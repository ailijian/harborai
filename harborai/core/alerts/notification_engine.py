"""
告警通知引擎

负责处理告警通知的发送、路由、重试、聚合等功能，
支持多种通知渠道和复杂的通知策略。
"""

import asyncio
import json
import smtplib
import ssl
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from typing import Dict, List, Optional, Any, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import aiohttp
import hashlib
from urllib.parse import urljoin


class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DINGTALK = "dingtalk"
    WECHAT = "wechat"
    SMS = "sms"
    CONSOLE = "console"
    FILE = "file"
    CUSTOM = "custom"


class NotificationPriority(Enum):
    """通知优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NotificationStatus(Enum):
    """通知状态"""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"
    SUPPRESSED = "suppressed"
    AGGREGATED = "aggregated"


@dataclass
class NotificationTemplate:
    """通知模板"""
    id: str
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    format_type: str = "text"  # text, html, markdown
    variables: List[str] = field(default_factory=list)
    
    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """渲染模板"""
        try:
            subject = self.subject_template.format(**context)
            body = self.body_template.format(**context)
            return {"subject": subject, "body": body}
        except KeyError as e:
            raise ValueError(f"模板变量缺失: {e}")


@dataclass
class NotificationRule:
    """通知规则"""
    id: str
    name: str
    conditions: Dict[str, Any]  # 匹配条件
    channels: List[NotificationChannel]
    priority: NotificationPriority
    template_id: str
    enabled: bool = True
    
    # 路由配置
    recipients: List[str] = field(default_factory=list)
    escalation_delay: Optional[int] = None  # 升级延迟（秒）
    escalation_recipients: List[str] = field(default_factory=list)
    
    # 限制配置
    rate_limit: Optional[int] = None  # 速率限制（每小时）
    quiet_hours: Optional[Dict[str, Any]] = None  # 静默时间
    
    def matches(self, alert_data: Dict[str, Any]) -> bool:
        """检查是否匹配告警"""
        for key, expected_value in self.conditions.items():
            if key not in alert_data:
                return False
            
            actual_value = alert_data[key]
            
            # 支持多种匹配方式
            if isinstance(expected_value, dict):
                operator = expected_value.get("operator", "eq")
                value = expected_value.get("value")
                
                if operator == "eq" and actual_value != value:
                    return False
                elif operator == "ne" and actual_value == value:
                    return False
                elif operator == "in" and actual_value not in value:
                    return False
                elif operator == "not_in" and actual_value in value:
                    return False
                elif operator == "regex":
                    import re
                    if not re.match(value, str(actual_value)):
                        return False
            else:
                if actual_value != expected_value:
                    return False
        
        return True


@dataclass
class NotificationConfig:
    """通知配置"""
    channel: NotificationChannel
    config: Dict[str, Any]
    enabled: bool = True
    
    # 重试配置
    max_retries: int = 3
    retry_delay: int = 60  # 秒
    retry_backoff: float = 2.0
    
    # 超时配置
    timeout: int = 30  # 秒


@dataclass
class NotificationMessage:
    """通知消息"""
    id: str
    alert_id: str
    channel: NotificationChannel
    recipients: List[str]
    subject: str
    body: str
    priority: NotificationPriority
    
    # 元数据
    template_id: Optional[str] = None
    rule_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    
    # 状态跟踪
    status: NotificationStatus = NotificationStatus.PENDING
    attempts: int = 0
    last_attempt_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # 聚合信息
    aggregated_alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "channel": self.channel.value,
            "recipients": self.recipients,
            "subject": self.subject,
            "body": self.body,
            "priority": self.priority.value,
            "template_id": self.template_id,
            "rule_id": self.rule_id,
            "created_at": self.created_at.isoformat(),
            "scheduled_at": self.scheduled_at.isoformat() if self.scheduled_at else None,
            "status": self.status.value,
            "attempts": self.attempts,
            "last_attempt_at": self.last_attempt_at.isoformat() if self.last_attempt_at else None,
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "error_message": self.error_message,
            "aggregated_alerts": self.aggregated_alerts
        }


class NotificationAggregator:
    """通知聚合器"""
    
    def __init__(self, window_size: int = 300, max_messages: int = 10):
        self.window_size = window_size  # 聚合窗口大小（秒）
        self.max_messages = max_messages  # 最大聚合消息数
        self.pending_messages: Dict[str, List[NotificationMessage]] = {}
        self.aggregation_tasks: Dict[str, asyncio.Task] = {}
    
    async def add_message(self, message: NotificationMessage) -> bool:
        """添加消息到聚合器"""
        # 生成聚合键
        aggregation_key = self._get_aggregation_key(message)
        
        if aggregation_key not in self.pending_messages:
            self.pending_messages[aggregation_key] = []
        
        self.pending_messages[aggregation_key].append(message)
        
        # 检查是否需要立即发送
        if len(self.pending_messages[aggregation_key]) >= self.max_messages:
            await self._flush_aggregation(aggregation_key)
            return True
        
        # 启动聚合任务
        if aggregation_key not in self.aggregation_tasks:
            task = asyncio.create_task(self._schedule_aggregation(aggregation_key))
            self.aggregation_tasks[aggregation_key] = task
        
        return False
    
    def _get_aggregation_key(self, message: NotificationMessage) -> str:
        """生成聚合键"""
        # 基于渠道、接收者、优先级生成聚合键
        key_parts = [
            message.channel.value,
            ",".join(sorted(message.recipients)),
            message.priority.value
        ]
        return "|".join(key_parts)
    
    async def _schedule_aggregation(self, aggregation_key: str):
        """调度聚合"""
        try:
            await asyncio.sleep(self.window_size)
            await self._flush_aggregation(aggregation_key)
        except asyncio.CancelledError:
            pass
        finally:
            if aggregation_key in self.aggregation_tasks:
                del self.aggregation_tasks[aggregation_key]
    
    async def _flush_aggregation(self, aggregation_key: str):
        """刷新聚合"""
        if aggregation_key not in self.pending_messages:
            return
        
        messages = self.pending_messages[aggregation_key]
        if not messages:
            return
        
        # 创建聚合消息
        aggregated_message = self._create_aggregated_message(messages)
        
        # 清理
        del self.pending_messages[aggregation_key]
        if aggregation_key in self.aggregation_tasks:
            self.aggregation_tasks[aggregation_key].cancel()
        
        # 返回聚合消息供发送
        return aggregated_message
    
    def _create_aggregated_message(self, messages: List[NotificationMessage]) -> NotificationMessage:
        """创建聚合消息"""
        if not messages:
            raise ValueError("消息列表不能为空")
        
        first_message = messages[0]
        
        # 聚合主题和内容
        if len(messages) == 1:
            return first_message
        
        aggregated_subject = f"[聚合告警] {len(messages)} 个告警"
        
        # 聚合内容
        aggregated_body_parts = [
            f"聚合了 {len(messages)} 个告警:",
            ""
        ]
        
        for i, msg in enumerate(messages, 1):
            aggregated_body_parts.extend([
                f"{i}. {msg.subject}",
                f"   时间: {msg.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
                f"   告警ID: {msg.alert_id}",
                ""
            ])
        
        aggregated_body = "\n".join(aggregated_body_parts)
        
        # 创建聚合消息
        aggregated_message = NotificationMessage(
            id=f"agg_{hashlib.md5(f'{first_message.channel.value}_{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}",
            alert_id="aggregated",
            channel=first_message.channel,
            recipients=first_message.recipients,
            subject=aggregated_subject,
            body=aggregated_body,
            priority=first_message.priority,
            aggregated_alerts=[msg.alert_id for msg in messages]
        )
        
        return aggregated_message


class NotificationEngine:
    """通知引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.templates: Dict[str, NotificationTemplate] = {}
        self.rules: Dict[str, NotificationRule] = {}
        self.configs: Dict[NotificationChannel, NotificationConfig] = {}
        self.custom_handlers: Dict[str, Callable] = {}
        
        # 状态跟踪
        self.pending_messages: List[NotificationMessage] = []
        self.sent_messages: List[NotificationMessage] = []
        self.failed_messages: List[NotificationMessage] = []
        
        # 速率限制
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        # 聚合器
        self.aggregator = NotificationAggregator()
        
        # 加载默认配置
        self._load_default_templates()
        self._load_default_rules()
    
    def _load_default_templates(self):
        """加载默认模板"""
        default_templates = [
            # 邮件模板
            NotificationTemplate(
                id="email_alert",
                name="邮件告警模板",
                channel=NotificationChannel.EMAIL,
                subject_template="[{severity}] {alert_name} - {status}",
                body_template="""
告警详情:

告警名称: {alert_name}
严重程度: {severity}
状态: {status}
触发时间: {triggered_at}
描述: {description}

指标信息:
- 指标名称: {metric_name}
- 当前值: {current_value}
- 阈值: {threshold}

标签:
{labels}

详细信息:
{details}

---
此邮件由HarborAI告警系统自动发送
                """.strip(),
                format_type="text",
                variables=["alert_name", "severity", "status", "triggered_at", "description", 
                          "metric_name", "current_value", "threshold", "labels", "details"]
            ),
            
            # Webhook模板
            NotificationTemplate(
                id="webhook_alert",
                name="Webhook告警模板",
                channel=NotificationChannel.WEBHOOK,
                subject_template="{alert_name}",
                body_template=json.dumps({
                    "alert_name": "{alert_name}",
                    "severity": "{severity}",
                    "status": "{status}",
                    "triggered_at": "{triggered_at}",
                    "description": "{description}",
                    "metric": {
                        "name": "{metric_name}",
                        "value": "{current_value}",
                        "threshold": "{threshold}"
                    },
                    "labels": "{labels}",
                    "details": "{details}"
                }, indent=2),
                format_type="json"
            ),
            
            # 钉钉模板
            NotificationTemplate(
                id="dingtalk_alert",
                name="钉钉告警模板",
                channel=NotificationChannel.DINGTALK,
                subject_template="{alert_name}",
                body_template="""
## [{severity}] {alert_name}

**状态**: {status}
**时间**: {triggered_at}
**描述**: {description}

### 指标信息
- **指标名称**: {metric_name}
- **当前值**: {current_value}
- **阈值**: {threshold}

### 标签
{labels}

### 详细信息
{details}
                """.strip(),
                format_type="markdown"
            ),
            
            # 控制台模板
            NotificationTemplate(
                id="console_alert",
                name="控制台告警模板",
                channel=NotificationChannel.CONSOLE,
                subject_template="[{severity}] {alert_name}",
                body_template="[{triggered_at}] [{severity}] {alert_name}: {description} (当前值: {current_value}, 阈值: {threshold})",
                format_type="text"
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
    
    def _load_default_rules(self):
        """加载默认规则"""
        default_rules = [
            # 关键告警规则
            NotificationRule(
                id="critical_alerts",
                name="关键告警通知",
                conditions={
                    "severity": "critical"
                },
                channels=[NotificationChannel.EMAIL, NotificationChannel.DINGTALK],
                priority=NotificationPriority.CRITICAL,
                template_id="email_alert",
                recipients=["admin@example.com", "ops@example.com"],
                escalation_delay=300,  # 5分钟后升级
                escalation_recipients=["manager@example.com"]
            ),
            
            # 高级告警规则
            NotificationRule(
                id="high_alerts",
                name="高级告警通知",
                conditions={
                    "severity": "high"
                },
                channels=[NotificationChannel.EMAIL],
                priority=NotificationPriority.HIGH,
                template_id="email_alert",
                recipients=["ops@example.com"],
                rate_limit=10  # 每小时最多10条
            ),
            
            # 中级告警规则
            NotificationRule(
                id="medium_alerts",
                name="中级告警通知",
                conditions={
                    "severity": "medium"
                },
                channels=[NotificationChannel.CONSOLE],
                priority=NotificationPriority.MEDIUM,
                template_id="console_alert",
                rate_limit=20
            ),
            
            # 数据库告警规则
            NotificationRule(
                id="database_alerts",
                name="数据库告警通知",
                conditions={
                    "category": "database",
                    "severity": {"operator": "in", "value": ["critical", "high"]}
                },
                channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK],
                priority=NotificationPriority.CRITICAL,
                template_id="webhook_alert",
                recipients=["dba@example.com", "ops@example.com"]
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
    
    async def add_template(self, template: NotificationTemplate) -> bool:
        """添加通知模板"""
        if template.id in self.templates:
            self.logger.warning(f"通知模板已存在: {template.id}")
            return False
        
        self.templates[template.id] = template
        self.logger.info(f"添加通知模板: {template.id}")
        return True
    
    async def add_rule(self, rule: NotificationRule) -> bool:
        """添加通知规则"""
        if rule.id in self.rules:
            self.logger.warning(f"通知规则已存在: {rule.id}")
            return False
        
        self.rules[rule.id] = rule
        self.logger.info(f"添加通知规则: {rule.id}")
        return True
    
    async def configure_channel(self, channel: NotificationChannel, config: NotificationConfig) -> bool:
        """配置通知渠道"""
        self.configs[channel] = config
        self.logger.info(f"配置通知渠道: {channel.value}")
        return True
    
    async def send_notification(self, alert_data: Dict[str, Any]) -> List[NotificationMessage]:
        """发送通知"""
        messages = []
        
        # 查找匹配的规则
        matching_rules = [
            rule for rule in self.rules.values()
            if rule.enabled and rule.matches(alert_data)
        ]
        
        if not matching_rules:
            self.logger.debug(f"没有匹配的通知规则: {alert_data.get('alert_id', 'unknown')}")
            return messages
        
        # 为每个匹配的规则创建通知消息
        for rule in matching_rules:
            # 检查速率限制
            if not await self._check_rate_limit(rule):
                self.logger.info(f"通知规则达到速率限制: {rule.id}")
                continue
            
            # 检查静默时间
            if not await self._check_quiet_hours(rule):
                self.logger.info(f"当前处于静默时间: {rule.id}")
                continue
            
            # 获取模板
            template = self.templates.get(rule.template_id)
            if not template:
                self.logger.error(f"通知模板不存在: {rule.template_id}")
                continue
            
            # 渲染消息
            try:
                rendered = template.render(alert_data)
            except ValueError as e:
                self.logger.error(f"模板渲染失败: {e}")
                continue
            
            # 为每个渠道创建消息
            for channel in rule.channels:
                if channel not in self.configs or not self.configs[channel].enabled:
                    self.logger.warning(f"通知渠道未配置或已禁用: {channel.value}")
                    continue
                
                message = NotificationMessage(
                    id=f"{alert_data.get('alert_id', 'unknown')}_{rule.id}_{channel.value}_{datetime.now().timestamp()}",
                    alert_id=alert_data.get('alert_id', 'unknown'),
                    channel=channel,
                    recipients=rule.recipients,
                    subject=rendered['subject'],
                    body=rendered['body'],
                    priority=rule.priority,
                    template_id=rule.template_id,
                    rule_id=rule.id
                )
                
                messages.append(message)
                
                # 添加到聚合器
                should_send_immediately = await self.aggregator.add_message(message)
                
                if should_send_immediately:
                    # 立即发送聚合消息
                    aggregated_message = await self.aggregator._flush_aggregation(
                        self.aggregator._get_aggregation_key(message)
                    )
                    if aggregated_message:
                        await self._send_message(aggregated_message)
                else:
                    # 添加到待发送队列
                    self.pending_messages.append(message)
        
        return messages
    
    async def _check_rate_limit(self, rule: NotificationRule) -> bool:
        """检查速率限制"""
        if not rule.rate_limit:
            return True
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # 清理过期记录
        if rule.id in self.rate_limits:
            self.rate_limits[rule.id] = [
                ts for ts in self.rate_limits[rule.id] if ts > hour_ago
            ]
        else:
            self.rate_limits[rule.id] = []
        
        # 检查是否超过限制
        if len(self.rate_limits[rule.id]) >= rule.rate_limit:
            return False
        
        # 记录本次发送
        self.rate_limits[rule.id].append(now)
        return True
    
    async def _check_quiet_hours(self, rule: NotificationRule) -> bool:
        """检查静默时间"""
        if not rule.quiet_hours:
            return True
        
        now = datetime.now()
        start_hour = rule.quiet_hours.get('start_hour', 22)
        end_hour = rule.quiet_hours.get('end_hour', 8)
        weekdays = rule.quiet_hours.get('weekdays', [])
        
        # 检查工作日
        if weekdays and now.weekday() not in weekdays:
            return True
        
        # 检查时间范围
        current_hour = now.hour
        if start_hour <= end_hour:
            # 同一天内的时间范围
            return not (start_hour <= current_hour < end_hour)
        else:
            # 跨天的时间范围
            return not (current_hour >= start_hour or current_hour < end_hour)
    
    async def _send_message(self, message: NotificationMessage) -> bool:
        """发送单个消息"""
        message.status = NotificationStatus.SENDING
        message.attempts += 1
        message.last_attempt_at = datetime.now()
        
        try:
            config = self.configs.get(message.channel)
            if not config:
                raise ValueError(f"通知渠道未配置: {message.channel.value}")
            
            # 根据渠道类型发送
            if message.channel == NotificationChannel.EMAIL:
                success = await self._send_email(message, config)
            elif message.channel == NotificationChannel.WEBHOOK:
                success = await self._send_webhook(message, config)
            elif message.channel == NotificationChannel.DINGTALK:
                success = await self._send_dingtalk(message, config)
            elif message.channel == NotificationChannel.CONSOLE:
                success = await self._send_console(message, config)
            elif message.channel == NotificationChannel.FILE:
                success = await self._send_file(message, config)
            elif message.channel == NotificationChannel.CUSTOM:
                success = await self._send_custom(message, config)
            else:
                raise ValueError(f"不支持的通知渠道: {message.channel.value}")
            
            if success:
                message.status = NotificationStatus.SENT
                message.sent_at = datetime.now()
                self.sent_messages.append(message)
                self.logger.info(f"通知发送成功: {message.id}")
                return True
            else:
                raise Exception("发送失败")
        
        except Exception as e:
            message.status = NotificationStatus.FAILED
            message.error_message = str(e)
            self.logger.error(f"通知发送失败: {message.id}, 错误: {e}")
            
            # 检查是否需要重试
            if message.attempts < config.max_retries:
                message.status = NotificationStatus.RETRYING
                # 计算重试延迟
                delay = config.retry_delay * (config.retry_backoff ** (message.attempts - 1))
                message.scheduled_at = datetime.now() + timedelta(seconds=delay)
                self.logger.info(f"将在 {delay} 秒后重试发送: {message.id}")
            else:
                self.failed_messages.append(message)
            
            return False
    
    async def _send_email(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """发送邮件"""
        try:
            smtp_config = config.config
            
            # 创建邮件
            msg = MimeMultipart()
            msg['From'] = smtp_config['from']
            msg['To'] = ', '.join(message.recipients)
            msg['Subject'] = message.subject
            
            # 添加邮件内容
            msg.attach(MimeText(message.body, 'plain', 'utf-8'))
            
            # 发送邮件
            context = ssl.create_default_context()
            with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
                if smtp_config.get('use_tls', True):
                    server.starttls(context=context)
                
                if 'username' in smtp_config and 'password' in smtp_config:
                    server.login(smtp_config['username'], smtp_config['password'])
                
                server.send_message(msg)
            
            return True
            
        except Exception as e:
            self.logger.error(f"邮件发送失败: {e}")
            return False
    
    async def _send_webhook(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """发送Webhook"""
        try:
            webhook_config = config.config
            url = webhook_config['url']
            
            # 准备数据
            if message.body.startswith('{'):
                # JSON格式
                data = json.loads(message.body)
            else:
                # 文本格式
                data = {
                    'subject': message.subject,
                    'body': message.body,
                    'alert_id': message.alert_id,
                    'priority': message.priority.value,
                    'timestamp': message.created_at.isoformat()
                }
            
            # 发送请求
            headers = webhook_config.get('headers', {})
            headers.setdefault('Content-Type', 'application/json')
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
                async with session.post(url, json=data, headers=headers) as response:
                    if response.status < 400:
                        return True
                    else:
                        self.logger.error(f"Webhook响应错误: {response.status}")
                        return False
        
        except Exception as e:
            self.logger.error(f"Webhook发送失败: {e}")
            return False
    
    async def _send_dingtalk(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """发送钉钉消息"""
        try:
            dingtalk_config = config.config
            webhook_url = dingtalk_config['webhook_url']
            
            # 准备钉钉消息格式
            data = {
                "msgtype": "markdown",
                "markdown": {
                    "title": message.subject,
                    "text": message.body
                }
            }
            
            # 添加@功能
            if 'at_mobiles' in dingtalk_config:
                data['at'] = {
                    'atMobiles': dingtalk_config['at_mobiles'],
                    'isAtAll': dingtalk_config.get('at_all', False)
                }
            
            # 发送请求
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
                async with session.post(webhook_url, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get('errcode') == 0:
                            return True
                        else:
                            self.logger.error(f"钉钉发送失败: {result.get('errmsg')}")
                            return False
                    else:
                        self.logger.error(f"钉钉请求失败: {response.status}")
                        return False
        
        except Exception as e:
            self.logger.error(f"钉钉发送失败: {e}")
            return False
    
    async def _send_console(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """发送控制台消息"""
        try:
            print(f"[ALERT] {message.subject}")
            if message.body != message.subject:
                print(f"[ALERT] {message.body}")
            return True
        except Exception as e:
            self.logger.error(f"控制台输出失败: {e}")
            return False
    
    async def _send_file(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """发送文件消息"""
        try:
            file_config = config.config
            file_path = file_config['file_path']
            
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"[{message.created_at.isoformat()}] {message.subject}\n")
                if message.body != message.subject:
                    f.write(f"{message.body}\n")
                f.write("-" * 80 + "\n")
            
            return True
        except Exception as e:
            self.logger.error(f"文件写入失败: {e}")
            return False
    
    async def _send_custom(self, message: NotificationMessage, config: NotificationConfig) -> bool:
        """发送自定义消息"""
        try:
            handler_name = config.config.get('handler')
            if handler_name not in self.custom_handlers:
                raise ValueError(f"自定义处理器不存在: {handler_name}")
            
            handler = self.custom_handlers[handler_name]
            return await handler(message, config)
        
        except Exception as e:
            self.logger.error(f"自定义发送失败: {e}")
            return False
    
    def register_custom_handler(self, name: str, handler: Callable):
        """注册自定义处理器"""
        self.custom_handlers[name] = handler
        self.logger.info(f"注册自定义处理器: {name}")
    
    async def process_pending_messages(self):
        """处理待发送消息"""
        now = datetime.now()
        
        # 处理重试消息
        retry_messages = [
            msg for msg in self.pending_messages
            if msg.status == NotificationStatus.RETRYING and 
               msg.scheduled_at and msg.scheduled_at <= now
        ]
        
        for message in retry_messages:
            await self._send_message(message)
            self.pending_messages.remove(message)
        
        # 处理普通待发送消息
        pending_messages = [
            msg for msg in self.pending_messages
            if msg.status == NotificationStatus.PENDING
        ]
        
        for message in pending_messages:
            await self._send_message(message)
            self.pending_messages.remove(message)
    
    async def get_notification_statistics(self) -> Dict[str, Any]:
        """获取通知统计信息"""
        total_sent = len(self.sent_messages)
        total_failed = len(self.failed_messages)
        total_pending = len(self.pending_messages)
        
        # 按渠道统计
        channel_stats = {}
        for channel in NotificationChannel:
            sent_count = len([msg for msg in self.sent_messages if msg.channel == channel])
            failed_count = len([msg for msg in self.failed_messages if msg.channel == channel])
            channel_stats[channel.value] = {
                "sent": sent_count,
                "failed": failed_count
            }
        
        # 按优先级统计
        priority_stats = {}
        for priority in NotificationPriority:
            sent_count = len([msg for msg in self.sent_messages if msg.priority == priority])
            failed_count = len([msg for msg in self.failed_messages if msg.priority == priority])
            priority_stats[priority.value] = {
                "sent": sent_count,
                "failed": failed_count
            }
        
        # 成功率
        total_attempts = total_sent + total_failed
        success_rate = (total_sent / total_attempts * 100) if total_attempts > 0 else 0
        
        return {
            "total_sent": total_sent,
            "total_failed": total_failed,
            "total_pending": total_pending,
            "success_rate": round(success_rate, 2),
            "channel_statistics": channel_stats,
            "priority_statistics": priority_stats,
            "templates_count": len(self.templates),
            "rules_count": len(self.rules),
            "configured_channels": len(self.configs)
        }
    
    async def cleanup_old_messages(self, retention_days: int = 30):
        """清理过期消息"""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        # 清理已发送消息
        old_sent_count = len(self.sent_messages)
        self.sent_messages = [
            msg for msg in self.sent_messages
            if msg.sent_at and msg.sent_at > cutoff_time
        ]
        
        # 清理失败消息
        old_failed_count = len(self.failed_messages)
        self.failed_messages = [
            msg for msg in self.failed_messages
            if msg.last_attempt_at and msg.last_attempt_at > cutoff_time
        ]
        
        cleaned_sent = old_sent_count - len(self.sent_messages)
        cleaned_failed = old_failed_count - len(self.failed_messages)
        
        self.logger.info(f"清理了 {cleaned_sent} 条已发送消息和 {cleaned_failed} 条失败消息")