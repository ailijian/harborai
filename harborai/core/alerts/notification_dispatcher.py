"""
告警通知分发器

负责管理告警通知的分发、路由、重试和聚合逻辑，
支持多种通知渠道和复杂的分发策略。
"""

import asyncio
import json
import smtplib
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import aiohttp
import hashlib
from collections import defaultdict, deque


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
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class NotificationStatus(Enum):
    """通知状态"""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"
    SUPPRESSED = "suppressed"


class AggregationStrategy(Enum):
    """聚合策略"""
    NONE = "none"              # 不聚合
    TIME_WINDOW = "time_window"  # 时间窗口聚合
    COUNT_THRESHOLD = "count_threshold"  # 数量阈值聚合
    SIMILARITY = "similarity"   # 相似性聚合
    CUSTOM = "custom"          # 自定义聚合


@dataclass
class NotificationTemplate:
    """通知模板"""
    id: str
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    format: str = "text"  # text, html, markdown
    variables: List[str] = field(default_factory=list)
    
    def render(self, data: Dict[str, Any]) -> Dict[str, str]:
        """渲染模板"""
        try:
            # 简单的模板变量替换
            subject = self.subject_template
            body = self.body_template
            
            for key, value in data.items():
                placeholder = f"{{{key}}}"
                subject = subject.replace(placeholder, str(value))
                body = body.replace(placeholder, str(value))
            
            return {
                "subject": subject,
                "body": body,
                "format": self.format
            }
        except Exception as e:
            logging.error(f"模板渲染失败: {e}")
            return {
                "subject": f"告警通知 - {data.get('alert_name', 'Unknown')}",
                "body": str(data),
                "format": "text"
            }


@dataclass
class NotificationRule:
    """通知规则"""
    id: str
    name: str
    conditions: List[Dict[str, Any]]  # 匹配条件
    channels: List[NotificationChannel]  # 通知渠道
    recipients: Dict[NotificationChannel, List[str]]  # 接收者
    template_id: Optional[str] = None
    priority: NotificationPriority = NotificationPriority.MEDIUM
    enabled: bool = True
    
    # 限制配置
    rate_limit: Optional[int] = None  # 速率限制（每分钟）
    quiet_hours: Optional[Dict[str, Any]] = None  # 静默时间
    escalation_delay: Optional[int] = None  # 升级延迟（秒）
    
    # 聚合配置
    aggregation_strategy: AggregationStrategy = AggregationStrategy.NONE
    aggregation_window: int = 300  # 聚合窗口（秒）
    aggregation_threshold: int = 5  # 聚合阈值
    
    def matches(self, alert_data: Dict[str, Any]) -> bool:
        """检查告警是否匹配规则"""
        if not self.enabled:
            return False
        
        for condition in self.conditions:
            field = condition.get("field")
            operator = condition.get("operator")
            value = condition.get("value")
            
            if field not in alert_data:
                return False
            
            alert_value = alert_data[field]
            
            if operator == "eq" and alert_value != value:
                return False
            elif operator == "ne" and alert_value == value:
                return False
            elif operator == "in" and alert_value not in value:
                return False
            elif operator == "not_in" and alert_value in value:
                return False
            elif operator == "regex":
                import re
                if not re.search(value, str(alert_value)):
                    return False
            elif operator == "gt" and alert_value <= value:
                return False
            elif operator == "lt" and alert_value >= value:
                return False
            elif operator == "gte" and alert_value < value:
                return False
            elif operator == "lte" and alert_value > value:
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
    retry_delay: int = 60  # 重试延迟（秒）
    retry_backoff: float = 2.0  # 退避倍数
    
    # 超时配置
    timeout: int = 30  # 超时时间（秒）


@dataclass
class NotificationMessage:
    """通知消息"""
    id: str
    alert_id: str
    rule_id: str
    channel: NotificationChannel
    recipients: List[str]
    subject: str
    body: str
    format: str = "text"
    priority: NotificationPriority = NotificationPriority.MEDIUM
    
    # 状态信息
    status: NotificationStatus = NotificationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    
    # 重试信息
    retry_count: int = 0
    last_error: Optional[str] = None
    
    # 聚合信息
    aggregated_alerts: List[str] = field(default_factory=list)
    aggregation_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "channel": self.channel.value,
            "recipients": self.recipients,
            "subject": self.subject,
            "body": self.body,
            "format": self.format,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "sent_at": self.sent_at.isoformat() if self.sent_at else None,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "aggregated_alerts": self.aggregated_alerts,
            "aggregation_key": self.aggregation_key
        }


@dataclass
class AggregationGroup:
    """聚合组"""
    key: str
    messages: List[NotificationMessage]
    created_at: datetime
    last_updated: datetime
    
    def should_send(self, strategy: AggregationStrategy, 
                   threshold: int, window: int) -> bool:
        """检查是否应该发送聚合消息"""
        now = datetime.now()
        
        if strategy == AggregationStrategy.COUNT_THRESHOLD:
            return len(self.messages) >= threshold
        
        elif strategy == AggregationStrategy.TIME_WINDOW:
            return (now - self.created_at).total_seconds() >= window
        
        elif strategy == AggregationStrategy.SIMILARITY:
            # 基于相似性的聚合逻辑
            return len(self.messages) >= threshold
        
        return False


class NotificationDispatcher:
    """通知分发器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 配置
        self.templates: Dict[str, NotificationTemplate] = {}
        self.rules: Dict[str, NotificationRule] = {}
        self.configs: Dict[NotificationChannel, NotificationConfig] = {}
        
        # 状态管理
        self.pending_messages: List[NotificationMessage] = []
        self.sent_messages: List[NotificationMessage] = []
        self.failed_messages: List[NotificationMessage] = []
        
        # 聚合管理
        self.aggregation_groups: Dict[str, AggregationGroup] = {}
        
        # 速率限制
        self.rate_limiters: Dict[str, deque] = defaultdict(deque)
        
        # 自定义处理器
        self.custom_handlers: Dict[str, Callable] = {}
        
        # 任务管理
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
        # 加载默认配置
        self._load_default_templates()
        self._load_default_rules()
    
    def _load_default_templates(self):
        """加载默认模板"""
        default_templates = [
            NotificationTemplate(
                id="critical_alert_email",
                name="严重告警邮件模板",
                channel=NotificationChannel.EMAIL,
                subject_template="🚨 严重告警: {alert_name}",
                body_template="""
严重告警详情:

告警名称: {alert_name}
严重级别: {severity}
主机: {host}
服务: {service}
时间: {timestamp}
描述: {description}

请立即处理此告警。

详细信息:
{details}
                """.strip(),
                format="text",
                variables=["alert_name", "severity", "host", "service", "timestamp", "description", "details"]
            ),
            
            NotificationTemplate(
                id="general_alert_webhook",
                name="通用告警Webhook模板",
                channel=NotificationChannel.WEBHOOK,
                subject_template="告警通知: {alert_name}",
                body_template="""{
    "alert_id": "{alert_id}",
    "alert_name": "{alert_name}",
    "severity": "{severity}",
    "host": "{host}",
    "service": "{service}",
    "timestamp": "{timestamp}",
    "description": "{description}",
    "status": "firing"
}""",
                format="json",
                variables=["alert_id", "alert_name", "severity", "host", "service", "timestamp", "description"]
            ),
            
            NotificationTemplate(
                id="dingtalk_alert",
                name="钉钉告警模板",
                channel=NotificationChannel.DINGTALK,
                subject_template="告警通知",
                body_template="""## 告警通知

**告警名称**: {alert_name}
**严重级别**: {severity}
**主机**: {host}
**服务**: {service}
**时间**: {timestamp}

**描述**: {description}

> 请及时处理此告警
                """.strip(),
                format="markdown",
                variables=["alert_name", "severity", "host", "service", "timestamp", "description"]
            ),
            
            NotificationTemplate(
                id="aggregated_alert",
                name="聚合告警模板",
                channel=NotificationChannel.EMAIL,
                subject_template="📊 聚合告警报告 ({count}个告警)",
                body_template="""
聚合告警报告

时间范围: {start_time} - {end_time}
告警数量: {count}

告警列表:
{alert_list}

请查看详细信息并处理相关告警。
                """.strip(),
                format="text",
                variables=["count", "start_time", "end_time", "alert_list"]
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
    
    def _load_default_rules(self):
        """加载默认规则"""
        default_rules = [
            NotificationRule(
                id="critical_alerts",
                name="严重告警通知",
                conditions=[
                    {"field": "severity", "operator": "eq", "value": "critical"}
                ],
                channels=[NotificationChannel.EMAIL, NotificationChannel.DINGTALK],
                recipients={
                    NotificationChannel.EMAIL: ["admin@example.com"],
                    NotificationChannel.DINGTALK: ["webhook_url"]
                },
                template_id="critical_alert_email",
                priority=NotificationPriority.CRITICAL,
                rate_limit=10,  # 每分钟最多10条
                escalation_delay=300  # 5分钟后升级
            ),
            
            NotificationRule(
                id="high_priority_alerts",
                name="高优先级告警通知",
                conditions=[
                    {"field": "severity", "operator": "eq", "value": "high"}
                ],
                channels=[NotificationChannel.EMAIL],
                recipients={
                    NotificationChannel.EMAIL: ["team@example.com"]
                },
                priority=NotificationPriority.HIGH,
                rate_limit=20,
                aggregation_strategy=AggregationStrategy.TIME_WINDOW,
                aggregation_window=600,  # 10分钟聚合窗口
                aggregation_threshold=3
            ),
            
            NotificationRule(
                id="database_alerts",
                name="数据库告警通知",
                conditions=[
                    {"field": "service", "operator": "eq", "value": "database"},
                    {"field": "severity", "operator": "in", "value": ["high", "critical"]}
                ],
                channels=[NotificationChannel.WEBHOOK, NotificationChannel.EMAIL],
                recipients={
                    NotificationChannel.WEBHOOK: ["http://monitoring.example.com/webhook"],
                    NotificationChannel.EMAIL: ["dba@example.com"]
                },
                template_id="general_alert_webhook",
                priority=NotificationPriority.HIGH
            ),
            
            NotificationRule(
                id="low_priority_aggregated",
                name="低优先级聚合告警",
                conditions=[
                    {"field": "severity", "operator": "in", "value": ["low", "info"]}
                ],
                channels=[NotificationChannel.EMAIL],
                recipients={
                    NotificationChannel.EMAIL: ["team@example.com"]
                },
                priority=NotificationPriority.LOW,
                aggregation_strategy=AggregationStrategy.COUNT_THRESHOLD,
                aggregation_threshold=10,
                aggregation_window=1800  # 30分钟聚合窗口
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
    
    async def configure_channel(self, channel: NotificationChannel, 
                              config: Dict[str, Any]) -> bool:
        """配置通知渠道"""
        try:
            notification_config = NotificationConfig(
                channel=channel,
                config=config,
                enabled=config.get("enabled", True),
                max_retries=config.get("max_retries", 3),
                retry_delay=config.get("retry_delay", 60),
                retry_backoff=config.get("retry_backoff", 2.0),
                timeout=config.get("timeout", 30)
            )
            
            self.configs[channel] = notification_config
            self.logger.info(f"配置通知渠道: {channel.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"配置通知渠道失败: {e}")
            return False
    
    async def add_template(self, template: NotificationTemplate) -> bool:
        """添加通知模板"""
        if template.id in self.templates:
            self.logger.warning(f"模板已存在: {template.id}")
            return False
        
        self.templates[template.id] = template
        self.logger.info(f"添加通知模板: {template.id}")
        return True
    
    async def add_rule(self, rule: NotificationRule) -> bool:
        """添加通知规则"""
        if rule.id in self.rules:
            self.logger.warning(f"规则已存在: {rule.id}")
            return False
        
        self.rules[rule.id] = rule
        self.logger.info(f"添加通知规则: {rule.id}")
        return True
    
    async def send_notification(self, alert_data: Dict[str, Any]) -> List[str]:
        """发送通知"""
        sent_message_ids = []
        
        # 查找匹配的规则
        matching_rules = [rule for rule in self.rules.values() if rule.matches(alert_data)]
        
        if not matching_rules:
            self.logger.warning(f"没有找到匹配的通知规则: {alert_data.get('alert_id')}")
            return sent_message_ids
        
        for rule in matching_rules:
            # 检查速率限制
            if not await self._check_rate_limit(rule):
                self.logger.warning(f"触发速率限制，跳过规则: {rule.id}")
                continue
            
            # 检查静默时间
            if not await self._check_quiet_hours(rule):
                self.logger.info(f"在静默时间内，跳过规则: {rule.id}")
                continue
            
            # 为每个渠道创建消息
            for channel in rule.channels:
                if channel not in self.configs or not self.configs[channel].enabled:
                    self.logger.warning(f"通知渠道未配置或已禁用: {channel.value}")
                    continue
                
                recipients = rule.recipients.get(channel, [])
                if not recipients:
                    self.logger.warning(f"规则 {rule.id} 的渠道 {channel.value} 没有配置接收者")
                    continue
                
                # 渲染消息内容
                content = await self._render_message_content(rule, alert_data)
                
                # 创建通知消息
                message = NotificationMessage(
                    id=f"{alert_data.get('alert_id')}_{rule.id}_{channel.value}_{datetime.now().timestamp()}",
                    alert_id=alert_data.get('alert_id', ''),
                    rule_id=rule.id,
                    channel=channel,
                    recipients=recipients,
                    subject=content["subject"],
                    body=content["body"],
                    format=content["format"],
                    priority=rule.priority
                )
                
                # 检查是否需要聚合
                if rule.aggregation_strategy != AggregationStrategy.NONE:
                    await self._handle_aggregation(message, rule)
                else:
                    # 直接发送
                    await self._queue_message(message)
                
                sent_message_ids.append(message.id)
        
        return sent_message_ids
    
    async def _render_message_content(self, rule: NotificationRule, 
                                    alert_data: Dict[str, Any]) -> Dict[str, str]:
        """渲染消息内容"""
        template_id = rule.template_id
        
        if template_id and template_id in self.templates:
            template = self.templates[template_id]
            return template.render(alert_data)
        else:
            # 使用默认模板
            return {
                "subject": f"告警通知: {alert_data.get('alert_name', 'Unknown')}",
                "body": json.dumps(alert_data, indent=2, ensure_ascii=False),
                "format": "text"
            }
    
    async def _check_rate_limit(self, rule: NotificationRule) -> bool:
        """检查速率限制"""
        if not rule.rate_limit:
            return True
        
        now = datetime.now()
        rate_key = f"rule_{rule.id}"
        
        # 清理过期记录
        cutoff_time = now - timedelta(minutes=1)
        rate_limiter = self.rate_limiters[rate_key]
        
        while rate_limiter and rate_limiter[0] < cutoff_time:
            rate_limiter.popleft()
        
        # 检查是否超过限制
        if len(rate_limiter) >= rule.rate_limit:
            return False
        
        # 添加当前记录
        rate_limiter.append(now)
        return True
    
    async def _check_quiet_hours(self, rule: NotificationRule) -> bool:
        """检查静默时间"""
        if not rule.quiet_hours:
            return True
        
        now = datetime.now()
        current_hour = now.hour
        current_weekday = now.weekday()  # 0=Monday, 6=Sunday
        
        # 检查工作日静默时间
        if "weekdays" in rule.quiet_hours:
            weekday_hours = rule.quiet_hours["weekdays"]
            if (current_weekday < 5 and  # Monday-Friday
                weekday_hours.get("start", 0) <= current_hour <= weekday_hours.get("end", 23)):
                return False
        
        # 检查周末静默时间
        if "weekends" in rule.quiet_hours:
            weekend_hours = rule.quiet_hours["weekends"]
            if (current_weekday >= 5 and  # Saturday-Sunday
                weekend_hours.get("start", 0) <= current_hour <= weekend_hours.get("end", 23)):
                return False
        
        return True
    
    async def _handle_aggregation(self, message: NotificationMessage, 
                                rule: NotificationRule):
        """处理消息聚合"""
        # 生成聚合键
        aggregation_key = self._generate_aggregation_key(message, rule)
        message.aggregation_key = aggregation_key
        
        # 获取或创建聚合组
        if aggregation_key not in self.aggregation_groups:
            self.aggregation_groups[aggregation_key] = AggregationGroup(
                key=aggregation_key,
                messages=[],
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
        
        group = self.aggregation_groups[aggregation_key]
        group.messages.append(message)
        group.last_updated = datetime.now()
        
        # 检查是否应该发送聚合消息
        if group.should_send(rule.aggregation_strategy, 
                           rule.aggregation_threshold, 
                           rule.aggregation_window):
            await self._send_aggregated_message(group, rule)
            # 清理已发送的聚合组
            del self.aggregation_groups[aggregation_key]
    
    def _generate_aggregation_key(self, message: NotificationMessage, 
                                rule: NotificationRule) -> str:
        """生成聚合键"""
        key_parts = [
            rule.id,
            message.channel.value,
            "|".join(sorted(message.recipients))
        ]
        
        # 可以根据需要添加更多聚合维度
        # 例如：按主机、服务、告警类型等聚合
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _send_aggregated_message(self, group: AggregationGroup, 
                                     rule: NotificationRule):
        """发送聚合消息"""
        if not group.messages:
            return
        
        # 使用第一个消息作为模板
        first_message = group.messages[0]
        
        # 构建聚合数据
        aggregated_data = {
            "count": len(group.messages),
            "start_time": group.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": group.last_updated.strftime("%Y-%m-%d %H:%M:%S"),
            "alert_list": "\n".join([
                f"- {msg.subject} ({msg.created_at.strftime('%H:%M:%S')})"
                for msg in group.messages
            ])
        }
        
        # 使用聚合模板
        template = self.templates.get("aggregated_alert")
        if template:
            content = template.render(aggregated_data)
        else:
            content = {
                "subject": f"聚合告警报告 ({len(group.messages)}个告警)",
                "body": f"聚合了{len(group.messages)}个告警，请查看详细信息。",
                "format": "text"
            }
        
        # 创建聚合消息
        aggregated_message = NotificationMessage(
            id=f"aggregated_{group.key}_{datetime.now().timestamp()}",
            alert_id="aggregated",
            rule_id=rule.id,
            channel=first_message.channel,
            recipients=first_message.recipients,
            subject=content["subject"],
            body=content["body"],
            format=content["format"],
            priority=rule.priority,
            aggregated_alerts=[msg.alert_id for msg in group.messages]
        )
        
        await self._queue_message(aggregated_message)
    
    async def _queue_message(self, message: NotificationMessage):
        """将消息加入发送队列"""
        self.pending_messages.append(message)
        self.logger.info(f"消息已加入队列: {message.id}")
    
    async def start(self):
        """启动通知分发器"""
        if self.running:
            self.logger.warning("通知分发器已在运行")
            return
        
        self.running = True
        self.logger.info("启动通知分发器")
        
        # 启动后台任务
        self.background_tasks = [
            asyncio.create_task(self._process_pending_messages()),
            asyncio.create_task(self._retry_failed_messages()),
            asyncio.create_task(self._process_aggregation_timeouts()),
            asyncio.create_task(self._cleanup_old_messages())
        ]
    
    async def stop(self):
        """停止通知分发器"""
        if not self.running:
            return
        
        self.running = False
        self.logger.info("停止通知分发器")
        
        # 取消后台任务
        for task in self.background_tasks:
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.background_tasks.clear()
    
    async def _process_pending_messages(self):
        """处理待发送消息"""
        while self.running:
            try:
                if self.pending_messages:
                    # 按优先级排序
                    self.pending_messages.sort(key=lambda m: m.priority.value)
                    
                    # 处理消息
                    messages_to_process = self.pending_messages[:10]  # 批量处理
                    self.pending_messages = self.pending_messages[10:]
                    
                    for message in messages_to_process:
                        await self._send_message(message)
                
                await asyncio.sleep(1)  # 1秒检查一次
                
            except Exception as e:
                self.logger.error(f"处理待发送消息时出错: {e}")
                await asyncio.sleep(5)
    
    async def _send_message(self, message: NotificationMessage):
        """发送单个消息"""
        try:
            message.status = NotificationStatus.SENDING
            
            config = self.configs.get(message.channel)
            if not config:
                raise Exception(f"通知渠道未配置: {message.channel.value}")
            
            # 根据渠道类型发送消息
            if message.channel == NotificationChannel.EMAIL:
                await self._send_email(message, config)
            elif message.channel == NotificationChannel.WEBHOOK:
                await self._send_webhook(message, config)
            elif message.channel == NotificationChannel.DINGTALK:
                await self._send_dingtalk(message, config)
            elif message.channel == NotificationChannel.SLACK:
                await self._send_slack(message, config)
            elif message.channel == NotificationChannel.CONSOLE:
                await self._send_console(message, config)
            elif message.channel == NotificationChannel.FILE:
                await self._send_file(message, config)
            elif message.channel == NotificationChannel.CUSTOM:
                await self._send_custom(message, config)
            else:
                raise Exception(f"不支持的通知渠道: {message.channel.value}")
            
            # 标记为已发送
            message.status = NotificationStatus.SENT
            message.sent_at = datetime.now()
            self.sent_messages.append(message)
            
            self.logger.info(f"消息发送成功: {message.id}")
            
        except Exception as e:
            message.status = NotificationStatus.FAILED
            message.last_error = str(e)
            self.failed_messages.append(message)
            
            self.logger.error(f"消息发送失败: {message.id}, 错误: {e}")
    
    async def _send_email(self, message: NotificationMessage, 
                         config: NotificationConfig):
        """发送邮件"""
        smtp_config = config.config
        
        # 创建邮件消息
        msg = MimeMultipart()
        msg['From'] = smtp_config['from']
        msg['Subject'] = message.subject
        
        # 添加邮件内容
        if message.format == "html":
            msg.attach(MimeText(message.body, 'html', 'utf-8'))
        else:
            msg.attach(MimeText(message.body, 'plain', 'utf-8'))
        
        # 发送给每个接收者
        for recipient in message.recipients:
            msg['To'] = recipient
            
            # 连接SMTP服务器
            context = ssl.create_default_context()
            
            if smtp_config.get('use_tls', True):
                with smtplib.SMTP(smtp_config['host'], smtp_config['port']) as server:
                    server.starttls(context=context)
                    if 'username' in smtp_config:
                        server.login(smtp_config['username'], smtp_config['password'])
                    server.send_message(msg)
            else:
                with smtplib.SMTP_SSL(smtp_config['host'], smtp_config['port'], context=context) as server:
                    if 'username' in smtp_config:
                        server.login(smtp_config['username'], smtp_config['password'])
                    server.send_message(msg)
    
    async def _send_webhook(self, message: NotificationMessage, 
                          config: NotificationConfig):
        """发送Webhook"""
        webhook_config = config.config
        
        # 构建请求数据
        if message.format == "json":
            try:
                payload = json.loads(message.body)
            except json.JSONDecodeError:
                payload = {"message": message.body}
        else:
            payload = {
                "subject": message.subject,
                "message": message.body,
                "alert_id": message.alert_id,
                "timestamp": message.created_at.isoformat()
            }
        
        # 发送HTTP请求
        headers = webhook_config.get('headers', {})
        headers.setdefault('Content-Type', 'application/json')
        
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for url in message.recipients:
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status >= 400:
                        raise Exception(f"Webhook请求失败: {response.status} {await response.text()}")
    
    async def _send_dingtalk(self, message: NotificationMessage, 
                           config: NotificationConfig):
        """发送钉钉消息"""
        dingtalk_config = config.config
        
        # 构建钉钉消息格式
        if message.format == "markdown":
            payload = {
                "msgtype": "markdown",
                "markdown": {
                    "title": message.subject,
                    "text": message.body
                }
            }
        else:
            payload = {
                "msgtype": "text",
                "text": {
                    "content": f"{message.subject}\n\n{message.body}"
                }
            }
        
        # 添加@功能
        if 'at_mobiles' in dingtalk_config:
            payload["at"] = {
                "atMobiles": dingtalk_config['at_mobiles'],
                "isAtAll": dingtalk_config.get('at_all', False)
            }
        
        # 发送请求
        headers = {'Content-Type': 'application/json'}
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for webhook_url in message.recipients:
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    if response.status >= 400:
                        raise Exception(f"钉钉消息发送失败: {response.status} {await response.text()}")
    
    async def _send_slack(self, message: NotificationMessage, 
                        config: NotificationConfig):
        """发送Slack消息"""
        slack_config = config.config
        
        payload = {
            "text": message.subject,
            "attachments": [
                {
                    "color": "warning",
                    "text": message.body,
                    "ts": int(message.created_at.timestamp())
                }
            ]
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f"Bearer {slack_config['token']}"
        }
        
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for channel in message.recipients:
                payload["channel"] = channel
                
                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status >= 400:
                        raise Exception(f"Slack消息发送失败: {response.status} {await response.text()}")
    
    async def _send_console(self, message: NotificationMessage, 
                          config: NotificationConfig):
        """发送控制台消息"""
        print(f"[ALERT] {message.subject}")
        print(f"Time: {message.created_at}")
        print(f"Content: {message.body}")
        print("-" * 50)
    
    async def _send_file(self, message: NotificationMessage, 
                       config: NotificationConfig):
        """发送文件消息"""
        file_config = config.config
        log_file = file_config.get('file_path', 'alerts.log')
        
        log_entry = {
            "timestamp": message.created_at.isoformat(),
            "alert_id": message.alert_id,
            "subject": message.subject,
            "body": message.body,
            "recipients": message.recipients
        }
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    async def _send_custom(self, message: NotificationMessage, 
                         config: NotificationConfig):
        """发送自定义消息"""
        custom_config = config.config
        handler_name = custom_config.get('handler')
        
        if handler_name in self.custom_handlers:
            handler = self.custom_handlers[handler_name]
            await handler(message, config)
        else:
            raise Exception(f"自定义处理器未找到: {handler_name}")
    
    def register_custom_handler(self, name: str, handler: Callable):
        """注册自定义处理器"""
        self.custom_handlers[name] = handler
        self.logger.info(f"注册自定义处理器: {name}")
    
    async def _retry_failed_messages(self):
        """重试失败的消息"""
        while self.running:
            try:
                messages_to_retry = []
                
                for message in self.failed_messages[:]:
                    config = self.configs.get(message.channel)
                    if not config:
                        continue
                    
                    # 检查是否应该重试
                    if (message.retry_count < config.max_retries and
                        message.status == NotificationStatus.FAILED):
                        
                        # 计算重试延迟
                        delay = config.retry_delay * (config.retry_backoff ** message.retry_count)
                        retry_time = message.created_at + timedelta(seconds=delay)
                        
                        if datetime.now() >= retry_time:
                            message.retry_count += 1
                            message.status = NotificationStatus.RETRYING
                            messages_to_retry.append(message)
                            self.failed_messages.remove(message)
                
                # 重试消息
                for message in messages_to_retry:
                    await self._send_message(message)
                
                await asyncio.sleep(30)  # 30秒检查一次
                
            except Exception as e:
                self.logger.error(f"重试失败消息时出错: {e}")
                await asyncio.sleep(60)
    
    async def _process_aggregation_timeouts(self):
        """处理聚合超时"""
        while self.running:
            try:
                now = datetime.now()
                expired_groups = []
                
                for key, group in self.aggregation_groups.items():
                    # 查找对应的规则
                    rule = None
                    for r in self.rules.values():
                        if group.messages and group.messages[0].rule_id == r.id:
                            rule = r
                            break
                    
                    if rule and group.should_send(
                        rule.aggregation_strategy,
                        rule.aggregation_threshold,
                        rule.aggregation_window
                    ):
                        await self._send_aggregated_message(group, rule)
                        expired_groups.append(key)
                
                # 清理已处理的聚合组
                for key in expired_groups:
                    del self.aggregation_groups[key]
                
                await asyncio.sleep(60)  # 1分钟检查一次
                
            except Exception as e:
                self.logger.error(f"处理聚合超时时出错: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_messages(self):
        """清理旧消息"""
        while self.running:
            try:
                cutoff_time = datetime.now() - timedelta(days=7)  # 保留7天
                
                # 清理已发送消息
                old_count = len(self.sent_messages)
                self.sent_messages = [
                    msg for msg in self.sent_messages
                    if msg.sent_at and msg.sent_at > cutoff_time
                ]
                
                # 清理失败消息
                self.failed_messages = [
                    msg for msg in self.failed_messages
                    if msg.created_at > cutoff_time
                ]
                
                cleaned_count = old_count - len(self.sent_messages)
                if cleaned_count > 0:
                    self.logger.info(f"清理了 {cleaned_count} 条旧消息")
                
                await asyncio.sleep(3600)  # 1小时清理一次
                
            except Exception as e:
                self.logger.error(f"清理旧消息时出错: {e}")
                await asyncio.sleep(3600)
    
    async def get_notification_statistics(self) -> Dict[str, Any]:
        """获取通知统计信息"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        # 统计最近24小时的消息
        recent_sent = [msg for msg in self.sent_messages if msg.sent_at and msg.sent_at > last_24h]
        recent_failed = [msg for msg in self.failed_messages if msg.created_at > last_24h]
        
        # 按渠道统计
        channel_stats = {}
        for channel in NotificationChannel:
            sent_count = len([msg for msg in recent_sent if msg.channel == channel])
            failed_count = len([msg for msg in recent_failed if msg.channel == channel])
            
            channel_stats[channel.value] = {
                "sent": sent_count,
                "failed": failed_count,
                "success_rate": sent_count / (sent_count + failed_count) if (sent_count + failed_count) > 0 else 0
            }
        
        # 按优先级统计
        priority_stats = {}
        for priority in NotificationPriority:
            sent_count = len([msg for msg in recent_sent if msg.priority == priority])
            failed_count = len([msg for msg in recent_failed if msg.priority == priority])
            
            priority_stats[priority.name.lower()] = {
                "sent": sent_count,
                "failed": failed_count
            }
        
        return {
            "total_rules": len(self.rules),
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_templates": len(self.templates),
            "configured_channels": len(self.configs),
            "pending_messages": len(self.pending_messages),
            "sent_messages_24h": len(recent_sent),
            "failed_messages_24h": len(recent_failed),
            "active_aggregation_groups": len(self.aggregation_groups),
            "channel_statistics": channel_stats,
            "priority_statistics": priority_stats,
            "success_rate_24h": len(recent_sent) / (len(recent_sent) + len(recent_failed)) if (len(recent_sent) + len(recent_failed)) > 0 else 0
        }