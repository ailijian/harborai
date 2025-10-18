"""
告警通知管理器

负责管理复杂的告警通知机制，包括多渠道通知、通知模板、通知路由、
通知重试、通知聚合、通知抑制等功能。
"""

import asyncio
import json
import smtplib
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import logging
import aiohttp
import jinja2


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


@dataclass
class NotificationTemplate:
    """通知模板"""
    id: str
    name: str
    channel: NotificationChannel
    subject_template: str
    body_template: str
    format: str = "text"  # text, html, markdown
    variables: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = {}
    
    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """渲染模板"""
        env = jinja2.Environment()
        
        subject = env.from_string(self.subject_template).render(context)
        body = env.from_string(self.body_template).render(context)
        
        return {
            "subject": subject,
            "body": body,
            "format": self.format
        }


@dataclass
class NotificationRule:
    """通知规则"""
    id: str
    name: str
    channels: List[NotificationChannel]
    conditions: List[Dict[str, Any]]  # 匹配条件
    template_id: Optional[str] = None
    recipients: List[str] = None
    priority: NotificationPriority = NotificationPriority.MEDIUM
    rate_limit: Optional[Dict[str, Any]] = None  # 频率限制
    retry_config: Optional[Dict[str, Any]] = None  # 重试配置
    enabled: bool = True
    
    def __post_init__(self):
        if self.recipients is None:
            self.recipients = []


@dataclass
class NotificationConfig:
    """通知配置"""
    channel: NotificationChannel
    config: Dict[str, Any]
    enabled: bool = True
    
    def validate(self) -> List[str]:
        """验证配置"""
        errors = []
        
        if self.channel == NotificationChannel.EMAIL:
            required_fields = ["smtp_host", "smtp_port", "username", "password"]
            for field in required_fields:
                if field not in self.config:
                    errors.append(f"邮件配置缺少字段: {field}")
        
        elif self.channel == NotificationChannel.WEBHOOK:
            if "url" not in self.config:
                errors.append("Webhook配置缺少URL")
        
        elif self.channel == NotificationChannel.SLACK:
            if "webhook_url" not in self.config:
                errors.append("Slack配置缺少webhook_url")
        
        elif self.channel == NotificationChannel.DINGTALK:
            if "webhook_url" not in self.config:
                errors.append("钉钉配置缺少webhook_url")
        
        return errors


@dataclass
class NotificationMessage:
    """通知消息"""
    id: str
    alert_id: str
    channel: NotificationChannel
    recipients: List[str]
    subject: str
    body: str
    format: str = "text"
    priority: NotificationPriority = NotificationPriority.MEDIUM
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class NotificationResult:
    """通知结果"""
    message_id: str
    status: NotificationStatus
    channel: NotificationChannel
    recipients: List[str]
    sent_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class NotificationManager:
    """通知管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 配置
        self.channel_configs: Dict[NotificationChannel, NotificationConfig] = {}
        self.templates: Dict[str, NotificationTemplate] = {}
        self.rules: Dict[str, NotificationRule] = {}
        
        # 状态
        self.pending_messages: List[NotificationMessage] = []
        self.notification_results: List[NotificationResult] = []
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        
        # 自定义通知处理器
        self.custom_handlers: Dict[str, Callable] = {}
        
        # 加载默认配置
        self._load_default_templates()
        self._load_default_rules()
    
    def _load_default_templates(self):
        """加载默认模板"""
        # 邮件模板
        email_template = NotificationTemplate(
            id="default_email",
            name="默认邮件模板",
            channel=NotificationChannel.EMAIL,
            subject_template="[{{ severity|upper }}] {{ rule_name }} - {{ summary }}",
            body_template="""
告警详情：

规则名称: {{ rule_name }}
严重程度: {{ severity }}
状态: {{ status }}
摘要: {{ summary }}
描述: {{ description }}

标签:
{% for key, value in labels.items() %}
- {{ key }}: {{ value }}
{% endfor %}

时间: {{ timestamp }}
告警ID: {{ alert_id }}

{% if annotations %}
注释:
{% for key, value in annotations.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}
            """.strip(),
            format="text"
        )
        
        # Webhook模板
        webhook_template = NotificationTemplate(
            id="default_webhook",
            name="默认Webhook模板",
            channel=NotificationChannel.WEBHOOK,
            subject_template="{{ rule_name }}",
            body_template=json.dumps({
                "alert_id": "{{ alert_id }}",
                "rule_name": "{{ rule_name }}",
                "severity": "{{ severity }}",
                "status": "{{ status }}",
                "summary": "{{ summary }}",
                "description": "{{ description }}",
                "labels": "{{ labels | tojson }}",
                "annotations": "{{ annotations | tojson }}",
                "timestamp": "{{ timestamp }}"
            }, indent=2),
            format="json"
        )
        
        # 钉钉模板
        dingtalk_template = NotificationTemplate(
            id="default_dingtalk",
            name="默认钉钉模板",
            channel=NotificationChannel.DINGTALK,
            subject_template="{{ rule_name }}",
            body_template="""
## 告警通知

**规则名称**: {{ rule_name }}
**严重程度**: {{ severity }}
**状态**: {{ status }}
**摘要**: {{ summary }}

**详细信息**:
{{ description }}

**时间**: {{ timestamp }}
**告警ID**: {{ alert_id }}
            """.strip(),
            format="markdown"
        )
        
        # 控制台模板
        console_template = NotificationTemplate(
            id="default_console",
            name="默认控制台模板",
            channel=NotificationChannel.CONSOLE,
            subject_template="[{{ severity|upper }}] {{ rule_name }}",
            body_template="[{{ timestamp }}] {{ severity|upper }}: {{ summary }} ({{ alert_id }})",
            format="text"
        )
        
        templates = [email_template, webhook_template, dingtalk_template, console_template]
        for template in templates:
            self.templates[template.id] = template
    
    def _load_default_rules(self):
        """加载默认通知规则"""
        # 关键告警规则
        critical_rule = NotificationRule(
            id="critical_alerts",
            name="关键告警通知",
            channels=[NotificationChannel.EMAIL, NotificationChannel.DINGTALK],
            conditions=[
                {"field": "severity", "operator": "eq", "value": "critical"}
            ],
            template_id="default_email",
            priority=NotificationPriority.CRITICAL,
            rate_limit={
                "max_notifications": 5,
                "window_seconds": 300
            },
            retry_config={
                "max_retries": 3,
                "retry_delay": 60,
                "backoff_factor": 2
            }
        )
        
        # 高优先级告警规则
        high_rule = NotificationRule(
            id="high_priority_alerts",
            name="高优先级告警通知",
            channels=[NotificationChannel.EMAIL],
            conditions=[
                {"field": "severity", "operator": "eq", "value": "high"}
            ],
            template_id="default_email",
            priority=NotificationPriority.HIGH,
            rate_limit={
                "max_notifications": 10,
                "window_seconds": 600
            }
        )
        
        # 一般告警规则
        medium_rule = NotificationRule(
            id="medium_priority_alerts",
            name="一般告警通知",
            channels=[NotificationChannel.CONSOLE],
            conditions=[
                {"field": "severity", "operator": "in", "value": ["medium", "low"]}
            ],
            template_id="default_console",
            priority=NotificationPriority.MEDIUM
        )
        
        rules = [critical_rule, high_rule, medium_rule]
        for rule in rules:
            self.rules[rule.id] = rule
    
    async def configure_channel(self, channel: NotificationChannel, config: Dict[str, Any]) -> bool:
        """配置通知渠道"""
        notification_config = NotificationConfig(channel=channel, config=config)
        
        # 验证配置
        errors = notification_config.validate()
        if errors:
            self.logger.error(f"通知渠道配置错误: {errors}")
            return False
        
        self.channel_configs[channel] = notification_config
        self.logger.info(f"配置通知渠道: {channel.value}")
        return True
    
    async def add_template(self, template: NotificationTemplate) -> bool:
        """添加通知模板"""
        self.templates[template.id] = template
        self.logger.info(f"添加通知模板: {template.id}")
        return True
    
    async def add_rule(self, rule: NotificationRule) -> bool:
        """添加通知规则"""
        self.rules[rule.id] = rule
        self.logger.info(f"添加通知规则: {rule.id}")
        return True
    
    async def send_notification(self, alert_data: Dict[str, Any]) -> List[NotificationResult]:
        """发送通知"""
        results = []
        
        # 查找匹配的规则
        matching_rules = await self._find_matching_rules(alert_data)
        
        for rule in matching_rules:
            if not rule.enabled:
                continue
            
            # 检查频率限制
            if await self._is_rate_limited(rule, alert_data):
                self.logger.info(f"通知被频率限制: {rule.id}")
                continue
            
            # 为每个渠道创建通知消息
            for channel in rule.channels:
                if channel not in self.channel_configs:
                    self.logger.warning(f"通知渠道未配置: {channel.value}")
                    continue
                
                if not self.channel_configs[channel].enabled:
                    continue
                
                # 创建通知消息
                message = await self._create_notification_message(rule, channel, alert_data)
                if message:
                    # 发送通知
                    result = await self._send_message(message)
                    results.append(result)
        
        return results
    
    async def _find_matching_rules(self, alert_data: Dict[str, Any]) -> List[NotificationRule]:
        """查找匹配的通知规则"""
        matching_rules = []
        
        for rule in self.rules.values():
            if await self._rule_matches(rule, alert_data):
                matching_rules.append(rule)
        
        # 按优先级排序
        matching_rules.sort(key=lambda r: r.priority.value)
        return matching_rules
    
    async def _rule_matches(self, rule: NotificationRule, alert_data: Dict[str, Any]) -> bool:
        """检查规则是否匹配"""
        for condition in rule.conditions:
            field = condition["field"]
            operator = condition["operator"]
            value = condition["value"]
            
            alert_value = self._get_nested_value(alert_data, field)
            
            if operator == "eq" and alert_value != value:
                return False
            elif operator == "ne" and alert_value == value:
                return False
            elif operator == "in" and alert_value not in value:
                return False
            elif operator == "not_in" and alert_value in value:
                return False
            elif operator == "gt" and (alert_value is None or alert_value <= value):
                return False
            elif operator == "lt" and (alert_value is None or alert_value >= value):
                return False
            elif operator == "gte" and (alert_value is None or alert_value < value):
                return False
            elif operator == "lte" and (alert_value is None or alert_value > value):
                return False
        
        return True
    
    def _get_nested_value(self, data: Dict[str, Any], field: str) -> Any:
        """获取嵌套字段值"""
        keys = field.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    async def _is_rate_limited(self, rule: NotificationRule, alert_data: Dict[str, Any]) -> bool:
        """检查是否被频率限制"""
        if not rule.rate_limit:
            return False
        
        max_notifications = rule.rate_limit.get("max_notifications", 10)
        window_seconds = rule.rate_limit.get("window_seconds", 300)
        
        # 构建限制键
        limit_key = f"{rule.id}:{alert_data.get('rule_id', 'unknown')}"
        
        now = datetime.now()
        if limit_key not in self.rate_limiters:
            self.rate_limiters[limit_key] = {
                "count": 0,
                "window_start": now
            }
        
        limiter = self.rate_limiters[limit_key]
        
        # 检查是否需要重置窗口
        if (now - limiter["window_start"]).total_seconds() >= window_seconds:
            limiter["count"] = 0
            limiter["window_start"] = now
        
        # 检查是否超过限制
        if limiter["count"] >= max_notifications:
            return True
        
        # 增加计数
        limiter["count"] += 1
        return False
    
    async def _create_notification_message(self, rule: NotificationRule, 
                                         channel: NotificationChannel, 
                                         alert_data: Dict[str, Any]) -> Optional[NotificationMessage]:
        """创建通知消息"""
        # 获取模板
        template_id = rule.template_id or f"default_{channel.value}"
        template = self.templates.get(template_id)
        
        if not template:
            self.logger.error(f"通知模板不存在: {template_id}")
            return None
        
        # 准备模板上下文
        context = {
            **alert_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rule_name": rule.name
        }
        
        # 渲染模板
        try:
            rendered = template.render(context)
        except Exception as e:
            self.logger.error(f"模板渲染失败: {e}")
            return None
        
        # 创建消息
        message = NotificationMessage(
            id=f"{alert_data.get('id', 'unknown')}_{channel.value}_{datetime.now().timestamp()}",
            alert_id=alert_data.get('id', 'unknown'),
            channel=channel,
            recipients=rule.recipients,
            subject=rendered["subject"],
            body=rendered["body"],
            format=rendered["format"],
            priority=rule.priority,
            metadata={
                "rule_id": rule.id,
                "template_id": template_id
            }
        )
        
        return message
    
    async def _send_message(self, message: NotificationMessage) -> NotificationResult:
        """发送消息"""
        result = NotificationResult(
            message_id=message.id,
            status=NotificationStatus.PENDING,
            channel=message.channel,
            recipients=message.recipients
        )
        
        try:
            result.status = NotificationStatus.SENDING
            
            if message.channel == NotificationChannel.EMAIL:
                await self._send_email(message, result)
            elif message.channel == NotificationChannel.WEBHOOK:
                await self._send_webhook(message, result)
            elif message.channel == NotificationChannel.SLACK:
                await self._send_slack(message, result)
            elif message.channel == NotificationChannel.DINGTALK:
                await self._send_dingtalk(message, result)
            elif message.channel == NotificationChannel.CONSOLE:
                await self._send_console(message, result)
            elif message.channel == NotificationChannel.FILE:
                await self._send_file(message, result)
            elif message.channel == NotificationChannel.CUSTOM:
                await self._send_custom(message, result)
            else:
                result.status = NotificationStatus.FAILED
                result.error = f"不支持的通知渠道: {message.channel.value}"
            
            if result.status == NotificationStatus.SENDING:
                result.status = NotificationStatus.SENT
                result.sent_at = datetime.now()
            
        except Exception as e:
            result.status = NotificationStatus.FAILED
            result.error = str(e)
            self.logger.error(f"发送通知失败: {e}")
        
        self.notification_results.append(result)
        return result
    
    async def _send_email(self, message: NotificationMessage, result: NotificationResult):
        """发送邮件"""
        config = self.channel_configs[NotificationChannel.EMAIL].config
        
        # 创建邮件
        msg = MimeMultipart()
        msg['From'] = config.get('from_address', config['username'])
        msg['To'] = ', '.join(message.recipients)
        msg['Subject'] = message.subject
        
        # 添加正文
        if message.format == "html":
            msg.attach(MimeText(message.body, 'html'))
        else:
            msg.attach(MimeText(message.body, 'plain'))
        
        # 发送邮件
        context = ssl.create_default_context()
        with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
            if config.get('use_tls', True):
                server.starttls(context=context)
            server.login(config['username'], config['password'])
            server.send_message(msg)
    
    async def _send_webhook(self, message: NotificationMessage, result: NotificationResult):
        """发送Webhook"""
        config = self.channel_configs[NotificationChannel.WEBHOOK].config
        
        # 准备数据
        if message.format == "json":
            try:
                data = json.loads(message.body)
            except json.JSONDecodeError:
                data = {"message": message.body}
        else:
            data = {
                "subject": message.subject,
                "body": message.body,
                "alert_id": message.alert_id,
                "timestamp": message.created_at.isoformat()
            }
        
        # 发送请求
        headers = config.get('headers', {})
        headers.setdefault('Content-Type', 'application/json')
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config['url'],
                json=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status >= 400:
                    raise Exception(f"Webhook请求失败: {response.status}")
    
    async def _send_slack(self, message: NotificationMessage, result: NotificationResult):
        """发送Slack消息"""
        config = self.channel_configs[NotificationChannel.SLACK].config
        
        # 准备Slack消息格式
        slack_data = {
            "text": message.subject,
            "attachments": [
                {
                    "color": self._get_slack_color(message.metadata.get('severity', 'medium')),
                    "text": message.body,
                    "ts": int(message.created_at.timestamp())
                }
            ]
        }
        
        # 发送到Slack
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config['webhook_url'],
                json=slack_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status >= 400:
                    raise Exception(f"Slack消息发送失败: {response.status}")
    
    async def _send_dingtalk(self, message: NotificationMessage, result: NotificationResult):
        """发送钉钉消息"""
        config = self.channel_configs[NotificationChannel.DINGTALK].config
        
        # 准备钉钉消息格式
        dingtalk_data = {
            "msgtype": "markdown",
            "markdown": {
                "title": message.subject,
                "text": message.body
            }
        }
        
        # 发送到钉钉
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config['webhook_url'],
                json=dingtalk_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status >= 400:
                    raise Exception(f"钉钉消息发送失败: {response.status}")
    
    async def _send_console(self, message: NotificationMessage, result: NotificationResult):
        """发送控制台消息"""
        print(f"[ALERT] {message.subject}")
        if message.body != message.subject:
            print(f"        {message.body}")
    
    async def _send_file(self, message: NotificationMessage, result: NotificationResult):
        """发送文件消息"""
        config = self.channel_configs[NotificationChannel.FILE].config
        log_file = config.get('log_file', 'alerts.log')
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{message.created_at.isoformat()}] {message.subject}\n")
            if message.body != message.subject:
                f.write(f"{message.body}\n")
            f.write("\n")
    
    async def _send_custom(self, message: NotificationMessage, result: NotificationResult):
        """发送自定义消息"""
        handler_name = message.metadata.get('custom_handler')
        if handler_name and handler_name in self.custom_handlers:
            handler = self.custom_handlers[handler_name]
            await handler(message, result)
        else:
            raise Exception("自定义处理器未找到")
    
    def _get_slack_color(self, severity: str) -> str:
        """获取Slack颜色"""
        color_map = {
            "critical": "danger",
            "high": "warning",
            "medium": "good",
            "low": "#36a64f"
        }
        return color_map.get(severity, "good")
    
    def register_custom_handler(self, name: str, handler: Callable):
        """注册自定义通知处理器"""
        self.custom_handlers[name] = handler
        self.logger.info(f"注册自定义通知处理器: {name}")
    
    async def retry_failed_notifications(self):
        """重试失败的通知"""
        failed_results = [
            r for r in self.notification_results
            if r.status == NotificationStatus.FAILED and r.retry_count < 3
        ]
        
        for result in failed_results:
            # 查找对应的规则
            rule = None
            for r in self.rules.values():
                if r.id == result.metadata.get('rule_id'):
                    rule = r
                    break
            
            if not rule or not rule.retry_config:
                continue
            
            max_retries = rule.retry_config.get('max_retries', 3)
            if result.retry_count >= max_retries:
                continue
            
            # 计算重试延迟
            retry_delay = rule.retry_config.get('retry_delay', 60)
            backoff_factor = rule.retry_config.get('backoff_factor', 1)
            delay = retry_delay * (backoff_factor ** result.retry_count)
            
            # 等待重试
            await asyncio.sleep(delay)
            
            # 重新发送（这里需要重新构造消息，简化处理）
            result.retry_count += 1
            result.status = NotificationStatus.RETRYING
            
            self.logger.info(f"重试通知: {result.message_id}, 第{result.retry_count}次")
    
    async def get_notification_statistics(self) -> Dict[str, Any]:
        """获取通知统计信息"""
        total_notifications = len(self.notification_results)
        
        # 按状态统计
        status_stats = {}
        for status in NotificationStatus:
            count = len([r for r in self.notification_results if r.status == status])
            status_stats[status.value] = count
        
        # 按渠道统计
        channel_stats = {}
        for channel in NotificationChannel:
            count = len([r for r in self.notification_results if r.channel == channel])
            channel_stats[channel.value] = count
        
        # 成功率统计
        sent_count = status_stats.get('sent', 0)
        success_rate = (sent_count / total_notifications * 100) if total_notifications > 0 else 0
        
        # 最近24小时统计
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_notifications = [
            r for r in self.notification_results
            if r.sent_at and r.sent_at > recent_cutoff
        ]
        
        return {
            "total_notifications": total_notifications,
            "status_distribution": status_stats,
            "channel_distribution": channel_stats,
            "success_rate": round(success_rate, 2),
            "recent_24h_count": len(recent_notifications),
            "configured_channels": len(self.channel_configs),
            "active_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_templates": len(self.templates)
        }
    
    async def cleanup_old_results(self, retention_days: int = 30):
        """清理过期的通知结果"""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        old_count = len(self.notification_results)
        self.notification_results = [
            result for result in self.notification_results
            if result.sent_at and result.sent_at > cutoff_time
        ]
        
        cleaned_count = old_count - len(self.notification_results)
        self.logger.info(f"清理了 {cleaned_count} 个过期通知结果")