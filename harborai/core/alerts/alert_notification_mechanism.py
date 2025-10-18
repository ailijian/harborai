"""
告警通知机制管理器

负责管理告警通知的发送机制、路由规则、重试策略、聚合逻辑和通知渠道，
支持多种通知方式、智能路由、失败重试和通知抑制。
"""

import json
import asyncio
import aiohttp
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
from collections import defaultdict, deque
import hashlib
import re
import time
from urllib.parse import urljoin
import ssl


class NotificationChannel(Enum):
    """通知渠道"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    DINGTALK = "dingtalk"
    WECHAT = "wechat"
    SMS = "sms"
    PHONE = "phone"
    CONSOLE = "console"
    FILE = "file"
    CUSTOM = "custom"


class NotificationPriority(Enum):
    """通知优先级"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class NotificationStatus(Enum):
    """通知状态"""
    PENDING = "pending"
    SENDING = "sending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


class EscalationLevel(Enum):
    """升级级别"""
    LEVEL_1 = "level_1"
    LEVEL_2 = "level_2"
    LEVEL_3 = "level_3"
    LEVEL_4 = "level_4"
    EXECUTIVE = "executive"


class AggregationStrategy(Enum):
    """聚合策略"""
    NONE = "none"                        # 不聚合
    BY_RULE = "by_rule"                  # 按规则聚合
    BY_SEVERITY = "by_severity"          # 按严重级别聚合
    BY_SOURCE = "by_source"              # 按来源聚合
    BY_TIME = "by_time"                  # 按时间聚合
    BY_LABELS = "by_labels"              # 按标签聚合
    CUSTOM = "custom"                    # 自定义聚合


@dataclass
class TimeWindow:
    """时间窗口"""
    duration_seconds: int
    start_time: Optional[datetime] = None
    
    def is_active(self, current_time: datetime = None) -> bool:
        """检查时间窗口是否活跃"""
        if not self.start_time:
            return True
        
        if current_time is None:
            current_time = datetime.now()
        
        return (current_time - self.start_time).total_seconds() <= self.duration_seconds


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    max_notifications: int               # 最大通知数
    time_window_seconds: int             # 时间窗口（秒）
    burst_limit: int = 0                 # 突发限制
    
    def __post_init__(self):
        if self.burst_limit == 0:
            self.burst_limit = self.max_notifications


@dataclass
class EscalationRule:
    """升级规则"""
    level: EscalationLevel
    delay_seconds: int                   # 升级延迟
    channels: List[NotificationChannel]  # 升级通知渠道
    recipients: List[str]                # 升级接收者
    conditions: Dict[str, Any] = field(default_factory=dict)  # 升级条件


@dataclass
class NotificationTemplate:
    """通知模板"""
    id: str
    name: str
    channel: NotificationChannel
    
    # 模板内容
    subject_template: str                # 主题模板
    body_template: str                   # 正文模板
    
    # 格式配置
    format_type: str = "text"            # 格式类型: text, html, markdown
    encoding: str = "utf-8"              # 编码
    
    # 元数据
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def render(self, context: Dict[str, Any]) -> Dict[str, str]:
        """渲染模板"""
        try:
            # 简单的模板渲染（可以替换为更复杂的模板引擎）
            subject = self.subject_template
            body = self.body_template
            
            for key, value in context.items():
                placeholder = f"{{{key}}}"
                subject = subject.replace(placeholder, str(value))
                body = body.replace(placeholder, str(value))
            
            return {
                "subject": subject,
                "body": body,
                "format": self.format_type
            }
            
        except Exception:
            return {
                "subject": f"告警通知 - {context.get('rule_name', 'Unknown')}",
                "body": f"告警详情: {context}",
                "format": "text"
            }


@dataclass
class NotificationRule:
    """通知规则"""
    id: str
    name: str
    description: str
    
    # 匹配条件
    rule_patterns: List[str] = field(default_factory=list)  # 规则模式
    severity_levels: List[NotificationPriority] = field(default_factory=list)  # 严重级别
    metric_patterns: List[str] = field(default_factory=list)  # 指标模式
    label_selectors: Dict[str, str] = field(default_factory=dict)  # 标签选择器
    
    # 通知配置
    channels: List[NotificationChannel] = field(default_factory=list)  # 通知渠道
    recipients: List[str] = field(default_factory=list)  # 接收者
    template_id: Optional[str] = None    # 模板ID
    
    # 时间配置
    quiet_hours: Optional[TimeWindow] = None  # 静默时间
    business_hours_only: bool = False    # 仅工作时间
    
    # 限制配置
    rate_limit: Optional[RateLimitConfig] = None  # 速率限制
    max_notifications: int = 0           # 最大通知数（0表示无限制）
    
    # 升级配置
    escalation_rules: List[EscalationRule] = field(default_factory=list)  # 升级规则
    auto_resolve_timeout: int = 0        # 自动解决超时（秒）
    
    # 聚合配置
    aggregation_strategy: AggregationStrategy = AggregationStrategy.NONE
    aggregation_window: int = 300        # 聚合窗口（秒）
    aggregation_threshold: int = 1       # 聚合阈值
    
    # 元数据
    enabled: bool = True
    priority: int = 100                  # 规则优先级（数字越小优先级越高）
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def matches(self, alert_data: Dict[str, Any]) -> bool:
        """检查规则是否匹配告警"""
        if not self.enabled:
            return False
        
        # 检查规则模式
        if self.rule_patterns:
            rule_name = alert_data.get("rule_name", "")
            if not any(self._match_pattern(pattern, rule_name) for pattern in self.rule_patterns):
                return False
        
        # 检查严重级别
        if self.severity_levels:
            severity = alert_data.get("severity")
            if severity not in [s.value for s in self.severity_levels]:
                return False
        
        # 检查指标模式
        if self.metric_patterns:
            metric_name = alert_data.get("metric_name", "")
            if not any(self._match_pattern(pattern, metric_name) for pattern in self.metric_patterns):
                return False
        
        # 检查标签选择器
        if self.label_selectors:
            labels = alert_data.get("labels", {})
            for key, value in self.label_selectors.items():
                if key not in labels or not self._match_pattern(value, labels[key]):
                    return False
        
        return True
    
    def _match_pattern(self, pattern: str, text: str) -> bool:
        """匹配模式"""
        if pattern.startswith("regex:"):
            # 正则表达式匹配
            regex_pattern = pattern[6:]
            return bool(re.match(regex_pattern, text))
        elif "*" in pattern or "?" in pattern:
            # 通配符匹配
            import fnmatch
            return fnmatch.fnmatch(text, pattern)
        else:
            # 精确匹配
            return pattern == text


@dataclass
class ChannelConfig:
    """渠道配置"""
    channel: NotificationChannel
    enabled: bool = True
    
    # 连接配置
    endpoint: Optional[str] = None       # 端点URL
    credentials: Dict[str, str] = field(default_factory=dict)  # 认证信息
    headers: Dict[str, str] = field(default_factory=dict)  # 请求头
    
    # 重试配置
    max_retries: int = 3                 # 最大重试次数
    retry_delay: int = 60                # 重试延迟（秒）
    retry_backoff: float = 2.0           # 重试退避因子
    
    # 超时配置
    connect_timeout: int = 10            # 连接超时
    read_timeout: int = 30               # 读取超时
    
    # 格式配置
    message_format: str = "json"         # 消息格式
    encoding: str = "utf-8"              # 编码
    
    # 特定配置
    extra_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationMessage:
    """通知消息"""
    id: str
    rule_id: str
    alert_id: str
    
    # 消息内容
    subject: str
    body: str
    format_type: str = "text"
    
    # 发送配置
    channel: NotificationChannel
    recipients: List[str] = field(default_factory=list)
    priority: NotificationPriority = NotificationPriority.MEDIUM
    
    # 状态信息
    status: NotificationStatus = NotificationStatus.PENDING
    attempts: int = 0
    max_attempts: int = 3
    
    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    
    # 聚合信息
    aggregation_key: Optional[str] = None
    aggregated_count: int = 1
    
    # 元数据
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def generate_id(self) -> str:
        """生成消息ID"""
        content = f"{self.rule_id}:{self.alert_id}:{self.created_at.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def is_expired(self, expiry_hours: int = 24) -> bool:
        """检查消息是否过期"""
        expiry_time = self.created_at + timedelta(hours=expiry_hours)
        return datetime.now() > expiry_time
    
    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return (self.status == NotificationStatus.FAILED and 
                self.attempts < self.max_attempts)


@dataclass
class NotificationResult:
    """通知结果"""
    message_id: str
    status: NotificationStatus
    sent_at: Optional[datetime] = None
    error_message: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    duration_ms: int = 0


class NotificationAggregator:
    """通知聚合器"""
    
    def __init__(self):
        self.aggregation_groups: Dict[str, List[NotificationMessage]] = defaultdict(list)
        self.group_timers: Dict[str, datetime] = {}
    
    def add_message(self, message: NotificationMessage, rule: NotificationRule) -> Optional[str]:
        """添加消息到聚合组"""
        if rule.aggregation_strategy == AggregationStrategy.NONE:
            return None
        
        # 生成聚合键
        aggregation_key = self._generate_aggregation_key(message, rule)
        message.aggregation_key = aggregation_key
        
        # 添加到聚合组
        self.aggregation_groups[aggregation_key].append(message)
        
        # 设置定时器
        if aggregation_key not in self.group_timers:
            self.group_timers[aggregation_key] = datetime.now()
        
        return aggregation_key
    
    def _generate_aggregation_key(self, message: NotificationMessage, rule: NotificationRule) -> str:
        """生成聚合键"""
        key_parts = []
        
        if rule.aggregation_strategy == AggregationStrategy.BY_RULE:
            key_parts.append(f"rule:{message.rule_id}")
        elif rule.aggregation_strategy == AggregationStrategy.BY_SEVERITY:
            key_parts.append(f"severity:{message.priority.value}")
        elif rule.aggregation_strategy == AggregationStrategy.BY_SOURCE:
            source = message.labels.get("source", "unknown")
            key_parts.append(f"source:{source}")
        elif rule.aggregation_strategy == AggregationStrategy.BY_LABELS:
            # 使用指定标签进行聚合
            label_keys = rule.label_selectors.keys()
            for key in sorted(label_keys):
                if key in message.labels:
                    key_parts.append(f"{key}:{message.labels[key]}")
        
        # 添加时间窗口
        if rule.aggregation_strategy != AggregationStrategy.NONE:
            window_start = int(time.time() // rule.aggregation_window) * rule.aggregation_window
            key_parts.append(f"window:{window_start}")
        
        return ":".join(key_parts)
    
    def get_ready_groups(self, rule: NotificationRule) -> List[Tuple[str, List[NotificationMessage]]]:
        """获取准备发送的聚合组"""
        ready_groups = []
        current_time = datetime.now()
        
        for aggregation_key, messages in list(self.aggregation_groups.items()):
            group_start_time = self.group_timers.get(aggregation_key)
            if not group_start_time:
                continue
            
            # 检查是否达到聚合阈值或时间窗口
            time_elapsed = (current_time - group_start_time).total_seconds()
            
            if (len(messages) >= rule.aggregation_threshold or 
                time_elapsed >= rule.aggregation_window):
                
                ready_groups.append((aggregation_key, messages))
                
                # 清理已处理的组
                del self.aggregation_groups[aggregation_key]
                del self.group_timers[aggregation_key]
        
        return ready_groups
    
    def create_aggregated_message(self, aggregation_key: str, 
                                messages: List[NotificationMessage]) -> NotificationMessage:
        """创建聚合消息"""
        if not messages:
            raise ValueError("消息列表不能为空")
        
        first_message = messages[0]
        
        # 创建聚合消息
        aggregated_message = NotificationMessage(
            id=f"agg_{hashlib.md5(aggregation_key.encode()).hexdigest()[:12]}",
            rule_id=first_message.rule_id,
            alert_id=f"aggregated_{len(messages)}_alerts",
            subject=f"聚合告警通知 ({len(messages)} 个告警)",
            body=self._create_aggregated_body(messages),
            channel=first_message.channel,
            recipients=first_message.recipients,
            priority=self._get_highest_priority(messages),
            aggregation_key=aggregation_key,
            aggregated_count=len(messages),
            labels=first_message.labels,
            metadata={
                "aggregated_message_ids": [msg.id for msg in messages],
                "aggregation_key": aggregation_key
            }
        )
        
        return aggregated_message
    
    def _create_aggregated_body(self, messages: List[NotificationMessage]) -> str:
        """创建聚合消息正文"""
        body_parts = [
            f"聚合了 {len(messages)} 个告警通知:",
            ""
        ]
        
        for i, message in enumerate(messages, 1):
            body_parts.append(f"{i}. {message.subject}")
            if message.body:
                # 截取前100个字符
                preview = message.body[:100]
                if len(message.body) > 100:
                    preview += "..."
                body_parts.append(f"   {preview}")
            body_parts.append("")
        
        return "\n".join(body_parts)
    
    def _get_highest_priority(self, messages: List[NotificationMessage]) -> NotificationPriority:
        """获取最高优先级"""
        priority_order = {
            NotificationPriority.CRITICAL: 5,
            NotificationPriority.HIGH: 4,
            NotificationPriority.MEDIUM: 3,
            NotificationPriority.LOW: 2,
            NotificationPriority.INFO: 1
        }
        
        highest_priority = NotificationPriority.INFO
        highest_value = 0
        
        for message in messages:
            value = priority_order.get(message.priority, 0)
            if value > highest_value:
                highest_value = value
                highest_priority = message.priority
        
        return highest_priority


class AlertNotificationMechanism:
    """告警通知机制管理器"""
    
    def __init__(self, config_dir: str = "config/alerts"):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置存储
        self.templates: Dict[str, NotificationTemplate] = {}
        self.rules: Dict[str, NotificationRule] = {}
        self.channel_configs: Dict[NotificationChannel, ChannelConfig] = {}
        
        # 消息队列和状态
        self.pending_messages: deque = deque()
        self.sent_messages: deque = deque(maxlen=10000)
        self.failed_messages: deque = deque(maxlen=1000)
        
        # 聚合器
        self.aggregator = NotificationAggregator()
        
        # 速率限制
        self.rate_limiters: Dict[str, deque] = defaultdict(lambda: deque())
        
        # 自定义处理器
        self.custom_handlers: Dict[str, Callable] = {}
        
        # 配置文件
        self.templates_file = self.config_dir / "notification_templates.json"
        self.rules_file = self.config_dir / "notification_rules.json"
        self.channels_file = self.config_dir / "notification_channels.json"
        
        # 加载配置
        self._load_default_templates()
        self._load_default_rules()
        self._load_default_channels()
        self._load_configurations()
    
    def _load_default_templates(self):
        """加载默认通知模板"""
        default_templates = [
            # 邮件模板
            NotificationTemplate(
                id="email_critical_alert",
                name="邮件关键告警模板",
                channel=NotificationChannel.EMAIL,
                subject_template="🚨 关键告警: {rule_name}",
                body_template="""
告警详情:
- 规则名称: {rule_name}
- 指标名称: {metric_name}
- 当前值: {current_value}
- 阈值: {threshold_value}
- 严重级别: {severity}
- 触发时间: {trigger_time}
- 持续时间: {duration}

标签信息:
{labels}

描述: {description}

请及时处理此告警。
                """.strip(),
                format_type="text",
                description="关键告警邮件模板",
                tags=["email", "critical"]
            ),
            
            NotificationTemplate(
                id="email_general_alert",
                name="邮件通用告警模板",
                channel=NotificationChannel.EMAIL,
                subject_template="⚠️ 告警通知: {rule_name}",
                body_template="""
告警信息:
- 规则: {rule_name}
- 指标: {metric_name}
- 当前值: {current_value}
- 阈值: {threshold_value}
- 级别: {severity}
- 时间: {trigger_time}

{description}
                """.strip(),
                format_type="text",
                description="通用告警邮件模板",
                tags=["email", "general"]
            ),
            
            # Webhook模板
            NotificationTemplate(
                id="webhook_alert",
                name="Webhook告警模板",
                channel=NotificationChannel.WEBHOOK,
                subject_template="{rule_name}",
                body_template="""{
    "alert_id": "{alert_id}",
    "rule_name": "{rule_name}",
    "metric_name": "{metric_name}",
    "current_value": {current_value},
    "threshold_value": {threshold_value},
    "severity": "{severity}",
    "trigger_time": "{trigger_time}",
    "labels": {labels_json},
    "description": "{description}"
}""",
                format_type="json",
                description="Webhook告警JSON模板",
                tags=["webhook", "json"]
            ),
            
            # 钉钉模板
            NotificationTemplate(
                id="dingtalk_alert",
                name="钉钉告警模板",
                channel=NotificationChannel.DINGTALK,
                subject_template="{rule_name}",
                body_template="""## {severity_emoji} {rule_name}

**指标名称:** {metric_name}
**当前值:** {current_value}
**阈值:** {threshold_value}
**严重级别:** {severity}
**触发时间:** {trigger_time}

{description}
                """.strip(),
                format_type="markdown",
                description="钉钉告警Markdown模板",
                tags=["dingtalk", "markdown"]
            ),
            
            # Slack模板
            NotificationTemplate(
                id="slack_alert",
                name="Slack告警模板",
                channel=NotificationChannel.SLACK,
                subject_template="{rule_name}",
                body_template="""{
    "text": "{severity_emoji} {rule_name}",
    "attachments": [
        {
            "color": "{color}",
            "fields": [
                {"title": "指标", "value": "{metric_name}", "short": true},
                {"title": "当前值", "value": "{current_value}", "short": true},
                {"title": "阈值", "value": "{threshold_value}", "short": true},
                {"title": "级别", "value": "{severity}", "short": true},
                {"title": "时间", "value": "{trigger_time}", "short": false}
            ],
            "footer": "HarborAI 监控系统",
            "ts": {timestamp}
        }
    ]
}""",
                format_type="json",
                description="Slack告警JSON模板",
                tags=["slack", "json"]
            ),
            
            # 控制台模板
            NotificationTemplate(
                id="console_alert",
                name="控制台告警模板",
                channel=NotificationChannel.CONSOLE,
                subject_template="[{severity}] {rule_name}",
                body_template="[{trigger_time}] {severity_emoji} {rule_name}: {metric_name}={current_value} (阈值: {threshold_value})",
                format_type="text",
                description="控制台告警模板",
                tags=["console", "text"]
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
    
    def _load_default_rules(self):
        """加载默认通知规则"""
        default_rules = [
            # 关键告警规则
            NotificationRule(
                id="critical_alerts",
                name="关键告警通知",
                description="所有关键级别告警的通知规则",
                severity_levels=[NotificationPriority.CRITICAL],
                channels=[NotificationChannel.EMAIL, NotificationChannel.DINGTALK, NotificationChannel.SLACK],
                recipients=["admin@company.com", "ops-team@company.com"],
                template_id="email_critical_alert",
                rate_limit=RateLimitConfig(
                    max_notifications=10,
                    time_window_seconds=3600,  # 1小时内最多10条
                    burst_limit=3
                ),
                escalation_rules=[
                    EscalationRule(
                        level=EscalationLevel.LEVEL_1,
                        delay_seconds=300,  # 5分钟后升级
                        channels=[NotificationChannel.PHONE],
                        recipients=["oncall@company.com"]
                    ),
                    EscalationRule(
                        level=EscalationLevel.LEVEL_2,
                        delay_seconds=900,  # 15分钟后升级
                        channels=[NotificationChannel.SMS],
                        recipients=["manager@company.com"]
                    )
                ],
                priority=1,
                tags=["critical", "escalation"]
            ),
            
            # 高级告警规则
            NotificationRule(
                id="high_alerts",
                name="高级告警通知",
                description="高级告警的通知规则",
                severity_levels=[NotificationPriority.HIGH],
                channels=[NotificationChannel.EMAIL, NotificationChannel.DINGTALK],
                recipients=["ops-team@company.com"],
                template_id="email_general_alert",
                rate_limit=RateLimitConfig(
                    max_notifications=20,
                    time_window_seconds=3600
                ),
                aggregation_strategy=AggregationStrategy.BY_RULE,
                aggregation_window=300,  # 5分钟聚合窗口
                aggregation_threshold=3,  # 3个告警聚合
                priority=2,
                tags=["high", "aggregation"]
            ),
            
            # 业务时间告警规则
            NotificationRule(
                id="business_hours_alerts",
                name="业务时间告警",
                description="仅在业务时间发送的告警",
                severity_levels=[NotificationPriority.MEDIUM, NotificationPriority.LOW],
                channels=[NotificationChannel.EMAIL],
                recipients=["dev-team@company.com"],
                template_id="email_general_alert",
                business_hours_only=True,
                quiet_hours=TimeWindow(duration_seconds=8*3600),  # 8小时静默
                aggregation_strategy=AggregationStrategy.BY_SEVERITY,
                aggregation_window=1800,  # 30分钟聚合
                priority=3,
                tags=["business", "medium", "low"]
            ),
            
            # 系统监控告警规则
            NotificationRule(
                id="system_monitoring",
                name="系统监控告警",
                description="系统基础设施监控告警",
                metric_patterns=["cpu_*", "memory_*", "disk_*", "network_*"],
                channels=[NotificationChannel.WEBHOOK, NotificationChannel.CONSOLE],
                recipients=["http://monitoring.company.com/webhook"],
                template_id="webhook_alert",
                rate_limit=RateLimitConfig(
                    max_notifications=50,
                    time_window_seconds=3600
                ),
                aggregation_strategy=AggregationStrategy.BY_SOURCE,
                aggregation_window=600,  # 10分钟聚合
                priority=4,
                tags=["system", "infrastructure"]
            ),
            
            # 应用性能告警规则
            NotificationRule(
                id="application_performance",
                name="应用性能告警",
                description="应用性能相关告警",
                metric_patterns=["api_*", "response_time_*", "error_rate_*"],
                label_selectors={"service": "harborai"},
                channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
                recipients=["#alerts", "dev-team@company.com"],
                template_id="slack_alert",
                aggregation_strategy=AggregationStrategy.BY_LABELS,
                aggregation_window=300,
                priority=5,
                tags=["application", "performance"]
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
    
    def _load_default_channels(self):
        """加载默认渠道配置"""
        default_channels = {
            NotificationChannel.EMAIL: ChannelConfig(
                channel=NotificationChannel.EMAIL,
                endpoint="smtp://localhost:587",
                credentials={
                    "username": "alerts@company.com",
                    "password": "password"
                },
                extra_config={
                    "smtp_server": "smtp.company.com",
                    "smtp_port": 587,
                    "use_tls": True,
                    "from_address": "alerts@company.com",
                    "from_name": "HarborAI 告警系统"
                },
                max_retries=3,
                retry_delay=60
            ),
            
            NotificationChannel.WEBHOOK: ChannelConfig(
                channel=NotificationChannel.WEBHOOK,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "HarborAI-AlertSystem/1.0"
                },
                max_retries=3,
                retry_delay=30,
                connect_timeout=10,
                read_timeout=30
            ),
            
            NotificationChannel.DINGTALK: ChannelConfig(
                channel=NotificationChannel.DINGTALK,
                endpoint="https://oapi.dingtalk.com/robot/send",
                headers={"Content-Type": "application/json"},
                extra_config={
                    "access_token": "your_dingtalk_token",
                    "secret": "your_dingtalk_secret"
                },
                max_retries=2,
                retry_delay=30
            ),
            
            NotificationChannel.SLACK: ChannelConfig(
                channel=NotificationChannel.SLACK,
                endpoint="https://hooks.slack.com/services/",
                headers={"Content-Type": "application/json"},
                credentials={
                    "webhook_url": "your_slack_webhook_url"
                },
                max_retries=2,
                retry_delay=30
            ),
            
            NotificationChannel.CONSOLE: ChannelConfig(
                channel=NotificationChannel.CONSOLE,
                enabled=True
            ),
            
            NotificationChannel.FILE: ChannelConfig(
                channel=NotificationChannel.FILE,
                extra_config={
                    "file_path": "logs/alerts.log",
                    "max_file_size": 10 * 1024 * 1024,  # 10MB
                    "backup_count": 5
                }
            )
        }
        
        self.channel_configs.update(default_channels)
    
    def _load_configurations(self):
        """加载配置文件"""
        # 加载模板
        if self.templates_file.exists():
            try:
                with open(self.templates_file, 'r', encoding='utf-8') as f:
                    templates_data = json.load(f)
                
                for template_data in templates_data:
                    template = self._dict_to_template(template_data)
                    if template:
                        self.templates[template.id] = template
                
                self.logger.info(f"加载了 {len(self.templates)} 个通知模板")
                
            except Exception as e:
                self.logger.error(f"加载通知模板失败: {e}")
        
        # 加载规则
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                
                for rule_data in rules_data:
                    rule = self._dict_to_rule(rule_data)
                    if rule:
                        self.rules[rule.id] = rule
                
                self.logger.info(f"加载了 {len(self.rules)} 个通知规则")
                
            except Exception as e:
                self.logger.error(f"加载通知规则失败: {e}")
        
        # 加载渠道配置
        if self.channels_file.exists():
            try:
                with open(self.channels_file, 'r', encoding='utf-8') as f:
                    channels_data = json.load(f)
                
                for channel_data in channels_data:
                    config = self._dict_to_channel_config(channel_data)
                    if config:
                        self.channel_configs[config.channel] = config
                
                self.logger.info(f"加载了 {len(self.channel_configs)} 个渠道配置")
                
            except Exception as e:
                self.logger.error(f"加载渠道配置失败: {e}")
    
    def _dict_to_template(self, template_data: Dict[str, Any]) -> Optional[NotificationTemplate]:
        """将字典转换为模板对象"""
        try:
            if "channel" in template_data:
                template_data["channel"] = NotificationChannel(template_data["channel"])
            
            if "created_at" in template_data:
                template_data["created_at"] = datetime.fromisoformat(template_data["created_at"])
            
            return NotificationTemplate(**template_data)
            
        except Exception as e:
            self.logger.error(f"转换通知模板失败: {e}")
            return None
    
    def _dict_to_rule(self, rule_data: Dict[str, Any]) -> Optional[NotificationRule]:
        """将字典转换为规则对象"""
        try:
            # 转换枚举类型
            if "severity_levels" in rule_data:
                rule_data["severity_levels"] = [
                    NotificationPriority(level) for level in rule_data["severity_levels"]
                ]
            
            if "channels" in rule_data:
                rule_data["channels"] = [
                    NotificationChannel(channel) for channel in rule_data["channels"]
                ]
            
            if "aggregation_strategy" in rule_data:
                rule_data["aggregation_strategy"] = AggregationStrategy(rule_data["aggregation_strategy"])
            
            # 转换时间窗口
            if "quiet_hours" in rule_data and rule_data["quiet_hours"]:
                quiet_hours_data = rule_data["quiet_hours"]
                if "start_time" in quiet_hours_data and quiet_hours_data["start_time"]:
                    quiet_hours_data["start_time"] = datetime.fromisoformat(quiet_hours_data["start_time"])
                rule_data["quiet_hours"] = TimeWindow(**quiet_hours_data)
            
            # 转换速率限制
            if "rate_limit" in rule_data and rule_data["rate_limit"]:
                rule_data["rate_limit"] = RateLimitConfig(**rule_data["rate_limit"])
            
            # 转换升级规则
            if "escalation_rules" in rule_data:
                escalation_rules = []
                for escalation_data in rule_data["escalation_rules"]:
                    escalation_data["level"] = EscalationLevel(escalation_data["level"])
                    escalation_data["channels"] = [
                        NotificationChannel(channel) for channel in escalation_data["channels"]
                    ]
                    escalation_rules.append(EscalationRule(**escalation_data))
                rule_data["escalation_rules"] = escalation_rules
            
            # 转换日期时间
            if "created_at" in rule_data:
                rule_data["created_at"] = datetime.fromisoformat(rule_data["created_at"])
            if "updated_at" in rule_data:
                rule_data["updated_at"] = datetime.fromisoformat(rule_data["updated_at"])
            
            return NotificationRule(**rule_data)
            
        except Exception as e:
            self.logger.error(f"转换通知规则失败: {e}")
            return None
    
    def _dict_to_channel_config(self, config_data: Dict[str, Any]) -> Optional[ChannelConfig]:
        """将字典转换为渠道配置对象"""
        try:
            if "channel" in config_data:
                config_data["channel"] = NotificationChannel(config_data["channel"])
            
            return ChannelConfig(**config_data)
            
        except Exception as e:
            self.logger.error(f"转换渠道配置失败: {e}")
            return None
    
    async def send_notification(self, alert_data: Dict[str, Any]) -> List[NotificationResult]:
        """发送通知"""
        results = []
        
        # 查找匹配的规则
        matching_rules = self._find_matching_rules(alert_data)
        
        if not matching_rules:
            self.logger.debug(f"没有找到匹配的通知规则: {alert_data.get('rule_name', 'Unknown')}")
            return results
        
        # 为每个匹配的规则创建通知消息
        for rule in matching_rules:
            try:
                # 检查速率限制
                if not self._check_rate_limit(rule, alert_data):
                    self.logger.info(f"规则 {rule.id} 触发速率限制，跳过通知")
                    continue
                
                # 创建通知消息
                messages = await self._create_notification_messages(rule, alert_data)
                
                for message in messages:
                    # 检查聚合
                    aggregation_key = self.aggregator.add_message(message, rule)
                    
                    if aggregation_key:
                        # 消息被聚合，暂不发送
                        self.logger.debug(f"消息被聚合: {message.id} -> {aggregation_key}")
                        continue
                    
                    # 立即发送消息
                    result = await self._send_message(message)
                    results.append(result)
                    
            except Exception as e:
                self.logger.error(f"处理通知规则失败 {rule.id}: {e}")
                results.append(NotificationResult(
                    message_id="unknown",
                    status=NotificationStatus.FAILED,
                    error_message=str(e)
                ))
        
        return results
    
    def _find_matching_rules(self, alert_data: Dict[str, Any]) -> List[NotificationRule]:
        """查找匹配的通知规则"""
        matching_rules = []
        
        for rule in self.rules.values():
            if rule.matches(alert_data):
                matching_rules.append(rule)
        
        # 按优先级排序
        matching_rules.sort(key=lambda r: r.priority)
        
        return matching_rules
    
    def _check_rate_limit(self, rule: NotificationRule, alert_data: Dict[str, Any]) -> bool:
        """检查速率限制"""
        if not rule.rate_limit:
            return True
        
        # 生成速率限制键
        rate_key = f"{rule.id}:{alert_data.get('rule_name', 'unknown')}"
        
        current_time = datetime.now()
        rate_limiter = self.rate_limiters[rate_key]
        
        # 清理过期的记录
        cutoff_time = current_time - timedelta(seconds=rule.rate_limit.time_window_seconds)
        while rate_limiter and rate_limiter[0] < cutoff_time:
            rate_limiter.popleft()
        
        # 检查是否超过限制
        if len(rate_limiter) >= rule.rate_limit.max_notifications:
            return False
        
        # 记录当前通知
        rate_limiter.append(current_time)
        
        return True
    
    async def _create_notification_messages(self, rule: NotificationRule, 
                                          alert_data: Dict[str, Any]) -> List[NotificationMessage]:
        """创建通知消息"""
        messages = []
        
        # 获取模板
        template = self.templates.get(rule.template_id)
        if not template:
            # 使用默认模板
            template = self._get_default_template(rule.channels[0] if rule.channels else NotificationChannel.CONSOLE)
        
        # 准备模板上下文
        context = self._prepare_template_context(alert_data)
        
        # 渲染模板
        rendered = template.render(context)
        
        # 为每个渠道创建消息
        for channel in rule.channels:
            # 获取渠道特定的模板
            channel_template = self._get_channel_template(channel, rule.template_id)
            if channel_template and channel_template != template:
                rendered = channel_template.render(context)
            
            message = NotificationMessage(
                id=self._generate_message_id(rule, alert_data, channel),
                rule_id=rule.id,
                alert_id=alert_data.get("alert_id", "unknown"),
                subject=rendered["subject"],
                body=rendered["body"],
                format_type=rendered["format"],
                channel=channel,
                recipients=rule.recipients.copy(),
                priority=self._get_notification_priority(alert_data),
                max_attempts=self.channel_configs.get(channel, ChannelConfig(channel)).max_retries,
                labels=alert_data.get("labels", {}),
                metadata={
                    "rule_name": alert_data.get("rule_name", ""),
                    "metric_name": alert_data.get("metric_name", ""),
                    "severity": alert_data.get("severity", ""),
                    "template_id": template.id
                }
            )
            
            messages.append(message)
        
        return messages
    
    def _get_default_template(self, channel: NotificationChannel) -> NotificationTemplate:
        """获取默认模板"""
        default_templates = {
            NotificationChannel.EMAIL: "email_general_alert",
            NotificationChannel.WEBHOOK: "webhook_alert",
            NotificationChannel.DINGTALK: "dingtalk_alert",
            NotificationChannel.SLACK: "slack_alert",
            NotificationChannel.CONSOLE: "console_alert"
        }
        
        template_id = default_templates.get(channel, "console_alert")
        return self.templates.get(template_id, self.templates["console_alert"])
    
    def _get_channel_template(self, channel: NotificationChannel, 
                            template_id: Optional[str]) -> Optional[NotificationTemplate]:
        """获取渠道特定模板"""
        if not template_id:
            return None
        
        # 查找渠道特定模板
        channel_template_id = f"{template_id}_{channel.value}"
        return self.templates.get(channel_template_id)
    
    def _prepare_template_context(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """准备模板上下文"""
        context = alert_data.copy()
        
        # 添加格式化的时间
        if "trigger_time" in context:
            if isinstance(context["trigger_time"], datetime):
                context["trigger_time"] = context["trigger_time"].strftime("%Y-%m-%d %H:%M:%S")
        else:
            context["trigger_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 添加严重级别表情符号
        severity_emojis = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️",
            "info": "📝"
        }
        
        severity = context.get("severity", "info").lower()
        context["severity_emoji"] = severity_emojis.get(severity, "📋")
        
        # 添加颜色（用于Slack等）
        severity_colors = {
            "critical": "danger",
            "high": "warning",
            "medium": "good",
            "low": "#36a64f",
            "info": "#36a64f"
        }
        
        context["color"] = severity_colors.get(severity, "#36a64f")
        
        # 添加时间戳
        context["timestamp"] = int(time.time())
        
        # 格式化标签
        labels = context.get("labels", {})
        if labels:
            labels_str = "\n".join([f"  {k}: {v}" for k, v in labels.items()])
            context["labels"] = labels_str
            context["labels_json"] = json.dumps(labels)
        else:
            context["labels"] = "无"
            context["labels_json"] = "{}"
        
        return context
    
    def _get_notification_priority(self, alert_data: Dict[str, Any]) -> NotificationPriority:
        """获取通知优先级"""
        severity = alert_data.get("severity", "info").lower()
        
        priority_mapping = {
            "critical": NotificationPriority.CRITICAL,
            "high": NotificationPriority.HIGH,
            "medium": NotificationPriority.MEDIUM,
            "low": NotificationPriority.LOW,
            "info": NotificationPriority.INFO
        }
        
        return priority_mapping.get(severity, NotificationPriority.MEDIUM)
    
    def _generate_message_id(self, rule: NotificationRule, alert_data: Dict[str, Any], 
                           channel: NotificationChannel) -> str:
        """生成消息ID"""
        content = f"{rule.id}:{alert_data.get('alert_id', 'unknown')}:{channel.value}:{time.time()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def _send_message(self, message: NotificationMessage) -> NotificationResult:
        """发送消息"""
        start_time = time.time()
        
        try:
            message.status = NotificationStatus.SENDING
            message.attempts += 1
            
            # 获取渠道配置
            channel_config = self.channel_configs.get(message.channel)
            if not channel_config or not channel_config.enabled:
                raise ValueError(f"渠道未配置或已禁用: {message.channel.value}")
            
            # 根据渠道类型发送消息
            if message.channel == NotificationChannel.EMAIL:
                await self._send_email(message, channel_config)
            elif message.channel == NotificationChannel.WEBHOOK:
                await self._send_webhook(message, channel_config)
            elif message.channel == NotificationChannel.DINGTALK:
                await self._send_dingtalk(message, channel_config)
            elif message.channel == NotificationChannel.SLACK:
                await self._send_slack(message, channel_config)
            elif message.channel == NotificationChannel.CONSOLE:
                await self._send_console(message, channel_config)
            elif message.channel == NotificationChannel.FILE:
                await self._send_file(message, channel_config)
            elif message.channel == NotificationChannel.CUSTOM:
                await self._send_custom(message, channel_config)
            else:
                raise ValueError(f"不支持的通知渠道: {message.channel.value}")
            
            # 发送成功
            message.status = NotificationStatus.SENT
            message.sent_at = datetime.now()
            
            self.sent_messages.append(message)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            result = NotificationResult(
                message_id=message.id,
                status=NotificationStatus.SENT,
                sent_at=message.sent_at,
                duration_ms=duration_ms
            )
            
            self.logger.info(f"通知发送成功: {message.id} via {message.channel.value}")
            
            return result
            
        except Exception as e:
            # 发送失败
            message.status = NotificationStatus.FAILED
            message.error_message = str(e)
            
            self.failed_messages.append(message)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            result = NotificationResult(
                message_id=message.id,
                status=NotificationStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms
            )
            
            self.logger.error(f"通知发送失败: {message.id} via {message.channel.value}: {e}")
            
            # 检查是否需要重试
            if message.should_retry():
                message.status = NotificationStatus.RETRYING
                # 计算重试延迟
                retry_delay = channel_config.retry_delay * (channel_config.retry_backoff ** (message.attempts - 1))
                message.scheduled_at = datetime.now() + timedelta(seconds=retry_delay)
                self.pending_messages.append(message)
                
                self.logger.info(f"消息将在 {retry_delay} 秒后重试: {message.id}")
            
            return result
    
    async def _send_email(self, message: NotificationMessage, config: ChannelConfig):
        """发送邮件"""
        smtp_config = config.extra_config
        
        # 创建邮件消息
        msg = MIMEMultipart()
        msg['From'] = f"{smtp_config.get('from_name', 'HarborAI')} <{smtp_config.get('from_address', 'alerts@company.com')}>"
        msg['Subject'] = message.subject
        
        # 添加正文
        if message.format_type == "html":
            msg.attach(MIMEText(message.body, 'html', 'utf-8'))
        else:
            msg.attach(MIMEText(message.body, 'plain', 'utf-8'))
        
        # 发送给每个接收者
        for recipient in message.recipients:
            if "@" not in recipient:
                continue
            
            msg['To'] = recipient
            
            # 连接SMTP服务器
            smtp_server = smtp_config.get('smtp_server', 'localhost')
            smtp_port = smtp_config.get('smtp_port', 587)
            use_tls = smtp_config.get('use_tls', True)
            
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=config.connect_timeout)
            
            try:
                if use_tls:
                    server.starttls(context=ssl.create_default_context())
                
                username = config.credentials.get('username')
                password = config.credentials.get('password')
                
                if username and password:
                    server.login(username, password)
                
                server.send_message(msg)
                
            finally:
                server.quit()
    
    async def _send_webhook(self, message: NotificationMessage, config: ChannelConfig):
        """发送Webhook"""
        for recipient in message.recipients:
            if not recipient.startswith('http'):
                continue
            
            # 准备请求数据
            if message.format_type == "json":
                try:
                    data = json.loads(message.body)
                except json.JSONDecodeError:
                    data = {"message": message.body}
            else:
                data = {
                    "subject": message.subject,
                    "message": message.body,
                    "priority": message.priority.value,
                    "timestamp": message.created_at.isoformat(),
                    "labels": message.labels,
                    "metadata": message.metadata
                }
            
            # 发送HTTP请求
            timeout = aiohttp.ClientTimeout(
                connect=config.connect_timeout,
                total=config.read_timeout
            )
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    recipient,
                    json=data,
                    headers=config.headers
                ) as response:
                    if response.status >= 400:
                        raise Exception(f"Webhook请求失败: {response.status} {await response.text()}")
    
    async def _send_dingtalk(self, message: NotificationMessage, config: ChannelConfig):
        """发送钉钉消息"""
        access_token = config.extra_config.get('access_token')
        if not access_token:
            raise ValueError("钉钉access_token未配置")
        
        url = f"{config.endpoint}?access_token={access_token}"
        
        # 准备钉钉消息格式
        if message.format_type == "markdown":
            data = {
                "msgtype": "markdown",
                "markdown": {
                    "title": message.subject,
                    "text": message.body
                }
            }
        else:
            data = {
                "msgtype": "text",
                "text": {
                    "content": f"{message.subject}\n\n{message.body}"
                }
            }
        
        # 添加@所有人或特定用户
        if message.priority == NotificationPriority.CRITICAL:
            data["at"] = {"isAtAll": True}
        
        # 发送请求
        timeout = aiohttp.ClientTimeout(
            connect=config.connect_timeout,
            total=config.read_timeout
        )
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                url,
                json=data,
                headers=config.headers
            ) as response:
                if response.status >= 400:
                    raise Exception(f"钉钉消息发送失败: {response.status} {await response.text()}")
    
    async def _send_slack(self, message: NotificationMessage, config: ChannelConfig):
        """发送Slack消息"""
        webhook_url = config.credentials.get('webhook_url')
        if not webhook_url:
            raise ValueError("Slack webhook_url未配置")
        
        # 准备Slack消息格式
        if message.format_type == "json":
            try:
                data = json.loads(message.body)
            except json.JSONDecodeError:
                data = {"text": f"{message.subject}\n{message.body}"}
        else:
            data = {"text": f"{message.subject}\n{message.body}"}
        
        # 发送请求
        timeout = aiohttp.ClientTimeout(
            connect=config.connect_timeout,
            total=config.read_timeout
        )
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                webhook_url,
                json=data,
                headers=config.headers
            ) as response:
                if response.status >= 400:
                    raise Exception(f"Slack消息发送失败: {response.status} {await response.text()}")
    
    async def _send_console(self, message: NotificationMessage, config: ChannelConfig):
        """发送控制台消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console_message = f"[{timestamp}] {message.subject}"
        
        if message.body and message.body != message.subject:
            console_message += f"\n{message.body}"
        
        print(console_message)
        self.logger.info(console_message)
    
    async def _send_file(self, message: NotificationMessage, config: ChannelConfig):
        """发送文件消息"""
        file_path = Path(config.extra_config.get('file_path', 'logs/alerts.log'))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message.subject}\n{message.body}\n{'-'*50}\n"
        
        # 检查文件大小
        max_size = config.extra_config.get('max_file_size', 10 * 1024 * 1024)
        if file_path.exists() and file_path.stat().st_size > max_size:
            # 轮转日志文件
            backup_count = config.extra_config.get('backup_count', 5)
            for i in range(backup_count - 1, 0, -1):
                old_file = file_path.with_suffix(f'.{i}')
                new_file = file_path.with_suffix(f'.{i + 1}')
                if old_file.exists():
                    old_file.rename(new_file)
            
            if file_path.exists():
                file_path.rename(file_path.with_suffix('.1'))
        
        # 写入日志
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    async def _send_custom(self, message: NotificationMessage, config: ChannelConfig):
        """发送自定义消息"""
        handler_name = config.extra_config.get('handler_name')
        if not handler_name or handler_name not in self.custom_handlers:
            raise ValueError(f"自定义处理器未找到: {handler_name}")
        
        handler = self.custom_handlers[handler_name]
        await handler(message, config)
    
    def register_custom_handler(self, name: str, handler: Callable):
        """注册自定义处理器"""
        self.custom_handlers[name] = handler
        self.logger.info(f"注册自定义通知处理器: {name}")
    
    async def process_pending_messages(self):
        """处理待发送消息"""
        current_time = datetime.now()
        ready_messages = []
        
        # 查找准备发送的消息
        while self.pending_messages:
            message = self.pending_messages[0]
            
            if message.scheduled_at and message.scheduled_at > current_time:
                break
            
            ready_messages.append(self.pending_messages.popleft())
        
        # 发送准备好的消息
        for message in ready_messages:
            try:
                await self._send_message(message)
            except Exception as e:
                self.logger.error(f"处理待发送消息失败: {message.id}: {e}")
        
        # 处理聚合消息
        await self._process_aggregated_messages()
    
    async def _process_aggregated_messages(self):
        """处理聚合消息"""
        for rule in self.rules.values():
            if rule.aggregation_strategy == AggregationStrategy.NONE:
                continue
            
            ready_groups = self.aggregator.get_ready_groups(rule)
            
            for aggregation_key, messages in ready_groups:
                try:
                    # 创建聚合消息
                    aggregated_message = self.aggregator.create_aggregated_message(
                        aggregation_key, messages
                    )
                    
                    # 发送聚合消息
                    await self._send_message(aggregated_message)
                    
                    self.logger.info(f"发送聚合消息: {aggregation_key} ({len(messages)} 个告警)")
                    
                except Exception as e:
                    self.logger.error(f"处理聚合消息失败: {aggregation_key}: {e}")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_sent = len(self.sent_messages)
        total_failed = len(self.failed_messages)
        total_pending = len(self.pending_messages)
        
        # 按渠道统计
        channel_stats = {}
        for channel in NotificationChannel:
            sent_count = len([m for m in self.sent_messages if m.channel == channel])
            failed_count = len([m for m in self.failed_messages if m.channel == channel])
            channel_stats[channel.value] = {
                "sent": sent_count,
                "failed": failed_count
            }
        
        # 按优先级统计
        priority_stats = {}
        for priority in NotificationPriority:
            sent_count = len([m for m in self.sent_messages if m.priority == priority])
            failed_count = len([m for m in self.failed_messages if m.priority == priority])
            priority_stats[priority.value] = {
                "sent": sent_count,
                "failed": failed_count
            }
        
        # 最近24小时统计
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_sent = len([
            m for m in self.sent_messages
            if m.sent_at and m.sent_at > last_24h
        ])
        
        recent_failed = len([
            m for m in self.failed_messages
            if m.created_at > last_24h
        ])
        
        return {
            "total_sent": total_sent,
            "total_failed": total_failed,
            "total_pending": total_pending,
            "success_rate": total_sent / (total_sent + total_failed) if (total_sent + total_failed) > 0 else 0,
            "channel_statistics": channel_stats,
            "priority_statistics": priority_stats,
            "sent_24h": recent_sent,
            "failed_24h": recent_failed,
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "total_templates": len(self.templates),
            "configured_channels": len(self.channel_configs),
            "aggregation_groups": len(self.aggregator.aggregation_groups),
            "rate_limiters": len(self.rate_limiters)
        }
    
    async def cleanup_old_messages(self, days: int = 7):
        """清理旧消息"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # 清理已发送消息
        recent_sent = deque([
            message for message in self.sent_messages
            if message.sent_at and message.sent_at > cutoff_time
        ], maxlen=10000)
        self.sent_messages = recent_sent
        
        # 清理失败消息
        recent_failed = deque([
            message for message in self.failed_messages
            if message.created_at > cutoff_time
        ], maxlen=1000)
        self.failed_messages = recent_failed
        
        # 清理过期的待发送消息
        current_pending = deque()
        while self.pending_messages:
            message = self.pending_messages.popleft()
            if not message.is_expired():
                current_pending.append(message)
        
        self.pending_messages = current_pending
        
        self.logger.info(f"清理了 {days} 天前的旧消息")