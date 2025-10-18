"""
告警通知配置管理器

负责管理告警通知的配置、路由、模板和渠道管理，
支持多渠道通知、智能路由、模板管理和通知策略配置。
"""

import json
import yaml
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
from collections import defaultdict, deque
import fnmatch
import hashlib


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
    AGGREGATED = "aggregated"


class EscalationLevel(Enum):
    """升级级别"""
    L1 = "l1"  # 一级支持
    L2 = "l2"  # 二级支持
    L3 = "l3"  # 三级支持
    MANAGER = "manager"  # 管理层
    EXECUTIVE = "executive"  # 高管层


class AggregationStrategy(Enum):
    """聚合策略"""
    COUNT = "count"           # 按数量聚合
    TIME = "time"             # 按时间聚合
    SEVERITY = "severity"     # 按严重级别聚合
    SERVICE = "service"       # 按服务聚合
    HOST = "host"             # 按主机聚合
    CUSTOM = "custom"         # 自定义聚合


@dataclass
class TimeWindow:
    """时间窗口"""
    start_time: str           # 开始时间 (HH:MM)
    end_time: str             # 结束时间 (HH:MM)
    weekdays: List[int] = field(default_factory=list)  # 工作日
    timezone: str = "UTC"     # 时区
    
    def is_in_window(self, timestamp: datetime) -> bool:
        """检查时间是否在窗口内"""
        if self.weekdays and timestamp.weekday() not in self.weekdays:
            return False
        
        current_time = timestamp.strftime("%H:%M")
        
        if self.start_time <= self.end_time:
            return self.start_time <= current_time <= self.end_time
        else:
            return current_time >= self.start_time or current_time <= self.end_time


@dataclass
class RateLimitConfig:
    """速率限制配置"""
    max_notifications: int    # 最大通知数
    time_window_seconds: int  # 时间窗口（秒）
    burst_limit: int = 0      # 突发限制
    
    def is_exceeded(self, count: int, window_start: datetime) -> bool:
        """检查是否超过速率限制"""
        if count >= self.max_notifications:
            window_age = (datetime.now() - window_start).total_seconds()
            return window_age < self.time_window_seconds
        return False


@dataclass
class EscalationRule:
    """升级规则"""
    trigger_condition: str    # 触发条件
    escalation_level: EscalationLevel  # 升级级别
    delay_minutes: int        # 延迟时间（分钟）
    max_escalations: int = 3  # 最大升级次数
    
    def should_escalate(self, alert_age_minutes: int, ack_status: bool) -> bool:
        """检查是否应该升级"""
        if ack_status:  # 已确认的告警不升级
            return False
        
        return alert_age_minutes >= self.delay_minutes


@dataclass
class NotificationTemplate:
    """通知模板"""
    id: str
    name: str
    description: str
    channel: NotificationChannel
    
    # 模板内容
    subject_template: str = ""
    body_template: str = ""
    
    # 模板变量
    variables: Dict[str, Any] = field(default_factory=dict)
    
    # 格式配置
    format_type: str = "text"  # text, html, markdown, json
    encoding: str = "utf-8"
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def render(self, context: Dict[str, Any]) -> Tuple[str, str]:
        """渲染模板"""
        try:
            # 合并变量和上下文
            render_context = {**self.variables, **context}
            
            # 渲染主题
            subject = self.subject_template
            for key, value in render_context.items():
                subject = subject.replace(f"{{{key}}}", str(value))
            
            # 渲染正文
            body = self.body_template
            for key, value in render_context.items():
                body = body.replace(f"{{{key}}}", str(value))
            
            return subject, body
            
        except Exception as e:
            return f"模板渲染错误: {e}", ""


@dataclass
class NotificationRule:
    """通知规则"""
    id: str
    name: str
    description: str
    
    # 匹配条件
    alert_patterns: List[str] = field(default_factory=list)  # 告警名称模式
    severity_levels: List[str] = field(default_factory=list)  # 严重级别
    service_patterns: List[str] = field(default_factory=list)  # 服务模式
    label_conditions: Dict[str, List[str]] = field(default_factory=dict)  # 标签条件
    
    # 通知配置
    channels: List[NotificationChannel] = field(default_factory=list)
    template_ids: Dict[NotificationChannel, str] = field(default_factory=dict)
    recipients: Dict[NotificationChannel, List[str]] = field(default_factory=dict)
    
    # 时间配置
    active_hours: Optional[TimeWindow] = None
    quiet_hours: Optional[TimeWindow] = None
    
    # 速率限制
    rate_limit: Optional[RateLimitConfig] = None
    
    # 升级配置
    escalation_rules: List[EscalationRule] = field(default_factory=list)
    
    # 聚合配置
    aggregation_strategy: Optional[AggregationStrategy] = None
    aggregation_window_seconds: int = 300
    aggregation_threshold: int = 5
    
    # 元数据
    priority: NotificationPriority = NotificationPriority.MEDIUM
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def matches_alert(self, alert_name: str, alert_severity: str,
                     alert_service: str, alert_labels: Dict[str, str]) -> bool:
        """检查规则是否匹配告警"""
        if not self.enabled:
            return False
        
        # 检查告警名称模式
        if self.alert_patterns:
            if not any(fnmatch.fnmatch(alert_name, pattern) for pattern in self.alert_patterns):
                return False
        
        # 检查严重级别
        if self.severity_levels and alert_severity not in self.severity_levels:
            return False
        
        # 检查服务模式
        if self.service_patterns:
            if not any(fnmatch.fnmatch(alert_service, pattern) for pattern in self.service_patterns):
                return False
        
        # 检查标签条件
        for label_key, label_values in self.label_conditions.items():
            alert_label_value = alert_labels.get(label_key, "")
            if alert_label_value not in label_values:
                return False
        
        return True
    
    def is_in_active_hours(self, timestamp: datetime) -> bool:
        """检查是否在活跃时间内"""
        if self.active_hours:
            return self.active_hours.is_in_window(timestamp)
        return True
    
    def is_in_quiet_hours(self, timestamp: datetime) -> bool:
        """检查是否在静默时间内"""
        if self.quiet_hours:
            return self.quiet_hours.is_in_window(timestamp)
        return False


@dataclass
class ChannelConfig:
    """渠道配置"""
    channel: NotificationChannel
    enabled: bool = True
    
    # 连接配置
    endpoint: str = ""
    api_key: str = ""
    username: str = ""
    password: str = ""
    
    # 特定配置
    config: Dict[str, Any] = field(default_factory=dict)
    
    # 重试配置
    max_retries: int = 3
    retry_delay_seconds: int = 60
    retry_backoff_factor: float = 2.0
    
    # 超时配置
    timeout_seconds: int = 30
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class NotificationMessage:
    """通知消息"""
    id: str
    rule_id: str
    template_id: str
    channel: NotificationChannel
    
    # 消息内容
    subject: str
    body: str
    recipients: List[str]
    
    # 告警信息
    alert_name: str
    alert_severity: str
    alert_labels: Dict[str, str]
    
    # 状态信息
    status: NotificationStatus = NotificationStatus.PENDING
    priority: NotificationPriority = NotificationPriority.MEDIUM
    
    # 时间信息
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    
    # 重试信息
    retry_count: int = 0
    max_retries: int = 3
    last_error: str = ""
    
    # 聚合信息
    aggregation_key: str = ""
    aggregated_count: int = 1
    
    def age_seconds(self) -> float:
        """获取消息年龄（秒）"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def should_retry(self) -> bool:
        """检查是否应该重试"""
        return (self.status == NotificationStatus.FAILED and 
                self.retry_count < self.max_retries)
    
    def calculate_next_retry_delay(self) -> int:
        """计算下次重试延迟"""
        base_delay = 60  # 基础延迟60秒
        return int(base_delay * (2 ** self.retry_count))


@dataclass
class NotificationResult:
    """通知结果"""
    message_id: str
    channel: NotificationChannel
    status: NotificationStatus
    sent_at: datetime
    response_data: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    duration_ms: int = 0


class AlertNotificationConfigManager:
    """告警通知配置管理器"""
    
    def __init__(self, config_dir: str = "config/alerts"):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置存储
        self.templates: Dict[str, NotificationTemplate] = {}
        self.rules: Dict[str, NotificationRule] = {}
        self.channels: Dict[NotificationChannel, ChannelConfig] = {}
        
        # 运行时状态
        self.pending_messages: deque = deque()
        self.sent_messages: deque = deque(maxlen=10000)
        self.rate_limit_counters: Dict[str, int] = defaultdict(int)
        self.rate_limit_windows: Dict[str, datetime] = {}
        self.aggregation_groups: Dict[str, List[NotificationMessage]] = defaultdict(list)
        
        # 配置文件路径
        self.templates_file = self.config_dir / "notification_templates.json"
        self.rules_file = self.config_dir / "notification_rules.json"
        self.channels_file = self.config_dir / "notification_channels.json"
        
        # 加载配置
        self._load_default_templates()
        self._load_default_rules()
        self._load_default_channels()
        self._load_templates()
        self._load_rules()
        self._load_channels()
    
    def _load_default_templates(self):
        """加载默认通知模板"""
        default_templates = [
            # 邮件模板
            NotificationTemplate(
                id="email_critical_alert",
                name="邮件关键告警模板",
                description="用于发送关键告警的邮件模板",
                channel=NotificationChannel.EMAIL,
                subject_template="🚨 关键告警: {alert_name}",
                body_template="""
告警详情:
- 告警名称: {alert_name}
- 严重级别: {alert_severity}
- 服务: {service}
- 主机: {host}
- 时间: {timestamp}
- 描述: {description}

标签信息:
{labels}

请立即处理此告警。

---
HarborAI 监控系统
                """.strip(),
                format_type="text",
                tags=["email", "critical"]
            ),
            
            NotificationTemplate(
                id="email_general_alert",
                name="邮件通用告警模板",
                description="用于发送一般告警的邮件模板",
                channel=NotificationChannel.EMAIL,
                subject_template="⚠️ 告警通知: {alert_name}",
                body_template="""
告警详情:
- 告警名称: {alert_name}
- 严重级别: {alert_severity}
- 服务: {service}
- 主机: {host}
- 时间: {timestamp}
- 描述: {description}

标签信息:
{labels}

---
HarborAI 监控系统
                """.strip(),
                format_type="text",
                tags=["email", "general"]
            ),
            
            # Webhook模板
            NotificationTemplate(
                id="webhook_alert",
                name="Webhook告警模板",
                description="用于发送告警到Webhook的JSON模板",
                channel=NotificationChannel.WEBHOOK,
                subject_template="",
                body_template="""{
    "alert_name": "{alert_name}",
    "severity": "{alert_severity}",
    "service": "{service}",
    "host": "{host}",
    "timestamp": "{timestamp}",
    "description": "{description}",
    "labels": {labels_json},
    "source": "HarborAI"
}""",
                format_type="json",
                tags=["webhook", "json"]
            ),
            
            # 钉钉模板
            NotificationTemplate(
                id="dingtalk_alert",
                name="钉钉告警模板",
                description="用于发送告警到钉钉的模板",
                channel=NotificationChannel.DINGTALK,
                subject_template="",
                body_template="""## {severity_emoji} {alert_name}

**告警详情:**
- **严重级别:** {alert_severity}
- **服务:** {service}
- **主机:** {host}
- **时间:** {timestamp}

**描述:** {description}

**标签:** {labels}

> 来自 HarborAI 监控系统""",
                format_type="markdown",
                variables={
                    "severity_emoji": "🚨"
                },
                tags=["dingtalk", "markdown"]
            ),
            
            # Slack模板
            NotificationTemplate(
                id="slack_alert",
                name="Slack告警模板",
                description="用于发送告警到Slack的模板",
                channel=NotificationChannel.SLACK,
                subject_template="",
                body_template="""{
    "text": "{severity_emoji} {alert_name}",
    "attachments": [
        {
            "color": "{color}",
            "fields": [
                {
                    "title": "严重级别",
                    "value": "{alert_severity}",
                    "short": true
                },
                {
                    "title": "服务",
                    "value": "{service}",
                    "short": true
                },
                {
                    "title": "主机",
                    "value": "{host}",
                    "short": true
                },
                {
                    "title": "时间",
                    "value": "{timestamp}",
                    "short": true
                },
                {
                    "title": "描述",
                    "value": "{description}",
                    "short": false
                }
            ],
            "footer": "HarborAI 监控系统"
        }
    ]
}""",
                format_type="json",
                variables={
                    "severity_emoji": "🚨",
                    "color": "danger"
                },
                tags=["slack", "json"]
            ),
            
            # 控制台模板
            NotificationTemplate(
                id="console_alert",
                name="控制台告警模板",
                description="用于在控制台显示告警的模板",
                channel=NotificationChannel.CONSOLE,
                subject_template="[{alert_severity}] {alert_name}",
                body_template="""[{timestamp}] {severity_emoji} {alert_name}
服务: {service} | 主机: {host} | 级别: {alert_severity}
描述: {description}
标签: {labels}""",
                format_type="text",
                variables={
                    "severity_emoji": "⚠️"
                },
                tags=["console", "text"]
            ),
            
            # 文件模板
            NotificationTemplate(
                id="file_alert",
                name="文件告警模板",
                description="用于写入文件的告警模板",
                channel=NotificationChannel.FILE,
                subject_template="",
                body_template="""{timestamp} | {alert_severity} | {alert_name} | {service} | {host} | {description}""",
                format_type="text",
                tags=["file", "log"]
            ),
            
            # 聚合模板
            NotificationTemplate(
                id="email_aggregated_alert",
                name="邮件聚合告警模板",
                description="用于发送聚合告警的邮件模板",
                channel=NotificationChannel.EMAIL,
                subject_template="📊 聚合告警报告 ({aggregated_count} 个告警)",
                body_template="""
聚合告警报告:

总计: {aggregated_count} 个告警
时间范围: {start_time} - {end_time}

告警列表:
{alert_list}

---
HarborAI 监控系统
                """.strip(),
                format_type="text",
                tags=["email", "aggregated"]
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
                description="立即通知所有关键告警",
                alert_patterns=["*"],
                severity_levels=["critical"],
                channels=[NotificationChannel.EMAIL, NotificationChannel.DINGTALK, NotificationChannel.SLACK],
                template_ids={
                    NotificationChannel.EMAIL: "email_critical_alert",
                    NotificationChannel.DINGTALK: "dingtalk_alert",
                    NotificationChannel.SLACK: "slack_alert"
                },
                recipients={
                    NotificationChannel.EMAIL: ["admin@example.com", "ops@example.com"],
                    NotificationChannel.DINGTALK: ["dingtalk_webhook_url"],
                    NotificationChannel.SLACK: ["#alerts"]
                },
                priority=NotificationPriority.CRITICAL,
                escalation_rules=[
                    EscalationRule(
                        trigger_condition="no_ack",
                        escalation_level=EscalationLevel.L2,
                        delay_minutes=15
                    ),
                    EscalationRule(
                        trigger_condition="no_ack",
                        escalation_level=EscalationLevel.MANAGER,
                        delay_minutes=30
                    )
                ],
                tags=["critical", "immediate"]
            ),
            
            # 高级告警规则
            NotificationRule(
                id="high_alerts",
                name="高级告警通知",
                description="通知高级告警",
                alert_patterns=["*"],
                severity_levels=["high"],
                channels=[NotificationChannel.EMAIL, NotificationChannel.DINGTALK],
                template_ids={
                    NotificationChannel.EMAIL: "email_general_alert",
                    NotificationChannel.DINGTALK: "dingtalk_alert"
                },
                recipients={
                    NotificationChannel.EMAIL: ["ops@example.com"],
                    NotificationChannel.DINGTALK: ["dingtalk_webhook_url"]
                },
                priority=NotificationPriority.HIGH,
                rate_limit=RateLimitConfig(
                    max_notifications=10,
                    time_window_seconds=3600  # 1小时内最多10条
                ),
                aggregation_strategy=AggregationStrategy.COUNT,
                aggregation_threshold=5,
                aggregation_window_seconds=600,  # 10分钟聚合窗口
                tags=["high", "rate_limited"]
            ),
            
            # 中级告警规则
            NotificationRule(
                id="medium_alerts",
                name="中级告警通知",
                description="通知中级告警",
                alert_patterns=["*"],
                severity_levels=["medium", "warning"],
                channels=[NotificationChannel.EMAIL],
                template_ids={
                    NotificationChannel.EMAIL: "email_general_alert"
                },
                recipients={
                    NotificationChannel.EMAIL: ["ops@example.com"]
                },
                priority=NotificationPriority.MEDIUM,
                rate_limit=RateLimitConfig(
                    max_notifications=5,
                    time_window_seconds=3600  # 1小时内最多5条
                ),
                aggregation_strategy=AggregationStrategy.SERVICE,
                aggregation_threshold=3,
                aggregation_window_seconds=1800,  # 30分钟聚合窗口
                quiet_hours=TimeWindow(
                    start_time="22:00",
                    end_time="08:00",
                    weekdays=[0, 1, 2, 3, 4, 5, 6]  # 每天
                ),
                tags=["medium", "aggregated", "quiet_hours"]
            ),
            
            # 数据库告警规则
            NotificationRule(
                id="database_alerts",
                name="数据库告警通知",
                description="专门处理数据库相关告警",
                alert_patterns=["database_*", "db_*", "*_database_*"],
                severity_levels=["critical", "high", "medium"],
                service_patterns=["*database*", "*db*", "*mysql*", "*postgres*"],
                channels=[NotificationChannel.EMAIL, NotificationChannel.WEBHOOK],
                template_ids={
                    NotificationChannel.EMAIL: "email_critical_alert",
                    NotificationChannel.WEBHOOK: "webhook_alert"
                },
                recipients={
                    NotificationChannel.EMAIL: ["dba@example.com", "ops@example.com"],
                    NotificationChannel.WEBHOOK: ["http://dba-system.example.com/webhook"]
                },
                priority=NotificationPriority.HIGH,
                tags=["database", "dba"]
            ),
            
            # 网络告警规则
            NotificationRule(
                id="network_alerts",
                name="网络告警通知",
                description="专门处理网络相关告警",
                alert_patterns=["network_*", "connectivity_*", "*_network_*"],
                severity_levels=["critical", "high"],
                channels=[NotificationChannel.EMAIL, NotificationChannel.DINGTALK],
                template_ids={
                    NotificationChannel.EMAIL: "email_critical_alert",
                    NotificationChannel.DINGTALK: "dingtalk_alert"
                },
                recipients={
                    NotificationChannel.EMAIL: ["network@example.com", "ops@example.com"],
                    NotificationChannel.DINGTALK: ["network_dingtalk_webhook"]
                },
                priority=NotificationPriority.HIGH,
                tags=["network", "infrastructure"]
            ),
            
            # 应用告警规则
            NotificationRule(
                id="application_alerts",
                name="应用告警通知",
                description="处理应用层告警",
                alert_patterns=["app_*", "application_*", "service_*"],
                severity_levels=["critical", "high", "medium"],
                channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
                template_ids={
                    NotificationChannel.EMAIL: "email_general_alert",
                    NotificationChannel.SLACK: "slack_alert"
                },
                recipients={
                    NotificationChannel.EMAIL: ["dev@example.com"],
                    NotificationChannel.SLACK: ["#dev-alerts"]
                },
                priority=NotificationPriority.MEDIUM,
                aggregation_strategy=AggregationStrategy.SERVICE,
                aggregation_threshold=3,
                aggregation_window_seconds=900,  # 15分钟聚合窗口
                tags=["application", "development"]
            ),
            
            # 测试环境告警规则
            NotificationRule(
                id="test_environment_alerts",
                name="测试环境告警通知",
                description="处理测试环境告警",
                alert_patterns=["*"],
                severity_levels=["critical", "high"],
                label_conditions={
                    "environment": ["test", "staging", "dev"]
                },
                channels=[NotificationChannel.SLACK],
                template_ids={
                    NotificationChannel.SLACK: "slack_alert"
                },
                recipients={
                    NotificationChannel.SLACK: ["#test-alerts"]
                },
                priority=NotificationPriority.LOW,
                rate_limit=RateLimitConfig(
                    max_notifications=20,
                    time_window_seconds=3600
                ),
                tags=["test", "development"]
            ),
            
            # 控制台日志规则
            NotificationRule(
                id="console_logging",
                name="控制台日志记录",
                description="将所有告警记录到控制台",
                alert_patterns=["*"],
                severity_levels=["critical", "high", "medium", "low", "info"],
                channels=[NotificationChannel.CONSOLE],
                template_ids={
                    NotificationChannel.CONSOLE: "console_alert"
                },
                recipients={
                    NotificationChannel.CONSOLE: ["console"]
                },
                priority=NotificationPriority.INFO,
                tags=["console", "logging"]
            ),
            
            # 文件日志规则
            NotificationRule(
                id="file_logging",
                name="文件日志记录",
                description="将所有告警记录到文件",
                alert_patterns=["*"],
                severity_levels=["critical", "high", "medium", "low", "info"],
                channels=[NotificationChannel.FILE],
                template_ids={
                    NotificationChannel.FILE: "file_alert"
                },
                recipients={
                    NotificationChannel.FILE: ["/var/log/alerts/alerts.log"]
                },
                priority=NotificationPriority.INFO,
                tags=["file", "logging"]
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
    
    def _load_default_channels(self):
        """加载默认通知渠道"""
        default_channels = [
            ChannelConfig(
                channel=NotificationChannel.EMAIL,
                enabled=True,
                endpoint="smtp://localhost:587",
                config={
                    "smtp_server": "localhost",
                    "smtp_port": 587,
                    "use_tls": True,
                    "from_address": "alerts@harborai.com",
                    "from_name": "HarborAI Alerts"
                },
                timeout_seconds=30,
                tags=["email", "smtp"]
            ),
            
            ChannelConfig(
                channel=NotificationChannel.WEBHOOK,
                enabled=True,
                config={
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json",
                        "User-Agent": "HarborAI-Alerts/1.0"
                    }
                },
                timeout_seconds=15,
                tags=["webhook", "http"]
            ),
            
            ChannelConfig(
                channel=NotificationChannel.DINGTALK,
                enabled=True,
                config={
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json"
                    }
                },
                timeout_seconds=10,
                tags=["dingtalk", "im"]
            ),
            
            ChannelConfig(
                channel=NotificationChannel.SLACK,
                enabled=True,
                config={
                    "method": "POST",
                    "headers": {
                        "Content-Type": "application/json"
                    }
                },
                timeout_seconds=10,
                tags=["slack", "im"]
            ),
            
            ChannelConfig(
                channel=NotificationChannel.CONSOLE,
                enabled=True,
                config={
                    "log_level": "INFO"
                },
                tags=["console", "logging"]
            ),
            
            ChannelConfig(
                channel=NotificationChannel.FILE,
                enabled=True,
                config={
                    "file_mode": "a",
                    "encoding": "utf-8",
                    "max_file_size": 100 * 1024 * 1024,  # 100MB
                    "backup_count": 5
                },
                tags=["file", "logging"]
            )
        ]
        
        for channel_config in default_channels:
            self.channels[channel_config.channel] = channel_config
    
    def _load_templates(self):
        """加载通知模板"""
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
    
    def _load_rules(self):
        """加载通知规则"""
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
    
    def _load_channels(self):
        """加载通知渠道"""
        if self.channels_file.exists():
            try:
                with open(self.channels_file, 'r', encoding='utf-8') as f:
                    channels_data = json.load(f)
                
                for channel_data in channels_data:
                    channel_config = self._dict_to_channel_config(channel_data)
                    if channel_config:
                        self.channels[channel_config.channel] = channel_config
                
                self.logger.info(f"加载了 {len(self.channels)} 个通知渠道")
                
            except Exception as e:
                self.logger.error(f"加载通知渠道失败: {e}")
    
    def _dict_to_template(self, template_data: Dict[str, Any]) -> Optional[NotificationTemplate]:
        """将字典转换为模板对象"""
        try:
            template_data["channel"] = NotificationChannel(template_data["channel"])
            
            if "created_at" in template_data:
                template_data["created_at"] = datetime.fromisoformat(template_data["created_at"])
            if "updated_at" in template_data:
                template_data["updated_at"] = datetime.fromisoformat(template_data["updated_at"])
            
            return NotificationTemplate(**template_data)
            
        except Exception as e:
            self.logger.error(f"转换通知模板失败: {e}")
            return None
    
    def _dict_to_rule(self, rule_data: Dict[str, Any]) -> Optional[NotificationRule]:
        """将字典转换为规则对象"""
        try:
            # 转换枚举类型
            if "channels" in rule_data:
                rule_data["channels"] = [NotificationChannel(ch) for ch in rule_data["channels"]]
            
            if "template_ids" in rule_data:
                template_ids = {}
                for ch, template_id in rule_data["template_ids"].items():
                    template_ids[NotificationChannel(ch)] = template_id
                rule_data["template_ids"] = template_ids
            
            if "recipients" in rule_data:
                recipients = {}
                for ch, recipient_list in rule_data["recipients"].items():
                    recipients[NotificationChannel(ch)] = recipient_list
                rule_data["recipients"] = recipients
            
            if "priority" in rule_data:
                rule_data["priority"] = NotificationPriority(rule_data["priority"])
            
            if "aggregation_strategy" in rule_data and rule_data["aggregation_strategy"]:
                rule_data["aggregation_strategy"] = AggregationStrategy(rule_data["aggregation_strategy"])
            
            # 转换时间窗口
            if "active_hours" in rule_data and rule_data["active_hours"]:
                rule_data["active_hours"] = TimeWindow(**rule_data["active_hours"])
            
            if "quiet_hours" in rule_data and rule_data["quiet_hours"]:
                rule_data["quiet_hours"] = TimeWindow(**rule_data["quiet_hours"])
            
            # 转换速率限制
            if "rate_limit" in rule_data and rule_data["rate_limit"]:
                rule_data["rate_limit"] = RateLimitConfig(**rule_data["rate_limit"])
            
            # 转换升级规则
            if "escalation_rules" in rule_data:
                escalation_rules = []
                for escalation_data in rule_data["escalation_rules"]:
                    escalation_data["escalation_level"] = EscalationLevel(escalation_data["escalation_level"])
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
    
    def _dict_to_channel_config(self, channel_data: Dict[str, Any]) -> Optional[ChannelConfig]:
        """将字典转换为渠道配置对象"""
        try:
            channel_data["channel"] = NotificationChannel(channel_data["channel"])
            
            if "created_at" in channel_data:
                channel_data["created_at"] = datetime.fromisoformat(channel_data["created_at"])
            if "updated_at" in channel_data:
                channel_data["updated_at"] = datetime.fromisoformat(channel_data["updated_at"])
            
            return ChannelConfig(**channel_data)
            
        except Exception as e:
            self.logger.error(f"转换渠道配置失败: {e}")
            return None
    
    def _template_to_dict(self, template: NotificationTemplate) -> Dict[str, Any]:
        """将模板对象转换为字典"""
        template_dict = asdict(template)
        template_dict["channel"] = template.channel.value
        
        if template.created_at:
            template_dict["created_at"] = template.created_at.isoformat()
        if template.updated_at:
            template_dict["updated_at"] = template.updated_at.isoformat()
        
        return template_dict
    
    def _rule_to_dict(self, rule: NotificationRule) -> Dict[str, Any]:
        """将规则对象转换为字典"""
        rule_dict = asdict(rule)
        
        # 转换枚举为字符串
        if rule.channels:
            rule_dict["channels"] = [ch.value for ch in rule.channels]
        
        if rule.template_ids:
            template_ids = {}
            for ch, template_id in rule.template_ids.items():
                template_ids[ch.value] = template_id
            rule_dict["template_ids"] = template_ids
        
        if rule.recipients:
            recipients = {}
            for ch, recipient_list in rule.recipients.items():
                recipients[ch.value] = recipient_list
            rule_dict["recipients"] = recipients
        
        rule_dict["priority"] = rule.priority.value
        
        if rule.aggregation_strategy:
            rule_dict["aggregation_strategy"] = rule.aggregation_strategy.value
        
        # 转换升级规则
        if rule.escalation_rules:
            escalation_rules = []
            for escalation_rule in rule.escalation_rules:
                escalation_dict = asdict(escalation_rule)
                escalation_dict["escalation_level"] = escalation_rule.escalation_level.value
                escalation_rules.append(escalation_dict)
            rule_dict["escalation_rules"] = escalation_rules
        
        # 转换日期时间为字符串
        if rule.created_at:
            rule_dict["created_at"] = rule.created_at.isoformat()
        if rule.updated_at:
            rule_dict["updated_at"] = rule.updated_at.isoformat()
        
        return rule_dict
    
    def _channel_config_to_dict(self, channel_config: ChannelConfig) -> Dict[str, Any]:
        """将渠道配置对象转换为字典"""
        config_dict = asdict(channel_config)
        config_dict["channel"] = channel_config.channel.value
        
        if channel_config.created_at:
            config_dict["created_at"] = channel_config.created_at.isoformat()
        if channel_config.updated_at:
            config_dict["updated_at"] = channel_config.updated_at.isoformat()
        
        return config_dict
    
    async def save_templates(self) -> bool:
        """保存通知模板"""
        try:
            templates_data = [self._template_to_dict(template) for template in self.templates.values()]
            
            with open(self.templates_file, 'w', encoding='utf-8') as f:
                json.dump(templates_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"保存了 {len(self.templates)} 个通知模板")
            return True
            
        except Exception as e:
            self.logger.error(f"保存通知模板失败: {e}")
            return False
    
    async def save_rules(self) -> bool:
        """保存通知规则"""
        try:
            rules_data = [self._rule_to_dict(rule) for rule in self.rules.values()]
            
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"保存了 {len(self.rules)} 个通知规则")
            return True
            
        except Exception as e:
            self.logger.error(f"保存通知规则失败: {e}")
            return False
    
    async def save_channels(self) -> bool:
        """保存通知渠道"""
        try:
            channels_data = [self._channel_config_to_dict(config) for config in self.channels.values()]
            
            with open(self.channels_file, 'w', encoding='utf-8') as f:
                json.dump(channels_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"保存了 {len(self.channels)} 个通知渠道")
            return True
            
        except Exception as e:
            self.logger.error(f"保存通知渠道失败: {e}")
            return False
    
    async def add_template(self, template: NotificationTemplate) -> bool:
        """添加通知模板"""
        if template.id in self.templates:
            self.logger.warning(f"通知模板已存在: {template.id}")
            return False
        
        self.templates[template.id] = template
        await self.save_templates()
        
        self.logger.info(f"添加通知模板: {template.id}")
        return True
    
    async def update_template(self, template_id: str, template: NotificationTemplate) -> bool:
        """更新通知模板"""
        if template_id not in self.templates:
            self.logger.warning(f"通知模板不存在: {template_id}")
            return False
        
        template.updated_at = datetime.now()
        self.templates[template_id] = template
        await self.save_templates()
        
        self.logger.info(f"更新通知模板: {template_id}")
        return True
    
    async def remove_template(self, template_id: str) -> bool:
        """删除通知模板"""
        if template_id not in self.templates:
            self.logger.warning(f"通知模板不存在: {template_id}")
            return False
        
        del self.templates[template_id]
        await self.save_templates()
        
        self.logger.info(f"删除通知模板: {template_id}")
        return True
    
    async def add_rule(self, rule: NotificationRule) -> bool:
        """添加通知规则"""
        if rule.id in self.rules:
            self.logger.warning(f"通知规则已存在: {rule.id}")
            return False
        
        self.rules[rule.id] = rule
        await self.save_rules()
        
        self.logger.info(f"添加通知规则: {rule.id}")
        return True
    
    async def update_rule(self, rule_id: str, rule: NotificationRule) -> bool:
        """更新通知规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"通知规则不存在: {rule_id}")
            return False
        
        rule.updated_at = datetime.now()
        self.rules[rule_id] = rule
        await self.save_rules()
        
        self.logger.info(f"更新通知规则: {rule_id}")
        return True
    
    async def remove_rule(self, rule_id: str) -> bool:
        """删除通知规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"通知规则不存在: {rule_id}")
            return False
        
        del self.rules[rule_id]
        await self.save_rules()
        
        self.logger.info(f"删除通知规则: {rule_id}")
        return True
    
    async def configure_channel(self, channel: NotificationChannel, config: ChannelConfig) -> bool:
        """配置通知渠道"""
        config.updated_at = datetime.now()
        self.channels[channel] = config
        await self.save_channels()
        
        self.logger.info(f"配置通知渠道: {channel.value}")
        return True
    
    def get_template(self, template_id: str) -> Optional[NotificationTemplate]:
        """获取通知模板"""
        return self.templates.get(template_id)
    
    def get_templates(self, channel: Optional[NotificationChannel] = None) -> List[NotificationTemplate]:
        """获取通知模板列表"""
        templates = list(self.templates.values())
        
        if channel:
            templates = [t for t in templates if t.channel == channel]
        
        return sorted(templates, key=lambda t: t.created_at)
    
    def get_rule(self, rule_id: str) -> Optional[NotificationRule]:
        """获取通知规则"""
        return self.rules.get(rule_id)
    
    def get_rules(self, enabled_only: bool = True) -> List[NotificationRule]:
        """获取通知规则列表"""
        rules = list(self.rules.values())
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        # 按优先级排序
        priority_order = {
            NotificationPriority.CRITICAL: 0,
            NotificationPriority.HIGH: 1,
            NotificationPriority.MEDIUM: 2,
            NotificationPriority.LOW: 3,
            NotificationPriority.INFO: 4
        }
        
        rules.sort(key=lambda r: priority_order.get(r.priority, 5))
        
        return rules
    
    def get_channel_config(self, channel: NotificationChannel) -> Optional[ChannelConfig]:
        """获取渠道配置"""
        return self.channels.get(channel)
    
    def get_matching_rules(self, alert_name: str, alert_severity: str,
                          alert_service: str, alert_labels: Dict[str, str]) -> List[NotificationRule]:
        """获取匹配的通知规则"""
        matching_rules = []
        
        for rule in self.get_rules(enabled_only=True):
            if rule.matches_alert(alert_name, alert_severity, alert_service, alert_labels):
                matching_rules.append(rule)
        
        return matching_rules
    
    def create_notification_message(self, rule: NotificationRule, template: NotificationTemplate,
                                  alert_name: str, alert_severity: str, alert_service: str,
                                  alert_labels: Dict[str, str], recipients: List[str]) -> NotificationMessage:
        """创建通知消息"""
        # 准备模板上下文
        context = {
            "alert_name": alert_name,
            "alert_severity": alert_severity,
            "service": alert_service,
            "host": alert_labels.get("host", "unknown"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": alert_labels.get("description", ""),
            "labels": ", ".join([f"{k}={v}" for k, v in alert_labels.items()]),
            "labels_json": json.dumps(alert_labels)
        }
        
        # 添加严重级别表情符号
        severity_emojis = {
            "critical": "🚨",
            "high": "⚠️",
            "medium": "⚡",
            "low": "ℹ️",
            "info": "📝"
        }
        context["severity_emoji"] = severity_emojis.get(alert_severity, "⚠️")
        
        # 添加颜色配置
        severity_colors = {
            "critical": "danger",
            "high": "warning",
            "medium": "good",
            "low": "good",
            "info": "good"
        }
        context["color"] = severity_colors.get(alert_severity, "warning")
        
        # 渲染模板
        subject, body = template.render(context)
        
        # 生成消息ID
        message_id = hashlib.md5(
            f"{rule.id}:{template.id}:{alert_name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()
        
        # 创建通知消息
        message = NotificationMessage(
            id=message_id,
            rule_id=rule.id,
            template_id=template.id,
            channel=template.channel,
            subject=subject,
            body=body,
            recipients=recipients,
            alert_name=alert_name,
            alert_severity=alert_severity,
            alert_labels=alert_labels,
            priority=rule.priority
        )
        
        return message
    
    async def get_notification_statistics(self) -> Dict[str, Any]:
        """获取通知统计信息"""
        total_templates = len(self.templates)
        total_rules = len(self.rules)
        enabled_rules = len([r for r in self.rules.values() if r.enabled])
        total_channels = len(self.channels)
        enabled_channels = len([c for c in self.channels.values() if c.enabled])
        
        # 按渠道统计模板
        template_by_channel = {}
        for channel in NotificationChannel:
            count = len([t for t in self.templates.values() if t.channel == channel])
            template_by_channel[channel.value] = count
        
        # 按优先级统计规则
        rule_by_priority = {}
        for priority in NotificationPriority:
            count = len([r for r in self.rules.values() if r.priority == priority])
            rule_by_priority[priority.value] = count
        
        # 消息统计
        pending_messages = len(self.pending_messages)
        sent_messages = len(self.sent_messages)
        
        # 最近24小时统计
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_sent = len([
            m for m in self.sent_messages
            if hasattr(m, 'sent_at') and m.sent_at and m.sent_at > last_24h
        ])
        
        return {
            "total_templates": total_templates,
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "total_channels": total_channels,
            "enabled_channels": enabled_channels,
            "template_distribution": template_by_channel,
            "rule_priority_distribution": rule_by_priority,
            "pending_messages": pending_messages,
            "sent_messages_total": sent_messages,
            "sent_messages_24h": recent_sent,
            "rate_limit_counters": len(self.rate_limit_counters),
            "aggregation_groups": len(self.aggregation_groups)
        }
    
    async def export_config(self, export_path: str) -> bool:
        """导出配置"""
        try:
            export_data = {
                "templates": [self._template_to_dict(template) for template in self.templates.values()],
                "rules": [self._rule_to_dict(rule) for rule in self.rules.values()],
                "channels": [self._channel_config_to_dict(config) for config in self.channels.values()],
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功导出通知配置到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出通知配置失败: {e}")
            return False
    
    async def import_config(self, import_path: str, overwrite: bool = False) -> bool:
        """导入配置"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 导入模板
            templates_data = import_data.get("templates", [])
            imported_templates = 0
            
            for template_data in templates_data:
                template = self._dict_to_template(template_data)
                if not template:
                    continue
                
                if template.id in self.templates and not overwrite:
                    self.logger.warning(f"通知模板已存在，跳过: {template.id}")
                    continue
                
                self.templates[template.id] = template
                imported_templates += 1
            
            # 导入规则
            rules_data = import_data.get("rules", [])
            imported_rules = 0
            
            for rule_data in rules_data:
                rule = self._dict_to_rule(rule_data)
                if not rule:
                    continue
                
                if rule.id in self.rules and not overwrite:
                    self.logger.warning(f"通知规则已存在，跳过: {rule.id}")
                    continue
                
                self.rules[rule.id] = rule
                imported_rules += 1
            
            # 导入渠道配置
            channels_data = import_data.get("channels", [])
            imported_channels = 0
            
            for channel_data in channels_data:
                channel_config = self._dict_to_channel_config(channel_data)
                if not channel_config:
                    continue
                
                if channel_config.channel in self.channels and not overwrite:
                    self.logger.warning(f"通知渠道已存在，跳过: {channel_config.channel.value}")
                    continue
                
                self.channels[channel_config.channel] = channel_config
                imported_channels += 1
            
            await self.save_templates()
            await self.save_rules()
            await self.save_channels()
            
            self.logger.info(f"成功导入 {imported_templates} 个模板, "
                           f"{imported_rules} 个规则, "
                           f"{imported_channels} 个渠道配置")
            return True
            
        except Exception as e:
            self.logger.error(f"导入通知配置失败: {e}")
            return False