"""
告警抑制规则配置管理器

负责管理告警抑制规则的配置、验证、加载和持久化，
支持多种抑制类型、智能抑制策略和动态规则管理。
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


class SuppressionType(Enum):
    """抑制类型"""
    TIME_WINDOW = "time_window"       # 时间窗口抑制
    FREQUENCY = "frequency"           # 频率抑制
    DEPENDENCY = "dependency"         # 依赖抑制
    PATTERN = "pattern"               # 模式抑制
    LABEL = "label"                   # 标签抑制
    MAINTENANCE = "maintenance"       # 维护抑制
    ESCALATION = "escalation"         # 升级抑制
    DUPLICATE = "duplicate"           # 重复抑制
    CORRELATION = "correlation"       # 关联抑制
    THRESHOLD = "threshold"           # 阈值抑制
    CONDITIONAL = "conditional"       # 条件抑制


class SuppressionAction(Enum):
    """抑制动作"""
    SUPPRESS = "suppress"             # 完全抑制
    DELAY = "delay"                   # 延迟发送
    AGGREGATE = "aggregate"           # 聚合发送
    DOWNGRADE = "downgrade"           # 降级处理
    REDIRECT = "redirect"             # 重定向通知


class SuppressionPriority(Enum):
    """抑制优先级"""
    CRITICAL = "critical"             # 关键（最高优先级）
    HIGH = "high"                     # 高
    MEDIUM = "medium"                 # 中
    LOW = "low"                       # 低
    IGNORE = "ignore"                 # 忽略（最低优先级）


@dataclass
class TimeWindow:
    """时间窗口"""
    start_time: str                   # 开始时间 (HH:MM)
    end_time: str                     # 结束时间 (HH:MM)
    weekdays: List[int] = field(default_factory=list)  # 工作日 (0=Monday, 6=Sunday)
    timezone: str = "UTC"             # 时区
    
    def is_in_window(self, timestamp: datetime) -> bool:
        """检查时间是否在窗口内"""
        # 检查工作日
        if self.weekdays and timestamp.weekday() not in self.weekdays:
            return False
        
        # 检查时间范围
        current_time = timestamp.strftime("%H:%M")
        
        # 处理跨日情况
        if self.start_time <= self.end_time:
            return self.start_time <= current_time <= self.end_time
        else:  # 跨日，如 22:00 到 06:00
            return current_time >= self.start_time or current_time <= self.end_time


@dataclass
class FrequencyLimit:
    """频率限制"""
    max_count: int                    # 最大次数
    time_window_seconds: int          # 时间窗口（秒）
    reset_on_success: bool = True     # 成功时重置计数
    
    def is_exceeded(self, count: int, window_start: datetime) -> bool:
        """检查是否超过频率限制"""
        if count >= self.max_count:
            window_age = (datetime.now() - window_start).total_seconds()
            return window_age < self.time_window_seconds
        return False


@dataclass
class DependencyRule:
    """依赖规则"""
    parent_alert_patterns: List[str]  # 父告警模式
    dependency_type: str = "service"  # 依赖类型: service, host, network
    timeout_seconds: int = 300        # 超时时间
    
    def matches_parent(self, alert_name: str) -> bool:
        """检查是否匹配父告警"""
        for pattern in self.parent_alert_patterns:
            if fnmatch.fnmatch(alert_name, pattern):
                return True
        return False


@dataclass
class PatternMatcher:
    """模式匹配器"""
    patterns: List[str]               # 匹配模式
    match_type: str = "any"           # 匹配类型: any, all
    case_sensitive: bool = False      # 是否区分大小写
    regex_enabled: bool = True        # 是否启用正则表达式
    
    def matches(self, text: str) -> bool:
        """检查文本是否匹配模式"""
        if not self.patterns:
            return False
        
        matches = []
        for pattern in self.patterns:
            if self.regex_enabled:
                flags = 0 if self.case_sensitive else re.IGNORECASE
                match = bool(re.search(pattern, text, flags))
            else:
                if self.case_sensitive:
                    match = pattern in text
                else:
                    match = pattern.lower() in text.lower()
            
            matches.append(match)
        
        if self.match_type == "all":
            return all(matches)
        else:  # any
            return any(matches)


@dataclass
class LabelCondition:
    """标签条件"""
    label_key: str                    # 标签键
    label_values: List[str]           # 标签值列表
    operator: str = "in"              # 操作符: in, not_in, equals, not_equals, exists, not_exists
    
    def matches(self, labels: Dict[str, str]) -> bool:
        """检查标签是否匹配条件"""
        if self.operator == "exists":
            return self.label_key in labels
        elif self.operator == "not_exists":
            return self.label_key not in labels
        
        label_value = labels.get(self.label_key, "")
        
        if self.operator == "in":
            return label_value in self.label_values
        elif self.operator == "not_in":
            return label_value not in self.label_values
        elif self.operator == "equals":
            return label_value == self.label_values[0] if self.label_values else False
        elif self.operator == "not_equals":
            return label_value != self.label_values[0] if self.label_values else True
        
        return False


@dataclass
class MaintenanceWindow:
    """维护窗口"""
    name: str
    description: str
    start_time: datetime
    end_time: datetime
    affected_services: List[str] = field(default_factory=list)
    affected_hosts: List[str] = field(default_factory=list)
    suppression_level: str = "all"    # all, critical_only, non_critical
    created_by: str = ""
    
    def is_active(self, timestamp: datetime = None) -> bool:
        """检查维护窗口是否活跃"""
        if timestamp is None:
            timestamp = datetime.now()
        
        return self.start_time <= timestamp <= self.end_time
    
    def affects_alert(self, alert_labels: Dict[str, str]) -> bool:
        """检查是否影响告警"""
        service = alert_labels.get("service", "")
        host = alert_labels.get("host", "")
        
        if self.affected_services and service:
            for pattern in self.affected_services:
                if fnmatch.fnmatch(service, pattern):
                    return True
        
        if self.affected_hosts and host:
            for pattern in self.affected_hosts:
                if fnmatch.fnmatch(host, pattern):
                    return True
        
        return not self.affected_services and not self.affected_hosts


@dataclass
class ConditionalRule:
    """条件规则"""
    condition_expression: str         # 条件表达式
    variables: Dict[str, Any] = field(default_factory=dict)  # 变量
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """评估条件"""
        try:
            # 简单的条件评估（实际应该使用更安全的表达式解析器）
            local_vars = {**self.variables, **context}
            
            # 替换变量
            expression = self.condition_expression
            for var, value in local_vars.items():
                expression = expression.replace(f"{{{var}}}", str(value))
            
            # 安全评估（仅支持基本比较操作）
            if any(op in expression for op in ["==", "!=", ">", "<", ">=", "<="]):
                return eval(expression, {"__builtins__": {}}, {})
            
            return False
            
        except Exception:
            return False


@dataclass
class SuppressionRule:
    """抑制规则"""
    id: str
    name: str
    description: str
    suppression_type: SuppressionType
    action: SuppressionAction
    priority: SuppressionPriority
    
    # 条件配置
    time_window: Optional[TimeWindow] = None
    frequency_limit: Optional[FrequencyLimit] = None
    dependency_rule: Optional[DependencyRule] = None
    pattern_matcher: Optional[PatternMatcher] = None
    label_conditions: List[LabelCondition] = field(default_factory=list)
    conditional_rule: Optional[ConditionalRule] = None
    
    # 动作配置
    delay_seconds: int = 0            # 延迟时间
    aggregate_window_seconds: int = 300  # 聚合窗口
    downgrade_severity: str = ""      # 降级严重级别
    redirect_channels: List[str] = field(default_factory=list)  # 重定向通道
    
    # 元数据
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    
    # 统计信息
    applied_count: int = 0
    last_applied: Optional[datetime] = None
    
    def matches_alert(self, alert_name: str, alert_labels: Dict[str, str],
                     alert_severity: str, timestamp: datetime = None) -> bool:
        """检查规则是否匹配告警"""
        if not self.enabled:
            return False
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # 检查时间窗口
        if self.time_window and not self.time_window.is_in_window(timestamp):
            return False
        
        # 检查模式匹配
        if self.pattern_matcher and not self.pattern_matcher.matches(alert_name):
            return False
        
        # 检查标签条件
        if self.label_conditions:
            for condition in self.label_conditions:
                if not condition.matches(alert_labels):
                    return False
        
        # 检查条件规则
        if self.conditional_rule:
            context = {
                "alert_name": alert_name,
                "alert_severity": alert_severity,
                "timestamp": timestamp.isoformat(),
                **alert_labels
            }
            if not self.conditional_rule.evaluate(context):
                return False
        
        return True
    
    def apply_suppression(self, timestamp: datetime = None):
        """应用抑制"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.applied_count += 1
        self.last_applied = timestamp


@dataclass
class SuppressionEvent:
    """抑制事件"""
    rule_id: str
    alert_name: str
    alert_labels: Dict[str, str]
    action: SuppressionAction
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_seconds: Optional[int] = None
    
    def age_seconds(self) -> float:
        """获取事件年龄（秒）"""
        return (datetime.now() - self.timestamp).total_seconds()


class AlertSuppressionConfigManager:
    """告警抑制规则配置管理器"""
    
    def __init__(self, config_dir: str = "config/alerts"):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 规则存储
        self.suppression_rules: Dict[str, SuppressionRule] = {}
        self.maintenance_windows: Dict[str, MaintenanceWindow] = {}
        
        # 运行时状态
        self.frequency_counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.frequency_windows: Dict[str, Dict[str, datetime]] = defaultdict(lambda: defaultdict(datetime))
        self.dependency_alerts: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.suppression_events: deque = deque(maxlen=10000)
        
        # 配置文件路径
        self.rules_file = self.config_dir / "suppression_rules.json"
        self.maintenance_file = self.config_dir / "maintenance_windows.json"
        
        # 加载配置
        self._load_default_rules()
        self._load_rules()
        self._load_maintenance_windows()
    
    def _load_default_rules(self):
        """加载默认抑制规则"""
        default_rules = [
            # 重复告警抑制
            SuppressionRule(
                id="duplicate_suppression",
                name="重复告警抑制",
                description="抑制相同告警的重复发送",
                suppression_type=SuppressionType.DUPLICATE,
                action=SuppressionAction.SUPPRESS,
                priority=SuppressionPriority.HIGH,
                frequency_limit=FrequencyLimit(
                    max_count=1,
                    time_window_seconds=300,
                    reset_on_success=True
                ),
                tags=["duplicate", "noise_reduction"]
            ),
            
            # 高频告警抑制
            SuppressionRule(
                id="high_frequency_suppression",
                name="高频告警抑制",
                description="抑制高频率的告警",
                suppression_type=SuppressionType.FREQUENCY,
                action=SuppressionAction.AGGREGATE,
                priority=SuppressionPriority.MEDIUM,
                frequency_limit=FrequencyLimit(
                    max_count=5,
                    time_window_seconds=600,
                    reset_on_success=False
                ),
                aggregate_window_seconds=300,
                tags=["frequency", "noise_reduction"]
            ),
            
            # 数据库依赖抑制
            SuppressionRule(
                id="database_dependency_suppression",
                name="数据库依赖抑制",
                description="当数据库告警时抑制相关服务告警",
                suppression_type=SuppressionType.DEPENDENCY,
                action=SuppressionAction.SUPPRESS,
                priority=SuppressionPriority.HIGH,
                dependency_rule=DependencyRule(
                    parent_alert_patterns=["database_*", "db_*"],
                    dependency_type="service",
                    timeout_seconds=600
                ),
                tags=["dependency", "database"]
            ),
            
            # 网络依赖抑制
            SuppressionRule(
                id="network_dependency_suppression",
                name="网络依赖抑制",
                description="当网络告警时抑制相关主机告警",
                suppression_type=SuppressionType.DEPENDENCY,
                action=SuppressionAction.SUPPRESS,
                priority=SuppressionPriority.HIGH,
                dependency_rule=DependencyRule(
                    parent_alert_patterns=["network_*", "connectivity_*"],
                    dependency_type="network",
                    timeout_seconds=300
                ),
                tags=["dependency", "network"]
            ),
            
            # 维护时间抑制
            SuppressionRule(
                id="maintenance_suppression",
                name="维护时间抑制",
                description="维护时间内抑制非关键告警",
                suppression_type=SuppressionType.MAINTENANCE,
                action=SuppressionAction.SUPPRESS,
                priority=SuppressionPriority.CRITICAL,
                label_conditions=[
                    LabelCondition(
                        label_key="severity",
                        label_values=["critical"],
                        operator="not_in"
                    )
                ],
                tags=["maintenance"]
            ),
            
            # 低优先级频率抑制
            SuppressionRule(
                id="low_priority_frequency_suppression",
                name="低优先级频率抑制",
                description="对低优先级告警进行更严格的频率控制",
                suppression_type=SuppressionType.FREQUENCY,
                action=SuppressionAction.DELAY,
                priority=SuppressionPriority.LOW,
                frequency_limit=FrequencyLimit(
                    max_count=3,
                    time_window_seconds=1800,  # 30分钟
                    reset_on_success=False
                ),
                delay_seconds=300,
                label_conditions=[
                    LabelCondition(
                        label_key="severity",
                        label_values=["low", "info"],
                        operator="in"
                    )
                ],
                tags=["frequency", "low_priority"]
            ),
            
            # 工作时间外抑制
            SuppressionRule(
                id="off_hours_suppression",
                name="工作时间外抑制",
                description="工作时间外抑制非关键告警",
                suppression_type=SuppressionType.TIME_WINDOW,
                action=SuppressionAction.DELAY,
                priority=SuppressionPriority.MEDIUM,
                time_window=TimeWindow(
                    start_time="18:00",
                    end_time="09:00",
                    weekdays=[0, 1, 2, 3, 4],  # Monday-Friday
                    timezone="UTC"
                ),
                delay_seconds=3600,  # 延迟1小时
                label_conditions=[
                    LabelCondition(
                        label_key="severity",
                        label_values=["critical", "high"],
                        operator="not_in"
                    )
                ],
                tags=["time_window", "off_hours"]
            ),
            
            # 测试环境抑制
            SuppressionRule(
                id="test_environment_suppression",
                name="测试环境抑制",
                description="抑制测试环境的告警",
                suppression_type=SuppressionType.LABEL,
                action=SuppressionAction.SUPPRESS,
                priority=SuppressionPriority.LOW,
                label_conditions=[
                    LabelCondition(
                        label_key="environment",
                        label_values=["test", "staging", "dev"],
                        operator="in"
                    )
                ],
                tags=["environment", "test"]
            ),
            
            # 模式匹配抑制
            SuppressionRule(
                id="pattern_suppression",
                name="模式匹配抑制",
                description="基于模式匹配的告警抑制",
                suppression_type=SuppressionType.PATTERN,
                action=SuppressionAction.DOWNGRADE,
                priority=SuppressionPriority.MEDIUM,
                pattern_matcher=PatternMatcher(
                    patterns=[".*test.*", ".*debug.*", ".*temp.*"],
                    match_type="any",
                    case_sensitive=False,
                    regex_enabled=True
                ),
                downgrade_severity="info",
                tags=["pattern", "downgrade"]
            ),
            
            # 阈值抑制
            SuppressionRule(
                id="threshold_suppression",
                name="阈值抑制",
                description="基于阈值的条件抑制",
                suppression_type=SuppressionType.THRESHOLD,
                action=SuppressionAction.SUPPRESS,
                priority=SuppressionPriority.MEDIUM,
                conditional_rule=ConditionalRule(
                    condition_expression="{cpu_usage} < 50 and {memory_usage} < 70",
                    variables={}
                ),
                tags=["threshold", "conditional"]
            )
        ]
        
        for rule in default_rules:
            self.suppression_rules[rule.id] = rule
    
    def _load_rules(self):
        """加载抑制规则"""
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                
                for rule_data in rules_data:
                    rule = self._dict_to_rule(rule_data)
                    if rule:
                        self.suppression_rules[rule.id] = rule
                
                self.logger.info(f"加载了 {len(self.suppression_rules)} 个抑制规则")
                
            except Exception as e:
                self.logger.error(f"加载抑制规则失败: {e}")
    
    def _load_maintenance_windows(self):
        """加载维护窗口"""
        if self.maintenance_file.exists():
            try:
                with open(self.maintenance_file, 'r', encoding='utf-8') as f:
                    windows_data = json.load(f)
                
                for window_data in windows_data:
                    window = self._dict_to_maintenance_window(window_data)
                    if window:
                        self.maintenance_windows[window.name] = window
                
                self.logger.info(f"加载了 {len(self.maintenance_windows)} 个维护窗口")
                
            except Exception as e:
                self.logger.error(f"加载维护窗口失败: {e}")
    
    def _dict_to_rule(self, rule_data: Dict[str, Any]) -> Optional[SuppressionRule]:
        """将字典转换为规则对象"""
        try:
            # 转换枚举类型
            rule_data["suppression_type"] = SuppressionType(rule_data["suppression_type"])
            rule_data["action"] = SuppressionAction(rule_data["action"])
            rule_data["priority"] = SuppressionPriority(rule_data["priority"])
            
            # 转换日期时间
            if "created_at" in rule_data:
                rule_data["created_at"] = datetime.fromisoformat(rule_data["created_at"])
            if "updated_at" in rule_data:
                rule_data["updated_at"] = datetime.fromisoformat(rule_data["updated_at"])
            if "last_applied" in rule_data and rule_data["last_applied"]:
                rule_data["last_applied"] = datetime.fromisoformat(rule_data["last_applied"])
            
            # 转换时间窗口
            if "time_window" in rule_data and rule_data["time_window"]:
                rule_data["time_window"] = TimeWindow(**rule_data["time_window"])
            
            # 转换频率限制
            if "frequency_limit" in rule_data and rule_data["frequency_limit"]:
                rule_data["frequency_limit"] = FrequencyLimit(**rule_data["frequency_limit"])
            
            # 转换依赖规则
            if "dependency_rule" in rule_data and rule_data["dependency_rule"]:
                rule_data["dependency_rule"] = DependencyRule(**rule_data["dependency_rule"])
            
            # 转换模式匹配器
            if "pattern_matcher" in rule_data and rule_data["pattern_matcher"]:
                rule_data["pattern_matcher"] = PatternMatcher(**rule_data["pattern_matcher"])
            
            # 转换标签条件
            if "label_conditions" in rule_data:
                conditions = []
                for cond_data in rule_data["label_conditions"]:
                    conditions.append(LabelCondition(**cond_data))
                rule_data["label_conditions"] = conditions
            
            # 转换条件规则
            if "conditional_rule" in rule_data and rule_data["conditional_rule"]:
                rule_data["conditional_rule"] = ConditionalRule(**rule_data["conditional_rule"])
            
            return SuppressionRule(**rule_data)
            
        except Exception as e:
            self.logger.error(f"转换抑制规则失败: {e}")
            return None
    
    def _dict_to_maintenance_window(self, window_data: Dict[str, Any]) -> Optional[MaintenanceWindow]:
        """将字典转换为维护窗口对象"""
        try:
            # 转换日期时间
            window_data["start_time"] = datetime.fromisoformat(window_data["start_time"])
            window_data["end_time"] = datetime.fromisoformat(window_data["end_time"])
            
            return MaintenanceWindow(**window_data)
            
        except Exception as e:
            self.logger.error(f"转换维护窗口失败: {e}")
            return None
    
    def _rule_to_dict(self, rule: SuppressionRule) -> Dict[str, Any]:
        """将规则对象转换为字典"""
        rule_dict = asdict(rule)
        
        # 转换枚举为字符串
        rule_dict["suppression_type"] = rule.suppression_type.value
        rule_dict["action"] = rule.action.value
        rule_dict["priority"] = rule.priority.value
        
        # 转换日期时间为字符串
        if rule.created_at:
            rule_dict["created_at"] = rule.created_at.isoformat()
        if rule.updated_at:
            rule_dict["updated_at"] = rule.updated_at.isoformat()
        if rule.last_applied:
            rule_dict["last_applied"] = rule.last_applied.isoformat()
        
        return rule_dict
    
    def _maintenance_window_to_dict(self, window: MaintenanceWindow) -> Dict[str, Any]:
        """将维护窗口对象转换为字典"""
        window_dict = asdict(window)
        
        # 转换日期时间为字符串
        window_dict["start_time"] = window.start_time.isoformat()
        window_dict["end_time"] = window.end_time.isoformat()
        
        return window_dict
    
    async def save_rules(self) -> bool:
        """保存抑制规则"""
        try:
            rules_data = [self._rule_to_dict(rule) for rule in self.suppression_rules.values()]
            
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"保存了 {len(self.suppression_rules)} 个抑制规则")
            return True
            
        except Exception as e:
            self.logger.error(f"保存抑制规则失败: {e}")
            return False
    
    async def save_maintenance_windows(self) -> bool:
        """保存维护窗口"""
        try:
            windows_data = [self._maintenance_window_to_dict(window) 
                          for window in self.maintenance_windows.values()]
            
            with open(self.maintenance_file, 'w', encoding='utf-8') as f:
                json.dump(windows_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"保存了 {len(self.maintenance_windows)} 个维护窗口")
            return True
            
        except Exception as e:
            self.logger.error(f"保存维护窗口失败: {e}")
            return False
    
    async def add_rule(self, rule: SuppressionRule) -> bool:
        """添加抑制规则"""
        if rule.id in self.suppression_rules:
            self.logger.warning(f"抑制规则已存在: {rule.id}")
            return False
        
        self.suppression_rules[rule.id] = rule
        await self.save_rules()
        
        self.logger.info(f"添加抑制规则: {rule.id}")
        return True
    
    async def update_rule(self, rule_id: str, rule: SuppressionRule) -> bool:
        """更新抑制规则"""
        if rule_id not in self.suppression_rules:
            self.logger.warning(f"抑制规则不存在: {rule_id}")
            return False
        
        rule.updated_at = datetime.now()
        self.suppression_rules[rule_id] = rule
        await self.save_rules()
        
        self.logger.info(f"更新抑制规则: {rule_id}")
        return True
    
    async def remove_rule(self, rule_id: str) -> bool:
        """删除抑制规则"""
        if rule_id not in self.suppression_rules:
            self.logger.warning(f"抑制规则不存在: {rule_id}")
            return False
        
        del self.suppression_rules[rule_id]
        await self.save_rules()
        
        self.logger.info(f"删除抑制规则: {rule_id}")
        return True
    
    async def add_maintenance_window(self, window: MaintenanceWindow) -> bool:
        """添加维护窗口"""
        if window.name in self.maintenance_windows:
            self.logger.warning(f"维护窗口已存在: {window.name}")
            return False
        
        self.maintenance_windows[window.name] = window
        await self.save_maintenance_windows()
        
        self.logger.info(f"添加维护窗口: {window.name}")
        return True
    
    async def remove_maintenance_window(self, window_name: str) -> bool:
        """删除维护窗口"""
        if window_name not in self.maintenance_windows:
            self.logger.warning(f"维护窗口不存在: {window_name}")
            return False
        
        del self.maintenance_windows[window_name]
        await self.save_maintenance_windows()
        
        self.logger.info(f"删除维护窗口: {window_name}")
        return True
    
    def get_rule(self, rule_id: str) -> Optional[SuppressionRule]:
        """获取抑制规则"""
        return self.suppression_rules.get(rule_id)
    
    def get_rules(self, suppression_type: Optional[SuppressionType] = None,
                  action: Optional[SuppressionAction] = None,
                  priority: Optional[SuppressionPriority] = None,
                  enabled_only: bool = True) -> List[SuppressionRule]:
        """获取抑制规则列表"""
        rules = list(self.suppression_rules.values())
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        if suppression_type:
            rules = [r for r in rules if r.suppression_type == suppression_type]
        
        if action:
            rules = [r for r in rules if r.action == action]
        
        if priority:
            rules = [r for r in rules if r.priority == priority]
        
        # 按优先级排序
        priority_order = {
            SuppressionPriority.CRITICAL: 0,
            SuppressionPriority.HIGH: 1,
            SuppressionPriority.MEDIUM: 2,
            SuppressionPriority.LOW: 3,
            SuppressionPriority.IGNORE: 4
        }
        
        rules.sort(key=lambda r: priority_order.get(r.priority, 5))
        
        return rules
    
    def get_active_maintenance_windows(self, timestamp: datetime = None) -> List[MaintenanceWindow]:
        """获取活跃的维护窗口"""
        if timestamp is None:
            timestamp = datetime.now()
        
        return [window for window in self.maintenance_windows.values() 
                if window.is_active(timestamp)]
    
    async def should_suppress_alert(self, alert_name: str, alert_labels: Dict[str, str],
                                  alert_severity: str, timestamp: datetime = None) -> Tuple[bool, List[SuppressionRule], str]:
        """检查是否应该抑制告警"""
        if timestamp is None:
            timestamp = datetime.now()
        
        applied_rules = []
        suppression_reasons = []
        
        # 检查维护窗口
        active_windows = self.get_active_maintenance_windows(timestamp)
        for window in active_windows:
            if window.affects_alert(alert_labels):
                if (window.suppression_level == "all" or
                    (window.suppression_level == "critical_only" and alert_severity == "critical") or
                    (window.suppression_level == "non_critical" and alert_severity != "critical")):
                    
                    suppression_reasons.append(f"维护窗口: {window.name}")
        
        # 检查抑制规则
        rules = self.get_rules(enabled_only=True)
        
        for rule in rules:
            if rule.matches_alert(alert_name, alert_labels, alert_severity, timestamp):
                
                # 检查频率限制
                if rule.frequency_limit:
                    alert_key = f"{alert_name}:{hash(str(sorted(alert_labels.items())))}"
                    
                    # 更新频率计数
                    if alert_key not in self.frequency_windows[rule.id]:
                        self.frequency_windows[rule.id][alert_key] = timestamp
                        self.frequency_counters[rule.id][alert_key] = 1
                    else:
                        window_start = self.frequency_windows[rule.id][alert_key]
                        window_age = (timestamp - window_start).total_seconds()
                        
                        if window_age >= rule.frequency_limit.time_window_seconds:
                            # 重置窗口
                            self.frequency_windows[rule.id][alert_key] = timestamp
                            self.frequency_counters[rule.id][alert_key] = 1
                        else:
                            self.frequency_counters[rule.id][alert_key] += 1
                    
                    # 检查是否超过频率限制
                    current_count = self.frequency_counters[rule.id][alert_key]
                    window_start = self.frequency_windows[rule.id][alert_key]
                    
                    if rule.frequency_limit.is_exceeded(current_count, window_start):
                        applied_rules.append(rule)
                        suppression_reasons.append(f"频率限制: {rule.name} (计数: {current_count})")
                        rule.apply_suppression(timestamp)
                        continue
                
                # 检查依赖规则
                if rule.dependency_rule:
                    # 检查是否有父告警
                    has_parent_alert = False
                    for parent_pattern in rule.dependency_rule.parent_alert_patterns:
                        if parent_pattern in self.dependency_alerts:
                            has_parent_alert = True
                            break
                    
                    if has_parent_alert:
                        applied_rules.append(rule)
                        suppression_reasons.append(f"依赖抑制: {rule.name}")
                        rule.apply_suppression(timestamp)
                        continue
                
                # 其他类型的规则直接应用
                if rule.suppression_type in [SuppressionType.TIME_WINDOW, SuppressionType.LABEL,
                                           SuppressionType.PATTERN, SuppressionType.CONDITIONAL,
                                           SuppressionType.THRESHOLD]:
                    applied_rules.append(rule)
                    suppression_reasons.append(f"{rule.suppression_type.value}: {rule.name}")
                    rule.apply_suppression(timestamp)
        
        # 记录抑制事件
        if applied_rules:
            for rule in applied_rules:
                event = SuppressionEvent(
                    rule_id=rule.id,
                    alert_name=alert_name,
                    alert_labels=alert_labels,
                    action=rule.action,
                    reason=f"{rule.suppression_type.value}: {rule.name}",
                    timestamp=timestamp
                )
                self.suppression_events.append(event)
        
        is_suppressed = len(applied_rules) > 0 or len(active_windows) > 0
        reason = "; ".join(suppression_reasons) if suppression_reasons else ""
        
        return is_suppressed, applied_rules, reason
    
    def register_parent_alert(self, alert_name: str, child_alerts: List[str]):
        """注册父告警"""
        self.dependency_alerts[alert_name].update(child_alerts)
        
        # 设置超时清理
        # 这里应该实现超时清理逻辑
    
    def unregister_parent_alert(self, alert_name: str):
        """注销父告警"""
        if alert_name in self.dependency_alerts:
            del self.dependency_alerts[alert_name]
    
    async def get_suppression_statistics(self) -> Dict[str, Any]:
        """获取抑制统计信息"""
        total_rules = len(self.suppression_rules)
        enabled_rules = len([r for r in self.suppression_rules.values() if r.enabled])
        
        # 按类型统计
        type_stats = {}
        for suppression_type in SuppressionType:
            count = len([r for r in self.suppression_rules.values() 
                        if r.suppression_type == suppression_type])
            type_stats[suppression_type.value] = count
        
        # 按动作统计
        action_stats = {}
        for action in SuppressionAction:
            count = len([r for r in self.suppression_rules.values() if r.action == action])
            action_stats[action.value] = count
        
        # 应用统计
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        applied_24h = len([
            r for r in self.suppression_rules.values()
            if r.last_applied and r.last_applied > last_24h
        ])
        
        # 事件统计
        events_24h = len([
            e for e in self.suppression_events
            if e.age_seconds() <= 86400
        ])
        
        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "total_maintenance_windows": len(self.maintenance_windows),
            "active_maintenance_windows": len(self.get_active_maintenance_windows()),
            "suppression_type_distribution": type_stats,
            "action_distribution": action_stats,
            "applied_rules_24h": applied_24h,
            "suppression_events_24h": events_24h,
            "total_suppression_events": len(self.suppression_events),
            "dependency_alerts": len(self.dependency_alerts)
        }
    
    async def cleanup_old_events(self, retention_hours: int = 168):  # 7天
        """清理旧事件"""
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        # 清理抑制事件
        original_count = len(self.suppression_events)
        self.suppression_events = deque(
            [e for e in self.suppression_events if e.timestamp >= cutoff_time],
            maxlen=self.suppression_events.maxlen
        )
        
        cleaned_events = original_count - len(self.suppression_events)
        
        # 清理频率计数器
        cleaned_counters = 0
        for rule_id in list(self.frequency_windows.keys()):
            for alert_key in list(self.frequency_windows[rule_id].keys()):
                if self.frequency_windows[rule_id][alert_key] < cutoff_time:
                    del self.frequency_windows[rule_id][alert_key]
                    if alert_key in self.frequency_counters[rule_id]:
                        del self.frequency_counters[rule_id][alert_key]
                    cleaned_counters += 1
        
        # 清理过期的维护窗口
        expired_windows = []
        for name, window in self.maintenance_windows.items():
            if window.end_time < cutoff_time:
                expired_windows.append(name)
        
        for name in expired_windows:
            del self.maintenance_windows[name]
        
        if cleaned_events > 0 or cleaned_counters > 0 or expired_windows:
            self.logger.info(f"清理了 {cleaned_events} 个事件, "
                           f"{cleaned_counters} 个频率计数器, "
                           f"{len(expired_windows)} 个过期维护窗口")
    
    async def export_rules(self, export_path: str, rule_ids: Optional[List[str]] = None) -> bool:
        """导出抑制规则"""
        try:
            if rule_ids:
                rules_to_export = [self.suppression_rules[rid] for rid in rule_ids 
                                 if rid in self.suppression_rules]
            else:
                rules_to_export = list(self.suppression_rules.values())
            
            export_data = {
                "rules": [self._rule_to_dict(rule) for rule in rules_to_export],
                "maintenance_windows": [self._maintenance_window_to_dict(window) 
                                      for window in self.maintenance_windows.values()],
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功导出 {len(rules_to_export)} 个抑制规则到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出抑制规则失败: {e}")
            return False
    
    async def import_rules(self, import_path: str, overwrite: bool = False) -> bool:
        """导入抑制规则"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 导入规则
            rules_data = import_data.get("rules", [])
            imported_rules = 0
            
            for rule_data in rules_data:
                rule = self._dict_to_rule(rule_data)
                if not rule:
                    continue
                
                if rule.id in self.suppression_rules and not overwrite:
                    self.logger.warning(f"抑制规则已存在，跳过: {rule.id}")
                    continue
                
                self.suppression_rules[rule.id] = rule
                imported_rules += 1
            
            # 导入维护窗口
            windows_data = import_data.get("maintenance_windows", [])
            imported_windows = 0
            
            for window_data in windows_data:
                window = self._dict_to_maintenance_window(window_data)
                if not window:
                    continue
                
                if window.name in self.maintenance_windows and not overwrite:
                    self.logger.warning(f"维护窗口已存在，跳过: {window.name}")
                    continue
                
                self.maintenance_windows[window.name] = window
                imported_windows += 1
            
            await self.save_rules()
            await self.save_maintenance_windows()
            
            self.logger.info(f"成功导入 {imported_rules} 个抑制规则和 {imported_windows} 个维护窗口")
            return True
            
        except Exception as e:
            self.logger.error(f"导入抑制规则失败: {e}")
            return False