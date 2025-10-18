"""
告警抑制引擎

负责管理复杂的告警抑制逻辑，包括时间窗口抑制、依赖关系抑制、
频率抑制、条件抑制等多种抑制策略。
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque


class SuppressionType(Enum):
    """抑制类型"""
    TIME_WINDOW = "time_window"        # 时间窗口抑制
    DEPENDENCY = "dependency"          # 依赖关系抑制
    FREQUENCY = "frequency"            # 频率抑制
    CONDITION = "condition"            # 条件抑制
    MAINTENANCE = "maintenance"        # 维护窗口抑制
    ESCALATION = "escalation"          # 升级抑制
    DUPLICATE = "duplicate"            # 重复抑制
    CORRELATION = "correlation"        # 关联抑制
    THRESHOLD = "threshold"            # 阈值抑制
    CUSTOM = "custom"                  # 自定义抑制


class SuppressionStatus(Enum):
    """抑制状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    DISABLED = "disabled"


class SuppressionPriority(Enum):
    """抑制优先级"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class TimeWindow:
    """时间窗口"""
    start_time: datetime
    end_time: datetime
    timezone: str = "UTC"
    
    def contains(self, timestamp: datetime) -> bool:
        """检查时间戳是否在窗口内"""
        return self.start_time <= timestamp <= self.end_time
    
    def overlaps(self, other: 'TimeWindow') -> bool:
        """检查是否与另一个时间窗口重叠"""
        return (self.start_time <= other.end_time and 
                self.end_time >= other.start_time)


@dataclass
class SuppressionCondition:
    """抑制条件"""
    field: str                          # 字段名
    operator: str                       # 操作符 (eq, ne, in, not_in, regex, gt, lt, gte, lte)
    value: Union[str, int, float, List] # 值
    case_sensitive: bool = True         # 是否区分大小写
    
    def matches(self, data: Dict[str, Any]) -> bool:
        """检查数据是否匹配条件"""
        if self.field not in data:
            return False
        
        field_value = data[self.field]
        
        # 字符串比较时考虑大小写
        if isinstance(field_value, str) and isinstance(self.value, str) and not self.case_sensitive:
            field_value = field_value.lower()
            compare_value = self.value.lower()
        else:
            compare_value = self.value
        
        if self.operator == "eq":
            return field_value == compare_value
        elif self.operator == "ne":
            return field_value != compare_value
        elif self.operator == "in":
            return field_value in compare_value
        elif self.operator == "not_in":
            return field_value not in compare_value
        elif self.operator == "regex":
            pattern = compare_value if self.case_sensitive else f"(?i){compare_value}"
            return bool(re.search(pattern, str(field_value)))
        elif self.operator == "gt":
            return field_value > compare_value
        elif self.operator == "lt":
            return field_value < compare_value
        elif self.operator == "gte":
            return field_value >= compare_value
        elif self.operator == "lte":
            return field_value <= compare_value
        elif self.operator == "contains":
            return str(compare_value) in str(field_value)
        elif self.operator == "starts_with":
            return str(field_value).startswith(str(compare_value))
        elif self.operator == "ends_with":
            return str(field_value).endswith(str(compare_value))
        
        return False


@dataclass
class DependencyRule:
    """依赖规则"""
    parent_conditions: List[SuppressionCondition]  # 父告警条件
    child_conditions: List[SuppressionCondition]   # 子告警条件
    dependency_type: str = "blocks"                # 依赖类型: blocks, requires
    timeout: Optional[int] = None                  # 超时时间（秒）
    
    def is_parent_alert(self, alert_data: Dict[str, Any]) -> bool:
        """检查是否为父告警"""
        return all(condition.matches(alert_data) for condition in self.parent_conditions)
    
    def is_child_alert(self, alert_data: Dict[str, Any]) -> bool:
        """检查是否为子告警"""
        return all(condition.matches(alert_data) for condition in self.child_conditions)


@dataclass
class FrequencyRule:
    """频率规则"""
    max_count: int                     # 最大次数
    time_window: int                   # 时间窗口（秒）
    conditions: List[SuppressionCondition] = field(default_factory=list)  # 匹配条件
    group_by: List[str] = field(default_factory=list)  # 分组字段
    
    def get_group_key(self, alert_data: Dict[str, Any]) -> str:
        """获取分组键"""
        if not self.group_by:
            return "default"
        
        key_parts = []
        for field in self.group_by:
            value = alert_data.get(field, "")
            key_parts.append(f"{field}={value}")
        
        return "|".join(key_parts)


@dataclass
class MaintenanceWindow:
    """维护窗口"""
    id: str
    name: str
    start_time: datetime
    end_time: datetime
    description: str = ""
    affected_services: List[str] = field(default_factory=list)
    affected_hosts: List[str] = field(default_factory=list)
    created_by: str = ""
    
    def is_active(self, timestamp: datetime = None) -> bool:
        """检查维护窗口是否激活"""
        if timestamp is None:
            timestamp = datetime.now()
        return self.start_time <= timestamp <= self.end_time
    
    def affects_alert(self, alert_data: Dict[str, Any]) -> bool:
        """检查是否影响告警"""
        # 检查服务
        alert_service = alert_data.get('service', '')
        if alert_service and alert_service in self.affected_services:
            return True
        
        # 检查主机
        alert_host = alert_data.get('host', '')
        if alert_host and alert_host in self.affected_hosts:
            return True
        
        return False


@dataclass
class SuppressionRule:
    """抑制规则"""
    id: str
    name: str
    suppression_type: SuppressionType
    priority: SuppressionPriority
    enabled: bool = True
    
    # 基本条件
    conditions: List[SuppressionCondition] = field(default_factory=list)
    
    # 时间相关
    time_window: Optional[TimeWindow] = None
    duration: Optional[int] = None  # 抑制持续时间（秒）
    
    # 类型特定配置
    dependency_rule: Optional[DependencyRule] = None
    frequency_rule: Optional[FrequencyRule] = None
    maintenance_window: Optional[MaintenanceWindow] = None
    
    # 元数据
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    
    # 统计信息
    suppressed_count: int = 0
    last_triggered: Optional[datetime] = None
    
    def matches_alert(self, alert_data: Dict[str, Any]) -> bool:
        """检查告警是否匹配规则"""
        if not self.enabled:
            return False
        
        # 检查基本条件
        if self.conditions:
            if not all(condition.matches(alert_data) for condition in self.conditions):
                return False
        
        # 检查时间窗口
        if self.time_window:
            alert_time = alert_data.get('timestamp')
            if isinstance(alert_time, str):
                alert_time = datetime.fromisoformat(alert_time)
            elif alert_time is None:
                alert_time = datetime.now()
            
            if not self.time_window.contains(alert_time):
                return False
        
        return True


@dataclass
class SuppressionEvent:
    """抑制事件"""
    id: str
    rule_id: str
    alert_id: str
    suppression_type: SuppressionType
    timestamp: datetime
    reason: str
    alert_data: Dict[str, Any]
    duration: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "alert_id": self.alert_id,
            "suppression_type": self.suppression_type.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "alert_data": self.alert_data,
            "duration": self.duration
        }


class SuppressionEngine:
    """抑制引擎"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules: Dict[str, SuppressionRule] = {}
        self.maintenance_windows: Dict[str, MaintenanceWindow] = {}
        
        # 状态跟踪
        self.suppression_events: List[SuppressionEvent] = []
        self.active_suppressions: Dict[str, List[str]] = defaultdict(list)  # alert_id -> rule_ids
        self.frequency_counters: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # parent -> children
        self.active_dependencies: Dict[str, datetime] = {}  # parent_alert_id -> timestamp
        
        # 缓存
        self.rule_cache: Dict[str, List[SuppressionRule]] = {}
        self.cache_ttl = 300  # 5分钟缓存
        self.last_cache_update = datetime.now()
        
        # 加载默认规则
        self._load_default_rules()
    
    def _load_default_rules(self):
        """加载默认抑制规则"""
        default_rules = [
            # 重复告警抑制
            SuppressionRule(
                id="duplicate_suppression",
                name="重复告警抑制",
                suppression_type=SuppressionType.DUPLICATE,
                priority=SuppressionPriority.HIGH,
                conditions=[],  # 将在运行时动态匹配
                duration=300,   # 5分钟内的重复告警
                description="抑制5分钟内的重复告警"
            ),
            
            # 高频告警抑制
            SuppressionRule(
                id="high_frequency_suppression",
                name="高频告警抑制",
                suppression_type=SuppressionType.FREQUENCY,
                priority=SuppressionPriority.MEDIUM,
                frequency_rule=FrequencyRule(
                    max_count=10,
                    time_window=300,  # 5分钟内超过10次
                    group_by=["alert_name", "host"]
                ),
                description="抑制5分钟内超过10次的高频告警"
            ),
            
            # 数据库依赖抑制
            SuppressionRule(
                id="database_dependency_suppression",
                name="数据库依赖抑制",
                suppression_type=SuppressionType.DEPENDENCY,
                priority=SuppressionPriority.CRITICAL,
                dependency_rule=DependencyRule(
                    parent_conditions=[
                        SuppressionCondition("service", "eq", "database"),
                        SuppressionCondition("severity", "in", ["critical", "high"])
                    ],
                    child_conditions=[
                        SuppressionCondition("category", "eq", "application"),
                        SuppressionCondition("error_type", "regex", ".*database.*|.*connection.*")
                    ],
                    timeout=1800  # 30分钟超时
                ),
                description="数据库故障时抑制相关应用告警"
            ),
            
            # 网络依赖抑制
            SuppressionRule(
                id="network_dependency_suppression",
                name="网络依赖抑制",
                suppression_type=SuppressionType.DEPENDENCY,
                priority=SuppressionPriority.CRITICAL,
                dependency_rule=DependencyRule(
                    parent_conditions=[
                        SuppressionCondition("category", "eq", "network"),
                        SuppressionCondition("severity", "in", ["critical", "high"])
                    ],
                    child_conditions=[
                        SuppressionCondition("error_type", "regex", ".*timeout.*|.*unreachable.*|.*connection.*")
                    ],
                    timeout=900  # 15分钟超时
                ),
                description="网络故障时抑制相关连接告警"
            ),
            
            # 维护窗口抑制
            SuppressionRule(
                id="maintenance_suppression",
                name="维护窗口抑制",
                suppression_type=SuppressionType.MAINTENANCE,
                priority=SuppressionPriority.HIGH,
                description="维护窗口期间抑制相关告警"
            ),
            
            # 低优先级告警频率抑制
            SuppressionRule(
                id="low_priority_frequency_suppression",
                name="低优先级告警频率抑制",
                suppression_type=SuppressionType.FREQUENCY,
                priority=SuppressionPriority.LOW,
                conditions=[
                    SuppressionCondition("severity", "in", ["low", "info"])
                ],
                frequency_rule=FrequencyRule(
                    max_count=5,
                    time_window=600,  # 10分钟内超过5次
                    group_by=["alert_name"]
                ),
                description="抑制10分钟内超过5次的低优先级告警"
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
    
    async def add_rule(self, rule: SuppressionRule) -> bool:
        """添加抑制规则"""
        if rule.id in self.rules:
            self.logger.warning(f"抑制规则已存在: {rule.id}")
            return False
        
        # 验证规则
        errors = await self._validate_rule(rule)
        if errors:
            self.logger.error(f"抑制规则验证失败: {errors}")
            return False
        
        self.rules[rule.id] = rule
        self._invalidate_cache()
        self.logger.info(f"添加抑制规则: {rule.id}")
        return True
    
    async def update_rule(self, rule_id: str, rule: SuppressionRule) -> bool:
        """更新抑制规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"抑制规则不存在: {rule_id}")
            return False
        
        # 验证规则
        errors = await self._validate_rule(rule)
        if errors:
            self.logger.error(f"抑制规则验证失败: {errors}")
            return False
        
        rule.updated_at = datetime.now()
        self.rules[rule_id] = rule
        self._invalidate_cache()
        self.logger.info(f"更新抑制规则: {rule_id}")
        return True
    
    async def remove_rule(self, rule_id: str) -> bool:
        """删除抑制规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"抑制规则不存在: {rule_id}")
            return False
        
        del self.rules[rule_id]
        
        # 清理相关数据
        self._cleanup_rule_data(rule_id)
        self._invalidate_cache()
        
        self.logger.info(f"删除抑制规则: {rule_id}")
        return True
    
    async def _validate_rule(self, rule: SuppressionRule) -> List[str]:
        """验证抑制规则"""
        errors = []
        
        # 基本验证
        if not rule.id:
            errors.append("规则ID不能为空")
        
        if not rule.name:
            errors.append("规则名称不能为空")
        
        # 类型特定验证
        if rule.suppression_type == SuppressionType.FREQUENCY:
            if not rule.frequency_rule:
                errors.append("频率抑制规则缺少频率配置")
            elif rule.frequency_rule.max_count <= 0:
                errors.append("频率抑制最大次数必须大于0")
            elif rule.frequency_rule.time_window <= 0:
                errors.append("频率抑制时间窗口必须大于0")
        
        if rule.suppression_type == SuppressionType.DEPENDENCY:
            if not rule.dependency_rule:
                errors.append("依赖抑制规则缺少依赖配置")
            elif not rule.dependency_rule.parent_conditions:
                errors.append("依赖抑制规则缺少父告警条件")
            elif not rule.dependency_rule.child_conditions:
                errors.append("依赖抑制规则缺少子告警条件")
        
        if rule.suppression_type == SuppressionType.TIME_WINDOW:
            if not rule.time_window:
                errors.append("时间窗口抑制规则缺少时间窗口配置")
            elif rule.time_window.start_time >= rule.time_window.end_time:
                errors.append("时间窗口开始时间必须小于结束时间")
        
        return errors
    
    def _cleanup_rule_data(self, rule_id: str):
        """清理规则相关数据"""
        # 清理活跃抑制
        for alert_id in list(self.active_suppressions.keys()):
            if rule_id in self.active_suppressions[alert_id]:
                self.active_suppressions[alert_id].remove(rule_id)
                if not self.active_suppressions[alert_id]:
                    del self.active_suppressions[alert_id]
        
        # 清理频率计数器
        if rule_id in self.frequency_counters:
            del self.frequency_counters[rule_id]
    
    def _invalidate_cache(self):
        """使缓存失效"""
        self.rule_cache.clear()
        self.last_cache_update = datetime.now()
    
    async def should_suppress_alert(self, alert_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """检查告警是否应该被抑制"""
        alert_id = alert_data.get('alert_id', '')
        suppression_reasons = []
        
        # 获取匹配的规则
        matching_rules = await self._get_matching_rules(alert_data)
        
        for rule in matching_rules:
            should_suppress, reason = await self._check_rule_suppression(rule, alert_data)
            
            if should_suppress:
                suppression_reasons.append(reason)
                
                # 记录抑制事件
                event = SuppressionEvent(
                    id=f"{alert_id}_{rule.id}_{datetime.now().timestamp()}",
                    rule_id=rule.id,
                    alert_id=alert_id,
                    suppression_type=rule.suppression_type,
                    timestamp=datetime.now(),
                    reason=reason,
                    alert_data=alert_data,
                    duration=rule.duration
                )
                
                self.suppression_events.append(event)
                self.active_suppressions[alert_id].append(rule.id)
                
                # 更新规则统计
                rule.suppressed_count += 1
                rule.last_triggered = datetime.now()
        
        is_suppressed = len(suppression_reasons) > 0
        
        if is_suppressed:
            self.logger.info(f"告警被抑制: {alert_id}, 原因: {suppression_reasons}")
        
        return is_suppressed, suppression_reasons
    
    async def _get_matching_rules(self, alert_data: Dict[str, Any]) -> List[SuppressionRule]:
        """获取匹配的规则"""
        # 检查缓存
        cache_key = self._get_cache_key(alert_data)
        if (cache_key in self.rule_cache and 
            datetime.now() - self.last_cache_update < timedelta(seconds=self.cache_ttl)):
            return self.rule_cache[cache_key]
        
        # 查找匹配的规则
        matching_rules = []
        for rule in self.rules.values():
            if rule.matches_alert(alert_data):
                matching_rules.append(rule)
        
        # 按优先级排序
        matching_rules.sort(key=lambda r: r.priority.value)
        
        # 缓存结果
        self.rule_cache[cache_key] = matching_rules
        
        return matching_rules
    
    def _get_cache_key(self, alert_data: Dict[str, Any]) -> str:
        """生成缓存键"""
        key_fields = ['alert_name', 'service', 'host', 'severity', 'category']
        key_parts = []
        
        for field in key_fields:
            value = alert_data.get(field, '')
            key_parts.append(f"{field}={value}")
        
        return "|".join(key_parts)
    
    async def _check_rule_suppression(self, rule: SuppressionRule, 
                                    alert_data: Dict[str, Any]) -> Tuple[bool, str]:
        """检查单个规则的抑制逻辑"""
        alert_id = alert_data.get('alert_id', '')
        
        if rule.suppression_type == SuppressionType.DUPLICATE:
            return await self._check_duplicate_suppression(rule, alert_data)
        
        elif rule.suppression_type == SuppressionType.FREQUENCY:
            return await self._check_frequency_suppression(rule, alert_data)
        
        elif rule.suppression_type == SuppressionType.DEPENDENCY:
            return await self._check_dependency_suppression(rule, alert_data)
        
        elif rule.suppression_type == SuppressionType.MAINTENANCE:
            return await self._check_maintenance_suppression(rule, alert_data)
        
        elif rule.suppression_type == SuppressionType.TIME_WINDOW:
            return await self._check_time_window_suppression(rule, alert_data)
        
        elif rule.suppression_type == SuppressionType.THRESHOLD:
            return await self._check_threshold_suppression(rule, alert_data)
        
        elif rule.suppression_type == SuppressionType.CORRELATION:
            return await self._check_correlation_suppression(rule, alert_data)
        
        else:
            return False, ""
    
    async def _check_duplicate_suppression(self, rule: SuppressionRule, 
                                         alert_data: Dict[str, Any]) -> Tuple[bool, str]:
        """检查重复告警抑制"""
        alert_id = alert_data.get('alert_id', '')
        alert_name = alert_data.get('alert_name', '')
        host = alert_data.get('host', '')
        
        # 生成重复检查键
        duplicate_key = f"{alert_name}_{host}"
        
        # 检查最近的抑制事件
        now = datetime.now()
        duration = rule.duration or 300  # 默认5分钟
        cutoff_time = now - timedelta(seconds=duration)
        
        recent_events = [
            event for event in self.suppression_events
            if (event.timestamp > cutoff_time and 
                event.alert_data.get('alert_name') == alert_name and
                event.alert_data.get('host') == host and
                event.alert_id != alert_id)
        ]
        
        if recent_events:
            return True, f"重复告警抑制: {duration}秒内已有相同告警"
        
        return False, ""
    
    async def _check_frequency_suppression(self, rule: SuppressionRule, 
                                         alert_data: Dict[str, Any]) -> Tuple[bool, str]:
        """检查频率抑制"""
        if not rule.frequency_rule:
            return False, ""
        
        freq_rule = rule.frequency_rule
        group_key = freq_rule.get_group_key(alert_data)
        
        # 获取或创建计数器
        if rule.id not in self.frequency_counters:
            self.frequency_counters[rule.id] = defaultdict(deque)
        
        counter = self.frequency_counters[rule.id][group_key]
        now = datetime.now()
        
        # 清理过期记录
        cutoff_time = now - timedelta(seconds=freq_rule.time_window)
        while counter and counter[0] < cutoff_time:
            counter.popleft()
        
        # 检查是否超过频率限制
        if len(counter) >= freq_rule.max_count:
            return True, f"频率抑制: {freq_rule.time_window}秒内超过{freq_rule.max_count}次"
        
        # 添加当前记录
        counter.append(now)
        
        return False, ""
    
    async def _check_dependency_suppression(self, rule: SuppressionRule, 
                                          alert_data: Dict[str, Any]) -> Tuple[bool, str]:
        """检查依赖抑制"""
        if not rule.dependency_rule:
            return False, ""
        
        dep_rule = rule.dependency_rule
        
        # 检查是否为子告警
        if not dep_rule.is_child_alert(alert_data):
            return False, ""
        
        # 查找活跃的父告警
        now = datetime.now()
        timeout = dep_rule.timeout or 3600  # 默认1小时超时
        
        for parent_alert_id, timestamp in list(self.active_dependencies.items()):
            # 检查超时
            if now - timestamp > timedelta(seconds=timeout):
                del self.active_dependencies[parent_alert_id]
                continue
            
            # 检查是否有匹配的父告警
            for event in self.suppression_events:
                if (event.alert_id == parent_alert_id and 
                    dep_rule.is_parent_alert(event.alert_data)):
                    return True, f"依赖抑制: 父告警 {parent_alert_id} 仍然活跃"
        
        return False, ""
    
    async def _check_maintenance_suppression(self, rule: SuppressionRule, 
                                           alert_data: Dict[str, Any]) -> Tuple[bool, str]:
        """检查维护窗口抑制"""
        now = datetime.now()
        
        # 检查所有活跃的维护窗口
        for window in self.maintenance_windows.values():
            if window.is_active(now) and window.affects_alert(alert_data):
                return True, f"维护窗口抑制: {window.name} ({window.id})"
        
        return False, ""
    
    async def _check_time_window_suppression(self, rule: SuppressionRule, 
                                           alert_data: Dict[str, Any]) -> Tuple[bool, str]:
        """检查时间窗口抑制"""
        if not rule.time_window:
            return False, ""
        
        alert_time = alert_data.get('timestamp')
        if isinstance(alert_time, str):
            alert_time = datetime.fromisoformat(alert_time)
        elif alert_time is None:
            alert_time = datetime.now()
        
        if rule.time_window.contains(alert_time):
            return True, f"时间窗口抑制: {rule.time_window.start_time} - {rule.time_window.end_time}"
        
        return False, ""
    
    async def _check_threshold_suppression(self, rule: SuppressionRule, 
                                         alert_data: Dict[str, Any]) -> Tuple[bool, str]:
        """检查阈值抑制"""
        # 实现阈值抑制逻辑
        # 例如：当告警值低于某个阈值时抑制
        return False, ""
    
    async def _check_correlation_suppression(self, rule: SuppressionRule, 
                                           alert_data: Dict[str, Any]) -> Tuple[bool, str]:
        """检查关联抑制"""
        # 实现关联抑制逻辑
        # 例如：当存在相关告警时抑制
        return False, ""
    
    async def add_maintenance_window(self, window: MaintenanceWindow) -> bool:
        """添加维护窗口"""
        if window.id in self.maintenance_windows:
            self.logger.warning(f"维护窗口已存在: {window.id}")
            return False
        
        self.maintenance_windows[window.id] = window
        self.logger.info(f"添加维护窗口: {window.id}")
        return True
    
    async def remove_maintenance_window(self, window_id: str) -> bool:
        """删除维护窗口"""
        if window_id not in self.maintenance_windows:
            self.logger.warning(f"维护窗口不存在: {window_id}")
            return False
        
        del self.maintenance_windows[window_id]
        self.logger.info(f"删除维护窗口: {window_id}")
        return True
    
    async def register_parent_alert(self, alert_data: Dict[str, Any]):
        """注册父告警"""
        alert_id = alert_data.get('alert_id', '')
        
        # 检查是否为任何依赖规则的父告警
        for rule in self.rules.values():
            if (rule.suppression_type == SuppressionType.DEPENDENCY and 
                rule.dependency_rule and 
                rule.dependency_rule.is_parent_alert(alert_data)):
                
                self.active_dependencies[alert_id] = datetime.now()
                self.logger.info(f"注册父告警: {alert_id} (规则: {rule.id})")
    
    async def unregister_parent_alert(self, alert_id: str):
        """注销父告警"""
        if alert_id in self.active_dependencies:
            del self.active_dependencies[alert_id]
            self.logger.info(f"注销父告警: {alert_id}")
    
    def get_rule(self, rule_id: str) -> Optional[SuppressionRule]:
        """获取抑制规则"""
        return self.rules.get(rule_id)
    
    def get_rules(self, suppression_type: Optional[SuppressionType] = None,
                  enabled_only: bool = True) -> List[SuppressionRule]:
        """获取抑制规则列表"""
        rules = list(self.rules.values())
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        if suppression_type:
            rules = [r for r in rules if r.suppression_type == suppression_type]
        
        return rules
    
    def get_maintenance_windows(self, active_only: bool = False) -> List[MaintenanceWindow]:
        """获取维护窗口列表"""
        windows = list(self.maintenance_windows.values())
        
        if active_only:
            now = datetime.now()
            windows = [w for w in windows if w.is_active(now)]
        
        return windows
    
    async def get_suppression_statistics(self) -> Dict[str, Any]:
        """获取抑制统计信息"""
        total_rules = len(self.rules)
        enabled_rules = len([r for r in self.rules.values() if r.enabled])
        
        # 按类型统计
        type_stats = {}
        for suppression_type in SuppressionType:
            count = len([r for r in self.rules.values() if r.suppression_type == suppression_type])
            type_stats[suppression_type.value] = count
        
        # 抑制事件统计
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        recent_events = [e for e in self.suppression_events if e.timestamp > last_24h]
        
        # 按类型统计抑制事件
        event_type_stats = {}
        for suppression_type in SuppressionType:
            count = len([e for e in recent_events if e.suppression_type == suppression_type])
            event_type_stats[suppression_type.value] = count
        
        # 活跃抑制统计
        active_suppressions_count = len(self.active_suppressions)
        active_dependencies_count = len(self.active_dependencies)
        active_maintenance_windows = len([w for w in self.maintenance_windows.values() if w.is_active(now)])
        
        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "rule_type_distribution": type_stats,
            "suppression_events_24h": len(recent_events),
            "event_type_distribution_24h": event_type_stats,
            "active_suppressions": active_suppressions_count,
            "active_dependencies": active_dependencies_count,
            "active_maintenance_windows": active_maintenance_windows,
            "total_maintenance_windows": len(self.maintenance_windows)
        }
    
    async def cleanup_old_events(self, retention_days: int = 30):
        """清理过期事件"""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        old_count = len(self.suppression_events)
        self.suppression_events = [
            event for event in self.suppression_events
            if event.timestamp > cutoff_time
        ]
        
        cleaned_count = old_count - len(self.suppression_events)
        
        # 清理过期的活跃抑制
        expired_suppressions = []
        for alert_id, rule_ids in list(self.active_suppressions.items()):
            # 检查是否有相关的活跃事件
            has_active_events = any(
                event.alert_id == alert_id and event.timestamp > cutoff_time
                for event in self.suppression_events
            )
            
            if not has_active_events:
                expired_suppressions.append(alert_id)
        
        for alert_id in expired_suppressions:
            del self.active_suppressions[alert_id]
        
        # 清理过期的依赖关系
        timeout_threshold = datetime.now() - timedelta(hours=24)  # 24小时超时
        expired_dependencies = [
            alert_id for alert_id, timestamp in self.active_dependencies.items()
            if timestamp < timeout_threshold
        ]
        
        for alert_id in expired_dependencies:
            del self.active_dependencies[alert_id]
        
        self.logger.info(f"清理了 {cleaned_count} 个过期抑制事件，{len(expired_suppressions)} 个过期抑制，{len(expired_dependencies)} 个过期依赖")
    
    async def export_rules(self, export_path: str) -> bool:
        """导出抑制规则"""
        try:
            export_data = {
                "suppression_rules": [
                    {
                        "id": rule.id,
                        "name": rule.name,
                        "suppression_type": rule.suppression_type.value,
                        "priority": rule.priority.value,
                        "enabled": rule.enabled,
                        "conditions": [
                            {
                                "field": cond.field,
                                "operator": cond.operator,
                                "value": cond.value,
                                "case_sensitive": cond.case_sensitive
                            }
                            for cond in rule.conditions
                        ],
                        "description": rule.description,
                        "tags": rule.tags,
                        "created_at": rule.created_at.isoformat(),
                        "updated_at": rule.updated_at.isoformat(),
                        "created_by": rule.created_by
                    }
                    for rule in self.rules.values()
                ],
                "maintenance_windows": [
                    {
                        "id": window.id,
                        "name": window.name,
                        "start_time": window.start_time.isoformat(),
                        "end_time": window.end_time.isoformat(),
                        "description": window.description,
                        "affected_services": window.affected_services,
                        "affected_hosts": window.affected_hosts,
                        "created_by": window.created_by
                    }
                    for window in self.maintenance_windows.values()
                ],
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功导出抑制规则到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出抑制规则失败: {e}")
            return False
    
    async def import_rules(self, import_path: str, overwrite: bool = False) -> bool:
        """导入抑制规则"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 导入抑制规则
            rules_data = import_data.get("suppression_rules", [])
            imported_rules = 0
            
            for rule_data in rules_data:
                # 重建条件对象
                conditions = [
                    SuppressionCondition(**cond_data)
                    for cond_data in rule_data.get("conditions", [])
                ]
                
                rule = SuppressionRule(
                    id=rule_data["id"],
                    name=rule_data["name"],
                    suppression_type=SuppressionType(rule_data["suppression_type"]),
                    priority=SuppressionPriority(rule_data["priority"]),
                    enabled=rule_data.get("enabled", True),
                    conditions=conditions,
                    description=rule_data.get("description", ""),
                    tags=rule_data.get("tags", []),
                    created_at=datetime.fromisoformat(rule_data["created_at"]),
                    updated_at=datetime.fromisoformat(rule_data["updated_at"]),
                    created_by=rule_data.get("created_by", "")
                )
                
                if rule.id in self.rules and not overwrite:
                    self.logger.warning(f"抑制规则已存在，跳过: {rule.id}")
                    continue
                
                await self.add_rule(rule)
                imported_rules += 1
            
            # 导入维护窗口
            windows_data = import_data.get("maintenance_windows", [])
            imported_windows = 0
            
            for window_data in windows_data:
                window = MaintenanceWindow(
                    id=window_data["id"],
                    name=window_data["name"],
                    start_time=datetime.fromisoformat(window_data["start_time"]),
                    end_time=datetime.fromisoformat(window_data["end_time"]),
                    description=window_data.get("description", ""),
                    affected_services=window_data.get("affected_services", []),
                    affected_hosts=window_data.get("affected_hosts", []),
                    created_by=window_data.get("created_by", "")
                )
                
                if window.id in self.maintenance_windows and not overwrite:
                    self.logger.warning(f"维护窗口已存在，跳过: {window.id}")
                    continue
                
                await self.add_maintenance_window(window)
                imported_windows += 1
            
            self.logger.info(f"成功导入 {imported_rules} 个抑制规则和 {imported_windows} 个维护窗口")
            return True
            
        except Exception as e:
            self.logger.error(f"导入抑制规则失败: {e}")
            return False