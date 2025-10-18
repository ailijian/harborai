"""
告警抑制规则管理器

负责管理复杂的告警抑制规则，包括时间窗口、依赖关系、模式匹配等。
支持动态抑制规则、智能抑制、抑制规则优先级等高级功能。
"""

import asyncio
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Pattern
from dataclasses import dataclass, asdict
from enum import Enum
import logging


class SuppressionRuleType(Enum):
    """抑制规则类型"""
    TIME_BASED = "time_based"           # 基于时间
    LABEL_BASED = "label_based"         # 基于标签
    PATTERN_BASED = "pattern_based"     # 基于模式
    DEPENDENCY = "dependency"           # 基于依赖
    MAINTENANCE = "maintenance"         # 维护窗口
    RATE_LIMIT = "rate_limit"          # 频率限制
    CONDITIONAL = "conditional"         # 条件抑制
    CASCADING = "cascading"            # 级联抑制


class SuppressionPriority(Enum):
    """抑制优先级"""
    CRITICAL = "critical"    # 关键抑制（最高优先级）
    HIGH = "high"           # 高优先级
    MEDIUM = "medium"       # 中等优先级
    LOW = "low"            # 低优先级


class SuppressionAction(Enum):
    """抑制动作"""
    SUPPRESS = "suppress"           # 完全抑制
    DELAY = "delay"                # 延迟发送
    REDUCE_SEVERITY = "reduce_severity"  # 降低严重程度
    AGGREGATE = "aggregate"         # 聚合告警
    REDIRECT = "redirect"          # 重定向通知


@dataclass
class TimeWindow:
    """时间窗口"""
    start_time: str  # HH:MM 格式
    end_time: str    # HH:MM 格式
    weekdays: List[int] = None  # 0-6, 0为周一
    timezone: str = "UTC"
    
    def __post_init__(self):
        if self.weekdays is None:
            self.weekdays = list(range(7))  # 默认所有工作日
    
    def is_in_window(self, dt: datetime) -> bool:
        """检查时间是否在窗口内"""
        if dt.weekday() not in self.weekdays:
            return False
        
        current_time = dt.time()
        start = datetime.strptime(self.start_time, "%H:%M").time()
        end = datetime.strptime(self.end_time, "%H:%M").time()
        
        if start <= end:
            return start <= current_time <= end
        else:
            # 跨天的情况
            return current_time >= start or current_time <= end


@dataclass
class SuppressionCondition:
    """抑制条件"""
    field: str                    # 字段名
    operator: str                 # 操作符: eq, ne, gt, lt, gte, lte, in, not_in, regex, exists
    value: Any                   # 比较值
    case_sensitive: bool = True  # 是否区分大小写
    
    def matches(self, data: Dict[str, Any]) -> bool:
        """检查条件是否匹配"""
        field_value = self._get_nested_value(data, self.field)
        
        if self.operator == "exists":
            return field_value is not None
        
        if field_value is None:
            return False
        
        if self.operator == "eq":
            return self._compare_values(field_value, self.value, "eq")
        elif self.operator == "ne":
            return self._compare_values(field_value, self.value, "ne")
        elif self.operator == "gt":
            return field_value > self.value
        elif self.operator == "lt":
            return field_value < self.value
        elif self.operator == "gte":
            return field_value >= self.value
        elif self.operator == "lte":
            return field_value <= self.value
        elif self.operator == "in":
            return field_value in self.value
        elif self.operator == "not_in":
            return field_value not in self.value
        elif self.operator == "regex":
            pattern = re.compile(self.value, 0 if self.case_sensitive else re.IGNORECASE)
            return bool(pattern.search(str(field_value)))
        
        return False
    
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
    
    def _compare_values(self, field_value: Any, compare_value: Any, operator: str) -> bool:
        """比较值"""
        if not self.case_sensitive and isinstance(field_value, str) and isinstance(compare_value, str):
            field_value = field_value.lower()
            compare_value = compare_value.lower()
        
        if operator == "eq":
            return field_value == compare_value
        elif operator == "ne":
            return field_value != compare_value
        
        return False


@dataclass
class SuppressionRule:
    """抑制规则"""
    id: str
    name: str
    rule_type: SuppressionRuleType
    priority: SuppressionPriority
    action: SuppressionAction
    conditions: List[SuppressionCondition]
    time_windows: List[TimeWindow] = None
    duration: Optional[int] = None  # 抑制持续时间（秒）
    max_suppressions: Optional[int] = None  # 最大抑制次数
    dependencies: List[str] = None  # 依赖的其他规则
    config: Dict[str, Any] = None
    enabled: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.time_windows is None:
            self.time_windows = []
        if self.dependencies is None:
            self.dependencies = []
        if self.config is None:
            self.config = {}
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'rule_type': self.rule_type.value,
            'priority': self.priority.value,
            'action': self.action.value,
            'conditions': [asdict(c) for c in self.conditions],
            'time_windows': [asdict(tw) for tw in self.time_windows],
            'duration': self.duration,
            'max_suppressions': self.max_suppressions,
            'dependencies': self.dependencies,
            'config': self.config,
            'enabled': self.enabled,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SuppressionRule':
        """从字典创建"""
        conditions = [
            SuppressionCondition(**c) for c in data.get('conditions', [])
        ]
        time_windows = [
            TimeWindow(**tw) for tw in data.get('time_windows', [])
        ]
        
        return cls(
            id=data['id'],
            name=data['name'],
            rule_type=SuppressionRuleType(data['rule_type']),
            priority=SuppressionPriority(data['priority']),
            action=SuppressionAction(data['action']),
            conditions=conditions,
            time_windows=time_windows,
            duration=data.get('duration'),
            max_suppressions=data.get('max_suppressions'),
            dependencies=data.get('dependencies', []),
            config=data.get('config', {}),
            enabled=data.get('enabled', True),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
        )


@dataclass
class SuppressionEvent:
    """抑制事件"""
    rule_id: str
    alert_id: str
    action: SuppressionAction
    timestamp: datetime
    reason: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SuppressionRuleManager:
    """抑制规则管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules: Dict[str, SuppressionRule] = {}
        self.suppression_events: List[SuppressionEvent] = []
        self.suppression_counters: Dict[str, int] = {}  # 规则ID -> 抑制次数
        self.active_suppressions: Dict[str, datetime] = {}  # 告警ID -> 抑制结束时间
        
        # 规则依赖图
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # 加载默认规则
        self._load_default_rules()
    
    def _load_default_rules(self):
        """加载默认抑制规则"""
        default_rules = [
            # 维护窗口抑制
            SuppressionRule(
                id="maintenance_window",
                name="维护窗口抑制",
                rule_type=SuppressionRuleType.MAINTENANCE,
                priority=SuppressionPriority.CRITICAL,
                action=SuppressionAction.SUPPRESS,
                conditions=[
                    SuppressionCondition("labels.maintenance", "eq", "true")
                ],
                time_windows=[
                    TimeWindow("02:00", "04:00", [0, 1, 2, 3, 4])  # 工作日凌晨2-4点
                ]
            ),
            
            # 测试环境抑制
            SuppressionRule(
                id="test_environment",
                name="测试环境抑制",
                rule_type=SuppressionRuleType.LABEL_BASED,
                priority=SuppressionPriority.HIGH,
                action=SuppressionAction.SUPPRESS,
                conditions=[
                    SuppressionCondition("labels.environment", "in", ["test", "dev", "staging"])
                ]
            ),
            
            # 频率限制抑制
            SuppressionRule(
                id="rate_limit_default",
                name="默认频率限制",
                rule_type=SuppressionRuleType.RATE_LIMIT,
                priority=SuppressionPriority.MEDIUM,
                action=SuppressionAction.AGGREGATE,
                conditions=[],
                config={
                    "max_alerts": 10,
                    "window_seconds": 300,
                    "group_by": ["rule_id", "severity"]
                }
            ),
            
            # 依赖关系抑制
            SuppressionRule(
                id="database_dependency",
                name="数据库依赖抑制",
                rule_type=SuppressionRuleType.DEPENDENCY,
                priority=SuppressionPriority.HIGH,
                action=SuppressionAction.SUPPRESS,
                conditions=[
                    SuppressionCondition("labels.component", "eq", "api")
                ],
                dependencies=["database_connection_failure"]
            ),
            
            # 低优先级告警延迟
            SuppressionRule(
                id="low_priority_delay",
                name="低优先级告警延迟",
                rule_type=SuppressionRuleType.CONDITIONAL,
                priority=SuppressionPriority.LOW,
                action=SuppressionAction.DELAY,
                conditions=[
                    SuppressionCondition("severity", "in", ["low", "info"])
                ],
                config={
                    "delay_seconds": 300
                }
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
        
        self._build_dependency_graph()
    
    def _build_dependency_graph(self):
        """构建依赖关系图"""
        self.dependency_graph.clear()
        
        for rule_id, rule in self.rules.items():
            self.dependency_graph[rule_id] = set(rule.dependencies)
    
    async def add_rule(self, rule: SuppressionRule) -> bool:
        """添加抑制规则"""
        if rule.id in self.rules:
            self.logger.warning(f"抑制规则已存在: {rule.id}")
            return False
        
        # 验证依赖关系
        for dep in rule.dependencies:
            if dep not in self.rules:
                self.logger.error(f"依赖的规则不存在: {dep}")
                return False
        
        self.rules[rule.id] = rule
        self.dependency_graph[rule.id] = set(rule.dependencies)
        
        self.logger.info(f"添加抑制规则: {rule.id}")
        return True
    
    async def update_rule(self, rule_id: str, rule: SuppressionRule) -> bool:
        """更新抑制规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"抑制规则不存在: {rule_id}")
            return False
        
        # 验证依赖关系
        for dep in rule.dependencies:
            if dep not in self.rules and dep != rule_id:
                self.logger.error(f"依赖的规则不存在: {dep}")
                return False
        
        rule.updated_at = datetime.now()
        self.rules[rule_id] = rule
        self.dependency_graph[rule_id] = set(rule.dependencies)
        
        self.logger.info(f"更新抑制规则: {rule_id}")
        return True
    
    async def remove_rule(self, rule_id: str) -> bool:
        """删除抑制规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"抑制规则不存在: {rule_id}")
            return False
        
        # 检查是否被其他规则依赖
        dependent_rules = []
        for rid, deps in self.dependency_graph.items():
            if rule_id in deps:
                dependent_rules.append(rid)
        
        if dependent_rules:
            self.logger.error(f"规则 {rule_id} 被其他规则依赖: {dependent_rules}")
            return False
        
        del self.rules[rule_id]
        if rule_id in self.dependency_graph:
            del self.dependency_graph[rule_id]
        
        self.logger.info(f"删除抑制规则: {rule_id}")
        return True
    
    async def check_suppression(self, alert_data: Dict[str, Any]) -> Tuple[bool, List[SuppressionEvent]]:
        """检查告警是否应该被抑制"""
        suppression_events = []
        should_suppress = False
        
        # 按优先级排序规则
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: (r.priority.value, r.id)
        )
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            # 检查规则是否匹配
            if await self._rule_matches(rule, alert_data):
                event = SuppressionEvent(
                    rule_id=rule.id,
                    alert_id=alert_data.get('id', 'unknown'),
                    action=rule.action,
                    timestamp=datetime.now(),
                    reason=f"匹配抑制规则: {rule.name}",
                    metadata={
                        'rule_type': rule.rule_type.value,
                        'priority': rule.priority.value
                    }
                )
                
                suppression_events.append(event)
                
                # 根据动作类型决定是否继续检查
                if rule.action == SuppressionAction.SUPPRESS:
                    should_suppress = True
                    break  # 完全抑制，不再检查其他规则
                elif rule.action in [SuppressionAction.DELAY, SuppressionAction.REDUCE_SEVERITY]:
                    # 这些动作不阻止后续规则检查
                    continue
        
        # 记录抑制事件
        self.suppression_events.extend(suppression_events)
        
        return should_suppress, suppression_events
    
    async def _rule_matches(self, rule: SuppressionRule, alert_data: Dict[str, Any]) -> bool:
        """检查规则是否匹配告警"""
        # 检查时间窗口
        if rule.time_windows:
            now = datetime.now()
            in_time_window = any(tw.is_in_window(now) for tw in rule.time_windows)
            if not in_time_window:
                return False
        
        # 检查条件
        if rule.conditions:
            conditions_met = all(condition.matches(alert_data) for condition in rule.conditions)
            if not conditions_met:
                return False
        
        # 检查依赖关系
        if rule.dependencies:
            dependencies_met = await self._check_dependencies(rule, alert_data)
            if not dependencies_met:
                return False
        
        # 检查抑制次数限制
        if rule.max_suppressions:
            current_count = self.suppression_counters.get(rule.id, 0)
            if current_count >= rule.max_suppressions:
                return False
        
        # 检查特定规则类型的逻辑
        if rule.rule_type == SuppressionRuleType.RATE_LIMIT:
            return await self._check_rate_limit(rule, alert_data)
        elif rule.rule_type == SuppressionRuleType.CASCADING:
            return await self._check_cascading(rule, alert_data)
        
        return True
    
    async def _check_dependencies(self, rule: SuppressionRule, alert_data: Dict[str, Any]) -> bool:
        """检查依赖关系"""
        for dep_rule_id in rule.dependencies:
            # 检查依赖的规则是否已经触发
            dep_triggered = any(
                event.rule_id == dep_rule_id and 
                event.timestamp > datetime.now() - timedelta(hours=1)
                for event in self.suppression_events
            )
            
            if not dep_triggered:
                return False
        
        return True
    
    async def _check_rate_limit(self, rule: SuppressionRule, alert_data: Dict[str, Any]) -> bool:
        """检查频率限制"""
        config = rule.config
        max_alerts = config.get('max_alerts', 10)
        window_seconds = config.get('window_seconds', 300)
        group_by = config.get('group_by', [])
        
        # 构建分组键
        group_key = self._build_group_key(alert_data, group_by)
        
        # 计算时间窗口内的告警数量
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_alerts = [
            event for event in self.suppression_events
            if (event.timestamp > cutoff_time and 
                event.metadata.get('group_key') == group_key)
        ]
        
        return len(recent_alerts) >= max_alerts
    
    async def _check_cascading(self, rule: SuppressionRule, alert_data: Dict[str, Any]) -> bool:
        """检查级联抑制"""
        # 级联抑制逻辑：如果上游组件有告警，则抑制下游组件告警
        upstream_components = rule.config.get('upstream_components', [])
        current_component = alert_data.get('labels', {}).get('component')
        
        if current_component in upstream_components:
            return False  # 上游组件不抑制
        
        # 检查是否有上游组件告警
        cutoff_time = datetime.now() - timedelta(minutes=30)
        upstream_alerts = [
            event for event in self.suppression_events
            if (event.timestamp > cutoff_time and
                event.metadata.get('component') in upstream_components)
        ]
        
        return len(upstream_alerts) > 0
    
    def _build_group_key(self, alert_data: Dict[str, Any], group_by: List[str]) -> str:
        """构建分组键"""
        key_parts = []
        for field in group_by:
            value = self._get_nested_value(alert_data, field)
            key_parts.append(f"{field}:{value}")
        return "|".join(key_parts)
    
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
    
    async def record_suppression(self, rule_id: str, alert_id: str, action: SuppressionAction):
        """记录抑制事件"""
        # 更新抑制计数器
        self.suppression_counters[rule_id] = self.suppression_counters.get(rule_id, 0) + 1
        
        # 如果是延迟动作，记录抑制结束时间
        if action == SuppressionAction.DELAY:
            rule = self.rules.get(rule_id)
            if rule and rule.config.get('delay_seconds'):
                end_time = datetime.now() + timedelta(seconds=rule.config['delay_seconds'])
                self.active_suppressions[alert_id] = end_time
    
    def get_rule(self, rule_id: str) -> Optional[SuppressionRule]:
        """获取抑制规则"""
        return self.rules.get(rule_id)
    
    def get_rules(self, rule_type: Optional[SuppressionRuleType] = None,
                 enabled_only: bool = True) -> List[SuppressionRule]:
        """获取抑制规则列表"""
        rules = list(self.rules.values())
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        if rule_type:
            rules = [r for r in rules if r.rule_type == rule_type]
        
        return rules
    
    async def get_suppression_statistics(self) -> Dict[str, Any]:
        """获取抑制统计信息"""
        total_rules = len(self.rules)
        enabled_rules = len([r for r in self.rules.values() if r.enabled])
        
        # 按类型统计
        type_stats = {}
        for rule_type in SuppressionRuleType:
            count = len([r for r in self.rules.values() if r.rule_type == rule_type])
            type_stats[rule_type.value] = count
        
        # 按优先级统计
        priority_stats = {}
        for priority in SuppressionPriority:
            count = len([r for r in self.rules.values() if r.priority == priority])
            priority_stats[priority.value] = count
        
        # 按动作统计
        action_stats = {}
        for action in SuppressionAction:
            count = len([r for r in self.rules.values() if r.action == action])
            action_stats[action.value] = count
        
        # 抑制事件统计
        recent_events = [
            event for event in self.suppression_events
            if event.timestamp > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "type_distribution": type_stats,
            "priority_distribution": priority_stats,
            "action_distribution": action_stats,
            "total_suppression_events": len(self.suppression_events),
            "recent_suppression_events": len(recent_events),
            "active_suppressions": len(self.active_suppressions),
            "suppression_counters": dict(self.suppression_counters)
        }
    
    async def export_rules(self, export_path: str) -> bool:
        """导出抑制规则"""
        try:
            export_data = {
                "suppression_rules": [rule.to_dict() for rule in self.rules.values()],
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
            
            rules_data = import_data.get("suppression_rules", [])
            imported_count = 0
            
            for rule_data in rules_data:
                rule = SuppressionRule.from_dict(rule_data)
                
                if rule.id in self.rules and not overwrite:
                    self.logger.warning(f"抑制规则已存在，跳过: {rule.id}")
                    continue
                
                await self.add_rule(rule)
                imported_count += 1
            
            self.logger.info(f"成功导入 {imported_count} 个抑制规则")
            return True
            
        except Exception as e:
            self.logger.error(f"导入抑制规则失败: {e}")
            return False
    
    async def cleanup_old_events(self, retention_days: int = 30):
        """清理过期的抑制事件"""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        old_count = len(self.suppression_events)
        self.suppression_events = [
            event for event in self.suppression_events
            if event.timestamp > cutoff_time
        ]
        
        # 清理过期的活跃抑制
        expired_suppressions = [
            alert_id for alert_id, end_time in self.active_suppressions.items()
            if end_time < datetime.now()
        ]
        
        for alert_id in expired_suppressions:
            del self.active_suppressions[alert_id]
        
        cleaned_events = old_count - len(self.suppression_events)
        cleaned_suppressions = len(expired_suppressions)
        
        self.logger.info(f"清理了 {cleaned_events} 个过期抑制事件，{cleaned_suppressions} 个过期抑制")
    
    async def validate_rule(self, rule: SuppressionRule) -> List[str]:
        """验证抑制规则"""
        errors = []
        
        # 基本验证
        if not rule.id:
            errors.append("规则ID不能为空")
        
        if not rule.name:
            errors.append("规则名称不能为空")
        
        # 验证时间窗口
        for tw in rule.time_windows:
            try:
                datetime.strptime(tw.start_time, "%H:%M")
                datetime.strptime(tw.end_time, "%H:%M")
            except ValueError:
                errors.append(f"时间格式错误: {tw.start_time} - {tw.end_time}")
        
        # 验证条件
        for condition in rule.conditions:
            if condition.operator not in ["eq", "ne", "gt", "lt", "gte", "lte", "in", "not_in", "regex", "exists"]:
                errors.append(f"不支持的操作符: {condition.operator}")
        
        # 验证依赖关系
        for dep in rule.dependencies:
            if dep not in self.rules and dep != rule.id:
                errors.append(f"依赖的规则不存在: {dep}")
        
        # 验证配置
        if rule.rule_type == SuppressionRuleType.RATE_LIMIT:
            if 'max_alerts' not in rule.config or 'window_seconds' not in rule.config:
                errors.append("频率限制规则缺少必要配置: max_alerts, window_seconds")
        
        return errors