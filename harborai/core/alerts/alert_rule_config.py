"""
告警规则配置管理器

负责管理告警规则的配置、验证、加载和持久化，
支持动态配置更新、规则模板和配置验证。
"""

import json
import yaml
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import jsonschema
from copy import deepcopy


class RuleType(Enum):
    """规则类型"""
    THRESHOLD = "threshold"        # 阈值规则
    ANOMALY = "anomaly"           # 异常检测规则
    PATTERN = "pattern"           # 模式匹配规则
    COMPOSITE = "composite"       # 复合规则
    CORRELATION = "correlation"   # 关联规则
    CUSTOM = "custom"             # 自定义规则


class Severity(Enum):
    """严重级别"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RuleStatus(Enum):
    """规则状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISABLED = "disabled"
    TESTING = "testing"


class ComparisonOperator(Enum):
    """比较操作符"""
    GT = "gt"          # 大于
    GTE = "gte"        # 大于等于
    LT = "lt"          # 小于
    LTE = "lte"        # 小于等于
    EQ = "eq"          # 等于
    NE = "ne"          # 不等于
    IN = "in"          # 包含
    NOT_IN = "not_in"  # 不包含
    REGEX = "regex"    # 正则匹配
    CONTAINS = "contains"  # 字符串包含


@dataclass
class ThresholdCondition:
    """阈值条件"""
    metric: str                           # 指标名称
    operator: ComparisonOperator          # 比较操作符
    value: Union[int, float, str, List]   # 阈值
    duration: int = 60                    # 持续时间（秒）
    aggregation: str = "avg"              # 聚合方式 (avg, sum, max, min, count)
    
    def evaluate(self, metric_value: Union[int, float], 
                 duration_met: bool = True) -> bool:
        """评估条件是否满足"""
        if not duration_met:
            return False
        
        if self.operator == ComparisonOperator.GT:
            return metric_value > self.value
        elif self.operator == ComparisonOperator.GTE:
            return metric_value >= self.value
        elif self.operator == ComparisonOperator.LT:
            return metric_value < self.value
        elif self.operator == ComparisonOperator.LTE:
            return metric_value <= self.value
        elif self.operator == ComparisonOperator.EQ:
            return metric_value == self.value
        elif self.operator == ComparisonOperator.NE:
            return metric_value != self.value
        elif self.operator == ComparisonOperator.IN:
            return metric_value in self.value
        elif self.operator == ComparisonOperator.NOT_IN:
            return metric_value not in self.value
        elif self.operator == ComparisonOperator.REGEX:
            import re
            return bool(re.search(str(self.value), str(metric_value)))
        elif self.operator == ComparisonOperator.CONTAINS:
            return str(self.value) in str(metric_value)
        
        return False


@dataclass
class AnomalyDetectionConfig:
    """异常检测配置"""
    algorithm: str = "statistical"       # 算法类型: statistical, ml, isolation_forest
    sensitivity: float = 0.95           # 敏感度 (0-1)
    window_size: int = 100              # 窗口大小
    min_samples: int = 50               # 最小样本数
    threshold_factor: float = 2.0       # 阈值因子
    
    # 统计方法参数
    std_dev_threshold: float = 2.0      # 标准差阈值
    percentile_threshold: float = 95.0  # 百分位阈值
    
    # 机器学习参数
    model_type: str = "isolation_forest"  # 模型类型
    contamination: float = 0.1          # 污染率
    
    def detect_anomaly(self, values: List[float], 
                      current_value: float) -> Tuple[bool, float]:
        """检测异常"""
        if len(values) < self.min_samples:
            return False, 0.0
        
        if self.algorithm == "statistical":
            return self._statistical_detection(values, current_value)
        elif self.algorithm == "percentile":
            return self._percentile_detection(values, current_value)
        else:
            # 其他算法的占位符
            return False, 0.0
    
    def _statistical_detection(self, values: List[float], 
                             current_value: float) -> Tuple[bool, float]:
        """统计方法异常检测"""
        import statistics
        
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values) if len(values) > 1 else 0
        
        if std_dev == 0:
            return False, 0.0
        
        z_score = abs(current_value - mean) / std_dev
        is_anomaly = z_score > self.std_dev_threshold
        
        return is_anomaly, z_score
    
    def _percentile_detection(self, values: List[float], 
                            current_value: float) -> Tuple[bool, float]:
        """百分位方法异常检测"""
        import statistics
        
        sorted_values = sorted(values)
        percentile_index = int(len(sorted_values) * self.percentile_threshold / 100)
        threshold = sorted_values[min(percentile_index, len(sorted_values) - 1)]
        
        is_anomaly = current_value > threshold
        score = current_value / threshold if threshold > 0 else 0
        
        return is_anomaly, score


@dataclass
class PatternMatchConfig:
    """模式匹配配置"""
    patterns: List[str]                 # 匹配模式列表
    match_type: str = "any"             # 匹配类型: any, all
    case_sensitive: bool = False        # 是否区分大小写
    regex_enabled: bool = True          # 是否启用正则表达式
    
    def matches(self, text: str) -> bool:
        """检查文本是否匹配模式"""
        if not self.patterns:
            return False
        
        matches = []
        for pattern in self.patterns:
            if self.regex_enabled:
                import re
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
class CompositeRuleConfig:
    """复合规则配置"""
    sub_rules: List[str]                # 子规则ID列表
    logic_operator: str = "AND"         # 逻辑操作符: AND, OR, NOT
    evaluation_window: int = 300        # 评估窗口（秒）
    
    def evaluate(self, sub_rule_results: Dict[str, bool]) -> bool:
        """评估复合规则"""
        if not self.sub_rules:
            return False
        
        results = [sub_rule_results.get(rule_id, False) for rule_id in self.sub_rules]
        
        if self.logic_operator == "AND":
            return all(results)
        elif self.logic_operator == "OR":
            return any(results)
        elif self.logic_operator == "NOT":
            return not any(results)
        
        return False


@dataclass
class CorrelationRuleConfig:
    """关联规则配置"""
    correlation_window: int = 300       # 关联窗口（秒）
    correlation_threshold: float = 0.8  # 关联阈值
    correlation_type: str = "temporal"  # 关联类型: temporal, causal, statistical
    related_metrics: List[str] = field(default_factory=list)  # 相关指标
    
    def calculate_correlation(self, metric_data: Dict[str, List[float]]) -> float:
        """计算关联度"""
        if len(self.related_metrics) < 2:
            return 0.0
        
        # 简单的皮尔逊相关系数计算
        try:
            import statistics
            
            metric1_data = metric_data.get(self.related_metrics[0], [])
            metric2_data = metric_data.get(self.related_metrics[1], [])
            
            if len(metric1_data) != len(metric2_data) or len(metric1_data) < 2:
                return 0.0
            
            # 计算皮尔逊相关系数
            mean1 = statistics.mean(metric1_data)
            mean2 = statistics.mean(metric2_data)
            
            numerator = sum((x - mean1) * (y - mean2) for x, y in zip(metric1_data, metric2_data))
            
            sum_sq1 = sum((x - mean1) ** 2 for x in metric1_data)
            sum_sq2 = sum((y - mean2) ** 2 for y in metric2_data)
            
            denominator = (sum_sq1 * sum_sq2) ** 0.5
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return abs(correlation)
            
        except Exception:
            return 0.0


@dataclass
class AlertRule:
    """告警规则"""
    id: str
    name: str
    description: str
    rule_type: RuleType
    severity: Severity
    status: RuleStatus = RuleStatus.ACTIVE
    
    # 条件配置
    threshold_conditions: List[ThresholdCondition] = field(default_factory=list)
    anomaly_config: Optional[AnomalyDetectionConfig] = None
    pattern_config: Optional[PatternMatchConfig] = None
    composite_config: Optional[CompositeRuleConfig] = None
    correlation_config: Optional[CorrelationRuleConfig] = None
    
    # 触发配置
    evaluation_interval: int = 60       # 评估间隔（秒）
    cooldown_period: int = 300          # 冷却期（秒）
    max_alerts_per_hour: int = 10       # 每小时最大告警数
    
    # 标签和分组
    tags: List[str] = field(default_factory=list)
    group: str = "default"
    
    # 通知配置
    notification_channels: List[str] = field(default_factory=list)
    escalation_rules: List[str] = field(default_factory=list)
    
    # 时间配置
    active_hours: Optional[Dict[str, Any]] = None  # 活跃时间
    timezone: str = "UTC"
    
    # 元数据
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = ""
    version: int = 1
    
    # 统计信息
    triggered_count: int = 0
    last_triggered: Optional[datetime] = None
    last_evaluated: Optional[datetime] = None
    
    def is_active(self, timestamp: datetime = None) -> bool:
        """检查规则是否在活跃时间内"""
        if self.status != RuleStatus.ACTIVE:
            return False
        
        if not self.active_hours:
            return True
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # 检查时间范围
        current_hour = timestamp.hour
        current_weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # 检查工作日
        if "weekdays" in self.active_hours:
            weekday_hours = self.active_hours["weekdays"]
            if (current_weekday < 5 and  # Monday-Friday
                weekday_hours.get("start", 0) <= current_hour <= weekday_hours.get("end", 23)):
                return True
        
        # 检查周末
        if "weekends" in self.active_hours:
            weekend_hours = self.active_hours["weekends"]
            if (current_weekday >= 5 and  # Saturday-Sunday
                weekend_hours.get("start", 0) <= current_hour <= weekend_hours.get("end", 23)):
                return True
        
        # 检查每日时间
        if "daily" in self.active_hours:
            daily_hours = self.active_hours["daily"]
            if daily_hours.get("start", 0) <= current_hour <= daily_hours.get("end", 23):
                return True
        
        return False
    
    def can_trigger(self, timestamp: datetime = None) -> bool:
        """检查是否可以触发告警"""
        if not self.is_active(timestamp):
            return False
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # 检查冷却期
        if self.last_triggered:
            cooldown_end = self.last_triggered + timedelta(seconds=self.cooldown_period)
            if timestamp < cooldown_end:
                return False
        
        # 检查每小时限制
        if self.last_triggered:
            hour_ago = timestamp - timedelta(hours=1)
            if self.last_triggered > hour_ago and self.triggered_count >= self.max_alerts_per_hour:
                return False
        
        return True
    
    def evaluate(self, metric_data: Dict[str, Any], 
                 historical_data: Dict[str, List[float]] = None) -> Tuple[bool, str]:
        """评估规则是否触发"""
        if not self.can_trigger():
            return False, "规则不在活跃状态或在冷却期内"
        
        self.last_evaluated = datetime.now()
        
        try:
            if self.rule_type == RuleType.THRESHOLD:
                return self._evaluate_threshold(metric_data)
            
            elif self.rule_type == RuleType.ANOMALY:
                return self._evaluate_anomaly(metric_data, historical_data)
            
            elif self.rule_type == RuleType.PATTERN:
                return self._evaluate_pattern(metric_data)
            
            elif self.rule_type == RuleType.COMPOSITE:
                return self._evaluate_composite(metric_data)
            
            elif self.rule_type == RuleType.CORRELATION:
                return self._evaluate_correlation(metric_data, historical_data)
            
            else:
                return False, f"不支持的规则类型: {self.rule_type.value}"
        
        except Exception as e:
            return False, f"规则评估出错: {str(e)}"
    
    def _evaluate_threshold(self, metric_data: Dict[str, Any]) -> Tuple[bool, str]:
        """评估阈值规则"""
        if not self.threshold_conditions:
            return False, "没有配置阈值条件"
        
        triggered_conditions = []
        
        for condition in self.threshold_conditions:
            metric_value = metric_data.get(condition.metric)
            if metric_value is None:
                continue
            
            # 简化的持续时间检查（实际应该基于历史数据）
            duration_met = True  # 这里应该实现真正的持续时间检查
            
            if condition.evaluate(metric_value, duration_met):
                triggered_conditions.append(f"{condition.metric} {condition.operator.value} {condition.value}")
        
        if triggered_conditions:
            return True, f"阈值条件触发: {', '.join(triggered_conditions)}"
        
        return False, "阈值条件未满足"
    
    def _evaluate_anomaly(self, metric_data: Dict[str, Any], 
                         historical_data: Dict[str, List[float]]) -> Tuple[bool, str]:
        """评估异常检测规则"""
        if not self.anomaly_config:
            return False, "没有配置异常检测"
        
        if not historical_data:
            return False, "缺少历史数据"
        
        anomalies = []
        
        for metric, current_value in metric_data.items():
            if metric in historical_data:
                is_anomaly, score = self.anomaly_config.detect_anomaly(
                    historical_data[metric], current_value
                )
                
                if is_anomaly:
                    anomalies.append(f"{metric} (异常分数: {score:.2f})")
        
        if anomalies:
            return True, f"检测到异常: {', '.join(anomalies)}"
        
        return False, "未检测到异常"
    
    def _evaluate_pattern(self, metric_data: Dict[str, Any]) -> Tuple[bool, str]:
        """评估模式匹配规则"""
        if not self.pattern_config:
            return False, "没有配置模式匹配"
        
        # 将所有指标值转换为字符串进行模式匹配
        text_data = " ".join(str(value) for value in metric_data.values())
        
        if self.pattern_config.matches(text_data):
            return True, f"模式匹配成功: {self.pattern_config.patterns}"
        
        return False, "模式匹配失败"
    
    def _evaluate_composite(self, metric_data: Dict[str, Any]) -> Tuple[bool, str]:
        """评估复合规则"""
        if not self.composite_config:
            return False, "没有配置复合规则"
        
        # 这里应该评估子规则，简化实现
        sub_rule_results = {}
        for rule_id in self.composite_config.sub_rules:
            # 假设子规则结果（实际应该递归评估）
            sub_rule_results[rule_id] = True
        
        if self.composite_config.evaluate(sub_rule_results):
            return True, f"复合规则触发: {self.composite_config.logic_operator}"
        
        return False, "复合规则条件未满足"
    
    def _evaluate_correlation(self, metric_data: Dict[str, Any], 
                            historical_data: Dict[str, List[float]]) -> Tuple[bool, str]:
        """评估关联规则"""
        if not self.correlation_config:
            return False, "没有配置关联规则"
        
        if not historical_data:
            return False, "缺少历史数据"
        
        correlation = self.correlation_config.calculate_correlation(historical_data)
        
        if correlation >= self.correlation_config.correlation_threshold:
            return True, f"关联度超过阈值: {correlation:.2f}"
        
        return False, f"关联度未达到阈值: {correlation:.2f}"
    
    def trigger(self, reason: str = ""):
        """触发告警"""
        self.triggered_count += 1
        self.last_triggered = datetime.now()
        
        # 这里可以添加触发后的处理逻辑
        logging.info(f"告警规则触发: {self.id} - {reason}")


@dataclass
class RuleTemplate:
    """规则模板"""
    id: str
    name: str
    description: str
    category: str
    rule_type: RuleType
    template_config: Dict[str, Any]
    variables: List[str] = field(default_factory=list)
    
    def create_rule(self, rule_id: str, rule_name: str, 
                   variables: Dict[str, Any]) -> AlertRule:
        """从模板创建规则"""
        config = deepcopy(self.template_config)
        
        # 替换变量
        config_str = json.dumps(config)
        for var, value in variables.items():
            config_str = config_str.replace(f"{{{var}}}", str(value))
        
        config = json.loads(config_str)
        
        # 创建规则对象
        rule = AlertRule(
            id=rule_id,
            name=rule_name,
            description=config.get("description", ""),
            rule_type=self.rule_type,
            severity=Severity(config.get("severity", "medium")),
            **config
        )
        
        return rule


class AlertRuleConfigManager:
    """告警规则配置管理器"""
    
    def __init__(self, config_dir: str = "config/alerts"):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 规则存储
        self.rules: Dict[str, AlertRule] = {}
        self.templates: Dict[str, RuleTemplate] = {}
        
        # 配置文件路径
        self.rules_file = self.config_dir / "rules.json"
        self.templates_file = self.config_dir / "templates.json"
        self.schema_file = self.config_dir / "rule_schema.json"
        
        # 加载配置
        self._load_schema()
        self._load_default_templates()
        self._load_rules()
    
    def _load_schema(self):
        """加载规则验证模式"""
        self.rule_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "name": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "rule_type": {"type": "string", "enum": [t.value for t in RuleType]},
                "severity": {"type": "string", "enum": [s.value for s in Severity]},
                "status": {"type": "string", "enum": [s.value for s in RuleStatus]},
                "evaluation_interval": {"type": "integer", "minimum": 1},
                "cooldown_period": {"type": "integer", "minimum": 0},
                "max_alerts_per_hour": {"type": "integer", "minimum": 1},
                "tags": {"type": "array", "items": {"type": "string"}},
                "group": {"type": "string"},
                "notification_channels": {"type": "array", "items": {"type": "string"}},
                "threshold_conditions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string", "minLength": 1},
                            "operator": {"type": "string", "enum": [op.value for op in ComparisonOperator]},
                            "value": {"oneOf": [
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "array"}
                            ]},
                            "duration": {"type": "integer", "minimum": 1},
                            "aggregation": {"type": "string"}
                        },
                        "required": ["metric", "operator", "value"]
                    }
                }
            },
            "required": ["id", "name", "rule_type", "severity"]
        }
    
    def _load_default_templates(self):
        """加载默认模板"""
        default_templates = [
            RuleTemplate(
                id="high_cpu_usage",
                name="高CPU使用率模板",
                description="监控CPU使用率超过阈值",
                category="system",
                rule_type=RuleType.THRESHOLD,
                template_config={
                    "description": "CPU使用率超过{cpu_threshold}%",
                    "severity": "high",
                    "threshold_conditions": [
                        {
                            "metric": "cpu_usage_percent",
                            "operator": "gt",
                            "value": "{cpu_threshold}",
                            "duration": "{duration}",
                            "aggregation": "avg"
                        }
                    ],
                    "evaluation_interval": 60,
                    "cooldown_period": 300
                },
                variables=["cpu_threshold", "duration"]
            ),
            
            RuleTemplate(
                id="high_memory_usage",
                name="高内存使用率模板",
                description="监控内存使用率超过阈值",
                category="system",
                rule_type=RuleType.THRESHOLD,
                template_config={
                    "description": "内存使用率超过{memory_threshold}%",
                    "severity": "high",
                    "threshold_conditions": [
                        {
                            "metric": "memory_usage_percent",
                            "operator": "gt",
                            "value": "{memory_threshold}",
                            "duration": "{duration}",
                            "aggregation": "avg"
                        }
                    ],
                    "evaluation_interval": 60,
                    "cooldown_period": 300
                },
                variables=["memory_threshold", "duration"]
            ),
            
            RuleTemplate(
                id="api_response_time",
                name="API响应时间模板",
                description="监控API响应时间超过阈值",
                category="application",
                rule_type=RuleType.THRESHOLD,
                template_config={
                    "description": "API响应时间超过{response_time_threshold}ms",
                    "severity": "medium",
                    "threshold_conditions": [
                        {
                            "metric": "api_response_time_ms",
                            "operator": "gt",
                            "value": "{response_time_threshold}",
                            "duration": "{duration}",
                            "aggregation": "avg"
                        }
                    ],
                    "evaluation_interval": 30,
                    "cooldown_period": 180
                },
                variables=["response_time_threshold", "duration"]
            ),
            
            RuleTemplate(
                id="error_rate_spike",
                name="错误率飙升模板",
                description="监控错误率异常增长",
                category="application",
                rule_type=RuleType.ANOMALY,
                template_config={
                    "description": "错误率异常增长",
                    "severity": "high",
                    "anomaly_config": {
                        "algorithm": "statistical",
                        "sensitivity": "{sensitivity}",
                        "window_size": 100,
                        "min_samples": 50,
                        "std_dev_threshold": "{std_threshold}"
                    },
                    "evaluation_interval": 60,
                    "cooldown_period": 300
                },
                variables=["sensitivity", "std_threshold"]
            ),
            
            RuleTemplate(
                id="database_connection_failure",
                name="数据库连接失败模板",
                description="监控数据库连接失败",
                category="database",
                rule_type=RuleType.PATTERN,
                template_config={
                    "description": "数据库连接失败",
                    "severity": "critical",
                    "pattern_config": {
                        "patterns": [
                            "connection.*failed",
                            "database.*unreachable",
                            "timeout.*database"
                        ],
                        "match_type": "any",
                        "case_sensitive": False,
                        "regex_enabled": True
                    },
                    "evaluation_interval": 30,
                    "cooldown_period": 120
                },
                variables=[]
            )
        ]
        
        for template in default_templates:
            self.templates[template.id] = template
    
    def _load_rules(self):
        """加载规则配置"""
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                
                for rule_data in rules_data:
                    rule = self._dict_to_rule(rule_data)
                    if rule:
                        self.rules[rule.id] = rule
                
                self.logger.info(f"加载了 {len(self.rules)} 个告警规则")
                
            except Exception as e:
                self.logger.error(f"加载规则配置失败: {e}")
    
    def _dict_to_rule(self, rule_data: Dict[str, Any]) -> Optional[AlertRule]:
        """将字典转换为规则对象"""
        try:
            # 转换枚举类型
            rule_data["rule_type"] = RuleType(rule_data["rule_type"])
            rule_data["severity"] = Severity(rule_data["severity"])
            rule_data["status"] = RuleStatus(rule_data.get("status", "active"))
            
            # 转换日期时间
            if "created_at" in rule_data:
                rule_data["created_at"] = datetime.fromisoformat(rule_data["created_at"])
            if "updated_at" in rule_data:
                rule_data["updated_at"] = datetime.fromisoformat(rule_data["updated_at"])
            if "last_triggered" in rule_data and rule_data["last_triggered"]:
                rule_data["last_triggered"] = datetime.fromisoformat(rule_data["last_triggered"])
            if "last_evaluated" in rule_data and rule_data["last_evaluated"]:
                rule_data["last_evaluated"] = datetime.fromisoformat(rule_data["last_evaluated"])
            
            # 转换阈值条件
            if "threshold_conditions" in rule_data:
                conditions = []
                for cond_data in rule_data["threshold_conditions"]:
                    cond_data["operator"] = ComparisonOperator(cond_data["operator"])
                    conditions.append(ThresholdCondition(**cond_data))
                rule_data["threshold_conditions"] = conditions
            
            # 转换异常检测配置
            if "anomaly_config" in rule_data and rule_data["anomaly_config"]:
                rule_data["anomaly_config"] = AnomalyDetectionConfig(**rule_data["anomaly_config"])
            
            # 转换模式匹配配置
            if "pattern_config" in rule_data and rule_data["pattern_config"]:
                rule_data["pattern_config"] = PatternMatchConfig(**rule_data["pattern_config"])
            
            # 转换复合规则配置
            if "composite_config" in rule_data and rule_data["composite_config"]:
                rule_data["composite_config"] = CompositeRuleConfig(**rule_data["composite_config"])
            
            # 转换关联规则配置
            if "correlation_config" in rule_data and rule_data["correlation_config"]:
                rule_data["correlation_config"] = CorrelationRuleConfig(**rule_data["correlation_config"])
            
            return AlertRule(**rule_data)
            
        except Exception as e:
            self.logger.error(f"转换规则数据失败: {e}")
            return None
    
    def _rule_to_dict(self, rule: AlertRule) -> Dict[str, Any]:
        """将规则对象转换为字典"""
        rule_dict = asdict(rule)
        
        # 转换枚举为字符串
        rule_dict["rule_type"] = rule.rule_type.value
        rule_dict["severity"] = rule.severity.value
        rule_dict["status"] = rule.status.value
        
        # 转换日期时间为字符串
        if rule.created_at:
            rule_dict["created_at"] = rule.created_at.isoformat()
        if rule.updated_at:
            rule_dict["updated_at"] = rule.updated_at.isoformat()
        if rule.last_triggered:
            rule_dict["last_triggered"] = rule.last_triggered.isoformat()
        if rule.last_evaluated:
            rule_dict["last_evaluated"] = rule.last_evaluated.isoformat()
        
        # 转换阈值条件
        if rule.threshold_conditions:
            conditions = []
            for condition in rule.threshold_conditions:
                cond_dict = asdict(condition)
                cond_dict["operator"] = condition.operator.value
                conditions.append(cond_dict)
            rule_dict["threshold_conditions"] = conditions
        
        return rule_dict
    
    async def save_rules(self) -> bool:
        """保存规则配置"""
        try:
            rules_data = [self._rule_to_dict(rule) for rule in self.rules.values()]
            
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"保存了 {len(self.rules)} 个告警规则")
            return True
            
        except Exception as e:
            self.logger.error(f"保存规则配置失败: {e}")
            return False
    
    async def add_rule(self, rule: AlertRule) -> bool:
        """添加规则"""
        if rule.id in self.rules:
            self.logger.warning(f"规则已存在: {rule.id}")
            return False
        
        # 验证规则
        errors = await self.validate_rule(rule)
        if errors:
            self.logger.error(f"规则验证失败: {errors}")
            return False
        
        self.rules[rule.id] = rule
        await self.save_rules()
        
        self.logger.info(f"添加告警规则: {rule.id}")
        return True
    
    async def update_rule(self, rule_id: str, rule: AlertRule) -> bool:
        """更新规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"规则不存在: {rule_id}")
            return False
        
        # 验证规则
        errors = await self.validate_rule(rule)
        if errors:
            self.logger.error(f"规则验证失败: {errors}")
            return False
        
        rule.updated_at = datetime.now()
        self.rules[rule_id] = rule
        await self.save_rules()
        
        self.logger.info(f"更新告警规则: {rule_id}")
        return True
    
    async def remove_rule(self, rule_id: str) -> bool:
        """删除规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"规则不存在: {rule_id}")
            return False
        
        del self.rules[rule_id]
        await self.save_rules()
        
        self.logger.info(f"删除告警规则: {rule_id}")
        return True
    
    async def validate_rule(self, rule: AlertRule) -> List[str]:
        """验证规则"""
        errors = []
        
        try:
            # 基本验证
            if not rule.id:
                errors.append("规则ID不能为空")
            
            if not rule.name:
                errors.append("规则名称不能为空")
            
            # 使用JSON Schema验证
            rule_dict = self._rule_to_dict(rule)
            jsonschema.validate(rule_dict, self.rule_schema)
            
            # 类型特定验证
            if rule.rule_type == RuleType.THRESHOLD:
                if not rule.threshold_conditions:
                    errors.append("阈值规则必须配置阈值条件")
            
            elif rule.rule_type == RuleType.ANOMALY:
                if not rule.anomaly_config:
                    errors.append("异常检测规则必须配置异常检测参数")
            
            elif rule.rule_type == RuleType.PATTERN:
                if not rule.pattern_config:
                    errors.append("模式匹配规则必须配置模式参数")
                elif not rule.pattern_config.patterns:
                    errors.append("模式匹配规则必须配置匹配模式")
            
            elif rule.rule_type == RuleType.COMPOSITE:
                if not rule.composite_config:
                    errors.append("复合规则必须配置复合参数")
                elif not rule.composite_config.sub_rules:
                    errors.append("复合规则必须配置子规则")
            
            elif rule.rule_type == RuleType.CORRELATION:
                if not rule.correlation_config:
                    errors.append("关联规则必须配置关联参数")
                elif len(rule.correlation_config.related_metrics) < 2:
                    errors.append("关联规则至少需要2个相关指标")
            
        except jsonschema.ValidationError as e:
            errors.append(f"JSON Schema验证失败: {e.message}")
        except Exception as e:
            errors.append(f"验证过程出错: {str(e)}")
        
        return errors
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """获取规则"""
        return self.rules.get(rule_id)
    
    def get_rules(self, status: Optional[RuleStatus] = None,
                  rule_type: Optional[RuleType] = None,
                  severity: Optional[Severity] = None,
                  group: Optional[str] = None,
                  tags: Optional[List[str]] = None) -> List[AlertRule]:
        """获取规则列表"""
        rules = list(self.rules.values())
        
        if status:
            rules = [r for r in rules if r.status == status]
        
        if rule_type:
            rules = [r for r in rules if r.rule_type == rule_type]
        
        if severity:
            rules = [r for r in rules if r.severity == severity]
        
        if group:
            rules = [r for r in rules if r.group == group]
        
        if tags:
            rules = [r for r in rules if any(tag in r.tags for tag in tags)]
        
        return rules
    
    def get_active_rules(self) -> List[AlertRule]:
        """获取活跃规则"""
        return [rule for rule in self.rules.values() if rule.is_active()]
    
    async def create_rule_from_template(self, template_id: str, rule_id: str,
                                      rule_name: str, variables: Dict[str, Any]) -> Optional[AlertRule]:
        """从模板创建规则"""
        if template_id not in self.templates:
            self.logger.error(f"模板不存在: {template_id}")
            return None
        
        if rule_id in self.rules:
            self.logger.error(f"规则ID已存在: {rule_id}")
            return None
        
        template = self.templates[template_id]
        
        try:
            rule = template.create_rule(rule_id, rule_name, variables)
            
            # 验证规则
            errors = await self.validate_rule(rule)
            if errors:
                self.logger.error(f"从模板创建的规则验证失败: {errors}")
                return None
            
            await self.add_rule(rule)
            return rule
            
        except Exception as e:
            self.logger.error(f"从模板创建规则失败: {e}")
            return None
    
    def get_template(self, template_id: str) -> Optional[RuleTemplate]:
        """获取模板"""
        return self.templates.get(template_id)
    
    def get_templates(self, category: Optional[str] = None,
                     rule_type: Optional[RuleType] = None) -> List[RuleTemplate]:
        """获取模板列表"""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if rule_type:
            templates = [t for t in templates if t.rule_type == rule_type]
        
        return templates
    
    async def get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则统计信息"""
        total_rules = len(self.rules)
        active_rules = len([r for r in self.rules.values() if r.status == RuleStatus.ACTIVE])
        
        # 按类型统计
        type_stats = {}
        for rule_type in RuleType:
            count = len([r for r in self.rules.values() if r.rule_type == rule_type])
            type_stats[rule_type.value] = count
        
        # 按严重级别统计
        severity_stats = {}
        for severity in Severity:
            count = len([r for r in self.rules.values() if r.severity == severity])
            severity_stats[severity.value] = count
        
        # 按状态统计
        status_stats = {}
        for status in RuleStatus:
            count = len([r for r in self.rules.values() if r.status == status])
            status_stats[status.value] = count
        
        # 触发统计
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        triggered_24h = len([
            r for r in self.rules.values()
            if r.last_triggered and r.last_triggered > last_24h
        ])
        
        return {
            "total_rules": total_rules,
            "active_rules": active_rules,
            "total_templates": len(self.templates),
            "rule_type_distribution": type_stats,
            "severity_distribution": severity_stats,
            "status_distribution": status_stats,
            "triggered_rules_24h": triggered_24h,
            "average_evaluation_interval": sum(r.evaluation_interval for r in self.rules.values()) / total_rules if total_rules > 0 else 0
        }
    
    async def export_rules(self, export_path: str, rule_ids: Optional[List[str]] = None) -> bool:
        """导出规则"""
        try:
            if rule_ids:
                rules_to_export = [self.rules[rid] for rid in rule_ids if rid in self.rules]
            else:
                rules_to_export = list(self.rules.values())
            
            export_data = {
                "rules": [self._rule_to_dict(rule) for rule in rules_to_export],
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功导出 {len(rules_to_export)} 个规则到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出规则失败: {e}")
            return False
    
    async def import_rules(self, import_path: str, overwrite: bool = False) -> bool:
        """导入规则"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            rules_data = import_data.get("rules", [])
            imported_count = 0
            
            for rule_data in rules_data:
                rule = self._dict_to_rule(rule_data)
                if not rule:
                    continue
                
                if rule.id in self.rules and not overwrite:
                    self.logger.warning(f"规则已存在，跳过: {rule.id}")
                    continue
                
                # 验证规则
                errors = await self.validate_rule(rule)
                if errors:
                    self.logger.error(f"导入的规则验证失败: {rule.id}, 错误: {errors}")
                    continue
                
                self.rules[rule.id] = rule
                imported_count += 1
            
            await self.save_rules()
            self.logger.info(f"成功导入 {imported_count} 个规则")
            return True
            
        except Exception as e:
            self.logger.error(f"导入规则失败: {e}")
            return False