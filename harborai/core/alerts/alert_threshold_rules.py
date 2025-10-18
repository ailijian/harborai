"""
告警阈值规则管理器

负责管理告警阈值规则的定义、验证、评估和动态调整，
支持多种阈值类型、智能阈值、自适应调整和规则组合。
"""

import json
import yaml
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
from collections import defaultdict, deque
import statistics
import math
import numpy as np
from scipy import stats


class ThresholdType(Enum):
    """阈值类型"""
    STATIC = "static"                    # 静态阈值
    DYNAMIC = "dynamic"                  # 动态阈值
    ADAPTIVE = "adaptive"                # 自适应阈值
    PERCENTILE = "percentile"            # 百分位阈值
    SEASONAL = "seasonal"                # 季节性阈值
    ANOMALY_DETECTION = "anomaly"        # 异常检测
    COMPOSITE = "composite"              # 复合阈值
    BUSINESS_HOURS = "business_hours"    # 业务时间阈值
    RATE_OF_CHANGE = "rate_of_change"    # 变化率阈值
    BASELINE = "baseline"                # 基线阈值


class ThresholdOperator(Enum):
    """阈值操作符"""
    GREATER_THAN = "gt"                  # 大于
    GREATER_EQUAL = "gte"                # 大于等于
    LESS_THAN = "lt"                     # 小于
    LESS_EQUAL = "lte"                   # 小于等于
    EQUAL = "eq"                         # 等于
    NOT_EQUAL = "ne"                     # 不等于
    BETWEEN = "between"                  # 在范围内
    NOT_BETWEEN = "not_between"          # 不在范围内
    INCREASE_BY = "increase_by"          # 增加了
    DECREASE_BY = "decrease_by"          # 减少了
    CHANGE_BY = "change_by"              # 变化了


class ThresholdSeverity(Enum):
    """阈值严重级别"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AggregationType(Enum):
    """聚合类型"""
    AVERAGE = "avg"                      # 平均值
    SUM = "sum"                          # 求和
    MIN = "min"                          # 最小值
    MAX = "max"                          # 最大值
    COUNT = "count"                      # 计数
    MEDIAN = "median"                    # 中位数
    PERCENTILE = "percentile"            # 百分位
    STDDEV = "stddev"                    # 标准差
    VARIANCE = "variance"                # 方差
    RATE = "rate"                        # 速率


class SeasonalPattern(Enum):
    """季节性模式"""
    HOURLY = "hourly"                    # 小时模式
    DAILY = "daily"                      # 日模式
    WEEKLY = "weekly"                    # 周模式
    MONTHLY = "monthly"                  # 月模式
    YEARLY = "yearly"                    # 年模式


@dataclass
class TimeWindow:
    """时间窗口"""
    duration_seconds: int                # 持续时间（秒）
    aggregation: AggregationType = AggregationType.AVERAGE  # 聚合方式
    min_samples: int = 1                 # 最小样本数
    
    def __post_init__(self):
        if self.duration_seconds <= 0:
            raise ValueError("时间窗口持续时间必须大于0")
        if self.min_samples <= 0:
            raise ValueError("最小样本数必须大于0")


@dataclass
class BusinessHours:
    """业务时间"""
    start_hour: int                      # 开始小时
    end_hour: int                        # 结束小时
    weekdays: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])  # 工作日
    timezone: str = "UTC"                # 时区
    
    def is_business_hours(self, timestamp: datetime) -> bool:
        """检查是否在业务时间内"""
        if timestamp.weekday() not in self.weekdays:
            return False
        
        hour = timestamp.hour
        if self.start_hour <= self.end_hour:
            return self.start_hour <= hour < self.end_hour
        else:
            return hour >= self.start_hour or hour < self.end_hour


@dataclass
class SeasonalConfig:
    """季节性配置"""
    pattern: SeasonalPattern             # 季节性模式
    window_size: int                     # 窗口大小
    min_periods: int = 3                 # 最小周期数
    trend_factor: float = 0.1            # 趋势因子
    seasonal_factor: float = 0.3         # 季节性因子


@dataclass
class AdaptiveConfig:
    """自适应配置"""
    learning_rate: float = 0.1           # 学习率
    adaptation_window: int = 100         # 适应窗口
    sensitivity: float = 2.0             # 敏感度
    min_samples: int = 10                # 最小样本数
    max_deviation: float = 3.0           # 最大偏差


@dataclass
class AnomalyConfig:
    """异常检测配置"""
    method: str = "zscore"               # 检测方法: zscore, iqr, isolation_forest
    window_size: int = 100               # 窗口大小
    threshold: float = 3.0               # 阈值
    min_samples: int = 30                # 最小样本数


@dataclass
class ThresholdCondition:
    """阈值条件"""
    operator: ThresholdOperator          # 操作符
    value: Union[float, List[float]]     # 阈值
    severity: ThresholdSeverity          # 严重级别
    
    # 时间配置
    duration_seconds: int = 0            # 持续时间
    evaluation_window: Optional[TimeWindow] = None  # 评估窗口
    
    # 业务时间配置
    business_hours_only: bool = False    # 仅在业务时间
    business_hours: Optional[BusinessHours] = None  # 业务时间配置
    
    def evaluate(self, current_value: float, previous_value: Optional[float] = None) -> bool:
        """评估阈值条件"""
        try:
            if self.operator == ThresholdOperator.GREATER_THAN:
                return current_value > self.value
            elif self.operator == ThresholdOperator.GREATER_EQUAL:
                return current_value >= self.value
            elif self.operator == ThresholdOperator.LESS_THAN:
                return current_value < self.value
            elif self.operator == ThresholdOperator.LESS_EQUAL:
                return current_value <= self.value
            elif self.operator == ThresholdOperator.EQUAL:
                return abs(current_value - self.value) < 1e-9
            elif self.operator == ThresholdOperator.NOT_EQUAL:
                return abs(current_value - self.value) >= 1e-9
            elif self.operator == ThresholdOperator.BETWEEN:
                if isinstance(self.value, list) and len(self.value) == 2:
                    return self.value[0] <= current_value <= self.value[1]
                return False
            elif self.operator == ThresholdOperator.NOT_BETWEEN:
                if isinstance(self.value, list) and len(self.value) == 2:
                    return not (self.value[0] <= current_value <= self.value[1])
                return False
            elif self.operator == ThresholdOperator.INCREASE_BY:
                if previous_value is not None:
                    return current_value - previous_value >= self.value
                return False
            elif self.operator == ThresholdOperator.DECREASE_BY:
                if previous_value is not None:
                    return previous_value - current_value >= self.value
                return False
            elif self.operator == ThresholdOperator.CHANGE_BY:
                if previous_value is not None:
                    return abs(current_value - previous_value) >= self.value
                return False
            
            return False
            
        except Exception:
            return False


@dataclass
class ThresholdRule:
    """阈值规则"""
    id: str
    name: str
    description: str
    
    # 指标配置
    metric_name: str                     # 指标名称
    metric_labels: Dict[str, str] = field(default_factory=dict)  # 指标标签
    
    # 阈值配置
    threshold_type: ThresholdType        # 阈值类型
    conditions: List[ThresholdCondition] = field(default_factory=list)  # 阈值条件
    
    # 时间配置
    evaluation_interval: int = 60        # 评估间隔（秒）
    evaluation_window: Optional[TimeWindow] = None  # 评估窗口
    
    # 特殊配置
    seasonal_config: Optional[SeasonalConfig] = None  # 季节性配置
    adaptive_config: Optional[AdaptiveConfig] = None  # 自适应配置
    anomaly_config: Optional[AnomalyConfig] = None    # 异常检测配置
    business_hours: Optional[BusinessHours] = None    # 业务时间
    
    # 复合规则配置
    composite_rules: List[str] = field(default_factory=list)  # 复合规则ID
    composite_operator: str = "AND"      # 复合操作符: AND, OR
    
    # 元数据
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def matches_metric(self, metric_name: str, metric_labels: Dict[str, str]) -> bool:
        """检查规则是否匹配指标"""
        if not self.enabled:
            return False
        
        # 检查指标名称
        if self.metric_name != metric_name:
            return False
        
        # 检查标签匹配
        for key, value in self.metric_labels.items():
            if key not in metric_labels or metric_labels[key] != value:
                return False
        
        return True
    
    def is_in_business_hours(self, timestamp: datetime) -> bool:
        """检查是否在业务时间内"""
        if not self.business_hours:
            return True
        
        return self.business_hours.is_business_hours(timestamp)


@dataclass
class MetricDataPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def age_seconds(self) -> float:
        """获取数据点年龄（秒）"""
        return (datetime.now() - self.timestamp).total_seconds()


@dataclass
class ThresholdEvaluation:
    """阈值评估结果"""
    rule_id: str
    metric_name: str
    current_value: float
    threshold_value: Union[float, List[float]]
    
    # 评估结果
    triggered: bool
    severity: Optional[ThresholdSeverity] = None
    condition_results: List[bool] = field(default_factory=list)
    
    # 时间信息
    evaluation_time: datetime = field(default_factory=datetime.now)
    duration_seconds: float = 0
    
    # 上下文信息
    previous_value: Optional[float] = None
    baseline_value: Optional[float] = None
    anomaly_score: Optional[float] = None
    
    # 元数据
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AlertThresholdRulesManager:
    """告警阈值规则管理器"""
    
    def __init__(self, config_dir: str = "config/alerts"):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 规则存储
        self.rules: Dict[str, ThresholdRule] = {}
        
        # 数据存储
        self.metric_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.baseline_data: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.seasonal_models: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.adaptive_models: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # 评估历史
        self.evaluation_history: deque = deque(maxlen=10000)
        self.triggered_rules: Dict[str, datetime] = {}
        
        # 配置文件
        self.rules_file = self.config_dir / "threshold_rules.json"
        
        # 加载配置
        self._load_default_rules()
        self._load_rules()
    
    def _load_default_rules(self):
        """加载默认阈值规则"""
        default_rules = [
            # CPU使用率规则
            ThresholdRule(
                id="cpu_usage_critical",
                name="CPU使用率关键告警",
                description="CPU使用率超过90%时触发关键告警",
                metric_name="cpu_usage_percent",
                threshold_type=ThresholdType.STATIC,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=90.0,
                        severity=ThresholdSeverity.CRITICAL,
                        duration_seconds=300,  # 持续5分钟
                        evaluation_window=TimeWindow(
                            duration_seconds=300,
                            aggregation=AggregationType.AVERAGE,
                            min_samples=5
                        )
                    )
                ],
                evaluation_interval=60,
                tags=["cpu", "critical", "infrastructure"]
            ),
            
            ThresholdRule(
                id="cpu_usage_high",
                name="CPU使用率高级告警",
                description="CPU使用率超过80%时触发高级告警",
                metric_name="cpu_usage_percent",
                threshold_type=ThresholdType.STATIC,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=80.0,
                        severity=ThresholdSeverity.HIGH,
                        duration_seconds=600,  # 持续10分钟
                        evaluation_window=TimeWindow(
                            duration_seconds=300,
                            aggregation=AggregationType.AVERAGE
                        )
                    )
                ],
                evaluation_interval=60,
                tags=["cpu", "high", "infrastructure"]
            ),
            
            # 内存使用率规则
            ThresholdRule(
                id="memory_usage_critical",
                name="内存使用率关键告警",
                description="内存使用率超过95%时触发关键告警",
                metric_name="memory_usage_percent",
                threshold_type=ThresholdType.STATIC,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=95.0,
                        severity=ThresholdSeverity.CRITICAL,
                        duration_seconds=180,  # 持续3分钟
                        evaluation_window=TimeWindow(
                            duration_seconds=180,
                            aggregation=AggregationType.AVERAGE
                        )
                    )
                ],
                evaluation_interval=30,
                tags=["memory", "critical", "infrastructure"]
            ),
            
            ThresholdRule(
                id="memory_usage_adaptive",
                name="内存使用率自适应告警",
                description="基于历史数据的自适应内存使用率告警",
                metric_name="memory_usage_percent",
                threshold_type=ThresholdType.ADAPTIVE,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=0.0,  # 动态计算
                        severity=ThresholdSeverity.HIGH,
                        duration_seconds=300
                    )
                ],
                adaptive_config=AdaptiveConfig(
                    learning_rate=0.1,
                    adaptation_window=200,
                    sensitivity=2.5,
                    min_samples=20
                ),
                evaluation_interval=60,
                tags=["memory", "adaptive", "ml"]
            ),
            
            # 磁盘使用率规则
            ThresholdRule(
                id="disk_usage_critical",
                name="磁盘使用率关键告警",
                description="磁盘使用率超过90%时触发关键告警",
                metric_name="disk_usage_percent",
                threshold_type=ThresholdType.STATIC,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=90.0,
                        severity=ThresholdSeverity.CRITICAL,
                        duration_seconds=300
                    )
                ],
                evaluation_interval=300,  # 5分钟检查一次
                tags=["disk", "critical", "storage"]
            ),
            
            # API响应时间规则
            ThresholdRule(
                id="api_response_time_high",
                name="API响应时间高级告警",
                description="API响应时间超过2秒时触发告警",
                metric_name="api_response_time_seconds",
                threshold_type=ThresholdType.PERCENTILE,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=2.0,
                        severity=ThresholdSeverity.HIGH,
                        duration_seconds=300,
                        evaluation_window=TimeWindow(
                            duration_seconds=300,
                            aggregation=AggregationType.PERCENTILE,
                            min_samples=10
                        )
                    )
                ],
                evaluation_interval=60,
                tags=["api", "performance", "latency"]
            ),
            
            # 错误率规则
            ThresholdRule(
                id="error_rate_critical",
                name="错误率关键告警",
                description="错误率超过5%时触发关键告警",
                metric_name="error_rate_percent",
                threshold_type=ThresholdType.STATIC,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=5.0,
                        severity=ThresholdSeverity.CRITICAL,
                        duration_seconds=180,
                        evaluation_window=TimeWindow(
                            duration_seconds=300,
                            aggregation=AggregationType.AVERAGE,
                            min_samples=5
                        )
                    )
                ],
                evaluation_interval=30,
                tags=["error", "critical", "reliability"]
            ),
            
            # 请求量异常检测规则
            ThresholdRule(
                id="request_volume_anomaly",
                name="请求量异常检测",
                description="基于异常检测的请求量告警",
                metric_name="request_count_per_minute",
                threshold_type=ThresholdType.ANOMALY_DETECTION,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=0.0,  # 动态计算
                        severity=ThresholdSeverity.MEDIUM,
                        duration_seconds=300
                    )
                ],
                anomaly_config=AnomalyConfig(
                    method="zscore",
                    window_size=100,
                    threshold=3.0,
                    min_samples=30
                ),
                evaluation_interval=60,
                tags=["request", "anomaly", "traffic"]
            ),
            
            # 数据库连接数规则
            ThresholdRule(
                id="db_connections_high",
                name="数据库连接数高级告警",
                description="数据库连接数超过阈值时告警",
                metric_name="database_connections_active",
                threshold_type=ThresholdType.STATIC,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=80.0,
                        severity=ThresholdSeverity.HIGH,
                        duration_seconds=300
                    )
                ],
                evaluation_interval=60,
                tags=["database", "connections", "resource"]
            ),
            
            # Token使用量规则
            ThresholdRule(
                id="token_usage_high",
                name="Token使用量高级告警",
                description="Token使用量超过阈值时告警",
                metric_name="token_usage_per_hour",
                threshold_type=ThresholdType.BUSINESS_HOURS,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=10000.0,
                        severity=ThresholdSeverity.HIGH,
                        duration_seconds=300,
                        business_hours_only=True
                    )
                ],
                business_hours=BusinessHours(
                    start_hour=9,
                    end_hour=18,
                    weekdays=[0, 1, 2, 3, 4]  # 工作日
                ),
                evaluation_interval=300,
                tags=["token", "usage", "business"]
            ),
            
            # 成本告警规则
            ThresholdRule(
                id="cost_daily_high",
                name="日成本高级告警",
                description="日成本超过预算时告警",
                metric_name="cost_daily_usd",
                threshold_type=ThresholdType.STATIC,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=100.0,
                        severity=ThresholdSeverity.HIGH,
                        duration_seconds=3600  # 持续1小时
                    )
                ],
                evaluation_interval=3600,  # 1小时检查一次
                tags=["cost", "budget", "financial"]
            ),
            
            # 变化率规则
            ThresholdRule(
                id="cpu_spike_detection",
                name="CPU使用率突增检测",
                description="检测CPU使用率的突然增加",
                metric_name="cpu_usage_percent",
                threshold_type=ThresholdType.RATE_OF_CHANGE,
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.INCREASE_BY,
                        value=30.0,  # 增加30%
                        severity=ThresholdSeverity.MEDIUM,
                        duration_seconds=120,
                        evaluation_window=TimeWindow(
                            duration_seconds=300,
                            aggregation=AggregationType.AVERAGE
                        )
                    )
                ],
                evaluation_interval=60,
                tags=["cpu", "spike", "change"]
            ),
            
            # 复合规则示例
            ThresholdRule(
                id="system_overload_composite",
                name="系统过载复合告警",
                description="CPU和内存同时高使用率时触发",
                metric_name="system_health",
                threshold_type=ThresholdType.COMPOSITE,
                composite_rules=["cpu_usage_high", "memory_usage_critical"],
                composite_operator="AND",
                conditions=[
                    ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=0.0,
                        severity=ThresholdSeverity.CRITICAL,
                        duration_seconds=300
                    )
                ],
                evaluation_interval=60,
                tags=["system", "composite", "overload"]
            )
        ]
        
        for rule in default_rules:
            self.rules[rule.id] = rule
    
    def _load_rules(self):
        """加载阈值规则"""
        if self.rules_file.exists():
            try:
                with open(self.rules_file, 'r', encoding='utf-8') as f:
                    rules_data = json.load(f)
                
                for rule_data in rules_data:
                    rule = self._dict_to_rule(rule_data)
                    if rule:
                        self.rules[rule.id] = rule
                
                self.logger.info(f"加载了 {len(self.rules)} 个阈值规则")
                
            except Exception as e:
                self.logger.error(f"加载阈值规则失败: {e}")
    
    def _dict_to_rule(self, rule_data: Dict[str, Any]) -> Optional[ThresholdRule]:
        """将字典转换为规则对象"""
        try:
            # 转换枚举类型
            if "threshold_type" in rule_data:
                rule_data["threshold_type"] = ThresholdType(rule_data["threshold_type"])
            
            # 转换条件
            if "conditions" in rule_data:
                conditions = []
                for condition_data in rule_data["conditions"]:
                    condition_data["operator"] = ThresholdOperator(condition_data["operator"])
                    condition_data["severity"] = ThresholdSeverity(condition_data["severity"])
                    
                    # 转换评估窗口
                    if "evaluation_window" in condition_data and condition_data["evaluation_window"]:
                        window_data = condition_data["evaluation_window"]
                        window_data["aggregation"] = AggregationType(window_data["aggregation"])
                        condition_data["evaluation_window"] = TimeWindow(**window_data)
                    
                    # 转换业务时间
                    if "business_hours" in condition_data and condition_data["business_hours"]:
                        condition_data["business_hours"] = BusinessHours(**condition_data["business_hours"])
                    
                    conditions.append(ThresholdCondition(**condition_data))
                
                rule_data["conditions"] = conditions
            
            # 转换评估窗口
            if "evaluation_window" in rule_data and rule_data["evaluation_window"]:
                window_data = rule_data["evaluation_window"]
                window_data["aggregation"] = AggregationType(window_data["aggregation"])
                rule_data["evaluation_window"] = TimeWindow(**window_data)
            
            # 转换特殊配置
            if "seasonal_config" in rule_data and rule_data["seasonal_config"]:
                seasonal_data = rule_data["seasonal_config"]
                seasonal_data["pattern"] = SeasonalPattern(seasonal_data["pattern"])
                rule_data["seasonal_config"] = SeasonalConfig(**seasonal_data)
            
            if "adaptive_config" in rule_data and rule_data["adaptive_config"]:
                rule_data["adaptive_config"] = AdaptiveConfig(**rule_data["adaptive_config"])
            
            if "anomaly_config" in rule_data and rule_data["anomaly_config"]:
                rule_data["anomaly_config"] = AnomalyConfig(**rule_data["anomaly_config"])
            
            if "business_hours" in rule_data and rule_data["business_hours"]:
                rule_data["business_hours"] = BusinessHours(**rule_data["business_hours"])
            
            # 转换日期时间
            if "created_at" in rule_data:
                rule_data["created_at"] = datetime.fromisoformat(rule_data["created_at"])
            if "updated_at" in rule_data:
                rule_data["updated_at"] = datetime.fromisoformat(rule_data["updated_at"])
            
            return ThresholdRule(**rule_data)
            
        except Exception as e:
            self.logger.error(f"转换阈值规则失败: {e}")
            return None
    
    def _rule_to_dict(self, rule: ThresholdRule) -> Dict[str, Any]:
        """将规则对象转换为字典"""
        rule_dict = asdict(rule)
        
        # 转换枚举为字符串
        rule_dict["threshold_type"] = rule.threshold_type.value
        
        # 转换条件
        if rule.conditions:
            conditions = []
            for condition in rule.conditions:
                condition_dict = asdict(condition)
                condition_dict["operator"] = condition.operator.value
                condition_dict["severity"] = condition.severity.value
                
                # 转换评估窗口
                if condition.evaluation_window:
                    window_dict = asdict(condition.evaluation_window)
                    window_dict["aggregation"] = condition.evaluation_window.aggregation.value
                    condition_dict["evaluation_window"] = window_dict
                
                conditions.append(condition_dict)
            
            rule_dict["conditions"] = conditions
        
        # 转换评估窗口
        if rule.evaluation_window:
            window_dict = asdict(rule.evaluation_window)
            window_dict["aggregation"] = rule.evaluation_window.aggregation.value
            rule_dict["evaluation_window"] = window_dict
        
        # 转换特殊配置
        if rule.seasonal_config:
            seasonal_dict = asdict(rule.seasonal_config)
            seasonal_dict["pattern"] = rule.seasonal_config.pattern.value
            rule_dict["seasonal_config"] = seasonal_dict
        
        # 转换日期时间为字符串
        if rule.created_at:
            rule_dict["created_at"] = rule.created_at.isoformat()
        if rule.updated_at:
            rule_dict["updated_at"] = rule.updated_at.isoformat()
        
        return rule_dict
    
    async def save_rules(self) -> bool:
        """保存阈值规则"""
        try:
            rules_data = [self._rule_to_dict(rule) for rule in self.rules.values()]
            
            with open(self.rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"保存了 {len(self.rules)} 个阈值规则")
            return True
            
        except Exception as e:
            self.logger.error(f"保存阈值规则失败: {e}")
            return False
    
    async def add_rule(self, rule: ThresholdRule) -> bool:
        """添加阈值规则"""
        if rule.id in self.rules:
            self.logger.warning(f"阈值规则已存在: {rule.id}")
            return False
        
        self.rules[rule.id] = rule
        await self.save_rules()
        
        self.logger.info(f"添加阈值规则: {rule.id}")
        return True
    
    async def update_rule(self, rule_id: str, rule: ThresholdRule) -> bool:
        """更新阈值规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"阈值规则不存在: {rule_id}")
            return False
        
        rule.updated_at = datetime.now()
        self.rules[rule_id] = rule
        await self.save_rules()
        
        self.logger.info(f"更新阈值规则: {rule_id}")
        return True
    
    async def remove_rule(self, rule_id: str) -> bool:
        """删除阈值规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"阈值规则不存在: {rule_id}")
            return False
        
        del self.rules[rule_id]
        await self.save_rules()
        
        self.logger.info(f"删除阈值规则: {rule_id}")
        return True
    
    def add_metric_data(self, metric_name: str, value: float, 
                       labels: Dict[str, str] = None, timestamp: datetime = None):
        """添加指标数据"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if labels is None:
            labels = {}
        
        data_point = MetricDataPoint(
            timestamp=timestamp,
            value=value,
            labels=labels
        )
        
        # 生成指标键
        metric_key = self._generate_metric_key(metric_name, labels)
        self.metric_data[metric_key].append(data_point)
        
        # 更新自适应模型
        self._update_adaptive_models(metric_key, value, timestamp)
    
    def _generate_metric_key(self, metric_name: str, labels: Dict[str, str]) -> str:
        """生成指标键"""
        if not labels:
            return metric_name
        
        label_str = ",".join([f"{k}={v}" for k, v in sorted(labels.items())])
        return f"{metric_name}{{{label_str}}}"
    
    def _update_adaptive_models(self, metric_key: str, value: float, timestamp: datetime):
        """更新自适应模型"""
        if metric_key not in self.adaptive_models:
            self.adaptive_models[metric_key] = {
                "mean": value,
                "variance": 0.0,
                "count": 1,
                "last_update": timestamp
            }
        else:
            model = self.adaptive_models[metric_key]
            count = model["count"]
            old_mean = model["mean"]
            
            # 更新均值和方差（在线算法）
            new_mean = old_mean + (value - old_mean) / (count + 1)
            new_variance = (model["variance"] * count + (value - old_mean) * (value - new_mean)) / (count + 1)
            
            model["mean"] = new_mean
            model["variance"] = new_variance
            model["count"] = count + 1
            model["last_update"] = timestamp
    
    def evaluate_rule(self, rule: ThresholdRule, metric_name: str, 
                     labels: Dict[str, str] = None) -> Optional[ThresholdEvaluation]:
        """评估阈值规则"""
        if not rule.enabled:
            return None
        
        if not rule.matches_metric(metric_name, labels or {}):
            return None
        
        # 获取指标数据
        metric_key = self._generate_metric_key(metric_name, labels or {})
        data_points = list(self.metric_data[metric_key])
        
        if not data_points:
            return None
        
        # 获取当前值
        current_data = data_points[-1]
        current_value = current_data.value
        
        # 获取前一个值
        previous_value = None
        if len(data_points) > 1:
            previous_value = data_points[-2].value
        
        # 计算阈值
        threshold_value = self._calculate_threshold(rule, metric_key, data_points)
        if threshold_value is None:
            return None
        
        # 评估条件
        condition_results = []
        triggered = False
        severity = None
        
        for condition in rule.conditions:
            # 检查业务时间
            if condition.business_hours_only and not rule.is_in_business_hours(current_data.timestamp):
                condition_results.append(False)
                continue
            
            # 使用计算出的阈值或条件中的阈值
            eval_value = threshold_value if rule.threshold_type != ThresholdType.STATIC else condition.value
            
            # 创建临时条件进行评估
            temp_condition = ThresholdCondition(
                operator=condition.operator,
                value=eval_value,
                severity=condition.severity
            )
            
            result = temp_condition.evaluate(current_value, previous_value)
            condition_results.append(result)
            
            if result:
                triggered = True
                if severity is None or self._compare_severity(condition.severity, severity) > 0:
                    severity = condition.severity
        
        # 计算异常分数
        anomaly_score = None
        if rule.threshold_type == ThresholdType.ANOMALY_DETECTION:
            anomaly_score = self._calculate_anomaly_score(rule, metric_key, current_value)
        
        # 创建评估结果
        evaluation = ThresholdEvaluation(
            rule_id=rule.id,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            triggered=triggered,
            severity=severity,
            condition_results=condition_results,
            previous_value=previous_value,
            anomaly_score=anomaly_score,
            labels=labels or {},
            metadata={
                "rule_type": rule.threshold_type.value,
                "evaluation_time": datetime.now().isoformat()
            }
        )
        
        # 记录评估历史
        self.evaluation_history.append(evaluation)
        
        # 更新触发状态
        if triggered:
            self.triggered_rules[rule.id] = datetime.now()
        elif rule.id in self.triggered_rules:
            del self.triggered_rules[rule.id]
        
        return evaluation
    
    def _calculate_threshold(self, rule: ThresholdRule, metric_key: str, 
                           data_points: List[MetricDataPoint]) -> Optional[Union[float, List[float]]]:
        """计算阈值"""
        if rule.threshold_type == ThresholdType.STATIC:
            # 静态阈值直接使用条件中的值
            return None
        
        elif rule.threshold_type == ThresholdType.ADAPTIVE:
            return self._calculate_adaptive_threshold(rule, metric_key)
        
        elif rule.threshold_type == ThresholdType.PERCENTILE:
            return self._calculate_percentile_threshold(rule, data_points)
        
        elif rule.threshold_type == ThresholdType.SEASONAL:
            return self._calculate_seasonal_threshold(rule, metric_key, data_points)
        
        elif rule.threshold_type == ThresholdType.ANOMALY_DETECTION:
            return self._calculate_anomaly_threshold(rule, metric_key, data_points)
        
        elif rule.threshold_type == ThresholdType.BASELINE:
            return self._calculate_baseline_threshold(rule, metric_key, data_points)
        
        return None
    
    def _calculate_adaptive_threshold(self, rule: ThresholdRule, metric_key: str) -> Optional[float]:
        """计算自适应阈值"""
        if metric_key not in self.adaptive_models:
            return None
        
        model = self.adaptive_models[metric_key]
        config = rule.adaptive_config
        
        if not config or model["count"] < config.min_samples:
            return None
        
        mean = model["mean"]
        std = math.sqrt(model["variance"])
        
        # 计算阈值
        threshold = mean + config.sensitivity * std
        
        return threshold
    
    def _calculate_percentile_threshold(self, rule: ThresholdRule, 
                                      data_points: List[MetricDataPoint]) -> Optional[float]:
        """计算百分位阈值"""
        if len(data_points) < 10:
            return None
        
        # 获取最近的数据点
        recent_points = data_points[-100:]  # 最近100个点
        values = [p.value for p in recent_points]
        
        # 计算95百分位
        percentile_95 = np.percentile(values, 95)
        
        return percentile_95
    
    def _calculate_seasonal_threshold(self, rule: ThresholdRule, metric_key: str,
                                    data_points: List[MetricDataPoint]) -> Optional[float]:
        """计算季节性阈值"""
        config = rule.seasonal_config
        if not config or len(data_points) < config.min_periods * config.window_size:
            return None
        
        # 简化的季节性计算
        values = [p.value for p in data_points[-config.window_size * config.min_periods:]]
        
        if config.pattern == SeasonalPattern.HOURLY:
            # 按小时分组
            hourly_means = {}
            for i, point in enumerate(data_points[-config.window_size:]):
                hour = point.timestamp.hour
                if hour not in hourly_means:
                    hourly_means[hour] = []
                hourly_means[hour].append(point.value)
            
            current_hour = data_points[-1].timestamp.hour
            if current_hour in hourly_means:
                hour_mean = statistics.mean(hourly_means[current_hour])
                hour_std = statistics.stdev(hourly_means[current_hour]) if len(hourly_means[current_hour]) > 1 else 0
                return hour_mean + 2 * hour_std
        
        # 默认返回均值 + 2倍标准差
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        
        return mean + 2 * std
    
    def _calculate_anomaly_threshold(self, rule: ThresholdRule, metric_key: str,
                                   data_points: List[MetricDataPoint]) -> Optional[float]:
        """计算异常检测阈值"""
        config = rule.anomaly_config
        if not config or len(data_points) < config.min_samples:
            return None
        
        values = [p.value for p in data_points[-config.window_size:]]
        
        if config.method == "zscore":
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0
            return mean + config.threshold * std
        
        elif config.method == "iqr":
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            return q3 + 1.5 * iqr
        
        return None
    
    def _calculate_baseline_threshold(self, rule: ThresholdRule, metric_key: str,
                                    data_points: List[MetricDataPoint]) -> Optional[float]:
        """计算基线阈值"""
        if metric_key not in self.baseline_data:
            # 计算基线
            if len(data_points) >= 100:
                baseline_values = [p.value for p in data_points[-100:]]
                baseline_mean = statistics.mean(baseline_values)
                baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
                
                self.baseline_data[metric_key] = {
                    "mean": baseline_mean,
                    "std": baseline_std,
                    "calculated_at": datetime.now()
                }
            else:
                return None
        
        baseline = self.baseline_data[metric_key]
        return baseline["mean"] + 2 * baseline["std"]
    
    def _calculate_anomaly_score(self, rule: ThresholdRule, metric_key: str, value: float) -> Optional[float]:
        """计算异常分数"""
        config = rule.anomaly_config
        if not config:
            return None
        
        data_points = list(self.metric_data[metric_key])
        if len(data_points) < config.min_samples:
            return None
        
        values = [p.value for p in data_points[-config.window_size:]]
        
        if config.method == "zscore":
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 1
            return abs(value - mean) / std
        
        return None
    
    def _compare_severity(self, severity1: ThresholdSeverity, severity2: ThresholdSeverity) -> int:
        """比较严重级别"""
        severity_order = {
            ThresholdSeverity.INFO: 0,
            ThresholdSeverity.LOW: 1,
            ThresholdSeverity.MEDIUM: 2,
            ThresholdSeverity.HIGH: 3,
            ThresholdSeverity.CRITICAL: 4
        }
        
        return severity_order.get(severity1, 0) - severity_order.get(severity2, 0)
    
    def get_rule(self, rule_id: str) -> Optional[ThresholdRule]:
        """获取阈值规则"""
        return self.rules.get(rule_id)
    
    def get_rules(self, enabled_only: bool = True, metric_name: str = None) -> List[ThresholdRule]:
        """获取阈值规则列表"""
        rules = list(self.rules.values())
        
        if enabled_only:
            rules = [r for r in rules if r.enabled]
        
        if metric_name:
            rules = [r for r in rules if r.metric_name == metric_name]
        
        return sorted(rules, key=lambda r: r.created_at)
    
    def get_matching_rules(self, metric_name: str, labels: Dict[str, str] = None) -> List[ThresholdRule]:
        """获取匹配的阈值规则"""
        matching_rules = []
        
        for rule in self.get_rules(enabled_only=True):
            if rule.matches_metric(metric_name, labels or {}):
                matching_rules.append(rule)
        
        return matching_rules
    
    def get_evaluation_history(self, rule_id: str = None, limit: int = 100) -> List[ThresholdEvaluation]:
        """获取评估历史"""
        evaluations = list(self.evaluation_history)
        
        if rule_id:
            evaluations = [e for e in evaluations if e.rule_id == rule_id]
        
        # 按时间倒序排列
        evaluations.sort(key=lambda e: e.evaluation_time, reverse=True)
        
        return evaluations[:limit]
    
    def get_triggered_rules(self) -> Dict[str, datetime]:
        """获取当前触发的规则"""
        return self.triggered_rules.copy()
    
    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_rules = len(self.rules)
        enabled_rules = len([r for r in self.rules.values() if r.enabled])
        triggered_rules = len(self.triggered_rules)
        
        # 按类型统计规则
        rule_by_type = {}
        for rule_type in ThresholdType:
            count = len([r for r in self.rules.values() if r.threshold_type == rule_type])
            rule_by_type[rule_type.value] = count
        
        # 按严重级别统计评估
        evaluation_by_severity = {}
        for severity in ThresholdSeverity:
            count = len([e for e in self.evaluation_history if e.severity == severity])
            evaluation_by_severity[severity.value] = count
        
        # 最近24小时统计
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_evaluations = len([
            e for e in self.evaluation_history
            if e.evaluation_time > last_24h
        ])
        
        recent_triggered = len([
            e for e in self.evaluation_history
            if e.evaluation_time > last_24h and e.triggered
        ])
        
        return {
            "total_rules": total_rules,
            "enabled_rules": enabled_rules,
            "triggered_rules": triggered_rules,
            "rule_type_distribution": rule_by_type,
            "evaluation_severity_distribution": evaluation_by_severity,
            "total_evaluations": len(self.evaluation_history),
            "evaluations_24h": recent_evaluations,
            "triggered_24h": recent_triggered,
            "metric_data_points": sum(len(data) for data in self.metric_data.values()),
            "adaptive_models": len(self.adaptive_models),
            "baseline_models": len(self.baseline_data)
        }
    
    async def export_rules(self, export_path: str) -> bool:
        """导出阈值规则"""
        try:
            export_data = {
                "rules": [self._rule_to_dict(rule) for rule in self.rules.values()],
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功导出阈值规则到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出阈值规则失败: {e}")
            return False
    
    async def import_rules(self, import_path: str, overwrite: bool = False) -> bool:
        """导入阈值规则"""
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
                    self.logger.warning(f"阈值规则已存在，跳过: {rule.id}")
                    continue
                
                self.rules[rule.id] = rule
                imported_count += 1
            
            await self.save_rules()
            
            self.logger.info(f"成功导入 {imported_count} 个阈值规则")
            return True
            
        except Exception as e:
            self.logger.error(f"导入阈值规则失败: {e}")
            return False
    
    async def cleanup_old_data(self, days: int = 7):
        """清理旧数据"""
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # 清理指标数据
        for metric_key in list(self.metric_data.keys()):
            data_points = self.metric_data[metric_key]
            # 保留最近的数据点
            recent_points = deque([
                point for point in data_points
                if point.timestamp > cutoff_time
            ], maxlen=10000)
            self.metric_data[metric_key] = recent_points
        
        # 清理评估历史
        recent_evaluations = deque([
            evaluation for evaluation in self.evaluation_history
            if evaluation.evaluation_time > cutoff_time
        ], maxlen=10000)
        self.evaluation_history = recent_evaluations
        
        self.logger.info(f"清理了 {days} 天前的旧数据"