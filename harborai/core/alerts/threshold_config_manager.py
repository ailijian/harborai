"""
告警阈值配置管理器

负责管理复杂的告警阈值配置，包括静态阈值、动态阈值、自适应阈值、
季节性阈值、业务时间阈值等多种阈值类型的配置和管理。
"""

import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging


class ThresholdType(Enum):
    """阈值类型"""
    STATIC = "static"                    # 静态阈值
    DYNAMIC = "dynamic"                  # 动态阈值
    ADAPTIVE = "adaptive"                # 自适应阈值
    SEASONAL = "seasonal"                # 季节性阈值
    BUSINESS_HOURS = "business_hours"    # 业务时间阈值
    PERCENTILE = "percentile"            # 百分位阈值
    ANOMALY_DETECTION = "anomaly_detection"  # 异常检测阈值
    COMPOSITE = "composite"              # 复合阈值


class ThresholdOperator(Enum):
    """阈值操作符"""
    GREATER_THAN = "gt"           # 大于
    GREATER_EQUAL = "gte"         # 大于等于
    LESS_THAN = "lt"             # 小于
    LESS_EQUAL = "lte"           # 小于等于
    EQUAL = "eq"                 # 等于
    NOT_EQUAL = "ne"             # 不等于
    BETWEEN = "between"          # 在范围内
    NOT_BETWEEN = "not_between"  # 不在范围内
    CHANGE_RATE = "change_rate"  # 变化率


class ThresholdSeverity(Enum):
    """阈值严重程度"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ThresholdCondition:
    """阈值条件"""
    operator: ThresholdOperator
    value: Union[float, List[float]]  # 单值或范围值
    duration: Optional[int] = None    # 持续时间（秒）
    evaluation_window: Optional[int] = None  # 评估窗口（秒）
    
    def evaluate(self, current_value: float, historical_values: List[float] = None) -> bool:
        """评估阈值条件"""
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
        elif self.operator == ThresholdOperator.NOT_BETWEEN:
            if isinstance(self.value, list) and len(self.value) == 2:
                return not (self.value[0] <= current_value <= self.value[1])
        elif self.operator == ThresholdOperator.CHANGE_RATE:
            if historical_values and len(historical_values) >= 2:
                old_value = historical_values[-2]
                if old_value != 0:
                    change_rate = abs((current_value - old_value) / old_value)
                    return change_rate > self.value
        
        return False


@dataclass
class BusinessHours:
    """业务时间配置"""
    start_hour: int = 9      # 开始小时
    end_hour: int = 18       # 结束小时
    weekdays: List[int] = None  # 工作日 (0-6, 0为周一)
    timezone: str = "UTC"    # 时区
    
    def __post_init__(self):
        if self.weekdays is None:
            self.weekdays = [0, 1, 2, 3, 4]  # 默认周一到周五
    
    def is_business_hours(self, dt: datetime) -> bool:
        """检查是否在业务时间内"""
        if dt.weekday() not in self.weekdays:
            return False
        
        return self.start_hour <= dt.hour < self.end_hour


@dataclass
class SeasonalPattern:
    """季节性模式"""
    pattern_type: str = "weekly"  # weekly, monthly, yearly
    multipliers: Dict[str, float] = None  # 时间段 -> 倍数
    
    def __post_init__(self):
        if self.multipliers is None:
            if self.pattern_type == "weekly":
                # 默认周末流量较低
                self.multipliers = {
                    "0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0,  # 工作日
                    "5": 0.7, "6": 0.6  # 周末
                }
            else:
                self.multipliers = {}
    
    def get_multiplier(self, dt: datetime) -> float:
        """获取时间点的倍数"""
        if self.pattern_type == "weekly":
            key = str(dt.weekday())
        elif self.pattern_type == "monthly":
            key = str(dt.day)
        elif self.pattern_type == "yearly":
            key = str(dt.month)
        else:
            return 1.0
        
        return self.multipliers.get(key, 1.0)


@dataclass
class AdaptiveConfig:
    """自适应配置"""
    learning_window: int = 3600 * 24 * 7  # 学习窗口（秒）
    sensitivity: float = 2.0              # 敏感度（标准差倍数）
    min_samples: int = 100                # 最小样本数
    update_interval: int = 3600           # 更新间隔（秒）
    outlier_threshold: float = 3.0        # 异常值阈值


@dataclass
class ThresholdConfig:
    """阈值配置"""
    id: str
    name: str
    metric_name: str
    threshold_type: ThresholdType
    conditions: Dict[ThresholdSeverity, ThresholdCondition]
    enabled: bool = True
    
    # 类型特定配置
    business_hours: Optional[BusinessHours] = None
    seasonal_pattern: Optional[SeasonalPattern] = None
    adaptive_config: Optional[AdaptiveConfig] = None
    percentile_config: Optional[Dict[str, Any]] = None
    composite_config: Optional[Dict[str, Any]] = None
    
    # 元数据
    description: str = ""
    tags: List[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = {
            'id': self.id,
            'name': self.name,
            'metric_name': self.metric_name,
            'threshold_type': self.threshold_type.value,
            'conditions': {
                severity.value: {
                    'operator': condition.operator.value,
                    'value': condition.value,
                    'duration': condition.duration,
                    'evaluation_window': condition.evaluation_window
                }
                for severity, condition in self.conditions.items()
            },
            'enabled': self.enabled,
            'description': self.description,
            'tags': self.tags,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        # 添加类型特定配置
        if self.business_hours:
            data['business_hours'] = asdict(self.business_hours)
        if self.seasonal_pattern:
            data['seasonal_pattern'] = asdict(self.seasonal_pattern)
        if self.adaptive_config:
            data['adaptive_config'] = asdict(self.adaptive_config)
        if self.percentile_config:
            data['percentile_config'] = self.percentile_config
        if self.composite_config:
            data['composite_config'] = self.composite_config
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThresholdConfig':
        """从字典创建"""
        conditions = {}
        for severity_str, condition_data in data.get('conditions', {}).items():
            severity = ThresholdSeverity(severity_str)
            condition = ThresholdCondition(
                operator=ThresholdOperator(condition_data['operator']),
                value=condition_data['value'],
                duration=condition_data.get('duration'),
                evaluation_window=condition_data.get('evaluation_window')
            )
            conditions[severity] = condition
        
        config = cls(
            id=data['id'],
            name=data['name'],
            metric_name=data['metric_name'],
            threshold_type=ThresholdType(data['threshold_type']),
            conditions=conditions,
            enabled=data.get('enabled', True),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else None
        )
        
        # 设置类型特定配置
        if 'business_hours' in data:
            config.business_hours = BusinessHours(**data['business_hours'])
        if 'seasonal_pattern' in data:
            config.seasonal_pattern = SeasonalPattern(**data['seasonal_pattern'])
        if 'adaptive_config' in data:
            config.adaptive_config = AdaptiveConfig(**data['adaptive_config'])
        if 'percentile_config' in data:
            config.percentile_config = data['percentile_config']
        if 'composite_config' in data:
            config.composite_config = data['composite_config']
        
        return config


@dataclass
class MetricDataPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


class ThresholdConfigManager:
    """阈值配置管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.configs: Dict[str, ThresholdConfig] = {}
        self.metric_data: Dict[str, List[MetricDataPoint]] = {}
        self.adaptive_thresholds: Dict[str, Dict[ThresholdSeverity, float]] = {}
        
        # 加载默认配置
        self._load_default_configs()
    
    def _load_default_configs(self):
        """加载默认阈值配置"""
        default_configs = [
            # CPU使用率阈值
            ThresholdConfig(
                id="cpu_usage",
                name="CPU使用率",
                metric_name="cpu_usage_percent",
                threshold_type=ThresholdType.STATIC,
                conditions={
                    ThresholdSeverity.CRITICAL: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=90.0,
                        duration=300
                    ),
                    ThresholdSeverity.HIGH: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=80.0,
                        duration=300
                    ),
                    ThresholdSeverity.MEDIUM: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=70.0,
                        duration=600
                    )
                },
                description="CPU使用率监控阈值"
            ),
            
            # 内存使用率阈值
            ThresholdConfig(
                id="memory_usage",
                name="内存使用率",
                metric_name="memory_usage_percent",
                threshold_type=ThresholdType.STATIC,
                conditions={
                    ThresholdSeverity.CRITICAL: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=95.0,
                        duration=180
                    ),
                    ThresholdSeverity.HIGH: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=85.0,
                        duration=300
                    )
                },
                description="内存使用率监控阈值"
            ),
            
            # API响应时间阈值（业务时间）
            ThresholdConfig(
                id="api_response_time_business",
                name="API响应时间（业务时间）",
                metric_name="api_response_time_ms",
                threshold_type=ThresholdType.BUSINESS_HOURS,
                conditions={
                    ThresholdSeverity.CRITICAL: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=5000.0,
                        duration=60
                    ),
                    ThresholdSeverity.HIGH: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=2000.0,
                        duration=120
                    ),
                    ThresholdSeverity.MEDIUM: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=1000.0,
                        duration=300
                    )
                },
                business_hours=BusinessHours(
                    start_hour=9,
                    end_hour=18,
                    weekdays=[0, 1, 2, 3, 4]
                ),
                description="业务时间内API响应时间监控"
            ),
            
            # 错误率阈值（自适应）
            ThresholdConfig(
                id="error_rate_adaptive",
                name="错误率（自适应）",
                metric_name="error_rate_percent",
                threshold_type=ThresholdType.ADAPTIVE,
                conditions={
                    ThresholdSeverity.CRITICAL: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=0.0,  # 将由自适应算法计算
                        duration=120
                    ),
                    ThresholdSeverity.HIGH: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=0.0,  # 将由自适应算法计算
                        duration=300
                    )
                },
                adaptive_config=AdaptiveConfig(
                    learning_window=3600 * 24 * 7,  # 7天学习窗口
                    sensitivity=2.5,
                    min_samples=200,
                    update_interval=3600
                ),
                description="基于历史数据的自适应错误率监控"
            ),
            
            # 请求量阈值（季节性）
            ThresholdConfig(
                id="request_volume_seasonal",
                name="请求量（季节性）",
                metric_name="request_count_per_minute",
                threshold_type=ThresholdType.SEASONAL,
                conditions={
                    ThresholdSeverity.HIGH: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=1000.0,  # 基准值，将根据季节性调整
                        duration=300
                    ),
                    ThresholdSeverity.MEDIUM: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=800.0,
                        duration=600
                    )
                },
                seasonal_pattern=SeasonalPattern(
                    pattern_type="weekly",
                    multipliers={
                        "0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0, "4": 1.0,  # 工作日
                        "5": 0.6, "6": 0.5  # 周末
                    }
                ),
                description="考虑季节性变化的请求量监控"
            ),
            
            # 数据库连接数阈值（百分位）
            ThresholdConfig(
                id="db_connections_percentile",
                name="数据库连接数（百分位）",
                metric_name="db_active_connections",
                threshold_type=ThresholdType.PERCENTILE,
                conditions={
                    ThresholdSeverity.CRITICAL: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=95.0,  # P95
                        duration=180
                    ),
                    ThresholdSeverity.HIGH: ThresholdCondition(
                        operator=ThresholdOperator.GREATER_THAN,
                        value=90.0,  # P90
                        duration=300
                    )
                },
                percentile_config={
                    "window_size": 3600,  # 1小时窗口
                    "percentiles": [90, 95, 99]
                },
                description="基于百分位的数据库连接数监控"
            )
        ]
        
        for config in default_configs:
            self.configs[config.id] = config
    
    async def add_config(self, config: ThresholdConfig) -> bool:
        """添加阈值配置"""
        if config.id in self.configs:
            self.logger.warning(f"阈值配置已存在: {config.id}")
            return False
        
        # 验证配置
        errors = await self.validate_config(config)
        if errors:
            self.logger.error(f"阈值配置验证失败: {errors}")
            return False
        
        self.configs[config.id] = config
        self.logger.info(f"添加阈值配置: {config.id}")
        return True
    
    async def update_config(self, config_id: str, config: ThresholdConfig) -> bool:
        """更新阈值配置"""
        if config_id not in self.configs:
            self.logger.warning(f"阈值配置不存在: {config_id}")
            return False
        
        # 验证配置
        errors = await self.validate_config(config)
        if errors:
            self.logger.error(f"阈值配置验证失败: {errors}")
            return False
        
        config.updated_at = datetime.now()
        self.configs[config_id] = config
        self.logger.info(f"更新阈值配置: {config_id}")
        return True
    
    async def remove_config(self, config_id: str) -> bool:
        """删除阈值配置"""
        if config_id not in self.configs:
            self.logger.warning(f"阈值配置不存在: {config_id}")
            return False
        
        del self.configs[config_id]
        
        # 清理相关数据
        if config_id in self.adaptive_thresholds:
            del self.adaptive_thresholds[config_id]
        
        self.logger.info(f"删除阈值配置: {config_id}")
        return True
    
    async def validate_config(self, config: ThresholdConfig) -> List[str]:
        """验证阈值配置"""
        errors = []
        
        # 基本验证
        if not config.id:
            errors.append("配置ID不能为空")
        
        if not config.name:
            errors.append("配置名称不能为空")
        
        if not config.metric_name:
            errors.append("指标名称不能为空")
        
        if not config.conditions:
            errors.append("阈值条件不能为空")
        
        # 验证条件
        for severity, condition in config.conditions.items():
            if condition.duration and condition.duration <= 0:
                errors.append(f"持续时间必须大于0: {severity.value}")
            
            if condition.evaluation_window and condition.evaluation_window <= 0:
                errors.append(f"评估窗口必须大于0: {severity.value}")
        
        # 验证类型特定配置
        if config.threshold_type == ThresholdType.BUSINESS_HOURS:
            if not config.business_hours:
                errors.append("业务时间阈值缺少业务时间配置")
            elif config.business_hours.start_hour >= config.business_hours.end_hour:
                errors.append("业务时间开始时间必须小于结束时间")
        
        if config.threshold_type == ThresholdType.ADAPTIVE:
            if not config.adaptive_config:
                errors.append("自适应阈值缺少自适应配置")
            elif config.adaptive_config.min_samples <= 0:
                errors.append("最小样本数必须大于0")
        
        if config.threshold_type == ThresholdType.PERCENTILE:
            if not config.percentile_config:
                errors.append("百分位阈值缺少百分位配置")
            elif 'percentiles' not in config.percentile_config:
                errors.append("百分位配置缺少percentiles字段")
        
        return errors
    
    async def add_metric_data(self, metric_name: str, data_point: MetricDataPoint):
        """添加指标数据"""
        if metric_name not in self.metric_data:
            self.metric_data[metric_name] = []
        
        self.metric_data[metric_name].append(data_point)
        
        # 限制数据量，保留最近的数据
        max_points = 10000
        if len(self.metric_data[metric_name]) > max_points:
            self.metric_data[metric_name] = self.metric_data[metric_name][-max_points:]
    
    async def evaluate_thresholds(self, metric_name: str, current_value: float, 
                                timestamp: datetime = None) -> List[Tuple[str, ThresholdSeverity, bool]]:
        """评估阈值"""
        if timestamp is None:
            timestamp = datetime.now()
        
        results = []
        
        # 查找相关的阈值配置
        relevant_configs = [
            config for config in self.configs.values()
            if config.metric_name == metric_name and config.enabled
        ]
        
        for config in relevant_configs:
            # 获取当前有效阈值
            effective_thresholds = await self._get_effective_thresholds(config, timestamp)
            
            # 获取历史数据
            historical_values = self._get_historical_values(metric_name, timestamp)
            
            # 评估每个严重程度的条件
            for severity, condition in config.conditions.items():
                # 使用有效阈值更新条件值
                if config.id in effective_thresholds and severity in effective_thresholds[config.id]:
                    condition.value = effective_thresholds[config.id][severity]
                
                # 评估条件
                is_triggered = condition.evaluate(current_value, historical_values)
                results.append((config.id, severity, is_triggered))
        
        return results
    
    async def _get_effective_thresholds(self, config: ThresholdConfig, 
                                      timestamp: datetime) -> Dict[str, Dict[ThresholdSeverity, float]]:
        """获取有效阈值"""
        effective_thresholds = {config.id: {}}
        
        if config.threshold_type == ThresholdType.STATIC:
            # 静态阈值直接使用配置值
            for severity, condition in config.conditions.items():
                effective_thresholds[config.id][severity] = condition.value
        
        elif config.threshold_type == ThresholdType.BUSINESS_HOURS:
            # 业务时间阈值
            multiplier = 1.0
            if config.business_hours and not config.business_hours.is_business_hours(timestamp):
                multiplier = 1.5  # 非业务时间放宽阈值
            
            for severity, condition in config.conditions.items():
                effective_thresholds[config.id][severity] = condition.value * multiplier
        
        elif config.threshold_type == ThresholdType.SEASONAL:
            # 季节性阈值
            multiplier = 1.0
            if config.seasonal_pattern:
                multiplier = config.seasonal_pattern.get_multiplier(timestamp)
            
            for severity, condition in config.conditions.items():
                effective_thresholds[config.id][severity] = condition.value * multiplier
        
        elif config.threshold_type == ThresholdType.ADAPTIVE:
            # 自适应阈值
            if config.id in self.adaptive_thresholds:
                effective_thresholds[config.id] = self.adaptive_thresholds[config.id]
            else:
                # 使用默认值
                for severity, condition in config.conditions.items():
                    effective_thresholds[config.id][severity] = condition.value
        
        elif config.threshold_type == ThresholdType.PERCENTILE:
            # 百分位阈值
            percentile_thresholds = await self._calculate_percentile_thresholds(config, timestamp)
            effective_thresholds[config.id] = percentile_thresholds
        
        return effective_thresholds
    
    async def _calculate_percentile_thresholds(self, config: ThresholdConfig, 
                                             timestamp: datetime) -> Dict[ThresholdSeverity, float]:
        """计算百分位阈值"""
        thresholds = {}
        
        if not config.percentile_config:
            return thresholds
        
        window_size = config.percentile_config.get('window_size', 3600)
        percentiles = config.percentile_config.get('percentiles', [90, 95, 99])
        
        # 获取窗口内的数据
        start_time = timestamp - timedelta(seconds=window_size)
        window_data = [
            dp.value for dp in self.metric_data.get(config.metric_name, [])
            if start_time <= dp.timestamp <= timestamp
        ]
        
        if len(window_data) < 10:  # 数据不足
            for severity, condition in config.conditions.items():
                thresholds[severity] = condition.value
            return thresholds
        
        # 计算百分位值
        window_data.sort()
        
        severity_percentile_map = {
            ThresholdSeverity.CRITICAL: 99,
            ThresholdSeverity.HIGH: 95,
            ThresholdSeverity.MEDIUM: 90,
            ThresholdSeverity.LOW: 75
        }
        
        for severity, condition in config.conditions.items():
            percentile = severity_percentile_map.get(severity, 90)
            if percentile in percentiles:
                index = int(len(window_data) * percentile / 100)
                thresholds[severity] = window_data[min(index, len(window_data) - 1)]
            else:
                thresholds[severity] = condition.value
        
        return thresholds
    
    def _get_historical_values(self, metric_name: str, timestamp: datetime, 
                             window_seconds: int = 3600) -> List[float]:
        """获取历史值"""
        start_time = timestamp - timedelta(seconds=window_seconds)
        
        historical_data = [
            dp.value for dp in self.metric_data.get(metric_name, [])
            if start_time <= dp.timestamp <= timestamp
        ]
        
        return historical_data
    
    async def update_adaptive_thresholds(self):
        """更新自适应阈值"""
        for config in self.configs.values():
            if config.threshold_type != ThresholdType.ADAPTIVE or not config.adaptive_config:
                continue
            
            metric_name = config.metric_name
            if metric_name not in self.metric_data:
                continue
            
            # 获取学习窗口内的数据
            now = datetime.now()
            learning_window = config.adaptive_config.learning_window
            start_time = now - timedelta(seconds=learning_window)
            
            learning_data = [
                dp.value for dp in self.metric_data[metric_name]
                if start_time <= dp.timestamp <= now
            ]
            
            if len(learning_data) < config.adaptive_config.min_samples:
                continue
            
            # 计算统计指标
            mean_value = statistics.mean(learning_data)
            std_value = statistics.stdev(learning_data) if len(learning_data) > 1 else 0
            
            # 移除异常值
            outlier_threshold = config.adaptive_config.outlier_threshold
            filtered_data = [
                value for value in learning_data
                if abs(value - mean_value) <= outlier_threshold * std_value
            ]
            
            if len(filtered_data) < config.adaptive_config.min_samples:
                continue
            
            # 重新计算统计指标
            mean_value = statistics.mean(filtered_data)
            std_value = statistics.stdev(filtered_data) if len(filtered_data) > 1 else 0
            
            # 计算自适应阈值
            sensitivity = config.adaptive_config.sensitivity
            
            adaptive_thresholds = {}
            severity_multipliers = {
                ThresholdSeverity.CRITICAL: 3.0,
                ThresholdSeverity.HIGH: 2.5,
                ThresholdSeverity.MEDIUM: 2.0,
                ThresholdSeverity.LOW: 1.5
            }
            
            for severity in config.conditions.keys():
                multiplier = severity_multipliers.get(severity, sensitivity)
                threshold = mean_value + multiplier * std_value
                adaptive_thresholds[severity] = threshold
            
            self.adaptive_thresholds[config.id] = adaptive_thresholds
            
            self.logger.info(f"更新自适应阈值: {config.id}, 均值={mean_value:.2f}, 标准差={std_value:.2f}")
    
    def get_config(self, config_id: str) -> Optional[ThresholdConfig]:
        """获取阈值配置"""
        return self.configs.get(config_id)
    
    def get_configs(self, metric_name: Optional[str] = None, 
                   threshold_type: Optional[ThresholdType] = None,
                   enabled_only: bool = True) -> List[ThresholdConfig]:
        """获取阈值配置列表"""
        configs = list(self.configs.values())
        
        if enabled_only:
            configs = [c for c in configs if c.enabled]
        
        if metric_name:
            configs = [c for c in configs if c.metric_name == metric_name]
        
        if threshold_type:
            configs = [c for c in configs if c.threshold_type == threshold_type]
        
        return configs
    
    async def get_threshold_statistics(self) -> Dict[str, Any]:
        """获取阈值统计信息"""
        total_configs = len(self.configs)
        enabled_configs = len([c for c in self.configs.values() if c.enabled])
        
        # 按类型统计
        type_stats = {}
        for threshold_type in ThresholdType:
            count = len([c for c in self.configs.values() if c.threshold_type == threshold_type])
            type_stats[threshold_type.value] = count
        
        # 按指标统计
        metric_stats = {}
        for config in self.configs.values():
            metric_name = config.metric_name
            metric_stats[metric_name] = metric_stats.get(metric_name, 0) + 1
        
        # 自适应阈值统计
        adaptive_count = len(self.adaptive_thresholds)
        
        return {
            "total_configs": total_configs,
            "enabled_configs": enabled_configs,
            "type_distribution": type_stats,
            "metric_distribution": metric_stats,
            "adaptive_thresholds_count": adaptive_count,
            "total_metric_data_points": sum(len(data) for data in self.metric_data.values())
        }
    
    async def export_configs(self, export_path: str) -> bool:
        """导出阈值配置"""
        try:
            export_data = {
                "threshold_configs": [config.to_dict() for config in self.configs.values()],
                "adaptive_thresholds": self.adaptive_thresholds,
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功导出阈值配置到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出阈值配置失败: {e}")
            return False
    
    async def import_configs(self, import_path: str, overwrite: bool = False) -> bool:
        """导入阈值配置"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            configs_data = import_data.get("threshold_configs", [])
            imported_count = 0
            
            for config_data in configs_data:
                config = ThresholdConfig.from_dict(config_data)
                
                if config.id in self.configs and not overwrite:
                    self.logger.warning(f"阈值配置已存在，跳过: {config.id}")
                    continue
                
                await self.add_config(config)
                imported_count += 1
            
            # 导入自适应阈值
            if "adaptive_thresholds" in import_data:
                self.adaptive_thresholds.update(import_data["adaptive_thresholds"])
            
            self.logger.info(f"成功导入 {imported_count} 个阈值配置")
            return True
            
        except Exception as e:
            self.logger.error(f"导入阈值配置失败: {e}")
            return False
    
    async def cleanup_old_data(self, retention_days: int = 30):
        """清理过期数据"""
        cutoff_time = datetime.now() - timedelta(days=retention_days)
        
        total_cleaned = 0
        for metric_name in self.metric_data:
            old_count = len(self.metric_data[metric_name])
            self.metric_data[metric_name] = [
                dp for dp in self.metric_data[metric_name]
                if dp.timestamp > cutoff_time
            ]
            cleaned = old_count - len(self.metric_data[metric_name])
            total_cleaned += cleaned
        
        self.logger.info(f"清理了 {total_cleaned} 个过期数据点")