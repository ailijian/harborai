"""
告警阈值配置管理器

负责管理告警阈值的配置、动态调整、自适应学习和阈值优化，
支持多种阈值类型、时间窗口和业务场景。
"""

import json
import yaml
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import statistics
import math
from collections import defaultdict, deque
import asyncio


class ThresholdType(Enum):
    """阈值类型"""
    STATIC = "static"              # 静态阈值
    DYNAMIC = "dynamic"            # 动态阈值
    ADAPTIVE = "adaptive"          # 自适应阈值
    PERCENTILE = "percentile"      # 百分位阈值
    SEASONAL = "seasonal"          # 季节性阈值
    BUSINESS_HOURS = "business_hours"  # 工作时间阈值
    ANOMALY_BASED = "anomaly_based"    # 基于异常检测的阈值
    COMPOSITE = "composite"        # 复合阈值


class ThresholdDirection(Enum):
    """阈值方向"""
    UPPER = "upper"               # 上限阈值
    LOWER = "lower"               # 下限阈值
    BOTH = "both"                 # 双向阈值


class ThresholdSeverity(Enum):
    """阈值严重级别"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AggregationType(Enum):
    """聚合类型"""
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    SUM = "sum"
    COUNT = "count"
    P50 = "p50"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"


@dataclass
class TimeWindow:
    """时间窗口"""
    duration: int                 # 持续时间（秒）
    evaluation_points: int = 1    # 评估点数
    slide_interval: int = 60      # 滑动间隔（秒）
    
    def is_valid_duration(self, start_time: datetime, end_time: datetime) -> bool:
        """检查时间窗口是否有效"""
        duration_seconds = (end_time - start_time).total_seconds()
        return duration_seconds >= self.duration


@dataclass
class BusinessHours:
    """工作时间配置"""
    weekday_start: int = 9        # 工作日开始时间
    weekday_end: int = 18         # 工作日结束时间
    weekend_start: int = 10       # 周末开始时间
    weekend_end: int = 16         # 周末结束时间
    timezone: str = "UTC"         # 时区
    holidays: List[str] = field(default_factory=list)  # 节假日列表
    
    def is_business_hours(self, timestamp: datetime) -> bool:
        """检查是否在工作时间内"""
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        hour = timestamp.hour
        
        # 检查是否为节假日
        date_str = timestamp.strftime("%Y-%m-%d")
        if date_str in self.holidays:
            return False
        
        # 工作日 (Monday-Friday)
        if weekday < 5:
            return self.weekday_start <= hour < self.weekday_end
        # 周末 (Saturday-Sunday)
        else:
            return self.weekend_start <= hour < self.weekend_end


@dataclass
class SeasonalPattern:
    """季节性模式"""
    pattern_type: str = "weekly"   # 模式类型: daily, weekly, monthly, yearly
    multipliers: Dict[str, float] = field(default_factory=dict)  # 时间段乘数
    base_threshold: float = 0.0    # 基础阈值
    
    def get_seasonal_multiplier(self, timestamp: datetime) -> float:
        """获取季节性乘数"""
        if self.pattern_type == "daily":
            key = str(timestamp.hour)
        elif self.pattern_type == "weekly":
            key = str(timestamp.weekday())
        elif self.pattern_type == "monthly":
            key = str(timestamp.day)
        elif self.pattern_type == "yearly":
            key = str(timestamp.month)
        else:
            return 1.0
        
        return self.multipliers.get(key, 1.0)
    
    def calculate_seasonal_threshold(self, timestamp: datetime) -> float:
        """计算季节性阈值"""
        multiplier = self.get_seasonal_multiplier(timestamp)
        return self.base_threshold * multiplier


@dataclass
class AdaptiveConfig:
    """自适应配置"""
    learning_rate: float = 0.1     # 学习率
    adaptation_window: int = 3600  # 适应窗口（秒）
    min_samples: int = 100         # 最小样本数
    sensitivity: float = 0.95      # 敏感度
    decay_factor: float = 0.99     # 衰减因子
    outlier_threshold: float = 3.0 # 异常值阈值（标准差倍数）
    
    def should_adapt(self, sample_count: int, last_adaptation: datetime) -> bool:
        """判断是否应该进行适应"""
        if sample_count < self.min_samples:
            return False
        
        time_since_adaptation = (datetime.now() - last_adaptation).total_seconds()
        return time_since_adaptation >= self.adaptation_window


@dataclass
class ThresholdValue:
    """阈值值"""
    value: float
    confidence: float = 1.0        # 置信度
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "manual"         # 来源: manual, adaptive, seasonal, etc.
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """检查阈值是否过期"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > ttl_seconds


@dataclass
class ThresholdConfig:
    """阈值配置"""
    metric_name: str
    threshold_type: ThresholdType
    direction: ThresholdDirection
    severity: ThresholdSeverity
    
    # 基础配置
    static_value: Optional[float] = None
    upper_threshold: Optional[ThresholdValue] = None
    lower_threshold: Optional[ThresholdValue] = None
    
    # 时间配置
    time_window: TimeWindow = field(default_factory=lambda: TimeWindow(duration=300))
    aggregation: AggregationType = AggregationType.AVG
    
    # 动态配置
    percentile: float = 95.0       # 百分位数
    adaptive_config: Optional[AdaptiveConfig] = None
    seasonal_pattern: Optional[SeasonalPattern] = None
    business_hours: Optional[BusinessHours] = None
    
    # 复合配置
    composite_rules: List[str] = field(default_factory=list)  # 复合规则列表
    composite_operator: str = "AND"  # 复合操作符: AND, OR
    
    # 元数据
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 统计信息
    evaluation_count: int = 0
    trigger_count: int = 0
    last_triggered: Optional[datetime] = None
    last_evaluated: Optional[datetime] = None
    
    def get_effective_threshold(self, timestamp: datetime = None,
                              historical_data: List[float] = None) -> Tuple[Optional[float], Optional[float]]:
        """获取有效阈值"""
        if not self.enabled:
            return None, None
        
        if timestamp is None:
            timestamp = datetime.now()
        
        upper_value = None
        lower_value = None
        
        if self.threshold_type == ThresholdType.STATIC:
            if self.direction in [ThresholdDirection.UPPER, ThresholdDirection.BOTH]:
                upper_value = self.static_value
            if self.direction in [ThresholdDirection.LOWER, ThresholdDirection.BOTH]:
                lower_value = self.static_value
        
        elif self.threshold_type == ThresholdType.PERCENTILE:
            if historical_data:
                threshold_value = self._calculate_percentile_threshold(historical_data)
                if self.direction in [ThresholdDirection.UPPER, ThresholdDirection.BOTH]:
                    upper_value = threshold_value
                if self.direction in [ThresholdDirection.LOWER, ThresholdDirection.BOTH]:
                    lower_value = threshold_value
        
        elif self.threshold_type == ThresholdType.SEASONAL:
            if self.seasonal_pattern:
                threshold_value = self.seasonal_pattern.calculate_seasonal_threshold(timestamp)
                if self.direction in [ThresholdDirection.UPPER, ThresholdDirection.BOTH]:
                    upper_value = threshold_value
                if self.direction in [ThresholdDirection.LOWER, ThresholdDirection.BOTH]:
                    lower_value = threshold_value
        
        elif self.threshold_type == ThresholdType.BUSINESS_HOURS:
            if self.business_hours:
                is_business = self.business_hours.is_business_hours(timestamp)
                base_value = self.static_value or 0
                multiplier = 1.0 if is_business else 0.5  # 非工作时间降低阈值
                threshold_value = base_value * multiplier
                
                if self.direction in [ThresholdDirection.UPPER, ThresholdDirection.BOTH]:
                    upper_value = threshold_value
                if self.direction in [ThresholdDirection.LOWER, ThresholdDirection.BOTH]:
                    lower_value = threshold_value
        
        elif self.threshold_type == ThresholdType.ADAPTIVE:
            if self.upper_threshold:
                upper_value = self.upper_threshold.value
            if self.lower_threshold:
                lower_value = self.lower_threshold.value
        
        return upper_value, lower_value
    
    def _calculate_percentile_threshold(self, data: List[float]) -> float:
        """计算百分位阈值"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * self.percentile / 100)
        index = min(index, len(sorted_data) - 1)
        
        return sorted_data[index]
    
    def evaluate(self, value: float, timestamp: datetime = None,
                historical_data: List[float] = None) -> Tuple[bool, str]:
        """评估阈值"""
        if not self.enabled:
            return False, "阈值配置已禁用"
        
        if timestamp is None:
            timestamp = datetime.now()
        
        self.evaluation_count += 1
        self.last_evaluated = timestamp
        
        upper_threshold, lower_threshold = self.get_effective_threshold(timestamp, historical_data)
        
        # 检查上限阈值
        if upper_threshold is not None and value > upper_threshold:
            self.trigger_count += 1
            self.last_triggered = timestamp
            return True, f"值 {value} 超过上限阈值 {upper_threshold}"
        
        # 检查下限阈值
        if lower_threshold is not None and value < lower_threshold:
            self.trigger_count += 1
            self.last_triggered = timestamp
            return True, f"值 {value} 低于下限阈值 {lower_threshold}"
        
        return False, "阈值检查通过"
    
    def update_adaptive_threshold(self, values: List[float], timestamp: datetime = None):
        """更新自适应阈值"""
        if self.threshold_type != ThresholdType.ADAPTIVE or not self.adaptive_config:
            return
        
        if not values or len(values) < self.adaptive_config.min_samples:
            return
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # 过滤异常值
        filtered_values = self._filter_outliers(values)
        
        if not filtered_values:
            return
        
        # 计算新阈值
        mean = statistics.mean(filtered_values)
        std_dev = statistics.stdev(filtered_values) if len(filtered_values) > 1 else 0
        
        # 计算置信区间
        confidence_factor = 2.0  # 95% 置信区间
        margin = confidence_factor * std_dev
        
        # 更新阈值
        if self.direction in [ThresholdDirection.UPPER, ThresholdDirection.BOTH]:
            new_upper = mean + margin
            if self.upper_threshold:
                # 使用学习率进行平滑更新
                old_value = self.upper_threshold.value
                new_value = old_value * (1 - self.adaptive_config.learning_rate) + new_upper * self.adaptive_config.learning_rate
                self.upper_threshold.value = new_value
            else:
                self.upper_threshold = ThresholdValue(
                    value=new_upper,
                    timestamp=timestamp,
                    source="adaptive"
                )
        
        if self.direction in [ThresholdDirection.LOWER, ThresholdDirection.BOTH]:
            new_lower = mean - margin
            if self.lower_threshold:
                old_value = self.lower_threshold.value
                new_value = old_value * (1 - self.adaptive_config.learning_rate) + new_lower * self.adaptive_config.learning_rate
                self.lower_threshold.value = new_value
            else:
                self.lower_threshold = ThresholdValue(
                    value=new_lower,
                    timestamp=timestamp,
                    source="adaptive"
                )
        
        self.updated_at = timestamp
    
    def _filter_outliers(self, values: List[float]) -> List[float]:
        """过滤异常值"""
        if not self.adaptive_config or len(values) < 3:
            return values
        
        mean = statistics.mean(values)
        std_dev = statistics.stdev(values)
        
        if std_dev == 0:
            return values
        
        threshold = self.adaptive_config.outlier_threshold
        filtered = []
        
        for value in values:
            z_score = abs(value - mean) / std_dev
            if z_score <= threshold:
                filtered.append(value)
        
        return filtered if filtered else values


@dataclass
class MetricDataPoint:
    """指标数据点"""
    metric_name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = field(default_factory=dict)
    
    def age_seconds(self) -> float:
        """获取数据点年龄（秒）"""
        return (datetime.now() - self.timestamp).total_seconds()


class AlertThresholdConfigManager:
    """告警阈值配置管理器"""
    
    def __init__(self, config_dir: str = "config/alerts"):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置存储
        self.threshold_configs: Dict[str, ThresholdConfig] = {}
        self.metric_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # 配置文件路径
        self.config_file = self.config_dir / "thresholds.json"
        
        # 自适应更新任务
        self.adaptive_update_task: Optional[asyncio.Task] = None
        self.update_interval = 300  # 5分钟
        
        # 加载配置
        self._load_default_configs()
        self._load_configs()
    
    def _load_default_configs(self):
        """加载默认阈值配置"""
        default_configs = [
            # CPU使用率阈值
            ThresholdConfig(
                metric_name="cpu_usage_percent",
                threshold_type=ThresholdType.STATIC,
                direction=ThresholdDirection.UPPER,
                severity=ThresholdSeverity.HIGH,
                static_value=80.0,
                time_window=TimeWindow(duration=300, evaluation_points=3),
                aggregation=AggregationType.AVG,
                description="CPU使用率超过80%告警",
                tags=["system", "cpu"]
            ),
            
            # 内存使用率阈值
            ThresholdConfig(
                metric_name="memory_usage_percent",
                threshold_type=ThresholdType.ADAPTIVE,
                direction=ThresholdDirection.UPPER,
                severity=ThresholdSeverity.HIGH,
                adaptive_config=AdaptiveConfig(
                    learning_rate=0.1,
                    adaptation_window=3600,
                    min_samples=100,
                    sensitivity=0.95
                ),
                time_window=TimeWindow(duration=300, evaluation_points=2),
                aggregation=AggregationType.AVG,
                description="内存使用率自适应阈值告警",
                tags=["system", "memory"]
            ),
            
            # API响应时间阈值
            ThresholdConfig(
                metric_name="api_response_time_ms",
                threshold_type=ThresholdType.PERCENTILE,
                direction=ThresholdDirection.UPPER,
                severity=ThresholdSeverity.MEDIUM,
                percentile=95.0,
                time_window=TimeWindow(duration=600, evaluation_points=5),
                aggregation=AggregationType.P95,
                description="API响应时间P95阈值告警",
                tags=["application", "api", "performance"]
            ),
            
            # 错误率阈值
            ThresholdConfig(
                metric_name="error_rate_percent",
                threshold_type=ThresholdType.BUSINESS_HOURS,
                direction=ThresholdDirection.UPPER,
                severity=ThresholdSeverity.CRITICAL,
                static_value=5.0,
                business_hours=BusinessHours(
                    weekday_start=9,
                    weekday_end=18,
                    weekend_start=10,
                    weekend_end=16
                ),
                time_window=TimeWindow(duration=180, evaluation_points=2),
                aggregation=AggregationType.AVG,
                description="错误率工作时间阈值告警",
                tags=["application", "error"]
            ),
            
            # 请求量季节性阈值
            ThresholdConfig(
                metric_name="request_count_per_minute",
                threshold_type=ThresholdType.SEASONAL,
                direction=ThresholdDirection.BOTH,
                severity=ThresholdSeverity.MEDIUM,
                seasonal_pattern=SeasonalPattern(
                    pattern_type="daily",
                    base_threshold=1000.0,
                    multipliers={
                        "9": 1.5,   # 9点高峰
                        "10": 1.3,
                        "11": 1.2,
                        "14": 1.4,  # 14点高峰
                        "15": 1.3,
                        "20": 1.1,  # 20点小高峰
                        "21": 1.0,
                        "22": 0.8,
                        "23": 0.6,
                        "0": 0.4,   # 夜间低谷
                        "1": 0.3,
                        "2": 0.3,
                        "3": 0.3,
                        "4": 0.3,
                        "5": 0.4,
                        "6": 0.6,
                        "7": 0.8,
                        "8": 1.0
                    }
                ),
                time_window=TimeWindow(duration=300, evaluation_points=3),
                aggregation=AggregationType.AVG,
                description="请求量季节性阈值告警",
                tags=["application", "traffic"]
            ),
            
            # 数据库连接数阈值
            ThresholdConfig(
                metric_name="db_connection_count",
                threshold_type=ThresholdType.STATIC,
                direction=ThresholdDirection.UPPER,
                severity=ThresholdSeverity.HIGH,
                static_value=80.0,
                time_window=TimeWindow(duration=120, evaluation_points=2),
                aggregation=AggregationType.MAX,
                description="数据库连接数阈值告警",
                tags=["database", "connection"]
            ),
            
            # 磁盘使用率阈值
            ThresholdConfig(
                metric_name="disk_usage_percent",
                threshold_type=ThresholdType.STATIC,
                direction=ThresholdDirection.UPPER,
                severity=ThresholdSeverity.CRITICAL,
                static_value=90.0,
                time_window=TimeWindow(duration=600, evaluation_points=1),
                aggregation=AggregationType.MAX,
                description="磁盘使用率阈值告警",
                tags=["system", "disk"]
            ),
            
            # Token使用量异常检测
            ThresholdConfig(
                metric_name="token_usage_per_hour",
                threshold_type=ThresholdType.ADAPTIVE,
                direction=ThresholdDirection.UPPER,
                severity=ThresholdSeverity.MEDIUM,
                adaptive_config=AdaptiveConfig(
                    learning_rate=0.05,
                    adaptation_window=7200,  # 2小时
                    min_samples=200,
                    sensitivity=0.98,
                    outlier_threshold=2.5
                ),
                time_window=TimeWindow(duration=3600, evaluation_points=1),
                aggregation=AggregationType.SUM,
                description="Token使用量异常检测",
                tags=["application", "token", "cost"]
            ),
            
            # 成本飙升检测
            ThresholdConfig(
                metric_name="cost_per_hour_usd",
                threshold_type=ThresholdType.PERCENTILE,
                direction=ThresholdDirection.UPPER,
                severity=ThresholdSeverity.HIGH,
                percentile=90.0,
                time_window=TimeWindow(duration=3600, evaluation_points=1),
                aggregation=AggregationType.SUM,
                description="成本飙升检测",
                tags=["application", "cost", "billing"]
            )
        ]
        
        for config in default_configs:
            self.threshold_configs[config.metric_name] = config
    
    def _load_configs(self):
        """加载阈值配置"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    configs_data = json.load(f)
                
                for config_data in configs_data:
                    config = self._dict_to_config(config_data)
                    if config:
                        self.threshold_configs[config.metric_name] = config
                
                self.logger.info(f"加载了 {len(self.threshold_configs)} 个阈值配置")
                
            except Exception as e:
                self.logger.error(f"加载阈值配置失败: {e}")
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> Optional[ThresholdConfig]:
        """将字典转换为配置对象"""
        try:
            # 转换枚举类型
            config_data["threshold_type"] = ThresholdType(config_data["threshold_type"])
            config_data["direction"] = ThresholdDirection(config_data["direction"])
            config_data["severity"] = ThresholdSeverity(config_data["severity"])
            config_data["aggregation"] = AggregationType(config_data.get("aggregation", "avg"))
            
            # 转换日期时间
            if "created_at" in config_data:
                config_data["created_at"] = datetime.fromisoformat(config_data["created_at"])
            if "updated_at" in config_data:
                config_data["updated_at"] = datetime.fromisoformat(config_data["updated_at"])
            if "last_triggered" in config_data and config_data["last_triggered"]:
                config_data["last_triggered"] = datetime.fromisoformat(config_data["last_triggered"])
            if "last_evaluated" in config_data and config_data["last_evaluated"]:
                config_data["last_evaluated"] = datetime.fromisoformat(config_data["last_evaluated"])
            
            # 转换时间窗口
            if "time_window" in config_data:
                config_data["time_window"] = TimeWindow(**config_data["time_window"])
            
            # 转换阈值值
            if "upper_threshold" in config_data and config_data["upper_threshold"]:
                threshold_data = config_data["upper_threshold"]
                if "timestamp" in threshold_data:
                    threshold_data["timestamp"] = datetime.fromisoformat(threshold_data["timestamp"])
                config_data["upper_threshold"] = ThresholdValue(**threshold_data)
            
            if "lower_threshold" in config_data and config_data["lower_threshold"]:
                threshold_data = config_data["lower_threshold"]
                if "timestamp" in threshold_data:
                    threshold_data["timestamp"] = datetime.fromisoformat(threshold_data["timestamp"])
                config_data["lower_threshold"] = ThresholdValue(**threshold_data)
            
            # 转换自适应配置
            if "adaptive_config" in config_data and config_data["adaptive_config"]:
                config_data["adaptive_config"] = AdaptiveConfig(**config_data["adaptive_config"])
            
            # 转换季节性模式
            if "seasonal_pattern" in config_data and config_data["seasonal_pattern"]:
                config_data["seasonal_pattern"] = SeasonalPattern(**config_data["seasonal_pattern"])
            
            # 转换工作时间
            if "business_hours" in config_data and config_data["business_hours"]:
                config_data["business_hours"] = BusinessHours(**config_data["business_hours"])
            
            return ThresholdConfig(**config_data)
            
        except Exception as e:
            self.logger.error(f"转换阈值配置失败: {e}")
            return None
    
    def _config_to_dict(self, config: ThresholdConfig) -> Dict[str, Any]:
        """将配置对象转换为字典"""
        config_dict = asdict(config)
        
        # 转换枚举为字符串
        config_dict["threshold_type"] = config.threshold_type.value
        config_dict["direction"] = config.direction.value
        config_dict["severity"] = config.severity.value
        config_dict["aggregation"] = config.aggregation.value
        
        # 转换日期时间为字符串
        if config.created_at:
            config_dict["created_at"] = config.created_at.isoformat()
        if config.updated_at:
            config_dict["updated_at"] = config.updated_at.isoformat()
        if config.last_triggered:
            config_dict["last_triggered"] = config.last_triggered.isoformat()
        if config.last_evaluated:
            config_dict["last_evaluated"] = config.last_evaluated.isoformat()
        
        # 转换阈值值
        if config.upper_threshold:
            threshold_dict = asdict(config.upper_threshold)
            threshold_dict["timestamp"] = config.upper_threshold.timestamp.isoformat()
            config_dict["upper_threshold"] = threshold_dict
        
        if config.lower_threshold:
            threshold_dict = asdict(config.lower_threshold)
            threshold_dict["timestamp"] = config.lower_threshold.timestamp.isoformat()
            config_dict["lower_threshold"] = threshold_dict
        
        return config_dict
    
    async def save_configs(self) -> bool:
        """保存阈值配置"""
        try:
            configs_data = [self._config_to_dict(config) for config in self.threshold_configs.values()]
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(configs_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"保存了 {len(self.threshold_configs)} 个阈值配置")
            return True
            
        except Exception as e:
            self.logger.error(f"保存阈值配置失败: {e}")
            return False
    
    async def add_config(self, config: ThresholdConfig) -> bool:
        """添加阈值配置"""
        if config.metric_name in self.threshold_configs:
            self.logger.warning(f"阈值配置已存在: {config.metric_name}")
            return False
        
        self.threshold_configs[config.metric_name] = config
        await self.save_configs()
        
        self.logger.info(f"添加阈值配置: {config.metric_name}")
        return True
    
    async def update_config(self, metric_name: str, config: ThresholdConfig) -> bool:
        """更新阈值配置"""
        if metric_name not in self.threshold_configs:
            self.logger.warning(f"阈值配置不存在: {metric_name}")
            return False
        
        config.updated_at = datetime.now()
        self.threshold_configs[metric_name] = config
        await self.save_configs()
        
        self.logger.info(f"更新阈值配置: {metric_name}")
        return True
    
    async def remove_config(self, metric_name: str) -> bool:
        """删除阈值配置"""
        if metric_name not in self.threshold_configs:
            self.logger.warning(f"阈值配置不存在: {metric_name}")
            return False
        
        del self.threshold_configs[metric_name]
        await self.save_configs()
        
        self.logger.info(f"删除阈值配置: {metric_name}")
        return True
    
    def get_config(self, metric_name: str) -> Optional[ThresholdConfig]:
        """获取阈值配置"""
        return self.threshold_configs.get(metric_name)
    
    def get_configs(self, threshold_type: Optional[ThresholdType] = None,
                   severity: Optional[ThresholdSeverity] = None,
                   tags: Optional[List[str]] = None,
                   enabled_only: bool = True) -> List[ThresholdConfig]:
        """获取阈值配置列表"""
        configs = list(self.threshold_configs.values())
        
        if enabled_only:
            configs = [c for c in configs if c.enabled]
        
        if threshold_type:
            configs = [c for c in configs if c.threshold_type == threshold_type]
        
        if severity:
            configs = [c for c in configs if c.severity == severity]
        
        if tags:
            configs = [c for c in configs if any(tag in c.tags for tag in tags)]
        
        return configs
    
    async def add_metric_data(self, metric_name: str, value: float,
                            timestamp: datetime = None, tags: Dict[str, str] = None):
        """添加指标数据"""
        if timestamp is None:
            timestamp = datetime.now()
        
        if tags is None:
            tags = {}
        
        data_point = MetricDataPoint(
            metric_name=metric_name,
            value=value,
            timestamp=timestamp,
            tags=tags
        )
        
        self.metric_data[metric_name].append(data_point)
        
        # 如果有自适应配置，检查是否需要更新阈值
        config = self.threshold_configs.get(metric_name)
        if (config and config.threshold_type == ThresholdType.ADAPTIVE and 
            config.adaptive_config):
            
            # 获取最近的数据点
            recent_data = [
                dp.value for dp in self.metric_data[metric_name]
                if dp.age_seconds() <= config.adaptive_config.adaptation_window
            ]
            
            if len(recent_data) >= config.adaptive_config.min_samples:
                config.update_adaptive_threshold(recent_data, timestamp)
    
    def get_metric_data(self, metric_name: str, 
                       time_window_seconds: int = 3600) -> List[MetricDataPoint]:
        """获取指标数据"""
        if metric_name not in self.metric_data:
            return []
        
        cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
        
        return [
            dp for dp in self.metric_data[metric_name]
            if dp.timestamp >= cutoff_time
        ]
    
    async def evaluate_threshold(self, metric_name: str, value: float,
                               timestamp: datetime = None) -> Tuple[bool, str, Optional[ThresholdConfig]]:
        """评估阈值"""
        config = self.threshold_configs.get(metric_name)
        if not config:
            return False, f"未找到指标 {metric_name} 的阈值配置", None
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # 获取历史数据用于百分位和自适应阈值计算
        historical_data = None
        if config.threshold_type in [ThresholdType.PERCENTILE, ThresholdType.ADAPTIVE]:
            data_points = self.get_metric_data(metric_name, 86400)  # 24小时数据
            historical_data = [dp.value for dp in data_points]
        
        # 评估阈值
        triggered, reason = config.evaluate(value, timestamp, historical_data)
        
        return triggered, reason, config
    
    async def evaluate_all_thresholds(self, metrics: Dict[str, float],
                                    timestamp: datetime = None) -> List[Tuple[str, bool, str, ThresholdConfig]]:
        """评估所有阈值"""
        results = []
        
        for metric_name, value in metrics.items():
            triggered, reason, config = await self.evaluate_threshold(metric_name, value, timestamp)
            if config:
                results.append((metric_name, triggered, reason, config))
        
        return results
    
    async def start_adaptive_updates(self):
        """启动自适应更新任务"""
        if self.adaptive_update_task and not self.adaptive_update_task.done():
            return
        
        self.adaptive_update_task = asyncio.create_task(self._adaptive_update_loop())
        self.logger.info("启动自适应阈值更新任务")
    
    async def stop_adaptive_updates(self):
        """停止自适应更新任务"""
        if self.adaptive_update_task and not self.adaptive_update_task.done():
            self.adaptive_update_task.cancel()
            try:
                await self.adaptive_update_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("停止自适应阈值更新任务")
    
    async def _adaptive_update_loop(self):
        """自适应更新循环"""
        while True:
            try:
                await asyncio.sleep(self.update_interval)
                await self._update_adaptive_thresholds()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"自适应阈值更新出错: {e}")
    
    async def _update_adaptive_thresholds(self):
        """更新自适应阈值"""
        updated_count = 0
        
        for metric_name, config in self.threshold_configs.items():
            if (config.threshold_type == ThresholdType.ADAPTIVE and 
                config.adaptive_config and config.enabled):
                
                # 获取最近的数据
                recent_data = [
                    dp.value for dp in self.metric_data[metric_name]
                    if dp.age_seconds() <= config.adaptive_config.adaptation_window
                ]
                
                if len(recent_data) >= config.adaptive_config.min_samples:
                    old_upper = config.upper_threshold.value if config.upper_threshold else None
                    old_lower = config.lower_threshold.value if config.lower_threshold else None
                    
                    config.update_adaptive_threshold(recent_data)
                    
                    new_upper = config.upper_threshold.value if config.upper_threshold else None
                    new_lower = config.lower_threshold.value if config.lower_threshold else None
                    
                    if old_upper != new_upper or old_lower != new_lower:
                        updated_count += 1
                        self.logger.debug(f"更新自适应阈值 {metric_name}: "
                                        f"上限 {old_upper} -> {new_upper}, "
                                        f"下限 {old_lower} -> {new_lower}")
        
        if updated_count > 0:
            await self.save_configs()
            self.logger.info(f"更新了 {updated_count} 个自适应阈值")
    
    async def cleanup_old_data(self, retention_hours: int = 168):  # 7天
        """清理旧数据"""
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        cleaned_count = 0
        
        for metric_name in list(self.metric_data.keys()):
            original_count = len(self.metric_data[metric_name])
            
            # 过滤掉旧数据
            self.metric_data[metric_name] = deque(
                [dp for dp in self.metric_data[metric_name] if dp.timestamp >= cutoff_time],
                maxlen=self.metric_data[metric_name].maxlen
            )
            
            cleaned = original_count - len(self.metric_data[metric_name])
            cleaned_count += cleaned
        
        if cleaned_count > 0:
            self.logger.info(f"清理了 {cleaned_count} 个旧数据点")
    
    async def get_threshold_statistics(self) -> Dict[str, Any]:
        """获取阈值统计信息"""
        total_configs = len(self.threshold_configs)
        enabled_configs = len([c for c in self.threshold_configs.values() if c.enabled])
        
        # 按类型统计
        type_stats = {}
        for threshold_type in ThresholdType:
            count = len([c for c in self.threshold_configs.values() if c.threshold_type == threshold_type])
            type_stats[threshold_type.value] = count
        
        # 按严重级别统计
        severity_stats = {}
        for severity in ThresholdSeverity:
            count = len([c for c in self.threshold_configs.values() if c.severity == severity])
            severity_stats[severity.value] = count
        
        # 触发统计
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        triggered_24h = len([
            c for c in self.threshold_configs.values()
            if c.last_triggered and c.last_triggered > last_24h
        ])
        
        # 数据统计
        total_data_points = sum(len(data) for data in self.metric_data.values())
        metrics_with_data = len([k for k, v in self.metric_data.items() if len(v) > 0])
        
        return {
            "total_configs": total_configs,
            "enabled_configs": enabled_configs,
            "total_data_points": total_data_points,
            "metrics_with_data": metrics_with_data,
            "threshold_type_distribution": type_stats,
            "severity_distribution": severity_stats,
            "triggered_configs_24h": triggered_24h,
            "adaptive_configs": len([c for c in self.threshold_configs.values() 
                                   if c.threshold_type == ThresholdType.ADAPTIVE]),
            "average_evaluation_count": sum(c.evaluation_count for c in self.threshold_configs.values()) / total_configs if total_configs > 0 else 0
        }
    
    async def export_configs(self, export_path: str, metric_names: Optional[List[str]] = None) -> bool:
        """导出阈值配置"""
        try:
            if metric_names:
                configs_to_export = [
                    self.threshold_configs[name] for name in metric_names 
                    if name in self.threshold_configs
                ]
            else:
                configs_to_export = list(self.threshold_configs.values())
            
            export_data = {
                "configs": [self._config_to_dict(config) for config in configs_to_export],
                "exported_at": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功导出 {len(configs_to_export)} 个阈值配置到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出阈值配置失败: {e}")
            return False
    
    async def import_configs(self, import_path: str, overwrite: bool = False) -> bool:
        """导入阈值配置"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            configs_data = import_data.get("configs", [])
            imported_count = 0
            
            for config_data in configs_data:
                config = self._dict_to_config(config_data)
                if not config:
                    continue
                
                if config.metric_name in self.threshold_configs and not overwrite:
                    self.logger.warning(f"阈值配置已存在，跳过: {config.metric_name}")
                    continue
                
                self.threshold_configs[config.metric_name] = config
                imported_count += 1
            
            await self.save_configs()
            self.logger.info(f"成功导入 {imported_count} 个阈值配置")
            return True
            
        except Exception as e:
            self.logger.error(f"导入阈值配置失败: {e}")
            return False