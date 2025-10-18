"""
告警阈值管理器

负责管理动态阈值、自适应阈值、阈值优化等功能。
支持基于历史数据的阈值调整、季节性阈值、业务时间阈值等。
"""

import asyncio
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import math


class ThresholdType(Enum):
    """阈值类型"""
    STATIC = "static"           # 静态阈值
    DYNAMIC = "dynamic"         # 动态阈值
    ADAPTIVE = "adaptive"       # 自适应阈值
    SEASONAL = "seasonal"       # 季节性阈值
    BUSINESS_HOURS = "business_hours"  # 业务时间阈值
    PERCENTILE = "percentile"   # 百分位阈值


class ThresholdDirection(Enum):
    """阈值方向"""
    ABOVE = "above"     # 超过阈值
    BELOW = "below"     # 低于阈值
    OUTSIDE = "outside" # 超出范围
    INSIDE = "inside"   # 在范围内


@dataclass
class ThresholdConfig:
    """阈值配置"""
    metric: str
    threshold_type: ThresholdType
    direction: ThresholdDirection
    value: Union[float, Dict[str, float]]
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    confidence_level: float = 0.95
    sensitivity: float = 1.0
    update_interval: int = 3600  # 更新间隔（秒）
    lookback_period: int = 86400 * 7  # 回看周期（秒）
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ThresholdConfig':
        """从字典创建"""
        return cls(
            metric=data['metric'],
            threshold_type=ThresholdType(data['threshold_type']),
            direction=ThresholdDirection(data['direction']),
            value=data['value'],
            min_value=data.get('min_value'),
            max_value=data.get('max_value'),
            confidence_level=data.get('confidence_level', 0.95),
            sensitivity=data.get('sensitivity', 1.0),
            update_interval=data.get('update_interval', 3600),
            lookback_period=data.get('lookback_period', 86400 * 7),
            enabled=data.get('enabled', True)
        )


@dataclass
class ThresholdValue:
    """阈值数值"""
    value: float
    upper_bound: Optional[float] = None
    lower_bound: Optional[float] = None
    confidence: float = 1.0
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'value': self.value,
            'upper_bound': self.upper_bound,
            'lower_bound': self.lower_bound,
            'confidence': self.confidence,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


@dataclass
class MetricDataPoint:
    """指标数据点"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = None
    
    def __post_init__(self):
        if self.labels is None:
            self.labels = {}


class ThresholdCalculator:
    """阈值计算器"""
    
    @staticmethod
    def calculate_static_threshold(config: ThresholdConfig) -> ThresholdValue:
        """计算静态阈值"""
        if isinstance(config.value, dict):
            value = config.value.get('default', 0)
        else:
            value = config.value
        
        return ThresholdValue(
            value=value,
            confidence=1.0,
            last_updated=datetime.now()
        )
    
    @staticmethod
    def calculate_percentile_threshold(data_points: List[MetricDataPoint],
                                     config: ThresholdConfig) -> ThresholdValue:
        """计算百分位阈值"""
        if not data_points:
            return ThresholdCalculator.calculate_static_threshold(config)
        
        values = [dp.value for dp in data_points]
        
        if isinstance(config.value, dict):
            percentile = config.value.get('percentile', 95)
        else:
            percentile = config.value
        
        threshold_value = statistics.quantiles(values, n=100)[int(percentile) - 1]
        
        # 应用敏感度调整
        if config.direction == ThresholdDirection.ABOVE:
            threshold_value *= config.sensitivity
        elif config.direction == ThresholdDirection.BELOW:
            threshold_value /= config.sensitivity
        
        return ThresholdValue(
            value=threshold_value,
            confidence=min(len(values) / 100, 1.0),
            last_updated=datetime.now()
        )
    
    @staticmethod
    def calculate_adaptive_threshold(data_points: List[MetricDataPoint],
                                   config: ThresholdConfig) -> ThresholdValue:
        """计算自适应阈值"""
        if not data_points:
            return ThresholdCalculator.calculate_static_threshold(config)
        
        values = [dp.value for dp in data_points]
        
        # 计算统计指标
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        
        # 计算置信区间
        z_score = 1.96 if config.confidence_level == 0.95 else 2.58  # 95% or 99%
        margin = z_score * stdev * config.sensitivity
        
        if config.direction == ThresholdDirection.ABOVE:
            threshold_value = mean + margin
        elif config.direction == ThresholdDirection.BELOW:
            threshold_value = mean - margin
        else:
            threshold_value = mean
        
        # 应用边界限制
        if config.min_value is not None:
            threshold_value = max(threshold_value, config.min_value)
        if config.max_value is not None:
            threshold_value = min(threshold_value, config.max_value)
        
        return ThresholdValue(
            value=threshold_value,
            upper_bound=mean + margin,
            lower_bound=mean - margin,
            confidence=min(len(values) / 50, 1.0),
            last_updated=datetime.now()
        )
    
    @staticmethod
    def calculate_seasonal_threshold(data_points: List[MetricDataPoint],
                                   config: ThresholdConfig) -> ThresholdValue:
        """计算季节性阈值"""
        if not data_points:
            return ThresholdCalculator.calculate_static_threshold(config)
        
        now = datetime.now()
        current_hour = now.hour
        current_weekday = now.weekday()
        
        # 过滤相同时间段的数据
        seasonal_points = []
        for dp in data_points:
            if (dp.timestamp.hour == current_hour and 
                dp.timestamp.weekday() == current_weekday):
                seasonal_points.append(dp)
        
        if not seasonal_points:
            # 如果没有相同时间段的数据，使用所有数据
            seasonal_points = data_points
        
        # 使用自适应方法计算阈值
        return ThresholdCalculator.calculate_adaptive_threshold(seasonal_points, config)
    
    @staticmethod
    def calculate_business_hours_threshold(data_points: List[MetricDataPoint],
                                         config: ThresholdConfig) -> ThresholdValue:
        """计算业务时间阈值"""
        if not data_points:
            return ThresholdCalculator.calculate_static_threshold(config)
        
        now = datetime.now()
        is_business_hours = 9 <= now.hour <= 17 and now.weekday() < 5
        
        # 过滤业务时间或非业务时间的数据
        filtered_points = []
        for dp in data_points:
            dp_is_business = 9 <= dp.timestamp.hour <= 17 and dp.timestamp.weekday() < 5
            if dp_is_business == is_business_hours:
                filtered_points.append(dp)
        
        if not filtered_points:
            filtered_points = data_points
        
        # 获取业务时间配置
        if isinstance(config.value, dict):
            if is_business_hours:
                threshold_config = config.value.get('business_hours', config.value.get('default', 0))
            else:
                threshold_config = config.value.get('off_hours', config.value.get('default', 0))
        else:
            threshold_config = config.value
        
        # 创建临时配置
        temp_config = ThresholdConfig(
            metric=config.metric,
            threshold_type=ThresholdType.ADAPTIVE,
            direction=config.direction,
            value=threshold_config,
            confidence_level=config.confidence_level,
            sensitivity=config.sensitivity
        )
        
        return ThresholdCalculator.calculate_adaptive_threshold(filtered_points, temp_config)


class ThresholdManager:
    """阈值管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.threshold_configs: Dict[str, ThresholdConfig] = {}
        self.current_thresholds: Dict[str, ThresholdValue] = {}
        self.metric_data: Dict[str, List[MetricDataPoint]] = {}
        self.calculator = ThresholdCalculator()
        self._update_tasks: Dict[str, asyncio.Task] = {}
    
    async def add_threshold_config(self, config: ThresholdConfig):
        """添加阈值配置"""
        self.threshold_configs[config.metric] = config
        
        # 初始化阈值
        await self.update_threshold(config.metric)
        
        # 启动自动更新任务
        if config.threshold_type != ThresholdType.STATIC:
            await self._start_update_task(config.metric)
        
        self.logger.info(f"添加阈值配置: {config.metric}")
    
    async def remove_threshold_config(self, metric: str):
        """删除阈值配置"""
        if metric in self.threshold_configs:
            del self.threshold_configs[metric]
        
        if metric in self.current_thresholds:
            del self.current_thresholds[metric]
        
        if metric in self.metric_data:
            del self.metric_data[metric]
        
        # 停止更新任务
        await self._stop_update_task(metric)
        
        self.logger.info(f"删除阈值配置: {metric}")
    
    async def update_threshold_config(self, metric: str, config: ThresholdConfig):
        """更新阈值配置"""
        old_config = self.threshold_configs.get(metric)
        self.threshold_configs[metric] = config
        
        # 重新计算阈值
        await self.update_threshold(metric)
        
        # 重启更新任务
        if old_config and old_config.threshold_type != config.threshold_type:
            await self._stop_update_task(metric)
            if config.threshold_type != ThresholdType.STATIC:
                await self._start_update_task(metric)
        
        self.logger.info(f"更新阈值配置: {metric}")
    
    async def add_metric_data(self, metric: str, data_point: MetricDataPoint):
        """添加指标数据"""
        if metric not in self.metric_data:
            self.metric_data[metric] = []
        
        self.metric_data[metric].append(data_point)
        
        # 清理过期数据
        await self._cleanup_old_data(metric)
    
    async def add_metric_data_batch(self, metric: str, data_points: List[MetricDataPoint]):
        """批量添加指标数据"""
        if metric not in self.metric_data:
            self.metric_data[metric] = []
        
        self.metric_data[metric].extend(data_points)
        
        # 清理过期数据
        await self._cleanup_old_data(metric)
    
    async def _cleanup_old_data(self, metric: str):
        """清理过期数据"""
        if metric not in self.threshold_configs:
            return
        
        config = self.threshold_configs[metric]
        cutoff_time = datetime.now() - timedelta(seconds=config.lookback_period)
        
        if metric in self.metric_data:
            self.metric_data[metric] = [
                dp for dp in self.metric_data[metric]
                if dp.timestamp > cutoff_time
            ]
    
    async def update_threshold(self, metric: str) -> Optional[ThresholdValue]:
        """更新阈值"""
        if metric not in self.threshold_configs:
            self.logger.warning(f"阈值配置不存在: {metric}")
            return None
        
        config = self.threshold_configs[metric]
        if not config.enabled:
            return None
        
        data_points = self.metric_data.get(metric, [])
        
        try:
            # 根据阈值类型计算阈值
            if config.threshold_type == ThresholdType.STATIC:
                threshold_value = self.calculator.calculate_static_threshold(config)
            elif config.threshold_type == ThresholdType.PERCENTILE:
                threshold_value = self.calculator.calculate_percentile_threshold(data_points, config)
            elif config.threshold_type == ThresholdType.ADAPTIVE:
                threshold_value = self.calculator.calculate_adaptive_threshold(data_points, config)
            elif config.threshold_type == ThresholdType.SEASONAL:
                threshold_value = self.calculator.calculate_seasonal_threshold(data_points, config)
            elif config.threshold_type == ThresholdType.BUSINESS_HOURS:
                threshold_value = self.calculator.calculate_business_hours_threshold(data_points, config)
            else:
                threshold_value = self.calculator.calculate_static_threshold(config)
            
            self.current_thresholds[metric] = threshold_value
            
            self.logger.debug(f"更新阈值 {metric}: {threshold_value.value}")
            return threshold_value
            
        except Exception as e:
            self.logger.error(f"更新阈值失败 {metric}: {e}")
            return None
    
    async def _start_update_task(self, metric: str):
        """启动自动更新任务"""
        if metric in self._update_tasks:
            await self._stop_update_task(metric)
        
        config = self.threshold_configs[metric]
        
        async def update_loop():
            while True:
                try:
                    await asyncio.sleep(config.update_interval)
                    await self.update_threshold(metric)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"阈值更新任务错误 {metric}: {e}")
        
        self._update_tasks[metric] = asyncio.create_task(update_loop())
        self.logger.debug(f"启动阈值更新任务: {metric}")
    
    async def _stop_update_task(self, metric: str):
        """停止自动更新任务"""
        if metric in self._update_tasks:
            self._update_tasks[metric].cancel()
            try:
                await self._update_tasks[metric]
            except asyncio.CancelledError:
                pass
            del self._update_tasks[metric]
            self.logger.debug(f"停止阈值更新任务: {metric}")
    
    def get_threshold(self, metric: str) -> Optional[ThresholdValue]:
        """获取当前阈值"""
        return self.current_thresholds.get(metric)
    
    def get_threshold_config(self, metric: str) -> Optional[ThresholdConfig]:
        """获取阈值配置"""
        return self.threshold_configs.get(metric)
    
    def check_threshold(self, metric: str, value: float) -> Tuple[bool, Optional[str]]:
        """检查是否超过阈值"""
        if metric not in self.current_thresholds:
            return False, "阈值未配置"
        
        if metric not in self.threshold_configs:
            return False, "阈值配置不存在"
        
        threshold = self.current_thresholds[metric]
        config = self.threshold_configs[metric]
        
        if config.direction == ThresholdDirection.ABOVE:
            if value > threshold.value:
                return True, f"值 {value} 超过阈值 {threshold.value}"
        elif config.direction == ThresholdDirection.BELOW:
            if value < threshold.value:
                return True, f"值 {value} 低于阈值 {threshold.value}"
        elif config.direction == ThresholdDirection.OUTSIDE:
            if threshold.upper_bound and threshold.lower_bound:
                if value > threshold.upper_bound or value < threshold.lower_bound:
                    return True, f"值 {value} 超出范围 [{threshold.lower_bound}, {threshold.upper_bound}]"
        elif config.direction == ThresholdDirection.INSIDE:
            if threshold.upper_bound and threshold.lower_bound:
                if threshold.lower_bound <= value <= threshold.upper_bound:
                    return True, f"值 {value} 在范围内 [{threshold.lower_bound}, {threshold.upper_bound}]"
        
        return False, None
    
    async def get_threshold_statistics(self) -> Dict[str, Any]:
        """获取阈值统计信息"""
        total_configs = len(self.threshold_configs)
        active_configs = len([c for c in self.threshold_configs.values() if c.enabled])
        
        # 按类型统计
        type_stats = {}
        for threshold_type in ThresholdType:
            count = len([c for c in self.threshold_configs.values() 
                        if c.threshold_type == threshold_type])
            type_stats[threshold_type.value] = count
        
        # 按方向统计
        direction_stats = {}
        for direction in ThresholdDirection:
            count = len([c for c in self.threshold_configs.values() 
                        if c.direction == direction])
            direction_stats[direction.value] = count
        
        # 数据点统计
        total_data_points = sum(len(data) for data in self.metric_data.values())
        metrics_with_data = len([m for m, data in self.metric_data.items() if data])
        
        # 更新任务统计
        active_update_tasks = len(self._update_tasks)
        
        return {
            "total_configs": total_configs,
            "active_configs": active_configs,
            "type_distribution": type_stats,
            "direction_distribution": direction_stats,
            "total_data_points": total_data_points,
            "metrics_with_data": metrics_with_data,
            "active_update_tasks": active_update_tasks,
            "current_thresholds": len(self.current_thresholds)
        }
    
    async def export_thresholds(self, export_path: str) -> bool:
        """导出阈值配置"""
        try:
            export_data = {
                "threshold_configs": {
                    metric: config.to_dict()
                    for metric, config in self.threshold_configs.items()
                },
                "current_thresholds": {
                    metric: threshold.to_dict()
                    for metric, threshold in self.current_thresholds.items()
                },
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
    
    async def import_thresholds(self, import_path: str) -> bool:
        """导入阈值配置"""
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            # 导入配置
            configs_data = import_data.get("threshold_configs", {})
            for metric, config_data in configs_data.items():
                config = ThresholdConfig.from_dict(config_data)
                await self.add_threshold_config(config)
            
            self.logger.info(f"成功导入 {len(configs_data)} 个阈值配置")
            return True
            
        except Exception as e:
            self.logger.error(f"导入阈值配置失败: {e}")
            return False
    
    async def cleanup(self):
        """清理资源"""
        # 停止所有更新任务
        for metric in list(self._update_tasks.keys()):
            await self._stop_update_task(metric)
        
        self.logger.info("阈值管理器清理完成")