#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能告警管理器

提供高级告警功能，包括：
- 智能阈值调整
- 数据质量异常告警
- 自适应告警抑制
- 告警升级策略
- 机器学习驱动的异常检测
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import numpy as np
from scipy import stats
import pickle
import os

from ..utils.logger import get_logger
from ..core.alerts import AlertManager, AlertRule, AlertSeverity, Alert, AlertStatus
from .prometheus_metrics import PrometheusMetrics, get_prometheus_metrics


class EnumJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，支持枚举类型序列化"""
    
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)

logger = get_logger(__name__)


class AlertType(Enum):
    """告警类型"""
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    SYSTEM_HEALTH = "system_health"
    BUSINESS_METRIC = "business_metric"
    SECURITY = "security"
    COST = "cost"


class ThresholdType(Enum):
    """阈值类型"""
    STATIC = "static"          # 静态阈值
    DYNAMIC = "dynamic"        # 动态阈值
    ADAPTIVE = "adaptive"      # 自适应阈值
    ML_BASED = "ml_based"      # 机器学习阈值


class AnomalyDetectionMethod(Enum):
    """异常检测方法"""
    STATISTICAL = "statistical"    # 统计方法
    ISOLATION_FOREST = "isolation_forest"  # 孤立森林
    ZSCORE = "zscore"             # Z分数
    IQR = "iqr"                   # 四分位距
    SEASONAL = "seasonal"         # 季节性检测


@dataclass
class IntelligentThreshold:
    """智能阈值配置"""
    metric_name: str
    threshold_type: ThresholdType
    base_value: float
    sensitivity: float = 1.0  # 敏感度（0.1-2.0）
    adaptation_rate: float = 0.1  # 适应速率
    min_samples: int = 100  # 最小样本数
    confidence_level: float = 0.95  # 置信水平
    seasonal_period: Optional[int] = None  # 季节性周期
    detection_method: AnomalyDetectionMethod = AnomalyDetectionMethod.STATISTICAL
    
    # 动态调整参数
    upper_multiplier: float = 1.5
    lower_multiplier: float = 0.5
    max_adjustment: float = 0.3  # 最大调整幅度
    
    # 历史数据
    historical_values: deque = field(default_factory=lambda: deque(maxlen=1000))
    last_update: Optional[datetime] = None
    current_threshold: Optional[float] = None


@dataclass
class DataQualityRule:
    """数据质量规则"""
    rule_id: str
    name: str
    description: str
    table_name: str
    column_name: Optional[str] = None
    rule_type: str = "completeness"  # completeness, consistency, accuracy, validity
    threshold: float = 0.95
    severity: AlertSeverity = AlertSeverity.MEDIUM
    enabled: bool = True
    
    # 检查SQL或函数
    check_query: Optional[str] = None
    check_function: Optional[Callable] = None
    
    # 历史结果
    last_check_time: Optional[datetime] = None
    last_result: Optional[float] = None
    trend_data: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class AlertSuppressionRule:
    """告警抑制规则"""
    rule_id: str
    name: str
    conditions: Dict[str, Any]  # 抑制条件
    duration: timedelta  # 抑制时长
    max_occurrences: int = 5  # 最大发生次数
    enabled: bool = True
    
    # 运行时状态
    active_suppressions: Set[str] = field(default_factory=set)
    occurrence_count: Dict[str, int] = field(default_factory=dict)
    last_triggered: Optional[datetime] = None


class IntelligentAlertManager:
    """智能告警管理器"""
    
    def __init__(self, 
                 prometheus_metrics: Optional[PrometheusMetrics] = None,
                 alert_manager: Optional[AlertManager] = None,
                 config_path: str = "config/intelligent_alerts.json"):
        """
        初始化智能告警管理器
        
        Args:
            prometheus_metrics: Prometheus指标实例
            alert_manager: 基础告警管理器
            config_path: 配置文件路径
        """
        self.prometheus_metrics = prometheus_metrics or get_prometheus_metrics()
        self.alert_manager = alert_manager or AlertManager()
        self.config_path = config_path
        
        # 智能阈值管理
        self.intelligent_thresholds: Dict[str, IntelligentThreshold] = {}
        self.threshold_models: Dict[str, Any] = {}  # ML模型缓存
        
        # 数据质量规则
        self.data_quality_rules: Dict[str, DataQualityRule] = {}
        
        # 告警抑制规则
        self.suppression_rules: Dict[str, AlertSuppressionRule] = {}
        
        # 运行时状态
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._last_evaluation = datetime.now()
        
        # 性能统计
        self.stats = {
            'total_evaluations': 0,
            'threshold_adjustments': 0,
            'data_quality_checks': 0,
            'alerts_suppressed': 0,
            'ml_predictions': 0
        }
        
        # 加载配置
        self._load_config()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 加载智能阈值配置
                for threshold_config in config.get('intelligent_thresholds', []):
                    # 转换枚举类型
                    if 'threshold_type' in threshold_config:
                        threshold_config['threshold_type'] = ThresholdType(threshold_config['threshold_type'])
                    if 'detection_method' in threshold_config:
                        threshold_config['detection_method'] = AnomalyDetectionMethod(threshold_config['detection_method'])
                    if 'last_update' in threshold_config and threshold_config['last_update']:
                        threshold_config['last_update'] = datetime.fromisoformat(threshold_config['last_update'])
                    
                    threshold = IntelligentThreshold(**threshold_config)
                    self.intelligent_thresholds[threshold.metric_name] = threshold
                
                # 加载数据质量规则
                for rule_config in config.get('data_quality_rules', []):
                    # 转换时间类型
                    if 'last_check_time' in rule_config and rule_config['last_check_time']:
                        rule_config['last_check_time'] = datetime.fromisoformat(rule_config['last_check_time'])
                    
                    rule = DataQualityRule(**rule_config)
                    self.data_quality_rules[rule.rule_id] = rule
                
                # 加载抑制规则
                for suppression_config in config.get('suppression_rules', []):
                    # 转换时间和集合类型
                    if 'last_triggered' in suppression_config and suppression_config['last_triggered']:
                        suppression_config['last_triggered'] = datetime.fromisoformat(suppression_config['last_triggered'])
                    if 'active_suppressions' in suppression_config:
                        suppression_config['active_suppressions'] = set(suppression_config['active_suppressions'])
                    if 'duration' in suppression_config and isinstance(suppression_config['duration'], (int, float)):
                        suppression_config['duration'] = timedelta(seconds=suppression_config['duration'])
                    
                    rule = AlertSuppressionRule(**suppression_config)
                    self.suppression_rules[rule.rule_id] = rule
                
                logger.info(f"已加载智能告警配置: {len(self.intelligent_thresholds)}个阈值, "
                           f"{len(self.data_quality_rules)}个数据质量规则, "
                           f"{len(self.suppression_rules)}个抑制规则")
            else:
                logger.info("配置文件不存在，使用默认配置")
                self._create_default_config()
        
        except Exception as e:
            logger.error(f"加载智能告警配置失败: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        # 默认智能阈值
        default_thresholds = [
            IntelligentThreshold(
                metric_name="api_response_time_p95",
                threshold_type=ThresholdType.ADAPTIVE,
                base_value=2.0,
                sensitivity=1.2,
                detection_method=AnomalyDetectionMethod.STATISTICAL
            ),
            IntelligentThreshold(
                metric_name="error_rate",
                threshold_type=ThresholdType.DYNAMIC,
                base_value=0.05,
                sensitivity=1.5,
                detection_method=AnomalyDetectionMethod.ZSCORE
            ),
            IntelligentThreshold(
                metric_name="memory_usage_percent",
                threshold_type=ThresholdType.STATIC,
                base_value=80.0,
                sensitivity=1.0
            )
        ]
        
        for threshold in default_thresholds:
            self.intelligent_thresholds[threshold.metric_name] = threshold
        
        # 默认数据质量规则
        default_quality_rules = [
            DataQualityRule(
                rule_id="token_completeness",
                name="Token数据完整性",
                description="检查Token数据的完整性",
                table_name="api_logs",
                column_name="tokens_used",
                rule_type="completeness",
                threshold=0.98,
                severity=AlertSeverity.HIGH
            ),
            DataQualityRule(
                rule_id="cost_consistency",
                name="成本数据一致性",
                description="检查成本计算的一致性",
                table_name="api_logs",
                rule_type="consistency",
                threshold=0.95,
                severity=AlertSeverity.MEDIUM
            )
        ]
        
        for rule in default_quality_rules:
            self.data_quality_rules[rule.rule_id] = rule
        
        # 默认抑制规则
        default_suppression_rules = [
            AlertSuppressionRule(
                rule_id="high_frequency_suppression",
                name="高频告警抑制",
                conditions={"frequency": "> 10/min"},
                duration=timedelta(minutes=15),
                max_occurrences=3
            )
        ]
        
        for rule in default_suppression_rules:
            self.suppression_rules[rule.rule_id] = rule
    
    async def start(self):
        """启动智能告警管理器"""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("智能告警管理器已启动")
    
    async def stop(self):
        """停止智能告警管理器"""
        if not self._running:
            return
        
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 保存配置
        await self._save_config()
        logger.info("智能告警管理器已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                # 评估智能阈值
                await self._evaluate_intelligent_thresholds()
                
                # 检查数据质量
                await self._check_data_quality()
                
                # 处理告警抑制
                await self._process_alert_suppression()
                
                # 更新统计信息
                self.stats['total_evaluations'] += 1
                self._last_evaluation = datetime.now()
                
                # 等待下一次检查
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"智能告警监控循环异常: {e}")
                await asyncio.sleep(10)
    
    async def _evaluate_intelligent_thresholds(self):
        """评估智能阈值"""
        for metric_name, threshold in self.intelligent_thresholds.items():
            try:
                # 获取当前指标值
                current_value = await self._get_metric_value(metric_name)
                if current_value is None:
                    continue
                
                # 更新历史数据
                threshold.historical_values.append({
                    'value': current_value,
                    'timestamp': datetime.now()
                })
                
                # 根据阈值类型调整
                if threshold.threshold_type == ThresholdType.ADAPTIVE:
                    await self._adjust_adaptive_threshold(threshold, current_value)
                elif threshold.threshold_type == ThresholdType.DYNAMIC:
                    await self._adjust_dynamic_threshold(threshold, current_value)
                elif threshold.threshold_type == ThresholdType.ML_BASED:
                    await self._adjust_ml_threshold(threshold, current_value)
                
                # 检查是否需要触发告警
                await self._check_threshold_alert(threshold, current_value)
                
            except Exception as e:
                logger.error(f"评估阈值 {metric_name} 失败: {e}")
    
    async def _adjust_adaptive_threshold(self, threshold: IntelligentThreshold, current_value: float):
        """调整自适应阈值"""
        if len(threshold.historical_values) < threshold.min_samples:
            return
        
        # 获取历史值
        values = [item['value'] for item in threshold.historical_values]
        
        # 计算统计指标
        mean_value = statistics.mean(values)
        std_value = statistics.stdev(values) if len(values) > 1 else 0
        
        # 根据检测方法调整阈值
        if threshold.detection_method == AnomalyDetectionMethod.STATISTICAL:
            # 使用均值 + N倍标准差
            new_threshold = mean_value + (threshold.sensitivity * std_value)
        elif threshold.detection_method == AnomalyDetectionMethod.ZSCORE:
            # 使用Z分数方法
            z_score = 2.0 * threshold.sensitivity  # 2倍敏感度作为Z分数阈值
            new_threshold = mean_value + (z_score * std_value)
        elif threshold.detection_method == AnomalyDetectionMethod.IQR:
            # 使用四分位距方法
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            new_threshold = q75 + (1.5 * threshold.sensitivity * iqr)
        else:
            new_threshold = threshold.base_value
        
        # 限制调整幅度
        if threshold.current_threshold is not None:
            max_change = threshold.current_threshold * threshold.max_adjustment
            change = new_threshold - threshold.current_threshold
            if abs(change) > max_change:
                new_threshold = threshold.current_threshold + (max_change if change > 0 else -max_change)
        
        # 应用适应速率
        if threshold.current_threshold is not None:
            new_threshold = (threshold.current_threshold * (1 - threshold.adaptation_rate) + 
                           new_threshold * threshold.adaptation_rate)
        
        # 更新阈值
        old_threshold = threshold.current_threshold
        threshold.current_threshold = new_threshold
        threshold.last_update = datetime.now()
        
        if old_threshold != new_threshold:
            self.stats['threshold_adjustments'] += 1
            logger.debug(f"调整自适应阈值 {threshold.metric_name}: {old_threshold} -> {new_threshold}")
    
    async def _adjust_dynamic_threshold(self, threshold: IntelligentThreshold, current_value: float):
        """调整动态阈值"""
        if len(threshold.historical_values) < 10:  # 需要至少10个样本
            return
        
        # 获取最近的值
        recent_values = [item['value'] for item in list(threshold.historical_values)[-50:]]
        
        # 计算趋势
        x = np.arange(len(recent_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_values)
        
        # 根据趋势调整阈值
        if abs(r_value) > 0.5:  # 有明显趋势
            # 预测下一个值
            next_value = slope * len(recent_values) + intercept
            
            # 根据趋势方向调整阈值
            if slope > 0:  # 上升趋势
                new_threshold = next_value * threshold.upper_multiplier * threshold.sensitivity
            else:  # 下降趋势
                new_threshold = next_value * threshold.lower_multiplier * threshold.sensitivity
        else:
            # 无明显趋势，使用统计方法
            mean_value = statistics.mean(recent_values)
            std_value = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
            new_threshold = mean_value + (2 * threshold.sensitivity * std_value)
        
        # 更新阈值
        threshold.current_threshold = new_threshold
        threshold.last_update = datetime.now()
        self.stats['threshold_adjustments'] += 1
    
    async def _adjust_ml_threshold(self, threshold: IntelligentThreshold, current_value: float):
        """调整基于机器学习的阈值"""
        if len(threshold.historical_values) < threshold.min_samples:
            return
        
        try:
            # 准备训练数据
            values = [item['value'] for item in threshold.historical_values]
            timestamps = [item['timestamp'] for item in threshold.historical_values]
            
            # 使用孤立森林进行异常检测
            from sklearn.ensemble import IsolationForest
            
            # 特征工程
            X = np.array(values).reshape(-1, 1)
            
            # 训练模型
            model = IsolationForest(contamination=0.1, random_state=42)
            model.fit(X)
            
            # 预测异常分数
            anomaly_scores = model.decision_function(X)
            
            # 计算阈值（基于异常分数的分位数）
            threshold_percentile = 95 - (threshold.sensitivity * 10)  # 敏感度影响分位数
            new_threshold = np.percentile(values, threshold_percentile)
            
            # 缓存模型
            self.threshold_models[threshold.metric_name] = model
            
            # 更新阈值
            threshold.current_threshold = new_threshold
            threshold.last_update = datetime.now()
            self.stats['threshold_adjustments'] += 1
            self.stats['ml_predictions'] += 1
            
            # 记录到Prometheus指标
            if self.prometheus_metrics:
                self.prometheus_metrics.record_intelligent_threshold_adjustment(
                    threshold.metric_name, 'ml_based'
                )
            
        except Exception as e:
            logger.error(f"ML阈值调整失败 {threshold.metric_name}: {e}")
    
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """获取指标当前值"""
        try:
            # 根据指标名称获取对应的Prometheus指标值
            if not self.prometheus_metrics:
                return None
            
            # 映射指标名称到Prometheus指标
            metric_mapping = {
                'api_response_time_p95': lambda: self._get_prometheus_gauge_value('harborai_api_response_time_p95_seconds'),
                'api_error_rate': lambda: self._get_prometheus_gauge_value('harborai_api_error_rate_percent'),
                'memory_usage_percent': lambda: self._get_prometheus_gauge_value('harborai_memory_usage_percent'),
                'cpu_usage_percent': lambda: self._get_prometheus_gauge_value('harborai_system_cpu_usage_percent'),
                'database_connection_pool_usage': lambda: self._get_prometheus_gauge_value('harborai_database_connection_pool_usage_percent'),
                'api_concurrent_connections': lambda: self._get_prometheus_gauge_value('harborai_api_concurrent_connections'),
                'database_query_duration_p99': lambda: self._get_prometheus_gauge_value('harborai_database_query_duration_p99_seconds'),
            }
            
            if metric_name in metric_mapping:
                return await metric_mapping[metric_name]()
            
            return None
            
        except Exception as e:
            logger.error(f"获取指标值失败 {metric_name}: {e}")
            return None
    
    async def _get_prometheus_gauge_value(self, metric_name: str) -> Optional[float]:
        """从Prometheus注册表获取Gauge指标值"""
        try:
            # 这里应该实现从Prometheus注册表获取指标值的逻辑
            # 由于这是一个复杂的实现，这里返回模拟值
            import random
            return random.uniform(0.1, 100.0)  # 模拟指标值
        except Exception as e:
            logger.error(f"获取Prometheus指标值失败 {metric_name}: {e}")
            return None
    
    async def _check_threshold_alert(self, threshold: IntelligentThreshold, current_value: float):
        """检查是否需要触发阈值告警"""
        if threshold.current_threshold is None:
            return
        
        # 检查是否超过阈值
        if current_value > threshold.current_threshold:
            # 创建告警
            alert = Alert(
                id=f"threshold_{threshold.metric_name}_{int(time.time())}",
                rule_id=f"intelligent_threshold_{threshold.metric_name}",
                name=f"智能阈值告警: {threshold.metric_name}",
                description=f"指标 {threshold.metric_name} 当前值 {current_value:.2f} 超过智能阈值 {threshold.current_threshold:.2f}",
                severity=AlertSeverity.HIGH,
                status=AlertStatus.FIRING,
                labels={
                    'metric_name': threshold.metric_name,
                    'threshold_type': threshold.threshold_type.value,
                    'detection_method': threshold.detection_method.value
                },
                annotations={
                    'current_value': str(current_value),
                    'threshold_value': str(threshold.current_threshold),
                    'sensitivity': str(threshold.sensitivity)
                },
                starts_at=datetime.now(),
                generator_url=f"intelligent_alert_manager/{threshold.metric_name}"
            )
            
            # 发送告警
            await self.alert_manager.send_alert(alert)
            
            # 记录到Prometheus指标
            if self.prometheus_metrics:
                self.prometheus_metrics.record_alert_rule_trigger(
                    f"intelligent_threshold_{threshold.metric_name}",
                    'high',
                    'fired'
                )

    async def _check_data_quality(self):
        """检查数据质量"""
        for rule_id, rule in self.data_quality_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # 执行数据质量检查
                quality_score = await self._execute_quality_check(rule)
                
                if quality_score is not None:
                    # 更新历史数据
                    rule.trend_data.append({
                        'score': quality_score,
                        'timestamp': datetime.now()
                    })
                    rule.last_result = quality_score
                    rule.last_check_time = datetime.now()
                    
                    # 检查是否低于阈值
                    if quality_score < rule.threshold:
                        await self._create_data_quality_alert(rule, quality_score)
                    
                    # 记录指标
                    self.prometheus_metrics.data_integrity_score.labels(
                        table_name=rule.table_name
                    ).set(quality_score * 100)
                
                self.stats['data_quality_checks'] += 1
                
            except Exception as e:
                logger.error(f"数据质量检查失败 {rule_id}: {e}")
    
    async def _execute_quality_check(self, rule: DataQualityRule) -> Optional[float]:
        """执行数据质量检查"""
        if rule.check_function:
            # 使用自定义函数
            return await rule.check_function(rule)
        elif rule.check_query:
            # 使用SQL查询
            # 这里需要数据库连接，简化实现
            return 0.95  # 模拟结果
        else:
            # 使用默认检查逻辑
            if rule.rule_type == "completeness":
                return await self._check_completeness(rule)
            elif rule.rule_type == "consistency":
                return await self._check_consistency(rule)
            elif rule.rule_type == "accuracy":
                return await self._check_accuracy(rule)
            elif rule.rule_type == "validity":
                return await self._check_validity(rule)
        
        return None
    
    async def _check_completeness(self, rule: DataQualityRule) -> float:
        """检查数据完整性"""
        # 模拟完整性检查
        # 实际实现需要查询数据库
        return 0.98
    
    async def _check_consistency(self, rule: DataQualityRule) -> float:
        """检查数据一致性"""
        # 模拟一致性检查
        return 0.96
    
    async def _check_accuracy(self, rule: DataQualityRule) -> float:
        """检查数据准确性"""
        # 模拟准确性检查
        return 0.94
    
    async def _check_validity(self, rule: DataQualityRule) -> float:
        """检查数据有效性"""
        # 模拟有效性检查
        return 0.97
    
    async def _create_data_quality_alert(self, rule: DataQualityRule, quality_score: float):
        """创建数据质量告警"""
        alert_id = f"data_quality_{rule.rule_id}_{int(time.time())}"
        
        alert = Alert(
            id=alert_id,
            rule_id=f"data_quality_{rule.rule_id}",
            rule_name=f"数据质量告警: {rule.name}",
            severity=rule.severity,
            status=AlertStatus.FIRING,
            message=f"表 {rule.table_name} 的 {rule.rule_type} 质量分数 {quality_score:.2f} 低于阈值 {rule.threshold:.2f}",
            metric_value=quality_score,
            threshold=rule.threshold,
            labels={
                'table': rule.table_name,
                'column': rule.column_name or 'all',
                'quality_type': rule.rule_type
            },
            annotations={
                'quality_score': str(quality_score),
                'threshold': str(rule.threshold),
                'rule_description': rule.description
            }
        )
        
        if not await self._should_suppress_alert(alert):
            await self.alert_manager.create_alert(alert)
            logger.warning(f"触发数据质量告警: {rule.name} = {quality_score:.2f} < {rule.threshold:.2f}")
    
    async def _process_alert_suppression(self):
        """处理告警抑制"""
        current_time = datetime.now()
        
        for rule_id, rule in self.suppression_rules.items():
            if not rule.enabled:
                continue
            
            # 清理过期的抑制
            expired_suppressions = set()
            for suppression_id in rule.active_suppressions:
                # 检查是否过期（简化实现）
                if rule.last_triggered and current_time - rule.last_triggered > rule.duration:
                    expired_suppressions.add(suppression_id)
            
            rule.active_suppressions -= expired_suppressions
            
            if expired_suppressions:
                logger.debug(f"清理过期抑制规则 {rule_id}: {len(expired_suppressions)}个")
    
    async def _should_suppress_alert(self, alert: Alert) -> bool:
        """检查是否应该抑制告警"""
        for rule_id, rule in self.suppression_rules.items():
            if not rule.enabled:
                continue
            
            # 检查抑制条件
            if await self._matches_suppression_conditions(alert, rule):
                # 检查发生次数
                alert_key = f"{alert.rule_id}_{alert.labels.get('metric', '')}"
                current_count = rule.occurrence_count.get(alert_key, 0)
                rule.occurrence_count[alert_key] = current_count + 1
                
                if rule.occurrence_count[alert_key] <= rule.max_occurrences:
                    rule.active_suppressions.add(alert.id)
                    rule.last_triggered = datetime.now()
                    self.stats['alerts_suppressed'] += 1
                    return True
        
        return False
    
    async def _matches_suppression_conditions(self, alert: Alert, rule: AlertSuppressionRule) -> bool:
        """检查告警是否匹配抑制条件"""
        # 简化实现，实际需要更复杂的条件匹配逻辑
        
        for condition_key, condition_value in rule.conditions.items():
            if condition_key == "frequency":
                # 检查频率条件
                return True  # 简化实现
            elif condition_key == "severity":
                if alert.severity.value != condition_value:
                    return False
            elif condition_key == "metric":
                if alert.labels.get('metric') != condition_value:
                    return False
            else:
                # 检查标签匹配
                alert_value = alert.labels.get(condition_key)
                if alert_value != condition_value:
                    return False
        
        return True
    
    async def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """获取指标值"""
        # 这里需要从Prometheus或其他监控系统获取实际指标值
        # 简化实现，返回模拟值
        import random
        
        if metric_name == "api_response_time_p95":
            return random.uniform(1.0, 3.0)
        elif metric_name == "error_rate":
            return random.uniform(0.01, 0.1)
        elif metric_name == "memory_usage_percent":
            return random.uniform(60.0, 90.0)
        
        return None
    
    async def _save_config(self):
        """保存配置"""
        try:
            config = {
                'intelligent_thresholds': [
                    {
                        **asdict(threshold),
                        'historical_values': [],  # 不保存历史数据
                        'last_update': threshold.last_update.isoformat() if threshold.last_update else None
                    }
                    for threshold in self.intelligent_thresholds.values()
                ],
                'data_quality_rules': [
                    {
                        **asdict(rule),
                        'trend_data': [],  # 不保存趋势数据
                        'last_check_time': rule.last_check_time.isoformat() if rule.last_check_time else None
                    }
                    for rule in self.data_quality_rules.values()
                ],
                'suppression_rules': [
                    {
                        **asdict(rule),
                        'active_suppressions': list(rule.active_suppressions),
                        'occurrence_count': rule.occurrence_count,
                        'last_triggered': rule.last_triggered.isoformat() if rule.last_triggered else None
                    }
                    for rule in self.suppression_rules.values()
                ]
            }
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False, cls=EnumJSONEncoder)
            
            logger.debug("智能告警配置已保存")
            
        except Exception as e:
            logger.error(f"保存智能告警配置失败: {e}")
    
    # === 公共API ===
    
    def add_intelligent_threshold(self, threshold: IntelligentThreshold):
        """添加智能阈值"""
        self.intelligent_thresholds[threshold.metric_name] = threshold
        logger.info(f"添加智能阈值: {threshold.metric_name}")
    
    def remove_intelligent_threshold(self, metric_name: str):
        """移除智能阈值"""
        if metric_name in self.intelligent_thresholds:
            del self.intelligent_thresholds[metric_name]
            logger.info(f"移除智能阈值: {metric_name}")
    
    def add_data_quality_rule(self, rule: DataQualityRule):
        """添加数据质量规则"""
        self.data_quality_rules[rule.rule_id] = rule
        logger.info(f"添加数据质量规则: {rule.rule_id}")
    
    def remove_data_quality_rule(self, rule_id: str):
        """移除数据质量规则"""
        if rule_id in self.data_quality_rules:
            del self.data_quality_rules[rule_id]
            logger.info(f"移除数据质量规则: {rule_id}")
    
    def add_suppression_rule(self, rule: AlertSuppressionRule):
        """添加抑制规则"""
        self.suppression_rules[rule.rule_id] = rule
        logger.info(f"添加抑制规则: {rule.rule_id}")
    
    def remove_suppression_rule(self, rule_id: str):
        """移除抑制规则"""
        if rule_id in self.suppression_rules:
            del self.suppression_rules[rule_id]
            logger.info(f"移除抑制规则: {rule_id}")
    
    def get_threshold_status(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """获取阈值状态"""
        if metric_name not in self.intelligent_thresholds:
            return None
        
        threshold = self.intelligent_thresholds[metric_name]
        return {
            'metric_name': threshold.metric_name,
            'threshold_type': threshold.threshold_type.value,
            'current_threshold': threshold.current_threshold,
            'base_value': threshold.base_value,
            'sensitivity': threshold.sensitivity,
            'last_update': threshold.last_update.isoformat() if threshold.last_update else None,
            'sample_count': len(threshold.historical_values),
            'detection_method': threshold.detection_method.value
        }
    
    def get_data_quality_status(self, rule_id: str) -> Optional[Dict[str, Any]]:
        """获取数据质量状态"""
        if rule_id not in self.data_quality_rules:
            return None
        
        rule = self.data_quality_rules[rule_id]
        return {
            'rule_id': rule.rule_id,
            'name': rule.name,
            'table_name': rule.table_name,
            'rule_type': rule.rule_type,
            'threshold': rule.threshold,
            'last_result': rule.last_result,
            'last_check_time': rule.last_check_time.isoformat() if rule.last_check_time else None,
            'trend_count': len(rule.trend_data),
            'enabled': rule.enabled
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'intelligent_thresholds_count': len(self.intelligent_thresholds),
            'data_quality_rules_count': len(self.data_quality_rules),
            'suppression_rules_count': len(self.suppression_rules),
            'last_evaluation': self._last_evaluation.isoformat(),
            'running': self._running
        }


# 全局实例
_intelligent_alert_manager: Optional[IntelligentAlertManager] = None


def get_intelligent_alert_manager() -> IntelligentAlertManager:
    """获取全局智能告警管理器实例"""
    global _intelligent_alert_manager
    if _intelligent_alert_manager is None:
        _intelligent_alert_manager = IntelligentAlertManager()
    return _intelligent_alert_manager


async def init_intelligent_alert_manager(
    prometheus_metrics: Optional[PrometheusMetrics] = None,
    alert_manager: Optional[AlertManager] = None,
    config_path: str = "config/intelligent_alerts.json"
) -> IntelligentAlertManager:
    """初始化智能告警管理器"""
    global _intelligent_alert_manager
    _intelligent_alert_manager = IntelligentAlertManager(
        prometheus_metrics=prometheus_metrics,
        alert_manager=alert_manager,
        config_path=config_path
    )
    await _intelligent_alert_manager.start()
    return _intelligent_alert_manager