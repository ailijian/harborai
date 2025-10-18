#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降级状态监控

监控系统降级状态、恢复状态，并提供告警机制。
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta

from ...utils.logger import get_logger
from ...monitoring.prometheus_metrics import get_prometheus_metrics

logger = get_logger(__name__)


class DegradationStatus(Enum):
    """降级状态枚举"""
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"


@dataclass
class DegradationEvent:
    """降级事件"""
    component: str
    degradation_type: str
    status: DegradationStatus
    reason: str
    details: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'component': self.component,
            'degradation_type': self.degradation_type,
            'status': self.status.value,
            'reason': self.reason,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class DegradationRule:
    """降级规则"""
    name: str
    component: str
    degradation_type: str
    condition: Callable[[Dict[str, Any]], bool]
    recovery_condition: Callable[[Dict[str, Any]], bool]
    threshold_duration: float  # 持续时间阈值（秒）
    description: str


class DegradationMonitor:
    """降级状态监控器"""
    
    def __init__(self):
        """初始化降级监控器"""
        self.rules: Dict[str, DegradationRule] = {}
        self.current_degradations: Dict[str, DegradationEvent] = {}
        self.degradation_history: List[DegradationEvent] = []
        self.condition_start_times: Dict[str, float] = {}
        self.recovery_start_times: Dict[str, float] = {}
        self.running = False
        self.background_task: Optional[asyncio.Task] = None
        self.alert_callbacks: List[Callable[[DegradationEvent], None]] = []
        
        # 注册默认降级规则
        self._register_default_rules()
        logger.info("降级状态监控器已初始化")
    
    def _register_default_rules(self):
        """注册默认降级规则"""
        # 数据库性能降级规则
        self.register_rule(DegradationRule(
            name='database_performance_degradation',
            component='database',
            degradation_type='performance',
            condition=lambda metrics: metrics.get('query_time_ms', 0) > 500,
            recovery_condition=lambda metrics: metrics.get('query_time_ms', 0) < 200,
            threshold_duration=60.0,  # 持续1分钟
            description='数据库查询性能降级'
        ))
        
        # 内存使用降级规则
        self.register_rule(DegradationRule(
            name='memory_usage_degradation',
            component='system',
            degradation_type='memory',
            condition=lambda metrics: metrics.get('memory_percent', 0) > 85,
            recovery_condition=lambda metrics: metrics.get('memory_percent', 0) < 75,
            threshold_duration=120.0,  # 持续2分钟
            description='系统内存使用过高'
        ))
        
        # CPU使用降级规则
        self.register_rule(DegradationRule(
            name='cpu_usage_degradation',
            component='system',
            degradation_type='cpu',
            condition=lambda metrics: metrics.get('cpu_percent', 0) > 85,
            recovery_condition=lambda metrics: metrics.get('cpu_percent', 0) < 70,
            threshold_duration=180.0,  # 持续3分钟
            description='系统CPU使用过高'
        ))
        
        # 磁盘空间降级规则
        self.register_rule(DegradationRule(
            name='disk_space_degradation',
            component='system',
            degradation_type='disk',
            condition=lambda metrics: metrics.get('disk_percent', 0) > 90,
            recovery_condition=lambda metrics: metrics.get('disk_percent', 0) < 85,
            threshold_duration=300.0,  # 持续5分钟
            description='磁盘空间不足'
        ))
        
        # API错误率降级规则
        self.register_rule(DegradationRule(
            name='api_error_rate_degradation',
            component='api',
            degradation_type='error_rate',
            condition=lambda metrics: metrics.get('error_rate_percent', 0) > 10,
            recovery_condition=lambda metrics: metrics.get('error_rate_percent', 0) < 5,
            threshold_duration=300.0,  # 持续5分钟
            description='API错误率过高'
        ))
        
        # 数据一致性降级规则
        self.register_rule(DegradationRule(
            name='data_consistency_degradation',
            component='data',
            degradation_type='consistency',
            condition=lambda metrics: metrics.get('critical_issues', 0) > 0,
            recovery_condition=lambda metrics: metrics.get('critical_issues', 0) == 0,
            threshold_duration=60.0,  # 持续1分钟
            description='数据一致性问题'
        ))
    
    def register_rule(self, rule: DegradationRule):
        """注册降级规则
        
        Args:
            rule: 降级规则
        """
        self.rules[rule.name] = rule
        logger.info(f"已注册降级规则: {rule.name} - {rule.description}")
    
    def register_alert_callback(self, callback: Callable[[DegradationEvent], None]):
        """注册告警回调函数
        
        Args:
            callback: 告警回调函数
        """
        self.alert_callbacks.append(callback)
        logger.info("已注册告警回调函数")
    
    async def check_degradations(self, metrics: Dict[str, Any]):
        """检查降级状态
        
        Args:
            metrics: 当前系统指标
        """
        current_time = time.time()
        
        for rule_name, rule in self.rules.items():
            try:
                # 检查降级条件
                if rule.condition(metrics):
                    # 记录条件开始时间
                    if rule_name not in self.condition_start_times:
                        self.condition_start_times[rule_name] = current_time
                    
                    # 检查是否达到持续时间阈值
                    if current_time - self.condition_start_times[rule_name] >= rule.threshold_duration:
                        await self._trigger_degradation(rule, metrics)
                
                else:
                    # 条件不满足，清除开始时间
                    if rule_name in self.condition_start_times:
                        del self.condition_start_times[rule_name]
                    
                    # 检查恢复条件
                    if rule_name in self.current_degradations:
                        if rule.recovery_condition(metrics):
                            # 记录恢复开始时间
                            if rule_name not in self.recovery_start_times:
                                self.recovery_start_times[rule_name] = current_time
                            
                            # 检查是否达到恢复持续时间（使用一半的阈值时间）
                            recovery_threshold = rule.threshold_duration / 2
                            if current_time - self.recovery_start_times[rule_name] >= recovery_threshold:
                                await self._trigger_recovery(rule, metrics)
                        else:
                            # 恢复条件不满足，清除恢复开始时间
                            if rule_name in self.recovery_start_times:
                                del self.recovery_start_times[rule_name]
                
            except Exception as e:
                logger.error(f"检查降级规则 {rule_name} 时发生异常: {e}")
    
    async def _trigger_degradation(self, rule: DegradationRule, metrics: Dict[str, Any]):
        """触发降级事件
        
        Args:
            rule: 降级规则
            metrics: 当前指标
        """
        # 如果已经在降级状态，不重复触发
        if rule.name in self.current_degradations:
            return
        
        # 创建降级事件
        event = DegradationEvent(
            component=rule.component,
            degradation_type=rule.degradation_type,
            status=DegradationStatus.DEGRADED,
            reason=rule.description,
            details=metrics.copy(),
            timestamp=datetime.now()
        )
        
        # 记录降级状态
        self.current_degradations[rule.name] = event
        self.degradation_history.append(event)
        
        # 记录Prometheus指标
        metrics_client = get_prometheus_metrics()
        if metrics_client:
            metrics_client.set_degradation_status(
                component=rule.component,
                degradation_type=rule.degradation_type,
                is_degraded=True
            )
        
        # 发送告警
        await self._send_alert(event)
        
        logger.warning(f"触发降级事件: {rule.component}.{rule.degradation_type} - {rule.description}")
    
    async def _trigger_recovery(self, rule: DegradationRule, metrics: Dict[str, Any]):
        """触发恢复事件
        
        Args:
            rule: 降级规则
            metrics: 当前指标
        """
        if rule.name not in self.current_degradations:
            return
        
        # 创建恢复事件
        event = DegradationEvent(
            component=rule.component,
            degradation_type=rule.degradation_type,
            status=DegradationStatus.NORMAL,
            reason=f"{rule.description} - 已恢复",
            details=metrics.copy(),
            timestamp=datetime.now()
        )
        
        # 移除降级状态
        del self.current_degradations[rule.name]
        if rule.name in self.recovery_start_times:
            del self.recovery_start_times[rule.name]
        
        # 记录恢复历史
        self.degradation_history.append(event)
        
        # 记录Prometheus指标
        metrics_client = get_prometheus_metrics()
        if metrics_client:
            metrics_client.set_degradation_status(
                component=rule.component,
                degradation_type=rule.degradation_type,
                is_degraded=False
            )
        
        # 发送恢复通知
        await self._send_alert(event)
        
        logger.info(f"触发恢复事件: {rule.component}.{rule.degradation_type} - 已恢复正常")
    
    async def _send_alert(self, event: DegradationEvent):
        """发送告警
        
        Args:
            event: 降级事件
        """
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"发送告警时发生异常: {e}")
    
    def get_current_degradations(self) -> Dict[str, DegradationEvent]:
        """获取当前降级状态
        
        Returns:
            当前降级状态字典
        """
        return self.current_degradations.copy()
    
    def get_degradation_history(self, hours: int = 24) -> List[DegradationEvent]:
        """获取降级历史
        
        Args:
            hours: 查询最近多少小时的历史
            
        Returns:
            降级历史列表
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.degradation_history
            if event.timestamp >= cutoff_time
        ]
    
    def get_degradation_summary(self) -> Dict[str, Any]:
        """获取降级状态摘要
        
        Returns:
            降级状态摘要
        """
        current_degradations = self.get_current_degradations()
        recent_history = self.get_degradation_history(24)
        
        # 统计各组件的降级次数
        component_stats = {}
        for event in recent_history:
            component = event.component
            if component not in component_stats:
                component_stats[component] = {
                    'degradation_count': 0,
                    'recovery_count': 0,
                    'current_degraded': False
                }
            
            if event.status == DegradationStatus.DEGRADED:
                component_stats[component]['degradation_count'] += 1
            elif event.status == DegradationStatus.NORMAL:
                component_stats[component]['recovery_count'] += 1
        
        # 标记当前降级状态
        for event in current_degradations.values():
            component = event.component
            if component in component_stats:
                component_stats[component]['current_degraded'] = True
        
        return {
            'current_degradations': {
                name: event.to_dict() 
                for name, event in current_degradations.items()
            },
            'degradation_count': len(current_degradations),
            'component_stats': component_stats,
            'total_events_24h': len(recent_history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def start_monitoring(self):
        """启动降级监控"""
        if self.running:
            logger.warning("降级监控已在运行")
            return
        
        self.running = True
        self.background_task = asyncio.create_task(self._monitoring_loop())
        logger.info("降级状态监控已启动")
    
    async def stop_monitoring(self):
        """停止降级监控"""
        if not self.running:
            return
        
        self.running = False
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        logger.info("降级状态监控已停止")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self.running:
            try:
                # 从健康检查服务获取最新指标
                from .health_check_service import get_health_service
                
                health_service = get_health_service()
                
                # 收集各种指标
                metrics = {}
                
                # 获取最新的健康检查结果
                for name, result in health_service.last_results.items():
                    if result.component == 'database' and result.check_type == 'performance':
                        metrics['query_time_ms'] = result.details.get('query_time_ms', 0)
                    elif result.component == 'system' and result.check_type == 'memory':
                        metrics['memory_percent'] = result.details.get('percent', 0)
                    elif result.component == 'system' and result.check_type == 'cpu':
                        metrics['cpu_percent'] = result.details.get('cpu_percent', 0)
                    elif result.component == 'system' and result.check_type == 'disk':
                        metrics['disk_percent'] = result.details.get('percent', 0)
                    elif result.component == 'data' and result.check_type == 'consistency':
                        metrics['critical_issues'] = result.details.get('critical_issues', 0)
                
                # TODO: 添加API错误率计算
                metrics['error_rate_percent'] = 0  # 暂时设为0，后续从Prometheus获取
                
                # 检查降级状态
                await self.check_degradations(metrics)
                
                # 等待30秒再检查
                await asyncio.sleep(30.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"降级监控循环异常: {e}")
                await asyncio.sleep(10.0)  # 出错时等待10秒


# 全局降级监控实例
_degradation_monitor: Optional[DegradationMonitor] = None


def get_degradation_monitor() -> DegradationMonitor:
    """获取全局降级监控实例"""
    global _degradation_monitor
    if _degradation_monitor is None:
        _degradation_monitor = DegradationMonitor()
    return _degradation_monitor


async def init_degradation_monitor() -> DegradationMonitor:
    """初始化降级监控"""
    monitor = get_degradation_monitor()
    await monitor.start_monitoring()
    return monitor