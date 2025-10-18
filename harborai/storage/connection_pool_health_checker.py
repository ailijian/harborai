#!/usr/bin/env python3
"""
连接池健康检查器模块

负责监控和检查数据库连接池的健康状态，包括：
- 连接池状态监控
- 连接质量评估
- 性能指标收集
- 健康报告生成
- 自动恢复建议

作者: HarborAI团队
创建时间: 2025-01-15
版本: v1.0.0
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from contextlib import asynccontextmanager

import structlog

from .connection_pool import ConnectionPool, ConnectionPoolConfig, ConnectionStats
from ..utils.exceptions import StorageError


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    status: HealthStatus
    score: float  # 0-100分
    timestamp: datetime
    details: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class ConnectionMetrics:
    """连接指标"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    connection_utilization: float = 0.0  # 连接利用率 (0-1)
    average_response_time: float = 0.0  # 平均响应时间(ms)
    error_rate: float = 0.0  # 错误率 (0-1)
    throughput: float = 0.0  # 吞吐量 (requests/second)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class PerformanceThresholds:
    """性能阈值配置"""
    max_response_time_ms: float = 1000.0  # 最大响应时间
    max_error_rate: float = 0.05  # 最大错误率 5%
    min_connection_utilization: float = 0.1  # 最小连接利用率 10%
    max_connection_utilization: float = 0.8  # 最大连接利用率 80%
    min_throughput: float = 1.0  # 最小吞吐量
    health_check_timeout: float = 5.0  # 健康检查超时时间


class ConnectionPoolHealthChecker:
    """
    连接池健康检查器
    
    功能：
    1. 监控连接池状态和性能
    2. 评估连接质量和健康度
    3. 收集性能指标和统计信息
    4. 生成健康报告和建议
    5. 提供自动恢复策略
    """
    
    def __init__(self,
                 connection_pool: ConnectionPool,
                 thresholds: Optional[PerformanceThresholds] = None,
                 check_interval: float = 30.0,
                 enable_auto_monitoring: bool = True):
        """
        初始化连接池健康检查器
        
        参数:
            connection_pool: 要监控的连接池
            thresholds: 性能阈值配置
            check_interval: 检查间隔（秒）
            enable_auto_monitoring: 是否启用自动监控
        """
        self.logger = structlog.get_logger(__name__)
        
        # 核心组件
        self.connection_pool = connection_pool
        self.thresholds = thresholds or PerformanceThresholds()
        self.check_interval = check_interval
        self.enable_auto_monitoring = enable_auto_monitoring
        
        # 状态管理
        self._is_running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._last_check_time: Optional[datetime] = None
        self._check_history: List[HealthCheckResult] = []
        self._max_history_size = 100
        
        # 性能指标
        self._metrics_history: List[ConnectionMetrics] = []
        self._performance_window = timedelta(minutes=5)  # 5分钟性能窗口
        
        # 回调函数
        self._health_change_callbacks: List[Callable[[HealthCheckResult], None]] = []
        
        # 锁
        self._lock = threading.RLock()
        
        self.logger.info(
            "连接池健康检查器初始化完成",
            check_interval=check_interval,
            auto_monitoring=enable_auto_monitoring
        )
    
    async def start_monitoring(self) -> None:
        """启动自动监控"""
        if self._is_running:
            self.logger.warning("健康检查器已在运行")
            return
        
        self._is_running = True
        
        if self.enable_auto_monitoring:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("连接池健康监控已启动")
    
    async def stop_monitoring(self) -> None:
        """停止自动监控"""
        self._is_running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            
        self.logger.info("连接池健康监控已停止")
    
    async def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._is_running:
            try:
                # 执行健康检查
                result = await self.check_health()
                
                # 记录检查历史
                with self._lock:
                    self._check_history.append(result)
                    if len(self._check_history) > self._max_history_size:
                        self._check_history.pop(0)
                
                # 触发回调
                await self._trigger_health_callbacks(result)
                
                # 等待下次检查
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "健康检查监控循环异常",
                    error=str(e)
                )
                await asyncio.sleep(self.check_interval)
    
    async def check_health(self) -> HealthCheckResult:
        """
        执行健康检查
        
        返回:
            HealthCheckResult: 健康检查结果
        """
        start_time = time.time()
        timestamp = datetime.now(timezone.utc)
        
        try:
            # 收集连接池指标
            metrics = await self._collect_metrics()
            
            # 评估健康状态
            status, score, details = self._evaluate_health(metrics)
            
            # 生成建议
            recommendations = self._generate_recommendations(status, metrics)
            
            # 创建结果
            result = HealthCheckResult(
                status=status,
                score=score,
                timestamp=timestamp,
                details=details,
                recommendations=recommendations
            )
            
            self._last_check_time = timestamp
            
            # 记录性能指标
            with self._lock:
                self._metrics_history.append(metrics)
                # 清理过期指标
                cutoff_time = timestamp - self._performance_window
                self._metrics_history = [
                    m for m in self._metrics_history 
                    if hasattr(m, 'timestamp') and m.timestamp > cutoff_time
                ]
            
            check_duration = (time.time() - start_time) * 1000
            
            self.logger.debug(
                "健康检查完成",
                status=status.value,
                score=score,
                duration_ms=check_duration
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "健康检查失败",
                error=str(e)
            )
            
            return HealthCheckResult(
                status=HealthStatus.UNKNOWN,
                score=0.0,
                timestamp=timestamp,
                details={"error": str(e)},
                recommendations=["检查连接池配置和网络连接"]
            )
    
    async def _collect_metrics(self) -> ConnectionMetrics:
        """收集连接池指标"""
        try:
            # 获取连接池统计信息
            stats = self.connection_pool._stats
            config = self.connection_pool.config
            
            # 检查stats是否为None
            if stats is None:
                raise ValueError("连接池统计信息不可用")
            
            # 计算连接利用率
            total_connections = stats.total_connections
            active_connections = stats.active_connections
            utilization = (active_connections / config.max_connections) if config.max_connections > 0 else 0
            
            # 计算错误率
            total_requests = stats.total_requests
            failed_requests = stats.failed_requests
            error_rate = (failed_requests / total_requests) if total_requests > 0 else 0
            
            # 计算吞吐量（基于最近的请求）
            throughput = self._calculate_throughput()
            
            metrics = ConnectionMetrics(
                total_connections=total_connections,
                active_connections=active_connections,
                idle_connections=total_connections - active_connections,
                failed_connections=stats.failed_connections,
                connection_utilization=utilization,
                average_response_time=stats.average_response_time,
                error_rate=error_rate,
                throughput=throughput
            )
            
            # 添加时间戳
            metrics.timestamp = datetime.now(timezone.utc)
            
            return metrics
            
        except Exception as e:
            self.logger.error(
                "收集连接池指标失败",
                error=str(e)
            )
            # 抛出异常以便上层处理
            raise
    
    def _calculate_throughput(self) -> float:
        """计算吞吐量"""
        try:
            if len(self._metrics_history) < 2:
                return 0.0
            
            # 获取最近两次的指标
            current_metrics = self._metrics_history[-1] if self._metrics_history else None
            previous_metrics = self._metrics_history[-2] if len(self._metrics_history) > 1 else None
            
            if not current_metrics or not previous_metrics:
                return 0.0
            
            # 计算时间差和请求差
            time_diff = (current_metrics.timestamp - previous_metrics.timestamp).total_seconds()
            if time_diff <= 0:
                return 0.0
            
            # 这里需要从连接池获取请求计数，暂时返回估算值
            return max(0.0, current_metrics.total_connections / time_diff)
            
        except Exception:
            return 0.0
    
    def _evaluate_health(self, metrics: ConnectionMetrics) -> tuple[HealthStatus, float, Dict[str, Any]]:
        """
        评估健康状态
        
        参数:
            metrics: 连接指标
            
        返回:
            tuple: (状态, 分数, 详细信息)
        """
        score = 100.0
        issues = []
        warnings = []
        
        # 检查响应时间
        if metrics.average_response_time > self.thresholds.max_response_time_ms:
            score -= 20
            issues.append(f"响应时间过高: {metrics.average_response_time:.2f}ms")
        elif metrics.average_response_time > self.thresholds.max_response_time_ms * 0.8:
            score -= 10
            warnings.append(f"响应时间偏高: {metrics.average_response_time:.2f}ms")
        
        # 检查错误率
        if metrics.error_rate > self.thresholds.max_error_rate:
            score -= 25
            issues.append(f"错误率过高: {metrics.error_rate:.2%}")
        elif metrics.error_rate > self.thresholds.max_error_rate * 0.5:
            score -= 10
            warnings.append(f"错误率偏高: {metrics.error_rate:.2%}")
        
        # 检查连接利用率
        if metrics.connection_utilization > self.thresholds.max_connection_utilization:
            score -= 15
            issues.append(f"连接利用率过高: {metrics.connection_utilization:.2%}")
        elif metrics.connection_utilization < self.thresholds.min_connection_utilization:
            score -= 5
            warnings.append(f"连接利用率过低: {metrics.connection_utilization:.2%}")
        
        # 检查吞吐量
        if metrics.throughput < self.thresholds.min_throughput:
            score -= 10
            warnings.append(f"吞吐量偏低: {metrics.throughput:.2f} req/s")
        
        # 检查失败连接
        if metrics.failed_connections > 0:
            failure_rate = metrics.failed_connections / max(metrics.total_connections, 1)
            if failure_rate > 0.1:  # 10%失败率
                score -= 20
                issues.append(f"连接失败率过高: {failure_rate:.2%}")
            else:
                score -= 5
                warnings.append(f"存在失败连接: {metrics.failed_connections}")
        
        # 确保分数在0-100范围内
        score = max(0.0, min(100.0, score))
        
        # 确定状态
        if score >= 80:
            status = HealthStatus.HEALTHY
        elif score >= 60:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.CRITICAL
        
        # 构建详细信息
        details = {
            "metrics": metrics.to_dict(),
            "score_breakdown": {
                "total_score": score,
                "issues": issues,
                "warnings": warnings
            },
            "thresholds": asdict(self.thresholds)
        }
        
        return status, score, details
    
    def _generate_recommendations(self, status: HealthStatus, metrics: ConnectionMetrics) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if status == HealthStatus.CRITICAL:
            recommendations.append("立即检查数据库连接和网络状态")
            recommendations.append("考虑重启连接池或增加连接数")
        
        if metrics.average_response_time > self.thresholds.max_response_time_ms:
            recommendations.append("优化数据库查询性能")
            recommendations.append("检查数据库服务器负载")
        
        if metrics.error_rate > self.thresholds.max_error_rate:
            recommendations.append("检查数据库错误日志")
            recommendations.append("验证连接字符串和权限")
        
        if metrics.connection_utilization > self.thresholds.max_connection_utilization:
            recommendations.append("增加最大连接数")
            recommendations.append("优化连接使用模式")
        elif metrics.connection_utilization < self.thresholds.min_connection_utilization:
            recommendations.append("减少最小连接数以节省资源")
        
        if metrics.failed_connections > 0:
            recommendations.append("检查网络连接稳定性")
            recommendations.append("增加连接重试机制")
        
        if not recommendations:
            recommendations.append("连接池运行正常，继续监控")
        
        return recommendations
    
    async def _trigger_health_callbacks(self, result: HealthCheckResult) -> None:
        """触发健康状态变化回调"""
        for callback in self._health_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                self.logger.error(
                    "健康状态回调执行失败",
                    error=str(e)
                )
    
    def add_health_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """添加健康状态变化回调"""
        self._health_change_callbacks.append(callback)
    
    def remove_health_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """移除健康状态变化回调"""
        if callback in self._health_change_callbacks:
            self._health_change_callbacks.remove(callback)
    
    def get_current_status(self) -> Optional[HealthCheckResult]:
        """获取当前健康状态"""
        with self._lock:
            return self._check_history[-1] if self._check_history else None
    
    def get_health_history(self, limit: int = 50) -> List[HealthCheckResult]:
        """获取健康检查历史"""
        with self._lock:
            return self._check_history[-limit:] if self._check_history else []
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self._lock:
            if not self._metrics_history:
                return {}
            
            recent_metrics = self._metrics_history[-10:]  # 最近10次
            
            return {
                "current": recent_metrics[-1].to_dict() if recent_metrics else {},
                "average": self._calculate_average_metrics(recent_metrics),
                "trend": self._calculate_trend(recent_metrics),
                "last_updated": self._last_check_time.isoformat() if self._last_check_time else None
            }
    
    def _calculate_average_metrics(self, metrics_list: List[ConnectionMetrics]) -> Dict[str, float]:
        """计算平均指标"""
        if not metrics_list:
            return {}
        
        total_count = len(metrics_list)
        
        return {
            "average_response_time": sum(m.average_response_time for m in metrics_list) / total_count,
            "average_utilization": sum(m.connection_utilization for m in metrics_list) / total_count,
            "average_error_rate": sum(m.error_rate for m in metrics_list) / total_count,
            "average_throughput": sum(m.throughput for m in metrics_list) / total_count
        }
    
    def _calculate_trend(self, metrics_list: List[ConnectionMetrics]) -> Dict[str, str]:
        """计算趋势"""
        if len(metrics_list) < 2:
            return {}
        
        first_half = metrics_list[:len(metrics_list)//2]
        second_half = metrics_list[len(metrics_list)//2:]
        
        def get_trend(first_avg: float, second_avg: float, threshold: float = 0.01) -> str:
            diff = abs(first_avg - second_avg)
            if diff < threshold:  # 变化很小
                return "stable"
            return "improving" if second_avg < first_avg else "degrading"
        
        first_avg_response = sum(m.average_response_time for m in first_half) / len(first_half)
        second_avg_response = sum(m.average_response_time for m in second_half) / len(second_half)
        
        first_avg_error = sum(m.error_rate for m in first_half) / len(first_half)
        second_avg_error = sum(m.error_rate for m in second_half) / len(second_half)
        
        return {
            "response_time": get_trend(first_avg_response, second_avg_response, 10.0),  # 响应时间阈值10ms
            "error_rate": get_trend(first_avg_error, second_avg_error, 0.001)  # 错误率阈值0.1%
        }
    
    async def force_health_check(self) -> HealthCheckResult:
        """强制执行健康检查"""
        return await self.check_health()
    
    def reset_history(self) -> None:
        """重置检查历史"""
        with self._lock:
            self._check_history.clear()
            self._metrics_history.clear()
        
        self.logger.info("健康检查历史已重置")


# 全局健康检查器实例
_global_health_checker: Optional[ConnectionPoolHealthChecker] = None


def get_global_health_checker() -> Optional[ConnectionPoolHealthChecker]:
    """获取全局健康检查器实例"""
    return _global_health_checker


def setup_global_health_checker(
    connection_pool: ConnectionPool,
    thresholds: Optional[PerformanceThresholds] = None
) -> ConnectionPoolHealthChecker:
    """设置全局健康检查器实例"""
    global _global_health_checker
    _global_health_checker = ConnectionPoolHealthChecker(connection_pool, thresholds)
    return _global_health_checker