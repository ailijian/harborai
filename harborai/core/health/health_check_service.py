#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健康检查服务

提供数据库健康检查、服务依赖检查、系统资源监控等功能。
"""

import asyncio
import time
import psutil
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Awaitable
from datetime import datetime, timedelta

from ...utils.logger import get_logger
from ...database.postgres_connection import get_postgres_connection
from ...monitoring.prometheus_metrics import get_prometheus_metrics

logger = get_logger(__name__)


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component: str
    check_type: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    duration_ms: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'component': self.component,
            'check_type': self.check_type,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'duration_ms': self.duration_ms,
            'timestamp': self.timestamp.isoformat()
        }


class HealthCheckService:
    """健康检查服务"""
    
    def __init__(self):
        """初始化健康检查服务"""
        self.checks: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.check_intervals: Dict[str, float] = {}
        self.running = False
        self.background_task: Optional[asyncio.Task] = None
        
        # 注册默认检查项
        self._register_default_checks()
        logger.info("健康检查服务已初始化")
    
    def _register_default_checks(self):
        """注册默认健康检查项"""
        # 数据库连接检查
        self.register_check(
            'database_connection',
            self._check_database_connection,
            interval=30.0  # 30秒检查一次
        )
        
        # 数据库查询性能检查
        self.register_check(
            'database_performance',
            self._check_database_performance,
            interval=60.0  # 1分钟检查一次
        )
        
        # 系统内存检查
        self.register_check(
            'system_memory',
            self._check_system_memory,
            interval=30.0
        )
        
        # 系统CPU检查
        self.register_check(
            'system_cpu',
            self._check_system_cpu,
            interval=30.0
        )
        
        # 磁盘空间检查
        self.register_check(
            'disk_space',
            self._check_disk_space,
            interval=60.0
        )
        
        # 数据一致性检查
        self.register_check(
            'data_consistency',
            self._check_data_consistency,
            interval=300.0  # 5分钟检查一次
        )
    
    def register_check(self, name: str, check_func: Callable[[], Awaitable[HealthCheckResult]], 
                      interval: float = 60.0):
        """注册健康检查项
        
        Args:
            name: 检查项名称
            check_func: 检查函数
            interval: 检查间隔（秒）
        """
        self.checks[name] = check_func
        self.check_intervals[name] = interval
        logger.info(f"已注册健康检查项: {name}, 间隔: {interval}秒")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """运行单个健康检查
        
        Args:
            name: 检查项名称
            
        Returns:
            健康检查结果
        """
        if name not in self.checks:
            return HealthCheckResult(
                component=name,
                check_type='unknown',
                status=HealthStatus.UNKNOWN,
                message=f"未知的检查项: {name}",
                details={},
                duration_ms=0.0,
                timestamp=datetime.now()
            )
        
        start_time = time.time()
        try:
            result = await self.checks[name]()
            result.duration_ms = (time.time() - start_time) * 1000
            
            # 记录Prometheus指标
            metrics = get_prometheus_metrics()
            if metrics:
                metrics.set_health_check_status(
                    component=result.component,
                    check_type=result.check_type,
                    is_healthy=(result.status == HealthStatus.HEALTHY)
                )
                metrics.record_health_check_duration(
                    component=result.component,
                    check_type=result.check_type,
                    duration=result.duration_ms / 1000.0
                )
            
            self.last_results[name] = result
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            error_result = HealthCheckResult(
                component=name,
                check_type='error',
                status=HealthStatus.UNHEALTHY,
                message=f"检查失败: {str(e)}",
                details={'error': str(e), 'error_type': type(e).__name__},
                duration_ms=duration_ms,
                timestamp=datetime.now()
            )
            
            # 记录错误指标
            metrics = get_prometheus_metrics()
            if metrics:
                metrics.set_health_check_status(
                    component=name,
                    check_type='error',
                    is_healthy=False
                )
            
            self.last_results[name] = error_result
            logger.error(f"健康检查失败 {name}: {e}")
            return error_result
    
    async def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """运行所有健康检查
        
        Returns:
            所有检查结果的字典
        """
        tasks = []
        for name in self.checks:
            tasks.append(self.run_check(name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        result_dict = {}
        for i, name in enumerate(self.checks):
            if isinstance(results[i], Exception):
                result_dict[name] = HealthCheckResult(
                    component=name,
                    check_type='error',
                    status=HealthStatus.UNHEALTHY,
                    message=f"检查异常: {str(results[i])}",
                    details={'error': str(results[i])},
                    duration_ms=0.0,
                    timestamp=datetime.now()
                )
            else:
                result_dict[name] = results[i]
        
        return result_dict
    
    async def get_overall_health(self) -> Dict[str, Any]:
        """获取系统整体健康状态
        
        Returns:
            整体健康状态信息
        """
        results = await self.run_all_checks()
        
        # 计算整体状态
        healthy_count = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY)
        total_count = len(results)
        
        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return {
            'overall_status': overall_status.value,
            'summary': {
                'total_checks': total_count,
                'healthy': healthy_count,
                'degraded': degraded_count,
                'unhealthy': unhealthy_count,
                'health_percentage': (healthy_count / total_count * 100) if total_count > 0 else 0
            },
            'checks': {name: result.to_dict() for name, result in results.items()},
            'timestamp': datetime.now().isoformat()
        }
    
    async def start_background_monitoring(self):
        """启动后台监控"""
        if self.running:
            logger.warning("后台监控已在运行")
            return
        
        self.running = True
        self.background_task = asyncio.create_task(self._background_monitor_loop())
        logger.info("后台健康监控已启动")
    
    async def stop_background_monitoring(self):
        """停止后台监控"""
        if not self.running:
            return
        
        self.running = False
        if self.background_task:
            self.background_task.cancel()
            try:
                await self.background_task
            except asyncio.CancelledError:
                pass
        
        logger.info("后台健康监控已停止")
    
    async def _background_monitor_loop(self):
        """后台监控循环"""
        last_check_times = {name: 0.0 for name in self.checks}
        
        while self.running:
            try:
                current_time = time.time()
                
                # 检查哪些项需要运行
                checks_to_run = []
                for name, interval in self.check_intervals.items():
                    if current_time - last_check_times[name] >= interval:
                        checks_to_run.append(name)
                        last_check_times[name] = current_time
                
                # 运行需要检查的项
                if checks_to_run:
                    tasks = [self.run_check(name) for name in checks_to_run]
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # 等待1秒再检查
                await asyncio.sleep(1.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"后台监控循环异常: {e}")
                await asyncio.sleep(5.0)  # 出错时等待5秒
    
    # === 具体检查方法实现 ===
    
    async def _check_database_connection(self) -> HealthCheckResult:
        """检查数据库连接"""
        try:
            conn = get_postgres_connection()
            
            # 执行简单查询测试连接
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
            
            if result and result[0] == 1:
                return HealthCheckResult(
                    component='database',
                    check_type='connection',
                    status=HealthStatus.HEALTHY,
                    message="数据库连接正常",
                    details={'connection_status': 'active'},
                    duration_ms=0.0,
                    timestamp=datetime.now()
                )
            else:
                return HealthCheckResult(
                    component='database',
                    check_type='connection',
                    status=HealthStatus.UNHEALTHY,
                    message="数据库查询返回异常结果",
                    details={'result': result},
                    duration_ms=0.0,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return HealthCheckResult(
                component='database',
                check_type='connection',
                status=HealthStatus.UNHEALTHY,
                message=f"数据库连接失败: {str(e)}",
                details={'error': str(e)},
                duration_ms=0.0,
                timestamp=datetime.now()
            )
    
    async def _check_database_performance(self) -> HealthCheckResult:
        """检查数据库性能"""
        try:
            conn = get_postgres_connection()
            
            # 测试查询性能
            start_time = time.time()
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT COUNT(*) FROM api_logs 
                    WHERE created_at > NOW() - INTERVAL '1 hour'
                """)
                count = cursor.fetchone()[0]
            query_time = (time.time() - start_time) * 1000
            
            # 判断性能状态
            if query_time < 100:  # 100ms以下为健康
                status = HealthStatus.HEALTHY
                message = "数据库性能良好"
            elif query_time < 500:  # 500ms以下为降级
                status = HealthStatus.DEGRADED
                message = "数据库性能轻微下降"
            else:  # 500ms以上为不健康
                status = HealthStatus.UNHEALTHY
                message = "数据库性能严重下降"
            
            return HealthCheckResult(
                component='database',
                check_type='performance',
                status=status,
                message=message,
                details={
                    'query_time_ms': query_time,
                    'recent_logs_count': count
                },
                duration_ms=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                component='database',
                check_type='performance',
                status=HealthStatus.UNHEALTHY,
                message=f"数据库性能检查失败: {str(e)}",
                details={'error': str(e)},
                duration_ms=0.0,
                timestamp=datetime.now()
            )
    
    async def _check_system_memory(self) -> HealthCheckResult:
        """检查系统内存"""
        try:
            memory = psutil.virtual_memory()
            
            # 判断内存使用状态
            if memory.percent < 70:
                status = HealthStatus.HEALTHY
                message = "内存使用正常"
            elif memory.percent < 85:
                status = HealthStatus.DEGRADED
                message = "内存使用偏高"
            else:
                status = HealthStatus.UNHEALTHY
                message = "内存使用过高"
            
            # 记录内存指标
            metrics = get_prometheus_metrics()
            if metrics:
                metrics.set_memory_usage('system', memory.used)
            
            return HealthCheckResult(
                component='system',
                check_type='memory',
                status=status,
                message=message,
                details={
                    'total_gb': round(memory.total / (1024**3), 2),
                    'used_gb': round(memory.used / (1024**3), 2),
                    'available_gb': round(memory.available / (1024**3), 2),
                    'percent': memory.percent
                },
                duration_ms=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                component='system',
                check_type='memory',
                status=HealthStatus.UNHEALTHY,
                message=f"内存检查失败: {str(e)}",
                details={'error': str(e)},
                duration_ms=0.0,
                timestamp=datetime.now()
            )
    
    async def _check_system_cpu(self) -> HealthCheckResult:
        """检查系统CPU"""
        try:
            # 获取CPU使用率（1秒平均）
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 判断CPU使用状态
            if cpu_percent < 70:
                status = HealthStatus.HEALTHY
                message = "CPU使用正常"
            elif cpu_percent < 85:
                status = HealthStatus.DEGRADED
                message = "CPU使用偏高"
            else:
                status = HealthStatus.UNHEALTHY
                message = "CPU使用过高"
            
            # 记录CPU指标
            metrics = get_prometheus_metrics()
            if metrics:
                metrics.set_cpu_usage('system', cpu_percent)
            
            return HealthCheckResult(
                component='system',
                check_type='cpu',
                status=status,
                message=message,
                details={
                    'cpu_percent': cpu_percent,
                    'cpu_count': psutil.cpu_count(),
                    'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                duration_ms=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                component='system',
                check_type='cpu',
                status=HealthStatus.UNHEALTHY,
                message=f"CPU检查失败: {str(e)}",
                details={'error': str(e)},
                duration_ms=0.0,
                timestamp=datetime.now()
            )
    
    async def _check_disk_space(self) -> HealthCheckResult:
        """检查磁盘空间"""
        try:
            disk = psutil.disk_usage('/')
            
            # 判断磁盘使用状态
            if disk.percent < 80:
                status = HealthStatus.HEALTHY
                message = "磁盘空间充足"
            elif disk.percent < 90:
                status = HealthStatus.DEGRADED
                message = "磁盘空间偏少"
            else:
                status = HealthStatus.UNHEALTHY
                message = "磁盘空间不足"
            
            return HealthCheckResult(
                component='system',
                check_type='disk',
                status=status,
                message=message,
                details={
                    'total_gb': round(disk.total / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'percent': disk.percent
                },
                duration_ms=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                component='system',
                check_type='disk',
                status=HealthStatus.UNHEALTHY,
                message=f"磁盘检查失败: {str(e)}",
                details={'error': str(e)},
                duration_ms=0.0,
                timestamp=datetime.now()
            )
    
    async def _check_data_consistency(self) -> HealthCheckResult:
        """检查数据一致性"""
        try:
            from ..consistency import DataConsistencyChecker
            
            checker = DataConsistencyChecker()
            report = await checker.check_all_consistency()
            
            # 根据问题数量判断状态
            total_issues = len(report.issues)
            critical_issues = len([i for i in report.issues if i.severity == 'critical'])
            high_issues = len([i for i in report.issues if i.severity == 'high'])
            
            if critical_issues > 0:
                status = HealthStatus.UNHEALTHY
                message = f"发现{critical_issues}个严重数据一致性问题"
            elif high_issues > 0:
                status = HealthStatus.DEGRADED
                message = f"发现{high_issues}个高优先级数据一致性问题"
            elif total_issues > 0:
                status = HealthStatus.DEGRADED
                message = f"发现{total_issues}个数据一致性问题"
            else:
                status = HealthStatus.HEALTHY
                message = "数据一致性良好"
            
            return HealthCheckResult(
                component='data',
                check_type='consistency',
                status=status,
                message=message,
                details={
                    'total_issues': total_issues,
                    'critical_issues': critical_issues,
                    'high_issues': high_issues,
                    'tables_checked': len(report.table_summaries)
                },
                duration_ms=0.0,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthCheckResult(
                component='data',
                check_type='consistency',
                status=HealthStatus.UNHEALTHY,
                message=f"数据一致性检查失败: {str(e)}",
                details={'error': str(e)},
                duration_ms=0.0,
                timestamp=datetime.now()
            )


# 全局健康检查服务实例
_health_service: Optional[HealthCheckService] = None


def get_health_service() -> HealthCheckService:
    """获取全局健康检查服务实例"""
    global _health_service
    if _health_service is None:
        _health_service = HealthCheckService()
    return _health_service


async def init_health_service() -> HealthCheckService:
    """初始化健康检查服务"""
    service = get_health_service()
    await service.start_background_monitoring()
    return service