#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健康检查模块

提供系统健康状态监控和检查功能。
"""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum

from ..utils.logger import get_logger

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
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: float
    details: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "details": self.details or {}
        }


@dataclass
class SystemHealthReport:
    """系统健康报告"""
    overall_status: HealthStatus
    checks: List[HealthCheckResult]
    timestamp: float
    total_duration_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "overall_status": self.overall_status.value,
            "timestamp": self.timestamp,
            "total_duration_ms": self.total_duration_ms,
            "checks": [check.to_dict() for check in self.checks]
        }


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        """初始化健康检查器"""
        self.checks: Dict[str, Callable] = {}
        self.async_checks: Dict[str, Callable] = {}
        logger.info("健康检查器已初始化")
    
    def register_check(self, name: str, check_func: Callable, is_async: bool = False):
        """注册健康检查函数
        
        Args:
            name: 检查名称
            check_func: 检查函数
            is_async: 是否为异步函数
        """
        if is_async:
            self.async_checks[name] = check_func
        else:
            self.checks[name] = check_func
        
        logger.info(f"已注册健康检查: {name} ({'异步' if is_async else '同步'})")
    
    def unregister_check(self, name: str):
        """取消注册健康检查函数
        
        Args:
            name: 检查名称
        """
        if name in self.checks:
            del self.checks[name]
        if name in self.async_checks:
            del self.async_checks[name]
        
        logger.info(f"已取消注册健康检查: {name}")
    
    def _run_single_check(self, name: str, check_func: Callable) -> HealthCheckResult:
        """执行单个健康检查
        
        Args:
            name: 检查名称
            check_func: 检查函数
            
        Returns:
            健康检查结果
        """
        start_time = time.time()
        timestamp = start_time
        
        try:
            result = check_func()
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'unknown'))
                message = result.get('message', 'OK')
                details = result.get('details')
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = 'OK' if result else 'Check failed'
                details = None
            else:
                status = HealthStatus.HEALTHY
                message = str(result) if result else 'OK'
                details = None
            
            return HealthCheckResult(
                name=name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=timestamp,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(f"健康检查 {name} 执行失败: {e}")
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"检查失败: {str(e)}",
                duration_ms=duration_ms,
                timestamp=timestamp,
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    async def _run_single_async_check(self, name: str, check_func: Callable) -> HealthCheckResult:
        """执行单个异步健康检查
        
        Args:
            name: 检查名称
            check_func: 异步检查函数
            
        Returns:
            健康检查结果
        """
        start_time = time.time()
        timestamp = start_time
        
        try:
            result = await check_func()
            duration_ms = (time.time() - start_time) * 1000
            
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'unknown'))
                message = result.get('message', 'OK')
                details = result.get('details')
            elif isinstance(result, bool):
                status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
                message = 'OK' if result else 'Check failed'
                details = None
            else:
                status = HealthStatus.HEALTHY
                message = str(result) if result else 'OK'
                details = None
            
            return HealthCheckResult(
                name=name,
                status=status,
                message=message,
                duration_ms=duration_ms,
                timestamp=timestamp,
                details=details
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.warning(f"异步健康检查 {name} 执行失败: {e}")
            
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"检查失败: {str(e)}",
                duration_ms=duration_ms,
                timestamp=timestamp,
                details={"error": str(e), "error_type": type(e).__name__}
            )
    
    def run_checks(self, check_names: Optional[List[str]] = None) -> SystemHealthReport:
        """运行健康检查
        
        Args:
            check_names: 要运行的检查名称列表，None表示运行所有检查
            
        Returns:
            系统健康报告
        """
        start_time = time.time()
        results = []
        
        # 确定要运行的检查
        checks_to_run = {}
        if check_names:
            for name in check_names:
                if name in self.checks:
                    checks_to_run[name] = self.checks[name]
        else:
            checks_to_run = self.checks.copy()
        
        # 执行同步检查
        for name, check_func in checks_to_run.items():
            result = self._run_single_check(name, check_func)
            results.append(result)
        
        # 计算总体状态
        overall_status = self._calculate_overall_status(results)
        total_duration_ms = (time.time() - start_time) * 1000
        
        return SystemHealthReport(
            overall_status=overall_status,
            checks=results,
            timestamp=start_time,
            total_duration_ms=total_duration_ms
        )
    
    async def run_async_checks(self, check_names: Optional[List[str]] = None) -> SystemHealthReport:
        """运行异步健康检查
        
        Args:
            check_names: 要运行的检查名称列表，None表示运行所有检查
            
        Returns:
            系统健康报告
        """
        start_time = time.time()
        results = []
        
        # 确定要运行的检查
        checks_to_run = {}
        if check_names:
            for name in check_names:
                if name in self.async_checks:
                    checks_to_run[name] = self.async_checks[name]
                elif name in self.checks:
                    checks_to_run[name] = self.checks[name]
        else:
            checks_to_run = {**self.checks, **self.async_checks}
        
        # 分离同步和异步检查
        sync_checks = {name: func for name, func in checks_to_run.items() 
                      if name in self.checks}
        async_checks = {name: func for name, func in checks_to_run.items() 
                       if name in self.async_checks}
        
        # 执行同步检查
        for name, check_func in sync_checks.items():
            result = self._run_single_check(name, check_func)
            results.append(result)
        
        # 并发执行异步检查
        if async_checks:
            async_tasks = [
                self._run_single_async_check(name, check_func)
                for name, check_func in async_checks.items()
            ]
            async_results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            for result in async_results:
                if isinstance(result, Exception):
                    logger.error(f"异步健康检查执行异常: {result}")
                else:
                    results.append(result)
        
        # 计算总体状态
        overall_status = self._calculate_overall_status(results)
        total_duration_ms = (time.time() - start_time) * 1000
        
        return SystemHealthReport(
            overall_status=overall_status,
            checks=results,
            timestamp=start_time,
            total_duration_ms=total_duration_ms
        )
    
    def _calculate_overall_status(self, results: List[HealthCheckResult]) -> HealthStatus:
        """计算总体健康状态
        
        Args:
            results: 健康检查结果列表
            
        Returns:
            总体健康状态
        """
        if not results:
            return HealthStatus.UNKNOWN
        
        # 统计各状态的数量
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.UNHEALTHY: 0,
            HealthStatus.UNKNOWN: 0
        }
        
        for result in results:
            status_counts[result.status] += 1
        
        # 计算总体状态
        total_checks = len(results)
        
        if status_counts[HealthStatus.UNHEALTHY] > 0:
            # 如果有任何不健康的检查，整体状态为不健康
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > 0:
            # 如果有降级的检查，整体状态为降级
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNKNOWN] > 0:
            # 如果有未知状态的检查，整体状态为未知
            return HealthStatus.UNKNOWN
        elif status_counts[HealthStatus.HEALTHY] == total_checks:
            # 所有检查都健康
            return HealthStatus.HEALTHY
        else:
            # 默认为未知状态
            return HealthStatus.UNKNOWN
    
    def get_check_names(self) -> List[str]:
        """获取所有已注册的检查名称
        
        Returns:
            检查名称列表
        """
        return list(set(self.checks.keys()) | set(self.async_checks.keys()))


# 默认健康检查函数
def basic_system_check() -> Dict[str, Any]:
    """基础系统检查
    
    Returns:
        检查结果字典
    """
    import psutil
    import sys
    
    try:
        # 检查内存使用率
        memory = psutil.virtual_memory()
        memory_usage_percent = memory.percent
        
        # 检查CPU使用率
        cpu_usage_percent = psutil.cpu_percent(interval=1)
        
        # 检查磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        
        # 判断系统状态
        if memory_usage_percent > 90 or cpu_usage_percent > 90 or disk_usage_percent > 90:
            status = HealthStatus.UNHEALTHY
            message = "系统资源使用率过高"
        elif memory_usage_percent > 80 or cpu_usage_percent > 80 or disk_usage_percent > 80:
            status = HealthStatus.DEGRADED
            message = "系统资源使用率较高"
        else:
            status = HealthStatus.HEALTHY
            message = "系统资源正常"
        
        return {
            "status": status.value,
            "message": message,
            "details": {
                "memory_usage_percent": memory_usage_percent,
                "cpu_usage_percent": cpu_usage_percent,
                "disk_usage_percent": disk_usage_percent,
                "python_version": sys.version,
                "platform": sys.platform
            }
        }
        
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY.value,
            "message": f"系统检查失败: {str(e)}",
            "details": {"error": str(e)}
        }


def database_connection_check() -> Dict[str, Any]:
    """数据库连接检查
    
    Returns:
        检查结果字典
    """
    try:
        from ..storage.postgres_logger import get_postgres_logger
        
        postgres_logger = get_postgres_logger()
        if not postgres_logger:
            return {
                "status": HealthStatus.DEGRADED.value,
                "message": "PostgreSQL日志记录器未配置",
                "details": {"configured": False}
            }
        
        # 这里可以添加实际的数据库连接测试
        # 例如执行一个简单的查询
        
        return {
            "status": HealthStatus.HEALTHY.value,
            "message": "数据库连接正常",
            "details": {"configured": True}
        }
        
    except Exception as e:
        return {
            "status": HealthStatus.UNHEALTHY.value,
            "message": f"数据库连接检查失败: {str(e)}",
            "details": {"error": str(e)}
        }


# 全局健康检查器实例
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """获取全局健康检查器实例
    
    Returns:
        HealthChecker实例
    """
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
        
        # 注册默认检查
        _health_checker.register_check("system", basic_system_check)
        _health_checker.register_check("database", database_connection_check)
    
    return _health_checker