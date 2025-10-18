#!/usr/bin/env python3
"""
健康检查API端点

提供系统健康状态监控、性能指标查询和故障诊断功能。
支持分布式追踪集成和实时监控数据。

作者: HarborAI团队
创建时间: 2025-01-15
版本: v1.0.0
"""

import time
import psutil
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..storage.enhanced_fallback_logger import get_enhanced_fallback_logger
from ..storage.optimized_postgresql_logger import get_optimized_postgres_logger
from ..storage.enhanced_file_logger import get_enhanced_file_logger
from ..core.tracing.data_collector import get_tracing_data_collector
from ..utils.logger import get_logger
from ..utils.timestamp import get_unified_timestamp

logger = get_logger(__name__)

# 创建健康检查路由器
health_router = APIRouter(prefix="/health", tags=["health"])


class HealthStatus(BaseModel):
    """健康状态响应模型"""
    status: str = Field(..., description="整体健康状态: healthy, degraded, unhealthy")
    timestamp: str = Field(..., description="检查时间戳")
    uptime_seconds: float = Field(..., description="系统运行时间（秒）")
    version: str = Field(default="2.0.0", description="系统版本")
    
    # 组件健康状态
    components: Dict[str, Dict[str, Any]] = Field(..., description="各组件健康状态")
    
    # 系统指标
    system_metrics: Dict[str, Any] = Field(..., description="系统性能指标")
    
    # 存储指标
    storage_metrics: Dict[str, Any] = Field(..., description="存储性能指标")
    
    # 追踪指标
    tracing_metrics: Dict[str, Any] = Field(..., description="追踪系统指标")


class DetailedHealthStatus(BaseModel):
    """详细健康状态响应模型"""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str
    
    # 详细组件信息
    postgres_logger: Optional[Dict[str, Any]] = None
    file_logger: Optional[Dict[str, Any]] = None
    fallback_logger: Optional[Dict[str, Any]] = None
    tracing_collector: Optional[Dict[str, Any]] = None
    
    # 详细系统信息
    system_info: Dict[str, Any]
    
    # 性能历史
    performance_history: List[Dict[str, Any]] = []
    
    # 错误统计
    error_statistics: Dict[str, Any] = {}


@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average: List[float]
    process_count: int
    thread_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HealthChecker:
    """健康检查器"""
    
    def __init__(self):
        self.start_time = time.time()
        self.check_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
    
    def get_system_metrics(self) -> SystemMetrics:
        """获取系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存信息
            memory = psutil.virtual_memory()
            
            # 磁盘信息
            disk = psutil.disk_usage('/')
            
            # 负载平均值（Windows上可能不可用）
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]  # Windows fallback
            
            # 进程信息
            process_count = len(psutil.pids())
            
            # 当前进程的线程数
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=disk.percent,
                disk_free_gb=disk.free / 1024 / 1024 / 1024,
                load_average=list(load_avg),
                process_count=process_count,
                thread_count=thread_count
            )
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            # 返回默认值
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                memory_available_mb=0.0,
                disk_usage_percent=0.0,
                disk_free_gb=0.0,
                load_average=[0.0, 0.0, 0.0],
                process_count=0,
                thread_count=0
            )
    
    def check_component_health(self, component_name: str, component) -> Dict[str, Any]:
        """检查组件健康状态"""
        if component is None:
            return {
                "status": "unavailable",
                "message": "Component not initialized",
                "last_check": get_unified_timestamp()
            }
        
        try:
            # 检查组件是否有健康检查方法
            if hasattr(component, 'get_health_status'):
                health_status = component.get_health_status()
                return {
                    "status": "healthy" if health_status.get("is_healthy", False) else "unhealthy",
                    "details": health_status,
                    "last_check": get_unified_timestamp()
                }
            elif hasattr(component, 'get_performance_stats'):
                stats = component.get_performance_stats()
                return {
                    "status": "healthy",
                    "details": stats,
                    "last_check": get_unified_timestamp()
                }
            else:
                return {
                    "status": "unknown",
                    "message": "No health check method available",
                    "last_check": get_unified_timestamp()
                }
                
        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "last_check": get_unified_timestamp()
            }
    
    def determine_overall_status(self, components: Dict[str, Dict[str, Any]]) -> str:
        """确定整体健康状态"""
        healthy_count = 0
        total_count = 0
        
        for component_name, component_status in components.items():
            if component_status["status"] != "unavailable":
                total_count += 1
                if component_status["status"] == "healthy":
                    healthy_count += 1
        
        if total_count == 0:
            return "unhealthy"
        
        health_ratio = healthy_count / total_count
        
        if health_ratio >= 0.8:
            return "healthy"
        elif health_ratio >= 0.5:
            return "degraded"
        else:
            return "unhealthy"
    
    def perform_health_check(self) -> HealthStatus:
        """执行健康检查"""
        timestamp = get_unified_timestamp()
        uptime = time.time() - self.start_time
        
        # 获取系统指标
        system_metrics = self.get_system_metrics()
        
        # 检查各组件健康状态
        components = {}
        
        # 检查增强降级日志记录器
        fallback_logger = get_enhanced_fallback_logger()
        components["fallback_logger"] = self.check_component_health("fallback_logger", fallback_logger)
        
        # 检查优化PostgreSQL日志记录器
        postgres_logger = get_optimized_postgres_logger()
        components["postgres_logger"] = self.check_component_health("postgres_logger", postgres_logger)
        
        # 检查增强文件日志记录器
        file_logger = get_enhanced_file_logger()
        components["file_logger"] = self.check_component_health("file_logger", file_logger)
        
        # 检查追踪数据收集器
        tracing_collector = get_tracing_data_collector()
        components["tracing_collector"] = self.check_component_health("tracing_collector", tracing_collector)
        
        # 确定整体状态
        overall_status = self.determine_overall_status(components)
        
        # 收集存储指标
        storage_metrics = {}
        if fallback_logger:
            try:
                storage_metrics = fallback_logger.get_performance_stats()
            except Exception as e:
                logger.debug(f"Failed to get storage metrics: {e}")
        
        # 收集追踪指标
        tracing_metrics = {}
        if tracing_collector:
            try:
                tracing_metrics = tracing_collector.get_performance_stats()
            except Exception as e:
                logger.debug(f"Failed to get tracing metrics: {e}")
        
        health_status = HealthStatus(
            status=overall_status,
            timestamp=timestamp,
            uptime_seconds=uptime,
            components=components,
            system_metrics=system_metrics.to_dict(),
            storage_metrics=storage_metrics,
            tracing_metrics=tracing_metrics
        )
        
        # 记录检查历史
        self._record_check_history(health_status)
        
        return health_status
    
    def _record_check_history(self, health_status: HealthStatus):
        """记录检查历史"""
        history_entry = {
            "timestamp": health_status.timestamp,
            "status": health_status.status,
            "uptime_seconds": health_status.uptime_seconds,
            "cpu_percent": health_status.system_metrics.get("cpu_percent", 0),
            "memory_percent": health_status.system_metrics.get("memory_percent", 0),
            "disk_usage_percent": health_status.system_metrics.get("disk_usage_percent", 0)
        }
        
        self.check_history.append(history_entry)
        
        # 保持历史记录大小限制
        if len(self.check_history) > self.max_history_size:
            self.check_history = self.check_history[-self.max_history_size:]


# 全局健康检查器实例
health_checker = HealthChecker()


@health_router.get("/", response_model=HealthStatus)
async def get_health_status():
    """获取系统健康状态
    
    Returns:
        HealthStatus: 系统健康状态信息
    """
    try:
        health_status = health_checker.perform_health_check()
        
        # 根据健康状态设置HTTP状态码
        status_code = 200
        if health_status.status == "degraded":
            status_code = 200  # 降级状态仍返回200，但在响应中标明
        elif health_status.status == "unhealthy":
            status_code = 503  # 服务不可用
        
        return JSONResponse(
            content=health_status.dict(),
            status_code=status_code
        )
        
    except Exception as e:
        logger.error(f"Health check endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


@health_router.get("/detailed", response_model=DetailedHealthStatus)
async def get_detailed_health_status():
    """获取详细的系统健康状态
    
    Returns:
        DetailedHealthStatus: 详细的系统健康状态信息
    """
    try:
        # 执行基础健康检查
        basic_health = health_checker.perform_health_check()
        
        # 收集详细信息
        detailed_info = {}
        
        # PostgreSQL日志记录器详细信息
        postgres_logger = get_optimized_postgres_logger()
        if postgres_logger:
            try:
                detailed_info["postgres_logger"] = postgres_logger.get_performance_stats()
            except Exception as e:
                detailed_info["postgres_logger"] = {"error": str(e)}
        
        # 文件日志记录器详细信息
        file_logger = get_enhanced_file_logger()
        if file_logger:
            try:
                detailed_info["file_logger"] = file_logger.get_health_status()
            except Exception as e:
                detailed_info["file_logger"] = {"error": str(e)}
        
        # 降级日志记录器详细信息
        fallback_logger = get_enhanced_fallback_logger()
        if fallback_logger:
            try:
                detailed_info["fallback_logger"] = {
                    "state": fallback_logger.get_state().value,
                    "health_metrics": fallback_logger.get_health_metrics(),
                    "performance_stats": fallback_logger.get_performance_stats()
                }
            except Exception as e:
                detailed_info["fallback_logger"] = {"error": str(e)}
        
        # 追踪收集器详细信息
        tracing_collector = get_tracing_data_collector()
        if tracing_collector:
            try:
                detailed_info["tracing_collector"] = tracing_collector.get_performance_stats()
            except Exception as e:
                detailed_info["tracing_collector"] = {"error": str(e)}
        
        # 系统详细信息
        system_info = {
            "platform": psutil.WINDOWS if hasattr(psutil, 'WINDOWS') else "unknown",
            "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            "cpu_count": psutil.cpu_count(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
        }
        
        # 性能历史
        performance_history = health_checker.check_history[-20:]  # 最近20次检查
        
        detailed_status = DetailedHealthStatus(
            status=basic_health.status,
            timestamp=basic_health.timestamp,
            uptime_seconds=basic_health.uptime_seconds,
            version=basic_health.version,
            postgres_logger=detailed_info.get("postgres_logger"),
            file_logger=detailed_info.get("file_logger"),
            fallback_logger=detailed_info.get("fallback_logger"),
            tracing_collector=detailed_info.get("tracing_collector"),
            system_info=system_info,
            performance_history=performance_history
        )
        
        return detailed_status
        
    except Exception as e:
        logger.error(f"Detailed health check endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detailed health check failed: {str(e)}"
        )


@health_router.get("/components/{component_name}")
async def get_component_health(component_name: str):
    """获取特定组件的健康状态
    
    Args:
        component_name: 组件名称 (postgres_logger, file_logger, fallback_logger, tracing_collector)
        
    Returns:
        Dict: 组件健康状态信息
    """
    component_map = {
        "postgres_logger": get_optimized_postgres_logger,
        "file_logger": get_enhanced_file_logger,
        "fallback_logger": get_enhanced_fallback_logger,
        "tracing_collector": get_tracing_data_collector
    }
    
    if component_name not in component_map:
        raise HTTPException(
            status_code=404,
            detail=f"Component '{component_name}' not found. Available components: {list(component_map.keys())}"
        )
    
    try:
        component = component_map[component_name]()
        health_info = health_checker.check_component_health(component_name, component)
        
        return JSONResponse(content=health_info)
        
    except Exception as e:
        logger.error(f"Component health check failed for {component_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Component health check failed: {str(e)}"
        )


@health_router.get("/metrics")
async def get_metrics():
    """获取Prometheus格式的指标数据
    
    Returns:
        str: Prometheus格式的指标数据
    """
    try:
        health_status = health_checker.perform_health_check()
        
        # 构建Prometheus格式的指标
        metrics_lines = []
        
        # 系统指标
        system_metrics = health_status.system_metrics
        metrics_lines.extend([
            f"# HELP harborai_cpu_usage_percent CPU usage percentage",
            f"# TYPE harborai_cpu_usage_percent gauge",
            f"harborai_cpu_usage_percent {system_metrics.get('cpu_percent', 0)}",
            "",
            f"# HELP harborai_memory_usage_percent Memory usage percentage",
            f"# TYPE harborai_memory_usage_percent gauge",
            f"harborai_memory_usage_percent {system_metrics.get('memory_percent', 0)}",
            "",
            f"# HELP harborai_disk_usage_percent Disk usage percentage",
            f"# TYPE harborai_disk_usage_percent gauge",
            f"harborai_disk_usage_percent {system_metrics.get('disk_usage_percent', 0)}",
            "",
            f"# HELP harborai_uptime_seconds System uptime in seconds",
            f"# TYPE harborai_uptime_seconds counter",
            f"harborai_uptime_seconds {health_status.uptime_seconds}",
            ""
        ])
        
        # 存储指标
        storage_metrics = health_status.storage_metrics
        if storage_metrics:
            metrics_lines.extend([
                f"# HELP harborai_postgres_logs_total Total PostgreSQL logs",
                f"# TYPE harborai_postgres_logs_total counter",
                f"harborai_postgres_logs_total {storage_metrics.get('postgres_logs', 0)}",
                "",
                f"# HELP harborai_file_logs_total Total file logs",
                f"# TYPE harborai_file_logs_total counter",
                f"harborai_file_logs_total {storage_metrics.get('file_logs', 0)}",
                "",
                f"# HELP harborai_storage_errors_total Total storage errors",
                f"# TYPE harborai_storage_errors_total counter",
                f"harborai_storage_errors_total {storage_metrics.get('total_errors', 0)}",
                ""
            ])
        
        # 追踪指标
        tracing_metrics = health_status.tracing_metrics
        if tracing_metrics:
            metrics_lines.extend([
                f"# HELP harborai_traces_collected_total Total traces collected",
                f"# TYPE harborai_traces_collected_total counter",
                f"harborai_traces_collected_total {tracing_metrics.get('traces_collected', 0)}",
                "",
                f"# HELP harborai_spans_created_total Total spans created",
                f"# TYPE harborai_spans_created_total counter",
                f"harborai_spans_created_total {tracing_metrics.get('spans_created', 0)}",
                ""
            ])
        
        # 健康状态指标
        status_value = 1 if health_status.status == "healthy" else 0
        metrics_lines.extend([
            f"# HELP harborai_health_status System health status (1=healthy, 0=unhealthy)",
            f"# TYPE harborai_health_status gauge",
            f"harborai_health_status {status_value}",
            ""
        ])
        
        metrics_text = "\n".join(metrics_lines)
        
        return JSONResponse(
            content=metrics_text,
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics collection failed: {str(e)}"
        )


@health_router.post("/force-check")
async def force_health_check():
    """强制执行健康检查
    
    Returns:
        Dict: 健康检查结果
    """
    try:
        # 强制执行各组件的健康检查
        fallback_logger = get_enhanced_fallback_logger()
        if fallback_logger:
            fallback_logger.force_health_check()
        
        # 执行新的健康检查
        health_status = health_checker.perform_health_check()
        
        return JSONResponse(content={
            "message": "Forced health check completed",
            "status": health_status.status,
            "timestamp": health_status.timestamp
        })
        
    except Exception as e:
        logger.error(f"Force health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Force health check failed: {str(e)}"
        )


@health_router.post("/recovery/{component_name}")
async def force_component_recovery(component_name: str):
    """强制组件恢复
    
    Args:
        component_name: 组件名称 (postgres, file)
        
    Returns:
        Dict: 恢复操作结果
    """
    fallback_logger = get_enhanced_fallback_logger()
    if not fallback_logger:
        raise HTTPException(
            status_code=404,
            detail="Fallback logger not available"
        )
    
    try:
        if component_name == "postgres":
            fallback_logger.force_postgres_recovery()
            message = "PostgreSQL recovery attempt initiated"
        elif component_name == "file":
            fallback_logger.force_file_recovery()
            message = "File system recovery attempt initiated"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid component name: {component_name}. Use 'postgres' or 'file'"
            )
        
        return JSONResponse(content={
            "message": message,
            "component": component_name,
            "timestamp": get_unified_timestamp()
        })
        
    except Exception as e:
        logger.error(f"Force recovery failed for {component_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Force recovery failed: {str(e)}"
        )


@health_router.get("/history")
async def get_health_history(
    limit: int = Query(default=50, ge=1, le=1000, description="历史记录数量限制")
):
    """获取健康检查历史
    
    Args:
        limit: 返回的历史记录数量
        
    Returns:
        Dict: 健康检查历史数据
    """
    try:
        history = health_checker.check_history[-limit:]
        
        return JSONResponse(content={
            "history": history,
            "total_records": len(health_checker.check_history),
            "returned_records": len(history),
            "oldest_timestamp": history[0]["timestamp"] if history else None,
            "newest_timestamp": history[-1]["timestamp"] if history else None
        })
        
    except Exception as e:
        logger.error(f"Health history endpoint failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health history retrieval failed: {str(e)}"
        )


# 导出路由器
__all__ = ["health_router", "HealthChecker", "health_checker"]