#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健康检查API路由

提供系统健康检查功能，包括：
- 基础健康检查
- 详细健康状态
- 系统指标监控
- 依赖服务检查
- 降级状态监控
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

from ...core.health import (
    HealthCheckService, HealthStatus, HealthCheckResult,
    DegradationMonitor, DegradationStatus, get_degradation_monitor
)
from ...database.manager import DatabaseManager
from ...utils.logger import get_logger
from ..schemas import StandardResponse, HealthCheckResponse, ResponseBuilder

logger = get_logger("harborai.api.health")

health_router = APIRouter(prefix="/v1/health", tags=["health"])


class HealthCheckType(str, Enum):
    """健康检查类型枚举"""
    BASIC = "basic"
    DETAILED = "detailed"
    DATABASE = "database"
    SYSTEM = "system"
    DEPENDENCIES = "dependencies"
    ALL = "all"


class SystemMetrics(BaseModel):
    """系统指标"""
    cpu_usage: float = Field(description="CPU使用率")
    memory_usage: float = Field(description="内存使用率")
    disk_usage: float = Field(description="磁盘使用率")
    database_connections: int = Field(description="数据库连接数")
    active_requests: int = Field(description="活跃请求数")
    uptime_seconds: float = Field(description="运行时间（秒）")


class DependencyStatus(BaseModel):
    """依赖服务状态"""
    name: str = Field(description="服务名称")
    status: HealthStatus = Field(description="健康状态")
    response_time_ms: Optional[float] = Field(description="响应时间（毫秒）")
    last_check: datetime = Field(description="最后检查时间")
    error_message: Optional[str] = Field(description="错误信息")


class DetailedHealthStatus(BaseModel):
    """详细健康状态"""
    overall_status: HealthStatus = Field(description="总体状态")
    system_metrics: SystemMetrics = Field(description="系统指标")
    database_status: HealthCheckResult = Field(description="数据库状态")
    dependencies: List[DependencyStatus] = Field(description="依赖服务状态")
    degradation_status: DegradationStatus = Field(description="降级状态")
    last_updated: datetime = Field(description="最后更新时间")
    checks_summary: Dict[str, int] = Field(description="检查结果汇总")


async def get_health_service() -> HealthCheckService:
    """获取健康检查服务依赖"""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    
    health_service = HealthCheckService(db_manager)
    await health_service.initialize()
    return health_service


@health_router.get("/", response_model=HealthCheckResponse)
async def basic_health_check():
    """
    基础健康检查
    
    快速检查系统是否正常运行。
    """
    try:
        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            service="HarborAI"
        )
    except Exception as e:
        logger.error(f"基础健康检查失败: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            service="HarborAI",
            error=str(e)
        )


@health_router.get("/status", response_model=StandardResponse[DetailedHealthStatus])
async def get_detailed_health_status(
    check_type: HealthCheckType = Query(HealthCheckType.ALL, description="检查类型"),
    include_metrics: bool = Query(True, description="是否包含系统指标"),
    health_service: HealthCheckService = Depends(get_health_service)
):
    """
    获取详细健康状态
    
    获取系统的详细健康状态信息，包括各个组件的状态。
    """
    try:
        # 执行健康检查
        if check_type == HealthCheckType.ALL:
            health_results = await health_service.run_all_checks()
        elif check_type == HealthCheckType.DATABASE:
            health_results = {
                'database_connection': await health_service.check_database_connection(),
                'database_performance': await health_service.check_database_performance()
            }
        elif check_type == HealthCheckType.SYSTEM:
            health_results = {
                'system_memory': await health_service.check_system_memory(),
                'system_cpu': await health_service.check_system_cpu(),
                'system_disk': await health_service.check_system_disk()
            }
        else:
            health_results = await health_service.run_all_checks()
        
        # 计算总体状态
        overall_status = _calculate_overall_status(health_results)
        
        # 获取系统指标
        system_metrics = await _get_system_metrics(health_service) if include_metrics else None
        
        # 获取数据库状态
        db_status = health_results.get('database_connection', 
                                     HealthCheckResult(name="database", status=HealthStatus.UNKNOWN, message="未检查"))
        
        # 获取依赖服务状态
        dependencies = await _get_dependencies_status(health_service)
        
        # 获取降级状态
        degradation_monitor = get_degradation_monitor()
        degradation_status = degradation_monitor.get_current_status()
        
        # 检查结果汇总
        checks_summary = _summarize_checks(health_results)
        
        detailed_status = DetailedHealthStatus(
            overall_status=overall_status,
            system_metrics=system_metrics or SystemMetrics(
                cpu_usage=0, memory_usage=0, disk_usage=0,
                database_connections=0, active_requests=0, uptime_seconds=0
            ),
            database_status=db_status,
            dependencies=dependencies,
            degradation_status=degradation_status,
            last_updated=datetime.now(),
            checks_summary=checks_summary
        )
        
        return StandardResponse(
            success=True,
            data=detailed_status,
            message="获取详细健康状态成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取详细健康状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取详细健康状态失败: {str(e)}")


@health_router.get("/database", response_model=StandardResponse[Dict[str, HealthCheckResult]])
async def check_database_health(
    health_service: HealthCheckService = Depends(get_health_service)
):
    """
    检查数据库健康状态
    
    专门检查数据库相关的健康状态。
    """
    try:
        db_checks = {
            'connection': await health_service.check_database_connection(),
            'performance': await health_service.check_database_performance(),
            'data_consistency': await health_service.check_data_consistency()
        }
        
        return StandardResponse(
            success=True,
            data=db_checks,
            message="数据库健康检查完成",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"数据库健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据库健康检查失败: {str(e)}")


@health_router.get("/system", response_model=StandardResponse[Dict[str, Any]])
async def check_system_health(
    health_service: HealthCheckService = Depends(get_health_service)
):
    """
    检查系统健康状态
    
    检查系统资源使用情况。
    """
    try:
        system_checks = {
            'memory': await health_service.check_system_memory(),
            'cpu': await health_service.check_system_cpu(),
            'disk': await health_service.check_system_disk()
        }
        
        # 获取系统指标
        system_metrics = await _get_system_metrics(health_service)
        
        return StandardResponse(
            success=True,
            data={
                'checks': system_checks,
                'metrics': system_metrics
            },
            message="系统健康检查完成",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"系统健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"系统健康检查失败: {str(e)}")


@health_router.get("/dependencies", response_model=StandardResponse[List[DependencyStatus]])
async def check_dependencies_health(
    health_service: HealthCheckService = Depends(get_health_service)
):
    """
    检查依赖服务健康状态
    
    检查所有外部依赖服务的状态。
    """
    try:
        dependencies = await _get_dependencies_status(health_service)
        
        return StandardResponse(
            success=True,
            data=dependencies,
            message="依赖服务健康检查完成",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"依赖服务健康检查失败: {e}")
        raise HTTPException(status_code=500, detail=f"依赖服务健康检查失败: {str(e)}")


@health_router.get("/degradation", response_model=StandardResponse[Dict[str, Any]])
async def get_degradation_status():
    """
    获取降级状态
    
    获取当前系统的降级状态和相关信息。
    """
    try:
        degradation_monitor = get_degradation_monitor()
        
        current_status = degradation_monitor.get_current_status()
        recent_events = degradation_monitor.get_recent_events(limit=10)
        active_rules = degradation_monitor.get_active_rules()
        
        return StandardResponse(
            success=True,
            data={
                'current_status': current_status,
                'recent_events': [
                    {
                        'timestamp': event.timestamp,
                        'event_type': event.event_type,
                        'rule_name': event.rule_name,
                        'message': event.message,
                        'severity': event.severity,
                        'metadata': event.metadata
                    }
                    for event in recent_events
                ],
                'active_rules': [
                    {
                        'name': rule.name,
                        'metric': rule.metric,
                        'threshold': rule.threshold,
                        'comparison': rule.comparison,
                        'enabled': rule.enabled
                    }
                    for rule in active_rules
                ]
            },
            message="获取降级状态成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取降级状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取降级状态失败: {str(e)}")


@health_router.get("/metrics", response_model=StandardResponse[SystemMetrics])
async def get_system_metrics(
    health_service: HealthCheckService = Depends(get_health_service)
):
    """
    获取系统指标
    
    获取当前系统的各项指标。
    """
    try:
        metrics = await _get_system_metrics(health_service)
        
        return StandardResponse(
            success=True,
            data=metrics,
            message="获取系统指标成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统指标失败: {str(e)}")


@health_router.post("/check/{check_name}", response_model=StandardResponse[HealthCheckResult])
async def run_specific_check(
    check_name: str,
    health_service: HealthCheckService = Depends(get_health_service)
):
    """
    运行特定的健康检查
    
    运行指定名称的健康检查。
    """
    try:
        # 检查映射
        check_methods = {
            'database_connection': health_service.check_database_connection,
            'database_performance': health_service.check_database_performance,
            'data_consistency': health_service.check_data_consistency,
            'system_memory': health_service.check_system_memory,
            'system_cpu': health_service.check_system_cpu,
            'system_disk': health_service.check_system_disk
        }
        
        if check_name not in check_methods:
            raise HTTPException(status_code=400, detail=f"未知的检查类型: {check_name}")
        
        result = await check_methods[check_name]()
        
        return StandardResponse(
            success=True,
            data=result,
            message=f"健康检查 {check_name} 完成",
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"运行健康检查 {check_name} 失败: {e}")
        raise HTTPException(status_code=500, detail=f"运行健康检查失败: {str(e)}")


@health_router.get("/history", response_model=StandardResponse[List[Dict[str, Any]]])
async def get_health_check_history(
    check_name: Optional[str] = Query(None, description="检查名称"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    limit: int = Query(100, description="返回数量限制"),
    health_service: HealthCheckService = Depends(get_health_service)
):
    """
    获取健康检查历史
    
    获取历史健康检查记录。
    """
    try:
        # 这里应该从数据库或缓存中获取历史记录
        # 暂时返回空列表，实际实现需要根据存储方案调整
        history = []
        
        return StandardResponse(
            success=True,
            data=history,
            message="获取健康检查历史成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取健康检查历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取健康检查历史失败: {str(e)}")


# 辅助函数
def _calculate_overall_status(health_results: Dict[str, HealthCheckResult]) -> HealthStatus:
    """计算总体健康状态"""
    if not health_results:
        return HealthStatus.UNKNOWN
    
    statuses = [result.status for result in health_results.values()]
    
    # 如果有任何检查失败，总体状态为不健康
    if HealthStatus.UNHEALTHY in statuses:
        return HealthStatus.UNHEALTHY
    
    # 如果有任何检查降级，总体状态为降级
    if HealthStatus.DEGRADED in statuses:
        return HealthStatus.DEGRADED
    
    # 如果所有检查都健康，总体状态为健康
    if all(status == HealthStatus.HEALTHY for status in statuses):
        return HealthStatus.HEALTHY
    
    # 其他情况为未知
    return HealthStatus.UNKNOWN


async def _get_system_metrics(health_service: HealthCheckService) -> SystemMetrics:
    """获取系统指标"""
    import psutil
    import time
    
    # 获取系统指标
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # 获取数据库连接数（需要实现）
    db_connections = 0  # 暂时设为0，实际需要从数据库获取
    
    # 获取活跃请求数（需要实现）
    active_requests = 0  # 暂时设为0，实际需要从应用监控获取
    
    # 获取运行时间
    boot_time = psutil.boot_time()
    uptime_seconds = time.time() - boot_time
    
    return SystemMetrics(
        cpu_usage=cpu_usage,
        memory_usage=memory.percent,
        disk_usage=disk.percent,
        database_connections=db_connections,
        active_requests=active_requests,
        uptime_seconds=uptime_seconds
    )


async def _get_dependencies_status(health_service: HealthCheckService) -> List[DependencyStatus]:
    """获取依赖服务状态"""
    dependencies = []
    
    # 检查数据库
    db_result = await health_service.check_database_connection()
    dependencies.append(DependencyStatus(
        name="PostgreSQL Database",
        status=db_result.status,
        response_time_ms=db_result.duration_ms,
        last_check=datetime.now(),
        error_message=db_result.message if db_result.status != HealthStatus.HEALTHY else None
    ))
    
    # 可以添加其他依赖服务的检查
    # 例如：Redis、外部API等
    
    return dependencies


def _summarize_checks(health_results: Dict[str, HealthCheckResult]) -> Dict[str, int]:
    """汇总检查结果"""
    summary = {
        'total': len(health_results),
        'healthy': 0,
        'degraded': 0,
        'unhealthy': 0,
        'unknown': 0
    }
    
    for result in health_results.values():
        if result.status == HealthStatus.HEALTHY:
            summary['healthy'] += 1
        elif result.status == HealthStatus.DEGRADED:
            summary['degraded'] += 1
        elif result.status == HealthStatus.UNHEALTHY:
            summary['unhealthy'] += 1
        else:
            summary['unknown'] += 1
    
    return summary