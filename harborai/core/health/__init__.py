#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
健康检查模块

提供系统健康检查、状态监控和降级状态管理功能。
"""

from .health_check_service import (
    HealthCheckService,
    HealthStatus,
    HealthCheckResult,
    get_health_service,
    init_health_service
)

from .degradation_monitor import (
    DegradationMonitor,
    DegradationStatus,
    DegradationEvent,
    DegradationRule,
    get_degradation_monitor,
    init_degradation_monitor
)

__all__ = [
    'HealthCheckService',
    'HealthStatus', 
    'HealthCheckResult',
    'DegradationMonitor',
    'DegradationStatus',
    'DegradationEvent',
    'DegradationRule',
    'get_health_service',
    'init_health_service',
    'get_degradation_monitor',
    'init_degradation_monitor'
]