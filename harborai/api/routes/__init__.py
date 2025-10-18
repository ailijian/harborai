#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API路由模块

包含所有API路由定义，支持：
- 日志记录和查询
- 追踪信息查询
- 统计信息获取
- 健康检查
"""

from .logs import logs_router
from .tracing import tracing_router
from .statistics import statistics_router
from .health import health_router

__all__ = [
    'logs_router',
    'tracing_router', 
    'statistics_router',
    'health_router'
]