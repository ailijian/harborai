#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志模块

提供结构化日志、异步日志、日志脱敏等功能。
支持控制台输出和数据库持久化。
"""

import logging
import sys
from typing import Dict, Any, Optional
from functools import lru_cache
import structlog
from structlog.stdlib import LoggerFactory

from ..config.settings import get_settings


def configure_structlog():
    """配置 structlog"""
    settings = get_settings()
    
    # 配置处理器链
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # 根据环境选择输出格式
    if settings.debug:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 配置标准库日志
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )


@lru_cache()
def get_logger(name: str = "harborai") -> structlog.BoundLogger:
    """获取结构化日志器"""
    configure_structlog()
    return structlog.get_logger(name)


class LogContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: structlog.BoundLogger, **context):
        self.logger = logger
        self.context = context
        self.bound_logger = None
    
    def __enter__(self) -> structlog.BoundLogger:
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.bound_logger.error(
                "Exception occurred in log context",
                exc_info=True,
                exception_type=exc_type.__name__,
                exception_message=str(exc_val),
            )


def sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """日志数据脱敏处理"""
    sensitive_keys = {
        "api_key", "token", "password", "secret", "authorization",
        "x-api-key", "bearer", "auth", "credential"
    }
    
    sanitized = {}
    for key, value in data.items():
        key_lower = key.lower()
        
        # 检查是否为敏感字段
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            if isinstance(value, str) and len(value) > 8:
                # 保留前4位和后4位，中间用*替代
                sanitized[key] = f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"
            else:
                sanitized[key] = "***"
        elif isinstance(value, dict):
            # 递归处理嵌套字典
            sanitized[key] = sanitize_log_data(value)
        elif isinstance(value, list):
            # 处理列表
            sanitized[key] = [sanitize_log_data(item) if isinstance(item, dict) else item for item in value]
        else:
            sanitized[key] = value
    
    return sanitized


class APICallLogger:
    """API 调用日志记录器"""
    
    def __init__(self, logger: Optional[structlog.BoundLogger] = None):
        self.logger = logger or get_logger("harborai.api")
    
    def log_request(
        self,
        method: str,
        url: str,
        params: Dict[str, Any],
        trace_id: str,
        **extra_context
    ):
        """记录请求日志"""
        sanitized_data = sanitize_log_data(params)
        
        self.logger.info(
            "API request started",
            trace_id=trace_id,
            method=method,
            url=url,
            params=sanitized_data,
            **extra_context
        )
    
    async def alog_request(
        self,
        method: str,
        url: str,
        params: Dict[str, Any],
        trace_id: str,
        **extra_context
    ):
        """异步记录请求日志"""
        self.log_request(method, url, params, trace_id, **extra_context)
    
    def log_response(
        self,
        response: Any,
        trace_id: str,
        **extra_context
    ):
        """记录响应日志"""
        response_data = {"type": type(response).__name__}
        if hasattr(response, "model_dump"):
            response_data.update(response.model_dump())
        elif hasattr(response, "__dict__"):
            response_data.update(response.__dict__)
        
        sanitized_data = sanitize_log_data(response_data)
        
        self.logger.info(
            "API request completed",
            trace_id=trace_id,
            response_data=sanitized_data,
            **extra_context
        )
    
    async def alog_response(
        self,
        response: Any,
        trace_id: str,
        **extra_context
    ):
        """异步记录响应日志"""
        self.log_response(response, trace_id, **extra_context)
    
    def log_error(
        self,
        error: Exception,
        model: str,
        plugin_name: str,
        latency_ms: float,
        trace_id: str,
        **extra_context
    ):
        """记录错误日志"""
        self.logger.error(
            "API request failed",
            trace_id=trace_id,
            model=model,
            plugin=plugin_name,
            error_type=type(error).__name__,
            error_message=str(error),
            latency_ms=latency_ms,
            exc_info=True,
            **extra_context
        )
    
    async def alog_error(
        self,
        error: Exception,
        model: str,
        plugin_name: str,
        latency_ms: float,
        trace_id: str,
        **extra_context
    ):
        """异步记录错误日志"""
        self.log_error(error, model, plugin_name, latency_ms, trace_id, **extra_context)


def create_api_logger() -> APICallLogger:
    """创建 API 调用日志记录器"""
    return APICallLogger()