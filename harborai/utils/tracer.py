#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
追踪模块

提供全链路追踪功能，包括 Trace ID 生成、上下文管理等。
"""

import uuid
import contextvars
from typing import Optional
import time

# 当前追踪上下文
_current_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'current_trace_id', default=None
)


def generate_trace_id() -> str:
    """生成新的 Trace ID"""
    timestamp = int(time.time() * 1000)  # 毫秒时间戳
    random_part = uuid.uuid4().hex[:8]  # 8位随机字符
    return f"harborai_{timestamp}_{random_part}"


def set_current_trace_id(trace_id: str) -> None:
    """设置当前的 Trace ID"""
    _current_trace_id.set(trace_id)


def get_current_trace_id() -> Optional[str]:
    """获取当前的 Trace ID"""
    return _current_trace_id.get()


def get_or_create_trace_id(custom_trace_id: Optional[str] = None) -> str:
    """获取或创建 Trace ID"""
    if custom_trace_id:
        return custom_trace_id
    
    current = get_current_trace_id()
    if current:
        return current
    
    new_trace_id = generate_trace_id()
    set_current_trace_id(new_trace_id)
    return new_trace_id


class TraceContext:
    """追踪上下文管理器"""
    
    def __init__(self, trace_id: Optional[str] = None):
        self.trace_id = trace_id or generate_trace_id()
        self.previous_trace_id = None
    
    def __enter__(self) -> str:
        self.previous_trace_id = get_current_trace_id()
        set_current_trace_id(self.trace_id)
        return self.trace_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        set_current_trace_id(self.previous_trace_id)
    
    async def __aenter__(self) -> str:
        self.previous_trace_id = get_current_trace_id()
        set_current_trace_id(self.trace_id)
        return self.trace_id
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        set_current_trace_id(self.previous_trace_id)


class SpanTimer:
    """时间跨度计时器"""
    
    def __init__(self, name: str, trace_id: Optional[str] = None):
        self.name = name
        self.trace_id = trace_id or get_current_trace_id()
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
    
    def __enter__(self) -> 'SpanTimer':
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def get_duration_ms(self) -> Optional[float]:
        """获取持续时间（毫秒）"""
        if self.duration_ms is not None:
            return self.duration_ms
        
        if self.start_time is not None:
            current_time = time.time()
            return (current_time - self.start_time) * 1000
        
        return None
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
        }


def trace_function(func_name: Optional[str] = None):
    """函数追踪装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            with SpanTimer(name) as timer:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    # 可以在这里记录异常信息
                    raise
        
        return wrapper
    return decorator


async def trace_async_function(func_name: Optional[str] = None):
    """异步函数追踪装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            with SpanTimer(name) as timer:
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    # 可以在这里记录异常信息
                    raise
        
        return wrapper
    return decorator