#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重试模块

提供智能重试、指数退避、条件重试等功能。
支持同步和异步函数的重试机制。
"""

import asyncio
import random
import time
from typing import Callable, Any, Optional, Union, Type, Tuple
from functools import wraps
from tenacity import (
    Retrying,
    AsyncRetrying,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    retry_if_result,
    before_sleep_log,
)

from .logger import get_logger
from .exceptions import (
    APIError,
    RateLimitError,
    TimeoutError,
    AuthenticationError,
)

logger = get_logger("harborai.retry")


# 默认可重试的异常类型
RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    TimeoutError,
    ConnectionError,
    # 不包括 AuthenticationError，认证错误通常不应该重试
)


def should_retry_api_error(exception: Exception) -> bool:
    """判断 API 错误是否应该重试"""
    if isinstance(exception, APIError):
        # 5xx 错误通常可以重试
        if exception.status_code and 500 <= exception.status_code < 600:
            return True
        
        # 429 (Rate Limit) 可以重试
        if exception.status_code == 429:
            return True
        
        # 408 (Request Timeout) 可以重试
        if exception.status_code == 408:
            return True
        
        # 502, 503, 504 网关错误可以重试
        if exception.status_code in (502, 503, 504):
            return True
    
    return isinstance(exception, RETRYABLE_EXCEPTIONS)


def calculate_backoff_delay(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True
) -> float:
    """计算退避延迟时间"""
    # 指数退避
    delay = base_delay * (exponential_base ** (attempt - 1))
    
    # 限制最大延迟
    delay = min(delay, max_delay)
    
    # 添加随机抖动，避免雷群效应
    if jitter:
        delay = delay * (0.5 + random.random() * 0.5)
    
    return delay


class RetryConfig:
    """重试配置类"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS,
        retry_condition: Optional[Callable[[Exception], bool]] = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.retry_condition = retry_condition or should_retry_api_error


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    trace_id: Optional[str] = None
):
    """重试装饰器（同步版本）"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # 检查是否应该重试
                    if not config.retry_condition(e):
                        logger.warning(
                            "Exception not retryable, failing immediately",
                            trace_id=trace_id,
                            exception_type=type(e).__name__,
                            exception_message=str(e),
                            attempt=attempt,
                        )
                        raise
                    
                    # 如果是最后一次尝试，直接抛出异常
                    if attempt == config.max_attempts:
                        logger.error(
                            "All retry attempts exhausted",
                            trace_id=trace_id,
                            exception_type=type(e).__name__,
                            exception_message=str(e),
                            total_attempts=attempt,
                        )
                        raise
                    
                    # 计算延迟时间
                    delay = calculate_backoff_delay(
                        attempt,
                        config.base_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.jitter,
                    )
                    
                    logger.warning(
                        "Retrying after exception",
                        trace_id=trace_id,
                        exception_type=type(e).__name__,
                        exception_message=str(e),
                        attempt=attempt,
                        max_attempts=config.max_attempts,
                        delay_seconds=delay,
                    )
                    
                    # 等待后重试
                    time.sleep(delay)
            
            # 理论上不会到达这里，但为了类型安全
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


def async_retry_with_backoff(
    config: Optional[RetryConfig] = None,
    trace_id: Optional[str] = None
):
    """异步重试装饰器"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # 检查是否应该重试
                    if not config.retry_condition(e):
                        logger.warning(
                            "Exception not retryable, failing immediately",
                            trace_id=trace_id,
                            exception_type=type(e).__name__,
                            exception_message=str(e),
                            attempt=attempt,
                        )
                        raise
                    
                    # 如果是最后一次尝试，直接抛出异常
                    if attempt == config.max_attempts:
                        logger.error(
                            "All retry attempts exhausted",
                            trace_id=trace_id,
                            exception_type=type(e).__name__,
                            exception_message=str(e),
                            total_attempts=attempt,
                        )
                        raise
                    
                    # 计算延迟时间
                    delay = calculate_backoff_delay(
                        attempt,
                        config.base_delay,
                        config.max_delay,
                        config.exponential_base,
                        config.jitter,
                    )
                    
                    logger.warning(
                        "Retrying after exception",
                        trace_id=trace_id,
                        exception_type=type(e).__name__,
                        exception_message=str(e),
                        attempt=attempt,
                        max_attempts=config.max_attempts,
                        delay_seconds=delay,
                    )
                    
                    # 异步等待后重试
                    await asyncio.sleep(delay)
            
            # 理论上不会到达这里，但为了类型安全
            if last_exception:
                raise last_exception
        
        return wrapper
    return decorator


# 使用 tenacity 库的高级重试功能
def create_tenacity_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    multiplier: float = 2.0,
    retry_condition: Optional[Callable] = None,
) -> Retrying:
    """创建 tenacity 重试器（同步版本）"""
    return Retrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        retry=retry_condition or retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def create_async_tenacity_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    multiplier: float = 2.0,
    retry_condition: Optional[Callable] = None,
) -> AsyncRetrying:
    """创建 tenacity 异步重试器"""
    return AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        retry=retry_condition or retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )