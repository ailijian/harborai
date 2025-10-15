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
    AuthenticationError,
)
# 导入核心异常模块中的 TimeoutError 和 RetryableError
from ..core.exceptions import TimeoutError, RetryableError

logger = get_logger("harborai.retry")


# 默认可重试的异常类型
RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    TimeoutError,
    RetryableError,  # 包含所有继承自 RetryableError 的异常
    ConnectionError,
    # 不包括 AuthenticationError，认证错误通常不应该重试
)


def should_retry_api_error(exception: Exception) -> bool:
    """判断 API 错误是否应该重试"""
    # 首先检查是否是 RetryableError 或其子类
    if isinstance(exception, RetryableError):
        return True
    
    # 检查是否在可重试异常列表中
    if isinstance(exception, RETRYABLE_EXCEPTIONS):
        return True
    
    # 检查 API 错误的状态码
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
    
    return False





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
        exponential_base: float = 2.0,  # 保持向后兼容
        backoff_factor: Optional[float] = None,  # 新参数
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = RETRYABLE_EXCEPTIONS,
        retry_condition: Optional[Callable[[Exception], bool]] = None,
        on_retry: Optional[Callable] = None,
        on_failure: Optional[Callable] = None,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        # 向后兼容：如果提供了 backoff_factor，使用它；否则使用 exponential_base
        self.backoff_factor = backoff_factor if backoff_factor is not None else exponential_base
        self.exponential_base = exponential_base  # 保持向后兼容
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.retry_condition = retry_condition or should_retry_api_error
        self.on_retry = on_retry
        self.on_failure = on_failure


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    trace_id: Optional[str] = None
):
    """重试装饰器"""
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            retry_context = {
                "function_name": func.__name__,
                "trace_id": trace_id,
                "max_attempts": config.max_attempts,
                "start_time": time.time()
            }
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # 如果之前有重试，记录成功恢复
                    if attempt > 1:
                        logger.info(
                            "Function succeeded after retry",
                            extra={
                                **retry_context,
                                "successful_attempt": attempt,
                                "total_duration": time.time() - retry_context["start_time"]
                            }
                        )
                    
                    return result
                except Exception as e:
                    last_exception = e
                    
                    # 详细的错误上下文
                    error_context = {
                        **retry_context,
                        "attempt": attempt,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "is_retryable": config.retry_condition(e),
                        "timestamp": time.time()
                    }
                    
                    # 检查是否应该重试
                    if not config.retry_condition(e):
                        logger.warning(
                            "Exception not retryable, failing immediately",
                            extra=error_context
                        )
                        
                        # 执行失败回调
                        if config.on_failure:
                            try:
                                config.on_failure(e, attempt)
                            except Exception as callback_error:
                                logger.error(
                                    "Failure callback execution failed",
                                    extra={
                                        **error_context,
                                        "callback_error": str(callback_error)
                                    }
                                )
                        
                        raise e
                    
                    # 如果是最后一次尝试
                    if attempt == config.max_attempts:
                        logger.error(
                            "All retry attempts exhausted",
                            extra={
                                **error_context,
                                "total_duration": time.time() - retry_context["start_time"]
                            }
                        )
                        
                        # 执行失败回调
                        if config.on_failure:
                            try:
                                config.on_failure(e, attempt)
                            except Exception as callback_error:
                                logger.error(
                                    "Failure callback execution failed",
                                    extra={
                                        **error_context,
                                        "callback_error": str(callback_error)
                                    }
                                )
                        
                        raise e
                    
                    # 计算延迟时间
                    delay = min(
                        config.base_delay * (config.backoff_factor ** (attempt - 1)),
                        config.max_delay
                    )
                    
                    # 添加抖动
                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        "Function failed, retrying",
                        extra={
                            **error_context,
                            "retry_delay": delay,
                            "next_attempt": attempt + 1
                        }
                    )
                    
                    # 执行重试回调
                    if config.on_retry:
                        try:
                            config.on_retry(attempt, e, delay)
                        except Exception as callback_error:
                            logger.error(
                                "Retry callback execution failed",
                                extra={
                                    **error_context,
                                    "callback_error": str(callback_error),
                                    "retry_delay": delay
                                }
                            )
                    
                    time.sleep(delay)
            
            # 这里不应该到达，但为了安全起见
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("Unexpected retry loop termination")
        
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
            retry_context = {
                "function_name": func.__name__,
                "trace_id": trace_id,
                "max_attempts": config.max_attempts,
                "start_time": time.time()
            }
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = await func(*args, **kwargs)
                    
                    # 如果之前有重试，记录成功恢复
                    if attempt > 1:
                        logger.info(
                            "Async function succeeded after retry",
                            extra={
                                **retry_context,
                                "successful_attempt": attempt,
                                "total_duration": time.time() - retry_context["start_time"]
                            }
                        )
                    
                    return result
                except Exception as e:
                    last_exception = e
                    
                    # 详细的错误上下文
                    error_context = {
                        **retry_context,
                        "attempt": attempt,
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                        "is_retryable": config.retry_condition(e),
                        "timestamp": time.time()
                    }
                    
                    # 检查是否应该重试
                    if not config.retry_condition(e):
                        logger.warning(
                            "Async exception not retryable, failing immediately",
                            extra=error_context
                        )
                        
                        # 执行失败回调
                        if config.on_failure:
                            try:
                                if asyncio.iscoroutinefunction(config.on_failure):
                                    await config.on_failure(e, attempt)
                                else:
                                    config.on_failure(e, attempt)
                            except Exception as callback_error:
                                logger.error(
                                    "Async failure callback execution failed",
                                    extra={
                                        **error_context,
                                        "callback_error": str(callback_error)
                                    }
                                )
                        
                        raise e
                    
                    # 如果是最后一次尝试
                    if attempt == config.max_attempts:
                        logger.error(
                            "All async retry attempts exhausted",
                            extra={
                                **error_context,
                                "total_duration": time.time() - retry_context["start_time"]
                            }
                        )
                        
                        # 执行失败回调
                        if config.on_failure:
                            try:
                                if asyncio.iscoroutinefunction(config.on_failure):
                                    await config.on_failure(e, attempt)
                                else:
                                    config.on_failure(e, attempt)
                            except Exception as callback_error:
                                logger.error(
                                    "Async failure callback execution failed",
                                    extra={
                                        **error_context,
                                        "callback_error": str(callback_error)
                                    }
                                )
                        
                        raise e
                    
                    # 计算延迟时间
                    delay = min(
                        config.base_delay * (config.backoff_factor ** (attempt - 1)),
                        config.max_delay
                    )
                    
                    # 添加抖动
                    if config.jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        "Async function failed, retrying",
                        extra={
                            **error_context,
                            "retry_delay": delay,
                            "next_attempt": attempt + 1
                        }
                    )
                    
                    # 执行重试回调
                    if config.on_retry:
                        try:
                            if asyncio.iscoroutinefunction(config.on_retry):
                                await config.on_retry(attempt, e, delay)
                            else:
                                config.on_retry(attempt, e, delay)
                        except Exception as callback_error:
                            logger.error(
                                "Async retry callback execution failed",
                                extra={
                                    **error_context,
                                    "callback_error": str(callback_error),
                                    "retry_delay": delay
                                }
                            )
                    
                    await asyncio.sleep(delay)
            
            # 这里不应该到达，但为了安全起见
            if last_exception:
                raise last_exception
            else:
                raise RuntimeError("Unexpected async retry loop termination")
        
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