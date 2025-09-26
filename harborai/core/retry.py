#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 重试机制模块

提供了灵活的重试机制，支持多种重试策略和条件判断。
"""

import asyncio
import functools
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from .exceptions import (
    HarborAIError,
    RetryableError,
    NonRetryableError,
    RateLimitError,
    APIError
)

logger = logging.getLogger(__name__)


class RetryStrategy(ABC):
    """重试策略抽象基类
    
    定义重试延迟计算的接口。
    """
    
    @abstractmethod
    def calculate_delay(
        self, 
        attempt: int, 
        base_delay: float, 
        max_delay: float, 
        **kwargs
    ) -> float:
        """计算延迟时间
        
        Args:
            attempt: 当前重试次数
            base_delay: 基础延迟时间
            max_delay: 最大延迟时间
            **kwargs: 其他参数
        
        Returns:
            延迟时间（秒）
        """
        pass


class ExponentialBackoff(RetryStrategy):
    """指数退避策略
    
    延迟时间按指数增长：base_delay * (multiplier ^ (attempt - 1))
    """
    
    def calculate_delay(
        self, 
        attempt: int, 
        base_delay: float, 
        max_delay: float, 
        backoff_multiplier: float = 2.0,
        jitter: bool = False,
        **kwargs
    ) -> float:
        """计算指数退避延迟
        
        Args:
            attempt: 当前重试次数
            base_delay: 基础延迟时间
            max_delay: 最大延迟时间
            backoff_multiplier: 退避乘数
            jitter: 是否添加抖动
            **kwargs: 其他参数
        
        Returns:
            延迟时间（秒）
        """
        delay = base_delay * (backoff_multiplier ** (attempt - 1))
        
        if jitter:
            # 在应用max_delay限制之前添加抖动，确保最终结果不超过max_delay
            jitter_range = delay * 0.25  # 25%的抖动范围
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)
        
        # 确保不超过最大延迟
        delay = min(delay, max_delay)
        
        return delay


class LinearBackoff(RetryStrategy):
    """线性退避策略
    
    延迟时间线性增长：base_delay * attempt
    """
    
    def calculate_delay(
        self, 
        attempt: int, 
        base_delay: float, 
        max_delay: float, 
        jitter: bool = False,
        **kwargs
    ) -> float:
        """计算线性退避延迟
        
        Args:
            attempt: 当前重试次数
            base_delay: 基础延迟时间
            max_delay: 最大延迟时间
            jitter: 是否添加抖动
            **kwargs: 其他参数
        
        Returns:
            延迟时间（秒）
        """
        delay = base_delay * attempt
        
        if jitter:
            # 在应用max_delay限制之前添加抖动，确保最终结果不超过max_delay
            jitter_range = delay * 0.25  # 25%的抖动范围
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)
        
        # 确保不超过最大延迟
        delay = min(delay, max_delay)
        
        return delay


class FixedBackoff(RetryStrategy):
    """固定退避策略
    
    延迟时间固定不变：base_delay
    """
    
    def calculate_delay(
        self, 
        attempt: int, 
        base_delay: float, 
        max_delay: float, 
        jitter: bool = False,
        **kwargs
    ) -> float:
        """计算固定延迟
        
        Args:
            attempt: 当前重试次数
            base_delay: 基础延迟时间
            max_delay: 最大延迟时间
            jitter: 是否添加抖动
            **kwargs: 其他参数
        
        Returns:
            延迟时间（秒）
        """
        delay = base_delay
        
        if jitter:
            # 在应用max_delay限制之前添加抖动，确保最终结果不超过max_delay
            jitter_range = delay * 0.25  # 25%的抖动范围
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)
        
        # 确保不超过最大延迟
        delay = min(delay, max_delay)
        
        return delay


class FixedDelay(RetryStrategy):
    """固定延迟重试策略（别名）"""
    
    def calculate_delay(
        self, 
        attempt: int, 
        base_delay: float, 
        max_delay: float, 
        jitter: bool = False,
        **kwargs
    ) -> float:
        """计算固定延迟时间
        
        Args:
            attempt: 当前重试次数
            base_delay: 基础延迟时间
            max_delay: 最大延迟时间
            jitter: 是否添加抖动
            **kwargs: 其他参数
        
        Returns:
            延迟时间（秒）
        """
        delay = base_delay
        
        if jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        return min(delay, max_delay)


class RetryStrategyEnum(Enum):
    """重试策略枚举"""
    FIXED = "fixed"  # 固定间隔
    LINEAR = "linear"  # 线性递增
    EXPONENTIAL = "exponential"  # 指数退避
    RANDOM = "random"  # 随机间隔
    CUSTOM = "custom"  # 自定义策略


@dataclass
class RetryConfig:
    """重试配置类
    
    Args:
        max_attempts: 最大重试次数（包括首次尝试）
        strategy: 重试策略
        base_delay: 基础延迟时间（秒）
        max_delay: 最大延迟时间（秒）
        multiplier: 指数退避的乘数
        backoff_multiplier: 指数退避的乘数（multiplier的别名）
        jitter: 是否添加随机抖动
        retryable_exceptions: 可重试的异常类型
        non_retryable_exceptions: 不可重试的异常类型
        retry_condition: 自定义重试条件函数
        on_retry: 重试时的回调函数
        on_failure: 最终失败时的回调函数
    """
    max_attempts: int = 3
    strategy: RetryStrategyEnum = RetryStrategyEnum.EXPONENTIAL
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (NonRetryableError,)
    retry_condition: Optional[Callable[[Exception], bool]] = None
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    on_failure: Optional[Callable[[Exception, int], None]] = None
    
    def __post_init__(self):
        """后初始化处理，确保multiplier和backoff_multiplier同步"""
        # 如果只设置了其中一个，同步到另一个
        if hasattr(self, '_multiplier_set') or hasattr(self, '_backoff_multiplier_set'):
            return
        # 默认情况下，两者应该相等
        if self.multiplier != self.backoff_multiplier:
            # 如果不相等，优先使用backoff_multiplier
            self.multiplier = self.backoff_multiplier


class RetryManager:
    """重试管理器
    
    负责执行重试逻辑和策略计算。
    """
    
    def __init__(self, config: Optional[RetryConfig] = None, strategy: Optional[RetryStrategy] = None):
        """初始化重试管理器
        
        Args:
            config: 重试配置，如果为None则使用默认配置
            strategy: 重试策略，如果提供则覆盖config中的策略
        """
        self.config = config or RetryConfig()
        
        # 根据config.strategy设置strategy对象
        if strategy:
            self.strategy = strategy
        else:
            if self.config.strategy == RetryStrategyEnum.EXPONENTIAL:
                self.strategy = ExponentialBackoff()
            elif self.config.strategy == RetryStrategyEnum.LINEAR:
                self.strategy = LinearBackoff()
            elif self.config.strategy == RetryStrategyEnum.FIXED:
                self.strategy = FixedBackoff()
            else:
                self.strategy = ExponentialBackoff()  # 默认
        
        self._attempt_count = 0
        self._total_delay = 0.0
    
    def get_strategy_name(self) -> str:
        """获取当前策略名称
        
        Returns:
            策略名称
        """
        if isinstance(self.strategy, ExponentialBackoff):
            return 'exponential_backoff'
        elif isinstance(self.strategy, LinearBackoff):
            return 'linear_backoff'
        elif isinstance(self.strategy, FixedBackoff):
            return 'fixed_backoff'
        elif isinstance(self.strategy, FixedDelay):
            return 'fixed_delay'
        else:
            return 'unknown'
    
    def should_retry(self, exception: Exception, attempt: int = None) -> bool:
        """判断是否应该重试
        
        Args:
            exception: 发生的异常
            attempt: 当前尝试次数（可选）
        
        Returns:
            是否应该重试
        """
        # 使用传入的attempt或内部计数
        current_attempt = attempt if attempt is not None else self._attempt_count
        
        # 检查重试次数
        if current_attempt >= self.config.max_attempts:
            return False
        
        # 检查不可重试异常
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False
        
        # 检查自定义重试条件
        if self.config.retry_condition:
            return self.config.retry_condition(exception)
        
        # 检查可重试异常
        if isinstance(exception, self.config.retryable_exceptions):
            return True
        
        # 检查特殊异常类型
        if isinstance(exception, RateLimitError):
            return True
        
        if isinstance(exception, APIError):
            # 5xx错误通常可以重试
            if exception.status_code and 500 <= exception.status_code < 600:
                return True
        
        return False
    
    def calculate_delay(self, attempt: int, exception: Exception = None) -> float:
        """计算延迟时间
        
        Args:
            attempt: 当前重试次数
            exception: 发生的异常
        
        Returns:
            延迟时间（秒）
        """
        # 如果是RateLimitError且有retry_after，优先使用
        if isinstance(exception, RateLimitError) and hasattr(exception, 'retry_after') and exception.retry_after:
            delay = float(exception.retry_after)
        else:
            # 如果有strategy对象，使用它来计算延迟
            if hasattr(self, 'strategy') and self.strategy:
                delay = self.strategy.calculate_delay(
                    attempt=attempt,
                    base_delay=self.config.base_delay,
                    max_delay=self.config.max_delay,
                    backoff_multiplier=self.config.backoff_multiplier,
                    jitter=self.config.jitter
                )
            else:
                # 根据策略枚举计算延迟（向后兼容）
                if self.config.strategy == RetryStrategyEnum.FIXED:
                    delay = self.config.base_delay
                elif self.config.strategy == RetryStrategyEnum.LINEAR:
                    delay = self.config.base_delay * attempt
                elif self.config.strategy == RetryStrategyEnum.EXPONENTIAL:
                    delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
                elif self.config.strategy == RetryStrategyEnum.RANDOM:
                    delay = random.uniform(0, self.config.base_delay * attempt)
                else:
                    delay = self.config.base_delay
                
                # 限制最大延迟
                delay = min(delay, self.config.max_delay)
                
                # 添加随机抖动
                if self.config.jitter:
                    jitter_range = delay * 0.1  # 10%的抖动
                    delay += random.uniform(-jitter_range, jitter_range)
                    delay = max(0, delay)  # 确保延迟不为负数
        
        return delay
    
    def execute_retry_callback(self, attempt: int, exception: Exception, delay: float):
        """执行重试回调
        
        Args:
            attempt: 当前重试次数
            exception: 发生的异常
            delay: 延迟时间
        """
        if self.config.on_retry:
            try:
                self.config.on_retry(attempt, exception, delay)
            except Exception as e:
                logger.warning(f"Error in retry callback: {e}")
        
        logger.info(
            f"Retrying attempt {attempt}/{self.config.max_attempts} "
            f"after {delay:.2f}s due to: {exception}"
        )
    
    def execute_failure_callback(self, exception: Exception, attempts: int):
        """执行失败回调
        
        Args:
            exception: 最终的异常
            attempts: 总尝试次数
        """
        if self.config.on_failure:
            try:
                self.config.on_failure(exception, attempts)
            except Exception as e:
                logger.warning(f"Error in failure callback: {e}")
        
        logger.error(
            f"All {attempts} retry attempts failed. Final error: {exception}"
        )
    
    def execute(self, func: Callable, *args, **kwargs):
        """执行函数并处理重试
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            函数执行结果
        
        Raises:
            最后一次执行的异常
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            self._attempt_count = attempt
            
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Function succeeded on attempt {attempt}")
                return result
            
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    break
                
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt, e)
                    self.execute_retry_callback(attempt, e, delay)
                    time.sleep(delay)
                    self._total_delay += delay
        
        # 所有重试都失败了
        self.execute_failure_callback(last_exception, self.config.max_attempts)
        raise last_exception
    
    async def execute_async(self, func: Callable, *args, **kwargs):
        """异步执行函数并处理重试
        
        Args:
            func: 要执行的异步函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            函数执行结果
        
        Raises:
            最后一次执行的异常
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            self._attempt_count = attempt
            
            try:
                result = await func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Async function succeeded on attempt {attempt}")
                return result
            
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    break
                
                if attempt < self.config.max_attempts:
                    delay = self.calculate_delay(attempt, e)
                    self.execute_retry_callback(attempt, e, delay)
                    await asyncio.sleep(delay)
                    self._total_delay += delay
        
        # 所有重试都失败了
        self.execute_failure_callback(last_exception, self.config.max_attempts)
        raise last_exception
    
    def reset(self):
        """重置重试状态"""
        self._attempt_count = 0
        self._total_delay = 0.0
    
    def execute_with_retry(self, func: Callable, *args, **kwargs):
        """使用重试机制执行函数（别名方法）
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            函数执行结果
        """
        return self.execute(func, *args, **kwargs)
    
    def execute_with_strategy(self, func: Callable, strategy: str = None, *args, **kwargs):
        """使用指定策略执行函数
        
        Args:
            func: 要执行的函数
            strategy: 重试策略名称
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            函数执行结果
        """
        if strategy:
            # 临时更改策略
            original_strategy = self.config.strategy
            if strategy == "exponential_backoff":
                self.config.strategy = RetryStrategyEnum.EXPONENTIAL
            elif strategy == "linear_backoff":
                self.config.strategy = RetryStrategyEnum.LINEAR
            elif strategy == "fixed_delay":
                self.config.strategy = RetryStrategyEnum.FIXED
            
            try:
                return self.execute(func, *args, **kwargs)
            finally:
                # 恢复原策略
                self.config.strategy = original_strategy
        else:
            return self.execute(func, *args, **kwargs)
    
    def execute_with_timeout(self, func: Callable, max_attempts: int = 3, timeout: float = 30.0, *args, **kwargs):
        """带超时的执行函数
        
        Args:
            func: 要执行的函数
            max_attempts: 最大重试次数
            timeout: 总超时时间（秒）
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            函数执行结果和尝试记录
        
        Raises:
            TimeoutError: 超时异常
        """
        import signal
        import threading
        
        start_time = time.time()
        attempts = []
        last_exception = None
        
        for attempt in range(1, max_attempts + 1):
            attempt_start = time.time()
            
            # 检查总超时
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                raise TimeoutError(f"Total timeout exceeded after {timeout}s")
            
            try:
                # 计算剩余时间
                remaining_time = timeout - elapsed
                if remaining_time <= 0:
                    raise TimeoutError("No time left for attempt")
                
                # 执行函数（简化版本，实际应该实现真正的超时控制）
                result = func(*args, **kwargs)
                
                attempts.append({
                    "attempt": attempt,
                    "success": True,
                    "result": result,
                    "duration": time.time() - attempt_start
                })
                
                return result, attempts
                
            except Exception as e:
                last_exception = e
                attempts.append({
                    "attempt": attempt,
                    "success": False,
                    "error": str(e),
                    "duration": time.time() - attempt_start
                })
                
                # 检查是否应该重试
                if attempt < max_attempts and self.should_retry(e, attempt):
                    # 计算延迟
                    delay = self.calculate_delay(attempt, e)
                    
                    # 检查延迟后是否会超时
                    if time.time() - start_time + delay >= timeout:
                        raise TimeoutError("Would exceed timeout with retry delay")
                    
                    # 执行延迟
                    time.sleep(delay)
        
        # 所有尝试都失败了
        if last_exception:
            raise last_exception
        else:
            raise TimeoutError("All attempts failed or timed out")


def retry(
    max_attempts: int = 3,
    strategy: RetryStrategyEnum = RetryStrategyEnum.EXPONENTIAL,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError, RateLimitError),
    non_retryable_exceptions: Tuple[Type[Exception], ...] = (NonRetryableError,),
    retry_condition: Optional[Callable[[Exception], bool]] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    on_failure: Optional[Callable[[Exception, int], None]] = None
):
    """重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        strategy: 重试策略
        base_delay: 基础延迟时间
        max_delay: 最大延迟时间
        multiplier: 指数退避乘数
        jitter: 是否添加随机抖动
        retryable_exceptions: 可重试的异常类型
        non_retryable_exceptions: 不可重试的异常类型
        retry_condition: 自定义重试条件
        on_retry: 重试回调
        on_failure: 失败回调
    
    Returns:
        装饰器函数
    """
    config = RetryConfig(
        max_attempts=max_attempts,
        strategy=strategy,
        base_delay=base_delay,
        max_delay=max_delay,
        multiplier=multiplier,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions,
        non_retryable_exceptions=non_retryable_exceptions,
        retry_condition=retry_condition,
        on_retry=on_retry,
        on_failure=on_failure
    )
    
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            return _async_retry_wrapper(func, config)
        else:
            return _sync_retry_wrapper(func, config)
    
    return decorator


def _sync_retry_wrapper(func: Callable, config: RetryConfig) -> Callable:
    """同步函数重试包装器"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        retry_manager = RetryManager(config)
        last_exception = None
        
        for attempt in range(1, config.max_attempts + 1):
            retry_manager._attempt_count = attempt
            
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Function succeeded on attempt {attempt}")
                return result
            
            except Exception as e:
                last_exception = e
                
                if not retry_manager.should_retry(e):
                    break
                
                if attempt < config.max_attempts:
                    delay = retry_manager.calculate_delay(attempt, e)
                    retry_manager.execute_retry_callback(attempt, e, delay)
                    time.sleep(delay)
                    retry_manager._total_delay += delay
        
        # 所有重试都失败了
        retry_manager.execute_failure_callback(last_exception, config.max_attempts)
        raise last_exception
    
    return wrapper


def _async_retry_wrapper(func: Callable, config: RetryConfig) -> Callable:
    """异步函数重试包装器"""
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        retry_manager = RetryManager(config)
        last_exception = None
        
        for attempt in range(1, config.max_attempts + 1):
            retry_manager._attempt_count = attempt
            
            try:
                result = await func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Function succeeded on attempt {attempt}")
                return result
            
            except Exception as e:
                last_exception = e
                
                if not retry_manager.should_retry(e):
                    break
                
                if attempt < config.max_attempts:
                    delay = retry_manager.calculate_delay(attempt, e)
                    retry_manager.execute_retry_callback(attempt, e, delay)
                    await asyncio.sleep(delay)
                    retry_manager._total_delay += delay
        
        # 所有重试都失败了
        retry_manager.execute_failure_callback(last_exception, config.max_attempts)
        raise last_exception
    
    return wrapper


class RetryableOperation:
    """可重试操作类
    
    提供了更灵活的重试控制，支持手动重试管理。
    """
    
    def __init__(self, config: RetryConfig):
        """初始化可重试操作
        
        Args:
            config: 重试配置
        """
        self.config = config
        self.retry_manager = RetryManager(config)
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """执行可重试操作（同步）
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            函数执行结果
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            self.retry_manager._attempt_count = attempt
            
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Operation succeeded on attempt {attempt}")
                return result
            
            except Exception as e:
                last_exception = e
                
                if not self.retry_manager.should_retry(e):
                    break
                
                if attempt < self.config.max_attempts:
                    delay = self.retry_manager.calculate_delay(attempt, e)
                    self.retry_manager.execute_retry_callback(attempt, e, delay)
                    time.sleep(delay)
        
        # 所有重试都失败了
        self.retry_manager.execute_failure_callback(last_exception, self.config.max_attempts)
        raise last_exception
    
    async def execute_async(self, func: Callable, *args, **kwargs) -> Any:
        """执行可重试操作（异步）
        
        Args:
            func: 要执行的异步函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            函数执行结果
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            self.retry_manager._attempt_count = attempt
            
            try:
                result = await func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Operation succeeded on attempt {attempt}")
                return result
            
            except Exception as e:
                last_exception = e
                
                if not self.retry_manager.should_retry(e):
                    break
                
                if attempt < self.config.max_attempts:
                    delay = self.retry_manager.calculate_delay(attempt, e)
                    self.retry_manager.execute_retry_callback(attempt, e, delay)
                    await asyncio.sleep(delay)
        
        # 所有重试都失败了
        self.retry_manager.execute_failure_callback(last_exception, self.config.max_attempts)
        raise last_exception


# 预定义的重试配置
DEFAULT_RETRY_CONFIG = RetryConfig()

API_RETRY_CONFIG = RetryConfig(
    max_attempts=5,
    strategy=RetryStrategyEnum.EXPONENTIAL,
    base_delay=1.0,
    max_delay=30.0,
    multiplier=2.0,
    jitter=True,
    retryable_exceptions=(RetryableError, RateLimitError, APIError)
)

QUICK_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    strategy=RetryStrategyEnum.FIXED,
    base_delay=0.5,
    max_delay=5.0,
    jitter=False
)

AGGRESSIVE_RETRY_CONFIG = RetryConfig(
    max_attempts=10,
    strategy=RetryStrategyEnum.EXPONENTIAL,
    base_delay=0.1,
    max_delay=60.0,
    multiplier=1.5,
    jitter=True
)


def retry_on_exception(
    retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError,),
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True
):
    """基于异常类型的重试装饰器
    
    Args:
        retryable_exceptions: 可重试的异常类型
        max_attempts: 最大重试次数
        base_delay: 基础延迟时间
        max_delay: 最大延迟时间
        backoff_multiplier: 退避乘数
        jitter: 是否添加抖动
    
    Returns:
        装饰器函数
    """
    return retry(
        max_attempts=max_attempts,
        strategy=RetryStrategyEnum.EXPONENTIAL,
        base_delay=base_delay,
        max_delay=max_delay,
        multiplier=backoff_multiplier,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions
    )


def async_retry_on_exception(
    retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError,),
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_multiplier: float = 2.0,
    jitter: bool = True
):
    """基于异常类型的异步重试装饰器
    
    Args:
        retryable_exceptions: 可重试的异常类型
        max_attempts: 最大重试次数
        base_delay: 基础延迟时间
        max_delay: 最大延迟时间
        backoff_multiplier: 退避乘数
        jitter: 是否添加抖动
    
    Returns:
        装饰器函数
    """
    return retry(
        max_attempts=max_attempts,
        strategy=RetryStrategyEnum.EXPONENTIAL,
        base_delay=base_delay,
        max_delay=max_delay,
        multiplier=backoff_multiplier,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions
    )


class CircuitBreaker:
    """熔断器实现
    
    当连续失败次数达到阈值时，熔断器会打开，阻止后续请求。
    经过一定时间后，熔断器会进入半开状态，允许少量请求通过。
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        """初始化熔断器
        
        Args:
            failure_threshold: 失败阈值
            recovery_timeout: 恢复超时时间
            expected_exception: 预期的异常类型
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """通过熔断器调用函数
        
        Args:
            func: 要调用的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        Returns:
            函数执行结果
        
        Raises:
            CircuitBreakerOpenError: 熔断器处于开启状态
        """
        if not self.can_execute():
            raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise
    
    def can_execute(self) -> bool:
        """检查是否可以执行
        
        Returns:
            是否可以执行
        """
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """记录成功"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
    
    def get_state(self) -> str:
        """获取熔断器当前状态
        
        Returns:
            熔断器状态: 'closed', 'open', 或 'half_open'
        """
        return self.state.lower()
    
    def _on_success(self):
        """成功时的处理"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """失败时的处理"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class RetryPolicy:
    """重试策略类
    
    封装了重试的各种配置和行为。
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        delay_strategy: str = 'exponential',
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (RetryableError,),
        non_retryable_exceptions: Tuple[Type[Exception], ...] = (NonRetryableError,)
    ):
        """初始化重试策略
        
        Args:
            max_attempts: 最大重试次数
            delay_strategy: 延迟策略
            base_delay: 基础延迟时间
            max_delay: 最大延迟时间
            multiplier: 指数退避乘数
            jitter: 是否添加抖动
            retryable_exceptions: 可重试的异常类型
            non_retryable_exceptions: 不可重试的异常类型
        """
        self.max_attempts = max_attempts
        self.delay_strategy = delay_strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
        self.non_retryable_exceptions = non_retryable_exceptions
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """判断是否应该重试
        
        Args:
            exception: 发生的异常
            attempt: 当前重试次数
        
        Returns:
            是否应该重试
        """
        if attempt >= self.max_attempts:
            return False
        
        if isinstance(exception, self.non_retryable_exceptions):
            return False
        
        if isinstance(exception, self.retryable_exceptions):
            return True
        
        return False
    
    def calculate_delay(self, attempt: int) -> float:
        """计算延迟时间
        
        Args:
            attempt: 当前重试次数
        
        Returns:
            延迟时间（秒）
        """
        if self.delay_strategy == 'fixed':
            delay = self.base_delay
        elif self.delay_strategy == 'linear':
            delay = self.base_delay * attempt
        elif self.delay_strategy == 'exponential':
            delay = self.base_delay * (self.multiplier ** (attempt - 1))
        else:
            delay = self.base_delay
        
        if self.jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        return min(delay, self.max_delay)