#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重试机制模块单元测试"""

import asyncio
import time
import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Any, Callable, Optional

from harborai.core.retry import (
    RetryConfig,
    RetryStrategy,
    ExponentialBackoff,
    LinearBackoff,
    FixedBackoff,
    RetryManager,
    retry_on_exception,
    async_retry_on_exception
)
from harborai.core.exceptions import (
    HarborAIError,
    RetryableError,
    NonRetryableError,
    RateLimitError,
    DatabaseError,
    AuthenticationError
)


class TestRetryConfig:
    """RetryConfig配置类测试"""
    
    def test_default_initialization(self):
        """测试默认初始化"""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_multiplier == 2.0
        assert config.jitter is True
        assert config.retryable_exceptions == (RetryableError,)
    
    def test_custom_initialization(self):
        """测试自定义初始化"""
        custom_exceptions = (RateLimitError, DatabaseError)
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            backoff_multiplier=3.0,
            jitter=False,
            retryable_exceptions=custom_exceptions
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.backoff_multiplier == 3.0
        assert config.jitter is False
        assert config.retryable_exceptions == custom_exceptions
    
    def test_validation(self):
        """测试参数验证"""
        # 测试有效参数
        config = RetryConfig(max_attempts=1, base_delay=0.1, max_delay=1.0)
        assert config.max_attempts == 1
        
        # 测试边界值
        config = RetryConfig(max_attempts=10, base_delay=0.0, max_delay=3600.0)
        assert config.max_attempts == 10
        assert config.base_delay == 0.0
        assert config.max_delay == 3600.0


class TestRetryStrategy:
    """RetryStrategy基础策略类测试"""
    
    def test_abstract_nature(self):
        """测试抽象类特性"""
        # RetryStrategy是抽象类，不能直接实例化
        with pytest.raises(TypeError):
            RetryStrategy()
    
    def test_subclass_implementation(self):
        """测试子类实现"""
        class CustomStrategy(RetryStrategy):
            def calculate_delay(self, attempt: int, base_delay: float, max_delay: float, **kwargs) -> float:
                return min(base_delay * attempt, max_delay)
        
        strategy = CustomStrategy()
        delay = strategy.calculate_delay(2, 1.0, 10.0)
        assert delay == 2.0


class TestExponentialBackoff:
    """ExponentialBackoff指数退避策略测试"""
    
    def test_basic_calculation(self):
        """测试基础计算"""
        strategy = ExponentialBackoff()
        
        # 测试不同尝试次数的延迟计算
        delay1 = strategy.calculate_delay(1, 1.0, 60.0, backoff_multiplier=2.0)
        delay2 = strategy.calculate_delay(2, 1.0, 60.0, backoff_multiplier=2.0)
        delay3 = strategy.calculate_delay(3, 1.0, 60.0, backoff_multiplier=2.0)
        
        # 指数增长：1.0, 2.0, 4.0
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0
    
    def test_max_delay_limit(self):
        """测试最大延迟限制"""
        strategy = ExponentialBackoff()
        
        # 测试超过最大延迟的情况
        delay = strategy.calculate_delay(10, 1.0, 5.0, backoff_multiplier=2.0)
        assert delay == 5.0  # 应该被限制在max_delay
    
    def test_custom_multiplier(self):
        """测试自定义倍数"""
        strategy = ExponentialBackoff()
        
        delay1 = strategy.calculate_delay(1, 1.0, 60.0, backoff_multiplier=3.0)
        delay2 = strategy.calculate_delay(2, 1.0, 60.0, backoff_multiplier=3.0)
        
        assert delay1 == 1.0
        assert delay2 == 3.0  # 1.0 * 3.0
    
    def test_jitter_application(self):
        """测试抖动应用"""
        strategy = ExponentialBackoff()
        
        # 测试多次计算，验证抖动效果
        delays = []
        for _ in range(10):
            delay = strategy.calculate_delay(2, 1.0, 60.0, backoff_multiplier=2.0, jitter=True)
            delays.append(delay)
        
        # 所有延迟应该在合理范围内
        for delay in delays:
            assert 1.0 <= delay <= 3.0  # 基础延迟2.0，抖动范围±50%
        
        # 应该有一些变化（不是所有值都相同）
        assert len(set(delays)) > 1
    
    def test_no_jitter(self):
        """测试无抖动"""
        strategy = ExponentialBackoff()
        
        delay1 = strategy.calculate_delay(2, 1.0, 60.0, backoff_multiplier=2.0, jitter=False)
        delay2 = strategy.calculate_delay(2, 1.0, 60.0, backoff_multiplier=2.0, jitter=False)
        
        # 无抖动时，相同参数应该产生相同结果
        assert delay1 == delay2 == 2.0


class TestLinearBackoff:
    """LinearBackoff线性退避策略测试"""
    
    def test_basic_calculation(self):
        """测试基础计算"""
        strategy = LinearBackoff()
        
        delay1 = strategy.calculate_delay(1, 1.0, 60.0)
        delay2 = strategy.calculate_delay(2, 1.0, 60.0)
        delay3 = strategy.calculate_delay(3, 1.0, 60.0)
        
        # 线性增长：1.0, 2.0, 3.0
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 3.0
    
    def test_max_delay_limit(self):
        """测试最大延迟限制"""
        strategy = LinearBackoff()
        
        delay = strategy.calculate_delay(10, 1.0, 5.0)
        assert delay == 5.0  # 应该被限制在max_delay
    
    def test_jitter_application(self):
        """测试抖动应用"""
        strategy = LinearBackoff()
        
        delays = []
        for _ in range(10):
            delay = strategy.calculate_delay(2, 1.0, 60.0, jitter=True)
            delays.append(delay)
        
        # 所有延迟应该在合理范围内
        for delay in delays:
            assert 1.0 <= delay <= 3.0  # 基础延迟2.0，抖动范围±50%
        
        # 应该有一些变化
        assert len(set(delays)) > 1


class TestFixedBackoff:
    """FixedBackoff固定退避策略测试"""
    
    def test_basic_calculation(self):
        """测试基础计算"""
        strategy = FixedBackoff()
        
        delay1 = strategy.calculate_delay(1, 2.0, 60.0)
        delay2 = strategy.calculate_delay(5, 2.0, 60.0)
        delay3 = strategy.calculate_delay(10, 2.0, 60.0)
        
        # 固定延迟：所有尝试都是相同的延迟
        assert delay1 == delay2 == delay3 == 2.0
    
    def test_max_delay_respect(self):
        """测试遵守最大延迟"""
        strategy = FixedBackoff()
        
        delay = strategy.calculate_delay(1, 10.0, 5.0)
        assert delay == 5.0  # 应该被限制在max_delay
    
    def test_jitter_application(self):
        """测试抖动应用"""
        strategy = FixedBackoff()
        
        delays = []
        for _ in range(10):
            delay = strategy.calculate_delay(1, 2.0, 60.0, jitter=True)
            delays.append(delay)
        
        # 所有延迟应该在合理范围内
        for delay in delays:
            assert 1.0 <= delay <= 3.0  # 基础延迟2.0，抖动范围±50%
        
        # 应该有一些变化
        assert len(set(delays)) > 1


class TestRetryManager:
    """RetryManager重试管理器测试"""
    
    def test_initialization_with_default_config(self):
        """测试使用默认配置初始化"""
        manager = RetryManager()
        
        assert isinstance(manager.config, RetryConfig)
        assert isinstance(manager.strategy, ExponentialBackoff)
    
    def test_initialization_with_custom_config(self):
        """测试使用自定义配置初始化"""
        config = RetryConfig(max_attempts=5, base_delay=2.0)
        strategy = LinearBackoff()
        manager = RetryManager(config=config, strategy=strategy)
        
        assert manager.config is config
        assert manager.strategy is strategy
    
    def test_should_retry_with_retryable_exception(self):
        """测试可重试异常的重试判断"""
        manager = RetryManager()
        
        # 测试可重试异常
        retryable_exc = RateLimitError("Rate limited")
        assert manager.should_retry(retryable_exc, 1) is True
        assert manager.should_retry(retryable_exc, 2) is True
    
    def test_should_retry_with_non_retryable_exception(self):
        """测试不可重试异常的重试判断"""
        manager = RetryManager()
        
        # 测试不可重试异常
        non_retryable_exc = AuthenticationError("Auth failed")
        assert manager.should_retry(non_retryable_exc, 1) is False
    
    def test_should_retry_with_max_attempts_exceeded(self):
        """测试超过最大尝试次数的重试判断"""
        config = RetryConfig(max_attempts=3)
        manager = RetryManager(config=config)
        
        retryable_exc = RateLimitError("Rate limited")
        assert manager.should_retry(retryable_exc, 3) is False  # 已达到最大尝试次数
        assert manager.should_retry(retryable_exc, 4) is False  # 超过最大尝试次数
    
    def test_calculate_delay(self):
        """测试延迟计算"""
        config = RetryConfig(base_delay=1.0, max_delay=10.0, backoff_multiplier=2.0, jitter=False)
        strategy = ExponentialBackoff()
        manager = RetryManager(config=config, strategy=strategy)
        
        delay1 = manager.calculate_delay(1)
        delay2 = manager.calculate_delay(2)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
    
    def test_execute_sync_success(self):
        """测试同步执行成功"""
        manager = RetryManager()
        
        def successful_func():
            return "success"
        
        result = manager.execute(successful_func)
        assert result == "success"
    
    def test_execute_sync_with_retries(self):
        """测试同步执行带重试"""
        config = RetryConfig(max_attempts=3, base_delay=0.01)  # 快速测试
        manager = RetryManager(config=config)
        
        call_count = 0
        
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited")
            return "success"
        
        result = manager.execute(failing_then_success)
        assert result == "success"
        assert call_count == 3
    
    def test_execute_sync_max_attempts_exceeded(self):
        """测试同步执行超过最大尝试次数"""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        manager = RetryManager(config=config)
        
        def always_failing():
            raise RateLimitError("Always fails")
        
        with pytest.raises(RateLimitError):
            manager.execute(always_failing)
    
    def test_execute_sync_non_retryable_exception(self):
        """测试同步执行不可重试异常"""
        manager = RetryManager()
        
        def auth_failing():
            raise AuthenticationError("Auth failed")
        
        with pytest.raises(AuthenticationError):
            manager.execute(auth_failing)
    
    @pytest.mark.asyncio
    async def test_execute_async_success(self):
        """测试异步执行成功"""
        manager = RetryManager()
        
        async def successful_async_func():
            return "async_success"
        
        result = await manager.execute_async(successful_async_func)
        assert result == "async_success"
    
    @pytest.mark.asyncio
    async def test_execute_async_with_retries(self):
        """测试异步执行带重试"""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        manager = RetryManager(config=config)
        
        call_count = 0
        
        async def failing_then_success_async():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DatabaseError("DB connection failed")
            return "async_success"
        
        result = await manager.execute_async(failing_then_success_async)
        assert result == "async_success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_execute_async_max_attempts_exceeded(self):
        """测试异步执行超过最大尝试次数"""
        config = RetryConfig(max_attempts=2, base_delay=0.01)
        manager = RetryManager(config=config)
        
        async def always_failing_async():
            raise DatabaseError("Always fails")
        
        with pytest.raises(DatabaseError):
            await manager.execute_async(always_failing_async)
    
    def test_execute_with_args_and_kwargs(self):
        """测试带参数的执行"""
        manager = RetryManager()
        
        def func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"
        
        result = manager.execute(func_with_args, "arg1", "arg2", c="kwarg1")
        assert result == "arg1-arg2-kwarg1"
    
    @pytest.mark.asyncio
    async def test_execute_async_with_args_and_kwargs(self):
        """测试带参数的异步执行"""
        manager = RetryManager()
        
        async def async_func_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"
        
        result = await manager.execute_async(async_func_with_args, "arg1", "arg2", c="kwarg1")
        assert result == "arg1-arg2-kwarg1"


class TestRetryDecorators:
    """重试装饰器测试"""
    
    def test_retry_on_exception_decorator_success(self):
        """测试重试装饰器成功情况"""
        @retry_on_exception(max_attempts=3, base_delay=0.01)
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
    
    def test_retry_on_exception_decorator_with_retries(self):
        """测试重试装饰器带重试"""
        call_count = 0
        
        @retry_on_exception(max_attempts=3, base_delay=0.01)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited")
            return "success"
        
        result = failing_then_success()
        assert result == "success"
        assert call_count == 3
    
    def test_retry_on_exception_decorator_max_attempts(self):
        """测试重试装饰器最大尝试次数"""
        @retry_on_exception(max_attempts=2, base_delay=0.01)
        def always_failing():
            raise RateLimitError("Always fails")
        
        with pytest.raises(RateLimitError):
            always_failing()
    
    def test_retry_on_exception_decorator_non_retryable(self):
        """测试重试装饰器不可重试异常"""
        @retry_on_exception(max_attempts=3, base_delay=0.01)
        def auth_failing():
            raise AuthenticationError("Auth failed")
        
        with pytest.raises(AuthenticationError):
            auth_failing()
    
    def test_retry_on_exception_decorator_with_args(self):
        """测试重试装饰器带参数"""
        @retry_on_exception(max_attempts=3, base_delay=0.01)
        def function_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"
        
        result = function_with_args("arg1", "arg2", c="kwarg1")
        assert result == "arg1-arg2-kwarg1"
    
    @pytest.mark.asyncio
    async def test_async_retry_on_exception_decorator_success(self):
        """测试异步重试装饰器成功情况"""
        @async_retry_on_exception(max_attempts=3, base_delay=0.01)
        async def successful_async_function():
            return "async_success"
        
        result = await successful_async_function()
        assert result == "async_success"
    
    @pytest.mark.asyncio
    async def test_async_retry_on_exception_decorator_with_retries(self):
        """测试异步重试装饰器带重试"""
        call_count = 0
        
        @async_retry_on_exception(max_attempts=3, base_delay=0.01)
        async def failing_then_success_async():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DatabaseError("DB failed")
            return "async_success"
        
        result = await failing_then_success_async()
        assert result == "async_success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_on_exception_decorator_max_attempts(self):
        """测试异步重试装饰器最大尝试次数"""
        @async_retry_on_exception(max_attempts=2, base_delay=0.01)
        async def always_failing_async():
            raise DatabaseError("Always fails")
        
        with pytest.raises(DatabaseError):
            await always_failing_async()
    
    @pytest.mark.asyncio
    async def test_async_retry_on_exception_decorator_with_args(self):
        """测试异步重试装饰器带参数"""
        @async_retry_on_exception(max_attempts=3, base_delay=0.01)
        async def async_function_with_args(a, b, c=None):
            return f"{a}-{b}-{c}"
        
        result = await async_function_with_args("arg1", "arg2", c="kwarg1")
        assert result == "arg1-arg2-kwarg1"


class TestRetryIntegration:
    """重试机制集成测试"""
    
    def test_rate_limit_with_retry_after(self):
        """测试带重试时间的速率限制"""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        manager = RetryManager(config=config)
        
        call_count = 0
        
        def rate_limited_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited", retry_after=1)
            return "success"
        
        start_time = time.time()
        result = manager.execute(rate_limited_func)
        end_time = time.time()
        
        assert result == "success"
        assert call_count == 3
        # 验证实际等待了一些时间（虽然我们设置的base_delay很小）
        assert end_time - start_time >= 0.01
    
    def test_custom_retry_strategy_integration(self):
        """测试自定义重试策略集成"""
        class CustomStrategy(RetryStrategy):
            def calculate_delay(self, attempt: int, base_delay: float, max_delay: float, **kwargs) -> float:
                # 自定义策略：第一次重试0.1秒，之后每次增加0.1秒
                return min(0.1 * attempt, max_delay)
        
        config = RetryConfig(max_attempts=3, base_delay=1.0, max_delay=1.0)
        strategy = CustomStrategy()
        manager = RetryManager(config=config, strategy=strategy)
        
        delays = []
        for attempt in range(1, 4):
            delay = manager.calculate_delay(attempt)
            delays.append(delay)
        
        # 使用近似比较避免浮点数精度问题
        expected_delays = [0.1, 0.2, 0.3]
        assert len(delays) == len(expected_delays)
        for actual, expected in zip(delays, expected_delays):
            assert abs(actual - expected) < 1e-10
    
    def test_mixed_exception_types(self):
        """测试混合异常类型"""
        config = RetryConfig(
            max_attempts=4,
            base_delay=0.01,
            retryable_exceptions=(RateLimitError, DatabaseError)
        )
        manager = RetryManager(config=config)
        
        call_count = 0
        
        def mixed_exceptions_func():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise RateLimitError("Rate limited")  # 可重试
            elif call_count == 2:
                raise DatabaseError("DB error")  # 可重试
            elif call_count == 3:
                raise AuthenticationError("Auth error")  # 不可重试
            else:
                return "success"
        
        # 应该在第3次调用时抛出AuthenticationError（不可重试）
        with pytest.raises(AuthenticationError):
            manager.execute(mixed_exceptions_func)
        
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_integration_with_real_delays(self):
        """测试异步集成与真实延迟"""
        config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)
        manager = RetryManager(config=config)
        
        call_count = 0
        start_time = time.time()
        
        async def delayed_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise DatabaseError("Temporary failure")
            return "success"
        
        result = await manager.execute_async(delayed_success)
        end_time = time.time()
        
        assert result == "success"
        assert call_count == 3
        # 验证实际等待了预期的时间（至少两次延迟：0.1 + 0.2 = 0.3秒）
        assert end_time - start_time >= 0.25  # 允许一些误差
    
    def test_decorator_and_manager_consistency(self):
        """测试装饰器和管理器的一致性"""
        # 使用管理器
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        manager = RetryManager(config=config)
        
        call_count_manager = 0
        
        def func_for_manager():
            nonlocal call_count_manager
            call_count_manager += 1
            if call_count_manager < 3:
                raise RateLimitError("Rate limited")
            return "manager_success"
        
        result_manager = manager.execute(func_for_manager)
        
        # 使用装饰器
        call_count_decorator = 0
        
        @retry_on_exception(max_attempts=3, base_delay=0.01)
        def func_for_decorator():
            nonlocal call_count_decorator
            call_count_decorator += 1
            if call_count_decorator < 3:
                raise RateLimitError("Rate limited")
            return "decorator_success"
        
        result_decorator = func_for_decorator()
        
        # 两种方式应该产生相同的行为
        assert result_manager == "manager_success"
        assert result_decorator == "decorator_success"
        assert call_count_manager == call_count_decorator == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])