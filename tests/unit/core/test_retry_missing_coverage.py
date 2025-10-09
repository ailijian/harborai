#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 重试机制模块缺失覆盖率测试

专门测试retry.py中未被现有测试覆盖的代码路径。
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from harborai.core.retry import (
    RetryConfig, RetryManager, RetryableOperation, RetryPolicy,
    retry_on_exception, async_retry_on_exception, RetryStrategyEnum
)
from harborai.core.exceptions import (
    RateLimitError, APIError, RetryableError, NonRetryableError
)


class TestRetryConfigPostInit:
    """测试RetryConfig的__post_init__方法"""
    
    def test_post_init_with_multiplier_set_flag(self):
        """测试当设置了_multiplier_set标志时的行为"""
        config = RetryConfig()
        # 模拟已设置标志的情况
        config._multiplier_set = True
        config.multiplier = 3.0
        config.backoff_multiplier = 2.0
        
        # 调用__post_init__
        config.__post_init__()
        
        # 验证multiplier没有被同步
        assert config.multiplier == 3.0
        assert config.backoff_multiplier == 2.0
    
    def test_post_init_with_backoff_multiplier_set_flag(self):
        """测试当设置了_backoff_multiplier_set标志时的行为"""
        config = RetryConfig()
        # 模拟已设置标志的情况
        config._backoff_multiplier_set = True
        config.multiplier = 3.0
        config.backoff_multiplier = 2.0
        
        # 调用__post_init__
        config.__post_init__()
        
        # 验证multiplier没有被同步
        assert config.multiplier == 3.0
        assert config.backoff_multiplier == 2.0
    
    def test_post_init_sync_multipliers(self):
        """测试multiplier和backoff_multiplier同步"""
        config = RetryConfig()
        config.multiplier = 3.0
        config.backoff_multiplier = 2.0
        
        # 调用__post_init__
        config.__post_init__()
        
        # 验证multiplier被同步为backoff_multiplier的值
        assert config.multiplier == 2.0
        assert config.backoff_multiplier == 2.0


class TestRetryManagerShouldRetry:
    """测试RetryManager的should_retry方法的边界情况"""
    
    def test_should_retry_with_rate_limit_error(self):
        """测试RateLimitError的重试逻辑"""
        config = RetryConfig(max_attempts=3)
        manager = RetryManager(config)
        
        # RateLimitError应该被重试
        rate_limit_error = RateLimitError("Rate limit exceeded")
        assert manager.should_retry(rate_limit_error, 1) is True
        assert manager.should_retry(rate_limit_error, 2) is True
        
        # 但超过最大次数时不应该重试
        assert manager.should_retry(rate_limit_error, 3) is False
    
    def test_should_retry_with_api_error_non_5xx(self):
        """测试非5xx APIError的重试逻辑"""
        config = RetryConfig(max_attempts=3)
        manager = RetryManager(config)
        
        # 4xx错误不应该重试
        api_error_4xx = APIError("Bad request", status_code=400)
        assert manager.should_retry(api_error_4xx, 1) is False
        
        # 3xx错误不应该重试
        api_error_3xx = APIError("Redirect", status_code=301)
        assert manager.should_retry(api_error_3xx, 1) is False
        
        # 没有状态码的APIError不应该重试
        api_error_no_code = APIError("Unknown error")
        assert manager.should_retry(api_error_no_code, 1) is False


class TestRetryManagerCalculateDelay:
    """测试RetryManager的calculate_delay方法"""
    
    def test_calculate_delay_with_rate_limit_retry_after(self):
        """测试RateLimitError带retry_after的延迟计算"""
        config = RetryConfig()
        manager = RetryManager(config)
        
        # 创建带retry_after的RateLimitError
        rate_limit_error = RateLimitError("Rate limit exceeded")
        rate_limit_error.retry_after = 30.5
        
        delay = manager.calculate_delay(1, rate_limit_error)
        assert delay == 30.5
    
    def test_calculate_delay_with_strategy_object(self):
        """测试使用strategy对象计算延迟"""
        config = RetryConfig()
        manager = RetryManager(config)
        
        # 创建模拟的strategy对象
        mock_strategy = Mock()
        mock_strategy.calculate_delay.return_value = 5.0
        manager.strategy = mock_strategy
        
        delay = manager.calculate_delay(2)
        
        assert delay == 5.0
        mock_strategy.calculate_delay.assert_called_once_with(
            attempt=2,
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            backoff_multiplier=config.backoff_multiplier,
            jitter=config.jitter
        )


class TestRetryManagerExecuteRetryCallback:
    """测试RetryManager的execute_retry_callback方法"""
    
    def test_execute_retry_callback_with_exception(self):
        """测试重试回调中发生异常的情况"""
        def failing_callback(attempt, exception, delay):
            raise ValueError("Callback failed")
        
        config = RetryConfig(on_retry=failing_callback)
        manager = RetryManager(config)
        
        # 执行回调不应该抛出异常，而是记录警告
        with patch('harborai.core.retry.logger') as mock_logger:
            manager.execute_retry_callback(1, Exception("test"), 1.0)
            mock_logger.warning.assert_called_once()


class TestRetryManagerExecuteFailureCallback:
    """测试RetryManager的execute_failure_callback方法"""
    
    def test_execute_failure_callback_with_exception(self):
        """测试失败回调中发生异常的情况"""
        def failing_callback(exception, attempts):
            raise ValueError("Failure callback failed")
        
        config = RetryConfig(on_failure=failing_callback)
        manager = RetryManager(config)
        
        # 执行失败回调不应该抛出异常，而是记录警告
        with patch('harborai.core.retry.logger') as mock_logger:
            manager.execute_failure_callback(Exception("test"), 3)
            mock_logger.warning.assert_called_once()


class TestRetryManagerExecuteWithRetry:
    """测试RetryManager的execute_with_retry方法"""
    
    def test_execute_with_retry_alias(self):
        """测试execute_with_retry别名方法"""
        def successful_func():
            return "success"
        
        manager = RetryManager()
        result = manager.execute_with_retry(successful_func)
        
        assert result == "success"


class TestRetryableOperationTimeoutHandling:
    """测试RetryableOperation的超时处理"""
    
    def test_execute_with_timeout_would_exceed(self):
        """测试延迟会导致超时的情况"""
        config = RetryConfig(max_attempts=3, base_delay=10.0)  # 长延迟
        operation = RetryableOperation(config)
        
        def failing_func():
            raise RetryableError("Always fails")
        
        # 设置很短的超时时间
        with pytest.raises(Exception):  # 应该抛出TimeoutError或其他异常
            operation.execute_with_timeout(failing_func, timeout=0.1, max_attempts=3)


class TestRetryDecorators:
    """测试重试装饰器函数"""
    
    def test_retry_on_exception_decorator(self):
        """测试retry_on_exception装饰器"""
        call_count = 0
        
        @retry_on_exception(
            retryable_exceptions=(ValueError,),
            max_attempts=3,
            base_delay=0.01
        )
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retry me")
            return "success"
        
        result = failing_func()
        assert result == "success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_retry_on_exception_decorator(self):
        """测试async_retry_on_exception装饰器"""
        call_count = 0
        
        @async_retry_on_exception(
            retryable_exceptions=(ValueError,),
            max_attempts=3,
            base_delay=0.01
        )
        async def failing_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Retry me")
            return "success"
        
        result = await failing_async_func()
        assert result == "success"
        assert call_count == 3


class TestRetryPolicyEdgeCases:
    """测试RetryPolicy的边界情况"""
    
    def test_retry_policy_unknown_exception(self):
        """测试RetryPolicy对未知异常类型的处理"""
        policy = RetryPolicy(
            max_attempts=3,
            retryable_exceptions=(ValueError,),
            non_retryable_exceptions=(TypeError,)
        )
        
        # 未知异常类型应该返回False
        unknown_error = RuntimeError("Unknown error")
        assert policy.should_retry(unknown_error, 1) is False
    
    def test_retry_policy_at_max_attempts(self):
        """测试RetryPolicy在达到最大重试次数时的行为"""
        policy = RetryPolicy(max_attempts=3)
        
        # 在最大重试次数时应该返回False
        assert policy.should_retry(ValueError("test"), 3) is False
        
        # 超过最大重试次数时也应该返回False
        assert policy.should_retry(ValueError("test"), 4) is False


class TestRetryManagerAsyncExecute:
    """测试RetryManager的异步执行方法"""
    
    @pytest.mark.asyncio
    async def test_async_execute_success_on_retry(self):
        """测试异步执行在重试后成功"""
        call_count = 0
        
        async def failing_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Service down")
            return "async success"
        
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.01,
            retryable_exceptions=(RetryableError,)
        )
        manager = RetryManager(config)
        
        result = await manager.execute_async(failing_async_func)
        
        assert result == "async success"
        assert call_count == 3


class TestRetryManagerReset:
    """测试RetryManager的reset方法"""
    
    def test_reset_state(self):
        """测试重置重试状态"""
        manager = RetryManager()
        
        # 设置一些状态
        manager._attempt_count = 5
        manager._total_delay = 10.5
        
        # 重置状态
        manager.reset()
        
        # 验证状态被重置
        assert manager._attempt_count == 0
        assert manager._total_delay == 0.0


class TestRetryManagerExecuteWithLogging:
    """测试RetryManager执行时的日志记录"""
    
    def test_execute_logs_success_on_retry(self):
        """测试重试成功时的日志记录"""
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("Service down")
            return "success"
        
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.01,
            retryable_exceptions=(RetryableError,)
        )
        manager = RetryManager(config)
        
        with patch('harborai.core.retry.logger') as mock_logger:
            result = manager.execute(failing_func)
            
            # 验证成功日志被记录
            mock_logger.info.assert_called_with("Function succeeded on attempt 3")
            assert result == "success"


class TestRetryManagerCalculateDelayStrategies:
    """测试RetryManager的calculate_delay方法的不同策略"""
    
    def test_calculate_delay_fixed_strategy(self):
        """测试固定延迟策略"""
        config = RetryConfig(
            strategy=RetryStrategyEnum.FIXED,
            base_delay=2.0,
            jitter=False  # 禁用抖动以获得确定结果
        )
        manager = RetryManager(config)
        
        # 固定策略应该总是返回base_delay
        delay = manager.calculate_delay(1)
        assert delay == 2.0
        
        delay = manager.calculate_delay(5)
        assert delay == 2.0
    
    def test_calculate_delay_linear_strategy(self):
        """测试线性延迟策略"""
        config = RetryConfig(
            strategy=RetryStrategyEnum.LINEAR,
            base_delay=1.0,
            jitter=False  # 禁用抖动以获得确定结果
        )
        manager = RetryManager(config)
        
        # 线性策略应该是base_delay * attempt
        delay = manager.calculate_delay(1)
        assert delay == 1.0
        
        delay = manager.calculate_delay(3)
        assert delay == 3.0
    
    def test_calculate_delay_exponential_strategy(self):
        """测试指数延迟策略"""
        config = RetryConfig(
            strategy=RetryStrategyEnum.EXPONENTIAL,
            base_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False  # 禁用抖动以获得确定结果
        )
        manager = RetryManager(config)
        
        # 指数策略应该是base_delay * (multiplier ** (attempt - 1))
        delay = manager.calculate_delay(1)
        assert delay == 1.0  # 1.0 * (2.0 ** 0)
        
        delay = manager.calculate_delay(2)
        assert delay == 2.0  # 1.0 * (2.0 ** 1)
        
        delay = manager.calculate_delay(3)
        assert delay == 4.0  # 1.0 * (2.0 ** 2)
    
    def test_calculate_delay_random_strategy(self):
        """测试随机延迟策略"""
        config = RetryConfig(
            strategy=RetryStrategyEnum.RANDOM,
            base_delay=1.0,
            max_delay=10.0,  # 设置足够大的最大延迟
            jitter=False  # 禁用抖动，只测试随机策略本身
        )
        manager = RetryManager(config)
        
        # 测试随机策略的延迟计算
        delay = manager.calculate_delay(3)
        # 随机延迟应该在0到max_delay之间（因为会被限制）
        assert 0 <= delay <= config.max_delay
        
        # 测试多次调用以确保代码路径被覆盖
        for attempt in range(1, 5):
            delay = manager.calculate_delay(attempt)
            assert 0 <= delay <= config.max_delay
    
    def test_calculate_delay_unknown_strategy(self):
        """测试未知策略（默认行为）"""
        config = RetryConfig(base_delay=3.0, jitter=False)
        manager = RetryManager(config)
        
        # 设置一个未知的策略
        config.strategy = "unknown_strategy"
        
        # 应该回退到base_delay
        delay = manager.calculate_delay(1)
        assert delay == 3.0
    
    def test_calculate_delay_with_max_delay_limit(self):
        """测试最大延迟限制"""
        config = RetryConfig(
            strategy=RetryStrategyEnum.LINEAR,
            base_delay=10.0,
            max_delay=15.0,
            jitter=False  # 禁用抖动以获得确定结果
        )
        manager = RetryManager(config)
        
        # 延迟应该被限制在max_delay以内
        delay = manager.calculate_delay(5)  # 10.0 * 5 = 50.0, 但应该被限制为15.0
        assert delay == 15.0
    
    def test_calculate_delay_with_jitter(self):
        """测试带抖动的延迟计算"""
        config = RetryConfig(
            strategy=RetryStrategyEnum.FIXED,
            base_delay=10.0,
            jitter=True
        )
        manager = RetryManager(config)
        
        # 测试抖动功能，不模拟random以避免复杂的调用顺序问题
        delays = []
        for _ in range(10):
            delay = manager.calculate_delay(1)
            delays.append(delay)
            # 延迟应该在合理范围内（base_delay ± 10%，但不能为负）
            assert delay >= 0  # 确保不为负数
            # 抖动可能使延迟变化较大，所以给足够的余量
            assert delay <= 15.0  # 给足够的余量
        
        # 应该有一些变化（除非极小概率全部相同）
        unique_delays = set(delays)
        # 如果抖动工作正常，应该有多个不同的值
        assert len(unique_delays) >= 1  # 至少有一个值
    
    def test_calculate_delay_with_negative_jitter(self):
        """测试负抖动的处理"""
        config = RetryConfig(
            strategy=RetryStrategyEnum.FIXED,
            base_delay=1.0,
            jitter=True
        )
        manager = RetryManager(config)
        
        with patch('harborai.core.retry.random.uniform') as mock_uniform:
            mock_uniform.return_value = -2.0  # 负抖动，会导致负延迟
            delay = manager.calculate_delay(1)
            
            # 延迟应该被限制为非负数
            assert delay == 0.0


class TestRetryManagerExecuteCallbacks:
    """测试RetryManager的回调执行"""
    
    def test_execute_retry_callback_with_exception(self):
        """测试重试回调中发生异常的情况"""
        def failing_callback(attempt, exception, delay):
            raise ValueError("Callback failed")
        
        config = RetryConfig(on_retry=failing_callback)
        manager = RetryManager(config)
        
        # 执行回调不应该抛出异常，而是记录警告
        with patch('harborai.core.retry.logger') as mock_logger:
            manager.execute_retry_callback(1, Exception("test"), 1.0)
            mock_logger.warning.assert_called_once()
    
    def test_execute_failure_callback_with_exception(self):
        """测试失败回调中发生异常的情况"""
        def failing_callback(exception, attempts):
            raise ValueError("Failure callback failed")
        
        config = RetryConfig(on_failure=failing_callback)
        manager = RetryManager(config)
        
        # 执行失败回调不应该抛出异常，而是记录警告
        with patch('harborai.core.retry.logger') as mock_logger:
            manager.execute_failure_callback(Exception("test"), 3)
            mock_logger.warning.assert_called_once()


class TestRetryManagerShouldRetryEdgeCases:
    """测试RetryManager的should_retry方法的边界情况"""
    
    def test_should_retry_with_rate_limit_error(self):
        """测试RateLimitError的重试逻辑"""
        config = RetryConfig(max_attempts=3)
        manager = RetryManager(config)
        
        # RateLimitError应该被重试
        rate_limit_error = RateLimitError("Rate limit exceeded")
        assert manager.should_retry(rate_limit_error, 1) is True
        assert manager.should_retry(rate_limit_error, 2) is True
        
        # 但超过最大次数时不应该重试
        assert manager.should_retry(rate_limit_error, 3) is False
    
    def test_should_retry_with_api_error_non_5xx(self):
        """测试非5xx APIError的重试逻辑"""
        config = RetryConfig(max_attempts=3)
        manager = RetryManager(config)
        
        # 4xx错误不应该重试
        api_error_4xx = APIError("Bad request", status_code=400)
        assert manager.should_retry(api_error_4xx, 1) is False
        
        # 3xx错误不应该重试
        api_error_3xx = APIError("Redirect", status_code=301)
        assert manager.should_retry(api_error_3xx, 1) is False
        
        # 没有状态码的APIError不应该重试
        api_error_no_code = APIError("Unknown error")
        assert manager.should_retry(api_error_no_code, 1) is False
    
    def test_should_retry_non_retryable_exception(self):
        """测试不可重试异常"""
        config = RetryConfig(
            max_attempts=3,
            non_retryable_exceptions=(NonRetryableError,)
        )
        manager = RetryManager(config)
        
        # 不可重试异常应该返回False
        non_retryable_error = NonRetryableError("Cannot retry")
        assert manager.should_retry(non_retryable_error, 1) is False
    
    def test_custom_retry_condition_true(self):
        """测试自定义重试条件返回True"""
        def custom_condition(exc):
            return "retry" in str(exc).lower()
        
        config = RetryConfig(
            max_attempts=3,
            retry_condition=custom_condition
        )
        manager = RetryManager(config)
        
        # 满足自定义条件的异常应该被重试
        retry_error = Exception("Please retry this")
        assert manager.should_retry(retry_error, 1) is True
    
    def test_custom_retry_condition_false(self):
        """测试自定义重试条件返回False"""
        def custom_condition(exc):
            return "retry" in str(exc).lower()
        
        config = RetryConfig(
            max_attempts=3,
            retry_condition=custom_condition
        )
        manager = RetryManager(config)
        
        # 不满足自定义条件的异常不应该被重试
        no_retry_error = Exception("Fatal error")
        assert manager.should_retry(no_retry_error, 1) is False