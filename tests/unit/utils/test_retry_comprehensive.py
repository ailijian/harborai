#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重试模块comprehensive测试

测试 HarborAI 重试机制的所有功能，确保重试逻辑正常工作。
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, call
from typing import Any

from harborai.utils.retry import (
    should_retry_api_error,
    calculate_backoff_delay,
    RetryConfig,
    retry_with_backoff,
    async_retry_with_backoff,
    RETRYABLE_EXCEPTIONS,
)
from harborai.utils.exceptions import (
    APIError,
    RateLimitError,
    TimeoutError,
    AuthenticationError,
)


class TestShouldRetryAPIError:
    """should_retry_api_error函数测试"""
    
    def test_retryable_exceptions(self):
        """测试可重试的异常类型"""
        assert should_retry_api_error(RateLimitError()) is True
        assert should_retry_api_error(TimeoutError()) is True
        assert should_retry_api_error(ConnectionError()) is True
    
    def test_non_retryable_exceptions(self):
        """测试不可重试的异常类型"""
        assert should_retry_api_error(AuthenticationError()) is False
        assert should_retry_api_error(ValueError()) is False
        assert should_retry_api_error(TypeError()) is False
    
    def test_api_error_5xx_status_codes(self):
        """测试5xx状态码的API错误"""
        assert should_retry_api_error(APIError("Server Error", status_code=500)) is True
        assert should_retry_api_error(APIError("Bad Gateway", status_code=502)) is True
        assert should_retry_api_error(APIError("Service Unavailable", status_code=503)) is True
        assert should_retry_api_error(APIError("Gateway Timeout", status_code=504)) is True
        assert should_retry_api_error(APIError("Internal Error", status_code=599)) is True
    
    def test_api_error_429_status_code(self):
        """测试429状态码（速率限制）"""
        assert should_retry_api_error(APIError("Rate Limited", status_code=429)) is True
    
    def test_api_error_408_status_code(self):
        """测试408状态码（请求超时）"""
        assert should_retry_api_error(APIError("Request Timeout", status_code=408)) is True
    
    def test_api_error_4xx_non_retryable(self):
        """测试不可重试的4xx状态码"""
        assert should_retry_api_error(APIError("Bad Request", status_code=400)) is False
        assert should_retry_api_error(APIError("Unauthorized", status_code=401)) is False
        assert should_retry_api_error(APIError("Forbidden", status_code=403)) is False
        assert should_retry_api_error(APIError("Not Found", status_code=404)) is False
    
    def test_api_error_without_status_code(self):
        """测试没有状态码的API错误"""
        assert should_retry_api_error(APIError("Generic API Error")) is False
    
    def test_api_error_none_status_code(self):
        """测试状态码为None的API错误"""
        error = APIError("Error")
        error.status_code = None
        assert should_retry_api_error(error) is False


class TestCalculateBackoffDelay:
    """calculate_backoff_delay函数测试"""
    
    def test_basic_exponential_backoff(self):
        """测试基础指数退避"""
        delay1 = calculate_backoff_delay(1, base_delay=1.0, exponential_base=2.0, jitter=False)
        delay2 = calculate_backoff_delay(2, base_delay=1.0, exponential_base=2.0, jitter=False)
        delay3 = calculate_backoff_delay(3, base_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert delay1 == 1.0  # 1.0 * 2^0
        assert delay2 == 2.0  # 1.0 * 2^1
        assert delay3 == 4.0  # 1.0 * 2^2
    
    def test_max_delay_limit(self):
        """测试最大延迟限制"""
        delay = calculate_backoff_delay(10, base_delay=1.0, max_delay=5.0, jitter=False)
        assert delay == 5.0
    
    def test_custom_base_delay(self):
        """测试自定义基础延迟"""
        delay1 = calculate_backoff_delay(1, base_delay=2.0, exponential_base=2.0, jitter=False)
        delay2 = calculate_backoff_delay(2, base_delay=2.0, exponential_base=2.0, jitter=False)
        
        assert delay1 == 2.0  # 2.0 * 2^0
        assert delay2 == 4.0  # 2.0 * 2^1
    
    def test_custom_exponential_base(self):
        """测试自定义指数基数"""
        delay1 = calculate_backoff_delay(1, base_delay=1.0, exponential_base=3.0, jitter=False)
        delay2 = calculate_backoff_delay(2, base_delay=1.0, exponential_base=3.0, jitter=False)
        
        assert delay1 == 1.0  # 1.0 * 3^0
        assert delay2 == 3.0  # 1.0 * 3^1
    
    def test_jitter_enabled(self):
        """测试启用抖动"""
        # 运行多次以确保抖动生效
        delays = []
        for _ in range(10):
            delay = calculate_backoff_delay(2, base_delay=1.0, exponential_base=2.0, jitter=True)
            delays.append(delay)
        
        # 所有延迟应该在[1.0, 2.0]范围内（2.0 * [0.5, 1.0]）
        for delay in delays:
            assert 1.0 <= delay <= 2.0
        
        # 应该有一些变化（不是所有值都相同）
        assert len(set(delays)) > 1
    
    def test_jitter_disabled(self):
        """测试禁用抖动"""
        delay1 = calculate_backoff_delay(2, base_delay=1.0, exponential_base=2.0, jitter=False)
        delay2 = calculate_backoff_delay(2, base_delay=1.0, exponential_base=2.0, jitter=False)
        
        assert delay1 == delay2 == 2.0


class TestRetryConfig:
    """RetryConfig类测试"""
    
    def test_default_initialization(self):
        """测试默认初始化"""
        config = RetryConfig()
        
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retryable_exceptions == RETRYABLE_EXCEPTIONS
        assert config.retry_condition is should_retry_api_error
    
    def test_custom_initialization(self):
        """测试自定义初始化"""
        custom_exceptions = (ValueError, TypeError)
        custom_condition = lambda e: isinstance(e, ValueError)
        
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False,
            retryable_exceptions=custom_exceptions,
            retry_condition=custom_condition
        )
        
        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retryable_exceptions == custom_exceptions
        assert config.retry_condition is custom_condition


class TestRetryWithBackoff:
    """retry_with_backoff装饰器测试"""
    
    def test_successful_function_no_retry(self):
        """测试成功函数不需要重试"""
        @retry_with_backoff()
        def successful_func():
            return "success"
        
        result = successful_func()
        assert result == "success"
    
    def test_function_with_retryable_exception(self):
        """测试可重试异常的函数"""
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_attempts=3, base_delay=0.01))
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited")
            return "success"
        
        with patch('time.sleep'):  # 避免实际等待
            result = failing_func()
        
        assert result == "success"
        assert call_count == 3
    
    def test_function_with_non_retryable_exception(self):
        """测试不可重试异常的函数"""
        call_count = 0
        
        @retry_with_backoff()
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Auth failed")
        
        with pytest.raises(AuthenticationError):
            failing_func()
        
        assert call_count == 1  # 只调用一次，不重试
    
    def test_exhausted_retries(self):
        """测试重试次数耗尽"""
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_attempts=2, base_delay=0.01))
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Always timeout")
        
        with patch('time.sleep'):  # 避免实际等待
            with pytest.raises(TimeoutError):
                always_failing_func()
        
        assert call_count == 2
    
    def test_custom_config(self):
        """测试自定义配置"""
        call_count = 0
        
        config = RetryConfig(
            max_attempts=4,
            base_delay=0.01,
            retry_condition=lambda e: isinstance(e, ValueError)
        )
        
        @retry_with_backoff(config)
        def custom_failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Custom error")
            return "success"
        
        with patch('time.sleep'):
            result = custom_failing_func()
        
        assert result == "success"
        assert call_count == 3
    
    def test_trace_id_logging(self):
        """测试trace_id日志记录"""
        @retry_with_backoff(RetryConfig(max_attempts=2, base_delay=0.01), trace_id="test_trace")
        def failing_func():
            raise RateLimitError("Rate limited")
        
        with patch('time.sleep'):
            with patch('harborai.utils.retry.logger') as mock_logger:
                with pytest.raises(RateLimitError):
                    failing_func()
                
                # 验证日志调用包含trace_id
                mock_logger.warning.assert_called()
                mock_logger.error.assert_called()
                
                # 检查日志调用的extra参数
                warning_call = mock_logger.warning.call_args
                error_call = mock_logger.error.call_args
                
                assert warning_call[1]['extra']['trace_id'] == "test_trace"
                assert error_call[1]['extra']['trace_id'] == "test_trace"
    
    def test_delay_calculation(self):
        """测试延迟计算"""
        call_count = 0
        delays = []
        
        @retry_with_backoff(RetryConfig(max_attempts=3, base_delay=1.0, jitter=False))
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Timeout")
            return "success"
        
        def mock_sleep(delay):
            delays.append(delay)
        
        with patch('time.sleep', side_effect=mock_sleep):
            result = failing_func()
        
        assert result == "success"
        assert len(delays) == 2  # 两次重试之间的延迟
        assert delays[0] == 1.0  # 第一次重试延迟
        assert delays[1] == 2.0  # 第二次重试延迟


class TestAsyncRetryWithBackoff:
    """async_retry_with_backoff装饰器测试"""
    
    @pytest.mark.asyncio
    async def test_successful_async_function_no_retry(self):
        """测试成功的异步函数不需要重试"""
        @async_retry_with_backoff()
        async def successful_async_func():
            return "async_success"
        
        result = await successful_async_func()
        assert result == "async_success"
    
    @pytest.mark.asyncio
    async def test_async_function_with_retryable_exception(self):
        """测试可重试异常的异步函数"""
        call_count = 0
        
        @async_retry_with_backoff(RetryConfig(max_attempts=3, base_delay=0.01))
        async def failing_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RateLimitError("Rate limited")
            return "async_success"
        
        with patch('asyncio.sleep'):  # 避免实际等待
            result = await failing_async_func()
        
        assert result == "async_success"
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_async_function_with_non_retryable_exception(self):
        """测试不可重试异常的异步函数"""
        call_count = 0
        
        @async_retry_with_backoff()
        async def failing_async_func():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Auth failed")
        
        with pytest.raises(AuthenticationError):
            await failing_async_func()
        
        assert call_count == 1  # 只调用一次，不重试
    
    @pytest.mark.asyncio
    async def test_async_exhausted_retries(self):
        """测试异步重试次数耗尽"""
        call_count = 0
        
        @async_retry_with_backoff(RetryConfig(max_attempts=2, base_delay=0.01))
        async def always_failing_async_func():
            nonlocal call_count
            call_count += 1
            raise TimeoutError("Always timeout")
        
        with patch('asyncio.sleep'):  # 避免实际等待
            with pytest.raises(TimeoutError):
                await always_failing_async_func()
        
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_async_delay_calculation(self):
        """测试异步延迟计算"""
        call_count = 0
        delays = []
        
        @async_retry_with_backoff(RetryConfig(max_attempts=3, base_delay=1.0, jitter=False))
        async def failing_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("Timeout")
            return "async_success"
        
        async def mock_async_sleep(delay):
            delays.append(delay)
        
        with patch('asyncio.sleep', side_effect=mock_async_sleep):
            result = await failing_async_func()
        
        assert result == "async_success"
        assert len(delays) == 2  # 两次重试之间的延迟
        assert delays[0] == 1.0  # 第一次重试延迟
        assert delays[1] == 2.0  # 第二次重试延迟
    
    @pytest.mark.asyncio
    async def test_async_trace_id_logging(self):
        """测试异步trace_id日志记录"""
        @async_retry_with_backoff(RetryConfig(max_attempts=2, base_delay=0.01), trace_id="async_trace")
        async def failing_async_func():
            raise RateLimitError("Rate limited")
        
        with patch('asyncio.sleep'):
            with patch('harborai.utils.retry.logger') as mock_logger:
                with pytest.raises(RateLimitError):
                    await failing_async_func()
                
                # 验证日志调用包含trace_id
                mock_logger.warning.assert_called()
                mock_logger.error.assert_called()
                
                # 检查日志调用的extra参数
                warning_call = mock_logger.warning.call_args
                error_call = mock_logger.error.call_args
                
                assert warning_call[1]['extra']['trace_id'] == "async_trace"
                assert error_call[1]['extra']['trace_id'] == "async_trace"


class TestIntegration:
    """集成测试"""
    
    def test_retry_with_api_error_integration(self):
        """测试重试与API错误的集成"""
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_attempts=4, base_delay=0.01))
        def api_call_simulation():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise APIError("Server Error", status_code=500)
            elif call_count == 2:
                raise APIError("Rate Limited", status_code=429)
            elif call_count == 3:
                raise APIError("Gateway Timeout", status_code=504)
            else:
                return {"status": "success", "data": "result"}
        
        with patch('time.sleep'):
            result = api_call_simulation()
        
        assert result == {"status": "success", "data": "result"}
        assert call_count == 4
    
    def test_retry_with_mixed_exceptions(self):
        """测试混合异常类型的重试"""
        call_count = 0
        
        @retry_with_backoff(RetryConfig(max_attempts=3, base_delay=0.01))
        def mixed_exception_func():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise RateLimitError("Rate limited")
            elif call_count == 2:
                raise AuthenticationError("Auth failed")  # 不可重试
            else:
                return "success"
        
        with patch('time.sleep'):
            with pytest.raises(AuthenticationError):
                mixed_exception_func()
        
        assert call_count == 2  # 第二次异常不可重试，停止
    
    @pytest.mark.asyncio
    async def test_async_retry_integration(self):
        """测试异步重试集成"""
        call_count = 0
        
        @async_retry_with_backoff(RetryConfig(max_attempts=3, base_delay=0.01))
        async def async_api_call():
            nonlocal call_count
            call_count += 1
            
            if call_count < 3:
                raise TimeoutError("Connection timeout")
            
            return {"async": True, "result": "success"}
        
        with patch('asyncio.sleep'):
            result = await async_api_call()
        
        assert result == {"async": True, "result": "success"}
        assert call_count == 3
    
    def test_custom_retry_condition_integration(self):
        """测试自定义重试条件集成"""
        call_count = 0
        
        def custom_retry_condition(exception):
            # 只重试包含"retry"的异常消息
            return "retry" in str(exception).lower()
        
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retry_condition=custom_retry_condition
        )
        
        @retry_with_backoff(config)
        def custom_condition_func():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise ValueError("Please retry this operation")
            elif call_count == 2:
                raise ValueError("Do not retry this")  # 不包含"retry"
            else:
                return "success"
        
        with patch('time.sleep'):
            with pytest.raises(ValueError, match="Do not retry this"):
                custom_condition_func()
        
        assert call_count == 2  # 第二次异常不满足重试条件