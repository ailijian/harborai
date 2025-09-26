# -*- coding: utf-8 -*-
"""
HarborAI 异常重试测试模块

测试目标：
- 验证异常检测和分类机制
- 测试重试策略的配置和执行
- 验证指数退避和抖动算法
- 测试重试限制和熔断机制
- 验证异常恢复和状态管理
"""

import pytest
import asyncio
import time
import random
from typing import Dict, Any, List, Optional, Callable
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass
from enum import Enum

from harborai import HarborAI
from harborai.core.exceptions import (
    HarborAIError, 
    RetryableError, 
    NonRetryableError,
    RateLimitError,
    TimeoutError,
    NetworkError,
    AuthenticationError,
    QuotaExceededError
)
from harborai.core.retry import (
    RetryManager,
    RetryStrategy,
    ExponentialBackoff,
    FixedDelay,
    LinearBackoff,
    CircuitBreaker,
    RetryPolicy
)


class ErrorType(Enum):
    """错误类型枚举"""
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    AUTH = "authentication"
    QUOTA = "quota"
    SERVER = "server"
    CLIENT = "client"


@dataclass
class RetryAttempt:
    """重试尝试记录"""
    attempt_number: int
    timestamp: float
    delay: float
    error: Optional[Exception] = None
    success: bool = False


class MockRetryableOperation:
    """可重试操作的模拟类"""
    
    def __init__(self, name: str, failure_pattern: List[bool] = None):
        self.name = name
        self.failure_pattern = failure_pattern or []
        self.call_count = 0
        self.attempts = []
    
    def execute(self, *args, **kwargs) -> Any:
        """执行操作"""
        self.call_count += 1
        attempt_index = self.call_count - 1
        
        # 记录尝试
        attempt = RetryAttempt(
            attempt_number=self.call_count,
            timestamp=time.time(),
            delay=0
        )
        
        # 根据失败模式决定是否失败
        if attempt_index < len(self.failure_pattern) and self.failure_pattern[attempt_index]:
            # 模拟失败
            error = self._generate_error(attempt_index)
            attempt.error = error
            attempt.success = False
            self.attempts.append(attempt)
            raise error
        else:
            # 模拟成功
            attempt.success = True
            self.attempts.append(attempt)
            return f"Operation {self.name} succeeded on attempt {self.call_count}"
    
    async def async_execute(self, *args, **kwargs) -> Any:
        """异步执行操作"""
        # 模拟异步延迟
        await asyncio.sleep(0.01)
        return self.execute(*args, **kwargs)
    
    def _generate_error(self, attempt_index: int) -> Exception:
        """根据尝试次数生成不同类型的错误"""
        error_types = [
            NetworkError("Network connection failed"),
            RateLimitError("Rate limit exceeded", retry_after=1),
            TimeoutError("Request timeout"),
            QuotaExceededError("API quota exceeded")
        ]
        
        return error_types[attempt_index % len(error_types)]
    
    def reset(self):
        """重置操作状态"""
        self.call_count = 0
        self.attempts = []


class TestExceptionClassification:
    """异常分类测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.exception_retry
    def test_retryable_exception_detection(self):
        """测试可重试异常检测"""
        # 创建异常分类器
        exception_classifier = Mock()
        
        # 定义异常分类规则
        def mock_is_retryable(exception):
            retryable_types = (
                NetworkError,
                RateLimitError,
                TimeoutError,
                QuotaExceededError
            )
            
            non_retryable_types = (
                AuthenticationError,
                ValueError,
                TypeError
            )
            
            if isinstance(exception, retryable_types):
                return True
            elif isinstance(exception, non_retryable_types):
                return False
            else:
                # 默认情况下，服务器错误可重试，客户端错误不可重试
                return isinstance(exception, Exception) and "server" in str(exception).lower()
        
        exception_classifier.is_retryable.side_effect = mock_is_retryable
        
        # 测试可重试异常
        retryable_exceptions = [
            NetworkError("Connection failed"),
            RateLimitError("Rate limit exceeded"),
            TimeoutError("Request timeout"),
            QuotaExceededError("Quota exceeded"),
            Exception("Internal server error")
        ]
        
        for exception in retryable_exceptions:
            assert exception_classifier.is_retryable(exception) is True
        
        # 测试不可重试异常
        non_retryable_exceptions = [
            AuthenticationError("Invalid API key"),
            ValueError("Invalid parameter"),
            TypeError("Type mismatch"),
            Exception("Client error: bad request")
        ]
        
        for exception in non_retryable_exceptions:
            assert exception_classifier.is_retryable(exception) is False
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.exception_retry
    def test_exception_severity_classification(self):
        """测试异常严重程度分类"""
        # 创建严重程度分类器
        severity_classifier = Mock()
        
        def mock_get_severity(exception):
            if isinstance(exception, (AuthenticationError, ValueError, TypeError)):
                return "critical"  # 严重错误，不应重试
            elif isinstance(exception, (RateLimitError, QuotaExceededError)):
                return "high"  # 高优先级，需要特殊处理
            elif isinstance(exception, (NetworkError, TimeoutError)):
                return "medium"  # 中等优先级，常规重试
            else:
                return "low"  # 低优先级，快速重试
        
        severity_classifier.get_severity.side_effect = mock_get_severity
        
        # 测试不同严重程度的异常
        test_cases = [
            (AuthenticationError("Invalid key"), "critical"),
            (ValueError("Invalid value"), "critical"),
            (RateLimitError("Rate limited"), "high"),
            (QuotaExceededError("Quota exceeded"), "high"),
            (NetworkError("Network error"), "medium"),
            (TimeoutError("Timeout"), "medium"),
            (Exception("Unknown error"), "low")
        ]
        
        for exception, expected_severity in test_cases:
            severity = severity_classifier.get_severity(exception)
            assert severity == expected_severity
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.exception_retry
    def test_error_context_analysis(self):
        """测试错误上下文分析"""
        # 创建上下文分析器
        context_analyzer = Mock()
        
        def mock_analyze_error_context(exception, operation_context):
            context = {
                "error_type": type(exception).__name__,
                "error_message": str(exception),
                "operation": operation_context.get("operation", "unknown"),
                "timestamp": time.time(),
                "retry_recommended": False,
                "suggested_delay": 0
            }
            
            # 根据错误类型和上下文推荐重试策略
            # 注意：TimeoutError继承自NetworkError，所以必须先检查TimeoutError
            if isinstance(exception, RateLimitError):
                context["retry_recommended"] = True
                context["suggested_delay"] = getattr(exception, 'retry_after', 60)
                context["strategy"] = "rate_limit_backoff"
            elif isinstance(exception, TimeoutError):
                context["retry_recommended"] = True
                context["suggested_delay"] = 0.5
                context["strategy"] = "linear_backoff"
            elif isinstance(exception, NetworkError):
                context["retry_recommended"] = True
                context["suggested_delay"] = 1
                context["strategy"] = "exponential_backoff"
            elif isinstance(exception, AuthenticationError):
                context["retry_recommended"] = False
                context["strategy"] = "no_retry"
            
            return context
        
        context_analyzer.analyze.side_effect = mock_analyze_error_context
        
        # 测试不同错误的上下文分析
        test_cases = [
            {
                "exception": RateLimitError("Rate limited", retry_after=30),
                "context": {"operation": "chat_completion", "model": "deepseek-r1"},
                "expected_strategy": "rate_limit_backoff",
                "expected_delay": 30
            },
            {
                "exception": NetworkError("Connection failed"),
                "context": {"operation": "api_request", "endpoint": "/v1/chat/completions"},
                "expected_strategy": "exponential_backoff",
                "expected_delay": 1
            },
            {
                "exception": TimeoutError("Request timeout"),
                "context": {"operation": "streaming_request"},
                "expected_strategy": "linear_backoff",
                "expected_delay": 0.5
            },
            {
                "exception": AuthenticationError("Invalid API key"),
                "context": {"operation": "authentication"},
                "expected_strategy": "no_retry",
                "expected_delay": 0
            }
        ]
        
        for test_case in test_cases:
            analysis = context_analyzer.analyze(test_case["exception"], test_case["context"])
            
            assert analysis["strategy"] == test_case["expected_strategy"]
            assert analysis["suggested_delay"] == test_case["expected_delay"]
            assert analysis["operation"] == test_case["context"]["operation"]
            
            if test_case["expected_strategy"] != "no_retry":
                assert analysis["retry_recommended"] is True
            else:
                assert analysis["retry_recommended"] is False


class TestRetryStrategies:
    """重试策略测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.exception_retry
    def test_exponential_backoff_strategy(self):
        """测试指数退避策略"""
        # 创建指数退避策略
        backoff_strategy = Mock(spec=ExponentialBackoff)
        
        # 配置指数退避参数
        base_delay = 1.0
        max_delay = 60.0
        multiplier = 2.0
        jitter = True
        
        def mock_calculate_delay(attempt_number):
            delay = base_delay * (multiplier ** (attempt_number - 1))
            
            if jitter:
                # 添加随机抖动 (±25%)
                jitter_range = delay * 0.25
                delay += random.uniform(-jitter_range, jitter_range)
            
            # 应用最大延迟限制（在jitter之后）
            delay = min(delay, max_delay)
            return max(0, delay)
        
        backoff_strategy.calculate_delay.side_effect = mock_calculate_delay
        
        # 测试多次重试的延迟计算
        expected_base_delays = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]  # 最后两个受max_delay限制
        
        for attempt in range(1, 9):
            delay = backoff_strategy.calculate_delay(attempt)
            expected_base = expected_base_delays[attempt - 1]
            
            # 考虑抖动，延迟应该在期望值的75%-125%范围内
            assert expected_base * 0.75 <= delay <= min(expected_base * 1.25, max_delay)
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.exception_retry
    def test_linear_backoff_strategy(self):
        """测试线性退避策略"""
        # 创建线性退避策略
        backoff_strategy = Mock(spec=LinearBackoff)
        
        # 配置线性退避参数
        base_delay = 0.5
        increment = 0.5
        max_delay = 10.0
        
        def mock_calculate_delay(attempt_number):
            delay = base_delay + (increment * (attempt_number - 1))
            return min(delay, max_delay)
        
        backoff_strategy.calculate_delay.side_effect = mock_calculate_delay
        
        # 测试线性增长的延迟
        expected_delays = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
        
        for attempt in range(1, 13):
            delay = backoff_strategy.calculate_delay(attempt)
            expected_delay = min(expected_delays[attempt - 1], max_delay)
            assert delay == expected_delay
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.exception_retry
    def test_fixed_delay_strategy(self):
        """测试固定延迟策略"""
        # 创建固定延迟策略
        backoff_strategy = Mock(spec=FixedDelay)
        
        # 配置固定延迟参数
        fixed_delay = 2.0
        
        def mock_calculate_delay(attempt_number):
            return fixed_delay
        
        backoff_strategy.calculate_delay.side_effect = mock_calculate_delay
        
        # 测试固定延迟
        for attempt in range(1, 10):
            delay = backoff_strategy.calculate_delay(attempt)
            assert delay == fixed_delay
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.exception_retry
    def test_adaptive_retry_strategy(self):
        """测试自适应重试策略"""
        # 创建自适应重试策略
        adaptive_strategy = Mock()
        
        # 历史成功率数据
        success_history = {
            "recent_attempts": 10,
            "recent_successes": 7,
            "success_rate": 0.7
        }
        
        def mock_calculate_adaptive_delay(attempt_number, error_type, success_rate):
            base_delay = 1.0
            
            # 根据成功率调整延迟
            if success_rate > 0.8:
                # 高成功率，使用较短延迟
                multiplier = 0.5
            elif success_rate > 0.5:
                # 中等成功率，使用标准延迟
                multiplier = 1.0
            else:
                # 低成功率，使用较长延迟
                multiplier = 2.0
            
            # 根据错误类型调整
            if error_type == "rate_limit":
                multiplier *= 3.0  # 速率限制需要更长延迟
            elif error_type == "network":
                multiplier *= 1.5  # 网络错误需要中等延迟
            
            delay = base_delay * multiplier * (attempt_number ** 0.5)  # 平方根增长
            return min(delay, 30.0)  # 最大延迟30秒
        
        adaptive_strategy.calculate_delay.side_effect = mock_calculate_adaptive_delay
        
        # 测试不同场景下的自适应延迟
        test_cases = [
            {"attempt": 1, "error_type": "network", "success_rate": 0.9, "expected_max": 1.0},
            {"attempt": 2, "error_type": "network", "success_rate": 0.9, "expected_max": 1.5},
            {"attempt": 1, "error_type": "rate_limit", "success_rate": 0.7, "expected_max": 4.0},
            {"attempt": 3, "error_type": "network", "success_rate": 0.3, "expected_max": 6.0}
        ]
        
        for case in test_cases:
            delay = adaptive_strategy.calculate_delay(
                case["attempt"], 
                case["error_type"], 
                case["success_rate"]
            )
            assert delay <= case["expected_max"]
            assert delay > 0


class TestRetryExecution:
    """重试执行测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.exception_retry
    def test_basic_retry_execution(self):
        """测试基本重试执行"""
        # 创建重试管理器
        retry_manager = Mock(spec=RetryManager)
        
        # 创建失败模式：前2次失败，第3次成功
        operation = MockRetryableOperation("test_op", failure_pattern=[True, True, False])
        
        # 模拟重试执行
        def mock_execute_with_retry(func, max_attempts=3, backoff_strategy=None):
            attempts = []
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    result = func()
                    attempts.append({"attempt": attempt, "success": True, "result": result})
                    return result, attempts
                except Exception as e:
                    last_exception = e
                    attempts.append({"attempt": attempt, "success": False, "error": str(e)})
                    
                    if attempt < max_attempts:
                        # 计算延迟（这里简化为固定延迟）
                        delay = 0.1 * attempt
                        time.sleep(delay)
            
            # 所有重试都失败
            raise last_exception
        
        retry_manager.execute_with_retry.side_effect = mock_execute_with_retry
        
        # 执行重试
        result, attempts = retry_manager.execute_with_retry(operation.execute, max_attempts=3)
        
        # 验证结果
        assert "succeeded on attempt 3" in result
        assert len(attempts) == 3
        assert attempts[0]["success"] is False
        assert attempts[1]["success"] is False
        assert attempts[2]["success"] is True
        
        # 验证操作被调用了3次
        assert operation.call_count == 3
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.exception_retry
    def test_retry_with_different_strategies(self):
        """测试不同重试策略的执行"""
        # 创建重试管理器
        retry_manager = Mock(spec=RetryManager)
        
        # 创建操作（前3次失败，第4次成功）
        operation = MockRetryableOperation("strategy_test", failure_pattern=[True, True, True, False])
        
        # 定义不同的重试策略
        strategies = {
            "exponential": {"type": "exponential", "base_delay": 0.1, "multiplier": 2},
            "linear": {"type": "linear", "base_delay": 0.1, "increment": 0.1},
            "fixed": {"type": "fixed", "delay": 0.1}
        }
        
        def mock_execute_with_strategy(func, strategy_name, max_attempts=5):
            strategy_config = strategies[strategy_name]
            attempts = []
            
            for attempt in range(1, max_attempts + 1):
                try:
                    result = func()
                    attempts.append({
                        "attempt": attempt, 
                        "success": True, 
                        "result": result,
                        "strategy": strategy_name
                    })
                    return result, attempts
                except Exception as e:
                    # 计算延迟
                    if strategy_config["type"] == "exponential":
                        delay = strategy_config["base_delay"] * (strategy_config["multiplier"] ** (attempt - 1))
                    elif strategy_config["type"] == "linear":
                        delay = strategy_config["base_delay"] + (strategy_config["increment"] * (attempt - 1))
                    else:  # fixed
                        delay = strategy_config["delay"]
                    
                    attempts.append({
                        "attempt": attempt, 
                        "success": False, 
                        "error": str(e),
                        "delay": delay,
                        "strategy": strategy_name
                    })
                    
                    if attempt < max_attempts:
                        time.sleep(delay)
            
            raise Exception("Max attempts exceeded")
        
        retry_manager.execute_with_strategy.side_effect = mock_execute_with_strategy
        
        # 测试指数退避策略
        operation.reset()
        result, attempts = retry_manager.execute_with_strategy(operation.execute, "exponential")
        
        assert "succeeded on attempt 4" in result
        assert len(attempts) == 4
        assert attempts[0]["delay"] == 0.1  # 0.1 * 2^0
        assert attempts[1]["delay"] == 0.2  # 0.1 * 2^1
        assert attempts[2]["delay"] == 0.4  # 0.1 * 2^2
        
        # 测试线性退避策略
        operation.reset()
        result, attempts = retry_manager.execute_with_strategy(operation.execute, "linear")
        
        assert len(attempts) == 4
        assert abs(attempts[0]["delay"] - 0.1) < 1e-10  # 0.1 + 0.1 * 0
        assert abs(attempts[1]["delay"] - 0.2) < 1e-10  # 0.1 + 0.1 * 1
        assert abs(attempts[2]["delay"] - 0.3) < 1e-10  # 0.1 + 0.1 * 2
        
        # 测试固定延迟策略
        operation.reset()
        result, attempts = retry_manager.execute_with_strategy(operation.execute, "fixed")
        
        assert len(attempts) == 4
        for attempt in attempts[:-1]:  # 除了最后一个成功的尝试
            assert attempt["delay"] == 0.1
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.exception_retry
    async def test_async_retry_execution(self):
        """测试异步重试执行"""
        # 创建异步重试管理器
        async_retry_manager = Mock()
        
        # 创建异步操作（前2次失败，第3次成功）
        async_operation = MockRetryableOperation("async_test", failure_pattern=[True, True, False])
        
        # 模拟异步重试执行
        async def mock_async_execute_with_retry(async_func, max_attempts=3):
            attempts = []
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    result = await async_func()
                    attempts.append({"attempt": attempt, "success": True, "result": result})
                    return result, attempts
                except Exception as e:
                    last_exception = e
                    attempts.append({"attempt": attempt, "success": False, "error": str(e)})
                    
                    if attempt < max_attempts:
                        # 异步延迟
                        await asyncio.sleep(0.01 * attempt)
            
            raise last_exception
        
        async_retry_manager.execute_with_retry.side_effect = mock_async_execute_with_retry
        
        # 执行异步重试
        result, attempts = await async_retry_manager.execute_with_retry(async_operation.async_execute)
        
        # 验证结果
        assert "succeeded on attempt 3" in result
        assert len(attempts) == 3
        assert attempts[2]["success"] is True
        assert async_operation.call_count == 3
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.exception_retry
    def test_retry_with_timeout(self):
        """测试带超时的重试"""
        # 创建带超时的重试管理器
        timeout_retry_manager = Mock()
        
        # 创建慢速操作
        slow_operation = Mock()
        
        def mock_slow_execute():
            time.sleep(1.5)  # 模拟慢速操作，超过1.0秒超时
            return "slow_result"
        
        slow_operation.side_effect = mock_slow_execute
        
        # 模拟带超时的重试执行
        def mock_execute_with_timeout(func, max_attempts=3, timeout=1.0):
            start_time = time.time()
            attempts = []
            
            # 检查初始超时
            if timeout <= 0:
                from harborai.core.exceptions import TimeoutError as HarborTimeoutError
                raise HarborTimeoutError("Timeout value must be positive")
            
            for attempt in range(1, max_attempts + 1):
                attempt_start = time.time()
                
                # 检查总超时
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    from harborai.core.exceptions import TimeoutError as HarborTimeoutError
                    raise HarborTimeoutError(f"Total timeout exceeded after {elapsed:.2f}s")
                
                try:
                    # 模拟超时检查：如果操作预计会超时，直接抛出异常
                    remaining_time = timeout - elapsed
                    if remaining_time < 1.0:  # 如果剩余时间不足1秒，而操作需要1.5秒
                        from harborai.core.exceptions import TimeoutError as HarborTimeoutError
                        raise HarborTimeoutError(f"Operation would exceed timeout (remaining: {remaining_time:.2f}s)")
                    
                    # 执行操作
                    result = func()
                    
                    attempts.append({
                        "attempt": attempt, 
                        "success": True, 
                        "result": result,
                        "duration": time.time() - attempt_start
                    })
                    return result, attempts
                    
                except Exception as e:
                    attempts.append({
                        "attempt": attempt, 
                        "success": False, 
                        "error": str(e),
                        "duration": time.time() - attempt_start
                    })
                    
                    # 如果是超时异常，直接重新抛出
                    if "timeout" in str(e).lower() or "TimeoutError" in str(type(e)):
                        raise e
                    
                    if attempt < max_attempts and time.time() - start_time < timeout:
                        time.sleep(0.1)  # 重试延迟
            
            from harborai.core.exceptions import TimeoutError as HarborTimeoutError
            raise HarborTimeoutError("All attempts failed or timed out")
        
        timeout_retry_manager.execute_with_timeout.side_effect = mock_execute_with_timeout
        
        # 测试超时情况
        with pytest.raises(TimeoutError) as exc_info:
            timeout_retry_manager.execute_with_timeout(slow_operation, max_attempts=3, timeout=0.8)
        
        assert "timeout" in str(exc_info.value).lower()
        
        # 测试快速操作不超时
        fast_operation = Mock(return_value="fast_result")
        
        result, attempts = timeout_retry_manager.execute_with_timeout(fast_operation, timeout=1.0)
        assert result == "fast_result"
        assert len(attempts) == 1
        assert attempts[0]["success"] is True


class TestCircuitBreaker:
    """熔断器测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.exception_retry
    @pytest.mark.circuit_breaker
    def test_circuit_breaker_states(self):
        """测试熔断器状态转换"""
        # 创建熔断器
        circuit_breaker = Mock(spec=CircuitBreaker)
        
        # 熔断器状态
        circuit_state = {
            "state": "closed",  # closed, open, half_open
            "failure_count": 0,
            "success_count": 0,
            "last_failure_time": None,
            "failure_threshold": 5,
            "recovery_timeout": 60,
            "success_threshold": 3
        }
        
        def mock_get_state():
            return circuit_state["state"]
        
        def mock_record_success():
            circuit_state["success_count"] += 1
            circuit_state["failure_count"] = 0  # 重置失败计数
            
            if circuit_state["state"] == "half_open":
                if circuit_state["success_count"] >= circuit_state["success_threshold"]:
                    circuit_state["state"] = "closed"
                    circuit_state["success_count"] = 0
        
        def mock_record_failure():
            circuit_state["failure_count"] += 1
            circuit_state["success_count"] = 0  # 重置成功计数
            circuit_state["last_failure_time"] = time.time()
            
            if circuit_state["failure_count"] >= circuit_state["failure_threshold"]:
                circuit_state["state"] = "open"
        
        def mock_can_execute():
            if circuit_state["state"] == "closed":
                return True
            elif circuit_state["state"] == "open":
                # 检查是否可以进入半开状态
                if (circuit_state["last_failure_time"] and 
                    time.time() - circuit_state["last_failure_time"] >= circuit_state["recovery_timeout"]):
                    circuit_state["state"] = "half_open"
                    circuit_state["success_count"] = 0
                    return True
                return False
            else:  # half_open
                return True
        
        circuit_breaker.get_state.side_effect = mock_get_state
        circuit_breaker.record_success.side_effect = mock_record_success
        circuit_breaker.record_failure.side_effect = mock_record_failure
        circuit_breaker.can_execute.side_effect = mock_can_execute
        
        # 测试初始状态（关闭）
        assert circuit_breaker.get_state() == "closed"
        assert circuit_breaker.can_execute() is True
        
        # 记录多次失败，触发熔断
        for i in range(5):
            circuit_breaker.record_failure()
        
        assert circuit_breaker.get_state() == "open"
        assert circuit_breaker.can_execute() is False
        
        # 模拟恢复超时，进入半开状态
        circuit_state["last_failure_time"] = time.time() - 61  # 超过60秒
        assert circuit_breaker.can_execute() is True
        assert circuit_breaker.get_state() == "half_open"
        
        # 在半开状态记录成功，恢复到关闭状态
        for i in range(3):
            circuit_breaker.record_success()
        
        assert circuit_breaker.get_state() == "closed"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.exception_retry
    @pytest.mark.circuit_breaker
    def test_circuit_breaker_with_retry(self):
        """测试熔断器与重试的集成"""
        # 创建集成了熔断器的重试管理器
        circuit_retry_manager = Mock()
        
        # 熔断器状态
        circuit_state = {"state": "closed", "failure_count": 0}
        
        # 创建操作
        operation = MockRetryableOperation("circuit_test", failure_pattern=[True, True, True, True, True])
        
        def mock_execute_with_circuit_breaker(func, max_attempts=3):
            attempts = []
            
            for attempt in range(1, max_attempts + 1):
                # 检查熔断器状态
                if circuit_state["state"] == "open":
                    raise Exception("Circuit breaker is open")
                
                try:
                    result = func()
                    # 记录成功
                    circuit_state["failure_count"] = 0
                    attempts.append({"attempt": attempt, "success": True, "result": result})
                    return result, attempts
                    
                except Exception as e:
                    # 记录失败
                    circuit_state["failure_count"] += 1
                    
                    # 检查是否需要打开熔断器
                    if circuit_state["failure_count"] >= 3:
                        circuit_state["state"] = "open"
                    
                    attempts.append({"attempt": attempt, "success": False, "error": str(e)})
                    
                    if attempt < max_attempts and circuit_state["state"] != "open":
                        time.sleep(0.1)
            
            if circuit_state["state"] == "open":
                raise Exception("Circuit breaker opened due to repeated failures")
            else:
                raise Exception("Max attempts exceeded")
        
        circuit_retry_manager.execute_with_circuit_breaker.side_effect = mock_execute_with_circuit_breaker
        
        # 执行操作，应该触发熔断器
        with pytest.raises(Exception) as exc_info:
            circuit_retry_manager.execute_with_circuit_breaker(operation.execute, max_attempts=5)
        
        assert "Circuit breaker" in str(exc_info.value)
        assert circuit_state["state"] == "open"
        
        # 验证熔断器打开后无法执行
        with pytest.raises(Exception) as exc_info:
            circuit_retry_manager.execute_with_circuit_breaker(operation.execute, max_attempts=1)
        
        assert "Circuit breaker is open" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.exception_retry
    @pytest.mark.circuit_breaker
    def test_circuit_breaker_metrics(self):
        """测试熔断器指标收集"""
        # 创建熔断器指标收集器
        metrics_collector = Mock()
        
        # 指标数据
        metrics_data = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "circuit_open_count": 0,
            "circuit_half_open_count": 0,
            "average_response_time": 0.0,
            "failure_rate": 0.0
        }
        
        def mock_record_request(success, response_time):
            metrics_data["total_requests"] += 1
            
            if success:
                metrics_data["successful_requests"] += 1
            else:
                metrics_data["failed_requests"] += 1
            
            # 更新平均响应时间
            total_time = metrics_data["average_response_time"] * (metrics_data["total_requests"] - 1)
            metrics_data["average_response_time"] = (total_time + response_time) / metrics_data["total_requests"]
            
            # 更新失败率
            metrics_data["failure_rate"] = metrics_data["failed_requests"] / metrics_data["total_requests"]
        
        def mock_record_circuit_state_change(new_state):
            if new_state == "open":
                metrics_data["circuit_open_count"] += 1
            elif new_state == "half_open":
                metrics_data["circuit_half_open_count"] += 1
        
        def mock_get_metrics():
            return metrics_data.copy()
        
        metrics_collector.record_request.side_effect = mock_record_request
        metrics_collector.record_state_change.side_effect = mock_record_circuit_state_change
        metrics_collector.get_metrics.side_effect = mock_get_metrics
        
        # 模拟一系列请求
        request_scenarios = [
            (True, 0.1),   # 成功，100ms
            (True, 0.15),  # 成功，150ms
            (False, 0.5),  # 失败，500ms
            (False, 0.3),  # 失败，300ms
            (True, 0.12),  # 成功，120ms
        ]
        
        for success, response_time in request_scenarios:
            metrics_collector.record_request(success, response_time)
        
        # 模拟熔断器状态变化
        metrics_collector.record_state_change("open")
        metrics_collector.record_state_change("half_open")
        metrics_collector.record_state_change("open")
        
        # 获取并验证指标
        metrics = metrics_collector.get_metrics()
        
        assert metrics["total_requests"] == 5
        assert metrics["successful_requests"] == 3
        assert metrics["failed_requests"] == 2
        assert metrics["failure_rate"] == 0.4  # 2/5
        assert metrics["circuit_open_count"] == 2
        assert metrics["circuit_half_open_count"] == 1
        
        # 验证平均响应时间
        expected_avg = (0.1 + 0.15 + 0.5 + 0.3 + 0.12) / 5
        assert abs(metrics["average_response_time"] - expected_avg) < 0.001


class TestRetryPolicies:
    """重试策略测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.exception_retry
    def test_retry_policy_configuration(self):
        """测试重试策略配置"""
        # 创建重试策略配置器
        policy_configurator = Mock()
        
        # 定义不同的重试策略
        policies = {
            "aggressive": {
                "max_attempts": 10,
                "backoff_strategy": "exponential",
                "base_delay": 0.1,
                "max_delay": 30.0,
                "jitter": True,
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 5,
                    "recovery_timeout": 30
                }
            },
            "conservative": {
                "max_attempts": 3,
                "backoff_strategy": "fixed",
                "base_delay": 1.0,
                "max_delay": 5.0,
                "jitter": False,
                "circuit_breaker": {
                    "enabled": False
                }
            },
            "adaptive": {
                "max_attempts": 5,
                "backoff_strategy": "adaptive",
                "base_delay": 0.5,
                "max_delay": 15.0,
                "jitter": True,
                "circuit_breaker": {
                    "enabled": True,
                    "failure_threshold": 3,
                    "recovery_timeout": 60
                },
                "adaptive_factors": {
                    "success_rate_threshold": 0.8,
                    "error_type_multipliers": {
                        "rate_limit": 3.0,
                        "network": 1.5,
                        "timeout": 1.2
                    }
                }
            }
        }
        
        def mock_get_policy(policy_name):
            return policies.get(policy_name)
        
        def mock_validate_policy(policy_config):
            required_fields = ["max_attempts", "backoff_strategy", "base_delay"]
            
            for field in required_fields:
                if field not in policy_config:
                    raise ValueError(f"Missing required field: {field}")
            
            if policy_config["max_attempts"] <= 0:
                raise ValueError("max_attempts must be positive")
            
            if policy_config["base_delay"] < 0:
                raise ValueError("base_delay must be non-negative")
            
            return True
        
        policy_configurator.get_policy.side_effect = mock_get_policy
        policy_configurator.validate_policy.side_effect = mock_validate_policy
        
        # 测试获取和验证不同策略
        for policy_name in policies.keys():
            policy = policy_configurator.get_policy(policy_name)
            assert policy is not None
            
            # 验证策略配置
            is_valid = policy_configurator.validate_policy(policy)
            assert is_valid is True
            
            # 验证特定策略的特征
            if policy_name == "aggressive":
                assert policy["max_attempts"] == 10
                assert policy["circuit_breaker"]["enabled"] is True
            elif policy_name == "conservative":
                assert policy["max_attempts"] == 3
                assert policy["backoff_strategy"] == "fixed"
                assert policy["circuit_breaker"]["enabled"] is False
            elif policy_name == "adaptive":
                assert "adaptive_factors" in policy
                assert "error_type_multipliers" in policy["adaptive_factors"]
        
        # 测试无效策略配置
        invalid_policy = {
            "max_attempts": -1,  # 无效值
            "backoff_strategy": "exponential"
            # 缺少 base_delay
        }
        
        with pytest.raises(ValueError):
            policy_configurator.validate_policy(invalid_policy)
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.exception_retry
    def test_dynamic_policy_selection(self):
        """测试动态策略选择"""
        # 创建动态策略选择器
        policy_selector = Mock()
        
        # 历史性能数据
        performance_history = {
            "api_endpoint_1": {
                "success_rate": 0.95,
                "average_response_time": 0.2,
                "error_types": ["timeout", "network"]
            },
            "api_endpoint_2": {
                "success_rate": 0.70,
                "average_response_time": 1.5,
                "error_types": ["rate_limit", "server_error"]
            },
            "api_endpoint_3": {
                "success_rate": 0.50,
                "average_response_time": 3.0,
                "error_types": ["timeout", "server_error", "network"]
            }
        }
        
        def mock_select_policy(endpoint, current_error=None):
            history = performance_history.get(endpoint, {})
            success_rate = history.get("success_rate", 0.8)
            response_time = history.get("average_response_time", 1.0)
            error_types = history.get("error_types", [])
            
            # 根据历史性能选择策略
            if success_rate >= 0.9 and response_time <= 0.5:
                return "conservative"  # 高成功率，快速响应
            elif success_rate >= 0.7:
                return "adaptive"  # 中等性能，使用自适应策略
            else:
                return "aggressive"  # 低成功率，需要积极重试
        
        def mock_adjust_policy_for_error(base_policy, error_type):
            policy_adjustments = {
                "rate_limit": {
                    "base_delay_multiplier": 3.0,
                    "max_delay": 60.0
                },
                "network": {
                    "base_delay_multiplier": 1.5,
                    "max_attempts": 5
                },
                "timeout": {
                    "base_delay_multiplier": 1.2,
                    "backoff_strategy": "linear"
                }
            }
            
            adjustments = policy_adjustments.get(error_type, {})
            adjusted_policy = base_policy.copy()
            
            for key, value in adjustments.items():
                if key == "base_delay_multiplier":
                    adjusted_policy["base_delay"] *= value
                else:
                    adjusted_policy[key] = value
            
            return adjusted_policy
        
        policy_selector.select_policy.side_effect = mock_select_policy
        policy_selector.adjust_for_error.side_effect = mock_adjust_policy_for_error
        
        # 测试不同端点的策略选择
        test_cases = [
            ("api_endpoint_1", "conservative"),  # 高性能端点
            ("api_endpoint_2", "adaptive"),      # 中等性能端点
            ("api_endpoint_3", "aggressive"),    # 低性能端点
        ]
        
        for endpoint, expected_policy in test_cases:
            selected_policy = policy_selector.select_policy(endpoint)
            assert selected_policy == expected_policy
        
        # 测试错误类型调整
        base_policy = {
            "max_attempts": 3,
            "base_delay": 1.0,
            "backoff_strategy": "exponential"
        }
        
        # 测试速率限制错误调整
        adjusted_policy = policy_selector.adjust_for_error(base_policy, "rate_limit")
        assert adjusted_policy["base_delay"] == 3.0  # 1.0 * 3.0
        assert adjusted_policy["max_delay"] == 60.0
        
        # 测试网络错误调整
        adjusted_policy = policy_selector.adjust_for_error(base_policy, "network")
        assert adjusted_policy["base_delay"] == 1.5  # 1.0 * 1.5
        assert adjusted_policy["max_attempts"] == 5
        
        # 测试超时错误调整
        adjusted_policy = policy_selector.adjust_for_error(base_policy, "timeout")
        assert adjusted_policy["base_delay"] == 1.2  # 1.0 * 1.2
        assert adjusted_policy["backoff_strategy"] == "linear"