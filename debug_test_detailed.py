from harborai.core.exceptions import NetworkError, TimeoutError, RateLimitError, AuthenticationError
from unittest.mock import Mock
import time

# 创建Mock对象
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
        print(f"RateLimitError detected, setting strategy to: {context['strategy']}")
    elif isinstance(exception, TimeoutError):
        context["retry_recommended"] = True
        context["suggested_delay"] = 0.5
        context["strategy"] = "linear_backoff"
        print(f"TimeoutError detected, setting strategy to: {context['strategy']}")
    elif isinstance(exception, NetworkError):
        context["retry_recommended"] = True
        context["suggested_delay"] = 1
        context["strategy"] = "exponential_backoff"
        print(f"NetworkError detected, setting strategy to: {context['strategy']}")
    elif isinstance(exception, AuthenticationError):
        context["retry_recommended"] = False
        context["strategy"] = "no_retry"
        print(f"AuthenticationError detected, setting strategy to: {context['strategy']}")
    
    return context

context_analyzer.analyze.side_effect = mock_analyze_error_context

# 测试不同错误的上下文分析
test_cases = [
    {
        "exception": RateLimitError("Rate limited"),
        "context": {"operation": "chat_completion", "model": "deepseek-r1"},
        "expected_strategy": "rate_limit_backoff",
        "expected_delay": 60
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

for i, test_case in enumerate(test_cases):
    print(f"\n=== Test Case {i+1} ===")
    print(f"Exception: {type(test_case['exception']).__name__}")
    print(f"Expected strategy: {test_case['expected_strategy']}")
    
    analysis = context_analyzer.analyze(test_case["exception"], test_case["context"])
    
    print(f"Actual strategy: {analysis['strategy']}")
    print(f"Match: {analysis['strategy'] == test_case['expected_strategy']}")
    
    if analysis["strategy"] != test_case["expected_strategy"]:
        print(f"❌ MISMATCH! Expected '{test_case['expected_strategy']}' but got '{analysis['strategy']}'")
        break
    else:
        print("✅ PASS")