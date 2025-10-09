#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 异常处理模块测试

测试自定义异常类的功能和层次结构。
"""

import pytest

from harborai.core.exceptions import (
    HarborAIError, RetryableError, NonRetryableError,
    ConfigurationError, ValidationError, ParameterValidationError,
    APIError, AuthenticationError, RateLimitError,
    ModelNotFoundError, ModelNotSupportedError, TokenLimitExceededError,
    PluginError, PluginLoadError, PluginNotFoundError,
    PluginConfigError, PluginExecutionError, BudgetExceededError,
    NetworkError, TimeoutError, QuotaExceededError,
    ServiceUnavailableError, DatabaseError
)


class TestHarborAIError:
    """测试基础异常类"""
    
    def test_basic_initialization(self):
        """测试基础初始化"""
        error = HarborAIError("测试错误")
        
        assert str(error) == "测试错误"
        assert error.message == "测试错误"
        assert error.error_code is None
        assert error.details == {}
        assert error.original_exception is None
    
    def test_full_initialization(self):
        """测试完整初始化"""
        details = {"key": "value", "status": 500}
        original_exception = ValueError("原始错误")
        
        error = HarborAIError(
            message="详细错误信息",
            error_code="TEST_ERROR",
            details=details,
            original_exception=original_exception
        )
        
        assert error.message == "详细错误信息"
        assert error.error_code == "TEST_ERROR"
        assert error.details == details
        assert error.original_exception == original_exception
    
    def test_kwargs_support(self):
        """测试关键字参数支持"""
        error = HarborAIError(
            "测试错误",
            custom_field="自定义值",
            timestamp=123456789
        )
        
        assert error.custom_field == "自定义值"
        assert error.timestamp == 123456789
    
    def test_details_copy(self):
        """测试details字典复制"""
        original_details = {"key": "value"}
        error = HarborAIError("测试", details=original_details)
        
        # 修改原始字典不应影响异常中的details
        original_details["new_key"] = "new_value"
        assert "new_key" not in error.details
    
    def test_inheritance(self):
        """测试继承关系"""
        error = HarborAIError("测试")
        assert isinstance(error, Exception)
        assert isinstance(error, HarborAIError)
    
    def test_string_representation(self):
        """测试字符串表示"""
        error = HarborAIError("测试错误", error_code="TEST_001")
        
        assert str(error) == "测试错误"
        assert "HarborAIError" in repr(error)
        assert "测试错误" in repr(error)
        assert "TEST_001" in repr(error)
    
    def test_repr_without_error_code(self):
        """测试无错误码的repr"""
        error = HarborAIError("测试错误")
        repr_str = repr(error)
        
        assert "HarborAIError" in repr_str
        assert "测试错误" in repr_str
        assert "error_code" not in repr_str


class TestRetryableError:
    """测试可重试异常类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = RetryableError("可重试错误")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, RetryableError)
    
    def test_initialization(self):
        """测试初始化"""
        error = RetryableError("网络错误", error_code="NETWORK_001")
        
        assert error.message == "网络错误"
        assert error.error_code == "NETWORK_001"


class TestNonRetryableError:
    """测试不可重试异常类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = NonRetryableError("不可重试错误")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, NonRetryableError)
    
    def test_initialization(self):
        """测试初始化"""
        error = NonRetryableError("永久错误", error_code="PERMANENT_001")
        
        assert error.message == "永久错误"
        assert error.error_code == "PERMANENT_001"


class TestConfigurationError:
    """测试配置错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = ConfigurationError("配置错误")
        assert isinstance(error, NonRetryableError)
        assert isinstance(error, ConfigurationError)
    
    def test_initialization(self):
        """测试初始化"""
        error = ConfigurationError("无效配置", config_key="api_key")
        
        assert error.message == "无效配置"
        assert error.config_key == "api_key"
    
    def test_config_key_none(self):
        """测试config_key为None"""
        error = ConfigurationError("配置问题")
        assert error.config_key is None


class TestValidationError:
    """测试验证错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = ValidationError("验证错误")
        assert isinstance(error, NonRetryableError)
        assert isinstance(error, ValidationError)
    
    def test_initialization(self):
        """测试初始化"""
        field_errors = {"temperature": "值必须在0-2之间", "max_tokens": "必须为正整数"}
        error = ValidationError("参数验证失败", field_errors=field_errors)
        
        assert error.message == "参数验证失败"
        assert error.field_errors == field_errors
    
    def test_field_errors_copy(self):
        """测试field_errors字典复制"""
        original_errors = {"field1": "error1"}
        error = ValidationError("验证失败", field_errors=original_errors)
        
        # 修改原始字典不应影响异常中的field_errors
        original_errors["field2"] = "error2"
        assert "field2" not in error.field_errors
    
    def test_default_values(self):
        """测试默认值"""
        error = ValidationError("验证失败")
        assert error.field_errors == {}


class TestParameterValidationError:
    """测试参数验证错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = ParameterValidationError("参数验证错误")
        assert isinstance(error, ValidationError)
        assert isinstance(error, NonRetryableError)
        assert isinstance(error, ParameterValidationError)


class TestAPIError:
    """测试API错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = APIError("API错误")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, APIError)
    
    def test_initialization(self):
        """测试初始化"""
        response_data = {"error": "内部错误", "code": 500}
        error = APIError("API调用失败", status_code=500, response_data=response_data)
        
        assert error.message == "API调用失败"
        assert error.status_code == 500
        assert error.response_data == response_data
    
    def test_default_values(self):
        """测试默认值"""
        error = APIError("API错误")
        assert error.status_code is None
        assert error.response_data is None


class TestAuthenticationError:
    """测试认证错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = AuthenticationError("认证失败")
        assert isinstance(error, APIError)
        assert isinstance(error, NonRetryableError)
        assert isinstance(error, AuthenticationError)
    
    def test_initialization(self):
        """测试初始化"""
        error = AuthenticationError("API密钥无效", status_code=401)
        
        assert error.message == "API密钥无效"
        assert error.status_code == 401


class TestRateLimitError:
    """测试速率限制错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = RateLimitError("速率限制")
        assert isinstance(error, APIError)
        assert isinstance(error, RetryableError)
        assert isinstance(error, RateLimitError)
    
    def test_initialization(self):
        """测试初始化"""
        error = RateLimitError(
            "速率限制超出",
            status_code=429,
            retry_after=60
        )
        
        assert error.message == "速率限制超出"
        assert error.status_code == 429
        assert error.retry_after == 60
    
    def test_default_values(self):
        """测试默认值"""
        error = RateLimitError("速率限制")
        assert error.retry_after is None


class TestModelNotFoundError:
    """测试模型未找到错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = ModelNotFoundError("模型未找到")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, ModelNotFoundError)


class TestModelNotSupportedError:
    """测试模型不支持错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = ModelNotSupportedError("模型不支持")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, ModelNotSupportedError)


class TestTokenLimitExceededError:
    """测试Token限制超出错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = TokenLimitExceededError("Token限制超出")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, TokenLimitExceededError)


class TestPluginError:
    """测试插件错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = PluginError("插件错误")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, PluginError)
    
    def test_initialization(self):
        """测试初始化"""
        error = PluginError("插件执行失败", plugin_name="test_plugin")
        
        assert error.message == "插件执行失败"
        assert error.plugin_name == "test_plugin"
    
    def test_default_values(self):
        """测试默认值"""
        error = PluginError("插件错误")
        assert error.plugin_name is None


class TestPluginLoadError:
    """测试插件加载错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = PluginLoadError("插件加载失败")
        assert isinstance(error, PluginError)
        assert isinstance(error, PluginLoadError)


class TestPluginNotFoundError:
    """测试插件未找到错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = PluginNotFoundError("插件未找到")
        assert isinstance(error, PluginError)
        assert isinstance(error, PluginNotFoundError)
    
    def test_default_message_with_plugin_name(self):
        """测试带插件名的默认消息"""
        error = PluginNotFoundError(plugin_name="test_plugin")
        assert error.message == "Plugin 'test_plugin' not found"
        assert error.plugin_name == "test_plugin"
    
    def test_default_message_without_plugin_name(self):
        """测试无插件名的默认消息"""
        error = PluginNotFoundError()
        assert error.message == "Plugin not found"
        assert error.plugin_name is None
    
    def test_custom_message(self):
        """测试自定义消息"""
        error = PluginNotFoundError("自定义错误消息", plugin_name="test_plugin")
        assert error.message == "自定义错误消息"
        assert error.plugin_name == "test_plugin"


class TestPluginConfigError:
    """测试插件配置错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = PluginConfigError("插件配置错误")
        assert isinstance(error, PluginError)
        assert isinstance(error, PluginConfigError)


class TestPluginExecutionError:
    """测试插件执行错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = PluginExecutionError("插件执行错误")
        assert isinstance(error, PluginError)
        assert isinstance(error, PluginExecutionError)


class TestBudgetExceededError:
    """测试预算超出错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = BudgetExceededError("预算超出")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, BudgetExceededError)


class TestNetworkError:
    """测试网络错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = NetworkError("网络错误")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, NetworkError)


class TestTimeoutError:
    """测试超时错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = TimeoutError("超时错误")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, TimeoutError)


class TestQuotaExceededError:
    """测试配额超出错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = QuotaExceededError("配额超出")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, QuotaExceededError)


class TestServiceUnavailableError:
    """测试服务不可用错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = ServiceUnavailableError("服务不可用")
        assert isinstance(error, HarborAIError)
        assert isinstance(error, ServiceUnavailableError)


class TestDatabaseError:
    """测试数据库错误类"""
    
    def test_inheritance(self):
        """测试继承关系"""
        error = DatabaseError("数据库错误")
        assert isinstance(error, RetryableError)
        assert isinstance(error, DatabaseError)
    
    def test_initialization(self):
        """测试初始化"""
        error = DatabaseError("查询失败", query="SELECT * FROM users")
        
        assert error.message == "查询失败"
        assert error.query == "SELECT * FROM users"
    
    def test_default_values(self):
        """测试默认值"""
        error = DatabaseError("数据库错误")
        assert error.query is None


class TestExceptionHierarchy:
    """测试异常层次结构"""
    
    def test_retryable_error_hierarchy(self):
        """测试可重试异常层次"""
        retryable_exceptions = [
            RetryableError("可重试"),
            RateLimitError("速率限制"),
            DatabaseError("数据库错误")
        ]
        
        for error in retryable_exceptions:
            assert isinstance(error, RetryableError)
            assert isinstance(error, HarborAIError)
    
    def test_non_retryable_error_hierarchy(self):
        """测试不可重试异常层次"""
        non_retryable_exceptions = [
            NonRetryableError("不可重试"),
            ConfigurationError("配置错误"),
            ValidationError("验证错误"),
            AuthenticationError("认证错误")
        ]
        
        for error in non_retryable_exceptions:
            assert isinstance(error, NonRetryableError)
            assert isinstance(error, HarborAIError)
    
    def test_plugin_error_hierarchy(self):
        """测试插件错误层次"""
        plugin_exceptions = [
            PluginError("插件错误"),
            PluginLoadError("加载错误"),
            PluginNotFoundError("未找到"),
            PluginConfigError("配置错误"),
            PluginExecutionError("执行错误")
        ]
        
        for error in plugin_exceptions:
            assert isinstance(error, PluginError)
            assert isinstance(error, HarborAIError)
    
    def test_api_error_hierarchy(self):
        """测试API错误层次"""
        api_exceptions = [
            APIError("API错误"),
            AuthenticationError("认证错误"),
            RateLimitError("速率限制")
        ]
        
        for error in api_exceptions:
            assert isinstance(error, APIError)
            assert isinstance(error, HarborAIError)
    
    def test_all_exceptions_inherit_from_harborai_error(self):
        """测试所有异常都继承自HarborAIError"""
        all_exceptions = [
            HarborAIError("基础错误"),
            RetryableError("可重试"),
            NonRetryableError("不可重试"),
            ConfigurationError("配置错误"),
            ValidationError("验证错误"),
            ParameterValidationError("参数验证错误"),
            APIError("API错误"),
            AuthenticationError("认证错误"),
            RateLimitError("速率限制"),
            ModelNotFoundError("模型未找到"),
            ModelNotSupportedError("模型不支持"),
            TokenLimitExceededError("Token限制"),
            PluginError("插件错误"),
            PluginLoadError("插件加载错误"),
            PluginNotFoundError("插件未找到"),
            PluginConfigError("插件配置错误"),
            PluginExecutionError("插件执行错误"),
            BudgetExceededError("预算超出"),
            NetworkError("网络错误"),
            TimeoutError("超时错误"),
            QuotaExceededError("配额超出"),
            ServiceUnavailableError("服务不可用"),
            DatabaseError("数据库错误")
        ]
        
        for error in all_exceptions:
            assert isinstance(error, HarborAIError)
            assert isinstance(error, Exception)


class TestExceptionUsagePatterns:
    """测试异常使用模式"""
    
    def test_exception_chaining(self):
        """测试异常链"""
        try:
            raise ValueError("原始错误")
        except ValueError as e:
            chained_error = APIError("API调用失败")
            chained_error.__cause__ = e
            
            assert chained_error.__cause__ is e
            assert isinstance(chained_error.__cause__, ValueError)
    
    def test_exception_with_context(self):
        """测试异常上下文"""
        try:
            try:
                raise ConnectionError("连接失败")
            except ConnectionError:
                raise NetworkError("网络错误")
        except NetworkError as e:
            assert e.__context__ is not None
            assert isinstance(e.__context__, ConnectionError)
    
    def test_exception_with_kwargs(self):
        """测试异常关键字参数"""
        error = APIError(
            "API错误",
            status_code=500,
            request_id="req_123",
            user_id="user_456"
        )
        
        assert error.status_code == 500
        assert error.request_id == "req_123"
        assert error.user_id == "user_456"
    
    def test_exception_filtering_by_type(self):
        """测试按类型过滤异常"""
        exceptions = [
            RetryableError("可重试1"),
            NonRetryableError("不可重试1"),
            RateLimitError("速率限制"),
            AuthenticationError("认证失败"),
            DatabaseError("数据库错误"),
            PluginError("插件错误")
        ]
        
        # 过滤可重试异常
        retryable_errors = [e for e in exceptions if isinstance(e, RetryableError)]
        assert len(retryable_errors) == 3  # RetryableError, RateLimitError, DatabaseError
        
        # 过滤不可重试异常
        non_retryable_errors = [e for e in exceptions if isinstance(e, NonRetryableError)]
        assert len(non_retryable_errors) == 2  # NonRetryableError, AuthenticationError
        
        # 过滤插件异常
        plugin_errors = [e for e in exceptions if isinstance(e, PluginError)]
        assert len(plugin_errors) == 1  # PluginError
    
    def test_retry_logic_pattern(self):
        """测试重试逻辑模式"""
        def should_retry(error):
            """判断是否应该重试"""
            return isinstance(error, RetryableError)
        
        # 可重试错误
        retryable_errors = [
            RetryableError("临时错误"),
            RateLimitError("速率限制"),
            DatabaseError("连接超时")
        ]
        
        for error in retryable_errors:
            assert should_retry(error) is True
        
        # 不可重试错误
        non_retryable_errors = [
            NonRetryableError("永久错误"),
            AuthenticationError("认证失败"),
            ConfigurationError("配置错误")
        ]
        
        for error in non_retryable_errors:
            assert should_retry(error) is False
    
    def test_error_details_handling(self):
        """测试错误详情处理"""
        details = {
            "request_id": "req_123",
            "timestamp": "2024-01-01T00:00:00Z",
            "endpoint": "/api/v1/chat/completions"
        }
        
        error = APIError("请求失败", status_code=500, details=details)
        
        assert error.details["request_id"] == "req_123"
        assert error.details["timestamp"] == "2024-01-01T00:00:00Z"
        assert error.details["endpoint"] == "/api/v1/chat/completions"
    
    def test_original_exception_tracking(self):
        """测试原始异常跟踪"""
        original = ConnectionError("网络连接失败")
        wrapped = NetworkError("网络错误", original_exception=original)
        
        assert wrapped.original_exception is original
        assert isinstance(wrapped.original_exception, ConnectionError)
        assert str(wrapped.original_exception) == "网络连接失败"