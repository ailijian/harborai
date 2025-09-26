#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""异常处理模块单元测试"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Any, Optional

from harborai.core.exceptions import (
    HarborAIError,
    PluginError,
    PluginNotFoundError,
    PluginLoadError,
    PluginExecutionError,
    APIError,
    RateLimitError,
    AuthenticationError,
    ValidationError,
    ConfigurationError,
    DatabaseError,
    RetryableError,
    NonRetryableError
)


class TestHarborAIError:
    """HarborAIError基础异常类测试"""
    
    def test_basic_initialization(self):
        """测试基础初始化"""
        error = HarborAIError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.details == {}
        assert error.original_exception is None
    
    def test_initialization_with_all_parameters(self):
        """测试带所有参数的初始化"""
        original_exc = ValueError("Original error")
        details = {"key": "value", "number": 42}
        
        error = HarborAIError(
            message="Custom error",
            error_code="CUSTOM_001",
            details=details,
            original_exception=original_exc
        )
        
        assert str(error) == "Custom error"
        assert error.message == "Custom error"
        assert error.error_code == "CUSTOM_001"
        assert error.details == details
        assert error.original_exception is original_exc
    
    def test_repr_method(self):
        """测试__repr__方法"""
        error = HarborAIError("Test message", error_code="TEST_001")
        repr_str = repr(error)
        
        assert "HarborAIError" in repr_str
        assert "Test message" in repr_str
        assert "TEST_001" in repr_str
    
    def test_inheritance(self):
        """测试继承关系"""
        error = HarborAIError("Test")
        
        assert isinstance(error, Exception)
        assert isinstance(error, HarborAIError)
    
    def test_details_immutability(self):
        """测试details字典的不可变性"""
        original_details = {"key": "value"}
        error = HarborAIError("Test", details=original_details)
        
        # 修改原始字典不应影响异常中的details
        original_details["new_key"] = "new_value"
        assert "new_key" not in error.details
        
        # 异常中的details应该是副本
        assert error.details == {"key": "value"}


class TestPluginError:
    """PluginError插件异常类测试"""
    
    def test_basic_initialization(self):
        """测试基础初始化"""
        error = PluginError("Plugin error", plugin_name="test_plugin")
        
        assert str(error) == "Plugin error"
        assert error.plugin_name == "test_plugin"
        assert isinstance(error, HarborAIError)
    
    def test_initialization_without_plugin_name(self):
        """测试不提供插件名的初始化"""
        error = PluginError("Plugin error")
        
        assert str(error) == "Plugin error"
        assert error.plugin_name is None
    
    def test_inheritance_chain(self):
        """测试继承链"""
        error = PluginError("Test", plugin_name="test")
        
        assert isinstance(error, Exception)
        assert isinstance(error, HarborAIError)
        assert isinstance(error, PluginError)


class TestPluginNotFoundError:
    """PluginNotFoundError插件未找到异常测试"""
    
    def test_initialization(self):
        """测试初始化"""
        error = PluginNotFoundError("Plugin not found", plugin_name="missing_plugin")
        
        assert str(error) == "Plugin not found"
        assert error.plugin_name == "missing_plugin"
        assert isinstance(error, PluginError)
    
    def test_default_message_formatting(self):
        """测试默认消息格式化"""
        error = PluginNotFoundError(plugin_name="missing_plugin")
        
        # 检查是否包含插件名
        assert "missing_plugin" in str(error)


class TestPluginLoadError:
    """PluginLoadError插件加载异常测试"""
    
    def test_initialization_with_original_exception(self):
        """测试带原始异常的初始化"""
        original_exc = ImportError("Module not found")
        error = PluginLoadError(
            "Failed to load plugin",
            plugin_name="test_plugin",
            original_exception=original_exc
        )
        
        assert str(error) == "Failed to load plugin"
        assert error.plugin_name == "test_plugin"
        assert error.original_exception is original_exc
        assert isinstance(error, PluginError)


class TestPluginExecutionError:
    """PluginExecutionError插件执行异常测试"""
    
    def test_initialization_with_details(self):
        """测试带详细信息的初始化"""
        details = {"method": "process", "input_data": "test"}
        error = PluginExecutionError(
            "Plugin execution failed",
            plugin_name="test_plugin",
            details=details
        )
        
        assert str(error) == "Plugin execution failed"
        assert error.plugin_name == "test_plugin"
        assert error.details == details
        assert isinstance(error, PluginError)


class TestAPIError:
    """APIError API异常类测试"""
    
    def test_initialization_with_status_code(self):
        """测试带状态码的初始化"""
        error = APIError("API request failed", status_code=404)
        
        assert str(error) == "API request failed"
        assert error.status_code == 404
        assert isinstance(error, HarborAIError)
    
    def test_initialization_with_response_data(self):
        """测试带响应数据的初始化"""
        response_data = {"error": "Not found", "code": "RESOURCE_NOT_FOUND"}
        error = APIError(
            "API error",
            status_code=404,
            response_data=response_data
        )
        
        assert error.response_data == response_data
        assert error.status_code == 404
    
    def test_initialization_without_optional_params(self):
        """测试不带可选参数的初始化"""
        error = APIError("Simple API error")
        
        assert str(error) == "Simple API error"
        assert error.status_code is None
        assert error.response_data is None


class TestRateLimitError:
    """RateLimitError速率限制异常测试"""
    
    def test_initialization_with_retry_after(self):
        """测试带重试时间的初始化"""
        error = RateLimitError("Rate limit exceeded", retry_after=60)
        
        assert str(error) == "Rate limit exceeded"
        assert error.retry_after == 60
        assert isinstance(error, APIError)
        assert isinstance(error, RetryableError)
    
    def test_initialization_without_retry_after(self):
        """测试不带重试时间的初始化"""
        error = RateLimitError("Rate limit exceeded")
        
        assert error.retry_after is None
    
    def test_inheritance_chain(self):
        """测试继承链"""
        error = RateLimitError("Test")
        
        assert isinstance(error, Exception)
        assert isinstance(error, HarborAIError)
        assert isinstance(error, APIError)
        assert isinstance(error, RetryableError)
        assert isinstance(error, RateLimitError)


class TestAuthenticationError:
    """AuthenticationError认证异常测试"""
    
    def test_initialization(self):
        """测试初始化"""
        error = AuthenticationError("Invalid API key")
        
        assert str(error) == "Invalid API key"
        assert isinstance(error, APIError)
        assert isinstance(error, NonRetryableError)
    
    def test_inheritance_chain(self):
        """测试继承链"""
        error = AuthenticationError("Test")
        
        assert isinstance(error, Exception)
        assert isinstance(error, HarborAIError)
        assert isinstance(error, APIError)
        assert isinstance(error, NonRetryableError)
        assert isinstance(error, AuthenticationError)


class TestValidationError:
    """ValidationError验证异常测试"""
    
    def test_initialization_with_field_errors(self):
        """测试带字段错误的初始化"""
        field_errors = {
            "email": "Invalid email format",
            "age": "Must be a positive integer"
        }
        error = ValidationError("Validation failed", field_errors=field_errors)
        
        assert str(error) == "Validation failed"
        assert error.field_errors == field_errors
        assert isinstance(error, HarborAIError)
        assert isinstance(error, NonRetryableError)
    
    def test_initialization_without_field_errors(self):
        """测试不带字段错误的初始化"""
        error = ValidationError("General validation error")
        
        assert error.field_errors == {}
    
    def test_field_errors_immutability(self):
        """测试字段错误的不可变性"""
        original_errors = {"field": "error"}
        error = ValidationError("Test", field_errors=original_errors)
        
        # 修改原始字典不应影响异常中的field_errors
        original_errors["new_field"] = "new_error"
        assert "new_field" not in error.field_errors


class TestConfigurationError:
    """ConfigurationError配置异常测试"""
    
    def test_initialization_with_config_key(self):
        """测试带配置键的初始化"""
        error = ConfigurationError(
            "Invalid configuration",
            config_key="database.host"
        )
        
        assert str(error) == "Invalid configuration"
        assert error.config_key == "database.host"
        assert isinstance(error, HarborAIError)
        assert isinstance(error, NonRetryableError)
    
    def test_initialization_without_config_key(self):
        """测试不带配置键的初始化"""
        error = ConfigurationError("General config error")
        
        assert error.config_key is None


class TestDatabaseError:
    """DatabaseError数据库异常测试"""
    
    def test_initialization_with_query(self):
        """测试带查询语句的初始化"""
        query = "SELECT * FROM users WHERE id = ?"
        error = DatabaseError("Query failed", query=query)
        
        assert str(error) == "Query failed"
        assert error.query == query
        assert isinstance(error, HarborAIError)
        assert isinstance(error, RetryableError)
    
    def test_initialization_without_query(self):
        """测试不带查询语句的初始化"""
        error = DatabaseError("Connection failed")
        
        assert error.query is None
    
    def test_inheritance_chain(self):
        """测试继承链"""
        error = DatabaseError("Test")
        
        assert isinstance(error, Exception)
        assert isinstance(error, HarborAIError)
        assert isinstance(error, RetryableError)
        assert isinstance(error, DatabaseError)


class TestRetryableError:
    """RetryableError可重试异常测试"""
    
    def test_initialization(self):
        """测试初始化"""
        error = RetryableError("Temporary failure")
        
        assert str(error) == "Temporary failure"
        assert isinstance(error, HarborAIError)
    
    def test_as_mixin(self):
        """测试作为混入类使用"""
        # 测试其他异常类是否正确继承了RetryableError
        rate_limit_error = RateLimitError("Rate limited")
        database_error = DatabaseError("DB error")
        
        assert isinstance(rate_limit_error, RetryableError)
        assert isinstance(database_error, RetryableError)


class TestNonRetryableError:
    """NonRetryableError不可重试异常测试"""
    
    def test_initialization(self):
        """测试初始化"""
        error = NonRetryableError("Permanent failure")
        
        assert str(error) == "Permanent failure"
        assert isinstance(error, HarborAIError)
    
    def test_as_mixin(self):
        """测试作为混入类使用"""
        # 测试其他异常类是否正确继承了NonRetryableError
        auth_error = AuthenticationError("Auth failed")
        validation_error = ValidationError("Validation failed")
        config_error = ConfigurationError("Config error")
        
        assert isinstance(auth_error, NonRetryableError)
        assert isinstance(validation_error, NonRetryableError)
        assert isinstance(config_error, NonRetryableError)


class TestExceptionHierarchy:
    """异常层次结构测试"""
    
    def test_retryable_vs_non_retryable_separation(self):
        """测试可重试和不可重试异常的分离"""
        # 可重试异常
        retryable_exceptions = [
            RateLimitError("Rate limited"),
            DatabaseError("DB error")
        ]
        
        # 不可重试异常
        non_retryable_exceptions = [
            AuthenticationError("Auth failed"),
            ValidationError("Validation failed"),
            ConfigurationError("Config error")
        ]
        
        # 验证可重试异常
        for exc in retryable_exceptions:
            assert isinstance(exc, RetryableError)
            assert not isinstance(exc, NonRetryableError)
        
        # 验证不可重试异常
        for exc in non_retryable_exceptions:
            assert isinstance(exc, NonRetryableError)
            assert not isinstance(exc, RetryableError)
    
    def test_all_exceptions_inherit_from_harborai_error(self):
        """测试所有异常都继承自HarborAIError"""
        exceptions = [
            PluginError("Plugin error"),
            PluginNotFoundError("Plugin not found"),
            PluginLoadError("Plugin load error"),
            PluginExecutionError("Plugin execution error"),
            APIError("API error"),
            RateLimitError("Rate limit error"),
            AuthenticationError("Auth error"),
            ValidationError("Validation error"),
            ConfigurationError("Config error"),
            DatabaseError("Database error"),
            RetryableError("Retryable error"),
            NonRetryableError("Non-retryable error")
        ]
        
        for exc in exceptions:
            assert isinstance(exc, HarborAIError)
            assert isinstance(exc, Exception)
    
    def test_plugin_error_hierarchy(self):
        """测试插件异常层次结构"""
        plugin_exceptions = [
            PluginNotFoundError("Not found"),
            PluginLoadError("Load error"),
            PluginExecutionError("Execution error")
        ]
        
        for exc in plugin_exceptions:
            assert isinstance(exc, PluginError)
            assert isinstance(exc, HarborAIError)
    
    def test_api_error_hierarchy(self):
        """测试API异常层次结构"""
        api_exceptions = [
            RateLimitError("Rate limited"),
            AuthenticationError("Auth failed")
        ]
        
        for exc in api_exceptions:
            assert isinstance(exc, APIError)
            assert isinstance(exc, HarborAIError)


class TestExceptionUsagePatterns:
    """异常使用模式测试"""
    
    def test_exception_chaining(self):
        """测试异常链"""
        original_exc = ValueError("Original error")
        
        try:
            raise original_exc
        except ValueError as e:
            harbor_exc = HarborAIError(
                "Wrapped error",
                original_exception=e
            )
            
            assert harbor_exc.original_exception is e
            assert str(harbor_exc.original_exception) == "Original error"
    
    def test_exception_with_context_details(self):
        """测试带上下文详情的异常"""
        context = {
            "user_id": "12345",
            "operation": "data_processing",
            "timestamp": "2024-01-01T00:00:00Z",
            "input_size": 1024
        }
        
        error = PluginExecutionError(
            "Processing failed",
            plugin_name="data_processor",
            details=context
        )
        
        assert error.details["user_id"] == "12345"
        assert error.details["operation"] == "data_processing"
        assert error.plugin_name == "data_processor"
    
    def test_error_code_usage(self):
        """测试错误码使用"""
        error = APIError(
            "Request failed",
            error_code="API_001",
            status_code=500
        )
        
        assert error.error_code == "API_001"
        assert error.status_code == 500
    
    def test_field_validation_error_usage(self):
        """测试字段验证错误使用"""
        field_errors = {
            "name": "Name is required",
            "email": "Invalid email format",
            "age": "Age must be between 18 and 100"
        }
        
        error = ValidationError(
            "User data validation failed",
            field_errors=field_errors
        )
        
        assert len(error.field_errors) == 3
        assert "name" in error.field_errors
        assert "email" in error.field_errors
        assert "age" in error.field_errors
    
    def test_retry_after_usage(self):
        """测试重试时间使用"""
        error = RateLimitError(
            "Too many requests",
            retry_after=300,  # 5分钟后重试
            status_code=429
        )
        
        assert error.retry_after == 300
        assert error.status_code == 429
        assert isinstance(error, RetryableError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])