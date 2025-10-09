#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常处理模块测试

测试 HarborAI 异常类的功能，确保错误处理机制正常工作。
"""

import pytest
from typing import Dict, Any

from harborai.utils.exceptions import (
    HarborAIError,
    APIError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ModelNotFoundError,
    PluginError,
    StructuredOutputError,
    ValidationError,
    StorageError,
)


class TestHarborAIError:
    """HarborAI基础异常类测试"""
    
    def test_basic_initialization(self):
        """测试基础初始化"""
        error = HarborAIError("测试错误")
        
        assert str(error) == "测试错误"
        assert error.message == "测试错误"
        assert error.error_code is None
        assert error.details == {}
        assert error.trace_id is None
    
    def test_full_initialization(self):
        """测试完整初始化"""
        details = {"key": "value", "count": 42}
        error = HarborAIError(
            message="完整错误信息",
            error_code="TEST_ERROR",
            details=details,
            trace_id="trace_123"
        )
        
        assert error.message == "完整错误信息"
        assert error.error_code == "TEST_ERROR"
        assert error.details == details
        assert error.trace_id == "trace_123"
    
    def test_str_representation(self):
        """测试字符串表示"""
        # 仅消息
        error1 = HarborAIError("简单错误")
        assert str(error1) == "简单错误"
        
        # 带错误码
        error2 = HarborAIError("错误消息", error_code="ERR001")
        assert str(error2) == "[ERR001] 错误消息"
        
        # 带trace_id
        error3 = HarborAIError("错误消息", trace_id="trace_456")
        assert str(error3) == "错误消息 (trace_id: trace_456)"
        
        # 完整信息
        error4 = HarborAIError(
            "完整错误",
            error_code="ERR002",
            trace_id="trace_789"
        )
        assert str(error4) == "[ERR002] 完整错误 (trace_id: trace_789)"
    
    def test_to_dict(self):
        """测试转换为字典"""
        details = {"field": "value"}
        error = HarborAIError(
            message="字典测试",
            error_code="DICT_TEST",
            details=details,
            trace_id="trace_dict"
        )
        
        result = error.to_dict()
        expected = {
            "error": "HarborAIError",
            "message": "字典测试",
            "error_code": "DICT_TEST",
            "details": details,
            "trace_id": "trace_dict",
        }
        
        assert result == expected
    
    def test_empty_details_default(self):
        """测试空details的默认值"""
        error = HarborAIError("测试", details=None)
        assert error.details == {}


class TestAPIError:
    """API异常类测试"""
    
    def test_basic_initialization(self):
        """测试基础初始化"""
        error = APIError("API错误")
        
        assert error.message == "API错误"
        assert error.status_code is None
        assert error.response_body is None
    
    def test_full_initialization(self):
        """测试完整初始化"""
        error = APIError(
            message="API调用失败",
            status_code=500,
            response_body='{"error": "Internal Server Error"}',
            error_code="API_FAIL",
            details={"endpoint": "/api/test"},
            trace_id="api_trace"
        )
        
        assert error.message == "API调用失败"
        assert error.status_code == 500
        assert error.response_body == '{"error": "Internal Server Error"}'
        assert error.error_code == "API_FAIL"
        assert error.details == {"endpoint": "/api/test"}
        assert error.trace_id == "api_trace"
    
    def test_to_dict(self):
        """测试转换为字典"""
        error = APIError(
            message="API错误",
            status_code=404,
            response_body="Not Found",
            error_code="NOT_FOUND"
        )
        
        result = error.to_dict()
        expected = {
            "error": "APIError",
            "message": "API错误",
            "error_code": "NOT_FOUND",
            "details": {},
            "trace_id": None,
            "status_code": 404,
            "response_body": "Not Found",
        }
        
        assert result == expected


class TestAuthenticationError:
    """认证异常类测试"""
    
    def test_default_message(self):
        """测试默认消息"""
        error = AuthenticationError()
        
        assert error.message == "Authentication failed"
        assert error.error_code == "AUTHENTICATION_ERROR"
    
    def test_custom_message(self):
        """测试自定义消息"""
        error = AuthenticationError("无效的API密钥")
        
        assert error.message == "无效的API密钥"
        assert error.error_code == "AUTHENTICATION_ERROR"
    
    def test_with_additional_params(self):
        """测试附加参数"""
        error = AuthenticationError(
            "认证失败",
            status_code=401,
            trace_id="auth_trace"
        )
        
        assert error.message == "认证失败"
        assert error.error_code == "AUTHENTICATION_ERROR"
        assert error.status_code == 401
        assert error.trace_id == "auth_trace"


class TestRateLimitError:
    """速率限制异常类测试"""
    
    def test_default_message(self):
        """测试默认消息"""
        error = RateLimitError()
        
        assert error.message == "Rate limit exceeded"
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.retry_after is None
    
    def test_with_retry_after(self):
        """测试带重试时间"""
        error = RateLimitError(
            message="请求过于频繁",
            retry_after=60
        )
        
        assert error.message == "请求过于频繁"
        assert error.error_code == "RATE_LIMIT_ERROR"
        assert error.retry_after == 60
    
    def test_to_dict(self):
        """测试转换为字典"""
        error = RateLimitError(
            message="速率限制",
            retry_after=30,
            status_code=429
        )
        
        result = error.to_dict()
        
        assert result["error"] == "RateLimitError"
        assert result["message"] == "速率限制"
        assert result["error_code"] == "RATE_LIMIT_ERROR"
        assert result["retry_after"] == 30
        assert result["status_code"] == 429


class TestTimeoutError:
    """超时异常类测试"""
    
    def test_default_message(self):
        """测试默认消息"""
        error = TimeoutError()
        
        assert error.message == "Request timeout"
        assert error.error_code == "TIMEOUT_ERROR"
    
    def test_custom_message(self):
        """测试自定义消息"""
        error = TimeoutError("连接超时")
        
        assert error.message == "连接超时"
        assert error.error_code == "TIMEOUT_ERROR"


class TestModelNotFoundError:
    """模型未找到异常类测试"""
    
    def test_initialization(self):
        """测试初始化"""
        error = ModelNotFoundError("gpt-4")
        
        assert error.message == "Model 'gpt-4' not found or not supported"
        assert error.error_code == "MODEL_NOT_FOUND"
        assert error.model_name == "gpt-4"
    
    def test_to_dict(self):
        """测试转换为字典"""
        error = ModelNotFoundError("claude-3")
        
        result = error.to_dict()
        
        assert result["error"] == "ModelNotFoundError"
        assert result["message"] == "Model 'claude-3' not found or not supported"
        assert result["error_code"] == "MODEL_NOT_FOUND"
        assert result["model_name"] == "claude-3"


class TestPluginError:
    """插件异常类测试"""
    
    def test_initialization(self):
        """测试初始化"""
        error = PluginError("test_plugin", "插件加载失败")
        
        assert error.message == "Plugin 'test_plugin': 插件加载失败"
        assert error.error_code == "PLUGIN_ERROR"
        assert error.plugin_name == "test_plugin"
    
    def test_to_dict(self):
        """测试转换为字典"""
        error = PluginError("auth_plugin", "认证插件错误")
        
        result = error.to_dict()
        
        assert result["error"] == "PluginError"
        assert result["message"] == "Plugin 'auth_plugin': 认证插件错误"
        assert result["error_code"] == "PLUGIN_ERROR"
        assert result["plugin_name"] == "auth_plugin"


class TestStructuredOutputError:
    """结构化输出异常类测试"""
    
    def test_basic_initialization(self):
        """测试基础初始化"""
        error = StructuredOutputError("解析失败")
        
        assert error.message == "解析失败"
        assert error.error_code == "STRUCTURED_OUTPUT_ERROR"
        assert error.provider is None
    
    def test_with_provider(self):
        """测试带提供商信息"""
        error = StructuredOutputError("JSON解析错误", provider="openai")
        
        assert error.message == "JSON解析错误"
        assert error.error_code == "STRUCTURED_OUTPUT_ERROR"
        assert error.provider == "openai"
    
    def test_to_dict(self):
        """测试转换为字典"""
        error = StructuredOutputError("格式错误", provider="anthropic")
        
        result = error.to_dict()
        
        assert result["error"] == "StructuredOutputError"
        assert result["message"] == "格式错误"
        assert result["error_code"] == "STRUCTURED_OUTPUT_ERROR"
        assert result["provider"] == "anthropic"


class TestValidationError:
    """验证异常类测试"""
    
    def test_basic_initialization(self):
        """测试基础初始化"""
        error = ValidationError("验证失败")
        
        assert error.message == "验证失败"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.field is None
    
    def test_with_field(self):
        """测试带字段信息"""
        error = ValidationError("字段值无效", field="email")
        
        assert error.message == "字段值无效"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.field == "email"
    
    def test_to_dict(self):
        """测试转换为字典"""
        error = ValidationError("必填字段缺失", field="username")
        
        result = error.to_dict()
        
        assert result["error"] == "ValidationError"
        assert result["message"] == "必填字段缺失"
        assert result["error_code"] == "VALIDATION_ERROR"
        assert result["field"] == "username"


class TestStorageError:
    """存储异常类测试"""
    
    def test_basic_initialization(self):
        """测试基础初始化"""
        error = StorageError("存储失败")
        
        assert error.message == "存储失败"
        assert error.error_code == "STORAGE_ERROR"
        assert error.storage_type is None
    
    def test_with_storage_type(self):
        """测试带存储类型"""
        error = StorageError("数据库连接失败", storage_type="postgresql")
        
        assert error.message == "数据库连接失败"
        assert error.error_code == "STORAGE_ERROR"
        assert error.storage_type == "postgresql"
    
    def test_to_dict(self):
        """测试转换为字典"""
        error = StorageError("文件写入失败", storage_type="filesystem")
        
        result = error.to_dict()
        
        assert result["error"] == "StorageError"
        assert result["message"] == "文件写入失败"
        assert result["error_code"] == "STORAGE_ERROR"
        assert result["storage_type"] == "filesystem"


class TestExceptionInheritance:
    """异常继承关系测试"""
    
    def test_inheritance_chain(self):
        """测试继承链"""
        # 所有异常都应该继承自HarborAIError
        assert issubclass(APIError, HarborAIError)
        assert issubclass(AuthenticationError, APIError)
        assert issubclass(RateLimitError, APIError)
        assert issubclass(TimeoutError, APIError)
        assert issubclass(ModelNotFoundError, HarborAIError)
        assert issubclass(PluginError, HarborAIError)
        assert issubclass(StructuredOutputError, HarborAIError)
        assert issubclass(ValidationError, HarborAIError)
        assert issubclass(StorageError, HarborAIError)
        
        # 所有异常都应该继承自Exception
        assert issubclass(HarborAIError, Exception)
    
    def test_exception_catching(self):
        """测试异常捕获"""
        # 测试可以用基类捕获子类异常
        try:
            raise AuthenticationError("认证失败")
        except HarborAIError as e:
            assert isinstance(e, AuthenticationError)
            assert isinstance(e, APIError)
            assert isinstance(e, HarborAIError)
        
        try:
            raise ModelNotFoundError("gpt-5")
        except HarborAIError as e:
            assert isinstance(e, ModelNotFoundError)
            assert isinstance(e, HarborAIError)


class TestEdgeCases:
    """边界情况测试"""
    
    def test_unicode_handling(self):
        """测试Unicode字符处理"""
        error = HarborAIError("错误：包含中文字符 🚨")
        
        assert "中文字符" in str(error)
        assert "🚨" in str(error)
        
        result = error.to_dict()
        assert "中文字符" in result["message"]
        assert "🚨" in result["message"]
    
    def test_empty_string_message(self):
        """测试空字符串消息"""
        error = HarborAIError("")
        
        assert error.message == ""
        assert str(error) == ""
    
    def test_none_values_handling(self):
        """测试None值处理"""
        error = APIError(
            message="测试",
            status_code=None,
            response_body=None,
            error_code=None,
            details=None,
            trace_id=None
        )
        
        result = error.to_dict()
        assert result["status_code"] is None
        assert result["response_body"] is None
        assert result["error_code"] is None
        assert result["details"] == {}
        assert result["trace_id"] is None
    
    def test_large_details_dict(self):
        """测试大型details字典"""
        large_details = {f"key_{i}": f"value_{i}" for i in range(1000)}
        error = HarborAIError("大型详情", details=large_details)
        
        assert len(error.details) == 1000
        assert error.details["key_500"] == "value_500"
        
        result = error.to_dict()
        assert len(result["details"]) == 1000
    
    def test_nested_details(self):
        """测试嵌套details"""
        nested_details = {
            "level1": {
                "level2": {
                    "level3": ["item1", "item2", "item3"]
                }
            },
            "simple": "value"
        }
        error = HarborAIError("嵌套详情", details=nested_details)
        
        assert error.details["level1"]["level2"]["level3"] == ["item1", "item2", "item3"]
        assert error.details["simple"] == "value"
        
        result = error.to_dict()
        assert result["details"]["level1"]["level2"]["level3"] == ["item1", "item2", "item3"]


if __name__ == '__main__':
    pytest.main([__file__])