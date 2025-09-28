#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HarborAI核心异常定义

定义HarborAI系统中使用的各种异常类型。
"""


class HarborAIError(Exception):
    """HarborAI基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None, original_exception: Exception = None, **kwargs):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details.copy() if details else {}
        self.original_exception = original_exception
        
        # 支持额外的关键字参数
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __str__(self):
        return self.message
    
    def __repr__(self):
        parts = [f"'{self.message}'"]
        if self.error_code:
            parts.append(f"error_code='{self.error_code}'")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class RetryableError(HarborAIError):
    """可重试错误异常"""
    pass


class NonRetryableError(HarborAIError):
    """不可重试错误异常"""
    pass


class ConfigurationError(NonRetryableError):
    """配置错误异常"""
    
    def __init__(self, message: str, config_key: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_key = config_key


class ValidationError(NonRetryableError):
    """验证错误异常"""
    
    def __init__(self, message: str, field_errors: dict = None, **kwargs):
        super().__init__(message, **kwargs)
        self.field_errors = field_errors.copy() if field_errors else {}


class ParameterValidationError(ValidationError):
    """参数验证错误异常"""
    pass


class APIError(HarborAIError):
    """API调用错误异常"""
    
    def __init__(self, message: str, status_code: int = None, response_data: dict = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_data = response_data


class AuthenticationError(APIError, NonRetryableError):
    """认证错误异常"""
    pass


class RateLimitError(APIError, RetryableError):
    """速率限制错误异常"""
    
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class ModelNotFoundError(HarborAIError):
    """模型未找到错误异常"""
    pass


class ModelNotSupportedError(HarborAIError):
    """模型不支持错误异常"""
    pass


class TokenLimitExceededError(HarborAIError):
    """Token限制超出错误异常"""
    pass


class PluginError(HarborAIError):
    """插件相关的基础异常"""
    
    def __init__(self, message: str, plugin_name: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.plugin_name = plugin_name


class PluginLoadError(PluginError):
    """插件加载失败时抛出的异常"""
    pass


class PluginNotFoundError(PluginError):
    """插件未找到时抛出的异常"""
    
    def __init__(self, message: str = None, plugin_name: str = None, **kwargs):
        if message is None and plugin_name:
            message = f"Plugin '{plugin_name}' not found"
        elif message is None:
            message = "Plugin not found"
        super().__init__(message, plugin_name=plugin_name, **kwargs)


class PluginConfigError(PluginError):
    """插件配置错误时抛出的异常"""
    pass


class PluginExecutionError(PluginError):
    """插件执行错误时抛出的异常"""
    pass


class BudgetExceededError(HarborAIError):
    """预算超限时抛出的异常"""
    pass


class NetworkError(HarborAIError):
    """网络错误异常"""
    pass


class TimeoutError(HarborAIError):
    """超时错误异常"""
    pass


class QuotaExceededError(HarborAIError):
    """配额超限错误异常"""
    pass


class ServiceUnavailableError(HarborAIError):
    """服务不可用错误异常"""
    pass


class DatabaseError(RetryableError):
    """数据库错误异常"""
    
    def __init__(self, message: str, query: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.query = query