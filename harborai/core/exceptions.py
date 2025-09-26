#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HarborAI核心异常定义

定义HarborAI系统中使用的各种异常类型。
"""


class HarborAIError(Exception):
    """HarborAI基础异常类"""
    
    def __init__(self, message: str, error_code: str = None, details: dict = None, trace_id: str = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.trace_id = trace_id
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(HarborAIError):
    """配置错误异常"""
    pass


class ValidationError(HarborAIError):
    """验证错误异常"""
    pass


class ParameterValidationError(ValidationError):
    """参数验证错误异常"""
    pass


class APIError(HarborAIError):
    """API调用错误异常"""
    pass


class AuthenticationError(HarborAIError):
    """认证错误异常"""
    pass


class RateLimitError(HarborAIError):
    """速率限制错误异常"""
    pass


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
    pass


class PluginLoadError(PluginError):
    """插件加载失败时抛出的异常"""
    pass


class PluginNotFoundError(PluginError):
    """插件未找到时抛出的异常"""
    pass


class PluginConfigError(PluginError):
    """插件配置错误时抛出的异常"""
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


class RetryableError(HarborAIError):
    """可重试错误异常"""
    pass


class NonRetryableError(HarborAIError):
    """不可重试错误异常"""
    pass


class QuotaExceededError(HarborAIError):
    """配额超限错误异常"""
    pass


class ServiceUnavailableError(HarborAIError):
    """服务不可用错误异常"""
    pass