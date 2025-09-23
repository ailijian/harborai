#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常处理模块

定义 HarborAI 中使用的各种异常类，提供标准化的错误处理机制。
"""

from typing import Optional, Dict, Any


class HarborAIError(Exception):
    """HarborAI 基础异常类"""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.trace_id = trace_id
    
    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.trace_id:
            base_msg = f"{base_msg} (trace_id: {self.trace_id})"
        return base_msg
    
    def to_dict(self) -> Dict[str, Any]:
        """将异常转换为字典格式"""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "trace_id": self.trace_id,
        }


class APIError(HarborAIError):
    """API 调用异常"""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.status_code = status_code
        self.response_body = response_body
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update({
            "status_code": self.status_code,
            "response_body": self.response_body,
        })
        return result


class AuthenticationError(APIError):
    """认证异常"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, error_code="AUTHENTICATION_ERROR", **kwargs)


class RateLimitError(APIError):
    """速率限制异常"""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        super().__init__(message, error_code="RATE_LIMIT_ERROR", **kwargs)
        self.retry_after = retry_after
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["retry_after"] = self.retry_after
        return result


class TimeoutError(APIError):
    """超时异常"""
    
    def __init__(self, message: str = "Request timeout", **kwargs):
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)


class ModelNotFoundError(HarborAIError):
    """模型未找到异常"""
    
    def __init__(self, model_name: str, **kwargs):
        message = f"Model '{model_name}' not found or not supported"
        super().__init__(message, error_code="MODEL_NOT_FOUND", **kwargs)
        self.model_name = model_name
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["model_name"] = self.model_name
        return result


class PluginError(HarborAIError):
    """插件异常"""
    
    def __init__(self, plugin_name: str, message: str, **kwargs):
        full_message = f"Plugin '{plugin_name}': {message}"
        super().__init__(full_message, error_code="PLUGIN_ERROR", **kwargs)
        self.plugin_name = plugin_name
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["plugin_name"] = self.plugin_name
        return result


class StructuredOutputError(HarborAIError):
    """结构化输出异常"""
    
    def __init__(self, message: str, provider: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="STRUCTURED_OUTPUT_ERROR", **kwargs)
        self.provider = provider
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["provider"] = self.provider
        return result


class ValidationError(HarborAIError):
    """参数验证异常"""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field
    
    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result["field"] = self.field
        return result