#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日志工具模块

提供统一的日志配置和获取功能。
"""

import logging
import sys
import json
import threading
from typing import Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
from contextvars import ContextVar
from dataclasses import dataclass, field


def get_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """获取配置好的日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        log_file: 日志文件路径（可选）
    
    Returns:
        配置好的日志器实例
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 设置日志级别
    logger.setLevel(getattr(logging, level.upper()))
    
    # 创建格式器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 添加文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """设置全局日志配置
    
    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ] + ([logging.FileHandler(log_file, encoding='utf-8')] if log_file else [])
    )


# 上下文变量
_log_context: ContextVar[Dict[str, Any]] = ContextVar('log_context', default={})


@dataclass
class LogContext:
    """日志上下文"""
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        if self.trace_id:
            result['trace_id'] = self.trace_id
        if self.span_id:
            result['span_id'] = self.span_id
        if self.user_id:
            result['user_id'] = self.user_id
        if self.session_id:
            result['session_id'] = self.session_id
        if self.request_id:
            result['request_id'] = self.request_id
        result.update(self.extra)
        return result


def sanitize_log_data(data: Any, max_length: int = 1000) -> Any:
    """清理日志数据，移除敏感信息并限制长度
    
    Args:
        data: 要清理的数据
        max_length: 字符串最大长度
    
    Returns:
        清理后的数据
    """
    if data is None:
        return None
    
    # 敏感字段列表
    sensitive_fields = {
        'password', 'token', 'key', 'secret', 'api_key', 'access_token',
        'refresh_token', 'authorization', 'auth', 'credential', 'private_key'
    }
    
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            key_lower = str(key).lower()
            if any(sensitive in key_lower for sensitive in sensitive_fields):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_log_data(value, max_length)
        return sanitized
    
    elif isinstance(data, (list, tuple)):
        return [sanitize_log_data(item, max_length) for item in data]
    
    elif isinstance(data, str):
        if len(data) > max_length:
            return data[:max_length] + "...[truncated]"
        return data
    
    elif isinstance(data, (int, float, bool)):
        return data
    
    else:
        # 对于其他类型，转换为字符串并限制长度
        str_data = str(data)
        if len(str_data) > max_length:
            return str_data[:max_length] + "...[truncated]"
        return str_data


class APICallLogger:
    """API调用日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._lock = threading.Lock()
        # 不在初始化时获取fallback_logger，而是在使用时动态获取
        
    def _get_fallback_logger(self):
        """动态获取fallback_logger"""
        try:
            from ..storage.fallback_logger import get_fallback_logger
            return get_fallback_logger()
        except ImportError:
            return None
    
    def log_request(self, context: LogContext, request_data: dict) -> None:
        """记录API请求"""
        import sys
        sys.stderr.write(f"[DEBUG] APICallLogger.log_request called\n")
        sys.stderr.flush()
        
        fallback_logger = self._get_fallback_logger()
        sys.stderr.write(f"[DEBUG] APICallLogger.log_request: fallback_logger = {fallback_logger}\n")
        sys.stderr.write(f"[DEBUG] APICallLogger.log_request: fallback_logger type = {type(fallback_logger)}\n")
        sys.stderr.flush()
        
        if fallback_logger:
            sys.stderr.write(f"[DEBUG] 使用 fallback_logger 记录请求\n")
            sys.stderr.flush()
            try:
                # 从request_data中提取参数以适配FallbackLogger接口
                model = request_data.get('model', 'unknown')
                messages = request_data.get('messages', [])
                
                # 创建kwargs包含其他参数
                kwargs = {k: v for k, v in request_data.items() if k not in ['model', 'messages']}
                
                fallback_logger.log_request(
                    trace_id=context.trace_id,
                    model=model,
                    messages=messages,
                    **kwargs
                )
                sys.stderr.write(f"[DEBUG] fallback_logger.log_request 成功\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"[DEBUG] fallback_logger.log_request 失败: {e}\n")
                sys.stderr.flush()
        else:
            sys.stderr.write(f"[DEBUG] 没有 fallback_logger，使用标准logger\n")
            sys.stderr.flush()
            self.logger.info(f"API Request [trace_id={context.trace_id}] {request_data}")

    def log_response(self, context: LogContext, response_data: dict) -> None:
        """记录API响应"""
        import sys
        sys.stderr.write(f"[DEBUG] APICallLogger.log_response called\n")
        sys.stderr.flush()
        
        fallback_logger = self._get_fallback_logger()
        sys.stderr.write(f"[DEBUG] APICallLogger.log_response: fallback_logger = {fallback_logger}\n")
        sys.stderr.write(f"[DEBUG] APICallLogger.log_response: fallback_logger type = {type(fallback_logger)}\n")
        sys.stderr.flush()
        
        if fallback_logger:
            sys.stderr.write(f"[DEBUG] 使用 fallback_logger 记录响应\n")
            sys.stderr.flush()
            try:
                # 从response_data中提取参数以适配FallbackLogger接口
                status_code = response_data.get('status_code', 200)
                response = response_data.get('response')
                duration = response_data.get('duration', 0.0)
                
                # 判断是否成功
                success = status_code < 400
                error = f"HTTP {status_code}" if not success else None
                
                fallback_logger.log_response(
                    trace_id=context.trace_id,
                    response=response,
                    latency=duration,
                    success=success,
                    error=error
                )
                sys.stderr.write(f"[DEBUG] fallback_logger.log_response 成功\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"[DEBUG] fallback_logger.log_response 失败: {e}\n")
                sys.stderr.flush()
        else:
            sys.stderr.write(f"[DEBUG] 没有 fallback_logger，使用标准logger\n")
            sys.stderr.flush()
            self.logger.info(f"API Response [trace_id={context.trace_id}] {response_data}")

    def log_error(self, context: LogContext, error_data: dict) -> None:
        """记录API错误"""
        import sys
        sys.stderr.write(f"[DEBUG] APICallLogger.log_error called\n")
        sys.stderr.flush()
        
        fallback_logger = self._get_fallback_logger()
        sys.stderr.write(f"[DEBUG] APICallLogger.log_error: fallback_logger = {fallback_logger}\n")
        sys.stderr.write(f"[DEBUG] APICallLogger.log_error: fallback_logger type = {type(fallback_logger)}\n")
        sys.stderr.flush()
        
        if fallback_logger:
            sys.stderr.write(f"[DEBUG] 使用 fallback_logger 记录错误\n")
            sys.stderr.flush()
            try:
                fallback_logger.log_error(context, error_data)
                sys.stderr.write(f"[DEBUG] fallback_logger.log_error 成功\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"[DEBUG] fallback_logger.log_error 失败: {e}\n")
                sys.stderr.flush()
        else:
            sys.stderr.write(f"[DEBUG] 没有 fallback_logger，使用标准logger\n")
            sys.stderr.flush()
            self.logger.error(f"API Error [trace_id={context.trace_id}] {error_data}")

    async def alog_request(self, context: LogContext, request_data: dict) -> None:
        """异步记录API请求"""
        self.log_request(context, request_data)
    
    async def alog_response(self, context: LogContext, response_data: dict) -> None:
        """异步记录API响应"""
        self.log_response(context, response_data)
    
    async def alog_error(self, context: LogContext, error_data: dict) -> None:
        """异步记录API错误"""
        self.log_error(context, error_data)