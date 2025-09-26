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
    
    def log_request(self, method: str, url: str, headers: Dict[str, str] = None,
                   body: Any = None, context: LogContext = None):
        """记录API请求
        
        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            body: 请求体
            context: 日志上下文
        """
        log_data = {
            'event': 'api_request',
            'method': method,
            'url': url,
            'timestamp': datetime.now().isoformat()
        }
        
        if headers:
            log_data['headers'] = sanitize_log_data(headers)
        
        if body:
            log_data['body'] = sanitize_log_data(body)
        
        if context:
            log_data.update(context.to_dict())
        
        with self._lock:
            self.logger.info("API Request", extra={'structured_data': log_data})
    
    def log_response(self, status_code: int, headers: Dict[str, str] = None,
                    body: Any = None, duration: float = None,
                    context: LogContext = None):
        """记录API响应
        
        Args:
            status_code: HTTP状态码
            headers: 响应头
            body: 响应体
            duration: 请求持续时间（秒）
            context: 日志上下文
        """
        log_data = {
            'event': 'api_response',
            'status_code': status_code,
            'timestamp': datetime.now().isoformat()
        }
        
        if headers:
            log_data['headers'] = sanitize_log_data(headers)
        
        if body:
            log_data['body'] = sanitize_log_data(body)
        
        if duration is not None:
            log_data['duration'] = duration
        
        if context:
            log_data.update(context.to_dict())
        
        with self._lock:
            if status_code >= 400:
                self.logger.error("API Response Error", extra={'structured_data': log_data})
            else:
                self.logger.info("API Response", extra={'structured_data': log_data})
    
    def log_error(self, error: Exception, context: LogContext = None):
        """记录API错误
        
        Args:
            error: 异常对象
            context: 日志上下文
        """
        log_data = {
            'event': 'api_error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat()
        }
        
        if context:
            log_data.update(context.to_dict())
        
        with self._lock:
            self.logger.error("API Error", exc_info=error, extra={'structured_data': log_data})
    
    async def alog_request(self, method: str, url: str, params: dict = None, headers: Dict[str, str] = None,
                          body: Any = None, context: LogContext = None, trace_id: str = None):
        """异步记录API请求
        
        Args:
            method: HTTP方法
            url: 请求URL
            params: 请求参数
            headers: 请求头
            body: 请求体
            context: 日志上下文
            trace_id: 追踪ID
        """
        self.log_request(method, url, headers, body, context)
    
    async def alog_response(self, response: Any = None, status_code: int = None,
                           headers: Dict[str, str] = None, body: Any = None,
                           duration: float = None, context: LogContext = None,
                           trace_id: str = None):
        """异步记录API响应
        
        Args:
            response: 响应对象
            status_code: HTTP状态码
            headers: 响应头
            body: 响应体
            duration: 请求持续时间（秒）
            context: 日志上下文
            trace_id: 追踪ID
        """
        # 如果传入了response对象，尝试从中提取信息
        if response and hasattr(response, '__dict__'):
            # 对于OpenAI风格的响应对象，记录基本信息
            log_data = {
                'event': 'api_response',
                'response_type': type(response).__name__,
                'timestamp': datetime.now().isoformat()
            }
            
            if trace_id:
                log_data['trace_id'] = trace_id
            
            if context:
                log_data.update(context.to_dict())
            
            with self._lock:
                self.logger.info("API Response", extra={'structured_data': log_data})
        else:
            # 使用传统方式记录响应
            self.log_response(status_code or 200, headers, body, duration, context)
    
    async def alog_error(self, error: Exception = None, model: str = None,
                        plugin_name: str = None, latency_ms: float = None,
                        context: LogContext = None, trace_id: str = None):
        """异步记录API错误
        
        Args:
            error: 异常对象
            model: 模型名称
            plugin_name: 插件名称
            latency_ms: 延迟毫秒数
            context: 日志上下文
            trace_id: 追踪ID
        """
        log_data = {
            'event': 'api_error',
            'timestamp': datetime.now().isoformat()
        }
        
        if error:
            log_data['error_type'] = type(error).__name__
            log_data['error_message'] = str(error)
        
        if model:
            log_data['model'] = model
        
        if plugin_name:
            log_data['plugin_name'] = plugin_name
        
        if latency_ms is not None:
            log_data['latency_ms'] = latency_ms
        
        if trace_id:
            log_data['trace_id'] = trace_id
        
        if context:
            log_data.update(context.to_dict())
        
        with self._lock:
            self.logger.error("API Error", exc_info=error, extra={'structured_data': log_data})