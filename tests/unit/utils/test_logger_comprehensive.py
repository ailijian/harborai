#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志工具模块comprehensive测试

测试 HarborAI 日志工具的所有功能，确保日志记录机制正常工作。
"""

import pytest
import logging
import sys
import json
import os
import threading
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from harborai.utils.logger import (
    get_logger,
    setup_logging,
    LogContext,
    sanitize_log_data,
    APICallLogger,
)


class TestGetLogger:
    """get_logger函数测试"""
    
    def test_basic_logger_creation(self):
        """测试基础日志器创建"""
        logger = get_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
    
    def test_custom_level(self):
        """测试自定义日志级别"""
        logger = get_logger("test_debug", level="DEBUG")
        assert logger.level == logging.DEBUG
        
        logger = get_logger("test_error", level="ERROR")
        assert logger.level == logging.ERROR
    
    def test_with_log_file(self):
        """测试带日志文件的logger获取"""
        with patch('pathlib.Path.mkdir'), \
             patch('logging.FileHandler') as mock_file_handler:
            
            logger = get_logger("test_file_logger", log_file="test.log")
            assert logger.name == "test_file_logger"
            assert len(logger.handlers) == 2  # console + file handler
            mock_file_handler.assert_called_once()
    
    def test_nested_log_file_creation(self):
        """测试嵌套目录日志文件创建"""
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('logging.FileHandler') as mock_file_handler:
            
            logger = get_logger("test_nested", log_file="nested/dir/test.log")
            
            # 验证目录创建被调用
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_file_handler.assert_called_once()
    
    def test_avoid_duplicate_handlers(self):
        """测试避免重复添加处理器"""
        logger1 = get_logger("duplicate_test")
        handler_count1 = len(logger1.handlers)
        
        logger2 = get_logger("duplicate_test")
        handler_count2 = len(logger2.handlers)
        
        assert logger1 is logger2
        assert handler_count1 == handler_count2
    
    def test_formatter_configuration(self):
        """测试格式器配置"""
        logger = get_logger("format_test")
        handler = logger.handlers[0]
        formatter = handler.formatter
        
        assert formatter is not None
        assert '%(asctime)s' in formatter._fmt
        assert '%(name)s' in formatter._fmt
        assert '%(levelname)s' in formatter._fmt
        assert '%(message)s' in formatter._fmt


class TestSetupLogging:
    """setup_logging函数测试"""
    
    def test_basic_setup(self):
        """测试基础日志设置"""
        with patch('logging.basicConfig') as mock_config:
            setup_logging()
            
            mock_config.assert_called_once()
            args, kwargs = mock_config.call_args
            assert kwargs['level'] == logging.INFO
            assert len(kwargs['handlers']) == 1
            assert isinstance(kwargs['handlers'][0], logging.StreamHandler)
    
    def test_custom_level_setup(self):
        """测试自定义级别设置"""
        with patch('logging.basicConfig') as mock_config:
            setup_logging(level="DEBUG")
            
            args, kwargs = mock_config.call_args
            assert kwargs['level'] == logging.DEBUG
    
    def test_with_log_file_setup(self):
        """测试带日志文件的setup_logging"""
        with patch('logging.basicConfig') as mock_config:
            setup_logging(log_file="test.log")
            
            args, kwargs = mock_config.call_args
            assert 'handlers' in kwargs
            assert len(kwargs['handlers']) == 2  # console + file handler


class TestLogContext:
    """LogContext类测试"""
    
    def test_basic_initialization(self):
        """测试基础初始化"""
        context = LogContext()
        
        assert context.trace_id is None
        assert context.span_id is None
        assert context.user_id is None
        assert context.session_id is None
        assert context.request_id is None
        assert context.extra == {}
    
    def test_full_initialization(self):
        """测试完整初始化"""
        extra = {"key": "value", "count": 42}
        context = LogContext(
            trace_id="trace_123",
            span_id="span_456",
            user_id="user_789",
            session_id="session_abc",
            request_id="request_def",
            extra=extra
        )
        
        assert context.trace_id == "trace_123"
        assert context.span_id == "span_456"
        assert context.user_id == "user_789"
        assert context.session_id == "session_abc"
        assert context.request_id == "request_def"
        assert context.extra == extra
    
    def test_to_dict_empty(self):
        """测试空上下文转换为字典"""
        context = LogContext()
        result = context.to_dict()
        
        assert result == {}
    
    def test_to_dict_partial(self):
        """测试部分字段转换为字典"""
        context = LogContext(
            trace_id="trace_123",
            user_id="user_789",
            extra={"custom": "data"}
        )
        result = context.to_dict()
        
        expected = {
            "trace_id": "trace_123",
            "user_id": "user_789",
            "custom": "data"
        }
        assert result == expected
    
    def test_to_dict_full(self):
        """测试完整字段转换为字典"""
        extra = {"key1": "value1", "key2": "value2"}
        context = LogContext(
            trace_id="trace_123",
            span_id="span_456",
            user_id="user_789",
            session_id="session_abc",
            request_id="request_def",
            extra=extra
        )
        result = context.to_dict()
        
        expected = {
            "trace_id": "trace_123",
            "span_id": "span_456",
            "user_id": "user_789",
            "session_id": "session_abc",
            "request_id": "request_def",
            "key1": "value1",
            "key2": "value2"
        }
        assert result == expected


class TestSanitizeLogData:
    """sanitize_log_data函数测试"""
    
    def test_none_input(self):
        """测试None输入"""
        result = sanitize_log_data(None)
        assert result is None
    
    def test_simple_types(self):
        """测试简单类型"""
        assert sanitize_log_data(42) == 42
        assert sanitize_log_data(3.14) == 3.14
        assert sanitize_log_data(True) is True
        assert sanitize_log_data(False) is False
    
    def test_string_normal(self):
        """测试正常字符串"""
        result = sanitize_log_data("normal string")
        assert result == "normal string"
    
    def test_string_truncation(self):
        """测试字符串截断"""
        long_string = "a" * 1500
        result = sanitize_log_data(long_string, max_length=1000)
        
        assert len(result) == 1000 + len("...[truncated]")
        assert result.endswith("...[truncated]")
        assert result.startswith("a" * 1000)
    
    def test_dict_normal(self):
        """测试正常字典"""
        data = {"name": "test", "count": 42}
        result = sanitize_log_data(data)
        
        assert result == data
    
    def test_dict_sensitive_fields(self):
        """测试敏感字段清理"""
        data = {
            "username": "user123",
            "password": "secret123",
            "api_key": "key_abc",
            "token": "token_xyz",
            "secret": "secret_data",
            "authorization": "Bearer token",
            "private_key": "private_data",
            "normal_field": "normal_value"
        }
        
        result = sanitize_log_data(data)
        
        assert result["username"] == "user123"
        assert result["normal_field"] == "normal_value"
        assert result["password"] == "[REDACTED]"
        assert result["api_key"] == "[REDACTED]"
        assert result["token"] == "[REDACTED]"
        assert result["secret"] == "[REDACTED]"
        assert result["authorization"] == "[REDACTED]"
        assert result["private_key"] == "[REDACTED]"
    
    def test_nested_dict(self):
        """测试嵌套字典"""
        data = {
            "user": {
                "name": "test",
                "password": "secret"
            },
            "config": {
                "api_key": "key123",
                "timeout": 30
            }
        }
        
        result = sanitize_log_data(data)
        
        assert result["user"]["name"] == "test"
        assert result["user"]["password"] == "[REDACTED]"
        assert result["config"]["api_key"] == "[REDACTED]"
        assert result["config"]["timeout"] == 30
    
    def test_list_processing(self):
        """测试列表处理"""
        data = [
            "normal",
            {"password": "secret", "name": "test"},
            42,
            ["nested", {"token": "abc"}]
        ]
        
        result = sanitize_log_data(data)
        
        assert result[0] == "normal"
        assert result[1]["name"] == "test"
        assert result[1]["password"] == "[REDACTED]"
        assert result[2] == 42
        assert result[3][0] == "nested"
        assert result[3][1]["token"] == "[REDACTED]"
    
    def test_tuple_processing(self):
        """测试元组处理"""
        data = ("normal", {"password": "secret"}, 42)
        result = sanitize_log_data(data)
        
        assert result[0] == "normal"
        assert result[1]["password"] == "[REDACTED]"
        assert result[2] == 42
    
    def test_other_types(self):
        """测试其他类型"""
        class CustomClass:
            def __str__(self):
                return "custom_object_string"
        
        obj = CustomClass()
        result = sanitize_log_data(obj)
        assert result == "custom_object_string"
    
    def test_other_types_truncation(self):
        """测试其他类型截断"""
        class LongStringClass:
            def __str__(self):
                return "x" * 1500
        
        obj = LongStringClass()
        result = sanitize_log_data(obj, max_length=1000)
        
        assert len(result) == 1000 + len("...[truncated]")
        assert result.endswith("...[truncated]")


class TestAPICallLogger:
    """APICallLogger类测试"""
    
    def test_initialization(self):
        """测试APICallLogger初始化"""
        mock_logger = Mock()
        api_logger = APICallLogger(mock_logger)
        assert api_logger.logger is mock_logger
        assert hasattr(api_logger, '_get_fallback_logger')
        assert hasattr(api_logger, '_lock')
    
    def test_get_fallback_logger_success(self):
        """测试fallback logger获取成功"""
        mock_logger = Mock()
        api_logger = APICallLogger(mock_logger)
        
        # Mock the fallback logger module and function
        mock_fallback = Mock()
        mock_module = Mock()
        mock_module.get_fallback_logger = Mock(return_value=mock_fallback)
        
        with patch.dict('sys.modules', {'harborai.storage.fallback_logger': mock_module}):
            fallback = api_logger._get_fallback_logger()
            assert fallback is mock_fallback
    
    def test_get_fallback_logger_import_error(self):
        """测试fallback logger导入错误处理"""
        mock_logger = Mock()
        api_logger = APICallLogger(mock_logger)
        
        # 模拟导入错误
        with patch('builtins.__import__', side_effect=ImportError("Module not found")):
            fallback = api_logger._get_fallback_logger()
            assert fallback is None
    
    def test_log_request_with_fallback_logger(self):
        """测试使用fallback_logger记录请求"""
        mock_logger = Mock()
        api_logger = APICallLogger(mock_logger)
        
        mock_fallback = Mock()
        context = LogContext(trace_id="test_trace")
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7
        }
        
        with patch.object(api_logger, '_get_fallback_logger', return_value=mock_fallback):
            with patch('sys.stderr'):  # 抑制调试输出
                api_logger.log_request(context, request_data)
            
            mock_fallback.log_request.assert_called_once_with(
                trace_id="test_trace",
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7
            )
    
    def test_log_request_without_fallback_logger(self):
        """测试没有fallback_logger时记录请求"""
        mock_logger = Mock()
        api_logger = APICallLogger(mock_logger)
        
        context = LogContext(trace_id="test_trace")
        request_data = {"model": "gpt-3.5-turbo"}
        
        with patch.object(api_logger, '_get_fallback_logger', return_value=None):
            with patch('sys.stderr'):  # 抑制调试输出
                # 应该不会抛出异常
                api_logger.log_request(context, request_data)
    
    def test_log_request_fallback_exception(self):
        """测试fallback_logger异常处理"""
        mock_logger = Mock()
        api_logger = APICallLogger(mock_logger)
        
        mock_fallback = Mock()
        mock_fallback.log_request.side_effect = Exception("Fallback error")
        
        context = LogContext(trace_id="test_trace")
        request_data = {"model": "gpt-3.5-turbo", "messages": []}
        
        with patch.object(api_logger, '_get_fallback_logger', return_value=mock_fallback):
            with patch('sys.stderr'):  # 抑制调试输出
                # 应该不会抛出异常，而是使用普通logger
                api_logger.log_request(context, request_data)
                
                # 验证fallback_logger被调用了
                mock_fallback.log_request.assert_called_once()


class TestIntegration:
    """集成测试"""
    
    def test_logger_with_context_and_sanitization(self):
        """测试logger与上下文和数据清理的集成"""
        # 只测试数据清理功能，不涉及文件操作
        logger = get_logger("integration_test")
        
        # 测试数据清理
        sensitive_data = {
            "username": "test_user",
            "password": "secret123",
            "api_key": "sk-1234567890",
            "normal_field": "normal_value"
        }
        
        sanitized = sanitize_log_data(sensitive_data)
        logger.info("Integration test")
        
        assert sanitized["username"] == "test_user"
        assert sanitized["password"] == "[REDACTED]"
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["normal_field"] == "normal_value"
    
    def test_api_call_logger_integration(self):
        """测试API调用日志器集成"""
        logger = get_logger("api_integration")
        api_logger = APICallLogger(logger)
        
        context = LogContext(
            trace_id="api_trace",
            request_id="req_123"
        )
        
        request_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.8,
            "api_key": "secret_key"  # 这个应该被清理
        }
        
        # 清理请求数据
        sanitized_request = sanitize_log_data(request_data)
        
        # 验证清理效果
        assert sanitized_request["api_key"] == "[REDACTED]"
        assert sanitized_request["model"] == "gpt-4"
        
        # 测试API日志记录（模拟没有fallback_logger的情况）
        with patch.object(api_logger, '_get_fallback_logger', return_value=None):
            with patch('sys.stderr'):  # 抑制调试输出
                api_logger.log_request(context, sanitized_request)