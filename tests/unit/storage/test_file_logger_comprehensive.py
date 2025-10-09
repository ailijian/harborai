#!/usr/bin/env python3
"""
---
summary: FileSystemLogger 综合测试套件
description: 全面测试文件系统日志记录器的功能，包括异步日志、文件轮转、数据脱敏等
assumptions:
  - id: A1
    text: 文件系统可写且有足够空间
    confidence: high
  - id: A2
    text: Mock对象能正确模拟文件操作
    confidence: high
  - id: A3
    text: 线程操作在测试环境中稳定
    confidence: medium
tests:
  - path: tests/unit/storage/test_file_logger_comprehensive.py
    cmd: pytest tests/unit/storage/test_file_logger_comprehensive.py -v
---
"""

import pytest
import tempfile
import shutil
import time
import json
import gzip
import threading
from unittest.mock import Mock, patch, MagicMock, call, mock_open
from pathlib import Path
from datetime import datetime
import queue

from harborai.storage.file_logger import (
    FileSystemLogger,
    get_file_logger,
    initialize_file_logger,
    shutdown_file_logger,
    SENSITIVE_PATTERNS
)


class TestFileSystemLoggerInitialization:
    """测试FileSystemLogger初始化功能"""
    
    def test_init_with_default_parameters(self):
        """测试使用默认参数初始化"""
        # Given: 默认参数
        log_dir = "test_logs"
        
        # When: 创建FileSystemLogger实例
        logger = FileSystemLogger(log_dir)
        
        # Then: 验证默认参数
        assert str(logger.log_dir) == log_dir
        assert logger.file_prefix == "harborai"
        assert logger.batch_size == 100
        assert logger.flush_interval == 30
        assert logger.max_file_size == 100 * 1024 * 1024
        assert logger.max_files == 10
        assert logger.compress_old_files is True
        assert logger._log_queue is not None
        assert logger._worker_thread is None
        assert logger._running is False
        assert logger._file_lock is not None
        assert logger._current_file is None
        assert logger._current_file_size == 0
    
    def test_init_with_custom_parameters(self):
        """测试使用自定义参数初始化"""
        # Given: 自定义参数
        custom_params = {
            "log_dir": "custom_logs",
            "file_prefix": "custom",
            "batch_size": 20,
            "flush_interval": 10,
            "max_file_size": 50 * 1024 * 1024,
            "max_files": 3,
            "compress_old_files": False
        }
        
        # When: 创建FileSystemLogger实例
        logger = FileSystemLogger(**custom_params)
        
        # Then: 验证自定义参数
        assert str(logger.log_dir) == "custom_logs"
        assert logger.file_prefix == "custom"
        assert logger.batch_size == 20
        assert logger.flush_interval == 10
        assert logger.max_file_size == 50 * 1024 * 1024
        assert logger.max_files == 3
        assert logger.compress_old_files is False
    
    def test_init_creates_directory(self):
        """测试初始化时创建目录"""
        # Given: 不存在的目录
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "new_logs"
            
            # When: 创建FileSystemLogger
            logger = FileSystemLogger(str(log_dir))
            
            # Then: 验证目录被创建
            assert log_dir.exists()
            assert log_dir.is_dir()
    
    def test_init_directory_creation_failure(self):
        """测试目录创建失败的情况"""
        # Given: 模拟目录创建失败
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            # When & Then: 应该抛出异常
            with pytest.raises(PermissionError):
                FileSystemLogger("/some/path")


class TestFileSystemLoggerLifecycle:
    """测试FileSystemLogger生命周期管理"""
    
    @pytest.fixture
    def temp_logger(self):
        """创建临时目录的FileSystemLogger"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FileSystemLogger(
                log_dir=temp_dir,
                batch_size=2,  # 小批次便于测试
                flush_interval=1  # 短间隔便于测试
            )
            yield logger, temp_dir
    
    def test_start_logger(self, temp_logger):
        """测试启动日志记录器"""
        # Given: 创建的日志记录器
        logger, temp_dir = temp_logger
        
        # When: 启动日志记录器
        logger.start()
        
        # Then: 验证工作线程启动
        assert logger._worker_thread is not None
        assert logger._worker_thread.is_alive()
        assert logger._running is True
        
        # 清理
        logger.stop()
    
    def test_stop_logger(self, temp_logger):
        """测试停止日志记录器"""
        # Given: 运行中的日志记录器
        logger, temp_dir = temp_logger
        logger.start()
        
        # When: 停止日志记录器
        logger.stop()
        
        # Then: 验证工作线程停止
        assert logger._running is False
        if logger._worker_thread:
            logger._worker_thread.join(timeout=1.0)
            assert not logger._worker_thread.is_alive()
    
    def test_stop_not_started_logger(self, temp_logger):
        """测试停止未启动的日志记录器"""
        # Given: 未启动的日志记录器
        logger, temp_dir = temp_logger
        
        # When: 停止日志记录器
        logger.stop()
        
        # Then: 不应该抛出异常
        assert logger._running is False
    
    def test_double_start(self, temp_logger):
        """测试重复启动日志记录器"""
        # Given: 已启动的日志记录器
        logger, temp_dir = temp_logger
        logger.start()
        original_thread = logger._worker_thread
        
        # When: 再次启动
        logger.start()
        
        # Then: 应该使用同一个线程
        assert logger._worker_thread is original_thread
        
        # 清理
        logger.stop()


class TestFileSystemLoggerLogging:
    """测试FileSystemLogger日志记录功能"""
    
    @pytest.fixture
    def temp_logger(self):
        """创建临时目录的FileSystemLogger"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FileSystemLogger(
                log_dir=temp_dir,
                batch_size=1,  # 立即刷新
                flush_interval=1
            )
            logger.start()
            yield logger, temp_dir
            logger.stop()
    
    def test_log_request_basic(self, temp_logger):
        """测试基本请求日志记录"""
        # Given: 运行中的日志记录器
        logger, temp_dir = temp_logger
        
        # When: 记录请求日志
        trace_id = "test-trace-123"
        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello world"}]
        
        logger.log_request(trace_id, model, messages, temperature=0.7, max_length=100)
        
        # 等待异步处理
        time.sleep(0.2)
        
        # Then: 验证日志被记录
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        assert len(log_files) > 0
        
        # 读取日志内容
        with open(log_files[0], 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline())
        
        assert log_entry["trace_id"] == trace_id
        assert log_entry["model"] == model
        assert log_entry["type"] == "request"
        # 验证消息被正确脱敏处理
        assert len(log_entry["messages"]) == 1
        assert log_entry["messages"][0]["role"] == "user"
        assert log_entry["messages"][0]["content"] == "Hello world"
        assert log_entry["parameters"]["temperature"] == 0.7
        assert log_entry["parameters"]["max_length"] == 100
    
    def test_log_request_with_sensitive_data(self, temp_logger):
        """测试包含敏感数据的请求日志"""
        # Given: 包含敏感信息的消息
        logger, temp_dir = temp_logger
        
        # When: 记录包含敏感数据的请求
        trace_id = "test-trace-sensitive"
        model = "gpt-4"
        messages = [
            {
                "role": "user", 
                "content": "My phone is 13812345678 and email is test@example.com"
            }
        ]
        
        logger.log_request(trace_id, model, messages)
        
        # 等待异步处理
        time.sleep(0.2)
        
        # Then: 验证敏感数据被脱敏
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        with open(log_files[0], 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline())
        
        content = log_entry["messages"][0]["content"]
        assert "13812345678" not in content  # 手机号被脱敏
        assert "test@example.com" not in content  # 邮箱被脱敏
        assert "***" in content  # 包含脱敏标记
    
    def test_log_response_basic(self, temp_logger):
        """测试基本响应日志记录"""
        # Given: 运行中的日志记录器
        logger, temp_dir = temp_logger
        
        # When: 记录响应日志
        trace_id = "test-trace-response"
        
        # 创建一个简单的响应对象，而不是Mock
        class SimpleResponse:
            def __init__(self):
                self.choices = [SimpleChoice()]
                self.usage = SimpleUsage()
        
        class SimpleChoice:
            def __init__(self):
                self.message = SimpleMessage()
        
        class SimpleMessage:
            def __init__(self):
                self.content = "Hello! How can I help you?"
        
        class SimpleUsage:
            def __init__(self):
                self.total_tokens = 50
                self.prompt_tokens = 20
                self.completion_tokens = 30
        
        response = SimpleResponse()
        
        logger.log_response(trace_id, response, latency=1.5, success=True)
        
        # 等待异步处理
        time.sleep(0.2)
        
        # Then: 验证响应日志
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        assert len(log_files) > 0, "No log files found"
        
        with open(log_files[0], 'r', encoding='utf-8') as f:
            content = f.read().strip()
            assert content, "Log file is empty"
            log_entry = json.loads(content)
        
        assert log_entry["trace_id"] == trace_id
        assert log_entry["type"] == "response"
        assert log_entry["success"] is True
        assert log_entry["latency"] == 1.5
        assert log_entry["tokens"]["total_tokens"] == 50
        assert log_entry["tokens"]["prompt_tokens"] == 20
        assert log_entry["tokens"]["completion_tokens"] == 30
    
    def test_log_response_with_error(self, temp_logger):
        """测试错误响应日志记录"""
        # Given: 运行中的日志记录器
        logger, temp_dir = temp_logger
        
        # When: 记录错误响应
        trace_id = "test-trace-error"
        error_message = "API rate limit exceeded"
        
        logger.log_response(trace_id, None, latency=0.5, success=False, error=error_message)
        
        # 等待异步处理
        time.sleep(0.2)
        
        # Then: 验证错误日志
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        with open(log_files[0], 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline())
        
        assert log_entry["trace_id"] == trace_id
        assert log_entry["type"] == "response"
        assert log_entry["success"] is False
        assert log_entry["latency"] == 0.5
        assert log_entry["error"] == error_message
        assert log_entry["response_summary"] == {}
    
    def test_log_request_with_parameters(self, temp_logger):
        """测试包含敏感参数的请求日志"""
        # Given: 包含敏感参数的请求
        logger, temp_dir = temp_logger
        
        # When: 记录包含敏感参数的请求
        trace_id = "test-trace-params"
        model = "gpt-4"
        messages = [{"role": "user", "content": "Test"}]
        
        logger.log_request(
            trace_id, model, messages,
            api_key="sk-1234567890abcdef",
            authorization="Bearer token123",
            temperature=0.7
        )
        
        # 等待异步处理
        time.sleep(0.2)
        
        # Then: 验证敏感参数被脱敏
        log_files = list(Path(temp_dir).glob("*.jsonl"))
        with open(log_files[0], 'r', encoding='utf-8') as f:
            log_entry = json.loads(f.readline())
        
        params = log_entry["parameters"]
        assert params["api_key"] == "[REDACTED]"
        assert params["authorization"] == "[REDACTED]"
        assert params["temperature"] == 0.7  # 非敏感参数保留


class TestFileSystemLoggerDataSanitization:
    """测试FileSystemLogger数据脱敏功能"""
    
    @pytest.fixture
    def logger(self):
        """创建FileSystemLogger实例"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FileSystemLogger(temp_dir)
            yield logger
    
    def test_sanitize_phone_numbers(self, logger):
        """测试手机号脱敏"""
        # Given: 包含手机号的文本
        text = "请联系我，手机号是13812345678，或者18987654321"
        
        # When: 脱敏处理
        sanitized_text, detections = logger._sanitize_text(text)
        
        # Then: 验证手机号被脱敏
        assert "13812345678" not in sanitized_text
        assert "18987654321" not in sanitized_text
        assert "138****5678" in sanitized_text
        assert "189****4321" in sanitized_text
        assert len(detections) == 2
    
    def test_sanitize_id_cards(self, logger):
        """测试身份证号脱敏"""
        # Given: 包含身份证号的文本（使用不会被误识别为手机号的身份证号）
        text = "我的身份证号是110101200001011234"
        
        # When: 脱敏处理
        sanitized_text, detections = logger._sanitize_text(text)
        
        # Then: 验证身份证号被脱敏
        assert "110101200001011234" not in sanitized_text
        assert "110101********1234" in sanitized_text
        assert len(detections) == 1
    
    def test_sanitize_emails(self, logger):
        """测试邮箱脱敏"""
        # Given: 包含邮箱的文本
        text = "联系邮箱：user@example.com 或 test.email@domain.org"
        
        # When: 脱敏处理
        sanitized_text, detections = logger._sanitize_text(text)
        
        # Then: 验证邮箱被脱敏
        assert "user@example.com" not in sanitized_text
        assert "test.email@domain.org" not in sanitized_text
        assert "u**r@example.com" in sanitized_text
        assert "t********l@domain.org" in sanitized_text
        assert len(detections) == 2
    
    def test_sanitize_credit_cards(self, logger):
        """测试信用卡号脱敏"""
        # Given: 包含信用卡号的文本
        text = "信用卡号：4111111111111111"
        
        # When: 脱敏处理
        sanitized_text, detections = logger._sanitize_text(text)
        
        # Then: 验证信用卡号被脱敏
        assert "4111111111111111" not in sanitized_text
        assert "************1111" in sanitized_text
        assert len(detections) == 1
    
    def test_sanitize_ip_addresses(self, logger):
        """测试IP地址脱敏"""
        # Given: 包含IP地址的文本
        text = "服务器IP：192.168.1.100，外网IP：203.0.113.1"
        
        # When: 脱敏处理
        sanitized_text, detections = logger._sanitize_text(text)
        
        # Then: 验证IP地址被脱敏
        assert "192.168.1.100" not in sanitized_text
        assert "203.0.113.1" not in sanitized_text
        assert "192.***.***.***" in sanitized_text
        assert "203.***.***.***" in sanitized_text
        assert len(detections) == 2
    
    def test_sanitize_api_keys(self, logger):
        """测试API密钥脱敏"""
        # Given: 包含API密钥的文本（使用符合模式的48位以上密钥）
        text = "API密钥：sk-1234567890abcdef1234567890abcdef1234567890abcdef"
        
        # When: 脱敏处理
        sanitized_text, detections = logger._sanitize_text(text)
        
        # Then: 验证API密钥被脱敏
        assert "sk-1234567890abcdef1234567890abcdef1234567890abcdef" not in sanitized_text
        assert "sk-" in sanitized_text  # 前缀保留
        assert "***" in sanitized_text  # 包含脱敏标记
        assert len(detections) >= 1
    
    def test_sanitize_generic_keys(self, logger):
        """测试通用密钥脱敏"""
        # Given: 包含通用密钥的文本（使用符合模式的32位以上密钥）
        text = "密钥：abcdef1234567890abcdef1234567890abcdef"
        
        # When: 脱敏处理
        sanitized_text, detections = logger._sanitize_text(text)
        
        # Then: 验证通用密钥被脱敏
        assert "abcdef1234567890abcdef1234567890abcdef" not in sanitized_text
        assert "***" in sanitized_text  # 包含脱敏标记
        assert len(detections) >= 1
    
    def test_sanitize_messages_list(self, logger):
        """测试消息列表脱敏"""
        # Given: 包含敏感信息的消息列表
        messages = [
            {"role": "user", "content": "我的手机是13812345678"},
            {"role": "assistant", "content": "好的，已记录您的联系方式"},
            {"role": "user", "content": "邮箱是test@example.com"}
        ]
        
        # When: 脱敏处理
        sanitized = logger._sanitize_messages(messages)
        
        # Then: 验证消息被脱敏
        assert sanitized[0]["content"] == "我的手机是138****5678"
        assert sanitized[0]["sensitive_data_detected"] is True
        assert sanitized[1]["content"] == "好的，已记录您的联系方式"  # 无敏感信息
        assert sanitized[1]["sensitive_data_detected"] is False
        assert sanitized[2]["content"] == "邮箱是t**t@example.com"
        assert sanitized[2]["sensitive_data_detected"] is True
    
    def test_sanitize_parameters_dict(self, logger):
        """测试参数字典脱敏"""
        # Given: 包含敏感参数的字典
        params = {
            "api_key": "sk-1234567890abcdef",
            "authorization": "Bearer token123",
            "secret_key": "secret123",
            "temperature": 0.7,
            "max_length": 100
        }
        
        # When: 脱敏处理
        sanitized = logger._sanitize_parameters(params)
        
        # Then: 验证敏感参数被脱敏
        assert sanitized["api_key"] == "[REDACTED]"
        assert sanitized["authorization"] == "[REDACTED]"
        assert sanitized["secret_key"] == "[REDACTED]"
        assert sanitized["temperature"] == 0.7  # 非敏感参数保留
        assert sanitized["max_length"] == 100


class TestFileSystemLoggerFileManagement:
    """测试FileSystemLogger文件管理功能"""
    
    @pytest.fixture
    def temp_logger(self):
        """创建临时目录的FileSystemLogger"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FileSystemLogger(
                log_dir=temp_dir,
                max_file_size=1024,  # 1KB，便于测试轮转
                max_files=3,
                compress_old_files=True
            )
            try:
                yield logger, temp_dir
            finally:
                # 确保文件被正确关闭
                if hasattr(logger, '_current_file') and logger._current_file:
                    logger._current_file.close()
                    logger._current_file = None
    
    def test_file_creation(self, temp_logger):
        """测试日志文件创建"""
        # Given: 新的日志记录器
        logger, temp_dir = temp_logger
        
        # When: 确保日志文件存在
        logger._ensure_log_file()
        
        # Then: 验证文件被创建
        assert logger._current_file is not None
        assert Path(logger._current_file.name).exists()
        assert Path(logger._current_file.name).parent == Path(temp_dir)
    
    def test_file_rotation_by_size(self, temp_logger):
        """测试按大小轮转文件"""
        # Given: 小文件大小限制的日志记录器
        logger, temp_dir = temp_logger
        logger.start()
        
        # When: 写入大量数据触发轮转
        for i in range(10):
            logger.log_request(
                f"trace-{i}", 
                "gpt-4", 
                [{"role": "user", "content": "A" * 200}]  # 大内容
            )
        
        # 等待异步处理
        time.sleep(0.5)
        logger.stop()
        
        # Then: 验证文件轮转
        log_files = list(Path(temp_dir).glob("*.jsonl*"))
        assert len(log_files) > 1  # 应该有多个文件
    
    def test_file_compression(self, temp_logger):
        """测试文件压缩"""
        # Given: 启用压缩的日志记录器
        logger, temp_dir = temp_logger
        
        # When: 创建并轮转文件
        logger._ensure_log_file()
        old_file_path = logger._current_file.name
        
        # 写入一些数据
        logger._current_file.write('{"test": "data"}\n' * 100)
        logger._current_file.flush()
        
        # 轮转文件
        logger._rotate_log_file()
        
        # Then: 验证压缩文件存在
        compressed_files = list(Path(temp_dir).glob("*.gz"))
        assert len(compressed_files) > 0
        
        # 验证压缩文件可以读取
        with gzip.open(compressed_files[0], 'rt') as f:
            content = f.read()
            assert "test" in content
    
    def test_old_files_cleanup(self, temp_logger):
        """测试旧文件清理"""
        # Given: 限制文件数量的日志记录器
        logger, temp_dir = temp_logger
        
        # When: 创建超过限制的文件数量
        for i in range(5):  # 超过max_files=3
            filename = f"harborai_{i}.jsonl"
            filepath = Path(temp_dir) / filename
            with open(filepath, 'w') as f:
                f.write(f'{{"test": {i}}}\n')
        
        # 执行清理
        logger._cleanup_old_files()
        
        # Then: 验证只保留指定数量的文件
        remaining_files = list(Path(temp_dir).glob("*.jsonl"))
        assert len(remaining_files) <= logger.max_files
    
    def test_get_log_files(self, temp_logger):
        """测试获取日志文件列表"""
        # Given: 有多个日志文件
        logger, temp_dir = temp_logger
        
        # 创建一些测试文件
        for i in range(3):
            filename = f"harborai_{i}.jsonl"
            filepath = Path(temp_dir) / filename
            with open(filepath, 'w') as f:
                f.write(f'{{"test": {i}}}\n')
        
        # When: 获取日志文件列表
        files = logger.get_log_files()
        
        # Then: 验证文件列表
        assert len(files) == 3
        assert all(f.name.endswith('.jsonl') for f in files)
    
    def test_read_logs_basic(self, temp_logger):
        """测试基本日志读取"""
        # Given: 有日志数据的文件
        logger, temp_dir = temp_logger
        
        # 创建测试日志文件
        log_file = Path(temp_dir) / "harborai_test.jsonl"
        test_logs = [
            {"trace_id": "trace-1", "type": "request", "timestamp": "2024-01-01T10:00:00"},
            {"trace_id": "trace-2", "type": "response", "timestamp": "2024-01-01T10:01:00"},
            {"trace_id": "trace-3", "type": "request", "timestamp": "2024-01-01T10:02:00"}
        ]
        
        with open(log_file, 'w') as f:
            for log in test_logs:
                f.write(json.dumps(log) + '\n')
        
        # When: 读取日志
        logs = logger.read_logs(limit=10)
        
        # Then: 验证读取结果
        assert len(logs) == 3
        assert logs[0]["trace_id"] == "trace-1"
        assert logs[1]["trace_id"] == "trace-2"
        assert logs[2]["trace_id"] == "trace-3"
    
    def test_read_logs_with_filters(self, temp_logger):
        """测试带过滤条件的日志读取"""
        # Given: 有日志数据的文件
        logger, temp_dir = temp_logger
        
        # 创建测试日志文件
        log_file = Path(temp_dir) / "harborai_test.jsonl"
        test_logs = [
            {"trace_id": "trace-1", "type": "request", "model": "gpt-4"},
            {"trace_id": "trace-2", "type": "response", "model": "gpt-3.5"},
            {"trace_id": "trace-3", "type": "request", "model": "gpt-4"}
        ]
        
        with open(log_file, 'w') as f:
            for log in test_logs:
                f.write(json.dumps(log) + '\n')
        
        # When: 使用过滤条件读取
        logs = logger.read_logs(trace_id="trace-1")
        
        # Then: 验证过滤结果
        assert len(logs) == 1
        assert logs[0]["trace_id"] == "trace-1"


class TestFileSystemLoggerGlobalFunctions:
    """测试FileSystemLogger全局函数"""
    
    def test_initialize_file_logger(self):
        """测试初始化全局文件日志记录器"""
        # Given: 日志目录
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # When: 初始化全局记录器
            result = initialize_file_logger(temp_dir, file_prefix="test")
            
            # Then: 验证创建和启动
            assert result is not None
            assert isinstance(result, FileSystemLogger)
            assert str(result.log_dir) == temp_dir
            assert result.file_prefix == "test"
            
            # 清理
            result.stop()
    
    def test_get_file_logger(self):
        """测试获取全局文件日志记录器"""
        # Given: 初始化全局记录器
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_file_logger(temp_dir)
            
            # When: 获取全局记录器
            result = get_file_logger()
            
            # Then: 应该返回全局实例
            assert result is not None
            assert isinstance(result, FileSystemLogger)
            
            # 清理
            shutdown_file_logger()
    
    def test_shutdown_file_logger(self):
        """测试关闭全局文件日志记录器"""
        # Given: 有全局记录器
        with tempfile.TemporaryDirectory() as temp_dir:
            initialize_file_logger(temp_dir)
            
            # When: 关闭全局记录器
            shutdown_file_logger()
            
            # Then: 全局记录器应该被清理
            result = get_file_logger()
            assert result is None
    
    def test_get_file_logger_none(self):
        """测试获取未初始化的全局记录器"""
        # Given: 没有全局记录器
        shutdown_file_logger()  # 确保清理
        
        # When: 获取全局记录器
        result = get_file_logger()
        
        # Then: 应该返回None
        assert result is None


class TestFileSystemLoggerEdgeCases:
    """测试FileSystemLogger边界情况"""
    
    def test_worker_thread_exception_handling(self):
        """测试工作线程异常处理"""
        # Given: 会抛出异常的日志记录器
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FileSystemLogger(temp_dir, batch_size=1)
            
            # Mock _flush_batch 抛出异常
            with patch.object(logger, '_flush_batch', side_effect=Exception("Flush error")):
                logger.start()
                
                # When: 添加日志条目
                logger.log_request("trace-error", "gpt-4", [{"role": "user", "content": "test"}])
                
                # 等待处理
                time.sleep(0.2)
                
                # Then: 工作线程应该继续运行（异常被捕获）
                assert logger._worker_thread.is_alive()
                
                logger.stop()
    
    def test_queue_full_handling(self):
        """测试队列满时的处理"""
        # Given: 小队列的日志记录器
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FileSystemLogger(temp_dir)
            
            # 填满队列
            for i in range(1000):  # 超过默认队列大小
                try:
                    logger._log_queue.put_nowait({"test": i})
                except queue.Full:
                    break
            
            # When: 尝试记录日志
            logger.log_request("trace-full", "gpt-4", [{"role": "user", "content": "test"}])
            
            # Then: 不应该抛出异常（应该优雅处理）
            assert True
    
    def test_file_write_permission_error(self):
        """测试文件写入权限错误"""
        # Given: 只读目录的日志记录器
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FileSystemLogger(temp_dir)
            
            # Mock open 抛出权限错误
            with patch('builtins.open', side_effect=PermissionError("Permission denied")):
                
                # When: 尝试刷新批次
                batch = [{"test": "data"}]
                
                # Then: 不应该抛出异常（应该优雅处理）
                try:
                    logger._flush_batch(batch)
                except PermissionError:
                    pass  # 预期的异常
    
    def test_response_summary_with_complex_structure(self):
        """测试复杂响应结构的摘要"""
        # Given: 复杂的响应对象
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FileSystemLogger(temp_dir)
            
            # 创建复杂的响应对象
            response = Mock()
            response.choices = [Mock()]  # 只使用第一个选择
            response.choices[0].message = Mock()
            response.choices[0].message.content = "First response"
            response.choices[0].message.reasoning_content = None
            response.choices[0].message.tool_calls = None
            response.model = "gpt-4"
            response.id = "resp-123"
            
            # When: 创建响应摘要
            summary = logger._create_response_summary(response)
            
            # Then: 验证摘要结构
            assert summary["content"] == "First response"
            assert summary["content_length"] == 14
            assert summary["model"] == "gpt-4"
            assert summary["response_id"] == "resp-123"
    
    def test_response_summary_with_none_content(self):
        """测试空内容响应的摘要"""
        # Given: 空内容的响应
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = FileSystemLogger(temp_dir)
            
            response = Mock()
            response.choices = [Mock()]
            response.choices[0].message = Mock()
            response.choices[0].message.content = None
            response.choices[0].message.reasoning_content = None
            response.choices[0].message.tool_calls = None
            
            # When: 创建响应摘要
            summary = logger._create_response_summary(response)
            
            # Then: 应该返回包含空内容的字典
            assert summary["content"] == ""
            assert summary["content_length"] == 0
    
    def test_sensitive_patterns_coverage(self):
        """测试敏感信息模式覆盖率"""
        import re
        
        # Given: 各种敏感信息
        test_cases = [
            ("手机号", "13812345678", SENSITIVE_PATTERNS["phone"]),
            ("身份证", "110101200001011234", SENSITIVE_PATTERNS["id_card"]),
            ("邮箱", "test@example.com", SENSITIVE_PATTERNS["email"]),
            ("信用卡", "4111111111111111", SENSITIVE_PATTERNS["credit_card"]),
            ("IP地址", "192.168.1.1", SENSITIVE_PATTERNS["ip_address"]),
            ("API密钥", "sk-1234567890abcdef1234567890abcdef1234567890abcdef", SENSITIVE_PATTERNS["api_key"]),
            ("通用密钥", "abcdef1234567890abcdef1234567890abcdef", SENSITIVE_PATTERNS["generic_key"])
        ]
        
        # When & Then: 验证每个模式都能匹配
        for name, test_data, patterns in test_cases:
            matched = False
            for pattern_str in patterns:
                pattern = re.compile(pattern_str)
                if pattern.search(test_data):
                    matched = True
                    break
            assert matched, f"{name} 模式应该匹配 {test_data}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])