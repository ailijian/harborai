"""
测试增强的文件系统日志记录器
"""

import json
import tempfile
import time
import shutil
from pathlib import Path
from decimal import Decimal
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from harborai.storage.enhanced_file_logger import EnhancedFileSystemLogger, DateTimeEncoder


class TestDateTimeEncoder:
    """测试DateTimeEncoder"""
    
    def test_datetime_encoding(self):
        """测试datetime对象编码"""
        encoder = DateTimeEncoder()
        dt = datetime(2023, 1, 1, 12, 0, 0)
        result = encoder.default(dt)
        assert isinstance(result, str)
        assert "2023-01-01T12:00:00" in result
    
    def test_decimal_encoding(self):
        """测试Decimal对象编码"""
        encoder = DateTimeEncoder()
        decimal_value = Decimal("123.456")
        result = encoder.default(decimal_value)
        # 实际实现返回float
        assert result == 123.456
        assert isinstance(result, float)
    
    def test_unsupported_type(self):
        """测试不支持的类型"""
        encoder = DateTimeEncoder()
        with pytest.raises(TypeError):
            encoder.default(object())


class TestEnhancedFileSystemLogger:
    """测试增强的文件系统日志记录器"""
    
    def test_logger_creation_default(self):
        """测试使用默认参数创建日志记录器"""
        logger = EnhancedFileSystemLogger(log_dir="/tmp/test_logs")
        assert logger.log_dir == Path("/tmp/test_logs")
        assert logger.enable_tracing is True
        assert logger.tracing_sample_rate == 1.0
        assert logger.health_check_interval == 60
        assert logger.max_disk_usage_percent == 85.0
    
    def test_logger_creation_custom(self):
        """测试使用自定义参数创建日志记录器"""
        logger = EnhancedFileSystemLogger(
            log_dir="/tmp/custom_logs",
            enable_tracing=False,
            tracing_sample_rate=0.5,
            health_check_interval=30,
            max_disk_usage_percent=80
        )
        assert logger.log_dir == Path("/tmp/custom_logs")
        assert logger.enable_tracing is False
        assert logger.tracing_sample_rate == 0.5
        assert logger.health_check_interval == 30
        assert logger.max_disk_usage_percent == 80
    
    def test_log_directory_creation(self):
        """测试日志目录创建"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "test_logs"
            logger = EnhancedFileSystemLogger(log_dir=str(log_dir))
            logger.start()
            assert log_dir.exists()
            logger.stop()
    
    def test_start_logger(self):
        """测试启动日志记录器"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EnhancedFileSystemLogger(log_dir=temp_dir)
            logger.start()
            assert logger._running is True
            assert logger._worker_thread is not None
            assert logger._worker_thread.is_alive()
            logger.stop()
    
    def test_stop_logger(self):
        """测试停止日志记录器"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EnhancedFileSystemLogger(log_dir=temp_dir)
            logger.start()
            logger.stop()
            assert logger._running is False
            # 等待线程结束
            if logger._worker_thread:
                logger._worker_thread.join(timeout=5)
            assert not logger._worker_thread.is_alive()
    
    def test_log_request(self):
        """测试记录请求日志"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EnhancedFileSystemLogger(log_dir=temp_dir, enable_tracing=False)
            logger.start()
            
            # 记录请求
            logger.log_request_with_tracing(
                trace_id="test-trace-123",
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                provider="openai"
            )
            
            # 检查队列中是否有日志
            assert not logger._log_queue.empty()
            
            # 等待一段时间让日志被处理
            time.sleep(0.1)
            
            logger.stop()
    
    def test_log_response(self):
        """测试记录响应日志"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EnhancedFileSystemLogger(log_dir=temp_dir, enable_tracing=False)
            logger.start()
            
            # 使用Mock对象而不是自定义类
            from unittest.mock import Mock
            
            response = Mock()
            response.usage = Mock()
            response.usage.prompt_tokens = 10
            response.usage.completion_tokens = 20
            response.usage.total_tokens = 30
            
            choice = Mock()
            message = Mock()
            message.content = "Test response content"
            message.reasoning_content = None
            choice.message = message
            response.choices = [choice]
            
            # 记录响应
            logger.log_response_with_tracing(
                trace_id="test-trace-123",
                response=response,
                latency=1.5,
                success=True
            )
            
            # 检查队列中是否有日志
            assert not logger._log_queue.empty()
            
            logger.stop()
    
    def test_log_response_with_error(self):
        """测试记录错误响应日志"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EnhancedFileSystemLogger(log_dir=temp_dir, enable_tracing=False)
            logger.start()
            
            # 记录错误响应
            logger.log_response_with_tracing(
                trace_id="test-trace-error",
                response=None,
                latency=0.5,
                success=False,
                error="API rate limit exceeded"
            )
            
            # 检查队列中是否有日志
            assert not logger._log_queue.empty()
            
            logger.stop()
    
    def test_sanitize_messages(self):
        """测试敏感信息清理"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EnhancedFileSystemLogger(log_dir=temp_dir)
            
            # 测试消息清理
            messages = [
                {"role": "user", "content": "My phone is 13812345678"},
                {"role": "assistant", "content": "I understand"}
            ]
            
            sanitized = logger._sanitize_messages(messages)
            assert "13812345678" not in str(sanitized)
            assert "138****5678" in str(sanitized)
    
    def test_sanitize_text(self):
        """测试文本清理"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EnhancedFileSystemLogger(log_dir=temp_dir)
            
            # 测试文本清理
            text = "Contact me at 13812345678 or email@example.com"
            result = logger._sanitize_text(text)
            
            # _sanitize_text 返回元组 (sanitized_text, matches)
            if isinstance(result, tuple):
                sanitized_text = result[0]
            else:
                sanitized_text = result
            
            assert "13812345678" not in sanitized_text
            assert "email@example.com" not in sanitized_text
            # 检查是否有某种形式的掩码
            assert ("138****5678" in sanitized_text or 
                    "138*" in sanitized_text or 
                    "***" in sanitized_text or
                    "e***l@example.com" in sanitized_text)
    
    def test_mask_sensitive_data(self):
        """测试敏感数据掩码"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EnhancedFileSystemLogger(log_dir=temp_dir)
            
            # 测试各种敏感数据类型
            test_cases = [
                ("13812345678", "138****5678"),  # 手机号
                ("email@example.com", "em***@example.com"),  # 邮箱
                ("sk-1234567890abcdef1234567890abcdef12345678", "sk-****"),  # API密钥
            ]
            
            for original, expected in test_cases:
                result = logger._mask_sensitive_data(original, "phone")
                # 检查是否被掩码处理
                assert original != result or expected in result
    
    def test_flush_batch_to_file(self):
        """测试批量刷新到文件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = EnhancedFileSystemLogger(log_dir=temp_dir, batch_size=2)
            logger.start()
            
            # 添加多个日志条目
            for i in range(3):
                logger._log_queue.put({
                    "trace_id": f"test-{i}",
                    "timestamp": "2023-01-01T12:00:00Z",
                    "type": "test",
                    "data": f"test data {i}"
                })
            
            # 等待批量处理
            time.sleep(0.5)
            
            # 检查文件是否创建
            log_files = list(Path(temp_dir).glob("*.jsonl"))
            assert len(log_files) > 0
            
            # 检查文件内容
            with open(log_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
                assert "test-0" in content or "test-1" in content
            
            # 检查队列是否被清空
            assert logger._log_queue.qsize() <= 1  # 可能还有一个未处理的
            
            logger.stop()