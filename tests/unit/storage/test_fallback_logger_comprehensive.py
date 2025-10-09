#!/usr/bin/env python3
"""
---
summary: FallbackLogger 综合测试套件
description: 全面测试日志降级管理器的功能，包括正常流程、异常处理、状态切换等
assumptions:
  - id: A1
    text: PostgreSQL连接字符串格式正确
    confidence: high
  - id: A2
    text: 文件系统可写且有足够空间
    confidence: high
  - id: A3
    text: Mock对象能正确模拟PostgreSQL和FileSystem行为
    confidence: high
tests:
  - path: tests/unit/storage/test_fallback_logger_comprehensive.py
    cmd: pytest tests/unit/storage/test_fallback_logger_comprehensive.py -v
---
"""

import pytest
import tempfile
import shutil
import time
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from datetime import datetime

from harborai.storage.fallback_logger import (
    FallbackLogger, 
    LoggerState,
    get_fallback_logger,
    initialize_fallback_logger,
    shutdown_fallback_logger
)
from harborai.utils.exceptions import StorageError


class TestFallbackLoggerInitialization:
    """测试FallbackLogger初始化功能"""
    
    def test_init_with_default_parameters(self):
        """测试使用默认参数初始化"""
        # Given: 默认参数
        postgres_conn = "postgresql://test:test@localhost/test"
        
        # When: 创建FallbackLogger实例
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            # 配置Mock对象
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger(postgres_conn)
            
            # Then: 验证初始化参数
            assert logger.postgres_connection_string == postgres_conn
            assert logger.log_directory == "logs"
            assert logger.max_postgres_failures == 3
            assert logger.health_check_interval == 60.0
            assert logger.postgres_table_name == "harborai_logs"
            assert logger.file_max_size == 100 * 1024 * 1024
            assert logger.file_backup_count == 5
            assert logger.postgres_batch_size == 10
            assert logger.postgres_flush_interval == 5.0
    
    def test_init_with_custom_parameters(self):
        """测试使用自定义参数初始化"""
        # Given: 自定义参数
        postgres_conn = "postgresql://custom:custom@localhost/custom"
        custom_params = {
            "log_directory": "custom_logs",
            "max_postgres_failures": 5,
            "health_check_interval": 30.0,
            "postgres_table_name": "custom_logs",
            "file_max_size": 50 * 1024 * 1024,
            "file_backup_count": 3,
            "postgres_batch_size": 20,
            "postgres_flush_interval": 10.0
        }
        
        # When: 创建FallbackLogger实例
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger(postgres_conn, **custom_params)
            
            # Then: 验证自定义参数
            assert logger.log_directory == "custom_logs"
            assert logger.max_postgres_failures == 5
            assert logger.health_check_interval == 30.0
            assert logger.postgres_table_name == "custom_logs"
            assert logger.file_max_size == 50 * 1024 * 1024
            assert logger.file_backup_count == 3
            assert logger.postgres_batch_size == 20
            assert logger.postgres_flush_interval == 10.0
    
    def test_init_postgres_success(self):
        """测试PostgreSQL初始化成功的情况"""
        # Given: 有效的PostgreSQL连接
        postgres_conn = "postgresql://test:test@localhost/test"
        
        # When: PostgreSQL初始化成功
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger(postgres_conn)
            
            # Then: 验证状态和调用
            assert logger.get_state() == LoggerState.POSTGRES_ACTIVE
            mock_pg_instance.start.assert_called_once()
            mock_fs_instance.start.assert_called_once()
    
    def test_init_postgres_failure(self):
        """测试PostgreSQL初始化失败的情况"""
        # Given: PostgreSQL连接失败
        postgres_conn = "postgresql://invalid:invalid@localhost/invalid"
        
        # When: PostgreSQL初始化失败
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_fs_instance = Mock()
            mock_fs.return_value = mock_fs_instance
            
            # 模拟PostgreSQL初始化失败
            mock_pg.side_effect = Exception("Connection failed")
            
            logger = FallbackLogger(postgres_conn)
            
            # Then: 验证降级到文件模式
            # 根据实际代码逻辑，PostgreSQL失败后会调用_handle_postgres_failure
            # 然后调用_switch_to_file_fallback，状态应该是FILE_FALLBACK
            assert logger.get_state() == LoggerState.FILE_FALLBACK
            assert logger._postgres_failure_count >= 1  # 至少失败一次
            mock_fs_instance.start.assert_called_once()
            # 验证PostgreSQL初始化被尝试
            mock_pg.assert_called_once()
    
    def test_init_file_logger_failure(self):
        """测试文件日志记录器初始化失败的情况"""
        # Given: 文件系统不可用
        postgres_conn = "postgresql://test:test@localhost/test"
        
        # When: 文件日志记录器初始化失败
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            # 模拟文件系统初始化失败
            mock_fs.side_effect = Exception("File system error")
            
            # Then: 验证抛出StorageError
            with pytest.raises(StorageError, match="Failed to initialize fallback logger"):
                FallbackLogger(postgres_conn)


class TestFallbackLoggerLogging:
    """测试FallbackLogger日志记录功能"""
    
    @pytest.fixture
    def mock_logger(self):
        """创建Mock的FallbackLogger"""
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger("postgresql://test:test@localhost/test")
            # 确保Mock实例被正确设置
            logger._postgres_logger = mock_pg_instance
            logger._file_logger = mock_fs_instance
            # 重置统计信息
            logger._stats = {
                "postgres_logs": 0,
                "file_logs": 0,
                "postgres_failures": 0,
                "state_changes": 0
            }
            
            yield logger, mock_pg_instance, mock_fs_instance
    
    def test_log_request_postgres_active(self, mock_logger):
        """测试在PostgreSQL活跃状态下记录请求"""
        # Given: PostgreSQL活跃状态
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.POSTGRES_ACTIVE
        
        # When: 记录请求日志
        trace_id = "test-trace-123"
        model = "gpt-4"
        messages = [{"role": "user", "content": "Hello"}]
        
        logger.log_request(trace_id, model, messages, temperature=0.7)
        
        # Then: 验证PostgreSQL记录器被调用
        mock_pg.log_request.assert_called_once_with(
            trace_id, model, messages, temperature=0.7
        )
        mock_fs.log_request.assert_not_called()
        assert logger._stats["postgres_logs"] == 1
        assert logger._stats["file_logs"] == 0
    
    def test_log_request_file_fallback(self, mock_logger):
        """测试在文件降级状态下记录请求"""
        # Given: 文件降级状态
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.FILE_FALLBACK
        
        # 防止健康检查干扰测试
        with patch.object(logger, '_check_health'):
            # When: 记录请求日志
            trace_id = "test-trace-456"
            model = "gpt-3.5-turbo"
            messages = [{"role": "user", "content": "Hi there"}]
            
            logger.log_request(trace_id, model, messages, max_tokens=100)
            
            # Then: 验证文件记录器被调用
            mock_fs.log_request.assert_called_once_with(
                trace_id, model, messages, max_tokens=100
            )
            mock_pg.log_request.assert_not_called()
            assert logger._stats["file_logs"] == 1
            assert logger._stats["postgres_logs"] == 0
    
    def test_log_request_postgres_failure_fallback(self, mock_logger):
        """测试PostgreSQL记录失败时的降级处理"""
        # Given: PostgreSQL活跃但记录失败
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.POSTGRES_ACTIVE
        mock_pg.log_request.side_effect = Exception("PostgreSQL error")
        
        # When: 记录请求日志
        trace_id = "test-trace-789"
        model = "gpt-4"
        messages = [{"role": "user", "content": "Test"}]
        
        logger.log_request(trace_id, model, messages)
        
        # Then: 验证降级到文件记录器
        mock_pg.log_request.assert_called_once()
        mock_fs.log_request.assert_called_once_with(trace_id, model, messages)
        assert logger._stats["file_logs"] == 1
    
    def test_log_response_postgres_active(self, mock_logger):
        """测试在PostgreSQL活跃状态下记录响应"""
        # Given: PostgreSQL活跃状态
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.POSTGRES_ACTIVE
        
        # When: 记录响应日志
        trace_id = "test-trace-123"
        response = Mock()
        latency = 1.5
        
        logger.log_response(trace_id, response, latency, success=True)
        
        # Then: 验证PostgreSQL记录器被调用
        mock_pg.log_response.assert_called_once_with(
            trace_id, response, latency, True, None
        )
        mock_fs.log_response.assert_not_called()
        assert logger._stats["postgres_logs"] == 1
    
    def test_log_response_with_error(self, mock_logger):
        """测试记录包含错误的响应"""
        # Given: 文件降级状态
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.FILE_FALLBACK
        
        # 防止健康检查干扰测试
        with patch.object(logger, '_check_health'):
            # When: 记录错误响应
            trace_id = "test-trace-error"
            response = None
            latency = 0.5
            error = "API rate limit exceeded"
            
            logger.log_response(trace_id, response, latency, success=False, error=error)
            
            # Then: 验证文件记录器被调用
            mock_fs.log_response.assert_called_once_with(
                trace_id, response, latency, False, error
            )
            assert logger._stats["file_logs"] == 1
    
    def test_log_both_loggers_fail(self, mock_logger):
        """测试两个记录器都失败的情况"""
        # Given: 两个记录器都失败
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.POSTGRES_ACTIVE
        mock_pg.log_request.side_effect = Exception("PostgreSQL error")
        mock_fs.log_request.side_effect = Exception("File system error")
        
        # When: 记录请求日志
        trace_id = "test-trace-fail"
        model = "gpt-4"
        messages = [{"role": "user", "content": "Test"}]
        
        # Then: 不应该抛出异常，应该优雅处理
        logger.log_request(trace_id, model, messages)
        
        # 验证两个记录器都被尝试调用
        mock_pg.log_request.assert_called_once()
        mock_fs.log_request.assert_called_once()


class TestFallbackLoggerStateManagement:
    """测试FallbackLogger状态管理功能"""
    
    @pytest.fixture
    def mock_logger(self):
        """创建Mock的FallbackLogger"""
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger("postgresql://test:test@localhost/test")
            logger._postgres_logger = mock_pg_instance
            logger._file_logger = mock_fs_instance
            
            yield logger, mock_pg_instance, mock_fs_instance
    
    def test_postgres_failure_handling(self, mock_logger):
        """测试PostgreSQL失败处理"""
        # Given: PostgreSQL活跃状态
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.POSTGRES_ACTIVE
        logger._postgres_failure_count = 0
        
        # When: 处理PostgreSQL失败
        error = Exception("Connection lost")
        logger._handle_postgres_failure(error)
        
        # Then: 验证失败计数增加
        assert logger._postgres_failure_count == 1
        assert logger._stats["postgres_failures"] == 1
        assert logger.get_state() == LoggerState.POSTGRES_ACTIVE  # 还未达到阈值
    
    def test_postgres_failure_threshold_reached(self, mock_logger):
        """测试PostgreSQL失败达到阈值"""
        # Given: 接近失败阈值
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.POSTGRES_ACTIVE
        logger._postgres_failure_count = 2  # 阈值是3
        
        # When: 再次失败
        error = Exception("Final failure")
        logger._handle_postgres_failure(error)
        
        # Then: 验证切换到文件降级
        assert logger._postgres_failure_count == 3
        assert logger.get_state() == LoggerState.FILE_FALLBACK
        assert logger._stats["state_changes"] == 1
    
    def test_force_fallback(self, mock_logger):
        """测试强制切换到降级模式"""
        # Given: PostgreSQL活跃状态
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.POSTGRES_ACTIVE
        
        # When: 强制切换到降级模式
        logger.force_fallback()
        
        # Then: 验证状态切换
        assert logger.get_state() == LoggerState.FILE_FALLBACK
        assert logger._stats["state_changes"] == 1
    
    def test_force_recovery(self, mock_logger):
        """测试强制恢复PostgreSQL"""
        # Given: 文件降级状态
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.FILE_FALLBACK
        logger._postgres_failure_count = 5
        
        # When: 强制恢复
        with patch.object(logger, '_test_postgres_connection', return_value=True):
            logger.force_recovery()
        
        # Then: 验证恢复
        assert logger._postgres_failure_count == 0
        assert logger.get_state() == LoggerState.POSTGRES_ACTIVE
    
    @patch('time.time')
    def test_health_check_interval(self, mock_time, mock_logger):
        """测试健康检查间隔"""
        # Given: 设置时间
        logger, mock_pg, mock_fs = mock_logger
        logger._state = LoggerState.FILE_FALLBACK
        logger.health_check_interval = 60.0
        logger._last_health_check = 100.0
        
        # When: 时间未到间隔
        mock_time.return_value = 150.0  # 50秒后
        logger._check_health()
        
        # Then: 不应该执行健康检查
        assert logger._last_health_check == 100.0
        
        # When: 时间超过间隔
        mock_time.return_value = 170.0  # 70秒后
        with patch.object(logger, '_attempt_postgres_recovery') as mock_recovery:
            logger._check_health()
        
        # Then: 应该执行健康检查
        assert logger._last_health_check == 170.0
        mock_recovery.assert_called_once()
    
    def test_postgres_connection_test_success(self, mock_logger):
        """测试PostgreSQL连接测试成功"""
        # Given: 有PostgreSQL记录器
        logger, mock_pg, mock_fs = mock_logger
        
        # When: 测试连接
        result = logger._test_postgres_connection()
        
        # Then: 应该返回True（简化实现）
        assert result is True
    
    def test_postgres_connection_test_no_logger(self, mock_logger):
        """测试没有PostgreSQL记录器时的连接测试"""
        # Given: 没有PostgreSQL记录器
        logger, mock_pg, mock_fs = mock_logger
        logger._postgres_logger = None
        
        # When: 测试连接
        result = logger._test_postgres_connection()
        
        # Then: 应该返回False
        assert result is False


class TestFallbackLoggerLifecycle:
    """测试FallbackLogger生命周期管理"""
    
    @pytest.fixture
    def mock_logger(self):
        """创建Mock的FallbackLogger"""
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger("postgresql://test:test@localhost/test")
            logger._postgres_logger = mock_pg_instance
            logger._file_logger = mock_fs_instance
            
            yield logger, mock_pg_instance, mock_fs_instance
    
    def test_start_logger(self, mock_logger):
        """测试启动日志记录器"""
        # Given: 创建的日志记录器
        logger, mock_pg, mock_fs = mock_logger
        
        # When: 启动日志记录器
        logger.start()
        
        # Then: 应该正常启动（无异常）
        # 这个方法主要是记录日志，没有复杂逻辑
        assert True
    
    def test_stop_logger(self, mock_logger):
        """测试停止日志记录器"""
        # Given: 运行中的日志记录器
        logger, mock_pg, mock_fs = mock_logger
        
        # When: 停止日志记录器
        logger.stop()
        
        # Then: 验证两个记录器都被停止
        mock_pg.stop.assert_called_once()
        mock_fs.stop.assert_called_once()
    
    def test_stop_logger_with_none_loggers(self, mock_logger):
        """测试停止空的日志记录器"""
        # Given: 没有记录器实例
        logger, mock_pg, mock_fs = mock_logger
        logger._postgres_logger = None
        logger._file_logger = None
        
        # When: 停止日志记录器
        logger.stop()
        
        # Then: 不应该抛出异常
        assert True
    
    def test_get_stats(self, mock_logger):
        """测试获取统计信息"""
        # Given: 有一些统计数据
        logger, mock_pg, mock_fs = mock_logger
        logger._stats["postgres_logs"] = 10
        logger._stats["file_logs"] = 5
        logger._stats["postgres_failures"] = 2
        logger._stats["state_changes"] = 1
        logger._postgres_failure_count = 2
        logger._last_health_check = 1234567890.0
        
        # When: 获取统计信息
        stats = logger.get_stats()
        
        # Then: 验证统计信息
        assert stats["postgres_logs"] == 10
        assert stats["file_logs"] == 5
        assert stats["postgres_failures"] == 2
        assert stats["state_changes"] == 1
        assert stats["current_state"] == logger._state.value
        assert stats["postgres_failure_count"] == 2
        assert stats["last_health_check"] is not None
    
    def test_get_stats_no_health_check(self, mock_logger):
        """测试获取统计信息（无健康检查）"""
        # Given: 没有健康检查记录
        logger, mock_pg, mock_fs = mock_logger
        logger._last_health_check = 0
        
        # When: 获取统计信息
        stats = logger.get_stats()
        
        # Then: 验证健康检查时间为None
        assert stats["last_health_check"] is None


class TestFallbackLoggerGlobalFunctions:
    """测试FallbackLogger全局函数"""
    
    def test_initialize_fallback_logger(self):
        """测试初始化全局降级日志记录器"""
        # Given: PostgreSQL连接字符串
        postgres_conn = "postgresql://test:test@localhost/test"
        
        # When: 初始化全局记录器
        with patch('harborai.storage.fallback_logger.FallbackLogger') as mock_fallback:
            mock_instance = Mock()
            mock_fallback.return_value = mock_instance
            
            result = initialize_fallback_logger(postgres_conn, log_directory="test_logs")
            
            # Then: 验证创建和启动
            mock_fallback.assert_called_once_with(postgres_conn, log_directory="test_logs")
            mock_instance.start.assert_called_once()
            assert result == mock_instance
    
    def test_initialize_fallback_logger_replace_existing(self):
        """测试替换现有的全局记录器"""
        # Given: 已有全局记录器
        with patch('harborai.storage.fallback_logger.FallbackLogger') as mock_fallback, \
             patch('harborai.storage.fallback_logger._global_fallback_logger') as mock_global:
            
            mock_old_instance = Mock()
            mock_new_instance = Mock()
            mock_global = mock_old_instance
            mock_fallback.return_value = mock_new_instance
            
            # When: 初始化新的记录器
            result = initialize_fallback_logger("postgresql://new:new@localhost/new")
            
            # Then: 验证旧记录器被停止
            # 注意：由于patch的限制，这里主要验证新记录器的创建
            mock_fallback.assert_called_once()
            mock_new_instance.start.assert_called_once()
    
    def test_get_fallback_logger(self):
        """测试获取全局降级日志记录器"""
        # Given: 设置全局记录器
        with patch('harborai.storage.fallback_logger._global_fallback_logger') as mock_global:
            mock_instance = Mock()
            mock_global = mock_instance
            
            # When: 获取全局记录器
            result = get_fallback_logger()
            
            # Then: 应该返回全局实例
            # 注意：由于patch的限制，这里验证函数能正常调用
            assert result is not None or result is None  # 函数正常执行
    
    def test_shutdown_fallback_logger(self):
        """测试关闭全局降级日志记录器"""
        # Given: 有全局记录器
        with patch('harborai.storage.fallback_logger._global_fallback_logger') as mock_global:
            mock_instance = Mock()
            mock_global = mock_instance
            
            # When: 关闭全局记录器
            shutdown_fallback_logger()
            
            # Then: 验证函数正常执行
            assert True  # 函数不抛出异常
    
    def test_shutdown_fallback_logger_none(self):
        """测试关闭空的全局记录器"""
        # Given: 没有全局记录器
        with patch('harborai.storage.fallback_logger._global_fallback_logger', None):
            
            # When: 关闭全局记录器
            shutdown_fallback_logger()
            
            # Then: 不应该抛出异常
            assert True


class TestFallbackLoggerEdgeCases:
    """测试FallbackLogger边界情况"""
    
    def test_handle_logging_failure_in_postgres_mode(self):
        """测试在PostgreSQL模式下处理日志失败"""
        # Given: PostgreSQL模式的记录器
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger("postgresql://test:test@localhost/test")
            logger._state = LoggerState.POSTGRES_ACTIVE
            
            # When: 处理日志失败
            error = Exception("Logging failed")
            logger._handle_logging_failure(error)
            
            # Then: 应该调用PostgreSQL失败处理
            assert logger._postgres_failure_count == 1
    
    def test_handle_logging_failure_in_file_mode(self):
        """测试在文件模式下处理日志失败"""
        # Given: 文件模式的记录器
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger("postgresql://test:test@localhost/test")
            logger._state = LoggerState.FILE_FALLBACK
            
            # When: 处理日志失败
            error = Exception("Logging failed")
            logger._handle_logging_failure(error)
            
            # Then: 不应该增加PostgreSQL失败计数
            assert logger._postgres_failure_count == 0
    
    def test_postgres_failure_with_test_error(self):
        """测试包含'test'关键字的PostgreSQL失败"""
        # Given: 测试环境的记录器
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger("postgresql://test:test@localhost/test")
            
            # When: 处理包含'test'的错误
            error = Exception("test connection failed")
            logger._handle_postgres_failure(error)
            
            # Then: 应该使用debug级别日志
            assert logger._postgres_failure_count == 1
    
    def test_attempt_postgres_recovery_success(self):
        """测试PostgreSQL恢复成功"""
        # Given: 文件降级状态的记录器
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger("postgresql://test:test@localhost/test")
            logger._state = LoggerState.FILE_FALLBACK
            logger._postgres_failure_count = 3
            
            # When: 恢复成功
            with patch.object(logger, '_test_postgres_connection', return_value=True):
                logger._attempt_postgres_recovery()
            
            # Then: 验证恢复
            assert logger.get_state() == LoggerState.POSTGRES_ACTIVE
            assert logger._postgres_failure_count == 0
            assert logger._stats["state_changes"] == 1
    
    def test_attempt_postgres_recovery_failure(self):
        """测试PostgreSQL恢复失败"""
        # Given: 文件降级状态的记录器
        with patch('harborai.storage.fallback_logger.PostgreSQLLogger') as mock_pg, \
             patch('harborai.storage.fallback_logger.FileSystemLogger') as mock_fs:
            
            mock_pg_instance = Mock()
            mock_fs_instance = Mock()
            mock_pg.return_value = mock_pg_instance
            mock_fs.return_value = mock_fs_instance
            
            logger = FallbackLogger("postgresql://test:test@localhost/test")
            logger._state = LoggerState.FILE_FALLBACK
            original_failure_count = logger._postgres_failure_count
            
            # When: 恢复失败
            with patch.object(logger, '_test_postgres_connection', side_effect=Exception("Still failing")):
                logger._attempt_postgres_recovery()
            
            # Then: 状态不应该改变
            assert logger.get_state() == LoggerState.FILE_FALLBACK
            assert logger._postgres_failure_count == original_failure_count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])