"""
增强的 PostgreSQL 日志记录器测试
测试连接池、批处理器和错误处理器的集成
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from decimal import Decimal

from harborai.storage.enhanced_postgres_logger import EnhancedPostgreSQLLogger, DateTimeEncoder
from harborai.storage.connection_pool import ConnectionPoolConfig
from harborai.storage.batch_processor import BatchConfig, BatchPriority
from harborai.storage.error_handler import RetryConfig, ErrorCategory, ErrorSeverity
from harborai.database.models import APILog, TokenUsageModel, TracingInfoModel
from harborai.utils.exceptions import StorageError


@pytest.fixture
def connection_string():
    """测试连接字符串"""
    return "postgresql://test:test@localhost:5432/test_db"


@pytest.fixture
def pool_config():
    """连接池配置"""
    return ConnectionPoolConfig(
        min_connections=2,
        max_connections=5,
        connection_timeout=10.0,
        idle_timeout=60.0,
        max_lifetime=300.0
    )


@pytest.fixture
def batch_config():
    """批处理配置"""
    return BatchConfig(
        max_batch_size=50,
        flush_interval=2.0,
        max_queue_size=1000
    )


@pytest.fixture
def retry_config():
    """重试配置"""
    return RetryConfig(
        max_retries=3,
        base_delay=1.0,
        max_delay=10.0
    )


@pytest.fixture
def sample_api_log():
    """示例 API 日志"""
    return APILog(
        id="test-log-1",
        timestamp=datetime.now(),
        provider="openai",
        model="gpt-3.5-turbo",
        request_data='{"model": "gpt-3.5-turbo", "messages": []}',
        response_data='{"choices": [{"message": {"content": "Hello"}}]}',
        status_code=200,
        error_message=None,
        duration_ms=1500.0
    )


@pytest.fixture
def sample_token_usage():
    """示例 Token 使用"""
    return TokenUsageModel(
        id="token-1",
        log_id="test-log-1",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )


@pytest.fixture
def sample_tracing_info():
    """示例追踪信息"""
    return TracingInfoModel(
        id="trace-1",
        log_id="test-log-1",
        hb_trace_id="hb_trace-789",
        otel_trace_id="otel_trace-789",
        span_id="span-abc",
        parent_span_id="parent-span-def",
        operation_name="chat.completion",
        start_time=datetime.now(),
        duration_ms=1000,
        status="ok",
        api_tags={"model": "gpt-3.5-turbo"},
        logs={"events": ["Request started", "Response received"]}
    )


class TestDateTimeEncoder:
    """测试 DateTimeEncoder"""
    
    def test_datetime_encoding(self):
        """测试 datetime 编码"""
        encoder = DateTimeEncoder()
        dt = datetime(2023, 1, 1, 12, 0, 0)
        result = encoder.default(dt)
        assert result == "2023-01-01T12:00:00"
    
    def test_decimal_encoding(self):
        """测试 Decimal 编码"""
        encoder = DateTimeEncoder()
        decimal_val = Decimal("123.45")
        result = encoder.default(decimal_val)
        assert result == 123.45
    
    def test_unsupported_type(self):
        """测试不支持的类型"""
        encoder = DateTimeEncoder()
        with pytest.raises(TypeError):
            encoder.default(object())


class TestEnhancedPostgreSQLLogger:
    """测试增强的 PostgreSQL 日志记录器"""
    
    @pytest.fixture
    def mock_connection_pool(self):
        """模拟连接池"""
        pool = AsyncMock()
        pool.initialize = AsyncMock()
        pool.shutdown = AsyncMock()
        pool.get_connection = AsyncMock()
        pool.health_check = AsyncMock(return_value={'healthy': True})
        pool.get_stats = Mock(return_value=Mock(
            total_connections=5,
            active_connections=2,
            idle_connections=3,
            failed_connections=0,
            average_connection_time=0.1,
            total_requests=100,
            successful_requests=95,
            failed_requests=5
        ))
        return pool
    
    @pytest.fixture
    def mock_batch_processor(self):
        """模拟批处理器"""
        processor = AsyncMock()
        processor.start = AsyncMock()
        processor.stop = AsyncMock()
        processor.add_item = AsyncMock()
        processor._running = True
        processor._queue = []
        processor._current_batch = []
        processor.get_stats = Mock(return_value=Mock(
            total_items=100,
            processed_items=95,
            failed_items=5,
            total_batches=10,
            successful_batches=9,
            failed_batches=1,
            current_batch_size=5,
            average_batch_size=10.0,
            average_processing_time=0.5,
            items_per_second=20.0
        ))
        return processor
    
    @pytest.fixture
    def mock_error_handler(self):
        """模拟错误处理器"""
        handler = Mock()
        handler.handle_error = Mock(return_value=True)
        handler.get_stats = Mock(return_value=Mock(
            total_errors=5,
            errors_by_category={ErrorCategory.CONNECTION: 2},
            errors_by_severity={ErrorSeverity.MEDIUM: 3},
            total_retries=10,
            successful_retries=8,
            failed_retries=2,
            average_retry_delay=1.5,
            error_rate=0.1
        ))
        handler.get_error_summary = Mock(return_value={
            'total_errors': 2,
            'error_rate': 0.5
        })
        return handler
    
    @pytest.fixture
    async def logger_instance(self, connection_string, pool_config, batch_config, retry_config):
        """创建日志记录器实例"""
        logger = EnhancedPostgreSQLLogger(
            connection_string=connection_string,
            pool_config=pool_config,
            batch_config=batch_config,
            retry_config=retry_config
        )
        yield logger
        
        # 清理
        if logger._running:
            await logger.stop()
    
    @pytest.mark.asyncio
    async def test_initialization(self, logger_instance):
        """测试初始化"""
        assert logger_instance.connection_string
        assert logger_instance.pool_config
        assert logger_instance.batch_config
        assert logger_instance.retry_config
        assert logger_instance.error_handler
        assert not logger_instance._running
    
    @pytest.mark.asyncio
    async def test_start_stop(self, logger_instance):
        """测试启动和停止"""
        with patch.object(logger_instance, '_initialize_database', new_callable=AsyncMock), \
             patch.object(logger_instance, '_initialize_connection_pool', new_callable=AsyncMock), \
             patch.object(logger_instance, '_initialize_batch_processor', new_callable=AsyncMock):
            
            # 启动
            await logger_instance.start()
            assert logger_instance._running
            
            # 重复启动应该被忽略
            await logger_instance.start()
            assert logger_instance._running
            
            # 停止
            await logger_instance.stop()
            assert not logger_instance._running
    
    @pytest.mark.asyncio
    async def test_start_failure(self, logger_instance):
        """测试启动失败"""
        with patch.object(logger_instance, '_initialize_database', new_callable=AsyncMock) as mock_init_db:
            mock_init_db.side_effect = Exception("Database connection failed")
            
            with pytest.raises(StorageError):
                await logger_instance.start()
            
            assert not logger_instance._running
    
    @pytest.mark.asyncio
    async def test_log_api_call_success(self, logger_instance, sample_api_log, sample_token_usage, sample_tracing_info):
        """测试成功记录 API 调用"""
        # 模拟运行状态
        logger_instance._running = True
        logger_instance.batch_processor = AsyncMock()
        
        await logger_instance.log_api_call(
            api_log=sample_api_log,
            token_usage=sample_token_usage,
            tracing_info=sample_tracing_info,
            priority=BatchPriority.HIGH
        )
        
        # 验证批处理器被调用
        logger_instance.batch_processor.add_item.assert_called_once()
        
        # 验证统计更新
        assert logger_instance._stats['total_logs'] == 1
    
    @pytest.mark.asyncio
    async def test_log_api_call_not_running(self, logger_instance, sample_api_log):
        """测试日志记录器未运行时记录 API 调用"""
        with pytest.raises(StorageError, match="日志记录器未运行"):
            await logger_instance.log_api_call(api_log=sample_api_log)
    
    @pytest.mark.asyncio
    async def test_log_api_call_failure(self, logger_instance, sample_api_log):
        """测试记录 API 调用失败"""
        logger_instance._running = True
        logger_instance.batch_processor = AsyncMock()
        logger_instance.batch_processor.add_item.side_effect = Exception("Batch processor error")
        
        with pytest.raises(StorageError):
            await logger_instance.log_api_call(api_log=sample_api_log)
        
        # 验证失败统计更新
        assert logger_instance._stats['failed_logs'] == 1
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self, logger_instance, sample_api_log):
        """测试成功处理批次"""
        # 准备测试数据
        batch_items = [Mock(data={'api_log': sample_api_log})]
        
        # 模拟连接池
        mock_conn = AsyncMock()
        mock_trans = Mock()
        mock_conn.begin.return_value = mock_trans
        mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_conn.__aexit__ = AsyncMock(return_value=None)
        
        logger_instance.connection_pool = Mock()
        logger_instance.connection_pool.get_connection.return_value = mock_conn
        
        # 模拟批量插入
        with patch.object(logger_instance, '_batch_insert_data', new_callable=AsyncMock):
            result = await logger_instance._process_batch(batch_items)
        
        assert result is True
        assert logger_instance._stats['successful_logs'] == 1
        assert logger_instance._stats['successful_batches'] == 1
    
    @pytest.mark.asyncio
    async def test_process_batch_failure(self, logger_instance, sample_api_log):
        """测试处理批次失败"""
        batch_items = [Mock(data={'api_log': sample_api_log})]
        
        # 模拟连接池错误
        logger_instance.connection_pool = Mock()
        logger_instance.connection_pool.get_connection.side_effect = Exception("Connection failed")
        
        # 模拟错误处理器
        logger_instance.error_handler.handle_error = Mock(return_value=False)
        
        result = await logger_instance._process_batch(batch_items)
        
        assert result is False
        assert logger_instance._stats['failed_logs'] == 1
        assert logger_instance._stats['failed_batches'] == 1
    
    @pytest.mark.asyncio
    async def test_batch_insert_data(self, logger_instance, sample_api_log, sample_token_usage, sample_tracing_info):
        """测试批量插入数据"""
        # 准备测试数据
        batch_items = [Mock(data={
            'api_log': sample_api_log,
            'token_usage': sample_token_usage,
            'tracing_info': sample_tracing_info
        })]
        
        # 模拟数据库连接
        mock_conn = Mock()
        
        # 模拟表对象
        logger_instance.api_logs_table = Mock()
        logger_instance.token_usage_table = Mock()
        logger_instance.tracing_info_table = Mock()
        
        with patch('asyncio.get_event_loop') as mock_loop:
            mock_loop.return_value.run_in_executor = AsyncMock()
            
            await logger_instance._batch_insert_data(mock_conn, batch_items)
            
            # 验证执行器被调用了3次（三个表）
            assert mock_loop.return_value.run_in_executor.call_count == 3
    
    def test_fallback_handler_connection_error(self, logger_instance):
        """测试连接错误的降级处理"""
        error_info = Mock()
        error_info.category = ErrorCategory.CONNECTION
        
        with patch.object(logger_instance, '_reinitialize_connection_pool', new_callable=AsyncMock):
            result = logger_instance._fallback_handler(error_info)
            assert result is True
    
    def test_fallback_handler_resource_exhausted(self, logger_instance):
        """测试资源耗尽的降级处理"""
        error_info = Mock()
        error_info.category = ErrorCategory.RESOURCE_EXHAUSTED
        
        # 模拟批处理器
        logger_instance.batch_processor = Mock()
        logger_instance.batch_processor.config = Mock()
        logger_instance.batch_processor.config.max_batch_size = 100
        
        result = logger_instance._fallback_handler(error_info)
        
        assert result is True
        assert logger_instance.batch_processor.config.max_batch_size == 50
    
    def test_fallback_handler_unknown_error(self, logger_instance):
        """测试未知错误的降级处理"""
        error_info = Mock()
        error_info.category = ErrorCategory.UNKNOWN
        
        result = logger_instance._fallback_handler(error_info)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_reinitialize_connection_pool(self, logger_instance):
        """测试重新初始化连接池"""
        # 模拟现有连接池
        old_pool = AsyncMock()
        logger_instance.connection_pool = old_pool
        
        with patch.object(logger_instance, '_initialize_connection_pool', new_callable=AsyncMock):
            await logger_instance._reinitialize_connection_pool()
            
            # 验证旧连接池被关闭
            old_pool.shutdown.assert_called_once()
    
    def test_alert_callback(self, logger_instance):
        """测试告警回调"""
        error_info = Mock()
        error_info.category = ErrorCategory.CONNECTION
        error_info.exception = Exception("Test error")
        
        logger_instance._alert_callback(error_info)
        
        # 验证连接错误统计更新
        assert logger_instance._stats['connection_errors'] == 1
    
    def test_get_statistics(self, logger_instance, mock_connection_pool, mock_batch_processor, mock_error_handler):
        """测试获取统计信息"""
        # 设置模拟组件
        logger_instance.connection_pool = mock_connection_pool
        logger_instance.batch_processor = mock_batch_processor
        logger_instance.error_handler = mock_error_handler
        
        stats = logger_instance.get_statistics()
        
        # 验证基本统计
        assert 'total_logs' in stats
        assert 'successful_logs' in stats
        assert 'failed_logs' in stats
        
        # 验证连接池统计
        assert 'connection_pool' in stats
        assert stats['connection_pool']['total_connections'] == 5
        
        # 验证批处理器统计
        assert 'batch_processor' in stats
        assert stats['batch_processor']['total_items'] == 100
        
        # 验证错误处理器统计
        assert 'error_handler' in stats
        assert stats['error_handler']['total_errors'] == 5
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, logger_instance, mock_connection_pool, mock_batch_processor, mock_error_handler):
        """测试健康检查 - 健康状态"""
        # 设置模拟组件
        logger_instance.connection_pool = mock_connection_pool
        logger_instance.batch_processor = mock_batch_processor
        logger_instance.error_handler = mock_error_handler
        
        health = await logger_instance.health_check()
        
        assert health['status'] == 'healthy'
        assert 'connection_pool' in health['components']
        assert 'batch_processor' in health['components']
        assert 'error_rate' in health['components']
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy_pool(self, logger_instance, mock_connection_pool, mock_batch_processor, mock_error_handler):
        """测试健康检查 - 连接池不健康"""
        # 设置不健康的连接池
        mock_connection_pool.health_check = AsyncMock(return_value={'healthy': False})
        
        logger_instance.connection_pool = mock_connection_pool
        logger_instance.batch_processor = mock_batch_processor
        logger_instance.error_handler = mock_error_handler
        
        health = await logger_instance.health_check()
        
        assert health['status'] == 'unhealthy'
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy_batch_processor(self, logger_instance, mock_connection_pool, mock_batch_processor, mock_error_handler):
        """测试健康检查 - 批处理器不健康"""
        # 设置不健康的批处理器
        mock_batch_processor._running = False
        
        logger_instance.connection_pool = mock_connection_pool
        logger_instance.batch_processor = mock_batch_processor
        logger_instance.error_handler = mock_error_handler
        
        health = await logger_instance.health_check()
        
        assert health['status'] == 'unhealthy'
    
    @pytest.mark.asyncio
    async def test_health_check_degraded_error_rate(self, logger_instance, mock_connection_pool, mock_batch_processor, mock_error_handler):
        """测试健康检查 - 错误率过高"""
        # 设置高错误率
        mock_error_handler.get_error_summary = Mock(return_value={'error_rate': 15.0})
        
        logger_instance.connection_pool = mock_connection_pool
        logger_instance.batch_processor = mock_batch_processor
        logger_instance.error_handler = mock_error_handler
        
        health = await logger_instance.health_check()
        
        assert health['status'] == 'degraded'
        assert health['components']['error_rate']['healthy'] is False
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, logger_instance):
        """测试健康检查异常"""
        # 模拟异常
        logger_instance.connection_pool = Mock()
        logger_instance.connection_pool.health_check.side_effect = Exception("Health check failed")
        
        health = await logger_instance.health_check()
        
        assert health['status'] == 'unhealthy'
        assert 'error' in health
    
    def test_sanitize_data_dict(self, logger_instance):
        """测试脱敏处理 - 字典"""
        data = {
            "api_key": "secret-key",
            "message": "Hello world",
            "authorization": "Bearer token",
            "normal_field": "normal_value"
        }
        
        result = logger_instance._sanitize_data(data)
        
        assert result["api_key"] == "[REDACTED]"
        assert result["authorization"] == "[REDACTED]"
        assert result["message"] == "Hello world"
        assert result["normal_field"] == "normal_value"
    
    def test_sanitize_data_list(self, logger_instance):
        """测试脱敏处理 - 列表"""
        data = [
            {"api_key": "secret"},
            {"message": "hello"}
        ]
        
        result = logger_instance._sanitize_data(data)
        
        assert result[0]["api_key"] == "[REDACTED]"
        assert result[1]["message"] == "hello"
    
    def test_sanitize_data_primitive(self, logger_instance):
        """测试脱敏处理 - 基本类型"""
        assert logger_instance._sanitize_data("string") == "string"
        assert logger_instance._sanitize_data(123) == 123
        assert logger_instance._sanitize_data(True) is True