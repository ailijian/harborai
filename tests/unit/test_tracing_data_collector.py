#!/usr/bin/env python3
"""
TracingDataCollector 单元测试

测试追踪数据收集器的核心功能：
- 追踪记录收集和存储
- 批量处理和异步写入
- 数据库连接管理
- 错误处理和重试机制
- 性能监控和统计

作者: HarborAI团队
创建时间: 2025-01-15
版本: v1.0.0
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

import asyncpg

from harborai.core.tracing.data_collector import (
    TracingDataCollector,
    TracingRecord,
    CollectorStatus,
    CollectorStatistics
)
from harborai.core.tracing.dual_trace_manager import DualTraceContext
from harborai.utils.exceptions import TracingError, DatabaseError


class TestTracingRecord:
    """TracingRecord 测试类"""
    
    def test_tracing_record_creation(self):
        """测试追踪记录创建"""
        start_time = datetime.now()
        record = TracingRecord(
            hb_trace_id="hb_test_123",
            otel_trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            operation_name="ai.chat.completion",
            start_time=start_time,
            duration_ms=150.5,
            status="completed",
            api_tags={"model": "gpt-4", "provider": "openai"},
            internal_tags={"service": "harborai", "version": "2.0.0"},
            logs=[{"level": "info", "message": "Request completed"}],
            error="",
            parent_span_id="parent123"
        )
        
        assert record.hb_trace_id == "hb_test_123"
        assert record.otel_trace_id == "0123456789abcdef0123456789abcdef"
        assert record.span_id == "0123456789abcdef"
        assert record.operation_name == "ai.chat.completion"
        assert record.start_time == start_time
        assert record.duration_ms == 150.5
        assert record.status == "completed"
        assert record.api_tags["model"] == "gpt-4"
        assert record.internal_tags["service"] == "harborai"
        assert len(record.logs) == 1
        assert record.error == ""
        assert record.parent_span_id == "parent123"
    
    def test_tracing_record_to_dict(self):
        """测试追踪记录转换为字典"""
        start_time = datetime.now()
        record = TracingRecord(
            hb_trace_id="hb_test_123",
            otel_trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            operation_name="ai.chat.completion",
            start_time=start_time,
            duration_ms=150.5,
            status="completed"
        )
        
        record_dict = record.to_dict()
        
        assert record_dict["hb_trace_id"] == "hb_test_123"
        assert record_dict["otel_trace_id"] == "0123456789abcdef0123456789abcdef"
        assert record_dict["span_id"] == "0123456789abcdef"
        assert record_dict["operation_name"] == "ai.chat.completion"
        assert record_dict["start_time"] == start_time.isoformat()
        assert record_dict["duration_ms"] == 150.5
        assert record_dict["status"] == "completed"
    
    def test_tracing_record_validation(self):
        """测试追踪记录验证"""
        # 测试有效记录
        valid_record = TracingRecord(
            hb_trace_id="hb_test_123",
            otel_trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            operation_name="ai.chat.completion",
            start_time=datetime.now(),
            duration_ms=150.5,
            status="completed"
        )
        
        assert valid_record.is_valid()
        
        # 测试无效记录（缺少必需字段）
        with pytest.raises(ValueError):
            TracingRecord(
                hb_trace_id="",  # 空的trace_id
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
        
        # 测试无效状态
        with pytest.raises(ValueError):
            TracingRecord(
                hb_trace_id="hb_test_123",
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="invalid_status"  # 无效状态
            )


class TestCollectorStatus:
    """CollectorStatus 测试类"""
    
    def test_collector_status_creation(self):
        """测试收集器状态创建"""
        status = CollectorStatus(
            is_running=True,
            is_healthy=True,
            last_error=None,
            queue_size=10,
            processed_count=100,
            error_count=2,
            last_flush_time=datetime.now()
        )
        
        assert status.is_running is True
        assert status.is_healthy is True
        assert status.last_error is None
        assert status.queue_size == 10
        assert status.processed_count == 100
        assert status.error_count == 2
        assert isinstance(status.last_flush_time, datetime)
    
    def test_collector_status_to_dict(self):
        """测试收集器状态转换为字典"""
        last_flush_time = datetime.now()
        status = CollectorStatus(
            is_running=True,
            is_healthy=True,
            last_error="Test error",
            queue_size=10,
            processed_count=100,
            error_count=2,
            last_flush_time=last_flush_time
        )
        
        status_dict = status.to_dict()
        
        assert status_dict["is_running"] is True
        assert status_dict["is_healthy"] is True
        assert status_dict["last_error"] == "Test error"
        assert status_dict["queue_size"] == 10
        assert status_dict["processed_count"] == 100
        assert status_dict["error_count"] == 2
        assert status_dict["last_flush_time"] == last_flush_time.isoformat()


class TestCollectorStatistics:
    """CollectorStatistics 测试类"""
    
    def test_collector_statistics_creation(self):
        """测试收集器统计创建"""
        stats = CollectorStatistics(
            total_records=1000,
            successful_records=950,
            failed_records=50,
            average_processing_time_ms=25.5,
            records_per_second=40.0,
            queue_size=15,
            uptime_seconds=3600
        )
        
        assert stats.total_records == 1000
        assert stats.successful_records == 950
        assert stats.failed_records == 50
        assert stats.average_processing_time_ms == 25.5
        assert stats.records_per_second == 40.0
        assert stats.queue_size == 15
        assert stats.uptime_seconds == 3600
    
    def test_collector_statistics_to_dict(self):
        """测试收集器统计转换为字典"""
        stats = CollectorStatistics(
            total_records=1000,
            successful_records=950,
            failed_records=50,
            average_processing_time_ms=25.5,
            records_per_second=40.0,
            queue_size=15,
            uptime_seconds=3600
        )
        
        stats_dict = stats.to_dict()
        
        assert stats_dict["total_records"] == 1000
        assert stats_dict["successful_records"] == 950
        assert stats_dict["failed_records"] == 50
        assert stats_dict["average_processing_time_ms"] == 25.5
        assert stats_dict["records_per_second"] == 40.0
        assert stats_dict["queue_size"] == 15
        assert stats_dict["uptime_seconds"] == 3600


class TestTracingDataCollector:
    """TracingDataCollector 测试类"""
    
    @pytest.fixture
    def mock_db_config(self):
        """模拟数据库配置"""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "test_harborai",
            "user": "test_user",
            "password": "test_password"
        }
    
    @pytest.fixture
    def collector(self, mock_db_config):
        """创建追踪数据收集器实例"""
        return TracingDataCollector(
            db_config=mock_db_config,
            batch_size=10,
            flush_interval=1.0,
            max_queue_size=100,
            max_retries=3
        )
    
    def test_collector_initialization(self, collector, mock_db_config):
        """测试收集器初始化"""
        assert collector.db_config == mock_db_config
        assert collector.batch_size == 10
        assert collector.flush_interval == 1.0
        assert collector.max_queue_size == 100
        assert collector.max_retries == 3
        assert collector.is_running is False
        assert collector.queue.empty()
        assert collector.processed_count == 0
        assert collector.error_count == 0
    
    @pytest.mark.asyncio
    async def test_collector_start_stop(self, collector):
        """测试收集器启动和停止"""
        with patch.object(collector, '_ensure_table_exists', new_callable=AsyncMock):
            with patch.object(collector, '_create_connection_pool', new_callable=AsyncMock):
                # 启动收集器
                await collector.start()
                
                assert collector.is_running is True
                assert collector.worker_task is not None
                
                # 停止收集器
                await collector.stop()
                
                assert collector.is_running is False
                assert collector.worker_task.cancelled() or collector.worker_task.done()
    
    @pytest.mark.asyncio
    async def test_collector_start_already_running(self, collector):
        """测试重复启动收集器"""
        with patch.object(collector, '_ensure_table_exists', new_callable=AsyncMock):
            with patch.object(collector, '_create_connection_pool', new_callable=AsyncMock):
                # 第一次启动
                await collector.start()
                assert collector.is_running is True
                
                # 第二次启动应该抛出异常
                with pytest.raises(TracingError, match="already running"):
                    await collector.start()
                
                await collector.stop()
    
    @pytest.mark.asyncio
    async def test_collector_stop_not_running(self, collector):
        """测试停止未运行的收集器"""
        # 停止未运行的收集器应该不抛出异常
        await collector.stop()
        assert collector.is_running is False
    
    def test_add_record(self, collector):
        """测试添加追踪记录"""
        record = TracingRecord(
            hb_trace_id="hb_test_123",
            otel_trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            operation_name="ai.chat.completion",
            start_time=datetime.now(),
            duration_ms=150.5,
            status="completed"
        )
        
        # 添加记录
        collector.add_record(record)
        
        assert collector.queue.qsize() == 1
        assert not collector.queue.empty()
    
    def test_add_record_queue_full(self, collector):
        """测试队列满时添加记录"""
        # 填满队列
        for i in range(collector.max_queue_size):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
            collector.add_record(record)
        
        # 队列已满，再添加应该抛出异常
        overflow_record = TracingRecord(
            hb_trace_id="hb_overflow",
            otel_trace_id="0123456789abcdef0123456789abcdef",
            span_id="0123456789abcdef",
            operation_name="ai.chat.completion",
            start_time=datetime.now(),
            duration_ms=150.5,
            status="completed"
        )
        
        with pytest.raises(TracingError, match="Queue is full"):
            collector.add_record(overflow_record)
    
    def test_add_invalid_record(self, collector):
        """测试添加无效记录"""
        # 创建无效记录（空的trace_id）
        with pytest.raises(ValueError):
            invalid_record = TracingRecord(
                hb_trace_id="",  # 空的trace_id
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
            collector.add_record(invalid_record)
    
    def test_get_status(self, collector):
        """测试获取收集器状态"""
        # 添加一些记录
        for i in range(5):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
            collector.add_record(record)
        
        status = collector.get_status()
        
        assert isinstance(status, CollectorStatus)
        assert status.is_running is False  # 未启动
        assert status.queue_size == 5
        assert status.processed_count == 0
        assert status.error_count == 0
    
    def test_get_statistics(self, collector):
        """测试获取收集器统计"""
        # 模拟一些处理统计
        collector.processed_count = 100
        collector.error_count = 5
        collector.start_time = time.time() - 3600  # 1小时前启动
        
        # 添加一些记录到队列
        for i in range(10):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
            collector.add_record(record)
        
        stats = collector.get_statistics()
        
        assert isinstance(stats, CollectorStatistics)
        assert stats.total_records == 105  # 100 processed + 5 failed
        assert stats.successful_records == 100
        assert stats.failed_records == 5
        assert stats.queue_size == 10
        assert stats.uptime_seconds > 3500  # 接近1小时
    
    @pytest.mark.asyncio
    async def test_create_connection_pool(self, collector):
        """测试创建连接池"""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_pool = Mock()
            mock_create_pool.return_value = mock_pool
            
            await collector._create_connection_pool()
            
            assert collector.pool == mock_pool
            mock_create_pool.assert_called_once_with(
                host=collector.db_config["host"],
                port=collector.db_config["port"],
                database=collector.db_config["database"],
                user=collector.db_config["user"],
                password=collector.db_config["password"],
                min_size=2,
                max_size=10,
                command_timeout=30
            )
    
    @pytest.mark.asyncio
    async def test_create_connection_pool_failure(self, collector):
        """测试连接池创建失败"""
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_create_pool.side_effect = Exception("Connection failed")
            
            with pytest.raises(DatabaseError, match="Failed to create connection pool"):
                await collector._create_connection_pool()
    
    @pytest.mark.asyncio
    async def test_ensure_table_exists(self, collector):
        """测试确保表存在"""
        mock_connection = AsyncMock()
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        collector.pool = mock_pool
        
        await collector._ensure_table_exists()
        
        # 验证执行了创建表的SQL
        mock_connection.execute.assert_called()
        call_args = mock_connection.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS tracing_info" in call_args
    
    @pytest.mark.asyncio
    async def test_ensure_table_exists_failure(self, collector):
        """测试表创建失败"""
        mock_connection = AsyncMock()
        mock_connection.execute.side_effect = Exception("Table creation failed")
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        collector.pool = mock_pool
        
        with pytest.raises(DatabaseError, match="Failed to ensure table exists"):
            await collector._ensure_table_exists()
    
    @pytest.mark.asyncio
    async def test_worker_loop(self, collector):
        """测试工作循环"""
        # 创建一些测试记录
        records = []
        for i in range(5):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
            records.append(record)
            collector.add_record(record)
        
        # 模拟批量处理
        with patch.object(collector, '_process_batch', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = True
            
            # 运行一次工作循环
            await collector._worker_loop_once()
            
            # 验证批量处理被调用
            mock_process.assert_called_once()
            call_args = mock_process.call_args[0][0]
            assert len(call_args) == 5  # 处理了5条记录
    
    @pytest.mark.asyncio
    async def test_process_batch(self, collector):
        """测试批量处理"""
        # 创建测试记录
        records = []
        for i in range(3):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
            records.append(record)
        
        # 模拟数据库连接
        mock_connection = AsyncMock()
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        collector.pool = mock_pool
        
        # 处理批量
        result = await collector._process_batch(records)
        
        assert result is True
        assert collector.processed_count == 3
        
        # 验证执行了插入SQL
        mock_connection.executemany.assert_called_once()
        call_args = mock_connection.executemany.call_args
        assert "INSERT INTO tracing_info" in call_args[0][0]
        assert len(call_args[0][1]) == 3  # 3条记录
    
    @pytest.mark.asyncio
    async def test_process_batch_failure(self, collector):
        """测试批量处理失败"""
        # 创建测试记录
        records = []
        for i in range(3):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
            records.append(record)
        
        # 模拟数据库连接失败
        mock_connection = AsyncMock()
        mock_connection.executemany.side_effect = Exception("Database error")
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        collector.pool = mock_pool
        
        # 处理批量应该失败
        result = await collector._process_batch(records)
        
        assert result is False
        assert collector.error_count == 1
    
    @pytest.mark.asyncio
    async def test_process_batch_with_retry(self, collector):
        """测试带重试的批量处理"""
        # 创建测试记录
        records = []
        for i in range(2):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
            records.append(record)
        
        # 模拟数据库连接（第一次失败，第二次成功）
        mock_connection = AsyncMock()
        mock_connection.executemany.side_effect = [
            Exception("Temporary error"),  # 第一次失败
            None  # 第二次成功
        ]
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        collector.pool = mock_pool
        
        # 处理批量（带重试）
        result = await collector._process_batch_with_retry(records)
        
        assert result is True
        assert collector.processed_count == 2
        assert mock_connection.executemany.call_count == 2  # 重试了一次
    
    @pytest.mark.asyncio
    async def test_process_batch_with_retry_exhausted(self, collector):
        """测试重试次数耗尽"""
        # 创建测试记录
        records = []
        for i in range(2):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
            records.append(record)
        
        # 模拟数据库连接始终失败
        mock_connection = AsyncMock()
        mock_connection.executemany.side_effect = Exception("Persistent error")
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        collector.pool = mock_pool
        
        # 处理批量（重试耗尽）
        result = await collector._process_batch_with_retry(records)
        
        assert result is False
        assert collector.error_count == 1
        assert mock_connection.executemany.call_count == collector.max_retries + 1  # 原始尝试 + 重试
    
    def test_clear_queue(self, collector):
        """测试清空队列"""
        # 添加一些记录
        for i in range(5):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id="0123456789abcdef0123456789abcdef",
                span_id="0123456789abcdef",
                operation_name="ai.chat.completion",
                start_time=datetime.now(),
                duration_ms=150.5,
                status="completed"
            )
            collector.add_record(record)
        
        assert collector.queue.qsize() == 5
        
        # 清空队列
        cleared_count = collector.clear_queue()
        
        assert cleared_count == 5
        assert collector.queue.empty()
    
    def test_is_healthy(self, collector):
        """测试健康检查"""
        # 初始状态应该是健康的
        assert collector.is_healthy() is True
        
        # 模拟一些错误
        collector.error_count = 10
        collector.last_error_time = time.time()
        
        # 错误率过高应该不健康
        collector.processed_count = 50  # 错误率 = 10/60 = 16.7% > 10%
        assert collector.is_healthy() is False
        
        # 错误率正常应该健康
        collector.processed_count = 200  # 错误率 = 10/210 = 4.8% < 10%
        assert collector.is_healthy() is True
        
        # 最近有错误但错误率低应该不健康
        collector.last_error_time = time.time() - 30  # 30秒前的错误
        assert collector.is_healthy() is False
        
        # 错误时间较久应该健康
        collector.last_error_time = time.time() - 120  # 2分钟前的错误
        assert collector.is_healthy() is True


class TestTracingDataCollectorIntegration:
    """TracingDataCollector 集成测试"""
    
    @pytest.fixture
    def mock_db_config(self):
        """模拟数据库配置"""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "test_harborai",
            "user": "test_user",
            "password": "test_password"
        }
    
    @pytest.fixture
    def collector(self, mock_db_config):
        """创建追踪数据收集器实例"""
        return TracingDataCollector(
            db_config=mock_db_config,
            batch_size=5,
            flush_interval=0.1,  # 快速刷新用于测试
            max_queue_size=50,
            max_retries=2
        )
    
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, collector):
        """测试完整生命周期"""
        # 模拟数据库操作
        mock_connection = AsyncMock()
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.close = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool):
            # 1. 启动收集器
            await collector.start()
            assert collector.is_running is True
            
            # 2. 添加一些记录
            records = []
            for i in range(12):  # 超过batch_size，会触发多次批量处理
                record = TracingRecord(
                    hb_trace_id=f"hb_integration_test_{i}",
                    otel_trace_id=f"{i:032x}",
                    span_id=f"{i:016x}",
                    operation_name=f"ai.chat.completion.{i}",
                    start_time=datetime.now(),
                    duration_ms=100.0 + i,
                    status="completed",
                    api_tags={"model": "gpt-4", "request_id": f"req_{i}"},
                    internal_tags={"service": "harborai", "version": "2.0.0"}
                )
                records.append(record)
                collector.add_record(record)
            
            # 3. 等待处理
            await asyncio.sleep(0.5)  # 等待批量处理
            
            # 4. 检查状态
            status = collector.get_status()
            assert status.is_running is True
            assert status.processed_count > 0  # 应该处理了一些记录
            
            # 5. 获取统计
            stats = collector.get_statistics()
            assert stats.total_records > 0
            assert stats.successful_records > 0
            
            # 6. 停止收集器
            await collector.stop()
            assert collector.is_running is False
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, collector):
        """测试错误恢复"""
        # 模拟数据库操作（先失败后成功）
        mock_connection = AsyncMock()
        call_count = 0
        
        def mock_executemany(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 前两次失败
                raise Exception("Database temporarily unavailable")
            return None  # 第三次成功
        
        mock_connection.executemany.side_effect = mock_executemany
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.close = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool):
            await collector.start()
            
            # 添加记录
            for i in range(3):
                record = TracingRecord(
                    hb_trace_id=f"hb_error_test_{i}",
                    otel_trace_id=f"{i:032x}",
                    span_id=f"{i:016x}",
                    operation_name=f"ai.chat.completion.{i}",
                    start_time=datetime.now(),
                    duration_ms=100.0,
                    status="completed"
                )
                collector.add_record(record)
            
            # 等待处理（包括重试）
            await asyncio.sleep(0.5)
            
            # 验证重试机制工作
            assert collector.error_count > 0  # 应该有错误记录
            assert collector.processed_count > 0  # 最终应该成功处理
            
            await collector.stop()
    
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, collector):
        """测试队列溢出处理"""
        # 创建一个小队列的收集器
        small_collector = TracingDataCollector(
            db_config=collector.db_config,
            batch_size=5,
            flush_interval=10.0,  # 长时间间隔，防止自动刷新
            max_queue_size=10,  # 小队列
            max_retries=1
        )
        
        # 填满队列
        for i in range(10):
            record = TracingRecord(
                hb_trace_id=f"hb_overflow_test_{i}",
                otel_trace_id=f"{i:032x}",
                span_id=f"{i:016x}",
                operation_name=f"ai.chat.completion.{i}",
                start_time=datetime.now(),
                duration_ms=100.0,
                status="completed"
            )
            small_collector.add_record(record)
        
        # 队列已满，再添加应该失败
        overflow_record = TracingRecord(
            hb_trace_id="hb_overflow",
            otel_trace_id="ffffffffffffffffffffffffffffffff",
            span_id="ffffffffffffffff",
            operation_name="ai.chat.completion.overflow",
            start_time=datetime.now(),
            duration_ms=100.0,
            status="completed"
        )
        
        with pytest.raises(TracingError, match="Queue is full"):
            small_collector.add_record(overflow_record)
        
        # 验证队列状态
        status = small_collector.get_status()
        assert status.queue_size == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, collector):
        """测试并发操作"""
        import threading
        
        # 模拟数据库操作
        mock_connection = AsyncMock()
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.close = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool):
            await collector.start()
            
            # 并发添加记录
            def add_records(thread_id, count):
                for i in range(count):
                    record = TracingRecord(
                        hb_trace_id=f"hb_concurrent_{thread_id}_{i}",
                        otel_trace_id=f"{thread_id:016x}{i:016x}",
                        span_id=f"{thread_id:08x}{i:08x}",
                        operation_name=f"ai.chat.completion.{thread_id}.{i}",
                        start_time=datetime.now(),
                        duration_ms=100.0,
                        status="completed",
                        api_tags={"thread_id": str(thread_id)}
                    )
                    try:
                        collector.add_record(record)
                    except TracingError:
                        # 队列满时忽略错误
                        pass
            
            # 创建多个线程并发添加记录
            threads = []
            for thread_id in range(5):
                thread = threading.Thread(target=add_records, args=(thread_id, 10))
                threads.append(thread)
            
            # 启动所有线程
            for thread in threads:
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 等待处理
            await asyncio.sleep(0.5)
            
            # 验证并发处理结果
            stats = collector.get_statistics()
            assert stats.total_records > 0
            
            await collector.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])