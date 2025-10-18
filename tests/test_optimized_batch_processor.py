#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的批量处理器测试

测试PostgreSQL存储架构优化功能，包括：
- 智能批量大小调整
- 数据库连接池优化
- 高性能批量插入策略
- 实时性能监控
- 自适应负载均衡
"""

import pytest
import asyncio
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from decimal import Decimal

from harborai.storage.optimized_batch_processor import (
    OptimizedBatchProcessor,
    OptimizedBatchConfig,
    OptimizationStrategy,
    PerformanceMetrics,
    get_optimized_batch_processor
)
from harborai.storage.batch_processor import BatchPriority, BatchItem


class TestOptimizedBatchProcessor:
    """优化的批量处理器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.connection_string = "postgresql://test:test@localhost:5432/test_db"
        self.table_name = "test_logs"
        
        # 创建测试配置
        self.config = OptimizedBatchConfig(
            min_batch_size=10,
            max_batch_size=100,
            flush_interval=1.0,
            optimization_strategy=OptimizationStrategy.BALANCED,
            use_copy_from=True,
            use_prepared_statements=True,
            enable_detailed_metrics=True,
            db_pool_min_connections=2,
            db_pool_max_connections=5
        )
    
    def teardown_method(self):
        """测试后清理"""
        pass
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self):
        """测试处理器初始化"""
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        assert processor.connection_string == self.connection_string
        assert processor.table_name == self.table_name
        assert processor.config.optimization_strategy == OptimizationStrategy.BALANCED
        assert processor._running is False
        assert processor._current_batch_size == self.config.min_batch_size
    
    @pytest.mark.asyncio
    @patch('harborai.storage.optimized_batch_processor.ThreadedConnectionPool')
    async def test_start_and_stop(self, mock_pool_class):
        """测试启动和停止"""
        # 模拟数据库连接池
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool
        
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        # 测试启动
        await processor.start()
        assert processor._running is True
        assert processor._db_pool is not None
        assert len(processor._worker_threads) > 0
        
        # 测试停止
        await processor.stop()
        assert processor._running is False
        mock_pool.closeall.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('harborai.storage.optimized_batch_processor.ThreadedConnectionPool')
    async def test_add_item(self, mock_pool_class):
        """测试添加项目"""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        await processor.start()
        
        # 测试添加普通优先级项目
        test_data = {"id": 1, "message": "test", "timestamp": datetime.now()}
        result = await processor.add_item(test_data, BatchPriority.NORMAL)
        
        assert result is True
        assert processor._stats.total_items == 1
        assert processor._stats.queue_size == 1
        assert processor._performance_metrics.queue_depth == 1
        
        # 测试添加高优先级项目
        high_priority_data = {"id": 2, "message": "urgent", "timestamp": datetime.now()}
        result = await processor.add_item(high_priority_data, BatchPriority.HIGH)
        
        assert result is True
        assert processor._stats.total_items == 2
        
        await processor.stop()
    
    @pytest.mark.asyncio
    @patch('harborai.storage.optimized_batch_processor.ThreadedConnectionPool')
    async def test_optimization_strategies(self, mock_pool_class):
        """测试不同的优化策略"""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool
        
        # 测试吞吐量优先策略
        throughput_config = OptimizedBatchConfig(
            optimization_strategy=OptimizationStrategy.THROUGHPUT,
            min_batch_size=50,
            max_batch_size=500
        )
        
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=throughput_config
        )
        
        await processor.start()
        
        # 验证工作线程数量（吞吐量优先应该有更多线程）
        assert len(processor._worker_threads) >= 4
        
        await processor.stop()
        
        # 测试延迟优先策略
        latency_config = OptimizedBatchConfig(
            optimization_strategy=OptimizationStrategy.LATENCY,
            min_batch_size=5,
            max_batch_size=50
        )
        
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=latency_config
        )
        
        await processor.start()
        
        # 验证工作线程数量（延迟优先应该有较少线程）
        assert len(processor._worker_threads) <= 2
        
        await processor.stop()
    
    def test_collect_optimized_batch(self):
        """测试优化的批次收集"""
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        # 添加测试项目到队列
        for i in range(20):
            item = BatchItem(
                data={"id": i, "message": f"test_{i}"},
                priority=BatchPriority.NORMAL if i % 2 == 0 else BatchPriority.HIGH
            )
            processor._priority_queue.put((item.priority.value, time.time(), item))
        
        # 收集批次
        batch = processor._collect_optimized_batch()
        
        # 验证批次大小
        assert len(batch) <= processor._current_batch_size
        assert len(batch) > 0
        
        # 验证高优先级项目优先处理
        if len(batch) > 1:
            priorities = [item.priority for item in batch]
            # 高优先级项目应该在前面
            high_priority_count = sum(1 for p in priorities if p == BatchPriority.HIGH)
            if high_priority_count > 0:
                # 至少第一个应该是高优先级
                assert batch[0].priority == BatchPriority.HIGH
    
    @patch('harborai.storage.optimized_batch_processor.ThreadedConnectionPool')
    def test_process_batch_copy_from(self, mock_pool_class):
        """测试COPY FROM批次处理"""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool
        
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        processor._db_pool = mock_pool
        
        # 创建测试批次
        batch = []
        for i in range(10):
            item = BatchItem(data={
                "id": i,
                "message": f"test_message_{i}",
                "timestamp": datetime.now(),
                "value": i * 10.5
            })
            batch.append(item)
        
        # 测试COPY FROM处理
        result = processor._process_batch_copy_from(mock_conn, batch)
        
        assert result is True
        mock_cursor.copy_from.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    @patch('harborai.storage.optimized_batch_processor.ThreadedConnectionPool')
    def test_process_batch_prepared(self, mock_pool_class):
        """测试预编译语句批次处理"""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool
        
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        processor._db_pool = mock_pool
        
        # 创建测试批次
        batch = []
        for i in range(5):
            item = BatchItem(data={
                "id": i,
                "message": f"test_message_{i}",
                "timestamp": datetime.now()
            })
            batch.append(item)
        
        # 测试预编译语句处理
        result = processor._process_batch_prepared(mock_conn, batch)
        
        assert result is True
        mock_cursor.executemany.assert_called_once()
        mock_conn.commit.assert_called_once()
    
    def test_adaptive_optimization(self):
        """测试自适应优化"""
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        initial_batch_size = processor._current_batch_size
        
        # 模拟高效率处理
        for i in range(10):
            processor._adaptive_optimization(
                processing_time=0.1,  # 快速处理
                batch_size=50,
                success=True
            )
        
        # 强制调整批次大小
        processor._last_optimization = 0  # 重置时间限制
        processor._adjust_batch_size()
        
        # 验证批次大小可能增加（取决于策略）
        if processor.config.optimization_strategy == OptimizationStrategy.THROUGHPUT:
            assert processor._current_batch_size >= initial_batch_size
    
    def test_performance_metrics_update(self):
        """测试性能指标更新"""
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        # 添加延迟样本
        latencies = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.12, 0.28, 0.16]
        processor._latency_samples.extend(latencies)
        
        # 更新性能指标
        processor._update_performance_metrics()
        
        # 验证百分位数计算
        assert processor._performance_metrics.latency_p50 > 0
        assert processor._performance_metrics.latency_p95 > processor._performance_metrics.latency_p50
        assert processor._performance_metrics.latency_p99 > processor._performance_metrics.latency_p95
    
    def test_batch_stats_update(self):
        """测试批次统计更新"""
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        # 创建测试批次
        batch = [BatchItem(data={"id": i}) for i in range(10)]
        
        # 更新成功批次统计
        processor._update_batch_stats(batch, success=True, processing_time=0.5)
        
        assert processor._stats.total_batches == 1
        assert processor._stats.successful_batches == 1
        assert processor._stats.processed_items == 10
        assert processor._stats.failed_batches == 0
        
        # 更新失败批次统计
        processor._update_batch_stats(batch, success=False, processing_time=1.0)
        
        assert processor._stats.total_batches == 2
        assert processor._stats.successful_batches == 1
        assert processor._stats.failed_batches == 1
        assert processor._stats.failed_items == 10
    
    def test_format_data_for_copy(self):
        """测试COPY FROM数据格式化"""
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        # 测试普通数据
        data = {
            "id": 123,
            "message": "test message",
            "value": 45.67,
            "flag": True,
            "empty": None
        }
        
        formatted = processor._format_data_for_copy(data)
        
        # 验证格式化结果
        assert "123" in formatted
        assert "test message" in formatted
        assert "45.67" in formatted
        assert "True" in formatted
        assert "\\N" in formatted  # NULL值
        
        # 测试特殊字符转义
        special_data = {
            "text": "line1\nline2\ttab",
            "value": 100
        }
        
        formatted_special = processor._format_data_for_copy(special_data)
        assert "\\n" in formatted_special
        assert "\\t" in formatted_special
    
    def test_format_data_for_prepared(self):
        """测试预编译语句数据格式化"""
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        data = {
            "id": 123,
            "message": "test",
            "value": 45.67
        }
        
        formatted = processor._format_data_for_prepared(data)
        
        # 验证返回元组
        assert isinstance(formatted, tuple)
        assert len(formatted) == len(data)
        assert 123 in formatted
        assert "test" in formatted
        assert 45.67 in formatted
    
    def test_build_insert_statement(self):
        """测试插入语句构建"""
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        data = {
            "id": 123,
            "message": "test",
            "timestamp": datetime.now()
        }
        
        stmt, values = processor._build_insert_statement(data)
        
        # 验证语句格式
        assert f"INSERT INTO {self.table_name}" in stmt
        assert "VALUES" in stmt
        assert stmt.count("%s") == len(data)
        
        # 验证值
        assert isinstance(values, tuple)
        assert len(values) == len(data)
        assert 123 in values
        assert "test" in values
    
    def test_get_stats(self):
        """测试获取统计信息"""
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        # 设置一些统计数据
        processor._stats.total_items = 100
        processor._stats.processed_items = 95
        processor._stats.failed_items = 5
        processor._stats.total_batches = 10
        processor._stats.successful_batches = 9
        processor._stats.failed_batches = 1
        
        stats = processor.get_stats()
        
        # 验证统计信息
        assert stats['total_items'] == 100
        assert stats['processed_items'] == 95
        assert stats['failed_items'] == 5
        assert stats['total_batches'] == 10
        assert stats['successful_batches'] == 9
        assert stats['failed_batches'] == 1
        assert 'performance_metrics' in stats
        assert 'current_batch_size' in stats
        assert 'optimization_strategy' in stats
    
    def test_get_health_status(self):
        """测试获取健康状态"""
        processor = OptimizedBatchProcessor(
            connection_string=self.connection_string,
            table_name=self.table_name,
            config=self.config
        )
        
        health = processor.get_health_status()
        
        # 验证健康状态字段
        assert 'running' in health
        assert 'worker_threads' in health
        assert 'queue_size' in health
        assert 'db_pool_status' in health
        assert 'error_rate' in health
        assert 'throughput' in health
        assert 'memory_usage' in health
        
        # 验证初始状态
        assert health['running'] is False
        assert health['worker_threads'] == 0
        assert health['queue_size'] == 0
        assert health['db_pool_status'] == 'unavailable'


class TestOptimizedBatchConfig:
    """优化批处理配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = OptimizedBatchConfig()
        
        # 验证默认值
        assert config.optimization_strategy == OptimizationStrategy.BALANCED
        assert config.use_copy_from is True
        assert config.use_prepared_statements is True
        assert config.enable_compression is True
        assert config.auto_tune_batch_size is True
        assert config.enable_detailed_metrics is True
        assert config.db_pool_min_connections == 5
        assert config.db_pool_max_connections == 20
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = OptimizedBatchConfig(
            min_batch_size=20,
            max_batch_size=200,
            optimization_strategy=OptimizationStrategy.THROUGHPUT,
            use_copy_from=False,
            db_pool_max_connections=50,
            compression_threshold=500
        )
        
        # 验证自定义值
        assert config.min_batch_size == 20
        assert config.max_batch_size == 200
        assert config.optimization_strategy == OptimizationStrategy.THROUGHPUT
        assert config.use_copy_from is False
        assert config.db_pool_max_connections == 50
        assert config.compression_threshold == 500


class TestPerformanceMetrics:
    """性能指标测试类"""
    
    def test_metrics_initialization(self):
        """测试指标初始化"""
        metrics = PerformanceMetrics()
        
        # 验证默认值
        assert metrics.throughput == 0.0
        assert metrics.latency_p50 == 0.0
        assert metrics.latency_p95 == 0.0
        assert metrics.latency_p99 == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.cpu_usage == 0.0
        assert metrics.error_rate == 0.0
        assert metrics.queue_depth == 0
        assert metrics.batch_efficiency == 0.0
    
    def test_metrics_update(self):
        """测试指标更新"""
        metrics = PerformanceMetrics()
        
        # 更新指标
        metrics.throughput = 1000.0
        metrics.latency_p50 = 0.1
        metrics.latency_p95 = 0.5
        metrics.latency_p99 = 1.0
        metrics.error_rate = 0.01
        metrics.batch_efficiency = 0.95
        
        # 验证更新
        assert metrics.throughput == 1000.0
        assert metrics.latency_p50 == 0.1
        assert metrics.latency_p95 == 0.5
        assert metrics.latency_p99 == 1.0
        assert metrics.error_rate == 0.01
        assert metrics.batch_efficiency == 0.95


class TestGlobalProcessorManager:
    """全局处理器管理测试类"""
    
    def test_get_optimized_batch_processor(self):
        """测试获取全局处理器实例"""
        connection_string = "postgresql://test:test@localhost:5432/test"
        table_name = "test_table"
        
        processor1 = get_optimized_batch_processor(connection_string, table_name)
        processor2 = get_optimized_batch_processor(connection_string, table_name)
        
        # 验证单例模式
        assert processor1 is processor2
        assert isinstance(processor1, OptimizedBatchProcessor)
        assert processor1.connection_string == connection_string
        assert processor1.table_name == table_name


class TestIntegrationScenarios:
    """集成场景测试类"""
    
    @pytest.mark.asyncio
    @patch('harborai.storage.optimized_batch_processor.ThreadedConnectionPool')
    async def test_high_throughput_scenario(self, mock_pool_class):
        """测试高吞吐量场景"""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool
        
        # 配置高吞吐量策略
        config = OptimizedBatchConfig(
            optimization_strategy=OptimizationStrategy.THROUGHPUT,
            min_batch_size=100,
            max_batch_size=1000,
            use_copy_from=True,
            db_pool_max_connections=20
        )
        
        processor = OptimizedBatchProcessor(
            connection_string="postgresql://test:test@localhost:5432/test",
            table_name="high_throughput_logs",
            config=config
        )
        
        await processor.start()
        
        # 添加大量数据
        start_time = time.time()
        for i in range(1000):
            await processor.add_item({
                "id": i,
                "message": f"high_throughput_test_{i}",
                "timestamp": datetime.now(),
                "value": i * 1.5
            }, BatchPriority.NORMAL)
        
        # 等待处理
        await asyncio.sleep(2)
        
        # 验证处理效果
        stats = processor.get_stats()
        assert stats['total_items'] == 1000
        
        # 验证高吞吐量配置生效
        assert len(processor._worker_threads) >= 4
        
        await processor.stop()
    
    @pytest.mark.asyncio
    @patch('harborai.storage.optimized_batch_processor.ThreadedConnectionPool')
    async def test_low_latency_scenario(self, mock_pool_class):
        """测试低延迟场景"""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool
        
        # 配置低延迟策略
        config = OptimizedBatchConfig(
            optimization_strategy=OptimizationStrategy.LATENCY,
            min_batch_size=5,
            max_batch_size=20,
            flush_interval=0.1,
            use_prepared_statements=True
        )
        
        processor = OptimizedBatchProcessor(
            connection_string="postgresql://test:test@localhost:5432/test",
            table_name="low_latency_logs",
            config=config
        )
        
        await processor.start()
        
        # 添加少量高优先级数据
        for i in range(10):
            await processor.add_item({
                "id": i,
                "urgent_message": f"low_latency_test_{i}",
                "timestamp": datetime.now()
            }, BatchPriority.HIGH)
        
        # 短暂等待
        await asyncio.sleep(0.5)
        
        # 验证低延迟配置生效
        assert len(processor._worker_threads) <= 2
        assert processor._current_batch_size <= 20
        
        await processor.stop()
    
    @pytest.mark.asyncio
    @patch('harborai.storage.optimized_batch_processor.ThreadedConnectionPool')
    async def test_mixed_priority_scenario(self, mock_pool_class):
        """测试混合优先级场景"""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool
        
        processor = OptimizedBatchProcessor(
            connection_string="postgresql://test:test@localhost:5432/test",
            table_name="mixed_priority_logs",
            config=OptimizedBatchConfig()
        )
        
        await processor.start()
        
        # 添加不同优先级的数据
        priorities = [BatchPriority.LOW, BatchPriority.NORMAL, BatchPriority.HIGH, BatchPriority.CRITICAL]
        
        for i in range(40):
            priority = priorities[i % len(priorities)]
            await processor.add_item({
                "id": i,
                "priority_level": priority.name,
                "message": f"mixed_priority_test_{i}",
                "timestamp": datetime.now()
            }, priority)
        
        # 等待处理
        await asyncio.sleep(1)
        
        # 验证统计信息
        stats = processor.get_stats()
        assert stats['total_items'] == 40
        
        await processor.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])