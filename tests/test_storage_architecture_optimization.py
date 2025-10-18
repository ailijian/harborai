#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
存储架构优化测试

测试PostgreSQL批量处理性能优化，包括：
- 批量插入策略优化
- 数据库连接池配置优化
- 批量操作监控增强
- EnhancedPostgreSQLLogger的批处理逻辑优化
"""

import pytest
import asyncio
import time
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
from decimal import Decimal

from harborai.storage.enhanced_postgres_logger import EnhancedPostgreSQLLogger
from harborai.storage.connection_pool import ConnectionPoolConfig, ConnectionPool
from harborai.storage.batch_processor import BatchConfig, BatchPriority, AdaptiveBatchProcessor
from harborai.storage.optimized_batch_processor import OptimizedBatchProcessor, OptimizedBatchConfig
from harborai.database.models import APILog, TokenUsageModel, TracingInfoModel
from harborai.utils.exceptions import StorageError


class TestStorageArchitectureOptimization:
    """存储架构优化测试类"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        import shutil
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def optimized_pool_config(self):
        """优化的连接池配置"""
        return ConnectionPoolConfig(
            min_connections=10,
            max_connections=50,
            connection_timeout=15.0,
            idle_timeout=180.0,  # 3分钟
            max_lifetime=1800.0,  # 30分钟
            health_check_interval=30.0,
            retry_attempts=5,
            retry_delay=0.5,
            enable_health_check=True
        )
    
    @pytest.fixture
    def optimized_batch_config(self):
        """优化的批处理配置"""
        return BatchConfig(
            min_batch_size=50,
            max_batch_size=500,
            flush_interval=2.0,
            max_wait_time=10.0,
            enable_compression=True,
            enable_priority=True,
            adaptive_threshold=0.9,
            performance_window=200
        )
    
    @pytest.fixture
    def enhanced_postgres_logger(self, optimized_pool_config, optimized_batch_config):
        """增强的PostgreSQL日志记录器"""
        return EnhancedPostgreSQLLogger(
            connection_string="postgresql://test:test@localhost:5432/test",
            batch_size=200,
            flush_interval=2.0,
            pool_config=optimized_pool_config,
            batch_config=optimized_batch_config
        )
    
    @pytest.fixture
    def sample_api_log(self):
        """示例API日志"""
        return APILog(
            id="test-request-123",
            timestamp=datetime.now(),
            provider="openai",
            model="gpt-4",
            request_data='{"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}',
            response_data='{"choices": [{"message": {"content": "Hello World"}}]}',
            status_code=200,
            error_message=None,
            duration_ms=1500.0
        )
    
    @pytest.fixture
    def sample_token_usage(self):
        """示例Token使用"""
        return TokenUsageModel(
            request_id="test-request-123",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            cost_breakdown={
                "prompt_cost": 0.002,
                "completion_cost": 0.003,
                "total_cost": 0.005
            }
        )
    
    @pytest.fixture
    def sample_tracing_info(self):
        """示例追踪信息"""
        return TracingInfoModel(
            request_id="test-request-123",
            trace_id="trace-789",
            span_id="span-101",
            parent_span_id="span-100",
            operation_name="chat_completion",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1),
            tags={"model": "gpt-4", "provider": "openai"},
            logs=[{"timestamp": datetime.now(), "message": "Request started"}]
        )
    
    @pytest.mark.asyncio
    async def test_optimized_connection_pool_performance(self, optimized_pool_config):
        """测试优化的连接池性能"""
        # 模拟数据库连接
        with patch('psycopg2.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn
            
            # 创建连接池
            pool = ConnectionPool(
                connection_string="postgresql://test:test@localhost:5432/test",
                config=optimized_pool_config
            )
            
            # 测试并发连接获取
            start_time = time.time()
            tasks = []
            
            async def get_connection_task():
                async with pool.get_connection() as conn:
                    # 模拟数据库操作
                    await asyncio.sleep(0.01)
                    return True
            
            # 创建100个并发任务
            for _ in range(100):
                tasks.append(get_connection_task())
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # 验证性能
            duration = end_time - start_time
            assert duration < 5.0, f"连接池性能不达标，耗时: {duration}秒"
            assert all(r is True for r in results if not isinstance(r, Exception))
            
            # 验证连接池统计
            stats = pool.get_stats()
            assert stats.total_requests >= 100
            assert stats.successful_requests >= 90  # 允许少量失败
    
    @pytest.mark.asyncio
    async def test_batch_insert_strategy_optimization(self, enhanced_postgres_logger, 
                                                     sample_api_log, sample_token_usage, 
                                                     sample_tracing_info):
        """测试批量插入策略优化"""
        # 模拟数据库操作
        with patch.object(enhanced_postgres_logger, '_process_batch', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = True
            
            # 启动日志记录器
            await enhanced_postgres_logger.start()
            
            # 批量添加日志
            start_time = time.time()
            tasks = []
            
            for i in range(1000):
                # 创建不同优先级的日志
                priority = BatchPriority.HIGH if i % 10 == 0 else BatchPriority.NORMAL
                
                task = enhanced_postgres_logger.log_api_call(
                    api_log=sample_api_log,
                    token_usage=sample_token_usage,
                    tracing_info=sample_tracing_info,
                    priority=priority
                )
                tasks.append(task)
            
            # 等待所有任务完成
            await asyncio.gather(*tasks)
            
            # 等待批处理完成
            await asyncio.sleep(3.0)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 验证性能
            assert duration < 10.0, f"批量插入性能不达标，耗时: {duration}秒"
            
            # 验证批处理调用
            assert mock_process.call_count > 0
            
            # 验证统计信息
            stats = enhanced_postgres_logger.get_statistics()
            assert stats['total_logs'] == 1000
            assert stats['successful_logs'] > 0
            assert stats['average_batch_size'] > 1
            
            await enhanced_postgres_logger.stop()
    
    @pytest.mark.asyncio
    async def test_adaptive_batch_size_adjustment(self, enhanced_postgres_logger):
        """测试自适应批量大小调整"""
        # 模拟不同性能的数据库操作
        call_count = 0
        
        async def mock_process_batch(items):
            nonlocal call_count
            call_count += 1
            
            # 模拟不同的处理时间
            if call_count <= 5:
                await asyncio.sleep(0.1)  # 快速处理
            else:
                await asyncio.sleep(0.5)  # 慢速处理
            
            return True
        
        with patch.object(enhanced_postgres_logger, '_process_batch', side_effect=mock_process_batch):
            await enhanced_postgres_logger.start()
            
            # 添加大量数据触发自适应调整
            for i in range(500):
                await enhanced_postgres_logger.log_api_call(
                    api_log=APILog(
                        id=f"test-{i}",
                        timestamp=datetime.now(),
                        provider="openai",
                        model="gpt-4",
                        request_data='{"model": "gpt-4", "messages": []}',
                        response_data='{"choices": [{"message": {"content": "Hello"}}]}',
                        status_code=200,
                        error_message=None,
                        duration_ms=1000.0
                    )
                )
            
            # 等待处理完成
            await asyncio.sleep(5.0)
            
            # 验证自适应调整
            stats = enhanced_postgres_logger.get_statistics()
            assert stats['total_batches'] > 1
            
            # 验证批处理器统计
            if 'batch_processor' in stats:
                batch_stats = stats['batch_processor']
                assert batch_stats['total_batches'] > 1
                assert batch_stats['processed_items'] > 0
            
            await enhanced_postgres_logger.stop()
    
    @pytest.mark.asyncio
    async def test_connection_pool_health_monitoring(self, optimized_pool_config):
        """测试连接池健康监控"""
        with patch('psycopg2.connect') as mock_connect:
            # 模拟健康和不健康的连接
            healthy_conn = MagicMock()
            unhealthy_conn = MagicMock()
            unhealthy_conn.closed = 1  # 模拟关闭的连接
            
            mock_connect.side_effect = [healthy_conn, unhealthy_conn, healthy_conn]
            
            pool = ConnectionPool(
                connection_string="postgresql://test:test@localhost:5432/test",
                config=optimized_pool_config
            )
            
            # 等待健康检查运行
            await asyncio.sleep(1.0)
            
            # 验证健康监控
            stats = pool.get_stats()
            assert stats.total_connections > 0
            
            # 测试连接获取
            async with pool.get_connection() as conn:
                assert conn is not None
    
    @pytest.mark.asyncio
    async def test_batch_processing_monitoring(self, enhanced_postgres_logger):
        """测试批量处理监控"""
        with patch.object(enhanced_postgres_logger, '_process_batch', new_callable=AsyncMock) as mock_process:
            mock_process.return_value = True
            
            await enhanced_postgres_logger.start()
            
            # 添加不同优先级的数据
            high_priority_count = 0
            normal_priority_count = 0
            
            for i in range(100):
                priority = BatchPriority.HIGH if i % 5 == 0 else BatchPriority.NORMAL
                if priority == BatchPriority.HIGH:
                    high_priority_count += 1
                else:
                    normal_priority_count += 1
                
                await enhanced_postgres_logger.log_api_call(
                    api_log=APILog(
                        id=f"monitor-test-{i}",
                        timestamp=datetime.now(),
                        provider="openai",
                        model="gpt-4",
                        request_data='{"model": "gpt-4", "messages": []}',
                        response_data='{"choices": [{"message": {"content": "Hello"}}]}',
                        status_code=200,
                        error_message=None,
                        duration_ms=1000.0
                    ),
                    priority=priority
                )
            
            # 等待处理完成
            await asyncio.sleep(3.0)
            
            # 验证监控数据
            stats = enhanced_postgres_logger.get_statistics()
            assert stats['total_logs'] == 100
            assert stats['total_batches'] > 0
            
            # 验证连接池监控
            if 'connection_pool' in stats:
                pool_stats = stats['connection_pool']
                assert 'total_connections' in pool_stats
                assert 'active_connections' in pool_stats
                assert 'total_requests' in pool_stats
            
            await enhanced_postgres_logger.stop()
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_retry(self, enhanced_postgres_logger):
        """测试错误恢复和重试机制"""
        call_count = 0
        
        async def mock_process_batch_with_errors(items):
            nonlocal call_count
            call_count += 1
            
            # 前两次调用失败，第三次成功
            if call_count <= 2:
                raise Exception("模拟数据库错误")
            return True
        
        with patch.object(enhanced_postgres_logger, '_process_batch', 
                         side_effect=mock_process_batch_with_errors):
            await enhanced_postgres_logger.start()
            
            # 添加数据
            await enhanced_postgres_logger.log_api_call(
                api_log=APILog(
                    id="error-test-1",
                    timestamp=datetime.now(),
                    provider="openai",
                    model="gpt-4",
                    request_data='{"model": "gpt-4", "messages": []}',
                    response_data='{"choices": [{"message": {"content": "Hello"}}]}',
                    status_code=200,
                    error_message=None,
                    duration_ms=1000.0
                )
            )
            
            # 等待重试完成
            await asyncio.sleep(5.0)
            
            # 验证重试机制
            assert call_count >= 3  # 至少重试了3次
            
            stats = enhanced_postgres_logger.get_statistics()
            assert stats['retry_count'] > 0
            
            await enhanced_postgres_logger.stop()
    
    def test_optimized_batch_config_validation(self):
        """测试优化的批处理配置验证"""
        # 测试有效配置
        config = BatchConfig(
            min_batch_size=50,
            max_batch_size=500,
            flush_interval=2.0,
            adaptive_threshold=0.9
        )
        
        assert config.min_batch_size < config.max_batch_size
        assert config.flush_interval > 0
        assert 0 < config.adaptive_threshold <= 1.0
        
        # 测试配置优化建议
        assert config.min_batch_size >= 50, "建议最小批量大小至少为50以提高性能"
        assert config.max_batch_size <= 1000, "建议最大批量大小不超过1000以避免内存问题"
        assert config.flush_interval <= 5.0, "建议刷新间隔不超过5秒以保证实时性"
    
    def test_connection_pool_config_optimization(self):
        """测试连接池配置优化"""
        config = ConnectionPoolConfig(
            min_connections=10,
            max_connections=50,
            connection_timeout=15.0,
            idle_timeout=180.0,
            max_lifetime=1800.0
        )
        
        # 验证配置合理性
        assert config.min_connections > 0
        assert config.max_connections > config.min_connections
        assert config.connection_timeout > 0
        assert config.idle_timeout > config.connection_timeout
        assert config.max_lifetime > config.idle_timeout
        
        # 验证性能优化建议
        assert config.min_connections >= 5, "建议最小连接数至少为5"
        assert config.max_connections <= 100, "建议最大连接数不超过100"
        assert config.connection_timeout <= 30.0, "建议连接超时不超过30秒"