#!/usr/bin/env python3
"""
追踪信息存储测试模块

测试TracingInfoStorage类的各种功能，包括：
- 存储模式测试
- 查询功能测试
- 缓存机制测试
- 性能优化测试
- 健康检查测试

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import asdict

from harborai.core.tracing.storage import (
    TracingInfoStorage,
    StorageConfig,
    StorageMode,
    QueryFilter,
    QueryOptions,
    QueryOptimization,
    StorageMetrics
)
from harborai.core.tracing.data_collector import TracingRecord


class TestTracingInfoStorage:
    """追踪信息存储测试类"""
    
    @pytest.fixture
    def storage_config(self):
        """创建存储配置"""
        return StorageConfig(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            batch_size=10,
            compression_enabled=True,
            cache_size=100,
            archive_after_days=7,
            cleanup_after_days=30,
            connection_pool_size=5,
            max_overflow=10,
            query_timeout=10
        )
    
    @pytest.fixture
    def mock_async_engine(self):
        """模拟异步数据库引擎"""
        engine = AsyncMock()
        engine.dispose = AsyncMock()
        return engine
    
    @pytest.fixture
    def mock_session_factory(self):
        """模拟异步会话工厂"""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        
        factory = Mock()
        factory.return_value = session
        return factory, session
    
    @pytest.fixture
    def sample_tracing_record(self):
        """创建示例追踪记录"""
        return TracingRecord(
            hb_trace_id="hb_test_123",
            otel_trace_id="otel_test_456",
            span_id="span_789",
            parent_span_id="parent_span_101",
            operation_name="test_operation",
            service_name="test_service",
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1),
            duration_ms=1000.0,
            provider="test_provider",
            model="test_model",
            status="ok",
            error_message=None,
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            parsing_method="json",
            confidence=0.95,
            input_cost=0.01,
            output_cost=0.02,
            total_cost=0.03,
            currency="USD",
            pricing_source="api",
            tags={"test": "value"},
            logs=[{"timestamp": datetime.now().isoformat(), "message": "test log"}],
            created_at=datetime.now()
        )
    
    @pytest.mark.asyncio
    async def test_storage_initialization(self, storage_config):
        """测试存储系统初始化"""
        with patch('harborai.core.tracing.storage.create_async_engine') as mock_create_engine, \
             patch('harborai.core.tracing.storage.async_sessionmaker') as mock_sessionmaker:
            
            mock_engine = AsyncMock()
            mock_create_engine.return_value = mock_engine
            mock_sessionmaker.return_value = Mock()
            
            storage = TracingInfoStorage(storage_config)
            
            # 测试初始化前状态
            assert not storage._initialized
            assert storage.async_engine is None
            
            # 执行初始化
            await storage.initialize()
            
            # 验证初始化结果
            assert storage._initialized
            assert storage.async_engine is not None
            assert storage.async_session_factory is not None
            
            # 验证引擎创建参数
            mock_create_engine.assert_called_once()
            call_args = mock_create_engine.call_args
            assert call_args[0][0] == storage_config.database_url
            
            # 清理
            await storage.shutdown()
    
    @pytest.mark.asyncio
    async def test_real_time_storage(self, storage_config, sample_tracing_record):
        """测试实时存储模式"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 执行实时存储
        result = await storage.store_record(sample_tracing_record, StorageMode.REAL_TIME)
        
        # 验证结果
        assert result is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
        
        # 验证SQL参数
        call_args = mock_session.execute.call_args
        assert "INSERT INTO tracing_info" in str(call_args[0][0])
        
        data = call_args[0][1]
        assert data["hb_trace_id"] == sample_tracing_record.hb_trace_id
        assert data["operation_name"] == sample_tracing_record.operation_name
    
    @pytest.mark.asyncio
    async def test_batch_storage(self, storage_config, sample_tracing_record):
        """测试批量存储模式"""
        storage = TracingInfoStorage(storage_config)
        storage._initialized = True
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 添加多个记录到批量缓冲区
        for i in range(5):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id=f"otel_test_{i}",
                span_id=f"span_{i}",
                operation_name=f"test_operation_{i}",
                service_name="test_service",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(seconds=1),
                duration_ms=1000.0,
                provider="test_provider",
                model="test_model",
                status="ok",
                created_at=datetime.now()
            )
            result = await storage.store_record(record, StorageMode.BATCH)
            assert result is True
        
        # 验证批量缓冲区
        assert len(storage._batch_buffer) == 5
        
        # 手动刷新批量缓冲区
        await storage._flush_batch()
        
        # 验证数据库操作
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
        
        # 验证缓冲区已清空
        assert len(storage._batch_buffer) == 0
    
    @pytest.mark.asyncio
    async def test_compressed_storage(self, storage_config, sample_tracing_record):
        """测试压缩存储模式"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 执行压缩存储
        result = await storage.store_record(sample_tracing_record, StorageMode.COMPRESSED)
        
        # 验证结果
        assert result is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
        
        # 验证SQL参数
        call_args = mock_session.execute.call_args
        assert "INSERT INTO tracing_info_compressed" in str(call_args[0][0])
        
        data = call_args[0][1]
        assert data["hb_trace_id"] == sample_tracing_record.hb_trace_id
        assert "compressed_data" in data
        assert "original_size" in data
        assert "compressed_size" in data
    
    @pytest.mark.asyncio
    async def test_archived_storage(self, storage_config, sample_tracing_record):
        """测试归档存储模式"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 执行归档存储
        result = await storage.store_record(sample_tracing_record, StorageMode.ARCHIVED)
        
        # 验证结果
        assert result is True
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
        
        # 验证SQL参数
        call_args = mock_session.execute.call_args
        assert "INSERT INTO tracing_info_archive" in str(call_args[0][0])
        
        data = call_args[0][1]
        assert data["hb_trace_id"] == sample_tracing_record.hb_trace_id
        assert "archived_at" in data
    
    @pytest.mark.asyncio
    async def test_query_records_basic(self, storage_config):
        """测试基本查询功能"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话和查询结果
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 模拟查询结果
        mock_row = Mock()
        mock_row._mapping = {
            "hb_trace_id": "hb_test_123",
            "operation_name": "test_operation",
            "tags": '{"test": "value"}',
            "logs": '[{"message": "test log"}]'
        }
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        
        # 创建查询过滤器
        filter_obj = QueryFilter(
            hb_trace_id="hb_test_123",
            operation_name="test_operation"
        )
        
        options = QueryOptions(
            limit=10,
            offset=0,
            sort_by="start_time",
            sort_order="desc"
        )
        
        # 执行查询
        results = await storage.query_records(filter_obj, options)
        
        # 验证结果
        assert len(results) == 1
        assert results[0]["hb_trace_id"] == "hb_test_123"
        assert results[0]["operation_name"] == "test_operation"
        assert isinstance(results[0]["tags"], dict)
        assert isinstance(results[0]["logs"], list)
        
        # 验证SQL查询
        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args
        assert "SELECT * FROM tracing_info" in str(call_args[0][0])
        assert "hb_trace_id = :hb_trace_id" in str(call_args[0][0])
        assert "operation_name = :operation_name" in str(call_args[0][0])
    
    @pytest.mark.asyncio
    async def test_query_with_cache(self, storage_config):
        """测试查询缓存功能"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 模拟查询结果
        mock_row = Mock()
        mock_row._mapping = {"hb_trace_id": "hb_test_123"}
        
        mock_result = Mock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        
        filter_obj = QueryFilter(hb_trace_id="hb_test_123")
        options = QueryOptions()
        
        # 第一次查询
        results1 = await storage.query_records(filter_obj, options)
        assert len(results1) == 1
        assert mock_session.execute.call_count == 1
        
        # 第二次查询（应该使用缓存）
        results2 = await storage.query_records(filter_obj, options)
        assert len(results2) == 1
        assert results1 == results2
        assert mock_session.execute.call_count == 1  # 没有增加
        
        # 验证缓存命中率
        assert storage._metrics.cache_hit_rate > 0
    
    @pytest.mark.asyncio
    async def test_query_conditions_building(self, storage_config):
        """测试查询条件构建"""
        storage = TracingInfoStorage(storage_config)
        
        # 创建复杂的查询过滤器
        filter_obj = QueryFilter(
            hb_trace_id="hb_test_123",
            operation_name="test_operation",
            provider="test_provider",
            status="ok",
            start_time_from=datetime(2025, 1, 1),
            start_time_to=datetime(2025, 1, 31),
            duration_min=100.0,
            duration_max=5000.0,
            has_errors=False,
            cost_min=0.01,
            cost_max=1.0
        )
        
        # 构建查询条件
        conditions, params = storage._build_query_conditions(filter_obj)
        
        # 验证条件
        expected_conditions = [
            "hb_trace_id = :hb_trace_id",
            "operation_name = :operation_name",
            "provider = :provider",
            "status = :status",
            "start_time >= :start_time_from",
            "start_time <= :start_time_to",
            "duration_ms >= :duration_min",
            "duration_ms <= :duration_max",
            "error_message IS NULL",
            "total_cost >= :cost_min",
            "total_cost <= :cost_max"
        ]
        
        for condition in expected_conditions:
            assert condition in conditions
        
        # 验证参数
        assert params["hb_trace_id"] == "hb_test_123"
        assert params["operation_name"] == "test_operation"
        assert params["provider"] == "test_provider"
        assert params["status"] == "ok"
        assert params["duration_min"] == 100.0
        assert params["duration_max"] == 5000.0
        assert params["cost_min"] == 0.01
        assert params["cost_max"] == 1.0
    
    @pytest.mark.asyncio
    async def test_sort_clause_building(self, storage_config):
        """测试排序子句构建"""
        storage = TracingInfoStorage(storage_config)
        
        # 测试有效的排序字段
        sort_clause = storage._build_sort_clause("start_time", "desc")
        assert sort_clause == "ORDER BY start_time DESC"
        
        sort_clause = storage._build_sort_clause("duration_ms", "asc")
        assert sort_clause == "ORDER BY duration_ms ASC"
        
        # 测试无效的排序字段（应该回退到默认值）
        sort_clause = storage._build_sort_clause("invalid_field", "desc")
        assert sort_clause == "ORDER BY start_time DESC"
        
        # 测试无效的排序顺序（应该回退到默认值）
        sort_clause = storage._build_sort_clause("start_time", "invalid_order")
        assert sort_clause == "ORDER BY start_time DESC"
    
    @pytest.mark.asyncio
    async def test_cache_key_generation(self, storage_config):
        """测试缓存键生成"""
        storage = TracingInfoStorage(storage_config)
        
        filter_obj1 = QueryFilter(hb_trace_id="test_123")
        options1 = QueryOptions(limit=10)
        
        filter_obj2 = QueryFilter(hb_trace_id="test_123")
        options2 = QueryOptions(limit=10)
        
        filter_obj3 = QueryFilter(hb_trace_id="test_456")
        options3 = QueryOptions(limit=10)
        
        # 相同的过滤器和选项应该生成相同的缓存键
        key1 = storage._generate_cache_key(filter_obj1, options1)
        key2 = storage._generate_cache_key(filter_obj2, options2)
        assert key1 == key2
        
        # 不同的过滤器应该生成不同的缓存键
        key3 = storage._generate_cache_key(filter_obj3, options3)
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_cache_operations(self, storage_config):
        """测试缓存操作"""
        storage = TracingInfoStorage(storage_config)
        
        cache_key = "test_cache_key"
        test_result = [{"test": "data"}]
        
        # 测试缓存不存在
        cached_result = storage._get_cached_result(cache_key)
        assert cached_result is None
        
        # 测试缓存存储
        storage._cache_result(cache_key, test_result)
        cached_result = storage._get_cached_result(cache_key)
        assert cached_result == test_result
        
        # 测试缓存过期
        storage._query_cache[cache_key] = (
            datetime.now() - timedelta(hours=1),  # 过期时间
            test_result
        )
        cached_result = storage._get_cached_result(cache_key)
        assert cached_result is None
        assert cache_key not in storage._query_cache
    
    @pytest.mark.asyncio
    async def test_batch_auto_flush(self, storage_config, sample_tracing_record):
        """测试批量自动刷新"""
        # 设置小的批量大小
        storage_config.batch_size = 3
        storage = TracingInfoStorage(storage_config)
        storage._initialized = True
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 添加记录直到触发自动刷新
        for i in range(3):
            record = TracingRecord(
                hb_trace_id=f"hb_test_{i}",
                otel_trace_id=f"otel_test_{i}",
                span_id=f"span_{i}",
                operation_name=f"test_operation_{i}",
                service_name="test_service",
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(seconds=1),
                duration_ms=1000.0,
                provider="test_provider",
                model="test_model",
                status="ok",
                created_at=datetime.now()
            )
            await storage.store_record(record, StorageMode.BATCH)
        
        # 验证自动刷新
        mock_session.execute.assert_called_once()
        mock_session.commit.assert_called_once()
        assert len(storage._batch_buffer) == 0
    
    @pytest.mark.asyncio
    async def test_storage_error_handling(self, storage_config, sample_tracing_record):
        """测试存储错误处理"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话错误
        mock_factory, mock_session = self.mock_session_factory()
        mock_session.execute.side_effect = Exception("Database error")
        storage.async_session_factory = mock_factory
        
        # 测试实时存储错误处理
        result = await storage.store_record(sample_tracing_record, StorageMode.REAL_TIME)
        assert result is False
        mock_session.rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_query_error_handling(self, storage_config):
        """测试查询错误处理"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话错误
        mock_factory, mock_session = self.mock_session_factory()
        mock_session.execute.side_effect = Exception("Query error")
        storage.async_session_factory = mock_factory
        
        filter_obj = QueryFilter(hb_trace_id="test_123")
        results = await storage.query_records(filter_obj)
        
        # 应该返回空列表而不是抛出异常
        assert results == []
    
    @pytest.mark.asyncio
    async def test_metrics_update(self, storage_config):
        """测试指标更新"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 模拟查询结果
        count_result = Mock()
        count_result.fetchone.return_value = [1000]
        
        size_result = Mock()
        size_result.fetchone.return_value = [1024000]
        
        mock_session.execute.side_effect = [count_result, size_result]
        
        # 更新指标
        await storage._update_metrics()
        
        # 验证指标
        assert storage._metrics.total_records == 1000
        assert storage._metrics.storage_size_bytes == 1024000
        assert storage._last_metrics_update is not None
    
    @pytest.mark.asyncio
    async def test_storage_optimization(self, storage_config):
        """测试存储优化"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 执行存储优化
        result = await storage.optimize_storage()
        
        # 验证结果
        assert result["status"] == "success"
        assert "optimizations" in result
        assert "timestamp" in result
        
        # 验证执行了优化操作
        assert mock_session.execute.call_count >= 1
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check(self, storage_config):
        """测试健康检查"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 执行健康检查
        health_status = await storage.health_check()
        
        # 验证结果
        assert health_status["status"] == "healthy"
        assert "checks" in health_status
        assert "timestamp" in health_status
        assert health_status["checks"]["database"] == "healthy"
        assert "batch_buffer" in health_status["checks"]
        assert "cache_size" in health_status["checks"]
        assert "background_tasks" in health_status["checks"]
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, storage_config):
        """测试不健康状态的健康检查"""
        storage = TracingInfoStorage(storage_config)
        
        # 不设置数据库会话（模拟数据库不可用）
        storage.async_session_factory = None
        
        # 执行健康检查
        health_status = await storage.health_check()
        
        # 验证结果
        assert health_status["status"] == "unhealthy"
        assert health_status["checks"]["database"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_shutdown(self, storage_config):
        """测试关闭操作"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟初始化状态
        storage._initialized = True
        storage.async_engine = AsyncMock()
        
        # 模拟数据库会话工厂
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 清空批量缓冲区（避免JSON序列化错误）
        storage._batch_buffer = []
        
        # 模拟后台任务为异步任务
        async def dummy_task():
            await asyncio.sleep(0.1)
        
        mock_task1 = asyncio.create_task(dummy_task())
        mock_task2 = asyncio.create_task(dummy_task())
        
        # 立即取消任务以模拟正常的取消流程
        mock_task1.cancel()
        mock_task2.cancel()
        
        storage._background_tasks = [mock_task1, mock_task2]
        
        # 执行关闭
        await storage.shutdown()
        
        # 验证关闭操作
        assert storage._shutdown_event.is_set()
        assert mock_task1.cancelled()
        assert mock_task2.cancelled()
        storage.async_engine.dispose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_data_archiver(self, storage_config):
        """测试数据归档功能"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 模拟需要归档的数据
        old_date = datetime.now() - timedelta(days=10)
        mock_row = Mock()
        mock_row._mapping = {
            "id": 1,
            "hb_trace_id": "hb_old_123",
            "operation_name": "old_operation",
            "start_time": old_date,
            "created_at": old_date
        }
        
        select_result = Mock()
        select_result.fetchall.return_value = [mock_row]
        
        mock_session.execute.side_effect = [select_result, None, None]
        
        # 执行数据归档
        await storage._archive_old_data()
        
        # 验证执行了归档操作
        assert mock_session.execute.call_count == 3  # SELECT, INSERT, DELETE
        mock_session.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_prepare_record_data(self, storage_config, sample_tracing_record):
        """测试记录数据准备"""
        storage = TracingInfoStorage(storage_config)
        
        # 准备记录数据
        data = storage._prepare_record_data(sample_tracing_record)
        
        # 验证数据
        assert data["hb_trace_id"] == sample_tracing_record.hb_trace_id
        assert data["otel_trace_id"] == sample_tracing_record.otel_trace_id
        assert data["operation_name"] == sample_tracing_record.operation_name
        assert data["provider"] == sample_tracing_record.provider
        assert data["total_cost"] == sample_tracing_record.total_cost
        
        # 验证JSON序列化
        assert isinstance(data["tags"], str)
        assert isinstance(data["logs"], str)
        
        # 验证JSON内容
        tags = json.loads(data["tags"])
        logs = json.loads(data["logs"])
        assert tags == sample_tracing_record.tags
        assert logs == sample_tracing_record.logs
    
    @pytest.mark.asyncio
    async def test_cache_size_limit(self, storage_config):
        """测试缓存大小限制"""
        # 设置小的缓存大小
        storage_config.cache_size = 3
        storage = TracingInfoStorage(storage_config)
        
        # 添加超过限制的缓存项
        for i in range(5):
            cache_key = f"test_key_{i}"
            test_result = [{"test": f"data_{i}"}]
            storage._cache_result(cache_key, test_result)
        
        # 验证缓存大小限制
        assert len(storage._query_cache) <= storage_config.cache_size
    
    @pytest.mark.asyncio
    async def test_compression_metrics(self, storage_config, sample_tracing_record):
        """测试压缩指标"""
        storage = TracingInfoStorage(storage_config)
        
        # 模拟数据库会话
        mock_factory, mock_session = self.mock_session_factory()
        storage.async_session_factory = mock_factory
        
        # 初始压缩比
        initial_ratio = storage._metrics.compression_ratio
        
        # 执行压缩存储
        await storage.store_record(sample_tracing_record, StorageMode.COMPRESSED)
        
        # 验证压缩比已更新
        assert storage._metrics.compression_ratio != initial_ratio
        assert 0 <= storage._metrics.compression_ratio <= 1
    
    def mock_session_factory(self):
        """创建模拟会话工厂的辅助方法"""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=None)
        
        factory = Mock()
        factory.return_value = session
        return factory, session


class TestStorageConfig:
    """存储配置测试类"""
    
    def test_storage_config_defaults(self):
        """测试存储配置默认值"""
        config = StorageConfig(database_url="test://localhost/test")
        
        assert config.database_url == "test://localhost/test"
        assert config.batch_size == 1000
        assert config.compression_enabled is True
        assert config.cache_size == 10000
        assert config.archive_after_days == 30
        assert config.cleanup_after_days == 90
        assert config.connection_pool_size == 20
        assert config.max_overflow == 30
        assert config.query_timeout == 30
        assert config.enable_partitioning is True
    
    def test_storage_config_custom_values(self):
        """测试存储配置自定义值"""
        config = StorageConfig(
            database_url="custom://localhost/custom",
            batch_size=500,
            compression_enabled=False,
            cache_size=5000,
            archive_after_days=15,
            cleanup_after_days=45,
            connection_pool_size=10,
            max_overflow=15,
            query_timeout=15,
            enable_partitioning=False
        )
        
        assert config.database_url == "custom://localhost/custom"
        assert config.batch_size == 500
        assert config.compression_enabled is False
        assert config.cache_size == 5000
        assert config.archive_after_days == 15
        assert config.cleanup_after_days == 45
        assert config.connection_pool_size == 10
        assert config.max_overflow == 15
        assert config.query_timeout == 15
        assert config.enable_partitioning is False


class TestQueryFilter:
    """查询过滤器测试类"""
    
    def test_query_filter_defaults(self):
        """测试查询过滤器默认值"""
        filter_obj = QueryFilter()
        
        assert filter_obj.hb_trace_id is None
        assert filter_obj.otel_trace_id is None
        assert filter_obj.span_id is None
        assert filter_obj.operation_name is None
        assert filter_obj.service_name is None
        assert filter_obj.provider is None
        assert filter_obj.model is None
        assert filter_obj.status is None
        assert filter_obj.start_time_from is None
        assert filter_obj.start_time_to is None
        assert filter_obj.duration_min is None
        assert filter_obj.duration_max is None
        assert filter_obj.has_errors is None
        assert filter_obj.cost_min is None
        assert filter_obj.cost_max is None
        assert filter_obj.tags is None
    
    def test_query_filter_custom_values(self):
        """测试查询过滤器自定义值"""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        tags = {"env": "test"}
        
        filter_obj = QueryFilter(
            hb_trace_id="hb_test_123",
            otel_trace_id="otel_test_456",
            span_id="span_789",
            operation_name="test_operation",
            service_name="test_service",
            provider="test_provider",
            model="test_model",
            status="ok",
            start_time_from=start_time,
            start_time_to=end_time,
            duration_min=100.0,
            duration_max=5000.0,
            has_errors=False,
            cost_min=0.01,
            cost_max=1.0,
            tags=tags
        )
        
        assert filter_obj.hb_trace_id == "hb_test_123"
        assert filter_obj.otel_trace_id == "otel_test_456"
        assert filter_obj.span_id == "span_789"
        assert filter_obj.operation_name == "test_operation"
        assert filter_obj.service_name == "test_service"
        assert filter_obj.provider == "test_provider"
        assert filter_obj.model == "test_model"
        assert filter_obj.status == "ok"
        assert filter_obj.start_time_from == start_time
        assert filter_obj.start_time_to == end_time
        assert filter_obj.duration_min == 100.0
        assert filter_obj.duration_max == 5000.0
        assert filter_obj.has_errors is False
        assert filter_obj.cost_min == 0.01
        assert filter_obj.cost_max == 1.0
        assert filter_obj.tags == tags


class TestQueryOptions:
    """查询选项测试类"""
    
    def test_query_options_defaults(self):
        """测试查询选项默认值"""
        options = QueryOptions()
        
        assert options.limit == 100
        assert options.offset == 0
        assert options.sort_by == "start_time"
        assert options.sort_order == "desc"
        assert options.include_logs is True
        assert options.include_tags is True
        assert options.optimization == QueryOptimization.INDEX_SCAN
    
    def test_query_options_custom_values(self):
        """测试查询选项自定义值"""
        options = QueryOptions(
            limit=50,
            offset=10,
            sort_by="duration_ms",
            sort_order="asc",
            include_logs=False,
            include_tags=False,
            optimization=QueryOptimization.FULL_SCAN
        )
        
        assert options.limit == 50
        assert options.offset == 10
        assert options.sort_by == "duration_ms"
        assert options.sort_order == "asc"
        assert options.include_logs is False
        assert options.include_tags is False
        assert options.optimization == QueryOptimization.FULL_SCAN


class TestStorageMetrics:
    """存储指标测试类"""
    
    def test_storage_metrics_defaults(self):
        """测试存储指标默认值"""
        metrics = StorageMetrics()
        
        assert metrics.total_records == 0
        assert metrics.storage_size_bytes == 0
        assert metrics.compression_ratio == 0.0
        assert metrics.query_performance_ms == 0.0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.index_efficiency == 0.0
        assert metrics.partition_distribution == {}
    
    def test_storage_metrics_custom_values(self):
        """测试存储指标自定义值"""
        partition_dist = {"2025-01": 1000, "2025-02": 1500}
        
        metrics = StorageMetrics(
            total_records=2500,
            storage_size_bytes=1024000,
            compression_ratio=0.75,
            query_performance_ms=150.5,
            cache_hit_rate=0.85,
            index_efficiency=0.92,
            partition_distribution=partition_dist
        )
        
        assert metrics.total_records == 2500
        assert metrics.storage_size_bytes == 1024000
        assert metrics.compression_ratio == 0.75
        assert metrics.query_performance_ms == 150.5
        assert metrics.cache_hit_rate == 0.85
        assert metrics.index_efficiency == 0.92
        assert metrics.partition_distribution == partition_dist