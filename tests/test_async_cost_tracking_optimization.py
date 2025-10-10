"""异步成本追踪优化测试

测试异步成本追踪的性能优化效果，确保不阻塞主流程。
"""

import asyncio
import pytest
import pytest_asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any
import uuid

from harborai.core.async_cost_tracking import AsyncCostTracker, get_async_cost_tracker
from harborai.core.unified_decorators import UnifiedDecorator, DecoratorConfig, DecoratorMode
from harborai.config.settings import get_settings


class TestAsyncCostTrackingOptimization:
    """异步成本追踪优化测试类"""
    
    @pytest_asyncio.fixture
    async def async_tracker(self):
        """创建异步成本追踪器实例"""
        tracker = AsyncCostTracker()
        yield tracker
        # 清理
        await tracker.close()
    
    @pytest.fixture
    def mock_settings(self):
        """模拟设置"""
        settings = Mock()
        settings.performance_mode = "balanced"
        settings.fast_path_skip_cost_tracking = False
        settings.enable_cost_tracking = True
        settings.enable_async_decorators = True
        settings.enable_fast_path = True
        settings.fast_path_models = ["gpt-3.5-turbo", "gpt-4o-mini"]
        settings.fast_path_max_tokens = None  # 无限制，由模型厂商控制
        settings.debug = False
        settings.enable_prometheus_metrics = True
        settings.enable_opentelemetry = True
        settings.enable_postgres_logging = True
        settings.enable_detailed_logging = True
        settings.enable_plugin_preload = True
        settings.enable_response_cache = True
        settings.enable_token_cache = True
        settings.enable_background_tasks = True
        settings.background_task_workers = 2
        settings.response_cache_ttl = 600
        settings.token_cache_ttl = 300
        settings.cache_cleanup_interval = 300
        settings.plugin_cache_size = 100
        settings.plugin_directories = ["harborai.core.plugins"]
        # 添加 is_fast_path_enabled 方法
        def is_fast_path_enabled(model, max_tokens=None):
            if settings.performance_mode == "fast":
                return True
            elif settings.performance_mode == "full":
                return False
            # balanced 模式逻辑
            if model in settings.fast_path_models:
                if max_tokens is None or max_tokens <= settings.fast_path_max_tokens:
                    return True
            return False
        settings.is_fast_path_enabled = is_fast_path_enabled
        return settings
    
    @pytest.mark.asyncio
    async def test_async_cost_tracking_non_blocking(self, async_tracker):
        """测试异步成本追踪不阻塞主流程"""
        # 记录开始时间
        start_time = time.time()
        
        # 异步追踪多个API调用
        tasks = []
        for i in range(10):
            task = asyncio.create_task(async_tracker.track_api_call_async(
                model=f"gpt-4-{i}",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
                cost=0.01,
                duration=0.1,
                success=True,
                trace_id=f"trace-{i}"
            ))
            tasks.append(task)
        
        # 等待所有追踪任务完成
        await asyncio.gather(*tasks)
        
        # 检查执行时间（应该很快，因为是异步的）
        execution_time = time.time() - start_time
        assert execution_time < 0.1, f"异步成本追踪耗时过长: {execution_time}s"
    
    @pytest.mark.asyncio
    async def test_batch_processing_efficiency(self, async_tracker):
        """测试批量处理效率"""
        # 设置较小的批量大小进行测试
        async_tracker._batch_size = 3
        
        # 添加多个调用，触发批量处理
        for i in range(5):
            await async_tracker.track_api_call_async(
                model=f"gpt-3.5-turbo-{i}",
                provider="openai",
                input_tokens=50,
                output_tokens=25,
                cost=0.005,
                duration=0.05,
                success=True,
                trace_id=f"batch-trace-{i}"
            )
        
        # 等待批量处理完成
        await async_tracker.flush_pending()
        
        # 验证所有调用都被处理
        assert len(async_tracker._pending_calls) == 0
        assert len(async_tracker._sync_tracker.api_calls) == 5
    
    @pytest.mark.asyncio
    async def test_fast_path_skip_cost_tracking(self, async_tracker, mock_settings):
        """测试快速路径跳过成本追踪"""
        # 设置快速路径模式
        mock_settings.performance_mode = "fast"
        mock_settings.fast_path_skip_cost_tracking = True
        
        with patch('harborai.core.async_cost_tracking.get_settings', return_value=mock_settings), \
             patch('harborai.config.settings.get_settings', return_value=mock_settings):
            # 创建新的追踪器实例
            fast_tracker = AsyncCostTracker()
            
            # 尝试追踪API调用（使用快速路径支持的模型）
            await fast_tracker.track_api_call_async(
                model="gpt-3.5-turbo",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
                cost=0.01,
                duration=0.1,
                success=True,
                trace_id="fast-trace"
            )
            
            # 验证没有调用被记录（因为快速路径跳过了）
            assert len(fast_tracker._pending_calls) == 0
            assert len(fast_tracker._sync_tracker.api_calls) == 0
    
    @pytest.mark.asyncio
    async def test_decorator_async_cost_tracking_integration(self, mock_settings):
        """测试装饰器与异步成本追踪的集成"""
        with patch('harborai.core.unified_decorators.get_settings', return_value=mock_settings):
            # 创建启用异步成本追踪的装饰器
            config = DecoratorConfig(
                mode=DecoratorMode.CUSTOM,
                enable_cost_tracking=True,
                async_cost_tracking=True
            )
            decorator = UnifiedDecorator(config)
            
            # 模拟API函数
            @decorator
            async def mock_api_call(model: str, **kwargs):
                """模拟API调用"""
                # 模拟返回带有usage信息的结果
                result = Mock()
                result.usage = Mock()
                result.usage.prompt_tokens = 100
                result.usage.completion_tokens = 50
                return result
            
            # 调用被装饰的函数
            start_time = time.time()
            result = await mock_api_call(model="gpt-4", trace_id="decorator-test")
            execution_time = time.time() - start_time
            
            # 验证函数执行很快（异步成本追踪不阻塞）
            assert execution_time < 0.1, f"装饰器执行耗时过长: {execution_time}s"
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_error_handling_in_async_tracking(self, async_tracker):
        """测试异步追踪中的错误处理"""
        # 模拟处理过程中的错误
        with patch.object(async_tracker, '_process_calls_sync', side_effect=Exception("处理错误")):
            # 添加一个调用
            await async_tracker.track_api_call_async(
                model="gpt-4",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
                cost=0.01,
                duration=0.1,
                success=True,
                trace_id="error-test"
            )
            
            # 尝试处理批量（应该失败但不崩溃）
            await async_tracker._process_batch()
            
            # 验证调用被重新加入队列
            assert len(async_tracker._pending_calls) > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_tracking_performance(self, async_tracker):
        """测试并发追踪性能"""
        # 创建大量并发追踪任务
        async def track_single_call(i):
            await async_tracker.track_api_call_async(
                model=f"model-{i}",
                provider="test",
                input_tokens=10,
                output_tokens=5,
                cost=0.001,
                duration=0.01,
                success=True,
                trace_id=f"concurrent-{i}"
            )
        
        # 并发执行100个追踪任务
        start_time = time.time()
        tasks = [track_single_call(i) for i in range(100)]
        await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # 验证并发性能
        assert execution_time < 1.0, f"并发追踪耗时过长: {execution_time}s"
        
        # 等待所有批量处理完成
        await async_tracker.flush_pending()
        
        # 验证所有调用都被处理
        assert len(async_tracker._sync_tracker.api_calls) == 100
    
    @pytest.mark.asyncio
    async def test_memory_efficiency_with_large_batches(self, async_tracker):
        """测试大批量处理的内存效率"""
        # 设置较大的批量大小
        async_tracker._batch_size = 50
        
        # 添加大量调用
        for i in range(200):
            await async_tracker.track_api_call_async(
                model="gpt-3.5-turbo",
                provider="openai",
                input_tokens=20,
                output_tokens=10,
                cost=0.002,
                duration=0.02,
                success=True,
                trace_id=f"memory-test-{i}"
            )
        
        # 验证待处理队列不会无限增长
        assert len(async_tracker._pending_calls) <= async_tracker._batch_size
        
        # 刷新所有待处理的调用
        await async_tracker.flush_pending()
        
        # 验证内存被正确清理
        assert len(async_tracker._pending_calls) == 0
    
    def test_global_async_tracker_singleton(self):
        """测试全局异步追踪器单例模式"""
        tracker1 = get_async_cost_tracker()
        tracker2 = get_async_cost_tracker()
        
        # 验证返回同一个实例
        assert tracker1 is tracker2
        assert isinstance(tracker1, AsyncCostTracker)