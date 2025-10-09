"""异步成本追踪模块测试

测试异步成本追踪功能，包括批量处理、并发安全、性能优化等。
遵循VIBE编码规范，使用TDD方法，目标覆盖率>90%。
"""

import pytest
import asyncio
import time
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

from harborai.core.async_cost_tracking import (
    AsyncCostTracker, get_async_cost_tracker, cleanup_async_cost_tracker
)
from harborai.core.cost_tracking import (
    TokenUsage, CostBreakdown, ApiCall, CostTracker
)


class TestAsyncCostTracker:
    """AsyncCostTracker类测试"""
    
    @pytest.fixture
    def async_tracker(self):
        """创建异步成本追踪器实例"""
        return AsyncCostTracker()
    
    @pytest.fixture
    def mock_settings(self):
        """Mock设置对象"""
        settings = Mock()
        settings.fast_path_skip_cost_tracking = False
        return settings
    
    @pytest.fixture
    def mock_perf_config(self):
        """Mock性能配置对象"""
        perf_config = Mock()
        perf_config.get_middleware_config.return_value = {
            'cost_tracking_middleware': True
        }
        perf_config.should_use_fast_path.return_value = False
        return perf_config
    
    def test_async_tracker_creation(self, async_tracker):
        """测试异步追踪器创建"""
        assert async_tracker is not None
        assert hasattr(async_tracker, '_sync_tracker')
        assert hasattr(async_tracker, '_executor')
        assert hasattr(async_tracker, '_pending_calls')
        assert hasattr(async_tracker, '_batch_size')
        assert hasattr(async_tracker, '_batch_timeout')
        assert isinstance(async_tracker._executor, ThreadPoolExecutor)
        assert async_tracker._batch_size == 10
        assert async_tracker._batch_timeout == 5.0
    
    @pytest.mark.asyncio
    async def test_track_api_call_async_basic(self, async_tracker, mock_settings, mock_perf_config):
        """测试基本异步API调用追踪"""
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config):
            
            await async_tracker.track_api_call_async(
                model="gpt-3.5-turbo",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
                duration=1.5,
                success=True,
                user_id="test-user",
                trace_id="test-trace"
            )
            
            # 验证调用被添加到待处理队列
            assert len(async_tracker._pending_calls) == 1
            
            call_data = async_tracker._pending_calls[0]
            assert call_data['model'] == "gpt-3.5-turbo"
            assert call_data['provider'] == "openai"
            assert call_data['input_tokens'] == 100
            assert call_data['output_tokens'] == 50
            assert call_data['cost'] == 0.001
            assert call_data['duration'] == 1.5
            assert call_data['success'] is True
            assert call_data['user_id'] == "test-user"
            assert call_data['trace_id'] == "test-trace"
    
    @pytest.mark.asyncio
    async def test_track_api_call_async_with_metadata(self, async_tracker, mock_settings, mock_perf_config):
        """测试带元数据的异步API调用追踪"""
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config):
            
            metadata = {"env": "test", "version": "1.0"}
            
            await async_tracker.track_api_call_async(
                model="gpt-4",
                provider="openai",
                input_tokens=200,
                output_tokens=100,
                cost=0.005,
                duration=2.0,
                success=True,
                **metadata
            )
            
            assert len(async_tracker._pending_calls) == 1
            call_data = async_tracker._pending_calls[0]
            assert call_data['metadata'] == metadata
    
    @pytest.mark.asyncio
    async def test_track_api_call_async_disabled_middleware(self, async_tracker, mock_settings):
        """测试中间件禁用时的行为"""
        mock_perf_config = Mock()
        mock_perf_config.get_middleware_config.return_value = {
            'cost_tracking_middleware': False
        }
        
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config):
            
            await async_tracker.track_api_call_async(
                model="gpt-3.5-turbo",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
                duration=1.5
            )
            
            # 中间件禁用时，不应该添加到队列
            assert len(async_tracker._pending_calls) == 0
    
    @pytest.mark.asyncio
    async def test_track_api_call_async_fast_path_skip(self, async_tracker):
        """测试快速路径跳过成本追踪"""
        mock_settings = Mock()
        mock_settings.fast_path_skip_cost_tracking = True
        
        mock_perf_config = Mock()
        mock_perf_config.get_middleware_config.return_value = {
            'cost_tracking_middleware': True
        }
        mock_perf_config.should_use_fast_path.return_value = True
        
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config):
            
            await async_tracker.track_api_call_async(
                model="gpt-3.5-turbo",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
                duration=1.5
            )
            
            # 快速路径跳过时，不应该添加到队列
            assert len(async_tracker._pending_calls) == 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_by_size(self, async_tracker, mock_settings, mock_perf_config):
        """测试按批次大小触发批量处理"""
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config), \
             patch.object(async_tracker, '_process_batch', new_callable=AsyncMock) as mock_process:
            
            # 设置较小的批次大小
            async_tracker._batch_size = 3
            
            # 添加调用直到触发批处理
            for i in range(3):
                await async_tracker.track_api_call_async(
                    model=f"model-{i}",
                    provider="openai",
                    input_tokens=100,
                    output_tokens=50,
                    cost=0.001,
                    duration=1.5
                )
            
            # 验证批处理被调用
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_processing_by_timeout(self, async_tracker, mock_settings, mock_perf_config):
        """测试按超时时间触发批量处理"""
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config), \
             patch.object(async_tracker, '_process_batch', new_callable=AsyncMock) as mock_process:
            
            # 设置较短的超时时间
            async_tracker._batch_timeout = 0.1
            async_tracker._last_batch_time = time.time() - 1.0  # 模拟超时
            
            await async_tracker.track_api_call_async(
                model="gpt-3.5-turbo",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
                duration=1.5
            )
            
            # 验证批处理被调用
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_batch_empty(self, async_tracker):
        """测试处理空批次"""
        # 确保待处理列表为空
        async_tracker._pending_calls.clear()
        
        # 处理空批次应该不会出错
        await async_tracker._process_batch()
        
        assert len(async_tracker._pending_calls) == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self, async_tracker):
        """测试成功的批量处理"""
        # 添加测试数据
        test_calls = [
            {
                'model': 'gpt-3.5-turbo',
                'provider': 'openai',
                'input_tokens': 100,
                'output_tokens': 50,
                'cost': 0.001,
                'duration': 1.5,
                'success': True,
                'user_id': 'test-user',
                'trace_id': 'test-trace',
                'timestamp': datetime.now(),
                'metadata': {}
            }
        ]
        
        async_tracker._pending_calls = test_calls.copy()
        
        with patch.object(async_tracker, '_process_calls_sync') as mock_process_sync:
            await async_tracker._process_batch()
            
            # 验证同步处理被调用
            mock_process_sync.assert_called_once()
            
            # 验证待处理列表被清空
            assert len(async_tracker._pending_calls) == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_failure(self, async_tracker):
        """测试批量处理失败的情况"""
        # 添加测试数据
        test_calls = [
            {
                'model': 'gpt-3.5-turbo',
                'provider': 'openai',
                'input_tokens': 100,
                'output_tokens': 50,
                'cost': 0.001,
                'duration': 1.5,
                'success': True,
                'timestamp': datetime.now(),
                'metadata': {}
            }
        ]
        
        async_tracker._pending_calls = test_calls.copy()
        
        with patch.object(async_tracker, '_process_calls_sync', side_effect=Exception("处理失败")):
            await async_tracker._process_batch()
            
            # 验证失败时调用被重新加入队列
            assert len(async_tracker._pending_calls) == 1
    
    def test_process_calls_sync(self, async_tracker):
        """测试同步处理调用列表"""
        test_calls = [
            {
                'model': 'gpt-3.5-turbo',
                'provider': 'openai',
                'input_tokens': 100,
                'output_tokens': 50,
                'cost': 0.001,
                'duration': 1.5,
                'success': True,
                'user_id': 'test-user',
                'trace_id': 'test-trace',
                'timestamp': datetime.now(),
                'metadata': {'env': 'test'}
            }
        ]
        
        # 确保同步追踪器的api_calls列表为空
        async_tracker._sync_tracker.api_calls.clear()
        
        async_tracker._process_calls_sync(test_calls)
        
        # 验证API调用被添加到同步追踪器
        assert len(async_tracker._sync_tracker.api_calls) == 1
        
        api_call = async_tracker._sync_tracker.api_calls[0]
        assert api_call.model == 'gpt-3.5-turbo'
        assert api_call.provider == 'openai'
        assert api_call.token_usage.input_tokens == 100
        assert api_call.token_usage.output_tokens == 50
        assert api_call.duration == 1.5
        assert api_call.status == "success"
        assert api_call.user_id == 'test-user'
        assert api_call.tags == {'env': 'test'}
    
    def test_process_calls_sync_with_error(self, async_tracker):
        """测试同步处理中的错误处理"""
        test_calls = [
            {
                'model': 'gpt-3.5-turbo',
                'provider': 'openai',
                'input_tokens': 100,
                'output_tokens': 50,
                'cost': 0.001,
                'duration': 1.5,
                'success': False,  # 失败的调用
                'timestamp': datetime.now(),
                'metadata': {}
            }
        ]
        
        async_tracker._sync_tracker.api_calls.clear()
        
        # 应该不会抛出异常
        async_tracker._process_calls_sync(test_calls)
        
        # 验证失败的调用也被记录
        assert len(async_tracker._sync_tracker.api_calls) == 1
        api_call = async_tracker._sync_tracker.api_calls[0]
        assert api_call.status == "error"
    
    @pytest.mark.asyncio
    async def test_get_cost_summary(self, async_tracker):
        """测试获取成本摘要"""
        # Mock同步追踪器的get_cost_summary方法
        expected_summary = {
            'total_cost': 0.001,
            'total_tokens': 150,
            'total_calls': 1
        }
        
        with patch.object(async_tracker._sync_tracker, 'get_cost_summary', return_value=expected_summary), \
             patch.object(async_tracker, 'flush_pending', new_callable=AsyncMock):
            
            summary = await async_tracker.get_cost_summary()
            
            assert summary == expected_summary
    
    @pytest.mark.asyncio
    async def test_flush_pending(self, async_tracker):
        """测试刷新待处理调用"""
        # 添加待处理调用
        async_tracker._pending_calls = [
            {
                'model': 'gpt-3.5-turbo',
                'provider': 'openai',
                'input_tokens': 100,
                'output_tokens': 50,
                'cost': 0.001,
                'duration': 1.5,
                'timestamp': datetime.now(),
                'metadata': {}
            }
        ]
        
        with patch.object(async_tracker, '_process_batch', new_callable=AsyncMock) as mock_process:
            await async_tracker.flush_pending()
            
            # 验证批处理被调用
            mock_process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_flush_pending_empty(self, async_tracker):
        """测试刷新空的待处理列表"""
        async_tracker._pending_calls.clear()
        
        with patch.object(async_tracker, '_process_batch', new_callable=AsyncMock) as mock_process:
            await async_tracker.flush_pending()
            
            # 空列表时不应该调用批处理
            mock_process.assert_not_called()
    
    def test_track_sync(self, async_tracker):
        """测试同步版本的成本追踪"""
        async_tracker._pending_calls.clear()
        
        async_tracker.track_sync(
            trace_id="sync-trace",
            function_name="test_function",
            model="gpt-3.5-turbo",
            input_tokens=100,
            output_tokens=50
        )
        
        # 验证调用被添加到待处理队列
        assert len(async_tracker._pending_calls) == 1
        
        call_info = async_tracker._pending_calls[0]
        assert call_info['trace_id'] == "sync-trace"
        assert call_info['function_name'] == "test_function"
        assert call_info['model'] == "gpt-3.5-turbo"
        assert call_info['input_tokens'] == 100
        assert call_info['output_tokens'] == 50
    
    def test_track_sync_batch_trigger(self, async_tracker):
        """测试同步追踪触发批处理"""
        async_tracker._pending_calls.clear()
        async_tracker._batch_size = 2
        
        with patch('threading.Thread') as mock_thread:
            # 添加足够的调用触发批处理
            for i in range(2):
                async_tracker.track_sync(
                    trace_id=f"sync-trace-{i}",
                    function_name="test_function",
                    model="gpt-3.5-turbo"
                )
            
            # 验证后台线程被创建
            mock_thread.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close(self, async_tracker):
        """测试关闭异步追踪器"""
        with patch.object(async_tracker, 'flush_pending', new_callable=AsyncMock) as mock_flush, \
             patch.object(async_tracker._executor, 'shutdown') as mock_shutdown:
            
            await async_tracker.close()
            
            # 验证刷新和关闭被调用
            mock_flush.assert_called_once()
            mock_shutdown.assert_called_once_with(wait=True)
    
    def test_destructor(self, async_tracker):
        """测试析构函数"""
        with patch.object(async_tracker._executor, 'shutdown') as mock_shutdown:
            async_tracker.__del__()
            
            # 验证线程池被关闭
            mock_shutdown.assert_called_once_with(wait=False)


class TestAsyncCostTrackerConcurrency:
    """异步成本追踪器并发测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_tracking(self):
        """测试并发追踪"""
        async_tracker = AsyncCostTracker()
        
        mock_settings = Mock()
        mock_settings.fast_path_skip_cost_tracking = False
        
        mock_perf_config = Mock()
        mock_perf_config.get_middleware_config.return_value = {
            'cost_tracking_middleware': True
        }
        mock_perf_config.should_use_fast_path.return_value = False
        
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config):
            
            # 创建多个并发任务
            tasks = []
            for i in range(10):
                task = async_tracker.track_api_call_async(
                    model=f"model-{i}",
                    provider="openai",
                    input_tokens=100 + i,
                    output_tokens=50 + i,
                    cost=0.001 * (i + 1),
                    duration=1.0 + i * 0.1
                )
                tasks.append(task)
            
            # 等待所有任务完成
            await asyncio.gather(*tasks)
            
            # 验证所有调用都被记录
            assert len(async_tracker._pending_calls) <= 10  # 可能有些已经被批处理
    
    @pytest.mark.asyncio
    async def test_concurrent_batch_processing(self):
        """测试并发批处理"""
        async_tracker = AsyncCostTracker()
        async_tracker._batch_size = 5
        
        mock_settings = Mock()
        mock_settings.fast_path_skip_cost_tracking = False
        
        mock_perf_config = Mock()
        mock_perf_config.get_middleware_config.return_value = {
            'cost_tracking_middleware': True
        }
        mock_perf_config.should_use_fast_path.return_value = False
        
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config):
            
            # 快速添加多个调用以触发多次批处理
            tasks = []
            for i in range(15):  # 3批次
                task = async_tracker.track_api_call_async(
                    model=f"model-{i}",
                    provider="openai",
                    input_tokens=100,
                    output_tokens=50,
                    cost=0.001,
                    duration=1.0
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            # 刷新剩余的调用
            await async_tracker.flush_pending()
            
            # 验证所有调用都被处理
            assert len(async_tracker._pending_calls) == 0


class TestGlobalAsyncCostTracker:
    """全局异步成本追踪器测试"""
    
    def test_get_async_cost_tracker_singleton(self):
        """测试全局追踪器单例模式"""
        tracker1 = get_async_cost_tracker()
        tracker2 = get_async_cost_tracker()
        
        # 应该返回同一个实例
        assert tracker1 is tracker2
        assert isinstance(tracker1, AsyncCostTracker)
    
    @pytest.mark.asyncio
    async def test_cleanup_async_cost_tracker(self):
        """测试清理全局追踪器"""
        # 获取全局追踪器
        tracker = get_async_cost_tracker()
        
        with patch.object(tracker, 'close', new_callable=AsyncMock) as mock_close:
            await cleanup_async_cost_tracker()
            
            # 验证close方法被调用
            mock_close.assert_called_once()
        
        # 获取新的追踪器应该是不同的实例
        new_tracker = get_async_cost_tracker()
        assert new_tracker is not tracker


class TestAsyncCostTrackerPerformance:
    """异步成本追踪器性能测试"""
    
    @pytest.mark.asyncio
    async def test_batch_processing_performance(self):
        """测试批处理性能"""
        async_tracker = AsyncCostTracker()
        async_tracker._batch_size = 100  # 较大的批次
        
        mock_settings = Mock()
        mock_settings.fast_path_skip_cost_tracking = False
        
        mock_perf_config = Mock()
        mock_perf_config.get_middleware_config.return_value = {
            'cost_tracking_middleware': True
        }
        mock_perf_config.should_use_fast_path.return_value = False
        
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config):
            
            start_time = time.time()
            
            # 添加大量调用
            tasks = []
            for i in range(50):
                task = async_tracker.track_api_call_async(
                    model="gpt-3.5-turbo",
                    provider="openai",
                    input_tokens=100,
                    output_tokens=50,
                    cost=0.001,
                    duration=1.0
                )
                tasks.append(task)
            
            await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 验证性能（应该很快完成）
            assert duration < 1.0  # 应该在1秒内完成


class TestAsyncCostTrackerEdgeCases:
    """异步成本追踪器边界情况测试"""
    
    @pytest.mark.asyncio
    async def test_middleware_disabled(self):
        """测试中间件禁用情况"""
        async_tracker = AsyncCostTracker()
        
        mock_settings = Mock()
        mock_settings.fast_path_skip_cost_tracking = False
        
        mock_perf_config = Mock()
        mock_perf_config.get_middleware_config.return_value = {
            'cost_tracking_middleware': False  # 禁用成本追踪中间件
        }
        mock_perf_config.should_use_fast_path.return_value = False
        
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config):
            
            await async_tracker.track_api_call_async(
                model="gpt-3.5-turbo",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
                duration=1.0
            )
            
            # 中间件禁用时，不应该添加到队列
            assert len(async_tracker._pending_calls) == 0
    
    @pytest.mark.asyncio
    async def test_fast_path_with_skip_enabled(self):
        """测试快速路径且跳过成本追踪启用的情况"""
        async_tracker = AsyncCostTracker()
        
        mock_settings = Mock()
        mock_settings.fast_path_skip_cost_tracking = True  # 启用跳过
        
        mock_perf_config = Mock()
        mock_perf_config.get_middleware_config.return_value = {
            'cost_tracking_middleware': True
        }
        mock_perf_config.should_use_fast_path.return_value = True  # 使用快速路径
        
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config):
            
            await async_tracker.track_api_call_async(
                model="gpt-3.5-turbo",
                provider="openai",
                input_tokens=100,
                output_tokens=50,
                cost=0.001,
                duration=1.0
            )
            
            # 快速路径跳过时，不应该添加到队列
            assert len(async_tracker._pending_calls) == 0
    
    def test_process_calls_sync_exception(self):
        """测试同步处理调用时的异常情况"""
        async_tracker = AsyncCostTracker()
        
        # 创建会导致异常的调用数据
        call_data = {
            'model': 'gpt-3.5-turbo',
            'provider': 'openai',
            'input_tokens': 100,
            'output_tokens': 50,
            'cost': 0.001,
            'duration': 1.0,
            'success': True,
            'timestamp': datetime.now(),
            'metadata': {}
        }
        
        # Mock同步追踪器的_update_cost_stats方法抛出异常
        with patch.object(async_tracker._sync_tracker, '_update_cost_stats', side_effect=Exception("测试异常")), \
             patch('harborai.core.async_cost_tracking.logger') as mock_logger:
            
            async_tracker._process_calls_sync([call_data])
            
            # 验证异常被记录
            mock_logger.error.assert_called()
    
    @pytest.mark.asyncio
    async def test_batch_processing_exception(self):
        """测试批处理异常情况"""
        async_tracker = AsyncCostTracker()
        
        # 添加测试数据
        test_calls = [
            {
                'model': 'gpt-3.5-turbo',
                'provider': 'openai',
                'input_tokens': 100,
                'output_tokens': 50,
                'cost': 0.001,
                'duration': 1.0,
                'success': True,
                'timestamp': datetime.now(),
                'metadata': {}
            }
        ]
        
        async_tracker._pending_calls = test_calls.copy()
        
        # Mock _process_calls_sync 抛出异常
        with patch.object(async_tracker, '_process_calls_sync', side_effect=Exception("批处理异常")), \
             patch('harborai.core.async_cost_tracking.logger') as mock_logger:
            
            await async_tracker._process_batch()
            
            # 验证异常被记录
            mock_logger.error.assert_called()
    
    def test_destructor_exception_handling(self):
        """测试析构函数异常处理"""
        async_tracker = AsyncCostTracker()
        
        # Mock executor.shutdown 抛出异常
        with patch.object(async_tracker._executor, 'shutdown', side_effect=Exception("关闭异常")):
            # 调用析构函数不应该抛出异常
            try:
                async_tracker.__del__()
            except Exception:
                pytest.fail("析构函数不应该抛出异常")
    
    @pytest.mark.asyncio
    async def test_cleanup_with_none_tracker(self):
        """测试清理空的全局追踪器"""
        # 确保全局追踪器为None
        import harborai.core.async_cost_tracking as module
        original_tracker = module._async_cost_tracker
        module._async_cost_tracker = None
        
        try:
            # 清理空的追踪器不应该抛出异常
            await cleanup_async_cost_tracker()
        finally:
            # 恢复原始状态
            module._async_cost_tracker = original_tracker
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_large_batch(self):
        """测试大批次的内存使用"""
        async_tracker = AsyncCostTracker()
        async_tracker._batch_size = 1000  # 很大的批次，不会自动触发
        
        mock_settings = Mock()
        mock_settings.fast_path_skip_cost_tracking = False
        
        mock_perf_config = Mock()
        mock_perf_config.get_middleware_config.return_value = {
            'cost_tracking_middleware': True
        }
        mock_perf_config.should_use_fast_path.return_value = False
        
        with patch.object(async_tracker, 'settings', mock_settings), \
             patch.object(async_tracker, 'perf_config', mock_perf_config):
            
            # 添加大量调用但不触发批处理
            for i in range(100):
                await async_tracker.track_api_call_async(
                    model="gpt-3.5-turbo",
                    provider="openai",
                    input_tokens=100,
                    output_tokens=50,
                    cost=0.001,
                    duration=1.0
                )
            
            # 验证内存中有待处理的调用
            assert len(async_tracker._pending_calls) == 100
            
            # 手动刷新
            await async_tracker.flush_pending()
            
            # 验证内存被清理
            assert len(async_tracker._pending_calls) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])