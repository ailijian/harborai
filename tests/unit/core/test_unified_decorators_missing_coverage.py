"""
unified_decorators.py 缺失覆盖率补充测试

专门针对覆盖率报告中缺失的行进行测试，以达到90%覆盖率目标
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from harborai.core.unified_decorators import (
    UnifiedDecorator, 
    DecoratorConfig, 
    DecoratorMode
)


class TestUnifiedDecoratorMissingCoverage:
    """补充缺失覆盖率的测试"""

    @pytest.fixture
    def mock_cache_manager(self):
        """模拟缓存管理器"""
        mock = Mock()
        mock.get = Mock()
        mock.set = Mock()
        return mock

    @pytest.fixture
    def mock_postgres_logger(self):
        """模拟PostgreSQL日志记录器"""
        mock = Mock()
        mock.log_async = AsyncMock()
        mock.log_sync = Mock()
        return mock

    @pytest.fixture
    def mock_cost_tracker(self):
        """模拟成本追踪器"""
        mock = AsyncMock()
        mock.track_api_call_async = AsyncMock()
        return mock

    def test_cache_result_exception_handling(self, mock_cache_manager):
        """测试缓存结果设置异常处理 - 覆盖行280-282, 287"""
        config = DecoratorConfig(enable_caching=True)
        decorator = UnifiedDecorator(config)
        decorator._cache_manager = mock_cache_manager
        
        # 模拟缓存设置异常
        mock_cache_manager.set.side_effect = Exception("缓存设置失败")
        
        # 测试异步版本 - 应该不抛出异常
        async def test_async():
            await decorator._cache_result("test_key", "test_value")
        
        # 不应该抛出异常
        asyncio.run(test_async())
        
        # 验证set方法被调用
        mock_cache_manager.set.assert_called_once()

    def test_cache_result_sync_exception_handling(self, mock_cache_manager):
        """测试同步缓存结果设置异常处理"""
        config = DecoratorConfig(enable_caching=True)
        decorator = UnifiedDecorator(config)
        decorator._cache_manager = mock_cache_manager
        
        # 模拟缓存设置异常
        mock_cache_manager.set.side_effect = Exception("缓存设置失败")
        
        # 测试同步版本 - 应该不抛出异常
        decorator._cache_result_sync("test_key", "test_value")
        
        # 验证set方法被调用
        mock_cache_manager.set.assert_called_once()

    def test_track_cost_functionality(self, mock_cost_tracker):
        """测试成本追踪功能 - 覆盖行369-370, 375"""
        config = DecoratorConfig(enable_cost_tracking=True)
        decorator = UnifiedDecorator(config)
        decorator._cost_tracker = mock_cost_tracker
        
        # 模拟_extract_cost_info返回有效成本信息
        with patch.object(decorator, '_extract_cost_info', return_value={
            'input_tokens': 100, 
            'output_tokens': 50, 
            'model': 'gpt-4', 
            'provider': 'openai'
        }):
            async def test_async():
                await decorator._track_cost("trace123", "test_func", (), {}, "result")
            
            asyncio.run(test_async())
            
            # 验证成本追踪被调用
            mock_cost_tracker.track_api_call_async.assert_called_once()

    def test_track_cost_no_cost_info(self, mock_cost_tracker):
        """测试成本追踪无成本信息情况"""
        config = DecoratorConfig(enable_cost_tracking=True)
        decorator = UnifiedDecorator(config)
        decorator._cost_tracker = mock_cost_tracker
        
        # 模拟_extract_cost_info返回None
        with patch.object(decorator, '_extract_cost_info', return_value=None):
            async def test_async():
                await decorator._track_cost("trace123", "test_func", (), {}, "result")
            
            asyncio.run(test_async())
            
            # 验证成本追踪未被调用
            mock_cost_tracker.track_api_call_async.assert_not_called()

    def test_track_cost_exception_handling(self, mock_cost_tracker):
        """测试成本追踪异常处理"""
        config = DecoratorConfig(enable_cost_tracking=True)
        decorator = UnifiedDecorator(config)
        decorator._cost_tracker = mock_cost_tracker
        
        # 模拟成本追踪异常
        mock_cost_tracker.track_api_call_async.side_effect = Exception("成本追踪失败")
        
        with patch.object(decorator, '_extract_cost_info', return_value={'cost': 0.01}):
            async def test_async():
                # 不应该抛出异常
                await decorator._track_cost("trace123", "test_func", (), {}, "result")
            
            asyncio.run(test_async())

    def test_log_error_async_with_postgres_logger(self, mock_postgres_logger):
        """测试异步错误日志记录 - 覆盖行441-451"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        decorator._postgres_logger = mock_postgres_logger
        
        async def test_async():
            await decorator._log_error("trace123", "test_func", "测试错误")
        
        asyncio.run(test_async())
        
        # 验证异步日志记录被调用
        mock_postgres_logger.log_async.assert_called_once()
        call_args = mock_postgres_logger.log_async.call_args[0][0]
        assert call_args['trace_id'] == "trace123"
        assert call_args['function_name'] == "test_func"
        assert call_args['error'] == "测试错误"
        assert call_args['level'] == 'ERROR'

    def test_log_error_async_exception_handling(self, mock_postgres_logger):
        """测试异步错误日志记录异常处理 - 覆盖行456"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        decorator._postgres_logger = mock_postgres_logger
        
        # 模拟日志记录异常
        mock_postgres_logger.log_async.side_effect = Exception("日志记录失败")
        
        async def test_async():
            # 不应该抛出异常
            await decorator._log_error("trace123", "test_func", "测试错误")
        
        asyncio.run(test_async())

    def test_log_error_sync_with_postgres_logger(self, mock_postgres_logger):
        """测试同步错误日志记录 - 覆盖行512-520"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        decorator._postgres_logger = mock_postgres_logger
        
        decorator._log_error_sync("trace123", "test_func", "测试错误")
        
        # 验证同步日志记录被调用
        mock_postgres_logger.log_sync.assert_called_once()
        call_args = mock_postgres_logger.log_sync.call_args[0][0]
        assert call_args['trace_id'] == "trace123"
        assert call_args['function_name'] == "test_func"
        assert call_args['error'] == "测试错误"
        assert call_args['level'] == 'ERROR'

    def test_log_error_sync_exception_handling(self, mock_postgres_logger):
        """测试同步错误日志记录异常处理"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        decorator._postgres_logger = mock_postgres_logger
        
        # 模拟日志记录异常
        mock_postgres_logger.log_sync.side_effect = Exception("日志记录失败")
        
        # 不应该抛出异常
        decorator._log_error_sync("trace123", "test_func", "测试错误")

    def test_update_execution_stats_first_time(self):
        """测试首次执行统计更新 - 覆盖行563"""
        config = DecoratorConfig()
        decorator = UnifiedDecorator(config)
        
        # 初始状态
        assert decorator._stats['avg_execution_time'] == 0.0
        
        # 首次更新
        decorator._update_execution_stats(1.5)
        
        # 验证首次更新直接设置值
        assert decorator._stats['avg_execution_time'] == 1.5

    def test_update_execution_stats_exponential_moving_average(self):
        """测试执行统计的指数移动平均 - 覆盖行569"""
        config = DecoratorConfig()
        decorator = UnifiedDecorator(config)
        
        # 设置初始平均值
        decorator._stats['avg_execution_time'] = 1.0
        
        # 更新统计
        decorator._update_execution_stats(2.0)
        
        # 验证指数移动平均计算
        # 新值 = 0.9 * 旧值 + 0.1 * 新值 = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        assert decorator._stats['avg_execution_time'] == 1.1



    def test_end_trace_with_context(self):
        """测试结束追踪上下文"""
        config = DecoratorConfig()
        decorator = UnifiedDecorator(config)
        
        # 模拟追踪上下文
        mock_trace_context = Mock()
        mock_trace_context.end = Mock()
        
        async def test_async():
            await decorator._end_trace(mock_trace_context)
        
        asyncio.run(test_async())
        
        # 验证end方法被调用
        mock_trace_context.end.assert_called_once()

    def test_end_trace_exception_handling(self):
        """测试结束追踪异常处理"""
        config = DecoratorConfig()
        decorator = UnifiedDecorator(config)
        
        # 模拟追踪上下文异常
        mock_trace_context = Mock()
        mock_trace_context.end.side_effect = Exception("结束追踪失败")
        
        async def test_async():
            # 不应该抛出异常
            await decorator._end_trace(mock_trace_context)
        
        asyncio.run(test_async())

    def test_end_trace_no_end_method(self):
        """测试结束追踪时上下文没有end方法"""
        config = DecoratorConfig()
        decorator = UnifiedDecorator(config)
        
        # 模拟没有end方法的上下文
        mock_trace_context = Mock(spec=[])  # 空spec，没有end方法
        
        async def test_async():
            # 不应该抛出异常
            await decorator._end_trace(mock_trace_context)
        
        asyncio.run(test_async())

    def test_cache_manager_none_scenarios(self):
        """测试缓存管理器为None的场景"""
        config = DecoratorConfig(enable_caching=True)
        decorator = UnifiedDecorator(config)
        # 不设置_cache_manager，保持为None
        
        # 测试异步版本
        async def test_async():
            result = await decorator._get_cached_result("test_key")
            assert result is None
            
            # 缓存结果也不应该抛出异常
            await decorator._cache_result("test_key", "test_value")
        
        asyncio.run(test_async())
        
        # 测试同步版本
        result = decorator._get_cached_result_sync("test_key")
        assert result is None
        
        # 缓存结果也不应该抛出异常
        decorator._cache_result_sync("test_key", "test_value")

    def test_cost_tracker_none_scenarios(self):
        """测试成本追踪器为None的场景"""
        config = DecoratorConfig(enable_cost_tracking=True)
        decorator = UnifiedDecorator(config)
        # 不设置_cost_tracker，保持为None
        
        async def test_async():
            # 不应该抛出异常
            await decorator._track_cost("trace123", "test_func", (), {}, "result")
        
        asyncio.run(test_async())

    def test_postgres_logger_none_scenarios(self):
        """测试PostgreSQL日志记录器为None的场景"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        # 不设置_postgres_logger，保持为None
        
        # 测试异步版本
        async def test_async():
            # 不应该抛出异常
            await decorator._log_to_postgres("trace123", "test_func", 0.5, "success")
            await decorator._log_error("trace123", "test_func", "测试错误")
        
        asyncio.run(test_async())
        
        # 测试同步版本
        decorator._log_to_postgres_sync("trace123", "test_func", 0.5, "success")
        decorator._log_error_sync("trace123", "test_func", "测试错误")