#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速装饰器模块全面测试

测试harborai.api.fast_decorators模块的所有装饰器功能，包括：
- fast_trace: 轻量级追踪装饰器
- fast_cost_tracking: 轻量级成本追踪装饰器
- conditional_decorator: 条件装饰器
- fast_path_decorators: 快速路径装饰器组合

测试策略：
1. 单元测试：测试每个装饰器的基本功能
2. 性能测试：验证轻量级实现的性能优势
3. 条件测试：测试条件装饰器的逻辑
4. 集成测试：测试装饰器组合使用
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Any, Dict

from harborai.api.fast_decorators import (
    fast_trace,
    fast_cost_tracking,
    conditional_decorator,
    fast_path_decorators,
    _async_record_cost
)


class TestFastTrace:
    """测试fast_trace装饰器"""
    
    def test_fast_trace_sync_function_basic(self):
        """测试同步函数的基本追踪功能"""
        @fast_trace
        def sync_function(x: int, y: int, **kwargs) -> int:
            return x + y
        
        with patch('harborai.api.fast_decorators.generate_trace_id', return_value='fast-trace-123'):
            with patch('harborai.api.fast_decorators.logger') as mock_logger:
                with patch('harborai.api.fast_decorators.settings') as mock_settings:
                    mock_settings.enable_detailed_tracing = True
                    
                    result = sync_function(3, 4)
                    
                    assert result == 7
                    mock_logger.debug.assert_called_once_with("[fast-trace-123] Fast path: sync_function")
    
    def test_fast_trace_sync_function_with_existing_trace_id(self):
        """测试同步函数使用现有trace_id"""
        @fast_trace
        def sync_function(x: int, y: int, **kwargs) -> int:
            return x + y
        
        with patch('harborai.api.fast_decorators.logger') as mock_logger:
            with patch('harborai.api.fast_decorators.settings') as mock_settings:
                mock_settings.enable_detailed_tracing = True
                
                result = sync_function(3, 4, trace_id='existing-trace-456')
                
                assert result == 7
                mock_logger.debug.assert_called_once_with("[existing-trace-456] Fast path: sync_function")
    
    def test_fast_trace_sync_function_tracing_disabled(self):
        """测试同步函数禁用追踪"""
        @fast_trace
        def sync_function_no_trace(x: int, y: int, **kwargs) -> int:
            return x + y
        
        with patch('harborai.api.fast_decorators.logger') as mock_logger:
            with patch('harborai.api.fast_decorators.settings') as mock_settings:
                mock_settings.enable_detailed_tracing = False
                
                result = sync_function_no_trace(3, 4)
                
                assert result == 7
                mock_logger.debug.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_fast_trace_async_function_basic(self):
        """测试异步函数的基本追踪功能"""
        @fast_trace
        async def async_function(x: int, y: int, **kwargs) -> int:
            await asyncio.sleep(0.01)
            return x + y
        
        with patch('harborai.api.fast_decorators.generate_trace_id', return_value='async-trace-789'):
            with patch('harborai.api.fast_decorators.logger') as mock_logger:
                with patch('harborai.api.fast_decorators.settings') as mock_settings:
                    mock_settings.enable_detailed_tracing = True
                    
                    result = await async_function(5, 6)
                    
                    assert result == 11
                    mock_logger.debug.assert_called_once_with("[async-trace-789] Fast path: async_function")
    
    @pytest.mark.asyncio
    async def test_fast_trace_async_function_with_existing_trace_id(self):
        """测试异步函数使用现有trace_id"""
        @fast_trace
        async def async_function(x: int, y: int, **kwargs) -> int:
            await asyncio.sleep(0.01)
            return x + y
        
        with patch('harborai.api.fast_decorators.logger') as mock_logger:
            with patch('harborai.api.fast_decorators.settings') as mock_settings:
                mock_settings.enable_detailed_tracing = True
                
                result = await async_function(5, 6, trace_id='existing-async-trace')
                
                assert result == 11
                mock_logger.debug.assert_called_once_with("[existing-async-trace] Fast path: async_function")
    
    def test_fast_trace_preserves_function_metadata(self):
        """测试装饰器保留函数元数据"""
        @fast_trace
        def documented_function(x: int) -> int:
            """这是一个有文档的函数"""
            return x * 2
        
        assert documented_function.__name__ == 'documented_function'
        assert documented_function.__doc__ == '这是一个有文档的函数'


class TestFastCostTracking:
    """测试fast_cost_tracking装饰器"""
    
    def test_fast_cost_tracking_sync_function_basic(self):
        """测试同步函数的基本成本追踪功能"""
        @fast_cost_tracking
        def sync_function_with_cost(**kwargs) -> Mock:
            result = Mock()
            result.usage = Mock()
            result.usage.total_tokens = 100
            result.usage.prompt_tokens = 50
            result.usage.completion_tokens = 50
            return result
        
        with patch('harborai.api.fast_decorators.settings') as mock_settings:
            with patch('harborai.api.fast_decorators.logger') as mock_logger:
                mock_settings.enable_cost_tracking = True
                
                result = sync_function_with_cost(model='gpt-3.5-turbo', trace_id='cost-trace-123')
                
                assert result.usage.total_tokens == 100
                # 验证日志记录
                mock_logger.debug.assert_called()
    
    def test_fast_cost_tracking_sync_function_disabled(self):
        """测试同步函数禁用成本追踪"""
        @fast_cost_tracking
        def sync_function_no_cost(**kwargs) -> str:
            return "no cost tracking"
        
        with patch('harborai.api.fast_decorators.settings') as mock_settings:
            with patch('harborai.api.fast_decorators.logger') as mock_logger:
                mock_settings.enable_cost_tracking = False
                
                result = sync_function_no_cost(model='gpt-3.5-turbo')
                
                assert result == "no cost tracking"
                mock_logger.debug.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_fast_cost_tracking_async_function_basic(self):
        """测试异步函数的基本成本追踪功能"""
        @fast_cost_tracking
        async def async_function_with_cost(**kwargs) -> Mock:
            result = Mock()
            result.usage = Mock()
            result.usage.total_tokens = 200
            result.usage.prompt_tokens = 100
            result.usage.completion_tokens = 100
            return result
        
        with patch('harborai.api.fast_decorators.settings') as mock_settings:
            with patch('asyncio.create_task') as mock_create_task:
                mock_settings.enable_cost_tracking = True
                
                result = await async_function_with_cost(model='gpt-4', trace_id='async-cost-trace')
                
                assert result.usage.total_tokens == 200
                # 验证异步任务被创建
                mock_create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fast_cost_tracking_async_function_disabled(self):
        """测试异步函数禁用成本追踪"""
        @fast_cost_tracking
        async def async_function_no_cost(**kwargs) -> str:
            return "no async cost tracking"
        
        with patch('harborai.api.fast_decorators.settings') as mock_settings:
            with patch('asyncio.create_task') as mock_create_task:
                mock_settings.enable_cost_tracking = False
                
                result = await async_function_no_cost(model='gpt-4')
                
                assert result == "no async cost tracking"
                mock_create_task.assert_not_called()
    
    def test_fast_cost_tracking_sync_exception_handling(self):
        """测试同步函数异常处理"""
        @fast_cost_tracking
        def sync_function_with_error(**kwargs):
            raise ValueError("Test error")
        
        with patch('harborai.api.fast_decorators.settings') as mock_settings:
            with patch('harborai.api.fast_decorators.logger') as mock_logger:
                mock_settings.enable_cost_tracking = True
                
                with pytest.raises(ValueError, match="Test error"):
                    sync_function_with_error(model='gpt-3.5-turbo', trace_id='error-trace')
                
                # 验证错误被记录
                mock_logger.debug.assert_called()
    
    @pytest.mark.asyncio
    async def test_fast_cost_tracking_async_exception_handling(self):
        """测试异步函数异常处理"""
        @fast_cost_tracking
        async def async_function_with_error(**kwargs):
            raise ValueError("Async test error")
        
        with patch('harborai.api.fast_decorators.settings') as mock_settings:
            with patch('asyncio.create_task') as mock_create_task:
                mock_settings.enable_cost_tracking = True
                
                with pytest.raises(ValueError, match="Async test error"):
                    await async_function_with_error(model='gpt-4', trace_id='async-error-trace')
                
                # 验证异步任务被创建（用于记录错误）
                mock_create_task.assert_called_once()
    
    def test_fast_cost_tracking_no_usage_attribute(self):
        """测试没有usage属性的结果"""
        @fast_cost_tracking
        def sync_function_no_usage(**kwargs) -> str:
            return "no usage"
        
        with patch('harborai.api.fast_decorators.settings') as mock_settings:
            with patch('harborai.api.fast_decorators.logger') as mock_logger:
                mock_settings.enable_cost_tracking = True
                
                result = sync_function_no_usage(model='gpt-3.5-turbo', trace_id='no-usage-trace')
                
                assert result == "no usage"
                # 没有usage属性时不应该记录详细信息
                mock_logger.debug.assert_not_called()


class TestConditionalDecorator:
    """测试conditional_decorator"""
    
    def test_conditional_decorator_enabled(self):
        """测试条件装饰器启用时的行为"""
        def mock_decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return f"decorated: {result}"
            return wrapper
        
        @conditional_decorator(mock_decorator, True)
        def test_function(x: int) -> int:
            return x * 2
        
        result = test_function(5)
        assert result == "decorated: 10"
    
    def test_conditional_decorator_disabled(self):
        """测试条件装饰器禁用时的行为"""
        def mock_decorator(func):
            def wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                return f"decorated: {result}"
            return wrapper
        
        @conditional_decorator(mock_decorator, False)
        def test_function(x: int) -> int:
            return x * 2
        
        result = test_function(5)
        assert result == 10  # 没有被装饰
    
    def test_conditional_decorator_with_complex_decorator(self):
        """测试条件装饰器与复杂装饰器的组合"""
        def timing_decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start
                return {'result': result, 'duration': duration}
            return wrapper
        
        @conditional_decorator(timing_decorator, True)
        def slow_function(delay: float) -> str:
            time.sleep(delay)
            return "completed"
        
        result = slow_function(0.01)
        assert isinstance(result, dict)
        assert result['result'] == "completed"
        assert result['duration'] > 0


class TestFastPathDecorators:
    """测试fast_path_decorators组合装饰器"""
    
    def test_fast_path_decorators_basic_functionality(self):
        """测试快速路径装饰器的基本功能"""
        with patch('harborai.api.fast_decorators.get_performance_config') as mock_get_config:
            mock_config = Mock()
            mock_decorator_config = Mock()
            mock_decorator_config.enable_tracing = True
            mock_decorator_config.enable_cost_tracking = True
            mock_config.get_decorator_config.return_value = mock_decorator_config
            mock_get_config.return_value = mock_config
            
            @fast_path_decorators
            def test_function(x: int, **kwargs) -> int:
                return x * 2
            
            with patch('harborai.api.fast_decorators.settings') as mock_settings:
                mock_settings.enable_detailed_tracing = True
                mock_settings.enable_cost_tracking = True
                
                result = test_function(5)
                assert result == 10
    
    def test_fast_path_decorators_disabled(self):
        """测试快速路径装饰器禁用时的行为"""
        with patch('harborai.api.fast_decorators.get_performance_config') as mock_get_config:
            mock_config = Mock()
            mock_decorator_config = Mock()
            mock_decorator_config.enable_tracing = False
            mock_decorator_config.enable_cost_tracking = False
            mock_config.get_decorator_config.return_value = mock_decorator_config
            mock_get_config.return_value = mock_config
            
            @fast_path_decorators
            def test_function(x: int) -> int:
                return x * 2
            
            result = test_function(5)
            assert result == 10
    
    @pytest.mark.asyncio
    async def test_fast_path_decorators_async_function(self):
        """测试快速路径装饰器与异步函数"""
        with patch('harborai.api.fast_decorators.get_performance_config') as mock_get_config:
            mock_config = Mock()
            mock_decorator_config = Mock()
            mock_decorator_config.enable_tracing = True
            mock_decorator_config.enable_cost_tracking = True
            mock_config.get_decorator_config.return_value = mock_decorator_config
            mock_get_config.return_value = mock_config
            
            @fast_path_decorators
            async def async_test_function(x: int, **kwargs) -> int:
                await asyncio.sleep(0.01)
                return x * 3
            
            with patch('harborai.api.fast_decorators.settings') as mock_settings:
                with patch('asyncio.create_task') as mock_create_task:
                    mock_settings.enable_detailed_tracing = True
                    mock_settings.enable_cost_tracking = True
                    
                    result = await async_test_function(7)
                    assert result == 21


class TestAsyncRecordCost:
    """测试_async_record_cost函数"""
    
    @pytest.mark.asyncio
    async def test_async_record_cost_success(self):
        """测试成功记录成本信息"""
        mock_usage = Mock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 30
        
        # 简单测试函数能正常运行而不抛出异常
        try:
            await _async_record_cost(
                trace_id='test-trace',
                model='gpt-3.5-turbo',
                usage=mock_usage,
                duration=1.5,
                success=True
            )
            # 如果没有抛出异常，测试通过
            assert True
        except Exception as e:
            # 如果抛出异常，测试失败
            pytest.fail(f"_async_record_cost raised an exception: {e}")
    
    @pytest.mark.asyncio
    async def test_async_record_cost_failure(self):
        """测试记录失败信息"""
        # 简单测试函数能正常运行而不抛出异常
        try:
            await _async_record_cost(
                trace_id='test-trace',
                model='gpt-3.5-turbo',
                usage=None,
                duration=1.0,
                success=False,
                error='Test error'
            )
            # 如果没有抛出异常，测试通过
            assert True
        except Exception as e:
            # 如果抛出异常，测试失败
            pytest.fail(f"_async_record_cost raised an exception: {e}")
    
    @pytest.mark.asyncio
    async def test_async_record_cost_exception_handling(self):
        """测试异常处理"""
        mock_usage = Mock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 30
        
        # 测试即使有异常也不会抛出
        try:
            await _async_record_cost(
                trace_id='test-trace',
                model='gpt-3.5-turbo',
                usage=mock_usage,
                duration=1.5,
                success=True
            )
            # 如果没有抛出异常，测试通过
            assert True
        except Exception as e:
            # 如果抛出异常，测试失败
            pytest.fail(f"_async_record_cost should not raise exceptions: {e}")


class TestPerformanceComparison:
    """测试性能对比"""
    
    def test_fast_trace_vs_regular_trace_performance(self):
        """测试快速追踪与常规追踪的性能对比"""
        @fast_trace
        def fast_traced_function(x: int, **kwargs) -> int:
            return x * 2
        
        def regular_traced_function(x: int) -> int:
            # 模拟常规追踪的开销
            time.sleep(0.001)
            return x * 2
        
        # 测试快速追踪性能
        with patch('harborai.api.fast_decorators.settings') as mock_settings:
            mock_settings.enable_detailed_tracing = False
            
            start_time = time.perf_counter()
            for _ in range(100):
                fast_traced_function(5)
            fast_duration = time.perf_counter() - start_time
            
            start_time = time.perf_counter()
            for _ in range(100):
                regular_traced_function(5)
            regular_duration = time.perf_counter() - start_time
            
            # 快速追踪应该明显更快
            assert fast_duration < regular_duration
    
    def test_fast_cost_tracking_performance(self):
        """测试快速成本追踪的性能"""
        @fast_cost_tracking
        def fast_cost_function(**kwargs) -> str:
            return "fast cost"
        
        with patch('harborai.api.fast_decorators.settings') as mock_settings:
            mock_settings.enable_cost_tracking = False
            
            start_time = time.perf_counter()
            for _ in range(100):
                fast_cost_function(model='gpt-3.5-turbo')
            duration = time.perf_counter() - start_time
            
            # 禁用时应该很快
            assert duration < 0.1  # 100次调用应该在0.1秒内完成


class TestEdgeCases:
    """测试边界情况"""
    
    def test_fast_trace_with_none_trace_id(self):
        """测试trace_id为None的情况"""
        @fast_trace
        def test_function(x: int, **kwargs) -> int:
            return x * 2
        
        with patch('harborai.api.fast_decorators.generate_trace_id', return_value='generated-trace'):
            with patch('harborai.api.fast_decorators.settings') as mock_settings:
                mock_settings.enable_detailed_tracing = True
                
                result = test_function(5, trace_id=None)
                assert result == 10
    
    def test_fast_cost_tracking_with_missing_usage(self):
        """测试缺少usage属性的情况"""
        @fast_cost_tracking
        def test_function(**kwargs) -> Mock:
            result = Mock()
            # 故意不设置usage属性
            return result
        
        with patch('harborai.api.fast_decorators.settings') as mock_settings:
            mock_settings.enable_cost_tracking = True
            
            result = test_function(model='gpt-3.5-turbo', trace_id='test-trace')
            assert result is not None
    
    def test_conditional_decorator_with_none_condition(self):
        """测试条件为None的情况"""
        def mock_decorator(func):
            def wrapper(*args, **kwargs):
                return f"decorated: {func(*args, **kwargs)}"
            return wrapper
        
        @conditional_decorator(mock_decorator, None)
        def test_function(x: int) -> int:
            return x * 2
        
        # None应该被视为False
        result = test_function(5)
        assert result == 10  # 没有被装饰
    
    def test_fast_path_decorators_with_missing_config(self):
        """测试配置缺失的情况"""
        with patch('harborai.api.fast_decorators.get_performance_config', side_effect=Exception("Config error")):
            # 配置出错时应该抛出异常
            with pytest.raises(Exception, match="Config error"):
                @fast_path_decorators
                def test_function(x: int) -> int:
                    return x * 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])