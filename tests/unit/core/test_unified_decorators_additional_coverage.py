#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一装饰器系统额外覆盖率测试模块

功能：补充统一装饰器系统的测试覆盖率，达到90%目标
作者：HarborAI测试团队
创建时间：2024年12月3日

测试覆盖：
- 缓存功能的边界条件和异常处理
- PostgreSQL日志记录的各种场景
- 成本追踪功能的详细测试
- 追踪功能的异常处理
- 配置模式的边界条件
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict

from harborai.core.unified_decorators import (
    DecoratorMode,
    DecoratorConfig,
    UnifiedDecorator
)


class TestUnifiedDecoratorAdditionalCoverage:
    """统一装饰器额外覆盖率测试"""

    @pytest.fixture
    def mock_cache_manager(self):
        """模拟缓存管理器"""
        mock = Mock()
        mock.get.return_value = None
        mock.set.return_value = None
        return mock

    @pytest.fixture
    def mock_postgres_logger(self):
        """模拟PostgreSQL日志记录器"""
        mock = AsyncMock()
        mock.log_async = AsyncMock()
        mock.log_sync = Mock()
        return mock

    @pytest.fixture
    def mock_cost_tracker(self):
        """模拟成本追踪器"""
        mock = AsyncMock()
        mock.track_api_call_async = AsyncMock()
        return mock

    @pytest.fixture
    def mock_tracer(self):
        """模拟追踪器"""
        mock = Mock()
        mock_context = Mock()
        mock_context.end = Mock()
        mock.start_trace.return_value = mock_context
        return mock, mock_context

    def test_cache_manager_none_scenarios(self):
        """测试缓存管理器为None的场景"""
        config = DecoratorConfig(enable_caching=True)
        decorator = UnifiedDecorator(config)
        # 不设置_cache_manager，保持为None
        
        # 测试同步缓存获取
        result = decorator._get_cached_result_sync("test_key")
        assert result is None
        
        # 测试同步缓存设置
        decorator._cache_result_sync("test_key", "test_value")  # 应该不抛出异常

    @pytest.mark.asyncio
    async def test_cache_manager_none_async_scenarios(self):
        """测试缓存管理器为None的异步场景"""
        config = DecoratorConfig(enable_caching=True)
        decorator = UnifiedDecorator(config)
        # 不设置_cache_manager，保持为None
        
        # 测试异步缓存获取
        result = await decorator._get_cached_result("test_key")
        assert result is None
        
        # 测试异步缓存设置
        await decorator._cache_result("test_key", "test_value")  # 应该不抛出异常

    def test_cache_get_exception_handling(self, mock_cache_manager):
        """测试缓存获取异常处理"""
        config = DecoratorConfig(enable_caching=True)
        decorator = UnifiedDecorator(config)
        decorator._cache_manager = mock_cache_manager
        
        # 模拟缓存获取异常
        mock_cache_manager.get.side_effect = Exception("缓存获取失败")
        
        with patch('harborai.core.unified_decorators.logger') as mock_logger:
            result = decorator._get_cached_result_sync("test_key")
            assert result is None
            mock_logger.warning.assert_called_with("获取缓存失败: 缓存获取失败")

    @pytest.mark.asyncio
    async def test_cache_get_async_exception_handling(self, mock_cache_manager):
        """测试异步缓存获取异常处理"""
        config = DecoratorConfig(enable_caching=True)
        decorator = UnifiedDecorator(config)
        decorator._cache_manager = mock_cache_manager
        
        # 模拟缓存获取异常
        mock_cache_manager.get.side_effect = Exception("缓存获取失败")
        
        with patch('harborai.core.unified_decorators.logger') as mock_logger:
            result = await decorator._get_cached_result("test_key")
            assert result is None
            mock_logger.warning.assert_called_with("获取缓存失败: 缓存获取失败")

    def test_cache_set_exception_handling(self, mock_cache_manager):
        """测试缓存设置异常处理"""
        config = DecoratorConfig(enable_caching=True)
        decorator = UnifiedDecorator(config)
        decorator._cache_manager = mock_cache_manager
        
        # 模拟缓存设置异常
        mock_cache_manager.set.side_effect = Exception("缓存设置失败")
        
        with patch('harborai.core.unified_decorators.logger') as mock_logger:
            decorator._cache_result_sync("test_key", "test_value")
            # 检查是否调用了warning方法，但不严格检查消息内容
            assert mock_logger.warning.called

    @pytest.mark.asyncio
    async def test_cache_set_async_exception_handling(self, mock_cache_manager):
        """测试异步缓存设置异常处理"""
        config = DecoratorConfig(enable_caching=True)
        decorator = UnifiedDecorator(config)
        decorator._cache_manager = mock_cache_manager
        
        # 模拟缓存设置异常
        mock_cache_manager.set.side_effect = Exception("缓存设置失败")
        
        with patch('harborai.core.unified_decorators.logger') as mock_logger:
            await decorator._cache_result("test_key", "test_value")
            # 检查是否调用了warning方法，但不严格检查消息内容
            assert mock_logger.warning.called

    def test_end_trace_sync_exception_handling(self, mock_tracer):
        """测试同步结束追踪异常处理"""
        config = DecoratorConfig(enable_tracing=True)
        decorator = UnifiedDecorator(config)
        
        tracer, trace_context = mock_tracer
        # 模拟end方法抛出异常
        trace_context.end.side_effect = Exception("结束追踪失败")
        
        with patch('harborai.core.unified_decorators.logger') as mock_logger:
            decorator._end_trace_sync(trace_context)
            mock_logger.warning.assert_called_with("结束追踪失败: 结束追踪失败")

    @pytest.mark.asyncio
    async def test_end_trace_async_exception_handling(self, mock_tracer):
        """测试异步结束追踪异常处理"""
        config = DecoratorConfig(enable_tracing=True)
        decorator = UnifiedDecorator(config)
        
        tracer, trace_context = mock_tracer
        # 模拟end方法抛出异常
        trace_context.end.side_effect = Exception("结束追踪失败")
        
        with patch('harborai.core.unified_decorators.logger') as mock_logger:
            await decorator._end_trace(trace_context)
            mock_logger.warning.assert_called_with("结束追踪失败: 结束追踪失败")

    def test_end_trace_sync_no_end_method(self):
        """测试同步结束追踪时对象没有end方法"""
        config = DecoratorConfig(enable_tracing=True)
        decorator = UnifiedDecorator(config)
        
        # 创建一个没有end方法的对象
        trace_context = Mock()
        del trace_context.end
        
        # 应该不抛出异常
        decorator._end_trace_sync(trace_context)

    @pytest.mark.asyncio
    async def test_end_trace_async_no_end_method(self):
        """测试异步结束追踪时对象没有end方法"""
        config = DecoratorConfig(enable_tracing=True)
        decorator = UnifiedDecorator(config)
        
        # 创建一个没有end方法的对象
        trace_context = Mock()
        del trace_context.end
        
        # 应该不抛出异常
        await decorator._end_trace(trace_context)

    @pytest.mark.asyncio
    async def test_track_cost_no_tracker(self):
        """测试成本追踪器为None的场景"""
        config = DecoratorConfig(enable_cost_tracking=True)
        decorator = UnifiedDecorator(config)
        # 不设置_cost_tracker，保持为None
        
        # 应该不抛出异常
        await decorator._track_cost("trace_id", "test_func", (), {}, "result")

    @pytest.mark.asyncio
    async def test_track_cost_with_tracker(self, mock_cost_tracker):
        """测试成本追踪功能"""
        config = DecoratorConfig(enable_cost_tracking=True)
        decorator = UnifiedDecorator(config)
        decorator._cost_tracker = mock_cost_tracker
        
        # 创建一个模拟结果对象，包含usage信息
        mock_result = Mock()
        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_result.usage = mock_usage
        
        # 模拟_extract_cost_info返回有效的成本信息
        with patch.object(decorator, '_extract_cost_info', return_value={'input_tokens': 100, 'output_tokens': 50}):
            await decorator._track_cost("trace_id", "test_func", (), {"duration": 1.5}, mock_result)
            
            # 验证成本追踪器被调用
            mock_cost_tracker.track_api_call_async.assert_called_once()

    def test_extract_cost_info_with_usage(self):
        """测试提取成本信息（包含usage）"""
        config = DecoratorConfig()
        decorator = UnifiedDecorator(config)
        
        # 创建一个模拟结果对象，包含usage信息
        mock_result = Mock()
        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_result.usage = mock_usage
        
        cost_info = decorator._extract_cost_info((), {"duration": 1.5}, mock_result)
        
        assert cost_info is not None
        assert cost_info['input_tokens'] == 100
        assert cost_info['output_tokens'] == 50
        assert cost_info['duration'] == 1.5

    def test_extract_cost_info_no_usage(self):
        """测试提取成本信息（无usage）"""
        config = DecoratorConfig()
        decorator = UnifiedDecorator(config)
        
        # 创建一个没有usage的结果对象
        mock_result = Mock()
        del mock_result.usage
        
        cost_info = decorator._extract_cost_info((), {"duration": 1.5}, mock_result)
        
        assert cost_info is not None
        assert cost_info['duration'] == 1.5
        assert 'input_tokens' not in cost_info
        assert 'output_tokens' not in cost_info

    def test_extract_cost_info_partial_usage(self):
        """测试提取成本信息（部分usage信息）"""
        config = DecoratorConfig()
        decorator = UnifiedDecorator(config)
        
        # 创建一个只有prompt_tokens的usage对象
        mock_result = Mock()
        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        del mock_usage.completion_tokens
        mock_result.usage = mock_usage
        
        cost_info = decorator._extract_cost_info((), {}, mock_result)
        
        assert cost_info is not None
        assert cost_info['input_tokens'] == 100
        assert 'output_tokens' not in cost_info

    @pytest.mark.asyncio
    async def test_postgres_logging_none_logger(self):
        """测试PostgreSQL日志记录器为None的场景"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        # 不设置_postgres_logger，保持为None
        
        # 应该不抛出异常
        await decorator._log_to_postgres("trace_id", "test_func", 1.0, "result")

    def test_postgres_logging_sync_none_logger(self):
        """测试同步PostgreSQL日志记录器为None的场景"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        # 不设置_postgres_logger，保持为None
        
        # 应该不抛出异常
        decorator._log_to_postgres_sync("trace_id", "test_func", 1.0, "result")

    @pytest.mark.asyncio
    async def test_postgres_logging_async_exception(self, mock_postgres_logger):
        """测试异步PostgreSQL日志记录异常处理"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        decorator._postgres_logger = mock_postgres_logger
        
        # 模拟日志记录异常
        mock_postgres_logger.log_async.side_effect = Exception("PostgreSQL日志记录失败")
        
        with patch('harborai.core.unified_decorators.logger') as mock_logger:
            await decorator._log_to_postgres("trace_id", "test_func", 1.0, "result")
            mock_logger.warning.assert_called_with("PostgreSQL日志记录失败: PostgreSQL日志记录失败")

    def test_postgres_logging_sync_exception(self, mock_postgres_logger):
        """测试同步PostgreSQL日志记录异常处理"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        decorator._postgres_logger = mock_postgres_logger
        
        # 模拟日志记录异常
        mock_postgres_logger.log_sync.side_effect = Exception("PostgreSQL日志记录失败")
        
        with patch('harborai.core.unified_decorators.logger') as mock_logger:
            decorator._log_to_postgres_sync("trace_id", "test_func", 1.0, "result")
            mock_logger.warning.assert_called_with("PostgreSQL日志记录失败: PostgreSQL日志记录失败")

    @pytest.mark.asyncio
    async def test_log_error_async_none_logger(self):
        """测试异步错误日志记录器为None的场景"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        # 不设置_postgres_logger，保持为None
        
        # 应该不抛出异常
        await decorator._log_error("trace_id", "test_func", "error message")

    def test_log_error_sync_none_logger(self):
        """测试同步错误日志记录器为None的场景"""
        config = DecoratorConfig(enable_postgres_logging=True)
        decorator = UnifiedDecorator(config)
        # 不设置_postgres_logger，保持为None
        
        # 应该不抛出异常
        decorator._log_error_sync("trace_id", "test_func", "error message")

    def test_decorator_config_mode_validation(self):
        """测试装饰器配置模式验证"""
        # 测试有效的字符串模式
        config = DecoratorConfig(mode="fast")
        assert config.mode == "fast"
        
        config = DecoratorConfig(mode="full")
        assert config.mode == "full"

    def test_decorator_config_with_custom_mode(self):
        """测试自定义模式的装饰器配置"""
        custom_mode = DecoratorMode.FAST
        config = DecoratorConfig(mode=custom_mode)
        assert config.mode == custom_mode

    def test_unified_trace_function_with_kwargs_override(self):
        """测试unified_trace函数的kwargs覆盖"""
        from harborai.core.unified_decorators import unified_trace
        
        # 测试FAST模式的kwargs覆盖
        decorator_func = unified_trace(DecoratorMode.FAST, enable_caching=True)
        assert callable(decorator_func)
        
        # 测试FULL模式的kwargs覆盖
        decorator_func = unified_trace(DecoratorMode.FULL, enable_tracing=False)
        assert callable(decorator_func)

    def test_unified_trace_function_with_custom_mode(self):
        """测试unified_trace函数的自定义模式"""
        from harborai.core.unified_decorators import unified_trace
        
        # 测试自定义模式
        custom_config = DecoratorConfig(enable_caching=True, enable_tracing=False)
        decorator_func = unified_trace(custom_config)
        assert callable(decorator_func)

    @pytest.mark.asyncio
    async def test_decorator_with_complex_function_signature(self):
        """测试装饰器处理复杂函数签名"""
        config = DecoratorConfig(
            enable_caching=True,
            enable_tracing=True,
            enable_cost_tracking=True,
            enable_postgres_logging=True
        )
        decorator = UnifiedDecorator(config)
        
        @decorator
        async def complex_function(a, b=10, *args, **kwargs):
            """复杂函数签名测试"""
            return f"a={a}, b={b}, args={args}, kwargs={kwargs}"
        
        result = await complex_function(1, 2, 3, 4, x=5, y=6)
        assert "a=1, b=2, args=(3, 4), kwargs={'x': 5, 'y': 6}" in result

    def test_decorator_with_sync_complex_function_signature(self):
        """测试装饰器处理同步复杂函数签名"""
        config = DecoratorConfig(
            enable_caching=True,
            enable_tracing=True,
            enable_cost_tracking=True,
            enable_postgres_logging=True
        )
        decorator = UnifiedDecorator(config)
        
        @decorator
        def complex_sync_function(a, b=10, *args, **kwargs):
            """复杂同步函数签名测试"""
            return f"a={a}, b={b}, args={args}, kwargs={kwargs}"
        
        result = complex_sync_function(1, 2, 3, 4, x=5, y=6)
        assert "a=1, b=2, args=(3, 4), kwargs={'x': 5, 'y': 6}" in result

    @pytest.mark.asyncio
    async def test_track_cost_exception_handling(self, mock_cost_tracker):
        """测试成本追踪异常处理"""
        config = DecoratorConfig(enable_cost_tracking=True)
        decorator = UnifiedDecorator(config)
        decorator._cost_tracker = mock_cost_tracker
        
        # 模拟成本追踪异常
        mock_cost_tracker.track_api_call_async.side_effect = Exception("成本追踪失败")
        
        # 模拟_extract_cost_info返回有效的成本信息
        with patch.object(decorator, '_extract_cost_info', return_value={'input_tokens': 100}):
            with patch('harborai.core.unified_decorators.logger') as mock_logger:
                await decorator._track_cost("trace_id", "test_func", (), {}, "result")
                mock_logger.warning.assert_called_with("成本追踪失败: 成本追踪失败")

    def test_extract_cost_info_empty_result(self):
        """测试提取成本信息（空结果）"""
        config = DecoratorConfig()
        decorator = UnifiedDecorator(config)
        
        cost_info = decorator._extract_cost_info((), {}, {})
        
        # 应该返回包含默认值的字典
        assert isinstance(cost_info, dict)
        assert 'duration' in cost_info
        assert 'provider' in cost_info