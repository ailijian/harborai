#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一装饰器系统最终覆盖率测试模块

功能：补充剩余的测试覆盖率，达到98%以上的目标
作者：HarborAI测试团队
创建时间：2024年12月3日

测试覆盖：
- 异步和同步错误处理中的background_logging分支
- 成本追踪功能的边界条件
- 其他未覆盖的代码路径
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from harborai.core.unified_decorators import (
    DecoratorConfig,
    UnifiedDecorator,
    DecoratorMode
)


class TestUnifiedDecoratorFinalCoverage:
    """测试统一装饰器的最终覆盖率"""

    @pytest.mark.asyncio
    async def test_async_error_with_background_logging(self):
        """测试异步函数错误处理中的background_logging分支"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_tracing=True,
            background_logging=True,  # 启用后台日志
            enable_cost_tracking=False
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.submit_background_task') as mock_submit:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = AsyncMock()
            mock_submit.return_value = asyncio.create_task(asyncio.sleep(0))
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            async def failing_function():
                """会抛出异常的异步函数"""
                raise ValueError("测试异常")
            
            # 执行函数并捕获异常
            with pytest.raises(ValueError, match="测试异常"):
                await failing_function()
            
            # 验证background_logging被调用
            mock_submit.assert_called()
            assert decorator._stats['error_count'] == 1

    def test_sync_error_with_background_logging(self):
        """测试同步函数错误处理中的background_logging分支"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_tracing=True,
            background_logging=True,  # 启用后台日志
            enable_cost_tracking=False
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            # 模拟_log_error_sync方法
            decorator._log_error_sync = Mock()
            
            @decorator
            def failing_function():
                """会抛出异常的同步函数"""
                raise ValueError("测试异常")
            
            # 执行函数并捕获异常
            with pytest.raises(ValueError, match="测试异常"):
                failing_function()
            
            # 验证_log_error_sync被调用
            decorator._log_error_sync.assert_called_once()
            assert decorator._stats['error_count'] == 1

    @pytest.mark.asyncio
    async def test_cost_tracking_with_cost_info(self):
        """测试成本追踪功能有成本信息的情况"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_cost_tracking=True,
            async_cost_tracking=True
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_cost_tracker = AsyncMock()
            mock_get_tracker.return_value = mock_cost_tracker
            
            decorator = UnifiedDecorator(config)
            
            # 模拟_extract_cost_info返回有效的成本信息
            decorator._extract_cost_info = Mock(return_value={
                'tokens': 100,
                'cost': 0.01,
                'model': 'test-model'
            })
            
            @decorator
            async def test_function():
                """测试函数"""
                return "result"
            
            result = await test_function()
            
            assert result == "result"
            # 验证成本追踪被调用
            mock_cost_tracker.track_api_call_async.assert_called()

    def test_sync_cost_tracking_with_cost_info(self):
        """测试同步成本追踪功能有成本信息的情况"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_cost_tracking=True,
            async_cost_tracking=True  # 需要有成本追踪器
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_cost_tracker = Mock()
            mock_cost_tracker.track_sync = Mock()
            mock_get_tracker.return_value = mock_cost_tracker
            
            decorator = UnifiedDecorator(config)
            
            # 模拟_extract_cost_info返回有效的成本信息
            decorator._extract_cost_info = Mock(return_value={
                'tokens': 100,
                'cost': 0.01,
                'model': 'test-model'
            })
            
            @decorator
            def test_function():
                """测试函数"""
                return "result"
            
            result = test_function()
            
            assert result == "result"
            # 验证同步成本追踪被调用
            mock_cost_tracker.track_sync.assert_called()

    @pytest.mark.asyncio
    async def test_cost_tracking_without_cost_info(self):
        """测试成本追踪功能没有成本信息的情况"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_cost_tracking=True,
            async_cost_tracking=True
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_cost_tracker = AsyncMock()
            mock_get_tracker.return_value = mock_cost_tracker
            
            decorator = UnifiedDecorator(config)
            
            # 模拟_extract_cost_info返回None（无成本信息）
            decorator._extract_cost_info = Mock(return_value=None)
            
            @decorator
            async def test_function():
                """测试函数"""
                return "result"
            
            result = await test_function()
            
            assert result == "result"
            # 验证成本追踪没有被调用（因为没有成本信息）
            mock_cost_tracker.track_api_call_async.assert_not_called()

    def test_sync_cost_tracking_without_cost_info(self):
        """测试同步成本追踪功能没有成本信息的情况"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_cost_tracking=True,
            async_cost_tracking=True  # 需要有成本追踪器
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_cost_tracker = Mock()
            mock_cost_tracker.track_sync = Mock()
            mock_get_tracker.return_value = mock_cost_tracker
            
            decorator = UnifiedDecorator(config)
            
            # 模拟_extract_cost_info返回None（无成本信息）
            decorator._extract_cost_info = Mock(return_value=None)
            
            @decorator
            def test_function():
                """测试函数"""
                return "result"
            
            result = test_function()
            
            assert result == "result"
            # 验证同步成本追踪没有被调用（因为没有成本信息）
            mock_cost_tracker.track_sync.assert_not_called()

    @pytest.mark.asyncio
    async def test_cost_tracking_without_tracker(self):
        """测试没有成本追踪器的情况"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_cost_tracking=False,  # 禁用成本追踪
            async_cost_tracking=False
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings:
            mock_get_settings.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            async def test_function():
                """测试函数"""
                return "result"
            
            result = await test_function()
            
            assert result == "result"
            # 验证没有成本追踪器
            assert decorator._cost_tracker is None

    def test_sync_cost_tracking_without_tracker(self):
        """测试同步版本没有成本追踪器的情况"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_cost_tracking=False,  # 禁用成本追踪
            async_cost_tracking=False
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings:
            mock_get_settings.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            def test_function():
                """测试函数"""
                return "result"
            
            result = test_function()
            
            assert result == "result"
            # 验证没有成本追踪器
            assert decorator._cost_tracker is None

    @pytest.mark.asyncio
    async def test_trace_end_failure(self):
        """测试追踪结束失败的情况"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_tracing=True
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.TraceContext') as mock_trace_context:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = AsyncMock()
            
            # 模拟追踪上下文的end方法抛出异常
            mock_context = Mock()
            mock_context.end.side_effect = Exception("追踪结束失败")
            mock_trace_context.return_value = mock_context
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            async def test_function():
                """测试函数"""
                return "result"
            
            # 函数应该正常执行，即使追踪结束失败
            result = await test_function()
            assert result == "result"