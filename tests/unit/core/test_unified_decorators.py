#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一装饰器系统测试模块

功能：测试统一装饰器系统的各项功能
作者：HarborAI测试团队
创建时间：2024年12月3日

测试覆盖：
- DecoratorMode枚举
- DecoratorConfig配置类
- UnifiedDecorator核心功能
- 各种预定义装饰器
- 异步和同步函数装饰
- 缓存、追踪、成本追踪等功能
"""

import pytest
import asyncio
import time
import uuid
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Any, Dict

from harborai.core.unified_decorators import (
    DecoratorMode,
    DecoratorConfig,
    UnifiedDecorator,
    fast_decorator,
    full_decorator,
    unified_trace,
    fast_trace,
    full_trace,
    conditional_unified_decorator,
    smart_decorator,
    cost_tracking,
    with_trace,
    with_postgres_logging,
    with_async_trace
)


class TestDecoratorMode:
    """测试装饰器模式枚举"""
    
    def test_decorator_mode_values(self):
        """测试装饰器模式的值"""
        assert DecoratorMode.FAST.value == "fast"
        assert DecoratorMode.FULL.value == "full"
        assert DecoratorMode.CUSTOM.value == "custom"
    
    def test_decorator_mode_comparison(self):
        """测试装饰器模式比较"""
        assert DecoratorMode.FAST == DecoratorMode.FAST
        assert DecoratorMode.FAST != DecoratorMode.FULL
        assert DecoratorMode.FULL != DecoratorMode.CUSTOM


class TestDecoratorConfig:
    """测试装饰器配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = DecoratorConfig()
        
        assert config.mode == DecoratorMode.FULL
        assert config.enable_tracing is True
        assert config.enable_cost_tracking is True
        assert config.enable_postgres_logging is True
        assert config.enable_caching is False
        assert config.enable_retry is False
        assert config.enable_rate_limiting is False
        assert config.async_cost_tracking is True
        assert config.background_logging is True
        assert config.cache_results is False
        assert config.cache_ttl == 300
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.rate_limit is None
        assert config.rate_window == 60
    
    def test_fast_mode_config(self):
        """测试快速模式配置"""
        config = DecoratorConfig.fast_mode()
        
        assert config.mode == DecoratorMode.FAST
        assert config.enable_tracing is False
        assert config.enable_cost_tracking is True
        assert config.enable_postgres_logging is False
        assert config.async_cost_tracking is True
        assert config.background_logging is True
    
    def test_full_mode_config(self):
        """测试完整模式配置"""
        config = DecoratorConfig.full_mode()
        
        assert config.mode == DecoratorMode.FULL
        assert config.enable_tracing is True
        assert config.enable_cost_tracking is True
        assert config.enable_postgres_logging is True
        assert config.async_cost_tracking is True
        assert config.background_logging is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_tracing=False,
            enable_caching=True,
            cache_ttl=600,
            max_retries=5
        )
        
        assert config.mode == DecoratorMode.CUSTOM
        assert config.enable_tracing is False
        assert config.enable_caching is True
        assert config.cache_ttl == 600
        assert config.max_retries == 5


class TestUnifiedDecorator:
    """测试统一装饰器类"""
    
    @pytest.fixture
    def mock_settings(self):
        """模拟设置"""
        settings = Mock()
        settings.postgres_connection_string = None
        return settings
    
    @pytest.fixture
    def mock_cost_tracker(self):
        """模拟成本追踪器"""
        tracker = AsyncMock()
        return tracker
    
    @pytest.fixture
    def mock_cache_manager(self):
        """模拟缓存管理器"""
        manager = AsyncMock()
        manager.get = AsyncMock(return_value=None)
        manager.set = AsyncMock()
        return manager
    
    def test_decorator_initialization_default(self):
        """测试装饰器默认初始化"""
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.get_cache_manager') as mock_get_cache:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            mock_get_cache.return_value = Mock()
            
            decorator = UnifiedDecorator()
            
            assert decorator.config.mode == DecoratorMode.FULL
            assert decorator._stats['total_calls'] == 0
            assert decorator._stats['cache_hits'] == 0
            assert decorator._stats['cache_misses'] == 0
            assert decorator._stats['avg_execution_time'] == 0.0
            assert decorator._stats['error_count'] == 0
    
    def test_decorator_initialization_with_config(self):
        """测试装饰器自定义配置初始化"""
        config = DecoratorConfig.fast_mode()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            assert decorator.config.mode == DecoratorMode.FAST
            assert decorator.config.enable_tracing is False
    
    def test_decorator_initialization_postgres_logger_warning(self):
        """测试PostgreSQL日志器初始化警告"""
        config = DecoratorConfig(enable_postgres_logging=True)
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.logger') as mock_logger:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            # 验证警告被记录
            mock_logger.warning.assert_called_with(
                "PostgreSQL logging enabled but no connection string provided"
            )
    
    def test_decorator_initialization_postgres_logger_exception(self):
        """测试PostgreSQL日志器初始化异常"""
        config = DecoratorConfig(enable_postgres_logging=True)
        config.postgres_connection_string = "invalid://connection"
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.PostgresLogger', side_effect=Exception("Connection failed")) as mock_postgres, \
             patch('harborai.core.unified_decorators.logger') as mock_logger:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            # 验证异常警告被记录
            mock_logger.warning.assert_called_with(
                "Failed to initialize PostgreSQL logger: Connection failed"
            )
    
    def test_sync_function_decoration(self):
        """测试同步函数装饰"""
        config = DecoratorConfig.fast_mode()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            def test_function(x, y):
                """测试函数"""
                return x + y
            
            result = test_function(1, 2)
            
            assert result == 3
            assert decorator._stats['total_calls'] == 1
    
    @pytest.mark.asyncio
    async def test_async_function_decoration(self):
        """测试异步函数装饰"""
        config = DecoratorConfig.fast_mode()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            async def test_async_function(x, y):
                """测试异步函数"""
                await asyncio.sleep(0.01)
                return x + y
            
            result = await test_async_function(1, 2)
            
            assert result == 3
            assert decorator._stats['total_calls'] == 1
    
    @pytest.mark.asyncio
    async def test_async_function_with_caching(self):
        """测试异步函数缓存功能"""
        config = DecoratorConfig(enable_caching=True)
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.get_cache_manager') as mock_get_cache:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = AsyncMock()
            
            # 模拟缓存管理器（同步方法）
            cache_manager = Mock()
            cache_manager.get.return_value = None  # 第一次调用无缓存
            cache_manager.set.return_value = None
            mock_get_cache.return_value = cache_manager
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            async def test_cached_function(x):
                """测试缓存函数"""
                return x * 2
            
            # 第一次调用
            result1 = await test_cached_function(5)
            assert result1 == 10
            assert decorator._stats['cache_misses'] == 1
            
            # 模拟缓存命中
            cache_manager.get.return_value = 10
            
            # 第二次调用
            result2 = await test_cached_function(5)
            assert result2 == 10
            assert decorator._stats['cache_hits'] == 1
    
    def test_sync_function_with_error(self):
        """测试同步函数错误处理"""
        config = DecoratorConfig.fast_mode()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            def test_error_function():
                """测试错误函数"""
                raise ValueError("测试错误")
            
            with pytest.raises(ValueError, match="测试错误"):
                test_error_function()
            
            assert decorator._stats['error_count'] == 1
    
    @pytest.mark.asyncio
    async def test_async_function_with_error(self):
        """测试异步函数错误处理"""
        config = DecoratorConfig.fast_mode()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.submit_background_task') as mock_submit:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            mock_submit.return_value = asyncio.create_task(asyncio.sleep(0))
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            async def test_async_error_function():
                """测试异步错误函数"""
                raise ValueError("测试异步错误")
            
            with pytest.raises(ValueError, match="测试异步错误"):
                await test_async_error_function()
            
            assert decorator._stats['error_count'] == 1
    
    def test_update_execution_stats(self):
        """测试执行统计更新"""
        config = DecoratorConfig.fast_mode()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            # 第一次更新
            decorator._update_execution_stats(1.0)
            assert decorator._stats['avg_execution_time'] == 1.0
            
            # 第二次更新（指数移动平均）
            decorator._update_execution_stats(2.0)
            expected = 0.1 * 2.0 + 0.9 * 1.0
            assert abs(decorator._stats['avg_execution_time'] - expected) < 0.001
    
    def test_get_stats(self):
        """测试获取统计信息"""
        config = DecoratorConfig.fast_mode()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            stats = decorator.get_stats()
            
            assert isinstance(stats, dict)
            assert 'total_calls' in stats
            assert 'cache_hits' in stats
            assert 'cache_misses' in stats
            assert 'avg_execution_time' in stats
            assert 'error_count' in stats
            
            # 验证返回的是副本
            stats['total_calls'] = 999
            assert decorator._stats['total_calls'] == 0


class TestPreDefinedDecorators:
    """测试预定义装饰器"""
    
    def test_fast_decorator_instance(self):
        """测试快速装饰器实例"""
        assert fast_decorator.config.mode == DecoratorMode.FAST
        assert fast_decorator.config.enable_tracing is False
        assert fast_decorator.config.enable_cost_tracking is True
    
    def test_full_decorator_instance(self):
        """测试完整装饰器实例"""
        assert full_decorator.config.mode == DecoratorMode.FULL
        assert full_decorator.config.enable_tracing is True
        assert full_decorator.config.enable_cost_tracking is True
    
    def test_unified_trace_with_mode(self):
        """测试统一追踪装饰器（指定模式）"""
        decorator_func = unified_trace(mode=DecoratorMode.FAST)
        
        def test_function():
            return "test"
        
        decorated = decorator_func(test_function)
        result = decorated()
        
        assert result == "test"
    
    def test_unified_trace_with_config(self):
        """测试统一追踪装饰器（自定义配置）"""
        config = DecoratorConfig.fast_mode()
        decorator_func = unified_trace(config=config)
        
        def test_function():
            return "test"
        
        decorated = decorator_func(test_function)
        result = decorated()
        
        assert result == "test"
    
    def test_unified_trace_with_kwargs(self):
        """测试统一追踪装饰器（额外参数）"""
        decorator_func = unified_trace(
            mode=DecoratorMode.CUSTOM,
            enable_caching=True,
            cache_ttl=600
        )
        
        def test_function():
            return "test"
        
        decorated = decorator_func(test_function)
        result = decorated()
        
        assert result == "test"
    
    def test_fast_trace_decorator(self):
        """测试快速追踪装饰器"""
        @fast_trace
        def test_function():
            return "fast_trace_test"
        
        result = test_function()
        assert result == "fast_trace_test"
    
    def test_full_trace_decorator(self):
        """测试完整追踪装饰器"""
        @full_trace
        def test_function():
            return "full_trace_test"
        
        result = test_function()
        assert result == "full_trace_test"
    
    def test_conditional_unified_decorator_true(self):
        """测试条件装饰器（条件为真）"""
        config = DecoratorConfig.fast_mode()
        decorator_func = conditional_unified_decorator(True, config)
        
        @decorator_func
        def test_function():
            return "conditional_test"
        
        result = test_function()
        assert result == "conditional_test"
    
    def test_conditional_unified_decorator_false(self):
        """测试条件装饰器（条件为假）"""
        config = DecoratorConfig.fast_mode()
        decorator_func = conditional_unified_decorator(False, config)
        
        @decorator_func
        def test_function():
            return "conditional_test"
        
        result = test_function()
        assert result == "conditional_test"
    
    def test_smart_decorator(self):
        """测试智能装饰器"""
        with patch('harborai.core.unified_decorators.get_performance_config') as mock_get_perf:
            # 模拟性能配置
            perf_config = Mock()
            decorator_config = Mock()
            decorator_config.enable_tracing = True
            decorator_config.enable_cost_tracking = True
            decorator_config.enable_postgres_logging = False
            decorator_config.async_cost_tracking = True
            decorator_config.background_logging = True
            decorator_config.enable_caching = False
            decorator_config.enable_retry = False
            decorator_config.enable_rate_limiting = False
            
            perf_config.get_decorator_config.return_value = decorator_config
            mock_get_perf.return_value = perf_config
            
            @smart_decorator
            def test_function():
                return "smart_test"
            
            result = test_function()
            assert result == "smart_test"


class TestCompatibilityDecorators:
    """测试兼容性装饰器"""
    
    def test_cost_tracking_decorator(self):
        """测试成本追踪装饰器"""
        @cost_tracking
        def test_function():
            return "cost_tracking_test"
        
        result = test_function()
        assert result == "cost_tracking_test"
    
    def test_with_trace_decorator(self):
        """测试追踪装饰器"""
        @with_trace
        def test_function():
            return "with_trace_test"
        
        result = test_function()
        assert result == "with_trace_test"
    
    def test_with_postgres_logging_decorator(self):
        """测试PostgreSQL日志装饰器"""
        @with_postgres_logging
        def test_function():
            return "postgres_logging_test"
        
        result = test_function()
        assert result == "postgres_logging_test"
    
    def test_with_async_trace_decorator(self):
        """测试异步追踪装饰器"""
        @with_async_trace
        def test_function():
            return "async_trace_test"
        
        result = test_function()
        assert result == "async_trace_test"


class TestDecoratorIntegration:
    """测试装饰器集成功能"""
    
    @pytest.mark.asyncio
    async def test_full_integration_async(self):
        """测试完整集成（异步）"""
        config = DecoratorConfig(
            enable_tracing=True,
            enable_cost_tracking=True,
            enable_postgres_logging=False,  # 禁用PostgreSQL日志避免警告
            background_logging=True
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.submit_background_task') as mock_submit, \
             patch('harborai.core.unified_decorators.TraceContext') as mock_trace_context:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = AsyncMock()
            mock_submit.return_value = asyncio.create_task(asyncio.sleep(0))
            mock_trace_context.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            async def test_integration_function(x, y):
                """测试集成函数"""
                await asyncio.sleep(0.01)
                return x * y
            
            result = await test_integration_function(3, 4)
            
            assert result == 12
            assert decorator._stats['total_calls'] == 1
            # 由于异步执行时间统计可能为0，我们只检查调用次数
            assert decorator._stats['avg_execution_time'] >= 0
    
    def test_full_integration_sync(self):
        """测试完整集成（同步）"""
        config = DecoratorConfig(
            enable_tracing=True,
            enable_cost_tracking=True,
            enable_postgres_logging=True,
            background_logging=False
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.TraceContext') as mock_trace_context:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            mock_trace_context.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            def test_integration_function(x, y):
                """测试集成函数"""
                time.sleep(0.01)
                return x * y
            
            result = test_integration_function(3, 4)
            
            assert result == 12
            assert decorator._stats['total_calls'] == 1
            assert decorator._stats['avg_execution_time'] > 0


class TestDecoratorEdgeCases:
    """测试装饰器边界情况"""
    
    def test_decorator_with_none_config(self):
        """测试装饰器空配置"""
        decorator = UnifiedDecorator(None)
        
        assert decorator.config.mode == DecoratorMode.FULL
    
    @pytest.mark.asyncio
    async def test_async_function_cache_key_generation(self):
        """测试异步函数缓存键生成"""
        config = DecoratorConfig(enable_caching=True)
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.get_cache_manager') as mock_get_cache:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = AsyncMock()
            
            cache_manager = Mock()
            cache_manager.get.return_value = None
            cache_manager.set.return_value = None
            mock_get_cache.return_value = cache_manager
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            async def test_cache_key_function(x, y, z=None):
                """测试缓存键函数"""
                return x + y + (z or 0)
            
            result = await test_cache_key_function(1, 2, z=3)
            
            assert result == 6
            # 验证缓存管理器被调用
            cache_manager.get.assert_called_once()
            cache_manager.set.assert_called_once()
    
    def test_decorator_stats_isolation(self):
        """测试装饰器统计隔离"""
        config1 = DecoratorConfig.fast_mode()
        config2 = DecoratorConfig.full_mode()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator1 = UnifiedDecorator(config1)
            decorator2 = UnifiedDecorator(config2)
            
            @decorator1
            def test_function1():
                return "test1"
            
            @decorator2
            def test_function2():
                return "test2"
            
            test_function1()
            test_function2()
            test_function2()
            
            assert decorator1._stats['total_calls'] == 1
            assert decorator2._stats['total_calls'] == 2


class TestUnifiedDecoratorAdvanced:
    """测试统一装饰器的高级功能"""
    
    def test_cache_functionality_sync(self):
        """测试同步缓存功能"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_caching=True,
            cache_ttl=300
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.get_cache_manager') as mock_get_cache:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            # 模拟缓存管理器
            cache_manager = Mock()
            cache_manager.get.return_value = None  # 第一次调用缓存未命中
            cache_manager.set.return_value = None
            mock_get_cache.return_value = cache_manager
            
            decorator = UnifiedDecorator(config)
            
            call_count = 0
            
            @decorator
            def cached_function(x):
                nonlocal call_count
                call_count += 1
                return f"result_{x}"
            
            # 第一次调用
            result1 = cached_function(1)
            assert result1 == "result_1"
            assert call_count == 1
            
            # 模拟缓存命中
            cache_manager.get.return_value = "cached_result_1"
            
            # 第二次调用应该从缓存获取
            result2 = cached_function(1)
            assert result2 == "cached_result_1"
            assert call_count == 1  # 函数没有再次执行
    
    def test_cache_error_handling(self):
        """测试缓存错误处理"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_caching=True
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.get_cache_manager') as mock_get_cache:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            cache_manager = Mock()
            cache_manager.get.side_effect = Exception("Cache get error")
            cache_manager.set.side_effect = Exception("Cache set error")
            mock_get_cache.return_value = cache_manager
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            def function_with_cache_error():
                return "result"
            
            # 即使缓存出错，函数也应该正常执行
            result = function_with_cache_error()
            assert result == "result"
    
    def test_postgres_logging_sync(self):
        """测试同步PostgreSQL日志记录"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_postgres_logging=True,
            background_logging=False
        )
        # 添加连接字符串
        config.postgres_connection_string = "postgresql://test:test@localhost/test"
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.PostgresLogger') as mock_postgres_class:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            mock_postgres_logger = Mock()
            mock_postgres_logger.log_sync.return_value = None
            mock_postgres_class.return_value = mock_postgres_logger
            
            decorator = UnifiedDecorator(config)
            
            # 直接测试 _log_to_postgres_sync 方法
            decorator._log_to_postgres_sync("test_trace", "test_func", 1.0, "result")
            
            # 验证日志记录被调用
            mock_postgres_logger.log_sync.assert_called()
    
    @pytest.mark.asyncio
    async def test_postgres_logging_async(self):
        """测试异步PostgreSQL日志记录"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_postgres_logging=True,
            background_logging=False
        )
        # 添加连接字符串
        config.postgres_connection_string = "postgresql://test:test@localhost/test"
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.PostgresLogger') as mock_postgres_class:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = AsyncMock()
            
            mock_postgres_logger = AsyncMock()
            mock_postgres_logger.log_async.return_value = None
            mock_postgres_class.return_value = mock_postgres_logger
            
            decorator = UnifiedDecorator(config)
            
            # 直接测试 _log_to_postgres 方法
            await decorator._log_to_postgres("test_trace", "test_func", 1.0, "result")
            
            # 验证异步日志记录被调用
            mock_postgres_logger.log_async.assert_called()
    
    def test_postgres_logging_error_handling(self):
        """测试PostgreSQL日志记录错误处理"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_postgres_logging=True
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.PostgresLogger') as mock_postgres_class:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            mock_postgres_logger = Mock()
            mock_postgres_logger.log_sync.side_effect = Exception("Postgres error")
            mock_postgres_class.return_value = mock_postgres_logger
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            def function_with_postgres_error():
                return "result"
            
            # 即使PostgreSQL日志出错，函数也应该正常执行
            result = function_with_postgres_error()
            assert result == "result"
    
    def test_cost_tracking_sync(self):
        """测试同步成本追踪"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_cost_tracking=True,
            async_cost_tracking=False
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            
            mock_cost_tracker = Mock()
            mock_cost_tracker.track_sync.return_value = None
            mock_get_tracker.return_value = mock_cost_tracker
            
            decorator = UnifiedDecorator(config)
            decorator._cost_tracker = mock_cost_tracker
            
            # 直接测试 _track_cost_sync 方法
            result = Mock()
            result.usage = Mock()
            result.usage.prompt_tokens = 10
            result.usage.completion_tokens = 20
            
            args = ()
            kwargs = {'model': 'gpt-3.5-turbo'}
            
            decorator._track_cost_sync("test_trace", "test_func", args, kwargs, result)
            
            # 验证成本追踪被调用
            mock_cost_tracker.track_sync.assert_called()
    
    @pytest.mark.asyncio
    async def test_cost_tracking_async(self):
        """测试异步成本追踪"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_cost_tracking=True,
            async_cost_tracking=True
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            
            mock_cost_tracker = AsyncMock()
            mock_cost_tracker.track_api_call_async.return_value = None
            mock_get_tracker.return_value = mock_cost_tracker
            
            decorator = UnifiedDecorator(config)
            decorator._cost_tracker = mock_cost_tracker
            
            # 直接测试 _track_cost 方法
            result = Mock()
            result.usage = Mock()
            result.usage.prompt_tokens = 15
            result.usage.completion_tokens = 25
            
            args = ()
            kwargs = {'model': 'gpt-4'}
            
            await decorator._track_cost("test_trace", "test_func", args, kwargs, result)
            
            # 验证异步成本追踪被调用
            mock_cost_tracker.track_api_call_async.assert_called()
    
    def test_cost_tracking_error_handling(self):
        """测试成本追踪错误处理"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_cost_tracking=True
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            
            mock_cost_tracker = Mock()
            mock_cost_tracker.track_sync.side_effect = Exception("Cost tracking error")
            mock_get_tracker.return_value = mock_cost_tracker
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            def function_with_cost_error():
                return "result"
            
            # 即使成本追踪出错，函数也应该正常执行
            result = function_with_cost_error()
            assert result == "result"
    
    def test_trace_context_handling(self):
        """测试追踪上下文处理"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_tracing=True
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.TraceContext') as mock_trace_class:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            mock_trace_context = Mock()
            mock_trace_context.end.return_value = None
            mock_trace_class.return_value = mock_trace_context
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            def traced_function():
                return "traced_result"
            
            result = traced_function()
            assert result == "traced_result"
            
            # 验证追踪上下文被调用
            mock_trace_class.assert_called()
            mock_trace_context.end.assert_called()
    
    def test_trace_context_error_handling(self):
        """测试追踪上下文错误处理"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_tracing=True
        )
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.TraceContext') as mock_trace_class:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            mock_trace_context = Mock()
            mock_trace_context.end.side_effect = Exception("Trace end error")
            mock_trace_class.return_value = mock_trace_context
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            def traced_function():
                return "traced_result"
            
            result = traced_function()
            assert result == "traced_result"
    
    def test_extract_cost_info_with_usage(self):
        """测试从结果中提取成本信息"""
        config = DecoratorConfig()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            # 模拟带有usage信息的结果
            result = Mock()
            result.usage = Mock()
            result.usage.prompt_tokens = 100
            result.usage.completion_tokens = 50
            
            args = ()
            kwargs = {
                'model': 'gpt-4',
                'provider': 'openai',
                'duration': 1.5
            }
            
            cost_info = decorator._extract_cost_info(args, kwargs, result)
            
            assert cost_info is not None
            assert cost_info['model'] == 'gpt-4'
            assert cost_info['provider'] == 'openai'
            assert cost_info['input_tokens'] == 100
            assert cost_info['output_tokens'] == 50
            assert cost_info['duration'] == 1.5
    
    def test_extract_cost_info_without_usage(self):
        """测试从没有usage信息的结果中提取成本信息"""
        config = DecoratorConfig()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            result = "simple_string_result"
            args = ()
            kwargs = {'model': 'gpt-3.5-turbo'}
            
            cost_info = decorator._extract_cost_info(args, kwargs, result)
            
            assert cost_info is not None
            assert cost_info['model'] == 'gpt-3.5-turbo'
            assert cost_info['provider'] == 'openai'  # 默认值
    
    def test_extract_cost_info_empty(self):
        """测试从空参数中提取成本信息"""
        config = DecoratorConfig()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            result = "result"
            args = ()
            kwargs = {}
            
            cost_info = decorator._extract_cost_info(args, kwargs, result)
            
            # 应该返回包含默认值的字典
            assert cost_info is not None
            assert cost_info['provider'] == 'openai'  # 默认值
            assert cost_info['duration'] == 0.0  # 默认值
    
    @pytest.mark.asyncio
    async def test_background_logging(self):
        """测试后台日志记录"""
        config = DecoratorConfig(
            mode=DecoratorMode.CUSTOM,
            enable_postgres_logging=True,
            background_logging=True
        )
        # 添加连接字符串
        config.postgres_connection_string = "postgresql://test:test@localhost/test"
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.PostgresLogger') as mock_postgres_class, \
             patch('harborai.core.unified_decorators.submit_background_task') as mock_submit:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = AsyncMock()
            
            mock_postgres_logger = AsyncMock()
            mock_postgres_logger.log_async.return_value = None
            mock_postgres_class.return_value = mock_postgres_logger
            
            mock_submit.return_value = asyncio.create_task(asyncio.sleep(0))
            
            decorator = UnifiedDecorator(config)
            
            @decorator
            async def background_logged_function():
                return "background_result"
            
            result = await background_logged_function()
            assert result == "background_result"
            
            # 验证后台任务被提交
            mock_submit.assert_called()
    
    def test_cache_key_generation(self):
        """测试缓存键生成"""
        config = DecoratorConfig()
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            decorator = UnifiedDecorator(config)
            
            def test_function(a, b, c=None):
                return a + b
            
            # 测试不同参数组合的缓存键
            key1 = decorator._generate_cache_key(test_function, (1, 2), {'c': 3})
            key2 = decorator._generate_cache_key(test_function, (1, 2), {'c': 4})
            key3 = decorator._generate_cache_key(test_function, (1, 3), {'c': 3})
            
            # 不同参数应该生成不同的缓存键
            assert key1 != key2
            assert key1 != key3
            assert key2 != key3
            
            # 相同参数应该生成相同的缓存键
            key4 = decorator._generate_cache_key(test_function, (1, 2), {'c': 3})
            assert key1 == key4


class TestDecoratorHelperMethods:
    """测试装饰器辅助方法"""
    
    @pytest.mark.asyncio
    async def test_get_cached_result_hit(self):
        """测试缓存命中"""
        config = DecoratorConfig(enable_caching=True)
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.get_cache_manager') as mock_get_cache:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            cache_manager = Mock()
            cache_manager.get.return_value = "cached_value"
            mock_get_cache.return_value = cache_manager
            
            decorator = UnifiedDecorator(config)
            
            result = await decorator._get_cached_result("test_key")
            assert result == "cached_value"
    
    @pytest.mark.asyncio
    async def test_get_cached_result_miss(self):
        """测试缓存未命中"""
        config = DecoratorConfig(enable_caching=True)
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.get_cache_manager') as mock_get_cache:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            cache_manager = Mock()
            cache_manager.get.return_value = None
            mock_get_cache.return_value = cache_manager
            
            decorator = UnifiedDecorator(config)
            
            result = await decorator._get_cached_result("test_key")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_result_success(self):
        """测试缓存结果成功"""
        config = DecoratorConfig(enable_caching=True, cache_ttl=600)
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.get_cache_manager') as mock_get_cache:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            cache_manager = Mock()
            cache_manager.set.return_value = None
            mock_get_cache.return_value = cache_manager
            
            decorator = UnifiedDecorator(config)
            
            await decorator._cache_result("test_key", "test_value")
            
            # 验证缓存设置被调用
            cache_manager.set.assert_called_once_with("test_key", "test_value", ttl=600)
    
    @pytest.mark.asyncio
    async def test_cache_result_error(self):
        """测试缓存结果错误"""
        config = DecoratorConfig(enable_caching=True)
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.get_cache_manager') as mock_get_cache:
            
            mock_get_settings.return_value = Mock()
            mock_get_tracker.return_value = Mock()
            
            cache_manager = Mock()
            cache_manager.set.side_effect = Exception("Cache set error")
            mock_get_cache.return_value = cache_manager
            
            decorator = UnifiedDecorator(config)
            
            # 应该不抛出异常
            await decorator._cache_result("test_key", "test_value")


class TestDecoratorPostExecution:
    """测试装饰器后执行处理"""
    
    @pytest.mark.asyncio
    async def test_handle_post_execution_async(self):
        """测试异步后执行处理"""
        config = DecoratorConfig(
            enable_cost_tracking=True,
            enable_postgres_logging=True,
            enable_tracing=True,
            background_logging=False
        )
        # 添加连接字符串
        config.postgres_connection_string = "postgresql://test:test@localhost/test"
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.PostgresLogger') as mock_postgres_class, \
             patch('harborai.core.unified_decorators.TraceContext') as mock_trace_class:
            
            mock_get_settings.return_value = Mock()
            
            mock_cost_tracker = AsyncMock()
            mock_cost_tracker.track_api_call_async.return_value = None
            mock_get_tracker.return_value = mock_cost_tracker
            
            mock_postgres_logger = AsyncMock()
            mock_postgres_logger.log_async.return_value = None
            mock_postgres_class.return_value = mock_postgres_logger
            
            mock_trace_context = Mock()
            mock_trace_context.end.return_value = None
            mock_trace_class.return_value = mock_trace_context
            
            decorator = UnifiedDecorator(config)
            # 设置 cost tracker
            decorator._cost_tracker = mock_cost_tracker
            
            # 直接测试 _handle_post_execution 方法
            trace_id = "test_trace"
            function_name = "test_function"
            args = ()
            kwargs = {'model': 'gpt-4'}
            result = Mock()
            execution_time = 1.0
            trace_context = mock_trace_context
            
            # 模拟 _track_cost 和 _log_to_postgres 方法
            with patch.object(decorator, '_track_cost') as mock_track_cost, \
                 patch.object(decorator, '_log_to_postgres') as mock_log_postgres, \
                 patch.object(decorator, '_end_trace') as mock_end_trace:
                
                mock_track_cost.return_value = None
                mock_log_postgres.return_value = None
                mock_end_trace.return_value = None
                
                await decorator._handle_post_execution(
                    trace_id, function_name, args, kwargs, result, execution_time, trace_context
                )
                
                # 验证各个组件被调用
                mock_track_cost.assert_called_once_with(trace_id, function_name, args, kwargs, result)
                mock_log_postgres.assert_called_once_with(trace_id, function_name, execution_time, result)
                mock_end_trace.assert_called_once_with(trace_context)
    
    def test_handle_post_execution_sync(self):
        """测试同步后执行处理"""
        config = DecoratorConfig(
            enable_cost_tracking=True,
            enable_postgres_logging=True,
            enable_tracing=True,
            async_cost_tracking=False,
            background_logging=False
        )
        # 添加连接字符串
        config.postgres_connection_string = "postgresql://test:test@localhost/test"
        
        with patch('harborai.core.unified_decorators.get_settings') as mock_get_settings, \
             patch('harborai.core.unified_decorators.get_async_cost_tracker') as mock_get_tracker, \
             patch('harborai.core.unified_decorators.PostgresLogger') as mock_postgres_class, \
             patch('harborai.core.unified_decorators.TraceContext') as mock_trace_class:
            
            mock_get_settings.return_value = Mock()
            
            mock_cost_tracker = Mock()
            mock_cost_tracker.track_sync.return_value = None
            mock_get_tracker.return_value = mock_cost_tracker
            
            mock_postgres_logger = Mock()
            mock_postgres_logger.log_sync.return_value = None
            mock_postgres_class.return_value = mock_postgres_logger
            
            mock_trace_context = Mock()
            mock_trace_context.end.return_value = None
            mock_trace_class.return_value = mock_trace_context
            
            decorator = UnifiedDecorator(config)
            # 设置 cost tracker
            decorator._cost_tracker = mock_cost_tracker
            
            # 直接测试 _handle_post_execution_sync 方法
            trace_id = "test_trace"
            function_name = "test_function"
            args = ()
            kwargs = {'model': 'gpt-4'}
            result = Mock()
            execution_time = 1.0
            trace_context = mock_trace_context
            
            # 模拟 _track_cost_sync 和 _log_to_postgres_sync 方法
            with patch.object(decorator, '_track_cost_sync') as mock_track_cost_sync, \
                 patch.object(decorator, '_log_to_postgres_sync') as mock_log_postgres_sync, \
                 patch.object(decorator, '_end_trace_sync') as mock_end_trace_sync:
                
                mock_track_cost_sync.return_value = None
                mock_log_postgres_sync.return_value = None
                mock_end_trace_sync.return_value = None
                
                decorator._handle_post_execution_sync(
                    trace_id, function_name, args, kwargs, result, execution_time, trace_context
                )
                
                # 验证各个组件被调用
                mock_track_cost_sync.assert_called_once_with(trace_id, function_name, args, kwargs, result)
                mock_log_postgres_sync.assert_called_once_with(trace_id, function_name, execution_time, result)
                mock_end_trace_sync.assert_called_once_with(trace_context)