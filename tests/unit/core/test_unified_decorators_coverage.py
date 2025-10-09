"""
unified_decorators.py 覆盖率提升测试

专门针对缺失覆盖的功能进行测试：
- 缓存功能异常处理
- PostgreSQL日志记录功能
- 同步版本的缓存和日志功能
- 边界条件和错误处理
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from harborai.core.unified_decorators import UnifiedDecorator, DecoratorConfig


class TestUnifiedDecoratorCoverage:
    """unified_decorators.py 覆盖率提升测试"""

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
    def decorator_with_cache_and_postgres(self, mock_cache_manager, mock_postgres_logger):
        """创建带缓存和PostgreSQL日志的装饰器"""
        config = DecoratorConfig(
            enable_caching=True,
            cache_ttl=300,
            enable_cost_tracking=True,
            enable_postgres_logging=True
        )
        decorator = UnifiedDecorator(config)
        decorator._cache_manager = mock_cache_manager
        decorator._postgres_logger = mock_postgres_logger
        return decorator

    def test_cache_get_exception_handling(self, decorator_with_cache_and_postgres):
        """测试缓存获取异常处理"""
        # 模拟缓存获取异常
        decorator_with_cache_and_postgres._cache_manager.get.side_effect = Exception("缓存获取失败")
        
        # 测试异步版本
        async def test_async():
            result = await decorator_with_cache_and_postgres._get_cached_result("test_key")
            assert result is None
        
        asyncio.run(test_async())
        
        # 测试同步版本
        result = decorator_with_cache_and_postgres._get_cached_result_sync("test_key")
        assert result is None

    def test_cache_set_exception_handling(self, decorator_with_cache_and_postgres):
        """测试缓存设置异常处理"""
        # 模拟缓存设置异常
        decorator_with_cache_and_postgres._cache_manager.set.side_effect = Exception("缓存设置失败")
        
        # 测试异步版本
        async def test_async():
            # 不应该抛出异常
            await decorator_with_cache_and_postgres._cache_result("test_key", "test_value")
        
        asyncio.run(test_async())
        
        # 测试同步版本
        decorator_with_cache_and_postgres._cache_result_sync("test_key", "test_value")

    def test_postgres_logging_async(self, decorator_with_cache_and_postgres):
        """测试PostgreSQL异步日志记录"""
        async def test_async():
            # 测试正常日志记录
            await decorator_with_cache_and_postgres._log_to_postgres(
                "trace123", "test_func", 0.5, "success"
            )
            
            # 验证调用
            decorator_with_cache_and_postgres._postgres_logger.log_async.assert_called_once()
            call_args = decorator_with_cache_and_postgres._postgres_logger.log_async.call_args[0][0]
            assert call_args['trace_id'] == "trace123"
            assert call_args['function_name'] == "test_func"
            assert call_args['execution_time'] == 0.5
            assert call_args['result_type'] == "str"  # type(result).__name__
        
        asyncio.run(test_async())

    def test_postgres_logging_sync(self, decorator_with_cache_and_postgres):
        """测试PostgreSQL同步日志记录"""
        # 测试正常日志记录
        decorator_with_cache_and_postgres._log_to_postgres_sync(
            "trace123", "test_func", 0.5, "success"
        )
        
        # 验证调用
        decorator_with_cache_and_postgres._postgres_logger.log_sync.assert_called_once()
        call_args = decorator_with_cache_and_postgres._postgres_logger.log_sync.call_args[0][0]
        assert call_args['trace_id'] == "trace123"
        assert call_args['function_name'] == "test_func"
        assert call_args['execution_time'] == 0.5
        assert call_args['result_type'] == "str"  # type(result).__name__

    def test_postgres_logging_exception_handling(self, decorator_with_cache_and_postgres):
        """测试PostgreSQL日志记录异常处理"""
        # 模拟日志记录异常
        decorator_with_cache_and_postgres._postgres_logger.log_async.side_effect = Exception("日志记录失败")
        decorator_with_cache_and_postgres._postgres_logger.log_sync.side_effect = Exception("日志记录失败")
        
        # 测试异步版本不抛出异常
        async def test_async():
            await decorator_with_cache_and_postgres._log_to_postgres(
                "trace123", "test_func", 0.5, "success"
            )
        
        asyncio.run(test_async())
        
        # 测试同步版本不抛出异常
        decorator_with_cache_and_postgres._log_to_postgres_sync(
            "trace123", "test_func", 0.5, "success"
        )

    def test_error_logging_async(self, decorator_with_cache_and_postgres):
        """测试异步错误日志记录"""
        async def test_async():
            await decorator_with_cache_and_postgres._log_error(
                "trace123", "test_func", "测试错误"
            )
            
            # 验证调用
            decorator_with_cache_and_postgres._postgres_logger.log_async.assert_called_once()
            call_args = decorator_with_cache_and_postgres._postgres_logger.log_async.call_args[0][0]
            assert call_args['trace_id'] == "trace123"
            assert call_args['function_name'] == "test_func"
            assert call_args['error'] == "测试错误"
            assert call_args['level'] == 'ERROR'
        
        asyncio.run(test_async())

    def test_error_logging_sync(self, decorator_with_cache_and_postgres):
        """测试同步错误日志记录"""
        decorator_with_cache_and_postgres._log_error_sync(
            "trace123", "test_func", "测试错误"
        )
        
        # 验证调用
        decorator_with_cache_and_postgres._postgres_logger.log_sync.assert_called_once()
        call_args = decorator_with_cache_and_postgres._postgres_logger.log_sync.call_args[0][0]
        assert call_args['trace_id'] == "trace123"
        assert call_args['function_name'] == "test_func"
        assert call_args['error'] == "测试错误"
        assert call_args['level'] == 'ERROR'

    def test_error_logging_exception_handling(self, decorator_with_cache_and_postgres):
        """测试错误日志记录异常处理"""
        # 模拟错误日志记录异常
        decorator_with_cache_and_postgres._postgres_logger.log_async.side_effect = Exception("错误日志记录失败")
        decorator_with_cache_and_postgres._postgres_logger.log_sync.side_effect = Exception("错误日志记录失败")
        
        # 测试异步版本不抛出异常
        async def test_async():
            await decorator_with_cache_and_postgres._log_error(
                "trace123", "test_func", "测试错误"
            )
        
        asyncio.run(test_async())
        
        # 测试同步版本不抛出异常
        decorator_with_cache_and_postgres._log_error_sync(
            "trace123", "test_func", "测试错误"
        )

    def test_cache_disabled_scenarios(self):
        """测试缓存禁用场景"""
        config = DecoratorConfig(enable_caching=False)
        decorator = UnifiedDecorator(config)
        
        # 测试异步版本
        async def test_async():
            result = await decorator._get_cached_result("test_key")
            assert result is None
            
            # 缓存结果应该不执行任何操作
            await decorator._cache_result("test_key", "test_value")
        
        asyncio.run(test_async())
        
        # 测试同步版本
        result = decorator._get_cached_result_sync("test_key")
        assert result is None
        
        decorator._cache_result_sync("test_key", "test_value")

    def test_postgres_logging_disabled_scenarios(self):
        """测试PostgreSQL日志记录禁用场景"""
        config = DecoratorConfig(enable_postgres_logging=False)
        decorator = UnifiedDecorator(config)
        
        # 测试异步版本
        async def test_async():
            await decorator._log_to_postgres("trace123", "test_func", 0.5, "success")
            await decorator._log_error("trace123", "test_func", "测试错误")
        
        asyncio.run(test_async())
        
        # 测试同步版本
        decorator._log_to_postgres_sync("trace123", "test_func", 0.5, "success")
        decorator._log_error_sync("trace123", "test_func", "测试错误")

    @pytest.mark.parametrize("cache_enabled,postgres_enabled", [
        (True, False),
        (False, True),
        (False, False)
    ])
    def test_mixed_feature_configurations(self, cache_enabled, postgres_enabled):
        """测试混合功能配置"""
        config = DecoratorConfig(
            enable_caching=cache_enabled,
            enable_postgres_logging=postgres_enabled
        )
        decorator = UnifiedDecorator(config)
        
        if cache_enabled:
            decorator._cache_manager = Mock()
            decorator._cache_manager.get = Mock(return_value=None)
            decorator._cache_manager.set = Mock()
        
        if postgres_enabled:
            decorator._postgres_logger = Mock()
            decorator._postgres_logger.log_async = AsyncMock()
            decorator._postgres_logger.log_sync = Mock()
        
        # 测试各种组合下的功能
        async def test_async():
            await decorator._get_cached_result("test_key")
            await decorator._cache_result("test_key", "test_value")
            await decorator._log_to_postgres("trace123", "test_func", 0.5, "success")
            await decorator._log_error("trace123", "test_func", "测试错误")
        
        asyncio.run(test_async())
        
        # 同步版本
        decorator._get_cached_result_sync("test_key")
        decorator._cache_result_sync("test_key", "test_value")
        decorator._log_to_postgres_sync("trace123", "test_func", 0.5, "success")
        decorator._log_error_sync("trace123", "test_func", "测试错误")

    def test_cache_manager_none_scenarios(self):
        """测试缓存管理器为None的场景"""
        config = DecoratorConfig(enable_caching=True)
        decorator = UnifiedDecorator(config)
        # 不设置_cache_manager，保持为None
        
        # 测试异步版本
        async def test_async():
            result = await decorator._get_cached_result("test_key")
            assert result is None
            
            # 不应该抛出异常
            await decorator._cache_result("test_key", "test_value")
        
        asyncio.run(test_async())
        
        # 测试同步版本
        result = decorator._get_cached_result_sync("test_key")
        assert result is None
        
        decorator._cache_result_sync("test_key", "test_value")

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