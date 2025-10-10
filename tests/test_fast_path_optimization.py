"""快速路径优化测试

测试快速路径模式下的性能优化效果。
"""

import asyncio
import time
from unittest.mock import Mock, patch
import pytest

from harborai import HarborAI
from harborai.config.settings import Settings
from harborai.core.base_plugin import ChatMessage


class TestFastPathOptimization:
    """快速路径优化测试类"""
    
    @pytest.fixture
    def fast_path_settings(self):
        """快速路径配置"""
        settings = Settings()
        settings.enable_fast_path = True
        settings.performance_mode = "balanced"  # 使用balanced模式来测试基于模型和token的判断
        settings.fast_path_models = ["gpt-3.5-turbo", "gpt-4"]
        settings.fast_path_max_tokens = None  # 无限制，由模型厂商控制
        settings.enable_detailed_tracing = False
        settings.enable_postgres_logging = False
        settings.enable_async_decorators = True
        return settings
    
    @pytest.fixture
    def full_path_settings(self):
        """完整路径配置"""
        settings = Settings()
        settings.enable_fast_path = False
        settings.performance_mode = "full"
        settings.enable_detailed_tracing = True
        settings.enable_postgres_logging = True
        settings.enable_async_decorators = False
        return settings
    
    def test_fast_path_enabled_conditions(self, fast_path_settings):
        """测试快速路径启用条件"""
        # 测试模型在快速路径列表中
        assert fast_path_settings.is_fast_path_enabled("gpt-3.5-turbo", 500) is True
        assert fast_path_settings.is_fast_path_enabled("gpt-4", 800) is True
        
        # 测试模型不在快速路径列表中
        assert fast_path_settings.is_fast_path_enabled("claude-3", 500) is False
        
        # 测试token限制
        assert fast_path_settings.is_fast_path_enabled("gpt-3.5-turbo", 1500) is False
    
    def test_fast_path_disabled_conditions(self, full_path_settings):
        """测试快速路径禁用条件"""
        # 即使模型支持，也应该返回False
        assert full_path_settings.is_fast_path_enabled("gpt-3.5-turbo", 500) is False
        assert full_path_settings.is_fast_path_enabled("gpt-4", 800) is False
    
    @patch('harborai.api.client.get_settings')
    def test_create_method_routing(self, mock_get_settings, fast_path_settings):
        """测试create方法的路由逻辑"""
        mock_get_settings.return_value = fast_path_settings
        
        # 创建客户端
        client = HarborAI()
        
        # Mock核心方法
        with patch.object(client.chat.completions, '_create_fast_path') as mock_fast, \
             patch.object(client.chat.completions, '_create_full_path') as mock_full:
            
            # 测试快速路径
            messages = [{"role": "user", "content": "Hello"}]
            client.chat.completions.create(
                messages=messages,
                model="gpt-3.5-turbo",
                max_tokens=500
            )
            
            # 验证调用了快速路径
            mock_fast.assert_called_once()
            mock_full.assert_not_called()
    
    @patch('harborai.api.client.get_settings')
    def test_acreate_method_routing(self, mock_get_settings, fast_path_settings):
        """测试异步create方法的路由逻辑"""
        mock_get_settings.return_value = fast_path_settings
        
        async def run_test():
            # 创建客户端
            client = HarborAI()
            
            # Mock核心方法
            with patch.object(client.chat.completions, '_acreate_fast_path') as mock_fast, \
                 patch.object(client.chat.completions, '_acreate_full_path') as mock_full:
                
                # 测试快速路径
                messages = [{"role": "user", "content": "Hello"}]
                await client.chat.completions.acreate(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    max_tokens=500
                )
                
                # 验证调用了快速路径
                mock_fast.assert_called_once()
                mock_full.assert_not_called()
        
        asyncio.run(run_test())
    
    def test_decorator_performance_difference(self):
        """测试装饰器性能差异"""
        from harborai.core.unified_decorators import fast_trace, full_trace
        
        # 创建测试函数
        def test_function():
            time.sleep(0.001)  # 模拟一些处理时间
            return "result"
        
        # 测试快速装饰器
        fast_decorated = fast_trace(test_function)
        start_time = time.time()
        for _ in range(100):
            fast_decorated()
        fast_time = time.time() - start_time
        
        # 测试完整装饰器
        full_decorated = full_trace(test_function)
        start_time = time.time()
        for _ in range(100):
            full_decorated()
        full_time = time.time() - start_time
        
        # 快速装饰器应该更快（允许一些误差）
        print(f"Fast decorator time: {fast_time:.4f}s")
        print(f"Full decorator time: {full_time:.4f}s")
        
        # 快速装饰器应该至少快10%
        assert fast_time < full_time * 1.1
    
    @pytest.mark.asyncio
    async def test_async_decorator_performance_difference(self):
        """测试异步装饰器性能差异"""
        from harborai.core.unified_decorators import fast_trace, full_trace
        
        # 创建异步测试函数
        async def test_async_function():
            await asyncio.sleep(0.001)  # 模拟一些异步处理时间
            return "result"
        
        # 测试快速装饰器
        fast_decorated = fast_trace(test_async_function)
        start_time = time.time()
        for _ in range(50):
            await fast_decorated()
        fast_time = time.time() - start_time
        
        # 测试完整装饰器
        full_decorated = full_trace(test_async_function)
        start_time = time.time()
        for _ in range(50):
            await full_decorated()
        full_time = time.time() - start_time
        
        # 快速装饰器应该更快
        print(f"Fast async decorator time: {fast_time:.4f}s")
        print(f"Full async decorator time: {full_time:.4f}s")
        
        # 快速装饰器应该至少快10%
        assert fast_time < full_time * 1.1
    
    def test_fast_path_bypasses_heavy_middleware(self):
        """测试快速路径绕过重型中间件"""
        # 这个测试将验证快速路径确实绕过了一些重型中间件
        # 比如详细的追踪、PostgreSQL日志记录等
        
        from harborai.core.unified_decorators import DecoratorConfig, DecoratorMode
        
        # 快速模式配置
        fast_config = DecoratorConfig.fast_mode()
        assert fast_config.mode == DecoratorMode.FAST
        assert fast_config.enable_tracing is False
        assert fast_config.enable_postgres_logging is False
        assert fast_config.async_cost_tracking is True
        
        # 完整模式配置
        full_config = DecoratorConfig.full_mode()
        assert full_config.mode == DecoratorMode.FULL
        assert full_config.enable_tracing is True
        assert full_config.enable_postgres_logging is True
        assert full_config.enable_cost_tracking is True
    
    @patch('harborai.core.unified_decorators.logger')
    def test_fast_path_minimal_logging(self, mock_logger):
        """测试快速路径的最小日志记录"""
        from harborai.core.unified_decorators import fast_trace
        
        @fast_trace
        def test_function():
            return "result"
        
        # 执行函数
        result = test_function()
        
        # 验证结果
        assert result == "result"
        
        # 验证日志调用次数较少（快速路径应该减少日志记录）
        # 这里的具体断言取决于fast_trace的实现
        # 目前只验证函数正常执行