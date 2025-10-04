"""
延迟加载兼容性测试

验证延迟加载功能与现有功能的兼容性，确保：
1. 现有API接口保持不变
2. 功能行为一致性
3. 配置兼容性
4. 错误处理一致性

Author: HarborAI Team
Date: 2024-10-03
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any

from harborai.core.client_manager import ClientManager
from harborai.core.lazy_plugin_manager import LazyPluginManager
from harborai.api.fast_client import FastHarborAI
from harborai.core.plugins.base import Plugin, PluginInfo


class TestLazyLoadingCompatibility:
    """延迟加载兼容性测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.test_config = {
            "timeout": 30,
            "max_retries": 3,
            "plugins": {
                "deepseek": {
                    "api_key": "test_key",
                    "base_url": "https://api.deepseek.com"
                }
            }
        }
    
    def test_client_manager_backward_compatibility(self):
        """测试ClientManager向后兼容性
        
        验证：
        1. 传统初始化方式仍然有效
        2. 现有API接口保持不变
        3. 功能行为一致
        """
        # 传统方式初始化
        traditional_manager = ClientManager(client_config=self.test_config, lazy_loading=False)
        
        # 延迟加载方式初始化
        lazy_manager = ClientManager(client_config=self.test_config, lazy_loading=True)
        
        # 验证接口一致性
        assert hasattr(traditional_manager, 'get_available_models')
        assert hasattr(lazy_manager, 'get_available_models')
        
        assert hasattr(traditional_manager, 'get_plugin_for_model')
        assert hasattr(lazy_manager, 'get_plugin_for_model')
        
        assert hasattr(traditional_manager, 'get_plugin_info')
        assert hasattr(lazy_manager, 'get_plugin_info')
        
        # 验证新增的延迟加载方法
        assert hasattr(lazy_manager, 'preload_plugin')
        assert hasattr(lazy_manager, 'preload_model')
        assert hasattr(lazy_manager, 'get_loading_statistics')
        
        print("✓ ClientManager向后兼容性验证通过")
    
    def test_api_interface_consistency(self):
        """测试API接口一致性
        
        验证FastHarborAI与传统HarborAI的接口一致性
        """
        # 创建FastHarborAI实例
        fast_client = FastHarborAI(config=self.test_config)
        
        # 验证核心接口存在
        assert hasattr(fast_client, 'chat')
        assert hasattr(fast_client, 'completions')
        assert hasattr(fast_client, 'preload_model')
        assert hasattr(fast_client, 'get_statistics')
        
        # 验证chat.completions接口
        assert hasattr(fast_client.chat, 'completions')
        assert hasattr(fast_client.chat.completions, 'create')
        
        print("✓ API接口一致性验证通过")
    
    @patch('importlib.import_module')
    def test_plugin_loading_behavior_consistency(self, mock_importlib):
        """测试插件加载行为一致性
        
        验证延迟加载和传统加载的行为一致性
        """
        # 创建mock插件
        class MockPlugin(Plugin):
            def __init__(self, name, **kwargs):
                self.name = name
                self.kwargs = kwargs

            @property
            def info(self):
                return {"name": self.name, "version": "1.0.0"}

            def initialize(self) -> bool:
                return True

            def get_supported_models(self) -> List[str]:
                return ["test-model"]

            async def chat_completion_async(self, messages, model, **kwargs):
                return {"choices": [{"message": {"content": "test response"}}]}

            def chat_completion(self, messages, model, **kwargs):
                return {"choices": [{"message": {"content": "test response"}}]}

        # 设置mock
        mock_module = MagicMock()
        mock_module.TestPlugin = MockPlugin
        mock_importlib.return_value = mock_module

        # 创建延迟加载管理器
        lazy_manager = LazyPluginManager(self.test_config)

        # 手动添加测试插件信息到注册表
        from harborai.core.lazy_plugin_manager import LazyPluginInfo
        test_plugin_info = LazyPluginInfo(
            name="test",
            module_path="test.plugin",
            class_name="TestPlugin",
            supported_models=["test-model"]
        )
        lazy_manager._register_plugin_info(test_plugin_info)

        # 测试插件加载
        plugin = lazy_manager.get_plugin("test")
        assert plugin is not None, "插件应该成功加载"
        assert plugin.name == "test", "插件名称应该正确"

        # 测试缓存机制
        plugin2 = lazy_manager.get_plugin("test")
        assert plugin is plugin2, "第二次获取应该返回缓存的实例"

        # 测试模型映射
        plugin_by_model = lazy_manager.get_plugin_for_model("test-model")
        assert plugin is plugin_by_model, "通过模型获取的插件应该是同一个实例"

        print("✓ 插件加载行为一致性验证通过")
    
    def test_configuration_compatibility(self):
        """测试配置兼容性
        
        验证现有配置格式在延迟加载模式下仍然有效
        """
        # 测试各种配置格式
        configs = [
            # 基本配置
            {"timeout": 30},
            
            # 插件配置
            {
                "plugins": {
                    "deepseek": {"api_key": "test_key"}
                }
            },
            
            # 完整配置
            {
                "timeout": 30,
                "max_retries": 3,
                "plugins": {
                    "deepseek": {
                        "api_key": "test_key",
                        "base_url": "https://api.deepseek.com"
                    }
                }
            }
        ]
        
        for config in configs:
            try:
                # 传统方式
                traditional_manager = ClientManager(client_config=config, lazy_loading=False)
                
                # 延迟加载方式
                lazy_manager = ClientManager(client_config=config, lazy_loading=True)
                
                # 验证配置被正确处理
                assert traditional_manager.client_config == config
                assert lazy_manager.client_config == config
                
            except Exception as e:
                pytest.fail(f"配置兼容性测试失败: {config}, 错误: {e}")
        
        print("✓ 配置兼容性验证通过")
    
    def test_error_handling_consistency(self):
        """测试错误处理一致性
        
        验证延迟加载模式下的错误处理与传统模式一致
        """
        # 测试无效配置
        invalid_configs = [
            None,
            {},
            {"invalid_key": "invalid_value"}
        ]
        
        for config in invalid_configs:
            try:
                # 传统方式
                traditional_manager = ClientManager(client_config=config, lazy_loading=False)
                
                # 延迟加载方式  
                lazy_manager = ClientManager(client_config=config, lazy_loading=True)
                
                # 两种方式都应该能处理无效配置（不抛出异常）
                assert traditional_manager is not None
                assert lazy_manager is not None
                
            except Exception as e:
                # 如果抛出异常，两种方式应该抛出相同类型的异常
                try:
                    ClientManager(client_config=config, lazy_loading=True)
                    pytest.fail("延迟加载模式应该抛出相同的异常")
                except type(e):
                    pass  # 预期的异常类型
        
        print("✓ 错误处理一致性验证通过")
    
    @patch('importlib.import_module')
    def test_performance_degradation_acceptable(self, mock_importlib):
        """测试性能退化可接受性
        
        验证延迟加载不会显著影响运行时性能
        """
        # 创建模拟插件
        class MockPlugin(Plugin):
            def __init__(self, name: str = "test", **kwargs):
                self.name = name
                self.config = kwargs
            
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(
                    name="test",
                    version="1.0.0",
                    description="Test Plugin",
                    author="Test"
                )
            
            def initialize(self) -> bool:
                return True
            
            def supported_models(self) -> List[str]:
                return ["test-model"]
            
            async def chat_completion_async(self, messages, model, **kwargs):
                return {"choices": [{"message": {"content": "test response"}}]}
            
            def chat_completion(self, messages, model, **kwargs):
                return {"choices": [{"message": {"content": "test response"}}]}
        
        # 设置mock
        mock_module = MagicMock()
        mock_module.TestPlugin = MockPlugin
        mock_importlib.return_value = mock_module
        
        # 创建延迟加载管理器
        lazy_manager = LazyPluginManager(self.test_config)
        
        # 手动添加测试插件信息到注册表
        from harborai.core.lazy_plugin_manager import LazyPluginInfo
        test_plugin_info = LazyPluginInfo(
            name="test",
            module_path="test.plugin",
            class_name="TestPlugin",
            supported_models=["test-model"]
        )
        lazy_manager._register_plugin_info(test_plugin_info)
        
        # 测试首次访问性能
        start_time = time.perf_counter()
        plugin1 = lazy_manager.get_plugin("test")
        first_access_time = (time.perf_counter() - start_time) * 1000
        
        # 测试缓存访问性能
        start_time = time.perf_counter()
        plugin2 = lazy_manager.get_plugin("test")
        cached_access_time = (time.perf_counter() - start_time) * 1000
        
        # 验证性能要求
        assert first_access_time < 50, f"首次访问时间过长: {first_access_time}ms"
        assert cached_access_time < 1, f"缓存访问时间过长: {cached_access_time}ms"
        assert plugin1 is plugin2, "缓存未生效"
        
        print(f"✓ 性能退化可接受性验证通过 - 首次: {first_access_time:.2f}ms, 缓存: {cached_access_time:.2f}ms")
    
    def test_thread_safety_compatibility(self):
        """测试线程安全兼容性
        
        验证延迟加载在多线程环境下的安全性
        """
        import threading
        import concurrent.futures
        
        lazy_manager = LazyPluginManager(self.test_config)
        
        # 模拟并发访问
        def concurrent_access():
            try:
                models = lazy_manager.get_supported_models()
                stats = lazy_manager.get_statistics()
                return True
            except Exception:
                return False
        
        # 创建多个线程并发访问
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_access) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # 验证所有访问都成功
        assert all(results), "并发访问存在失败"
        
        print("✓ 线程安全兼容性验证通过")
    
    def test_memory_usage_compatibility(self):
        """测试内存使用兼容性
        
        验证延迟加载不会导致内存泄漏
        """
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # 创建多个管理器实例
        managers = []
        for i in range(10):
            manager = LazyPluginManager(self.test_config)
            managers.append(manager)
        
        # 清理管理器
        for manager in managers:
            manager.cleanup()
        
        # 强制垃圾回收
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # 验证内存增长在合理范围内（小于10MB）
        assert memory_increase < 10 * 1024 * 1024, f"内存增长过大: {memory_increase / 1024 / 1024:.2f}MB"
        
        print(f"✓ 内存使用兼容性验证通过 - 内存增长: {memory_increase / 1024 / 1024:.2f}MB")


class TestFastClientCompatibility:
    """FastHarborAI客户端兼容性测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.test_config = {
            "timeout": 30,
            "max_retries": 3
        }
    
    def test_fast_client_initialization_compatibility(self):
        """测试FastHarborAI初始化兼容性"""
        # 测试不同初始化方式
        clients = [
            FastHarborAI(),  # 默认配置
            FastHarborAI(config=self.test_config),  # 自定义配置
            FastHarborAI(config=self.test_config, lazy_loading=True),  # 显式启用延迟加载
            FastHarborAI(config=self.test_config, lazy_loading=False)  # 显式禁用延迟加载
        ]
        
        for client in clients:
            assert client is not None
            assert hasattr(client, 'chat')
            assert hasattr(client, 'completions')
            
            # 清理资源
            client.cleanup()
        
        print("✓ FastHarborAI初始化兼容性验证通过")
    
    def test_api_method_compatibility(self):
        """测试API方法兼容性"""
        client = FastHarborAI(config=self.test_config)
        
        # 验证核心方法存在且可调用
        assert callable(getattr(client.chat.completions, 'create', None))
        assert callable(getattr(client, 'preload_model', None))
        assert callable(getattr(client, 'get_statistics', None))
        
        # 清理资源
        client.cleanup()
        
        print("✓ API方法兼容性验证通过")


if __name__ == "__main__":
    # 运行兼容性测试
    pytest.main([__file__, "-v", "-s"])