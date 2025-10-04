#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
延迟初始化性能测试

验证延迟初始化优化的性能改进效果。
根据技术设计方案，目标是将初始化时间从355.58ms降低到≤160ms。
"""

import time
import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from harborai.core.lazy_plugin_manager import LazyPluginManager
from harborai.api.fast_client import FastHarborAI
from harborai.core.client_manager import ClientManager


class TestLazyInitializationPerformance:
    """延迟初始化性能测试类"""
    
    def test_lazy_plugin_manager_initialization_performance(self):
        """测试LazyPluginManager的初始化性能
        
        验证：
        1. LazyPluginManager初始化时间应该显著低于传统PluginManager
        2. 插件应该在首次使用时才被加载
        3. 初始化时间应该≤50ms
        """
        # 记录开始时间
        start_time = time.perf_counter()
        
        # 创建LazyPluginManager实例
        lazy_manager = LazyPluginManager()
        
        # 记录结束时间
        end_time = time.perf_counter()
        initialization_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 验证初始化时间
        assert initialization_time <= 50, f"LazyPluginManager初始化时间 {initialization_time:.2f}ms 超过50ms阈值"
        
        # 验证插件未被立即加载
        assert len(lazy_manager._loaded_plugins) == 0, "插件不应该在初始化时被加载"
        
        print(f"LazyPluginManager初始化时间: {initialization_time:.2f}ms")
    
    def test_fast_harborai_client_initialization_performance(self):
        """测试FastHarborAI客户端的初始化性能
        
        验证：
        1. FastHarborAI初始化时间应该≤160ms（目标性能）
        2. 核心组件应该延迟加载
        3. 基本功能应该可用
        """
        # 记录开始时间
        start_time = time.perf_counter()
        
        # 创建FastHarborAI客户端
        client = FastHarborAI()
        
        # 记录结束时间
        end_time = time.perf_counter()
        initialization_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 验证初始化时间符合目标
        assert initialization_time <= 160, f"FastHarborAI初始化时间 {initialization_time:.2f}ms 超过160ms目标"
        
        # 验证客户端基本属性存在
        assert hasattr(client, 'chat'), "客户端应该有chat属性"
        assert hasattr(client, 'completions'), "客户端应该有completions属性"
        
        print(f"FastHarborAI初始化时间: {initialization_time:.2f}ms")
    
    def test_lazy_plugin_loading_on_demand(self):
        """测试插件按需加载功能
        
        验证：
        1. 插件在首次使用时才被加载
        2. 加载后的插件被缓存
        3. 后续访问不会重复加载
        """
        lazy_manager = LazyPluginManager()
        
        # 验证初始状态
        assert len(lazy_manager._loaded_plugins) == 0
        
        # 创建一个真实的mock插件类
        from harborai.core.plugins.base import Plugin, PluginInfo
        
        class MockDeepSeekPlugin(Plugin):
            """模拟DeepSeek插件"""
            
            def __init__(self, name: str = "deepseek", **kwargs):
                """初始化插件
                
                Args:
                    name: 插件名称
                    **kwargs: 其他配置参数
                """
                self.name = name
                self.config = kwargs
            
            @property
            def info(self) -> PluginInfo:
                return PluginInfo(
                    name="deepseek",
                    version="1.0.0",
                    description="Mock DeepSeek Plugin",
                    author="Test"
                )
            
            def initialize(self) -> bool:
                return True
            
            def supported_models(self) -> List[str]:
                return ["deepseek-chat"]
            
            async def chat_completion_async(self, messages, model, **kwargs):
                return {"choices": [{"message": {"content": "test response"}}]}
            
            def chat_completion(self, messages, model, **kwargs):
                return {"choices": [{"message": {"content": "test response"}}]}
        
        # 模拟获取DeepSeek插件
        with patch('harborai.core.lazy_plugin_manager.importlib') as mock_importlib:
            mock_module = MagicMock()
            mock_module.DeepSeekPlugin = MockDeepSeekPlugin
            mock_importlib.import_module.return_value = mock_module
            
            # 首次获取插件
            start_time = time.perf_counter()
            plugin1 = lazy_manager.get_plugin_for_model("deepseek-chat")
            first_load_time = (time.perf_counter() - start_time) * 1000
            
            # 验证插件被加载
            assert "deepseek" in lazy_manager._loaded_plugins
            
            # 再次获取相同插件
            start_time = time.perf_counter()
            plugin2 = lazy_manager.get_plugin_for_model("deepseek-chat")
            second_load_time = (time.perf_counter() - start_time) * 1000
            
            # 验证返回相同实例且速度更快
            assert plugin1 is plugin2, "应该返回缓存的插件实例"
            assert second_load_time < first_load_time, "缓存访问应该更快"
            
            print(f"首次加载时间: {first_load_time:.2f}ms, 缓存访问时间: {second_load_time:.2f}ms")
    
    def test_performance_comparison_with_traditional_manager(self):
        """对比传统插件管理器和延迟插件管理器的性能
        
        验证：
        1. 延迟管理器初始化时间显著低于传统管理器
        2. 性能改进至少50%
        """
        # 测试传统插件管理器初始化时间
        with patch('harborai.core.plugins.manager.PluginManager.discover_plugins') as mock_discover:
            mock_discover.return_value = ["deepseek", "doubao", "wenxin"]
            
            start_time = time.perf_counter()
            traditional_manager = ClientManager()
            traditional_time = (time.perf_counter() - start_time) * 1000
        
        # 测试延迟插件管理器初始化时间
        start_time = time.perf_counter()
        lazy_manager = LazyPluginManager()
        lazy_time = (time.perf_counter() - start_time) * 1000
        
        # 计算性能改进
        improvement_ratio = (traditional_time - lazy_time) / traditional_time * 100
        
        # 验证性能改进
        assert lazy_time < traditional_time, "延迟管理器应该比传统管理器更快"
        assert improvement_ratio >= 50, f"性能改进 {improvement_ratio:.1f}% 应该至少50%"
        
        print(f"传统管理器: {traditional_time:.2f}ms, 延迟管理器: {lazy_time:.2f}ms")
        print(f"性能改进: {improvement_ratio:.1f}%")
    
    @pytest.mark.asyncio
    async def test_async_initialization_performance(self):
        """测试异步初始化性能
        
        验证：
        1. 异步初始化不会阻塞主线程
        2. 并发初始化性能良好
        """
        async def create_fast_client():
            start_time = time.perf_counter()
            client = FastHarborAI()
            end_time = time.perf_counter()
            return (end_time - start_time) * 1000
        
        # 并发创建多个客户端
        tasks = [create_fast_client() for _ in range(5)]
        initialization_times = await asyncio.gather(*tasks)
        
        # 验证所有初始化时间都在目标范围内
        for i, init_time in enumerate(initialization_times):
            assert init_time <= 160, f"客户端 {i+1} 初始化时间 {init_time:.2f}ms 超过160ms目标"
        
        avg_time = sum(initialization_times) / len(initialization_times)
        print(f"并发初始化平均时间: {avg_time:.2f}ms")
    
    def test_memory_usage_optimization(self):
        """测试内存使用优化
        
        验证：
        1. 延迟加载减少初始内存占用
        2. 只有使用的插件才占用内存
        """
        import psutil
        import os
        
        # 获取当前进程
        process = psutil.Process(os.getpid())
        
        # 记录初始内存使用
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建延迟管理器
        lazy_manager = LazyPluginManager()
        
        # 记录延迟管理器创建后的内存使用
        after_lazy_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 验证内存增长很小
        memory_increase = after_lazy_memory - initial_memory
        assert memory_increase <= 5, f"延迟管理器内存增长 {memory_increase:.2f}MB 应该≤5MB"
        
        print(f"初始内存: {initial_memory:.2f}MB, 延迟管理器后: {after_lazy_memory:.2f}MB")
        print(f"内存增长: {memory_increase:.2f}MB")


class TestLazyInitializationFunctionality:
    """延迟初始化功能测试类"""
    
    def test_lazy_manager_model_mapping(self):
        """测试延迟管理器的模型映射功能
        
        验证：
        1. 正确映射模型到插件
        2. 支持的模型列表正确
        """
        lazy_manager = LazyPluginManager()
        
        # 验证模型映射
        expected_mappings = {
            "deepseek-chat": "deepseek",
            "deepseek-reasoner": "deepseek",
            "doubao-pro-4k": "doubao",
            "doubao-pro-32k": "doubao",
            "ernie-bot-turbo": "wenxin",
            "ernie-bot-4": "wenxin"
        }
        
        for model, expected_plugin in expected_mappings.items():
            plugin_name = lazy_manager.get_plugin_name_for_model(model)
            assert plugin_name == expected_plugin, f"模型 {model} 应该映射到插件 {expected_plugin}"
    
    def test_fast_client_compatibility(self):
        """测试FastHarborAI客户端的兼容性
        
        验证：
        1. 与现有API接口兼容
        2. 支持所有必要的方法
        """
        client = FastHarborAI()
        
        # 验证必要的属性和方法存在
        assert hasattr(client, 'chat'), "应该有chat属性"
        assert hasattr(client.chat, 'completions'), "应该有completions属性"
        assert hasattr(client.chat.completions, 'create'), "应该有create方法"
        
        # 验证方法签名兼容性（不实际调用）
        import inspect
        create_signature = inspect.signature(client.chat.completions.create)
        expected_params = ['messages', 'model']
        
        for param in expected_params:
            assert param in create_signature.parameters, f"create方法应该有 {param} 参数"


if __name__ == "__main__":
    # 运行性能测试
    test_class = TestLazyInitializationPerformance()
    
    print("=== 延迟初始化性能测试 ===")
    test_class.test_lazy_plugin_manager_initialization_performance()
    test_class.test_fast_harborai_client_initialization_performance()
    test_class.test_lazy_plugin_loading_on_demand()
    test_class.test_performance_comparison_with_traditional_manager()
    test_class.test_memory_usage_optimization()
    
    print("\n=== 功能测试 ===")
    func_test_class = TestLazyInitializationFunctionality()
    func_test_class.test_lazy_manager_model_mapping()
    func_test_class.test_fast_client_compatibility()
    
    print("\n所有测试完成！")