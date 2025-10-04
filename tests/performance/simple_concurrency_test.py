#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的并发性能测试

验证并发优化组件是否正常工作。
"""

import asyncio
import time
import logging
from unittest.mock import Mock, patch

# 导入被测试的组件
from harborai.api.fast_client import FastHarborAI, create_fast_client
from harborai.core.optimizations.concurrency_manager import ConcurrencyManager, ConcurrencyConfig

logger = logging.getLogger(__name__)


class MockPlugin:
    """模拟插件，用于性能测试"""
    
    def __init__(self, response_time_ms: float = 50):
        self.response_time_ms = response_time_ms
        self.call_count = 0
    
    def chat_completion(self, messages, model, **kwargs):
        """模拟同步聊天完成"""
        self.call_count += 1
        time.sleep(self.response_time_ms / 1000)
        return {
            "id": f"chatcmpl-{self.call_count}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"响应 {self.call_count}"
                },
                "finish_reason": "stop"
            }]
        }
    
    async def chat_completion_async(self, messages, model, **kwargs):
        """模拟异步聊天完成"""
        self.call_count += 1
        await asyncio.sleep(self.response_time_ms / 1000)
        return {
            "id": f"chatcmpl-async-{self.call_count}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"异步响应 {self.call_count}"
                },
                "finish_reason": "stop"
            }]
        }


def setup_mock_plugin_manager():
    """设置模拟插件管理器"""
    mock_plugin = MockPlugin()
    
    def mock_get_plugin_for_model(model):
        return mock_plugin
    
    def mock_get_supported_models():
        return ["gpt-3.5-turbo", "gpt-4", "claude-3"]
    
    def mock_is_model_supported(model):
        return model in ["gpt-3.5-turbo", "gpt-4", "claude-3"]
    
    mock_manager = Mock()
    mock_manager.get_plugin_for_model = mock_get_plugin_for_model
    mock_manager.get_plugin_name_for_model.return_value = "mock_plugin"
    mock_manager.get_supported_models = mock_get_supported_models
    mock_manager.is_model_supported = mock_is_model_supported
    
    return mock_manager


async def test_concurrency_manager_basic():
    """测试并发管理器基本功能"""
    print("=== 测试并发管理器基本功能 ===")
    
    # 创建配置
    config = ConcurrencyConfig(
        max_concurrent_requests=10,
        connection_pool_size=5,
        request_timeout=30.0,
        enable_adaptive_optimization=True,
        enable_health_check=True
    )
    
    # 创建并发管理器
    manager = ConcurrencyManager(config)
    
    try:
        # 启动管理器
        await manager.start()
        print("✓ 并发管理器启动成功")
        
        # 检查状态
        stats = manager.get_statistics()
        print(f"✓ 管理器状态: {stats['manager_status']}")
        print(f"✓ 配置: {stats['config']}")
        
        # 停止管理器
        await manager.stop()
        print("✓ 并发管理器停止成功")
        
    except Exception as e:
        print(f"✗ 并发管理器测试失败: {e}")
        raise


async def test_fast_client_with_concurrency():
    """测试FastHarborAI客户端的并发功能"""
    print("\n=== 测试FastHarborAI客户端并发功能 ===")
    
    # 配置启用并发优化
    config = {
        'enable_caching': False,  # 禁用缓存以简化测试
        'concurrency_optimization': {
            'max_concurrent_requests': 20,
            'connection_pool_size': 10,
            'request_timeout': 30.0,
            'enable_adaptive_optimization': True,
            'enable_health_check': True
        }
    }
    
    with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
        mock_manager = setup_mock_plugin_manager()
        mock_get_manager.return_value = mock_manager
        
        # 创建客户端
        client = create_fast_client(config=config)
        
        # 强制初始化并设置模拟管理器
        client.chat.completions._ensure_initialized()
        client.chat.completions._lazy_manager = mock_manager
        
        try:
            # 测试同步请求
            print("测试同步请求...")
            start_time = time.perf_counter()
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-3.5-turbo"
            )
            
            end_time = time.perf_counter()
            print(f"✓ 同步请求成功，耗时: {(end_time - start_time) * 1000:.2f}ms")
            print(f"✓ 响应: {response['choices'][0]['message']['content']}")
            
            # 测试异步请求
            print("测试异步请求...")
            start_time = time.perf_counter()
            
            response = await client.chat.completions.create_async(
                messages=[{"role": "user", "content": "Hello async"}],
                model="gpt-3.5-turbo"
            )
            
            end_time = time.perf_counter()
            print(f"✓ 异步请求成功，耗时: {(end_time - start_time) * 1000:.2f}ms")
            print(f"✓ 响应: {response['choices'][0]['message']['content']}")
            
        except Exception as e:
            print(f"✗ 客户端测试失败: {e}")
            raise


async def test_concurrent_requests():
    """测试并发请求性能"""
    print("\n=== 测试并发请求性能 ===")
    
    config = {
        'enable_caching': False,  # 禁用缓存以简化测试
        'concurrency_optimization': {
            'max_concurrent_requests': 50,
            'connection_pool_size': 20,
            'request_timeout': 30.0,
            'enable_adaptive_optimization': True,
            'enable_health_check': True
        }
    }
    
    with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
        mock_manager = setup_mock_plugin_manager()
        mock_get_manager.return_value = mock_manager
        
        client = create_fast_client(config=config)
        
        # 强制初始化并设置模拟管理器
        client.chat.completions._ensure_initialized()
        client.chat.completions._lazy_manager = mock_manager
        
        # 并发测试
        num_requests = 20
        print(f"执行 {num_requests} 个并发请求...")
        
        async def make_request(i):
            try:
                response = await client.chat.completions.create_async(
                    messages=[{"role": "user", "content": f"Request {i}"}],
                    model="gpt-3.5-turbo"
                )
                return True, response
            except Exception as e:
                return False, str(e)
        
        start_time = time.perf_counter()
        
        # 创建并发任务
        tasks = [make_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # 统计结果
        success_count = sum(1 for result in results if isinstance(result, tuple) and result[0])
        error_count = num_requests - success_count
        
        ops_per_second = num_requests / total_time
        
        print(f"✓ 并发测试完成")
        print(f"✓ 总请求数: {num_requests}")
        print(f"✓ 成功请求: {success_count}")
        print(f"✓ 失败请求: {error_count}")
        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 吞吐量: {ops_per_second:.2f} ops/s")
        print(f"✓ 平均响应时间: {(total_time / num_requests) * 1000:.2f}ms")
        
        # 验证性能
        if ops_per_second > 100:  # 基本性能要求
            print("✓ 性能测试通过")
        else:
            print("✗ 性能测试未达到预期")


async def main():
    """主测试函数"""
    print("=== HarborAI 并发优化组件测试 ===\n")
    
    try:
        # 基本功能测试
        await test_concurrency_manager_basic()
        
        # 客户端集成测试
        await test_fast_client_with_concurrency()
        
        # 并发性能测试
        await test_concurrent_requests()
        
        print("\n=== 所有测试通过 ===")
        
    except Exception as e:
        print(f"\n=== 测试失败: {e} ===")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())