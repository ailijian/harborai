#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速并发性能测试
验证并发优化的基本功能和性能提升
"""

import asyncio
import time
from unittest.mock import Mock, patch
from harborai.api.fast_client import create_fast_client


class MockPlugin:
    """模拟插件"""
    
    def __init__(self):
        self.call_count = 0
    
    def chat_completion(self, messages, model, **kwargs):
        """同步聊天完成"""
        self.call_count += 1
        time.sleep(0.01)  # 模拟10ms处理时间
        return {
            "id": f"chatcmpl-{self.call_count}",
            "choices": [{"message": {"content": f"响应 {self.call_count}"}}]
        }
    
    async def chat_completion_async(self, messages, model, **kwargs):
        """异步聊天完成"""
        self.call_count += 1
        await asyncio.sleep(0.01)  # 模拟10ms处理时间
        return {
            "id": f"chatcmpl-async-{self.call_count}",
            "choices": [{"message": {"content": f"异步响应 {self.call_count}"}}]
        }


def setup_mock_plugin_manager():
    """设置模拟插件管理器"""
    mock_plugin = MockPlugin()
    
    mock_manager = Mock()
    mock_manager.get_plugin_for_model.return_value = mock_plugin
    mock_manager.get_supported_models.return_value = ['gpt-3.5-turbo', 'gpt-4']
    mock_manager.is_model_supported.return_value = True
    mock_manager.get_plugin_name_for_model.return_value = "mock_plugin"
    
    return mock_manager


async def test_concurrent_performance():
    """测试并发性能"""
    print("=== 并发性能测试 ===")
    
    with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
        mock_manager = setup_mock_plugin_manager()
        mock_get_manager.return_value = mock_manager
        
        # 创建客户端（启用并发优化）
        config = {
            'enable_caching': False,
            'concurrency_optimization': {
                'max_concurrent_requests': 50,
                'connection_pool_size': 25,
                'request_timeout': 30.0
            }
        }
        
        client = create_fast_client(config=config)
        client.chat.completions._ensure_initialized()
        client.chat.completions._lazy_manager = mock_manager
        
        # 测试不同并发级别
        test_cases = [
            (10, "低并发"),
            (50, "中并发"),
            (100, "高并发")
        ]
        
        for num_requests, test_name in test_cases:
            print(f"\n--- {test_name}测试 ({num_requests} 请求) ---")
            
            start_time = time.perf_counter()
            
            # 创建并发任务
            tasks = []
            for i in range(num_requests):
                task = client.chat.completions.create_async(
                    messages=[{"role": "user", "content": f"测试请求 {i}"}],
                    model="gpt-3.5-turbo"
                )
                tasks.append(task)
            
            # 执行并发请求
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # 统计结果
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            error_count = num_requests - success_count
            throughput = num_requests / total_time
            
            print(f"✓ 总时间: {total_time:.3f}s")
            print(f"✓ 成功请求: {success_count}/{num_requests}")
            print(f"✓ 失败请求: {error_count}")
            print(f"✓ 吞吐量: {throughput:.2f} ops/s")
            
            if error_count > 0:
                print("错误详情:")
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        print(f"  请求 {i}: {result}")


async def test_baseline_vs_optimized():
    """对比基线性能和优化性能"""
    print("\n=== 基线 vs 优化性能对比 ===")
    
    with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
        mock_manager = setup_mock_plugin_manager()
        mock_get_manager.return_value = mock_manager
        
        num_requests = 50
        
        # 1. 基线性能（无优化）
        print(f"\n--- 基线性能测试 ({num_requests} 请求) ---")
        config_baseline = {'enable_caching': False}
        
        client_baseline = create_fast_client(config=config_baseline)
        client_baseline.chat.completions._ensure_initialized()
        client_baseline.chat.completions._lazy_manager = mock_manager
        
        start_time = time.perf_counter()
        tasks_baseline = []
        for i in range(num_requests):
            task = client_baseline.chat.completions.create_async(
                messages=[{"role": "user", "content": f"基线测试 {i}"}],
                model="gpt-3.5-turbo"
            )
            tasks_baseline.append(task)
        
        results_baseline = await asyncio.gather(*tasks_baseline, return_exceptions=True)
        baseline_time = time.perf_counter() - start_time
        baseline_throughput = num_requests / baseline_time
        
        print(f"✓ 基线吞吐量: {baseline_throughput:.2f} ops/s")
        
        # 2. 优化性能
        print(f"\n--- 优化性能测试 ({num_requests} 请求) ---")
        config_optimized = {
            'enable_caching': False,
            'concurrency_optimization': {
                'max_concurrent_requests': 50,
                'connection_pool_size': 25,
                'request_timeout': 30.0
            }
        }
        
        client_optimized = create_fast_client(config=config_optimized)
        client_optimized.chat.completions._ensure_initialized()
        client_optimized.chat.completions._lazy_manager = mock_manager
        
        start_time = time.perf_counter()
        tasks_optimized = []
        for i in range(num_requests):
            task = client_optimized.chat.completions.create_async(
                messages=[{"role": "user", "content": f"优化测试 {i}"}],
                model="gpt-3.5-turbo"
            )
            tasks_optimized.append(task)
        
        results_optimized = await asyncio.gather(*tasks_optimized, return_exceptions=True)
        optimized_time = time.perf_counter() - start_time
        optimized_throughput = num_requests / optimized_time
        
        print(f"✓ 优化吞吐量: {optimized_throughput:.2f} ops/s")
        
        # 3. 性能对比
        improvement = (optimized_throughput / baseline_throughput - 1) * 100
        print(f"\n--- 性能提升分析 ---")
        print(f"✓ 基线性能: {baseline_throughput:.2f} ops/s")
        print(f"✓ 优化性能: {optimized_throughput:.2f} ops/s")
        print(f"✓ 性能提升: {improvement:+.1f}%")
        
        # 验证目标
        target_throughput = 1000.0
        if optimized_throughput >= target_throughput:
            print(f"✓ 目标达成: {optimized_throughput:.2f} ops/s ≥ {target_throughput} ops/s")
        else:
            print(f"⚠ 目标未达成: {optimized_throughput:.2f} ops/s < {target_throughput} ops/s")
            print("  注意: 这可能是由于模拟环境的限制")


async def main():
    """主测试函数"""
    print("=== HarborAI 快速并发性能测试 ===")
    
    try:
        await test_concurrent_performance()
        await test_baseline_vs_optimized()
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"\n=== 测试失败: {e} ===")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())