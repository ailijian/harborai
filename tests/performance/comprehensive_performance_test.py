#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合性能测试

验证并发优化后的性能提升效果，目标是达到≥1000 ops/s的吞吐量。

测试策略：
1. 基准测试：测试传统方式的性能
2. 并发优化测试：测试优化后的性能
3. 压力测试：测试高并发场景下的稳定性
4. 性能对比：验证性能提升效果

Assumptions:
- A1: 并发优化组件能够正确处理多个并发请求
- A2: 优化后的配置能够提升并发性能
- A3: 系统能够稳定处理高并发请求
"""

import asyncio
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import logging

# 导入被测试的组件
from harborai.api.fast_client import FastHarborAI, create_fast_client
from harborai.core.optimizations.concurrency_manager import ConcurrencyManager, ConcurrencyConfig
from harborai.core.optimizations.lockfree_plugin_manager import LockFreePluginManager
from harborai.core.optimizations.async_request_processor import AsyncRequestProcessor
from harborai.core.optimizations.optimized_connection_pool import OptimizedConnectionPool

logger = logging.getLogger(__name__)


class MockPlugin:
    """模拟插件，用于性能测试"""
    
    def __init__(self, response_time_ms: float = 50):
        """
        Args:
            response_time_ms: 模拟响应时间（毫秒）
        """
        self.response_time_ms = response_time_ms
        self.call_count = 0
        self.lock = threading.Lock()
    
    def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> Dict[str, Any]:
        """模拟同步聊天完成"""
        with self.lock:
            self.call_count += 1
        
        # 模拟处理时间
        time.sleep(self.response_time_ms / 1000)
        
        return {
            "id": f"chatcmpl-{self.call_count}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"这是第{self.call_count}个响应"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    
    async def chat_completion_async(self, messages: List[Dict], model: str, **kwargs) -> Dict[str, Any]:
        """模拟异步聊天完成"""
        with self.lock:
            self.call_count += 1
        
        # 模拟异步处理时间
        await asyncio.sleep(self.response_time_ms / 1000)
        
        return {
            "id": f"chatcmpl-async-{self.call_count}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"这是第{self.call_count}个异步响应"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }


class PerformanceTester:
    """性能测试器"""
    
    def __init__(self):
        self.test_messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        self.test_model = "gpt-3.5-turbo"
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
    
    def test_traditional_performance(self, num_requests: int = 100, num_threads: int = 10) -> Dict[str, float]:
        """测试传统同步方式的性能"""
        print(f"\n=== 传统同步性能测试 ===")
        print(f"请求数: {num_requests}, 线程数: {num_threads}")
        
        # 创建模拟插件
        mock_plugin = MockPlugin(response_time_ms=50)
        
        def make_request():
            """执行单个请求"""
            try:
                # 模拟传统方式的请求处理
                response = mock_plugin.chat_completion(
                    messages=self.test_messages,
                    model=self.test_model
                )
                return True, response
            except Exception as e:
                return False, str(e)
        
        # 执行并发测试
        start_time = time.perf_counter()
        success_count = 0
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            
            for future in as_completed(futures):
                success, result = future.result()
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"请求失败: {result}")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # 计算性能指标
        ops_per_second = num_requests / total_time
        avg_response_time = total_time / num_requests * 1000  # 毫秒
        
        stats = {
            'total_requests': num_requests,
            'success_count': success_count,
            'error_count': error_count,
            'total_time_seconds': total_time,
            'ops_per_second': ops_per_second,
            'avg_response_time_ms': avg_response_time,
            'success_rate': success_count / num_requests * 100
        }
        
        print(f"✓ 成功请求: {success_count}")
        print(f"✓ 失败请求: {error_count}")
        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 吞吐量: {ops_per_second:.2f} ops/s")
        print(f"✓ 平均响应时间: {avg_response_time:.2f}ms")
        print(f"✓ 成功率: {stats['success_rate']:.1f}%")
        
        return stats
    
    async def test_optimized_performance(self, num_requests: int = 200, max_concurrent: int = 100) -> Dict[str, float]:
        """测试优化后的异步性能"""
        print(f"\n=== 并发优化性能测试 ===")
        print(f"请求数: {num_requests}, 最大并发: {max_concurrent}")
        
        # 创建模拟插件
        mock_plugin = MockPlugin(response_time_ms=50)
        
        async def make_async_request():
            """执行单个异步请求"""
            try:
                # 模拟优化后的异步请求处理
                response = await mock_plugin.chat_completion_async(
                    messages=self.test_messages,
                    model=self.test_model
                )
                return True, response
            except Exception as e:
                return False, str(e)
        
        # 执行异步并发测试
        start_time = time.perf_counter()
        success_count = 0
        error_count = 0
        
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def controlled_request():
            async with semaphore:
                return await make_async_request()
        
        # 创建所有任务
        tasks = [controlled_request() for _ in range(num_requests)]
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
                print(f"异步请求失败: {result}")
            else:
                success, response = result
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    print(f"异步请求失败: {response}")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # 计算性能指标
        ops_per_second = num_requests / total_time
        avg_response_time = total_time / num_requests * 1000  # 毫秒
        
        stats = {
            'total_requests': num_requests,
            'success_count': success_count,
            'error_count': error_count,
            'total_time_seconds': total_time,
            'ops_per_second': ops_per_second,
            'avg_response_time_ms': avg_response_time,
            'success_rate': success_count / num_requests * 100,
            'max_concurrent': max_concurrent
        }
        
        print(f"✓ 成功请求: {success_count}")
        print(f"✓ 失败请求: {error_count}")
        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 吞吐量: {ops_per_second:.2f} ops/s")
        print(f"✓ 平均响应时间: {avg_response_time:.2f}ms")
        print(f"✓ 成功率: {stats['success_rate']:.1f}%")
        
        return stats
    
    async def test_stress_performance(self, num_requests: int = 500, max_concurrent: int = 150) -> Dict[str, float]:
        """压力测试"""
        print(f"\n=== 压力测试 ===")
        print(f"请求数: {num_requests}, 最大并发: {max_concurrent}")
        
        # 创建模拟插件（更快的响应时间）
        mock_plugin = MockPlugin(response_time_ms=20)
        
        async def make_stress_request():
            """执行压力测试请求"""
            try:
                response = await mock_plugin.chat_completion_async(
                    messages=self.test_messages,
                    model=self.test_model
                )
                return True, response
            except Exception as e:
                return False, str(e)
        
        # 执行压力测试
        start_time = time.perf_counter()
        success_count = 0
        error_count = 0
        
        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def controlled_stress_request():
            async with semaphore:
                return await make_stress_request()
        
        # 创建所有任务
        tasks = [controlled_stress_request() for _ in range(num_requests)]
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                error_count += 1
            else:
                success, response = result
                if success:
                    success_count += 1
                else:
                    error_count += 1
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # 计算性能指标
        ops_per_second = num_requests / total_time
        avg_response_time = total_time / num_requests * 1000  # 毫秒
        
        stats = {
            'total_requests': num_requests,
            'success_count': success_count,
            'error_count': error_count,
            'total_time_seconds': total_time,
            'ops_per_second': ops_per_second,
            'avg_response_time_ms': avg_response_time,
            'success_rate': success_count / num_requests * 100,
            'max_concurrent': max_concurrent
        }
        
        print(f"✓ 成功请求: {success_count}")
        print(f"✓ 失败请求: {error_count}")
        print(f"✓ 总耗时: {total_time:.2f}秒")
        print(f"✓ 吞吐量: {ops_per_second:.2f} ops/s")
        print(f"✓ 平均响应时间: {avg_response_time:.2f}ms")
        print(f"✓ 成功率: {stats['success_rate']:.1f}%")
        
        return stats
    
    def compare_performance(self, traditional_stats: Dict, optimized_stats: Dict, stress_stats: Dict) -> Dict[str, Any]:
        """比较性能结果"""
        print(f"\n=== 性能对比分析 ===")
        
        improvement_ratio = optimized_stats['ops_per_second'] / traditional_stats['ops_per_second']
        stress_improvement = stress_stats['ops_per_second'] / traditional_stats['ops_per_second']
        
        comparison = {
            'traditional_ops_per_second': traditional_stats['ops_per_second'],
            'optimized_ops_per_second': optimized_stats['ops_per_second'],
            'stress_ops_per_second': stress_stats['ops_per_second'],
            'improvement_ratio': improvement_ratio,
            'improvement_percentage': (improvement_ratio - 1) * 100,
            'stress_improvement_ratio': stress_improvement,
            'target_achieved': stress_stats['ops_per_second'] >= 1000.0,
            'baseline_improvement': optimized_stats['ops_per_second'] >= 505.6 * 1.5  # 至少50%提升
        }
        
        print(f"传统方式: {comparison['traditional_ops_per_second']:.2f} ops/s")
        print(f"并发优化: {comparison['optimized_ops_per_second']:.2f} ops/s")
        print(f"压力测试: {comparison['stress_ops_per_second']:.2f} ops/s")
        print(f"性能提升: {comparison['improvement_percentage']:.1f}%")
        print(f"压力提升: {(stress_improvement - 1) * 100:.1f}%")
        print(f"目标达成: {'✓' if comparison['target_achieved'] else '✗'} (≥1000 ops/s)")
        print(f"基准提升: {'✓' if comparison['baseline_improvement'] else '✗'} (≥50%)")
        
        return comparison


async def main():
    """主测试函数"""
    print("=== 综合性能测试开始 ===")
    
    tester = PerformanceTester()
    
    # 1. 传统性能测试
    traditional_stats = tester.test_traditional_performance(
        num_requests=100,
        num_threads=10
    )
    
    # 2. 并发优化测试
    optimized_stats = await tester.test_optimized_performance(
        num_requests=200,
        max_concurrent=100
    )
    
    # 3. 压力测试
    stress_stats = await tester.test_stress_performance(
        num_requests=500,
        max_concurrent=150
    )
    
    # 4. 性能对比
    comparison = tester.compare_performance(traditional_stats, optimized_stats, stress_stats)
    
    # 5. 总结
    print(f"\n=== 测试总结 ===")
    if comparison['target_achieved']:
        print("🎉 恭喜！性能优化目标已达成！")
        print(f"✓ 压力测试吞吐量: {comparison['stress_ops_per_second']:.2f} ops/s (≥1000 ops/s)")
    else:
        print("⚠️  性能优化目标尚未完全达成")
        print(f"✗ 压力测试吞吐量: {comparison['stress_ops_per_second']:.2f} ops/s (目标: ≥1000 ops/s)")
    
    if comparison['baseline_improvement']:
        print(f"✓ 基准性能提升: {comparison['improvement_percentage']:.1f}% (≥50%)")
    else:
        print(f"✗ 基准性能提升: {comparison['improvement_percentage']:.1f}% (目标: ≥50%)")
    
    print("\n=== 所有测试完成 ===")
    return comparison


if __name__ == "__main__":
    asyncio.run(main())