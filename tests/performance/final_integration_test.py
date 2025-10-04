#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终集成测试

验证所有并发优化组件的协同工作：
- 基线性能测试
- 并发优化性能测试
- 性能提升验证
- 目标达成确认
"""

import asyncio
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from dataclasses import dataclass
from harborai.api.fast_client import create_fast_client


@dataclass
class TestResult:
    """测试结果"""
    test_name: str
    requests: int
    concurrency: int
    total_time: float
    throughput: float
    avg_response_time: float
    success_rate: float


class MockPlugin:
    """模拟插件"""
    
    def __init__(self, response_time_ms: float = 10):
        self.response_time_ms = response_time_ms
        self.call_count = 0
        self.lock = threading.Lock()
    
    def chat_completion(self, messages, model, **kwargs):
        """同步聊天完成"""
        with self.lock:
            self.call_count += 1
            call_id = self.call_count
        
        time.sleep(self.response_time_ms / 1000)
        return {
            "id": f"chatcmpl-{call_id}",
            "choices": [{"message": {"content": f"响应 {call_id}"}}]
        }
    
    async def chat_completion_async(self, messages, model, **kwargs):
        """异步聊天完成"""
        with self.lock:
            self.call_count += 1
            call_id = self.call_count
        
        await asyncio.sleep(self.response_time_ms / 1000)
        return {
            "id": f"chatcmpl-async-{call_id}",
            "choices": [{"message": {"content": f"异步响应 {call_id}"}}]
        }


def setup_mock_plugin_manager(response_time_ms: float = 10):
    """设置模拟插件管理器"""
    mock_plugin = MockPlugin(response_time_ms)
    
    mock_manager = Mock()
    mock_manager.get_plugin_for_model.return_value = mock_plugin
    mock_manager.get_supported_models.return_value = ["gpt-3.5-turbo", "gpt-4"]
    mock_manager.is_model_supported.return_value = True
    mock_manager.get_plugin_name_for_model.return_value = "mock_plugin"
    
    return mock_manager


class FinalIntegrationTester:
    """最终集成测试器"""
    
    def __init__(self):
        self.results: List[TestResult] = []
    
    def test_baseline_performance(self, num_requests: int = 100, concurrency: int = 10) -> TestResult:
        """测试基线性能（无并发优化）"""
        print(f"\n=== 基线性能测试 ===")
        print(f"请求数: {num_requests}, 并发度: {concurrency}")
        
        config = {
            'enable_caching': False,
            'enable_performance_optimization': False,
            'concurrency_optimization': None
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = setup_mock_plugin_manager(response_time_ms=10)
            mock_get_manager.return_value = mock_manager
            
            client = create_fast_client(config=config)
            client.chat.completions._ensure_initialized()
            client.chat.completions._lazy_manager = mock_manager
            
            successful_requests = 0
            failed_requests = 0
            response_times = []
            
            start_time = time.perf_counter()
            
            # 使用线程池模拟并发
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                
                for i in range(num_requests):
                    future = executor.submit(self._make_sync_request, client, i, response_times)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        future.result()
                        successful_requests += 1
                    except Exception as e:
                        failed_requests += 1
                        print(f"请求失败: {e}")
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            result = TestResult(
                test_name="基线性能",
                requests=num_requests,
                concurrency=concurrency,
                total_time=total_time,
                throughput=num_requests / total_time,
                avg_response_time=statistics.mean(response_times) if response_times else 0,
                success_rate=(successful_requests / num_requests * 100) if num_requests > 0 else 0
            )
            
            self._print_result(result)
            self.results.append(result)
            return result
    
    async def test_optimized_performance(self, num_requests: int = 100, concurrency: int = 50) -> TestResult:
        """测试优化性能（启用并发优化）"""
        print(f"\n=== 并发优化性能测试 ===")
        print(f"请求数: {num_requests}, 并发度: {concurrency}")
        
        config = {
            'enable_caching': False,
            'concurrency_optimization': {
                'max_concurrent_requests': concurrency,
                'connection_pool_size': concurrency // 2,
                'request_timeout': 30.0,
                'enable_adaptive_optimization': True,
                'enable_health_check': True
            }
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = setup_mock_plugin_manager(response_time_ms=10)
            mock_get_manager.return_value = mock_manager
            
            client = create_fast_client(config=config)
            client.chat.completions._ensure_initialized()
            client.chat.completions._lazy_manager = mock_manager
            
            successful_requests = 0
            failed_requests = 0
            response_times = []
            
            start_time = time.perf_counter()
            
            # 创建异步任务
            tasks = []
            for i in range(num_requests):
                task = self._make_async_request(client, i, response_times)
                tasks.append(task)
            
            # 并发执行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # 统计结果
            for result in results:
                if isinstance(result, Exception):
                    failed_requests += 1
                    print(f"请求失败: {result}")
                else:
                    successful_requests += 1
            
            result = TestResult(
                test_name="并发优化性能",
                requests=num_requests,
                concurrency=concurrency,
                total_time=total_time,
                throughput=num_requests / total_time,
                avg_response_time=statistics.mean(response_times) if response_times else 0,
                success_rate=(successful_requests / num_requests * 100) if num_requests > 0 else 0
            )
            
            self._print_result(result)
            self.results.append(result)
            return result
    
    def _make_sync_request(self, client, request_id: int, response_times: List[float]):
        """执行同步请求"""
        start_time = time.perf_counter()
        
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": f"基线测试 {request_id}"}],
            model="gpt-3.5-turbo"
        )
        
        end_time = time.perf_counter()
        response_time = end_time - start_time
        response_times.append(response_time)
        
        return response
    
    async def _make_async_request(self, client, request_id: int, response_times: List[float]):
        """执行异步请求"""
        start_time = time.perf_counter()
        
        response = await client.chat.completions.create_async(
            messages=[{"role": "user", "content": f"优化测试 {request_id}"}],
            model="gpt-3.5-turbo"
        )
        
        end_time = time.perf_counter()
        response_time = end_time - start_time
        response_times.append(response_time)
        
        return response
    
    def _print_result(self, result: TestResult):
        """打印测试结果"""
        print(f"✓ {result.test_name}测试完成:")
        print(f"  总请求数: {result.requests}")
        print(f"  并发度: {result.concurrency}")
        print(f"  成功率: {result.success_rate:.1f}%")
        print(f"  总时间: {result.total_time:.3f}s")
        print(f"  吞吐量: {result.throughput:.2f} ops/s")
        print(f"  平均响应时间: {result.avg_response_time*1000:.2f}ms")
    
    def analyze_final_results(self):
        """分析最终结果"""
        print("\n" + "="*80)
        print("最终集成测试结果分析")
        print("="*80)
        
        if len(self.results) < 2:
            print("缺少对比数据")
            return
        
        baseline = self.results[0]
        optimized = self.results[1]
        
        # 性能提升计算
        throughput_improvement = (optimized.throughput / baseline.throughput - 1) * 100
        response_time_improvement = (baseline.avg_response_time / optimized.avg_response_time - 1) * 100
        
        print(f"基线性能:")
        print(f"  吞吐量: {baseline.throughput:.2f} ops/s")
        print(f"  平均响应时间: {baseline.avg_response_time*1000:.2f}ms")
        print(f"  成功率: {baseline.success_rate:.1f}%")
        
        print(f"\n并发优化性能:")
        print(f"  吞吐量: {optimized.throughput:.2f} ops/s")
        print(f"  平均响应时间: {optimized.avg_response_time*1000:.2f}ms")
        print(f"  成功率: {optimized.success_rate:.1f}%")
        
        print(f"\n性能提升:")
        print(f"  吞吐量提升: {throughput_improvement:+.1f}%")
        print(f"  响应时间改善: {response_time_improvement:+.1f}%")
        
        # 目标验证
        target_throughput = 1000.0
        original_throughput = 505.6  # 原始基线
        
        print(f"\n目标验证:")
        print(f"  原始基线: {original_throughput:.1f} ops/s")
        print(f"  目标吞吐量: {target_throughput:.1f} ops/s")
        print(f"  当前优化吞吐量: {optimized.throughput:.2f} ops/s")
        
        if optimized.throughput >= target_throughput:
            print(f"  ✓ 性能目标达成: {optimized.throughput:.2f} ops/s ≥ {target_throughput} ops/s")
            improvement_vs_original = (optimized.throughput / original_throughput - 1) * 100
            print(f"  ✓ 相比原始基线提升: {improvement_vs_original:+.1f}%")
        else:
            print(f"  ⚠ 性能目标未完全达成: {optimized.throughput:.2f} ops/s < {target_throughput} ops/s")
            improvement_vs_original = (optimized.throughput / original_throughput - 1) * 100
            print(f"  但相比原始基线已提升: {improvement_vs_original:+.1f}%")
        
        # 稳定性评估
        print(f"\n稳定性评估:")
        if optimized.success_rate >= 99:
            print(f"  ✓ 稳定性优秀: 成功率 {optimized.success_rate:.1f}%")
        elif optimized.success_rate >= 95:
            print(f"  ✓ 稳定性良好: 成功率 {optimized.success_rate:.1f}%")
        else:
            print(f"  ⚠ 稳定性需改善: 成功率 {optimized.success_rate:.1f}%")
        
        # 总体评估
        print(f"\n总体评估:")
        if optimized.throughput >= target_throughput and optimized.success_rate >= 95:
            print("  ✓ 并发性能优化成功！所有目标均已达成")
        elif optimized.throughput >= target_throughput:
            print("  ✓ 性能目标达成，但稳定性需要进一步优化")
        elif throughput_improvement >= 50:
            print("  ✓ 性能显著提升，接近目标")
        else:
            print("  ⚠ 性能提升有限，需要进一步优化")


async def main():
    """主测试函数"""
    print("=== HarborAI 最终集成测试 ===")
    print("验证所有并发优化组件的协同工作")
    
    tester = FinalIntegrationTester()
    
    try:
        # 1. 基线性能测试
        tester.test_baseline_performance(100, 10)
        
        # 2. 并发优化性能测试
        await tester.test_optimized_performance(100, 50)
        
        # 3. 最终结果分析
        tester.analyze_final_results()
        
        print("\n=== 最终集成测试完成 ===")
        
    except Exception as e:
        print(f"\n=== 测试失败: {e} ===")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())