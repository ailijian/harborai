#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能对比测试

对比启用和禁用并发优化的性能差异：
- 基线性能（无优化）
- 并发优化性能
- 详细的性能指标对比
- 性能提升分析
"""

import asyncio
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from harborai.api.fast_client import create_fast_client


@dataclass
class PerformanceResult:
    """性能测试结果"""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    throughput: float
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    success_rate: float
    response_times: List[float]


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


class PerformanceComparator:
    """性能对比器"""
    
    def __init__(self):
        self.results: Dict[str, PerformanceResult] = {}
    
    def test_baseline_performance(self, num_requests: int = 100, concurrency: int = 10) -> PerformanceResult:
        """测试基线性能（无并发优化）"""
        print(f"\n=== 基线性能测试 ({num_requests} 请求, 并发度 {concurrency}) ===")
        
        config = {
            'enable_caching': False,
            'enable_performance_optimization': False,
            # 明确禁用所有优化
            'concurrency_optimization': None
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = setup_mock_plugin_manager(response_time_ms=10)
            mock_get_manager.return_value = mock_manager
            
            client = create_fast_client(config=config)
            client.chat.completions._ensure_initialized()
            client.chat.completions._lazy_manager = mock_manager
            
            response_times = []
            successful_requests = 0
            failed_requests = 0
            
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
                        print(f"基线测试请求失败: {e}")
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            result = self._create_performance_result(
                "基线性能", num_requests, successful_requests, failed_requests,
                total_time, response_times
            )
            
            self._print_result(result)
            self.results['baseline'] = result
            return result
    
    async def test_optimized_performance(self, num_requests: int = 100, concurrency: int = 50) -> PerformanceResult:
        """测试优化性能（启用并发优化）"""
        print(f"\n=== 优化性能测试 ({num_requests} 请求, 并发度 {concurrency}) ===")
        
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
            
            response_times = []
            successful_requests = 0
            failed_requests = 0
            
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
                    print(f"优化测试请求失败: {result}")
                else:
                    successful_requests += 1
            
            result = self._create_performance_result(
                "优化性能", num_requests, successful_requests, failed_requests,
                total_time, response_times
            )
            
            self._print_result(result)
            self.results['optimized'] = result
            return result
    
    async def test_scalability_comparison(self) -> Dict[str, List[PerformanceResult]]:
        """可扩展性对比测试"""
        print(f"\n=== 可扩展性对比测试 ===")
        
        test_cases = [
            (50, 10),   # 50请求, 10并发
            (100, 20),  # 100请求, 20并发
            (200, 50),  # 200请求, 50并发
        ]
        
        baseline_results = []
        optimized_results = []
        
        for num_requests, concurrency in test_cases:
            print(f"\n--- 测试场景: {num_requests} 请求, {concurrency} 并发 ---")
            
            # 基线测试
            baseline_result = self.test_baseline_performance(num_requests, concurrency)
            baseline_results.append(baseline_result)
            
            # 优化测试
            optimized_result = await self.test_optimized_performance(num_requests, concurrency)
            optimized_results.append(optimized_result)
            
            # 对比分析
            improvement = (optimized_result.throughput / baseline_result.throughput - 1) * 100
            print(f"性能提升: {improvement:+.1f}%")
        
        return {
            'baseline': baseline_results,
            'optimized': optimized_results
        }
    
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
    
    def _create_performance_result(self, test_name: str, total_requests: int,
                                 successful_requests: int, failed_requests: int,
                                 total_time: float, response_times: List[float]) -> PerformanceResult:
        """创建性能结果"""
        throughput = total_requests / total_time if total_time > 0 else 0
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.median(response_times)
            
            if len(response_times) >= 20:
                p95_response_time = statistics.quantiles(response_times, n=20)[18]
            else:
                p95_response_time = max(response_times) if response_times else 0
            
            if len(response_times) >= 100:
                p99_response_time = statistics.quantiles(response_times, n=100)[98]
            else:
                p99_response_time = max(response_times) if response_times else 0
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0
        
        return PerformanceResult(
            test_name=test_name,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_time=total_time,
            throughput=throughput,
            avg_response_time=avg_response_time,
            p50_response_time=p50_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            success_rate=success_rate,
            response_times=response_times
        )
    
    def _print_result(self, result: PerformanceResult):
        """打印测试结果"""
        print(f"✓ {result.test_name}测试完成:")
        print(f"  总请求数: {result.total_requests}")
        print(f"  成功请求: {result.successful_requests}")
        print(f"  失败请求: {result.failed_requests}")
        print(f"  成功率: {result.success_rate:.1f}%")
        print(f"  总时间: {result.total_time:.3f}s")
        print(f"  吞吐量: {result.throughput:.2f} ops/s")
        print(f"  平均响应时间: {result.avg_response_time*1000:.2f}ms")
        print(f"  P50响应时间: {result.p50_response_time*1000:.2f}ms")
        print(f"  P95响应时间: {result.p95_response_time*1000:.2f}ms")
    
    def analyze_comparison(self):
        """分析对比结果"""
        print("\n" + "="*60)
        print("性能对比分析")
        print("="*60)
        
        if 'baseline' not in self.results or 'optimized' not in self.results:
            print("缺少对比数据")
            return
        
        baseline = self.results['baseline']
        optimized = self.results['optimized']
        
        # 吞吐量对比
        throughput_improvement = (optimized.throughput / baseline.throughput - 1) * 100
        
        # 响应时间对比
        response_time_improvement = (baseline.avg_response_time / optimized.avg_response_time - 1) * 100
        
        # 成功率对比
        success_rate_diff = optimized.success_rate - baseline.success_rate
        
        print(f"基线性能:")
        print(f"  吞吐量: {baseline.throughput:.2f} ops/s")
        print(f"  平均响应时间: {baseline.avg_response_time*1000:.2f}ms")
        print(f"  P95响应时间: {baseline.p95_response_time*1000:.2f}ms")
        print(f"  成功率: {baseline.success_rate:.1f}%")
        
        print(f"\n优化性能:")
        print(f"  吞吐量: {optimized.throughput:.2f} ops/s")
        print(f"  平均响应时间: {optimized.avg_response_time*1000:.2f}ms")
        print(f"  P95响应时间: {optimized.p95_response_time*1000:.2f}ms")
        print(f"  成功率: {optimized.success_rate:.1f}%")
        
        print(f"\n性能提升:")
        print(f"  吞吐量提升: {throughput_improvement:+.1f}%")
        print(f"  响应时间改善: {response_time_improvement:+.1f}%")
        print(f"  成功率变化: {success_rate_diff:+.1f}%")
        
        # 评估结果
        print(f"\n评估结果:")
        
        if throughput_improvement > 50:
            print(f"✓ 吞吐量显著提升: {throughput_improvement:.1f}%")
        elif throughput_improvement > 20:
            print(f"✓ 吞吐量明显提升: {throughput_improvement:.1f}%")
        elif throughput_improvement > 0:
            print(f"✓ 吞吐量有所提升: {throughput_improvement:.1f}%")
        else:
            print(f"⚠ 吞吐量未提升: {throughput_improvement:.1f}%")
        
        if response_time_improvement > 20:
            print(f"✓ 响应时间显著改善: {response_time_improvement:.1f}%")
        elif response_time_improvement > 10:
            print(f"✓ 响应时间明显改善: {response_time_improvement:.1f}%")
        elif response_time_improvement > 0:
            print(f"✓ 响应时间有所改善: {response_time_improvement:.1f}%")
        else:
            print(f"⚠ 响应时间未改善: {response_time_improvement:.1f}%")
        
        if optimized.success_rate >= 95:
            print(f"✓ 稳定性良好: 成功率 {optimized.success_rate:.1f}%")
        else:
            print(f"⚠ 稳定性需改善: 成功率 {optimized.success_rate:.1f}%")
        
        # 目标验证
        target_throughput = 1000.0
        if optimized.throughput >= target_throughput:
            print(f"\n✓ 性能目标达成: {optimized.throughput:.2f} ops/s ≥ {target_throughput} ops/s")
        else:
            print(f"\n⚠ 性能目标未完全达成: {optimized.throughput:.2f} ops/s < {target_throughput} ops/s")
            print(f"  但相比基线已有 {throughput_improvement:.1f}% 的提升")


async def main():
    """主测试函数"""
    print("=== HarborAI 性能对比测试 ===")
    print("对比启用和禁用并发优化的性能差异")
    
    comparator = PerformanceComparator()
    
    try:
        # 1. 基本性能对比
        print("\n--- 基本性能对比 ---")
        comparator.test_baseline_performance(100, 10)
        await comparator.test_optimized_performance(100, 50)
        comparator.analyze_comparison()
        
        # 2. 可扩展性对比
        print("\n--- 可扩展性对比 ---")
        scalability_results = await comparator.test_scalability_comparison()
        
        # 3. 可扩展性分析
        print(f"\n--- 可扩展性分析 ---")
        baseline_results = scalability_results['baseline']
        optimized_results = scalability_results['optimized']
        
        for i, (baseline, optimized) in enumerate(zip(baseline_results, optimized_results)):
            improvement = (optimized.throughput / baseline.throughput - 1) * 100
            print(f"场景 {i+1}: 基线 {baseline.throughput:.1f} ops/s → 优化 {optimized.throughput:.1f} ops/s (提升 {improvement:+.1f}%)")
        
        print("\n=== 性能对比测试完成 ===")
        
    except Exception as e:
        print(f"\n=== 测试失败: {e} ===")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())