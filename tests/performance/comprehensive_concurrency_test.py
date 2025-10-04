#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全面的并发性能测试

验证HarborAI并发优化组件的性能提升：
- 目标：从505.6ops/s提升到≥1000ops/s
- 测试场景：不同并发级别、不同负载模式
- 稳定性验证：高并发下的系统稳定性
"""

import asyncio
import time
import statistics
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Tuple
import threading

# 导入被测试的组件
from harborai.api.fast_client import FastHarborAI, create_fast_client
from harborai.core.optimizations.concurrency_manager import ConcurrencyManager, ConcurrencyConfig

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """性能指标收集器"""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.success_count = 0
        self.error_count = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.lock = threading.Lock()
    
    def record_request(self, response_time: float, success: bool = True):
        """记录请求结果"""
        with self.lock:
            self.response_times.append(response_time)
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
    
    def start_timing(self):
        """开始计时"""
        self.start_time = time.perf_counter()
    
    def stop_timing(self):
        """停止计时"""
        self.end_time = time.perf_counter()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_requests = self.success_count + self.error_count
        total_time = self.end_time - self.start_time
        
        if not self.response_times:
            return {
                'total_requests': total_requests,
                'success_count': self.success_count,
                'error_count': self.error_count,
                'success_rate': 0.0,
                'total_time': total_time,
                'throughput': 0.0,
                'avg_response_time': 0.0,
                'p50_response_time': 0.0,
                'p95_response_time': 0.0,
                'p99_response_time': 0.0
            }
        
        return {
            'total_requests': total_requests,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': (self.success_count / total_requests * 100) if total_requests > 0 else 0,
            'total_time': total_time,
            'throughput': total_requests / total_time if total_time > 0 else 0,
            'avg_response_time': statistics.mean(self.response_times),
            'p50_response_time': statistics.median(self.response_times),
            'p95_response_time': statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else max(self.response_times),
            'p99_response_time': statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else max(self.response_times)
        }


class MockPlugin:
    """高性能模拟插件"""
    
    def __init__(self, response_time_ms: float = 10):
        self.response_time_ms = response_time_ms
        self.call_count = 0
        self.lock = threading.Lock()
    
    def chat_completion(self, messages, model, **kwargs):
        """模拟同步聊天完成"""
        with self.lock:
            self.call_count += 1
            call_id = self.call_count
        
        # 模拟处理时间
        time.sleep(self.response_time_ms / 1000)
        
        return {
            "id": f"chatcmpl-{call_id}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"响应 {call_id}"
                },
                "finish_reason": "stop"
            }]
        }
    
    async def chat_completion_async(self, messages, model, **kwargs):
        """模拟异步聊天完成"""
        with self.lock:
            self.call_count += 1
            call_id = self.call_count
        
        # 模拟异步处理时间
        await asyncio.sleep(self.response_time_ms / 1000)
        
        return {
            "id": f"chatcmpl-async-{call_id}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"异步响应 {call_id}"
                },
                "finish_reason": "stop"
            }]
        }


def setup_mock_plugin_manager(response_time_ms: float = 10):
    """设置模拟插件管理器"""
    mock_plugin = MockPlugin(response_time_ms)
    
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


class ConcurrencyPerformanceTester:
    """并发性能测试器"""
    
    def __init__(self):
        self.results = {}
    
    def test_baseline_performance(self, num_requests: int = 100) -> Dict[str, Any]:
        """测试基线性能（无并发优化）"""
        print(f"\n=== 基线性能测试 ({num_requests} 请求) ===")
        
        config = {
            'enable_caching': False,
            'enable_performance_optimization': False,
            # 不启用并发优化
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = setup_mock_plugin_manager(response_time_ms=5)  # 5ms模拟响应时间
            mock_get_manager.return_value = mock_manager
            
            client = create_fast_client(config=config)
            client.chat.completions._ensure_initialized()
            client.chat.completions._lazy_manager = mock_manager
            
            metrics = PerformanceMetrics()
            metrics.start_timing()
            
            # 使用线程池进行并发测试
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                
                for i in range(num_requests):
                    future = executor.submit(self._make_sync_request, client, i, metrics)
                    futures.append(future)
                
                # 等待所有请求完成
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        metrics.record_request(0.0, success=False)
                        logger.error(f"请求失败: {e}")
            
            metrics.stop_timing()
            stats = metrics.get_statistics()
            
            print(f"✓ 基线性能测试完成")
            print(f"✓ 总请求数: {stats['total_requests']}")
            print(f"✓ 成功率: {stats['success_rate']:.1f}%")
            print(f"✓ 吞吐量: {stats['throughput']:.2f} ops/s")
            print(f"✓ 平均响应时间: {stats['avg_response_time']*1000:.2f}ms")
            print(f"✓ P95响应时间: {stats['p95_response_time']*1000:.2f}ms")
            
            self.results['baseline'] = stats
            return stats
    
    async def test_concurrency_performance(self, num_requests: int = 100, concurrency_level: int = 50) -> Dict[str, Any]:
        """测试并发优化性能"""
        print(f"\n=== 并发优化性能测试 ({num_requests} 请求, 并发度 {concurrency_level}) ===")
        
        config = {
            'enable_caching': False,
            'concurrency_optimization': {
                'max_concurrent_requests': concurrency_level,
                'connection_pool_size': concurrency_level // 2,
                'request_timeout': 30.0,
                'enable_adaptive_optimization': True,
                'enable_health_check': True
            }
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = setup_mock_plugin_manager(response_time_ms=5)  # 5ms模拟响应时间
            mock_get_manager.return_value = mock_manager
            
            client = create_fast_client(config=config)
            client.chat.completions._ensure_initialized()
            client.chat.completions._lazy_manager = mock_manager
            
            metrics = PerformanceMetrics()
            metrics.start_timing()
            
            # 创建异步任务
            tasks = []
            for i in range(num_requests):
                task = self._make_async_request(client, i, metrics)
                tasks.append(task)
            
            # 并发执行所有任务
            await asyncio.gather(*tasks, return_exceptions=True)
            
            metrics.stop_timing()
            stats = metrics.get_statistics()
            
            print(f"✓ 并发优化测试完成")
            print(f"✓ 总请求数: {stats['total_requests']}")
            print(f"✓ 成功率: {stats['success_rate']:.1f}%")
            print(f"✓ 吞吐量: {stats['throughput']:.2f} ops/s")
            print(f"✓ 平均响应时间: {stats['avg_response_time']*1000:.2f}ms")
            print(f"✓ P95响应时间: {stats['p95_response_time']*1000:.2f}ms")
            
            self.results['concurrency'] = stats
            return stats
    
    async def test_high_concurrency_stress(self, num_requests: int = 500, concurrency_level: int = 100) -> Dict[str, Any]:
        """高并发压力测试"""
        print(f"\n=== 高并发压力测试 ({num_requests} 请求, 并发度 {concurrency_level}) ===")
        
        config = {
            'enable_caching': False,
            'concurrency_optimization': {
                'max_concurrent_requests': concurrency_level,
                'connection_pool_size': concurrency_level // 2,
                'request_timeout': 30.0,
                'enable_adaptive_optimization': True,
                'enable_health_check': True
            }
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = setup_mock_plugin_manager(response_time_ms=10)  # 10ms模拟响应时间
            mock_get_manager.return_value = mock_manager
            
            client = create_fast_client(config=config)
            client.chat.completions._ensure_initialized()
            client.chat.completions._lazy_manager = mock_manager
            
            metrics = PerformanceMetrics()
            metrics.start_timing()
            
            # 分批处理以避免过度并发
            batch_size = concurrency_level
            for batch_start in range(0, num_requests, batch_size):
                batch_end = min(batch_start + batch_size, num_requests)
                batch_tasks = []
                
                for i in range(batch_start, batch_end):
                    task = self._make_async_request(client, i, metrics)
                    batch_tasks.append(task)
                
                # 执行当前批次
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # 短暂休息以避免系统过载
                await asyncio.sleep(0.01)
            
            metrics.stop_timing()
            stats = metrics.get_statistics()
            
            print(f"✓ 高并发压力测试完成")
            print(f"✓ 总请求数: {stats['total_requests']}")
            print(f"✓ 成功率: {stats['success_rate']:.1f}%")
            print(f"✓ 吞吐量: {stats['throughput']:.2f} ops/s")
            print(f"✓ 平均响应时间: {stats['avg_response_time']*1000:.2f}ms")
            print(f"✓ P95响应时间: {stats['p95_response_time']*1000:.2f}ms")
            
            self.results['stress'] = stats
            return stats
    
    def _make_sync_request(self, client, request_id: int, metrics: PerformanceMetrics):
        """执行同步请求"""
        try:
            start_time = time.perf_counter()
            
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": f"Request {request_id}"}],
                model="gpt-3.5-turbo"
            )
            
            end_time = time.perf_counter()
            response_time = end_time - start_time
            
            metrics.record_request(response_time, success=True)
            return response
            
        except Exception as e:
            metrics.record_request(0.0, success=False)
            raise
    
    async def _make_async_request(self, client, request_id: int, metrics: PerformanceMetrics):
        """执行异步请求"""
        try:
            start_time = time.perf_counter()
            
            response = await client.chat.completions.create_async(
                messages=[{"role": "user", "content": f"Request {request_id}"}],
                model="gpt-3.5-turbo"
            )
            
            end_time = time.perf_counter()
            response_time = end_time - start_time
            
            metrics.record_request(response_time, success=True)
            return response
            
        except Exception as e:
            metrics.record_request(0.0, success=False)
            logger.error(f"异步请求失败 {request_id}: {e}")
    
    def analyze_results(self):
        """分析测试结果"""
        print("\n" + "="*60)
        print("性能测试结果分析")
        print("="*60)
        
        if 'baseline' in self.results and 'concurrency' in self.results:
            baseline = self.results['baseline']
            concurrency = self.results['concurrency']
            
            throughput_improvement = (concurrency['throughput'] / baseline['throughput'] - 1) * 100
            response_time_improvement = (baseline['avg_response_time'] / concurrency['avg_response_time'] - 1) * 100
            
            print(f"基线性能:")
            print(f"  - 吞吐量: {baseline['throughput']:.2f} ops/s")
            print(f"  - 平均响应时间: {baseline['avg_response_time']*1000:.2f}ms")
            print(f"  - 成功率: {baseline['success_rate']:.1f}%")
            
            print(f"\n并发优化性能:")
            print(f"  - 吞吐量: {concurrency['throughput']:.2f} ops/s")
            print(f"  - 平均响应时间: {concurrency['avg_response_time']*1000:.2f}ms")
            print(f"  - 成功率: {concurrency['success_rate']:.1f}%")
            
            print(f"\n性能提升:")
            print(f"  - 吞吐量提升: {throughput_improvement:+.1f}%")
            print(f"  - 响应时间改善: {response_time_improvement:+.1f}%")
            
            # 验证目标
            target_throughput = 1000.0
            if concurrency['throughput'] >= target_throughput:
                print(f"\n✓ 性能目标达成: {concurrency['throughput']:.2f} ops/s ≥ {target_throughput} ops/s")
            else:
                print(f"\n✗ 性能目标未达成: {concurrency['throughput']:.2f} ops/s < {target_throughput} ops/s")
        
        if 'stress' in self.results:
            stress = self.results['stress']
            print(f"\n高并发压力测试:")
            print(f"  - 吞吐量: {stress['throughput']:.2f} ops/s")
            print(f"  - 平均响应时间: {stress['avg_response_time']*1000:.2f}ms")
            print(f"  - 成功率: {stress['success_rate']:.1f}%")
            
            if stress['success_rate'] >= 95.0:
                print(f"✓ 稳定性测试通过: 成功率 {stress['success_rate']:.1f}% ≥ 95%")
            else:
                print(f"✗ 稳定性测试失败: 成功率 {stress['success_rate']:.1f}% < 95%")


async def main():
    """主测试函数"""
    print("=== HarborAI 全面并发性能测试 ===")
    print("目标: 验证吞吐量从505.6ops/s提升到≥1000ops/s")
    
    tester = ConcurrencyPerformanceTester()
    
    try:
        # 1. 基线性能测试
        await asyncio.get_event_loop().run_in_executor(
            None, tester.test_baseline_performance, 200
        )
        
        # 2. 并发优化性能测试
        await tester.test_concurrency_performance(200, 50)
        
        # 3. 高并发压力测试
        await tester.test_high_concurrency_stress(500, 100)
        
        # 4. 结果分析
        tester.analyze_results()
        
        print("\n=== 全面性能测试完成 ===")
        
    except Exception as e:
        print(f"\n=== 测试失败: {e} ===")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # 减少日志输出
    asyncio.run(main())