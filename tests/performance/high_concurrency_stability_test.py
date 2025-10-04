#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高并发稳定性测试

验证HarborAI在高并发场景下的稳定性：
- 长时间运行稳定性
- 内存泄漏检测
- 错误率监控
- 资源使用监控
"""

import asyncio
import time
import psutil
import gc
import threading
from unittest.mock import Mock, patch
from typing import List, Dict, Any
from dataclasses import dataclass
from harborai.api.fast_client import create_fast_client


@dataclass
class StabilityMetrics:
    """稳定性指标"""
    start_time: float
    end_time: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    memory_usage_mb: List[float]
    cpu_usage_percent: List[float]
    response_times: List[float]
    error_messages: List[str]


class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.monitoring = False
        self.memory_usage = []
        self.cpu_usage = []
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
    
    def _monitor_loop(self):
        """监控循环"""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # 获取内存使用情况
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                # 获取CPU使用情况
                cpu_percent = process.cpu_percent()
                
                with self.lock:
                    self.memory_usage.append(memory_mb)
                    self.cpu_usage.append(cpu_percent)
                
                time.sleep(0.5)  # 每0.5秒采样一次
                
            except Exception as e:
                print(f"监控错误: {e}")
                break
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        with self.lock:
            if not self.memory_usage or not self.cpu_usage:
                return {
                    'memory_usage_mb': [],
                    'cpu_usage_percent': [],
                    'avg_memory_mb': 0.0,
                    'max_memory_mb': 0.0,
                    'avg_cpu_percent': 0.0,
                    'max_cpu_percent': 0.0
                }
            
            return {
                'memory_usage_mb': self.memory_usage.copy(),
                'cpu_usage_percent': self.cpu_usage.copy(),
                'avg_memory_mb': sum(self.memory_usage) / len(self.memory_usage),
                'max_memory_mb': max(self.memory_usage),
                'avg_cpu_percent': sum(self.cpu_usage) / len(self.cpu_usage),
                'max_cpu_percent': max(self.cpu_usage)
            }


class MockPlugin:
    """高性能模拟插件"""
    
    def __init__(self, response_time_ms: float = 10, error_rate: float = 0.0):
        self.response_time_ms = response_time_ms
        self.error_rate = error_rate
        self.call_count = 0
        self.lock = threading.Lock()
    
    async def chat_completion_async(self, messages, model, **kwargs):
        """异步聊天完成"""
        with self.lock:
            self.call_count += 1
            call_id = self.call_count
        
        # 模拟错误
        if self.error_rate > 0 and (call_id % int(1 / self.error_rate)) == 0:
            raise Exception(f"模拟错误 {call_id}")
        
        # 模拟处理时间
        await asyncio.sleep(self.response_time_ms / 1000)
        
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


def setup_mock_plugin_manager(response_time_ms: float = 10, error_rate: float = 0.0):
    """设置模拟插件管理器"""
    mock_plugin = MockPlugin(response_time_ms, error_rate)
    
    mock_manager = Mock()
    mock_manager.get_plugin_for_model.return_value = mock_plugin
    mock_manager.get_supported_models.return_value = ["gpt-3.5-turbo", "gpt-4"]
    mock_manager.is_model_supported.return_value = True
    mock_manager.get_plugin_name_for_model.return_value = "mock_plugin"
    
    return mock_manager


class HighConcurrencyStabilityTester:
    """高并发稳定性测试器"""
    
    def __init__(self):
        self.results = {}
    
    async def test_long_running_stability(self, duration_seconds: int = 60, concurrency: int = 50) -> StabilityMetrics:
        """长时间运行稳定性测试"""
        print(f"\n=== 长时间运行稳定性测试 ({duration_seconds}秒, 并发度{concurrency}) ===")
        
        # 启动资源监控
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        
        config = {
            'enable_caching': False,
            'concurrency_optimization': {
                'max_concurrent_requests': concurrency,
                'connection_pool_size': concurrency // 2,
                'request_timeout': 30.0,
                'enable_adaptive_optimization': True
            }
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = setup_mock_plugin_manager(response_time_ms=20, error_rate=0.01)  # 1%错误率
            mock_get_manager.return_value = mock_manager
            
            client = create_fast_client(config=config)
            client.chat.completions._ensure_initialized()
            client.chat.completions._lazy_manager = mock_manager
            
            # 初始化指标
            start_time = time.perf_counter()
            total_requests = 0
            successful_requests = 0
            failed_requests = 0
            response_times = []
            error_messages = []
            
            print(f"开始长时间稳定性测试...")
            
            # 持续发送请求
            end_time = start_time + duration_seconds
            
            while time.perf_counter() < end_time:
                # 创建一批并发请求
                batch_size = min(concurrency, 20)  # 限制批次大小
                tasks = []
                
                for i in range(batch_size):
                    task = self._make_request_with_metrics(
                        client, total_requests + i, response_times, error_messages
                    )
                    tasks.append(task)
                
                # 执行批次
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 统计结果
                for result in results:
                    total_requests += 1
                    if isinstance(result, Exception):
                        failed_requests += 1
                    else:
                        successful_requests += 1
                
                # 短暂休息
                await asyncio.sleep(0.1)
                
                # 定期输出进度
                if total_requests % 100 == 0:
                    elapsed = time.perf_counter() - start_time
                    current_throughput = total_requests / elapsed
                    print(f"进度: {total_requests} 请求, {elapsed:.1f}s, {current_throughput:.1f} ops/s")
            
            # 停止监控
            monitor.stop_monitoring()
            actual_end_time = time.perf_counter()
            
            # 收集资源指标
            resource_metrics = monitor.get_metrics()
            
            # 创建稳定性指标
            metrics = StabilityMetrics(
                start_time=start_time,
                end_time=actual_end_time,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                memory_usage_mb=resource_metrics['memory_usage_mb'],
                cpu_usage_percent=resource_metrics['cpu_usage_percent'],
                response_times=response_times,
                error_messages=error_messages
            )
            
            # 输出结果
            self._print_stability_results(metrics, resource_metrics)
            
            return metrics
    
    async def test_memory_leak_detection(self, iterations: int = 10, requests_per_iteration: int = 100) -> Dict[str, Any]:
        """内存泄漏检测测试"""
        print(f"\n=== 内存泄漏检测测试 ({iterations}轮, 每轮{requests_per_iteration}请求) ===")
        
        memory_snapshots = []
        
        config = {
            'enable_caching': False,
            'concurrency_optimization': {
                'max_concurrent_requests': 50,
                'connection_pool_size': 25,
                'request_timeout': 30.0
            }
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = setup_mock_plugin_manager(response_time_ms=5)
            mock_get_manager.return_value = mock_manager
            
            for iteration in range(iterations):
                print(f"执行第 {iteration + 1}/{iterations} 轮测试...")
                
                # 记录开始内存
                gc.collect()  # 强制垃圾回收
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # 创建新客户端
                client = create_fast_client(config=config)
                client.chat.completions._ensure_initialized()
                client.chat.completions._lazy_manager = mock_manager
                
                # 执行请求
                tasks = []
                for i in range(requests_per_iteration):
                    task = client.chat.completions.create_async(
                        messages=[{"role": "user", "content": f"内存测试 {iteration}-{i}"}],
                        model="gpt-3.5-turbo"
                    )
                    tasks.append(task)
                
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # 清理客户端
                await client.cleanup()
                del client
                
                # 记录结束内存
                gc.collect()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                memory_snapshots.append({
                    'iteration': iteration + 1,
                    'start_memory_mb': start_memory,
                    'end_memory_mb': end_memory,
                    'memory_diff_mb': end_memory - start_memory
                })
                
                print(f"  内存使用: {start_memory:.1f}MB -> {end_memory:.1f}MB (差异: {end_memory - start_memory:+.1f}MB)")
        
        # 分析内存趋势
        memory_diffs = [snapshot['memory_diff_mb'] for snapshot in memory_snapshots]
        avg_memory_diff = sum(memory_diffs) / len(memory_diffs)
        max_memory_diff = max(memory_diffs)
        
        print(f"\n内存泄漏分析:")
        print(f"  平均内存差异: {avg_memory_diff:+.1f}MB")
        print(f"  最大内存差异: {max_memory_diff:+.1f}MB")
        
        if avg_memory_diff > 10.0:  # 平均增长超过10MB认为可能有内存泄漏
            print(f"⚠ 可能存在内存泄漏: 平均增长 {avg_memory_diff:.1f}MB")
        else:
            print(f"✓ 内存使用正常: 平均增长 {avg_memory_diff:.1f}MB")
        
        return {
            'memory_snapshots': memory_snapshots,
            'avg_memory_diff_mb': avg_memory_diff,
            'max_memory_diff_mb': max_memory_diff,
            'has_memory_leak': avg_memory_diff > 10.0
        }
    
    async def _make_request_with_metrics(self, client, request_id: int, response_times: List[float], error_messages: List[str]):
        """执行请求并收集指标"""
        try:
            start_time = time.perf_counter()
            
            response = await client.chat.completions.create_async(
                messages=[{"role": "user", "content": f"稳定性测试 {request_id}"}],
                model="gpt-3.5-turbo"
            )
            
            end_time = time.perf_counter()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            return response
            
        except Exception as e:
            error_messages.append(str(e))
            raise
    
    def _print_stability_results(self, metrics: StabilityMetrics, resource_metrics: Dict[str, Any]):
        """输出稳定性测试结果"""
        total_time = metrics.end_time - metrics.start_time
        success_rate = (metrics.successful_requests / metrics.total_requests * 100) if metrics.total_requests > 0 else 0
        throughput = metrics.total_requests / total_time if total_time > 0 else 0
        
        print(f"\n稳定性测试结果:")
        print(f"  总运行时间: {total_time:.1f}s")
        print(f"  总请求数: {metrics.total_requests}")
        print(f"  成功请求: {metrics.successful_requests}")
        print(f"  失败请求: {metrics.failed_requests}")
        print(f"  成功率: {success_rate:.2f}%")
        print(f"  平均吞吐量: {throughput:.2f} ops/s")
        
        if metrics.response_times:
            avg_response_time = sum(metrics.response_times) / len(metrics.response_times)
            print(f"  平均响应时间: {avg_response_time*1000:.2f}ms")
        
        print(f"\n资源使用情况:")
        print(f"  平均内存使用: {resource_metrics['avg_memory_mb']:.1f}MB")
        print(f"  最大内存使用: {resource_metrics['max_memory_mb']:.1f}MB")
        print(f"  平均CPU使用: {resource_metrics['avg_cpu_percent']:.1f}%")
        print(f"  最大CPU使用: {resource_metrics['max_cpu_percent']:.1f}%")
        
        # 稳定性评估
        if success_rate >= 95.0:
            print(f"✓ 稳定性测试通过: 成功率 {success_rate:.2f}% ≥ 95%")
        else:
            print(f"✗ 稳定性测试失败: 成功率 {success_rate:.2f}% < 95%")


async def main():
    """主测试函数"""
    print("=== HarborAI 高并发稳定性测试 ===")
    
    tester = HighConcurrencyStabilityTester()
    
    try:
        # 1. 长时间运行稳定性测试
        await tester.test_long_running_stability(duration_seconds=30, concurrency=50)
        
        # 2. 内存泄漏检测测试
        await tester.test_memory_leak_detection(iterations=5, requests_per_iteration=50)
        
        print("\n=== 高并发稳定性测试完成 ===")
        
    except Exception as e:
        print(f"\n=== 测试失败: {e} ===")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())