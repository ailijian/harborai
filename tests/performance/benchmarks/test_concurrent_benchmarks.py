# -*- coding: utf-8 -*-
"""
并发基准测试

本模块实现了HarborAI项目的并发基准测试，包括：
- 线程池并发基准测试
- 异步并发基准测试
- 连接池并发基准测试
- 并发扩展性基准测试
- 并发错误处理基准测试

作者: HarborAI Team
版本: 1.0.0
创建时间: 2024-01-20
"""

import asyncio
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from unittest.mock import Mock, AsyncMock, patch
import pytest
import json
from datetime import datetime
import statistics
import queue
import random
from contextlib import asynccontextmanager

from tests.performance.benchmarks import BENCHMARK_CONFIG, PERFORMANCE_GRADES
from tests.performance import PERFORMANCE_CONFIG, SUPPORTED_VENDORS


@dataclass
class ConcurrentBenchmark:
    """
    并发基准测试结果
    
    记录并发基准测试的详细结果
    """
    test_name: str
    vendor: str
    model: str
    concurrency_type: str  # 'thread_pool', 'async', 'connection_pool'
    max_concurrency: int
    
    # 并发指标
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    concurrent_requests: int = 0
    
    # 性能指标
    total_duration: float = 0.0
    avg_response_time: float = 0.0
    concurrent_throughput: float = 0.0
    efficiency_ratio: float = 0.0  # 实际并发数/理论并发数
    
    # 响应时间分布
    response_times: List[float] = field(default_factory=list)
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # 并发特定指标
    thread_utilization: float = 0.0
    connection_reuse_rate: float = 0.0
    queue_wait_time: float = 0.0
    
    # 错误统计
    timeout_errors: int = 0
    connection_errors: int = 0
    other_errors: int = 0
    
    # 资源使用
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    active_threads: int = 0
    
    # 性能等级
    concurrent_grade: str = 'F'
    baseline_comparison: str = 'unknown'
    
    # 测试元数据
    test_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def calculate_metrics(self):
        """
        计算并发指标
        """
        if self.total_duration > 0:
            self.concurrent_throughput = self.successful_requests / self.total_duration
        
        if self.successful_requests > 0 and self.response_times:
            self.avg_response_time = sum(self.response_times) / len(self.response_times)
        
        if self.max_concurrency > 0:
            actual_concurrency = min(self.concurrent_requests, self.max_concurrency)
            self.efficiency_ratio = actual_concurrency / self.max_concurrency
        
        if self.response_times:
            self.response_times.sort()
            n = len(self.response_times)
            self.min_response_time = self.response_times[0]
            self.max_response_time = self.response_times[-1]
            self.p50_response_time = self.response_times[n // 2]
            if n >= 20:
                self.p95_response_time = self.response_times[int(n * 0.95)]
            if n >= 100:
                self.p99_response_time = self.response_times[int(n * 0.99)]
    
    def evaluate_performance(self):
        """
        评估并发性能等级
        """
        thresholds = BENCHMARK_CONFIG['baseline_thresholds']['concurrent_performance']
        
        # 综合评估：并发吞吐量 + 效率比
        score = self.concurrent_throughput * self.efficiency_ratio
        
        if score >= thresholds['excellent']:
            self.concurrent_grade = 'A+'
            self.baseline_comparison = 'excellent'
        elif score >= thresholds['good']:
            self.concurrent_grade = 'A'
            self.baseline_comparison = 'good'
        elif score >= thresholds['acceptable']:
            self.concurrent_grade = 'B'
            self.baseline_comparison = 'acceptable'
        elif score >= thresholds['poor']:
            self.concurrent_grade = 'C'
            self.baseline_comparison = 'poor'
        else:
            self.concurrent_grade = 'F'
            self.baseline_comparison = 'unacceptable'
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        返回:
            Dict[str, Any]: 基准测试结果字典
        """
        return {
            'test_name': self.test_name,
            'vendor': self.vendor,
            'model': self.model,
            'concurrency_config': {
                'type': self.concurrency_type,
                'max_concurrency': self.max_concurrency,
                'concurrent_requests': self.concurrent_requests
            },
            'performance_metrics': {
                'concurrent_throughput': self.concurrent_throughput,
                'efficiency_ratio': self.efficiency_ratio,
                'avg_response_time': self.avg_response_time,
                'thread_utilization': self.thread_utilization
            },
            'response_time_distribution': {
                'min': self.min_response_time,
                'max': self.max_response_time,
                'p50': self.p50_response_time,
                'p95': self.p95_response_time,
                'p99': self.p99_response_time
            },
            'error_statistics': {
                'timeout_errors': self.timeout_errors,
                'connection_errors': self.connection_errors,
                'other_errors': self.other_errors,
                'total_errors': self.failed_requests
            },
            'resource_usage': {
                'peak_memory_mb': self.peak_memory_mb,
                'avg_cpu_percent': self.avg_cpu_percent,
                'active_threads': self.active_threads
            },
            'performance': {
                'grade': self.concurrent_grade,
                'baseline_comparison': self.baseline_comparison
            },
            'summary': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
                'total_duration': self.total_duration,
                'test_timestamp': self.test_timestamp
            }
        }


class MockConcurrentAPI:
    """
    模拟并发测试API客户端
    
    用于并发基准测试的模拟API客户端
    """
    
    def __init__(self, vendor: str, model: str):
        self.vendor = vendor
        self.model = model
        self.request_count = 0
        self.active_connections = 0
        self.connection_pool_size = 20
        self.lock = threading.Lock()
        self.connection_reuse_count = 0
        self.total_connections_created = 0
        
        # 并发配置
        self.concurrent_config = self._get_concurrent_config()
    
    def _get_concurrent_config(self) -> Dict[str, Any]:
        """
        获取并发配置
        
        返回:
            Dict[str, Any]: 并发配置
        """
        configs = {
            'deepseek': {
                'deepseek-chat': {
                    'max_concurrent': 50,
                    'base_response_time': 0.5,
                    'concurrent_penalty': 0.02,  # 每增加1个并发请求增加2%延迟
                    'connection_overhead': 0.05
                },
                'deepseek-coder': {
                    'max_concurrent': 30,
                    'base_response_time': 1.2,
                    'concurrent_penalty': 0.03,
                    'connection_overhead': 0.08
                }
            },
            'ernie': {
                'ernie-3.5-8k': {
                    'max_concurrent': 60,
                    'base_response_time': 0.4,
                    'concurrent_penalty': 0.015,
                    'connection_overhead': 0.04
                },
                'ernie-4.0-8k': {
                    'max_concurrent': 45,
                    'base_response_time': 0.7,
                    'concurrent_penalty': 0.025,
                    'connection_overhead': 0.06
                }
            },
            'doubao': {
                'doubao-pro-4k': {
                    'max_concurrent': 35,
                    'base_response_time': 0.9,
                    'concurrent_penalty': 0.03,
                    'connection_overhead': 0.07
                }
            }
        }
        
        return configs.get(self.vendor, {}).get(self.model, {
            'max_concurrent': 40,
            'base_response_time': 1.0,
            'concurrent_penalty': 0.025,
            'connection_overhead': 0.06
        })
    
    def _acquire_connection(self) -> bool:
        """
        获取连接
        
        返回:
            bool: 是否成功获取连接
        """
        with self.lock:
            if self.active_connections < self.connection_pool_size:
                self.active_connections += 1
                
                # 模拟连接复用
                if self.total_connections_created > 0 and random.random() < 0.7:
                    self.connection_reuse_count += 1
                else:
                    self.total_connections_created += 1
                
                return True
            return False
    
    def _release_connection(self):
        """
        释放连接
        """
        with self.lock:
            if self.active_connections > 0:
                self.active_connections -= 1
    
    def send_request_sync(self, request_id: str = None, use_connection_pool: bool = True) -> Dict[str, Any]:
        """
        发送同步API请求
        
        参数:
            request_id: 请求ID
            use_connection_pool: 是否使用连接池
        
        返回:
            Dict[str, Any]: 模拟响应
        """
        start_time = time.time()
        
        # 连接池管理
        connection_acquired = True
        connection_wait_time = 0.0
        
        if use_connection_pool:
            connection_acquired = self._acquire_connection()
            if not connection_acquired:
                # 模拟等待连接
                wait_start = time.time()
                while not self._acquire_connection() and time.time() - wait_start < 2.0:
                    time.sleep(0.01)
                connection_wait_time = time.time() - wait_start
                connection_acquired = self.active_connections < self.connection_pool_size
        
        if not connection_acquired:
            raise Exception("连接池已满，无法获取连接")
        
        try:
            with self.lock:
                self.request_count += 1
                current_count = self.request_count
            
            # 计算并发影响
            concurrent_factor = 1.0 + (self.active_connections - 1) * self.concurrent_config['concurrent_penalty']
            
            # 计算响应时间
            base_time = self.concurrent_config['base_response_time']
            connection_overhead = self.concurrent_config['connection_overhead'] if use_connection_pool else 0
            
            response_time = (base_time + connection_overhead) * concurrent_factor
            response_time = max(0.05, response_time)  # 最小响应时间50ms
            
            # 模拟网络延迟
            time.sleep(response_time)
            
            return {
                'vendor': self.vendor,
                'model': self.model,
                'response_time': response_time,
                'connection_wait_time': connection_wait_time,
                'content': f"并发响应 #{current_count} 来自 {self.vendor}/{self.model}",
                'request_id': request_id or f"{self.vendor}_{self.model}_{current_count}",
                'timestamp': time.time(),
                'concurrent_level': self.active_connections
            }
        
        finally:
            if use_connection_pool:
                self._release_connection()
    
    async def send_request_async(self, request_id: str = None, semaphore: asyncio.Semaphore = None) -> Dict[str, Any]:
        """
        发送异步API请求
        
        参数:
            request_id: 请求ID
            semaphore: 异步信号量
        
        返回:
            Dict[str, Any]: 模拟响应
        """
        if semaphore:
            async with semaphore:
                # 获取连接并跟踪并发数
                self._acquire_connection()
                try:
                    return await self._send_async_request(request_id)
                finally:
                    self._release_connection()
        else:
            # 获取连接并跟踪并发数
            self._acquire_connection()
            try:
                return await self._send_async_request(request_id)
            finally:
                self._release_connection()
    
    async def _send_async_request(self, request_id: str = None) -> Dict[str, Any]:
        """
        内部异步请求方法
        
        参数:
            request_id: 请求ID
        
        返回:
            Dict[str, Any]: 模拟响应
        """
        with self.lock:
            self.request_count += 1
            current_count = self.request_count
            current_concurrent = self.active_connections
        
        # 异步请求通常有更好的并发性能
        concurrent_factor = 1.0 + (current_concurrent - 1) * self.concurrent_config['concurrent_penalty'] * 0.7
        
        base_time = self.concurrent_config['base_response_time'] * 0.8  # 异步减少20%响应时间
        response_time = base_time * concurrent_factor
        response_time = max(0.03, response_time)  # 最小响应时间30ms
        
        # 使用异步睡眠
        await asyncio.sleep(response_time)
        
        return {
            'vendor': self.vendor,
            'model': self.model,
            'response_time': response_time,
            'content': f"异步并发响应 #{current_count} 来自 {self.vendor}/{self.model}",
            'request_id': request_id or f"{self.vendor}_{self.model}_async_{current_count}",
            'timestamp': time.time(),
            'concurrent_level': current_concurrent
        }
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        获取连接统计信息
        
        返回:
            Dict[str, Any]: 连接统计
        """
        with self.lock:
            reuse_rate = self.connection_reuse_count / max(1, self.total_connections_created)
            return {
                'active_connections': self.active_connections,
                'total_connections_created': self.total_connections_created,
                'connection_reuse_count': self.connection_reuse_count,
                'connection_reuse_rate': reuse_rate,
                'pool_size': self.connection_pool_size
            }


class ConcurrentBenchmarkRunner:
    """
    并发基准测试运行器
    
    执行各种并发基准测试
    """
    
    def __init__(self):
        self.config = BENCHMARK_CONFIG
        self.results: List[ConcurrentBenchmark] = []
    
    def run_thread_pool_benchmark(self,
                                vendor: str,
                                model: str,
                                num_requests: int = 100,
                                max_workers: int = 10,
                                test_name: str = "thread_pool") -> ConcurrentBenchmark:
        """
        运行线程池并发基准测试
        
        参数:
            vendor: 厂商名称
            model: 模型名称
            num_requests: 总请求数量
            max_workers: 最大工作线程数
            test_name: 测试名称
        
        返回:
            ConcurrentBenchmark: 基准测试结果
        """
        client = MockConcurrentAPI(vendor, model)
        benchmark = ConcurrentBenchmark(
            test_name=test_name,
            vendor=vendor,
            model=model,
            concurrency_type='thread_pool',
            max_concurrency=max_workers
        )
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_id = {
                executor.submit(client.send_request_sync, f"thread_pool_{i}"): i
                for i in range(num_requests)
            }
            
            # 收集结果
            for future in as_completed(future_to_id):
                request_id = future_to_id[future]
                benchmark.total_requests += 1
                
                try:
                    response = future.result()
                    benchmark.successful_requests += 1
                    benchmark.response_times.append(response['response_time'])
                    benchmark.concurrent_requests = max(benchmark.concurrent_requests, response['concurrent_level'])
                except Exception as e:
                    benchmark.failed_requests += 1
                    print(f"线程池请求失败 {request_id}: {e}")
        
        end_time = time.time()
        benchmark.total_duration = end_time - start_time
        
        # 获取连接统计
        connection_stats = client.get_connection_stats()
        benchmark.connection_reuse_rate = connection_stats['connection_reuse_rate']
        benchmark.active_threads = max_workers
        
        # 计算指标和性能等级
        benchmark.calculate_metrics()
        benchmark.evaluate_performance()
        
        self.results.append(benchmark)
        return benchmark
    
    async def run_async_benchmark(self,
                                vendor: str,
                                model: str,
                                num_requests: int = 100,
                                max_concurrency: int = 20,
                                test_name: str = "async_concurrent") -> ConcurrentBenchmark:
        """
        运行异步并发基准测试
        
        参数:
            vendor: 厂商名称
            model: 模型名称
            num_requests: 总请求数量
            max_concurrency: 最大并发数
            test_name: 测试名称
        
        返回:
            ConcurrentBenchmark: 基准测试结果
        """
        client = MockConcurrentAPI(vendor, model)
        benchmark = ConcurrentBenchmark(
            test_name=test_name,
            vendor=vendor,
            model=model,
            concurrency_type='async',
            max_concurrency=max_concurrency
        )
        
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def send_single_request(request_id: str) -> Dict[str, Any]:
            """发送单个异步请求"""
            try:
                response = await client.send_request_async(request_id, semaphore)
                return {
                    'success': True,
                    'response_time': response['response_time'],
                    'concurrent_level': response['concurrent_level']
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        start_time = time.time()
        
        # 创建所有任务
        tasks = [send_single_request(f"async_{i}") for i in range(num_requests)]
        
        # 执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        benchmark.total_duration = end_time - start_time
        
        # 处理结果
        for i, result in enumerate(results):
            benchmark.total_requests += 1
            
            if isinstance(result, Exception):
                benchmark.failed_requests += 1
            elif result['success']:
                benchmark.successful_requests += 1
                benchmark.response_times.append(result['response_time'])
                benchmark.concurrent_requests = max(benchmark.concurrent_requests, result['concurrent_level'])
            else:
                benchmark.failed_requests += 1
        
        # 获取连接统计
        connection_stats = client.get_connection_stats()
        benchmark.connection_reuse_rate = connection_stats['connection_reuse_rate']
        
        # 计算指标和性能等级
        benchmark.calculate_metrics()
        benchmark.evaluate_performance()
        
        self.results.append(benchmark)
        return benchmark
    
    def run_connection_pool_benchmark(self,
                                    vendor: str,
                                    model: str,
                                    num_requests: int = 100,
                                    pool_size: int = 15,
                                    test_name: str = "connection_pool") -> ConcurrentBenchmark:
        """
        运行连接池并发基准测试
        
        参数:
            vendor: 厂商名称
            model: 模型名称
            num_requests: 总请求数量
            pool_size: 连接池大小
            test_name: 测试名称
        
        返回:
            ConcurrentBenchmark: 基准测试结果
        """
        client = MockConcurrentAPI(vendor, model)
        client.connection_pool_size = pool_size
        
        benchmark = ConcurrentBenchmark(
            test_name=test_name,
            vendor=vendor,
            model=model,
            concurrency_type='connection_pool',
            max_concurrency=pool_size
        )
        
        # 使用线程池模拟连接池并发
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=pool_size * 2) as executor:  # 允许更多线程等待连接
            future_to_id = {
                executor.submit(client.send_request_sync, f"pool_{i}", True): i
                for i in range(num_requests)
            }
            
            total_wait_time = 0.0
            
            for future in as_completed(future_to_id):
                request_id = future_to_id[future]
                benchmark.total_requests += 1
                
                try:
                    response = future.result()
                    benchmark.successful_requests += 1
                    benchmark.response_times.append(response['response_time'])
                    total_wait_time += response.get('connection_wait_time', 0)
                    benchmark.concurrent_requests = max(benchmark.concurrent_requests, response['concurrent_level'])
                except Exception as e:
                    benchmark.failed_requests += 1
                    if "连接池已满" in str(e):
                        benchmark.connection_errors += 1
                    else:
                        benchmark.other_errors += 1
        
        end_time = time.time()
        benchmark.total_duration = end_time - start_time
        
        # 计算连接池特定指标
        if benchmark.successful_requests > 0:
            benchmark.queue_wait_time = total_wait_time / benchmark.successful_requests
        
        connection_stats = client.get_connection_stats()
        benchmark.connection_reuse_rate = connection_stats['connection_reuse_rate']
        
        # 计算指标和性能等级
        benchmark.calculate_metrics()
        benchmark.evaluate_performance()
        
        self.results.append(benchmark)
        return benchmark
    
    def run_scalability_benchmark(self,
                                vendor: str,
                                model: str,
                                concurrency_levels: List[int],
                                requests_per_level: int = 50) -> List[ConcurrentBenchmark]:
        """
        运行并发扩展性基准测试
        
        参数:
            vendor: 厂商名称
            model: 模型名称
            concurrency_levels: 并发级别列表
            requests_per_level: 每个级别的请求数量
        
        返回:
            List[ConcurrentBenchmark]: 基准测试结果列表
        """
        results = []
        
        for concurrency in concurrency_levels:
            test_name = f"scalability_c{concurrency}"
            
            # 使用线程池测试不同并发级别
            result = self.run_thread_pool_benchmark(
                vendor, model, requests_per_level, concurrency, test_name
            )
            
            results.append(result)
        
        return results
    
    async def run_mixed_concurrency_benchmark(self,
                                            vendor: str,
                                            model: str,
                                            num_requests: int = 100,
                                            thread_workers: int = 5,
                                            async_concurrency: int = 10,
                                            test_name: str = "mixed_concurrency") -> ConcurrentBenchmark:
        """
        运行混合并发基准测试（线程池 + 异步）
        
        参数:
            vendor: 厂商名称
            model: 模型名称
            num_requests: 总请求数量
            thread_workers: 线程池工作线程数
            async_concurrency: 异步并发数
            test_name: 测试名称
        
        返回:
            ConcurrentBenchmark: 基准测试结果
        """
        client = MockConcurrentAPI(vendor, model)
        benchmark = ConcurrentBenchmark(
            test_name=test_name,
            vendor=vendor,
            model=model,
            concurrency_type='mixed',
            max_concurrency=thread_workers + async_concurrency
        )
        
        start_time = time.time()
        
        # 分配请求：一半给线程池，一半给异步
        thread_requests = num_requests // 2
        async_requests = num_requests - thread_requests
        
        # 线程池任务
        def thread_task():
            with ThreadPoolExecutor(max_workers=thread_workers) as executor:
                futures = [executor.submit(client.send_request_sync, f"mixed_thread_{i}")
                          for i in range(thread_requests)]
                
                thread_results = []
                for future in as_completed(futures):
                    try:
                        response = future.result()
                        thread_results.append({
                            'success': True,
                            'response_time': response['response_time'],
                            'concurrent_level': response['concurrent_level']
                        })
                    except Exception as e:
                        thread_results.append({'success': False, 'error': str(e)})
                
                return thread_results
        
        # 异步任务
        async def async_task():
            semaphore = asyncio.Semaphore(async_concurrency)
            
            async def send_async_request(request_id: str):
                try:
                    response = await client.send_request_async(request_id, semaphore)
                    return {
                        'success': True,
                        'response_time': response['response_time'],
                        'concurrent_level': response['concurrent_level']
                    }
                except Exception as e:
                    return {'success': False, 'error': str(e)}
            
            tasks = [send_async_request(f"mixed_async_{i}") for i in range(async_requests)]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # 并行执行线程池和异步任务
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            thread_future = executor.submit(thread_task)
            async_results = await async_task()
            thread_results = thread_future.result()
        
        end_time = time.time()
        benchmark.total_duration = end_time - start_time
        
        # 合并结果
        all_results = thread_results + list(async_results)
        
        for result in all_results:
            benchmark.total_requests += 1
            
            if isinstance(result, Exception) or not result.get('success', False):
                benchmark.failed_requests += 1
            else:
                benchmark.successful_requests += 1
                benchmark.response_times.append(result['response_time'])
                benchmark.concurrent_requests = max(benchmark.concurrent_requests, result['concurrent_level'])
        
        # 计算指标和性能等级
        benchmark.calculate_metrics()
        benchmark.evaluate_performance()
        
        self.results.append(benchmark)
        return benchmark
    
    def generate_concurrent_report(self, results: List[ConcurrentBenchmark]) -> Dict[str, Any]:
        """
        生成并发基准测试报告
        
        参数:
            results: 基准测试结果列表
        
        返回:
            Dict[str, Any]: 基准测试报告
        """
        if not results:
            return {'error': '没有并发基准测试结果'}
        
        report = {
            'summary': {
                'total_benchmarks': len(results),
                'test_timestamp': datetime.now().isoformat(),
                'concurrency_types': list(set(r.concurrency_type for r in results)),
                'avg_efficiency': 0.0,
                'max_throughput': 0.0
            },
            'results': [result.to_dict() for result in results],
            'rankings': {
                'highest_throughput': None,
                'best_efficiency': None,
                'most_scalable': None
            },
            'analysis': {
                'concurrency_comparison': {},
                'scalability_analysis': None
            },
            'recommendations': []
        }
        
        # 统计分析
        efficiencies = [r.efficiency_ratio for r in results if r.efficiency_ratio > 0]
        throughputs = [r.concurrent_throughput for r in results]
        
        if efficiencies:
            report['summary']['avg_efficiency'] = statistics.mean(efficiencies)
        if throughputs:
            report['summary']['max_throughput'] = max(throughputs)
        
        # 排名分析
        if results:
            # 最高吞吐量
            highest = max(results, key=lambda x: x.concurrent_throughput)
            report['rankings']['highest_throughput'] = {
                'test_name': highest.test_name,
                'vendor': highest.vendor,
                'model': highest.model,
                'concurrency_type': highest.concurrency_type,
                'throughput': highest.concurrent_throughput
            }
            
            # 最佳效率
            if efficiencies:
                best_efficiency = max(results, key=lambda x: x.efficiency_ratio)
                report['rankings']['best_efficiency'] = {
                    'test_name': best_efficiency.test_name,
                    'vendor': best_efficiency.vendor,
                    'model': best_efficiency.model,
                    'efficiency_ratio': best_efficiency.efficiency_ratio
                }
        
        # 并发类型对比
        type_stats = {}
        for concurrency_type in report['summary']['concurrency_types']:
            type_results = [r for r in results if r.concurrency_type == concurrency_type]
            if type_results:
                type_stats[concurrency_type] = {
                    'count': len(type_results),
                    'avg_throughput': statistics.mean([r.concurrent_throughput for r in type_results]),
                    'avg_efficiency': statistics.mean([r.efficiency_ratio for r in type_results if r.efficiency_ratio > 0])
                }
        
        report['analysis']['concurrency_comparison'] = type_stats
        
        # 生成建议
        if type_stats:
            best_type = max(type_stats.items(), key=lambda x: x[1]['avg_throughput'])
            report['recommendations'].append(
                f"推荐使用 {best_type[0]} 并发模式，平均吞吐量最高: {best_type[1]['avg_throughput']:.2f} req/s"
            )
        
        poor_efficiency = [r for r in results if r.efficiency_ratio < 0.6]
        if poor_efficiency:
            report['recommendations'].append(
                f"发现 {len(poor_efficiency)} 个低效率配置（效率 < 60%），建议优化并发策略"
            )
        
        return report
    
    def reset_results(self):
        """重置基准测试结果"""
        self.results.clear()


class TestConcurrentBenchmarks:
    """
    并发基准测试类
    
    包含各种并发基准测试场景
    """
    
    def setup_method(self):
        """测试方法设置"""
        self.benchmark_runner = ConcurrentBenchmarkRunner()
        self.config = BENCHMARK_CONFIG
    
    def teardown_method(self):
        """测试方法清理"""
        self.benchmark_runner.reset_results()
    
    def _print_concurrent_summary(self, results: List[ConcurrentBenchmark]):
        """打印并发基准测试摘要"""
        if not results:
            print("\n没有并发基准测试结果")
            return
        
        print(f"\n=== 并发基准测试结果 ===")
        print(f"测试数量: {len(results)}")
        
        for result in results:
            print(f"\n{result.test_name} - {result.vendor}/{result.model}:")
            print(f"  并发类型: {result.concurrency_type}")
            print(f"  最大并发数: {result.max_concurrency}")
            print(f"  实际并发数: {result.concurrent_requests}")
            print(f"  并发等级: {result.concurrent_grade} ({result.baseline_comparison})")
            print(f"  并发吞吐量: {result.concurrent_throughput:.2f} req/s")
            print(f"  效率比: {result.efficiency_ratio:.2%}")
            print(f"  平均响应时间: {result.avg_response_time*1000:.1f}ms")
            print(f"  连接复用率: {result.connection_reuse_rate:.2%}")
            print(f"  成功率: {result.successful_requests}/{result.total_requests} ({result.successful_requests/result.total_requests*100:.1f}%)")
            if result.queue_wait_time > 0:
                print(f"  平均等待时间: {result.queue_wait_time*1000:.1f}ms")
    
    @pytest.mark.benchmark
    @pytest.mark.quick_benchmark
    def test_thread_pool_concurrent_benchmark(self):
        """
        线程池并发基准测试
        
        测试线程池环境下的并发性能
        """
        vendor = 'deepseek'
        model = 'deepseek-chat'
        num_requests = 60
        max_workers = 8
        
        # 运行线程池并发基准测试
        result = self.benchmark_runner.run_thread_pool_benchmark(
            vendor, model, num_requests, max_workers
        )
        
        self._print_concurrent_summary([result])
        
        # 基准测试断言
        assert result.total_requests == num_requests
        assert result.successful_requests > 0
        assert result.concurrency_type == 'thread_pool'
        assert result.max_concurrency == max_workers
        assert result.concurrent_throughput > 0
        assert result.concurrent_grade in ['A+', 'A', 'B', 'C', 'F']
        
        # 并发性能要求
        assert result.efficiency_ratio >= 0.3  # 至少30%的效率
        assert result.successful_requests / result.total_requests >= 0.9  # 90%成功率
    
    @pytest.mark.benchmark
    @pytest.mark.standard_benchmark
    @pytest.mark.asyncio
    async def test_async_concurrent_benchmark(self):
        """
        异步并发基准测试
        
        测试异步环境下的并发性能
        """
        vendor = 'ernie'
        model = 'ernie-3.5-8k'
        num_requests = 80
        max_concurrency = 15
        
        # 运行异步并发基准测试
        result = await self.benchmark_runner.run_async_benchmark(
            vendor, model, num_requests, max_concurrency
        )
        
        self._print_concurrent_summary([result])
        
        # 基准测试断言
        assert result.total_requests == num_requests
        assert result.successful_requests > 0
        assert result.concurrency_type == 'async'
        assert result.max_concurrency == max_concurrency
        assert result.concurrent_throughput > 0
        
        # 异步并发性能要求
        assert result.efficiency_ratio >= 0.4  # 异步应该有更好的效率
        assert result.successful_requests / result.total_requests >= 0.85  # 85%成功率
    
    @pytest.mark.benchmark
    @pytest.mark.standard_benchmark
    def test_connection_pool_benchmark(self):
        """
        连接池并发基准测试
        
        测试连接池环境下的并发性能
        """
        vendor = 'doubao'
        model = 'doubao-pro-4k'
        num_requests = 70
        pool_size = 12
        
        # 运行连接池并发基准测试
        result = self.benchmark_runner.run_connection_pool_benchmark(
            vendor, model, num_requests, pool_size
        )
        
        self._print_concurrent_summary([result])
        
        # 基准测试断言
        assert result.total_requests == num_requests
        assert result.successful_requests > 0
        assert result.concurrency_type == 'connection_pool'
        assert result.max_concurrency == pool_size
        
        # 连接池特定断言
        assert result.connection_reuse_rate >= 0.0  # 连接复用率
        assert result.queue_wait_time >= 0.0  # 等待时间
        
        # 连接池性能要求
        assert result.concurrent_throughput >= self.config['baseline_thresholds']['concurrent_performance']['poor']
    
    @pytest.mark.benchmark
    @pytest.mark.comprehensive_benchmark
    def test_concurrent_scalability_benchmark(self):
        """
        并发扩展性基准测试
        
        测试不同并发级别下的扩展性
        """
        vendor = 'deepseek'
        model = 'deepseek-coder'
        concurrency_levels = [2, 4, 8, 16]
        requests_per_level = 40
        
        # 运行并发扩展性基准测试
        results = self.benchmark_runner.run_scalability_benchmark(
            vendor, model, concurrency_levels, requests_per_level
        )
        
        self._print_concurrent_summary(results)
        
        # 生成扩展性报告
        report = self.benchmark_runner.generate_concurrent_report(results)
        
        print(f"\n=== 并发扩展性分析报告 ===")
        print(f"测试的并发级别: {concurrency_levels}")
        print(f"平均效率: {report['summary']['avg_efficiency']:.2%}")
        print(f"最大吞吐量: {report['summary']['max_throughput']:.2f} req/s")
        
        if report['rankings']['highest_throughput']:
            best = report['rankings']['highest_throughput']
            print(f"最佳并发配置: {best['concurrency_type']} ({best['throughput']:.2f} req/s)")
        
        # 扩展性断言
        assert len(results) == len(concurrency_levels)
        assert all(r.total_requests == requests_per_level for r in results)
        
        # 检查并发扩展性
        throughputs = [r.concurrent_throughput for r in results]
        max_throughput = max(throughputs)
        min_throughput = min(throughputs)
        
        # 最大吞吐量应该明显高于最小吞吐量
        assert max_throughput > min_throughput * 1.5  # 至少50%的提升
    
    @pytest.mark.benchmark
    @pytest.mark.comprehensive_benchmark
    @pytest.mark.asyncio
    async def test_mixed_concurrency_benchmark(self):
        """
        混合并发基准测试
        
        测试线程池和异步混合并发性能
        """
        vendor = 'ernie'
        model = 'ernie-4.0-8k'
        num_requests = 100
        thread_workers = 6
        async_concurrency = 12
        
        # 运行混合并发基准测试
        result = await self.benchmark_runner.run_mixed_concurrency_benchmark(
            vendor, model, num_requests, thread_workers, async_concurrency
        )
        
        self._print_concurrent_summary([result])
        
        # 基准测试断言
        assert result.total_requests == num_requests
        assert result.successful_requests > 0
        assert result.concurrency_type == 'mixed'
        assert result.max_concurrency == thread_workers + async_concurrency
        
        # 混合并发性能要求
        assert result.concurrent_throughput > 0
        assert result.efficiency_ratio >= 0.25  # 混合模式可能效率稍低
        assert result.successful_requests / result.total_requests >= 0.8  # 80%成功率
    
    @pytest.mark.benchmark
    @pytest.mark.comprehensive_benchmark
    def test_concurrency_type_comparison_benchmark(self):
        """
        并发类型对比基准测试
        
        对比不同并发类型的性能
        """
        vendor = 'deepseek'
        model = 'deepseek-chat'
        num_requests = 60
        concurrency_level = 10
        
        all_results = []
        
        # 线程池并发测试
        thread_result = self.benchmark_runner.run_thread_pool_benchmark(
            vendor, model, num_requests, concurrency_level, "type_comparison_thread"
        )
        all_results.append(thread_result)
        
        # 连接池并发测试
        pool_result = self.benchmark_runner.run_connection_pool_benchmark(
            vendor, model, num_requests, concurrency_level, "type_comparison_pool"
        )
        all_results.append(pool_result)
        
        self._print_concurrent_summary(all_results)
        
        # 生成对比报告
        report = self.benchmark_runner.generate_concurrent_report(all_results)
        
        print(f"\n=== 并发类型对比报告 ===")
        print(f"参与对比的并发类型: {report['summary']['concurrency_types']}")
        
        if report['analysis']['concurrency_comparison']:
            for ctype, stats in report['analysis']['concurrency_comparison'].items():
                print(f"{ctype}: 平均吞吐量 {stats['avg_throughput']:.2f} req/s, 平均效率 {stats['avg_efficiency']:.2%}")
        
        if report['recommendations']:
            print(f"\n建议:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        # 对比断言
        assert len(all_results) >= 2
        assert all(r.total_requests == num_requests for r in all_results)
        
        # 至少有一种并发类型达到可接受的性能
        acceptable_results = [r for r in all_results if r.concurrent_grade in ['A+', 'A', 'B']]
        assert len(acceptable_results) > 0, "没有并发类型达到可接受的性能水平"
    
    @pytest.mark.benchmark
    @pytest.mark.performance_benchmark
    def test_concurrent_benchmark_with_pytest_benchmark(self, benchmark):
        """
        使用pytest-benchmark的并发基准测试
        
        集成pytest-benchmark进行精确的并发性能测量
        """
        vendor = 'ernie'
        model = 'ernie-3.5-8k'
        
        def concurrent_test_function():
            """被基准测试的函数"""
            return self.benchmark_runner.run_thread_pool_benchmark(
                vendor, model, 30, 6, "pytest_concurrent_benchmark"
            )
        
        # 使用pytest-benchmark运行基准测试
        result = benchmark(concurrent_test_function)
        
        self._print_concurrent_summary([result])
        
        print(f"\n=== pytest-benchmark 并发统计 ===")
        print(f"基准测试函数: {concurrent_test_function.__name__}")
        
        # pytest-benchmark断言
        assert result.total_requests == 30
        assert result.successful_requests > 0
        assert result.concurrent_throughput > 0
        
        # 并发性能基准断言
        assert result.concurrent_throughput >= self.config['baseline_thresholds']['concurrent_performance']['poor']
        assert result.efficiency_ratio >= 0.2  # 至少20%的效率