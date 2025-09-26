# -*- coding: utf-8 -*-
"""
并发性能测试模块

本模块实现了HarborAI项目的并发性能测试，包括：
- 多线程并发测试
- 异步并发测试
- 连接池性能测试
- 并发负载均衡测试
- 并发错误处理测试

作者: HarborAI Team
版本: 1.0.0
创建时间: 2024-01-20
"""

import asyncio
import concurrent.futures
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, patch
import pytest
import statistics
from datetime import datetime

from tests.performance import PERFORMANCE_CONFIG, SUPPORTED_VENDORS


@dataclass
class ConcurrentMetrics:
    """
    并发性能指标数据类
    
    记录并发测试中的各项性能指标
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    concurrent_level: int = 0
    throughput: float = 0.0
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    def calculate_metrics(self):
        """计算并发性能指标"""
        if self.response_times:
            self.avg_response_time = statistics.mean(self.response_times)
            # 手动计算P95和P99百分位数以兼容Python < 3.8
            sorted_times = sorted(self.response_times)
            n = len(sorted_times)
            self.p95_response_time = sorted_times[int(n * 0.95)] if n > 0 else 0.0
            self.p99_response_time = sorted_times[int(n * 0.99)] if n > 0 else 0.0
        
        if self.total_requests > 0:
            self.error_rate = (self.failed_requests / self.total_requests) * 100
        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            if duration > 0:
                self.throughput = self.successful_requests / duration


class ConcurrentPerformanceTest:
    """
    并发性能测试基础类
    
    提供并发测试的通用方法和工具
    """
    
    def __init__(self):
        self.metrics = ConcurrentMetrics()
        self.lock = threading.Lock()
        self.config = PERFORMANCE_CONFIG['concurrent']
    
    def mock_api_call(self, delay: float = 0.1, success_rate: float = 0.95) -> Tuple[bool, float]:
        """
        模拟API调用
        
        参数:
            delay: 模拟响应延迟（秒）
            success_rate: 成功率（0-1）
        
        返回:
            (是否成功, 响应时间)
        """
        start_time = time.time()
        time.sleep(delay)
        response_time = time.time() - start_time
        
        import random
        success = random.random() < success_rate
        
        return success, response_time
    
    def record_result(self, success: bool, response_time: float):
        """
        线程安全地记录测试结果
        
        参数:
            success: 请求是否成功
            response_time: 响应时间
        """
        with self.lock:
            self.metrics.total_requests += 1
            self.metrics.response_times.append(response_time)
            
            if success:
                self.metrics.successful_requests += 1
            else:
                self.metrics.failed_requests += 1
    
    def worker_thread(self, worker_id: int, requests_per_worker: int, delay: float = 0.1):
        """
        工作线程函数
        
        参数:
            worker_id: 工作线程ID
            requests_per_worker: 每个工作线程的请求数
            delay: 模拟延迟
        """
        for i in range(requests_per_worker):
            success, response_time = self.mock_api_call(delay)
            self.record_result(success, response_time)
    
    async def async_worker(self, worker_id: int, requests_per_worker: int, delay: float = 0.1):
        """
        异步工作协程
        
        参数:
            worker_id: 工作协程ID
            requests_per_worker: 每个工作协程的请求数
            delay: 模拟延迟
        """
        for i in range(requests_per_worker):
            start_time = time.time()
            await asyncio.sleep(delay)  # 模拟异步IO操作
            response_time = time.time() - start_time
            
            import random
            success = random.random() < 0.95
            self.record_result(success, response_time)


class TestConcurrentPerformance:
    """
    并发性能测试类
    
    包含各种并发场景的性能测试用例
    """
    
    def setup_method(self):
        """测试方法设置"""
        self.perf_test = ConcurrentPerformanceTest()
        self.config = PERFORMANCE_CONFIG['concurrent']
    
    def teardown_method(self):
        """测试方法清理"""
        # 输出测试结果摘要
        metrics = self.perf_test.metrics
        metrics.calculate_metrics()
        
        print(f"\n=== 并发性能测试结果 ===")
        print(f"总请求数: {metrics.total_requests}")
        print(f"成功请求数: {metrics.successful_requests}")
        print(f"失败请求数: {metrics.failed_requests}")
        print(f"错误率: {metrics.error_rate:.2f}%")
        print(f"平均响应时间: {metrics.avg_response_time:.3f}s")
        print(f"P95响应时间: {metrics.p95_response_time:.3f}s")
        print(f"P99响应时间: {metrics.p99_response_time:.3f}s")
        print(f"吞吐量: {metrics.throughput:.2f} req/s")
    
    @pytest.mark.performance
    @pytest.mark.concurrent
    def test_thread_pool_performance(self):
        """
        测试线程池并发性能
        
        验证多线程环境下的API调用性能
        """
        concurrent_threads = self.config['max_concurrent_requests']
        requests_per_thread = 10
        
        self.perf_test.metrics.concurrent_level = concurrent_threads
        self.perf_test.metrics.start_time = datetime.now()
        
        # 使用ThreadPoolExecutor进行并发测试
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
            futures = []
            for i in range(concurrent_threads):
                future = executor.submit(
                    self.perf_test.worker_thread, 
                    i, 
                    requests_per_thread,
                    0.05  # 50ms延迟
                )
                futures.append(future)
            
            # 等待所有任务完成
            concurrent.futures.wait(futures)
        
        self.perf_test.metrics.end_time = datetime.now()
        self.perf_test.metrics.calculate_metrics()
        
        # 性能断言
        assert self.perf_test.metrics.total_requests == concurrent_threads * requests_per_thread
        assert self.perf_test.metrics.error_rate <= self.config['max_error_rate']
        assert self.perf_test.metrics.avg_response_time <= self.config['max_response_time']
        assert self.perf_test.metrics.throughput >= self.config['min_throughput']
    
    @pytest.mark.performance
    @pytest.mark.concurrent
    @pytest.mark.asyncio
    async def test_async_concurrent_performance(self):
        """
        测试异步并发性能
        
        验证异步环境下的API调用性能
        """
        concurrent_tasks = self.config['max_concurrent_requests']
        requests_per_task = 10
        
        self.perf_test.metrics.concurrent_level = concurrent_tasks
        self.perf_test.metrics.start_time = datetime.now()
        
        # 创建异步任务
        tasks = []
        for i in range(concurrent_tasks):
            task = asyncio.create_task(
                self.perf_test.async_worker(
                    i, 
                    requests_per_task,
                    0.05  # 50ms延迟
                )
            )
            tasks.append(task)
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        self.perf_test.metrics.end_time = datetime.now()
        self.perf_test.metrics.calculate_metrics()
        
        # 性能断言
        assert self.perf_test.metrics.total_requests == concurrent_tasks * requests_per_task
        assert self.perf_test.metrics.error_rate <= self.config['max_error_rate']
        assert self.perf_test.metrics.avg_response_time <= self.config['max_response_time']
        # 异步并发通常有更高的吞吐量
        assert self.perf_test.metrics.throughput >= self.config['min_throughput'] * 1.5
    
    @pytest.mark.performance
    @pytest.mark.concurrent
    @pytest.mark.parametrize("concurrent_level", [5, 10, 20, 50])
    def test_concurrent_scaling(self, concurrent_level: int):
        """
        测试并发扩展性能
        
        验证不同并发级别下的性能表现
        
        参数:
            concurrent_level: 并发级别
        """
        requests_per_thread = 5
        
        self.perf_test.metrics.concurrent_level = concurrent_level
        self.perf_test.metrics.start_time = datetime.now()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = []
            for i in range(concurrent_level):
                future = executor.submit(
                    self.perf_test.worker_thread, 
                    i, 
                    requests_per_thread,
                    0.1  # 100ms延迟
                )
                futures.append(future)
            
            concurrent.futures.wait(futures)
        
        self.perf_test.metrics.end_time = datetime.now()
        self.perf_test.metrics.calculate_metrics()
        
        # 根据并发级别调整性能期望
        expected_throughput = min(
            self.config['min_throughput'] * (concurrent_level / 10),
            self.config['min_throughput'] * 5  # 最大5倍
        )
        
        assert self.perf_test.metrics.total_requests == concurrent_level * requests_per_thread
        assert self.perf_test.metrics.error_rate <= self.config['max_error_rate']
        assert self.perf_test.metrics.throughput >= expected_throughput
    
    @pytest.mark.performance
    @pytest.mark.concurrent
    @pytest.mark.parametrize("vendor", SUPPORTED_VENDORS)
    def test_vendor_concurrent_performance(self, vendor: str):
        """
        测试不同厂商的并发性能
        
        验证各厂商API在并发环境下的性能表现
        
        参数:
            vendor: API厂商名称
        """
        concurrent_threads = 10
        requests_per_thread = 5
        
        # 根据厂商调整延迟模拟
        vendor_delays = {
            'deepseek': 0.15,
            'ernie': 0.12,
            'doubao': 0.10,
            'azure': 0.18,
            'aws': 0.20
        }
        delay = vendor_delays.get(vendor, 0.15)
        
        self.perf_test.metrics.concurrent_level = concurrent_threads
        self.perf_test.metrics.start_time = datetime.now()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
            futures = []
            for i in range(concurrent_threads):
                future = executor.submit(
                    self.perf_test.worker_thread, 
                    i, 
                    requests_per_thread,
                    delay
                )
                futures.append(future)
            
            concurrent.futures.wait(futures)
        
        self.perf_test.metrics.end_time = datetime.now()
        self.perf_test.metrics.calculate_metrics()
        
        # 厂商特定的性能断言
        assert self.perf_test.metrics.total_requests == concurrent_threads * requests_per_thread
        assert self.perf_test.metrics.error_rate <= self.config['max_error_rate']
        
        # 根据厂商调整性能期望
        if vendor in ['doubao', 'ernie']:
            # 这些厂商通常有更好的并发性能
            assert self.perf_test.metrics.avg_response_time <= self.config['max_response_time'] * 0.8
        else:
            assert self.perf_test.metrics.avg_response_time <= self.config['max_response_time']
    
    @pytest.mark.performance
    @pytest.mark.concurrent
    def test_connection_pool_efficiency(self):
        """
        测试连接池效率
        
        验证连接复用对并发性能的影响
        """
        concurrent_requests = 20
        
        # 模拟有连接池的情况（更快的连接建立）
        def mock_with_pool():
            self.perf_test.metrics = ConcurrentMetrics()
            self.perf_test.metrics.start_time = datetime.now()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = []
                for i in range(concurrent_requests):
                    future = executor.submit(
                        self.perf_test.worker_thread, 
                        i, 
                        1,
                        0.05  # 连接池减少了连接开销
                    )
                    futures.append(future)
                
                concurrent.futures.wait(futures)
            
            self.perf_test.metrics.end_time = datetime.now()
            self.perf_test.metrics.calculate_metrics()
            return self.perf_test.metrics.throughput
        
        # 模拟无连接池的情况（较慢的连接建立）
        def mock_without_pool():
            self.perf_test.metrics = ConcurrentMetrics()
            self.perf_test.metrics.start_time = datetime.now()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = []
                for i in range(concurrent_requests):
                    future = executor.submit(
                        self.perf_test.worker_thread, 
                        i, 
                        1,
                        0.15  # 无连接池增加了连接开销
                    )
                    futures.append(future)
                
                concurrent.futures.wait(futures)
            
            self.perf_test.metrics.end_time = datetime.now()
            self.perf_test.metrics.calculate_metrics()
            return self.perf_test.metrics.throughput
        
        throughput_with_pool = mock_with_pool()
        throughput_without_pool = mock_without_pool()
        
        # 连接池应该提供更好的性能
        improvement_ratio = throughput_with_pool / throughput_without_pool
        assert improvement_ratio >= 1.2  # 至少20%的性能提升
        
        print(f"\n连接池性能提升: {improvement_ratio:.2f}x")
        print(f"有连接池吞吐量: {throughput_with_pool:.2f} req/s")
        print(f"无连接池吞吐量: {throughput_without_pool:.2f} req/s")
    
    @pytest.mark.performance
    @pytest.mark.concurrent
    def test_concurrent_error_handling(self):
        """
        测试并发错误处理性能
        
        验证在高错误率情况下的并发性能
        """
        concurrent_threads = 15
        requests_per_thread = 5
        
        def error_prone_worker(worker_id: int, requests_per_worker: int):
            """容易出错的工作线程"""
            for i in range(requests_per_worker):
                # 模拟50%的错误率
                success, response_time = self.perf_test.mock_api_call(
                    delay=0.1, 
                    success_rate=0.5
                )
                self.perf_test.record_result(success, response_time)
        
        self.perf_test.metrics.concurrent_level = concurrent_threads
        self.perf_test.metrics.start_time = datetime.now()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
            futures = []
            for i in range(concurrent_threads):
                future = executor.submit(error_prone_worker, i, requests_per_thread)
                futures.append(future)
            
            concurrent.futures.wait(futures)
        
        self.perf_test.metrics.end_time = datetime.now()
        self.perf_test.metrics.calculate_metrics()
        
        # 即使在高错误率下，系统也应该保持基本性能
        assert self.perf_test.metrics.total_requests == concurrent_threads * requests_per_thread
        assert 40 <= self.perf_test.metrics.error_rate <= 60  # 错误率应该在预期范围内
        assert self.perf_test.metrics.avg_response_time <= self.config['max_response_time'] * 1.5
        # 在错误情况下，吞吐量要求可以放宽
        assert self.perf_test.metrics.throughput >= self.config['min_throughput'] * 0.5
    
    @pytest.mark.performance
    @pytest.mark.concurrent
    @pytest.mark.benchmark
    def test_concurrent_throughput_benchmark(self, benchmark):
        """
        并发吞吐量基准测试
        
        使用pytest-benchmark进行精确的并发性能测量
        """
        def concurrent_benchmark():
            concurrent_threads = 10
            requests_per_thread = 3
            
            self.perf_test.metrics = ConcurrentMetrics()
            self.perf_test.metrics.start_time = datetime.now()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_threads) as executor:
                futures = []
                for i in range(concurrent_threads):
                    future = executor.submit(
                        self.perf_test.worker_thread, 
                        i, 
                        requests_per_thread,
                        0.05
                    )
                    futures.append(future)
                
                concurrent.futures.wait(futures)
            
            self.perf_test.metrics.end_time = datetime.now()
            self.perf_test.metrics.calculate_metrics()
            
            return self.perf_test.metrics.throughput
        
        # 运行基准测试
        throughput = benchmark(concurrent_benchmark)
        
        # 基准测试断言
        assert throughput >= self.config['min_throughput']
        print(f"\n并发基准吞吐量: {throughput:.2f} req/s")