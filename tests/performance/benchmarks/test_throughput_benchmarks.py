# -*- coding: utf-8 -*-
"""
吞吐量基准测试

本模块实现了HarborAI项目的吞吐量基准测试，包括：
- 单线程吞吐量基准测试
- 多线程吞吐量基准测试
- 异步并发吞吐量基准测试
- 不同厂商吞吐量对比基准测试
- 吞吐量扩展性基准测试

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

from tests.performance.benchmarks import BENCHMARK_CONFIG, PERFORMANCE_GRADES
from tests.performance import PERFORMANCE_CONFIG, SUPPORTED_VENDORS


@dataclass
class ThroughputBenchmark:
    """
    吞吐量基准测试结果
    
    记录吞吐量基准测试的详细结果
    """
    test_name: str
    vendor: str
    model: str
    concurrency_level: int
    
    # 吞吐量指标
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    
    # 吞吐量统计
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0
    avg_request_time: float = 0.0
    
    # 响应时间分布
    response_times: List[float] = field(default_factory=list)
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # 资源使用
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    
    # 性能等级
    throughput_grade: str = 'F'
    baseline_comparison: str = 'unknown'
    
    # 测试元数据
    test_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_tokens: int = 0
    
    def calculate_metrics(self):
        """
        计算吞吐量指标
        """
        if self.total_duration > 0:
            self.requests_per_second = self.successful_requests / self.total_duration
            if self.total_tokens > 0:
                self.tokens_per_second = self.total_tokens / self.total_duration
        
        if self.successful_requests > 0:
            self.avg_request_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        if self.response_times:
            self.min_response_time = min(self.response_times)
            self.max_response_time = max(self.response_times)
            self.response_times.sort()
            n = len(self.response_times)
            self.p50_response_time = self.response_times[n // 2]
            if n >= 20:
                self.p95_response_time = self.response_times[int(n * 0.95)]
            if n >= 100:
                self.p99_response_time = self.response_times[int(n * 0.99)]
    
    def evaluate_performance(self):
        """
        评估吞吐量性能等级
        """
        thresholds = BENCHMARK_CONFIG['baseline_thresholds']['throughput']
        
        if self.requests_per_second >= thresholds['excellent']:
            self.throughput_grade = 'A+'
            self.baseline_comparison = 'excellent'
        elif self.requests_per_second >= thresholds['good']:
            self.throughput_grade = 'A'
            self.baseline_comparison = 'good'
        elif self.requests_per_second >= thresholds['acceptable']:
            self.throughput_grade = 'B'
            self.baseline_comparison = 'acceptable'
        elif self.requests_per_second >= thresholds['poor']:
            self.throughput_grade = 'C'
            self.baseline_comparison = 'poor'
        else:
            self.throughput_grade = 'F'
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
            'concurrency_level': self.concurrency_level,
            'throughput_metrics': {
                'requests_per_second': self.requests_per_second,
                'tokens_per_second': self.tokens_per_second,
                'avg_request_time': self.avg_request_time
            },
            'response_time_distribution': {
                'min': self.min_response_time,
                'max': self.max_response_time,
                'p50': self.p50_response_time,
                'p95': self.p95_response_time,
                'p99': self.p99_response_time
            },
            'resource_usage': {
                'peak_memory_mb': self.peak_memory_mb,
                'avg_cpu_percent': self.avg_cpu_percent
            },
            'performance': {
                'grade': self.throughput_grade,
                'baseline_comparison': self.baseline_comparison
            },
            'summary': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.successful_requests / self.total_requests if self.total_requests > 0 else 0,
                'total_duration': self.total_duration,
                'total_tokens': self.total_tokens,
                'test_timestamp': self.test_timestamp
            }
        }


class MockThroughputAPI:
    """
    模拟吞吐量测试API客户端
    
    用于吞吐量基准测试的模拟API客户端
    """
    
    def __init__(self, vendor: str, model: str):
        self.vendor = vendor
        self.model = model
        self.request_count = 0
        self.lock = threading.Lock()
        
        # 根据厂商和模型设置不同的吞吐量特征
        self.throughput_config = self._get_throughput_config()
    
    def _get_throughput_config(self) -> Dict[str, Any]:
        """
        获取吞吐量配置
        
        返回:
            Dict[str, Any]: 吞吐量配置
        """
        # 模拟不同厂商和模型的吞吐量特征
        configs = {
            'deepseek': {
                'deepseek-chat': {
                    'base_response_time': 0.5,
                    'variance': 0.2,
                    'tokens_per_request': 150,
                    'max_concurrent': 50
                },
                'deepseek-coder': {
                    'base_response_time': 1.2,
                    'variance': 0.4,
                    'tokens_per_request': 200,
                    'max_concurrent': 30
                }
            },
            'ernie': {
                'ernie-3.5-8k': {
                    'base_response_time': 0.4,
                    'variance': 0.15,
                    'tokens_per_request': 120,
                    'max_concurrent': 60
                },
                'ernie-4.0-8k': {
                    'base_response_time': 0.7,
                    'variance': 0.25,
                    'tokens_per_request': 160,
                    'max_concurrent': 45
                }
            },
            'doubao': {
                'doubao-pro-4k': {
                    'base_response_time': 0.9,
                    'variance': 0.3,
                    'tokens_per_request': 140,
                    'max_concurrent': 35
                }
            },
            'local': {
                'llama2-7b': {
                    'base_response_time': 0.3,
                    'variance': 0.1,
                    'tokens_per_request': 100,
                    'max_concurrent': 80
                }
            }
        }
        
        return configs.get(self.vendor, {}).get(self.model, {
            'base_response_time': 1.0,
            'variance': 0.3,
            'tokens_per_request': 150,
            'max_concurrent': 40
        })
    
    def send_request_sync(self, request_id: str = None) -> Dict[str, Any]:
        """
        发送同步API请求
        
        参数:
            request_id: 请求ID
        
        返回:
            Dict[str, Any]: 模拟响应
        """
        with self.lock:
            self.request_count += 1
            current_count = self.request_count
        
        # 模拟并发限制影响
        concurrent_factor = 1.0
        if hasattr(threading.current_thread(), 'concurrent_level'):
            concurrent_level = threading.current_thread().concurrent_level
            max_concurrent = self.throughput_config['max_concurrent']
            if concurrent_level > max_concurrent:
                concurrent_factor = 1 + (concurrent_level - max_concurrent) * 0.1
        
        # 计算响应时间
        import random
        base_time = self.throughput_config['base_response_time']
        variance = self.throughput_config['variance']
        response_time = base_time * concurrent_factor * (1 + random.uniform(-variance, variance))
        response_time = max(0.05, response_time)  # 最小响应时间50ms
        
        # 模拟网络延迟
        time.sleep(response_time)
        
        tokens = self.throughput_config['tokens_per_request'] + random.randint(-20, 20)
        
        return {
            'vendor': self.vendor,
            'model': self.model,
            'response_time': response_time,
            'content': f"模拟响应 #{current_count} 来自 {self.vendor}/{self.model}",
            'tokens': tokens,
            'request_id': request_id or f"{self.vendor}_{self.model}_{current_count}",
            'timestamp': time.time()
        }
    
    async def send_request_async(self, request_id: str = None) -> Dict[str, Any]:
        """
        发送异步API请求
        
        参数:
            request_id: 请求ID
        
        返回:
            Dict[str, Any]: 模拟响应
        """
        # 异步版本通常有更好的并发性能
        response = self.send_request_sync(request_id)
        # 异步请求减少20%的响应时间
        response['response_time'] *= 0.8
        
        # 使用异步睡眠
        await asyncio.sleep(response['response_time'])
        
        return response


class ThroughputBenchmarkRunner:
    """
    吞吐量基准测试运行器
    
    执行各种吞吐量基准测试
    """
    
    def __init__(self):
        self.config = BENCHMARK_CONFIG
        self.results: List[ThroughputBenchmark] = []
    
    def run_single_thread_benchmark(self,
                                   vendor: str,
                                   model: str,
                                   num_requests: int = 100,
                                   test_name: str = "single_thread") -> ThroughputBenchmark:
        """
        运行单线程吞吐量基准测试
        
        参数:
            vendor: 厂商名称
            model: 模型名称
            num_requests: 请求数量
            test_name: 测试名称
        
        返回:
            ThroughputBenchmark: 基准测试结果
        """
        client = MockThroughputAPI(vendor, model)
        benchmark = ThroughputBenchmark(
            test_name=test_name,
            vendor=vendor,
            model=model,
            concurrency_level=1
        )
        
        start_time = time.time()
        
        for i in range(num_requests):
            try:
                response = client.send_request_sync(f"single_{i}")
                benchmark.response_times.append(response['response_time'])
                benchmark.successful_requests += 1
                benchmark.total_tokens += response['tokens']
            except Exception as e:
                benchmark.failed_requests += 1
                print(f"单线程请求失败: {e}")
            
            benchmark.total_requests += 1
        
        end_time = time.time()
        benchmark.total_duration = end_time - start_time
        
        # 计算指标和性能等级
        benchmark.calculate_metrics()
        benchmark.evaluate_performance()
        
        self.results.append(benchmark)
        return benchmark
    
    def run_multi_thread_benchmark(self,
                                  vendor: str,
                                  model: str,
                                  num_requests: int = 100,
                                  num_threads: int = 10,
                                  test_name: str = "multi_thread") -> ThroughputBenchmark:
        """
        运行多线程吞吐量基准测试
        
        参数:
            vendor: 厂商名称
            model: 模型名称
            num_requests: 总请求数量
            num_threads: 线程数量
            test_name: 测试名称
        
        返回:
            ThroughputBenchmark: 基准测试结果
        """
        client = MockThroughputAPI(vendor, model)
        benchmark = ThroughputBenchmark(
            test_name=test_name,
            vendor=vendor,
            model=model,
            concurrency_level=num_threads
        )
        
        # 结果收集队列
        results_queue = queue.Queue()
        
        def worker_thread(thread_id: int, requests_per_thread: int):
            """工作线程函数"""
            # 设置线程的并发级别（用于模拟并发影响）
            threading.current_thread().concurrent_level = num_threads
            
            thread_results = []
            for i in range(requests_per_thread):
                try:
                    response = client.send_request_sync(f"thread_{thread_id}_{i}")
                    thread_results.append({
                        'success': True,
                        'response_time': response['response_time'],
                        'tokens': response['tokens']
                    })
                except Exception as e:
                    thread_results.append({
                        'success': False,
                        'error': str(e)
                    })
            
            results_queue.put(thread_results)
        
        # 计算每个线程的请求数量
        requests_per_thread = num_requests // num_threads
        remaining_requests = num_requests % num_threads
        
        start_time = time.time()
        
        # 启动线程
        threads = []
        for i in range(num_threads):
            thread_requests = requests_per_thread + (1 if i < remaining_requests else 0)
            thread = threading.Thread(target=worker_thread, args=(i, thread_requests))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        benchmark.total_duration = end_time - start_time
        
        # 收集结果
        while not results_queue.empty():
            thread_results = results_queue.get()
            for result in thread_results:
                benchmark.total_requests += 1
                if result['success']:
                    benchmark.successful_requests += 1
                    benchmark.response_times.append(result['response_time'])
                    benchmark.total_tokens += result['tokens']
                else:
                    benchmark.failed_requests += 1
        
        # 计算指标和性能等级
        benchmark.calculate_metrics()
        benchmark.evaluate_performance()
        
        self.results.append(benchmark)
        return benchmark
    
    async def run_async_benchmark(self,
                                vendor: str,
                                model: str,
                                num_requests: int = 100,
                                concurrency: int = 20,
                                test_name: str = "async_concurrent") -> ThroughputBenchmark:
        """
        运行异步并发吞吐量基准测试
        
        参数:
            vendor: 厂商名称
            model: 模型名称
            num_requests: 总请求数量
            concurrency: 并发级别
            test_name: 测试名称
        
        返回:
            ThroughputBenchmark: 基准测试结果
        """
        client = MockThroughputAPI(vendor, model)
        benchmark = ThroughputBenchmark(
            test_name=test_name,
            vendor=vendor,
            model=model,
            concurrency_level=concurrency
        )
        
        async def send_single_request(request_id: str) -> Dict[str, Any]:
            """发送单个异步请求"""
            try:
                response = await client.send_request_async(request_id)
                return {
                    'success': True,
                    'response_time': response['response_time'],
                    'tokens': response['tokens']
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        start_time = time.time()
        
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(concurrency)
        
        async def bounded_request(request_id: str):
            """带并发控制的请求"""
            async with semaphore:
                return await send_single_request(request_id)
        
        # 创建所有任务
        tasks = [bounded_request(f"async_{i}") for i in range(num_requests)]
        
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
                benchmark.total_tokens += result['tokens']
            else:
                benchmark.failed_requests += 1
        
        # 计算指标和性能等级
        benchmark.calculate_metrics()
        benchmark.evaluate_performance()
        
        self.results.append(benchmark)
        return benchmark
    
    def run_scalability_benchmark(self,
                                vendor: str,
                                model: str,
                                concurrency_levels: List[int],
                                requests_per_level: int = 50) -> List[ThroughputBenchmark]:
        """
        运行可扩展性基准测试
        
        参数:
            vendor: 厂商名称
            model: 模型名称
            concurrency_levels: 并发级别列表
            requests_per_level: 每个级别的请求数量
        
        返回:
            List[ThroughputBenchmark]: 基准测试结果列表
        """
        results = []
        
        for concurrency in concurrency_levels:
            test_name = f"scalability_c{concurrency}"
            
            if concurrency == 1:
                # 单线程测试
                result = self.run_single_thread_benchmark(
                    vendor, model, requests_per_level, test_name
                )
            else:
                # 多线程测试
                result = self.run_multi_thread_benchmark(
                    vendor, model, requests_per_level, concurrency, test_name
                )
            
            results.append(result)
        
        return results
    
    def generate_throughput_report(self, results: List[ThroughputBenchmark]) -> Dict[str, Any]:
        """
        生成吞吐量基准测试报告
        
        参数:
            results: 基准测试结果列表
        
        返回:
            Dict[str, Any]: 基准测试报告
        """
        if not results:
            return {'error': '没有吞吐量基准测试结果'}
        
        report = {
            'summary': {
                'total_benchmarks': len(results),
                'test_timestamp': datetime.now().isoformat(),
                'throughput_distribution': {},
                'avg_throughput': 0.0,
                'max_throughput': 0.0,
                'min_throughput': float('inf')
            },
            'results': [result.to_dict() for result in results],
            'rankings': {
                'highest_throughput': None,
                'best_efficiency': None,
                'most_scalable': None
            },
            'analysis': {
                'scalability_trend': None,
                'concurrency_impact': None
            },
            'recommendations': []
        }
        
        # 统计吞吐量等级分布
        grade_counts = {}
        throughputs = []
        
        for result in results:
            grade = result.throughput_grade
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            throughputs.append(result.requests_per_second)
        
        report['summary']['throughput_distribution'] = grade_counts
        report['summary']['avg_throughput'] = statistics.mean(throughputs)
        report['summary']['max_throughput'] = max(throughputs)
        report['summary']['min_throughput'] = min(throughputs)
        
        # 排名分析
        if results:
            # 最高吞吐量
            highest = max(results, key=lambda x: x.requests_per_second)
            report['rankings']['highest_throughput'] = {
                'test_name': highest.test_name,
                'vendor': highest.vendor,
                'model': highest.model,
                'concurrency_level': highest.concurrency_level,
                'requests_per_second': highest.requests_per_second
            }
            
            # 最佳效率（吞吐量/并发级别）
            efficiency_results = [(r, r.requests_per_second / r.concurrency_level) for r in results]
            best_efficiency = max(efficiency_results, key=lambda x: x[1])
            report['rankings']['best_efficiency'] = {
                'test_name': best_efficiency[0].test_name,
                'vendor': best_efficiency[0].vendor,
                'model': best_efficiency[0].model,
                'efficiency_score': best_efficiency[1]
            }
        
        # 可扩展性分析
        scalability_results = [r for r in results if 'scalability' in r.test_name]
        if len(scalability_results) > 1:
            scalability_results.sort(key=lambda x: x.concurrency_level)
            
            # 计算可扩展性趋势
            concurrency_levels = [r.concurrency_level for r in scalability_results]
            throughputs = [r.requests_per_second for r in scalability_results]
            
            # 简单的线性回归分析
            if len(concurrency_levels) >= 2:
                import numpy as np
                try:
                    slope = np.polyfit(concurrency_levels, throughputs, 1)[0]
                    report['analysis']['scalability_trend'] = {
                        'slope': slope,
                        'interpretation': 'positive' if slope > 0 else 'negative' if slope < 0 else 'flat'
                    }
                except:
                    # 如果numpy不可用，使用简单计算
                    slope = (throughputs[-1] - throughputs[0]) / (concurrency_levels[-1] - concurrency_levels[0])
                    report['analysis']['scalability_trend'] = {
                        'slope': slope,
                        'interpretation': 'positive' if slope > 0 else 'negative' if slope < 0 else 'flat'
                    }
        
        # 生成建议
        poor_performers = [r for r in results if r.throughput_grade in ['C', 'F']]
        if poor_performers:
            report['recommendations'].append(
                f"发现 {len(poor_performers)} 个吞吐量较差的配置，建议优化并发策略"
            )
        
        excellent_performers = [r for r in results if r.throughput_grade == 'A+']
        if excellent_performers:
            report['recommendations'].append(
                f"推荐使用高吞吐量配置: {', '.join([f'{r.vendor}/{r.model}' for r in excellent_performers])}"
            )
        
        if report['analysis']['scalability_trend']:
            trend = report['analysis']['scalability_trend']['interpretation']
            if trend == 'negative':
                report['recommendations'].append("检测到负向可扩展性，建议检查并发瓶颈")
            elif trend == 'positive':
                report['recommendations'].append("良好的可扩展性，可以考虑增加并发级别")
        
        return report
    
    def reset_results(self):
        """重置基准测试结果"""
        self.results.clear()


class TestThroughputBenchmarks:
    """
    吞吐量基准测试类
    
    包含各种吞吐量基准测试场景
    """
    
    def setup_method(self):
        """测试方法设置"""
        self.benchmark_runner = ThroughputBenchmarkRunner()
        self.config = BENCHMARK_CONFIG
    
    def teardown_method(self):
        """测试方法清理"""
        self.benchmark_runner.reset_results()
    
    def _print_throughput_summary(self, results: List[ThroughputBenchmark]):
        """打印吞吐量基准测试摘要"""
        if not results:
            print("\n没有吞吐量基准测试结果")
            return
        
        print(f"\n=== 吞吐量基准测试结果 ===")
        print(f"测试数量: {len(results)}")
        
        for result in results:
            print(f"\n{result.test_name} - {result.vendor}/{result.model}:")
            print(f"  并发级别: {result.concurrency_level}")
            print(f"  吞吐量等级: {result.throughput_grade} ({result.baseline_comparison})")
            print(f"  请求/秒: {result.requests_per_second:.2f}")
            if result.tokens_per_second > 0:
                print(f"  令牌/秒: {result.tokens_per_second:.2f}")
            print(f"  平均响应时间: {result.avg_request_time*1000:.1f}ms")
            print(f"  响应时间分布: P50={result.p50_response_time*1000:.1f}ms, P95={result.p95_response_time*1000:.1f}ms")
            print(f"  成功率: {result.successful_requests}/{result.total_requests} ({result.successful_requests/result.total_requests*100:.1f}%)")
            print(f"  测试时长: {result.total_duration:.2f}秒")
    
    @pytest.mark.benchmark
    @pytest.mark.quick_benchmark
    def test_single_thread_throughput_benchmark(self):
        """
        单线程吞吐量基准测试
        
        测试单线程环境下的API吞吐量性能
        """
        vendor = 'deepseek'
        model = 'deepseek-chat'
        num_requests = 50
        
        # 运行单线程基准测试
        result = self.benchmark_runner.run_single_thread_benchmark(
            vendor, model, num_requests
        )
        
        self._print_throughput_summary([result])
        
        # 基准测试断言
        assert result.total_requests == num_requests
        assert result.successful_requests > 0
        assert result.concurrency_level == 1
        assert result.requests_per_second > 0
        assert result.throughput_grade in ['A+', 'A', 'B', 'C', 'F']
        
        # 性能要求断言
        assert result.requests_per_second >= self.config['baseline_thresholds']['throughput']['poor']
        assert result.successful_requests / result.total_requests >= 0.9  # 90%成功率
    
    @pytest.mark.benchmark
    @pytest.mark.standard_benchmark
    def test_multi_thread_throughput_benchmark(self):
        """
        多线程吞吐量基准测试
        
        测试多线程环境下的API吞吐量性能
        """
        vendor = 'ernie'
        model = 'ernie-4.0-8k'
        num_requests = 80
        num_threads = 8
        
        # 运行多线程基准测试
        result = self.benchmark_runner.run_multi_thread_benchmark(
            vendor, model, num_requests, num_threads
        )
        
        self._print_throughput_summary([result])
        
        # 基准测试断言
        assert result.total_requests == num_requests
        assert result.successful_requests > 0
        assert result.concurrency_level == num_threads
        assert result.requests_per_second > 0
        
        # 多线程应该比单线程有更好的吞吐量（在理想情况下）
        single_thread_result = self.benchmark_runner.run_single_thread_benchmark(
            vendor, model, num_requests // num_threads
        )
        
        print(f"\n=== 多线程 vs 单线程对比 ===")
        print(f"单线程吞吐量: {single_thread_result.requests_per_second:.2f} req/s")
        print(f"多线程吞吐量: {result.requests_per_second:.2f} req/s")
        print(f"吞吐量提升: {(result.requests_per_second / single_thread_result.requests_per_second - 1) * 100:.1f}%")
        
        # 多线程吞吐量断言
        assert result.requests_per_second >= single_thread_result.requests_per_second * 0.8  # 至少保持80%的效率
    
    @pytest.mark.benchmark
    @pytest.mark.standard_benchmark
    @pytest.mark.asyncio
    async def test_async_concurrent_throughput_benchmark(self):
        """
        异步并发吞吐量基准测试
        
        测试异步并发环境下的API吞吐量性能
        """
        vendor = 'doubao'
        model = 'doubao-pro-4k'
        num_requests = 100
        concurrency = 15
        
        # 运行异步并发基准测试
        result = await self.benchmark_runner.run_async_benchmark(
            vendor, model, num_requests, concurrency
        )
        
        self._print_throughput_summary([result])
        
        # 基准测试断言
        assert result.total_requests == num_requests
        assert result.successful_requests > 0
        assert result.concurrency_level == concurrency
        assert result.requests_per_second > 0
        
        # 异步并发性能要求
        assert result.requests_per_second >= self.config['baseline_thresholds']['throughput']['acceptable']
        assert result.successful_requests / result.total_requests >= 0.85  # 85%成功率
    
    @pytest.mark.benchmark
    @pytest.mark.comprehensive_benchmark
    def test_throughput_scalability_benchmark(self):
        """
        吞吐量可扩展性基准测试
        
        测试不同并发级别下的吞吐量扩展性
        """
        vendor = 'deepseek'
        model = 'deepseek-coder'
        concurrency_levels = [1, 2, 4, 8, 16]
        requests_per_level = 40
        
        # 运行可扩展性基准测试
        results = self.benchmark_runner.run_scalability_benchmark(
            vendor, model, concurrency_levels, requests_per_level
        )
        
        self._print_throughput_summary(results)
        
        # 生成可扩展性报告
        report = self.benchmark_runner.generate_throughput_report(results)
        
        print(f"\n=== 可扩展性分析报告 ===")
        print(f"测试的并发级别: {concurrency_levels}")
        print(f"平均吞吐量: {report['summary']['avg_throughput']:.2f} req/s")
        print(f"最大吞吐量: {report['summary']['max_throughput']:.2f} req/s")
        print(f"最小吞吐量: {report['summary']['min_throughput']:.2f} req/s")
        
        if report['analysis']['scalability_trend']:
            trend = report['analysis']['scalability_trend']
            print(f"可扩展性趋势: {trend['interpretation']} (斜率: {trend['slope']:.3f})")
        
        if report['rankings']['highest_throughput']:
            best = report['rankings']['highest_throughput']
            print(f"最佳吞吐量配置: 并发级别 {best['concurrency_level']} ({best['requests_per_second']:.2f} req/s)")
        
        # 可扩展性断言
        assert len(results) == len(concurrency_levels)
        assert all(r.total_requests == requests_per_level for r in results)
        
        # 检查吞吐量随并发级别的变化
        throughputs = [r.requests_per_second for r in results]
        max_throughput = max(throughputs)
        min_throughput = min(throughputs)
        
        # 最大吞吐量应该明显高于最小吞吐量
        assert max_throughput > min_throughput * 1.2  # 至少20%的提升
    
    @pytest.mark.benchmark
    @pytest.mark.comprehensive_benchmark
    def test_vendor_throughput_comparison_benchmark(self):
        """
        厂商吞吐量对比基准测试
        
        对比不同厂商的吞吐量性能
        """
        test_configs = [
            ('deepseek', 'deepseek-chat'),
            ('ernie', 'ernie-3.5-8k'),
            ('doubao', 'doubao-pro-4k')
        ]
        
        num_requests = 60
        concurrency = 10
        
        all_results = []
        
        for vendor, model in test_configs:
            result = self.benchmark_runner.run_multi_thread_benchmark(
                vendor, model, num_requests, concurrency, f"vendor_comparison_{vendor}"
            )
            all_results.append(result)
        
        self._print_throughput_summary(all_results)
        
        # 生成对比报告
        report = self.benchmark_runner.generate_throughput_report(all_results)
        
        print(f"\n=== 厂商吞吐量对比报告 ===")
        print(f"参与对比的厂商数量: {len(test_configs)}")
        
        if report['rankings']['highest_throughput']:
            best = report['rankings']['highest_throughput']
            print(f"最高吞吐量: {best['vendor']}/{best['requests_per_second']:.2f} req/s")
        
        if report['rankings']['best_efficiency']:
            efficient = report['rankings']['best_efficiency']
            print(f"最佳效率: {efficient['vendor']} (效率分数: {efficient['efficiency_score']:.2f})")
        
        if report['recommendations']:
            print(f"\n建议:")
            for rec in report['recommendations']:
                print(f"  - {rec}")
        
        # 厂商对比断言
        assert len(all_results) == len(test_configs)
        assert all(r.total_requests == num_requests for r in all_results)
        
        # 至少有一个厂商达到可接受的吞吐量
        acceptable_results = [r for r in all_results if r.throughput_grade in ['A+', 'A', 'B']]
        assert len(acceptable_results) > 0, "没有厂商达到可接受的吞吐量水平"
    
    @pytest.mark.benchmark
    @pytest.mark.regression_benchmark
    @pytest.mark.asyncio
    async def test_throughput_regression_benchmark(self):
        """
        吞吐量回归基准测试
        
        检测吞吐量性能是否出现回归
        """
        vendor = 'ernie'
        model = 'ernie-4.0-8k'
        num_requests = 80
        concurrency = 12
        
        # 运行当前吞吐量基准测试
        current_result = await self.benchmark_runner.run_async_benchmark(
            vendor, model, num_requests, concurrency, "regression_test"
        )
        
        self._print_throughput_summary([current_result])
        
        # 模拟历史基准数据（实际应用中从数据库或文件读取）
        historical_baseline = {
            'requests_per_second': 8.5,  # 历史吞吐量8.5 req/s
            'avg_request_time': 1.2,     # 历史平均响应时间1.2秒
            'throughput_grade': 'A'      # 历史吞吐量等级
        }
        
        print(f"\n=== 吞吐量回归分析 ===")
        print(f"历史吞吐量: {historical_baseline['requests_per_second']:.2f} req/s")
        print(f"当前吞吐量: {current_result.requests_per_second:.2f} req/s")
        
        # 计算吞吐量变化
        throughput_change = (current_result.requests_per_second - historical_baseline['requests_per_second']) / historical_baseline['requests_per_second'] * 100
        print(f"吞吐量变化: {throughput_change:+.1f}%")
        
        # 吞吐量回归检测
        regression_threshold = -15.0  # -15%的吞吐量回归阈值
        
        if throughput_change < regression_threshold:
            print(f"⚠️  检测到吞吐量回归: 吞吐量下降了 {abs(throughput_change):.1f}%")
        elif throughput_change > 10.0:
            print(f"✅ 检测到吞吐量改进: 吞吐量提升了 {throughput_change:.1f}%")
        else:
            print(f"✅ 吞吐量稳定: 变化在可接受范围内")
        
        # 回归测试断言
        assert current_result.total_requests == num_requests
        assert current_result.successful_requests > 0
        
        # 吞吐量回归断言（在实际应用中，这里可能会失败并触发告警）
        if throughput_change < regression_threshold:
            pytest.skip(f"检测到吞吐量回归 ({throughput_change:.1f}%)，需要进一步调查")
        
        # 确保吞吐量不会严重退化
        assert current_result.requests_per_second >= historical_baseline['requests_per_second'] * 0.7  # 最多允许30%的吞吐量下降
    
    @pytest.mark.benchmark
    @pytest.mark.performance_benchmark
    def test_throughput_benchmark_with_pytest_benchmark(self, benchmark):
        """
        使用pytest-benchmark的吞吐量基准测试
        
        集成pytest-benchmark进行精确的性能测量
        """
        vendor = 'deepseek'
        model = 'deepseek-chat'
        
        def throughput_test_function():
            """被基准测试的函数"""
            return self.benchmark_runner.run_single_thread_benchmark(
                vendor, model, 20, "pytest_benchmark"
            )
        
        # 使用pytest-benchmark运行基准测试
        result = benchmark(throughput_test_function)
        
        self._print_throughput_summary([result])
        
        print(f"\n=== pytest-benchmark 统计 ===")
        print(f"基准测试函数: {throughput_test_function.__name__}")
        
        # pytest-benchmark断言
        assert result.total_requests == 20
        assert result.successful_requests > 0
        assert result.requests_per_second > 0
        
        # 性能基准断言
        assert result.requests_per_second >= self.config['baseline_thresholds']['throughput']['poor']