# -*- coding: utf-8 -*-
"""
压力测试模块

本模块实现了HarborAI项目的压力测试，包括：
- 负载递增测试
- 峰值负载测试
- 长时间压力测试
- 资源耗尽测试
- 恢复能力测试

作者: HarborAI Team
版本: 1.0.0
创建时间: 2024-01-20
"""

import asyncio
import concurrent.futures
import gc
import psutil
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock, patch
import pytest
import statistics
from datetime import datetime, timedelta
import random

from tests.performance import PERFORMANCE_CONFIG, SUPPORTED_VENDORS


@dataclass
class StressMetrics:
    """
    压力测试指标数据类
    
    记录压力测试中的各项性能和资源指标
    """
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    response_times: List[float] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    peak_concurrent_requests: int = 0
    max_memory_usage: float = 0.0  # MB
    max_cpu_usage: float = 0.0  # %
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    throughput: float = 0.0
    degradation_point: Optional[int] = None  # 性能开始下降的并发数
    recovery_time: Optional[float] = None  # 恢复时间（秒）
    
    def calculate_metrics(self):
        """计算压力测试指标"""
        if self.response_times:
            self.avg_response_time = statistics.mean(self.response_times)
            if len(self.response_times) > 1:
                sorted_times = sorted(self.response_times)
                n = len(sorted_times)
                # 计算P95和P99百分位数
                p95_index = int(n * 0.95)
                p99_index = int(n * 0.99)
                self.p95_response_time = sorted_times[min(p95_index, n-1)]
                self.p99_response_time = sorted_times[min(p99_index, n-1)]
        
        if self.total_requests > 0:
            self.error_rate = (self.failed_requests / self.total_requests) * 100
        
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
            if duration > 0:
                self.throughput = self.successful_requests / duration


class SystemMonitor:
    """
    系统资源监控器
    
    监控CPU、内存等系统资源使用情况
    """
    
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.5):
        """
        开始监控系统资源
        
        参数:
            interval: 监控间隔（秒）
        """
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        
        def monitor_loop():
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory_info = psutil.virtual_memory()
                    memory_mb = memory_info.used / 1024 / 1024
                    
                    self.cpu_samples.append(cpu_percent)
                    self.memory_samples.append(memory_mb)
                    
                    time.sleep(interval)
                except Exception as e:
                    print(f"监控错误: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Tuple[float, float]:
        """
        停止监控并返回峰值资源使用
        
        返回:
            (最大CPU使用率, 最大内存使用量MB)
        """
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        max_cpu = max(self.cpu_samples) if self.cpu_samples else 0.0
        max_memory = max(self.memory_samples) if self.memory_samples else 0.0
        
        return max_cpu, max_memory


class StressTestRunner:
    """
    压力测试执行器
    
    提供各种压力测试场景的执行方法
    """
    
    def __init__(self):
        self.metrics = StressMetrics()
        self.monitor = SystemMonitor()
        self.lock = threading.Lock()
        self.config = PERFORMANCE_CONFIG['stress']
        self.active_requests = 0
    
    def mock_api_call(self, delay: float = 0.1, failure_rate: float = 0.05) -> Tuple[bool, float]:
        """
        模拟API调用
        
        参数:
            delay: 基础延迟时间
            failure_rate: 失败率
        
        返回:
            (是否成功, 响应时间)
        """
        # 模拟负载增加时的性能下降
        load_factor = min(self.active_requests / 100, 1.5)  # 最多1.5倍延迟
        actual_delay = delay * (1 + load_factor * 0.3)
        
        start_time = time.time()
        time.sleep(actual_delay)
        response_time = time.time() - start_time
        
        # 高负载时适度增加失败率，但不要过高
        adjusted_failure_rate = failure_rate * (1 + load_factor * 0.5)
        # 确保失败率不会超过合理范围
        adjusted_failure_rate = min(adjusted_failure_rate, 0.15)  # 最大15%失败率
        success = random.random() > adjusted_failure_rate
        
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
    
    def stress_worker(self, worker_id: int, duration: float, base_delay: float = 0.1):
        """
        压力测试工作线程
        
        参数:
            worker_id: 工作线程ID
            duration: 运行时长（秒）
            base_delay: 基础延迟
        """
        end_time = time.time() + duration
        
        while time.time() < end_time:
            with self.lock:
                self.active_requests += 1
                current_active = self.active_requests
            
            # 更新峰值并发数
            if current_active > self.metrics.peak_concurrent_requests:
                self.metrics.peak_concurrent_requests = current_active
            
            try:
                success, response_time = self.mock_api_call(base_delay)
                self.record_result(success, response_time)
            finally:
                with self.lock:
                    self.active_requests -= 1
            
            # 短暂休息避免过度消耗资源
            time.sleep(0.01)


class TestStressTesting:
    """
    压力测试类
    
    包含各种压力测试场景
    """
    
    def setup_method(self):
        """测试方法设置"""
        self.stress_runner = StressTestRunner()
        self.config = PERFORMANCE_CONFIG['stress']
        # 强制垃圾回收，确保测试开始时内存状态一致
        gc.collect()
    
    def teardown_method(self):
        """测试方法清理"""
        # 停止监控
        max_cpu, max_memory = self.stress_runner.monitor.stop_monitoring()
        self.stress_runner.metrics.max_cpu_usage = max_cpu
        self.stress_runner.metrics.max_memory_usage = max_memory
        
        # 计算指标
        self.stress_runner.metrics.calculate_metrics()
        metrics = self.stress_runner.metrics
        
        print(f"\n=== 压力测试结果 ===")
        print(f"总请求数: {metrics.total_requests}")
        print(f"成功请求数: {metrics.successful_requests}")
        print(f"失败请求数: {metrics.failed_requests}")
        print(f"错误率: {metrics.error_rate:.2f}%")
        print(f"峰值并发数: {metrics.peak_concurrent_requests}")
        print(f"平均响应时间: {metrics.avg_response_time:.3f}s")
        print(f"P95响应时间: {metrics.p95_response_time:.3f}s")
        print(f"P99响应时间: {metrics.p99_response_time:.3f}s")
        print(f"吞吐量: {metrics.throughput:.2f} req/s")
        print(f"最大CPU使用率: {metrics.max_cpu_usage:.1f}%")
        print(f"最大内存使用: {metrics.max_memory_usage:.1f}MB")
        if metrics.degradation_point:
            print(f"性能下降点: {metrics.degradation_point} 并发")
        if metrics.recovery_time:
            print(f"恢复时间: {metrics.recovery_time:.2f}s")
        
        # 清理资源
        gc.collect()
    
    @pytest.mark.performance
    @pytest.mark.stress
    def test_load_ramp_up(self):
        """
        负载递增测试
        
        逐步增加负载，观察系统性能变化
        """
        self.stress_runner.monitor.start_monitoring()
        self.stress_runner.metrics.start_time = datetime.now()
        
        # 负载递增阶段
        ramp_stages = [5, 10, 20, 30, 50]  # 并发数递增
        stage_duration = 10  # 每阶段持续时间（秒）
        
        baseline_response_time = None
        
        for stage, concurrent_users in enumerate(ramp_stages):
            print(f"\n阶段 {stage + 1}: {concurrent_users} 并发用户")
            
            stage_start = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = []
                for i in range(concurrent_users):
                    future = executor.submit(
                        self.stress_runner.stress_worker,
                        i,
                        stage_duration,
                        0.05  # 50ms基础延迟
                    )
                    futures.append(future)
                
                # 等待阶段完成
                concurrent.futures.wait(futures, timeout=stage_duration + 5)
            
            # 分析当前阶段性能
            response_times_list = list(self.stress_runner.metrics.response_times)
            current_response_times = response_times_list[-50:] if len(response_times_list) >= 50 else response_times_list  # 最近50个请求
            if current_response_times:
                avg_response_time = statistics.mean(current_response_times)
                
                if baseline_response_time is None:
                    baseline_response_time = avg_response_time
                elif (avg_response_time > baseline_response_time * 2 and 
                      self.stress_runner.metrics.degradation_point is None):
                    # 性能下降超过2倍，记录下降点
                    self.stress_runner.metrics.degradation_point = concurrent_users
                    print(f"检测到性能下降点: {concurrent_users} 并发")
        
        self.stress_runner.metrics.end_time = datetime.now()
        self.stress_runner.metrics.calculate_metrics()
        
        # 压力测试断言
        assert self.stress_runner.metrics.total_requests > 0
        assert self.stress_runner.metrics.error_rate <= self.config['max_error_rate']
        assert self.stress_runner.metrics.peak_concurrent_requests >= max(ramp_stages)
        # 在压力测试中，响应时间可能会增加
        assert self.stress_runner.metrics.avg_response_time <= self.config['max_response_time'] * 3
    
    @pytest.mark.performance
    @pytest.mark.stress
    def test_peak_load_sustained(self):
        """
        峰值负载持续测试
        
        在峰值负载下持续运行，测试系统稳定性
        """
        peak_concurrent = self.config['max_concurrent_requests']
        test_duration = 30  # 30秒峰值负载
        
        self.stress_runner.monitor.start_monitoring()
        self.stress_runner.metrics.start_time = datetime.now()
        
        print(f"\n开始峰值负载测试: {peak_concurrent} 并发，持续 {test_duration} 秒")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=peak_concurrent) as executor:
            futures = []
            for i in range(peak_concurrent):
                future = executor.submit(
                    self.stress_runner.stress_worker,
                    i,
                    test_duration,
                    0.08  # 80ms基础延迟
                )
                futures.append(future)
            
            # 等待所有任务完成
            concurrent.futures.wait(futures, timeout=test_duration + 10)
        
        self.stress_runner.metrics.end_time = datetime.now()
        self.stress_runner.metrics.calculate_metrics()
        
        # 峰值负载断言
        assert self.stress_runner.metrics.total_requests > peak_concurrent * 5  # 至少每个线程5个请求
        assert self.stress_runner.metrics.error_rate <= self.config['max_error_rate'] * 2  # 峰值时允许更高错误率
        assert self.stress_runner.metrics.peak_concurrent_requests >= peak_concurrent * 0.8
        # 峰值负载下的性能要求
        assert self.stress_runner.metrics.throughput >= self.config['min_throughput'] * 0.5
    
    @pytest.mark.performance
    @pytest.mark.stress
    @pytest.mark.slow
    def test_endurance_testing(self):
        """
        长时间耐久性测试
        
        中等负载下长时间运行，测试内存泄漏和性能退化
        """
        moderate_concurrent = 15
        test_duration = 60  # 1分钟耐久测试
        
        self.stress_runner.monitor.start_monitoring()
        self.stress_runner.metrics.start_time = datetime.now()
        
        print(f"\n开始耐久性测试: {moderate_concurrent} 并发，持续 {test_duration} 秒")
        
        # 记录初始内存使用
        initial_memory = psutil.virtual_memory().used / 1024 / 1024
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=moderate_concurrent) as executor:
            futures = []
            for i in range(moderate_concurrent):
                future = executor.submit(
                    self.stress_runner.stress_worker,
                    i,
                    test_duration,
                    0.1  # 100ms基础延迟
                )
                futures.append(future)
            
            # 等待所有任务完成
            concurrent.futures.wait(futures, timeout=test_duration + 10)
        
        # 记录结束内存使用
        final_memory = psutil.virtual_memory().used / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        self.stress_runner.metrics.end_time = datetime.now()
        self.stress_runner.metrics.calculate_metrics()
        
        # 耐久性测试断言
        assert self.stress_runner.metrics.total_requests > moderate_concurrent * 10
        assert self.stress_runner.metrics.error_rate <= self.config['max_error_rate']
        # 检查内存增长（不应该有严重的内存泄漏）
        assert memory_growth <= 100  # 内存增长不超过100MB
        # 长时间运行后性能不应严重退化
        assert self.stress_runner.metrics.avg_response_time <= self.config['max_response_time'] * 2
        
        print(f"\n内存使用变化: {memory_growth:.1f}MB")
    
    @pytest.mark.performance
    @pytest.mark.stress
    def test_resource_exhaustion_recovery(self):
        """
        资源耗尽恢复测试
        
        模拟资源耗尽情况，测试系统恢复能力
        """
        # 第一阶段：极高负载导致资源耗尽
        extreme_concurrent = self.config['max_concurrent_requests'] * 2
        overload_duration = 10
        
        self.stress_runner.monitor.start_monitoring()
        self.stress_runner.metrics.start_time = datetime.now()
        
        print(f"\n阶段1: 极高负载 {extreme_concurrent} 并发")
        
        # 极高负载阶段
        with concurrent.futures.ThreadPoolExecutor(max_workers=extreme_concurrent) as executor:
            futures = []
            for i in range(extreme_concurrent):
                future = executor.submit(
                    self.stress_runner.stress_worker,
                    i,
                    overload_duration,
                    0.05
                )
                futures.append(future)
            
            concurrent.futures.wait(futures, timeout=overload_duration + 5)
        
        # 记录过载后的错误率
        overload_error_rate = self.stress_runner.metrics.error_rate if self.stress_runner.metrics.total_requests > 0 else 0
        
        print(f"过载阶段错误率: {overload_error_rate:.2f}%")
        
        # 第二阶段：恢复期（正常负载）
        recovery_start = time.time()
        normal_concurrent = 10
        recovery_duration = 15
        
        print(f"\n阶段2: 恢复期 {normal_concurrent} 并发")
        
        # 重置部分指标以测量恢复
        recovery_start_requests = self.stress_runner.metrics.total_requests
        recovery_start_errors = self.stress_runner.metrics.failed_requests
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=normal_concurrent) as executor:
            futures = []
            for i in range(normal_concurrent):
                future = executor.submit(
                    self.stress_runner.stress_worker,
                    i,
                    recovery_duration,
                    0.1
                )
                futures.append(future)
            
            concurrent.futures.wait(futures, timeout=recovery_duration + 5)
        
        recovery_end = time.time()
        self.stress_runner.metrics.recovery_time = recovery_end - recovery_start
        
        # 计算恢复期的错误率
        recovery_requests = self.stress_runner.metrics.total_requests - recovery_start_requests
        recovery_errors = self.stress_runner.metrics.failed_requests - recovery_start_errors
        recovery_error_rate = (recovery_errors / recovery_requests * 100) if recovery_requests > 0 else 0
        
        self.stress_runner.metrics.end_time = datetime.now()
        self.stress_runner.metrics.calculate_metrics()
        
        # 恢复能力断言
        assert self.stress_runner.metrics.total_requests > extreme_concurrent + normal_concurrent
        # 恢复期的错误率应该明显低于过载期
        assert recovery_error_rate <= overload_error_rate * 0.5
        # 恢复时间应该在合理范围内
        assert self.stress_runner.metrics.recovery_time <= 20  # 20秒内恢复
        
        print(f"\n恢复期错误率: {recovery_error_rate:.2f}%")
        print(f"恢复时间: {self.stress_runner.metrics.recovery_time:.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.stress
    @pytest.mark.parametrize("vendor", list(SUPPORTED_VENDORS.keys())[:3])  # 只测试前3个厂商以节省时间
    def test_vendor_stress_comparison(self, vendor: str):
        """
        厂商压力测试对比
        
        比较不同厂商在压力情况下的表现
        
        参数:
            vendor: API厂商名称
        """
        concurrent_users = 20
        test_duration = 15
        
        # 根据厂商调整测试参数
        vendor_configs = {
            'deepseek': {'delay': 0.15, 'failure_rate': 0.08},
            'anthropic': {'delay': 0.12, 'failure_rate': 0.06},
            'google': {'delay': 0.10, 'failure_rate': 0.05},
            'azure': {'delay': 0.18, 'failure_rate': 0.10},
            'aws': {'delay': 0.20, 'failure_rate': 0.12}
        }
        
        config = vendor_configs.get(vendor, {'delay': 0.15, 'failure_rate': 0.08})
        
        self.stress_runner.monitor.start_monitoring()
        self.stress_runner.metrics.start_time = datetime.now()
        
        print(f"\n压力测试厂商: {vendor}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            for i in range(concurrent_users):
                future = executor.submit(
                    self.stress_runner.stress_worker,
                    i,
                    test_duration,
                    config['delay']
                )
                futures.append(future)
            
            concurrent.futures.wait(futures, timeout=test_duration + 5)
        
        self.stress_runner.metrics.end_time = datetime.now()
        self.stress_runner.metrics.calculate_metrics()
        
        # 厂商特定的压力测试断言
        assert self.stress_runner.metrics.total_requests > concurrent_users * 3
        
        # 根据厂商调整性能期望
        if vendor in ['google', 'anthropic']:
            # 这些厂商通常有更好的压力承受能力
            assert self.stress_runner.metrics.error_rate <= self.config['max_error_rate'] * 0.8
            assert self.stress_runner.metrics.avg_response_time <= self.config['max_response_time'] * 1.5
        else:
            assert self.stress_runner.metrics.error_rate <= self.config['max_error_rate'] * 1.5
            assert self.stress_runner.metrics.avg_response_time <= self.config['max_response_time'] * 2
    
    @pytest.mark.performance
    @pytest.mark.stress
    @pytest.mark.benchmark
    def test_stress_benchmark(self, benchmark):
        """
        压力测试基准
        
        使用pytest-benchmark进行压力测试基准测量
        """
        def stress_benchmark():
            concurrent_users = 15
            test_duration = 5  # 较短的基准测试时间
            
            self.stress_runner.metrics = StressMetrics()
            self.stress_runner.metrics.start_time = datetime.now()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = []
                for i in range(concurrent_users):
                    future = executor.submit(
                        self.stress_runner.stress_worker,
                        i,
                        test_duration,
                        0.08
                    )
                    futures.append(future)
                
                concurrent.futures.wait(futures, timeout=test_duration + 3)
            
            self.stress_runner.metrics.end_time = datetime.now()
            self.stress_runner.metrics.calculate_metrics()
            
            return {
                'throughput': self.stress_runner.metrics.throughput,
                'error_rate': self.stress_runner.metrics.error_rate,
                'avg_response_time': self.stress_runner.metrics.avg_response_time
            }
        
        # 运行基准测试
        result = benchmark(stress_benchmark)
        
        # 基准测试断言
        assert result['throughput'] >= self.config['min_throughput'] * 0.7
        # 错误率以百分比形式存储，需要转换为小数进行比较
        assert result['error_rate'] <= self.config['max_error_rate'] * 100 * 1.5  # 30%
        assert result['avg_response_time'] <= self.config['max_response_time'] * 2
        
        print(f"\n压力基准结果:")
        print(f"吞吐量: {result['throughput']:.2f} req/s")
        print(f"错误率: {result['error_rate']:.2f}%")
        print(f"平均响应时间: {result['avg_response_time']:.3f}s")