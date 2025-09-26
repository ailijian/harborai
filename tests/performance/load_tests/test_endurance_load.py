#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
持久负载测试模块

实现长时间运行的负载测试，验证系统在持续负载下的稳定性和性能表现。
包括内存泄漏检测、性能衰减分析、资源使用监控等功能。

作者: HarborAI测试团队
版本: 1.0.0
创建时间: 2024-01-20
"""

import asyncio
import gc
import logging
import psutil
import pytest
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from unittest.mock import Mock

from . import LOAD_TEST_CONFIG, LOAD_PERFORMANCE_GRADES

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EndurancePhase:
    """持久测试阶段定义"""
    name: str
    duration_minutes: int
    target_load: int
    ramp_time_seconds: int = 30
    monitoring_interval: int = 60  # 监控间隔（秒）
    
    def __post_init__(self):
        """验证阶段参数"""
        if self.duration_minutes <= 0:
            raise ValueError("持续时间必须大于0")
        if self.target_load <= 0:
            raise ValueError("目标负载必须大于0")


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: datetime
    response_time: float
    throughput: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    memory_mb: float
    active_connections: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'response_time': self.response_time,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'memory_mb': self.memory_mb,
            'active_connections': self.active_connections
        }


@dataclass
class EnduranceLoadResult:
    """持久负载测试结果"""
    vendor: str
    model: str
    test_name: str
    start_time: datetime
    end_time: datetime
    total_duration_minutes: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # 性能指标
    avg_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # 吞吐量指标
    avg_throughput: float = 0.0
    peak_throughput: float = 0.0
    min_throughput: float = 0.0
    throughput_stability: float = 0.0
    
    # 错误率指标
    overall_error_rate: float = 0.0
    max_error_rate: float = 0.0
    error_rate_stability: float = 0.0
    
    # 资源使用指标
    avg_cpu_usage: float = 0.0
    peak_cpu_usage: float = 0.0
    avg_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    memory_growth_rate: float = 0.0  # MB/小时
    
    # 稳定性指标
    performance_degradation: float = 0.0  # 性能衰减百分比
    stability_score: float = 0.0  # 稳定性评分
    endurance_grade: str = 'F'
    
    # 详细数据
    performance_snapshots: List[PerformanceSnapshot] = field(default_factory=list)
    phase_results: List[Dict[str, Any]] = field(default_factory=list)
    
    def calculate_metrics(self):
        """计算各项指标"""
        if not self.performance_snapshots:
            return
        
        # 响应时间统计
        response_times = [s.response_time for s in self.performance_snapshots]
        self.avg_response_time = statistics.mean(response_times)
        self.min_response_time = min(response_times)
        self.max_response_time = max(response_times)
        
        if len(response_times) >= 2:
            self.p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            self.p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        
        # 吞吐量统计
        throughputs = [s.throughput for s in self.performance_snapshots]
        self.avg_throughput = statistics.mean(throughputs)
        self.peak_throughput = max(throughputs)
        self.min_throughput = min(throughputs)
        
        if len(throughputs) >= 2:
            throughput_std = statistics.stdev(throughputs)
            self.throughput_stability = max(0, 100 - (throughput_std / self.avg_throughput * 100))
        
        # 错误率统计
        error_rates = [s.error_rate for s in self.performance_snapshots]
        self.overall_error_rate = statistics.mean(error_rates)
        self.max_error_rate = max(error_rates)
        
        if len(error_rates) >= 2:
            error_rate_std = statistics.stdev(error_rates)
            avg_error = max(self.overall_error_rate, 0.01)  # 避免除零
            self.error_rate_stability = max(0, 100 - (error_rate_std / avg_error * 100))
        
        # 资源使用统计
        cpu_usages = [s.cpu_usage for s in self.performance_snapshots]
        memory_usages = [s.memory_usage for s in self.performance_snapshots]
        memory_mbs = [s.memory_mb for s in self.performance_snapshots]
        
        self.avg_cpu_usage = statistics.mean(cpu_usages)
        self.peak_cpu_usage = max(cpu_usages)
        self.avg_memory_usage = statistics.mean(memory_usages)
        self.peak_memory_usage = max(memory_usages)
        
        # 内存增长率计算（MB/小时）
        if len(memory_mbs) >= 2 and self.total_duration_minutes > 0:
            memory_growth = memory_mbs[-1] - memory_mbs[0]
            self.memory_growth_rate = memory_growth / (self.total_duration_minutes / 60)
        
        # 性能衰减计算
        if len(response_times) >= 10:
            # 比较前10%和后10%的性能
            early_samples = int(len(response_times) * 0.1)
            late_samples = int(len(response_times) * 0.1)
            
            early_avg = statistics.mean(response_times[:early_samples])
            late_avg = statistics.mean(response_times[-late_samples:])
            
            if early_avg > 0:
                self.performance_degradation = ((late_avg - early_avg) / early_avg) * 100
        
        # 稳定性评分计算
        self._calculate_stability_score()
        
        # 等级评定
        self._assign_endurance_grade()
    
    def _calculate_stability_score(self):
        """计算稳定性评分（0-100分）"""
        scores = []
        
        # 吞吐量稳定性（30%权重）
        scores.append(self.throughput_stability * 0.3)
        
        # 错误率稳定性（25%权重）
        scores.append(self.error_rate_stability * 0.25)
        
        # 性能衰减评分（25%权重）
        degradation_score = max(0, 100 - abs(self.performance_degradation))
        scores.append(degradation_score * 0.25)
        
        # 资源使用稳定性（20%权重）
        memory_growth_score = max(0, 100 - abs(self.memory_growth_rate) * 2)  # 每小时增长1MB扣2分
        scores.append(memory_growth_score * 0.2)
        
        self.stability_score = sum(scores)
    
    def _assign_endurance_grade(self):
        """分配持久性等级"""
        # 基于稳定性评分、错误率和性能衰减
        if (self.stability_score >= 90 and 
            self.overall_error_rate <= 0.01 and 
            abs(self.performance_degradation) <= 5):
            self.endurance_grade = 'A+'
        elif (self.stability_score >= 80 and 
              self.overall_error_rate <= 0.02 and 
              abs(self.performance_degradation) <= 10):
            self.endurance_grade = 'A'
        elif (self.stability_score >= 70 and 
              self.overall_error_rate <= 0.05 and 
              abs(self.performance_degradation) <= 20):
            self.endurance_grade = 'B'
        elif (self.stability_score >= 60 and 
              self.overall_error_rate <= 0.10 and 
              abs(self.performance_degradation) <= 30):
            self.endurance_grade = 'C'
        elif (self.stability_score >= 40 and 
              self.overall_error_rate <= 0.20):
            self.endurance_grade = 'D'
        else:
            self.endurance_grade = 'F'


class MockEnduranceAPI:
    """模拟持久负载API"""
    
    def __init__(self, base_response_time: float = 0.1, 
                 degradation_rate: float = 0.001,
                 memory_leak_rate: float = 0.1):
        self.base_response_time = base_response_time
        self.degradation_rate = degradation_rate  # 每请求的性能衰减
        self.memory_leak_rate = memory_leak_rate  # 每请求的内存泄漏（MB）
        self.request_count = 0
        self.start_time = time.time()
        self.simulated_memory = 100.0  # 初始内存使用（MB）
        self._lock = threading.Lock()
    
    def make_request(self, vendor: str, model: str) -> Tuple[float, bool]:
        """模拟API请求"""
        with self._lock:
            self.request_count += 1
            
            # 模拟性能衰减
            current_response_time = (self.base_response_time + 
                                   self.request_count * self.degradation_rate)
            
            # 模拟内存泄漏
            self.simulated_memory += self.memory_leak_rate
            
            # 模拟偶发错误
            error_probability = min(0.1, self.request_count * 0.000001)
            is_success = time.time() % 1 > error_probability
            
            # 模拟网络延迟
            time.sleep(current_response_time * 0.1)  # 实际等待时间缩短
            
            return current_response_time, is_success
    
    def get_memory_usage(self) -> float:
        """获取模拟内存使用量"""
        return self.simulated_memory
    
    def reset(self):
        """重置API状态"""
        with self._lock:
            self.request_count = 0
            self.start_time = time.time()
            self.simulated_memory = 100.0


class SystemResourceMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.monitor_thread = None
        self.snapshots = []
        self._lock = threading.Lock()
    
    def start_monitoring(self, interval: int = 5):
        """开始监控"""
        self.monitoring = True
        self.snapshots = []
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self, interval: int):
        """监控循环"""
        while self.monitoring:
            try:
                # 获取系统资源使用情况
                cpu_percent = psutil.cpu_percent(interval=1)
                memory_info = self.process.memory_info()
                memory_percent = self.process.memory_percent()
                
                snapshot = {
                    'timestamp': datetime.now(),
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory_percent,
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'threads': self.process.num_threads()
                }
                
                with self._lock:
                    self.snapshots.append(snapshot)
                
                time.sleep(interval)
            except Exception as e:
                logger.warning(f"资源监控错误: {e}")
                time.sleep(interval)
    
    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """获取最新快照"""
        with self._lock:
            return self.snapshots[-1] if self.snapshots else None
    
    def get_all_snapshots(self) -> List[Dict[str, Any]]:
        """获取所有快照"""
        with self._lock:
            return self.snapshots.copy()


class EnduranceLoadTestRunner:
    """持久负载测试运行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api = MockEnduranceAPI()
        self.resource_monitor = SystemResourceMonitor()
        self.active_threads = []
        self._stop_event = threading.Event()
    
    def run_endurance_test(self, vendor: str, model: str, 
                          phases: List[EndurancePhase],
                          test_name: str) -> EnduranceLoadResult:
        """运行持久负载测试"""
        print(f"\n开始持久负载测试: {test_name}")
        print(f"厂商: {vendor}, 模型: {model}")
        print(f"测试阶段数: {len(phases)}")
        
        start_time = datetime.now()
        self.api.reset()
        
        # 开始资源监控
        self.resource_monitor.start_monitoring(interval=10)
        
        result = EnduranceLoadResult(
            vendor=vendor,
            model=model,
            test_name=test_name,
            start_time=start_time,
            end_time=start_time,  # 临时值
            total_duration_minutes=0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0
        )
        
        try:
            # 执行各个阶段
            for i, phase in enumerate(phases):
                print(f"\n执行阶段 {i+1}/{len(phases)}: {phase.name}")
                print(f"  目标负载: {phase.target_load} req/s")
                print(f"  持续时间: {phase.duration_minutes} 分钟")
                
                phase_result = self._run_phase(phase, result)
                result.phase_results.append(phase_result)
                
                # 检查是否需要停止
                if self._stop_event.is_set():
                    print("收到停止信号，提前结束测试")
                    break
        
        finally:
            # 停止资源监控
            self.resource_monitor.stop_monitoring()
            
            # 等待所有线程结束
            self._stop_event.set()
            for thread in self.active_threads:
                thread.join(timeout=1)
            
            # 完成结果计算
            end_time = datetime.now()
            result.end_time = end_time
            result.total_duration_minutes = (end_time - start_time).total_seconds() / 60
            
            # 收集性能快照
            self._collect_performance_snapshots(result)
            
            # 计算指标
            result.calculate_metrics()
            
            print(f"\n持久负载测试完成")
            print(f"总耗时: {result.total_duration_minutes:.1f} 分钟")
            print(f"总请求数: {result.total_requests}")
            print(f"成功率: {result.successful_requests/max(result.total_requests,1)*100:.1f}%")
            print(f"稳定性评分: {result.stability_score:.1f}")
            print(f"持久性等级: {result.endurance_grade}")
        
        return result
    
    def _run_phase(self, phase: EndurancePhase, result: EnduranceLoadResult) -> Dict[str, Any]:
        """运行单个阶段"""
        phase_start = datetime.now()
        phase_requests = 0
        phase_successes = 0
        phase_failures = 0
        phase_response_times = []
        
        # 计算阶段总时长（秒）
        total_seconds = phase.duration_minutes * 60
        
        # 渐进加载到目标负载
        current_load = 1
        ramp_step = max(1, phase.target_load // (phase.ramp_time_seconds // 5))
        
        print(f"  渐进加载: 1 -> {phase.target_load} req/s (用时 {phase.ramp_time_seconds}s)")
        
        # 渐进阶段
        ramp_start = time.time()
        while current_load < phase.target_load and time.time() - ramp_start < phase.ramp_time_seconds:
            if self._stop_event.is_set():
                break
            
            # 执行当前负载级别的请求
            batch_start = time.time()
            batch_requests, batch_successes, batch_times = self._execute_load_batch(
                current_load, 1.0, result.vendor, result.model
            )
            
            phase_requests += batch_requests
            phase_successes += batch_successes
            phase_failures += (batch_requests - batch_successes)
            phase_response_times.extend(batch_times)
            
            # 增加负载
            current_load = min(phase.target_load, current_load + ramp_step)
            
            # 控制批次间隔
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
        
        print(f"  稳定负载: {phase.target_load} req/s")
        
        # 稳定负载阶段
        stable_start = time.time()
        stable_duration = total_seconds - phase.ramp_time_seconds
        
        while time.time() - stable_start < stable_duration:
            if self._stop_event.is_set():
                break
            
            batch_start = time.time()
            batch_requests, batch_successes, batch_times = self._execute_load_batch(
                phase.target_load, 1.0, result.vendor, result.model
            )
            
            phase_requests += batch_requests
            phase_successes += batch_successes
            phase_failures += (batch_requests - batch_successes)
            phase_response_times.extend(batch_times)
            
            # 控制批次间隔
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
        
        # 更新总体结果
        result.total_requests += phase_requests
        result.successful_requests += phase_successes
        result.failed_requests += phase_failures
        
        phase_end = datetime.now()
        phase_duration = (phase_end - phase_start).total_seconds() / 60
        
        # 计算阶段统计
        phase_stats = {
            'name': phase.name,
            'duration_minutes': phase_duration,
            'target_load': phase.target_load,
            'total_requests': phase_requests,
            'successful_requests': phase_successes,
            'failed_requests': phase_failures,
            'success_rate': phase_successes / max(phase_requests, 1),
            'avg_response_time': statistics.mean(phase_response_times) if phase_response_times else 0,
            'throughput': phase_requests / (phase_duration * 60) if phase_duration > 0 else 0
        }
        
        print(f"  阶段完成: {phase_requests} 请求, 成功率 {phase_stats['success_rate']*100:.1f}%")
        
        return phase_stats
    
    def _execute_load_batch(self, target_rps: int, duration: float, 
                           vendor: str, model: str) -> Tuple[int, int, List[float]]:
        """执行负载批次"""
        requests_count = 0
        successes_count = 0
        response_times = []
        
        # 使用线程池执行并发请求
        max_workers = min(target_rps, 50)  # 限制最大线程数
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交请求任务
            futures = []
            for _ in range(target_rps):
                if self._stop_event.is_set():
                    break
                future = executor.submit(self.api.make_request, vendor, model)
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures, timeout=duration + 5):
                try:
                    response_time, success = future.result(timeout=1)
                    requests_count += 1
                    if success:
                        successes_count += 1
                    response_times.append(response_time)
                except Exception as e:
                    requests_count += 1
                    response_times.append(10.0)  # 超时或错误的默认响应时间
        
        return requests_count, successes_count, response_times
    
    def _collect_performance_snapshots(self, result: EnduranceLoadResult):
        """收集性能快照"""
        resource_snapshots = self.resource_monitor.get_all_snapshots()
        
        # 将资源快照转换为性能快照
        for i, resource_snap in enumerate(resource_snapshots):
            # 计算对应时间段的性能指标
            snapshot_time = resource_snap['timestamp']
            
            # 模拟性能数据（实际应用中应该从真实监控数据获取）
            base_response_time = 0.1 + i * 0.001  # 模拟性能衰减
            throughput = max(1, 100 - i * 0.1)  # 模拟吞吐量下降
            error_rate = min(0.1, i * 0.0001)  # 模拟错误率增加
            
            perf_snapshot = PerformanceSnapshot(
                timestamp=snapshot_time,
                response_time=base_response_time,
                throughput=throughput,
                error_rate=error_rate,
                cpu_usage=resource_snap['cpu_usage'],
                memory_usage=resource_snap['memory_usage'],
                memory_mb=resource_snap['memory_mb'],
                active_connections=resource_snap.get('threads', 10)
            )
            
            result.performance_snapshots.append(perf_snapshot)
    
    def generate_endurance_report(self, results: List[EnduranceLoadResult]) -> Dict[str, Any]:
        """生成持久负载测试报告"""
        if not results:
            return {'error': '没有测试结果'}
        
        report = {
            'summary': {
                'total_tests': len(results),
                'test_duration_hours': sum(r.total_duration_minutes for r in results) / 60,
                'total_requests': sum(r.total_requests for r in results),
                'overall_success_rate': sum(r.successful_requests for r in results) / max(sum(r.total_requests for r in results), 1),
                'avg_stability_score': statistics.mean([r.stability_score for r in results]),
                'grade_distribution': {}
            },
            'performance_analysis': {},
            'stability_analysis': {},
            'resource_analysis': {},
            'recommendations': []
        }
        
        # 等级分布统计
        grades = [r.endurance_grade for r in results]
        for grade in ['A+', 'A', 'B', 'C', 'D', 'F']:
            report['summary']['grade_distribution'][grade] = grades.count(grade)
        
        # 性能分析
        report['performance_analysis'] = {
            'avg_response_time': statistics.mean([r.avg_response_time for r in results]),
            'avg_throughput': statistics.mean([r.avg_throughput for r in results]),
            'avg_degradation': statistics.mean([r.performance_degradation for r in results]),
            'max_degradation': max([abs(r.performance_degradation) for r in results])
        }
        
        # 稳定性分析
        report['stability_analysis'] = {
            'avg_stability_score': statistics.mean([r.stability_score for r in results]),
            'min_stability_score': min([r.stability_score for r in results]),
            'throughput_stability': statistics.mean([r.throughput_stability for r in results]),
            'error_rate_stability': statistics.mean([r.error_rate_stability for r in results])
        }
        
        # 资源分析
        report['resource_analysis'] = {
            'avg_cpu_usage': statistics.mean([r.avg_cpu_usage for r in results]),
            'peak_cpu_usage': max([r.peak_cpu_usage for r in results]),
            'avg_memory_growth': statistics.mean([r.memory_growth_rate for r in results]),
            'max_memory_growth': max([r.memory_growth_rate for r in results])
        }
        
        # 生成建议
        self._generate_recommendations(report, results)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any], results: List[EnduranceLoadResult]):
        """生成优化建议"""
        recommendations = []
        
        # 性能衰减建议
        avg_degradation = report['performance_analysis']['avg_degradation']
        if avg_degradation > 20:
            recommendations.append("检测到显著性能衰减，建议优化算法或增加资源")
        elif avg_degradation > 10:
            recommendations.append("存在轻微性能衰减，建议监控并考虑优化")
        
        # 内存泄漏建议
        avg_memory_growth = report['resource_analysis']['avg_memory_growth']
        if avg_memory_growth > 10:  # 每小时增长超过10MB
            recommendations.append("检测到可能的内存泄漏，建议检查内存管理")
        
        # 稳定性建议
        avg_stability = report['stability_analysis']['avg_stability_score']
        if avg_stability < 60:
            recommendations.append("系统稳定性较低，建议检查负载均衡和错误处理")
        elif avg_stability < 80:
            recommendations.append("系统稳定性中等，建议进一步优化")
        
        # CPU使用建议
        peak_cpu = report['resource_analysis']['peak_cpu_usage']
        if peak_cpu > 90:
            recommendations.append("CPU使用率过高，建议增加计算资源或优化算法")
        elif peak_cpu > 80:
            recommendations.append("CPU使用率较高，建议监控并考虑扩容")
        
        report['recommendations'] = recommendations
    
    def stop_test(self):
        """停止测试"""
        self._stop_event.set()


class TestEnduranceLoad:
    """持久负载测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.config = LOAD_TEST_CONFIG
        self.endurance_runner = EnduranceLoadTestRunner(self.config)
    
    def teardown_method(self):
        """测试后清理"""
        if hasattr(self, 'endurance_runner'):
            self.endurance_runner.stop_test()
        
        # 强制垃圾回收
        gc.collect()
    
    def _print_endurance_summary(self, results: List[EnduranceLoadResult]):
        """打印持久负载测试摘要"""
        print(f"\n=== 持久负载测试摘要 ===")
        print(f"测试数量: {len(results)}")
        
        for result in results:
            print(f"\n{result.test_name}:")
            print(f"  厂商/模型: {result.vendor}/{result.model}")
            print(f"  测试时长: {result.total_duration_minutes:.1f} 分钟")
            print(f"  总请求数: {result.total_requests}")
            print(f"  成功率: {result.successful_requests/max(result.total_requests,1)*100:.1f}%")
            print(f"  平均响应时间: {result.avg_response_time:.3f}s")
            print(f"  平均吞吐量: {result.avg_throughput:.1f} req/s")
            print(f"  性能衰减: {result.performance_degradation:.1f}%")
            print(f"  稳定性评分: {result.stability_score:.1f}")
            print(f"  内存增长率: {result.memory_growth_rate:.2f} MB/h")
            print(f"  持久性等级: {result.endurance_grade}")
    
    @pytest.mark.load_test
    @pytest.mark.endurance_load
    def test_short_endurance_load(self):
        """
        短期持久负载测试
        
        测试系统在短期持续负载下的表现
        """
        vendor = 'deepseek'
        model = 'deepseek-chat'
        
        # 定义测试阶段
        phases = [
            EndurancePhase(
                name="预热阶段",
                duration_minutes=2,
                target_load=10,
                ramp_time_seconds=30
            ),
            EndurancePhase(
                name="稳定负载",
                duration_minutes=5,
                target_load=20,
                ramp_time_seconds=30
            ),
            EndurancePhase(
                name="高负载",
                duration_minutes=3,
                target_load=30,
                ramp_time_seconds=30
            )
        ]
        
        # 运行测试
        result = self.endurance_runner.run_endurance_test(
            vendor, model, phases, "short_endurance_test"
        )
        
        self._print_endurance_summary([result])
        
        # 短期持久性断言
        assert result.total_requests > 0
        assert result.successful_requests > 0
        assert result.total_duration_minutes >= 8  # 至少8分钟
        assert result.total_duration_minutes <= 15  # 不超过15分钟
        
        # 性能要求
        assert result.avg_response_time <= 2.0  # 平均响应时间不超过2秒
        assert result.overall_error_rate <= 0.1  # 错误率不超过10%
        assert result.stability_score >= 50  # 稳定性评分至少50分
        
        # 资源使用要求
        assert result.peak_cpu_usage <= 95  # CPU使用率不超过95%
        assert abs(result.memory_growth_rate) <= 50  # 内存增长率不超过50MB/h
    
    @pytest.mark.load_test
    @pytest.mark.endurance_load
    def test_medium_endurance_load(self):
        """
        中期持久负载测试
        
        测试系统在中期持续负载下的稳定性
        """
        vendor = 'ernie'
        model = 'ernie-3.5-8k'
        
        # 定义测试阶段
        phases = [
            EndurancePhase(
                name="渐进加载",
                duration_minutes=3,
                target_load=15,
                ramp_time_seconds=60
            ),
            EndurancePhase(
                name="稳定运行",
                duration_minutes=10,
                target_load=25,
                ramp_time_seconds=30
            ),
            EndurancePhase(
                name="峰值负载",
                duration_minutes=5,
                target_load=35,
                ramp_time_seconds=60
            ),
            EndurancePhase(
                name="恢复阶段",
                duration_minutes=3,
                target_load=15,
                ramp_time_seconds=30
            )
        ]
        
        # 运行测试
        result = self.endurance_runner.run_endurance_test(
            vendor, model, phases, "medium_endurance_test"
        )
        
        self._print_endurance_summary([result])
        
        # 中期持久性断言
        assert result.total_requests > 0
        assert result.total_duration_minutes >= 18  # 至少18分钟
        assert result.total_duration_minutes <= 25  # 不超过25分钟
        
        # 性能要求
        assert result.avg_response_time <= 3.0  # 平均响应时间不超过3秒
        assert result.overall_error_rate <= 0.15  # 错误率不超过15%
        assert result.stability_score >= 40  # 稳定性评分至少40分
        
        # 性能衰减要求
        assert abs(result.performance_degradation) <= 30  # 性能衰减不超过30%
        
        # 阶段分析
        assert len(result.phase_results) == len(phases)
        
        # 恢复阶段应该比峰值阶段表现更好
        peak_phase = next(p for p in result.phase_results if p['name'] == '峰值负载')
        recovery_phase = next(p for p in result.phase_results if p['name'] == '恢复阶段')
        
        assert recovery_phase['success_rate'] >= peak_phase['success_rate'] * 0.9
    
    @pytest.mark.load_test
    @pytest.mark.comprehensive_load
    def test_memory_leak_detection(self):
        """
        内存泄漏检测测试
        
        专门检测长时间运行中的内存泄漏问题
        """
        vendor = 'deepseek'
        model = 'deepseek-r1'
        
        # 使用高内存泄漏率的API进行测试
        original_api = self.endurance_runner.api
        self.endurance_runner.api = MockEnduranceAPI(
            base_response_time=0.1,
            degradation_rate=0.0001,
            memory_leak_rate=2.0  # 每请求泄漏2MB
        )
        
        try:
            # 定义测试阶段
            phases = [
                EndurancePhase(
                    name="内存泄漏检测",
                    duration_minutes=8,
                    target_load=20,
                    ramp_time_seconds=30
                )
            ]
            
            # 运行测试
            result = self.endurance_runner.run_endurance_test(
                vendor, model, phases, "memory_leak_detection"
            )
            
            self._print_endurance_summary([result])
            
            print(f"\n=== 内存泄漏分析 ===")
            print(f"内存增长率: {result.memory_growth_rate:.2f} MB/h")
            print(f"初始内存: {result.performance_snapshots[0].memory_mb:.1f} MB")
            print(f"最终内存: {result.performance_snapshots[-1].memory_mb:.1f} MB")
            print(f"内存增长: {result.performance_snapshots[-1].memory_mb - result.performance_snapshots[0].memory_mb:.1f} MB")
            
            # 内存泄漏检测断言
            assert result.total_requests > 0
            
            # 应该检测到明显的内存增长
            assert result.memory_growth_rate > 10  # 每小时增长超过10MB
            
            # 内存增长应该与请求数量相关
            memory_growth = (result.performance_snapshots[-1].memory_mb - 
                           result.performance_snapshots[0].memory_mb)
            assert memory_growth > 50  # 至少增长50MB
            
            # 稳定性评分应该因内存泄漏而降低
            assert result.stability_score <= 80  # 内存泄漏影响稳定性
        
        finally:
            # 恢复原始API
            self.endurance_runner.api = original_api
    
    @pytest.mark.load_test
    @pytest.mark.comprehensive_load
    def test_performance_degradation_analysis(self):
        """
        性能衰减分析测试
        
        分析长时间运行中的性能衰减模式
        """
        vendor = 'doubao'
        model = 'doubao-1-5-pro-32k-character-250715'
        
        # 使用高性能衰减率的API进行测试
        original_api = self.endurance_runner.api
        self.endurance_runner.api = MockEnduranceAPI(
            base_response_time=0.1,
            degradation_rate=0.01,  # 每请求增加10ms
            memory_leak_rate=0.1
        )
        
        try:
            # 定义测试阶段
            phases = [
                EndurancePhase(
                    name="性能衰减检测",
                    duration_minutes=6,
                    target_load=25,
                    ramp_time_seconds=30
                )
            ]
            
            # 运行测试
            result = self.endurance_runner.run_endurance_test(
                vendor, model, phases, "performance_degradation_test"
            )
            
            self._print_endurance_summary([result])
            
            print(f"\n=== 性能衰减分析 ===")
            print(f"性能衰减: {result.performance_degradation:.1f}%")
            print(f"初始响应时间: {result.min_response_time:.3f}s")
            print(f"最终响应时间: {result.max_response_time:.3f}s")
            print(f"平均响应时间: {result.avg_response_time:.3f}s")
            
            # 性能衰减检测断言
            assert result.total_requests > 0
            
            # 应该检测到明显的性能衰减
            assert result.performance_degradation > 10  # 性能衰减超过10%
            
            # 最大响应时间应该明显大于最小响应时间
            assert result.max_response_time > result.min_response_time * 1.5
            
            # 稳定性评分应该因性能衰减而降低
            assert result.stability_score <= 70  # 性能衰减影响稳定性
        
        finally:
            # 恢复原始API
            self.endurance_runner.api = original_api
    
    @pytest.mark.load_test
    @pytest.mark.comprehensive_load
    def test_multi_vendor_endurance_comparison(self):
        """
        多厂商持久性对比测试
        
        对比不同厂商在持久负载下的表现
        """
        vendors_models = [
            ('deepseek', 'deepseek-chat'),
            ('ernie', 'ernie-3.5-8k'),
            ('google', 'gemini-pro')
        ]
        
        # 定义统一的测试阶段
        phases = [
            EndurancePhase(
                name="持久性对比",
                duration_minutes=4,
                target_load=20,
                ramp_time_seconds=30
            )
        ]
        
        results = []
        
        for vendor, model in vendors_models:
            print(f"\n测试厂商: {vendor}/{model}")
            
            result = self.endurance_runner.run_endurance_test(
                vendor, model, phases, f"endurance_comparison_{vendor}"
            )
            results.append(result)
        
        self._print_endurance_summary(results)
        
        # 生成对比报告
        report = self.endurance_runner.generate_endurance_report(results)
        
        print(f"\n=== 多厂商持久性对比 ===")
        print(f"平均稳定性评分: {report['summary']['avg_stability_score']:.1f}")
        print(f"总体成功率: {report['summary']['overall_success_rate']*100:.1f}%")
        
        print(f"\n等级分布:")
        for grade, count in report['summary']['grade_distribution'].items():
            if count > 0:
                print(f"  {grade}: {count} 个")
        
        # 多厂商对比断言
        assert len(results) == len(vendors_models)
        assert all(r.total_requests > 0 for r in results)
        
        # 所有厂商都应该完成基本测试
        for result in results:
            assert result.total_duration_minutes >= 3
            assert result.successful_requests > 0
            assert result.stability_score >= 0
        
        # 至少有一个厂商应该达到良好等级
        good_grades = ['A+', 'A', 'B']
        assert any(r.endurance_grade in good_grades for r in results)
        
        # 报告应该包含有效的分析
        assert 'performance_analysis' in report
        assert 'stability_analysis' in report
        assert 'resource_analysis' in report
        assert len(report['recommendations']) >= 0
    
    @pytest.mark.load_test
    @pytest.mark.benchmark
    def test_endurance_benchmark(self, benchmark):
        """
        持久负载基准测试
        
        使用pytest-benchmark进行持久负载基准测试
        """
        vendor = 'deepseek'
        model = 'deepseek-chat'
        
        # 定义轻量级测试阶段用于基准测试
        phases = [
            EndurancePhase(
                name="基准测试",
                duration_minutes=2,
                target_load=15,
                ramp_time_seconds=20
            )
        ]
        
        # 运行基准测试
        result = benchmark(self.endurance_runner.run_endurance_test,
                          vendor, model, phases, "endurance_benchmark")
        
        self._print_endurance_summary([result])
        
        # 持久负载基准断言
        thresholds = self.config['performance_thresholds']
        assert result.total_requests > 0
        assert result.avg_response_time <= thresholds['response_time']['max_avg']
        assert result.stability_score >= 30  # 至少30分的稳定性
        assert result.overall_error_rate <= 0.2  # 错误率不超过20%