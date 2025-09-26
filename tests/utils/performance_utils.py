# -*- coding: utf-8 -*-
"""
性能测试工具模块

功能：提供性能指标收集、资源监控、性能测试器、基准测试等功能
作者：HarborAI测试团队
创建时间：2024
"""

import time
import psutil
import threading
import asyncio
import statistics
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import functools


@dataclass
class PerformanceMetrics:
    """性能指标数据类
    
    功能：存储各种性能指标数据
    参数：
        response_time: 响应时间（毫秒）
        throughput: 吞吐量（请求/秒）
        error_rate: 错误率（0-1）
        cpu_usage: CPU使用率（0-100）
        memory_usage: 内存使用率（0-100）
        concurrent_users: 并发用户数
    """
    response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    concurrent_users: int = 0
    timestamp: float = field(default_factory=time.time)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThresholds:
    """性能阈值配置类
    
    功能：定义性能测试的各种阈值
    参数：
        max_response_time: 最大响应时间（毫秒）
        min_throughput: 最小吞吐量（请求/秒）
        max_error_rate: 最大错误率（0-1）
        max_cpu_usage: 最大CPU使用率（0-100）
        max_memory_usage: 最大内存使用率（0-100）
    """
    max_response_time: float = 1000.0  # 1秒
    min_throughput: float = 10.0  # 10 RPS
    max_error_rate: float = 0.05  # 5%
    max_cpu_usage: float = 80.0  # 80%
    max_memory_usage: float = 80.0  # 80%
    max_concurrent_users: int = 1000


class SystemMonitor:
    """系统资源监控器
    
    功能：监控系统CPU、内存、网络等资源使用情况
    假设：系统支持psutil库的所有功能
    不确定点：某些系统可能不支持特定的监控指标
    验证方法：pytest tests/test_performance_utils.py::TestSystemMonitor
    """
    
    def __init__(self, interval: float = 1.0):
        """初始化系统监控器
        
        参数：
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self.monitoring = False
        self.metrics_history: List[Dict[str, Any]] = []
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logging.info("系统监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logging.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self.collect_system_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
                    # 保持最近1000个数据点
                    if len(self.metrics_history) > 1000:
                        self.metrics_history.pop(0)
                
                time.sleep(self.interval)
            except Exception as e:
                logging.error(f"监控过程中发生错误：{e}")
                time.sleep(self.interval)
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标
        
        功能：收集当前系统的各项性能指标
        返回：系统指标字典
        """
        try:
            # CPU指标
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # 内存指标
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # 磁盘指标
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # 网络指标
            network_io = psutil.net_io_counters()
            
            # 进程指标
            process_count = len(psutil.pids())
            
            return {
                'timestamp': time.time(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': cpu_freq.current if cpu_freq else None
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'percent': swap.percent
                },
                'disk': {
                    'total': disk_usage.total,
                    'used': disk_usage.used,
                    'free': disk_usage.free,
                    'percent': (disk_usage.used / disk_usage.total) * 100,
                    'read_bytes': disk_io.read_bytes if disk_io else 0,
                    'write_bytes': disk_io.write_bytes if disk_io else 0
                },
                'network': {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv
                },
                'processes': {
                    'count': process_count
                }
            }
        except Exception as e:
            logging.error(f"收集系统指标时发生错误：{e}")
            return {
                'timestamp': time.time(),
                'error': str(e)
            }
    
    def get_metrics_summary(self, duration_seconds: int = None) -> Dict[str, Any]:
        """获取指标汇总
        
        功能：计算指定时间段内的指标统计信息
        参数：
            duration_seconds: 统计时间段（秒），None表示全部历史
        返回：指标汇总字典
        """
        with self._lock:
            metrics = self.metrics_history.copy()
        
        if not metrics:
            return {}
        
        # 过滤时间范围
        if duration_seconds:
            cutoff_time = time.time() - duration_seconds
            metrics = [m for m in metrics if m.get('timestamp', 0) >= cutoff_time]
        
        if not metrics:
            return {}
        
        # 计算统计信息
        cpu_values = [m.get('cpu', {}).get('percent', 0) for m in metrics if 'error' not in m]
        memory_values = [m.get('memory', {}).get('percent', 0) for m in metrics if 'error' not in m]
        
        summary = {
            'period': {
                'start_time': min(m.get('timestamp', 0) for m in metrics),
                'end_time': max(m.get('timestamp', 0) for m in metrics),
                'duration_seconds': duration_seconds or (max(m.get('timestamp', 0) for m in metrics) - min(m.get('timestamp', 0) for m in metrics)),
                'data_points': len(metrics)
            },
            'cpu': {
                'avg': statistics.mean(cpu_values) if cpu_values else 0,
                'min': min(cpu_values) if cpu_values else 0,
                'max': max(cpu_values) if cpu_values else 0,
                'median': statistics.median(cpu_values) if cpu_values else 0
            },
            'memory': {
                'avg': statistics.mean(memory_values) if memory_values else 0,
                'min': min(memory_values) if memory_values else 0,
                'max': max(memory_values) if memory_values else 0,
                'median': statistics.median(memory_values) if memory_values else 0
            }
        }
        
        return summary
    
    def clear_history(self):
        """清空历史数据"""
        with self._lock:
            self.metrics_history.clear()


class PerformanceTester:
    """性能测试器
    
    功能：执行各种类型的性能测试
    假设：被测试的函数是线程安全的
    不确定点：某些异步函数可能需要特殊处理
    验证方法：pytest tests/test_performance_utils.py::TestPerformanceTester
    """
    
    def __init__(self, thresholds: PerformanceThresholds = None):
        """初始化性能测试器
        
        参数：
            thresholds: 性能阈值配置
        """
        self.thresholds = thresholds or PerformanceThresholds()
        self.system_monitor = SystemMonitor()
        self.test_results: List[Dict[str, Any]] = []
    
    def load_test(
        self,
        target_function: Callable,
        concurrent_users: int = 10,
        duration_seconds: int = 60,
        ramp_up_seconds: int = 10,
        **function_kwargs
    ) -> Dict[str, Any]:
        """负载测试
        
        功能：模拟多用户并发访问，测试系统在正常负载下的性能
        参数：
            target_function: 目标测试函数
            concurrent_users: 并发用户数
            duration_seconds: 测试持续时间
            ramp_up_seconds: 用户增长时间
            **function_kwargs: 传递给目标函数的参数
        返回：测试结果字典
        """
        logging.info(f"开始负载测试：{concurrent_users}并发用户，持续{duration_seconds}秒")
        
        # 启动系统监控
        self.system_monitor.start_monitoring()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        results = []
        errors = []
        
        def worker():
            """工作线程函数"""
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    result = target_function(**function_kwargs)
                    request_end = time.time()
                    
                    results.append({
                        'timestamp': request_start,
                        'response_time': (request_end - request_start) * 1000,  # 转换为毫秒
                        'success': True,
                        'result': result
                    })
                except Exception as e:
                    errors.append({
                        'timestamp': time.time(),
                        'error': str(e),
                        'type': type(e).__name__
                    })
                
                # 简单的速率控制
                time.sleep(0.1)
        
        # 逐渐增加用户数
        threads = []
        for i in range(concurrent_users):
            if time.time() >= end_time:
                break
            
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
            
            # 渐进式增加用户
            if ramp_up_seconds > 0:
                time.sleep(ramp_up_seconds / concurrent_users)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 停止系统监控
        self.system_monitor.stop_monitoring()
        
        # 分析结果
        test_result = self._analyze_results(results, errors, start_time, end_time)
        test_result['test_type'] = 'load_test'
        test_result['concurrent_users'] = concurrent_users
        test_result['system_metrics'] = self.system_monitor.get_metrics_summary(duration_seconds)
        
        self.test_results.append(test_result)
        return test_result
    
    def stress_test(
        self,
        target_function: Callable,
        max_users: int = 100,
        step_size: int = 10,
        step_duration: int = 30,
        **function_kwargs
    ) -> Dict[str, Any]:
        """压力测试
        
        功能：逐步增加负载直到系统达到极限
        参数：
            target_function: 目标测试函数
            max_users: 最大用户数
            step_size: 每步增加的用户数
            step_duration: 每步持续时间
            **function_kwargs: 传递给目标函数的参数
        返回：测试结果字典
        """
        logging.info(f"开始压力测试：最大{max_users}用户，每{step_duration}秒增加{step_size}用户")
        
        self.system_monitor.start_monitoring()
        
        all_results = []
        step_results = []
        
        for current_users in range(step_size, max_users + 1, step_size):
            logging.info(f"压力测试步骤：{current_users}并发用户")
            
            step_result = self.load_test(
                target_function,
                concurrent_users=current_users,
                duration_seconds=step_duration,
                ramp_up_seconds=5,
                **function_kwargs
            )
            
            step_results.append({
                'users': current_users,
                'metrics': step_result
            })
            
            # 检查是否达到性能阈值
            if (step_result['avg_response_time'] > self.thresholds.max_response_time or
                step_result['error_rate'] > self.thresholds.max_error_rate):
                logging.warning(f"达到性能阈值，停止压力测试")
                break
        
        self.system_monitor.stop_monitoring()
        
        # 分析整体结果
        total_requests = sum(step['metrics']['total_requests'] for step in step_results)
        total_errors = sum(step['metrics']['total_errors'] for step in step_results)
        
        stress_result = {
            'test_type': 'stress_test',
            'max_users_tested': max(step['users'] for step in step_results),
            'total_requests': total_requests,
            'total_errors': total_errors,
            'overall_error_rate': total_errors / total_requests if total_requests > 0 else 0,
            'step_results': step_results,
            'breaking_point': self._find_breaking_point(step_results)
        }
        
        self.test_results.append(stress_result)
        return stress_result
    
    def spike_test(
        self,
        target_function: Callable,
        normal_users: int = 10,
        spike_users: int = 100,
        spike_duration: int = 30,
        total_duration: int = 120,
        **function_kwargs
    ) -> Dict[str, Any]:
        """尖峰测试
        
        功能：测试系统在突然增加负载时的表现
        参数：
            target_function: 目标测试函数
            normal_users: 正常负载用户数
            spike_users: 尖峰负载用户数
            spike_duration: 尖峰持续时间
            total_duration: 总测试时间
            **function_kwargs: 传递给目标函数的参数
        返回：测试结果字典
        """
        logging.info(f"开始尖峰测试：正常{normal_users}用户，尖峰{spike_users}用户")
        
        self.system_monitor.start_monitoring()
        
        start_time = time.time()
        spike_start = start_time + (total_duration - spike_duration) / 2
        spike_end = spike_start + spike_duration
        end_time = start_time + total_duration
        
        results = []
        errors = []
        
        def worker(is_spike_user=False):
            """工作线程函数"""
            while time.time() < end_time:
                current_time = time.time()
                
                # 尖峰用户只在尖峰期间工作
                if is_spike_user and not (spike_start <= current_time <= spike_end):
                    time.sleep(0.1)
                    continue
                
                try:
                    request_start = time.time()
                    result = target_function(**function_kwargs)
                    request_end = time.time()
                    
                    results.append({
                        'timestamp': request_start,
                        'response_time': (request_end - request_start) * 1000,
                        'success': True,
                        'is_spike': spike_start <= request_start <= spike_end,
                        'result': result
                    })
                except Exception as e:
                    errors.append({
                        'timestamp': time.time(),
                        'error': str(e),
                        'type': type(e).__name__,
                        'is_spike': spike_start <= time.time() <= spike_end
                    })
                
                time.sleep(0.1)
        
        # 启动正常用户线程
        threads = []
        for _ in range(normal_users):
            thread = threading.Thread(target=worker, args=(False,))
            thread.start()
            threads.append(thread)
        
        # 启动尖峰用户线程
        for _ in range(spike_users - normal_users):
            thread = threading.Thread(target=worker, args=(True,))
            thread.start()
            threads.append(thread)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        self.system_monitor.stop_monitoring()
        
        # 分析结果
        spike_result = self._analyze_spike_results(results, errors, spike_start, spike_end)
        spike_result['test_type'] = 'spike_test'
        spike_result['normal_users'] = normal_users
        spike_result['spike_users'] = spike_users
        spike_result['system_metrics'] = self.system_monitor.get_metrics_summary(total_duration)
        
        self.test_results.append(spike_result)
        return spike_result
    
    def _analyze_results(
        self, 
        results: List[Dict], 
        errors: List[Dict], 
        start_time: float, 
        end_time: float
    ) -> Dict[str, Any]:
        """分析测试结果"""
        if not results and not errors:
            return {
                'total_requests': 0,
                'total_errors': 0,
                'error_rate': 0,
                'avg_response_time': 0,
                'throughput': 0
            }
        
        total_requests = len(results) + len(errors)
        total_errors = len(errors)
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        
        if results:
            response_times = [r['response_time'] for r in results]
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = self._calculate_percentile(response_times, 0.95)
            p99_response_time = self._calculate_percentile(response_times, 0.99)
        else:
            avg_response_time = 0
            min_response_time = 0
            max_response_time = 0
            p50_response_time = 0
            p95_response_time = 0
            p99_response_time = 0
        
        duration = end_time - start_time
        throughput = len(results) / duration if duration > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': len(results),
            'total_errors': total_errors,
            'error_rate': error_rate,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time,
            'p50_response_time': p50_response_time,
            'p95_response_time': p95_response_time,
            'p99_response_time': p99_response_time,
            'throughput': throughput,
            'duration': duration,
            'start_time': start_time,
            'end_time': end_time,
            'threshold_violations': self._check_thresholds({
                'avg_response_time': avg_response_time,
                'error_rate': error_rate,
                'throughput': throughput
            })
        }
    
    def _analyze_spike_results(
        self, 
        results: List[Dict], 
        errors: List[Dict], 
        spike_start: float, 
        spike_end: float
    ) -> Dict[str, Any]:
        """分析尖峰测试结果"""
        # 分离正常期间和尖峰期间的结果
        normal_results = [r for r in results if not r.get('is_spike', False)]
        spike_results = [r for r in results if r.get('is_spike', False)]
        
        normal_errors = [e for e in errors if not e.get('is_spike', False)]
        spike_errors = [e for e in errors if e.get('is_spike', False)]
        
        # 分别分析
        normal_analysis = self._analyze_results(
            normal_results, normal_errors, 
            spike_start - 60, spike_start  # 假设正常期间为尖峰前60秒
        )
        
        spike_analysis = self._analyze_results(
            spike_results, spike_errors,
            spike_start, spike_end
        )
        
        return {
            'normal_period': normal_analysis,
            'spike_period': spike_analysis,
            'performance_degradation': {
                'response_time_increase': (
                    spike_analysis['avg_response_time'] - normal_analysis['avg_response_time']
                ) / normal_analysis['avg_response_time'] if normal_analysis['avg_response_time'] > 0 else 0,
                'throughput_decrease': (
                    normal_analysis['throughput'] - spike_analysis['throughput']
                ) / normal_analysis['throughput'] if normal_analysis['throughput'] > 0 else 0,
                'error_rate_increase': spike_analysis['error_rate'] - normal_analysis['error_rate']
            }
        }
    
    def _find_breaking_point(self, step_results: List[Dict]) -> Dict[str, Any]:
        """找到系统的破坏点"""
        for i, step in enumerate(step_results):
            metrics = step['metrics']
            if (metrics['avg_response_time'] > self.thresholds.max_response_time or
                metrics['error_rate'] > self.thresholds.max_error_rate):
                return {
                    'users': step['users'],
                    'step_index': i,
                    'reason': 'threshold_exceeded',
                    'metrics': metrics
                }
        
        return {
            'users': step_results[-1]['users'] if step_results else 0,
            'step_index': len(step_results) - 1,
            'reason': 'test_completed',
            'metrics': step_results[-1]['metrics'] if step_results else {}
        }
    
    def _check_thresholds(self, metrics: Dict[str, float]) -> List[str]:
        """检查性能阈值违规"""
        violations = []
        
        if metrics.get('avg_response_time', 0) > self.thresholds.max_response_time:
            violations.append(f"响应时间超过阈值：{metrics['avg_response_time']:.2f}ms > {self.thresholds.max_response_time}ms")
        
        if metrics.get('error_rate', 0) > self.thresholds.max_error_rate:
            violations.append(f"错误率超过阈值：{metrics['error_rate']:.2%} > {self.thresholds.max_error_rate:.2%}")
        
        if metrics.get('throughput', 0) < self.thresholds.min_throughput:
            violations.append(f"吞吐量低于阈值：{metrics['throughput']:.2f} RPS < {self.thresholds.min_throughput} RPS")
        
        return violations
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def generate_report(self, output_file: str = None) -> Dict[str, Any]:
        """生成性能测试报告
        
        功能：生成包含所有测试结果的详细报告
        参数：
            output_file: 输出文件路径（可选）
        返回：报告字典
        """
        report = {
            'summary': {
                'total_tests': len(self.test_results),
                'test_types': list(set(result.get('test_type', 'unknown') for result in self.test_results)),
                'generated_at': datetime.now().isoformat(),
                'thresholds': self.thresholds.__dict__
            },
            'test_results': self.test_results,
            'recommendations': self._generate_recommendations()
        }
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        if not self.test_results:
            return ["暂无测试结果，无法生成建议"]
        
        # 分析所有测试结果
        avg_response_times = []
        error_rates = []
        throughputs = []
        
        for result in self.test_results:
            if 'avg_response_time' in result:
                avg_response_times.append(result['avg_response_time'])
                error_rates.append(result.get('error_rate', 0))
                throughputs.append(result.get('throughput', 0))
        
        if avg_response_times:
            avg_response_time = statistics.mean(avg_response_times)
            avg_error_rate = statistics.mean(error_rates)
            avg_throughput = statistics.mean(throughputs)
            
            if avg_response_time > self.thresholds.max_response_time:
                recommendations.append(f"响应时间过高（{avg_response_time:.2f}ms），建议优化代码逻辑或增加缓存")
            
            if avg_error_rate > self.thresholds.max_error_rate:
                recommendations.append(f"错误率过高（{avg_error_rate:.2%}），建议检查错误处理和系统稳定性")
            
            if avg_throughput < self.thresholds.min_throughput:
                recommendations.append(f"吞吐量过低（{avg_throughput:.2f} RPS），建议优化并发处理能力")
            
            # 基于系统监控数据的建议
            for result in self.test_results:
                system_metrics = result.get('system_metrics', {})
                cpu_metrics = system_metrics.get('cpu', {})
                memory_metrics = system_metrics.get('memory', {})
                
                if cpu_metrics.get('max', 0) > 90:
                    recommendations.append("CPU使用率过高，建议优化计算密集型操作")
                
                if memory_metrics.get('max', 0) > 90:
                    recommendations.append("内存使用率过高，建议检查内存泄漏或优化内存使用")
        
        if not recommendations:
            recommendations.append("性能表现良好，所有指标都在正常范围内")
        
        return recommendations


class BenchmarkSuite:
    """基准测试套件
    
    功能：提供标准化的基准测试功能
    假设：基准测试函数是可重复执行的
    不确定点：某些基准测试可能受到系统状态影响
    验证方法：pytest tests/test_performance_utils.py::TestBenchmarkSuite
    """
    
    def __init__(self):
        """初始化基准测试套件"""
        self.benchmarks: Dict[str, Dict[str, Any]] = {}
        self.baseline_results: Dict[str, Dict[str, Any]] = {}
    
    def register_benchmark(
        self,
        name: str,
        function: Callable,
        setup_function: Callable = None,
        teardown_function: Callable = None,
        iterations: int = 100,
        **kwargs
    ):
        """注册基准测试
        
        功能：注册一个基准测试函数
        参数：
            name: 基准测试名称
            function: 测试函数
            setup_function: 设置函数
            teardown_function: 清理函数
            iterations: 迭代次数
            **kwargs: 传递给测试函数的参数
        """
        self.benchmarks[name] = {
            'function': function,
            'setup_function': setup_function,
            'teardown_function': teardown_function,
            'iterations': iterations,
            'kwargs': kwargs
        }
    
    def run_benchmark(self, name: str) -> Dict[str, Any]:
        """运行单个基准测试
        
        功能：执行指定的基准测试并返回结果
        参数：
            name: 基准测试名称
        返回：基准测试结果字典
        """
        if name not in self.benchmarks:
            raise ValueError(f"基准测试 '{name}' 未注册")
        
        benchmark = self.benchmarks[name]
        function = benchmark['function']
        setup_function = benchmark.get('setup_function')
        teardown_function = benchmark.get('teardown_function')
        iterations = benchmark['iterations']
        kwargs = benchmark['kwargs']
        
        logging.info(f"运行基准测试：{name}（{iterations}次迭代）")
        
        execution_times = []
        errors = []
        
        for i in range(iterations):
            try:
                # 执行设置函数
                if setup_function:
                    setup_function()
                
                # 执行基准测试
                start_time = time.perf_counter()
                result = function(**kwargs)
                end_time = time.perf_counter()
                
                execution_times.append((end_time - start_time) * 1000)  # 转换为毫秒
                
                # 执行清理函数
                if teardown_function:
                    teardown_function()
                    
            except Exception as e:
                errors.append({
                    'iteration': i,
                    'error': str(e),
                    'type': type(e).__name__
                })
        
        # 计算统计信息
        if execution_times:
            result = {
                'name': name,
                'iterations': iterations,
                'successful_iterations': len(execution_times),
                'failed_iterations': len(errors),
                'avg_time_ms': statistics.mean(execution_times),
                'min_time_ms': min(execution_times),
                'max_time_ms': max(execution_times),
                'median_time_ms': statistics.median(execution_times),
                'std_dev_ms': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'p95_time_ms': self._calculate_percentile(execution_times, 0.95),
                'p99_time_ms': self._calculate_percentile(execution_times, 0.99),
                'operations_per_second': 1000 / statistics.mean(execution_times),
                'errors': errors,
                'timestamp': time.time()
            }
        else:
            result = {
                'name': name,
                'iterations': iterations,
                'successful_iterations': 0,
                'failed_iterations': len(errors),
                'errors': errors,
                'timestamp': time.time()
            }
        
        return result
    
    def run_all_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """运行所有基准测试
        
        功能：执行所有注册的基准测试
        返回：所有基准测试结果字典
        """
        results = {}
        
        for name in self.benchmarks:
            try:
                results[name] = self.run_benchmark(name)
            except Exception as e:
                logging.error(f"基准测试 '{name}' 执行失败：{e}")
                results[name] = {
                    'name': name,
                    'error': str(e),
                    'timestamp': time.time()
                }
        
        return results
    
    def set_baseline(self, name: str, result: Dict[str, Any] = None):
        """设置基准线
        
        功能：设置基准测试的基准线结果
        参数：
            name: 基准测试名称
            result: 基准线结果，None表示使用当前运行结果
        """
        if result is None:
            result = self.run_benchmark(name)
        
        self.baseline_results[name] = result
        logging.info(f"已设置基准测试 '{name}' 的基准线")
    
    def compare_with_baseline(self, name: str) -> Dict[str, Any]:
        """与基准线比较
        
        功能：将当前测试结果与基准线进行比较
        参数：
            name: 基准测试名称
        返回：比较结果字典
        """
        if name not in self.baseline_results:
            raise ValueError(f"基准测试 '{name}' 没有设置基准线")
        
        current_result = self.run_benchmark(name)
        baseline_result = self.baseline_results[name]
        
        if 'avg_time_ms' not in current_result or 'avg_time_ms' not in baseline_result:
            return {
                'name': name,
                'comparison': 'failed',
                'reason': 'missing_performance_data'
            }
        
        current_avg = current_result['avg_time_ms']
        baseline_avg = baseline_result['avg_time_ms']
        
        performance_change = ((current_avg - baseline_avg) / baseline_avg) * 100
        
        return {
            'name': name,
            'current_result': current_result,
            'baseline_result': baseline_result,
            'performance_change_percent': performance_change,
            'performance_status': self._get_performance_status(performance_change),
            'comparison_timestamp': time.time()
        }
    
    def _get_performance_status(self, change_percent: float) -> str:
        """获取性能状态"""
        if change_percent < -10:
            return 'significantly_improved'
        elif change_percent < -5:
            return 'improved'
        elif change_percent < 5:
            return 'stable'
        elif change_percent < 10:
            return 'degraded'
        else:
            return 'significantly_degraded'
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]


# 装饰器和上下文管理器

def measure_performance(func: Callable = None, *, name: str = None, iterations: int = 1):
    """性能测量装饰器
    
    功能：装饰器形式的性能测量工具
    参数：
        func: 被装饰的函数
        name: 测量名称
        iterations: 迭代次数
    返回：装饰后的函数
    边界条件：处理异步函数和生成器函数
    假设：被装饰的函数是可重复执行的
    不确定点：某些函数可能有副作用
    验证方法：@measure_performance(iterations=10)
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            test_name = name or f.__name__
            execution_times = []
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                result = f(*args, **kwargs)
                end_time = time.perf_counter()
                execution_times.append((end_time - start_time) * 1000)
            
            avg_time = statistics.mean(execution_times)
            logging.info(f"性能测量 '{test_name}': 平均执行时间 {avg_time:.2f}ms")
            
            return result
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def performance_monitor(name: str = "operation"):
    """性能监控上下文管理器
    
    功能：在代码块执行期间监控性能
    参数：
        name: 操作名称
    边界条件：处理异常情况
    假设：代码块执行时间可测量
    不确定点：某些操作可能包含异步调用
    验证方法：with performance_monitor("test_operation"):
    """
    start_time = time.perf_counter()
    start_memory = psutil.Process().memory_info().rss
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss
        
        execution_time = (end_time - start_time) * 1000
        memory_delta = end_memory - start_memory
        
        logging.info(
            f"性能监控 '{name}': 执行时间 {execution_time:.2f}ms, "
            f"内存变化 {memory_delta / 1024 / 1024:.2f}MB"
        )


# 全局实例
system_monitor = SystemMonitor()
performance_tester = PerformanceTester()
benchmark_suite = BenchmarkSuite()