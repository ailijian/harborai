# -*- coding: utf-8 -*-
"""
渐进式负载测试

本模块实现了HarborAI项目的渐进式负载测试，包括：
- 负载渐进增长测试
- 多阶段负载测试
- 负载稳定性测试
- 负载恢复测试
- 厂商负载对比测试

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
from datetime import datetime, timedelta
import statistics
import queue
import random
import psutil
import gc
from contextlib import contextmanager

from tests.performance.load_tests import LOAD_TEST_CONFIG, LOAD_PERFORMANCE_GRADES
from tests.performance import PERFORMANCE_CONFIG, SUPPORTED_VENDORS


@dataclass
class LoadPhase:
    """
    负载阶段定义
    
    定义负载测试中的单个阶段
    """
    name: str
    concurrent_users: int
    requests_per_second: int
    duration_minutes: int
    ramp_up_seconds: int
    
    # 阶段状态
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    actual_duration: float = 0.0
    
    # 阶段指标
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    
    # 资源使用
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_percent: float = 0.0
    avg_memory_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        返回:
            Dict[str, Any]: 阶段数据字典
        """
        return {
            'name': self.name,
            'config': {
                'concurrent_users': self.concurrent_users,
                'requests_per_second': self.requests_per_second,
                'duration_minutes': self.duration_minutes,
                'ramp_up_seconds': self.ramp_up_seconds
            },
            'timing': {
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'actual_duration': self.actual_duration
            },
            'performance': {
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'avg_response_time': self.avg_response_time,
                'throughput': self.throughput,
                'error_rate': self.error_rate
            },
            'resources': {
                'peak_cpu_percent': self.peak_cpu_percent,
                'peak_memory_mb': self.peak_memory_mb,
                'avg_cpu_percent': self.avg_cpu_percent,
                'avg_memory_mb': self.avg_memory_mb
            }
        }


@dataclass
class GradualLoadResult:
    """
    渐进式负载测试结果
    
    记录渐进式负载测试的完整结果
    """
    test_name: str
    vendor: str
    model: str
    scenario: str
    
    # 测试配置
    phases: List[LoadPhase] = field(default_factory=list)
    total_duration: float = 0.0
    
    # 整体指标
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    overall_throughput: float = 0.0
    overall_error_rate: float = 0.0
    
    # 响应时间统计
    response_times: List[float] = field(default_factory=list)
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    avg_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # 负载稳定性指标
    load_stability_score: float = 0.0
    performance_degradation: float = 0.0
    recovery_time_seconds: float = 0.0
    
    # 资源使用峰值
    peak_cpu_percent: float = 0.0
    peak_memory_mb: float = 0.0
    resource_efficiency: float = 0.0
    
    # 性能等级
    load_grade: str = 'F'
    stability_grade: str = 'F'
    overall_grade: str = 'F'
    
    # 测试元数据
    test_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def calculate_metrics(self):
        """
        计算负载测试指标
        """
        if self.total_duration > 0:
            self.overall_throughput = self.successful_requests / self.total_duration
        
        if self.total_requests > 0:
            self.overall_error_rate = self.failed_requests / self.total_requests
        
        if self.response_times:
            self.response_times.sort()
            n = len(self.response_times)
            self.min_response_time = self.response_times[0]
            self.max_response_time = self.response_times[-1]
            self.avg_response_time = sum(self.response_times) / n
            
            if n >= 20:
                self.p95_response_time = self.response_times[int(n * 0.95)]
            if n >= 100:
                self.p99_response_time = self.response_times[int(n * 0.99)]
        
        # 计算负载稳定性
        self._calculate_stability_metrics()
        
        # 计算资源效率
        if self.peak_cpu_percent > 0 and self.peak_memory_mb > 0:
            # 资源效率 = 吞吐量 / (CPU使用率 * 内存使用率)
            resource_usage = (self.peak_cpu_percent / 100) * (self.peak_memory_mb / 1024)  # GB
            if resource_usage > 0:
                self.resource_efficiency = self.overall_throughput / resource_usage
    
    def _calculate_stability_metrics(self):
        """
        计算负载稳定性指标
        """
        if len(self.phases) < 2:
            return
        
        # 计算各阶段吞吐量变化
        phase_throughputs = [phase.throughput for phase in self.phases if phase.throughput > 0]
        
        if len(phase_throughputs) >= 2:
            # 稳定性评分：基于吞吐量变异系数
            mean_throughput = statistics.mean(phase_throughputs)
            if mean_throughput > 0:
                throughput_cv = statistics.stdev(phase_throughputs) / mean_throughput
                self.load_stability_score = max(0, 100 - throughput_cv * 100)
            
            # 性能退化：最高吞吐量与最低吞吐量的差异
            max_throughput = max(phase_throughputs)
            min_throughput = min(phase_throughputs)
            if max_throughput > 0:
                self.performance_degradation = (max_throughput - min_throughput) / max_throughput
        
        # 恢复时间：从最高负载阶段到稳定的时间
        if len(self.phases) >= 3:
            # 简化计算：假设最后阶段是恢复阶段
            last_phase = self.phases[-1]
            self.recovery_time_seconds = last_phase.ramp_up_seconds
    
    def evaluate_performance(self):
        """
        评估负载性能等级
        """
        thresholds = LOAD_TEST_CONFIG['performance_thresholds']
        
        # 负载性能评分
        load_score = 0
        
        # 吞吐量评分 (40%)
        if self.overall_throughput >= thresholds['throughput']['optimal_rps']:
            load_score += 40
        elif self.overall_throughput >= thresholds['throughput']['target_rps']:
            load_score += 30
        elif self.overall_throughput >= thresholds['throughput']['min_rps']:
            load_score += 20
        
        # 响应时间评分 (30%)
        if self.avg_response_time <= thresholds['response_time']['excellent_ms'] / 1000:
            load_score += 30
        elif self.avg_response_time <= thresholds['response_time']['good_ms'] / 1000:
            load_score += 25
        elif self.avg_response_time <= thresholds['response_time']['acceptable_ms'] / 1000:
            load_score += 15
        
        # 错误率评分 (20%)
        if self.overall_error_rate <= thresholds['error_rate']['excellent']:
            load_score += 20
        elif self.overall_error_rate <= thresholds['error_rate']['target']:
            load_score += 15
        elif self.overall_error_rate <= thresholds['error_rate']['max_acceptable']:
            load_score += 10
        
        # 资源效率评分 (10%)
        if self.resource_efficiency >= 1.0:
            load_score += 10
        elif self.resource_efficiency >= 0.5:
            load_score += 7
        elif self.resource_efficiency >= 0.2:
            load_score += 5
        
        # 确定负载等级
        for grade, info in LOAD_PERFORMANCE_GRADES.items():
            if load_score >= info['min_score']:
                self.load_grade = grade
                break
        
        # 稳定性等级
        if self.load_stability_score >= 90:
            self.stability_grade = 'A+'
        elif self.load_stability_score >= 80:
            self.stability_grade = 'A'
        elif self.load_stability_score >= 70:
            self.stability_grade = 'B'
        elif self.load_stability_score >= 60:
            self.stability_grade = 'C'
        elif self.load_stability_score >= 50:
            self.stability_grade = 'D'
        else:
            self.stability_grade = 'F'
        
        # 综合等级（负载性能权重70%，稳定性权重30%）
        combined_score = load_score * 0.7 + self.load_stability_score * 0.3
        
        for grade, info in LOAD_PERFORMANCE_GRADES.items():
            if combined_score >= info['min_score']:
                self.overall_grade = grade
                break
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        返回:
            Dict[str, Any]: 负载测试结果字典
        """
        return {
            'test_info': {
                'test_name': self.test_name,
                'vendor': self.vendor,
                'model': self.model,
                'scenario': self.scenario,
                'test_timestamp': self.test_timestamp
            },
            'phases': [phase.to_dict() for phase in self.phases],
            'overall_metrics': {
                'total_duration': self.total_duration,
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'overall_throughput': self.overall_throughput,
                'overall_error_rate': self.overall_error_rate
            },
            'response_time_stats': {
                'min': self.min_response_time,
                'max': self.max_response_time,
                'avg': self.avg_response_time,
                'p95': self.p95_response_time,
                'p99': self.p99_response_time
            },
            'stability_metrics': {
                'load_stability_score': self.load_stability_score,
                'performance_degradation': self.performance_degradation,
                'recovery_time_seconds': self.recovery_time_seconds
            },
            'resource_usage': {
                'peak_cpu_percent': self.peak_cpu_percent,
                'peak_memory_mb': self.peak_memory_mb,
                'resource_efficiency': self.resource_efficiency
            },
            'performance_grades': {
                'load_grade': self.load_grade,
                'stability_grade': self.stability_grade,
                'overall_grade': self.overall_grade
            }
        }


class MockLoadTestAPI:
    """
    模拟负载测试API客户端
    
    用于渐进式负载测试的模拟API客户端
    """
    
    def __init__(self, vendor: str, model: str):
        self.vendor = vendor
        self.model = model
        self.request_count = 0
        self.active_requests = 0
        self.lock = threading.Lock()
        
        # 负载特性配置
        self.load_config = self._get_load_config()
        
        # 性能退化模拟
        self.performance_degradation_factor = 1.0
        self.overload_threshold = 100  # 超过此并发数开始性能退化
    
    def _get_load_config(self) -> Dict[str, Any]:
        """
        获取负载配置
        
        返回:
            Dict[str, Any]: 负载配置
        """
        configs = {
            'deepseek': {
                'deepseek-chat': {
                    'base_response_time': 0.6,
                    'load_factor': 0.01,  # 每增加1个并发请求增加1%延迟
                    'max_stable_load': 80,
                    'degradation_rate': 0.02
                },
                'deepseek-r1': {
                    'base_response_time': 1.5,
                    'load_factor': 0.015,
                    'max_stable_load': 60,
                    'degradation_rate': 0.025
                }
            },
            'ernie': {
                'ernie-3.5-8k': {
                    'base_response_time': 0.5,
                    'load_factor': 0.008,
                    'max_stable_load': 100,
                    'degradation_rate': 0.015
                },
                'ernie-4.0-turbo-8k': {
                    'base_response_time': 0.8,
                    'load_factor': 0.012,
                    'max_stable_load': 70,
                    'degradation_rate': 0.02
                }
            },
            'doubao': {
                'doubao-1-5-pro-32k-character-250715': {
                    'base_response_time': 1.0,
                    'load_factor': 0.015,
                    'max_stable_load': 65,
                    'degradation_rate': 0.03
                }
            }
        }
        
        return configs.get(self.vendor, {}).get(self.model, {
            'base_response_time': 1.0,
            'load_factor': 0.02,
            'max_stable_load': 50,
            'degradation_rate': 0.025
        })
    
    def update_load_conditions(self, current_load: int):
        """
        更新负载条件
        
        参数:
            current_load: 当前负载级别
        """
        # 模拟性能退化
        if current_load > self.load_config['max_stable_load']:
            excess_load = current_load - self.load_config['max_stable_load']
            self.performance_degradation_factor = 1.0 + (excess_load * self.load_config['degradation_rate'])
        else:
            self.performance_degradation_factor = 1.0
    
    def send_request(self, request_id: str = None, current_load: int = 1) -> Dict[str, Any]:
        """
        发送负载测试请求
        
        参数:
            request_id: 请求ID
            current_load: 当前负载级别
        
        返回:
            Dict[str, Any]: 模拟响应
        """
        start_time = time.time()
        
        with self.lock:
            self.request_count += 1
            self.active_requests += 1
            current_count = self.request_count
        
        try:
            # 更新负载条件
            self.update_load_conditions(current_load)
            
            # 计算响应时间
            base_time = self.load_config['base_response_time']
            load_impact = 1.0 + (current_load * self.load_config['load_factor'])
            
            response_time = base_time * load_impact * self.performance_degradation_factor
            response_time = max(0.05, response_time)  # 最小响应时间50ms
            
            # 模拟负载下的错误率
            error_probability = 0.0
            if current_load > self.load_config['max_stable_load']:
                excess_load = current_load - self.load_config['max_stable_load']
                error_probability = min(0.1, excess_load * 0.001)  # 最高10%错误率
            
            # 模拟网络延迟和处理时间
            time.sleep(response_time)
            
            # 模拟错误
            if random.random() < error_probability:
                raise Exception(f"负载过高导致的模拟错误 (负载: {current_load})")
            
            return {
                'vendor': self.vendor,
                'model': self.model,
                'response_time': response_time,
                'content': f"负载测试响应 #{current_count} 来自 {self.vendor}/{self.model}",
                'request_id': request_id or f"{self.vendor}_{self.model}_load_{current_count}",
                'timestamp': time.time(),
                'current_load': current_load,
                'performance_factor': self.performance_degradation_factor,
                'success': True
            }
        
        except Exception as e:
            return {
                'vendor': self.vendor,
                'model': self.model,
                'error': str(e),
                'request_id': request_id or f"{self.vendor}_{self.model}_error_{current_count}",
                'timestamp': time.time(),
                'current_load': current_load,
                'success': False
            }
        
        finally:
            with self.lock:
                self.active_requests -= 1
    
    def get_current_stats(self) -> Dict[str, Any]:
        """
        获取当前统计信息
        
        返回:
            Dict[str, Any]: 当前统计
        """
        with self.lock:
            return {
                'total_requests': self.request_count,
                'active_requests': self.active_requests,
                'performance_degradation_factor': self.performance_degradation_factor,
                'max_stable_load': self.load_config['max_stable_load']
            }


class SystemResourceMonitor:
    """
    系统资源监控器
    
    监控负载测试期间的系统资源使用情况
    """
    
    def __init__(self):
        self.monitoring = False
        self.cpu_samples = []
        self.memory_samples = []
        self.monitor_thread = None
        self.sample_interval = 1.0  # 秒
    
    def start_monitoring(self):
        """
        开始资源监控
        """
        if self.monitoring:
            return
        
        self.monitoring = True
        self.cpu_samples.clear()
        self.memory_samples.clear()
        
        def monitor_loop():
            while self.monitoring:
                try:
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory_info = psutil.virtual_memory()
                    
                    self.cpu_samples.append(cpu_percent)
                    self.memory_samples.append(memory_info.used / 1024 / 1024)  # MB
                    
                    time.sleep(self.sample_interval)
                except Exception as e:
                    print(f"资源监控错误: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """
        停止资源监控
        
        返回:
            Dict[str, float]: 资源使用统计
        """
        self.monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        stats = {
            'peak_cpu_percent': 0.0,
            'avg_cpu_percent': 0.0,
            'peak_memory_mb': 0.0,
            'avg_memory_mb': 0.0
        }
        
        if self.cpu_samples:
            stats['peak_cpu_percent'] = max(self.cpu_samples)
            stats['avg_cpu_percent'] = statistics.mean(self.cpu_samples)
        
        if self.memory_samples:
            stats['peak_memory_mb'] = max(self.memory_samples)
            stats['avg_memory_mb'] = statistics.mean(self.memory_samples)
        
        return stats


class GradualLoadTestRunner:
    """
    渐进式负载测试运行器
    
    执行渐进式负载测试
    """
    
    def __init__(self):
        self.config = LOAD_TEST_CONFIG
        self.resource_monitor = SystemResourceMonitor()
        self.results: List[GradualLoadResult] = []
    
    def create_load_phases(self, scenario: str) -> List[LoadPhase]:
        """
        创建负载阶段
        
        参数:
            scenario: 测试场景名称
        
        返回:
            List[LoadPhase]: 负载阶段列表
        """
        scenario_config = self.config['test_scenarios'].get(scenario)
        if not scenario_config:
            raise ValueError(f"未知的测试场景: {scenario}")
        
        phases = []
        
        if scenario == 'gradual_ramp':
            # 渐进式负载增长
            for phase_name in scenario_config['phases']:
                load_config = self.config['load_levels'][phase_name]
                phase = LoadPhase(
                    name=phase_name,
                    concurrent_users=load_config['concurrent_users'],
                    requests_per_second=load_config['requests_per_second'],
                    duration_minutes=scenario_config['phase_duration_minutes'],
                    ramp_up_seconds=load_config['ramp_up_seconds']
                )
                phases.append(phase)
        
        elif scenario == 'spike_test':
            # 突发负载测试
            base_config = self.config['load_levels'][scenario_config['base_load']]
            spike_config = self.config['load_levels'][scenario_config['spike_load']]
            
            # 基础负载阶段
            phases.append(LoadPhase(
                name='base_load',
                concurrent_users=base_config['concurrent_users'],
                requests_per_second=base_config['requests_per_second'],
                duration_minutes=scenario_config['spike_interval_minutes'],
                ramp_up_seconds=base_config['ramp_up_seconds']
            ))
            
            # 突发负载阶段
            phases.append(LoadPhase(
                name='spike_load',
                concurrent_users=spike_config['concurrent_users'],
                requests_per_second=spike_config['requests_per_second'],
                duration_minutes=scenario_config['spike_duration_minutes'],
                ramp_up_seconds=30  # 快速突发
            ))
            
            # 恢复阶段
            phases.append(LoadPhase(
                name='recovery',
                concurrent_users=base_config['concurrent_users'],
                requests_per_second=base_config['requests_per_second'],
                duration_minutes=scenario_config['spike_interval_minutes'],
                ramp_up_seconds=60
            ))
        
        return phases
    
    def run_load_phase(self,
                      client: MockLoadTestAPI,
                      phase: LoadPhase) -> LoadPhase:
        """
        运行单个负载阶段
        
        参数:
            client: API客户端
            phase: 负载阶段
        
        返回:
            LoadPhase: 更新后的负载阶段
        """
        print(f"\n开始负载阶段: {phase.name}")
        print(f"  并发用户: {phase.concurrent_users}")
        print(f"  目标RPS: {phase.requests_per_second}")
        print(f"  持续时间: {phase.duration_minutes}分钟")
        print(f"  爬坡时间: {phase.ramp_up_seconds}秒")
        
        phase.start_time = datetime.now()
        start_time = time.time()
        
        # 计算总请求数
        total_requests = phase.requests_per_second * phase.duration_minutes * 60
        request_interval = 1.0 / phase.requests_per_second if phase.requests_per_second > 0 else 1.0
        
        response_times = []
        successful_count = 0
        failed_count = 0
        
        # 使用线程池模拟并发用户
        with ThreadPoolExecutor(max_workers=phase.concurrent_users) as executor:
            futures = []
            
            # 提交请求
            for i in range(total_requests):
                if time.time() - start_time >= phase.duration_minutes * 60:
                    break
                
                future = executor.submit(
                    client.send_request,
                    f"{phase.name}_{i}",
                    phase.concurrent_users
                )
                futures.append(future)
                
                # 控制请求速率
                time.sleep(request_interval)
            
            # 收集结果
            for future in as_completed(futures):
                try:
                    response = future.result(timeout=30)
                    
                    if response.get('success', False):
                        successful_count += 1
                        response_times.append(response['response_time'])
                    else:
                        failed_count += 1
                
                except Exception as e:
                    failed_count += 1
                    print(f"请求执行失败: {e}")
        
        end_time = time.time()
        phase.end_time = datetime.now()
        phase.actual_duration = end_time - start_time
        
        # 更新阶段指标
        phase.total_requests = successful_count + failed_count
        phase.successful_requests = successful_count
        phase.failed_requests = failed_count
        
        if phase.actual_duration > 0:
            phase.throughput = successful_count / phase.actual_duration
        
        if phase.total_requests > 0:
            phase.error_rate = failed_count / phase.total_requests
        
        if response_times:
            phase.avg_response_time = statistics.mean(response_times)
        
        print(f"阶段 {phase.name} 完成:")
        print(f"  实际持续时间: {phase.actual_duration:.1f}秒")
        print(f"  总请求数: {phase.total_requests}")
        print(f"  成功请求: {phase.successful_requests}")
        print(f"  失败请求: {phase.failed_requests}")
        print(f"  吞吐量: {phase.throughput:.2f} req/s")
        print(f"  错误率: {phase.error_rate:.2%}")
        print(f"  平均响应时间: {phase.avg_response_time*1000:.1f}ms")
        
        return phase
    
    def run_gradual_load_test(self,
                            vendor: str,
                            model: str,
                            scenario: str = 'gradual_ramp',
                            test_name: str = None) -> GradualLoadResult:
        """
        运行渐进式负载测试
        
        参数:
            vendor: 厂商名称
            model: 模型名称
            scenario: 测试场景
            test_name: 测试名称
        
        返回:
            GradualLoadResult: 负载测试结果
        """
        if not test_name:
            test_name = f"gradual_load_{scenario}"
        
        print(f"\n=== 开始渐进式负载测试 ===")
        print(f"厂商: {vendor}")
        print(f"模型: {model}")
        print(f"场景: {scenario}")
        
        # 创建负载阶段
        phases = self.create_load_phases(scenario)
        
        # 创建API客户端
        client = MockLoadTestAPI(vendor, model)
        
        # 创建结果对象
        result = GradualLoadResult(
            test_name=test_name,
            vendor=vendor,
            model=model,
            scenario=scenario
        )
        
        # 开始资源监控
        self.resource_monitor.start_monitoring()
        
        test_start_time = time.time()
        
        try:
            # 执行各个阶段
            for phase in phases:
                completed_phase = self.run_load_phase(client, phase)
                result.phases.append(completed_phase)
                
                # 累计统计
                result.total_requests += completed_phase.total_requests
                result.successful_requests += completed_phase.successful_requests
                result.failed_requests += completed_phase.failed_requests
                
                # 收集响应时间（简化处理）
                if completed_phase.avg_response_time > 0:
                    # 模拟响应时间分布
                    for _ in range(min(100, completed_phase.successful_requests)):
                        # 基于平均响应时间生成模拟分布
                        simulated_time = completed_phase.avg_response_time * random.uniform(0.5, 1.5)
                        result.response_times.append(simulated_time)
                
                # 阶段间休息
                if phase != phases[-1]:  # 不是最后一个阶段
                    print(f"阶段间休息 30 秒...")
                    time.sleep(30)
        
        finally:
            # 停止资源监控
            resource_stats = self.resource_monitor.stop_monitoring()
            
            result.peak_cpu_percent = resource_stats['peak_cpu_percent']
            result.peak_memory_mb = resource_stats['peak_memory_mb']
            
            # 更新阶段资源信息
            for phase in result.phases:
                phase.peak_cpu_percent = resource_stats['peak_cpu_percent']
                phase.peak_memory_mb = resource_stats['peak_memory_mb']
                phase.avg_cpu_percent = resource_stats['avg_cpu_percent']
                phase.avg_memory_mb = resource_stats['avg_memory_mb']
        
        test_end_time = time.time()
        result.total_duration = test_end_time - test_start_time
        
        # 计算指标和性能等级
        result.calculate_metrics()
        result.evaluate_performance()
        
        print(f"\n=== 渐进式负载测试完成 ===")
        print(f"总持续时间: {result.total_duration:.1f}秒")
        print(f"总请求数: {result.total_requests}")
        print(f"成功请求: {result.successful_requests}")
        print(f"整体吞吐量: {result.overall_throughput:.2f} req/s")
        print(f"整体错误率: {result.overall_error_rate:.2%}")
        print(f"负载稳定性评分: {result.load_stability_score:.1f}")
        print(f"性能等级: {result.overall_grade} (负载: {result.load_grade}, 稳定性: {result.stability_grade})")
        
        self.results.append(result)
        return result
    
    def generate_load_test_report(self, results: List[GradualLoadResult]) -> Dict[str, Any]:
        """
        生成负载测试报告
        
        参数:
            results: 负载测试结果列表
        
        返回:
            Dict[str, Any]: 负载测试报告
        """
        if not results:
            return {'error': '没有负载测试结果'}
        
        report = {
            'summary': {
                'total_tests': len(results),
                'test_timestamp': datetime.now().isoformat(),
                'scenarios': list(set(r.scenario for r in results)),
                'avg_throughput': 0.0,
                'avg_stability_score': 0.0
            },
            'results': [result.to_dict() for result in results],
            'analysis': {
                'best_performance': None,
                'most_stable': None,
                'scenario_comparison': {}
            },
            'recommendations': []
        }
        
        # 统计分析
        throughputs = [r.overall_throughput for r in results]
        stability_scores = [r.load_stability_score for r in results]
        
        if throughputs:
            report['summary']['avg_throughput'] = statistics.mean(throughputs)
        if stability_scores:
            report['summary']['avg_stability_score'] = statistics.mean(stability_scores)
        
        # 最佳性能
        if results:
            best_perf = max(results, key=lambda x: x.overall_throughput)
            report['analysis']['best_performance'] = {
                'test_name': best_perf.test_name,
                'vendor': best_perf.vendor,
                'model': best_perf.model,
                'throughput': best_perf.overall_throughput,
                'grade': best_perf.overall_grade
            }
            
            most_stable = max(results, key=lambda x: x.load_stability_score)
            report['analysis']['most_stable'] = {
                'test_name': most_stable.test_name,
                'vendor': most_stable.vendor,
                'model': most_stable.model,
                'stability_score': most_stable.load_stability_score,
                'stability_grade': most_stable.stability_grade
            }
        
        # 场景对比
        scenario_stats = {}
        for scenario in report['summary']['scenarios']:
            scenario_results = [r for r in results if r.scenario == scenario]
            if scenario_results:
                scenario_stats[scenario] = {
                    'count': len(scenario_results),
                    'avg_throughput': statistics.mean([r.overall_throughput for r in scenario_results]),
                    'avg_stability': statistics.mean([r.load_stability_score for r in scenario_results])
                }
        
        report['analysis']['scenario_comparison'] = scenario_stats
        
        # 生成建议
        poor_performance = [r for r in results if r.overall_grade in ['D', 'F']]
        if poor_performance:
            report['recommendations'].append(
                f"发现 {len(poor_performance)} 个性能不佳的配置，建议优化负载处理策略"
            )
        
        unstable_results = [r for r in results if r.load_stability_score < 70]
        if unstable_results:
            report['recommendations'].append(
                f"发现 {len(unstable_results)} 个稳定性较差的配置，建议检查负载均衡和资源分配"
            )
        
        return report
    
    def reset_results(self):
        """重置测试结果"""
        self.results.clear()


class TestGradualLoad:
    """
    渐进式负载测试类
    
    包含各种渐进式负载测试场景
    """
    
    def setup_method(self):
        """测试方法设置"""
        self.load_runner = GradualLoadTestRunner()
        self.config = LOAD_TEST_CONFIG
    
    def teardown_method(self):
        """测试方法清理"""
        self.load_runner.reset_results()
        # 强制垃圾回收
        gc.collect()
    
    def _print_load_summary(self, results: List[GradualLoadResult]):
        """打印负载测试摘要"""
        if not results:
            print("\n没有负载测试结果")
            return
        
        print(f"\n=== 渐进式负载测试结果摘要 ===")
        print(f"测试数量: {len(results)}")
        
        for result in results:
            print(f"\n{result.test_name} - {result.vendor}/{result.model}:")
            print(f"  场景: {result.scenario}")
            print(f"  阶段数: {len(result.phases)}")
            print(f"  总持续时间: {result.total_duration:.1f}秒")
            print(f"  整体等级: {result.overall_grade} (负载: {result.load_grade}, 稳定性: {result.stability_grade})")
            print(f"  整体吞吐量: {result.overall_throughput:.2f} req/s")
            print(f"  整体错误率: {result.overall_error_rate:.2%}")
            print(f"  稳定性评分: {result.load_stability_score:.1f}")
            print(f"  资源效率: {result.resource_efficiency:.2f}")
            print(f"  峰值CPU: {result.peak_cpu_percent:.1f}%")
            print(f"  峰值内存: {result.peak_memory_mb:.1f}MB")
    
    @pytest.mark.load_test
    @pytest.mark.quick_load
    def test_gradual_ramp_load(self):
        """
        渐进式负载增长测试
        
        测试系统在负载逐步增长时的性能表现
        """
        vendor = 'deepseek'
        model = 'deepseek-chat'
        scenario = 'gradual_ramp'
        
        # 运行渐进式负载测试
        result = self.load_runner.run_gradual_load_test(
            vendor, model, scenario, "test_gradual_ramp"
        )
        
        self._print_load_summary([result])
        
        # 基本断言
        assert result.total_requests > 0
        assert result.successful_requests > 0
        assert len(result.phases) >= 2  # 至少有两个阶段
        assert result.scenario == scenario
        assert result.overall_grade in ['A+', 'A', 'B', 'C', 'D', 'F']
        
        # 负载测试特定断言
        assert result.overall_throughput > 0
        assert result.load_stability_score >= 0
        assert result.total_duration > 0
        
        # 性能要求
        thresholds = self.config['performance_thresholds']
        assert result.overall_error_rate <= thresholds['error_rate']['max_acceptable']
        assert result.overall_throughput >= thresholds['throughput']['min_rps']
    
    @pytest.mark.load_test
    @pytest.mark.standard_load
    def test_spike_load(self):
        """
        突发负载测试
        
        测试系统在突发负载下的性能表现和恢复能力
        """
        vendor = 'ernie'
        model = 'ernie-3.5-8k'
        scenario = 'spike_test'
        
        # 运行突发负载测试
        result = self.load_runner.run_gradual_load_test(
            vendor, model, scenario, "test_spike_load"
        )
        
        self._print_load_summary([result])
        
        # 基本断言
        assert result.total_requests > 0
        assert result.successful_requests > 0
        assert len(result.phases) == 3  # 基础-突发-恢复三个阶段
        assert result.scenario == scenario
        
        # 突发测试特定断言
        phase_names = [phase.name for phase in result.phases]
        assert 'base_load' in phase_names
        assert 'spike_load' in phase_names
        assert 'recovery' in phase_names
        
        # 检查突发阶段的性能影响
        spike_phase = next(p for p in result.phases if p.name == 'spike_load')
        base_phase = next(p for p in result.phases if p.name == 'base_load')
        
        # 突发阶段的并发数应该更高
        assert spike_phase.concurrent_users > base_phase.concurrent_users
        
        # 恢复时间应该合理
        assert result.recovery_time_seconds >= 0
    
    @pytest.mark.load_test
    @pytest.mark.comprehensive_load
    def test_multi_vendor_load_comparison(self):
        """
        多厂商负载对比测试
        
        对比不同厂商在相同负载下的性能表现
        """
        vendors_models = [
            ('deepseek', 'deepseek-chat'),
            ('ernie', 'ernie-3.5-8k'),
            ('doubao', 'doubao-1-5-pro-32k-character-250715')
        ]
        scenario = 'gradual_ramp'
        
        all_results = []
        
        for vendor, model in vendors_models:
            print(f"\n测试厂商: {vendor}/{model}")
            
            result = self.load_runner.run_gradual_load_test(
                vendor, model, scenario, f"comparison_{vendor}_{model}"
            )
            all_results.append(result)
        
        self._print_load_summary(all_results)
        
        # 生成对比报告
        report = self.load_runner.generate_load_test_report(all_results)
        
        print(f"\n=== 多厂商负载对比报告 ===")
        print(f"参与对比的厂商: {len(vendors_models)}")
        print(f"平均吞吐量: {report['summary']['avg_throughput']:.2f} req/s")
        print(f"平均稳定性评分: {report['summary']['avg_stability_score']:.1f}")
        
        if report['analysis']['best_performance']:
            best = report['analysis']['best_performance']
            print(f"最佳性能: {best['vendor']}/{best['model']} ({best['throughput']:.2f} req/s, {best['grade']})")
        
        if report['analysis']['most_stable']:
            stable = report['analysis']['most_stable']
            print(f"最稳定: {stable['vendor']}/{stable['model']} (稳定性: {stable['stability_score']:.1f}, {stable['stability_grade']})")
        
        # 对比断言
        assert len(all_results) == len(vendors_models)
        assert all(r.total_requests > 0 for r in all_results)
        
        # 至少有一个厂商达到可接受的性能
        acceptable_results = [r for r in all_results if r.overall_grade in ['A+', 'A', 'B', 'C']]
        assert len(acceptable_results) > 0, "没有厂商达到可接受的负载性能水平"
        
        # 性能差异应该在合理范围内
        throughputs = [r.overall_throughput for r in all_results]
        max_throughput = max(throughputs)
        min_throughput = min(throughputs)
        
        # 最大吞吐量不应该是最小吞吐量的10倍以上（避免极端差异）
        if min_throughput > 0:
            assert max_throughput / min_throughput <= 10, "厂商间性能差异过大"
    
    @pytest.mark.load_test
    @pytest.mark.comprehensive_load
    def test_load_stability_analysis(self):
        """
        负载稳定性分析测试
        
        深入分析系统在不同负载阶段的稳定性
        """
        vendor = 'deepseek'
        model = 'deepseek-r1'
        scenario = 'gradual_ramp'
        
        # 运行负载测试
        result = self.load_runner.run_gradual_load_test(
            vendor, model, scenario, "stability_analysis"
        )
        
        self._print_load_summary([result])
        
        print(f"\n=== 负载稳定性详细分析 ===")
        print(f"稳定性评分: {result.load_stability_score:.1f}")
        print(f"性能退化率: {result.performance_degradation:.2%}")
        print(f"恢复时间: {result.recovery_time_seconds:.1f}秒")
        
        # 分析各阶段性能
        print(f"\n各阶段性能分析:")
        for i, phase in enumerate(result.phases):
            print(f"  阶段 {i+1} ({phase.name}):")
            print(f"    吞吐量: {phase.throughput:.2f} req/s")
            print(f"    错误率: {phase.error_rate:.2%}")
            print(f"    平均响应时间: {phase.avg_response_time*1000:.1f}ms")
            print(f"    CPU峰值: {phase.peak_cpu_percent:.1f}%")
            print(f"    内存峰值: {phase.peak_memory_mb:.1f}MB")
        
        # 稳定性断言
        assert result.load_stability_score >= 0
        assert result.performance_degradation >= 0
        assert result.recovery_time_seconds >= 0
        
        # 稳定性要求
        assert result.load_stability_score >= 30, "负载稳定性评分过低"
        assert result.performance_degradation <= 0.5, "性能退化率过高"
        
        # 各阶段都应该有合理的性能
        for phase in result.phases:
            assert phase.throughput >= 0, f"阶段 {phase.name} 吞吐量异常"
            assert phase.error_rate <= 0.2, f"阶段 {phase.name} 错误率过高"
    
    @pytest.mark.load_test
    @pytest.mark.performance_load
    def test_load_performance_benchmark(self, benchmark):
        """
        负载性能基准测试
        
        使用pytest-benchmark进行精确的负载性能测量
        """
        vendor = 'doubao'
        model = 'doubao-1-5-pro-32k-character-250715'
        
        def load_test_function():
            """被基准测试的函数"""
            return self.load_runner.run_gradual_load_test(
                vendor, model, 'gradual_ramp', "benchmark_load_test"
            )
        
        # 使用pytest-benchmark运行基准测试
        result = benchmark(load_test_function)
        
        self._print_load_summary([result])
        
        print(f"\n=== pytest-benchmark 负载统计 ===")
        print(f"基准测试函数: {load_test_function.__name__}")
        
        # pytest-benchmark断言
        assert result.total_requests > 0
        assert result.successful_requests > 0
        assert result.overall_throughput > 0
        
        # 负载性能基准断言
        thresholds = self.config['performance_thresholds']
        assert result.overall_throughput >= thresholds['throughput']['min_rps']
        assert result.overall_error_rate <= thresholds['error_rate']['max_acceptable']
        assert result.load_stability_score >= 40  # 至少40分的稳定性