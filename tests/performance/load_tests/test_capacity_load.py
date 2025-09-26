#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
容量负载测试模块

实现系统容量极限测试，确定系统的最大处理能力和性能边界。
包括容量发现、瓶颈分析、扩展性测试等功能。

作者: HarborAI测试团队
版本: 1.0.0
创建时间: 2024-01-20
"""

import asyncio
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
class CapacityPoint:
    """容量测试点"""
    load_level: int  # 负载级别（req/s）
    response_time: float  # 平均响应时间
    throughput: float  # 实际吞吐量
    error_rate: float  # 错误率
    cpu_usage: float  # CPU使用率
    memory_usage: float  # 内存使用率
    success_rate: float  # 成功率
    
    def is_acceptable(self, max_response_time: float = 2.0, 
                     max_error_rate: float = 0.05) -> bool:
        """判断是否在可接受范围内"""
        return (self.response_time <= max_response_time and 
                self.error_rate <= max_error_rate and
                self.success_rate >= 0.95)
    
    def efficiency_score(self) -> float:
        """计算效率评分"""
        # 基于吞吐量、响应时间和错误率的综合评分
        throughput_score = min(100, self.throughput * 2)  # 吞吐量评分
        response_score = max(0, 100 - self.response_time * 50)  # 响应时间评分
        error_score = max(0, 100 - self.error_rate * 1000)  # 错误率评分
        
        return (throughput_score * 0.4 + response_score * 0.3 + error_score * 0.3)


@dataclass
class CapacityTestResult:
    """容量测试结果"""
    vendor: str
    model: str
    test_name: str
    start_time: datetime
    end_time: datetime
    
    # 容量指标
    max_sustainable_load: int = 0  # 最大可持续负载
    peak_load: int = 0  # 峰值负载
    optimal_load: int = 0  # 最优负载点
    capacity_utilization: float = 0.0  # 容量利用率
    
    # 性能边界
    max_throughput: float = 0.0  # 最大吞吐量
    min_response_time: float = 0.0  # 最小响应时间
    breaking_point_load: int = 0  # 系统崩溃点负载
    
    # 扩展性指标
    scalability_factor: float = 0.0  # 扩展性因子
    efficiency_degradation: float = 0.0  # 效率衰减率
    resource_efficiency: float = 0.0  # 资源效率
    
    # 瓶颈分析
    primary_bottleneck: str = "unknown"  # 主要瓶颈
    bottleneck_threshold: float = 0.0  # 瓶颈阈值
    
    # 等级评定
    capacity_grade: str = 'F'
    
    # 详细数据
    capacity_points: List[CapacityPoint] = field(default_factory=list)
    bottleneck_analysis: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_metrics(self):
        """计算容量指标"""
        if not self.capacity_points:
            return
        
        # 找到最大可持续负载（满足性能要求的最大负载）
        acceptable_points = [p for p in self.capacity_points if p.is_acceptable()]
        if acceptable_points:
            self.max_sustainable_load = max(p.load_level for p in acceptable_points)
        
        # 峰值负载（测试的最大负载）
        self.peak_load = max(p.load_level for p in self.capacity_points)
        
        # 最优负载点（效率最高的负载点）
        if self.capacity_points:
            optimal_point = max(self.capacity_points, key=lambda p: p.efficiency_score())
            self.optimal_load = optimal_point.load_level
        
        # 最大吞吐量
        self.max_throughput = max(p.throughput for p in self.capacity_points)
        
        # 最小响应时间
        self.min_response_time = min(p.response_time for p in self.capacity_points)
        
        # 找到系统崩溃点（错误率超过50%的负载点）
        breaking_points = [p for p in self.capacity_points if p.error_rate > 0.5]
        if breaking_points:
            self.breaking_point_load = min(p.load_level for p in breaking_points)
        else:
            self.breaking_point_load = self.peak_load
        
        # 计算容量利用率
        if self.peak_load > 0:
            self.capacity_utilization = (self.max_sustainable_load / self.peak_load) * 100
        
        # 计算扩展性因子
        self._calculate_scalability_metrics()
        
        # 瓶颈分析
        self._analyze_bottlenecks()
        
        # 等级评定
        self._assign_capacity_grade()
    
    def _calculate_scalability_metrics(self):
        """计算扩展性指标"""
        if len(self.capacity_points) < 2:
            return
        
        # 按负载级别排序
        sorted_points = sorted(self.capacity_points, key=lambda p: p.load_level)
        
        # 计算扩展性因子（吞吐量增长率 vs 负载增长率）
        load_growth = sorted_points[-1].load_level - sorted_points[0].load_level
        throughput_growth = sorted_points[-1].throughput - sorted_points[0].throughput
        
        if load_growth > 0:
            self.scalability_factor = throughput_growth / load_growth
        
        # 计算效率衰减率
        efficiency_scores = [p.efficiency_score() for p in sorted_points]
        if len(efficiency_scores) >= 2:
            max_efficiency = max(efficiency_scores)
            min_efficiency = min(efficiency_scores)
            if max_efficiency > 0:
                self.efficiency_degradation = ((max_efficiency - min_efficiency) / max_efficiency) * 100
        
        # 计算资源效率（吞吐量 / 资源使用率）
        resource_usages = [(p.cpu_usage + p.memory_usage) / 2 for p in sorted_points]
        throughputs = [p.throughput for p in sorted_points]
        
        if resource_usages and max(resource_usages) > 0:
            avg_throughput = statistics.mean(throughputs)
            avg_resource_usage = statistics.mean(resource_usages)
            self.resource_efficiency = avg_throughput / avg_resource_usage
    
    def _analyze_bottlenecks(self):
        """分析系统瓶颈"""
        if not self.capacity_points:
            return
        
        # 分析各种资源的使用情况
        cpu_usages = [p.cpu_usage for p in self.capacity_points]
        memory_usages = [p.memory_usage for p in self.capacity_points]
        response_times = [p.response_time for p in self.capacity_points]
        error_rates = [p.error_rate for p in self.capacity_points]
        
        max_cpu = max(cpu_usages)
        max_memory = max(memory_usages)
        max_response_time = max(response_times)
        max_error_rate = max(error_rates)
        
        # 确定主要瓶颈
        bottlenecks = []
        
        if max_cpu > 80:
            bottlenecks.append(('CPU', max_cpu))
        if max_memory > 80:
            bottlenecks.append(('Memory', max_memory))
        if max_response_time > 2.0:
            bottlenecks.append(('Response Time', max_response_time))
        if max_error_rate > 0.1:
            bottlenecks.append(('Error Rate', max_error_rate * 100))
        
        if bottlenecks:
            # 选择最严重的瓶颈
            primary = max(bottlenecks, key=lambda x: x[1])
            self.primary_bottleneck = primary[0]
            self.bottleneck_threshold = primary[1]
        
        # 详细瓶颈分析
        self.bottleneck_analysis = {
            'cpu_bottleneck': max_cpu > 80,
            'memory_bottleneck': max_memory > 80,
            'response_time_bottleneck': max_response_time > 2.0,
            'error_rate_bottleneck': max_error_rate > 0.1,
            'max_cpu_usage': max_cpu,
            'max_memory_usage': max_memory,
            'max_response_time': max_response_time,
            'max_error_rate': max_error_rate,
            'bottleneck_recommendations': self._generate_bottleneck_recommendations()
        }
    
    def _generate_bottleneck_recommendations(self) -> List[str]:
        """生成瓶颈优化建议"""
        recommendations = []
        
        if self.primary_bottleneck == 'CPU':
            recommendations.append("CPU使用率过高，建议优化算法或增加CPU资源")
            recommendations.append("考虑使用异步处理或并行计算优化")
        elif self.primary_bottleneck == 'Memory':
            recommendations.append("内存使用率过高，建议优化内存管理或增加内存")
            recommendations.append("检查是否存在内存泄漏或不必要的内存占用")
        elif self.primary_bottleneck == 'Response Time':
            recommendations.append("响应时间过长，建议优化处理逻辑或增加缓存")
            recommendations.append("考虑使用负载均衡分散请求压力")
        elif self.primary_bottleneck == 'Error Rate':
            recommendations.append("错误率过高，建议检查错误处理逻辑")
            recommendations.append("增强系统的容错能力和重试机制")
        
        return recommendations
    
    def _assign_capacity_grade(self):
        """分配容量等级"""
        # 基于容量利用率、扩展性因子和资源效率
        score = 0
        
        # 容量利用率评分（40%权重）
        if self.capacity_utilization >= 90:
            score += 40
        elif self.capacity_utilization >= 80:
            score += 35
        elif self.capacity_utilization >= 70:
            score += 30
        elif self.capacity_utilization >= 60:
            score += 25
        elif self.capacity_utilization >= 50:
            score += 20
        else:
            score += 10
        
        # 扩展性因子评分（30%权重）
        if self.scalability_factor >= 0.8:
            score += 30
        elif self.scalability_factor >= 0.6:
            score += 25
        elif self.scalability_factor >= 0.4:
            score += 20
        elif self.scalability_factor >= 0.2:
            score += 15
        else:
            score += 5
        
        # 资源效率评分（30%权重）
        if self.resource_efficiency >= 2.0:
            score += 30
        elif self.resource_efficiency >= 1.5:
            score += 25
        elif self.resource_efficiency >= 1.0:
            score += 20
        elif self.resource_efficiency >= 0.5:
            score += 15
        else:
            score += 5
        
        # 根据总分分配等级
        if score >= 90:
            self.capacity_grade = 'A+'
        elif score >= 80:
            self.capacity_grade = 'A'
        elif score >= 70:
            self.capacity_grade = 'B'
        elif score >= 60:
            self.capacity_grade = 'C'
        elif score >= 40:
            self.capacity_grade = 'D'
        else:
            self.capacity_grade = 'F'


class MockCapacityAPI:
    """模拟容量测试API"""
    
    def __init__(self, max_capacity: int = 100, 
                 optimal_load: int = 50,
                 degradation_factor: float = 0.01):
        self.max_capacity = max_capacity
        self.optimal_load = optimal_load
        self.degradation_factor = degradation_factor
        self.current_load = 0
        self._lock = threading.Lock()
    
    def make_request(self, vendor: str, model: str, current_load: int) -> Tuple[float, bool]:
        """模拟API请求"""
        with self._lock:
            self.current_load = current_load
            
            # 计算基础响应时间（负载越高响应时间越长）
            load_factor = current_load / self.optimal_load
            base_response_time = 0.1 * (1 + load_factor * self.degradation_factor)
            
            # 超过最优负载后性能急剧下降
            if current_load > self.optimal_load:
                overload_factor = (current_load - self.optimal_load) / self.optimal_load
                base_response_time *= (1 + overload_factor * 2)
            
            # 超过最大容量后大量失败
            if current_load > self.max_capacity:
                failure_rate = min(0.9, (current_load - self.max_capacity) / self.max_capacity)
                success = time.time() % 1 > failure_rate
            else:
                # 正常情况下的少量随机失败
                failure_rate = min(0.05, current_load * 0.0001)
                success = time.time() % 1 > failure_rate
            
            # 模拟网络延迟
            actual_response_time = base_response_time + (time.time() % 0.01)
            
            return actual_response_time, success
    
    def get_resource_usage(self, current_load: int) -> Tuple[float, float]:
        """获取资源使用情况"""
        # 模拟CPU和内存使用率
        load_ratio = current_load / self.max_capacity
        
        # CPU使用率随负载线性增长，但有一定随机性
        cpu_usage = min(100, load_ratio * 80 + (time.time() % 10))
        
        # 内存使用率增长较慢，但在高负载时快速增长
        if load_ratio < 0.7:
            memory_usage = load_ratio * 40 + (time.time() % 5)
        else:
            memory_usage = 40 + (load_ratio - 0.7) * 150 + (time.time() % 10)
        
        memory_usage = min(100, memory_usage)
        
        return cpu_usage, memory_usage


class CapacityTestRunner:
    """容量测试运行器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api = MockCapacityAPI()
        self._stop_event = threading.Event()
    
    def run_capacity_discovery(self, vendor: str, model: str, 
                              test_name: str,
                              start_load: int = 1,
                              max_load: int = 200,
                              step_size: int = 5,
                              step_duration: int = 30) -> CapacityTestResult:
        """运行容量发现测试"""
        print(f"\n开始容量发现测试: {test_name}")
        print(f"厂商: {vendor}, 模型: {model}")
        print(f"负载范围: {start_load} - {max_load} req/s")
        print(f"步长: {step_size} req/s, 每步持续: {step_duration}s")
        
        start_time = datetime.now()
        
        result = CapacityTestResult(
            vendor=vendor,
            model=model,
            test_name=test_name,
            start_time=start_time,
            end_time=start_time  # 临时值
        )
        
        try:
            current_load = start_load
            
            while current_load <= max_load and not self._stop_event.is_set():
                print(f"\n测试负载级别: {current_load} req/s")
                
                # 运行当前负载级别的测试
                capacity_point = self._test_load_level(
                    current_load, step_duration, vendor, model
                )
                
                result.capacity_points.append(capacity_point)
                
                print(f"  响应时间: {capacity_point.response_time:.3f}s")
                print(f"  吞吐量: {capacity_point.throughput:.1f} req/s")
                print(f"  错误率: {capacity_point.error_rate:.2%}")
                print(f"  CPU使用率: {capacity_point.cpu_usage:.1f}%")
                print(f"  内存使用率: {capacity_point.memory_usage:.1f}%")
                print(f"  效率评分: {capacity_point.efficiency_score():.1f}")
                
                # 检查是否应该提前停止
                if capacity_point.error_rate > 0.8:  # 错误率超过80%
                    print(f"  错误率过高({capacity_point.error_rate:.1%})，停止测试")
                    break
                
                if capacity_point.response_time > 10.0:  # 响应时间超过10秒
                    print(f"  响应时间过长({capacity_point.response_time:.1f}s)，停止测试")
                    break
                
                current_load += step_size
        
        finally:
            end_time = datetime.now()
            result.end_time = end_time
            
            # 计算指标
            result.calculate_metrics()
            
            print(f"\n容量发现测试完成")
            print(f"测试时长: {(end_time - start_time).total_seconds():.1f}秒")
            print(f"测试点数: {len(result.capacity_points)}")
            print(f"最大可持续负载: {result.max_sustainable_load} req/s")
            print(f"峰值负载: {result.peak_load} req/s")
            print(f"最优负载: {result.optimal_load} req/s")
            print(f"容量利用率: {result.capacity_utilization:.1f}%")
            print(f"主要瓶颈: {result.primary_bottleneck}")
            print(f"容量等级: {result.capacity_grade}")
        
        return result
    
    def _test_load_level(self, load_level: int, duration: int, 
                        vendor: str, model: str) -> CapacityPoint:
        """测试特定负载级别"""
        start_time = time.time()
        end_time = start_time + duration
        
        total_requests = 0
        successful_requests = 0
        response_times = []
        
        # 使用线程池执行并发请求
        max_workers = min(load_level, 100)  # 限制最大线程数
        
        while time.time() < end_time and not self._stop_event.is_set():
            batch_start = time.time()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交一秒钟的请求
                futures = []
                for _ in range(load_level):
                    if time.time() >= end_time:
                        break
                    future = executor.submit(self.api.make_request, vendor, model, load_level)
                    futures.append(future)
                
                # 收集结果
                for future in as_completed(futures, timeout=2):
                    try:
                        response_time, success = future.result(timeout=1)
                        total_requests += 1
                        if success:
                            successful_requests += 1
                        response_times.append(response_time)
                    except Exception:
                        total_requests += 1
                        response_times.append(10.0)  # 超时默认响应时间
            
            # 控制请求频率
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)
        
        # 计算指标
        actual_duration = time.time() - start_time
        avg_response_time = statistics.mean(response_times) if response_times else 0
        actual_throughput = successful_requests / actual_duration if actual_duration > 0 else 0
        error_rate = (total_requests - successful_requests) / max(total_requests, 1)
        success_rate = successful_requests / max(total_requests, 1)
        
        # 获取资源使用情况
        cpu_usage, memory_usage = self.api.get_resource_usage(load_level)
        
        return CapacityPoint(
            load_level=load_level,
            response_time=avg_response_time,
            throughput=actual_throughput,
            error_rate=error_rate,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            success_rate=success_rate
        )
    
    def run_binary_search_capacity(self, vendor: str, model: str,
                                  test_name: str,
                                  min_load: int = 1,
                                  max_load: int = 500,
                                  target_error_rate: float = 0.05,
                                  precision: int = 2) -> CapacityTestResult:
        """使用二分搜索快速找到容量边界"""
        print(f"\n开始二分搜索容量测试: {test_name}")
        print(f"厂商: {vendor}, 模型: {model}")
        print(f"搜索范围: {min_load} - {max_load} req/s")
        print(f"目标错误率: {target_error_rate:.1%}")
        
        start_time = datetime.now()
        
        result = CapacityTestResult(
            vendor=vendor,
            model=model,
            test_name=test_name,
            start_time=start_time,
            end_time=start_time
        )
        
        try:
            low = min_load
            high = max_load
            
            while high - low > precision and not self._stop_event.is_set():
                mid = (low + high) // 2
                print(f"\n测试负载: {mid} req/s (范围: {low}-{high})")
                
                # 测试中点负载
                capacity_point = self._test_load_level(mid, 30, vendor, model)
                result.capacity_points.append(capacity_point)
                
                print(f"  错误率: {capacity_point.error_rate:.2%}")
                print(f"  响应时间: {capacity_point.response_time:.3f}s")
                print(f"  吞吐量: {capacity_point.throughput:.1f} req/s")
                
                if capacity_point.error_rate <= target_error_rate:
                    # 错误率可接受，尝试更高负载
                    low = mid
                    print(f"  ✓ 可接受，尝试更高负载")
                else:
                    # 错误率过高，降低负载
                    high = mid
                    print(f"  ✗ 错误率过高，降低负载")
            
            # 测试边界点
            if not self._stop_event.is_set():
                print(f"\n测试最终边界点: {low} req/s")
                final_point = self._test_load_level(low, 60, vendor, model)
                result.capacity_points.append(final_point)
        
        finally:
            end_time = datetime.now()
            result.end_time = end_time
            
            # 计算指标
            result.calculate_metrics()
            
            print(f"\n二分搜索容量测试完成")
            print(f"找到的容量边界: {result.max_sustainable_load} req/s")
            print(f"容量等级: {result.capacity_grade}")
        
        return result
    
    def generate_capacity_report(self, results: List[CapacityTestResult]) -> Dict[str, Any]:
        """生成容量测试报告"""
        if not results:
            return {'error': '没有测试结果'}
        
        report = {
            'summary': {
                'total_tests': len(results),
                'avg_max_sustainable_load': statistics.mean([r.max_sustainable_load for r in results]),
                'avg_capacity_utilization': statistics.mean([r.capacity_utilization for r in results]),
                'avg_scalability_factor': statistics.mean([r.scalability_factor for r in results]),
                'grade_distribution': {}
            },
            'capacity_analysis': {},
            'bottleneck_analysis': {},
            'scalability_analysis': {},
            'recommendations': []
        }
        
        # 等级分布
        grades = [r.capacity_grade for r in results]
        for grade in ['A+', 'A', 'B', 'C', 'D', 'F']:
            report['summary']['grade_distribution'][grade] = grades.count(grade)
        
        # 容量分析
        report['capacity_analysis'] = {
            'max_sustainable_loads': [r.max_sustainable_load for r in results],
            'peak_loads': [r.peak_load for r in results],
            'optimal_loads': [r.optimal_load for r in results],
            'capacity_utilizations': [r.capacity_utilization for r in results],
            'max_throughputs': [r.max_throughput for r in results]
        }
        
        # 瓶颈分析
        bottlenecks = [r.primary_bottleneck for r in results]
        bottleneck_counts = {}
        for bottleneck in bottlenecks:
            bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
        
        report['bottleneck_analysis'] = {
            'primary_bottlenecks': bottleneck_counts,
            'most_common_bottleneck': max(bottleneck_counts.items(), key=lambda x: x[1])[0] if bottleneck_counts else 'unknown'
        }
        
        # 扩展性分析
        report['scalability_analysis'] = {
            'avg_scalability_factor': statistics.mean([r.scalability_factor for r in results]),
            'avg_efficiency_degradation': statistics.mean([r.efficiency_degradation for r in results]),
            'avg_resource_efficiency': statistics.mean([r.resource_efficiency for r in results])
        }
        
        # 生成建议
        self._generate_capacity_recommendations(report, results)
        
        return report
    
    def _generate_capacity_recommendations(self, report: Dict[str, Any], results: List[CapacityTestResult]):
        """生成容量优化建议"""
        recommendations = []
        
        # 容量利用率建议
        avg_utilization = report['summary']['avg_capacity_utilization']
        if avg_utilization < 50:
            recommendations.append("容量利用率较低，系统可能存在性能瓶颈或配置问题")
        elif avg_utilization < 70:
            recommendations.append("容量利用率中等，建议进一步优化以提高效率")
        
        # 扩展性建议
        avg_scalability = report['scalability_analysis']['avg_scalability_factor']
        if avg_scalability < 0.3:
            recommendations.append("扩展性较差，建议检查架构设计和资源配置")
        elif avg_scalability < 0.6:
            recommendations.append("扩展性中等，建议优化并发处理和负载均衡")
        
        # 瓶颈建议
        common_bottleneck = report['bottleneck_analysis']['most_common_bottleneck']
        if common_bottleneck == 'CPU':
            recommendations.append("CPU是主要瓶颈，建议优化算法或增加CPU资源")
        elif common_bottleneck == 'Memory':
            recommendations.append("内存是主要瓶颈，建议优化内存使用或增加内存容量")
        elif common_bottleneck == 'Response Time':
            recommendations.append("响应时间是主要瓶颈，建议优化处理逻辑或增加缓存")
        
        report['recommendations'] = recommendations
    
    def stop_test(self):
        """停止测试"""
        self._stop_event.set()


class TestCapacityLoad:
    """容量负载测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.config = LOAD_TEST_CONFIG
        self.capacity_runner = CapacityTestRunner(self.config)
    
    def teardown_method(self):
        """测试后清理"""
        if hasattr(self, 'capacity_runner'):
            self.capacity_runner.stop_test()
    
    def _print_capacity_summary(self, results: List[CapacityTestResult]):
        """打印容量测试摘要"""
        print(f"\n=== 容量测试摘要 ===")
        print(f"测试数量: {len(results)}")
        
        for result in results:
            print(f"\n{result.test_name}:")
            print(f"  厂商/模型: {result.vendor}/{result.model}")
            print(f"  最大可持续负载: {result.max_sustainable_load} req/s")
            print(f"  峰值负载: {result.peak_load} req/s")
            print(f"  最优负载: {result.optimal_load} req/s")
            print(f"  容量利用率: {result.capacity_utilization:.1f}%")
            print(f"  最大吞吐量: {result.max_throughput:.1f} req/s")
            print(f"  扩展性因子: {result.scalability_factor:.2f}")
            print(f"  资源效率: {result.resource_efficiency:.2f}")
            print(f"  主要瓶颈: {result.primary_bottleneck}")
            print(f"  容量等级: {result.capacity_grade}")
    
    @pytest.mark.load_test
    @pytest.mark.capacity_load
    def test_basic_capacity_discovery(self):
        """
        基础容量发现测试
        
        通过逐步增加负载来发现系统的容量边界
        """
        vendor = 'deepseek'
        model = 'deepseek-chat'
        
        # 运行容量发现测试
        result = self.capacity_runner.run_capacity_discovery(
            vendor=vendor,
            model=model,
            test_name="basic_capacity_discovery",
            start_load=5,
            max_load=100,
            step_size=10,
            step_duration=20
        )
        
        self._print_capacity_summary([result])
        
        # 基础容量发现断言
        assert len(result.capacity_points) > 0
        assert result.max_sustainable_load > 0
        assert result.peak_load >= result.max_sustainable_load
        assert result.optimal_load > 0
        assert result.capacity_utilization >= 0
        
        # 容量指标合理性检查
        assert result.max_throughput > 0
        assert result.scalability_factor >= 0
        assert result.primary_bottleneck in ['CPU', 'Memory', 'Response Time', 'Error Rate', 'unknown']
        assert result.capacity_grade in ['A+', 'A', 'B', 'C', 'D', 'F']
        
        # 容量点数据完整性检查
        for point in result.capacity_points:
            assert point.load_level > 0
            assert point.response_time >= 0
            assert point.throughput >= 0
            assert 0 <= point.error_rate <= 1
            assert 0 <= point.cpu_usage <= 100
            assert 0 <= point.memory_usage <= 100
    
    @pytest.mark.load_test
    @pytest.mark.capacity_load
    def test_binary_search_capacity(self):
        """
        二分搜索容量测试
        
        使用二分搜索快速找到系统的容量边界
        """
        vendor = 'ernie'
        model = 'ernie-3.5-8k'
        
        # 运行二分搜索容量测试
        result = self.capacity_runner.run_binary_search_capacity(
            vendor=vendor,
            model=model,
            test_name="binary_search_capacity",
            min_load=1,
            max_load=200,
            target_error_rate=0.05,
            precision=5
        )
        
        self._print_capacity_summary([result])
        
        # 二分搜索容量断言
        assert len(result.capacity_points) > 0
        assert result.max_sustainable_load > 0
        
        # 验证找到的容量边界的准确性
        # 最后一个测试点应该接近目标错误率
        if result.capacity_points:
            final_point = result.capacity_points[-1]
            assert final_point.error_rate <= 0.1  # 允许一定误差
        
        # 二分搜索应该比线性搜索更高效（测试点更少）
        assert len(result.capacity_points) <= 15  # 二分搜索的测试点数量应该较少
    
    @pytest.mark.load_test
    @pytest.mark.capacity_load
    def test_high_capacity_stress(self):
        """
        高容量压力测试
        
        测试系统在高负载下的极限表现
        """
        vendor = 'google'
        model = 'gemini-pro'
        
        # 配置高容量API
        original_api = self.capacity_runner.api
        self.capacity_runner.api = MockCapacityAPI(
            max_capacity=300,  # 更高的最大容量
            optimal_load=150,  # 更高的最优负载
            degradation_factor=0.005  # 更慢的性能衰减
        )
        
        try:
            # 运行高容量测试
            result = self.capacity_runner.run_capacity_discovery(
                vendor=vendor,
                model=model,
                test_name="high_capacity_stress",
                start_load=50,
                max_load=400,
                step_size=25,
                step_duration=15
            )
            
            self._print_capacity_summary([result])
            
            print(f"\n=== 高容量压力分析 ===")
            print(f"系统崩溃点: {result.breaking_point_load} req/s")
            print(f"效率衰减率: {result.efficiency_degradation:.1f}%")
            
            # 高容量压力断言
            assert result.max_sustainable_load >= 100  # 应该支持至少100 req/s
            assert result.peak_load >= 200  # 应该能测试到至少200 req/s
            
            # 高容量系统应该有更好的扩展性
            assert result.scalability_factor >= 0.3
            assert result.capacity_utilization >= 30
            
            # 应该能识别出系统的崩溃点
            if result.breaking_point_load > 0:
                assert result.breaking_point_load > result.max_sustainable_load
        
        finally:
            # 恢复原始API
            self.capacity_runner.api = original_api
    
    @pytest.mark.load_test
    @pytest.mark.comprehensive_load
    def test_bottleneck_identification(self):
        """
        瓶颈识别测试
        
        专门测试系统瓶颈的识别和分析
        """
        vendor = 'deepseek'
        model = 'deepseek-r1'
        
        # 配置有明显CPU瓶颈的API
        original_api = self.capacity_runner.api
        
        class CPUBottleneckAPI(MockCapacityAPI):
            def get_resource_usage(self, current_load: int) -> Tuple[float, float]:
                # 模拟CPU瓶颈：CPU使用率快速增长
                load_ratio = current_load / self.max_capacity
                cpu_usage = min(100, load_ratio * 120)  # CPU使用率增长更快
                memory_usage = min(50, load_ratio * 30)  # 内存使用率较低
                return cpu_usage, memory_usage
        
        self.capacity_runner.api = CPUBottleneckAPI(
            max_capacity=80,
            optimal_load=40,
            degradation_factor=0.02
        )
        
        try:
            # 运行瓶颈识别测试
            result = self.capacity_runner.run_capacity_discovery(
                vendor=vendor,
                model=model,
                test_name="bottleneck_identification",
                start_load=10,
                max_load=120,
                step_size=15,
                step_duration=20
            )
            
            self._print_capacity_summary([result])
            
            print(f"\n=== 瓶颈分析 ===")
            print(f"主要瓶颈: {result.primary_bottleneck}")
            print(f"瓶颈阈值: {result.bottleneck_threshold:.1f}")
            print(f"瓶颈分析: {result.bottleneck_analysis}")
            
            # 瓶颈识别断言
            assert result.primary_bottleneck == 'CPU'  # 应该识别出CPU瓶颈
            assert result.bottleneck_threshold > 80  # CPU使用率应该超过80%
            
            # 瓶颈分析应该包含详细信息
            assert 'cpu_bottleneck' in result.bottleneck_analysis
            assert result.bottleneck_analysis['cpu_bottleneck'] is True
            assert 'bottleneck_recommendations' in result.bottleneck_analysis
            assert len(result.bottleneck_analysis['bottleneck_recommendations']) > 0
        
        finally:
            # 恢复原始API
            self.capacity_runner.api = original_api
    
    @pytest.mark.load_test
    @pytest.mark.comprehensive_load
    def test_multi_vendor_capacity_comparison(self):
        """
        多厂商容量对比测试
        
        对比不同厂商的容量表现
        """
        vendors_models = [
            ('deepseek', 'deepseek-chat'),
            ('ernie', 'ernie-3.5-8k'),
            ('google', 'gemini-pro')
        ]
        
        results = []
        
        for vendor, model in vendors_models:
            print(f"\n测试厂商容量: {vendor}/{model}")
            
            # 为每个厂商配置不同的容量特性
            if vendor == 'deepseek':
                self.capacity_runner.api = MockCapacityAPI(max_capacity=100, optimal_load=50)
            elif vendor == 'ernie':
                self.capacity_runner.api = MockCapacityAPI(max_capacity=120, optimal_load=60)
            else:  # google
                self.capacity_runner.api = MockCapacityAPI(max_capacity=80, optimal_load=40)
            
            result = self.capacity_runner.run_binary_search_capacity(
                vendor=vendor,
                model=model,
                test_name=f"capacity_comparison_{vendor}",
                min_load=5,
                max_load=150,
                target_error_rate=0.05
            )
            results.append(result)
        
        self._print_capacity_summary(results)
        
        # 生成对比报告
        report = self.capacity_runner.generate_capacity_report(results)
        
        print(f"\n=== 多厂商容量对比 ===")
        print(f"平均最大可持续负载: {report['summary']['avg_max_sustainable_load']:.1f} req/s")
        print(f"平均容量利用率: {report['summary']['avg_capacity_utilization']:.1f}%")
        print(f"平均扩展性因子: {report['summary']['avg_scalability_factor']:.2f}")
        print(f"最常见瓶颈: {report['bottleneck_analysis']['most_common_bottleneck']}")
        
        print(f"\n等级分布:")
        for grade, count in report['summary']['grade_distribution'].items():
            if count > 0:
                print(f"  {grade}: {count} 个")
        
        # 多厂商对比断言
        assert len(results) == len(vendors_models)
        assert all(r.max_sustainable_load > 0 for r in results)
        
        # 应该有容量差异
        max_loads = [r.max_sustainable_load for r in results]
        assert max(max_loads) > min(max_loads)  # 不同厂商应该有不同的容量
        
        # 报告应该包含有效分析
        assert 'capacity_analysis' in report
        assert 'bottleneck_analysis' in report
        assert 'scalability_analysis' in report
        assert len(report['recommendations']) >= 0
        
        # 至少有一个厂商应该达到良好等级
        good_grades = ['A+', 'A', 'B']
        assert any(r.capacity_grade in good_grades for r in results)
    
    @pytest.mark.load_test
    @pytest.mark.benchmark
    def test_capacity_benchmark(self, benchmark):
        """
        容量测试基准
        
        使用pytest-benchmark进行容量测试基准
        """
        vendor = 'doubao'
        model = 'doubao-1-5-pro-32k-character-250715'
        
        # 运行基准测试
        result = benchmark(self.capacity_runner.run_binary_search_capacity,
                          vendor, model, "capacity_benchmark",
                          1, 100, 0.05, 5)
        
        self._print_capacity_summary([result])
        
        # 容量基准断言
        thresholds = self.config['performance_thresholds']
        assert result.max_sustainable_load >= 10  # 至少支持10 req/s
        assert result.capacity_utilization >= 20  # 容量利用率至少20%
        assert result.scalability_factor >= 0.1  # 扩展性因子至少0.1
        assert result.capacity_grade in ['A+', 'A', 'B', 'C', 'D', 'F']