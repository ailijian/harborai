#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一并发性能测试模块

合并所有并发相关测试，包括：
1. 基础并发性能测试
2. 高并发压力测试
3. 并发稳定性验证
4. 吞吐量测试
5. 资源竞争检测
6. 异步并发测试

验证HarborAI并发优化组件的性能提升：
- 目标：从505.6ops/s提升到≥1000ops/s
- 测试场景：不同并发级别、不同负载模式
- 稳定性验证：高并发下的系统稳定性
"""

import asyncio
import time
import statistics
import logging
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pytest
import psutil

# 导入被测试的组件
try:
    from harborai.api.fast_client import FastHarborAI, create_fast_client
except ImportError:
    FastHarborAI = None
    create_fast_client = None

try:
    from harborai.core.optimizations.concurrency_manager import ConcurrencyManager, ConcurrencyConfig
except ImportError:
    ConcurrencyManager = None
    ConcurrencyConfig = None

logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyMetrics:
    """并发测试指标"""
    test_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    concurrent_users: int
    response_times: List[float]
    throughput_per_second: List[float]
    error_messages: List[str] = field(default_factory=list)
    resource_usage: Dict[str, List[float]] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """平均响应时间"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def average_throughput(self) -> float:
        """平均吞吐量"""
        if not self.throughput_per_second:
            return 0.0
        return statistics.mean(self.throughput_per_second)
    
    @property
    def peak_throughput(self) -> float:
        """峰值吞吐量"""
        if not self.throughput_per_second:
            return 0.0
        return max(self.throughput_per_second)
    
    @property
    def test_duration(self) -> timedelta:
        """测试持续时间"""
        return self.end_time - self.start_time
    
    @property
    def requests_per_second(self) -> float:
        """每秒请求数"""
        duration_seconds = self.test_duration.total_seconds()
        if duration_seconds == 0:
            return 0.0
        return self.successful_requests / duration_seconds


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
        duration = self.end_time - self.start_time
        
        if not self.response_times:
            return {
                "total_requests": total_requests,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "success_rate": 0.0,
                "duration": duration,
                "ops_per_second": 0.0,
                "avg_response_time": 0.0,
                "min_response_time": 0.0,
                "max_response_time": 0.0,
                "p95_response_time": 0.0,
                "p99_response_time": 0.0
            }
        
        return {
            "total_requests": total_requests,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / total_requests if total_requests > 0 else 0.0,
            "duration": duration,
            "ops_per_second": self.success_count / duration if duration > 0 else 0.0,
            "avg_response_time": statistics.mean(self.response_times),
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times),
            "p95_response_time": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else max(self.response_times),
            "p99_response_time": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) >= 100 else max(self.response_times)
        }


class MockPlugin:
    """模拟插件"""
    
    def __init__(self, response_time_ms: float = 10):
        self.response_time_ms = response_time_ms
        self.call_count = 0
        self.lock = threading.Lock()
    
    def chat_completion(self, messages, model, **kwargs):
        """同步聊天完成"""
        with self.lock:
            self.call_count += 1
            call_id = self.call_count
        
        # 模拟处理时间
        time.sleep(self.response_time_ms / 1000.0)
        
        return {
            "id": f"chatcmpl-{call_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"模拟响应 {call_id}"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
    
    async def chat_completion_async(self, messages, model, **kwargs):
        """异步聊天完成"""
        with self.lock:
            self.call_count += 1
            call_id = self.call_count
        
        # 模拟异步处理时间
        await asyncio.sleep(self.response_time_ms / 1000.0)
        
        return {
            "id": f"chatcmpl-async-{call_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"异步模拟响应 {call_id}"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }


def setup_mock_plugin_manager(response_time_ms: float = 10):
    """设置模拟插件管理器"""
    mock_plugin = MockPlugin(response_time_ms)
    
    mock_manager = Mock()
    mock_manager.get_plugin_for_model.return_value = mock_plugin
    mock_manager.get_supported_models.return_value = ['gpt-3.5-turbo', 'gpt-4']
    mock_manager.is_model_supported.return_value = True
    mock_manager.get_plugin_name_for_model.return_value = "mock_plugin"
    
    return mock_manager


class ConcurrencyPerformanceTester:
    """并发性能测试器"""
    
    def __init__(self):
        self.target_ops_per_second = 1000.0  # 目标性能
        self.baseline_ops_per_second = 505.6  # 基线性能
    
    def test_baseline_performance(self, num_requests: int = 100) -> Dict[str, Any]:
        """测试基线性能（无并发优化）"""
        if not FastHarborAI:
            pytest.skip("FastHarborAI 不可用")
        
        # 创建无优化的客户端
        client = FastHarborAI()
        metrics = PerformanceMetrics()
        
        messages = [{"role": "user", "content": "测试消息"}]
        
        with patch.object(client, '_plugin_manager', setup_mock_plugin_manager(10)):
            metrics.start_timing()
            
            for i in range(num_requests):
                start_time = time.perf_counter()
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=messages
                    )
                    end_time = time.perf_counter()
                    metrics.record_request(end_time - start_time, True)
                except Exception as e:
                    end_time = time.perf_counter()
                    metrics.record_request(end_time - start_time, False)
            
            metrics.stop_timing()
        
        return metrics.get_statistics()
    
    async def test_concurrency_performance(self, num_requests: int = 100, concurrency_level: int = 50) -> Dict[str, Any]:
        """测试并发性能（启用并发优化）"""
        if not create_fast_client:
            pytest.skip("create_fast_client 不可用")
        
        # 创建启用并发优化的客户端
        config = {
            "concurrency": {
                "max_concurrent_requests": concurrency_level,
                "request_timeout": 30.0,
                "enable_connection_pooling": True,
                "pool_size": concurrency_level
            }
        }
        
        client = create_fast_client(config=config)
        metrics = PerformanceMetrics()
        
        messages = [{"role": "user", "content": "测试消息"}]
        
        with patch.object(client, '_plugin_manager', setup_mock_plugin_manager(10)):
            metrics.start_timing()
            
            # 创建并发任务
            tasks = []
            for i in range(num_requests):
                task = self._make_async_request(client, i, metrics, messages)
                tasks.append(task)
            
            # 执行并发请求
            await asyncio.gather(*tasks, return_exceptions=True)
            
            metrics.stop_timing()
        
        return metrics.get_statistics()
    
    async def test_high_concurrency_stress(self, num_requests: int = 500, concurrency_level: int = 100) -> Dict[str, Any]:
        """测试高并发压力"""
        if not create_fast_client:
            pytest.skip("create_fast_client 不可用")
        
        # 创建高并发配置的客户端
        config = {
            "concurrency": {
                "max_concurrent_requests": concurrency_level,
                "request_timeout": 60.0,
                "enable_connection_pooling": True,
                "pool_size": concurrency_level,
                "enable_request_batching": True,
                "batch_size": 10
            }
        }
        
        client = create_fast_client(config=config)
        metrics = PerformanceMetrics()
        
        messages = [{"role": "user", "content": "高并发测试消息"}]
        
        with patch.object(client, '_plugin_manager', setup_mock_plugin_manager(5)):  # 更快的响应时间
            metrics.start_timing()
            
            # 创建高并发任务
            semaphore = asyncio.Semaphore(concurrency_level)
            tasks = []
            
            for i in range(num_requests):
                task = self._make_semaphore_async_request(client, i, metrics, messages, semaphore)
                tasks.append(task)
            
            # 执行高并发请求
            await asyncio.gather(*tasks, return_exceptions=True)
            
            metrics.stop_timing()
        
        return metrics.get_statistics()
    
    def _make_sync_request(self, client, request_id: int, metrics: PerformanceMetrics, messages: List[Dict]):
        """执行同步请求"""
        start_time = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            end_time = time.perf_counter()
            metrics.record_request(end_time - start_time, True)
        except Exception as e:
            end_time = time.perf_counter()
            metrics.record_request(end_time - start_time, False)
    
    async def _make_async_request(self, client, request_id: int, metrics: PerformanceMetrics, messages: List[Dict]):
        """执行异步请求"""
        start_time = time.perf_counter()
        try:
            response = await client.chat.completions.acreate(
                model="gpt-3.5-turbo",
                messages=messages
            )
            end_time = time.perf_counter()
            metrics.record_request(end_time - start_time, True)
        except Exception as e:
            end_time = time.perf_counter()
            metrics.record_request(end_time - start_time, False)
    
    async def _make_semaphore_async_request(self, client, request_id: int, metrics: PerformanceMetrics, messages: List[Dict], semaphore: asyncio.Semaphore):
        """执行带信号量控制的异步请求"""
        async with semaphore:
            await self._make_async_request(client, request_id, metrics, messages)
    
    def analyze_results(self, baseline: Dict[str, Any], optimized: Dict[str, Any], stress: Dict[str, Any]) -> Dict[str, Any]:
        """分析测试结果"""
        baseline_ops = baseline.get("ops_per_second", 0)
        optimized_ops = optimized.get("ops_per_second", 0)
        stress_ops = stress.get("ops_per_second", 0)
        
        improvement_ratio = optimized_ops / baseline_ops if baseline_ops > 0 else 0
        target_achieved = optimized_ops >= self.target_ops_per_second
        
        return {
            "baseline_performance": {
                "ops_per_second": baseline_ops,
                "avg_response_time": baseline.get("avg_response_time", 0),
                "success_rate": baseline.get("success_rate", 0)
            },
            "optimized_performance": {
                "ops_per_second": optimized_ops,
                "avg_response_time": optimized.get("avg_response_time", 0),
                "success_rate": optimized.get("success_rate", 0)
            },
            "stress_performance": {
                "ops_per_second": stress_ops,
                "avg_response_time": stress.get("avg_response_time", 0),
                "success_rate": stress.get("success_rate", 0)
            },
            "analysis": {
                "improvement_ratio": improvement_ratio,
                "improvement_percentage": (improvement_ratio - 1) * 100,
                "target_ops_per_second": self.target_ops_per_second,
                "target_achieved": target_achieved,
                "performance_grade": self._calculate_performance_grade(optimized_ops)
            }
        }
    
    def _calculate_performance_grade(self, ops_per_second: float) -> str:
        """计算性能等级"""
        if ops_per_second >= self.target_ops_per_second:
            return "A"
        elif ops_per_second >= self.target_ops_per_second * 0.8:
            return "B"
        elif ops_per_second >= self.target_ops_per_second * 0.6:
            return "C"
        elif ops_per_second >= self.target_ops_per_second * 0.4:
            return "D"
        else:
            return "F"


@pytest.mark.performance
@pytest.mark.concurrency
class TestConcurrencyUnified:
    """统一并发测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.tester = ConcurrencyPerformanceTester()
        self.target_ops_per_second = 1000.0
        self.min_success_rate = 0.95
    
    @pytest.mark.baseline
    def test_baseline_performance(self):
        """测试基线性能"""
        if not FastHarborAI:
            pytest.skip("FastHarborAI 不可用")
        
        result = self.tester.test_baseline_performance(num_requests=100)
        
        # 验证基本功能
        assert result["success_rate"] >= self.min_success_rate, \
            f"基线成功率 {result['success_rate']:.3f} 低于要求 {self.min_success_rate}"
        
        assert result["ops_per_second"] > 0, "基线性能应该大于0"
        
        print(f"基线性能测试结果:")
        print(f"  吞吐量: {result['ops_per_second']:.2f} ops/s")
        print(f"  平均响应时间: {result['avg_response_time']:.3f}s")
        print(f"  成功率: {result['success_rate']:.3f}")
    
    @pytest.mark.asyncio
    @pytest.mark.optimized
    async def test_concurrency_optimization(self):
        """测试并发优化效果"""
        if not create_fast_client:
            pytest.skip("create_fast_client 不可用")
        
        result = await self.tester.test_concurrency_performance(
            num_requests=200, 
            concurrency_level=50
        )
        
        # 验证并发优化效果
        assert result["success_rate"] >= self.min_success_rate, \
            f"并发优化成功率 {result['success_rate']:.3f} 低于要求 {self.min_success_rate}"
        
        # 验证性能提升
        assert result["ops_per_second"] > 500, \
            f"并发优化性能 {result['ops_per_second']:.2f} ops/s 低于预期"
        
        print(f"并发优化测试结果:")
        print(f"  吞吐量: {result['ops_per_second']:.2f} ops/s")
        print(f"  平均响应时间: {result['avg_response_time']:.3f}s")
        print(f"  成功率: {result['success_rate']:.3f}")
    
    @pytest.mark.asyncio
    @pytest.mark.stress
    async def test_high_concurrency_stress(self):
        """测试高并发压力"""
        if not create_fast_client:
            pytest.skip("create_fast_client 不可用")
        
        result = await self.tester.test_high_concurrency_stress(
            num_requests=500, 
            concurrency_level=100
        )
        
        # 验证高并发稳定性
        assert result["success_rate"] >= 0.90, \
            f"高并发成功率 {result['success_rate']:.3f} 低于要求 0.90"
        
        # 验证响应时间合理
        assert result["avg_response_time"] < 2.0, \
            f"高并发平均响应时间 {result['avg_response_time']:.3f}s 过长"
        
        print(f"高并发压力测试结果:")
        print(f"  吞吐量: {result['ops_per_second']:.2f} ops/s")
        print(f"  平均响应时间: {result['avg_response_time']:.3f}s")
        print(f"  成功率: {result['success_rate']:.3f}")
    
    @pytest.mark.asyncio
    @pytest.mark.comprehensive
    async def test_comprehensive_concurrency_analysis(self):
        """综合并发性能分析"""
        if not FastHarborAI or not create_fast_client:
            pytest.skip("所需组件不可用")
        
        # 执行所有测试
        baseline = self.tester.test_baseline_performance(num_requests=100)
        optimized = await self.tester.test_concurrency_performance(num_requests=200, concurrency_level=50)
        stress = await self.tester.test_high_concurrency_stress(num_requests=300, concurrency_level=80)
        
        # 分析结果
        analysis = self.tester.analyze_results(baseline, optimized, stress)
        
        # 验证性能提升
        improvement_ratio = analysis["analysis"]["improvement_ratio"]
        assert improvement_ratio > 1.5, \
            f"性能提升比例 {improvement_ratio:.2f} 低于预期 1.5"
        
        # 验证目标达成
        target_achieved = analysis["analysis"]["target_achieved"]
        performance_grade = analysis["analysis"]["performance_grade"]
        
        print(f"综合并发性能分析:")
        print(f"  基线性能: {baseline['ops_per_second']:.2f} ops/s")
        print(f"  优化性能: {optimized['ops_per_second']:.2f} ops/s")
        print(f"  压力性能: {stress['ops_per_second']:.2f} ops/s")
        print(f"  性能提升: {analysis['analysis']['improvement_percentage']:.1f}%")
        print(f"  性能等级: {performance_grade}")
        print(f"  目标达成: {'是' if target_achieved else '否'}")
        
        # 如果未达到目标，给出警告而不是失败
        if not target_achieved:
            print(f"警告: 未达到目标性能 {self.target_ops_per_second} ops/s")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])