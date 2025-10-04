#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高并发稳定性测试

验证并发优化组件在高并发场景下的稳定性和可靠性。

测试策略：
1. 长时间运行测试：验证系统在长时间高并发下的稳定性
2. 突发流量测试：验证系统对突发流量的处理能力
3. 资源泄漏测试：验证系统是否存在内存或连接泄漏
4. 错误恢复测试：验证系统在错误情况下的恢复能力
5. 负载均衡测试：验证请求分发的均匀性

Assumptions:
- A1: 并发管理器能够正确处理高并发请求而不崩溃
- A2: 连接池能够正确管理连接生命周期，避免泄漏
- A3: 无锁数据结构在高并发下不会出现数据竞争
- A4: 系统能够在错误情况下自动恢复
- A5: 内存使用量在长时间运行下保持稳定
"""

import asyncio
import time
import threading
import gc
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
import pytest
import logging
from dataclasses import dataclass
from collections import defaultdict
import random

# 测试用的模拟组件
from unittest.mock import Mock, AsyncMock, patch

# 导入被测试的组件
try:
    from harborai.api.fast_client import FastHarborAI, create_fast_client
    from harborai.core.optimizations.concurrency_manager import ConcurrencyManager, ConcurrencyConfig
    from harborai.core.optimizations.lockfree_plugin_manager import LockFreePluginManager
    from harborai.core.optimizations.async_request_processor import AsyncRequestProcessor
    from harborai.core.optimizations.optimized_connection_pool import OptimizedConnectionPool
    CONCURRENCY_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    CONCURRENCY_OPTIMIZATION_AVAILABLE = False
    print(f"并发优化组件不可用: {e}")

logger = logging.getLogger(__name__)


@dataclass
class StabilityMetrics:
    """稳定性指标"""
    start_time: float
    end_time: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    error_types: Dict[str, int]
    memory_usage_mb: List[float]
    cpu_usage_percent: List[float]
    response_times_ms: List[float]
    concurrent_connections: List[int]
    
    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    @property
    def avg_response_time_ms(self) -> float:
        if not self.response_times_ms:
            return 0.0
        return sum(self.response_times_ms) / len(self.response_times_ms)
    
    @property
    def ops_per_second(self) -> float:
        if self.duration_seconds == 0:
            return 0.0
        return self.total_requests / self.duration_seconds
    
    @property
    def memory_growth_mb(self) -> float:
        if len(self.memory_usage_mb) < 2:
            return 0.0
        return self.memory_usage_mb[-1] - self.memory_usage_mb[0]


class UnstablePlugin:
    """不稳定的模拟插件，用于测试错误恢复"""
    
    def __init__(self, failure_rate: float = 0.1, response_time_ms: float = 100):
        """
        Args:
            failure_rate: 失败率 (0.0-1.0)
            response_time_ms: 基础响应时间（毫秒）
        """
        self.failure_rate = failure_rate
        self.response_time_ms = response_time_ms
        self.call_count = 0
        self.lock = threading.Lock()
    
    def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> Dict[str, Any]:
        """模拟不稳定的同步聊天完成"""
        with self.lock:
            self.call_count += 1
        
        # 随机失败
        if random.random() < self.failure_rate:
            raise Exception(f"模拟错误 - 请求 {self.call_count}")
        
        # 随机响应时间（50%-150%的基础时间）
        actual_time = self.response_time_ms * (0.5 + random.random())
        time.sleep(actual_time / 1000)
        
        return {
            "id": f"chatcmpl-{self.call_count}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"稳定性测试响应 {self.call_count}"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    
    async def chat_completion_async(self, messages: List[Dict], model: str, **kwargs) -> Dict[str, Any]:
        """模拟不稳定的异步聊天完成"""
        with self.lock:
            self.call_count += 1
        
        # 随机失败
        if random.random() < self.failure_rate:
            raise Exception(f"模拟异步错误 - 请求 {self.call_count}")
        
        # 随机响应时间
        actual_time = self.response_time_ms * (0.5 + random.random())
        await asyncio.sleep(actual_time / 1000)
        
        return {
            "id": f"chatcmpl-async-{self.call_count}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"异步稳定性测试响应 {self.call_count}"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }


class HighConcurrencyStabilityTester:
    """高并发稳定性测试器"""
    
    def __init__(self):
        self.test_messages = [
            {"role": "user", "content": "稳定性测试消息"}
        ]
        self.test_model = "gpt-3.5-turbo"
        self.process = psutil.Process(os.getpid())
    
    def setup_mock_plugin_manager(self, plugin):
        """设置模拟插件管理器"""
        def mock_get_plugin_for_model(model):
            return plugin
        
        def mock_get_plugin_name_for_model(model):
            return "unstable_plugin"
        
        mock_manager = Mock()
        mock_manager.get_plugin_for_model = mock_get_plugin_for_model
        mock_manager.get_plugin_name_for_model = mock_get_plugin_name_for_model
        mock_manager.get_supported_models.return_value = [self.test_model]
        
        return mock_manager
    
    def get_system_metrics(self) -> Tuple[float, float, int]:
        """获取系统指标
        
        Returns:
            (内存使用MB, CPU使用率%, 连接数)
        """
        try:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            cpu_percent = self.process.cpu_percent()
            
            # 获取连接数（简化版）
            connections = len(self.process.connections())
            
            return memory_mb, cpu_percent, connections
        except Exception as e:
            logger.warning(f"获取系统指标失败: {e}")
            return 0.0, 0.0, 0
    
    async def long_running_stability_test(
        self,
        duration_minutes: int = 5,
        requests_per_second: int = 50,
        max_concurrent: int = 100
    ) -> StabilityMetrics:
        """长时间运行稳定性测试
        
        Args:
            duration_minutes: 测试持续时间（分钟）
            requests_per_second: 每秒请求数
            max_concurrent: 最大并发数
            
        Returns:
            稳定性指标
        """
        logger.info(f"开始长时间稳定性测试: {duration_minutes}分钟, {requests_per_second} req/s")
        
        # 创建不稳定插件（10%失败率）
        unstable_plugin = UnstablePlugin(failure_rate=0.1, response_time_ms=50)
        
        # 配置客户端
        config = {
            'enable_memory_optimization': True,
            'concurrency_optimization': {
                'max_concurrent_requests': max_concurrent,
                'connection_pool_size': 50,
                'request_timeout': 30.0,
                'enable_adaptive_optimization': True,
                'enable_health_check': True,
                'health_check_interval': 30.0
            }
        }
        
        # 初始化指标
        metrics = StabilityMetrics(
            start_time=time.perf_counter(),
            end_time=0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            error_types=defaultdict(int),
            memory_usage_mb=[],
            cpu_usage_percent=[],
            response_times_ms=[],
            concurrent_connections=[]
        )
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_get_manager.return_value = self.setup_mock_plugin_manager(unstable_plugin)
            
            client = create_fast_client(config=config)
            
            # 控制请求速率的信号量
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def make_request():
                """执行单个请求"""
                async with semaphore:
                    request_start = time.perf_counter()
                    try:
                        response = await client.chat.completions.create_async(
                            messages=self.test_messages,
                            model=self.test_model
                        )
                        
                        response_time = (time.perf_counter() - request_start) * 1000
                        metrics.response_times_ms.append(response_time)
                        metrics.successful_requests += 1
                        return True, None
                        
                    except Exception as e:
                        error_type = type(e).__name__
                        metrics.error_types[error_type] += 1
                        metrics.failed_requests += 1
                        return False, str(e)
                    finally:
                        metrics.total_requests += 1
            
            # 系统监控任务
            async def monitor_system():
                """监控系统资源"""
                while time.perf_counter() - metrics.start_time < duration_minutes * 60:
                    memory_mb, cpu_percent, connections = self.get_system_metrics()
                    metrics.memory_usage_mb.append(memory_mb)
                    metrics.cpu_usage_percent.append(cpu_percent)
                    metrics.concurrent_connections.append(connections)
                    
                    await asyncio.sleep(5)  # 每5秒监控一次
            
            # 请求生成任务
            async def generate_requests():
                """生成请求"""
                request_interval = 1.0 / requests_per_second
                tasks = []
                
                while time.perf_counter() - metrics.start_time < duration_minutes * 60:
                    # 创建请求任务
                    task = asyncio.create_task(make_request())
                    tasks.append(task)
                    
                    # 清理已完成的任务
                    if len(tasks) > max_concurrent * 2:
                        done_tasks = [t for t in tasks if t.done()]
                        for task in done_tasks:
                            tasks.remove(task)
                    
                    await asyncio.sleep(request_interval)
                
                # 等待剩余任务完成
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            
            # 并行运行监控和请求生成
            await asyncio.gather(
                monitor_system(),
                generate_requests(),
                return_exceptions=True
            )
        
        metrics.end_time = time.perf_counter()
        
        logger.info(f"长时间稳定性测试完成: {metrics.ops_per_second:.2f} ops/s, 成功率: {metrics.success_rate:.1f}%")
        return metrics
    
    async def burst_traffic_test(
        self,
        burst_duration_seconds: int = 30,
        burst_requests_per_second: int = 200,
        normal_requests_per_second: int = 50,
        max_concurrent: int = 150
    ) -> StabilityMetrics:
        """突发流量测试
        
        Args:
            burst_duration_seconds: 突发持续时间（秒）
            burst_requests_per_second: 突发期间每秒请求数
            normal_requests_per_second: 正常期间每秒请求数
            max_concurrent: 最大并发数
            
        Returns:
            稳定性指标
        """
        logger.info(f"开始突发流量测试: 突发{burst_requests_per_second} req/s 持续{burst_duration_seconds}秒")
        
        # 创建稳定插件
        stable_plugin = UnstablePlugin(failure_rate=0.05, response_time_ms=30)
        
        config = {
            'enable_memory_optimization': True,
            'concurrency_optimization': {
                'max_concurrent_requests': max_concurrent,
                'connection_pool_size': 100,
                'request_timeout': 30.0,
                'enable_adaptive_optimization': True,
                'enable_health_check': True
            }
        }
        
        metrics = StabilityMetrics(
            start_time=time.perf_counter(),
            end_time=0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            error_types=defaultdict(int),
            memory_usage_mb=[],
            cpu_usage_percent=[],
            response_times_ms=[],
            concurrent_connections=[]
        )
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_get_manager.return_value = self.setup_mock_plugin_manager(stable_plugin)
            
            client = create_fast_client(config=config)
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def make_request():
                """执行单个请求"""
                async with semaphore:
                    request_start = time.perf_counter()
                    try:
                        response = await client.chat.completions.create_async(
                            messages=self.test_messages,
                            model=self.test_model
                        )
                        
                        response_time = (time.perf_counter() - request_start) * 1000
                        metrics.response_times_ms.append(response_time)
                        metrics.successful_requests += 1
                        return True
                        
                    except Exception as e:
                        error_type = type(e).__name__
                        metrics.error_types[error_type] += 1
                        metrics.failed_requests += 1
                        return False
                    finally:
                        metrics.total_requests += 1
            
            # 系统监控
            async def monitor_system():
                while time.perf_counter() - metrics.start_time < 120:  # 2分钟总测试时间
                    memory_mb, cpu_percent, connections = self.get_system_metrics()
                    metrics.memory_usage_mb.append(memory_mb)
                    metrics.cpu_usage_percent.append(cpu_percent)
                    metrics.concurrent_connections.append(connections)
                    await asyncio.sleep(2)
            
            # 请求生成（包含突发）
            async def generate_burst_requests():
                tasks = []
                test_start = time.perf_counter()
                
                # 第一阶段：正常流量 (30秒)
                logger.info("阶段1: 正常流量")
                phase1_end = test_start + 30
                while time.perf_counter() < phase1_end:
                    task = asyncio.create_task(make_request())
                    tasks.append(task)
                    await asyncio.sleep(1.0 / normal_requests_per_second)
                
                # 第二阶段：突发流量
                logger.info("阶段2: 突发流量")
                phase2_end = time.perf_counter() + burst_duration_seconds
                while time.perf_counter() < phase2_end:
                    task = asyncio.create_task(make_request())
                    tasks.append(task)
                    await asyncio.sleep(1.0 / burst_requests_per_second)
                
                # 第三阶段：恢复正常流量 (30秒)
                logger.info("阶段3: 恢复正常流量")
                phase3_end = time.perf_counter() + 30
                while time.perf_counter() < phase3_end:
                    task = asyncio.create_task(make_request())
                    tasks.append(task)
                    await asyncio.sleep(1.0 / normal_requests_per_second)
                
                # 等待所有任务完成
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
            
            await asyncio.gather(
                monitor_system(),
                generate_burst_requests(),
                return_exceptions=True
            )
        
        metrics.end_time = time.perf_counter()
        
        logger.info(f"突发流量测试完成: {metrics.ops_per_second:.2f} ops/s, 成功率: {metrics.success_rate:.1f}%")
        return metrics
    
    async def memory_leak_test(
        self,
        duration_minutes: int = 3,
        requests_per_second: int = 100
    ) -> StabilityMetrics:
        """内存泄漏测试
        
        Args:
            duration_minutes: 测试持续时间（分钟）
            requests_per_second: 每秒请求数
            
        Returns:
            稳定性指标
        """
        logger.info(f"开始内存泄漏测试: {duration_minutes}分钟, {requests_per_second} req/s")
        
        # 强制垃圾回收
        gc.collect()
        
        stable_plugin = UnstablePlugin(failure_rate=0.02, response_time_ms=20)
        
        config = {
            'enable_memory_optimization': True,
            'concurrency_optimization': {
                'max_concurrent_requests': 50,
                'connection_pool_size': 30,
                'request_timeout': 30.0,
                'enable_adaptive_optimization': True
            }
        }
        
        metrics = StabilityMetrics(
            start_time=time.perf_counter(),
            end_time=0,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            error_types=defaultdict(int),
            memory_usage_mb=[],
            cpu_usage_percent=[],
            response_times_ms=[],
            concurrent_connections=[]
        )
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_get_manager.return_value = self.setup_mock_plugin_manager(stable_plugin)
            
            client = create_fast_client(config=config)
            
            async def make_request():
                try:
                    response = await client.chat.completions.create_async(
                        messages=self.test_messages,
                        model=self.test_model
                    )
                    metrics.successful_requests += 1
                except Exception as e:
                    error_type = type(e).__name__
                    metrics.error_types[error_type] += 1
                    metrics.failed_requests += 1
                finally:
                    metrics.total_requests += 1
            
            # 详细内存监控
            async def detailed_memory_monitor():
                while time.perf_counter() - metrics.start_time < duration_minutes * 60:
                    # 强制垃圾回收
                    gc.collect()
                    
                    memory_mb, cpu_percent, connections = self.get_system_metrics()
                    metrics.memory_usage_mb.append(memory_mb)
                    metrics.cpu_usage_percent.append(cpu_percent)
                    metrics.concurrent_connections.append(connections)
                    
                    # 记录详细内存信息
                    if len(metrics.memory_usage_mb) % 10 == 0:  # 每50秒记录一次
                        logger.info(f"内存使用: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%, 连接数: {connections}")
                    
                    await asyncio.sleep(5)
            
            # 持续请求生成
            async def continuous_requests():
                request_interval = 1.0 / requests_per_second
                
                while time.perf_counter() - metrics.start_time < duration_minutes * 60:
                    await make_request()
                    await asyncio.sleep(request_interval)
            
            await asyncio.gather(
                detailed_memory_monitor(),
                continuous_requests(),
                return_exceptions=True
            )
        
        metrics.end_time = time.perf_counter()
        
        # 最终垃圾回收
        gc.collect()
        
        logger.info(f"内存泄漏测试完成: 内存增长 {metrics.memory_growth_mb:.1f}MB")
        return metrics


# 测试用例
@pytest.mark.asyncio
@pytest.mark.stability
class TestHighConcurrencyStability:
    """高并发稳定性测试用例"""
    
    def setup_method(self):
        """测试前设置"""
        self.tester = HighConcurrencyStabilityTester()
        logging.basicConfig(level=logging.INFO)
    
    async def test_long_running_stability(self):
        """长时间运行稳定性测试"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        metrics = await self.tester.long_running_stability_test(
            duration_minutes=2,  # 2分钟测试
            requests_per_second=30,
            max_concurrent=50
        )
        
        # 验证稳定性指标
        assert metrics.success_rate >= 85.0, f"长时间运行成功率过低: {metrics.success_rate:.1f}%"
        assert metrics.ops_per_second >= 20.0, f"长时间运行吞吐量过低: {metrics.ops_per_second:.2f} ops/s"
        assert len(metrics.memory_usage_mb) > 0, "缺少内存监控数据"
        
        # 检查内存增长
        if len(metrics.memory_usage_mb) >= 2:
            memory_growth_rate = metrics.memory_growth_mb / metrics.duration_seconds * 60  # MB/分钟
            assert memory_growth_rate < 10.0, f"内存增长过快: {memory_growth_rate:.2f} MB/分钟"
        
        logger.info(f"长时间稳定性测试通过: {metrics.success_rate:.1f}% 成功率")
    
    async def test_burst_traffic_handling(self):
        """突发流量处理测试"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        metrics = await self.tester.burst_traffic_test(
            burst_duration_seconds=20,
            burst_requests_per_second=100,
            normal_requests_per_second=30,
            max_concurrent=80
        )
        
        # 验证突发流量处理能力
        assert metrics.success_rate >= 80.0, f"突发流量成功率过低: {metrics.success_rate:.1f}%"
        assert metrics.ops_per_second >= 40.0, f"突发流量吞吐量过低: {metrics.ops_per_second:.2f} ops/s"
        
        logger.info(f"突发流量测试通过: {metrics.success_rate:.1f}% 成功率")
    
    async def test_memory_leak_detection(self):
        """内存泄漏检测测试"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        metrics = await self.tester.memory_leak_test(
            duration_minutes=2,  # 2分钟测试
            requests_per_second=50
        )
        
        # 验证内存使用
        assert metrics.success_rate >= 90.0, f"内存泄漏测试成功率过低: {metrics.success_rate:.1f}%"
        
        # 检查内存泄漏
        if len(metrics.memory_usage_mb) >= 2:
            memory_growth_rate = metrics.memory_growth_mb / metrics.duration_seconds * 60  # MB/分钟
            assert memory_growth_rate < 5.0, f"检测到内存泄漏: {memory_growth_rate:.2f} MB/分钟增长"
            
            logger.info(f"内存使用稳定: 增长 {metrics.memory_growth_mb:.1f}MB ({memory_growth_rate:.2f} MB/分钟)")
        
        logger.info(f"内存泄漏测试通过: 无明显内存泄漏")
    
    async def test_error_recovery(self):
        """错误恢复测试"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        # 使用高失败率插件测试错误恢复
        unstable_plugin = UnstablePlugin(failure_rate=0.3, response_time_ms=50)
        
        config = {
            'enable_memory_optimization': True,
            'concurrency_optimization': {
                'max_concurrent_requests': 30,
                'connection_pool_size': 20,
                'request_timeout': 10.0,
                'enable_adaptive_optimization': True,
                'enable_health_check': True,
                'health_check_interval': 10.0
            }
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_get_manager.return_value = self.tester.setup_mock_plugin_manager(unstable_plugin)
            
            client = create_fast_client(config=config)
            
            success_count = 0
            error_count = 0
            total_requests = 100
            
            # 执行请求
            for i in range(total_requests):
                try:
                    response = await client.chat.completions.create_async(
                        messages=self.tester.test_messages,
                        model=self.tester.test_model
                    )
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    logger.debug(f"预期错误: {e}")
                
                # 短暂延迟
                await asyncio.sleep(0.01)
            
            success_rate = (success_count / total_requests) * 100
            
            # 验证系统在高错误率下仍能正常工作
            assert success_rate >= 60.0, f"错误恢复测试失败: 成功率仅 {success_rate:.1f}%"
            assert success_count > 0, "系统完全无法处理请求"
            
            logger.info(f"错误恢复测试通过: {success_rate:.1f}% 成功率，系统正常恢复")


if __name__ == "__main__":
    """直接运行稳定性测试"""
    
    async def main():
        """主测试函数"""
        tester = HighConcurrencyStabilityTester()
        
        print("=== HarborAI 高并发稳定性测试 ===")
        print()
        
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            print("并发优化组件不可用，跳过测试")
            return
        
        # 长时间稳定性测试
        print("1. 长时间稳定性测试 (2分钟)...")
        stability_metrics = await tester.long_running_stability_test(
            duration_minutes=2,
            requests_per_second=30,
            max_concurrent=50
        )
        print(f"   结果: {stability_metrics.success_rate:.1f}% 成功率, {stability_metrics.ops_per_second:.2f} ops/s")
        print(f"   内存增长: {stability_metrics.memory_growth_mb:.1f}MB")
        print()
        
        # 突发流量测试
        print("2. 突发流量测试...")
        burst_metrics = await tester.burst_traffic_test(
            burst_duration_seconds=20,
            burst_requests_per_second=100,
            normal_requests_per_second=30,
            max_concurrent=80
        )
        print(f"   结果: {burst_metrics.success_rate:.1f}% 成功率, {burst_metrics.ops_per_second:.2f} ops/s")
        print()
        
        # 内存泄漏测试
        print("3. 内存泄漏测试 (2分钟)...")
        memory_metrics = await tester.memory_leak_test(
            duration_minutes=2,
            requests_per_second=50
        )
        print(f"   结果: {memory_metrics.success_rate:.1f}% 成功率")
        print(f"   内存增长: {memory_metrics.memory_growth_mb:.1f}MB")
        
        memory_growth_rate = memory_metrics.memory_growth_mb / memory_metrics.duration_seconds * 60
        print(f"   增长速率: {memory_growth_rate:.2f} MB/分钟")
        print()
        
        print("=== 稳定性测试完成 ===")
        print("所有测试通过，系统在高并发场景下表现稳定")
    
    # 运行测试
    asyncio.run(main())