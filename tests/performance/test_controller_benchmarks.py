#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerformanceTestController 基准测试

本模块包含对性能测试控制器的基准测试。

作者: HarborAI Team
创建时间: 2024-01-20
遵循: VIBE Coding 规范
"""

import pytest
import asyncio
import time
from datetime import datetime

try:
    from .core_performance_framework import (
        PerformanceTestController,
        PerformanceConfig,
        TestType,
        TestStatus,
        TestMetrics
    )
except ImportError:
    from core_performance_framework import (
        PerformanceTestController,
        PerformanceConfig,
        TestType,
        TestStatus,
        TestMetrics
    )


class TestPerformanceTestControllerBenchmarks:
    """PerformanceTestController 基准测试类"""
    
    @pytest.fixture
    def benchmark_config(self):
        """基准测试配置fixture"""
        return PerformanceConfig(
            test_duration=2.0,
            warmup_duration=0.5,
            cooldown_duration=0.2,
            max_concurrent_users=10,
            response_time_threshold=1.0,
            cpu_usage_threshold=90.0
        )
    
    @pytest.fixture
    def benchmark_controller(self, benchmark_config):
        """基准测试控制器fixture"""
        return PerformanceTestController(benchmark_config)

    def test_controller_creation_benchmark(self, benchmark_config):
        """基准测试：控制器创建性能"""
        
        start_time = time.time()
        controller = PerformanceTestController(benchmark_config)
        end_time = time.time()
        
        creation_time = end_time - start_time
        assert creation_time < 1.0  # 创建时间应该小于1秒
        assert controller is not None

    @pytest.mark.asyncio
    async def test_single_test_execution_benchmark(self, benchmark_controller):
        """基准测试：单个测试执行性能"""
        
        async def fast_test(**kwargs):
            """快速测试运行器"""
            await asyncio.sleep(0.01)  # 10ms 模拟
            return {
                'response_time': 0.01,
                'success_rate': 1.0,
                'throughput': 100.0,
                'error_count': 0
            }
        
        benchmark_controller.register_test_runner(TestType.RESPONSE_TIME, fast_test)
        
        start_time = time.time()
        metrics = await benchmark_controller.run_single_test(
            TestType.RESPONSE_TIME, 
            "基准测试"
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time < 5.0  # 执行时间应该合理
        assert metrics.status == TestStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_multiple_tests_throughput(self, benchmark_controller):
        """基准测试：多个测试的吞吐量"""
        
        async def throughput_test(**kwargs):
            """吞吐量测试运行器"""
            await asyncio.sleep(0.005)  # 5ms 模拟
            return {
                'response_time': 0.005,
                'success_rate': 1.0,
                'throughput': 200.0,
                'error_count': 0
            }
        
        benchmark_controller.register_test_runner(TestType.RESPONSE_TIME, throughput_test)
        
        # 测试连续执行多个测试的性能
        start_time = time.time()
        
        tasks = []
        for i in range(5):  # 减少测试数量以避免过长执行时间
            task = benchmark_controller.run_single_test(
                TestType.RESPONSE_TIME, 
                f"吞吐量测试 {i+1}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # 验证所有测试都成功完成
        for metrics in results:
            assert metrics.status == TestStatus.COMPLETED
        
        # 验证吞吐量性能
        tests_per_second = len(results) / total_duration
        assert tests_per_second > 0.5  # 至少每秒0.5个测试
        
        print(f"吞吐量: {tests_per_second:.2f} 测试/秒")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])