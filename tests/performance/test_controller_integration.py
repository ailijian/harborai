#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerformanceTestController 集成测试

本模块包含对性能测试控制器的集成测试。

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


class TestPerformanceTestControllerIntegration:
    """PerformanceTestController 集成测试类"""
    
    @pytest.fixture
    def integration_config(self):
        """集成测试配置fixture"""
        return PerformanceConfig(
            test_duration=5.0,
            warmup_duration=1.0,
            cooldown_duration=0.5,
            max_concurrent_users=3,
            response_time_threshold=2.0,
            cpu_usage_threshold=80.0
        )
    
    @pytest.fixture
    def integration_controller(self, integration_config):
        """集成测试控制器fixture"""
        return PerformanceTestController(integration_config)

    @pytest.mark.asyncio
    async def test_real_performance_test_simulation(self, integration_controller):
        """测试真实性能测试模拟"""
        
        # 注册CPU密集型测试
        async def cpu_intensive_test(**kwargs):
            """模拟CPU密集型测试"""
            start = time.time()
            # 模拟CPU密集型操作
            total = 0
            for i in range(100000):
                total += i * i
            
            duration = time.time() - start
            return {
                'response_time': duration,
                'success_rate': 1.0,
                'throughput': 1.0 / duration if duration > 0 else 0,
                'error_count': 0,
                'cpu_usage': 50.0
            }
        
        # 注册内存测试
        async def memory_test(**kwargs):
            """模拟内存测试"""
            # 创建一些内存使用
            data = [i for i in range(10000)]
            await asyncio.sleep(0.1)
            
            return {
                'response_time': 0.1,
                'success_rate': 1.0,
                'throughput': 10.0,
                'error_count': 0,
                'memory_usage': len(data) * 8  # 估算内存使用
            }
        
        integration_controller.register_test_runner(TestType.EXECUTION_EFFICIENCY, cpu_intensive_test)
        integration_controller.register_test_runner(TestType.MEMORY_MONITORING, memory_test)
        
        # 运行CPU测试
        cpu_metrics = await integration_controller.run_single_test(
            TestType.EXECUTION_EFFICIENCY, 
            "CPU密集型测试"
        )
        
        assert cpu_metrics.status == TestStatus.COMPLETED
        assert cpu_metrics.response_time > 0
        assert cpu_metrics.success_rate == 1.0
        
        # 运行内存测试
        memory_metrics = await integration_controller.run_single_test(
            TestType.MEMORY_MONITORING, 
            "内存使用测试"
        )
        
        assert memory_metrics.status == TestStatus.COMPLETED
        assert memory_metrics.response_time > 0
        assert memory_metrics.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_error_recovery_and_continuation(self, integration_controller):
        """测试错误恢复和继续执行"""
        
        call_count = 0
        
        async def intermittent_failing_test(**kwargs):
            """间歇性失败的测试"""
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # 第一次调用失败
                raise Exception("模拟网络错误")
            else:
                # 后续调用成功
                return {
                    'response_time': 0.2,
                    'success_rate': 1.0,
                    'throughput': 5.0,
                    'error_count': 0
                }
        
        integration_controller.register_test_runner(
            TestType.RESPONSE_TIME, 
            intermittent_failing_test
        )
        
        # 第一次运行应该失败
        first_metrics = await integration_controller.run_single_test(
            TestType.RESPONSE_TIME, 
            "间歇性失败测试"
        )
        
        assert first_metrics.status == TestStatus.FAILED
        assert len(first_metrics.error_details) > 0
        
        # 第二次运行应该成功
        second_metrics = await integration_controller.run_single_test(
            TestType.RESPONSE_TIME, 
            "间歇性失败测试重试"
        )
        
        assert second_metrics.status == TestStatus.COMPLETED
        assert second_metrics.success_rate == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])