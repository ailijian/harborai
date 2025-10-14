#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerformanceTestController 综合测试模块

本模块整合了性能测试控制器的所有测试类型：
- 单元测试：基础功能验证
- 集成测试：组件协作验证
- 基准测试：性能指标验证

遵循VIBE编码规范的测试金字塔原则：
- 第一层：单元测试 (Unit Tests) - 验证单个组件行为
- 第二层：集成测试 (Integration Tests) - 验证组件间交互
- 第三层：基准测试 (Benchmark Tests) - 验证性能指标

作者: HarborAI Team
创建时间: 2025-01-27
遵循: VIBE Coding 规范
"""

import pytest
import asyncio
import time
from unittest.mock import Mock
from datetime import datetime
from typing import Dict, Any, List

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


class TestPerformanceTestControllerUnit:
    """
    第一层：单元测试
    
    验证PerformanceTestController的基础功能和单个方法行为。
    """
    
    @pytest.fixture
    def default_config(self):
        """默认配置fixture"""
        return PerformanceConfig(
            max_concurrent_tests=5,
            test_timeout=30.0,
            retry_attempts=3,
            metrics_collection_interval=1.0,
            enable_detailed_logging=True
        )
    
    @pytest.fixture
    def controller(self, default_config):
        """控制器实例fixture"""
        return PerformanceTestController(default_config)
    
    @pytest.fixture
    def mock_test_runner(self):
        """模拟测试运行器fixture"""
        mock_runner = Mock()
        mock_runner.run_test.return_value = {
            "status": "success",
            "metrics": TestMetrics(
                response_time=0.5,
                throughput=100.0,
                error_rate=0.0,
                resource_usage={"cpu": 10.0, "memory": 50.0}
            )
        }
        return mock_runner
    
    def test_controller_initialization(self, default_config):
        """测试控制器初始化"""
        controller = PerformanceTestController(default_config)
        
        assert controller.config == default_config
        assert controller.status == TestStatus.IDLE
        assert len(controller.test_runners) == 0
        assert len(controller.test_results) == 0
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        valid_config = PerformanceConfig(
            max_concurrent_tests=1,
            test_timeout=10.0,
            retry_attempts=1
        )
        controller = PerformanceTestController(valid_config)
        assert controller.config.max_concurrent_tests == 1
        
        # 测试无效配置
        with pytest.raises(ValueError):
            PerformanceConfig(max_concurrent_tests=0)
    
    def test_register_test_runner(self, controller, mock_test_runner):
        """测试注册测试运行器"""
        controller.register_test_runner("test_runner", mock_test_runner)
        
        assert "test_runner" in controller.test_runners
        assert controller.test_runners["test_runner"] == mock_test_runner
    
    @pytest.mark.asyncio
    async def test_run_single_test_success(self, controller, mock_test_runner):
        """测试单个测试成功执行"""
        controller.register_test_runner("test_runner", mock_test_runner)
        
        result = await controller.run_test("test_runner", TestType.LATENCY)
        
        assert result["status"] == "success"
        assert "metrics" in result
        assert mock_test_runner.run_test.called
    
    def test_get_test_status_idle(self, controller):
        """测试获取空闲状态"""
        status = controller.get_test_status()
        
        assert status["current_status"] == TestStatus.IDLE
        assert status["active_tests"] == 0
        assert status["completed_tests"] == 0
    
    def test_stop_current_test(self, controller):
        """测试停止当前测试"""
        result = controller.stop_current_test()
        
        assert result["stopped"] is True
        assert result["previous_status"] == TestStatus.IDLE


class TestPerformanceTestControllerIntegration:
    """
    第二层：集成测试
    
    验证PerformanceTestController与其他组件的协作和完整工作流程。
    """
    
    @pytest.fixture
    def integration_config(self):
        """集成测试配置fixture"""
        return PerformanceConfig(
            max_concurrent_tests=3,
            test_timeout=60.0,
            retry_attempts=2,
            metrics_collection_interval=0.5,
            enable_detailed_logging=True
        )
    
    @pytest.fixture
    def integration_controller(self, integration_config):
        """集成测试控制器fixture"""
        return PerformanceTestController(integration_config)
    
    @pytest.mark.asyncio
    async def test_real_performance_test_simulation(self, integration_controller):
        """测试真实性能测试模拟"""
        # 模拟真实的测试运行器
        class MockRealTestRunner:
            async def run_test(self, test_type: TestType) -> Dict[str, Any]:
                # 模拟真实测试执行时间
                await asyncio.sleep(0.1)
                
                return {
                    "status": "success",
                    "metrics": TestMetrics(
                        response_time=0.15,
                        throughput=85.0,
                        error_rate=0.02,
                        resource_usage={"cpu": 15.0, "memory": 75.0}
                    ),
                    "test_type": test_type,
                    "timestamp": datetime.now()
                }
        
        # 注册测试运行器
        test_runner = MockRealTestRunner()
        integration_controller.register_test_runner("real_runner", test_runner)
        
        # 执行测试
        result = await integration_controller.run_test("real_runner", TestType.THROUGHPUT)
        
        # 验证结果
        assert result["status"] == "success"
        assert result["metrics"].throughput == 85.0
        assert result["test_type"] == TestType.THROUGHPUT
        assert "timestamp" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_test_execution(self, integration_controller):
        """测试并发测试执行"""
        class MockConcurrentTestRunner:
            def __init__(self, runner_id: str):
                self.runner_id = runner_id
            
            async def run_test(self, test_type: TestType) -> Dict[str, Any]:
                # 模拟不同的执行时间
                await asyncio.sleep(0.05 + (hash(self.runner_id) % 10) * 0.01)
                
                return {
                    "status": "success",
                    "runner_id": self.runner_id,
                    "metrics": TestMetrics(
                        response_time=0.1,
                        throughput=90.0,
                        error_rate=0.0,
                        resource_usage={"cpu": 12.0, "memory": 60.0}
                    )
                }
        
        # 注册多个测试运行器
        runners = [MockConcurrentTestRunner(f"runner_{i}") for i in range(3)]
        for i, runner in enumerate(runners):
            integration_controller.register_test_runner(f"runner_{i}", runner)
        
        # 并发执行测试
        tasks = [
            integration_controller.run_test(f"runner_{i}", TestType.LATENCY)
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 验证所有测试都成功
        assert len(results) == 3
        for result in results:
            assert result["status"] == "success"
            assert "runner_id" in result
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_continuation(self, integration_controller):
        """测试错误恢复和继续执行"""
        class MockErrorProneTestRunner:
            def __init__(self):
                self.call_count = 0
            
            async def run_test(self, test_type: TestType) -> Dict[str, Any]:
                self.call_count += 1
                
                # 第一次调用失败，第二次成功
                if self.call_count == 1:
                    raise Exception("模拟测试失败")
                
                return {
                    "status": "success",
                    "call_count": self.call_count,
                    "metrics": TestMetrics(
                        response_time=0.2,
                        throughput=70.0,
                        error_rate=0.1,
                        resource_usage={"cpu": 20.0, "memory": 80.0}
                    )
                }
        
        # 注册容易出错的测试运行器
        error_runner = MockErrorProneTestRunner()
        integration_controller.register_test_runner("error_runner", error_runner)
        
        # 第一次执行应该失败
        with pytest.raises(Exception):
            await integration_controller.run_test("error_runner", TestType.STRESS)
        
        # 第二次执行应该成功
        result = await integration_controller.run_test("error_runner", TestType.STRESS)
        assert result["status"] == "success"
        assert result["call_count"] == 2


class TestPerformanceTestControllerBenchmarks:
    """
    第三层：基准测试
    
    验证PerformanceTestController的性能指标和基准表现。
    """
    
    @pytest.fixture
    def benchmark_config(self):
        """基准测试配置fixture"""
        return PerformanceConfig(
            max_concurrent_tests=10,
            test_timeout=120.0,
            retry_attempts=1,
            metrics_collection_interval=0.1,
            enable_detailed_logging=False  # 减少日志开销
        )
    
    @pytest.fixture
    def benchmark_controller(self, benchmark_config):
        """基准测试控制器fixture"""
        return PerformanceTestController(benchmark_config)
    
    def test_controller_creation_benchmark(self, benchmark_config):
        """基准测试：控制器创建性能"""
        def create_controller():
            return PerformanceTestController(benchmark_config)
        
        # 测试控制器创建时间
        start_time = time.time()
        controller = create_controller()
        creation_time = time.time() - start_time
        
        # 验证创建时间在合理范围内（< 10ms）
        assert creation_time < 0.01
        assert controller is not None
    
    @pytest.mark.asyncio
    async def test_single_test_execution_benchmark(self, benchmark_controller):
        """基准测试：单个测试执行性能"""
        class BenchmarkTestRunner:
            async def run_test(self, test_type: TestType) -> Dict[str, Any]:
                # 最小化测试执行时间
                return {
                    "status": "success",
                    "metrics": TestMetrics(
                        response_time=0.001,
                        throughput=1000.0,
                        error_rate=0.0,
                        resource_usage={"cpu": 5.0, "memory": 30.0}
                    )
                }
        
        # 注册基准测试运行器
        benchmark_runner = BenchmarkTestRunner()
        benchmark_controller.register_test_runner("benchmark_runner", benchmark_runner)
        
        # 测试执行时间
        start_time = time.time()
        result = await benchmark_controller.run_test("benchmark_runner", TestType.LATENCY)
        execution_time = time.time() - start_time
        
        # 验证执行时间在合理范围内（< 50ms）
        assert execution_time < 0.05
        assert result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_multiple_tests_throughput(self, benchmark_controller):
        """基准测试：多测试吞吐量性能"""
        class HighThroughputTestRunner:
            async def run_test(self, test_type: TestType) -> Dict[str, Any]:
                return {
                    "status": "success",
                    "metrics": TestMetrics(
                        response_time=0.001,
                        throughput=2000.0,
                        error_rate=0.0,
                        resource_usage={"cpu": 3.0, "memory": 20.0}
                    )
                }
        
        # 注册高吞吐量测试运行器
        throughput_runner = HighThroughputTestRunner()
        benchmark_controller.register_test_runner("throughput_runner", throughput_runner)
        
        # 执行多个测试并测量吞吐量
        test_count = 100
        start_time = time.time()
        
        tasks = [
            benchmark_controller.run_test("throughput_runner", TestType.THROUGHPUT)
            for _ in range(test_count)
        ]
        
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # 计算吞吐量（测试/秒）
        throughput = test_count / total_time
        
        # 验证吞吐量满足基准要求（> 500 测试/秒）
        assert throughput > 500
        assert len(results) == test_count
        assert all(result["status"] == "success" for result in results)
    
    @pytest.mark.asyncio
    async def test_memory_usage_benchmark(self, benchmark_controller):
        """基准测试：内存使用性能"""
        import psutil
        import os
        
        class MemoryTestRunner:
            async def run_test(self, test_type: TestType) -> Dict[str, Any]:
                # 模拟一些内存使用
                data = [i for i in range(1000)]
                return {
                    "status": "success",
                    "data_size": len(data),
                    "metrics": TestMetrics(
                        response_time=0.01,
                        throughput=100.0,
                        error_rate=0.0,
                        resource_usage={"cpu": 8.0, "memory": 40.0}
                    )
                }
        
        # 注册内存测试运行器
        memory_runner = MemoryTestRunner()
        benchmark_controller.register_test_runner("memory_runner", memory_runner)
        
        # 测量内存使用
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 执行测试
        result = await benchmark_controller.run_test("memory_runner", TestType.STRESS)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # 验证内存增长在合理范围内（< 10MB）
        assert memory_increase < 10
        assert result["status"] == "success"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])