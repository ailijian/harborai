#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试控制器测试模块

测试performance_test_controller.py的核心功能
遵循VIBE Coding规范
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil

try:
    from .performance_test_controller import (
        PerformanceTestController,
        TestType,
        TestConfiguration,
        TestResult,
        TestStatus
    )
except ImportError:
    from performance_test_controller import (
        PerformanceTestController,
        TestType,
        TestConfiguration,
        TestResult,
        TestStatus
    )


class TestPerformanceTestController:
    """性能测试控制器测试类"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def controller(self, temp_output_dir):
        """创建性能测试控制器实例"""
        return PerformanceTestController(output_dir=temp_output_dir)
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_controller_initialization(self, controller, temp_output_dir):
        """测试控制器初始化"""
        assert controller is not None
        assert controller.output_dir == temp_output_dir
        assert controller.status == TestStatus.IDLE
        assert len(controller.test_results) == 0
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_configure_test(self, controller):
        """测试测试配置"""
        config = TestConfiguration(
            test_type=TestType.EXECUTION_EFFICIENCY,
            duration=60,
            iterations=100
        )
        
        controller.configure_test(config)
        assert controller.current_config == config
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_get_available_tests(self, controller):
        """测试获取可用测试类型"""
        available_tests = controller.get_available_tests()
        
        assert TestType.EXECUTION_EFFICIENCY in available_tests
        assert TestType.MEMORY_LEAK in available_tests
        assert TestType.RESOURCE_UTILIZATION in available_tests
        assert TestType.RESPONSE_TIME in available_tests
        assert TestType.CONCURRENCY in available_tests
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_validate_configuration(self, controller):
        """测试配置验证"""
        # 有效配置
        valid_config = TestConfiguration(
            test_type=TestType.EXECUTION_EFFICIENCY,
            duration=60,
            iterations=100
        )
        assert controller.validate_configuration(valid_config) is True
        
        # 无效配置 - 持续时间太短
        invalid_config = TestConfiguration(
            test_type=TestType.EXECUTION_EFFICIENCY,
            duration=0,
            iterations=100
        )
        assert controller.validate_configuration(invalid_config) is False
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_get_test_status(self, controller):
        """测试获取测试状态"""
        assert controller.get_test_status() == TestStatus.IDLE
        
        # 模拟运行状态
        controller.status = TestStatus.RUNNING
        assert controller.get_test_status() == TestStatus.RUNNING
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_get_test_results(self, controller):
        """测试获取测试结果"""
        # 初始状态应该没有结果
        results = controller.get_test_results()
        assert len(results) == 0
        
        # 添加模拟结果
        mock_result = TestResult(
            test_type=TestType.EXECUTION_EFFICIENCY,
            status=TestStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time() + 60,
            metrics={"avg_response_time": 0.5}
        )
        controller.test_results.append(mock_result)
        
        results = controller.get_test_results()
        assert len(results) == 1
        assert results[0].test_type == TestType.EXECUTION_EFFICIENCY
    
    @pytest.mark.integration
    @pytest.mark.p2
    @patch('tests.performance.performance_test_controller.ExecutionEfficiencyTester')
    def test_run_single_test_execution_efficiency(self, mock_tester, controller):
        """测试运行单个执行效率测试"""
        # 配置模拟测试器
        mock_instance = Mock()
        mock_instance.run_test.return_value = {
            "avg_response_time": 0.5,
            "throughput": 100,
            "success_rate": 0.99
        }
        mock_tester.return_value = mock_instance
        
        config = TestConfiguration(
            test_type=TestType.EXECUTION_EFFICIENCY,
            duration=10,  # 短时间测试
            iterations=10
        )
        
        result = controller.run_single_test(config)
        
        assert result is not None
        assert result.test_type == TestType.EXECUTION_EFFICIENCY
        assert result.status == TestStatus.COMPLETED
        assert "avg_response_time" in result.metrics
    
    @pytest.mark.integration
    @pytest.mark.p2
    @patch('tests.performance.performance_test_controller.MemoryLeakDetector')
    def test_run_single_test_memory_leak(self, mock_detector, controller):
        """测试运行单个内存泄漏测试"""
        # 配置模拟检测器
        mock_instance = Mock()
        mock_instance.run_test.return_value = {
            "memory_growth": 0.1,
            "leak_detected": False,
            "peak_memory": 100
        }
        mock_detector.return_value = mock_instance
        
        config = TestConfiguration(
            test_type=TestType.MEMORY_LEAK,
            duration=10,
            iterations=10
        )
        
        result = controller.run_single_test(config)
        
        assert result is not None
        assert result.test_type == TestType.MEMORY_LEAK
        assert result.status == TestStatus.COMPLETED
        assert "memory_growth" in result.metrics
    
    @pytest.mark.integration
    @pytest.mark.p3
    def test_run_test_suite_quick(self, controller):
        """测试运行快速测试套件"""
        with patch.multiple(
            'tests.performance.performance_test_controller',
            ExecutionEfficiencyTester=Mock(),
            MemoryLeakDetector=Mock(),
            ResourceUtilizationMonitor=Mock()
        ):
            # 配置所有模拟对象返回成功结果
            for mock_class in [
                'ExecutionEfficiencyTester',
                'MemoryLeakDetector', 
                'ResourceUtilizationMonitor'
            ]:
                mock_instance = Mock()
                mock_instance.run_test.return_value = {"test": "success"}
                
            results = controller.run_test_suite(
                test_types=[TestType.EXECUTION_EFFICIENCY],
                duration=5,  # 很短的测试时间
                quick_mode=True
            )
            
            assert len(results) >= 1
            assert all(result.status == TestStatus.COMPLETED for result in results)
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_stop_current_test(self, controller):
        """测试停止当前测试"""
        # 模拟运行状态
        controller.status = TestStatus.RUNNING
        controller._stop_requested = False
        
        controller.stop_current_test()
        
        assert controller._stop_requested is True
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_cleanup_resources(self, controller):
        """测试资源清理"""
        # 添加一些模拟资源
        controller.test_results = [Mock(), Mock()]
        controller.status = TestStatus.RUNNING
        
        controller.cleanup_resources()
        
        # 验证资源被清理
        assert controller.status == TestStatus.IDLE
        # 注意：test_results通常不会被清理，因为它们是有价值的历史数据
    
    @pytest.mark.unit
    @pytest.mark.p3
    def test_error_handling(self, controller):
        """测试错误处理"""
        config = TestConfiguration(
            test_type=TestType.EXECUTION_EFFICIENCY,
            duration=10,
            iterations=10
        )
        
        # 模拟测试器抛出异常
        with patch('tests.performance.performance_test_controller.ExecutionEfficiencyTester') as mock_tester:
            mock_instance = Mock()
            mock_instance.run_test.side_effect = Exception("测试失败")
            mock_tester.return_value = mock_instance
            
            result = controller.run_single_test(config)
            
            assert result is not None
            assert result.status == TestStatus.FAILED
            assert "error" in result.metrics


class TestPerformanceTestControllerIntegration:
    """性能测试控制器集成测试类"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def controller(self, temp_output_dir):
        """创建性能测试控制器实例"""
        return PerformanceTestController(output_dir=temp_output_dir)
    
    @pytest.mark.integration
    @pytest.mark.p3
    def test_full_test_workflow(self, controller):
        """测试完整的测试工作流程"""
        # 这是一个集成测试，验证完整的工作流程
        with patch.multiple(
            'tests.performance.performance_test_controller',
            ExecutionEfficiencyTester=Mock(),
            PerformanceReportGenerator=Mock()
        ):
            # 1. 配置测试
            config = TestConfiguration(
                test_type=TestType.EXECUTION_EFFICIENCY,
                duration=5,
                iterations=5
            )
            controller.configure_test(config)
            
            # 2. 验证配置
            assert controller.validate_configuration(config)
            
            # 3. 运行测试
            result = controller.run_single_test(config)
            
            # 4. 验证结果
            assert result is not None
            assert result.test_type == TestType.EXECUTION_EFFICIENCY
            
            # 5. 获取结果
            results = controller.get_test_results()
            assert len(results) >= 1


class TestPerformanceTestControllerBenchmarks:
    """性能测试控制器基准测试类"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """创建临时输出目录"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def controller(self, temp_output_dir):
        """创建性能测试控制器实例"""
        return PerformanceTestController(output_dir=temp_output_dir)
    
    @pytest.mark.benchmark
    @pytest.mark.p3
    def test_controller_creation_benchmark(self, benchmark, temp_output_dir):
        """基准测试：控制器创建性能"""
        def create_controller():
            return PerformanceTestController(output_dir=temp_output_dir)
        
        result = benchmark(create_controller)
        assert result is not None
    
    @pytest.mark.benchmark
    @pytest.mark.p3
    def test_configuration_validation_benchmark(self, benchmark, controller):
        """基准测试：配置验证性能"""
        config = TestConfiguration(
            test_type=TestType.EXECUTION_EFFICIENCY,
            duration=60,
            iterations=100
        )
        
        def validate_config():
            return controller.validate_configuration(config)
        
        result = benchmark(validate_config)
        assert result is True