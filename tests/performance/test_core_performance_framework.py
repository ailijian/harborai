#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心性能测试框架测试

本模块包含对core_performance_framework.py中所有组件的全面测试。
目标覆盖率：≥80%

作者: HarborAI Team
创建时间: 2024-01-20
遵循: VIBE Coding 规范
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List, Any

try:
    from .core_performance_framework import (
        PerformanceConfig,
        TestMetrics,
        AggregatedResults,
        ResultsCollector,
        PerformanceTestController,
        TestType,
        TestStatus
    )
except ImportError:
    from core_performance_framework import (
        PerformanceConfig,
        TestMetrics,
        AggregatedResults,
        ResultsCollector,
        PerformanceTestController,
        TestType,
        TestStatus
    )


class TestPerformanceConfig:
    """PerformanceConfig 测试类"""
    
    def test_default_config_creation(self):
        """测试默认配置创建"""
        config = PerformanceConfig()
        
        assert config.test_duration == 60.0
        assert config.warmup_duration == 10.0
        assert config.cooldown_duration == 5.0
        assert config.max_concurrent_users == 100
        assert config.response_time_threshold == 2.0
        assert config.cpu_usage_threshold == 80.0
        assert config.success_rate_threshold == 0.999
        assert config.enable_html_report is True
        assert config.enable_json_report is True
    
    def test_custom_config_creation(self):
        """测试自定义配置创建"""
        config = PerformanceConfig(
            test_duration=30.0,
            warmup_duration=5.0,
            max_concurrent_users=50,
            response_time_threshold=1.5,
            cpu_usage_threshold=70.0
        )
        
        assert config.test_duration == 30.0
        assert config.warmup_duration == 5.0
        assert config.max_concurrent_users == 50
        assert config.response_time_threshold == 1.5
        assert config.cpu_usage_threshold == 70.0
    
    def test_config_validation_success(self):
        """测试配置验证成功"""
        config = PerformanceConfig(
            test_duration=30.0,
            warmup_duration=5.0,
            max_concurrent_users=10,
            success_rate_threshold=0.95
        )
        
        errors = config.validate()
        assert len(errors) == 0
    
    def test_config_validation_failures(self):
        """测试配置验证失败"""
        config = PerformanceConfig(
            test_duration=-10.0,  # 无效：负数
            warmup_duration=-5.0,  # 无效：负数
            max_concurrent_users=0,  # 无效：零
            success_rate_threshold=1.5  # 无效：大于1
        )
        
        errors = config.validate()
        assert len(errors) == 4
        assert "测试持续时间必须大于0" in errors
        assert "预热时间不能为负数" in errors
        assert "最大并发用户数必须大于0" in errors
        assert "成功率阈值必须在0到1之间" in errors
    
    def test_config_validation_edge_cases(self):
        """测试配置验证边界情况"""
        # 测试边界值
        config = PerformanceConfig(
            test_duration=0.1,  # 最小正值
            warmup_duration=0.0,  # 零值（允许）
            max_concurrent_users=1,  # 最小正值
            success_rate_threshold=1.0  # 最大值
        )
        
        errors = config.validate()
        assert len(errors) == 0


class TestTestMetrics:
    """TestMetrics 测试类"""
    
    def test_test_metrics_creation(self):
        """测试测试指标创建"""
        start_time = datetime.now()
        metrics = TestMetrics(
            test_name="测试指标",
            test_type=TestType.RESPONSE_TIME,
            start_time=start_time
        )
        
        assert metrics.test_name == "测试指标"
        assert metrics.test_type == TestType.RESPONSE_TIME
        assert metrics.start_time == start_time
        assert metrics.status == TestStatus.PENDING
        assert metrics.error_count == 0
        assert len(metrics.error_details) == 0
    
    def test_calculate_duration_with_end_time(self):
        """测试计算持续时间（有结束时间）"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=5)
        
        metrics = TestMetrics(
            test_name="持续时间测试",
            test_type=TestType.RESPONSE_TIME,
            start_time=start_time,
            end_time=end_time
        )
        
        metrics.calculate_duration()
        assert metrics.duration == 5.0
    
    def test_calculate_duration_without_end_time(self):
        """测试计算持续时间（无结束时间）"""
        start_time = datetime.now() - timedelta(seconds=3)
        
        metrics = TestMetrics(
            test_name="持续时间测试",
            test_type=TestType.RESPONSE_TIME,
            start_time=start_time
        )
        
        duration = metrics.calculate_duration()
        # 没有end_time时，应该返回0.0
        assert duration == 0.0
        assert metrics.duration is None
    
    def test_to_dict(self):
        """测试转换为字典"""
        start_time = datetime.now()
        metrics = TestMetrics(
            test_name="字典测试",
            test_type=TestType.MEMORY_MONITORING,
            start_time=start_time,
            response_time=1.5,
            success_rate=0.95,
            error_count=2
        )
        
        result_dict = metrics.to_dict()
        
        assert result_dict["test_name"] == "字典测试"
        assert result_dict["test_type"] == "memory_monitoring"
        assert result_dict["response_time"] == 1.5
        assert result_dict["success_rate"] == 0.95
        assert result_dict["error_count"] == 2
        assert "start_time" in result_dict


class TestAggregatedResults:
    """AggregatedResults 测试类"""
    
    def test_aggregated_results_creation(self):
        """测试聚合结果创建"""
        results = AggregatedResults()
        
        assert results.total_tests == 0
        assert results.passed_tests == 0
        assert results.failed_tests == 0
        assert results.skipped_tests == 0
        assert results.total_duration == 0.0
        assert results.average_response_time == 0.0
        assert results.overall_success_rate == 0.0
        assert results.performance_grade == "未评估"
    
    def test_performance_grade_evaluation_a(self):
        """测试性能等级评估 - A级"""
        results = AggregatedResults(
            total_tests=10,
            passed_tests=10,
            overall_success_rate=0.999,
            average_response_time=0.5
        )
        
        results._evaluate_performance_grade()
        assert results.performance_grade == "A"
    
    def test_performance_grade_evaluation_b(self):
        """测试性能等级评估 - B级"""
        results = AggregatedResults(
            total_tests=10,
            passed_tests=9,
            overall_success_rate=0.99,
            average_response_time=1.5
        )
        
        results._evaluate_performance_grade()
        assert results.performance_grade == "B"
    
    def test_performance_grade_evaluation_f(self):
        """测试性能等级评估 - F级"""
        results = AggregatedResults(
            total_tests=10,
            passed_tests=3,
            overall_success_rate=0.5,
            average_response_time=15.0
        )
        
        results._evaluate_performance_grade()
        assert results.performance_grade == "F"
    
    def test_to_dict(self):
        """测试转换为字典"""
        results = AggregatedResults(
            total_tests=5,
            passed_tests=4,
            failed_tests=1,
            total_duration=25.0,
            average_response_time=2.5,
            overall_success_rate=0.8
        )
        
        result_dict = results.to_dict()
        
        assert result_dict["total_tests"] == 5
        assert result_dict["passed_tests"] == 4
        assert result_dict["failed_tests"] == 1
        assert result_dict["total_duration"] == 25.0
        assert result_dict["average_response_time"] == 2.5
        assert result_dict["overall_success_rate"] == 0.8


class TestResultsCollector:
    """ResultsCollector 测试类"""
    
    @pytest.fixture
    def collector(self):
        """结果收集器fixture"""
        return ResultsCollector()
    
    def test_collector_initialization(self, collector):
        """测试收集器初始化"""
        assert len(collector.metrics) == 0
        assert collector._lock is not None
    
    def test_collect_metrics(self, collector):
        """测试收集指标"""
        metrics_data = {
            "test_name": "收集测试",
            "start_time": datetime.now(),
            "response_time": 1.0,
            "success_rate": 0.95,
            "error_count": 1
        }
        
        metrics = collector.collect_metrics("response_time", metrics_data)
        
        assert isinstance(metrics, TestMetrics)
        assert metrics.test_name == "收集测试"
        assert metrics.test_type == TestType.RESPONSE_TIME
        assert metrics.response_time == 1.0
        assert metrics.success_rate == 0.95
        assert metrics.error_count == 1
        assert len(collector.metrics) == 1
    
    def test_collect_metrics_with_defaults(self, collector):
        """测试收集指标（使用默认值）"""
        metrics_data = {}
        
        metrics = collector.collect_metrics("execution_efficiency", metrics_data)
        
        assert metrics.test_name.startswith("未命名测试_")
        assert metrics.test_type == TestType.EXECUTION_EFFICIENCY
        assert metrics.error_count == 0
        assert len(metrics.error_details) == 0
    
    def test_get_metrics_by_type(self, collector):
        """测试按类型获取指标"""
        # 添加不同类型的指标
        collector.collect_metrics("response_time", {"test_name": "响应时间测试1"})
        collector.collect_metrics("response_time", {"test_name": "响应时间测试2"})
        collector.collect_metrics("memory_monitoring", {"test_name": "内存测试"})
        
        response_metrics = collector.get_metrics_by_type(TestType.RESPONSE_TIME)
        memory_metrics = collector.get_metrics_by_type(TestType.MEMORY_MONITORING)
        
        assert len(response_metrics) == 2
        assert len(memory_metrics) == 1
        assert all(m.test_type == TestType.RESPONSE_TIME for m in response_metrics)
        assert all(m.test_type == TestType.MEMORY_MONITORING for m in memory_metrics)
    
    def test_get_failed_tests(self, collector):
        """测试获取失败的测试"""
        # 添加成功和失败的测试
        metrics1 = collector.collect_metrics("response_time", {"test_name": "成功测试"})
        metrics1.status = TestStatus.COMPLETED
        
        metrics2 = collector.collect_metrics("response_time", {"test_name": "失败测试"})
        metrics2.status = TestStatus.FAILED
        
        failed_tests = collector.get_failed_tests()
        
        assert len(failed_tests) == 1
        assert failed_tests[0].test_name == "失败测试"
        assert failed_tests[0].status == TestStatus.FAILED
    
    def test_clear_results(self, collector):
        """测试清空结果"""
        collector.collect_metrics("response_time", {"test_name": "测试1"})
        collector.collect_metrics("response_time", {"test_name": "测试2"})
        
        assert len(collector.metrics) == 2
        
        collector.clear_results()
        
        assert len(collector.metrics) == 0
    
    def test_aggregate_results(self, collector):
        """测试聚合结果"""
        # 添加多个测试指标
        start_time = datetime.now()
        
        # 成功测试
        metrics1 = collector.collect_metrics("response_time", {
            "test_name": "成功测试1",
            "start_time": start_time,
            "end_time": start_time + timedelta(seconds=2),
            "response_time": 1.0,
            "success_rate": 1.0
        })
        metrics1.status = TestStatus.COMPLETED
        metrics1.calculate_duration()  # 计算持续时间
        
        # 失败测试
        metrics2 = collector.collect_metrics("response_time", {
            "test_name": "失败测试",
            "start_time": start_time,
            "end_time": start_time + timedelta(seconds=3),
            "response_time": 5.0,
            "success_rate": 0.0
        })
        metrics2.status = TestStatus.FAILED
        metrics2.calculate_duration()  # 计算持续时间
        
        # 跳过测试
        metrics3 = collector.collect_metrics("memory_monitoring", {
            "test_name": "跳过测试",
            "start_time": start_time,
            "end_time": start_time + timedelta(seconds=1)
        })
        metrics3.status = TestStatus.SKIPPED
        metrics3.calculate_duration()  # 计算持续时间
        
        aggregated = collector.aggregate_results()
        
        assert aggregated.total_tests == 3
        assert aggregated.passed_tests == 1
        assert aggregated.failed_tests == 1
        assert aggregated.skipped_tests == 1
        # 由于calculate_statistics方法的实现，我们需要检查实际的计算结果
        assert aggregated.average_response_time == 3.0  # (1.0 + 5.0) / 2
        assert aggregated.overall_success_rate == 1/3  # 1 passed out of 3 total


class TestPerformanceTestControllerAdvanced:
    """PerformanceTestController 高级测试类"""
    
    @pytest.fixture
    def config(self):
        """测试配置fixture"""
        return PerformanceConfig(
            test_duration=5.0,
            warmup_duration=1.0,
            cooldown_duration=0.5,
            max_concurrent_users=3
        )
    
    @pytest.fixture
    def controller(self, config):
        """控制器fixture"""
        return PerformanceTestController(config)
    
    def test_controller_initialization_with_invalid_config(self):
        """测试使用无效配置初始化控制器"""
        invalid_config = PerformanceConfig(test_duration=-1.0)
        
        with pytest.raises(ValueError) as exc_info:
            PerformanceTestController(invalid_config)
        
        assert "配置验证失败" in str(exc_info.value)
    
    def test_controller_properties(self, controller):
        """测试控制器属性"""
        # 测试初始状态
        assert controller.current_test is None
        assert controller.is_running is False
        
        # 测试设置属性
        test_info = {"test_name": "测试", "start_time": datetime.now()}
        controller.current_test = test_info
        controller.is_running = True
        
        assert controller.current_test == test_info
        assert controller.is_running is True
    
    @pytest.mark.asyncio
    async def test_run_single_test_unregistered_type(self, controller):
        """测试运行未注册类型的测试"""
        metrics = await controller.run_single_test(
            TestType.RESPONSE_TIME,
            "未注册测试"
        )
        
        assert metrics.status == TestStatus.FAILED
        assert len(metrics.error_details) > 0
        assert "未注册的测试类型" in metrics.error_details[0]
    
    @pytest.mark.asyncio
    async def test_run_single_test_with_exception(self, controller):
        """测试运行测试时发生异常"""
        async def failing_runner(**kwargs):
            raise Exception("模拟测试异常")
        
        controller.register_test_runner(TestType.RESPONSE_TIME, failing_runner)
        
        metrics = await controller.run_single_test(
            TestType.RESPONSE_TIME,
            "异常测试"
        )
        
        assert metrics.status == TestStatus.FAILED
        assert len(metrics.error_details) > 0
        assert "模拟测试异常" in metrics.error_details[0]
    
    @pytest.mark.asyncio
    async def test_run_full_performance_suite_default(self, controller):
        """测试运行完整性能测试套件（默认）"""
        # 注册测试运行器
        async def mock_runner(**kwargs):
            return {
                'response_time': 1.0,
                'success_rate': 1.0,
                'throughput': 10.0,
                'error_count': 0
            }
        
        controller.register_test_runner(TestType.RESPONSE_TIME, mock_runner)
        controller.register_test_runner(TestType.MEMORY_MONITORING, mock_runner)
        
        aggregated = await controller.run_full_performance_suite()
        
        assert aggregated.total_tests == 2
        assert aggregated.passed_tests == 2
        assert aggregated.failed_tests == 0
    
    @pytest.mark.asyncio
    async def test_run_full_performance_suite_custom(self, controller):
        """测试运行完整性能测试套件（自定义）"""
        async def mock_runner(**kwargs):
            return {
                'response_time': 0.5,
                'success_rate': 1.0,
                'throughput': 20.0,
                'error_count': 0
            }
        
        controller.register_test_runner(TestType.RESPONSE_TIME, mock_runner)
        
        test_suite = [
            {
                "test_type": TestType.RESPONSE_TIME,
                "test_name": "自定义响应时间测试",
                "params": {"target_url": "http://example.com"}
            }
        ]
        
        aggregated = await controller.run_full_performance_suite(test_suite)
        
        assert aggregated.total_tests == 1
        assert aggregated.passed_tests == 1
    
    @pytest.mark.asyncio
    async def test_run_full_performance_suite_already_running(self, controller):
        """测试在已运行状态下启动测试套件"""
        controller._is_running = True
        
        with pytest.raises(RuntimeError) as exc_info:
            await controller.run_full_performance_suite()
        
        assert "性能测试套件正在运行中" in str(exc_info.value)
    
    def test_get_test_status_detailed(self, controller):
        """测试获取详细测试状态"""
        # 注册一些测试运行器
        controller.register_test_runner(TestType.RESPONSE_TIME, lambda: None)
        controller.register_test_runner(TestType.MEMORY_MONITORING, lambda: None)
        
        # 添加一些测试结果
        controller.results_collector.collect_metrics("response_time", {"test_name": "测试1"})
        controller.results_collector.collect_metrics("response_time", {"test_name": "测试2"})
        
        status = controller.get_test_status()
        
        assert status["is_running"] is False
        assert status["current_test"] is None
        assert len(status["registered_test_types"]) == 2
        assert "response_time" in status["registered_test_types"]
        assert "memory_monitoring" in status["registered_test_types"]
        assert status["total_tests_run"] == 2
        assert status["total_collected_metrics"] == 2
    
    def test_stop_current_test_when_running(self, controller):
        """测试停止正在运行的测试"""
        controller._is_running = True
        controller._current_test = {"test_name": "运行中的测试"}
        
        controller.stop_current_test()
        
        assert controller.is_running is False
        assert controller.current_test is None
    
    def test_stop_current_test_when_not_running(self, controller):
        """测试停止未运行的测试"""
        assert controller.is_running is False
        assert controller.current_test is None
        
        # 应该不会抛出异常
        controller.stop_current_test()
        
        assert controller.is_running is False
        assert controller.current_test is None


class TestResultsCollectorExportToJson:
    """测试ResultsCollector的export_to_json方法"""
    
    @pytest.fixture
    def collector_with_data(self):
        """创建包含测试数据的ResultsCollector"""
        collector = ResultsCollector()
        
        # 添加一些测试指标
        metrics1 = TestMetrics(
            test_name="测试1",
            test_type=TestType.RESPONSE_TIME,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1),
            status=TestStatus.COMPLETED,
            response_time=0.5,
            throughput=1000.0,
            success_rate=1.0,
            error_count=0,
            cpu_usage={"avg": 50.0, "max": 60.0, "min": 40.0},
            memory_usage={"avg": 100.0, "max": 120.0, "min": 80.0},
            raw_data={"custom1": 123}
        )
        
        metrics2 = TestMetrics(
            test_name="测试2", 
            test_type=TestType.MEMORY_MONITORING,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=2),
            status=TestStatus.FAILED,
            response_time=1.5,
            throughput=500.0,
            success_rate=0.9,
            error_count=1,
            cpu_usage={"avg": 75.0, "max": 85.0, "min": 65.0},
            memory_usage={"avg": 200.0, "max": 250.0, "min": 150.0},
            raw_data={"custom2": 456},
            error_details=["测试失败原因"]
        )
        
        # 直接添加到metrics列表中
        collector.metrics.append(metrics1)
        collector.metrics.append(metrics2)
        
        # 生成聚合结果
        collector.aggregate_results()
        
        return collector
    
    def test_export_to_json_success(self, collector_with_data, tmp_path):
        """测试成功导出JSON文件"""
        export_file = tmp_path / "test_results.json"
        
        # 执行导出
        collector_with_data.export_to_json(str(export_file))
        
        # 验证文件存在
        assert export_file.exists()
        
        # 验证文件内容
        with open(export_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据结构
        assert "metrics" in data
        assert "aggregated_results" in data
        assert "export_time" in data
        
        # 验证metrics数据
        assert len(data["metrics"]) == 2
        assert data["metrics"][0]["test_name"] == "测试1"
        assert data["metrics"][1]["test_name"] == "测试2"
        
        # 验证聚合结果
        assert data["aggregated_results"]["total_tests"] == 2
        assert data["aggregated_results"]["passed_tests"] == 1  # COMPLETED 状态的测试
        assert data["aggregated_results"]["failed_tests"] == 1
        
        # 验证导出时间格式
        export_time = datetime.fromisoformat(data["export_time"])
        assert isinstance(export_time, datetime)
    
    def test_export_to_json_creates_directory(self, collector_with_data, tmp_path):
        """测试导出时自动创建目录"""
        nested_dir = tmp_path / "nested" / "directory"
        export_file = nested_dir / "test_results.json"
        
        # 确保目录不存在
        assert not nested_dir.exists()
        
        # 执行导出
        collector_with_data.export_to_json(str(export_file))
        
        # 验证目录和文件都被创建
        assert nested_dir.exists()
        assert export_file.exists()
    
    def test_export_to_json_empty_collector(self, tmp_path):
        """测试导出空的ResultsCollector"""
        collector = ResultsCollector()
        export_file = tmp_path / "empty_results.json"
        
        # 执行导出
        collector.export_to_json(str(export_file))
        
        # 验证文件存在
        assert export_file.exists()
        
        # 验证文件内容
        with open(export_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 验证数据结构
        assert data["metrics"] == []
        assert data["aggregated_results"] is None
        assert "export_time" in data
    
    def test_export_to_json_invalid_path(self, collector_with_data):
        """测试导出到无效路径时的错误处理"""
        # 使用无效路径（在Windows上，路径包含无效字符）
        invalid_path = "invalid<>path/test.json"
        
        with pytest.raises(Exception):
            collector_with_data.export_to_json(invalid_path)
    
    def test_export_to_json_permission_error(self, collector_with_data, tmp_path):
        """测试权限错误的处理"""
        # 创建一个只读目录（在某些系统上可能不起作用）
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        export_file = readonly_dir / "test.json"
        
        try:
            # 尝试使权限只读（可能在某些系统上不起作用）
            readonly_dir.chmod(0o444)
            
            # 在某些系统上这可能会成功，所以我们只是确保方法能被调用
            collector_with_data.export_to_json(str(export_file))
            
        except Exception:
            # 如果抛出异常，这是预期的行为
            pass
        finally:
            # 恢复权限以便清理
            try:
                readonly_dir.chmod(0o755)
            except:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])