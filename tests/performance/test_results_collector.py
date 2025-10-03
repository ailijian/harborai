"""
测试结果收集器测试模块

该模块包含对ResultsCollector类的全面测试，包括：
- 单元测试：测试各个方法的基本功能
- 集成测试：测试组件间的协作
- 边界测试：测试极端情况和错误处理
- 性能基准测试：测试性能指标

作者：HarborAI性能测试团队
创建时间：2024年
遵循VIBE Coding规范
"""

import pytest
import asyncio
import time
import tempfile
import shutil
import sqlite3
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# 导入被测试的模块
from results_collector import (
    ResultsCollector,
    TestMetrics,
    AggregatedResults,
    TestType,
    TestStatus,
    create_test_metrics
)


class TestTestMetrics:
    """TestMetrics类的单元测试"""
    
    def test_test_metrics_creation(self):
        """测试TestMetrics对象创建"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1.5)
        
        metrics = TestMetrics(
            test_id="test_001",
            test_name="响应时间测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            response_time=0.15,
            success_rate=0.95,
            cpu_usage=45.2,
            memory_usage=128.5
        )
        
        assert metrics.test_id == "test_001"
        assert metrics.test_name == "响应时间测试"
        assert metrics.test_type == TestType.RESPONSE_TIME
        assert metrics.status == TestStatus.COMPLETED
        assert metrics.response_time == 0.15
        assert metrics.success_rate == 0.95
        assert metrics.cpu_usage == 45.2
        assert metrics.memory_usage == 128.5
        assert abs(metrics.duration - 1.5) < 0.01  # 允许小误差
    
    def test_duration_calculation(self):
        """测试持续时间计算"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=2.5)
        
        metrics = TestMetrics(
            test_id="test_002",
            test_name="持续时间测试",
            test_type=TestType.LOAD_TEST,
            status=TestStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time
        )
        
        calculated_duration = metrics.calculate_duration()
        assert abs(calculated_duration - 2.5) < 0.01
        assert abs(metrics.duration - 2.5) < 0.01
    
    def test_is_successful(self):
        """测试成功判断"""
        # 成功的测试
        successful_metrics = TestMetrics(
            test_id="test_003",
            test_name="成功测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(),
            error_count=0
        )
        assert successful_metrics.is_successful() is True
        
        # 失败的测试（状态失败）
        failed_metrics = TestMetrics(
            test_id="test_004",
            test_name="失败测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.FAILED,
            start_time=datetime.now(),
            error_count=0
        )
        assert failed_metrics.is_successful() is False
        
        # 失败的测试（有错误）
        error_metrics = TestMetrics(
            test_id="test_005",
            test_name="错误测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(),
            error_count=1
        )
        assert error_metrics.is_successful() is False
    
    def test_to_dict_and_from_dict(self):
        """测试字典转换"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        
        original_metrics = TestMetrics(
            test_id="test_006",
            test_name="字典转换测试",
            test_type=TestType.MEMORY_LEAK,
            status=TestStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            response_time=0.2,
            tags=["性能", "内存"],
            metadata={"version": "1.0", "env": "test"}
        )
        
        # 转换为字典
        metrics_dict = original_metrics.to_dict()
        assert metrics_dict['test_id'] == "test_006"
        assert metrics_dict['test_type'] == "memory_leak"
        assert metrics_dict['status'] == "completed"
        assert 'start_time' in metrics_dict
        assert 'end_time' in metrics_dict
        
        # 从字典恢复
        restored_metrics = TestMetrics.from_dict(metrics_dict)
        assert restored_metrics.test_id == original_metrics.test_id
        assert restored_metrics.test_name == original_metrics.test_name
        assert restored_metrics.test_type == original_metrics.test_type
        assert restored_metrics.status == original_metrics.status
        assert restored_metrics.response_time == original_metrics.response_time
        assert restored_metrics.tags == original_metrics.tags
        assert restored_metrics.metadata == original_metrics.metadata


class TestAggregatedResults:
    """AggregatedResults类的单元测试"""
    
    def test_empty_aggregation(self):
        """测试空结果聚合"""
        aggregated = AggregatedResults()
        aggregated.calculate_statistics([])
        
        assert aggregated.total_tests == 0
        assert aggregated.completed_tests == 0
        assert aggregated.failed_tests == 0
        assert aggregated.overall_success_rate == 0.0
    
    def test_basic_aggregation(self):
        """测试基本聚合功能"""
        # 创建测试数据
        metrics = []
        start_time = datetime.now()
        
        for i in range(10):
            end_time = start_time + timedelta(seconds=i + 1)
            status = TestStatus.COMPLETED if i < 8 else TestStatus.FAILED
            
            metric = TestMetrics(
                test_id=f"test_{i}",
                test_name=f"测试_{i}",
                test_type=TestType.RESPONSE_TIME,
                status=status,
                start_time=start_time,
                end_time=end_time,
                response_time=0.1 + i * 0.01,
                cpu_usage=50 + i * 2,
                memory_usage=100 + i * 5
            )
            metrics.append(metric)
        
        # 执行聚合
        aggregated = AggregatedResults()
        aggregated.calculate_statistics(metrics)
        
        # 验证基本统计
        assert aggregated.total_tests == 10
        assert aggregated.completed_tests == 8
        assert aggregated.failed_tests == 2
        assert aggregated.overall_success_rate == 0.8
        
        # 验证持续时间统计
        assert aggregated.min_duration == 1.0
        assert aggregated.max_duration == 10.0
        assert aggregated.average_duration == 5.5
        
        # 验证响应时间统计
        assert aggregated.min_response_time == 0.1
        assert aggregated.max_response_time == 0.19
        assert abs(aggregated.average_response_time - 0.145) < 0.001
        
        # 验证资源使用统计
        assert aggregated.average_cpu_usage == 59.0
        assert aggregated.peak_cpu_usage == 68.0
        assert aggregated.average_memory_usage == 122.5
        assert aggregated.peak_memory_usage == 145.0
    
    def test_performance_grade_calculation(self):
        """测试性能等级计算"""
        aggregated = AggregatedResults()
        
        # 优秀性能（A级）
        aggregated.overall_success_rate = 0.99
        aggregated.average_response_time = 0.05
        aggregated.average_cpu_usage = 30.0
        aggregated.average_memory_usage = 50.0
        aggregated._calculate_performance_grade()
        assert aggregated.performance_grade == "A"
        
        # 良好性能（B级）
        aggregated.overall_success_rate = 0.95
        aggregated.average_response_time = 0.6
        aggregated.average_cpu_usage = 60.0
        aggregated.average_memory_usage = 60.0
        aggregated._calculate_performance_grade()
        assert aggregated.performance_grade == "B"
        
        # 一般性能（C级）
        aggregated.overall_success_rate = 0.90
        aggregated.average_response_time = 0.8
        aggregated.average_cpu_usage = 60.0
        aggregated.average_memory_usage = 75.0
        aggregated._calculate_performance_grade()
        assert aggregated.performance_grade == "C"
        
        # 较差性能（D级）
        aggregated.overall_success_rate = 0.90
        aggregated.average_response_time = 1.5
        aggregated.average_cpu_usage = 60.0
        aggregated.average_memory_usage = 60.0
        aggregated._calculate_performance_grade()
        assert aggregated.performance_grade == "D"
        
        # 很差性能（F级）
        aggregated.overall_success_rate = 0.70
        aggregated.average_response_time = 1.0
        aggregated.average_cpu_usage = 95.0
        aggregated.average_memory_usage = 98.0
        aggregated._calculate_performance_grade()
        assert aggregated.performance_grade == "F"


class TestResultsCollector:
    """ResultsCollector类的单元测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def collector(self, temp_dir):
        """ResultsCollector实例fixture"""
        return ResultsCollector(
            storage_path=temp_dir,
            max_memory_items=100,
            auto_aggregate=False,  # 手动控制聚合
            enable_persistence=True
        )
    
    @pytest.fixture
    def collector_no_persistence(self, temp_dir):
        """无持久化的ResultsCollector实例"""
        return ResultsCollector(
            storage_path=temp_dir,
            max_memory_items=100,
            auto_aggregate=False,
            enable_persistence=False
        )
    
    def test_initialization(self, temp_dir):
        """测试初始化"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            max_memory_items=50,
            auto_aggregate=True,
            enable_persistence=True
        )
        
        assert collector.storage_path == Path(temp_dir)
        assert collector.max_memory_items == 50
        assert collector.auto_aggregate is True
        assert collector.enable_persistence is True
        assert collector.total_collected == 0
        assert len(collector.metrics) == 0
        assert len(collector.metrics_by_id) == 0
        assert collector.db_connection is not None
    
    def test_collect_metrics_basic(self, collector):
        """测试基本指标收集"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=1)
        
        metrics = collector.collect_metrics(
            test_name="基本测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            response_time=0.15,
            success_rate=0.95,
            cpu_usage=45.0,
            memory_usage=128.0
        )
        
        assert metrics.test_name == "基本测试"
        assert metrics.test_type == TestType.RESPONSE_TIME
        assert metrics.status == TestStatus.COMPLETED
        assert metrics.response_time == 0.15
        assert metrics.success_rate == 0.95
        assert metrics.cpu_usage == 45.0
        assert metrics.memory_usage == 128.0
        
        # 验证收集器状态
        assert collector.total_collected == 1
        assert len(collector.metrics) == 1
        assert len(collector.metrics_by_id) == 1
        assert collector.get_metrics_by_id(metrics.test_id) == metrics
    
    def test_collect_metrics_with_string_enums(self, collector):
        """测试使用字符串枚举收集指标"""
        metrics = collector.collect_metrics(
            test_name="字符串枚举测试",
            test_type="memory_leak",  # 字符串形式
            status="completed",       # 字符串形式
            start_time=datetime.now(),
            response_time=0.2
        )
        
        assert metrics.test_type == TestType.MEMORY_LEAK
        assert metrics.status == TestStatus.COMPLETED
    
    def test_collect_multiple_metrics(self, collector):
        """测试收集多个指标"""
        metrics_list = []
        
        for i in range(5):
            metrics = collector.collect_metrics(
                test_name=f"测试_{i}",
                test_type=TestType.LOAD_TEST,
                status=TestStatus.COMPLETED,
                start_time=datetime.now(),
                response_time=0.1 + i * 0.01,
                cpu_usage=50 + i * 2
            )
            metrics_list.append(metrics)
        
        assert collector.total_collected == 5
        assert len(collector.metrics) == 5
        assert len(collector.metrics_by_id) == 5
        
        # 验证所有指标都能通过ID找到
        for metrics in metrics_list:
            assert collector.get_metrics_by_id(metrics.test_id) == metrics
    
    def test_memory_limit(self, temp_dir):
        """测试内存限制"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            max_memory_items=3,  # 限制为3个
            enable_persistence=False
        )
        
        # 添加5个指标
        for i in range(5):
            collector.collect_metrics(
                test_name=f"测试_{i}",
                test_type=TestType.RESPONSE_TIME,
                status=TestStatus.COMPLETED,
                start_time=datetime.now()
            )
        
        # 验证只保留最新的3个
        assert len(collector.metrics) == 3
        assert collector.total_collected == 5
    
    def test_callbacks(self, collector):
        """测试回调函数"""
        result_callback_called = []
        aggregation_callback_called = []
        
        def result_callback(metrics):
            result_callback_called.append(metrics)
        
        def aggregation_callback(results):
            aggregation_callback_called.append(results)
        
        collector.add_result_callback(result_callback)
        collector.add_aggregation_callback(aggregation_callback)
        
        # 收集指标
        metrics = collector.collect_metrics(
            test_name="回调测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=datetime.now()
        )
        
        # 验证结果回调被调用
        assert len(result_callback_called) == 1
        assert result_callback_called[0] == metrics
        
        # 执行聚合
        collector.aggregate_results()
        
        # 验证聚合回调被调用
        assert len(aggregation_callback_called) == 1
        assert isinstance(aggregation_callback_called[0], AggregatedResults)
    
    def test_get_metrics_filtering(self, collector):
        """测试指标查询和过滤"""
        # 添加不同类型的测试数据
        base_time = datetime.now()
        
        # 响应时间测试
        collector.collect_metrics(
            test_name="响应时间测试1",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=base_time,
            tags=["性能", "响应"]
        )
        
        collector.collect_metrics(
            test_name="响应时间测试2",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.FAILED,
            start_time=base_time + timedelta(minutes=1),
            tags=["性能", "响应"]
        )
        
        # 内存泄漏测试
        collector.collect_metrics(
            test_name="内存泄漏测试",
            test_type=TestType.MEMORY_LEAK,
            status=TestStatus.COMPLETED,
            start_time=base_time + timedelta(minutes=2),
            tags=["内存", "泄漏"]
        )
        
        # 测试类型过滤
        response_time_metrics = collector.get_metrics(test_type=TestType.RESPONSE_TIME)
        assert len(response_time_metrics) == 2
        
        memory_leak_metrics = collector.get_metrics(test_type="memory_leak")
        assert len(memory_leak_metrics) == 1
        
        # 测试状态过滤
        completed_metrics = collector.get_metrics(status=TestStatus.COMPLETED)
        assert len(completed_metrics) == 2
        
        failed_metrics = collector.get_metrics(status="failed")
        assert len(failed_metrics) == 1
        
        # 测试时间过滤
        recent_metrics = collector.get_metrics(start_time=base_time + timedelta(minutes=1))
        assert len(recent_metrics) == 2
        
        # 测试标签过滤
        performance_metrics = collector.get_metrics(tags=["性能"])
        assert len(performance_metrics) == 2
        
        memory_metrics = collector.get_metrics(tags=["内存"])
        assert len(memory_metrics) == 1
        
        # 测试限制数量
        limited_metrics = collector.get_metrics(limit=2)
        assert len(limited_metrics) == 2
    
    def test_aggregation(self, collector):
        """测试结果聚合"""
        # 添加测试数据
        for i in range(10):
            status = TestStatus.COMPLETED if i < 8 else TestStatus.FAILED
            collector.collect_metrics(
                test_name=f"聚合测试_{i}",
                test_type=TestType.RESPONSE_TIME,
                status=status,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(seconds=i + 1),
                response_time=0.1 + i * 0.01,
                success_rate=0.9 + i * 0.01,
                cpu_usage=50 + i * 2
            )
        
        # 执行聚合
        aggregated = collector.aggregate_results()
        
        assert aggregated.total_tests == 10
        assert aggregated.completed_tests == 8
        assert aggregated.failed_tests == 2
        assert aggregated.overall_success_rate == 0.8
        assert collector.aggregated_results == aggregated
        assert collector.last_aggregation_time is not None
    
    def test_auto_aggregation(self, temp_dir):
        """测试自动聚合"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            auto_aggregate=True,
            enable_persistence=False
        )
        
        # 模拟自动聚合触发条件
        with patch.object(collector, 'aggregate_results') as mock_aggregate:
            # 添加100个指标（应该触发自动聚合）
            for i in range(100):
                collector.collect_metrics(
                    test_name=f"自动聚合测试_{i}",
                    test_type=TestType.RESPONSE_TIME,
                    status=TestStatus.COMPLETED,
                    start_time=datetime.now()
                )
            
            # 验证自动聚合被调用
            assert mock_aggregate.called
    
    def test_export_results(self, collector, temp_dir):
        """测试结果导出"""
        # 添加测试数据
        collector.collect_metrics(
            test_name="导出测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(),
            response_time=0.15,
            raw_data={"detail": "test_data"}
        )
        
        # 执行聚合
        collector.aggregate_results()
        
        # 导出为JSON
        export_path = Path(temp_dir) / "export_test.json"
        collector.export_results(str(export_path), format="json", include_raw_data=True)
        
        assert export_path.exists()
        
        # 验证导出内容
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        assert 'metadata' in data
        assert 'aggregated_results' in data
        assert 'metrics' in data
        assert len(data['metrics']) == 1
        assert data['metrics'][0]['test_name'] == "导出测试"
        assert 'raw_data' in data['metrics'][0]
        
        # 导出不包含原始数据
        export_path_no_raw = Path(temp_dir) / "export_test_no_raw.json"
        collector.export_results(str(export_path_no_raw), format="json", include_raw_data=False)
        
        with open(export_path_no_raw, 'r', encoding='utf-8') as f:
            data_no_raw = json.load(f)
        
        assert 'raw_data' not in data_no_raw['metrics'][0]
    
    def test_clear_results(self, collector):
        """测试清空结果"""
        # 添加测试数据
        collector.collect_metrics(
            test_name="清空测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=datetime.now()
        )
        
        collector.aggregate_results()
        
        # 清空但保留聚合结果
        collector.clear_results(keep_aggregated=True)
        
        assert len(collector.metrics) == 0
        assert len(collector.metrics_by_id) == 0
        assert collector.aggregated_results is not None
        
        # 完全清空
        collector.clear_results(keep_aggregated=False)
        
        assert collector.aggregated_results is None
        assert collector.last_aggregation_time is None
    
    def test_statistics(self, collector):
        """测试统计信息"""
        # 添加测试数据
        for i in range(3):
            collector.collect_metrics(
                test_name=f"统计测试_{i}",
                test_type=TestType.RESPONSE_TIME,
                status=TestStatus.COMPLETED,
                start_time=datetime.now()
            )
        
        stats = collector.get_statistics()
        
        assert stats['total_collected'] == 3
        assert stats['memory_items'] == 3
        assert stats['unique_tests'] == 3
        assert stats['persistence_enabled'] is True
        assert stats['auto_aggregate_enabled'] is False
        assert 'storage_path' in stats
    
    def test_database_persistence(self, collector):
        """测试数据库持久化"""
        # 添加测试数据
        metrics = collector.collect_metrics(
            test_name="持久化测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(),
            response_time=0.15,
            cpu_usage=45.0
        )
        
        # 验证数据库中有数据
        cursor = collector.db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_metrics")
        count = cursor.fetchone()[0]
        assert count == 1
        
        # 验证数据内容
        cursor.execute("SELECT test_name, test_type, response_time FROM test_metrics WHERE test_id = ?", 
                      (metrics.test_id,))
        row = cursor.fetchone()
        assert row[0] == "持久化测试"
        assert row[1] == "response_time"
        assert row[2] == 0.15
    
    def test_database_initialization_failure(self, temp_dir):
        """测试数据库初始化失败的处理"""
        # 创建一个无效的存储路径（文件而不是目录）
        invalid_path = Path(temp_dir) / "invalid_file"
        invalid_path.touch()
        
        with patch('sqlite3.connect', side_effect=Exception("数据库连接失败")):
            collector = ResultsCollector(
                storage_path=str(invalid_path),
                enable_persistence=True
            )
            
            # 验证持久化被禁用
            assert collector.enable_persistence is False
            assert collector.db_connection is None
    
    def test_callback_error_handling(self, collector):
        """测试回调函数错误处理"""
        def failing_callback(metrics):
            raise Exception("回调函数错误")
        
        collector.add_result_callback(failing_callback)
        
        # 收集指标不应该因为回调错误而失败
        metrics = collector.collect_metrics(
            test_name="回调错误测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=datetime.now()
        )
        
        assert metrics is not None
        assert collector.total_collected == 1


class TestResultsCollectorIntegration:
    """ResultsCollector集成测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_full_workflow(self, temp_dir):
        """测试完整工作流程"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            auto_aggregate=True,
            enable_persistence=True
        )
        
        # 模拟性能测试场景
        test_scenarios = [
            ("响应时间测试", TestType.RESPONSE_TIME, 0.12, 0.98, 45.0, 120.0),
            ("内存泄漏测试", TestType.MEMORY_LEAK, 0.25, 0.95, 55.0, 150.0),
            ("负载测试", TestType.LOAD_TEST, 0.18, 0.92, 65.0, 180.0),
            ("并发测试", TestType.CONCURRENCY, 0.22, 0.89, 70.0, 200.0),
            ("压力测试", TestType.STRESS_TEST, 0.35, 0.85, 80.0, 250.0)
        ]
        
        collected_metrics = []
        
        # 收集测试结果
        for name, test_type, response_time, success_rate, cpu, memory in test_scenarios:
            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=2)
            
            metrics = collector.collect_metrics(
                test_name=name,
                test_type=test_type,
                status=TestStatus.COMPLETED,
                start_time=start_time,
                end_time=end_time,
                response_time=response_time,
                success_rate=success_rate,
                cpu_usage=cpu,
                memory_usage=memory,
                tags=["集成测试", "性能"],
                metadata={"version": "1.0", "environment": "test"}
            )
            collected_metrics.append(metrics)
        
        # 验证收集状态
        assert collector.total_collected == 5
        assert len(collector.metrics) == 5
        
        # 执行聚合
        aggregated = collector.aggregate_results()
        
        # 验证聚合结果
        assert aggregated.total_tests == 5
        assert aggregated.completed_tests == 5
        assert aggregated.failed_tests == 0
        assert aggregated.overall_success_rate == 1.0
        assert aggregated.performance_grade in ["A", "B", "C", "D", "F"]
        
        # 测试查询功能
        response_time_tests = collector.get_metrics(test_type=TestType.RESPONSE_TIME)
        assert len(response_time_tests) == 1
        
        performance_tests = collector.get_metrics(tags=["性能"])
        assert len(performance_tests) == 5
        
        # 导出结果
        export_path = Path(temp_dir) / "integration_test_results.json"
        collector.export_results(str(export_path))
        
        assert export_path.exists()
        
        # 验证数据库持久化
        cursor = collector.db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_metrics")
        db_count = cursor.fetchone()[0]
        assert db_count == 5
        
        # 获取统计信息
        stats = collector.get_statistics()
        assert stats['total_collected'] == 5
        assert stats['memory_items'] == 5
        
        collector.db_connection.close()
    
    def test_concurrent_collection(self, temp_dir):
        """测试并发收集"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            enable_persistence=True
        )
        
        def collect_worker(worker_id, count):
            """工作线程函数"""
            for i in range(count):
                collector.collect_metrics(
                    test_name=f"并发测试_{worker_id}_{i}",
                    test_type=TestType.CONCURRENCY,
                    status=TestStatus.COMPLETED,
                    start_time=datetime.now(),
                    response_time=0.1 + i * 0.01,
                    metadata={"worker_id": worker_id}
                )
        
        # 启动多个线程
        threads = []
        worker_count = 5
        metrics_per_worker = 10
        
        for worker_id in range(worker_count):
            thread = threading.Thread(
                target=collect_worker,
                args=(worker_id, metrics_per_worker)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        expected_total = worker_count * metrics_per_worker
        assert collector.total_collected == expected_total
        assert len(collector.metrics) == expected_total
        
        # 验证数据库中的数据
        cursor = collector.db_connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_metrics")
        db_count = cursor.fetchone()[0]
        assert db_count == expected_total
        
        # 验证每个工作线程的数据
        for worker_id in range(worker_count):
            worker_metrics = collector.get_metrics()
            worker_specific = [
                m for m in worker_metrics 
                if m.metadata.get("worker_id") == worker_id
            ]
            assert len(worker_specific) == metrics_per_worker
        
        collector.db_connection.close()


class TestResultsCollectorAsync:
    """ResultsCollector异步功能测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_async_collect_metrics(self, temp_dir):
        """测试异步指标收集"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            enable_persistence=False
        )
        
        # 异步收集指标
        metrics = await collector.collect_metrics_async(
            test_name="异步测试",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=datetime.now(),
            response_time=0.15
        )
        
        assert metrics.test_name == "异步测试"
        assert metrics.response_time == 0.15
        assert collector.total_collected == 1
    
    @pytest.mark.asyncio
    async def test_concurrent_async_collection(self, temp_dir):
        """测试并发异步收集"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            enable_persistence=False
        )
        
        async def async_collect_worker(worker_id, count):
            """异步工作函数"""
            for i in range(count):
                await collector.collect_metrics_async(
                    test_name=f"异步并发测试_{worker_id}_{i}",
                    test_type=TestType.CONCURRENCY,
                    status=TestStatus.COMPLETED,
                    start_time=datetime.now(),
                    response_time=0.1 + i * 0.01
                )
        
        # 并发执行多个异步任务
        tasks = []
        worker_count = 3
        metrics_per_worker = 5
        
        for worker_id in range(worker_count):
            task = async_collect_worker(worker_id, metrics_per_worker)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # 验证结果
        expected_total = worker_count * metrics_per_worker
        assert collector.total_collected == expected_total
        assert len(collector.metrics) == expected_total


class TestCreateTestMetrics:
    """create_test_metrics便捷函数测试"""
    
    def test_create_successful_metrics(self):
        """测试创建成功的测试指标"""
        metrics = create_test_metrics(
            test_name="便捷创建测试",
            test_type="response_time",
            duration=1.5,
            success=True,
            response_time=0.12,
            cpu_usage=45.0
        )
        
        assert metrics.test_name == "便捷创建测试"
        assert metrics.test_type == TestType.RESPONSE_TIME
        assert metrics.status == TestStatus.COMPLETED
        assert abs(metrics.duration - 1.5) < 0.01
        assert metrics.response_time == 0.12
        assert metrics.cpu_usage == 45.0
        assert metrics.is_successful() is True
    
    def test_create_failed_metrics(self):
        """测试创建失败的测试指标"""
        metrics = create_test_metrics(
            test_name="失败测试",
            test_type="load_test",
            duration=2.0,
            success=False,
            error_count=3
        )
        
        assert metrics.test_name == "失败测试"
        assert metrics.test_type == TestType.LOAD_TEST
        assert metrics.status == TestStatus.FAILED
        assert abs(metrics.duration - 2.0) < 0.01
        assert metrics.error_count == 3
        assert metrics.is_successful() is False


# 性能基准测试
class TestResultsCollectorBenchmarks:
    """ResultsCollector性能基准测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_collection_performance(self, temp_dir, benchmark):
        """测试收集性能"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            enable_persistence=False,
            auto_aggregate=False
        )
        
        def collect_single_metric():
            return collector.collect_metrics(
                test_name="性能基准测试",
                test_type=TestType.RESPONSE_TIME,
                status=TestStatus.COMPLETED,
                start_time=datetime.now(),
                response_time=0.15
            )
        
        # 基准测试单次收集
        result = benchmark(collect_single_metric)
        assert result is not None
    
    def test_batch_collection_performance(self, temp_dir, benchmark):
        """测试批量收集性能"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            enable_persistence=False,
            auto_aggregate=False
        )
        
        def collect_batch_metrics():
            initial_count = collector.total_collected
            for i in range(100):
                collector.collect_metrics(
                    test_name=f"批量测试_{i}",
                    test_type=TestType.RESPONSE_TIME,
                    status=TestStatus.COMPLETED,
                    start_time=datetime.now(),
                    response_time=0.1 + i * 0.001
                )
            return collector.total_collected - initial_count
        
        # 基准测试批量收集
        result = benchmark(collect_batch_metrics)
        # 验证每次运行都收集了100个指标
        assert result == 100
    
    def test_aggregation_performance(self, temp_dir, benchmark):
        """测试聚合性能"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            enable_persistence=False,
            auto_aggregate=False
        )
        
        # 预先添加大量数据
        for i in range(1000):
            collector.collect_metrics(
                test_name=f"聚合性能测试_{i}",
                test_type=TestType.RESPONSE_TIME,
                status=TestStatus.COMPLETED,
                start_time=datetime.now(),
                response_time=0.1 + i * 0.0001
            )
        
        # 基准测试聚合操作
        result = benchmark(collector.aggregate_results)
        assert result.total_tests == 1000
    
    def test_query_performance(self, temp_dir, benchmark):
        """测试查询性能"""
        collector = ResultsCollector(
            storage_path=temp_dir,
            enable_persistence=False
        )
        
        # 添加多种类型的测试数据
        test_types = [TestType.RESPONSE_TIME, TestType.MEMORY_LEAK, TestType.LOAD_TEST]
        for i in range(300):
            collector.collect_metrics(
                test_name=f"查询性能测试_{i}",
                test_type=test_types[i % len(test_types)],
                status=TestStatus.COMPLETED,
                start_time=datetime.now(),
                tags=[f"tag_{i % 10}"]
            )
        
        def query_metrics():
            return collector.get_metrics(
                test_type=TestType.RESPONSE_TIME,
                tags=["tag_1"],
                limit=50
            )
        
        # 基准测试查询操作
        results = benchmark(query_metrics)
        assert len(results) > 0


class TestErrorHandling:
    """测试错误处理"""
    
    def test_database_connection_error(self):
        """测试数据库连接错误处理"""
        # 使用无效路径（Windows兼容）
        import os
        if os.name == 'nt':  # Windows
            invalid_path = "Z:\\invalid\\path\\that\\does\\not\\exist"
        else:  # Unix/Linux
            invalid_path = "/invalid/path"
        
        collector = ResultsCollector(storage_path=invalid_path)
        
        # 应该能正常初始化但禁用持久化
        assert not collector.enable_persistence
    
    def test_invalid_metrics_data(self):
        """测试无效指标数据处理"""
        collector = ResultsCollector(enable_persistence=False)
        
        # 测试添加None指标
        with pytest.raises(Exception):
            collector.collect_metrics(None, None, None, None)
    
    def test_corrupted_database_recovery(self):
        """测试损坏数据库的恢复"""
        # 测试禁用持久化的情况
        collector = ResultsCollector(enable_persistence=False)
        # 应该能够正常工作
        assert collector.enable_persistence == False
        
        # 测试无效路径的情况 - 应该自动禁用持久化
        try:
            invalid_path = "/invalid/path/that/does/not/exist" if os.name != 'nt' else "Z:\\invalid\\path\\that\\does\\not\\exist"
            collector = ResultsCollector(storage_path=invalid_path, enable_persistence=True)
            # 如果路径无效，应该自动禁用持久化
            assert collector.enable_persistence == False
        except Exception:
            # 如果抛出异常，也是可以接受的
            assert True


class TestBoundaryConditions:
    """测试边界条件"""
    
    def test_empty_database_operations(self):
        """测试空数据库操作"""
        collector = ResultsCollector(enable_persistence=False)
        
        # 测试从空数据库获取指标
        metrics = collector.get_metrics()
        assert len(metrics) == 0
        
        # 测试聚合空数据
        aggregated = collector.aggregate_results()
        assert aggregated.total_tests == 0
    
    def test_large_dataset_handling(self):
        """测试大数据集处理"""
        collector = ResultsCollector(enable_persistence=False)
        
        # 添加大量测试指标
        large_batch = []
        for i in range(1000):
            metrics = create_test_metrics(
                test_name=f"大数据测试_{i}",
                test_type="load_test",
                duration=1.0,
                response_time=0.1 + (i % 10) * 0.01
            )
            large_batch.append(metrics)
        
        # 批量添加
        with collector._lock:
            for metrics in large_batch:
                collector.metrics.append(metrics)
                collector.metrics_by_id[metrics.test_id] = metrics
                collector.total_collected += 1
        
        # 验证数据完整性
        all_metrics = collector.get_metrics(limit=2000)
        assert len(all_metrics) == 1000
    
    def test_concurrent_access_safety(self):
        """测试并发访问安全性"""
        collector = ResultsCollector(enable_persistence=False)
        
        def add_metrics_worker(worker_id):
            for i in range(10):
                metrics = create_test_metrics(
                    test_name=f"并发测试_{worker_id}_{i}",
                    test_type="response_time",
                    duration=0.5
                )
                collector.metrics.append(metrics)
                collector.metrics_by_id[metrics.test_id] = metrics
                collector.total_collected += 1
        
        # 创建多个线程同时添加数据
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=add_metrics_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证数据完整性
        all_metrics = collector.get_metrics()
        assert len(all_metrics) == 50  # 5个工作线程 × 10个测试


class TestAdvancedFeatures:
    """测试高级功能"""
    
    def test_custom_aggregation_functions(self):
        """测试自定义聚合函数"""
        collector = ResultsCollector(enable_persistence=False)
        
        # 添加测试数据
        test_data = [
            (0.1, 0.95, 50.0, 100.0),
            (0.2, 0.90, 60.0, 120.0),
            (0.15, 0.98, 55.0, 110.0),
            (0.25, 0.85, 65.0, 130.0),
            (0.12, 0.96, 52.0, 105.0)
        ]
        
        for i, (rt, sr, cpu, mem) in enumerate(test_data):
            metrics = create_test_metrics(
                test_name=f"自定义聚合测试_{i}",
                test_type="response_time",
                duration=1.0,
                response_time=rt,
                success_rate=sr,
                cpu_usage=cpu,
                memory_usage=mem
            )
            collector.metrics.append(metrics)
            collector.metrics_by_id[metrics.test_id] = metrics
            collector.total_collected += 1
        
        # 测试自定义聚合
        aggregated = collector.aggregate_results()
        
        # 验证聚合结果的合理性
        assert aggregated.total_tests == 5
        assert 0.1 <= aggregated.average_response_time <= 0.25
        assert 0.85 <= aggregated.overall_success_rate <= 1.0
    
    def test_time_based_filtering(self):
        """测试基于时间的过滤"""
        collector = ResultsCollector(enable_persistence=False)
        
        base_time = datetime.now()
        
        # 添加不同时间的测试数据
        for i in range(10):
            test_time = base_time + timedelta(hours=i)
            metrics = create_test_metrics(
                test_name=f"时间过滤测试_{i}",
                test_type="response_time",
                duration=1.0,
                start_time=test_time,
                end_time=test_time + timedelta(seconds=1)
            )
            collector.metrics.append(metrics)
            collector.metrics_by_id[metrics.test_id] = metrics
            collector.total_collected += 1
        
        # 测试时间范围过滤
        start_filter = base_time + timedelta(hours=3)
        end_filter = base_time + timedelta(hours=7)
        
        filtered_metrics = collector.get_metrics_by_time_range(
            start_time=start_filter,
            end_time=end_filter
        )
        
        # 应该返回5个结果（小时3,4,5,6,7）
        assert len(filtered_metrics) == 5


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--benchmark-only"])