# -*- coding: utf-8 -*-
"""
HarborAI 性能测试核心框架

本模块实现了HarborAI项目的核心性能测试框架，包括：
- 性能测试控制器 (PerformanceTestController)
- 测试结果收集器 (ResultsCollector)
- 性能配置管理 (PerformanceConfig)
- 性能指标数据结构

作者: HarborAI Team
版本: 1.0.0
创建时间: 2024-01-20
遵循: VIBE Coding 规范
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Union, Callable, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path

# 设置日志
logger = logging.getLogger(__name__)


class TestType(Enum):
    """测试类型枚举"""
    EXECUTION_EFFICIENCY = "execution_efficiency"
    MEMORY_MONITORING = "memory_monitoring"
    RESPONSE_TIME = "response_time"
    CONCURRENCY = "concurrency"
    RESOURCE_UTILIZATION = "resource_utilization"


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PerformanceConfig:
    """
    性能测试配置类
    
    包含所有性能测试的配置参数和阈值设置
    """
    # 基础配置
    test_duration: float = 60.0  # 测试持续时间（秒）
    warmup_duration: float = 10.0  # 预热时间（秒）
    cooldown_duration: float = 5.0  # 冷却时间（秒）
    
    # 并发配置
    max_concurrent_users: int = 100
    concurrent_ramp_up_time: float = 30.0  # 并发爬坡时间
    
    # 资源监控配置
    resource_monitor_interval: float = 1.0  # 资源监控间隔（秒）
    memory_monitor_interval: float = 0.5  # 内存监控间隔（秒）
    
    # 性能阈值配置
    response_time_threshold: float = 2.0  # 响应时间阈值（秒）
    api_call_overhead_threshold: float = 0.001  # API调用开销阈值（秒）
    plugin_loading_threshold: float = 0.1  # 插件加载时间阈值（秒）
    memory_growth_threshold: float = 0.1  # 内存增长率阈值（10%）
    cpu_usage_threshold: float = 80.0  # CPU使用率阈值（%）
    success_rate_threshold: float = 0.999  # 成功率阈值（99.9%）
    
    # 报告配置
    enable_html_report: bool = True
    enable_json_report: bool = True
    report_output_dir: str = "reports/performance"
    
    def validate(self) -> List[str]:
        """
        验证配置参数的有效性
        
        Returns:
            List[str]: 验证错误信息列表，空列表表示验证通过
        """
        errors = []
        
        if self.test_duration <= 0:
            errors.append("测试持续时间必须大于0")
        
        if self.warmup_duration < 0:
            errors.append("预热时间不能为负数")
        
        if self.max_concurrent_users <= 0:
            errors.append("最大并发用户数必须大于0")
        
        if not (0 < self.success_rate_threshold <= 1):
            errors.append("成功率阈值必须在0到1之间")
        
        return errors


@dataclass
class TestMetrics:
    """
    测试指标数据结构
    
    存储单个测试的性能指标数据
    """
    test_name: str
    test_type: TestType
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    status: TestStatus = TestStatus.PENDING
    
    # 基础性能指标
    response_time: Optional[float] = None
    throughput: Optional[float] = None  # 吞吐量（请求/秒）
    success_rate: Optional[float] = None
    error_count: int = 0
    
    # 资源使用指标
    cpu_usage: Optional[Dict[str, float]] = None  # {"avg": 50.0, "max": 80.0, "min": 20.0}
    memory_usage: Optional[Dict[str, float]] = None  # {"avg": 100.0, "max": 150.0, "min": 80.0}
    
    # 详细数据
    raw_data: Dict[str, Any] = field(default_factory=dict)
    error_details: List[str] = field(default_factory=list)
    
    def calculate_duration(self) -> float:
        """计算测试持续时间"""
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
        return self.duration or 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "test_name": self.test_name,
            "test_type": self.test_type.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "status": self.status.value,
            "response_time": self.response_time,
            "throughput": self.throughput,
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "raw_data": self.raw_data,
            "error_details": self.error_details
        }


@dataclass
class AggregatedResults:
    """
    聚合测试结果
    
    包含所有测试的汇总统计信息
    """
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    
    total_duration: float = 0.0
    average_response_time: float = 0.0
    overall_success_rate: float = 0.0
    
    # 按测试类型分组的结果
    results_by_type: Dict[TestType, List[TestMetrics]] = field(default_factory=dict)
    
    # 性能等级评估
    performance_grade: str = "未评估"  # A, B, C, D, F
    
    def calculate_statistics(self, metrics_list: List[TestMetrics]):
        """计算聚合统计信息"""
        if not metrics_list:
            return
        
        self.total_tests = len(metrics_list)
        self.passed_tests = sum(1 for m in metrics_list if m.status == TestStatus.COMPLETED)
        self.failed_tests = sum(1 for m in metrics_list if m.status == TestStatus.FAILED)
        self.skipped_tests = sum(1 for m in metrics_list if m.status == TestStatus.SKIPPED)
        
        # 计算平均响应时间
        response_times = [m.response_time for m in metrics_list if m.response_time is not None]
        if response_times:
            self.average_response_time = sum(response_times) / len(response_times)
        
        # 计算总体成功率（基于通过的测试数量）
        if self.total_tests > 0:
            self.overall_success_rate = self.passed_tests / self.total_tests
        
        # 按类型分组
        for metric in metrics_list:
            if metric.test_type not in self.results_by_type:
                self.results_by_type[metric.test_type] = []
            self.results_by_type[metric.test_type].append(metric)
        
        # 评估性能等级
        self._evaluate_performance_grade()
    
    def _evaluate_performance_grade(self):
        """评估性能等级"""
        score = 0
        
        # 成功率评分 (40%)
        if self.overall_success_rate >= 0.999:
            score += 40
        elif self.overall_success_rate >= 0.99:
            score += 35
        elif self.overall_success_rate >= 0.95:
            score += 25
        elif self.overall_success_rate >= 0.90:
            score += 15
        
        # 响应时间评分 (30%)
        if self.average_response_time <= 1.0:
            score += 30
        elif self.average_response_time <= 2.0:
            score += 25
        elif self.average_response_time <= 5.0:
            score += 15
        elif self.average_response_time <= 10.0:
            score += 10
        
        # 测试通过率评分 (30%)
        pass_rate = self.passed_tests / self.total_tests if self.total_tests > 0 else 0
        if pass_rate >= 0.95:
            score += 30
        elif pass_rate >= 0.90:
            score += 25
        elif pass_rate >= 0.80:
            score += 20
        elif pass_rate >= 0.70:
            score += 15
        
        # 等级评定
        if score >= 90:
            self.performance_grade = "A"
        elif score >= 80:
            self.performance_grade = "B"
        elif score >= 70:
            self.performance_grade = "C"
        elif score >= 60:
            self.performance_grade = "D"
        else:
            self.performance_grade = "F"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "skipped_tests": self.skipped_tests,
            "total_duration": self.total_duration,
            "average_response_time": self.average_response_time,
            "overall_success_rate": self.overall_success_rate,
            "performance_grade": self.performance_grade,
            "results_by_type": {
                test_type.value: [metric.to_dict() for metric in metrics]
                for test_type, metrics in self.results_by_type.items()
            }
        }


class ResultsCollector:
    """
    测试结果收集和聚合器
    
    职责：
    - 收集各类性能指标数据
    - 数据标准化和聚合
    - 提供查询和分析接口
    """
    
    def __init__(self):
        self.metrics: List[TestMetrics] = []
        self.aggregated_results: Optional[AggregatedResults] = None
        self._lock = threading.Lock()
        
        logger.info("测试结果收集器初始化完成")
    
    def collect_metrics(self, test_type: str, metrics: Dict[str, Any]) -> TestMetrics:
        """
        收集性能指标
        
        Args:
            test_type: 测试类型
            metrics: 指标数据字典
            
        Returns:
            TestMetrics: 创建的测试指标对象
        """
        with self._lock:
            # 创建测试指标对象
            test_metrics = TestMetrics(
                test_name=metrics.get("test_name", f"未命名测试_{len(self.metrics)}"),
                test_type=TestType(test_type),
                start_time=metrics.get("start_time", datetime.now()),
                end_time=metrics.get("end_time"),
                response_time=metrics.get("response_time"),
                throughput=metrics.get("throughput"),
                success_rate=metrics.get("success_rate"),
                error_count=metrics.get("error_count", 0),
                cpu_usage=metrics.get("cpu_usage"),
                memory_usage=metrics.get("memory_usage"),
                raw_data=metrics.get("raw_data", {}),
                error_details=metrics.get("error_details", [])
            )
            
            # 计算持续时间
            test_metrics.calculate_duration()
            
            # 添加到收集列表
            self.metrics.append(test_metrics)
            
            logger.debug(f"收集到测试指标: {test_metrics.test_name}")
            return test_metrics
    
    def aggregate_results(self) -> AggregatedResults:
        """
        聚合测试结果
        
        Returns:
            AggregatedResults: 聚合后的测试结果
        """
        with self._lock:
            self.aggregated_results = AggregatedResults()
            self.aggregated_results.calculate_statistics(self.metrics)
            
            logger.info(f"聚合测试结果完成，共 {len(self.metrics)} 个测试")
            return self.aggregated_results
    
    def get_metrics_by_type(self, test_type: TestType) -> List[TestMetrics]:
        """根据测试类型获取指标"""
        return [m for m in self.metrics if m.test_type == test_type]
    
    def get_failed_tests(self) -> List[TestMetrics]:
        """获取失败的测试"""
        return [m for m in self.metrics if m.status == TestStatus.FAILED]
    
    def clear_results(self):
        """清空收集的结果"""
        with self._lock:
            self.metrics.clear()
            self.aggregated_results = None
            logger.info("测试结果已清空")
    
    def export_to_json(self, file_path: str):
        """导出结果到JSON文件"""
        try:
            data = {
                "metrics": [m.to_dict() for m in self.metrics],
                "aggregated_results": {
                    "total_tests": self.aggregated_results.total_tests if self.aggregated_results else 0,
                    "passed_tests": self.aggregated_results.passed_tests if self.aggregated_results else 0,
                    "failed_tests": self.aggregated_results.failed_tests if self.aggregated_results else 0,
                    "performance_grade": self.aggregated_results.performance_grade if self.aggregated_results else "未评估"
                } if self.aggregated_results else None,
                "export_time": datetime.now().isoformat()
            }
            
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"测试结果已导出到: {file_path}")
        except Exception as e:
            logger.error(f"导出测试结果失败: {e}")
            raise


class PerformanceTestController:
    """
    性能测试主控制器
    
    职责：
    - 协调各类性能测试的执行
    - 管理测试环境和配置
    - 收集和整合测试结果
    """
    
    def __init__(self, config: PerformanceConfig):
        """
        初始化性能测试控制器
        
        Args:
            config: 性能测试配置
        """
        # 验证配置
        config_errors = config.validate()
        if config_errors:
            raise ValueError(f"配置验证失败: {', '.join(config_errors)}")
        
        self.config = config
        self.test_runners: Dict[TestType, Callable] = {}
        self.results_collector = ResultsCollector()
        self._is_running = False
        self._current_test: Optional[Dict[str, Any]] = None
        
        logger.info("性能测试控制器初始化完成")
    
    @property
    def current_test(self) -> Optional[Dict[str, Any]]:
        """获取当前正在运行的测试"""
        return self._current_test
    
    @current_test.setter
    def current_test(self, value: Optional[Dict[str, Any]]):
        """设置当前正在运行的测试"""
        self._current_test = value
    
    @property
    def is_running(self) -> bool:
        """获取是否正在运行测试"""
        return self._is_running
    
    @is_running.setter
    def is_running(self, value: bool):
        """设置是否正在运行测试"""
        self._is_running = value
    
    def register_test_runner(self, test_type: TestType, runner: Callable):
        """
        注册测试运行器
        
        Args:
            test_type: 测试类型
            runner: 测试运行器函数
        """
        self.test_runners[test_type] = runner
        logger.info(f"注册测试运行器: {test_type.value}")
    
    async def run_single_test(self, test_type: TestType, test_name: str, **kwargs) -> TestMetrics:
        """
        运行单个性能测试
        
        Args:
            test_type: 测试类型
            test_name: 测试名称
            **kwargs: 测试参数
            
        Returns:
            TestMetrics: 测试结果指标
        """
        start_time = datetime.now()
        
        if test_type not in self.test_runners:
            # 创建失败的测试结果而不是抛出异常
            end_time = datetime.now()
            logger.error(f"未注册的测试类型: {test_type.value}")
            
            metrics_data = {
                "test_name": test_name,
                "start_time": start_time,
                "end_time": end_time,
                "error_details": [f"未注册的测试类型: {test_type.value}"]
            }
            
            metrics = self.results_collector.collect_metrics(test_type.value, metrics_data)
            metrics.status = TestStatus.FAILED
            
            return metrics
        
        self._current_test = {
            "test_name": test_name,
            "test_type": test_type.value,
            "start_time": start_time.isoformat()
        }
        self._is_running = True
        
        logger.info(f"开始执行测试: {test_name} ({test_type.value})")
        
        try:
            # 预热阶段
            if self.config.warmup_duration > 0:
                logger.info(f"预热阶段开始，持续 {self.config.warmup_duration} 秒")
                await asyncio.sleep(self.config.warmup_duration)
            
            # 执行测试（带超时）
            runner = self.test_runners[test_type]
            try:
                if asyncio.iscoroutinefunction(runner):
                    test_result = await asyncio.wait_for(
                        runner(test_name=test_name, config=self.config, **kwargs),
                        timeout=self.config.test_duration
                    )
                else:
                    test_result = runner(test_name=test_name, config=self.config, **kwargs)
            except asyncio.TimeoutError:
                raise Exception(f"测试超时: {test_name} (超过 {self.config.test_duration} 秒)")
            
            # 冷却阶段
            if self.config.cooldown_duration > 0:
                logger.info(f"冷却阶段开始，持续 {self.config.cooldown_duration} 秒")
                await asyncio.sleep(self.config.cooldown_duration)
            
            end_time = datetime.now()
            
            # 收集测试结果
            metrics_data = {
                "test_name": test_name,
                "start_time": start_time,
                "end_time": end_time,
                **test_result
            }
            
            metrics = self.results_collector.collect_metrics(test_type.value, metrics_data)
            metrics.status = TestStatus.COMPLETED
            
            logger.info(f"测试完成: {test_name}")
            return metrics
            
        except Exception as e:
            end_time = datetime.now()
            logger.error(f"测试失败: {test_name}, 错误: {e}")
            
            # 收集失败的测试结果
            metrics_data = {
                "test_name": test_name,
                "start_time": start_time,
                "end_time": end_time,
                "error_details": [str(e)]
            }
            
            metrics = self.results_collector.collect_metrics(test_type.value, metrics_data)
            metrics.status = TestStatus.FAILED
            
            return metrics
        
        finally:
            self._current_test = None
            self._is_running = False
    
    async def run_full_performance_suite(self, test_suite: Optional[List[Dict[str, Any]]] = None) -> AggregatedResults:
        """
        执行完整性能测试套件
        
        Args:
            test_suite: 测试套件配置列表，如果为None则运行所有注册的测试
            
        Returns:
            AggregatedResults: 聚合测试结果
        """
        if self._is_running:
            raise RuntimeError("性能测试套件正在运行中")
        
        self._is_running = True
        logger.info("开始执行完整性能测试套件")
        
        try:
            # 清空之前的结果
            self.results_collector.clear_results()
            
            # 如果没有提供测试套件，则运行所有注册的测试
            if test_suite is None:
                test_suite = [
                    {"test_type": test_type, "test_name": f"{test_type.value}_default_test"}
                    for test_type in self.test_runners.keys()
                ]
            
            # 执行所有测试
            for test_config in test_suite:
                test_type = TestType(test_config["test_type"])
                test_name = test_config["test_name"]
                test_params = test_config.get("params", {})
                
                await self.run_single_test(test_type, test_name, **test_params)
            
            # 聚合结果
            aggregated_results = self.results_collector.aggregate_results()
            
            logger.info(f"性能测试套件执行完成，性能等级: {aggregated_results.performance_grade}")
            return aggregated_results
            
        finally:
            self._is_running = False
    
    def get_test_status(self) -> Dict[str, Any]:
        """
        获取当前测试状态
        
        Returns:
            Dict[str, Any]: 测试状态信息
        """
        return {
            "is_running": self._is_running,
            "current_test": self._current_test,
            "registered_test_types": [t.value for t in self.test_runners.keys()],
            "total_tests_run": len(self.results_collector.metrics),
            "total_collected_metrics": len(self.results_collector.metrics)
        }
    
    def stop_current_test(self):
        """停止当前正在运行的测试"""
        if self._is_running or self._current_test is not None:
            logger.warning("请求停止当前测试")
            self._is_running = False
            self._current_test = None
        else:
            logger.info("当前没有正在运行的测试")


# 导出主要类和函数
__all__ = [
    'PerformanceTestController',
    'ResultsCollector', 
    'PerformanceConfig',
    'TestMetrics',
    'AggregatedResults',
    'TestType',
    'TestStatus'
]