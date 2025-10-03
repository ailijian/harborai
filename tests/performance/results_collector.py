"""
测试结果收集器模块

该模块提供全面的性能测试结果收集和聚合功能，支持：
- 多种测试类型的结果收集
- 实时数据聚合和统计分析
- 结果数据的持久化存储
- 灵活的查询和过滤接口
- 异步数据处理支持

作者：HarborAI性能测试团队
创建时间：2024年
遵循VIBE Coding规范
"""

import os
import json
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import statistics
import logging
from enum import Enum
import sqlite3
import pickle

# 配置日志
logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """测试状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class TestType(Enum):
    """测试类型枚举"""
    RESPONSE_TIME = "response_time"
    MEMORY_LEAK = "memory_leak"
    RESOURCE_UTILIZATION = "resource_utilization"
    CONCURRENCY = "concurrency"
    EXECUTION_EFFICIENCY = "execution_efficiency"
    STRESS_TEST = "stress_test"
    LOAD_TEST = "load_test"
    INTEGRATION = "integration"


@dataclass
class TestMetrics:
    """
    测试指标数据结构
    
    包含单个测试的完整指标信息
    """
    test_id: str
    test_name: str
    test_type: TestType
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    
    # 性能指标
    response_time: Optional[float] = None
    throughput: Optional[float] = None
    success_rate: Optional[float] = None
    error_count: int = 0
    
    # 资源使用指标
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    disk_io: Optional[Dict[str, float]] = None
    network_io: Optional[Dict[str, float]] = None
    
    # 详细数据
    raw_data: Dict[str, Any] = field(default_factory=dict)
    error_details: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
    
    def calculate_duration(self) -> float:
        """计算测试持续时间"""
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()
            return self.duration
        return 0.0
    
    def is_successful(self) -> bool:
        """判断测试是否成功"""
        return self.status == TestStatus.COMPLETED and self.error_count == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        # 处理枚举类型
        data['test_type'] = self.test_type.value
        data['status'] = self.status.value
        # 处理时间格式
        data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestMetrics':
        """从字典创建实例"""
        # 处理枚举类型
        data['test_type'] = TestType(data['test_type'])
        data['status'] = TestStatus(data['status'])
        # 处理时间格式
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        return cls(**data)


@dataclass
class AggregatedResults:
    """
    聚合测试结果
    
    包含所有测试的汇总统计信息
    """
    total_tests: int = 0
    completed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    timeout_tests: int = 0
    
    total_duration: float = 0.0
    average_duration: float = 0.0
    min_duration: float = 0.0
    max_duration: float = 0.0
    
    # 性能统计
    average_response_time: float = 0.0
    min_response_time: float = 0.0
    max_response_time: float = 0.0
    overall_success_rate: float = 0.0
    total_throughput: float = 0.0
    
    # 资源使用统计
    average_cpu_usage: float = 0.0
    peak_cpu_usage: float = 0.0
    average_memory_usage: float = 0.0
    peak_memory_usage: float = 0.0
    
    # 按测试类型分组的结果
    results_by_type: Dict[TestType, List[TestMetrics]] = field(default_factory=dict)
    
    # 性能等级评估
    performance_grade: str = "未评估"  # A, B, C, D, F
    
    # 时间戳
    aggregation_time: datetime = field(default_factory=datetime.now)
    
    def calculate_statistics(self, metrics: List[TestMetrics]) -> None:
        """
        计算聚合统计信息
        
        参数:
            metrics: 测试指标列表
        """
        if not metrics:
            return
        
        self.total_tests = len(metrics)
        
        # 按状态统计
        status_counts = defaultdict(int)
        for metric in metrics:
            status_counts[metric.status] += 1
        
        self.completed_tests = status_counts[TestStatus.COMPLETED]
        self.failed_tests = status_counts[TestStatus.FAILED]
        self.skipped_tests = status_counts[TestStatus.SKIPPED]
        self.timeout_tests = status_counts[TestStatus.TIMEOUT]
        
        # 计算成功率
        if self.total_tests > 0:
            self.overall_success_rate = self.completed_tests / self.total_tests
        
        # 持续时间统计
        durations = [m.duration for m in metrics if m.duration is not None]
        if durations:
            self.total_duration = sum(durations)
            self.average_duration = statistics.mean(durations)
            self.min_duration = min(durations)
            self.max_duration = max(durations)
        
        # 响应时间统计
        response_times = [m.response_time for m in metrics if m.response_time is not None]
        if response_times:
            self.average_response_time = statistics.mean(response_times)
            self.min_response_time = min(response_times)
            self.max_response_time = max(response_times)
        
        # 吞吐量统计
        throughputs = [m.throughput for m in metrics if m.throughput is not None]
        if throughputs:
            self.total_throughput = sum(throughputs)
        
        # 资源使用统计
        cpu_usages = [m.cpu_usage for m in metrics if m.cpu_usage is not None]
        if cpu_usages:
            self.average_cpu_usage = statistics.mean(cpu_usages)
            self.peak_cpu_usage = max(cpu_usages)
        
        memory_usages = [m.memory_usage for m in metrics if m.memory_usage is not None]
        if memory_usages:
            self.average_memory_usage = statistics.mean(memory_usages)
            self.peak_memory_usage = max(memory_usages)
        
        # 按类型分组
        self.results_by_type.clear()
        for metric in metrics:
            if metric.test_type not in self.results_by_type:
                self.results_by_type[metric.test_type] = []
            self.results_by_type[metric.test_type].append(metric)
        
        # 计算性能等级
        self._calculate_performance_grade()
    
    def _calculate_performance_grade(self) -> None:
        """计算性能等级"""
        score = 0
        
        # 成功率权重 40%
        if self.overall_success_rate >= 0.95:
            score += 40
        elif self.overall_success_rate >= 0.90:
            score += 35
        elif self.overall_success_rate >= 0.80:
            score += 25
        elif self.overall_success_rate >= 0.70:
            score += 15
        
        # 响应时间权重 30%
        if self.average_response_time <= 0.1:
            score += 30
        elif self.average_response_time <= 0.5:
            score += 25
        elif self.average_response_time <= 1.0:
            score += 20
        elif self.average_response_time <= 2.0:
            score += 10
        
        # 资源使用权重 30%
        if self.average_cpu_usage <= 50:
            score += 15
        elif self.average_cpu_usage <= 70:
            score += 10
        elif self.average_cpu_usage <= 85:
            score += 5
        
        if self.average_memory_usage <= 50:
            score += 15
        elif self.average_memory_usage <= 70:
            score += 10
        elif self.average_memory_usage <= 85:
            score += 5
        
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
        data = asdict(self)
        # 处理时间格式
        data['aggregation_time'] = self.aggregation_time.isoformat()
        # 处理嵌套的TestMetrics对象
        data['results_by_type'] = {
            test_type.value: [metric.to_dict() for metric in metrics]
            for test_type, metrics in self.results_by_type.items()
        }
        return data


class ResultsCollector:
    """
    测试结果收集器
    
    功能特性：
    - 实时收集测试结果
    - 多线程安全的数据存储
    - 灵活的查询和过滤
    - 数据持久化支持
    - 异步处理能力
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_memory_items: int = 10000,
        auto_aggregate: bool = True,
        enable_persistence: bool = True
    ):
        """
        初始化结果收集器
        
        参数:
            storage_path: 数据存储路径
            max_memory_items: 内存中最大存储项数
            auto_aggregate: 是否自动聚合结果
            enable_persistence: 是否启用持久化
        """
        self.storage_path = Path(storage_path) if storage_path else Path("test_results")
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
        except (OSError, FileExistsError) as e:
            logger.warning(f"无法创建存储目录 {self.storage_path}: {e}")
            self.enable_persistence = False
        
        self.max_memory_items = max_memory_items
        self.auto_aggregate = auto_aggregate
        self.enable_persistence = enable_persistence
        
        # 数据存储
        self.metrics: deque = deque(maxlen=max_memory_items)
        self.metrics_by_id: Dict[str, TestMetrics] = {}
        self.aggregated_results: Optional[AggregatedResults] = None
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 回调函数
        self.result_callbacks: List[Callable[[TestMetrics], None]] = []
        self.aggregation_callbacks: List[Callable[[AggregatedResults], None]] = []
        
        # 统计信息
        self.total_collected = 0
        self.last_aggregation_time: Optional[datetime] = None
        
        # 数据库连接（如果启用持久化）
        self.db_connection: Optional[sqlite3.Connection] = None
        if self.enable_persistence:
            self._initialize_database()
        
        logger.info(f"测试结果收集器初始化完成，存储路径: {self.storage_path}")
    
    def _initialize_database(self) -> None:
        """初始化数据库"""
        try:
            db_path = self.storage_path / "test_results.db"
            self.db_connection = sqlite3.connect(str(db_path), check_same_thread=False)
            
            # 创建表结构
            self.db_connection.execute("""
                CREATE TABLE IF NOT EXISTS test_metrics (
                    test_id TEXT PRIMARY KEY,
                    test_name TEXT NOT NULL,
                    test_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration REAL,
                    response_time REAL,
                    throughput REAL,
                    success_rate REAL,
                    error_count INTEGER,
                    cpu_usage REAL,
                    memory_usage REAL,
                    raw_data TEXT,
                    error_details TEXT,
                    tags TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.db_connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_test_type ON test_metrics(test_type)
            """)
            
            self.db_connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON test_metrics(status)
            """)
            
            self.db_connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_start_time ON test_metrics(start_time)
            """)
            
            self.db_connection.commit()
            logger.info("数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            self.enable_persistence = False
    
    def add_result_callback(self, callback: Callable[[TestMetrics], None]) -> None:
        """添加结果回调函数"""
        self.result_callbacks.append(callback)
    
    def add_aggregation_callback(self, callback: Callable[[AggregatedResults], None]) -> None:
        """添加聚合回调函数"""
        self.aggregation_callbacks.append(callback)
    
    def collect_metrics(
        self,
        test_name: str,
        test_type: Union[str, TestType],
        status: Union[str, TestStatus],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        **kwargs
    ) -> TestMetrics:
        """
        收集测试指标
        
        参数:
            test_name: 测试名称
            test_type: 测试类型
            status: 测试状态
            start_time: 开始时间
            end_time: 结束时间
            **kwargs: 其他指标数据
        
        返回:
            TestMetrics: 创建的测试指标对象
        """
        with self._lock:
            # 生成测试ID
            test_id = kwargs.get('test_id', f"{test_name}_{int(time.time() * 1000)}")
            
            # 处理枚举类型
            if isinstance(test_type, str):
                test_type = TestType(test_type)
            if isinstance(status, str):
                status = TestStatus(status)
            
            # 创建测试指标对象
            metrics = TestMetrics(
                test_id=test_id,
                test_name=test_name,
                test_type=test_type,
                status=status,
                start_time=start_time,
                end_time=end_time,
                response_time=kwargs.get('response_time'),
                throughput=kwargs.get('throughput'),
                success_rate=kwargs.get('success_rate'),
                error_count=kwargs.get('error_count', 0),
                cpu_usage=kwargs.get('cpu_usage'),
                memory_usage=kwargs.get('memory_usage'),
                disk_io=kwargs.get('disk_io'),
                network_io=kwargs.get('network_io'),
                raw_data=kwargs.get('raw_data', {}),
                error_details=kwargs.get('error_details', []),
                tags=kwargs.get('tags', []),
                metadata=kwargs.get('metadata', {})
            )
            
            # 计算持续时间
            metrics.calculate_duration()
            
            # 存储到内存
            self.metrics.append(metrics)
            self.metrics_by_id[test_id] = metrics
            self.total_collected += 1
            
            # 持久化存储
            if self.enable_persistence:
                self._persist_metrics(metrics)
            
            # 触发回调
            self._notify_result_callbacks(metrics)
            
            # 自动聚合
            if self.auto_aggregate:
                self._auto_aggregate()
            
            logger.debug(f"收集测试指标: {test_name} ({test_type.value})")
            return metrics
    
    def _persist_metrics(self, metrics: TestMetrics) -> None:
        """持久化测试指标"""
        if not self.db_connection:
            return
        
        try:
            self.db_connection.execute("""
                INSERT OR REPLACE INTO test_metrics (
                    test_id, test_name, test_type, status, start_time, end_time,
                    duration, response_time, throughput, success_rate, error_count,
                    cpu_usage, memory_usage, raw_data, error_details, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.test_id,
                metrics.test_name,
                metrics.test_type.value,
                metrics.status.value,
                metrics.start_time.isoformat(),
                metrics.end_time.isoformat() if metrics.end_time else None,
                metrics.duration,
                metrics.response_time,
                metrics.throughput,
                metrics.success_rate,
                metrics.error_count,
                metrics.cpu_usage,
                metrics.memory_usage,
                json.dumps(metrics.raw_data),
                json.dumps(metrics.error_details),
                json.dumps(metrics.tags),
                json.dumps(metrics.metadata)
            ))
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"持久化测试指标失败: {e}")
    
    def _notify_result_callbacks(self, metrics: TestMetrics) -> None:
        """通知结果回调"""
        for callback in self.result_callbacks:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"结果回调执行失败: {e}")
    
    def _notify_aggregation_callbacks(self, results: AggregatedResults) -> None:
        """通知聚合回调"""
        for callback in self.aggregation_callbacks:
            try:
                callback(results)
            except Exception as e:
                logger.error(f"聚合回调执行失败: {e}")
    
    def _auto_aggregate(self) -> None:
        """自动聚合结果"""
        # 每收集100个结果或每5分钟聚合一次
        should_aggregate = (
            self.total_collected % 100 == 0 or
            (self.last_aggregation_time is None) or
            (datetime.now() - self.last_aggregation_time).total_seconds() > 300
        )
        
        if should_aggregate:
            self.aggregate_results()
    
    def aggregate_results(self, metrics: Optional[List[TestMetrics]] = None) -> AggregatedResults:
        """
        聚合测试结果
        
        参数:
            metrics: 要聚合的指标列表（None表示使用所有指标）
        
        返回:
            AggregatedResults: 聚合后的测试结果
        """
        with self._lock:
            if metrics is None:
                metrics = list(self.metrics)
            
            self.aggregated_results = AggregatedResults()
            self.aggregated_results.calculate_statistics(metrics)
            self.last_aggregation_time = datetime.now()
            
            # 触发聚合回调
            self._notify_aggregation_callbacks(self.aggregated_results)
            
            logger.info(f"聚合测试结果完成，共 {len(metrics)} 个测试")
            return self.aggregated_results
    
    def get_metrics(
        self,
        test_type: Optional[Union[str, TestType]] = None,
        status: Optional[Union[str, TestStatus]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        tags: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[TestMetrics]:
        """
        查询测试指标
        
        参数:
            test_type: 测试类型过滤
            status: 状态过滤
            start_time: 开始时间过滤
            end_time: 结束时间过滤
            tags: 标签过滤
            limit: 结果数量限制
        
        返回:
            符合条件的测试指标列表
        """
        with self._lock:
            results = list(self.metrics)
            
            # 类型过滤
            if test_type:
                if isinstance(test_type, str):
                    test_type = TestType(test_type)
                results = [m for m in results if m.test_type == test_type]
            
            # 状态过滤
            if status:
                if isinstance(status, str):
                    status = TestStatus(status)
                results = [m for m in results if m.status == status]
            
            # 时间过滤
            if start_time:
                results = [m for m in results if m.start_time >= start_time]
            if end_time:
                results = [m for m in results if m.start_time <= end_time]
            
            # 标签过滤
            if tags:
                results = [
                    m for m in results
                    if any(tag in m.tags for tag in tags)
                ]
            
            # 限制结果数量
            if limit:
                results = results[-limit:]
            
            return results
    
    def get_metrics_by_id(self, test_id: str) -> Optional[TestMetrics]:
        """根据ID获取测试指标"""
        return self.metrics_by_id.get(test_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取收集器统计信息"""
        with self._lock:
            return {
                'total_collected': self.total_collected,
                'memory_items': len(self.metrics),
                'unique_tests': len(self.metrics_by_id),
                'last_aggregation': self.last_aggregation_time.isoformat() if self.last_aggregation_time else None,
                'storage_path': str(self.storage_path),
                'persistence_enabled': self.enable_persistence,
                'auto_aggregate_enabled': self.auto_aggregate
            }
    
    def get_metrics_by_time_range(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[TestMetrics]:
        """
        根据时间范围获取测试指标
        
        参数:
            start_time: 开始时间（包含）
            end_time: 结束时间（包含）
        
        返回:
            符合时间范围的测试指标列表
        """
        with self._lock:
            results = list(self.metrics)
            
            # 时间过滤
            if start_time:
                results = [m for m in results if m.start_time >= start_time]
            if end_time:
                results = [m for m in results if m.start_time <= end_time]
            
            return results
    
    def export_results(
        self,
        filepath: str,
        format: str = "json",
        include_raw_data: bool = False
    ) -> None:
        """
        导出测试结果
        
        参数:
            filepath: 导出文件路径
            format: 导出格式（json, csv, pickle）
            include_raw_data: 是否包含原始数据
        """
        with self._lock:
            data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'total_metrics': len(self.metrics),
                    'collector_stats': self.get_statistics()
                },
                'aggregated_results': self.aggregated_results.to_dict() if self.aggregated_results else None,
                'metrics': []
            }
            
            for metric in self.metrics:
                metric_data = metric.to_dict()
                if not include_raw_data:
                    metric_data.pop('raw_data', None)
                data['metrics'].append(metric_data)
            
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "json":
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            logger.info(f"测试结果已导出到: {filepath}")
    
    def clear_results(self, keep_aggregated: bool = True) -> None:
        """
        清空测试结果
        
        参数:
            keep_aggregated: 是否保留聚合结果
        """
        with self._lock:
            self.metrics.clear()
            self.metrics_by_id.clear()
            
            if not keep_aggregated:
                self.aggregated_results = None
                self.last_aggregation_time = None
            
            logger.info("测试结果已清空")
    
    async def collect_metrics_async(
        self,
        test_name: str,
        test_type: Union[str, TestType],
        status: Union[str, TestStatus],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        **kwargs
    ) -> TestMetrics:
        """
        异步收集测试指标
        
        参数同collect_metrics
        """
        # 在线程池中执行同步方法
        loop = asyncio.get_event_loop()
        
        # 创建一个包装函数来处理关键字参数
        def _collect_wrapper():
            return self.collect_metrics(
                test_name=test_name,
                test_type=test_type,
                status=status,
                start_time=start_time,
                end_time=end_time,
                **kwargs
            )
        
        return await loop.run_in_executor(None, _collect_wrapper)
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if self.db_connection:
            self.db_connection.close()


# 便捷函数
def create_test_metrics(
    test_name: str,
    test_type: str,
    duration: float,
    success: bool = True,
    **kwargs
) -> TestMetrics:
    """
    便捷的测试指标创建函数
    
    参数:
        test_name: 测试名称
        test_type: 测试类型
        duration: 测试持续时间
        success: 是否成功
        **kwargs: 其他指标数据（可包含start_time, end_time等）
    
    返回:
        TestMetrics: 创建的测试指标对象
    """
    # 允许通过kwargs传递自定义时间
    if 'start_time' in kwargs:
        start_time = kwargs.pop('start_time')
        end_time = kwargs.pop('end_time', start_time + timedelta(seconds=duration))
    else:
        start_time = datetime.now() - timedelta(seconds=duration)
        end_time = datetime.now()
    
    status = TestStatus.COMPLETED if success else TestStatus.FAILED
    
    return TestMetrics(
        test_id=kwargs.get('test_id', f"{test_name}_{int(time.time() * 1000)}"),
        test_name=test_name,
        test_type=TestType(test_type),
        status=status,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        **kwargs
    )


if __name__ == "__main__":
    # 示例使用
    collector = ResultsCollector()
    
    # 收集一些示例数据
    for i in range(10):
        metrics = collector.collect_metrics(
            test_name=f"测试_{i}",
            test_type=TestType.RESPONSE_TIME,
            status=TestStatus.COMPLETED,
            start_time=datetime.now() - timedelta(seconds=1),
            end_time=datetime.now(),
            response_time=0.1 + i * 0.01,
            success_rate=0.95 + i * 0.005,
            cpu_usage=50 + i * 2,
            memory_usage=60 + i * 3
        )
    
    # 聚合结果
    aggregated = collector.aggregate_results()
    print(f"聚合结果: 总测试数 {aggregated.total_tests}, 成功率 {aggregated.overall_success_rate:.2%}")
    
    # 导出结果
    collector.export_results("test_results.json")
    print("结果已导出到 test_results.json")