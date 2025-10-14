# -*- coding: utf-8 -*-
"""
统一资源监控测试模块

本模块整合了所有资源监控相关的测试，包括：
- 基础资源监控测试（CPU、内存、磁盘、网络）
- 资源利用率监控器测试
- 资源泄漏检测测试
- 资源限制和阈值测试
- 性能基准测试

遵循VIBE编码规范的测试金字塔原则：
- 第一层：单元测试 - 验证各个指标收集方法
- 第二层：集成测试 - 验证完整的资源监控流程
- 第三层：基准测试 - 验证监控器本身的性能开销

作者: HarborAI Team
创建时间: 2025-01-27
遵循: VIBE Coding 规范
"""

import asyncio
import gc
import os
import psutil
import threading
import time
import platform
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from unittest.mock import Mock, patch, MagicMock
import pytest
import statistics
from datetime import datetime, timedelta
import concurrent.futures

from tests.performance import PERFORMANCE_CONFIG, SUPPORTED_VENDORS

try:
    from resource_utilization_monitor import (
        ResourceUtilizationMonitor,
        CPUMetrics,
        MemoryMetrics,
        DiskMetrics,
        NetworkMetrics,
        GPUMetrics,
        ProcessMetrics,
        SystemResourceSnapshot,
        ResourceThresholds,
        monitor_resource_usage
    )
except ImportError:
    # 如果导入失败，创建Mock类
    ResourceUtilizationMonitor = Mock
    CPUMetrics = Mock
    MemoryMetrics = Mock
    DiskMetrics = Mock
    NetworkMetrics = Mock
    GPUMetrics = Mock
    ProcessMetrics = Mock
    SystemResourceSnapshot = Mock
    ResourceThresholds = Mock
    monitor_resource_usage = Mock


class ResourceSnapshot(NamedTuple):
    """
    资源快照数据结构
    
    记录某个时间点的系统资源使用情况
    """
    timestamp: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    disk_read_mb: float
    disk_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    threads_count: int


@dataclass
class ResourceMetrics:
    """
    资源监控指标数据类
    
    汇总和分析资源使用情况的统计数据
    """
    # 基础数据
    snapshots: List[ResourceSnapshot] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # CPU指标
    avg_cpu_usage: float = 0.0
    max_cpu_usage: float = 0.0
    cpu_spikes_count: int = 0
    
    # 内存指标
    avg_memory_usage: float = 0.0
    max_memory_usage: float = 0.0
    memory_growth: float = 0.0
    memory_leaks_detected: bool = False
    
    # 磁盘I/O指标
    total_disk_read: float = 0.0
    total_disk_write: float = 0.0
    avg_disk_io_rate: float = 0.0
    
    # 网络I/O指标
    total_network_sent: float = 0.0
    total_network_recv: float = 0.0
    avg_network_io_rate: float = 0.0
    
    # 系统资源指标
    max_open_files: int = 0
    max_threads: int = 0
    resource_warnings: List[str] = field(default_factory=list)
    
    def calculate_metrics(self):
        """计算汇总指标"""
        if not self.snapshots:
            return
        
        # CPU指标计算
        cpu_values = [s.cpu_percent for s in self.snapshots]
        self.avg_cpu_usage = statistics.mean(cpu_values)
        self.max_cpu_usage = max(cpu_values)
        self.cpu_spikes_count = sum(1 for cpu in cpu_values if cpu > 80.0)
        
        # 内存指标计算
        memory_values = [s.memory_mb for s in self.snapshots]
        self.avg_memory_usage = statistics.mean(memory_values)
        self.max_memory_usage = max(memory_values)
        
        if len(memory_values) > 1:
            self.memory_growth = memory_values[-1] - memory_values[0]
            # 简单的内存泄漏检测：内存持续增长且增长量超过100MB
            if self.memory_growth > 100 and all(
                memory_values[i] <= memory_values[i+1] 
                for i in range(len(memory_values)-1)
            ):
                self.memory_leaks_detected = True
        
        # 磁盘I/O指标计算
        if len(self.snapshots) > 1:
            first_snapshot = self.snapshots[0]
            last_snapshot = self.snapshots[-1]
            
            self.total_disk_read = last_snapshot.disk_read_mb - first_snapshot.disk_read_mb
            self.total_disk_write = last_snapshot.disk_write_mb - first_snapshot.disk_write_mb
            
            duration = last_snapshot.timestamp - first_snapshot.timestamp
            if duration > 0:
                self.avg_disk_io_rate = (self.total_disk_read + self.total_disk_write) / duration
        
        # 网络I/O指标计算
        if len(self.snapshots) > 1:
            first_snapshot = self.snapshots[0]
            last_snapshot = self.snapshots[-1]
            
            self.total_network_sent = last_snapshot.network_sent_mb - first_snapshot.network_sent_mb
            self.total_network_recv = last_snapshot.network_recv_mb - first_snapshot.network_recv_mb
            
            duration = last_snapshot.timestamp - first_snapshot.timestamp
            if duration > 0:
                self.avg_network_io_rate = (self.total_network_sent + self.total_network_recv) / duration
        
        # 系统资源指标
        self.max_open_files = max(s.open_files for s in self.snapshots)
        self.max_threads = max(s.threads_count for s in self.snapshots)
        
        # 生成警告
        self._generate_warnings()
    
    def _generate_warnings(self):
        """生成资源使用警告"""
        self.resource_warnings.clear()
        
        if self.max_cpu_usage > 90:
            self.resource_warnings.append(f"CPU使用率过高: {self.max_cpu_usage:.1f}%")
        
        if self.max_memory_usage > 1000:  # 1GB
            self.resource_warnings.append(f"内存使用量过高: {self.max_memory_usage:.1f}MB")
        
        if self.memory_leaks_detected:
            self.resource_warnings.append(f"检测到可能的内存泄漏: 增长{self.memory_growth:.1f}MB")
        
        if self.max_open_files > 1000:
            self.resource_warnings.append(f"打开文件数过多: {self.max_open_files}")


class ResourceMonitor:
    """
    资源监控器
    
    实时监控系统资源使用情况
    """
    
    def __init__(self, interval: float = 0.5):
        """
        初始化资源监控器
        
        Args:
            interval: 监控间隔（秒）
        """
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.snapshots = []
        self.start_time = None
        self.process = psutil.Process()
        
        # 初始化基准值
        self._initial_disk_io = None
        self._initial_network_io = None
        
        try:
            self._initial_disk_io = psutil.disk_io_counters()
            self._initial_network_io = psutil.net_io_counters()
        except (AttributeError, OSError):
            # 某些系统可能不支持这些指标
            pass
    
    def _get_resource_snapshot(self) -> ResourceSnapshot:
        """获取当前资源快照"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # 内存使用情况
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            virtual_memory = psutil.virtual_memory()
            memory_percent = virtual_memory.percent
            
            # 磁盘I/O
            disk_read_mb = 0.0
            disk_write_mb = 0.0
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io and self._initial_disk_io:
                    disk_read_mb = (disk_io.read_bytes - self._initial_disk_io.read_bytes) / 1024 / 1024
                    disk_write_mb = (disk_io.write_bytes - self._initial_disk_io.write_bytes) / 1024 / 1024
            except (AttributeError, OSError):
                pass
            
            # 网络I/O
            network_sent_mb = 0.0
            network_recv_mb = 0.0
            try:
                network_io = psutil.net_io_counters()
                if network_io and self._initial_network_io:
                    network_sent_mb = (network_io.bytes_sent - self._initial_network_io.bytes_sent) / 1024 / 1024
                    network_recv_mb = (network_io.bytes_recv - self._initial_network_io.bytes_recv) / 1024 / 1024
            except (AttributeError, OSError):
                pass
            
            # 进程信息
            try:
                open_files = len(self.process.open_files())
            except (psutil.AccessDenied, OSError):
                open_files = 0
            
            try:
                threads_count = self.process.num_threads()
            except (psutil.AccessDenied, OSError):
                threads_count = 0
            
            return ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                disk_read_mb=disk_read_mb,
                disk_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                open_files=open_files,
                threads_count=threads_count
            )
        
        except Exception as e:
            # 如果获取快照失败，返回默认值
            return ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                disk_read_mb=0.0,
                disk_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                open_files=0,
                threads_count=0
            )
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            snapshot = self._get_resource_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(self.interval)
    
    def start_monitoring(self):
        """开始监控"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.snapshots.clear()
        self.start_time = datetime.now()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> ResourceMetrics:
        """
        停止监控并返回指标
        
        Returns:
            ResourceMetrics: 监控期间的资源使用指标
        """
        if not self.monitoring:
            return ResourceMetrics()
        
        self.monitoring = False
        
        # 等待监控线程结束
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)
        
        # 计算指标
        metrics = ResourceMetrics(
            snapshots=self.snapshots.copy(),
            start_time=self.start_time,
            end_time=datetime.now()
        )
        metrics.calculate_metrics()
        
        return metrics


class ResourceTestRunner:
    """
    资源测试运行器
    
    提供各种资源密集型任务用于测试
    """
    
    def __init__(self):
        self.temp_files = []
    
    def cpu_intensive_task(self, duration: float):
        """CPU密集型任务"""
        start_time = time.time()
        while time.time() - start_time < duration:
            # 执行一些CPU密集型计算
            sum(i * i for i in range(1000))
    
    def memory_intensive_task(self, duration: float, allocation_mb: int = 10):
        """内存密集型任务"""
        start_time = time.time()
        allocated_data = []
        
        try:
            while time.time() - start_time < duration:
                # 分配内存
                data = bytearray(allocation_mb * 1024 * 1024)  # allocation_mb MB
                allocated_data.append(data)
                time.sleep(0.1)
        finally:
            # 清理内存
            del allocated_data
            gc.collect()
    
    def io_intensive_task(self, duration: float):
        """I/O密集型任务"""
        import tempfile
        
        start_time = time.time()
        temp_files = []
        
        try:
            while time.time() - start_time < duration:
                # 创建临时文件并写入数据
                with tempfile.NamedTemporaryFile(delete=False) as f:
                    temp_files.append(f.name)
                    f.write(b"x" * 1024 * 1024)  # 1MB
                
                time.sleep(0.1)
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass
    
    def mixed_workload_task(self, duration: float):
        """混合工作负载任务"""
        start_time = time.time()
        allocated_data = []
        
        try:
            while time.time() - start_time < duration:
                # CPU计算
                sum(i * i for i in range(500))
                
                # 内存分配
                data = bytearray(1024 * 1024)  # 1MB
                allocated_data.append(data)
                
                # 短暂休眠
                time.sleep(0.05)
        finally:
            del allocated_data
            gc.collect()


# ==================== 第一层：单元测试 ====================

class TestCPUMetrics:
    """CPU指标单元测试"""
    
    def test_cpu_metrics_creation(self):
        """测试CPU指标创建"""
        if CPUMetrics == Mock:
            pytest.skip("ResourceUtilizationMonitor模块未导入")
        
        cpu_metrics = CPUMetrics(
            usage_percent=45.5,
            core_count=8,
            frequency_mhz=2400.0,
            load_average=[1.2, 1.5, 1.8],
            context_switches=12345,
            interrupts=6789
        )
        
        assert cpu_metrics.usage_percent == 45.5
        assert cpu_metrics.core_count == 8
        assert cpu_metrics.frequency_mhz == 2400.0
        assert cpu_metrics.load_average == [1.2, 1.5, 1.8]
        assert cpu_metrics.context_switches == 12345
        assert cpu_metrics.interrupts == 6789
    
    def test_cpu_metrics_to_dict(self):
        """测试CPU指标转换为字典"""
        if CPUMetrics == Mock:
            pytest.skip("ResourceUtilizationMonitor模块未导入")
        
        cpu_metrics = CPUMetrics(
            usage_percent=50.0,
            core_count=4,
            frequency_mhz=3000.0,
            load_average=[0.8, 1.0, 1.2],
            context_switches=10000,
            interrupts=5000
        )
        
        result = cpu_metrics.to_dict()
        
        assert result["usage_percent"] == 50.0
        assert result["core_count"] == 4
        assert result["frequency_mhz"] == 3000.0
        assert result["load_average"] == [0.8, 1.0, 1.2]
        assert result["context_switches"] == 10000
        assert result["interrupts"] == 5000


class TestMemoryMetrics:
    """内存指标单元测试"""
    
    def test_memory_metrics_creation(self):
        """测试内存指标创建"""
        if MemoryMetrics == Mock:
            pytest.skip("ResourceUtilizationMonitor模块未导入")
        
        memory_metrics = MemoryMetrics(
            total_mb=8192.0,
            available_mb=4096.0,
            used_mb=4096.0,
            usage_percent=50.0,
            swap_total_mb=2048.0,
            swap_used_mb=512.0,
            swap_percent=25.0
        )
        
        assert memory_metrics.total_mb == 8192.0
        assert memory_metrics.available_mb == 4096.0
        assert memory_metrics.used_mb == 4096.0
        assert memory_metrics.usage_percent == 50.0
        assert memory_metrics.swap_total_mb == 2048.0
        assert memory_metrics.swap_used_mb == 512.0
        assert memory_metrics.swap_percent == 25.0
    
    def test_memory_metrics_to_dict(self):
        """测试内存指标转换为字典"""
        if MemoryMetrics == Mock:
            pytest.skip("ResourceUtilizationMonitor模块未导入")
        
        memory_metrics = MemoryMetrics(
            total_mb=16384.0,
            available_mb=8192.0,
            used_mb=8192.0,
            usage_percent=50.0,
            swap_total_mb=4096.0,
            swap_used_mb=1024.0,
            swap_percent=25.0
        )
        
        result = memory_metrics.to_dict()
        
        assert result["total_mb"] == 16384.0
        assert result["available_mb"] == 8192.0
        assert result["used_mb"] == 8192.0
        assert result["usage_percent"] == 50.0
        assert result["swap_total_mb"] == 4096.0
        assert result["swap_used_mb"] == 1024.0
        assert result["swap_percent"] == 25.0


class TestResourceThresholds:
    """资源阈值单元测试"""
    
    def test_default_thresholds(self):
        """测试默认阈值"""
        if ResourceThresholds == Mock:
            pytest.skip("ResourceUtilizationMonitor模块未导入")
        
        thresholds = ResourceThresholds()
        
        assert thresholds.cpu_warning_percent == 80.0
        assert thresholds.cpu_critical_percent == 95.0
        assert thresholds.memory_warning_percent == 80.0
        assert thresholds.memory_critical_percent == 95.0
    
    def test_custom_thresholds(self):
        """测试自定义阈值"""
        if ResourceThresholds == Mock:
            pytest.skip("ResourceUtilizationMonitor模块未导入")
        
        thresholds = ResourceThresholds(
            cpu_warning_percent=70.0,
            cpu_critical_percent=90.0,
            memory_warning_percent=75.0,
            memory_critical_percent=90.0
        )
        
        assert thresholds.cpu_warning_percent == 70.0
        assert thresholds.cpu_critical_percent == 90.0
        assert thresholds.memory_warning_percent == 75.0
        assert thresholds.memory_critical_percent == 90.0


# ==================== 第二层：集成测试 ====================

class TestResourceMonitoringIntegration:
    """资源监控集成测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.monitor = ResourceMonitor(interval=0.1)
        self.test_runner = ResourceTestRunner()
    
    def teardown_method(self):
        """测试后清理"""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
    
    def _print_resource_summary(self, metrics: ResourceMetrics):
        """打印资源使用摘要"""
        print(f"\n=== 资源使用摘要 ===")
        print(f"监控时长: {(metrics.end_time - metrics.start_time).total_seconds():.1f}秒")
        print(f"快照数量: {len(metrics.snapshots)}")
        print(f"平均CPU使用率: {metrics.avg_cpu_usage:.1f}%")
        print(f"峰值CPU使用率: {metrics.max_cpu_usage:.1f}%")
        print(f"平均内存使用: {metrics.avg_memory_usage:.1f}MB")
        print(f"峰值内存使用: {metrics.max_memory_usage:.1f}MB")
        print(f"内存增长: {metrics.memory_growth:.1f}MB")
        print(f"磁盘读取: {metrics.total_disk_read:.1f}MB")
        print(f"磁盘写入: {metrics.total_disk_write:.1f}MB")
        print(f"网络发送: {metrics.total_network_sent:.1f}MB")
        print(f"网络接收: {metrics.total_network_recv:.1f}MB")
        
        if metrics.resource_warnings:
            print(f"资源警告:")
            for warning in metrics.resource_warnings:
                print(f"  - {warning}")
    
    @pytest.mark.performance
    @pytest.mark.resource
    def test_cpu_usage_monitoring(self):
        """测试CPU使用率监控"""
        self.monitor.start_monitoring()
        
        # 执行CPU密集型任务
        self.test_runner.cpu_intensive_task(duration=1.0)
        
        metrics = self.monitor.stop_monitoring()
        self._print_resource_summary(metrics)
        
        # 验证监控结果
        assert len(metrics.snapshots) > 0
        assert metrics.avg_cpu_usage >= 0
        assert metrics.max_cpu_usage >= metrics.avg_cpu_usage
        assert metrics.start_time is not None
        assert metrics.end_time is not None
        assert metrics.end_time > metrics.start_time
    
    @pytest.mark.performance
    @pytest.mark.resource
    def test_memory_usage_monitoring(self):
        """测试内存使用监控"""
        self.monitor.start_monitoring()
        
        # 执行内存密集型任务
        self.test_runner.memory_intensive_task(duration=1.0, allocation_mb=5)
        
        metrics = self.monitor.stop_monitoring()
        self._print_resource_summary(metrics)
        
        # 验证监控结果
        assert len(metrics.snapshots) > 0
        assert metrics.avg_memory_usage > 0
        assert metrics.max_memory_usage >= metrics.avg_memory_usage
    
    @pytest.mark.performance
    @pytest.mark.resource
    def test_io_performance_monitoring(self):
        """测试I/O性能监控"""
        self.monitor.start_monitoring()
        
        # 执行I/O密集型任务
        self.test_runner.io_intensive_task(duration=1.0)
        
        metrics = self.monitor.stop_monitoring()
        self._print_resource_summary(metrics)
        
        # 验证监控结果
        assert len(metrics.snapshots) > 0
        # I/O指标可能为0（取决于系统支持）
        assert metrics.total_disk_read >= 0
        assert metrics.total_disk_write >= 0
    
    @pytest.mark.performance
    @pytest.mark.resource
    def test_mixed_workload_monitoring(self):
        """测试混合工作负载监控"""
        self.monitor.start_monitoring()
        
        # 执行混合工作负载
        self.test_runner.mixed_workload_task(duration=1.5)
        
        metrics = self.monitor.stop_monitoring()
        self._print_resource_summary(metrics)
        
        # 验证监控结果
        assert len(metrics.snapshots) > 0
        assert metrics.avg_cpu_usage > 0
        assert metrics.avg_memory_usage > 0
        
        # 验证指标计算
        assert metrics.max_cpu_usage >= metrics.avg_cpu_usage
        assert metrics.max_memory_usage >= metrics.avg_memory_usage
    
    @pytest.mark.performance
    @pytest.mark.resource
    @pytest.mark.slow
    def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        self.monitor.start_monitoring()
        
        # 模拟内存泄漏（持续分配内存不释放）
        leaked_data = []
        for i in range(10):
            leaked_data.append(bytearray(10 * 1024 * 1024))  # 10MB
            time.sleep(0.2)
        
        metrics = self.monitor.stop_monitoring()
        self._print_resource_summary(metrics)
        
        # 清理泄漏的内存
        del leaked_data
        gc.collect()
        
        # 验证内存泄漏检测
        assert len(metrics.snapshots) > 0
        assert metrics.memory_growth > 50  # 应该检测到显著的内存增长
    
    @pytest.mark.performance
    @pytest.mark.resource
    @pytest.mark.parametrize("concurrent_level", [1, 5, 10])
    def test_resource_scaling(self, concurrent_level: int):
        """测试资源扩展性"""
        self.monitor.start_monitoring()
        
        # 并发执行任务
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_level) as executor:
            futures = [
                executor.submit(self.test_runner.mixed_workload_task, 0.5)
                for _ in range(concurrent_level)
            ]
            
            # 等待所有任务完成
            concurrent.futures.wait(futures)
        
        metrics = self.monitor.stop_monitoring()
        self._print_resource_summary(metrics)
        
        # 验证资源使用随并发级别增长
        assert len(metrics.snapshots) > 0
        assert metrics.avg_cpu_usage > 0
        assert metrics.avg_memory_usage > 0
        
        # 高并发级别应该有更高的资源使用
        if concurrent_level >= 5:
            assert metrics.max_cpu_usage > 10  # 至少10%的CPU使用率


# ==================== 第三层：基准测试 ====================

class TestResourceMonitoringBenchmarks:
    """资源监控基准测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.monitor = ResourceMonitor(interval=0.01)  # 高频监控
    
    def teardown_method(self):
        """测试后清理"""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
    
    @pytest.mark.performance
    @pytest.mark.resource
    @pytest.mark.benchmark
    def test_resource_monitoring_benchmark(self, benchmark):
        """基准测试：资源监控性能"""
        def monitor_task():
            self.monitor.start_monitoring()
            time.sleep(0.5)  # 监控0.5秒
            metrics = self.monitor.stop_monitoring()
            return {
                "snapshots_count": len(metrics.snapshots),
                "avg_cpu_usage": metrics.avg_cpu_usage,
                "max_memory_usage": metrics.max_memory_usage,
                "monitoring_overhead": "minimal"
            }
        
        result = benchmark(monitor_task)
        
        # 验证基准测试结果
        assert result["snapshots_count"] > 0
        assert result["avg_cpu_usage"] >= 0
        assert result["max_memory_usage"] > 0
        
        print(f"\n=== 基准测试结果 ===")
        print(f"快照数量: {result['snapshots_count']}")
        print(f"平均CPU: {result['avg_cpu_usage']:.1f}%")
        print(f"峰值内存: {result['max_memory_usage']:.1f}MB")
    
    @pytest.mark.performance
    @pytest.mark.resource
    def test_monitoring_overhead(self):
        """测试监控开销"""
        # 测试无监控情况下的基准性能
        start_time = time.time()
        test_runner = ResourceTestRunner()
        test_runner.cpu_intensive_task(duration=0.5)
        baseline_time = time.time() - start_time
        
        # 测试有监控情况下的性能
        self.monitor.start_monitoring()
        start_time = time.time()
        test_runner.cpu_intensive_task(duration=0.5)
        monitored_time = time.time() - start_time
        metrics = self.monitor.stop_monitoring()
        
        # 计算监控开销
        overhead_percent = ((monitored_time - baseline_time) / baseline_time) * 100
        
        print(f"\n=== 监控开销分析 ===")
        print(f"基准执行时间: {baseline_time:.3f}秒")
        print(f"监控执行时间: {monitored_time:.3f}秒")
        print(f"监控开销: {overhead_percent:.1f}%")
        print(f"快照数量: {len(metrics.snapshots)}")
        
        # 验证监控开销在可接受范围内（< 10%）
        assert overhead_percent < 10.0
        assert len(metrics.snapshots) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])