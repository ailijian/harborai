#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResourceUtilizationMonitor 完整测试用例

本模块包含对资源利用率监控器的全面测试，包括：
- 单元测试：测试各个指标收集方法
- 集成测试：测试完整的资源监控流程
- 边界条件测试：测试极端情况和错误处理
- 性能基准测试：验证监控器本身的性能开销
- 阈值告警测试：测试资源阈值监控和告警

作者: HarborAI Team
创建时间: 2024-01-20
遵循: VIBE Coding 规范
"""

import pytest
import asyncio
import time
import threading
import os
import platform
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any
import psutil

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


class TestCPUMetrics:
    """CPUMetrics 测试类"""
    
    def test_cpu_metrics_creation(self):
        """测试CPU指标创建"""
        timestamp = datetime.now()
        cpu_metrics = CPUMetrics(
            timestamp=timestamp,
            total_percent=45.5,
            per_cpu_percent=[40.0, 50.0, 45.0, 46.0],
            load_average=[1.5, 1.2, 1.0],
            context_switches=12345,
            interrupts=6789,
            frequency=2400.0
        )
        
        assert cpu_metrics.timestamp == timestamp
        assert cpu_metrics.total_percent == 45.5
        assert cpu_metrics.per_cpu_percent == [40.0, 50.0, 45.0, 46.0]
        assert cpu_metrics.load_average == [1.5, 1.2, 1.0]
        assert cpu_metrics.context_switches == 12345
        assert cpu_metrics.interrupts == 6789
        assert cpu_metrics.frequency == 2400.0
    
    def test_cpu_metrics_to_dict(self):
        """测试CPU指标转换为字典"""
        timestamp = datetime.now()
        cpu_metrics = CPUMetrics(
            timestamp=timestamp,
            total_percent=50.0,
            per_cpu_percent=[45.0, 55.0],
            load_average=[1.0, 0.8, 0.6],
            context_switches=1000,
            interrupts=500,
            frequency=2000.0
        )
        
        result_dict = cpu_metrics.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['timestamp'] == timestamp.isoformat()
        assert result_dict['total_percent'] == 50.0
        assert result_dict['per_cpu_percent'] == [45.0, 55.0]
        assert result_dict['load_average'] == [1.0, 0.8, 0.6]
        assert result_dict['context_switches'] == 1000
        assert result_dict['interrupts'] == 500
        assert result_dict['frequency'] == 2000.0


class TestMemoryMetrics:
    """MemoryMetrics 测试类"""
    
    def test_memory_metrics_creation(self):
        """测试内存指标创建"""
        timestamp = datetime.now()
        memory_metrics = MemoryMetrics(
            timestamp=timestamp,
            total=8 * 1024**3,  # 8GB
            available=4 * 1024**3,  # 4GB
            used=4 * 1024**3,  # 4GB
            percent=50.0,
            swap_total=2 * 1024**3,  # 2GB
            swap_used=512 * 1024**2,  # 512MB
            swap_percent=25.0,
            buffers=256 * 1024**2,  # 256MB
            cached=512 * 1024**2  # 512MB
        )
        
        assert memory_metrics.timestamp == timestamp
        assert memory_metrics.total == 8 * 1024**3
        assert memory_metrics.available == 4 * 1024**3
        assert memory_metrics.used == 4 * 1024**3
        assert memory_metrics.percent == 50.0
        assert memory_metrics.swap_total == 2 * 1024**3
        assert memory_metrics.swap_used == 512 * 1024**2
        assert memory_metrics.swap_percent == 25.0
        assert memory_metrics.buffers == 256 * 1024**2
        assert memory_metrics.cached == 512 * 1024**2
    
    def test_memory_metrics_to_dict(self):
        """测试内存指标转换为字典"""
        timestamp = datetime.now()
        memory_metrics = MemoryMetrics(
            timestamp=timestamp,
            total=1024,
            available=512,
            used=512,
            percent=50.0,
            swap_total=256,
            swap_used=128,
            swap_percent=50.0,
            buffers=64,
            cached=128
        )
        
        result_dict = memory_metrics.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['timestamp'] == timestamp.isoformat()
        assert result_dict['total'] == 1024
        assert result_dict['available'] == 512
        assert result_dict['used'] == 512
        assert result_dict['percent'] == 50.0
        assert result_dict['swap_total'] == 256
        assert result_dict['swap_used'] == 128
        assert result_dict['swap_percent'] == 50.0
        assert result_dict['buffers'] == 64
        assert result_dict['cached'] == 128


class TestResourceThresholds:
    """ResourceThresholds 测试类"""
    
    def test_default_thresholds(self):
        """测试默认阈值"""
        thresholds = ResourceThresholds()
        
        assert thresholds.cpu_warning == 80.0
        assert thresholds.cpu_critical == 95.0
        assert thresholds.memory_warning == 80.0
        assert thresholds.memory_critical == 95.0
        assert thresholds.disk_warning == 80.0
        assert thresholds.disk_critical == 95.0
        assert thresholds.gpu_warning == 80.0
        assert thresholds.gpu_critical == 95.0
    
    def test_custom_thresholds(self):
        """测试自定义阈值"""
        thresholds = ResourceThresholds(
            cpu_warning=70.0,
            cpu_critical=90.0,
            memory_warning=75.0,
            memory_critical=92.0
        )
        
        assert thresholds.cpu_warning == 70.0
        assert thresholds.cpu_critical == 90.0
        assert thresholds.memory_warning == 75.0
        assert thresholds.memory_critical == 92.0


class TestResourceUtilizationMonitor:
    """ResourceUtilizationMonitor 测试类"""
    
    @pytest.fixture
    def monitor(self):
        """资源监控器fixture"""
        return ResourceUtilizationMonitor(
            monitoring_interval=0.1,  # 快速测试
            max_snapshots=50,
            monitor_gpu=False  # 避免GPU依赖问题
        )
    
    @pytest.fixture
    def custom_thresholds(self):
        """自定义阈值fixture"""
        return ResourceThresholds(
            cpu_warning=60.0,
            cpu_critical=85.0,
            memory_warning=70.0,
            memory_critical=90.0
        )
    
    def test_monitor_initialization(self):
        """测试监控器初始化"""
        monitor = ResourceUtilizationMonitor(
            monitoring_interval=2.0,
            max_snapshots=200,
            monitor_gpu=True,
            target_process=os.getpid()
        )
        
        assert monitor.monitoring_interval == 2.0
        assert monitor.max_snapshots == 200
        # GPU监控取决于GPU_AVAILABLE的值
        from .resource_utilization_monitor import GPU_AVAILABLE
        assert monitor.monitor_gpu == (True and GPU_AVAILABLE)
        assert monitor.target_process.pid == os.getpid()
        assert monitor.is_monitoring is False
        assert len(monitor.snapshots) == 0
        assert len(monitor.alert_callbacks) == 0
    
    def test_add_alert_callback(self, monitor):
        """测试添加告警回调"""
        callback1 = Mock()
        callback2 = Mock()
        
        monitor.add_alert_callback(callback1)
        monitor.add_alert_callback(callback2)
        
        assert len(monitor.alert_callbacks) == 2
        assert callback1 in monitor.alert_callbacks
        assert callback2 in monitor.alert_callbacks
    
    @patch('tests.performance.resource_utilization_monitor.psutil.cpu_percent')
    @patch('tests.performance.resource_utilization_monitor.psutil.cpu_count')
    @patch('tests.performance.resource_utilization_monitor.psutil.cpu_freq')
    @patch('tests.performance.resource_utilization_monitor.psutil.cpu_stats')
    def test_get_cpu_metrics(self, mock_cpu_stats, mock_cpu_freq, mock_cpu_count, mock_cpu_percent, monitor):
        """测试获取CPU指标"""
        # 设置mock返回值
        mock_cpu_percent.side_effect = [50.0, [45.0, 55.0, 48.0, 52.0]]
        mock_cpu_count.return_value = 4
        mock_cpu_freq.return_value = Mock(current=2400.0)
        mock_cpu_stats.return_value = Mock(ctx_switches=12345, interrupts=6789)
        
        # 模拟load average（仅在Unix系统上可用）
        if hasattr(os, 'getloadavg'):
            with patch('os.getloadavg', return_value=(1.5, 1.2, 1.0)):
                cpu_metrics = monitor._get_cpu_metrics(datetime.now())
        else:
            cpu_metrics = monitor._get_cpu_metrics(datetime.now())
        
        # 验证CPU指标
        assert isinstance(cpu_metrics, CPUMetrics)
        assert cpu_metrics.total_percent == 50.0
        assert cpu_metrics.per_cpu_percent == [45.0, 55.0, 48.0, 52.0]
        assert cpu_metrics.context_switches == 12345
        assert cpu_metrics.interrupts == 6789
        assert cpu_metrics.frequency == 2400.0
    
    @patch('tests.performance.resource_utilization_monitor.psutil.virtual_memory')
    @patch('tests.performance.resource_utilization_monitor.psutil.swap_memory')
    def test_get_memory_metrics(self, mock_swap_memory, mock_virtual_memory, monitor):
        """测试获取内存指标"""
        # 设置mock返回值
        mock_virtual_memory.return_value = Mock(
            total=8 * 1024**3,
            available=4 * 1024**3,
            used=4 * 1024**3,
            percent=50.0,
            buffers=256 * 1024**2,
            cached=512 * 1024**2
        )
        mock_swap_memory.return_value = Mock(
            total=2 * 1024**3,
            used=512 * 1024**2,
            percent=25.0
        )
        
        memory_metrics = monitor._get_memory_metrics(datetime.now())
        
        # 验证内存指标
        assert isinstance(memory_metrics, MemoryMetrics)
        assert memory_metrics.total == 8 * 1024**3
        assert memory_metrics.available == 4 * 1024**3
        assert memory_metrics.used == 4 * 1024**3
        assert memory_metrics.percent == 50.0
        assert memory_metrics.swap_total == 2 * 1024**3
        assert memory_metrics.swap_used == 512 * 1024**2
        assert memory_metrics.swap_percent == 25.0
    
    @patch('tests.performance.resource_utilization_monitor.psutil.disk_io_counters')
    @patch('tests.performance.resource_utilization_monitor.psutil.disk_usage')
    @patch('tests.performance.resource_utilization_monitor.psutil.disk_partitions')
    def test_get_disk_metrics(self, mock_disk_partitions, mock_disk_usage, mock_disk_io_counters, monitor):
        """测试获取磁盘指标"""
        # 设置mock返回值
        mock_disk_io_counters.return_value = Mock(
            read_bytes=1024 * 1024,
            write_bytes=512 * 1024,
            read_count=100,
            write_count=50,
            read_time=1000,
            write_time=500
        )
        mock_disk_partitions.return_value = [
            Mock(mountpoint='C:\\' if platform.system() == 'Windows' else '/')
        ]
        mock_disk_usage.return_value = Mock(
            total=100 * 1024**3,
            used=50 * 1024**3,
            free=50 * 1024**3
        )
        
        disk_metrics = monitor._get_disk_metrics(datetime.now())
        
        # 验证磁盘指标
        assert isinstance(disk_metrics, DiskMetrics)
        assert disk_metrics.read_bytes == 1024 * 1024
        assert disk_metrics.write_bytes == 512 * 1024
        assert disk_metrics.read_count == 100
        assert disk_metrics.write_count == 50
        assert disk_metrics.read_time == 1000
        assert disk_metrics.write_time == 500
        assert len(disk_metrics.disk_usage) > 0
    
    @patch('tests.performance.resource_utilization_monitor.psutil.net_io_counters')
    @patch('tests.performance.resource_utilization_monitor.psutil.net_connections')
    def test_get_network_metrics(self, mock_net_connections, mock_net_io_counters, monitor):
        """测试获取网络指标"""
        # 设置mock返回值
        mock_net_io_counters.return_value = Mock(
            bytes_sent=1024 * 1024,
            bytes_recv=2048 * 1024,
            packets_sent=1000,
            packets_recv=2000,
            errin=5,
            errout=3,
            dropin=2,
            dropout=1
        )
        mock_net_connections.return_value = [Mock() for _ in range(10)]  # 10个连接
        
        network_metrics = monitor._get_network_metrics(datetime.now())
        
        # 验证网络指标
        assert isinstance(network_metrics, NetworkMetrics)
        assert network_metrics.bytes_sent == 1024 * 1024
        assert network_metrics.bytes_recv == 2048 * 1024
        assert network_metrics.packets_sent == 1000
        assert network_metrics.packets_recv == 2000
        assert network_metrics.errin == 5
        assert network_metrics.errout == 3
        assert network_metrics.dropin == 2
        assert network_metrics.dropout == 1
        assert network_metrics.connections == 10
    
    @patch('tests.performance.resource_utilization_monitor.psutil.Process')
    def test_get_process_metrics(self, mock_process_class, monitor):
        """测试获取进程指标"""
        
        # 设置mock进程
        mock_process = Mock()
        mock_process.pid = 1234
        mock_process.name.return_value = "test_process"
        mock_process.cpu_percent.return_value = 25.5
        mock_process.memory_percent.return_value = 15.0
        mock_process.memory_info.return_value = Mock(
            rss=100 * 1024 * 1024,  # 100MB
            vms=200 * 1024 * 1024   # 200MB
        )
        mock_process.num_threads.return_value = 8
        
        # 模拟文件描述符数（仅Unix系统）
        if hasattr(psutil.Process, 'num_fds'):
            mock_process.num_fds.return_value = 50
        else:
            mock_process.num_fds = Mock(side_effect=AttributeError())
        
        # 模拟IO统计
        mock_process.io_counters.return_value = Mock(
            read_bytes=1024 * 1024,
            write_bytes=512 * 1024
        )
        
        mock_process_class.return_value = mock_process
        
        # 设置目标进程为mock对象
        monitor.target_process = mock_process
        
        process_metrics = monitor._get_process_metrics(datetime.now())
        
        # 验证进程指标
        assert isinstance(process_metrics, ProcessMetrics)
        assert process_metrics.pid == 1234
        assert process_metrics.name == "test_process"
        assert process_metrics.cpu_percent == 25.5
        assert process_metrics.memory_percent == 15.0
        assert process_metrics.memory_rss == 100 * 1024 * 1024
        assert process_metrics.memory_vms == 200 * 1024 * 1024
        assert process_metrics.num_threads == 8
        assert process_metrics.io_read_bytes == 1024 * 1024
        assert process_metrics.io_write_bytes == 512 * 1024
    
    def test_start_stop_monitoring(self, monitor):
        """测试开始和停止监控"""
        # 测试开始监控
        monitor.start_monitoring()
        assert monitor.is_monitoring is True
        assert monitor.monitor_thread is not None
        assert monitor.monitor_thread.is_alive()
        
        # 等待一小段时间让监控运行
        time.sleep(0.3)
        
        # 验证收集了数据
        assert len(monitor.snapshots) > 0
        
        # 测试停止监控
        monitor.stop_monitoring()
        assert monitor.is_monitoring is False
        
        # 等待线程结束
        if monitor.monitor_thread:
            monitor.monitor_thread.join(timeout=1.0)
        
        assert not monitor.monitor_thread.is_alive()
    
    def test_take_snapshot(self, monitor):
        """测试获取系统资源快照"""
        snapshot = monitor._take_snapshot()
        
        # 验证快照
        assert isinstance(snapshot, SystemResourceSnapshot)
        assert isinstance(snapshot.cpu_metrics, CPUMetrics)
        assert isinstance(snapshot.memory_metrics, MemoryMetrics)
        assert isinstance(snapshot.disk_metrics, DiskMetrics)
        assert isinstance(snapshot.network_metrics, NetworkMetrics)
        assert isinstance(snapshot.process_metrics, ProcessMetrics)
        # GPU指标可能为None（如果未启用或不可用）
        assert snapshot.gpu_metrics is None or isinstance(snapshot.gpu_metrics, GPUMetrics)
    
    def test_threshold_checking(self, custom_thresholds):
        """测试阈值检查"""
        monitor = ResourceUtilizationMonitor(
            monitoring_interval=0.1,
            thresholds=custom_thresholds,
            monitor_gpu=False
        )
        
        # 添加告警回调
        alert_calls = []
        def alert_callback(alert_type: str, alert_data: Dict[str, Any]):
            alert_calls.append((alert_type, alert_data))
        
        monitor.add_alert_callback(alert_callback)
        
        # 创建一个超过阈值的快照
        timestamp = datetime.now()
        high_cpu_snapshot = SystemResourceSnapshot(
            timestamp=timestamp,
            cpu_metrics=CPUMetrics(
                timestamp=timestamp,
                total_percent=75.0,  # 超过警告阈值60.0但低于严重阈值85.0
                per_cpu_percent=[70.0, 80.0],
                load_average=None,
                context_switches=1000,
                interrupts=500,
                frequency=2400.0
            ),
            memory_metrics=MemoryMetrics(
                timestamp=timestamp,
                total=1024,
                available=200,
                used=824,
                percent=80.5,  # 超过警告阈值70.0
                swap_total=512,
                swap_used=100,
                swap_percent=19.5,
                buffers=50,
                cached=100
            ),
            disk_metrics=DiskMetrics(
                timestamp=timestamp,
                read_bytes=1000,
                write_bytes=500,
                read_count=10,
                write_count=5,
                read_time=100,
                write_time=50,
                disk_usage={}
            ),
            network_metrics=NetworkMetrics(
                timestamp=timestamp,
                bytes_sent=1000,
                bytes_recv=2000,
                packets_sent=10,
                packets_recv=20,
                errin=0,
                errout=0,
                dropin=0,
                dropout=0,
                connections=5
            ),
            gpu_metrics=None,
            process_metrics=ProcessMetrics(
                timestamp=timestamp,
                pid=1234,
                name="test",
                cpu_percent=50.0,
                memory_percent=30.0,
                memory_rss=1024,
                memory_vms=2048,
                num_threads=4,
                num_fds=20,
                io_read_bytes=1000,
                io_write_bytes=500
            )
        )
        
        # 检查阈值
        monitor._check_thresholds(high_cpu_snapshot)
        
        # 验证告警被触发
        assert len(alert_calls) >= 2  # CPU和内存都应该触发告警
        alert_types = [call[0] for call in alert_calls]
        assert 'CPU_WARNING' in alert_types
        assert 'MEMORY_WARNING' in alert_types
    
    def test_get_resource_statistics(self, monitor):
        """测试获取资源统计"""
        # 添加一些模拟快照
        base_time = datetime.now()
        for i in range(5):
            timestamp = base_time + timedelta(seconds=i * 0.1)
            snapshot = SystemResourceSnapshot(
                timestamp=timestamp,
                cpu_metrics=CPUMetrics(
                    timestamp=timestamp,
                    total_percent=50.0 + i * 5,
                    per_cpu_percent=[45.0, 55.0],
                    load_average=None,
                    context_switches=1000,
                    interrupts=500,
                    frequency=2400.0
                ),
                memory_metrics=MemoryMetrics(
                    timestamp=timestamp,
                    total=1024,
                    available=512,
                    used=512,
                    percent=50.0,
                    swap_total=256,
                    swap_used=128,
                    swap_percent=50.0,
                    buffers=64,
                    cached=128
                ),
                disk_metrics=DiskMetrics(
                    timestamp=timestamp,
                    read_bytes=1000,
                    write_bytes=500,
                    read_count=10,
                    write_count=5,
                    read_time=100,
                    write_time=50,
                    disk_usage={}
                ),
                network_metrics=NetworkMetrics(
                    timestamp=timestamp,
                    bytes_sent=1000,
                    bytes_recv=2000,
                    packets_sent=10,
                    packets_recv=20,
                    errin=0,
                    errout=0,
                    dropin=0,
                    dropout=0,
                    connections=5
                ),
                gpu_metrics=None,
                process_metrics=ProcessMetrics(
                    timestamp=timestamp,
                    pid=1234,
                    name="test",
                    cpu_percent=25.0,
                    memory_percent=15.0,
                    memory_rss=1024,
                    memory_vms=2048,
                    num_threads=4,
                    num_fds=20,
                    io_read_bytes=1000,
                    io_write_bytes=500
                )
            )
            monitor.snapshots.append(snapshot)
        
        stats = monitor.get_resource_statistics()
        
        # 验证统计信息
        assert isinstance(stats, dict)
        assert 'monitoring_duration' in stats
        assert 'snapshot_count' in stats
        assert 'cpu' in stats
        assert 'memory' in stats
        assert 'disk' in stats
        assert 'network' in stats
        assert 'process' in stats
        
        assert stats['snapshot_count'] == 5
        assert stats['cpu']['average_percent'] > 0
        assert stats['cpu']['peak_percent'] > 0
        assert stats['memory']['average_percent'] == 50.0
    
    def test_context_manager(self, monitor):
        """测试上下文管理器"""
        with monitor as m:
            assert m is monitor
            assert m.is_monitoring is True
            time.sleep(0.2)  # 让监控运行一段时间
        
        # 退出上下文后应该停止监控
        assert monitor.is_monitoring is False


class TestResourceUtilizationMonitorIntegration:
    """ResourceUtilizationMonitor 集成测试类"""
    
    @pytest.mark.asyncio
    async def test_monitor_resources_async(self):
        """测试异步资源监控"""
        monitor = ResourceUtilizationMonitor(
            monitoring_interval=0.1,
            max_snapshots=20,
            monitor_gpu=False
        )
        
        # 运行异步监控
        snapshots = await monitor.monitor_resources_async(duration=1.0, interval=0.1)
        
        # 验证结果
        assert isinstance(snapshots, list)
        assert len(snapshots) > 0
        assert all(isinstance(snapshot, SystemResourceSnapshot) for snapshot in snapshots)
        
        # 验证时间间隔
        if len(snapshots) > 1:
            time_diff = (snapshots[1].timestamp - snapshots[0].timestamp).total_seconds()
            assert 0.05 <= time_diff <= 0.2  # 允许一些误差
    
    def test_monitor_resource_usage_function(self):
        """测试资源使用监控函数"""
        def cpu_intensive_task():
            """CPU密集型任务"""
            result = 0
            for i in range(100000):
                result += i * i
            return result
        
        # 监控资源使用
        stats = monitor_resource_usage(
            test_function=cpu_intensive_task,
            duration=2.0,
            monitoring_interval=0.2
        )
        
        # 验证统计结果
        assert isinstance(stats, dict)
        assert 'monitoring_duration' in stats
        assert 'snapshot_count' in stats
        assert 'cpu' in stats
        assert 'memory' in stats
        assert 'function_result' in stats
        
        assert stats['snapshot_count'] > 0
        assert stats['cpu']['peak_percent'] > 0
        assert stats['function_result'] is not None
    
    def test_real_system_monitoring(self):
        """测试真实系统监控"""
        monitor = ResourceUtilizationMonitor(
            monitoring_interval=0.2,
            max_snapshots=10,
            monitor_gpu=False
        )
        
        try:
            # 启动监控
            monitor.start_monitoring()
            
            # 执行一些系统操作
            data = []
            for i in range(1000):
                data.append([j for j in range(100)])
            
            # 等待收集数据
            time.sleep(1.0)
            
            # 获取统计信息
            stats = monitor.get_resource_statistics()
            
            # 验证收集了真实数据
            assert stats['snapshot_count'] > 0
            assert stats['cpu']['average_percent'] >= 0
            assert stats['memory']['average_percent'] > 0
            assert stats['process']['average_cpu_percent'] >= 0
            
        finally:
            monitor.stop_monitoring()


class TestResourceUtilizationMonitorBenchmarks:
    """ResourceUtilizationMonitor 性能基准测试类"""
    
    def test_snapshot_creation_performance(self, benchmark):
        """基准测试：快照创建性能"""
        monitor = ResourceUtilizationMonitor(monitor_gpu=False)
        
        def create_snapshot():
            return monitor._take_snapshot()
        
        # 运行基准测试
        result = benchmark(create_snapshot)
        assert isinstance(result, SystemResourceSnapshot)
    
    def test_monitoring_overhead(self):
        """测试监控开销"""
        monitor = ResourceUtilizationMonitor(
            monitoring_interval=0.01,  # 高频监控
            max_snapshots=100,
            monitor_gpu=False
        )
        
        # 测量监控前的CPU使用率
        initial_cpu = psutil.cpu_percent(interval=0.1)
        
        try:
            # 启动监控
            start_time = time.time()
            monitor.start_monitoring()
            
            # 运行一段时间
            time.sleep(1.0)
            
            # 测量监控期间的CPU使用率
            monitoring_cpu = psutil.cpu_percent(interval=0.1)
            end_time = time.time()
            
            # 计算开销
            cpu_overhead = monitoring_cpu - initial_cpu
            time_overhead = end_time - start_time
            
            # 验证开销在合理范围内
            assert cpu_overhead < 20.0  # CPU开销小于20%
            assert time_overhead >= 1.0  # 至少运行了1秒
            
            # 验证收集了数据
            assert len(monitor.snapshots) > 0
            
        finally:
            monitor.stop_monitoring()
    
    def test_large_dataset_performance(self, benchmark):
        """基准测试：大数据集处理性能"""
        monitor = ResourceUtilizationMonitor(monitor_gpu=False)
        
        # 添加大量模拟快照
        base_time = datetime.now()
        for i in range(1000):
            timestamp = base_time + timedelta(seconds=i * 0.1)
            snapshot = SystemResourceSnapshot(
                timestamp=timestamp,
                cpu_metrics=CPUMetrics(
                    timestamp=timestamp,
                    total_percent=50.0 + (i % 50),
                    per_cpu_percent=[45.0, 55.0],
                    load_average=None,
                    context_switches=1000 + i,
                    interrupts=500 + i,
                    frequency=2400.0
                ),
                memory_metrics=MemoryMetrics(
                    timestamp=timestamp,
                    total=1024 * 1024 * 1024,
                    available=512 * 1024 * 1024,
                    used=512 * 1024 * 1024,
                    percent=50.0,
                    swap_total=256 * 1024 * 1024,
                    swap_used=128 * 1024 * 1024,
                    swap_percent=50.0,
                    buffers=64 * 1024 * 1024,
                    cached=128 * 1024 * 1024
                ),
                disk_metrics=DiskMetrics(
                    timestamp=timestamp,
                    read_bytes=1000 + i,
                    write_bytes=500 + i,
                    read_count=10 + i,
                    write_count=5 + i,
                    read_time=100,
                    write_time=50,
                    disk_usage={}
                ),
                network_metrics=NetworkMetrics(
                    timestamp=timestamp,
                    bytes_sent=1000 + i,
                    bytes_recv=2000 + i,
                    packets_sent=10 + i,
                    packets_recv=20 + i,
                    errin=0,
                    errout=0,
                    dropin=0,
                    dropout=0,
                    connections=5
                ),
                gpu_metrics=None,
                process_metrics=ProcessMetrics(
                    timestamp=timestamp,
                    pid=1234,
                    name="test",
                    cpu_percent=25.0,
                    memory_percent=15.0,
                    memory_rss=1024 * 1024,
                    memory_vms=2048 * 1024,
                    num_threads=4,
                    num_fds=20,
                    io_read_bytes=1000 + i,
                    io_write_bytes=500 + i
                )
            )
            monitor.snapshots.append(snapshot)
        
        def get_statistics():
            return monitor.get_resource_statistics()
        
        # 运行基准测试
        result = benchmark(get_statistics)
        assert isinstance(result, dict)
        assert result['snapshot_count'] == 1000


class TestResourceUtilizationMonitorEdgeCases:
    """ResourceUtilizationMonitor 边界条件测试类"""
    
    def test_invalid_monitoring_interval(self):
        """测试无效的监控间隔"""
        # 测试负数间隔
        with pytest.raises(ValueError):
            ResourceUtilizationMonitor(monitoring_interval=-1.0)
        
        # 测试零间隔
        with pytest.raises(ValueError):
            ResourceUtilizationMonitor(monitoring_interval=0.0)
    
    def test_invalid_max_snapshots(self):
        """测试无效的最大快照数"""
        # 测试负数
        with pytest.raises(ValueError):
            ResourceUtilizationMonitor(max_snapshots=-1)
        
        # 测试零
        with pytest.raises(ValueError):
            ResourceUtilizationMonitor(max_snapshots=0)
    
    def test_nonexistent_target_process(self):
        """测试不存在的目标进程"""
        # 测试初始化时传入不存在的进程PID应该抛出异常
        with pytest.raises(psutil.NoSuchProcess):
            ResourceUtilizationMonitor(
                target_process=999999,  # 不太可能存在的PID
                monitor_gpu=False
            )
    
    def test_max_snapshots_limit(self):
        """测试最大快照数量限制"""
        monitor = ResourceUtilizationMonitor(
            monitoring_interval=0.01,
            max_snapshots=5,
            monitor_gpu=False
        )
        
        try:
            # 启动监控
            monitor.start_monitoring()
            
            # 等待收集超过限制的快照
            time.sleep(0.5)
            
            # 验证快照数量限制
            assert len(monitor.snapshots) <= monitor.max_snapshots
            
        finally:
            monitor.stop_monitoring()
    
    def test_concurrent_monitoring_calls(self):
        """测试并发监控调用"""
        monitor = ResourceUtilizationMonitor(
            monitoring_interval=0.1,
            monitor_gpu=False
        )
        
        try:
            # 启动第一次监控
            monitor.start_monitoring()
            assert monitor.is_monitoring is True
            
            # 尝试再次启动监控（应该被忽略）
            monitor.start_monitoring()
            assert monitor.is_monitoring is True
            
            # 应该只有一个监控线程
            thread_count = sum(1 for t in threading.enumerate() if 'resource_monitor' in t.name.lower())
            assert thread_count <= 1
            
        finally:
            monitor.stop_monitoring()


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short", "--benchmark-skip"])