"""
资源利用率监控器模块

该模块提供全面的系统资源监控功能，支持：
- CPU使用率监控（总体和各核心）
- 内存使用监控（物理内存、虚拟内存、交换空间）
- 磁盘I/O监控（读写速度、IOPS）
- 网络I/O监控（带宽使用、连接数）
- GPU资源监控（如果可用）
- 进程级资源监控

作者：HarborAI性能测试团队
创建时间：2024年
"""

import os
import sys
import time
import psutil
import threading
import platform
import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import statistics
import logging
from pathlib import Path
import json

# 尝试导入GPU监控库
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class CPUMetrics:
    """CPU指标数据结构"""
    timestamp: datetime
    total_percent: float  # 总体CPU使用率
    per_cpu_percent: List[float]  # 各核心CPU使用率
    load_average: Optional[List[float]]  # 负载平均值（Linux/macOS）
    context_switches: int  # 上下文切换次数
    interrupts: int  # 中断次数
    frequency: float  # CPU频率（MHz）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_percent': self.total_percent,
            'per_cpu_percent': self.per_cpu_percent,
            'load_average': self.load_average,
            'context_switches': self.context_switches,
            'interrupts': self.interrupts,
            'frequency': self.frequency
        }


@dataclass
class MemoryMetrics:
    """内存指标数据结构"""
    timestamp: datetime
    total: int  # 总内存（bytes）
    available: int  # 可用内存（bytes）
    used: int  # 已用内存（bytes）
    percent: float  # 内存使用百分比
    swap_total: int  # 交换空间总量（bytes）
    swap_used: int  # 交换空间使用量（bytes）
    swap_percent: float  # 交换空间使用百分比
    buffers: int  # 缓冲区（Linux）
    cached: int  # 缓存（Linux）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total': self.total,
            'available': self.available,
            'used': self.used,
            'percent': self.percent,
            'swap_total': self.swap_total,
            'swap_used': self.swap_used,
            'swap_percent': self.swap_percent,
            'buffers': self.buffers,
            'cached': self.cached
        }


@dataclass
class DiskMetrics:
    """磁盘I/O指标数据结构"""
    timestamp: datetime
    read_bytes: int  # 读取字节数
    write_bytes: int  # 写入字节数
    read_count: int  # 读取次数
    write_count: int  # 写入次数
    read_time: int  # 读取时间（ms）
    write_time: int  # 写入时间（ms）
    disk_usage: Dict[str, Dict[str, Union[int, float]]]  # 各分区使用情况
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'read_bytes': self.read_bytes,
            'write_bytes': self.write_bytes,
            'read_count': self.read_count,
            'write_count': self.write_count,
            'read_time': self.read_time,
            'write_time': self.write_time,
            'disk_usage': self.disk_usage
        }


@dataclass
class NetworkMetrics:
    """网络I/O指标数据结构"""
    timestamp: datetime
    bytes_sent: int  # 发送字节数
    bytes_recv: int  # 接收字节数
    packets_sent: int  # 发送包数
    packets_recv: int  # 接收包数
    errin: int  # 接收错误数
    errout: int  # 发送错误数
    dropin: int  # 接收丢包数
    dropout: int  # 发送丢包数
    connections: int  # 活跃连接数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'bytes_sent': self.bytes_sent,
            'bytes_recv': self.bytes_recv,
            'packets_sent': self.packets_sent,
            'packets_recv': self.packets_recv,
            'errin': self.errin,
            'errout': self.errout,
            'dropin': self.dropin,
            'dropout': self.dropout,
            'connections': self.connections
        }


@dataclass
class GPUMetrics:
    """GPU指标数据结构"""
    timestamp: datetime
    gpu_count: int  # GPU数量
    gpu_utilization: List[float]  # 各GPU使用率
    memory_total: List[int]  # 各GPU总内存（MB）
    memory_used: List[int]  # 各GPU已用内存（MB）
    memory_percent: List[float]  # 各GPU内存使用百分比
    temperature: List[float]  # 各GPU温度（°C）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'gpu_count': self.gpu_count,
            'gpu_utilization': self.gpu_utilization,
            'memory_total': self.memory_total,
            'memory_used': self.memory_used,
            'memory_percent': self.memory_percent,
            'temperature': self.temperature
        }


@dataclass
class ProcessMetrics:
    """进程指标数据结构"""
    timestamp: datetime
    pid: int  # 进程ID
    name: str  # 进程名称
    cpu_percent: float  # CPU使用率
    memory_percent: float  # 内存使用率
    memory_rss: int  # 常驻内存（bytes）
    memory_vms: int  # 虚拟内存（bytes）
    num_threads: int  # 线程数
    num_fds: int  # 文件描述符数（Unix）
    io_read_bytes: int  # 读取字节数
    io_write_bytes: int  # 写入字节数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'pid': self.pid,
            'name': self.name,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_rss': self.memory_rss,
            'memory_vms': self.memory_vms,
            'num_threads': self.num_threads,
            'num_fds': self.num_fds,
            'io_read_bytes': self.io_read_bytes,
            'io_write_bytes': self.io_write_bytes
        }


@dataclass
class SystemResourceSnapshot:
    """系统资源快照"""
    timestamp: datetime
    cpu_metrics: CPUMetrics
    memory_metrics: MemoryMetrics
    disk_metrics: DiskMetrics
    network_metrics: NetworkMetrics
    gpu_metrics: Optional[GPUMetrics]
    process_metrics: ProcessMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_metrics': self.cpu_metrics.to_dict(),
            'memory_metrics': self.memory_metrics.to_dict(),
            'disk_metrics': self.disk_metrics.to_dict(),
            'network_metrics': self.network_metrics.to_dict(),
            'gpu_metrics': self.gpu_metrics.to_dict() if self.gpu_metrics else None,
            'process_metrics': self.process_metrics.to_dict()
        }


@dataclass
class ResourceThresholds:
    """资源阈值配置"""
    cpu_warning: float = 80.0  # CPU使用率警告阈值（%）
    cpu_critical: float = 95.0  # CPU使用率严重阈值（%）
    memory_warning: float = 80.0  # 内存使用率警告阈值（%）
    memory_critical: float = 95.0  # 内存使用率严重阈值（%）
    disk_warning: float = 80.0  # 磁盘使用率警告阈值（%）
    disk_critical: float = 95.0  # 磁盘使用率严重阈值（%）
    gpu_warning: float = 80.0  # GPU使用率警告阈值（%）
    gpu_critical: float = 95.0  # GPU使用率严重阈值（%）


class ResourceUtilizationMonitor:
    """
    资源利用率监控器
    
    功能特性：
    - 全面的系统资源监控
    - 实时资源使用率统计
    - 资源阈值告警
    - 历史数据分析
    - 性能瓶颈识别
    """
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        max_snapshots: int = 1000,
        thresholds: Optional[ResourceThresholds] = None,
        monitor_gpu: bool = True,
        target_process: Optional[int] = None
    ):
        """
        初始化资源利用率监控器
        
        参数:
            monitoring_interval: 监控间隔（秒）
            max_snapshots: 最大快照数量
            thresholds: 资源阈值配置
            monitor_gpu: 是否监控GPU
            target_process: 目标进程PID（None表示当前进程）
        """
        # 参数验证
        if monitoring_interval <= 0:
            raise ValueError("监控间隔必须大于0")
        if max_snapshots <= 0:
            raise ValueError("最大快照数量必须大于0")
        
        self.monitoring_interval = monitoring_interval
        self.max_snapshots = max_snapshots
        self.thresholds = thresholds or ResourceThresholds()
        self.monitor_gpu = monitor_gpu and GPU_AVAILABLE
        
        # 目标进程
        if target_process is None:
            self.target_process = psutil.Process()
        else:
            try:
                self.target_process = psutil.Process(target_process)
            except psutil.NoSuchProcess:
                raise psutil.NoSuchProcess(f"进程 PID {target_process} 不存在")
        
        # 资源快照存储
        self.snapshots: deque = deque(maxlen=max_snapshots)
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 告警回调
        self.alert_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        
        # 统计信息
        self.total_snapshots = 0
        self.alert_count = 0
        
        # 基线数据（用于计算增量）
        self.baseline_disk_io = None
        self.baseline_network_io = None
        
        logger.info(f"资源监控器初始化完成，目标进程: {self.target_process.pid}")
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """开始资源监控"""
        if self.is_monitoring:
            logger.warning("资源监控已在运行中")
            return
        
        # 获取基线数据
        self._initialize_baselines()
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="ResourceUtilizationMonitor"
        )
        self.monitor_thread.start()
        logger.info("资源利用率监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止资源监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("资源利用率监控已停止")
    
    def _initialize_baselines(self) -> None:
        """初始化基线数据"""
        try:
            self.baseline_disk_io = psutil.disk_io_counters()
            self.baseline_network_io = psutil.net_io_counters()
        except Exception as e:
            logger.warning(f"初始化基线数据失败: {e}")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                self.total_snapshots += 1
                
                # 检查资源阈值
                self._check_thresholds(snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"资源监控循环出错: {e}")
                time.sleep(self.monitoring_interval)
    
    def _take_snapshot(self) -> SystemResourceSnapshot:
        """获取系统资源快照"""
        timestamp = datetime.now()
        
        # 获取CPU指标
        cpu_metrics = self._get_cpu_metrics(timestamp)
        
        # 获取内存指标
        memory_metrics = self._get_memory_metrics(timestamp)
        
        # 获取磁盘I/O指标
        disk_metrics = self._get_disk_metrics(timestamp)
        
        # 获取网络I/O指标
        network_metrics = self._get_network_metrics(timestamp)
        
        # 获取GPU指标（如果可用）
        gpu_metrics = self._get_gpu_metrics(timestamp) if self.monitor_gpu else None
        
        # 获取进程指标
        process_metrics = self._get_process_metrics(timestamp)
        
        return SystemResourceSnapshot(
            timestamp=timestamp,
            cpu_metrics=cpu_metrics,
            memory_metrics=memory_metrics,
            disk_metrics=disk_metrics,
            network_metrics=network_metrics,
            gpu_metrics=gpu_metrics,
            process_metrics=process_metrics
        )
    
    def _get_cpu_metrics(self, timestamp: datetime) -> CPUMetrics:
        """获取CPU指标"""
        try:
            # CPU使用率
            total_percent = psutil.cpu_percent(interval=None)
            per_cpu_percent = psutil.cpu_percent(interval=None, percpu=True)
            
            # 负载平均值（仅Linux/macOS）
            load_average = None
            if hasattr(os, 'getloadavg'):
                try:
                    load_average = list(os.getloadavg())
                except OSError:
                    pass
            
            # CPU统计信息
            cpu_stats = psutil.cpu_stats()
            context_switches = cpu_stats.ctx_switches
            interrupts = cpu_stats.interrupts
            
            # CPU频率
            cpu_freq = psutil.cpu_freq()
            frequency = cpu_freq.current if cpu_freq else 0.0
            
            return CPUMetrics(
                timestamp=timestamp,
                total_percent=total_percent,
                per_cpu_percent=per_cpu_percent,
                load_average=load_average,
                context_switches=context_switches,
                interrupts=interrupts,
                frequency=frequency
            )
            
        except Exception as e:
            logger.error(f"获取CPU指标失败: {e}")
            return CPUMetrics(
                timestamp=timestamp,
                total_percent=0.0,
                per_cpu_percent=[],
                load_average=None,
                context_switches=0,
                interrupts=0,
                frequency=0.0
            )
    
    def _get_memory_metrics(self, timestamp: datetime) -> MemoryMetrics:
        """获取内存指标"""
        try:
            # 虚拟内存
            virtual_memory = psutil.virtual_memory()
            
            # 交换空间
            swap_memory = psutil.swap_memory()
            
            return MemoryMetrics(
                timestamp=timestamp,
                total=virtual_memory.total,
                available=virtual_memory.available,
                used=virtual_memory.used,
                percent=virtual_memory.percent,
                swap_total=swap_memory.total,
                swap_used=swap_memory.used,
                swap_percent=swap_memory.percent,
                buffers=getattr(virtual_memory, 'buffers', 0),
                cached=getattr(virtual_memory, 'cached', 0)
            )
            
        except Exception as e:
            logger.error(f"获取内存指标失败: {e}")
            return MemoryMetrics(
                timestamp=timestamp,
                total=0, available=0, used=0, percent=0.0,
                swap_total=0, swap_used=0, swap_percent=0.0,
                buffers=0, cached=0
            )
    
    def _get_disk_metrics(self, timestamp: datetime) -> DiskMetrics:
        """获取磁盘I/O指标"""
        try:
            # 磁盘I/O统计
            disk_io = psutil.disk_io_counters()
            if disk_io is None:
                disk_io = psutil.disk_io_counters(perdisk=False)
            
            # 磁盘使用情况
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.device] = {
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': (usage.used / usage.total) * 100 if usage.total > 0 else 0
                    }
                except (PermissionError, OSError):
                    continue
            
            return DiskMetrics(
                timestamp=timestamp,
                read_bytes=disk_io.read_bytes if disk_io else 0,
                write_bytes=disk_io.write_bytes if disk_io else 0,
                read_count=disk_io.read_count if disk_io else 0,
                write_count=disk_io.write_count if disk_io else 0,
                read_time=disk_io.read_time if disk_io else 0,
                write_time=disk_io.write_time if disk_io else 0,
                disk_usage=disk_usage
            )
            
        except Exception as e:
            logger.error(f"获取磁盘指标失败: {e}")
            return DiskMetrics(
                timestamp=timestamp,
                read_bytes=0, write_bytes=0, read_count=0, write_count=0,
                read_time=0, write_time=0, disk_usage={}
            )
    
    def _get_network_metrics(self, timestamp: datetime) -> NetworkMetrics:
        """获取网络I/O指标"""
        try:
            # 网络I/O统计
            net_io = psutil.net_io_counters()
            
            # 网络连接数
            connections = len(psutil.net_connections())
            
            return NetworkMetrics(
                timestamp=timestamp,
                bytes_sent=net_io.bytes_sent if net_io else 0,
                bytes_recv=net_io.bytes_recv if net_io else 0,
                packets_sent=net_io.packets_sent if net_io else 0,
                packets_recv=net_io.packets_recv if net_io else 0,
                errin=net_io.errin if net_io else 0,
                errout=net_io.errout if net_io else 0,
                dropin=net_io.dropin if net_io else 0,
                dropout=net_io.dropout if net_io else 0,
                connections=connections
            )
            
        except Exception as e:
            logger.error(f"获取网络指标失败: {e}")
            return NetworkMetrics(
                timestamp=timestamp,
                bytes_sent=0, bytes_recv=0, packets_sent=0, packets_recv=0,
                errin=0, errout=0, dropin=0, dropout=0, connections=0
            )
    
    def _get_gpu_metrics(self, timestamp: datetime) -> Optional[GPUMetrics]:
        """获取GPU指标"""
        if not GPU_AVAILABLE:
            return None
        
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return None
            
            gpu_utilization = [gpu.load * 100 for gpu in gpus]
            memory_total = [gpu.memoryTotal for gpu in gpus]
            memory_used = [gpu.memoryUsed for gpu in gpus]
            memory_percent = [gpu.memoryUtil * 100 for gpu in gpus]
            temperature = [gpu.temperature for gpu in gpus]
            
            return GPUMetrics(
                timestamp=timestamp,
                gpu_count=len(gpus),
                gpu_utilization=gpu_utilization,
                memory_total=memory_total,
                memory_used=memory_used,
                memory_percent=memory_percent,
                temperature=temperature
            )
            
        except Exception as e:
            logger.error(f"获取GPU指标失败: {e}")
            return None
    
    def _get_process_metrics(self, timestamp: datetime) -> ProcessMetrics:
        """获取进程指标"""
        try:
            # 进程基本信息
            pid = self.target_process.pid
            name = self.target_process.name()
            
            # CPU和内存使用率
            cpu_percent = self.target_process.cpu_percent()
            memory_percent = self.target_process.memory_percent()
            
            # 内存信息
            memory_info = self.target_process.memory_info()
            memory_rss = memory_info.rss
            memory_vms = memory_info.vms
            
            # 线程和文件描述符
            num_threads = self.target_process.num_threads()
            try:
                num_fds = self.target_process.num_fds()
            except (AttributeError, OSError):
                num_fds = 0  # Windows不支持
            
            # I/O信息
            try:
                io_counters = self.target_process.io_counters()
                io_read_bytes = io_counters.read_bytes
                io_write_bytes = io_counters.write_bytes
            except (AttributeError, OSError):
                io_read_bytes = 0
                io_write_bytes = 0
            
            return ProcessMetrics(
                timestamp=timestamp,
                pid=pid,
                name=name,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_rss=memory_rss,
                memory_vms=memory_vms,
                num_threads=num_threads,
                num_fds=num_fds,
                io_read_bytes=io_read_bytes,
                io_write_bytes=io_write_bytes
            )
            
        except Exception as e:
            logger.error(f"获取进程指标失败: {e}")
            return ProcessMetrics(
                timestamp=timestamp,
                pid=0, name="unknown", cpu_percent=0.0, memory_percent=0.0,
                memory_rss=0, memory_vms=0, num_threads=0, num_fds=0,
                io_read_bytes=0, io_write_bytes=0
            )
    
    def _check_thresholds(self, snapshot: SystemResourceSnapshot) -> None:
        """检查资源阈值"""
        alerts = []
        
        # 检查CPU阈值
        if snapshot.cpu_metrics.total_percent >= self.thresholds.cpu_critical:
            alerts.append(("CPU_CRITICAL", {
                'usage': snapshot.cpu_metrics.total_percent,
                'threshold': self.thresholds.cpu_critical
            }))
        elif snapshot.cpu_metrics.total_percent >= self.thresholds.cpu_warning:
            alerts.append(("CPU_WARNING", {
                'usage': snapshot.cpu_metrics.total_percent,
                'threshold': self.thresholds.cpu_warning
            }))
        
        # 检查内存阈值
        if snapshot.memory_metrics.percent >= self.thresholds.memory_critical:
            alerts.append(("MEMORY_CRITICAL", {
                'usage': snapshot.memory_metrics.percent,
                'threshold': self.thresholds.memory_critical
            }))
        elif snapshot.memory_metrics.percent >= self.thresholds.memory_warning:
            alerts.append(("MEMORY_WARNING", {
                'usage': snapshot.memory_metrics.percent,
                'threshold': self.thresholds.memory_warning
            }))
        
        # 检查磁盘阈值
        for device, usage_info in snapshot.disk_metrics.disk_usage.items():
            usage_percent = usage_info['percent']
            if usage_percent >= self.thresholds.disk_critical:
                alerts.append(("DISK_CRITICAL", {
                    'device': device,
                    'usage': usage_percent,
                    'threshold': self.thresholds.disk_critical
                }))
            elif usage_percent >= self.thresholds.disk_warning:
                alerts.append(("DISK_WARNING", {
                    'device': device,
                    'usage': usage_percent,
                    'threshold': self.thresholds.disk_warning
                }))
        
        # 检查GPU阈值
        if snapshot.gpu_metrics:
            for i, usage in enumerate(snapshot.gpu_metrics.gpu_utilization):
                if usage >= self.thresholds.gpu_critical:
                    alerts.append(("GPU_CRITICAL", {
                        'gpu_id': i,
                        'usage': usage,
                        'threshold': self.thresholds.gpu_critical
                    }))
                elif usage >= self.thresholds.gpu_warning:
                    alerts.append(("GPU_WARNING", {
                        'gpu_id': i,
                        'usage': usage,
                        'threshold': self.thresholds.gpu_warning
                    }))
        
        # 触发告警回调
        for alert_type, alert_data in alerts:
            self.alert_count += 1
            self._notify_alert_callbacks(alert_type, alert_data)
    
    def _notify_alert_callbacks(self, alert_type: str, alert_data: Dict[str, Any]) -> None:
        """通知告警回调"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")
    
    def get_current_snapshot(self) -> Optional[SystemResourceSnapshot]:
        """获取当前资源快照"""
        if not self.snapshots:
            return None
        return self.snapshots[-1]
    
    def get_current_stats(self) -> Dict[str, Any]:
        """
        获取当前系统资源统计信息
        
        实时获取系统当前的CPU、内存、磁盘、网络等资源使用状态，
        包含内存优化策略以减少数据收集过程中的内存占用。
        
        返回:
            Dict[str, Any]: 包含当前系统资源状态的字典
            
        内存优化特性:
            - 使用生成器减少内存占用
            - 及时清理临时数据
            - 避免大量数据缓存
        """
        import gc
        
        try:
            current_time = datetime.now()
            
            # 获取当前快照（不存储到历史记录中）
            current_snapshot = self._take_snapshot()
            
            # CPU信息
            cpu_info = {
                "usage_percent": current_snapshot.cpu_metrics.total_percent,
                "per_cpu_percent": current_snapshot.cpu_metrics.per_cpu_percent,
                "load_average": current_snapshot.cpu_metrics.load_average,
                "frequency_mhz": current_snapshot.cpu_metrics.frequency,
                "context_switches": current_snapshot.cpu_metrics.context_switches,
                "interrupts": current_snapshot.cpu_metrics.interrupts,
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True)
            }
            
            # 内存信息
            memory_info = {
                "total_mb": current_snapshot.memory_metrics.total / 1024 / 1024,
                "available_mb": current_snapshot.memory_metrics.available / 1024 / 1024,
                "used_mb": current_snapshot.memory_metrics.used / 1024 / 1024,
                "usage_percent": current_snapshot.memory_metrics.percent,
                "swap_total_mb": current_snapshot.memory_metrics.swap_total / 1024 / 1024,
                "swap_used_mb": current_snapshot.memory_metrics.swap_used / 1024 / 1024,
                "swap_percent": current_snapshot.memory_metrics.swap_percent,
                "buffers_mb": current_snapshot.memory_metrics.buffers / 1024 / 1024,
                "cached_mb": current_snapshot.memory_metrics.cached / 1024 / 1024
            }
            
            # 磁盘信息
            disk_info = {
                "read_mb": current_snapshot.disk_metrics.read_bytes / 1024 / 1024,
                "write_mb": current_snapshot.disk_metrics.write_bytes / 1024 / 1024,
                "read_count": current_snapshot.disk_metrics.read_count,
                "write_count": current_snapshot.disk_metrics.write_count,
                "read_time_ms": current_snapshot.disk_metrics.read_time,
                "write_time_ms": current_snapshot.disk_metrics.write_time,
                "disk_usage": current_snapshot.disk_metrics.disk_usage
            }
            
            # 网络信息
            network_info = {
                "bytes_sent_mb": current_snapshot.network_metrics.bytes_sent / 1024 / 1024,
                "bytes_recv_mb": current_snapshot.network_metrics.bytes_recv / 1024 / 1024,
                "packets_sent": current_snapshot.network_metrics.packets_sent,
                "packets_recv": current_snapshot.network_metrics.packets_recv,
                "errors_in": current_snapshot.network_metrics.errin,
                "errors_out": current_snapshot.network_metrics.errout,
                "drops_in": current_snapshot.network_metrics.dropin,
                "drops_out": current_snapshot.network_metrics.dropout,
                "connections": current_snapshot.network_metrics.connections
            }
            
            # 进程信息
            process_info = {
                "pid": current_snapshot.process_metrics.pid,
                "name": current_snapshot.process_metrics.name,
                "cpu_percent": current_snapshot.process_metrics.cpu_percent,
                "memory_percent": current_snapshot.process_metrics.memory_percent,
                "memory_rss_mb": current_snapshot.process_metrics.memory_rss / 1024 / 1024,
                "memory_vms_mb": current_snapshot.process_metrics.memory_vms / 1024 / 1024,
                "num_threads": current_snapshot.process_metrics.num_threads,
                "num_fds": current_snapshot.process_metrics.num_fds,
                "io_read_mb": current_snapshot.process_metrics.io_read_bytes / 1024 / 1024,
                "io_write_mb": current_snapshot.process_metrics.io_write_bytes / 1024 / 1024
            }
            
            # GPU信息（如果可用）
            gpu_info = None
            if current_snapshot.gpu_metrics:
                gpu_info = {
                    "gpu_count": current_snapshot.gpu_metrics.gpu_count,
                    "utilization_percent": current_snapshot.gpu_metrics.gpu_utilization,
                    "memory_total_mb": current_snapshot.gpu_metrics.memory_total,
                    "memory_used_mb": current_snapshot.gpu_metrics.memory_used,
                    "memory_percent": current_snapshot.gpu_metrics.memory_percent,
                    "temperature_celsius": current_snapshot.gpu_metrics.temperature
                }
            
            # 系统信息
            system_info = {
                "platform": platform.platform(),
                "architecture": platform.architecture()[0],
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
            
            # 监控状态信息
            monitoring_info = {
                "is_monitoring": self.is_monitoring,
                "total_snapshots_collected": self.total_snapshots,
                "current_snapshots_in_memory": len(self.snapshots),
                "monitoring_interval": self.monitoring_interval,
                "max_snapshots": self.max_snapshots
            }
            
            # 内存使用警告
            memory_warnings = []
            if memory_info["usage_percent"] > 90:
                memory_warnings.append("系统内存使用率超过90%")
            if memory_info["swap_percent"] > 50:
                memory_warnings.append("交换空间使用率超过50%")
            
            # 清理临时变量以释放内存
            del current_snapshot
            gc.collect()
            
            return {
                "timestamp": current_time.isoformat(),
                "cpu": cpu_info,
                "memory": memory_info,
                "disk": disk_info,
                "network": network_info,
                "process": process_info,
                "gpu": gpu_info,
                "system": system_info,
                "monitoring": monitoring_info,
                "warnings": memory_warnings
            }
            
        except Exception as e:
            logger.error(f"获取当前系统资源统计信息失败: {e}")
            return {
                "error": f"获取系统资源信息失败: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def get_resource_statistics(self) -> Dict[str, Any]:
        """获取资源统计信息"""
        if not self.snapshots:
            return {
                'snapshot_count': 0,
                'monitoring_duration': 0,
                'cpu': {},
                'memory': {},
                'disk': {},
                'network': {},
                'process': {}
            }
        
        # 提取各类指标数据
        cpu_data = [s.cpu_metrics.total_percent for s in self.snapshots]
        memory_data = [s.memory_metrics.percent for s in self.snapshots]
        disk_read_data = [s.disk_metrics.read_bytes for s in self.snapshots]
        disk_write_data = [s.disk_metrics.write_bytes for s in self.snapshots]
        network_sent_data = [s.network_metrics.bytes_sent for s in self.snapshots]
        network_recv_data = [s.network_metrics.bytes_recv for s in self.snapshots]
        process_cpu_data = [s.process_metrics.cpu_percent for s in self.snapshots if s.process_metrics]
        process_memory_data = [s.process_metrics.memory_percent for s in self.snapshots if s.process_metrics]
        
        return {
            'snapshot_count': len(self.snapshots),
            'total_snapshots': self.total_snapshots,
            'alert_count': self.alert_count,
            'monitoring_duration': (
                (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds()
                if len(self.snapshots) > 1 else 0
            ),
            'cpu': {
                'current_percent': cpu_data[-1] if cpu_data else 0,
                'average_percent': statistics.mean(cpu_data) if cpu_data else 0,
                'peak_percent': max(cpu_data) if cpu_data else 0,
                'min_percent': min(cpu_data) if cpu_data else 0,
                'std_percent': statistics.stdev(cpu_data) if len(cpu_data) > 1 else 0
            },
            'memory': {
                'current_percent': memory_data[-1] if memory_data else 0,
                'average_percent': statistics.mean(memory_data) if memory_data else 0,
                'peak_percent': max(memory_data) if memory_data else 0,
                'min_percent': min(memory_data) if memory_data else 0,
                'std_percent': statistics.stdev(memory_data) if len(memory_data) > 1 else 0
            },
            'disk': {
                'total_read_bytes': disk_read_data[-1] if disk_read_data else 0,
                'total_write_bytes': disk_write_data[-1] if disk_write_data else 0,
                'average_read_rate': statistics.mean(disk_read_data) if disk_read_data else 0,
                'average_write_rate': statistics.mean(disk_write_data) if disk_write_data else 0
            },
            'network': {
                'total_sent_bytes': network_sent_data[-1] if network_sent_data else 0,
                'total_recv_bytes': network_recv_data[-1] if network_recv_data else 0,
                'average_sent_rate': statistics.mean(network_sent_data) if network_sent_data else 0,
                'average_recv_rate': statistics.mean(network_recv_data) if network_recv_data else 0
            },
            'process': {
                'average_cpu_percent': statistics.mean(process_cpu_data) if process_cpu_data else 0,
                'peak_cpu_percent': max(process_cpu_data) if process_cpu_data else 0,
                'average_memory_percent': statistics.mean(process_memory_data) if process_memory_data else 0,
                'peak_memory_percent': max(process_memory_data) if process_memory_data else 0
            }
        }
    
    def export_snapshots(self, filepath: str) -> None:
        """导出资源快照数据"""
        data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_snapshots': len(self.snapshots),
                'monitoring_interval': self.monitoring_interval,
                'target_process': self.target_process.pid,
                'system_info': {
                    'platform': platform.platform(),
                    'cpu_count': psutil.cpu_count(),
                    'memory_total': psutil.virtual_memory().total
                }
            },
            'snapshots': [snapshot.to_dict() for snapshot in self.snapshots]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"资源快照数据已导出到: {filepath}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
    
    async def monitor_resources_async(
        self,
        duration: float = 60.0,
        interval: float = 1.0
    ) -> List[SystemResourceSnapshot]:
        """
        异步资源监控方法
        
        参数:
            duration: 监控持续时间（秒）
            interval: 监控间隔（秒）
        
        返回:
            资源快照列表
        """
        snapshots = []
        start_time = time.time()
        
        logger.info(f"开始异步资源监控，持续时间: {duration}秒")
        
        while time.time() - start_time < duration:
            try:
                snapshot = self._take_snapshot()
                snapshots.append(snapshot)
                
                # 检查资源阈值
                self._check_thresholds(snapshot)
                
                # 异步等待
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"异步资源监控出错: {e}")
                await asyncio.sleep(interval)
        
        logger.info(f"异步资源监控完成，共收集 {len(snapshots)} 个快照")
        return snapshots


# 便捷函数
def monitor_resource_usage(
    test_function: Callable,
    duration: float = 60.0,
    monitoring_interval: float = 1.0,
    thresholds: Optional[ResourceThresholds] = None
) -> Dict[str, Any]:
    """
    便捷的资源使用监控函数
    
    参数:
        test_function: 要测试的函数
        duration: 测试持续时间（秒）
        monitoring_interval: 监控间隔（秒）
        thresholds: 资源阈值配置
    
    返回:
        资源使用统计信息
    """
    monitor = ResourceUtilizationMonitor(
        monitoring_interval=monitoring_interval,
        thresholds=thresholds
    )
    
    function_result = None
    with monitor:
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                function_result = test_function()
                time.sleep(0.1)  # 短暂休息
            except Exception as e:
                logger.error(f"测试函数执行出错: {e}")
                break
    
    stats = monitor.get_resource_statistics()
    stats['function_result'] = function_result
    return stats


if __name__ == "__main__":
    # 示例使用
    def example_test_function():
        """示例测试函数"""
        # 模拟一些计算密集型操作
        result = sum(i * i for i in range(10000))
        return result
    
    # 运行资源监控
    stats = monitor_resource_usage(example_test_function, duration=30.0)
    print(f"资源使用统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")