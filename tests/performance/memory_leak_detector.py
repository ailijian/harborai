"""
内存泄漏检测器模块

提供内存泄漏检测、分析和监控功能。
"""

import gc
import time
import threading
import tracemalloc
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import psutil


class SnapshotList:
    """支持自动限制的快照列表包装器"""
    
    def __init__(self, snapshots: List, max_snapshots: int):
        self._snapshots = snapshots
        self._max_snapshots = max_snapshots
    
    def append(self, item):
        """添加快照并自动限制数量"""
        self._snapshots.append(item)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots.pop(0)
    
    def __len__(self):
        return len(self._snapshots)
    
    def __getitem__(self, index):
        return self._snapshots[index]
    
    def __setitem__(self, index, value):
        self._snapshots[index] = value
    
    def __iter__(self):
        return iter(self._snapshots)
    
    def __repr__(self):
        return repr(self._snapshots)
    
    def pop(self, index=-1):
        return self._snapshots.pop(index)
    
    def clear(self):
        self._snapshots.clear()


@dataclass
class MemorySnapshot:
    """内存快照数据类"""
    timestamp: datetime
    rss_memory: float  # RSS内存使用量（字节）
    vms_memory: float  # VMS内存使用量（字节）
    heap_memory: float  # 堆内存使用量（字节）
    gc_count: Dict[int, int]  # GC计数
    object_count: int  # 对象数量
    tracemalloc_peak: float  # tracemalloc峰值（字节）
    memory_percent: float  # 内存使用百分比
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'rss_memory': self.rss_memory,
            'vms_memory': self.vms_memory,
            'heap_memory': self.heap_memory,
            'gc_count': self.gc_count,
            'object_count': self.object_count,
            'tracemalloc_peak': self.tracemalloc_peak,
            'memory_percent': self.memory_percent
        }


@dataclass
class MemoryLeakAnalysis:
    """内存泄漏分析结果"""
    is_leak_detected: bool
    leak_rate: float  # MB/s
    confidence_level: float  # 0-1
    trend_analysis: str
    recommendations: List[str]
    peak_memory: int
    average_memory: float
    memory_growth: float
    gc_efficiency: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'is_leak_detected': self.is_leak_detected,
            'leak_rate': self.leak_rate,
            'confidence_level': self.confidence_level,
            'trend_analysis': self.trend_analysis,
            'recommendations': self.recommendations,
            'peak_memory': self.peak_memory,
            'average_memory': self.average_memory,
            'memory_growth': self.memory_growth,
            'gc_efficiency': self.gc_efficiency
        }


class MemoryLeakDetector:
    """内存泄漏检测器"""
    
    def __init__(self, monitoring_interval: float = 1.0, max_snapshots: int = 100, 
                 leak_threshold: int = 1024, confidence_threshold: float = 0.8):
        """
        初始化内存泄漏检测器
        
        Args:
            monitoring_interval: 监控间隔（秒）
            max_snapshots: 最大快照数量
            leak_threshold: 泄漏阈值（字节）
            confidence_threshold: 置信度阈值
        """
        if monitoring_interval <= 0:
            raise ValueError("监控间隔必须大于0")
            
        self.monitoring_interval = monitoring_interval
        self.max_snapshots = max_snapshots
        self.leak_threshold = leak_threshold
        self.confidence_threshold = confidence_threshold
        self._snapshots: List[MemorySnapshot] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.leak_callbacks: List[Callable[[MemoryLeakAnalysis], None]] = []
        self.process = psutil.Process()
    
    @property
    def snapshots(self):
        """获取快照列表的包装器，支持自动限制"""
        return SnapshotList(self._snapshots, self.max_snapshots)
        
    def add_leak_callback(self, callback: Callable[[MemoryLeakAnalysis], None]):
        """添加内存泄漏回调函数"""
        self.leak_callbacks.append(callback)
    
    def _notify_leak_callbacks(self, analysis: MemoryLeakAnalysis):
        """通知所有注册的回调函数"""
        for callback in self.leak_callbacks:
            try:
                callback(analysis)
            except Exception as e:
                # 记录回调错误但不中断程序
                print(f"回调函数执行错误: {e}")
        
    def _take_snapshot(self) -> MemorySnapshot:
        """内部方法：获取内存快照"""
        return self.take_snapshot()
    
    def take_snapshot(self) -> MemorySnapshot:
        """获取当前内存快照"""
        memory_info = self.process.memory_info()
        
        # 获取GC计数
        gc_count_tuple = gc.get_count()
        gc_count = {i: count for i, count in enumerate(gc_count_tuple)}
        
        # 获取tracemalloc峰值
        tracemalloc_peak = 0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_peak = peak
            
        # 估算对象数量（简化版本）
        object_count = len(gc.get_objects())
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            rss_memory=memory_info.rss,
            vms_memory=memory_info.vms,
            heap_memory=tracemalloc_peak,
            gc_count=gc_count,
            object_count=object_count,
            tracemalloc_peak=tracemalloc_peak,
            memory_percent=self.process.memory_percent()
        )
        
        return snapshot
        
    def start_monitoring(self):
        """开始监控内存"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控内存"""
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                snapshot = self.take_snapshot()
                self._snapshots.append(snapshot)
                
                # 限制快照数量
                if len(self._snapshots) > self.max_snapshots:
                    self._snapshots.pop(0)
                    
            except Exception as e:
                print(f"监控过程中发生错误: {e}")
                
            time.sleep(self.monitoring_interval)
            
    def _calculate_leak_rate(self, memory_data: List[float], timestamps: List[datetime]) -> tuple[float, float]:
        """内部方法：计算泄漏率和置信度"""
        if len(memory_data) < 2 or len(timestamps) < 2:
            return 0.0, 0.0
        
        # 计算时间差（秒）
        time_diff = (timestamps[-1] - timestamps[0]).total_seconds()
        if time_diff <= 0:
            return 0.0, 0.0
        
        # 计算内存变化（MB/s）
        memory_diff = memory_data[-1] - memory_data[0]
        leak_rate = memory_diff / time_diff
        
        # 计算置信度（基于数据点数量和趋势一致性）
        confidence = min(1.0, len(memory_data) / 5.0)
        
        return leak_rate, confidence
    
    def calculate_leak_rate(self) -> float:
        """计算内存泄漏率"""
        if len(self.snapshots) < 2:
            return 0.0
            
        # 使用最近的快照计算趋势
        recent_snapshots = self.snapshots[-10:]
        if len(recent_snapshots) < 2:
            return 0.0
            
        first = recent_snapshots[0]
        last = recent_snapshots[-1]
        
        time_diff = (last.timestamp - first.timestamp).total_seconds()
        if time_diff <= 0:
            return 0.0
            
        # 使用RSS内存计算泄漏率（转换为MB/s）
        memory_diff = (last.rss_memory - first.rss_memory) / 1024 / 1024
        return memory_diff / time_diff
        
    def _analyze_gc_efficiency(self) -> float:
        """内部方法：分析GC效率"""
        result = self.analyze_gc_efficiency()
        return result.get('efficiency', 0.0)
    
    def analyze_gc_efficiency(self) -> Dict[str, Any]:
        """分析垃圾回收效率"""
        if not self.snapshots:
            return {'efficiency': 0.0}
            
        latest = self.snapshots[-1]
        
        # 简单的效率计算：基于GC次数和对象数量
        total_gc = sum(latest.gc_count.values())
        efficiency = min(1.0, 1.0 - (total_gc / max(latest.object_count, 1)) * 100)
        efficiency = max(0.0, efficiency)
        
        return {
            'gc_count': latest.gc_count,
            'object_count': latest.object_count,
            'efficiency': efficiency
        }
        
    def _generate_trend_analysis(self, memory_data: List[float], leak_rate: float) -> str:
        """内部方法：生成趋势分析"""
        if not memory_data:
            return "无数据"
        
        if abs(leak_rate) < 10:
            return "稳定"
        elif leak_rate > 0:
            return "增长"
        else:
            return "下降"
    
    def generate_trend_analysis(self) -> Dict[str, Any]:
        """生成趋势分析"""
        if len(self.snapshots) < 3:
            return {}
            
        # 使用RSS内存进行趋势分析（转换为MB）
        memory_values = [s.rss_memory / 1024 / 1024 for s in self.snapshots]
        
        # 简单的趋势分析
        increasing_trend = 0
        for i in range(1, len(memory_values)):
            if memory_values[i] > memory_values[i-1]:
                increasing_trend += 1
                
        trend_ratio = increasing_trend / (len(memory_values) - 1)
        
        return {
            'trend_ratio': trend_ratio,
            'memory_range': {
                'min': min(memory_values),
                'max': max(memory_values),
                'avg': sum(memory_values) / len(memory_values)
            }
        }
        
    def _generate_recommendations(self, is_leak_detected: bool, leak_rate: float, gc_efficiency: float) -> List[str]:
        """内部方法：生成优化建议"""
        recommendations = []
        
        if is_leak_detected:
            if leak_rate > 1024:  # 每秒泄漏超过1MB
                recommendations.append("检测到严重内存泄漏，建议立即排查")
            else:
                recommendations.append("检测到轻微内存泄漏，建议监控")
        
        if gc_efficiency < 0.7:
            recommendations.append("垃圾回收效率较低，建议优化对象生命周期")
        
        if not recommendations:
            recommendations.append("内存使用正常，继续监控")
        
        return recommendations
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        leak_rate = analysis.get('leak_rate', 0)
        if leak_rate > 1.0:  # 每秒泄漏超过1MB
            recommendations.append("检测到严重内存泄漏，建议立即排查")
        elif leak_rate > 0.1:  # 每秒泄漏超过0.1MB
            recommendations.append("检测到轻微内存泄漏，建议监控")
            
        trend = analysis.get('trend_analysis', {})
        if trend.get('trend_ratio', 0) > 0.8:
            recommendations.append("内存使用呈上升趋势，建议检查对象生命周期")
            
        return recommendations
        
    def _analyze_memory_leak(self) -> MemoryLeakAnalysis:
        """内部方法：分析内存泄漏"""
        return self.analyze_leak()
    
    def analyze_leak(self) -> MemoryLeakAnalysis:
        """分析内存泄漏"""
        leak_rate = self.calculate_leak_rate()
        trend_analysis = self.generate_trend_analysis()
        
        # 简单的泄漏检测逻辑
        is_leak_detected = leak_rate > 0.1  # 每秒泄漏超过0.1MB
        confidence_level = min(1.0, abs(leak_rate) / 1.0)  # 基于泄漏率计算置信度
        
        # 计算统计数据
        if self.snapshots:
            memory_values = [s.rss_memory / 1024 / 1024 for s in self.snapshots]  # 转换为MB
            peak_memory = int(max(memory_values) * 1024)  # 转换为KB
            average_memory = sum(memory_values) / len(memory_values)
            memory_growth = memory_values[-1] - memory_values[0] if len(memory_values) > 1 else 0
        else:
            peak_memory = 0
            average_memory = 0.0
            memory_growth = 0.0
        
        # 计算GC效率
        gc_efficiency = self.analyze_gc_efficiency().get('efficiency', 0.0)
        
        # 生成趋势分析文本
        trend_text = "稳定"
        if trend_analysis.get('trend_ratio', 0) > 0.7:
            trend_text = "上升"
        elif trend_analysis.get('trend_ratio', 0) < 0.3:
            trend_text = "下降"
        
        recommendations = self.generate_recommendations({
            'leak_rate': leak_rate,
            'trend_analysis': trend_analysis
        })
        
        return MemoryLeakAnalysis(
            is_leak_detected=is_leak_detected,
            leak_rate=leak_rate,
            confidence_level=confidence_level,
            trend_analysis=trend_text,
            recommendations=recommendations,
            peak_memory=peak_memory,
            average_memory=average_memory,
            memory_growth=memory_growth,
            gc_efficiency=gc_efficiency
        )
    
    async def detect_memory_leak_async(self, duration: float, interval: float) -> MemoryLeakAnalysis:
        """异步检测内存泄漏"""
        import asyncio
        
        # 开始监控
        self.start_monitoring()
        
        try:
            # 等待指定时间
            await asyncio.sleep(duration)
            
            # 分析结果
            return self.analyze_leak()
        finally:
            # 停止监控
            self.stop_monitoring()
        
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        if not self.snapshots:
            return {
                'snapshot_count': 0,
                'monitoring_duration': 0.0,
                'average_memory': 0.0,
                'peak_memory': 0.0,
                'memory_growth': 0.0,
                'monitoring_active': self.is_monitoring
            }
            
        # 使用RSS内存进行统计（转换为MB）
        memory_values = [s.rss_memory / 1024 / 1024 for s in self.snapshots]
        
        # 计算监控持续时间
        monitoring_duration = 0.0
        if len(self.snapshots) > 1:
            monitoring_duration = (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds()
        
        # 计算内存增长
        memory_growth = 0.0
        if len(memory_values) > 1:
            memory_growth = memory_values[-1] - memory_values[0]
        
        return {
            'snapshot_count': len(self.snapshots),
            'monitoring_duration': monitoring_duration,
            'average_memory': sum(memory_values) / len(memory_values),
            'peak_memory': max(memory_values),
            'memory_growth': memory_growth,
            'monitoring_active': self.is_monitoring
        }
        
    def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        self.snapshots.clear()
        self.leak_callbacks.clear()
        
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()


def detect_memory_leak(test_function: Callable = None, func: Callable = None, 
                      duration: float = 1.0, monitoring_interval: float = 0.1, 
                      *args, **kwargs) -> MemoryLeakAnalysis:
    """
    检测函数执行过程中的内存泄漏
    
    Args:
        test_function: 要检测的函数（新参数名）
        func: 要检测的函数（兼容旧参数名）
        duration: 监控持续时间
        monitoring_interval: 监控间隔
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    Returns:
        内存泄漏分析结果
    """
    # 兼容两种参数名
    target_func = test_function or func
    if target_func is None:
        raise TypeError("detect_memory_leak() missing 1 required positional argument: 'func' or 'test_function'")
    
    detector = MemoryLeakDetector(monitoring_interval=monitoring_interval)
    
    try:
        detector.start_monitoring()
        time.sleep(monitoring_interval * 2)  # 获取基线
        
        # 执行函数
        result = target_func(*args, **kwargs)
        
        time.sleep(duration)  # 等待内存稳定
        
        analysis = detector.analyze_leak()
        return analysis
        
    finally:
        detector.cleanup()