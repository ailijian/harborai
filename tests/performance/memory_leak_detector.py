"""
内存泄漏检测器模块

该模块提供长期内存监控和泄漏检测功能，支持：
- 实时内存使用监控
- 内存泄漏模式识别
- 内存增长趋势分析
- 垃圾回收效率监控
- 内存碎片化检测

作者：HarborAI性能测试团队
创建时间：2024年
"""

import gc
import os
import sys
import time
import psutil
import threading
import tracemalloc
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import statistics
import logging
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """内存快照数据结构"""
    timestamp: datetime
    rss_memory: int  # 常驻内存集大小 (bytes)
    vms_memory: int  # 虚拟内存大小 (bytes)
    heap_memory: int  # 堆内存大小 (bytes)
    gc_count: Dict[int, int]  # 各代垃圾回收次数
    object_count: int  # 对象总数
    tracemalloc_peak: int  # tracemalloc峰值内存
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
    leak_rate: float  # bytes/second
    confidence_level: float  # 0.0-1.0
    trend_analysis: str
    recommendations: List[str]
    peak_memory: int
    average_memory: int
    memory_growth: float  # 内存增长百分比
    gc_efficiency: float  # 垃圾回收效率
    
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
    """
    内存泄漏检测器
    
    功能特性：
    - 持续监控内存使用情况
    - 检测内存泄漏模式
    - 分析内存增长趋势
    - 提供优化建议
    """
    
    def __init__(
        self,
        monitoring_interval: float = 1.0,
        max_snapshots: int = 1000,
        leak_threshold: float = 1024 * 1024,  # 1MB/s
        confidence_threshold: float = 0.8
    ):
        """
        初始化内存泄漏检测器
        
        参数:
            monitoring_interval: 监控间隔（秒）
            max_snapshots: 最大快照数量
            leak_threshold: 泄漏阈值（bytes/s）
            confidence_threshold: 置信度阈值
        """
        # 参数验证
        if monitoring_interval <= 0:
            raise ValueError("监控间隔必须大于0")
        if max_snapshots <= 0:
            raise ValueError("最大快照数量必须大于0")
        if confidence_threshold < 0 or confidence_threshold > 1:
            raise ValueError("置信度阈值必须在0-1之间")
        
        self.monitoring_interval = monitoring_interval
        self.max_snapshots = max_snapshots
        self.leak_threshold = leak_threshold
        self.confidence_threshold = confidence_threshold
        
        # 内存快照存储
        self.snapshots: deque = deque(maxlen=max_snapshots)
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.process = psutil.Process()
        
        # 回调函数
        self.leak_callbacks: List[Callable[[MemoryLeakAnalysis], None]] = []
        
        # 统计信息
        self.total_snapshots = 0
        self.leak_detections = 0
        
        # 启用tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    
    def add_leak_callback(self, callback: Callable[[MemoryLeakAnalysis], None]) -> None:
        """添加泄漏检测回调函数"""
        self.leak_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """开始内存监控"""
        if self.is_monitoring:
            logger.warning("内存监控已在运行中")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryLeakDetector"
        )
        self.monitor_thread.start()
        logger.info("内存泄漏监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止内存监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("内存泄漏监控已停止")
    
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                self.total_snapshots += 1
                
                # 定期分析内存泄漏
                if len(self.snapshots) >= 10:  # 至少需要10个样本
                    analysis = self._analyze_memory_leak()
                    if analysis.is_leak_detected:
                        self.leak_detections += 1
                        self._notify_leak_callbacks(analysis)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"内存监控循环出错: {e}")
                time.sleep(self.monitoring_interval)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """获取内存快照"""
        try:
            # 获取进程内存信息
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # 获取垃圾回收统计
            gc_stats = {}
            for i in range(3):  # Python有3代垃圾回收
                gc_stats[i] = gc.get_count()[i]
            
            # 获取对象数量
            object_count = len(gc.get_objects())
            
            # 获取tracemalloc信息
            current, peak = tracemalloc.get_traced_memory()
            
            return MemorySnapshot(
                timestamp=datetime.now(),
                rss_memory=memory_info.rss,
                vms_memory=memory_info.vms,
                heap_memory=current,
                gc_count=gc_stats,
                object_count=object_count,
                tracemalloc_peak=peak,
                memory_percent=memory_percent
            )
            
        except Exception as e:
            logger.error(f"获取内存快照失败: {e}")
            raise
    
    def _analyze_memory_leak(self) -> MemoryLeakAnalysis:
        """分析内存泄漏"""
        if len(self.snapshots) < 10:
            return MemoryLeakAnalysis(
                is_leak_detected=False,
                leak_rate=0.0,
                confidence_level=0.0,
                trend_analysis="样本数量不足",
                recommendations=[],
                peak_memory=0,
                average_memory=0,
                memory_growth=0.0,
                gc_efficiency=0.0
            )
        
        # 提取内存数据
        memory_data = [s.rss_memory for s in self.snapshots]
        timestamps = [s.timestamp for s in self.snapshots]
        
        # 计算统计信息
        peak_memory = max(memory_data)
        average_memory = statistics.mean(memory_data)
        
        # 计算内存增长率
        if len(memory_data) >= 2:
            initial_memory = memory_data[0]
            final_memory = memory_data[-1]
            memory_growth = ((final_memory - initial_memory) / initial_memory) * 100
        else:
            memory_growth = 0.0
        
        # 计算泄漏率（线性回归）
        leak_rate, confidence = self._calculate_leak_rate(memory_data, timestamps)
        
        # 分析垃圾回收效率
        gc_efficiency = self._analyze_gc_efficiency()
        
        # 判断是否存在泄漏
        is_leak_detected = (
            leak_rate > self.leak_threshold and
            confidence > self.confidence_threshold
        )
        
        # 生成趋势分析
        trend_analysis = self._generate_trend_analysis(memory_data, leak_rate)
        
        # 生成建议
        recommendations = self._generate_recommendations(
            is_leak_detected, leak_rate, gc_efficiency
        )
        
        return MemoryLeakAnalysis(
            is_leak_detected=is_leak_detected,
            leak_rate=leak_rate,
            confidence_level=confidence,
            trend_analysis=trend_analysis,
            recommendations=recommendations,
            peak_memory=peak_memory,
            average_memory=average_memory,
            memory_growth=memory_growth,
            gc_efficiency=gc_efficiency
        )
    
    def _calculate_leak_rate(
        self, 
        memory_data: List[int], 
        timestamps: List[datetime]
    ) -> Tuple[float, float]:
        """计算内存泄漏率和置信度"""
        if len(memory_data) < 2:
            return 0.0, 0.0
        
        # 转换时间戳为秒数
        time_points = [(t - timestamps[0]).total_seconds() for t in timestamps]
        
        # 简单线性回归计算斜率
        n = len(memory_data)
        sum_x = sum(time_points)
        sum_y = sum(memory_data)
        sum_xy = sum(x * y for x, y in zip(time_points, memory_data))
        sum_x2 = sum(x * x for x in time_points)
        
        # 计算斜率（内存增长率 bytes/second）
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0, 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        
        # 计算相关系数作为置信度
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(time_points, memory_data))
        denominator_x = sum((x - mean_x) ** 2 for x in time_points)
        denominator_y = sum((y - mean_y) ** 2 for y in memory_data)
        
        if denominator_x == 0 or denominator_y == 0:
            # 如果数据完全稳定（denominator_y == 0），置信度应该很高
            # 如果时间点相同（denominator_x == 0），置信度为0
            if denominator_y == 0 and denominator_x > 0:
                correlation = 1.0  # 完全稳定的数据，高置信度
            else:
                correlation = 0.0
        else:
            correlation = numerator / ((denominator_x * denominator_y) ** 0.5)
        
        confidence = abs(correlation)
        
        return slope, confidence
    
    def _analyze_gc_efficiency(self) -> float:
        """分析垃圾回收效率"""
        if len(self.snapshots) < 2:
            return 0.0
        
        # 计算垃圾回收前后的内存变化
        recent_snapshots = list(self.snapshots)[-10:]  # 最近10个快照
        
        total_gc_events = 0
        total_memory_freed = 0
        
        for i in range(1, len(recent_snapshots)):
            prev_snapshot = recent_snapshots[i-1]
            curr_snapshot = recent_snapshots[i]
            
            # 检查是否发生了垃圾回收
            gc_occurred = any(
                curr_snapshot.gc_count[gen] > prev_snapshot.gc_count[gen]
                for gen in range(3)
            )
            
            if gc_occurred:
                total_gc_events += 1
                memory_change = prev_snapshot.rss_memory - curr_snapshot.rss_memory
                if memory_change > 0:
                    total_memory_freed += memory_change
        
        if total_gc_events == 0:
            return 0.0
        
        # 计算平均每次GC释放的内存比例
        average_memory = statistics.mean(s.rss_memory for s in recent_snapshots)
        if average_memory == 0:
            return 0.0
        
        efficiency = (total_memory_freed / total_gc_events) / average_memory
        return min(efficiency, 1.0)  # 限制在0-1之间
    
    def _generate_trend_analysis(self, memory_data: List[int], leak_rate: float) -> str:
        """生成趋势分析"""
        if leak_rate > self.leak_threshold:
            return f"检测到内存持续增长，增长率: {leak_rate:.2f} bytes/s"
        elif leak_rate > 0:
            return f"内存轻微增长，增长率: {leak_rate:.2f} bytes/s"
        elif leak_rate < 0:
            return f"内存使用稳定或下降，变化率: {leak_rate:.2f} bytes/s"
        else:
            return "内存使用保持稳定"
    
    def _generate_recommendations(
        self, 
        is_leak_detected: bool, 
        leak_rate: float, 
        gc_efficiency: float
    ) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if is_leak_detected:
            recommendations.append("检测到内存泄漏，建议进行详细的内存分析")
            recommendations.append("使用memory_profiler或pympler进行深度分析")
            recommendations.append("检查是否存在循环引用或未释放的资源")
        
        if leak_rate > self.leak_threshold * 0.5:
            recommendations.append("内存增长较快，建议优化内存使用")
            recommendations.append("检查大对象的生命周期管理")
        
        if gc_efficiency < 0.1:
            recommendations.append("垃圾回收效率较低，建议调整GC参数")
            recommendations.append("考虑手动调用gc.collect()在适当时机")
        
        if not recommendations:
            recommendations.append("内存使用正常，继续监控")
        
        return recommendations
    
    def _notify_leak_callbacks(self, analysis: MemoryLeakAnalysis) -> None:
        """通知泄漏检测回调"""
        for callback in self.leak_callbacks:
            try:
                callback(analysis)
            except Exception as e:
                logger.error(f"泄漏回调执行失败: {e}")
    
    def get_current_analysis(self) -> Optional[MemoryLeakAnalysis]:
        """获取当前内存分析结果"""
        if len(self.snapshots) < 10:
            return None
        return self._analyze_memory_leak()
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        if not self.snapshots:
            return {
                'snapshot_count': 0,
                'total_snapshots': 0,
                'leak_detections': 0,
                'monitoring_duration': 0,
                'average_memory': 0,
                'peak_memory': 0,
                'memory_growth': 0.0
            }
        
        memory_data = [s.rss_memory for s in self.snapshots]
        
        # 计算内存增长
        memory_growth = 0.0
        if len(memory_data) >= 2:
            initial_memory = memory_data[0]
            final_memory = memory_data[-1]
            memory_growth = ((final_memory - initial_memory) / initial_memory) * 100
        
        return {
            'snapshot_count': len(self.snapshots),
            'total_snapshots': self.total_snapshots,
            'leak_detections': self.leak_detections,
            'current_memory': memory_data[-1] if memory_data else 0,
            'peak_memory': max(memory_data) if memory_data else 0,
            'average_memory': statistics.mean(memory_data) if memory_data else 0,
            'memory_std': statistics.stdev(memory_data) if len(memory_data) > 1 else 0,
            'memory_growth': memory_growth,
            'monitoring_duration': (
                (self.snapshots[-1].timestamp - self.snapshots[0].timestamp).total_seconds()
                if len(self.snapshots) > 1 else 0
            )
        }
    
    async def detect_memory_leak_async(
        self,
        duration: float = 60.0,
        interval: float = 1.0
    ) -> MemoryLeakAnalysis:
        """
        异步内存泄漏检测
        
        Args:
            duration: 监控持续时间（秒）
            interval: 采样间隔（秒）
            
        Returns:
            MemoryLeakAnalysis: 内存泄漏分析结果
        """
        import asyncio
        
        logger.info(f"开始异步内存泄漏检测，持续时间: {duration}秒")
        
        # 清空之前的快照
        self.snapshots.clear()
        self.current_analysis = None
        
        # 启用tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        start_time = time.time()
        end_time = start_time + duration
        
        try:
            while time.time() < end_time:
                # 拍摄内存快照
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                
                # 如果快照数量超过限制，移除最旧的
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots.popleft()
                
                # 等待下一次采样
                await asyncio.sleep(interval)
            
            # 分析内存泄漏
            analysis = self._analyze_memory_leak()
            self.current_analysis = analysis
            
            logger.info(f"异步内存泄漏检测完成，检测到泄漏: {analysis.is_leak_detected}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"异步内存泄漏检测失败: {e}")
            raise
        finally:
            # 停止tracemalloc
            if tracemalloc.is_tracing():
                tracemalloc.stop()
    
    def cleanup(self):
        """清理资源"""
        logger.info("开始清理内存泄漏检测器资源")
        
        try:
            # 停止监控
            self.stop_monitoring()
            
            # 清空快照数据
            self.snapshots.clear()
            
            # 停止tracemalloc
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            
            logger.info("内存泄漏检测器资源清理完成")
            
        except Exception as e:
            logger.error(f"清理内存泄漏检测器资源时发生错误: {e}")
    
    def export_snapshots(self, filepath: str) -> None:
        """导出内存快照数据到文件"""
        import json
        
        data = {
            'snapshots': [snapshot.to_dict() for snapshot in self.snapshots],
            'analysis': self.current_analysis.to_dict() if hasattr(self, 'current_analysis') and self.current_analysis else None,
            'export_time': datetime.now().isoformat()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"内存快照数据已导出到: {filepath}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
        return False


# 便捷函数
def detect_memory_leak(
    test_function: Callable,
    duration: float = 60.0,
    monitoring_interval: float = 1.0
) -> MemoryLeakAnalysis:
    """
    便捷的内存泄漏检测函数
    
    参数:
        test_function: 要测试的函数
        duration: 测试持续时间（秒）
        monitoring_interval: 监控间隔（秒）
    
    返回:
        内存泄漏分析结果
    """
    detector = MemoryLeakDetector(monitoring_interval=monitoring_interval)
    
    with detector:
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                test_function()
                time.sleep(0.1)  # 短暂休息
            except Exception as e:
                logger.error(f"测试函数执行出错: {e}")
                break
    
    analysis = detector.get_current_analysis()
    return analysis if analysis else MemoryLeakAnalysis(
        is_leak_detected=False,
        leak_rate=0.0,
        confidence_level=0.0,
        trend_analysis="测试时间不足",
        recommendations=["延长测试时间以获得更准确的结果"],
        peak_memory=0,
        average_memory=0,
        memory_growth=0.0,
        gc_efficiency=0.0
    )


if __name__ == "__main__":
    # 示例使用
    def example_test_function():
        """示例测试函数"""
        # 模拟一些内存操作
        data = [i for i in range(1000)]
        return sum(data)
    
    # 运行内存泄漏检测
    result = detect_memory_leak(example_test_function, duration=30.0)
    print(f"内存泄漏检测结果: {result.to_dict()}")