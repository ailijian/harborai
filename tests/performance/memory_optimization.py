"""
性能测试框架内存优化工具模块

提供内存优化策略、生成器、上下文管理器和内存监控功能，
用于减少性能测试过程中的内存使用率。

作者: AI Assistant
创建时间: 2024
"""

import gc
import sys
import psutil
import logging
import threading
import time
from typing import Iterator, List, Dict, Any, Optional, Callable, Generator, Union
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import weakref
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """内存快照数据类"""
    timestamp: datetime
    rss_mb: float  # 常驻内存（MB）
    vms_mb: float  # 虚拟内存（MB）
    percent: float  # 内存使用百分比
    available_mb: float  # 可用内存（MB）
    gc_count: Dict[str, int]  # 垃圾回收计数


class MemoryMonitor:
    """
    内存监控器
    
    实时监控内存使用情况，提供内存警告和自动清理功能。
    """
    
    def __init__(
        self,
        warning_threshold_mb: float = 500.0,
        critical_threshold_mb: float = 1000.0,
        auto_gc: bool = True,
        monitoring_interval: float = 1.0
    ):
        """
        初始化内存监控器
        
        参数:
            warning_threshold_mb: 内存警告阈值（MB）
            critical_threshold_mb: 内存严重阈值（MB）
            auto_gc: 是否自动垃圾回收
            monitoring_interval: 监控间隔（秒）
        """
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.auto_gc = auto_gc
        self.monitoring_interval = monitoring_interval
        
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.snapshots = deque(maxlen=100)  # 限制快照数量
        self.callbacks: List[Callable[[MemorySnapshot], None]] = []
        
        self.process = psutil.Process()
        self.initial_memory = self._get_current_memory()
        
    def add_callback(self, callback: Callable[[MemorySnapshot], None]) -> None:
        """添加内存状态变化回调"""
        self.callbacks.append(callback)
        
    def start_monitoring(self) -> None:
        """开始内存监控"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("内存监控已启动")
        
    def stop_monitoring(self) -> None:
        """停止内存监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("内存监控已停止")
        
    def _monitoring_loop(self) -> None:
        """监控循环"""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                
                # 检查阈值
                self._check_thresholds(snapshot)
                
                # 通知回调
                for callback in self.callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.error(f"内存监控回调执行失败: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"内存监控循环出错: {e}")
                time.sleep(self.monitoring_interval)
                
    def _take_snapshot(self) -> MemorySnapshot:
        """获取内存快照"""
        memory_info = self.process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=self.process.memory_percent(),
            available_mb=system_memory.available / 1024 / 1024,
            gc_count={
                "gen0": gc.get_count()[0],
                "gen1": gc.get_count()[1],
                "gen2": gc.get_count()[2]
            }
        )
        
    def _check_thresholds(self, snapshot: MemorySnapshot) -> None:
        """检查内存阈值"""
        if snapshot.rss_mb > self.critical_threshold_mb:
            logger.critical(f"内存使用严重超标: {snapshot.rss_mb:.2f} MB")
            if self.auto_gc:
                self.force_gc()
                
        elif snapshot.rss_mb > self.warning_threshold_mb:
            logger.warning(f"内存使用警告: {snapshot.rss_mb:.2f} MB")
            
    def _get_current_memory(self) -> float:
        """获取当前内存使用（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
        
    def get_memory_increase(self) -> float:
        """获取内存增长量（MB）"""
        return self._get_current_memory() - self.initial_memory
        
    def force_gc(self) -> Dict[str, int]:
        """强制垃圾回收"""
        before_count = gc.get_count()
        collected = gc.collect()
        after_count = gc.get_count()
        
        logger.info(f"强制垃圾回收: 回收对象 {collected} 个")
        
        return {
            "collected": collected,
            "before": {"gen0": before_count[0], "gen1": before_count[1], "gen2": before_count[2]},
            "after": {"gen0": after_count[0], "gen1": after_count[1], "gen2": after_count[2]}
        }
        
    def get_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        current_memory = self._get_current_memory()
        
        return {
            "current_memory_mb": current_memory,
            "initial_memory_mb": self.initial_memory,
            "memory_increase_mb": current_memory - self.initial_memory,
            "snapshots_count": len(self.snapshots),
            "is_monitoring": self.is_monitoring,
            "thresholds": {
                "warning_mb": self.warning_threshold_mb,
                "critical_mb": self.critical_threshold_mb
            }
        }


@contextmanager
def memory_optimized_context(
    auto_gc: bool = True,
    gc_threshold: Optional[int] = None,
    monitor: bool = True
):
    """
    内存优化上下文管理器
    
    在代码块执行期间自动进行内存优化和监控。
    
    参数:
        auto_gc: 是否在退出时自动垃圾回收
        gc_threshold: 垃圾回收阈值
        monitor: 是否启用内存监控
    """
    # 记录初始状态
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    initial_gc_count = gc.get_count()
    
    # 设置垃圾回收阈值
    old_threshold = None
    if gc_threshold:
        old_threshold = gc.get_threshold()
        gc.set_threshold(gc_threshold, gc_threshold * 10, gc_threshold * 10)
    
    # 启动内存监控
    memory_monitor = None
    if monitor:
        memory_monitor = MemoryMonitor()
        memory_monitor.start_monitoring()
    
    try:
        logger.info(f"进入内存优化上下文，初始内存: {initial_memory:.2f} MB")
        yield memory_monitor
        
    finally:
        # 停止监控
        if memory_monitor:
            memory_monitor.stop_monitoring()
        
        # 恢复垃圾回收阈值
        if old_threshold:
            gc.set_threshold(*old_threshold)
        
        # 自动垃圾回收
        if auto_gc:
            collected = gc.collect()
            logger.info(f"上下文退出时垃圾回收: {collected} 个对象")
        
        # 记录最终状态
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        logger.info(f"退出内存优化上下文，最终内存: {final_memory:.2f} MB，"
                   f"增长: {memory_increase:.2f} MB")


def batch_generator(
    data: List[Any],
    batch_size: int = 100,
    auto_cleanup: bool = True
) -> Generator[List[Any], None, None]:
    """
    批处理生成器
    
    将大数据集分批处理，减少内存占用。
    
    参数:
        data: 要处理的数据列表
        batch_size: 批处理大小
        auto_cleanup: 是否自动清理已处理的数据
        
    生成:
        每批数据的列表
    """
    total_batches = (len(data) + batch_size - 1) // batch_size
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        logger.debug(f"处理批次 {batch_num}/{total_batches}，大小: {len(batch)}")
        
        yield batch
        
        # 自动清理
        if auto_cleanup:
            del batch
            if batch_num % 10 == 0:  # 每10批强制垃圾回收
                gc.collect()


def memory_efficient_iterator(
    items: List[Any],
    chunk_size: int = 1000,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Iterator[Any]:
    """
    内存高效迭代器
    
    逐个处理大数据集中的项目，避免一次性加载所有数据。
    
    参数:
        items: 要迭代的项目列表
        chunk_size: 块大小，用于定期清理
        progress_callback: 进度回调函数
        
    生成:
        单个数据项
    """
    total_items = len(items)
    
    for i, item in enumerate(items):
        yield item
        
        # 进度回调
        if progress_callback and (i + 1) % 100 == 0:
            progress_callback(i + 1, total_items)
        
        # 定期清理
        if (i + 1) % chunk_size == 0:
            gc.collect()


class MemoryEfficientDataCollector:
    """
    内存高效数据收集器
    
    使用弱引用和定期清理策略收集数据，避免内存泄漏。
    """
    
    def __init__(
        self,
        max_items: int = 10000,
        auto_cleanup_interval: int = 1000,
        use_weak_refs: bool = True
    ):
        """
        初始化数据收集器
        
        参数:
            max_items: 最大项目数
            auto_cleanup_interval: 自动清理间隔
            use_weak_refs: 是否使用弱引用
        """
        self.max_items = max_items
        self.auto_cleanup_interval = auto_cleanup_interval
        self.use_weak_refs = use_weak_refs
        
        self.items = deque(maxlen=max_items)
        self.weak_refs = weakref.WeakSet() if use_weak_refs else None
        self.item_count = 0
        
    def add_item(self, item: Any) -> None:
        """添加数据项"""
        self.items.append(item)
        
        if self.weak_refs is not None:
            try:
                self.weak_refs.add(item)
            except TypeError:
                # 某些类型不支持弱引用
                pass
        
        self.item_count += 1
        
        # 定期清理
        if self.item_count % self.auto_cleanup_interval == 0:
            self.cleanup()
    
    def get_items(self) -> List[Any]:
        """获取所有项目"""
        return list(self.items)
    
    def cleanup(self) -> int:
        """清理无效引用和过期数据"""
        initial_count = len(self.items)
        
        # 强制垃圾回收
        collected = gc.collect()
        
        final_count = len(self.items)
        cleaned = initial_count - final_count
        
        logger.debug(f"数据收集器清理: 移除 {cleaned} 项，垃圾回收 {collected} 对象")
        
        return cleaned
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_items": len(self.items),
            "max_items": self.max_items,
            "items_added": self.item_count,
            "weak_refs_count": len(self.weak_refs) if self.weak_refs else 0,
            "memory_usage_mb": sys.getsizeof(self.items) / 1024 / 1024
        }


def optimize_memory_usage(
    func: Callable,
    *args,
    gc_before: bool = True,
    gc_after: bool = True,
    monitor: bool = True,
    **kwargs
) -> Any:
    """
    内存优化装饰器函数
    
    在函数执行前后进行内存优化操作。
    
    参数:
        func: 要执行的函数
        gc_before: 执行前是否垃圾回收
        gc_after: 执行后是否垃圾回收
        monitor: 是否监控内存使用
        
    返回:
        函数执行结果
    """
    # 执行前内存状态
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    
    if gc_before:
        gc.collect()
    
    # 启动内存监控
    memory_monitor = None
    if monitor:
        memory_monitor = MemoryMonitor()
        memory_monitor.start_monitoring()
    
    try:
        logger.info(f"开始执行函数 {func.__name__}，初始内存: {initial_memory:.2f} MB")
        result = func(*args, **kwargs)
        return result
        
    finally:
        # 停止监控
        if memory_monitor:
            memory_monitor.stop_monitoring()
        
        if gc_after:
            collected = gc.collect()
            logger.info(f"函数执行后垃圾回收: {collected} 个对象")
        
        # 最终内存状态
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory
        
        logger.info(f"函数 {func.__name__} 执行完成，最终内存: {final_memory:.2f} MB，"
                   f"增长: {memory_increase:.2f} MB")


# 内存优化工具函数
def clear_large_objects(*objects) -> None:
    """清理大对象"""
    for obj in objects:
        if hasattr(obj, 'clear'):
            obj.clear()
        elif hasattr(obj, '__del__'):
            del obj
    gc.collect()


def get_memory_usage() -> Dict[str, float]:
    """获取当前内存使用情况"""
    process = psutil.Process()
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()
    
    return {
        "process_rss_mb": memory_info.rss / 1024 / 1024,
        "process_vms_mb": memory_info.vms / 1024 / 1024,
        "process_percent": process.memory_percent(),
        "system_total_mb": system_memory.total / 1024 / 1024,
        "system_available_mb": system_memory.available / 1024 / 1024,
        "system_used_percent": system_memory.percent
    }


def log_memory_usage(prefix: str = "") -> None:
    """记录当前内存使用情况"""
    usage = get_memory_usage()
    logger.info(f"{prefix}内存使用: 进程 {usage['process_rss_mb']:.2f} MB "
               f"({usage['process_percent']:.1f}%), "
               f"系统 {usage['system_used_percent']:.1f}%")


if __name__ == "__main__":
    # 示例用法
    def test_memory_optimization():
        """测试内存优化功能"""
        
        # 测试内存监控
        with memory_optimized_context(monitor=True) as monitor:
            # 模拟内存密集型操作
            large_data = [i for i in range(100000)]
            
            # 使用批处理生成器
            for batch in batch_generator(large_data, batch_size=1000):
                # 处理批次数据
                processed = [x * 2 for x in batch]
                del processed
            
            # 使用内存高效迭代器
            for item in memory_efficient_iterator(large_data[:1000]):
                # 处理单个项目
                result = item ** 2
            
            del large_data
        
        print("内存优化测试完成")
    
    test_memory_optimization()