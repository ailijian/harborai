"""
智能批处理管理模块
提供高性能、自适应的批处理功能
"""

import asyncio
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from queue import Queue, Empty, Full
from enum import Enum
import json

from ..utils.logger import get_logger
from ..utils.exceptions import StorageError

logger = get_logger(__name__)


class BatchStrategy(Enum):
    """批处理策略"""
    SIZE_BASED = "size_based"  # 基于大小
    TIME_BASED = "time_based"  # 基于时间
    ADAPTIVE = "adaptive"      # 自适应
    PRIORITY_BASED = "priority_based"  # 基于优先级


class BatchPriority(Enum):
    """批处理优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class BatchConfig:
    """批处理配置"""
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    min_batch_size: int = 10
    max_batch_size: int = 100
    flush_interval: float = 5.0
    max_wait_time: float = 30.0
    enable_compression: bool = True
    enable_priority: bool = True
    adaptive_threshold: float = 0.8  # 自适应阈值
    performance_window: int = 100    # 性能窗口大小


@dataclass
class BatchItem:
    """批处理项"""
    data: Any
    priority: BatchPriority = BatchPriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class BatchStats:
    """批处理统计信息"""
    total_items: int = 0
    processed_items: int = 0
    failed_items: int = 0
    total_batches: int = 0
    successful_batches: int = 0
    failed_batches: int = 0
    average_batch_size: float = 0.0
    average_processing_time: float = 0.0
    throughput: float = 0.0  # 每秒处理项数
    last_flush_time: Optional[datetime] = None
    queue_size: int = 0


class AdaptiveBatchProcessor:
    """自适应批处理器"""
    
    def __init__(self, 
                 processor_func: Callable[[List[BatchItem]], bool],
                 config: Optional[BatchConfig] = None,
                 error_callback: Optional[Callable[[Exception, List[BatchItem]], None]] = None):
        """初始化批处理器
        
        Args:
            processor_func: 批处理函数，接收BatchItem列表，返回是否成功
            config: 批处理配置
            error_callback: 错误回调函数
        """
        self.processor_func = processor_func
        self.config = config or BatchConfig()
        self.error_callback = error_callback
        
        # 队列和状态
        self._queue: Queue[BatchItem] = Queue()
        self._priority_queues: Dict[BatchPriority, Queue[BatchItem]] = {
            priority: Queue() for priority in BatchPriority
        }
        self._stats = BatchStats()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 自适应参数
        self._performance_history: List[float] = []
        self._current_batch_size = self.config.min_batch_size
        self._last_adjustment = time.time()
        
        # 压缩相关
        self._compression_enabled = self.config.enable_compression
    
    def start(self):
        """启动批处理器"""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="BatchProcessor-Worker"
        )
        self._worker_thread.start()
        logger.info("批处理器已启动")
    
    def stop(self, timeout: float = 30.0):
        """停止批处理器
        
        Args:
            timeout: 停止超时时间
        """
        logger.info("正在停止批处理器...")
        self._running = False
        
        # 等待工作线程结束
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)
        
        # 处理剩余项目
        self._flush_remaining_items()
        
        logger.info("批处理器已停止")
    
    def add_item(self, 
                 data: Any, 
                 priority: BatchPriority = BatchPriority.NORMAL,
                 callback: Optional[Callable] = None,
                 max_retries: int = 3) -> bool:
        """添加项目到批处理队列
        
        Args:
            data: 要处理的数据
            priority: 优先级
            callback: 完成回调
            max_retries: 最大重试次数
            
        Returns:
            bool: 是否成功添加
        """
        if not self._running:
            return False
        
        item = BatchItem(
            data=data,
            priority=priority,
            callback=callback,
            max_retries=max_retries
        )
        
        try:
            if self.config.enable_priority:
                self._priority_queues[priority].put_nowait(item)
            else:
                self._queue.put_nowait(item)
            
            with self._lock:
                self._stats.total_items += 1
                self._stats.queue_size += 1
            
            return True
        except Full:
            logger.warning("批处理队列已满，无法添加项目")
            return False
    
    def _worker_loop(self):
        """工作线程主循环"""
        last_flush = time.time()
        
        while self._running:
            try:
                # 收集批次
                batch = self._collect_batch()
                
                current_time = time.time()
                should_flush = self._should_flush(batch, current_time - last_flush)
                
                if should_flush and batch:
                    success = self._process_batch(batch)
                    if success:
                        last_flush = current_time
                    else:
                        # 处理失败，重新排队或丢弃
                        self._handle_failed_batch(batch)
                
                # 短暂休眠避免CPU占用过高
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"批处理工作循环错误: {e}")
                time.sleep(1)  # 错误时等待更长时间
    
    def _collect_batch(self) -> List[BatchItem]:
        """收集批次"""
        batch = []
        
        if self.config.enable_priority:
            # 按优先级收集
            for priority in sorted(BatchPriority, key=lambda x: x.value, reverse=True):
                queue = self._priority_queues[priority]
                while len(batch) < self._current_batch_size and not queue.empty():
                    try:
                        item = queue.get_nowait()
                        batch.append(item)
                    except Empty:
                        break
        else:
            # 普通收集
            while len(batch) < self._current_batch_size and not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    batch.append(item)
                except Empty:
                    break
        
        return batch
    
    def _should_flush(self, batch: List[BatchItem], time_since_last: float) -> bool:
        """判断是否应该刷新批次"""
        if not batch:
            return False
        
        # 基于策略判断
        if self.config.strategy == BatchStrategy.SIZE_BASED:
            return len(batch) >= self._current_batch_size
        
        elif self.config.strategy == BatchStrategy.TIME_BASED:
            return time_since_last >= self.config.flush_interval
        
        elif self.config.strategy == BatchStrategy.PRIORITY_BASED:
            # 检查是否有高优先级项目
            has_critical = any(item.priority == BatchPriority.CRITICAL for item in batch)
            if has_critical:
                return True
            
            has_high = any(item.priority == BatchPriority.HIGH for item in batch)
            if has_high and len(batch) >= self.config.min_batch_size:
                return True
            
            return len(batch) >= self._current_batch_size or time_since_last >= self.config.flush_interval
        
        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            # 自适应策略
            return self._adaptive_should_flush(batch, time_since_last)
        
        return len(batch) >= self._current_batch_size or time_since_last >= self.config.flush_interval
    
    def _adaptive_should_flush(self, batch: List[BatchItem], time_since_last: float) -> bool:
        """自适应刷新判断"""
        # 基本条件
        if len(batch) >= self.config.max_batch_size:
            return True
        
        if time_since_last >= self.config.max_wait_time:
            return True
        
        # 检查队列压力
        total_queue_size = self._get_total_queue_size()
        if total_queue_size > self.config.max_batch_size * 2:
            return True
        
        # 检查最老项目的等待时间
        if batch:
            oldest_item = min(batch, key=lambda x: x.timestamp)
            wait_time = (datetime.now() - oldest_item.timestamp).total_seconds()
            if wait_time > self.config.max_wait_time * 0.5:
                return True
        
        # 基于性能历史调整
        if len(self._performance_history) > 10:
            avg_performance = sum(self._performance_history[-10:]) / 10
            if avg_performance < self.config.adaptive_threshold:
                # 性能较差，减小批次大小，增加刷新频率
                return len(batch) >= max(self.config.min_batch_size, self._current_batch_size // 2)
        
        return len(batch) >= self._current_batch_size or time_since_last >= self.config.flush_interval
    
    def _process_batch(self, batch: List[BatchItem]) -> bool:
        """处理批次"""
        if not batch:
            return True
        
        start_time = time.time()
        
        try:
            # 压缩数据（如果启用）
            if self._compression_enabled:
                batch = self._compress_batch(batch)
            
            # 调用处理函数
            success = self.processor_func(batch)
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            with self._lock:
                self._stats.processed_items += len(batch)
                self._stats.total_batches += 1
                self._stats.queue_size -= len(batch)
                
                if success:
                    self._stats.successful_batches += 1
                else:
                    self._stats.failed_batches += 1
                    self._stats.failed_items += len(batch)
                
                # 更新平均值
                if self._stats.total_batches > 0:
                    self._stats.average_batch_size = self._stats.processed_items / self._stats.total_batches
                
                if self._stats.successful_batches > 0:
                    old_avg = self._stats.average_processing_time
                    self._stats.average_processing_time = (
                        (old_avg * (self._stats.successful_batches - 1) + processing_time) /
                        self._stats.successful_batches
                    )
                
                # 计算吞吐量
                if processing_time > 0:
                    self._stats.throughput = len(batch) / processing_time
                
                self._stats.last_flush_time = datetime.now()
            
            # 更新性能历史
            self._update_performance_history(processing_time, len(batch))
            
            # 自适应调整
            if self.config.strategy == BatchStrategy.ADAPTIVE:
                self._adjust_batch_size(success, processing_time, len(batch))
            
            # 调用回调
            for item in batch:
                if item.callback:
                    try:
                        item.callback(success, None if success else "批处理失败")
                    except Exception as e:
                        logger.warning(f"回调执行失败: {e}")
            
            if success:
                logger.debug(f"成功处理批次，大小: {len(batch)}, 耗时: {processing_time:.3f}s")
            else:
                logger.warning(f"批次处理失败，大小: {len(batch)}")
            
            return success
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            with self._lock:
                self._stats.failed_batches += 1
                self._stats.failed_items += len(batch)
                self._stats.queue_size -= len(batch)
            
            logger.error(f"批处理过程中发生错误: {e}")
            
            if self.error_callback:
                try:
                    self.error_callback(e, batch)
                except Exception as callback_error:
                    logger.error(f"错误回调执行失败: {callback_error}")
            
            return False
    
    def _compress_batch(self, batch: List[BatchItem]) -> List[BatchItem]:
        """压缩批次数据"""
        # 这里可以实现数据压缩逻辑
        # 例如：合并相似项目、去重等
        return batch
    
    def _handle_failed_batch(self, batch: List[BatchItem]):
        """处理失败的批次"""
        for item in batch:
            item.retry_count += 1
            
            if item.retry_count <= item.max_retries:
                # 重新排队
                try:
                    if self.config.enable_priority:
                        self._priority_queues[item.priority].put_nowait(item)
                    else:
                        self._queue.put_nowait(item)
                except Full:
                    logger.warning("重试队列已满，丢弃项目")
                    if item.callback:
                        item.callback(False, "重试队列已满")
            else:
                # 超过最大重试次数，丢弃
                logger.warning(f"项目超过最大重试次数，丢弃: {item.retry_count}")
                if item.callback:
                    item.callback(False, "超过最大重试次数")
    
    def _update_performance_history(self, processing_time: float, batch_size: int):
        """更新性能历史"""
        if batch_size > 0:
            performance = batch_size / processing_time  # 每秒处理项数
            self._performance_history.append(performance)
            
            # 保持历史记录在合理范围内
            if len(self._performance_history) > self.config.performance_window:
                self._performance_history.pop(0)
    
    def _adjust_batch_size(self, success: bool, processing_time: float, batch_size: int):
        """自适应调整批次大小"""
        current_time = time.time()
        
        # 限制调整频率
        if current_time - self._last_adjustment < 10:  # 10秒内不重复调整
            return
        
        if not success:
            # 处理失败，减小批次大小
            self._current_batch_size = max(
                self.config.min_batch_size,
                int(self._current_batch_size * 0.8)
            )
            self._last_adjustment = current_time
            logger.debug(f"处理失败，调整批次大小为: {self._current_batch_size}")
            return
        
        # 基于性能调整
        if len(self._performance_history) >= 5:
            recent_performance = sum(self._performance_history[-5:]) / 5
            
            if recent_performance > self.config.adaptive_threshold * 1.2:
                # 性能良好，可以增大批次
                new_size = min(
                    self.config.max_batch_size,
                    int(self._current_batch_size * 1.1)
                )
                if new_size != self._current_batch_size:
                    self._current_batch_size = new_size
                    self._last_adjustment = current_time
                    logger.debug(f"性能良好，调整批次大小为: {self._current_batch_size}")
            
            elif recent_performance < self.config.adaptive_threshold * 0.8:
                # 性能较差，减小批次
                new_size = max(
                    self.config.min_batch_size,
                    int(self._current_batch_size * 0.9)
                )
                if new_size != self._current_batch_size:
                    self._current_batch_size = new_size
                    self._last_adjustment = current_time
                    logger.debug(f"性能较差，调整批次大小为: {self._current_batch_size}")
    
    def _get_total_queue_size(self) -> int:
        """获取总队列大小"""
        if self.config.enable_priority:
            return sum(queue.qsize() for queue in self._priority_queues.values())
        else:
            return self._queue.qsize()
    
    def _flush_remaining_items(self):
        """刷新剩余项目"""
        logger.info("处理剩余的批处理项目...")
        
        # 收集所有剩余项目
        remaining_items = []
        
        if self.config.enable_priority:
            for queue in self._priority_queues.values():
                while not queue.empty():
                    try:
                        item = queue.get_nowait()
                        remaining_items.append(item)
                    except Empty:
                        break
        else:
            while not self._queue.empty():
                try:
                    item = self._queue.get_nowait()
                    remaining_items.append(item)
                except Empty:
                    break
        
        # 分批处理剩余项目
        while remaining_items:
            batch = remaining_items[:self.config.max_batch_size]
            remaining_items = remaining_items[self.config.max_batch_size:]
            
            try:
                self._process_batch(batch)
            except Exception as e:
                logger.error(f"处理剩余项目时发生错误: {e}")
    
    def get_stats(self) -> BatchStats:
        """获取批处理统计信息"""
        with self._lock:
            stats = BatchStats(
                total_items=self._stats.total_items,
                processed_items=self._stats.processed_items,
                failed_items=self._stats.failed_items,
                total_batches=self._stats.total_batches,
                successful_batches=self._stats.successful_batches,
                failed_batches=self._stats.failed_batches,
                average_batch_size=self._stats.average_batch_size,
                average_processing_time=self._stats.average_processing_time,
                throughput=self._stats.throughput,
                last_flush_time=self._stats.last_flush_time,
                queue_size=self._get_total_queue_size()
            )
        return stats
    
    def get_current_batch_size(self) -> int:
        """获取当前批次大小"""
        return self._current_batch_size
    
    def force_flush(self) -> bool:
        """强制刷新当前批次"""
        batch = self._collect_batch()
        if batch:
            return self._process_batch(batch)
        return True