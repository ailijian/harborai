#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的批量处理器

专门针对PostgreSQL存储架构优化，提供：
- 智能批量大小调整
- 数据库连接池优化
- 高性能批量插入策略
- 实时性能监控
- 自适应负载均衡

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import asyncio
import time
import threading
import json
import hashlib
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from queue import Queue, Empty, Full, PriorityQueue
from enum import Enum
from collections import deque, defaultdict
import psycopg2
import psycopg2.extras
from psycopg2.pool import ThreadedConnectionPool
from contextlib import contextmanager

from ..utils.logger import get_logger
from ..utils.exceptions import StorageError
from .batch_processor import BatchStrategy, BatchPriority, BatchConfig, BatchItem, BatchStats

logger = get_logger(__name__)


class OptimizationStrategy(Enum):
    """优化策略"""
    THROUGHPUT = "throughput"      # 吞吐量优先
    LATENCY = "latency"           # 延迟优先
    BALANCED = "balanced"         # 平衡模式
    MEMORY = "memory"             # 内存优化


@dataclass
class PerformanceMetrics:
    """性能指标"""
    throughput: float = 0.0           # 每秒处理项数
    latency_p50: float = 0.0          # 50%延迟
    latency_p95: float = 0.0          # 95%延迟
    latency_p99: float = 0.0          # 99%延迟
    memory_usage: float = 0.0         # 内存使用量(MB)
    cpu_usage: float = 0.0            # CPU使用率
    db_connection_usage: float = 0.0   # 数据库连接使用率
    error_rate: float = 0.0           # 错误率
    queue_depth: int = 0              # 队列深度
    batch_efficiency: float = 0.0     # 批处理效率


@dataclass
class OptimizedBatchConfig(BatchConfig):
    """优化的批处理配置"""
    # 基础配置继承自BatchConfig
    
    # 数据库优化配置
    db_pool_min_connections: int = 5
    db_pool_max_connections: int = 20
    db_connection_timeout: float = 30.0
    db_statement_timeout: float = 60.0
    
    # 批量插入优化
    use_copy_from: bool = True        # 使用COPY FROM优化
    use_prepared_statements: bool = True
    enable_compression: bool = True
    compression_threshold: int = 1000  # 压缩阈值
    
    # 性能优化
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    auto_tune_batch_size: bool = True
    performance_sample_rate: float = 0.1  # 性能采样率
    metrics_window_size: int = 1000
    
    # 内存管理
    max_memory_usage_mb: float = 512.0
    memory_check_interval: float = 10.0
    
    # 监控配置
    enable_detailed_metrics: bool = True
    metrics_export_interval: float = 60.0


class OptimizedBatchProcessor:
    """优化的批量处理器
    
    专门针对PostgreSQL存储架构优化，提供高性能批量处理能力。
    """
    
    def __init__(self, 
                 connection_string: str,
                 table_name: str,
                 config: Optional[OptimizedBatchConfig] = None,
                 error_callback: Optional[Callable[[Exception, List[BatchItem]], None]] = None):
        """初始化优化的批量处理器
        
        Args:
            connection_string: PostgreSQL连接字符串
            table_name: 目标表名
            config: 优化配置
            error_callback: 错误回调函数
        """
        self.connection_string = connection_string
        self.table_name = table_name
        self.config = config or OptimizedBatchConfig()
        self.error_callback = error_callback
        
        # 数据库连接池
        self._db_pool: Optional[ThreadedConnectionPool] = None
        
        # 队列系统
        self._priority_queue = PriorityQueue()
        self._batch_queue: Queue[List[BatchItem]] = Queue()
        
        # 统计和监控
        self._stats = BatchStats()
        self._performance_metrics = PerformanceMetrics()
        self._latency_samples = deque(maxlen=self.config.metrics_window_size)
        self._throughput_samples = deque(maxlen=100)
        
        # 线程控制
        self._running = False
        self._worker_threads: List[threading.Thread] = []
        self._metrics_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 自适应参数
        self._current_batch_size = self.config.min_batch_size
        self._last_optimization = time.time()
        self._optimization_history = deque(maxlen=50)
        
        # 预编译语句缓存
        self._prepared_statements: Dict[str, str] = {}
        
        # 内存监控
        self._memory_usage_history = deque(maxlen=100)
        
        logger.info(f"优化的批量处理器已初始化，目标表: {table_name}")
    
    async def start(self):
        """启动批量处理器"""
        if self._running:
            logger.warning("批量处理器已在运行")
            return
        
        try:
            # 初始化数据库连接池
            await self._initialize_db_pool()
            
            # 启动工作线程
            self._start_worker_threads()
            
            # 启动监控线程
            if self.config.enable_detailed_metrics:
                self._start_metrics_thread()
            
            self._running = True
            logger.info("优化的批量处理器已启动")
            
        except Exception as e:
            logger.error(f"启动批量处理器失败: {e}")
            await self.stop()
            raise StorageError(f"启动批量处理器失败: {e}") from e
    
    async def stop(self, timeout: float = 30.0):
        """停止批量处理器
        
        Args:
            timeout: 停止超时时间
        """
        logger.info("正在停止优化的批量处理器...")
        self._running = False
        
        # 等待工作线程结束
        for thread in self._worker_threads:
            if thread.is_alive():
                thread.join(timeout=timeout / len(self._worker_threads))
        
        # 停止监控线程
        if self._metrics_thread and self._metrics_thread.is_alive():
            self._metrics_thread.join(timeout=5.0)
        
        # 处理剩余项目
        await self._flush_remaining_items()
        
        # 关闭数据库连接池
        if self._db_pool:
            self._db_pool.closeall()
        
        logger.info("优化的批量处理器已停止")
    
    async def add_item(self, 
                      data: Any, 
                      priority: BatchPriority = BatchPriority.NORMAL,
                      callback: Optional[Callable] = None) -> bool:
        """添加项目到批处理队列
        
        Args:
            data: 要处理的数据
            priority: 优先级
            callback: 完成回调
            
        Returns:
            bool: 是否成功添加
        """
        if not self._running:
            return False
        
        item = BatchItem(
            data=data,
            priority=priority,
            callback=callback
        )
        
        try:
            # 使用优先级队列
            priority_value = priority.value
            self._priority_queue.put_nowait((priority_value, time.time(), item))
            
            with self._lock:
                self._stats.total_items += 1
                self._stats.queue_size += 1
                self._performance_metrics.queue_depth = self._priority_queue.qsize()
            
            return True
        except Full:
            logger.warning("批处理队列已满，无法添加项目")
            return False
    
    async def _initialize_db_pool(self):
        """初始化数据库连接池"""
        try:
            self._db_pool = ThreadedConnectionPool(
                minconn=self.config.db_pool_min_connections,
                maxconn=self.config.db_pool_max_connections,
                dsn=self.connection_string
            )
            
            # 测试连接
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
            
            logger.info(f"数据库连接池已初始化: {self.config.db_pool_min_connections}-{self.config.db_pool_max_connections}")
            
        except Exception as e:
            logger.error(f"初始化数据库连接池失败: {e}")
            raise
    
    @contextmanager
    def _get_db_connection(self):
        """获取数据库连接"""
        if not self._db_pool:
            raise StorageError("数据库连接池未初始化")
        
        conn = None
        try:
            conn = self._db_pool.getconn()
            yield conn
        finally:
            if conn:
                self._db_pool.putconn(conn)
    
    def _start_worker_threads(self):
        """启动工作线程"""
        # 根据优化策略确定线程数
        if self.config.optimization_strategy == OptimizationStrategy.THROUGHPUT:
            num_workers = min(8, self.config.db_pool_max_connections // 2)
        elif self.config.optimization_strategy == OptimizationStrategy.LATENCY:
            num_workers = 2
        else:  # BALANCED or MEMORY
            num_workers = 4
        
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"OptimizedBatchWorker-{i}"
            )
            worker.start()
            self._worker_threads.append(worker)
        
        logger.info(f"已启动 {num_workers} 个工作线程")
    
    def _start_metrics_thread(self):
        """启动监控线程"""
        self._metrics_thread = threading.Thread(
            target=self._metrics_loop,
            daemon=True,
            name="OptimizedBatchMetrics"
        )
        self._metrics_thread.start()
        logger.info("监控线程已启动")
    
    def _worker_loop(self):
        """工作线程主循环"""
        while self._running:
            try:
                # 收集批次
                batch = self._collect_optimized_batch()
                
                if not batch:
                    time.sleep(0.1)
                    continue
                
                # 处理批次
                start_time = time.time()
                success = self._process_batch_optimized(batch)
                processing_time = time.time() - start_time
                
                # 更新统计信息
                self._update_batch_stats(batch, success, processing_time)
                
                # 自适应优化
                if self.config.auto_tune_batch_size:
                    self._adaptive_optimization(processing_time, len(batch), success)
                
            except Exception as e:
                logger.error(f"工作线程处理批次时发生错误: {e}")
                if self.error_callback:
                    self.error_callback(e, batch if 'batch' in locals() else [])
    
    def _collect_optimized_batch(self) -> List[BatchItem]:
        """收集优化的批次"""
        batch = []
        batch_size = self._current_batch_size
        start_time = time.time()
        
        # 根据优化策略调整收集逻辑
        if self.config.optimization_strategy == OptimizationStrategy.LATENCY:
            # 延迟优先：快速收集小批次
            timeout = 0.1
        elif self.config.optimization_strategy == OptimizationStrategy.THROUGHPUT:
            # 吞吐量优先：等待更大批次
            timeout = self.config.flush_interval
        else:
            # 平衡模式
            timeout = self.config.flush_interval / 2
        
        while len(batch) < batch_size and (time.time() - start_time) < timeout:
            try:
                # 从优先级队列获取项目
                priority, timestamp, item = self._priority_queue.get_nowait()
                batch.append(item)
                
                with self._lock:
                    self._stats.queue_size -= 1
                    
            except Empty:
                if batch:  # 如果已有项目，直接返回
                    break
                time.sleep(0.01)
        
        return batch
    
    def _process_batch_optimized(self, batch: List[BatchItem]) -> bool:
        """优化的批次处理"""
        if not batch:
            return True
        
        try:
            with self._get_db_connection() as conn:
                # 根据配置选择最优的插入策略
                if self.config.use_copy_from and len(batch) >= self.config.compression_threshold:
                    return self._process_batch_copy_from(conn, batch)
                elif self.config.use_prepared_statements:
                    return self._process_batch_prepared(conn, batch)
                else:
                    return self._process_batch_standard(conn, batch)
                    
        except Exception as e:
            logger.error(f"批次处理失败: {e}")
            return False
    
    def _process_batch_copy_from(self, conn, batch: List[BatchItem]) -> bool:
        """使用COPY FROM优化的批次处理"""
        try:
            import io
            
            # 准备数据流
            data_stream = io.StringIO()
            for item in batch:
                # 将数据转换为CSV格式
                row_data = self._format_data_for_copy(item.data)
                data_stream.write(row_data + '\n')
            
            data_stream.seek(0)
            
            # 使用COPY FROM插入
            with conn.cursor() as cursor:
                cursor.copy_from(
                    data_stream,
                    self.table_name,
                    sep='\t',
                    null='\\N'
                )
                conn.commit()
            
            logger.debug(f"使用COPY FROM成功处理 {len(batch)} 项")
            return True
            
        except Exception as e:
            logger.error(f"COPY FROM处理失败: {e}")
            conn.rollback()
            return False
    
    def _process_batch_prepared(self, conn, batch: List[BatchItem]) -> bool:
        """使用预编译语句的批次处理"""
        try:
            # 获取或创建预编译语句
            stmt_key = f"insert_{self.table_name}"
            if stmt_key not in self._prepared_statements:
                self._prepared_statements[stmt_key] = self._create_prepared_statement()
            
            prepared_stmt = self._prepared_statements[stmt_key]
            
            # 批量执行
            with conn.cursor() as cursor:
                data_list = [self._format_data_for_prepared(item.data) for item in batch]
                cursor.executemany(prepared_stmt, data_list)
                conn.commit()
            
            logger.debug(f"使用预编译语句成功处理 {len(batch)} 项")
            return True
            
        except Exception as e:
            logger.error(f"预编译语句处理失败: {e}")
            conn.rollback()
            return False
    
    def _process_batch_standard(self, conn, batch: List[BatchItem]) -> bool:
        """标准批次处理"""
        try:
            with conn.cursor() as cursor:
                for item in batch:
                    # 构建插入语句
                    insert_stmt, values = self._build_insert_statement(item.data)
                    cursor.execute(insert_stmt, values)
                
                conn.commit()
            
            logger.debug(f"使用标准方式成功处理 {len(batch)} 项")
            return True
            
        except Exception as e:
            logger.error(f"标准处理失败: {e}")
            conn.rollback()
            return False
    
    def _format_data_for_copy(self, data: Dict[str, Any]) -> str:
        """格式化数据用于COPY FROM"""
        # 根据表结构格式化数据
        # 这里需要根据具体的表结构来实现
        values = []
        for key, value in data.items():
            if value is None:
                values.append('\\N')
            elif isinstance(value, str):
                values.append(value.replace('\t', '\\t').replace('\n', '\\n'))
            else:
                values.append(str(value))
        
        return '\t'.join(values)
    
    def _format_data_for_prepared(self, data: Dict[str, Any]) -> Tuple:
        """格式化数据用于预编译语句"""
        # 根据表结构返回值元组
        return tuple(data.values())
    
    def _create_prepared_statement(self) -> str:
        """创建预编译插入语句"""
        # 这里需要根据具体的表结构来实现
        # 示例：INSERT INTO table_name (col1, col2, ...) VALUES (%s, %s, ...)
        return f"INSERT INTO {self.table_name} VALUES (%s, %s, %s, %s, %s)"
    
    def _build_insert_statement(self, data: Dict[str, Any]) -> Tuple[str, Tuple]:
        """构建插入语句"""
        columns = list(data.keys())
        values = list(data.values())
        
        columns_str = ', '.join(columns)
        placeholders = ', '.join(['%s'] * len(values))
        
        stmt = f"INSERT INTO {self.table_name} ({columns_str}) VALUES ({placeholders})"
        return stmt, tuple(values)
    
    def _update_batch_stats(self, batch: List[BatchItem], success: bool, processing_time: float):
        """更新批次统计信息"""
        with self._lock:
            self._stats.total_batches += 1
            
            if success:
                self._stats.successful_batches += 1
                self._stats.processed_items += len(batch)
            else:
                self._stats.failed_batches += 1
                self._stats.failed_items += len(batch)
            
            # 更新平均值
            total_processed = self._stats.processed_items + self._stats.failed_items
            if total_processed > 0:
                self._stats.average_batch_size = total_processed / self._stats.total_batches
            
            # 记录延迟样本
            self._latency_samples.append(processing_time)
            
            # 计算吞吐量
            if processing_time > 0:
                throughput = len(batch) / processing_time
                self._throughput_samples.append(throughput)
                
                if self._throughput_samples:
                    self._stats.throughput = sum(self._throughput_samples) / len(self._throughput_samples)
    
    def _adaptive_optimization(self, processing_time: float, batch_size: int, success: bool):
        """自适应优化"""
        current_time = time.time()
        
        # 限制优化频率
        if current_time - self._last_optimization < 10.0:
            return
        
        # 记录性能数据
        efficiency = batch_size / processing_time if processing_time > 0 else 0
        self._optimization_history.append({
            'batch_size': batch_size,
            'processing_time': processing_time,
            'efficiency': efficiency,
            'success': success,
            'timestamp': current_time
        })
        
        # 分析性能趋势
        if len(self._optimization_history) >= 5:
            self._adjust_batch_size()
        
        self._last_optimization = current_time
    
    def _adjust_batch_size(self):
        """调整批次大小"""
        recent_data = list(self._optimization_history)[-5:]
        
        # 计算平均效率
        avg_efficiency = sum(d['efficiency'] for d in recent_data) / len(recent_data)
        success_rate = sum(1 for d in recent_data if d['success']) / len(recent_data)
        
        # 根据策略调整
        if self.config.optimization_strategy == OptimizationStrategy.THROUGHPUT:
            # 吞吐量优先：增加批次大小
            if success_rate > 0.9 and avg_efficiency > 100:
                self._current_batch_size = min(
                    self._current_batch_size * 1.2,
                    self.config.max_batch_size
                )
        elif self.config.optimization_strategy == OptimizationStrategy.LATENCY:
            # 延迟优先：减少批次大小
            if avg_efficiency < 50:
                self._current_batch_size = max(
                    self._current_batch_size * 0.8,
                    self.config.min_batch_size
                )
        else:
            # 平衡模式：根据成功率和效率调整
            if success_rate > 0.95 and avg_efficiency > 80:
                self._current_batch_size = min(
                    self._current_batch_size * 1.1,
                    self.config.max_batch_size
                )
            elif success_rate < 0.8 or avg_efficiency < 30:
                self._current_batch_size = max(
                    self._current_batch_size * 0.9,
                    self.config.min_batch_size
                )
        
        logger.debug(f"批次大小调整为: {self._current_batch_size}")
    
    def _metrics_loop(self):
        """监控线程循环"""
        while self._running:
            try:
                self._update_performance_metrics()
                time.sleep(self.config.metrics_export_interval)
            except Exception as e:
                logger.error(f"监控线程错误: {e}")
    
    def _update_performance_metrics(self):
        """更新性能指标"""
        with self._lock:
            # 计算延迟百分位数
            if self._latency_samples:
                sorted_latencies = sorted(self._latency_samples)
                n = len(sorted_latencies)
                
                self._performance_metrics.latency_p50 = sorted_latencies[int(n * 0.5)]
                self._performance_metrics.latency_p95 = sorted_latencies[int(n * 0.95)]
                self._performance_metrics.latency_p99 = sorted_latencies[int(n * 0.99)]
            
            # 计算错误率
            total_batches = self._stats.total_batches
            if total_batches > 0:
                self._performance_metrics.error_rate = self._stats.failed_batches / total_batches
            
            # 计算批处理效率
            if self._stats.total_batches > 0:
                self._performance_metrics.batch_efficiency = (
                    self._stats.successful_batches / self._stats.total_batches
                )
            
            # 更新队列深度
            self._performance_metrics.queue_depth = self._priority_queue.qsize()
            
            # 数据库连接使用率
            if self._db_pool:
                # 这里需要根据具体的连接池实现来获取使用率
                pass
    
    async def _flush_remaining_items(self):
        """刷新剩余项目"""
        logger.info("正在处理剩余项目...")
        
        remaining_items = []
        while not self._priority_queue.empty():
            try:
                _, _, item = self._priority_queue.get_nowait()
                remaining_items.append(item)
            except Empty:
                break
        
        if remaining_items:
            logger.info(f"处理剩余 {len(remaining_items)} 项")
            success = self._process_batch_optimized(remaining_items)
            if success:
                logger.info("剩余项目处理完成")
            else:
                logger.warning("剩余项目处理失败")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = asdict(self._stats)
            stats['performance_metrics'] = asdict(self._performance_metrics)
            stats['current_batch_size'] = self._current_batch_size
            stats['optimization_strategy'] = self.config.optimization_strategy.value
            
            return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'running': self._running,
            'worker_threads': len([t for t in self._worker_threads if t.is_alive()]),
            'queue_size': self._priority_queue.qsize(),
            'db_pool_status': 'healthy' if self._db_pool else 'unavailable',
            'error_rate': self._performance_metrics.error_rate,
            'throughput': self._stats.throughput,
            'memory_usage': self._performance_metrics.memory_usage
        }


# 全局实例管理
_optimized_batch_processor: Optional[OptimizedBatchProcessor] = None


def get_optimized_batch_processor(
    connection_string: str,
    table_name: str,
    config: Optional[OptimizedBatchConfig] = None
) -> OptimizedBatchProcessor:
    """获取优化的批量处理器实例
    
    Args:
        connection_string: 数据库连接字符串
        table_name: 目标表名
        config: 配置
        
    Returns:
        OptimizedBatchProcessor: 批量处理器实例
    """
    global _optimized_batch_processor
    
    if _optimized_batch_processor is None:
        _optimized_batch_processor = OptimizedBatchProcessor(
            connection_string=connection_string,
            table_name=table_name,
            config=config
        )
    
    return _optimized_batch_processor


if __name__ == "__main__":
    # 示例用法
    import asyncio
    
    async def main():
        config = OptimizedBatchConfig(
            min_batch_size=50,
            max_batch_size=500,
            optimization_strategy=OptimizationStrategy.THROUGHPUT,
            use_copy_from=True,
            enable_detailed_metrics=True
        )
        
        processor = OptimizedBatchProcessor(
            connection_string="postgresql://user:pass@localhost:5432/db",
            table_name="test_table",
            config=config
        )
        
        await processor.start()
        
        # 添加测试数据
        for i in range(1000):
            await processor.add_item({
                'id': i,
                'data': f'test_data_{i}',
                'timestamp': datetime.now()
            })
        
        # 等待处理完成
        await asyncio.sleep(10)
        
        # 获取统计信息
        stats = processor.get_stats()
        print(f"处理统计: {stats}")
        
        await processor.stop()
    
    asyncio.run(main())