#!/usr/bin/env python3
"""
增强的文件系统日志记录器

基于现有FileSystemLogger的增强版本，专注于：
- 分布式追踪集成
- 健康检查机制
- 性能优化
- 追踪信息存储

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import asyncio
import json
import os
import time
import threading
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Union
from queue import Queue
from threading import Thread, Lock
from dataclasses import asdict
import gzip
import shutil

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .file_logger import FileSystemLogger
from ..core.tracing.dual_trace_manager import DualTraceIDManager, DualTraceContext
from ..core.tracing.data_collector import TracingDataCollector, TracingRecord
from ..utils.logger import get_logger
from ..utils.exceptions import StorageError
from ..utils.timestamp import get_unified_timestamp

logger = get_logger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理datetime和Decimal对象"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


class HealthStatus:
    """健康状态管理类"""
    
    def __init__(self):
        self.is_healthy = True
        self.last_check = time.time()
        self.error_count = 0
        self.last_error = None
        self.disk_usage_percent = 0.0
        self.write_latency_ms = 0.0
        self.queue_size = 0
        
    def update_health(self, is_healthy: bool, error: Optional[Exception] = None):
        """更新健康状态"""
        self.is_healthy = is_healthy
        self.last_check = time.time()
        
        if error:
            self.error_count += 1
            self.last_error = str(error)
        else:
            self.error_count = 0
            self.last_error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "is_healthy": self.is_healthy,
            "last_check": datetime.fromtimestamp(self.last_check).isoformat(),
            "error_count": self.error_count,
            "last_error": self.last_error,
            "disk_usage_percent": self.disk_usage_percent,
            "write_latency_ms": self.write_latency_ms,
            "queue_size": self.queue_size
        }


class EnhancedFileSystemLogger(FileSystemLogger):
    """基于现有FileSystemLogger的增强版本
    
    增强功能：
    - 分布式追踪集成
    - 健康检查机制
    - 性能监控
    - 追踪信息记录
    - 自动恢复机制
    """
    
    def __init__(self, 
                 log_dir: str = "./logs",
                 file_prefix: str = "harborai",
                 batch_size: int = 100,
                 flush_interval: int = 30,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 max_files: int = 10,
                 compress_old_files: bool = True,
                 enable_tracing: bool = True,
                 tracing_sample_rate: float = 1.0,
                 health_check_interval: float = 60.0,
                 max_disk_usage_percent: float = 85.0,
                 error_callback: Optional[Callable[[Exception], None]] = None):
        """初始化增强的文件系统日志记录器
        
        Args:
            log_dir: 日志目录路径
            file_prefix: 日志文件前缀
            batch_size: 批量写入大小
            flush_interval: 刷新间隔（秒）
            max_file_size: 单个日志文件最大大小（字节）
            max_files: 保留的最大文件数量
            compress_old_files: 是否压缩旧文件
            enable_tracing: 是否启用分布式追踪
            tracing_sample_rate: 追踪采样率
            health_check_interval: 健康检查间隔（秒）
            max_disk_usage_percent: 最大磁盘使用率
            error_callback: 错误回调函数
        """
        super().__init__(
            log_dir=log_dir,
            file_prefix=file_prefix,
            batch_size=batch_size,
            flush_interval=flush_interval,
            max_file_size=max_file_size,
            max_files=max_files,
            compress_old_files=compress_old_files
        )
        
        # 追踪相关配置
        self.enable_tracing = enable_tracing
        self.tracing_sample_rate = tracing_sample_rate
        self.error_callback = error_callback
        
        # 健康检查配置
        self.health_check_interval = health_check_interval
        self.max_disk_usage_percent = max_disk_usage_percent
        self._health_status = HealthStatus()
        self._health_check_thread = None
        
        # 性能监控
        self._performance_metrics = {
            "total_logs": 0,
            "total_bytes": 0,
            "avg_write_latency": 0.0,
            "max_write_latency": 0.0,
            "error_count": 0,
            "last_flush_time": 0.0
        }
        
        # 初始化追踪组件
        if self.enable_tracing:
            self.dual_trace_manager = DualTraceIDManager()
            self.tracer = trace.get_tracer(__name__)
            # 注意：这里不初始化TracingDataCollector，因为它需要数据库连接
            self.tracing_collector = None
        else:
            self.dual_trace_manager = None
            self.tracer = None
            self.tracing_collector = None
    
    def set_tracing_collector(self, collector: TracingDataCollector):
        """设置追踪数据收集器
        
        Args:
            collector: 追踪数据收集器实例
        """
        self.tracing_collector = collector
    
    def start(self):
        """启动增强的文件系统日志记录器"""
        super().start()
        
        # 启动健康检查线程
        self._start_health_check()
        
        logger.info(
            "Enhanced FileSystem logger started",
            extra={
                "log_dir": str(self.log_dir),
                "tracing_enabled": self.enable_tracing,
                "health_check_interval": self.health_check_interval
            }
        )
    
    def stop(self):
        """停止增强的文件系统日志记录器"""
        # 停止健康检查线程
        self._stop_health_check()
        
        super().stop()
        logger.info("Enhanced FileSystem logger stopped")
    
    def log_request_with_tracing(self, 
                               trace_id: str,
                               model: str,
                               messages: List[Dict[str, Any]],
                               provider: str = "unknown",
                               operation_name: str = "ai.chat.completion",
                               **kwargs) -> Optional[DualTraceContext]:
        """记录请求日志并创建追踪span
        
        Args:
            trace_id: HarborAI追踪ID
            model: 模型名称
            messages: 消息列表
            provider: AI提供商
            operation_name: 操作名称
            **kwargs: 其他参数
            
        Returns:
            DualTraceContext: 双追踪上下文（如果启用追踪）
        """
        dual_context = None
        
        # 创建追踪span
        if self.enable_tracing and self.tracer:
            with self.tracer.start_as_current_span(operation_name) as span:
                try:
                    # 创建双追踪上下文
                    dual_context = self.dual_trace_manager.create_trace_context(
                        hb_trace_id=trace_id,
                        operation_name=operation_name,
                        service_name="harborai-file-logging"
                    )
                    
                    # 设置span属性
                    span.set_attributes({
                        "ai.provider": provider,
                        "ai.model": model,
                        "ai.operation": operation_name,
                        "harborai.trace_id": trace_id,
                        "harborai.message_count": len(messages),
                        "storage.type": "filesystem"
                    })
                    
                    # 记录请求日志
                    self.log_request(
                        trace_id=trace_id,
                        model=model,
                        messages=messages,
                        **kwargs
                    )
                    
                    span.set_status(Status(StatusCode.OK))
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    logger.error(
                        "Failed to log request with tracing",
                        extra={
                            "trace_id": trace_id,
                            "error": str(e),
                            "provider": provider,
                            "model": model
                        }
                    )
                    
                    # 调用错误回调
                    if self.error_callback:
                        self.error_callback(e)
                    
                    raise
        else:
            # 不启用追踪时，直接记录日志
            self.log_request(
                trace_id=trace_id,
                model=model,
                messages=messages,
                **kwargs
            )
        
        return dual_context
    
    def log_response_with_tracing(self,
                                trace_id: str,
                                response: Any,
                                latency: float,
                                success: bool = True,
                                error: Optional[str] = None,
                                dual_context: Optional[DualTraceContext] = None,
                                **kwargs):
        """记录响应日志并包含追踪信息
        
        Args:
            trace_id: 追踪ID
            response: 响应对象
            latency: 延迟时间
            success: 是否成功
            error: 错误信息
            dual_context: 双追踪上下文
            **kwargs: 其他参数
        """
        # 记录响应日志
        self.log_response(
            trace_id=trace_id,
            response=response,
            latency=latency,
            success=success,
            error=error
        )
        
        # 记录追踪信息到文件
        if self.enable_tracing and dual_context:
            try:
                tracing_entry = {
                    "trace_id": trace_id,
                    "timestamp": get_unified_timestamp(),
                    "type": "tracing",
                    "hb_trace_id": dual_context.hb_trace_id,
                    "otel_trace_id": dual_context.otel_trace_id,
                    "span_id": dual_context.span_id,
                    "operation_name": dual_context.operation_name,
                    "service_name": dual_context.service_name,
                    "start_time": dual_context.created_at.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_ms": int((datetime.now() - dual_context.created_at).total_seconds() * 1000),
                    "status": "completed" if success else "error",
                    "ai_provider": kwargs.get("provider", "unknown"),
                    "ai_model": kwargs.get("model", "unknown"),
                    "latency": latency,
                    "error": error
                }
                
                # 添加到队列
                self._log_queue.put(tracing_entry)
                
                # 如果有追踪数据收集器，也发送给它
                if self.tracing_collector:
                    tracing_record = TracingRecord(
                        hb_trace_id=dual_context.hb_trace_id,
                        otel_trace_id=dual_context.otel_trace_id,
                        span_id=dual_context.span_id,
                        parent_span_id=dual_context.parent_span_id,
                        operation_name=dual_context.operation_name,
                        service_name=dual_context.service_name,
                        start_time=dual_context.created_at,
                        end_time=datetime.now(),
                        status="completed" if success else "error",
                        ai_provider=kwargs.get("provider", "unknown"),
                        ai_model=kwargs.get("model", "unknown"),
                        error_message=error
                    )
                    
                    self.tracing_collector.collect_trace_data(tracing_record)
                
            except Exception as e:
                logger.error(
                    "Failed to record tracing information",
                    extra={
                        "trace_id": trace_id,
                        "error": str(e)
                    }
                )
    
    def _flush_batch(self, batch: List[Dict[str, Any]]):
        """批量写入日志到文件（重写以添加性能监控）"""
        if not batch:
            return
        
        start_time = time.time()
        
        try:
            # 调用父类方法
            super()._flush_batch(batch)
            
            # 更新性能指标
            write_latency = (time.time() - start_time) * 1000  # 转换为毫秒
            self._update_performance_metrics(len(batch), write_latency)
            
            # 更新健康状态
            self._health_status.update_health(True)
            self._health_status.write_latency_ms = write_latency
            self._health_status.queue_size = self._log_queue.qsize()
            
        except Exception as e:
            # 更新性能指标
            self._performance_metrics["error_count"] += 1
            
            # 更新健康状态
            self._health_status.update_health(False, e)
            
            # 调用错误回调
            if self.error_callback:
                self.error_callback(e)
            
            logger.error(f"Failed to flush batch to enhanced file system: {e}")
            raise
    
    def _update_performance_metrics(self, batch_size: int, write_latency: float):
        """更新性能指标"""
        self._performance_metrics["total_logs"] += batch_size
        self._performance_metrics["last_flush_time"] = time.time()
        
        # 更新平均写入延迟
        current_avg = self._performance_metrics["avg_write_latency"]
        total_logs = self._performance_metrics["total_logs"]
        
        if total_logs > 0:
            self._performance_metrics["avg_write_latency"] = (
                (current_avg * (total_logs - batch_size) + write_latency) / total_logs
            )
        
        # 更新最大写入延迟
        if write_latency > self._performance_metrics["max_write_latency"]:
            self._performance_metrics["max_write_latency"] = write_latency
    
    def _start_health_check(self):
        """启动健康检查线程"""
        if self._health_check_thread is None or not self._health_check_thread.is_alive():
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="EnhancedFileLogger-HealthCheck"
            )
            self._health_check_thread.start()
    
    def _stop_health_check(self):
        """停止健康检查线程"""
        # 健康检查线程是daemon线程，会自动停止
        pass
    
    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                time.sleep(self.health_check_interval)
                
                if not self._running:
                    break
                
                self._perform_health_check()
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                time.sleep(5)  # 出错时短暂等待
    
    def _perform_health_check(self):
        """执行健康检查"""
        try:
            # 检查磁盘使用率
            disk_usage = self._get_disk_usage()
            self._health_status.disk_usage_percent = disk_usage
            
            # 检查队列大小
            queue_size = self._log_queue.qsize()
            self._health_status.queue_size = queue_size
            
            # 检查日志目录是否可写
            is_writable = os.access(self.log_dir, os.W_OK)
            
            # 综合判断健康状态
            is_healthy = (
                disk_usage < self.max_disk_usage_percent and
                queue_size < self.batch_size * 10 and  # 队列不能太大
                is_writable
            )
            
            if not is_healthy:
                error_msg = []
                if disk_usage >= self.max_disk_usage_percent:
                    error_msg.append(f"Disk usage too high: {disk_usage:.1f}%")
                if queue_size >= self.batch_size * 10:
                    error_msg.append(f"Queue size too large: {queue_size}")
                if not is_writable:
                    error_msg.append("Log directory not writable")
                
                error = Exception("; ".join(error_msg))
                self._health_status.update_health(False, error)
                
                # 调用错误回调
                if self.error_callback:
                    self.error_callback(error)
            else:
                self._health_status.update_health(True)
            
            logger.debug(
                "Health check completed",
                extra={
                    "is_healthy": is_healthy,
                    "disk_usage": disk_usage,
                    "queue_size": queue_size,
                    "is_writable": is_writable
                }
            )
            
        except Exception as e:
            self._health_status.update_health(False, e)
            logger.error(f"Health check failed: {e}")
    
    def _get_disk_usage(self) -> float:
        """获取磁盘使用率"""
        try:
            statvfs = os.statvfs(self.log_dir)
            total_space = statvfs.f_frsize * statvfs.f_blocks
            free_space = statvfs.f_frsize * statvfs.f_available
            used_space = total_space - free_space
            
            if total_space > 0:
                return (used_space / total_space) * 100
            else:
                return 0.0
                
        except (OSError, AttributeError):
            # Windows系统或其他不支持statvfs的系统
            try:
                import shutil
                total, used, free = shutil.disk_usage(self.log_dir)
                return (used / total) * 100 if total > 0 else 0.0
            except Exception:
                return 0.0
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        return self._health_status.to_dict()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return {
            **self._performance_metrics,
            "uptime_seconds": time.time() - self._performance_metrics.get("start_time", time.time()),
            "logs_per_second": self._calculate_logs_per_second()
        }
    
    def _calculate_logs_per_second(self) -> float:
        """计算每秒日志数"""
        total_logs = self._performance_metrics["total_logs"]
        last_flush = self._performance_metrics["last_flush_time"]
        
        if last_flush > 0 and total_logs > 0:
            uptime = time.time() - (last_flush - 60)  # 估算运行时间
            return total_logs / max(uptime, 1)
        
        return 0.0
    
    def force_health_check(self):
        """强制执行健康检查"""
        self._perform_health_check()


# 全局实例管理
_global_enhanced_file_logger: Optional[EnhancedFileSystemLogger] = None


def get_enhanced_file_logger() -> Optional[EnhancedFileSystemLogger]:
    """获取全局增强文件系统日志记录器实例"""
    return _global_enhanced_file_logger


def initialize_enhanced_file_logger(log_dir: str = "./logs", **kwargs) -> EnhancedFileSystemLogger:
    """初始化全局增强文件系统日志记录器
    
    Args:
        log_dir: 日志目录
        **kwargs: 其他初始化参数
        
    Returns:
        EnhancedFileSystemLogger: 日志记录器实例
    """
    global _global_enhanced_file_logger
    
    _global_enhanced_file_logger = EnhancedFileSystemLogger(log_dir=log_dir, **kwargs)
    _global_enhanced_file_logger.start()
    
    return _global_enhanced_file_logger


def shutdown_enhanced_file_logger():
    """关闭全局增强文件系统日志记录器"""
    global _global_enhanced_file_logger
    
    if _global_enhanced_file_logger:
        _global_enhanced_file_logger.stop()
        _global_enhanced_file_logger = None