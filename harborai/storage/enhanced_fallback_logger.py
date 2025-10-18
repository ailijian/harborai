#!/usr/bin/env python3
"""
增强的日志降级管理器

基于现有FallbackLogger的增强版本，专注于：
- 智能健康检查和自动恢复
- 分布式追踪集成
- 性能监控和指标收集
- 高级错误处理和降级策略

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import asyncio
import os
import time
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .fallback_logger import FallbackLogger, LoggerState
from .optimized_postgresql_logger import OptimizedPostgreSQLLogger
from .enhanced_file_logger import EnhancedFileSystemLogger
from ..core.tracing.dual_trace_manager import DualTraceIDManager, DualTraceContext
from ..core.tracing.data_collector import TracingDataCollector
from ..utils.logger import get_logger
from ..utils.exceptions import StorageError
from ..utils.timestamp import get_unified_timestamp

logger = get_logger(__name__)


@dataclass
class RecoveryStrategy:
    """恢复策略配置"""
    max_retry_attempts: int = 5
    base_retry_delay: float = 2.0  # 基础重试延迟（秒）
    max_retry_delay: float = 300.0  # 最大重试延迟（秒）
    exponential_backoff: bool = True
    jitter_factor: float = 0.1  # 抖动因子
    
    def calculate_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        if self.exponential_backoff:
            delay = self.base_retry_delay * (2 ** attempt)
        else:
            delay = self.base_retry_delay
        
        # 应用最大延迟限制
        delay = min(delay, self.max_retry_delay)
        
        # 添加抖动
        if self.jitter_factor > 0:
            import random
            jitter = delay * self.jitter_factor * random.random()
            delay += jitter
        
        return delay


@dataclass
class HealthMetrics:
    """健康指标"""
    postgres_health: bool = False
    file_health: bool = False
    overall_health: bool = False
    postgres_error_rate: float = 0.0
    file_error_rate: float = 0.0
    postgres_latency_ms: float = 0.0
    file_latency_ms: float = 0.0
    queue_size: int = 0
    disk_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    last_check_time: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            **asdict(self),
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None
        }


class EnhancedFallbackLogger:
    """增强的日志降级管理器
    
    提供智能的PostgreSQL和文件系统之间的自动降级功能，
    包含健康检查、自动恢复、性能监控等高级特性。
    """
    
    def __init__(self,
                 postgres_connection_string: str,
                 log_directory: str = "logs",
                 enable_tracing: bool = True,
                 tracing_sample_rate: float = 1.0,
                 health_check_interval: float = 30.0,
                 recovery_check_interval: float = 120.0,
                 max_postgres_failures: int = 3,
                 max_file_failures: int = 5,
                 postgres_table_name: str = "harborai_logs",
                 file_max_size: int = 100 * 1024 * 1024,  # 100MB
                 file_backup_count: int = 10,
                 postgres_batch_size: int = 50,
                 postgres_flush_interval: float = 5.0,
                 file_batch_size: int = 20,
                 file_flush_interval: float = 10.0,
                 error_callback: Optional[Callable[[Exception], None]] = None):
        """初始化增强的日志降级管理器
        
        Args:
            postgres_connection_string: PostgreSQL连接字符串
            log_directory: 日志文件目录
            enable_tracing: 是否启用分布式追踪
            tracing_sample_rate: 追踪采样率
            health_check_interval: 健康检查间隔（秒）
            recovery_check_interval: 恢复检查间隔（秒）
            max_postgres_failures: PostgreSQL最大失败次数
            max_file_failures: 文件系统最大失败次数
            postgres_table_name: PostgreSQL表名
            file_max_size: 单个日志文件最大大小
            file_backup_count: 日志文件备份数量
            postgres_batch_size: PostgreSQL批处理大小
            postgres_flush_interval: PostgreSQL刷新间隔（秒）
            file_batch_size: 文件系统批处理大小
            file_flush_interval: 文件系统刷新间隔（秒）
            error_callback: 错误回调函数
        """
        self.postgres_connection_string = postgres_connection_string
        self.log_directory = log_directory
        self.enable_tracing = enable_tracing
        self.tracing_sample_rate = tracing_sample_rate
        self.health_check_interval = health_check_interval
        self.recovery_check_interval = recovery_check_interval
        self.max_postgres_failures = max_postgres_failures
        self.max_file_failures = max_file_failures
        self.error_callback = error_callback
        
        # 状态管理
        self._state = LoggerState.INITIALIZING
        self._postgres_failure_count = 0
        self._file_failure_count = 0
        self._last_health_check = 0
        self._last_recovery_check = 0
        
        # 恢复策略
        self._recovery_strategy = RecoveryStrategy()
        self._postgres_recovery_attempts = 0
        self._file_recovery_attempts = 0
        
        # 健康指标
        self._health_metrics = HealthMetrics()
        
        # 性能统计
        self._stats = {
            "postgres_logs": 0,
            "file_logs": 0,
            "postgres_failures": 0,
            "file_failures": 0,
            "state_changes": 0,
            "recovery_attempts": 0,
            "successful_recoveries": 0,
            "total_errors": 0,
            "uptime_start": time.time()
        }
        
        # 日志记录器实例
        self._postgres_logger: Optional[OptimizedPostgreSQLLogger] = None
        self._file_logger: Optional[EnhancedFileSystemLogger] = None
        
        # 追踪组件
        if self.enable_tracing:
            self.dual_trace_manager = DualTraceIDManager()
            self.tracing_collector = None  # 将在初始化时设置
            self.tracer = trace.get_tracer(__name__)
        else:
            self.dual_trace_manager = None
            self.tracing_collector = None
            self.tracer = None
        
        # 健康检查线程
        self._health_check_thread = None
        self._recovery_thread = None
        self._running = False
        
        # 初始化日志记录器
        self._initialize_loggers()
    
    def _initialize_loggers(self):
        """初始化日志记录器"""
        try:
            # 初始化追踪数据收集器
            if self.enable_tracing:
                self.tracing_collector = TracingDataCollector(self.postgres_connection_string)
            
            # 初始化增强文件日志记录器
            self._file_logger = EnhancedFileSystemLogger(
                log_dir=self.log_directory,
                max_file_size=file_max_size,
                max_files=file_backup_count,
                batch_size=file_batch_size,
                flush_interval=file_flush_interval,
                enable_tracing=self.enable_tracing,
                tracing_sample_rate=self.tracing_sample_rate,
                error_callback=self._handle_file_failure
            )
            
            # 设置追踪收集器
            if self.tracing_collector:
                self._file_logger.set_tracing_collector(self.tracing_collector)
            
            self._file_logger.start()
            logger.info("Enhanced file logger initialized successfully")
            
            # 尝试初始化优化的PostgreSQL日志记录器
            try:
                self._postgres_logger = OptimizedPostgreSQLLogger(
                    connection_string=self.postgres_connection_string,
                    table_name=postgres_table_name,
                    batch_size=postgres_batch_size,
                    flush_interval=postgres_flush_interval,
                    enable_tracing=self.enable_tracing,
                    tracing_sample_rate=self.tracing_sample_rate,
                    error_callback=self._handle_postgres_failure
                )
                self._postgres_logger.start()
                self._state = LoggerState.POSTGRES_ACTIVE
                logger.info("Optimized PostgreSQL logger initialized successfully")
                
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL logger: {e}")
                self._handle_postgres_failure(e)
                self._switch_to_file_fallback()
                
        except Exception as e:
            logger.error(f"Failed to initialize enhanced loggers: {e}")
            self._state = LoggerState.ERROR
            raise StorageError(f"Failed to initialize enhanced fallback logger: {e}")
    
    def start(self):
        """启动增强的日志降级管理器"""
        self._running = True
        
        # 启动健康检查线程
        self._start_health_check_thread()
        
        # 启动恢复检查线程
        self._start_recovery_thread()
        
        # 启动追踪数据收集器
        if self.tracing_collector:
            self.tracing_collector.start()
        
        logger.info(
            f"Enhanced fallback logger started in state: {self._state.value}",
            extra={
                "tracing_enabled": self.enable_tracing,
                "health_check_interval": self.health_check_interval,
                "recovery_check_interval": self.recovery_check_interval
            }
        )
    
    def stop(self):
        """停止增强的日志降级管理器"""
        self._running = False
        
        # 停止追踪数据收集器
        if self.tracing_collector:
            self.tracing_collector.stop()
        
        # 停止日志记录器
        if self._postgres_logger:
            self._postgres_logger.stop()
        
        if self._file_logger:
            self._file_logger.stop()
        
        logger.info("Enhanced fallback logger stopped")
    
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
        self._check_health()
        
        dual_context = None
        
        # 创建追踪span
        if self.enable_tracing and self.tracer:
            with self.tracer.start_as_current_span(operation_name) as span:
                try:
                    # 创建双追踪上下文
                    dual_context = self.dual_trace_manager.create_trace_context(
                        hb_trace_id=trace_id,
                        operation_name=operation_name,
                        service_name="harborai-enhanced-logging"
                    )
                    
                    # 设置span属性
                    span.set_attributes({
                        "ai.provider": provider,
                        "ai.model": model,
                        "ai.operation": operation_name,
                        "harborai.trace_id": trace_id,
                        "harborai.message_count": len(messages),
                        "storage.state": self._state.value,
                        "storage.type": "enhanced_fallback"
                    })
                    
                    # 记录请求日志
                    self._log_request_internal(trace_id, model, messages, dual_context, **kwargs)
                    
                    span.set_status(Status(StatusCode.OK))
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    logger.error(
                        "Failed to log request with enhanced tracing",
                        extra={
                            "trace_id": trace_id,
                            "error": str(e),
                            "provider": provider,
                            "model": model,
                            "state": self._state.value
                        }
                    )
                    raise
        else:
            # 不启用追踪时，直接记录日志
            self._log_request_internal(trace_id, model, messages, None, **kwargs)
        
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
        self._check_health()
        
        try:
            if self._state == LoggerState.POSTGRES_ACTIVE and self._postgres_logger:
                if hasattr(self._postgres_logger, 'log_response_with_cost_breakdown'):
                    self._postgres_logger.log_response_with_cost_breakdown(
                        trace_id=trace_id,
                        response=response,
                        dual_context=dual_context,
                        latency=latency,
                        success=success,
                        error=error,
                        **kwargs
                    )
                else:
                    self._postgres_logger.log_response(trace_id, response, latency, success, error)
                self._stats["postgres_logs"] += 1
            else:
                if self._file_logger:
                    if hasattr(self._file_logger, 'log_response_with_tracing'):
                        self._file_logger.log_response_with_tracing(
                            trace_id=trace_id,
                            response=response,
                            latency=latency,
                            success=success,
                            error=error,
                            dual_context=dual_context,
                            **kwargs
                        )
                    else:
                        self._file_logger.log_response(trace_id, response, latency, success, error)
                    self._stats["file_logs"] += 1
                    
        except Exception as e:
            self._handle_logging_failure(e, "response")
            self._attempt_fallback_response_logging(trace_id, response, latency, success, error, dual_context, **kwargs)
    
    def _log_request_internal(self,
                            trace_id: str,
                            model: str,
                            messages: List[Dict[str, Any]],
                            dual_context: Optional[DualTraceContext],
                            **kwargs):
        """内部请求日志记录方法"""
        try:
            if self._state == LoggerState.POSTGRES_ACTIVE and self._postgres_logger:
                if hasattr(self._postgres_logger, 'log_request_with_tracing'):
                    self._postgres_logger.log_request_with_tracing(
                        trace_id=trace_id,
                        model=model,
                        messages=messages,
                        **kwargs
                    )
                else:
                    self._postgres_logger.log_request(trace_id, model, messages, **kwargs)
                self._stats["postgres_logs"] += 1
            else:
                if self._file_logger:
                    if hasattr(self._file_logger, 'log_request_with_tracing'):
                        self._file_logger.log_request_with_tracing(
                            trace_id=trace_id,
                            model=model,
                            messages=messages,
                            **kwargs
                        )
                    else:
                        self._file_logger.log_request(trace_id, model, messages, **kwargs)
                    self._stats["file_logs"] += 1
                    
        except Exception as e:
            self._handle_logging_failure(e, "request")
            self._attempt_fallback_request_logging(trace_id, model, messages, dual_context, **kwargs)
    
    def _attempt_fallback_request_logging(self,
                                        trace_id: str,
                                        model: str,
                                        messages: List[Dict[str, Any]],
                                        dual_context: Optional[DualTraceContext],
                                        **kwargs):
        """尝试降级请求日志记录"""
        try:
            if self._file_logger and self._state != LoggerState.ERROR:
                logger.info(f"Attempting fallback request logging for trace_id: {trace_id}")
                if hasattr(self._file_logger, 'log_request_with_tracing'):
                    self._file_logger.log_request_with_tracing(
                        trace_id=trace_id,
                        model=model,
                        messages=messages,
                        **kwargs
                    )
                else:
                    self._file_logger.log_request(trace_id, model, messages, **kwargs)
                self._stats["file_logs"] += 1
                self._stats["fallback_successes"] = self._stats.get("fallback_successes", 0) + 1
            else:
                logger.warning(f"No fallback logger available for trace_id: {trace_id}")
                self._stats["lost_logs"] = self._stats.get("lost_logs", 0) + 1
        except Exception as fallback_error:
            logger.critical(
                "Both primary and fallback request logging failed",
                extra={
                    "trace_id": trace_id,
                    "fallback_error": str(fallback_error)
                }
            )
            self._stats["lost_logs"] = self._stats.get("lost_logs", 0) + 1
    
    def _attempt_fallback_response_logging(self,
                                         trace_id: str,
                                         response: Any,
                                         latency: float,
                                         success: bool,
                                         error: Optional[str],
                                         dual_context: Optional[DualTraceContext],
                                         **kwargs):
        """尝试降级响应日志记录"""
        try:
            if self._file_logger and self._state != LoggerState.ERROR:
                logger.info(f"Attempting fallback response logging for trace_id: {trace_id}")
                if hasattr(self._file_logger, 'log_response_with_tracing'):
                    self._file_logger.log_response_with_tracing(
                        trace_id=trace_id,
                        response=response,
                        latency=latency,
                        success=success,
                        error=error,
                        dual_context=dual_context,
                        **kwargs
                    )
                else:
                    self._file_logger.log_response(trace_id, response, latency, success, error)
                self._stats["file_logs"] += 1
                self._stats["fallback_successes"] = self._stats.get("fallback_successes", 0) + 1
            else:
                logger.warning(f"No fallback logger available for response trace_id: {trace_id}")
                self._stats["lost_logs"] = self._stats.get("lost_logs", 0) + 1
        except Exception as fallback_error:
            logger.critical(
                "Both primary and fallback response logging failed",
                extra={
                    "trace_id": trace_id,
                    "fallback_error": str(fallback_error)
                }
            )
            self._stats["lost_logs"] = self._stats.get("lost_logs", 0) + 1
    
    def _handle_logging_failure(self, error: Exception, log_type: str):
        """处理日志记录失败"""
        self._stats["total_errors"] += 1
        
        error_type = type(error).__name__
        error_msg = str(error)
        
        logger.error(
            f"Enhanced logging failure in {log_type}",
            extra={
                "error_type": error_type,
                "error_message": error_msg,
                "current_state": self._state.value,
                "log_type": log_type
            }
        )
        
        # 根据当前状态和错误类型决定处理策略
        if self._state == LoggerState.POSTGRES_ACTIVE:
            self._handle_postgres_failure(error)
        elif self._state == LoggerState.FILE_FALLBACK:
            self._handle_file_failure(error)
        
        # 调用错误回调
        if self.error_callback:
            self.error_callback(error)
    
    def _handle_postgres_failure(self, error: Exception):
        """处理PostgreSQL失败"""
        self._postgres_failure_count += 1
        self._stats["postgres_failures"] += 1
        
        logger.error(
            "Enhanced PostgreSQL logger failure",
            extra={
                "error": str(error),
                "failure_count": self._postgres_failure_count,
                "max_failures": self.max_postgres_failures
            }
        )
        
        if self._postgres_failure_count >= self.max_postgres_failures:
            self._switch_to_file_fallback()
    
    def _handle_file_failure(self, error: Exception):
        """处理文件系统失败"""
        self._file_failure_count += 1
        self._stats["file_failures"] += 1
        
        logger.error(
            "Enhanced file logger failure",
            extra={
                "error": str(error),
                "failure_count": self._file_failure_count,
                "max_failures": self.max_file_failures
            }
        )
        
        if self._file_failure_count >= self.max_file_failures:
            self._state = LoggerState.ERROR
            logger.critical("Both PostgreSQL and file logging have failed")
    
    def _switch_to_file_fallback(self):
        """切换到文件系统降级"""
        if self._state != LoggerState.FILE_FALLBACK:
            self._state = LoggerState.FILE_FALLBACK
            self._stats["state_changes"] += 1
            logger.info("Switched to enhanced file fallback mode")
    
    def _check_health(self):
        """检查系统健康状态"""
        current_time = time.time()
        
        if current_time - self._last_health_check < self.health_check_interval:
            return
        
        self._last_health_check = current_time
        self._perform_health_check()
    
    def _perform_health_check(self):
        """执行健康检查"""
        try:
            # 检查PostgreSQL健康状态
            postgres_health = self._check_postgres_health()
            
            # 检查文件系统健康状态
            file_health = self._check_file_health()
            
            # 更新健康指标
            self._health_metrics.postgres_health = postgres_health
            self._health_metrics.file_health = file_health
            self._health_metrics.overall_health = postgres_health or file_health
            self._health_metrics.last_check_time = datetime.now()
            
            # 获取文件系统健康状态详情
            if self._file_logger and hasattr(self._file_logger, 'get_health_status'):
                file_status = self._file_logger.get_health_status()
                self._health_metrics.disk_usage_percent = file_status.get("disk_usage_percent", 0.0)
                self._health_metrics.file_latency_ms = file_status.get("write_latency_ms", 0.0)
                self._health_metrics.queue_size = file_status.get("queue_size", 0)
            
            logger.debug(
                "Enhanced health check completed",
                extra={
                    "postgres_health": postgres_health,
                    "file_health": file_health,
                    "overall_health": self._health_metrics.overall_health,
                    "current_state": self._state.value
                }
            )
            
        except Exception as e:
            logger.error(f"Enhanced health check failed: {e}")
            self._health_metrics.overall_health = False
    
    def _check_postgres_health(self) -> bool:
        """检查PostgreSQL健康状态"""
        if not self._postgres_logger:
            return False
        
        try:
            # 这里可以添加更详细的PostgreSQL健康检查
            # 例如执行简单查询、检查连接池状态等
            return True
        except Exception:
            return False
    
    def _check_file_health(self) -> bool:
        """检查文件系统健康状态"""
        if not self._file_logger:
            return False
        
        try:
            # 检查文件系统健康状态
            if hasattr(self._file_logger, 'get_health_status'):
                status = self._file_logger.get_health_status()
                return status.get("is_healthy", False)
            return True
        except Exception:
            return False
    
    def _start_health_check_thread(self):
        """启动健康检查线程"""
        if self._health_check_thread is None or not self._health_check_thread.is_alive():
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True,
                name="EnhancedFallbackLogger-HealthCheck"
            )
            self._health_check_thread.start()
    
    def _start_recovery_thread(self):
        """启动恢复检查线程"""
        if self._recovery_thread is None or not self._recovery_thread.is_alive():
            self._recovery_thread = threading.Thread(
                target=self._recovery_loop,
                daemon=True,
                name="EnhancedFallbackLogger-Recovery"
            )
            self._recovery_thread.start()
    
    def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                time.sleep(self.health_check_interval)
                
                if not self._running:
                    break
                
                self._perform_health_check()
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                time.sleep(5)
    
    def _recovery_loop(self):
        """恢复检查循环"""
        while self._running:
            try:
                time.sleep(self.recovery_check_interval)
                
                if not self._running:
                    break
                
                self._attempt_recovery()
                
            except Exception as e:
                logger.error(f"Recovery loop error: {e}")
                time.sleep(10)
    
    def _attempt_recovery(self):
        """尝试恢复"""
        current_time = time.time()
        
        if current_time - self._last_recovery_check < self.recovery_check_interval:
            return
        
        self._last_recovery_check = current_time
        
        # 如果当前是文件降级模式，尝试恢复PostgreSQL
        if self._state == LoggerState.FILE_FALLBACK:
            self._attempt_postgres_recovery()
        
        # 如果当前是错误状态，尝试恢复文件系统
        elif self._state == LoggerState.ERROR:
            self._attempt_file_recovery()
    
    def _attempt_postgres_recovery(self):
        """尝试恢复PostgreSQL连接"""
        if self._postgres_recovery_attempts >= self._recovery_strategy.max_retry_attempts:
            logger.debug("PostgreSQL recovery attempts exhausted")
            return
        
        try:
            delay = self._recovery_strategy.calculate_delay(self._postgres_recovery_attempts)
            logger.info(f"Attempting PostgreSQL recovery (attempt {self._postgres_recovery_attempts + 1})")
            
            if self._test_postgres_connection():
                logger.info("PostgreSQL connection recovered, switching back")
                self._postgres_failure_count = 0
                self._postgres_recovery_attempts = 0
                self._state = LoggerState.POSTGRES_ACTIVE
                self._stats["state_changes"] += 1
                self._stats["successful_recoveries"] += 1
            else:
                self._postgres_recovery_attempts += 1
                logger.debug(f"PostgreSQL recovery failed, will retry in {delay:.1f} seconds")
                
        except Exception as e:
            self._postgres_recovery_attempts += 1
            logger.debug(f"PostgreSQL recovery attempt failed: {e}")
        
        self._stats["recovery_attempts"] += 1
    
    def _attempt_file_recovery(self):
        """尝试恢复文件系统"""
        if self._file_recovery_attempts >= self._recovery_strategy.max_retry_attempts:
            logger.debug("File system recovery attempts exhausted")
            return
        
        try:
            delay = self._recovery_strategy.calculate_delay(self._file_recovery_attempts)
            logger.info(f"Attempting file system recovery (attempt {self._file_recovery_attempts + 1})")
            
            if self._test_file_system():
                logger.info("File system recovered, switching to fallback mode")
                self._file_failure_count = 0
                self._file_recovery_attempts = 0
                self._state = LoggerState.FILE_FALLBACK
                self._stats["state_changes"] += 1
                self._stats["successful_recoveries"] += 1
            else:
                self._file_recovery_attempts += 1
                logger.debug(f"File system recovery failed, will retry in {delay:.1f} seconds")
                
        except Exception as e:
            self._file_recovery_attempts += 1
            logger.debug(f"File system recovery attempt failed: {e}")
        
        self._stats["recovery_attempts"] += 1
    
    def _test_postgres_connection(self) -> bool:
        """测试PostgreSQL连接"""
        if not self._postgres_logger:
            return False
        
        try:
            # 这里可以添加更详细的连接测试
            return True
        except Exception:
            return False
    
    def _test_file_system(self) -> bool:
        """测试文件系统"""
        if not self._file_logger:
            return False
        
        try:
            # 检查文件系统是否可用
            return os.access(self.log_directory, os.W_OK)
        except Exception:
            return False
    
    def get_state(self) -> LoggerState:
        """获取当前状态"""
        return self._state
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """获取健康指标"""
        return self._health_metrics.to_dict()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        uptime = time.time() - self._stats["uptime_start"]
        
        return {
            **self._stats,
            "current_state": self._state.value,
            "postgres_failure_count": self._postgres_failure_count,
            "file_failure_count": self._file_failure_count,
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "logs_per_second": (self._stats["postgres_logs"] + self._stats["file_logs"]) / max(uptime, 1),
            "error_rate": self._stats["total_errors"] / max(uptime, 1),
            "recovery_success_rate": (
                self._stats["successful_recoveries"] / max(self._stats["recovery_attempts"], 1) * 100
            )
        }
    
    def force_postgres_recovery(self):
        """强制尝试PostgreSQL恢复"""
        self._postgres_recovery_attempts = 0
        self._attempt_postgres_recovery()
        logger.info("Forced PostgreSQL recovery attempt")
    
    def force_file_recovery(self):
        """强制尝试文件系统恢复"""
        self._file_recovery_attempts = 0
        self._attempt_file_recovery()
        logger.info("Forced file system recovery attempt")
    
    def force_health_check(self):
        """强制执行健康检查"""
        self._perform_health_check()
    
    def reset_failure_counts(self):
        """重置失败计数"""
        self._postgres_failure_count = 0
        self._file_failure_count = 0
        self._postgres_recovery_attempts = 0
        self._file_recovery_attempts = 0
        logger.info("Failure counts reset")


# 全局实例管理
_global_enhanced_fallback_logger: Optional[EnhancedFallbackLogger] = None


def get_enhanced_fallback_logger() -> Optional[EnhancedFallbackLogger]:
    """获取全局增强降级日志记录器实例"""
    return _global_enhanced_fallback_logger


def initialize_enhanced_fallback_logger(postgres_connection_string: str, **kwargs) -> EnhancedFallbackLogger:
    """初始化全局增强降级日志记录器
    
    Args:
        postgres_connection_string: PostgreSQL连接字符串
        **kwargs: 其他初始化参数
        
    Returns:
        EnhancedFallbackLogger: 日志记录器实例
    """
    global _global_enhanced_fallback_logger
    
    if _global_enhanced_fallback_logger:
        _global_enhanced_fallback_logger.stop()
    
    _global_enhanced_fallback_logger = EnhancedFallbackLogger(postgres_connection_string, **kwargs)
    _global_enhanced_fallback_logger.start()
    
    return _global_enhanced_fallback_logger


def shutdown_enhanced_fallback_logger():
    """关闭全局增强降级日志记录器"""
    global _global_enhanced_fallback_logger
    
    if _global_enhanced_fallback_logger:
        _global_enhanced_fallback_logger.stop()
        _global_enhanced_fallback_logger = None