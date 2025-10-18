"""日志降级管理器。"""

import os
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger
from ..utils.exceptions import StorageError
from ..utils.timestamp import create_timestamp_context, validate_timestamp_order
from .file_logger import FileSystemLogger
from .postgres_logger import PostgreSQLLogger

logger = get_logger(__name__)


class LoggerState(Enum):
    """日志记录器状态枚举。"""
    POSTGRES_ACTIVE = "postgres_active"
    FILE_FALLBACK = "file_fallback"
    INITIALIZING = "initializing"
    ERROR = "error"


class FallbackLogger:
    """日志降级管理器。
    
    提供PostgreSQL和文件系统之间的自动降级功能。
    当PostgreSQL不可用时，自动切换到文件系统日志记录。
    """
    
    def __init__(self,
                 postgres_connection_string: str,
                 log_directory: str = "logs",
                 max_postgres_failures: int = 3,
                 health_check_interval: float = 60.0,
                 postgres_table_name: str = "harborai_logs",
                 file_max_size: int = 100 * 1024 * 1024,  # 100MB
                 file_backup_count: int = 5,
                 postgres_batch_size: int = 10,
                 postgres_flush_interval: float = 5.0):
        """初始化日志降级管理器。
        
        Args:
            postgres_connection_string: PostgreSQL连接字符串
            log_directory: 日志文件目录
            max_postgres_failures: PostgreSQL最大失败次数
            health_check_interval: 健康检查间隔（秒）
            postgres_table_name: PostgreSQL表名
            file_max_size: 单个日志文件最大大小
            file_backup_count: 日志文件备份数量
            postgres_batch_size: PostgreSQL批处理大小
            postgres_flush_interval: PostgreSQL刷新间隔（秒）
        """
        self.postgres_connection_string = postgres_connection_string
        self.log_directory = log_directory
        self.max_postgres_failures = max_postgres_failures
        self.health_check_interval = health_check_interval
        self.postgres_table_name = postgres_table_name
        self.file_max_size = file_max_size
        self.file_backup_count = file_backup_count
        self.postgres_batch_size = postgres_batch_size
        self.postgres_flush_interval = postgres_flush_interval
        
        # 状态管理
        self._state = LoggerState.INITIALIZING
        self._postgres_failure_count = 0
        self._last_health_check = 0
        self._stats = {
            "postgres_logs": 0,
            "file_logs": 0,
            "postgres_failures": 0,
            "state_changes": 0
        }
        
        # 日志记录器实例
        self._postgres_logger: Optional[PostgreSQLLogger] = None
        self._file_logger: Optional[FileSystemLogger] = None
        
        # 时间戳上下文管理器缓存
        self._timestamp_contexts: Dict[str, Any] = {}
        
        # 初始化日志记录器
        self._initialize_loggers()
    
    def _initialize_loggers(self):
        """初始化日志记录器。"""
        try:
            # 初始化文件日志记录器
            self._file_logger = FileSystemLogger(
                log_dir=self.log_directory,
                max_file_size=self.file_max_size,
                max_files=self.file_backup_count,
                batch_size=1,  # 设置为1以便立即写入，方便测试
                flush_interval=1  # 设置为1秒以便快速刷新
            )
            self._file_logger.start()
            logger.info("File logger initialized successfully")
            
            # 尝试初始化PostgreSQL日志记录器
            try:
                self._postgres_logger = PostgreSQLLogger(
                    connection_string=self.postgres_connection_string,
                    table_name=self.postgres_table_name,
                    batch_size=self.postgres_batch_size,
                    flush_interval=self.postgres_flush_interval,
                    error_callback=self._handle_postgres_failure  # 传递错误回调
                )
                self._postgres_logger.start()
                self._state = LoggerState.POSTGRES_ACTIVE
                logger.info("PostgreSQL logger initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL logger: {e}")
                self._handle_postgres_failure(e)
                # 在初始化阶段，如果PostgreSQL失败，立即切换到文件降级模式
                self._switch_to_file_fallback()
                
        except Exception as e:
            logger.error(f"Failed to initialize loggers: {e}")
            self._state = LoggerState.ERROR
            raise StorageError(f"Failed to initialize fallback logger: {e}")
    
    def start(self):
        """启动日志降级管理器。"""
        logger.info(f"Fallback logger started in state: {self._state.value}")
    
    def stop(self):
        """停止日志降级管理器。"""
        if self._postgres_logger:
            self._postgres_logger.stop()
        
        if self._file_logger:
            self._file_logger.stop()
        
        logger.info("Fallback logger stopped")
    
    def log_request(self,
                   trace_id: str,
                   model: str,
                   messages: List[Dict[str, Any]],
                   **kwargs):
        """记录请求日志。
        
        Args:
            trace_id: 追踪ID
            model: 模型名称
            messages: 消息列表
            **kwargs: 其他参数
        """
        self._check_health()
        
        # 创建时间戳上下文管理器
        if trace_id not in self._timestamp_contexts:
            self._timestamp_contexts[trace_id] = create_timestamp_context(trace_id)
        
        # 标记请求时间戳
        timestamp_context = self._timestamp_contexts[trace_id]
        timestamp_context.mark_request()
        
        try:
            if self._state == LoggerState.POSTGRES_ACTIVE and self._postgres_logger:
                self._postgres_logger.log_request(trace_id, model, messages, **kwargs)
                self._stats["postgres_logs"] += 1
            else:
                if self._file_logger:
                    self._file_logger.log_request(trace_id, model, messages, **kwargs)
                    self._stats["file_logs"] += 1
        except Exception as e:
            # 详细的错误处理和分类
            error_context = {
                "trace_id": trace_id,
                "model": model,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "current_state": self._state.value,
                "timestamp": time.time()
            }
            
            logger.error("Failed to log request", extra=error_context)
            self._handle_logging_failure(e)
            
            # 尝试使用备用日志记录器，增强错误处理
            try:
                if self._file_logger and self._state != LoggerState.ERROR:
                    logger.info(f"Attempting fallback logging for trace_id: {trace_id}")
                    self._file_logger.log_request(trace_id, model, messages, **kwargs)
                    self._stats["file_logs"] += 1
                    self._stats["fallback_successes"] = self._stats.get("fallback_successes", 0) + 1
                else:
                    logger.warning(f"No fallback logger available for trace_id: {trace_id}")
                    self._stats["lost_logs"] = self._stats.get("lost_logs", 0) + 1
            except Exception as fallback_error:
                fallback_context = {
                    "trace_id": trace_id,
                    "original_error": str(e),
                    "fallback_error_type": type(fallback_error).__name__,
                    "fallback_error_message": str(fallback_error),
                    "timestamp": time.time()
                }
                logger.critical("Both primary and fallback logging failed", extra=fallback_context)
                self._stats["lost_logs"] = self._stats.get("lost_logs", 0) + 1
    
    def log_response(self,
                    trace_id: str,
                    response: Any,
                    latency: float,
                    success: bool = True,
                    error: Optional[str] = None):
        """记录响应日志。
        
        Args:
            trace_id: 追踪ID
            response: 响应对象
            latency: 延迟时间
            success: 是否成功
            error: 错误信息
        """
        self._check_health()
        
        # 获取时间戳上下文管理器并标记响应时间戳
        if trace_id in self._timestamp_contexts:
            timestamp_context = self._timestamp_contexts[trace_id]
            timestamp_context.mark_response()
            
            # 清理上下文缓存（避免内存泄漏）
            del self._timestamp_contexts[trace_id]
        else:
            logger.warning(f"未找到 trace_id {trace_id} 的请求时间戳上下文")
        
        try:
            if self._state == LoggerState.POSTGRES_ACTIVE and self._postgres_logger:
                self._postgres_logger.log_response(trace_id, response, latency, success, error)
                self._stats["postgres_logs"] += 1
            else:
                if self._file_logger:
                    self._file_logger.log_response(trace_id, response, latency, success, error)
                    self._stats["file_logs"] += 1
        except Exception as e:
            # 详细的错误处理
            error_context = {
                "trace_id": trace_id,
                "latency": latency,
                "success": success,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "current_state": self._state.value,
                "timestamp": time.time()
            }
            
            logger.error("Failed to log response", extra=error_context)
            self._handle_logging_failure(e)
            
            # 尝试使用备用日志记录器
            try:
                if self._file_logger and self._state != LoggerState.ERROR:
                    logger.info(f"Attempting fallback response logging for trace_id: {trace_id}")
                    self._file_logger.log_response(trace_id, response, latency, success, error)
                    self._stats["file_logs"] += 1
                    self._stats["fallback_successes"] = self._stats.get("fallback_successes", 0) + 1
                else:
                    logger.warning(f"No fallback logger available for response trace_id: {trace_id}")
                    self._stats["lost_logs"] = self._stats.get("lost_logs", 0) + 1
            except Exception as fallback_error:
                fallback_context = {
                    "trace_id": trace_id,
                    "original_error": str(e),
                    "fallback_error_type": type(fallback_error).__name__,
                    "fallback_error_message": str(fallback_error),
                    "timestamp": time.time()
                }
                logger.critical("Both primary and fallback response logging failed", extra=fallback_context)
                self._stats["lost_logs"] = self._stats.get("lost_logs", 0) + 1
    
    def _handle_logging_failure(self, error: Exception):
        """处理日志记录失败。
        
        Args:
            error: 异常对象
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        # 增强错误分析和处理
        if self._state == LoggerState.POSTGRES_ACTIVE:
            # PostgreSQL相关错误
            if any(keyword in error_msg.lower() for keyword in ['connection', 'timeout', 'network']):
                logger.warning("PostgreSQL connection issue detected, switching to file fallback")
                self._switch_to_file_fallback()
                self._handle_postgres_failure(error)
            elif any(keyword in error_msg.lower() for keyword in ['disk', 'space', 'permission']):
                logger.error("PostgreSQL storage issue detected")
                self._switch_to_file_fallback()
            else:
                logger.error(f"Unknown PostgreSQL error: {error_type} - {error_msg}")
                self._handle_postgres_failure(error)
        else:
            # 文件日志相关错误
            if any(keyword in error_msg.lower() for keyword in ['disk', 'space', 'permission']):
                logger.critical("File logging also failing due to storage issues")
                self._state = LoggerState.ERROR
            else:
                logger.error(f"File logging error: {error_type} - {error_msg}")
    
    def _handle_postgres_failure(self, error: Exception):
        """处理PostgreSQL失败。
        
        Args:
            error: 异常对象
        """
        self._postgres_failure_count += 1
        self._stats["postgres_failures"] += 1
        
        # 详细的错误分类和处理
        error_type = type(error).__name__
        error_msg = str(error)
        
        # 记录详细的错误信息
        logger.error(
            "PostgreSQL logger failure detected",
            extra={
                "error_type": error_type,
                "error_message": error_msg,
                "failure_count": self._stats["postgres_failures"],
                "current_state": self._state.value,
                "timestamp": time.time()
            }
        )
        
        # 检查是否达到失败阈值
        if self._postgres_failure_count >= self.max_postgres_failures:
            logger.warning(f"PostgreSQL failure count ({self._postgres_failure_count}) reached threshold ({self.max_postgres_failures}), switching to file fallback")
            self._switch_to_file_fallback()
            return
        
        # 根据错误类型决定处理策略
        if "connection" in error_msg.lower() or "timeout" in error_msg.lower():
            # 连接相关错误，可能是临时的
            logger.warning("Connection-related error detected, will attempt recovery")
            self._schedule_postgres_recovery()
        elif "authentication" in error_msg.lower() or "permission" in error_msg.lower():
            # 认证或权限错误，通常是配置问题
            logger.error("Authentication/permission error detected, manual intervention required")
            self._state = LoggerState.ERROR
        else:
            # 其他错误，尝试恢复
            logger.warning("Unknown PostgreSQL error, attempting recovery")
            self._schedule_postgres_recovery()
    
    def _schedule_postgres_recovery(self):
        """安排PostgreSQL恢复尝试"""
        # 实现指数退避策略
        recovery_delay = min(60, 2 ** min(self._stats["postgres_failures"], 6))
        logger.info(f"Scheduling PostgreSQL recovery in {recovery_delay} seconds")
        
        # 这里可以添加定时器或异步任务来尝试恢复
        # 为了简化，暂时只记录日志

    def _switch_to_file_fallback(self):
        """切换到文件系统降级。"""
        if self._state != LoggerState.FILE_FALLBACK:
            self._state = LoggerState.FILE_FALLBACK
            self._stats["state_changes"] += 1
            logger.info("Switched to file fallback mode")
    
    def _check_health(self):
        """检查系统健康状态。"""
        current_time = time.time()
        
        if current_time - self._last_health_check < self.health_check_interval:
            return
        
        self._last_health_check = current_time
        
        # 如果当前是文件降级模式，尝试恢复PostgreSQL
        if self._state == LoggerState.FILE_FALLBACK:
            self._attempt_postgres_recovery()
    
    def _attempt_postgres_recovery(self):
        """尝试恢复PostgreSQL连接。"""
        try:
            if self._test_postgres_connection():
                logger.info("PostgreSQL connection recovered, switching back")
                self._postgres_failure_count = 0
                self._state = LoggerState.POSTGRES_ACTIVE
                self._stats["state_changes"] += 1
        except Exception as e:
            logger.debug(f"PostgreSQL recovery failed: {e}")
    
    def _test_postgres_connection(self) -> bool:
        """测试PostgreSQL连接。"""
        if not self._postgres_logger:
            return False
        
        try:
            # 这里可以添加一个简单的连接测试
            # 例如执行一个简单的查询
            return True
        except Exception:
            return False
    
    def get_state(self) -> LoggerState:
        """获取当前状态。"""
        return self._state
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息。"""
        return {
            **self._stats,
            "current_state": self._state.value,
            "postgres_failure_count": self._postgres_failure_count,
            "last_health_check": datetime.fromtimestamp(self._last_health_check).isoformat() if self._last_health_check else None
        }
    
    def force_fallback(self):
        """强制切换到文件降级模式。"""
        self._switch_to_file_fallback()
        logger.info("Forced switch to file fallback mode")
    
    def force_recovery(self):
        """强制尝试恢复PostgreSQL。"""
        self._postgres_failure_count = 0
        self._attempt_postgres_recovery()
        logger.info("Forced PostgreSQL recovery attempt")


# 全局降级日志记录器实例
_global_fallback_logger: Optional[FallbackLogger] = None


def get_fallback_logger() -> Optional[FallbackLogger]:
    """获取全局降级日志记录器。"""
    return _global_fallback_logger


def initialize_fallback_logger(postgres_connection_string: str, **kwargs) -> FallbackLogger:
    """初始化全局降级日志记录器。"""
    global _global_fallback_logger
    
    if _global_fallback_logger:
        _global_fallback_logger.stop()
    
    _global_fallback_logger = FallbackLogger(postgres_connection_string, **kwargs)
    _global_fallback_logger.start()
    
    return _global_fallback_logger


def shutdown_fallback_logger():
    """关闭全局降级日志记录器。"""
    global _global_fallback_logger
    
    if _global_fallback_logger:
        _global_fallback_logger.stop()
        _global_fallback_logger = None