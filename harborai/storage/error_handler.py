"""
增强错误处理和重试机制模块
提供智能的错误分类、重试策略和降级处理
"""

import time
import random
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback
import logging

from ..utils.logger import get_logger
from ..utils.exceptions import StorageError

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = 1      # 轻微错误，可以忽略
    MEDIUM = 2   # 中等错误，需要重试
    HIGH = 3     # 严重错误，需要降级
    CRITICAL = 4 # 致命错误，需要立即处理


class ErrorCategory(Enum):
    """错误类别"""
    CONNECTION = "connection"           # 连接错误
    TIMEOUT = "timeout"                # 超时错误
    AUTHENTICATION = "authentication"  # 认证错误
    PERMISSION = "permission"          # 权限错误
    DATA_VALIDATION = "data_validation" # 数据验证错误
    RESOURCE_EXHAUSTED = "resource_exhausted"  # 资源耗尽
    INTERNAL_ERROR = "internal_error"   # 内部错误
    UNKNOWN = "unknown"                # 未知错误


class RetryStrategy(Enum):
    """重试策略"""
    FIXED_DELAY = "fixed_delay"         # 固定延迟
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # 指数退避
    LINEAR_BACKOFF = "linear_backoff"   # 线性退避
    JITTERED_BACKOFF = "jittered_backoff"  # 带抖动的退避
    NO_RETRY = "no_retry"              # 不重试


@dataclass
class ErrorInfo:
    """错误信息"""
    exception: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    last_retry: Optional[datetime] = None
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class RetryConfig:
    """重试配置"""
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter_range: float = 0.1
    timeout: float = 300.0  # 总超时时间
    
    # 错误类别特定配置
    category_configs: Dict[ErrorCategory, 'RetryConfig'] = field(default_factory=dict)


@dataclass
class ErrorStats:
    """错误统计信息"""
    total_errors: int = 0
    errors_by_category: Dict[ErrorCategory, int] = field(default_factory=dict)
    errors_by_severity: Dict[ErrorSeverity, int] = field(default_factory=dict)
    total_retries: int = 0
    successful_retries: int = 0
    failed_retries: int = 0
    average_retry_delay: float = 0.0
    last_error_time: Optional[datetime] = None
    error_rate: float = 0.0  # 错误率（每分钟）


class ErrorClassifier:
    """错误分类器"""
    
    def __init__(self):
        # 错误模式映射
        self._error_patterns = {
            # 连接错误
            ErrorCategory.CONNECTION: [
                "connection refused", "connection reset", "connection timeout",
                "network unreachable", "host unreachable", "connection failed",
                "psycopg2.OperationalError", "connection closed"
            ],
            
            # 超时错误
            ErrorCategory.TIMEOUT: [
                "timeout", "timed out", "deadline exceeded", "request timeout",
                "read timeout", "write timeout"
            ],
            
            # 认证错误
            ErrorCategory.AUTHENTICATION: [
                "authentication failed", "invalid credentials", "unauthorized",
                "login failed", "password incorrect", "auth error"
            ],
            
            # 权限错误
            ErrorCategory.PERMISSION: [
                "permission denied", "access denied", "forbidden",
                "insufficient privileges", "not authorized"
            ],
            
            # 数据验证错误
            ErrorCategory.DATA_VALIDATION: [
                "validation error", "invalid data", "constraint violation",
                "foreign key", "unique constraint", "check constraint",
                "data too long", "invalid format"
            ],
            
            # 资源耗尽
            ErrorCategory.RESOURCE_EXHAUSTED: [
                "out of memory", "disk full", "too many connections",
                "resource exhausted", "quota exceeded", "rate limit"
            ],
            
            # 内部错误
            ErrorCategory.INTERNAL_ERROR: [
                "internal error", "server error", "unexpected error",
                "assertion error", "null pointer", "index out of bounds"
            ]
        }
        
        # 严重程度映射
        self._severity_mapping = {
            ErrorCategory.CONNECTION: ErrorSeverity.HIGH,
            ErrorCategory.TIMEOUT: ErrorSeverity.MEDIUM,
            ErrorCategory.AUTHENTICATION: ErrorSeverity.HIGH,
            ErrorCategory.PERMISSION: ErrorSeverity.HIGH,
            ErrorCategory.DATA_VALIDATION: ErrorSeverity.LOW,
            ErrorCategory.RESOURCE_EXHAUSTED: ErrorSeverity.CRITICAL,
            ErrorCategory.INTERNAL_ERROR: ErrorSeverity.HIGH,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM
        }
    
    def classify_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorInfo:
        """分类错误
        
        Args:
            exception: 异常对象
            context: 错误上下文
            
        Returns:
            ErrorInfo: 错误信息
        """
        error_message = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        # 确定错误类别
        category = ErrorCategory.UNKNOWN
        for cat, patterns in self._error_patterns.items():
            for pattern in patterns:
                if pattern.lower() in error_message or pattern.lower() in exception_type:
                    category = cat
                    break
            if category != ErrorCategory.UNKNOWN:
                break
        
        # 确定严重程度
        severity = self._severity_mapping.get(category, ErrorSeverity.MEDIUM)
        
        # 根据上下文调整严重程度
        if context:
            if context.get('is_critical_operation', False):
                severity = ErrorSeverity.CRITICAL
            elif context.get('retry_count', 0) > 5:
                severity = ErrorSeverity.HIGH
        
        return ErrorInfo(
            exception=exception,
            category=category,
            severity=severity,
            context=context or {}
        )


class RetryManager:
    """重试管理器"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self._stats = ErrorStats()
        self._lock = threading.RLock()
    
    def should_retry(self, error_info: ErrorInfo) -> bool:
        """判断是否应该重试
        
        Args:
            error_info: 错误信息
            
        Returns:
            bool: 是否应该重试
        """
        # 检查重试次数
        if error_info.retry_count >= error_info.max_retries:
            return False
        
        # 检查错误严重程度
        if error_info.severity == ErrorSeverity.CRITICAL:
            return False
        
        # 检查错误类别
        if error_info.category in [ErrorCategory.AUTHENTICATION, ErrorCategory.PERMISSION]:
            return False
        
        # 检查总超时时间
        if error_info.last_retry:
            total_time = (datetime.now() - error_info.timestamp).total_seconds()
            if total_time > self.config.timeout:
                return False
        
        return True
    
    def calculate_delay(self, error_info: ErrorInfo) -> float:
        """计算重试延迟
        
        Args:
            error_info: 错误信息
            
        Returns:
            float: 延迟时间（秒）
        """
        # 获取配置
        config = self.config.category_configs.get(error_info.category, self.config)
        
        if config.strategy == RetryStrategy.NO_RETRY:
            return 0
        
        elif config.strategy == RetryStrategy.FIXED_DELAY:
            delay = config.base_delay
        
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = config.base_delay * (error_info.retry_count + 1)
        
        elif config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay * (config.backoff_multiplier ** error_info.retry_count)
        
        elif config.strategy == RetryStrategy.JITTERED_BACKOFF:
            base_delay = config.base_delay * (config.backoff_multiplier ** error_info.retry_count)
            jitter = base_delay * config.jitter_range * (random.random() * 2 - 1)
            delay = base_delay + jitter
        
        else:
            delay = config.base_delay
        
        # 限制最大延迟
        delay = min(delay, config.max_delay)
        
        return max(0, delay)
    
    def record_error(self, error_info: ErrorInfo):
        """记录错误
        
        Args:
            error_info: 错误信息
        """
        with self._lock:
            self._stats.total_errors += 1
            self._stats.last_error_time = error_info.timestamp
            
            # 按类别统计
            if error_info.category not in self._stats.errors_by_category:
                self._stats.errors_by_category[error_info.category] = 0
            self._stats.errors_by_category[error_info.category] += 1
            
            # 按严重程度统计
            if error_info.severity not in self._stats.errors_by_severity:
                self._stats.errors_by_severity[error_info.severity] = 0
            self._stats.errors_by_severity[error_info.severity] += 1
            
            # 计算错误率（每分钟）
            if self._stats.total_errors > 1:
                time_span = (datetime.now() - error_info.timestamp).total_seconds() / 60
                if time_span > 0:
                    self._stats.error_rate = self._stats.total_errors / time_span
    
    def record_retry(self, error_info: ErrorInfo, success: bool, delay: float):
        """记录重试
        
        Args:
            error_info: 错误信息
            success: 是否成功
            delay: 延迟时间
        """
        with self._lock:
            self._stats.total_retries += 1
            
            if success:
                self._stats.successful_retries += 1
            else:
                self._stats.failed_retries += 1
            
            # 更新平均延迟
            if self._stats.total_retries > 0:
                old_avg = self._stats.average_retry_delay
                self._stats.average_retry_delay = (
                    (old_avg * (self._stats.total_retries - 1) + delay) /
                    self._stats.total_retries
                )
    
    def get_stats(self) -> ErrorStats:
        """获取错误统计信息"""
        with self._lock:
            return ErrorStats(
                total_errors=self._stats.total_errors,
                errors_by_category=self._stats.errors_by_category.copy(),
                errors_by_severity=self._stats.errors_by_severity.copy(),
                total_retries=self._stats.total_retries,
                successful_retries=self._stats.successful_retries,
                failed_retries=self._stats.failed_retries,
                average_retry_delay=self._stats.average_retry_delay,
                last_error_time=self._stats.last_error_time,
                error_rate=self._stats.error_rate
            )


class EnhancedErrorHandler:
    """增强错误处理器"""
    
    def __init__(self, 
                 retry_config: Optional[RetryConfig] = None,
                 fallback_handler: Optional[Callable[[ErrorInfo], Any]] = None,
                 alert_callback: Optional[Callable[[ErrorInfo], None]] = None):
        """初始化错误处理器
        
        Args:
            retry_config: 重试配置
            fallback_handler: 降级处理函数
            alert_callback: 告警回调函数
        """
        self.classifier = ErrorClassifier()
        self.retry_manager = RetryManager(retry_config)
        self.fallback_handler = fallback_handler
        self.alert_callback = alert_callback
        
        # 错误历史
        self._error_history: List[ErrorInfo] = []
        self._max_history_size = 1000
        self._lock = threading.RLock()
    
    def handle_error(self, 
                    exception: Exception,
                    operation: Callable,
                    context: Optional[Dict[str, Any]] = None,
                    max_retries: Optional[int] = None) -> Any:
        """处理错误
        
        Args:
            exception: 异常对象
            operation: 要重试的操作
            context: 错误上下文
            max_retries: 最大重试次数
            
        Returns:
            操作结果或降级结果
        """
        # 分类错误
        error_info = self.classifier.classify_error(exception, context)
        if max_retries is not None:
            error_info.max_retries = max_retries
        
        # 记录错误
        self.retry_manager.record_error(error_info)
        self._add_to_history(error_info)
        
        # 发送告警
        if self.alert_callback and error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            try:
                self.alert_callback(error_info)
            except Exception as e:
                logger.warning(f"告警回调执行失败: {e}")
        
        # 尝试重试
        while self.retry_manager.should_retry(error_info):
            delay = self.retry_manager.calculate_delay(error_info)
            
            logger.info(f"重试操作，延迟 {delay:.2f} 秒 (第 {error_info.retry_count + 1} 次)")
            
            if delay > 0:
                time.sleep(delay)
            
            error_info.retry_count += 1
            error_info.last_retry = datetime.now()
            
            try:
                result = operation()
                self.retry_manager.record_retry(error_info, True, delay)
                logger.info(f"重试成功 (第 {error_info.retry_count} 次)")
                return result
            
            except Exception as retry_exception:
                self.retry_manager.record_retry(error_info, False, delay)
                
                # 更新错误信息
                new_error_info = self.classifier.classify_error(retry_exception, context)
                new_error_info.retry_count = error_info.retry_count
                new_error_info.max_retries = error_info.max_retries
                new_error_info.timestamp = error_info.timestamp
                error_info = new_error_info
                
                logger.warning(f"重试失败 (第 {error_info.retry_count} 次): {retry_exception}")
        
        # 重试失败，尝试降级处理
        logger.error(f"操作失败，已达到最大重试次数: {error_info.max_retries}")
        
        if self.fallback_handler:
            try:
                logger.info("执行降级处理")
                return self.fallback_handler(error_info)
            except Exception as fallback_exception:
                logger.error(f"降级处理失败: {fallback_exception}")
        
        # 最终失败
        raise StorageError(f"操作失败: {error_info.exception}") from error_info.exception
    
    def _add_to_history(self, error_info: ErrorInfo):
        """添加到错误历史"""
        with self._lock:
            self._error_history.append(error_info)
            
            # 限制历史大小
            if len(self._error_history) > self._max_history_size:
                self._error_history.pop(0)
    
    def get_error_history(self, 
                         category: Optional[ErrorCategory] = None,
                         severity: Optional[ErrorSeverity] = None,
                         limit: int = 100) -> List[ErrorInfo]:
        """获取错误历史
        
        Args:
            category: 错误类别过滤
            severity: 严重程度过滤
            limit: 返回数量限制
            
        Returns:
            List[ErrorInfo]: 错误历史列表
        """
        with self._lock:
            history = self._error_history.copy()
        
        # 过滤
        if category:
            history = [e for e in history if e.category == category]
        
        if severity:
            history = [e for e in history if e.severity == severity]
        
        # 按时间倒序排列
        history.sort(key=lambda x: x.timestamp, reverse=True)
        
        return history[:limit]
    
    def get_error_summary(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """获取错误摘要
        
        Args:
            time_window: 时间窗口
            
        Returns:
            Dict: 错误摘要
        """
        cutoff_time = datetime.now() - time_window
        
        with self._lock:
            recent_errors = [e for e in self._error_history if e.timestamp >= cutoff_time]
        
        summary = {
            'total_errors': len(recent_errors),
            'by_category': {},
            'by_severity': {},
            'most_common_errors': {},
            'error_rate': 0.0
        }
        
        # 按类别统计
        for error in recent_errors:
            category = error.category.value
            if category not in summary['by_category']:
                summary['by_category'][category] = 0
            summary['by_category'][category] += 1
        
        # 按严重程度统计
        for error in recent_errors:
            severity = error.severity.value
            if severity not in summary['by_severity']:
                summary['by_severity'][severity] = 0
            summary['by_severity'][severity] += 1
        
        # 最常见错误
        error_types = {}
        for error in recent_errors:
            error_type = type(error.exception).__name__
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        summary['most_common_errors'] = dict(
            sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        # 错误率
        if time_window.total_seconds() > 0:
            summary['error_rate'] = len(recent_errors) / (time_window.total_seconds() / 60)
        
        return summary
    
    def clear_history(self):
        """清空错误历史"""
        with self._lock:
            self._error_history.clear()
    
    def get_stats(self) -> ErrorStats:
        """获取错误统计信息"""
        return self.retry_manager.get_stats()