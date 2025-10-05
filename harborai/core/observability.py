#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 可观测性模块

提供统一的可观测性接口，包括日志记录、指标收集、链路追踪、告警管理、性能监控等功能。
整合现有的logger、tracer、metrics等组件，为测试和生产环境提供完整的可观测性支持。
"""

import time
import uuid
import asyncio
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging

# 导入现有组件
from ..utils.logger import get_logger, APICallLogger, LogContext, sanitize_log_data
from ..utils.tracer import (
    generate_trace_id, get_current_trace_id, set_current_trace_id,
    TraceContext, SpanTimer, trace_function
)
from ..monitoring.prometheus_metrics import PrometheusMetrics
from ..storage.postgres_logger import PostgreSQLLogger
from ..config.settings import get_settings


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """告警严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime
    level: LogLevel
    message: str
    logger_name: str
    module: str
    function: str
    line_number: int
    thread_id: str
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    stack_trace: Optional[str] = None


@dataclass
class Metric:
    """指标数据"""
    name: str
    type: MetricType
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    description: Optional[str] = None
    unit: Optional[str] = None


@dataclass
class Span:
    """追踪跨度"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]  # 毫秒
    status: str  # "ok", "error", "timeout"
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """告警信息"""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_message: Optional[str] = None


class Logger:
    """统一日志记录器"""
    
    def __init__(self, name: str = "harborai"):
        self.name = name
        self._structlog_logger = get_logger(name)
        self._api_logger = APICallLogger(self._structlog_logger)
        self.logs: List[LogEntry] = []
        self.handlers: List[Callable] = []
        self.level = LogLevel.INFO
        self.structured_logging = True
        self.context = {}
        self._lock = threading.Lock()
    
    def set_level(self, level: LogLevel):
        """设置日志级别"""
        self.level = level
    
    def add_handler(self, handler: Callable):
        """添加日志处理器"""
        self.handlers.append(handler)
    
    def set_context(self, **kwargs):
        """设置日志上下文"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """清除日志上下文"""
        self.context.clear()
    
    def _should_log(self, level: LogLevel) -> bool:
        """检查是否应该记录日志"""
        level_order = [LogLevel.DEBUG, LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
        return level_order.index(level) >= level_order.index(self.level)
    
    def _create_log_entry(self, level: LogLevel, message: str, **kwargs) -> LogEntry:
        """创建日志条目"""
        import inspect
        frame = inspect.currentframe().f_back.f_back
        
        entry = LogEntry(
            timestamp=datetime.now(),
            level=level,
            message=message,
            logger_name=self.name,
            module=frame.f_globals.get('__name__', 'unknown'),
            function=frame.f_code.co_name,
            line_number=frame.f_lineno,
            thread_id=str(threading.current_thread().ident),
            extra_fields={**self.context, **kwargs}
        )
        
        # 添加追踪信息（如果存在）
        if 'trace_id' in self.context:
            entry.trace_id = self.context['trace_id']
        if 'span_id' in self.context:
            entry.span_id = self.context['span_id']
        
        return entry
    
    def _notify_handlers(self, entry: LogEntry):
        """通知所有处理器"""
        for handler in self.handlers:
            try:
                handler(entry)
            except Exception as e:
                # 避免处理器错误影响日志记录
                pass
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        if self._should_log(LogLevel.DEBUG):
            entry = self._create_log_entry(LogLevel.DEBUG, message, **kwargs)
            with self._lock:
                self.logs.append(entry)
            self._notify_handlers(entry)
            self._structlog_logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        if self._should_log(LogLevel.INFO):
            entry = self._create_log_entry(LogLevel.INFO, message, **kwargs)
            with self._lock:
                self.logs.append(entry)
            self._notify_handlers(entry)
            self._structlog_logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        if self._should_log(LogLevel.WARNING):
            entry = self._create_log_entry(LogLevel.WARNING, message, **kwargs)
            with self._lock:
                self.logs.append(entry)
            self._notify_handlers(entry)
            self._structlog_logger.warning(message, **kwargs)
    
    def error(self, message: str, exception: Exception = None, **kwargs):
        """记录错误日志"""
        if self._should_log(LogLevel.ERROR):
            entry = self._create_log_entry(LogLevel.ERROR, message, **kwargs)
            if exception:
                entry.exception = str(exception)
                entry.stack_trace = self._get_stack_trace(exception)
            with self._lock:
                self.logs.append(entry)
            self._notify_handlers(entry)
            self._structlog_logger.error(message, exc_info=exception, **kwargs)
    
    def critical(self, message: str, exception: Exception = None, **kwargs):
        """记录严重错误日志"""
        if self._should_log(LogLevel.CRITICAL):
            entry = self._create_log_entry(LogLevel.CRITICAL, message, **kwargs)
            if exception:
                entry.exception = str(exception)
                entry.stack_trace = self._get_stack_trace(exception)
            with self._lock:
                self.logs.append(entry)
            self._notify_handlers(entry)
            self._structlog_logger.critical(message, exc_info=exception, **kwargs)
    
    def _get_stack_trace(self, exception: Exception) -> str:
        """获取异常堆栈跟踪"""
        import traceback
        return ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    
    def get_logs(self, level: Optional[LogLevel] = None, limit: Optional[int] = None) -> List[LogEntry]:
        """获取日志条目"""
        with self._lock:
            logs = self.logs.copy()
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        if limit:
            logs = logs[-limit:]
        
        return logs
    
    def clear_logs(self):
        """清除日志条目"""
        with self._lock:
            self.logs.clear()


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.prometheus_metrics = None
        self._lock = threading.Lock()
        
        try:
            self.prometheus_metrics = PrometheusMetrics()
        except Exception as e:
            # 如果Prometheus不可用，继续使用内存存储
            pass
    
    def record_counter(self, name: str, value: Union[int, float] = 1, 
                      labels: Dict[str, str] = None, **kwargs):
        """记录计数器指标"""
        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            **kwargs
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def record_gauge(self, name: str, value: Union[int, float], 
                    labels: Dict[str, str] = None, **kwargs):
        """记录仪表盘指标"""
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            **kwargs
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def record_histogram(self, name: str, value: Union[int, float], 
                        labels: Dict[str, str] = None, **kwargs):
        """记录直方图指标"""
        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            timestamp=datetime.now(),
            labels=labels or {},
            **kwargs
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, List[Metric]]:
        """获取指标数据"""
        with self._lock:
            if name:
                return {name: self.metrics.get(name, [])}
            return dict(self.metrics)
    
    def clear_metrics(self):
        """清除指标数据"""
        with self._lock:
            self.metrics.clear()


class TracingManager:
    """链路追踪管理器"""
    
    def __init__(self):
        self.spans: Dict[str, List[Span]] = defaultdict(list)
        self.active_spans: Dict[str, Span] = {}
        self._lock = threading.Lock()
    
    def start_trace(self, operation_name: str, trace_id: Optional[str] = None) -> str:
        """开始一个新的追踪"""
        if not trace_id:
            trace_id = generate_trace_id()
            set_current_trace_id(trace_id)
        
        # 创建根跨度
        span_id = self.start_span(operation_name, trace_id=trace_id)
        return trace_id
    
    def start_span(self, operation_name: str, trace_id: Optional[str] = None, 
                  parent_span_id: Optional[str] = None, **tags) -> str:
        """开始一个新的跨度"""
        if not trace_id:
            trace_id = get_current_trace_id() or generate_trace_id()
            set_current_trace_id(trace_id)
        
        span_id = str(uuid.uuid4())
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            status="ok",
            tags=tags
        )
        
        with self._lock:
            self.active_spans[span_id] = span
            self.spans[trace_id].append(span)
        
        return span_id
    
    def get_active_span_id(self) -> Optional[str]:
        """获取当前活跃的跨度ID"""
        with self._lock:
            if self.active_spans:
                # 返回最新的活跃跨度ID
                return list(self.active_spans.keys())[-1]
            return None
    
    def finish_span(self, span_or_id, status: str = "ok"):
        """结束跨度"""
        if isinstance(span_or_id, str):
            # 如果传入的是span_id
            span_id = span_or_id
            with self._lock:
                if span_id in self.active_spans:
                    span = self.active_spans[span_id]
                else:
                    return  # 跨度不存在
        else:
            # 如果传入的是Span对象
            span = span_or_id
            span_id = span.span_id
        
        span.end_time = datetime.now()
        span.duration = (span.end_time - span.start_time).total_seconds() * 1000
        span.status = status
        
        with self._lock:
            if span_id in self.active_spans:
                del self.active_spans[span_id]
    
    def add_span_log(self, span_or_id, **log_data):
        """添加跨度日志"""
        if isinstance(span_or_id, str):
            # 如果传入的是span_id
            span_id = span_or_id
            with self._lock:
                if span_id in self.active_spans:
                    span = self.active_spans[span_id]
                else:
                    return  # 跨度不存在
        else:
            # 如果传入的是Span对象
            span = span_or_id
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            **log_data
        }
        span.logs.append(log_entry)
    
    def set_span_tag(self, span_or_id, key: str, value: Any):
        """设置跨度标签"""
        if isinstance(span_or_id, str):
            # 如果传入的是span_id
            span_id = span_or_id
            with self._lock:
                if span_id in self.active_spans:
                    span = self.active_spans[span_id]
                else:
                    return  # 跨度不存在
        else:
            # 如果传入的是Span对象
            span = span_or_id
        
        span.tags[key] = value
    
    def get_spans(self, trace_id: Optional[str] = None) -> Dict[str, List[Span]]:
        """获取跨度数据"""
        with self._lock:
            if trace_id:
                return {trace_id: self.spans.get(trace_id, [])}
            return dict(self.spans)
    
    def clear_spans(self):
        """清除跨度数据"""
        with self._lock:
            self.spans.clear()
            self.active_spans.clear()


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.alerts: List[Alert] = []
        self.alert_rules: List[Callable] = []
        self._lock = threading.Lock()
    
    def create_alert(self, name: str, severity: AlertSeverity, message: str, 
                    source: str, labels: Dict[str, str] = None) -> Alert:
        """创建告警"""
        alert = Alert(
            id=str(uuid.uuid4()),
            name=name,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            source=source,
            labels=labels or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
        
        return alert
    
    def resolve_alert(self, alert_id: str, resolution_message: str = None):
        """解决告警"""
        with self._lock:
            for alert in self.alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    alert.resolved_at = datetime.now()
                    alert.resolution_message = resolution_message
                    break
    
    def add_alert_rule(self, rule: Callable):
        """添加告警规则"""
        self.alert_rules.append(rule)
    
    def evaluate_rules(self, metrics: Dict[str, Any]):
        """评估告警规则"""
        for rule in self.alert_rules:
            try:
                rule(metrics, self)
            except Exception as e:
                # 记录规则评估错误，但不影响其他规则
                pass
    
    def get_alerts(self, resolved: Optional[bool] = None) -> List[Alert]:
        """获取告警列表"""
        with self._lock:
            alerts = self.alerts.copy()
        
        if resolved is not None:
            alerts = [alert for alert in alerts if alert.resolved == resolved]
        
        return alerts
    
    def clear_alerts(self):
        """清除告警"""
        with self._lock:
            self.alerts.clear()


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
    
    def record_latency(self, operation: str, latency_ms: float, **labels):
        """记录延迟"""
        data = {
            "timestamp": datetime.now(),
            "latency_ms": latency_ms,
            "labels": labels
        }
        
        with self._lock:
            self.performance_data[f"{operation}_latency"].append(data)
    
    def record_throughput(self, operation: str, count: int, **labels):
        """记录吞吐量"""
        data = {
            "timestamp": datetime.now(),
            "count": count,
            "labels": labels
        }
        
        with self._lock:
            self.performance_data[f"{operation}_throughput"].append(data)
    
    def record_resource_usage(self, resource_type: str, usage: float, **labels):
        """记录资源使用情况"""
        data = {
            "timestamp": datetime.now(),
            "usage": usage,
            "labels": labels
        }
        
        with self._lock:
            self.performance_data[f"{resource_type}_usage"].append(data)
    
    def get_performance_data(self, metric_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """获取性能数据"""
        with self._lock:
            if metric_name:
                return {metric_name: list(self.performance_data.get(metric_name, []))}
            return {k: list(v) for k, v in self.performance_data.items()}
    
    def get_average_latency(self, operation: str, time_window_minutes: int = 5) -> Optional[float]:
        """获取平均延迟"""
        metric_name = f"{operation}_latency"
        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            data = self.performance_data.get(metric_name, [])
            recent_data = [d for d in data if d["timestamp"] >= cutoff_time]
            
            if not recent_data:
                return None
            
            return sum(d["latency_ms"] for d in recent_data) / len(recent_data)
    
    def clear_performance_data(self):
        """清除性能数据"""
        with self._lock:
            self.performance_data.clear()


class ObservabilityExporter:
    """可观测性数据导出器"""
    
    def __init__(self):
        self.exporters: Dict[str, Callable] = {}
    
    def register_exporter(self, name: str, exporter: Callable):
        """注册导出器"""
        self.exporters[name] = exporter
    
    def export_logs(self, logs: List[LogEntry], format: str = "json") -> str:
        """导出日志数据"""
        if format == "json":
            return json.dumps([
                {
                    "timestamp": log.timestamp.isoformat(),
                    "level": log.level.value,
                    "message": log.message,
                    "logger_name": log.logger_name,
                    "module": log.module,
                    "function": log.function,
                    "line_number": log.line_number,
                    "thread_id": log.thread_id,
                    "trace_id": log.trace_id,
                    "span_id": log.span_id,
                    "extra_fields": log.extra_fields,
                    "exception": log.exception,
                    "stack_trace": log.stack_trace
                }
                for log in logs
            ], indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_metrics(self, metrics: Dict[str, List[Metric]], format: str = "json") -> str:
        """导出指标数据"""
        if format == "json":
            result = {}
            for name, metric_list in metrics.items():
                result[name] = [
                    {
                        "name": metric.name,
                        "type": metric.type.value,
                        "value": metric.value,
                        "timestamp": metric.timestamp.isoformat(),
                        "labels": metric.labels,
                        "description": metric.description,
                        "unit": metric.unit
                    }
                    for metric in metric_list
                ]
            return json.dumps(result, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_spans(self, spans: Dict[str, List[Span]], format: str = "json") -> str:
        """导出跨度数据"""
        if format == "json":
            result = {}
            for trace_id, span_list in spans.items():
                result[trace_id] = [
                    {
                        "trace_id": span.trace_id,
                        "span_id": span.span_id,
                        "parent_span_id": span.parent_span_id,
                        "operation_name": span.operation_name,
                        "start_time": span.start_time.isoformat(),
                        "end_time": span.end_time.isoformat() if span.end_time else None,
                        "duration": span.duration,
                        "status": span.status,
                        "tags": span.tags,
                        "logs": span.logs,
                        "baggage": span.baggage
                    }
                    for span in span_list
                ]
            return json.dumps(result, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def export_all(self, logger: Logger, metrics_collector: MetricsCollector, 
                  tracing_manager: TracingManager, format: str = "json") -> Dict[str, str]:
        """导出所有可观测性数据"""
        return {
            "logs": self.export_logs(logger.get_logs(), format),
            "metrics": self.export_metrics(metrics_collector.get_metrics(), format),
            "spans": self.export_spans(tracing_manager.get_spans(), format)
        }


# 全局实例（用于测试和简单使用场景）
_global_logger = None
_global_metrics_collector = None
_global_tracing_manager = None
_global_alert_manager = None
_global_performance_monitor = None
_global_observability_exporter = None


def get_global_logger() -> Logger:
    """获取全局日志记录器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger


def get_global_metrics_collector() -> MetricsCollector:
    """获取全局指标收集器"""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector


def get_global_tracing_manager() -> TracingManager:
    """获取全局链路追踪管理器"""
    global _global_tracing_manager
    if _global_tracing_manager is None:
        _global_tracing_manager = TracingManager()
    return _global_tracing_manager


def get_global_alert_manager() -> AlertManager:
    """获取全局告警管理器"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
    return _global_alert_manager


def get_global_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    global _global_performance_monitor
    if _global_performance_monitor is None:
        _global_performance_monitor = PerformanceMonitor()
    return _global_performance_monitor


def get_global_observability_exporter() -> ObservabilityExporter:
    """获取全局可观测性导出器"""
    global _global_observability_exporter
    if _global_observability_exporter is None:
        _global_observability_exporter = ObservabilityExporter()
    return _global_observability_exporter