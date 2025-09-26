# -*- coding: utf-8 -*-
"""
HarborAI 可观测性测试模块

测试目标：
- 验证日志记录和结构化日志功能
- 测试指标收集和监控系统
- 验证链路追踪和分布式追踪
- 测试告警和通知机制
- 验证性能监控和资源使用情况
- 测试可观测性数据的导出和集成
"""

import pytest
import asyncio
import time
import json
import logging
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import uuid
import os
import tempfile

from harborai import HarborAI
from harborai.core.observability import (
    Logger,
    MetricsCollector,
    TracingManager,
    AlertManager,
    PerformanceMonitor,
    ObservabilityExporter
)
from harborai.core.exceptions import HarborAIError


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


class MockLogger:
    """模拟日志记录器"""
    
    def __init__(self, name: str = "harborai"):
        self.name = name
        self.logs: List[LogEntry] = []
        self.handlers: List[Callable] = []
        self.level = LogLevel.INFO
        self.structured_logging = True
        self.context = {}
    
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
    
    def debug(self, message: str, **kwargs):
        """记录调试日志"""
        if self._should_log(LogLevel.DEBUG):
            entry = self._create_log_entry(LogLevel.DEBUG, message, **kwargs)
            self.logs.append(entry)
            self._notify_handlers(entry)
    
    def info(self, message: str, **kwargs):
        """记录信息日志"""
        if self._should_log(LogLevel.INFO):
            entry = self._create_log_entry(LogLevel.INFO, message, **kwargs)
            self.logs.append(entry)
            self._notify_handlers(entry)
    
    def warning(self, message: str, **kwargs):
        """记录警告日志"""
        if self._should_log(LogLevel.WARNING):
            entry = self._create_log_entry(LogLevel.WARNING, message, **kwargs)
            self.logs.append(entry)
            self._notify_handlers(entry)
    
    def error(self, message: str, exception: Exception = None, **kwargs):
        """记录错误日志"""
        if self._should_log(LogLevel.ERROR):
            entry = self._create_log_entry(LogLevel.ERROR, message, **kwargs)
            if exception:
                entry.exception = str(exception)
                entry.stack_trace = self._get_stack_trace(exception)
            self.logs.append(entry)
            self._notify_handlers(entry)
    
    def critical(self, message: str, exception: Exception = None, **kwargs):
        """记录严重错误日志"""
        if self._should_log(LogLevel.CRITICAL):
            entry = self._create_log_entry(LogLevel.CRITICAL, message, **kwargs)
            if exception:
                entry.exception = str(exception)
                entry.stack_trace = self._get_stack_trace(exception)
            self.logs.append(entry)
            self._notify_handlers(entry)
    
    def _get_stack_trace(self, exception: Exception) -> str:
        """获取异常堆栈跟踪"""
        import traceback
        return ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    
    def _notify_handlers(self, entry: LogEntry):
        """通知所有处理器"""
        for handler in self.handlers:
            try:
                handler(entry)
            except Exception:
                pass  # 忽略处理器错误
    
    def get_logs(self, level: LogLevel = None, since: datetime = None) -> List[LogEntry]:
        """获取日志"""
        logs = self.logs
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        if since:
            logs = [log for log in logs if log.timestamp >= since]
        
        return logs
    
    def clear_logs(self):
        """清除日志"""
        self.logs.clear()


class MockMetricsCollector:
    """模拟指标收集器"""
    
    def __init__(self):
        self.metrics: List[Metric] = []
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.summaries: Dict[str, List[float]] = defaultdict(list)
    
    def counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None, description: str = None):
        """记录计数器指标"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.counters[key] += value
        
        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=self.counters[key],
            timestamp=datetime.now(),
            labels=labels,
            description=description
        )
        self.metrics.append(metric)
    
    def gauge(self, name: str, value: float, labels: Dict[str, str] = None, description: str = None):
        """记录仪表盘指标"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.gauges[key] = value
        
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            timestamp=datetime.now(),
            labels=labels,
            description=description
        )
        self.metrics.append(metric)
    
    def histogram(self, name: str, value: float, labels: Dict[str, str] = None, description: str = None):
        """记录直方图指标"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.histograms[key].append(value)
        
        metric = Metric(
            name=name,
            type=MetricType.HISTOGRAM,
            value=value,
            timestamp=datetime.now(),
            labels=labels,
            description=description
        )
        self.metrics.append(metric)
    
    def summary(self, name: str, value: float, labels: Dict[str, str] = None, description: str = None):
        """记录摘要指标"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        self.summaries[key].append(value)
        
        metric = Metric(
            name=name,
            type=MetricType.SUMMARY,
            value=value,
            timestamp=datetime.now(),
            labels=labels,
            description=description
        )
        self.metrics.append(metric)
    
    def get_counter_value(self, name: str, labels: Dict[str, str] = None) -> float:
        """获取计数器值"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        return self.counters.get(key, 0.0)
    
    def get_gauge_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """获取仪表盘值"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        return self.gauges.get(key)
    
    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """获取直方图统计"""
        labels = labels or {}
        key = f"{name}:{json.dumps(labels, sort_keys=True)}"
        values = self.histograms.get(key, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(values)
        
        def percentile(data, p):
            """计算百分位数"""
            if not data:
                return 0
            k = (len(data) - 1) * p
            f = int(k)
            c = k - f
            if f + 1 < len(data):
                return data[f] * (1 - c) + data[f + 1] * c
            else:
                return data[f]
        
        return {
            "count": count,
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / count,
            "p50": percentile(sorted_values, 0.5),
            "p90": percentile(sorted_values, 0.9),
            "p95": percentile(sorted_values, 0.95),
            "p99": percentile(sorted_values, 0.99)
        }
    
    def get_metrics(self, name_pattern: str = None, since: datetime = None) -> List[Metric]:
        """获取指标"""
        metrics = self.metrics
        
        if name_pattern:
            import re
            pattern = re.compile(name_pattern)
            metrics = [m for m in metrics if pattern.match(m.name)]
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        return metrics
    
    def clear_metrics(self):
        """清除指标"""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.histograms.clear()
        self.summaries.clear()


class MockTracingManager:
    """模拟追踪管理器"""
    
    def __init__(self):
        self.spans: Dict[str, Span] = {}
        self.active_spans: Dict[str, str] = {}  # thread_id -> span_id
        self.traces: Dict[str, List[str]] = defaultdict(list)  # trace_id -> span_ids
    
    def start_trace(self, operation_name: str, trace_id: str = None) -> str:
        """开始新的追踪"""
        trace_id = trace_id or str(uuid.uuid4())
        span_id = self.start_span(operation_name, trace_id=trace_id)
        return trace_id
    
    def start_span(self, operation_name: str, parent_span_id: str = None, trace_id: str = None) -> str:
        """开始新的跨度"""
        span_id = str(uuid.uuid4())
        thread_id = str(threading.current_thread().ident)
        
        # 如果没有指定父跨度，尝试使用当前活跃跨度
        if parent_span_id is None and thread_id in self.active_spans:
            parent_span_id = self.active_spans[thread_id]
        
        # 如果没有指定追踪ID，尝试从父跨度获取
        if trace_id is None and parent_span_id:
            parent_span = self.spans.get(parent_span_id)
            if parent_span:
                trace_id = parent_span.trace_id
        
        # 如果仍然没有追踪ID，创建新的
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            status="active"
        )
        
        self.spans[span_id] = span
        self.traces[trace_id].append(span_id)
        self.active_spans[thread_id] = span_id
        
        return span_id
    
    def finish_span(self, span_id: str, status: str = "ok", tags: Dict[str, Any] = None):
        """结束跨度"""
        span = self.spans.get(span_id)
        if not span:
            return
        
        span.end_time = datetime.now()
        span.duration = (span.end_time - span.start_time).total_seconds() * 1000  # 毫秒
        span.status = status
        
        if tags:
            span.tags.update(tags)
        
        # 移除活跃跨度
        thread_id = str(threading.current_thread().ident)
        if self.active_spans.get(thread_id) == span_id:
            # 恢复父跨度为活跃跨度
            if span.parent_span_id:
                self.active_spans[thread_id] = span.parent_span_id
            else:
                del self.active_spans[thread_id]
    
    def add_span_tag(self, span_id: str, key: str, value: Any):
        """添加跨度标签"""
        span = self.spans.get(span_id)
        if span:
            span.tags[key] = value
    
    def add_span_log(self, span_id: str, message: str, level: str = "info", **fields):
        """添加跨度日志"""
        span = self.spans.get(span_id)
        if span:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": level,
                "message": message,
                **fields
            }
            span.logs.append(log_entry)
    
    def get_active_span_id(self) -> Optional[str]:
        """获取当前活跃跨度ID"""
        thread_id = str(threading.current_thread().ident)
        return self.active_spans.get(thread_id)
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """获取跨度"""
        return self.spans.get(span_id)
    
    def get_trace_spans(self, trace_id: str) -> List[Span]:
        """获取追踪的所有跨度"""
        span_ids = self.traces.get(trace_id, [])
        return [self.spans[span_id] for span_id in span_ids if span_id in self.spans]
    
    def get_trace_duration(self, trace_id: str) -> Optional[float]:
        """获取追踪总时长"""
        spans = self.get_trace_spans(trace_id)
        if not spans:
            return None
        
        start_times = [span.start_time for span in spans]
        end_times = [span.end_time for span in spans if span.end_time]
        
        if not start_times or not end_times:
            return None
        
        total_start = min(start_times)
        total_end = max(end_times)
        
        return (total_end - total_start).total_seconds() * 1000  # 毫秒
    
    def clear_traces(self):
        """清除所有追踪数据"""
        self.spans.clear()
        self.active_spans.clear()
        self.traces.clear()


class TestLogging:
    """日志记录测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.observability
    def test_basic_logging(self):
        """测试基本日志记录功能"""
        logger = MockLogger("test_logger")
        
        # 测试不同级别的日志
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # 验证日志记录
        logs = logger.get_logs()
        assert len(logs) == 4  # DEBUG级别默认不记录
        
        # 验证日志级别
        levels = [log.level for log in logs]
        assert LogLevel.INFO in levels
        assert LogLevel.WARNING in levels
        assert LogLevel.ERROR in levels
        assert LogLevel.CRITICAL in levels
        
        # 验证日志内容
        messages = [log.message for log in logs]
        assert "Info message" in messages
        assert "Warning message" in messages
        assert "Error message" in messages
        assert "Critical message" in messages
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.observability
    def test_structured_logging(self):
        """测试结构化日志记录"""
        logger = MockLogger("structured_logger")
        
        # 测试带额外字段的日志
        logger.info(
            "User action performed",
            user_id="user123",
            action="login",
            ip_address="192.168.1.1",
            duration=1.23
        )
        
        logger.error(
            "Database connection failed",
            database="postgres",
            host="db.example.com",
            port=5432,
            retry_count=3
        )
        
        # 验证结构化字段
        logs = logger.get_logs()
        assert len(logs) == 2
        
        info_log = logs[0]
        assert info_log.extra_fields["user_id"] == "user123"
        assert info_log.extra_fields["action"] == "login"
        assert info_log.extra_fields["ip_address"] == "192.168.1.1"
        assert info_log.extra_fields["duration"] == 1.23
        
        error_log = logs[1]
        assert error_log.extra_fields["database"] == "postgres"
        assert error_log.extra_fields["host"] == "db.example.com"
        assert error_log.extra_fields["port"] == 5432
        assert error_log.extra_fields["retry_count"] == 3
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.observability
    def test_logging_context(self):
        """测试日志上下文"""
        logger = MockLogger("context_logger")
        
        # 设置全局上下文
        logger.set_context(
            request_id="req-123",
            user_id="user456",
            session_id="sess-789"
        )
        
        # 记录日志
        logger.info("Processing request")
        logger.warning("Slow query detected", query_time=2.5)
        
        # 验证上下文字段
        logs = logger.get_logs()
        assert len(logs) == 2
        
        for log in logs:
            assert log.extra_fields["request_id"] == "req-123"
            assert log.extra_fields["user_id"] == "user456"
            assert log.extra_fields["session_id"] == "sess-789"
        
        # 验证额外字段也被保留
        warning_log = logs[1]
        assert warning_log.extra_fields["query_time"] == 2.5
        
        # 清除上下文
        logger.clear_context()
        logger.info("Context cleared")
        
        # 验证上下文已清除
        all_logs = logger.get_logs()
        context_cleared_log = all_logs[-1]  # 最新的日志
        assert context_cleared_log.extra_fields.get("request_id") is None
        assert context_cleared_log.extra_fields.get("user_id") is None
        assert context_cleared_log.extra_fields.get("session_id") is None
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.observability
    def test_exception_logging(self):
        """测试异常日志记录"""
        logger = MockLogger("exception_logger")
        
        # 创建测试异常
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            logger.error("An error occurred", exception=e)
        
        try:
            raise RuntimeError("Critical error")
        except RuntimeError as e:
            logger.critical("Critical error occurred", exception=e)
        
        # 验证异常信息
        logs = logger.get_logs()
        assert len(logs) == 2
        
        error_log = logs[0]
        assert error_log.exception == "Test exception"
        assert error_log.stack_trace is not None
        assert "ValueError" in error_log.stack_trace
        
        critical_log = logs[1]
        assert critical_log.exception == "Critical error"
        assert critical_log.stack_trace is not None
        assert "RuntimeError" in critical_log.stack_trace
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.observability
    def test_log_filtering(self):
        """测试日志过滤"""
        logger = MockLogger("filter_logger")
        
        # 设置不同的日志级别
        logger.set_level(LogLevel.WARNING)
        
        # 记录不同级别的日志
        logger.debug("Debug message")    # 不应记录
        logger.info("Info message")      # 不应记录
        logger.warning("Warning message") # 应记录
        logger.error("Error message")    # 应记录
        logger.critical("Critical message") # 应记录
        
        # 验证过滤结果
        logs = logger.get_logs()
        assert len(logs) == 3
        
        levels = [log.level for log in logs]
        assert LogLevel.DEBUG not in levels
        assert LogLevel.INFO not in levels
        assert LogLevel.WARNING in levels
        assert LogLevel.ERROR in levels
        assert LogLevel.CRITICAL in levels
        
        # 测试按级别获取日志
        warning_logs = logger.get_logs(level=LogLevel.WARNING)
        assert len(warning_logs) == 1
        assert warning_logs[0].message == "Warning message"
        
        error_logs = logger.get_logs(level=LogLevel.ERROR)
        assert len(error_logs) == 1
        assert error_logs[0].message == "Error message"
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.observability
    def test_log_handlers(self):
        """测试日志处理器"""
        logger = MockLogger("handler_logger")
        
        # 创建处理器
        handled_logs = []
        
        def test_handler(log_entry: LogEntry):
            handled_logs.append(log_entry)
        
        def error_handler(log_entry: LogEntry):
            if log_entry.level in [LogLevel.ERROR, LogLevel.CRITICAL]:
                # 模拟发送告警
                handled_logs.append(f"ALERT: {log_entry.message}")
        
        # 添加处理器
        logger.add_handler(test_handler)
        logger.add_handler(error_handler)
        
        # 记录日志
        logger.info("Info message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # 验证处理器被调用
        assert len(handled_logs) == 5  # 3个日志条目 + 2个告警
        
        # 验证日志条目
        log_entries = [item for item in handled_logs if isinstance(item, LogEntry)]
        assert len(log_entries) == 3
        
        # 验证告警
        alerts = [item for item in handled_logs if isinstance(item, str)]
        assert len(alerts) == 2
        assert "ALERT: Error message" in alerts
        assert "ALERT: Critical message" in alerts


class TestMetrics:
    """指标收集测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.observability
    def test_counter_metrics(self):
        """测试计数器指标"""
        collector = MockMetricsCollector()
        
        # 测试基本计数器
        collector.counter("requests_total")
        collector.counter("requests_total")
        collector.counter("requests_total", value=3)
        
        # 验证计数器值
        assert collector.get_counter_value("requests_total") == 5.0
        
        # 测试带标签的计数器
        collector.counter("http_requests_total", labels={"method": "GET", "status": "200"})
        collector.counter("http_requests_total", labels={"method": "GET", "status": "200"})
        collector.counter("http_requests_total", labels={"method": "POST", "status": "201"})
        
        # 验证带标签的计数器
        get_200_count = collector.get_counter_value(
            "http_requests_total", 
            labels={"method": "GET", "status": "200"}
        )
        assert get_200_count == 2.0
        
        post_201_count = collector.get_counter_value(
            "http_requests_total", 
            labels={"method": "POST", "status": "201"}
        )
        assert post_201_count == 1.0
        
        # 验证指标记录
        metrics = collector.get_metrics()
        counter_metrics = [m for m in metrics if m.type == MetricType.COUNTER]
        assert len(counter_metrics) >= 6  # 至少6个计数器指标
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.observability
    def test_gauge_metrics(self):
        """测试仪表盘指标"""
        collector = MockMetricsCollector()
        
        # 测试基本仪表盘
        collector.gauge("cpu_usage_percent", 45.5)
        collector.gauge("cpu_usage_percent", 67.2)
        collector.gauge("cpu_usage_percent", 23.8)
        
        # 验证仪表盘值（应该是最后设置的值）
        assert collector.get_gauge_value("cpu_usage_percent") == 23.8
        
        # 测试带标签的仪表盘
        collector.gauge("memory_usage_bytes", 1024*1024*512, labels={"instance": "server1"})
        collector.gauge("memory_usage_bytes", 1024*1024*768, labels={"instance": "server2"})
        
        # 验证带标签的仪表盘
        server1_memory = collector.get_gauge_value(
            "memory_usage_bytes", 
            labels={"instance": "server1"}
        )
        assert server1_memory == 1024*1024*512
        
        server2_memory = collector.get_gauge_value(
            "memory_usage_bytes", 
            labels={"instance": "server2"}
        )
        assert server2_memory == 1024*1024*768
        
        # 验证指标记录
        metrics = collector.get_metrics()
        gauge_metrics = [m for m in metrics if m.type == MetricType.GAUGE]
        assert len(gauge_metrics) >= 5  # 至少5个仪表盘指标
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.observability
    def test_histogram_metrics(self):
        """测试直方图指标"""
        collector = MockMetricsCollector()
        
        # 测试响应时间直方图
        response_times = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.28, 0.16, 0.24]
        
        for time_val in response_times:
            collector.histogram("response_time_seconds", time_val)
        
        # 获取直方图统计
        stats = collector.get_histogram_stats("response_time_seconds")
        
        assert stats["count"] == 10
        assert abs(stats["sum"] - sum(response_times)) < 0.001
        assert stats["min"] == min(response_times)
        assert stats["max"] == max(response_times)
        assert abs(stats["mean"] - (sum(response_times) / len(response_times))) < 0.001
        
        # 验证百分位数
        assert stats["p50"] > 0
        assert stats["p90"] > stats["p50"]
        assert stats["p95"] > stats["p90"]
        assert stats["p99"] > stats["p95"]
        
        # 测试带标签的直方图
        for time_val in [0.5, 0.6, 0.7]:
            collector.histogram(
                "api_response_time", 
                time_val, 
                labels={"endpoint": "/api/users", "method": "GET"}
            )
        
        api_stats = collector.get_histogram_stats(
            "api_response_time", 
            labels={"endpoint": "/api/users", "method": "GET"}
        )
        
        assert api_stats["count"] == 3
        assert api_stats["mean"] == 0.6
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.observability
    def test_metrics_filtering(self):
        """测试指标过滤"""
        collector = MockMetricsCollector()
        
        # 记录不同类型的指标
        collector.counter("requests_total")
        collector.gauge("cpu_usage", 50.0)
        collector.histogram("response_time", 0.2)
        collector.summary("request_size", 1024)
        
        time.sleep(0.1)  # 确保时间戳不同
        
        collector.counter("errors_total")
        collector.gauge("memory_usage", 75.0)
        
        # 测试按名称模式过滤
        request_metrics = collector.get_metrics(name_pattern=r".*_total")
        assert len(request_metrics) == 2
        
        usage_metrics = collector.get_metrics(name_pattern=r".*_usage")
        assert len(usage_metrics) == 2
        
        # 测试按时间过滤
        recent_time = datetime.now() - timedelta(seconds=0.05)
        recent_metrics = collector.get_metrics(since=recent_time)
        assert len(recent_metrics) == 2  # 最近的两个指标
        
        # 验证指标类型
        all_metrics = collector.get_metrics()
        counter_count = len([m for m in all_metrics if m.type == MetricType.COUNTER])
        gauge_count = len([m for m in all_metrics if m.type == MetricType.GAUGE])
        histogram_count = len([m for m in all_metrics if m.type == MetricType.HISTOGRAM])
        summary_count = len([m for m in all_metrics if m.type == MetricType.SUMMARY])
        
        assert counter_count == 2
        assert gauge_count == 2
        assert histogram_count == 1
        assert summary_count == 1


class TestTracing:
    """链路追踪测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.observability
    def test_basic_tracing(self):
        """测试基本链路追踪"""
        tracer = MockTracingManager()
        
        # 开始追踪
        trace_id = tracer.start_trace("user_request")
        assert trace_id is not None
        
        # 获取根跨度
        root_span_id = tracer.get_active_span_id()
        assert root_span_id is not None
        
        root_span = tracer.get_span(root_span_id)
        assert root_span is not None
        assert root_span.trace_id == trace_id
        assert root_span.operation_name == "user_request"
        assert root_span.parent_span_id is None
        
        # 结束跨度
        tracer.finish_span(root_span_id, status="ok")
        
        # 验证跨度已结束
        finished_span = tracer.get_span(root_span_id)
        assert finished_span.end_time is not None
        assert finished_span.duration is not None
        assert finished_span.status == "ok"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.observability
    def test_nested_spans(self):
        """测试嵌套跨度"""
        tracer = MockTracingManager()
        
        # 开始根追踪
        trace_id = tracer.start_trace("api_request")
        root_span_id = tracer.get_active_span_id()
        
        # 添加子跨度
        db_span_id = tracer.start_span("database_query")
        db_span = tracer.get_span(db_span_id)
        
        assert db_span.trace_id == trace_id
        assert db_span.parent_span_id == root_span_id
        assert db_span.operation_name == "database_query"
        
        # 添加孙跨度
        cache_span_id = tracer.start_span("cache_lookup")
        cache_span = tracer.get_span(cache_span_id)
        
        assert cache_span.trace_id == trace_id
        assert cache_span.parent_span_id == db_span_id
        assert cache_span.operation_name == "cache_lookup"
        
        # 按顺序结束跨度
        tracer.finish_span(cache_span_id, status="ok")
        tracer.finish_span(db_span_id, status="ok")
        tracer.finish_span(root_span_id, status="ok")
        
        # 验证追踪结构
        trace_spans = tracer.get_trace_spans(trace_id)
        assert len(trace_spans) == 3
        
        # 验证父子关系
        span_by_id = {span.span_id: span for span in trace_spans}
        
        root = span_by_id[root_span_id]
        db = span_by_id[db_span_id]
        cache = span_by_id[cache_span_id]
        
        assert root.parent_span_id is None
        assert db.parent_span_id == root.span_id
        assert cache.parent_span_id == db.span_id
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.observability
    def test_span_tags_and_logs(self):
        """测试跨度标签和日志"""
        tracer = MockTracingManager()
        
        # 开始跨度
        trace_id = tracer.start_trace("http_request")
        span_id = tracer.get_active_span_id()
        
        # 添加标签
        tracer.add_span_tag(span_id, "http.method", "GET")
        tracer.add_span_tag(span_id, "http.url", "/api/users")
        tracer.add_span_tag(span_id, "http.status_code", 200)
        tracer.add_span_tag(span_id, "user.id", "user123")
        
        # 添加日志
        tracer.add_span_log(span_id, "Request received", level="info")
        tracer.add_span_log(span_id, "Validating request", level="debug", validation_time=0.05)
        tracer.add_span_log(span_id, "Database query executed", level="info", query_time=0.12)
        tracer.add_span_log(span_id, "Response sent", level="info", response_size=1024)
        
        # 结束跨度
        tracer.finish_span(span_id, status="ok", tags={"response.size": 1024})
        
        # 验证标签
        span = tracer.get_span(span_id)
        assert span.tags["http.method"] == "GET"
        assert span.tags["http.url"] == "/api/users"
        assert span.tags["http.status_code"] == 200
        assert span.tags["user.id"] == "user123"
        assert span.tags["response.size"] == 1024
        
        # 验证日志
        assert len(span.logs) == 4
        
        log_messages = [log["message"] for log in span.logs]
        assert "Request received" in log_messages
        assert "Validating request" in log_messages
        assert "Database query executed" in log_messages
        assert "Response sent" in log_messages
        
        # 验证日志字段
        validation_log = next(log for log in span.logs if "validation_time" in log)
        assert validation_log["validation_time"] == 0.05
        
        query_log = next(log for log in span.logs if "query_time" in log)
        assert query_log["query_time"] == 0.12
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.observability
    def test_trace_duration(self):
        """测试追踪时长计算"""
        tracer = MockTracingManager()
        
        # 开始追踪
        trace_id = tracer.start_trace("complex_operation")
        root_span_id = tracer.get_active_span_id()
        
        # 模拟一些操作时间
        time.sleep(0.1)
        
        # 添加子跨度
        child1_span_id = tracer.start_span("operation_1")
        time.sleep(0.05)
        tracer.finish_span(child1_span_id, status="ok")
        
        child2_span_id = tracer.start_span("operation_2")
        time.sleep(0.03)
        tracer.finish_span(child2_span_id, status="ok")
        
        # 结束根跨度
        tracer.finish_span(root_span_id, status="ok")
        
        # 验证跨度时长
        root_span = tracer.get_span(root_span_id)
        child1_span = tracer.get_span(child1_span_id)
        child2_span = tracer.get_span(child2_span_id)
        
        assert root_span.duration > 150  # 至少150ms
        assert child1_span.duration > 40   # 至少40ms
        assert child2_span.duration > 20   # 至少20ms
        
        # 验证追踪总时长
        trace_duration = tracer.get_trace_duration(trace_id)
        assert trace_duration is not None
        assert trace_duration > 150  # 应该接近根跨度的时长
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.observability
    def test_error_tracing(self):
        """测试错误追踪"""
        tracer = MockTracingManager()
        
        # 开始追踪
        trace_id = tracer.start_trace("error_operation")
        span_id = tracer.get_active_span_id()
        
        # 模拟错误情况
        try:
            raise ValueError("Something went wrong")
        except ValueError as e:
            # 记录错误信息
            tracer.add_span_tag(span_id, "error", True)
            tracer.add_span_tag(span_id, "error.type", type(e).__name__)
            tracer.add_span_tag(span_id, "error.message", str(e))
            
            tracer.add_span_log(
                span_id, 
                "Exception occurred", 
                level="error",
                exception_type=type(e).__name__,
                exception_message=str(e)
            )
            
            # 结束跨度并标记为错误
            tracer.finish_span(span_id, status="error")
        
        # 验证错误信息
        span = tracer.get_span(span_id)
        assert span.status == "error"
        assert span.tags["error"] is True
        assert span.tags["error.type"] == "ValueError"
        assert span.tags["error.message"] == "Something went wrong"
        
        # 验证错误日志
        error_logs = [log for log in span.logs if log["level"] == "error"]
        assert len(error_logs) == 1
        assert error_logs[0]["exception_type"] == "ValueError"
        assert error_logs[0]["exception_message"] == "Something went wrong"