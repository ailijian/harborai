#!/usr/bin/env python3
"""
性能指标收集器模块

提供全面的性能指标收集和监控功能，包括：
- 延迟和吞吐量监控
- 资源使用率追踪
- 错误率统计
- 自定义指标收集
- OpenTelemetry集成

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import time
import asyncio
import threading
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from enum import Enum

import structlog
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    type: MetricType
    description: str
    unit: str = ""
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # 用于直方图
    
    
@dataclass
class PerformanceThreshold:
    """性能阈值"""
    metric_name: str
    warning_threshold: float
    error_threshold: float
    critical_threshold: float
    comparison_operator: str = ">"  # >, <, >=, <=, ==, !=
    time_window_seconds: int = 60
    min_samples: int = 5


@dataclass
class MetricSample:
    """指标样本"""
    timestamp: datetime
    value: Union[int, float]
    labels: Dict[str, str] = field(default_factory=dict)
    
    
@dataclass
class PerformanceAlert:
    """性能告警"""
    metric_name: str
    level: AlertLevel
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceReport:
    """性能报告"""
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    error_rate: float
    cpu_usage_avg: float
    memory_usage_avg: float
    alerts: List[PerformanceAlert] = field(default_factory=list)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class PerformanceMetricsCollector:
    """
    性能指标收集器
    
    功能：
    1. 收集系统和应用性能指标
    2. 实时监控和告警
    3. 与OpenTelemetry集成
    4. 生成性能报告
    5. 支持自定义指标
    """
    
    def __init__(
        self,
        service_name: str = "harborai-logging",
        otlp_endpoint: Optional[str] = None,
        collection_interval: int = 10,
        enable_system_metrics: bool = True,
        enable_opentelemetry: bool = True
    ):
        """
        初始化性能指标收集器
        
        参数:
            service_name: 服务名称
            otlp_endpoint: OpenTelemetry导出端点
            collection_interval: 收集间隔（秒）
            enable_system_metrics: 是否启用系统指标收集
            enable_opentelemetry: 是否启用OpenTelemetry集成
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.collection_interval = collection_interval
        self.enable_system_metrics = enable_system_metrics
        self.enable_opentelemetry = enable_opentelemetry
        
        self.logger = structlog.get_logger(__name__)
        
        # 指标存储
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._metric_definitions: Dict[str, MetricDefinition] = {}
        self._thresholds: Dict[str, PerformanceThreshold] = {}
        self._alerts: List[PerformanceAlert] = []
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 收集状态
        self._collecting = False
        self._collection_task: Optional[asyncio.Task] = None
        
        # OpenTelemetry组件
        self._meter_provider: Optional[MeterProvider] = None
        self._meter: Optional[metrics.Meter] = None
        self._otel_instruments: Dict[str, Any] = {}
        
        # 性能统计
        self._start_time = datetime.now()
        self._request_count = 0
        self._error_count = 0
        self._latency_samples: deque = deque(maxlen=10000)
        
        # 初始化
        self._setup_default_metrics()
        if self.enable_opentelemetry:
            self._setup_opentelemetry()
    
    def _setup_default_metrics(self) -> None:
        """设置默认指标"""
        default_metrics = [
            MetricDefinition(
                name="request_duration_ms",
                type=MetricType.HISTOGRAM,
                description="请求处理时间",
                unit="ms",
                labels=["provider", "model", "operation"],
                buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
            ),
            MetricDefinition(
                name="request_count",
                type=MetricType.COUNTER,
                description="请求总数",
                labels=["provider", "model", "status"]
            ),
            MetricDefinition(
                name="error_rate",
                type=MetricType.GAUGE,
                description="错误率",
                unit="percent",
                labels=["provider", "model"]
            ),
            MetricDefinition(
                name="throughput_rps",
                type=MetricType.GAUGE,
                description="每秒请求数",
                unit="rps",
                labels=["provider", "model"]
            ),
            MetricDefinition(
                name="cpu_usage",
                type=MetricType.GAUGE,
                description="CPU使用率",
                unit="percent"
            ),
            MetricDefinition(
                name="memory_usage",
                type=MetricType.GAUGE,
                description="内存使用率",
                unit="percent"
            ),
            MetricDefinition(
                name="token_usage",
                type=MetricType.COUNTER,
                description="Token使用量",
                labels=["provider", "model", "type"]
            ),
            MetricDefinition(
                name="cost_total",
                type=MetricType.COUNTER,
                description="总成本",
                unit="CNY",
                labels=["provider", "model"]
            )
        ]
        
        for metric_def in default_metrics:
            self._metric_definitions[metric_def.name] = metric_def
        
        # 设置默认阈值
        self._setup_default_thresholds()
    
    def _setup_default_thresholds(self) -> None:
        """设置默认阈值"""
        default_thresholds = [
            PerformanceThreshold(
                metric_name="request_duration_ms",
                warning_threshold=1000,
                error_threshold=3000,
                critical_threshold=5000,
                comparison_operator=">",
                time_window_seconds=60
            ),
            PerformanceThreshold(
                metric_name="error_rate",
                warning_threshold=5.0,
                error_threshold=10.0,
                critical_threshold=20.0,
                comparison_operator=">",
                time_window_seconds=300
            ),
            PerformanceThreshold(
                metric_name="cpu_usage",
                warning_threshold=70.0,
                error_threshold=85.0,
                critical_threshold=95.0,
                comparison_operator=">",
                time_window_seconds=120
            ),
            PerformanceThreshold(
                metric_name="memory_usage",
                warning_threshold=80.0,
                error_threshold=90.0,
                critical_threshold=95.0,
                comparison_operator=">",
                time_window_seconds=120
            )
        ]
        
        for threshold in default_thresholds:
            self._thresholds[threshold.metric_name] = threshold
    
    def _setup_opentelemetry(self) -> None:
        """设置OpenTelemetry指标导出"""
        try:
            # 创建资源
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": "2.0.0",
                "ai.system": "harborai"
            })
            
            # 创建导出器
            readers = []
            if self.otlp_endpoint:
                otlp_exporter = OTLPMetricExporter(endpoint=self.otlp_endpoint)
                reader = PeriodicExportingMetricReader(
                    exporter=otlp_exporter,
                    export_interval_millis=self.collection_interval * 1000
                )
                readers.append(reader)
            
            # 创建MeterProvider
            self._meter_provider = MeterProvider(
                resource=resource,
                metric_readers=readers
            )
            
            # 设置全局MeterProvider
            metrics.set_meter_provider(self._meter_provider)
            
            # 获取Meter
            self._meter = metrics.get_meter(__name__)
            
            # 创建OpenTelemetry仪表
            self._create_otel_instruments()
            
            self.logger.info(
                "OpenTelemetry指标导出器初始化成功",
                service_name=self.service_name,
                otlp_endpoint=self.otlp_endpoint
            )
            
        except Exception as e:
            self.logger.error(
                "OpenTelemetry指标导出器初始化失败",
                error=str(e)
            )
            self.enable_opentelemetry = False
    
    def _create_otel_instruments(self) -> None:
        """创建OpenTelemetry仪表"""
        if not self._meter:
            return
        
        for name, metric_def in self._metric_definitions.items():
            try:
                if metric_def.type == MetricType.COUNTER:
                    instrument = self._meter.create_counter(
                        name=name,
                        description=metric_def.description,
                        unit=metric_def.unit
                    )
                elif metric_def.type == MetricType.GAUGE:
                    instrument = self._meter.create_up_down_counter(
                        name=name,
                        description=metric_def.description,
                        unit=metric_def.unit
                    )
                elif metric_def.type == MetricType.HISTOGRAM:
                    instrument = self._meter.create_histogram(
                        name=name,
                        description=metric_def.description,
                        unit=metric_def.unit
                    )
                else:
                    continue
                
                self._otel_instruments[name] = instrument
                
            except Exception as e:
                self.logger.warning(
                    "创建OpenTelemetry仪表失败",
                    metric_name=name,
                    error=str(e)
                )
    
    async def start_collection(self) -> None:
        """开始指标收集"""
        if self._collecting:
            return
        
        self._collecting = True
        self._start_time = datetime.now()
        
        if self.enable_system_metrics:
            self._collection_task = asyncio.create_task(self._collection_loop())
        
        self.logger.info("性能指标收集已启动")
    
    async def stop_collection(self) -> None:
        """停止指标收集"""
        if not self._collecting:
            return
        
        self._collecting = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("性能指标收集已停止")
    
    async def _collection_loop(self) -> None:
        """指标收集循环"""
        while self._collecting:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(
                    "收集系统指标失败",
                    error=str(e)
                )
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self) -> None:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=None)
            await self.record_gauge("cpu_usage", cpu_percent)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            await self.record_gauge("memory_usage", memory.percent)
            
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            await self.record_gauge("disk_usage", disk_percent)
            
            # 网络IO
            net_io = psutil.net_io_counters()
            await self.record_gauge("network_bytes_sent", net_io.bytes_sent)
            await self.record_gauge("network_bytes_recv", net_io.bytes_recv)
            
        except Exception as e:
            self.logger.error(
                "收集系统指标失败",
                error=str(e)
            )
    
    async def record_counter(
        self,
        name: str,
        value: Union[int, float] = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """记录计数器指标"""
        await self._record_metric(name, value, MetricType.COUNTER, labels)
    
    async def record_gauge(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """记录仪表指标"""
        await self._record_metric(name, value, MetricType.GAUGE, labels)
    
    async def record_histogram(
        self,
        name: str,
        value: Union[int, float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """记录直方图指标"""
        await self._record_metric(name, value, MetricType.HISTOGRAM, labels)
    
    async def _record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """记录指标"""
        if labels is None:
            labels = {}
        
        sample = MetricSample(
            timestamp=datetime.now(),
            value=value,
            labels=labels
        )
        
        with self._lock:
            self._metrics[name].append(sample)
        
        # 记录到OpenTelemetry
        if self.enable_opentelemetry and name in self._otel_instruments:
            try:
                instrument = self._otel_instruments[name]
                if metric_type == MetricType.COUNTER:
                    instrument.add(value, labels)
                elif metric_type == MetricType.GAUGE:
                    instrument.add(value, labels)
                elif metric_type == MetricType.HISTOGRAM:
                    instrument.record(value, labels)
            except Exception as e:
                self.logger.debug(
                    "记录OpenTelemetry指标失败",
                    metric_name=name,
                    error=str(e)
                )
        
        # 检查阈值
        await self._check_thresholds(name, value, labels)
    
    async def _check_thresholds(
        self,
        metric_name: str,
        value: Union[int, float],
        labels: Dict[str, str]
    ) -> None:
        """检查指标阈值"""
        threshold = self._thresholds.get(metric_name)
        if not threshold:
            return
        
        # 获取时间窗口内的样本
        cutoff_time = datetime.now() - timedelta(seconds=threshold.time_window_seconds)
        
        with self._lock:
            samples = [
                s for s in self._metrics[metric_name]
                if s.timestamp >= cutoff_time
            ]
        
        if len(samples) < threshold.min_samples:
            return
        
        # 计算平均值
        avg_value = sum(s.value for s in samples) / len(samples)
        
        # 检查阈值
        alert_level = None
        threshold_value = None
        
        if self._compare_value(avg_value, threshold.critical_threshold, threshold.comparison_operator):
            alert_level = AlertLevel.CRITICAL
            threshold_value = threshold.critical_threshold
        elif self._compare_value(avg_value, threshold.error_threshold, threshold.comparison_operator):
            alert_level = AlertLevel.ERROR
            threshold_value = threshold.error_threshold
        elif self._compare_value(avg_value, threshold.warning_threshold, threshold.comparison_operator):
            alert_level = AlertLevel.WARNING
            threshold_value = threshold.warning_threshold
        
        if alert_level:
            alert = PerformanceAlert(
                metric_name=metric_name,
                level=alert_level,
                message=f"{metric_name} {threshold.comparison_operator} {threshold_value} (当前值: {avg_value:.2f})",
                current_value=avg_value,
                threshold=threshold_value,
                timestamp=datetime.now(),
                labels=labels
            )
            
            with self._lock:
                self._alerts.append(alert)
            
            self.logger.warning(
                "性能指标告警",
                metric_name=metric_name,
                level=alert_level.value,
                current_value=avg_value,
                threshold=threshold_value,
                labels=labels
            )
    
    def _compare_value(self, value: float, threshold: float, operator: str) -> bool:
        """比较值与阈值"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return value == threshold
        elif operator == "!=":
            return value != threshold
        else:
            return False
    
    @asynccontextmanager
    async def measure_latency(
        self,
        operation_name: str,
        labels: Optional[Dict[str, str]] = None
    ):
        """测量操作延迟的上下文管理器"""
        start_time = time.time()
        
        try:
            yield
            
            # 成功完成
            duration_ms = (time.time() - start_time) * 1000
            await self.record_histogram("request_duration_ms", duration_ms, labels)
            await self.record_counter("request_count", 1, {**(labels or {}), "status": "success"})
            
            with self._lock:
                self._request_count += 1
                self._latency_samples.append(duration_ms)
            
        except Exception as e:
            # 发生错误
            duration_ms = (time.time() - start_time) * 1000
            await self.record_histogram("request_duration_ms", duration_ms, labels)
            await self.record_counter("request_count", 1, {**(labels or {}), "status": "error"})
            
            with self._lock:
                self._request_count += 1
                self._error_count += 1
                self._latency_samples.append(duration_ms)
            
            raise
    
    async def record_ai_operation(
        self,
        provider: str,
        model: str,
        operation: str,
        duration_ms: float,
        token_usage: Optional[Dict[str, int]] = None,
        cost: Optional[float] = None,
        success: bool = True
    ) -> None:
        """记录AI操作指标"""
        labels = {
            "provider": provider,
            "model": model,
            "operation": operation
        }
        
        # 记录延迟
        await self.record_histogram("request_duration_ms", duration_ms, labels)
        
        # 记录请求计数
        status_labels = {**labels, "status": "success" if success else "error"}
        await self.record_counter("request_count", 1, status_labels)
        
        # 记录Token使用量
        if token_usage:
            for token_type, count in token_usage.items():
                token_labels = {**labels, "type": token_type}
                await self.record_counter("token_usage", count, token_labels)
        
        # 记录成本
        if cost is not None:
            await self.record_counter("cost_total", cost, labels)
        
        # 更新统计
        with self._lock:
            self._request_count += 1
            if not success:
                self._error_count += 1
            self._latency_samples.append(duration_ms)
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        time_window_minutes: Optional[int] = None
    ) -> Dict[str, List[MetricSample]]:
        """获取指标数据"""
        cutoff_time = None
        if time_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            if metric_name:
                samples = list(self._metrics.get(metric_name, []))
                if cutoff_time:
                    samples = [s for s in samples if s.timestamp >= cutoff_time]
                return {metric_name: samples}
            else:
                result = {}
                for name, samples in self._metrics.items():
                    filtered_samples = list(samples)
                    if cutoff_time:
                        filtered_samples = [s for s in filtered_samples if s.timestamp >= cutoff_time]
                    result[name] = filtered_samples
                return result
    
    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        time_window_minutes: Optional[int] = None
    ) -> List[PerformanceAlert]:
        """获取告警信息"""
        cutoff_time = None
        if time_window_minutes:
            cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            alerts = list(self._alerts)
        
        if cutoff_time:
            alerts = [a for a in alerts if a.timestamp >= cutoff_time]
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts
    
    def generate_performance_report(
        self,
        time_window_minutes: int = 60
    ) -> PerformanceReport:
        """生成性能报告"""
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=time_window_minutes)
        
        with self._lock:
            # 获取时间窗口内的延迟样本
            latency_samples = [
                sample for sample in self._latency_samples
                if len(self._latency_samples) > 0  # 确保有样本
            ]
            
            # 计算统计信息
            if latency_samples:
                latency_samples_sorted = sorted(latency_samples)
                avg_latency = sum(latency_samples) / len(latency_samples)
                p95_index = int(len(latency_samples_sorted) * 0.95)
                p99_index = int(len(latency_samples_sorted) * 0.99)
                p95_latency = latency_samples_sorted[p95_index] if p95_index < len(latency_samples_sorted) else 0
                p99_latency = latency_samples_sorted[p99_index] if p99_index < len(latency_samples_sorted) else 0
            else:
                avg_latency = p95_latency = p99_latency = 0
            
            # 计算吞吐量和错误率
            total_requests = self._request_count
            failed_requests = self._error_count
            successful_requests = total_requests - failed_requests
            
            # 使用时间窗口计算吞吐量，而不是从启动时间开始
            duration_seconds = time_window_minutes * 60
            throughput_rps = total_requests / duration_seconds if duration_seconds > 0 else 0
            error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
            
            # 获取系统指标
            cpu_samples = [s.value for s in self._metrics.get("cpu_usage", []) if s.timestamp >= start_time]
            memory_samples = [s.value for s in self._metrics.get("memory_usage", []) if s.timestamp >= start_time]
            
            cpu_usage_avg = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
            memory_usage_avg = sum(memory_samples) / len(memory_samples) if memory_samples else 0
            
            # 获取告警
            alerts = [a for a in self._alerts if a.timestamp >= start_time]
        
        return PerformanceReport(
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput_rps=throughput_rps,
            error_rate=error_rate,
            cpu_usage_avg=cpu_usage_avg,
            memory_usage_avg=memory_usage_avg,
            alerts=alerts
        )
    
    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """添加性能阈值"""
        with self._lock:
            self._thresholds[threshold.metric_name] = threshold
    
    def remove_threshold(self, metric_name: str) -> None:
        """移除性能阈值"""
        with self._lock:
            self._thresholds.pop(metric_name, None)
    
    def clear_alerts(self) -> None:
        """清除所有告警"""
        with self._lock:
            self._alerts.clear()
    
    def clear_metrics(self) -> None:
        """清除所有指标数据"""
        with self._lock:
            self._metrics.clear()
            self._request_count = 0
            self._error_count = 0
            self._latency_samples.clear()
            self._alerts.clear()
    
    async def shutdown(self) -> None:
        """关闭指标收集器"""
        await self.stop_collection()
        
        if self._meter_provider:
            try:
                self._meter_provider.shutdown()
                self.logger.info("OpenTelemetry指标提供器已关闭")
            except Exception as e:
                self.logger.error(
                    "关闭OpenTelemetry指标提供器失败",
                    error=str(e)
                )


# 全局实例
_global_metrics_collector: Optional[PerformanceMetricsCollector] = None


def get_global_metrics_collector() -> Optional[PerformanceMetricsCollector]:
    """获取全局指标收集器实例"""
    return _global_metrics_collector


def setup_global_metrics_collector(
    service_name: str = "harborai-logging",
    otlp_endpoint: Optional[str] = None,
    collection_interval: int = 10,
    enable_system_metrics: bool = True,
    enable_opentelemetry: bool = True
) -> PerformanceMetricsCollector:
    """设置全局指标收集器实例"""
    global _global_metrics_collector
    _global_metrics_collector = PerformanceMetricsCollector(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        collection_interval=collection_interval,
        enable_system_metrics=enable_system_metrics,
        enable_opentelemetry=enable_opentelemetry
    )
    return _global_metrics_collector


def create_metrics_collector_from_env() -> PerformanceMetricsCollector:
    """从环境变量创建指标收集器"""
    import os
    
    return PerformanceMetricsCollector(
        service_name=os.getenv("OTEL_SERVICE_NAME", "harborai-logging"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"),
        collection_interval=int(os.getenv("METRICS_COLLECTION_INTERVAL", "10")),
        enable_system_metrics=os.getenv("ENABLE_SYSTEM_METRICS", "true").lower() == "true",
        enable_opentelemetry=os.getenv("ENABLE_OPENTELEMETRY_METRICS", "true").lower() == "true"
    )