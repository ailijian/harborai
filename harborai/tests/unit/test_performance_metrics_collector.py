#!/usr/bin/env python3
"""
性能指标收集器单元测试

测试覆盖：
- 指标收集和记录
- 阈值检查和告警
- OpenTelemetry集成
- 性能报告生成
- 系统指标收集

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from harborai.core.tracing.performance_metrics_collector import (
    PerformanceMetricsCollector,
    MetricType,
    AlertLevel,
    MetricDefinition,
    PerformanceThreshold,
    MetricSample,
    PerformanceAlert,
    PerformanceReport,
    get_global_metrics_collector,
    setup_global_metrics_collector,
    create_metrics_collector_from_env
)


class TestPerformanceMetricsCollector:
    """性能指标收集器测试类"""
    
    @pytest.fixture
    def collector(self):
        """创建测试用的指标收集器"""
        return PerformanceMetricsCollector(
            service_name="test-service",
            enable_opentelemetry=False,  # 测试时禁用OpenTelemetry
            enable_system_metrics=False,  # 测试时禁用系统指标收集
            collection_interval=1
        )
    
    @pytest.fixture
    def mock_psutil(self):
        """模拟psutil"""
        with patch('harborai.core.tracing.performance_metrics_collector.psutil') as mock:
            mock.cpu_percent.return_value = 50.0
            mock.virtual_memory.return_value = Mock(percent=60.0)
            mock.disk_usage.return_value = Mock(used=500, total=1000)
            mock.net_io_counters.return_value = Mock(bytes_sent=1000, bytes_recv=2000)
            yield mock
    
    def test_init_default_config(self):
        """测试默认配置初始化"""
        collector = PerformanceMetricsCollector()
        
        assert collector.service_name == "harborai-logging"
        assert collector.collection_interval == 10
        assert collector.enable_system_metrics is True
        assert collector.enable_opentelemetry is True
        assert len(collector._metric_definitions) > 0
        assert len(collector._thresholds) > 0
    
    def test_init_custom_config(self):
        """测试自定义配置初始化"""
        collector = PerformanceMetricsCollector(
            service_name="custom-service",
            collection_interval=5,
            enable_system_metrics=False,
            enable_opentelemetry=False
        )
        
        assert collector.service_name == "custom-service"
        assert collector.collection_interval == 5
        assert collector.enable_system_metrics is False
        assert collector.enable_opentelemetry is False
    
    @pytest.mark.asyncio
    async def test_record_counter(self, collector):
        """测试记录计数器指标"""
        await collector.record_counter("test_counter", 5, {"label": "value"})
        
        metrics = collector.get_metrics("test_counter")
        assert "test_counter" in metrics
        assert len(metrics["test_counter"]) == 1
        
        sample = metrics["test_counter"][0]
        assert sample.value == 5
        assert sample.labels == {"label": "value"}
        assert isinstance(sample.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_record_gauge(self, collector):
        """测试记录仪表指标"""
        await collector.record_gauge("test_gauge", 75.5)
        
        metrics = collector.get_metrics("test_gauge")
        assert "test_gauge" in metrics
        assert len(metrics["test_gauge"]) == 1
        assert metrics["test_gauge"][0].value == 75.5
    
    @pytest.mark.asyncio
    async def test_record_histogram(self, collector):
        """测试记录直方图指标"""
        await collector.record_histogram("test_histogram", 123.45, {"type": "test"})
        
        metrics = collector.get_metrics("test_histogram")
        assert "test_histogram" in metrics
        assert len(metrics["test_histogram"]) == 1
        assert metrics["test_histogram"][0].value == 123.45
        assert metrics["test_histogram"][0].labels == {"type": "test"}
    
    @pytest.mark.asyncio
    async def test_measure_latency_success(self, collector):
        """测试成功操作的延迟测量"""
        labels = {"operation": "test"}
        
        async with collector.measure_latency("test_operation", labels):
            await asyncio.sleep(0.01)  # 模拟操作耗时
        
        # 检查延迟指标
        duration_metrics = collector.get_metrics("request_duration_ms")
        assert "request_duration_ms" in duration_metrics
        assert len(duration_metrics["request_duration_ms"]) == 1
        assert duration_metrics["request_duration_ms"][0].value >= 10  # 至少10ms
        
        # 检查计数指标
        count_metrics = collector.get_metrics("request_count")
        assert "request_count" in count_metrics
        assert len(count_metrics["request_count"]) == 1
        assert count_metrics["request_count"][0].labels["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_measure_latency_error(self, collector):
        """测试错误操作的延迟测量"""
        labels = {"operation": "test"}
        
        with pytest.raises(ValueError):
            async with collector.measure_latency("test_operation", labels):
                await asyncio.sleep(0.01)
                raise ValueError("Test error")
        
        # 检查计数指标
        count_metrics = collector.get_metrics("request_count")
        assert "request_count" in count_metrics
        assert len(count_metrics["request_count"]) == 1
        assert count_metrics["request_count"][0].labels["status"] == "error"
    
    @pytest.mark.asyncio
    async def test_record_ai_operation(self, collector):
        """测试记录AI操作指标"""
        await collector.record_ai_operation(
            provider="openai",
            model="gpt-4",
            operation="completion",
            duration_ms=1500,
            token_usage={"prompt_tokens": 100, "completion_tokens": 50},
            cost=0.05,
            success=True
        )
        
        # 检查延迟指标
        duration_metrics = collector.get_metrics("request_duration_ms")
        assert len(duration_metrics["request_duration_ms"]) == 1
        assert duration_metrics["request_duration_ms"][0].value == 1500
        
        # 检查计数指标
        count_metrics = collector.get_metrics("request_count")
        assert len(count_metrics["request_count"]) == 1
        assert count_metrics["request_count"][0].labels["status"] == "success"
        
        # 检查Token使用量
        token_metrics = collector.get_metrics("token_usage")
        assert len(token_metrics["token_usage"]) == 2  # prompt_tokens + completion_tokens
        
        # 检查成本
        cost_metrics = collector.get_metrics("cost_total")
        assert len(cost_metrics["cost_total"]) == 1
        assert cost_metrics["cost_total"][0].value == 0.05
    
    @pytest.mark.asyncio
    async def test_threshold_checking_warning(self, collector):
        """测试阈值检查 - 警告级别"""
        # 添加测试阈值
        threshold = PerformanceThreshold(
            metric_name="test_metric",
            warning_threshold=50.0,
            error_threshold=80.0,
            critical_threshold=95.0,
            comparison_operator=">",
            time_window_seconds=60,
            min_samples=1
        )
        collector.add_threshold(threshold)
        
        # 记录超过警告阈值的指标
        await collector.record_gauge("test_metric", 60.0)
        
        # 等待阈值检查
        await asyncio.sleep(0.1)
        
        # 检查告警
        alerts = collector.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.WARNING
        assert alerts[0].metric_name == "test_metric"
        assert alerts[0].current_value == 60.0
    
    @pytest.mark.asyncio
    async def test_threshold_checking_critical(self, collector):
        """测试阈值检查 - 严重级别"""
        # 添加测试阈值
        threshold = PerformanceThreshold(
            metric_name="test_metric",
            warning_threshold=50.0,
            error_threshold=80.0,
            critical_threshold=95.0,
            comparison_operator=">",
            time_window_seconds=60,
            min_samples=1
        )
        collector.add_threshold(threshold)
        
        # 记录超过严重阈值的指标
        await collector.record_gauge("test_metric", 98.0)
        
        # 等待阈值检查
        await asyncio.sleep(0.1)
        
        # 检查告警
        alerts = collector.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].level == AlertLevel.CRITICAL
        assert alerts[0].current_value == 98.0
    
    @pytest.mark.asyncio
    async def test_threshold_checking_no_alert(self, collector):
        """测试阈值检查 - 无告警"""
        # 添加测试阈值
        threshold = PerformanceThreshold(
            metric_name="test_metric",
            warning_threshold=50.0,
            error_threshold=80.0,
            critical_threshold=95.0,
            comparison_operator=">",
            time_window_seconds=60,
            min_samples=1
        )
        collector.add_threshold(threshold)
        
        # 记录未超过阈值的指标
        await collector.record_gauge("test_metric", 30.0)
        
        # 等待阈值检查
        await asyncio.sleep(0.1)
        
        # 检查告警
        alerts = collector.get_alerts()
        assert len(alerts) == 0
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, collector, mock_psutil):
        """测试系统指标收集"""
        collector.enable_system_metrics = True
        
        # 手动触发系统指标收集
        await collector._collect_system_metrics()
        
        # 检查CPU使用率
        cpu_metrics = collector.get_metrics("cpu_usage")
        assert len(cpu_metrics["cpu_usage"]) == 1
        assert cpu_metrics["cpu_usage"][0].value == 50.0
        
        # 检查内存使用率
        memory_metrics = collector.get_metrics("memory_usage")
        assert len(memory_metrics["memory_usage"]) == 1
        assert memory_metrics["memory_usage"][0].value == 60.0
    
    @pytest.mark.asyncio
    async def test_collection_start_stop(self, collector):
        """测试收集启动和停止"""
        assert not collector._collecting
        
        await collector.start_collection()
        assert collector._collecting
        
        await collector.stop_collection()
        assert not collector._collecting
    
    def test_get_metrics_with_time_window(self, collector):
        """测试带时间窗口的指标获取"""
        # 添加一些测试数据
        now = datetime.now()
        old_sample = MetricSample(
            timestamp=now - timedelta(minutes=30),
            value=10.0
        )
        recent_sample = MetricSample(
            timestamp=now - timedelta(minutes=5),
            value=20.0
        )
        
        collector._metrics["test_metric"].extend([old_sample, recent_sample])
        
        # 获取最近10分钟的指标
        metrics = collector.get_metrics("test_metric", time_window_minutes=10)
        assert len(metrics["test_metric"]) == 1
        assert metrics["test_metric"][0].value == 20.0
    
    def test_get_alerts_with_filters(self, collector):
        """测试带过滤器的告警获取"""
        # 添加测试告警
        now = datetime.now()
        old_alert = PerformanceAlert(
            metric_name="test1",
            level=AlertLevel.WARNING,
            message="Old warning",
            current_value=60.0,
            threshold=50.0,
            timestamp=now - timedelta(minutes=30)
        )
        recent_alert = PerformanceAlert(
            metric_name="test2",
            level=AlertLevel.ERROR,
            message="Recent error",
            current_value=90.0,
            threshold=80.0,
            timestamp=now - timedelta(minutes=5)
        )
        
        collector._alerts.extend([old_alert, recent_alert])
        
        # 按级别过滤
        error_alerts = collector.get_alerts(level=AlertLevel.ERROR)
        assert len(error_alerts) == 1
        assert error_alerts[0].level == AlertLevel.ERROR
        
        # 按时间窗口过滤
        recent_alerts = collector.get_alerts(time_window_minutes=10)
        assert len(recent_alerts) == 1
        assert recent_alerts[0].message == "Recent error"
    
    @pytest.mark.asyncio
    async def test_generate_performance_report(self, collector):
        """测试性能报告生成"""
        # 模拟一些请求数据
        await collector.record_ai_operation(
            provider="openai",
            model="gpt-4",
            operation="completion",
            duration_ms=150.0,
            token_usage={"prompt": 100, "completion": 50},
            cost=0.01,
            success=True
        )
        
        await collector.record_ai_operation(
            provider="openai",
            model="gpt-4",
            operation="completion",
            duration_ms=200.0,
            token_usage={"prompt": 120, "completion": 60},
            cost=0.015,
            success=False
        )
        
        # 等待一小段时间确保数据被记录
        await asyncio.sleep(0.1)
        
        # 生成报告
        report = collector.generate_performance_report(time_window_minutes=60)
        
        # 验证报告内容
        assert isinstance(report, PerformanceReport)
        assert report.total_requests == 2
        assert report.successful_requests == 1
        assert report.failed_requests == 1
        assert report.average_latency_ms > 0
        assert report.throughput_rps > 0  # 现在应该大于0
        assert report.error_rate == 50.0  # 50% 错误率
    
    def test_generate_performance_report_with_system_metrics(self, collector):
        """测试带系统指标的性能报告生成"""
        # 模拟一些数据
        collector._request_count = 100
        collector._error_count = 5
        collector._latency_samples.extend([10, 20, 30, 40, 50] * 20)  # 100个样本
        
        # 添加系统指标
        now = datetime.now()
        for i in range(10):
            collector._metrics["cpu_usage"].append(
                MetricSample(timestamp=now - timedelta(minutes=i), value=50.0 + i)
            )
            collector._metrics["memory_usage"].append(
                MetricSample(timestamp=now - timedelta(minutes=i), value=60.0 + i)
            )
        
        # 生成报告
        report = collector.generate_performance_report(time_window_minutes=60)
        
        assert report.total_requests == 100
        assert report.successful_requests == 95
        assert report.failed_requests == 5
        assert report.error_rate == 5.0
        assert report.average_latency_ms == 30.0  # (10+20+30+40+50)/5 = 30
        assert report.throughput_rps > 0
        assert report.cpu_usage_avg > 0
        assert report.memory_usage_avg > 0
    
    def test_threshold_management(self, collector):
        """测试阈值管理"""
        threshold = PerformanceThreshold(
            metric_name="test_metric",
            warning_threshold=50.0,
            error_threshold=80.0,
            critical_threshold=95.0
        )
        
        # 添加阈值
        collector.add_threshold(threshold)
        assert "test_metric" in collector._thresholds
        
        # 移除阈值
        collector.remove_threshold("test_metric")
        assert "test_metric" not in collector._thresholds
    
    def test_clear_operations(self, collector):
        """测试清除操作"""
        # 添加一些数据
        collector._metrics["test"].append(MetricSample(datetime.now(), 10.0))
        collector._alerts.append(PerformanceAlert(
            metric_name="test",
            level=AlertLevel.WARNING,
            message="Test",
            current_value=60.0,
            threshold=50.0,
            timestamp=datetime.now()
        ))
        collector._request_count = 10
        collector._error_count = 2
        collector._latency_samples.append(100.0)
        
        # 清除告警
        collector.clear_alerts()
        assert len(collector._alerts) == 0
        
        # 清除指标
        collector.clear_metrics()
        assert len(collector._metrics) == 0
        assert collector._request_count == 0
        assert collector._error_count == 0
        assert len(collector._latency_samples) == 0
    
    @pytest.mark.asyncio
    async def test_shutdown(self, collector):
        """测试关闭操作"""
        await collector.start_collection()
        assert collector._collecting
        
        await collector.shutdown()
        assert not collector._collecting


class TestGlobalFunctions:
    """全局函数测试类"""
    
    def test_setup_and_get_global_collector(self):
        """测试设置和获取全局收集器"""
        # 初始状态
        assert get_global_metrics_collector() is None
        
        # 设置全局收集器
        collector = setup_global_metrics_collector(
            service_name="test-global",
            enable_opentelemetry=False
        )
        
        assert collector is not None
        assert collector.service_name == "test-global"
        assert get_global_metrics_collector() is collector
    
    @patch.dict('os.environ', {
        'OTEL_SERVICE_NAME': 'env-service',
        'OTEL_EXPORTER_OTLP_METRICS_ENDPOINT': 'http://localhost:4318/v1/metrics',
        'METRICS_COLLECTION_INTERVAL': '5',
        'ENABLE_SYSTEM_METRICS': 'false',
        'ENABLE_OPENTELEMETRY_METRICS': 'false'
    })
    def test_create_collector_from_env(self):
        """测试从环境变量创建收集器"""
        collector = create_metrics_collector_from_env()
        
        assert collector.service_name == "env-service"
        assert collector.otlp_endpoint == "http://localhost:4318/v1/metrics"
        assert collector.collection_interval == 5
        assert collector.enable_system_metrics is False
        assert collector.enable_opentelemetry is False


class TestMetricDefinition:
    """指标定义测试类"""
    
    def test_metric_definition_creation(self):
        """测试指标定义创建"""
        metric_def = MetricDefinition(
            name="test_metric",
            type=MetricType.HISTOGRAM,
            description="Test metric",
            unit="ms",
            labels=["provider", "model"],
            buckets=[1, 5, 10, 25, 50, 100]
        )
        
        assert metric_def.name == "test_metric"
        assert metric_def.type == MetricType.HISTOGRAM
        assert metric_def.description == "Test metric"
        assert metric_def.unit == "ms"
        assert metric_def.labels == ["provider", "model"]
        assert metric_def.buckets == [1, 5, 10, 25, 50, 100]


class TestPerformanceThreshold:
    """性能阈值测试类"""
    
    def test_threshold_creation(self):
        """测试阈值创建"""
        threshold = PerformanceThreshold(
            metric_name="latency",
            warning_threshold=100.0,
            error_threshold=500.0,
            critical_threshold=1000.0,
            comparison_operator=">",
            time_window_seconds=300,
            min_samples=10
        )
        
        assert threshold.metric_name == "latency"
        assert threshold.warning_threshold == 100.0
        assert threshold.error_threshold == 500.0
        assert threshold.critical_threshold == 1000.0
        assert threshold.comparison_operator == ">"
        assert threshold.time_window_seconds == 300
        assert threshold.min_samples == 10


class TestComparisonOperators:
    """比较操作符测试类"""
    
    def test_compare_value_operators(self):
        """测试值比较操作符"""
        collector = PerformanceMetricsCollector(enable_opentelemetry=False)
        
        # 测试各种操作符
        assert collector._compare_value(10, 5, ">") is True
        assert collector._compare_value(5, 10, ">") is False
        assert collector._compare_value(5, 10, "<") is True
        assert collector._compare_value(10, 5, "<") is False
        assert collector._compare_value(10, 10, ">=") is True
        assert collector._compare_value(9, 10, ">=") is False
        assert collector._compare_value(10, 10, "<=") is True
        assert collector._compare_value(11, 10, "<=") is False
        assert collector._compare_value(10, 10, "==") is True
        assert collector._compare_value(10, 11, "==") is False
        assert collector._compare_value(10, 11, "!=") is True
        assert collector._compare_value(10, 10, "!=") is False
        
        # 测试无效操作符
        assert collector._compare_value(10, 5, "invalid") is False


if __name__ == "__main__":
    pytest.main([__file__])