#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenTelemetry追踪器模块测试

测试分布式追踪、性能瓶颈识别和调用链分析功能。
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from harborai.monitoring.opentelemetry_tracer import (
    OpenTelemetryTracer,
    get_otel_tracer,
    init_otel_tracer,
    otel_trace,
    otel_trace_async,
    OTEL_AVAILABLE
)


class TestOpenTelemetryTracer:
    """OpenTelemetryTracer类测试"""
    
    def setup_method(self):
        """测试前设置"""
        # 重置全局追踪器
        import harborai.monitoring.opentelemetry_tracer as otel_module
        otel_module._otel_tracer = None
    
    def test_init_without_opentelemetry(self):
        """测试在没有OpenTelemetry的情况下初始化"""
        with patch('harborai.monitoring.opentelemetry_tracer.OTEL_AVAILABLE', False):
            tracer = OpenTelemetryTracer()
            assert tracer.enabled is False
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_init_with_opentelemetry(self):
        """测试在有OpenTelemetry的情况下初始化"""
        tracer = OpenTelemetryTracer(
            service_name="test_service",
            service_version="2.0.0"
        )
        assert tracer.enabled is True
        assert tracer.service_name == "test_service"
        assert tracer.service_version == "2.0.0"
        assert hasattr(tracer, 'tracer')
        assert hasattr(tracer, 'tracer_provider')
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_init_with_jaeger_endpoint(self):
        """测试使用Jaeger端点初始化"""
        with patch('harborai.monitoring.opentelemetry_tracer.JaegerExporter') as mock_jaeger:
            with patch('harborai.monitoring.opentelemetry_tracer.BatchSpanProcessor') as mock_processor:
                tracer = OpenTelemetryTracer(
                    jaeger_endpoint="http://localhost:14268/api/traces"
                )
                assert tracer.enabled is True
                mock_jaeger.assert_called_once()
                mock_processor.assert_called()
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_init_with_otlp_endpoint(self):
        """测试使用OTLP端点初始化"""
        with patch('harborai.monitoring.opentelemetry_tracer.OTLPSpanExporter') as mock_otlp:
            with patch('harborai.monitoring.opentelemetry_tracer.BatchSpanProcessor') as mock_processor:
                tracer = OpenTelemetryTracer(
                    otlp_endpoint="http://localhost:4317"
                )
                assert tracer.enabled is True
                mock_otlp.assert_called_once()
                mock_processor.assert_called()
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_init_with_exporter_error(self):
        """测试导出器配置错误的情况"""
        with patch('harborai.monitoring.opentelemetry_tracer.JaegerExporter', side_effect=Exception("Connection error")):
            # 应该不会抛出异常，只是记录警告
            tracer = OpenTelemetryTracer(
                jaeger_endpoint="http://localhost:14268/api/traces"
            )
            assert tracer.enabled is True
    
    def test_start_span_disabled(self):
        """测试在禁用状态下启动跨度"""
        tracer = OpenTelemetryTracer()
        tracer.enabled = False
        
        with tracer.start_span("test_span") as span:
            assert span is None
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_start_span_enabled(self):
        """测试在启用状态下启动跨度"""
        tracer = OpenTelemetryTracer()
        
        with patch.object(tracer, 'tracer') as mock_tracer:
            mock_span = Mock()
            mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
            
            with tracer.start_span("test_span", {"key": "value"}) as span:
                assert span == mock_span
                mock_tracer.start_as_current_span.assert_called_once()
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_start_span_with_exception(self):
        """测试跨度中发生异常的情况"""
        tracer = OpenTelemetryTracer()
        
        with patch.object(tracer, 'tracer') as mock_tracer:
            mock_span = Mock()
            mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
            
            with pytest.raises(ValueError):
                with tracer.start_span("test_span") as span:
                    raise ValueError("Test error")
            
            # 验证异常信息被记录
            mock_span.set_status.assert_called_once()
            mock_span.set_attribute.assert_any_call("error.type", "ValueError")
            mock_span.set_attribute.assert_any_call("error.message", "Test error")
    
    def test_trace_api_call_disabled(self):
        """测试在禁用状态下追踪API调用"""
        tracer = OpenTelemetryTracer()
        tracer.enabled = False
        
        # 应该不会抛出异常
        tracer.trace_api_call("chat", "gpt-4", "openai")
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_trace_api_call_enabled(self):
        """测试在启用状态下追踪API调用"""
        tracer = OpenTelemetryTracer()
        mock_span = Mock()
        
        tracer.trace_api_call(
            "chat", "gpt-4", "openai", 
            span=mock_span,
            temperature=0.7,
            max_tokens=1000
        )
        
        # 验证属性被设置
        mock_span.set_attribute.assert_any_call("harborai.api.method", "chat")
        mock_span.set_attribute.assert_any_call("harborai.api.model", "gpt-4")
        mock_span.set_attribute.assert_any_call("harborai.api.provider", "openai")
        mock_span.set_attribute.assert_any_call("harborai.api.temperature", "0.7")
        mock_span.set_attribute.assert_any_call("harborai.api.max_tokens", "1000")
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_trace_token_usage(self):
        """测试Token使用量追踪"""
        tracer = OpenTelemetryTracer()
        mock_span = Mock()
        
        tracer.trace_token_usage(mock_span, 100, 50, 150)
        
        # 验证Token使用量属性被设置
        mock_span.set_attribute.assert_any_call("harborai.tokens.prompt", 100)
        mock_span.set_attribute.assert_any_call("harborai.tokens.completion", 50)
        mock_span.set_attribute.assert_any_call("harborai.tokens.total", 150)
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_trace_cost(self):
        """测试成本追踪"""
        tracer = OpenTelemetryTracer()
        mock_span = Mock()
        
        tracer.trace_cost(mock_span, 0.05, "USD")
        
        # 验证成本属性被设置
        mock_span.set_attribute.assert_any_call("harborai.cost.amount", 0.05)
        mock_span.set_attribute.assert_any_call("harborai.cost.currency", "USD")
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_trace_performance(self):
        """测试性能追踪"""
        tracer = OpenTelemetryTracer()
        mock_span = Mock()
        
        tracer.trace_performance(mock_span, 1500.5, 100.2, 1400.3)
        
        # 验证性能属性被设置
        mock_span.set_attribute.assert_any_call("harborai.performance.duration_ms", 1500.5)
        mock_span.set_attribute.assert_any_call("harborai.performance.queue_time_ms", 100.2)
        mock_span.set_attribute.assert_any_call("harborai.performance.processing_time_ms", 1400.3)
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_trace_performance_minimal(self):
        """测试最小性能追踪（只有持续时间）"""
        tracer = OpenTelemetryTracer()
        mock_span = Mock()
        
        tracer.trace_performance(mock_span, 1500.5)
        
        # 验证只有持续时间属性被设置
        mock_span.set_attribute.assert_called_with("harborai.performance.duration_ms", 1500.5)
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_add_event(self):
        """测试添加事件"""
        tracer = OpenTelemetryTracer()
        mock_span = Mock()
        
        tracer.add_event(mock_span, "test_event", {"key": "value"})
        
        # 验证事件被添加
        mock_span.add_event.assert_called_once_with("test_event", {"key": "value"})
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_add_event_no_attributes(self):
        """测试添加事件（无属性）"""
        tracer = OpenTelemetryTracer()
        mock_span = Mock()
        
        tracer.add_event(mock_span, "test_event")
        
        # 验证事件被添加，使用空字典作为属性
        mock_span.add_event.assert_called_once_with("test_event", {})
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_inject_context(self):
        """测试注入追踪上下文"""
        tracer = OpenTelemetryTracer()
        headers = {"Content-Type": "application/json"}
        
        with patch('harborai.monitoring.opentelemetry_tracer.inject') as mock_inject:
            result = tracer.inject_context(headers)
            
            mock_inject.assert_called_once_with(headers)
            assert result == headers
    
    def test_inject_context_disabled(self):
        """测试在禁用状态下注入追踪上下文"""
        tracer = OpenTelemetryTracer()
        tracer.enabled = False
        headers = {"Content-Type": "application/json"}
        
        result = tracer.inject_context(headers)
        assert result == headers
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_extract_context(self):
        """测试提取追踪上下文"""
        tracer = OpenTelemetryTracer()
        headers = {"traceparent": "00-trace-span-01"}
        
        with patch('harborai.monitoring.opentelemetry_tracer.extract') as mock_extract:
            mock_extract.return_value = "mock_context"
            result = tracer.extract_context(headers)
            
            mock_extract.assert_called_once_with(headers)
            assert result == "mock_context"
    
    def test_extract_context_disabled(self):
        """测试在禁用状态下提取追踪上下文"""
        tracer = OpenTelemetryTracer()
        tracer.enabled = False
        headers = {"traceparent": "00-trace-span-01"}
        
        result = tracer.extract_context(headers)
        assert result is None
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_shutdown(self):
        """测试关闭追踪器"""
        tracer = OpenTelemetryTracer()
        
        with patch.object(tracer, 'tracer_provider') as mock_provider:
            tracer.shutdown()
            mock_provider.shutdown.assert_called_once()


class TestGlobalFunctions:
    """全局函数测试"""
    
    def setup_method(self):
        """测试前设置"""
        # 重置全局追踪器
        import harborai.monitoring.opentelemetry_tracer as otel_module
        otel_module._otel_tracer = None
    
    def test_get_otel_tracer_none(self):
        """测试获取未初始化的追踪器"""
        tracer = get_otel_tracer()
        assert tracer is None
    
    def test_init_otel_tracer(self):
        """测试初始化全局追踪器"""
        tracer = init_otel_tracer(
            service_name="test_service",
            service_version="1.0.0"
        )
        
        assert tracer is not None
        if tracer.enabled:
            assert tracer.service_name == "test_service"
            assert tracer.service_version == "1.0.0"
        
        # 验证全局追踪器被设置
        global_tracer = get_otel_tracer()
        assert global_tracer == tracer


class TestOtelTraceDecorator:
    """otel_trace装饰器测试"""
    
    def setup_method(self):
        """测试前设置"""
        # 重置全局追踪器
        import harborai.monitoring.opentelemetry_tracer as otel_module
        otel_module._otel_tracer = None
    
    def test_decorator_no_tracer(self):
        """测试在没有追踪器的情况下使用装饰器"""
        @otel_trace()
        def test_function():
            return "result"
        
        result = test_function()
        assert result == "result"
    
    def test_decorator_disabled_tracer(self):
        """测试在禁用追踪器的情况下使用装饰器"""
        mock_tracer = Mock()
        mock_tracer.enabled = False
        
        with patch('harborai.monitoring.opentelemetry_tracer.get_otel_tracer', return_value=mock_tracer):
            @otel_trace()
            def test_function():
                return "result"
            
            result = test_function()
            assert result == "result"
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_decorator_enabled_tracer(self):
        """测试在启用追踪器的情况下使用装饰器"""
        mock_tracer = Mock()
        mock_tracer.enabled = True
        mock_span = Mock()
        
        # 设置上下文管理器
        mock_tracer.start_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_span.return_value.__exit__ = Mock(return_value=None)
        
        with patch('harborai.monitoring.opentelemetry_tracer.get_otel_tracer', return_value=mock_tracer):
            @otel_trace(operation_name="custom_operation", attributes={"key": "value"})
            def test_function(model="gpt-4", provider="openai"):
                return "result"
            
            result = test_function()
            
            assert result == "result"
            mock_tracer.start_span.assert_called_once()
            mock_tracer.trace_performance.assert_called_once()
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_decorator_with_exception(self):
        """测试装饰器处理异常的情况"""
        mock_tracer = Mock()
        mock_tracer.enabled = True
        mock_span = Mock()
        
        # 设置上下文管理器抛出异常
        mock_tracer.start_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_span.return_value.__exit__ = Mock(side_effect=ValueError("Test error"))
        
        with patch('harborai.monitoring.opentelemetry_tracer.get_otel_tracer', return_value=mock_tracer):
            @otel_trace()
            def test_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                test_function()
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_decorator_with_usage_result(self):
        """测试装饰器处理包含使用量信息的结果"""
        mock_tracer = Mock()
        mock_tracer.enabled = True
        mock_span = Mock()
        
        # 设置上下文管理器
        mock_tracer.start_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_span.return_value.__exit__ = Mock(return_value=None)
        
        # 模拟包含使用量信息的结果
        mock_usage = Mock()
        mock_usage.prompt_tokens = 100
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 150
        
        mock_result = Mock()
        mock_result.usage = mock_usage
        
        with patch('harborai.monitoring.opentelemetry_tracer.get_otel_tracer', return_value=mock_tracer):
            with patch('harborai.monitoring.opentelemetry_tracer.PricingCalculator') as mock_pricing:
                mock_pricing.calculate_cost.return_value = 0.05
                
                @otel_trace()
                def test_function(model="gpt-4"):
                    return mock_result
                
                result = test_function()
                
                assert result == mock_result
                mock_tracer.trace_token_usage.assert_called_once_with(
                    mock_span, 100, 50, 150
                )
                mock_tracer.trace_cost.assert_called_once_with(mock_span, 0.05)


class TestOtelTraceAsyncDecorator:
    """otel_trace_async装饰器测试"""
    
    def setup_method(self):
        """测试前设置"""
        # 重置全局追踪器
        import harborai.monitoring.opentelemetry_tracer as otel_module
        otel_module._otel_tracer = None
    
    @pytest.mark.asyncio
    async def test_async_decorator_no_tracer(self):
        """测试异步装饰器在没有追踪器的情况下"""
        @otel_trace_async()
        async def test_async_function():
            return "async_result"
        
        result = await test_async_function()
        assert result == "async_result"
    
    @pytest.mark.asyncio
    async def test_async_decorator_disabled_tracer(self):
        """测试异步装饰器在禁用追踪器的情况下"""
        mock_tracer = Mock()
        mock_tracer.enabled = False
        
        with patch('harborai.monitoring.opentelemetry_tracer.get_otel_tracer', return_value=mock_tracer):
            @otel_trace_async()
            async def test_async_function():
                return "async_result"
            
            result = await test_async_function()
            assert result == "async_result"
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    async def test_async_decorator_enabled_tracer(self):
        """测试异步装饰器在启用追踪器的情况下"""
        mock_tracer = Mock()
        mock_tracer.enabled = True
        mock_span = Mock()
        
        # 设置上下文管理器
        mock_tracer.start_span.return_value.__enter__ = Mock(return_value=mock_span)
        mock_tracer.start_span.return_value.__exit__ = Mock(return_value=None)
        
        with patch('harborai.monitoring.opentelemetry_tracer.get_otel_tracer', return_value=mock_tracer):
            @otel_trace_async(operation_name="async_operation")
            async def test_async_function(model="gpt-4", provider="openai"):
                return "async_result"
            
            result = await test_async_function()
            
            assert result == "async_result"
            mock_tracer.start_span.assert_called_once()
            mock_tracer.trace_performance.assert_called_once()


class TestEdgeCases:
    """边界条件和异常情况测试"""
    
    def test_none_span_operations(self):
        """测试传入None跨度的操作"""
        tracer = OpenTelemetryTracer()
        
        # 这些操作应该不会抛出异常
        tracer.trace_api_call("chat", "gpt-4", "openai", span=None)
        tracer.trace_token_usage(None, 100, 50, 150)
        tracer.trace_cost(None, 0.05)
        tracer.trace_performance(None, 1500.5)
        tracer.add_event(None, "test_event")
    
    def test_invalid_parameters(self):
        """测试无效参数"""
        tracer = OpenTelemetryTracer()
        
        # 如果追踪器被禁用，直接返回
        if not tracer.enabled:
            return
        
        mock_span = Mock()
        
        # 测试None值参数
        tracer.trace_api_call("chat", "gpt-4", "openai", span=mock_span, invalid_param=None)
        
        # 验证None值不会被设置为属性
        mock_span.set_attribute.assert_any_call("harborai.api.method", "chat")
        mock_span.set_attribute.assert_any_call("harborai.api.model", "gpt-4")
        mock_span.set_attribute.assert_any_call("harborai.api.provider", "openai")
        
        # 验证None值参数没有被设置
        call_args_list = [call[0] for call in mock_span.set_attribute.call_args_list]
        assert not any("invalid_param" in str(args) for args in call_args_list)
    
    @pytest.mark.skipif(not OTEL_AVAILABLE, reason="OpenTelemetry not available")
    def test_shutdown_without_tracer_provider(self):
        """测试在没有tracer_provider的情况下关闭"""
        tracer = OpenTelemetryTracer()
        # 删除tracer_provider属性
        if hasattr(tracer, 'tracer_provider'):
            delattr(tracer, 'tracer_provider')
        
        # 应该不会抛出异常
        tracer.shutdown()
    
    def test_unicode_and_special_characters(self):
        """测试Unicode和特殊字符处理"""
        tracer = OpenTelemetryTracer()
        
        # 如果追踪器被禁用，直接返回
        if not tracer.enabled:
            return
        
        mock_span = Mock()
        
        # 测试包含Unicode字符的操作名称和属性
        with patch.object(tracer, 'tracer') as mock_tracer:
            mock_tracer.start_as_current_span.return_value.__enter__ = Mock(return_value=mock_span)
            mock_tracer.start_as_current_span.return_value.__exit__ = Mock(return_value=None)
            
            with tracer.start_span("测试跨度", {"中文键": "中文值", "emoji": "🚀"}):
                pass
            
            # 验证Unicode字符被正确处理
            mock_tracer.start_as_current_span.assert_called_once()