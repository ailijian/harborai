#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenTelemetry集成模块

提供分布式追踪、性能瓶颈识别和调用链分析功能。
"""

import time
import functools
from typing import Optional, Dict, Any, Callable, Union
from contextlib import contextmanager

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.propagate import inject, extract
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    trace = None

from ..utils.logger import get_logger
from ..utils.tracer import get_current_trace_id, generate_trace_id
from ..utils.exceptions import HarborAIError

logger = get_logger(__name__)

# 全局OpenTelemetry追踪器实例
_otel_tracer: Optional['OpenTelemetryTracer'] = None


class OpenTelemetryTracer:
    """OpenTelemetry分布式追踪器"""
    
    def __init__(self, service_name: str = "harborai", 
                 jaeger_endpoint: Optional[str] = None,
                 otlp_endpoint: Optional[str] = None,
                 service_version: str = "1.0.0"):
        """初始化OpenTelemetry追踪器
        
        Args:
            service_name: 服务名称
            jaeger_endpoint: Jaeger导出器端点
            otlp_endpoint: OTLP导出器端点
            service_version: 服务版本
        """
        if not OTEL_AVAILABLE:
            logger.warning("OpenTelemetry未安装，追踪功能将被禁用")
            self.enabled = False
            return
        
        self.service_name = service_name
        self.service_version = service_version
        self.enabled = True
        
        # 创建资源
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: service_version,
        })
        
        # 创建追踪器提供者
        self.tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(self.tracer_provider)
        
        # 配置导出器
        self._setup_exporters(jaeger_endpoint, otlp_endpoint)
        
        # 获取追踪器
        self.tracer = trace.get_tracer(__name__)
        
        logger.info(f"OpenTelemetry追踪器已初始化，服务名称: {service_name}")
    
    def _setup_exporters(self, jaeger_endpoint: Optional[str], 
                        otlp_endpoint: Optional[str]):
        """设置追踪数据导出器
        
        Args:
            jaeger_endpoint: Jaeger端点
            otlp_endpoint: OTLP端点
        """
        if jaeger_endpoint:
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=6831,
                    collector_endpoint=jaeger_endpoint,
                )
                span_processor = BatchSpanProcessor(jaeger_exporter)
                self.tracer_provider.add_span_processor(span_processor)
                logger.info(f"Jaeger导出器已配置: {jaeger_endpoint}")
            except Exception as e:
                logger.warning(f"配置Jaeger导出器失败: {e}")
        
        if otlp_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=otlp_endpoint,
                    insecure=True
                )
                span_processor = BatchSpanProcessor(otlp_exporter)
                self.tracer_provider.add_span_processor(span_processor)
                logger.info(f"OTLP导出器已配置: {otlp_endpoint}")
            except Exception as e:
                logger.warning(f"配置OTLP导出器失败: {e}")
    
    @contextmanager
    def start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None,
                   parent_context: Optional[Any] = None):
        """启动一个新的追踪跨度
        
        Args:
            name: 跨度名称
            attributes: 跨度属性
            parent_context: 父级上下文
            
        Yields:
            追踪跨度对象
        """
        if not self.enabled:
            yield None
            return
        
        # 获取或生成trace_id
        trace_id = get_current_trace_id() or generate_trace_id()
        
        # 设置默认属性
        span_attributes = {
            "harborai.trace_id": trace_id,
            "harborai.service": self.service_name,
            "harborai.version": self.service_version,
        }
        
        if attributes:
            span_attributes.update(attributes)
        
        with self.tracer.start_as_current_span(
            name, 
            context=parent_context,
            attributes=span_attributes
        ) as span:
            try:
                yield span
            except Exception as e:
                # 记录异常信息
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.set_attribute("error.type", type(e).__name__)
                span.set_attribute("error.message", str(e))
                raise
    
    def trace_api_call(self, method: str, model: str, provider: str,
                      span: Optional[Any] = None, **kwargs):
        """为API调用添加追踪信息
        
        Args:
            method: API方法名
            model: 模型名称
            provider: 提供商名称
            span: 当前跨度对象
            **kwargs: 其他属性
        """
        if not self.enabled or not span:
            return
        
        # 设置API调用相关属性
        span.set_attribute("harborai.api.method", method)
        span.set_attribute("harborai.api.model", model)
        span.set_attribute("harborai.api.provider", provider)
        
        # 添加其他属性
        for key, value in kwargs.items():
            if value is not None:
                span.set_attribute(f"harborai.api.{key}", str(value))
    
    def trace_token_usage(self, span: Optional[Any], prompt_tokens: int,
                         completion_tokens: int, total_tokens: int):
        """记录Token使用量追踪信息
        
        Args:
            span: 当前跨度对象
            prompt_tokens: 输入Token数量
            completion_tokens: 输出Token数量
            total_tokens: 总Token数量
        """
        if not self.enabled or not span:
            return
        
        span.set_attribute("harborai.tokens.prompt", prompt_tokens)
        span.set_attribute("harborai.tokens.completion", completion_tokens)
        span.set_attribute("harborai.tokens.total", total_tokens)
    
    def trace_cost(self, span: Optional[Any], cost: float, currency: str = "USD"):
        """记录成本追踪信息
        
        Args:
            span: 当前跨度对象
            cost: 成本金额
            currency: 货币类型
        """
        if not self.enabled or not span:
            return
        
        span.set_attribute("harborai.cost.amount", cost)
        span.set_attribute("harborai.cost.currency", currency)
    
    def trace_performance(self, span: Optional[Any], duration_ms: float,
                         queue_time_ms: Optional[float] = None,
                         processing_time_ms: Optional[float] = None):
        """记录性能追踪信息
        
        Args:
            span: 当前跨度对象
            duration_ms: 总持续时间（毫秒）
            queue_time_ms: 队列等待时间（毫秒）
            processing_time_ms: 处理时间（毫秒）
        """
        if not self.enabled or not span:
            return
        
        span.set_attribute("harborai.performance.duration_ms", duration_ms)
        
        if queue_time_ms is not None:
            span.set_attribute("harborai.performance.queue_time_ms", queue_time_ms)
        
        if processing_time_ms is not None:
            span.set_attribute("harborai.performance.processing_time_ms", processing_time_ms)
    
    def add_event(self, span: Optional[Any], name: str, 
                  attributes: Optional[Dict[str, Any]] = None):
        """向跨度添加事件
        
        Args:
            span: 当前跨度对象
            name: 事件名称
            attributes: 事件属性
        """
        if not self.enabled or not span:
            return
        
        span.add_event(name, attributes or {})
    
    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """将追踪上下文注入到HTTP头中
        
        Args:
            headers: HTTP头字典
            
        Returns:
            包含追踪上下文的HTTP头字典
        """
        if not self.enabled:
            return headers
        
        inject(headers)
        return headers
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[Any]:
        """从HTTP头中提取追踪上下文
        
        Args:
            headers: HTTP头字典
            
        Returns:
            追踪上下文对象
        """
        if not self.enabled:
            return None
        
        return extract(headers)
    
    def shutdown(self):
        """关闭追踪器并清理资源"""
        if self.enabled and hasattr(self, 'tracer_provider'):
            self.tracer_provider.shutdown()
            logger.info("OpenTelemetry追踪器已关闭")


def get_otel_tracer() -> Optional[OpenTelemetryTracer]:
    """获取全局OpenTelemetry追踪器实例
    
    Returns:
        OpenTelemetryTracer实例，如果未初始化则返回None
    """
    return _otel_tracer


def init_otel_tracer(service_name: str = "harborai",
                     jaeger_endpoint: Optional[str] = None,
                     otlp_endpoint: Optional[str] = None,
                     service_version: str = "1.0.0") -> OpenTelemetryTracer:
    """初始化全局OpenTelemetry追踪器实例
    
    Args:
        service_name: 服务名称
        jaeger_endpoint: Jaeger导出器端点
        otlp_endpoint: OTLP导出器端点
        service_version: 服务版本
        
    Returns:
        OpenTelemetryTracer实例
    """
    global _otel_tracer
    _otel_tracer = OpenTelemetryTracer(
        service_name=service_name,
        jaeger_endpoint=jaeger_endpoint,
        otlp_endpoint=otlp_endpoint,
        service_version=service_version
    )
    return _otel_tracer


def otel_trace(operation_name: Optional[str] = None,
               attributes: Optional[Dict[str, Any]] = None):
    """OpenTelemetry追踪装饰器
    
    Args:
        operation_name: 操作名称，默认使用函数名
        attributes: 额外的跨度属性
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_otel_tracer()
            if not tracer or not tracer.enabled:
                return func(*args, **kwargs)
            
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # 提取函数参数作为属性
            func_attributes = {
                "function.name": func.__name__,
                "function.module": func.__module__,
            }
            
            # 添加自定义属性
            if attributes:
                func_attributes.update(attributes)
            
            # 从kwargs中提取常用参数
            if 'model' in kwargs:
                func_attributes['harborai.model'] = kwargs['model']
            if 'provider' in kwargs:
                func_attributes['harborai.provider'] = kwargs['provider']
            
            with tracer.start_span(name, func_attributes) as span:
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # 记录成功执行的性能信息
                    duration_ms = (time.time() - start_time) * 1000
                    tracer.trace_performance(span, duration_ms)
                    
                    # 记录Token使用量和成本（如果结果包含这些信息）
                    if hasattr(result, 'usage') and result.usage:
                        usage = result.usage
                        tracer.trace_token_usage(
                            span,
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            usage.total_tokens
                        )
                        
                        # 计算并记录成本
                        from ..core.pricing import PricingCalculator
                        model = kwargs.get('model', 'unknown')
                        cost = PricingCalculator.calculate_cost(
                            input_tokens=usage.prompt_tokens,
                            output_tokens=usage.completion_tokens,
                            model_name=model
                        )
                        if cost is not None:
                            tracer.trace_cost(span, cost)
                    
                    return result
                    
                except Exception as e:
                    # 异常已在start_span中处理
                    raise
        
        return wrapper
    
    return decorator


# 异步版本的装饰器
def otel_trace_async(operation_name: Optional[str] = None,
                    attributes: Optional[Dict[str, Any]] = None):
    """异步OpenTelemetry追踪装饰器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_otel_tracer()
            if not tracer or not tracer.enabled:
                return await func(*args, **kwargs)
            
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            # 提取函数参数作为属性
            func_attributes = {
                "function.name": func.__name__,
                "function.module": func.__module__,
            }
            
            # 添加自定义属性
            if attributes:
                func_attributes.update(attributes)
            
            # 从kwargs中提取常用参数
            if 'model' in kwargs:
                func_attributes['harborai.model'] = kwargs['model']
            if 'provider' in kwargs:
                func_attributes['harborai.provider'] = kwargs['provider']
            
            with tracer.start_span(name, func_attributes) as span:
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    
                    # 记录成功执行的性能信息
                    duration_ms = (time.time() - start_time) * 1000
                    tracer.trace_performance(span, duration_ms)
                    
                    # 记录Token使用量和成本（如果结果包含这些信息）
                    if hasattr(result, 'usage') and result.usage:
                        usage = result.usage
                        tracer.trace_token_usage(
                            span,
                            usage.prompt_tokens,
                            usage.completion_tokens,
                            usage.total_tokens
                        )
                        
                        # 计算并记录成本
                        from ..core.pricing import PricingCalculator
                        model = kwargs.get('model', 'unknown')
                        cost = PricingCalculator.calculate_cost(
                            input_tokens=usage.prompt_tokens,
                            output_tokens=usage.completion_tokens,
                            model_name=model
                        )
                        if cost is not None:
                            tracer.trace_cost(span, cost)
                    
                    return result
                    
                except Exception as e:
                    # 异常已在start_span中处理
                    raise
        
        return wrapper
    
    return decorator