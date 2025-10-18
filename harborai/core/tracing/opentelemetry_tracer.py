#!/usr/bin/env python3
"""
OpenTelemetry分布式追踪集成模块

实现OpenTelemetry分布式追踪功能，支持：
- AI操作的追踪span创建和管理
- 双Trace ID策略（HarborAI + OpenTelemetry）
- 追踪上下文传播和管理
- 性能指标和标签记录
- 与现有日志系统的无缝集成

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import time
import uuid
import asyncio
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ..token_usage import TokenUsage


@dataclass
class TracingConfig:
    """追踪配置类"""
    enabled: bool = True
    service_name: str = "harborai-logging"
    service_version: str = "2.0.0"
    environment: str = "production"
    
    # OpenTelemetry配置
    otlp_endpoint: Optional[str] = None
    otlp_headers: Dict[str, str] = field(default_factory=dict)
    batch_export_timeout: int = 30000  # 30秒
    max_export_batch_size: int = 512
    
    # 采样配置
    sampling_ratio: float = 1.0  # 100%采样
    
    # HarborAI特定配置
    hb_trace_id_prefix: str = "hb"
    include_sensitive_data: bool = False


@dataclass
class AISpanContext:
    """AI操作Span上下文"""
    hb_trace_id: str
    otel_trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    operation_name: str = "ai.chat.completion"
    provider: Optional[str] = None
    model: Optional[str] = None
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, Any] = field(default_factory=dict)


class OpenTelemetryTracer:
    """
    OpenTelemetry分布式追踪集成器
    
    功能：
    1. 创建和管理AI操作的追踪span
    2. 实现双Trace ID策略
    3. 记录性能指标和业务标签
    4. 支持追踪上下文传播
    5. 与现有日志系统集成
    """
    
    def __init__(self, config: TracingConfig):
        """
        初始化OpenTelemetry追踪器
        
        参数:
            config: 追踪配置
        """
        self.config = config
        self.logger = structlog.get_logger(__name__)
        self._tracer: Optional[trace.Tracer] = None
        self._tracer_provider: Optional[TracerProvider] = None
        self._propagator = TraceContextTextMapPropagator()
        
        # 初始化追踪器
        if config.enabled:
            self._setup_tracer()
    
    def _setup_tracer(self) -> None:
        """设置OpenTelemetry追踪器"""
        try:
            # 创建资源信息
            resource = Resource.create({
                "service.name": self.config.service_name,
                "service.version": self.config.service_version,
                "deployment.environment": self.config.environment,
                "ai.system": "harborai",
                "ai.version": self.config.service_version,
            })
            
            # 创建TracerProvider
            self._tracer_provider = TracerProvider(resource=resource)
            
            # 配置OTLP导出器（如果提供了endpoint）
            if self.config.otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(
                    endpoint=self.config.otlp_endpoint,
                    headers=self.config.otlp_headers
                )
                
                span_processor = BatchSpanProcessor(
                    otlp_exporter,
                    max_export_batch_size=self.config.max_export_batch_size,
                    export_timeout_millis=self.config.batch_export_timeout
                )
                
                self._tracer_provider.add_span_processor(span_processor)
            
            # 设置全局TracerProvider
            trace.set_tracer_provider(self._tracer_provider)
            
            # 获取Tracer实例
            self._tracer = trace.get_tracer(
                __name__,
                version=self.config.service_version
            )
            
            self.logger.info(
                "OpenTelemetry追踪器初始化成功",
                service_name=self.config.service_name,
                otlp_endpoint=self.config.otlp_endpoint,
                sampling_ratio=self.config.sampling_ratio
            )
            
        except Exception as e:
            self.logger.error(
                "OpenTelemetry追踪器初始化失败",
                error=str(e),
                config=self.config
            )
            self._tracer = None
    
    def _generate_hb_trace_id(self) -> str:
        """生成HarborAI内部追踪ID"""
        timestamp = int(time.time() * 1000)  # 毫秒时间戳
        random_part = uuid.uuid4().hex[:8]
        return f"{self.config.hb_trace_id_prefix}_{timestamp}_{random_part}"
    
    def _extract_otel_trace_id(self, span: trace.Span) -> str:
        """从OpenTelemetry Span提取追踪ID"""
        if not span or not span.get_span_context():
            return ""
        
        trace_id = span.get_span_context().trace_id
        return f"{trace_id:032x}"
    
    def _extract_span_id(self, span: trace.Span) -> str:
        """从OpenTelemetry Span提取Span ID"""
        if not span or not span.get_span_context():
            return ""
        
        span_id = span.get_span_context().span_id
        return f"{span_id:016x}"
    
    async def create_ai_span(
        self,
        operation_name: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        trace_context: Optional[Dict[str, str]] = None,
        parent_span: Optional[trace.Span] = None
    ) -> Optional[AISpanContext]:
        """
        创建AI操作的追踪span
        
        参数:
            operation_name: 操作名称，如 'ai.chat.completion'
            provider: AI提供商，如 'deepseek', 'openai'
            model: 模型名称，如 'deepseek-chat'
            trace_context: 外部追踪上下文
            parent_span: 父span
            
        返回:
            AISpanContext: AI Span上下文，如果追踪未启用则返回None
        """
        if not self.config.enabled or not self._tracer:
            return None
        
        try:
            # 生成HarborAI追踪ID
            hb_trace_id = self._generate_hb_trace_id()
            
            # 创建span
            with self._tracer.start_as_current_span(
                operation_name,
                context=trace.set_span_in_context(parent_span) if parent_span else None
            ) as span:
                
                # 设置基础属性
                span.set_attribute("ai.system", "harborai")
                span.set_attribute("ai.version", self.config.service_version)
                span.set_attribute("ai.operation", operation_name)
                span.set_attribute("service.name", self.config.service_name)
                span.set_attribute("harborai.trace_id", hb_trace_id)
                
                # 设置AI相关属性
                if provider:
                    span.set_attribute("ai.provider", provider)
                if model:
                    span.set_attribute("ai.model", model)
                
                # 设置环境信息
                span.set_attribute("deployment.environment", self.config.environment)
                
                # 提取OpenTelemetry追踪信息
                otel_trace_id = self._extract_otel_trace_id(span)
                span_id = self._extract_span_id(span)
                
                # 获取父span ID
                parent_span_id = None
                if parent_span:
                    parent_span_id = self._extract_span_id(parent_span)
                
                # 创建AI Span上下文
                span_context = AISpanContext(
                    hb_trace_id=hb_trace_id,
                    otel_trace_id=otel_trace_id,
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                    operation_name=operation_name,
                    provider=provider,
                    model=model,
                    tags={
                        "ai.system": "harborai",
                        "ai.version": self.config.service_version,
                        "service.name": self.config.service_name,
                        "deployment.environment": self.config.environment
                    }
                )
                
                self.logger.debug(
                    "AI追踪span创建成功",
                    hb_trace_id=hb_trace_id,
                    otel_trace_id=otel_trace_id,
                    span_id=span_id,
                    operation_name=operation_name,
                    provider=provider,
                    model=model
                )
                
                return span_context
                
        except Exception as e:
            self.logger.error(
                "创建AI追踪span失败",
                error=str(e),
                operation_name=operation_name,
                provider=provider,
                model=model
            )
            return None
    
    async def record_ai_metrics(
        self,
        span_context: AISpanContext,
        token_usage: Optional[TokenUsage] = None,
        cost_info: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None,
        custom_tags: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        记录AI操作的指标和标签
        
        参数:
            span_context: AI Span上下文
            token_usage: Token使用量信息
            cost_info: 成本信息
            performance_metrics: 性能指标
            custom_tags: 自定义标签
        """
        if not self.config.enabled or not self._tracer:
            return
        
        try:
            # 获取当前span
            current_span = trace.get_current_span()
            if not current_span or not current_span.is_recording():
                return
            
            # 记录Token使用量
            if token_usage:
                current_span.set_attribute("ai.usage.prompt_tokens", token_usage.prompt_tokens)
                current_span.set_attribute("ai.usage.completion_tokens", token_usage.completion_tokens)
                current_span.set_attribute("ai.usage.total_tokens", token_usage.total_tokens)
                current_span.set_attribute("ai.usage.parsing_method", token_usage.parsing_method)
                current_span.set_attribute("ai.usage.confidence", token_usage.confidence)
            
            # 记录成本信息
            if cost_info:
                current_span.set_attribute("ai.cost.input_cost", float(cost_info.get("input_cost", 0)))
                current_span.set_attribute("ai.cost.output_cost", float(cost_info.get("output_cost", 0)))
                current_span.set_attribute("ai.cost.total_cost", float(cost_info.get("total_cost", 0)))
                current_span.set_attribute("ai.cost.currency", cost_info.get("currency", "CNY"))
                current_span.set_attribute("ai.cost.pricing_source", cost_info.get("pricing_source", "unknown"))
            
            # 记录性能指标
            if performance_metrics:
                for key, value in performance_metrics.items():
                    if isinstance(value, (int, float)):
                        current_span.set_attribute(f"ai.performance.{key}", value)
            
            # 记录自定义标签
            if custom_tags:
                for key, value in custom_tags.items():
                    if isinstance(value, (str, int, float, bool)):
                        current_span.set_attribute(f"ai.custom.{key}", value)
            
            # 更新span上下文的标签
            span_context.tags.update({
                "token_usage_recorded": token_usage is not None,
                "cost_info_recorded": cost_info is not None,
                "performance_metrics_recorded": performance_metrics is not None,
                "custom_tags_recorded": custom_tags is not None
            })
            
            self.logger.debug(
                "AI指标记录成功",
                hb_trace_id=span_context.hb_trace_id,
                otel_trace_id=span_context.otel_trace_id,
                token_usage_recorded=token_usage is not None,
                cost_info_recorded=cost_info is not None
            )
            
        except Exception as e:
            self.logger.error(
                "记录AI指标失败",
                error=str(e),
                hb_trace_id=span_context.hb_trace_id if span_context else None
            )
    
    async def finish_span(
        self,
        span_context: AISpanContext,
        status: str = "ok",
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        完成追踪span并返回追踪信息
        
        参数:
            span_context: AI Span上下文
            status: 操作状态 ('ok', 'error', 'timeout', 'cancelled')
            error_message: 错误信息（如果有）
            
        返回:
            Dict: 追踪信息摘要
        """
        if not self.config.enabled or not span_context:
            return {}
        
        try:
            # 获取当前span
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                
                # 设置span状态
                if status == "ok":
                    current_span.set_status(Status(StatusCode.OK))
                else:
                    current_span.set_status(Status(StatusCode.ERROR, error_message or f"Operation failed with status: {status}"))
                    if error_message:
                        current_span.set_attribute("error.message", error_message)
                
                # 设置最终状态
                current_span.set_attribute("ai.status", status)
                
                # 计算持续时间
                end_time = datetime.now(timezone.utc)
                duration_ms = int((end_time - span_context.start_time).total_seconds() * 1000)
                current_span.set_attribute("ai.duration_ms", duration_ms)
                
                # 结束span
                current_span.end()
                
                # 返回追踪信息摘要
                tracing_summary = {
                    "hb_trace_id": span_context.hb_trace_id,
                    "otel_trace_id": span_context.otel_trace_id,
                    "span_id": span_context.span_id,
                    "parent_span_id": span_context.parent_span_id,
                    "operation_name": span_context.operation_name,
                    "provider": span_context.provider,
                    "model": span_context.model,
                    "start_time": span_context.start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_ms": duration_ms,
                    "status": status,
                    "tags": span_context.tags
                }
                
                self.logger.debug(
                    "AI追踪span完成",
                    **tracing_summary
                )
                
                return tracing_summary
            
        except Exception as e:
            self.logger.error(
                "完成追踪span失败",
                error=str(e),
                hb_trace_id=span_context.hb_trace_id if span_context else None
            )
        
        return {}
    
    @asynccontextmanager
    async def trace_ai_operation(
        self,
        operation_name: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        trace_context: Optional[Dict[str, str]] = None
    ):
        """
        AI操作追踪的上下文管理器
        
        使用示例:
            async with tracer.trace_ai_operation("ai.chat.completion", "deepseek", "deepseek-chat") as span_ctx:
                # 执行AI操作
                result = await ai_operation()
                
                # 记录指标
                await tracer.record_ai_metrics(span_ctx, token_usage, cost_info)
        """
        span_context = None
        try:
            # 创建span
            span_context = await self.create_ai_span(
                operation_name=operation_name,
                provider=provider,
                model=model,
                trace_context=trace_context
            )
            
            yield span_context
            
            # 正常完成
            if span_context:
                await self.finish_span(span_context, status="ok")
                
        except Exception as e:
            # 异常完成
            if span_context:
                await self.finish_span(
                    span_context,
                    status="error",
                    error_message=str(e)
                )
            raise
    
    def extract_trace_context(self, headers: Dict[str, str]) -> Optional[Dict[str, str]]:
        """
        从HTTP头部提取追踪上下文
        
        参数:
            headers: HTTP请求头部
            
        返回:
            Dict: 追踪上下文，如果没有则返回None
        """
        if not self.config.enabled:
            return None
        
        try:
            # 使用TraceContext传播器提取上下文
            context = self._propagator.extract(headers)
            if context:
                return {"trace_context": context}
        except Exception as e:
            self.logger.debug(
                "提取追踪上下文失败",
                error=str(e),
                headers=headers
            )
        
        return None
    
    def inject_trace_context(self, span_context: AISpanContext) -> Dict[str, str]:
        """
        将追踪上下文注入到HTTP头部
        
        参数:
            span_context: AI Span上下文
            
        返回:
            Dict: 包含追踪信息的HTTP头部
        """
        headers = {}
        
        if not self.config.enabled or not span_context:
            return headers
        
        try:
            # 获取当前span的上下文
            current_span = trace.get_current_span()
            if current_span:
                context = trace.set_span_in_context(current_span)
                self._propagator.inject(headers, context)
            
            # 添加HarborAI特定的追踪头部
            headers["X-HarborAI-Trace-ID"] = span_context.hb_trace_id
            headers["X-HarborAI-Span-ID"] = span_context.span_id
            
        except Exception as e:
            self.logger.debug(
                "注入追踪上下文失败",
                error=str(e),
                hb_trace_id=span_context.hb_trace_id
            )
        
        return headers
    
    async def shutdown(self) -> None:
        """关闭追踪器并清理资源"""
        if self._tracer_provider:
            try:
                # 强制导出所有待处理的span
                self._tracer_provider.force_flush(timeout_millis=5000)
                
                # 关闭TracerProvider
                self._tracer_provider.shutdown()
                
                self.logger.info("OpenTelemetry追踪器已关闭")
                
            except Exception as e:
                self.logger.error(
                    "关闭OpenTelemetry追踪器失败",
                    error=str(e)
                )


# 全局追踪器实例
_global_tracer: Optional[OpenTelemetryTracer] = None


def get_global_tracer() -> Optional[OpenTelemetryTracer]:
    """获取全局追踪器实例"""
    return _global_tracer


def setup_global_tracer(config: TracingConfig) -> OpenTelemetryTracer:
    """设置全局追踪器实例"""
    global _global_tracer
    _global_tracer = OpenTelemetryTracer(config)
    return _global_tracer


def create_tracing_config_from_env() -> TracingConfig:
    """从环境变量创建追踪配置"""
    import os
    
    return TracingConfig(
        enabled=os.getenv("OTEL_ENABLED", "true").lower() == "true",
        service_name=os.getenv("OTEL_SERVICE_NAME", "harborai-logging"),
        service_version=os.getenv("OTEL_SERVICE_VERSION", "2.0.0"),
        environment=os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
        otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        otlp_headers={
            k[len("OTEL_EXPORTER_OTLP_HEADERS_"):]: v
            for k, v in os.environ.items()
            if k.startswith("OTEL_EXPORTER_OTLP_HEADERS_")
        },
        sampling_ratio=float(os.getenv("OTEL_SAMPLING_RATIO", "1.0")),
        hb_trace_id_prefix=os.getenv("HARBORAI_TRACE_ID_PREFIX", "hb"),
        include_sensitive_data=os.getenv("HARBORAI_INCLUDE_SENSITIVE_DATA", "false").lower() == "true"
    )