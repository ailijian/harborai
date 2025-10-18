#!/usr/bin/env python3
"""
HarborAI分布式追踪模块

提供OpenTelemetry集成和分布式追踪功能，包括：
- 自动化追踪器初始化
- AI操作追踪
- 双Trace ID支持
- 追踪数据收集和存储
- 性能监控和指标记录

使用示例:
    from harborai.core.tracing import setup_tracing, get_tracer
    
    # 初始化追踪
    setup_tracing()
    
    # 获取追踪器
    tracer = get_tracer()
    
    # 追踪AI操作
    async with tracer.trace_ai_operation("ai.chat.completion", "deepseek", "deepseek-chat") as span_ctx:
        # 执行AI操作
        result = await ai_operation()
        
        # 记录指标
        await tracer.record_ai_metrics(span_ctx, token_usage, cost_info)

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import os
import atexit
from typing import Optional

import structlog
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

from .config import load_tracing_config, validate_tracing_config, get_sampler_from_config
from .opentelemetry_tracer import OpenTelemetryTracer, setup_global_tracer
from .data_collector import TracingDataCollector


# 全局变量
_tracer_provider: Optional[TracerProvider] = None
_is_initialized: bool = False
_logger = structlog.get_logger(__name__)


def setup_tracing(config_override: Optional[dict] = None) -> bool:
    """
    设置和初始化分布式追踪系统
    
    参数:
        config_override: 配置覆盖字典
        
    返回:
        bool: 初始化是否成功
    """
    global _tracer_provider, _is_initialized
    
    if _is_initialized:
        _logger.warning("追踪系统已经初始化，跳过重复初始化")
        return True
    
    try:
        # 加载配置
        config = load_tracing_config()
        
        # 应用配置覆盖
        if config_override:
            for key, value in config_override.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # 验证配置
        if not validate_tracing_config(config):
            _logger.error("追踪配置验证失败")
            return False
        
        if not config.enabled:
            _logger.info("追踪功能已禁用")
            return True
        
        # 创建资源
        resource_attributes = {
            "service.name": config.resource.service_name,
            "service.version": config.resource.service_version,
            "service.namespace": config.resource.service_namespace,
            "deployment.environment": config.resource.deployment_environment,
        }
        resource_attributes.update(config.resource.custom_attributes)
        
        resource = Resource.create(resource_attributes)
        
        # 创建采样器
        sampler = get_sampler_from_config(config.sampling)
        
        # 创建TracerProvider
        _tracer_provider = TracerProvider(
            resource=resource,
            sampler=sampler
        )
        
        # 配置导出器
        _setup_exporters(config)
        
        # 设置全局TracerProvider
        trace.set_tracer_provider(_tracer_provider)
        
        # 初始化HarborAI追踪器
        harborai_tracer = setup_global_tracer(config)
        
        # 初始化追踪数据收集器
        data_collector = TracingDataCollector()
        
        # 设置自动化仪表
        _setup_auto_instrumentation(config)
        
        # 注册清理函数
        atexit.register(shutdown_tracing)
        
        _is_initialized = True
        
        _logger.info(
            "分布式追踪系统初始化成功",
            service_name=config.resource.service_name,
            environment=config.resource.deployment_environment,
            otlp_endpoint=config.otlp.endpoint,
            sampling_strategy=config.sampling.strategy,
            sampling_ratio=config.sampling.ratio
        )
        
        return True
        
    except Exception as e:
        _logger.error(
            "分布式追踪系统初始化失败",
            error=str(e)
        )
        return False


def _setup_exporters(config) -> None:
    """设置追踪导出器"""
    if not _tracer_provider:
        return
    
    # 控制台导出器（调试模式）
    if config.console_exporter or config.debug_mode:
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        _tracer_provider.add_span_processor(console_processor)
        _logger.debug("已启用控制台追踪导出器")
    
    # OTLP导出器
    if config.otlp.endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=config.otlp.endpoint,
                headers=config.otlp.headers,
                timeout=config.otlp.timeout,
                compression=config.otlp.compression,
                insecure=config.otlp.insecure
            )
            
            otlp_processor = BatchSpanProcessor(
                otlp_exporter,
                max_export_batch_size=config.max_export_batch_size,
                export_timeout_millis=config.batch_export_timeout,
                max_queue_size=config.max_queue_size
            )
            
            _tracer_provider.add_span_processor(otlp_processor)
            
            _logger.info(
                "OTLP追踪导出器配置成功",
                endpoint=config.otlp.endpoint,
                compression=config.otlp.compression
            )
            
        except Exception as e:
            _logger.error(
                "OTLP追踪导出器配置失败",
                error=str(e),
                endpoint=config.otlp.endpoint
            )


def _setup_auto_instrumentation(config) -> None:
    """设置自动化仪表"""
    try:
        # FastAPI自动仪表
        FastAPIInstrumentor().instrument()
        _logger.debug("FastAPI自动仪表已启用")
        
        # HTTPX客户端自动仪表
        HTTPXClientInstrumentor().instrument()
        _logger.debug("HTTPX客户端自动仪表已启用")
        
        # SQLAlchemy自动仪表
        SQLAlchemyInstrumentor().instrument()
        _logger.debug("SQLAlchemy自动仪表已启用")
        
    except Exception as e:
        _logger.warning(
            "自动化仪表设置失败",
            error=str(e)
        )


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    获取OpenTelemetry追踪器
    
    参数:
        name: 追踪器名称
        
    返回:
        trace.Tracer: OpenTelemetry追踪器实例
    """
    return trace.get_tracer(name)


def get_harborai_tracer() -> Optional[OpenTelemetryTracer]:
    """
    获取HarborAI追踪器
    
    返回:
        Optional[OpenTelemetryTracer]: HarborAI追踪器实例
    """
    from .opentelemetry_tracer import get_global_tracer
    return get_global_tracer()


def get_data_collector() -> Optional[TracingDataCollector]:
    """
    获取追踪数据收集器
    
    返回:
        Optional[TracingDataCollector]: 追踪数据收集器实例
    """
    try:
        return TracingDataCollector()
    except Exception as e:
        _logger.error(
            "获取追踪数据收集器失败",
            error=str(e)
        )
        return None


def is_tracing_enabled() -> bool:
    """
    检查追踪是否已启用
    
    返回:
        bool: 追踪是否已启用
    """
    return _is_initialized and _tracer_provider is not None


def shutdown_tracing() -> None:
    """关闭追踪系统并清理资源"""
    global _tracer_provider, _is_initialized
    
    if not _is_initialized:
        return
    
    try:
        # 关闭HarborAI追踪器
        harborai_tracer = get_harborai_tracer()
        if harborai_tracer:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，创建任务
                    loop.create_task(harborai_tracer.shutdown())
                else:
                    # 如果事件循环未运行，直接运行
                    loop.run_until_complete(harborai_tracer.shutdown())
            except RuntimeError:
                # 如果没有事件循环，创建新的
                asyncio.run(harborai_tracer.shutdown())
        
        # 强制导出所有待处理的span
        if _tracer_provider:
            _tracer_provider.force_flush(timeout_millis=5000)
            _tracer_provider.shutdown()
        
        _is_initialized = False
        _tracer_provider = None
        
        _logger.info("分布式追踪系统已关闭")
        
    except Exception as e:
        _logger.error(
            "关闭追踪系统失败",
            error=str(e)
        )


# 导出主要接口
__all__ = [
    "setup_tracing",
    "get_tracer", 
    "get_harborai_tracer",
    "get_data_collector",
    "is_tracing_enabled",
    "shutdown_tracing",
    "OpenTelemetryTracer",
    "TracingDataCollector"
]