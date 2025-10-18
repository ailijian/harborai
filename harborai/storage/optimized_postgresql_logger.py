#!/usr/bin/env python3
"""
优化的PostgreSQL日志记录器

基于现有PostgreSQLLogger的增强版本，专注于：
- 分布式追踪集成
- Token字段对齐和成本细分
- 性能优化和批量处理
- 追踪信息存储

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from queue import Queue
from threading import Thread
from dataclasses import asdict

import structlog
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .postgres_logger import PostgreSQLLogger, DateTimeEncoder
from ..core.tracing.dual_trace_manager import DualTraceIDManager, DualTraceContext
from ..core.tracing.data_collector import TracingDataCollector, TracingRecord
from ..utils.logger import get_logger
from ..utils.exceptions import StorageError
from ..utils.timestamp import get_unified_timestamp

logger = get_logger(__name__)


class OptimizedPostgreSQLLogger(PostgreSQLLogger):
    """基于现有PostgreSQLLogger的优化版本
    
    增强功能：
    - 分布式追踪集成
    - Token字段对齐（prompt_tokens, completion_tokens）
    - 成本细分存储
    - 追踪信息记录
    - 性能优化
    """
    
    def __init__(self, 
                 connection_string: str,
                 table_name: str = "harborai_logs",
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 error_callback: Optional[Callable[[Exception], None]] = None,
                 enable_tracing: bool = True,
                 tracing_sample_rate: float = 1.0):
        """初始化优化的PostgreSQL日志记录器
        
        Args:
            connection_string: PostgreSQL连接字符串
            table_name: 日志表名
            batch_size: 批量写入大小
            flush_interval: 刷新间隔（秒）
            error_callback: 错误回调函数
            enable_tracing: 是否启用分布式追踪
            tracing_sample_rate: 追踪采样率
        """
        super().__init__(
            connection_string=connection_string,
            table_name=table_name,
            batch_size=batch_size,
            flush_interval=flush_interval,
            error_callback=error_callback
        )
        
        # 追踪相关配置
        self.enable_tracing = enable_tracing
        self.tracing_sample_rate = tracing_sample_rate
        
        # 初始化追踪组件
        if self.enable_tracing:
            self.dual_trace_manager = DualTraceIDManager()
            self.tracing_collector = TracingDataCollector(connection_string)
            self.tracer = trace.get_tracer(__name__)
        else:
            self.dual_trace_manager = None
            self.tracing_collector = None
            self.tracer = None
    
    def log_request_with_tracing(self, 
                               trace_id: str,
                               model: str,
                               messages: List[Dict[str, Any]],
                               provider: str = "unknown",
                               operation_name: str = "ai.chat.completion",
                               **kwargs) -> Optional[DualTraceContext]:
        """记录请求日志并创建追踪span
        
        Args:
            trace_id: HarborAI追踪ID
            model: 模型名称
            messages: 消息列表
            provider: AI提供商
            operation_name: 操作名称
            **kwargs: 其他参数
            
        Returns:
            DualTraceContext: 双追踪上下文（如果启用追踪）
        """
        dual_context = None
        
        # 创建追踪span
        if self.enable_tracing and self.tracer:
            with self.tracer.start_as_current_span(operation_name) as span:
                try:
                    # 创建双追踪上下文
                    dual_context = self.dual_trace_manager.create_trace_context(
                        hb_trace_id=trace_id,
                        operation_name=operation_name,
                        service_name="harborai-logging"
                    )
                    
                    # 设置span属性
                    span.set_attributes({
                        "ai.provider": provider,
                        "ai.model": model,
                        "ai.operation": operation_name,
                        "harborai.trace_id": trace_id,
                        "harborai.message_count": len(messages)
                    })
                    
                    # 记录请求日志
                    self.log_request(
                        trace_id=trace_id,
                        model=model,
                        messages=messages,
                        **kwargs
                    )
                    
                    span.set_status(Status(StatusCode.OK))
                    
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    logger.error(
                        "Failed to log request with tracing",
                        extra={
                            "trace_id": trace_id,
                            "error": str(e),
                            "provider": provider,
                            "model": model
                        }
                    )
                    raise
        else:
            # 不启用追踪时，直接记录日志
            self.log_request(
                trace_id=trace_id,
                model=model,
                messages=messages,
                **kwargs
            )
        
        return dual_context
    
    def log_response_with_cost_breakdown(self,
                                       trace_id: str,
                                       response: Dict[str, Any],
                                       cost_breakdown: Optional[Dict[str, Any]] = None,
                                       token_usage: Optional[Dict[str, Any]] = None,
                                       dual_context: Optional[DualTraceContext] = None,
                                       **kwargs):
        """记录响应日志并包含成本细分
        
        Args:
            trace_id: 追踪ID
            response: 响应数据
            cost_breakdown: 成本细分信息
            token_usage: Token使用情况
            dual_context: 双追踪上下文
            **kwargs: 其他参数
        """
        # 标准化token字段名称
        if token_usage:
            standardized_usage = self._standardize_token_fields(token_usage)
            kwargs.update(standardized_usage)
        
        # 添加成本细分信息
        if cost_breakdown:
            kwargs.update({
                "cost_breakdown": cost_breakdown,
                "total_cost": cost_breakdown.get("total_cost", 0.0),
                "currency": cost_breakdown.get("currency", "USD")
            })
        
        # 记录响应日志
        self.log_response(
            trace_id=trace_id,
            response=response,
            **kwargs
        )
        
        # 记录追踪信息
        if self.enable_tracing and dual_context and self.tracing_collector:
            try:
                tracing_record = TracingRecord(
                    hb_trace_id=dual_context.hb_trace_id,
                    otel_trace_id=dual_context.otel_trace_id,
                    span_id=dual_context.span_id,
                    parent_span_id=dual_context.parent_span_id,
                    operation_name=dual_context.operation_name,
                    service_name=dual_context.service_name,
                    start_time=dual_context.created_at,
                    end_time=datetime.now(),
                    status="completed",
                    ai_provider=kwargs.get("provider", "unknown"),
                    ai_model=kwargs.get("model", "unknown"),
                    token_usage=token_usage,
                    cost_info=cost_breakdown
                )
                
                self.tracing_collector.collect_trace_data(tracing_record)
                
            except Exception as e:
                logger.error(
                    "Failed to collect tracing data",
                    extra={
                        "trace_id": trace_id,
                        "error": str(e)
                    }
                )
    
    def _standardize_token_fields(self, token_usage: Dict[str, Any]) -> Dict[str, Any]:
        """标准化token字段名称
        
        将各种token字段名称统一为标准格式：
        - input_tokens -> prompt_tokens
        - output_tokens -> completion_tokens
        
        Args:
            token_usage: 原始token使用数据
            
        Returns:
            Dict: 标准化后的token数据
        """
        standardized = {}
        
        # 映射字段名称
        field_mapping = {
            "input_tokens": "prompt_tokens",
            "output_tokens": "completion_tokens",
            "prompt_tokens": "prompt_tokens",
            "completion_tokens": "completion_tokens",
            "total_tokens": "total_tokens"
        }
        
        for original_key, value in token_usage.items():
            standard_key = field_mapping.get(original_key, original_key)
            standardized[standard_key] = value
        
        # 确保total_tokens存在
        if "total_tokens" not in standardized:
            prompt_tokens = standardized.get("prompt_tokens", 0)
            completion_tokens = standardized.get("completion_tokens", 0)
            standardized["total_tokens"] = prompt_tokens + completion_tokens
        
        return standardized
    
    def _ensure_tracing_table_exists(self):
        """确保追踪信息表存在"""
        if not self.enable_tracing or not self._connection:
            return
        
        try:
            with self._connection.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tracing_info (
                        id SERIAL PRIMARY KEY,
                        hb_trace_id VARCHAR(64) NOT NULL,
                        otel_trace_id VARCHAR(32),
                        span_id VARCHAR(16),
                        parent_span_id VARCHAR(16),
                        operation_name VARCHAR(255) NOT NULL,
                        service_name VARCHAR(255) NOT NULL DEFAULT 'harborai-logging',
                        start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        end_time TIMESTAMP WITH TIME ZONE,
                        duration_ms INTEGER,
                        status VARCHAR(50) DEFAULT 'pending',
                        ai_provider VARCHAR(100),
                        ai_model VARCHAR(255),
                        token_usage JSONB,
                        cost_info JSONB,
                        tags JSONB DEFAULT '{}',
                        logs JSONB DEFAULT '[]',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_tracing_hb_trace_id ON tracing_info(hb_trace_id);
                    CREATE INDEX IF NOT EXISTS idx_tracing_otel_trace_id ON tracing_info(otel_trace_id);
                    CREATE INDEX IF NOT EXISTS idx_tracing_operation ON tracing_info(operation_name);
                    CREATE INDEX IF NOT EXISTS idx_tracing_service ON tracing_info(service_name);
                    CREATE INDEX IF NOT EXISTS idx_tracing_start_time ON tracing_info(start_time);
                    CREATE INDEX IF NOT EXISTS idx_tracing_provider_model ON tracing_info(ai_provider, ai_model);
                """)
                self._connection.commit()
                
        except Exception as e:
            logger.error(
                "Failed to create tracing table",
                extra={
                    "error": str(e),
                    "table_name": "tracing_info"
                }
            )
            raise StorageError(f"Failed to create tracing table: {e}")
    
    def start(self):
        """启动优化的日志记录器"""
        super().start()
        
        # 确保追踪表存在
        if self.enable_tracing:
            self._ensure_tracing_table_exists()
            
            # 启动追踪数据收集器
            if self.tracing_collector:
                self.tracing_collector.start()
        
        logger.info(
            "Optimized PostgreSQL logger started",
            extra={
                "tracing_enabled": self.enable_tracing,
                "sample_rate": self.tracing_sample_rate
            }
        )
    
    def stop(self):
        """停止优化的日志记录器"""
        # 停止追踪数据收集器
        if self.enable_tracing and self.tracing_collector:
            self.tracing_collector.stop()
        
        super().stop()
        logger.info("Optimized PostgreSQL logger stopped")


# 全局实例管理
_global_optimized_logger: Optional[OptimizedPostgreSQLLogger] = None


def get_optimized_postgres_logger() -> Optional[OptimizedPostgreSQLLogger]:
    """获取全局优化PostgreSQL日志记录器实例"""
    return _global_optimized_logger


def initialize_optimized_postgres_logger(connection_string: str, **kwargs) -> OptimizedPostgreSQLLogger:
    """初始化全局优化PostgreSQL日志记录器
    
    Args:
        connection_string: PostgreSQL连接字符串
        **kwargs: 其他初始化参数
        
    Returns:
        OptimizedPostgreSQLLogger: 日志记录器实例
    """
    global _global_optimized_logger
    
    _global_optimized_logger = OptimizedPostgreSQLLogger(connection_string, **kwargs)
    _global_optimized_logger.start()
    
    return _global_optimized_logger


def shutdown_optimized_postgres_logger():
    """关闭全局优化PostgreSQL日志记录器"""
    global _global_optimized_logger
    
    if _global_optimized_logger:
        _global_optimized_logger.stop()
        _global_optimized_logger = None