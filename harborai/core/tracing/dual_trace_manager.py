#!/usr/bin/env python3
"""
双Trace ID管理器模块

负责管理HarborAI和OpenTelemetry双追踪ID系统，包括：
- 双Trace ID生成和关联
- 追踪上下文传播
- ID映射和查找
- 跨服务追踪链路管理
- 追踪ID验证和格式化

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import time
import uuid
import hashlib
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timezone
from dataclasses import dataclass
from contextlib import contextmanager

import structlog
from opentelemetry import trace
from opentelemetry.trace import SpanContext
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


@dataclass
class DualTraceContext:
    """双追踪上下文"""
    hb_trace_id: str
    otel_trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    
    # 关联信息
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    
    # 元数据
    created_at: datetime = None
    service_name: str = "harborai-logging"
    operation_name: str = "ai.chat.completion"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class DualTraceIDManager:
    """
    双Trace ID管理器
    
    功能：
    1. 生成和管理HarborAI内部追踪ID
    2. 关联OpenTelemetry追踪ID
    3. 提供追踪上下文传播
    4. 支持跨服务追踪链路
    5. 追踪ID验证和格式化
    """
    
    def __init__(self, hb_prefix: str = "hb", service_name: str = "harborai-logging"):
        """
        初始化双Trace ID管理器
        
        参数:
            hb_prefix: HarborAI追踪ID前缀
            service_name: 服务名称
        """
        self.hb_prefix = hb_prefix
        self.service_name = service_name
        self.logger = structlog.get_logger(__name__)
        self.propagator = TraceContextTextMapPropagator()
        
        # 内存映射缓存
        self._trace_mapping: Dict[str, DualTraceContext] = {}
        self._otel_to_hb_mapping: Dict[str, str] = {}
        
        # 配置
        self.max_cache_size = 10000
        self.cache_ttl_seconds = 3600  # 1小时
    
    def generate_hb_trace_id(
        self,
        operation_name: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        生成HarborAI追踪ID
        
        参数:
            operation_name: 操作名称
            correlation_id: 关联ID
            
        返回:
            str: HarborAI追踪ID
        """
        try:
            # 时间戳（毫秒）
            timestamp = int(time.time() * 1000)
            
            # 随机部分
            random_part = uuid.uuid4().hex[:8]
            
            # 服务标识
            service_hash = hashlib.md5(self.service_name.encode()).hexdigest()[:4]
            
            # 操作标识（可选）
            operation_hash = ""
            if operation_name:
                operation_hash = hashlib.md5(operation_name.encode()).hexdigest()[:4]
            
            # 构建追踪ID
            if operation_hash:
                hb_trace_id = f"{self.hb_prefix}_{timestamp}_{service_hash}_{operation_hash}_{random_part}"
            else:
                hb_trace_id = f"{self.hb_prefix}_{timestamp}_{service_hash}_{random_part}"
            
            self.logger.debug(
                "HarborAI追踪ID生成成功",
                hb_trace_id=hb_trace_id,
                operation_name=operation_name,
                correlation_id=correlation_id
            )
            
            return hb_trace_id
            
        except Exception as e:
            self.logger.error(
                "生成HarborAI追踪ID失败",
                error=str(e),
                operation_name=operation_name
            )
            # 回退到简单格式
            return f"{self.hb_prefix}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    def extract_otel_trace_id(self, span: Optional[trace.Span] = None) -> str:
        """
        提取OpenTelemetry追踪ID
        
        参数:
            span: OpenTelemetry Span，如果为None则使用当前span
            
        返回:
            str: OpenTelemetry追踪ID
        """
        try:
            if span is None:
                span = trace.get_current_span()
            
            if span and span.get_span_context():
                trace_id = span.get_span_context().trace_id
                return f"{trace_id:032x}"
            
            return ""
            
        except Exception as e:
            self.logger.debug(
                "提取OpenTelemetry追踪ID失败",
                error=str(e)
            )
            return ""
    
    def extract_span_id(self, span: Optional[trace.Span] = None) -> str:
        """
        提取Span ID
        
        参数:
            span: OpenTelemetry Span，如果为None则使用当前span
            
        返回:
            str: Span ID
        """
        try:
            if span is None:
                span = trace.get_current_span()
            
            if span and span.get_span_context():
                span_id = span.get_span_context().span_id
                return f"{span_id:016x}"
            
            return ""
            
        except Exception as e:
            self.logger.debug(
                "提取Span ID失败",
                error=str(e)
            )
            return ""
    
    def create_dual_trace_context(
        self,
        operation_name: str = "ai.chat.completion",
        correlation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_span: Optional[trace.Span] = None
    ) -> DualTraceContext:
        """
        创建双追踪上下文
        
        参数:
            operation_name: 操作名称
            correlation_id: 关联ID
            session_id: 会话ID
            user_id: 用户ID
            parent_span: 父span
            
        返回:
            DualTraceContext: 双追踪上下文
        """
        try:
            # 生成HarborAI追踪ID
            hb_trace_id = self.generate_hb_trace_id(operation_name, correlation_id)
            
            # 提取OpenTelemetry追踪信息
            current_span = trace.get_current_span()
            otel_trace_id = self.extract_otel_trace_id(current_span)
            span_id = self.extract_span_id(current_span)
            
            # 获取父span ID
            parent_span_id = None
            if parent_span:
                parent_span_id = self.extract_span_id(parent_span)
            
            # 创建双追踪上下文
            dual_context = DualTraceContext(
                hb_trace_id=hb_trace_id,
                otel_trace_id=otel_trace_id,
                span_id=span_id,
                parent_span_id=parent_span_id,
                correlation_id=correlation_id,
                session_id=session_id,
                user_id=user_id,
                service_name=self.service_name,
                operation_name=operation_name
            )
            
            # 缓存映射关系
            self._cache_trace_mapping(dual_context)
            
            self.logger.debug(
                "双追踪上下文创建成功",
                hb_trace_id=hb_trace_id,
                otel_trace_id=otel_trace_id,
                span_id=span_id,
                operation_name=operation_name
            )
            
            return dual_context
            
        except Exception as e:
            self.logger.error(
                "创建双追踪上下文失败",
                error=str(e),
                operation_name=operation_name
            )
            raise
    
    def _cache_trace_mapping(self, dual_context: DualTraceContext) -> None:
        """缓存追踪映射关系"""
        try:
            # 清理过期缓存
            self._cleanup_expired_cache()
            
            # 添加新映射
            self._trace_mapping[dual_context.hb_trace_id] = dual_context
            
            if dual_context.otel_trace_id:
                self._otel_to_hb_mapping[dual_context.otel_trace_id] = dual_context.hb_trace_id
            
            # 限制缓存大小
            if len(self._trace_mapping) > self.max_cache_size:
                self._evict_oldest_cache_entries()
            
        except Exception as e:
            self.logger.warning(
                "缓存追踪映射失败",
                error=str(e),
                hb_trace_id=dual_context.hb_trace_id
            )
    
    def _cleanup_expired_cache(self) -> None:
        """清理过期缓存"""
        try:
            current_time = datetime.now(timezone.utc)
            expired_keys = []
            
            for hb_trace_id, context in self._trace_mapping.items():
                if context.created_at:
                    age_seconds = (current_time - context.created_at).total_seconds()
                    if age_seconds > self.cache_ttl_seconds:
                        expired_keys.append(hb_trace_id)
            
            # 删除过期条目
            for key in expired_keys:
                context = self._trace_mapping.pop(key, None)
                if context and context.otel_trace_id:
                    self._otel_to_hb_mapping.pop(context.otel_trace_id, None)
            
            if expired_keys:
                self.logger.debug(
                    "清理过期缓存完成",
                    expired_count=len(expired_keys)
                )
                
        except Exception as e:
            self.logger.warning(
                "清理过期缓存失败",
                error=str(e)
            )
    
    def _evict_oldest_cache_entries(self) -> None:
        """驱逐最旧的缓存条目"""
        try:
            # 按创建时间排序，删除最旧的条目
            sorted_items = sorted(
                self._trace_mapping.items(),
                key=lambda x: x[1].created_at or datetime.min.replace(tzinfo=timezone.utc)
            )
            
            # 删除最旧的10%条目
            evict_count = max(1, len(sorted_items) // 10)
            
            for i in range(evict_count):
                hb_trace_id, context = sorted_items[i]
                self._trace_mapping.pop(hb_trace_id, None)
                if context.otel_trace_id:
                    self._otel_to_hb_mapping.pop(context.otel_trace_id, None)
            
            self.logger.debug(
                "驱逐最旧缓存条目完成",
                evicted_count=evict_count
            )
            
        except Exception as e:
            self.logger.warning(
                "驱逐最旧缓存条目失败",
                error=str(e)
            )
    
    def get_dual_context_by_hb_id(self, hb_trace_id: str) -> Optional[DualTraceContext]:
        """
        通过HarborAI追踪ID获取双追踪上下文
        
        参数:
            hb_trace_id: HarborAI追踪ID
            
        返回:
            Optional[DualTraceContext]: 双追踪上下文
        """
        return self._trace_mapping.get(hb_trace_id)
    
    def get_hb_id_by_otel_id(self, otel_trace_id: str) -> Optional[str]:
        """
        通过OpenTelemetry追踪ID获取HarborAI追踪ID
        
        参数:
            otel_trace_id: OpenTelemetry追踪ID
            
        返回:
            Optional[str]: HarborAI追踪ID
        """
        return self._otel_to_hb_mapping.get(otel_trace_id)
    
    def get_dual_context_by_otel_id(self, otel_trace_id: str) -> Optional[DualTraceContext]:
        """
        通过OpenTelemetry追踪ID获取双追踪上下文
        
        参数:
            otel_trace_id: OpenTelemetry追踪ID
            
        返回:
            Optional[DualTraceContext]: 双追踪上下文
        """
        hb_trace_id = self.get_hb_id_by_otel_id(otel_trace_id)
        if hb_trace_id:
            return self.get_dual_context_by_hb_id(hb_trace_id)
        return None
    
    def inject_trace_headers(self, dual_context: DualTraceContext) -> Dict[str, str]:
        """
        注入追踪头部
        
        参数:
            dual_context: 双追踪上下文
            
        返回:
            Dict[str, str]: 包含追踪信息的HTTP头部
        """
        headers = {}
        
        try:
            # 注入OpenTelemetry追踪头部
            current_span = trace.get_current_span()
            if current_span:
                context = trace.set_span_in_context(current_span)
                self.propagator.inject(headers, context)
            
            # 注入HarborAI特定头部
            headers["X-HarborAI-Trace-ID"] = dual_context.hb_trace_id
            headers["X-HarborAI-Span-ID"] = dual_context.span_id
            headers["X-HarborAI-Service"] = dual_context.service_name
            headers["X-HarborAI-Operation"] = dual_context.operation_name
            
            # 注入关联信息
            if dual_context.correlation_id:
                headers["X-HarborAI-Correlation-ID"] = dual_context.correlation_id
            if dual_context.session_id:
                headers["X-HarborAI-Session-ID"] = dual_context.session_id
            if dual_context.user_id:
                headers["X-HarborAI-User-ID"] = dual_context.user_id
            
            self.logger.debug(
                "追踪头部注入成功",
                hb_trace_id=dual_context.hb_trace_id,
                headers_count=len(headers)
            )
            
        except Exception as e:
            self.logger.warning(
                "注入追踪头部失败",
                error=str(e),
                hb_trace_id=dual_context.hb_trace_id
            )
        
        return headers
    
    def extract_trace_headers(self, headers: Dict[str, str]) -> Optional[DualTraceContext]:
        """
        从HTTP头部提取追踪上下文
        
        参数:
            headers: HTTP请求头部
            
        返回:
            Optional[DualTraceContext]: 双追踪上下文
        """
        try:
            # 提取HarborAI追踪信息
            hb_trace_id = headers.get("X-HarborAI-Trace-ID")
            span_id = headers.get("X-HarborAI-Span-ID", "")
            service_name = headers.get("X-HarborAI-Service", self.service_name)
            operation_name = headers.get("X-HarborAI-Operation", "ai.chat.completion")
            
            # 提取关联信息
            correlation_id = headers.get("X-HarborAI-Correlation-ID")
            session_id = headers.get("X-HarborAI-Session-ID")
            user_id = headers.get("X-HarborAI-User-ID")
            
            if not hb_trace_id:
                return None
            
            # 提取OpenTelemetry追踪上下文
            otel_context = self.propagator.extract(headers)
            otel_trace_id = ""
            
            if otel_context:
                span_context = trace.get_current_span(otel_context).get_span_context()
                if span_context:
                    otel_trace_id = f"{span_context.trace_id:032x}"
            
            # 创建双追踪上下文
            dual_context = DualTraceContext(
                hb_trace_id=hb_trace_id,
                otel_trace_id=otel_trace_id,
                span_id=span_id,
                correlation_id=correlation_id,
                session_id=session_id,
                user_id=user_id,
                service_name=service_name,
                operation_name=operation_name
            )
            
            # 缓存映射关系
            self._cache_trace_mapping(dual_context)
            
            self.logger.debug(
                "追踪头部提取成功",
                hb_trace_id=hb_trace_id,
                otel_trace_id=otel_trace_id
            )
            
            return dual_context
            
        except Exception as e:
            self.logger.warning(
                "提取追踪头部失败",
                error=str(e),
                headers=headers
            )
            return None
    
    def validate_hb_trace_id(self, hb_trace_id: str) -> bool:
        """
        验证HarborAI追踪ID格式
        
        参数:
            hb_trace_id: HarborAI追踪ID
            
        返回:
            bool: 是否有效
        """
        try:
            if not hb_trace_id or not isinstance(hb_trace_id, str):
                return False
            
            # 检查前缀
            if not hb_trace_id.startswith(f"{self.hb_prefix}_"):
                return False
            
            # 检查格式：prefix_timestamp_service_[operation_]random
            parts = hb_trace_id.split("_")
            if len(parts) < 4:
                return False
            
            # 验证时间戳部分
            try:
                timestamp = int(parts[1])
                if timestamp <= 0:
                    return False
            except ValueError:
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(
                "验证HarborAI追踪ID失败",
                error=str(e),
                hb_trace_id=hb_trace_id
            )
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        返回:
            Dict[str, Any]: 缓存统计信息
        """
        return {
            "trace_mapping_size": len(self._trace_mapping),
            "otel_to_hb_mapping_size": len(self._otel_to_hb_mapping),
            "max_cache_size": self.max_cache_size,
            "cache_ttl_seconds": self.cache_ttl_seconds
        }
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self._trace_mapping.clear()
        self._otel_to_hb_mapping.clear()
        self.logger.info("双追踪ID缓存已清空")


# 全局双Trace ID管理器实例
_global_dual_trace_manager: Optional[DualTraceIDManager] = None


def get_global_dual_trace_manager() -> Optional[DualTraceIDManager]:
    """获取全局双Trace ID管理器实例"""
    return _global_dual_trace_manager


def setup_global_dual_trace_manager(
    hb_prefix: str = "hb",
    service_name: str = "harborai-logging"
) -> DualTraceIDManager:
    """设置全局双Trace ID管理器实例"""
    global _global_dual_trace_manager
    _global_dual_trace_manager = DualTraceIDManager(hb_prefix, service_name)
    return _global_dual_