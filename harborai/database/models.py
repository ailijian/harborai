"""
HarborAI 数据库模型定义
包含API日志、追踪日志、Token使用量、成本信息和追踪信息的数据结构
根据重构设计方案更新，支持增强的数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from decimal import Decimal
import uuid


@dataclass
class APILog:
    """API调用日志数据模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    provider: str = ""
    model: str = ""
    request_data: str = ""
    response_data: str = ""
    status_code: int = 0
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'provider': self.provider,
            'model': self.model,
            'request_data': self.request_data,
            'response_data': self.response_data,
            'status_code': self.status_code,
            'error_message': self.error_message,
            'duration_ms': self.duration_ms
        }


@dataclass
class TraceLog:
    """追踪日志数据模型（保持向后兼容）"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'trace_id': self.trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'duration_ms': self.duration_ms,
            'tags': self.tags,
            'logs': self.logs,
            'status': self.status
        }


@dataclass
class TokenUsageModel:
    """Token使用量数据模型（数据库存储）"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    log_id: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    parsing_method: str = "direct_extraction"
    confidence: float = 1.0
    raw_usage_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """数据一致性验证"""
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            # 自动修正total_tokens
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        
        # 确保confidence在有效范围内
        if not (0.0 <= self.confidence <= 1.0):
            self.confidence = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'log_id': self.log_id,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'parsing_method': self.parsing_method,
            'confidence': self.confidence,
            'raw_usage_data': self.raw_usage_data,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class CostInfoModel:
    """成本信息数据模型（数据库存储）"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    log_id: str = ""
    input_cost: Decimal = field(default_factory=lambda: Decimal('0.0'))
    output_cost: Decimal = field(default_factory=lambda: Decimal('0.0'))
    total_cost: Decimal = field(default_factory=lambda: Decimal('0.0'))
    currency: str = "CNY"
    pricing_source: str = "environment_variable"
    pricing_timestamp: datetime = field(default_factory=datetime.now)
    pricing_details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """数据一致性验证"""
        if self.total_cost != self.input_cost + self.output_cost:
            # 自动修正total_cost
            self.total_cost = self.input_cost + self.output_cost
        
        # 确保成本为非负数
        if self.input_cost < 0:
            self.input_cost = Decimal('0.0')
        if self.output_cost < 0:
            self.output_cost = Decimal('0.0')
        if self.total_cost < 0:
            self.total_cost = Decimal('0.0')
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'log_id': self.log_id,
            'input_cost': float(self.input_cost),
            'output_cost': float(self.output_cost),
            'total_cost': float(self.total_cost),
            'currency': self.currency,
            'pricing_source': self.pricing_source,
            'pricing_timestamp': self.pricing_timestamp.isoformat(),
            'pricing_details': self.pricing_details,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class TracingInfoModel:
    """分布式追踪信息数据模型（数据库存储）"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    log_id: str = ""
    hb_trace_id: str = ""  # HarborAI内部trace_id
    otel_trace_id: str = ""  # OpenTelemetry trace_id
    span_id: str = ""
    parent_span_id: Optional[str] = None
    operation_name: str = "ai.chat.completion"
    start_time: datetime = field(default_factory=datetime.now)
    duration_ms: int = 0
    status: str = "ok"
    trace_flags: str = "01"
    trace_state: str = ""
    api_tags: Dict[str, Any] = field(default_factory=dict)  # 精简版标签
    internal_tags: Dict[str, Any] = field(default_factory=dict)  # 完整版标签
    logs: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """数据验证"""
        # 确保duration_ms为非负数
        if self.duration_ms < 0:
            self.duration_ms = 0
        
        # 验证状态值
        valid_statuses = {'ok', 'error', 'timeout', 'cancelled'}
        if self.status not in valid_statuses:
            self.status = 'ok'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'log_id': self.log_id,
            'hb_trace_id': self.hb_trace_id,
            'otel_trace_id': self.otel_trace_id,
            'span_id': self.span_id,
            'parent_span_id': self.parent_span_id,
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'duration_ms': self.duration_ms,
            'status': self.status,
            'trace_flags': self.trace_flags,
            'trace_state': self.trace_state,
            'api_tags': self.api_tags,
            'internal_tags': self.internal_tags,
            'logs': self.logs,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }


@dataclass
class ModelUsage:
    """模型使用统计数据模型（保持向后兼容）"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    currency: str = "CNY"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'provider': self.provider,
            'model': self.model,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'total_tokens': self.total_tokens,
            'cost': self.cost,
            'currency': self.currency
        }


@dataclass
class LogSummary:
    """日志摘要数据模型（视图查询结果）"""
    log_id: str
    timestamp: datetime
    provider: str
    model: str
    status_code: int
    duration_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    parsing_method: Optional[str] = None
    token_confidence: Optional[float] = None
    input_cost: Optional[Decimal] = None
    output_cost: Optional[Decimal] = None
    total_cost: Optional[Decimal] = None
    currency: Optional[str] = None
    pricing_source: Optional[str] = None
    hb_trace_id: Optional[str] = None
    otel_trace_id: Optional[str] = None
    operation_name: Optional[str] = None
    trace_status: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'log_id': self.log_id,
            'timestamp': self.timestamp.isoformat(),
            'provider': self.provider,
            'model': self.model,
            'status_code': self.status_code,
            'duration_ms': self.duration_ms,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'parsing_method': self.parsing_method,
            'token_confidence': self.token_confidence,
            'input_cost': float(self.input_cost) if self.input_cost else None,
            'output_cost': float(self.output_cost) if self.output_cost else None,
            'total_cost': float(self.total_cost) if self.total_cost else None,
            'currency': self.currency,
            'pricing_source': self.pricing_source,
            'hb_trace_id': self.hb_trace_id,
            'otel_trace_id': self.otel_trace_id,
            'operation_name': self.operation_name,
            'trace_status': self.trace_status
        }