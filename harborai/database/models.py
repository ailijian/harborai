#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库模型定义

定义API日志、跟踪日志和模型使用统计的数据模型。
"""

from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
import json

@dataclass
class APILog:
    """API调用日志模型"""
    id: Optional[int] = None
    timestamp: Optional[datetime] = None
    provider: str = ""
    model: str = ""
    request_data: Optional[str] = None
    response_data: Optional[str] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "provider": self.provider,
            "model": self.model,
            "request_data": json.loads(self.request_data) if self.request_data else None,
            "response_data": json.loads(self.response_data) if self.response_data else None,
            "status_code": self.status_code,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms
        }
    
    @classmethod
    def from_row(cls, row) -> 'APILog':
        """从数据库行创建实例"""
        return cls(
            id=row['id'],
            timestamp=datetime.fromisoformat(row['timestamp']) if row['timestamp'] else None,
            provider=row['provider'],
            model=row['model'],
            request_data=row['request_data'],
            response_data=row['response_data'],
            status_code=row['status_code'],
            error_message=row['error_message'],
            duration_ms=row['duration_ms']
        )

@dataclass
class TraceLog:
    """跟踪日志模型"""
    id: Optional[int] = None
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: Optional[str] = None
    operation_name: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Optional[str] = None
    logs: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "tags": json.loads(self.tags) if self.tags else None,
            "logs": json.loads(self.logs) if self.logs else None
        }
    
    @classmethod
    def from_row(cls, row) -> 'TraceLog':
        """从数据库行创建实例"""
        return cls(
            id=row['id'],
            trace_id=row['trace_id'],
            span_id=row['span_id'],
            parent_span_id=row['parent_span_id'],
            operation_name=row['operation_name'],
            start_time=datetime.fromisoformat(row['start_time']) if row['start_time'] else None,
            end_time=datetime.fromisoformat(row['end_time']) if row['end_time'] else None,
            duration_ms=row['duration_ms'],
            tags=row['tags'],
            logs=row['logs']
        )

@dataclass
class ModelUsage:
    """模型使用统计模型"""
    id: Optional[int] = None
    date: Optional[datetime] = None
    provider: str = ""
    model: str = ""
    request_count: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "date": self.date.date().isoformat() if self.date else None,
            "provider": self.provider,
            "model": self.model,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_cost": self.total_cost
        }
    
    @classmethod
    def from_row(cls, row) -> 'ModelUsage':
        """从数据库行创建实例"""
        return cls(
            id=row['id'],
            date=datetime.fromisoformat(row['date']) if row['date'] else None,
            provider=row['provider'],
            model=row['model'],
            request_count=row['request_count'],
            total_tokens=row['total_tokens'],
            input_tokens=row['input_tokens'],
            output_tokens=row['output_tokens'],
            total_cost=row['total_cost']
        )