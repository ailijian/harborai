#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token使用量数据模型

定义Token使用量的数据结构，保持与厂商响应字段名的一致性。
根据HarborAI日志系统重构设计方案实现。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class TokenUsage:
    """Token使用量数据模型 - 保持厂商原始字段名
    
    根据重构设计方案，保持prompt_tokens和completion_tokens字段名
    与厂商响应完全一致，确保数据的准确性和一致性。
    """
    prompt_tokens: int      # 与厂商响应字段名保持一致
    completion_tokens: int  # 与厂商响应字段名保持一致
    total_tokens: int
    parsing_method: str = "direct_extraction"
    confidence: float = 1.0
    raw_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """数据一致性自动修正策略
        
        根据重构设计方案，实现自动数据一致性验证和修正：
        1. 验证total_tokens = prompt_tokens + completion_tokens
        2. 如果不一致，优先信任厂商提供的total_tokens
        3. 降低置信度以标识数据质量问题
        """
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
            
        # 数据一致性验证和自动修正
        expected_total = self.prompt_tokens + self.completion_tokens
        
        if self.total_tokens != expected_total:
            logger.warning(
                "Token数据不一致，执行自动修正",
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                reported_total=self.total_tokens,
                expected_total=expected_total,
                parsing_method=self.parsing_method
            )
            
            # 自动修正策略
            if self.total_tokens > 0 and expected_total == 0:
                # 如果只有total_tokens有值，保持不变
                logger.info("保持厂商提供的total_tokens值", total_tokens=self.total_tokens)
            else:
                # 否则使用计算值
                self.total_tokens = expected_total
                self.confidence = 0.8  # 降低置信度
                logger.info("使用计算值修正total_tokens", corrected_total=self.total_tokens)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            包含所有字段的字典
        """
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "parsing_method": self.parsing_method,
            "confidence": self.confidence,
            "raw_data": self.raw_data,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenUsage':
        """从字典创建TokenUsage实例
        
        Args:
            data: 包含token使用量数据的字典
            
        Returns:
            TokenUsage实例
        """
        timestamp = None
        if data.get("timestamp"):
            if isinstance(data["timestamp"], str):
                timestamp = datetime.fromisoformat(data["timestamp"])
            elif isinstance(data["timestamp"], datetime):
                timestamp = data["timestamp"]
        
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            parsing_method=data.get("parsing_method", "direct_extraction"),
            confidence=data.get("confidence", 1.0),
            raw_data=data.get("raw_data", {}),
            timestamp=timestamp
        )
    
    def validate_consistency(self) -> bool:
        """验证数据一致性
        
        Returns:
            True if consistent, False otherwise
        """
        expected_total = self.prompt_tokens + self.completion_tokens
        return self.total_tokens == expected_total
    
    def get_quality_score(self) -> float:
        """获取数据质量评分
        
        基于置信度和数据一致性计算质量评分
        
        Returns:
            0.0-1.0之间的质量评分
        """
        consistency_score = 1.0 if self.validate_consistency() else 0.7
        return self.confidence * consistency_score
    
    def is_valid(self) -> bool:
        """检查数据是否有效
        
        Returns:
            True if valid, False otherwise
        """
        return (
            self.prompt_tokens >= 0 and
            self.completion_tokens >= 0 and
            self.total_tokens >= 0 and
            0.0 <= self.confidence <= 1.0
        )

@dataclass
class TokenUsageStats:
    """Token使用量统计数据模型
    
    用于聚合和统计分析Token使用情况
    """
    total_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_total_tokens: float = 0.0
    date_range_start: Optional[datetime] = None
    date_range_end: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "total_requests": self.total_requests,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "avg_prompt_tokens": self.avg_prompt_tokens,
            "avg_completion_tokens": self.avg_completion_tokens,
            "avg_total_tokens": self.avg_total_tokens,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None
        }