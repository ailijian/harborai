# -*- coding: utf-8 -*-
"""
统一时间戳生成工具模块

功能：提供统一的时间戳生成和验证功能，确保所有日志记录使用一致的时间源（北京时间 UTC+8）
作者：HarborAI开发团队
创建时间：2024
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Union
from threading import Lock

from .logger import get_logger

logger = get_logger(__name__)

# 北京时区（UTC+8）
BEIJING_TIMEZONE = timezone(timedelta(hours=8))

# 全局锁，确保时间戳生成的线程安全
_timestamp_lock = Lock()

# 上一次生成的时间戳，用于确保时间戳的单调递增
_last_timestamp: Optional[float] = None


def get_unified_timestamp() -> datetime:
    """
    获取统一的时间戳
    
    使用北京时间（UTC+8）确保一致性，并保证时间戳的单调递增特性
    
    Returns:
        datetime: 北京时间戳对象
    """
    global _last_timestamp
    
    with _timestamp_lock:
        current_time = datetime.now(BEIJING_TIMEZONE)
        current_timestamp = current_time.timestamp()
        
        # 确保时间戳单调递增
        if _last_timestamp is not None and current_timestamp <= _last_timestamp:
            # 如果当前时间戳不大于上一次的时间戳，则在上一次基础上增加1微秒
            current_timestamp = _last_timestamp + 0.000001
            current_time = datetime.fromtimestamp(current_timestamp, BEIJING_TIMEZONE)
            
            logger.warning(
                f"检测到时间戳回退，已调整为单调递增: {current_time.isoformat()}"
            )
        
        _last_timestamp = current_timestamp
        return current_time


def get_unified_timestamp_iso() -> str:
    """
    获取统一的 ISO 格式时间戳字符串
    
    Returns:
        str: ISO 格式的北京时间戳字符串（包含 +08:00 时区信息）
    """
    return get_unified_timestamp().isoformat()


def get_unified_timestamp_float() -> float:
    """
    获取统一的浮点数时间戳
    
    Returns:
        float: 北京时间戳（秒，Unix 时间戳）
    """
    return get_unified_timestamp().timestamp()


def validate_timestamp_order(
    request_timestamp: Union[str, datetime, float],
    response_timestamp: Union[str, datetime, float],
    trace_id: Optional[str] = None
) -> bool:
    """
    验证请求和响应时间戳的逻辑顺序
    
    Args:
        request_timestamp: 请求时间戳
        response_timestamp: 响应时间戳
        trace_id: 追踪ID（用于日志记录）
    
    Returns:
        bool: 如果时间戳顺序正确返回 True，否则返回 False
    """
    try:
        # 统一转换为 datetime 对象
        req_dt = _normalize_timestamp(request_timestamp)
        resp_dt = _normalize_timestamp(response_timestamp)
        
        if req_dt is None or resp_dt is None:
            logger.warning(
                f"无法解析时间戳 [trace_id={trace_id}] "
                f"request={request_timestamp} response={response_timestamp}"
            )
            return False
        
        # 检查时间戳顺序
        if resp_dt < req_dt:
            logger.error(
                f"检测到异常时间戳顺序 [trace_id={trace_id}] "
                f"响应时间 {resp_dt.isoformat()} 早于请求时间 {req_dt.isoformat()}"
            )
            return False
        
        # 检查时间差是否合理（响应时间不应该比请求时间早，也不应该相差太久）
        time_diff = (resp_dt - req_dt).total_seconds()
        if time_diff > 300:  # 5分钟
            logger.warning(
                f"检测到异常长的响应时间 [trace_id={trace_id}] "
                f"时间差: {time_diff:.2f}秒"
            )
        
        return True
        
    except Exception as e:
        logger.error(
            f"时间戳验证失败 [trace_id={trace_id}] error={str(e)}"
        )
        return False


def _normalize_timestamp(timestamp: Union[str, datetime, float]) -> Optional[datetime]:
    """
    将各种格式的时间戳标准化为 datetime 对象
    
    Args:
        timestamp: 时间戳（字符串、datetime对象或浮点数）
    
    Returns:
        Optional[datetime]: 标准化后的 datetime 对象（北京时间），失败时返回 None
    """
    try:
        if isinstance(timestamp, datetime):
            # 确保是北京时间
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=BEIJING_TIMEZONE)
            return timestamp.astimezone(BEIJING_TIMEZONE)
        
        elif isinstance(timestamp, str):
            # 尝试解析 ISO 格式字符串
            try:
                # 处理各种 ISO 格式
                if timestamp.endswith('Z'):
                    timestamp = timestamp[:-1] + '+00:00'
                return datetime.fromisoformat(timestamp).astimezone(BEIJING_TIMEZONE)
            except ValueError:
                # 尝试其他格式
                try:
                    return datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').replace(tzinfo=BEIJING_TIMEZONE)
                except ValueError:
                    return None
        
        elif isinstance(timestamp, (int, float)):
            # 浮点数时间戳
            return datetime.fromtimestamp(timestamp, BEIJING_TIMEZONE)
        
        else:
            return None
            
    except Exception:
        return None


def create_timestamp_context(trace_id: str) -> 'TimestampContext':
    """
    创建时间戳上下文管理器，用于跟踪单个请求的时间戳
    
    Args:
        trace_id: 追踪ID
    
    Returns:
        TimestampContext: 时间戳上下文管理器
    """
    return TimestampContext(trace_id)


class TimestampContext:
    """
    时间戳上下文管理器
    
    用于跟踪单个请求的时间戳，确保请求-响应时间戳的逻辑一致性
    """
    
    def __init__(self, trace_id: str):
        self.trace_id = trace_id
        self.request_timestamp: Optional[datetime] = None
        self.response_timestamp: Optional[datetime] = None
    
    def mark_request(self) -> datetime:
        """
        标记请求时间戳
        
        Returns:
            datetime: 请求时间戳
        """
        self.request_timestamp = get_unified_timestamp()
        logger.debug(f"标记请求时间戳 [trace_id={self.trace_id}] {self.request_timestamp.isoformat()}")
        return self.request_timestamp
    
    def mark_response(self) -> datetime:
        """
        标记响应时间戳并验证时间戳顺序
        
        Returns:
            datetime: 响应时间戳
        """
        self.response_timestamp = get_unified_timestamp()
        logger.debug(f"标记响应时间戳 [trace_id={self.trace_id}] {self.response_timestamp.isoformat()}")
        
        # 验证时间戳顺序
        if self.request_timestamp:
            validate_timestamp_order(
                self.request_timestamp,
                self.response_timestamp,
                self.trace_id
            )
        
        return self.response_timestamp
    
    def get_duration_ms(self) -> Optional[float]:
        """
        获取请求-响应的持续时间（毫秒）
        
        Returns:
            Optional[float]: 持续时间（毫秒），如果时间戳不完整则返回 None
        """
        if self.request_timestamp and self.response_timestamp:
            duration = (self.response_timestamp - self.request_timestamp).total_seconds() * 1000
            return max(0, duration)  # 确保不返回负数
        return None