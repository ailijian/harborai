#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token统计收集器模块

提供Token使用量的统计和分析功能
"""

import asyncio
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Deque
from concurrent.futures import ThreadPoolExecutor

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TokenUsageRecord:
    """Token使用记录"""
    timestamp: datetime
    model_name: str
    provider: str
    request_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    latency_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class ModelStatistics:
    """模型统计信息"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_latency: float = 0.0
    last_used: Optional[datetime] = None
    
    @property
    def average_latency(self) -> float:
        """平均延迟"""
        return self.total_latency / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        return self.failed_requests / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def average_tokens_per_request(self) -> float:
        """每次请求平均Token数"""
        return self.total_tokens / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def cost_per_token(self) -> float:
        """每Token成本"""
        return self.total_cost / self.total_tokens if self.total_tokens > 0 else 0.0


@dataclass
class TimeWindowStatistics:
    """时间窗口统计信息"""
    window_start: datetime
    window_end: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    unique_models: int = 0
    average_latency: float = 0.0


class TokenStatisticsCollector:
    """Token统计收集器"""
    
    def __init__(self, max_records: int = 10000, cleanup_interval: int = 3600):
        """
        初始化统计收集器
        
        Args:
            max_records: 最大记录数
            cleanup_interval: 清理间隔（秒）
        """
        self.max_records = max_records
        self.cleanup_interval = cleanup_interval
        
        # 存储原始记录
        self._records: Deque[TokenUsageRecord] = deque(maxlen=max_records)
        
        # 模型统计缓存
        self._model_stats: Dict[str, ModelStatistics] = {}
        
        # 时间窗口缓存
        self._hourly_stats: Dict[datetime, TimeWindowStatistics] = {}
        self._daily_stats: Dict[datetime, TimeWindowStatistics] = {}
        
        # 线程安全锁
        self._lock = threading.RLock()
        
        # 后台清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="token_stats")
        
        # 启动后台任务
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """启动后台任务"""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                self._cleanup_task = loop.create_task(self._periodic_cleanup())
        except RuntimeError:
            # 如果没有事件循环，稍后再启动
            logger.debug("No event loop available, background tasks will start later")
    
    async def _periodic_cleanup(self):
        """定期清理过期数据"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_old_records()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    def record_usage(self, 
                    trace_id: Optional[str] = None,
                    model: Optional[str] = None,
                    input_tokens: Optional[int] = None,
                    output_tokens: Optional[int] = None,
                    duration: Optional[float] = None,
                    success: Optional[bool] = None,
                    error: Optional[str] = None,
                    provider: str = "unknown",
                    cost: Optional[float] = None,
                    record: Optional[TokenUsageRecord] = None):
        """
        记录Token使用
        
        Args:
            trace_id: 追踪ID
            model: 模型名称
            input_tokens: 输入Token数
            output_tokens: 输出Token数
            duration: 持续时间（秒）
            success: 是否成功
            error: 错误信息
            provider: 服务提供商
            cost: 成本
            record: Token使用记录对象（如果提供，将忽略其他参数）
        """
        if record is not None:
            # 使用提供的记录对象
            usage_record = record
        else:
            # 从参数创建记录对象
            if any(param is None for param in [trace_id, model, input_tokens, output_tokens, duration, success]):
                raise ValueError("Missing required parameters for token usage recording")
            
            # 计算成本（如果未提供）
            if cost is None:
                try:
                    from ..core.cost_tracking import PricingCalculator
                    calculator = PricingCalculator()
                    cost = calculator.calculate_cost(input_tokens, output_tokens, model)
                except Exception as e:
                    logger.warning(f"Failed to calculate cost: {e}")
                    cost = 0.0
            
            usage_record = TokenUsageRecord(
                timestamp=datetime.now(),
                model_name=model,
                provider=provider,
                request_id=trace_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost=cost,
                latency_ms=duration * 1000,  # 转换为毫秒
                success=success,
                error_message=error
            )
        
        with self._lock:
            # 添加到记录队列
            self._records.append(usage_record)
            
            # 更新模型统计
            self._update_model_statistics(usage_record)
            
            # 更新时间窗口统计
            self._update_time_window_statistics(usage_record)
    
    def _update_model_statistics(self, record: TokenUsageRecord):
        """更新模型统计信息"""
        model_name = record.model_name
        
        if model_name not in self._model_stats:
            self._model_stats[model_name] = ModelStatistics(model_name=model_name)
        
        stats = self._model_stats[model_name]
        stats.total_requests += 1
        
        if record.success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
        
        stats.total_input_tokens += record.input_tokens
        stats.total_output_tokens += record.output_tokens
        stats.total_tokens += record.total_tokens
        stats.total_cost += record.cost
        stats.total_latency += record.latency_ms
        stats.last_used = record.timestamp
    
    def _update_time_window_statistics(self, record: TokenUsageRecord):
        """更新时间窗口统计"""
        # 小时级统计
        hour_key = record.timestamp.replace(minute=0, second=0, microsecond=0)
        if hour_key not in self._hourly_stats:
            self._hourly_stats[hour_key] = TimeWindowStatistics(
                window_start=hour_key,
                window_end=hour_key + timedelta(hours=1)
            )
        
        hourly_stat = self._hourly_stats[hour_key]
        self._update_window_stat(hourly_stat, record)
        
        # 日级统计
        day_key = record.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        if day_key not in self._daily_stats:
            self._daily_stats[day_key] = TimeWindowStatistics(
                window_start=day_key,
                window_end=day_key + timedelta(days=1)
            )
        
        daily_stat = self._daily_stats[day_key]
        self._update_window_stat(daily_stat, record)
    
    def _update_window_stat(self, stat: TimeWindowStatistics, record: TokenUsageRecord):
        """更新窗口统计"""
        stat.total_requests += 1
        
        if record.success:
            stat.successful_requests += 1
        else:
            stat.failed_requests += 1
        
        stat.total_tokens += record.total_tokens
        stat.total_cost += record.cost
        
        # 更新平均延迟
        if stat.total_requests > 1:
            stat.average_latency = (stat.average_latency * (stat.total_requests - 1) + record.latency_ms) / stat.total_requests
        else:
            stat.average_latency = record.latency_ms
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        获取汇总统计信息
        
        Returns:
            汇总统计信息字典
        """
        with self._lock:
            total_records = len(self._records)
            
            if total_records == 0:
                return {
                    "total_records": 0,
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "unique_models": 0,
                    "success_rate": 0.0,
                    "average_latency": 0.0,
                    "time_range": None
                }
            
            # 计算汇总数据
            total_requests = sum(stats.total_requests for stats in self._model_stats.values())
            successful_requests = sum(stats.successful_requests for stats in self._model_stats.values())
            total_tokens = sum(stats.total_tokens for stats in self._model_stats.values())
            total_cost = sum(stats.total_cost for stats in self._model_stats.values())
            total_latency = sum(stats.total_latency for stats in self._model_stats.values())
            
            # 时间范围
            oldest_record = self._records[0]
            newest_record = self._records[-1]
            
            return {
                "total_records": total_records,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": total_requests - successful_requests,
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 6),
                "unique_models": len(self._model_stats),
                "success_rate": successful_requests / total_requests if total_requests > 0 else 0.0,
                "average_latency": total_latency / total_requests if total_requests > 0 else 0.0,
                "time_range": {
                    "start": oldest_record.timestamp.isoformat(),
                    "end": newest_record.timestamp.isoformat()
                }
            }
    
    def get_model_statistics(self, model_name: Optional[str] = None) -> Dict[str, ModelStatistics]:
        """
        获取模型统计信息
        
        Args:
            model_name: 可选的模型名称，如果提供则只返回该模型的统计
            
        Returns:
            模型统计信息字典
        """
        with self._lock:
            if model_name:
                return {model_name: self._model_stats.get(model_name)} if model_name in self._model_stats else {}
            return self._model_stats.copy()
    
    def get_time_window_stats(self, window_type: str = "hour", count: int = 24) -> List[TimeWindowStatistics]:
        """
        获取时间窗口统计
        
        Args:
            window_type: 窗口类型（"hour" 或 "day"）
            count: 返回的窗口数量
            
        Returns:
            时间窗口统计列表
        """
        with self._lock:
            if window_type == "hour":
                stats_dict = self._hourly_stats
            elif window_type == "day":
                stats_dict = self._daily_stats
            else:
                raise ValueError(f"Unsupported window_type: {window_type}")
            
            # 获取最近的窗口
            sorted_windows = sorted(stats_dict.keys(), reverse=True)
            recent_windows = sorted_windows[:count]
            
            return [stats_dict[window] for window in reversed(recent_windows)]
    
    def get_recent_records(self, count: int = 100) -> List[TokenUsageRecord]:
        """
        获取最近的记录
        
        Args:
            count: 返回的记录数量
            
        Returns:
            最近的记录列表
        """
        with self._lock:
            records_list = list(self._records)
            return records_list[-count:] if len(records_list) > count else records_list
    
    async def cleanup_old_records(self, max_age_hours: int = 168):  # 7天
        """
        清理过期记录
        
        Args:
            max_age_hours: 最大保留时间（小时）
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        def _cleanup():
            with self._lock:
                # 清理时间窗口统计
                expired_hours = [k for k in self._hourly_stats.keys() if k < cutoff_time]
                for key in expired_hours:
                    del self._hourly_stats[key]
                
                expired_days = [k for k in self._daily_stats.keys() if k < cutoff_time]
                for key in expired_days:
                    del self._daily_stats[key]
                
                logger.info(f"Cleaned up {len(expired_hours)} hourly and {len(expired_days)} daily statistics")
        
        # 在线程池中执行清理
        await asyncio.get_event_loop().run_in_executor(self._executor, _cleanup)
    
    def export_statistics(self, format_type: str = "json") -> str:
        """
        导出统计信息
        
        Args:
            format_type: 导出格式（"json" 或 "csv"）
            
        Returns:
            导出的数据字符串
        """
        with self._lock:
            if format_type == "json":
                import json
                data = {
                    "summary": self.get_summary_stats(),
                    "models": {name: {
                        "model_name": stats.model_name,
                        "total_requests": stats.total_requests,
                        "successful_requests": stats.successful_requests,
                        "failed_requests": stats.failed_requests,
                        "total_tokens": stats.total_tokens,
                        "total_cost": stats.total_cost,
                        "average_latency": stats.average_latency,
                        "error_rate": stats.error_rate,
                        "last_used": stats.last_used.isoformat() if stats.last_used else None
                    } for name, stats in self._model_stats.items()},
                    "export_time": datetime.now().isoformat()
                }
                return json.dumps(data, indent=2, ensure_ascii=False)
            
            elif format_type == "csv":
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # 写入标题
                writer.writerow([
                    "model_name", "total_requests", "successful_requests", "failed_requests",
                    "total_tokens", "total_cost", "average_latency", "error_rate", "last_used"
                ])
                
                # 写入数据
                for stats in self._model_stats.values():
                    writer.writerow([
                        stats.model_name,
                        stats.total_requests,
                        stats.successful_requests,
                        stats.failed_requests,
                        stats.total_tokens,
                        round(stats.total_cost, 6),
                        round(stats.average_latency, 3),
                        round(stats.error_rate, 4),
                        stats.last_used.isoformat() if stats.last_used else ""
                    ])
                
                return output.getvalue()
            
            else:
                raise ValueError(f"Unsupported format_type: {format_type}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        with self._lock:
            return {
                "status": "healthy",
                "total_records": len(self._records),
                "total_models": len(self._model_stats),
                "hourly_windows": len(self._hourly_stats),
                "daily_windows": len(self._daily_stats),
                "memory_usage": {
                    "records_count": len(self._records),
                    "max_records": self.max_records,
                    "usage_percentage": len(self._records) / self.max_records * 100
                },
                "last_record_time": self._records[-1].timestamp.isoformat() if self._records else None,
                "cleanup_task_running": self._cleanup_task is not None and not self._cleanup_task.done()
            }
    
    def __del__(self):
        """清理资源"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        
        if self._executor:
            self._executor.shutdown(wait=False)


# 全局实例
_token_statistics_collector: Optional[TokenStatisticsCollector] = None


def get_token_statistics_collector() -> TokenStatisticsCollector:
    """
    获取Token统计收集器实例（单例模式）
    
    Returns:
        Token统计收集器实例
    """
    global _token_statistics_collector
    
    if _token_statistics_collector is None:
        _token_statistics_collector = TokenStatisticsCollector()
    
    return _token_statistics_collector


def record_token_usage(
    trace_id: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration: float,
    success: bool,
    error: Optional[str] = None,
    provider: str = "unknown",
    cost: Optional[float] = None
):
    """
    记录Token使用情况
    
    Args:
        trace_id: 追踪ID
        model: 模型名称
        input_tokens: 输入Token数
        output_tokens: 输出Token数
        duration: 持续时间（秒）
        success: 是否成功
        error: 错误信息（如果有）
        provider: 服务提供商
        cost: 成本（如果有）
    """
    try:
        collector = get_token_statistics_collector()
        
        record = TokenUsageRecord(
            timestamp=datetime.now(),
            model_name=model,
            provider=provider,
            request_id=trace_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost or 0.0,
            latency_ms=duration * 1000,  # 转换为毫秒
            success=success,
            error_message=error
        )
        
        collector.record_usage(record)
        
    except Exception as e:
        logger.error(f"Failed to record token usage: {e}")


def reset_token_statistics_collector():
    """重置Token统计收集器（主要用于测试）"""
    global _token_statistics_collector
    
    if _token_statistics_collector:
        if hasattr(_token_statistics_collector, '_cleanup_task') and _token_statistics_collector._cleanup_task:
            _token_statistics_collector._cleanup_task.cancel()
    
    _token_statistics_collector = None