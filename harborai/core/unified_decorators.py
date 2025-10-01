"""统一装饰器系统

合并所有装饰器功能，提供高性能的统一装饰器接口。
"""

import asyncio
import functools
import inspect
import logging
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Union, TypeVar, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager, contextmanager

from ..config.settings import get_settings
from ..config.performance import get_performance_config, PerformanceMode
from .async_cost_tracking import get_async_cost_tracker
from .background_tasks import submit_background_task
from .cache_manager import get_cache_manager
from ..storage.postgres_logger import PostgreSQLLogger as PostgresLogger
from ..utils.tracer import TraceContext, SpanTimer, get_or_create_trace_id

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class DecoratorMode(Enum):
    """装饰器模式"""
    FAST = "fast"  # 快速模式，最小开销
    FULL = "full"  # 完整模式，所有功能
    CUSTOM = "custom"  # 自定义模式


@dataclass
class DecoratorConfig:
    """装饰器配置"""
    mode: DecoratorMode = DecoratorMode.FULL
    enable_tracing: bool = True
    enable_cost_tracking: bool = True
    enable_postgres_logging: bool = True
    enable_caching: bool = False
    enable_retry: bool = False
    enable_rate_limiting: bool = False
    
    # 性能优化选项
    async_cost_tracking: bool = True
    background_logging: bool = True
    cache_results: bool = False
    cache_ttl: int = 300
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # 限流配置
    rate_limit: Optional[int] = None
    rate_window: int = 60
    
    @classmethod
    def fast_mode(cls) -> 'DecoratorConfig':
        """快速模式配置"""
        return cls(
            mode=DecoratorMode.FAST,
            enable_tracing=False,
            enable_cost_tracking=True,
            enable_postgres_logging=False,
            async_cost_tracking=True,
            background_logging=True
        )
    
    @classmethod
    def full_mode(cls) -> 'DecoratorConfig':
        """完整模式配置"""
        return cls(
            mode=DecoratorMode.FULL,
            enable_tracing=True,
            enable_cost_tracking=True,
            enable_postgres_logging=True,
            async_cost_tracking=True,
            background_logging=True
        )


class UnifiedDecorator:
    """统一装饰器类
    
    提供高性能的统一装饰器功能。
    """
    
    def __init__(self, config: Optional[DecoratorConfig] = None):
        self.config = config or DecoratorConfig()
        self.settings = get_settings()
        
        # 初始化组件
        self._cost_tracker = get_async_cost_tracker() if self.config.async_cost_tracking else None
        self._cache_manager = get_cache_manager() if self.config.enable_caching else None
        self._postgres_logger = None
        if self.config.enable_postgres_logging:
            try:
                # 从配置获取连接字符串，如果没有则跳过
                connection_string = getattr(self.config, 'postgres_connection_string', None)
                if connection_string:
                    self._postgres_logger = PostgresLogger(connection_string)
                else:
                    logger.warning("PostgreSQL logging enabled but no connection string provided")
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL logger: {e}")
        self._enable_tracing = self.config.enable_tracing
        
        # 性能统计
        self._stats = {
            'total_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_execution_time': 0.0,
            'error_count': 0
        }
    
    def __call__(self, func: F) -> F:
        """装饰器调用"""
        if asyncio.iscoroutinefunction(func):
            return self._wrap_async(func)
        else:
            return self._wrap_sync(func)
    
    def _wrap_async(self, func: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """包装异步函数"""
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            return await self._execute_async(func, args, kwargs)
        
        return async_wrapper
    
    def _wrap_sync(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """包装同步函数"""
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            return self._execute_sync(func, args, kwargs)
        
        return sync_wrapper
    
    async def _execute_async(self, func: Callable[..., Awaitable[Any]], args: tuple, kwargs: dict) -> Any:
        """执行异步函数"""
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        function_name = func.__name__
        
        # 更新统计
        self._stats['total_calls'] += 1
        
        try:
            # 检查缓存
            if self.config.enable_caching and self._cache_manager:
                cache_key = self._generate_cache_key(func, args, kwargs)
                cached_result = await self._get_cached_result(cache_key)
                if cached_result is not None:
                    self._stats['cache_hits'] += 1
                    return cached_result
                else:
                    self._stats['cache_misses'] += 1
            
            # 开始追踪
            trace_context = None
            if self._enable_tracing:
                trace_context = TraceContext(trace_id)
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 记录执行时间
            execution_time = time.time() - start_time
            self._update_execution_stats(execution_time)
            
            # 异步处理后续任务
            await self._handle_post_execution(
                trace_id, function_name, args, kwargs, result, 
                execution_time, trace_context
            )
            
            # 缓存结果
            if self.config.enable_caching and self._cache_manager:
                await self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self._stats['error_count'] += 1
            
            # 记录错误
            if self.config.background_logging:
                await submit_background_task(
                    self._log_error,
                    trace_id, function_name, str(e)
                )
            
            # 结束追踪 - 异常情况下自动处理
            pass
            
            raise
    
    def _execute_sync(self, func: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
        """执行同步函数"""
        start_time = time.time()
        trace_id = str(uuid.uuid4())
        function_name = func.__name__
        
        # 更新统计
        self._stats['total_calls'] += 1
        
        try:
            # 检查缓存（同步版本）
            if self.config.enable_caching and self._cache_manager:
                cache_key = self._generate_cache_key(func, args, kwargs)
                cached_result = self._get_cached_result_sync(cache_key)
                if cached_result is not None:
                    self._stats['cache_hits'] += 1
                    return cached_result
                else:
                    self._stats['cache_misses'] += 1
            
            # 开始追踪
            trace_context = None
            if self._enable_tracing:
                trace_context = TraceContext(trace_id)
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 记录执行时间
            execution_time = time.time() - start_time
            self._update_execution_stats(execution_time)
            
            # 同步处理后续任务
            self._handle_post_execution_sync(
                trace_id, function_name, args, kwargs, result, 
                execution_time, trace_context
            )
            
            # 缓存结果（同步版本）
            if self.config.enable_caching and self._cache_manager:
                self._cache_result_sync(cache_key, result)
            
            return result
            
        except Exception as e:
            self._stats['error_count'] += 1
            
            # 记录错误（同步版本）
            if self.config.background_logging:
                self._log_error_sync(trace_id, function_name, str(e))
            
            raise
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 简化的缓存键生成，实际应用中可能需要更复杂的逻辑
        key_parts = [func.__name__]
        
        # 添加参数
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
        
        return ":".join(key_parts)
    
    async def _get_cached_result(self, cache_key: str) -> Any:
        """获取缓存结果"""
        if not self._cache_manager:
            return None
        
        try:
            return self._cache_manager.get(cache_key)
        except Exception as e:
            logger.warning(f"获取缓存失败: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, result: Any) -> None:
        """缓存结果"""
        if not self._cache_manager:
            return
        
        try:
            self._cache_manager.set(cache_key, result, ttl=self.config.cache_ttl)
        except Exception as e:
            logger.warning(f"缓存结果失败: {e}")
    
    # 追踪功能现在通过 TraceContext 自动管理
    
    async def _handle_post_execution(
        self,
        trace_id: str,
        function_name: str,
        args: tuple,
        kwargs: dict,
        result: Any,
        execution_time: float,
        trace_context: Any
    ) -> None:
        """处理执行后任务"""
        tasks = []
        
        # 成本追踪
        if self.config.enable_cost_tracking and self._cost_tracker:
            tasks.append(self._track_cost(trace_id, function_name, args, kwargs, result))
        
        # PostgreSQL 日志记录
        if self.config.enable_postgres_logging and self._postgres_logger:
            if self.config.background_logging:
                tasks.append(submit_background_task(
                    self._log_to_postgres,
                    trace_id, function_name, execution_time, result
                ))
            else:
                tasks.append(self._log_to_postgres(
                    trace_id, function_name, execution_time, result
                ))
        
        # 结束追踪
        if trace_context and self.config.enable_tracing:
            tasks.append(self._end_trace(trace_context))
        
        # 并发执行所有任务
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _handle_post_execution_sync(
        self,
        trace_id: str,
        function_name: str,
        args: tuple,
        kwargs: dict,
        result: Any,
        execution_time: float,
        trace_context: Any
    ) -> None:
        """处理执行后任务（同步版本）"""
        # 成本追踪（同步版本）
        if self.config.enable_cost_tracking and self._cost_tracker:
            self._track_cost_sync(trace_id, function_name, args, kwargs, result)
        
        # PostgreSQL 日志记录（同步版本）
        if self.config.enable_postgres_logging and self._postgres_logger:
            self._log_to_postgres_sync(trace_id, function_name, execution_time, result)
        
        # 结束追踪（同步版本）
        if trace_context and self.config.enable_tracing:
            self._end_trace_sync(trace_context)
    
    def _end_trace_sync(self, trace_context: Any) -> None:
        """结束追踪（同步版本）"""
        try:
            if hasattr(trace_context, 'end'):
                trace_context.end()
        except Exception as e:
            logger.warning(f"结束追踪失败: {e}")
    
    async def _end_trace(self, trace_context: Any) -> None:
        """结束追踪"""
        try:
            if hasattr(trace_context, 'end'):
                trace_context.end()
        except Exception as e:
            logger.warning(f"结束追踪失败: {e}")
    
    async def _track_cost(self, trace_id: str, function_name: str, args: tuple, kwargs: dict, result: Any) -> None:
        """追踪成本"""
        if not self._cost_tracker:
            return
        
        try:
            # 提取成本相关信息
            cost_info = self._extract_cost_info(args, kwargs, result)
            if cost_info:
                # 计算成本（简化计算，实际应根据模型定价）
                input_tokens = cost_info.get('input_tokens', 0)
                output_tokens = cost_info.get('output_tokens', 0)
                estimated_cost = (input_tokens * 0.0015 + output_tokens * 0.002) / 1000  # GPT-4 定价示例
                
                await self._cost_tracker.track_api_call_async(
                    model=cost_info.get('model', 'unknown'),
                    provider=cost_info.get('provider', 'openai'),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost=estimated_cost,
                    duration=cost_info.get('duration', 0.0)
                )
        except Exception as e:
            logger.warning(f"成本追踪失败: {e}")
    
    def _extract_cost_info(self, args: tuple, kwargs: dict, result: Any) -> Optional[Dict[str, Any]]:
        """提取成本信息"""
        # 这里需要根据实际的API响应格式来提取成本信息
        # 这是一个简化的实现
        cost_info = {}
        
        # 从kwargs中提取模型信息
        if 'model' in kwargs:
            cost_info['model'] = kwargs['model']
        
        # 提取提供商信息
        cost_info['provider'] = kwargs.get('provider', 'openai')
        
        # 从结果中提取token信息
        if hasattr(result, 'usage'):
            usage = result.usage
            if hasattr(usage, 'prompt_tokens'):
                cost_info['input_tokens'] = usage.prompt_tokens
            if hasattr(usage, 'completion_tokens'):
                cost_info['output_tokens'] = usage.completion_tokens
        
        # 添加持续时间（如果可用）
        cost_info['duration'] = kwargs.get('duration', 0.0)
        
        return cost_info if cost_info else None
    
    async def _log_to_postgres(self, trace_id: str, function_name: str, execution_time: float, result: Any) -> None:
        """记录到PostgreSQL"""
        if not self._postgres_logger:
            return
        
        try:
            await self._postgres_logger.log_async({
                'trace_id': trace_id,
                'function_name': function_name,
                'execution_time': execution_time,
                'timestamp': time.time(),
                'result_type': type(result).__name__
            })
        except Exception as e:
            logger.warning(f"PostgreSQL日志记录失败: {e}")
    
    async def _log_error(self, trace_id: str, function_name: str, error: str) -> None:
        """记录错误"""
        try:
            if self._postgres_logger:
                await self._postgres_logger.log_async({
                    'trace_id': trace_id,
                    'function_name': function_name,
                    'error': error,
                    'timestamp': time.time(),
                    'level': 'ERROR'
                })
        except Exception as e:
            logger.warning(f"错误日志记录失败: {e}")
    
    def _get_cached_result_sync(self, cache_key: str) -> Any:
        """获取缓存结果（同步版本）"""
        if not self._cache_manager:
            return None
        
        try:
            return self._cache_manager.get(cache_key)
        except Exception as e:
            logger.warning(f"获取缓存失败: {e}")
            return None
    
    def _cache_result_sync(self, cache_key: str, result: Any) -> None:
        """缓存结果（同步版本）"""
        if not self._cache_manager:
            return
        
        try:
            self._cache_manager.set(cache_key, result, ttl=self.config.cache_ttl)
        except Exception as e:
            logger.warning(f"缓存结果失败: {e}")
    
    def _track_cost_sync(self, trace_id: str, function_name: str, args: tuple, kwargs: dict, result: Any) -> None:
        """追踪成本（同步版本）"""
        if not self._cost_tracker:
            return
        
        try:
            cost_info = self._extract_cost_info(args, kwargs, result)
            if cost_info:
                # 同步版本的成本追踪
                self._cost_tracker.track_sync(
                    trace_id=trace_id,
                    function_name=function_name,
                    **cost_info
                )
        except Exception as e:
            logger.warning(f"成本追踪失败: {e}")
    
    def _log_to_postgres_sync(self, trace_id: str, function_name: str, execution_time: float, result: Any) -> None:
        """记录到PostgreSQL（同步版本）"""
        if not self._postgres_logger:
            return
        
        try:
            # 同步版本的PostgreSQL日志记录
            self._postgres_logger.log_sync({
                'trace_id': trace_id,
                'function_name': function_name,
                'execution_time': execution_time,
                'timestamp': time.time(),
                'result_type': type(result).__name__
            })
        except Exception as e:
            logger.warning(f"PostgreSQL日志记录失败: {e}")
    
    def _log_error_sync(self, trace_id: str, function_name: str, error: str) -> None:
        """记录错误（同步版本）"""
        try:
            if self._postgres_logger:
                self._postgres_logger.log_sync({
                    'trace_id': trace_id,
                    'function_name': function_name,
                    'error': error,
                    'timestamp': time.time(),
                    'level': 'ERROR'
                })
        except Exception as e:
            logger.warning(f"错误日志记录失败: {e}")
    
    def _update_execution_stats(self, execution_time: float) -> None:
        """更新执行统计"""
        if self._stats['avg_execution_time'] == 0.0:
            self._stats['avg_execution_time'] = execution_time
        else:
            # 使用指数移动平均
            alpha = 0.1
            self._stats['avg_execution_time'] = (
                alpha * execution_time + 
                (1 - alpha) * self._stats['avg_execution_time']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self._stats.copy()


# 预定义的装饰器实例
fast_decorator = UnifiedDecorator(DecoratorConfig.fast_mode())
full_decorator = UnifiedDecorator(DecoratorConfig.full_mode())


def unified_trace(
    mode: DecoratorMode = DecoratorMode.FULL,
    config: Optional[DecoratorConfig] = None,
    **kwargs
) -> Callable[[F], F]:
    """统一追踪装饰器
    
    Args:
        mode: 装饰器模式
        config: 自定义配置
        **kwargs: 额外配置参数
    
    Returns:
        装饰器函数
    """
    if config is None:
        if mode == DecoratorMode.FAST:
            config = DecoratorConfig.fast_mode()
        elif mode == DecoratorMode.FULL:
            config = DecoratorConfig.full_mode()
        else:
            config = DecoratorConfig(mode=mode)
        
        # 应用额外配置
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    decorator = UnifiedDecorator(config)
    return decorator


def fast_trace(func: F) -> F:
    """快速追踪装饰器"""
    return fast_decorator(func)


def full_trace(func: F) -> F:
    """完整追踪装饰器"""
    return full_decorator(func)


def conditional_unified_decorator(condition: bool, config: DecoratorConfig) -> Callable[[F], F]:
    """条件统一装饰器
    
    根据条件决定是否应用装饰器。
    
    Args:
        condition: 是否应用装饰器的条件
        config: 装饰器配置
    
    Returns:
        装饰器函数
    """
    def decorator(func: F) -> F:
        if condition:
            return UnifiedDecorator(config)(func)
        else:
            return func
    
    return decorator


def smart_decorator(func: F) -> F:
    """智能装饰器
    
    根据系统配置自动选择最适合的装饰器模式。
    
    Args:
        func: 要装饰的函数
    
    Returns:
        装饰后的函数
    """
    perf_config = get_performance_config()
    decorator_config = perf_config.get_decorator_config()
    
    # 转换为 DecoratorConfig
    config = DecoratorConfig(
        mode=DecoratorMode.CUSTOM,
        enable_tracing=decorator_config.enable_tracing,
        enable_cost_tracking=decorator_config.enable_cost_tracking,
        enable_postgres_logging=decorator_config.enable_postgres_logging,
        async_cost_tracking=decorator_config.async_cost_tracking,
        background_logging=decorator_config.background_logging,
        enable_caching=decorator_config.enable_caching,
        enable_retry=decorator_config.enable_retry,
        enable_rate_limiting=decorator_config.enable_rate_limiting
    )
    
    return UnifiedDecorator(config)(func)


# 兼容性装饰器，保持向后兼容
def cost_tracking(func: F) -> F:
    """成本追踪装饰器（兼容性）"""
    config = DecoratorConfig(
        mode=DecoratorMode.CUSTOM,
        enable_tracing=False,
        enable_cost_tracking=True,
        enable_postgres_logging=False,
        async_cost_tracking=True
    )
    return UnifiedDecorator(config)(func)


def with_trace(func: F) -> F:
    """追踪装饰器（兼容性）"""
    config = DecoratorConfig(
        mode=DecoratorMode.CUSTOM,
        enable_tracing=True,
        enable_cost_tracking=False,
        enable_postgres_logging=False
    )
    return UnifiedDecorator(config)(func)


def with_postgres_logging(func: F) -> F:
    """PostgreSQL日志装饰器（兼容性）"""
    config = DecoratorConfig(
        mode=DecoratorMode.CUSTOM,
        enable_tracing=False,
        enable_cost_tracking=False,
        enable_postgres_logging=True,
        background_logging=True
    )
    return UnifiedDecorator(config)(func)


def with_async_trace(func: F) -> F:
    """异步追踪装饰器（兼容性）"""
    config = DecoratorConfig(
        mode=DecoratorMode.CUSTOM,
        enable_tracing=True,
        enable_cost_tracking=False,
        enable_postgres_logging=False
    )
    return UnifiedDecorator(config)(func)