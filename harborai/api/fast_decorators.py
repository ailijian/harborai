"""快速路径装饰器模块，提供轻量级的装饰器实现。"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Optional

from ..utils.tracer import generate_trace_id
from ..config.settings import get_settings
from ..config.performance import get_performance_config
from ..core.async_cost_tracking import get_async_cost_tracker

logger = logging.getLogger(__name__)
settings = get_settings()


def fast_trace(func: Callable) -> Callable:
    """轻量级追踪装饰器，仅记录基本信息。"""
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace_id = kwargs.get('trace_id') or generate_trace_id()
            kwargs['trace_id'] = trace_id
            
            if settings.enable_detailed_tracing:
                logger.debug(f"[{trace_id}] Fast path: {func.__name__}")
            
            return await func(*args, **kwargs)
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            trace_id = kwargs.get('trace_id') or generate_trace_id()
            kwargs['trace_id'] = trace_id
            
            if settings.enable_detailed_tracing:
                logger.debug(f"[{trace_id}] Fast path: {func.__name__}")
            
            return func(*args, **kwargs)
        return sync_wrapper


def fast_cost_tracking(func: Callable) -> Callable:
    """轻量级成本追踪装饰器，异步记录成本信息。"""
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cost_tracking_enabled = kwargs.get('cost_tracking', True)
            
            if not cost_tracking_enabled or not settings.enable_cost_tracking:
                return await func(*args, **kwargs)
            
            start_time = time.time()
            model = kwargs.get('model', 'unknown')
            trace_id = kwargs.get('trace_id', 'unknown')
            
            try:
                result = await func(*args, **kwargs)
                
                # 异步记录成本信息，不阻塞主流程
                if hasattr(result, 'usage') and result.usage:
                    asyncio.create_task(_async_record_cost(
                        trace_id=trace_id,
                        model=model,
                        usage=result.usage,
                        duration=time.time() - start_time,
                        success=True
                    ))
                
                return result
            except Exception as e:
                # 异步记录失败信息
                asyncio.create_task(_async_record_cost(
                    trace_id=trace_id,
                    model=model,
                    usage=None,
                    duration=time.time() - start_time,
                    success=False,
                    error=str(e)
                ))
                raise
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cost_tracking_enabled = kwargs.get('cost_tracking', True)
            
            if not cost_tracking_enabled or not settings.enable_cost_tracking:
                return func(*args, **kwargs)
            
            start_time = time.time()
            model = kwargs.get('model', 'unknown')
            trace_id = kwargs.get('trace_id', 'unknown')
            
            try:
                result = func(*args, **kwargs)
                
                # 同步版本仍然需要记录，但尽量简化
                if hasattr(result, 'usage') and result.usage:
                    logger.debug(
                        f"[{trace_id}] Fast cost - Model: {model}, "
                        f"Tokens: {result.usage.total_tokens}, "
                        f"Duration: {time.time() - start_time:.3f}s"
                    )
                
                return result
            except Exception as e:
                logger.debug(
                    f"[{trace_id}] Fast cost error - Model: {model}, "
                    f"Duration: {time.time() - start_time:.3f}s, Error: {str(e)}"
                )
                raise
        return sync_wrapper


async def _async_record_cost(
    trace_id: str,
    model: str,
    usage: Optional[Any],
    duration: float,
    success: bool,
    error: Optional[str] = None
):
    """异步记录成本信息，避免阻塞主流程。"""
    try:
        from harborai.core.pricing import PricingCalculator
        from harborai.monitoring.token_statistics import record_token_usage
        
        if success and usage:
            # 计算成本
            cost = PricingCalculator.calculate_cost(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                model_name=model
            )
            
            logger.debug(
                f"[{trace_id}] Fast cost tracking - Model: {model}, "
                f"Input: {usage.prompt_tokens}, Output: {usage.completion_tokens}, "
                f"Cost: ¥{cost:.6f if cost else 'N/A'}, Duration: {duration:.3f}s"
            )
            
            # 使用异步成本追踪器
            async_tracker = get_async_cost_tracker()
            await async_tracker.track_api_call_async(
                model=model,
                provider="unknown",
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cost=cost or 0.0,
                duration=duration,
                success=True,
                trace_id=trace_id
            )
        else:
            # 使用异步成本追踪器记录失败
            async_tracker = get_async_cost_tracker()
            await async_tracker.track_api_call_async(
                model=model,
                provider="unknown",
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                duration=duration,
                success=False,
                trace_id=trace_id
            )
    except Exception as e:
        # 记录成本信息失败不应该影响主流程
        logger.warning(f"Failed to record cost asynchronously: {e}")


def conditional_decorator(decorator_func: Callable, condition: bool):
    """条件装饰器，根据条件决定是否应用装饰器。"""
    def decorator(func: Callable) -> Callable:
        if condition:
            return decorator_func(func)
        return func
    return decorator


def fast_path_decorators(func: Callable) -> Callable:
    """快速路径装饰器组合，根据配置动态应用装饰器。"""
    perf_config = get_performance_config()
    decorator_config = perf_config.get_decorator_config()
    
    # 应用轻量级追踪
    if decorator_config.enable_tracing:
        func = fast_trace(func)
    
    # 应用轻量级成本追踪
    if decorator_config.enable_cost_tracking:
        func = fast_cost_tracking(func)
    
    return func