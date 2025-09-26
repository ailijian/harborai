"""装饰器模块，提供日志、重试、追踪等功能。"""

import asyncio
import functools
import time
from typing import Any, Callable, Optional

from harborai.utils.logger import get_logger
from harborai.storage.postgres_logger import get_postgres_logger
from harborai.utils.exceptions import HarborAIError
from harborai.utils.tracer import generate_trace_id
from harborai.core.pricing import PricingCalculator
from harborai.monitoring.token_statistics import record_token_usage

logger = get_logger(__name__)


def with_trace(func: Callable) -> Callable:
    """为函数调用添加追踪ID。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        trace_id = kwargs.get('trace_id') or generate_trace_id()
        kwargs['trace_id'] = trace_id
        
        logger.info(f"[{trace_id}] Starting {func.__name__}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"[{trace_id}] Completed {func.__name__} in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{trace_id}] Failed {func.__name__} after {duration:.3f}s: {e}")
            raise
    
    return wrapper


def with_async_trace(func: Callable) -> Callable:
    """为异步函数调用添加追踪ID。"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        trace_id = kwargs.get('trace_id') or generate_trace_id()
        kwargs['trace_id'] = trace_id
        
        logger.info(f"[{trace_id}] Starting async {func.__name__}")
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"[{trace_id}] Completed async {func.__name__} in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{trace_id}] Failed async {func.__name__} after {duration:.3f}s: {e}")
            raise
    
    return wrapper


def with_logging(func: Callable) -> Callable:
    """为函数调用添加详细日志记录。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 记录请求参数（脱敏处理）
        safe_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['api_key', 'authorization']}
        logger.debug(f"Calling {func.__name__} with args: {safe_kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Successfully completed {func.__name__}")
            return result
        except HarborAIError as e:
            logger.warning(f"HarborAI error in {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    
    return wrapper


def with_async_logging(func: Callable) -> Callable:
    """为异步函数调用添加详细日志记录。"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # 记录请求参数（脱敏处理）
        safe_kwargs = {k: v for k, v in kwargs.items() 
                      if k not in ['api_key', 'authorization']}
        logger.debug(f"Calling async {func.__name__} with args: {safe_kwargs}")
        
        try:
            result = await func(*args, **kwargs)
            logger.debug(f"Successfully completed async {func.__name__}")
            return result
        except HarborAIError as e:
            logger.warning(f"HarborAI error in async {func.__name__}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in async {func.__name__}: {e}")
            raise
    
    return wrapper


def cost_tracking(func: Callable) -> Callable:
    """为函数调用添加成本追踪。"""
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            cost_tracking_enabled = kwargs.get('cost_tracking', True)
            
            if not cost_tracking_enabled:
                return await func(*args, **kwargs)
            
            start_time = time.time()
            model = kwargs.get('model', 'unknown')
            trace_id = kwargs.get('trace_id', 'unknown')
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 记录成本相关信息
                if hasattr(result, 'usage') and result.usage:
                    usage = result.usage
                    
                    # 计算成本
                    cost = PricingCalculator.calculate_cost(
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        model_name=model
                    )
                    
                    cost_info = f", Cost: ${cost:.6f}" if cost is not None else ", Cost: N/A"
                    
                    logger.info(
                        f"[{trace_id}] Cost tracking - Model: {model}, "
                        f"Input tokens: {usage.prompt_tokens}, "
                        f"Output tokens: {usage.completion_tokens}, "
                        f"Total tokens: {usage.total_tokens}, "
                        f"Duration: {duration:.3f}s{cost_info}"
                    )
                    
                    # 记录Token使用统计
                    record_token_usage(
                        trace_id=trace_id,
                        model=model,
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        duration=duration,
                        success=True
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # 记录失败的Token使用统计
                record_token_usage(
                    trace_id=trace_id,
                    model=model,
                    input_tokens=0,  # 失败时无法获取准确的token数
                    output_tokens=0,
                    duration=duration,
                    success=False,
                    error=str(e)
                )
                
                raise
        
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            cost_tracking_enabled = kwargs.get('cost_tracking', True)
            
            if not cost_tracking_enabled:
                return func(*args, **kwargs)
            
            start_time = time.time()
            model = kwargs.get('model', 'unknown')
            trace_id = kwargs.get('trace_id', 'unknown')
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 记录成本相关信息
                if hasattr(result, 'usage') and result.usage:
                    usage = result.usage
                    
                    # 计算成本
                    cost = PricingCalculator.calculate_cost(
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        model_name=model
                    )
                    
                    cost_info = f", Cost: ${cost:.6f}" if cost is not None else ", Cost: N/A"
                    
                    logger.info(
                        f"[{trace_id}] Cost tracking - Model: {model}, "
                        f"Input tokens: {usage.prompt_tokens}, "
                        f"Output tokens: {usage.completion_tokens}, "
                        f"Total tokens: {usage.total_tokens}, "
                        f"Duration: {duration:.3f}s{cost_info}"
                    )
                    
                    # 记录Token使用统计
                    record_token_usage(
                        trace_id=trace_id,
                        model=model,
                        input_tokens=usage.prompt_tokens,
                        output_tokens=usage.completion_tokens,
                        duration=duration,
                        success=True
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # 记录失败的Token使用统计
                record_token_usage(
                    trace_id=trace_id,
                    model=model,
                    input_tokens=0,  # 失败时无法获取准确的token数
                    output_tokens=0,
                    duration=duration,
                    success=False,
                    error=str(e)
                )
                
                raise
        
        return sync_wrapper


def with_postgres_logging(func: Callable) -> Callable:
    """为函数调用添加PostgreSQL日志持久化。"""
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            postgres_logger = get_postgres_logger()
            if not postgres_logger:
                # 如果没有配置PostgreSQL日志记录器，直接执行原函数
                return await func(*args, **kwargs)
            
            trace_id = kwargs.get('trace_id') or generate_trace_id()
            kwargs['trace_id'] = trace_id
            
            # 记录请求日志
            model = kwargs.get('model', 'unknown')
            messages = kwargs.get('messages', [])
            postgres_logger.log_request(
                trace_id=trace_id,
                model=model,
                messages=messages,
                **{k: v for k, v in kwargs.items() if k not in ['messages', 'model', 'trace_id']}
            )
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                latency = time.time() - start_time
                
                # 记录响应日志
                postgres_logger.log_response(
                    trace_id=trace_id,
                    response=result,
                    latency=latency,
                    success=True
                )
                
                return result
            except Exception as e:
                latency = time.time() - start_time
                
                # 记录错误日志
                postgres_logger.log_response(
                    trace_id=trace_id,
                    response=None,
                    latency=latency,
                    success=False,
                    error=str(e)
                )
                
                raise
        
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            postgres_logger = get_postgres_logger()
            if not postgres_logger:
                # 如果没有配置PostgreSQL日志记录器，直接执行原函数
                return func(*args, **kwargs)
            
            trace_id = kwargs.get('trace_id') or generate_trace_id()
            kwargs['trace_id'] = trace_id
            
            # 记录请求日志
            model = kwargs.get('model', 'unknown')
            messages = kwargs.get('messages', [])
            postgres_logger.log_request(
                trace_id=trace_id,
                model=model,
                messages=messages,
                **{k: v for k, v in kwargs.items() if k not in ['messages', 'model', 'trace_id']}
            )
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency = time.time() - start_time
                
                # 记录响应日志
                postgres_logger.log_response(
                    trace_id=trace_id,
                    response=result,
                    latency=latency,
                    success=True
                )
                
                return result
            except Exception as e:
                latency = time.time() - start_time
                
                # 记录错误日志
                postgres_logger.log_response(
                    trace_id=trace_id,
                    response=None,
                    latency=latency,
                    success=False,
                    error=str(e)
                )
                
                raise
        
        return sync_wrapper