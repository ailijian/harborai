"""装饰器模块，提供日志、重试、追踪等功能。"""

import functools
import time
from typing import Any, Callable, Optional

from ..utils.logger import get_logger
from ..utils.tracer import generate_trace_id
from ..utils.exceptions import HarborAIError

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
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cost_tracking_enabled = kwargs.get('cost_tracking', True)
        
        if not cost_tracking_enabled:
            return func(*args, **kwargs)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        
        # 记录成本相关信息
        if hasattr(result, 'usage') and result.usage:
            usage = result.usage
            model = kwargs.get('model', 'unknown')
            trace_id = kwargs.get('trace_id', 'unknown')
            
            logger.info(
                f"[{trace_id}] Cost tracking - Model: {model}, "
                f"Input tokens: {usage.prompt_tokens}, "
                f"Output tokens: {usage.completion_tokens}, "
                f"Total tokens: {usage.total_tokens}, "
                f"Duration: {duration:.3f}s"
            )
        
        return result
    
    return wrapper