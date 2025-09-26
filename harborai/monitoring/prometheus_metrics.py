#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prometheus指标导出模块

提供API调用次数、响应时间分布、错误率等关键指标的监控。
"""

import time
import functools
from typing import Optional, Dict, Any, Callable
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)
from ..utils.logger import get_logger
from ..utils.exceptions import HarborAIError

logger = get_logger(__name__)

# 全局Prometheus指标实例
_prometheus_metrics: Optional['PrometheusMetrics'] = None


class PrometheusMetrics:
    """Prometheus指标收集器"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """初始化Prometheus指标收集器
        
        Args:
            registry: Prometheus注册表，默认使用全局注册表
        """
        self.registry = registry or CollectorRegistry()
        self._init_metrics()
        logger.info("Prometheus指标收集器已初始化")
    
    def _init_metrics(self):
        """初始化所有指标"""
        # API调用计数器
        self.api_requests_total = Counter(
            'harborai_api_requests_total',
            'API请求总数',
            ['method', 'model', 'provider', 'status'],
            registry=self.registry
        )
        
        # API响应时间直方图
        self.api_request_duration_seconds = Histogram(
            'harborai_api_request_duration_seconds',
            'API请求持续时间（秒）',
            ['method', 'model', 'provider'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0),
            registry=self.registry
        )
        
        # Token使用量计数器
        self.tokens_used_total = Counter(
            'harborai_tokens_used_total',
            'Token使用总量',
            ['model', 'provider', 'token_type'],
            registry=self.registry
        )
        
        # 成本计数器
        self.cost_total = Counter(
            'harborai_cost_total',
            'API调用总成本（美元）',
            ['model', 'provider'],
            registry=self.registry
        )
        
        # 错误率计数器
        self.api_errors_total = Counter(
            'harborai_api_errors_total',
            'API错误总数',
            ['method', 'model', 'provider', 'error_type'],
            registry=self.registry
        )
        
        # 当前活跃连接数
        self.active_connections = Gauge(
            'harborai_active_connections',
            '当前活跃连接数',
            ['provider'],
            registry=self.registry
        )
        
        # 重试次数计数器
        self.retries_total = Counter(
            'harborai_retries_total',
            '重试总次数',
            ['model', 'provider', 'retry_reason'],
            registry=self.registry
        )
        
        # 缓存命中率
        self.cache_hits_total = Counter(
            'harborai_cache_hits_total',
            '缓存命中总数',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            'harborai_cache_misses_total',
            '缓存未命中总数',
            ['cache_type'],
            registry=self.registry
        )
        
        # 系统信息
        self.system_info = Info(
            'harborai_system_info',
            'HarborAI系统信息',
            registry=self.registry
        )
    
    def record_api_request(self, method: str, model: str, provider: str, 
                          duration: float, status: str = 'success',
                          error_type: Optional[str] = None):
        """记录API请求指标
        
        Args:
            method: API方法名
            model: 模型名称
            provider: 提供商名称
            duration: 请求持续时间（秒）
            status: 请求状态（success/error）
            error_type: 错误类型（仅在status为error时使用）
        """
        # 记录请求总数
        self.api_requests_total.labels(
            method=method,
            model=model,
            provider=provider,
            status=status
        ).inc()
        
        # 记录响应时间
        self.api_request_duration_seconds.labels(
            method=method,
            model=model,
            provider=provider
        ).observe(duration)
        
        # 记录错误
        if status == 'error' and error_type:
            self.api_errors_total.labels(
                method=method,
                model=model,
                provider=provider,
                error_type=error_type
            ).inc()
    
    def record_token_usage(self, model: str, provider: str, 
                          prompt_tokens: int, completion_tokens: int):
        """记录Token使用量
        
        Args:
            model: 模型名称
            provider: 提供商名称
            prompt_tokens: 输入Token数量
            completion_tokens: 输出Token数量
        """
        self.tokens_used_total.labels(
            model=model,
            provider=provider,
            token_type='prompt'
        ).inc(prompt_tokens)
        
        self.tokens_used_total.labels(
            model=model,
            provider=provider,
            token_type='completion'
        ).inc(completion_tokens)
    
    def record_cost(self, model: str, provider: str, cost: float):
        """记录API调用成本
        
        Args:
            model: 模型名称
            provider: 提供商名称
            cost: 成本（美元）
        """
        self.cost_total.labels(
            model=model,
            provider=provider
        ).inc(cost)
    
    def record_retry(self, model: str, provider: str, retry_reason: str):
        """记录重试事件
        
        Args:
            model: 模型名称
            provider: 提供商名称
            retry_reason: 重试原因
        """
        self.retries_total.labels(
            model=model,
            provider=provider,
            retry_reason=retry_reason
        ).inc()
    
    def record_cache_hit(self, cache_type: str):
        """记录缓存命中
        
        Args:
            cache_type: 缓存类型
        """
        self.cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """记录缓存未命中
        
        Args:
            cache_type: 缓存类型
        """
        self.cache_misses_total.labels(cache_type=cache_type).inc()
    
    def set_active_connections(self, provider: str, count: int):
        """设置活跃连接数
        
        Args:
            provider: 提供商名称
            count: 连接数
        """
        self.active_connections.labels(provider=provider).set(count)
    
    def set_system_info(self, info: Dict[str, str]):
        """设置系统信息
        
        Args:
            info: 系统信息字典
        """
        self.system_info.info(info)
    
    def get_metrics(self) -> str:
        """获取Prometheus格式的指标数据
        
        Returns:
            Prometheus格式的指标字符串
        """
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """获取指标数据的Content-Type
        
        Returns:
            Content-Type字符串
        """
        return CONTENT_TYPE_LATEST


def get_prometheus_metrics() -> Optional[PrometheusMetrics]:
    """获取全局Prometheus指标实例
    
    Returns:
        PrometheusMetrics实例，如果未初始化则返回None
    """
    return _prometheus_metrics


def init_prometheus_metrics(registry: Optional[CollectorRegistry] = None) -> PrometheusMetrics:
    """初始化全局Prometheus指标实例
    
    Args:
        registry: Prometheus注册表
        
    Returns:
        PrometheusMetrics实例
    """
    global _prometheus_metrics
    _prometheus_metrics = PrometheusMetrics(registry)
    return _prometheus_metrics


def prometheus_middleware(func: Callable) -> Callable:
    """Prometheus监控中间件装饰器
    
    自动记录函数调用的指标信息。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        metrics = get_prometheus_metrics()
        if not metrics:
            return func(*args, **kwargs)
        
        # 提取参数
        method = func.__name__
        model = kwargs.get('model', 'unknown')
        provider = kwargs.get('provider', 'unknown')
        
        start_time = time.time()
        status = 'success'
        error_type = None
        
        try:
            result = func(*args, **kwargs)
            
            # 记录Token使用量和成本
            if hasattr(result, 'usage') and result.usage:
                usage = result.usage
                metrics.record_token_usage(
                    model=model,
                    provider=provider,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens
                )
                
                # 计算并记录成本
                from ..core.pricing import PricingCalculator
                cost = PricingCalculator.calculate_cost(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    model_name=model
                )
                if cost is not None:
                    metrics.record_cost(model=model, provider=provider, cost=cost)
            
            return result
            
        except HarborAIError as e:
            status = 'error'
            error_type = type(e).__name__
            raise
        except Exception as e:
            status = 'error'
            error_type = 'UnexpectedError'
            raise
        finally:
            duration = time.time() - start_time
            metrics.record_api_request(
                method=method,
                model=model,
                provider=provider,
                duration=duration,
                status=status,
                error_type=error_type
            )
    
    return wrapper


# 异步版本的中间件
def prometheus_async_middleware(func: Callable) -> Callable:
    """异步Prometheus监控中间件装饰器"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        metrics = get_prometheus_metrics()
        if not metrics:
            return await func(*args, **kwargs)
        
        # 提取参数
        method = func.__name__
        model = kwargs.get('model', 'unknown')
        provider = kwargs.get('provider', 'unknown')
        
        start_time = time.time()
        status = 'success'
        error_type = None
        
        try:
            result = await func(*args, **kwargs)
            
            # 记录Token使用量和成本
            if hasattr(result, 'usage') and result.usage:
                usage = result.usage
                metrics.record_token_usage(
                    model=model,
                    provider=provider,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens
                )
                
                # 计算并记录成本
                from ..core.pricing import PricingCalculator
                cost = PricingCalculator.calculate_cost(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    model_name=model
                )
                if cost is not None:
                    metrics.record_cost(model=model, provider=provider, cost=cost)
            
            return result
            
        except HarborAIError as e:
            status = 'error'
            error_type = type(e).__name__
            raise
        except Exception as e:
            status = 'error'
            error_type = 'UnexpectedError'
            raise
        finally:
            duration = time.time() - start_time
            metrics.record_api_request(
                method=method,
                model=model,
                provider=provider,
                duration=duration,
                status=status,
                error_type=error_type
            )
    
    return wrapper