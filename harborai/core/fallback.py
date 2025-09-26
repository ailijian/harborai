#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 降级策略模块

提供服务降级、故障转移和健康监控功能，确保系统在部分服务不可用时仍能正常运行。
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor

from .exceptions import (
    HarborAIError,
    ServiceUnavailableError,
    RateLimitError,
    TimeoutError,
    QuotaExceededError,
    ModelNotAvailableError
)


logger = logging.getLogger(__name__)


class FallbackTrigger(Enum):
    """降级触发条件枚举"""
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    AVAILABILITY = "availability"
    QUOTA_LIMIT = "quota_limit"
    MANUAL = "manual"


class ServiceStatus(Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class FallbackRule:
    """降级规则配置"""
    trigger: FallbackTrigger
    threshold: float
    window_size: int = 10
    min_requests: int = 5
    cooldown_period: int = 60  # 冷却期（秒）
    enabled: bool = True
    priority: int = 1  # 优先级，数字越小优先级越高


@dataclass
class ServiceEndpoint:
    """服务端点配置"""
    name: str
    url: str
    priority: int  # 优先级，数字越小优先级越高
    model_type: str
    capabilities: List[str] = field(default_factory=list)
    cost_per_token: float = 0.0
    max_tokens: int = 4096
    rate_limit: int = 100  # 每分钟请求数
    timeout: float = 30.0
    status: ServiceStatus = ServiceStatus.HEALTHY
    last_check: Optional[datetime] = None
    health_check_url: Optional[str] = None
    retry_count: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 60


@dataclass
class FallbackAttempt:
    """降级尝试记录"""
    timestamp: datetime
    original_service: str
    fallback_service: str
    trigger_reason: str
    success: bool
    response_time: float
    error: Optional[str] = None
    cost_impact: float = 0.0
    trace_id: Optional[str] = None


class FallbackMetrics:
    """降级指标收集器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.attempts = deque(maxlen=window_size)
        self.service_metrics = defaultdict(lambda: {
            'requests': deque(maxlen=window_size),
            'successes': 0,
            'failures': 0,
            'total_response_time': 0.0,
            'last_request_time': None
        })
        self._lock = threading.Lock()
    
    def record_attempt(self, attempt: FallbackAttempt):
        """记录降级尝试"""
        with self._lock:
            self.attempts.append(attempt)
            
            # 更新服务指标
            service_name = attempt.fallback_service
            metrics = self.service_metrics[service_name]
            
            metrics['requests'].append({
                'timestamp': attempt.timestamp,
                'success': attempt.success,
                'response_time': attempt.response_time
            })
            
            if attempt.success:
                metrics['successes'] += 1
            else:
                metrics['failures'] += 1
            
            metrics['total_response_time'] += attempt.response_time
            metrics['last_request_time'] = attempt.timestamp
    
    def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """获取服务指标"""
        with self._lock:
            metrics = self.service_metrics[service_name]
            requests = list(metrics['requests'])
            
            if not requests:
                return {
                    'request_count': 0,
                    'success_rate': 0.0,
                    'failure_rate': 0.0,
                    'average_response_time': 0.0,
                    'last_request_time': None
                }
            
            total_requests = len(requests)
            successful_requests = sum(1 for req in requests if req['success'])
            total_response_time = sum(req['response_time'] for req in requests)
            
            return {
                'request_count': total_requests,
                'success_rate': successful_requests / total_requests,
                'failure_rate': (total_requests - successful_requests) / total_requests,
                'average_response_time': total_response_time / total_requests,
                'last_request_time': metrics['last_request_time']
            }
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """获取降级统计信息"""
        with self._lock:
            attempts = list(self.attempts)
            
            if not attempts:
                return {
                    'total_attempts': 0,
                    'success_rate': 0.0,
                    'most_common_trigger': None,
                    'average_cost_impact': 0.0
                }
            
            total_attempts = len(attempts)
            successful_attempts = sum(1 for attempt in attempts if attempt.success)
            
            # 统计触发原因
            trigger_counts = defaultdict(int)
            for attempt in attempts:
                trigger_counts[attempt.trigger_reason] += 1
            
            most_common_trigger = max(trigger_counts.items(), key=lambda x: x[1])[0] if trigger_counts else None
            
            # 计算平均成本影响
            total_cost_impact = sum(attempt.cost_impact for attempt in attempts)
            average_cost_impact = total_cost_impact / total_attempts
            
            return {
                'total_attempts': total_attempts,
                'success_rate': successful_attempts / total_attempts,
                'most_common_trigger': most_common_trigger,
                'average_cost_impact': average_cost_impact,
                'trigger_distribution': dict(trigger_counts)
            }


class ServiceHealthMonitor:
    """服务健康监控器"""
    
    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.endpoints: Dict[str, ServiceEndpoint] = {}
        self.health_status: Dict[str, ServiceStatus] = {}
        self.last_check_times: Dict[str, datetime] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._monitor_thread = None
        self._lock = threading.Lock()
    
    def register_endpoint(self, endpoint: ServiceEndpoint):
        """注册服务端点"""
        with self._lock:
            self.endpoints[endpoint.name] = endpoint
            self.health_status[endpoint.name] = ServiceStatus.UNKNOWN
            self.circuit_breakers[endpoint.name] = {
                'failure_count': 0,
                'last_failure_time': None,
                'state': 'closed'  # closed, open, half-open
            }
    
    def start_monitoring(self):
        """开始健康监控"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Service health monitoring started")
    
    def stop_monitoring(self):
        """停止健康监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Service health monitoring stopped")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                self._check_all_endpoints()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def _check_all_endpoints(self):
        """检查所有端点健康状态"""
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for endpoint_name in self.endpoints:
                future = executor.submit(self._check_endpoint_health, endpoint_name)
                futures.append((endpoint_name, future))
            
            for endpoint_name, future in futures:
                try:
                    future.result(timeout=10)
                except Exception as e:
                    logger.error(f"Health check failed for {endpoint_name}: {e}")
                    self._update_health_status(endpoint_name, ServiceStatus.UNAVAILABLE)
    
    def _check_endpoint_health(self, endpoint_name: str):
        """检查单个端点健康状态"""
        endpoint = self.endpoints[endpoint_name]
        
        # 检查熔断器状态
        circuit_breaker = self.circuit_breakers[endpoint_name]
        if circuit_breaker['state'] == 'open':
            # 检查是否可以进入半开状态
            if (circuit_breaker['last_failure_time'] and 
                datetime.now() - circuit_breaker['last_failure_time'] > 
                timedelta(seconds=endpoint.circuit_breaker_timeout)):
                circuit_breaker['state'] = 'half-open'
                logger.info(f"Circuit breaker for {endpoint_name} moved to half-open state")
            else:
                self._update_health_status(endpoint_name, ServiceStatus.UNAVAILABLE)
                return
        
        try:
            # 这里应该实现实际的健康检查逻辑
            # 暂时使用简单的状态检查
            if endpoint.health_check_url:
                # 实际项目中应该发送HTTP请求到健康检查端点
                pass
            
            # 模拟健康检查成功
            self._update_health_status(endpoint_name, ServiceStatus.HEALTHY)
            
            # 重置熔断器
            if circuit_breaker['state'] in ['open', 'half-open']:
                circuit_breaker['state'] = 'closed'
                circuit_breaker['failure_count'] = 0
                logger.info(f"Circuit breaker for {endpoint_name} reset to closed state")
                
        except Exception as e:
            logger.error(f"Health check failed for {endpoint_name}: {e}")
            self._handle_health_check_failure(endpoint_name)
    
    def _handle_health_check_failure(self, endpoint_name: str):
        """处理健康检查失败"""
        circuit_breaker = self.circuit_breakers[endpoint_name]
        circuit_breaker['failure_count'] += 1
        circuit_breaker['last_failure_time'] = datetime.now()
        
        endpoint = self.endpoints[endpoint_name]
        if circuit_breaker['failure_count'] >= endpoint.circuit_breaker_threshold:
            circuit_breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened for {endpoint_name} after {circuit_breaker['failure_count']} failures")
        
        self._update_health_status(endpoint_name, ServiceStatus.UNAVAILABLE)
    
    def _update_health_status(self, endpoint_name: str, status: ServiceStatus):
        """更新健康状态"""
        with self._lock:
            old_status = self.health_status.get(endpoint_name)
            self.health_status[endpoint_name] = status
            self.last_check_times[endpoint_name] = datetime.now()
            
            if old_status != status:
                logger.info(f"Service {endpoint_name} status changed from {old_status} to {status}")
    
    def get_health_status(self, endpoint_name: str) -> ServiceStatus:
        """获取服务健康状态"""
        with self._lock:
            return self.health_status.get(endpoint_name, ServiceStatus.UNKNOWN)
    
    def is_service_available(self, endpoint_name: str) -> bool:
        """检查服务是否可用"""
        status = self.get_health_status(endpoint_name)
        circuit_breaker = self.circuit_breakers.get(endpoint_name, {})
        
        return (status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED] and 
                circuit_breaker.get('state', 'closed') != 'open')


class FallbackDecisionEngine:
    """降级决策引擎"""
    
    def __init__(self, rules: List[FallbackRule] = None):
        self.rules = rules or []
        self.metrics = FallbackMetrics()
        self.health_monitor = ServiceHealthMonitor()
        self._lock = threading.Lock()
    
    def add_rule(self, rule: FallbackRule):
        """添加降级规则"""
        with self._lock:
            self.rules.append(rule)
            # 按优先级排序
            self.rules.sort(key=lambda r: r.priority)
    
    def should_trigger_fallback(self, service_name: str, error: Exception = None) -> tuple[bool, str]:
        """判断是否应该触发降级"""
        with self._lock:
            # 检查服务健康状态
            if not self.health_monitor.is_service_available(service_name):
                return True, "Service unavailable"
            
            # 检查错误类型
            if error:
                if isinstance(error, (ServiceUnavailableError, QuotaExceededError)):
                    return True, f"Service error: {type(error).__name__}"
                elif isinstance(error, RateLimitError):
                    return True, "Rate limit exceeded"
                elif isinstance(error, TimeoutError):
                    return True, "Request timeout"
            
            # 检查降级规则
            service_metrics = self.metrics.get_service_metrics(service_name)
            
            for rule in self.rules:
                if not rule.enabled:
                    continue
                
                if self._evaluate_rule(rule, service_metrics):
                    return True, f"Rule triggered: {rule.trigger.value}"
            
            return False, "No fallback needed"
    
    def _evaluate_rule(self, rule: FallbackRule, metrics: Dict[str, Any]) -> bool:
        """评估降级规则"""
        if metrics['request_count'] < rule.min_requests:
            return False
        
        if rule.trigger == FallbackTrigger.ERROR_RATE:
            return metrics['failure_rate'] >= rule.threshold
        elif rule.trigger == FallbackTrigger.RESPONSE_TIME:
            return metrics['average_response_time'] >= rule.threshold
        elif rule.trigger == FallbackTrigger.AVAILABILITY:
            return metrics['success_rate'] < rule.threshold
        
        return False
    
    def select_fallback_service(self, 
                              original_service: str, 
                              available_services: List[str],
                              request_context: Dict[str, Any] = None) -> Optional[str]:
        """选择降级服务"""
        # 过滤掉原始服务和不可用的服务
        candidates = [
            s for s in available_services 
            if s != original_service and self.health_monitor.is_service_available(s)
        ]
        
        if not candidates:
            return None
        
        # 简单按名称排序，实际应该根据策略选择
        return sorted(candidates)[0]
    
    def get_fallback_chain(self, 
                          original_service: str, 
                          available_services: List[str]) -> List[str]:
        """获取降级链"""
        chain = []
        remaining_services = available_services.copy()
        current_service = original_service
        
        # 移除原始服务
        if current_service in remaining_services:
            remaining_services.remove(current_service)
        
        # 构建降级链
        while remaining_services:
            next_service = self.select_fallback_service(
                current_service, remaining_services
            )
            if next_service:
                chain.append(next_service)
                remaining_services.remove(next_service)
                current_service = next_service
            else:
                break
        
        return chain


class FallbackStrategy(ABC):
    """降级策略抽象基类"""
    
    @abstractmethod
    def select_fallback_service(self, 
                              original_service: str, 
                              available_services: List[str],
                              request_context: Dict[str, Any] = None) -> Optional[str]:
        """选择降级服务"""
        pass


class PriorityFallbackStrategy(FallbackStrategy):
    """基于优先级的降级策略"""
    
    def __init__(self, service_priorities: Dict[str, int]):
        self.service_priorities = service_priorities
    
    def select_fallback_service(self, 
                              original_service: str, 
                              available_services: List[str],
                              request_context: Dict[str, Any] = None) -> Optional[str]:
        """根据优先级选择降级服务"""
        # 过滤掉原始服务
        candidates = [s for s in available_services if s != original_service]
        
        if not candidates:
            return None
        
        # 按优先级排序（数字越小优先级越高）
        candidates.sort(key=lambda s: self.service_priorities.get(s, float('inf')))
        
        return candidates[0]


class CostOptimizedFallbackStrategy(FallbackStrategy):
    """成本优化的降级策略"""
    
    def __init__(self, service_costs: Dict[str, float]):
        self.service_costs = service_costs
    
    def select_fallback_service(self, 
                              original_service: str, 
                              available_services: List[str],
                              request_context: Dict[str, Any] = None) -> Optional[str]:
        """根据成本选择降级服务"""
        candidates = [s for s in available_services if s != original_service]
        
        if not candidates:
            return None
        
        # 按成本排序（成本越低越优先）
        candidates.sort(key=lambda s: self.service_costs.get(s, float('inf')))
        
        return candidates[0]


class FallbackManager:
    """降级管理器"""
    
    def __init__(self, 
                 strategy: FallbackStrategy = None,
                 decision_engine: FallbackDecisionEngine = None):
        self.strategy = strategy or PriorityFallbackStrategy({})
        self.decision_engine = decision_engine or FallbackDecisionEngine()
        self.endpoints: Dict[str, ServiceEndpoint] = {}
        self.metrics = FallbackMetrics()
        self._lock = threading.Lock()
    
    def register_endpoint(self, endpoint: ServiceEndpoint):
        """注册服务端点"""
        with self._lock:
            self.endpoints[endpoint.name] = endpoint
            self.decision_engine.health_monitor.register_endpoint(endpoint)
    
    def start_monitoring(self):
        """开始监控"""
        self.decision_engine.health_monitor.start_monitoring()
    
    def stop_monitoring(self):
        """停止监控"""
        self.decision_engine.health_monitor.stop_monitoring()
    
    def execute_with_fallback(self, 
                            primary_service: str,
                            request_func: Callable,
                            request_data: Dict[str, Any],
                            trace_id: str = None) -> Dict[str, Any]:
        """执行带降级的请求"""
        start_time = time.time()
        original_service = primary_service
        
        # 生成trace_id如果没有提供
        if trace_id is None:
            trace_id = f"fallback_{int(time.time() * 1000)}_{hash(primary_service) % 10000}"
        
        # 获取完整的降级链（按优先级顺序）
        fallback_chain = self._build_fallback_chain(primary_service)
        
        # 记录所有尝试的错误信息
        attempt_errors = []
        
        for attempt_count, current_service in enumerate(fallback_chain):
            attempt_start_time = time.time()
            
            try:
                # 检查服务是否可用
                if not self.decision_engine.health_monitor.is_service_available(current_service):
                    error_msg = f"Service {current_service} is not available"
                    attempt_errors.append({
                        'service': current_service,
                        'error': error_msg,
                        'attempt': attempt_count + 1,
                        'trace_id': trace_id
                    })
                    continue
                
                # 执行请求
                logger.info(f"[{trace_id}] Attempting request to {current_service} (attempt {attempt_count + 1})")
                result = request_func(current_service, request_data)
                
                # 记录成功的尝试
                attempt = FallbackAttempt(
                    timestamp=datetime.now(),
                    original_service=original_service,
                    fallback_service=current_service,
                    trigger_reason="Sequential fallback" if attempt_count > 0 else "Primary service",
                    success=True,
                    response_time=time.time() - attempt_start_time,
                    trace_id=trace_id
                )
                self.metrics.record_attempt(attempt)
                
                logger.info(f"[{trace_id}] Request successful on {current_service}")
                return result
                
            except Exception as e:
                response_time = time.time() - attempt_start_time
                error_info = {
                    'service': current_service,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'attempt': attempt_count + 1,
                    'response_time': response_time,
                    'trace_id': trace_id
                }
                attempt_errors.append(error_info)
                
                # 记录失败的尝试
                attempt = FallbackAttempt(
                    timestamp=datetime.now(),
                    original_service=original_service,
                    fallback_service=current_service,
                    trigger_reason="Sequential fallback" if attempt_count > 0 else "Primary service",
                    success=False,
                    response_time=response_time,
                    error=str(e),
                    trace_id=trace_id
                )
                self.metrics.record_attempt(attempt)
                
                logger.warning(f"[{trace_id}] Request to {current_service} failed: {e}")
        
        # 所有服务都失败，抛出复合错误
        self._raise_composite_error(original_service, attempt_errors, trace_id)
    
    def _build_fallback_chain(self, primary_service: str) -> List[str]:
        """构建按优先级排序的降级链"""
        # 获取所有已注册的服务端点
        all_services = list(self.endpoints.keys())
        
        # 确保主服务在第一位
        fallback_chain = [primary_service]
        
        # 添加其他服务，按优先级排序
        other_services = [s for s in all_services if s != primary_service]
        other_services.sort(key=lambda s: self.endpoints[s].priority)
        
        fallback_chain.extend(other_services)
        
        return fallback_chain
    
    def _raise_composite_error(self, original_service: str, attempt_errors: List[Dict], trace_id: str):
        """抛出包含所有失败信息的复合错误"""
        error_summary = {
            'trace_id': trace_id,
            'original_service': original_service,
            'total_attempts': len(attempt_errors),
            'failed_services': [err['service'] for err in attempt_errors],
            'errors': attempt_errors
        }
        
        # 构建详细的错误消息
        error_msg = f"[{trace_id}] All fallback attempts failed for service '{original_service}'. "
        error_msg += f"Attempted {len(attempt_errors)} services: "
        
        for i, err in enumerate(attempt_errors, 1):
            error_msg += f"\n  {i}. {err['service']}: {err['error_type']} - {err['error']}"
        
        # 创建复合错误异常
        composite_error = ServiceUnavailableError(error_msg)
        composite_error.error_details = error_summary
        
        logger.error(f"[{trace_id}] Composite error: {error_msg}")
        raise composite_error
    
    async def async_execute_with_fallback(self, 
                                         primary_service: str,
                                         request_func: Callable,
                                         request_data: Dict[str, Any],
                                         trace_id: str = None,
                                         timeout: float = None) -> Dict[str, Any]:
        """异步执行带降级的请求"""
        import asyncio
        
        start_time = time.time()
        original_service = primary_service
        
        # 生成trace_id如果没有提供
        if trace_id is None:
            trace_id = f"async_fallback_{int(time.time() * 1000)}_{hash(primary_service) % 10000}"
        
        # 获取完整的降级链（按优先级顺序）
        fallback_chain = self._build_fallback_chain(primary_service)
        
        # 记录所有尝试的错误信息
        attempt_errors = []
        
        for attempt_count, current_service in enumerate(fallback_chain):
            attempt_start_time = time.time()
            
            try:
                # 检查服务是否可用
                if not self.decision_engine.health_monitor.is_service_available(current_service):
                    error_msg = f"Service {current_service} is not available"
                    attempt_errors.append({
                        'service': current_service,
                        'error': error_msg,
                        'attempt': attempt_count + 1,
                        'trace_id': trace_id
                    })
                    continue
                
                # 执行异步请求
                logger.info(f"[{trace_id}] Async attempting request to {current_service} (attempt {attempt_count + 1})")
                
                if timeout:
                    result = await asyncio.wait_for(
                        request_func(current_service, request_data), 
                        timeout=timeout
                    )
                else:
                    result = await request_func(current_service, request_data)
                
                # 记录成功的尝试
                attempt = FallbackAttempt(
                    timestamp=datetime.now(),
                    original_service=original_service,
                    fallback_service=current_service,
                    trigger_reason="Async sequential fallback" if attempt_count > 0 else "Primary async service",
                    success=True,
                    response_time=time.time() - attempt_start_time,
                    trace_id=trace_id
                )
                self.metrics.record_attempt(attempt)
                
                logger.info(f"[{trace_id}] Async request successful on {current_service}")
                return result
                
            except Exception as e:
                response_time = time.time() - attempt_start_time
                error_info = {
                    'service': current_service,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'attempt': attempt_count + 1,
                    'response_time': response_time,
                    'trace_id': trace_id
                }
                attempt_errors.append(error_info)
                
                # 记录失败的尝试
                attempt = FallbackAttempt(
                    timestamp=datetime.now(),
                    original_service=original_service,
                    fallback_service=current_service,
                    trigger_reason="Async sequential fallback" if attempt_count > 0 else "Primary async service",
                    success=False,
                    response_time=response_time,
                    error=str(e),
                    trace_id=trace_id
                )
                self.metrics.record_attempt(attempt)
                
                logger.warning(f"[{trace_id}] Async request to {current_service} failed: {e}")
        
        # 所有服务都失败，抛出复合错误
        self._raise_composite_error(original_service, attempt_errors, trace_id)
    
    def execute_cascading_fallback(self, 
                                  primary_service: str,
                                  request_func: Callable,
                                  request_data: Dict[str, Any],
                                  trace_id: str = None,
                                  max_attempts: int = 3) -> Dict[str, Any]:
        """执行级联降级（限制最大尝试次数）"""
        start_time = time.time()
        original_service = primary_service
        
        # 生成trace_id如果没有提供
        if trace_id is None:
            trace_id = f"cascading_{int(time.time() * 1000)}_{hash(primary_service) % 10000}"
        
        # 获取降级链，但限制最大尝试次数
        fallback_chain = self._build_fallback_chain(primary_service)[:max_attempts]
        
        # 记录所有尝试的错误信息
        attempt_errors = []
        
        for attempt_count, current_service in enumerate(fallback_chain):
            attempt_start_time = time.time()
            
            try:
                # 检查服务是否可用
                if not self.decision_engine.health_monitor.is_service_available(current_service):
                    error_msg = f"Service {current_service} is not available"
                    attempt_errors.append({
                        'service': current_service,
                        'error': error_msg,
                        'attempt': attempt_count + 1,
                        'trace_id': trace_id
                    })
                    continue
                
                # 执行请求
                logger.info(f"[{trace_id}] Cascading attempt {attempt_count + 1}/{max_attempts} to {current_service}")
                result = request_func(current_service, request_data)
                
                # 记录成功的尝试
                attempt = FallbackAttempt(
                    timestamp=datetime.now(),
                    original_service=original_service,
                    fallback_service=current_service,
                    trigger_reason=f"Cascading fallback {attempt_count + 1}/{max_attempts}" if attempt_count > 0 else "Primary cascading service",
                    success=True,
                    response_time=time.time() - attempt_start_time,
                    trace_id=trace_id
                )
                self.metrics.record_attempt(attempt)
                
                logger.info(f"[{trace_id}] Cascading request successful on {current_service}")
                return result
                
            except Exception as e:
                response_time = time.time() - attempt_start_time
                error_info = {
                    'service': current_service,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'attempt': attempt_count + 1,
                    'response_time': response_time,
                    'trace_id': trace_id
                }
                attempt_errors.append(error_info)
                
                # 记录失败的尝试
                attempt = FallbackAttempt(
                    timestamp=datetime.now(),
                    original_service=original_service,
                    fallback_service=current_service,
                    trigger_reason=f"Cascading fallback {attempt_count + 1}/{max_attempts}" if attempt_count > 0 else "Primary cascading service",
                    success=False,
                    response_time=response_time,
                    error=str(e),
                    trace_id=trace_id
                )
                self.metrics.record_attempt(attempt)
                
                logger.warning(f"[{trace_id}] Cascading request to {current_service} failed: {e}")
        
        # 所有尝试都失败，抛出复合错误
        self._raise_composite_error(original_service, attempt_errors, trace_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取降级指标"""
        return {
            'fallback_statistics': self.metrics.get_fallback_statistics(),
            'service_metrics': {
                name: self.metrics.get_service_metrics(name)
                for name in self.endpoints.keys()
            }
        }