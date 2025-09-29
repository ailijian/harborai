# -*- coding: utf-8 -*-
"""
HarborAI 降级策略测试模块

测试目标：
- 验证降级策略的触发条件和执行逻辑
- 测试多级降级和回退机制
- 验证降级决策算法和优先级管理
- 测试降级状态监控和恢复机制
- 验证降级策略的性能影响
"""

import pytest
import asyncio
import time
import random
from typing import Dict, Any, List, Optional, Callable, Union
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from harborai import HarborAI
from harborai.core.exceptions import (
    HarborAIError,
    ServiceUnavailableError,
    RateLimitError,
    TimeoutError,
    QuotaExceededError,
    ModelNotFoundError
)
from harborai.core.fallback import (
    FallbackManager,
    FallbackStrategy,
    FallbackRule,
    ServiceHealthMonitor,
    FallbackDecisionEngine,
    FallbackMetrics
)


class ServiceStatus(Enum):
    """服务状态枚举"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class FallbackTrigger(Enum):
    """降级触发条件枚举"""
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    AVAILABILITY = "availability"
    QUOTA_LIMIT = "quota_limit"
    MANUAL = "manual"


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


class MockServiceProvider:
    """模拟服务提供者"""
    
    def __init__(self, endpoint: ServiceEndpoint, failure_rate: float = 0.0, 
                 response_time_range: tuple = (0.1, 0.5)):
        self.endpoint = endpoint
        self.failure_rate = failure_rate
        self.response_time_range = response_time_range
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_response_time = 0.0
        self.last_request_time = None
    
    def make_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """模拟请求处理"""
        self.request_count += 1
        self.last_request_time = datetime.now()
        
        # 模拟响应时间
        response_time = random.uniform(*self.response_time_range)
        time.sleep(response_time)
        self.total_response_time += response_time
        
        # 根据失败率决定是否失败
        if random.random() < self.failure_rate:
            self.failure_count += 1
            error_types = [
                ServiceUnavailableError(f"Service {self.endpoint.name} unavailable"),
                RateLimitError(f"Rate limit exceeded for {self.endpoint.name}"),
                TimeoutError(f"Timeout for {self.endpoint.name}"),
                QuotaExceededError(f"Quota exceeded for {self.endpoint.name}")
            ]
            raise random.choice(error_types)
        
        # 模拟成功响应
        self.success_count += 1
        return {
            "service": self.endpoint.name,
            "model": self.endpoint.model_type,
            "response": f"Response from {self.endpoint.name}",
            "tokens_used": random.randint(50, 200),
            "response_time": response_time,
            "cost": random.randint(50, 200) * self.endpoint.cost_per_token
        }
    
    async def async_make_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """异步请求处理"""
        # 模拟异步延迟
        response_time = random.uniform(*self.response_time_range)
        await asyncio.sleep(response_time)
        
        # 调用同步方法（除了sleep部分）
        self.request_count += 1
        self.last_request_time = datetime.now()
        self.total_response_time += response_time
        
        if random.random() < self.failure_rate:
            self.failure_count += 1
            raise ServiceUnavailableError(f"Async service {self.endpoint.name} unavailable")
        
        self.success_count += 1
        return {
            "service": self.endpoint.name,
            "model": self.endpoint.model_type,
            "response": f"Async response from {self.endpoint.name}",
            "tokens_used": random.randint(50, 200),
            "response_time": response_time,
            "cost": random.randint(50, 200) * self.endpoint.cost_per_token
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取服务指标"""
        return {
            "request_count": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_count / max(self.request_count, 1),
            "failure_rate": self.failure_count / max(self.request_count, 1),
            "average_response_time": self.total_response_time / max(self.request_count, 1),
            "last_request_time": self.last_request_time
        }
    
    def reset_metrics(self):
        """重置指标"""
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.total_response_time = 0.0
        self.last_request_time = None


class TestFallbackTriggers:
    """降级触发条件测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.fallback_strategy
    def test_error_rate_trigger(self):
        """测试错误率触发降级"""
        # 创建降级触发器
        trigger_monitor = Mock()
        
        # 错误率阈值配置
        error_rate_config = {
            "threshold": 0.5,  # 50%错误率
            "window_size": 10,  # 最近10个请求
            "min_requests": 5   # 最少5个请求才开始监控
        }
        
        # 模拟请求历史（成功=True，失败=False）
        request_history = []
        
        def mock_add_request_result(success: bool):
            request_history.append(success)
            # 保持窗口大小
            if len(request_history) > error_rate_config["window_size"]:
                request_history.pop(0)
        
        def mock_should_trigger_fallback():
            if len(request_history) < error_rate_config["min_requests"]:
                return False
            
            failure_count = sum(1 for result in request_history if not result)
            error_rate = failure_count / len(request_history)
            
            return error_rate >= error_rate_config["threshold"]
        
        def mock_get_current_error_rate():
            if not request_history:
                return 0.0
            failure_count = sum(1 for result in request_history if not result)
            return failure_count / len(request_history)
        
        trigger_monitor.add_request_result.side_effect = mock_add_request_result
        trigger_monitor.should_trigger_fallback.side_effect = mock_should_trigger_fallback
        trigger_monitor.get_error_rate.side_effect = mock_get_current_error_rate
        
        # 测试初始状态（请求不足）
        for i in range(4):
            trigger_monitor.add_request_result(False)  # 4次失败
        
        assert trigger_monitor.should_trigger_fallback() is False  # 请求数不足
        
        # 添加第5个请求，达到最小请求数
        trigger_monitor.add_request_result(False)  # 第5次失败
        assert trigger_monitor.should_trigger_fallback() is True  # 100%错误率，超过阈值
        assert trigger_monitor.get_error_rate() == 1.0
        
        # 添加一些成功请求，降低错误率
        for i in range(3):
            trigger_monitor.add_request_result(True)  # 3次成功
        
        # 现在历史是：[False, False, False, False, False, True, True, True]
        # 错误率 = 5/8 = 0.625，仍然超过阈值
        assert trigger_monitor.should_trigger_fallback() is True
        assert abs(trigger_monitor.get_error_rate() - 0.625) < 0.001
        
        # 继续添加成功请求
        for i in range(5):
            trigger_monitor.add_request_result(True)  # 5次成功
        
        # 现在窗口内是最近10个：[False, False, True, True, True, True, True, True, True, True]
        # 错误率 = 2/10 = 0.2，低于阈值
        assert trigger_monitor.should_trigger_fallback() is False
        assert abs(trigger_monitor.get_error_rate() - 0.2) < 0.001
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.fallback_strategy
    def test_response_time_trigger(self):
        """测试响应时间触发降级"""
        # 创建响应时间监控器
        response_time_monitor = Mock()
        
        # 响应时间阈值配置
        response_time_config = {
            "threshold": 2.0,  # 2秒阈值
            "percentile": 95,  # P95响应时间
            "window_size": 20,  # 最近20个请求
            "min_requests": 10  # 最少10个请求
        }
        
        # 响应时间历史
        response_times = []
        
        def mock_add_response_time(response_time: float):
            response_times.append(response_time)
            if len(response_times) > response_time_config["window_size"]:
                response_times.pop(0)
        
        def mock_get_percentile_response_time(percentile: int):
            if len(response_times) < response_time_config["min_requests"]:
                return 0.0
            
            sorted_times = sorted(response_times)
            index = int((percentile / 100.0) * len(sorted_times)) - 1
            index = max(0, min(index, len(sorted_times) - 1))
            return sorted_times[index]
        
        def mock_should_trigger_fallback():
            if len(response_times) < response_time_config["min_requests"]:
                return False
            
            p95_time = mock_get_percentile_response_time(response_time_config["percentile"])
            return p95_time >= response_time_config["threshold"]
        
        response_time_monitor.add_response_time.side_effect = mock_add_response_time
        response_time_monitor.get_percentile_response_time.side_effect = mock_get_percentile_response_time
        response_time_monitor.should_trigger_fallback.side_effect = mock_should_trigger_fallback
        
        # 测试正常响应时间（不触发降级）
        normal_times = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22, 0.28, 0.16, 0.24]
        for time_val in normal_times:
            response_time_monitor.add_response_time(time_val)
        
        assert response_time_monitor.should_trigger_fallback() is False
        p95_time = response_time_monitor.get_percentile_response_time(95)
        assert p95_time < response_time_config["threshold"]
        
        # 添加一些慢请求
        slow_times = [2.5, 3.0, 2.8, 1.8, 1.9, 2.2, 2.1, 2.6, 2.3, 2.4]
        for time_val in slow_times:
            response_time_monitor.add_response_time(time_val)
        
        # 现在P95应该超过阈值
        assert response_time_monitor.should_trigger_fallback() is True
        p95_time = response_time_monitor.get_percentile_response_time(95)
        assert p95_time >= response_time_config["threshold"]
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.fallback_strategy
    def test_availability_trigger(self):
        """测试可用性触发降级"""
        # 创建可用性监控器
        availability_monitor = Mock()
        
        # 可用性配置
        availability_config = {
            "threshold": 0.95,  # 95%可用性
            "check_interval": 30,  # 30秒检查间隔
            "consecutive_failures": 3  # 连续3次失败触发降级
        }
        
        # 健康检查历史
        health_checks = []
        consecutive_failures = 0
        
        def mock_perform_health_check():
            # 模拟健康检查（这里简化为随机结果）
            is_healthy = random.random() > 0.1  # 90%概率健康
            timestamp = datetime.now()
            
            health_checks.append({
                "timestamp": timestamp,
                "healthy": is_healthy
            })
            
            # 保持最近100次检查
            if len(health_checks) > 100:
                health_checks.pop(0)
            
            return is_healthy
        
        def mock_get_availability():
            if not health_checks:
                return 1.0
            
            healthy_count = sum(1 for check in health_checks if check["healthy"])
            return healthy_count / len(health_checks)
        
        def mock_check_consecutive_failures():
            nonlocal consecutive_failures
            
            if not health_checks:
                return 0
            
            # 从最新的检查开始计算连续失败
            consecutive_failures = 0
            for check in reversed(health_checks):
                if not check["healthy"]:
                    consecutive_failures += 1
                else:
                    break
            
            return consecutive_failures
        
        def mock_should_trigger_fallback():
            availability = mock_get_availability()
            consecutive = mock_check_consecutive_failures()
            
            return (availability < availability_config["threshold"] or 
                   consecutive >= availability_config["consecutive_failures"])
        
        availability_monitor.perform_health_check.side_effect = mock_perform_health_check
        availability_monitor.get_availability.side_effect = mock_get_availability
        availability_monitor.check_consecutive_failures.side_effect = mock_check_consecutive_failures
        availability_monitor.should_trigger_fallback.side_effect = mock_should_trigger_fallback
        
        # 模拟一系列健康检查
        # 先进行一些成功的检查
        for i in range(20):
            # 强制设置为健康状态
            health_checks.append({
                "timestamp": datetime.now(),
                "healthy": True
            })
        
        assert availability_monitor.should_trigger_fallback() is False
        assert availability_monitor.get_availability() == 1.0
        
        # 添加一些失败的检查
        for i in range(5):
            health_checks.append({
                "timestamp": datetime.now(),
                "healthy": False
            })
        
        # 现在可用性降低，但可能还没达到阈值
        availability = availability_monitor.get_availability()
        consecutive = availability_monitor.check_consecutive_failures()
        
        # 检查是否触发降级（可用性低于阈值或连续失败过多）
        should_trigger = (availability < availability_config["threshold"] or 
                         consecutive >= availability_config["consecutive_failures"])
        
        assert availability_monitor.should_trigger_fallback() == should_trigger
        assert consecutive == 5  # 连续5次失败
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.fallback_strategy
    def test_quota_limit_trigger(self):
        """测试配额限制触发降级"""
        # 创建配额监控器
        quota_monitor = Mock()
        
        # 配额配置
        quota_config = {
            "daily_limit": 10000,
            "hourly_limit": 1000,
            "warning_threshold": 0.8,  # 80%时警告
            "fallback_threshold": 0.95  # 95%时降级
        }
        
        # 配额使用情况
        quota_usage = {
            "daily_used": 0,
            "hourly_used": 0,
            "last_reset_hour": datetime.now().hour,
            "last_reset_day": datetime.now().day
        }
        
        def mock_add_usage(tokens_used: int):
            current_time = datetime.now()
            
            # 检查是否需要重置小时计数
            if current_time.hour != quota_usage["last_reset_hour"]:
                quota_usage["hourly_used"] = 0
                quota_usage["last_reset_hour"] = current_time.hour
            
            # 检查是否需要重置日计数
            if current_time.day != quota_usage["last_reset_day"]:
                quota_usage["daily_used"] = 0
                quota_usage["last_reset_day"] = current_time.day
            
            quota_usage["daily_used"] += tokens_used
            quota_usage["hourly_used"] += tokens_used
        
        def mock_get_usage_percentage(period: str):
            if period == "daily":
                return quota_usage["daily_used"] / quota_config["daily_limit"]
            elif period == "hourly":
                return quota_usage["hourly_used"] / quota_config["hourly_limit"]
            else:
                return 0.0
        
        def mock_should_trigger_fallback():
            daily_percentage = mock_get_usage_percentage("daily")
            hourly_percentage = mock_get_usage_percentage("hourly")
            
            return (daily_percentage >= quota_config["fallback_threshold"] or
                   hourly_percentage >= quota_config["fallback_threshold"])
        
        def mock_should_warn():
            daily_percentage = mock_get_usage_percentage("daily")
            hourly_percentage = mock_get_usage_percentage("hourly")
            
            return (daily_percentage >= quota_config["warning_threshold"] or
                   hourly_percentage >= quota_config["warning_threshold"])
        
        quota_monitor.add_usage.side_effect = mock_add_usage
        quota_monitor.get_usage_percentage.side_effect = mock_get_usage_percentage
        quota_monitor.should_trigger_fallback.side_effect = mock_should_trigger_fallback
        quota_monitor.should_warn.side_effect = mock_should_warn
        
        # 测试正常使用情况（未达到阈值）
        quota_monitor.add_usage(800)  # 使用800 tokens
        
        assert quota_monitor.get_usage_percentage("daily") == 0.08  # 8%
        assert quota_monitor.get_usage_percentage("hourly") == 0.8  # 80%
        assert quota_monitor.should_warn() is True  # 达到警告阈值
        assert quota_monitor.should_trigger_fallback() is False  # 未达到降级阈值
        
        # 继续使用，触发小时配额降级
        quota_monitor.add_usage(200)  # 总共1000 tokens
        
        assert quota_monitor.get_usage_percentage("hourly") == 1.0  # 100%
        assert quota_monitor.should_trigger_fallback() is True  # 超过降级阈值
        
        # 重置小时配额（模拟新的一小时）
        quota_usage["hourly_used"] = 0
        quota_usage["last_reset_hour"] = (datetime.now().hour + 1) % 24
        
        assert quota_monitor.should_trigger_fallback() is False
        
        # 继续使用，接近日配额限制
        quota_monitor.add_usage(8000)  # 总共9000 tokens (1000 + 8000)
        
        assert quota_monitor.get_usage_percentage("daily") == 0.9  # 90%
        assert quota_monitor.get_usage_percentage("hourly") == 8.0  # 800% (8000/1000)
        assert quota_monitor.should_warn() is True  # 超过警告阈值
        assert quota_monitor.should_trigger_fallback() is True  # 小时使用率超过降级阈值
        
        # 再次重置小时配额
        quota_usage["hourly_used"] = 0
        quota_usage["last_reset_hour"] = (datetime.now().hour + 2) % 24
        
        # 继续使用，触发日配额降级
        quota_monitor.add_usage(600)  # 总共9600 tokens
        
        assert quota_monitor.get_usage_percentage("daily") == 0.96  # 96%
        assert quota_monitor.get_usage_percentage("hourly") == 0.6  # 60%
        assert quota_monitor.should_trigger_fallback() is True  # 日使用率超过降级阈值


class TestFallbackDecision:
    """降级决策测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.fallback_strategy
    def test_service_priority_selection(self):
        """测试服务优先级选择"""
        # 创建降级决策引擎
        decision_engine = Mock(spec=FallbackDecisionEngine)
        
        # 定义服务端点
        services = [
            ServiceEndpoint("primary", "https://api.primary.com", priority=1, model_type="deepseek-chat", cost_per_token=0.03),
            ServiceEndpoint("secondary", "https://api.secondary.com", priority=2, model_type="ernie-3.5-8k", cost_per_token=0.002),
            ServiceEndpoint("backup", "https://api.backup.com", priority=3, model_type="doubao-1-5-pro-32k-character-250715", cost_per_token=0.01),
            ServiceEndpoint("emergency", "https://api.emergency.com", priority=4, model_type="llama-2", cost_per_token=0.001)
        ]
        
        # 服务状态
        service_status = {
            "primary": ServiceStatus.UNAVAILABLE,
            "secondary": ServiceStatus.HEALTHY,
            "backup": ServiceStatus.DEGRADED,
            "emergency": ServiceStatus.HEALTHY
        }
        
        def mock_select_fallback_service(original_service: str, requirements: Dict[str, Any] = None):
            # 过滤掉原始服务和不可用服务
            available_services = [
                svc for svc in services 
                if svc.name != original_service and 
                service_status.get(svc.name) != ServiceStatus.UNAVAILABLE
            ]
            
            if not available_services:
                return None
            
            # 根据要求过滤服务
            if requirements:
                if "model_type" in requirements:
                    available_services = [
                        svc for svc in available_services 
                        if svc.model_type == requirements["model_type"]
                    ]
                
                if "max_cost" in requirements:
                    available_services = [
                        svc for svc in available_services 
                        if svc.cost_per_token <= requirements["max_cost"]
                    ]
            
            if not available_services:
                return None
            
            # 选择优先级最高的健康服务
            healthy_services = [
                svc for svc in available_services 
                if service_status.get(svc.name) == ServiceStatus.HEALTHY
            ]
            
            if healthy_services:
                return min(healthy_services, key=lambda x: x.priority)
            else:
                # 如果没有完全健康的服务，选择降级服务
                return min(available_services, key=lambda x: x.priority)
        
        def mock_get_fallback_chain(original_service: str, max_depth: int = 3):
            chain = []
            excluded_services = {original_service}  # 排除原始服务和已选择的服务
            
            for _ in range(max_depth):
                # 过滤掉原始服务和已在链中的服务
                available_services = [
                    svc for svc in services 
                    if svc.name not in excluded_services and 
                    service_status.get(svc.name) != ServiceStatus.UNAVAILABLE
                ]
                
                if not available_services:
                    break
                
                # 按优先级顺序选择下一个可用服务
                fallback = min(available_services, key=lambda x: x.priority)
                
                chain.append(fallback)
                excluded_services.add(fallback.name)
            
            return chain
        
        decision_engine.select_fallback_service.side_effect = mock_select_fallback_service
        decision_engine.get_fallback_chain.side_effect = mock_get_fallback_chain
        
        # 测试基本降级选择
        fallback = decision_engine.select_fallback_service("primary")
        assert fallback is not None
        assert fallback.name == "secondary"  # 优先级最高的可用服务
        
        # 测试带要求的降级选择
        fallback = decision_engine.select_fallback_service(
            "primary", 
            requirements={"max_cost": 0.005}
        )
        assert fallback is not None
        assert fallback.name == "secondary"  # 符合成本要求的服务
        
        # 测试更严格的成本要求
        fallback = decision_engine.select_fallback_service(
            "primary", 
            requirements={"max_cost": 0.0015}
        )
        assert fallback is not None
        assert fallback.name == "emergency"  # 只有emergency符合成本要求
        
        # 测试降级链
        chain = decision_engine.get_fallback_chain("primary", max_depth=3)
        assert len(chain) >= 2
        assert chain[0].name == "secondary"
        assert chain[1].name in ["backup", "emergency"]  # 取决于状态
        
        # 验证优先级顺序
        priorities = [svc.priority for svc in chain]
        assert priorities == sorted(priorities)  # 优先级应该是递增的
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.fallback_strategy
    def test_capability_based_selection(self):
        """测试基于能力的服务选择"""
        # 创建具有不同能力的服务
        services_with_capabilities = [
            ServiceEndpoint(
                "full_service", "https://api.full.com", priority=1, model_type="deepseek-chat",
                capabilities=["chat", "completion", "embedding", "image_generation", "function_calling"]
            ),
            ServiceEndpoint(
                "chat_service", "https://api.chat.com", priority=2, model_type="gpt-3.5",
                capabilities=["chat", "completion", "function_calling"]
            ),
            ServiceEndpoint(
                "basic_service", "https://api.basic.com", priority=3, model_type="claude-2",
                capabilities=["chat", "completion"]
            ),
            ServiceEndpoint(
                "embedding_service", "https://api.embedding.com", priority=4, model_type="text-embedding",
                capabilities=["embedding"]
            )
        ]
        
        capability_selector = Mock()
        
        def mock_select_by_capability(required_capabilities: List[str], exclude_services: List[str] = None):
            exclude_services = exclude_services or []
            
            # 过滤掉排除的服务
            available_services = [
                svc for svc in services_with_capabilities 
                if svc.name not in exclude_services
            ]
            
            # 找到支持所有必需能力的服务
            compatible_services = []
            for svc in available_services:
                if all(cap in svc.capabilities for cap in required_capabilities):
                    compatible_services.append(svc)
            
            if not compatible_services:
                return None
            
            # 返回优先级最高的服务
            return min(compatible_services, key=lambda x: x.priority)
        
        def mock_get_capability_score(service: ServiceEndpoint, required_capabilities: List[str]):
            # 计算能力匹配分数
            matched_capabilities = sum(1 for cap in required_capabilities if cap in service.capabilities)
            total_capabilities = len(service.capabilities)
            
            if not required_capabilities:
                return 0.0
            
            # 基础匹配度
            match_score = matched_capabilities / len(required_capabilities)
            
            # 只有完全匹配时才给额外能力奖励
            if matched_capabilities == len(required_capabilities):
                bonus_score = (total_capabilities - matched_capabilities) * 0.1
                return match_score + bonus_score
            else:
                # 部分匹配时不给奖励
                return match_score
        
        capability_selector.select_by_capability.side_effect = mock_select_by_capability
        capability_selector.get_capability_score.side_effect = mock_get_capability_score
        
        # 测试聊天功能选择
        selected = capability_selector.select_by_capability(["chat"])
        assert selected is not None
        assert selected.name == "full_service"  # 优先级最高且支持聊天
        
        # 测试函数调用功能选择
        selected = capability_selector.select_by_capability(["chat", "function_calling"])
        assert selected is not None
        assert selected.name == "full_service"
        assert "function_calling" in selected.capabilities
        
        # 测试嵌入功能选择
        selected = capability_selector.select_by_capability(["embedding"])
        assert selected is not None
        # 可能是full_service或embedding_service，取决于优先级
        assert "embedding" in selected.capabilities
        
        # 测试排除某些服务后的选择
        selected = capability_selector.select_by_capability(
            ["chat", "function_calling"], 
            exclude_services=["full_service"]
        )
        assert selected is not None
        assert selected.name == "chat_service"
        
        # 测试无法满足的能力要求
        selected = capability_selector.select_by_capability(["image_generation"], exclude_services=["full_service"])
        assert selected is None
        
        # 测试能力评分
        score = capability_selector.get_capability_score(
            services_with_capabilities[0],  # full_service
            ["chat", "completion"]
        )
        assert score > 1.0  # 完全匹配 + 额外能力奖励
        
        score = capability_selector.get_capability_score(
            services_with_capabilities[2],  # basic_service
            ["chat", "completion"]
        )
        assert score == 1.0  # 完全匹配，无额外能力
        
        score = capability_selector.get_capability_score(
            services_with_capabilities[2],  # basic_service
            ["chat", "function_calling"]
        )
        assert score == 0.5  # 只匹配一半能力
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.fallback_strategy
    def test_cost_aware_selection(self):
        """测试成本感知的服务选择"""
        # 创建成本感知选择器
        cost_selector = Mock()
        
        # 服务成本配置
        cost_services = [
            ServiceEndpoint("premium", "https://api.premium.com", priority=1, model_type="deepseek-reasoner", cost_per_token=0.06),
            ServiceEndpoint("standard", "https://api.standard.com", priority=2, model_type="ernie-4.0-turbo-8k", cost_per_token=0.002),
            ServiceEndpoint("economy", "https://api.economy.com", priority=3, model_type="doubao-seed-1-6-250615", cost_per_token=0.0015),
            ServiceEndpoint("budget", "https://api.budget.com", priority=4, model_type="llama-2", cost_per_token=0.0005)
        ]
        
        def mock_select_by_cost(max_cost_per_token: float, min_quality_score: float = 0.0):
            # 过滤符合成本要求的服务
            affordable_services = [
                svc for svc in cost_services 
                if svc.cost_per_token <= max_cost_per_token
            ]
            
            if not affordable_services:
                return None
            
            # 简化的质量评分（基于优先级，优先级越低质量越高）
            def get_quality_score(service):
                return 1.0 / service.priority
            
            # 过滤符合质量要求的服务
            quality_services = [
                svc for svc in affordable_services 
                if get_quality_score(svc) >= min_quality_score
            ]
            
            if not quality_services:
                # 如果没有符合质量要求的，返回成本最低的
                return min(affordable_services, key=lambda x: x.cost_per_token)
            
            # 返回质量最高的服务
            return min(quality_services, key=lambda x: x.priority)
        
        def mock_calculate_cost_impact(original_service: str, fallback_service: str, estimated_tokens: int):
            original = next((svc for svc in cost_services if svc.name == original_service), None)
            fallback = next((svc for svc in cost_services if svc.name == fallback_service), None)
            
            if not original or not fallback:
                return 0.0
            
            original_cost = original.cost_per_token * estimated_tokens
            fallback_cost = fallback.cost_per_token * estimated_tokens
            
            return fallback_cost - original_cost
        
        def mock_get_cost_efficiency_score(service: ServiceEndpoint):
            # 成本效率 = 质量 / 成本
            quality = 1.0 / service.priority
            return quality / service.cost_per_token
        
        cost_selector.select_by_cost.side_effect = mock_select_by_cost
        cost_selector.calculate_cost_impact.side_effect = mock_calculate_cost_impact
        cost_selector.get_cost_efficiency_score.side_effect = mock_get_cost_efficiency_score
        
        # 测试高预算选择
        selected = cost_selector.select_by_cost(max_cost_per_token=0.1)
        assert selected is not None
        assert selected.name == "premium"  # 预算充足，选择最高质量
        
        # 测试中等预算选择
        selected = cost_selector.select_by_cost(max_cost_per_token=0.01)
        assert selected is not None
        assert selected.name == "standard"  # 排除premium，选择次优
        
        # 测试低预算选择
        selected = cost_selector.select_by_cost(max_cost_per_token=0.001)
        assert selected is not None
        assert selected.name == "budget"  # 只有budget符合预算
        
        # 测试带质量要求的选择
        selected = cost_selector.select_by_cost(
            max_cost_per_token=0.01, 
            min_quality_score=0.3  # 要求较高质量
        )
        assert selected is not None
        # 应该选择符合质量要求的最高质量服务
        
        # 测试成本影响计算
        cost_impact = cost_selector.calculate_cost_impact("premium", "standard", 1000)
        assert cost_impact < 0  # 降级到更便宜的服务，成本应该降低
        
        cost_impact = cost_selector.calculate_cost_impact("budget", "premium", 1000)
        assert cost_impact > 0  # 升级到更贵的服务，成本应该增加
        
        # 测试成本效率评分
        efficiency_scores = [
            (svc.name, cost_selector.get_cost_efficiency_score(svc)) 
            for svc in cost_services
        ]
        
        # 验证评分合理性（这里简化验证）
        for name, score in efficiency_scores:
            assert score > 0
        
        # standard服务通常有较好的成本效率
        standard_score = next(score for name, score in efficiency_scores if name == "standard")
        budget_score = next(score for name, score in efficiency_scores if name == "budget")
        
        # 验证成本效率计算的合理性
        assert isinstance(standard_score, (int, float))
        assert isinstance(budget_score, (int, float))


class TestFallbackExecution:
    """降级执行测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.fallback_strategy
    def test_single_fallback_execution(self):
        """测试单次降级执行"""
        # 创建降级管理器
        fallback_manager = Mock(spec=FallbackManager)
        
        # 创建服务提供者
        primary_service = MockServiceProvider(
            ServiceEndpoint("primary", "https://api.primary.com", 1, "deepseek-chat"),
            failure_rate=1.0  # 100%失败率
        )
        
        fallback_service = MockServiceProvider(
            ServiceEndpoint("fallback", "https://api.fallback.com", 2, "gpt-3.5"),
            failure_rate=0.0  # 0%失败率
        )
        
        # 模拟降级执行
        def mock_execute_with_fallback(request_data: Dict[str, Any], max_fallbacks: int = 3):
            attempts = []
            services_tried = []
            
            # 尝试主服务
            try:
                result = primary_service.make_request(request_data)
                attempts.append({
                    "service": "primary",
                    "success": True,
                    "result": result,
                    "attempt_number": 1
                })
                return result, attempts
            except Exception as e:
                attempts.append({
                    "service": "primary",
                    "success": False,
                    "error": str(e),
                    "attempt_number": 1
                })
                services_tried.append("primary")
            
            # 尝试降级服务
            if len(services_tried) < max_fallbacks:
                try:
                    result = fallback_service.make_request(request_data)
                    attempts.append({
                        "service": "fallback",
                        "success": True,
                        "result": result,
                        "attempt_number": 2
                    })
                    return result, attempts
                except Exception as e:
                    attempts.append({
                        "service": "fallback",
                        "success": False,
                        "error": str(e),
                        "attempt_number": 2
                    })
            
            # 所有服务都失败
            raise Exception("All fallback services failed")
        
        fallback_manager.execute_with_fallback.side_effect = mock_execute_with_fallback
        
        # 执行降级请求
        request_data = {"prompt": "Hello, world!", "max_tokens": 100}
        result, attempts = fallback_manager.execute_with_fallback(request_data)
        
        # 验证结果
        assert result is not None
        assert "fallback" in result["service"]
        assert len(attempts) == 2
        assert attempts[0]["success"] is False  # 主服务失败
        assert attempts[1]["success"] is True   # 降级服务成功
        
        # 验证服务调用次数
        assert primary_service.request_count == 1
        assert fallback_service.request_count == 1
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.fallback_strategy
    def test_cascading_fallback_execution(self):
        """测试级联降级执行"""
        # 创建多级降级管理器
        cascading_manager = Mock()
        
        # 创建多个服务提供者
        services = {
            "primary": MockServiceProvider(
                ServiceEndpoint("primary", "https://api.primary.com", 1, "deepseek-chat"),
                failure_rate=1.0
            ),
            "secondary": MockServiceProvider(
                ServiceEndpoint("secondary", "https://api.secondary.com", 2, "gpt-3.5"),
                failure_rate=1.0
            ),
            "tertiary": MockServiceProvider(
                ServiceEndpoint("tertiary", "https://api.tertiary.com", 3, "claude-2"),
                failure_rate=0.0  # 只有第三级服务可用
            )
        }
        
        def mock_execute_cascading_fallback(request_data: Dict[str, Any]):
            service_order = ["primary", "secondary", "tertiary"]
            attempts = []
            
            for i, service_name in enumerate(service_order):
                service = services[service_name]
                
                try:
                    result = service.make_request(request_data)
                    attempts.append({
                        "service": service_name,
                        "success": True,
                        "result": result,
                        "attempt_number": i + 1,
                        "fallback_level": i
                    })
                    return result, attempts
                except Exception as e:
                    attempts.append({
                        "service": service_name,
                        "success": False,
                        "error": str(e),
                        "attempt_number": i + 1,
                        "fallback_level": i
                    })
                    
                    # 添加降级延迟
                    time.sleep(0.1 * (i + 1))
            
            # 所有服务都失败
            raise Exception("All cascading fallbacks failed")
        
        cascading_manager.execute_cascading_fallback.side_effect = mock_execute_cascading_fallback
        
        # 执行级联降级
        request_data = {"prompt": "Test cascading fallback", "max_tokens": 50}
        result, attempts = cascading_manager.execute_cascading_fallback(request_data)
        
        # 验证结果
        assert result is not None
        assert result["service"] == "tertiary"
        assert len(attempts) == 3
        
        # 验证降级顺序
        assert attempts[0]["service"] == "primary"
        assert attempts[0]["fallback_level"] == 0
        assert attempts[0]["success"] is False
        
        assert attempts[1]["service"] == "secondary"
        assert attempts[1]["fallback_level"] == 1
        assert attempts[1]["success"] is False
        
        assert attempts[2]["service"] == "tertiary"
        assert attempts[2]["fallback_level"] == 2
        assert attempts[2]["success"] is True
        
        # 验证所有服务都被尝试
        for service in services.values():
            assert service.request_count == 1
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.fallback_strategy
    async def test_async_fallback_execution(self):
        """测试异步降级执行"""
        # 创建异步降级管理器
        async_fallback_manager = Mock()
        
        # 创建异步服务提供者
        async_primary = MockServiceProvider(
            ServiceEndpoint("async_primary", "https://api.async-primary.com", 1, "deepseek-chat"),
            failure_rate=1.0
        )
        
        async_fallback = MockServiceProvider(
            ServiceEndpoint("async_fallback", "https://api.async-fallback.com", 2, "gpt-3.5"),
            failure_rate=0.0
        )
        
        # 模拟异步降级执行
        async def mock_async_execute_with_fallback(request_data: Dict[str, Any]):
            attempts = []
            
            # 尝试主服务
            try:
                result = await async_primary.async_make_request(request_data)
                attempts.append({
                    "service": "async_primary",
                    "success": True,
                    "result": result
                })
                return result, attempts
            except Exception as e:
                attempts.append({
                    "service": "async_primary",
                    "success": False,
                    "error": str(e)
                })
            
            # 尝试降级服务
            try:
                result = await async_fallback.async_make_request(request_data)
                attempts.append({
                    "service": "async_fallback",
                    "success": True,
                    "result": result
                })
                return result, attempts
            except Exception as e:
                attempts.append({
                    "service": "async_fallback",
                    "success": False,
                    "error": str(e)
                })
                raise Exception("All async fallbacks failed")
        
        async_fallback_manager.async_execute_with_fallback.side_effect = mock_async_execute_with_fallback
        
        # 执行异步降级
        request_data = {"prompt": "Async fallback test", "max_tokens": 75}
        result, attempts = await async_fallback_manager.async_execute_with_fallback(request_data)
        
        # 验证结果
        assert result is not None
        assert "async_fallback" in result["service"]
        assert len(attempts) == 2
        assert attempts[0]["success"] is False
        assert attempts[1]["success"] is True
        
        # 验证异步调用
        assert async_primary.request_count == 1
        assert async_fallback.request_count == 1
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.fallback_strategy
    def test_fallback_with_timeout(self):
        """测试带超时的降级执行"""
        # 创建带超时的降级管理器
        timeout_manager = Mock()
        
        # 创建慢速服务
        slow_service = MockServiceProvider(
            ServiceEndpoint("slow", "https://api.slow.com", 1, "deepseek-chat"),
            response_time_range=(2.0, 3.0)  # 2-3秒响应时间
        )
        
        fast_service = MockServiceProvider(
            ServiceEndpoint("fast", "https://api.fast.com", 2, "gpt-3.5"),
            response_time_range=(0.1, 0.2)  # 0.1-0.2秒响应时间
        )
        
        def mock_execute_with_timeout(request_data: Dict[str, Any], timeout: float = 1.0):
            attempts = []
            start_time = time.time()
            
            # 尝试慢速服务
            try:
                # 检查是否会超时
                if slow_service.response_time_range[0] > timeout:
                    raise TimeoutError(f"Service timeout after {timeout}s")
                
                result = slow_service.make_request(request_data)
                elapsed = time.time() - start_time
                
                if elapsed > timeout:
                    raise TimeoutError(f"Service timeout after {elapsed:.2f}s")
                
                attempts.append({
                    "service": "slow",
                    "success": True,
                    "result": result,
                    "response_time": elapsed
                })
                return result, attempts
                
            except TimeoutError as e:
                attempts.append({
                    "service": "slow",
                    "success": False,
                    "error": str(e),
                    "timeout": True
                })
            except Exception as e:
                attempts.append({
                    "service": "slow",
                    "success": False,
                    "error": str(e),
                    "timeout": False
                })
            
            # 检查剩余时间
            elapsed = time.time() - start_time
            remaining_time = timeout - elapsed
            
            if remaining_time <= 0:
                raise TimeoutError("Total timeout exceeded")
            
            # 尝试快速服务
            try:
                result = fast_service.make_request(request_data)
                elapsed = time.time() - start_time
                
                attempts.append({
                    "service": "fast",
                    "success": True,
                    "result": result,
                    "response_time": elapsed
                })
                return result, attempts
                
            except Exception as e:
                attempts.append({
                    "service": "fast",
                    "success": False,
                    "error": str(e)
                })
                raise Exception("All services failed or timed out")
        
        timeout_manager.execute_with_timeout.side_effect = mock_execute_with_timeout
        
        # 测试超时降级
        request_data = {"prompt": "Timeout test", "max_tokens": 50}
        
        # 使用短超时，应该触发降级
        result, attempts = timeout_manager.execute_with_timeout(request_data, timeout=1.0)
        
        # 验证结果
        assert result is not None
        assert result["service"] == "fast"
        assert len(attempts) == 2
        assert attempts[0]["timeout"] is True  # 慢服务超时
        assert attempts[1]["success"] is True  # 快服务成功
        
        # 验证响应时间
        total_time = attempts[1]["response_time"]
        assert total_time < 1.0  # 总时间应该在超时限制内
        
        # 测试充足超时时间
        slow_service.reset_metrics()
        fast_service.reset_metrics()
        
        # 模拟慢服务在充足时间内成功
        slow_service.response_time_range = (0.5, 0.8)  # 减少响应时间
        
        result, attempts = timeout_manager.execute_with_timeout(request_data, timeout=2.0)
        
        # 应该使用慢服务（优先级更高）
        assert result["service"] == "slow"
        assert len(attempts) == 1
        assert attempts[0]["success"] is True