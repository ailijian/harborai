#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 降级策略模块测试

测试故障转移、健康监控和降级决策功能。
"""

import asyncio
import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from collections import deque

from harborai.core.fallback import (
    FallbackTrigger,
    ServiceStatus,
    FallbackRule,
    ServiceEndpoint,
    FallbackAttempt,
    FallbackMetrics,
    ServiceHealthMonitor,
    FallbackDecisionEngine,
    FallbackStrategy,
    PriorityFallbackStrategy,
    CostOptimizedFallbackStrategy,
    FallbackManager
)
from harborai.core.exceptions import (
    ServiceUnavailableError,
    RateLimitError,
    TimeoutError,
    QuotaExceededError
)


class TestFallbackTrigger:
    """测试降级触发条件枚举"""
    
    def test_trigger_values(self):
        """测试触发条件值"""
        assert FallbackTrigger.ERROR_RATE.value == "error_rate"
        assert FallbackTrigger.RESPONSE_TIME.value == "response_time"
        assert FallbackTrigger.AVAILABILITY.value == "availability"
        assert FallbackTrigger.QUOTA_LIMIT.value == "quota_limit"
        assert FallbackTrigger.MANUAL.value == "manual"


class TestServiceStatus:
    """测试服务状态枚举"""
    
    def test_status_values(self):
        """测试状态值"""
        assert ServiceStatus.HEALTHY.value == "healthy"
        assert ServiceStatus.DEGRADED.value == "degraded"
        assert ServiceStatus.UNAVAILABLE.value == "unavailable"
        assert ServiceStatus.UNKNOWN.value == "unknown"


class TestFallbackRule:
    """测试降级规则配置"""
    
    def test_rule_creation(self):
        """测试规则创建"""
        rule = FallbackRule(
            trigger=FallbackTrigger.ERROR_RATE,
            threshold=0.5,
            window_size=20,
            min_requests=10,
            cooldown_period=120,
            enabled=True,
            priority=2
        )
        
        assert rule.trigger == FallbackTrigger.ERROR_RATE
        assert rule.threshold == 0.5
        assert rule.window_size == 20
        assert rule.min_requests == 10
        assert rule.cooldown_period == 120
        assert rule.enabled is True
        assert rule.priority == 2
    
    def test_rule_defaults(self):
        """测试规则默认值"""
        rule = FallbackRule(
            trigger=FallbackTrigger.RESPONSE_TIME,
            threshold=5.0
        )
        
        assert rule.window_size == 10
        assert rule.min_requests == 5
        assert rule.cooldown_period == 60
        assert rule.enabled is True
        assert rule.priority == 1


class TestServiceEndpoint:
    """测试服务端点配置"""
    
    def test_endpoint_creation(self):
        """测试端点创建"""
        endpoint = ServiceEndpoint(
            name="test_service",
            url="https://api.test.com",
            priority=1,
            model_type="gpt-3.5",
            capabilities=["chat", "completion"],
            cost_per_token=0.002,
            max_tokens=8192,
            rate_limit=200,
            timeout=45.0
        )
        
        assert endpoint.name == "test_service"
        assert endpoint.url == "https://api.test.com"
        assert endpoint.priority == 1
        assert endpoint.model_type == "gpt-3.5"
        assert endpoint.capabilities == ["chat", "completion"]
        assert endpoint.cost_per_token == 0.002
        assert endpoint.max_tokens == 8192
        assert endpoint.rate_limit == 200
        assert endpoint.timeout == 45.0
        assert endpoint.status == ServiceStatus.HEALTHY
    
    def test_endpoint_defaults(self):
        """测试端点默认值"""
        endpoint = ServiceEndpoint(
            name="basic_service",
            url="https://api.basic.com",
            priority=1,
            model_type="basic"
        )
        
        assert endpoint.capabilities == []
        assert endpoint.cost_per_token == 0.0
        assert endpoint.max_tokens == 4096
        assert endpoint.rate_limit == 100
        assert endpoint.timeout == 30.0
        assert endpoint.status == ServiceStatus.HEALTHY
        assert endpoint.last_check is None
        assert endpoint.health_check_url is None
        assert endpoint.retry_count == 3
        assert endpoint.circuit_breaker_threshold == 5
        assert endpoint.circuit_breaker_timeout == 60


class TestFallbackAttempt:
    """测试降级尝试记录"""
    
    def test_attempt_creation(self):
        """测试尝试记录创建"""
        timestamp = datetime.now()
        attempt = FallbackAttempt(
            timestamp=timestamp,
            original_service="primary",
            fallback_service="secondary",
            trigger_reason="Service unavailable",
            success=True,
            response_time=1.5,
            error=None,
            cost_impact=0.1,
            trace_id="trace_123"
        )
        
        assert attempt.timestamp == timestamp
        assert attempt.original_service == "primary"
        assert attempt.fallback_service == "secondary"
        assert attempt.trigger_reason == "Service unavailable"
        assert attempt.success is True
        assert attempt.response_time == 1.5
        assert attempt.error is None
        assert attempt.cost_impact == 0.1
        assert attempt.trace_id == "trace_123"
    
    def test_attempt_defaults(self):
        """测试尝试记录默认值"""
        timestamp = datetime.now()
        attempt = FallbackAttempt(
            timestamp=timestamp,
            original_service="primary",
            fallback_service="secondary",
            trigger_reason="Test",
            success=False,
            response_time=2.0
        )
        
        assert attempt.error is None
        assert attempt.cost_impact == 0.0
        assert attempt.trace_id is None


class TestFallbackMetrics:
    """测试降级指标收集器"""
    
    def test_metrics_initialization(self):
        """测试指标初始化"""
        metrics = FallbackMetrics(window_size=50)
        
        assert metrics.window_size == 50
        assert len(metrics.attempts) == 0
        assert len(metrics.service_metrics) == 0
    
    def test_record_attempt(self):
        """测试记录尝试"""
        metrics = FallbackMetrics()
        timestamp = datetime.now()
        
        attempt = FallbackAttempt(
            timestamp=timestamp,
            original_service="primary",
            fallback_service="secondary",
            trigger_reason="Test",
            success=True,
            response_time=1.0
        )
        
        metrics.record_attempt(attempt)
        
        assert len(metrics.attempts) == 1
        assert "secondary" in metrics.service_metrics
        
        service_metrics = metrics.service_metrics["secondary"]
        assert len(service_metrics["requests"]) == 1
        assert service_metrics["successes"] == 1
        assert service_metrics["failures"] == 0
        assert service_metrics["total_response_time"] == 1.0
        assert service_metrics["last_request_time"] == timestamp
    
    def test_get_service_metrics_empty(self):
        """测试获取空服务指标"""
        metrics = FallbackMetrics()
        service_metrics = metrics.get_service_metrics("nonexistent")
        
        assert service_metrics["request_count"] == 0
        assert service_metrics["success_rate"] == 0.0
        assert service_metrics["failure_rate"] == 0.0
        assert service_metrics["average_response_time"] == 0.0
        assert service_metrics["last_request_time"] is None
    
    def test_get_service_metrics_with_data(self):
        """测试获取有数据的服务指标"""
        metrics = FallbackMetrics()
        timestamp = datetime.now()
        
        # 记录成功尝试
        success_attempt = FallbackAttempt(
            timestamp=timestamp,
            original_service="primary",
            fallback_service="test_service",
            trigger_reason="Test",
            success=True,
            response_time=1.0
        )
        metrics.record_attempt(success_attempt)
        
        # 记录失败尝试
        failure_attempt = FallbackAttempt(
            timestamp=timestamp,
            original_service="primary",
            fallback_service="test_service",
            trigger_reason="Test",
            success=False,
            response_time=2.0
        )
        metrics.record_attempt(failure_attempt)
        
        service_metrics = metrics.get_service_metrics("test_service")
        
        assert service_metrics["request_count"] == 2
        assert service_metrics["success_rate"] == 0.5
        assert service_metrics["failure_rate"] == 0.5
        assert service_metrics["average_response_time"] == 1.5
        assert service_metrics["last_request_time"] == timestamp
    
    def test_get_fallback_statistics_empty(self):
        """测试获取空降级统计"""
        metrics = FallbackMetrics()
        stats = metrics.get_fallback_statistics()
        
        assert stats["total_attempts"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["most_common_trigger"] is None
        assert stats["average_cost_impact"] == 0.0
    
    def test_get_fallback_statistics_with_data(self):
        """测试获取有数据的降级统计"""
        metrics = FallbackMetrics()
        timestamp = datetime.now()
        
        # 记录多个尝试
        attempts = [
            FallbackAttempt(
                timestamp=timestamp,
                original_service="primary",
                fallback_service="secondary",
                trigger_reason="Service unavailable",
                success=True,
                response_time=1.0,
                cost_impact=0.1
            ),
            FallbackAttempt(
                timestamp=timestamp,
                original_service="primary",
                fallback_service="tertiary",
                trigger_reason="Service unavailable",
                success=False,
                response_time=2.0,
                cost_impact=0.2
            ),
            FallbackAttempt(
                timestamp=timestamp,
                original_service="primary",
                fallback_service="quaternary",
                trigger_reason="Rate limit",
                success=True,
                response_time=1.5,
                cost_impact=0.15
            )
        ]
        
        for attempt in attempts:
            metrics.record_attempt(attempt)
        
        stats = metrics.get_fallback_statistics()
        
        assert stats["total_attempts"] == 3
        assert stats["success_rate"] == 2/3
        assert stats["most_common_trigger"] == "Service unavailable"
        assert abs(stats["average_cost_impact"] - 0.15) < 0.001  # 使用浮点数比较
        assert stats["trigger_distribution"]["Service unavailable"] == 2
        assert stats["trigger_distribution"]["Rate limit"] == 1


class TestServiceHealthMonitor:
    """测试服务健康监控器"""
    
    def test_monitor_initialization(self):
        """测试监控器初始化"""
        monitor = ServiceHealthMonitor(check_interval=60)
        
        assert monitor.check_interval == 60
        assert len(monitor.endpoints) == 0
        assert len(monitor.health_status) == 0
        assert len(monitor.last_check_times) == 0
        assert len(monitor.circuit_breakers) == 0
        assert monitor._running is False
        assert monitor._monitor_thread is None
    
    def test_register_endpoint(self):
        """测试注册端点"""
        monitor = ServiceHealthMonitor()
        endpoint = ServiceEndpoint(
            name="test_service",
            url="https://api.test.com",
            priority=1,
            model_type="test"
        )
        
        monitor.register_endpoint(endpoint)
        
        assert "test_service" in monitor.endpoints
        assert monitor.endpoints["test_service"] == endpoint
        assert monitor.health_status["test_service"] == ServiceStatus.UNKNOWN
        assert "test_service" in monitor.circuit_breakers
        
        circuit_breaker = monitor.circuit_breakers["test_service"]
        assert circuit_breaker["failure_count"] == 0
        assert circuit_breaker["last_failure_time"] is None
        assert circuit_breaker["state"] == "closed"
    
    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        monitor = ServiceHealthMonitor(check_interval=0.1)
        
        # 启动监控
        monitor.start_monitoring()
        assert monitor._running is True
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.is_alive()
        
        # 停止监控
        monitor.stop_monitoring()
        assert monitor._running is False
    
    def test_get_health_status(self):
        """测试获取健康状态"""
        monitor = ServiceHealthMonitor()
        endpoint = ServiceEndpoint(
            name="test_service",
            url="https://api.test.com",
            priority=1,
            model_type="test"
        )
        
        monitor.register_endpoint(endpoint)
        
        # 初始状态
        assert monitor.get_health_status("test_service") == ServiceStatus.UNKNOWN
        assert monitor.get_health_status("nonexistent") == ServiceStatus.UNKNOWN
        
        # 更新状态
        monitor._update_health_status("test_service", ServiceStatus.HEALTHY)
        assert monitor.get_health_status("test_service") == ServiceStatus.HEALTHY
    
    def test_is_service_available(self):
        """测试服务可用性检查"""
        monitor = ServiceHealthMonitor()
        endpoint = ServiceEndpoint(
            name="test_service",
            url="https://api.test.com",
            priority=1,
            model_type="test"
        )
        
        monitor.register_endpoint(endpoint)
        
        # 健康状态 + 熔断器关闭
        monitor._update_health_status("test_service", ServiceStatus.HEALTHY)
        assert monitor.is_service_available("test_service") is True
        
        # 降级状态 + 熔断器关闭
        monitor._update_health_status("test_service", ServiceStatus.DEGRADED)
        assert monitor.is_service_available("test_service") is True
        
        # 不可用状态
        monitor._update_health_status("test_service", ServiceStatus.UNAVAILABLE)
        assert monitor.is_service_available("test_service") is False
        
        # 熔断器打开
        monitor._update_health_status("test_service", ServiceStatus.HEALTHY)
        monitor.circuit_breakers["test_service"]["state"] = "open"
        assert monitor.is_service_available("test_service") is False
    
    def test_handle_health_check_failure(self):
        """测试健康检查失败处理"""
        monitor = ServiceHealthMonitor()
        endpoint = ServiceEndpoint(
            name="test_service",
            url="https://api.test.com",
            priority=1,
            model_type="test",
            circuit_breaker_threshold=3
        )
        
        monitor.register_endpoint(endpoint)
        
        # 第一次失败
        monitor._handle_health_check_failure("test_service")
        circuit_breaker = monitor.circuit_breakers["test_service"]
        assert circuit_breaker["failure_count"] == 1
        assert circuit_breaker["state"] == "closed"
        assert monitor.get_health_status("test_service") == ServiceStatus.UNAVAILABLE
        
        # 第二次失败
        monitor._handle_health_check_failure("test_service")
        assert circuit_breaker["failure_count"] == 2
        assert circuit_breaker["state"] == "closed"
        
        # 第三次失败，触发熔断器
        monitor._handle_health_check_failure("test_service")
        assert circuit_breaker["failure_count"] == 3
        assert circuit_breaker["state"] == "open"
        assert circuit_breaker["last_failure_time"] is not None


class TestFallbackDecisionEngine:
    """测试降级决策引擎"""
    
    def test_engine_initialization(self):
        """测试引擎初始化"""
        rules = [
            FallbackRule(FallbackTrigger.ERROR_RATE, 0.5),
            FallbackRule(FallbackTrigger.RESPONSE_TIME, 5.0)
        ]
        
        engine = FallbackDecisionEngine(rules)
        
        assert len(engine.rules) == 2
        assert isinstance(engine.metrics, FallbackMetrics)
        assert isinstance(engine.health_monitor, ServiceHealthMonitor)
    
    def test_add_rule(self):
        """测试添加规则"""
        engine = FallbackDecisionEngine()
        
        rule1 = FallbackRule(FallbackTrigger.ERROR_RATE, 0.5, priority=2)
        rule2 = FallbackRule(FallbackTrigger.RESPONSE_TIME, 5.0, priority=1)
        
        engine.add_rule(rule1)
        engine.add_rule(rule2)
        
        assert len(engine.rules) == 2
        # 应该按优先级排序
        assert engine.rules[0].priority == 1
        assert engine.rules[1].priority == 2
    
    def test_should_trigger_fallback_service_unavailable(self):
        """测试服务不可用时触发降级"""
        engine = FallbackDecisionEngine()
        
        # Mock健康监控器返回不可用
        engine.health_monitor.is_service_available = Mock(return_value=False)
        
        should_trigger, reason = engine.should_trigger_fallback("test_service")
        
        assert should_trigger is True
        assert reason == "Service unavailable"
    
    def test_should_trigger_fallback_error_types(self):
        """测试错误类型触发降级"""
        engine = FallbackDecisionEngine()
        
        # Mock健康监控器返回可用
        engine.health_monitor.is_service_available = Mock(return_value=True)
        
        # 测试不同错误类型
        test_cases = [
            (ServiceUnavailableError("Service down"), True, "Service error: ServiceUnavailableError"),
            (QuotaExceededError("Quota exceeded"), True, "Service error: QuotaExceededError"),
            (RateLimitError("Rate limited"), True, "Rate limit exceeded"),
            (TimeoutError("Timeout"), True, "Request timeout"),
            (ValueError("Invalid value"), False, "No fallback needed")
        ]
        
        for error, expected_trigger, expected_reason in test_cases:
            should_trigger, reason = engine.should_trigger_fallback("test_service", error)
            assert should_trigger == expected_trigger
            assert reason == expected_reason
    
    def test_should_trigger_fallback_rules(self):
        """测试规则触发降级"""
        rules = [
            FallbackRule(FallbackTrigger.ERROR_RATE, 0.5, min_requests=2)
        ]
        engine = FallbackDecisionEngine(rules)
        
        # Mock健康监控器返回可用
        engine.health_monitor.is_service_available = Mock(return_value=True)
        
        # Mock服务指标
        engine.metrics.get_service_metrics = Mock(return_value={
            'request_count': 5,
            'failure_rate': 0.6,  # 超过阈值
            'success_rate': 0.4,
            'average_response_time': 2.0
        })
        
        should_trigger, reason = engine.should_trigger_fallback("test_service")
        
        assert should_trigger is True
        assert reason == "Rule triggered: error_rate"
    
    def test_select_fallback_service(self):
        """测试选择降级服务"""
        engine = FallbackDecisionEngine()
        
        # Mock健康监控器
        def mock_is_available(service):
            return service != "unavailable_service"
        
        engine.health_monitor.is_service_available = mock_is_available
        
        available_services = ["primary", "secondary", "tertiary", "unavailable_service"]
        
        # 选择降级服务（排除原始服务和不可用服务）
        fallback = engine.select_fallback_service("primary", available_services)
        
        assert fallback in ["secondary", "tertiary"]
        assert fallback != "primary"
        assert fallback != "unavailable_service"
    
    def test_get_fallback_chain(self):
        """测试获取降级链"""
        engine = FallbackDecisionEngine()
        
        # Mock健康监控器
        engine.health_monitor.is_service_available = Mock(return_value=True)
        
        available_services = ["primary", "secondary", "tertiary"]
        
        chain = engine.get_fallback_chain("primary", available_services)
        
        assert "primary" not in chain
        assert len(chain) <= 2  # 排除原始服务后的数量
        assert all(service in available_services for service in chain)


class TestFallbackStrategies:
    """测试降级策略"""
    
    def test_priority_fallback_strategy(self):
        """测试基于优先级的降级策略"""
        priorities = {
            "service_a": 1,
            "service_b": 3,
            "service_c": 2
        }
        
        strategy = PriorityFallbackStrategy(priorities)
        available_services = ["original", "service_a", "service_b", "service_c"]
        
        fallback = strategy.select_fallback_service("original", available_services)
        
        # 应该选择优先级最高的（数字最小）
        assert fallback == "service_a"
    
    def test_priority_fallback_strategy_no_candidates(self):
        """测试优先级策略无候选服务"""
        strategy = PriorityFallbackStrategy({})
        
        fallback = strategy.select_fallback_service("only_service", ["only_service"])
        
        assert fallback is None
    
    def test_cost_optimized_fallback_strategy(self):
        """测试成本优化的降级策略"""
        costs = {
            "expensive_service": 0.1,
            "cheap_service": 0.01,
            "medium_service": 0.05
        }
        
        strategy = CostOptimizedFallbackStrategy(costs)
        available_services = ["original", "expensive_service", "cheap_service", "medium_service"]
        
        fallback = strategy.select_fallback_service("original", available_services)
        
        # 应该选择成本最低的
        assert fallback == "cheap_service"
    
    def test_cost_optimized_fallback_strategy_no_candidates(self):
        """测试成本策略无候选服务"""
        strategy = CostOptimizedFallbackStrategy({})
        
        fallback = strategy.select_fallback_service("only_service", ["only_service"])
        
        assert fallback is None


class TestFallbackManager:
    """测试降级管理器"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        strategy = PriorityFallbackStrategy({})
        engine = FallbackDecisionEngine()
        
        manager = FallbackManager(strategy, engine)
        
        assert manager.strategy == strategy
        assert manager.decision_engine == engine
        assert len(manager.endpoints) == 0
        assert isinstance(manager.metrics, FallbackMetrics)
    
    def test_manager_default_initialization(self):
        """测试管理器默认初始化"""
        manager = FallbackManager()
        
        assert isinstance(manager.strategy, PriorityFallbackStrategy)
        assert isinstance(manager.decision_engine, FallbackDecisionEngine)
        assert isinstance(manager.metrics, FallbackMetrics)
    
    def test_register_endpoint(self):
        """测试注册端点"""
        manager = FallbackManager()
        endpoint = ServiceEndpoint(
            name="test_service",
            url="https://api.test.com",
            priority=1,
            model_type="test"
        )
        
        manager.register_endpoint(endpoint)
        
        assert "test_service" in manager.endpoints
        assert manager.endpoints["test_service"] == endpoint
    
    def test_start_stop_monitoring(self):
        """测试启动停止监控"""
        manager = FallbackManager()
        
        # Mock决策引擎的健康监控器
        manager.decision_engine.health_monitor.start_monitoring = Mock()
        manager.decision_engine.health_monitor.stop_monitoring = Mock()
        
        manager.start_monitoring()
        manager.decision_engine.health_monitor.start_monitoring.assert_called_once()
        
        manager.stop_monitoring()
        manager.decision_engine.health_monitor.stop_monitoring.assert_called_once()
    
    def test_build_fallback_chain(self):
        """测试构建降级链"""
        manager = FallbackManager()
        
        # 注册多个端点
        endpoints = [
            ServiceEndpoint("primary", "url1", priority=1, model_type="test"),
            ServiceEndpoint("secondary", "url2", priority=2, model_type="test"),
            ServiceEndpoint("tertiary", "url3", priority=3, model_type="test")
        ]
        
        for endpoint in endpoints:
            manager.register_endpoint(endpoint)
        
        chain = manager._build_fallback_chain("primary")
        
        assert chain[0] == "primary"  # 主服务在第一位
        assert chain[1] == "secondary"  # 按优先级排序
        assert chain[2] == "tertiary"
        assert len(chain) == 3
    
    def test_execute_with_fallback_success(self):
        """测试降级执行成功"""
        manager = FallbackManager()
        
        # 注册端点
        endpoints = [
            ServiceEndpoint("primary", "url1", priority=1, model_type="test"),
            ServiceEndpoint("secondary", "url2", priority=2, model_type="test")
        ]
        
        for endpoint in endpoints:
            manager.register_endpoint(endpoint)
        
        # Mock健康监控器
        manager.decision_engine.health_monitor.is_service_available = Mock(return_value=True)
        
        # Mock请求函数
        def mock_request_func(service, data):
            if service == "primary":
                return {"result": "success", "service": service}
            raise Exception("Service failed")
        
        result = manager.execute_with_fallback(
            "primary",
            mock_request_func,
            {"test": "data"}
        )
        
        assert result["result"] == "success"
        assert result["service"] == "primary"
    
    def test_execute_with_fallback_with_fallback(self):
        """测试降级执行使用降级服务"""
        manager = FallbackManager()
        
        # 注册端点
        endpoints = [
            ServiceEndpoint("primary", "url1", priority=1, model_type="test"),
            ServiceEndpoint("secondary", "url2", priority=2, model_type="test")
        ]
        
        for endpoint in endpoints:
            manager.register_endpoint(endpoint)
        
        # Mock健康监控器
        manager.decision_engine.health_monitor.is_service_available = Mock(return_value=True)
        
        # Mock请求函数
        def mock_request_func(service, data):
            if service == "primary":
                raise ServiceUnavailableError("Primary failed")
            elif service == "secondary":
                return {"result": "success", "service": service}
            raise Exception("Service failed")
        
        result = manager.execute_with_fallback(
            "primary",
            mock_request_func,
            {"test": "data"}
        )
        
        assert result["result"] == "success"
        assert result["service"] == "secondary"
    
    def test_execute_with_fallback_all_fail(self):
        """测试降级执行全部失败"""
        manager = FallbackManager()
        
        # 注册端点
        endpoints = [
            ServiceEndpoint("primary", "url1", priority=1, model_type="test"),
            ServiceEndpoint("secondary", "url2", priority=2, model_type="test")
        ]
        
        for endpoint in endpoints:
            manager.register_endpoint(endpoint)
        
        # Mock健康监控器
        manager.decision_engine.health_monitor.is_service_available = Mock(return_value=True)
        
        # Mock请求函数（全部失败）
        def mock_request_func(service, data):
            raise ServiceUnavailableError(f"{service} failed")
        
        with pytest.raises(ServiceUnavailableError) as exc_info:
            manager.execute_with_fallback(
                "primary",
                mock_request_func,
                {"test": "data"}
            )
        
        error = exc_info.value
        assert hasattr(error, 'error_details')
        assert error.error_details['original_service'] == "primary"
        assert error.error_details['total_attempts'] == 2
    
    @pytest.mark.asyncio
    async def test_async_execute_with_fallback_success(self):
        """测试异步降级执行成功"""
        manager = FallbackManager()
        
        # 注册端点
        endpoint = ServiceEndpoint("primary", "url1", priority=1, model_type="test")
        manager.register_endpoint(endpoint)
        
        # Mock健康监控器
        manager.decision_engine.health_monitor.is_service_available = Mock(return_value=True)
        
        # Mock异步请求函数
        async def mock_async_request_func(service, data):
            await asyncio.sleep(0.01)  # 模拟异步操作
            return {"result": "async_success", "service": service}
        
        result = await manager.async_execute_with_fallback(
            "primary",
            mock_async_request_func,
            {"test": "data"}
        )
        
        assert result["result"] == "async_success"
        assert result["service"] == "primary"
    
    @pytest.mark.asyncio
    async def test_async_execute_with_fallback_timeout(self):
        """测试异步降级执行超时"""
        manager = FallbackManager()
        
        # 注册端点
        endpoint = ServiceEndpoint("primary", "url1", priority=1, model_type="test")
        manager.register_endpoint(endpoint)
        
        # Mock健康监控器
        manager.decision_engine.health_monitor.is_service_available = Mock(return_value=True)
        
        # Mock异步请求函数（超时）
        async def mock_slow_request_func(service, data):
            await asyncio.sleep(2.0)  # 超过超时时间
            return {"result": "success"}
        
        with pytest.raises((asyncio.TimeoutError, ServiceUnavailableError)):
            await manager.async_execute_with_fallback(
                "primary",
                mock_slow_request_func,
                {"test": "data"},
                timeout=0.1
            )
    
    def test_execute_cascading_fallback(self):
        """测试级联降级"""
        manager = FallbackManager()
        
        # 注册多个端点
        endpoints = [
            ServiceEndpoint("primary", "url1", priority=1, model_type="test"),
            ServiceEndpoint("secondary", "url2", priority=2, model_type="test"),
            ServiceEndpoint("tertiary", "url3", priority=3, model_type="test")
        ]
        
        for endpoint in endpoints:
            manager.register_endpoint(endpoint)
        
        # Mock健康监控器
        manager.decision_engine.health_monitor.is_service_available = Mock(return_value=True)
        
        # Mock请求函数
        def mock_request_func(service, data):
            if service == "tertiary":
                return {"result": "success", "service": service}
            raise ServiceUnavailableError(f"{service} failed")
        
        result = manager.execute_cascading_fallback(
            "primary",
            mock_request_func,
            {"test": "data"},
            max_attempts=3
        )
        
        assert result["result"] == "success"
        assert result["service"] == "tertiary"
    
    def test_get_metrics(self):
        """测试获取指标"""
        manager = FallbackManager()
        
        # 注册端点
        endpoint = ServiceEndpoint("test_service", "url1", priority=1, model_type="test")
        manager.register_endpoint(endpoint)
        
        metrics = manager.get_metrics()
        
        assert "fallback_statistics" in metrics
        assert "service_metrics" in metrics
        assert "test_service" in metrics["service_metrics"]


class TestFallbackIntegration:
    """测试降级功能集成"""
    
    def test_full_fallback_lifecycle(self):
        """测试完整降级生命周期"""
        # 创建管理器
        manager = FallbackManager()
        
        # 注册多个服务端点
        endpoints = [
            ServiceEndpoint("primary", "https://primary.api.com", priority=1, model_type="gpt-4"),
            ServiceEndpoint("secondary", "https://secondary.api.com", priority=2, model_type="gpt-3.5"),
            ServiceEndpoint("tertiary", "https://tertiary.api.com", priority=3, model_type="claude")
        ]
        
        for endpoint in endpoints:
            manager.register_endpoint(endpoint)
        
        # 启动监控
        manager.start_monitoring()
        
        try:
            # Mock健康监控器
            manager.decision_engine.health_monitor.is_service_available = Mock(return_value=True)
            
            # 模拟请求函数
            call_count = {"primary": 0, "secondary": 0, "tertiary": 0}
            
            def mock_request_func(service, data):
                call_count[service] += 1
                if service == "primary" and call_count[service] <= 2:
                    raise ServiceUnavailableError("Primary temporarily down")
                elif service == "secondary":
                    raise RateLimitError("Secondary rate limited")
                elif service == "tertiary":
                    return {"result": "success", "service": service, "model": "claude"}
                raise Exception("Unexpected service")
            
            # 执行带降级的请求
            result = manager.execute_with_fallback(
                "primary",
                mock_request_func,
                {"prompt": "Hello, world!"}
            )
            
            # 验证结果
            assert result["result"] == "success"
            assert result["service"] == "tertiary"
            assert result["model"] == "claude"
            
            # 验证指标
            metrics = manager.get_metrics()
            assert metrics["fallback_statistics"]["total_attempts"] > 0
            
        finally:
            # 停止监控
            manager.stop_monitoring()
    
    def test_circuit_breaker_integration(self):
        """测试熔断器集成"""
        manager = FallbackManager()
        
        # 注册端点
        endpoint = ServiceEndpoint(
            "test_service", 
            "https://test.api.com", 
            priority=1, 
            model_type="test",
            circuit_breaker_threshold=2
        )
        manager.register_endpoint(endpoint)
        
        # 模拟多次健康检查失败
        for _ in range(3):
            manager.decision_engine.health_monitor._handle_health_check_failure("test_service")
        
        # 验证熔断器状态
        circuit_breaker = manager.decision_engine.health_monitor.circuit_breakers["test_service"]
        assert circuit_breaker["state"] == "open"
        assert not manager.decision_engine.health_monitor.is_service_available("test_service")
    
    def test_metrics_collection_integration(self):
        """测试指标收集集成"""
        manager = FallbackManager()
        
        # 注册端点
        endpoints = [
            ServiceEndpoint("service_a", "url_a", priority=1, model_type="test"),
            ServiceEndpoint("service_b", "url_b", priority=2, model_type="test")
        ]
        
        for endpoint in endpoints:
            manager.register_endpoint(endpoint)
        
        # Mock健康监控器
        manager.decision_engine.health_monitor.is_service_available = Mock(return_value=True)
        
        # 执行多次请求以收集指标
        def mock_request_func(service, data):
            if service == "service_a":
                raise ServiceUnavailableError("Service A down")
            return {"result": "success", "service": service}
        
        # 执行多次降级请求
        for i in range(3):
            try:
                manager.execute_with_fallback(
                    "service_a",
                    mock_request_func,
                    {"request_id": i}
                )
            except Exception:
                pass
        
        # 验证指标收集
        metrics = manager.get_metrics()
        fallback_stats = metrics["fallback_statistics"]
        
        assert fallback_stats["total_attempts"] > 0
        assert "service_b" in metrics["service_metrics"]
        
        service_b_metrics = metrics["service_metrics"]["service_b"]
        assert service_b_metrics["request_count"] > 0
        assert service_b_metrics["success_rate"] > 0


class TestServiceHealthMonitorAdvanced:
    """测试服务健康监控器的高级功能"""
    
    def test_monitor_loop_exception_handling(self):
        """测试监控循环中的异常处理"""
        monitor = ServiceHealthMonitor(check_interval=0.1)
        
        endpoint = ServiceEndpoint(
            name="test_service",
            url="http://test.com",
            priority=1,
            model_type="chat"
        )
        monitor.register_endpoint(endpoint)
        
        # Mock _check_all_endpoints 抛出异常
        with patch.object(monitor, '_check_all_endpoints', side_effect=Exception("Test error")):
            monitor.start_monitoring()
            time.sleep(0.2)  # 让监控循环运行一段时间
            monitor.stop_monitoring()
        
        # 验证监控器能够处理异常并继续运行
        assert not monitor._running
    
    def test_circuit_breaker_half_open_state(self):
        """测试熔断器半开状态逻辑"""
        monitor = ServiceHealthMonitor()
        
        endpoint = ServiceEndpoint(
            name="test_service",
            url="http://test.com",
            priority=1,
            model_type="chat",
            circuit_breaker_timeout=1  # 1秒超时
        )
        monitor.register_endpoint(endpoint)
        
        # 设置熔断器为开启状态，且超时已过
        monitor.circuit_breakers["test_service"] = {
            'state': 'open',
            'failure_count': 5,
            'last_failure_time': datetime.now() - timedelta(seconds=2)  # 2秒前失败
        }
        
        # Mock健康检查成功
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            
            # 检查端点健康状态
            monitor._check_endpoint_health("test_service")
        
        # 验证熔断器状态变为关闭（因为健康检查成功）
        assert monitor.circuit_breakers["test_service"]['state'] == 'closed'
        assert monitor.circuit_breakers["test_service"]['failure_count'] == 0
    
    def test_circuit_breaker_open_state_timeout_not_reached(self):
        """测试熔断器开启状态但超时未到达"""
        monitor = ServiceHealthMonitor()
        
        endpoint = ServiceEndpoint(
            name="test_service",
            url="http://test.com",
            priority=1,
            model_type="chat",
            circuit_breaker_timeout=10  # 10秒超时
        )
        monitor.register_endpoint(endpoint)
        
        # 设置熔断器为开启状态，但超时未到达
        monitor.circuit_breakers["test_service"] = {
            'state': 'open',
            'failure_count': 5,
            'last_failure_time': datetime.now() - timedelta(seconds=1)  # 1秒前失败
        }
        
        # 检查端点健康状态
        monitor._check_endpoint_health("test_service")
        
        # 验证服务状态被设置为不可用
        assert monitor.health_status["test_service"] == ServiceStatus.UNAVAILABLE
        # 验证熔断器仍为开启状态
        assert monitor.circuit_breakers["test_service"]['state'] == 'open'
    
    def test_health_check_with_url(self):
        """测试带健康检查URL的端点"""
        monitor = ServiceHealthMonitor()
        
        endpoint = ServiceEndpoint(
            name="test_service",
            url="http://test.com",
            priority=1,
            model_type="chat",
            health_check_url="http://test.com/health"
        )
        monitor.register_endpoint(endpoint)
        
        # Mock健康检查成功
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            
            # 检查端点健康状态
            monitor._check_endpoint_health("test_service")
        
        # 验证健康状态被设置为健康
        assert monitor.health_status["test_service"] == ServiceStatus.HEALTHY