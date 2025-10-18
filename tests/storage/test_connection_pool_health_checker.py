#!/usr/bin/env python3
"""
连接池健康检查器测试模块

测试ConnectionPoolHealthChecker的各项功能：
- 健康状态检查
- 性能指标收集
- 监控和告警
- 自动恢复建议

作者: HarborAI团队
创建时间: 2025-01-15
版本: v1.0.0
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timezone, timedelta

from harborai.storage.connection_pool_health_checker import (
    ConnectionPoolHealthChecker,
    HealthStatus,
    HealthCheckResult,
    ConnectionMetrics,
    PerformanceThresholds,
    get_global_health_checker,
    setup_global_health_checker
)
from harborai.storage.connection_pool import ConnectionPool, ConnectionPoolConfig, ConnectionStats


class TestConnectionPoolHealthChecker:
    """连接池健康检查器测试类"""
    
    @pytest.fixture
    def mock_connection_pool(self):
        """模拟连接池"""
        pool = Mock(spec=ConnectionPool)
        pool.config = ConnectionPoolConfig(
            max_connections=10,
            min_connections=2,
            connection_timeout=30.0
        )
        
        # 模拟统计信息
        stats = ConnectionStats()
        stats.total_connections = 5
        stats.active_connections = 3
        stats.failed_connections = 0
        stats.total_requests = 100
        stats.failed_requests = 2
        stats.average_response_time = 150.0
        
        pool._stats = stats
        return pool
    
    @pytest.fixture
    def performance_thresholds(self):
        """性能阈值配置"""
        return PerformanceThresholds(
            max_response_time_ms=200.0,
            max_error_rate=0.05,
            min_connection_utilization=0.1,
            max_connection_utilization=0.8,
            min_throughput=1.0
        )
    
    @pytest.fixture
    def health_checker(self, mock_connection_pool, performance_thresholds):
        """健康检查器实例"""
        return ConnectionPoolHealthChecker(
            connection_pool=mock_connection_pool,
            thresholds=performance_thresholds,
            check_interval=1.0,
            enable_auto_monitoring=False
        )
    
    def test_init(self, mock_connection_pool, performance_thresholds):
        """测试初始化"""
        checker = ConnectionPoolHealthChecker(
            connection_pool=mock_connection_pool,
            thresholds=performance_thresholds,
            check_interval=5.0,
            enable_auto_monitoring=True
        )
        
        assert checker.connection_pool == mock_connection_pool
        assert checker.thresholds == performance_thresholds
        assert checker.check_interval == 5.0
        assert checker.enable_auto_monitoring is True
        assert checker._is_running is False
        assert checker._check_history == []
    
    @pytest.mark.asyncio
    async def test_collect_metrics(self, health_checker):
        """测试指标收集"""
        metrics = await health_checker._collect_metrics()
        
        assert isinstance(metrics, ConnectionMetrics)
        assert metrics.total_connections == 5
        assert metrics.active_connections == 3
        assert metrics.idle_connections == 2
        assert metrics.failed_connections == 0
        assert metrics.connection_utilization == 0.3  # 3/10
        assert metrics.average_response_time == 150.0
        assert metrics.error_rate == 0.02  # 2/100
        assert hasattr(metrics, 'timestamp')
    
    def test_evaluate_health_healthy(self, health_checker):
        """测试健康状态评估 - 健康状态"""
        metrics = ConnectionMetrics(
            total_connections=5,
            active_connections=3,
            idle_connections=2,
            failed_connections=0,
            connection_utilization=0.3,
            average_response_time=100.0,  # 低于阈值
            error_rate=0.01,  # 低于阈值
            throughput=5.0
        )
        
        status, score, details = health_checker._evaluate_health(metrics)
        
        assert status == HealthStatus.HEALTHY
        assert score >= 80
        assert "metrics" in details
        assert "score_breakdown" in details
    
    def test_evaluate_health_warning(self, health_checker):
        """测试健康状态评估 - 警告状态"""
        metrics = ConnectionMetrics(
            total_connections=8,
            active_connections=7,
            idle_connections=1,
            failed_connections=1,
            connection_utilization=0.7,  # 接近阈值
            average_response_time=180.0,  # 接近阈值
            error_rate=0.03,  # 中等错误率
            throughput=2.0
        )
        
        status, score, details = health_checker._evaluate_health(metrics)
        
        assert status == HealthStatus.WARNING
        assert 60 <= score < 80
        assert len(details["score_breakdown"]["warnings"]) > 0
    
    def test_evaluate_health_critical(self, health_checker):
        """测试健康状态评估 - 严重状态"""
        metrics = ConnectionMetrics(
            total_connections=10,
            active_connections=9,
            idle_connections=1,
            failed_connections=3,
            connection_utilization=0.9,  # 超过阈值
            average_response_time=500.0,  # 超过阈值
            error_rate=0.1,  # 超过阈值
            throughput=0.5
        )
        
        status, score, details = health_checker._evaluate_health(metrics)
        
        assert status == HealthStatus.CRITICAL
        assert score < 60
        assert len(details["score_breakdown"]["issues"]) > 0
    
    def test_generate_recommendations(self, health_checker):
        """测试建议生成"""
        # 健康状态
        recommendations = health_checker._generate_recommendations(
            HealthStatus.HEALTHY,
            ConnectionMetrics(connection_utilization=0.5, error_rate=0.01)
        )
        assert "连接池运行正常" in recommendations[0]
        
        # 严重状态
        metrics = ConnectionMetrics(
            connection_utilization=0.9,
            average_response_time=600.0,
            error_rate=0.1,
            failed_connections=5
        )
        recommendations = health_checker._generate_recommendations(
            HealthStatus.CRITICAL,
            metrics
        )
        assert len(recommendations) > 1
        assert any("立即检查" in rec for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_check_health(self, health_checker):
        """测试健康检查"""
        result = await health_checker.check_health()
        
        assert isinstance(result, HealthCheckResult)
        assert isinstance(result.status, HealthStatus)
        assert 0 <= result.score <= 100
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.details, dict)
        assert isinstance(result.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_check_health_with_exception(self, health_checker):
        """测试健康检查异常处理"""
        # 模拟异常
        health_checker.connection_pool._stats = None
        
        result = await health_checker.check_health()
        
        assert result.status == HealthStatus.UNKNOWN
        assert result.score == 0.0
        assert "error" in result.details
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, health_checker):
        """测试启动和停止监控"""
        # 启动监控
        health_checker.enable_auto_monitoring = True
        await health_checker.start_monitoring()
        
        assert health_checker._is_running is True
        assert health_checker._monitoring_task is not None
        
        # 等待一小段时间让监控运行
        await asyncio.sleep(0.1)
        
        # 停止监控
        await health_checker.stop_monitoring()
        
        assert health_checker._is_running is False
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self, health_checker):
        """测试监控循环"""
        health_checker.check_interval = 0.1  # 快速检查
        health_checker.enable_auto_monitoring = True
        
        await health_checker.start_monitoring()
        
        # 等待几次检查
        await asyncio.sleep(0.3)
        
        await health_checker.stop_monitoring()
        
        # 验证检查历史
        assert len(health_checker._check_history) > 0
    
    def test_health_callbacks(self, health_checker):
        """测试健康状态回调"""
        callback_results = []
        
        def test_callback(result: HealthCheckResult):
            callback_results.append(result)
        
        # 添加回调
        health_checker.add_health_callback(test_callback)
        
        # 验证回调已添加
        assert test_callback in health_checker._health_change_callbacks
        
        # 移除回调
        health_checker.remove_health_callback(test_callback)
        
        # 验证回调已移除
        assert test_callback not in health_checker._health_change_callbacks
    
    @pytest.mark.asyncio
    async def test_trigger_health_callbacks(self, health_checker):
        """测试触发健康状态回调"""
        callback_results = []
        
        async def async_callback(result: HealthCheckResult):
            callback_results.append(result)
        
        def sync_callback(result: HealthCheckResult):
            callback_results.append(result)
        
        health_checker.add_health_callback(async_callback)
        health_checker.add_health_callback(sync_callback)
        
        # 创建测试结果
        test_result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            score=90.0,
            timestamp=datetime.now(timezone.utc),
            details={},
            recommendations=[]
        )
        
        # 触发回调
        await health_checker._trigger_health_callbacks(test_result)
        
        # 验证回调被调用
        assert len(callback_results) == 2
        assert all(r == test_result for r in callback_results)
    
    def test_get_current_status(self, health_checker):
        """测试获取当前状态"""
        # 初始状态
        assert health_checker.get_current_status() is None
        
        # 添加检查结果
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            score=90.0,
            timestamp=datetime.now(timezone.utc),
            details={},
            recommendations=[]
        )
        
        health_checker._check_history.append(result)
        
        # 获取当前状态
        current = health_checker.get_current_status()
        assert current == result
    
    def test_get_health_history(self, health_checker):
        """测试获取健康检查历史"""
        # 添加多个检查结果
        results = []
        for i in range(10):
            result = HealthCheckResult(
                status=HealthStatus.HEALTHY,
                score=90.0 - i,
                timestamp=datetime.now(timezone.utc),
                details={},
                recommendations=[]
            )
            results.append(result)
            health_checker._check_history.append(result)
        
        # 获取历史（限制5个）
        history = health_checker.get_health_history(limit=5)
        assert len(history) == 5
        assert history == results[-5:]
    
    def test_get_metrics_summary(self, health_checker):
        """测试获取指标摘要"""
        # 空历史
        summary = health_checker.get_metrics_summary()
        assert summary == {}
        
        # 添加指标历史
        for i in range(5):
            metrics = ConnectionMetrics(
                total_connections=5 + i,
                active_connections=3 + i,
                connection_utilization=0.3 + i * 0.1,
                average_response_time=100.0 + i * 10,
                error_rate=0.01 + i * 0.005,
                throughput=5.0 + i
            )
            metrics.timestamp = datetime.now(timezone.utc)
            health_checker._metrics_history.append(metrics)
        
        health_checker._last_check_time = datetime.now(timezone.utc)
        
        # 获取摘要
        summary = health_checker.get_metrics_summary()
        
        assert "current" in summary
        assert "average" in summary
        assert "trend" in summary
        assert "last_updated" in summary
    
    def test_calculate_average_metrics(self, health_checker):
        """测试计算平均指标"""
        metrics_list = [
            ConnectionMetrics(
                average_response_time=100.0,
                connection_utilization=0.3,
                error_rate=0.01,
                throughput=5.0
            ),
            ConnectionMetrics(
                average_response_time=200.0,
                connection_utilization=0.5,
                error_rate=0.02,
                throughput=7.0
            )
        ]
        
        avg = health_checker._calculate_average_metrics(metrics_list)
        
        assert avg["average_response_time"] == 150.0
        assert avg["average_utilization"] == 0.4
        assert avg["average_error_rate"] == 0.015
        assert avg["average_throughput"] == 6.0
    
    def test_calculate_trend(self, health_checker):
        """测试计算趋势"""
        # 创建趋势数据（响应时间改善，错误率恶化）
        metrics_list = [
            ConnectionMetrics(average_response_time=200.0, error_rate=0.01),
            ConnectionMetrics(average_response_time=190.0, error_rate=0.015),
            ConnectionMetrics(average_response_time=150.0, error_rate=0.02),
            ConnectionMetrics(average_response_time=140.0, error_rate=0.025)
        ]
        
        trend = health_checker._calculate_trend(metrics_list)
        
        assert trend["response_time"] == "improving"
        assert trend["error_rate"] == "degrading"
    
    @pytest.mark.asyncio
    async def test_force_health_check(self, health_checker):
        """测试强制健康检查"""
        result = await health_checker.force_health_check()
        
        assert isinstance(result, HealthCheckResult)
        assert result.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
    
    def test_reset_history(self, health_checker):
        """测试重置历史"""
        # 添加一些历史数据
        health_checker._check_history.append(
            HealthCheckResult(
                status=HealthStatus.HEALTHY,
                score=90.0,
                timestamp=datetime.now(timezone.utc),
                details={},
                recommendations=[]
            )
        )
        
        health_checker._metrics_history.append(ConnectionMetrics())
        
        # 重置历史
        health_checker.reset_history()
        
        assert len(health_checker._check_history) == 0
        assert len(health_checker._metrics_history) == 0
    
    def test_health_check_result_to_dict(self):
        """测试健康检查结果转换为字典"""
        result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            score=90.0,
            timestamp=datetime.now(timezone.utc),
            details={"test": "data"},
            recommendations=["建议1", "建议2"]
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["status"] == "healthy"
        assert result_dict["score"] == 90.0
        assert isinstance(result_dict["timestamp"], str)
        assert result_dict["details"] == {"test": "data"}
        assert result_dict["recommendations"] == ["建议1", "建议2"]
    
    def test_connection_metrics_to_dict(self):
        """测试连接指标转换为字典"""
        metrics = ConnectionMetrics(
            total_connections=10,
            active_connections=5,
            idle_connections=5,
            failed_connections=1,
            connection_utilization=0.5,
            average_response_time=150.0,
            error_rate=0.02,
            throughput=8.0
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["total_connections"] == 10
        assert metrics_dict["active_connections"] == 5
        assert metrics_dict["connection_utilization"] == 0.5
        assert metrics_dict["average_response_time"] == 150.0


class TestGlobalHealthChecker:
    """全局健康检查器测试类"""
    
    def test_global_health_checker_management(self):
        """测试全局健康检查器管理"""
        # 初始状态
        assert get_global_health_checker() is None
        
        # 设置全局实例
        mock_pool = Mock(spec=ConnectionPool)
        mock_pool.config = ConnectionPoolConfig()
        mock_pool._stats = ConnectionStats()
        
        checker = setup_global_health_checker(mock_pool)
        
        # 验证设置成功
        assert get_global_health_checker() == checker
        assert checker.connection_pool == mock_pool


class TestPerformanceThresholds:
    """性能阈值测试类"""
    
    def test_default_thresholds(self):
        """测试默认阈值"""
        thresholds = PerformanceThresholds()
        
        assert thresholds.max_response_time_ms == 1000.0
        assert thresholds.max_error_rate == 0.05
        assert thresholds.min_connection_utilization == 0.1
        assert thresholds.max_connection_utilization == 0.8
        assert thresholds.min_throughput == 1.0
        assert thresholds.health_check_timeout == 5.0
    
    def test_custom_thresholds(self):
        """测试自定义阈值"""
        thresholds = PerformanceThresholds(
            max_response_time_ms=500.0,
            max_error_rate=0.02,
            min_connection_utilization=0.2,
            max_connection_utilization=0.9,
            min_throughput=2.0,
            health_check_timeout=10.0
        )
        
        assert thresholds.max_response_time_ms == 500.0
        assert thresholds.max_error_rate == 0.02
        assert thresholds.min_connection_utilization == 0.2
        assert thresholds.max_connection_utilization == 0.9
        assert thresholds.min_throughput == 2.0
        assert thresholds.health_check_timeout == 10.0


@pytest.mark.asyncio
async def test_integration_health_monitoring():
    """集成测试：健康监控"""
    # 创建模拟连接池
    mock_pool = Mock(spec=ConnectionPool)
    mock_pool.config = ConnectionPoolConfig(max_connections=10)
    
    stats = ConnectionStats()
    stats.total_connections = 5
    stats.active_connections = 3
    stats.failed_connections = 0
    stats.total_requests = 100
    stats.failed_requests = 1
    stats.average_response_time = 120.0
    
    mock_pool._stats = stats
    
    # 创建健康检查器
    checker = ConnectionPoolHealthChecker(
        connection_pool=mock_pool,
        check_interval=0.1,
        enable_auto_monitoring=True
    )
    
    # 启动监控
    await checker.start_monitoring()
    
    # 等待几次检查
    await asyncio.sleep(0.3)
    
    # 验证监控结果
    current_status = checker.get_current_status()
    assert current_status is not None
    assert current_status.status in [HealthStatus.HEALTHY, HealthStatus.WARNING, HealthStatus.CRITICAL]
    
    # 获取指标摘要
    summary = checker.get_metrics_summary()
    assert "current" in summary
    
    # 停止监控
    await checker.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])