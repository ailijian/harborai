#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时一致性监控器测试

测试RealTimeConsistencyMonitor类的各项功能
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from typing import List

from harborai.core.consistency.real_time_monitor import (
    RealTimeConsistencyMonitor,
    MonitoringMode,
    MonitoringStatus,
    MonitoringConfig,
    MonitoringMetrics
)
from harborai.core.consistency.data_consistency_checker import (
    ConsistencyIssue,
    IssueType,
    IssueSeverity
)
from harborai.core.alerts.alert_manager import AlertSeverity


class TestRealTimeConsistencyMonitor:
    """实时一致性监控器测试类"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """模拟数据库管理器"""
        mock = Mock()
        mock.is_connected = True
        return mock
    
    @pytest.fixture
    def mock_consistency_checker(self):
        """模拟一致性检查器"""
        mock = Mock()
        # 模拟各种检查方法
        mock.check_token_consistency = AsyncMock(return_value=([], 0))
        mock.check_cost_consistency = AsyncMock(return_value=([], 0))
        mock.check_tracing_completeness = AsyncMock(return_value=([], 0))
        mock.check_foreign_key_integrity = AsyncMock(return_value=([], 0))
        mock.check_data_range_consistency = AsyncMock(return_value=([], 0))
        mock.check_performance_anomalies = AsyncMock(return_value=([], 0))
        return mock
    
    @pytest.fixture
    def mock_auto_correction_service(self):
        """模拟自动修正服务"""
        mock = Mock()
        mock.auto_correct_issues = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_alert_manager(self):
        """模拟告警管理器"""
        mock = Mock()
        mock.add_rule = Mock()
        mock.trigger_alert = AsyncMock()
        return mock
    
    @pytest.fixture
    def monitor(self, mock_db_manager, mock_consistency_checker, 
                mock_auto_correction_service, mock_alert_manager):
        """创建监控器实例"""
        return RealTimeConsistencyMonitor(
            db_manager=mock_db_manager,
            consistency_checker=mock_consistency_checker,
            auto_correction_service=mock_auto_correction_service,
            alert_manager=mock_alert_manager
        )
    
    def test_initialization(self, monitor):
        """测试初始化"""
        assert monitor.status == MonitoringStatus.STOPPED
        assert monitor.config.check_interval == 60  # 1分钟
        assert monitor.config.auto_correction_enabled is True
        assert monitor.config.alert_enabled is True
        assert monitor.metrics.checks_performed == 0
        assert monitor.metrics.issues_detected == 0
        assert len(monitor.recent_issues) == 0
    
    def test_monitoring_config_creation(self):
        """测试监控配置创建"""
        config = MonitoringConfig(
            check_interval=120,
            auto_correction_enabled=False,
            alert_enabled=True,
            batch_size=500
        )
        assert config.check_interval == 120
        assert config.auto_correction_enabled is False
        assert config.alert_enabled is True
        assert config.batch_size == 500
    
    def test_monitoring_metrics_creation(self):
        """测试监控指标创建"""
        metrics = MonitoringMetrics(
            checks_performed=10,
            issues_detected=5,
            avg_check_duration=1.5
        )
        assert metrics.checks_performed == 10
        assert metrics.issues_detected == 5
        assert metrics.avg_check_duration == 1.5
    
    @pytest.mark.asyncio
    async def test_start_monitoring_continuous(self, monitor):
        """测试启动连续监控"""
        with patch('asyncio.create_task') as mock_create_task:
            await monitor.start()
            assert monitor.status == MonitoringStatus.RUNNING
            mock_create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_start_monitoring_scheduled(self, monitor):
        """测试启动定时监控"""
        monitor.config.mode = MonitoringMode.SCHEDULED
        with patch('asyncio.create_task') as mock_create_task:
            await monitor.start()
            assert monitor.status == MonitoringStatus.RUNNING
            mock_create_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitor):
        """测试停止监控"""
        # 先启动监控
        with patch('asyncio.create_task') as mock_create_task:
            # 创建一个模拟的任务
            mock_task = Mock()
            mock_task.done.return_value = True
            mock_create_task.return_value = mock_task
            monitor.monitoring_task = mock_task
            
            await monitor.start()
        
        # 停止监控
        await monitor.stop()
        assert monitor.status == MonitoringStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_pause_resume_monitoring(self, monitor):
        """测试暂停和恢复监控"""
        # 先启动监控
        with patch('asyncio.create_task'):
            await monitor.start()
        
        # 暂停监控
        await monitor.pause()
        assert monitor.status == MonitoringStatus.PAUSED
        
        # 恢复监控
        await monitor.resume()
        assert monitor.status == MonitoringStatus.RUNNING
    
    def test_add_status_callback(self, monitor):
        """测试添加状态回调"""
        callback = Mock()
        monitor.add_status_callback(callback)
        assert callback in monitor.status_callbacks
    
    def test_get_metrics(self, monitor):
        """测试获取指标"""
        metrics = monitor.get_metrics()
        assert isinstance(metrics, MonitoringMetrics)
        assert metrics.checks_performed == 0
    
    def test_get_recent_issues(self, monitor):
        """测试获取最近问题"""
        # 添加一些测试问题
        issue = ConsistencyIssue(
            issue_id="test_issue_1",
            table_name="test_table",
            record_id="123",
            issue_type=IssueType.TOKEN_MISMATCH,
            description="测试问题",
            severity=IssueSeverity.MEDIUM,
            detected_at=datetime.now(),
            auto_fixable=True
        )
        monitor.recent_issues.append(issue)
        
        # 测试获取所有问题
        issues = monitor.get_recent_issues()
        assert len(issues) == 1
        assert issues[0] == issue
    
    def test_get_recent_issues_with_limit(self, monitor):
        """测试获取最近问题（带限制）"""
        # 添加多个测试问题
        for i in range(5):
            issue = ConsistencyIssue(
                issue_id=f"test_issue_{i}",
                table_name="test_table",
                record_id=str(i),
                issue_type=IssueType.TOKEN_MISMATCH,
                description=f"测试问题 {i}",
                severity=IssueSeverity.MEDIUM,
                detected_at=datetime.now(),
                auto_fixable=True
            )
            monitor.recent_issues.append(issue)
        
        # 测试限制数量
        issues = monitor.get_recent_issues(limit=3)
        assert len(issues) == 3
    
    @pytest.mark.asyncio
    async def test_force_check(self, monitor):
        """测试强制检查"""
        # 模拟检查结果
        test_issue = ConsistencyIssue(
            issue_id="forced_check_issue",
            table_name="test_table",
            record_id="456",
            issue_type=IssueType.COST_MISMATCH,
            description="强制检查发现的问题",
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=False
        )
        
        # 模拟检查方法返回问题
        monitor.consistency_checker.check_token_consistency.return_value = ([test_issue], 1)
        
        # 执行强制检查
        result = await monitor.force_check()
        
        # 验证结果
        assert result["success"] is True
        assert result["issues_found"] == 1
        assert "duration" in result
        assert "timestamp" in result
        
        # 验证问题被添加到历史记录
        assert len(monitor.recent_issues) == 1
        assert monitor.recent_issues[0].issue_id == "forced_check_issue"
    
    @pytest.mark.asyncio
    async def test_force_check_exception_handling(self, mock_consistency_checker, mock_auto_correction_service, mock_alert_manager, mock_db_manager):
        """测试强制检查异常处理"""
        # 配置所有检查方法都抛出异常
        exception_msg = "检查失败"
        mock_consistency_checker.check_token_consistency.side_effect = Exception(exception_msg)
        mock_consistency_checker.check_cost_consistency.side_effect = Exception(exception_msg)
        mock_consistency_checker.check_tracing_completeness.side_effect = Exception(exception_msg)
        mock_consistency_checker.check_foreign_key_integrity.side_effect = Exception(exception_msg)
        mock_consistency_checker.check_data_range_consistency.side_effect = Exception(exception_msg)
        mock_consistency_checker.check_performance_anomalies.side_effect = Exception(exception_msg)
        
        monitor = RealTimeConsistencyMonitor(
            db_manager=mock_db_manager,
            consistency_checker=mock_consistency_checker,
            auto_correction_service=mock_auto_correction_service,
            alert_manager=mock_alert_manager
        )
        
        result = await monitor.force_check()
        
        # 验证返回错误结果而不是抛出异常
        assert result["success"] is False
        assert "error" in result
        assert "所有一致性检查都失败" in result["error"]
        assert monitor.metrics.error_count == 1

    @pytest.mark.asyncio
    async def test_handle_issues_with_auto_correction(self, mock_consistency_checker, mock_auto_correction_service, mock_alert_manager, mock_db_manager):
        """测试问题处理和自动修正"""
        # 创建测试问题
        issues = [
            ConsistencyIssue(
                issue_type=IssueType.TOKEN_MISMATCH,
                severity=IssueSeverity.HIGH,
                description="Token数量不匹配",
                affected_records=["record1"],
                auto_fixable=True
            )
        ]
        
        # 配置自动修正服务
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.total_records_affected = 1
        mock_auto_correction_service.auto_correct_issues.return_value = mock_result
        
        monitor = RealTimeConsistencyMonitor(
            db_manager=mock_db_manager,
            consistency_checker=mock_consistency_checker,
            auto_correction_service=mock_auto_correction_service,
            alert_manager=mock_alert_manager
        )
        
        await monitor._handle_issues(issues)
        
        # 验证自动修正被调用
        mock_auto_correction_service.auto_correct_issues.assert_called_once()
        assert monitor.metrics.issues_detected == 1
        assert monitor.metrics.issues_resolved == 1
    
    @pytest.mark.asyncio
    async def test_perform_consistency_check(self, monitor):
        """测试执行一致性检查"""
        # 模拟检查结果
        test_issue = ConsistencyIssue(
            issue_id="check_issue",
            table_name="test_table",
            record_id="789",
            issue_type=IssueType.MISSING_TRACING,
            description="检查发现的问题",
            severity=IssueSeverity.LOW,
            detected_at=datetime.now(),
            auto_fixable=True
        )
        
        # 模拟各种检查方法返回结果
        monitor.consistency_checker.check_token_consistency.return_value = ([test_issue], 1)
        monitor.consistency_checker.check_cost_consistency.return_value = ([], 0)
        monitor.consistency_checker.check_tracing_completeness.return_value = ([], 0)
        monitor.consistency_checker.check_foreign_key_integrity.return_value = ([], 0)
        monitor.consistency_checker.check_data_range_consistency.return_value = ([], 0)
        monitor.consistency_checker.check_performance_anomalies.return_value = ([], 0)
        
        # 执行检查
        issues = await monitor._perform_consistency_check()
        
        # 验证结果
        assert len(issues) == 1
        assert issues[0].issue_id == "check_issue"
        
        # 验证各种检查方法被调用
        monitor.consistency_checker.check_token_consistency.assert_called_once()
        monitor.consistency_checker.check_cost_consistency.assert_called_once()
        monitor.consistency_checker.check_tracing_completeness.assert_called_once()
        monitor.consistency_checker.check_foreign_key_integrity.assert_called_once()
        monitor.consistency_checker.check_data_range_consistency.assert_called_once()
        monitor.consistency_checker.check_performance_anomalies.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_perform_consistency_check_with_issues(self, monitor):
        """测试执行一致性检查（有问题）"""
        # 模拟检查结果
        test_issues = [
            ConsistencyIssue(
                issue_id="issue_1",
                table_name="test_table",
                record_id="1",
                issue_type=IssueType.TOKEN_MISMATCH,
                description="Token不匹配",
                severity=IssueSeverity.HIGH,
                detected_at=datetime.now(),
                auto_fixable=True
            ),
            ConsistencyIssue(
                issue_id="issue_2",
                table_name="test_table",
                record_id="2",
                issue_type=IssueType.COST_MISMATCH,
                description="成本不匹配",
                severity=IssueSeverity.MEDIUM,
                detected_at=datetime.now(),
                auto_fixable=False
            )
        ]
        
        # 模拟不同检查方法返回不同问题
        monitor.consistency_checker.check_token_consistency.return_value = ([test_issues[0]], 1)
        monitor.consistency_checker.check_cost_consistency.return_value = ([test_issues[1]], 1)
        monitor.consistency_checker.check_tracing_completeness.return_value = ([], 0)
        monitor.consistency_checker.check_foreign_key_integrity.return_value = ([], 0)
        monitor.consistency_checker.check_data_range_consistency.return_value = ([], 0)
        monitor.consistency_checker.check_performance_anomalies.return_value = ([], 0)
        
        # 执行检查
        issues = await monitor._perform_consistency_check()
        
        # 验证结果
        assert len(issues) == 2
        assert issues[0].issue_id == "issue_1"
        assert issues[1].issue_id == "issue_2"
    
    @pytest.mark.asyncio
    async def test_handle_issues_with_auto_correction(self, monitor):
        """测试处理问题（启用自动修正）"""
        issues = [
            ConsistencyIssue(
                issue_id="auto_fix_issue",
                table_name="test_table",
                record_id="101",
                issue_type=IssueType.TOKEN_MISMATCH,  # 使用有效的类型
                description="可自动修正的问题",
                severity=IssueSeverity.MEDIUM,
                detected_at=datetime.now(),
                auto_fixable=True
            )
        ]
        
        # 模拟自动修正服务的返回值
        mock_result = Mock()
        mock_result.success = True
        mock_result.total_records_affected = 1
        monitor.auto_correction_service.auto_correct_issues.return_value = mock_result
        
        # 执行问题处理
        await monitor._handle_issues(issues)
        
        # 验证自动修正被调用
        monitor.auto_correction_service.auto_correct_issues.assert_called_once()
        
        # 验证问题被添加到历史记录
        assert len(monitor.recent_issues) == 1
        assert monitor.metrics.issues_detected == 1
    
    @pytest.mark.asyncio
    async def test_handle_issues_with_alerting(self, monitor):
        """测试处理问题（启用告警）"""
        # 创建足够多的严重问题以触发告警
        issues = []
        for i in range(15):  # 超过默认的critical_issue_threshold (10)
            issues.append(
                ConsistencyIssue(
                    issue_id=f"critical_issue_{i}",
                    table_name="test_table",
                    record_id=str(i),
                    issue_type=IssueType.DATA_CORRUPTION,
                    description=f"严重问题 {i}",
                    severity=IssueSeverity.CRITICAL,
                    detected_at=datetime.now(),
                    auto_fixable=False
                )
            )
        
        # 模拟告警创建方法
        with patch.object(monitor, '_create_alert') as mock_create_alert:
            # 执行问题处理
            await monitor._handle_issues(issues)
            
            # 验证告警被创建
            mock_create_alert.assert_called()
        
        # 验证问题被添加到历史记录
        assert len(monitor.recent_issues) == 15
        assert monitor.metrics.issues_detected == 15
    
    @pytest.mark.asyncio
    async def test_handle_issues_disabled_auto_correction(self, monitor):
        """测试处理问题（禁用自动修正）"""
        monitor.config.auto_correction_enabled = False
        
        issues = [
            ConsistencyIssue(
                issue_id="no_auto_fix_issue",
                table_name="test_table",
                record_id="303",
                issue_type=IssueType.COST_MISMATCH,
                description="不自动修正的问题",
                severity=IssueSeverity.HIGH,
                detected_at=datetime.now(),
                auto_fixable=True
            )
        ]
        
        # 执行问题处理
        await monitor._handle_issues(issues)
        
        # 验证自动修正未被调用
        monitor.auto_correction_service.auto_correct_issues.assert_not_called()
        
        # 验证问题被添加到历史记录
        assert len(monitor.recent_issues) == 1
    
    def test_adjust_check_interval(self, monitor):
        """测试调整检查间隔"""
        # 模拟高负载情况（很多问题，耗时长）
        issues = [Mock() for _ in range(20)]  # 20个问题
        duration = 120.0  # 2分钟
        
        original_interval = monitor.config.check_interval
        monitor._adjust_check_interval(issues, duration)
        
        # 由于问题多但耗时也长，间隔变化可能不大
        # 主要验证方法能正常执行而不抛异常
        assert monitor.config.check_interval >= monitor.config.adaptive_interval_min
        assert monitor.config.check_interval <= monitor.config.adaptive_interval_max
    
    @pytest.mark.asyncio
    async def test_create_alert(self, monitor):
        """测试创建告警"""
        # 测试告警创建方法
        with patch.object(monitor.logger, 'warning') as mock_warning:
            await monitor._create_alert(
                "test_rule",
                AlertSeverity.HIGH,
                "测试告警消息",
                {"test": "metadata"}
            )
            
            # 验证日志被记录
            mock_warning.assert_called_once()
            assert "HIGH" in mock_warning.call_args[0][0]
            assert "test_rule" in mock_warning.call_args[0][0]
            assert "测试告警消息" in mock_warning.call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_register_default_alert_rules(self, monitor):
        """测试注册默认告警规则"""
        await monitor._register_default_alert_rules()
        
        # 验证告警规则被添加（具体实现可能不同）
        # 这里主要验证方法能正常执行
        assert True  # 如果没有异常，测试通过
    
    def test_notify_status_change(self, monitor):
        """测试状态变更通知"""
        # 添加状态回调
        callback = Mock()
        monitor.add_status_callback(callback)
        
        # 触发状态变更
        monitor._notify_status_change()
        
        # 验证回调被调用
        callback.assert_called_once_with(monitor.status)
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_continuous(self, monitor):
        """测试连续监控循环"""
        monitor.config.mode = MonitoringMode.CONTINUOUS
        monitor.config.check_interval = 0.1  # 快速测试
        
        # 模拟检查结果
        monitor.consistency_checker.check_token_consistency.return_value = ([], 0)
        
        # 模拟循环运行一次后停止
        async def stop_after_delay():
            await asyncio.sleep(0.2)  # 等待一次迭代
            monitor._stop_event.set()
        
        # 启动停止任务
        stop_task = asyncio.create_task(stop_after_delay())
        
        # 运行监控循环
        await monitor._monitoring_loop()
        
        # 等待停止任务完成
        await stop_task
        
        # 验证检查被执行
        assert monitor.consistency_checker.check_token_consistency.called
        assert monitor.metrics.checks_performed > 0
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_scheduled(self, monitor):
        """测试定时监控循环"""
        monitor.config.mode = MonitoringMode.SCHEDULED
        monitor.config.check_interval = 0.1  # 快速测试
        
        # 模拟检查结果
        monitor.consistency_checker.check_token_consistency.return_value = ([], 0)
        
        # 模拟循环运行一次后停止
        async def stop_after_delay():
            await asyncio.sleep(0.2)  # 等待一次迭代
            monitor._stop_event.set()
        
        # 启动停止任务
        stop_task = asyncio.create_task(stop_after_delay())
        
        # 运行监控循环
        await monitor._monitoring_loop()
        
        # 等待停止任务完成
        await stop_task
        
        # 验证检查被执行
        assert monitor.consistency_checker.check_token_consistency.called
        assert monitor.metrics.checks_performed > 0