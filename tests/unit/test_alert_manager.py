#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警管理器单元测试

测试告警管理器的核心功能
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from harborai.core.alerts.alert_manager import (
    AlertManager, AlertRule, Alert, AlertSeverity, AlertStatus, AlertCondition
)


class TestAlertManager:
    """告警管理器测试"""
    
    @pytest.fixture
    async def alert_manager(self):
        """告警管理器实例"""
        manager = AlertManager()
        await manager.initialize()
        return manager
        
    @pytest.fixture
    def sample_rule(self):
        """示例告警规则"""
        return AlertRule(
            id="test_rule",
            name="测试规则",
            description="测试用的告警规则",
            severity=AlertSeverity.HIGH,
            condition=AlertCondition.THRESHOLD,
            metric="test_metric",
            threshold=10.0,
            duration=timedelta(minutes=5),
            labels={"component": "test", "env": "dev"},
            annotations={
                "summary": "测试告警",
                "description": "这是一个测试告警",
                "runbook": "检查测试组件"
            }
        )
        
    async def test_initialization(self, alert_manager):
        """测试初始化"""
        assert alert_manager.rules == {}
        assert alert_manager.alerts == {}
        assert alert_manager.running is False
        assert alert_manager.notification_service is None
        assert alert_manager.suppression_service is None
        
    async def test_add_rule(self, alert_manager, sample_rule):
        """测试添加规则"""
        result = await alert_manager.add_rule(sample_rule)
        assert result is True
        assert "test_rule" in alert_manager.rules
        assert alert_manager.rules["test_rule"] == sample_rule
        
    async def test_add_duplicate_rule(self, alert_manager, sample_rule):
        """测试添加重复规则"""
        await alert_manager.add_rule(sample_rule)
        
        # 添加相同ID的规则应该失败
        duplicate_rule = AlertRule(
            id="test_rule",
            name="重复规则",
            description="重复的规则",
            severity=AlertSeverity.LOW,
            condition=AlertCondition.THRESHOLD,
            metric="other_metric",
            threshold=5.0,
            duration=timedelta(minutes=1),
            labels={},
            annotations={}
        )
        
        result = await alert_manager.add_rule(duplicate_rule)
        assert result is False
        assert alert_manager.rules["test_rule"].name == "测试规则"  # 原规则未被覆盖
        
    async def test_update_rule(self, alert_manager, sample_rule):
        """测试更新规则"""
        await alert_manager.add_rule(sample_rule)
        
        # 更新规则
        updated_rule = AlertRule(
            id="test_rule",
            name="更新的规则",
            description="更新后的规则",
            severity=AlertSeverity.CRITICAL,
            condition=AlertCondition.THRESHOLD,
            metric="test_metric",
            threshold=20.0,
            duration=timedelta(minutes=10),
            labels={"component": "test", "env": "prod"},
            annotations={"summary": "更新的告警"}
        )
        
        result = await alert_manager.update_rule(updated_rule)
        assert result is True
        assert alert_manager.rules["test_rule"].name == "更新的规则"
        assert alert_manager.rules["test_rule"].severity == AlertSeverity.CRITICAL
        assert alert_manager.rules["test_rule"].threshold == 20.0
        
    async def test_update_nonexistent_rule(self, alert_manager, sample_rule):
        """测试更新不存在的规则"""
        result = await alert_manager.update_rule(sample_rule)
        assert result is False
        assert "test_rule" not in alert_manager.rules
        
    async def test_remove_rule(self, alert_manager, sample_rule):
        """测试删除规则"""
        await alert_manager.add_rule(sample_rule)
        assert "test_rule" in alert_manager.rules
        
        result = await alert_manager.remove_rule("test_rule")
        assert result is True
        assert "test_rule" not in alert_manager.rules
        
    async def test_remove_nonexistent_rule(self, alert_manager):
        """测试删除不存在的规则"""
        result = await alert_manager.remove_rule("nonexistent_rule")
        assert result is False
        
    async def test_get_rules(self, alert_manager, sample_rule):
        """测试获取规则列表"""
        rules = await alert_manager.get_rules()
        assert len(rules) == 0
        
        await alert_manager.add_rule(sample_rule)
        rules = await alert_manager.get_rules()
        assert len(rules) == 1
        assert rules[0] == sample_rule
        
    async def test_register_metric_provider(self, alert_manager):
        """测试注册指标提供者"""
        mock_provider = Mock()
        alert_manager.register_metric_provider("test_provider", mock_provider)
        
        assert "test_provider" in alert_manager.metric_providers
        assert alert_manager.metric_providers["test_provider"] == mock_provider
        
    async def test_evaluate_threshold_condition(self, alert_manager, sample_rule):
        """测试阈值条件评估"""
        # 测试超过阈值
        result = await alert_manager._evaluate_condition(sample_rule, 15.0)
        assert result is True
        
        # 测试未超过阈值
        result = await alert_manager._evaluate_condition(sample_rule, 5.0)
        assert result is False
        
        # 测试等于阈值
        result = await alert_manager._evaluate_condition(sample_rule, 10.0)
        assert result is False
        
    async def test_evaluate_anomaly_condition(self, alert_manager):
        """测试异常检测条件评估"""
        anomaly_rule = AlertRule(
            id="anomaly_rule",
            name="异常检测规则",
            description="异常检测",
            severity=AlertSeverity.MEDIUM,
            condition=AlertCondition.ANOMALY,
            metric="anomaly_metric",
            threshold=2.0,  # 2倍标准差
            duration=timedelta(minutes=5),
            labels={},
            annotations={}
        )
        
        # 模拟历史数据（正常值在10左右）
        with patch.object(alert_manager, '_get_historical_data') as mock_history:
            mock_history.return_value = [9.0, 10.0, 11.0, 9.5, 10.5] * 10  # 50个正常值
            
            # 测试正常值
            result = await alert_manager._evaluate_condition(anomaly_rule, 10.0)
            assert result is False
            
            # 测试异常值
            result = await alert_manager._evaluate_condition(anomaly_rule, 25.0)
            assert result is True
            
    async def test_create_alert(self, alert_manager, sample_rule):
        """测试创建告警"""
        await alert_manager.add_rule(sample_rule)
        
        alert = await alert_manager._create_alert(sample_rule, 15.0)
        
        assert alert.rule_id == "test_rule"
        assert alert.rule_name == "测试规则"
        assert alert.severity == AlertSeverity.HIGH
        assert alert.status == AlertStatus.FIRING
        assert alert.metric_value == 15.0
        assert alert.threshold == 10.0
        assert alert.labels == {"component": "test", "env": "dev"}
        assert alert.annotations["summary"] == "测试告警"
        assert alert.started_at is not None
        
    async def test_acknowledge_alert(self, alert_manager, sample_rule):
        """测试确认告警"""
        await alert_manager.add_rule(sample_rule)
        alert = await alert_manager._create_alert(sample_rule, 15.0)
        alert_manager.alerts[alert.id] = alert
        
        result = await alert_manager.acknowledge_alert(alert.id, "test_user")
        assert result is True
        
        updated_alert = alert_manager.alerts[alert.id]
        assert updated_alert.status == AlertStatus.ACKNOWLEDGED
        assert updated_alert.acknowledged_by == "test_user"
        assert updated_alert.acknowledged_at is not None
        
    async def test_acknowledge_nonexistent_alert(self, alert_manager):
        """测试确认不存在的告警"""
        result = await alert_manager.acknowledge_alert("nonexistent_alert", "test_user")
        assert result is False
        
    async def test_suppress_alert(self, alert_manager, sample_rule):
        """测试抑制告警"""
        await alert_manager.add_rule(sample_rule)
        alert = await alert_manager._create_alert(sample_rule, 15.0)
        alert_manager.alerts[alert.id] = alert
        
        result = await alert_manager.suppress_alert(alert.id, timedelta(hours=1), "test_user")
        assert result is True
        
        updated_alert = alert_manager.alerts[alert.id]
        assert updated_alert.status == AlertStatus.SUPPRESSED
        assert updated_alert.suppressed_until is not None
        
    async def test_get_active_alerts(self, alert_manager, sample_rule):
        """测试获取活跃告警"""
        await alert_manager.add_rule(sample_rule)
        
        # 创建多个告警
        alert1 = await alert_manager._create_alert(sample_rule, 15.0)
        alert2 = await alert_manager._create_alert(sample_rule, 20.0)
        alert3 = await alert_manager._create_alert(sample_rule, 25.0)
        
        # 设置不同状态
        alert2.status = AlertStatus.ACKNOWLEDGED
        alert3.status = AlertStatus.RESOLVED
        
        alert_manager.alerts[alert1.id] = alert1
        alert_manager.alerts[alert2.id] = alert2
        alert_manager.alerts[alert3.id] = alert3
        
        # 获取活跃告警
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 2  # FIRING 和 ACKNOWLEDGED
        
        firing_alerts = [a for a in active_alerts if a.status == AlertStatus.FIRING]
        acknowledged_alerts = [a for a in active_alerts if a.status == AlertStatus.ACKNOWLEDGED]
        
        assert len(firing_alerts) == 1
        assert len(acknowledged_alerts) == 1
        
    async def test_get_alerts_by_severity(self, alert_manager):
        """测试按严重程度获取告警"""
        # 创建不同严重程度的规则和告警
        rules = [
            AlertRule(
                id="critical_rule",
                name="严重规则",
                description="严重告警",
                severity=AlertSeverity.CRITICAL,
                condition=AlertCondition.THRESHOLD,
                metric="critical_metric",
                threshold=10.0,
                duration=timedelta(minutes=1),
                labels={},
                annotations={}
            ),
            AlertRule(
                id="high_rule",
                name="高级规则",
                description="高级告警",
                severity=AlertSeverity.HIGH,
                condition=AlertCondition.THRESHOLD,
                metric="high_metric",
                threshold=10.0,
                duration=timedelta(minutes=1),
                labels={},
                annotations={}
            ),
            AlertRule(
                id="medium_rule",
                name="中级规则",
                description="中级告警",
                severity=AlertSeverity.MEDIUM,
                condition=AlertCondition.THRESHOLD,
                metric="medium_metric",
                threshold=10.0,
                duration=timedelta(minutes=1),
                labels={},
                annotations={}
            )
        ]
        
        for rule in rules:
            await alert_manager.add_rule(rule)
            alert = await alert_manager._create_alert(rule, 15.0)
            alert_manager.alerts[alert.id] = alert
            
        # 获取严重告警
        critical_alerts = await alert_manager.get_alerts_by_severity(AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL
        
        # 获取高级告警
        high_alerts = await alert_manager.get_alerts_by_severity(AlertSeverity.HIGH)
        assert len(high_alerts) == 1
        assert high_alerts[0].severity == AlertSeverity.HIGH
        
    async def test_get_statistics(self, alert_manager):
        """测试获取统计信息"""
        # 创建规则和告警
        rules = [
            AlertRule(
                id=f"rule_{i}",
                name=f"规则 {i}",
                description=f"规则 {i}",
                severity=AlertSeverity.HIGH if i % 2 == 0 else AlertSeverity.MEDIUM,
                condition=AlertCondition.THRESHOLD,
                metric=f"metric_{i}",
                threshold=10.0,
                duration=timedelta(minutes=1),
                labels={},
                annotations={}
            )
            for i in range(5)
        ]
        
        for rule in rules:
            await alert_manager.add_rule(rule)
            
        # 创建告警
        for i, rule in enumerate(rules):
            if i < 3:  # 只为前3个规则创建告警
                alert = await alert_manager._create_alert(rule, 15.0)
                if i == 2:  # 第3个告警设为已确认
                    alert.status = AlertStatus.ACKNOWLEDGED
                alert_manager.alerts[alert.id] = alert
                
        stats = await alert_manager.get_statistics()
        
        assert stats["total_rules"] == 5
        assert stats["active_alerts"] == 3
        assert stats["alerts_by_severity"]["high"] == 2
        assert stats["alerts_by_severity"]["medium"] == 1
        assert stats["alerts_by_status"]["firing"] == 2
        assert stats["alerts_by_status"]["acknowledged"] == 1
        
    async def test_start_stop(self, alert_manager):
        """测试启动和停止"""
        assert alert_manager.running is False
        
        await alert_manager.start()
        assert alert_manager.running is True
        
        await alert_manager.stop()
        assert alert_manager.running is False
        
    async def test_evaluation_loop(self, alert_manager, sample_rule):
        """测试评估循环"""
        # 注册模拟指标提供者
        mock_provider = Mock()
        mock_provider.get_metric = AsyncMock(return_value=15.0)
        alert_manager.register_metric_provider("test", mock_provider)
        
        await alert_manager.add_rule(sample_rule)
        
        # 启动管理器
        await alert_manager.start()
        
        # 等待评估
        await asyncio.sleep(0.1)
        
        # 检查告警是否被创建
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        
        alert = active_alerts[0]
        assert alert.rule_id == "test_rule"
        assert alert.metric_value == 15.0
        
        await alert_manager.stop()
        
    async def test_notification_integration(self, alert_manager, sample_rule):
        """测试通知集成"""
        # 设置模拟通知服务
        mock_notification = Mock()
        mock_notification.send_alert_notification = AsyncMock(return_value=True)
        alert_manager.set_notification_service(mock_notification)
        
        # 注册指标提供者
        mock_provider = Mock()
        mock_provider.get_metric = AsyncMock(return_value=15.0)
        alert_manager.register_metric_provider("test", mock_provider)
        
        await alert_manager.add_rule(sample_rule)
        await alert_manager.start()
        await asyncio.sleep(0.1)
        
        # 检查通知是否被发送
        assert mock_notification.send_alert_notification.called
        
        await alert_manager.stop()
        
    async def test_suppression_integration(self, alert_manager, sample_rule):
        """测试抑制集成"""
        # 设置模拟抑制服务
        mock_suppression = Mock()
        mock_suppression.should_suppress = AsyncMock(return_value=True)
        alert_manager.set_suppression_service(mock_suppression)
        
        # 注册指标提供者
        mock_provider = Mock()
        mock_provider.get_metric = AsyncMock(return_value=15.0)
        alert_manager.register_metric_provider("test", mock_provider)
        
        await alert_manager.add_rule(sample_rule)
        await alert_manager.start()
        await asyncio.sleep(0.1)
        
        # 检查告警是否被抑制
        active_alerts = await alert_manager.get_active_alerts()
        if active_alerts:
            assert active_alerts[0].status == AlertStatus.SUPPRESSED
            
        await alert_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])