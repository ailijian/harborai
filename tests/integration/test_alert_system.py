#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警系统集成测试

测试告警管理器、通知服务、抑制管理器和历史服务的集成功能
"""

import pytest
import asyncio
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from harborai.core.alerts.alert_manager import (
    AlertManager, AlertRule, Alert, AlertSeverity, AlertStatus, AlertCondition
)
from harborai.core.alerts.notification_service import (
    NotificationService, NotificationChannel, NotificationPriority
)
from harborai.core.alerts.suppression_manager import (
    SuppressionManager, SuppressionRule, SuppressionType
)
from harborai.core.alerts.alert_history import AlertHistory, AlertEventType
from harborai.core.alerts.config import get_default_config


class TestAlertSystemIntegration:
    """告警系统集成测试"""
    
    @pytest.fixture
    async def temp_db(self):
        """临时数据库文件"""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        yield path
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
            
    @pytest.fixture
    async def alert_manager(self, temp_db):
        """告警管理器实例"""
        manager = AlertManager()
        await manager.initialize()
        return manager
        
    @pytest.fixture
    async def notification_service(self):
        """通知服务实例"""
        service = NotificationService()
        await service.initialize()
        return service
        
    @pytest.fixture
    async def suppression_manager(self):
        """抑制管理器实例"""
        manager = SuppressionManager()
        await manager.initialize()
        return manager
        
    @pytest.fixture
    async def alert_history(self, temp_db):
        """告警历史服务实例"""
        history = AlertHistory(db_path=temp_db)
        await history.initialize()
        return history
        
    @pytest.fixture
    async def integrated_system(self, alert_manager, notification_service, suppression_manager, alert_history):
        """集成的告警系统"""
        # 设置服务依赖
        alert_manager.set_notification_service(notification_service)
        alert_manager.set_suppression_service(suppression_manager)
        alert_manager.history_service = alert_history
        
        # 注册模拟指标提供者
        mock_provider = Mock()
        mock_provider.get_metric = AsyncMock()
        alert_manager.register_metric_provider("test", mock_provider)
        
        return {
            "alert_manager": alert_manager,
            "notification_service": notification_service,
            "suppression_manager": suppression_manager,
            "alert_history": alert_history,
            "metric_provider": mock_provider
        }
        
    async def test_complete_alert_lifecycle(self, integrated_system):
        """测试完整的告警生命周期"""
        system = integrated_system
        alert_manager = system["alert_manager"]
        metric_provider = system["metric_provider"]
        
        # 创建测试规则
        rule = AlertRule(
            id="test_rule",
            name="测试规则",
            description="测试用规则",
            severity=AlertSeverity.HIGH,
            condition=AlertCondition.THRESHOLD,
            metric="test_metric",
            threshold=10.0,
            duration=timedelta(seconds=1),
            labels={"component": "test"},
            annotations={"summary": "测试告警"}
        )
        
        # 添加规则
        await alert_manager.add_rule(rule)
        
        # 设置指标值触发告警
        metric_provider.get_metric.return_value = 15.0
        
        # 启动告警管理器
        await alert_manager.start()
        
        # 等待告警触发
        await asyncio.sleep(2)
        
        # 检查告警是否创建
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        
        alert = active_alerts[0]
        assert alert.rule_id == "test_rule"
        assert alert.status == AlertStatus.FIRING
        assert alert.metric_value == 15.0
        
        # 确认告警
        await alert_manager.acknowledge_alert(alert.id, "test_user")
        
        # 检查告警状态
        updated_alert = (await alert_manager.get_active_alerts())[0]
        assert updated_alert.status == AlertStatus.ACKNOWLEDGED
        assert updated_alert.acknowledged_by == "test_user"
        
        # 设置指标值解决告警
        metric_provider.get_metric.return_value = 5.0
        
        # 等待告警解决
        await asyncio.sleep(2)
        
        # 检查告警是否解决
        active_alerts = await alert_manager.get_active_alerts()
        resolved_alerts = [a for a in active_alerts if a.status == AlertStatus.RESOLVED]
        assert len(resolved_alerts) == 1
        
        # 停止告警管理器
        await alert_manager.stop()
        
    async def test_notification_integration(self, integrated_system):
        """测试通知集成"""
        system = integrated_system
        alert_manager = system["alert_manager"]
        notification_service = system["notification_service"]
        metric_provider = system["metric_provider"]
        
        # 模拟通知发送
        with patch.object(notification_service, 'send_alert_notification', new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True
            
            # 创建高严重程度规则
            rule = AlertRule(
                id="critical_rule",
                name="严重告警",
                description="严重级别告警",
                severity=AlertSeverity.CRITICAL,
                condition=AlertCondition.THRESHOLD,
                metric="critical_metric",
                threshold=100.0,
                duration=timedelta(seconds=1),
                labels={"component": "critical"},
                annotations={"summary": "严重告警"}
            )
            
            await alert_manager.add_rule(rule)
            metric_provider.get_metric.return_value = 150.0
            
            await alert_manager.start()
            await asyncio.sleep(2)
            
            # 检查通知是否发送
            assert mock_send.called
            call_args = mock_send.call_args[0]
            assert call_args[0].severity == AlertSeverity.CRITICAL
            
            await alert_manager.stop()
            
    async def test_suppression_integration(self, integrated_system):
        """测试抑制集成"""
        system = integrated_system
        alert_manager = system["alert_manager"]
        suppression_manager = system["suppression_manager"]
        metric_provider = system["metric_provider"]
        
        # 创建抑制规则
        suppression_rule = SuppressionRule(
            id="test_suppression",
            name="测试抑制",
            type=SuppressionType.LABEL_BASED,
            config={
                "match_labels": {"component": "test"},
                "duration": timedelta(minutes=10)
            },
            labels={}
        )
        
        await suppression_manager.add_rule(suppression_rule)
        
        # 创建告警规则
        alert_rule = AlertRule(
            id="suppressed_rule",
            name="被抑制的规则",
            description="会被抑制的规则",
            severity=AlertSeverity.MEDIUM,
            condition=AlertCondition.THRESHOLD,
            metric="suppressed_metric",
            threshold=10.0,
            duration=timedelta(seconds=1),
            labels={"component": "test"},
            annotations={"summary": "被抑制的告警"}
        )
        
        await alert_manager.add_rule(alert_rule)
        metric_provider.get_metric.return_value = 15.0
        
        await alert_manager.start()
        await asyncio.sleep(2)
        
        # 检查告警是否被抑制
        active_alerts = await alert_manager.get_active_alerts()
        suppressed_alerts = [a for a in active_alerts if a.status == AlertStatus.SUPPRESSED]
        assert len(suppressed_alerts) == 1
        
        await alert_manager.stop()
        
    async def test_history_integration(self, integrated_system):
        """测试历史记录集成"""
        system = integrated_system
        alert_manager = system["alert_manager"]
        alert_history = system["alert_history"]
        metric_provider = system["metric_provider"]
        
        # 创建告警规则
        rule = AlertRule(
            id="history_rule",
            name="历史记录规则",
            description="用于测试历史记录的规则",
            severity=AlertSeverity.MEDIUM,
            condition=AlertCondition.THRESHOLD,
            metric="history_metric",
            threshold=10.0,
            duration=timedelta(seconds=1),
            labels={"component": "history"},
            annotations={"summary": "历史记录告警"}
        )
        
        await alert_manager.add_rule(rule)
        metric_provider.get_metric.return_value = 15.0
        
        await alert_manager.start()
        await asyncio.sleep(2)
        
        # 获取活跃告警
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        
        alert = active_alerts[0]
        
        # 检查历史记录
        history_records = await alert_history.get_alert_history(alert_id=alert.id)
        assert len(history_records) == 1
        
        record = history_records[0]
        assert record.alert_id == alert.id
        assert record.rule_id == "history_rule"
        assert record.severity == "medium"
        
        # 检查事件记录
        events = await alert_history.get_alert_events(alert.id)
        assert len(events) >= 1
        
        create_event = next((e for e in events if e.event_type == AlertEventType.CREATED), None)
        assert create_event is not None
        assert create_event.alert_id == alert.id
        
        await alert_manager.stop()
        
    async def test_metric_provider_integration(self, integrated_system):
        """测试指标提供者集成"""
        system = integrated_system
        alert_manager = system["alert_manager"]
        metric_provider = system["metric_provider"]
        
        # 测试多个指标
        metrics = {
            "cpu_usage": 85.0,
            "memory_usage": 90.0,
            "disk_usage": 95.0
        }
        
        def get_metric_side_effect(metric_name):
            return metrics.get(metric_name, 0.0)
            
        metric_provider.get_metric.side_effect = get_metric_side_effect
        
        # 创建多个规则
        rules = [
            AlertRule(
                id="cpu_rule",
                name="CPU使用率",
                description="CPU使用率过高",
                severity=AlertSeverity.HIGH,
                condition=AlertCondition.THRESHOLD,
                metric="cpu_usage",
                threshold=80.0,
                duration=timedelta(seconds=1),
                labels={"component": "system", "type": "cpu"},
                annotations={"summary": "CPU使用率过高"}
            ),
            AlertRule(
                id="memory_rule",
                name="内存使用率",
                description="内存使用率过高",
                severity=AlertSeverity.HIGH,
                condition=AlertCondition.THRESHOLD,
                metric="memory_usage",
                threshold=85.0,
                duration=timedelta(seconds=1),
                labels={"component": "system", "type": "memory"},
                annotations={"summary": "内存使用率过高"}
            ),
            AlertRule(
                id="disk_rule",
                name="磁盘使用率",
                description="磁盘使用率过高",
                severity=AlertSeverity.CRITICAL,
                condition=AlertCondition.THRESHOLD,
                metric="disk_usage",
                threshold=90.0,
                duration=timedelta(seconds=1),
                labels={"component": "system", "type": "disk"},
                annotations={"summary": "磁盘使用率过高"}
            )
        ]
        
        for rule in rules:
            await alert_manager.add_rule(rule)
            
        await alert_manager.start()
        await asyncio.sleep(2)
        
        # 检查所有告警都被触发
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 3
        
        # 检查告警内容
        alert_by_rule = {a.rule_id: a for a in active_alerts}
        
        assert "cpu_rule" in alert_by_rule
        assert alert_by_rule["cpu_rule"].metric_value == 85.0
        
        assert "memory_rule" in alert_by_rule
        assert alert_by_rule["memory_rule"].metric_value == 90.0
        
        assert "disk_rule" in alert_by_rule
        assert alert_by_rule["disk_rule"].metric_value == 95.0
        
        await alert_manager.stop()
        
    async def test_configuration_loading(self, integrated_system):
        """测试配置加载"""
        system = integrated_system
        alert_manager = system["alert_manager"]
        
        # 加载默认配置
        config = get_default_config()
        
        # 加载告警规则
        for rule_config in config["alert_rules"]:
            rule = AlertRule(
                id=rule_config["id"],
                name=rule_config["name"],
                description=rule_config["description"],
                severity=rule_config["severity"],
                condition=rule_config["condition"],
                metric=rule_config["metric"],
                threshold=rule_config["threshold"],
                duration=rule_config["duration"],
                labels=rule_config["labels"],
                annotations=rule_config["annotations"]
            )
            await alert_manager.add_rule(rule)
            
        # 检查规则是否加载
        rules = await alert_manager.get_rules()
        assert len(rules) == len(config["alert_rules"])
        
        # 检查特定规则
        db_rule = next((r for r in rules if r.id == "database_connection_failure"), None)
        assert db_rule is not None
        assert db_rule.severity == AlertSeverity.CRITICAL
        
    async def test_error_handling(self, integrated_system):
        """测试错误处理"""
        system = integrated_system
        alert_manager = system["alert_manager"]
        metric_provider = system["metric_provider"]
        
        # 模拟指标获取失败
        metric_provider.get_metric.side_effect = Exception("指标获取失败")
        
        # 创建规则
        rule = AlertRule(
            id="error_rule",
            name="错误测试规则",
            description="用于测试错误处理",
            severity=AlertSeverity.LOW,
            condition=AlertCondition.THRESHOLD,
            metric="error_metric",
            threshold=10.0,
            duration=timedelta(seconds=1),
            labels={"component": "error"},
            annotations={"summary": "错误测试"}
        )
        
        await alert_manager.add_rule(rule)
        await alert_manager.start()
        await asyncio.sleep(2)
        
        # 检查没有告警被创建（因为指标获取失败）
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 0
        
        await alert_manager.stop()
        
    async def test_performance_under_load(self, integrated_system):
        """测试负载下的性能"""
        system = integrated_system
        alert_manager = system["alert_manager"]
        metric_provider = system["metric_provider"]
        
        # 创建大量规则
        rules = []
        for i in range(50):
            rule = AlertRule(
                id=f"load_rule_{i}",
                name=f"负载测试规则 {i}",
                description=f"负载测试规则 {i}",
                severity=AlertSeverity.LOW,
                condition=AlertCondition.THRESHOLD,
                metric=f"load_metric_{i}",
                threshold=10.0,
                duration=timedelta(seconds=1),
                labels={"component": "load", "index": str(i)},
                annotations={"summary": f"负载测试 {i}"}
            )
            rules.append(rule)
            await alert_manager.add_rule(rule)
            
        # 设置指标值
        def get_metric_side_effect(metric_name):
            if metric_name.startswith("load_metric_"):
                index = int(metric_name.split("_")[-1])
                return 15.0 if index % 2 == 0 else 5.0  # 一半触发告警
            return 0.0
            
        metric_provider.get_metric.side_effect = get_metric_side_effect
        
        # 测试启动时间
        start_time = datetime.now()
        await alert_manager.start()
        await asyncio.sleep(3)  # 等待所有规则评估
        
        # 检查告警数量
        active_alerts = await alert_manager.get_active_alerts()
        expected_alerts = 25  # 一半的规则应该触发告警
        assert len(active_alerts) == expected_alerts
        
        # 测试查询性能
        query_start = datetime.now()
        stats = await alert_manager.get_statistics()
        query_time = (datetime.now() - query_start).total_seconds()
        
        assert query_time < 1.0  # 查询应该在1秒内完成
        assert stats["total_rules"] == 50
        assert stats["active_alerts"] == expected_alerts
        
        await alert_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])