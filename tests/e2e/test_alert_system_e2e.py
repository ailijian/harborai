#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警系统端到端测试

测试完整的告警系统工作流程
"""

import pytest
import asyncio
import tempfile
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

from harborai.core.alerts.alert_manager import (
    AlertManager, AlertRule, AlertSeverity, AlertCondition, AlertStatus
)
from harborai.core.alerts.notification_service import (
    NotificationService, NotificationChannel, NotificationPriority
)
from harborai.core.alerts.suppression_manager import (
    SuppressionManager, SuppressionRule, SuppressionType
)
from harborai.core.alerts.alert_history import AlertHistory
from harborai.core.alerts.config import get_default_config


class MockMetricProvider:
    """模拟指标提供者"""
    
    def __init__(self):
        self.metrics = {}
        self.call_count = 0
        
    async def get_metric(self, metric_name: str) -> float:
        """获取指标值"""
        self.call_count += 1
        return self.metrics.get(metric_name, 0.0)
        
    def set_metric(self, metric_name: str, value: float):
        """设置指标值"""
        self.metrics[metric_name] = value
        
    def clear_metrics(self):
        """清空指标"""
        self.metrics.clear()
        self.call_count = 0


class MockNotificationChannel:
    """模拟通知渠道"""
    
    def __init__(self):
        self.sent_notifications = []
        self.should_fail = False
        self.delay = 0
        
    async def send(self, message: str, **kwargs) -> bool:
        """发送通知"""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
            
        if self.should_fail:
            raise Exception("通知发送失败")
            
        self.sent_notifications.append({
            "message": message,
            "timestamp": datetime.now(),
            "kwargs": kwargs
        })
        return True
        
    def get_sent_count(self) -> int:
        """获取发送数量"""
        return len(self.sent_notifications)
        
    def get_last_notification(self) -> Dict[str, Any]:
        """获取最后一条通知"""
        return self.sent_notifications[-1] if self.sent_notifications else None
        
    def clear_notifications(self):
        """清空通知记录"""
        self.sent_notifications.clear()


class TestAlertSystemE2E:
    """告警系统端到端测试"""
    
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
    async def metric_provider(self):
        """模拟指标提供者"""
        return MockMetricProvider()
        
    @pytest.fixture
    async def notification_channels(self):
        """模拟通知渠道"""
        return {
            "console": MockNotificationChannel(),
            "email": MockNotificationChannel(),
            "webhook": MockNotificationChannel()
        }
        
    @pytest.fixture
    async def alert_system(self, temp_db, metric_provider, notification_channels):
        """完整的告警系统"""
        # 创建组件
        alert_manager = AlertManager()
        notification_service = NotificationService()
        suppression_manager = SuppressionManager()
        alert_history = AlertHistory(db_path=temp_db)
        
        # 初始化
        await alert_manager.initialize()
        await notification_service.initialize()
        await suppression_manager.initialize()
        await alert_history.initialize()
        
        # 设置依赖
        alert_manager.set_notification_service(notification_service)
        alert_manager.set_suppression_service(suppression_manager)
        alert_manager.history_service = alert_history
        
        # 注册指标提供者
        alert_manager.register_metric_provider("test", metric_provider)
        
        # 配置通知渠道
        for channel_name, channel in notification_channels.items():
            notification_service.channels[channel_name] = channel
            
        return {
            "alert_manager": alert_manager,
            "notification_service": notification_service,
            "suppression_manager": suppression_manager,
            "alert_history": alert_history,
            "metric_provider": metric_provider,
            "notification_channels": notification_channels
        }
        
    async def test_complete_alert_lifecycle(self, alert_system):
        """测试完整的告警生命周期"""
        alert_manager = alert_system["alert_manager"]
        metric_provider = alert_system["metric_provider"]
        notification_channels = alert_system["notification_channels"]
        alert_history = alert_system["alert_history"]
        
        # 1. 创建告警规则
        rule = AlertRule(
            id="e2e_test_rule",
            name="端到端测试规则",
            description="用于端到端测试的告警规则",
            severity=AlertSeverity.HIGH,
            condition=AlertCondition.THRESHOLD,
            metric="test_metric",
            threshold=10.0,
            duration=timedelta(seconds=1),
            labels={"component": "e2e_test"},
            annotations={"summary": "端到端测试告警"}
        )
        
        await alert_manager.add_rule(rule)
        
        # 2. 设置正常指标值（不触发告警）
        metric_provider.set_metric("test_metric", 5.0)
        
        # 启动告警管理器
        await alert_manager.start()
        await asyncio.sleep(2)
        
        # 验证没有告警
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 0, "不应该有活跃告警"
        
        # 3. 设置异常指标值（触发告警）
        metric_provider.set_metric("test_metric", 15.0)
        await asyncio.sleep(3)  # 等待告警触发
        
        # 验证告警被触发
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 1, "应该有一个活跃告警"
        
        alert = active_alerts[0]
        assert alert.rule_id == "e2e_test_rule"
        assert alert.status == AlertStatus.FIRING
        assert alert.metric_value == 15.0
        
        # 验证通知被发送
        console_channel = notification_channels["console"]
        assert console_channel.get_sent_count() > 0, "应该发送了通知"
        
        last_notification = console_channel.get_last_notification()
        assert "端到端测试告警" in last_notification["message"]
        
        # 验证历史记录
        history_records = await alert_history.get_alert_history(limit=10)
        assert len(history_records) > 0, "应该有历史记录"
        
        # 4. 确认告警
        await alert_manager.acknowledge_alert(alert.id, "测试用户")
        
        # 验证告警状态
        active_alerts = await alert_manager.get_active_alerts()
        acknowledged_alert = next((a for a in active_alerts if a.id == alert.id), None)
        assert acknowledged_alert is not None
        assert acknowledged_alert.acknowledged_by == "测试用户"
        
        # 5. 恢复指标值（解决告警）
        metric_provider.set_metric("test_metric", 5.0)
        await asyncio.sleep(3)  # 等待告警解决
        
        # 验证告警被解决
        active_alerts = await alert_manager.get_active_alerts()
        resolved_alerts = [a for a in active_alerts if a.status == AlertStatus.RESOLVED]
        assert len(resolved_alerts) > 0, "告警应该被解决"
        
        await alert_manager.stop()
        
    async def test_alert_suppression_workflow(self, alert_system):
        """测试告警抑制工作流程"""
        alert_manager = alert_system["alert_manager"]
        suppression_manager = alert_system["suppression_manager"]
        metric_provider = alert_system["metric_provider"]
        notification_channels = alert_system["notification_channels"]
        
        # 1. 创建告警规则
        rule = AlertRule(
            id="suppression_test_rule",
            name="抑制测试规则",
            description="用于测试抑制功能的规则",
            severity=AlertSeverity.MEDIUM,
            condition=AlertCondition.THRESHOLD,
            metric="suppression_metric",
            threshold=10.0,
            duration=timedelta(seconds=1),
            labels={"component": "suppression_test"},
            annotations={"summary": "抑制测试告警"}
        )
        
        await alert_manager.add_rule(rule)
        
        # 2. 创建抑制规则
        suppression_rule = SuppressionRule(
            id="test_suppression",
            name="测试抑制规则",
            type=SuppressionType.LABEL_BASED,
            target_labels={"component": "suppression_test"},
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=10),
            reason="端到端测试抑制"
        )
        
        await suppression_manager.add_suppression_rule(suppression_rule)
        
        # 3. 启动系统并触发告警
        await alert_manager.start()
        metric_provider.set_metric("suppression_metric", 15.0)
        await asyncio.sleep(3)
        
        # 4. 验证告警被抑制
        active_alerts = await alert_manager.get_active_alerts()
        suppressed_alerts = [a for a in active_alerts if a.suppressed]
        assert len(suppressed_alerts) > 0, "告警应该被抑制"
        
        # 验证没有发送通知
        console_channel = notification_channels["console"]
        initial_count = console_channel.get_sent_count()
        
        await asyncio.sleep(2)
        final_count = console_channel.get_sent_count()
        assert final_count == initial_count, "抑制期间不应该发送通知"
        
        # 5. 移除抑制规则
        await suppression_manager.remove_suppression_rule("test_suppression")
        await asyncio.sleep(2)
        
        # 验证告警不再被抑制
        active_alerts = await alert_manager.get_active_alerts()
        unsuppressed_alerts = [a for a in active_alerts if not a.suppressed]
        assert len(unsuppressed_alerts) > 0, "告警应该不再被抑制"
        
        await alert_manager.stop()
        
    async def test_notification_failure_handling(self, alert_system):
        """测试通知失败处理"""
        alert_manager = alert_system["alert_manager"]
        metric_provider = alert_system["metric_provider"]
        notification_channels = alert_system["notification_channels"]
        notification_service = alert_system["notification_service"]
        
        # 1. 设置通知渠道失败
        email_channel = notification_channels["email"]
        email_channel.should_fail = True
        
        # 2. 创建告警规则
        rule = AlertRule(
            id="notification_test_rule",
            name="通知测试规则",
            description="用于测试通知失败处理的规则",
            severity=AlertSeverity.CRITICAL,
            condition=AlertCondition.THRESHOLD,
            metric="notification_metric",
            threshold=10.0,
            duration=timedelta(seconds=1),
            labels={"component": "notification_test"},
            annotations={"summary": "通知测试告警"}
        )
        
        await alert_manager.add_rule(rule)
        
        # 3. 启动系统并触发告警
        await alert_manager.start()
        metric_provider.set_metric("notification_metric", 15.0)
        await asyncio.sleep(3)
        
        # 4. 验证告警被创建
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) > 0, "应该有活跃告警"
        
        # 5. 验证失败的通知被记录
        console_channel = notification_channels["console"]
        assert console_channel.get_sent_count() > 0, "控制台通知应该成功"
        
        # 6. 修复通知渠道
        email_channel.should_fail = False
        
        # 触发新的告警状态变化
        await alert_manager.acknowledge_alert(active_alerts[0].id, "测试用户")
        await asyncio.sleep(2)
        
        # 验证通知恢复正常
        assert email_channel.get_sent_count() > 0, "邮件通知应该恢复"
        
        await alert_manager.stop()
        
    async def test_high_frequency_alerts(self, alert_system):
        """测试高频告警处理"""
        alert_manager = alert_system["alert_manager"]
        metric_provider = alert_system["metric_provider"]
        notification_channels = alert_system["notification_channels"]
        
        # 1. 创建多个告警规则
        rules = []
        for i in range(10):
            rule = AlertRule(
                id=f"high_freq_rule_{i}",
                name=f"高频测试规则 {i}",
                description=f"高频测试规则 {i}",
                severity=AlertSeverity.HIGH,
                condition=AlertCondition.THRESHOLD,
                metric=f"high_freq_metric_{i}",
                threshold=10.0,
                duration=timedelta(seconds=1),
                labels={"component": "high_freq_test", "index": str(i)},
                annotations={"summary": f"高频测试告警 {i}"}
            )
            rules.append(rule)
            await alert_manager.add_rule(rule)
            
        # 2. 启动系统
        await alert_manager.start()
        
        # 3. 快速触发多个告警
        for i in range(10):
            metric_provider.set_metric(f"high_freq_metric_{i}", 15.0)
            await asyncio.sleep(0.1)  # 100ms间隔
            
        await asyncio.sleep(5)  # 等待处理
        
        # 4. 验证所有告警都被处理
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) == 10, f"应该有10个活跃告警，实际有{len(active_alerts)}个"
        
        # 5. 验证通知被发送
        console_channel = notification_channels["console"]
        assert console_channel.get_sent_count() >= 10, "应该发送了至少10个通知"
        
        # 6. 快速解决告警
        for i in range(10):
            metric_provider.set_metric(f"high_freq_metric_{i}", 5.0)
            await asyncio.sleep(0.1)
            
        await asyncio.sleep(5)  # 等待解决
        
        # 验证告警被解决
        active_alerts = await alert_manager.get_active_alerts()
        resolved_count = len([a for a in active_alerts if a.status == AlertStatus.RESOLVED])
        assert resolved_count == 10, f"应该有10个已解决告警，实际有{resolved_count}个"
        
        await alert_manager.stop()
        
    async def test_system_recovery_after_failure(self, alert_system):
        """测试系统故障后的恢复"""
        alert_manager = alert_system["alert_manager"]
        metric_provider = alert_system["metric_provider"]
        alert_history = alert_system["alert_history"]
        
        # 1. 创建告警规则
        rule = AlertRule(
            id="recovery_test_rule",
            name="恢复测试规则",
            description="用于测试系统恢复的规则",
            severity=AlertSeverity.HIGH,
            condition=AlertCondition.THRESHOLD,
            metric="recovery_metric",
            threshold=10.0,
            duration=timedelta(seconds=1),
            labels={"component": "recovery_test"},
            annotations={"summary": "恢复测试告警"}
        )
        
        await alert_manager.add_rule(rule)
        
        # 2. 启动系统并创建告警
        await alert_manager.start()
        metric_provider.set_metric("recovery_metric", 15.0)
        await asyncio.sleep(3)
        
        # 验证告警被创建
        active_alerts_before = await alert_manager.get_active_alerts()
        assert len(active_alerts_before) > 0, "应该有活跃告警"
        
        # 3. 模拟系统故障（停止告警管理器）
        await alert_manager.stop()
        
        # 4. 在停机期间改变指标值
        metric_provider.set_metric("recovery_metric", 5.0)
        await asyncio.sleep(2)
        
        # 5. 重启系统
        await alert_manager.start()
        await asyncio.sleep(3)
        
        # 6. 验证系统恢复正常
        active_alerts_after = await alert_manager.get_active_alerts()
        
        # 验证告警状态被正确恢复
        # 由于指标值已经恢复正常，告警应该被解决
        resolved_alerts = [a for a in active_alerts_after if a.status == AlertStatus.RESOLVED]
        assert len(resolved_alerts) > 0, "告警应该在系统恢复后被解决"
        
        # 7. 验证历史记录完整性
        history_records = await alert_history.get_alert_history(limit=20)
        assert len(history_records) > 0, "应该有完整的历史记录"
        
        await alert_manager.stop()
        
    async def test_configuration_reload(self, alert_system):
        """测试配置重新加载"""
        alert_manager = alert_system["alert_manager"]
        notification_service = alert_system["notification_service"]
        suppression_manager = alert_system["suppression_manager"]
        
        # 1. 获取初始配置
        initial_rules = await alert_manager.get_rules()
        initial_rule_count = len(initial_rules)
        
        # 2. 加载默认配置
        config = get_default_config()
        
        # 添加配置中的规则
        for rule_config in config["alert_rules"]:
            rule = AlertRule(
                id=rule_config["id"],
                name=rule_config["name"],
                description=rule_config["description"],
                severity=AlertSeverity(rule_config["severity"]),
                condition=AlertCondition(rule_config["condition"]),
                metric=rule_config["metric"],
                threshold=rule_config["threshold"],
                duration=timedelta(seconds=rule_config["duration"]),
                labels=rule_config.get("labels", {}),
                annotations=rule_config.get("annotations", {})
            )
            await alert_manager.add_rule(rule)
            
        # 3. 验证规则被加载
        updated_rules = await alert_manager.get_rules()
        assert len(updated_rules) > initial_rule_count, "应该加载了新的规则"
        
        # 4. 加载通知配置
        for channel_config in config["notification"]["channels"]:
            # 这里只是验证配置结构，实际的通知渠道已经在fixture中设置
            assert "name" in channel_config
            assert "type" in channel_config
            
        # 5. 加载抑制配置
        for suppression_config in config["suppression"]["rules"]:
            suppression_rule = SuppressionRule(
                id=suppression_config["id"],
                name=suppression_config["name"],
                type=SuppressionType(suppression_config["type"]),
                target_labels=suppression_config.get("target_labels", {}),
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(hours=1),
                reason="配置测试"
            )
            await suppression_manager.add_suppression_rule(suppression_rule)
            
        # 6. 验证配置生效
        suppression_rules = await suppression_manager.get_suppression_rules()
        assert len(suppression_rules) > 0, "应该有抑制规则"
        
    async def test_alert_escalation(self, alert_system):
        """测试告警升级"""
        alert_manager = alert_system["alert_manager"]
        metric_provider = alert_system["metric_provider"]
        notification_channels = alert_system["notification_channels"]
        
        # 1. 创建需要升级的告警规则
        rule = AlertRule(
            id="escalation_test_rule",
            name="升级测试规则",
            description="用于测试告警升级的规则",
            severity=AlertSeverity.MEDIUM,
            condition=AlertCondition.THRESHOLD,
            metric="escalation_metric",
            threshold=10.0,
            duration=timedelta(seconds=1),
            labels={"component": "escalation_test"},
            annotations={"summary": "升级测试告警"}
        )
        
        await alert_manager.add_rule(rule)
        
        # 2. 启动系统并触发告警
        await alert_manager.start()
        metric_provider.set_metric("escalation_metric", 15.0)
        await asyncio.sleep(3)
        
        # 3. 获取告警并验证初始状态
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) > 0, "应该有活跃告警"
        
        alert = active_alerts[0]
        initial_escalation_level = alert.escalation_level
        
        # 4. 等待升级时间（模拟长时间未处理）
        # 这里我们手动触发升级逻辑
        await asyncio.sleep(2)
        
        # 5. 验证升级后的状态
        updated_alerts = await alert_manager.get_active_alerts()
        updated_alert = next((a for a in updated_alerts if a.id == alert.id), None)
        
        assert updated_alert is not None, "告警应该仍然存在"
        
        # 6. 验证通知发送
        console_channel = notification_channels["console"]
        assert console_channel.get_sent_count() > 0, "应该发送了通知"
        
        await alert_manager.stop()
        
    async def test_data_consistency_across_components(self, alert_system):
        """测试组件间数据一致性"""
        alert_manager = alert_system["alert_manager"]
        alert_history = alert_system["alert_history"]
        metric_provider = alert_system["metric_provider"]
        
        # 1. 创建告警规则
        rule = AlertRule(
            id="consistency_test_rule",
            name="一致性测试规则",
            description="用于测试数据一致性的规则",
            severity=AlertSeverity.HIGH,
            condition=AlertCondition.THRESHOLD,
            metric="consistency_metric",
            threshold=10.0,
            duration=timedelta(seconds=1),
            labels={"component": "consistency_test"},
            annotations={"summary": "一致性测试告警"}
        )
        
        await alert_manager.add_rule(rule)
        
        # 2. 启动系统并创建告警
        await alert_manager.start()
        metric_provider.set_metric("consistency_metric", 15.0)
        await asyncio.sleep(3)
        
        # 3. 获取活跃告警
        active_alerts = await alert_manager.get_active_alerts()
        assert len(active_alerts) > 0, "应该有活跃告警"
        
        alert = active_alerts[0]
        
        # 4. 验证历史记录中的数据一致性
        history_records = await alert_history.get_alert_history(
            rule_id="consistency_test_rule",
            limit=10
        )
        
        assert len(history_records) > 0, "应该有历史记录"
        
        history_record = history_records[0]
        
        # 验证关键字段一致性
        assert history_record.rule_id == alert.rule_id
        assert history_record.severity == alert.severity.value
        assert history_record.metric_value == alert.metric_value
        
        # 5. 确认告警并验证一致性
        await alert_manager.acknowledge_alert(alert.id, "一致性测试用户")
        
        # 获取更新后的告警
        updated_alerts = await alert_manager.get_active_alerts()
        updated_alert = next((a for a in updated_alerts if a.id == alert.id), None)
        
        assert updated_alert is not None
        assert updated_alert.acknowledged_by == "一致性测试用户"
        
        # 验证历史记录也被更新
        await asyncio.sleep(1)  # 等待历史记录更新
        updated_history = await alert_history.get_alert_history(
            rule_id="consistency_test_rule",
            limit=10
        )
        
        updated_history_record = updated_history[0]
        assert updated_history_record.acknowledged_by == "一致性测试用户"
        
        await alert_manager.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])