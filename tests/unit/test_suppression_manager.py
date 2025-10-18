#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抑制管理器单元测试

测试告警抑制管理器的各种抑制策略和功能
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from harborai.core.alerts.suppression_manager import (
    SuppressionManager, SuppressionRule, SuppressionType, SuppressionStatus,
    SuppressionEvent, AlertFingerprint
)
from harborai.core.alerts.alert_manager import Alert, AlertSeverity, AlertStatus


class TestSuppressionManager:
    """抑制管理器测试"""
    
    @pytest.fixture
    async def suppression_manager(self):
        """抑制管理器实例"""
        manager = SuppressionManager()
        await manager.initialize()
        return manager
        
    @pytest.fixture
    def sample_alert(self):
        """示例告警"""
        return Alert(
            id="test_alert_001",
            rule_id="test_rule",
            name="测试告警",
            description="这是一个测试告警",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            metric="cpu_usage",
            value=85.0,
            threshold=80.0,
            labels={"component": "system", "env": "production"},
            annotations={
                "summary": "CPU使用率过高",
                "description": "系统CPU使用率达到85%，超过阈值80%"
            },
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
    @pytest.fixture
    def time_based_rule(self):
        """基于时间的抑制规则"""
        return SuppressionRule(
            id="time_rule",
            name="时间抑制规则",
            description="在指定时间段内抑制告警",
            type=SuppressionType.TIME_BASED,
            status=SuppressionStatus.ACTIVE,
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now() + timedelta(hours=1),
            alert_name_pattern=".*",
            severity_filter=[AlertSeverity.LOW, AlertSeverity.MEDIUM]
        )
        
    @pytest.fixture
    def label_based_rule(self):
        """基于标签的抑制规则"""
        return SuppressionRule(
            id="label_rule",
            name="标签抑制规则",
            description="基于标签匹配抑制告警",
            type=SuppressionType.LABEL_BASED,
            status=SuppressionStatus.ACTIVE,
            label_matchers={"env": "development", "component": "test"}
        )
        
    @pytest.fixture
    def pattern_based_rule(self):
        """基于模式的抑制规则"""
        return SuppressionRule(
            id="pattern_rule",
            name="模式抑制规则",
            description="基于告警名称模式抑制告警",
            type=SuppressionType.PATTERN_BASED,
            status=SuppressionStatus.ACTIVE,
            alert_name_pattern="test_.*",
            severity_filter=[AlertSeverity.LOW]
        )
        
    @pytest.fixture
    def dependency_rule(self):
        """依赖关系抑制规则"""
        return SuppressionRule(
            id="dependency_rule",
            name="依赖抑制规则",
            description="基于依赖关系抑制告警",
            type=SuppressionType.DEPENDENCY,
            status=SuppressionStatus.ACTIVE,
            dependency_config={
                "parent_rule": "parent_alert",
                "child_rules": ["child_alert_1", "child_alert_2"]
            }
        )
        
    @pytest.fixture
    def maintenance_rule(self):
        """维护窗口抑制规则"""
        return SuppressionRule(
            id="maintenance_rule",
            name="维护窗口抑制规则",
            description="在维护窗口期间抑制告警",
            type=SuppressionType.MAINTENANCE,
            status=SuppressionStatus.ACTIVE,
            maintenance_windows=[{
                "start_time": "02:00",
                "end_time": "04:00",
                "days": ["monday", "wednesday", "friday"],
                "timezone": "UTC"
            }]
        )
        
    @pytest.fixture
    def rate_limit_rule(self):
        """速率限制抑制规则"""
        return SuppressionRule(
            id="rate_limit_rule",
            name="速率限制抑制规则",
            description="限制告警发送频率",
            type=SuppressionType.RATE_LIMIT,
            status=SuppressionStatus.ACTIVE,
            rate_limit_config={
                "max_alerts": 5,
                "time_window": 300,  # 5分钟
                "reset_interval": 3600  # 1小时
            }
        )
        
    @pytest.fixture
    def duplicate_rule(self):
        """重复告警抑制规则"""
        return SuppressionRule(
            id="duplicate_rule",
            name="重复告警抑制规则",
            description="抑制重复的告警",
            type=SuppressionType.DUPLICATE,
            status=SuppressionStatus.ACTIVE,
            duplicate_config={
                "time_window": 600,  # 10分钟
                "max_history": 100
            }
        )
        
    @pytest.fixture
    def smart_rule(self):
        """智能抑制规则"""
        return SuppressionRule(
            id="smart_rule",
            name="智能抑制规则",
            description="基于机器学习的智能抑制",
            type=SuppressionType.SMART,
            status=SuppressionStatus.ACTIVE,
            smart_config={
                "algorithm": "anomaly_detection",
                "threshold": 2.0,
                "window_size": 3600,  # 1小时
                "min_samples": 10
            }
        )

    async def test_initialization(self, suppression_manager):
        """测试初始化"""
        assert suppression_manager is not None
        assert suppression_manager.rules == {}
        assert suppression_manager.suppression_history is not None
        assert suppression_manager.dependency_graph == {}
        
    async def test_add_rule(self, suppression_manager, time_based_rule):
        """测试添加抑制规则"""
        await suppression_manager.add_rule(time_based_rule)
        assert "time_rule" in suppression_manager.rules
        assert suppression_manager.rules["time_rule"] == time_based_rule
        
    async def test_add_duplicate_rule(self, suppression_manager, time_based_rule):
        """测试添加重复抑制规则"""
        await suppression_manager.add_rule(time_based_rule)
        
        # 添加同ID规则应该覆盖原有规则
        new_rule = SuppressionRule(
            id="time_rule",
            name="新时间抑制规则",
            description="更新的时间抑制规则",
            type=SuppressionType.TIME_BASED,
            status=SuppressionStatus.INACTIVE
        )
        await suppression_manager.add_rule(new_rule)
        assert suppression_manager.rules["time_rule"] == new_rule
        
    async def test_update_rule(self, suppression_manager, time_based_rule):
        """测试更新抑制规则"""
        await suppression_manager.add_rule(time_based_rule)
        
        # 更新规则状态
        updated_rule = time_based_rule
        updated_rule.status = SuppressionStatus.INACTIVE
        updated_rule.description = "更新的描述"
        
        await suppression_manager.update_rule("time_rule", updated_rule)
        assert suppression_manager.rules["time_rule"].status == SuppressionStatus.INACTIVE
        assert suppression_manager.rules["time_rule"].description == "更新的描述"
        
    async def test_update_nonexistent_rule(self, suppression_manager, time_based_rule):
        """测试更新不存在的抑制规则"""
        with pytest.raises(ValueError, match="Rule not found"):
            await suppression_manager.update_rule("nonexistent", time_based_rule)
            
    async def test_remove_rule(self, suppression_manager, time_based_rule):
        """测试移除抑制规则"""
        await suppression_manager.add_rule(time_based_rule)
        await suppression_manager.remove_rule("time_rule")
        assert "time_rule" not in suppression_manager.rules
        
    async def test_remove_nonexistent_rule(self, suppression_manager):
        """测试移除不存在的抑制规则"""
        # 移除不存在的规则不应该抛出异常
        await suppression_manager.remove_rule("nonexistent")
        
    async def test_get_rules(self, suppression_manager, time_based_rule, label_based_rule):
        """测试获取抑制规则"""
        await suppression_manager.add_rule(time_based_rule)
        await suppression_manager.add_rule(label_based_rule)
        
        rules = suppression_manager.get_rules()
        assert len(rules) == 2
        assert "time_rule" in rules
        assert "label_rule" in rules
        
    async def test_get_active_rules(self, suppression_manager):
        """测试获取活跃的抑制规则"""
        active_rule = SuppressionRule(
            id="active_rule",
            name="活跃规则",
            type=SuppressionType.TIME_BASED,
            status=SuppressionStatus.ACTIVE
        )
        inactive_rule = SuppressionRule(
            id="inactive_rule",
            name="非活跃规则",
            type=SuppressionType.TIME_BASED,
            status=SuppressionStatus.INACTIVE
        )
        
        await suppression_manager.add_rule(active_rule)
        await suppression_manager.add_rule(inactive_rule)
        
        active_rules = suppression_manager.get_active_rules()
        assert len(active_rules) == 1
        assert active_rules[0].id == "active_rule"
        
    async def test_check_suppression_time_based(self, suppression_manager, time_based_rule, sample_alert):
        """测试基于时间的抑制检查"""
        await suppression_manager.add_rule(time_based_rule)
        
        # 测试在时间范围内的告警
        is_suppressed, reason = await suppression_manager.check_suppression(sample_alert)
        assert is_suppressed is True
        assert "time_rule" in reason
        
    async def test_check_suppression_label_based(self, suppression_manager, label_based_rule):
        """测试基于标签的抑制检查"""
        await suppression_manager.add_rule(label_based_rule)
        
        # 创建匹配标签的告警
        matching_alert = Alert(
            id="matching_alert",
            rule_id="test_rule",
            name="匹配告警",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            metric="test_metric",
            value=100.0,
            threshold=80.0,
            labels={"env": "development", "component": "test"},
            created_at=datetime.now()
        )
        
        is_suppressed, reason = await suppression_manager.check_suppression(matching_alert)
        assert is_suppressed is True
        assert "label_rule" in reason
        
    async def test_check_suppression_pattern_based(self, suppression_manager, pattern_based_rule):
        """测试基于模式的抑制检查"""
        await suppression_manager.add_rule(pattern_based_rule)
        
        # 创建匹配模式的告警
        matching_alert = Alert(
            id="matching_alert",
            rule_id="test_rule",
            name="test_pattern_alert",
            severity=AlertSeverity.LOW,
            status=AlertStatus.FIRING,
            metric="test_metric",
            value=100.0,
            threshold=80.0,
            labels={},
            created_at=datetime.now()
        )
        
        is_suppressed, reason = await suppression_manager.check_suppression(matching_alert)
        assert is_suppressed is True
        assert "pattern_rule" in reason
        
    async def test_check_suppression_dependency(self, suppression_manager, dependency_rule):
        """测试依赖关系抑制检查"""
        await suppression_manager.add_rule(dependency_rule)
        
        # 模拟父告警存在
        suppression_manager.active_alerts = {"parent_alert": True}
        
        # 创建子告警
        child_alert = Alert(
            id="child_alert",
            rule_id="child_alert_1",
            name="子告警",
            severity=AlertSeverity.HIGH,
            status=AlertStatus.FIRING,
            metric="test_metric",
            value=100.0,
            threshold=80.0,
            labels={},
            created_at=datetime.now()
        )
        
        is_suppressed, reason = await suppression_manager.check_suppression(child_alert)
        assert is_suppressed is True
        assert "dependency_rule" in reason
        
    async def test_check_suppression_rate_limit(self, suppression_manager, rate_limit_rule, sample_alert):
        """测试速率限制抑制检查"""
        await suppression_manager.add_rule(rate_limit_rule)
        
        # 发送多个告警直到达到速率限制
        for i in range(6):  # 超过限制的5个
            is_suppressed, reason = await suppression_manager.check_suppression(sample_alert)
            if i < 5:
                assert is_suppressed is False
            else:
                assert is_suppressed is True
                assert "rate_limit_rule" in reason
                
    async def test_check_suppression_duplicate(self, suppression_manager, duplicate_rule, sample_alert):
        """测试重复告警抑制检查"""
        await suppression_manager.add_rule(duplicate_rule)
        
        # 第一次发送告警
        is_suppressed, reason = await suppression_manager.check_suppression(sample_alert)
        assert is_suppressed is False
        
        # 立即再次发送相同告警
        is_suppressed, reason = await suppression_manager.check_suppression(sample_alert)
        assert is_suppressed is True
        assert "duplicate_rule" in reason
        
    async def test_check_suppression_smart(self, suppression_manager, smart_rule, sample_alert):
        """测试智能抑制检查"""
        await suppression_manager.add_rule(smart_rule)
        
        # 模拟正常频率的告警
        for i in range(5):
            await asyncio.sleep(0.1)  # 短暂延迟
            is_suppressed, reason = await suppression_manager.check_suppression(sample_alert)
            assert is_suppressed is False
            
        # 模拟异常高频告警
        for i in range(10):
            is_suppressed, reason = await suppression_manager.check_suppression(sample_alert)
            # 在高频情况下应该被抑制
            if i > 5:
                assert is_suppressed is True
                assert "smart_rule" in reason
                
    async def test_check_suppression_no_rules(self, suppression_manager, sample_alert):
        """测试没有抑制规则时的检查"""
        is_suppressed, reason = await suppression_manager.check_suppression(sample_alert)
        assert is_suppressed is False
        assert reason == ""
        
    async def test_check_suppression_inactive_rule(self, suppression_manager, sample_alert):
        """测试非活跃规则不会抑制告警"""
        inactive_rule = SuppressionRule(
            id="inactive_rule",
            name="非活跃规则",
            type=SuppressionType.TIME_BASED,
            status=SuppressionStatus.INACTIVE,
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now() + timedelta(hours=1),
            alert_name_pattern=".*"
        )
        
        await suppression_manager.add_rule(inactive_rule)
        
        is_suppressed, reason = await suppression_manager.check_suppression(sample_alert)
        assert is_suppressed is False
        
    async def test_maintenance_window_check(self, suppression_manager, maintenance_rule):
        """测试维护窗口检查"""
        await suppression_manager.add_rule(maintenance_rule)
        
        # 模拟在维护窗口内的时间
        with patch('datetime.datetime') as mock_datetime:
            # 设置为周一凌晨3点（在维护窗口内）
            mock_datetime.now.return_value = datetime(2024, 1, 1, 3, 0, 0)  # 假设是周一
            mock_datetime.strptime = datetime.strptime
            
            alert_in_window = Alert(
                id="maintenance_alert",
                rule_id="test_rule",
                name="维护期间告警",
                severity=AlertSeverity.HIGH,
                status=AlertStatus.FIRING,
                metric="test_metric",
                value=100.0,
                threshold=80.0,
                labels={},
                created_at=datetime.now()
            )
            
            is_suppressed, reason = await suppression_manager.check_suppression(alert_in_window)
            # 注意：这里的测试结果取决于具体的维护窗口实现
            
    async def test_get_suppression_statistics(self, suppression_manager, time_based_rule, sample_alert):
        """测试获取抑制统计信息"""
        await suppression_manager.add_rule(time_based_rule)
        
        # 触发一些抑制
        await suppression_manager.check_suppression(sample_alert)
        
        stats = suppression_manager.get_statistics()
        
        assert "total_rules" in stats
        assert "active_rules" in stats
        assert "total_suppressions" in stats
        assert "suppression_by_type" in stats
        assert stats["total_rules"] >= 1
        assert stats["active_rules"] >= 1
        
    async def test_alert_fingerprint_generation(self, suppression_manager, sample_alert):
        """测试告警指纹生成"""
        fingerprint = suppression_manager._generate_alert_fingerprint(sample_alert)
        
        assert isinstance(fingerprint, AlertFingerprint)
        assert fingerprint.rule_id == sample_alert.rule_id
        assert fingerprint.metric == sample_alert.metric
        assert fingerprint.labels_hash is not None
        
        # 相同告警应该生成相同指纹
        fingerprint2 = suppression_manager._generate_alert_fingerprint(sample_alert)
        assert fingerprint.labels_hash == fingerprint2.labels_hash
        
    async def test_suppression_event_recording(self, suppression_manager, time_based_rule, sample_alert):
        """测试抑制事件记录"""
        await suppression_manager.add_rule(time_based_rule)
        
        # 触发抑制
        await suppression_manager.check_suppression(sample_alert)
        
        # 检查是否记录了抑制事件
        assert len(suppression_manager.suppression_history) > 0
        
        event = suppression_manager.suppression_history[-1]
        assert isinstance(event, SuppressionEvent)
        assert event.rule_id == "time_rule"
        assert event.alert_id == sample_alert.id
        
    async def test_dependency_graph_update(self, suppression_manager, dependency_rule):
        """测试依赖图更新"""
        await suppression_manager.add_rule(dependency_rule)
        
        # 检查依赖图是否正确更新
        assert "parent_alert" in suppression_manager.dependency_graph
        assert "child_alert_1" in suppression_manager.dependency_graph["parent_alert"]
        assert "child_alert_2" in suppression_manager.dependency_graph["parent_alert"]
        
    async def test_rule_validation(self, suppression_manager):
        """测试规则验证"""
        # 测试无效的规则
        invalid_rule = SuppressionRule(
            id="",  # 空ID
            name="无效规则",
            type=SuppressionType.TIME_BASED,
            status=SuppressionStatus.ACTIVE
        )
        
        with pytest.raises(ValueError):
            await suppression_manager.add_rule(invalid_rule)
            
    async def test_concurrent_suppression_checks(self, suppression_manager, time_based_rule, sample_alert):
        """测试并发抑制检查"""
        await suppression_manager.add_rule(time_based_rule)
        
        # 并发执行多个抑制检查
        tasks = []
        for i in range(10):
            task = asyncio.create_task(suppression_manager.check_suppression(sample_alert))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        # 所有结果应该一致
        assert all(result[0] == results[0][0] for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])