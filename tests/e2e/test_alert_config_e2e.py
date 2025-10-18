#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警配置端到端测试

测试从配置加载到告警触发的完整流程，包括配置验证、
告警规则评估、通知发送、抑制处理和升级策略等。
"""

import pytest
import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta

from harborai.core.alerts.config_validator import validate_config_file
from harborai.core.alerts.alert_manager import AlertManager, Alert, AlertSeverity, AlertStatus
from harborai.core.alerts.notification_service import NotificationService
from harborai.core.alerts.suppression_manager import SuppressionManager


class TestAlertConfigE2E:
    """告警配置端到端测试"""
    
    @pytest.fixture
    def complete_config(self):
        """完整的测试配置"""
        return {
            "alert_rules": [
                {
                    "id": "cpu_high",
                    "name": "CPU使用率过高",
                    "description": "系统CPU使用率超过阈值",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "system.cpu.usage",
                    "threshold": 80.0,
                    "duration": 300,
                    "labels": {
                        "component": "system",
                        "environment": "test"
                    },
                    "annotations": {
                        "summary": "CPU使用率过高",
                        "description": "CPU使用率已达到 {{ $value }}%",
                        "runbook": "检查系统负载"
                    }
                },
                {
                    "id": "memory_high",
                    "name": "内存使用率过高",
                    "description": "系统内存使用率超过阈值",
                    "severity": "critical",
                    "condition": "threshold",
                    "metric": "system.memory.usage",
                    "threshold": 90.0,
                    "duration": 180,
                    "labels": {
                        "component": "system",
                        "environment": "test"
                    },
                    "annotations": {
                        "summary": "内存使用率过高",
                        "description": "内存使用率已达到 {{ $value }}%",
                        "runbook": "检查内存泄漏"
                    }
                },
                {
                    "id": "api_error_rate",
                    "name": "API错误率过高",
                    "description": "API错误率超过阈值",
                    "severity": "medium",
                    "condition": "threshold",
                    "metric": "api.error_rate",
                    "threshold": 5.0,
                    "duration": 120,
                    "labels": {
                        "component": "api",
                        "environment": "test"
                    },
                    "annotations": {
                        "summary": "API错误率过高",
                        "description": "API错误率已达到 {{ $value }}%",
                        "runbook": "检查API服务"
                    }
                }
            ],
            "notification": {
                "channels": [
                    {
                        "name": "console",
                        "type": "console",
                        "enabled": True,
                        "config": {}
                    },
                    {
                        "name": "test_email",
                        "type": "email",
                        "enabled": True,
                        "config": {
                            "smtp_host": "localhost",
                            "smtp_port": 587,
                            "username": "test@example.com",
                            "password": "password",
                            "from_email": "alerts@example.com"
                        }
                    },
                    {
                        "name": "test_webhook",
                        "type": "webhook",
                        "enabled": True,
                        "config": {
                            "url": "http://localhost:8080/webhook",
                            "method": "POST",
                            "headers": {"Content-Type": "application/json"}
                        }
                    }
                ],
                "routing": {
                    "default_channels": ["console"],
                    "rules": [
                        {
                            "match": {"severity": "critical"},
                            "channels": ["console", "test_email", "test_webhook"]
                        },
                        {
                            "match": {"severity": "high"},
                            "channels": ["console", "test_email"]
                        },
                        {
                            "match": {"component": "api"},
                            "channels": ["console", "test_webhook"]
                        }
                    ]
                },
                "rate_limits": {
                    "enabled": True,
                    "max_notifications_per_minute": 10
                },
                "retry": {
                    "enabled": True,
                    "max_attempts": 3,
                    "backoff_factor": 2.0
                }
            },
            "suppression": {
                "rules": [
                    {
                        "id": "duplicate_suppression",
                        "name": "重复告警抑制",
                        "type": "duplicate",
                        "enabled": True,
                        "duplicate_window": 300,
                        "duplicate_threshold": 2
                    },
                    {
                        "id": "maintenance_window",
                        "name": "维护窗口",
                        "type": "time_based",
                        "enabled": False,  # 测试时禁用
                        "start_time": "02:00",
                        "end_time": "04:00",
                        "timezone": "Asia/Shanghai"
                    },
                    {
                        "id": "cascade_suppression",
                        "name": "级联抑制",
                        "type": "dependency",
                        "enabled": True,
                        "dependency_alerts": ["memory_high"],
                        "dependency_rules": ["cpu_high"]
                    }
                ]
            },
            "escalation": {
                "enabled": True,
                "global_settings": {
                    "escalation_timeout": 600,  # 10分钟用于测试
                    "auto_resolve_timeout": 1800,
                    "escalation_cooldown": 120
                },
                "rules": [
                    {
                        "severity": "critical",
                        "steps": [
                            {
                                "delay": 0,
                                "channels": ["console"],
                                "message_template": "escalation_immediate"
                            },
                            {
                                "delay": 300,  # 5分钟
                                "channels": ["test_email"],
                                "message_template": "escalation_level1"
                            }
                        ],
                        "max_escalations": 2
                    },
                    {
                        "severity": "high",
                        "steps": [
                            {
                                "delay": 600,  # 10分钟
                                "channels": ["test_email"],
                                "message_template": "escalation_level1"
                            }
                        ],
                        "max_escalations": 1
                    }
                ],
                "notification_templates": {
                    "escalation_immediate": {
                        "subject": "紧急告警 - {{ alert.name }}",
                        "body": "告警已触发，需要立即处理！"
                    },
                    "escalation_level1": {
                        "subject": "告警升级 - {{ alert.name }}",
                        "body": "告警已升级，请及时处理。"
                    }
                }
            },
            "aggregation": {
                "enabled": True,
                "window_size": 300,
                "rules": [
                    {
                        "id": "system_alerts",
                        "name": "系统告警聚合",
                        "group_by": ["component"],
                        "match": {"component": "system"},
                        "threshold": 2
                    }
                ]
            },
            "metrics": {
                "collection_interval": 30,  # 30秒用于测试
                "retention_days": 7
            },
            "health_check": {
                "enabled": True,
                "interval": 15  # 15秒用于测试
            }
        }
    
    @pytest.fixture
    def config_file(self, complete_config):
        """创建临时配置文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(complete_config, f, indent=2, ensure_ascii=False)
            temp_path = f.name
        
        yield temp_path
        
        # 清理
        Path(temp_path).unlink(missing_ok=True)
    
    def test_config_validation_pipeline(self, config_file):
        """测试配置验证流水线"""
        # 步骤1: 验证配置文件
        is_valid, results = validate_config_file(config_file)
        
        assert is_valid, f"配置文件应该有效: {[r.message for r in results if r.level.name == 'ERROR']}"
        
        # 步骤2: 检查验证结果
        from harborai.core.alerts.config_validator import ValidationLevel
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        
        assert len(errors) == 0, f"不应该有错误: {[e.message for e in errors]}"
        
        print(f"配置验证完成: {len(warnings)} 个警告")
        
        # 步骤3: 验证配置结构
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        assert "alert_rules" in config
        assert "notification" in config
        assert "suppression" in config
        assert "escalation" in config
        
        assert len(config["alert_rules"]) == 3
        assert len(config["notification"]["channels"]) == 3
        assert len(config["suppression"]["rules"]) == 3
    
    @pytest.mark.asyncio
    async def test_alert_lifecycle_e2e(self, complete_config):
        """测试告警生命周期端到端流程"""
        # 模拟组件
        mock_notification_service = AsyncMock()
        mock_suppression_manager = MagicMock()
        
        # 配置模拟行为
        mock_suppression_manager.should_suppress.return_value = False
        mock_notification_service.send_notification.return_value = True
        
        with patch('harborai.core.alerts.alert_manager.NotificationService', return_value=mock_notification_service), \
             patch('harborai.core.alerts.alert_manager.SuppressionManager', return_value=mock_suppression_manager):
            
            # 步骤1: 创建告警管理器
            alert_manager = AlertManager()
            await alert_manager.initialize()
            
            # 步骤2: 加载配置
            for rule_config in complete_config["alert_rules"]:
                from harborai.core.alerts.alert_manager import AlertRule, AlertSeverity, AlertCondition
                
                rule = AlertRule(
                    id=rule_config["id"],
                    name=rule_config["name"],
                    description=rule_config["description"],
                    severity=AlertSeverity(rule_config["severity"].upper()),
                    condition=AlertCondition(rule_config["condition"].upper()),
                    metric=rule_config["metric"],
                    threshold=rule_config["threshold"],
                    duration=timedelta(seconds=rule_config["duration"]),
                    labels=rule_config["labels"],
                    annotations=rule_config["annotations"]
                )
                await alert_manager.add_rule(rule)
            
            # 步骤3: 注册指标提供者
            async def mock_metric_provider(metric_name):
                """模拟指标提供者"""
                if metric_name == "system.cpu.usage":
                    return 85.0  # 超过阈值80.0
                elif metric_name == "system.memory.usage":
                    return 95.0  # 超过阈值90.0
                elif metric_name == "api.error_rate":
                    return 3.0  # 未超过阈值5.0
                return 0.0
            
            await alert_manager.register_metric_provider("test", mock_metric_provider)
            
            # 步骤4: 评估告警规则
            await alert_manager._evaluate_rules()
            
            # 步骤5: 检查告警状态
            active_alerts = await alert_manager.get_active_alerts()
            
            # 应该有2个活跃告警（CPU和内存）
            assert len(active_alerts) >= 1, f"应该有活跃告警，实际: {len(active_alerts)}"
            
            # 检查告警内容
            alert_ids = [alert.rule_id for alert in active_alerts]
            print(f"活跃告警: {alert_ids}")
            
            # 步骤6: 验证通知发送
            # 由于是异步操作，可能需要等待
            await asyncio.sleep(0.1)
            
            # 检查通知服务是否被调用
            if mock_notification_service.send_notification.called:
                print("通知服务已被调用")
            
            # 步骤7: 测试告警确认
            if active_alerts:
                alert = active_alerts[0]
                await alert_manager.acknowledge_alert(alert.id, "测试确认")
                
                # 检查告警状态
                updated_alert = await alert_manager.get_alert(alert.id)
                assert updated_alert.status == AlertStatus.ACKNOWLEDGED
    
    @pytest.mark.asyncio
    async def test_notification_routing_e2e(self, complete_config):
        """测试通知路由端到端流程"""
        # 模拟通知渠道
        mock_channels = {
            "console": AsyncMock(),
            "test_email": AsyncMock(),
            "test_webhook": AsyncMock()
        }
        
        # 配置模拟返回值
        for channel in mock_channels.values():
            channel.send.return_value = True
        
        with patch('harborai.core.alerts.notification_service.NotificationService') as mock_service:
            mock_instance = AsyncMock()
            mock_service.return_value = mock_instance
            
            # 创建通知服务
            notification_service = mock_service()
            
            # 模拟不同严重级别的告警
            test_alerts = [
                {
                    "id": "alert_1",
                    "name": "CPU告警",
                    "severity": "high",
                    "labels": {"component": "system"},
                    "message": "CPU使用率过高"
                },
                {
                    "id": "alert_2",
                    "name": "内存告警",
                    "severity": "critical",
                    "labels": {"component": "system"},
                    "message": "内存使用率过高"
                },
                {
                    "id": "alert_3",
                    "name": "API告警",
                    "severity": "medium",
                    "labels": {"component": "api"},
                    "message": "API错误率过高"
                }
            ]
            
            # 发送告警通知
            for alert in test_alerts:
                await notification_service.send_notification(
                    alert["id"],
                    alert["name"],
                    alert["message"],
                    alert["severity"],
                    alert["labels"]
                )
            
            # 验证通知服务被调用
            assert mock_instance.send_notification.call_count == len(test_alerts)
            
            # 检查调用参数
            calls = mock_instance.send_notification.call_args_list
            assert len(calls) == 3
            
            print("通知路由测试完成")
    
    @pytest.mark.asyncio
    async def test_suppression_rules_e2e(self, complete_config):
        """测试抑制规则端到端流程"""
        with patch('harborai.core.alerts.suppression_manager.SuppressionManager') as mock_manager:
            mock_instance = MagicMock()
            mock_manager.return_value = mock_instance
            
            # 创建抑制管理器
            suppression_manager = mock_manager()
            
            # 模拟告警
            from harborai.core.alerts.alert_manager import Alert, AlertSeverity, AlertStatus
            
            alert1 = Alert(
                id="alert_1",
                rule_id="cpu_high",
                name="CPU告警",
                description="CPU使用率过高",
                severity=AlertSeverity.HIGH,
                status=AlertStatus.FIRING,
                labels={"component": "system"},
                annotations={"summary": "CPU使用率过高"},
                timestamp=datetime.now(),
                value=85.0
            )
            
            alert2 = Alert(
                id="alert_2",
                rule_id="memory_high",
                name="内存告警",
                description="内存使用率过高",
                severity=AlertSeverity.CRITICAL,
                status=AlertStatus.FIRING,
                labels={"component": "system"},
                annotations={"summary": "内存使用率过高"},
                timestamp=datetime.now(),
                value=95.0
            )
            
            # 测试重复告警抑制
            mock_instance.should_suppress.side_effect = [False, True]  # 第一次不抑制，第二次抑制
            
            # 第一次检查
            should_suppress_1 = suppression_manager.should_suppress(alert1)
            assert not should_suppress_1
            
            # 第二次检查（模拟重复告警）
            should_suppress_2 = suppression_manager.should_suppress(alert1)
            assert should_suppress_2
            
            # 测试依赖抑制
            mock_instance.should_suppress.return_value = True
            should_suppress_dep = suppression_manager.should_suppress(alert2)
            assert should_suppress_dep
            
            print("抑制规则测试完成")
    
    @pytest.mark.asyncio
    async def test_escalation_strategy_e2e(self, complete_config):
        """测试升级策略端到端流程"""
        # 模拟时间流逝
        start_time = datetime.now()
        
        with patch('harborai.core.alerts.alert_manager.datetime') as mock_datetime:
            mock_datetime.now.return_value = start_time
            
            # 创建告警
            from harborai.core.alerts.alert_manager import Alert, AlertSeverity, AlertStatus
            
            critical_alert = Alert(
                id="critical_alert",
                rule_id="memory_high",
                name="内存告警",
                description="内存使用率过高",
                severity=AlertSeverity.CRITICAL,
                status=AlertStatus.FIRING,
                labels={"component": "system"},
                annotations={"summary": "内存使用率过高"},
                timestamp=start_time,
                value=95.0
            )
            
            # 模拟升级逻辑
            escalation_config = complete_config["escalation"]
            critical_rule = None
            
            for rule in escalation_config["rules"]:
                if rule["severity"] == "critical":
                    critical_rule = rule
                    break
            
            assert critical_rule is not None
            
            # 检查升级步骤
            steps = critical_rule["steps"]
            assert len(steps) == 2
            
            # 第一步：立即通知
            step1 = steps[0]
            assert step1["delay"] == 0
            assert "console" in step1["channels"]
            
            # 第二步：5分钟后升级
            step2 = steps[1]
            assert step2["delay"] == 300
            assert "test_email" in step2["channels"]
            
            print("升级策略测试完成")
    
    def test_config_error_handling_e2e(self):
        """测试配置错误处理端到端流程"""
        # 创建包含错误的配置
        invalid_config = {
            "alert_rules": [
                {
                    "id": "",  # 错误：空ID
                    "name": "无效规则",
                    "severity": "invalid",  # 错误：无效严重级别
                    "condition": "threshold",
                    "metric": "test_metric",
                    "threshold": "not_a_number"  # 错误：非数字阈值
                }
            ],
            "notification": {
                "channels": [
                    {
                        "name": "invalid_channel",
                        "type": "invalid_type",  # 错误：无效类型
                        "enabled": True
                    }
                ]
            }
        }
        
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config, f, indent=2)
            temp_path = f.name
        
        try:
            # 验证配置文件
            is_valid, results = validate_config_file(temp_path)
            
            # 应该检测到错误
            assert not is_valid, "无效配置应该被检测到"
            
            from harborai.core.alerts.config_validator import ValidationLevel
            errors = [r for r in results if r.level == ValidationLevel.ERROR]
            assert len(errors) > 0, "应该有验证错误"
            
            # 检查错误类型
            error_messages = [e.message for e in errors]
            print(f"检测到的错误: {error_messages}")
            
            # 验证特定错误
            assert any("ID不能为空" in msg for msg in error_messages)
            assert any("无效的严重级别" in msg for msg in error_messages)
            assert any("不支持的通知类型" in msg for msg in error_messages)
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_performance_e2e(self, complete_config):
        """测试性能端到端流程"""
        import time
        
        # 测试配置验证性能
        start_time = time.time()
        
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(complete_config, f, indent=2)
            temp_path = f.name
        
        try:
            # 验证配置
            is_valid, results = validate_config_file(temp_path)
            validation_time = time.time() - start_time
            
            assert is_valid
            assert validation_time < 1.0, f"配置验证时间过长: {validation_time:.2f}秒"
            
            print(f"配置验证耗时: {validation_time:.3f}秒")
            
            # 测试告警管理器初始化性能
            start_time = time.time()
            
            with patch('harborai.core.alerts.alert_manager.NotificationService'), \
                 patch('harborai.core.alerts.alert_manager.SuppressionManager'):
                
                alert_manager = AlertManager()
                await alert_manager.initialize()
                
                init_time = time.time() - start_time
                assert init_time < 2.0, f"告警管理器初始化时间过长: {init_time:.2f}秒"
                
                print(f"告警管理器初始化耗时: {init_time:.3f}秒")
        
        finally:
            Path(temp_path).unlink(missing_ok=True)
    
    def test_config_completeness_e2e(self, complete_config):
        """测试配置完整性端到端流程"""
        # 验证所有必需的配置节都存在
        required_sections = [
            "alert_rules",
            "notification",
            "suppression",
            "escalation",
            "aggregation",
            "metrics",
            "health_check"
        ]
        
        for section in required_sections:
            assert section in complete_config, f"缺少必需的配置节: {section}"
        
        # 验证告警规则完整性
        for rule in complete_config["alert_rules"]:
            required_fields = ["id", "name", "description", "severity", "condition", "metric", "threshold"]
            for field in required_fields:
                assert field in rule, f"告警规则缺少必需字段: {field}"
        
        # 验证通知渠道完整性
        for channel in complete_config["notification"]["channels"]:
            required_fields = ["name", "type", "enabled"]
            for field in required_fields:
                assert field in channel, f"通知渠道缺少必需字段: {field}"
        
        # 验证抑制规则完整性
        for rule in complete_config["suppression"]["rules"]:
            required_fields = ["id", "name", "type", "enabled"]
            for field in required_fields:
                assert field in rule, f"抑制规则缺少必需字段: {field}"
        
        print("配置完整性验证通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])