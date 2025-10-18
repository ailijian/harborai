#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警配置验证集成测试

测试配置验证器与其他组件的集成，包括实际配置文件的验证、
配置热重载、配置冲突检测等完整流程。
"""

import pytest
import json
import yaml
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from harborai.core.alerts.config_validator import (
    ConfigValidator, ValidationLevel, ValidationResult,
    validate_config_file, validate_default_config
)
from harborai.core.alerts.alert_manager import AlertManager
from harborai.core.alerts.notification_service import NotificationService
from harborai.core.alerts.suppression_manager import SuppressionManager


class TestConfigValidationIntegration:
    """配置验证集成测试"""
    
    @pytest.fixture
    def production_like_config(self):
        """生产环境类似的配置"""
        return {
            "alert_rules": [
                {
                    "id": "cpu_high",
                    "name": "CPU使用率过高",
                    "description": "系统CPU使用率超过阈值",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "system.cpu.usage",
                    "threshold": 85.0,
                    "duration": 300,
                    "labels": {
                        "component": "system",
                        "team": "infrastructure",
                        "environment": "production"
                    },
                    "annotations": {
                        "summary": "CPU使用率过高: {{ $value }}%",
                        "description": "主机 {{ $labels.instance }} 的CPU使用率已达到 {{ $value }}%，超过阈值 {{ $threshold }}%",
                        "runbook": "https://wiki.company.com/runbooks/high-cpu",
                        "dashboard": "https://grafana.company.com/d/system-overview"
                    }
                },
                {
                    "id": "memory_high",
                    "name": "内存使用率过高",
                    "description": "系统内存使用率超过阈值",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "system.memory.usage",
                    "threshold": 90.0,
                    "duration": 180,
                    "labels": {
                        "component": "system",
                        "team": "infrastructure",
                        "environment": "production"
                    },
                    "annotations": {
                        "summary": "内存使用率过高: {{ $value }}%",
                        "description": "主机 {{ $labels.instance }} 的内存使用率已达到 {{ $value }}%",
                        "runbook": "https://wiki.company.com/runbooks/high-memory"
                    }
                },
                {
                    "id": "disk_space_low",
                    "name": "磁盘空间不足",
                    "description": "磁盘可用空间低于阈值",
                    "severity": "critical",
                    "condition": "threshold",
                    "metric": "system.disk.free_percent",
                    "threshold": 10.0,
                    "duration": 60,
                    "labels": {
                        "component": "system",
                        "team": "infrastructure",
                        "environment": "production"
                    },
                    "annotations": {
                        "summary": "磁盘空间不足: {{ $value }}%",
                        "description": "主机 {{ $labels.instance }} 的磁盘可用空间仅剩 {{ $value }}%",
                        "runbook": "https://wiki.company.com/runbooks/disk-space"
                    }
                },
                {
                    "id": "api_response_time_high",
                    "name": "API响应时间过长",
                    "description": "API平均响应时间超过阈值",
                    "severity": "medium",
                    "condition": "threshold",
                    "metric": "api.response_time.avg",
                    "threshold": 2000.0,
                    "duration": 300,
                    "labels": {
                        "component": "api",
                        "team": "backend",
                        "environment": "production"
                    },
                    "annotations": {
                        "summary": "API响应时间过长: {{ $value }}ms",
                        "description": "API {{ $labels.endpoint }} 的平均响应时间为 {{ $value }}ms",
                        "runbook": "https://wiki.company.com/runbooks/api-performance"
                    }
                },
                {
                    "id": "error_rate_high",
                    "name": "错误率过高",
                    "description": "API错误率超过阈值",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "api.error_rate",
                    "threshold": 5.0,
                    "duration": 120,
                    "labels": {
                        "component": "api",
                        "team": "backend",
                        "environment": "production"
                    },
                    "annotations": {
                        "summary": "错误率过高: {{ $value }}%",
                        "description": "API错误率已达到 {{ $value }}%，需要立即检查",
                        "runbook": "https://wiki.company.com/runbooks/high-error-rate"
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
                        "name": "ops_email",
                        "type": "email",
                        "enabled": True,
                        "config": {
                            "smtp_host": "smtp.company.com",
                            "smtp_port": 587,
                            "username": "alerts@company.com",
                            "password": "${SMTP_PASSWORD}",
                            "from_email": "alerts@company.com",
                            "use_tls": True
                        }
                    },
                    {
                        "name": "ops_slack",
                        "type": "slack",
                        "enabled": True,
                        "config": {
                            "webhook_url": "${SLACK_WEBHOOK_URL}",
                            "channel": "#ops-alerts",
                            "username": "AlertBot",
                            "icon_emoji": ":warning:"
                        }
                    },
                    {
                        "name": "dev_dingtalk",
                        "type": "dingtalk",
                        "enabled": True,
                        "config": {
                            "webhook_url": "${DINGTALK_WEBHOOK_URL}",
                            "secret": "${DINGTALK_SECRET}",
                            "at_all": False,
                            "at_mobiles": []
                        }
                    }
                ],
                "routing": {
                    "default_channels": ["console"],
                    "rules": [
                        {
                            "match": {"severity": "critical"},
                            "channels": ["console", "ops_email", "ops_slack"]
                        },
                        {
                            "match": {"severity": "high"},
                            "channels": ["console", "ops_slack"]
                        },
                        {
                            "match": {"team": "backend"},
                            "channels": ["console", "dev_dingtalk"]
                        },
                        {
                            "match": {"component": "system"},
                            "channels": ["console", "ops_email"]
                        }
                    ]
                },
                "rate_limits": {
                    "enabled": True,
                    "max_notifications_per_minute": 20,
                    "burst_limit": 50
                },
                "retry": {
                    "enabled": True,
                    "max_attempts": 3,
                    "backoff_factor": 2.0,
                    "max_delay": 300
                }
            },
            "suppression": {
                "rules": [
                    {
                        "id": "maintenance_window",
                        "name": "维护窗口",
                        "description": "夜间维护窗口期间抑制告警",
                        "type": "time_based",
                        "enabled": True,
                        "start_time": "02:00",
                        "end_time": "04:00",
                        "timezone": "Asia/Shanghai",
                        "weekdays": [1, 2, 3, 4, 5]  # 工作日
                    },
                    {
                        "id": "duplicate_alerts",
                        "name": "重复告警抑制",
                        "description": "抑制短时间内的重复告警",
                        "type": "duplicate",
                        "enabled": True,
                        "duplicate_window": 300,
                        "duplicate_threshold": 3
                    },
                    {
                        "id": "cascade_suppression",
                        "name": "级联抑制",
                        "description": "当主要服务告警时抑制依赖服务告警",
                        "type": "dependency",
                        "enabled": True,
                        "dependency_alerts": ["api_response_time_high"],
                        "dependency_rules": ["error_rate_high"]
                    },
                    {
                        "id": "low_priority_night",
                        "name": "夜间低优先级抑制",
                        "description": "夜间抑制低优先级告警",
                        "type": "time_based",
                        "enabled": True,
                        "start_time": "22:00",
                        "end_time": "08:00",
                        "timezone": "Asia/Shanghai",
                        "severity_filter": ["low", "medium"]
                    }
                ]
            },
            "escalation": {
                "enabled": True,
                "global_settings": {
                    "escalation_timeout": 1800,
                    "auto_resolve_timeout": 3600,
                    "escalation_cooldown": 300,
                    "max_total_escalations": 5,
                    "business_hours": {
                        "start": "09:00",
                        "end": "18:00",
                        "timezone": "Asia/Shanghai",
                        "weekdays": [1, 2, 3, 4, 5]
                    }
                },
                "rules": [
                    {
                        "severity": "critical",
                        "steps": [
                            {
                                "delay": 0,
                                "channels": ["ops_slack"],
                                "message_template": "escalation_immediate",
                                "conditions": {
                                    "require_ack": False,
                                    "business_hours_only": False
                                }
                            },
                            {
                                "delay": 300,
                                "channels": ["ops_email"],
                                "message_template": "escalation_level1",
                                "conditions": {
                                    "require_ack": True,
                                    "business_hours_only": False
                                },
                                "auto_actions": ["create_incident"]
                            },
                            {
                                "delay": 900,
                                "channels": ["ops_email"],
                                "message_template": "escalation_level2",
                                "conditions": {
                                    "escalate_to": "manager",
                                    "business_hours_only": False
                                },
                                "auto_actions": ["page_on_call"]
                            }
                        ],
                        "max_escalations": 3
                    },
                    {
                        "severity": "high",
                        "steps": [
                            {
                                "delay": 600,
                                "channels": ["ops_slack"],
                                "message_template": "escalation_level1",
                                "conditions": {
                                    "require_ack": True,
                                    "business_hours_only": True
                                }
                            },
                            {
                                "delay": 1800,
                                "channels": ["ops_email"],
                                "message_template": "escalation_level2",
                                "conditions": {
                                    "business_hours_only": True
                                }
                            }
                        ],
                        "max_escalations": 2
                    }
                ],
                "notification_templates": {
                    "escalation_immediate": {
                        "subject": "🚨 紧急告警 - {{ alert.name }}",
                        "body": "告警已触发，需要立即处理！\n\n详情：{{ alert.description }}\n时间：{{ alert.timestamp }}"
                    },
                    "escalation_level1": {
                        "subject": "⚠️ 告警升级 - {{ alert.name }}",
                        "body": "告警已升级，请及时处理。\n\n详情：{{ alert.description }}\n持续时间：{{ alert.duration }}"
                    },
                    "escalation_level2": {
                        "subject": "🔥 高级别告警升级 - {{ alert.name }}",
                        "body": "告警已升级到高级别，需要管理层介入。\n\n详情：{{ alert.description }}\n影响：{{ alert.impact }}"
                    }
                },
                "escalation_policies": {
                    "default": {
                        "on_call_schedule": {
                            "primary": ["ops-team@company.com"],
                            "secondary": ["dev-team@company.com"],
                            "manager": ["manager@company.com"]
                        }
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
                        "group_by": ["instance", "component"],
                        "match": {"component": "system"},
                        "threshold": 3
                    },
                    {
                        "id": "api_alerts",
                        "name": "API告警聚合",
                        "group_by": ["endpoint"],
                        "match": {"component": "api"},
                        "threshold": 2
                    }
                ]
            },
            "metrics": {
                "collection_interval": 60,
                "retention_days": 30,
                "storage_path": "/var/lib/harborai/metrics",
                "compression": True
            },
            "health_check": {
                "enabled": True,
                "interval": 30,
                "timeout": 10,
                "endpoints": [
                    "http://localhost:8080/health",
                    "http://localhost:9090/metrics"
                ]
            }
        }
    
    @pytest.fixture
    def config_with_errors(self):
        """包含错误的配置"""
        return {
            "alert_rules": [
                {
                    "id": "",  # 错误：空ID
                    "name": "无效规则",
                    "severity": "invalid_severity",  # 错误：无效严重级别
                    "condition": "threshold",
                    "metric": "test_metric",
                    "threshold": "not_a_number"  # 错误：非数字阈值
                },
                {
                    "id": "duplicate_id",
                    "name": "规则1",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "metric1",
                    "threshold": 80.0
                },
                {
                    "id": "duplicate_id",  # 错误：重复ID
                    "name": "规则2",
                    "severity": "medium",
                    "condition": "threshold",
                    "metric": "metric2",
                    "threshold": 70.0
                }
            ],
            "notification": {
                "channels": [
                    {
                        "name": "",  # 错误：空名称
                        "type": "invalid_type",  # 错误：无效类型
                        "enabled": "not_boolean"  # 错误：非布尔值
                    },
                    {
                        "name": "email_channel",
                        "type": "email",
                        "config": {}  # 错误：缺少必需的邮件配置
                    }
                ],
                "routing": {
                    "rules": [
                        {
                            "match": {"severity": "critical"},
                            "channels": ["nonexistent_channel"]  # 错误：不存在的渠道
                        }
                    ]
                }
            },
            "suppression": {
                "rules": [
                    {
                        "id": "time_rule",
                        "name": "时间规则",
                        "type": "time_based"
                        # 错误：缺少时间配置
                    },
                    {
                        "id": "dep_rule",
                        "name": "依赖规则",
                        "type": "dependency",
                        "dependency_alerts": ["nonexistent_alert"]  # 错误：不存在的告警
                    }
                ]
            },
            "escalation": {
                "enabled": True,
                "rules": [
                    {
                        "severity": "critical",
                        "escalation_channels": ["nonexistent_channel"]  # 错误：不存在的渠道
                    }
                ]
            }
        }
    
    def test_validate_production_config(self, production_like_config):
        """测试验证生产环境配置"""
        validator = ConfigValidator()
        results = validator.validate_config(production_like_config)
        
        # 统计结果
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        info = [r for r in results if r.level == ValidationLevel.INFO]
        
        # 生产配置不应该有错误
        assert len(errors) == 0, f"生产配置不应该有错误: {[e.message for e in errors]}"
        
        # 可能有一些警告（这是正常的）
        print(f"验证结果: {len(errors)} 错误, {len(warnings)} 警告, {len(info)} 信息")
        
        # 检查配置完整性
        summary = validator.get_summary()
        assert summary["errors"] == 0
    
    def test_validate_config_with_errors(self, config_with_errors):
        """测试验证包含错误的配置"""
        validator = ConfigValidator()
        results = validator.validate_config(config_with_errors)
        
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # 应该检测到所有错误
        assert len(errors) > 0, "应该检测到配置错误"
        
        # 检查特定错误类型
        error_messages = [e.message for e in errors]
        
        # 告警规则错误
        assert any("ID不能为空" in msg for msg in error_messages)
        assert any("无效的严重级别" in msg for msg in error_messages)
        assert any("重复的告警规则ID" in msg for msg in error_messages)
        
        # 通知渠道错误
        assert any("名称不能为空" in msg for msg in error_messages)
        assert any("不支持的通知类型" in msg for msg in error_messages)
        assert any("邮件渠道缺少必需的配置" in msg for msg in error_messages)
        
        # 交叉引用错误
        assert any("引用了不存在的通知渠道" in msg for msg in error_messages)
    
    def test_config_file_validation_json(self, production_like_config):
        """测试JSON配置文件验证"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(production_like_config, f, indent=2, ensure_ascii=False)
            temp_path = f.name
        
        try:
            is_valid, results = validate_config_file(temp_path)
            
            assert is_valid, f"配置文件应该有效: {[r.message for r in results if r.level == ValidationLevel.ERROR]}"
            
            # 检查结果格式
            assert isinstance(results, list)
            for result in results:
                assert isinstance(result, ValidationResult)
                assert hasattr(result, 'level')
                assert hasattr(result, 'category')
                assert hasattr(result, 'message')
        finally:
            Path(temp_path).unlink()
    
    def test_config_file_validation_yaml(self, production_like_config):
        """测试YAML配置文件验证"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(production_like_config, f, default_flow_style=False, allow_unicode=True)
            temp_path = f.name
        
        try:
            # 修改验证函数以支持YAML
            with open(temp_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            validator = ConfigValidator()
            results = validator.validate_config(config)
            
            errors = [r for r in results if r.level == ValidationLevel.ERROR]
            assert len(errors) == 0, f"YAML配置不应该有错误: {[e.message for e in errors]}"
        finally:
            Path(temp_path).unlink()
    
    def test_config_hot_reload_validation(self, production_like_config):
        """测试配置热重载验证"""
        # 创建初始配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(production_like_config, f, indent=2)
            temp_path = f.name
        
        try:
            # 初始验证
            is_valid, results = validate_config_file(temp_path)
            assert is_valid
            
            # 修改配置（添加错误）
            invalid_config = production_like_config.copy()
            invalid_config["alert_rules"][0]["severity"] = "invalid_severity"
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(invalid_config, f, indent=2)
            
            # 重新验证
            is_valid, results = validate_config_file(temp_path)
            assert not is_valid
            
            errors = [r for r in results if r.level == ValidationLevel.ERROR]
            assert any("无效的严重级别" in e.message for e in errors)
        finally:
            Path(temp_path).unlink()
    
    def test_cross_component_validation(self, production_like_config):
        """测试跨组件验证"""
        validator = ConfigValidator()
        results = validator.validate_config(production_like_config)
        
        # 检查交叉引用验证
        # 1. 通知路由中的渠道应该存在
        # 2. 抑制规则中的依赖告警应该存在
        # 3. 升级配置中的渠道应该存在
        
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        cross_ref_errors = [e for e in errors if "引用" in e.message or "不存在" in e.message]
        
        # 生产配置的交叉引用应该是正确的
        assert len(cross_ref_errors) == 0, f"交叉引用错误: {[e.message for e in cross_ref_errors]}"
    
    def test_performance_with_large_config(self):
        """测试大型配置的验证性能"""
        import time
        
        # 生成大型配置
        large_config = {
            "alert_rules": [],
            "notification": {"channels": []},
            "suppression": {"rules": []},
            "escalation": {"enabled": False},
            "aggregation": {"enabled": False},
            "metrics": {},
            "health_check": {"enabled": False}
        }
        
        # 生成1000个告警规则
        for i in range(1000):
            rule = {
                "id": f"rule_{i}",
                "name": f"规则 {i}",
                "description": f"测试规则 {i}",
                "severity": "medium",
                "condition": "threshold",
                "metric": f"metric_{i}",
                "threshold": 80.0,
                "duration": 300,
                "labels": {"component": f"comp_{i % 10}"},
                "annotations": {"summary": f"告警 {i}"}
            }
            large_config["alert_rules"].append(rule)
        
        # 生成100个通知渠道
        for i in range(100):
            channel = {
                "name": f"channel_{i}",
                "type": "console",
                "enabled": True,
                "config": {}
            }
            large_config["notification"]["channels"].append(channel)
        
        # 测试验证性能
        validator = ConfigValidator()
        start_time = time.time()
        results = validator.validate_config(large_config)
        end_time = time.time()
        
        validation_time = end_time - start_time
        print(f"验证1000个规则和100个渠道耗时: {validation_time:.2f}秒")
        
        # 验证应该在合理时间内完成（比如5秒）
        assert validation_time < 5.0, f"验证时间过长: {validation_time:.2f}秒"
        
        # 检查结果
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0, "大型配置不应该有错误"
    
    @pytest.mark.asyncio
    async def test_integration_with_alert_manager(self, production_like_config):
        """测试与告警管理器的集成"""
        # 首先验证配置
        validator = ConfigValidator()
        results = validator.validate_config(production_like_config)
        
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0, "配置必须有效才能集成"
        
        # 模拟告警管理器使用配置
        with patch('harborai.core.alerts.alert_manager.AlertManager') as mock_manager:
            mock_instance = AsyncMock()
            mock_manager.return_value = mock_instance
            
            # 模拟加载配置
            mock_instance.load_rules.return_value = True
            
            # 创建告警管理器实例
            alert_manager = mock_manager()
            
            # 加载告警规则
            rules_loaded = await alert_manager.load_rules(production_like_config["alert_rules"])
            assert rules_loaded
            
            # 验证调用
            mock_instance.load_rules.assert_called_once()
    
    def test_default_config_validation(self):
        """测试默认配置验证"""
        is_valid, results = validate_default_config()
        
        # 默认配置应该是有效的
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0, f"默认配置不应该有错误: {[e.message for e in errors]}"
        
        # 可能有一些信息性消息
        info_count = len([r for r in results if r.level == ValidationLevel.INFO])
        print(f"默认配置验证: {info_count} 条信息")
    
    def test_config_schema_validation(self):
        """测试配置模式验证"""
        from harborai.core.alerts.config_validator import create_config_schema
        
        schema = create_config_schema()
        
        # 验证模式结构
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        
        # 验证必需字段
        required_fields = schema["required"]
        assert "alert_rules" in required_fields
        assert "notification" in required_fields
        
        # 验证属性定义
        properties = schema["properties"]
        assert "alert_rules" in properties
        assert "notification" in properties
        assert "suppression" in properties
        assert "escalation" in properties
    
    def test_validation_result_formatting(self, config_with_errors):
        """测试验证结果格式化"""
        validator = ConfigValidator()
        results = validator.validate_config(config_with_errors)
        
        # 测试文本格式
        text_output = validator.format_results(format_type="text")
        assert isinstance(text_output, str)
        assert len(text_output) > 0
        assert "错误" in text_output or "ERROR" in text_output
        
        # 测试JSON格式
        json_output = validator.format_results(format_type="json")
        parsed = json.loads(json_output)
        
        assert "summary" in parsed
        assert "results" in parsed
        assert "total_results" in parsed["summary"]
        assert "errors" in parsed["summary"]
        assert "warnings" in parsed["summary"]
        
        # 验证结果数据
        assert parsed["summary"]["errors"] > 0
        assert len(parsed["results"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])