#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警配置验证器单元测试

测试配置验证器的各种验证功能
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from harborai.core.alerts.config_validator import (
    ConfigValidator, ValidationLevel, ValidationResult,
    validate_config_file, validate_default_config, create_config_schema
)


class TestConfigValidator:
    """配置验证器测试"""
    
    @pytest.fixture
    def validator(self):
        """配置验证器实例"""
        return ConfigValidator()
    
    @pytest.fixture
    def valid_config(self):
        """有效的配置"""
        return {
            "alert_rules": [
                {
                    "id": "test_rule_1",
                    "name": "测试规则1",
                    "description": "测试用的告警规则",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 80.0,
                    "duration": 300,
                    "labels": {"component": "system"},
                    "annotations": {"summary": "CPU使用率过高"}
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
                        "name": "email",
                        "type": "email",
                        "enabled": True,
                        "config": {
                            "smtp_server": "smtp.example.com",
                            "smtp_port": 587,
                            "username": "test@example.com",
                            "password": "password",
                            "from_email": "alerts@example.com"
                        }
                    }
                ],
                "routing": {
                    "default_channels": ["console"],
                    "rules": [
                        {
                            "match": {"severity": "critical"},
                            "channels": ["console", "email"]
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
                        "id": "maintenance_window",
                        "name": "维护窗口",
                        "type": "time_based",
                        "enabled": True,
                        "time_config": {
                            "start_time": "02:00",
                            "end_time": "04:00",
                            "timezone": "Asia/Shanghai"
                        }
                    }
                ]
            },
            "escalation": {
                "enabled": True,
                "global_settings": {
                    "escalation_timeout": 1800,
                    "auto_resolve_timeout": 3600
                },
                "rules": [
                    {
                        "severity": "critical",
                        "escalation_time": 300,
                        "escalation_channels": ["email"],
                        "max_escalations": 3
                    }
                ]
            },
            "aggregation": {
                "enabled": True,
                "window_size": 300,
                "rules": []
            },
            "metrics": {
                "collection_interval": 60,
                "retention_days": 30
            },
            "health_check": {
                "enabled": True,
                "interval": 30
            }
        }
    
    @pytest.fixture
    def invalid_config(self):
        """无效的配置"""
        return {
            "alert_rules": [
                {
                    "id": "",  # 空ID
                    "name": "",  # 空名称
                    "severity": "invalid",  # 无效严重级别
                    "condition": "unknown",  # 无效条件
                    "metric": "",  # 空指标
                    "threshold": "not_a_number"  # 非数字阈值
                }
            ],
            "notification": {
                "channels": [
                    {
                        "name": "",  # 空名称
                        "type": "invalid_type",  # 无效类型
                        "enabled": "not_boolean"  # 非布尔值
                    }
                ]
            }
        }
    
    def test_validate_valid_config(self, validator, valid_config):
        """测试验证有效配置"""
        results = validator.validate_config(valid_config)
        
        # 不应该有错误
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0, f"不应该有错误，但发现: {[e.message for e in errors]}"
    
    def test_validate_invalid_config(self, validator, invalid_config):
        """测试验证无效配置"""
        results = validator.validate_config(invalid_config)
        
        # 应该有错误
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0, "应该有验证错误"
        
        # 检查特定错误
        error_messages = [e.message for e in errors]
        assert any("ID不能为空" in msg for msg in error_messages)
        assert any("名称不能为空" in msg for msg in error_messages)
        assert any("无效的严重级别" in msg for msg in error_messages)
    
    def test_validate_missing_required_fields(self, validator):
        """测试缺少必需字段"""
        config = {}
        results = validator.validate_config(config)
        
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0
        
        error_messages = [e.message for e in errors]
        assert any("缺少必需的配置节" in msg for msg in error_messages)
    
    def test_validate_alert_rules(self, validator):
        """测试告警规则验证"""
        config = {
            "alert_rules": [
                {
                    "id": "rule1",
                    "name": "规则1",
                    "description": "描述",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "cpu",
                    "threshold": 80.0
                },
                {
                    "id": "rule1",  # 重复ID
                    "name": "规则2",
                    "description": "描述",
                    "severity": "medium",
                    "condition": "threshold",
                    "metric": "memory",
                    "threshold": 90.0
                }
            ],
            "notification": {
                "channels": [{"name": "console", "type": "console"}]
            }
        }
        
        results = validator.validate_config(config)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # 应该检测到重复ID
        assert any("重复的告警规则ID" in e.message for e in errors)
    
    def test_validate_notification_channels(self, validator):
        """测试通知渠道验证"""
        config = {
            "alert_rules": [],
            "notification": {
                "channels": [
                    {
                        "name": "email1",
                        "type": "email",
                        "config": {}  # 缺少必需的邮件配置
                    },
                    {
                        "name": "webhook1",
                        "type": "webhook",
                        "config": {}  # 缺少必需的webhook配置
                    }
                ]
            }
        }
        
        results = validator.validate_config(config)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # 应该检测到缺少配置
        assert any("邮件渠道缺少必需配置" in e.message for e in errors)
        assert any("Webhook渠道缺少URL配置" in e.message for e in errors)
    
    def test_validate_suppression_rules(self, validator):
        """测试抑制规则验证"""
        config = {
            "alert_rules": [],
            "notification": {"channels": []},
            "suppression": {
                "rules": [
                    {
                        "id": "rule1",
                        "name": "规则1",
                        "type": "time_based"
                        # 缺少时间配置
                    },
                    {
                        "id": "rule2",
                        "name": "规则2",
                        "type": "dependency",
                        "dependency_alerts": ["nonexistent_alert"]  # 不存在的告警
                    }
                ]
            }
        }
        
        results = validator.validate_config(config)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # 应该检测到配置问题
        assert any("时间抑制规则缺少时间配置" in e.message for e in errors)
    
    def test_validate_escalation_config(self, validator):
        """测试升级配置验证"""
        config = {
            "alert_rules": [],
            "notification": {"channels": [{"name": "console", "type": "console"}]},
            "escalation": {
                "enabled": True,
                "rules": [
                    {
                        "severity": "critical",
                        "escalation_channels": ["nonexistent_channel"]  # 不存在的渠道
                    }
                ]
            }
        }
        
        results = validator.validate_config(config)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # 应该检测到不存在的渠道
        assert any("升级配置引用了不存在的通知渠道" in e.message for e in errors)
    
    def test_validate_config_completeness(self, validator):
        """测试配置完整性验证"""
        config = {
            "alert_rules": [],  # 没有告警规则
            "notification": {
                "channels": [
                    {"name": "disabled", "type": "console", "enabled": False}  # 没有启用的渠道
                ]
            },
            "escalation": {
                "enabled": False  # 没有启用升级
            }
        }
        
        results = validator.validate_config(config)
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        
        # 应该有完整性警告
        assert any("没有定义任何告警规则" in w.message for w in warnings)
        assert any("没有启用的通知渠道" in w.message for w in warnings)
    
    def test_validate_config_conflicts(self, validator):
        """测试配置冲突验证"""
        config = {
            "alert_rules": [
                {
                    "id": "rule1",
                    "name": "规则1",
                    "description": "描述",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 90.0  # 高严重级别，低阈值
                },
                {
                    "id": "rule2",
                    "name": "规则2",
                    "description": "描述",
                    "severity": "medium",
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 80.0  # 中等严重级别，更低阈值
                }
            ],
            "notification": {"channels": [{"name": "console", "type": "console"}]},
            "suppression": {
                "rules": [
                    {
                        "id": "time1",
                        "name": "时间1",
                        "type": "time_based",
                        "start_time": "02:00",
                        "end_time": "04:00"
                    },
                    {
                        "id": "time2",
                        "name": "时间2",
                        "type": "time_based",
                        "start_time": "03:00",  # 重叠时间
                        "end_time": "05:00"
                    }
                ]
            }
        }
        
        results = validator.validate_config(config)
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        
        # 应该检测到阈值冲突和时间重叠
        assert any("阈值配置可能存在冲突" in w.message for w in warnings)
        assert any("时间抑制规则存在重叠" in w.message for w in warnings)
    
    def test_get_summary(self, validator, valid_config):
        """测试获取验证摘要"""
        results = validator.validate_config(valid_config)
        summary = validator.get_summary()
        
        assert "total_results" in summary
        assert "errors" in summary
        assert "warnings" in summary
        assert "info" in summary
        assert summary["total_results"] == len(results)
    
    def test_format_results_text(self, validator, invalid_config):
        """测试文本格式化结果"""
        results = validator.validate_config(invalid_config)
        formatted = validator.format_results(format_type="text")
        
        assert isinstance(formatted, str)
        assert "错误" in formatted or "ERROR" in formatted
    
    def test_format_results_json(self, validator, invalid_config):
        """测试JSON格式化结果"""
        results = validator.validate_config(invalid_config)
        formatted = validator.format_results(format_type="json")
        
        # 应该是有效的JSON
        parsed = json.loads(formatted)
        assert "summary" in parsed
        assert "results" in parsed


class TestConfigValidatorFunctions:
    """配置验证器函数测试"""
    
    def test_validate_config_file_success(self):
        """测试成功验证配置文件"""
        config = {
            "alert_rules": [],
            "notification": {"channels": [{"name": "console", "type": "console"}]}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            is_valid, results = validate_config_file(temp_path)
            assert is_valid
            assert isinstance(results, list)
        finally:
            Path(temp_path).unlink()
    
    def test_validate_config_file_not_found(self):
        """测试配置文件不存在"""
        is_valid, results = validate_config_file("nonexistent.json")
        
        assert not is_valid
        assert len(results) == 1
        assert "配置文件不存在" in results[0].message
    
    def test_validate_config_file_invalid_json(self):
        """测试无效JSON文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            is_valid, results = validate_config_file(temp_path)
            assert not is_valid
            assert len(results) == 1
            assert "JSON格式错误" in results[0].message
        finally:
            Path(temp_path).unlink()
    
    @patch('harborai.core.alerts.config_validator.DEFAULT_ALERT_RULES', [])
    @patch('harborai.core.alerts.config_validator.DEFAULT_NOTIFICATION_CONFIG', {"channels": []})
    @patch('harborai.core.alerts.config_validator.DEFAULT_SUPPRESSION_RULES', [])
    @patch('harborai.core.alerts.config_validator.ESCALATION_CONFIG', {})
    @patch('harborai.core.alerts.config_validator.AGGREGATION_CONFIG', {})
    @patch('harborai.core.alerts.config_validator.METRICS_CONFIG', {})
    @patch('harborai.core.alerts.config_validator.HEALTH_CHECK_CONFIG', {})
    def test_validate_default_config(self):
        """测试验证默认配置"""
        is_valid, results = validate_default_config()
        
        # 默认配置应该是有效的（即使可能有警告）
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0, f"默认配置不应该有错误: {[e.message for e in errors]}"
    
    def test_create_config_schema(self):
        """测试创建配置模式"""
        schema = create_config_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "alert_rules" in schema["properties"]
        assert "notification" in schema["properties"]
        assert "required" in schema
        assert "alert_rules" in schema["required"]
        assert "notification" in schema["required"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])