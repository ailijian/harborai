#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警配置场景测试

测试各种复杂的告警配置场景和边界情况
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

from harborai.core.alerts.config_validator import (
    ConfigValidator, ValidationLevel, ValidationResult
)
from harborai.core.alerts.alert_manager import AlertSeverity, AlertCondition
from harborai.core.alerts.suppression_manager import SuppressionType
from harborai.core.alerts.notification_service import NotificationPriority


class TestAlertConfigScenarios:
    """告警配置场景测试"""
    
    @pytest.fixture
    def validator(self):
        """配置验证器实例"""
        return ConfigValidator()
        
    @pytest.fixture
    def minimal_valid_config(self):
        """最小有效配置"""
        return {
            "alert_rules": [
                {
                    "id": "minimal_rule",
                    "name": "最小规则",
                    "severity": "medium",
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 80.0,
                    "duration": 300
                }
            ],
            "notification": {
                "channels": [
                    {
                        "name": "console",
                        "type": "console",
                        "enabled": True
                    }
                ]
            }
        }
        
    @pytest.fixture
    def complex_valid_config(self):
        """复杂有效配置"""
        return {
            "alert_rules": [
                {
                    "id": "cpu_high",
                    "name": "CPU使用率过高",
                    "description": "监控CPU使用率",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 85.0,
                    "duration": 300,
                    "labels": {
                        "component": "system",
                        "env": "production"
                    },
                    "annotations": {
                        "summary": "CPU使用率超过85%",
                        "description": "系统CPU使用率持续5分钟超过85%",
                        "runbook": "检查系统负载和进程"
                    }
                },
                {
                    "id": "memory_high",
                    "name": "内存使用率过高",
                    "description": "监控内存使用率",
                    "severity": "critical",
                    "condition": "threshold",
                    "metric": "memory_usage",
                    "threshold": 90.0,
                    "duration": 180,
                    "labels": {
                        "component": "system",
                        "env": "production"
                    },
                    "annotations": {
                        "summary": "内存使用率超过90%",
                        "description": "系统内存使用率持续3分钟超过90%"
                    }
                },
                {
                    "id": "disk_space_low",
                    "name": "磁盘空间不足",
                    "description": "监控磁盘空间",
                    "severity": "warning",
                    "condition": "threshold",
                    "metric": "disk_usage",
                    "threshold": 95.0,
                    "duration": 600,
                    "labels": {
                        "component": "storage",
                        "env": "production"
                    }
                }
            ],
            "notification": {
                "channels": [
                    {
                        "name": "console",
                        "type": "console",
                        "enabled": True
                    },
                    {
                        "name": "email",
                        "type": "email",
                        "enabled": True,
                        "config": {
                            "smtp_server": "smtp.company.com",
                            "smtp_port": 587,
                            "username": "alerts@company.com",
                            "password": "secure_password",
                            "from_email": "alerts@company.com",
                            "to_emails": ["admin@company.com", "ops@company.com"]
                        }
                    },
                    {
                        "name": "slack",
                        "type": "slack",
                        "enabled": True,
                        "config": {
                            "webhook_url": "https://hooks.slack.com/services/TXXXXXXXX/BXXXXXXXX/XXXXXXXXXXXXXXXXXXXXXXXX",
                            "channel": "#alerts",
                            "username": "AlertBot"
                        }
                    },
                    {
                        "name": "webhook",
                        "type": "webhook",
                        "enabled": False,
                        "config": {
                            "url": "https://api.company.com/alerts",
                            "method": "POST",
                            "headers": {
                                "Content-Type": "application/json",
                                "Authorization": "Bearer token123"
                            }
                        }
                    }
                ],
                "routing": {
                    "rules": [
                        {
                            "name": "critical_alerts",
                            "match": {
                                "severity": ["critical"]
                            },
                            "channels": ["email", "slack"]
                        },
                        {
                            "name": "high_alerts",
                            "match": {
                                "severity": ["high"]
                            },
                            "channels": ["slack"]
                        },
                        {
                            "name": "default",
                            "match": {},
                            "channels": ["console"]
                        }
                    ]
                },
                "rate_limit": {
                    "enabled": True,
                    "max_notifications_per_minute": 10
                }
            },
            "suppression": {
                "rules": [
                    {
                        "id": "maintenance_window",
                        "name": "维护窗口抑制",
                        "description": "在维护窗口期间抑制所有告警",
                        "type": "maintenance",
                        "status": "active",
                        "maintenance_windows": [
                            {
                                "start_time": "02:00",
                                "end_time": "04:00",
                                "days": ["sunday"],
                                "timezone": "UTC"
                            }
                        ]
                    },
                    {
                        "id": "low_priority_rate_limit",
                        "name": "低优先级速率限制",
                        "description": "限制低优先级告警的发送频率",
                        "type": "rate_limit",
                        "status": "active",
                        "severity_filter": ["low", "info"],
                        "rate_limit_config": {
                            "max_alerts": 5,
                            "time_window": 300
                        }
                    }
                ]
            },
            "escalation": {
                "enabled": True,
                "policies": [
                    {
                        "name": "critical_escalation",
                        "levels": [
                            {
                                "level": 1,
                                "delay": 300,
                                "channels": ["email"]
                            },
                            {
                                "level": 2,
                                "delay": 900,
                                "channels": ["email", "slack"]
                            }
                        ]
                    }
                ]
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

    def test_minimal_config_validation(self, validator, minimal_valid_config):
        """测试最小配置验证"""
        results = validator.validate_config(minimal_valid_config)
        
        # 应该没有错误
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0
        
        # 可能有一些警告或信息
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        infos = [r for r in results if r.level == ValidationLevel.INFO]
        
        # 验证基本结构
        assert "alert_rules" in minimal_valid_config
        assert "notification" in minimal_valid_config
        
    def test_complex_config_validation(self, validator, complex_valid_config):
        """测试复杂配置验证"""
        results = validator.validate_config(complex_valid_config)
        
        # 应该没有错误
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0
        
        # 验证所有组件都存在
        assert "alert_rules" in complex_valid_config
        assert "notification" in complex_valid_config
        assert "suppression" in complex_valid_config
        assert "escalation" in complex_valid_config
        assert "metrics" in complex_valid_config
        assert "health_check" in complex_valid_config
        
    def test_missing_required_sections(self, validator):
        """测试缺少必需部分的配置"""
        incomplete_config = {
            "notification": {
                "channels": []
            }
        }
        
        results = validator.validate_config(incomplete_config)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # 应该有缺少alert_rules的错误
        assert any("alert_rules" in error.message for error in errors)
        
    def test_invalid_alert_rule_fields(self, validator):
        """测试无效的告警规则字段"""
        invalid_config = {
            "alert_rules": [
                {
                    "id": "",  # 空ID
                    "name": "无效规则",
                    "severity": "invalid_severity",  # 无效严重级别
                    "condition": "invalid_condition",  # 无效条件
                    "metric": "",  # 空指标
                    "threshold": "not_a_number",  # 非数字阈值
                    "duration": -1  # 负数持续时间
                }
            ],
            "notification": {
                "channels": [{"name": "console", "type": "console", "enabled": True}]
            }
        }
        
        results = validator.validate_config(invalid_config)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # 应该有多个错误
        assert len(errors) > 0
        
        # 检查特定错误
        error_messages = [error.message for error in errors]
        assert any("ID不能为空" in msg for msg in error_messages)
        assert any("无效的严重级别" in msg for msg in error_messages)
        assert any("无效的条件类型" in msg for msg in error_messages)
        
    def test_duplicate_alert_rule_ids(self, validator):
        """测试重复的告警规则ID"""
        config_with_duplicates = {
            "alert_rules": [
                {
                    "id": "duplicate_id",
                    "name": "规则1",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 80.0,
                    "duration": 300
                },
                {
                    "id": "duplicate_id",  # 重复ID
                    "name": "规则2",
                    "severity": "medium",
                    "condition": "threshold",
                    "metric": "memory_usage",
                    "threshold": 85.0,
                    "duration": 300
                }
            ],
            "notification": {
                "channels": [{"name": "console", "type": "console", "enabled": True}]
            }
        }
        
        results = validator.validate_config(config_with_duplicates)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # 应该有重复ID的错误
        assert any("重复的告警规则ID" in error.message for error in errors)
        
    def test_invalid_notification_channels(self, validator):
        """测试无效的通知渠道"""
        invalid_notification_config = {
            "alert_rules": [
                {
                    "id": "test_rule",
                    "name": "测试规则",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 80.0,
                    "duration": 300
                }
            ],
            "notification": {
                "channels": [
                    {
                        "name": "",  # 空名称
                        "type": "invalid_type",  # 无效类型
                        "enabled": "not_boolean"  # 非布尔值
                    },
                    {
                        "name": "email_without_config",
                        "type": "email",
                        "enabled": True
                        # 缺少email配置
                    },
                    {
                        "name": "webhook_invalid_url",
                        "type": "webhook",
                        "enabled": True,
                        "config": {
                            "url": "not_a_valid_url"  # 无效URL
                        }
                    }
                ]
            }
        }
        
        results = validator.validate_config(invalid_notification_config)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # 应该有多个错误
        assert len(errors) > 0
        
        error_messages = [error.message for error in errors]
        assert any("渠道名称不能为空" in msg for msg in error_messages)
        assert any("不支持的渠道类型" in msg for msg in error_messages)
        
    def test_cross_reference_validation(self, validator):
        """测试交叉引用验证"""
        config_with_invalid_references = {
            "alert_rules": [
                {
                    "id": "test_rule",
                    "name": "测试规则",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 80.0,
                    "duration": 300
                }
            ],
            "notification": {
                "channels": [
                    {
                        "name": "console",
                        "type": "console",
                        "enabled": True
                    }
                ],
                "routing": {
                    "rules": [
                        {
                            "name": "test_routing",
                            "match": {"severity": ["high"]},
                            "channels": ["nonexistent_channel"]  # 引用不存在的渠道
                        }
                    ]
                }
            },
            "suppression": {
                "rules": [
                    {
                        "id": "dependency_rule",
                        "name": "依赖抑制",
                        "type": "dependency",
                        "status": "active",
                        "config": {
                            "parent_rule": "nonexistent_parent",  # 引用不存在的父规则
                            "child_rules": ["nonexistent_child"]   # 引用不存在的子规则
                        }
                    }
                ]
            }
        }
        
        results = validator.validate_config(config_with_invalid_references)
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        
        # 应该有交叉引用错误
        error_messages = [error.message for error in errors]
        assert any("引用了不存在的渠道" in msg for msg in error_messages)
        assert any("引用了不存在的父告警规则" in msg or "引用了不存在的子告警规则" in msg for msg in error_messages)
        
    def test_threshold_consistency_validation(self, validator):
        """测试阈值一致性验证"""
        config_with_threshold_conflicts = {
            "alert_rules": [
                {
                    "id": "cpu_rule_1",
                    "name": "CPU规则1",
                    "severity": "high",
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 80.0,
                    "duration": 300
                },
                {
                    "id": "cpu_rule_2",
                    "name": "CPU规则2",
                    "severity": "critical",
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 70.0,  # 更低的阈值但更高的严重级别
                    "duration": 300
                }
            ],
            "notification": {
                "channels": [{"name": "console", "type": "console", "enabled": True}]
            }
        }
        
        results = validator.validate_config(config_with_threshold_conflicts)
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        
        # 应该有阈值冲突的警告
        warning_messages = [warning.message for warning in warnings]
        assert any("阈值设置可能存在冲突" in msg for msg in warning_messages)
        
    def test_completeness_validation(self, validator):
        """测试配置完整性验证"""
        incomplete_config = {
            "alert_rules": [
                {
                    "id": "test_rule",
                    "name": "测试规则",
                    "severity": "low",  # 只有低级别告警
                    "condition": "threshold",
                    "metric": "cpu_usage",
                    "threshold": 80.0,
                    "duration": 300
                }
            ],
            "notification": {
                "channels": [
                    {
                        "name": "disabled_channel",
                        "type": "console",
                        "enabled": False  # 所有渠道都被禁用
                    }
                ]
            }
            # 缺少升级配置
        }
        
        results = validator.validate_config(incomplete_config)
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        infos = [r for r in results if r.level == ValidationLevel.INFO]
        
        # 应该有完整性相关的警告或信息
        all_messages = [r.message for r in warnings + infos]
        assert any("没有启用的通知渠道" in msg for msg in all_messages)
        assert any("没有定义严重级别的告警规则" in msg for msg in all_messages)
        assert any("没有启用告警升级策略" in msg for msg in all_messages)
        
    def test_config_file_validation(self, validator, complex_valid_config):
        """测试配置文件验证"""
        # 创建临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(complex_valid_config, f, indent=2, ensure_ascii=False)
            temp_file = f.name
            
        try:
            from harborai.core.alerts.config_validator import validate_config_file
            
            is_valid, results = validate_config_file(temp_file)
            
            assert is_valid is True
            errors = [r for r in results if r.level == ValidationLevel.ERROR]
            assert len(errors) == 0
            
        finally:
            os.unlink(temp_file)
            
    def test_invalid_json_file(self, validator):
        """测试无效JSON文件"""
        # 创建包含无效JSON的临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json,}')  # 无效JSON
            temp_file = f.name
            
        try:
            from harborai.core.alerts.config_validator import validate_config_file
            
            is_valid, results = validate_config_file(temp_file)
            
            assert is_valid is False
            assert len(results) > 0
            assert any("JSON格式错误" in result.message for result in results)
            
        finally:
            os.unlink(temp_file)
            
    def test_nonexistent_config_file(self, validator):
        """测试不存在的配置文件"""
        from harborai.core.alerts.config_validator import validate_config_file
        
        is_valid, results = validate_config_file("nonexistent_file.json")
        
        assert is_valid is False
        assert len(results) > 0
        assert any("文件不存在" in result.message for result in results)
        
    def test_config_schema_generation(self):
        """测试配置模式生成"""
        from harborai.core.alerts.config_validator import create_config_schema
        
        schema = create_config_schema()
        
        assert isinstance(schema, dict)
        assert "type" in schema
        assert "properties" in schema
        assert "alert_rules" in schema["properties"]
        assert "notification" in schema["properties"]
        
    def test_validation_result_serialization(self, validator, minimal_valid_config):
        """测试验证结果序列化"""
        results = validator.validate_config(minimal_valid_config)
        
        # 测试转换为字典
        for result in results:
            result_dict = result.to_dict()
            assert isinstance(result_dict, dict)
            assert "level" in result_dict
            assert "category" in result_dict
            assert "message" in result_dict
            
        # 测试格式化输出
        text_output = validator.format_results("text")
        assert isinstance(text_output, str)
        
        json_output = validator.format_results("json")
        assert isinstance(json_output, str)
        
        # 验证JSON输出可以解析
        import json
        parsed_json = json.loads(json_output)
        assert isinstance(parsed_json, list)
        
    def test_validation_summary(self, validator, minimal_valid_config):
        """测试验证摘要"""
        results = validator.validate_config(minimal_valid_config)
        summary = validator.get_summary()
        
        assert isinstance(summary, dict)
        assert "total_issues" in summary
        assert "errors" in summary
        assert "warnings" in summary
        assert "infos" in summary
        assert "is_valid" in summary
        
        # 验证计数正确性
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        infos = [r for r in results if r.level == ValidationLevel.INFO]
        
        assert summary["errors"] == len(errors)
        assert summary["warnings"] == len(warnings)
        assert summary["infos"] == len(infos)
        assert summary["total_issues"] == len(results)
        assert summary["is_valid"] == (len(errors) == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])