"""
告警配置集成测试

测试告警配置系统的完整集成流程，包括配置加载、验证、通知和抑制等功能。
"""

import os
import pytest
import tempfile
import yaml
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

from harborai.core.alerts.alert_config_loader import AlertConfigManager
from harborai.core.alerts.alert_manager import AlertManager
from harborai.core.alerts.notification_manager import NotificationManager
from harborai.core.alerts.suppression_manager import SuppressionManager
from harborai.core.alerts.threshold_manager import ThresholdManager


class TestAlertConfigIntegration:
    """告警配置集成测试"""
    
    @pytest.fixture
    def comprehensive_config(self):
        """创建完整的告警配置"""
        return {
            "global": {
                "enabled": True,
                "defaults": {
                    "check_interval": 30,
                    "evaluation_window": 300,
                    "retention_days": 30
                },
                "timezone": "Asia/Shanghai",
                "data_sources": {
                    "prometheus": {
                        "enabled": True,
                        "url": "http://localhost:9090"
                    },
                    "database": {
                        "enabled": True,
                        "connection_string": "postgresql://test:test@localhost:5432/test"
                    }
                }
            },
            "alert_rules": {
                "system_resources": [
                    {
                        "id": "cpu_usage_critical",
                        "name": "CPU使用率严重告警",
                        "type": "threshold",
                        "severity": "critical",
                        "metric": "system.cpu.usage_percent",
                        "conditions": [
                            {
                                "operator": ">",
                                "threshold": 90,
                                "duration": "5m"
                            }
                        ],
                        "labels": {
                            "component": "system",
                            "team": "infrastructure"
                        }
                    },
                    {
                        "id": "memory_usage_high",
                        "name": "内存使用率高告警",
                        "type": "threshold",
                        "severity": "high",
                        "metric": "system.memory.usage_percent",
                        "conditions": [
                            {
                                "operator": ">",
                                "threshold": 85,
                                "duration": "3m"
                            }
                        ]
                    }
                ],
                "application_performance": [
                    {
                        "id": "api_response_time_high",
                        "name": "API响应时间过高",
                        "type": "threshold",
                        "severity": "high",
                        "metric": "http.request.duration",
                        "conditions": [
                            {
                                "operator": ">",
                                "threshold": 2000,
                                "duration": "2m"
                            }
                        ]
                    },
                    {
                        "id": "error_rate_high",
                        "name": "错误率过高",
                        "type": "threshold",
                        "severity": "critical",
                        "metric": "http.request.error_rate",
                        "conditions": [
                            {
                                "operator": ">",
                                "threshold": 5,
                                "duration": "1m"
                            }
                        ]
                    }
                ]
            },
            "notifications": {
                "channels": {
                    "email": {
                        "enabled": True,
                        "smtp_server": "smtp.example.com",
                        "smtp_port": 587,
                        "username": "alerts@example.com",
                        "password": "password",
                        "from_address": "alerts@example.com",
                        "retry_policy": {
                            "max_retries": 3,
                            "retry_delay": 30
                        }
                    },
                    "dingtalk": {
                        "enabled": True,
                        "webhook_url": "https://oapi.dingtalk.com/robot/send?access_token=test",
                        "secret": "test_secret",
                        "retry_policy": {
                            "max_retries": 2,
                            "retry_delay": 10
                        }
                    },
                    "slack": {
                        "enabled": False,
                        "webhook_url": "https://hooks.slack.com/services/test",
                        "channel": "#alerts"
                    }
                },
                "rules": [
                    {
                        "id": "critical_alerts",
                        "name": "关键告警通知",
                        "severity_levels": ["critical"],
                        "channels": ["email", "dingtalk"],
                        "recipients": ["admin@example.com", "ops@example.com"],
                        "rate_limit": {
                            "max_notifications": 10,
                            "time_window": 3600
                        },
                        "escalation": {
                            "enabled": True,
                            "escalation_time": 900
                        }
                    },
                    {
                        "id": "high_alerts",
                        "name": "高级告警通知",
                        "severity_levels": ["high"],
                        "channels": ["email"],
                        "recipients": ["ops@example.com"],
                        "rate_limit": {
                            "max_notifications": 20,
                            "time_window": 3600
                        }
                    },
                    {
                        "id": "business_hours_alerts",
                        "name": "工作时间告警",
                        "severity_levels": ["medium", "high", "critical"],
                        "channels": ["email"],
                        "schedule": {
                            "enabled": True,
                            "business_hours": {
                                "start": "09:00",
                                "end": "18:00",
                                "weekdays": [1, 2, 3, 4, 5]
                            }
                        }
                    }
                ]
            },
            "suppression": {
                "rules": [
                    {
                        "id": "duplicate_suppression",
                        "name": "重复告警抑制",
                        "type": "duplicate",
                        "action": "suppress",
                        "time_window": 300,
                        "match_fields": ["alert_id", "instance"]
                    },
                    {
                        "id": "maintenance_window",
                        "name": "维护窗口抑制",
                        "type": "maintenance",
                        "action": "suppress",
                        "schedule": {
                            "start_time": "02:00",
                            "end_time": "04:00",
                            "weekdays": [0, 6]
                        }
                    },
                    {
                        "id": "dependency_suppression",
                        "name": "依赖关系抑制",
                        "type": "dependency",
                        "action": "suppress",
                        "dependencies": {
                            "database_down": ["api_response_time_high", "error_rate_high"],
                            "network_issue": ["cpu_usage_critical"]
                        }
                    }
                ]
            },
            "escalation": {
                "policies": [
                    {
                        "id": "critical_escalation",
                        "name": "关键告警升级策略",
                        "trigger_severities": ["critical"],
                        "levels": [
                            {
                                "level": 1,
                                "delay": "5m",
                                "channels": ["email"],
                                "recipients": ["oncall@example.com"]
                            },
                            {
                                "level": 2,
                                "delay": "15m",
                                "channels": ["email", "dingtalk"],
                                "recipients": ["manager@example.com"]
                            },
                            {
                                "level": 3,
                                "delay": "30m",
                                "channels": ["email"],
                                "recipients": ["director@example.com"]
                            }
                        ]
                    },
                    {
                        "id": "high_escalation",
                        "name": "高级告警升级策略",
                        "trigger_severities": ["high"],
                        "levels": [
                            {
                                "level": 1,
                                "delay": "10m",
                                "channels": ["email"],
                                "recipients": ["oncall@example.com"]
                            },
                            {
                                "level": 2,
                                "delay": "30m",
                                "channels": ["email"],
                                "recipients": ["manager@example.com"]
                            }
                        ]
                    }
                ]
            },
            "thresholds": {
                "static": {
                    "cpu_usage": 80,
                    "memory_usage": 85,
                    "disk_usage": 90,
                    "api_response_time": 1000,
                    "error_rate": 2
                },
                "dynamic": {
                    "enabled": True,
                    "learning_period_days": 7,
                    "sensitivity": 0.8,
                    "min_samples": 100
                },
                "business_hours": {
                    "enabled": True,
                    "multipliers": {
                        "cpu_usage": 1.2,
                        "memory_usage": 1.1,
                        "api_response_time": 0.8
                    }
                }
            }
        }
    
    @pytest.fixture
    def config_file(self, comprehensive_config):
        """创建配置文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(comprehensive_config, f)
            temp_path = f.name
        
        yield temp_path
        
        os.unlink(temp_path)
    
    @pytest.fixture
    def config_manager(self, config_file):
        """创建配置管理器"""
        return AlertConfigManager(config_file)
    
    def test_complete_config_loading_and_validation(self, config_manager):
        """测试完整配置加载和验证"""
        # 加载所有配置
        configs = config_manager.load_all_configs()
        
        # 验证配置结构
        assert "global" in configs
        assert "alert_rules" in configs
        assert "notifications" in configs
        assert "suppression_rules" in configs
        assert "escalation_policies" in configs
        assert "thresholds" in configs
        
        # 验证告警规则
        assert len(configs["alert_rules"]) == 4  # 2个系统资源 + 2个应用性能
        
        # 验证通知配置
        notification_config = configs["notifications"]
        assert len(notification_config["channels"]) == 3
        assert len(notification_config["rules"]) == 3
        
        # 验证抑制规则
        assert len(configs["suppression_rules"]) == 3
        
        # 验证升级策略
        assert len(configs["escalation_policies"]) == 2
        
        # 验证阈值配置
        thresholds = configs["thresholds"]
        assert "static" in thresholds
        assert "dynamic" in thresholds
        assert "business_hours" in thresholds
        
        # 验证配置有效性
        validation_result = config_manager.validate_configuration()
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0
    
    def test_alert_rule_categorization(self, config_manager):
        """测试告警规则分类"""
        alert_rules = config_manager.get_alert_rules()
        
        # 按类别分组
        categories = {}
        for rule in alert_rules:
            category = rule.get("category", "unknown")
            if category not in categories:
                categories[category] = []
            categories[category].append(rule)
        
        # 验证分类
        assert "system_resources" in categories
        assert "application_performance" in categories
        assert len(categories["system_resources"]) == 2
        assert len(categories["application_performance"]) == 2
        
        # 验证规则内容
        cpu_rule = next(rule for rule in categories["system_resources"] 
                       if rule["id"] == "cpu_usage_critical")
        assert cpu_rule["severity"] == "critical"
        assert cpu_rule["metric"] == "system.cpu.usage_percent"
        
        api_rule = next(rule for rule in categories["application_performance"] 
                       if rule["id"] == "api_response_time_high")
        assert api_rule["severity"] == "high"
        assert api_rule["metric"] == "http.request.duration"
    
    def test_notification_channel_configuration(self, config_manager):
        """测试通知渠道配置"""
        notification_config = config_manager.get_notification_config()
        channels = notification_config["channels"]
        
        # 验证邮件渠道
        email_config = channels["email"]
        assert email_config["enabled"] is True
        assert email_config["smtp_server"] == "smtp.example.com"
        assert email_config["retry_policy"]["max_retries"] == 3
        
        # 验证钉钉渠道
        dingtalk_config = channels["dingtalk"]
        assert dingtalk_config["enabled"] is True
        assert "webhook_url" in dingtalk_config
        assert "secret" in dingtalk_config
        
        # 验证Slack渠道（已禁用）
        slack_config = channels["slack"]
        assert slack_config["enabled"] is False
    
    def test_notification_rules_and_routing(self, config_manager):
        """测试通知规则和路由"""
        notification_config = config_manager.get_notification_config()
        rules = notification_config["rules"]
        
        # 验证关键告警规则
        critical_rule = next(rule for rule in rules if rule["id"] == "critical_alerts")
        assert "critical" in critical_rule["severity_levels"]
        assert "email" in critical_rule["channels"]
        assert "dingtalk" in critical_rule["channels"]
        assert critical_rule["rate_limit"]["max_notifications"] == 10
        
        # 验证高级告警规则
        high_rule = next(rule for rule in rules if rule["id"] == "high_alerts")
        assert "high" in high_rule["severity_levels"]
        assert len(high_rule["channels"]) == 1
        assert high_rule["channels"][0] == "email"
        
        # 验证工作时间告警规则
        business_rule = next(rule for rule in rules if rule["id"] == "business_hours_alerts")
        assert "schedule" in business_rule
        assert business_rule["schedule"]["enabled"] is True
        assert business_rule["schedule"]["business_hours"]["start"] == "09:00"
    
    def test_suppression_rules_configuration(self, config_manager):
        """测试抑制规则配置"""
        suppression_rules = config_manager.get_suppression_rules()
        
        # 验证重复告警抑制
        duplicate_rule = next(rule for rule in suppression_rules 
                             if rule["id"] == "duplicate_suppression")
        assert duplicate_rule["type"] == "duplicate"
        assert duplicate_rule["time_window"] == 300
        assert "alert_id" in duplicate_rule["match_fields"]
        
        # 验证维护窗口抑制
        maintenance_rule = next(rule for rule in suppression_rules 
                               if rule["id"] == "maintenance_window")
        assert maintenance_rule["type"] == "maintenance"
        assert "schedule" in maintenance_rule
        
        # 验证依赖关系抑制
        dependency_rule = next(rule for rule in suppression_rules 
                              if rule["id"] == "dependency_suppression")
        assert dependency_rule["type"] == "dependency"
        assert "dependencies" in dependency_rule
        assert "database_down" in dependency_rule["dependencies"]
    
    def test_escalation_policies_configuration(self, config_manager):
        """测试升级策略配置"""
        escalation_policies = config_manager.get_escalation_policies()
        
        # 验证关键告警升级策略
        critical_policy = next(policy for policy in escalation_policies 
                              if policy["id"] == "critical_escalation")
        assert "critical" in critical_policy["trigger_severities"]
        assert len(critical_policy["levels"]) == 3
        
        # 验证升级级别
        level1 = critical_policy["levels"][0]
        assert level1["level"] == 1
        assert level1["delay"] == "5m"
        assert "oncall@example.com" in level1["recipients"]
        
        level3 = critical_policy["levels"][2]
        assert level3["level"] == 3
        assert level3["delay"] == "30m"
        assert "director@example.com" in level3["recipients"]
        
        # 验证高级告警升级策略
        high_policy = next(policy for policy in escalation_policies 
                          if policy["id"] == "high_escalation")
        assert "high" in high_policy["trigger_severities"]
        assert len(high_policy["levels"]) == 2
    
    def test_threshold_configuration_types(self, config_manager):
        """测试阈值配置类型"""
        thresholds = config_manager.get_thresholds()
        
        # 验证静态阈值
        static_thresholds = thresholds["static"]
        assert static_thresholds["cpu_usage"] == 80
        assert static_thresholds["memory_usage"] == 85
        assert static_thresholds["api_response_time"] == 1000
        
        # 验证动态阈值
        dynamic_thresholds = thresholds["dynamic"]
        assert dynamic_thresholds["enabled"] is True
        assert dynamic_thresholds["learning_period_days"] == 7
        assert dynamic_thresholds["sensitivity"] == 0.8
        
        # 验证工作时间阈值
        business_thresholds = thresholds["business_hours"]
        assert business_thresholds["enabled"] is True
        assert business_thresholds["multipliers"]["cpu_usage"] == 1.2
        assert business_thresholds["multipliers"]["api_response_time"] == 0.8
    
    def test_config_export_and_import(self, config_manager):
        """测试配置导出和导入"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 导出YAML格式
            yaml_path = os.path.join(temp_dir, "exported_config.yaml")
            success = config_manager.export_config(yaml_path, "yaml")
            assert success is True
            assert os.path.exists(yaml_path)
            
            # 导出JSON格式
            json_path = os.path.join(temp_dir, "exported_config.json")
            success = config_manager.export_config(json_path, "json")
            assert success is True
            assert os.path.exists(json_path)
            
            # 验证导出的YAML可以重新加载
            new_manager = AlertConfigManager(yaml_path)
            new_configs = new_manager.load_all_configs()
            
            # 比较关键配置项
            original_configs = config_manager.load_all_configs()
            assert len(new_configs["alert_rules"]) == len(original_configs["alert_rules"])
            assert len(new_configs["suppression_rules"]) == len(original_configs["suppression_rules"])
    
    def test_config_summary_generation(self, config_manager):
        """测试配置摘要生成"""
        summary = config_manager.get_config_summary()
        
        # 验证摘要内容
        assert summary["alert_rules_count"] == 4
        assert summary["notification_rules_count"] == 3
        assert summary["suppression_rules_count"] == 3
        assert summary["escalation_policies_count"] == 2
        assert summary["global_enabled"] is True
        
        # 验证启用的渠道
        enabled_channels = summary["enabled_channels"]
        assert "email" in enabled_channels
        assert "dingtalk" in enabled_channels
        assert "slack" not in enabled_channels  # 已禁用
        
        # 验证时间戳
        assert "last_loaded" in summary
        assert isinstance(summary["last_loaded"], str)
    
    @patch('harborai.core.alerts.alert_config_loader.AlertConfigLoader.load_config')
    def test_config_reload_with_watchers(self, mock_load_config, config_manager):
        """测试配置重新加载和观察者通知"""
        # 模拟配置变更
        original_config = {"global": {"enabled": True}}
        updated_config = {"global": {"enabled": False}}
        
        mock_load_config.side_effect = [original_config, updated_config]
        
        # 添加配置观察者
        watcher_calls = []
        
        def config_watcher(config):
            watcher_calls.append(config)
        
        config_manager.add_config_watcher(config_watcher)
        
        # 初始加载
        config_manager.load_all_configs()
        
        # 重新加载配置
        config_manager.reload_configuration()
        
        # 验证观察者被调用
        assert len(watcher_calls) == 1
        assert watcher_calls[0]["global"]["enabled"] is False
    
    def test_config_validation_with_errors(self, config_file):
        """测试包含错误的配置验证"""
        # 创建包含错误的配置
        invalid_config = {
            "global": {"enabled": True},
            "alert_rules": {
                "invalid_rules": [
                    {
                        # 缺少必需字段
                        "name": "无效规则",
                        "type": "threshold"
                        # 缺少id, severity, metric
                    }
                ]
            },
            "notifications": {
                "channels": {},  # 空渠道配置
                "rules": [
                    {
                        "id": "invalid_rule",
                        "severity_levels": ["invalid_severity"],  # 无效严重级别
                        "channels": ["nonexistent_channel"]  # 不存在的渠道
                    }
                ]
            }
        }
        
        # 写入无效配置
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = AlertConfigManager(config_file)
        
        # 验证配置应该失败
        validation_result = config_manager.validate_configuration()
        
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
        
        # 检查特定错误
        error_messages = " ".join(validation_result.errors)
        assert "缺少必需字段" in error_messages
        assert "无效的严重级别" in error_messages or "不存在的通知渠道" in error_messages
    
    def test_performance_with_large_config(self):
        """测试大型配置的性能"""
        # 创建大型配置（模拟生产环境）
        large_config = {
            "global": {"enabled": True},
            "alert_rules": {},
            "notifications": {"channels": {}, "rules": []},
            "suppression": {"rules": []},
            "escalation": {"policies": []},
            "thresholds": {"static": {}}
        }
        
        # 生成大量告警规则
        for category in ["system", "application", "database", "network"]:
            large_config["alert_rules"][f"{category}_rules"] = []
            for i in range(50):  # 每个类别50个规则
                rule = {
                    "id": f"{category}_rule_{i}",
                    "name": f"{category.title()} Rule {i}",
                    "type": "threshold",
                    "severity": ["low", "medium", "high", "critical"][i % 4],
                    "metric": f"{category}.metric_{i}",
                    "conditions": [{"operator": ">", "threshold": i * 10}]
                }
                large_config["alert_rules"][f"{category}_rules"].append(rule)
        
        # 生成大量通知规则
        for i in range(20):
            rule = {
                "id": f"notification_rule_{i}",
                "name": f"Notification Rule {i}",
                "severity_levels": ["high", "critical"],
                "channels": ["email"]
            }
            large_config["notifications"]["rules"].append(rule)
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(large_config, f)
            temp_path = f.name
        
        try:
            # 测试加载性能
            start_time = time.time()
            config_manager = AlertConfigManager(temp_path)
            configs = config_manager.load_all_configs()
            load_time = time.time() - start_time
            
            # 验证加载时间合理（应该在几秒内）
            assert load_time < 10.0, f"配置加载时间过长: {load_time}秒"
            
            # 验证配置正确加载
            assert len(configs["alert_rules"]) == 200  # 4个类别 * 50个规则
            assert len(configs["notifications"]["rules"]) == 20
            
            # 测试验证性能
            start_time = time.time()
            validation_result = config_manager.validate_configuration()
            validation_time = time.time() - start_time
            
            # 验证时间应该合理
            assert validation_time < 5.0, f"配置验证时间过长: {validation_time}秒"
            assert validation_result.is_valid is True
            
        finally:
            os.unlink(temp_path)