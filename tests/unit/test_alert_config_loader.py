"""
告警配置加载器测试

测试告警配置的加载、验证和管理功能。
"""

import os
import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from harborai.core.alerts.alert_config_loader import (
    AlertConfigLoader,
    AlertConfigManager,
    ValidationResult,
    ConfigValidationError,
    ConfigLoadError
)


class TestAlertConfigLoader:
    """告警配置加载器测试"""
    
    @pytest.fixture
    def temp_config_file(self):
        """创建临时配置文件"""
        config_data = {
            "global": {
                "enabled": True,
                "defaults": {
                    "check_interval": 60,
                    "evaluation_window": 300
                }
            },
            "alert_rules": {
                "system_resources": [
                    {
                        "id": "cpu_usage_high",
                        "name": "CPU使用率过高",
                        "type": "threshold",
                        "severity": "high",
                        "metric": "system.cpu.usage_percent",
                        "conditions": [
                            {
                                "operator": ">",
                                "threshold": 80,
                                "duration": "5m"
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
                        "username": "test@example.com",
                        "password": "password",
                        "from_address": "alerts@example.com"
                    }
                },
                "rules": [
                    {
                        "id": "critical_alerts",
                        "name": "关键告警通知",
                        "severity_levels": ["critical"],
                        "channels": ["email"]
                    }
                ]
            },
            "suppression": {
                "rules": [
                    {
                        "id": "duplicate_suppression",
                        "name": "重复告警抑制",
                        "type": "duplicate",
                        "action": "suppress"
                    }
                ]
            },
            "escalation": {
                "policies": [
                    {
                        "id": "critical_escalation",
                        "name": "关键告警升级",
                        "trigger_severities": ["critical"],
                        "levels": [
                            {
                                "level": 1,
                                "delay": "5m",
                                "channels": ["email"]
                            }
                        ]
                    }
                ]
            },
            "thresholds": {
                "static": {
                    "cpu_usage": 80,
                    "memory_usage": 85
                },
                "dynamic": {
                    "enabled": True,
                    "learning_period_days": 7,
                    "sensitivity": 0.8
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # 清理
        os.unlink(temp_path)
    
    @pytest.fixture
    def config_loader(self, temp_config_file):
        """创建配置加载器实例"""
        return AlertConfigLoader(temp_config_file)
    
    def test_load_config_success(self, config_loader):
        """测试成功加载配置"""
        config = config_loader.load_config()
        
        assert config is not None
        assert config["global"]["enabled"] is True
        assert "alert_rules" in config
        assert "notifications" in config
        assert "suppression" in config
        assert "escalation" in config
        assert "thresholds" in config
    
    def test_load_config_with_cache(self, config_loader):
        """测试配置缓存功能"""
        # 第一次加载
        config1 = config_loader.load_config()
        
        # 第二次加载应该使用缓存
        config2 = config_loader.load_config()
        
        assert config1 is config2  # 应该是同一个对象
    
    def test_load_config_force_reload(self, config_loader):
        """测试强制重新加载配置"""
        # 第一次加载
        config1 = config_loader.load_config()
        
        # 强制重新加载
        config2 = config_loader.load_config(force_reload=True)
        
        assert config1 is not config2  # 应该是不同的对象
        assert config1 == config2  # 但内容应该相同
    
    def test_load_config_file_not_exists(self):
        """测试配置文件不存在的情况"""
        loader = AlertConfigLoader("nonexistent.yaml")
        
        with pytest.raises(ConfigLoadError, match="配置文件不存在"):
            loader.load_config()
    
    def test_load_config_invalid_yaml(self):
        """测试无效YAML格式"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            temp_path = f.name
        
        try:
            loader = AlertConfigLoader(temp_path)
            with pytest.raises(ConfigLoadError, match="YAML解析错误"):
                loader.load_config()
        finally:
            os.unlink(temp_path)
    
    def test_load_config_empty_file(self):
        """测试空配置文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name
        
        try:
            loader = AlertConfigLoader(temp_path)
            with pytest.raises(ConfigLoadError, match="配置文件为空"):
                loader.load_config()
        finally:
            os.unlink(temp_path)
    
    @patch.dict(os.environ, {"TEST_VAR": "test_value", "SMTP_SERVER": "smtp.test.com"})
    def test_env_var_substitution(self):
        """测试环境变量替换"""
        config_data = {
            "global": {"enabled": True},
            "test_config": {
                "simple_var": "${TEST_VAR}",
                "var_with_default": "${NONEXISTENT:default_value}",
                "smtp_server": "${SMTP_SERVER}"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = AlertConfigLoader(temp_path)
            config = loader.load_config()
            
            assert config["test_config"]["simple_var"] == "test_value"
            assert config["test_config"]["var_with_default"] == "default_value"
            assert config["test_config"]["smtp_server"] == "smtp.test.com"
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_success(self, config_loader):
        """测试配置验证成功"""
        config = config_loader.load_config()
        result = config_loader.validate_config(config)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_config_missing_required_fields(self):
        """测试缺少必需字段的配置验证"""
        config_data = {
            "global": {"enabled": True},
            "alert_rules": {
                "test_rules": [
                    {
                        "name": "测试规则",  # 缺少id字段
                        "type": "threshold",
                        "severity": "high"
                        # 缺少metric字段
                    }
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = AlertConfigLoader(temp_path)
            config = loader.load_config(validate=False)  # 跳过加载时验证
            result = loader.validate_config(config)
            
            assert result.is_valid is False
            assert any("'id' is a required property" in error for error in result.errors)
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_duplicate_rule_ids(self):
        """测试重复规则ID的验证"""
        config_data = {
            "global": {"enabled": True},
            "alert_rules": {
                "test_rules": [
                    {
                        "id": "duplicate_id",
                        "name": "规则1",
                        "type": "threshold",
                        "severity": "high",
                        "metric": "test.metric1"
                    },
                    {
                        "id": "duplicate_id",  # 重复ID
                        "name": "规则2",
                        "type": "threshold",
                        "severity": "medium",
                        "metric": "test.metric2"
                    }
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = AlertConfigLoader(temp_path)
            config = loader.load_config(validate=False)  # 跳过加载时验证
            result = loader.validate_config(config)
            
            assert result.is_valid is False
            assert any("duplicate_id" in error for error in result.errors)
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_invalid_severity(self):
        """测试无效严重级别的验证"""
        config_data = {
            "global": {"enabled": True},
            "alert_rules": {
                "test_rules": [
                    {
                        "id": "test_rule",
                        "name": "测试规则",
                        "type": "threshold",
                        "severity": "invalid_severity",  # 无效严重级别
                        "metric": "test.metric"
                    }
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loader = AlertConfigLoader(temp_path)
            config = loader.load_config(validate=False)  # 跳过加载时验证
            result = loader.validate_config(config)
            
            assert result.is_valid is False
            assert any("'invalid_severity' is not one of" in error for error in result.errors)
        finally:
            os.unlink(temp_path)
    
    def test_get_alert_rules(self, config_loader):
        """测试获取告警规则"""
        rules = config_loader.get_alert_rules()
        
        assert len(rules) == 1
        assert rules[0]["id"] == "cpu_usage_high"
        assert rules[0]["category"] == "system_resources"
    
    def test_get_notification_config(self, config_loader):
        """测试获取通知配置"""
        notification_config = config_loader.get_notification_config()
        
        assert "channels" in notification_config
        assert "rules" in notification_config
        assert notification_config["channels"]["email"]["enabled"] is True
    
    def test_get_suppression_rules(self, config_loader):
        """测试获取抑制规则"""
        suppression_rules = config_loader.get_suppression_rules()
        
        assert len(suppression_rules) == 1
        assert suppression_rules[0]["id"] == "duplicate_suppression"
    
    def test_get_escalation_policies(self, config_loader):
        """测试获取升级策略"""
        escalation_policies = config_loader.get_escalation_policies()
        
        assert len(escalation_policies) == 1
        assert escalation_policies[0]["id"] == "critical_escalation"
    
    def test_get_thresholds(self, config_loader):
        """测试获取阈值配置"""
        thresholds = config_loader.get_thresholds()
        
        assert "static" in thresholds
        assert "dynamic" in thresholds
        assert thresholds["static"]["cpu_usage"] == 80
        assert thresholds["dynamic"]["enabled"] is True
    
    def test_get_global_config(self, config_loader):
        """测试获取全局配置"""
        global_config = config_loader.get_global_config()
        
        assert global_config["enabled"] is True
        assert global_config["defaults"]["check_interval"] == 60


class TestAlertConfigManager:
    """告警配置管理器测试"""
    
    @pytest.fixture
    def temp_config_file(self):
        """创建临时配置文件"""
        config_data = {
            "global": {
                "enabled": True,
                "defaults": {"check_interval": 60}
            },
            "alert_rules": {
                "test_rules": [
                    {
                        "id": "test_rule",
                        "name": "测试规则",
                        "type": "threshold",
                        "severity": "high",
                        "metric": "test.metric"
                    }
                ]
            },
            "notifications": {
                "channels": {
                    "email": {
                        "enabled": True,
                        "smtp_server": "smtp.example.com",
                        "smtp_port": 587,
                        "username": "test@example.com",
                        "password": "password",
                        "from_address": "test@example.com"
                    }
                },
                "rules": []
            },
            "suppression": {"rules": []},
            "escalation": {"policies": []},
            "thresholds": {"static": {}}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        os.unlink(temp_path)
    
    @pytest.fixture
    def config_manager(self, temp_config_file):
        """创建配置管理器实例"""
        return AlertConfigManager(temp_config_file)
    
    def test_load_all_configs(self, config_manager):
        """测试加载所有配置"""
        # 先加载配置（跳过验证）
        config_manager.loader.load_config(validate=False)
        
        configs = config_manager.load_all_configs()
        
        assert "global" in configs
        assert "alert_rules" in configs
        assert "notifications" in configs
        assert "suppression_rules" in configs
        assert "escalation_policies" in configs
        assert "thresholds" in configs
        assert "raw_config" in configs
        
        assert len(configs["alert_rules"]) == 1
        assert configs["alert_rules"][0]["id"] == "test_rule"
    
    def test_validate_configuration(self, config_manager):
        """测试配置验证"""
        # 先加载配置（跳过验证）
        config_manager.loader.load_config(validate=False)
        
        result = config_manager.validate_configuration()
        
        assert isinstance(result, ValidationResult)
        # 配置可能有警告但应该是有效的
        assert result.is_valid is True or len(result.errors) == 0
    
    def test_reload_configuration(self, config_manager):
        """测试重新加载配置"""
        # 先加载配置（跳过验证）
        config_manager.loader.load_config(validate=False)
        
        # 添加配置观察者
        watcher_called = False
        
        def config_watcher(config):
            nonlocal watcher_called
            watcher_called = True
        
        config_manager.add_config_watcher(config_watcher)
        
        # 重新加载配置
        config = config_manager.reload_configuration()
        
        assert config is not None
        assert watcher_called is True
    
    def test_config_watcher_management(self, config_manager):
        """测试配置观察者管理"""
        def watcher1(config):
            pass
        
        def watcher2(config):
            pass
        
        # 添加观察者
        config_manager.add_config_watcher(watcher1)
        config_manager.add_config_watcher(watcher2)
        
        assert len(config_manager._watchers) == 2
        
        # 移除观察者
        config_manager.remove_config_watcher(watcher1)
        
        assert len(config_manager._watchers) == 1
        assert watcher2 in config_manager._watchers
    
    def test_export_config_yaml(self, config_manager):
        """测试导出YAML配置"""
        # 先加载配置（跳过验证）
        config_manager.loader.load_config(validate=False)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "exported_config.yaml")
            
            success = config_manager.export_config(output_path, "yaml")
            
            assert success is True
            assert os.path.exists(output_path)
            
            # 验证导出的文件可以正常加载
            with open(output_path, 'r', encoding='utf-8') as f:
                exported_config = yaml.safe_load(f)
            
            assert exported_config is not None
            assert "global" in exported_config
    
    def test_export_config_json(self, config_manager):
        """测试导出JSON配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "exported_config.json")
            
            success = config_manager.export_config(output_path, "json")
            
            assert success is True
            assert os.path.exists(output_path)
            
            # 验证导出的文件可以正常加载
            import json
            with open(output_path, 'r', encoding='utf-8') as f:
                exported_config = json.load(f)
            
            assert exported_config is not None
            assert "global" in exported_config
    
    def test_export_config_invalid_format(self, config_manager):
        """测试导出无效格式"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "exported_config.xml")
            
            success = config_manager.export_config(output_path, "xml")
            
            assert success is False
    
    def test_get_config_summary(self, config_manager):
        """测试获取配置摘要"""
        # 先加载配置（跳过验证）
        config_manager.loader.load_config(validate=False)
        
        summary = config_manager.get_config_summary()
        
        assert "alert_rules_count" in summary
        assert "notification_rules_count" in summary
        assert "suppression_rules_count" in summary
        assert "escalation_policies_count" in summary
        assert "enabled_channels" in summary
        assert "global_enabled" in summary
        assert "last_loaded" in summary
        
        assert summary["alert_rules_count"] == 1
        assert summary["global_enabled"] is True


class TestValidationResult:
    """验证结果测试"""
    
    def test_validation_result_initial_state(self):
        """测试验证结果初始状态"""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_add_error(self):
        """测试添加错误"""
        result = ValidationResult(is_valid=True)
        
        result.add_error("测试错误")
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0] == "测试错误"
    
    def test_add_warning(self):
        """测试添加警告"""
        result = ValidationResult(is_valid=True)
        
        result.add_warning("测试警告")
        
        assert result.is_valid is True  # 警告不影响有效性
        assert len(result.warnings) == 1
        assert result.warnings[0] == "测试警告"
    
    def test_multiple_errors_and_warnings(self):
        """测试多个错误和警告"""
        result = ValidationResult(is_valid=True)
        
        result.add_error("错误1")
        result.add_error("错误2")
        result.add_warning("警告1")
        result.add_warning("警告2")
        
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 2