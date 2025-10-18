"""
告警配置加载器

负责加载、验证和管理告警系统的统一配置文件，
支持YAML格式配置、环境变量替换、配置验证和热重载。
"""

import os
import re
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """配置验证错误"""
    pass


class ConfigLoadError(Exception):
    """配置加载错误"""
    pass


@dataclass
class ValidationResult:
    """配置验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)


class AlertConfigLoader:
    """告警配置加载器"""
    
    def __init__(self, config_path: str = "config/alerts/alert_system_config.yaml"):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self._config_cache: Optional[Dict[str, Any]] = None
        self._last_modified: Optional[float] = None
        
        # 配置验证schema
        self._schema = self._load_config_schema()
    
    def load_config(self, force_reload: bool = False, validate: bool = True) -> Dict[str, Any]:
        """
        加载告警配置
        
        Args:
            force_reload: 是否强制重新加载
            validate: 是否进行配置验证
            
        Returns:
            配置字典
            
        Raises:
            ConfigLoadError: 配置加载失败
        """
        try:
            # 检查是否需要重新加载
            if not force_reload and self._should_use_cache():
                return self._config_cache
            
            # 检查配置文件是否存在
            if not self.config_path.exists():
                raise ConfigLoadError(f"配置文件不存在: {self.config_path}")
            
            # 读取配置文件
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = yaml.safe_load(f)
            
            if not raw_config:
                raise ConfigLoadError("配置文件为空")
            
            # 环境变量替换
            config = self._substitute_env_vars(raw_config)
            
            # 可选的配置验证
            if validate:
                validation_result = self.validate_config(config)
                if not validation_result.is_valid:
                    error_msg = "配置验证失败:\n" + "\n".join(validation_result.errors)
                    raise ConfigValidationError(error_msg)
                
                # 记录警告
                for warning in validation_result.warnings:
                    self.logger.warning(f"配置警告: {warning}")
            
            # 缓存配置
            self._config_cache = config
            self._last_modified = self.config_path.stat().st_mtime
            
            self.logger.info(f"成功加载告警配置: {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigLoadError(f"YAML解析错误: {e}")
        except Exception as e:
            raise ConfigLoadError(f"配置加载失败: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        验证配置
        
        Args:
            config: 配置字典
            
        Returns:
            验证结果
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # JSON Schema验证
            validate(instance=config, schema=self._schema)
            
            # 自定义验证规则
            self._validate_alert_rules(config, result)
            self._validate_notification_config(config, result)
            self._validate_suppression_rules(config, result)
            self._validate_escalation_policies(config, result)
            self._validate_thresholds(config, result)
            
        except ValidationError as e:
            result.add_error(f"Schema验证失败: {e.message}")
        except Exception as e:
            result.add_error(f"配置验证异常: {e}")
        
        return result
    
    def get_alert_rules(self) -> List[Dict[str, Any]]:
        """获取告警规则配置"""
        config = self.load_config()
        rules = []
        
        for category, rule_list in config.get("alert_rules", {}).items():
            for rule in rule_list:
                rule["category"] = category
                rules.append(rule)
        
        return rules
    
    def get_notification_config(self) -> Dict[str, Any]:
        """获取通知配置"""
        config = self.load_config()
        return config.get("notifications", {})
    
    def get_suppression_rules(self) -> List[Dict[str, Any]]:
        """获取抑制规则配置"""
        config = self.load_config()
        return config.get("suppression", {}).get("rules", [])
    
    def get_escalation_policies(self) -> List[Dict[str, Any]]:
        """获取升级策略配置"""
        config = self.load_config()
        return config.get("escalation", {}).get("policies", [])
    
    def get_thresholds(self) -> Dict[str, Any]:
        """获取阈值配置"""
        config = self.load_config()
        return config.get("thresholds", {})
    
    def get_global_config(self) -> Dict[str, Any]:
        """获取全局配置"""
        config = self.load_config()
        return config.get("global", {})
    
    def reload_config(self) -> Dict[str, Any]:
        """重新加载配置"""
        return self.load_config(force_reload=True)
    
    def _should_use_cache(self) -> bool:
        """检查是否应该使用缓存"""
        if self._config_cache is None or self._last_modified is None:
            return False
        
        if not self.config_path.exists():
            return False
        
        current_mtime = self.config_path.stat().st_mtime
        return current_mtime == self._last_modified
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        递归替换环境变量
        
        支持格式: ${VAR_NAME} 或 ${VAR_NAME:default_value}
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_var_string(obj)
        else:
            return obj
    
    def _substitute_env_var_string(self, text: str) -> str:
        """替换字符串中的环境变量"""
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name, default_value = var_expr, ''
            
            return os.getenv(var_name, default_value)
        
        return re.sub(pattern, replace_var, text)
    
    def _validate_alert_rules(self, config: Dict[str, Any], result: ValidationResult):
        """验证告警规则配置"""
        alert_rules = config.get("alert_rules", {})
        
        if not alert_rules:
            result.add_warning("未配置告警规则")
            return
        
        rule_ids = set()
        
        for category, rules in alert_rules.items():
            if not isinstance(rules, list):
                result.add_error(f"告警规则类别 '{category}' 必须是列表")
                continue
            
            for i, rule in enumerate(rules):
                rule_path = f"alert_rules.{category}[{i}]"
                
                # 检查必需字段
                required_fields = ["id", "name", "type", "severity", "metric"]
                for field in required_fields:
                    if field not in rule:
                        result.add_error(f"{rule_path}: 缺少必需字段 '{field}'")
                
                # 检查规则ID唯一性
                rule_id = rule.get("id")
                if rule_id:
                    if rule_id in rule_ids:
                        result.add_error(f"{rule_path}: 规则ID '{rule_id}' 重复")
                    else:
                        rule_ids.add(rule_id)
                
                # 验证严重级别
                severity = rule.get("severity")
                if severity and severity not in ["critical", "high", "medium", "low", "info"]:
                    result.add_error(f"{rule_path}: 无效的严重级别 '{severity}'")
                
                # 验证条件
                conditions = rule.get("conditions", [])
                if not conditions:
                    result.add_warning(f"{rule_path}: 未配置告警条件")
                
                for j, condition in enumerate(conditions):
                    if isinstance(condition, dict):
                        if "operator" in condition and "threshold" in condition:
                            operator = condition["operator"]
                            if operator not in [">", "<", ">=", "<=", "==", "!="]:
                                result.add_error(f"{rule_path}.conditions[{j}]: 无效的操作符 '{operator}'")
    
    def _validate_notification_config(self, config: Dict[str, Any], result: ValidationResult):
        """验证通知配置"""
        notifications = config.get("notifications", {})
        
        if not notifications:
            result.add_warning("未配置通知设置")
            return
        
        # 验证通知渠道
        channels = notifications.get("channels", {})
        for channel_name, channel_config in channels.items():
            if not isinstance(channel_config, dict):
                result.add_error(f"通知渠道 '{channel_name}' 配置必须是字典")
                continue
            
            enabled = channel_config.get("enabled", False)
            if enabled:
                # 检查必需的配置项
                if channel_name == "email":
                    required = ["smtp_server", "smtp_port", "username", "password", "from_address"]
                    for field in required:
                        if not channel_config.get(field):
                            result.add_error(f"邮件通知渠道缺少必需配置: {field}")
                
                elif channel_name == "dingtalk":
                    if not channel_config.get("webhook_url"):
                        result.add_error("钉钉通知渠道缺少webhook_url配置")
                
                elif channel_name == "slack":
                    if not channel_config.get("webhook_url"):
                        result.add_error("Slack通知渠道缺少webhook_url配置")
        
        # 验证通知规则
        rules = notifications.get("rules", [])
        for i, rule in enumerate(rules):
            rule_path = f"notifications.rules[{i}]"
            
            # 检查必需字段
            required_fields = ["id", "name", "severity_levels", "channels"]
            for field in required_fields:
                if field not in rule:
                    result.add_error(f"{rule_path}: 缺少必需字段 '{field}'")
            
            # 验证严重级别
            severity_levels = rule.get("severity_levels", [])
            valid_severities = ["critical", "high", "medium", "low", "info"]
            for severity in severity_levels:
                if severity not in valid_severities:
                    result.add_error(f"{rule_path}: 无效的严重级别 '{severity}'")
            
            # 验证通知渠道
            rule_channels = rule.get("channels", [])
            available_channels = list(channels.keys())
            for channel in rule_channels:
                if channel not in available_channels:
                    result.add_error(f"{rule_path}: 未定义的通知渠道 '{channel}'")
    
    def _validate_suppression_rules(self, config: Dict[str, Any], result: ValidationResult):
        """验证抑制规则配置"""
        suppression = config.get("suppression", {})
        
        if not suppression:
            result.add_warning("未配置抑制规则")
            return
        
        rules = suppression.get("rules", [])
        for i, rule in enumerate(rules):
            rule_path = f"suppression.rules[{i}]"
            
            # 检查必需字段
            required_fields = ["id", "name", "type", "action"]
            for field in required_fields:
                if field not in rule:
                    result.add_error(f"{rule_path}: 缺少必需字段 '{field}'")
            
            # 验证抑制类型
            rule_type = rule.get("type")
            valid_types = ["duplicate", "maintenance", "dependency", "frequency", "correlation"]
            if rule_type and rule_type not in valid_types:
                result.add_error(f"{rule_path}: 无效的抑制类型 '{rule_type}'")
            
            # 验证抑制动作
            action = rule.get("action")
            valid_actions = ["suppress", "delay", "aggregate", "downgrade"]
            if action and action not in valid_actions:
                result.add_error(f"{rule_path}: 无效的抑制动作 '{action}'")
    
    def _validate_escalation_policies(self, config: Dict[str, Any], result: ValidationResult):
        """验证升级策略配置"""
        escalation = config.get("escalation", {})
        
        if not escalation:
            result.add_warning("未配置升级策略")
            return
        
        policies = escalation.get("policies", [])
        for i, policy in enumerate(policies):
            policy_path = f"escalation.policies[{i}]"
            
            # 检查必需字段
            required_fields = ["id", "name", "trigger_severities", "levels"]
            for field in required_fields:
                if field not in policy:
                    result.add_error(f"{policy_path}: 缺少必需字段 '{field}'")
            
            # 验证触发严重级别
            trigger_severities = policy.get("trigger_severities", [])
            valid_severities = ["critical", "high", "medium", "low", "info"]
            for severity in trigger_severities:
                if severity not in valid_severities:
                    result.add_error(f"{policy_path}: 无效的触发严重级别 '{severity}'")
            
            # 验证升级级别
            levels = policy.get("levels", [])
            if not levels:
                result.add_error(f"{policy_path}: 升级策略必须包含至少一个级别")
            
            for j, level in enumerate(levels):
                level_path = f"{policy_path}.levels[{j}]"
                
                # 检查必需字段
                level_required = ["level", "delay", "channels"]
                for field in level_required:
                    if field not in level:
                        result.add_error(f"{level_path}: 缺少必需字段 '{field}'")
                
                # 验证延迟格式
                delay = level.get("delay")
                if delay and not self._is_valid_duration(delay):
                    result.add_error(f"{level_path}: 无效的延迟格式 '{delay}'")
    
    def _validate_thresholds(self, config: Dict[str, Any], result: ValidationResult):
        """验证阈值配置"""
        thresholds = config.get("thresholds", {})
        
        if not thresholds:
            result.add_warning("未配置阈值设置")
            return
        
        # 验证静态阈值
        static = thresholds.get("static", {})
        for metric, threshold in static.items():
            if not isinstance(threshold, (int, float)):
                result.add_error(f"静态阈值 '{metric}' 必须是数值")
        
        # 验证动态阈值
        dynamic = thresholds.get("dynamic", {})
        if dynamic.get("enabled", False):
            learning_period = dynamic.get("learning_period_days")
            if learning_period and (not isinstance(learning_period, int) or learning_period <= 0):
                result.add_error("动态阈值学习周期必须是正整数")
            
            sensitivity = dynamic.get("sensitivity")
            if sensitivity and (not isinstance(sensitivity, (int, float)) or not 0 < sensitivity <= 1):
                result.add_error("动态阈值敏感度必须在0-1之间")
    
    def _is_valid_duration(self, duration: str) -> bool:
        """验证时间间隔格式"""
        pattern = r'^\d+[smhd]$'
        return bool(re.match(pattern, duration))
    
    def _load_config_schema(self) -> Dict[str, Any]:
        """加载配置验证schema"""
        return {
            "type": "object",
            "properties": {
                "global": {
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "defaults": {"type": "object"},
                        "datasources": {"type": "object"}
                    }
                },
                "alert_rules": {
                    "type": "object",
                    "patternProperties": {
                        ".*": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["id", "name", "type", "severity"],
                                "properties": {
                                    "id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "type": {"type": "string"},
                                    "severity": {
                                        "type": "string",
                                        "enum": ["critical", "high", "medium", "low", "info"]
                                    },
                                    "enabled": {"type": "boolean"},
                                    "metric": {"type": "string"},
                                    "conditions": {"type": "array"},
                                    "labels": {"type": "object"},
                                    "annotations": {"type": "object"}
                                }
                            }
                        }
                    }
                },
                "thresholds": {"type": "object"},
                "notifications": {"type": "object"},
                "suppression": {"type": "object"},
                "escalation": {"type": "object"},
                "templates": {"type": "object"},
                "integrations": {"type": "object"},
                "storage": {"type": "object"},
                "performance": {"type": "object"},
                "security": {"type": "object"},
                "logging": {"type": "object"}
            },
            "required": ["global"]
        }


class AlertConfigManager:
    """告警配置管理器"""
    
    def __init__(self, config_path: str = "config/alerts/alert_system_config.yaml"):
        self.loader = AlertConfigLoader(config_path)
        self.logger = logging.getLogger(__name__)
        self._watchers: List[callable] = []
    
    def load_all_configs(self) -> Dict[str, Any]:
        """加载所有配置"""
        try:
            config = self.loader.load_config()
            
            return {
                "global": self.loader.get_global_config(),
                "alert_rules": self.loader.get_alert_rules(),
                "notifications": self.loader.get_notification_config(),
                "suppression_rules": self.loader.get_suppression_rules(),
                "escalation_policies": self.loader.get_escalation_policies(),
                "thresholds": self.loader.get_thresholds(),
                "raw_config": config
            }
        except Exception as e:
            self.logger.error(f"加载告警配置失败: {e}")
            raise
    
    def validate_configuration(self) -> ValidationResult:
        """验证配置"""
        try:
            config = self.loader.load_config()
            return self.loader.validate_config(config)
        except Exception as e:
            result = ValidationResult(is_valid=False)
            result.add_error(f"配置验证异常: {e}")
            return result
    
    def reload_configuration(self) -> Dict[str, Any]:
        """重新加载配置"""
        try:
            config = self.loader.load_config(force_reload=True)
            
            # 通知观察者
            for watcher in self._watchers:
                try:
                    watcher(config)
                except Exception as e:
                    self.logger.error(f"配置变更通知失败: {e}")
            
            return config
        except Exception as e:
            self.logger.error(f"重新加载配置失败: {e}")
            raise
    
    def add_config_watcher(self, callback: callable):
        """添加配置变更观察者"""
        self._watchers.append(callback)
    
    def remove_config_watcher(self, callback: callable):
        """移除配置变更观察者"""
        if callback in self._watchers:
            self._watchers.remove(callback)
    
    def export_config(self, output_path: str, format: str = "yaml") -> bool:
        """导出配置"""
        try:
            config = self.loader.load_config()
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == "yaml":
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            elif format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"配置已导出到: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出配置失败: {e}")
            return False
    
    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        try:
            configs = self.load_all_configs()
            
            return {
                "alert_rules_count": len(configs["alert_rules"]),
                "notification_rules_count": len(configs["notifications"].get("rules", [])),
                "suppression_rules_count": len(configs["suppression_rules"]),
                "escalation_policies_count": len(configs["escalation_policies"]),
                "enabled_channels": [
                    name for name, config in configs["notifications"].get("channels", {}).items()
                    if config.get("enabled", False)
                ],
                "global_enabled": configs["global"].get("enabled", False),
                "last_loaded": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"获取配置摘要失败: {e}")
            return {}