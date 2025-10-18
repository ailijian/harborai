#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警系统配置验证器

验证告警系统配置的正确性和完整性
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .alert_manager import AlertSeverity, AlertCondition
from .suppression_manager import SuppressionType
from .notification_service import NotificationPriority

# 默认配置常量
DEFAULT_ALERT_RULES = []
DEFAULT_NOTIFICATION_CONFIG = {"channels": []}
DEFAULT_SUPPRESSION_RULES = []
ESCALATION_CONFIG = {}
AGGREGATION_CONFIG = {}
METRICS_CONFIG = {}
HEALTH_CHECK_CONFIG = {}


class ValidationLevel(Enum):
    """验证级别"""
    ERROR = "error"      # 错误，必须修复
    WARNING = "warning"  # 警告，建议修复
    INFO = "info"       # 信息，可选修复


@dataclass
class ValidationResult:
    """验证结果"""
    level: ValidationLevel
    category: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "level": self.level.value,
            "category": self.category,
            "message": self.message,
            "field": self.field,
            "suggestion": self.suggestion
        }


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        
    def validate_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """验证完整配置"""
        self.results.clear()
        
        # 验证顶级结构
        self._validate_top_level_structure(config)
        
        # 验证告警规则
        if "alert_rules" in config:
            self._validate_alert_rules(config["alert_rules"])
            
        # 验证通知配置
        if "notification" in config:
            self._validate_notification_config(config["notification"])
            
        # 验证抑制配置
        if "suppression" in config:
            self._validate_suppression_config(config["suppression"])
            
        # 验证升级配置
        if "escalation" in config:
            self._validate_escalation_config(config["escalation"])
            
        # 验证聚合配置
        if "aggregation" in config:
            self._validate_aggregation_config(config["aggregation"])
            
        # 验证指标配置
        if "metrics" in config:
            self._validate_metrics_config(config["metrics"])
            
        # 验证健康检查配置
        if "health_check" in config:
            self._validate_health_check_config(config["health_check"])
            
        # 验证配置间的一致性
        self._validate_cross_references(config)
        
        return self.results.copy()
        
    def _validate_top_level_structure(self, config: Dict[str, Any]):
        """验证顶级结构"""
        required_sections = ["alert_rules", "notification"]
        optional_sections = ["suppression", "escalation", "aggregation", "metrics", "health_check"]
        
        # 检查必需的部分
        for section in required_sections:
            if section not in config:
                self._add_error(
                    "structure",
                    f"缺少必需的配置部分: {section}",
                    field=section,
                    suggestion=f"添加 {section} 配置部分"
                )
                
        # 检查未知的部分
        known_sections = set(required_sections + optional_sections)
        for section in config:
            if section not in known_sections:
                self._add_warning(
                    "structure",
                    f"未知的配置部分: {section}",
                    field=section,
                    suggestion="检查配置部分名称是否正确"
                )
                
    def _validate_alert_rules(self, rules: List[Dict[str, Any]]):
        """验证告警规则"""
        if not isinstance(rules, list):
            self._add_error(
                "alert_rules",
                "alert_rules 必须是列表",
                field="alert_rules"
            )
            return
            
        if len(rules) == 0:
            self._add_warning(
                "alert_rules",
                "没有定义任何告警规则",
                field="alert_rules",
                suggestion="添加至少一个告警规则"
            )
            return
            
        rule_ids = set()
        rule_names = set()
        
        for i, rule in enumerate(rules):
            self._validate_single_alert_rule(rule, i, rule_ids, rule_names)
            
    def _validate_single_alert_rule(
        self, 
        rule: Dict[str, Any], 
        index: int, 
        rule_ids: set, 
        rule_names: set
    ):
        """验证单个告警规则"""
        field_prefix = f"alert_rules[{index}]"
        
        # 验证必需字段
        required_fields = ["id", "name", "description", "severity", "condition", "metric", "threshold"]
        for field in required_fields:
            if field not in rule:
                self._add_error(
                    "alert_rules",
                    f"告警规则缺少必需字段: {field}",
                    field=f"{field_prefix}.{field}"
                )
                
        # 验证ID唯一性
        if "id" in rule:
            rule_id = rule["id"]
            if not isinstance(rule_id, str) or not rule_id.strip():
                self._add_error(
                    "alert_rules",
                    "告警规则ID必须是非空字符串",
                    field=f"{field_prefix}.id"
                )
            elif rule_id in rule_ids:
                self._add_error(
                    "alert_rules",
                    f"重复的告警规则ID: {rule_id}",
                    field=f"{field_prefix}.id"
                )
            else:
                rule_ids.add(rule_id)
                
            # 验证ID格式
            if not re.match(r'^[a-zA-Z0-9_-]+$', rule_id):
                self._add_warning(
                    "alert_rules",
                    f"告警规则ID包含特殊字符: {rule_id}",
                    field=f"{field_prefix}.id",
                    suggestion="使用字母、数字、下划线和连字符"
                )
                
        # 验证名称唯一性
        if "name" in rule:
            rule_name = rule["name"]
            if not isinstance(rule_name, str) or not rule_name.strip():
                self._add_error(
                    "alert_rules",
                    "告警规则名称必须是非空字符串",
                    field=f"{field_prefix}.name"
                )
            elif rule_name in rule_names:
                self._add_warning(
                    "alert_rules",
                    f"重复的告警规则名称: {rule_name}",
                    field=f"{field_prefix}.name",
                    suggestion="使用唯一的规则名称"
                )
            else:
                rule_names.add(rule_name)
                
        # 验证严重级别
        if "severity" in rule:
            try:
                AlertSeverity(rule["severity"])
            except ValueError:
                self._add_error(
                    "alert_rules",
                    f"无效的严重级别: {rule['severity']}",
                    field=f"{field_prefix}.severity",
                    suggestion=f"使用有效值: {[s.value for s in AlertSeverity]}"
                )
                
        # 验证条件类型
        if "condition" in rule:
            try:
                AlertCondition(rule["condition"])
            except ValueError:
                self._add_error(
                    "alert_rules",
                    f"无效的条件类型: {rule['condition']}",
                    field=f"{field_prefix}.condition",
                    suggestion=f"使用有效值: {[c.value for c in AlertCondition]}"
                )
                
        # 验证指标名称
        if "metric" in rule:
            metric = rule["metric"]
            if not isinstance(metric, str) or not metric.strip():
                self._add_error(
                    "alert_rules",
                    "指标名称必须是非空字符串",
                    field=f"{field_prefix}.metric"
                )
                
        # 验证阈值
        if "threshold" in rule:
            threshold = rule["threshold"]
            if not isinstance(threshold, (int, float)):
                self._add_error(
                    "alert_rules",
                    "阈值必须是数字",
                    field=f"{field_prefix}.threshold"
                )
                
        # 验证持续时间
        if "duration" in rule:
            duration = rule["duration"]
            if not isinstance(duration, (int, float)) or duration <= 0:
                self._add_error(
                    "alert_rules",
                    "持续时间必须是正数",
                    field=f"{field_prefix}.duration"
                )
            elif duration < 1:
                self._add_warning(
                    "alert_rules",
                    f"持续时间过短: {duration}秒",
                    field=f"{field_prefix}.duration",
                    suggestion="建议至少1秒以避免误报"
                )
                
        # 验证标签
        if "labels" in rule:
            labels = rule["labels"]
            if not isinstance(labels, dict):
                self._add_error(
                    "alert_rules",
                    "标签必须是字典",
                    field=f"{field_prefix}.labels"
                )
            else:
                for key, value in labels.items():
                    if not isinstance(key, str) or not isinstance(value, str):
                        self._add_error(
                            "alert_rules",
                            "标签键值必须是字符串",
                            field=f"{field_prefix}.labels.{key}"
                        )
                        
        # 验证注解
        if "annotations" in rule:
            annotations = rule["annotations"]
            if not isinstance(annotations, dict):
                self._add_error(
                    "alert_rules",
                    "注解必须是字典",
                    field=f"{field_prefix}.annotations"
                )
                
    def _validate_notification_config(self, notification: Dict[str, Any]):
        """验证通知配置"""
        # 验证渠道配置
        if "channels" in notification:
            self._validate_notification_channels(notification["channels"])
            
        # 验证路由配置
        if "routing" in notification:
            self._validate_notification_routing(notification["routing"])
            
        # 验证限流配置
        if "rate_limit" in notification:
            self._validate_rate_limit_config(notification["rate_limit"])
            
    def _validate_notification_channels(self, channels: List[Dict[str, Any]]):
        """验证通知渠道"""
        if not isinstance(channels, list):
            self._add_error(
                "notification",
                "通知渠道必须是列表",
                field="notification.channels"
            )
            return
            
        if len(channels) == 0:
            self._add_warning(
                "notification",
                "没有配置任何通知渠道",
                field="notification.channels",
                suggestion="至少配置一个通知渠道"
            )
            return
            
        channel_names = set()
        supported_types = ["console", "email", "webhook", "slack", "dingtalk", "wechat", "sms"]
        
        for i, channel in enumerate(channels):
            field_prefix = f"notification.channels[{i}]"
            
            # 验证必需字段
            required_fields = ["name", "type"]
            for field in required_fields:
                if field not in channel:
                    self._add_error(
                        "notification",
                        f"通知渠道缺少必需字段: {field}",
                        field=f"{field_prefix}.{field}"
                    )
                    
            # 验证名称唯一性
            if "name" in channel:
                name = channel["name"]
                if name in channel_names:
                    self._add_error(
                        "notification",
                        f"重复的通知渠道名称: {name}",
                        field=f"{field_prefix}.name"
                    )
                else:
                    channel_names.add(name)
                    
            # 验证类型
            if "type" in channel:
                channel_type = channel["type"]
                if channel_type not in supported_types:
                    self._add_error(
                        "notification",
                        f"不支持的通知渠道类型: {channel_type}",
                        field=f"{field_prefix}.type",
                        suggestion=f"使用支持的类型: {supported_types}"
                    )
                    
            # 验证特定类型的配置
            self._validate_channel_specific_config(channel, field_prefix)
            
    def _validate_channel_specific_config(self, channel: Dict[str, Any], field_prefix: str):
        """验证特定渠道类型的配置"""
        channel_type = channel.get("type")
        
        if channel_type == "email":
            required_fields = ["smtp_server", "smtp_port", "username", "password"]
            for field in required_fields:
                if field not in channel.get("config", {}):
                    self._add_error(
                        "notification",
                        f"邮件渠道缺少必需配置: {field}",
                        field=f"{field_prefix}.config.{field}"
                    )
                    
        elif channel_type == "webhook":
            if "url" not in channel.get("config", {}):
                self._add_error(
                    "notification",
                    "Webhook渠道缺少URL配置",
                    field=f"{field_prefix}.config.url"
                )
            else:
                url = channel["config"]["url"]
                if not url.startswith(("http://", "https://")):
                    self._add_error(
                        "notification",
                        "Webhook URL必须以http://或https://开头",
                        field=f"{field_prefix}.config.url"
                    )
                    
        elif channel_type == "slack":
            required_fields = ["webhook_url"]
            for field in required_fields:
                if field not in channel.get("config", {}):
                    self._add_error(
                        "notification",
                        f"Slack渠道缺少必需配置: {field}",
                        field=f"{field_prefix}.config.{field}"
                    )
                    
    def _validate_notification_routing(self, routing: Dict[str, Any]):
        """验证通知路由"""
        if "rules" in routing:
            rules = routing["rules"]
            if not isinstance(rules, list):
                self._add_error(
                    "notification",
                    "路由规则必须是列表",
                    field="notification.routing.rules"
                )
                return
                
            for i, rule in enumerate(rules):
                field_prefix = f"notification.routing.rules[{i}]"
                
                # 验证匹配条件
                if "match" not in rule:
                    self._add_error(
                        "notification",
                        "路由规则缺少匹配条件",
                        field=f"{field_prefix}.match"
                    )
                    
                # 验证目标渠道
                if "channels" not in rule:
                    self._add_error(
                        "notification",
                        "路由规则缺少目标渠道",
                        field=f"{field_prefix}.channels"
                    )
                elif not isinstance(rule["channels"], list):
                    self._add_error(
                        "notification",
                        "目标渠道必须是列表",
                        field=f"{field_prefix}.channels"
                    )
                    
    def _validate_rate_limit_config(self, rate_limit: Dict[str, Any]):
        """验证限流配置"""
        if "enabled" in rate_limit and not isinstance(rate_limit["enabled"], bool):
            self._add_error(
                "notification",
                "限流启用标志必须是布尔值",
                field="notification.rate_limit.enabled"
            )
            
        if "max_notifications_per_minute" in rate_limit:
            max_rate = rate_limit["max_notifications_per_minute"]
            if not isinstance(max_rate, int) or max_rate <= 0:
                self._add_error(
                    "notification",
                    "每分钟最大通知数必须是正整数",
                    field="notification.rate_limit.max_notifications_per_minute"
                )
                
    def _validate_suppression_config(self, suppression: Dict[str, Any]):
        """验证抑制配置"""
        if "rules" in suppression:
            rules = suppression["rules"]
            if not isinstance(rules, list):
                self._add_error(
                    "suppression",
                    "抑制规则必须是列表",
                    field="suppression.rules"
                )
                return
                
            for i, rule in enumerate(rules):
                self._validate_suppression_rule(rule, i)
                
    def _validate_suppression_rule(self, rule: Dict[str, Any], index: int):
        """验证抑制规则"""
        field_prefix = f"suppression.rules[{index}]"
        
        # 验证必需字段
        required_fields = ["id", "name", "type"]
        for field in required_fields:
            if field not in rule:
                self._add_error(
                    "suppression",
                    f"抑制规则缺少必需字段: {field}",
                    field=f"{field_prefix}.{field}"
                )
                
        # 验证类型
        if "type" in rule:
            try:
                rule_type = SuppressionType(rule["type"])
                # 验证特定类型的配置
                self._validate_suppression_rule_type_specific(rule, rule_type, field_prefix)
            except ValueError:
                self._add_error(
                    "suppression",
                    f"无效的抑制类型: {rule['type']}",
                    field=f"{field_prefix}.type",
                    suggestion=f"使用有效值: {[t.value for t in SuppressionType]}"
                )
                
    def _validate_suppression_rule_type_specific(self, rule: Dict[str, Any], rule_type: SuppressionType, field_prefix: str):
        """验证特定类型抑制规则的配置"""
        if rule_type == SuppressionType.TIME_BASED:
            if "time_config" not in rule:
                self._add_error(
                    "suppression",
                    "时间抑制规则缺少时间配置",
                    field=f"{field_prefix}.time_config"
                )
            else:
                time_config = rule["time_config"]
                if not isinstance(time_config, dict):
                    self._add_error(
                        "suppression",
                        "时间配置必须是对象",
                        field=f"{field_prefix}.time_config"
                    )
                else:
                    # 验证时间配置的必需字段
                    if "start_time" not in time_config or "end_time" not in time_config:
                        self._add_error(
                            "suppression",
                            "时间配置缺少开始时间或结束时间",
                            field=f"{field_prefix}.time_config"
                        )
                        
        elif rule_type == SuppressionType.DEPENDENCY:
            if "dependency_config" not in rule:
                self._add_error(
                    "suppression",
                    "依赖抑制规则缺少依赖配置",
                    field=f"{field_prefix}.dependency_config"
                )
            else:
                dep_config = rule["dependency_config"]
                if "depends_on" not in dep_config:
                    self._add_error(
                        "suppression",
                        "依赖配置缺少依赖目标",
                        field=f"{field_prefix}.dependency_config.depends_on"
                    )
                    
        elif rule_type == SuppressionType.LABEL_BASED:
            if "label_config" not in rule:
                self._add_error(
                    "suppression",
                    "标签抑制规则缺少标签配置",
                    field=f"{field_prefix}.label_config"
                )
                
        elif rule_type == SuppressionType.PATTERN_BASED:
            if "pattern_config" not in rule:
                self._add_error(
                    "suppression",
                    "模式抑制规则缺少模式配置",
                    field=f"{field_prefix}.pattern_config"
                )
                
        elif rule_type == SuppressionType.RATE_LIMIT:
            if "rate_limit_config" not in rule:
                self._add_error(
                    "suppression",
                    "频率限制抑制规则缺少频率配置",
                    field=f"{field_prefix}.rate_limit_config"
                )
                
    def _validate_escalation_config(self, escalation: Dict[str, Any]):
        """验证升级配置"""
        if "policies" in escalation:
            policies = escalation["policies"]
            if not isinstance(policies, list):
                self._add_error(
                    "escalation",
                    "升级策略必须是列表",
                    field="escalation.policies"
                )
                return
                
            for i, policy in enumerate(policies):
                self._validate_escalation_policy(policy, i)
                
    def _validate_escalation_policy(self, policy: Dict[str, Any], index: int):
        """验证升级策略"""
        field_prefix = f"escalation.policies[{index}]"
        
        # 验证必需字段
        required_fields = ["name", "levels"]
        for field in required_fields:
            if field not in policy:
                self._add_error(
                    "escalation",
                    f"升级策略缺少必需字段: {field}",
                    field=f"{field_prefix}.{field}"
                )
                
        # 验证升级级别
        if "levels" in policy:
            levels = policy["levels"]
            if not isinstance(levels, list):
                self._add_error(
                    "escalation",
                    "升级级别必须是列表",
                    field=f"{field_prefix}.levels"
                )
            elif len(levels) == 0:
                self._add_warning(
                    "escalation",
                    "升级策略没有定义任何级别",
                    field=f"{field_prefix}.levels"
                )
                
    def _validate_aggregation_config(self, aggregation: Dict[str, Any]):
        """验证聚合配置"""
        if "rules" in aggregation:
            rules = aggregation["rules"]
            if not isinstance(rules, list):
                self._add_error(
                    "aggregation",
                    "聚合规则必须是列表",
                    field="aggregation.rules"
                )
                
    def _validate_metrics_config(self, metrics: Dict[str, Any]):
        """验证指标配置"""
        if "collection_interval" in metrics:
            interval = metrics["collection_interval"]
            if not isinstance(interval, (int, float)) or interval <= 0:
                self._add_error(
                    "metrics",
                    "指标收集间隔必须是正数",
                    field="metrics.collection_interval"
                )
                
    def _validate_health_check_config(self, health_check: Dict[str, Any]):
        """验证健康检查配置"""
        if "enabled" in health_check and not isinstance(health_check["enabled"], bool):
            self._add_error(
                "health_check",
                "健康检查启用标志必须是布尔值",
                field="health_check.enabled"
            )
            
        if "interval" in health_check:
            interval = health_check["interval"]
            if not isinstance(interval, (int, float)) or interval <= 0:
                self._add_error(
                    "health_check",
                    "健康检查间隔必须是正数",
                    field="health_check.interval"
                )
                
    def _validate_cross_references(self, config: Dict[str, Any]):
        """验证配置间的交叉引用"""
        # 收集所有渠道名称
        channel_names = set()
        if "notification" in config and "channels" in config["notification"]:
            for channel in config["notification"]["channels"]:
                if "name" in channel:
                    channel_names.add(channel["name"])
                    
        # 验证路由规则中的渠道引用
        if ("notification" in config and 
            "routing" in config["notification"] and 
            "rules" in config["notification"]["routing"]):
            
            for i, rule in enumerate(config["notification"]["routing"]["rules"]):
                if "channels" in rule:
                    for j, channel_name in enumerate(rule["channels"]):
                        if channel_name not in channel_names:
                            self._add_error(
                                "notification",
                                f"路由规则引用了不存在的渠道: {channel_name}",
                                field=f"notification.routing.rules[{i}].channels[{j}]",
                                suggestion="确保渠道已在channels部分定义"
                            )
                            
        # 验证升级配置中的渠道引用
        if "escalation" in config and "rules" in config["escalation"]:
            for i, rule in enumerate(config["escalation"]["rules"]):
                # 支持两种格式：escalation_steps.channels 和 escalation_channels
                if "escalation_steps" in rule:
                    for j, step in enumerate(rule["escalation_steps"]):
                        if "channels" in step:
                            for k, channel_name in enumerate(step["channels"]):
                                if channel_name not in channel_names:
                                    self._add_error(
                                        "escalation",
                                        f"升级配置引用了不存在的通知渠道: {channel_name}",
                                        field=f"escalation.rules[{i}].escalation_steps[{j}].channels[{k}]",
                                        suggestion="确保渠道已在notification.channels部分定义"
                                    )
                elif "escalation_channels" in rule:
                    for j, channel_name in enumerate(rule["escalation_channels"]):
                        if channel_name not in channel_names:
                            self._add_error(
                                "escalation",
                                f"升级配置引用了不存在的通知渠道: {channel_name}",
                                field=f"escalation.rules[{i}].escalation_channels[{j}]",
                                suggestion="确保渠道已在notification.channels部分定义"
                            )
                                    
        # 验证告警规则和抑制规则的依赖关系
        alert_rule_ids = set()
        if "alert_rules" in config:
            for rule in config["alert_rules"]:
                if "id" in rule:
                    alert_rule_ids.add(rule["id"])
                    
        if "suppression" in config and "rules" in config["suppression"]:
            for i, rule in enumerate(config["suppression"]["rules"]):
                if rule.get("type") == "dependency":
                    config_data = rule.get("config", {})
                    
                    # 验证父规则引用
                    if "parent_rule" in config_data:
                        parent_rule = config_data["parent_rule"]
                        if parent_rule not in alert_rule_ids:
                            self._add_error(
                                "suppression",
                                f"抑制规则引用了不存在的父告警规则: {parent_rule}",
                                field=f"suppression.rules[{i}].config.parent_rule",
                                suggestion="确保父规则已在alert_rules部分定义"
                            )
                            
                    # 验证子规则引用
                    if "child_rules" in config_data:
                        for j, child_rule in enumerate(config_data["child_rules"]):
                            if child_rule not in alert_rule_ids:
                                self._add_error(
                                    "suppression",
                                    f"抑制规则引用了不存在的子告警规则: {child_rule}",
                                    field=f"suppression.rules[{i}].config.child_rules[{j}]",
                                    suggestion="确保子规则已在alert_rules部分定义"
                                )
                                
        # 验证配置完整性
        self._validate_config_completeness(config)
        
        # 验证配置冲突
        self._validate_config_conflicts(config)
        
    def _validate_config_completeness(self, config: Dict[str, Any]):
        """验证配置完整性"""
        # 检查是否有启用的通知渠道
        enabled_channels = 0
        if "notification" in config and "channels" in config["notification"]:
            for channel in config["notification"]["channels"]:
                if channel.get("enabled", True):
                    enabled_channels += 1
                    
        if enabled_channels == 0:
            self._add_warning(
                "completeness",
                "没有启用的通知渠道，告警将无法发送",
                suggestion="至少启用一个通知渠道"
            )
            
        # 检查是否有严重级别的告警规则
        has_critical_rules = False
        if "alert_rules" in config:
            for rule in config["alert_rules"]:
                if rule.get("severity") == AlertSeverity.CRITICAL.value:
                    has_critical_rules = True
                    break
                    
        if not has_critical_rules:
            self._add_info(
                "completeness",
                "没有定义严重级别的告警规则",
                suggestion="考虑添加关键系统的严重告警规则"
            )
            
        # 检查是否配置了升级策略
        if "escalation" not in config or not config["escalation"].get("enabled", False):
            self._add_info(
                "completeness",
                "没有启用告警升级策略",
                suggestion="考虑为重要告警配置升级策略"
            )
            
    def _validate_config_conflicts(self, config: Dict[str, Any]):
        """验证配置冲突"""
        # 检查告警规则的阈值冲突
        if "alert_rules" in config:
            metric_rules = {}
            for rule in config["alert_rules"]:
                metric = rule.get("metric")
                severity = rule.get("severity")
                threshold = rule.get("threshold")
                
                if metric and severity and threshold is not None:
                    if metric not in metric_rules:
                        metric_rules[metric] = []
                    metric_rules[metric].append({
                        "id": rule.get("id"),
                        "severity": severity,
                        "threshold": threshold,
                        "condition": rule.get("condition")
                    })
                    
            # 检查同一指标的不同严重级别阈值是否合理
            for metric, rules in metric_rules.items():
                if len(rules) > 1:
                    self._validate_threshold_consistency(metric, rules)
                    
        # 检查抑制规则的时间冲突
        if "suppression" in config and "rules" in config["suppression"]:
            time_based_rules = []
            for rule in config["suppression"]["rules"]:
                if rule.get("type") == "time_based":
                    time_based_rules.append(rule)
                    
            self._validate_time_based_suppression_conflicts(time_based_rules)
            
    def _validate_threshold_consistency(self, metric: str, rules: List[Dict[str, Any]]):
        """验证阈值一致性"""
        severity_order = {
            AlertSeverity.LOW.value: 1,
            AlertSeverity.MEDIUM.value: 2,
            AlertSeverity.HIGH.value: 3,
            AlertSeverity.CRITICAL.value: 4
        }
        
        # 按严重级别排序
        sorted_rules = sorted(rules, key=lambda r: severity_order.get(r["severity"], 0))
        
        for i in range(len(sorted_rules) - 1):
            current = sorted_rules[i]
            next_rule = sorted_rules[i + 1]
            
            # 检查阈值是否递增（对于大于类型的条件）
            if (current["condition"] == "threshold" and 
                next_rule["condition"] == "threshold"):
                
                if current["threshold"] >= next_rule["threshold"]:
                    self._add_warning(
                        "conflicts",
                        f"阈值配置可能存在冲突: 指标 {metric} 的 "
                        f"{current['severity']}级别({current['threshold']}) >= "
                        f"{next_rule['severity']}级别({next_rule['threshold']})",
                        suggestion="确保更高严重级别的阈值更严格"
                    )
                    
    def _validate_time_based_suppression_conflicts(self, rules: List[Dict[str, Any]]):
        """验证基于时间的抑制规则冲突"""
        for i, rule1 in enumerate(rules):
            for j, rule2 in enumerate(rules[i + 1:], i + 1):
                if self._check_time_overlap(rule1, rule2):
                    self._add_warning(
                        "conflicts",
                        f"时间抑制规则存在重叠: {rule1.get('name')} 和 {rule2.get('name')}",
                        suggestion="检查时间窗口设置，避免不必要的重叠"
                    )
                    
    def _check_time_overlap(self, rule1: Dict[str, Any], rule2: Dict[str, Any]) -> bool:
        """检查两个时间规则是否重叠"""
        # 支持两种格式：config.start_time/end_time 和 直接的 start_time/end_time
        config1 = rule1.get("config", {})
        config2 = rule2.get("config", {})
        
        start1 = config1.get("start_time") or rule1.get("start_time")
        end1 = config1.get("end_time") or rule1.get("end_time")
        start2 = config2.get("start_time") or rule2.get("start_time")
        end2 = config2.get("end_time") or rule2.get("end_time")
        
        if not all([start1, end1, start2, end2]):
            return False
            
        # 检查天数重叠（如果指定了天数）
        days1 = set(config1.get("days", []) or rule1.get("days", []))
        days2 = set(config2.get("days", []) or rule2.get("days", []))
        
        # 如果都指定了天数且没有交集，则不重叠
        if days1 and days2 and not days1.intersection(days2):
            return False
            
        # 简化的时间重叠检查
        try:
            from datetime import time
            
            # 处理 HH:MM 格式的时间
            def parse_time(time_str):
                if ':' in time_str:
                    hour, minute = map(int, time_str.split(':'))
                    return time(hour, minute)
                return time.fromisoformat(time_str)
            
            t1_start = parse_time(start1)
            t1_end = parse_time(end1)
            t2_start = parse_time(start2)
            t2_end = parse_time(end2)
            
            # 检查时间段是否重叠
            return not (t1_end <= t2_start or t2_end <= t1_start)
        except:
            return False
                            
    def _add_error(self, category: str, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """添加错误"""
        self.results.append(ValidationResult(
            level=ValidationLevel.ERROR,
            category=category,
            message=message,
            field=field,
            suggestion=suggestion
        ))
        
    def _add_warning(self, category: str, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """添加警告"""
        self.results.append(ValidationResult(
            level=ValidationLevel.WARNING,
            category=category,
            message=message,
            field=field,
            suggestion=suggestion
        ))
        
    def _add_info(self, category: str, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """添加信息"""
        self.results.append(ValidationResult(
            level=ValidationLevel.INFO,
            category=category,
            message=message,
            field=field,
            suggestion=suggestion
        ))
        
    def get_summary(self) -> Dict[str, Any]:
        """获取验证摘要"""
        error_count = len([r for r in self.results if r.level == ValidationLevel.ERROR])
        warning_count = len([r for r in self.results if r.level == ValidationLevel.WARNING])
        info_count = len([r for r in self.results if r.level == ValidationLevel.INFO])
        
        return {
            "total_results": len(self.results),
            "errors": error_count,
            "warnings": warning_count,
            "info": info_count,
            "is_valid": error_count == 0
        }
        
    def format_results(self, format_type: str = "text") -> str:
        """格式化验证结果"""
        if format_type == "json":
            return json.dumps({
                "summary": self.get_summary(),
                "results": [r.to_dict() for r in self.results]
            }, indent=2, ensure_ascii=False)
            
        elif format_type == "text":
            lines = []
            summary = self.get_summary()
            
            lines.append("=== 告警系统配置验证结果 ===")
            lines.append(f"总问题数: {summary['total_results']}")
            lines.append(f"错误: {summary['errors']}")
            lines.append(f"警告: {summary['warnings']}")
            lines.append(f"信息: {summary['info']}")
            lines.append(f"配置有效: {'是' if summary['is_valid'] else '否'}")
            lines.append("")
            
            if self.results:
                # 按级别分组
                by_level = {}
                for result in self.results:
                    level = result.level.value
                    if level not in by_level:
                        by_level[level] = []
                    by_level[level].append(result)
                    
                # 输出每个级别的问题
                for level in ["error", "warning", "info"]:
                    if level in by_level:
                        level_name = {"error": "错误", "warning": "警告", "info": "信息"}[level]
                        lines.append(f"=== {level_name} ===")
                        
                        for result in by_level[level]:
                            lines.append(f"[{result.category}] {result.message}")
                            if result.field:
                                lines.append(f"  字段: {result.field}")
                            if result.suggestion:
                                lines.append(f"  建议: {result.suggestion}")
                            lines.append("")
                            
            return "\n".join(lines)
            
        else:
            raise ValueError(f"不支持的格式类型: {format_type}")


def validate_config_file(config_path: str) -> Tuple[bool, List[ValidationResult]]:
    """验证配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        validator = ConfigValidator()
        results = validator.validate_config(config)
        
        # 检查是否有错误
        has_errors = any(r.level == ValidationLevel.ERROR for r in results)
        
        return not has_errors, results
        
    except FileNotFoundError:
        return False, [ValidationResult(
            level=ValidationLevel.ERROR,
            category="file",
            message=f"配置文件不存在: {config_path}"
        )]
    except json.JSONDecodeError as e:
        return False, [ValidationResult(
            level=ValidationLevel.ERROR,
            category="file",
            message=f"配置文件JSON格式错误: {e}"
        )]
    except Exception as e:
        return False, [ValidationResult(
            level=ValidationLevel.ERROR,
            category="file",
            message=f"读取配置文件时发生错误: {e}"
        )]


def validate_default_config() -> Tuple[bool, List[ValidationResult]]:
    """验证默认配置"""
    from .config import (
        DEFAULT_ALERT_RULES, 
        DEFAULT_NOTIFICATION_CONFIG, 
        DEFAULT_SUPPRESSION_RULES,
        ESCALATION_CONFIG,
        AGGREGATION_CONFIG,
        METRICS_CONFIG,
        HEALTH_CHECK_CONFIG
    )
    
    # 构建完整的配置字典
    config = {
        "alert_rules": DEFAULT_ALERT_RULES,
        "notification": DEFAULT_NOTIFICATION_CONFIG,
        "suppression": {"rules": DEFAULT_SUPPRESSION_RULES},
        "escalation": ESCALATION_CONFIG,
        "aggregation": AGGREGATION_CONFIG,
        "metrics": METRICS_CONFIG,
        "health_check": HEALTH_CHECK_CONFIG
    }
    
    validator = ConfigValidator()
    results = validator.validate_config(config)
    
    # 检查是否有错误
    has_errors = any(r.level == ValidationLevel.ERROR for r in results)
    
    return not has_errors, results


def create_config_schema() -> Dict[str, Any]:
    """创建配置模式定义"""
    return {
        "type": "object",
        "properties": {
            "alert_rules": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "name", "description", "severity", "condition", "metric", "threshold"],
                    "properties": {
                        "id": {"type": "string", "pattern": "^[a-zA-Z0-9_-]+$"},
                        "name": {"type": "string", "minLength": 1},
                        "description": {"type": "string"},
                        "severity": {"enum": ["low", "medium", "high", "critical"]},
                        "condition": {"enum": ["threshold", "anomaly", "change"]},
                        "metric": {"type": "string", "minLength": 1},
                        "threshold": {"type": "number"},
                        "duration": {"type": "number", "minimum": 0},
                        "labels": {"type": "object"},
                        "annotations": {"type": "object"}
                    }
                }
            },
            "notification": {
                "type": "object",
                "required": ["channels"],
                "properties": {
                    "channels": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["name", "type"],
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "type": {"enum": ["console", "email", "webhook", "slack", "dingtalk", "wechat", "sms"]},
                                "enabled": {"type": "boolean"},
                                "config": {"type": "object"}
                            }
                        }
                    },
                    "routing": {"type": "object"},
                    "rate_limits": {"type": "object"},
                    "retry": {"type": "object"}
                }
            },
            "suppression": {
                "type": "object",
                "properties": {
                    "rules": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "name", "type"],
                            "properties": {
                                "id": {"type": "string"},
                                "name": {"type": "string"},
                                "type": {"enum": ["time_based", "label_based", "pattern_based", "dependency", "maintenance", "rate_limit", "duplicate", "smart"]},
                                "enabled": {"type": "boolean"},
                                "config": {"type": "object"}
                            }
                        }
                    }
                }
            },
            "escalation": {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                    "global_settings": {"type": "object"},
                    "rules": {"type": "array"},
                    "notification_templates": {"type": "object"},
                    "escalation_policies": {"type": "object"}
                }
            },
            "aggregation": {"type": "object"},
            "metrics": {"type": "object"},
            "health_check": {"type": "object"}
        },
        "required": ["alert_rules", "notification"]
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("用法: python config_validator.py <config_file>")
        sys.exit(1)
        
    config_file = sys.argv[1]
    is_valid, results = validate_config_file(config_file)
    
    validator = ConfigValidator()
    validator.results = results
    
    print(validator.format_results("text"))
    
    if not is_valid:
        sys.exit(1)