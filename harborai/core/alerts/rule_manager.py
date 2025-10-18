"""
告警规则管理器

负责管理告警规则的加载、验证、更新和持久化。
支持动态规则更新、规则模板、规则继承等高级功能。
"""

import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum

from .alert_manager import AlertRule, AlertSeverity, AlertCondition
from .config_validator import ConfigValidator, ValidationResult


class RuleStatus(Enum):
    """规则状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    DEPRECATED = "deprecated"


class RuleSource(Enum):
    """规则来源"""
    FILE = "file"
    API = "api"
    TEMPLATE = "template"
    INHERITED = "inherited"


@dataclass
class RuleMetadata:
    """规则元数据"""
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    version: int
    source: RuleSource
    status: RuleStatus
    tags: List[str]
    dependencies: List[str]
    parent_rule: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_by': self.created_by,
            'updated_by': self.updated_by,
            'version': self.version,
            'source': self.source.value,
            'status': self.status.value,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'parent_rule': self.parent_rule
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RuleMetadata':
        """从字典创建"""
        return cls(
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            created_by=data['created_by'],
            updated_by=data['updated_by'],
            version=data['version'],
            source=RuleSource(data['source']),
            status=RuleStatus(data['status']),
            tags=data['tags'],
            dependencies=data['dependencies'],
            parent_rule=data.get('parent_rule')
        )


@dataclass
class ManagedAlertRule:
    """带元数据的告警规则"""
    rule: AlertRule
    metadata: RuleMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'rule': asdict(self.rule),
            'metadata': self.metadata.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ManagedAlertRule':
        """从字典创建"""
        rule_data = data['rule']
        rule = AlertRule(
            id=rule_data['id'],
            name=rule_data['name'],
            description=rule_data['description'],
            severity=AlertSeverity(rule_data['severity']),
            condition=AlertCondition(rule_data['condition']),
            metric=rule_data['metric'],
            threshold=rule_data['threshold'],
            duration=rule_data['duration'],
            labels=rule_data['labels'],
            annotations=rule_data['annotations'],
            enabled=rule_data.get('enabled', True)
        )
        metadata = RuleMetadata.from_dict(data['metadata'])
        return cls(rule=rule, metadata=metadata)


class RuleTemplate:
    """规则模板"""
    
    def __init__(self, template_id: str, name: str, description: str, 
                 template_data: Dict[str, Any]):
        self.template_id = template_id
        self.name = name
        self.description = description
        self.template_data = template_data
    
    def create_rule(self, rule_id: str, parameters: Dict[str, Any]) -> AlertRule:
        """基于模板创建规则"""
        rule_data = self.template_data.copy()
        
        # 替换模板参数
        for key, value in parameters.items():
            if isinstance(value, dict):
                rule_data[key].update(value)
            else:
                rule_data[key] = value
        
        # 设置规则ID
        rule_data['id'] = rule_id
        
        return AlertRule(
            id=rule_data['id'],
            name=rule_data['name'],
            description=rule_data['description'],
            severity=AlertSeverity(rule_data['severity']),
            condition=AlertCondition(rule_data['condition']),
            metric=rule_data['metric'],
            threshold=rule_data['threshold'],
            duration=rule_data['duration'],
            labels=rule_data.get('labels', {}),
            annotations=rule_data.get('annotations', {}),
            enabled=rule_data.get('enabled', True)
        )


class RuleManager:
    """告警规则管理器"""
    
    def __init__(self, config_dir: str = "config", 
                 rules_file: str = "alert_rules.json"):
        self.config_dir = Path(config_dir)
        self.rules_file = self.config_dir / rules_file
        self.logger = logging.getLogger(__name__)
        
        # 规则存储
        self.rules: Dict[str, ManagedAlertRule] = {}
        self.templates: Dict[str, RuleTemplate] = {}
        self.rule_dependencies: Dict[str, Set[str]] = {}
        
        # 配置验证器
        self.validator = ConfigValidator()
        
        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载默认模板
        self._load_default_templates()
    
    def _load_default_templates(self):
        """加载默认规则模板"""
        templates = {
            "threshold_template": {
                "name": "阈值告警模板",
                "description": "基于阈值的告警规则模板",
                "template_data": {
                    "name": "阈值告警",
                    "description": "当指标超过阈值时触发告警",
                    "severity": "medium",
                    "condition": "threshold",
                    "metric": "metric.name",
                    "threshold": 100,
                    "duration": 300,
                    "labels": {},
                    "annotations": {},
                    "enabled": True
                }
            },
            "anomaly_template": {
                "name": "异常检测模板",
                "description": "基于异常检测的告警规则模板",
                "template_data": {
                    "name": "异常检测告警",
                    "description": "当指标出现异常时触发告警",
                    "severity": "medium",
                    "condition": "anomaly",
                    "metric": "metric.name",
                    "threshold": 3.0,
                    "duration": 600,
                    "labels": {},
                    "annotations": {},
                    "enabled": True
                }
            },
            "rate_template": {
                "name": "变化率模板",
                "description": "基于变化率的告警规则模板",
                "template_data": {
                    "name": "变化率告警",
                    "description": "当指标变化率超过阈值时触发告警",
                    "severity": "medium",
                    "condition": "rate",
                    "metric": "metric.name",
                    "threshold": 0.1,
                    "duration": 300,
                    "labels": {},
                    "annotations": {},
                    "enabled": True
                }
            }
        }
        
        for template_id, template_config in templates.items():
            template = RuleTemplate(
                template_id=template_id,
                name=template_config["name"],
                description=template_config["description"],
                template_data=template_config["template_data"]
            )
            self.templates[template_id] = template
    
    async def load_rules(self, config_file: Optional[str] = None) -> ValidationResult:
        """加载告警规则"""
        if config_file:
            config_path = Path(config_file)
        else:
            config_path = self.rules_file
        
        if not config_path.exists():
            self.logger.warning(f"规则文件不存在: {config_path}")
            return ValidationResult()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 验证配置
            validation_result = self.validator.validate_config(config_data)
            if validation_result.has_errors():
                self.logger.error(f"规则配置验证失败: {validation_result.get_summary()}")
                return validation_result
            
            # 加载规则
            rules_data = config_data.get('alert_rules', [])
            for rule_data in rules_data:
                await self._load_rule_from_data(rule_data)
            
            # 构建依赖关系
            self._build_dependencies()
            
            self.logger.info(f"成功加载 {len(self.rules)} 个告警规则")
            return validation_result
            
        except Exception as e:
            self.logger.error(f"加载规则失败: {e}")
            validation_result = ValidationResult()
            validation_result.add_error("rule_loading", f"加载规则失败: {e}")
            return validation_result
    
    async def _load_rule_from_data(self, rule_data: Dict[str, Any]):
        """从数据加载单个规则"""
        try:
            # 创建规则
            rule = AlertRule(
                id=rule_data['id'],
                name=rule_data['name'],
                description=rule_data['description'],
                severity=AlertSeverity(rule_data['severity']),
                condition=AlertCondition(rule_data['condition']),
                metric=rule_data['metric'],
                threshold=rule_data['threshold'],
                duration=rule_data['duration'],
                labels=rule_data.get('labels', {}),
                annotations=rule_data.get('annotations', {}),
                enabled=rule_data.get('enabled', True)
            )
            
            # 创建元数据
            now = datetime.now()
            metadata = RuleMetadata(
                created_at=now,
                updated_at=now,
                created_by="system",
                updated_by="system",
                version=1,
                source=RuleSource.FILE,
                status=RuleStatus.ACTIVE,
                tags=rule_data.get('tags', []),
                dependencies=rule_data.get('dependencies', [])
            )
            
            # 创建管理规则
            managed_rule = ManagedAlertRule(rule=rule, metadata=metadata)
            self.rules[rule.id] = managed_rule
            
        except Exception as e:
            self.logger.error(f"加载规则失败 {rule_data.get('id', 'unknown')}: {e}")
    
    def _build_dependencies(self):
        """构建规则依赖关系"""
        self.rule_dependencies.clear()
        
        for rule_id, managed_rule in self.rules.items():
            dependencies = set(managed_rule.metadata.dependencies)
            self.rule_dependencies[rule_id] = dependencies
    
    async def save_rules(self, config_file: Optional[str] = None) -> bool:
        """保存告警规则"""
        if config_file:
            config_path = Path(config_file)
        else:
            config_path = self.rules_file
        
        try:
            # 构建配置数据
            config_data = {
                "alert_rules": [
                    {
                        **asdict(managed_rule.rule),
                        "tags": managed_rule.metadata.tags,
                        "dependencies": managed_rule.metadata.dependencies
                    }
                    for managed_rule in self.rules.values()
                    if managed_rule.metadata.status == RuleStatus.ACTIVE
                ]
            }
            
            # 保存到文件
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功保存 {len(config_data['alert_rules'])} 个告警规则到 {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存规则失败: {e}")
            return False
    
    async def add_rule(self, rule: AlertRule, metadata: Optional[RuleMetadata] = None,
                      created_by: str = "system") -> bool:
        """添加告警规则"""
        if rule.id in self.rules:
            self.logger.warning(f"规则已存在: {rule.id}")
            return False
        
        # 创建元数据
        if metadata is None:
            now = datetime.now()
            metadata = RuleMetadata(
                created_at=now,
                updated_at=now,
                created_by=created_by,
                updated_by=created_by,
                version=1,
                source=RuleSource.API,
                status=RuleStatus.ACTIVE,
                tags=[],
                dependencies=[]
            )
        
        # 创建管理规则
        managed_rule = ManagedAlertRule(rule=rule, metadata=metadata)
        self.rules[rule.id] = managed_rule
        
        # 更新依赖关系
        self.rule_dependencies[rule.id] = set(metadata.dependencies)
        
        self.logger.info(f"添加告警规则: {rule.id}")
        return True
    
    async def update_rule(self, rule_id: str, rule: AlertRule, 
                         updated_by: str = "system") -> bool:
        """更新告警规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"规则不存在: {rule_id}")
            return False
        
        managed_rule = self.rules[rule_id]
        
        # 更新规则
        managed_rule.rule = rule
        
        # 更新元数据
        managed_rule.metadata.updated_at = datetime.now()
        managed_rule.metadata.updated_by = updated_by
        managed_rule.metadata.version += 1
        
        self.logger.info(f"更新告警规则: {rule_id}")
        return True
    
    async def remove_rule(self, rule_id: str) -> bool:
        """删除告警规则"""
        if rule_id not in self.rules:
            self.logger.warning(f"规则不存在: {rule_id}")
            return False
        
        # 检查依赖关系
        dependent_rules = self._get_dependent_rules(rule_id)
        if dependent_rules:
            self.logger.warning(f"规则 {rule_id} 被其他规则依赖: {dependent_rules}")
            return False
        
        # 删除规则
        del self.rules[rule_id]
        if rule_id in self.rule_dependencies:
            del self.rule_dependencies[rule_id]
        
        self.logger.info(f"删除告警规则: {rule_id}")
        return True
    
    def _get_dependent_rules(self, rule_id: str) -> List[str]:
        """获取依赖指定规则的规则列表"""
        dependent_rules = []
        for rid, dependencies in self.rule_dependencies.items():
            if rule_id in dependencies:
                dependent_rules.append(rid)
        return dependent_rules
    
    async def create_rule_from_template(self, template_id: str, rule_id: str,
                                      parameters: Dict[str, Any],
                                      created_by: str = "system") -> Optional[AlertRule]:
        """基于模板创建规则"""
        if template_id not in self.templates:
            self.logger.error(f"模板不存在: {template_id}")
            return None
        
        if rule_id in self.rules:
            self.logger.error(f"规则已存在: {rule_id}")
            return None
        
        try:
            template = self.templates[template_id]
            rule = template.create_rule(rule_id, parameters)
            
            # 创建元数据
            now = datetime.now()
            metadata = RuleMetadata(
                created_at=now,
                updated_at=now,
                created_by=created_by,
                updated_by=created_by,
                version=1,
                source=RuleSource.TEMPLATE,
                status=RuleStatus.ACTIVE,
                tags=[f"template:{template_id}"],
                dependencies=[]
            )
            
            # 添加规则
            await self.add_rule(rule, metadata, created_by)
            
            self.logger.info(f"基于模板 {template_id} 创建规则: {rule_id}")
            return rule
            
        except Exception as e:
            self.logger.error(f"基于模板创建规则失败: {e}")
            return None
    
    def get_rule(self, rule_id: str) -> Optional[ManagedAlertRule]:
        """获取规则"""
        return self.rules.get(rule_id)
    
    def get_rules(self, status: Optional[RuleStatus] = None,
                 tags: Optional[List[str]] = None) -> List[ManagedAlertRule]:
        """获取规则列表"""
        rules = list(self.rules.values())
        
        # 按状态过滤
        if status:
            rules = [r for r in rules if r.metadata.status == status]
        
        # 按标签过滤
        if tags:
            rules = [r for r in rules if any(tag in r.metadata.tags for tag in tags)]
        
        return rules
    
    def get_active_rules(self) -> List[AlertRule]:
        """获取活跃规则列表"""
        return [
            managed_rule.rule
            for managed_rule in self.rules.values()
            if managed_rule.metadata.status == RuleStatus.ACTIVE
            and managed_rule.rule.enabled
        ]
    
    def get_templates(self) -> List[RuleTemplate]:
        """获取模板列表"""
        return list(self.templates.values())
    
    def get_template(self, template_id: str) -> Optional[RuleTemplate]:
        """获取模板"""
        return self.templates.get(template_id)
    
    async def validate_rule(self, rule: AlertRule) -> ValidationResult:
        """验证单个规则"""
        validation_result = ValidationResult()
        
        try:
            # 基本验证
            if not rule.id:
                validation_result.add_error("rule_id", "规则ID不能为空")
            
            if not rule.name:
                validation_result.add_error("rule_name", "规则名称不能为空")
            
            if not rule.metric:
                validation_result.add_error("rule_metric", "监控指标不能为空")
            
            if rule.threshold is None:
                validation_result.add_error("rule_threshold", "阈值不能为空")
            
            if rule.duration <= 0:
                validation_result.add_error("rule_duration", "持续时间必须大于0")
            
            # 检查重复ID
            if rule.id in self.rules:
                validation_result.add_error("rule_duplicate", f"规则ID重复: {rule.id}")
            
        except Exception as e:
            validation_result.add_error("rule_validation", f"规则验证失败: {e}")
        
        return validation_result
    
    async def get_rule_statistics(self) -> Dict[str, Any]:
        """获取规则统计信息"""
        total_rules = len(self.rules)
        active_rules = len([r for r in self.rules.values() 
                           if r.metadata.status == RuleStatus.ACTIVE])
        inactive_rules = len([r for r in self.rules.values() 
                             if r.metadata.status == RuleStatus.INACTIVE])
        draft_rules = len([r for r in self.rules.values() 
                          if r.metadata.status == RuleStatus.DRAFT])
        
        # 按严重程度统计
        severity_stats = {}
        for severity in AlertSeverity:
            count = len([r for r in self.rules.values() 
                        if r.rule.severity == severity])
            severity_stats[severity.value] = count
        
        # 按条件类型统计
        condition_stats = {}
        for condition in AlertCondition:
            count = len([r for r in self.rules.values() 
                        if r.rule.condition == condition])
            condition_stats[condition.value] = count
        
        # 按来源统计
        source_stats = {}
        for source in RuleSource:
            count = len([r for r in self.rules.values() 
                        if r.metadata.source == source])
            source_stats[source.value] = count
        
        return {
            "total_rules": total_rules,
            "active_rules": active_rules,
            "inactive_rules": inactive_rules,
            "draft_rules": draft_rules,
            "severity_distribution": severity_stats,
            "condition_distribution": condition_stats,
            "source_distribution": source_stats,
            "templates_count": len(self.templates),
            "dependencies_count": sum(len(deps) for deps in self.rule_dependencies.values())
        }
    
    async def export_rules(self, export_path: str, 
                          include_metadata: bool = True) -> bool:
        """导出规则"""
        try:
            export_data = {}
            
            if include_metadata:
                export_data = {
                    "rules": [managed_rule.to_dict() for managed_rule in self.rules.values()],
                    "templates": {
                        tid: {
                            "name": template.name,
                            "description": template.description,
                            "template_data": template.template_data
                        }
                        for tid, template in self.templates.items()
                    },
                    "exported_at": datetime.now().isoformat(),
                    "version": "1.0"
                }
            else:
                export_data = {
                    "alert_rules": [asdict(managed_rule.rule) 
                                   for managed_rule in self.rules.values()]
                }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"成功导出规则到: {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出规则失败: {e}")
            return False
    
    async def import_rules(self, import_path: str, 
                          overwrite: bool = False) -> ValidationResult:
        """导入规则"""
        validation_result = ValidationResult()
        
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            # 导入规则
            if "rules" in import_data:
                # 带元数据的导入
                for rule_data in import_data["rules"]:
                    managed_rule = ManagedAlertRule.from_dict(rule_data)
                    
                    if managed_rule.rule.id in self.rules and not overwrite:
                        validation_result.add_warning(
                            "rule_exists", 
                            f"规则已存在，跳过: {managed_rule.rule.id}"
                        )
                        continue
                    
                    self.rules[managed_rule.rule.id] = managed_rule
                    imported_count += 1
            
            elif "alert_rules" in import_data:
                # 仅规则数据的导入
                for rule_data in import_data["alert_rules"]:
                    await self._load_rule_from_data(rule_data)
                    imported_count += 1
            
            # 重建依赖关系
            self._build_dependencies()
            
            validation_result.add_info(
                "import_success", 
                f"成功导入 {imported_count} 个规则"
            )
            
            self.logger.info(f"成功导入 {imported_count} 个规则")
            
        except Exception as e:
            validation_result.add_error("import_failed", f"导入规则失败: {e}")
            self.logger.error(f"导入规则失败: {e}")
        
        return validation_result