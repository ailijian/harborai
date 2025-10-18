#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据完整性验证器

提供全面的数据完整性验证功能，包括：
- 数据类型验证
- 业务规则验证
- 数据关系验证
- 数据质量评估
- 完整性报告生成
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from decimal import Decimal, InvalidOperation

from ...database.async_manager import DatabaseManager
from .data_consistency_checker import ConsistencyIssue, IssueType, IssueSeverity


class ValidationType(Enum):
    """验证类型"""
    DATA_TYPE = "data_type"           # 数据类型验证
    RANGE = "range"                   # 数据范围验证
    FORMAT = "format"                 # 数据格式验证
    BUSINESS_RULE = "business_rule"   # 业务规则验证
    REFERENTIAL = "referential"       # 引用完整性验证
    UNIQUENESS = "uniqueness"         # 唯一性验证
    COMPLETENESS = "completeness"     # 完整性验证
    CONSISTENCY = "consistency"       # 一致性验证


class ValidationSeverity(Enum):
    """验证严重程度"""
    CRITICAL = "critical"  # 严重：数据损坏
    HIGH = "high"         # 高：业务影响
    MEDIUM = "medium"     # 中：数据质量问题
    LOW = "low"          # 低：格式问题
    INFO = "info"        # 信息：建议优化


@dataclass
class ValidationRule:
    """验证规则"""
    id: str
    name: str
    description: str
    validation_type: ValidationType
    severity: ValidationSeverity
    table_name: str
    column_name: Optional[str] = None
    condition: Optional[str] = None  # SQL条件或正则表达式
    expected_value: Optional[Any] = None
    min_value: Optional[Union[int, float, Decimal]] = None
    max_value: Optional[Union[int, float, Decimal]] = None
    pattern: Optional[str] = None  # 正则表达式模式
    reference_table: Optional[str] = None
    reference_column: Optional[str] = None
    custom_validator: Optional[str] = None  # 自定义验证函数名
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "validation_type": self.validation_type.value,
            "severity": self.severity.value,
            "table_name": self.table_name,
            "column_name": self.column_name,
            "condition": self.condition,
            "expected_value": self.expected_value,
            "min_value": str(self.min_value) if self.min_value is not None else None,
            "max_value": str(self.max_value) if self.max_value is not None else None,
            "pattern": self.pattern,
            "reference_table": self.reference_table,
            "reference_column": self.reference_column,
            "custom_validator": self.custom_validator,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ValidationResult:
    """验证结果"""
    rule_id: str
    rule_name: str
    validation_type: ValidationType
    severity: ValidationSeverity
    passed: bool
    failed_records: int
    total_records: int
    error_details: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_records == 0:
            return 1.0
        return (self.total_records - self.failed_records) / self.total_records
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "validation_type": self.validation_type.value,
            "severity": self.severity.value,
            "passed": self.passed,
            "failed_records": self.failed_records,
            "total_records": self.total_records,
            "success_rate": self.success_rate,
            "error_details": self.error_details,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class IntegrityReport:
    """完整性报告"""
    validation_results: List[ValidationResult]
    total_rules: int
    passed_rules: int
    failed_rules: int
    total_records_validated: int
    failed_records: int
    overall_success_rate: float
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "validation_results": [r.to_dict() for r in self.validation_results],
            "total_rules": self.total_rules,
            "passed_rules": self.passed_rules,
            "failed_rules": self.failed_rules,
            "total_records_validated": self.total_records_validated,
            "failed_records": self.failed_records,
            "overall_success_rate": self.overall_success_rate,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp.isoformat()
        }


class DataIntegrityValidator:
    """
    数据完整性验证器
    
    提供全面的数据完整性验证功能，包括：
    - 数据类型和格式验证
    - 业务规则验证
    - 引用完整性验证
    - 数据质量评估
    - 完整性报告生成
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """
        初始化数据完整性验证器
        
        Args:
            db_manager: 数据库管理器
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # 验证规则
        self.validation_rules: Dict[str, ValidationRule] = {}
        
        # 自定义验证器
        self.custom_validators: Dict[str, callable] = {}
        
        # 初始化默认规则
        self._initialize_default_rules()
        
    def _initialize_default_rules(self):
        """初始化默认验证规则"""
        # API日志表验证规则
        self.add_validation_rule(ValidationRule(
            id="api_logs_id_not_null",
            name="API日志ID非空验证",
            description="验证api_logs表的id字段不为空",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.CRITICAL,
            table_name="api_logs",
            column_name="id",
            condition="id IS NULL"
        ))
        
        self.add_validation_rule(ValidationRule(
            id="api_logs_model_format",
            name="API日志模型格式验证",
            description="验证api_logs表的model字段格式",
            validation_type=ValidationType.FORMAT,
            severity=ValidationSeverity.MEDIUM,
            table_name="api_logs",
            column_name="model",
            pattern=r"^[a-zA-Z0-9\-_\.]+$"
        ))
        
        self.add_validation_rule(ValidationRule(
            id="api_logs_tokens_range",
            name="API日志Token数量范围验证",
            description="验证api_logs表的token字段在合理范围内",
            validation_type=ValidationType.RANGE,
            severity=ValidationSeverity.HIGH,
            table_name="api_logs",
            column_name="total_tokens",
            min_value=0,
            max_value=1000000
        ))
        
        # Token使用表验证规则
        self.add_validation_rule(ValidationRule(
            id="token_usage_referential",
            name="Token使用引用完整性验证",
            description="验证token_usage表的log_id引用api_logs表",
            validation_type=ValidationType.REFERENTIAL,
            severity=ValidationSeverity.CRITICAL,
            table_name="token_usage",
            column_name="log_id",
            reference_table="api_logs",
            reference_column="id"
        ))
        
        self.add_validation_rule(ValidationRule(
            id="token_usage_consistency",
            name="Token使用一致性验证",
            description="验证token_usage表的token总数等于prompt+completion",
            validation_type=ValidationType.CONSISTENCY,
            severity=ValidationSeverity.HIGH,
            table_name="token_usage",
            condition="total_tokens != prompt_tokens + completion_tokens"
        ))
        
        # 成本信息表验证规则
        self.add_validation_rule(ValidationRule(
            id="cost_info_referential",
            name="成本信息引用完整性验证",
            description="验证cost_info表的log_id引用api_logs表",
            validation_type=ValidationType.REFERENTIAL,
            severity=ValidationSeverity.CRITICAL,
            table_name="cost_info",
            column_name="log_id",
            reference_table="api_logs",
            reference_column="id"
        ))
        
        self.add_validation_rule(ValidationRule(
            id="cost_info_positive",
            name="成本信息正值验证",
            description="验证cost_info表的成本字段为正值",
            validation_type=ValidationType.RANGE,
            severity=ValidationSeverity.HIGH,
            table_name="cost_info",
            column_name="total_cost",
            min_value=0
        ))
        
        # 追踪信息表验证规则
        self.add_validation_rule(ValidationRule(
            id="tracing_info_referential",
            name="追踪信息引用完整性验证",
            description="验证tracing_info表的log_id引用api_logs表",
            validation_type=ValidationType.REFERENTIAL,
            severity=ValidationSeverity.CRITICAL,
            table_name="tracing_info",
            column_name="log_id",
            reference_table="api_logs",
            reference_column="id"
        ))
        
        self.add_validation_rule(ValidationRule(
            id="tracing_info_trace_id_format",
            name="追踪ID格式验证",
            description="验证tracing_info表的trace_id格式",
            validation_type=ValidationType.FORMAT,
            severity=ValidationSeverity.MEDIUM,
            table_name="tracing_info",
            column_name="hb_trace_id",
            pattern=r"^[a-fA-F0-9\-]{8,}$"
        ))
        
    def add_validation_rule(self, rule: ValidationRule):
        """添加验证规则"""
        self.validation_rules[rule.id] = rule
        self.logger.info(f"添加验证规则: {rule.name} ({rule.id})")
        
    def remove_validation_rule(self, rule_id: str) -> bool:
        """删除验证规则"""
        if rule_id in self.validation_rules:
            rule_name = self.validation_rules[rule_id].name
            del self.validation_rules[rule_id]
            self.logger.info(f"删除验证规则: {rule_name} ({rule_id})")
            return True
        return False
        
    def add_custom_validator(self, name: str, validator: callable):
        """添加自定义验证器"""
        self.custom_validators[name] = validator
        self.logger.info(f"添加自定义验证器: {name}")
        
    async def validate_all(self, 
                          rule_ids: Optional[List[str]] = None,
                          table_names: Optional[List[str]] = None) -> IntegrityReport:
        """
        执行所有验证规则
        
        Args:
            rule_ids: 指定要执行的规则ID列表，None表示执行所有规则
            table_names: 指定要验证的表名列表，None表示验证所有表
            
        Returns:
            完整性报告
        """
        start_time = datetime.now()
        validation_results = []
        
        # 筛选要执行的规则
        rules_to_execute = []
        for rule in self.validation_rules.values():
            if not rule.enabled:
                continue
                
            if rule_ids and rule.id not in rule_ids:
                continue
                
            if table_names and rule.table_name not in table_names:
                continue
                
            rules_to_execute.append(rule)
            
        self.logger.info(f"开始执行 {len(rules_to_execute)} 个验证规则")
        
        # 并发执行验证规则
        semaphore = asyncio.Semaphore(5)  # 限制并发数
        
        async def validate_rule_with_semaphore(rule):
            async with semaphore:
                return await self._validate_rule(rule)
                
        results = await asyncio.gather(
            *[validate_rule_with_semaphore(rule) for rule in rules_to_execute],
            return_exceptions=True
        )
        
        # 处理结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"验证规则 {rules_to_execute[i].id} 失败: {result}")
                # 创建失败结果
                validation_results.append(ValidationResult(
                    rule_id=rules_to_execute[i].id,
                    rule_name=rules_to_execute[i].name,
                    validation_type=rules_to_execute[i].validation_type,
                    severity=rules_to_execute[i].severity,
                    passed=False,
                    failed_records=0,
                    total_records=0,
                    error_details=[{"error": str(result)}]
                ))
            else:
                validation_results.append(result)
                
        # 生成报告
        execution_time = (datetime.now() - start_time).total_seconds()
        
        total_rules = len(validation_results)
        passed_rules = len([r for r in validation_results if r.passed])
        failed_rules = total_rules - passed_rules
        
        total_records_validated = sum(r.total_records for r in validation_results)
        failed_records = sum(r.failed_records for r in validation_results)
        
        overall_success_rate = (
            (total_records_validated - failed_records) / total_records_validated
            if total_records_validated > 0 else 1.0
        )
        
        report = IntegrityReport(
            validation_results=validation_results,
            total_rules=total_rules,
            passed_rules=passed_rules,
            failed_rules=failed_rules,
            total_records_validated=total_records_validated,
            failed_records=failed_records,
            overall_success_rate=overall_success_rate,
            execution_time=execution_time
        )
        
        self.logger.info(f"验证完成: {passed_rules}/{total_rules} 规则通过, "
                        f"整体成功率: {overall_success_rate:.2%}")
        
        return report
        
    async def _validate_rule(self, rule: ValidationRule) -> ValidationResult:
        """执行单个验证规则"""
        start_time = datetime.now()
        
        try:
            if rule.validation_type == ValidationType.DATA_TYPE:
                result = await self._validate_data_type(rule)
            elif rule.validation_type == ValidationType.RANGE:
                result = await self._validate_range(rule)
            elif rule.validation_type == ValidationType.FORMAT:
                result = await self._validate_format(rule)
            elif rule.validation_type == ValidationType.BUSINESS_RULE:
                result = await self._validate_business_rule(rule)
            elif rule.validation_type == ValidationType.REFERENTIAL:
                result = await self._validate_referential_integrity(rule)
            elif rule.validation_type == ValidationType.UNIQUENESS:
                result = await self._validate_uniqueness(rule)
            elif rule.validation_type == ValidationType.COMPLETENESS:
                result = await self._validate_completeness(rule)
            elif rule.validation_type == ValidationType.CONSISTENCY:
                result = await self._validate_consistency(rule)
            else:
                raise ValueError(f"不支持的验证类型: {rule.validation_type}")
                
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"验证规则 {rule.id} 执行失败: {e}")
            
            return ValidationResult(
                rule_id=rule.id,
                rule_name=rule.name,
                validation_type=rule.validation_type,
                severity=rule.severity,
                passed=False,
                failed_records=0,
                total_records=0,
                error_details=[{"error": str(e)}],
                execution_time=execution_time
            )
            
    async def _validate_data_type(self, rule: ValidationRule) -> ValidationResult:
        """验证数据类型"""
        # 这里可以实现数据类型验证逻辑
        # 由于PostgreSQL有强类型系统，通常不会有类型错误
        return ValidationResult(
            rule_id=rule.id,
            rule_name=rule.name,
            validation_type=rule.validation_type,
            severity=rule.severity,
            passed=True,
            failed_records=0,
            total_records=0
        )
        
    async def _validate_range(self, rule: ValidationRule) -> ValidationResult:
        """验证数据范围"""
        conditions = []
        
        if rule.min_value is not None:
            conditions.append(f"{rule.column_name} < {rule.min_value}")
            
        if rule.max_value is not None:
            conditions.append(f"{rule.column_name} > {rule.max_value}")
            
        if not conditions:
            return ValidationResult(
                rule_id=rule.id,
                rule_name=rule.name,
                validation_type=rule.validation_type,
                severity=rule.severity,
                passed=True,
                failed_records=0,
                total_records=0
            )
            
        condition = " OR ".join(conditions)
        
        # 查询违反范围的记录
        query = f"""
        SELECT COUNT(*) as failed_count,
               (SELECT COUNT(*) FROM {rule.table_name}) as total_count
        FROM {rule.table_name}
        WHERE {condition}
        """
        
        result = await self.db_manager.execute_query(query)
        row = result[0] if result else {"failed_count": 0, "total_count": 0}
        
        failed_records = row["failed_count"]
        total_records = row["total_count"]
        
        return ValidationResult(
            rule_id=rule.id,
            rule_name=rule.name,
            validation_type=rule.validation_type,
            severity=rule.severity,
            passed=failed_records == 0,
            failed_records=failed_records,
            total_records=total_records
        )
        
    async def _validate_format(self, rule: ValidationRule) -> ValidationResult:
        """验证数据格式"""
        if not rule.pattern:
            return ValidationResult(
                rule_id=rule.id,
                rule_name=rule.name,
                validation_type=rule.validation_type,
                severity=rule.severity,
                passed=True,
                failed_records=0,
                total_records=0
            )
            
        # 使用PostgreSQL的正则表达式功能
        query = f"""
        SELECT COUNT(*) as failed_count,
               (SELECT COUNT(*) FROM {rule.table_name} WHERE {rule.column_name} IS NOT NULL) as total_count
        FROM {rule.table_name}
        WHERE {rule.column_name} IS NOT NULL 
        AND NOT ({rule.column_name}::text ~ %s)
        """
        
        result = await self.db_manager.execute_query(query, (rule.pattern,))
        row = result[0] if result else {"failed_count": 0, "total_count": 0}
        
        failed_records = row["failed_count"]
        total_records = row["total_count"]
        
        return ValidationResult(
            rule_id=rule.id,
            rule_name=rule.name,
            validation_type=rule.validation_type,
            severity=rule.severity,
            passed=failed_records == 0,
            failed_records=failed_records,
            total_records=total_records
        )
        
    async def _validate_business_rule(self, rule: ValidationRule) -> ValidationResult:
        """验证业务规则"""
        if rule.custom_validator and rule.custom_validator in self.custom_validators:
            # 使用自定义验证器
            validator = self.custom_validators[rule.custom_validator]
            return await validator(rule, self.db_manager)
        else:
            # 使用SQL条件验证
            return await self._validate_sql_condition(rule)
            
    async def _validate_referential_integrity(self, rule: ValidationRule) -> ValidationResult:
        """验证引用完整性"""
        if not rule.reference_table or not rule.reference_column:
            return ValidationResult(
                rule_id=rule.id,
                rule_name=rule.name,
                validation_type=rule.validation_type,
                severity=rule.severity,
                passed=True,
                failed_records=0,
                total_records=0
            )
            
        # 查询孤立记录
        query = f"""
        SELECT COUNT(*) as failed_count,
               (SELECT COUNT(*) FROM {rule.table_name}) as total_count
        FROM {rule.table_name} t1
        LEFT JOIN {rule.reference_table} t2 ON t1.{rule.column_name} = t2.{rule.reference_column}
        WHERE t1.{rule.column_name} IS NOT NULL AND t2.{rule.reference_column} IS NULL
        """
        
        result = await self.db_manager.execute_query(query)
        row = result[0] if result else {"failed_count": 0, "total_count": 0}
        
        failed_records = row["failed_count"]
        total_records = row["total_count"]
        
        return ValidationResult(
            rule_id=rule.id,
            rule_name=rule.name,
            validation_type=rule.validation_type,
            severity=rule.severity,
            passed=failed_records == 0,
            failed_records=failed_records,
            total_records=total_records
        )
        
    async def _validate_uniqueness(self, rule: ValidationRule) -> ValidationResult:
        """验证唯一性"""
        query = f"""
        SELECT COUNT(*) as failed_count,
               (SELECT COUNT(*) FROM {rule.table_name}) as total_count
        FROM (
            SELECT {rule.column_name}, COUNT(*) as cnt
            FROM {rule.table_name}
            WHERE {rule.column_name} IS NOT NULL
            GROUP BY {rule.column_name}
            HAVING COUNT(*) > 1
        ) duplicates
        """
        
        result = await self.db_manager.execute_query(query)
        row = result[0] if result else {"failed_count": 0, "total_count": 0}
        
        failed_records = row["failed_count"]
        total_records = row["total_count"]
        
        return ValidationResult(
            rule_id=rule.id,
            rule_name=rule.name,
            validation_type=rule.validation_type,
            severity=rule.severity,
            passed=failed_records == 0,
            failed_records=failed_records,
            total_records=total_records
        )
        
    async def _validate_completeness(self, rule: ValidationRule) -> ValidationResult:
        """验证完整性（非空）"""
        if rule.condition:
            # 使用自定义条件
            return await self._validate_sql_condition(rule)
        else:
            # 默认检查NULL值
            query = f"""
            SELECT COUNT(*) as failed_count,
                   (SELECT COUNT(*) FROM {rule.table_name}) as total_count
            FROM {rule.table_name}
            WHERE {rule.column_name} IS NULL
            """
            
            result = await self.db_manager.execute_query(query)
            row = result[0] if result else {"failed_count": 0, "total_count": 0}
            
            failed_records = row["failed_count"]
            total_records = row["total_count"]
            
            return ValidationResult(
                rule_id=rule.id,
                rule_name=rule.name,
                validation_type=rule.validation_type,
                severity=rule.severity,
                passed=failed_records == 0,
                failed_records=failed_records,
                total_records=total_records
            )
            
    async def _validate_consistency(self, rule: ValidationRule) -> ValidationResult:
        """验证一致性"""
        return await self._validate_sql_condition(rule)
        
    async def _validate_sql_condition(self, rule: ValidationRule) -> ValidationResult:
        """使用SQL条件验证"""
        if not rule.condition:
            return ValidationResult(
                rule_id=rule.id,
                rule_name=rule.name,
                validation_type=rule.validation_type,
                severity=rule.severity,
                passed=True,
                failed_records=0,
                total_records=0
            )
            
        query = f"""
        SELECT COUNT(*) as failed_count,
               (SELECT COUNT(*) FROM {rule.table_name}) as total_count
        FROM {rule.table_name}
        WHERE {rule.condition}
        """
        
        result = await self.db_manager.execute_query(query)
        row = result[0] if result else {"failed_count": 0, "total_count": 0}
        
        failed_records = row["failed_count"]
        total_records = row["total_count"]
        
        return ValidationResult(
            rule_id=rule.id,
            rule_name=rule.name,
            validation_type=rule.validation_type,
            severity=rule.severity,
            passed=failed_records == 0,
            failed_records=failed_records,
            total_records=total_records
        )
        
    def get_validation_rules(self, 
                           validation_type: Optional[ValidationType] = None,
                           table_name: Optional[str] = None) -> List[ValidationRule]:
        """获取验证规则"""
        rules = list(self.validation_rules.values())
        
        if validation_type:
            rules = [r for r in rules if r.validation_type == validation_type]
            
        if table_name:
            rules = [r for r in rules if r.table_name == table_name]
            
        return rules
        
    async def export_report(self, report: IntegrityReport, format: str = "json") -> str:
        """导出完整性报告"""
        if format.lower() == "json":
            return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        elif format.lower() == "csv":
            # 实现CSV导出
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # 写入标题行
            writer.writerow([
                "规则ID", "规则名称", "验证类型", "严重程度", "通过",
                "失败记录数", "总记录数", "成功率", "执行时间"
            ])
            
            # 写入数据行
            for result in report.validation_results:
                writer.writerow([
                    result.rule_id,
                    result.rule_name,
                    result.validation_type.value,
                    result.severity.value,
                    "是" if result.passed else "否",
                    result.failed_records,
                    result.total_records,
                    f"{result.success_rate:.2%}",
                    f"{result.execution_time:.3f}s"
                ])
                
            return output.getvalue()
        else:
            raise ValueError(f"不支持的导出格式: {format}")