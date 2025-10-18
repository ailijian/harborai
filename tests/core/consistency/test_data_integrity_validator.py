#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据完整性验证器测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal

from harborai.core.consistency.data_integrity_validator import (
    DataIntegrityValidator,
    ValidationRule,
    ValidationResult,
    IntegrityReport,
    ValidationType,
    ValidationSeverity
)
from harborai.database.async_manager import DatabaseManager


class TestDataIntegrityValidator:
    """数据完整性验证器测试类"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """模拟数据库管理器"""
        db_manager = AsyncMock(spec=DatabaseManager)
        return db_manager
        
    @pytest.fixture
    def validator(self, mock_db_manager):
        """创建验证器实例"""
        return DataIntegrityValidator(mock_db_manager)
        
    def test_initialization(self, validator):
        """测试初始化"""
        assert validator.db_manager is not None
        assert len(validator.validation_rules) > 0
        assert isinstance(validator.custom_validators, dict)
        
    def test_add_validation_rule(self, validator):
        """测试添加验证规则"""
        rule = ValidationRule(
            id="test_rule",
            name="测试规则",
            description="测试描述",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            table_name="test_table",
            column_name="test_column"
        )
        
        initial_count = len(validator.validation_rules)
        validator.add_validation_rule(rule)
        
        assert len(validator.validation_rules) == initial_count + 1
        assert validator.validation_rules["test_rule"] == rule
        
    def test_remove_validation_rule(self, validator):
        """测试删除验证规则"""
        # 添加一个规则
        rule = ValidationRule(
            id="test_rule_to_remove",
            name="待删除规则",
            description="测试描述",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            table_name="test_table"
        )
        validator.add_validation_rule(rule)
        
        # 删除规则
        result = validator.remove_validation_rule("test_rule_to_remove")
        assert result is True
        assert "test_rule_to_remove" not in validator.validation_rules
        
        # 删除不存在的规则
        result = validator.remove_validation_rule("non_existent_rule")
        assert result is False
        
    def test_add_custom_validator(self, validator):
        """测试添加自定义验证器"""
        def custom_validator(rule, db_manager):
            return ValidationResult(
                rule_id=rule.id,
                rule_name=rule.name,
                validation_type=rule.validation_type,
                severity=rule.severity,
                passed=True,
                failed_records=0,
                total_records=100
            )
            
        validator.add_custom_validator("test_validator", custom_validator)
        assert "test_validator" in validator.custom_validators
        assert validator.custom_validators["test_validator"] == custom_validator
        
    @pytest.mark.asyncio
    async def test_validate_range_success(self, validator, mock_db_manager):
        """测试范围验证 - 成功"""
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 0, "total_count": 100}
        ]
        
        rule = ValidationRule(
            id="range_test",
            name="范围测试",
            description="测试范围验证",
            validation_type=ValidationType.RANGE,
            severity=ValidationSeverity.HIGH,
            table_name="test_table",
            column_name="test_column",
            min_value=0,
            max_value=1000
        )
        
        result = await validator._validate_range(rule)
        
        assert result.passed is True
        assert result.failed_records == 0
        assert result.total_records == 100
        assert result.success_rate == 1.0
        
    @pytest.mark.asyncio
    async def test_validate_range_failure(self, validator, mock_db_manager):
        """测试范围验证 - 失败"""
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 5, "total_count": 100}
        ]
        
        rule = ValidationRule(
            id="range_test",
            name="范围测试",
            description="测试范围验证",
            validation_type=ValidationType.RANGE,
            severity=ValidationSeverity.HIGH,
            table_name="test_table",
            column_name="test_column",
            min_value=0,
            max_value=1000
        )
        
        result = await validator._validate_range(rule)
        
        assert result.passed is False
        assert result.failed_records == 5
        assert result.total_records == 100
        assert result.success_rate == 0.95
        
    @pytest.mark.asyncio
    async def test_validate_format_success(self, validator, mock_db_manager):
        """测试格式验证 - 成功"""
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 0, "total_count": 50}
        ]
        
        rule = ValidationRule(
            id="format_test",
            name="格式测试",
            description="测试格式验证",
            validation_type=ValidationType.FORMAT,
            severity=ValidationSeverity.MEDIUM,
            table_name="test_table",
            column_name="email",
            pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        )
        
        result = await validator._validate_format(rule)
        
        assert result.passed is True
        assert result.failed_records == 0
        assert result.total_records == 50
        
    @pytest.mark.asyncio
    async def test_validate_referential_integrity_success(self, validator, mock_db_manager):
        """测试引用完整性验证 - 成功"""
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 0, "total_count": 200}
        ]
        
        rule = ValidationRule(
            id="ref_test",
            name="引用完整性测试",
            description="测试引用完整性验证",
            validation_type=ValidationType.REFERENTIAL,
            severity=ValidationSeverity.CRITICAL,
            table_name="child_table",
            column_name="parent_id",
            reference_table="parent_table",
            reference_column="id"
        )
        
        result = await validator._validate_referential_integrity(rule)
        
        assert result.passed is True
        assert result.failed_records == 0
        assert result.total_records == 200
        
    @pytest.mark.asyncio
    async def test_validate_referential_integrity_failure(self, validator, mock_db_manager):
        """测试引用完整性验证 - 失败"""
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 3, "total_count": 200}
        ]
        
        rule = ValidationRule(
            id="ref_test",
            name="引用完整性测试",
            description="测试引用完整性验证",
            validation_type=ValidationType.REFERENTIAL,
            severity=ValidationSeverity.CRITICAL,
            table_name="child_table",
            column_name="parent_id",
            reference_table="parent_table",
            reference_column="id"
        )
        
        result = await validator._validate_referential_integrity(rule)
        
        assert result.passed is False
        assert result.failed_records == 3
        assert result.total_records == 200
        assert result.success_rate == 0.985
        
    @pytest.mark.asyncio
    async def test_validate_completeness_success(self, validator, mock_db_manager):
        """测试完整性验证 - 成功"""
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 0, "total_count": 150}
        ]
        
        rule = ValidationRule(
            id="completeness_test",
            name="完整性测试",
            description="测试完整性验证",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            table_name="test_table",
            column_name="required_field"
        )
        
        result = await validator._validate_completeness(rule)
        
        assert result.passed is True
        assert result.failed_records == 0
        assert result.total_records == 150
        
    @pytest.mark.asyncio
    async def test_validate_consistency_with_condition(self, validator, mock_db_manager):
        """测试一致性验证 - 使用条件"""
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 2, "total_count": 100}
        ]
        
        rule = ValidationRule(
            id="consistency_test",
            name="一致性测试",
            description="测试一致性验证",
            validation_type=ValidationType.CONSISTENCY,
            severity=ValidationSeverity.HIGH,
            table_name="token_usage",
            condition="total_tokens != prompt_tokens + completion_tokens"
        )
        
        result = await validator._validate_consistency(rule)
        
        assert result.passed is False
        assert result.failed_records == 2
        assert result.total_records == 100
        assert result.success_rate == 0.98
        
    @pytest.mark.asyncio
    async def test_validate_uniqueness(self, validator, mock_db_manager):
        """测试唯一性验证"""
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 1, "total_count": 100}
        ]
        
        rule = ValidationRule(
            id="uniqueness_test",
            name="唯一性测试",
            description="测试唯一性验证",
            validation_type=ValidationType.UNIQUENESS,
            severity=ValidationSeverity.HIGH,
            table_name="users",
            column_name="email"
        )
        
        result = await validator._validate_uniqueness(rule)
        
        assert result.passed is False
        assert result.failed_records == 1
        assert result.total_records == 100
        
    @pytest.mark.asyncio
    async def test_validate_all_success(self, validator, mock_db_manager):
        """测试执行所有验证规则 - 成功"""
        # 清空默认规则，添加测试规则
        validator.validation_rules.clear()
        
        # 添加测试规则
        rule1 = ValidationRule(
            id="test_rule_1",
            name="测试规则1",
            description="测试描述1",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            table_name="test_table1",
            column_name="test_column1"
        )
        
        rule2 = ValidationRule(
            id="test_rule_2",
            name="测试规则2",
            description="测试描述2",
            validation_type=ValidationType.RANGE,
            severity=ValidationSeverity.MEDIUM,
            table_name="test_table2",
            column_name="test_column2",
            min_value=0,
            max_value=100
        )
        
        validator.add_validation_rule(rule1)
        validator.add_validation_rule(rule2)
        
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 0, "total_count": 100}
        ]
        
        report = await validator.validate_all()
        
        assert isinstance(report, IntegrityReport)
        assert report.total_rules == 2
        assert report.passed_rules == 2
        assert report.failed_rules == 0
        assert report.overall_success_rate == 1.0
        assert len(report.validation_results) == 2
        
    @pytest.mark.asyncio
    async def test_validate_all_with_failures(self, validator, mock_db_manager):
        """测试执行所有验证规则 - 有失败"""
        # 清空默认规则，添加测试规则
        validator.validation_rules.clear()
        
        rule = ValidationRule(
            id="test_rule_fail",
            name="失败测试规则",
            description="测试失败情况",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.CRITICAL,
            table_name="test_table",
            column_name="test_column"
        )
        
        validator.add_validation_rule(rule)
        
        # 模拟数据库查询结果 - 有失败记录
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 5, "total_count": 100}
        ]
        
        report = await validator.validate_all()
        
        assert report.total_rules == 1
        assert report.passed_rules == 0
        assert report.failed_rules == 1
        assert report.failed_records == 5
        assert report.overall_success_rate == 0.95
        
    @pytest.mark.asyncio
    async def test_validate_all_with_rule_filter(self, validator, mock_db_manager):
        """测试执行指定验证规则"""
        # 清空默认规则，添加测试规则
        validator.validation_rules.clear()
        
        rule1 = ValidationRule(
            id="rule_1",
            name="规则1",
            description="描述1",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            table_name="table1",
            column_name="column1"
        )
        
        rule2 = ValidationRule(
            id="rule_2",
            name="规则2",
            description="描述2",
            validation_type=ValidationType.RANGE,
            severity=ValidationSeverity.MEDIUM,
            table_name="table2",
            column_name="column2",
            min_value=0
        )
        
        validator.add_validation_rule(rule1)
        validator.add_validation_rule(rule2)
        
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 0, "total_count": 50}
        ]
        
        # 只执行rule_1
        report = await validator.validate_all(rule_ids=["rule_1"])
        
        assert report.total_rules == 1
        assert len(report.validation_results) == 1
        assert report.validation_results[0].rule_id == "rule_1"
        
    @pytest.mark.asyncio
    async def test_validate_all_with_table_filter(self, validator, mock_db_manager):
        """测试执行指定表的验证规则"""
        # 清空默认规则，添加测试规则
        validator.validation_rules.clear()
        
        rule1 = ValidationRule(
            id="rule_table1",
            name="表1规则",
            description="表1描述",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            table_name="table1",
            column_name="column1"
        )
        
        rule2 = ValidationRule(
            id="rule_table2",
            name="表2规则",
            description="表2描述",
            validation_type=ValidationType.RANGE,
            severity=ValidationSeverity.MEDIUM,
            table_name="table2",
            column_name="column2",
            min_value=0
        )
        
        validator.add_validation_rule(rule1)
        validator.add_validation_rule(rule2)
        
        # 模拟数据库查询结果
        mock_db_manager.execute_query.return_value = [
            {"failed_count": 0, "total_count": 50}
        ]
        
        # 只验证table1
        report = await validator.validate_all(table_names=["table1"])
        
        assert report.total_rules == 1
        assert len(report.validation_results) == 1
        assert report.validation_results[0].rule_id == "rule_table1"
        
    @pytest.mark.asyncio
    async def test_validate_rule_with_exception(self, validator, mock_db_manager):
        """测试验证规则异常处理"""
        # 模拟数据库查询异常
        mock_db_manager.execute_query.side_effect = Exception("数据库连接失败")
        
        rule = ValidationRule(
            id="exception_test",
            name="异常测试",
            description="测试异常处理",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            table_name="test_table",
            column_name="test_column"
        )
        
        result = await validator._validate_rule(rule)
        
        assert result.passed is False
        assert len(result.error_details) > 0
        assert "数据库连接失败" in result.error_details[0]["error"]
        
    def test_get_validation_rules_all(self, validator):
        """测试获取所有验证规则"""
        rules = validator.get_validation_rules()
        assert len(rules) > 0
        
    def test_get_validation_rules_by_type(self, validator):
        """测试按类型获取验证规则"""
        rules = validator.get_validation_rules(validation_type=ValidationType.REFERENTIAL)
        assert all(rule.validation_type == ValidationType.REFERENTIAL for rule in rules)
        
    def test_get_validation_rules_by_table(self, validator):
        """测试按表名获取验证规则"""
        rules = validator.get_validation_rules(table_name="api_logs")
        assert all(rule.table_name == "api_logs" for rule in rules)
        
    @pytest.mark.asyncio
    async def test_export_report_json(self, validator, mock_db_manager):
        """测试导出JSON格式报告"""
        # 创建测试报告
        result = ValidationResult(
            rule_id="test_rule",
            rule_name="测试规则",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            passed=True,
            failed_records=0,
            total_records=100
        )
        
        report = IntegrityReport(
            validation_results=[result],
            total_rules=1,
            passed_rules=1,
            failed_rules=0,
            total_records_validated=100,
            failed_records=0,
            overall_success_rate=1.0,
            execution_time=0.5
        )
        
        json_output = await validator.export_report(report, "json")
        
        assert isinstance(json_output, str)
        assert "test_rule" in json_output
        assert "测试规则" in json_output
        
    @pytest.mark.asyncio
    async def test_export_report_csv(self, validator, mock_db_manager):
        """测试导出CSV格式报告"""
        # 创建测试报告
        result = ValidationResult(
            rule_id="test_rule",
            rule_name="测试规则",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            passed=True,
            failed_records=0,
            total_records=100
        )
        
        report = IntegrityReport(
            validation_results=[result],
            total_rules=1,
            passed_rules=1,
            failed_rules=0,
            total_records_validated=100,
            failed_records=0,
            overall_success_rate=1.0,
            execution_time=0.5
        )
        
        csv_output = await validator.export_report(report, "csv")
        
        assert isinstance(csv_output, str)
        assert "规则ID" in csv_output
        assert "test_rule" in csv_output
        assert "测试规则" in csv_output
        
    @pytest.mark.asyncio
    async def test_export_report_unsupported_format(self, validator):
        """测试导出不支持的格式"""
        result = ValidationResult(
            rule_id="test_rule",
            rule_name="测试规则",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            passed=True,
            failed_records=0,
            total_records=100
        )
        
        report = IntegrityReport(
            validation_results=[result],
            total_rules=1,
            passed_rules=1,
            failed_rules=0,
            total_records_validated=100,
            failed_records=0,
            overall_success_rate=1.0,
            execution_time=0.5
        )
        
        with pytest.raises(ValueError, match="不支持的导出格式"):
            await validator.export_report(report, "xml")


class TestValidationRule:
    """验证规则测试类"""
    
    def test_validation_rule_creation(self):
        """测试验证规则创建"""
        rule = ValidationRule(
            id="test_rule",
            name="测试规则",
            description="测试描述",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            table_name="test_table",
            column_name="test_column",
            min_value=0,
            max_value=100,
            pattern=r"^\d+$",
            reference_table="ref_table",
            reference_column="ref_column"
        )
        
        assert rule.id == "test_rule"
        assert rule.name == "测试规则"
        assert rule.validation_type == ValidationType.COMPLETENESS
        assert rule.severity == ValidationSeverity.HIGH
        assert rule.min_value == 0
        assert rule.max_value == 100
        assert rule.enabled is True
        
    def test_validation_rule_to_dict(self):
        """测试验证规则转换为字典"""
        rule = ValidationRule(
            id="test_rule",
            name="测试规则",
            description="测试描述",
            validation_type=ValidationType.RANGE,
            severity=ValidationSeverity.MEDIUM,
            table_name="test_table",
            column_name="test_column",
            min_value=Decimal("0.5"),
            max_value=Decimal("99.9")
        )
        
        rule_dict = rule.to_dict()
        
        assert rule_dict["id"] == "test_rule"
        assert rule_dict["name"] == "测试规则"
        assert rule_dict["validation_type"] == "range"
        assert rule_dict["severity"] == "medium"
        assert rule_dict["min_value"] == "0.5"
        assert rule_dict["max_value"] == "99.9"


class TestValidationResult:
    """验证结果测试类"""
    
    def test_validation_result_creation(self):
        """测试验证结果创建"""
        result = ValidationResult(
            rule_id="test_rule",
            rule_name="测试规则",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            passed=False,
            failed_records=5,
            total_records=100,
            execution_time=0.5
        )
        
        assert result.rule_id == "test_rule"
        assert result.passed is False
        assert result.failed_records == 5
        assert result.total_records == 100
        assert result.success_rate == 0.95
        assert result.execution_time == 0.5
        
    def test_validation_result_success_rate_zero_records(self):
        """测试零记录时的成功率"""
        result = ValidationResult(
            rule_id="test_rule",
            rule_name="测试规则",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            passed=True,
            failed_records=0,
            total_records=0
        )
        
        assert result.success_rate == 1.0
        
    def test_validation_result_to_dict(self):
        """测试验证结果转换为字典"""
        result = ValidationResult(
            rule_id="test_rule",
            rule_name="测试规则",
            validation_type=ValidationType.FORMAT,
            severity=ValidationSeverity.LOW,
            passed=True,
            failed_records=0,
            total_records=50,
            error_details=[{"error": "测试错误"}],
            execution_time=0.3
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["rule_id"] == "test_rule"
        assert result_dict["validation_type"] == "format"
        assert result_dict["severity"] == "low"
        assert result_dict["passed"] is True
        assert result_dict["success_rate"] == 1.0
        assert len(result_dict["error_details"]) == 1


class TestIntegrityReport:
    """完整性报告测试类"""
    
    def test_integrity_report_creation(self):
        """测试完整性报告创建"""
        result1 = ValidationResult(
            rule_id="rule1",
            rule_name="规则1",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            passed=True,
            failed_records=0,
            total_records=100
        )
        
        result2 = ValidationResult(
            rule_id="rule2",
            rule_name="规则2",
            validation_type=ValidationType.RANGE,
            severity=ValidationSeverity.MEDIUM,
            passed=False,
            failed_records=5,
            total_records=200
        )
        
        report = IntegrityReport(
            validation_results=[result1, result2],
            total_rules=2,
            passed_rules=1,
            failed_rules=1,
            total_records_validated=300,
            failed_records=5,
            overall_success_rate=0.983,
            execution_time=1.5
        )
        
        assert len(report.validation_results) == 2
        assert report.total_rules == 2
        assert report.passed_rules == 1
        assert report.failed_rules == 1
        assert report.overall_success_rate == 0.983
        
    def test_integrity_report_to_dict(self):
        """测试完整性报告转换为字典"""
        result = ValidationResult(
            rule_id="test_rule",
            rule_name="测试规则",
            validation_type=ValidationType.COMPLETENESS,
            severity=ValidationSeverity.HIGH,
            passed=True,
            failed_records=0,
            total_records=100
        )
        
        report = IntegrityReport(
            validation_results=[result],
            total_rules=1,
            passed_rules=1,
            failed_rules=0,
            total_records_validated=100,
            failed_records=0,
            overall_success_rate=1.0,
            execution_time=0.5
        )
        
        report_dict = report.to_dict()
        
        assert len(report_dict["validation_results"]) == 1
        assert report_dict["total_rules"] == 1
        assert report_dict["passed_rules"] == 1
        assert report_dict["overall_success_rate"] == 1.0