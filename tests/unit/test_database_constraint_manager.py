#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库约束管理器单元测试

测试DatabaseConstraintManager的各项功能，包括：
- 约束信息获取
- 外键约束检查
- 数据完整性约束验证
- 业务规则约束检查
- 约束创建和管理
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from harborai.core.consistency.database_constraint_manager import (
    DatabaseConstraintManager,
    ConstraintInfo,
    ConstraintViolation,
    ConstraintReport,
    ConstraintType,
    ViolationSeverity
)


class TestDatabaseConstraintManager:
    """数据库约束管理器测试类"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """模拟数据库管理器"""
        db_manager = AsyncMock()
        return db_manager
    
    @pytest.fixture
    def constraint_manager(self, mock_db_manager):
        """创建数据库约束管理器实例"""
        return DatabaseConstraintManager(mock_db_manager)
    
    @pytest.mark.asyncio
    async def test_get_constraint_info_success(self, constraint_manager, mock_db_manager):
        """测试获取约束信息 - 成功场景"""
        # 准备测试数据
        mock_db_manager.execute_query.return_value = [
            {
                'constraint_name': 'token_usage_log_id_fkey',
                'constraint_type': 'FOREIGN KEY',
                'table_name': 'token_usage',
                'column_names': ['log_id'],
                'definition': 'FOREIGN KEY (log_id) REFERENCES api_logs(id)',
                'is_valid': True
            },
            {
                'constraint_name': 'cost_info_log_id_fkey',
                'constraint_type': 'FOREIGN KEY',
                'table_name': 'cost_info',
                'column_names': ['log_id'],
                'definition': 'FOREIGN KEY (log_id) REFERENCES api_logs(id)',
                'is_valid': True
            }
        ]
        
        # 执行获取约束信息
        constraints = await constraint_manager.get_constraint_info()
        
        # 验证结果
        assert len(constraints) == 2
        assert all(isinstance(constraint, ConstraintInfo) for constraint in constraints)
        assert constraints[0].constraint_name == 'token_usage_log_id_fkey'
        assert constraints[0].constraint_type == 'FOREIGN KEY'
        assert constraints[1].table_name == 'cost_info'
    
    @pytest.mark.asyncio
    async def test_check_foreign_key_violations_none(self, constraint_manager, mock_db_manager):
        """测试外键约束检查 - 无违反场景"""
        # 准备测试数据
        mock_db_manager.execute_query.return_value = []  # 没有违反外键约束的记录
        
        # 执行检查
        violations = await constraint_manager.check_foreign_key_violations()
        
        # 验证结果
        assert len(violations) == 0
    
    @pytest.mark.asyncio
    async def test_check_foreign_key_violations_found(self, constraint_manager, mock_db_manager):
        """测试外键约束检查 - 发现违反场景"""
        # 准备测试数据 - 有违反外键约束的记录
        mock_db_manager.execute_query.side_effect = [
            [{'id': 1, 'log_id': 999}],  # token_usage中的孤立记录
            [{'id': 2, 'log_id': 888}],  # cost_info中的孤立记录
            [{'id': 3, 'log_id': 777}]   # tracing_info中的孤立记录
        ]
        
        # 执行检查
        violations = await constraint_manager.check_foreign_key_violations()
        
        # 验证结果
        assert len(violations) == 3
        assert all('foreign_key_violation' in violation.violation_type for violation in violations)
    
    @pytest.mark.asyncio
    async def test_check_data_integrity_violations_none(self, constraint_manager, mock_db_manager):
        """测试数据完整性约束检查 - 无违反场景"""
        # 准备测试数据
        mock_db_manager.execute_query.return_value = []  # 没有违反数据完整性的记录
        
        # 执行检查
        violations = await constraint_manager.check_data_integrity_violations()
        
        # 验证结果
        assert len(violations) == 0
    
    @pytest.mark.asyncio
    async def test_check_data_integrity_violations_found(self, constraint_manager, mock_db_manager):
        """测试数据完整性约束检查 - 发现违反场景"""
        # 准备测试数据 - 有违反数据完整性的记录
        mock_db_manager.execute_query.side_effect = [
            [{'id': 1, 'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 50}],  # token不一致
            [{'id': 2, 'confidence': 1.5}],  # 置信度超出范围
            [{'id': 3, 'prompt_tokens': -5, 'completion_tokens': 10, 'total_tokens': 5}],  # 负数token
            [{'id': 4, 'input_cost': -0.01, 'output_cost': 0.02, 'total_cost': 0.01}],  # 负数成本
            [{'id': 5, 'input_cost': 0.01, 'output_cost': 0.02, 'total_cost': 0.05}],  # 成本不一致
            [{'id': 6, 'duration_ms': -100}]  # 负数持续时间
        ]
        
        # 执行检查
        violations = await constraint_manager.check_data_integrity_violations()
        
        # 验证结果
        assert len(violations) == 6
        assert all('check_constraint_violation' in violation.violation_type for violation in violations)
    
    @pytest.mark.asyncio
    async def test_check_business_rule_violations_none(self, constraint_manager, mock_db_manager):
        """测试业务规则约束检查 - 无违反场景"""
        # 准备测试数据
        mock_db_manager.execute_query.return_value = []  # 没有违反业务规则的记录
        
        # 执行检查
        violations = await constraint_manager.check_business_rule_violations()
        
        # 验证结果
        assert len(violations) == 0
    
    @pytest.mark.asyncio
    async def test_check_business_rule_violations_found(self, constraint_manager, mock_db_manager):
        """测试业务规则约束检查 - 发现违反场景"""
        # 准备测试数据 - 有违反业务规则的记录
        mock_db_manager.execute_query.side_effect = [
            [{'id': 1}],  # 缺少token记录
            [{'id': 2}],  # 缺少成本记录
            [{'id': 3, 'total_tokens': 0}],  # 成功调用但token无效
            [{'id': 4, 'api_time': '2023-01-01 10:00:00', 'trace_time': '2023-01-01 10:10:00'}]  # 时间不一致
        ]
        
        # 执行检查
        violations = await constraint_manager.check_business_rule_violations()
        
        # 验证结果
        assert len(violations) == 4
        assert all('business_rule_violation' in violation.violation_type for violation in violations)
    
    @pytest.mark.asyncio
    async def test_create_missing_constraints_success(self, constraint_manager, mock_db_manager):
        """测试创建缺失约束 - 成功场景"""
        # 模拟约束创建成功
        mock_db_manager.execute_query.return_value = None
        
        # 执行创建约束
        result = await constraint_manager.create_missing_constraints()
        
        # 验证结果
        assert result is True
        # 验证调用了创建约束的SQL
        assert mock_db_manager.execute_query.call_count > 0
    
    @pytest.mark.asyncio
    async def test_create_missing_constraints_failure(self, constraint_manager, mock_db_manager):
        """测试创建缺失约束 - 失败场景"""
        # 模拟约束创建失败
        mock_db_manager.execute_query.side_effect = Exception("约束创建失败")
        
        # 执行创建约束
        result = await constraint_manager.create_missing_constraints()
        
        # 验证结果
        assert result is False
    
    @pytest.mark.asyncio
    async def test_generate_constraint_report(self, constraint_manager, mock_db_manager):
        """测试生成约束报告"""
        # 模拟约束信息查询
        mock_db_manager.execute_query.return_value = [
            {
                'constraint_name': 'test_constraint',
                'table_name': 'test_table',
                'constraint_type': 'FOREIGN KEY',
                'column_names': ['test_column'],
                'definition': 'FOREIGN KEY (test_column) REFERENCES other_table(id)',
                'is_valid': True
            }
        ]
        
        # 创建测试违反
        violations = [
            ConstraintViolation(
                violation_id="test_violation_1",
                constraint_name="test_constraint",
                table_name="test_table",
                record_id="1",
                violation_type="foreign_key_violation",
                description="测试违反",
                detected_at=datetime.now()
            )
        ]
        
        # 执行报告生成
        report = await constraint_manager.generate_constraint_report(violations)
        
        # 验证结果
        assert isinstance(report, ConstraintReport)
        assert len(report.violations_found) == 1
        assert report.violations_found[0].violation_id == "test_violation_1"
    
    @pytest.mark.asyncio
    async def test_validate_table_constraints_success(self, constraint_manager, mock_db_manager):
        """测试验证表约束 - 成功场景"""
        # 模拟没有违反
        mock_db_manager.execute_query.return_value = []
        
        # 执行验证
        result = await constraint_manager.validate_table_constraints('test_table')
        
        # 验证结果
        assert result is True
    
    @pytest.mark.asyncio
    async def test_validate_table_constraints_with_violations(self, constraint_manager, mock_db_manager):
        """测试验证表约束 - 有违反场景"""
        # 模拟有违反
        mock_db_manager.execute_query.return_value = [
            {'id': 1, 'log_id': 999}  # 外键违反
        ]
        
        # 执行验证
        result = await constraint_manager.validate_table_constraints('test_table')
        
        # 验证结果
        assert result is False
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, constraint_manager, mock_db_manager):
        """测试数据库错误处理"""
        # 模拟数据库连接失败
        mock_db_manager.execute_query.side_effect = Exception("数据库连接失败")
        
        # 执行检查，应该捕获异常
        with pytest.raises(Exception, match="数据库连接失败"):
            await constraint_manager.check_foreign_key_violations()
        
    def test_constraint_info_creation(self):
        """测试约束信息对象创建"""
        constraint = ConstraintInfo(
            constraint_name='test_constraint',
            table_name='test_table',
            constraint_type='FOREIGN KEY',
            column_names=['test_column'],
            definition='FOREIGN KEY (test_column) REFERENCES other_table(id)',
            is_enabled=True,
            is_valid=True
        )
        
        assert constraint.constraint_name == 'test_constraint'
        assert constraint.constraint_type == 'FOREIGN KEY'
        assert constraint.table_name == 'test_table'
        assert constraint.column_names == ['test_column']
        assert constraint.is_enabled is True
        assert constraint.is_valid is True
    
    def test_constraint_violation_creation(self):
        """测试约束违反对象创建"""
        violation = ConstraintViolation(
            violation_id="test_violation",
            constraint_name="test_constraint",
            table_name='test_table',
            record_id="1",
            violation_type="foreign_key_violation",
            description='测试违反',
            detected_at=datetime.now(),
            fix_suggestion="修复建议"
        )
        
        assert violation.violation_id == "test_violation"
        assert violation.constraint_name == "test_constraint"
        assert violation.table_name == 'test_table'
        assert violation.record_id == "1"
        assert violation.violation_type == "foreign_key_violation"
        assert violation.description == '测试违反'
        assert violation.fix_suggestion == "修复建议"
        assert isinstance(violation.detected_at, datetime)
    
    def test_constraint_report_creation(self):
        """测试约束报告对象创建"""
        violations = [
            ConstraintViolation(
                violation_id="test_violation_1",
                constraint_name="test_constraint",
                table_name='table1',
                record_id="1",
                violation_type="foreign_key_violation",
                description='高严重性违反',
                detected_at=datetime.now()
            ),
            ConstraintViolation(
                violation_id="test_violation_2",
                constraint_name="test_constraint2",
                table_name='table2',
                record_id="2",
                violation_type="check_constraint_violation",
                description='中等严重性违反',
                detected_at=datetime.now()
            )
        ]
        
        constraints = [
            ConstraintInfo(
                constraint_name='test_constraint',
                table_name='test_table',
                constraint_type='FOREIGN KEY',
                column_names=['test_column'],
                definition='FOREIGN KEY (test_column) REFERENCES other_table(id)',
                is_enabled=True,
                is_valid=True
            )
        ]
        
        report = ConstraintReport(
            check_id='test_check_123',
            check_timestamp=datetime.now(),
            constraints_checked=constraints,
            violations_found=violations,
            check_duration_ms=100.5,
            summary={'total_violations': 2, 'foreign_key_violations': 1, 'check_constraint_violations': 1}
        )
        
        assert report.check_id == 'test_check_123'
        assert len(report.violations_found) == 2
        assert len(report.constraints_checked) == 1
        assert report.check_duration_ms == 100.5
        assert report.summary['total_violations'] == 2
        assert isinstance(report.check_timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_concurrent_constraint_checks(self, constraint_manager, mock_db_manager):
        """测试并发约束检查"""
        # 模拟各项检查
        mock_db_manager.execute_query.return_value = []
        
        # 并发执行多个检查
        tasks = [
            constraint_manager.check_foreign_key_violations(),
            constraint_manager.check_data_integrity_violations(),
            constraint_manager.check_business_rule_violations()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 验证结果
        assert len(results) == 3
        assert all(isinstance(result, list) for result in results)
    
    @pytest.mark.asyncio
    async def test_constraint_type_filtering(self, constraint_manager, mock_db_manager):
        """测试约束类型过滤"""
        # 准备测试数据
        mock_db_manager.execute_query.return_value = [
            {
                'constraint_name': 'fk_constraint',
                'constraint_type': 'FOREIGN KEY',
                'table_name': 'test_table',
                'column_names': ['test_column'],
                'definition': 'FOREIGN KEY (test_column) REFERENCES other_table(id)',
                'is_enabled': True,
                'is_valid': True
            },
            {
                'constraint_name': 'check_constraint',
                'constraint_type': 'CHECK',
                'table_name': 'test_table',
                'column_names': ['test_column'],
                'definition': 'CHECK (test_column > 0)',
                'is_enabled': True,
                'is_valid': True
            }
        ]
        
        # 执行获取约束信息
        constraints = await constraint_manager.get_constraint_info()
        
        # 验证结果
        fk_constraints = [c for c in constraints if c.constraint_type == 'FOREIGN KEY']
        check_constraints = [c for c in constraints if c.constraint_type == 'CHECK']
        
        assert len(fk_constraints) == 1
        assert len(check_constraints) == 1
    
    @pytest.mark.asyncio
    async def test_constraint_repair_simulation(self, constraint_manager, mock_db_manager):
        """测试约束修复模拟"""
        # 模拟约束修复过程
        mock_db_manager.execute_query.side_effect = [
            # 第一次检查发现违反 - token_usage表
            [{'id': 1, 'log_id': 999}],
            # cost_info表检查
            [],
            # tracing_info表检查
            [],
            # create_missing_constraints 调用 - 6个SQL语句 (3个外键 + 3个CHECK约束)
            None, None, None, None, None, None,
            # 修复后检查无违反 - token_usage表
            [],
            # cost_info表检查
            [],
            # tracing_info表检查
            []
        ]
        
        # 第一次检查
        violations_before = await constraint_manager.check_foreign_key_violations()
        assert len(violations_before) == 1
        
        # 模拟修复过程
        repair_result = await constraint_manager.create_missing_constraints()
        assert repair_result is True
        
        # 第二次检查
        violations_after = await constraint_manager.check_foreign_key_violations()
        assert len(violations_after) == 0