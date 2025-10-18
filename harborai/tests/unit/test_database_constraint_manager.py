"""
数据库约束管理器测试

测试DatabaseConstraintManager的异步功能
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from harborai.core.consistency.database_constraint_manager import (
    DatabaseConstraintManager,
    ConstraintInfo,
    ConstraintViolation,
    ConstraintReport
)
from harborai.database.async_manager import DatabaseManager


@pytest.fixture
def mock_db_manager():
    """创建模拟的数据库管理器"""
    return AsyncMock(spec=DatabaseManager)


@pytest.fixture
def constraint_manager(mock_db_manager):
    """创建约束管理器实例"""
    return DatabaseConstraintManager(mock_db_manager)


@pytest.mark.asyncio
async def test_get_all_constraints_success(constraint_manager, mock_db_manager):
    """测试成功获取所有约束信息"""
    # 模拟数据库返回
    mock_db_manager.execute_query.return_value = [
        {
            'constraint_name': 'token_usage_log_id_fkey',
            'table_name': 'token_usage',
            'constraint_type': 'FOREIGN KEY',
            'column_names': ['log_id'],
            'definition': 'FOREIGN KEY (log_id) REFERENCES api_logs(id)',
            'is_valid': True
        },
        {
            'constraint_name': 'token_usage_consistency_check',
            'table_name': 'token_usage',
            'constraint_type': 'CHECK',
            'column_names': ['total_tokens', 'prompt_tokens', 'completion_tokens'],
            'definition': 'CHECK (total_tokens = prompt_tokens + completion_tokens)',
            'is_valid': True
        }
    ]
    
    # 执行测试
    constraints = await constraint_manager._get_all_constraints()
    
    # 验证结果
    assert len(constraints) == 2
    assert constraints[0].constraint_name == 'token_usage_log_id_fkey'
    assert constraints[0].constraint_type == 'FOREIGN KEY'
    assert constraints[1].constraint_name == 'token_usage_consistency_check'
    assert constraints[1].constraint_type == 'CHECK'


@pytest.mark.asyncio
async def test_check_foreign_key_constraints_with_violations(constraint_manager, mock_db_manager):
    """测试外键约束检查发现违反"""
    # 模拟数据库返回孤立记录
    mock_db_manager.execute_query.side_effect = [
        # token_usage孤立记录
        [{'id': 1, 'log_id': 999}],
        # cost_info孤立记录
        [{'id': 2, 'log_id': 888}],
        # tracing_info孤立记录
        [{'id': 3, 'log_id': 777}]
    ]
    
    # 执行测试
    violations = await constraint_manager._check_foreign_key_constraints()
    
    # 验证结果
    assert len(violations) == 3
    assert violations[0].table_name == 'token_usage'
    assert violations[0].violation_type == 'foreign_key_violation'
    assert violations[1].table_name == 'cost_info'
    assert violations[2].table_name == 'tracing_info'


@pytest.mark.asyncio
async def test_check_check_constraints_with_violations(constraint_manager, mock_db_manager):
    """测试CHECK约束检查发现违反"""
    # 模拟数据库返回违反记录
    mock_db_manager.execute_query.side_effect = [
        # token一致性违反
        [{'id': 1, 'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 25}],
        # 置信度范围违反
        [{'id': 2, 'confidence': 1.5}],
        # 负数token
        [{'id': 3, 'prompt_tokens': -5, 'completion_tokens': 10, 'total_tokens': 5}],
        # 负数成本
        [{'id': 4, 'input_cost': -0.01, 'output_cost': 0.02, 'total_cost': 0.01}],
        # 成本一致性违反
        [{'id': 5, 'input_cost': 0.01, 'output_cost': 0.02, 'total_cost': 0.05}],
        # 负数持续时间
        [{'id': 6, 'duration_ms': -100}]
    ]
    
    # 执行测试
    violations = await constraint_manager._check_check_constraints()
    
    # 验证结果
    assert len(violations) == 6
    assert violations[0].constraint_name == 'token_usage_consistency'
    assert violations[1].constraint_name == 'token_usage_confidence_range'
    assert violations[2].constraint_name == 'token_usage_positive_tokens'


@pytest.mark.asyncio
async def test_check_business_rule_constraints_with_violations(constraint_manager, mock_db_manager):
    """测试业务规则约束检查发现违反"""
    # 模拟数据库返回违反记录
    mock_db_manager.execute_query.side_effect = [
        # 缺少token记录
        [{'id': 1}],
        # 缺少成本记录
        [{'id': 2}],
        # 无效成功token
        [{'id': 3, 'total_tokens': 0}],
        # 时间不匹配
        [{'id': 4, 'api_time': datetime.now(), 'trace_time': datetime.now()}]
    ]
    
    # 执行测试
    violations = await constraint_manager._check_business_rule_constraints()
    
    # 验证结果
    assert len(violations) == 4
    assert violations[0].constraint_name == 'business_rule_token_required'
    assert violations[1].constraint_name == 'business_rule_cost_required'
    assert violations[2].constraint_name == 'business_rule_success_token_valid'
    assert violations[3].constraint_name == 'business_rule_time_consistency'


@pytest.mark.asyncio
async def test_check_all_constraints_success(constraint_manager, mock_db_manager):
    """测试完整的约束检查流程"""
    # 模拟_get_all_constraints返回
    constraint_manager._get_all_constraints = AsyncMock(return_value=[
        ConstraintInfo(
            constraint_name='test_constraint',
            table_name='test_table',
            constraint_type='CHECK',
            column_names=['test_column'],
            definition='CHECK (test_column > 0)',
            is_enabled=True,
            is_valid=True
        )
    ])
    
    # 模拟各种约束检查返回
    constraint_manager._check_foreign_key_constraints = AsyncMock(return_value=[])
    constraint_manager._check_check_constraints = AsyncMock(return_value=[])
    constraint_manager._check_business_rule_constraints = AsyncMock(return_value=[])
    
    # 执行测试
    report = await constraint_manager.check_all_constraints()
    
    # 验证结果
    assert isinstance(report, ConstraintReport)
    assert len(report.constraints_checked) == 1
    assert len(report.violations_found) == 0
    assert report.check_duration_ms >= 0  # 允许为0，因为测试执行很快
    assert report.summary['total_violations'] == 0


@pytest.mark.asyncio
async def test_create_missing_constraints_success(constraint_manager, mock_db_manager):
    """测试创建缺失约束"""
    # 模拟数据库执行成功
    mock_db_manager.execute_query.return_value = None
    
    # 执行测试
    executed_sqls = await constraint_manager.create_missing_constraints()
    
    # 验证结果
    assert len(executed_sqls) == 6  # 3个外键 + 3个CHECK约束
    assert mock_db_manager.execute_query.call_count == 6


@pytest.mark.asyncio
async def test_validate_constraint_exists(constraint_manager, mock_db_manager):
    """测试验证约束存在"""
    # 模拟数据库返回约束存在且有效
    mock_db_manager.execute_query.return_value = [{'convalidated': True}]
    
    # 执行测试
    result = await constraint_manager.validate_constraint('test_constraint')
    
    # 验证结果
    assert result is True


@pytest.mark.asyncio
async def test_validate_constraint_not_exists(constraint_manager, mock_db_manager):
    """测试验证约束不存在"""
    # 模拟数据库返回空结果
    mock_db_manager.execute_query.return_value = []
    
    # 执行测试
    result = await constraint_manager.validate_constraint('nonexistent_constraint')
    
    # 验证结果
    assert result is False


@pytest.mark.asyncio
async def test_generate_constraint_summary(constraint_manager):
    """测试生成约束摘要"""
    # 创建测试违反记录
    violations = [
        ConstraintViolation(
            violation_id='fk_1',
            constraint_name='test_fk',
            table_name='test_table',
            record_id='1',
            violation_type='foreign_key_violation',
            description='Test FK violation',
            detected_at=datetime.now()
        ),
        ConstraintViolation(
            violation_id='check_1',
            constraint_name='test_check',
            table_name='test_table',
            record_id='2',
            violation_type='check_constraint_violation',
            description='Test CHECK violation',
            detected_at=datetime.now()
        ),
        ConstraintViolation(
            violation_id='business_1',
            constraint_name='test_business',
            table_name='test_table',
            record_id='3',
            violation_type='business_rule_violation',
            description='Test business violation',
            detected_at=datetime.now()
        )
    ]
    
    # 执行测试
    summary = constraint_manager._generate_constraint_summary(violations)
    
    # 验证结果
    assert summary['total_violations'] == 3
    assert summary['foreign_key_violations'] == 1
    assert summary['check_constraint_violations'] == 1
    assert summary['business_rule_violations'] == 1


@pytest.mark.asyncio
async def test_get_constraint_violations_by_table(constraint_manager):
    """测试按表名筛选约束违反"""
    # 创建测试报告
    violations = [
        ConstraintViolation(
            violation_id='1',
            constraint_name='test1',
            table_name='table1',
            record_id='1',
            violation_type='foreign_key_violation',
            description='Test 1',
            detected_at=datetime.now()
        ),
        ConstraintViolation(
            violation_id='2',
            constraint_name='test2',
            table_name='table2',
            record_id='2',
            violation_type='check_constraint_violation',
            description='Test 2',
            detected_at=datetime.now()
        ),
        ConstraintViolation(
            violation_id='3',
            constraint_name='test3',
            table_name='table1',
            record_id='3',
            violation_type='business_rule_violation',
            description='Test 3',
            detected_at=datetime.now()
        )
    ]
    
    report = ConstraintReport(
        check_id='test_check',
        check_timestamp=datetime.now(),
        constraints_checked=[],
        violations_found=violations,
        check_duration_ms=100.0,
        summary={}
    )
    
    # 执行测试
    table1_violations = constraint_manager.get_constraint_violations_by_table(report, 'table1')
    
    # 验证结果
    assert len(table1_violations) == 2
    assert all(v.table_name == 'table1' for v in table1_violations)


@pytest.mark.asyncio
async def test_database_error_handling(constraint_manager, mock_db_manager):
    """测试数据库错误处理"""
    # 模拟数据库错误
    mock_db_manager.execute_query.side_effect = Exception("Database connection failed")
    
    # 执行测试并验证异常
    with pytest.raises(Exception) as exc_info:
        await constraint_manager._get_all_constraints()
    
    assert "Database connection failed" in str(exc_info.value)