#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据一致性检查器测试

测试DataConsistencyChecker的各项功能，包括：
1. Token数据一致性检查
2. 成本数据一致性检查
3. 追踪数据完整性检查
4. 外键完整性检查
5. 数据范围检查
6. 性能异常检测
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from harborai.core.consistency.data_consistency_checker import (
    DataConsistencyChecker,
    ConsistencyIssue,
    ConsistencyReport,
    IssueType,
    IssueSeverity
)


@pytest.fixture
def mock_db_manager():
    """模拟数据库管理器"""
    mock = AsyncMock()
    return mock


@pytest.fixture
def consistency_checker(mock_db_manager):
    """创建数据一致性检查器实例"""
    return DataConsistencyChecker(mock_db_manager)


@pytest.mark.asyncio
async def test_check_token_consistency_success(consistency_checker, mock_db_manager):
    """测试Token一致性检查 - 成功场景"""
    # 模拟数据库返回的记录
    mock_records = [
        {
            'log_id': 1,
            'api_prompt_tokens': 100,
            'api_completion_tokens': 50,
            'api_total_tokens': 150,
            'token_prompt_tokens': 100,
            'token_completion_tokens': 50,
            'token_total_tokens': 150,
            'confidence': 0.95,
            'parsing_method': 'api',
            'token_id': 1
        }
    ]
    
    mock_db_manager.execute_query.return_value = mock_records
    
    issues, count = await consistency_checker.check_token_consistency(days_back=7, batch_size=1000)
    
    assert count == 1
    assert len(issues) == 0  # 没有一致性问题
    mock_db_manager.execute_query.assert_called_once()


@pytest.mark.asyncio
async def test_check_token_consistency_mismatch(consistency_checker, mock_db_manager):
    """测试Token一致性检查 - 数据不匹配"""
    # 模拟数据库返回的记录 - 包含不匹配数据
    mock_records = [
        {
            'log_id': 1,
            'api_prompt_tokens': 100,
            'api_completion_tokens': 50,
            'api_total_tokens': 150,
            'token_prompt_tokens': 90,  # 不匹配
            'token_completion_tokens': 45,  # 不匹配
            'token_total_tokens': 140,  # 不匹配
            'confidence': 1.5,  # 超出范围
            'parsing_method': 'api',
            'token_id': 1
        }
    ]
    
    mock_db_manager.execute_query.return_value = mock_records
    
    issues, count = await consistency_checker.check_token_consistency(days_back=7, batch_size=1000)
    
    assert count == 1
    assert len(issues) == 5  # 3个token不匹配 + 1个token总数计算错误 + 1个confidence超范围
    
    # 检查问题类型
    token_issues = [issue for issue in issues if issue.issue_type == IssueType.TOKEN_MISMATCH]
    range_issues = [issue for issue in issues if issue.issue_type == IssueType.INVALID_DATA_RANGE]
    
    assert len(token_issues) == 4  # 3个字段不匹配 + 1个总数计算错误
    assert len(range_issues) == 1
    assert all(issue.auto_fixable for issue in issues)


@pytest.mark.asyncio
async def test_check_token_consistency_missing_record(consistency_checker, mock_db_manager):
    """测试Token一致性检查 - 缺失token_usage记录"""
    # 模拟数据库返回的记录 - 缺失token_usage记录
    mock_records = [
        {
            'log_id': 1,
            'api_prompt_tokens': 100,
            'api_completion_tokens': 50,
            'api_total_tokens': 150,
            'token_prompt_tokens': None,
            'token_completion_tokens': None,
            'token_total_tokens': None,
            'confidence': None,
            'parsing_method': None,
            'token_id': None  # 缺失记录
        }
    ]
    
    mock_db_manager.execute_query.return_value = mock_records
    
    issues, count = await consistency_checker.check_token_consistency(days_back=7, batch_size=1000)
    
    assert count == 1
    assert len(issues) == 1
    assert issues[0].issue_type == IssueType.MISSING_TRACING
    assert not issues[0].auto_fixable


@pytest.mark.asyncio
async def test_check_cost_consistency_success(consistency_checker, mock_db_manager):
    """测试成本一致性检查 - 成功场景"""
    mock_records = [
        {
            'log_id': 1,
            'api_total_cost': 0.001500,
            'cost_prompt_cost': 0.001000,
            'cost_completion_cost': 0.000500,
            'cost_total_cost': 0.001500,
            'currency': 'USD',
            'pricing_source': 'api',
            'cost_id': 1
        }
    ]
    
    mock_db_manager.execute_query.return_value = mock_records
    
    issues, count = await consistency_checker.check_cost_consistency(days_back=7, batch_size=1000)
    
    assert count == 1
    assert len(issues) == 0


@pytest.mark.asyncio
async def test_check_cost_consistency_mismatch(consistency_checker, mock_db_manager):
    """测试成本一致性检查 - 数据不匹配"""
    mock_records = [
        {
            'log_id': 1,
            'api_total_cost': 0.001500,
            'cost_prompt_cost': 0.001000,
            'cost_completion_cost': 0.000500,
            'cost_total_cost': 0.002000,  # 与API不匹配
            'currency': 'USD',
            'pricing_source': 'api',
            'cost_id': 1
        }
    ]
    
    mock_db_manager.execute_query.return_value = mock_records
    
    issues, count = await consistency_checker.check_cost_consistency(days_back=7, batch_size=1000)
    
    assert count == 1
    assert len(issues) == 2  # API不匹配 + 总额计算错误
    
    cost_issues = [issue for issue in issues if issue.issue_type == IssueType.COST_MISMATCH]
    assert len(cost_issues) == 2
    assert all(issue.auto_fixable for issue in issues)


@pytest.mark.asyncio
async def test_check_tracing_completeness(consistency_checker, mock_db_manager):
    """测试追踪数据完整性检查"""
    mock_records = [
        {'log_id': 1, 'trace_id': 'trace-123'},
        {'log_id': 2, 'trace_id': 'trace-456'}
    ]
    
    mock_db_manager.execute_query.return_value = mock_records
    
    issues, count = await consistency_checker.check_tracing_completeness(days_back=7, batch_size=1000)
    
    assert count == 2
    assert len(issues) == 2
    assert all(issue.issue_type == IssueType.MISSING_TRACING for issue in issues)
    assert all(not issue.auto_fixable for issue in issues)


@pytest.mark.asyncio
async def test_check_foreign_key_integrity(consistency_checker, mock_db_manager):
    """测试外键完整性检查"""
    # 模拟三次查询的返回值
    mock_db_manager.execute_query.side_effect = [
        [{'id': 1, 'log_id': 999}],  # 孤立的token_usage
        [{'id': 2, 'log_id': 998}],  # 孤立的cost_info
        [{'id': 3, 'log_id': 997}]   # 孤立的tracing_info
    ]
    
    issues, count = await consistency_checker.check_foreign_key_integrity(days_back=7, batch_size=1000)
    
    assert count == 3
    assert len(issues) == 3
    assert all(issue.issue_type == IssueType.ORPHANED_RECORD for issue in issues)
    assert all(not issue.auto_fixable for issue in issues)


@pytest.mark.asyncio
async def test_check_data_ranges(consistency_checker, mock_db_manager):
    """测试数据范围检查"""
    # 模拟两次查询的返回值
    mock_db_manager.execute_query.side_effect = [
        [{'id': 1, 'duration_ms': -100, 'provider': 'openai', 'model': 'gpt-4'}],  # 异常响应时间
        [{'id': 2, 'prompt_tokens': 150000, 'completion_tokens': 80000, 'total_tokens': 230000}]  # 异常token数量
    ]
    
    issues, count = await consistency_checker.check_data_ranges(days_back=7, batch_size=1000)
    
    assert count == 2
    assert len(issues) == 2
    assert all(issue.issue_type == IssueType.INVALID_DATA_RANGE for issue in issues)
    assert all(not issue.auto_fixable for issue in issues)


@pytest.mark.asyncio
async def test_check_performance_anomalies(consistency_checker, mock_db_manager):
    """测试性能异常检测"""
    # 模拟统计查询和异常请求查询
    mock_db_manager.execute_query.side_effect = [
        [{'provider': 'openai', 'model': 'gpt-4', 'avg_duration': 1000.0, 'stddev_duration': 200.0, 'request_count': 100}],
        [{'id': 1, 'duration_ms': 2000.0, 'created_at': datetime.now()}]  # 异常慢的请求
    ]
    
    issues, count = await consistency_checker.check_performance_anomalies(days_back=7, batch_size=1000)
    
    assert count == 1
    assert len(issues) == 1
    assert issues[0].issue_type == IssueType.PERFORMANCE_ANOMALY
    assert not issues[0].auto_fixable


@pytest.mark.asyncio
async def test_generate_report_success(consistency_checker, mock_db_manager):
    """测试生成完整报告 - 成功场景"""
    # 模拟所有检查都返回空结果
    mock_db_manager.execute_query.return_value = []
    
    report = await consistency_checker.generate_report(days_back=7, batch_size=1000)
    
    assert isinstance(report, ConsistencyReport)
    assert report.total_issues == 0
    assert report.check_duration_ms >= 0
    assert len(report.summary) > 0
    assert 'total' in report.summary


@pytest.mark.asyncio
async def test_generate_report_with_issues(consistency_checker, mock_db_manager):
    """测试生成完整报告 - 包含问题"""
    # 模拟token检查返回问题
    mock_records = [
        {
            'log_id': 1,
            'api_prompt_tokens': 100,
            'api_completion_tokens': 50,
            'api_total_tokens': 150,
            'token_prompt_tokens': 90,  # 不匹配 - 问题1
            'token_completion_tokens': 50,  # 匹配
            'token_total_tokens': 140,  # 不匹配 - 问题2
            'confidence': 0.95,
            'parsing_method': 'api',
            'token_id': 1
        }
    ]
    
    # 设置所有查询返回值
    mock_db_manager.execute_query.side_effect = [
        mock_records,  # token检查
        [],  # cost检查
        [],  # tracing检查
        [], [], [],  # foreign key检查 (3次查询)
        [], [],  # data ranges检查 (2次查询)
        [], []  # performance检查 (2次查询)
    ]
    
    report = await consistency_checker.generate_report(days_back=7, batch_size=1000, include_performance_check=True)
    
    assert report.total_issues == 2  # prompt_tokens不匹配 + total_tokens不匹配
    assert len(report.issues) == 2
    assert all(issue.issue_type == IssueType.TOKEN_MISMATCH for issue in report.issues)
    assert len(report.recommendations) > 0


@pytest.mark.asyncio
async def test_get_issues_by_severity(consistency_checker):
    """测试按严重程度筛选问题"""
    issues = [
        ConsistencyIssue(
            issue_id="test1",
            table_name="test_table",
            record_id="1",
            issue_type=IssueType.TOKEN_MISMATCH,
            description="Test issue",
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True
        ),
        ConsistencyIssue(
            issue_id="test2",
            table_name="test_table",
            record_id="2",
            issue_type=IssueType.COST_MISMATCH,
            description="Test issue",
            severity=IssueSeverity.LOW,
            detected_at=datetime.now(),
            auto_fixable=True
        )
    ]
    
    report = ConsistencyReport(
        check_id="test",
        check_timestamp=datetime.now(),
        total_records_checked=2,
        issues=issues,
        check_duration_ms=100.0,
        summary={}
    )
    
    high_issues = await consistency_checker.get_issues_by_severity(report, IssueSeverity.HIGH)
    assert len(high_issues) == 1
    assert high_issues[0].severity == IssueSeverity.HIGH


@pytest.mark.asyncio
async def test_get_issues_by_type(consistency_checker):
    """测试按问题类型筛选问题"""
    issues = [
        ConsistencyIssue(
            issue_id="test1",
            table_name="test_table",
            record_id="1",
            issue_type=IssueType.TOKEN_MISMATCH,
            description="Test issue",
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True
        ),
        ConsistencyIssue(
            issue_id="test2",
            table_name="test_table",
            record_id="2",
            issue_type=IssueType.COST_MISMATCH,
            description="Test issue",
            severity=IssueSeverity.LOW,
            detected_at=datetime.now(),
            auto_fixable=True
        )
    ]
    
    report = ConsistencyReport(
        check_id="test",
        check_timestamp=datetime.now(),
        total_records_checked=2,
        issues=issues,
        check_duration_ms=100.0,
        summary={}
    )
    
    token_issues = await consistency_checker.get_issues_by_type(report, IssueType.TOKEN_MISMATCH)
    assert len(token_issues) == 1
    assert token_issues[0].issue_type == IssueType.TOKEN_MISMATCH


@pytest.mark.asyncio
async def test_export_report_json(consistency_checker):
    """测试导出JSON格式报告"""
    issues = [
        ConsistencyIssue(
            issue_id="test1",
            table_name="test_table",
            record_id="1",
            issue_type=IssueType.TOKEN_MISMATCH,
            description="Test issue",
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True
        )
    ]
    
    report = ConsistencyReport(
        check_id="test",
        check_timestamp=datetime.now(),
        total_records_checked=1,
        issues=issues,
        check_duration_ms=100.0,
        summary={'total': 1}
    )
    
    json_output = await consistency_checker.export_report(report, format="json")
    assert isinstance(json_output, str)
    assert "test1" in json_output
    assert "token_mismatch" in json_output


@pytest.mark.asyncio
async def test_export_report_csv(consistency_checker):
    """测试导出CSV格式报告"""
    issues = [
        ConsistencyIssue(
            issue_id="test1",
            table_name="test_table",
            record_id="1",
            issue_type=IssueType.TOKEN_MISMATCH,
            description="Test issue",
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True
        )
    ]
    
    report = ConsistencyReport(
        check_id="test",
        check_timestamp=datetime.now(),
        total_records_checked=1,
        issues=issues,
        check_duration_ms=100.0,
        summary={'total': 1}
    )
    
    csv_output = await consistency_checker.export_report(report, format="csv")
    assert isinstance(csv_output, str)
    assert "test1" in csv_output
    assert "token_mismatch" in csv_output
    assert "问题ID,表名,记录ID" in csv_output


@pytest.mark.asyncio
async def test_export_report_invalid_format(consistency_checker):
    """测试导出无效格式报告"""
    report = ConsistencyReport(
        check_id="test",
        check_timestamp=datetime.now(),
        total_records_checked=0,
        issues=[],
        check_duration_ms=100.0,
        summary={}
    )
    
    with pytest.raises(ValueError, match="不支持的导出格式"):
        await consistency_checker.export_report(report, format="xml")


@pytest.mark.asyncio
async def test_database_error_handling(consistency_checker, mock_db_manager):
    """测试数据库错误处理"""
    mock_db_manager.execute_query.side_effect = Exception("Database connection failed")
    
    with pytest.raises(Exception, match="Database connection failed"):
        await consistency_checker.check_token_consistency(days_back=7, batch_size=1000)


def test_consistency_issue_to_dict():
    """测试ConsistencyIssue转换为字典"""
    issue = ConsistencyIssue(
        issue_id="test1",
        table_name="test_table",
        record_id="1",
        issue_type=IssueType.TOKEN_MISMATCH,
        description="Test issue",
        severity=IssueSeverity.HIGH,
        detected_at=datetime(2024, 1, 1, 12, 0, 0),
        auto_fixable=True,
        fix_suggestion="Fix suggestion",
        affected_fields=["field1", "field2"],
        metadata={"key": "value"}
    )
    
    result = issue.to_dict()
    
    assert result["issue_id"] == "test1"
    assert result["issue_type"] == "token_mismatch"
    assert result["severity"] == "high"
    assert result["auto_fixable"] is True
    assert result["affected_fields"] == ["field1", "field2"]
    assert result["metadata"] == {"key": "value"}


def test_consistency_report_properties():
    """测试ConsistencyReport属性"""
    issues = [
        ConsistencyIssue(
            issue_id="test1",
            table_name="test_table",
            record_id="1",
            issue_type=IssueType.TOKEN_MISMATCH,
            description="Test issue",
            severity=IssueSeverity.CRITICAL,
            detected_at=datetime.now(),
            auto_fixable=True
        ),
        ConsistencyIssue(
            issue_id="test2",
            table_name="test_table",
            record_id="2",
            issue_type=IssueType.COST_MISMATCH,
            description="Test issue",
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=False
        )
    ]
    
    report = ConsistencyReport(
        check_id="test",
        check_timestamp=datetime.now(),
        total_records_checked=2,
        issues=issues,
        check_duration_ms=100.0,
        summary={'total': 2}
    )
    
    assert report.total_issues == 2
    assert len(report.critical_issues) == 1
    assert len(report.auto_fixable_issues) == 1
    assert report.critical_issues[0].severity == IssueSeverity.CRITICAL
    assert report.auto_fixable_issues[0].auto_fixable is True