#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DataConsistencyChecker 单元测试

测试数据一致性检查器的各项功能，包括token一致性、成本一致性、追踪完整性等
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any

from harborai.core.consistency.data_consistency_checker import (
    DataConsistencyChecker,
    ConsistencyIssue,
    ConsistencyReport,
    IssueType,
    IssueSeverity
)
from harborai.database.async_manager import DatabaseManager


class TestDataConsistencyChecker:
    """DataConsistencyChecker 测试类"""
    
    @pytest.fixture
    async def mock_db_manager(self):
        """模拟数据库管理器"""
        db_manager = AsyncMock(spec=DatabaseManager)
        return db_manager
    
    @pytest.fixture
    async def checker(self, mock_db_manager):
        """创建数据一致性检查器实例"""
        return DataConsistencyChecker(mock_db_manager)
    
    @pytest.mark.asyncio
    async def test_token_consistency_check_success(self, checker, mock_db_manager):
        """测试token一致性检查 - 成功场景"""
        # 模拟数据库查询结果 - 一致的数据
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
                'parsing_method': 'direct_extraction',
                'token_id': 1
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_records
        
        issues, count = await checker.check_token_consistency(days_back=7, batch_size=1000)
        
        assert count == 1
        assert len(issues) == 0  # 没有一致性问题
        
        # 验证数据库查询被正确调用
        mock_db_manager.execute_query.assert_called_once()
        call_args = mock_db_manager.execute_query.call_args
        assert "api_logs al" in call_args[0][0]  # 检查SQL查询
        assert "LEFT JOIN token_usage tu" in call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_token_consistency_check_mismatch(self, checker, mock_db_manager):
        """测试token一致性检查 - 数据不匹配"""
        # 模拟数据库查询结果 - 不一致的数据
        mock_records = [
            {
                'log_id': 1,
                'api_prompt_tokens': 100,
                'api_completion_tokens': 50,
                'api_total_tokens': 150,
                'token_prompt_tokens': 90,  # 不匹配
                'token_completion_tokens': 45,  # 不匹配
                'token_total_tokens': 140,  # 不匹配
                'confidence': 0.95,
                'parsing_method': 'direct_extraction',
                'token_id': 1
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_records
        
        issues, count = await checker.check_token_consistency(days_back=7, batch_size=1000)
        
        assert count == 1
        assert len(issues) == 3  # 3个不匹配问题
        
        # 检查问题类型
        issue_types = [issue.issue_type for issue in issues]
        assert all(issue_type == IssueType.TOKEN_MISMATCH for issue_type in issue_types)
        
        # 检查严重程度
        severities = [issue.severity for issue in issues]
        assert all(severity == IssueSeverity.HIGH for severity in severities)
        
        # 检查是否可自动修复
        assert all(issue.auto_fixable for issue in issues)
    
    @pytest.mark.asyncio
    async def test_token_consistency_check_missing_record(self, checker, mock_db_manager):
        """测试token一致性检查 - 缺失token_usage记录"""
        # 模拟数据库查询结果 - 缺失token_usage记录
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
        
        issues, count = await checker.check_token_consistency(days_back=7, batch_size=1000)
        
        assert count == 1
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.MISSING_TRACING
        assert issues[0].severity == IssueSeverity.MEDIUM
        assert not issues[0].auto_fixable
    
    @pytest.mark.asyncio
    async def test_cost_consistency_check_success(self, checker, mock_db_manager):
        """测试成本一致性检查 - 成功场景"""
        # 模拟数据库查询结果 - 一致的数据
        mock_records = [
            {
                'log_id': 1,
                'api_total_cost': 0.001500,
                'cost_prompt_cost': 0.001000,
                'cost_completion_cost': 0.000500,
                'cost_total_cost': 0.001500,
                'currency': 'USD',
                'pricing_source': 'builtin',
                'cost_id': 1
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_records
        
        issues, count = await checker.check_cost_consistency(days_back=7, batch_size=1000)
        
        assert count == 1
        assert len(issues) == 0  # 没有一致性问题
    
    @pytest.mark.asyncio
    async def test_cost_consistency_check_mismatch(self, checker, mock_db_manager):
        """测试成本一致性检查 - 数据不匹配"""
        # 模拟数据库查询结果 - 不一致的数据
        mock_records = [
            {
                'log_id': 1,
                'api_total_cost': 0.001500,
                'cost_prompt_cost': 0.001000,
                'cost_completion_cost': 0.000500,
                'cost_total_cost': 0.001400,  # 与API不匹配
                'currency': 'USD',
                'pricing_source': 'builtin',
                'cost_id': 1
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_records
        
        issues, count = await checker.check_cost_consistency(days_back=7, batch_size=1000)
        
        assert count == 1
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.COST_MISMATCH
        assert issues[0].severity == IssueSeverity.HIGH
        assert issues[0].auto_fixable
    
    @pytest.mark.asyncio
    async def test_cost_consistency_check_negative_values(self, checker, mock_db_manager):
        """测试成本一致性检查 - 负数值"""
        # 模拟数据库查询结果 - 包含负数
        mock_records = [
            {
                'log_id': 1,
                'api_total_cost': 0.001500,
                'cost_prompt_cost': -0.001000,  # 负数
                'cost_completion_cost': 0.000500,
                'cost_total_cost': 0.001500,
                'currency': 'USD',
                'pricing_source': 'builtin',
                'cost_id': 1
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_records
        
        issues, count = await checker.check_cost_consistency(days_back=7, batch_size=1000)
        
        assert count == 1
        assert len(issues) >= 1
        
        # 检查是否有负数问题
        negative_issues = [issue for issue in issues if "负数" in issue.description]
        assert len(negative_issues) >= 1
        assert negative_issues[0].issue_type == IssueType.INVALID_DATA_RANGE
        assert negative_issues[0].severity == IssueSeverity.HIGH
    
    @pytest.mark.asyncio
    async def test_tracing_completeness_check(self, checker, mock_db_manager):
        """测试追踪完整性检查"""
        # 模拟数据库查询结果 - 缺失tracing_info的记录
        mock_records = [
            {
                'log_id': 1,
                'trace_id': 'trace_123'
            },
            {
                'log_id': 2,
                'trace_id': 'trace_456'
            }
        ]
        
        mock_db_manager.execute_query.return_value = mock_records
        
        issues, count = await checker.check_tracing_completeness(days_back=7, batch_size=1000)
        
        assert count == 2
        assert len(issues) == 2
        
        for issue in issues:
            assert issue.issue_type == IssueType.MISSING_TRACING
            assert issue.severity == IssueSeverity.MEDIUM
            assert not issue.auto_fixable
            assert "trace_id" in issue.metadata
    
    @pytest.mark.asyncio
    async def test_foreign_key_integrity_check(self, checker, mock_db_manager):
        """测试外键完整性检查"""
        # 模拟三次数据库查询的结果
        mock_db_manager.execute_query.side_effect = [
            # 孤立的token_usage记录
            [{'id': 1, 'log_id': 999}],
            # 孤立的cost_info记录
            [{'id': 2, 'log_id': 998}],
            # 孤立的tracing_info记录
            [{'id': 3, 'log_id': 997}]
        ]
        
        issues, count = await checker.check_foreign_key_integrity(days_back=7, batch_size=1000)
        
        assert count == 3
        assert len(issues) == 3
        
        # 检查问题类型
        for issue in issues:
            assert issue.issue_type == IssueType.ORPHANED_RECORD
            assert issue.severity == IssueSeverity.HIGH
            assert not issue.auto_fixable
    
    @pytest.mark.asyncio
    async def test_data_ranges_check(self, checker, mock_db_manager):
        """测试数据范围检查"""
        # 模拟两次数据库查询的结果
        mock_db_manager.execute_query.side_effect = [
            # 异常的响应时间
            [
                {
                    'id': 1,
                    'duration_ms': -100,  # 负数
                    'provider': 'openai',
                    'model': 'gpt-3.5-turbo'
                },
                {
                    'id': 2,
                    'duration_ms': 400000,  # 超过5分钟
                    'provider': 'anthropic',
                    'model': 'claude-3'
                }
            ],
            # 异常的token数量
            [
                {
                    'id': 3,
                    'prompt_tokens': 150000,  # 超过限制
                    'completion_tokens': 50000,
                    'total_tokens': 200000
                }
            ]
        ]
        
        issues, count = await checker.check_data_ranges(days_back=7, batch_size=1000)
        
        assert count == 3
        assert len(issues) == 3
        
        # 检查问题类型
        for issue in issues:
            assert issue.issue_type == IssueType.INVALID_DATA_RANGE
            assert not issue.auto_fixable
    
    @pytest.mark.asyncio
    async def test_performance_anomalies_check(self, checker, mock_db_manager):
        """测试性能异常检查"""
        # 模拟数据库查询结果
        mock_db_manager.execute_query.side_effect = [
            # 平均统计数据
            [
                {
                    'provider': 'openai',
                    'model': 'gpt-3.5-turbo',
                    'avg_duration': 1000.0,
                    'stddev_duration': 200.0,
                    'request_count': 100
                }
            ],
            # 异常慢的请求
            [
                {
                    'id': 1,
                    'duration_ms': 2000.0,  # 超过阈值
                    'created_at': datetime.now()
                }
            ]
        ]
        
        issues, count = await checker.check_performance_anomalies(days_back=7, batch_size=1000)
        
        assert count == 1
        assert len(issues) == 1
        assert issues[0].issue_type == IssueType.PERFORMANCE_ANOMALY
        assert issues[0].severity == IssueSeverity.LOW
        assert not issues[0].auto_fixable
    
    @pytest.mark.asyncio
    async def test_generate_report_comprehensive(self, checker, mock_db_manager):
        """测试生成完整报告"""
        # 模拟各种检查的返回值
        with patch.object(checker, 'check_token_consistency') as mock_token, \
             patch.object(checker, 'check_cost_consistency') as mock_cost, \
             patch.object(checker, 'check_tracing_completeness') as mock_tracing, \
             patch.object(checker, 'check_foreign_key_integrity') as mock_fk, \
             patch.object(checker, 'check_data_ranges') as mock_ranges, \
             patch.object(checker, 'check_performance_anomalies') as mock_perf:
            
            # 设置模拟返回值
            mock_token.return_value = ([
                ConsistencyIssue(
                    issue_id="test_1",
                    table_name="token_usage",
                    record_id="1",
                    issue_type=IssueType.TOKEN_MISMATCH,
                    description="测试问题",
                    severity=IssueSeverity.HIGH,
                    detected_at=datetime.now(),
                    auto_fixable=True
                )
            ], 100)
            
            mock_cost.return_value = ([], 50)
            mock_tracing.return_value = ([], 75)
            mock_fk.return_value = ([], 25)
            mock_ranges.return_value = ([], 30)
            mock_perf.return_value = ([], 20)
            
            # 生成报告
            report = await checker.generate_report(
                days_back=7,
                batch_size=1000,
                include_performance_check=True
            )
            
            # 验证报告内容
            assert isinstance(report, ConsistencyReport)
            assert report.total_records_checked == 300  # 100+50+75+25+30+20
            assert report.total_issues == 1
            assert len(report.issues) == 1
            assert len(report.auto_fixable_issues) == 1
            assert len(report.critical_issues) == 0
            assert report.check_duration_ms > 0
            
            # 验证摘要
            assert 'total' in report.summary
            assert 'high' in report.summary
            assert 'auto_fixable' in report.summary
            
            # 验证建议
            assert len(report.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_export_report_json(self, checker):
        """测试导出JSON格式报告"""
        # 创建测试报告
        issue = ConsistencyIssue(
            issue_id="test_1",
            table_name="token_usage",
            record_id="1",
            issue_type=IssueType.TOKEN_MISMATCH,
            description="测试问题",
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True
        )
        
        report = ConsistencyReport(
            check_id="test_check",
            check_timestamp=datetime.now(),
            total_records_checked=100,
            issues=[issue],
            check_duration_ms=1000.0,
            summary={'total': 1, 'high': 1},
            recommendations=["测试建议"]
        )
        
        # 导出JSON
        json_output = await checker.export_report(report, format="json")
        
        assert isinstance(json_output, str)
        assert "test_check" in json_output
        assert "token_mismatch" in json_output
        assert "测试问题" in json_output
    
    @pytest.mark.asyncio
    async def test_export_report_csv(self, checker):
        """测试导出CSV格式报告"""
        # 创建测试报告
        issue = ConsistencyIssue(
            issue_id="test_1",
            table_name="token_usage",
            record_id="1",
            issue_type=IssueType.TOKEN_MISMATCH,
            description="测试问题",
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True
        )
        
        report = ConsistencyReport(
            check_id="test_check",
            check_timestamp=datetime.now(),
            total_records_checked=100,
            issues=[issue],
            check_duration_ms=1000.0,
            summary={'total': 1, 'high': 1},
            recommendations=["测试建议"]
        )
        
        # 导出CSV
        csv_output = await checker.export_report(report, format="csv")
        
        assert isinstance(csv_output, str)
        assert "问题ID,表名,记录ID" in csv_output
        assert "test_1,token_usage,1" in csv_output
        assert "token_mismatch,high" in csv_output
    
    @pytest.mark.asyncio
    async def test_get_issues_by_severity(self, checker):
        """测试按严重程度筛选问题"""
        # 创建测试报告
        issues = [
            ConsistencyIssue(
                issue_id="high_1",
                table_name="token_usage",
                record_id="1",
                issue_type=IssueType.TOKEN_MISMATCH,
                description="高严重度问题",
                severity=IssueSeverity.HIGH,
                detected_at=datetime.now(),
                auto_fixable=True
            ),
            ConsistencyIssue(
                issue_id="low_1",
                table_name="api_logs",
                record_id="2",
                issue_type=IssueType.PERFORMANCE_ANOMALY,
                description="低严重度问题",
                severity=IssueSeverity.LOW,
                detected_at=datetime.now(),
                auto_fixable=False
            )
        ]
        
        report = ConsistencyReport(
            check_id="test_check",
            check_timestamp=datetime.now(),
            total_records_checked=100,
            issues=issues,
            check_duration_ms=1000.0,
            summary={'total': 2, 'high': 1, 'low': 1},
            recommendations=[]
        )
        
        # 筛选高严重度问题
        high_issues = await checker.get_issues_by_severity(report, IssueSeverity.HIGH)
        assert len(high_issues) == 1
        assert high_issues[0].issue_id == "high_1"
        
        # 筛选低严重度问题
        low_issues = await checker.get_issues_by_severity(report, IssueSeverity.LOW)
        assert len(low_issues) == 1
        assert low_issues[0].issue_id == "low_1"
    
    @pytest.mark.asyncio
    async def test_get_issues_by_type(self, checker):
        """测试按问题类型筛选问题"""
        # 创建测试报告
        issues = [
            ConsistencyIssue(
                issue_id="token_1",
                table_name="token_usage",
                record_id="1",
                issue_type=IssueType.TOKEN_MISMATCH,
                description="Token问题",
                severity=IssueSeverity.HIGH,
                detected_at=datetime.now(),
                auto_fixable=True
            ),
            ConsistencyIssue(
                issue_id="cost_1",
                table_name="cost_info",
                record_id="2",
                issue_type=IssueType.COST_MISMATCH,
                description="成本问题",
                severity=IssueSeverity.MEDIUM,
                detected_at=datetime.now(),
                auto_fixable=True
            )
        ]
        
        report = ConsistencyReport(
            check_id="test_check",
            check_timestamp=datetime.now(),
            total_records_checked=100,
            issues=issues,
            check_duration_ms=1000.0,
            summary={'total': 2, 'high': 1, 'medium': 1},
            recommendations=[]
        )
        
        # 筛选Token问题
        token_issues = await checker.get_issues_by_type(report, IssueType.TOKEN_MISMATCH)
        assert len(token_issues) == 1
        assert token_issues[0].issue_id == "token_1"
        
        # 筛选成本问题
        cost_issues = await checker.get_issues_by_type(report, IssueType.COST_MISMATCH)
        assert len(cost_issues) == 1
        assert cost_issues[0].issue_id == "cost_1"
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, checker, mock_db_manager):
        """测试数据库错误处理"""
        # 模拟数据库错误
        mock_db_manager.execute_query.side_effect = Exception("数据库连接失败")
        
        # 测试各种检查方法的错误处理
        with pytest.raises(Exception, match="数据库连接失败"):
            await checker.check_token_consistency()
        
        with pytest.raises(Exception, match="数据库连接失败"):
            await checker.check_cost_consistency()
        
        with pytest.raises(Exception, match="数据库连接失败"):
            await checker.check_tracing_completeness()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])