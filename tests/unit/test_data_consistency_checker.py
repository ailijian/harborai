#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据一致性检查器单元测试

测试DataConsistencyChecker的各项功能，包括：
- token数据一致性验证
- 成本数据一致性检查
- 追踪数据完整性验证
- 跨表关联一致性检查
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from harborai.core.consistency.data_consistency_checker import (
    DataConsistencyChecker,
    ConsistencyIssue,
    ConsistencyReport,
    IssueType,
    IssueSeverity
)


class TestDataConsistencyChecker:
    """数据一致性检查器测试类"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """模拟数据库管理器"""
        db_manager = AsyncMock()
        return db_manager
    
    @pytest.fixture
    def consistency_checker(self, mock_db_manager):
        """创建数据一致性检查器实例"""
        return DataConsistencyChecker(mock_db_manager)
    
    @pytest.mark.asyncio
    async def test_check_token_consistency_success(self, consistency_checker, mock_db_manager):
        """测试token数据一致性检查 - 成功场景"""
        # 准备测试数据
        mock_db_manager.execute_query.return_value = [
            {
                'log_id': 1,
                'token_id': 1,  # 添加token_id字段
                'api_prompt_tokens': 100,
                'api_completion_tokens': 50,
                'api_total_tokens': 150,
                'token_prompt_tokens': 100,
                'token_completion_tokens': 50,
                'token_total_tokens': 150,
                'confidence': 0.95  # 添加confidence字段
            }
        ]
        
        # 执行检查
        issues, total_count = await consistency_checker.check_token_consistency()
        
        # 验证结果
        assert len(issues) == 0
        assert total_count == 1
        mock_db_manager.execute_query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_token_consistency_mismatch(self, consistency_checker, mock_db_manager):
        """测试token数据一致性检查 - 不匹配场景"""
        # 准备测试数据 - token数量不匹配
        mock_db_manager.execute_query.return_value = [
            {
                'log_id': 1,
                'token_id': 1,  # 添加token_id字段
                'api_prompt_tokens': 100,
                'api_completion_tokens': 50,
                'api_total_tokens': 150,
                'token_prompt_tokens': 90,  # 不匹配
                'token_completion_tokens': 45,  # 不匹配
                'token_total_tokens': 135,  # 不匹配
                'confidence': 0.95  # 添加confidence字段
            }
        ]
        
        # 执行检查
        issues, total_count = await consistency_checker.check_token_consistency()
        
        # 验证结果
        assert len(issues) == 3  # prompt, completion, total tokens都不匹配
        assert total_count == 1
        assert all(issue.issue_type == IssueType.TOKEN_MISMATCH for issue in issues)
        assert all(issue.severity == IssueSeverity.HIGH for issue in issues)
    
    @pytest.mark.asyncio
    async def test_check_cost_consistency_success(self, consistency_checker, mock_db_manager):
        """测试成本数据一致性检查 - 成功场景"""
        # 准备测试数据
        mock_db_manager.execute_query.return_value = [
            {
                'log_id': 1,
                'cost_id': 1,  # 添加cost_id字段
                'api_total_cost': 0.05,
                'cost_total_cost': 0.05,
                'cost_prompt_cost': 0.03,
                'cost_completion_cost': 0.02
            }
        ]
        
        # 执行检查
        issues, total_count = await consistency_checker.check_cost_consistency()
        
        # 验证结果
        assert len(issues) == 0
        assert total_count == 1
    
    @pytest.mark.asyncio
    async def test_check_cost_consistency_mismatch(self, consistency_checker, mock_db_manager):
        """测试成本数据一致性检查 - 不匹配场景"""
        # 准备测试数据 - 成本不匹配
        mock_db_manager.execute_query.return_value = [
            {
                'log_id': 1,
                'cost_id': 1,  # 添加cost_id字段
                'api_total_cost': 0.05,
                'cost_total_cost': 0.04,  # 不匹配
                'cost_prompt_cost': 0.03,
                'cost_completion_cost': 0.02
            }
        ]
        
        # 执行检查
        issues, total_count = await consistency_checker.check_cost_consistency()
        
        # 验证结果
        assert len(issues) == 2  # API不匹配 + 总额计算错误
        assert total_count == 1
        assert all(issue.issue_type == IssueType.COST_MISMATCH for issue in issues)
        # 第一个问题是API不匹配（HIGH），第二个是总额计算错误（MEDIUM）
        assert any(issue.severity == IssueSeverity.HIGH for issue in issues)
        assert any(issue.severity == IssueSeverity.MEDIUM for issue in issues)
    
    @pytest.mark.asyncio
    async def test_check_tracing_completeness_success(self, consistency_checker, mock_db_manager):
        """测试追踪数据完整性检查 - 成功场景"""
        # 准备测试数据
        mock_db_manager.execute_query.return_value = []  # 没有缺失的追踪数据
        
        # 执行检查
        issues, total_count = await consistency_checker.check_tracing_completeness()
        
        # 验证结果
        assert len(issues) == 0
        assert total_count == 0
    
    @pytest.mark.asyncio
    async def test_check_tracing_completeness_missing(self, consistency_checker, mock_db_manager):
        """测试追踪数据完整性检查 - 缺失场景"""
        # 准备测试数据 - 有缺失的追踪数据
        mock_db_manager.execute_query.return_value = [
            {'log_id': 1, 'trace_id': 'trace-123'},
            {'log_id': 2, 'trace_id': 'trace-456'}
        ]
        
        # 执行检查
        issues, total_count = await consistency_checker.check_tracing_completeness()
        
        # 验证结果
        assert len(issues) == 2
        assert total_count == 2
        assert all(issue.issue_type == IssueType.MISSING_TRACING for issue in issues)
        assert all(issue.severity == IssueSeverity.MEDIUM for issue in issues)
    
    @pytest.mark.asyncio
    async def test_check_foreign_key_integrity_success(self, consistency_checker, mock_db_manager):
        """测试外键完整性检查 - 成功场景"""
        # 准备测试数据
        mock_db_manager.execute_query.return_value = []  # 没有孤立记录
        
        # 执行检查
        issues, total_count = await consistency_checker.check_foreign_key_integrity()
        
        # 验证结果
        assert len(issues) == 0
        assert total_count == 0
    
    @pytest.mark.asyncio
    async def test_check_foreign_key_integrity_orphaned(self, consistency_checker, mock_db_manager):
        """测试外键完整性检查 - 孤立记录场景"""
        # 准备测试数据 - 有孤立记录
        mock_db_manager.execute_query.side_effect = [
            [{'id': 1, 'log_id': 999}],  # token_usage中的孤立记录
            [{'id': 2, 'log_id': 888}],  # cost_info中的孤立记录
            [{'id': 3, 'log_id': 777}]   # tracing_info中的孤立记录
        ]
        
        # 执行检查
        issues, total_count = await consistency_checker.check_foreign_key_integrity()
        
        # 验证结果
        assert len(issues) == 3
        assert total_count == 3
        assert all(issue.issue_type == IssueType.ORPHANED_RECORD for issue in issues)
        assert all(issue.severity == IssueSeverity.HIGH for issue in issues)
    
    @pytest.mark.asyncio
    async def test_check_data_ranges_success(self, consistency_checker, mock_db_manager):
        """测试数据范围检查 - 成功场景"""
        # 准备测试数据
        mock_db_manager.execute_query.return_value = []  # 没有异常数据
        
        # 执行检查
        issues, total_count = await consistency_checker.check_data_ranges()
        
        # 验证结果
        assert len(issues) == 0
        assert total_count == 0
    
    @pytest.mark.asyncio
    async def test_check_data_ranges_invalid(self, consistency_checker, mock_db_manager):
        """测试数据范围检查 - 无效数据场景"""
        # 准备测试数据 - 有无效数据
        mock_db_manager.execute_query.side_effect = [
            [{'id': 1, 'duration_ms': -100, 'provider': 'openai', 'model': 'gpt-4'}],  # 负数持续时间
            [{'id': 2, 'prompt_tokens': 150000, 'completion_tokens': 150000, 'total_tokens': 300000}]  # 异常token数量
        ]
        
        # 执行检查
        issues, total_count = await consistency_checker.check_data_ranges()
        
        # 验证结果
        assert len(issues) == 2
        assert total_count == 2
        assert all(issue.issue_type == IssueType.INVALID_DATA_RANGE for issue in issues)
        # 第一个是MEDIUM（异常持续时间），第二个是LOW（异常token数量）
        assert issues[0].severity == IssueSeverity.MEDIUM
        assert issues[1].severity == IssueSeverity.LOW
    

    @pytest.mark.asyncio
    async def test_generate_report(self, consistency_checker, mock_db_manager):
        """测试生成一致性报告"""
        # 准备测试数据
        issues = [
            ConsistencyIssue(
                issue_id="test_token_issue",
                table_name="api_logs",
                record_id="1",
                issue_type=IssueType.TOKEN_MISMATCH,
                severity=IssueSeverity.HIGH,
                description="Token不匹配",
                detected_at=datetime.now(),
                auto_fixable=False
            ),
            ConsistencyIssue(
                issue_id="test_cost_issue",
                table_name="cost_info",
                record_id="2",
                issue_type=IssueType.COST_MISMATCH,
                severity=IssueSeverity.MEDIUM,
                description="成本不匹配",
                detected_at=datetime.now(),
                auto_fixable=False
            )
        ]
        
        # 模拟各项检查
        with patch.object(consistency_checker, 'check_token_consistency', return_value=([issues[0]], 1)), \
             patch.object(consistency_checker, 'check_cost_consistency', return_value=([issues[1]], 1)), \
             patch.object(consistency_checker, 'check_tracing_completeness', return_value=([], 0)), \
             patch.object(consistency_checker, 'check_foreign_key_integrity', return_value=([], 0)), \
             patch.object(consistency_checker, 'check_data_ranges', return_value=([], 0)):
            
            # 执行报告生成
            report = await consistency_checker.generate_report()
            
            # 验证结果
            assert isinstance(report, ConsistencyReport)
            assert len(report.issues) == 2
            assert report.total_issues == 2
            # 检查严重性分布
            high_issues = [issue for issue in report.issues if issue.severity == IssueSeverity.HIGH]
            medium_issues = [issue for issue in report.issues if issue.severity == IssueSeverity.MEDIUM]
            assert len(high_issues) == 1  # HIGH severity
            assert len(medium_issues) == 1   # MEDIUM severity
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, consistency_checker, mock_db_manager):
        """测试数据库错误处理"""
        # 模拟数据库异常
        mock_db_manager.execute_query.side_effect = Exception("数据库连接失败")
        
        # 执行检查，应该抛出异常
        with pytest.raises(Exception, match="数据库连接失败"):
            await consistency_checker.check_token_consistency()
    
    def test_consistency_issue_creation(self):
        """测试一致性问题对象创建"""
        issue = ConsistencyIssue(
            issue_id="test_issue_1",
            table_name="api_logs",
            record_id="123",
            issue_type=IssueType.TOKEN_MISMATCH,
            severity=IssueSeverity.HIGH,
            description="测试问题",
            detected_at=datetime.now(),
            auto_fixable=False,
            metadata={'key': 'value'}
        )
        
        assert issue.issue_type == IssueType.TOKEN_MISMATCH
        assert issue.severity == IssueSeverity.HIGH
        assert issue.description == "测试问题"
        assert issue.record_id == "123"
        assert issue.metadata == {'key': 'value'}
        assert isinstance(issue.detected_at, datetime)
    
    def test_consistency_report_creation(self):
        """测试一致性报告对象创建"""
        issues = [
            ConsistencyIssue(
                issue_id="test_issue_1",
                table_name="api_logs",
                record_id="1",
                issue_type=IssueType.TOKEN_MISMATCH,
                severity=IssueSeverity.HIGH,
                description="高严重性问题",
                detected_at=datetime.now(),
                auto_fixable=False
            ),
            ConsistencyIssue(
                issue_id="test_issue_2",
                table_name="cost_info",
                record_id="2",
                issue_type=IssueType.COST_MISMATCH,
                severity=IssueSeverity.MEDIUM,
                description="中等严重性问题",
                detected_at=datetime.now(),
                auto_fixable=False
            ),
            ConsistencyIssue(
                issue_id="test_issue_3",
                table_name="tracing_info",
                record_id="3",
                issue_type=IssueType.MISSING_TRACING,
                severity=IssueSeverity.LOW,
                description="低严重性问题",
                detected_at=datetime.now(),
                auto_fixable=False
            )
        ]
        
        report = ConsistencyReport(
            check_id="test_check_1",
            check_timestamp=datetime.now(),
            total_records_checked=100,
            issues=issues,
            check_duration_ms=1500.0,
            summary={"high": 1, "medium": 1, "low": 1}
        )
        
        assert len(report.issues) == 3
        assert report.total_issues == 3
        assert len(report.critical_issues) == 0  # 没有CRITICAL级别的问题
        assert len(report.auto_fixable_issues) == 0  # 没有可自动修复的问题
        assert report.check_duration_ms == 1500.0
        assert isinstance(report.check_timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_concurrent_checks(self, consistency_checker, mock_db_manager):
        """测试并发检查"""
        # 模拟各项检查
        mock_db_manager.execute_query.return_value = []
        
        # 并发执行多个检查
        tasks = [
            consistency_checker.check_token_consistency(),
            consistency_checker.check_cost_consistency(),
            consistency_checker.check_tracing_completeness()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 验证结果
        assert len(results) == 3
        # 每个结果都是 (issues, total_count) 元组
        assert all(isinstance(result, tuple) and len(result) == 2 for result in results)
        assert all(isinstance(result[0], list) for result in results)  # issues 列表
        assert all(isinstance(result[1], int) for result in results)   # total_count 整数
    
    @pytest.mark.asyncio
    async def test_large_dataset_handling(self, consistency_checker, mock_db_manager):
        """测试大数据集处理"""
        # 模拟大量数据
        large_dataset = [
            {
                'log_id': i,
                'token_id': f'token_{i}',
                'api_prompt_tokens': 100,
                'api_completion_tokens': 50,
                'api_total_tokens': 150,
                'token_prompt_tokens': 100,
                'token_completion_tokens': 50,
                'token_total_tokens': 150,
                'confidence': 0.95
            }
            for i in range(1000)
        ]
        
        mock_db_manager.execute_query.return_value = large_dataset
        
        # 执行检查
        issues, total_count = await consistency_checker.check_token_consistency()
        
        # 验证结果
        assert isinstance(issues, list)
        # 由于数据一致，应该没有问题
        assert len(issues) == 0
        assert total_count == 1000