"""
自动修正服务测试模块

测试自动修正服务的各种功能：
- Token不匹配修正
- 成本不匹配修正
- 缺失追踪数据修正
- 孤立记录修正
- 数据范围问题修正
- 约束违反修正
- 数据损坏修正
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from typing import Dict, Any

from harborai.core.consistency.auto_correction_service import (
    AutoCorrectionService,
    CorrectionAction,
    CorrectionResult
)
from harborai.core.consistency.data_consistency_checker import (
    ConsistencyIssue,
    IssueType,
    IssueSeverity
)


class TestAutoCorrectionService:
    """自动修正服务测试类"""
    
    @pytest.fixture
    def mock_db_client(self):
        """模拟数据库客户端"""
        mock_client = AsyncMock()
        return mock_client
    
    @pytest.fixture
    def mock_consistency_checker(self):
        """模拟一致性检查器"""
        mock_checker = MagicMock()
        return mock_checker
    
    @pytest.fixture
    def auto_correction_service(self, mock_db_client, mock_consistency_checker):
        """创建自动修正服务实例"""
        return AutoCorrectionService(mock_db_client, mock_consistency_checker)
    
    @pytest.fixture
    def sample_issues(self):
        """示例一致性问题"""
        return [
            ConsistencyIssue(
                issue_id="issue_1",
                issue_type=IssueType.TOKEN_MISMATCH,
                severity=IssueSeverity.HIGH,
                description="Token数量不匹配",
                record_id="1",
                table_name="token_usage",
                detected_at=datetime.now(),
                auto_fixable=True,
                fix_suggestion="重新计算token数量",
                affected_fields=["total_tokens"],
                metadata={"expected_value": 100, "actual_value": 90}
            ),
            ConsistencyIssue(
                issue_id="issue_2",
                issue_type=IssueType.COST_MISMATCH,
                severity=IssueSeverity.MEDIUM,
                description="成本计算不匹配",
                record_id="2",
                table_name="cost_info",
                detected_at=datetime.now(),
                auto_fixable=True,
                fix_suggestion="重新计算成本",
                affected_fields=["total_cost"],
                metadata={"expected_value": 0.05, "actual_value": 0.04}
            ),
            ConsistencyIssue(
                issue_id="issue_3",
                issue_type=IssueType.MISSING_TRACING,
                severity=IssueSeverity.LOW,
                description="缺失追踪数据",
                record_id="3",
                table_name="tracing_info",
                detected_at=datetime.now(),
                auto_fixable=True,
                fix_suggestion="生成追踪数据",
                affected_fields=None,
                metadata={}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_auto_correct_issues_success(self, auto_correction_service, sample_issues):
        """测试自动修正问题成功"""
        # 模拟数据库查询
        auto_correction_service._get_api_log_data = AsyncMock(return_value={
            'id': 1,
            'trace_id': 'test-trace-id',
            'model': 'gpt-3.5-turbo',
            'prompt_tokens': 50,
            'completion_tokens': 50,
            'total_tokens': 100,
            'created_at': datetime.now()
        })
        
        auto_correction_service._get_token_usage_data = AsyncMock(return_value={
            'log_id': 1,
            'prompt_tokens': 45,
            'completion_tokens': 45,
            'total_tokens': 90
        })
        
        auto_correction_service._get_cost_info_data = AsyncMock(return_value={
            'log_id': 2,
            'input_cost': 0.02,
            'output_cost': 0.02,
            'total_cost': 0.04
        })
        
        # 模拟数据库操作
        auto_correction_service._update_token_usage = AsyncMock()
        auto_correction_service._update_cost_info = AsyncMock()
        auto_correction_service._insert_tracing_info = AsyncMock()
        
        # 执行修正
        result = await auto_correction_service.auto_correct_issues(sample_issues)
        
        # 验证结果
        assert result.success is True
        assert len(result.actions_performed) == 3
        assert result.total_records_affected == 3
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_auto_correct_issues_dry_run(self, auto_correction_service, sample_issues):
        """测试试运行模式"""
        # 模拟数据库查询
        auto_correction_service._get_api_log_data = AsyncMock(return_value={
            'id': 1,
            'trace_id': 'test-trace-id',
            'model': 'gpt-3.5-turbo',
            'prompt_tokens': 50,
            'completion_tokens': 50,
            'total_tokens': 100,
            'created_at': datetime.now()
        })
        
        auto_correction_service._get_token_usage_data = AsyncMock(return_value={
            'log_id': 1,
            'prompt_tokens': 45,
            'completion_tokens': 45,
            'total_tokens': 90
        })
        
        # 模拟数据库操作（不应该被调用）
        auto_correction_service._update_token_usage = AsyncMock()
        auto_correction_service._update_cost_info = AsyncMock()
        auto_correction_service._insert_tracing_info = AsyncMock()
        
        # 执行试运行
        result = await auto_correction_service.auto_correct_issues(sample_issues, dry_run=True)
        
        # 验证结果
        assert result.success is True
        assert len(result.actions_performed) > 0
        
        # 验证数据库操作没有被调用
        auto_correction_service._update_token_usage.assert_not_called()
        auto_correction_service._update_cost_info.assert_not_called()
        auto_correction_service._insert_tracing_info.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_correct_token_mismatch(self, auto_correction_service):
        """测试修正Token不匹配问题"""
        issue = ConsistencyIssue(
            issue_id="token_issue_1",
            issue_type=IssueType.TOKEN_MISMATCH,
            severity=IssueSeverity.HIGH,
            description="Token数量不匹配",
            record_id="1",
            table_name="token_usage",
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion="重新计算token数量",
            affected_fields=["total_tokens"],
            metadata={"expected_value": 100, "actual_value": 90}
        )
        
        # 模拟数据库查询
        auto_correction_service._get_api_log_data = AsyncMock(return_value={
            'id': 1,
            'prompt_tokens': 50,
            'completion_tokens': 50,
            'total_tokens': 100
        })
        
        auto_correction_service._get_token_usage_data = AsyncMock(return_value={
            'log_id': 1,
            'prompt_tokens': 45,
            'completion_tokens': 45,
            'total_tokens': 90
        })
        
        auto_correction_service._update_token_usage = AsyncMock()
        
        # 执行修正
        result = await auto_correction_service._correct_token_mismatch([issue])
        
        # 验证结果
        assert result.success is True
        assert len(result.actions_performed) == 1
        assert result.actions_performed[0].action_type == 'update'
        assert result.actions_performed[0].table_name == 'token_usage'
        auto_correction_service._update_token_usage.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_correct_cost_mismatch(self, auto_correction_service):
        """测试修正成本不匹配问题"""
        issue = ConsistencyIssue(
            issue_id="cost_issue_1",
            issue_type=IssueType.COST_MISMATCH,
            severity=IssueSeverity.MEDIUM,
            description="成本计算不匹配",
            record_id="2",
            table_name="cost_info",
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion="重新计算成本",
            affected_fields=["total_cost"],
            metadata={"expected_value": 0.05, "actual_value": 0.04}
        )
        
        # 模拟数据库查询
        auto_correction_service._get_token_usage_data = AsyncMock(return_value={
            'log_id': 2,
            'prompt_tokens': 50,
            'completion_tokens': 50,
            'total_tokens': 100
        })
        
        auto_correction_service._get_cost_info_data = AsyncMock(return_value={
            'log_id': 2,
            'input_cost': 0.02,
            'output_cost': 0.02,
            'total_cost': 0.04
        })
        
        auto_correction_service._update_cost_info = AsyncMock()
        
        # 执行修正
        result = await auto_correction_service._correct_cost_mismatch([issue])
        
        # 验证结果
        assert result.success is True
        assert len(result.actions_performed) == 1
        assert result.actions_performed[0].action_type == 'update'
        assert result.actions_performed[0].table_name == 'cost_info'
        auto_correction_service._update_cost_info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_correct_missing_tracing_data(self, auto_correction_service):
        """测试修正缺失追踪数据"""
        issue = ConsistencyIssue(
            issue_id="tracing_issue_1",
            issue_type=IssueType.MISSING_TRACING,
            severity=IssueSeverity.LOW,
            description="缺失追踪数据",
            record_id="3",
            table_name="tracing_info",
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion="生成追踪数据",
            affected_fields=None,
            metadata={}
        )
        
        # 模拟数据库查询
        auto_correction_service._get_api_log_data = AsyncMock(return_value={
            'id': 3,
            'trace_id': 'test-trace-id',
            'created_at': datetime.now()
        })
        
        auto_correction_service._insert_tracing_info = AsyncMock()
        
        # 执行修正
        result = await auto_correction_service._correct_missing_tracing_data([issue])
        
        # 验证结果
        assert result.success is True
        assert len(result.actions_performed) == 1
        assert result.actions_performed[0].action_type == 'insert'
        assert result.actions_performed[0].table_name == 'tracing_info'
        auto_correction_service._insert_tracing_info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_correct_orphaned_records(self, auto_correction_service):
        """测试修正孤立记录"""
        issue = ConsistencyIssue(
            issue_id="orphaned_issue_1",
            issue_type=IssueType.ORPHANED_RECORD,
            severity=IssueSeverity.MEDIUM,
            description="孤立的token_usage记录",
            record_id="4",
            table_name="token_usage",
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion="删除孤立记录",
            affected_fields=None,
            metadata={}
        )
        
        auto_correction_service._delete_orphaned_token_usage = AsyncMock()
        
        # 执行修正
        result = await auto_correction_service._correct_orphaned_records([issue])
        
        # 验证结果
        assert result.success is True
        assert len(result.actions_performed) == 1
        assert result.actions_performed[0].action_type == 'delete'
        assert result.actions_performed[0].table_name == 'token_usage'
        auto_correction_service._delete_orphaned_token_usage.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_correct_invalid_data_range(self, auto_correction_service):
        """测试修正无效数据范围问题"""
        issue = ConsistencyIssue(
            issue_id="range_issue_1",
            issue_type=IssueType.INVALID_DATA_RANGE,
            severity=IssueSeverity.HIGH,
            description="Token数量超出有效范围",
            record_id="5",
            table_name="token_usage",
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion="修正数据范围",
            affected_fields=["total_tokens"],
            metadata={"expected_value": "0-1000000", "actual_value": -100}
        )
        
        # 模拟数据库查询
        auto_correction_service._get_api_log_data = AsyncMock(return_value={
            'id': 5,
            'prompt_tokens': 50,
            'completion_tokens': 50,
            'total_tokens': 100
        })
        
        auto_correction_service._update_token_usage = AsyncMock()
        
        # 执行修正
        result = await auto_correction_service._correct_invalid_data_range([issue])
        
        # 验证结果
        assert result.success is True
        assert len(result.actions_performed) == 1
        assert result.actions_performed[0].action_type == 'update'
        assert result.actions_performed[0].table_name == 'token_usage'
        auto_correction_service._update_token_usage.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_correct_constraint_violation(self, auto_correction_service):
        """测试修正约束违反问题"""
        issue = ConsistencyIssue(
            issue_id="constraint_issue_1",
            issue_type=IssueType.CONSTRAINT_VIOLATION,
            severity=IssueSeverity.HIGH,
            description="foreign key constraint violation",
            record_id="6",
            table_name="token_usage",
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion="修正约束违反",
            affected_fields=["log_id"],
            metadata={"expected_value": "存在的log_id", "actual_value": 999999}
        )
        
        # 模拟孤立记录修正
        auto_correction_service._correct_orphaned_records = AsyncMock(return_value=CorrectionResult(
            success=True,
            actions_performed=[],
            errors=[],
            warnings=[],
            total_records_affected=1,
            execution_time=0.1
        ))
        
        # 执行修正
        result = await auto_correction_service._correct_constraint_violation([issue])
        
        # 验证结果
        assert result.success is True
        auto_correction_service._correct_orphaned_records.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_correct_data_corruption(self, auto_correction_service):
        """测试修正数据损坏问题"""
        issue = ConsistencyIssue(
            issue_id="corruption_issue_1",
            issue_type=IssueType.DATA_CORRUPTION,
            severity=IssueSeverity.CRITICAL,
            description="数据损坏",
            record_id="7",
            table_name="api_logs",
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion="重建数据",
            affected_fields=None,
            metadata={}
        )
        
        # 模拟数据库查询
        auto_correction_service._get_api_log_data = AsyncMock(return_value={
            'id': 7,
            'prompt_tokens': 50,
            'completion_tokens': 50,
            'total_tokens': 100
        })
        
        auto_correction_service._update_token_usage = AsyncMock()
        auto_correction_service._update_cost_info = AsyncMock()
        
        # 执行修正
        result = await auto_correction_service._correct_data_corruption([issue])
        
        # 验证结果
        assert result.success is True
        assert len(result.actions_performed) == 1
        assert result.actions_performed[0].action_type == 'rebuild'
        auto_correction_service._update_token_usage.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling(self, auto_correction_service):
        """测试错误处理"""
        issue = ConsistencyIssue(
            issue_id="error_issue_1",
            issue_type=IssueType.TOKEN_MISMATCH,
            severity=IssueSeverity.HIGH,
            description="Token数量不匹配",
            record_id="1",
            table_name="token_usage",
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion="重新计算token数量",
            affected_fields=["total_tokens"],
            metadata={"expected_value": 100, "actual_value": 90}
        )
        
        # 模拟数据库错误
        auto_correction_service._get_api_log_data = AsyncMock(side_effect=Exception("数据库连接失败"))
        
        # 执行修正
        result = await auto_correction_service.auto_correct_issues([issue])
        
        # 验证错误处理
        assert result.success is False
        assert len(result.errors) > 0
        assert "数据库连接失败" in str(result.errors)
    
    @pytest.mark.asyncio
    async def test_unsupported_issue_type(self, auto_correction_service):
        """测试不支持的问题类型"""
        issue = ConsistencyIssue(
            issue_id="unsupported_issue_1",
            issue_type=IssueType.PERFORMANCE_ANOMALY,
            severity=IssueSeverity.LOW,
            description="性能异常",
            record_id="8",
            table_name="api_logs",
            detected_at=datetime.now(),
            auto_fixable=False,
            fix_suggestion="需要人工分析",
            affected_fields=None,
            metadata={}
        )
        
        # 执行修正
        result = await auto_correction_service.auto_correct_issues([issue])
        
        # 验证结果
        assert result.success is True
        assert len(result.warnings) > 0
        assert "性能异常" in str(result.warnings)
    
    def test_group_issues_by_type(self, auto_correction_service, sample_issues):
        """测试按问题类型分组"""
        grouped = auto_correction_service._group_issues_by_type(sample_issues)
        
        assert len(grouped) == 3
        assert IssueType.TOKEN_MISMATCH in grouped
        assert IssueType.COST_MISMATCH in grouped
        assert IssueType.MISSING_TRACING in grouped
        assert len(grouped[IssueType.TOKEN_MISMATCH]) == 1
        assert len(grouped[IssueType.COST_MISMATCH]) == 1
        assert len(grouped[IssueType.MISSING_TRACING]) == 1
    
    @pytest.mark.asyncio
    async def test_recalculate_token_usage(self, auto_correction_service):
        """测试重新计算token使用量"""
        log_data = {
            'prompt_tokens': 50,
            'completion_tokens': 50,
            'total_tokens': 100
        }
        
        result = await auto_correction_service._recalculate_token_usage(log_data)
        
        assert result is not None
        assert result['prompt_tokens'] == 50
        assert result['completion_tokens'] == 50
        assert result['total_tokens'] == 100
    
    @pytest.mark.asyncio
    async def test_recalculate_cost(self, auto_correction_service):
        """测试重新计算成本"""
        token_data = {
            'prompt_tokens': 50,
            'completion_tokens': 50,
            'total_tokens': 100
        }
        
        result = await auto_correction_service._recalculate_cost(token_data)
        
        assert result is not None
        assert result['input_cost'] > 0
        assert result['output_cost'] > 0
        assert result['total_cost'] == result['input_cost'] + result['output_cost']
        assert result['currency'] == 'USD'
    
    @pytest.mark.asyncio
    async def test_extract_or_generate_tracing_data(self, auto_correction_service):
        """测试提取或生成追踪数据"""
        log_data = {
            'trace_id': 'test-trace-id',
            'created_at': datetime.now()
        }
        
        result = await auto_correction_service._extract_or_generate_tracing_data(log_data)
        
        assert result is not None
        assert result['hb_trace_id'] == 'test-trace-id'
        assert result['operation_name'] == 'api_call'
        assert result['status'] == 'completed'