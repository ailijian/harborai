#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动修正服务单元测试

测试AutoCorrectionService的各项功能，包括：
- 缺失数据修正
- 不一致数据修正
- 孤立记录处理
- 数据重新计算
- 干运行模式
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

from harborai.core.consistency.auto_correction_service import (
    AutoCorrectionService,
    CorrectionAction,
    CorrectionResult,
    ActionType,
    CorrectionStatus
)


class TestAutoCorrectionService:
    """自动修正服务测试类"""
    
    @pytest.fixture
    def mock_db_manager(self):
        """创建模拟数据库管理器"""
        mock = AsyncMock()
        # 添加fetch_one方法的mock
        mock.fetch_one = AsyncMock()
        mock.execute = AsyncMock()
        return mock
    
    @pytest.fixture
    def mock_consistency_checker(self):
        """模拟数据一致性检查器"""
        checker = AsyncMock()
        return checker
    
    @pytest.fixture
    def correction_service(self, mock_db_manager, mock_consistency_checker):
        """创建自动修正服务实例"""
        return AutoCorrectionService(mock_db_manager, mock_consistency_checker)
    
    @pytest.mark.asyncio
    async def test_correct_missing_token_data_success(self, correction_service, mock_db_manager):
        """测试修正缺失token数据 - 成功场景"""
        # 准备测试数据
        log_id = 123
        # Mock fetch_one调用
        mock_db_manager.fetch_one.side_effect = [
            # 获取API日志数据
            {
                'id': log_id,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150,
                'model': 'gpt-3.5-turbo',
                'created_at': datetime.now()
            },
            # 获取token使用数据（不存在）
            None
        ]
        # Mock execute调用（插入操作）
        mock_db_manager.execute.return_value = None
        
        # 执行修正
        result = await correction_service.correct_missing_token_data(log_id)
        
        # 验证结果
        assert result.status == CorrectionStatus.SUCCESS
        assert len(result.actions_performed) > 0
        assert result.actions_performed[0].action_type == 'insert'
        assert result.actions_performed[0].table_name == 'token_usage'
        assert result.actions_performed[0].record_id == str(log_id)
        
        # 验证数据库调用
        assert mock_db_manager.fetch_one.call_count >= 1
        assert mock_db_manager.execute.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_correct_missing_token_data_not_found(self, correction_service, mock_db_manager):
        """测试修正缺失token数据 - 源数据不存在场景"""
        # 准备测试数据
        log_id = 999
        mock_db_manager.fetch_one.return_value = None  # 没有找到API日志
        
        # 执行修正
        result = await correction_service.correct_missing_token_data(log_id)
        
        # 验证结果
        assert result.status == CorrectionStatus.FAILED
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_correct_missing_cost_data_success(self, correction_service, mock_db_manager):
        """测试修正缺失成本数据 - 成功场景"""
        # 准备测试数据
        log_id = 123
        # Mock fetch_one调用
        mock_db_manager.fetch_one.side_effect = [
            # 获取API日志数据
            {
                'id': log_id,
                'model': 'gpt-3.5-turbo',
                'total_cost': 0.05,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'created_at': datetime.now()
            },
            # 获取成本数据（不存在）
            None
        ]
        # Mock execute调用（插入操作）
        mock_db_manager.execute.return_value = None
        
        # 执行修正
        result = await correction_service.correct_missing_cost_data(log_id)
        
        # 验证结果
        assert result.status == CorrectionStatus.SUCCESS
        assert len(result.actions_performed) > 0
        assert result.actions_performed[0].action_type == 'insert'
        assert result.actions_performed[0].table_name == 'cost_info'
        assert result.actions_performed[0].record_id == str(log_id)
    
    @pytest.mark.asyncio
    async def test_correct_missing_tracing_data_success(self, correction_service, mock_db_manager):
        """测试修正缺失追踪数据 - 成功场景"""
        # 准备测试数据
        log_id = 123
        # Mock fetch_one调用
        mock_db_manager.fetch_one.side_effect = [
            # 获取API日志数据
            {
                'id': log_id,
                'trace_id': 'trace-123',
                'span_id': 'span-456',
                'parent_span_id': 'parent-789',
                'duration_ms': 1500,
                'created_at': datetime.now()
            },
            # 获取追踪数据（不存在）
            None
        ]
        # Mock execute调用（插入操作）
        mock_db_manager.execute.return_value = None
        
        # 执行修正
        result = await correction_service.correct_missing_tracing_data(log_id)
        
        # 验证结果
        assert result.status == CorrectionStatus.SUCCESS
        assert len(result.actions_performed) > 0
        assert result.actions_performed[0].action_type == 'insert'
        assert result.actions_performed[0].table_name == 'tracing_info'
        assert result.actions_performed[0].record_id == str(log_id)
    
    @pytest.mark.asyncio
    async def test_correct_inconsistent_token_counts_success(self, correction_service, mock_db_manager):
        """测试修正不一致token计数 - 成功场景"""
        # 准备测试数据
        log_id = 123
        # Mock fetch_one调用
        mock_db_manager.fetch_one.side_effect = [
            # 获取API日志数据
            {
                'id': log_id,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150,
                'model': 'gpt-3.5-turbo',
                'created_at': datetime.now()
            },
            # 获取当前token使用数据
            {
                'log_id': log_id,
                'prompt_tokens': 90,  # 不一致的数据
                'completion_tokens': 40,
                'total_tokens': 130
            }
        ]
        # Mock execute调用（更新操作）
        mock_db_manager.execute.return_value = None
        
        # 执行修正
        result = await correction_service.correct_inconsistent_token_counts(log_id)
        
        # 验证结果
        assert result.status == CorrectionStatus.SUCCESS
        assert len(result.actions_performed) > 0
        assert result.actions_performed[0].action_type == 'update'
        assert result.actions_performed[0].table_name == 'token_usage'
        assert result.actions_performed[0].record_id == str(log_id)
    
    @pytest.mark.asyncio
    async def test_correct_inconsistent_costs_success(self, correction_service, mock_db_manager):
        """测试修正不一致成本 - 成功场景"""
        # 准备测试数据
        log_id = 123
        # Mock fetch_one调用
        mock_db_manager.fetch_one.side_effect = [
            # 获取token使用数据
            {
                'log_id': log_id,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150
            },
            # 获取当前成本数据
            {
                'log_id': log_id,
                'input_cost': 0.02,  # 不一致的数据
                'output_cost': 0.02,
                'total_cost': 0.04
            }
        ]
        # Mock execute调用（更新操作）
        mock_db_manager.execute.return_value = None
        
        # 执行修正
        result = await correction_service.correct_inconsistent_costs(log_id)
        
        # 验证结果
        assert result.status == CorrectionStatus.SUCCESS
        assert len(result.actions_performed) > 0
        assert result.actions_performed[0].action_type == 'update'
        assert result.actions_performed[0].table_name == 'cost_info'
        assert result.actions_performed[0].record_id == str(log_id)
    
    @pytest.mark.asyncio
    async def test_remove_orphaned_records_success(self, correction_service, mock_db_manager):
        """测试移除孤立记录 - 成功场景"""
        # 准备测试数据
        table_name = 'token_usage'
        orphaned_ids = [999, 888, 777]
        
        # 模拟删除操作成功
        mock_db_manager.execute.return_value = None
        
        # 执行移除孤立记录
        result = await correction_service.remove_orphaned_records(table_name, orphaned_ids)
        
        # 验证结果
        assert result.status == CorrectionStatus.SUCCESS
        assert len(result.actions_performed) > 0
        assert result.actions_performed[0].action_type == 'delete'
        assert result.actions_performed[0].table_name == table_name
    
    @pytest.mark.asyncio
    async def test_recalculate_token_totals_success(self, correction_service, mock_db_manager):
        """测试重新计算token总数 - 成功场景"""
        # 准备测试数据
        log_id = 123
        # Mock fetch_one调用
        mock_db_manager.fetch_one.side_effect = [
            # 获取API日志数据
            {
                'id': log_id,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150,
                'model': 'gpt-3.5-turbo',
                'created_at': datetime.now()
            },
            # 获取当前token使用数据
            {
                'log_id': log_id,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 140  # 不一致的总数
            }
        ]
        # Mock execute调用（更新操作）
        mock_db_manager.execute.return_value = None
        
        # 执行重新计算
        result = await correction_service.recalculate_token_totals([log_id])
        
        # 验证结果
        assert result.status == CorrectionStatus.SUCCESS
        assert len(result.actions_performed) > 0
        assert result.actions_performed[0].action_type == 'update'
        assert result.actions_performed[0].table_name == 'token_usage'
        assert result.actions_performed[0].record_id == str(log_id)
    
    @pytest.mark.asyncio
    async def test_recalculate_cost_totals_success(self, correction_service, mock_db_manager):
        """测试重新计算成本总数 - 成功场景"""
        # 准备测试数据
        log_id = 123
        # Mock fetch_one调用
        mock_db_manager.fetch_one.side_effect = [
            # 获取token使用数据
            {
                'log_id': log_id,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150
            },
            # 获取当前成本数据
            {
                'log_id': log_id,
                'input_cost': 0.03,
                'output_cost': 0.02,
                'total_cost': 0.04  # 不一致的总成本
            }
        ]
        # Mock execute调用（更新操作）
        mock_db_manager.execute.return_value = None
        
        # 执行重新计算
        result = await correction_service.recalculate_cost_totals([log_id])
        
        # 验证结果
        assert result.status == CorrectionStatus.SUCCESS
        assert len(result.actions_performed) > 0
        assert result.actions_performed[0].action_type == 'update'
        assert result.actions_performed[0].table_name == 'cost_info'
        assert result.actions_performed[0].record_id == str(log_id)
    
    @pytest.mark.asyncio
    async def test_dry_run_mode(self, correction_service, mock_db_manager):
        """测试干运行模式"""
        # 启用干运行模式
        correction_service.dry_run_mode = True
        
        # 准备测试数据
        log_id = 123
        # Mock fetch_one调用
        mock_db_manager.fetch_one.side_effect = [
            # 获取API日志数据
            {
                'id': log_id,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150,
                'model': 'gpt-3.5-turbo',
                'created_at': datetime.now()
            },
            # 获取token使用数据（不存在）
            None
        ]
        
        # 执行修正（干运行）
        result = await correction_service.correct_missing_token_data(log_id, dry_run=True)
        
        # 验证结果
        assert result.status == CorrectionStatus.SUCCESS
        assert len(result.actions_performed) > 0
        assert result.actions_performed[0].action_type == 'insert'
        
        # 验证没有执行实际的插入操作
        # 只调用了fetch_one（获取数据），没有调用execute
        assert mock_db_manager.fetch_one.call_count >= 1
        # 在干运行模式下，不应该调用execute
        assert mock_db_manager.execute.call_count == 0
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, correction_service, mock_db_manager):
        """测试数据库错误处理"""
        # 模拟数据库异常
        mock_db_manager.fetch_one.side_effect = Exception("数据库连接失败")
        
        # 执行修正
        result = await correction_service.correct_missing_token_data(123)
        
        # 验证结果
        assert result.status == CorrectionStatus.FAILED
        assert len(result.errors) > 0
    
    def test_correction_action_creation(self):
        """测试修正动作对象创建"""
        action = CorrectionAction(
            action_type=ActionType.INSERT.value,
            table_name='test_table',
            record_id=123,
            old_values={},
            new_values={'key': 'value'},
            reason='测试动作',
            confidence=0.9
        )
        
        assert action.action_type == ActionType.INSERT.value
        assert action.table_name == 'test_table'
        assert action.record_id == 123
        assert action.reason == '测试动作'
        assert action.new_values == {'key': 'value'}
        assert action.confidence == 0.9
    
    def test_correction_result_creation(self):
        """测试修正结果对象创建"""
        action = CorrectionAction(
            action_type=ActionType.UPDATE.value,
            table_name='test_table',
            record_id=456,
            old_values={'old_key': 'old_value'},
            new_values={'new_key': 'new_value'},
            reason='测试更新',
            confidence=0.8
        )
        
        result = CorrectionResult(
            success=True,
            status=CorrectionStatus.SUCCESS,
            actions_performed=[action],
            errors=[],
            warnings=[],
            total_records_affected=1,
            execution_time=0.5
        )
        
        assert result.actions_performed[0] == action
        assert result.status == CorrectionStatus.SUCCESS
        assert result.total_records_affected == 1
        assert result.execution_time == 0.5
        assert len(result.errors) == 0
        assert result.success == True
    
    @pytest.mark.asyncio
    async def test_batch_correction_operations(self, correction_service, mock_db_manager):
        """测试批量修正操作"""
        # 准备测试数据
        log_ids = [1, 2, 3]
        
        # 模拟每个ID的数据查询和修正
        mock_db_manager.fetch_one.side_effect = [
            # 第一个ID的数据
            {'id': 1, 'prompt_tokens': 100, 'completion_tokens': 50, 'total_tokens': 150, 'model': 'gpt-3.5-turbo', 'created_at': datetime.now()},
            None,  # 第一个ID的token使用数据（不存在）
            # 第二个ID的数据
            {'id': 2, 'prompt_tokens': 200, 'completion_tokens': 100, 'total_tokens': 300, 'model': 'gpt-3.5-turbo', 'created_at': datetime.now()},
            None,  # 第二个ID的token使用数据（不存在）
            # 第三个ID的数据
            {'id': 3, 'prompt_tokens': 150, 'completion_tokens': 75, 'total_tokens': 225, 'model': 'gpt-3.5-turbo', 'created_at': datetime.now()},
            None   # 第三个ID的token使用数据（不存在）
        ]
        
        # 模拟插入操作成功
        mock_db_manager.execute.return_value = None
        
        # 并发执行修正
        tasks = [
            correction_service.correct_missing_token_data(log_id)
            for log_id in log_ids
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 验证结果
        assert len(results) == 3
        assert all(result.status == CorrectionStatus.SUCCESS for result in results)
        assert all(len(result.actions_performed) > 0 for result in results)
    
    @pytest.mark.asyncio
    async def test_correction_with_validation(self, correction_service, mock_db_manager):
        """测试带验证的修正"""
        # 准备测试数据
        log_id = 123
        
        # 模拟修正前后的验证
        mock_db_manager.fetch_one.side_effect = [
            # 获取API日志数据
            {
                'id': log_id,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150,
                'model': 'gpt-3.5-turbo',
                'created_at': datetime.now()
            },
            # 获取token使用数据（不存在）
            None
        ]
        
        # 模拟插入操作成功
        mock_db_manager.execute.return_value = None
        
        # 模拟验证查询
        mock_db_manager.execute_query.return_value = [{
            'log_id': log_id,
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150
        }]
        
        # 执行修正
        result = await correction_service.correct_missing_token_data(log_id)
        
        # 验证结果
        assert result.status == CorrectionStatus.SUCCESS
        
        # 验证数据是否正确插入（额外的验证查询）
        verification_result = await mock_db_manager.execute_query(
            "SELECT * FROM token_usage WHERE log_id = %s", (log_id,)
        )
        assert len(verification_result) == 1
        assert verification_result[0]['prompt_tokens'] == 100
    
    @pytest.mark.asyncio
    async def test_partial_correction_failure(self, correction_service, mock_db_manager):
        """测试部分修正失败场景"""
        # 准备测试数据
        log_id = 123
        
        # 模拟部分操作失败
        mock_db_manager.fetch_one.side_effect = [
            # 获取API日志数据成功
            {
                'id': log_id,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150,
                'model': 'gpt-3.5-turbo',
                'created_at': datetime.now()
            },
            # 获取token使用数据（不存在）
            None
        ]
        
        # 模拟插入操作失败
        mock_db_manager.execute.side_effect = Exception("插入失败")
        
        # 执行修正
        result = await correction_service.correct_missing_token_data(log_id)
        
        # 验证结果
        assert result.status == CorrectionStatus.FAILED
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_correction_performance_tracking(self, correction_service, mock_db_manager):
        """测试修正性能跟踪"""
        # 准备测试数据
        log_id = 123
        mock_db_manager.fetch_one.side_effect = [
            {
                'id': log_id,
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150,
                'model': 'gpt-3.5-turbo',
                'created_at': datetime.now()
            },
            # 获取token使用数据（不存在）
            None
        ]
        
        # 模拟插入操作成功
        mock_db_manager.execute.return_value = None
        
        # 执行修正
        result = await correction_service.correct_missing_token_data(log_id)
        
        # 验证性能跟踪
        assert result.execution_time is not None
        assert result.execution_time >= 0
        assert isinstance(result.execution_time, float)