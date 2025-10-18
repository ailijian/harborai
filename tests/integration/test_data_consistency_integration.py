#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据一致性系统集成测试

测试DataConsistencyChecker、DatabaseConstraintManager和AutoCorrectionService
的协同工作，包括：
- 端到端的数据一致性检查和修正流程
- 多组件协作场景
- 实际数据库操作测试
- 性能和可靠性测试
"""

import pytest
import asyncio
import asyncpg
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import patch

from harborai.core.consistency import (
    DataConsistencyChecker,
    DatabaseConstraintManager,
    AutoCorrectionService,
    ConsistencyIssue,
    ConstraintViolation,
    CorrectionResult,
    IssueType,
    IssueSeverity,
    ConstraintType,
    ViolationSeverity,
    ActionType,
    CorrectionStatus
)
from harborai.database.async_manager import DatabaseManager


class TestDataConsistencyIntegration:
    """数据一致性系统集成测试类"""
    
    @pytest.fixture
    async def db_manager(self):
        """创建数据库管理器实例"""
        # 使用测试数据库配置
        db_manager = DatabaseManager(
            host='localhost',
            port=5432,
            database='harborai_test',
            user='test_user',
            password='test_password'
        )
        await db_manager.initialize()
        yield db_manager
        await db_manager.close()
    
    @pytest.fixture
    async def consistency_system(self, db_manager):
        """创建完整的数据一致性系统"""
        checker = DataConsistencyChecker(db_manager)
        constraint_manager = DatabaseConstraintManager(db_manager)
        correction_service = AutoCorrectionService(db_manager)
        
        return {
            'checker': checker,
            'constraint_manager': constraint_manager,
            'correction_service': correction_service,
            'db_manager': db_manager
        }
    
    @pytest.fixture
    async def test_data_setup(self, db_manager):
        """设置测试数据"""
        # 清理测试数据
        await self._cleanup_test_data(db_manager)
        
        # 插入测试数据
        test_data = await self._insert_test_data(db_manager)
        
        yield test_data
        
        # 清理测试数据
        await self._cleanup_test_data(db_manager)
    
    async def _cleanup_test_data(self, db_manager):
        """清理测试数据"""
        cleanup_queries = [
            "DELETE FROM tracing_info WHERE log_id >= 9000",
            "DELETE FROM cost_info WHERE log_id >= 9000",
            "DELETE FROM token_usage WHERE log_id >= 9000",
            "DELETE FROM api_logs WHERE id >= 9000"
        ]
        
        for query in cleanup_queries:
            try:
                await db_manager.execute_query(query)
            except Exception:
                pass  # 忽略清理错误
    
    async def _insert_test_data(self, db_manager):
        """插入测试数据"""
        # 插入API日志
        api_logs = []
        for i in range(5):
            log_id = 9000 + i
            api_log = {
                'id': log_id,
                'trace_id': f'trace-{log_id}',
                'span_id': f'span-{log_id}',
                'model': 'gpt-3.5-turbo',
                'prompt_tokens': 100 + i * 10,
                'completion_tokens': 50 + i * 5,
                'total_tokens': 150 + i * 15,
                'total_cost': 0.05 + i * 0.01,
                'duration_ms': 1000 + i * 100,
                'status': 'success',
                'created_at': datetime.now()
            }
            
            await db_manager.execute_query("""
                INSERT INTO api_logs (
                    id, trace_id, span_id, model, prompt_tokens, completion_tokens,
                    total_tokens, total_cost, duration_ms, status, created_at
                ) VALUES (
                    %(id)s, %(trace_id)s, %(span_id)s, %(model)s, %(prompt_tokens)s,
                    %(completion_tokens)s, %(total_tokens)s, %(total_cost)s,
                    %(duration_ms)s, %(status)s, %(created_at)s
                )
            """, api_log)
            
            api_logs.append(api_log)
        
        return {'api_logs': api_logs}
    
    @pytest.mark.asyncio
    async def test_end_to_end_consistency_check_and_correction(self, consistency_system, test_data_setup):
        """测试端到端的一致性检查和修正流程"""
        checker = consistency_system['checker']
        correction_service = consistency_system['correction_service']
        
        # 第一步：检查数据一致性（应该发现缺失的token、cost、tracing数据）
        report = await checker.generate_report()
        
        # 验证发现了问题
        assert report.total_issues > 0
        missing_token_issues = [
            issue for issue in report.issues 
            if issue.issue_type == IssueType.MISSING_TOKEN_DATA
        ]
        missing_cost_issues = [
            issue for issue in report.issues 
            if issue.issue_type == IssueType.MISSING_COST_DATA
        ]
        missing_tracing_issues = [
            issue for issue in report.issues 
            if issue.issue_type == IssueType.MISSING_TRACING
        ]
        
        # 第二步：自动修正缺失数据
        correction_results = []
        
        # 修正缺失的token数据
        for issue in missing_token_issues:
            result = await correction_service.correct_missing_token_data(issue.log_id)
            correction_results.append(result)
        
        # 修正缺失的cost数据
        for issue in missing_cost_issues:
            result = await correction_service.correct_missing_cost_data(issue.log_id)
            correction_results.append(result)
        
        # 修正缺失的tracing数据
        for issue in missing_tracing_issues:
            result = await correction_service.correct_missing_tracing_data(issue.log_id)
            correction_results.append(result)
        
        # 验证修正结果
        successful_corrections = [
            result for result in correction_results 
            if result.status == CorrectionStatus.SUCCESS
        ]
        assert len(successful_corrections) > 0
        
        # 第三步：重新检查数据一致性
        final_report = await checker.generate_report()
        
        # 验证问题已被修正
        assert final_report.total_issues < report.total_issues
    
    @pytest.mark.asyncio
    async def test_constraint_violation_detection_and_repair(self, consistency_system, test_data_setup):
        """测试约束违反检测和修复"""
        constraint_manager = consistency_system['constraint_manager']
        db_manager = consistency_system['db_manager']
        
        # 人为创建约束违反（插入孤立记录）
        await db_manager.execute_query("""
            INSERT INTO token_usage (log_id, prompt_tokens, completion_tokens, total_tokens)
            VALUES (99999, 100, 50, 150)
        """)
        
        # 检查约束违反
        report = await constraint_manager.generate_constraint_report()
        
        # 验证发现了违反
        assert report.total_violations > 0
        fk_violations = [
            violation for violation in report.violations
            if violation.constraint_type == ConstraintType.FOREIGN_KEY
        ]
        assert len(fk_violations) > 0
        
        # 修复约束违反（移除孤立记录）
        await db_manager.execute_query("DELETE FROM token_usage WHERE log_id = 99999")
        
        # 重新检查
        final_report = await constraint_manager.generate_constraint_report()
        
        # 验证违反已被修复
        final_fk_violations = [
            violation for violation in final_report.violations
            if violation.constraint_type == ConstraintType.FOREIGN_KEY
        ]
        assert len(final_fk_violations) < len(fk_violations)
    
    @pytest.mark.asyncio
    async def test_data_inconsistency_correction_workflow(self, consistency_system, test_data_setup):
        """测试数据不一致修正工作流"""
        checker = consistency_system['checker']
        correction_service = consistency_system['correction_service']
        db_manager = consistency_system['db_manager']
        
        # 首先创建完整的数据
        log_id = 9000
        
        # 插入token数据
        await db_manager.execute_query("""
            INSERT INTO token_usage (log_id, prompt_tokens, completion_tokens, total_tokens)
            VALUES (%(log_id)s, 100, 50, 150)
        """, {'log_id': log_id})
        
        # 插入cost数据
        await db_manager.execute_query("""
            INSERT INTO cost_info (log_id, prompt_cost, completion_cost, total_cost)
            VALUES (%(log_id)s, 0.03, 0.02, 0.05)
        """, {'log_id': log_id})
        
        # 人为创建数据不一致（修改token数据使其与API日志不匹配）
        await db_manager.execute_query("""
            UPDATE token_usage 
            SET prompt_tokens = 90, completion_tokens = 45, total_tokens = 135
            WHERE log_id = %(log_id)s
        """, {'log_id': log_id})
        
        # 检查不一致
        issues = await checker.check_token_consistency()
        token_mismatch_issues = [
            issue for issue in issues
            if issue.issue_type == IssueType.TOKEN_MISMATCH and issue.log_id == log_id
        ]
        assert len(token_mismatch_issues) > 0
        
        # 修正不一致
        result = await correction_service.correct_inconsistent_token_counts(log_id)
        assert result.status == CorrectionStatus.SUCCESS
        
        # 重新检查
        final_issues = await checker.check_token_consistency()
        final_token_mismatch_issues = [
            issue for issue in final_issues
            if issue.issue_type == IssueType.TOKEN_MISMATCH and issue.log_id == log_id
        ]
        assert len(final_token_mismatch_issues) == 0
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, consistency_system, test_data_setup):
        """测试并发操作"""
        checker = consistency_system['checker']
        correction_service = consistency_system['correction_service']
        constraint_manager = consistency_system['constraint_manager']
        
        # 并发执行多种检查
        check_tasks = [
            checker.check_token_consistency(),
            checker.check_cost_consistency(),
            checker.check_tracing_completeness(),
            constraint_manager.check_foreign_key_violations(),
            constraint_manager.check_data_integrity_violations()
        ]
        
        check_results = await asyncio.gather(*check_tasks, return_exceptions=True)
        
        # 验证所有检查都成功完成
        assert len(check_results) == 5
        assert all(not isinstance(result, Exception) for result in check_results)
        
        # 并发执行修正操作
        log_ids = [9000, 9001, 9002]
        correction_tasks = [
            correction_service.correct_missing_token_data(log_id)
            for log_id in log_ids
        ]
        
        correction_results = await asyncio.gather(*correction_tasks, return_exceptions=True)
        
        # 验证修正操作
        assert len(correction_results) == 3
        successful_corrections = [
            result for result in correction_results
            if not isinstance(result, Exception) and result.status == CorrectionStatus.SUCCESS
        ]
        assert len(successful_corrections) > 0
    
    @pytest.mark.asyncio
    async def test_dry_run_mode_integration(self, consistency_system, test_data_setup):
        """测试干运行模式集成"""
        correction_service = consistency_system['correction_service']
        db_manager = consistency_system['db_manager']
        
        # 启用干运行模式
        correction_service.dry_run = True
        
        # 记录修正前的数据状态
        before_count = await db_manager.execute_query(
            "SELECT COUNT(*) as count FROM token_usage WHERE log_id >= 9000"
        )
        before_count = before_count[0]['count']
        
        # 执行修正（干运行）
        result = await correction_service.correct_missing_token_data(9000)
        
        # 验证干运行结果
        assert result.status == CorrectionStatus.SUCCESS
        assert "干运行模式" in result.action.description
        
        # 验证数据没有实际改变
        after_count = await db_manager.execute_query(
            "SELECT COUNT(*) as count FROM token_usage WHERE log_id >= 9000"
        )
        after_count = after_count[0]['count']
        
        assert after_count == before_count
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, consistency_system, test_data_setup):
        """测试错误处理和恢复"""
        checker = consistency_system['checker']
        correction_service = consistency_system['correction_service']
        db_manager = consistency_system['db_manager']
        
        # 模拟数据库连接问题
        with patch.object(db_manager, 'execute_query', side_effect=Exception("连接失败")):
            # 检查应该优雅处理错误
            issues = await checker.check_token_consistency()
            assert isinstance(issues, list)  # 应该返回空列表而不是抛出异常
            
            # 修正应该返回失败状态
            result = await correction_service.correct_missing_token_data(9000)
            assert result.status == CorrectionStatus.FAILED
            assert "连接失败" in result.error_message
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, consistency_system, test_data_setup):
        """测试负载下的性能"""
        checker = consistency_system['checker']
        db_manager = consistency_system['db_manager']
        
        # 插入大量测试数据
        batch_size = 100
        for i in range(batch_size):
            log_id = 10000 + i
            await db_manager.execute_query("""
                INSERT INTO api_logs (
                    id, trace_id, model, prompt_tokens, completion_tokens,
                    total_tokens, total_cost, status, created_at
                ) VALUES (
                    %(id)s, %(trace_id)s, 'gpt-3.5-turbo', 100, 50,
                    150, 0.05, 'success', NOW()
                )
            """, {'id': log_id, 'trace_id': f'trace-{log_id}'})
        
        try:
            # 测试大数据集下的检查性能
            start_time = datetime.now()
            report = await checker.generate_report()
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            
            # 验证性能（应该在合理时间内完成）
            assert execution_time < 30.0  # 30秒内完成
            assert report.check_duration < 30.0
            
            # 验证结果正确性
            assert isinstance(report.total_issues, int)
            assert report.total_issues >= 0
            
        finally:
            # 清理大量测试数据
            await db_manager.execute_query("DELETE FROM api_logs WHERE id >= 10000")
    
    @pytest.mark.asyncio
    async def test_data_consistency_monitoring_integration(self, consistency_system, test_data_setup):
        """测试数据一致性监控集成"""
        checker = consistency_system['checker']
        
        # 模拟监控场景：定期检查数据一致性
        monitoring_results = []
        
        for i in range(3):
            # 每次检查间隔
            await asyncio.sleep(0.1)
            
            # 执行检查
            report = await checker.generate_report()
            monitoring_results.append({
                'timestamp': datetime.now(),
                'total_issues': report.total_issues,
                'critical_issues': report.critical_issues,
                'check_duration': report.check_duration
            })
        
        # 验证监控结果
        assert len(monitoring_results) == 3
        assert all(result['total_issues'] >= 0 for result in monitoring_results)
        assert all(result['check_duration'] > 0 for result in monitoring_results)
        
        # 验证监控趋势（问题数量应该相对稳定或减少）
        issue_counts = [result['total_issues'] for result in monitoring_results]
        assert max(issue_counts) - min(issue_counts) <= len(test_data_setup['api_logs'])
    
    @pytest.mark.asyncio
    async def test_cross_component_data_flow(self, consistency_system, test_data_setup):
        """测试跨组件数据流"""
        checker = consistency_system['checker']
        constraint_manager = consistency_system['constraint_manager']
        correction_service = consistency_system['correction_service']
        
        # 第一阶段：数据一致性检查
        consistency_report = await checker.generate_report()
        
        # 第二阶段：约束检查
        constraint_report = await constraint_manager.generate_constraint_report()
        
        # 第三阶段：基于检查结果进行修正
        all_issues = consistency_report.issues
        correction_results = []
        
        for issue in all_issues[:3]:  # 限制修正数量以控制测试时间
            if issue.issue_type == IssueType.MISSING_TOKEN_DATA:
                result = await correction_service.correct_missing_token_data(issue.log_id)
                correction_results.append(result)
            elif issue.issue_type == IssueType.MISSING_COST_DATA:
                result = await correction_service.correct_missing_cost_data(issue.log_id)
                correction_results.append(result)
        
        # 第四阶段：验证修正效果
        final_consistency_report = await checker.generate_report()
        final_constraint_report = await constraint_manager.generate_constraint_report()
        
        # 验证数据流的完整性
        assert isinstance(consistency_report.total_issues, int)
        assert isinstance(constraint_report.total_violations, int)
        assert len(correction_results) <= 3
        assert isinstance(final_consistency_report.total_issues, int)
        assert isinstance(final_constraint_report.total_violations, int)
        
        # 验证修正效果（问题数量应该减少或保持不变）
        assert final_consistency_report.total_issues <= consistency_report.total_issues