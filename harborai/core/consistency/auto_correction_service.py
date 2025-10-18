"""
自动修正服务模块

该模块提供数据自动修正功能，包括：
- Token数据自动修正
- 成本数据修正逻辑
- 追踪数据修复
- 批量数据修正
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ...database.postgres_client import PostgreSQLClient
from .data_consistency_checker import DataConsistencyChecker, ConsistencyIssue, IssueType, IssueSeverity


class ActionType(Enum):
    """修正操作类型枚举"""
    UPDATE = "update"
    INSERT = "insert"
    DELETE = "delete"
    RECALCULATE = "recalculate"


class CorrectionStatus(Enum):
    """修正状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class CorrectionAction:
    """修正操作信息"""
    action_type: str  # 'update', 'insert', 'delete', 'recalculate'
    table_name: str
    record_id: Optional[int]
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    reason: str
    confidence: float  # 0.0-1.0，修正置信度


@dataclass
class CorrectionResult:
    """修正结果"""
    success: bool
    status: CorrectionStatus
    actions_performed: List[CorrectionAction]
    errors: List[str]
    warnings: List[str]
    total_records_affected: int
    execution_time: float


class AutoCorrectionService:
    """
    自动修正服务
    
    提供数据自动修正功能，包括：
    - 检测并修正数据不一致问题
    - 重新计算错误的token和成本数据
    - 修复缺失的追踪信息
    - 批量数据修正操作
    """
    
    def __init__(self, db_client: PostgreSQLClient, consistency_checker: DataConsistencyChecker):
        """
        初始化自动修正服务
        
        Args:
            db_client: 数据库客户端
            consistency_checker: 数据一致性检查器
        """
        self.db_client = db_client
        self.consistency_checker = consistency_checker
        self.logger = logging.getLogger(__name__)
        
        # 修正配置
        self.max_batch_size = 1000
        self.confidence_threshold = 0.8  # 只执行高置信度的修正
        self.dry_run_mode = False  # 是否为试运行模式
    
    async def auto_correct_issues(
        self, 
        issues: List[ConsistencyIssue],
        dry_run: bool = False
    ) -> CorrectionResult:
        """
        自动修正一致性问题
        
        Args:
            issues: 一致性问题列表
            dry_run: 是否为试运行模式
            
        Returns:
            修正结果
        """
        start_time = datetime.now()
        actions_performed = []
        errors = []
        warnings = []
        total_affected = 0
        
        self.dry_run_mode = dry_run
        
        try:
            # 按问题类型分组处理
            grouped_issues = self._group_issues_by_type(issues)
            
            for issue_type, issue_list in grouped_issues.items():
                try:
                    if issue_type == IssueType.TOKEN_MISMATCH:
                        result = await self._correct_token_mismatch(issue_list)
                    elif issue_type == IssueType.COST_MISMATCH:
                        result = await self._correct_cost_mismatch(issue_list)
                    elif issue_type == IssueType.MISSING_TRACING:
                        result = await self._correct_missing_tracing_data(issue_list)
                    elif issue_type == IssueType.ORPHANED_RECORD:
                        result = await self._correct_orphaned_records(issue_list)
                    elif issue_type == IssueType.INVALID_DATA_RANGE:
                        result = await self._correct_invalid_data_range(issue_list)
                    elif issue_type == IssueType.CONSTRAINT_VIOLATION:
                        result = await self._correct_constraint_violation(issue_list)
                    elif issue_type == IssueType.DATA_CORRUPTION:
                        result = await self._correct_data_corruption(issue_list)
                    elif issue_type == IssueType.PERFORMANCE_ANOMALY:
                        # 性能异常通常不需要数据修正，只记录警告
                        warnings.append(f"检测到{len(issue_list)}个性能异常问题，建议检查系统性能")
                        continue
                    else:
                        warnings.append(f"未支持的问题类型: {issue_type}")
                        continue
                    
                    actions_performed.extend(result.actions_performed)
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)
                    total_affected += result.total_records_affected
                    
                except Exception as e:
                    error_msg = f"修正{issue_type}时发生错误: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    errors.append(error_msg)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 确定整体状态
            if len(errors) == 0:
                overall_status = CorrectionStatus.SUCCESS
            elif total_affected > 0:
                overall_status = CorrectionStatus.PARTIAL
            else:
                overall_status = CorrectionStatus.FAILED
            
            return CorrectionResult(
                success=len(errors) == 0,
                status=overall_status,
                actions_performed=actions_performed,
                errors=errors,
                warnings=warnings,
                total_records_affected=total_affected,
                execution_time=execution_time
            )
            
        except Exception as e:
            error_msg = f"自动修正过程中发生严重错误: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            return CorrectionResult(
                success=False,
                status=CorrectionStatus.FAILED,
                actions_performed=actions_performed,
                errors=[error_msg],
                warnings=warnings,
                total_records_affected=total_affected,
                execution_time=execution_time
            )

    async def correct_missing_token_data(self, log_id: int, dry_run: bool = False) -> CorrectionResult:
        """
        修正缺失的token数据
        
        Args:
            log_id: API日志ID
            
        Returns:
            修正结果
        """
        # 创建一个虚拟的一致性问题
        issue = ConsistencyIssue(
            issue_id=f'missing_token_{log_id}',
            table_name='token_usage',
            record_id=str(log_id),
            issue_type=IssueType.TOKEN_MISMATCH,
            description=f'修正缺失的token数据 for log_id={log_id}',
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion='创建缺失的token使用记录',
            affected_fields=['prompt_tokens', 'completion_tokens', 'total_tokens'],
            metadata={}
        )
        return await self.auto_correct_issues([issue], dry_run=dry_run)

    async def correct_missing_cost_data(self, log_id: int) -> CorrectionResult:
        """
        修正缺失的成本数据
        
        Args:
            log_id: API日志ID
            
        Returns:
            修正结果
        """
        issue = ConsistencyIssue(
            issue_id=f'missing_cost_{log_id}',
            table_name='cost_info',
            record_id=str(log_id),
            issue_type=IssueType.COST_MISMATCH,
            description=f'修正缺失的成本数据 for log_id={log_id}',
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion='创建缺失的成本记录',
            affected_fields=['input_cost', 'output_cost', 'total_cost'],
            metadata={}
        )
        return await self.auto_correct_issues([issue])

    async def correct_missing_tracing_data(self, log_id: int) -> CorrectionResult:
        """
        修正缺失的追踪数据
        
        Args:
            log_id: API日志ID
            
        Returns:
            修正结果
        """
        issue = ConsistencyIssue(
            issue_id=f'missing_tracing_{log_id}',
            table_name='tracing_info',
            record_id=str(log_id),
            issue_type=IssueType.MISSING_TRACING,
            description=f'修正缺失的追踪数据 for log_id={log_id}',
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion='创建缺失的追踪记录',
            affected_fields=['trace_id', 'span_id', 'parent_span_id'],
            metadata={}
        )
        return await self.auto_correct_issues([issue])

    async def correct_inconsistent_token_counts(self, log_id: int) -> CorrectionResult:
        """
        修正不一致的token计数
        
        Args:
            log_id: API日志ID
            
        Returns:
            修正结果
        """
        issue = ConsistencyIssue(
            issue_id=f'token_inconsistent_{log_id}',
            table_name='token_usage',
            record_id=str(log_id),
            issue_type=IssueType.TOKEN_MISMATCH,
            description=f'Token计数不一致 for log_id={log_id}',
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion='重新计算token计数',
            affected_fields=['total_tokens'],
            metadata={}
        )
        return await self.auto_correct_issues([issue])

    async def correct_inconsistent_costs(self, log_id: int) -> CorrectionResult:
        """
        修正不一致的成本
        
        Args:
            log_id: API日志ID
            
        Returns:
            修正结果
        """
        issue = ConsistencyIssue(
            issue_id=f'cost_inconsistent_{log_id}',
            table_name='cost_info',
            record_id=str(log_id),
            issue_type=IssueType.COST_MISMATCH,
            description=f'成本计算不一致 for log_id={log_id}',
            severity=IssueSeverity.HIGH,
            detected_at=datetime.now(),
            auto_fixable=True,
            fix_suggestion='重新计算成本',
            affected_fields=['total_cost'],
            metadata={}
        )
        return await self.auto_correct_issues([issue])

    async def remove_orphaned_records(self, table_name: str, record_ids: List[int]) -> CorrectionResult:
        """
        移除孤立记录
        
        Args:
            table_name: 表名
            record_ids: 记录ID列表
            
        Returns:
            修正结果
        """
        issues = []
        for record_id in record_ids:
            issue = ConsistencyIssue(
                issue_id=f'orphaned_{table_name}_{record_id}',
                table_name=table_name,
                record_id=str(record_id),
                issue_type=IssueType.ORPHANED_RECORD,
                description=f'孤立记录 in {table_name}',
                severity=IssueSeverity.MEDIUM,
                detected_at=datetime.now(),
                auto_fixable=True,
                fix_suggestion='删除孤立记录',
                affected_fields=[],
                metadata={}
            )
            issues.append(issue)
        return await self.auto_correct_issues(issues)

    async def recalculate_token_totals(self, log_ids: List[int]) -> CorrectionResult:
        """
        重新计算token总数
        
        Args:
            log_ids: API日志ID列表
            
        Returns:
            修正结果
        """
        issues = []
        for log_id in log_ids:
            issue = ConsistencyIssue(
                issue_id=f'recalc_token_{log_id}',
                table_name='token_usage',
                record_id=str(log_id),
                issue_type=IssueType.TOKEN_MISMATCH,
                description=f'重新计算token总数 for log_id={log_id}',
                severity=IssueSeverity.MEDIUM,
                detected_at=datetime.now(),
                auto_fixable=True,
                fix_suggestion='重新计算token总数',
                affected_fields=['total_tokens'],
                metadata={}
            )
            issues.append(issue)
        return await self.auto_correct_issues(issues)

    async def recalculate_cost_totals(self, log_ids: List[int]) -> CorrectionResult:
        """
        重新计算成本总数
        
        Args:
            log_ids: API日志ID列表
            
        Returns:
            修正结果
        """
        issues = []
        for log_id in log_ids:
            issue = ConsistencyIssue(
                issue_id=f'recalc_cost_{log_id}',
                table_name='cost_info',
                record_id=str(log_id),
                issue_type=IssueType.COST_MISMATCH,
                description=f'重新计算成本总数 for log_id={log_id}',
                severity=IssueSeverity.MEDIUM,
                detected_at=datetime.now(),
                auto_fixable=True,
                fix_suggestion='重新计算成本总数',
                affected_fields=['total_cost'],
                metadata={}
            )
            issues.append(issue)
        return await self.auto_correct_issues(issues)

    async def _correct_token_mismatch(self, issues: List[ConsistencyIssue]) -> CorrectionResult:
        """修正Token不匹配问题"""
        actions = []
        errors = []
        warnings = []
        affected_count = 0
        
        for issue in issues:
            try:
                log_id = issue.record_id
                
                # 获取原始数据
                log_data = await self._get_api_log_data(log_id)
                current_token_data = await self._get_token_usage_data(log_id)
                
                if not log_data:
                    error_msg = f"无法获取log_id={log_id}的API日志数据，无法进行修正"
                    errors.append(error_msg)
                    continue
                
                # 重新计算正确的token使用量
                correct_token_data = await self._recalculate_token_usage(log_data)
                if not correct_token_data:
                    warnings.append(f"无法重新计算log_id={log_id}的token使用量")
                    continue
                
                # 如果没有现有数据，插入新数据；否则更新
                if not current_token_data:
                    if not self.dry_run_mode:
                        await self._insert_token_usage(log_id, correct_token_data)
                    action_type = 'insert'
                    old_values = {}
                else:
                    if not self.dry_run_mode:
                        await self._update_token_usage(log_id, correct_token_data)
                    action_type = 'update'
                    old_values = current_token_data
                
                action = CorrectionAction(
                    action_type=action_type,
                    table_name='token_usage',
                    record_id=log_id,
                    old_values=old_values,
                    new_values=correct_token_data,
                    reason=f"修正Token不匹配问题",
                    confidence=0.9
                )
                actions.append(action)
                affected_count += 1
                
            except Exception as e:
                error_msg = f"修正log_id={issue.record_id}的Token不匹配问题时出错: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        return CorrectionResult(
            success=len(errors) == 0,
            status=CorrectionStatus.SUCCESS if len(errors) == 0 else CorrectionStatus.FAILED,
            actions_performed=actions,
            errors=errors,
            warnings=warnings,
            total_records_affected=affected_count,
            execution_time=0.0
        )

    async def _correct_cost_mismatch(self, issues: List[ConsistencyIssue]) -> CorrectionResult:
        """修正成本不匹配问题"""
        actions = []
        errors = []
        warnings = []
        affected_count = 0
        
        for issue in issues:
            try:
                log_id = issue.record_id
                
                # 获取当前数据
                token_data = await self._get_token_usage_data(log_id)
                current_cost_data = await self._get_cost_info_data(log_id)
                
                if not token_data:
                    warnings.append(f"无法获取log_id={log_id}的token使用数据，无法计算成本")
                    continue
                
                # 重新计算正确的成本
                correct_cost_data = await self._recalculate_cost(token_data)
                if not correct_cost_data:
                    warnings.append(f"无法重新计算log_id={log_id}的成本数据")
                    continue
                
                # 如果没有现有数据，插入新数据；否则更新
                if not current_cost_data:
                    if not self.dry_run_mode:
                        await self._insert_cost_info(log_id, correct_cost_data)
                    action_type = 'insert'
                    old_values = {}
                else:
                    if not self.dry_run_mode:
                        await self._update_cost_info(log_id, correct_cost_data)
                    action_type = 'update'
                    old_values = current_cost_data
                
                action = CorrectionAction(
                    action_type=action_type,
                    table_name='cost_info',
                    record_id=log_id,
                    old_values=old_values,
                    new_values=correct_cost_data,
                    reason=f"修正成本不匹配问题",
                    confidence=0.95
                )
                actions.append(action)
                affected_count += 1
                
            except Exception as e:
                error_msg = f"修正log_id={issue.record_id}的成本不匹配问题时出错: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        return CorrectionResult(
            success=len(errors) == 0,
            status=CorrectionStatus.SUCCESS if len(errors) == 0 else CorrectionStatus.FAILED,
            actions_performed=actions,
            errors=errors,
            warnings=warnings,
            total_records_affected=affected_count,
            execution_time=0.0
        )

    async def _correct_invalid_data_range(self, issues: List[ConsistencyIssue]) -> CorrectionResult:
        """修正无效数据范围问题"""
        actions = []
        errors = []
        warnings = []
        affected_count = 0
        
        for issue in issues:
            try:
                log_id = issue.record_id
                
                # 根据问题描述确定需要修正的字段
                if "token" in issue.description.lower():
                    # 修正token相关的无效数据
                    log_data = await self._get_api_log_data(log_id)
                    if log_data:
                        correct_token_data = await self._recalculate_token_usage(log_data)
                        if correct_token_data and not self.dry_run_mode:
                            await self._update_token_usage(log_id, correct_token_data)
                        
                        action = CorrectionAction(
                            action_type='update',
                            table_name='token_usage',
                            record_id=log_id,
                            old_values={'invalid_range': True},
                            new_values=correct_token_data or {},
                            reason=f"修正Token数据范围问题",
                            confidence=0.8
                        )
                        actions.append(action)
                        affected_count += 1
                
                elif "cost" in issue.description.lower():
                    # 修正成本相关的无效数据
                    token_data = await self._get_token_usage_data(log_id)
                    if token_data:
                        correct_cost_data = await self._recalculate_cost(token_data)
                        if correct_cost_data and not self.dry_run_mode:
                            await self._update_cost_info(log_id, correct_cost_data)
                        
                        action = CorrectionAction(
                            action_type='update',
                            table_name='cost_info',
                            record_id=log_id,
                            old_values={'invalid_range': True},
                            new_values=correct_cost_data or {},
                            reason=f"修正成本数据范围问题",
                            confidence=0.8
                        )
                        actions.append(action)
                        affected_count += 1
                
                else:
                    warnings.append(f"未知的数据范围问题类型: {issue.description}")
                
            except Exception as e:
                error_msg = f"修正log_id={issue.record_id}的数据范围问题时出错: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        return CorrectionResult(
            success=len(errors) == 0,
            status=CorrectionStatus.SUCCESS if len(errors) == 0 else CorrectionStatus.FAILED,
            actions_performed=actions,
            errors=errors,
            warnings=warnings,
            total_records_affected=affected_count,
            execution_time=0.0
        )

    async def _correct_constraint_violation(self, issues: List[ConsistencyIssue]) -> CorrectionResult:
        """修正约束违反问题"""
        actions = []
        errors = []
        warnings = []
        affected_count = 0
        
        for issue in issues:
            try:
                # 约束违反通常需要删除或修正违反约束的记录
                if "foreign key" in issue.description.lower():
                    # 外键约束违反，删除孤立记录
                    result = await self._correct_orphaned_records([issue])
                    actions.extend(result.actions_performed)
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)
                    affected_count += result.total_records_affected
                
                elif "unique" in issue.description.lower():
                    # 唯一约束违反，保留最新记录，删除重复记录
                    warnings.append(f"检测到唯一约束违反: {issue.description}，需要手动处理")
                
                elif "check" in issue.description.lower():
                    # 检查约束违反，修正数据值
                    warnings.append(f"检测到检查约束违反: {issue.description}，需要手动处理")
                
                else:
                    warnings.append(f"未知的约束违反类型: {issue.description}")
                
            except Exception as e:
                error_msg = f"修正约束违反问题时出错: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        return CorrectionResult(
            success=len(errors) == 0,
            status=CorrectionStatus.SUCCESS if len(errors) == 0 else CorrectionStatus.FAILED,
            actions_performed=actions,
            errors=errors,
            warnings=warnings,
            total_records_affected=affected_count,
            execution_time=0.0
        )

    async def _correct_missing_tracing_data(self, issues: List[ConsistencyIssue]) -> CorrectionResult:
        """修正缺失的追踪数据"""
        actions = []
        errors = []
        warnings = []
        affected_count = 0
        
        for issue in issues:
            try:
                log_id = issue.record_id
                
                # 从api_logs获取追踪信息
                log_data = await self._get_api_log_data(log_id)
                if not log_data:
                    error_msg = f"无法找到log_id={log_id}的API日志数据，无法生成追踪数据"
                    errors.append(error_msg)
                    continue
                
                # 提取或生成追踪数据
                tracing_data = await self._extract_or_generate_tracing_data(log_data)
                if not tracing_data:
                    warnings.append(f"无法为log_id={log_id}生成追踪数据")
                    continue
                
                # 插入追踪数据
                if not self.dry_run_mode:
                    await self._insert_tracing_info(log_id, tracing_data)
                
                action = CorrectionAction(
                    action_type='insert',
                    table_name='tracing_info',
                    record_id=log_id,
                    old_values={},
                    new_values=tracing_data,
                    reason=f"提取或生成缺失的追踪数据",
                    confidence=0.7  # 生成的追踪数据置信度较低
                )
                actions.append(action)
                affected_count += 1
                
            except Exception as e:
                error_msg = f"修正log_id={issue.record_id}的追踪数据时出错: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        return CorrectionResult(
            success=len(errors) == 0,
            status=CorrectionStatus.SUCCESS if len(errors) == 0 else CorrectionStatus.FAILED,
            actions_performed=actions,
            errors=errors,
            warnings=warnings,
            total_records_affected=affected_count,
            execution_time=0.0
        )

    async def _correct_data_corruption(self, issues: List[ConsistencyIssue]) -> CorrectionResult:
        """修正数据损坏问题"""
        actions = []
        errors = []
        warnings = []
        affected_count = 0
        
        for issue in issues:
            try:
                log_id = issue.record_id
                
                # 数据损坏通常需要从原始数据重新生成
                log_data = await self._get_api_log_data(log_id)
                if not log_data:
                    warnings.append(f"无法获取log_id={log_id}的原始数据，无法修复损坏")
                    continue
                
                # 重新生成所有相关数据
                token_data = await self._recalculate_token_usage(log_data)
                if token_data and not self.dry_run_mode:
                    await self._update_token_usage(log_id, token_data)
                    
                    cost_data = await self._recalculate_cost(token_data)
                    if cost_data:
                        await self._update_cost_info(log_id, cost_data)
                
                action = CorrectionAction(
                    action_type='rebuild',
                    table_name='multiple',
                    record_id=log_id,
                    old_values={'corrupted': True},
                    new_values={'rebuilt': True},
                    reason=f"重建损坏的数据",
                    confidence=0.7
                )
                actions.append(action)
                affected_count += 1
                
            except Exception as e:
                error_msg = f"修正数据损坏问题时出错: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        return CorrectionResult(
            success=len(errors) == 0,
            status=CorrectionStatus.SUCCESS if len(errors) == 0 else CorrectionStatus.FAILED,
            actions_performed=actions,
            errors=errors,
            warnings=warnings,
            total_records_affected=affected_count,
            execution_time=0.0
        )
    
    async def _correct_orphaned_records(self, issues: List[ConsistencyIssue]) -> CorrectionResult:
        """修正孤立记录"""
        actions = []
        errors = []
        warnings = []
        affected_count = 0
        
        for issue in issues:
            try:
                # 根据问题描述确定处理策略
                if "token_usage" in issue.description:
                    # 删除孤立的token_usage记录
                    if not self.dry_run_mode:
                        await self._delete_orphaned_token_usage(issue.record_id)
                    
                    action = CorrectionAction(
                        action_type='delete',
                        table_name='token_usage',
                        record_id=issue.record_id,
                        old_values={'log_id': issue.record_id},
                        new_values={},
                        reason="删除孤立的token_usage记录",
                        confidence=0.8
                    )
                    
                elif "cost_info" in issue.description:
                    # 删除孤立的cost_info记录
                    if not self.dry_run_mode:
                        await self._delete_orphaned_cost_info(issue.record_id)
                    
                    action = CorrectionAction(
                        action_type='delete',
                        table_name='cost_info',
                        record_id=issue.record_id,
                        old_values={'log_id': issue.record_id},
                        new_values={},
                        reason="删除孤立的cost_info记录",
                        confidence=0.8
                    )
                    
                elif "tracing_info" in issue.description:
                    # 删除孤立的tracing_info记录
                    if not self.dry_run_mode:
                        await self._delete_orphaned_tracing_info(issue.record_id)
                    
                    action = CorrectionAction(
                        action_type='delete',
                        table_name='tracing_info',
                        record_id=issue.record_id,
                        old_values={'log_id': issue.record_id},
                        new_values={},
                        reason="删除孤立的tracing_info记录",
                        confidence=0.8
                    )
                else:
                    warnings.append(f"未知的孤立记录类型: {issue.description}")
                    continue
                
                actions.append(action)
                affected_count += 1
                
            except Exception as e:
                error_msg = f"修正孤立记录{issue.record_id}时出错: {str(e)}"
                self.logger.error(error_msg)
                errors.append(error_msg)
        
        return CorrectionResult(
            success=len(errors) == 0,
            status=CorrectionStatus.SUCCESS if len(errors) == 0 else CorrectionStatus.FAILED,
            actions_performed=actions,
            errors=errors,
            warnings=warnings,
            total_records_affected=affected_count,
            execution_time=0.0
        )
    
    def _group_issues_by_type(self, issues: List[ConsistencyIssue]) -> Dict[IssueType, List[ConsistencyIssue]]:
        """按问题类型分组"""
        grouped = {}
        for issue in issues:
            if issue.issue_type not in grouped:
                grouped[issue.issue_type] = []
            grouped[issue.issue_type].append(issue)
        return grouped
    
    async def _get_api_log_data(self, log_id: int) -> Optional[Dict[str, Any]]:
        """获取API日志数据"""
        try:
            query = """
            SELECT id, trace_id, model, prompt_tokens, completion_tokens, 
                   total_tokens, request_data, response_data, created_at
            FROM api_logs 
            WHERE id = %s
            """
            result = await self.db_client.fetch_one(query, (log_id,))
            return dict(result) if result else None
        except Exception as e:
            self.logger.error(f"获取API日志数据失败: {e}")
            return None
    
    async def _get_token_usage_data(self, log_id: int) -> Optional[Dict[str, Any]]:
        """获取token使用数据"""
        try:
            query = """
            SELECT log_id, prompt_tokens, completion_tokens, total_tokens,
                   prompt_cost, completion_cost, total_cost
            FROM token_usage 
            WHERE log_id = %s
            """
            result = await self.db_client.fetch_one(query, (log_id,))
            return dict(result) if result else None
        except Exception as e:
            self.logger.error(f"获取token使用数据失败: {e}")
            return None
    
    async def _get_cost_info_data(self, log_id: int) -> Optional[Dict[str, Any]]:
        """获取成本信息数据"""
        try:
            query = """
            SELECT log_id, input_cost, output_cost, total_cost, 
                   cost_per_token, currency, pricing_model
            FROM cost_info 
            WHERE log_id = %s
            """
            result = await self.db_client.fetch_one(query, (log_id,))
            return dict(result) if result else None
        except Exception as e:
            self.logger.error(f"获取成本信息数据失败: {e}")
            return None
    
    async def _recalculate_token_usage(self, log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """重新计算token使用量"""
        try:
            # 从API日志中提取token信息
            prompt_tokens = log_data.get('prompt_tokens', 0)
            completion_tokens = log_data.get('completion_tokens', 0)
            total_tokens = log_data.get('total_tokens', prompt_tokens + completion_tokens)
            
            # 如果API日志中没有token信息，尝试从请求/响应数据中解析
            if total_tokens == 0:
                # 这里可以添加更复杂的token计算逻辑
                # 例如基于模型和文本内容估算token数量
                pass
            
            return {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'prompt_cost': 0.0,  # 将在成本计算中更新
                'completion_cost': 0.0,
                'total_cost': 0.0
            }
        except Exception as e:
            self.logger.error(f"重新计算token使用量失败: {e}")
            return None
    
    async def _recalculate_cost(self, token_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """重新计算成本"""
        try:
            # 这里应该使用实际的定价模型
            # 暂时使用简单的固定价格
            prompt_tokens = token_data.get('prompt_tokens', 0)
            completion_tokens = token_data.get('completion_tokens', 0)
            
            # 示例定价（实际应该从定价服务获取）
            prompt_price_per_token = 0.0001
            completion_price_per_token = 0.0002
            
            input_cost = prompt_tokens * prompt_price_per_token
            output_cost = completion_tokens * completion_price_per_token
            total_cost = input_cost + output_cost
            
            return {
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost,
                'cost_per_token': total_cost / max(prompt_tokens + completion_tokens, 1),
                'currency': 'USD',
                'pricing_model': 'standard'
            }
        except Exception as e:
            self.logger.error(f"重新计算成本失败: {e}")
            return None
    
    async def _extract_or_generate_tracing_data(self, log_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取或生成追踪数据"""
        try:
            trace_id = log_data.get('trace_id')
            if not trace_id:
                # 生成新的trace_id
                import uuid
                trace_id = str(uuid.uuid4())
            
            return {
                'hb_trace_id': trace_id,
                'otel_trace_id': None,  # 如果没有OpenTelemetry追踪
                'span_id': None,
                'operation_name': 'api_call',
                'start_time': log_data.get('created_at'),
                'end_time': log_data.get('created_at'),  # 如果没有结束时间，使用创建时间
                'duration_ms': 0,
                'status': 'completed',
                'tags': {},
                'metadata': {}
            }
        except Exception as e:
            self.logger.error(f"提取或生成追踪数据失败: {e}")
            return None
    
    # 数据库操作方法
    async def _insert_token_usage(self, log_id: int, token_data: Dict[str, Any]):
        """插入token使用数据"""
        query = """
        INSERT INTO token_usage (log_id, prompt_tokens, completion_tokens, total_tokens,
                                prompt_cost, completion_cost, total_cost)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        await self.db_client.execute(query, (
            log_id,
            token_data['prompt_tokens'],
            token_data['completion_tokens'],
            token_data['total_tokens'],
            token_data['prompt_cost'],
            token_data['completion_cost'],
            token_data['total_cost']
        ))
    
    async def _insert_cost_info(self, log_id: int, cost_data: Dict[str, Any]):
        """插入成本信息数据"""
        query = """
        INSERT INTO cost_info (log_id, input_cost, output_cost, total_cost,
                              cost_per_token, currency, pricing_model)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        await self.db_client.execute(query, (
            log_id,
            cost_data['input_cost'],
            cost_data['output_cost'],
            cost_data['total_cost'],
            cost_data['cost_per_token'],
            cost_data['currency'],
            cost_data['pricing_model']
        ))
    
    async def _insert_tracing_info(self, log_id: int, tracing_data: Dict[str, Any]):
        """插入追踪信息数据"""
        query = """
        INSERT INTO tracing_info (log_id, hb_trace_id, otel_trace_id, span_id,
                                 operation_name, start_time, end_time, duration_ms,
                                 status, tags, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        await self.db_client.execute(query, (
            log_id,
            tracing_data['hb_trace_id'],
            tracing_data['otel_trace_id'],
            tracing_data['span_id'],
            tracing_data['operation_name'],
            tracing_data['start_time'],
            tracing_data['end_time'],
            tracing_data['duration_ms'],
            tracing_data['status'],
            tracing_data['tags'],
            tracing_data['metadata']
        ))
    
    async def _update_token_usage(self, log_id: int, token_data: Dict[str, Any]):
        """更新token使用数据"""
        query = """
        UPDATE token_usage 
        SET prompt_tokens = %s, completion_tokens = %s, total_tokens = %s,
            prompt_cost = %s, completion_cost = %s, total_cost = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE log_id = %s
        """
        await self.db_client.execute(query, (
            token_data['prompt_tokens'],
            token_data['completion_tokens'],
            token_data['total_tokens'],
            token_data['prompt_cost'],
            token_data['completion_cost'],
            token_data['total_cost'],
            log_id
        ))
    
    async def _update_cost_info(self, log_id: int, cost_data: Dict[str, Any]):
        """更新成本信息数据"""
        query = """
        UPDATE cost_info 
        SET input_cost = %s, output_cost = %s, total_cost = %s,
            cost_per_token = %s, currency = %s, pricing_model = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE log_id = %s
        """
        await self.db_client.execute(query, (
            cost_data['input_cost'],
            cost_data['output_cost'],
            cost_data['total_cost'],
            cost_data['cost_per_token'],
            cost_data['currency'],
            cost_data['pricing_model'],
            log_id
        ))
    
    async def _delete_orphaned_token_usage(self, log_id: int):
        """删除孤立的token使用记录"""
        query = "DELETE FROM token_usage WHERE log_id = %s"
        await self.db_client.execute(query, (log_id,))
    
    async def _delete_orphaned_cost_info(self, log_id: int):
        """删除孤立的成本信息记录"""
        query = "DELETE FROM cost_info WHERE log_id = %s"
        await self.db_client.execute(query, (log_id,))
    
    async def _delete_orphaned_tracing_info(self, log_id: int):
        """删除孤立的追踪信息记录"""
        query = "DELETE FROM tracing_info WHERE log_id = %s"
        await self.db_client.execute(query, (log_id,))