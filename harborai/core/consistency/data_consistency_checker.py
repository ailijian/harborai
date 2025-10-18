#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据一致性检查器

负责检查token数据、成本数据和追踪数据的一致性，支持异步操作和智能分析
"""

import logging
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib

from ...database.async_manager import DatabaseManager

logger = logging.getLogger(__name__)


class IssueType(Enum):
    """问题类型枚举"""
    TOKEN_MISMATCH = "token_mismatch"
    COST_MISMATCH = "cost_mismatch"
    MISSING_TRACING = "missing_tracing"
    ORPHANED_RECORD = "orphaned_record"
    INVALID_DATA_RANGE = "invalid_data_range"
    CONSTRAINT_VIOLATION = "constraint_violation"
    DATA_CORRUPTION = "data_corruption"
    PERFORMANCE_ANOMALY = "performance_anomaly"


class IssueSeverity(Enum):
    """问题严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConsistencyIssue:
    """一致性问题数据类"""
    issue_id: str
    table_name: str
    record_id: str
    issue_type: IssueType
    description: str
    severity: IssueSeverity
    detected_at: datetime
    auto_fixable: bool
    fix_suggestion: Optional[str] = None
    affected_fields: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "issue_id": self.issue_id,
            "table_name": self.table_name,
            "record_id": self.record_id,
            "issue_type": self.issue_type.value,
            "description": self.description,
            "severity": self.severity.value,
            "detected_at": self.detected_at.isoformat(),
            "auto_fixable": self.auto_fixable,
            "fix_suggestion": self.fix_suggestion,
            "affected_fields": self.affected_fields,
            "metadata": self.metadata
        }


@dataclass
class ConsistencyReport:
    """一致性检查报告"""
    check_id: str
    check_timestamp: datetime
    total_records_checked: int
    issues: List[ConsistencyIssue]
    check_duration_ms: float
    summary: Dict[str, int]
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def total_issues(self) -> int:
        """总问题数量"""
        return len(self.issues)
    
    @property
    def critical_issues(self) -> List[ConsistencyIssue]:
        """关键问题列表"""
        return [issue for issue in self.issues if issue.severity == IssueSeverity.CRITICAL]
    
    @property
    def auto_fixable_issues(self) -> List[ConsistencyIssue]:
        """可自动修复的问题列表"""
        return [issue for issue in self.issues if issue.auto_fixable]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "check_id": self.check_id,
            "check_timestamp": self.check_timestamp.isoformat(),
            "total_records_checked": self.total_records_checked,
            "total_issues": self.total_issues,
            "issues": [issue.to_dict() for issue in self.issues],
            "check_duration_ms": self.check_duration_ms,
            "summary": self.summary,
            "recommendations": self.recommendations
        }


class DataConsistencyChecker:
    """数据一致性检查器
    
    功能：
    1. Token数据一致性验证
    2. 成本数据一致性检查  
    3. 追踪数据完整性验证
    4. 跨表关联数据一致性检查
    5. 数据质量分析
    6. 性能异常检测
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """初始化数据一致性检查器
        
        Args:
            db_manager: 异步数据库管理器实例
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # 检查配置
        self.batch_size = 1000
        self.max_concurrent_checks = 5
        
        # 缓存检查结果
        self._check_cache: Dict[str, Any] = {}
        
    async def generate_report(self, 
                            days_back: int = 7,
                            batch_size: int = 1000,
                            include_performance_check: bool = True) -> ConsistencyReport:
        """生成完整的一致性检查报告
        
        Args:
            days_back: 检查最近几天的数据
            batch_size: 批处理大小
            include_performance_check: 是否包含性能检查
            
        Returns:
            ConsistencyReport: 一致性检查报告
        """
        start_time = datetime.now()
        check_id = f"consistency_check_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"开始执行数据一致性检查 - ID: {check_id}")
        
        all_issues = []
        total_records = 0
        
        try:
            # 并发执行各项检查
            check_tasks = [
                self.check_token_consistency(days_back, batch_size),
                self.check_cost_consistency(days_back, batch_size),
                self.check_tracing_completeness(days_back, batch_size),
                self.check_foreign_key_integrity(days_back, batch_size),
                self.check_data_ranges(days_back, batch_size)
            ]
            
            if include_performance_check:
                check_tasks.append(self.check_performance_anomalies(days_back, batch_size))
            
            # 执行所有检查
            results = await asyncio.gather(*check_tasks, return_exceptions=True)
            
            # 处理检查结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"检查任务 {i} 失败: {result}")
                    continue
                
                issues, count = result
                all_issues.extend(issues)
                total_records += count
            
            # 生成摘要统计
            summary = self._generate_summary(all_issues)
            
            # 生成建议
            recommendations = self._generate_recommendations(all_issues)
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            report = ConsistencyReport(
                check_id=check_id,
                check_timestamp=start_time,
                total_records_checked=total_records,
                issues=all_issues,
                check_duration_ms=duration_ms,
                summary=summary,
                recommendations=recommendations
            )
            
            self.logger.info(f"数据一致性检查完成 - 检查记录数: {total_records}, "
                           f"发现问题: {len(all_issues)}, 耗时: {duration_ms:.2f}ms")
            
            return report
            
        except Exception as e:
            self.logger.error(f"数据一致性检查失败: {str(e)}")
            raise
    
    async def check_token_consistency(self, 
                                    days_back: int = 7, 
                                    batch_size: int = 1000) -> Tuple[List[ConsistencyIssue], int]:
        """检查Token数据一致性
        
        检查项目：
        1. API日志与token_usage表的token数量一致性
        2. total_tokens = prompt_tokens + completion_tokens
        3. token数量为非负数
        4. confidence值在0-1范围内
        """
        issues = []
        total_count = 0
        
        try:
            # 查询API日志与token_usage的对比数据
            query = """
                SELECT 
                    al.id as log_id,
                    al.prompt_tokens as api_prompt_tokens,
                    al.completion_tokens as api_completion_tokens,
                    al.total_tokens as api_total_tokens,
                    tu.prompt_tokens as token_prompt_tokens,
                    tu.completion_tokens as token_completion_tokens,
                    tu.total_tokens as token_total_tokens,
                    tu.confidence,
                    tu.parsing_method,
                    tu.id as token_id
                FROM api_logs al
                LEFT JOIN token_usage tu ON al.id = tu.log_id
                WHERE al.created_at >= $1
                ORDER BY al.created_at DESC
                LIMIT $2
            """
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            records = await self.db_manager.execute_query(query, (cutoff_date, batch_size))
            total_count = len(records)
            
            for record in records:
                record_issues = await self._validate_token_record(record)
                issues.extend(record_issues)
                
        except Exception as e:
            self.logger.error(f"Token一致性检查失败: {str(e)}")
            raise
            
        return issues, total_count
    
    async def _validate_token_record(self, record: Dict[str, Any]) -> List[ConsistencyIssue]:
        """验证单个Token记录的一致性"""
        issues = []
        log_id = str(record['log_id'])
        
        # 检查1: API日志与token_usage表的一致性
        if record['token_id'] is not None:  # 存在token_usage记录
            # 检查prompt_tokens一致性
            if record['api_prompt_tokens'] != record['token_prompt_tokens']:
                issues.append(ConsistencyIssue(
                    issue_id=f"token_prompt_mismatch_{log_id}",
                    table_name="token_usage",
                    record_id=log_id,
                    issue_type=IssueType.TOKEN_MISMATCH,
                    description=f"Prompt tokens不匹配: API={record['api_prompt_tokens']}, "
                              f"Token表={record['token_prompt_tokens']}",
                    severity=IssueSeverity.HIGH,
                    detected_at=datetime.now(),
                    auto_fixable=True,
                    fix_suggestion=f"将token_usage.prompt_tokens更新为{record['api_prompt_tokens']}",
                    affected_fields=["prompt_tokens"]
                ))
            
            # 检查completion_tokens一致性
            if record['api_completion_tokens'] != record['token_completion_tokens']:
                issues.append(ConsistencyIssue(
                    issue_id=f"token_completion_mismatch_{log_id}",
                    table_name="token_usage",
                    record_id=log_id,
                    issue_type=IssueType.TOKEN_MISMATCH,
                    description=f"Completion tokens不匹配: API={record['api_completion_tokens']}, "
                              f"Token表={record['token_completion_tokens']}",
                    severity=IssueSeverity.HIGH,
                    detected_at=datetime.now(),
                    auto_fixable=True,
                    fix_suggestion=f"将token_usage.completion_tokens更新为{record['api_completion_tokens']}",
                    affected_fields=["completion_tokens"]
                ))
            
            # 检查total_tokens一致性
            if record['api_total_tokens'] != record['token_total_tokens']:
                issues.append(ConsistencyIssue(
                    issue_id=f"token_total_mismatch_{log_id}",
                    table_name="token_usage",
                    record_id=log_id,
                    issue_type=IssueType.TOKEN_MISMATCH,
                    description=f"Total tokens不匹配: API={record['api_total_tokens']}, "
                              f"Token表={record['token_total_tokens']}",
                    severity=IssueSeverity.HIGH,
                    detected_at=datetime.now(),
                    auto_fixable=True,
                    fix_suggestion=f"将token_usage.total_tokens更新为{record['api_total_tokens']}",
                    affected_fields=["total_tokens"]
                ))
            
            # 检查2: total_tokens = prompt_tokens + completion_tokens (在token_usage表内)
            if record['token_total_tokens'] is not None:
                expected_total = (record['token_prompt_tokens'] or 0) + (record['token_completion_tokens'] or 0)
                if record['token_total_tokens'] != expected_total:
                    issues.append(ConsistencyIssue(
                        issue_id=f"token_sum_mismatch_{log_id}",
                        table_name="token_usage",
                        record_id=log_id,
                        issue_type=IssueType.TOKEN_MISMATCH,
                        description=f"Token总数计算错误: 实际={record['token_total_tokens']}, "
                                  f"期望={expected_total}",
                        severity=IssueSeverity.MEDIUM,
                        detected_at=datetime.now(),
                        auto_fixable=True,
                        fix_suggestion=f"将total_tokens更新为{expected_total}",
                        affected_fields=["total_tokens"]
                    ))
            
            # 检查3: confidence值范围
            if record['confidence'] is not None:
                if not (0.0 <= record['confidence'] <= 1.0):
                    issues.append(ConsistencyIssue(
                        issue_id=f"confidence_range_{log_id}",
                        table_name="token_usage",
                        record_id=log_id,
                        issue_type=IssueType.INVALID_DATA_RANGE,
                        description=f"置信度超出范围: {record['confidence']} (应在0.0-1.0之间)",
                        severity=IssueSeverity.MEDIUM,
                        detected_at=datetime.now(),
                        auto_fixable=True,
                        fix_suggestion="将confidence限制在0.0-1.0范围内",
                        affected_fields=["confidence"]
                    ))
            
            # 检查4: Token数量非负
            for field in ['token_prompt_tokens', 'token_completion_tokens', 'token_total_tokens']:
                if record[field] is not None and record[field] < 0:
                    issues.append(ConsistencyIssue(
                        issue_id=f"negative_tokens_{field}_{log_id}",
                        table_name="token_usage",
                        record_id=log_id,
                        issue_type=IssueType.INVALID_DATA_RANGE,
                        description=f"{field}为负数: {record[field]}",
                        severity=IssueSeverity.HIGH,
                        detected_at=datetime.now(),
                        auto_fixable=True,
                        fix_suggestion=f"将{field}设置为0",
                        affected_fields=[field.replace('token_', '')]
                    ))
        
        else:
            # 缺失token_usage记录
            issues.append(ConsistencyIssue(
                issue_id=f"missing_token_usage_{log_id}",
                table_name="token_usage",
                record_id=log_id,
                issue_type=IssueType.MISSING_TRACING,
                description=f"API日志缺少对应的token_usage记录",
                severity=IssueSeverity.MEDIUM,
                detected_at=datetime.now(),
                auto_fixable=False,
                fix_suggestion="创建对应的token_usage记录",
                affected_fields=["log_id"]
            ))
        
        return issues
    
    async def check_cost_consistency(self, 
                                   days_back: int = 7, 
                                   batch_size: int = 1000) -> Tuple[List[ConsistencyIssue], int]:
        """检查成本数据一致性"""
        issues = []
        total_count = 0
        
        try:
            # 查询API日志与cost_info的对比数据
            query = """
                SELECT 
                    al.id as log_id,
                    al.total_cost as api_total_cost,
                    ci.input_cost as cost_prompt_cost,
                    ci.output_cost as cost_completion_cost,
                    ci.total_cost as cost_total_cost,
                    ci.currency,
                    ci.pricing_source,
                    ci.id as cost_id
                FROM api_logs al
                LEFT JOIN cost_info ci ON al.id = ci.log_id
                WHERE al.created_at >= $1
                ORDER BY al.created_at DESC
                LIMIT $2
            """
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            records = await self.db_manager.execute_query(query, (cutoff_date, batch_size))
            total_count = len(records)
            
            for record in records:
                record_issues = await self._validate_cost_record(record)
                issues.extend(record_issues)
                
        except Exception as e:
            self.logger.error(f"成本一致性检查失败: {str(e)}")
            raise
            
        return issues, total_count
    
    async def _validate_cost_record(self, record: Dict[str, Any]) -> List[ConsistencyIssue]:
        """验证单个成本记录的一致性"""
        issues = []
        log_id = str(record['log_id'])
        
        if record['cost_id'] is not None:  # 存在cost_info记录
            # 检查1: API日志与cost_info表的总成本一致性
            if (record['api_total_cost'] is not None and 
                record['cost_total_cost'] is not None):
                
                api_cost = float(record['api_total_cost'])
                cost_total = float(record['cost_total_cost'])
                
                if abs(api_cost - cost_total) > 0.000001:
                    issues.append(ConsistencyIssue(
                        issue_id=f"cost_api_mismatch_{log_id}",
                        table_name="cost_info",
                        record_id=log_id,
                        issue_type=IssueType.COST_MISMATCH,
                        description=f"API总成本与cost_info不匹配: API={api_cost:.6f}, "
                                  f"Cost表={cost_total:.6f}",
                        severity=IssueSeverity.HIGH,
                        detected_at=datetime.now(),
                        auto_fixable=True,
                        fix_suggestion=f"将cost_info.total_cost更新为{api_cost:.6f}",
                        affected_fields=["total_cost"]
                    ))
            
            # 检查2: total_cost = input_cost + output_cost
            if (record['cost_prompt_cost'] is not None and 
                record['cost_completion_cost'] is not None and
                record['cost_total_cost'] is not None):
                
                expected_total = float(record['cost_prompt_cost']) + float(record['cost_completion_cost'])
                actual_total = float(record['cost_total_cost'])
                
                if abs(actual_total - expected_total) > 0.000001:
                    issues.append(ConsistencyIssue(
                        issue_id=f"cost_sum_mismatch_{log_id}",
                        table_name="cost_info",
                        record_id=log_id,
                        issue_type=IssueType.COST_MISMATCH,
                        description=f"成本总额计算错误: 实际={actual_total:.6f}, "
                                  f"期望={expected_total:.6f}",
                        severity=IssueSeverity.MEDIUM,
                        detected_at=datetime.now(),
                        auto_fixable=True,
                        fix_suggestion=f"将total_cost更新为{expected_total:.6f}",
                        affected_fields=["total_cost"]
                    ))
            
            # 检查3: 成本值非负
            cost_fields = {
                'cost_prompt_cost': 'input_cost',
                'cost_completion_cost': 'output_cost',
                'cost_total_cost': 'total_cost'
            }
            
            for field, db_field in cost_fields.items():
                if record[field] is not None and float(record[field]) < 0:
                    issues.append(ConsistencyIssue(
                        issue_id=f"negative_cost_{db_field}_{log_id}",
                        table_name="cost_info",
                        record_id=log_id,
                        issue_type=IssueType.INVALID_DATA_RANGE,
                        description=f"{db_field}为负数: {record[field]}",
                        severity=IssueSeverity.HIGH,
                        detected_at=datetime.now(),
                        auto_fixable=True,
                        fix_suggestion=f"将{db_field}设置为0.0",
                        affected_fields=[db_field]
                    ))
        
        else:
            # 缺失cost_info记录
            if record['api_total_cost'] is not None:
                issues.append(ConsistencyIssue(
                    issue_id=f"missing_cost_info_{log_id}",
                    table_name="cost_info",
                    record_id=log_id,
                    issue_type=IssueType.MISSING_TRACING,
                    description=f"API日志缺少对应的cost_info记录",
                    severity=IssueSeverity.MEDIUM,
                    detected_at=datetime.now(),
                    auto_fixable=False,
                    fix_suggestion="创建对应的cost_info记录",
                    affected_fields=["log_id"]
                ))
        
        return issues
    
    async def check_tracing_completeness(self, 
                                       days_back: int = 7, 
                                       batch_size: int = 1000) -> Tuple[List[ConsistencyIssue], int]:
        """检查追踪数据完整性"""
        issues = []
        total_count = 0
        
        try:
            # 查询缺少tracing_info的API日志
            query = """
                SELECT al.id as log_id, al.trace_id
                FROM api_logs al
                LEFT JOIN tracing_info ti ON al.id = ti.log_id
                WHERE al.created_at >= $1 
                  AND ti.id IS NULL
                ORDER BY al.created_at DESC
                LIMIT $2
            """
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            records = await self.db_manager.execute_query(query, (cutoff_date, batch_size))
            total_count = len(records)
            
            for record in records:
                issues.append(ConsistencyIssue(
                    issue_id=f"missing_tracing_{record['log_id']}",
                    table_name="tracing_info",
                    record_id=str(record['log_id']),
                    issue_type=IssueType.MISSING_TRACING,
                    description=f"API日志缺少对应的tracing_info记录",
                    severity=IssueSeverity.MEDIUM,
                    detected_at=datetime.now(),
                    auto_fixable=False,
                    fix_suggestion="创建对应的tracing_info记录",
                    affected_fields=["log_id"],
                    metadata={"trace_id": record['trace_id']}
                ))
                
        except Exception as e:
            self.logger.error(f"追踪完整性检查失败: {str(e)}")
            raise
            
        return issues, total_count
    
    async def check_foreign_key_integrity(self, 
                                        days_back: int = 7, 
                                        batch_size: int = 1000) -> Tuple[List[ConsistencyIssue], int]:
        """检查外键完整性"""
        issues = []
        total_count = 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # 检查孤立的token_usage记录
            orphaned_tokens = await self.db_manager.execute_query("""
                SELECT tu.id, tu.log_id
                FROM token_usage tu
                LEFT JOIN api_logs al ON tu.log_id = al.id
                WHERE al.id IS NULL
                  AND tu.created_at >= $1
                LIMIT $2
            """, (cutoff_date, batch_size))
            
            for record in orphaned_tokens:
                issues.append(ConsistencyIssue(
                    issue_id=f"orphaned_token_{record['id']}",
                    table_name="token_usage",
                    record_id=str(record['id']),
                    issue_type=IssueType.ORPHANED_RECORD,
                    description=f"孤立的token_usage记录，关联的api_logs记录不存在: log_id={record['log_id']}",
                    severity=IssueSeverity.HIGH,
                    detected_at=datetime.now(),
                    auto_fixable=False,
                    fix_suggestion="删除孤立记录或修复log_id关联",
                    affected_fields=["log_id"]
                ))
            
            # 检查孤立的cost_info记录
            orphaned_costs = await self.db_manager.execute_query("""
                SELECT ci.id, ci.log_id
                FROM cost_info ci
                LEFT JOIN api_logs al ON ci.log_id = al.id
                WHERE al.id IS NULL
                  AND ci.created_at >= $1
                LIMIT $2
            """, (cutoff_date, batch_size))
            
            for record in orphaned_costs:
                issues.append(ConsistencyIssue(
                    issue_id=f"orphaned_cost_{record['id']}",
                    table_name="cost_info",
                    record_id=str(record['id']),
                    issue_type=IssueType.ORPHANED_RECORD,
                    description=f"孤立的cost_info记录，关联的api_logs记录不存在: log_id={record['log_id']}",
                    severity=IssueSeverity.HIGH,
                    detected_at=datetime.now(),
                    auto_fixable=False,
                    fix_suggestion="删除孤立记录或修复log_id关联",
                    affected_fields=["log_id"]
                ))
            
            # 检查孤立的tracing_info记录
            orphaned_traces = await self.db_manager.execute_query("""
                SELECT ti.id, ti.log_id
                FROM tracing_info ti
                LEFT JOIN api_logs al ON ti.log_id = al.id
                WHERE al.id IS NULL
                  AND ti.created_at >= $1
                LIMIT $2
            """, (cutoff_date, batch_size))
            
            for record in orphaned_traces:
                issues.append(ConsistencyIssue(
                    issue_id=f"orphaned_trace_{record['id']}",
                    table_name="tracing_info",
                    record_id=str(record['id']),
                    issue_type=IssueType.ORPHANED_RECORD,
                    description=f"孤立的tracing_info记录，关联的api_logs记录不存在: log_id={record['log_id']}",
                    severity=IssueSeverity.HIGH,
                    detected_at=datetime.now(),
                    auto_fixable=False,
                    fix_suggestion="删除孤立记录或修复log_id关联",
                    affected_fields=["log_id"]
                ))
            
            total_count = len(orphaned_tokens) + len(orphaned_costs) + len(orphaned_traces)
            
        except Exception as e:
            self.logger.error(f"外键完整性检查失败: {str(e)}")
            raise
            
        return issues, total_count
    
    async def check_data_ranges(self, 
                              days_back: int = 7, 
                              batch_size: int = 1000) -> Tuple[List[ConsistencyIssue], int]:
        """检查数据范围异常"""
        issues = []
        total_count = 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # 检查异常的响应时间
            abnormal_durations = await self.db_manager.execute_query("""
                SELECT id, duration_ms, provider, model
                FROM api_logs
                WHERE created_at >= $1
                  AND (duration_ms < 0 OR duration_ms > 300000)  -- 负数或超过5分钟
                LIMIT $2
            """, (cutoff_date, batch_size))
            
            for record in abnormal_durations:
                issues.append(ConsistencyIssue(
                    issue_id=f"abnormal_duration_{record['id']}",
                    table_name="api_logs",
                    record_id=str(record['id']),
                    issue_type=IssueType.INVALID_DATA_RANGE,
                    description=f"异常的响应时间: {record['duration_ms']}ms",
                    severity=IssueSeverity.MEDIUM,
                    detected_at=datetime.now(),
                    auto_fixable=False,
                    fix_suggestion="检查响应时间计算逻辑",
                    affected_fields=["duration_ms"],
                    metadata={
                        "provider": record['provider'],
                        "model": record['model']
                    }
                ))
            
            # 检查异常的token数量
            abnormal_tokens = await self.db_manager.execute_query("""
                SELECT id, prompt_tokens, completion_tokens, total_tokens
                FROM api_logs
                WHERE created_at >= $1
                  AND (prompt_tokens > 100000 OR completion_tokens > 100000 OR total_tokens > 200000)
                LIMIT $2
            """, (cutoff_date, batch_size))
            
            for record in abnormal_tokens:
                issues.append(ConsistencyIssue(
                    issue_id=f"abnormal_tokens_{record['id']}",
                    table_name="api_logs",
                    record_id=str(record['id']),
                    issue_type=IssueType.INVALID_DATA_RANGE,
                    description=f"异常的token数量: prompt={record['prompt_tokens']}, "
                              f"completion={record['completion_tokens']}, total={record['total_tokens']}",
                    severity=IssueSeverity.LOW,
                    detected_at=datetime.now(),
                    auto_fixable=False,
                    fix_suggestion="检查token计算逻辑",
                    affected_fields=["prompt_tokens", "completion_tokens", "total_tokens"]
                ))
            
            total_count = len(abnormal_durations) + len(abnormal_tokens)
            
        except Exception as e:
            self.logger.error(f"数据范围检查失败: {str(e)}")
            raise
            
        return issues, total_count
    
    async def check_performance_anomalies(self, 
                                        days_back: int = 7, 
                                        batch_size: int = 1000) -> Tuple[List[ConsistencyIssue], int]:
        """检查性能异常"""
        issues = []
        total_count = 0
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # 计算平均响应时间
            avg_stats = await self.db_manager.execute_query("""
                SELECT 
                    provider,
                    model,
                    AVG(duration_ms) as avg_duration,
                    STDDEV(duration_ms) as stddev_duration,
                    COUNT(*) as request_count
                FROM api_logs
                WHERE created_at >= $1
                  AND duration_ms > 0
                GROUP BY provider, model
                HAVING COUNT(*) >= 10
            """, (cutoff_date,))
            
            # 查找异常慢的请求
            for stat in avg_stats:
                threshold = stat['avg_duration'] + 3 * (stat['stddev_duration'] or 0)
                
                slow_requests = await self.db_manager.execute_query("""
                    SELECT id, duration_ms, created_at
                    FROM api_logs
                    WHERE created_at >= $1
                      AND provider = $2
                      AND model = $3
                      AND duration_ms > $4
                    LIMIT 10
                """, (cutoff_date, stat['provider'], stat['model'], threshold))
                
                for request in slow_requests:
                    issues.append(ConsistencyIssue(
                        issue_id=f"performance_anomaly_{request['id']}",
                        table_name="api_logs",
                        record_id=str(request['id']),
                        issue_type=IssueType.PERFORMANCE_ANOMALY,
                        description=f"性能异常: 响应时间{request['duration_ms']:.2f}ms "
                                  f"超过阈值{threshold:.2f}ms",
                        severity=IssueSeverity.LOW,
                        detected_at=datetime.now(),
                        auto_fixable=False,
                        fix_suggestion="检查网络或服务性能",
                        affected_fields=["duration_ms"],
                        metadata={
                            "provider": stat['provider'],
                            "model": stat['model'],
                            "avg_duration": stat['avg_duration'],
                            "threshold": threshold
                        }
                    ))
                    total_count += 1
            
        except Exception as e:
            self.logger.error(f"性能异常检查失败: {str(e)}")
            raise
            
        return issues, total_count
    
    def _generate_summary(self, issues: List[ConsistencyIssue]) -> Dict[str, int]:
        """生成问题摘要统计"""
        summary = {
            'total': len(issues),
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'auto_fixable': 0
        }
        
        # 按严重程度统计
        for issue in issues:
            summary[issue.severity.value] += 1
            if issue.auto_fixable:
                summary['auto_fixable'] += 1
        
        # 按问题类型统计
        type_counts = {}
        for issue in issues:
            issue_type = issue.issue_type.value
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        
        summary.update(type_counts)
        
        return summary
    
    def _generate_recommendations(self, issues: List[ConsistencyIssue]) -> List[str]:
        """生成修复建议"""
        recommendations = []
        
        # 统计问题类型
        type_counts = {}
        for issue in issues:
            issue_type = issue.issue_type.value
            type_counts[issue_type] = type_counts.get(issue_type, 0) + 1
        
        # 生成针对性建议
        if type_counts.get('token_mismatch', 0) > 0:
            recommendations.append(
                f"发现{type_counts['token_mismatch']}个token数据不一致问题，"
                "建议检查token解析逻辑和数据同步机制"
            )
        
        if type_counts.get('cost_mismatch', 0) > 0:
            recommendations.append(
                f"发现{type_counts['cost_mismatch']}个成本数据不一致问题，"
                "建议检查成本计算逻辑和定价配置"
            )
        
        if type_counts.get('missing_tracing', 0) > 0:
            recommendations.append(
                f"发现{type_counts['missing_tracing']}个缺失追踪数据问题，"
                "建议检查追踪数据收集和存储流程"
            )
        
        if type_counts.get('orphaned_record', 0) > 0:
            recommendations.append(
                f"发现{type_counts['orphaned_record']}个孤立记录问题，"
                "建议执行数据清理和外键约束检查"
            )
        
        if type_counts.get('performance_anomaly', 0) > 0:
            recommendations.append(
                f"发现{type_counts['performance_anomaly']}个性能异常问题，"
                "建议检查网络连接和服务性能"
            )
        
        # 自动修复建议
        auto_fixable_count = sum(1 for issue in issues if issue.auto_fixable)
        if auto_fixable_count > 0:
            recommendations.append(
                f"有{auto_fixable_count}个问题可以自动修复，"
                "建议使用AutoCorrectionService进行自动修复"
            )
        
        return recommendations
    
    async def get_issues_by_severity(self, 
                                   report: ConsistencyReport, 
                                   severity: IssueSeverity) -> List[ConsistencyIssue]:
        """根据严重程度筛选问题"""
        return [issue for issue in report.issues if issue.severity == severity]
    
    async def get_issues_by_type(self, 
                               report: ConsistencyReport, 
                               issue_type: IssueType) -> List[ConsistencyIssue]:
        """根据问题类型筛选问题"""
        return [issue for issue in report.issues if issue.issue_type == issue_type]
    
    async def export_report(self, 
                          report: ConsistencyReport, 
                          format: str = "json") -> str:
        """导出检查报告
        
        Args:
            report: 一致性检查报告
            format: 导出格式 (json, csv)
            
        Returns:
            导出的报告内容
        """
        if format.lower() == "json":
            return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)
        elif format.lower() == "csv":
            # 简化的CSV导出
            lines = ["问题ID,表名,记录ID,问题类型,严重程度,描述,可自动修复"]
            for issue in report.issues:
                lines.append(f"{issue.issue_id},{issue.table_name},{issue.record_id},"
                           f"{issue.issue_type.value},{issue.severity.value},"
                           f'"{issue.description}",{issue.auto_fixable}')
            return "\n".join(lines)
        else:
            raise ValueError(f"不支持的导出格式: {format}")