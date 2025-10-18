"""
数据库约束管理器

负责管理数据库约束、外键检查和数据完整性约束
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ...database.async_manager import DatabaseManager


class ConstraintType(Enum):
    """约束类型枚举"""
    CHECK = "check"
    FOREIGN_KEY = "foreign_key"
    UNIQUE = "unique"
    PRIMARY_KEY = "primary_key"
    NOT_NULL = "not_null"


class ViolationSeverity(Enum):
    """违反严重程度枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ConstraintInfo:
    """约束信息数据类"""
    constraint_name: str
    table_name: str
    constraint_type: str  # 'CHECK', 'FOREIGN KEY', 'UNIQUE', 'PRIMARY KEY'
    column_names: List[str]
    definition: str
    is_enabled: bool
    is_valid: bool


@dataclass
class ConstraintViolation:
    """约束违反数据类"""
    violation_id: str
    constraint_name: str
    table_name: str
    record_id: str
    violation_type: str
    description: str
    detected_at: datetime
    fix_suggestion: Optional[str] = None


@dataclass
class ConstraintReport:
    """约束检查报告"""
    check_id: str
    check_timestamp: datetime
    constraints_checked: List[ConstraintInfo]
    violations_found: List[ConstraintViolation]
    check_duration_ms: float
    summary: Dict[str, int]


class DatabaseConstraintManager:
    """数据库约束管理器
    
    功能：
    1. 外键约束检查和管理
    2. 数据完整性约束验证
    3. 业务规则约束实施
    4. 约束违反检测和报告
    """
    
    def __init__(self, db_manager: DatabaseManager):
        """初始化数据库约束管理器
        
        Args:
            db_manager: 数据库管理器实例
        """
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # 定义核心表的约束配置
        self.core_tables = ['api_logs', 'token_usage', 'cost_info', 'tracing_info']
        
    async def check_all_constraints(self) -> ConstraintReport:
        """检查所有数据库约束
        
        Returns:
            ConstraintReport: 约束检查报告
        """
        start_time = datetime.now()
        check_id = f"constraint_check_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"开始执行数据库约束检查 - ID: {check_id}")
        
        try:
            # 1. 获取所有约束信息
            constraints = await self._get_all_constraints()
            
            # 2. 检查约束违反
            violations = []
            
            # 检查外键约束
            fk_violations = await self._check_foreign_key_constraints()
            violations.extend(fk_violations)
            
            # 检查CHECK约束
            check_violations = await self._check_check_constraints()
            violations.extend(check_violations)
            
            # 检查业务规则约束
            business_violations = await self._check_business_rule_constraints()
            violations.extend(business_violations)
            
            # 生成摘要
            summary = self._generate_constraint_summary(violations)
            
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            report = ConstraintReport(
                check_id=check_id,
                check_timestamp=start_time,
                constraints_checked=constraints,
                violations_found=violations,
                check_duration_ms=duration_ms,
                summary=summary
            )
            
            self.logger.info(f"数据库约束检查完成 - 检查约束数: {len(constraints)}, "
                           f"发现违反: {len(violations)}, 耗时: {duration_ms:.2f}ms")
            
            return report
            
        except Exception as e:
            self.logger.error(f"数据库约束检查失败: {str(e)}")
            raise
    
    async def _get_all_constraints(self) -> List[ConstraintInfo]:
        """获取所有约束信息"""
        constraints = []
        
        try:
            # 查询所有约束信息
            query = """
                SELECT 
                    tc.constraint_name,
                    tc.table_name,
                    tc.constraint_type,
                    array_agg(kcu.column_name) as column_names,
                    pg_get_constraintdef(pgc.oid) as definition,
                    pgc.convalidated as is_valid
                FROM information_schema.table_constraints tc
                LEFT JOIN information_schema.key_column_usage kcu 
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                LEFT JOIN pg_constraint pgc 
                    ON pgc.conname = tc.constraint_name
                WHERE tc.table_schema = 'public'
                    AND tc.table_name = ANY($1)
                GROUP BY tc.constraint_name, tc.table_name, tc.constraint_type, 
                         pgc.oid, pgc.convalidated
                ORDER BY tc.table_name, tc.constraint_type, tc.constraint_name
            """
            
            records = await self.db_manager.execute_query(query, (self.core_tables,))
            
            for record in records:
                constraints.append(ConstraintInfo(
                    constraint_name=record['constraint_name'],
                    table_name=record['table_name'],
                    constraint_type=record['constraint_type'],
                    column_names=record['column_names'] or [],
                    definition=record['definition'] or '',
                    is_enabled=True,  # PostgreSQL中约束默认启用
                    is_valid=record['is_valid'] if record['is_valid'] is not None else True
                ))
                    
        except Exception as e:
            self.logger.error(f"获取约束信息失败: {str(e)}")
            raise
            
        return constraints
    
    async def _check_foreign_key_constraints(self) -> List[ConstraintViolation]:
        """检查外键约束违反"""
        violations = []
        
        try:
            # 检查token_usage表的外键约束
            query1 = """
                SELECT tu.id, tu.log_id
                FROM token_usage tu
                LEFT JOIN api_logs al ON tu.log_id = al.id
                WHERE al.id IS NULL
                LIMIT 100
            """
            
            orphaned_tokens = await self.db_manager.execute_query(query1)
            for record in orphaned_tokens:
                violations.append(ConstraintViolation(
                    violation_id=f"fk_token_usage_{record['id']}",
                    constraint_name="token_usage_log_id_fkey",
                    table_name="token_usage",
                    record_id=str(record['id']),
                    violation_type="foreign_key_violation",
                    description=f"token_usage记录引用不存在的api_logs记录: log_id={record['log_id']}",
                    detected_at=datetime.now(),
                    fix_suggestion="删除孤立记录或修复log_id引用"
                ))
            
            # 检查cost_info表的外键约束
            query2 = """
                SELECT ci.id, ci.log_id
                FROM cost_info ci
                LEFT JOIN api_logs al ON ci.log_id = al.id
                WHERE al.id IS NULL
                LIMIT 100
            """
            
            orphaned_costs = await self.db_manager.execute_query(query2)
            for record in orphaned_costs:
                violations.append(ConstraintViolation(
                    violation_id=f"fk_cost_info_{record['id']}",
                    constraint_name="cost_info_log_id_fkey",
                    table_name="cost_info",
                    record_id=str(record['id']),
                    violation_type="foreign_key_violation",
                    description=f"cost_info记录引用不存在的api_logs记录: log_id={record['log_id']}",
                    detected_at=datetime.now(),
                    fix_suggestion="删除孤立记录或修复log_id引用"
                ))
            
            # 检查tracing_info表的外键约束
            query3 = """
                SELECT ti.id, ti.log_id
                FROM tracing_info ti
                LEFT JOIN api_logs al ON ti.log_id = al.id
                WHERE al.id IS NULL
                LIMIT 100
            """
            
            orphaned_traces = await self.db_manager.execute_query(query3)
            for record in orphaned_traces:
                violations.append(ConstraintViolation(
                    violation_id=f"fk_tracing_info_{record['id']}",
                    constraint_name="tracing_info_log_id_fkey",
                    table_name="tracing_info",
                    record_id=str(record['id']),
                    violation_type="foreign_key_violation",
                    description=f"tracing_info记录引用不存在的api_logs记录: log_id={record['log_id']}",
                    detected_at=datetime.now(),
                    fix_suggestion="删除孤立记录或修复log_id引用"
                ))
                    
        except Exception as e:
            self.logger.error(f"外键约束检查失败: {str(e)}")
            raise
            
        return violations
    
    async def _check_check_constraints(self) -> List[ConstraintViolation]:
        """检查CHECK约束违反"""
        violations = []
        
        try:
                
            # 检查token_usage表的CHECK约束
            
            # 1. token_usage_consistency: total_tokens = prompt_tokens + completion_tokens
            query1 = """
                SELECT id, prompt_tokens, completion_tokens, total_tokens
                FROM token_usage
                WHERE total_tokens != (prompt_tokens + completion_tokens)
                LIMIT 100
            """
            
            inconsistent_tokens = await self.db_manager.execute_query(query1)
            for record in inconsistent_tokens:
                violations.append(ConstraintViolation(
                    violation_id=f"check_token_consistency_{record['id']}",
                    constraint_name="token_usage_consistency",
                    table_name="token_usage",
                    record_id=str(record['id']),
                    violation_type="check_constraint_violation",
                    description=f"Token总数不一致: total={record['total_tokens']}, "
                              f"expected={record['prompt_tokens'] + record['completion_tokens']}",
                    detected_at=datetime.now(),
                    fix_suggestion=f"更新total_tokens为{record['prompt_tokens'] + record['completion_tokens']}"
                ))
                
            # 2. token_usage_confidence_range: confidence BETWEEN 0.0 AND 1.0
            query2 = """
                SELECT id, confidence
                FROM token_usage
                WHERE confidence IS NOT NULL 
                    AND (confidence < 0.0 OR confidence > 1.0)
                LIMIT 100
            """
            
            invalid_confidence = await self.db_manager.execute_query(query2)
            for record in invalid_confidence:
                violations.append(ConstraintViolation(
                    violation_id=f"check_confidence_range_{record['id']}",
                    constraint_name="token_usage_confidence_range",
                    table_name="token_usage",
                    record_id=str(record['id']),
                    violation_type="check_constraint_violation",
                    description=f"置信度超出范围: {record['confidence']} (应在0.0-1.0之间)",
                    detected_at=datetime.now(),
                    fix_suggestion="将confidence设置在0.0-1.0范围内"
                ))
                
            # 3. token_usage_positive_tokens: 所有token数量 >= 0
            query3 = """
                SELECT id, prompt_tokens, completion_tokens, total_tokens
                FROM token_usage
                WHERE prompt_tokens < 0 OR completion_tokens < 0 OR total_tokens < 0
                LIMIT 100
            """
            
            negative_tokens = await self.db_manager.execute_query(query3)
            for record in negative_tokens:
                violations.append(ConstraintViolation(
                    violation_id=f"check_positive_tokens_{record['id']}",
                    constraint_name="token_usage_positive_tokens",
                    table_name="token_usage",
                    record_id=str(record['id']),
                    violation_type="check_constraint_violation",
                    description=f"Token数量为负数: prompt={record['prompt_tokens']}, "
                              f"completion={record['completion_tokens']}, total={record['total_tokens']}",
                    detected_at=datetime.now(),
                    fix_suggestion="将负数token设置为0"
                ))
            
            # 检查cost_info表的CHECK约束
            
            # 1. cost_info_positive_costs: 所有成本 >= 0
            query4 = """
                SELECT id, input_cost, output_cost, total_cost
                FROM cost_info
                WHERE input_cost < 0 OR output_cost < 0 OR total_cost < 0
                LIMIT 100
            """
            
            negative_costs = await self.db_manager.execute_query(query4)
            for record in negative_costs:
                violations.append(ConstraintViolation(
                    violation_id=f"check_positive_costs_{record['id']}",
                    constraint_name="cost_info_positive_costs",
                    table_name="cost_info",
                    record_id=str(record['id']),
                    violation_type="check_constraint_violation",
                    description=f"成本为负数: input={record['input_cost']}, "
                              f"output={record['output_cost']}, total={record['total_cost']}",
                    detected_at=datetime.now(),
                    fix_suggestion="将负数成本设置为0.0"
                ))
            
            # 2. cost_info_consistency: total_cost = input_cost + output_cost
            query5 = """
                SELECT id, input_cost, output_cost, total_cost
                FROM cost_info
                WHERE ABS(total_cost - (input_cost + output_cost)) > 0.000001
                LIMIT 100
            """
            
            inconsistent_costs = await self.db_manager.execute_query(query5)
            for record in inconsistent_costs:
                expected_total = float(record['input_cost']) + float(record['output_cost'])
                violations.append(ConstraintViolation(
                    violation_id=f"check_cost_consistency_{record['id']}",
                    constraint_name="cost_info_consistency",
                    table_name="cost_info",
                    record_id=str(record['id']),
                    violation_type="check_constraint_violation",
                    description=f"成本总额不一致: total={record['total_cost']}, "
                              f"expected={expected_total:.6f}",
                    detected_at=datetime.now(),
                    fix_suggestion=f"更新total_cost为{expected_total:.6f}"
                ))
            
            # 检查tracing_info表的CHECK约束
            
            # 1. tracing_info_positive_duration: duration_ms >= 0
            query6 = """
                SELECT id, duration_ms
                FROM tracing_info
                WHERE duration_ms IS NOT NULL AND duration_ms < 0
                LIMIT 100
            """
            
            negative_durations = await self.db_manager.execute_query(query6)
            for record in negative_durations:
                violations.append(ConstraintViolation(
                    violation_id=f"check_positive_duration_{record['id']}",
                    constraint_name="tracing_info_positive_duration",
                    table_name="tracing_info",
                    record_id=str(record['id']),
                    violation_type="check_constraint_violation",
                    description=f"持续时间为负数: {record['duration_ms']}ms",
                    detected_at=datetime.now(),
                    fix_suggestion="将duration_ms设置为0"
                ))
                    
        except Exception as e:
            self.logger.error(f"CHECK约束检查失败: {str(e)}")
            raise
            
        return violations
    
    async def _check_business_rule_constraints(self) -> List[ConstraintViolation]:
        """检查业务规则约束违反"""
        violations = []
        
        try:
            # 业务规则1: 每个api_logs记录应该有对应的token_usage记录
            query1 = """
                SELECT al.id
                FROM api_logs al
                LEFT JOIN token_usage tu ON al.id = tu.log_id
                WHERE tu.log_id IS NULL
                    AND al.created_at >= NOW() - INTERVAL '7 days'
                LIMIT 50
            """
            
            missing_tokens = await self.db_manager.execute_query(query1)
            for record in missing_tokens:
                violations.append(ConstraintViolation(
                    violation_id=f"business_missing_token_{record['id']}",
                    constraint_name="business_rule_token_required",
                    table_name="api_logs",
                    record_id=str(record['id']),
                    violation_type="business_rule_violation",
                    description=f"API日志缺少对应的token使用记录: log_id={record['id']}",
                    detected_at=datetime.now(),
                    fix_suggestion="为该API日志创建token_usage记录"
                ))
                
            # 业务规则2: 每个api_logs记录应该有对应的cost_info记录
            query2 = """
                SELECT al.id
                FROM api_logs al
                LEFT JOIN cost_info ci ON al.id = ci.log_id
                WHERE ci.log_id IS NULL
                    AND al.created_at >= NOW() - INTERVAL '7 days'
                LIMIT 50
            """
            
            missing_costs = await self.db_manager.execute_query(query2)
            for record in missing_costs:
                violations.append(ConstraintViolation(
                    violation_id=f"business_missing_cost_{record['id']}",
                    constraint_name="business_rule_cost_required",
                    table_name="api_logs",
                    record_id=str(record['id']),
                    violation_type="business_rule_violation",
                    description=f"API日志缺少对应的成本信息记录: log_id={record['id']}",
                    detected_at=datetime.now(),
                    fix_suggestion="为该API日志创建cost_info记录"
                ))
                
            # 业务规则3: 成功的API调用(status_code=200)应该有有效的token数据
            query3 = """
                SELECT al.id, tu.total_tokens
                FROM api_logs al
                JOIN token_usage tu ON al.id = tu.log_id
                WHERE al.status_code = 200
                    AND (tu.total_tokens IS NULL OR tu.total_tokens <= 0)
                    AND al.created_at >= NOW() - INTERVAL '7 days'
                LIMIT 50
            """
            
            invalid_success_tokens = await self.db_manager.execute_query(query3)
            for record in invalid_success_tokens:
                violations.append(ConstraintViolation(
                    violation_id=f"business_invalid_success_token_{record['id']}",
                    constraint_name="business_rule_success_token_valid",
                    table_name="api_logs",
                    record_id=str(record['id']),
                    violation_type="business_rule_violation",
                    description=f"成功的API调用缺少有效的token数据: log_id={record['id']}, "
                              f"total_tokens={record['total_tokens']}",
                    detected_at=datetime.now(),
                    fix_suggestion="重新解析或估算token使用量"
                ))
                
            # 业务规则4: 追踪信息的时间戳应该与API日志时间戳接近
            query4 = """
                SELECT al.id, al.timestamp as api_time, ti.start_time as trace_time
                FROM api_logs al
                JOIN tracing_info ti ON al.id = ti.log_id
                WHERE ABS(EXTRACT(EPOCH FROM (al.timestamp - ti.start_time))) > 300
                    AND al.created_at >= NOW() - INTERVAL '7 days'
                LIMIT 50
            """
            
            time_mismatches = await self.db_manager.execute_query(query4)
            for record in time_mismatches:
                violations.append(ConstraintViolation(
                    violation_id=f"business_time_mismatch_{record['id']}",
                    constraint_name="business_rule_time_consistency",
                    table_name="tracing_info",
                    record_id=str(record['id']),
                    violation_type="business_rule_violation",
                    description=f"追踪时间与API时间不一致: api_time={record['api_time']}, "
                              f"trace_time={record['trace_time']}",
                    detected_at=datetime.now(),
                    fix_suggestion="同步追踪时间戳与API时间戳"
                ))
                    
        except Exception as e:
            self.logger.error(f"业务规则约束检查失败: {str(e)}")
            raise
            
        return violations
    
    def _generate_constraint_summary(self, violations: List[ConstraintViolation]) -> Dict[str, int]:
        """生成约束检查摘要"""
        summary = {
            'total_violations': len(violations),
            'foreign_key_violations': 0,
            'check_constraint_violations': 0,
            'business_rule_violations': 0
        }
        
        for violation in violations:
            if violation.violation_type == 'foreign_key_violation':
                summary['foreign_key_violations'] += 1
            elif violation.violation_type == 'check_constraint_violation':
                summary['check_constraint_violations'] += 1
            elif violation.violation_type == 'business_rule_violation':
                summary['business_rule_violations'] += 1
        
        return summary
    
    async def create_missing_constraints(self) -> bool:
        """创建缺失的约束
        
        Returns:
            bool: 是否成功创建约束
        """
        try:
            success_count = 0
            total_constraints = 0
            
            # 创建外键约束（如果不存在）
            foreign_key_sqls = [
                """
                ALTER TABLE token_usage 
                ADD CONSTRAINT IF NOT EXISTS token_usage_log_id_fkey 
                FOREIGN KEY (log_id) REFERENCES api_logs(id) ON DELETE CASCADE
                """,
                """
                ALTER TABLE cost_info 
                ADD CONSTRAINT IF NOT EXISTS cost_info_log_id_fkey 
                FOREIGN KEY (log_id) REFERENCES api_logs(id) ON DELETE CASCADE
                """,
                """
                ALTER TABLE tracing_info 
                ADD CONSTRAINT IF NOT EXISTS tracing_info_log_id_fkey 
                FOREIGN KEY (log_id) REFERENCES api_logs(id) ON DELETE CASCADE
                """
            ]
            
            for sql in foreign_key_sqls:
                total_constraints += 1
                try:
                    await self.db_manager.execute_query(sql)
                    success_count += 1
                    self.logger.info(f"成功创建外键约束")
                except Exception as e:
                    if "already exists" not in str(e):
                        self.logger.warning(f"创建外键约束失败: {str(e)}")
                        # 如果不是"已存在"错误，则认为是真正的失败
                        return False
                    else:
                        # 约束已存在，也算成功
                        success_count += 1
            
            # 创建CHECK约束（如果不存在）
            check_constraint_sqls = [
                """
                ALTER TABLE token_usage 
                ADD CONSTRAINT IF NOT EXISTS token_usage_consistency_check 
                CHECK (total_tokens = prompt_tokens + completion_tokens)
                """,
                """
                ALTER TABLE cost_info 
                ADD CONSTRAINT IF NOT EXISTS cost_info_consistency_check 
                CHECK (ABS(total_cost - (input_cost + output_cost)) < 0.000001)
                """,
                """
                ALTER TABLE tracing_info 
                ADD CONSTRAINT IF NOT EXISTS tracing_info_duration_check 
                CHECK (duration_ms IS NULL OR duration_ms >= 0)
                """
            ]
            
            for sql in check_constraint_sqls:
                total_constraints += 1
                try:
                    await self.db_manager.execute_query(sql)
                    success_count += 1
                    self.logger.info(f"成功创建CHECK约束")
                except Exception as e:
                    if "already exists" not in str(e):
                        self.logger.warning(f"创建CHECK约束失败: {str(e)}")
                        # 如果不是"已存在"错误，则认为是真正的失败
                        return False
                    else:
                        # 约束已存在，也算成功
                        success_count += 1
            
            return success_count > 0
                
        except Exception as e:
            self.logger.error(f"创建约束失败: {str(e)}")
            return False
    
    async def validate_table_constraints(self, table_name: str) -> bool:
        """验证特定表的约束
        
        Args:
            table_name: 表名
            
        Returns:
            bool: 表约束是否全部有效
        """
        try:
            # 检查外键约束违反
            fk_violations = await self.check_foreign_key_violations()
            table_fk_violations = [v for v in fk_violations if v.table_name == table_name]
            
            # 检查数据完整性约束违反
            integrity_violations = await self.check_data_integrity_violations()
            table_integrity_violations = [v for v in integrity_violations if v.table_name == table_name]
            
            # 检查业务规则约束违反
            business_violations = await self.check_business_rule_violations()
            table_business_violations = [v for v in business_violations if v.table_name == table_name]
            
            # 如果没有任何违反，则表约束有效
            total_violations = len(table_fk_violations) + len(table_integrity_violations) + len(table_business_violations)
            return total_violations == 0
            
        except Exception as e:
            self.logger.error(f"验证表约束失败: {str(e)}")
            return False
    
    async def validate_constraint(self, constraint_name: str) -> bool:
        """验证特定约束是否有效
        
        Args:
            constraint_name: 约束名称
            
        Returns:
            bool: 约束是否有效
        """
        try:
            query = """
                SELECT convalidated
                FROM pg_constraint
                WHERE conname = $1
            """
            
            result = await self.db_manager.execute_query(query, (constraint_name,))
            return result[0]['convalidated'] if result else False
                
        except Exception as e:
            self.logger.error(f"验证约束失败: {str(e)}")
            return False
    
    def get_constraint_violations_by_table(self, 
                                         report: ConstraintReport, 
                                         table_name: str) -> List[ConstraintViolation]:
        """根据表名筛选约束违反"""
        return [v for v in report.violations_found if v.table_name == table_name]

    async def get_constraint_info(self) -> List[ConstraintInfo]:
        """获取约束信息
        
        Returns:
            List[ConstraintInfo]: 约束信息列表
        """
        return await self._get_all_constraints()
    
    async def check_foreign_key_violations(self) -> List[ConstraintViolation]:
        """检查外键约束违反
        
        Returns:
            List[ConstraintViolation]: 外键约束违反列表
        """
        return await self._check_foreign_key_constraints()
    
    async def check_business_rule_violations(self) -> List[ConstraintViolation]:
        """检查业务规则约束违反
        
        Returns:
            List[ConstraintViolation]: 业务规则约束违反列表
        """
        return await self._check_business_rule_constraints()
    
    async def check_data_integrity_violations(self) -> List[ConstraintViolation]:
        """检查数据完整性约束违反
        
        Returns:
            List[ConstraintViolation]: 数据完整性约束违反列表
        """
        violations = []
        
        # 检查CHECK约束
        check_violations = await self._check_check_constraints()
        violations.extend(check_violations)
        
        return violations
    
    async def generate_constraint_report(self, violations: List[ConstraintViolation]) -> ConstraintReport:
        """生成约束报告
        
        Args:
            violations: 约束违反列表
            
        Returns:
            ConstraintReport: 约束检查报告
        """
        check_id = f"constraint_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        check_timestamp = datetime.now()
        
        # 获取所有约束信息
        constraints_checked = await self.get_constraint_info()
        
        # 生成摘要
        summary = self._generate_constraint_summary(violations)
        
        return ConstraintReport(
            check_id=check_id,
            check_timestamp=check_timestamp,
            constraints_checked=constraints_checked,
            violations_found=violations,
            check_duration_ms=0.0,  # 这里可以根据需要计算实际时间
            summary=summary
        )