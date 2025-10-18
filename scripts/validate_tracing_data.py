#!/usr/bin/env python3
"""
HarborAI 追踪系统数据验证脚本

此脚本负责验证追踪系统的数据完整性和一致性，包括：
- 数据库表结构验证
- 数据完整性检查
- 性能指标验证
- 追踪链路完整性验证
- 数据质量报告生成

功能特性：
- 全面的数据验证
- 性能基准测试
- 数据质量评分
- 详细的验证报告
- 自动修复建议

作者: HarborAI团队
创建时间: 2025-01-15
版本: v1.0.0
"""

import asyncio
import asyncpg
import argparse
import json
import sys
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import statistics

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from harborai.utils.exceptions import DatabaseError, ValidationError


@dataclass
class ValidationResult:
    """验证结果"""
    category: str
    test_name: str
    status: str  # "pass", "fail", "warning"
    message: str
    details: Dict[str, Any] = None
    score: float = 0.0  # 0-100分
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class DataQualityMetrics:
    """数据质量指标"""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    duplicate_records: int = 0
    orphaned_records: int = 0
    data_completeness: float = 0.0
    data_accuracy: float = 0.0
    data_consistency: float = 0.0
    overall_score: float = 0.0


@dataclass
class PerformanceMetrics:
    """性能指标"""
    avg_query_time_ms: float = 0.0
    max_query_time_ms: float = 0.0
    min_query_time_ms: float = 0.0
    p95_query_time_ms: float = 0.0
    throughput_qps: float = 0.0
    index_efficiency: float = 0.0
    storage_efficiency: float = 0.0


@dataclass
class ValidationReport:
    """验证报告"""
    timestamp: datetime
    database_info: Dict[str, Any]
    schema_validation: List[ValidationResult]
    data_validation: List[ValidationResult]
    performance_validation: List[ValidationResult]
    data_quality: DataQualityMetrics
    performance_metrics: PerformanceMetrics
    overall_score: float = 0.0
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class TracingDataValidator:
    """追踪数据验证器"""
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        初始化验证器
        
        Args:
            db_config: 数据库连接配置
        """
        self.db_config = db_config
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.validation_results: List[ValidationResult] = []
        
    async def connect(self) -> None:
        """建立数据库连接"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            print(f"✅ 数据库连接成功: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
        except Exception as e:
            raise DatabaseError(f"数据库连接失败: {e}")
    
    async def disconnect(self) -> None:
        """关闭数据库连接"""
        if self.connection_pool:
            await self.connection_pool.close()
            print("✅ 数据库连接已关闭")
    
    async def validate_schema(self) -> List[ValidationResult]:
        """验证数据库模式"""
        results = []
        
        if not self.connection_pool:
            raise DatabaseError("数据库连接未建立")
        
        async with self.connection_pool.acquire() as connection:
            # 1. 验证必需的表存在
            required_tables = ["migration_history", "tracing_info"]
            
            for table_name in required_tables:
                exists = await connection.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = $1
                    )
                """, table_name)
                
                if exists:
                    results.append(ValidationResult(
                        category="schema",
                        test_name=f"table_{table_name}_exists",
                        status="pass",
                        message=f"表 {table_name} 存在",
                        score=100.0
                    ))
                else:
                    results.append(ValidationResult(
                        category="schema",
                        test_name=f"table_{table_name}_exists",
                        status="fail",
                        message=f"表 {table_name} 不存在",
                        score=0.0
                    ))
            
            # 2. 验证tracing_info表结构
            if await self._table_exists(connection, "tracing_info"):
                columns = await connection.fetch("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'tracing_info'
                    ORDER BY ordinal_position
                """)
                
                required_columns = {
                    "id": "bigint",
                    "hb_trace_id": "character varying",
                    "otel_trace_id": "character varying",
                    "span_id": "character varying",
                    "operation_name": "character varying",
                    "service_name": "character varying",
                    "start_time": "timestamp with time zone",
                    "status": "character varying",
                    "tags": "jsonb",
                    "logs": "jsonb"
                }
                
                existing_columns = {col["column_name"]: col["data_type"] for col in columns}
                
                for col_name, expected_type in required_columns.items():
                    if col_name in existing_columns:
                        actual_type = existing_columns[col_name]
                        if expected_type in actual_type or actual_type in expected_type:
                            results.append(ValidationResult(
                                category="schema",
                                test_name=f"column_{col_name}_type",
                                status="pass",
                                message=f"列 {col_name} 类型正确 ({actual_type})",
                                score=100.0
                            ))
                        else:
                            results.append(ValidationResult(
                                category="schema",
                                test_name=f"column_{col_name}_type",
                                status="fail",
                                message=f"列 {col_name} 类型错误: 期望 {expected_type}, 实际 {actual_type}",
                                score=0.0
                            ))
                    else:
                        results.append(ValidationResult(
                            category="schema",
                            test_name=f"column_{col_name}_exists",
                            status="fail",
                            message=f"缺少必需列 {col_name}",
                            score=0.0
                        ))
            
            # 3. 验证索引
            indexes = await connection.fetch("""
                SELECT indexname, tablename
                FROM pg_indexes
                WHERE schemaname = 'public' AND tablename = 'tracing_info'
            """)
            
            required_indexes = [
                "idx_tracing_info_hb_trace_id",
                "idx_tracing_info_otel_trace_id",
                "idx_tracing_info_start_time",
                "idx_tracing_info_status"
            ]
            
            existing_indexes = [idx["indexname"] for idx in indexes]
            
            for idx_name in required_indexes:
                if idx_name in existing_indexes:
                    results.append(ValidationResult(
                        category="schema",
                        test_name=f"index_{idx_name}_exists",
                        status="pass",
                        message=f"索引 {idx_name} 存在",
                        score=100.0
                    ))
                else:
                    results.append(ValidationResult(
                        category="schema",
                        test_name=f"index_{idx_name}_exists",
                        status="warning",
                        message=f"建议的索引 {idx_name} 不存在",
                        score=70.0
                    ))
            
            # 4. 验证约束
            constraints = await connection.fetch("""
                SELECT constraint_name, constraint_type
                FROM information_schema.table_constraints
                WHERE table_schema = 'public' AND table_name = 'tracing_info'
            """)
            
            constraint_types = [c["constraint_type"] for c in constraints]
            
            if "PRIMARY KEY" in constraint_types:
                results.append(ValidationResult(
                    category="schema",
                    test_name="primary_key_exists",
                    status="pass",
                    message="主键约束存在",
                    score=100.0
                ))
            else:
                results.append(ValidationResult(
                    category="schema",
                    test_name="primary_key_exists",
                    status="fail",
                    message="缺少主键约束",
                    score=0.0
                ))
            
            # 5. 验证视图
            views = await connection.fetch("""
                SELECT table_name
                FROM information_schema.views
                WHERE table_schema = 'public' AND table_name LIKE 'v_%'
            """)
            
            expected_views = ["v_active_traces", "v_tracing_stats", "v_error_traces"]
            existing_views = [v["table_name"] for v in views]
            
            for view_name in expected_views:
                if view_name in existing_views:
                    results.append(ValidationResult(
                        category="schema",
                        test_name=f"view_{view_name}_exists",
                        status="pass",
                        message=f"视图 {view_name} 存在",
                        score=100.0
                    ))
                else:
                    results.append(ValidationResult(
                        category="schema",
                        test_name=f"view_{view_name}_exists",
                        status="warning",
                        message=f"建议的视图 {view_name} 不存在",
                        score=80.0
                    ))
        
        return results
    
    async def validate_data_integrity(self) -> List[ValidationResult]:
        """验证数据完整性"""
        results = []
        
        if not self.connection_pool:
            raise DatabaseError("数据库连接未建立")
        
        async with self.connection_pool.acquire() as connection:
            # 1. 检查基本数据统计
            total_records = await connection.fetchval(
                "SELECT COUNT(*) FROM tracing_info"
            )
            
            results.append(ValidationResult(
                category="data",
                test_name="total_records_count",
                status="pass" if total_records >= 0 else "fail",
                message=f"总记录数: {total_records}",
                details={"count": total_records},
                score=100.0 if total_records >= 0 else 0.0
            ))
            
            if total_records == 0:
                results.append(ValidationResult(
                    category="data",
                    test_name="data_exists",
                    status="warning",
                    message="数据库中没有追踪数据",
                    score=50.0
                ))
                return results
            
            # 2. 检查必需字段的完整性
            required_fields = ["hb_trace_id", "otel_trace_id", "operation_name", "service_name", "start_time", "status"]
            
            for field in required_fields:
                null_count = await connection.fetchval(
                    f"SELECT COUNT(*) FROM tracing_info WHERE {field} IS NULL"
                )
                
                completeness = ((total_records - null_count) / total_records * 100) if total_records > 0 else 0
                
                if completeness == 100:
                    status = "pass"
                    score = 100.0
                elif completeness >= 95:
                    status = "warning"
                    score = 80.0
                else:
                    status = "fail"
                    score = 50.0
                
                results.append(ValidationResult(
                    category="data",
                    test_name=f"field_{field}_completeness",
                    status=status,
                    message=f"字段 {field} 完整性: {completeness:.1f}%",
                    details={"completeness": completeness, "null_count": null_count},
                    score=score
                ))
            
            # 3. 检查数据格式有效性
            # 检查otel_trace_id格式（32位十六进制）
            invalid_otel_trace_ids = await connection.fetchval("""
                SELECT COUNT(*) FROM tracing_info 
                WHERE otel_trace_id !~ '^[0-9a-f]{32}$'
            """)
            
            otel_validity = ((total_records - invalid_otel_trace_ids) / total_records * 100) if total_records > 0 else 0
            
            results.append(ValidationResult(
                category="data",
                test_name="otel_trace_id_format",
                status="pass" if otel_validity == 100 else "fail",
                message=f"OpenTelemetry Trace ID 格式有效性: {otel_validity:.1f}%",
                details={"invalid_count": invalid_otel_trace_ids},
                score=100.0 if otel_validity == 100 else 60.0
            ))
            
            # 检查span_id格式（16位十六进制）
            invalid_span_ids = await connection.fetchval("""
                SELECT COUNT(*) FROM tracing_info 
                WHERE span_id !~ '^[0-9a-f]{16}$'
            """)
            
            span_validity = ((total_records - invalid_span_ids) / total_records * 100) if total_records > 0 else 0
            
            results.append(ValidationResult(
                category="data",
                test_name="span_id_format",
                status="pass" if span_validity == 100 else "fail",
                message=f"Span ID 格式有效性: {span_validity:.1f}%",
                details={"invalid_count": invalid_span_ids},
                score=100.0 if span_validity == 100 else 60.0
            ))
            
            # 4. 检查状态值有效性
            valid_statuses = ["pending", "active", "completed", "error"]
            invalid_statuses = await connection.fetchval("""
                SELECT COUNT(*) FROM tracing_info 
                WHERE status NOT IN ('pending', 'active', 'completed', 'error')
            """)
            
            status_validity = ((total_records - invalid_statuses) / total_records * 100) if total_records > 0 else 0
            
            results.append(ValidationResult(
                category="data",
                test_name="status_values_validity",
                status="pass" if status_validity == 100 else "fail",
                message=f"状态值有效性: {status_validity:.1f}%",
                details={"invalid_count": invalid_statuses, "valid_statuses": valid_statuses},
                score=100.0 if status_validity == 100 else 50.0
            ))
            
            # 5. 检查时间逻辑一致性
            invalid_time_logic = await connection.fetchval("""
                SELECT COUNT(*) FROM tracing_info 
                WHERE end_time IS NOT NULL AND end_time < start_time
            """)
            
            time_consistency = ((total_records - invalid_time_logic) / total_records * 100) if total_records > 0 else 0
            
            results.append(ValidationResult(
                category="data",
                test_name="time_logic_consistency",
                status="pass" if time_consistency == 100 else "fail",
                message=f"时间逻辑一致性: {time_consistency:.1f}%",
                details={"invalid_count": invalid_time_logic},
                score=100.0 if time_consistency == 100 else 40.0
            ))
            
            # 6. 检查重复记录
            duplicate_hb_trace_ids = await connection.fetchval("""
                SELECT COUNT(*) - COUNT(DISTINCT hb_trace_id) FROM tracing_info
            """)
            
            results.append(ValidationResult(
                category="data",
                test_name="duplicate_hb_trace_ids",
                status="pass" if duplicate_hb_trace_ids == 0 else "warning",
                message=f"重复的HB Trace ID: {duplicate_hb_trace_ids}",
                details={"duplicate_count": duplicate_hb_trace_ids},
                score=100.0 if duplicate_hb_trace_ids == 0 else 70.0
            ))
            
            # 7. 检查JSON字段有效性
            invalid_tags = await connection.fetchval("""
                SELECT COUNT(*) FROM tracing_info 
                WHERE tags IS NOT NULL AND NOT (tags::text ~ '^{.*}$')
            """)
            
            invalid_logs = await connection.fetchval("""
                SELECT COUNT(*) FROM tracing_info 
                WHERE logs IS NOT NULL AND NOT (logs::text ~ '^\\[.*\\]$')
            """)
            
            json_validity = ((total_records - invalid_tags - invalid_logs) / total_records * 100) if total_records > 0 else 0
            
            results.append(ValidationResult(
                category="data",
                test_name="json_fields_validity",
                status="pass" if json_validity == 100 else "fail",
                message=f"JSON字段有效性: {json_validity:.1f}%",
                details={"invalid_tags": invalid_tags, "invalid_logs": invalid_logs},
                score=100.0 if json_validity == 100 else 60.0
            ))
            
            # 8. 检查数据分布
            status_distribution = await connection.fetch("""
                SELECT status, COUNT(*) as count
                FROM tracing_info
                GROUP BY status
                ORDER BY count DESC
            """)
            
            status_dist = {row["status"]: row["count"] for row in status_distribution}
            
            results.append(ValidationResult(
                category="data",
                test_name="status_distribution",
                status="pass",
                message="状态分布统计",
                details={"distribution": status_dist},
                score=100.0
            ))
            
            # 9. 检查最近数据活跃度
            recent_records = await connection.fetchval("""
                SELECT COUNT(*) FROM tracing_info 
                WHERE created_at >= NOW() - INTERVAL '24 hours'
            """)
            
            activity_score = min(100.0, (recent_records / max(1, total_records * 0.1)) * 100)
            
            results.append(ValidationResult(
                category="data",
                test_name="recent_activity",
                status="pass" if recent_records > 0 else "warning",
                message=f"24小时内新增记录: {recent_records}",
                details={"recent_count": recent_records},
                score=activity_score
            ))
        
        return results
    
    async def validate_performance(self) -> Tuple[List[ValidationResult], PerformanceMetrics]:
        """验证性能指标"""
        results = []
        metrics = PerformanceMetrics()
        
        if not self.connection_pool:
            raise DatabaseError("数据库连接未建立")
        
        # 性能测试查询
        test_queries = [
            ("simple_count", "SELECT COUNT(*) FROM tracing_info"),
            ("hb_trace_id_lookup", "SELECT * FROM tracing_info WHERE hb_trace_id = 'test_trace_001' LIMIT 1"),
            ("status_filter", "SELECT COUNT(*) FROM tracing_info WHERE status = 'completed'"),
            ("time_range_query", "SELECT COUNT(*) FROM tracing_info WHERE start_time >= NOW() - INTERVAL '1 hour'"),
            ("json_tag_query", "SELECT COUNT(*) FROM tracing_info WHERE tags->>'model' = 'gpt-4'"),
            ("complex_aggregation", """
                SELECT service_name, operation_name, COUNT(*), AVG(duration_ms)
                FROM tracing_info 
                WHERE start_time >= NOW() - INTERVAL '24 hours'
                GROUP BY service_name, operation_name
            """)
        ]
        
        query_times = []
        
        async with self.connection_pool.acquire() as connection:
            for query_name, query_sql in test_queries:
                times = []
                
                # 执行每个查询3次取平均值
                for i in range(3):
                    start_time = time.time()
                    try:
                        await connection.fetch(query_sql)
                        end_time = time.time()
                        query_time_ms = (end_time - start_time) * 1000
                        times.append(query_time_ms)
                    except Exception as e:
                        results.append(ValidationResult(
                            category="performance",
                            test_name=f"query_{query_name}",
                            status="fail",
                            message=f"查询 {query_name} 执行失败: {e}",
                            score=0.0
                        ))
                        continue
                
                if times:
                    avg_time = statistics.mean(times)
                    query_times.extend(times)
                    
                    # 性能评分（基于查询时间）
                    if avg_time < 10:  # 10ms以下
                        score = 100.0
                        status = "pass"
                    elif avg_time < 50:  # 50ms以下
                        score = 80.0
                        status = "pass"
                    elif avg_time < 200:  # 200ms以下
                        score = 60.0
                        status = "warning"
                    else:
                        score = 40.0
                        status = "fail"
                    
                    results.append(ValidationResult(
                        category="performance",
                        test_name=f"query_{query_name}_performance",
                        status=status,
                        message=f"查询 {query_name} 平均耗时: {avg_time:.2f}ms",
                        details={"avg_time_ms": avg_time, "times": times},
                        score=score
                    ))
            
            # 计算整体性能指标
            if query_times:
                metrics.avg_query_time_ms = statistics.mean(query_times)
                metrics.max_query_time_ms = max(query_times)
                metrics.min_query_time_ms = min(query_times)
                
                if len(query_times) > 1:
                    sorted_times = sorted(query_times)
                    p95_index = int(len(sorted_times) * 0.95)
                    metrics.p95_query_time_ms = sorted_times[p95_index]
                
                # 估算吞吐量（基于平均查询时间）
                if metrics.avg_query_time_ms > 0:
                    metrics.throughput_qps = 1000 / metrics.avg_query_time_ms
            
            # 检查索引使用情况
            index_usage = await connection.fetch("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes 
                WHERE tablename = 'tracing_info'
            """)
            
            total_index_reads = sum(row["idx_tup_read"] or 0 for row in index_usage)
            
            if total_index_reads > 0:
                metrics.index_efficiency = 100.0
                results.append(ValidationResult(
                    category="performance",
                    test_name="index_usage",
                    status="pass",
                    message=f"索引使用良好，总读取次数: {total_index_reads}",
                    details={"total_reads": total_index_reads},
                    score=100.0
                ))
            else:
                metrics.index_efficiency = 50.0
                results.append(ValidationResult(
                    category="performance",
                    test_name="index_usage",
                    status="warning",
                    message="索引使用情况不明确或较少",
                    score=50.0
                ))
            
            # 检查表大小和存储效率
            table_size = await connection.fetchval("""
                SELECT pg_total_relation_size('tracing_info')
            """)
            
            row_count = await connection.fetchval("SELECT COUNT(*) FROM tracing_info")
            
            if row_count > 0:
                avg_row_size = table_size / row_count
                
                # 评估存储效率（基于平均行大小）
                if avg_row_size < 1024:  # 1KB以下
                    storage_score = 100.0
                    storage_status = "pass"
                elif avg_row_size < 4096:  # 4KB以下
                    storage_score = 80.0
                    storage_status = "pass"
                elif avg_row_size < 10240:  # 10KB以下
                    storage_score = 60.0
                    storage_status = "warning"
                else:
                    storage_score = 40.0
                    storage_status = "warning"
                
                metrics.storage_efficiency = storage_score
                
                results.append(ValidationResult(
                    category="performance",
                    test_name="storage_efficiency",
                    status=storage_status,
                    message=f"存储效率: 平均行大小 {avg_row_size:.0f} 字节",
                    details={
                        "table_size_bytes": table_size,
                        "row_count": row_count,
                        "avg_row_size_bytes": avg_row_size
                    },
                    score=storage_score
                ))
        
        return results, metrics
    
    async def calculate_data_quality_metrics(self) -> DataQualityMetrics:
        """计算数据质量指标"""
        metrics = DataQualityMetrics()
        
        if not self.connection_pool:
            raise DatabaseError("数据库连接未建立")
        
        async with self.connection_pool.acquire() as connection:
            # 总记录数
            metrics.total_records = await connection.fetchval(
                "SELECT COUNT(*) FROM tracing_info"
            )
            
            if metrics.total_records == 0:
                return metrics
            
            # 有效记录数（所有必需字段都不为空）
            metrics.valid_records = await connection.fetchval("""
                SELECT COUNT(*) FROM tracing_info 
                WHERE hb_trace_id IS NOT NULL 
                  AND otel_trace_id IS NOT NULL 
                  AND operation_name IS NOT NULL 
                  AND service_name IS NOT NULL 
                  AND start_time IS NOT NULL 
                  AND status IS NOT NULL
            """)
            
            metrics.invalid_records = metrics.total_records - metrics.valid_records
            
            # 重复记录数
            metrics.duplicate_records = await connection.fetchval("""
                SELECT COUNT(*) - COUNT(DISTINCT hb_trace_id) FROM tracing_info
            """)
            
            # 孤立记录数（这里简化为状态异常的记录）
            metrics.orphaned_records = await connection.fetchval("""
                SELECT COUNT(*) FROM tracing_info 
                WHERE status NOT IN ('pending', 'active', 'completed', 'error')
            """)
            
            # 计算质量指标
            if metrics.total_records > 0:
                metrics.data_completeness = (metrics.valid_records / metrics.total_records) * 100
                
                # 数据准确性（基于格式验证）
                format_valid_records = await connection.fetchval("""
                    SELECT COUNT(*) FROM tracing_info 
                    WHERE otel_trace_id ~ '^[0-9a-f]{32}$'
                      AND span_id ~ '^[0-9a-f]{16}$'
                      AND status IN ('pending', 'active', 'completed', 'error')
                      AND (end_time IS NULL OR end_time >= start_time)
                """)
                
                metrics.data_accuracy = (format_valid_records / metrics.total_records) * 100
                
                # 数据一致性（基于逻辑验证）
                consistent_records = await connection.fetchval("""
                    SELECT COUNT(*) FROM tracing_info 
                    WHERE (status = 'completed' AND end_time IS NOT NULL AND duration_ms IS NOT NULL)
                       OR (status IN ('pending', 'active') AND end_time IS NULL)
                       OR (status = 'error')
                """)
                
                metrics.data_consistency = (consistent_records / metrics.total_records) * 100
                
                # 总体评分
                metrics.overall_score = (
                    metrics.data_completeness * 0.4 +
                    metrics.data_accuracy * 0.3 +
                    metrics.data_consistency * 0.3
                )
        
        return metrics
    
    async def generate_recommendations(self, validation_results: List[ValidationResult], 
                                     data_quality: DataQualityMetrics,
                                     performance_metrics: PerformanceMetrics) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 基于验证结果的建议
        failed_tests = [r for r in validation_results if r.status == "fail"]
        warning_tests = [r for r in validation_results if r.status == "warning"]
        
        if failed_tests:
            recommendations.append("🔴 发现严重问题，需要立即处理:")
            for test in failed_tests:
                recommendations.append(f"   - {test.message}")
        
        if warning_tests:
            recommendations.append("🟡 发现潜在问题，建议优化:")
            for test in warning_tests:
                recommendations.append(f"   - {test.message}")
        
        # 基于数据质量的建议
        if data_quality.data_completeness < 95:
            recommendations.append(f"📊 数据完整性较低 ({data_quality.data_completeness:.1f}%)，建议:")
            recommendations.append("   - 检查数据写入逻辑，确保必需字段不为空")
            recommendations.append("   - 添加数据验证规则")
        
        if data_quality.data_accuracy < 90:
            recommendations.append(f"🎯 数据准确性较低 ({data_quality.data_accuracy:.1f}%)，建议:")
            recommendations.append("   - 验证数据格式规则")
            recommendations.append("   - 添加输入数据校验")
        
        if data_quality.duplicate_records > 0:
            recommendations.append(f"🔄 发现重复记录 ({data_quality.duplicate_records} 条)，建议:")
            recommendations.append("   - 检查数据写入逻辑，避免重复插入")
            recommendations.append("   - 考虑添加唯一约束")
        
        # 基于性能指标的建议
        if performance_metrics.avg_query_time_ms > 100:
            recommendations.append(f"⚡ 查询性能较慢 (平均 {performance_metrics.avg_query_time_ms:.1f}ms)，建议:")
            recommendations.append("   - 检查并优化索引")
            recommendations.append("   - 考虑查询优化")
            recommendations.append("   - 评估是否需要分区")
        
        if performance_metrics.storage_efficiency < 80:
            recommendations.append(f"💾 存储效率较低 ({performance_metrics.storage_efficiency:.1f}%)，建议:")
            recommendations.append("   - 检查数据类型是否合适")
            recommendations.append("   - 考虑数据压缩")
            recommendations.append("   - 清理历史数据")
        
        # 通用建议
        if not recommendations:
            recommendations.append("✅ 系统状态良好，建议:")
            recommendations.append("   - 定期执行数据验证")
            recommendations.append("   - 监控性能指标")
            recommendations.append("   - 保持数据备份")
        
        return recommendations
    
    async def _table_exists(self, connection, table_name: str) -> bool:
        """检查表是否存在"""
        return await connection.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = $1
            )
        """, table_name)
    
    async def generate_full_report(self) -> ValidationReport:
        """生成完整的验证报告"""
        print("🔍 开始数据验证...")
        
        # 获取数据库信息
        async with self.connection_pool.acquire() as connection:
            db_version = await connection.fetchval("SELECT version()")
            db_size = await connection.fetchval("""
                SELECT pg_size_pretty(pg_database_size(current_database()))
            """)
            
            database_info = {
                "version": db_version,
                "size": db_size,
                "host": self.db_config["host"],
                "port": self.db_config["port"],
                "database": self.db_config["database"]
            }
        
        # 执行各项验证
        print("📋 验证数据库模式...")
        schema_results = await self.validate_schema()
        
        print("🔍 验证数据完整性...")
        data_results = await self.validate_data_integrity()
        
        print("⚡ 验证性能指标...")
        performance_results, performance_metrics = await self.validate_performance()
        
        print("📊 计算数据质量指标...")
        data_quality = await self.calculate_data_quality_metrics()
        
        # 计算总体评分
        all_results = schema_results + data_results + performance_results
        if all_results:
            overall_score = sum(r.score for r in all_results) / len(all_results)
        else:
            overall_score = 0.0
        
        # 生成建议
        print("💡 生成改进建议...")
        recommendations = await self.generate_recommendations(
            all_results, data_quality, performance_metrics
        )
        
        # 创建报告
        report = ValidationReport(
            timestamp=datetime.now(timezone.utc),
            database_info=database_info,
            schema_validation=schema_results,
            data_validation=data_results,
            performance_validation=performance_results,
            data_quality=data_quality,
            performance_metrics=performance_metrics,
            overall_score=overall_score,
            recommendations=recommendations
        )
        
        return report


def print_validation_report(report: ValidationReport):
    """打印验证报告"""
    print("\n" + "="*80)
    print("🔍 HarborAI 追踪系统数据验证报告")
    print("="*80)
    
    print(f"\n📅 报告时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"🗄️  数据库: {report.database_info['host']}:{report.database_info['port']}/{report.database_info['database']}")
    print(f"📦 数据库版本: {report.database_info['version']}")
    print(f"💾 数据库大小: {report.database_info['size']}")
    
    # 总体评分
    print(f"\n🎯 总体评分: {report.overall_score:.1f}/100")
    
    if report.overall_score >= 90:
        status_emoji = "🟢"
        status_text = "优秀"
    elif report.overall_score >= 75:
        status_emoji = "🟡"
        status_text = "良好"
    elif report.overall_score >= 60:
        status_emoji = "🟠"
        status_text = "一般"
    else:
        status_emoji = "🔴"
        status_text = "需要改进"
    
    print(f"📊 系统状态: {status_emoji} {status_text}")
    
    # 数据质量指标
    print(f"\n📊 数据质量指标:")
    print(f"   总记录数: {report.data_quality.total_records:,}")
    print(f"   有效记录: {report.data_quality.valid_records:,}")
    print(f"   无效记录: {report.data_quality.invalid_records:,}")
    print(f"   重复记录: {report.data_quality.duplicate_records:,}")
    print(f"   数据完整性: {report.data_quality.data_completeness:.1f}%")
    print(f"   数据准确性: {report.data_quality.data_accuracy:.1f}%")
    print(f"   数据一致性: {report.data_quality.data_consistency:.1f}%")
    print(f"   质量评分: {report.data_quality.overall_score:.1f}/100")
    
    # 性能指标
    print(f"\n⚡ 性能指标:")
    print(f"   平均查询时间: {report.performance_metrics.avg_query_time_ms:.2f}ms")
    print(f"   最大查询时间: {report.performance_metrics.max_query_time_ms:.2f}ms")
    print(f"   P95查询时间: {report.performance_metrics.p95_query_time_ms:.2f}ms")
    print(f"   估算吞吐量: {report.performance_metrics.throughput_qps:.1f} QPS")
    print(f"   索引效率: {report.performance_metrics.index_efficiency:.1f}%")
    print(f"   存储效率: {report.performance_metrics.storage_efficiency:.1f}%")
    
    # 验证结果统计
    all_results = report.schema_validation + report.data_validation + report.performance_validation
    
    pass_count = len([r for r in all_results if r.status == "pass"])
    warning_count = len([r for r in all_results if r.status == "warning"])
    fail_count = len([r for r in all_results if r.status == "fail"])
    
    print(f"\n📋 验证结果统计:")
    print(f"   ✅ 通过: {pass_count}")
    print(f"   ⚠️  警告: {warning_count}")
    print(f"   ❌ 失败: {fail_count}")
    print(f"   📊 总计: {len(all_results)}")
    
    # 详细结果（仅显示警告和失败）
    issues = [r for r in all_results if r.status in ["warning", "fail"]]
    if issues:
        print(f"\n⚠️  需要关注的问题:")
        for result in issues:
            emoji = "❌" if result.status == "fail" else "⚠️"
            print(f"   {emoji} [{result.category}] {result.test_name}: {result.message}")
    
    # 改进建议
    if report.recommendations:
        print(f"\n💡 改进建议:")
        for recommendation in report.recommendations:
            print(f"   {recommendation}")
    
    print("\n" + "="*80)


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HarborAI 追踪系统数据验证工具")
    parser.add_argument("--host", default="localhost", help="数据库主机")
    parser.add_argument("--port", type=int, default=5432, help="数据库端口")
    parser.add_argument("--database", required=True, help="数据库名称")
    parser.add_argument("--user", required=True, help="数据库用户")
    parser.add_argument("--password", required=True, help="数据库密码")
    parser.add_argument("--output", help="输出报告文件路径（JSON格式）")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 数据库配置
    db_config = {
        "host": args.host,
        "port": args.port,
        "database": args.database,
        "user": args.user,
        "password": args.password
    }
    
    # 创建验证器
    validator = TracingDataValidator(db_config)
    
    try:
        # 连接数据库
        await validator.connect()
        
        # 生成验证报告
        report = await validator.generate_full_report()
        
        # 打印报告
        print_validation_report(report)
        
        # 保存报告到文件
        if args.output:
            report_data = {
                "timestamp": report.timestamp.isoformat(),
                "database_info": report.database_info,
                "overall_score": report.overall_score,
                "data_quality": asdict(report.data_quality),
                "performance_metrics": asdict(report.performance_metrics),
                "schema_validation": [asdict(r) for r in report.schema_validation],
                "data_validation": [asdict(r) for r in report.data_validation],
                "performance_validation": [asdict(r) for r in report.performance_validation],
                "recommendations": report.recommendations
            }
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 报告已保存到: {args.output}")
        
        # 根据评分设置退出码
        if report.overall_score >= 75:
            sys.exit(0)  # 成功
        elif report.overall_score >= 50:
            sys.exit(1)  # 警告
        else:
            sys.exit(2)  # 错误
    
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        sys.exit(3)
    
    finally:
        await validator.disconnect()


if __name__ == "__main__":
    asyncio.run(main())