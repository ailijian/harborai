#!/usr/bin/env python3
"""
HarborAI 追踪系统数据库迁移脚本

此脚本负责创建和更新追踪系统所需的数据库表结构，包括：
- tracing_info 表：存储追踪信息
- 相关索引和约束
- 数据迁移和验证

功能特性：
- 支持增量迁移
- 数据完整性验证
- 回滚机制
- 性能优化索引
- 兼容性检查

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
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from harborai.utils.exceptions import DatabaseError, MigrationError


@dataclass
class MigrationStep:
    """迁移步骤定义"""
    version: str
    name: str
    description: str
    sql_up: str
    sql_down: str
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class MigrationResult:
    """迁移结果"""
    success: bool
    version: str
    message: str
    execution_time_ms: float
    affected_rows: int = 0
    error: Optional[str] = None


class TracingMigrationManager:
    """追踪系统迁移管理器"""
    
    def __init__(self, db_config: Dict[str, Any]):
        """
        初始化迁移管理器
        
        Args:
            db_config: 数据库连接配置
        """
        self.db_config = db_config
        self.connection_pool: Optional[asyncpg.Pool] = None
        self.migration_steps = self._define_migration_steps()
        
    async def connect(self) -> None:
        """建立数据库连接"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                host=self.db_config["host"],
                port=self.db_config["port"],
                database=self.db_config["database"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                min_size=1,
                max_size=5,
                command_timeout=30
            )
            print(f"✅ 数据库连接成功: {self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}")
        except Exception as e:
            raise DatabaseError(f"数据库连接失败: {e}")
    
    async def disconnect(self) -> None:
        """关闭数据库连接"""
        if self.connection_pool:
            await self.connection_pool.close()
            print("✅ 数据库连接已关闭")
    
    def _define_migration_steps(self) -> List[MigrationStep]:
        """定义迁移步骤"""
        return [
            MigrationStep(
                version="001",
                name="create_migration_history",
                description="创建迁移历史表",
                sql_up="""
                CREATE TABLE IF NOT EXISTS migration_history (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(50) NOT NULL UNIQUE,
                    name VARCHAR(255) NOT NULL,
                    description TEXT,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    execution_time_ms FLOAT,
                    success BOOLEAN DEFAULT TRUE,
                    error_message TEXT,
                    checksum VARCHAR(64)
                );
                
                CREATE INDEX IF NOT EXISTS idx_migration_history_version 
                ON migration_history(version);
                
                CREATE INDEX IF NOT EXISTS idx_migration_history_applied_at 
                ON migration_history(applied_at);
                """,
                sql_down="""
                DROP TABLE IF EXISTS migration_history CASCADE;
                """
            ),
            
            MigrationStep(
                version="002",
                name="create_tracing_info_table",
                description="创建追踪信息表",
                sql_up="""
                CREATE TABLE IF NOT EXISTS tracing_info (
                    id BIGSERIAL PRIMARY KEY,
                    hb_trace_id VARCHAR(255) NOT NULL,
                    otel_trace_id VARCHAR(32) NOT NULL,
                    span_id VARCHAR(16) NOT NULL,
                    operation_name VARCHAR(255) NOT NULL,
                    service_name VARCHAR(100) NOT NULL,
                    service_version VARCHAR(50),
                    environment VARCHAR(50),
                    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                    end_time TIMESTAMP WITH TIME ZONE,
                    duration_ms FLOAT,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    success BOOLEAN,
                    error_message TEXT,
                    tags JSONB DEFAULT '{}',
                    logs JSONB DEFAULT '[]',
                    parent_span_id VARCHAR(16),
                    trace_flags INTEGER DEFAULT 0,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
                
                -- 添加注释
                COMMENT ON TABLE tracing_info IS '追踪信息表，存储HarborAI和OpenTelemetry的追踪数据';
                COMMENT ON COLUMN tracing_info.hb_trace_id IS 'HarborAI内部追踪ID';
                COMMENT ON COLUMN tracing_info.otel_trace_id IS 'OpenTelemetry追踪ID（32位十六进制）';
                COMMENT ON COLUMN tracing_info.span_id IS 'OpenTelemetry Span ID（16位十六进制）';
                COMMENT ON COLUMN tracing_info.operation_name IS '操作名称，如ai.chat.completion';
                COMMENT ON COLUMN tracing_info.service_name IS '服务名称';
                COMMENT ON COLUMN tracing_info.status IS '追踪状态：pending, active, completed, error';
                COMMENT ON COLUMN tracing_info.tags IS '追踪标签，JSON格式';
                COMMENT ON COLUMN tracing_info.logs IS '追踪日志，JSON数组格式';
                """,
                sql_down="""
                DROP TABLE IF EXISTS tracing_info CASCADE;
                """,
                dependencies=["001"]
            ),
            
            MigrationStep(
                version="003",
                name="create_tracing_indexes",
                description="创建追踪信息表索引",
                sql_up="""
                -- 主要查询索引
                CREATE UNIQUE INDEX IF NOT EXISTS idx_tracing_info_hb_trace_id 
                ON tracing_info(hb_trace_id);
                
                CREATE INDEX IF NOT EXISTS idx_tracing_info_otel_trace_id 
                ON tracing_info(otel_trace_id);
                
                CREATE INDEX IF NOT EXISTS idx_tracing_info_operation_name 
                ON tracing_info(operation_name);
                
                CREATE INDEX IF NOT EXISTS idx_tracing_info_service_name 
                ON tracing_info(service_name);
                
                CREATE INDEX IF NOT EXISTS idx_tracing_info_status 
                ON tracing_info(status);
                
                -- 时间范围查询索引
                CREATE INDEX IF NOT EXISTS idx_tracing_info_start_time 
                ON tracing_info(start_time);
                
                CREATE INDEX IF NOT EXISTS idx_tracing_info_created_at 
                ON tracing_info(created_at);
                
                -- 复合索引用于常见查询模式
                CREATE INDEX IF NOT EXISTS idx_tracing_info_service_operation 
                ON tracing_info(service_name, operation_name);
                
                CREATE INDEX IF NOT EXISTS idx_tracing_info_status_start_time 
                ON tracing_info(status, start_time);
                
                CREATE INDEX IF NOT EXISTS idx_tracing_info_success_start_time 
                ON tracing_info(success, start_time) WHERE success IS NOT NULL;
                
                -- JSONB索引用于标签查询
                CREATE INDEX IF NOT EXISTS idx_tracing_info_tags_gin 
                ON tracing_info USING GIN(tags);
                
                -- 部分索引用于活跃追踪
                CREATE INDEX IF NOT EXISTS idx_tracing_info_active_traces 
                ON tracing_info(hb_trace_id, start_time) 
                WHERE status IN ('pending', 'active');
                """,
                sql_down="""
                DROP INDEX IF EXISTS idx_tracing_info_hb_trace_id;
                DROP INDEX IF EXISTS idx_tracing_info_otel_trace_id;
                DROP INDEX IF EXISTS idx_tracing_info_operation_name;
                DROP INDEX IF EXISTS idx_tracing_info_service_name;
                DROP INDEX IF EXISTS idx_tracing_info_status;
                DROP INDEX IF EXISTS idx_tracing_info_start_time;
                DROP INDEX IF EXISTS idx_tracing_info_created_at;
                DROP INDEX IF EXISTS idx_tracing_info_service_operation;
                DROP INDEX IF EXISTS idx_tracing_info_status_start_time;
                DROP INDEX IF EXISTS idx_tracing_info_success_start_time;
                DROP INDEX IF EXISTS idx_tracing_info_tags_gin;
                DROP INDEX IF EXISTS idx_tracing_info_active_traces;
                """,
                dependencies=["002"]
            ),
            
            MigrationStep(
                version="004",
                name="create_tracing_constraints",
                description="创建追踪信息表约束",
                sql_up="""
                -- 检查约束
                ALTER TABLE tracing_info 
                ADD CONSTRAINT chk_tracing_info_status 
                CHECK (status IN ('pending', 'active', 'completed', 'error'));
                
                ALTER TABLE tracing_info 
                ADD CONSTRAINT chk_tracing_info_duration 
                CHECK (duration_ms IS NULL OR duration_ms >= 0);
                
                ALTER TABLE tracing_info 
                ADD CONSTRAINT chk_tracing_info_times 
                CHECK (end_time IS NULL OR end_time >= start_time);
                
                ALTER TABLE tracing_info 
                ADD CONSTRAINT chk_tracing_info_otel_trace_id_format 
                CHECK (otel_trace_id ~ '^[0-9a-f]{32}$');
                
                ALTER TABLE tracing_info 
                ADD CONSTRAINT chk_tracing_info_span_id_format 
                CHECK (span_id ~ '^[0-9a-f]{16}$');
                
                -- 触发器：自动更新updated_at字段
                CREATE OR REPLACE FUNCTION update_tracing_info_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
                
                CREATE TRIGGER trg_tracing_info_updated_at
                    BEFORE UPDATE ON tracing_info
                    FOR EACH ROW
                    EXECUTE FUNCTION update_tracing_info_updated_at();
                """,
                sql_down="""
                DROP TRIGGER IF EXISTS trg_tracing_info_updated_at ON tracing_info;
                DROP FUNCTION IF EXISTS update_tracing_info_updated_at();
                
                ALTER TABLE tracing_info DROP CONSTRAINT IF EXISTS chk_tracing_info_status;
                ALTER TABLE tracing_info DROP CONSTRAINT IF EXISTS chk_tracing_info_duration;
                ALTER TABLE tracing_info DROP CONSTRAINT IF EXISTS chk_tracing_info_times;
                ALTER TABLE tracing_info DROP CONSTRAINT IF EXISTS chk_tracing_info_otel_trace_id_format;
                ALTER TABLE tracing_info DROP CONSTRAINT IF EXISTS chk_tracing_info_span_id_format;
                """,
                dependencies=["003"]
            ),
            
            MigrationStep(
                version="005",
                name="create_tracing_partitions",
                description="创建追踪信息表分区（按月）",
                sql_up="""
                -- 创建分区函数
                CREATE OR REPLACE FUNCTION create_tracing_info_partition(
                    partition_date DATE
                ) RETURNS VOID AS $$
                DECLARE
                    partition_name TEXT;
                    start_date DATE;
                    end_date DATE;
                BEGIN
                    -- 计算分区名称和日期范围
                    partition_name := 'tracing_info_' || TO_CHAR(partition_date, 'YYYY_MM');
                    start_date := DATE_TRUNC('month', partition_date);
                    end_date := start_date + INTERVAL '1 month';
                    
                    -- 创建分区表（如果不存在）
                    EXECUTE format('
                        CREATE TABLE IF NOT EXISTS %I PARTITION OF tracing_info
                        FOR VALUES FROM (%L) TO (%L)',
                        partition_name, start_date, end_date
                    );
                    
                    -- 创建分区特定的索引
                    EXECUTE format('
                        CREATE INDEX IF NOT EXISTS %I 
                        ON %I(start_time)',
                        'idx_' || partition_name || '_start_time',
                        partition_name
                    );
                    
                    RAISE NOTICE '分区 % 创建成功', partition_name;
                END;
                $$ LANGUAGE plpgsql;
                
                -- 将现有表转换为分区表（如果有数据需要先备份）
                -- 注意：这是一个破坏性操作，在生产环境中需要谨慎执行
                
                -- 创建当前月份和下个月的分区
                SELECT create_tracing_info_partition(CURRENT_DATE);
                SELECT create_tracing_info_partition(CURRENT_DATE + INTERVAL '1 month');
                """,
                sql_down="""
                -- 删除分区函数
                DROP FUNCTION IF EXISTS create_tracing_info_partition(DATE);
                
                -- 注意：删除分区需要手动处理，因为涉及数据迁移
                -- 在生产环境中，这个回滚操作需要特别小心
                """,
                dependencies=["004"]
            ),
            
            MigrationStep(
                version="006",
                name="create_tracing_views",
                description="创建追踪信息视图",
                sql_up="""
                -- 活跃追踪视图
                CREATE OR REPLACE VIEW v_active_traces AS
                SELECT 
                    hb_trace_id,
                    otel_trace_id,
                    operation_name,
                    service_name,
                    start_time,
                    EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - start_time)) * 1000 AS running_time_ms,
                    tags,
                    status
                FROM tracing_info
                WHERE status IN ('pending', 'active')
                ORDER BY start_time DESC;
                
                -- 追踪统计视图
                CREATE OR REPLACE VIEW v_tracing_stats AS
                SELECT 
                    service_name,
                    operation_name,
                    DATE_TRUNC('hour', start_time) AS hour_bucket,
                    COUNT(*) AS total_traces,
                    COUNT(*) FILTER (WHERE success = true) AS successful_traces,
                    COUNT(*) FILTER (WHERE success = false) AS failed_traces,
                    AVG(duration_ms) AS avg_duration_ms,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) AS median_duration_ms,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95_duration_ms,
                    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) AS p99_duration_ms
                FROM tracing_info
                WHERE status = 'completed' 
                  AND start_time >= CURRENT_TIMESTAMP - INTERVAL '24 hours'
                GROUP BY service_name, operation_name, hour_bucket
                ORDER BY hour_bucket DESC, service_name, operation_name;
                
                -- 错误追踪视图
                CREATE OR REPLACE VIEW v_error_traces AS
                SELECT 
                    hb_trace_id,
                    otel_trace_id,
                    operation_name,
                    service_name,
                    start_time,
                    end_time,
                    duration_ms,
                    error_message,
                    tags
                FROM tracing_info
                WHERE status = 'error' OR success = false
                ORDER BY start_time DESC;
                
                -- 性能慢查询视图
                CREATE OR REPLACE VIEW v_slow_traces AS
                SELECT 
                    hb_trace_id,
                    otel_trace_id,
                    operation_name,
                    service_name,
                    start_time,
                    duration_ms,
                    tags
                FROM tracing_info
                WHERE status = 'completed' 
                  AND duration_ms > 1000  -- 超过1秒的请求
                ORDER BY duration_ms DESC;
                """,
                sql_down="""
                DROP VIEW IF EXISTS v_active_traces;
                DROP VIEW IF EXISTS v_tracing_stats;
                DROP VIEW IF EXISTS v_error_traces;
                DROP VIEW IF EXISTS v_slow_traces;
                """,
                dependencies=["005"]
            )
        ]
    
    async def get_applied_migrations(self) -> List[str]:
        """获取已应用的迁移版本"""
        if not self.connection_pool:
            raise DatabaseError("数据库连接未建立")
        
        async with self.connection_pool.acquire() as connection:
            try:
                # 检查迁移历史表是否存在
                exists = await connection.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'migration_history'
                    )
                """)
                
                if not exists:
                    return []
                
                # 获取已应用的迁移
                rows = await connection.fetch("""
                    SELECT version 
                    FROM migration_history 
                    WHERE success = true 
                    ORDER BY applied_at
                """)
                
                return [row['version'] for row in rows]
                
            except Exception as e:
                raise DatabaseError(f"获取迁移历史失败: {e}")
    
    async def apply_migration(self, step: MigrationStep) -> MigrationResult:
        """应用单个迁移步骤"""
        if not self.connection_pool:
            raise DatabaseError("数据库连接未建立")
        
        start_time = datetime.now()
        
        async with self.connection_pool.acquire() as connection:
            async with connection.transaction():
                try:
                    print(f"🔄 应用迁移 {step.version}: {step.name}")
                    print(f"   描述: {step.description}")
                    
                    # 执行迁移SQL
                    result = await connection.execute(step.sql_up)
                    
                    # 计算执行时间
                    end_time = datetime.now()
                    execution_time_ms = (end_time - start_time).total_seconds() * 1000
                    
                    # 记录迁移历史
                    await connection.execute("""
                        INSERT INTO migration_history 
                        (version, name, description, execution_time_ms, success)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (version) DO UPDATE SET
                            applied_at = CURRENT_TIMESTAMP,
                            execution_time_ms = EXCLUDED.execution_time_ms,
                            success = EXCLUDED.success,
                            error_message = NULL
                    """, step.version, step.name, step.description, execution_time_ms, True)
                    
                    print(f"✅ 迁移 {step.version} 应用成功 ({execution_time_ms:.2f}ms)")
                    
                    return MigrationResult(
                        success=True,
                        version=step.version,
                        message=f"迁移 {step.version} 应用成功",
                        execution_time_ms=execution_time_ms
                    )
                    
                except Exception as e:
                    # 记录失败的迁移
                    end_time = datetime.now()
                    execution_time_ms = (end_time - start_time).total_seconds() * 1000
                    
                    try:
                        await connection.execute("""
                            INSERT INTO migration_history 
                            (version, name, description, execution_time_ms, success, error_message)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (version) DO UPDATE SET
                                applied_at = CURRENT_TIMESTAMP,
                                execution_time_ms = EXCLUDED.execution_time_ms,
                                success = EXCLUDED.success,
                                error_message = EXCLUDED.error_message
                        """, step.version, step.name, step.description, execution_time_ms, False, str(e))
                    except:
                        pass  # 如果连迁移历史都无法记录，说明是严重错误
                    
                    print(f"❌ 迁移 {step.version} 应用失败: {e}")
                    
                    return MigrationResult(
                        success=False,
                        version=step.version,
                        message=f"迁移 {step.version} 应用失败",
                        execution_time_ms=execution_time_ms,
                        error=str(e)
                    )
    
    async def rollback_migration(self, step: MigrationStep) -> MigrationResult:
        """回滚单个迁移步骤"""
        if not self.connection_pool:
            raise DatabaseError("数据库连接未建立")
        
        start_time = datetime.now()
        
        async with self.connection_pool.acquire() as connection:
            async with connection.transaction():
                try:
                    print(f"🔄 回滚迁移 {step.version}: {step.name}")
                    
                    # 执行回滚SQL
                    await connection.execute(step.sql_down)
                    
                    # 删除迁移历史记录
                    await connection.execute("""
                        DELETE FROM migration_history WHERE version = $1
                    """, step.version)
                    
                    # 计算执行时间
                    end_time = datetime.now()
                    execution_time_ms = (end_time - start_time).total_seconds() * 1000
                    
                    print(f"✅ 迁移 {step.version} 回滚成功 ({execution_time_ms:.2f}ms)")
                    
                    return MigrationResult(
                        success=True,
                        version=step.version,
                        message=f"迁移 {step.version} 回滚成功",
                        execution_time_ms=execution_time_ms
                    )
                    
                except Exception as e:
                    end_time = datetime.now()
                    execution_time_ms = (end_time - start_time).total_seconds() * 1000
                    
                    print(f"❌ 迁移 {step.version} 回滚失败: {e}")
                    
                    return MigrationResult(
                        success=False,
                        version=step.version,
                        message=f"迁移 {step.version} 回滚失败",
                        execution_time_ms=execution_time_ms,
                        error=str(e)
                    )
    
    async def migrate_up(self, target_version: Optional[str] = None) -> List[MigrationResult]:
        """执行向上迁移"""
        applied_migrations = await self.get_applied_migrations()
        results = []
        
        print(f"📋 已应用的迁移: {applied_migrations}")
        
        for step in self.migration_steps:
            # 检查是否已应用
            if step.version in applied_migrations:
                print(f"⏭️  跳过已应用的迁移 {step.version}")
                continue
            
            # 检查依赖
            for dep in step.dependencies:
                if dep not in applied_migrations:
                    raise MigrationError(f"迁移 {step.version} 依赖 {dep}，但 {dep} 尚未应用")
            
            # 应用迁移
            result = await self.apply_migration(step)
            results.append(result)
            
            if not result.success:
                print(f"❌ 迁移失败，停止后续迁移")
                break
            
            applied_migrations.append(step.version)
            
            # 检查是否达到目标版本
            if target_version and step.version == target_version:
                print(f"🎯 已达到目标版本 {target_version}")
                break
        
        return results
    
    async def migrate_down(self, target_version: str) -> List[MigrationResult]:
        """执行向下迁移（回滚）"""
        applied_migrations = await self.get_applied_migrations()
        results = []
        
        # 找到需要回滚的迁移（按逆序）
        steps_to_rollback = []
        for step in reversed(self.migration_steps):
            if step.version in applied_migrations:
                steps_to_rollback.append(step)
                if step.version == target_version:
                    break
        
        print(f"📋 需要回滚的迁移: {[s.version for s in steps_to_rollback]}")
        
        for step in steps_to_rollback:
            if step.version == target_version:
                print(f"🎯 已达到目标版本 {target_version}，停止回滚")
                break
            
            result = await self.rollback_migration(step)
            results.append(result)
            
            if not result.success:
                print(f"❌ 回滚失败，停止后续回滚")
                break
        
        return results
    
    async def validate_schema(self) -> Dict[str, Any]:
        """验证数据库模式"""
        if not self.connection_pool:
            raise DatabaseError("数据库连接未建立")
        
        validation_results = {
            "tables": {},
            "indexes": {},
            "constraints": {},
            "views": {},
            "functions": {},
            "overall_status": "unknown"
        }
        
        async with self.connection_pool.acquire() as connection:
            try:
                # 验证表
                tables = await connection.fetch("""
                    SELECT table_name, 
                           (SELECT COUNT(*) FROM information_schema.columns 
                            WHERE table_name = t.table_name AND table_schema = 'public') as column_count
                    FROM information_schema.tables t
                    WHERE table_schema = 'public' 
                    AND table_name IN ('migration_history', 'tracing_info')
                """)
                
                for table in tables:
                    validation_results["tables"][table["table_name"]] = {
                        "exists": True,
                        "column_count": table["column_count"]
                    }
                
                # 验证索引
                indexes = await connection.fetch("""
                    SELECT indexname, tablename
                    FROM pg_indexes
                    WHERE schemaname = 'public'
                    AND tablename IN ('migration_history', 'tracing_info')
                """)
                
                for index in indexes:
                    table_name = index["tablename"]
                    if table_name not in validation_results["indexes"]:
                        validation_results["indexes"][table_name] = []
                    validation_results["indexes"][table_name].append(index["indexname"])
                
                # 验证约束
                constraints = await connection.fetch("""
                    SELECT constraint_name, table_name, constraint_type
                    FROM information_schema.table_constraints
                    WHERE table_schema = 'public'
                    AND table_name IN ('migration_history', 'tracing_info')
                """)
                
                for constraint in constraints:
                    table_name = constraint["table_name"]
                    if table_name not in validation_results["constraints"]:
                        validation_results["constraints"][table_name] = []
                    validation_results["constraints"][table_name].append({
                        "name": constraint["constraint_name"],
                        "type": constraint["constraint_type"]
                    })
                
                # 验证视图
                views = await connection.fetch("""
                    SELECT table_name
                    FROM information_schema.views
                    WHERE table_schema = 'public'
                    AND table_name LIKE 'v_%'
                """)
                
                validation_results["views"] = [view["table_name"] for view in views]
                
                # 验证函数
                functions = await connection.fetch("""
                    SELECT routine_name
                    FROM information_schema.routines
                    WHERE routine_schema = 'public'
                    AND routine_type = 'FUNCTION'
                    AND routine_name LIKE '%tracing%'
                """)
                
                validation_results["functions"] = [func["routine_name"] for func in functions]
                
                # 总体状态评估
                required_tables = {"migration_history", "tracing_info"}
                existing_tables = set(validation_results["tables"].keys())
                
                if required_tables.issubset(existing_tables):
                    validation_results["overall_status"] = "healthy"
                else:
                    validation_results["overall_status"] = "incomplete"
                
            except Exception as e:
                validation_results["overall_status"] = "error"
                validation_results["error"] = str(e)
        
        return validation_results
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态"""
        applied_migrations = await self.get_applied_migrations()
        
        status = {
            "applied_migrations": applied_migrations,
            "pending_migrations": [],
            "total_migrations": len(self.migration_steps),
            "completion_percentage": 0
        }
        
        for step in self.migration_steps:
            if step.version not in applied_migrations:
                status["pending_migrations"].append({
                    "version": step.version,
                    "name": step.name,
                    "description": step.description
                })
        
        if status["total_migrations"] > 0:
            status["completion_percentage"] = (
                len(applied_migrations) / status["total_migrations"] * 100
            )
        
        return status


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HarborAI 追踪系统数据库迁移工具")
    parser.add_argument("--host", default="localhost", help="数据库主机")
    parser.add_argument("--port", type=int, default=5432, help="数据库端口")
    parser.add_argument("--database", required=True, help="数据库名称")
    parser.add_argument("--user", required=True, help="数据库用户")
    parser.add_argument("--password", required=True, help="数据库密码")
    parser.add_argument("--action", choices=["up", "down", "status", "validate"], 
                       default="up", help="执行的操作")
    parser.add_argument("--target", help="目标版本（用于up/down操作）")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要执行的操作，不实际执行")
    
    args = parser.parse_args()
    
    # 数据库配置
    db_config = {
        "host": args.host,
        "port": args.port,
        "database": args.database,
        "user": args.user,
        "password": args.password
    }
    
    # 创建迁移管理器
    migration_manager = TracingMigrationManager(db_config)
    
    try:
        # 连接数据库
        await migration_manager.connect()
        
        if args.action == "up":
            print("🚀 开始执行向上迁移...")
            if args.dry_run:
                print("🔍 DRY RUN 模式 - 仅显示将要执行的操作")
                status = await migration_manager.get_migration_status()
                print(f"📊 待执行的迁移:")
                for migration in status["pending_migrations"]:
                    print(f"   - {migration['version']}: {migration['name']}")
            else:
                results = await migration_manager.migrate_up(args.target)
                
                success_count = sum(1 for r in results if r.success)
                total_count = len(results)
                
                print(f"\n📊 迁移结果: {success_count}/{total_count} 成功")
                
                if success_count == total_count and total_count > 0:
                    print("🎉 所有迁移执行成功！")
                elif success_count < total_count:
                    print("⚠️  部分迁移执行失败，请检查错误信息")
                    sys.exit(1)
                else:
                    print("ℹ️  没有需要执行的迁移")
        
        elif args.action == "down":
            if not args.target:
                print("❌ 回滚操作需要指定目标版本 (--target)")
                sys.exit(1)
            
            print(f"🔄 开始回滚到版本 {args.target}...")
            if args.dry_run:
                print("🔍 DRY RUN 模式 - 仅显示将要执行的操作")
                # 这里可以添加显示将要回滚的迁移
            else:
                results = await migration_manager.migrate_down(args.target)
                
                success_count = sum(1 for r in results if r.success)
                total_count = len(results)
                
                print(f"\n📊 回滚结果: {success_count}/{total_count} 成功")
                
                if success_count == total_count:
                    print("🎉 回滚执行成功！")
                else:
                    print("⚠️  部分回滚执行失败，请检查错误信息")
                    sys.exit(1)
        
        elif args.action == "status":
            print("📊 获取迁移状态...")
            status = await migration_manager.get_migration_status()
            
            print(f"\n迁移状态:")
            print(f"  总迁移数: {status['total_migrations']}")
            print(f"  已应用: {len(status['applied_migrations'])}")
            print(f"  待应用: {len(status['pending_migrations'])}")
            print(f"  完成度: {status['completion_percentage']:.1f}%")
            
            if status['applied_migrations']:
                print(f"\n已应用的迁移:")
                for version in status['applied_migrations']:
                    print(f"  ✅ {version}")
            
            if status['pending_migrations']:
                print(f"\n待应用的迁移:")
                for migration in status['pending_migrations']:
                    print(f"  ⏳ {migration['version']}: {migration['name']}")
        
        elif args.action == "validate":
            print("🔍 验证数据库模式...")
            validation = await migration_manager.validate_schema()
            
            print(f"\n验证结果:")
            print(f"  总体状态: {validation['overall_status']}")
            
            print(f"\n表:")
            for table_name, info in validation["tables"].items():
                print(f"  ✅ {table_name} ({info['column_count']} 列)")
            
            print(f"\n索引:")
            for table_name, indexes in validation["indexes"].items():
                print(f"  {table_name}: {len(indexes)} 个索引")
            
            print(f"\n视图: {len(validation['views'])} 个")
            print(f"函数: {len(validation['functions'])} 个")
            
            if validation['overall_status'] == 'healthy':
                print("\n🎉 数据库模式验证通过！")
            else:
                print("\n⚠️  数据库模式存在问题，请检查迁移状态")
                if 'error' in validation:
                    print(f"错误: {validation['error']}")
                sys.exit(1)
    
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        sys.exit(1)
    
    finally:
        await migration_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())