#!/usr/bin/env python3
"""
数据库迁移集成测试

测试数据库迁移脚本的功能，包括：
- 迁移表创建
- 迁移文件发现
- 迁移状态管理
- 表结构验证

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import pytest
import asyncio
from pathlib import Path
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from harborai.config.settings import get_settings
from scripts.run_migration import MigrationRunner
from scripts.validate_migration import MigrationValidator


class TestMigrationRunner:
    """迁移运行器测试"""
    
    @pytest.fixture
    def mock_connection(self):
        """模拟数据库连接"""
        conn = AsyncMock()
        conn.execute = AsyncMock()
        conn.fetch = AsyncMock()
        conn.fetchrow = AsyncMock()
        return conn
    
    @pytest.fixture
    def migration_runner(self, mock_connection):
        """迁移运行器fixture"""
        return MigrationRunner(mock_connection)
    
    @pytest.mark.asyncio
    async def test_create_migration_table(self, migration_runner, mock_connection):
        """测试创建迁移表"""
        # 模拟表不存在
        mock_connection.fetchrow.return_value = None
        
        await migration_runner.create_migration_table()
        
        # 验证执行了创建表的SQL
        mock_connection.execute.assert_called()
        call_args = mock_connection.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS schema_migrations" in call_args
    
    @pytest.mark.asyncio
    async def test_get_applied_migrations(self, migration_runner, mock_connection):
        """测试获取已应用的迁移"""
        # 模拟返回的迁移记录
        mock_connection.fetch.return_value = [
            {"version": "001"},
            {"version": "002"}
        ]
        
        migrations = await migration_runner.get_applied_migrations()
        
        assert len(migrations) == 2
        assert migrations[0] == "001"
        assert migrations[1] == "002"
    
    def test_get_migration_files(self, migration_runner):
        """测试获取迁移文件"""
        # 创建临时迁移目录和文件
        migration_dir = Path("test_migrations")
        migration_dir.mkdir(exist_ok=True)
        
        try:
            # 创建测试迁移文件
            (migration_dir / "001_initial.sql").write_text("CREATE TABLE test;")
            (migration_dir / "002_add_index.sql").write_text("CREATE INDEX test_idx ON test(id);")
            (migration_dir / "invalid.txt").write_text("Not a migration")
            
            migration_runner.migrations_dir = migration_dir
            files = migration_runner.get_migration_files()
            
            assert len(files) == 2
            assert any("001_initial.sql" in str(f) for f in files)
            assert any("002_add_index.sql" in str(f) for f in files)
            
        finally:
            # 清理测试文件
            for file in migration_dir.glob("*"):
                file.unlink()
            migration_dir.rmdir()
    
    def test_extract_version_from_filename(self, migration_runner):
        """测试提取版本号"""
        assert migration_runner.extract_version_from_filename("001_initial.sql") == "001"
        assert migration_runner.extract_version_from_filename("002_add_index.sql") == "002"
    
    def test_calculate_checksum(self, migration_runner):
        """测试计算校验和"""
        content = "CREATE TABLE test (id INTEGER);"
        checksum1 = migration_runner.calculate_checksum(content)
        checksum2 = migration_runner.calculate_checksum(content)
        
        # 相同内容应该产生相同的校验和
        assert checksum1 == checksum2
        assert len(checksum1) == 64  # SHA256 hex digest length
        
        # 不同内容应该产生不同的校验和
        different_content = "CREATE TABLE test2 (id INTEGER);"
        checksum3 = migration_runner.calculate_checksum(different_content)
        assert checksum1 != checksum3


class TestMigrationValidator:
    """迁移验证器测试"""
    
    @pytest.fixture
    def mock_connection(self):
        """模拟数据库连接"""
        conn = AsyncMock()
        conn.execute = AsyncMock()
        conn.fetch = AsyncMock()
        conn.fetchrow = AsyncMock()
        return conn
    
    @pytest.fixture
    def migration_validator(self, mock_connection):
        """迁移验证器fixture"""
        return MigrationValidator(mock_connection)
    
    @pytest.mark.asyncio
    async def test_validate_table_exists(self, migration_validator, mock_connection):
        """测试表存在检查"""
        # 模拟表存在
        mock_connection.fetchval.return_value = True
        
        exists = await migration_validator.validate_table_exists("tracing_info")
        assert exists is True
        
        # 模拟表不存在
        mock_connection.fetchval.return_value = False
        
        exists = await migration_validator.validate_table_exists("nonexistent")
        assert exists is False
    
    @pytest.mark.asyncio
    async def test_validate_table_structure(self, migration_validator, mock_connection):
        """测试表结构验证"""
        # 模拟表列信息
        mock_connection.fetch.return_value = [
            {"column_name": "hb_trace_id", "data_type": "character varying"},
            {"column_name": "otel_trace_id", "data_type": "character varying"},
            {"column_name": "created_at", "data_type": "timestamp with time zone"}
        ]
        
        expected_columns = {
            "hb_trace_id": "VARCHAR",
            "otel_trace_id": "VARCHAR",
            "created_at": "TIMESTAMP WITH TIME ZONE"
        }
        
        result = await migration_validator.validate_table_structure("tracing_info", expected_columns)
        assert result["table_exists"] is True
        assert len(result["missing_columns"]) == 0
        assert len(result["type_mismatches"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_indexes(self, migration_validator, mock_connection):
        """测试索引验证"""
        # 模拟索引信息
        mock_connection.fetch.return_value = [
            {"indexname": "idx_tracing_info_hb_trace_id"},
            {"indexname": "idx_tracing_info_otel_trace_id"}
        ]
        
        expected_indexes = [
            "idx_tracing_info_hb_trace_id",
            "idx_tracing_info_otel_trace_id"
        ]
        
        result = await migration_validator.validate_indexes("tracing_info", expected_indexes)
        assert "missing_indexes" in result
        assert "extra_indexes" in result
        assert len(result["missing_indexes"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_constraints(self, migration_validator, mock_connection):
        """测试约束验证"""
        # 模拟约束信息
        mock_connection.fetch.return_value = [
            {"constraint_name": "tracing_info_pkey", "constraint_type": "PRIMARY KEY"},
            {"constraint_name": "tracing_info_duration_positive", "constraint_type": "CHECK"}
        ]
        
        expected_constraints = [
            "tracing_info_pkey",
            "tracing_info_duration_positive"
        ]
        
        result = await migration_validator.validate_constraints("tracing_info", expected_constraints)
        assert "missing_constraints" in result
        assert "extra_constraints" in result
        assert len(result["missing_constraints"]) == 0


class TestMigrationIntegration:
    """迁移集成测试"""
    
    @pytest.mark.asyncio
    async def test_migration_workflow(self):
        """测试完整的迁移工作流"""
        # 模拟数据库连接
        mock_connection = AsyncMock()
        
        # 模拟迁移表不存在
        mock_connection.fetchrow.return_value = None
        mock_connection.fetch.return_value = []
        
        runner = MigrationRunner(mock_connection)
        
        # 测试创建迁移表
        await runner.create_migration_table()
        mock_connection.execute.assert_called()
        
        # 测试获取迁移状态
        migrations = await runner.get_applied_migrations()
        assert isinstance(migrations, list)
    
    def test_migration_file_validation(self):
        """测试迁移文件验证"""
        # 检查实际的迁移文件是否存在
        migration_dir = Path("migrations")
        if migration_dir.exists():
            sql_files = list(migration_dir.glob("*.sql"))
            assert len(sql_files) > 0, "应该至少有一个迁移文件"
            
            # 检查文件命名格式
            for sql_file in sql_files:
                assert sql_file.name.endswith(".sql"), f"文件 {sql_file.name} 应该以 .sql 结尾"
                # 检查是否有版本号前缀
                parts = sql_file.stem.split("_", 1)
                assert len(parts) >= 2, f"文件 {sql_file.name} 应该包含版本号前缀"
                assert parts[0].isdigit(), f"文件 {sql_file.name} 的版本号应该是数字"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])