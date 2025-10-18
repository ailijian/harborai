#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库迁移工具

用于执行SQL迁移脚本，支持版本控制和回滚功能。
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from ..config.settings import get_settings
from ..utils.logger import get_logger
from .postgres_connection import get_postgres_url, get_postgres_connection

logger = get_logger(__name__)


class MigrationError(Exception):
    """迁移错误异常"""
    pass


class DatabaseMigrator:
    """数据库迁移工具类"""
    
    def __init__(self, migration_dirs: Optional[List[str]] = None):
        """初始化迁移工具
        
        Args:
            migration_dirs: 迁移脚本目录列表，默认为项目标准目录
        """
        self.migration_dirs = migration_dirs or [
            "migrations",
            "harborai/database/migrations"
        ]
        self.project_root = self._find_project_root()
        
    def _find_project_root(self) -> Path:
        """查找项目根目录"""
        current = Path.cwd()
        while current != current.parent:
            if (current / "pyproject.toml").exists() or (current / "setup.py").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def _ensure_migration_table(self, conn) -> None:
        """确保迁移记录表存在"""
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    version VARCHAR(100) NOT NULL UNIQUE,
                    filename VARCHAR(255) NOT NULL,
                    checksum VARCHAR(64) NOT NULL,
                    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    execution_time_ms INTEGER DEFAULT 0
                )
            """)
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_schema_migrations_version 
                ON schema_migrations(version)
            """)
            
            conn.commit()
    
    def _get_migration_files(self) -> List[Tuple[str, Path]]:
        """获取所有迁移文件
        
        Returns:
            List[Tuple[str, Path]]: (版本号, 文件路径) 的列表，按版本号排序
        """
        migration_files = []
        
        for migration_dir in self.migration_dirs:
            dir_path = self.project_root / migration_dir
            if not dir_path.exists():
                continue
                
            for file_path in dir_path.glob("*.sql"):
                # 从文件名提取版本号 (例如: 001_add_enhanced_tables.sql -> 001)
                match = re.match(r'^(\d+)_.*\.sql$', file_path.name)
                if match:
                    version = match.group(1)
                    migration_files.append((version, file_path))
        
        # 按版本号排序
        migration_files.sort(key=lambda x: int(x[0]))
        return migration_files
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        with open(file_path, 'rb') as f:
            content = f.read()
            return hashlib.sha256(content).hexdigest()
    
    def _split_sql_statements(self, sql_content: str) -> List[str]:
        """智能分割SQL语句，正确处理DO $$块和其他复杂语句
        
        Args:
            sql_content: SQL文件内容
            
        Returns:
            List[str]: 分割后的SQL语句列表
        """
        statements = []
        current_statement = ""
        in_dollar_quote = False
        dollar_tag = ""
        i = 0
        
        while i < len(sql_content):
            char = sql_content[i]
            
            # 检查是否进入或退出dollar-quoted字符串
            if char == '$':
                # 查找完整的dollar标签
                tag_start = i
                i += 1
                while i < len(sql_content) and sql_content[i] not in ['$', ' ', '\n', '\t']:
                    i += 1
                
                if i < len(sql_content) and sql_content[i] == '$':
                    tag = sql_content[tag_start:i+1]
                    
                    if not in_dollar_quote:
                        # 进入dollar-quoted字符串
                        in_dollar_quote = True
                        dollar_tag = tag
                        current_statement += tag
                    elif tag == dollar_tag:
                        # 退出dollar-quoted字符串
                        in_dollar_quote = False
                        dollar_tag = ""
                        current_statement += tag
                    else:
                        # 不匹配的dollar标签，继续
                        current_statement += tag
                else:
                    # 不是完整的dollar标签
                    current_statement += char
                    i = tag_start
            elif char == ';' and not in_dollar_quote:
                # 语句结束
                current_statement += char
                stmt = current_statement.strip()
                # 移除注释行，但保留包含有效SQL的语句
                if stmt and not self._is_comment_only(stmt):
                    statements.append(stmt)
                current_statement = ""
            else:
                current_statement += char
            
            i += 1
        
        # 处理最后一个语句（如果没有以分号结尾）
        if current_statement.strip():
            stmt = current_statement.strip()
            if stmt and not self._is_comment_only(stmt):
                statements.append(stmt)
        
        return statements
    
    def _is_comment_only(self, stmt: str) -> bool:
        """检查语句是否只包含注释
        
        Args:
            stmt: SQL语句
            
        Returns:
            bool: 如果语句只包含注释返回True，否则返回False
        """
        lines = stmt.split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('--'):
                return False
        return True
    
    def _get_applied_migrations(self, conn) -> Dict[str, Dict]:
        """获取已应用的迁移记录"""
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT version, filename, checksum, applied_at, execution_time_ms
                FROM schema_migrations
                ORDER BY version
            """)
            
            applied = {}
            for row in cursor.fetchall():
                applied[row['version']] = {
                    'filename': row['filename'],
                    'checksum': row['checksum'],
                    'applied_at': row['applied_at'],
                    'execution_time_ms': row['execution_time_ms']
                }
            
            return applied
    
    def _execute_migration(self, conn, version: str, file_path: Path) -> int:
        """执行单个迁移脚本
        
        Returns:
            int: 执行时间（毫秒）
        """
        start_time = datetime.now()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            # 智能分割SQL语句，处理DO $$块和其他复杂语句
            statements = self._split_sql_statements(sql_content)
            
            with conn.cursor() as cursor:
                for i, statement in enumerate(statements, 1):
                    if statement:
                        try:
                            logger.debug(f"执行语句 {i}/{len(statements)}: {statement[:100]}...")
                            cursor.execute(statement)
                        except Exception as e:
                            logger.error(f"语句 {i} 执行失败: {statement[:200]}...")
                            raise MigrationError(f"迁移 {version} 的第 {i} 个语句执行失败: {e}")
            
            conn.commit()
            
            end_time = datetime.now()
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            
            # 记录迁移
            checksum = self._calculate_checksum(file_path)
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO schema_migrations (version, filename, checksum, execution_time_ms)
                    VALUES (%s, %s, %s, %s)
                """, (version, file_path.name, checksum, execution_time))
            
            conn.commit()
            
            logger.info(f"迁移 {version} ({file_path.name}) 执行成功，耗时 {execution_time}ms")
            return execution_time
            
        except Exception as e:
            conn.rollback()
            raise MigrationError(f"迁移 {version} 执行失败: {e}")
    
    def check_migrations(self) -> Dict:
        """检查迁移状态
        
        Returns:
            Dict: 迁移状态信息
        """
        if not PSYCOPG2_AVAILABLE:
            raise MigrationError("psycopg2 未安装，无法连接 PostgreSQL")
        
        migration_files = self._get_migration_files()
        
        with get_postgres_connection() as conn:
            if conn is None:
                raise MigrationError("无法连接到 PostgreSQL 数据库")
            
            self._ensure_migration_table(conn)
            applied_migrations = self._get_applied_migrations(conn)
        
        pending_migrations = []
        applied_list = []
        
        for version, file_path in migration_files:
            if version in applied_migrations:
                applied_info = applied_migrations[version]
                current_checksum = self._calculate_checksum(file_path)
                
                applied_list.append({
                    'version': version,
                    'filename': file_path.name,
                    'applied_at': applied_info['applied_at'],
                    'execution_time_ms': applied_info['execution_time_ms'],
                    'checksum_match': current_checksum == applied_info['checksum']
                })
            else:
                pending_migrations.append({
                    'version': version,
                    'filename': file_path.name,
                    'file_path': str(file_path)
                })
        
        return {
            'total_migrations': len(migration_files),
            'applied_count': len(applied_list),
            'pending_count': len(pending_migrations),
            'applied_migrations': applied_list,
            'pending_migrations': pending_migrations
        }
    
    def migrate(self, target_version: Optional[str] = None, dry_run: bool = False) -> Dict:
        """执行数据库迁移
        
        Args:
            target_version: 目标版本，None表示迁移到最新版本
            dry_run: 是否为试运行模式
            
        Returns:
            Dict: 迁移结果
        """
        if not PSYCOPG2_AVAILABLE:
            raise MigrationError("psycopg2 未安装，无法连接 PostgreSQL")
        
        migration_files = self._get_migration_files()
        
        if not migration_files:
            return {
                'status': 'success',
                'message': '没有找到迁移文件',
                'executed_migrations': []
            }
        
        with get_postgres_connection() as conn:
            if conn is None:
                raise MigrationError("无法连接到 PostgreSQL 数据库")
            
            self._ensure_migration_table(conn)
            applied_migrations = self._get_applied_migrations(conn)
        
        # 确定需要执行的迁移
        migrations_to_execute = []
        for version, file_path in migration_files:
            if version not in applied_migrations:
                migrations_to_execute.append((version, file_path))
                
                # 如果指定了目标版本，检查是否已达到
                if target_version and version == target_version:
                    break
        
        if not migrations_to_execute:
            return {
                'status': 'success',
                'message': '所有迁移都已应用',
                'executed_migrations': []
            }
        
        if dry_run:
            return {
                'status': 'dry_run',
                'message': f'将执行 {len(migrations_to_execute)} 个迁移',
                'migrations_to_execute': [
                    {'version': v, 'filename': p.name} 
                    for v, p in migrations_to_execute
                ]
            }
        
        # 执行迁移
        executed_migrations = []
        total_time = 0
        
        with get_postgres_connection() as conn:
            if conn is None:
                raise MigrationError("无法连接到 PostgreSQL 数据库")
            
            try:
                for version, file_path in migrations_to_execute:
                    execution_time = self._execute_migration(conn, version, file_path)
                    executed_migrations.append({
                        'version': version,
                        'filename': file_path.name,
                        'execution_time_ms': execution_time
                    })
                    total_time += execution_time
                
                return {
                    'status': 'success',
                    'message': f'成功执行 {len(executed_migrations)} 个迁移',
                    'executed_migrations': executed_migrations,
                    'total_execution_time_ms': total_time
                }
                
            except Exception as e:
                logger.error(f"迁移执行失败: {e}")
                return {
                    'status': 'failed',
                    'message': str(e),
                    'executed_migrations': executed_migrations,
                    'total_execution_time_ms': total_time
                }
    
    def rollback(self, target_version: str) -> Dict:
        """回滚到指定版本（简单实现，仅删除迁移记录）
        
        Args:
            target_version: 目标版本
            
        Returns:
            Dict: 回滚结果
        """
        if not PSYCOPG2_AVAILABLE:
            raise MigrationError("psycopg2 未安装，无法连接 PostgreSQL")
        
        with get_postgres_connection() as conn:
            if conn is None:
                raise MigrationError("无法连接到 PostgreSQL 数据库")
            
            self._ensure_migration_table(conn)
            
            try:
                with conn.cursor() as cursor:
                    # 删除大于目标版本的迁移记录
                    cursor.execute("""
                        DELETE FROM schema_migrations 
                        WHERE CAST(version AS INTEGER) > %s
                        RETURNING version, filename
                    """, (int(target_version),))
                    
                    rolled_back = cursor.fetchall()
                    conn.commit()
                
                return {
                    'status': 'success',
                    'message': f'回滚到版本 {target_version}',
                    'rolled_back_migrations': [
                        {'version': row['version'], 'filename': row['filename']}
                        for row in rolled_back
                    ]
                }
                
            except Exception as e:
                conn.rollback()
                logger.error(f"回滚失败: {e}")
                return {
                    'status': 'failed',
                    'message': str(e)
                }


def create_migrator() -> DatabaseMigrator:
    """创建数据库迁移工具实例"""
    return DatabaseMigrator()