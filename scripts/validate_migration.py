#!/usr/bin/env python3
"""
数据库迁移验证脚本

用于验证数据库迁移的正确性和完整性
"""

import asyncio
import logging
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

import asyncpg
from asyncpg import Connection

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from harborai.config.settings import get_settings


class MigrationValidator:
    """数据库迁移验证器"""
    
    def __init__(self, connection: Connection):
        """
        初始化迁移验证器
        
        Args:
            connection: 数据库连接
        """
        self.conn = connection
        self.logger = logging.getLogger(__name__)
    
    async def validate_table_exists(self, table_name: str) -> bool:
        """
        验证表是否存在
        
        Args:
            table_name: 表名
            
        Returns:
            bool: 表是否存在
        """
        query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = $1
        );
        """
        result = await self.conn.fetchval(query, table_name)
        return result
    
    async def validate_table_structure(self, table_name: str, expected_columns: Dict[str, str]) -> Dict[str, Any]:
        """
        验证表结构
        
        Args:
            table_name: 表名
            expected_columns: 期望的列定义 {列名: 数据类型}
            
        Returns:
            Dict: 验证结果
        """
        query = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = $1
        ORDER BY ordinal_position;
        """
        
        columns = await self.conn.fetch(query, table_name)
        actual_columns = {
            col['column_name']: col['data_type'] 
            for col in columns
        }
        
        missing_columns = set(expected_columns.keys()) - set(actual_columns.keys())
        extra_columns = set(actual_columns.keys()) - set(expected_columns.keys())
        type_mismatches = []
        
        for col_name, expected_type in expected_columns.items():
            if col_name in actual_columns:
                actual_type = actual_columns[col_name]
                if not self._types_compatible(expected_type, actual_type):
                    type_mismatches.append({
                        'column': col_name,
                        'expected': expected_type,
                        'actual': actual_type
                    })
        
        return {
            'table_exists': True,
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns),
            'type_mismatches': type_mismatches,
            'all_columns': [dict(col) for col in columns]
        }
    
    def _types_compatible(self, expected: str, actual: str) -> bool:
        """
        检查数据类型是否兼容
        
        Args:
            expected: 期望的数据类型
            actual: 实际的数据类型
            
        Returns:
            bool: 类型是否兼容
        """
        # 类型映射表
        type_mappings = {
            'UUID': ['uuid'],
            'VARCHAR': ['character varying'],
            'TEXT': ['text'],
            'INTEGER': ['integer'],
            'TIMESTAMP WITH TIME ZONE': ['timestamp with time zone'],
            'JSONB': ['jsonb'],
            'BOOLEAN': ['boolean']
        }
        
        expected_upper = expected.upper()
        actual_lower = actual.lower()
        
        # 直接匹配
        if expected_upper.lower() == actual_lower:
            return True
        
        # 通过映射表匹配
        for exp_type, actual_types in type_mappings.items():
            if expected_upper.startswith(exp_type) and actual_lower in actual_types:
                return True
        
        return False
    
    async def validate_indexes(self, table_name: str, expected_indexes: List[str]) -> Dict[str, Any]:
        """
        验证索引
        
        Args:
            table_name: 表名
            expected_indexes: 期望的索引名列表
            
        Returns:
            Dict: 验证结果
        """
        query = """
        SELECT indexname
        FROM pg_indexes
        WHERE tablename = $1 AND schemaname = 'public';
        """
        
        indexes = await self.conn.fetch(query, table_name)
        actual_indexes = {idx['indexname'] for idx in indexes}
        expected_indexes_set = set(expected_indexes)
        
        missing_indexes = expected_indexes_set - actual_indexes
        extra_indexes = actual_indexes - expected_indexes_set
        
        return {
            'missing_indexes': list(missing_indexes),
            'extra_indexes': list(extra_indexes),
            'all_indexes': list(actual_indexes)
        }
    
    async def validate_constraints(self, table_name: str, expected_constraints: List[str]) -> Dict[str, Any]:
        """
        验证约束
        
        Args:
            table_name: 表名
            expected_constraints: 期望的约束名列表
            
        Returns:
            Dict: 验证结果
        """
        query = """
        SELECT constraint_name, constraint_type
        FROM information_schema.table_constraints
        WHERE table_schema = 'public' AND table_name = $1;
        """
        
        constraints = await self.conn.fetch(query, table_name)
        actual_constraints = {const['constraint_name'] for const in constraints}
        expected_constraints_set = set(expected_constraints)
        
        missing_constraints = expected_constraints_set - actual_constraints
        extra_constraints = actual_constraints - expected_constraints_set
        
        return {
            'missing_constraints': list(missing_constraints),
            'extra_constraints': list(extra_constraints),
            'all_constraints': [dict(const) for const in constraints]
        }
    
    async def validate_tracing_info_table(self) -> Dict[str, Any]:
        """
        验证tracing_info表的完整性
        
        Returns:
            Dict: 验证结果
        """
        table_name = 'tracing_info'
        
        # 检查表是否存在
        table_exists = await self.validate_table_exists(table_name)
        if not table_exists:
            return {
                'table_exists': False,
                'error': f'Table {table_name} does not exist'
            }
        
        # 期望的列定义
        expected_columns = {
            'id': 'UUID',
            'log_id': 'UUID',
            'hb_trace_id': 'VARCHAR',
            'otel_trace_id': 'VARCHAR',
            'span_id': 'VARCHAR',
            'parent_span_id': 'VARCHAR',
            'operation_name': 'VARCHAR',
            'start_time': 'TIMESTAMP WITH TIME ZONE',
            'duration_ms': 'INTEGER',
            'status': 'VARCHAR',
            'trace_flags': 'VARCHAR',
            'trace_state': 'TEXT',
            'api_tags': 'JSONB',
            'internal_tags': 'JSONB',
            'logs': 'JSONB',
            'created_at': 'TIMESTAMP WITH TIME ZONE',
            'updated_at': 'TIMESTAMP WITH TIME ZONE'
        }
        
        # 期望的索引
        expected_indexes = [
            'idx_tracing_info_hb_trace_id',
            'idx_tracing_info_otel_trace_id',
            'idx_tracing_info_log_id',
            'idx_tracing_info_operation_name',
            'idx_tracing_info_start_time',
            'idx_tracing_info_status',
            'idx_tracing_info_trace_span',
            'idx_tracing_info_time_status'
        ]
        
        # 期望的约束
        expected_constraints = [
            'tracing_info_pkey',
            'tracing_info_duration_positive',
            'tracing_info_status_valid',
            'tracing_info_trace_flags_valid',
            'tracing_info_unique_span'
        ]
        
        # 执行验证
        structure_result = await self.validate_table_structure(table_name, expected_columns)
        indexes_result = await self.validate_indexes(table_name, expected_indexes)
        constraints_result = await self.validate_constraints(table_name, expected_constraints)
        
        return {
            'table_name': table_name,
            'structure': structure_result,
            'indexes': indexes_result,
            'constraints': constraints_result
        }
    
    async def validate_foreign_keys(self) -> Dict[str, Any]:
        """
        验证外键约束
        
        Returns:
            Dict: 验证结果
        """
        query = """
        SELECT
            tc.constraint_name,
            tc.table_name,
            kcu.column_name,
            ccu.table_name AS foreign_table_name,
            ccu.column_name AS foreign_column_name
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
            ON tc.constraint_name = kcu.constraint_name
            AND tc.table_schema = kcu.table_schema
        JOIN information_schema.constraint_column_usage AS ccu
            ON ccu.constraint_name = tc.constraint_name
            AND ccu.table_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = 'public'
            AND tc.table_name = 'tracing_info';
        """
        
        foreign_keys = await self.conn.fetch(query)
        return {
            'foreign_keys': [dict(fk) for fk in foreign_keys]
        }
    
    async def test_basic_operations(self) -> Dict[str, Any]:
        """
        测试基本的数据库操作
        
        Returns:
            Dict: 测试结果
        """
        test_results = []
        
        try:
            # 测试插入
            insert_query = """
            INSERT INTO tracing_info (
                log_id, hb_trace_id, otel_trace_id, span_id,
                operation_name, status
            ) VALUES (
                gen_random_uuid(), 'test_trace_123', 'otel_trace_456', 'span_789',
                'test.operation', 'ok'
            ) RETURNING id;
            """
            
            test_id = await self.conn.fetchval(insert_query)
            test_results.append({
                'operation': 'insert',
                'success': True,
                'test_id': str(test_id)
            })
            
            # 测试查询
            select_query = "SELECT * FROM tracing_info WHERE id = $1;"
            record = await self.conn.fetchrow(select_query, test_id)
            test_results.append({
                'operation': 'select',
                'success': record is not None,
                'record_found': record is not None
            })
            
            # 测试更新
            update_query = """
            UPDATE tracing_info 
            SET duration_ms = 100, status = 'completed'
            WHERE id = $1;
            """
            await self.conn.execute(update_query, test_id)
            test_results.append({
                'operation': 'update',
                'success': True
            })
            
            # 测试删除
            delete_query = "DELETE FROM tracing_info WHERE id = $1;"
            await self.conn.execute(delete_query, test_id)
            test_results.append({
                'operation': 'delete',
                'success': True
            })
            
        except Exception as e:
            test_results.append({
                'operation': 'error',
                'success': False,
                'error': str(e)
            })
        
        return {
            'basic_operations': test_results
        }


async def main():
    """主函数"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 获取配置
        settings = get_settings()
        
        # 连接数据库
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            user=settings.postgres_user,
            password=settings.postgres_password,
            database=settings.postgres_database
        )
        
        logger.info("已连接到数据库，开始验证迁移...")
        
        # 创建验证器
        validator = MigrationValidator(conn)
        
        # 执行验证
        results = {}
        
        # 验证tracing_info表
        logger.info("验证tracing_info表结构...")
        results['tracing_info'] = await validator.validate_tracing_info_table()
        
        # 验证外键
        logger.info("验证外键约束...")
        results['foreign_keys'] = await validator.validate_foreign_keys()
        
        # 测试基本操作
        logger.info("测试基本数据库操作...")
        results['basic_operations'] = await validator.test_basic_operations()
        
        # 输出结果
        print("\n" + "="*60)
        print("数据库迁移验证结果")
        print("="*60)
        
        # 验证tracing_info表
        tracing_result = results['tracing_info']
        if tracing_result.get('table_exists', False):
            print("✅ tracing_info表存在")
            
            structure = tracing_result['structure']
            if not structure['missing_columns'] and not structure['type_mismatches']:
                print("✅ 表结构正确")
            else:
                print("❌ 表结构问题:")
                if structure['missing_columns']:
                    print(f"  缺少列: {structure['missing_columns']}")
                if structure['type_mismatches']:
                    print(f"  类型不匹配: {structure['type_mismatches']}")
            
            indexes = tracing_result['indexes']
            if not indexes['missing_indexes']:
                print("✅ 索引完整")
            else:
                print(f"❌ 缺少索引: {indexes['missing_indexes']}")
            
            constraints = tracing_result['constraints']
            if not constraints['missing_constraints']:
                print("✅ 约束完整")
            else:
                print(f"❌ 缺少约束: {constraints['missing_constraints']}")
        else:
            print("❌ tracing_info表不存在")
        
        # 验证外键
        fk_result = results['foreign_keys']
        if fk_result['foreign_keys']:
            print("✅ 外键约束存在")
        else:
            print("⚠️  未找到外键约束")
        
        # 验证基本操作
        ops_result = results['basic_operations']
        all_ops_success = all(
            op.get('success', False) 
            for op in ops_result['basic_operations']
            if op['operation'] != 'error'
        )
        if all_ops_success:
            print("✅ 基本数据库操作正常")
        else:
            print("❌ 基本数据库操作失败")
            for op in ops_result['basic_operations']:
                if not op.get('success', False):
                    print(f"  失败操作: {op}")
        
        print("\n验证完成!")
        
        # 关闭连接
        await conn.close()
        
        # 返回退出码
        if (tracing_result.get('table_exists', False) and 
            not tracing_result['structure']['missing_columns'] and
            not tracing_result['indexes']['missing_indexes'] and
            not tracing_result['constraints']['missing_constraints'] and
            all_ops_success):
            sys.exit(0)  # 成功
        else:
            sys.exit(1)  # 失败
            
    except Exception as e:
        logger.error(f"验证过程中发生错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())