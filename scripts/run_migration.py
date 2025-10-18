#!/usr/bin/env python3
"""
数据库迁移执行脚本

用于执行数据库迁移并验证结果
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import asyncpg
from asyncpg import Connection

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from harborai.config.settings import get_settings


class MigrationRunner:
    """数据库迁移执行器"""
    
    def __init__(self, connection: Connection):
        """
        初始化迁移执行器
        
        Args:
            connection: 数据库连接
        """
        self.conn = connection
        self.logger = logging.getLogger(__name__)
        self.migrations_dir = Path(__file__).parent.parent / "migrations"
    
    async def create_migration_table(self):
        """创建迁移记录表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version VARCHAR(255) PRIMARY KEY,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            description TEXT,
            checksum VARCHAR(64)
        );
        """
        await self.conn.execute(create_table_sql)
        self.logger.info("迁移记录表已创建或已存在")
    
    async def get_applied_migrations(self) -> List[str]:
        """
        获取已应用的迁移版本
        
        Returns:
            List[str]: 已应用的迁移版本列表
        """
        query = "SELECT version FROM schema_migrations ORDER BY version;"
        rows = await self.conn.fetch(query)
        return [row['version'] for row in rows]
    
    def get_migration_files(self) -> List[Path]:
        """
        获取迁移文件列表
        
        Returns:
            List[Path]: 迁移文件路径列表
        """
        if not self.migrations_dir.exists():
            self.logger.warning(f"迁移目录不存在: {self.migrations_dir}")
            return []
        
        migration_files = []
        for file_path in self.migrations_dir.glob("*.sql"):
            if file_path.is_file():
                migration_files.append(file_path)
        
        # 按文件名排序
        migration_files.sort(key=lambda x: x.name)
        return migration_files
    
    def extract_version_from_filename(self, filename: str) -> str:
        """
        从文件名提取版本号
        
        Args:
            filename: 文件名
            
        Returns:
            str: 版本号
        """
        # 假设文件名格式为: 001_description.sql
        return filename.split('_')[0]
    
    def calculate_checksum(self, content: str) -> str:
        """
        计算文件内容的校验和
        
        Args:
            content: 文件内容
            
        Returns:
            str: 校验和
        """
        import hashlib
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    async def apply_migration(self, migration_file: Path) -> Dict[str, Any]:
        """
        应用单个迁移
        
        Args:
            migration_file: 迁移文件路径
            
        Returns:
            Dict: 应用结果
        """
        try:
            # 读取迁移文件
            with open(migration_file, 'r', encoding='utf-8') as f:
                sql_content = f.read()
            
            if not sql_content.strip():
                return {
                    'success': False,
                    'error': '迁移文件为空'
                }
            
            # 提取版本和描述
            version = self.extract_version_from_filename(migration_file.name)
            description = migration_file.stem.replace(f"{version}_", "")
            checksum = self.calculate_checksum(sql_content)
            
            # 开始事务
            async with self.conn.transaction():
                # 执行迁移SQL
                await self.conn.execute(sql_content)
                
                # 记录迁移
                insert_sql = """
                INSERT INTO schema_migrations (version, description, checksum)
                VALUES ($1, $2, $3)
                ON CONFLICT (version) DO UPDATE SET
                    applied_at = NOW(),
                    description = EXCLUDED.description,
                    checksum = EXCLUDED.checksum;
                """
                await self.conn.execute(insert_sql, version, description, checksum)
            
            self.logger.info(f"迁移 {version} 应用成功: {description}")
            return {
                'success': True,
                'version': version,
                'description': description,
                'checksum': checksum
            }
            
        except Exception as e:
            self.logger.error(f"应用迁移 {migration_file.name} 失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'file': str(migration_file)
            }
    
    async def run_migrations(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """
        运行迁移
        
        Args:
            target_version: 目标版本，None表示运行所有未应用的迁移
            
        Returns:
            Dict: 运行结果
        """
        # 创建迁移记录表
        await self.create_migration_table()
        
        # 获取已应用的迁移
        applied_migrations = await self.get_applied_migrations()
        self.logger.info(f"已应用的迁移: {applied_migrations}")
        
        # 获取迁移文件
        migration_files = self.get_migration_files()
        if not migration_files:
            return {
                'success': True,
                'message': '没有找到迁移文件',
                'applied_migrations': []
            }
        
        # 确定需要应用的迁移
        pending_migrations = []
        for migration_file in migration_files:
            version = self.extract_version_from_filename(migration_file.name)
            
            # 检查是否已应用
            if version in applied_migrations:
                continue
            
            # 检查是否超过目标版本
            if target_version and version > target_version:
                break
            
            pending_migrations.append(migration_file)
        
        if not pending_migrations:
            return {
                'success': True,
                'message': '所有迁移都已应用',
                'applied_migrations': []
            }
        
        # 应用迁移
        applied_results = []
        failed_migrations = []
        
        for migration_file in pending_migrations:
            result = await self.apply_migration(migration_file)
            applied_results.append(result)
            
            if not result['success']:
                failed_migrations.append(result)
                # 如果有迁移失败，停止后续迁移
                break
        
        return {
            'success': len(failed_migrations) == 0,
            'applied_migrations': applied_results,
            'failed_migrations': failed_migrations,
            'total_applied': len([r for r in applied_results if r['success']])
        }
    
    async def rollback_migration(self, target_version: str) -> Dict[str, Any]:
        """
        回滚迁移到指定版本
        
        Args:
            target_version: 目标版本
            
        Returns:
            Dict: 回滚结果
        """
        # 获取已应用的迁移
        applied_migrations = await self.get_applied_migrations()
        
        # 找到需要回滚的迁移
        migrations_to_rollback = [
            version for version in applied_migrations 
            if version > target_version
        ]
        
        if not migrations_to_rollback:
            return {
                'success': True,
                'message': f'已经在版本 {target_version} 或更早版本',
                'rolled_back': []
            }
        
        # 注意：这里只是从记录表中删除，实际的数据库结构回滚需要专门的回滚脚本
        rolled_back = []
        try:
            for version in reversed(migrations_to_rollback):
                delete_sql = "DELETE FROM schema_migrations WHERE version = $1;"
                await self.conn.execute(delete_sql, version)
                rolled_back.append(version)
                self.logger.info(f"已从迁移记录中移除版本: {version}")
            
            return {
                'success': True,
                'message': f'已回滚到版本 {target_version}',
                'rolled_back': rolled_back
            }
            
        except Exception as e:
            self.logger.error(f"回滚失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'rolled_back': rolled_back
            }
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """
        获取迁移状态
        
        Returns:
            Dict: 迁移状态信息
        """
        # 确保迁移记录表存在
        await self.create_migration_table()
        
        # 获取已应用的迁移
        applied_migrations = await self.get_applied_migrations()
        
        # 获取所有迁移文件
        migration_files = self.get_migration_files()
        all_migrations = [
            self.extract_version_from_filename(f.name) 
            for f in migration_files
        ]
        
        # 计算待应用的迁移
        pending_migrations = [
            version for version in all_migrations 
            if version not in applied_migrations
        ]
        
        return {
            'applied_migrations': applied_migrations,
            'pending_migrations': pending_migrations,
            'all_migrations': all_migrations,
            'total_applied': len(applied_migrations),
            'total_pending': len(pending_migrations)
        }


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据库迁移工具')
    parser.add_argument('command', choices=['migrate', 'rollback', 'status'], 
                       help='要执行的命令')
    parser.add_argument('--target', help='目标版本')
    parser.add_argument('--validate', action='store_true', 
                       help='迁移后运行验证')
    
    args = parser.parse_args()
    
    # 配置日志
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
        
        logger.info("已连接到数据库")
        
        # 创建迁移执行器
        runner = MigrationRunner(conn)
        
        if args.command == 'migrate':
            logger.info("开始执行迁移...")
            result = await runner.run_migrations(args.target)
            
            if result['success']:
                logger.info(f"迁移完成，共应用 {result['total_applied']} 个迁移")
                print("✅ 迁移成功完成")
                
                # 如果指定了验证，运行验证脚本
                if args.validate:
                    logger.info("开始验证迁移...")
                    from validate_migration import MigrationValidator
                    validator = MigrationValidator(conn)
                    
                    # 验证tracing_info表
                    validation_result = await validator.validate_tracing_info_table()
                    if validation_result.get('table_exists', False):
                        print("✅ 迁移验证通过")
                    else:
                        print("❌ 迁移验证失败")
                        sys.exit(1)
            else:
                logger.error("迁移失败")
                print("❌ 迁移失败")
                for failed in result.get('failed_migrations', []):
                    print(f"  失败的迁移: {failed}")
                sys.exit(1)
        
        elif args.command == 'rollback':
            if not args.target:
                print("回滚命令需要指定目标版本")
                sys.exit(1)
            
            logger.info(f"开始回滚到版本 {args.target}...")
            result = await runner.rollback_migration(args.target)
            
            if result['success']:
                logger.info(result['message'])
                print("✅ 回滚成功完成")
            else:
                logger.error(f"回滚失败: {result.get('error', '未知错误')}")
                print("❌ 回滚失败")
                sys.exit(1)
        
        elif args.command == 'status':
            logger.info("获取迁移状态...")
            status = await runner.get_migration_status()
            
            print("\n" + "="*50)
            print("数据库迁移状态")
            print("="*50)
            print(f"已应用迁移: {status['total_applied']}")
            print(f"待应用迁移: {status['total_pending']}")
            
            if status['applied_migrations']:
                print(f"\n已应用的迁移:")
                for version in status['applied_migrations']:
                    print(f"  ✅ {version}")
            
            if status['pending_migrations']:
                print(f"\n待应用的迁移:")
                for version in status['pending_migrations']:
                    print(f"  ⏳ {version}")
            
            print()
        
        # 关闭连接
        await conn.close()
        
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        print(f"❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())