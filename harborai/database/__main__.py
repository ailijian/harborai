#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库迁移工具CLI入口

用法:
    python -m harborai.database apply    # 应用所有待执行的迁移
    python -m harborai.database status   # 查看迁移状态
    python -m harborai.database rollback <version>  # 回滚到指定版本
"""

import sys
import argparse
from .migration_tool import create_migrator, MigrationError


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HarborAI 数据库迁移工具')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # apply 命令
    apply_parser = subparsers.add_parser('apply', help='应用所有待执行的迁移')
    apply_parser.add_argument('--dry-run', action='store_true', help='仅显示将要执行的迁移，不实际执行')
    
    # status 命令
    subparsers.add_parser('status', help='查看迁移状态')
    
    # rollback 命令
    rollback_parser = subparsers.add_parser('rollback', help='回滚到指定版本')
    rollback_parser.add_argument('version', help='目标版本号')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        migrator = create_migrator()
        
        if args.command == 'apply':
            result = migrator.migrate(dry_run=getattr(args, 'dry_run', False))
            print(f"状态: {result['status']}")
            print(f"消息: {result['message']}")
            
            if 'executed_migrations' in result and result['executed_migrations']:
                print("\n执行的迁移:")
                for migration in result['executed_migrations']:
                    print(f"  - {migration['version']}: {migration['filename']} ({migration.get('execution_time_ms', 0)}ms)")
            
            if 'migrations_to_execute' in result and result['migrations_to_execute']:
                print("\n待执行的迁移:")
                for migration in result['migrations_to_execute']:
                    print(f"  - {migration['version']}: {migration['filename']}")
        
        elif args.command == 'status':
            result = migrator.check_migrations()
            print(f"总迁移数: {result['total_migrations']}")
            print(f"已应用: {result['applied_count']}")
            print(f"待执行: {result['pending_count']}")
            
            if result['applied_migrations']:
                print("\n已应用的迁移:")
                for migration in result['applied_migrations']:
                    checksum_status = "✓" if migration['checksum_match'] else "✗"
                    print(f"  - {migration['version']}: {migration['filename']} (应用于: {migration['applied_at']}) {checksum_status}")
            
            if result['pending_migrations']:
                print("\n待执行的迁移:")
                for migration in result['pending_migrations']:
                    print(f"  - {migration['version']}: {migration['filename']}")
        
        elif args.command == 'rollback':
            result = migrator.rollback(args.version)
            print(f"状态: {result['status']}")
            print(f"消息: {result['message']}")
            
            if 'rolled_back_migrations' in result and result['rolled_back_migrations']:
                print("\n回滚的迁移:")
                for migration in result['rolled_back_migrations']:
                    print(f"  - {migration['version']}: {migration['filename']}")
        
        # 根据结果状态设置退出码
        if result['status'] == 'failed':
            sys.exit(1)
        
    except MigrationError as e:
        print(f"迁移错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"未知错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()