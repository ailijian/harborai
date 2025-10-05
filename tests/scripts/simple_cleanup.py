#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 项目测试结果文件清理脚本 (简化版)

专业的测试结果文件清理工具，用于清理 HarborAI 项目内所有测试报告、缓存文件和临时结果文件。
遵循 VIBE Coding 规范，提供安全的清理操作，支持备份和干运行模式。

作者: HarborAI Team
版本: 1.0.0
创建日期: 2025-01-05
遵循: VIBE Coding 规范
"""

import os
import sys
import shutil
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
import glob


def setup_logging(project_root: Path) -> logging.Logger:
    """设置日志配置"""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"cleanup_test_results_{timestamp}.log"
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("测试结果清理脚本启动")
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"日志文件: {log_file}")
    
    return logger


def format_file_size(bytes_size: int) -> str:
    """格式化文件大小"""
    if bytes_size >= 1024**3:
        return f"{bytes_size / (1024**3):.2f} GB"
    elif bytes_size >= 1024**2:
        return f"{bytes_size / (1024**2):.2f} MB"
    elif bytes_size >= 1024:
        return f"{bytes_size / 1024:.2f} KB"
    else:
        return f"{bytes_size} 字节"


def get_directory_size(path: Path) -> int:
    """获取目录大小"""
    try:
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    except Exception:
        return 0


def scan_test_result_files(project_root: Path, logger: logging.Logger) -> Tuple[List[Path], List[Path]]:
    """扫描并返回所有需要清理的测试结果文件和目录"""
    logger.info("=" * 60)
    logger.info("  扫描测试结果文件")
    logger.info("=" * 60)
    
    files_to_delete = []
    directories_to_delete = []
    
    # 定义需要清理的目录
    directory_patterns = [
        "tests/reports",
        "tests/performance/performance_reports",
        "tests/performance/.benchmarks",
        "tests/performance/htmlcov_*",
        "tests/performance/metrics",
        "tests/performance/test_results",
        "reports",
        "htmlcov",
        ".benchmarks",
        "metrics",
        ".pytest_cache"
    ]
    
    # 定义需要清理的文件模式
    file_patterns = [
        ".coverage",
        "*.coverage",
        "tests/performance/*.log",
        "tests/performance/*.json",
        "tests/performance/*.html",
        "tests/performance/*.csv",
        "tests/performance/*.md",
        "tests/reports/*.txt",
        "tests/reports/*.log"
    ]
    
    logger.info("开始扫描目录...")
    
    # 扫描目录
    for pattern in directory_patterns:
        full_pattern = project_root / pattern
        try:
            # 使用 glob 处理通配符
            if '*' in pattern:
                for path in project_root.glob(pattern):
                    if path.is_dir() and path.exists():
                        directories_to_delete.append(path)
                        logger.info(f"发现目录: {path}")
            else:
                if full_pattern.exists() and full_pattern.is_dir():
                    directories_to_delete.append(full_pattern)
                    logger.info(f"发现目录: {full_pattern}")
        except Exception as e:
            logger.warning(f"扫描目录模式失败 {pattern}: {e}")
    
    # 递归查找 __pycache__ 目录
    try:
        for pycache_dir in project_root.rglob("__pycache__"):
            if pycache_dir.is_dir():
                directories_to_delete.append(pycache_dir)
                logger.info(f"发现 __pycache__ 目录: {pycache_dir}")
    except Exception as e:
        logger.warning(f"扫描 __pycache__ 目录时出错: {e}")
    
    # 递归查找 .pytest_cache 目录
    try:
        for pytest_cache_dir in project_root.rglob(".pytest_cache"):
            if pytest_cache_dir.is_dir():
                directories_to_delete.append(pytest_cache_dir)
                logger.info(f"发现 .pytest_cache 目录: {pytest_cache_dir}")
    except Exception as e:
        logger.warning(f"扫描 .pytest_cache 目录时出错: {e}")
    
    logger.info("开始扫描文件...")
    
    # 扫描文件
    for pattern in file_patterns:
        try:
            for path in project_root.glob(pattern):
                if path.is_file() and path.exists():
                    files_to_delete.append(path)
                    logger.info(f"发现文件: {path}")
        except Exception as e:
            logger.warning(f"扫描文件模式失败 {pattern}: {e}")
    
    logger.info(f"扫描完成 - 发现 {len(files_to_delete)} 个文件，{len(directories_to_delete)} 个目录")
    
    return files_to_delete, directories_to_delete


def create_backup(files: List[Path], directories: List[Path], project_root: Path, logger: logging.Logger) -> str:
    """创建备份"""
    logger.info("=" * 60)
    logger.info("  创建备份")
    logger.info("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = project_root / f"backup_test_results_{timestamp}"
    
    try:
        backup_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建备份目录: {backup_path}")
        
        # 备份文件
        for file_path in files:
            try:
                relative_path = file_path.relative_to(project_root)
                backup_file = backup_path / relative_path
                backup_file.parent.mkdir(parents=True, exist_ok=True)
                
                shutil.copy2(file_path, backup_file)
                logger.info(f"备份文件: {relative_path}")
            except Exception as e:
                logger.error(f"备份文件失败 {file_path}: {e}")
        
        # 备份目录
        for dir_path in directories:
            try:
                relative_path = dir_path.relative_to(project_root)
                backup_dir_path = backup_path / relative_path
                
                shutil.copytree(dir_path, backup_dir_path, dirs_exist_ok=True)
                logger.info(f"备份目录: {relative_path}")
            except Exception as e:
                logger.error(f"备份目录失败 {dir_path}: {e}")
        
        logger.info("备份完成")
        return str(backup_path)
        
    except Exception as e:
        logger.error(f"备份过程中发生错误: {e}")
        raise RuntimeError("备份失败")


def remove_test_results(files: List[Path], directories: List[Path], 
                       dry_run: bool, force: bool, logger: logging.Logger) -> dict:
    """清理测试结果文件"""
    logger.info("=" * 60)
    logger.info("  清理测试结果文件")
    logger.info("=" * 60)
    
    cleanup_stats = {
        'deleted_files': [],
        'deleted_directories': [],
        'total_size': 0,
        'errors': []
    }
    
    if dry_run:
        logger.warning("*** 干运行模式 - 仅显示将要删除的项目 ***")
        
        print("\n将要删除的文件:")
        for file_path in files:
            size = format_file_size(file_path.stat().st_size)
            print(f"  - {file_path} ({size})")
        
        print("\n将要删除的目录:")
        for dir_path in directories:
            size = format_file_size(get_directory_size(dir_path))
            print(f"  - {dir_path} ({size})")
        
        total_size = sum(f.stat().st_size for f in files)
        total_size += sum(get_directory_size(d) for d in directories)
        
        print(f"\n总计: {len(files)} 个文件，{len(directories)} 个目录，{format_file_size(total_size)}")
        return cleanup_stats
    
    # 确认删除
    if not force:
        total_items = len(files) + len(directories)
        total_size = sum(f.stat().st_size for f in files)
        total_size += sum(get_directory_size(d) for d in directories)
        
        print(f"\n即将删除 {total_items} 个项目，总大小 {format_file_size(total_size)}")
        confirmation = input("确认继续删除？(y/N): ").strip().lower()
        
        if confirmation not in ['y', 'yes']:
            logger.info("用户取消操作")
            return cleanup_stats
    
    # 删除文件
    logger.info("开始删除文件...")
    for file_path in files:
        try:
            size = file_path.stat().st_size
            file_path.unlink()
            cleanup_stats['deleted_files'].append(str(file_path))
            cleanup_stats['total_size'] += size
            logger.info(f"删除文件: {file_path} ({format_file_size(size)})")
        except Exception as e:
            error_msg = f"删除文件失败 {file_path}: {e}"
            logger.error(error_msg)
            cleanup_stats['errors'].append(error_msg)
    
    # 删除目录
    logger.info("开始删除目录...")
    for dir_path in directories:
        try:
            size = get_directory_size(dir_path)
            shutil.rmtree(dir_path)
            cleanup_stats['deleted_directories'].append(str(dir_path))
            cleanup_stats['total_size'] += size
            logger.info(f"删除目录: {dir_path} ({format_file_size(size)})")
        except Exception as e:
            error_msg = f"删除目录失败 {dir_path}: {e}"
            logger.error(error_msg)
            cleanup_stats['errors'].append(error_msg)
    
    logger.info("清理操作完成")
    return cleanup_stats


def generate_cleanup_report(cleanup_stats: dict, backup_path: str, 
                          start_time: datetime, project_root: Path, logger: logging.Logger):
    """生成清理操作报告"""
    logger.info("=" * 60)
    logger.info("  生成清理报告")
    logger.info("=" * 60)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # 创建报告内容
    report_content = f"""# HarborAI 测试结果清理报告

## 清理摘要
- **开始时间**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}
- **结束时间**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
- **持续时间**: {str(duration).split('.')[0]}
- **删除文件数**: {len(cleanup_stats['deleted_files'])}
- **删除目录数**: {len(cleanup_stats['deleted_directories'])}
- **释放空间**: {format_file_size(cleanup_stats['total_size'])}
- **错误数量**: {len(cleanup_stats['errors'])}

## 删除的文件
{chr(10).join(f"- {f}" for f in cleanup_stats['deleted_files'])}

## 删除的目录
{chr(10).join(f"- {d}" for d in cleanup_stats['deleted_directories'])}

## 错误信息
{chr(10).join(f"- {e}" for e in cleanup_stats['errors'])}

## 备份信息
{'备份路径: ' + backup_path if backup_path else '未创建备份'}

---
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
脚本版本: 1.0.0
"""
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = project_root / "logs" / f"cleanup_report_{timestamp}.md"
    report_file.write_text(report_content, encoding='utf-8')
    
    logger.info(f"清理报告已保存: {report_file}")
    
    # 显示摘要
    print(f"\n清理完成摘要:")
    print(f"  删除文件: {len(cleanup_stats['deleted_files'])} 个")
    print(f"  删除目录: {len(cleanup_stats['deleted_directories'])} 个")
    print(f"  释放空间: {format_file_size(cleanup_stats['total_size'])}")
    print(f"  错误数量: {len(cleanup_stats['errors'])} 个")
    print(f"  持续时间: {str(duration).split('.')[0]}")
    
    if backup_path:
        print(f"  备份路径: {backup_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="HarborAI 项目测试结果文件清理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python simple_cleanup.py --dry-run
    预览将要删除的文件
    
  python simple_cleanup.py --backup
    创建备份并清理
    
  python simple_cleanup.py --force
    强制清理所有测试结果文件
        """
    )
    
    parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='干运行模式，仅显示将要删除的文件，不执行实际删除操作'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        help='在删除前创建备份'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='强制删除，跳过确认提示'
    )
    
    parser.add_argument(
        '--project-root',
        type=str,
        default='.',
        help='项目根目录路径，默认为当前目录'
    )
    
    args = parser.parse_args()
    
    # 设置项目根目录
    project_root = Path(args.project_root).resolve()
    
    # 验证项目结构
    tests_dir = project_root / "tests"
    if not tests_dir.exists():
        print("错误: 未找到 tests 目录，请确认在正确的项目根目录下运行脚本")
        sys.exit(1)
    
    # 设置日志
    logger = setup_logging(project_root)
    
    try:
        start_time = datetime.now()
        
        # 扫描测试结果文件
        files, directories = scan_test_result_files(project_root, logger)
        
        if not files and not directories:
            logger.info("未发现需要清理的测试结果文件")
            return
        
        # 创建备份
        backup_path = ""
        if args.backup:
            backup_path = create_backup(files, directories, project_root, logger)
        
        # 执行清理
        cleanup_stats = remove_test_results(files, directories, args.dry_run, args.force, logger)
        
        # 生成报告
        if not args.dry_run:
            generate_cleanup_report(cleanup_stats, backup_path, start_time, project_root, logger)
        
        logger.info("测试结果清理脚本执行完成")
        
    except Exception as e:
        logger.error(f"脚本执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()