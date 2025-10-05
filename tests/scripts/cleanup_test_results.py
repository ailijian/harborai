#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 项目测试结果文件清理脚本 (Python版本)

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
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import glob


class TestResultsCleaner:
    """测试结果文件清理器"""
    
    def __init__(self, project_root: str, dry_run: bool = False, 
                 backup: bool = False, force: bool = False, 
                 backup_path: Optional[str] = None):
        """
        初始化清理器
        
        Args:
            project_root: 项目根目录路径
            dry_run: 是否为干运行模式
            backup: 是否创建备份
            force: 是否强制删除
            backup_path: 备份目录路径
        """
        self.project_root = Path(project_root).resolve()
        self.dry_run = dry_run
        self.backup = backup
        self.force = force
        self.backup_path = backup_path
        
        # 初始化日志
        self._setup_logging()
        
        # 清理报告
        self.cleanup_report = {
            'start_time': datetime.now(),
            'deleted_files': [],
            'deleted_directories': [],
            'errors': [],
            'total_size': 0,
            'backup_path': ''
        }
        
        # 验证项目结构
        self._validate_project_structure()
    
    def _setup_logging(self) -> None:
        """设置日志配置"""
        log_dir = self.project_root / "logs"
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
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("测试结果清理脚本启动")
        self.logger.info(f"项目根目录: {self.project_root}")
        self.logger.info(f"日志文件: {log_file}")
    
    def _validate_project_structure(self) -> None:
        """验证项目结构"""
        tests_dir = self.project_root / "tests"
        if not tests_dir.exists():
            self.logger.error("错误: 未找到 tests 目录，请确认在正确的项目根目录下运行脚本")
            raise FileNotFoundError("项目结构验证失败")
        
        self.logger.info("项目结构验证通过")
    
    def _get_directory_size(self, path: Path) -> int:
        """获取目录大小"""
        try:
            total_size = 0
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception as e:
            self.logger.warning(f"计算目录大小失败 {path}: {e}")
            return 0
    
    def _format_file_size(self, bytes_size: int) -> str:
        """格式化文件大小"""
        if bytes_size >= 1024**3:
            return f"{bytes_size / (1024**3):.2f} GB"
        elif bytes_size >= 1024**2:
            return f"{bytes_size / (1024**2):.2f} MB"
        elif bytes_size >= 1024:
            return f"{bytes_size / 1024:.2f} KB"
        else:
            return f"{bytes_size} 字节"
    
    def scan_test_result_files(self) -> Tuple[List[Path], List[Path]]:
        """
        扫描并返回所有需要清理的测试结果文件和目录
        
        Returns:
            Tuple[List[Path], List[Path]]: (文件列表, 目录列表)
        """
        self.logger.info("=" * 60)
        self.logger.info("  扫描测试结果文件")
        self.logger.info("=" * 60)
        
        files_to_delete = []
        directories_to_delete = []
        
        # 定义需要清理的目录模式
        directory_patterns = [
            # 测试报告目录
            "tests/reports/*",
            "tests/performance/performance_reports",
            "tests/performance/.benchmarks",
            "tests/performance/htmlcov_*",
            "tests/performance/metrics",
            "tests/performance/test_results",
            "tests/logs/*",
            
            # 根目录下的测试相关目录
            "reports",
            "htmlcov",
            ".benchmarks",
            "metrics",
            
            # 缓存目录
            ".pytest_cache",
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
            "tests/reports/*.log",
        ]
        
        self.logger.info("开始扫描目录...")
        
        # 扫描目录
        for pattern in directory_patterns:
            full_pattern = self.project_root / pattern
            try:
                for path in glob.glob(str(full_pattern)):
                    path_obj = Path(path)
                    if path_obj.is_dir() and path_obj.exists():
                        directories_to_delete.append(path_obj)
                        self.logger.info(f"发现目录: {path_obj}")
            except Exception as e:
                self.logger.warning(f"扫描目录模式失败 {pattern}: {e}")
        
        # 递归查找 __pycache__ 目录
        try:
            for pycache_dir in self.project_root.rglob("__pycache__"):
                if pycache_dir.is_dir():
                    directories_to_delete.append(pycache_dir)
                    self.logger.info(f"发现 __pycache__ 目录: {pycache_dir}")
        except Exception as e:
            self.logger.warning(f"扫描 __pycache__ 目录时出错: {e}")
        
        # 递归查找 .pytest_cache 目录
        try:
            for pytest_cache_dir in self.project_root.rglob(".pytest_cache"):
                if pytest_cache_dir.is_dir():
                    directories_to_delete.append(pytest_cache_dir)
                    self.logger.info(f"发现 .pytest_cache 目录: {pytest_cache_dir}")
        except Exception as e:
            self.logger.warning(f"扫描 .pytest_cache 目录时出错: {e}")
        
        self.logger.info("开始扫描文件...")
        
        # 扫描文件
        for pattern in file_patterns:
            full_pattern = self.project_root / pattern
            try:
                for path in glob.glob(str(full_pattern)):
                    path_obj = Path(path)
                    if path_obj.is_file() and path_obj.exists():
                        files_to_delete.append(path_obj)
                        self.logger.info(f"发现文件: {path_obj}")
            except Exception as e:
                self.logger.warning(f"扫描文件模式失败 {pattern}: {e}")
        
        # 查找根目录下的 .coverage 文件
        coverage_file = self.project_root / ".coverage"
        if coverage_file.exists() and coverage_file.is_file():
            files_to_delete.append(coverage_file)
            self.logger.info(f"发现覆盖率文件: {coverage_file}")
        
        self.logger.info(f"扫描完成 - 发现 {len(files_to_delete)} 个文件，{len(directories_to_delete)} 个目录")
        
        return files_to_delete, directories_to_delete
    
    def create_backup(self, files: List[Path], directories: List[Path]) -> None:
        """创建备份"""
        if not self.backup:
            return
        
        self.logger.info("=" * 60)
        self.logger.info("  创建备份")
        self.logger.info("=" * 60)
        
        try:
            # 设置备份路径
            if not self.backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.cleanup_report['backup_path'] = str(self.project_root / f"backup_test_results_{timestamp}")
            else:
                self.cleanup_report['backup_path'] = self.backup_path
            
            backup_dir = Path(self.cleanup_report['backup_path'])
            backup_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"创建备份目录: {backup_dir}")
            
            # 备份文件
            for file_path in files:
                try:
                    relative_path = file_path.relative_to(self.project_root)
                    backup_file = backup_dir / relative_path
                    backup_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(file_path, backup_file)
                    self.logger.info(f"备份文件: {relative_path}")
                except Exception as e:
                    error_msg = f"备份文件失败 {file_path}: {e}"
                    self.logger.error(error_msg)
                    self.cleanup_report['errors'].append(error_msg)
            
            # 备份目录
            for dir_path in directories:
                try:
                    relative_path = dir_path.relative_to(self.project_root)
                    backup_dir_path = backup_dir / relative_path
                    
                    shutil.copytree(dir_path, backup_dir_path, dirs_exist_ok=True)
                    self.logger.info(f"备份目录: {relative_path}")
                except Exception as e:
                    error_msg = f"备份目录失败 {dir_path}: {e}"
                    self.logger.error(error_msg)
                    self.cleanup_report['errors'].append(error_msg)
            
            self.logger.info("备份完成")
            
        except Exception as e:
            self.logger.error(f"备份过程中发生错误: {e}")
            raise RuntimeError("备份失败")
    
    def remove_test_results(self, files: List[Path], directories: List[Path]) -> None:
        """清理测试结果文件"""
        self.logger.info("=" * 60)
        self.logger.info("  清理测试结果文件")
        self.logger.info("=" * 60)
        
        if self.dry_run:
            self.logger.warning("*** 干运行模式 - 仅显示将要删除的项目 ***")
            
            print("\n将要删除的文件:")
            for file_path in files:
                size = self._format_file_size(file_path.stat().st_size)
                print(f"  - {file_path} ({size})")
            
            print("\n将要删除的目录:")
            for dir_path in directories:
                size = self._format_file_size(self._get_directory_size(dir_path))
                print(f"  - {dir_path} ({size})")
            
            total_size = sum(f.stat().st_size for f in files)
            total_size += sum(self._get_directory_size(d) for d in directories)
            
            print(f"\n总计: {len(files)} 个文件，{len(directories)} 个目录，{self._format_file_size(total_size)}")
            return
        
        # 确认删除
        if not self.force:
            total_items = len(files) + len(directories)
            total_size = sum(f.stat().st_size for f in files)
            total_size += sum(self._get_directory_size(d) for d in directories)
            
            print(f"\n即将删除 {total_items} 个项目，总大小 {self._format_file_size(total_size)}")
            confirmation = input("确认继续删除？(y/N): ").strip().lower()
            
            if confirmation not in ['y', 'yes']:
                self.logger.info("用户取消操作")
                return
        
        # 删除文件
        self.logger.info("开始删除文件...")
        for file_path in files:
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                self.cleanup_report['deleted_files'].append(str(file_path))
                self.cleanup_report['total_size'] += size
                self.logger.info(f"删除文件: {file_path} ({self._format_file_size(size)})")
            except Exception as e:
                error_msg = f"删除文件失败 {file_path}: {e}"
                self.logger.error(error_msg)
                self.cleanup_report['errors'].append(error_msg)
        
        # 删除目录
        self.logger.info("开始删除目录...")
        for dir_path in directories:
            try:
                size = self._get_directory_size(dir_path)
                shutil.rmtree(dir_path)
                self.cleanup_report['deleted_directories'].append(str(dir_path))
                self.cleanup_report['total_size'] += size
                self.logger.info(f"删除目录: {dir_path} ({self._format_file_size(size)})")
            except Exception as e:
                error_msg = f"删除目录失败 {dir_path}: {e}"
                self.logger.error(error_msg)
                self.cleanup_report['errors'].append(error_msg)
        
        self.logger.info("清理操作完成")
    
    def generate_cleanup_report(self) -> None:
        """生成清理操作报告"""
        self.logger.info("=" * 60)
        self.logger.info("  生成清理报告")
        self.logger.info("=" * 60)
        
        end_time = datetime.now()
        duration = end_time - self.cleanup_report['start_time']
        
        # 创建报告内容
        report_content = f"""# HarborAI 测试结果清理报告

## 清理摘要
- **开始时间**: {self.cleanup_report['start_time'].strftime('%Y-%m-%d %H:%M:%S')}
- **结束时间**: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
- **持续时间**: {str(duration).split('.')[0]}
- **删除文件数**: {len(self.cleanup_report['deleted_files'])}
- **删除目录数**: {len(self.cleanup_report['deleted_directories'])}
- **释放空间**: {self._format_file_size(self.cleanup_report['total_size'])}
- **错误数量**: {len(self.cleanup_report['errors'])}

## 删除的文件
{chr(10).join(f"- {f}" for f in self.cleanup_report['deleted_files'])}

## 删除的目录
{chr(10).join(f"- {d}" for d in self.cleanup_report['deleted_directories'])}

## 错误信息
{chr(10).join(f"- {e}" for e in self.cleanup_report['errors'])}

## 备份信息
{'备份路径: ' + self.cleanup_report['backup_path'] if self.cleanup_report['backup_path'] else '未创建备份'}

---
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
脚本版本: 1.0.0
"""
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / "logs" / f"cleanup_report_{timestamp}.md"
        report_file.write_text(report_content, encoding='utf-8')
        
        self.logger.info(f"清理报告已保存: {report_file}")
        
        # 显示摘要
        print(f"\n清理完成摘要:")
        print(f"  删除文件: {len(self.cleanup_report['deleted_files'])} 个")
        print(f"  删除目录: {len(self.cleanup_report['deleted_directories'])} 个")
        print(f"  释放空间: {self._format_file_size(self.cleanup_report['total_size'])}")
        print(f"  错误数量: {len(self.cleanup_report['errors'])} 个")
        print(f"  持续时间: {str(duration).split('.')[0]}")
        
        if self.cleanup_report['backup_path']:
            print(f"  备份路径: {self.cleanup_report['backup_path']}")
    
    def run(self) -> None:
        """执行清理操作"""
        try:
            self.cleanup_report['start_time'] = datetime.now()
            
            # 扫描测试结果文件
            files, directories = self.scan_test_result_files()
            
            if not files and not directories:
                self.logger.info("未发现需要清理的测试结果文件")
                return
            
            # 创建备份
            if self.backup:
                self.create_backup(files, directories)
            
            # 执行清理
            self.remove_test_results(files, directories)
            
            # 生成报告
            if not self.dry_run:
                self.generate_cleanup_report()
            
            self.logger.info("测试结果清理脚本执行完成")
            
        except Exception as e:
            self.logger.error(f"脚本执行失败: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="HarborAI 项目测试结果文件清理脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python cleanup_test_results.py --dry-run
    预览将要删除的文件
    
  python cleanup_test_results.py --backup
    创建备份并清理
    
  python cleanup_test_results.py --force
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
    
    parser.add_argument(
        '--backup-path',
        type=str,
        help='自定义备份路径'
    )
    
    args = parser.parse_args()
    
    try:
        cleaner = TestResultsCleaner(
            project_root=args.project_root,
            dry_run=args.dry_run,
            backup=args.backup,
            force=args.force,
            backup_path=args.backup_path
        )
        
        cleaner.run()
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()