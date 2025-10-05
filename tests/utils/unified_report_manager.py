#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一测试报告管理器

功能：
- 统一管理所有测试报告的输出路径
- 提供标准化的报告命名规范
- 自动创建必要的目录结构
- 支持报告归档和清理

验证方法：pytest tests/test_unified_report_manager.py
作者：HarborAI测试团队
创建时间：2024年12月3日
"""

import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class UnifiedReportManager:
    """
    统一报告管理器
    
    负责管理所有测试报告的输出路径和格式
    确保报告输出的一致性和可维护性
    """
    
    def __init__(self, base_dir: str = None):
        """
        初始化报告管理器
        
        参数：
            base_dir: 报告根目录，默认为 tests/reports
        """
        if base_dir is None:
            # 自动检测项目根目录
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            base_dir = project_root / "tests" / "reports"
        
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 目录结构定义
        self.directory_structure = {
            "unit": ["html", "xml", "json"],
            "integration": ["html", "xml", "json"],
            "functional": ["html", "xml", "json"],
            "performance": ["benchmarks", "load_tests", "metrics", "html", "json", "markdown"],
            "security": ["html", "json", "xml"],
            "coverage": ["html", "xml", "json"],
            "allure": ["results", "report"],
            "dashboard": ["html", "assets"],
            "archive": []
        }
        
        self._ensure_directories()
        logger.info(f"报告管理器初始化完成，根目录：{self.base_dir}")
    
    def _ensure_directories(self):
        """确保所有必要的目录存在"""
        try:
            for test_type, formats in self.directory_structure.items():
                if formats:
                    for format_type in formats:
                        dir_path = self.base_dir / test_type / format_type
                        dir_path.mkdir(parents=True, exist_ok=True)
                else:
                    # 对于archive等特殊目录
                    dir_path = self.base_dir / test_type
                    dir_path.mkdir(parents=True, exist_ok=True)
            
            logger.info("报告目录结构创建完成")
        except Exception as e:
            logger.error(f"创建目录结构失败：{e}")
            raise
    
    def get_report_path(self, test_type: str, format_type: str, filename: str = None) -> Path:
        """
        获取标准报告输出路径
        
        参数：
            test_type: 测试类型 (unit, integration, functional, security)
            format_type: 报告格式 (html, xml, json)
            filename: 文件名（可选，默认自动生成）
        
        返回：完整的报告文件路径
        """
        if test_type not in self.directory_structure:
            raise ValueError(f"不支持的测试类型：{test_type}")
        
        if format_type not in self.directory_structure[test_type]:
            raise ValueError(f"测试类型 {test_type} 不支持格式：{format_type}")
        
        if filename is None:
            ext = self._get_file_extension(format_type)
            filename = f"{test_type}_{format_type}_{self.timestamp}.{ext}"
        
        return self.base_dir / test_type / format_type / filename
    
    def get_performance_path(self, subtype: str, format_type: str = None, filename: str = None) -> Path:
        """
        获取性能测试报告路径
        
        参数：
            subtype: 性能测试子类型 (benchmarks, load_tests, metrics, etc.)
            format_type: 报告格式（可选）
            filename: 文件名（可选）
        
        返回：完整的性能报告文件路径
        """
        if subtype not in self.directory_structure["performance"]:
            raise ValueError(f"不支持的性能测试子类型：{subtype}")
        
        base_path = self.base_dir / "performance" / subtype
        
        if filename is None and format_type:
            ext = self._get_file_extension(format_type)
            filename = f"performance_{subtype}_{self.timestamp}.{ext}"
        elif filename is None:
            filename = f"performance_{subtype}_{self.timestamp}"
        
        return base_path / filename
    
    def get_coverage_path(self, format_type: str, filename: str = None) -> Path:
        """
        获取覆盖率报告路径
        
        参数：
            format_type: 报告格式 (html, xml, json)
            filename: 文件名（可选）
        
        返回：完整的覆盖率报告路径
        """
        return self.get_report_path("coverage", format_type, filename)
    
    def get_allure_path(self, path_type: str = "results") -> Path:
        """
        获取Allure报告路径
        
        参数：
            path_type: 路径类型 (results, report)
        
        返回：Allure报告目录路径
        """
        if path_type not in ["results", "report"]:
            raise ValueError(f"不支持的Allure路径类型：{path_type}")
        
        return self.base_dir / "allure" / path_type
    
    def _get_file_extension(self, format_type: str) -> str:
        """根据格式类型获取文件扩展名"""
        extension_map = {
            "html": "html",
            "xml": "xml", 
            "json": "json",
            "markdown": "md",
            "csv": "csv",
            "txt": "txt"
        }
        return extension_map.get(format_type, format_type)
    
    def archive_old_reports(self, days: int = 30) -> int:
        """
        归档旧报告
        
        参数：
            days: 保留天数，超过此天数的报告将被归档
        
        返回：归档的文件数量
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        archived_count = 0
        
        try:
            for test_type in self.directory_structure.keys():
                if test_type == "archive":
                    continue
                
                test_dir = self.base_dir / test_type
                if not test_dir.exists():
                    continue
                
                for file_path in test_dir.rglob("*"):
                    if file_path.is_file():
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_time < cutoff_date:
                            # 创建归档目录
                            archive_date = file_time.strftime("%Y-%m-%d")
                            archive_dir = self.base_dir / "archive" / archive_date / test_type
                            archive_dir.mkdir(parents=True, exist_ok=True)
                            
                            # 移动文件到归档目录
                            archive_path = archive_dir / file_path.name
                            shutil.move(str(file_path), str(archive_path))
                            archived_count += 1
            
            logger.info(f"归档完成，共归档 {archived_count} 个文件")
            return archived_count
            
        except Exception as e:
            logger.error(f"归档过程中出错：{e}")
            raise
    
    def cleanup_empty_directories(self):
        """清理空目录"""
        try:
            for root, dirs, files in os.walk(self.base_dir, topdown=False):
                for dir_name in dirs:
                    dir_path = Path(root) / dir_name
                    if dir_path.exists() and not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        logger.debug(f"删除空目录：{dir_path}")
        except Exception as e:
            logger.error(f"清理空目录时出错：{e}")
    
    def get_report_summary(self) -> Dict[str, int]:
        """
        获取报告统计摘要
        
        返回：各类型报告的文件数量统计
        """
        summary = {}
        
        try:
            for test_type in self.directory_structure.keys():
                test_dir = self.base_dir / test_type
                if test_dir.exists():
                    file_count = sum(1 for _ in test_dir.rglob("*") if _.is_file())
                    summary[test_type] = file_count
                else:
                    summary[test_type] = 0
            
            return summary
            
        except Exception as e:
            logger.error(f"获取报告摘要时出错：{e}")
            return {}


# 全局单例实例
_report_manager_instance = None


def get_report_manager() -> UnifiedReportManager:
    """获取全局报告管理器实例"""
    global _report_manager_instance
    if _report_manager_instance is None:
        _report_manager_instance = UnifiedReportManager()
    return _report_manager_instance


# 便捷函数
def get_unit_report_path(format_type: str, filename: str = None) -> Path:
    """获取单元测试报告路径"""
    return get_report_manager().get_report_path("unit", format_type, filename)


def get_integration_report_path(format_type: str, filename: str = None) -> Path:
    """获取集成测试报告路径"""
    return get_report_manager().get_report_path("integration", format_type, filename)


def get_performance_report_path(subtype: str, format_type: str = None, filename: str = None) -> Path:
    """获取性能测试报告路径"""
    return get_report_manager().get_performance_path(subtype, format_type, filename)


def get_coverage_report_path(format_type: str, filename: str = None) -> Path:
    """获取覆盖率报告路径"""
    return get_report_manager().get_coverage_path(format_type, filename)


def get_allure_report_path(path_type: str = "results") -> Path:
    """获取Allure报告路径"""
    return get_report_manager().get_allure_path(path_type)