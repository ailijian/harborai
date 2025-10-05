#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一报告管理器测试模块

功能：测试统一报告管理器的各项功能
作者：HarborAI测试团队
创建时间：2024年12月3日

测试覆盖：
- 路径标准化功能
- 目录结构管理
- 报告归档功能
- 错误处理机制
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import os
import sys

# 添加路径以导入被测试模块
sys.path.insert(0, str(Path(__file__).parent / "utils"))

from utils.unified_report_manager import (
    UnifiedReportManager,
    get_report_manager,
    get_unit_report_path,
    get_integration_report_path,
    get_performance_report_path,
    get_coverage_report_path,
    get_allure_report_path
)


class TestUnifiedReportManager:
    """统一报告管理器测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        # 创建临时目录作为测试基础目录
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir) / "test_reports"
        self.manager = UnifiedReportManager(str(self.base_dir))
    
    def teardown_method(self):
        """测试后置清理"""
        # 清理临时目录
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_init_creates_directory_structure(self):
        """测试初始化时创建目录结构"""
        # 验证基础目录存在
        assert self.base_dir.exists()
        
        # 验证各类型测试目录存在
        expected_dirs = [
            "unit/html", "unit/xml", "unit/json",
            "integration/html", "integration/xml", "integration/json",
            "functional/html", "functional/xml", "functional/json",
            "performance/benchmarks", "performance/load_tests", 
            "performance/metrics", "performance/html", "performance/json", "performance/markdown",
            "security/html", "security/json", "security/xml",
            "coverage/html", "coverage/xml", "coverage/json",
            "allure/results", "allure/report",
            "dashboard/html", "dashboard/assets",
            "archive"
        ]
        
        for dir_path in expected_dirs:
            full_path = self.base_dir / dir_path
            assert full_path.exists(), f"目录 {dir_path} 应该存在"
            assert full_path.is_dir(), f"{dir_path} 应该是目录"
    
    def test_get_report_path_basic(self):
        """测试基本报告路径获取"""
        # 测试单元测试HTML报告路径
        path = self.manager.get_report_path("unit", "html", "test_report.html")
        expected = self.base_dir / "unit" / "html" / "test_report.html"
        assert path == expected
        
        # 测试集成测试XML报告路径
        path = self.manager.get_report_path("integration", "xml", "integration_report.xml")
        expected = self.base_dir / "integration" / "xml" / "integration_report.xml"
        assert path == expected
    
    def test_get_report_path_auto_filename(self):
        """测试自动生成文件名"""
        path = self.manager.get_report_path("unit", "html")
        # 验证路径结构和文件名格式，而不是具体的时间戳
        assert path.parent == self.base_dir / "unit" / "html"
        assert path.name.startswith("unit_html_")
        assert path.name.endswith(".html")
        # 验证时间戳格式 (YYYYMMDD_HHMMSS)
        timestamp_part = path.stem.split("_", 2)[2]  # 获取时间戳部分
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS 长度为15
    
    def test_get_report_path_invalid_test_type(self):
        """测试无效测试类型"""
        with pytest.raises(ValueError, match="不支持的测试类型"):
            self.manager.get_report_path("invalid_type", "html")
    
    def test_get_report_path_invalid_format_type(self):
        """测试无效格式类型"""
        with pytest.raises(ValueError, match="不支持格式"):
            self.manager.get_report_path("unit", "invalid_format")
    
    def test_get_performance_path(self):
        """测试性能测试报告路径获取"""
        # 测试基准测试路径
        path = self.manager.get_performance_path("benchmarks", "json", "benchmark_results.json")
        expected = self.base_dir / "performance" / "benchmarks" / "benchmark_results.json"
        assert path == expected
        
        # 测试负载测试路径
        path = self.manager.get_performance_path("load_tests", "html", "load_test_report.html")
        expected = self.base_dir / "performance" / "load_tests" / "load_test_report.html"
        assert path == expected
    
    def test_get_performance_path_auto_filename(self):
        """测试性能测试自动文件名生成"""
        path = self.manager.get_performance_path("metrics", "json")
        # 验证路径结构和文件名格式，而不是具体的时间戳
        assert path.parent == self.base_dir / "performance" / "metrics"
        assert path.name.startswith("performance_metrics_")
        assert path.name.endswith(".json")
        # 验证时间戳格式
        timestamp_part = path.stem.split("_", 2)[2]  # 获取时间戳部分
        assert len(timestamp_part) == 15  # YYYYMMDD_HHMMSS 长度为15
    
    def test_get_performance_path_invalid_subtype(self):
        """测试无效性能测试子类型"""
        with pytest.raises(ValueError, match="不支持的性能测试子类型"):
            self.manager.get_performance_path("invalid_subtype", "json")
    
    def test_get_coverage_path(self):
        """测试覆盖率报告路径获取"""
        path = self.manager.get_coverage_path("html", "coverage_report.html")
        expected = self.base_dir / "coverage" / "html" / "coverage_report.html"
        assert path == expected
    
    def test_get_allure_path(self):
        """测试Allure报告路径获取"""
        # 测试结果目录
        path = self.manager.get_allure_path("results")
        expected = self.base_dir / "allure" / "results"
        assert path == expected
        
        # 测试报告目录
        path = self.manager.get_allure_path("report")
        expected = self.base_dir / "allure" / "report"
        assert path == expected
    
    def test_get_allure_path_invalid_type(self):
        """测试无效Allure路径类型"""
        with pytest.raises(ValueError, match="不支持的Allure路径类型"):
            self.manager.get_allure_path("invalid_type")
    
    def test_file_extension_mapping(self):
        """测试文件扩展名映射"""
        test_cases = [
            ("html", "html"),
            ("xml", "xml"),
            ("json", "json"),
            ("markdown", "md"),
            ("csv", "csv"),
            ("txt", "txt")
        ]
        
        for format_type, expected_ext in test_cases:
            ext = self.manager._get_file_extension(format_type)
            assert ext == expected_ext
    
    def test_archive_old_reports(self):
        """测试报告归档功能"""
        # 创建一些测试文件
        test_file1 = self.base_dir / "unit" / "html" / "old_report.html"
        test_file2 = self.base_dir / "performance" / "json" / "old_perf.json"
        
        test_file1.parent.mkdir(parents=True, exist_ok=True)
        test_file2.parent.mkdir(parents=True, exist_ok=True)
        
        test_file1.write_text("test content 1")
        test_file2.write_text("test content 2")
        
        # 修改文件时间为31天前
        old_time = datetime.now() - timedelta(days=31)
        old_timestamp = old_time.timestamp()
        
        os.utime(test_file1, (old_timestamp, old_timestamp))
        os.utime(test_file2, (old_timestamp, old_timestamp))
        
        # 执行归档
        archived_count = self.manager.archive_old_reports(days=30)
        
        # 验证文件被归档
        assert archived_count == 2
        assert not test_file1.exists()
        assert not test_file2.exists()
        
        # 验证归档目录存在
        archive_date = old_time.strftime("%Y-%m-%d")
        archive_dir = self.base_dir / "archive" / archive_date
        assert archive_dir.exists()
        
        # 验证文件在归档目录中
        archived_file1 = archive_dir / "unit" / "old_report.html"
        archived_file2 = archive_dir / "performance" / "old_perf.json"
        assert archived_file1.exists()
        assert archived_file2.exists()
    
    def test_cleanup_empty_directories(self):
        """测试清理空目录功能"""
        # 创建一些空目录
        empty_dir1 = self.base_dir / "unit" / "empty_subdir"
        empty_dir2 = self.base_dir / "performance" / "empty_subdir"
        
        empty_dir1.mkdir(parents=True)
        empty_dir2.mkdir(parents=True)
        
        # 创建一个非空目录
        non_empty_dir = self.base_dir / "integration" / "non_empty"
        non_empty_dir.mkdir(parents=True)
        (non_empty_dir / "test_file.txt").write_text("content")
        
        # 执行清理
        self.manager.cleanup_empty_directories()
        
        # 验证空目录被删除
        assert not empty_dir1.exists()
        assert not empty_dir2.exists()
        
        # 验证非空目录保留
        assert non_empty_dir.exists()
        assert (non_empty_dir / "test_file.txt").exists()
    
    def test_get_report_summary(self):
        """测试报告统计摘要"""
        # 创建一些测试文件
        test_files = [
            self.base_dir / "unit" / "html" / "test1.html",
            self.base_dir / "unit" / "xml" / "test2.xml",
            self.base_dir / "performance" / "json" / "perf1.json",
            self.base_dir / "performance" / "json" / "perf2.json",
            self.base_dir / "coverage" / "html" / "coverage.html"
        ]
        
        for file_path in test_files:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("test content")
        
        # 获取统计摘要
        summary = self.manager.get_report_summary()
        
        # 验证统计结果
        assert summary["unit"] == 2  # test1.html, test2.xml
        assert summary["performance"] == 2  # perf1.json, perf2.json
        assert summary["coverage"] == 1  # coverage.html
        assert summary["integration"] == 0  # 没有文件
        assert summary["security"] == 0  # 没有文件


class TestGlobalFunctions:
    """测试全局便捷函数"""
    
    def setup_method(self):
        """测试前置设置"""
        self.temp_dir = tempfile.mkdtemp()
        self.base_dir = Path(self.temp_dir) / "test_reports"
    
    def teardown_method(self):
        """测试后置清理"""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('utils.unified_report_manager._report_manager_instance', None)
    def test_get_report_manager_singleton(self):
        """测试全局报告管理器单例"""
        # 第一次调用
        manager1 = get_report_manager()
        
        # 第二次调用应该返回同一个实例
        manager2 = get_report_manager()
        
        assert manager1 is manager2
    
    @patch('utils.unified_report_manager.get_report_manager')
    def test_convenience_functions(self, mock_get_manager):
        """测试便捷函数"""
        mock_manager = MagicMock()
        mock_get_manager.return_value = mock_manager
        
        # 测试单元测试报告路径函数
        get_unit_report_path("html", "test.html")
        mock_manager.get_report_path.assert_called_with("unit", "html", "test.html")
        
        # 测试集成测试报告路径函数
        get_integration_report_path("xml", "integration.xml")
        mock_manager.get_report_path.assert_called_with("integration", "xml", "integration.xml")
        
        # 测试性能测试报告路径函数
        get_performance_report_path("benchmarks", "json", "bench.json")
        mock_manager.get_performance_path.assert_called_with("benchmarks", "json", "bench.json")
        
        # 测试覆盖率报告路径函数
        get_coverage_report_path("html", "coverage.html")
        mock_manager.get_coverage_path.assert_called_with("html", "coverage.html")
        
        # 测试Allure报告路径函数
        get_allure_report_path("results")
        mock_manager.get_allure_path.assert_called_with("results")


class TestErrorHandling:
    """测试错误处理"""
    
    def test_init_with_permission_error(self):
        """测试初始化时权限错误处理"""
        # 尝试在只读目录创建报告管理器
        with patch('pathlib.Path.mkdir') as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Permission denied")
            
            with pytest.raises(PermissionError):
                UnifiedReportManager("/readonly/path")
    
    def test_archive_with_file_error(self):
        """测试归档时文件操作错误"""
        temp_dir = tempfile.mkdtemp()
        try:
            manager = UnifiedReportManager(temp_dir)
            
            with patch('shutil.move') as mock_move:
                mock_move.side_effect = OSError("File operation failed")
                
                # 创建测试文件
                test_file = Path(temp_dir) / "unit" / "html" / "test.html"
                test_file.parent.mkdir(parents=True, exist_ok=True)
                test_file.write_text("test")
                
                # 修改文件时间
                old_time = datetime.now() - timedelta(days=31)
                os.utime(test_file, (old_time.timestamp(), old_time.timestamp()))
                
                # 归档应该抛出异常
                with pytest.raises(OSError):
                    manager.archive_old_reports(days=30)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])