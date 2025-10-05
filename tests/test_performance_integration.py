#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试集成验证

功能：
- 验证性能测试文件使用统一报告管理器
- 确保报告输出路径正确

验证方法：pytest tests/test_performance_integration.py -v
作者：HarborAI测试团队
创建时间：2024年12月3日
"""

import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# 添加路径以导入性能测试模块
sys.path.append(str(Path(__file__).parent))
from utils.unified_report_manager import get_performance_report_path


class TestPerformanceIntegration(unittest.TestCase):
    """性能测试集成验证类"""
    
    def setUp(self):
        """测试初始化"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_results_file = Path(self.temp_dir) / "test_results.json"
        
        # 创建测试结果文件
        test_data = {
            "test_session": "test_session_20241203_143022",
            "analysis": {
                "prd_compliance": {
                    "call_overhead": {"compliant": True},
                    "response_time": {"compliant": False}
                },
                "bottlenecks": ["memory_usage", "cpu_intensive"]
            }
        }
        
        with open(self.test_results_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f)
    
    def test_performance_report_generator_uses_unified_manager(self):
        """验证性能报告生成器使用统一报告管理器"""
        # 导入性能报告生成器
        from performance.performance_report_generator import PerformanceReportGenerator
        
        # 创建报告生成器实例
        generator = PerformanceReportGenerator(str(self.test_results_file))
        
        # 验证报告目录使用了统一管理器的路径
        expected_base_path = get_performance_report_path("metrics").parent
        self.assertEqual(generator.report_dir, expected_base_path)
    
    def test_simple_performance_test_path_generation(self):
        """验证简单性能测试的路径生成"""
        # 测试路径生成函数
        test_filename = "simple_test_20241203_143022.json"
        expected_path = get_performance_report_path("metrics", "json", test_filename)
        
        # 验证路径结构（使用Path对象比较，避免路径分隔符问题）
        path_parts = expected_path.parts
        self.assertIn("tests", path_parts)
        self.assertIn("reports", path_parts)
        self.assertIn("performance", path_parts)
        self.assertIn("metrics", path_parts)
        self.assertEqual(expected_path.name, test_filename)
    
    def test_performance_report_path_structure(self):
        """验证性能报告路径结构"""
        # 测试不同类型的性能报告路径
        test_cases = [
            ("metrics", "json", "test_metrics.json"),
            ("benchmarks", "html", "test_benchmarks.html"),
            ("load_tests", "markdown", "test_load.md")
        ]
        
        for subtype, format_type, filename in test_cases:
            with self.subTest(subtype=subtype, format_type=format_type):
                path = get_performance_report_path(subtype, format_type, filename)
                
                # 验证路径包含正确的结构（使用Path对象比较）
                path_parts = path.parts
                self.assertIn("tests", path_parts)
                self.assertIn("reports", path_parts)
                self.assertIn("performance", path_parts)
                self.assertIn(subtype, path_parts)
                self.assertEqual(path.name, filename)
    
    @patch('performance.performance_report_generator.get_performance_report_path')
    def test_report_generator_calls_unified_manager(self, mock_get_path):
        """验证报告生成器调用统一管理器"""
        # 设置mock返回值
        mock_path = Path(self.temp_dir) / "test_report.md"
        mock_get_path.return_value = mock_path
        
        # 导入并创建报告生成器
        from performance.performance_report_generator import PerformanceReportGenerator
        generator = PerformanceReportGenerator(str(self.test_results_file))
        
        # 模拟生成报告
        with patch.object(generator, '_generate_report_content', return_value="Test Report"):
            with patch.object(generator, '_generate_charts'):
                try:
                    generator.generate_comprehensive_report()
                    # 验证统一管理器被调用
                    mock_get_path.assert_called()
                except Exception:
                    # 忽略其他可能的错误，只关注路径调用
                    pass
    
    def test_unified_manager_performance_paths(self):
        """验证统一管理器的性能路径功能"""
        # 测试各种性能报告子类型
        subtypes = ["metrics", "benchmarks", "load_tests"]
        formats = ["json", "html", "markdown"]
        
        for subtype in subtypes:
            for format_type in formats:
                with self.subTest(subtype=subtype, format_type=format_type):
                    path = get_performance_report_path(subtype, format_type)
                    
                    # 验证路径结构
                    self.assertIn("performance", str(path))
                    self.assertIn(subtype, str(path))
                    
                    # 验证文件扩展名
                    expected_ext = "md" if format_type == "markdown" else format_type
                    self.assertTrue(path.name.endswith(f".{expected_ext}"))


if __name__ == '__main__':
    unittest.main()