#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest配置验证测试

功能：
- 验证pytest.ini配置文件的报告路径设置
- 确保所有报告输出路径符合统一标准

验证方法：pytest tests/test_pytest_config.py -v
作者：HarborAI测试团队
创建时间：2024年12月3日
"""

import configparser
import unittest
from pathlib import Path


class TestPytestConfig(unittest.TestCase):
    """pytest配置验证测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.config_file = Path(__file__).parent / "pytest.ini"
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
    
    def test_pytest_ini_exists(self):
        """验证pytest.ini文件存在"""
        self.assertTrue(self.config_file.exists(), "pytest.ini文件不存在")
    
    def test_coverage_report_paths(self):
        """验证覆盖率报告路径配置"""
        addopts = self.config.get('tool:pytest', 'addopts')
        
        # 验证HTML覆盖率报告路径
        self.assertIn('--cov-report=html:tests/reports/coverage/html', addopts)
        
        # 验证XML覆盖率报告路径
        self.assertIn('--cov-report=xml:tests/reports/coverage/xml/coverage.xml', addopts)
        
        # 验证JSON覆盖率报告路径
        self.assertIn('--cov-report=json:tests/reports/coverage/json/coverage.json', addopts)
    
    def test_html_report_path(self):
        """验证HTML测试报告路径配置"""
        addopts = self.config.get('tool:pytest', 'addopts')
        self.assertIn('--html=tests/reports/unit/html/pytest_report.html', addopts)
    
    def test_allure_report_path(self):
        """验证Allure报告路径配置"""
        addopts = self.config.get('tool:pytest', 'addopts')
        self.assertIn('--alluredir=tests/reports/allure/results', addopts)
    
    def test_unified_report_structure(self):
        """验证统一报告结构"""
        addopts = self.config.get('tool:pytest', 'addopts')
        
        # 所有报告路径都应该以tests/reports/开头
        report_paths = [
            'tests/reports/coverage/html',
            'tests/reports/coverage/xml/coverage.xml',
            'tests/reports/coverage/json/coverage.json',
            'tests/reports/unit/html/pytest_report.html',
            'tests/reports/allure/results'
        ]
        
        for path in report_paths:
            self.assertIn(path, addopts, f"报告路径 {path} 未在配置中找到")
    
    def test_no_old_report_paths(self):
        """验证不包含旧的报告路径"""
        addopts = self.config.get('tool:pytest', 'addopts')
        
        # 确保不包含旧的报告路径（精确匹配）
        old_path_patterns = [
            '--cov-report=html:reports/coverage/html',
            '--cov-report=xml:reports/coverage/coverage.xml',
            '--html=reports/html/report.html',
            '--alluredir=reports/allure'
        ]
        
        for old_pattern in old_path_patterns:
            self.assertNotIn(old_pattern, addopts, f"发现旧的报告路径配置 {old_pattern}，应该已被移除")


if __name__ == '__main__':
    unittest.main()