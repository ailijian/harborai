#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PowerShell脚本路径配置验证测试

验证所有PowerShell脚本都使用了统一的报告路径配置
"""

import unittest
import os
import re
from pathlib import Path


class TestPowerShellScripts(unittest.TestCase):
    """PowerShell脚本配置测试类"""
    
    def setUp(self):
        """测试初始化"""
        self.project_root = Path(__file__).parent.parent
        self.scripts_dir = self.project_root / "tests" / "scripts"
        
    def test_scripts_directory_exists(self):
        """测试脚本目录存在"""
        self.assertTrue(self.scripts_dir.exists(), f"脚本目录不存在: {self.scripts_dir}")
        
    def test_powershell_scripts_exist(self):
        """测试PowerShell脚本文件存在"""
        expected_scripts = [
            "generate_reports.ps1",
            "run_all_tests.ps1", 
            "run_performance_tests.ps1",
            "setup_test_env.ps1",
            "cleanup_test_env.ps1"
        ]
        
        for script_name in expected_scripts:
            script_path = self.scripts_dir / script_name
            self.assertTrue(script_path.exists(), f"脚本文件不存在: {script_path}")
            
    def test_generate_reports_script_paths(self):
        """测试generate_reports.ps1使用正确的路径配置"""
        script_path = self.scripts_dir / "generate_reports.ps1"
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 验证使用了统一的报告目录结构
        self.assertIn('$DefaultOutputDir = Join-Path $TestsDir "reports"', content)
        self.assertIn('$ActualOutputDir = if ($OutputDir) { $OutputDir } else { $DefaultOutputDir }', content)
        
        # 验证没有使用旧的硬编码路径
        old_patterns = [
            r'reports/coverage',
            r'reports/html',
            r'reports/allure'
        ]
        
        for pattern in old_patterns:
            matches = re.findall(pattern, content)
            # 允许在注释或字符串中出现，但不应该在路径配置中出现
            if matches:
                # 检查是否在注释中
                lines = content.split('\n')
                for line_num, line in enumerate(lines, 1):
                    if pattern in line and not line.strip().startswith('#'):
                        # 进一步检查是否在字符串字面量中（HTML内容等）
                        if not ('"' in line and pattern in line):
                            self.fail(f"在第{line_num}行发现旧路径配置: {line.strip()}")
                            
    def test_run_all_tests_script_paths(self):
        """测试run_all_tests.ps1使用正确的路径配置"""
        script_path = self.scripts_dir / "run_all_tests.ps1"
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 验证使用了统一的报告目录结构
        self.assertIn('$ReportsDir = Join-Path $TestsDir "reports"', content)
        
        # 验证覆盖率报告路径使用了变量
        coverage_patterns = [
            r'--cov-report=html:\$ReportsDir',
            r'--cov-report=html:\$ReportsDir/coverage'
        ]
        
        found_coverage_config = False
        for pattern in coverage_patterns:
            if re.search(pattern, content):
                found_coverage_config = True
                break
                
        if not found_coverage_config:
            # 检查是否有其他形式的覆盖率配置
            if '--cov-report=html:' in content:
                self.assertIn('$ReportsDir', content, "覆盖率报告路径应该使用$ReportsDir变量")
                
    def test_run_performance_tests_script_paths(self):
        """测试run_performance_tests.ps1使用正确的路径配置"""
        script_path = self.scripts_dir / "run_performance_tests.ps1"
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 验证使用了统一的性能测试报告目录结构
        self.assertIn('$ReportsDir = Join-Path $TestsDir "reports" "performance"', content)
        
        # 验证性能测试目录配置
        self.assertIn('$PerformanceDir = Join-Path $TestsDir "performance"', content)
        
    def test_setup_test_env_script_paths(self):
        """测试setup_test_env.ps1使用正确的路径配置"""
        script_path = self.scripts_dir / "setup_test_env.ps1"
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 验证使用了统一的报告目录结构
        self.assertIn('$ReportsDir = Join-Path $TestsDir "reports"', content)
        
    def test_cleanup_test_env_script_paths(self):
        """测试cleanup_test_env.ps1的路径配置"""
        script_path = self.scripts_dir / "cleanup_test_env.ps1"
        
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 这个脚本主要是清理，应该引用正确的目录结构
        # 验证日志目录配置
        self.assertIn('Join-Path $TestsDir "logs"', content)
        
    def test_unified_directory_structure(self):
        """测试所有脚本都遵循统一的目录结构"""
        script_files = list(self.scripts_dir.glob("*.ps1"))
        
        for script_path in script_files:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 验证基本目录变量定义
            self.assertIn('$TestsDir = Join-Path $ProjectRoot "tests"', content,
                         f"脚本 {script_path.name} 应该定义$TestsDir变量")
                         
            # 验证项目根目录变量定义
            self.assertIn('$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)', content,
                         f"脚本 {script_path.name} 应该定义$ProjectRoot变量")
                         
    def test_no_hardcoded_report_paths(self):
        """测试没有硬编码的报告路径"""
        script_files = list(self.scripts_dir.glob("*.ps1"))
        
        # 不应该出现的硬编码路径模式
        forbidden_patterns = [
            r'reports/coverage/html',
            r'reports/html/report\.html',
            r'reports/allure(?!/)',  # 允许 reports/allure/ 但不允许 reports/allure
        ]
        
        for script_path in script_files:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern in forbidden_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    # 检查是否在注释或HTML字符串中
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line):
                            # 跳过注释行
                            if line.strip().startswith('#'):
                                continue
                            # 跳过HTML内容字符串
                            if '"' in line and any(html_tag in line for html_tag in ['<', '>', 'html', 'href']):
                                continue
                            # 如果不是注释或HTML，则报告错误
                            self.fail(f"在 {script_path.name} 第{line_num}行发现硬编码路径: {line.strip()}")


if __name__ == '__main__':
    unittest.main()