"""
HarborAI测试报告验证脚本测试

此模块测试验证脚本的功能，确保验证逻辑正确。
"""

import os
import subprocess
import tempfile
import pytest
from pathlib import Path


class TestVerificationScript:
    """验证脚本测试类"""
    
    @pytest.fixture
    def project_root(self):
        """获取项目根目录"""
        return Path(__file__).parent.parent
    
    @pytest.fixture
    def verification_script(self, project_root):
        """获取验证脚本路径"""
        return project_root / "tests" / "scripts" / "verify_reports.ps1"
    
    def test_verification_script_exists(self, verification_script):
        """测试验证脚本文件存在"""
        assert verification_script.exists(), f"验证脚本不存在: {verification_script}"
    
    def test_verification_script_has_help(self, verification_script):
        """测试验证脚本包含帮助信息"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查帮助相关内容
        assert ".SYNOPSIS" in content, "脚本应包含SYNOPSIS"
        assert ".DESCRIPTION" in content, "脚本应包含DESCRIPTION"
        assert ".PARAMETER" in content, "脚本应包含PARAMETER说明"
        assert ".EXAMPLE" in content, "脚本应包含EXAMPLE"
        assert "param(" in content, "脚本应包含参数定义"
        assert "[switch]$Help" in content, "脚本应包含Help参数"
    
    def test_verification_script_has_required_functions(self, verification_script):
        """测试验证脚本包含必需的函数"""
        content = verification_script.read_text(encoding='utf-8')
        
        required_functions = [
            "function Write-Log",
            "function Add-Issue", 
            "function Add-Success",
            "function Test-DirectoryStructure",
            "function Test-PytestConfig",
            "function Test-PowerShellScripts",
            "function Test-UnifiedManager",
            "function Test-PerformanceIntegration",
            "function Test-TestExecution",
            "function New-VerificationReport",
            "function Start-ReportVerification"
        ]
        
        for func in required_functions:
            assert func in content, f"脚本应包含函数: {func}"
    
    def test_verification_script_has_proper_structure(self, verification_script):
        """测试验证脚本具有正确的结构"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查脚本结构
        assert "$ErrorActionPreference = \"Stop\"" in content, "应设置错误处理"
        assert "$VerificationResults = @{" in content, "应定义验证结果存储"
        assert "Start-ReportVerification" in content, "应调用主执行函数"
        
        # 检查路径变量
        assert "$ProjectRoot" in content, "应定义项目根目录变量"
        assert "$TestsDir" in content, "应定义测试目录变量"
        assert "$ActualReportsDir" in content, "应定义报告目录变量"
    
    def test_verification_script_parameters(self, verification_script):
        """测试验证脚本参数定义"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查参数定义
        expected_params = [
            "[string]$ReportsDir",
            "[switch]$Verbose",
            "[switch]$GenerateReport", 
            "[switch]$FixIssues",
            "[switch]$Help"
        ]
        
        for param in expected_params:
            assert param in content, f"脚本应包含参数: {param}"
    
    def test_verification_script_validation_categories(self, verification_script):
        """测试验证脚本包含所有验证类别"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查验证类别
        validation_categories = [
            "DirectoryStructure",
            "PytestConfig", 
            "PowerShellScripts",
            "UnifiedManager",
            "PerformanceIntegration"
        ]
        
        for category in validation_categories:
            assert category in content, f"脚本应包含验证类别: {category}"
    
    def test_verification_script_required_directories(self, verification_script):
        """测试验证脚本检查必需的目录"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查必需目录列表
        required_dirs = [
            '"coverage"',
            '"html"',
            '"allure"', 
            '"performance"',
            '"performance/metrics"',
            '"performance/reports"',
            '"security"',
            '"functional"',
            '"dashboard"'
        ]
        
        for dir_name in required_dirs:
            assert dir_name in content, f"脚本应检查目录: {dir_name}"
    
    def test_verification_script_pytest_configs(self, verification_script):
        """测试验证脚本检查pytest配置"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查pytest配置项 - 现在使用动态路径构建
        pytest_config_patterns = [
            "--cov-report=html:",
            "--cov-report=xml:",
            "--cov-report=json:",
            "--html=",
            "--alluredir="
        ]
        
        # 检查动态路径变量
        assert "$ReportsRelativePath" in content, "脚本应使用动态路径变量"
        
        for pattern in pytest_config_patterns:
            assert pattern in content, f"脚本应检查pytest配置模式: {pattern}"
    
    def test_verification_script_powershell_scripts(self, verification_script):
        """测试验证脚本检查PowerShell脚本"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查PowerShell脚本列表
        ps_scripts = [
            '"generate_reports.ps1"',
            '"run_all_tests.ps1"',
            '"run_performance_tests.ps1"',
            '"setup_test_env.ps1"',
            '"cleanup_test_env.ps1"'
        ]
        
        for script in ps_scripts:
            assert script in content, f"脚本应检查PowerShell脚本: {script}"
    
    def test_verification_script_html_report_generation(self, verification_script):
        """测试验证脚本包含HTML报告生成"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查HTML报告相关内容
        html_elements = [
            "<!DOCTYPE html>",
            "<title>HarborAI测试报告验证报告</title>",
            "verification_report_",
            ".html",
            "总体状态:",
            "验证概览",
            "目录结构验证",
            "发现的问题",
            "系统信息",
            "下一步操作"
        ]
        
        for element in html_elements:
            assert element in content, f"脚本应包含HTML报告元素: {element}"
    
    def test_verification_script_logging(self, verification_script):
        """测试验证脚本包含日志功能"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查日志相关功能
        log_elements = [
            "Write-Log",
            "$LogFile",
            "Add-Content -Path $LogFile",
            "verify_reports_",
            ".log"
        ]
        
        for element in log_elements:
            assert element in content, f"脚本应包含日志元素: {element}"
    
    def test_verification_script_issue_tracking(self, verification_script):
        """测试验证脚本包含问题跟踪"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查问题跟踪功能
        issue_elements = [
            "Add-Issue",
            "Add-Success",
            "$VerificationResults.Issues",
            "Severity",
            "Recommendation",
            "Category",
            "Description"
        ]
        
        for element in issue_elements:
            assert element in content, f"脚本应包含问题跟踪元素: {element}"
    
    def test_verification_script_fix_issues_mode(self, verification_script):
        """测试验证脚本包含自动修复模式"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查自动修复功能
        fix_elements = [
            "if ($FixIssues)",
            "New-Item -Path $DirPath -ItemType Directory -Force",
            "自动修复模式已启用"
        ]
        
        for element in fix_elements:
            assert element in content, f"脚本应包含自动修复元素: {element}"
    
    def test_verification_script_exit_codes(self, verification_script):
        """测试验证脚本包含正确的退出代码"""
        content = verification_script.read_text(encoding='utf-8')
        
        # 检查退出代码
        assert "exit 0" in content, "脚本应包含成功退出代码"
        assert "exit 1" in content, "脚本应包含失败退出代码"
        assert "if ($OverallStatus -eq \"FAIL\")" in content, "脚本应根据状态设置退出代码"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])