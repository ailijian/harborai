#!/usr/bin/env python3
"""
HarborAI 测试运行脚本

此脚本负责运行追踪系统的所有测试，包括：
- 单元测试
- 集成测试
- 数据验证测试
- 性能基准测试

功能特性：
- 自动发现测试
- 并行测试执行
- 详细的测试报告
- 覆盖率统计
- 失败测试重试

作者: HarborAI团队
创建时间: 2025-01-15
版本: v1.0.0
"""

import asyncio
import subprocess
import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TestResult:
    """测试结果"""
    name: str
    status: str  # "pass", "fail", "skip", "error"
    duration: float
    output: str = ""
    error: str = ""
    coverage: float = 0.0


@dataclass
class TestSuite:
    """测试套件"""
    name: str
    tests: List[TestResult]
    total_duration: float
    pass_count: int = 0
    fail_count: int = 0
    skip_count: int = 0
    error_count: int = 0
    coverage: float = 0.0
    
    def __post_init__(self):
        self.pass_count = len([t for t in self.tests if t.status == "pass"])
        self.fail_count = len([t for t in self.tests if t.status == "fail"])
        self.skip_count = len([t for t in self.tests if t.status == "skip"])
        self.error_count = len([t for t in self.tests if t.status == "error"])


@dataclass
class TestReport:
    """测试报告"""
    timestamp: datetime
    project_info: Dict[str, str]
    test_suites: List[TestSuite]
    overall_coverage: float = 0.0
    total_duration: float = 0.0
    total_tests: int = 0
    total_pass: int = 0
    total_fail: int = 0
    total_skip: int = 0
    total_error: int = 0
    
    def __post_init__(self):
        self.total_duration = sum(suite.total_duration for suite in self.test_suites)
        self.total_tests = sum(len(suite.tests) for suite in self.test_suites)
        self.total_pass = sum(suite.pass_count for suite in self.test_suites)
        self.total_fail = sum(suite.fail_count for suite in self.test_suites)
        self.total_skip = sum(suite.skip_count for suite in self.test_suites)
        self.total_error = sum(suite.error_count for suite in self.test_suites)


class TestRunner:
    """测试运行器"""
    
    def __init__(self, project_root: Path):
        """
        初始化测试运行器
        
        Args:
            project_root: 项目根目录
        """
        self.project_root = project_root
        self.test_suites: List[TestSuite] = []
        
    def discover_tests(self) -> Dict[str, List[Path]]:
        """发现测试文件"""
        test_categories = {
            "unit": [],
            "integration": [],
            "performance": []
        }
        
        # 单元测试
        unit_test_dir = self.project_root / "tests" / "unit"
        if unit_test_dir.exists():
            test_categories["unit"] = list(unit_test_dir.glob("test_*.py"))
        
        # 集成测试
        integration_test_dir = self.project_root / "tests" / "integration"
        if integration_test_dir.exists():
            test_categories["integration"] = list(integration_test_dir.glob("test_*.py"))
        
        # 性能测试
        performance_test_dir = self.project_root / "tests" / "performance"
        if performance_test_dir.exists():
            test_categories["performance"] = list(performance_test_dir.glob("test_*.py"))
        
        return test_categories
    
    async def run_pytest_suite(self, name: str, test_files: List[Path], 
                              extra_args: List[str] = None) -> TestSuite:
        """运行pytest测试套件"""
        if not test_files:
            return TestSuite(name=name, tests=[], total_duration=0.0)
        
        print(f"🧪 运行 {name} 测试...")
        
        # 构建pytest命令
        cmd = [
            sys.executable, "-m", "pytest",
            "--tb=short",
            "--json-report",
            f"--json-report-file={self.project_root}/test_results_{name}.json",
            "-v"
        ]
        
        if extra_args:
            cmd.extend(extra_args)
        
        # 添加测试文件
        cmd.extend(str(f) for f in test_files)
        
        start_time = time.time()
        
        try:
            # 运行pytest
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 解析JSON报告
            json_report_file = self.project_root / f"test_results_{name}.json"
            tests = []
            
            if json_report_file.exists():
                try:
                    with open(json_report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    for test_data in report_data.get("tests", []):
                        test_result = TestResult(
                            name=test_data.get("nodeid", "unknown"),
                            status=self._map_pytest_outcome(test_data.get("outcome", "error")),
                            duration=test_data.get("duration", 0.0),
                            output=test_data.get("call", {}).get("stdout", ""),
                            error=test_data.get("call", {}).get("stderr", "")
                        )
                        tests.append(test_result)
                    
                    # 清理临时文件
                    json_report_file.unlink()
                    
                except Exception as e:
                    print(f"⚠️  解析测试报告失败: {e}")
            
            # 如果没有解析到测试结果，创建一个基于返回码的结果
            if not tests:
                status = "pass" if result.returncode == 0 else "fail"
                tests.append(TestResult(
                    name=f"{name}_suite",
                    status=status,
                    duration=duration,
                    output=result.stdout,
                    error=result.stderr
                ))
            
            return TestSuite(
                name=name,
                tests=tests,
                total_duration=duration
            )
        
        except subprocess.TimeoutExpired:
            return TestSuite(
                name=name,
                tests=[TestResult(
                    name=f"{name}_timeout",
                    status="error",
                    duration=300.0,
                    error="测试超时"
                )],
                total_duration=300.0
            )
        
        except Exception as e:
            return TestSuite(
                name=name,
                tests=[TestResult(
                    name=f"{name}_error",
                    status="error",
                    duration=0.0,
                    error=str(e)
                )],
                total_duration=0.0
            )
    
    def _map_pytest_outcome(self, outcome: str) -> str:
        """映射pytest结果到标准状态"""
        mapping = {
            "passed": "pass",
            "failed": "fail",
            "skipped": "skip",
            "error": "error",
            "xfail": "skip",
            "xpass": "pass"
        }
        return mapping.get(outcome, "error")
    
    async def run_coverage_analysis(self) -> float:
        """运行覆盖率分析"""
        print("📊 分析代码覆盖率...")
        
        try:
            # 运行覆盖率测试
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=harborai",
                "--cov-report=json",
                f"--cov-report-file={self.project_root}/coverage.json",
                "tests/"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # 解析覆盖率报告
            coverage_file = self.project_root / "coverage.json"
            if coverage_file.exists():
                try:
                    with open(coverage_file, 'r', encoding='utf-8') as f:
                        coverage_data = json.load(f)
                    
                    total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
                    
                    # 清理临时文件
                    coverage_file.unlink()
                    
                    return total_coverage
                
                except Exception as e:
                    print(f"⚠️  解析覆盖率报告失败: {e}")
                    return 0.0
            
            return 0.0
        
        except Exception as e:
            print(f"⚠️  覆盖率分析失败: {e}")
            return 0.0
    
    async def run_all_tests(self, include_coverage: bool = True, 
                           parallel: bool = True) -> TestReport:
        """运行所有测试"""
        print("🚀 开始运行测试套件...")
        
        # 发现测试
        test_categories = self.discover_tests()
        
        print(f"📋 发现测试:")
        for category, files in test_categories.items():
            print(f"   {category}: {len(files)} 个文件")
        
        # 运行测试套件
        test_suites = []
        
        if parallel and len(test_categories) > 1:
            # 并行运行
            tasks = []
            for category, files in test_categories.items():
                if files:
                    task = self.run_pytest_suite(category, files)
                    tasks.append(task)
            
            if tasks:
                test_suites = await asyncio.gather(*tasks)
        else:
            # 串行运行
            for category, files in test_categories.items():
                if files:
                    suite = await self.run_pytest_suite(category, files)
                    test_suites.append(suite)
        
        # 运行覆盖率分析
        overall_coverage = 0.0
        if include_coverage:
            overall_coverage = await self.run_coverage_analysis()
        
        # 获取项目信息
        project_info = {
            "name": "HarborAI",
            "version": "1.0.0",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform
        }
        
        # 创建测试报告
        report = TestReport(
            timestamp=datetime.now(timezone.utc),
            project_info=project_info,
            test_suites=test_suites,
            overall_coverage=overall_coverage
        )
        
        return report
    
    def print_test_report(self, report: TestReport):
        """打印测试报告"""
        print("\n" + "="*80)
        print("🧪 HarborAI 测试报告")
        print("="*80)
        
        print(f"\n📅 测试时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"🐍 Python版本: {report.project_info['python_version']}")
        print(f"💻 平台: {report.project_info['platform']}")
        
        # 总体统计
        print(f"\n📊 总体统计:")
        print(f"   总测试数: {report.total_tests}")
        print(f"   ✅ 通过: {report.total_pass}")
        print(f"   ❌ 失败: {report.total_fail}")
        print(f"   ⏭️  跳过: {report.total_skip}")
        print(f"   💥 错误: {report.total_error}")
        print(f"   ⏱️  总耗时: {report.total_duration:.2f}s")
        print(f"   📈 覆盖率: {report.overall_coverage:.1f}%")
        
        # 成功率
        if report.total_tests > 0:
            success_rate = (report.total_pass / report.total_tests) * 100
            print(f"   🎯 成功率: {success_rate:.1f}%")
            
            if success_rate == 100:
                status_emoji = "🟢"
                status_text = "全部通过"
            elif success_rate >= 90:
                status_emoji = "🟡"
                status_text = "基本通过"
            elif success_rate >= 70:
                status_emoji = "🟠"
                status_text = "部分失败"
            else:
                status_emoji = "🔴"
                status_text = "大量失败"
            
            print(f"   📊 测试状态: {status_emoji} {status_text}")
        
        # 各套件详情
        print(f"\n📋 测试套件详情:")
        for suite in report.test_suites:
            print(f"\n   📦 {suite.name}:")
            print(f"      测试数: {len(suite.tests)}")
            print(f"      通过: {suite.pass_count}")
            print(f"      失败: {suite.fail_count}")
            print(f"      跳过: {suite.skip_count}")
            print(f"      错误: {suite.error_count}")
            print(f"      耗时: {suite.total_duration:.2f}s")
            
            # 显示失败的测试
            failed_tests = [t for t in suite.tests if t.status in ["fail", "error"]]
            if failed_tests:
                print(f"      ❌ 失败测试:")
                for test in failed_tests[:5]:  # 最多显示5个
                    print(f"         - {test.name}: {test.error[:100]}...")
                
                if len(failed_tests) > 5:
                    print(f"         ... 还有 {len(failed_tests) - 5} 个失败测试")
        
        # 覆盖率评估
        if report.overall_coverage > 0:
            print(f"\n📈 覆盖率评估:")
            if report.overall_coverage >= 90:
                coverage_status = "🟢 优秀"
            elif report.overall_coverage >= 80:
                coverage_status = "🟡 良好"
            elif report.overall_coverage >= 70:
                coverage_status = "🟠 一般"
            else:
                coverage_status = "🔴 需要改进"
            
            print(f"   状态: {coverage_status} ({report.overall_coverage:.1f}%)")
        
        print("\n" + "="*80)
    
    def save_report(self, report: TestReport, output_file: Path):
        """保存测试报告"""
        report_data = {
            "timestamp": report.timestamp.isoformat(),
            "project_info": report.project_info,
            "overall_coverage": report.overall_coverage,
            "total_duration": report.total_duration,
            "total_tests": report.total_tests,
            "total_pass": report.total_pass,
            "total_fail": report.total_fail,
            "total_skip": report.total_skip,
            "total_error": report.total_error,
            "test_suites": []
        }
        
        for suite in report.test_suites:
            suite_data = {
                "name": suite.name,
                "total_duration": suite.total_duration,
                "pass_count": suite.pass_count,
                "fail_count": suite.fail_count,
                "skip_count": suite.skip_count,
                "error_count": suite.error_count,
                "coverage": suite.coverage,
                "tests": [asdict(test) for test in suite.tests]
            }
            report_data["test_suites"].append(suite_data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 测试报告已保存到: {output_file}")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HarborAI 测试运行工具")
    parser.add_argument("--no-coverage", action="store_true", help="跳过覆盖率分析")
    parser.add_argument("--no-parallel", action="store_true", help="串行运行测试")
    parser.add_argument("--output", help="输出报告文件路径（JSON格式）")
    parser.add_argument("--category", choices=["unit", "integration", "performance"], 
                       help="只运行指定类别的测试")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    # 创建测试运行器
    runner = TestRunner(project_root)
    
    try:
        # 如果指定了类别，只运行该类别的测试
        if args.category:
            test_categories = runner.discover_tests()
            if args.category in test_categories:
                files = test_categories[args.category]
                if files:
                    suite = await runner.run_pytest_suite(args.category, files)
                    
                    # 创建简化报告
                    report = TestReport(
                        timestamp=datetime.now(timezone.utc),
                        project_info={
                            "name": "HarborAI",
                            "version": "1.0.0",
                            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                            "platform": sys.platform
                        },
                        test_suites=[suite]
                    )
                else:
                    print(f"❌ 没有找到 {args.category} 类别的测试")
                    sys.exit(1)
            else:
                print(f"❌ 未知的测试类别: {args.category}")
                sys.exit(1)
        else:
            # 运行所有测试
            report = await runner.run_all_tests(
                include_coverage=not args.no_coverage,
                parallel=not args.no_parallel
            )
        
        # 打印报告
        runner.print_test_report(report)
        
        # 保存报告
        if args.output:
            runner.save_report(report, Path(args.output))
        
        # 根据测试结果设置退出码
        if report.total_fail > 0 or report.total_error > 0:
            sys.exit(1)  # 有失败或错误
        elif report.total_tests == 0:
            sys.exit(2)  # 没有测试
        else:
            sys.exit(0)  # 成功
    
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())