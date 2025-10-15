#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 日志系统统一演示入口

这个脚本提供了一个统一的入口来运行所有日志系统相关的演示和示例。
包括基础演示、中级演示、高级分析和完整系统演示。

使用方法：
    python run_all_log_demos.py                    # 运行所有演示
    python run_all_log_demos.py --basic            # 仅运行基础演示
    python run_all_log_demos.py --intermediate     # 仅运行中级演示
    python run_all_log_demos.py --advanced         # 仅运行高级演示
    python run_all_log_demos.py --complete         # 仅运行完整演示
    python run_all_log_demos.py --verification     # 仅运行验证脚本
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Tuple

# 尝试导入Rich库用于美化输出
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


class LogDemoRunner:
    """日志演示运行器"""
    
    def __init__(self):
        """初始化演示运行器"""
        self.project_root = Path(__file__).parent.parent
        self.examples_dir = Path(__file__).parent
        
        # 定义所有可用的演示脚本
        self.demos = {
            "verification": {
                "name": "日志系统验证",
                "script": "log_system_verification_report.py",
                "description": "验证日志系统的基本功能和数据完整性"
            },
            "basic": {
                "name": "基础日志演示",
                "script": "basic/log_basics.py",
                "description": "演示基础的日志记录和查看功能"
            },
            "intermediate": {
                "name": "中级日志监控",
                "script": "intermediate/logging_monitoring.py",
                "description": "演示日志监控、性能指标和告警功能"
            },
            "advanced": {
                "name": "高级日志分析",
                "script": "advanced/log_analysis.py",
                "description": "演示高级的日志分析和统计功能"
            },
            "complete": {
                "name": "完整系统演示",
                "script": "demo_complete_logging_system.py",
                "description": "全面演示所有日志系统功能特性"
            }
        }
    
    def print_section(self, title: str, emoji: str = "📋"):
        """打印章节标题"""
        if HAS_RICH:
            console.print(Panel(f"{emoji} {title}", style="bold blue"))
        else:
            print(f"\n{'='*60}")
            print(f"  {emoji} {title}")
            print(f"{'='*60}")
    
    def print_step(self, step: str, emoji: str = "🔸"):
        """打印步骤"""
        if HAS_RICH:
            console.print(f"{emoji} {step}", style="cyan")
        else:
            print(f"\n{emoji} {step}")
    
    def print_success(self, message: str):
        """打印成功信息"""
        if HAS_RICH:
            console.print(f"✅ {message}", style="green")
        else:
            print(f"✅ {message}")
    
    def print_warning(self, message: str):
        """打印警告信息"""
        if HAS_RICH:
            console.print(f"⚠️ {message}", style="yellow")
        else:
            print(f"⚠️ {message}")
    
    def print_error(self, message: str):
        """打印错误信息"""
        if HAS_RICH:
            console.print(f"❌ {message}", style="red")
        else:
            print(f"❌ {message}")
    
    def print_info(self, message: str):
        """打印信息"""
        if HAS_RICH:
            console.print(f"ℹ️ {message}", style="blue")
        else:
            print(f"ℹ️ {message}")
    
    def check_script_exists(self, script_path: str) -> bool:
        """检查脚本文件是否存在"""
        full_path = self.examples_dir / script_path
        return full_path.exists()
    
    def run_script(self, script_path: str, args: List[str] = None) -> Tuple[bool, str, str]:
        """运行指定的脚本"""
        full_path = self.examples_dir / script_path
        
        if not full_path.exists():
            return False, "", f"脚本文件不存在: {full_path}"
        
        cmd = ["python", str(full_path)]
        if args:
            cmd.extend(args)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore',
                timeout=120,  # 2分钟超时
                cwd=str(self.project_root)
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "脚本执行超时"
        except Exception as e:
            return False, "", f"执行脚本时出错: {e}"
    
    def run_demo(self, demo_key: str) -> bool:
        """运行单个演示"""
        if demo_key not in self.demos:
            self.print_error(f"未知的演示类型: {demo_key}")
            return False
        
        demo = self.demos[demo_key]
        self.print_step(f"运行 {demo['name']}")
        self.print_info(demo['description'])
        
        if not self.check_script_exists(demo['script']):
            self.print_error(f"演示脚本不存在: {demo['script']}")
            return False
        
        # 根据不同的演示类型设置不同的参数
        args = []
        if demo_key == "complete":
            args = ["--no-create-logs"]  # 避免重复创建日志
        elif demo_key == "basic":
            args = ["--view-only"]  # 仅查看日志
        
        success, stdout, stderr = self.run_script(demo['script'], args)
        
        if success:
            self.print_success(f"{demo['name']} 运行成功")
            # 显示部分输出
            if stdout:
                output_preview = stdout[:300] + "..." if len(stdout) > 300 else stdout
                if HAS_RICH:
                    console.print(f"输出预览:\n{output_preview}")
                else:
                    print(f"输出预览:\n{output_preview}")
        else:
            self.print_error(f"{demo['name']} 运行失败")
            if stderr:
                error_preview = stderr[:200] + "..." if len(stderr) > 200 else stderr
                if HAS_RICH:
                    console.print(f"错误信息: {error_preview}", style="red")
                else:
                    print(f"错误信息: {error_preview}")
        
        return success
    
    def show_available_demos(self):
        """显示可用的演示"""
        self.print_section("可用的日志系统演示", "📋")
        
        if HAS_RICH:
            table = Table(title="HarborAI 日志系统演示列表")
            table.add_column("演示类型", style="cyan")
            table.add_column("名称", style="green")
            table.add_column("描述", style="white")
            table.add_column("状态", style="yellow")
            
            for key, demo in self.demos.items():
                status = "✅ 可用" if self.check_script_exists(demo['script']) else "❌ 缺失"
                table.add_row(key, demo['name'], demo['description'], status)
            
            console.print(table)
        else:
            print("\n📋 可用的日志系统演示:")
            for key, demo in self.demos.items():
                status = "✅ 可用" if self.check_script_exists(demo['script']) else "❌ 缺失"
                print(f"  {key:12} - {demo['name']:15} - {demo['description']} [{status}]")
    
    def run_all_demos(self):
        """运行所有演示"""
        self.print_section("运行所有日志系统演示", "🚀")
        
        # 按顺序运行演示
        demo_order = ["verification", "basic", "intermediate", "advanced", "complete"]
        results = {}
        
        for demo_key in demo_order:
            if demo_key in self.demos:
                results[demo_key] = self.run_demo(demo_key)
                print()  # 添加空行分隔
        
        # 显示总结
        self.show_summary(results)
    
    def show_summary(self, results: dict):
        """显示运行总结"""
        self.print_section("演示运行总结", "📊")
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        if HAS_RICH:
            table = Table(title="演示运行结果")
            table.add_column("演示", style="cyan")
            table.add_column("状态", style="white")
            
            for demo_key, success in results.items():
                demo_name = self.demos[demo_key]['name']
                status = "✅ 成功" if success else "❌ 失败"
                table.add_row(demo_name, status)
            
            console.print(table)
            
            # 总体状态
            if success_count == total_count:
                console.print(f"🎉 所有演示运行成功！({success_count}/{total_count})", style="bold green")
            else:
                console.print(f"⚠️ 部分演示失败 ({success_count}/{total_count})", style="bold yellow")
        else:
            print(f"\n📊 演示运行结果:")
            for demo_key, success in results.items():
                demo_name = self.demos[demo_key]['name']
                status = "✅ 成功" if success else "❌ 失败"
                print(f"  {demo_name:20} - {status}")
            
            print(f"\n总体状态: {success_count}/{total_count} 成功")
        
        # 使用建议
        if HAS_RICH:
            suggestions = Panel(
                "💡 使用建议:\n"
                "• 如果某个演示失败，可以单独运行该演示进行调试\n"
                "• 使用 --help 查看每个演示脚本的详细参数\n"
                "• 定期运行验证脚本确保日志系统正常工作\n"
                "• 查看 LOG_FEATURES_GUIDE.md 了解更多功能特性",
                title="使用建议",
                style="blue"
            )
            console.print(suggestions)
        else:
            print("\n💡 使用建议:")
            print("• 如果某个演示失败，可以单独运行该演示进行调试")
            print("• 使用 --help 查看每个演示脚本的详细参数")
            print("• 定期运行验证脚本确保日志系统正常工作")
            print("• 查看 LOG_FEATURES_GUIDE.md 了解更多功能特性")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HarborAI 日志系统统一演示入口")
    parser.add_argument("--basic", action="store_true", help="仅运行基础演示")
    parser.add_argument("--intermediate", action="store_true", help="仅运行中级演示")
    parser.add_argument("--advanced", action="store_true", help="仅运行高级演示")
    parser.add_argument("--complete", action="store_true", help="仅运行完整演示")
    parser.add_argument("--verification", action="store_true", help="仅运行验证脚本")
    parser.add_argument("--list", action="store_true", help="列出所有可用的演示")
    
    args = parser.parse_args()
    
    runner = LogDemoRunner()
    
    # 显示欢迎信息
    if HAS_RICH:
        console.print(Panel(
            "🚀 HarborAI 日志系统统一演示入口\n\n"
            "这里提供了所有日志系统相关的演示和示例，\n"
            "帮助您全面了解 HarborAI 的日志功能。",
            title="欢迎使用 HarborAI 日志演示",
            style="bold green"
        ))
    else:
        print("🚀 HarborAI 日志系统统一演示入口")
        print("这里提供了所有日志系统相关的演示和示例")
    
    # 根据参数运行相应的演示
    if args.list:
        runner.show_available_demos()
    elif args.verification:
        runner.run_demo("verification")
    elif args.basic:
        runner.run_demo("basic")
    elif args.intermediate:
        runner.run_demo("intermediate")
    elif args.advanced:
        runner.run_demo("advanced")
    elif args.complete:
        runner.run_demo("complete")
    else:
        # 运行所有演示
        runner.run_all_demos()


if __name__ == "__main__":
    main()