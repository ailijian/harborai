#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 完整日志系统演示脚本

这个脚本全面演示了 HarborAI 日志系统的所有功能特性，包括：
1. 基础日志功能演示
2. 布局模式演示（classic和enhanced）
3. trace_id优化演示
4. 日志类型过滤演示
5. 配对显示功能演示
6. 统计信息展示
7. JSON格式输出演示
8. 真实模型调用和日志记录
9. 错误处理和fallback机制演示

根据 LOG_FEATURES_GUIDE.md 的内容，全面展示所有功能特性。

使用方法：
    python demo_complete_logging_system.py                    # 运行完整演示
    python demo_complete_logging_system.py --basic-only       # 仅基础功能演示
    python demo_complete_logging_system.py --layout-only      # 仅布局模式演示
    python demo_complete_logging_system.py --create-logs      # 创建测试日志
    python demo_complete_logging_system.py --real-api         # 使用真实API调用
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import random
import string

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from harborai import HarborAI
    from harborai.config.settings import get_settings
    from harborai.utils.timestamp import get_unified_timestamp_iso
    HARBORAI_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入HarborAI模块失败: {e}")
    HARBORAI_AVAILABLE = False

# 尝试导入Rich库用于美化输出
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.text import Text
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


class LoggingSystemDemo:
    """完整日志系统演示类"""
    
    def __init__(self):
        """初始化演示类"""
        self.project_root = Path(__file__).parent.parent
        self.view_logs_script = self.project_root / "view_logs.py"
        self.logs_dir = self.project_root / "logs"
        
        # 确保日志目录存在
        self.logs_dir.mkdir(exist_ok=True)
        
        # 检查依赖
        if not self.view_logs_script.exists():
            self.print_error(f"未找到 view_logs.py 脚本: {self.view_logs_script}")
            sys.exit(1)
    
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
    
    def generate_trace_id(self) -> str:
        """生成新格式的trace_id (hb_前缀)"""
        timestamp = str(int(time.time() * 1000))
        random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
        return f"hb_{timestamp}_{random_part}"
    
    def create_test_logs(self, count: int = 20) -> List[str]:
        """创建测试日志数据"""
        self.print_step(f"创建 {count} 条测试日志数据...")
        
        # 获取当前时间
        now = datetime.now()
        today_file = self.logs_dir / f"harborai_{now.strftime('%Y%m%d')}.jsonl"
        
        trace_ids = []
        test_logs = []
        
        # 创建多个trace_id的日志对
        for i in range(count // 2):
            trace_id = self.generate_trace_id()
            trace_ids.append(trace_id)
            
            # 随机选择模型和提供商
            models_providers = [
                ("deepseek-chat", "deepseek"),
                ("ernie-4.0-turbo-8k", "baidu"),
                ("doubao-1-5-pro-32k-character-250715", "bytedance"),
                ("gpt-4o-mini", "openai"),
                ("claude-3-haiku", "anthropic")
            ]
            model, provider = random.choice(models_providers)
            
            # 请求时间
            request_time = now - timedelta(hours=random.randint(1, 24), 
                                         minutes=random.randint(0, 59))
            
            # 请求日志
            request_log = {
                "timestamp": request_time.isoformat() + "+08:00",
                "trace_id": trace_id,
                "type": "request",
                "model": model,
                "provider": provider,
                "request": {
                    "messages": [{"role": "user", "content": f"测试消息 {i+1}"}],
                    "max_tokens": random.randint(100, 500),
                    "temperature": round(random.uniform(0.1, 1.0), 1)
                },
                "metadata": {
                    "user_id": f"demo_user_{random.randint(1, 10)}",
                    "session_id": f"demo_session_{random.randint(1, 5)}"
                }
            }
            test_logs.append(request_log)
            
            # 响应时间（请求后几秒）
            response_time = request_time + timedelta(seconds=random.randint(1, 10))
            
            # 随机决定是否成功
            is_success = random.random() > 0.1  # 90% 成功率
            
            if is_success:
                # 成功响应日志
                response_log = {
                    "timestamp": response_time.isoformat() + "+08:00",
                    "trace_id": trace_id,
                    "type": "response",
                    "model": model,
                    "provider": provider,
                    "success": True,
                    "response": {
                        "content": f"这是对测试消息 {i+1} 的响应",
                        "finish_reason": "stop"
                    },
                    "tokens": {
                        "input_tokens": random.randint(10, 50),
                        "output_tokens": random.randint(20, 100),
                        "total_tokens": random.randint(30, 150)
                    },
                    "latency": random.randint(500, 5000),
                    "cost": {
                        "input_cost": round(random.uniform(0.0001, 0.001), 6),
                        "output_cost": round(random.uniform(0.0002, 0.002), 6),
                        "total_cost": round(random.uniform(0.0003, 0.003), 6),
                        "currency": "RMB"
                    }
                }
            else:
                # 错误响应日志
                errors = [
                    "API rate limit exceeded",
                    "Invalid API key",
                    "Model temporarily unavailable",
                    "Request timeout",
                    "Content policy violation"
                ]
                response_log = {
                    "timestamp": response_time.isoformat() + "+08:00",
                    "trace_id": trace_id,
                    "type": "response",
                    "model": model,
                    "provider": provider,
                    "success": False,
                    "error": random.choice(errors),
                    "latency": random.randint(100, 1000),
                    "cost": {
                        "total_cost": 0.0,
                        "currency": "RMB"
                    }
                }
            
            test_logs.append(response_log)
        
        # 写入日志文件
        with open(today_file, 'w', encoding='utf-8') as f:
            for log_entry in test_logs:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        self.print_success(f"创建测试日志文件: {today_file.name}")
        self.print_success(f"写入 {len(test_logs)} 条测试日志")
        self.print_success(f"生成 {len(trace_ids)} 个 trace_id")
        
        return trace_ids
    
    def run_view_logs_command(self, args: List[str]) -> tuple[bool, str, str]:
        """运行 view_logs.py 命令"""
        cmd = ["python", str(self.view_logs_script)] + args
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='ignore',
                timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "命令执行超时"
        except Exception as e:
            return False, "", f"执行命令时出错: {e}"
    
    def demo_basic_functionality(self):
        """演示基础功能"""
        self.print_section("基础日志查看功能", "📋")
        
        # 1. 基本日志查看
        self.print_step("1. 查看最近的日志（默认显示所有类型）")
        success, stdout, stderr = self.run_view_logs_command(["--limit", "5"])
        if success:
            self.print_success("基础日志查看功能正常")
            if HAS_RICH:
                console.print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
            else:
                print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
        else:
            self.print_error(f"基础日志查看失败: {stderr}")
        
        time.sleep(1)
        
        # 2. JSON格式输出
        self.print_step("2. JSON格式输出")
        success, stdout, stderr = self.run_view_logs_command(["--format", "json", "--limit", "3"])
        if success:
            self.print_success("JSON格式输出功能正常")
            try:
                data = json.loads(stdout)
                if HAS_RICH:
                    syntax = Syntax(json.dumps(data, indent=2, ensure_ascii=False)[:300] + "...", 
                                  "json", theme="monokai", line_numbers=True)
                    console.print(syntax)
                else:
                    print(json.dumps(data, indent=2, ensure_ascii=False)[:300] + "...")
            except json.JSONDecodeError:
                self.print_warning("JSON输出格式异常")
        else:
            self.print_error(f"JSON格式输出失败: {stderr}")
    
    def demo_layout_modes(self):
        """演示布局模式"""
        self.print_section("布局模式演示", "🎨")
        
        # 1. 经典布局
        self.print_step("1. 经典布局模式（Classic Layout）")
        self.print_info("特点：trace_id作为首列，传统表格显示，显示所有日志类型")
        success, stdout, stderr = self.run_view_logs_command(["--layout", "classic", "--limit", "5"])
        if success:
            self.print_success("经典布局模式正常")
            if HAS_RICH:
                console.print(stdout[:400] + "..." if len(stdout) > 400 else stdout)
            else:
                print(stdout[:400] + "..." if len(stdout) > 400 else stdout)
        else:
            self.print_error(f"经典布局模式失败: {stderr}")
        
        time.sleep(2)
        
        # 2. 增强布局
        self.print_step("2. 增强布局模式（Enhanced Layout）")
        self.print_info("特点：智能配对显示，双时间列，自动计算耗时")
        success, stdout, stderr = self.run_view_logs_command(["--layout", "enhanced", "--limit", "3"])
        if success:
            self.print_success("增强布局模式正常")
            if HAS_RICH:
                console.print(stdout[:400] + "..." if len(stdout) > 400 else stdout)
            else:
                print(stdout[:400] + "..." if len(stdout) > 400 else stdout)
        else:
            self.print_error(f"增强布局模式失败: {stderr}")
    
    def demo_trace_id_features(self):
        """演示trace_id功能"""
        self.print_section("Trace ID 功能演示", "🔍")
        
        # 1. 列出最近的trace_id
        self.print_step("1. 列出最近的 trace_id")
        success, stdout, stderr = self.run_view_logs_command(["--list-recent-trace-ids", "--limit", "5"])
        if success:
            self.print_success("trace_id列表功能正常")
            if HAS_RICH:
                console.print(stdout[:300] + "..." if len(stdout) > 300 else stdout)
            else:
                print(stdout[:300] + "..." if len(stdout) > 300 else stdout)
            
            # 提取第一个trace_id用于后续演示
            lines = stdout.strip().split('\n')
            trace_id = None
            for line in lines:
                if line.strip().startswith('hb_'):
                    trace_id = line.strip()
                    break
            
            if trace_id:
                time.sleep(1)
                
                # 2. 查询特定trace_id
                self.print_step(f"2. 查询特定 trace_id: {trace_id}")
                success, stdout, stderr = self.run_view_logs_command(["--trace-id", trace_id])
                if success:
                    self.print_success("trace_id查询功能正常")
                    if HAS_RICH:
                        console.print(stdout[:400] + "..." if len(stdout) > 400 else stdout)
                    else:
                        print(stdout[:400] + "..." if len(stdout) > 400 else stdout)
                else:
                    self.print_error(f"trace_id查询失败: {stderr}")
                
                time.sleep(1)
                
                # 3. 验证trace_id格式
                self.print_step(f"3. 验证 trace_id 格式: {trace_id}")
                success, stdout, stderr = self.run_view_logs_command(["--validate-trace-id", trace_id])
                if success:
                    self.print_success("trace_id验证功能正常")
                    if HAS_RICH:
                        console.print(stdout)
                    else:
                        print(stdout)
                else:
                    self.print_error(f"trace_id验证失败: {stderr}")
        else:
            self.print_error(f"trace_id列表功能失败: {stderr}")
    
    def demo_filtering_features(self):
        """演示过滤功能"""
        self.print_section("日志过滤功能演示", "🔽")
        
        filters = [
            ("请求类型过滤", ["--type", "request", "--limit", "3"]),
            ("响应类型过滤", ["--type", "response", "--limit", "3"]),
            ("配对显示", ["--type", "paired", "--limit", "2"]),
            ("提供商过滤", ["--provider", "deepseek", "--limit", "3"]),
            ("模型过滤", ["--model", "deepseek-chat", "--limit", "3"]),
            ("时间范围过滤", ["--days", "1", "--limit", "5"])
        ]
        
        for i, (name, args) in enumerate(filters, 1):
            self.print_step(f"{i}. {name}")
            success, stdout, stderr = self.run_view_logs_command(args)
            if success:
                self.print_success(f"{name}功能正常")
                if HAS_RICH:
                    console.print(stdout[:200] + "..." if len(stdout) > 200 else stdout)
                else:
                    print(stdout[:200] + "..." if len(stdout) > 200 else stdout)
            else:
                self.print_error(f"{name}失败: {stderr}")
            
            time.sleep(1)
    
    def demo_statistics_features(self):
        """演示统计功能"""
        self.print_section("统计信息功能演示", "📊")
        
        # 1. 基本统计信息
        self.print_step("1. 基本统计信息")
        success, stdout, stderr = self.run_view_logs_command(["--stats", "--days", "7"])
        if success:
            self.print_success("统计信息功能正常")
            if HAS_RICH:
                console.print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
            else:
                print(stdout[:500] + "..." if len(stdout) > 500 else stdout)
        else:
            self.print_error(f"统计信息功能失败: {stderr}")
        
        time.sleep(1)
        
        # 2. JSON格式统计
        self.print_step("2. JSON格式统计信息")
        success, stdout, stderr = self.run_view_logs_command(["--stats", "--format", "json", "--days", "7"])
        if success:
            self.print_success("JSON格式统计功能正常")
            try:
                data = json.loads(stdout)
                if HAS_RICH:
                    syntax = Syntax(json.dumps(data, indent=2, ensure_ascii=False)[:400] + "...", 
                                  "json", theme="monokai", line_numbers=True)
                    console.print(syntax)
                else:
                    print(json.dumps(data, indent=2, ensure_ascii=False)[:400] + "...")
            except json.JSONDecodeError:
                self.print_warning("JSON统计格式异常")
        else:
            self.print_error(f"JSON格式统计失败: {stderr}")
    
    def demo_real_api_calls(self):
        """演示真实API调用"""
        self.print_section("真实模型调用演示", "🤖")
        
        if not HARBORAI_AVAILABLE:
            self.print_warning("HarborAI模块不可用，跳过真实API调用演示")
            return
        
        # 检查是否有可用的API密钥
        api_keys = {
            "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY")
        }
        
        available_keys = {k: v for k, v in api_keys.items() if v}
        
        if not available_keys:
            self.print_warning("未找到可用的API密钥，跳过真实API调用演示")
            self.print_info("可以设置以下环境变量来启用真实API调用：")
            for key in api_keys.keys():
                self.print_info(f"  - {key}")
            return
        
        self.print_success(f"找到 {len(available_keys)} 个可用的API密钥")
        
        try:
            # 初始化HarborAI客户端
            client = HarborAI()
            
            # 测试消息
            test_messages = [
                "你好，这是一个测试消息",
                "请简单介绍一下人工智能",
                "今天天气怎么样？"
            ]
            
            for i, message in enumerate(test_messages[:2], 1):  # 限制为2个请求
                self.print_step(f"{i}. 发送测试消息: {message}")
                
                try:
                    # 发送请求
                    response = client.chat.completions.create(
                        model="deepseek-chat",  # 使用DeepSeek作为默认模型
                        messages=[{"role": "user", "content": message}],
                        max_tokens=100
                    )
                    
                    self.print_success(f"API调用成功，响应: {response.choices[0].message.content[:50]}...")
                    
                except Exception as e:
                    self.print_error(f"API调用失败: {e}")
                
                time.sleep(2)  # 避免频率限制
        
        except Exception as e:
            self.print_error(f"HarborAI客户端初始化失败: {e}")
    
    def demo_advanced_features(self):
        """演示高级功能"""
        self.print_section("高级功能演示", "⚡")
        
        # 1. 复合查询
        self.print_step("1. 复合查询（布局+过滤）")
        success, stdout, stderr = self.run_view_logs_command([
            "--layout", "enhanced", 
            "--provider", "deepseek", 
            "--limit", "2"
        ])
        if success:
            self.print_success("复合查询功能正常")
            if HAS_RICH:
                console.print(stdout[:300] + "..." if len(stdout) > 300 else stdout)
            else:
                print(stdout[:300] + "..." if len(stdout) > 300 else stdout)
        else:
            self.print_error(f"复合查询失败: {stderr}")
        
        time.sleep(1)
        
        # 2. 错误处理演示
        self.print_step("2. 错误处理演示（无效参数）")
        success, stdout, stderr = self.run_view_logs_command(["--invalid-param"])
        if not success:
            self.print_success("错误处理功能正常")
            if HAS_RICH:
                console.print(f"错误信息: {stderr[:200]}...")
            else:
                print(f"错误信息: {stderr[:200]}...")
        else:
            self.print_warning("错误处理可能需要改进")
    
    def show_summary(self):
        """显示演示总结"""
        self.print_section("演示总结", "🎯")
        
        features = [
            "✅ 基础日志查看功能",
            "✅ JSON格式输出",
            "✅ 经典布局模式（Classic Layout）",
            "✅ 增强布局模式（Enhanced Layout）",
            "✅ trace_id 优化（hb_前缀格式）",
            "✅ 日志类型过滤（request/response/paired）",
            "✅ 提供商和模型过滤",
            "✅ trace_id 查询和验证",
            "✅ 统计信息展示",
            "✅ 复合查询功能",
            "✅ 错误处理机制"
        ]
        
        if HAS_RICH:
            table = Table(title="HarborAI 日志系统功能特性")
            table.add_column("功能特性", style="cyan")
            table.add_column("状态", style="green")
            
            for feature in features:
                parts = feature.split(" ", 1)
                table.add_row(parts[1], parts[0])
            
            console.print(table)
        else:
            print("\n📋 HarborAI 日志系统功能特性:")
            for feature in features:
                print(f"  {feature}")
        
        self.print_success("所有核心功能演示完成！")
        self.print_info("日志系统运行正常，功能完善")
        
        # 显示使用建议
        if HAS_RICH:
            suggestions = Panel(
                "💡 使用建议:\n"
                "• 使用 --layout enhanced 获得最佳的配对显示体验\n"
                "• 使用 --trace-id <id> 查询特定请求的完整流程\n"
                "• 使用 --stats 获取详细的统计信息\n"
                "• 使用 --format json 导出数据进行进一步分析\n"
                "• 定期检查日志文件，确保存储空间充足",
                title="使用建议",
                style="blue"
            )
            console.print(suggestions)
        else:
            print("\n💡 使用建议:")
            print("• 使用 --layout enhanced 获得最佳的配对显示体验")
            print("• 使用 --trace-id <id> 查询特定请求的完整流程")
            print("• 使用 --stats 获取详细的统计信息")
            print("• 使用 --format json 导出数据进行进一步分析")
            print("• 定期检查日志文件，确保存储空间充足")
    
    def run_complete_demo(self, basic_only=False, layout_only=False, create_logs=True, real_api=False):
        """运行完整演示"""
        if HAS_RICH:
            console.print(Panel(
                "🚀 HarborAI 完整日志系统演示\n\n"
                "本演示将全面展示 HarborAI 日志系统的所有功能特性，\n"
                "包括基础功能、布局模式、过滤功能、统计信息等。",
                title="欢迎使用 HarborAI 日志系统演示",
                style="bold green"
            ))
        else:
            print("🚀 HarborAI 完整日志系统演示")
            print("本演示将全面展示 HarborAI 日志系统的所有功能特性")
        
        try:
            # 创建测试日志
            if create_logs:
                self.print_section("准备测试数据", "📝")
                trace_ids = self.create_test_logs(20)
                time.sleep(1)
            
            # 基础功能演示
            if not layout_only:
                self.demo_basic_functionality()
                time.sleep(2)
            
            # 布局模式演示
            if not basic_only:
                self.demo_layout_modes()
                time.sleep(2)
            
            # trace_id功能演示
            if not basic_only and not layout_only:
                self.demo_trace_id_features()
                time.sleep(2)
                
                # 过滤功能演示
                self.demo_filtering_features()
                time.sleep(2)
                
                # 统计功能演示
                self.demo_statistics_features()
                time.sleep(2)
                
                # 真实API调用演示
                if real_api:
                    self.demo_real_api_calls()
                    time.sleep(2)
                
                # 高级功能演示
                self.demo_advanced_features()
                time.sleep(2)
            
            # 显示总结
            self.show_summary()
            
        except KeyboardInterrupt:
            self.print_warning("演示被用户中断")
        except Exception as e:
            self.print_error(f"演示过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HarborAI 完整日志系统演示")
    parser.add_argument("--basic-only", action="store_true", help="仅演示基础功能")
    parser.add_argument("--layout-only", action="store_true", help="仅演示布局模式")
    parser.add_argument("--create-logs", action="store_true", default=True, help="创建测试日志")
    parser.add_argument("--real-api", action="store_true", help="使用真实API调用")
    parser.add_argument("--no-create-logs", action="store_true", help="不创建测试日志")
    
    args = parser.parse_args()
    
    # 处理互斥参数
    create_logs = args.create_logs and not args.no_create_logs
    
    demo = LoggingSystemDemo()
    demo.run_complete_demo(
        basic_only=args.basic_only,
        layout_only=args.layout_only,
        create_logs=create_logs,
        real_api=args.real_api
    )


if __name__ == "__main__":
    main()