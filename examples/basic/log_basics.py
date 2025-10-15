#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 基础日志功能演示

这个脚本演示了 HarborAI 日志系统的基础功能，包括：
1. 创建测试日志数据（支持新的 hb_ 前缀 trace_id 格式）
2. 查看日志文件位置和内容
3. 使用基本的日志查看命令
4. 演示不同的布局模式（classic 和 enhanced）
5. 展示 trace_id 查询功能

使用方法：
    python log_basics.py                    # 运行完整演示
    python log_basics.py --create-only      # 仅创建日志
    python log_basics.py --view-only        # 仅查看日志
    python log_basics.py --demo-layouts     # 演示布局模式
    python log_basics.py --demo-trace-id    # 演示 trace_id 功能

更新内容：
- 支持新的 hb_ 前缀 trace_id 格式（长度从 31 字符减少到 25 字符）
- 添加布局模式演示（classic 和 enhanced）
- 增强 trace_id 查询功能演示
- 改进中文输出和用户体验

作者: HarborAI Team
版本: 2.0.0
更新时间: 2025-01-14
"""

import os
import sys
import json
import time
import random
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from harborai.config.settings import get_settings
    from harborai.database.file_log_parser import FileLogParser
    from harborai.utils.timestamp import get_unified_timestamp, get_unified_timestamp_iso
except ImportError as e:
    print(f"❌ 导入 HarborAI 模块失败: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


class BasicLoggingDemo:
    """基础日志功能演示类"""
    
    def __init__(self):
        """初始化演示类"""
        self.settings = get_settings()
        self.log_directory = Path(self.settings.file_log_directory)
        self.view_logs_script = project_root / "view_logs.py"
        
        # 确保日志目录存在
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 日志目录: {self.log_directory}")
        print(f"🔧 view_logs.py 脚本: {self.view_logs_script}")
        
        # 检查 view_logs.py 是否存在
        if not self.view_logs_script.exists():
            print(f"❌ 未找到 view_logs.py 脚本: {self.view_logs_script}")
            print("💡 请确保在项目根目录运行此脚本")
    
    def generate_new_trace_id(self) -> str:
        """生成新的 trace_id（hb_ 前缀格式）
        
        新格式特点：
        - 前缀从 'harborai_' 简化为 'hb_'
        - 长度从 31 字符减少到 25 字符
        - 格式: hb_{timestamp}_{random_part}
        """
        timestamp = str(int(time.time() * 1000))
        random_part = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        return f"hb_{timestamp}_{random_part}"
    
    def generate_legacy_trace_id(self) -> str:
        """生成旧格式的 trace_id（用于兼容性测试）"""
        timestamp = str(int(time.time() * 1000))
        random_part = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=8))
        return f"harborai_{timestamp}_{random_part}"
    
    def create_sample_logs_today(self):
        """创建今天的示例日志数据"""
        print("\n📝 创建今天的示例日志数据...")
        
        # 获取今天的日志文件路径
        today = datetime.now()
        log_file = self.log_directory / f"harborai_{today.strftime('%Y%m%d')}.jsonl"
        
        # 创建示例日志条目
        sample_logs = []
        
        # 1. 成功的请求-响应对（使用新格式 trace_id）
        for i in range(3):
            trace_id = self.generate_new_trace_id()
            request_time = today - timedelta(hours=i+1)
            response_time = request_time + timedelta(seconds=random.randint(1, 3))
            
            # 请求日志
            sample_logs.append({
                "timestamp": get_unified_timestamp_iso(),
                "trace_id": trace_id,
                "type": "request",
                "model": random.choice(["deepseek-chat", "ernie-4.0-turbo-8k", "doubao-1-5-pro-32k-character-250715"]),
                "provider": random.choice(["deepseek", "ernie", "doubao"]),
                "request": {
                    "messages": [{"role": "user", "content": f"基础演示测试消息 {i+1}"}],
                    "max_tokens": random.randint(100, 300),
                    "temperature": round(random.uniform(0.1, 1.0), 1)
                },
                "metadata": {
                    "user_id": f"basic_demo_user_{i+1}",
                    "session_id": f"basic_demo_session_{i+1}",
                    "demo_type": "basic_success"
                }
            })
            
            # 响应日志
            sample_logs.append({
                "timestamp": get_unified_timestamp_iso(),
                "trace_id": trace_id,
                "type": "response",
                "model": sample_logs[-1]["model"],
                "provider": sample_logs[-1]["provider"],
                "success": True,
                "response": {
                    "content": f"这是对基础演示测试消息 {i+1} 的回复",
                    "finish_reason": "stop"
                },
                "tokens": {
                    "input": random.randint(15, 40),
                    "output": random.randint(25, 80),
                    "total": random.randint(40, 120)
                },
                "latency": random.randint(800, 2500),
                "cost": {
                    "input_cost": round(random.uniform(0.0001, 0.0008), 6),
                    "output_cost": round(random.uniform(0.0002, 0.0015), 6),
                    "total_cost": round(random.uniform(0.0003, 0.0023), 6),
                    "currency": "RMB"
                }
            })
        
        # 2. 失败的请求（使用新格式 trace_id）
        trace_id = self.generate_new_trace_id()
        request_time = today - timedelta(minutes=30)
        response_time = request_time + timedelta(seconds=1)
        
        # 失败请求日志
        sample_logs.append({
            "timestamp": get_unified_timestamp_iso(),
            "trace_id": trace_id,
            "type": "request",
            "model": "gpt-4",
            "provider": "openai",
            "request": {
                "messages": [{"role": "user", "content": "基础演示失败测试消息"}],
                "max_tokens": 200
            },
            "metadata": {
                "user_id": "basic_demo_user_fail",
                "session_id": "basic_demo_session_fail",
                "demo_type": "basic_failure"
            }
        })
        
        # 失败响应日志
        sample_logs.append({
            "timestamp": get_unified_timestamp_iso(),
            "trace_id": trace_id,
            "type": "response",
            "model": "gpt-4",
            "provider": "openai",
            "success": False,
            "error": "API rate limit exceeded",
            "latency": random.randint(100, 500),
            "cost": {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0,
                "currency": "RMB"
            }
        })
        
        # 3. 兼容性测试：添加一个旧格式的 trace_id
        legacy_trace_id = self.generate_legacy_trace_id()
        legacy_time = today - timedelta(minutes=45)
        
        sample_logs.append({
            "timestamp": get_unified_timestamp_iso(),
            "trace_id": legacy_trace_id,
            "type": "request",
            "model": "deepseek-chat",
            "provider": "deepseek",
            "request": {
                "messages": [{"role": "user", "content": "兼容性测试消息（旧格式 trace_id）"}],
                "max_tokens": 150
            },
            "metadata": {
                "user_id": "basic_demo_user_legacy",
                "session_id": "basic_demo_session_legacy",
                "demo_type": "legacy_compatibility"
            }
        })
        
        # 按时间排序
        sample_logs.sort(key=lambda x: x["timestamp"])
        
        # 写入日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            for log_entry in sample_logs:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        print(f"✅ 创建日志文件: {log_file.name}")
        print(f"📊 写入 {len(sample_logs)} 条日志记录")
        
        # 显示 trace_id 格式统计
        new_format_count = sum(1 for log in sample_logs if log['trace_id'].startswith('hb_'))
        old_format_count = sum(1 for log in sample_logs if log['trace_id'].startswith('harborai_'))
        
        print(f"🆔 trace_id 格式统计:")
        print(f"   - 新格式 (hb_): {new_format_count}")
        print(f"   - 旧格式 (harborai_): {old_format_count}")
        
        return len(sample_logs)
    
    def create_sample_logs_yesterday(self):
        """创建昨天的示例日志数据"""
        print("\n📝 创建昨天的示例日志数据...")
        
        # 获取昨天的日志文件路径
        yesterday = datetime.now() - timedelta(days=1)
        log_file = self.log_directory / f"harborai_{yesterday.strftime('%Y%m%d')}.jsonl"
        
        # 创建昨天的示例日志
        sample_logs = []
        
        for i in range(2):
            trace_id = self.generate_new_trace_id()
            request_time = yesterday - timedelta(hours=i*2+1)
            response_time = request_time + timedelta(seconds=random.randint(1, 4))
            
            # 请求日志
            sample_logs.append({
                "timestamp": get_unified_timestamp_iso(),
                "trace_id": trace_id,
                "type": "request",
                "model": "deepseek-chat",
                "provider": "deepseek",
                "request": {
                    "messages": [{"role": "user", "content": f"昨天的基础演示消息 {i+1}"}],
                    "max_tokens": random.randint(100, 250)
                },
                "metadata": {
                    "user_id": f"basic_demo_user_yesterday_{i+1}",
                    "session_id": f"basic_demo_session_yesterday_{i+1}",
                    "demo_type": "basic_yesterday"
                }
            })
            
            # 响应日志
            sample_logs.append({
                "timestamp": get_unified_timestamp_iso(),
                "trace_id": trace_id,
                "type": "response",
                "model": "deepseek-chat",
                "provider": "deepseek",
                "success": True,
                "response": {
                    "content": f"这是对昨天基础演示消息 {i+1} 的回复",
                    "finish_reason": "stop"
                },
                "tokens": {
                    "input": random.randint(20, 50),
                    "output": random.randint(30, 90),
                    "total": random.randint(50, 140)
                },
                "latency": random.randint(1000, 3000),
                "cost": {
                    "input_cost": round(random.uniform(0.0002, 0.001), 6),
                    "output_cost": round(random.uniform(0.0003, 0.002), 6),
                    "total_cost": round(random.uniform(0.0005, 0.003), 6),
                    "currency": "RMB"
                }
            })
        
        # 写入日志文件
        with open(log_file, 'w', encoding='utf-8') as f:
            for log_entry in sample_logs:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        print(f"✅ 创建昨天的日志文件: {log_file.name}")
        print(f"📊 写入 {len(sample_logs)} 条日志记录")
        
        return len(sample_logs)
    
    def show_log_directory_info(self):
        """显示日志目录信息"""
        print("\n📁 日志目录信息:")
        print(f"   路径: {self.log_directory}")
        
        # 列出所有日志文件
        log_files = list(self.log_directory.glob("harborai_*.jsonl"))
        
        if log_files:
            print(f"   文件数量: {len(log_files)}")
            print("   文件列表:")
            
            for log_file in sorted(log_files):
                file_size = log_file.stat().st_size
                mod_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                print(f"     - {log_file.name} ({file_size} bytes, {mod_time.strftime('%Y-%m-%d %H:%M:%S')})")
        else:
            print("   ⚠️  未找到日志文件")
    
    def show_sample_log_content(self):
        """显示示例日志内容"""
        print("\n📄 示例日志内容:")
        
        log_files = list(self.log_directory.glob("harborai_*.jsonl"))
        
        if not log_files:
            print("   ⚠️  未找到日志文件")
            return
        
        # 选择最新的日志文件
        latest_file = sorted(log_files)[-1]
        print(f"   文件: {latest_file.name}")
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
                print(f"   总行数: {len(lines)}")
                print("   前 3 条记录:")
                
                for i, line in enumerate(lines[:3], 1):
                    try:
                        log_entry = json.loads(line.strip())
                        trace_id = log_entry.get('trace_id', 'unknown')
                        log_type = log_entry.get('type', 'unknown')
                        timestamp = log_entry.get('timestamp', 'unknown')
                        model = log_entry.get('model', 'unknown')
                        
                        print(f"     {i}. [{log_type.upper()}] {trace_id} - {model} ({timestamp})")
                        
                        # 显示 trace_id 格式信息
                        if trace_id.startswith('hb_'):
                            print(f"        🆔 新格式 trace_id (长度: {len(trace_id)})")
                        elif trace_id.startswith('harborai_'):
                            print(f"        🆔 旧格式 trace_id (长度: {len(trace_id)})")
                        
                    except json.JSONDecodeError:
                        print(f"     {i}. ❌ 解析失败")
                        
        except Exception as e:
            print(f"   ❌ 读取文件失败: {e}")
    
    def demo_file_log_parser(self):
        """演示 FileLogParser 的使用"""
        print("\n🔍 FileLogParser 使用演示:")
        
        try:
            parser = FileLogParser(str(self.log_directory))
            
            # 获取最近的日志
            print("   获取最近 5 条日志:")
            recent_logs = parser.get_recent_logs(limit=5)
            
            if recent_logs:
                for i, log in enumerate(recent_logs, 1):
                    trace_id = log.get('trace_id', 'unknown')
                    log_type = log.get('type', 'unknown')
                    model = log.get('model', 'unknown')
                    print(f"     {i}. [{log_type.upper()}] {trace_id} - {model}")
            else:
                print("     ⚠️  未找到日志记录")
            
            # 获取统计信息
            print("\n   统计信息:")
            stats = parser.get_statistics(days=1)
            
            if stats:
                print(f"     - 总请求数: {stats.get('total_requests', 0)}")
                print(f"     - 成功响应数: {stats.get('successful_responses', 0)}")
                print(f"     - 失败响应数: {stats.get('failed_responses', 0)}")
                print(f"     - 唯一 trace_id 数: {stats.get('unique_trace_ids', 0)}")
            else:
                print("     ⚠️  无统计数据")
                
        except Exception as e:
            print(f"   ❌ FileLogParser 演示失败: {e}")
    
    def run_view_logs_command(self, args: List[str]) -> tuple[bool, str, str]:
        """运行 view_logs.py 命令"""
        if not self.view_logs_script.exists():
            return False, "", "view_logs.py 脚本不存在"
        
        cmd = ["python", str(self.view_logs_script)] + args
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='ignore',
                timeout=30,
                cwd=str(project_root)
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "命令执行超时"
        except Exception as e:
            return False, "", f"执行命令时出错: {e}"
    
    def demo_basic_log_viewing(self):
        """演示基本日志查看命令"""
        print("\n📋 基本日志查看命令演示:")
        
        # 1. 查看最近的日志
        print("   1. 查看最近 5 条日志:")
        success, stdout, stderr = self.run_view_logs_command(["--limit", "5"])
        
        if success:
            print("     ✅ 命令执行成功")
            # 显示部分输出
            lines = stdout.split('\n')[:10]  # 显示前10行
            for line in lines:
                if line.strip():
                    print(f"     {line}")
            if len(stdout.split('\n')) > 10:
                print("     ... (更多内容)")
        else:
            print(f"     ❌ 命令执行失败: {stderr}")
        
        print("\n   2. 查看统计信息:")
        success, stdout, stderr = self.run_view_logs_command(["--stats", "--days", "1"])
        
        if success:
            print("     ✅ 统计信息:")
            for line in stdout.split('\n'):
                if line.strip():
                    print(f"     {line}")
        else:
            print(f"     ❌ 获取统计信息失败: {stderr}")
    
    def demo_layout_modes(self):
        """演示布局模式"""
        print("\n🎨 布局模式演示:")
        
        # 1. Classic 布局
        print("   1. Classic 布局（默认）:")
        print("      特点: trace_id 作为首列，传统表格显示")
        
        success, stdout, stderr = self.run_view_logs_command([
            "--layout", "classic", "--limit", "4"
        ])
        
        if success:
            print("     ✅ Classic 布局示例:")
            lines = stdout.split('\n')[:8]  # 显示前8行
            for line in lines:
                if line.strip():
                    print(f"     {line}")
        else:
            print(f"     ❌ Classic 布局演示失败: {stderr}")
        
        print("\n   2. Enhanced 布局:")
        print("      特点: 智能配对显示，双时间列，自动计算耗时")
        
        success, stdout, stderr = self.run_view_logs_command([
            "--layout", "enhanced", "--limit", "3"
        ])
        
        if success:
            print("     ✅ Enhanced 布局示例:")
            lines = stdout.split('\n')[:8]  # 显示前8行
            for line in lines:
                if line.strip():
                    print(f"     {line}")
        else:
            print(f"     ❌ Enhanced 布局演示失败: {stderr}")
    
    def demo_trace_id_features(self):
        """演示 trace_id 功能"""
        print("\n🆔 Trace ID 功能演示:")
        
        # 1. 列出最近的 trace_id
        print("   1. 列出最近的 trace_id:")
        success, stdout, stderr = self.run_view_logs_command([
            "--list-recent-trace-ids", "--limit", "8"
        ])
        
        if success:
            print("     ✅ 最近的 trace_id:")
            for line in stdout.split('\n'):
                if line.strip():
                    print(f"     {line}")
        else:
            print(f"     ❌ 获取 trace_id 列表失败: {stderr}")
        
        # 2. 查询特定 trace_id
        print("\n   2. 查询特定 trace_id:")
        
        # 先获取一个可用的 trace_id
        success, stdout, stderr = self.run_view_logs_command([
            "--format", "json", "--limit", "1"
        ])
        
        if success:
            try:
                data = json.loads(stdout)
                logs = data.get('logs', [])
                if logs:
                    trace_id = logs[0].get('trace_id')
                    if trace_id:
                        print(f"     🔍 查询 trace_id: {trace_id}")
                        
                        # 显示 trace_id 格式信息
                        if trace_id.startswith('hb_'):
                            print(f"     🆔 新格式 trace_id (长度: {len(trace_id)})")
                        elif trace_id.startswith('harborai_'):
                            print(f"     🆔 旧格式 trace_id (长度: {len(trace_id)})")
                        
                        success2, stdout2, stderr2 = self.run_view_logs_command([
                            "--trace-id", trace_id
                        ])
                        
                        if success2:
                            print("     ✅ 查询结果:")
                            for line in stdout2.split('\n')[:6]:  # 显示前6行
                                if line.strip():
                                    print(f"     {line}")
                        else:
                            print(f"     ❌ 查询失败: {stderr2}")
                    else:
                        print("     ⚠️  未找到有效的 trace_id")
                else:
                    print("     ⚠️  没有可用的日志记录")
            except json.JSONDecodeError:
                print("     ❌ 解析日志数据失败")
        else:
            print(f"     ❌ 获取日志数据失败: {stderr}")
    
    def show_common_commands(self):
        """显示常用的日志查看命令"""
        print("\n💡 常用日志查看命令:")
        
        commands = [
            ("查看最近 10 条日志", "python view_logs.py --limit 10"),
            ("查看今天的统计信息", "python view_logs.py --stats --days 1"),
            ("仅查看请求日志", "python view_logs.py --type request --limit 5"),
            ("仅查看响应日志", "python view_logs.py --type response --limit 5"),
            ("配对显示请求-响应", "python view_logs.py --type paired --limit 5"),
            ("使用 Enhanced 布局", "python view_logs.py --layout enhanced --limit 5"),
            ("JSON 格式输出", "python view_logs.py --format json --limit 3"),
            ("查询特定 trace_id", "python view_logs.py --trace-id <trace_id>"),
            ("列出最近的 trace_id", "python view_logs.py --list-recent-trace-ids --limit 10"),
            ("查看最近 7 天统计", "python view_logs.py --stats --days 7")
        ]
        
        for i, (desc, cmd) in enumerate(commands, 1):
            print(f"   {i:2d}. {desc}:")
            print(f"       {cmd}")
    
    def run_complete_demo(self):
        """运行完整的基础日志演示"""
        print("="*80)
        print("  🚀 HarborAI 基础日志功能演示")
        print("="*80)
        
        print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 创建示例日志数据
        today_logs = self.create_sample_logs_today()
        yesterday_logs = self.create_sample_logs_yesterday()
        
        # 显示日志目录信息
        self.show_log_directory_info()
        
        # 显示示例日志内容
        self.show_sample_log_content()
        
        # 演示 FileLogParser
        self.demo_file_log_parser()
        
        # 演示基本日志查看
        self.demo_basic_log_viewing()
        
        # 演示布局模式
        self.demo_layout_modes()
        
        # 演示 trace_id 功能
        self.demo_trace_id_features()
        
        # 显示常用命令
        self.show_common_commands()
        
        # 演示总结
        print("\n" + "="*80)
        print("  📊 基础日志演示总结")
        print("="*80)
        print(f"✅ 创建日志记录: {today_logs + yesterday_logs} 条")
        print(f"✅ 演示功能模块: 8 个")
        print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n🎉 基础日志功能演示完成！")
        print("\n💡 下一步建议:")
        print("   1. 尝试运行上面列出的常用命令")
        print("   2. 查看 intermediate/logging_monitoring.py 了解更多功能")
        print("   3. 查看 advanced/log_analysis.py 了解高级分析功能")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="HarborAI 基础日志功能演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python log_basics.py                    # 运行完整演示
  python log_basics.py --create-only      # 仅创建日志
  python log_basics.py --view-only        # 仅查看日志
  python log_basics.py --demo-layouts     # 演示布局模式
  python log_basics.py --demo-trace-id    # 演示 trace_id 功能
        """
    )
    
    parser.add_argument(
        '--create-only',
        action='store_true',
        help='仅创建示例日志数据'
    )
    
    parser.add_argument(
        '--view-only',
        action='store_true',
        help='仅查看现有日志（不创建新数据）'
    )
    
    parser.add_argument(
        '--demo-layouts',
        action='store_true',
        help='仅演示布局模式'
    )
    
    parser.add_argument(
        '--demo-trace-id',
        action='store_true',
        help='仅演示 trace_id 功能'
    )
    
    args = parser.parse_args()
    
    # 创建演示实例
    demo = BasicLoggingDemo()
    
    try:
        if args.create_only:
            # 仅创建日志数据
            demo.create_sample_logs_today()
            demo.create_sample_logs_yesterday()
            demo.show_log_directory_info()
            
        elif args.view_only:
            # 仅查看现有日志
            demo.show_log_directory_info()
            demo.show_sample_log_content()
            demo.demo_file_log_parser()
            demo.demo_basic_log_viewing()
            
        elif args.demo_layouts:
            # 仅演示布局模式
            demo.demo_layout_modes()
            
        elif args.demo_trace_id:
            # 仅演示 trace_id 功能
            demo.demo_trace_id_features()
            
        else:
            # 运行完整演示
            demo.run_complete_demo()
            
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"\n\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()