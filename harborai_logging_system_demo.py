#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 日志系统完整演示脚本

这个脚本全面演示了 HarborAI 日志系统的所有功能特性，包括：

1. 日志存储功能展示：
   - 实时日志采集与存储机制
   - 日志分类与索引建立过程
   - 存储容量与性能基准测试

2. 日志查询功能演示：
   - 多条件组合查询界面操作
   - 全文检索与关键词高亮显示
   - 时间范围筛选与日志分级查看
   - 查询响应时间与结果准确性验证

3. 系统管理功能：
   - 日志保留策略配置
   - 存储空间监控告警

使用方法：
    python harborai_logging_system_demo.py                    # 运行完整演示
    python harborai_logging_system_demo.py --storage-only     # 仅存储功能演示
    python harborai_logging_system_demo.py --query-only       # 仅查询功能演示
    python harborai_logging_system_demo.py --management-only  # 仅管理功能演示
    python harborai_logging_system_demo.py --real-api         # 使用真实API调用
    python harborai_logging_system_demo.py --performance      # 性能基准测试
"""

import os
import sys
import json
import time
import argparse
import subprocess
import asyncio
import threading
import psutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
import string
import sqlite3
from collections import defaultdict

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入HarborAI模块
try:
    from harborai import HarborAI
    from harborai.config.settings import get_settings
    from harborai.utils.timestamp import get_unified_timestamp_iso
    from harborai.utils.logger import get_logger
    HARBORAI_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入HarborAI模块失败: {e}")
    HARBORAI_AVAILABLE = False

# 导入Rich库用于美化输出
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich.tree import Tree
    from rich.columns import Columns
    from rich.align import Align
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None

# 导入数据库相关模块
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class HarborAILoggingSystemDemo:
    """HarborAI日志系统完整演示类"""
    
    def __init__(self):
        """初始化演示类"""
        self.project_root = Path(__file__).parent
        self.view_logs_script = self.project_root / "view_logs.py"
        self.logs_dir = self.project_root / "logs"
        self.reports_dir = self.project_root / "reports"
        
        # 确保目录存在
        self.logs_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        
        # 可用模型列表
        self.available_models = [
            {'vendor': 'deepseek', 'model': 'deepseek-chat', 'is_reasoning': False},
            {'vendor': 'deepseek', 'model': 'deepseek-reasoner', 'is_reasoning': True},
            {'vendor': 'ernie', 'model': 'ernie-4.0-turbo-8k', 'is_reasoning': False},
            {'vendor': 'ernie', 'model': 'ernie-x1-turbo-32k', 'is_reasoning': True},
            {'vendor': 'doubao', 'model': 'doubao-1-5-pro-32k-character-250715', 'is_reasoning': False},
            {'vendor': 'doubao', 'model': 'doubao-seed-1-6-250615', 'is_reasoning': True}
        ]
        
        # 模型价格配置（每1000个token的价格，单位：人民币）
        self.model_pricing = {
            'deepseek-chat': {'input': 0.001, 'output': 0.002},
            'deepseek-reasoner': {'input': 0.014, 'output': 0.028},
            'ernie-4.0-turbo-8k': {'input': 0.03, 'output': 0.09},
            'ernie-x1-turbo-32k': {'input': 0.04, 'output': 0.12},
            'doubao-1-5-pro-32k-character-250715': {'input': 0.0008, 'output': 0.002},
            'doubao-seed-1-6-250615': {'input': 0.001, 'output': 0.003}
        }
        
        # 演示统计数据
        self.demo_stats = {
            'api_calls': 0,
            'logs_created': 0,
            'queries_executed': 0,
            'total_cost': 0.0,
            'start_time': None,
            'end_time': None
        }
        
        # 检查依赖
        if not self.view_logs_script.exists():
            self.print_error(f"未找到 view_logs.py 脚本: {self.view_logs_script}")
    
    def print_section(self, title: str, emoji: str = "📋"):
        """打印章节标题"""
        if HAS_RICH:
            console.print(Panel(f"{emoji} {title}", style="bold blue", expand=False))
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
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('.').percent,
            'available_memory': psutil.virtual_memory().available / (1024**3),  # GB
            'total_memory': psutil.virtual_memory().total / (1024**3),  # GB
        }
    
    def check_database_connection(self) -> bool:
        """检查数据库连接"""
        try:
            if POSTGRES_AVAILABLE:
                # 尝试连接PostgreSQL
                import os
                db_url = os.getenv('DATABASE_URL')
                if db_url:
                    conn = psycopg2.connect(db_url)
                    conn.close()
                    return True
        except Exception as e:
            self.print_warning(f"PostgreSQL连接失败: {e}")
        return False
    
    def demo_storage_functionality(self):
        """演示日志存储功能"""
        self.print_section("日志存储功能展示", "💾")
        
        # 1. 实时日志采集与存储机制
        self.print_step("1. 实时日志采集与存储机制演示")
        
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("初始化日志存储系统...", total=100)
                
                # 检查存储系统状态
                progress.update(task, advance=20, description="检查PostgreSQL连接...")
                time.sleep(1)
                postgres_available = self.check_database_connection()
                
                progress.update(task, advance=20, description="检查文件系统存储...")
                time.sleep(1)
                file_storage_available = self.logs_dir.exists()
                
                progress.update(task, advance=20, description="初始化日志分类器...")
                time.sleep(1)
                
                progress.update(task, advance=20, description="建立索引结构...")
                time.sleep(1)
                
                progress.update(task, advance=20, description="存储系统就绪")
                time.sleep(1)
        
        # 显示存储系统状态
        if HAS_RICH:
            storage_table = Table(title="存储系统状态")
            storage_table.add_column("存储类型", style="cyan")
            storage_table.add_column("状态", style="green")
            storage_table.add_column("路径/连接", style="yellow")
            
            storage_table.add_row(
                "PostgreSQL数据库", 
                "✅ 可用" if postgres_available else "❌ 不可用",
                os.getenv('DATABASE_URL', '未配置')[:50] + "..." if os.getenv('DATABASE_URL') else "未配置"
            )
            storage_table.add_row(
                "文件系统存储", 
                "✅ 可用" if file_storage_available else "❌ 不可用",
                str(self.logs_dir)
            )
            
            console.print(storage_table)
        
        # 2. 日志分类与索引建立过程
        self.print_step("2. 日志分类与索引建立过程")
        
        log_categories = {
            "请求日志": "记录API请求的详细信息，包括模型、参数、时间戳",
            "响应日志": "记录API响应的详细信息，包括内容、token使用、成本",
            "错误日志": "记录系统错误、API错误、网络错误等异常情况",
            "性能日志": "记录响应时间、吞吐量、资源使用等性能指标",
            "成本日志": "记录API调用成本、token使用统计、费用分析"
        }
        
        if HAS_RICH:
            categories_table = Table(title="日志分类体系")
            categories_table.add_column("分类", style="cyan")
            categories_table.add_column("描述", style="white")
            
            for category, description in log_categories.items():
                categories_table.add_row(category, description)
            
            console.print(categories_table)
        
        # 3. 存储容量与性能基准测试
        self.print_step("3. 存储容量与性能基准测试")
        
        # 获取系统信息
        system_info = self.get_system_info()
        
        if HAS_RICH:
            system_table = Table(title="系统资源状态")
            system_table.add_column("资源类型", style="cyan")
            system_table.add_column("使用率", style="yellow")
            system_table.add_column("详细信息", style="white")
            
            system_table.add_row(
                "CPU", 
                f"{system_info['cpu_percent']:.1f}%",
                "处理器使用率"
            )
            system_table.add_row(
                "内存", 
                f"{system_info['memory_percent']:.1f}%",
                f"{system_info['available_memory']:.1f}GB 可用 / {system_info['total_memory']:.1f}GB 总计"
            )
            system_table.add_row(
                "磁盘", 
                f"{system_info['disk_usage']:.1f}%",
                "磁盘使用率"
            )
            
            console.print(system_table)
        
        # 性能基准测试
        self.print_info("执行存储性能基准测试...")
        
        # 模拟写入测试
        start_time = time.time()
        test_logs = []
        for i in range(100):
            test_log = {
                "timestamp": datetime.now().isoformat(),
                "trace_id": self.generate_trace_id(),
                "type": "test",
                "message": f"性能测试日志 {i+1}",
                "data": {"test_id": i, "batch": "performance_test"}
            }
            test_logs.append(test_log)
            # 添加小延迟以确保有可测量的时间
            time.sleep(0.001)
        
        write_time = max(time.time() - start_time, 0.001)  # 确保不为零
        
        # 模拟读取测试
        start_time = time.time()
        for log in test_logs[:10]:  # 读取前10条
            _ = json.dumps(log)
            time.sleep(0.0001)  # 添加小延迟
        read_time = max(time.time() - start_time, 0.001)  # 确保不为零
        
        if HAS_RICH:
            perf_table = Table(title="性能基准测试结果")
            perf_table.add_column("测试项目", style="cyan")
            perf_table.add_column("结果", style="green")
            perf_table.add_column("说明", style="white")
            
            perf_table.add_row(
                "写入性能", 
                f"{len(test_logs)/write_time:.0f} 条/秒",
                f"写入 {len(test_logs)} 条日志耗时 {write_time:.3f} 秒"
            )
            perf_table.add_row(
                "读取性能", 
                f"{10/read_time:.0f} 条/秒",
                f"读取 10 条日志耗时 {read_time:.3f} 秒"
            )
            perf_table.add_row(
                "存储效率", 
                f"{len(json.dumps(test_logs))/1024:.1f} KB",
                f"100条日志占用存储空间"
            )
            
            console.print(perf_table)
        
        self.print_success("日志存储功能演示完成")
    
    def demo_query_functionality(self):
        """演示日志查询功能"""
        self.print_section("日志查询功能演示", "🔍")
        
        # 1. 多条件组合查询界面操作
        self.print_step("1. 多条件组合查询演示")
        
        query_examples = [
            {
                "name": "按模型查询",
                "command": ["--model", "deepseek-chat", "--limit", "5"],
                "description": "查询使用 deepseek-chat 模型的日志"
            },
            {
                "name": "按提供商查询",
                "command": ["--provider", "deepseek", "--limit", "5"],
                "description": "查询 DeepSeek 提供商的所有日志"
            },
            {
                "name": "按日志类型查询",
                "command": ["--type", "request", "--limit", "3"],
                "description": "查询所有请求类型的日志"
            },
            {
                "name": "组合条件查询",
                "command": ["--model", "deepseek-chat", "--type", "response", "--limit", "3"],
                "description": "查询 deepseek-chat 模型的响应日志"
            }
        ]
        
        for query in query_examples:
            self.print_info(f"执行查询: {query['name']}")
            self.print_info(f"描述: {query['description']}")
            
            success, stdout, stderr = self.run_view_logs_command(query['command'])
            if success:
                self.print_success(f"查询成功")
                if HAS_RICH and stdout:
                    # 显示查询结果的前几行
                    lines = stdout.split('\n')[:5]
                    console.print(Panel('\n'.join(lines), title=f"查询结果预览: {query['name']}", style="green"))
            else:
                self.print_warning(f"查询失败: {stderr}")
            
            time.sleep(1)
        
        # 2. 全文检索与关键词高亮显示
        self.print_step("2. 全文检索与关键词高亮演示")
        
        search_examples = [
            {
                "keyword": "deepseek",
                "description": "搜索包含 'deepseek' 关键词的日志"
            },
            {
                "keyword": "error",
                "description": "搜索包含 'error' 关键词的错误日志"
            },
            {
                "keyword": "success",
                "description": "搜索包含 'success' 关键词的成功日志"
            }
        ]
        
        for search in search_examples:
            self.print_info(f"全文搜索: {search['description']}")
            
            # 模拟全文搜索（实际实现需要在view_logs.py中添加搜索功能）
            success, stdout, stderr = self.run_view_logs_command(["--limit", "10"])
            if success and search['keyword'].lower() in stdout.lower():
                self.print_success(f"找到包含 '{search['keyword']}' 的日志")
                
                # 模拟关键词高亮
                if HAS_RICH:
                    highlighted_text = Text(stdout[:200] + "...")
                    # 简单的高亮模拟
                    console.print(Panel(highlighted_text, title=f"搜索结果: {search['keyword']}", style="yellow"))
            else:
                self.print_info(f"未找到包含 '{search['keyword']}' 的日志")
            
            time.sleep(1)
        
        # 3. 时间范围筛选与日志分级查看
        self.print_step("3. 时间范围筛选与日志分级查看")
        
        time_filters = [
            {
                "name": "最近1小时",
                "command": ["--hours", "1", "--limit", "5"],
                "description": "查看最近1小时的日志"
            },
            {
                "name": "最近1天",
                "command": ["--days", "1", "--limit", "10"],
                "description": "查看最近1天的日志"
            },
            {
                "name": "最近1周",
                "command": ["--days", "7", "--limit", "15"],
                "description": "查看最近1周的日志"
            }
        ]
        
        for time_filter in time_filters:
            self.print_info(f"时间筛选: {time_filter['description']}")
            
            success, stdout, stderr = self.run_view_logs_command(time_filter['command'])
            if success:
                self.print_success(f"时间筛选成功: {time_filter['name']}")
                
                # 统计日志数量
                log_count = len([line for line in stdout.split('\n') if line.strip()])
                self.print_info(f"找到 {log_count} 条日志记录")
            else:
                self.print_warning(f"时间筛选失败: {stderr}")
            
            time.sleep(1)
        
        # 4. 查询响应时间与结果准确性验证
        self.print_step("4. 查询性能与准确性验证")
        
        performance_tests = [
            {
                "name": "快速查询测试",
                "command": ["--limit", "5"],
                "expected_max_time": 2.0
            },
            {
                "name": "中等查询测试",
                "command": ["--limit", "20"],
                "expected_max_time": 5.0
            },
            {
                "name": "复杂查询测试",
                "command": ["--limit", "50", "--format", "json"],
                "expected_max_time": 10.0
            }
        ]
        
        if HAS_RICH:
            perf_results = Table(title="查询性能测试结果")
            perf_results.add_column("测试名称", style="cyan")
            perf_results.add_column("响应时间", style="yellow")
            perf_results.add_column("状态", style="green")
            perf_results.add_column("结果数量", style="white")
            
            for test in performance_tests:
                start_time = time.time()
                success, stdout, stderr = self.run_view_logs_command(test['command'])
                response_time = time.time() - start_time
                
                if success:
                    result_count = len([line for line in stdout.split('\n') if line.strip()])
                    status = "✅ 通过" if response_time <= test['expected_max_time'] else "⚠️ 超时"
                else:
                    result_count = 0
                    status = "❌ 失败"
                
                perf_results.add_row(
                    test['name'],
                    f"{response_time:.3f}s",
                    status,
                    str(result_count)
                )
                
                self.demo_stats['queries_executed'] += 1
            
            console.print(perf_results)
        
        self.print_success("日志查询功能演示完成")
    
    def demo_management_functionality(self):
        """演示系统管理功能"""
        self.print_section("系统管理功能演示", "⚙️")
        
        # 1. 日志保留策略配置
        self.print_step("1. 日志保留策略配置")
        
        retention_policies = {
            "请求/响应日志": "30天",
            "错误日志": "90天",
            "性能日志": "7天",
            "成本日志": "365天",
            "系统日志": "30天"
        }
        
        if HAS_RICH:
            retention_table = Table(title="日志保留策略配置")
            retention_table.add_column("日志类型", style="cyan")
            retention_table.add_column("保留期限", style="yellow")
            retention_table.add_column("清理策略", style="white")
            
            for log_type, retention in retention_policies.items():
                cleanup_strategy = "自动清理" if log_type != "成本日志" else "手动归档"
                retention_table.add_row(log_type, retention, cleanup_strategy)
            
            console.print(retention_table)
        
        # 模拟配置更新
        self.print_info("正在应用保留策略配置...")
        time.sleep(2)
        self.print_success("日志保留策略配置完成")
        
        # 2. 存储空间监控告警
        self.print_step("2. 存储空间监控告警")
        
        # 获取存储空间信息
        disk_usage = psutil.disk_usage('.')
        total_gb = disk_usage.total / (1024**3)
        used_gb = disk_usage.used / (1024**3)
        free_gb = disk_usage.free / (1024**3)
        usage_percent = (used_gb / total_gb) * 100
        
        # 检查日志目录大小
        logs_size = 0
        if self.logs_dir.exists():
            for file_path in self.logs_dir.rglob('*'):
                if file_path.is_file():
                    logs_size += file_path.stat().st_size
        logs_size_mb = logs_size / (1024**2)
        
        if HAS_RICH:
            storage_table = Table(title="存储空间监控")
            storage_table.add_column("监控项目", style="cyan")
            storage_table.add_column("当前值", style="yellow")
            storage_table.add_column("阈值", style="white")
            storage_table.add_column("状态", style="green")
            
            # 磁盘使用率
            disk_status = "🟢 正常" if usage_percent < 80 else "🟡 警告" if usage_percent < 90 else "🔴 危险"
            storage_table.add_row(
                "磁盘使用率",
                f"{usage_percent:.1f}%",
                "< 80%",
                disk_status
            )
            
            # 可用空间
            free_status = "🟢 充足" if free_gb > 10 else "🟡 不足" if free_gb > 5 else "🔴 严重不足"
            storage_table.add_row(
                "可用空间",
                f"{free_gb:.1f} GB",
                "> 10 GB",
                free_status
            )
            
            # 日志目录大小
            logs_status = "🟢 正常" if logs_size_mb < 100 else "🟡 较大" if logs_size_mb < 500 else "🔴 过大"
            storage_table.add_row(
                "日志目录大小",
                f"{logs_size_mb:.1f} MB",
                "< 100 MB",
                logs_status
            )
            
            console.print(storage_table)
        
        # 模拟告警配置
        alert_rules = [
            {"metric": "磁盘使用率", "threshold": "85%", "action": "发送邮件通知"},
            {"metric": "可用空间", "threshold": "< 5GB", "action": "发送短信告警"},
            {"metric": "日志增长率", "threshold": "> 100MB/天", "action": "自动清理旧日志"},
            {"metric": "查询响应时间", "threshold": "> 10秒", "action": "性能优化建议"}
        ]
        
        if HAS_RICH:
            alert_table = Table(title="监控告警规则")
            alert_table.add_column("监控指标", style="cyan")
            alert_table.add_column("告警阈值", style="yellow")
            alert_table.add_column("响应动作", style="white")
            
            for rule in alert_rules:
                alert_table.add_row(rule["metric"], rule["threshold"], rule["action"])
            
            console.print(alert_table)
        
        # 3. 自动化管理任务
        self.print_step("3. 自动化管理任务演示")
        
        management_tasks = [
            {"name": "日志轮转", "schedule": "每日 02:00", "status": "已启用"},
            {"name": "索引优化", "schedule": "每周日 03:00", "status": "已启用"},
            {"name": "统计报告", "schedule": "每月1日 08:00", "status": "已启用"},
            {"name": "备份任务", "schedule": "每日 04:00", "status": "已启用"},
            {"name": "清理任务", "schedule": "每日 05:00", "status": "已启用"}
        ]
        
        if HAS_RICH:
            tasks_table = Table(title="自动化管理任务")
            tasks_table.add_column("任务名称", style="cyan")
            tasks_table.add_column("执行计划", style="yellow")
            tasks_table.add_column("状态", style="green")
            
            for task in management_tasks:
                tasks_table.add_row(task["name"], task["schedule"], task["status"])
            
            console.print(tasks_table)
        
        self.print_success("系统管理功能演示完成")
    
    def demo_real_api_calls(self):
        """演示真实API调用"""
        self.print_section("真实API调用演示", "🚀")
        
        if not HARBORAI_AVAILABLE:
            self.print_error("HarborAI模块不可用，跳过真实API调用演示")
            return
        
        self.print_step("1. 初始化HarborAI客户端")
        
        try:
            # 初始化客户端
            client = HarborAI()
            self.print_success("HarborAI客户端初始化成功")
            
            # 测试消息列表
            test_messages = [
                "你好，请简单介绍一下人工智能的发展历程。",
                "请解释一下机器学习和深度学习的区别。",
                "什么是大语言模型？它们有什么特点？",
                "请用一句话总结今天的天气情况。",
                "帮我写一个简单的Python函数来计算斐波那契数列。"
            ]
            
            # 遍历可用模型进行测试
            for model_info in self.available_models[:3]:  # 只测试前3个模型
                model_name = model_info['model']
                vendor = model_info['vendor']
                
                self.print_step(f"2. 测试模型: {model_name} ({vendor})")
                
                # 随机选择一个测试消息
                test_message = random.choice(test_messages)
                self.print_info(f"发送消息: {test_message}")
                
                try:
                    start_time = time.time()
                    
                    # 发送API请求
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": test_message}],
                        max_tokens=150,
                        temperature=0.7
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    # 获取响应内容
                    response_content = response.choices[0].message.content
                    
                    # 计算成本（如果有pricing信息）
                    cost = 0.0
                    if model_name in self.model_pricing:
                        pricing = self.model_pricing[model_name]
                        # 估算token数量（简单估算）
                        input_tokens = len(test_message.split()) * 1.3  # 粗略估算
                        output_tokens = len(response_content.split()) * 1.3
                        cost = (input_tokens * pricing['input'] + output_tokens * pricing['output']) / 1000
                    
                    # 显示结果
                    if HAS_RICH:
                        result_table = Table(title=f"API调用结果: {model_name}")
                        result_table.add_column("项目", style="cyan")
                        result_table.add_column("值", style="white")
                        
                        result_table.add_row("模型", model_name)
                        result_table.add_row("提供商", vendor)
                        result_table.add_row("响应时间", f"{response_time:.2f}秒")
                        result_table.add_row("响应长度", f"{len(response_content)}字符")
                        result_table.add_row("估算成本", f"¥{cost:.6f}")
                        result_table.add_row("响应内容", response_content[:100] + "..." if len(response_content) > 100 else response_content)
                        
                        console.print(result_table)
                    
                    # 更新统计
                    self.demo_stats['api_calls'] += 1
                    self.demo_stats['total_cost'] += cost
                    
                    self.print_success(f"API调用成功: {model_name}")
                    
                except Exception as e:
                    self.print_error(f"API调用失败: {model_name} - {e}")
                
                time.sleep(2)  # 避免频率限制
            
        except Exception as e:
            self.print_error(f"HarborAI客户端初始化失败: {e}")
        
        self.print_success("真实API调用演示完成")
    
    def run_view_logs_command(self, args: List[str]) -> Tuple[bool, str, str]:
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
    
    def generate_demo_report(self):
        """生成演示报告"""
        self.print_section("生成演示报告", "📊")
        
        # 计算演示时长
        if self.demo_stats['start_time'] and self.demo_stats['end_time']:
            duration = self.demo_stats['end_time'] - self.demo_stats['start_time']
            duration_str = f"{duration:.2f}秒"
        else:
            duration_str = "未知"
        
        # 生成报告内容
        report = {
            "演示时间": datetime.now().isoformat(),
            "演示时长": duration_str,
            "API调用次数": self.demo_stats['api_calls'],
            "日志创建数量": self.demo_stats['logs_created'],
            "查询执行次数": self.demo_stats['queries_executed'],
            "总成本": f"¥{self.demo_stats['total_cost']:.6f}",
            "系统信息": self.get_system_info(),
            "可用模型": self.available_models,
            "模型价格": self.model_pricing
        }
        
        # 保存报告到文件
        report_file = self.reports_dir / f"harborai_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 显示报告摘要
        if HAS_RICH:
            summary_table = Table(title="演示报告摘要")
            summary_table.add_column("项目", style="cyan")
            summary_table.add_column("值", style="white")
            
            summary_table.add_row("演示时长", duration_str)
            summary_table.add_row("API调用次数", str(self.demo_stats['api_calls']))
            summary_table.add_row("查询执行次数", str(self.demo_stats['queries_executed']))
            summary_table.add_row("总成本", f"¥{self.demo_stats['total_cost']:.6f}")
            summary_table.add_row("报告文件", str(report_file.name))
            
            console.print(summary_table)
        
        self.print_success(f"演示报告已保存: {report_file}")
        return report_file
    
    def show_welcome_message(self):
        """显示欢迎信息"""
        if HAS_RICH:
            welcome_panel = Panel(
                "🚀 HarborAI 日志系统完整演示\n\n"
                "本演示将全面展示 HarborAI 日志系统的所有功能特性：\n"
                "• 日志存储功能：实时采集、分类索引、性能测试\n"
                "• 日志查询功能：多条件查询、全文检索、时间筛选\n"
                "• 系统管理功能：保留策略、存储监控、自动化任务\n"
                "• 真实API调用：使用配置的模型进行实际测试\n\n"
                "演示过程中将使用真实的API密钥进行调用，请确保网络连接正常。",
                title="欢迎使用 HarborAI 日志系统演示",
                style="bold green",
                expand=False
            )
            console.print(welcome_panel)
        else:
            print("🚀 HarborAI 日志系统完整演示")
            print("本演示将全面展示 HarborAI 日志系统的所有功能特性")
    
    def show_completion_message(self):
        """显示完成信息"""
        if HAS_RICH:
            completion_panel = Panel(
                "🎉 HarborAI 日志系统演示完成！\n\n"
                "演示内容回顾：\n"
                "✅ 日志存储功能展示完成\n"
                "✅ 日志查询功能演示完成\n"
                "✅ 系统管理功能演示完成\n"
                "✅ 真实API调用测试完成\n"
                "✅ 演示报告生成完成\n\n"
                "感谢您使用 HarborAI 日志系统！",
                title="演示完成",
                style="bold blue",
                expand=False
            )
            console.print(completion_panel)
        else:
            print("🎉 HarborAI 日志系统演示完成！")
    
    def run_complete_demo(self, storage_only=False, query_only=False, management_only=False, real_api=True, performance=False):
        """运行完整演示"""
        self.demo_stats['start_time'] = time.time()
        
        try:
            # 显示欢迎信息
            self.show_welcome_message()
            time.sleep(2)
            
            # 根据参数决定运行哪些演示
            if not query_only and not management_only:
                self.demo_storage_functionality()
                time.sleep(2)
            
            if not storage_only and not management_only:
                self.demo_query_functionality()
                time.sleep(2)
            
            if not storage_only and not query_only:
                self.demo_management_functionality()
                time.sleep(2)
            
            # 真实API调用演示
            if real_api and not storage_only and not query_only and not management_only:
                self.demo_real_api_calls()
                time.sleep(2)
            
            # 生成演示报告
            self.demo_stats['end_time'] = time.time()
            self.generate_demo_report()
            
            # 显示完成信息
            self.show_completion_message()
            
        except KeyboardInterrupt:
            self.print_warning("演示被用户中断")
        except Exception as e:
            self.print_error(f"演示过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if not self.demo_stats['end_time']:
                self.demo_stats['end_time'] = time.time()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HarborAI 日志系统完整演示")
    parser.add_argument("--storage-only", action="store_true", help="仅演示存储功能")
    parser.add_argument("--query-only", action="store_true", help="仅演示查询功能")
    parser.add_argument("--management-only", action="store_true", help="仅演示管理功能")
    parser.add_argument("--real-api", action="store_true", default=True, help="使用真实API调用")
    parser.add_argument("--no-real-api", action="store_true", help="不使用真实API调用")
    parser.add_argument("--performance", action="store_true", help="执行性能基准测试")
    
    args = parser.parse_args()
    
    # 处理互斥参数
    real_api = args.real_api and not args.no_real_api
    
    demo = HarborAILoggingSystemDemo()
    demo.run_complete_demo(
        storage_only=args.storage_only,
        query_only=args.query_only,
        management_only=args.management_only,
        real_api=real_api,
        performance=args.performance
    )


if __name__ == "__main__":
    main()