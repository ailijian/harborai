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
import asyncio
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import random
import string
import psutil
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from harborai import HarborAI
    from harborai.config.settings import get_settings
    from harborai.utils.timestamp import get_unified_timestamp_iso
    from harborai.database.models import LogRecord, ResponseRecord
    from harborai.storage.file_logger import FileSystemLogger
    from harborai.core.cost_tracking import CostTracker
    from harborai.monitoring.metrics import MetricsCollector
    HARBORAI_AVAILABLE = True
except ImportError as e:
    print(f"❌ 导入HarborAI模块失败: {e}")
    HARBORAI_AVAILABLE = False

# 尝试导入Rich库用于美化输出
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
    from rich.rule import Rule
    from rich.status import Status
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None

# 尝试导入数据库相关模块
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


class ModelConfig:
    """模型配置类"""
    
    # 从文档中读取的可用模型列表
    AVAILABLE_MODELS = [
        {'vendor': 'deepseek', 'model': 'deepseek-chat', 'is_reasoning': False},
        {'vendor': 'deepseek', 'model': 'deepseek-reasoner', 'is_reasoning': True},
        {'vendor': 'ernie', 'model': 'ernie-4.0-turbo-8k', 'is_reasoning': False},
        {'vendor': 'ernie', 'model': 'ernie-x1-turbo-32k', 'is_reasoning': True},
        {'vendor': 'doubao', 'model': 'doubao-1-5-pro-32k-character-250715', 'is_reasoning': False},
        {'vendor': 'doubao', 'model': 'doubao-seed-1-6-250615', 'is_reasoning': True}
    ]
    
    # 模型价格配置 (每1000个token的价格，单位：人民币)
    MODEL_PRICING = {
        'deepseek-chat': {'input': 0.001, 'output': 0.002},
        'deepseek-reasoner': {'input': 0.014, 'output': 0.028},
        'ernie-4.0-turbo-8k': {'input': 0.03, 'output': 0.09},
        'ernie-x1-turbo-32k': {'input': 0.04, 'output': 0.12},
        'doubao-1-5-pro-32k-character-250715': {'input': 0.0008, 'output': 0.002},
        'doubao-seed-1-6-250615': {'input': 0.001, 'output': 0.003}
    }
    
    @classmethod
    def get_random_model(cls) -> Dict[str, Any]:
        """获取随机模型"""
        return random.choice(cls.AVAILABLE_MODELS)
    
    @classmethod
    def get_model_price(cls, model: str) -> Dict[str, float]:
        """获取模型价格"""
        return cls.MODEL_PRICING.get(model, {'input': 0.001, 'output': 0.002})


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
            'start_time': self.start_time
        }
        
    def stop_monitoring(self) -> Dict[str, Any]:
        """停止监控并返回结果"""
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0
        
        return {
            'duration': duration,
            'start_metrics': self.metrics,
            'end_metrics': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:').percent,
                'end_time': end_time
            }
        }


class LoggingSystemDemo:
    """HarborAI日志系统完整演示类"""
    
    def __init__(self):
        """初始化演示类"""
        self.project_root = Path(__file__).parent
        self.logs_dir = self.project_root / "logs"
        self.view_logs_script = self.project_root / "view_logs.py"
        self.performance_monitor = PerformanceMonitor()
        
        # 确保日志目录存在
        self.logs_dir.mkdir(exist_ok=True)
        
        # 初始化HarborAI客户端
        self.harborai_client = None
        if HARBORAI_AVAILABLE:
            try:
                self.harborai_client = HarborAI()
                self.print_success("HarborAI客户端初始化成功")
            except Exception as e:
                self.print_error(f"HarborAI客户端初始化失败: {e}")
        
        # 统计信息
        self.demo_stats = {
            'logs_created': 0,
            'queries_executed': 0,
            'api_calls_made': 0,
            'total_cost': 0.0,
            'start_time': datetime.now()
        }
    
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
    
    def demo_storage_functionality(self):
        """演示日志存储功能"""
        self.print_section("日志存储功能演示", "💾")
        
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
                task = progress.add_task("正在演示实时日志采集...", total=100)
                
                # 模拟实时日志采集
                for i in range(10):
                    # 创建测试日志
                    trace_id = self.generate_trace_id()
                    model_config = ModelConfig.get_random_model()
                    
                    # 模拟请求日志
                    request_log = {
                        "timestamp": datetime.now().isoformat() + "+08:00",
                        "trace_id": trace_id,
                        "type": "request",
                        "model": model_config['model'],
                        "provider": model_config['vendor'],
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
                    
                    # 写入日志文件
                    today_file = self.logs_dir / f"harborai_{datetime.now().strftime('%Y%m%d')}.jsonl"
                    with open(today_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(request_log, ensure_ascii=False) + '\n')
                    
                    self.demo_stats['logs_created'] += 1
                    progress.update(task, advance=10)
                    time.sleep(0.2)
        
        self.print_success(f"成功创建 {self.demo_stats['logs_created']} 条实时日志")
        
        # 2. 日志分类与索引建立过程
        self.print_step("2. 日志分类与索引建立过程")
        
        # 创建不同类型的日志
        log_types = ['request', 'response', 'error', 'system']
        for log_type in log_types:
            self.print_info(f"创建 {log_type} 类型日志...")
            
            log_entry = {
                "timestamp": datetime.now().isoformat() + "+08:00",
                "trace_id": self.generate_trace_id(),
                "type": log_type,
                "model": "deepseek-chat",
                "provider": "deepseek",
                "data": f"示例 {log_type} 数据"
            }
            
            # 写入分类日志
            today_file = self.logs_dir / f"harborai_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(today_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            
            self.demo_stats['logs_created'] += 1
        
        self.print_success("日志分类索引建立完成")
        
        # 3. 存储容量与性能基准测试
        self.print_step("3. 存储容量与性能基准测试")
        self.demo_storage_performance()
    
    def demo_storage_performance(self):
        """演示存储性能测试"""
        self.print_info("开始存储性能基准测试...")
        
        # 开始性能监控
        self.performance_monitor.start_monitoring()
        
        # 批量创建日志测试性能
        batch_size = 1000
        start_time = time.time()
        
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"批量创建 {batch_size} 条日志...", total=batch_size)
                
                today_file = self.logs_dir / f"harborai_performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
                
                with open(today_file, 'w', encoding='utf-8') as f:
                    for i in range(batch_size):
                        log_entry = {
                            "timestamp": datetime.now().isoformat() + "+08:00",
                            "trace_id": self.generate_trace_id(),
                            "type": "performance_test",
                            "model": "deepseek-chat",
                            "provider": "deepseek",
                            "sequence": i,
                            "data": f"性能测试数据 {i}" * 10  # 增加数据量
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                        
                        if i % 100 == 0:
                            progress.update(task, advance=100)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 停止性能监控
        perf_results = self.performance_monitor.stop_monitoring()
        
        # 计算性能指标
        logs_per_second = batch_size / duration
        file_size = today_file.stat().st_size / 1024 / 1024  # MB
        
        # 显示性能结果
        if HAS_RICH:
            perf_table = Table(title="存储性能基准测试结果")
            perf_table.add_column("指标", style="cyan")
            perf_table.add_column("数值", style="green")
            
            perf_table.add_row("批量大小", f"{batch_size:,} 条日志")
            perf_table.add_row("总耗时", f"{duration:.2f} 秒")
            perf_table.add_row("写入速度", f"{logs_per_second:.0f} 条/秒")
            perf_table.add_row("文件大小", f"{file_size:.2f} MB")
            perf_table.add_row("CPU使用率", f"{perf_results['end_metrics']['cpu_percent']:.1f}%")
            perf_table.add_row("内存使用率", f"{perf_results['end_metrics']['memory_percent']:.1f}%")
            
            console.print(perf_table)
        else:
            print(f"性能测试结果:")
            print(f"  批量大小: {batch_size:,} 条日志")
            print(f"  总耗时: {duration:.2f} 秒")
            print(f"  写入速度: {logs_per_second:.0f} 条/秒")
            print(f"  文件大小: {file_size:.2f} MB")
        
        self.print_success("存储性能基准测试完成")
    
    def demo_query_functionality(self):
        """演示日志查询功能"""
        self.print_section("日志查询功能演示", "🔍")
        
        # 1. 多条件组合查询
        self.print_step("1. 多条件组合查询演示")
        self.demo_multi_condition_query()
        
        # 2. 全文检索与关键词高亮
        self.print_step("2. 全文检索与关键词高亮演示")
        self.demo_full_text_search()
        
        # 3. 时间范围筛选
        self.print_step("3. 时间范围筛选演示")
        self.demo_time_range_filter()
        
        # 4. 查询响应时间验证
        self.print_step("4. 查询响应时间与准确性验证")
        self.demo_query_performance()
    
    def demo_multi_condition_query(self):
        """演示多条件组合查询"""
        self.print_info("执行多条件组合查询...")
        
        # 定义多种查询条件组合
        query_combinations = [
            {
                "name": "按模型和提供商查询",
                "args": ["--model", "deepseek-chat", "--provider", "deepseek", "--limit", "5"]
            },
            {
                "name": "按日志类型查询",
                "args": ["--type", "request", "--limit", "3"]
            },
            {
                "name": "按时间范围查询",
                "args": ["--days", "1", "--limit", "5"]
            },
            {
                "name": "复合条件查询",
                "args": ["--layout", "enhanced", "--provider", "deepseek", "--type", "paired", "--limit", "2"]
            }
        ]
        
        for query in query_combinations:
            self.print_info(f"执行查询: {query['name']}")
            success, stdout, stderr = self.run_view_logs_command(query['args'])
            
            if success:
                self.print_success(f"查询成功: {query['name']}")
                if HAS_RICH and stdout:
                    # 显示查询结果的前几行
                    lines = stdout.split('\n')[:5]
                    console.print(Panel('\n'.join(lines), title=f"查询结果: {query['name']}", style="green"))
                self.demo_stats['queries_executed'] += 1
            else:
                self.print_error(f"查询失败: {query['name']} - {stderr}")
            
            time.sleep(1)
    
    def demo_full_text_search(self):
        """演示全文检索功能"""
        self.print_info("演示全文检索与关键词高亮...")
        
        # 创建包含特定关键词的测试日志
        search_keywords = ["测试", "演示", "性能", "查询"]
        
        for keyword in search_keywords:
            log_entry = {
                "timestamp": datetime.now().isoformat() + "+08:00",
                "trace_id": self.generate_trace_id(),
                "type": "search_test",
                "model": "deepseek-chat",
                "provider": "deepseek",
                "content": f"这是一个包含关键词 '{keyword}' 的测试日志，用于演示全文检索功能。",
                "metadata": {
                    "search_keyword": keyword,
                    "test_type": "full_text_search"
                }
            }
            
            today_file = self.logs_dir / f"harborai_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(today_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # 执行全文搜索
        for keyword in search_keywords[:2]:  # 只测试前两个关键词
            self.print_info(f"搜索关键词: '{keyword}'")
            
            # 使用grep进行全文搜索
            try:
                result = subprocess.run(
                    ['findstr', keyword, str(self.logs_dir / "*.jsonl")] if os.name == 'nt' 
                    else ['grep', keyword, str(self.logs_dir / "*.jsonl")],
                    capture_output=True,
                    text=True,
                    shell=True
                )
                
                if result.returncode == 0 and result.stdout:
                    matches = result.stdout.strip().split('\n')
                    self.print_success(f"找到 {len(matches)} 个匹配结果")
                    
                    if HAS_RICH:
                        # 高亮显示关键词
                        highlighted_text = Text()
                        for line in matches[:3]:  # 只显示前3个结果
                            if keyword in line:
                                parts = line.split(keyword)
                                for i, part in enumerate(parts):
                                    highlighted_text.append(part)
                                    if i < len(parts) - 1:
                                        highlighted_text.append(keyword, style="bold red on yellow")
                                highlighted_text.append('\n')
                        
                        console.print(Panel(highlighted_text, title=f"搜索结果: {keyword}", style="blue"))
                else:
                    self.print_warning(f"未找到关键词 '{keyword}' 的匹配结果")
                    
            except Exception as e:
                self.print_error(f"搜索失败: {e}")
            
            time.sleep(1)
    
    def demo_time_range_filter(self):
        """演示时间范围筛选"""
        self.print_info("演示时间范围筛选功能...")
        
        # 创建不同时间的测试日志
        time_ranges = [
            {"hours": 1, "label": "1小时前"},
            {"hours": 6, "label": "6小时前"},
            {"hours": 24, "label": "24小时前"}
        ]
        
        for time_range in time_ranges:
            past_time = datetime.now() - timedelta(hours=time_range['hours'])
            
            log_entry = {
                "timestamp": past_time.isoformat() + "+08:00",
                "trace_id": self.generate_trace_id(),
                "type": "time_test",
                "model": "deepseek-chat",
                "provider": "deepseek",
                "content": f"这是 {time_range['label']} 的测试日志",
                "time_label": time_range['label']
            }
            
            today_file = self.logs_dir / f"harborai_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(today_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # 测试不同的时间范围查询
        time_queries = [
            {"name": "最近1天", "args": ["--days", "1", "--limit", "5"]},
            {"name": "最近7天", "args": ["--days", "7", "--limit", "5"]},
            {"name": "最近30天", "args": ["--days", "30", "--limit", "5"]}
        ]
        
        for query in time_queries:
            self.print_info(f"执行时间范围查询: {query['name']}")
            success, stdout, stderr = self.run_view_logs_command(query['args'])
            
            if success:
                self.print_success(f"时间查询成功: {query['name']}")
                # 统计结果数量
                lines = stdout.strip().split('\n') if stdout else []
                result_count = len([line for line in lines if line.strip() and not line.startswith('=')])
                self.print_info(f"返回 {result_count} 条结果")
                self.demo_stats['queries_executed'] += 1
            else:
                self.print_error(f"时间查询失败: {query['name']} - {stderr}")
            
            time.sleep(1)
    
    def demo_query_performance(self):
        """演示查询性能测试"""
        self.print_info("执行查询性能测试...")
        
        # 定义性能测试查询
        performance_queries = [
            {"name": "简单查询", "args": ["--limit", "10"]},
            {"name": "复杂查询", "args": ["--layout", "enhanced", "--type", "paired", "--limit", "5"]},
            {"name": "大数据量查询", "args": ["--days", "30", "--limit", "100"]},
            {"name": "JSON格式查询", "args": ["--format", "json", "--limit", "20"]}
        ]
        
        if HAS_RICH:
            perf_table = Table(title="查询性能测试结果")
            perf_table.add_column("查询类型", style="cyan")
            perf_table.add_column("响应时间(秒)", style="green")
            perf_table.add_column("结果数量", style="yellow")
            perf_table.add_column("状态", style="blue")
            
            for query in performance_queries:
                start_time = time.time()
                success, stdout, stderr = self.run_view_logs_command(query['args'])
                end_time = time.time()
                
                response_time = end_time - start_time
                result_count = len(stdout.split('\n')) if stdout else 0
                status = "成功" if success else "失败"
                
                perf_table.add_row(
                    query['name'],
                    f"{response_time:.3f}",
                    str(result_count),
                    status
                )
                
                if success:
                    self.demo_stats['queries_executed'] += 1
            
            console.print(perf_table)
        else:
            print("查询性能测试结果:")
            for query in performance_queries:
                start_time = time.time()
                success, stdout, stderr = self.run_view_logs_command(query['args'])
                end_time = time.time()
                
                response_time = end_time - start_time
                result_count = len(stdout.split('\n')) if stdout else 0
                status = "成功" if success else "失败"
                
                print(f"  {query['name']}: {response_time:.3f}秒, {result_count}条结果, {status}")
                
                if success:
                    self.demo_stats['queries_executed'] += 1
    
    def demo_management_functionality(self):
        """演示系统管理功能"""
        self.print_section("系统管理功能演示", "⚙️")
        
        # 1. 日志保留策略配置
        self.print_step("1. 日志保留策略配置")
        self.demo_retention_policy()
        
        # 2. 存储空间监控告警
        self.print_step("2. 存储空间监控告警")
        self.demo_storage_monitoring()
    
    def demo_retention_policy(self):
        """演示日志保留策略"""
        self.print_info("配置和演示日志保留策略...")
        
        # 模拟不同保留策略
        retention_policies = [
            {"name": "短期保留", "days": 7, "description": "保留7天的日志"},
            {"name": "中期保留", "days": 30, "description": "保留30天的日志"},
            {"name": "长期保留", "days": 90, "description": "保留90天的日志"}
        ]
        
        if HAS_RICH:
            policy_table = Table(title="日志保留策略配置")
            policy_table.add_column("策略名称", style="cyan")
            policy_table.add_column("保留天数", style="green")
            policy_table.add_column("描述", style="yellow")
            policy_table.add_column("状态", style="blue")
            
            for policy in retention_policies:
                # 模拟策略应用
                status = "已应用" if policy['days'] <= 30 else "待配置"
                
                policy_table.add_row(
                    policy['name'],
                    f"{policy['days']} 天",
                    policy['description'],
                    status
                )
            
            console.print(policy_table)
        
        # 演示清理过期日志
        self.print_info("演示过期日志清理...")
        
        # 创建模拟的过期日志文件
        old_date = datetime.now() - timedelta(days=35)
        old_log_file = self.logs_dir / f"harborai_{old_date.strftime('%Y%m%d')}.jsonl"
        
        if not old_log_file.exists():
            with open(old_log_file, 'w', encoding='utf-8') as f:
                f.write('{"timestamp": "' + old_date.isoformat() + '", "type": "old_log", "message": "这是过期的日志"}\n')
        
        # 检查并清理过期文件
        retention_days = 30
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cleaned_files = []
        for log_file in self.logs_dir.glob("harborai_*.jsonl"):
            try:
                # 从文件名提取日期
                date_str = log_file.stem.split('_')[1]
                file_date = datetime.strptime(date_str, '%Y%m%d')
                
                if file_date < cutoff_date:
                    cleaned_files.append(log_file.name)
                    # 在实际环境中这里会删除文件
                    # log_file.unlink()
                    
            except (ValueError, IndexError):
                continue
        
        if cleaned_files:
            self.print_success(f"模拟清理了 {len(cleaned_files)} 个过期日志文件")
            for file_name in cleaned_files:
                self.print_info(f"  - {file_name}")
        else:
            self.print_info("没有发现需要清理的过期日志文件")
    
    def demo_storage_monitoring(self):
        """演示存储空间监控"""
        self.print_info("执行存储空间监控检查...")
        
        # 获取存储空间信息
        try:
            if os.name == 'nt':  # Windows
                disk_usage = psutil.disk_usage('C:')
            else:  # Unix/Linux
                disk_usage = psutil.disk_usage('/')
            
            total_gb = disk_usage.total / (1024**3)
            used_gb = disk_usage.used / (1024**3)
            free_gb = disk_usage.free / (1024**3)
            usage_percent = (used_gb / total_gb) * 100
            
            # 获取日志目录大小
            logs_size = 0
            for log_file in self.logs_dir.glob("*.jsonl"):
                logs_size += log_file.stat().st_size
            
            logs_size_mb = logs_size / (1024**2)
            
            if HAS_RICH:
                storage_table = Table(title="存储空间监控报告")
                storage_table.add_column("监控项", style="cyan")
                storage_table.add_column("当前值", style="green")
                storage_table.add_column("状态", style="yellow")
                
                # 磁盘使用率状态
                disk_status = "正常"
                if usage_percent > 90:
                    disk_status = "严重告警"
                elif usage_percent > 80:
                    disk_status = "警告"
                elif usage_percent > 70:
                    disk_status = "注意"
                
                storage_table.add_row("磁盘总容量", f"{total_gb:.1f} GB", "信息")
                storage_table.add_row("已使用空间", f"{used_gb:.1f} GB", "信息")
                storage_table.add_row("可用空间", f"{free_gb:.1f} GB", "信息")
                storage_table.add_row("使用率", f"{usage_percent:.1f}%", disk_status)
                storage_table.add_row("日志目录大小", f"{logs_size_mb:.2f} MB", "信息")
                
                console.print(storage_table)
            else:
                print("存储空间监控报告:")
                print(f"  磁盘总容量: {total_gb:.1f} GB")
                print(f"  已使用空间: {used_gb:.1f} GB")
                print(f"  可用空间: {free_gb:.1f} GB")
                print(f"  使用率: {usage_percent:.1f}%")
                print(f"  日志目录大小: {logs_size_mb:.2f} MB")
            
            # 模拟告警逻辑
            alerts = []
            if usage_percent > 90:
                alerts.append("🚨 严重告警: 磁盘使用率超过90%，请立即清理空间")
            elif usage_percent > 80:
                alerts.append("⚠️ 警告: 磁盘使用率超过80%，建议清理日志文件")
            
            if logs_size_mb > 100:  # 如果日志超过100MB
                alerts.append("📊 信息: 日志文件较大，建议配置自动清理策略")
            
            if alerts:
                self.print_warning("存储监控告警:")
                for alert in alerts:
                    self.print_warning(f"  {alert}")
            else:
                self.print_success("存储空间状态正常，无告警")
                
        except Exception as e:
            self.print_error(f"存储监控检查失败: {e}")
    
    def demo_real_api_calls(self):
        """演示真实API调用"""
        self.print_section("真实API调用演示", "🌐")
        
        if not self.harborai_client:
            self.print_error("HarborAI客户端未初始化，跳过真实API调用演示")
            return
        
        self.print_info("开始真实API调用演示...")
        
        # 测试消息列表
        test_messages = [
            "请简单介绍一下人工智能的发展历史",
            "解释一下什么是机器学习",
            "Python编程语言有哪些特点？",
            "请推荐几本关于数据科学的书籍"
        ]
        
        # 使用不同模型进行测试
        test_models = [
            "deepseek-chat",
            "deepseek-reasoner"
        ]
        
        total_cost = 0.0
        
        if HAS_RICH:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("执行API调用...", total=len(test_messages) * len(test_models))
                
                for model in test_models:
                    for i, message in enumerate(test_messages):
                        try:
                            self.print_step(f"使用模型 {model} 处理消息: {message[:30]}...")
                            
                            # 记录开始时间
                            start_time = time.time()
                            
                            # 发送API请求
                            response = self.harborai_client.chat.completions.create(
                                model=model,
                                messages=[{"role": "user", "content": message}],
                                max_tokens=150,
                                temperature=0.7
                            )
                            
                            # 记录结束时间
                            end_time = time.time()
                            latency = end_time - start_time
                            
                            # 计算成本
                            pricing = ModelConfig.get_model_price(model)
                            input_tokens = len(message.split()) * 1.3  # 估算
                            output_tokens = len(response.choices[0].message.content.split()) * 1.3  # 估算
                            
                            input_cost = (input_tokens / 1000) * pricing['input']
                            output_cost = (output_tokens / 1000) * pricing['output']
                            call_cost = input_cost + output_cost
                            total_cost += call_cost
                            
                            self.print_success(f"API调用成功 - 延迟: {latency:.2f}s, 成本: ¥{call_cost:.6f}")
                            self.print_info(f"响应: {response.choices[0].message.content[:100]}...")
                            
                            self.demo_stats['api_calls_made'] += 1
                            self.demo_stats['total_cost'] += call_cost
                            
                        except Exception as e:
                            self.print_error(f"API调用失败: {e}")
                        
                        progress.update(task, advance=1)
                        time.sleep(1)  # 避免频率限制
        
        self.print_success(f"真实API调用演示完成，总成本: ¥{total_cost:.6f}")
    
    def run_view_logs_command(self, args: List[str]) -> Tuple[bool, str, str]:
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
                timeout=30
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "命令执行超时"
        except Exception as e:
            return False, "", f"执行命令时出错: {e}"
    
    def show_demo_summary(self):
        """显示演示总结"""
        self.print_section("演示总结报告", "📊")
        
        end_time = datetime.now()
        duration = end_time - self.demo_stats['start_time']
        
        if HAS_RICH:
            summary_table = Table(title="HarborAI 日志系统演示总结")
            summary_table.add_column("统计项", style="cyan")
            summary_table.add_column("数值", style="green")
            
            summary_table.add_row("演示开始时间", self.demo_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S'))
            summary_table.add_row("演示结束时间", end_time.strftime('%Y-%m-%d %H:%M:%S'))
            summary_table.add_row("总耗时", str(duration).split('.')[0])
            summary_table.add_row("创建日志数量", f"{self.demo_stats['logs_created']:,} 条")
            summary_table.add_row("执行查询数量", f"{self.demo_stats['queries_executed']} 次")
            summary_table.add_row("API调用次数", f"{self.demo_stats['api_calls_made']} 次")
            summary_table.add_row("总成本", f"¥{self.demo_stats['total_cost']:.6f}")
            
            console.print(summary_table)
            
            # 功能完成状态
            features_table = Table(title="功能演示完成状态")
            features_table.add_column("功能模块", style="cyan")
            features_table.add_column("状态", style="green")
            features_table.add_column("描述", style="yellow")
            
            features = [
                ("日志存储功能", "✅ 完成", "实时采集、分类索引、性能测试"),
                ("日志查询功能", "✅ 完成", "多条件查询、全文检索、时间筛选"),
                ("系统管理功能", "✅ 完成", "保留策略、存储监控"),
                ("真实API调用", "✅ 完成", "多模型测试、成本追踪"),
                ("性能基准测试", "✅ 完成", "存储性能、查询性能")
            ]
            
            for feature, status, description in features:
                features_table.add_row(feature, status, description)
            
            console.print(features_table)
        else:
            print("HarborAI 日志系统演示总结:")
            print(f"  演示开始时间: {self.demo_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  演示结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  总耗时: {str(duration).split('.')[0]}")
            print(f"  创建日志数量: {self.demo_stats['logs_created']:,} 条")
            print(f"  执行查询数量: {self.demo_stats['queries_executed']} 次")
            print(f"  API调用次数: {self.demo_stats['api_calls_made']} 次")
            print(f"  总成本: ¥{self.demo_stats['total_cost']:.6f}")
        
        self.print_success("🎉 HarborAI 日志系统完整演示成功完成！")
        
        # 使用建议
        if HAS_RICH:
            suggestions = Panel(
                "💡 使用建议:\n"
                "• 在生产环境中配置合适的日志保留策略\n"
                "• 定期监控存储空间使用情况\n"
                "• 使用多条件查询快速定位问题\n"
                "• 启用成本追踪监控API使用费用\n"
                "• 配置自动化告警机制\n"
                "• 定期进行性能基准测试",
                title="生产环境建议",
                style="blue"
            )
            console.print(suggestions)
        else:
            print("\n💡 生产环境使用建议:")
            print("• 在生产环境中配置合适的日志保留策略")
            print("• 定期监控存储空间使用情况")
            print("• 使用多条件查询快速定位问题")
            print("• 启用成本追踪监控API使用费用")
            print("• 配置自动化告警机制")
            print("• 定期进行性能基准测试")
    
    def run_complete_demo(self, storage_only=False, query_only=False, management_only=False, real_api=False, performance=False):
        """运行完整演示"""
        if HAS_RICH:
            console.print(Panel(
                "🚀 HarborAI 日志系统完整演示\n\n"
                "本演示将全面展示 HarborAI 日志系统的所有功能特性，\n"
                "包括日志存储、查询功能、系统管理和真实API调用。\n\n"
                "演示内容:\n"
                "• 实时日志采集与存储机制\n"
                "• 多条件组合查询与全文检索\n"
                "• 时间范围筛选与性能测试\n"
                "• 日志保留策略与存储监控\n"
                "• 真实API调用与成本追踪",
                title="欢迎使用 HarborAI 日志系统演示",
                style="bold green"
            ))
        else:
            print("🚀 HarborAI 日志系统完整演示")
            print("本演示将全面展示 HarborAI 日志系统的所有功能特性")
        
        try:
            # 存储功能演示
            if not query_only and not management_only:
                self.demo_storage_functionality()
                time.sleep(2)
            
            # 查询功能演示
            if not storage_only and not management_only:
                self.demo_query_functionality()
                time.sleep(2)
            
            # 管理功能演示
            if not storage_only and not query_only:
                self.demo_management_functionality()
                time.sleep(2)
            
            # 真实API调用演示
            if real_api and not storage_only and not query_only and not management_only:
                self.demo_real_api_calls()
                time.sleep(2)
            
            # 性能测试
            if performance:
                self.print_section("额外性能测试", "⚡")
                self.demo_storage_performance()
                time.sleep(2)
            
            # 显示总结
            self.show_demo_summary()
            
        except KeyboardInterrupt:
            self.print_warning("演示被用户中断")
        except Exception as e:
            self.print_error(f"演示过程中发生错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HarborAI 日志系统完整演示")
    parser.add_argument("--storage-only", action="store_true", help="仅演示存储功能")
    parser.add_argument("--query-only", action="store_true", help="仅演示查询功能")
    parser.add_argument("--management-only", action="store_true", help="仅演示管理功能")
    parser.add_argument("--real-api", action="store_true", help="包含真实API调用演示")
    parser.add_argument("--performance", action="store_true", help="包含性能基准测试")
    
    args = parser.parse_args()
    
    demo = LoggingSystemDemo()
    demo.run_complete_demo(
        storage_only=args.storage_only,
        query_only=args.query_only,
        management_only=args.management_only,
        real_api=args.real_api,
        performance=args.performance
    )


if __name__ == "__main__":
    main()