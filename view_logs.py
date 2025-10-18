#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 日志查看工具

这个脚本提供了便捷的日志查看功能，支持：
1. 查看PostgreSQL日志
2. 查看文件日志
3. 统一的日志查看接口
4. 多种过滤和格式化选项

使用方法：
    python view_logs.py                    # 查看最近的日志
    python view_logs.py --days 7           # 查看最近7天的日志
    python view_logs.py --model gpt-4      # 查看特定模型的日志
    python view_logs.py --provider openai  # 查看特定提供商的日志
    python view_logs.py --source file      # 强制查看文件日志
    python view_logs.py --source postgres  # 强制查看PostgreSQL日志
    python view_logs.py --format json      # JSON格式输出
    python view_logs.py --stats            # 显示统计信息

注意：
- 自动检测可用的日志源
- 优先使用PostgreSQL，降级到文件日志
- 支持丰富的过滤和格式化选项
"""

import argparse
import json
import sys
import os
import re
import time
import threading
import configparser
import csv
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# 导出功能相关依赖
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

# Windows环境下的编码设置
if os.name == 'nt':  # Windows
    # 设置控制台代码页为UTF-8
    try:
        import subprocess
        subprocess.run(['chcp', '65001'], shell=True, capture_output=True)
    except:
        pass
    
    # 设置环境变量
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# HarborAI 导入
from harborai.database.postgres_client import get_postgres_client
from harborai.database.file_log_parser import FileLogParser
from harborai.config.settings import get_settings
from harborai.storage.file_logger import FileSystemLogger
from harborai.utils.logger import get_logger

# 尝试导入 PostgreSQL 相关模块
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# 用于美化输出
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.syntax import Syntax
    from rich.live import Live
    from rich.layout import Layout
    from rich.text import Text
    
    # 在Windows环境下配置控制台以支持UTF-8编码
    if os.name == 'nt':  # Windows
        # 设置控制台编码为UTF-8
        import codecs
        import locale
        
        # 尝试设置控制台编码
        try:
            # 设置标准输出编码
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            if hasattr(sys.stderr, 'reconfigure'):
                sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass
        
        # 创建支持UTF-8的控制台
        console = Console(
            force_terminal=True,
            legacy_windows=False,
            width=120,
            file=sys.stdout
        )
    else:
        console = Console()
    
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self._get_default_config_file()
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def _get_default_config_file(self) -> str:
        """获取默认配置文件路径"""
        return os.path.join(os.path.expanduser("~"), ".harborai_view_logs.ini")
    
    def load_config(self):
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                self.config.read(self.config_file, encoding='utf-8')
            except Exception as e:
                print(f"[WARNING] 加载配置文件失败: {e}")
    
    def save_config(self):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)
        except Exception as e:
            print(f"[WARNING] 保存配置文件失败: {e}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """获取配置值"""
        try:
            return self.config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def set(self, section: str, key: str, value: Any):
        """设置配置值"""
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))


class DemoDataGenerator:
    """演示数据生成器"""
    
    @staticmethod
    def generate_sample_logs(count: int = 50) -> List[Dict[str, Any]]:
        """生成示例日志数据"""
        import random
        import uuid
        
        providers = ['openai', 'anthropic', 'google', 'baidu', 'deepseek']
        models = {
            'openai': ['gpt-4', 'gpt-3.5-turbo', 'gpt-4-turbo'],
            'anthropic': ['claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'],
            'google': ['gemini-pro', 'gemini-pro-vision'],
            'baidu': ['ernie-bot', 'ernie-bot-turbo'],
            'deepseek': ['deepseek-chat', 'deepseek-coder']
        }
        
        # 模型价格表 (每1K tokens的价格，单位：CNY)
        model_pricing = {
            'gpt-4': {'input': 0.21, 'output': 0.42},
            'gpt-3.5-turbo': {'input': 0.007, 'output': 0.014},
            'gpt-4-turbo': {'input': 0.07, 'output': 0.21},
            'claude-3-opus': {'input': 0.105, 'output': 0.525},
            'claude-3-sonnet': {'input': 0.021, 'output': 0.105},
            'claude-3-haiku': {'input': 0.0014, 'output': 0.007},
            'gemini-pro': {'input': 0.0035, 'output': 0.0105},
            'gemini-pro-vision': {'input': 0.0035, 'output': 0.0105},
            'ernie-bot': {'input': 0.008, 'output': 0.016},
            'ernie-bot-turbo': {'input': 0.004, 'output': 0.008},
            'deepseek-chat': {'input': 0.001, 'output': 0.002},
            'deepseek-coder': {'input': 0.001, 'output': 0.002}
        }
        
        logs = []
        base_time = datetime.now()
        
        for i in range(count):
            provider = random.choice(providers)
            model = random.choice(models[provider])
            trace_id = f"hb_{int((base_time - timedelta(minutes=random.randint(0, 10080))).timestamp() * 1000)}_{random.randint(10000000, 99999999):08x}"
            
            # 生成请求日志
            request_log = {
                'id': str(uuid.uuid4()),
                'hb_trace_id': trace_id,  # 使用设计文档中的字段名
                'trace_id': trace_id,  # 保持向后兼容性
                'timestamp': (base_time - timedelta(minutes=random.randint(0, 10080))).isoformat(),
                'type': 'request',
                'model': model,
                'provider': provider,
                'request_data': {
                    'messages': [{'role': 'user', 'content': f'示例请求 {i+1}'}],
                    'max_tokens': random.randint(100, 2000),
                    'temperature': round(random.uniform(0.1, 1.0), 2)
                },
                'response_data': None,
                'error_message': None,
                'success': None,  # 请求日志没有成功状态
                'created_at': (base_time - timedelta(minutes=random.randint(0, 10080))).isoformat(),
                'source': 'demo'
            }
            logs.append(request_log)
            
            # 生成对应的响应日志
            success = random.random() > 0.1  # 90% 成功率
            
            # 生成Token使用量
            prompt_tokens = random.randint(10, 100)
            completion_tokens = random.randint(20, 200)
            total_tokens = prompt_tokens + completion_tokens
            
            # 计算成本
            cost = 0.0
            if success and model in model_pricing:
                pricing = model_pricing[model]
                cost = (prompt_tokens / 1000 * pricing['input']) + (completion_tokens / 1000 * pricing['output'])
            
            response_log = {
                'id': str(uuid.uuid4()),
                'hb_trace_id': trace_id,  # 使用设计文档中的字段名
                'trace_id': trace_id,  # 保持向后兼容性
                'timestamp': (base_time - timedelta(minutes=random.randint(0, 10080))).isoformat(),
                'type': 'response',
                'model': model,
                'provider': provider,
                'request_data': None,
                'response_data': {
                    'choices': [{'message': {'content': f'示例响应 {i+1}'}}],
                    'usage': {
                        'prompt_tokens': prompt_tokens,
                        'completion_tokens': completion_tokens,
                        'total_tokens': total_tokens
                    }
                } if success else None,
                'error_message': f'示例错误 {i+1}' if not success else None,
                'success': success,  # 明确设置成功状态
                'total_tokens': total_tokens if success else 0,  # 按设计文档添加顶级字段
                'total_cost': cost if success else 0.0,  # 按设计文档添加顶级字段
                'pricing_source': 'estimated' if success else 'unknown',  # 价格来源
                'created_at': (base_time - timedelta(minutes=random.randint(0, 10080))).isoformat(),
                'source': 'demo'
            }
            logs.append(response_log)
        
        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)


def infer_provider_from_model(model: str) -> str:
    """从模型名称推断提供商
    
    Args:
        model: 模型名称
        
    Returns:
        推断的提供商名称
    """
    if not model:
        return 'unknown'
    
    model_lower = model.lower()
    
    # 根据模型名称推断真实的提供商
    if 'ernie' in model_lower:
        return 'baidu'
    elif 'doubao' in model_lower:
        return 'bytedance'
    elif 'deepseek' in model_lower:
        return 'deepseek'
    elif 'gpt' in model_lower or 'openai' in model_lower:
        return 'openai'
    elif 'claude' in model_lower:
        return 'anthropic'
    elif 'gemini' in model_lower:
        return 'google'
    else:
        return 'unknown'


def infer_log_type(log: Dict[str, Any]) -> str:
    """智能推断日志类型
    
    Args:
        log: 日志数据
        
    Returns:
        推断的日志类型: 'request', 'response', 或 'unknown'
    """
    # 首先检查显式的 type 字段
    log_type = log.get('type', '').lower()
    if log_type in ['request', 'response']:
        return log_type
    
    # 基于日志内容智能推断
    has_request_data = bool(log.get('request_data'))
    has_response_data = bool(log.get('response_data'))
    has_success = log.get('success') is not None
    has_duration = log.get('duration_ms') is not None
    has_tokens = bool(log.get('tokens') or log.get('total_tokens'))
    has_cost = bool(log.get('total_cost') or log.get('estimated_cost') or log.get('cost'))
    
    # 推断逻辑：
    # 1. 如果有 response_data 或者有 success/duration/tokens/cost，很可能是响应日志
    if has_response_data or has_success or has_duration or has_tokens or has_cost:
        return 'response'
    
    # 2. 如果只有 request_data 但没有响应相关字段，很可能是请求日志
    if has_request_data and not (has_response_data or has_success or has_duration):
        return 'request'
    
    # 3. 根据消息内容判断
    message = log.get('message', '').lower()
    if 'request' in message and 'completed' not in message:
        return 'request'
    elif 'completed' in message or 'response' in message:
        return 'response'
    
    # 4. 默认返回 unknown
    return 'unknown'


class LogViewer:
    """日志查看器类 - 增强版"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.postgres_client = None
        self.file_parser = None
        self.file_logger = None
        self.postgres_conn = None
        self._watch_active = False
        self._init_clients()
        self._init_trace_id_support()
    
    def _init_clients(self):
        """初始化客户端"""
        try:
            self.postgres_client = get_postgres_client()
        except Exception as e:
            self._print_debug(f"PostgreSQL客户端初始化失败: {e}")
        
        try:
            log_dir = self._get_log_directory()
            self.file_parser = FileLogParser(str(log_dir))
        except Exception as e:
            self._print_debug(f"文件日志解析器初始化失败: {e}")
    
    def _get_log_directory(self) -> Path:
        """获取日志目录路径"""
        try:
            settings = get_settings()
            if hasattr(settings, 'file_log_directory'):
                return Path(settings.file_log_directory)
        except Exception:
            pass
        
        # 从配置文件获取
        config_log_dir = self.config_manager.get('paths', 'log_directory')
        if config_log_dir:
            return Path(config_log_dir)
        
        # 默认日志目录（项目根目录下的logs文件夹）
        return Path("./logs")
    
    def _init_trace_id_support(self):
        """初始化 trace_id 查询支持"""
        # 初始化文件日志系统（用于 trace_id 查询）
        self._init_file_logger_for_trace_id()
        
        # 初始化 PostgreSQL 连接（用于 trace_id 查询）
        self._init_postgres_connection_for_trace_id()
    
    def _init_file_logger_for_trace_id(self):
        """初始化文件日志系统（用于 trace_id 查询）"""
        try:
            # 获取日志目录
            log_dir = self._get_trace_id_log_directory()
            if log_dir and log_dir.exists():
                self.file_logger = FileSystemLogger(log_dir=str(log_dir))
        except Exception as e:
            self._print_debug(f"文件日志系统初始化失败: {e}")
    
    def _init_postgres_connection_for_trace_id(self):
        """初始化 PostgreSQL 连接（用于 trace_id 查询）"""
        if not POSTGRES_AVAILABLE:
            return
        
        try:
            # 构建连接字符串
            conn_str = self._build_postgres_connection_string()
            if conn_str:
                self.postgres_conn = psycopg2.connect(conn_str)
        except Exception as e:
            self._print_debug(f"PostgreSQL连接初始化失败: {e}")
    
    def _get_trace_id_log_directory(self) -> Optional[Path]:
        """获取 trace_id 查询用的日志目录路径"""
        # 尝试多个可能的日志目录位置
        possible_dirs = [
            Path("logs"),
            Path("harborai_logs"),
            Path.home() / ".harborai" / "logs",
            Path("/tmp/harborai_logs"),
        ]
        
        for log_dir in possible_dirs:
            if log_dir.exists():
                return log_dir
        
        # 如果都不存在，返回默认的 logs 目录
        return Path("logs")
    
    def _build_postgres_connection_string(self) -> Optional[str]:
        """构建 PostgreSQL 连接字符串"""
        try:
            # 从环境变量或配置获取 PostgreSQL 连接信息
            host = os.getenv('POSTGRES_HOST', 'localhost')
            port = os.getenv('POSTGRES_PORT', '5432')
            database = os.getenv('POSTGRES_DB', 'harborai')
            user = os.getenv('POSTGRES_USER', 'harborai')
            password = os.getenv('POSTGRES_PASSWORD', '')
            
            if not password:
                return None
            
            # 使用标准的 PostgreSQL URL 格式
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        except Exception:
            return None
    
    def _print_info(self, message: str):
        """打印信息"""
        if HAS_RICH and console:
            console.print(f"[cyan][INFO] {message}[/cyan]")
        else:
            print(f"[INFO] {message}")
    
    def _print_success(self, message: str):
        """打印成功信息"""
        if HAS_RICH and console:
            console.print(f"[green][SUCCESS] {message}[/green]")
        else:
            print(f"[SUCCESS] {message}")
    
    def _print_warning(self, message: str):
        """打印警告信息"""
        if HAS_RICH and console:
            console.print(f"[yellow][WARNING] {message}[/yellow]")
        else:
            print(f"[WARNING] {message}")
    
    def _print_error(self, message: str):
        """打印错误信息"""
        if HAS_RICH and console:
            console.print(f"[red][ERROR] {message}[/red]")
        else:
            print(f"[ERROR] {message}")
    
    def _print_debug(self, message: str):
        """打印调试信息（仅在调试模式下显示）"""
        debug_mode = self.config_manager.get('general', 'debug_mode', 'false').lower() == 'true'
        if debug_mode:
            if HAS_RICH and console:
                console.print(f"[dim][DEBUG] {message}[/dim]")
            else:
                print(f"[DEBUG] {message}")
    
    def parse_time_string(self, time_str: str) -> Optional[datetime]:
        """解析时间字符串，支持多种格式
        
        支持的格式：
        - ISO格式: 2024-01-15T10:30:00 或 2024-01-15T10:30:00Z
        - 标准格式: 2024-01-15 10:30:00
        - 简化格式: 01-15 10:30 (当年)
        - 仅时间: 10:30 (当天)
        
        Args:
            time_str: 时间字符串
            
        Returns:
            解析后的datetime对象，失败返回None
        """
        if not time_str:
            return None
        
        time_str = time_str.strip()
        now = datetime.now()
        
        # 定义支持的时间格式
        time_formats = [
            # ISO格式
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            # 标准格式
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M',
            # 简化格式（当年）
            '%m-%d %H:%M:%S',
            '%m-%d %H:%M',
            # 仅时间（当天）
            '%H:%M:%S',
            '%H:%M',
        ]
        
        for fmt in time_formats:
            try:
                parsed_time = datetime.strptime(time_str, fmt)
                
                # 对于简化格式，补充年份
                if fmt.startswith('%m-%d'):
                    parsed_time = parsed_time.replace(year=now.year)
                # 对于仅时间格式，补充日期
                elif fmt.startswith('%H:%M'):
                    parsed_time = parsed_time.replace(
                        year=now.year,
                        month=now.month,
                        day=now.day
                    )
                
                return parsed_time
                
            except ValueError:
                continue
        
        # 尝试使用 fromisoformat（Python 3.7+）
        try:
            # 处理带时区的ISO格式
            if time_str.endswith('Z'):
                time_str = time_str[:-1] + '+00:00'
            return datetime.fromisoformat(time_str)
        except (ValueError, AttributeError):
            pass
        
        return None
    
    def validate_time_range(self, start_time: Optional[datetime], end_time: Optional[datetime]) -> Tuple[bool, str]:
        """验证时间范围的有效性
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            (是否有效, 错误信息)
        """
        if start_time is None and end_time is None:
            return True, ""
        
        if start_time and end_time:
            if start_time >= end_time:
                return False, "开始时间不能晚于或等于结束时间"
        
        # 检查时间是否过于久远或未来
        now = datetime.now()
        max_past = now - timedelta(days=365 * 2)  # 最多2年前
        max_future = now + timedelta(days=1)  # 最多1天后
        
        if start_time:
            if start_time < max_past:
                return False, f"开始时间过于久远（超过2年前）"
            if start_time > max_future:
                return False, f"开始时间不能是未来时间"
        
        if end_time:
            if end_time < max_past:
                return False, f"结束时间过于久远（超过2年前）"
            if end_time > max_future:
                return False, f"结束时间不能是未来时间"
        
        return True, ""
    
    def check_postgres_availability(self) -> bool:
        """检查PostgreSQL可用性"""
        if not self.postgres_client:
            return False
        
        try:
            result = self.postgres_client.query_api_logs(days=1, limit=1)
            return result.source == "postgresql"
        except Exception:
            return False
    
    def check_file_logs_availability(self) -> bool:
        """检查文件日志可用性"""
        if not self.file_parser:
            return False
        
        log_dir = self._get_log_directory()
        return log_dir.exists() and any(log_dir.glob("**/*.jsonl"))
    
    def get_postgres_logs(self, days: int = 7, model: Optional[str] = None, 
                         provider: Optional[str] = None, limit: int = 50, 
                         log_type: str = "response", start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取PostgreSQL日志
        
        Args:
            days: 查询最近几天的日志（当start_time和end_time都为None时使用）
            model: 过滤特定模型
            provider: 过滤特定提供商
            limit: 限制返回条数
            log_type: 日志类型
            start_time: 开始时间（优先级高于days参数）
            end_time: 结束时间
        """
        if not self.postgres_client:
            return {"error": "PostgreSQL客户端不可用", "data": [], "source": "error"}
        
        try:
            # 如果指定了时间范围，使用自定义查询
            if start_time or end_time:
                return self._query_postgres_with_time_range(
                    start_time=start_time,
                    end_time=end_time,
                    model=model,
                    provider=provider,
                    limit=limit,
                    log_type=log_type
                )
            else:
                # 使用原有的days参数查询
                result = self.postgres_client.query_api_logs(
                    days=days,
                    model=model,
                    provider=provider,
                    limit=limit
                )
                return {
                    "error": result.error,
                    "data": result.data,
                    "source": result.source,
                    "total_count": getattr(result, 'total_count', len(result.data))
                }
        except Exception as e:
            return {"error": str(e), "data": [], "source": "error"}
    
    def _query_postgres_with_time_range(self, start_time: Optional[datetime] = None,
                                       end_time: Optional[datetime] = None,
                                       model: Optional[str] = None,
                                       provider: Optional[str] = None,
                                       limit: int = 50,
                                       log_type: str = "response") -> Dict[str, Any]:
        """使用时间范围查询PostgreSQL日志"""
        if not self.postgres_conn:
            return {"error": "PostgreSQL连接不可用", "data": [], "source": "error"}
        
        try:
            # 构建查询条件
            where_conditions = []
            params = []
            
            if start_time:
                where_conditions.append("timestamp >= %s")
                params.append(start_time)
            
            if end_time:
                where_conditions.append("timestamp <= %s")
                params.append(end_time)
            
            if model:
                where_conditions.append("model = %s")
                params.append(model)
            
            if provider:
                where_conditions.append("provider = %s")
                params.append(provider)
            
            where_clause = ""
            if where_conditions:
                where_clause = "WHERE " + " AND ".join(where_conditions)
            
            # 构建查询语句
            query = f"""
                SELECT 
                    id, timestamp, provider, model, 
                    request_data, response_data, status_code,
                    error_message, duration_ms, created_at, updated_at
                FROM api_logs 
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT %s
            """
            params.append(limit)
            
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                logs = []
                for row in rows:
                    log_entry = {
                        "id": row["id"],
                        "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                        "provider": row["provider"],
                        "model": row["model"],
                        "request_data": row["request_data"],
                        "response_data": row["response_data"],
                        "status_code": row["status_code"],
                        "error_message": row["error_message"],
                        "duration_ms": row["duration_ms"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                        "type": log_type,
                        "source": "postgresql"
                    }
                    logs.append(log_entry)
                
                return {
                    "error": None,
                    "data": logs,
                    "source": "postgresql",
                    "total_count": len(logs)
                }
                
        except Exception as e:
            self._print_debug(f"PostgreSQL时间范围查询失败: {e}")
            return {"error": str(e), "data": [], "source": "error"}
    
    def get_file_logs(self, days: int = 7, model: Optional[str] = None, 
                     provider: Optional[str] = None, limit: int = 50, 
                     log_type: str = "response", start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取文件日志"""
        if not self.file_parser:
            return {"error": "文件日志解析器不可用", "data": [], "source": "error"}
        
        try:
            # 如果指定了时间范围，使用时间范围查询
            if start_time or end_time:
                return self._query_file_logs_with_time_range(
                    start_time=start_time,
                    end_time=end_time,
                    model=model,
                    provider=provider,
                    limit=limit,
                    log_type=log_type
                )
            else:
                # 使用原有的days参数查询
                result = self.file_parser.query_api_logs(
                    days=days,
                    model=model,
                    provider=provider,
                    limit=limit,
                    log_type=log_type
                )
                return {
                    "error": result.error,
                    "data": result.data,
                    "source": result.source,
                    "total_count": result.total_count
                }
        except Exception as e:
            return {"error": str(e), "data": [], "source": "error"}
    
    def _query_file_logs_with_time_range(self, start_time: Optional[datetime] = None,
                                        end_time: Optional[datetime] = None,
                                        model: Optional[str] = None,
                                        provider: Optional[str] = None,
                                        limit: int = 50,
                                        log_type: str = "response") -> Dict[str, Any]:
        """使用时间范围查询文件日志"""
        try:
            # 检查是否有FileSystemLogger可用
            if hasattr(self, 'file_logger') and self.file_logger:
                # 使用FileSystemLogger的read_logs方法，它支持start_time和end_time
                logs = self.file_logger.read_logs(
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit
                )
                
                # 过滤日志
                filtered_logs = []
                for log in logs:
                    # 检查模型过滤
                    if model and log.get('model') != model:
                        continue
                    
                    # 检查提供商过滤
                    if provider and log.get('provider') != provider:
                        continue
                    
                    # 添加source和type信息
                    log['source'] = 'file'
                    log['type'] = log_type
                    filtered_logs.append(log)
                
                return {
                    "error": None,
                    "data": filtered_logs[:limit],
                    "source": "file",
                    "total_count": len(filtered_logs)
                }
            else:
                # 回退到使用file_parser的方式，但需要手动过滤时间
                # 先获取足够多的日志，然后手动过滤时间范围
                days_range = 30  # 获取30天的日志进行时间过滤
                result = self.file_parser.query_api_logs(
                    days=days_range,
                    model=model,
                    provider=provider,
                    limit=limit * 5,  # 获取更多日志以便过滤
                    log_type=log_type
                )
                
                if result.error:
                    return {
                        "error": result.error,
                        "data": [],
                        "source": "error",
                        "total_count": 0
                    }
                
                # 手动过滤时间范围
                filtered_logs = []
                for log in result.data:
                    log_time = None
                    if 'timestamp' in log and log['timestamp']:
                        try:
                            if isinstance(log['timestamp'], str):
                                log_time = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                            elif isinstance(log['timestamp'], datetime):
                                log_time = log['timestamp']
                        except:
                            continue
                    
                    if log_time:
                        # 检查时间范围
                        if start_time and log_time < start_time:
                            continue
                        if end_time and log_time > end_time:
                            continue
                    
                    filtered_logs.append(log)
                    
                    if len(filtered_logs) >= limit:
                        break
                
                return {
                    "error": None,
                    "data": filtered_logs,
                    "source": result.source,
                    "total_count": len(filtered_logs)
                }
                
        except Exception as e:
            self._print_debug(f"文件日志时间范围查询失败: {e}")
            return {"error": str(e), "data": [], "source": "error"}
    
    def get_logs(self, source: Optional[str] = None, start_time: Optional[datetime] = None, 
                 end_time: Optional[datetime] = None, **kwargs) -> Dict[str, Any]:
        """获取日志（自动选择源或指定源）
        
        Args:
            source: 指定数据源 ("postgres", "file", None)
            start_time: 开始时间
            end_time: 结束时间
            **kwargs: 其他参数
        """
        # 验证时间范围
        if start_time or end_time:
            is_valid, error_msg = self.validate_time_range(start_time, end_time)
            if not is_valid:
                return {"error": f"时间范围无效: {error_msg}", "data": [], "source": "error"}
        
        if source == "postgres":
            return self.get_postgres_logs(start_time=start_time, end_time=end_time, **kwargs)
        elif source == "file":
            return self.get_file_logs(start_time=start_time, end_time=end_time, **kwargs)
        else:
            # 自动选择：优先PostgreSQL，降级到文件
            if self.check_postgres_availability():
                return self.get_postgres_logs(start_time=start_time, end_time=end_time, **kwargs)
            elif self.check_file_logs_availability():
                return self.get_file_logs(start_time=start_time, end_time=end_time, **kwargs)
            else:
                return {"error": "没有可用的日志源", "data": [], "source": "error"}
    
    def get_stats(self, days: int = 30, provider: Optional[str] = None, 
                 model: Optional[str] = None) -> Dict[str, Any]:
        """获取统计信息"""
        if self.postgres_client:
            try:
                result = self.postgres_client.query_model_usage(
                    days=days,
                    provider=provider,
                    model=model
                )
                return {
                    "error": result.error,
                    "data": result.data,
                    "source": result.source
                }
            except Exception as e:
                self._print_warning(f"PostgreSQL统计查询失败: {e}")
        
        # 从文件日志计算统计（简化版）
        if self.file_parser:
            try:
                result = self.file_parser.query_model_usage(days=days, model=model, provider=provider)
                return {
                    "error": result.error,
                    "data": result.data,
                    "source": result.source
                }
            except Exception as e:
                return {"error": str(e), "data": [], "source": "error"}
        
        return {"error": "没有可用的统计源", "data": [], "source": "error"}
    
    def get_log_type_stats(self, days: int = 7, **kwargs) -> Dict[str, Any]:
        """获取日志类型统计信息"""
        # 获取所有类型的日志
        logs_result = self.get_logs(log_type="all", days=days, limit=1000, **kwargs)
        
        if logs_result.get("error"):
            return {"error": logs_result["error"], "data": {}, "source": logs_result["source"]}
        
        logs = logs_result.get("data", [])
        
        # 统计各种类型的日志数量
        type_stats = {
            "request": 0,
            "response": 0,
            "unknown": 0,
            "total": len(logs)
        }
        
        for log in logs:
            log_type = log.get('type', 'unknown')
            if log_type in type_stats:
                type_stats[log_type] += 1
            else:
                type_stats['unknown'] += 1
        
        return {
            "error": None,
            "data": type_stats,
            "source": logs_result["source"]
        }
    
    def search_logs(self, logs: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
        """搜索日志"""
        if not search_term:
            return logs
        
        search_term_lower = search_term.lower()
        filtered_logs = []
        
        for log in logs:
            # 搜索多个字段
            searchable_text = ""
            
            # 添加基本字段
            for field in ['model', 'provider', 'error_message']:
                if log.get(field):
                    searchable_text += str(log[field]).lower() + " "
            
            # 搜索请求和响应数据
            if log.get('request_data'):
                searchable_text += json.dumps(log['request_data'], ensure_ascii=False).lower() + " "
            
            if log.get('response_data'):
                searchable_text += json.dumps(log['response_data'], ensure_ascii=False).lower() + " "
            
            if search_term_lower in searchable_text:
                filtered_logs.append(log)
        
        return filtered_logs
    
    def search_logs_with_trace_pairs(self, logs: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
        """搜索日志并返回完整的trace对（用于增强布局）"""
        if not search_term:
            return logs
        
        # 首先进行常规搜索
        matching_logs = self.search_logs(logs, search_term)
        
        if not matching_logs:
            return []
        
        # 获取匹配日志的所有trace_id
        matching_trace_ids = set()
        for log in matching_logs:
            trace_id = log.get('trace_id')
            if trace_id:
                matching_trace_ids.add(trace_id)
        
        # 返回这些trace_id对应的所有日志（包括请求和响应）
        complete_logs = []
        for log in logs:
            if log.get('trace_id') in matching_trace_ids:
                complete_logs.append(log)
        
        return complete_logs
    
    def filter_logs(self, logs: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """根据过滤条件过滤日志"""
        filtered_logs = logs
        
        for filter_key, filter_value in filters.items():
            if not filter_value:
                continue
            
            if filter_key == 'has_error':
                filtered_logs = [log for log in filtered_logs if bool(log.get('error_message'))]
            elif filter_key == 'no_error':
                filtered_logs = [log for log in filtered_logs if not log.get('error_message')]
            elif filter_key == 'min_tokens':
                filtered_logs = [
                    log for log in filtered_logs 
                    if log.get('response_data', {}).get('usage', {}).get('total_tokens', 0) >= int(filter_value)
                ]
            elif filter_key == 'max_tokens':
                filtered_logs = [
                    log for log in filtered_logs 
                    if log.get('response_data', {}).get('usage', {}).get('total_tokens', float('inf')) <= int(filter_value)
                ]
        
        return filtered_logs
    
    def filter_logs_with_trace_pairs(self, logs: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """过滤日志并返回完整的trace对（用于增强布局）"""
        if not filters:
            return logs
        
        # 首先进行常规过滤
        matching_logs = self.filter_logs(logs, filters)
        
        if not matching_logs:
            return []
        
        # 获取匹配日志的所有trace_id
        matching_trace_ids = set()
        for log in matching_logs:
            trace_id = log.get('trace_id')
            if trace_id:
                matching_trace_ids.add(trace_id)
        
        # 返回这些trace_id对应的所有日志（包括请求和响应）
        complete_logs = []
        for log in logs:
            if log.get('trace_id') in matching_trace_ids:
                complete_logs.append(log)
        
        return complete_logs
    
    def paginate_logs(self, logs: List[Dict[str, Any]], page: int = 1, page_size: int = 50) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """分页处理日志"""
        total_count = len(logs)
        total_pages = (total_count + page_size - 1) // page_size
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        paginated_logs = logs[start_idx:end_idx]
        
        pagination_info = {
            'current_page': page,
            'page_size': page_size,
            'total_count': total_count,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
        
        return paginated_logs, pagination_info
    
    def query_logs_by_trace_id(self, trace_id: str, output_format: str = "table") -> Dict[str, Any]:
        """根据 trace_id 查询日志"""
        if not self.validate_trace_id(trace_id):
            return {
                "error": f"无效的 trace_id 格式: {trace_id}",
                "data": [],
                "source": "error"
            }
        
        # 首先尝试从 PostgreSQL 查询
        postgres_result = self._query_postgres_by_trace_id(trace_id)
        if postgres_result["data"]:
            return postgres_result
        
        # 如果 PostgreSQL 没有结果，尝试文件日志
        file_result = self._query_file_logs_by_trace_id(trace_id)
        if file_result["data"]:
            return file_result
        
        # 如果都没有结果
        return {
            "error": None,
            "data": [],
            "source": "both",
            "message": f"未找到 trace_id '{trace_id}' 的日志记录"
        }
    
    def _query_postgres_by_trace_id(self, trace_id: str) -> Dict[str, Any]:
        """从 PostgreSQL 查询指定 trace_id 的日志"""
        if not self.postgres_conn:
            return {"error": None, "data": [], "source": "postgres_unavailable"}
        
        try:
            with self.postgres_conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # 查询 API 日志表
                query = """
                    SELECT 
                        id, trace_id, timestamp, model, provider, 
                        request_data, response_data, error_message,
                        created_at, updated_at
                    FROM api_logs 
                    WHERE trace_id = %s 
                    ORDER BY timestamp ASC
                """
                cursor.execute(query, (trace_id,))
                rows = cursor.fetchall()
                
                logs = []
                for row in rows:
                    log_entry = {
                        "id": row["id"],
                        "trace_id": row["trace_id"],
                        "timestamp": row["timestamp"].isoformat() if row["timestamp"] else None,
                        "model": row["model"],
                        "provider": row["provider"],
                        "request_data": row["request_data"],
                        "response_data": row["response_data"],
                        "error_message": row["error_message"],
                        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
                        "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                        "source": "postgresql"
                    }
                    logs.append(log_entry)
                
                return {
                    "error": None,
                    "data": logs,
                    "source": "postgresql",
                    "total_count": len(logs)
                }
        
        except Exception as e:
            return {
                "error": f"PostgreSQL 查询失败: {str(e)}",
                "data": [],
                "source": "postgresql_error"
            }
    
    def _query_file_logs_by_trace_id(self, trace_id: str) -> Dict[str, Any]:
        """从文件日志查询指定 trace_id 的日志"""
        if not self.file_logger:
            return {"error": None, "data": [], "source": "file_unavailable"}
        
        try:
            # 获取日志目录
            log_dir = self._get_trace_id_log_directory()
            if not log_dir or not log_dir.exists():
                return {"error": None, "data": [], "source": "file_unavailable"}
            
            logs = []
            
            # 遍历日志文件（支持 .log 和 .jsonl 格式）
            for log_file in list(log_dir.glob("*.log")) + list(log_dir.glob("*.jsonl")):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line_num, line in enumerate(f, 1):
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                log_data = json.loads(line)
                                if log_data.get('trace_id') == trace_id:
                                    # 添加文件信息
                                    log_data['source'] = 'file'
                                    log_data['file_name'] = log_file.name
                                    log_data['line_number'] = line_num
                                    logs.append(log_data)
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    continue
            
            # 按时间戳排序
            logs.sort(key=lambda x: x.get('timestamp', ''))
            
            return {
                "error": None,
                "data": logs,
                "source": "file",
                "total_count": len(logs)
            }
        
        except Exception as e:
            return {
                "error": f"文件日志查询失败: {str(e)}",
                "data": [],
                "source": "file_error"
            }
    
    def list_recent_trace_ids(self, days: int = 7, limit: int = 20) -> Dict[str, Any]:
        """列出最近的 trace_id"""
        trace_ids = set()
        
        # 从 PostgreSQL 获取
        postgres_trace_ids = self._get_postgres_trace_ids(days, limit)
        trace_ids.update(postgres_trace_ids)
        
        # 从文件日志获取
        file_trace_ids = self._get_file_trace_ids(days, limit)
        trace_ids.update(file_trace_ids)
        
        # 转换为列表并排序
        trace_id_list = sorted(list(trace_ids))[:limit]
        
        return {
            "error": None,
            "data": trace_id_list,
            "source": "both",
            "total_count": len(trace_id_list)
        }
    
    def _get_postgres_trace_ids(self, days: int, limit: int) -> List[str]:
        """从 PostgreSQL 获取 trace_id 列表"""
        if not self.postgres_conn:
            return []
        
        try:
            with self.postgres_conn.cursor() as cursor:
                # 计算日期范围
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                query = """
                    SELECT DISTINCT trace_id 
                    FROM api_logs 
                    WHERE timestamp >= %s AND timestamp <= %s 
                    AND trace_id IS NOT NULL
                    ORDER BY trace_id DESC
                    LIMIT %s
                """
                cursor.execute(query, (start_date, end_date, limit))
                rows = cursor.fetchall()
                
                return [row[0] for row in rows if row[0]]
        
        except Exception as e:
            return []
    
    def _get_file_trace_ids(self, days: int, limit: int) -> List[str]:
        """从文件日志获取 trace_id 列表"""
        if not self.file_logger:
            return []
        
        try:
            log_dir = self._get_trace_id_log_directory()
            if not log_dir or not log_dir.exists():
                return []
            
            trace_ids = set()
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 遍历日志文件（支持 .log 和 .jsonl 格式）
            for log_file in list(log_dir.glob("*.log")) + list(log_dir.glob("*.jsonl")):
                try:
                    # 检查文件修改时间
                    file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_mtime < cutoff_date:
                        continue
                    
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                log_data = json.loads(line)
                                trace_id = log_data.get('trace_id')
                                if trace_id:
                                    trace_ids.add(trace_id)
                                    
                                    # 限制数量
                                    if len(trace_ids) >= limit:
                                        return list(trace_ids)
                            except json.JSONDecodeError:
                                continue
                except Exception as e:
                    continue
            
            return list(trace_ids)
        
        except Exception as e:
            return []
    
    def validate_trace_id(self, trace_id: str) -> bool:
        """验证 trace_id 格式"""
        if not trace_id or not isinstance(trace_id, str):
            return False
        
        # 支持多种 trace_id 格式：
        # 1. HarborAI 格式：hb_timestamp_randomstring
        # 2. 简单字母数字格式：8位字母数字字符
        # 3. UUID 格式：xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
        patterns = [
            r'^hb_\d+_[a-z0-9]+$',  # HarborAI 格式
            r'^[a-z0-9]{8,}$',      # 简单字母数字格式（至少8位）
            r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'  # UUID 格式
        ]
        
        return any(re.match(pattern, trace_id) for pattern in patterns)
    
    def _calculate_file_stats(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """从文件日志计算统计信息"""
        stats = {}
        
        for log in logs:
            model = log.get('model', 'unknown')
            provider = log.get('provider', 'unknown')
            key = f"{provider}:{model}"
            
            if key not in stats:
                stats[key] = {
                    'model': model,
                    'provider': provider,
                    'request_count': 0,
                    'success_count': 0,
                    'total_tokens': 0,
                    'total_cost': 0.0
                }
            
            stats[key]['request_count'] += 1
            
            if log.get('success', True):
                stats[key]['success_count'] += 1
            
            tokens = log.get('tokens') or log.get('total_tokens', 0)
            if tokens:
                stats[key]['total_tokens'] += tokens
            
            cost = log.get('cost') or log.get('estimated_cost', 0.0)
            if cost:
                stats[key]['total_cost'] += cost
        
        return list(stats.values())
    
    def _create_enhanced_logs(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """创建增强型日志表格数据，将请求和响应配对显示"""
        # 按 trace_id 分组
        trace_groups = defaultdict(list)
        for log in logs:
            trace_id = log.get('trace_id')
            if trace_id:
                trace_groups[trace_id].append(log)
        
        enhanced_logs = []
        for trace_id, group_logs in trace_groups.items():
            # 分离请求和响应日志
            request_logs = [log for log in group_logs if log.get('type') == 'request']
            response_logs = [log for log in group_logs if log.get('type') == 'response']
            
            # 如果有配对的请求和响应
            if request_logs and response_logs:
                # 按时间戳排序，选择最新的请求和响应
                request_log = max(request_logs, key=lambda x: x.get('timestamp', datetime.min) if isinstance(x.get('timestamp'), datetime) else datetime.min)
                response_log = max(response_logs, key=lambda x: x.get('timestamp', datetime.min) if isinstance(x.get('timestamp'), datetime) else datetime.min)
                
                # 解析时间戳
                request_time = None
                response_time = None
                
                # 处理请求时间
                req_ts = request_log.get('timestamp')
                if isinstance(req_ts, datetime):
                    request_time = req_ts
                elif isinstance(req_ts, str):
                    try:
                        request_time = datetime.fromisoformat(req_ts.replace('Z', '+00:00'))
                    except ValueError:
                        try:
                            request_time = datetime.strptime(req_ts, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            pass
                
                # 处理响应时间
                resp_ts = response_log.get('timestamp')
                if isinstance(resp_ts, datetime):
                    response_time = resp_ts
                elif isinstance(resp_ts, str):
                    try:
                        response_time = datetime.fromisoformat(resp_ts.replace('Z', '+00:00'))
                    except ValueError:
                        try:
                            response_time = datetime.strptime(resp_ts, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            pass
                
                # 计算耗时
                duration_ms = "N/A"
                if request_time and response_time:
                    # 确保响应时间晚于请求时间
                    if response_time > request_time:
                        calculated_duration = (response_time - request_time).total_seconds() * 1000
                        duration_ms = f"{calculated_duration:.1f}"
                    else:
                        # 如果时间顺序不对，尝试从response_log中获取latency
                        latency = response_log.get('latency')
                        if latency is not None:
                            duration_ms = f"{latency:.1f}"
                else:
                    # 尝试从response_log中获取latency
                    latency = response_log.get('latency')
                    if latency is not None:
                        duration_ms = f"{latency:.1f}"
                
                # 智能截断trace_id（显示前8位）
                short_trace_id = trace_id[-8:] if len(trace_id) > 8 else trace_id
                
                # 获取模型名称（移除硬截断，让Rich库自动处理）
                model = response_log.get('model', request_log.get('model', 'unknown'))
                
                # 格式化时间显示（分别显示请求时间和响应时间）
                request_time_str = request_time.strftime("%m-%d %H:%M:%S") if request_time else "N/A"
                response_time_str = response_time.strftime("%m-%d %H:%M:%S") if response_time else "N/A"
                
                # 从success字段推导状态
                success = response_log.get('success')
                if success is True:
                    status = 'success'
                elif success is False:
                    status = 'failed'
                else:
                    status = 'unknown'
                
                # 提取Token信息 - 按设计文档中的TracingRecord字段名称
                tokens = response_log.get('total_tokens')
                if not tokens:
                    # 尝试从response_data.usage中获取
                    response_data = response_log.get('response_data', {})
                    if response_data and isinstance(response_data, dict):
                        usage = response_data.get('usage', {})
                        if usage and isinstance(usage, dict):
                            tokens = usage.get('total_tokens')
                    
                    # 如果还是没有，尝试其他可能的字段名称
                    if not tokens:
                        tokens = response_log.get('tokens')
                        if isinstance(tokens, dict):
                            tokens = tokens.get('total_tokens')
                
                # 提取成本信息 - 按设计文档中的TracingRecord字段名称
                cost = response_log.get('total_cost')
                if cost is None:
                    # 尝试其他可能的字段名称以保持向后兼容性
                    cost = response_log.get('estimated_cost') or response_log.get('cost')
                    if isinstance(cost, dict):
                        # 如果成本是字典，尝试获取总成本
                        cost = cost.get('total_cost') or cost.get('total') or 0.0
                
                enhanced_log = {
                    'Trace ID': short_trace_id,
                    '请求时间': request_time_str,
                    '响应时间': response_time_str,
                    '耗时(ms)': duration_ms,
                    '模型': model,
                    '提供商': response_log.get('provider', 'unknown'),
                    '状态': status,
                    'Token': tokens if tokens else 'N/A',
                    '成本': cost if cost else 'N/A'
                }
                enhanced_logs.append(enhanced_log)
        
        return enhanced_logs
    
    def watch_logs(self, interval: int = 5, max_duration: int = 300):
        """实时监控日志"""
        self._print_info(f"开始实时监控日志 (每 {interval} 秒刷新一次，最多运行 {max_duration} 秒)")
        self._print_info("按 Ctrl+C 停止监控")
        
        self._watch_active = True
        start_time = time.time()
        last_log_count = 0
        
        try:
            while self._watch_active and (time.time() - start_time) < max_duration:
                # 获取最新日志
                logs_result = self.get_logs(days=1, limit=10)
                
                if logs_result["error"]:
                    self._print_error(f"获取日志失败: {logs_result['error']}")
                    time.sleep(interval)
                    continue
                
                current_logs = logs_result["data"]
                current_count = len(current_logs)
                
                # 清屏并显示最新状态
                if HAS_RICH and console:
                    console.clear()
                    
                    # 显示监控状态
                    status_text = f"🔍 实时监控中... | 运行时间: {int(time.time() - start_time)}s | 最新日志数: {current_count}"
                    if current_count > last_log_count:
                        status_text += f" | 新增: {current_count - last_log_count}"
                    
                    console.print(Panel(status_text, style="bold green"))
                    
                    # 显示最新日志
                    if current_logs:
                        self.format_logs_table(current_logs, logs_result["source"], "classic")
                    else:
                        console.print("暂无日志数据", style="yellow")
                else:
                    print(f"\n=== 实时监控 ({int(time.time() - start_time)}s) ===")
                    if current_logs:
                        self.format_logs_table(current_logs, logs_result["source"], "classic")
                    else:
                        print("暂无日志数据")
                
                last_log_count = current_count
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self._watch_active = False
            self._print_info("\n监控已停止")
        except Exception as e:
            self._watch_active = False
            self._print_error(f"监控过程中发生错误: {e}")
    
    def stop_watch(self):
        """停止实时监控"""
        self._watch_active = False
    
    def format_logs_table(self, logs: List[Dict[str, Any]], source: str, layout: str = "classic"):
        """格式化日志为表格"""
        if not logs:
            self._print_info("没有找到日志数据")
            return
        
        self._print_success(f"数据源: {source}")
        
        # 如果是增强型布局，需要进行请求-响应配对
        if layout == "enhanced":
            logs = self._create_enhanced_logs(logs)
            if not logs:
                self._print_info("没有找到配对的请求-响应日志")
                return
        
        if HAS_RICH and console:
            table = Table(show_header=True, header_style="bold magenta")
            
            if layout == "enhanced":
                # 增强型布局的列定义 - 两个时间列，适中的模型列
                table.add_column("Trace ID", style="cyan", width=8)
                table.add_column("请求时间", style="blue", width=15)  # 增加宽度以容纳完整时间
                table.add_column("响应时间", style="green", width=15)  # 增加宽度以容纳完整时间
                table.add_column("耗时(ms)", justify="right", width=10)
                table.add_column("模型", style="magenta", width=18)  # 调整为18宽度
                table.add_column("提供商", style="yellow", width=12)
                table.add_column("状态", justify="center", width=8)
                table.add_column("Token", justify="right", width=8)
                table.add_column("成本", justify="right", width=10)
            else:
                # 传统布局的列定义 - Trace ID 作为首列
                table.add_column("Trace ID", style="cyan", width=10)
                table.add_column("类型", style="yellow", width=8)
                table.add_column("时间", style="blue", width=17)
                table.add_column("模型", style="magenta", width=22)
                table.add_column("提供商", style="green", width=12)
                table.add_column("状态", justify="center", width=8)
                table.add_column("Token", justify="right", width=8)
                table.add_column("成本", justify="right", width=10)
            
            for log in logs:
                if layout == "enhanced":
                    # 增强型布局的行数据处理 - 使用两个时间字段
                    trace_id = log.get('Trace ID', 'N/A')
                    request_time_display = log.get('请求时间', 'N/A')
                    response_time_display = log.get('响应时间', 'N/A')
                    duration_display = log.get('耗时(ms)', 'N/A')
                    model = log.get('模型', 'N/A')
                    provider = log.get('提供商', 'N/A')
                    
                    # 处理状态
                    status = log.get('状态', 'unknown')
                    if status == 'success':
                        status_style = "green"
                    elif status == 'failed':
                        status_style = "red"
                    else:
                        status_style = "yellow"
                    status_display = f"[{status_style}]{status}[/{status_style}]"
                    
                    # 处理Token数
                    tokens = log.get('Token', 'N/A')
                    if isinstance(tokens, dict):
                        # 如果是字典，提取total_tokens
                        total_tokens = tokens.get('total_tokens', 0)
                        tokens_display = f"{total_tokens:,}" if total_tokens else "N/A"
                    elif isinstance(tokens, (int, float)):
                        tokens_display = f"{tokens:,}"
                    else:
                        tokens_display = str(tokens)
                    
                    # 处理成本 - 使用ASCII兼容的货币符号
                    cost = log.get('成本', 'N/A')
                    cost_display = f"CNY{cost:.4f}" if cost != 'N/A' and isinstance(cost, (int, float)) else str(cost)
                    
                    table.add_row(
                        trace_id,
                        request_time_display,
                        response_time_display,
                        duration_display,
                        model,  # 让Rich库自动处理截断
                        provider,
                        status_display,
                        tokens_display,
                        cost_display
                    )
                else:
                    # 传统布局的行数据处理
                    # 处理时间戳
                    timestamp = log.get('timestamp')
                    if isinstance(timestamp, str):
                        try:
                            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        except ValueError:
                            timestamp = None
                    
                    timestamp_display = timestamp.strftime("%m-%d %H:%M:%S") if timestamp else "N/A"
                    
                    # 处理状态
                    success = log.get('success')
                    if success is True:
                        status = "success"
                        status_style = "green"
                    elif success is False:
                        status = "failed"
                        status_style = "red"
                    else:
                        status = "unknown"
                        status_style = "yellow"
                    status_display = f"[{status_style}]{status}[/{status_style}]"
                    
                    # 处理Token数 - 按设计文档中的TracingRecord字段名称
                    tokens = log.get('total_tokens')
                    if not tokens:
                        # 尝试其他可能的字段名称以保持向后兼容性
                        tokens = log.get('tokens')
                        if isinstance(tokens, dict):
                            tokens = tokens.get('total_tokens')
                    tokens_display = f"{tokens:,}" if tokens and isinstance(tokens, (int, float)) else "N/A"
                    
                    # 处理耗时
                    duration = log.get('duration_ms') or log.get('latency')
                    duration_display = f"{duration:.1f}" if duration else "N/A"
                    
                    # 处理成本 - 按设计文档中的TracingRecord字段名称
                    cost = log.get('total_cost')
                    if cost is None:
                        # 尝试其他可能的字段名称以保持向后兼容性
                        cost = log.get('estimated_cost') or log.get('cost')
                        if isinstance(cost, dict):
                            # 如果成本是字典，尝试获取总成本
                            cost = cost.get('total_cost') or cost.get('total') or 0.0
                    cost_display = f"CNY{cost:.4f}" if cost and isinstance(cost, (int, float)) else "N/A"
                    
                    # 处理日志类型并添加视觉区分 - 使用智能推断
                    inferred_type = infer_log_type(log)
                    if inferred_type == 'request':
                        type_display = "[blue]REQ[/blue]"
                    elif inferred_type == 'response':
                        type_display = "[green]RES[/green]"
                    else:
                        type_display = "[yellow]UNK[/yellow]"
                    
                    # 处理 trace_id - 按设计文档中的TracingRecord字段名称
                    trace_id = log.get('hb_trace_id') or log.get('trace_id', 'N/A')
                    # 截取 trace_id 的后8位以适应列宽
                    if trace_id != 'N/A' and len(trace_id) > 8:
                        trace_id_display = trace_id[-8:]
                    else:
                        trace_id_display = trace_id
                    
                    # 正确提取 model 字段
                    model = log.get('model', 'N/A')
                    if model == 'N/A' and log.get('response_summary'):
                        model = log.get('response_summary', {}).get('model', 'N/A')
                    
                    # 正确提取 provider 字段，如果缺失则从模型名称推断
                    provider = log.get('provider')
                    if not provider:
                        provider = log.get('structured_provider')
                    if not provider or provider == 'agently':
                        provider = infer_provider_from_model(model)
                    if not provider:
                        provider = 'N/A'
                    
                    table.add_row(
                        trace_id_display,  # 首列显示 trace_id
                        type_display,
                        timestamp_display,
                        model,  # 移除硬截断，让Rich库自动处理
                        provider,  # 移除硬截断，让Rich库自动处理
                        status_display,
                        tokens_display,
                        cost_display
                    )
            
            console.print(table)
        else:
            # 简单文本格式
            print(f"{'Trace ID':<10} {'类型':<6} {'时间':<12} {'模型':<15} {'提供商':<10} {'状态':<8} {'Token':<8} {'成本':<10}")
            print("-" * 86)
            
            for log in logs:
                timestamp = log.get('timestamp', 'N/A')
                if isinstance(timestamp, str):
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        timestamp = timestamp.strftime("%m-%d %H:%M:%S")
                    except:
                        pass
                
                # 正确提取 model 字段
                model = log.get('model', 'N/A')
                if model == 'N/A' and log.get('response_summary'):
                    model = log.get('response_summary', {}).get('model', 'N/A')
                model = model[:14]
                
                # 正确提取 provider 字段，如果缺失则从模型名称推断
                provider = log.get('provider')
                if not provider:
                    provider = log.get('structured_provider')
                if not provider or provider == 'agently':
                    provider = infer_provider_from_model(model)
                if not provider:
                    provider = 'N/A'
                provider = provider[:9]
                
                # 改进状态判断逻辑
                success = log.get('success')
                if success is True:
                    status = "success"
                elif success is False or log.get('error_message') or log.get('error'):
                    status = "failed"
                else:
                    status = "unknown"
                
                # 按设计文档中的TracingRecord字段名称处理
                tokens = log.get('total_tokens') or log.get('tokens', 'N/A')
                cost = log.get('total_cost') or log.get('estimated_cost') or log.get('cost', 'N/A')
                
                # 处理日志类型 - 使用智能推断
                inferred_type = infer_log_type(log)
                if inferred_type == 'request':
                    type_display = "REQ"
                elif inferred_type == 'response':
                    type_display = "RES"
                else:
                    type_display = "UNK"
                
                # 处理 trace_id - 按设计文档中的TracingRecord字段名称
                trace_id = log.get('hb_trace_id') or log.get('trace_id', 'N/A')
                if trace_id != 'N/A' and len(trace_id) > 8:
                    trace_id_display = trace_id[-8:]
                else:
                    trace_id_display = str(trace_id)
                
                print(f"{trace_id_display:<10} {type_display:<6} {str(timestamp):<12} {model:<15} {provider:<10} {status:<8} {str(tokens):<8} {str(cost):<10}")
    
    def format_trace_id_logs(self, logs: List[Dict[str, Any]], trace_id: str, 
                            output_format: str = "table", source: str = "unknown") -> str:
        """格式化 trace_id 查询结果"""
        if not logs:
            return f"[ERROR] 未找到 trace_id '{trace_id}' 的日志记录"
        
        if output_format == "json":
            return self._format_trace_id_logs_json(logs, trace_id, source)
        else:
            return self._format_trace_id_logs_table(logs, trace_id, source)
    
    def _format_trace_id_logs_table(self, logs: List[Dict[str, Any]], 
                                   trace_id: str, source: str) -> str:
        """以表格格式显示 trace_id 查询结果"""
        output = []
        
        # 标题
        output.append(f"[SEARCH] Trace ID 查询结果: {trace_id}")
        output.append(f"[DATA] 数据源: {source}")
        output.append(f"[INFO] 找到 {len(logs)} 条记录")
        output.append("=" * 80)
        
        if HAS_RICH and console:
            # 使用 Rich 表格
            table = Table(title=f"Trace ID: {trace_id}")
            table.add_column("序号", style="cyan", no_wrap=True)
            table.add_column("时间", style="green")
            table.add_column("模型", style="yellow")
            table.add_column("提供商", style="blue")
            table.add_column("类型", style="magenta")
            table.add_column("状态", style="red")
            
            for i, log in enumerate(logs, 1):
                timestamp = log.get('timestamp', 'N/A')
                if timestamp and timestamp != 'N/A':
                    try:
                        # 尝试解析时间戳
                        if 'T' in timestamp:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except:
                        pass
                
                # 正确提取 model 字段
                model = log.get('model', 'N/A')
                if model == 'N/A' and log.get('response_summary'):
                    model = log.get('response_summary', {}).get('model', 'N/A')
                
                # 正确提取 provider 字段，如果缺失则从模型名称推断
                provider = log.get('provider')
                if not provider:
                    provider = log.get('structured_provider')
                if not provider or provider == 'agently':
                    provider = infer_provider_from_model(model)
                if not provider:
                    provider = 'N/A'
                
                log_type = log.get('type', 'N/A')
                
                # 改进状态判断逻辑
                success = log.get('success')
                if success is True:
                    status = "[SUCCESS] 成功"
                elif success is False or log.get('error_message') or log.get('error'):
                    status = "[ERROR] 错误"
                elif success is None and log_type == 'request':
                    status = "[PENDING] 处理中"
                else:
                    status = "[PENDING] 处理中"
                
                table.add_row(
                    str(i),
                    timestamp,
                    model,
                    provider,
                    log_type,
                    status
                )
            
            # 使用 Rich 渲染表格
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            console.print(table)
            sys.stdout = old_stdout
            output.append(buffer.getvalue())
        else:
            # 简单表格格式
            output.append(f"{'序号':<4} {'时间':<20} {'模型':<15} {'提供商':<10} {'类型':<8} {'状态':<8}")
            output.append("-" * 80)
            
            for i, log in enumerate(logs, 1):
                timestamp = log.get('timestamp', 'N/A')
                if timestamp and timestamp != 'N/A':
                    try:
                        if 'T' in timestamp:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            timestamp = dt.strftime('%m-%d %H:%M:%S')
                    except:
                        timestamp = timestamp[:19] if len(timestamp) > 19 else timestamp
                
                # 正确提取 model 字段
                model = log.get('model', 'N/A')
                if model == 'N/A' and log.get('response_summary'):
                    model = log.get('response_summary', {}).get('model', 'N/A')
                model = (model[:14] + '..') if len(model) > 15 else model
                
                # 正确提取 provider 字段，如果缺失则从模型名称推断
                provider = log.get('provider')
                if not provider:
                    provider = log.get('structured_provider')
                if not provider or provider == 'agently':
                    provider = infer_provider_from_model(model)
                if not provider:
                    provider = 'N/A'
                provider = (provider[:9] + '..') if len(provider) > 10 else provider
                
                log_type = log.get('type', 'N/A')[:7]
                
                # 改进状态判断逻辑
                success = log.get('success')
                if success is True:
                    status = "成功"
                elif success is False or log.get('error_message') or log.get('error'):
                    status = "错误"
                elif success is None and log_type == 'request':
                    status = "处理中"
                else:
                    status = "处理中"
                
                output.append(f"{i:<4} {timestamp:<20} {model:<15} {provider:<10} {log_type:<8} {status:<8}")
        
        # 显示第一条记录的详细信息
        if logs:
            output.append("\n" + "=" * 80)
            output.append("[DETAIL] 第一条记录详细信息:")
            output.append("-" * 40)
            
            first_log = logs[0]
            
            # 优先显示关键字段（按设计文档中的TracingRecord结构）
            priority_fields = [
                'hb_trace_id', 'otel_trace_id', 'span_id', 'operation_name', 'service_name',
                'start_time', 'end_time', 'duration_ms', 'provider', 'model', 'status', 'error_message',
                'prompt_tokens', 'completion_tokens', 'total_tokens', 'parsing_method', 'confidence',
                'input_cost', 'output_cost', 'total_cost', 'currency', 'pricing_source',
                'tags', 'logs', 'created_at'
            ]
            
            # 首先显示优先字段
            displayed_keys = set()
            for field in priority_fields:
                if field in first_log:
                    value = first_log[field]
                    if field in ['request_data', 'response_data', 'request', 'response', 'logs']:
                        # 对于大型数据，只显示摘要
                        if isinstance(value, (dict, list)):
                            try:
                                value_str = json.dumps(value, ensure_ascii=False, indent=2)
                                if len(value_str) > 500:
                                    value_str = value_str[:500] + "... (截断)"
                            except:
                                value_str = str(value)[:500] + "... (截断)"
                        else:
                            value_str = str(value)[:500] + "... (截断)" if len(str(value)) > 500 else str(value)
                        output.append(f"{field}: {value_str}")
                    else:
                        # 特殊处理Token解析相关字段
                        if field == 'parsing_method':
                            output.append(f"Token解析方法: {value if value else 'N/A'}")
                        elif field == 'confidence':
                            confidence_str = f"{value:.2f}" if isinstance(value, (int, float)) else str(value)
                            output.append(f"解析置信度: {confidence_str if value is not None else 'N/A'}")
                        elif field == 'pricing_source':
                            output.append(f"价格数据源: {value if value else 'N/A'}")
                        else:
                            output.append(f"{field}: {value}")
                    displayed_keys.add(field)
            
            # 然后显示其他字段
            for key, value in first_log.items():
                if key not in displayed_keys:
                    if key in ['request_data', 'response_data', 'request', 'response']:
                        # 对于大型数据，只显示摘要
                        if isinstance(value, (dict, list)):
                            try:
                                value_str = json.dumps(value, ensure_ascii=False, indent=2)
                                if len(value_str) > 500:
                                    value_str = value_str[:500] + "... (截断)"
                            except:
                                value_str = str(value)[:500] + "... (截断)"
                        else:
                            value_str = str(value)[:500] + "... (截断)" if len(str(value)) > 500 else str(value)
                        output.append(f"{key}: {value_str}")
                    else:
                        output.append(f"{key}: {value}")
            
            # 添加Token解析调试信息部分
            if any(field in first_log for field in ['parsing_method', 'confidence']):
                output.append("\n" + "-" * 40)
                output.append("[DEBUG] Token解析调试信息:")
                parsing_method = first_log.get('parsing_method', 'N/A')
                confidence = first_log.get('confidence')
                confidence_str = f"{confidence:.2f}" if isinstance(confidence, (int, float)) else str(confidence) if confidence is not None else 'N/A'
                output.append(f"  解析方法: {parsing_method}")
                output.append(f"  置信度: {confidence_str}")
                if confidence is not None and isinstance(confidence, (int, float)):
                    if confidence >= 0.9:
                        output.append(f"  质量评估: 高置信度 ✓")
                    elif confidence >= 0.7:
                        output.append(f"  质量评估: 中等置信度 ⚠")
                    else:
                        output.append(f"  质量评估: 低置信度 ⚠ (建议检查)")
            
            # 添加成本计算调试信息部分
            if 'pricing_source' in first_log:
                output.append("\n" + "-" * 40)
                output.append("[DEBUG] 成本计算调试信息:")
                pricing_source = first_log.get('pricing_source', 'N/A')
                output.append(f"  价格数据源: {pricing_source}")
                input_cost = first_log.get('input_cost')
                output_cost = first_log.get('output_cost')
                total_cost = first_log.get('total_cost')
                currency = first_log.get('currency', 'CNY')
                if input_cost is not None:
                    output.append(f"  输入成本: {currency}{input_cost:.6f}")
                if output_cost is not None:
                    output.append(f"  输出成本: {currency}{output_cost:.6f}")
                if total_cost is not None:
                    output.append(f"  总成本: {currency}{total_cost:.6f}")
        
        return "\n".join(output)
    
    def _format_trace_id_logs_json(self, logs: List[Dict[str, Any]], 
                                  trace_id: str, source: str) -> str:
        """以 JSON 格式显示 trace_id 查询结果"""
        # 清理日志数据以优化JSON输出
        cleaned_logs = self._clean_logs_for_json_output(logs)
        
        result = {
            "trace_id": trace_id,
            "source": source,
            "total_count": len(cleaned_logs),
            "logs": cleaned_logs
        }
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    def format_recent_trace_ids(self, trace_ids: List[str], source: str) -> str:
        """格式化最近的 trace_id 列表"""
        if not trace_ids:
            return "[ERROR] 未找到最近的 trace_id"
        
        output = []
        output.append(f"[SEARCH] 最近的 Trace ID 列表")
        output.append(f"[DATA] 数据源: {source}")
        output.append(f"[INFO] 找到 {len(trace_ids)} 个 trace_id")
        output.append("=" * 60)
        
        if HAS_RICH and console:
            # 使用 Rich 表格
            table = Table(title="最近的 Trace ID")
            table.add_column("序号", style="cyan", no_wrap=True)
            table.add_column("Trace ID", style="green")
            
            for i, trace_id in enumerate(trace_ids, 1):
                table.add_row(str(i), trace_id)
            
            # 使用 Rich 渲染表格
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            console.print(table)
            sys.stdout = old_stdout
            output.append(buffer.getvalue())
        else:
            # 简单列表格式
            for i, trace_id in enumerate(trace_ids, 1):
                output.append(f"{i:>3}. {trace_id}")
        
        output.append("\n[TIP] 使用方法:")
        output.append("   python view_logs.py --trace-id <trace_id>")
        output.append("   例如: python view_logs.py --trace-id " + (trace_ids[0] if trace_ids else "hb_1234567890_abcd1234"))
        
        return "\n".join(output)

    def format_stats_table(self, stats: List[Dict[str, Any]], source: str):
        """格式化统计信息为表格"""
        if not stats:
            self._print_info("没有找到统计数据")
            return
        
        self._print_success(f"统计数据源: {source}")
        
        # 计算总计
        total_requests = sum(stat.get('request_count', 0) for stat in stats)
        total_success = sum(stat.get('success_count', 0) for stat in stats)
        total_tokens = sum(stat.get('total_tokens', 0) for stat in stats)
        total_cost = sum(stat.get('total_cost', 0.0) for stat in stats)
        success_rate = (total_success / total_requests * 100) if total_requests > 0 else 0
        
        # 统计价格数据源分布
        pricing_sources = {}
        for stat in stats:
            pricing_source = stat.get('pricing_source', 'unknown')
            pricing_sources[pricing_source] = pricing_sources.get(pricing_source, 0) + stat.get('request_count', 0)
        
        # 显示总计
        if HAS_RICH and console:
            # 构建总体统计面板内容
            summary_content = (
                f"总请求数: {total_requests:,}\n"
                f"成功请求: {total_success:,}\n"
                f"成功率: {success_rate:.1f}%\n"
                f"总Token数: {total_tokens:,}\n"
                f"总成本: CNY{total_cost:.4f}"
            )
            
            # 添加价格数据源统计
            if pricing_sources:
                summary_content += "\n\n价格数据源分布:"
                for source, count in sorted(pricing_sources.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_requests * 100) if total_requests > 0 else 0
                    summary_content += f"\n  {source}: {count:,} ({percentage:.1f}%)"
            
            summary_panel = Panel(
                summary_content,
                title="总体统计",
                border_style="blue"
            )
            console.print(summary_panel)
            
            # 详细表格
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("模型", style="cyan")
            table.add_column("提供商", style="green")
            table.add_column("请求数", justify="right")
            table.add_column("成功数", justify="right")
            table.add_column("成功率", justify="right")
            table.add_column("总Token", justify="right")
            table.add_column("总成本", justify="right")
            table.add_column("价格源", style="yellow")
            
            for stat in stats:
                request_count = stat.get('request_count', 0)
                success_count = stat.get('success_count', 0)
                success_rate = (success_count / request_count * 100) if request_count > 0 else 0
                pricing_source = stat.get('pricing_source', 'N/A')
                
                table.add_row(
                    stat.get('model', 'N/A'),
                    stat.get('provider', 'N/A'),
                    f"{request_count:,}",
                    f"{success_count:,}",
                    f"{success_rate:.1f}%",
                    f"{stat.get('total_tokens', 0):,}",
                    f"CNY{stat.get('total_cost', 0.0):.4f}",
                    pricing_source
                )
            
            console.print(table)
            
            # 显示Token解析质量统计（如果有相关数据）
            parsing_stats = {}
            confidence_stats = []
            for stat in stats:
                parsing_method = stat.get('parsing_method')
                if parsing_method:
                    parsing_stats[parsing_method] = parsing_stats.get(parsing_method, 0) + stat.get('request_count', 0)
                
                confidence = stat.get('confidence')
                if confidence is not None and isinstance(confidence, (int, float)):
                    confidence_stats.append(confidence)
            
            if parsing_stats or confidence_stats:
                debug_content = ""
                if parsing_stats:
                    debug_content += "Token解析方法分布:\n"
                    for method, count in sorted(parsing_stats.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / total_requests * 100) if total_requests > 0 else 0
                        debug_content += f"  {method}: {count:,} ({percentage:.1f}%)\n"
                
                if confidence_stats:
                    avg_confidence = sum(confidence_stats) / len(confidence_stats)
                    high_conf = sum(1 for c in confidence_stats if c >= 0.9)
                    med_conf = sum(1 for c in confidence_stats if 0.7 <= c < 0.9)
                    low_conf = sum(1 for c in confidence_stats if c < 0.7)
                    
                    if debug_content:
                        debug_content += "\n"
                    debug_content += f"Token解析质量统计:\n"
                    debug_content += f"  平均置信度: {avg_confidence:.3f}\n"
                    debug_content += f"  高置信度(≥0.9): {high_conf} ({high_conf/len(confidence_stats)*100:.1f}%)\n"
                    debug_content += f"  中等置信度(0.7-0.9): {med_conf} ({med_conf/len(confidence_stats)*100:.1f}%)\n"
                    debug_content += f"  低置信度(<0.7): {low_conf} ({low_conf/len(confidence_stats)*100:.1f}%)"
                
                debug_panel = Panel(
                    debug_content.rstrip(),
                    title="调试信息",
                    border_style="yellow"
                )
                console.print(debug_panel)
        else:
            # 简单文本格式
            print(f"总请求数: {total_requests:,}")
            print(f"成功请求: {total_success:,}")
            print(f"成功率: {success_rate:.1f}%")
            print(f"总Token数: {total_tokens:,}")
            print(f"总成本: CNY{total_cost:.4f}")
            
            # 显示价格数据源分布
            if pricing_sources:
                print("\n价格数据源分布:")
                for source, count in sorted(pricing_sources.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_requests * 100) if total_requests > 0 else 0
                    print(f"  {source}: {count:,} ({percentage:.1f}%)")
            
            print()
            
            print(f"{'模型':<20} {'提供商':<10} {'请求数':<8} {'成功数':<8} {'成功率':<8} {'Token':<10} {'成本':<10} {'价格源':<12}")
            print("-" * 100)
            
            for stat in stats:
                request_count = stat.get('request_count', 0)
                success_count = stat.get('success_count', 0)
                success_rate = (success_count / request_count * 100) if request_count > 0 else 0
                pricing_source = stat.get('pricing_source', 'N/A')
                
                print(f"{stat.get('model', 'N/A'):<20} "
                      f"{stat.get('provider', 'N/A'):<10} "
                      f"{request_count:<8} "
                      f"{success_count:<8} "
                      f"{success_rate:.1f}%{'':<3} "
                      f"{stat.get('total_tokens', 0):<10} "
                      f"CNY{stat.get('total_cost', 0.0):.4f}{'':<2} "
                      f"{pricing_source:<12}")

    def run_demo(self, interactive: bool = True, layout: str = "classic"):
        """运行演示模式"""
        self._print_success("🚀 HarborAI 日志查看工具演示")
        self._print_info("正在生成演示数据...")
        
        # 生成演示数据
        sample_logs = DemoDataGenerator.generate_sample_logs(50)
        
        # 演示场景1: 基本日志查看
        self._demo_basic_viewing(sample_logs, interactive, layout)
        
        # 演示场景2: 过滤功能
        self._demo_filtering(sample_logs, interactive, layout)
        
        # 演示场景3: 统计信息
        self._demo_statistics(sample_logs, interactive)
        
        # 演示场景4: trace_id 查询
        self._demo_trace_id_query(sample_logs, interactive)
        
        # 演示场景5: 输出格式
        self._demo_output_formats(sample_logs, interactive)
        
        # 演示场景6: 搜索和高级过滤
        self._demo_search_and_filtering(sample_logs, interactive, layout)
        
        # 演示场景7: 分页功能
        self._demo_pagination(sample_logs, interactive, layout)
        
        # 演示场景8: 错误处理
        self._demo_error_handling(interactive)
        
        self._print_success("🎉 演示完成！")
    
    def _demo_basic_viewing(self, sample_logs: List[Dict[str, Any]], interactive: bool = True, layout: str = "classic"):
        """演示基本日志查看"""
        self._print_success("\n📋 演示场景1: 基本日志查看")
        self._print_info(f"显示最近的日志记录（布局: {layout}）...")
        
        # 显示前10条日志
        recent_logs = sample_logs[:10]
        self.format_logs_table(recent_logs, "演示数据", layout)
        
        if interactive:
            input("\n按回车键继续下一个演示...")
        else:
            import time
            time.sleep(2)
    
    def _demo_filtering(self, sample_logs: List[Dict[str, Any]], interactive: bool = True, layout: str = "classic"):
        """演示过滤功能"""
        self._print_success("\n🔍 演示场景2: 按模型和提供商过滤")
        
        # 按模型过滤
        gpt4_logs = [log for log in sample_logs if 'gpt-4' in log.get('model', '').lower()]
        self._print_info(f"GPT-4 模型的日志 ({len(gpt4_logs)} 条):")
        if gpt4_logs:
            self.format_logs_table(gpt4_logs[:5], "演示数据 (GPT-4)", layout)
        
        # 按提供商过滤
        openai_logs = [log for log in sample_logs if log.get('provider') == 'openai']
        self._print_info(f"\nOpenAI 提供商的日志 ({len(openai_logs)} 条):")
        if openai_logs:
            self.format_logs_table(openai_logs[:5], "演示数据 (OpenAI)", layout)
        
        if interactive:
            input("\n按回车键继续下一个演示...")
        else:
            import time
            time.sleep(2)
    
    def _demo_statistics(self, sample_logs: List[Dict[str, Any]], interactive: bool = True):
        """演示统计信息"""
        self._print_success("\n📊 演示场景3: 统计信息查看")
        
        # 计算统计信息
        stats = self._calculate_file_stats(sample_logs)
        self.format_stats_table(stats, "演示数据")
        
        if interactive:
            input("\n按回车键继续下一个演示...")
        else:
            import time
            time.sleep(2)
    
    def _demo_trace_id_query(self, sample_logs: List[Dict[str, Any]], interactive: bool = True):
        """演示 trace_id 查询"""
        self._print_success("\n🔎 演示场景4: trace_id 查询")
        
        # 获取一个示例 trace_id
        if sample_logs:
            sample_trace_id = sample_logs[0].get('trace_id')
            self._print_info(f"查询 trace_id: {sample_trace_id}")
            
            # 查找相关日志
            trace_logs = [log for log in sample_logs if log.get('trace_id') == sample_trace_id]
            if trace_logs:
                self.format_trace_id_logs(trace_logs, sample_trace_id, "table", "演示数据")
            else:
                self._print_warning("未找到相关日志")
        
        if interactive:
            input("\n按回车键继续下一个演示...")
        else:
            import time
            time.sleep(2)
    
    def _demo_output_formats(self, sample_logs: List[Dict[str, Any]], interactive: bool = True):
        """演示不同输出格式"""
        self._print_success("\n📄 演示场景5: 不同输出格式")
        
        # 表格格式
        self._print_info("表格格式:")
        self.format_logs_table(sample_logs[:3], "演示数据", "classic")
        
        # JSON 格式
        self._print_info("\nJSON 格式:")
        if sample_logs:
            sample_trace_id = sample_logs[0].get('trace_id')
            trace_logs = [log for log in sample_logs if log.get('trace_id') == sample_trace_id]
            if trace_logs:
                json_output = self.format_trace_id_logs(trace_logs, sample_trace_id, "json", "演示数据")
                if HAS_RICH and console:
                    syntax = Syntax(json_output, "json", theme="monokai", line_numbers=True)
                    console.print(syntax)
                else:
                    print(json_output)
        
        if interactive:
            input("\n按回车键继续下一个演示...")
        else:
            import time
            time.sleep(2)
    
    def _demo_search_and_filtering(self, sample_logs: List[Dict[str, Any]], interactive: bool = True, layout: str = "classic"):
        """演示搜索和高级过滤"""
        self._print_success("\n🔍 演示场景6: 搜索和高级过滤")
        
        # 搜索包含错误的日志
        if layout == "enhanced":
            # 对于增强布局，使用新的方法确保返回完整的trace对
            error_logs = [log for log in sample_logs if log.get('error_message')]
            if error_logs:
                # 获取错误日志的trace_id
                error_trace_ids = set(log.get('trace_id') for log in error_logs if log.get('trace_id'))
                # 获取这些trace_id的所有日志
                error_logs_with_pairs = [log for log in sample_logs if log.get('trace_id') in error_trace_ids]
            else:
                error_logs_with_pairs = []
            
            self._print_info(f"包含错误的日志 ({len(error_logs)} 条错误，{len(error_logs_with_pairs)} 条完整trace):")
            if error_logs_with_pairs:
                self.format_logs_table(error_logs_with_pairs, "演示数据 (错误日志)", layout)
        else:
            error_logs = [log for log in sample_logs if log.get('error_message')]
            self._print_info(f"包含错误的日志 ({len(error_logs)} 条):")
            if error_logs:
                self.format_logs_table(error_logs[:3], "演示数据 (错误日志)", layout)
        
        # 搜索特定内容
        if layout == "enhanced":
            search_results = self.search_logs_with_trace_pairs(sample_logs, "gpt")
        else:
            search_results = self.search_logs(sample_logs, "gpt")
        
        self._print_info(f"\n搜索 'gpt' 的结果 ({len(search_results)} 条):")
        if search_results:
            if layout == "enhanced":
                self.format_logs_table(search_results, "演示数据 (搜索结果)", layout)
            else:
                self.format_logs_table(search_results[:3], "演示数据 (搜索结果)", layout)
        
        # 高级过滤
        filters = {'has_error': True}
        if layout == "enhanced":
            filtered_logs = self.filter_logs_with_trace_pairs(sample_logs, filters)
        else:
            filtered_logs = self.filter_logs(sample_logs, filters)
        
        self._print_info(f"\n高级过滤 (仅错误日志): {len(filtered_logs)} 条")
        if filtered_logs and layout == "enhanced":
            self.format_logs_table(filtered_logs, "演示数据 (过滤结果)", layout)
        
        if interactive:
            input("\n按回车键继续下一个演示...")
        else:
            import time
            time.sleep(2)
    
    def _demo_pagination(self, sample_logs: List[Dict[str, Any]], interactive: bool = True, layout: str = "classic"):
        """演示分页功能"""
        self._print_success("\n📄 演示场景7: 分页功能")
        
        # 分页显示
        page_size = 5
        paginated_logs, pagination_info = self.paginate_logs(sample_logs, page=1, page_size=page_size)
        
        self._print_info(f"分页信息: 第 {pagination_info['current_page']} 页，共 {pagination_info['total_pages']} 页")
        self._print_info(f"总记录数: {pagination_info['total_count']}，每页显示: {pagination_info['page_size']}")
        
        self.format_logs_table(paginated_logs, "演示数据 (第1页)", layout)
        
        # 显示第二页
        if pagination_info['has_next']:
            paginated_logs_2, pagination_info_2 = self.paginate_logs(sample_logs, page=2, page_size=page_size)
            self._print_info(f"\n第 {pagination_info_2['current_page']} 页:")
            self.format_logs_table(paginated_logs_2, "演示数据 (第2页)", layout)
        
        if interactive:
            input("\n按回车键继续下一个演示...")
        else:
            import time
            time.sleep(2)
    
    # ==================== 导出功能 ====================
    
    def export_logs(self, logs: List[Dict[str, Any]], file_path: str, 
                   export_format: Optional[str] = None) -> Dict[str, Any]:
        """
        导出日志数据到文件
        
        Args:
            logs: 要导出的日志数据
            file_path: 导出文件路径
            export_format: 导出格式 (csv, json, excel)，如果为None则根据文件扩展名判断
            
        Returns:
            包含导出结果的字典
        """
        try:
            # 确定导出格式
            if export_format is None:
                file_ext = Path(file_path).suffix.lower()
                if file_ext == '.csv':
                    export_format = 'csv'
                elif file_ext == '.json':
                    export_format = 'json'
                elif file_ext in ['.xlsx', '.xls']:
                    export_format = 'excel'
                else:
                    return {
                        "success": False,
                        "error": f"无法从文件扩展名 '{file_ext}' 判断导出格式，请使用 --export-format 参数指定"
                    }
            
            # 检查依赖
            if export_format == 'excel' and not HAS_PANDAS:
                return {
                    "success": False,
                    "error": "Excel 导出需要 pandas 库，请运行: pip install pandas"
                }
            
            if export_format == 'excel' and not HAS_OPENPYXL:
                return {
                    "success": False,
                    "error": "Excel 导出需要 openpyxl 库，请运行: pip install openpyxl"
                }
            
            # 预处理数据
            processed_logs = self._prepare_export_data(logs)
            
            # 创建目录
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 根据格式导出
            if export_format == 'csv':
                return self._export_to_csv(processed_logs, file_path)
            elif export_format == 'json':
                return self._export_to_json(processed_logs, file_path)
            elif export_format == 'excel':
                return self._export_to_excel(processed_logs, file_path)
            else:
                return {
                    "success": False,
                    "error": f"不支持的导出格式: {export_format}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"导出失败: {str(e)}"
            }
    
    def _prepare_export_data(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        预处理导出数据，展平嵌套结构并格式化字段
        
        Args:
            logs: 原始日志数据
            
        Returns:
            处理后的日志数据
        """
        processed_logs = []
        
        for log in logs:
            processed_log = {}
            
            # 基础字段
            processed_log['trace_id'] = log.get('trace_id', '')
            processed_log['timestamp'] = self._format_timestamp_for_export(log.get('timestamp'))
            processed_log['log_type'] = log.get('log_type', '')
            processed_log['model'] = log.get('model', '')
            processed_log['provider'] = log.get('provider', '')
            processed_log['status'] = log.get('status', '')
            
            # Token 信息
            tokens = log.get('tokens', {})
            if isinstance(tokens, dict):
                processed_log['prompt_tokens'] = tokens.get('prompt_tokens', 0)
                processed_log['completion_tokens'] = tokens.get('completion_tokens', 0)
                processed_log['total_tokens'] = tokens.get('total_tokens', 0)
            else:
                processed_log['prompt_tokens'] = 0
                processed_log['completion_tokens'] = 0
                processed_log['total_tokens'] = 0
            
            # 成本信息
            cost = log.get('cost', {})
            if isinstance(cost, dict):
                processed_log['prompt_cost'] = cost.get('prompt_cost', 0.0)
                processed_log['completion_cost'] = cost.get('completion_cost', 0.0)
                processed_log['total_cost'] = cost.get('total_cost', 0.0)
            else:
                processed_log['prompt_cost'] = 0.0
                processed_log['completion_cost'] = 0.0
                processed_log['total_cost'] = 0.0
            
            # 错误信息
            processed_log['error_message'] = log.get('error_message', '')
            processed_log['error_type'] = log.get('error_type', '')
            
            # 请求和响应内容（截断长文本）
            request_data = log.get('request_data', {})
            response_data = log.get('response_data', {})
            
            if isinstance(request_data, dict):
                processed_log['request_content'] = str(request_data.get('messages', ''))[:500]
                processed_log['request_temperature'] = request_data.get('temperature', '')
                processed_log['request_max_tokens'] = request_data.get('max_tokens', '')
            else:
                processed_log['request_content'] = ''
                processed_log['request_temperature'] = ''
                processed_log['request_max_tokens'] = ''
            
            if isinstance(response_data, dict):
                processed_log['response_content'] = str(response_data.get('choices', ''))[:500]
                processed_log['response_finish_reason'] = response_data.get('finish_reason', '')
            else:
                processed_log['response_content'] = ''
                processed_log['response_finish_reason'] = ''
            
            # 时间信息
            processed_log['duration_ms'] = log.get('duration_ms', 0)
            processed_log['created_at'] = self._format_timestamp_for_export(log.get('created_at'))
            
            processed_logs.append(processed_log)
        
        return processed_logs
    
    def _format_timestamp_for_export(self, timestamp) -> str:
        """格式化时间戳为易读格式"""
        if not timestamp:
            return ''
        
        try:
            if isinstance(timestamp, str):
                # 尝试解析字符串时间戳
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            elif isinstance(timestamp, datetime):
                dt = timestamp
            else:
                return str(timestamp)
            
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            return str(timestamp)
    
    def _export_to_csv(self, logs: List[Dict[str, Any]], file_path: str) -> Dict[str, Any]:
        """导出为 CSV 格式"""
        try:
            if not logs:
                return {
                    "success": False,
                    "error": "没有数据可导出"
                }
            
            with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = logs[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for log in logs:
                    writer.writerow(log)
            
            return {
                "success": True,
                "file_path": file_path,
                "record_count": len(logs),
                "format": "CSV"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"CSV 导出失败: {str(e)}"
            }
    
    def _export_to_json(self, logs: List[Dict[str, Any]], file_path: str) -> Dict[str, Any]:
        """导出为 JSON 格式"""
        try:
            # 清理日志数据以优化JSON输出
            cleaned_logs = self._clean_logs_for_json_output(logs)
            
            export_data = {
                "export_info": {
                    "timestamp": datetime.now().isoformat(),
                    "record_count": len(cleaned_logs),
                    "format": "JSON"
                },
                "logs": cleaned_logs
            }
            
            with open(file_path, 'w', encoding='utf-8') as jsonfile:
                json.dump(export_data, jsonfile, ensure_ascii=False, indent=2, default=str)
            
            return {
                "success": True,
                "file_path": file_path,
                "record_count": len(cleaned_logs),
                "format": "JSON"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"JSON 导出失败: {str(e)}"
            }
    
    def _export_to_excel(self, logs: List[Dict[str, Any]], file_path: str) -> Dict[str, Any]:
        """导出为 Excel 格式"""
        try:
            if not logs:
                return {
                    "success": False,
                    "error": "没有数据可导出"
                }
            
            # 创建 DataFrame
            df = pd.DataFrame(logs)
            
            # 创建 Excel writer
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # 主数据表
                df.to_excel(writer, sheet_name='日志数据', index=False)
                
                # 统计摘要表
                summary_data = self._generate_export_summary(logs)
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='统计摘要', index=False)
                
                # 错误统计表
                error_logs = [log for log in logs if log.get('error_message')]
                if error_logs:
                    error_df = pd.DataFrame(error_logs)
                    error_df.to_excel(writer, sheet_name='错误日志', index=False)
            
            return {
                "success": True,
                "file_path": file_path,
                "record_count": len(logs),
                "format": "Excel",
                "sheets": ["日志数据", "统计摘要"] + (["错误日志"] if error_logs else [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Excel 导出失败: {str(e)}"
            }
    
    def _generate_export_summary(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成导出统计摘要"""
        summary = []
        
        # 基础统计
        total_logs = len(logs)
        request_logs = len([log for log in logs if log.get('log_type') == 'request'])
        response_logs = len([log for log in logs if log.get('log_type') == 'response'])
        error_logs = len([log for log in logs if log.get('error_message')])
        
        summary.append({"指标": "总日志数", "数值": total_logs})
        summary.append({"指标": "请求日志数", "数值": request_logs})
        summary.append({"指标": "响应日志数", "数值": response_logs})
        summary.append({"指标": "错误日志数", "数值": error_logs})
        
        # Token 统计
        total_tokens = sum(log.get('total_tokens', 0) for log in logs)
        total_cost = sum(log.get('total_cost', 0.0) for log in logs)
        
        summary.append({"指标": "总 Token 数", "数值": total_tokens})
        summary.append({"指标": "总成本", "数值": f"${total_cost:.4f}"})
        
        # 模型统计
        models = {}
        providers = {}
        
        for log in logs:
            model = log.get('model', 'unknown')
            provider = log.get('provider', 'unknown')
            
            models[model] = models.get(model, 0) + 1
            providers[provider] = providers.get(provider, 0) + 1
        
        summary.append({"指标": "使用的模型数", "数值": len(models)})
        summary.append({"指标": "使用的提供商数", "数值": len(providers)})
        
        # 最常用的模型和提供商
        if models:
            most_used_model = max(models.items(), key=lambda x: x[1])
            summary.append({"指标": "最常用模型", "数值": f"{most_used_model[0]} ({most_used_model[1]}次)"})
        
        if providers:
            most_used_provider = max(providers.items(), key=lambda x: x[1])
            summary.append({"指标": "最常用提供商", "数值": f"{most_used_provider[0]} ({most_used_provider[1]}次)"})
        
        return summary

    def _clean_logs_for_json_output(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        清理日志数据以优化JSON输出，将Token和成本信息统一到response_data.usage中
        
        Args:
            logs: 原始日志数据
            
        Returns:
            清理后的日志数据
        """
        cleaned_logs = []
        
        for log in logs:
            cleaned_log = log.copy()
            
            # 1. 确保response_data和usage对象存在
            if 'response_data' not in cleaned_log:
                cleaned_log['response_data'] = {}
            if not isinstance(cleaned_log['response_data'], dict):
                cleaned_log['response_data'] = {}
            if 'usage' not in cleaned_log['response_data']:
                cleaned_log['response_data']['usage'] = {}
            
            usage = cleaned_log['response_data']['usage']
            
            # 2. 收集Token信息 - 优先使用现有usage中的值，然后是根级别的值
            prompt_tokens = usage.get('prompt_tokens') or cleaned_log.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens') or cleaned_log.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens') or cleaned_log.get('total_tokens', 0)
            
            # 处理旧的tokens字段
            if 'tokens' in cleaned_log and not total_tokens:
                total_tokens = cleaned_log.get('tokens', 0)
            
            # 3. 收集成本信息 - 确保类型安全，处理None值
            input_cost = cleaned_log.get('input_cost') or 0.0
            output_cost = cleaned_log.get('output_cost') or 0.0
            total_cost = cleaned_log.get('total_cost') or 0.0
            currency = cleaned_log.get('currency') or 'CNY'
            
            # 确保成本字段是数值类型
            try:
                input_cost = float(input_cost) if input_cost is not None else 0.0
                output_cost = float(output_cost) if output_cost is not None else 0.0
                total_cost = float(total_cost) if total_cost is not None else 0.0
            except (ValueError, TypeError):
                input_cost = 0.0
                output_cost = 0.0
                total_cost = 0.0
            
            # 处理旧的cost字段
            if 'cost' in cleaned_log and total_cost == 0.0:
                try:
                    old_cost = cleaned_log.get('cost')
                    total_cost = float(old_cost) if old_cost is not None else 0.0
                except (ValueError, TypeError):
                    total_cost = 0.0
            
            # 如果total_cost为0但有input_cost和output_cost，计算total_cost
            if total_cost == 0.0 and (input_cost > 0 or output_cost > 0):
                total_cost = input_cost + output_cost
            
            # 4. 将所有Token和成本信息统一到usage对象中
            usage.update({
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'input_cost': input_cost,
                'output_cost': output_cost,
                'total_cost': total_cost,
                'currency': currency
            })
            
            # 5. 删除根级别的重复字段
            fields_to_remove = [
                'prompt_tokens', 'completion_tokens', 'total_tokens', 'tokens',
                'input_cost', 'output_cost', 'total_cost', 'cost', 'currency'
            ]
            
            for field in fields_to_remove:
                if field in cleaned_log:
                    del cleaned_log[field]
            
            # 6. 清理空值和None值（但保留usage对象）
            cleaned_log = {k: v for k, v in cleaned_log.items() if v is not None}
            
            cleaned_logs.append(cleaned_log)
        
        return cleaned_logs

    def _demo_error_handling(self, interactive: bool = True):
        """演示错误处理"""
        self._print_success("\n⚠️ 演示场景8: 错误处理")
        
        # 模拟各种错误情况
        self._print_info("模拟无效 trace_id 查询:")
        invalid_trace_id = "invalid-trace-id-12345"
        is_valid = self.validate_trace_id(invalid_trace_id)
        if not is_valid:
            self._print_warning(f"trace_id '{invalid_trace_id}' 格式无效")
        
        self._print_info("\n模拟数据库连接错误:")
        self._print_error("模拟错误: 无法连接到 PostgreSQL 数据库")
        self._print_info("系统自动切换到文件日志模式")
        
        self._print_info("\n模拟文件不存在错误:")
        self._print_error("模拟错误: 日志文件不存在或无法读取")
        self._print_info("系统提供友好的错误提示和解决建议")
        
        if interactive:
            input("\n按回车键完成演示...")
        else:
            import time
            time.sleep(2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="HarborAI 日志查看工具")
    parser.add_argument("--days", type=int, default=7, help="查看最近几天的日志 (默认: 7)")
    parser.add_argument("--model", help="过滤特定模型")
    parser.add_argument("--provider", help="过滤特定提供商")
    parser.add_argument("--source", choices=["auto", "postgres", "file"], default="auto", 
                       help="指定日志源 (默认: auto)")
    parser.add_argument("--limit", type=int, default=50, help="限制显示条数 (默认: 50)")
    parser.add_argument("--format", choices=["table", "json"], default="table", 
                       help="输出格式 (默认: table)")
    parser.add_argument("--stats", action="store_true", help="显示统计信息而不是日志列表")
    parser.add_argument("--type", choices=["all", "request", "response", "paired"], default="all",
                       help="日志类型过滤 (默认: all) - all: 显示所有日志, request: 仅请求日志, response: 仅响应日志, paired: 配对显示")
    parser.add_argument("--show-request-response-pairs", action="store_true", 
                       help="以配对方式显示请求-响应日志 (等同于 --type paired)")
    parser.add_argument("--layout", choices=["classic", "enhanced"], default="classic",
                       help="表格布局模式 (默认: classic) - classic: 传统布局, enhanced: 基于trace_id的增强布局")
    
    # Trace ID 相关参数
    parser.add_argument("--trace-id", help="根据指定的 trace_id 查询日志")
    parser.add_argument("--list-recent-trace-ids", action="store_true", 
                       help="列出最近的 trace_id 列表")
    parser.add_argument("--validate-trace-id", help="验证指定的 trace_id 格式是否正确")
    
    # 新增功能参数
    parser.add_argument("--demo", action="store_true", help="运行演示模式")
    parser.add_argument("--watch", action="store_true", help="实时监控日志")
    parser.add_argument("--watch-interval", type=int, default=5, help="监控刷新间隔(秒) (默认: 5)")
    parser.add_argument("--watch-duration", type=int, default=300, help="监控最大持续时间(秒) (默认: 300)")
    parser.add_argument("--search", help="搜索日志内容")
    parser.add_argument("--page", type=int, default=1, help="分页页码 (默认: 1)")
    parser.add_argument("--page-size", type=int, default=50, help="每页显示条数 (默认: 50)")
    parser.add_argument("--has-error", action="store_true", help="仅显示包含错误的日志")
    parser.add_argument("--no-error", action="store_true", help="仅显示无错误的日志")
    parser.add_argument("--min-tokens", type=int, help="最小token数过滤")
    parser.add_argument("--max-tokens", type=int, help="最大token数过滤")
    parser.add_argument("--config", help="指定配置文件路径")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    # 导出功能参数
    parser.add_argument("--export", help="导出日志到指定文件路径")
    parser.add_argument("--export-format", choices=["csv", "json", "excel"], 
                       help="导出格式 (csv, json, excel) - 如果未指定，将根据文件扩展名自动判断")
    
    # 时间范围过滤参数
    parser.add_argument("--start-time", help="开始时间，支持多种格式：\n"
                       "  - ISO格式: 2024-01-15T10:30:00 或 2024-01-15T10:30:00Z\n"
                       "  - 标准格式: 2024-01-15 10:30:00\n"
                       "  - 简化格式: 01-15 10:30 (当年)\n"
                       "  - 仅时间: 10:30 (当天)")
    parser.add_argument("--end-time", help="结束时间，支持格式同 --start-time")
    
    args = parser.parse_args()
    
    # 处理 --show-request-response-pairs 参数
    if args.show_request_response_pairs:
        args.type = "paired"
    
    # 创建配置管理器
    config_manager = ConfigManager(args.config) if args.config else ConfigManager()
    
    # 设置调试模式
    if args.debug:
        config_manager.set('general', 'debug', True)
    
    # 创建日志查看器
    viewer = LogViewer(config_manager)
    
    # 处理演示模式
    if args.demo:
        viewer.run_demo(interactive=False, layout=args.layout)  # 非交互模式
        sys.exit(0)
    
    # 处理实时监控
    if args.watch:
        viewer.watch_logs(args.watch_interval, args.watch_duration)
        sys.exit(0)
    
    # 处理 trace_id 验证
    if args.validate_trace_id:
        is_valid = viewer.validate_trace_id(args.validate_trace_id)
        if is_valid:
            print(f"[SUCCESS] trace_id '{args.validate_trace_id}' 格式正确")
        else:
            print(f"[ERROR] trace_id '{args.validate_trace_id}' 格式无效")
            print("[TIP] 正确格式: hb_<timestamp>_<random_string>")
            print("   例如: hb_1760458760039_257c901f")
        sys.exit(0)
    
    # 处理列出最近的 trace_id
    if args.list_recent_trace_ids:
        result = viewer.list_recent_trace_ids(days=args.days, limit=args.limit)
        if result["error"]:
            viewer._print_error(f"获取 trace_id 列表失败: {result['error']}")
            sys.exit(1)
        
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            output = viewer.format_recent_trace_ids(result["data"], result["source"])
            print(output)
        sys.exit(0)
    
    # 处理 trace_id 查询
    if args.trace_id:
        result = viewer.query_logs_by_trace_id(args.trace_id, args.format)
        if result["error"]:
            viewer._print_error(f"查询失败: {result['error']}")
            sys.exit(1)
        
        if args.format == "json":
            # 清理日志数据以优化JSON输出
            if result.get("data"):
                cleaned_logs = viewer._clean_logs_for_json_output(result["data"])
                result["data"] = cleaned_logs
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            if result["data"]:
                output = viewer.format_trace_id_logs(
                    result["data"], 
                    args.trace_id, 
                    args.format, 
                    result["source"]
                )
                print(output)
            else:
                message = result.get("message", f"未找到 trace_id '{args.trace_id}' 的日志记录")
                print(f"[ERROR] {message}")
                print("\n[TIP] 提示:")
                print("   1. 检查 trace_id 是否正确")
                print("   2. 尝试使用 --list-recent-trace-ids 查看可用的 trace_id")
                print("   3. 调整 --days 参数扩大搜索范围")
        sys.exit(0)
    
    # 检查可用性
    postgres_available = viewer.check_postgres_availability()
    file_available = viewer.check_file_logs_availability()
    
    if not postgres_available and not file_available:
        viewer._print_error("没有可用的日志源")
        sys.exit(1)
    
    # 显示可用性信息
    if args.format == "table":
        if postgres_available:
            viewer._print_success("PostgreSQL日志可用")
        else:
            viewer._print_warning("PostgreSQL日志不可用")
        
        if file_available:
            viewer._print_success("文件日志可用")
        else:
            viewer._print_warning("文件日志不可用")
    
    # 获取数据
    if args.stats:
        # 获取统计信息
        result = viewer.get_stats(
            days=args.days,
            provider=args.provider,
            model=args.model
        )
        
        if result["error"]:
            viewer._print_error(f"获取统计信息失败: {result['error']}")
            sys.exit(1)
        
        # 获取日志类型统计
        type_stats_result = viewer.get_log_type_stats(
            days=args.days,
            provider=args.provider,
            model=args.model
        )
        
        if args.format == "json":
            # 合并统计信息和类型统计
            combined_result = {
                "stats": result,
                "type_stats": type_stats_result
            }
            print(json.dumps(combined_result, ensure_ascii=False, indent=2, default=str))
        else:
            viewer.format_stats_table(result["data"], result["source"])
            
            # 显示日志类型统计
            if not type_stats_result["error"]:
                print()  # 空行分隔
                viewer._print_info("[STATS] 日志类型分布:")
                type_stats = type_stats_result["data"]
                
                if HAS_RICH:
                    type_table = Table(title="日志类型分布", show_header=True, header_style="bold magenta")
                    type_table.add_column("类型", style="cyan", width=10)
                    type_table.add_column("数量", style="green", width=10)
                    type_table.add_column("占比", style="yellow", width=10)
                    
                    total = type_stats["total"]
                    for log_type in ["request", "response", "unknown"]:
                        count = type_stats[log_type]
                        percentage = (count / total * 100) if total > 0 else 0
                        type_table.add_row(
                            log_type.upper(),
                            str(count),
                            f"{percentage:.1f}%"
                        )
                    
                    type_table.add_row("TOTAL", str(total), "100.0%", style="bold")
                    console.print(type_table)
                else:
                    # 简单文本格式
                    print("类型      数量      占比")
                    print("-" * 25)
                    total = type_stats["total"]
                    for log_type in ["request", "response", "unknown"]:
                        count = type_stats[log_type]
                        percentage = (count / total * 100) if total > 0 else 0
                        print(f"{log_type.upper():<8} {count:<8} {percentage:.1f}%")
                    print("-" * 25)
                    print(f"TOTAL    {total:<8} 100.0%")
    else:
        # 解析时间参数
        start_time = None
        end_time = None
        
        if args.start_time:
            start_time = viewer.parse_time_string(args.start_time)
            if start_time is None:
                viewer._print_error(f"开始时间格式错误: 无法解析时间字符串 '{args.start_time}'")
                viewer._print_error("支持的时间格式:")
                viewer._print_error("  - ISO格式: 2024-01-15T10:30:00 或 2024-01-15T10:30:00Z")
                viewer._print_error("  - 标准格式: 2024-01-15 10:30:00")
                viewer._print_error("  - 简化格式: 01-15 10:30 (当年)")
                viewer._print_error("  - 仅时间: 10:30 (当天)")
                sys.exit(1)
            viewer._print_info(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if args.end_time:
            end_time = viewer.parse_time_string(args.end_time)
            if end_time is None:
                viewer._print_error(f"结束时间格式错误: 无法解析时间字符串 '{args.end_time}'")
                viewer._print_error("支持的时间格式:")
                viewer._print_error("  - ISO格式: 2024-01-15T10:30:00 或 2024-01-15T10:30:00Z")
                viewer._print_error("  - 标准格式: 2024-01-15 10:30:00")
                viewer._print_error("  - 简化格式: 01-15 10:30 (当年)")
                viewer._print_error("  - 仅时间: 10:30 (当天)")
                sys.exit(1)
            viewer._print_info(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 验证时间范围
        if start_time or end_time:
            try:
                viewer.validate_time_range(start_time, end_time)
            except ValueError as e:
                viewer._print_error(f"时间范围无效: {e}")
                sys.exit(1)
        
        # 获取日志
        source = None if args.source == "auto" else args.source
        
        # 如果使用增强型布局，需要获取所有类型的日志进行配对
        log_type = "all" if args.layout == "enhanced" else args.type
        
        result = viewer.get_logs(
            source=source,
            start_time=start_time,
            end_time=end_time,
            days=args.days,
            model=args.model,
            provider=args.provider,
            limit=args.limit,
            log_type=log_type
        )
        
        if result["error"]:
            viewer._print_error(f"获取日志失败: {result['error']}")
            sys.exit(1)
        
        logs = result["data"]
        
        # 应用搜索过滤
        if args.search:
            logs = viewer.search_logs(logs, args.search)
            viewer._print_info(f"搜索 '{args.search}' 找到 {len(logs)} 条记录")
        
        # 应用高级过滤
        filters = {}
        if args.has_error:
            filters['has_error'] = True
        if args.no_error:
            filters['no_error'] = True
        if args.min_tokens:
            filters['min_tokens'] = args.min_tokens
        if args.max_tokens:
            filters['max_tokens'] = args.max_tokens
        
        if filters:
            logs = viewer.filter_logs(logs, filters)
            filter_desc = ", ".join([f"{k}={v}" for k, v in filters.items()])
            viewer._print_info(f"应用过滤条件 ({filter_desc}) 后剩余 {len(logs)} 条记录")
        
        # 应用分页
        if args.page_size != args.limit or args.page != 1:
            paginated_logs, pagination_info = viewer.paginate_logs(logs, args.page, args.page_size)
            logs = paginated_logs
            
            # 显示分页信息
            if args.format == "table":
                viewer._print_info(
                    f"第 {pagination_info['current_page']}/{pagination_info['total_pages']} 页 "
                    f"(共 {pagination_info['total_count']} 条记录)"
                )
        
        # 处理导出功能
        if args.export:
            try:
                export_result = viewer.export_logs(logs, args.export, args.export_format)
                if export_result["success"]:
                    viewer._print_success(f"导出成功: {export_result['file_path']}")
                    viewer._print_info(f"导出记录数: {export_result['record_count']}")
                    viewer._print_info(f"导出格式: {export_result['format']}")
                    if export_result.get('summary'):
                        summary = export_result['summary']
                        viewer._print_info(f"统计信息: 总请求 {summary.get('total_requests', 0)}, "
                                         f"成功 {summary.get('success_count', 0)}, "
                                         f"错误 {summary.get('error_count', 0)}")
                else:
                    viewer._print_error(f"导出失败: {export_result['error']}")
                    sys.exit(1)
            except Exception as e:
                viewer._print_error(f"导出过程中发生错误: {e}")
                sys.exit(1)
            
            # 如果只是导出，不显示表格
            if not args.format:
                return
        
        if args.format == "json":
            # 清理日志数据以优化JSON输出
            cleaned_logs = viewer._clean_logs_for_json_output(logs)
            
            output_data = {
                "logs": cleaned_logs,
                "source": result["source"],
                "total_count": len(cleaned_logs),
                "filters_applied": bool(args.search or filters),
                "pagination": pagination_info if 'pagination_info' in locals() else None
            }
            print(json.dumps(output_data, ensure_ascii=False, indent=2, default=str))
        else:
            if not logs:
                viewer._print_warning("没有找到符合条件的日志")
            else:
                viewer.format_logs_table(logs, result["source"], args.layout)
                
                # 显示总计信息
                if 'pagination_info' in locals():
                    if pagination_info['has_next'] or pagination_info['has_prev']:
                        viewer._print_info(f"使用 --page {pagination_info['current_page'] + 1} 查看下一页")
                elif result.get("total_count", 0) > len(logs):
                    viewer._print_info(f"显示 {len(logs)} 条记录，共 {result['total_count']} 条")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)