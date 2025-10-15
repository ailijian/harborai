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
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

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
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    console = None


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


class LogViewer:
    """日志查看器类"""
    
    def __init__(self):
        self.postgres_client = None
        self.file_parser = None
        self.file_logger = None
        self.postgres_conn = None
        self._init_clients()
        self._init_trace_id_support()
    
    def _init_clients(self):
        """初始化客户端"""
        try:
            self.postgres_client = get_postgres_client()
        except Exception as e:
            self._print_warning(f"PostgreSQL客户端初始化失败: {e}")
        
        try:
            log_dir = self._get_log_directory()
            self.file_parser = FileLogParser(str(log_dir))
        except Exception as e:
            self._print_warning(f"文件日志解析器初始化失败: {e}")
    
    def _get_log_directory(self) -> Path:
        """获取日志目录路径"""
        try:
            settings = get_settings()
            if hasattr(settings, 'file_log_directory'):
                return Path(settings.file_log_directory)
        except Exception:
            pass
        
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
                # 不在这里打印成功信息，避免干扰正常输出
            else:
                # 不在这里打印警告信息，避免干扰正常输出
                pass
        except Exception as e:
            # 不在这里打印错误信息，避免干扰正常输出
            pass
    
    def _init_postgres_connection_for_trace_id(self):
        """初始化 PostgreSQL 连接（用于 trace_id 查询）"""
        if not POSTGRES_AVAILABLE:
            return
        
        try:
            # 构建连接字符串
            conn_str = self._build_postgres_connection_string()
            if conn_str:
                self.postgres_conn = psycopg2.connect(conn_str)
                # 不在这里打印成功信息，避免干扰正常输出
            else:
                # 不在这里打印警告信息，避免干扰正常输出
                pass
        except Exception as e:
            # 不在这里打印警告信息，避免干扰正常输出
            pass
    
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
                         log_type: str = "response") -> Dict[str, Any]:
        """获取PostgreSQL日志"""
        if not self.postgres_client:
            return {"error": "PostgreSQL客户端不可用", "data": [], "source": "error"}
        
        try:
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
    
    def get_file_logs(self, days: int = 7, model: Optional[str] = None, 
                     provider: Optional[str] = None, limit: int = 50, 
                     log_type: str = "response") -> Dict[str, Any]:
        """获取文件日志"""
        if not self.file_parser:
            return {"error": "文件日志解析器不可用", "data": [], "source": "error"}
        
        try:
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
    
    def get_logs(self, source: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """获取日志（自动选择源或指定源）"""
        if source == "postgres":
            return self.get_postgres_logs(**kwargs)
        elif source == "file":
            return self.get_file_logs(**kwargs)
        else:
            # 自动选择：优先PostgreSQL，降级到文件
            if self.check_postgres_availability():
                return self.get_postgres_logs(**kwargs)
            elif self.check_file_logs_availability():
                return self.get_file_logs(**kwargs)
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
        
        # HarborAI trace_id 格式：hb_timestamp_randomstring
        # 支持十六进制字符和字母数字字符
        pattern = r'^hb_\d+_[a-z0-9]+$'
        return bool(re.match(pattern, trace_id))
    
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
                
                enhanced_log = {
                    'Trace ID': short_trace_id,
                    '请求时间': request_time_str,
                    '响应时间': response_time_str,
                    '耗时(ms)': duration_ms,
                    '模型': model,
                    '提供商': response_log.get('provider', 'unknown'),
                    '状态': status,
                    'Token': response_log.get('tokens', 'N/A'),
                    '成本': response_log.get('cost', 'N/A')
                }
                enhanced_logs.append(enhanced_log)
        
        return enhanced_logs
    
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
                table.add_column("请求时间", style="blue", width=14)
                table.add_column("响应时间", style="green", width=14)
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
                    
                    # 处理成本
                    cost = log.get('成本', 'N/A')
                    cost_display = f"¥{cost:.4f}" if cost != 'N/A' and isinstance(cost, (int, float)) else str(cost)
                    
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
                    
                    # 处理Token数
                    tokens = log.get('total_tokens')
                    if not tokens and log.get('tokens'):
                        # 如果 tokens 是字典，提取 total_tokens
                        tokens_dict = log.get('tokens')
                        if isinstance(tokens_dict, dict):
                            tokens = tokens_dict.get('total_tokens')
                        else:
                            tokens = tokens_dict
                    tokens_display = f"{tokens:,}" if tokens and isinstance(tokens, (int, float)) else "N/A"
                    
                    # 处理耗时
                    duration = log.get('duration_ms') or log.get('latency')
                    duration_display = f"{duration:.1f}" if duration else "N/A"
                    
                    # 处理成本
                    cost = log.get('estimated_cost') or log.get('cost')
                    if isinstance(cost, dict):
                        # 如果成本是字典，尝试获取总成本
                        cost = cost.get('total_cost') or cost.get('total') or 0.0
                    cost_display = f"¥{cost:.4f}" if cost and isinstance(cost, (int, float)) else "N/A"
                    
                    # 处理日志类型并添加视觉区分
                    log_type = log.get('type', 'unknown')
                    if log_type == 'request':
                        type_display = "[blue]REQ[/blue]"
                    elif log_type == 'response':
                        type_display = "[green]RES[/green]"
                    else:
                        type_display = "[yellow]UNK[/yellow]"
                    
                    # 处理 trace_id
                    trace_id = log.get('trace_id', 'N/A')
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
                
                tokens = log.get('total_tokens') or log.get('tokens', 'N/A')
                cost = log.get('estimated_cost') or log.get('cost', 'N/A')
                
                # 处理日志类型
                log_type = log.get('type', 'unknown')
                if log_type == 'request':
                    type_display = "REQ"
                elif log_type == 'response':
                    type_display = "RES"
                else:
                    type_display = "UNK"
                
                # 处理 trace_id
                trace_id = log.get('trace_id', 'N/A')
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
            for key, value in first_log.items():
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
        
        return "\n".join(output)
    
    def _format_trace_id_logs_json(self, logs: List[Dict[str, Any]], 
                                  trace_id: str, source: str) -> str:
        """以 JSON 格式显示 trace_id 查询结果"""
        result = {
            "trace_id": trace_id,
            "source": source,
            "total_count": len(logs),
            "logs": logs
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
        
        # 显示总计
        if HAS_RICH and console:
            summary_panel = Panel(
                f"总请求数: {total_requests:,}\n"
                f"成功请求: {total_success:,}\n"
                f"成功率: {success_rate:.1f}%\n"
                f"总Token数: {total_tokens:,}\n"
                f"总成本: ¥{total_cost:.4f}",
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
            
            for stat in stats:
                request_count = stat.get('request_count', 0)
                success_count = stat.get('success_count', 0)
                success_rate = (success_count / request_count * 100) if request_count > 0 else 0
                
                table.add_row(
                    stat.get('model', 'N/A'),
                    stat.get('provider', 'N/A'),
                    f"{request_count:,}",
                    f"{success_count:,}",
                    f"{success_rate:.1f}%",
                    f"{stat.get('total_tokens', 0):,}",
                    f"¥{stat.get('total_cost', 0.0):.4f}"
                )
            
            console.print(table)
        else:
            # 简单文本格式
            print(f"总请求数: {total_requests:,}")
            print(f"成功请求: {total_success:,}")
            print(f"成功率: {success_rate:.1f}%")
            print(f"总Token数: {total_tokens:,}")
            print(f"总成本: ¥{total_cost:.4f}")
            print()
            
            print(f"{'模型':<20} {'提供商':<10} {'请求数':<8} {'成功数':<8} {'成功率':<8} {'Token':<10} {'成本':<10}")
            print("-" * 80)
            
            for stat in stats:
                request_count = stat.get('request_count', 0)
                success_count = stat.get('success_count', 0)
                success_rate = (success_count / request_count * 100) if request_count > 0 else 0
                
                print(f"{stat.get('model', 'N/A'):<20} "
                      f"{stat.get('provider', 'N/A'):<10} "
                      f"{request_count:<8} "
                      f"{success_count:<8} "
                      f"{success_rate:.1f}%{'':<3} "
                      f"{stat.get('total_tokens', 0):<10} "
                      f"¥{stat.get('total_cost', 0.0):.4f}")


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
    
    args = parser.parse_args()
    
    # 处理 --show-request-response-pairs 参数
    if args.show_request_response_pairs:
        args.type = "paired"
    
    # 创建日志查看器
    viewer = LogViewer()
    
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
        # 获取日志
        source = None if args.source == "auto" else args.source
        
        # 如果使用增强型布局，需要获取所有类型的日志进行配对
        log_type = "all" if args.layout == "enhanced" else args.type
        
        result = viewer.get_logs(
            source=source,
            days=args.days,
            model=args.model,
            provider=args.provider,
            limit=args.limit,
            log_type=log_type
        )
        
        if result["error"]:
            viewer._print_error(f"获取日志失败: {result['error']}")
            sys.exit(1)
        
        if args.format == "json":
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            viewer.format_logs_table(result["data"], result["source"], args.layout)
            
            # 显示总计信息
            if result.get("total_count", 0) > len(result["data"]):
                viewer._print_info(f"显示 {len(result['data'])} 条记录，共 {result['total_count']} 条")


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