#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件日志解析器

当 PostgreSQL 不可用时，解析文件系统中的日志文件，为 CLI 工具提供数据查询功能。
"""

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterator
from collections import defaultdict

from ..config.settings import get_settings
from ..utils.logger import get_logger
from ..core.pricing import PricingCalculator

logger = get_logger(__name__)


class QueryResult:
    """查询结果封装（与 postgres_client.py 中的定义保持一致）"""
    
    def __init__(self, data: List[Dict[str, Any]], total_count: int, source: str, error: Optional[str] = None):
        self.data = data
        self.total_count = total_count
        self.source = source
        self.error = error


class FileLogParser:
    """文件日志解析器
    
    解析 HarborAI 的文件日志，提供与 PostgreSQL 客户端兼容的查询接口。
    """
    
    def __init__(self, log_directory: Optional[str] = None):
        """初始化文件日志解析器
        
        Args:
            log_directory: 日志文件目录，如果为 None 则从配置获取
        """
        self.settings = get_settings()
        self.log_directory = Path(log_directory or self._get_default_log_directory())
        
        # 确保日志目录存在，如果不存在则自动创建
        self._ensure_log_directory_exists()
    
    def _get_default_log_directory(self) -> str:
        """获取默认日志目录"""
        # 从设置中获取文件日志目录
        if hasattr(self.settings, 'file_log_directory'):
            return self.settings.file_log_directory
        
        # 默认日志目录（项目根目录下的logs文件夹）
        return "./logs"
    
    def _ensure_log_directory_exists(self) -> None:
        """确保日志目录存在，如果不存在则自动创建"""
        try:
            if not self.log_directory.exists():
                logger.info(f"创建日志目录: {self.log_directory}")
                self.log_directory.mkdir(parents=True, exist_ok=True)
                
                # 检查目录是否可写
                if not os.access(self.log_directory, os.W_OK):
                    logger.error(f"日志目录不可写: {self.log_directory}")
                    raise PermissionError(f"日志目录不可写: {self.log_directory}")
                    
                logger.info(f"日志目录创建成功: {self.log_directory}")
            else:
                # 目录存在，检查是否可写
                if not os.access(self.log_directory, os.W_OK):
                    logger.warning(f"日志目录不可写: {self.log_directory}")
                    
        except (OSError, PermissionError) as e:
            logger.error(f"无法创建或访问日志目录 {self.log_directory}: {e}")
            logger.error("建议解决方案:")
            logger.error(f"1. 检查目录权限: {self.log_directory.parent}")
            logger.error("2. 手动创建目录或使用其他位置")
            logger.error("3. 设置环境变量 HARBORAI_LOG_DIR 指定其他目录")
            # 不抛出异常，允许程序继续运行，但日志功能可能受限
    
    def _find_log_files(self, days: int) -> List[Path]:
        """查找指定天数内的日志文件
        
        Args:
            days: 查找最近几天的日志文件
            
        Returns:
            List[Path]: 日志文件路径列表
        """
        if not self.log_directory.exists():
            return []
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        log_files = []
        
        # 查找所有 .log 和 .jsonl 文件
        for log_file in self.log_directory.glob("**/*.log"):
            try:
                # 检查文件修改时间
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime >= cutoff_date:
                    log_files.append(log_file)
            except (OSError, ValueError) as e:
                logger.warning(f"无法检查日志文件 {log_file}: {e}")
        
        # 查找所有 .jsonl 文件
        for log_file in self.log_directory.glob("**/*.jsonl"):
            try:
                # 检查文件修改时间
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime >= cutoff_date:
                    log_files.append(log_file)
            except (OSError, ValueError) as e:
                logger.warning(f"无法检查日志文件 {log_file}: {e}")
        
        return sorted(log_files, key=lambda f: f.stat().st_mtime, reverse=True)
    
    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析单行日志
        
        Args:
            line: 日志行内容
            
        Returns:
            Optional[Dict[str, Any]]: 解析后的日志数据，如果解析失败返回 None
        """
        line = line.strip()
        if not line:
            return None
        
        try:
            # 尝试解析 JSON 格式的日志
            if line.startswith('{') and line.endswith('}'):
                log_data = json.loads(line)
                return self._normalize_log_data(log_data)
            
            # 尝试解析结构化文本日志
            return self._parse_structured_text_log(line)
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"无法解析日志行: {line[:100]}... 错误: {e}")
            return None
    
    def _normalize_log_data(self, log_data: Dict[str, Any]) -> Dict[str, Any]:
        """标准化日志数据格式
        
        Args:
            log_data: 原始日志数据
            
        Returns:
            Dict[str, Any]: 标准化后的日志数据
        """
        # 标准化时间戳
        timestamp = log_data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    # 尝试解析 ISO 格式时间戳
                    log_data['timestamp'] = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    # 尝试其他时间格式
                    try:
                        log_data['timestamp'] = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        logger.warning(f"无法解析时间戳: {timestamp}")
                        log_data['timestamp'] = None
        
        # 处理 HarborAI 日志格式
        log_type = log_data.get('type', 'unknown')
        
        # 提取模型名称
        # 优先从顶级 model 字段提取模型名称
        model = log_data.get('model', 'unknown')
        
        # 如果顶级没有找到，尝试从其他位置提取
        if model == 'unknown':
            if log_type == 'response':
                # 从 response_summary.model 提取模型名称
                response_summary = log_data.get('response_summary', {})
                if isinstance(response_summary, dict):
                    model = response_summary.get('model', 'unknown')
        
        # 提取提供商信息 - 优先使用日志中的provider字段
        provider = log_data.get('provider', 'unknown')
        
        # 只有当provider字段为'unknown'时，才根据模型名称推断
        if provider == 'unknown':
            if 'ernie' in model.lower():
                provider = 'baidu'
            elif 'doubao' in model.lower():
                provider = 'bytedance'
            elif 'deepseek' in model.lower():
                provider = 'deepseek'
            elif 'gpt' in model.lower() or 'openai' in model.lower():
                provider = 'openai'
            elif 'claude' in model.lower():
                provider = 'anthropic'
            elif 'gemini' in model.lower():
                provider = 'google'
        
        # 提取成功状态
        success = log_data.get('success')
        status_code = None
        if success is True:
            status_code = 200
        elif success is False:
            status_code = 500
        
        # 提取 token 信息
        tokens = log_data.get('tokens', {})
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        
        if isinstance(tokens, dict):
            # 支持多种Token字段格式
            # 优先使用原始字段名，然后尝试标准字段名
            prompt_tokens = (tokens.get('input', 0) or 
                           tokens.get('prompt_tokens', 0) or 0)
            completion_tokens = (tokens.get('output', 0) or 
                               tokens.get('completion_tokens', 0) or 0)
            total_tokens = (tokens.get('total', 0) or 
                          tokens.get('total_tokens', 0) or 0)
        
        # 提取成本信息
        cost_data = log_data.get('cost', 0.0)
        cost = 0.0
        
        if isinstance(cost_data, dict):
            # 如果cost是对象，提取total_cost
            cost = cost_data.get('total_cost', 0.0) or 0.0
        elif isinstance(cost_data, (int, float)):
            # 如果cost是数字，直接使用
            cost = float(cost_data) if cost_data else 0.0
        
        if cost == 0.0:
            # 如果日志中没有成本信息或成本为0，使用PricingCalculator计算
            if prompt_tokens > 0 and completion_tokens > 0 and model != 'unknown':
                calculated_cost = PricingCalculator.calculate_cost(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    model_name=model
                )
                if calculated_cost is not None:
                    cost = calculated_cost
        
        # 提取延迟信息
        duration_ms = log_data.get('latency', 0.0)
        if duration_ms is None:
            duration_ms = 0.0
        # 转换为毫秒
        if duration_ms < 1:  # 如果小于1，可能是秒为单位
            duration_ms = duration_ms * 1000
        
        # 提取错误信息
        error_message = log_data.get('error')
        if error_message and isinstance(error_message, dict):
            error_message = str(error_message)
        
        # 更新标准化字段
        log_data.update({
            'provider': provider,
            'model': model,
            'status_code': status_code,
            'duration_ms': duration_ms,
            'error_message': error_message,
            'success': success,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'cost': cost,
            'type': log_type
        })
        
        return log_data
    
    def _parse_structured_text_log(self, line: str) -> Optional[Dict[str, Any]]:
        """解析结构化文本日志
        
        Args:
            line: 日志行内容
            
        Returns:
            Optional[Dict[str, Any]]: 解析后的日志数据
        """
        # 简单的正则表达式解析（可根据实际日志格式调整）
        pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+(\w+)\s+(.+)'
        match = re.match(pattern, line)
        
        if match:
            timestamp_str, provider, model, message = match.groups()
            
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                timestamp = None
            
            return {
                'timestamp': timestamp,
                'provider': provider,
                'model': model,
                'message': message,
                'status_code': None,
                'duration_ms': None,
                'error_message': None
            }
        
        return None
    
    def _filter_logs(
        self,
        logs: List[Dict[str, Any]],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """过滤日志数据
        
        Args:
            logs: 日志数据列表
            model: 过滤特定模型
            provider: 过滤特定提供商
            days: 过滤最近几天的日志
            
        Returns:
            List[Dict[str, Any]]: 过滤后的日志数据
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        filtered_logs = []
        
        for log in logs:
            # 时间过滤
            log_timestamp = log.get('timestamp')
            if log_timestamp:
                # 确保时间戳有时区信息
                if log_timestamp.tzinfo is None:
                    log_timestamp = log_timestamp.replace(tzinfo=timezone.utc)
                if log_timestamp < cutoff_date:
                    continue
            
            # 模型过滤
            if model and log.get('model') != model:
                continue
            
            # 提供商过滤
            if provider and log.get('provider') != provider:
                continue
            
            filtered_logs.append(log)
        
        return filtered_logs
    
    def query_api_logs(
        self,
        days: int = 7,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        limit: int = 50,
        log_type: str = "response"
    ) -> QueryResult:
        """查询 API 日志
        
        Args:
            days: 查询最近几天的日志
            model: 过滤特定模型
            provider: 过滤特定提供商
            limit: 限制返回条数
            log_type: 日志类型过滤 ("all", "request", "response", "paired")
            
        Returns:
            QueryResult: 查询结果
        """
        try:
            log_files = self._find_log_files(days)
            if not log_files:
                return QueryResult(
                    data=[],
                    total_count=0,
                    source='file',
                    error="未找到日志文件"
                )
            
            all_logs = []
            
            # 解析所有日志文件
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            log_data = self._parse_log_line(line)
                            if log_data:
                                # 根据 log_type 参数过滤日志
                                current_log_type = log_data.get('type', 'unknown')
                                if log_type == "all" or log_type == current_log_type or log_type == "paired":
                                    all_logs.append(log_data)
                except (IOError, UnicodeDecodeError) as e:
                    logger.warning(f"无法读取日志文件 {log_file}: {e}")
            
            # 过滤日志
            filtered_logs = self._filter_logs(all_logs, model, provider, days)
            
            # 处理 paired 类型：按 trace_id 配对并排序
            if log_type == "paired":
                filtered_logs = self._create_paired_logs(filtered_logs)
            
            # 按时间戳排序（最新的在前）
            filtered_logs.sort(key=lambda x: x.get('timestamp') or datetime.min, reverse=True)
            
            # 限制返回条数
            limited_logs = filtered_logs[:limit]
            
            return QueryResult(
                data=limited_logs,
                total_count=len(limited_logs),
                source='file'
            )
            
        except Exception as e:
            logger.error(f"查询文件日志失败: {e}")
            return QueryResult(
                data=[],
                total_count=0,
                source='file',
                error=str(e)
            )
    
    def _create_paired_logs(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """创建配对的请求-响应日志
        
        Args:
            logs: 原始日志列表
            
        Returns:
            List[Dict[str, Any]]: 配对后的日志列表
        """
        # 按 trace_id 分组
        trace_groups = defaultdict(list)
        for log in logs:
            trace_id = log.get('trace_id')
            if trace_id:
                trace_groups[trace_id].append(log)
        
        # 创建配对的日志
        paired_logs = []
        for trace_id, group_logs in trace_groups.items():
            # 按类型分组
            request_logs = [log for log in group_logs if log.get('type') == 'request']
            response_logs = [log for log in group_logs if log.get('type') == 'response']
            
            # 如果有配对的请求和响应，按时间顺序添加
            if request_logs and response_logs:
                # 取最新的请求和响应
                request_log = max(request_logs, key=lambda x: x.get('timestamp', datetime.min))
                response_log = max(response_logs, key=lambda x: x.get('timestamp', datetime.min))
                
                # 先添加请求，再添加响应
                paired_logs.append(request_log)
                paired_logs.append(response_log)
            else:
                # 如果没有配对，添加所有日志
                paired_logs.extend(group_logs)
        
        return paired_logs
    
    def query_model_usage(
        self,
        days: int = 30,
        provider: Optional[str] = None,
        model: Optional[str] = None
    ) -> QueryResult:
        """查询模型使用统计
        
        Args:
            days: 查询最近几天的统计
            provider: 过滤特定提供商
            model: 过滤特定模型
            
        Returns:
            QueryResult: 查询结果
        """
        try:
            # 先获取所有日志
            api_logs_result = self.query_api_logs(days, model, provider, limit=10000)
            
            if api_logs_result.error:
                return QueryResult(
                    data=[],
                    total_count=0,
                    source='file',
                    error=api_logs_result.error
                )
            
            # 聚合统计
            usage_stats = defaultdict(lambda: {
                'provider': '',
                'model': '',
                'request_count': 0,
                'success_count': 0,
                'error_count': 0,
                'total_duration': 0,
                'avg_duration': 0,
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'total_cost': 0.0
            })
            
            for log in api_logs_result.data:
                # 只统计 response 类型的日志，避免重复计算
                if log.get('type') != 'response':
                    continue
                    
                key = (log.get('provider', 'unknown'), log.get('model', 'unknown'))
                stats = usage_stats[key]
                
                stats['provider'] = log.get('provider', 'unknown')
                stats['model'] = log.get('model', 'unknown')
                stats['request_count'] += 1
                
                # 统计成功和失败
                success = log.get('success')
                if success is True:
                    stats['success_count'] += 1
                elif success is False:
                    stats['error_count'] += 1
                
                # 统计响应时间
                duration = log.get('duration_ms', 0)
                if duration:
                    stats['total_duration'] += duration
                
                # 统计 token 使用量
                total_tokens = log.get('total_tokens', 0) or 0
                prompt_tokens = log.get('prompt_tokens', 0) or 0
                completion_tokens = log.get('completion_tokens', 0) or 0
                
                stats['total_tokens'] += total_tokens
                stats['prompt_tokens'] += prompt_tokens
                stats['completion_tokens'] += completion_tokens
                
                # 统计成本 - 使用PricingCalculator计算实际成本
                cost = log.get('cost', 0.0) or 0.0
                
                # 处理字典类型的成本数据
                if isinstance(cost, dict):
                    cost = cost.get('total_cost', cost.get('total', 0.0))
                
                # 确保成本是数字类型
                if not isinstance(cost, (int, float)):
                    cost = 0.0
                
                if cost == 0.0 and prompt_tokens > 0 and completion_tokens > 0:
                    # 如果日志中没有成本信息，使用PricingCalculator计算
                    model_name = log.get('model', '')
                    calculated_cost = PricingCalculator.calculate_cost(
                        input_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                        model_name=model_name
                    )
                    if calculated_cost is not None:
                        cost = calculated_cost
                
                stats['total_cost'] += cost
            
            # 计算平均响应时间
            result_data = []
            for stats in usage_stats.values():
                if stats['request_count'] > 0:
                    stats['avg_duration'] = stats['total_duration'] / stats['request_count']
                result_data.append(stats)
            
            # 按请求数量排序
            result_data.sort(key=lambda x: x['request_count'], reverse=True)
            
            return QueryResult(
                data=result_data,
                total_count=len(result_data),
                source='file'
            )
            
        except Exception as e:
            logger.error(f"查询模型使用统计失败: {e}")
            return QueryResult(
                data=[],
                total_count=0,
                source='file',
                error=str(e)
            )