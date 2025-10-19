#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL 查询客户端

为 CLI 工具提供 PostgreSQL 数据库查询功能，支持自动降级到文件日志解析。
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from ..config.settings import get_settings
from ..utils.logger import get_logger
from .models import APILog, TraceLog, ModelUsage

logger = get_logger(__name__)


@dataclass
class QueryResult:
    """查询结果封装"""
    data: List[Dict[str, Any]]
    total_count: int
    source: str  # 'postgresql' 或 'file'
    error: Optional[str] = None


class PostgreSQLClient:
    """PostgreSQL 查询客户端
    
    提供与 SQLite 兼容的查询接口，支持 CLI 工具的数据查询需求。
    当 PostgreSQL 不可用时，自动降级到文件日志解析。
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """初始化 PostgreSQL 客户端
        
        Args:
            connection_string: PostgreSQL 连接字符串，如果为 None 则从配置获取
        """
        self.settings = get_settings()
        self.connection_string = connection_string or self._build_connection_string()
        self._connection = None
        self._available = False
        
        # 尝试连接 PostgreSQL
        self._test_connection()
    
    def _build_connection_string(self) -> str:
        """构建 PostgreSQL 连接字符串"""
        if self.settings.postgres_url:
            # 将 asyncpg 格式转换为 psycopg2 格式
            url = self.settings.postgres_url
            if url.startswith("postgresql+asyncpg://"):
                url = url.replace("postgresql+asyncpg://", "postgresql://")
            return url
        
        return (
            f"postgresql://{self.settings.postgres_user}:{self.settings.postgres_password}"
            f"@{self.settings.postgres_host}:{self.settings.postgres_port}"
            f"/{self.settings.postgres_database}"
        )
    
    def _test_connection(self) -> bool:
        """测试 PostgreSQL 连接"""
        if not PSYCOPG2_AVAILABLE:
            logger.warning("psycopg2 未安装，PostgreSQL 查询功能不可用")
            return False
        
        try:
            conn = psycopg2.connect(self.connection_string)
            conn.close()
            self._available = True
            logger.info("PostgreSQL 连接测试成功")
            return True
        except Exception as e:
            logger.warning(f"PostgreSQL 连接失败，将使用文件降级: {e}")
            self._available = False
            return False
    
    def _get_connection(self):
        """获取数据库连接"""
        if not self._available:
            return None
        
        try:
            if self._connection is None or self._connection.closed:
                self._connection = psycopg2.connect(
                    self.connection_string,
                    cursor_factory=RealDictCursor
                )
            return self._connection
        except Exception as e:
            logger.error(f"获取 PostgreSQL 连接失败: {e}")
            self._available = False
            return None
    
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
        # 尝试从 PostgreSQL 查询
        if self._available:
            try:
                return self._query_api_logs_from_postgres(days, model, provider, limit, log_type)
            except Exception as e:
                logger.error(f"PostgreSQL 查询失败: {e}")
                self._available = False
        
        # 降级到文件日志解析
        logger.info("降级到文件日志解析")
        return self._query_api_logs_from_files(days, model, provider, limit, log_type)
    
    def _query_api_logs_from_postgres(
        self,
        days: int,
        model: Optional[str],
        provider: Optional[str],
        limit: int,
        log_type: str = "response"
    ) -> QueryResult:
        """从 PostgreSQL 查询 API 日志"""
        conn = self._get_connection()
        if not conn:
            raise Exception("无法获取 PostgreSQL 连接")
        
        # 构建查询条件 - 匹配harborai_logs表结构
        where_conditions = ["timestamp >= %s"]
        params = [datetime.utcnow() - timedelta(days=days)]
        
        if model:
            where_conditions.append("model = %s")
            params.append(model)
        
        if provider:
            where_conditions.append("structured_provider = %s")
            params.append(provider)
        
        # 根据 log_type 添加类型过滤
        if log_type == "request":
            where_conditions.append("type = %s")
            params.append("request")
        elif log_type == "response":
            where_conditions.append("type = %s")
            params.append("response")
        # log_type == "all" 或 "paired" 时不添加类型过滤
        
        where_clause = " AND ".join(where_conditions)
        
        # 根据 log_type 调整查询字段 - 匹配harborai_logs表结构
        if log_type == "request":
            # 只查询请求相关字段
            select_fields = """
                trace_id,
                timestamp,
                structured_provider as provider,
                model,
                messages as request_data,
                parameters,
                'request' as type
            """
        else:
            # 查询所有字段（response, all, paired）
            select_fields = """
                trace_id,
                timestamp,
                structured_provider as provider,
                model,
                messages as request_data,
                response_summary as response_data,
                CASE WHEN success THEN 200 ELSE 500 END as status_code,
                error as error_message,
                latency as duration_ms,
                success,
                type,
                tokens,
                cost
            """
        
        # 执行查询
        query = f"""
            SELECT {select_fields}
            FROM harborai_logs 
            WHERE {where_clause}
            ORDER BY timestamp DESC 
            LIMIT %s
        """
        params.append(limit)
        
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # 转换为字典列表
            data = []
            for row in rows:
                log_data = dict(row)
                # 解析 JSON 字段 - 适配harborai_logs表结构
                if log_data.get('request_data'):
                    try:
                        log_data['request_data'] = json.loads(log_data['request_data'])
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                if log_data.get('response_data'):
                    try:
                        log_data['response_data'] = json.loads(log_data['response_data'])
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                if log_data.get('parameters'):
                    try:
                        log_data['parameters'] = json.loads(log_data['parameters'])
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                # 处理 paired 类型：为每个记录创建 request 和 response 两条记录
                if log_type == "paired":
                    # 创建 request 记录
                    request_data = log_data.copy()
                    request_data['type'] = 'request'
                    # 移除响应相关字段
                    request_data.pop('response_data', None)
                    request_data.pop('status_code', None)
                    request_data.pop('error_message', None)
                    request_data.pop('duration_ms', None)
                    data.append(request_data)
                    
                    # 创建 response 记录
                    response_data = log_data.copy()
                    response_data['type'] = 'response'
                    data.append(response_data)
                else:
                    data.append(log_data)
        
        return QueryResult(
            data=data,
            total_count=len(data),
            source='postgresql'
        )
    
    def _query_api_logs_from_files(
        self,
        days: int,
        model: Optional[str],
        provider: Optional[str],
        limit: int,
        log_type: str = "response"
    ) -> QueryResult:
        """从文件日志解析 API 日志"""
        # 这里需要导入文件日志解析器
        from .file_log_parser import FileLogParser
        
        parser = FileLogParser()
        return parser.query_api_logs(days, model, provider, limit, log_type)
    
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
        # 尝试从 PostgreSQL 查询
        if self._available:
            try:
                return self._query_model_usage_from_postgres(days, provider, model)
            except Exception as e:
                logger.error(f"PostgreSQL 查询失败: {e}")
                self._available = False
        
        # 降级到文件日志解析
        logger.info("降级到文件日志解析")
        return self._query_model_usage_from_files(days, provider, model)
    
    def _query_model_usage_from_postgres(
        self,
        days: int,
        provider: Optional[str],
        model: Optional[str]
    ) -> QueryResult:
        """从 PostgreSQL 查询模型使用统计"""
        conn = self._get_connection()
        if not conn:
            raise Exception("无法获取 PostgreSQL 连接")
        
        # 构建查询条件
        where_conditions = ["timestamp >= %s"]
        params = [datetime.utcnow() - timedelta(days=days)]
        
        if provider:
            where_conditions.append("structured_provider = %s")
            params.append(provider)
        
        if model:
            where_conditions.append("model = %s")
            params.append(model)
        
        where_clause = " AND ".join(where_conditions)
        
        # 聚合查询
        query = f"""
            SELECT 
                structured_provider as provider,
                model,
                COUNT(*) as request_count,
                AVG(latency) as avg_duration,
                SUM(CASE WHEN success = true THEN 1 ELSE 0 END) as success_count,
                SUM(CASE WHEN success = false THEN 1 ELSE 0 END) as error_count
            FROM harborai_logs 
            WHERE {where_clause}
            GROUP BY structured_provider, model
            ORDER BY request_count DESC
        """
        
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            data = [dict(row) for row in rows]
        
        return QueryResult(
            data=data,
            total_count=len(data),
            source='postgresql'
        )
    
    def _query_model_usage_from_files(
        self,
        days: int,
        provider: Optional[str],
        model: Optional[str]
    ) -> QueryResult:
        """从文件日志解析模型使用统计"""
        from .file_log_parser import FileLogParser
        
        parser = FileLogParser()
        return parser.query_model_usage(days, provider, model)
    
    def is_available(self) -> bool:
        """检查 PostgreSQL 是否可用"""
        return self._available
    
    def close(self):
        """关闭连接"""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None


# 全局客户端实例
_global_client: Optional[PostgreSQLClient] = None


def get_postgres_client() -> PostgreSQLClient:
    """获取全局 PostgreSQL 客户端实例"""
    global _global_client
    
    if _global_client is None:
        _global_client = PostgreSQLClient()
    
    return _global_client


def close_postgres_client():
    """关闭全局 PostgreSQL 客户端"""
    global _global_client
    
    if _global_client:
        _global_client.close()
        _global_client = None