#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL 数据库连接管理

提供 PostgreSQL 数据库初始化和连接管理功能。
"""

import os
from typing import Optional
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


def get_postgres_url() -> Optional[str]:
    """获取 PostgreSQL 连接字符串"""
    settings = get_settings()
    
    if settings.postgres_url:
        # 将 asyncpg 格式转换为 psycopg2 格式
        url = settings.postgres_url
        if url.startswith("postgresql+asyncpg://"):
            url = url.replace("postgresql+asyncpg://", "postgresql://")
        return url
    
    # 从环境变量构建连接字符串
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "harborai")
    user = os.getenv("POSTGRES_USER", "harborai")
    password = os.getenv("POSTGRES_PASSWORD")
    
    if not password:
        logger.warning("PostgreSQL 密码未设置")
        return None
    
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def init_postgres_database() -> bool:
    """初始化 PostgreSQL 数据库表结构
    
    Returns:
        bool: 初始化是否成功
    """
    if not PSYCOPG2_AVAILABLE:
        logger.error("psycopg2 未安装，无法连接 PostgreSQL")
        return False
    
    postgres_url = get_postgres_url()
    if not postgres_url:
        logger.error("PostgreSQL 连接配置不完整")
        return False
    
    try:
        with psycopg2.connect(postgres_url) as conn:
            with conn.cursor() as cursor:
                # 创建 API 日志表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        provider VARCHAR(100) NOT NULL,
                        model VARCHAR(100) NOT NULL,
                        request_data TEXT,
                        response_data TEXT,
                        status_code INTEGER,
                        error_message TEXT,
                        duration_ms REAL
                    )
                """)
                
                # 创建跟踪日志表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trace_logs (
                        id SERIAL PRIMARY KEY,
                        trace_id VARCHAR(100) NOT NULL,
                        span_id VARCHAR(100) NOT NULL,
                        parent_span_id VARCHAR(100),
                        operation_name VARCHAR(200) NOT NULL,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP,
                        duration_ms REAL,
                        tags TEXT,
                        logs TEXT
                    )
                """)
                
                # 创建模型使用统计表
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS model_usage (
                        id SERIAL PRIMARY KEY,
                        date DATE NOT NULL,
                        provider VARCHAR(100) NOT NULL,
                        model VARCHAR(100) NOT NULL,
                        request_count INTEGER DEFAULT 0,
                        total_tokens INTEGER DEFAULT 0,
                        input_tokens INTEGER DEFAULT 0,
                        output_tokens INTEGER DEFAULT 0,
                        total_cost REAL DEFAULT 0.0,
                        UNIQUE(date, provider, model)
                    )
                """)
                
                # 创建索引
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_logs_timestamp 
                    ON api_logs(timestamp)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_api_logs_provider_model 
                    ON api_logs(provider, model)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_trace_logs_trace_id 
                    ON trace_logs(trace_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model_usage_date 
                    ON model_usage(date)
                """)
                
                conn.commit()
                logger.info("PostgreSQL 数据库表结构初始化完成")
                return True
                
    except Exception as e:
        logger.error(f"PostgreSQL 数据库初始化失败: {e}")
        return False


@contextmanager
def get_postgres_connection():
    """获取 PostgreSQL 连接上下文管理器"""
    if not PSYCOPG2_AVAILABLE:
        logger.warning("psycopg2 未安装，无法连接 PostgreSQL")
        yield None
        return
    
    postgres_url = get_postgres_url()
    if not postgres_url:
        logger.warning("PostgreSQL 连接配置不完整")
        yield None
        return
    
    try:
        conn = psycopg2.connect(postgres_url, cursor_factory=RealDictCursor)
        try:
            yield conn
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"PostgreSQL 连接失败: {e}")
        yield None


def test_postgres_connection() -> bool:
    """测试 PostgreSQL 连接
    
    Returns:
        bool: 连接是否成功
    """
    try:
        with get_postgres_connection() as conn:
            if conn is None:
                return False
            
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
                
    except Exception as e:
        logger.error(f"PostgreSQL 连接测试失败: {e}")
        return False