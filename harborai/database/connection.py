#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库连接管理 (已弃用)

⚠️ 警告: 此模块已弃用，请使用 postgres_connection.py

此模块提供 SQLite 数据库连接，仅用于向后兼容和数据迁移。
新的实现应使用 PostgreSQL 连接。
"""

import warnings

# 发出弃用警告
warnings.warn(
    "harborai.database.connection 模块已弃用，请使用 postgres_connection.py",
    DeprecationWarning,
    stacklevel=2
)

import sqlite3
import os
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

# 数据库文件路径
DB_PATH = Path.home() / ".harborai" / "harborai.db"

def init_database_sync() -> None:
    """同步初始化数据库"""
    # 确保目录存在
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建数据库连接
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    try:
        # 创建API日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                parent_span_id TEXT,
                operation_name TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                duration_ms REAL,
                tags TEXT,
                logs TEXT
            )
        """)
        
        # 创建模型使用统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                request_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_cost REAL DEFAULT 0.0
            )
        """)
        
        conn.commit()
        
    finally:
        conn.close()

@contextmanager
def get_db_session():
    """获取数据库会话上下文管理器"""
    # 检查数据库文件是否存在
    if not DB_PATH.exists():
        yield None
        return
    
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row  # 使结果可以按列名访问
        
        try:
            yield conn
        finally:
            conn.close()
    except Exception:
        yield None

def get_db_connection() -> sqlite3.Connection:
    """获取数据库连接"""
    return sqlite3.connect(str(DB_PATH))