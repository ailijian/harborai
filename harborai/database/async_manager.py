#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步数据库管理器

提供异步数据库操作接口，支持连接池管理和事务处理
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from contextlib import asynccontextmanager
import asyncpg
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DatabaseManager:
    """异步数据库管理器
    
    功能：
    1. 异步连接池管理
    2. 查询执行和事务处理
    3. 连接健康检查
    4. 自动重连机制
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "harborai",
                 user: str = "postgres",
                 password: str = "",
                 min_size: int = 5,
                 max_size: int = 20,
                 command_timeout: float = 60.0):
        """初始化数据库管理器
        
        Args:
            host: 数据库主机
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            min_size: 连接池最小连接数
            max_size: 连接池最大连接数
            command_timeout: 命令超时时间（秒）
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.min_size = min_size
        self.max_size = max_size
        self.command_timeout = command_timeout
        
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """初始化数据库连接池"""
        if self._initialized:
            return
            
        async with self._lock:
            if self._initialized:
                return
                
            try:
                self.pool = await asyncpg.create_pool(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    command_timeout=self.command_timeout
                )
                
                # 测试连接
                async with self.pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                
                self._initialized = True
                logger.info(f"数据库连接池初始化成功: {self.host}:{self.port}/{self.database}")
                
            except Exception as e:
                logger.error(f"数据库连接池初始化失败: {e}")
                raise
    
    async def close(self):
        """关闭数据库连接池"""
        if self.pool:
            await self.pool.close()
            self.pool = None
            self._initialized = False
            logger.info("数据库连接池已关闭")
    
    async def execute_query(self, 
                          query: str, 
                          params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """执行查询并返回结果
        
        Args:
            query: SQL查询语句
            params: 查询参数
            
        Returns:
            查询结果列表
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as conn:
                if params:
                    rows = await conn.fetch(query, *params)
                else:
                    rows = await conn.fetch(query)
                
                # 转换为字典列表
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"查询执行失败: {e}, SQL: {query}")
            raise
    
    async def execute_command(self, 
                            command: str, 
                            params: Optional[tuple] = None) -> str:
        """执行命令（INSERT, UPDATE, DELETE）
        
        Args:
            command: SQL命令
            params: 命令参数
            
        Returns:
            执行结果状态
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            async with self.pool.acquire() as conn:
                if params:
                    result = await conn.execute(command, *params)
                else:
                    result = await conn.execute(command)
                
                return result
                
        except Exception as e:
            logger.error(f"命令执行失败: {e}, SQL: {command}")
            raise
    
    @asynccontextmanager
    async def transaction(self):
        """事务上下文管理器"""
        if not self._initialized:
            await self.initialize()
            
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                yield conn
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            健康状态信息
        """
        try:
            if not self._initialized:
                return {
                    "status": "unhealthy",
                    "error": "数据库未初始化"
                }
            
            start_time = datetime.now()
            
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "pool_size": self.pool.get_size(),
                "pool_idle": self.pool.get_idle_size(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        if not self.pool:
            return {"error": "连接池未初始化"}
        
        return {
            "total_size": self.pool.get_size(),
            "idle_size": self.pool.get_idle_size(),
            "min_size": self.min_size,
            "max_size": self.max_size,
            "command_timeout": self.command_timeout
        }
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()


# 全局数据库管理器实例
_global_db_manager: Optional[DatabaseManager] = None


async def get_global_db_manager() -> DatabaseManager:
    """获取全局数据库管理器实例"""
    global _global_db_manager
    
    if _global_db_manager is None:
        # 从环境变量或配置文件读取数据库配置
        import os
        
        _global_db_manager = DatabaseManager(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=int(os.getenv("POSTGRES_PORT", "5432")),
            database=os.getenv("POSTGRES_DB", "harborai"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
            min_size=int(os.getenv("DB_POOL_MIN_SIZE", "5")),
            max_size=int(os.getenv("DB_POOL_MAX_SIZE", "20"))
        )
        
        await _global_db_manager.initialize()
    
    return _global_db_manager


async def close_global_db_manager():
    """关闭全局数据库管理器"""
    global _global_db_manager
    
    if _global_db_manager:
        await _global_db_manager.close()
        _global_db_manager = None