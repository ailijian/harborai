#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化连接池

实现异步连接池优化，支持动态调整连接池大小，提升并发性能。
根据技术设计方案，支持连接复用、健康检查、负载均衡和自适应调整。

设计原则：
1. 异步连接管理，支持高并发访问
2. 动态连接池大小调整，根据负载自适应
3. 连接健康检查和自动恢复
4. 连接复用和生命周期管理
5. 负载均衡和故障转移
6. 性能监控和统计

技术特性：
- 异步连接池
- 动态大小调整
- 健康检查
- 连接复用
- 负载均衡
- 故障转移
- 性能监控
"""

import asyncio
import aiohttp
import time
import threading
import weakref
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import ssl
from urllib.parse import urlparse
import json

from .lockfree_plugin_manager import AtomicInteger, AtomicReference

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """连接状态枚举"""
    IDLE = "idle"           # 空闲
    ACTIVE = "active"       # 活跃
    CHECKING = "checking"   # 健康检查中
    FAILED = "failed"       # 失败
    CLOSED = "closed"       # 已关闭


@dataclass
class ConnectionInfo:
    """连接信息"""
    session: aiohttp.ClientSession
    created_at: float
    last_used: float
    use_count: int
    state: 'AtomicReference'  # AtomicReference[ConnectionState]
    endpoint: str
    health_check_count: AtomicInteger = field(default_factory=lambda: AtomicInteger(0))
    error_count: AtomicInteger = field(default_factory=lambda: AtomicInteger(0))
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.use_count, int):
            self.use_count = AtomicInteger(self.use_count)
        if isinstance(self.state, ConnectionState):
            self.state = AtomicReference(self.state)
        if isinstance(self.health_check_count, int):
            self.health_check_count = AtomicInteger(self.health_check_count)
        if isinstance(self.error_count, int):
            self.error_count = AtomicInteger(self.error_count)


@dataclass
class PoolConfig:
    """连接池配置"""
    min_size: int = 5                    # 提升最小连接数到5
    max_size: int = 50                   # 提升最大连接数到50
    max_idle_time: float = 180.0         # 减少最大空闲时间到3分钟
    max_lifetime: float = 1800.0         # 减少连接最大生命周期到30分钟
    health_check_interval: float = 30.0  # 减少健康检查间隔到30秒
    connection_timeout: float = 10.0     # 减少连接超时到10秒
    read_timeout: float = 30.0           # 减少读取超时到30秒
    max_retries: int = 5                 # 增加最大重试次数到5
    retry_delay: float = 0.5             # 减少重试延迟到0.5秒
    enable_ssl_verify: bool = True       # 启用SSL验证
    max_connections_per_host: int = 20   # 提升每个主机的最大连接数到20


class OptimizedConnectionPool:
    """优化连接池
    
    实现异步连接池，支持动态调整、健康检查和负载均衡。
    
    主要特性：
    1. 异步连接管理：使用aiohttp实现高性能异步连接
    2. 动态调整：根据负载自动调整连接池大小
    3. 健康检查：定期检查连接健康状态
    4. 连接复用：最大化连接利用率
    5. 负载均衡：在多个连接间分配请求
    6. 故障转移：自动处理连接失败和恢复
    """
    
    def __init__(self, config: Optional[PoolConfig] = None):
        """初始化连接池
        
        Args:
            config: 连接池配置
        """
        self.config = config or PoolConfig()
        
        # 连接池：按端点分组的连接
        self._pools: Dict[str, List[ConnectionInfo]] = defaultdict(list)
        
        # 连接池锁：保护连接池操作
        self._pool_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        
        # 统计信息
        self._stats = {
            'total_connections': AtomicInteger(0),
            'active_connections': AtomicInteger(0),
            'idle_connections': AtomicInteger(0),
            'failed_connections': AtomicInteger(0),
            'total_requests': AtomicInteger(0),
            'successful_requests': AtomicInteger(0),
            'failed_requests': AtomicInteger(0),
            'pool_hits': AtomicInteger(0),
            'pool_misses': AtomicInteger(0),
            'health_checks': AtomicInteger(0),
            'health_check_failures': AtomicInteger(0),
        }
        
        # 性能监控
        self._performance = {
            'avg_response_time': 0.0,
            'max_response_time': 0.0,
            'min_response_time': float('inf'),
            'total_response_time': 0.0,
            'response_time_samples': 0,
        }
        self._perf_lock = threading.Lock()
        
        # 健康检查任务
        self._health_check_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = True
        
        # SSL上下文
        self._ssl_context = self._create_ssl_context()
        
        logger.info("OptimizedConnectionPool初始化完成，配置: %s", self.config)
    
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """创建SSL上下文
        
        Returns:
            SSL上下文，如果禁用SSL验证则返回False
        """
        if not self.config.enable_ssl_verify:
            return False
        
        context = ssl.create_default_context()
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    
    async def start(self):
        """启动连接池
        
        启动健康检查和清理任务。
        """
        if not self._running:
            self._running = True
        
        # 启动健康检查任务
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        # 启动清理任务
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("OptimizedConnectionPool已启动")
    
    async def stop(self):
        """停止连接池
        
        停止所有任务并关闭所有连接。
        """
        self._running = False
        
        # 停止后台任务
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        await self._close_all_connections()
        
        logger.info("OptimizedConnectionPool已停止")
    
    async def get_session(self, endpoint: str) -> Optional[aiohttp.ClientSession]:
        """获取连接会话
        
        从连接池获取可用连接，如果没有可用连接则创建新连接。
        
        Args:
            endpoint: 目标端点URL
            
        Returns:
            连接会话，如果获取失败则返回None
        """
        self._stats['total_requests'].increment()
        
        # 解析端点
        parsed = urlparse(endpoint)
        pool_key = f"{parsed.scheme}://{parsed.netloc}"
        
        async with self._pool_locks[pool_key]:
            # 尝试从池中获取空闲连接
            connection = await self._get_idle_connection(pool_key)
            
            if connection:
                # 找到空闲连接
                self._stats['pool_hits'].increment()
                connection.state.set(ConnectionState.ACTIVE)
                connection.last_used = time.time()
                connection.use_count.increment()
                self._stats['active_connections'].increment()
                self._stats['idle_connections'].decrement()
                return connection.session
            
            # 没有空闲连接，尝试创建新连接
            self._stats['pool_misses'].increment()
            
            if len(self._pools[pool_key]) < self.config.max_size:
                connection = await self._create_connection(pool_key)
                if connection:
                    self._pools[pool_key].append(connection)
                    self._stats['total_connections'].increment()
                    self._stats['active_connections'].increment()
                    return connection.session
            
            # 连接池已满，等待空闲连接
            return await self._wait_for_idle_connection(pool_key)
    
    async def return_session(self, endpoint: str, session: aiohttp.ClientSession, 
                           success: bool = True):
        """归还连接会话
        
        将使用完的连接归还到连接池。
        
        Args:
            endpoint: 目标端点URL
            session: 连接会话
            success: 请求是否成功
        """
        parsed = urlparse(endpoint)
        pool_key = f"{parsed.scheme}://{parsed.netloc}"
        
        async with self._pool_locks[pool_key]:
            # 查找对应的连接
            for connection in self._pools[pool_key]:
                if connection.session is session:
                    if success:
                        # 请求成功，将连接标记为空闲
                        connection.state.set(ConnectionState.IDLE)
                        connection.last_used = time.time()
                        self._stats['active_connections'].decrement()
                        self._stats['idle_connections'].increment()
                        self._stats['successful_requests'].increment()
                    else:
                        # 请求失败，增加错误计数
                        connection.error_count.increment()
                        self._stats['failed_requests'].increment()
                        
                        # 如果错误过多，关闭连接
                        if connection.error_count.get() >= self.config.max_retries:
                            await self._close_connection(connection, pool_key)
                        else:
                            connection.state.set(ConnectionState.IDLE)
                            self._stats['active_connections'].decrement()
                            self._stats['idle_connections'].increment()
                    break
    
    async def _get_idle_connection(self, pool_key: str) -> Optional[ConnectionInfo]:
        """获取空闲连接
        
        Args:
            pool_key: 连接池键
            
        Returns:
            空闲连接，如果没有则返回None
        """
        for connection in self._pools[pool_key]:
            if connection.state.get() == ConnectionState.IDLE:
                # 检查连接是否仍然有效
                if await self._is_connection_valid(connection):
                    return connection
                else:
                    # 连接无效，移除
                    await self._close_connection(connection, pool_key)
        
        return None
    
    async def _create_connection(self, pool_key: str) -> Optional[ConnectionInfo]:
        """创建新连接
        
        Args:
            pool_key: 连接池键
            
        Returns:
            新连接，如果创建失败则返回None
        """
        try:
            # 创建连接器
            connector = aiohttp.TCPConnector(
                limit=self.config.max_connections_per_host,
                limit_per_host=self.config.max_connections_per_host,
                ssl=self._ssl_context,
                enable_cleanup_closed=True,
                keepalive_timeout=self.config.max_idle_time,
            )
            
            # 创建超时配置
            timeout = aiohttp.ClientTimeout(
                total=self.config.connection_timeout,
                connect=self.config.connection_timeout,
                sock_read=self.config.read_timeout,
            )
            
            # 创建会话
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    'User-Agent': 'HarborAI-OptimizedPool/1.0',
                    'Connection': 'keep-alive',
                }
            )
            
            # 创建连接信息
            connection = ConnectionInfo(
                session=session,
                created_at=time.time(),
                last_used=time.time(),
                use_count=AtomicInteger(1),
                state=AtomicReference(ConnectionState.ACTIVE),
                endpoint=pool_key,
            )
            
            logger.debug("创建新连接: %s", pool_key)
            return connection
            
        except Exception as e:
            logger.error("创建连接失败 %s: %s", pool_key, str(e))
            return None
    
    async def _wait_for_idle_connection(self, pool_key: str, 
                                      timeout: float = 5.0) -> Optional[aiohttp.ClientSession]:
        """等待空闲连接
        
        Args:
            pool_key: 连接池键
            timeout: 超时时间（秒）
            
        Returns:
            连接会话，如果超时则返回None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            async with self._pool_locks[pool_key]:
                connection = await self._get_idle_connection(pool_key)
                if connection:
                    connection.state.set(ConnectionState.ACTIVE)
                    connection.last_used = time.time()
                    connection.use_count.increment()
                    self._stats['active_connections'].increment()
                    self._stats['idle_connections'].decrement()
                    return connection.session
            
            # 短暂等待
            await asyncio.sleep(0.1)
        
        logger.warning("等待空闲连接超时: %s", pool_key)
        return None
    
    async def _is_connection_valid(self, connection: ConnectionInfo) -> bool:
        """检查连接是否有效
        
        Args:
            connection: 连接信息
            
        Returns:
            连接是否有效
        """
        current_time = time.time()
        
        # 检查连接年龄
        if current_time - connection.created_at > self.config.max_lifetime:
            logger.debug("连接超过最大生命周期: %s", connection.endpoint)
            return False
        
        # 检查空闲时间
        if current_time - connection.last_used > self.config.max_idle_time:
            logger.debug("连接空闲时间过长: %s", connection.endpoint)
            return False
        
        # 检查会话状态
        if connection.session.closed:
            logger.debug("连接会话已关闭: %s", connection.endpoint)
            return False
        
        return True
    
    async def _close_connection(self, connection: ConnectionInfo, pool_key: str):
        """关闭连接
        
        Args:
            connection: 连接信息
            pool_key: 连接池键
        """
        try:
            # 更新状态
            old_state = connection.state.get()
            connection.state.set(ConnectionState.CLOSED)
            
            # 关闭会话
            if not connection.session.closed:
                await connection.session.close()
            
            # 从池中移除
            if connection in self._pools[pool_key]:
                self._pools[pool_key].remove(connection)
                self._stats['total_connections'].decrement()
                
                if old_state == ConnectionState.ACTIVE:
                    self._stats['active_connections'].decrement()
                elif old_state == ConnectionState.IDLE:
                    self._stats['idle_connections'].decrement()
                elif old_state == ConnectionState.FAILED:
                    self._stats['failed_connections'].decrement()
            
            logger.debug("关闭连接: %s", connection.endpoint)
            
        except Exception as e:
            logger.error("关闭连接失败 %s: %s", connection.endpoint, str(e))
    
    async def _close_all_connections(self):
        """关闭所有连接"""
        for pool_key, connections in self._pools.items():
            for connection in connections.copy():
                await self._close_connection(connection, pool_key)
        
        self._pools.clear()
        logger.info("所有连接已关闭")
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("健康检查失败: %s", str(e))
                await asyncio.sleep(self.config.health_check_interval)
    
    async def _perform_health_checks(self):
        """执行健康检查"""
        for pool_key, connections in self._pools.items():
            for connection in connections.copy():
                if connection.state.get() == ConnectionState.IDLE:
                    await self._health_check_connection(connection, pool_key)
    
    async def _health_check_connection(self, connection: ConnectionInfo, pool_key: str):
        """健康检查单个连接
        
        Args:
            connection: 连接信息
            pool_key: 连接池键
        """
        try:
            # 设置检查状态
            if not connection.state.compare_and_swap(ConnectionState.IDLE, ConnectionState.CHECKING):
                return  # 连接状态已改变，跳过检查
            
            self._stats['health_checks'].increment()
            connection.health_check_count.increment()
            
            # 执行简单的健康检查（HEAD请求）
            try:
                async with connection.session.head(pool_key, timeout=aiohttp.ClientTimeout(total=5.0)) as response:
                    if response.status < 500:
                        # 健康检查通过
                        connection.state.set(ConnectionState.IDLE)
                        connection.error_count.set(0)  # 重置错误计数
                        return
            except Exception:
                pass  # 健康检查失败，继续处理
            
            # 健康检查失败
            self._stats['health_check_failures'].increment()
            connection.error_count.increment()
            
            if connection.error_count.get() >= self.config.max_retries:
                # 错误过多，关闭连接
                await self._close_connection(connection, pool_key)
            else:
                # 标记为失败状态
                connection.state.set(ConnectionState.FAILED)
                self._stats['failed_connections'].increment()
                self._stats['idle_connections'].decrement()
            
        except Exception as e:
            logger.error("健康检查连接失败 %s: %s", connection.endpoint, str(e))
            connection.state.set(ConnectionState.FAILED)
    
    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                await self._cleanup_expired_connections()
                await asyncio.sleep(60.0)  # 每分钟清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("连接清理失败: %s", str(e))
                await asyncio.sleep(60.0)
    
    async def _cleanup_expired_connections(self):
        """清理过期连接"""
        current_time = time.time()
        
        for pool_key, connections in self._pools.items():
            for connection in connections.copy():
                # 检查是否需要清理
                should_cleanup = False
                
                # 检查生命周期
                if current_time - connection.created_at > self.config.max_lifetime:
                    should_cleanup = True
                
                # 检查空闲时间
                elif (connection.state.get() == ConnectionState.IDLE and 
                      current_time - connection.last_used > self.config.max_idle_time):
                    should_cleanup = True
                
                # 检查失败状态
                elif connection.state.get() == ConnectionState.FAILED:
                    should_cleanup = True
                
                if should_cleanup:
                    await self._close_connection(connection, pool_key)
    
    def update_response_time(self, response_time: float):
        """更新响应时间统计
        
        Args:
            response_time: 响应时间（毫秒）
        """
        with self._perf_lock:
            self._performance['total_response_time'] += response_time
            self._performance['response_time_samples'] += 1
            
            if response_time > self._performance['max_response_time']:
                self._performance['max_response_time'] = response_time
            
            if response_time < self._performance['min_response_time']:
                self._performance['min_response_time'] = response_time
            
            # 计算平均响应时间
            self._performance['avg_response_time'] = (
                self._performance['total_response_time'] / 
                self._performance['response_time_samples']
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取连接池统计信息
        
        Returns:
            统计信息字典
        """
        total_requests = self._stats['total_requests'].get()
        successful_requests = self._stats['successful_requests'].get()
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        pool_hits = self._stats['pool_hits'].get()
        hit_rate = (pool_hits / total_requests * 100) if total_requests > 0 else 0
        
        health_checks = self._stats['health_checks'].get()
        health_check_failures = self._stats['health_check_failures'].get()
        health_success_rate = ((health_checks - health_check_failures) / health_checks * 100) if health_checks > 0 else 0
        
        with self._perf_lock:
            perf_stats = self._performance.copy()
        
        return {
            "total_connections": self._stats['total_connections'].get(),
            "active_connections": self._stats['active_connections'].get(),
            "idle_connections": self._stats['idle_connections'].get(),
            "failed_connections": self._stats['failed_connections'].get(),
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": self._stats['failed_requests'].get(),
            "success_rate_percent": round(success_rate, 2),
            "pool_hits": pool_hits,
            "pool_misses": self._stats['pool_misses'].get(),
            "hit_rate_percent": round(hit_rate, 2),
            "health_checks": health_checks,
            "health_check_failures": health_check_failures,
            "health_success_rate_percent": round(health_success_rate, 2),
            "performance": perf_stats,
            "pool_sizes": {key: len(connections) for key, connections in self._pools.items()},
            "config": {
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "max_idle_time": self.config.max_idle_time,
                "max_lifetime": self.config.max_lifetime,
                "health_check_interval": self.config.health_check_interval,
            }
        }


# 全局连接池实例
_global_connection_pool: Optional[OptimizedConnectionPool] = None
_pool_ref = AtomicReference(None)


async def get_connection_pool(config: Optional[PoolConfig] = None) -> OptimizedConnectionPool:
    """获取全局连接池实例
    
    Args:
        config: 连接池配置
        
    Returns:
        连接池实例
    """
    pool = _pool_ref.get()
    
    if pool is None:
        # 创建新连接池
        new_pool = OptimizedConnectionPool(config)
        
        # 使用CAS操作设置全局实例
        if _pool_ref.compare_and_swap(None, new_pool):
            await new_pool.start()
            return new_pool
        else:
            # 其他协程已经创建了实例
            await new_pool.stop()  # 清理未使用的实例
            return _pool_ref.get()
    
    return pool


async def reset_connection_pool():
    """重置全局连接池
    
    主要用于测试场景。
    """
    pool = _pool_ref.get()
    
    if pool is not None:
        await pool.stop()
        _pool_ref.set(None)