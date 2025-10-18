"""
PostgreSQL连接池管理模块
提供高性能、可靠的数据库连接池功能
"""

import asyncio
import time
import threading
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
from queue import Queue, Empty, Full
import logging

from ..utils.logger import get_logger
from ..utils.exceptions import StorageError

logger = get_logger(__name__)


@dataclass
class ConnectionPoolConfig:
    """连接池配置"""
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: float = 30.0
    idle_timeout: float = 300.0  # 5分钟
    max_lifetime: float = 3600.0  # 1小时
    health_check_interval: float = 60.0  # 1分钟
    retry_attempts: int = 3
    retry_delay: float = 1.0
    enable_health_check: bool = True


@dataclass
class ConnectionStats:
    """连接统计信息"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    failed_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0


class PooledConnection:
    """池化连接包装器"""
    
    def __init__(self, connection, pool: 'ConnectionPool'):
        self.connection = connection
        self.pool = pool
        self.created_at = datetime.now()
        self.last_used = datetime.now()
        self.is_healthy = True
        self.usage_count = 0
        self._lock = threading.Lock()
    
    def __enter__(self):
        """进入上下文管理器"""
        with self._lock:
            self.last_used = datetime.now()
            self.usage_count += 1
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器"""
        # 如果发生异常，标记连接为不健康
        if exc_type is not None:
            self.is_healthy = False
        
        # 将连接返回到池中
        self.pool._return_connection(self)
    
    def is_expired(self, max_lifetime: float, idle_timeout: float) -> bool:
        """检查连接是否过期"""
        now = datetime.now()
        
        # 检查最大生命周期
        if (now - self.created_at).total_seconds() > max_lifetime:
            return True
        
        # 检查空闲超时
        if (now - self.last_used).total_seconds() > idle_timeout:
            return True
        
        return False
    
    def test_health(self) -> bool:
        """测试连接健康状态"""
        try:
            with self._lock:
                if not self.connection or self.connection.closed:
                    return False
                
                # 执行简单的健康检查查询
                with self.connection.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                
                self.is_healthy = True
                return True
        except Exception as e:
            logger.warning(f"连接健康检查失败: {e}")
            self.is_healthy = False
            return False
    
    def close(self):
        """关闭连接"""
        try:
            if self.connection and not self.connection.closed:
                self.connection.close()
        except Exception as e:
            logger.warning(f"关闭连接时发生错误: {e}")


class ConnectionPool:
    """PostgreSQL连接池"""
    
    def __init__(self, 
                 connection_string: str,
                 config: Optional[ConnectionPoolConfig] = None):
        """初始化连接池
        
        Args:
            connection_string: PostgreSQL连接字符串
            config: 连接池配置
        """
        self.connection_string = connection_string
        self.config = config or ConnectionPoolConfig()
        
        # 连接池状态
        self._pool: Queue[PooledConnection] = Queue(maxsize=self.config.max_connections)
        self._all_connections: List[PooledConnection] = []
        self._stats = ConnectionStats()
        self._lock = threading.RLock()
        self._shutdown = False
        
        # 健康检查线程
        self._health_check_thread: Optional[threading.Thread] = None
        
        # 初始化连接池
        self._initialize_pool()
        
        # 启动健康检查
        if self.config.enable_health_check:
            self._start_health_check()
    
    def _initialize_pool(self):
        """初始化连接池"""
        logger.info(f"初始化连接池，最小连接数: {self.config.min_connections}")
        
        for _ in range(self.config.min_connections):
            try:
                conn = self._create_connection()
                if conn:
                    self._pool.put_nowait(conn)
            except Exception as e:
                logger.error(f"初始化连接失败: {e}")
    
    def _create_connection(self) -> Optional[PooledConnection]:
        """创建新连接"""
        try:
            import psycopg2
            
            # 创建原始连接
            raw_conn = psycopg2.connect(self.connection_string)
            raw_conn.autocommit = False
            
            # 包装为池化连接
            pooled_conn = PooledConnection(raw_conn, self)
            
            with self._lock:
                self._all_connections.append(pooled_conn)
                self._stats.total_connections += 1
            
            logger.debug("创建新数据库连接")
            return pooled_conn
            
        except ImportError:
            raise StorageError("psycopg2 未安装。请安装 psycopg2 以使用 PostgreSQL 连接池。")
        except Exception as e:
            with self._lock:
                self._stats.failed_connections += 1
            logger.error(f"创建数据库连接失败: {e}")
            return None
    
    @contextmanager
    def get_connection(self, timeout: Optional[float] = None):
        """获取连接（上下文管理器）
        
        Args:
            timeout: 获取连接的超时时间
            
        Yields:
            数据库连接对象
        """
        if self._shutdown:
            raise StorageError("连接池已关闭")
        
        timeout = timeout or self.config.connection_timeout
        start_time = time.time()
        
        connection = None
        try:
            # 尝试从池中获取连接
            connection = self._get_pooled_connection(timeout)
            
            if not connection:
                raise StorageError("无法获取数据库连接")
            
            # 测试连接健康状态
            if not connection.test_health():
                # 连接不健康，尝试创建新连接
                self._remove_connection(connection)
                connection = self._get_or_create_connection(timeout - (time.time() - start_time))
                
                if not connection:
                    raise StorageError("无法获取健康的数据库连接")
            
            with self._lock:
                self._stats.total_requests += 1
                self._stats.active_connections += 1
            
            # 使用连接
            yield connection.connection
            
            with self._lock:
                self._stats.successful_requests += 1
            
        except Exception as e:
            with self._lock:
                self._stats.failed_requests += 1
            logger.error(f"使用数据库连接时发生错误: {e}")
            
            # 如果连接出现问题，标记为不健康
            if connection:
                connection.is_healthy = False
            
            raise
        finally:
            # 更新统计信息
            if connection:
                with self._lock:
                    self._stats.active_connections -= 1
                    
                    # 更新平均响应时间
                    duration = time.time() - start_time
                    if self._stats.successful_requests > 0:
                        self._stats.average_response_time = (
                            (self._stats.average_response_time * (self._stats.successful_requests - 1) + duration) /
                            self._stats.successful_requests
                        )
    
    def _get_pooled_connection(self, timeout: float) -> Optional[PooledConnection]:
        """从池中获取连接"""
        try:
            return self._pool.get(timeout=timeout)
        except Empty:
            # 池中没有可用连接，尝试创建新连接
            with self._lock:
                if len(self._all_connections) < self.config.max_connections:
                    return self._create_connection()
            return None
    
    def _get_or_create_connection(self, timeout: float) -> Optional[PooledConnection]:
        """获取或创建连接"""
        # 首先尝试从池中获取
        try:
            return self._pool.get(timeout=min(timeout, 1.0))
        except Empty:
            pass
        
        # 尝试创建新连接
        with self._lock:
            if len(self._all_connections) < self.config.max_connections:
                return self._create_connection()
        
        # 等待池中有可用连接
        try:
            return self._pool.get(timeout=timeout)
        except Empty:
            return None
    
    def _return_connection(self, connection: PooledConnection):
        """将连接返回到池中"""
        if self._shutdown:
            connection.close()
            return
        
        # 检查连接是否过期或不健康
        if (not connection.is_healthy or 
            connection.is_expired(self.config.max_lifetime, self.config.idle_timeout)):
            self._remove_connection(connection)
            
            # 如果连接数低于最小值，创建新连接
            with self._lock:
                if len(self._all_connections) < self.config.min_connections:
                    new_conn = self._create_connection()
                    if new_conn:
                        try:
                            self._pool.put_nowait(new_conn)
                        except Full:
                            new_conn.close()
            return
        
        # 将健康连接返回到池中
        try:
            self._pool.put_nowait(connection)
        except Full:
            # 池已满，关闭连接
            self._remove_connection(connection)
    
    def _remove_connection(self, connection: PooledConnection):
        """从池中移除连接"""
        with self._lock:
            if connection in self._all_connections:
                self._all_connections.remove(connection)
                self._stats.total_connections -= 1
        
        connection.close()
    
    def _start_health_check(self):
        """启动健康检查线程"""
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True,
            name="ConnectionPool-HealthCheck"
        )
        self._health_check_thread.start()
        logger.info("连接池健康检查线程已启动")
    
    def _health_check_loop(self):
        """健康检查循环"""
        while not self._shutdown:
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"健康检查过程中发生错误: {e}")
                time.sleep(5)  # 错误时短暂等待
    
    def _perform_health_check(self):
        """执行健康检查"""
        logger.debug("执行连接池健康检查")
        
        unhealthy_connections = []
        
        with self._lock:
            connections_to_check = self._all_connections.copy()
            self._stats.last_health_check = datetime.now()
        
        for conn in connections_to_check:
            if not conn.test_health():
                unhealthy_connections.append(conn)
        
        # 移除不健康的连接
        for conn in unhealthy_connections:
            self._remove_connection(conn)
            with self._lock:
                self._stats.health_check_failures += 1
        
        # 确保最小连接数
        with self._lock:
            current_count = len(self._all_connections)
            needed = self.config.min_connections - current_count
        
        for _ in range(needed):
            new_conn = self._create_connection()
            if new_conn:
                try:
                    self._pool.put_nowait(new_conn)
                except Full:
                    new_conn.close()
                    break
        
        if unhealthy_connections:
            logger.info(f"健康检查完成，移除了 {len(unhealthy_connections)} 个不健康连接")
    
    def get_stats(self) -> ConnectionStats:
        """获取连接池统计信息"""
        with self._lock:
            stats = ConnectionStats(
                total_connections=self._stats.total_connections,
                active_connections=self._stats.active_connections,
                idle_connections=self._pool.qsize(),
                failed_connections=self._stats.failed_connections,
                total_requests=self._stats.total_requests,
                successful_requests=self._stats.successful_requests,
                failed_requests=self._stats.failed_requests,
                average_response_time=self._stats.average_response_time,
                last_health_check=self._stats.last_health_check,
                health_check_failures=self._stats.health_check_failures
            )
        return stats
    
    def shutdown(self, timeout: float = 30.0):
        """关闭连接池
        
        Args:
            timeout: 关闭超时时间
        """
        logger.info("开始关闭连接池")
        self._shutdown = True
        
        # 等待健康检查线程结束
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)
        
        # 关闭所有连接
        with self._lock:
            connections_to_close = self._all_connections.copy()
            self._all_connections.clear()
        
        for conn in connections_to_close:
            conn.close()
        
        # 清空池
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                conn.close()
            except Empty:
                break
        
        logger.info("连接池已关闭")