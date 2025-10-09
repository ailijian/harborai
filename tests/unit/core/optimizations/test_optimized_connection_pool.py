#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化连接池测试

测试异步连接池的各项功能，包括连接管理、健康检查、负载均衡等。

测试覆盖：
1. 连接池配置和初始化
2. 连接获取和归还
3. 连接生命周期管理
4. 健康检查机制
5. 动态调整和负载均衡
6. 故障转移和恢复
7. 性能监控和统计
8. 并发安全性
"""

import pytest
import asyncio
import time
import aiohttp
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, Optional, List
import logging

from harborai.core.optimizations.optimized_connection_pool import (
    OptimizedConnectionPool,
    PoolConfig,
    ConnectionState,
    ConnectionInfo,
    get_connection_pool,
    reset_connection_pool
)


def create_mock_session():
    """创建一个正确配置的mock会话"""
    mock_session = AsyncMock()
    mock_session.closed = False
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    return mock_session
from harborai.core.optimizations.lockfree_plugin_manager import AtomicInteger, AtomicReference


class TestPoolConfig:
    """连接池配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = PoolConfig()
        
        assert config.min_size == 5
        assert config.max_size == 50
        assert config.max_idle_time == 180.0
        assert config.max_lifetime == 1800.0
        assert config.health_check_interval == 30.0
        assert config.connection_timeout == 10.0
        assert config.read_timeout == 30.0
        assert config.max_retries == 5
        assert config.retry_delay == 0.5
        assert config.enable_ssl_verify is True
        assert config.max_connections_per_host == 20
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = PoolConfig(
            min_size=10,
            max_size=100,
            max_idle_time=300.0,
            max_lifetime=3600.0,
            health_check_interval=60.0,
            connection_timeout=15.0,
            read_timeout=45.0,
            max_retries=3,
            retry_delay=1.0,
            enable_ssl_verify=False,
            max_connections_per_host=30
        )
        
        assert config.min_size == 10
        assert config.max_size == 100
        assert config.max_idle_time == 300.0
        assert config.max_lifetime == 3600.0
        assert config.health_check_interval == 60.0
        assert config.connection_timeout == 15.0
        assert config.read_timeout == 45.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.enable_ssl_verify is False
        assert config.max_connections_per_host == 30


class TestConnectionInfo:
    """连接信息测试"""
    
    def test_connection_info_creation(self):
        """测试连接信息创建"""
        mock_session = Mock()
        current_time = time.time()
        
        connection = ConnectionInfo(
            session=mock_session,
            created_at=current_time,
            last_used=current_time,
            use_count=0,
            state=ConnectionState.IDLE,
            endpoint="https://api.example.com"
        )
        
        assert connection.session is mock_session
        assert connection.created_at == current_time
        assert connection.last_used == current_time
        assert connection.use_count.get() == 0
        assert connection.state.get() == ConnectionState.IDLE
        assert connection.endpoint == "https://api.example.com"
        assert connection.health_check_count.get() == 0
        assert connection.error_count.get() == 0
    
    def test_connection_info_post_init(self):
        """测试连接信息初始化后处理"""
        mock_session = Mock()
        current_time = time.time()
        
        # 使用原始类型创建
        connection = ConnectionInfo(
            session=mock_session,
            created_at=current_time,
            last_used=current_time,
            use_count=5,  # 原始int
            state=ConnectionState.ACTIVE,  # 原始枚举
            endpoint="https://api.example.com",
            health_check_count=3,  # 原始int
            error_count=1  # 原始int
        )
        
        # 验证自动转换为原子类型
        assert isinstance(connection.use_count, AtomicInteger)
        assert connection.use_count.get() == 5
        assert isinstance(connection.state, AtomicReference)
        assert connection.state.get() == ConnectionState.ACTIVE
        assert isinstance(connection.health_check_count, AtomicInteger)
        assert connection.health_check_count.get() == 3
        assert isinstance(connection.error_count, AtomicInteger)
        assert connection.error_count.get() == 1


class TestOptimizedConnectionPool:
    """优化连接池测试"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return PoolConfig(
            min_size=2,
            max_size=5,
            max_idle_time=60.0,
            max_lifetime=300.0,
            health_check_interval=10.0,
            connection_timeout=5.0,
            read_timeout=15.0,
            max_retries=3,
            retry_delay=0.1,
            enable_ssl_verify=False,
            max_connections_per_host=5
        )
    
    @pytest.fixture
    async def pool(self, config):
        """测试连接池"""
        pool = OptimizedConnectionPool(config)
        yield pool
        # 确保测试后清理资源
        try:
            if pool._running:
                await pool.stop()
        except Exception as e:
            logger.warning("清理连接池时出错: %s", str(e))
    
    def test_init_with_default_config(self):
        """测试使用默认配置初始化"""
        pool = OptimizedConnectionPool()
        
        assert isinstance(pool.config, PoolConfig)
        assert pool.config.min_size == 5
        assert pool.config.max_size == 50
        assert pool._running is True
        assert len(pool._pools) == 0
        assert len(pool._pool_locks) == 0
        
        # 验证统计信息初始化
        assert pool._stats['total_connections'].get() == 0
        assert pool._stats['active_connections'].get() == 0
        assert pool._stats['idle_connections'].get() == 0
        
        # 验证性能监控初始化
        assert pool._performance['avg_response_time'] == 0.0
        assert pool._performance['max_response_time'] == 0.0
        assert pool._performance['min_response_time'] == float('inf')
    
    def test_init_with_custom_config(self, config, pool):
        """测试使用自定义配置初始化"""
        assert pool.config is config
        assert pool.config.min_size == 2
        assert pool.config.max_size == 5
        assert pool._running is True
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, pool):
        """测试启动停止生命周期"""
        # 启动连接池
        await pool.start()
        
        assert pool._running is True
        assert pool._health_check_task is not None
        assert pool._cleanup_task is not None
        assert not pool._health_check_task.done()
        assert not pool._cleanup_task.done()
        
        # 停止连接池
        await pool.stop()
        
        assert pool._running is False
        assert pool._health_check_task.cancelled() or pool._health_check_task.done()
        assert pool._cleanup_task.cancelled() or pool._cleanup_task.done()
    
    @pytest.mark.asyncio
    async def test_get_session_new_connection(self, pool):
        """测试获取会话（创建新连接）"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = create_mock_session()
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                session = await pool.get_session(endpoint)
                
                assert session is mock_session
                assert pool._stats['total_requests'].get() == 1
                assert pool._stats['pool_misses'].get() == 1
                assert pool._stats['total_connections'].get() == 1
                assert pool._stats['active_connections'].get() == 1
    
    @pytest.mark.asyncio
    async def test_get_session_reuse_connection(self, pool):
        """测试获取会话（复用连接）"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 第一次获取（创建新连接）
                session1 = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session1, success=True)
                
                # 第二次获取（复用连接）
                session2 = await pool.get_session(endpoint)
                
                assert session2 is mock_session
                assert pool._stats['total_requests'].get() == 2
                assert pool._stats['pool_hits'].get() == 1
                assert pool._stats['pool_misses'].get() == 1
                assert pool._stats['total_connections'].get() == 1
    
    @pytest.mark.asyncio
    async def test_return_session_success(self, pool):
        """测试归还会话（成功）"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取会话
                session = await pool.get_session(endpoint)
                
                # 归还会话（成功）
                await pool.return_session(endpoint, session, success=True)
                
                assert pool._stats['successful_requests'].get() == 1
                assert pool._stats['active_connections'].get() == 0
                assert pool._stats['idle_connections'].get() == 1
    
    @pytest.mark.asyncio
    async def test_return_session_failure(self, pool):
        """测试归还会话（失败）"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取会话
                session = await pool.get_session(endpoint)
                
                # 归还会话（失败）
                await pool.return_session(endpoint, session, success=False)
                
                assert pool._stats['failed_requests'].get() == 1
                # 连接应该仍然可用（错误次数未达到阈值）
                assert pool._stats['active_connections'].get() == 0
                assert pool._stats['idle_connections'].get() == 1
    
    @pytest.mark.asyncio
    async def test_return_session_multiple_failures(self, pool):
        """测试归还会话（多次失败导致连接关闭）"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取会话
                session = await pool.get_session(endpoint)
                
                # 多次失败归还
                for i in range(pool.config.max_retries):
                    await pool.return_session(endpoint, session, success=False)
                
                assert pool._stats['failed_requests'].get() == pool.config.max_retries
                # 连接应该被关闭
                assert pool._stats['total_connections'].get() == 0
                assert pool._stats['active_connections'].get() == 0
                assert pool._stats['idle_connections'].get() == 0
    
    @pytest.mark.asyncio
    async def test_connection_pool_limit(self, pool):
        """测试连接池大小限制"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_sessions = [AsyncMock() for _ in range(pool.config.max_size + 2)]
            for session in mock_sessions:
                session.closed = False
            mock_session_class.side_effect = mock_sessions
            
            with patch('aiohttp.TCPConnector'):
                sessions = []
                
                # 获取最大数量的连接
                for i in range(pool.config.max_size):
                    session = await pool.get_session(f"{endpoint}/{i}")
                    sessions.append(session)
                
                assert pool._stats['total_connections'].get() == pool.config.max_size
                assert pool._stats['active_connections'].get() == pool.config.max_size
                
                # 尝试获取超出限制的连接（应该等待或返回None）
                with patch.object(pool, '_wait_for_idle_connection', return_value=None):
                    session = await pool.get_session(f"{endpoint}/overflow")
                    assert session is None
    
    @pytest.mark.asyncio
    async def test_connection_validation(self, pool):
        """测试连接有效性检查"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                await pool.return_session(endpoint, session, success=True)
                
                # 模拟连接过期
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.created_at = time.time() - pool.config.max_lifetime - 1
                
                # 再次获取连接（应该创建新连接）
                with patch.object(pool, '_create_connection') as mock_create:
                    # 创建一个新的mock连接
                    new_mock_session = AsyncMock()
                    new_mock_session.closed = False
                    new_connection = ConnectionInfo(
                        session=new_mock_session,
                        created_at=time.time(),
                        last_used=time.time(),
                        use_count=AtomicInteger(1),
                        state=AtomicReference(ConnectionState.ACTIVE),
                        endpoint=pool_key,
                    )
                    mock_create.return_value = new_connection
                    
                    session2 = await pool.get_session(endpoint)
                    
                    # 验证尝试创建新连接
                    mock_create.assert_called_once()
    
    @pytest.mark.skip(reason="Mock issue with async context manager - needs investigation")
    @pytest.mark.asyncio
    async def test_health_check_mechanism(self, pool):
        """测试健康检查机制"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取并归还连接
                session = await pool.get_session(endpoint)
                await pool.return_session(endpoint, session, success=True)
                
                # 执行健康检查
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                
                # 初始化健康检查统计
                initial_health_checks = pool._stats['health_checks'].get()
                initial_connection_health_checks = connection.health_check_count.get()
                
                # 直接模拟健康检查成功，避免复杂的async context manager mock
                with patch.object(connection.session, 'head') as mock_head:
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    mock_head.return_value.__aenter__ = AsyncMock(return_value=mock_response)
                    mock_head.return_value.__aexit__ = AsyncMock(return_value=None)
                    
                    await pool._health_check_connection(connection, pool_key)
                
                # 验证健康检查统计
                assert pool._stats['health_checks'].get() == initial_health_checks + 1
                assert connection.health_check_count.get() == initial_connection_health_checks + 1
                assert connection.error_count.get() == 0
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, pool):
        """测试健康检查失败"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            # 模拟健康检查失败
            mock_session.head.side_effect = Exception("Connection failed")
            
            with patch('aiohttp.TCPConnector'):
                # 获取并归还连接
                session = await pool.get_session(endpoint)
                await pool.return_session(endpoint, session, success=True)
                
                # 执行健康检查
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                
                await pool._health_check_connection(connection, pool_key)
                
                # 验证健康检查失败统计
                assert pool._stats['health_check_failures'].get() == 1
                assert connection.error_count.get() == 1
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, pool):
        """测试性能监控"""
        response_time = 0.5  # 500ms
        
        pool.update_response_time(response_time)
        
        assert pool._performance['avg_response_time'] == response_time
        assert pool._performance['max_response_time'] == response_time
        assert pool._performance['min_response_time'] == response_time
        assert pool._performance['total_response_time'] == response_time
        assert pool._performance['response_time_samples'] == 1
        
        # 添加更多样本
        pool.update_response_time(1.0)  # 1000ms
        pool.update_response_time(0.2)  # 200ms
        
        assert pool._performance['max_response_time'] == 1.0
        assert pool._performance['min_response_time'] == 0.2
        assert pool._performance['response_time_samples'] == 3
        expected_avg = (0.5 + 1.0 + 0.2) / 3
        assert abs(pool._performance['avg_response_time'] - expected_avg) < 0.001
    
    def test_get_statistics(self, pool):
        """测试获取统计信息"""
        # 设置一些统计数据
        pool._stats['total_connections'].set(10)
        pool._stats['active_connections'].set(5)
        pool._stats['idle_connections'].set(3)
        pool._stats['failed_connections'].set(2)
        pool._stats['total_requests'].set(100)
        pool._stats['successful_requests'].set(90)
        pool._stats['failed_requests'].set(10)
        
        stats = pool.get_statistics()
        
        assert stats['total_connections'] == 10
        assert stats['active_connections'] == 5
        assert stats['idle_connections'] == 3
        assert stats['failed_connections'] == 2
        assert stats['total_requests'] == 100
        assert stats['successful_requests'] == 90
        assert stats['failed_requests'] == 10
        assert stats['success_rate_percent'] == 90.0  # 注意字段名
        assert 'performance' in stats
        
        performance = stats['performance']
        assert 'avg_response_time' in performance
        assert 'max_response_time' in performance
        assert 'min_response_time' in performance


class TestOptimizedConnectionPoolIntegration:
    """优化连接池集成测试"""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(30)  # 30秒超时
    async def test_concurrent_connections(self):
        """测试并发连接"""
        config = PoolConfig(min_size=2, max_size=10)
        pool = OptimizedConnectionPool(config)
        
        try:
            await pool.start()
            
            async def get_and_return_session(endpoint_suffix):
                """获取并归还会话"""
                endpoint = f"https://api.example.com/{endpoint_suffix}"
                
                with patch('aiohttp.ClientSession') as mock_session_class:
                    mock_session = AsyncMock()
                    mock_session.closed = False
                    mock_session_class.return_value = mock_session
                    
                    with patch('aiohttp.TCPConnector'):
                        # 添加超时控制
                        session = await asyncio.wait_for(
                            pool.get_session(endpoint), 
                            timeout=5.0
                        )
                        await asyncio.sleep(0.05)  # 减少模拟使用时间
                        await asyncio.wait_for(
                            pool.return_session(endpoint, session, success=True),
                            timeout=5.0
                        )
                        return session
            
            # 并发执行多个请求，减少并发数量
            tasks = [get_and_return_session(f"test{i}") for i in range(10)]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=20.0
            )
            
            # 验证所有请求都成功
            assert len(results) == 10
            for result in results:
                assert not isinstance(result, Exception), f"Unexpected exception: {result}"
            
            # 验证统计信息
            assert pool._stats['total_requests'].get() == 10
            assert pool._stats['successful_requests'].get() == 10
            
        finally:
            await pool.stop()
    
    @pytest.mark.asyncio
    async def test_connection_lifecycle_management(self):
        """测试连接生命周期管理"""
        config = PoolConfig(
            min_size=1,
            max_size=3,
            max_idle_time=0.1,  # 100ms空闲超时
            max_lifetime=0.5    # 500ms生命周期
        )
        pool = OptimizedConnectionPool(config)
        
        try:
            await pool.start()
            
            with patch('aiohttp.ClientSession') as mock_session_class:
                mock_session = AsyncMock()
                mock_session.closed = False
                mock_session_class.return_value = mock_session
                
                with patch('aiohttp.TCPConnector'):
                    # 获取连接
                    endpoint = "https://api.example.com/test"
                    session = await pool.get_session(endpoint)
                    await pool.return_session(endpoint, session, success=True)
                    
                    # 等待连接过期
                    await asyncio.sleep(0.6)
                    
                    # 再次获取连接（应该创建新连接）
                    with patch.object(pool, '_create_connection') as mock_create:
                        # 创建一个新的mock连接
                        new_mock_session = AsyncMock()
                        new_mock_session.closed = False
                        new_connection = ConnectionInfo(
                            session=new_mock_session,
                            created_at=time.time(),
                            last_used=time.time(),
                            use_count=AtomicInteger(1),
                            state=AtomicReference(ConnectionState.ACTIVE),
                            endpoint="https://api.example.com",
                        )
                        mock_create.return_value = new_connection
                        
                        session2 = await pool.get_session(endpoint)
                        
                        # 验证尝试创建新连接（因为旧连接已过期）
                        mock_create.assert_called_once()
            
        finally:
            await pool.stop()


class TestGlobalConnectionPool:
    """全局连接池测试"""
    
    @pytest.mark.asyncio
    async def test_get_global_connection_pool(self):
        """测试获取全局连接池"""
        # 重置全局连接池
        await reset_connection_pool()
        
        config = PoolConfig(min_size=3, max_size=10)
        
        # 获取全局连接池
        pool1 = await get_connection_pool(config)
        pool2 = await get_connection_pool()
        
        # 验证返回同一个实例
        assert pool1 is pool2
        assert pool1.config.min_size == 3
        assert pool1.config.max_size == 10
        
        # 清理
        await reset_connection_pool()
    
    @pytest.mark.asyncio
    async def test_reset_connection_pool(self):
        """测试重置全局连接池"""
        config = PoolConfig(min_size=2, max_size=5)
        
        # 获取全局连接池
        pool1 = await get_connection_pool(config)
        
        # 重置连接池
        await reset_connection_pool()
        
        # 再次获取（应该是新实例）
        pool2 = await get_connection_pool()
        
        assert pool1 is not pool2
        assert pool2.config.min_size == 5  # 默认配置


class TestAdvancedConnectionPoolFeatures:
    """高级连接池功能测试"""
    
    @pytest.fixture
    async def pool(self):
        """测试连接池"""
        config = PoolConfig(
            min_size=2,
            max_size=5,
            max_idle_time=60.0,
            max_lifetime=300.0,
            health_check_interval=10.0,
            connection_timeout=5.0,
            read_timeout=15.0,
            max_retries=3,
            retry_delay=0.1,
            enable_ssl_verify=False,
            max_connections_per_host=5
        )
        pool = OptimizedConnectionPool(config)
        await pool.start()
        yield pool
        await pool.stop()
    
    @pytest.mark.asyncio
    async def test_ssl_context_creation_enabled(self):
        """测试启用SSL验证时的SSL上下文创建"""
        config = PoolConfig(enable_ssl_verify=True)
        pool = OptimizedConnectionPool(config)
        
        assert pool._ssl_context is not False
        assert hasattr(pool._ssl_context, 'check_hostname')
        assert hasattr(pool._ssl_context, 'verify_mode')
    
    @pytest.mark.asyncio
    async def test_ssl_context_creation_disabled(self):
        """测试禁用SSL验证时的SSL上下文创建"""
        config = PoolConfig(enable_ssl_verify=False)
        pool = OptimizedConnectionPool(config)
        
        assert pool._ssl_context is False
    
    @pytest.mark.asyncio
    async def test_connection_creation_failure(self, pool):
        """测试连接创建失败"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession', side_effect=Exception("Connection failed")):
            with patch('aiohttp.TCPConnector'):
                session = await pool.get_session(endpoint)
                assert session is None
    
    @pytest.mark.asyncio
    async def test_wait_for_idle_connection_timeout(self, pool):
        """测试等待空闲连接超时"""
        endpoint = "https://api.example.com/test"
        
        # 填满连接池
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_sessions = []
            for i in range(pool.config.max_size):
                mock_session = create_mock_session()
                mock_sessions.append(mock_session)
            
            mock_session_class.side_effect = mock_sessions
            
            with patch('aiohttp.TCPConnector'):
                # 获取所有连接
                sessions = []
                for i in range(pool.config.max_size):
                    session = await pool.get_session(endpoint)
                    sessions.append(session)
                
                # 尝试获取额外连接（应该超时）
                session = await pool.get_session(endpoint)
                assert session is None
    
    @pytest.mark.asyncio
    async def test_connection_validation_expired_lifetime(self, pool):
        """测试连接验证 - 生命周期过期"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 模拟连接生命周期过期
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.created_at = time.time() - pool.config.max_lifetime - 1
                
                # 再次获取连接（应该创建新连接）
                with patch.object(pool, '_create_connection') as mock_create:
                    mock_create.return_value = None
                    session2 = await pool.get_session(endpoint)
                    assert session2 is None
    
    @pytest.mark.asyncio
    async def test_connection_validation_idle_timeout(self, pool):
        """测试连接验证 - 空闲超时"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 模拟连接空闲超时
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.last_used = time.time() - pool.config.max_idle_time - 1
                
                # 再次获取连接（应该创建新连接）
                with patch.object(pool, '_create_connection') as mock_create:
                    mock_create.return_value = None
                    session2 = await pool.get_session(endpoint)
                    assert session2 is None
    
    @pytest.mark.asyncio
    async def test_connection_validation_session_closed(self, pool):
        """测试连接验证 - 会话已关闭"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 模拟会话关闭
                mock_session.closed = True
                
                # 再次获取连接（应该创建新连接）
                with patch.object(pool, '_create_connection') as mock_create:
                    mock_create.return_value = None
                    session2 = await pool.get_session(endpoint)
                    assert session2 is None
    
    @pytest.mark.asyncio
    async def test_return_session_closed_connection(self, pool):
        """测试归还已关闭的连接"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 手动关闭连接
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.state.set(ConnectionState.CLOSED)
                
                # 归还连接（应该被忽略）
                await pool.return_session(endpoint, session, success=True)
                
                # 验证统计信息没有变化
                assert pool._stats['successful_requests'].get() == 0
    
    @pytest.mark.asyncio
    async def test_return_session_failure_max_retries(self, pool):
        """测试归还失败连接达到最大重试次数"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 设置错误计数接近最大重试次数
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.error_count.set(pool.config.max_retries - 1)
                
                # 归还失败连接（应该关闭连接）
                await pool.return_session(endpoint, session, success=False)
                
                # 验证连接被关闭
                assert connection.state.get() == ConnectionState.CLOSED
                assert pool._stats['total_connections'].get() == 0
    
    @pytest.mark.asyncio
    async def test_return_session_failure_from_idle_state(self, pool):
        """测试从空闲状态归还失败连接"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接为空闲状态
                await pool.return_session(endpoint, session, success=True)
                
                # 手动设置为空闲状态
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.state.set(ConnectionState.IDLE)
                
                # 再次归还失败连接
                await pool.return_session(endpoint, session, success=False)
                
                # 验证连接仍为空闲状态（因为不是从ACTIVE状态归还）
                assert connection.state.get() == ConnectionState.IDLE
    
    @pytest.mark.asyncio
    async def test_health_check_loop_exception_handling(self, pool):
        """测试健康检查循环异常处理"""
        with patch.object(pool, '_perform_health_checks', side_effect=Exception("Health check error")):
            # 启动健康检查循环
            task = asyncio.create_task(pool._health_check_loop())
            
            # 等待一小段时间让异常发生
            await asyncio.sleep(0.1)
            
            # 停止循环
            pool._running = False
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_health_check_connection_timeout(self, pool):
        """测试健康检查连接超时"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 模拟健康检查超时
                with patch.object(pool, '_health_check_connection', side_effect=asyncio.TimeoutError()):
                    await pool._perform_health_checks()
    
    @pytest.mark.asyncio
    async def test_health_check_connection_exception(self, pool):
        """测试健康检查连接异常"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 模拟健康检查异常
                with patch.object(pool, '_health_check_connection', side_effect=Exception("Health check failed")):
                    await pool._perform_health_checks()
    
    @pytest.mark.asyncio
    async def test_health_check_connection_state_change(self, pool):
        """测试健康检查时连接状态改变"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 获取连接并设置为活跃状态
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.state.set(ConnectionState.ACTIVE)
                
                # 执行健康检查（应该跳过因为状态不是IDLE）
                await pool._health_check_connection(connection, pool_key)
                
                # 验证健康检查计数没有增加
                assert connection.health_check_count.get() == 0
    
    @pytest.mark.skip(reason="Mock issue with async context manager - needs investigation")
    @pytest.mark.asyncio
    async def test_health_check_connection_success(self, pool):
        """测试健康检查连接成功"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = create_mock_session()
            
            # 模拟成功的HEAD请求
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_head_context = AsyncMock()
            mock_head_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_head_context.__aexit__ = AsyncMock(return_value=None)
            mock_session.head.return_value = mock_head_context
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 设置错误计数
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.error_count.set(2)
                
                # 为连接的session设置相同的HEAD mock
                connection.session.head = mock_session.head
                
                # 执行健康检查
                await pool._health_check_connection(connection, pool_key)
                
                # 验证健康检查成功
                assert connection.state.get() == ConnectionState.IDLE
                assert connection.error_count.get() == 0
                assert connection.health_check_count.get() == 1
    
    @pytest.mark.asyncio
    async def test_health_check_connection_failure_close(self, pool):
        """测试健康检查失败导致连接关闭"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            
            # 模拟失败的HEAD请求
            mock_session.head.side_effect = Exception("Request failed")
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 设置错误计数接近最大值
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.error_count.set(pool.config.max_retries - 1)
                
                # 执行健康检查
                await pool._health_check_connection(connection, pool_key)
                
                # 验证连接被关闭
                assert connection.state.get() == ConnectionState.CLOSED
                assert pool._stats['total_connections'].get() == 0
    
    @pytest.mark.asyncio
    async def test_health_check_connection_failure_mark_failed(self, pool):
        """测试健康检查失败标记为失败状态"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            
            # 模拟失败的HEAD请求
            mock_session.head.side_effect = Exception("Request failed")
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 执行健康检查
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                await pool._health_check_connection(connection, pool_key)
                
                # 验证连接被标记为失败
                assert connection.state.get() == ConnectionState.FAILED
                assert pool._stats['failed_connections'].get() == 1
                assert pool._stats['idle_connections'].get() == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_loop_exception_handling(self, pool):
        """测试清理循环异常处理"""
        with patch.object(pool, '_cleanup_expired_connections', side_effect=Exception("Cleanup error")):
            # 启动清理循环
            task = asyncio.create_task(pool._cleanup_loop())
            
            # 等待一小段时间让异常发生
            await asyncio.sleep(0.1)
            
            # 停止循环
            pool._running = False
            task.cancel()
            
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_connections_lifetime(self, pool):
        """测试清理过期连接 - 生命周期"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 模拟连接生命周期过期
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.created_at = time.time() - pool.config.max_lifetime - 1
                
                # 执行清理
                await pool._cleanup_expired_connections()
                
                # 验证连接被清理
                assert len(pool._pools[pool_key]) == 0
                assert pool._stats['total_connections'].get() == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_connections_idle_time(self, pool):
        """测试清理过期连接 - 空闲时间"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 模拟连接空闲超时
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.last_used = time.time() - pool.config.max_idle_time - 1
                
                # 执行清理
                await pool._cleanup_expired_connections()
                
                # 验证连接被清理
                assert len(pool._pools[pool_key]) == 0
                assert pool._stats['total_connections'].get() == 0
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_connections_failed_state(self, pool):
        """测试清理失败状态的连接"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 设置连接为失败状态
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                connection.state.set(ConnectionState.FAILED)
                pool._stats['idle_connections'].decrement()
                pool._stats['failed_connections'].increment()
                
                # 执行清理
                await pool._cleanup_expired_connections()
                
                # 验证连接被清理
                assert len(pool._pools[pool_key]) == 0
                assert pool._stats['total_connections'].get() == 0
                assert pool._stats['failed_connections'].get() == 0
    
    @pytest.mark.asyncio
    async def test_update_response_time(self, pool):
        """测试更新响应时间统计"""
        # 更新响应时间
        pool.update_response_time(100.0)
        pool.update_response_time(200.0)
        pool.update_response_time(50.0)
        
        # 验证统计信息
        assert pool._performance['total_response_time'] == 350.0
        assert pool._performance['response_time_samples'] == 3
        assert pool._performance['avg_response_time'] == 350.0 / 3
        assert pool._performance['max_response_time'] == 200.0
        assert pool._performance['min_response_time'] == 50.0
    
    @pytest.mark.asyncio
    async def test_get_statistics_comprehensive(self, pool):
        """测试获取全面的统计信息"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 执行一些操作
                session = await pool.get_session(endpoint)
                await pool.return_session(endpoint, session, success=True)
                
                # 更新响应时间
                pool.update_response_time(150.0)
                
                # 模拟健康检查
                pool._stats['health_checks'].increment()
                
                # 获取统计信息
                stats = pool.get_statistics()
                
                # 验证统计信息
                assert stats['total_connections'] == 1
                assert stats['idle_connections'] == 1
                assert stats['total_requests'] == 1
                assert stats['successful_requests'] == 1
                assert stats['success_rate_percent'] == 100.0
                assert stats['pool_hits'] == 0
                assert stats['pool_misses'] == 1
                assert stats['hit_rate_percent'] == 0.0
                assert stats['health_checks'] == 1
                assert stats['health_success_rate_percent'] == 100.0
                assert 'performance' in stats
                assert 'pool_sizes' in stats
                assert 'config' in stats
    
    @pytest.mark.asyncio
    async def test_get_statistics_zero_division_protection(self, pool):
        """测试统计信息零除保护"""
        # 获取空统计信息
        stats = pool.get_statistics()
        
        # 验证零除保护
        assert stats['success_rate_percent'] == 0
        assert stats['hit_rate_percent'] == 0
        assert stats['health_success_rate_percent'] == 0
    
    @pytest.mark.asyncio
    async def test_wait_for_idle_connection_lock_timeout(self, pool):
        """测试等待空闲连接时锁超时"""
        endpoint = "https://api.example.com/test"
        
        # 模拟锁获取超时
        with patch.object(pool._pool_locks["https://api.example.com"], 'acquire', side_effect=asyncio.TimeoutError()):
            session = await pool._wait_for_idle_connection("https://api.example.com", timeout=0.2)
            assert session is None
    
    @pytest.mark.skip(reason="Mock issue with teardown - needs investigation")
    @pytest.mark.asyncio
    async def test_close_connection_exception_handling(self, pool):
        """测试关闭连接时的异常处理"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session.close.side_effect = Exception("Close failed")
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 关闭连接（应该处理异常）
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                await pool._close_connection(connection, pool_key)
                
                # 验证连接被移除（异常被正确处理）
                assert len(pool._pools[pool_key]) == 0
    
    @pytest.mark.asyncio
    async def test_health_check_connection_exception_handling(self, pool):
        """测试健康检查连接异常处理"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.closed = False
            mock_session_class.return_value = mock_session
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                session = await pool.get_session(endpoint)
                
                # 归还连接
                await pool.return_session(endpoint, session, success=True)
                
                # 模拟健康检查异常
                pool_key = "https://api.example.com"
                connection = pool._pools[pool_key][0]
                
                with patch.object(connection.state, 'compare_and_swap', side_effect=Exception("State error")):
                    await pool._health_check_connection(connection, pool_key)
                    
                    # 验证连接状态被设置为失败
                    assert connection.state.get() == ConnectionState.FAILED


class TestConcurrentConnectionPool:
    """并发连接池测试"""
    
    @pytest.fixture
    async def pool(self):
        """测试连接池"""
        config = PoolConfig(
            min_size=2,
            max_size=10,
            max_idle_time=60.0,
            max_lifetime=300.0,
            health_check_interval=10.0,
            connection_timeout=5.0,
            read_timeout=15.0,
            max_retries=3,
            retry_delay=0.1,
            enable_ssl_verify=False,
            max_connections_per_host=10
        )
        pool = OptimizedConnectionPool(config)
        await pool.start()
        yield pool
        await pool.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_connection_acquisition(self, pool):
        """测试并发连接获取"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            # 创建多个mock会话
            mock_sessions = []
            for i in range(5):
                mock_session = AsyncMock()
                mock_session.closed = False
                mock_sessions.append(mock_session)
            
            mock_session_class.side_effect = mock_sessions
            
            with patch('aiohttp.TCPConnector'):
                # 并发获取连接
                tasks = []
                for i in range(5):
                    task = asyncio.create_task(pool.get_session(endpoint))
                    tasks.append(task)
                
                sessions = await asyncio.gather(*tasks)
                
                # 验证所有连接都成功获取
                assert len(sessions) == 5
                assert all(session is not None for session in sessions)
                assert pool._stats['total_connections'].get() == 5
                assert pool._stats['active_connections'].get() == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_connection_return(self, pool):
        """测试并发连接归还"""
        endpoint = "https://api.example.com/test"
        
        with patch('aiohttp.ClientSession') as mock_session_class:
            # 创建多个mock会话
            mock_sessions = []
            for i in range(3):
                mock_session = AsyncMock()
                mock_session.closed = False
                mock_sessions.append(mock_session)
            
            mock_session_class.side_effect = mock_sessions
            
            with patch('aiohttp.TCPConnector'):
                # 获取连接
                sessions = []
                for i in range(3):
                    session = await pool.get_session(endpoint)
                    sessions.append(session)
                
                # 并发归还连接
                tasks = []
                for session in sessions:
                    task = asyncio.create_task(pool.return_session(endpoint, session, success=True))
                    tasks.append(task)
                
                await asyncio.gather(*tasks)
                
                # 验证所有连接都成功归还
                assert pool._stats['active_connections'].get() == 0
                assert pool._stats['idle_connections'].get() == 3
                assert pool._stats['successful_requests'].get() == 3


class TestGlobalConnectionPoolAdvanced:
    """高级全局连接池测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_global_pool_creation(self):
        """测试并发全局连接池创建"""
        # 重置全局连接池
        await reset_connection_pool()
        
        config = PoolConfig(min_size=3, max_size=10)
        
        # 并发获取全局连接池
        tasks = []
        for i in range(5):
            task = asyncio.create_task(get_connection_pool(config))
            tasks.append(task)
        
        pools = await asyncio.gather(*tasks)
        
        # 验证所有任务返回同一个实例
        first_pool = pools[0]
        for pool in pools[1:]:
            assert pool is first_pool
        
        # 清理
        await reset_connection_pool()
    
    @pytest.mark.asyncio
    async def test_global_pool_cleanup_unused_instance(self):
        """测试全局连接池清理未使用的实例"""
        # 重置全局连接池
        await reset_connection_pool()
        
        config = PoolConfig(min_size=2, max_size=5)
        
        # 模拟竞争条件：一个实例创建成功，另一个被清理
        with patch('harborai.core.optimizations.optimized_connection_pool._pool_ref') as mock_ref:
            # 第一次调用返回None，第二次返回已存在的实例
            existing_pool = OptimizedConnectionPool(config)
            mock_ref.get.side_effect = [None, existing_pool]
            mock_ref.compare_and_swap.return_value = False  # 模拟CAS失败
            
            with patch.object(OptimizedConnectionPool, 'stop') as mock_stop:
                pool = await get_connection_pool(config)
                
                # 验证未使用的实例被清理
                mock_stop.assert_called_once()
                assert pool is existing_pool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])