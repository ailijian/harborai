#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发管理器测试

测试并发管理器的核心功能：
1. 配置管理和初始化
2. 组件生命周期管理
3. 性能监控和统计
4. 自适应优化策略
5. 故障检测和恢复
6. 资源管理和清理
7. 并发控制策略

测试覆盖率目标：≥85%
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, Optional, List
import logging

from harborai.core.optimizations.concurrency_manager import (
    ConcurrencyManager,
    ConcurrencyConfig,
    ConcurrencyMode,
    ComponentStatus,
    ComponentInfo,
    PerformanceMetrics,
    get_concurrency_manager,
    reset_concurrency_manager
)
from harborai.core.optimizations.lockfree_plugin_manager import AtomicInteger, AtomicReference


class TestConcurrencyConfig:
    """测试并发配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ConcurrencyConfig()
        
        assert config.max_concurrent_requests == 100
        assert config.connection_pool_size == 50
        assert config.request_timeout == 15.0
        assert config.enable_adaptive_optimization is True
        assert config.enable_health_check is True
        assert config.health_check_interval == 30.0
        assert config.memory_threshold_mb == 2048
        assert config.cpu_threshold_percent == 85.0
        assert config.response_time_threshold_ms == 500.0
        assert config.error_rate_threshold_percent == 3.0
        assert config.max_recovery_attempts == 5
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = ConcurrencyConfig(
            max_concurrent_requests=200,
            connection_pool_size=100,
            request_timeout=30.0,
            enable_adaptive_optimization=False,
            enable_health_check=False,
            health_check_interval=60.0,
            memory_threshold_mb=4096,
            cpu_threshold_percent=90.0,
            response_time_threshold_ms=1000.0,
            error_rate_threshold_percent=5.0,
            max_recovery_attempts=10
        )
        
        assert config.max_concurrent_requests == 200
        assert config.connection_pool_size == 100
        assert config.request_timeout == 30.0
        assert config.enable_adaptive_optimization is False
        assert config.enable_health_check is False
        assert config.health_check_interval == 60.0
        assert config.memory_threshold_mb == 4096
        assert config.cpu_threshold_percent == 90.0
        assert config.response_time_threshold_ms == 1000.0
        assert config.error_rate_threshold_percent == 5.0
        assert config.max_recovery_attempts == 10


class TestComponentInfo:
    """测试组件信息类"""
    
    def test_component_info_creation(self):
        """测试组件信息创建"""
        status = AtomicReference(ComponentStatus.STOPPED)
        mock_instance = Mock()
        
        info = ComponentInfo(
            name="test_component",
            status=status,
            instance=mock_instance,
            start_time=time.time()
        )
        
        assert info.name == "test_component"
        assert info.status.get() == ComponentStatus.STOPPED
        assert info.instance == mock_instance
        assert info.start_time is not None
        assert info.error_count.get() == 0
        assert info.last_error is None
        assert info.recovery_attempts.get() == 0
    
    def test_component_info_error_tracking(self):
        """测试组件错误跟踪"""
        status = AtomicReference(ComponentStatus.RUNNING)
        mock_instance = Mock()
        
        info = ComponentInfo(
            name="test_component",
            status=status,
            instance=mock_instance
        )
        
        # 模拟错误
        info.error_count.increment()
        info.last_error = "Test error"
        info.recovery_attempts.increment()
        
        assert info.error_count.get() == 1
        assert info.last_error == "Test error"
        assert info.recovery_attempts.get() == 1


class TestPerformanceMetrics:
    """测试性能指标类"""
    
    def test_performance_metrics_creation(self):
        """测试性能指标创建"""
        timestamp = time.time()
        metrics = PerformanceMetrics(
            timestamp=timestamp,
            cpu_usage=50.0,
            memory_usage=60.0,
            active_threads=10,
            active_connections=5,
            request_throughput=100.0,
            avg_response_time=200.0,
            error_rate=1.0,
            plugin_cache_hit_rate=95.0
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0
        assert metrics.active_threads == 10
        assert metrics.active_connections == 5
        assert metrics.request_throughput == 100.0
        assert metrics.avg_response_time == 200.0
        assert metrics.error_rate == 1.0
        assert metrics.plugin_cache_hit_rate == 95.0


class TestConcurrencyManager:
    """测试并发管理器"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return ConcurrencyConfig(
            max_concurrent_requests=10,
            connection_pool_size=5,
            request_timeout=5.0,
            enable_adaptive_optimization=True,
            enable_health_check=True,
            health_check_interval=0.1,  # 短间隔用于测试
            memory_threshold_mb=1024,
            cpu_threshold_percent=80.0,
            response_time_threshold_ms=300.0,
            error_rate_threshold_percent=2.0,
            max_recovery_attempts=3
        )
    
    @pytest.fixture
    def manager(self, config):
        """测试管理器实例"""
        return ConcurrencyManager(config)
    
    def test_initialization(self, manager, config):
        """测试初始化"""
        assert manager.config == config
        assert manager.config.max_concurrent_requests == 10
        assert manager.config.connection_pool_size == 5
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, manager):
        """测试启动停止生命周期"""
        # 模拟依赖组件
        with patch('harborai.core.optimizations.memory_manager.get_memory_manager') as mock_get_memory, \
             patch('harborai.core.optimizations.concurrency_manager.get_connection_pool') as mock_get_pool, \
             patch('harborai.core.optimizations.concurrency_manager.get_request_processor') as mock_get_processor, \
             patch('harborai.core.optimizations.concurrency_manager.get_lockfree_plugin_manager') as mock_get_plugin:
            
            # 设置模拟返回值
            mock_memory = AsyncMock()
            mock_pool = AsyncMock()
            mock_processor = AsyncMock()
            mock_plugin = AsyncMock()
            
            mock_get_memory.return_value = mock_memory
            mock_get_pool.return_value = mock_pool
            mock_get_processor.return_value = mock_processor
            mock_get_plugin.return_value = mock_plugin
            
            # 测试启动
            assert manager._status.get() == ComponentStatus.STOPPED
            await manager.start()
            
            # 验证状态
            assert manager._status.get() == ComponentStatus.RUNNING
            assert manager._running is True
            assert manager._start_time is not None
            assert manager._thread_pool is not None
            
            # 测试停止
            await manager.stop()
            
            # 验证状态
            assert manager._status.get() == ComponentStatus.STOPPED
            assert manager._running is False
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, manager):
        """测试重复启动"""
        with patch('harborai.core.optimizations.memory_manager.get_memory_manager') as mock_get_memory:
            mock_memory = AsyncMock()
            mock_get_memory.return_value = mock_memory
            
            # 第一次启动
            await manager.start()
            assert manager._running is True
            
            # 第二次启动应该直接返回
            await manager.start()
            assert manager._running is True
    
    @pytest.mark.asyncio
    async def test_start_component_failure(self, manager):
        """测试组件启动失败"""
        with patch('harborai.core.optimizations.memory_manager.get_memory_manager') as mock_get_memory:
            # 模拟组件启动失败
            mock_get_memory.side_effect = Exception("Memory manager start failed")
            
            # 启动不会抛出异常，但会记录错误
            await manager.start()
            
            # 验证管理器仍然启动，但组件可能有错误
            assert manager._status.get() == ComponentStatus.RUNNING
            assert manager._running is True
    
    def test_get_statistics(self, manager):
        """测试获取统计信息"""
        stats = manager.get_statistics()
        
        assert isinstance(stats, dict)
        assert 'manager_status' in stats
        assert 'config' in stats
        assert 'components' in stats
        assert 'performance' in stats
        assert 'uptime' in stats
        assert 'optimization_history' in stats
        
        assert stats['manager_status'] == ComponentStatus.STOPPED.value
        assert stats['components'] == {}
        assert stats['performance'] == {}
        assert stats['uptime'] == 0
        assert stats['optimization_history'] == 0
    
    @pytest.mark.asyncio
    async def test_register_component(self, manager):
        """测试注册组件"""
        mock_component = Mock()
        component_name = "test_component"
        
        await manager._register_component(
            component_name,
            mock_component,
            ComponentStatus.RUNNING
        )
        
        assert component_name in manager._components
        component_info = manager._components[component_name]
        assert component_info.name == component_name
        assert component_info.instance == mock_component
        assert component_info.status.get() == ComponentStatus.RUNNING
        assert component_info.start_time is not None
    
    @pytest.mark.asyncio
    async def test_stop_component(self, manager):
        """测试停止组件"""
        # 先注册一个组件
        mock_component = Mock()
        component_name = "test_component"
        
        await manager._register_component(
            component_name,
            mock_component,
            ComponentStatus.RUNNING
        )
        
        # 停止组件
        await manager._stop_component(component_name)
        
        # 验证组件状态
        component_info = manager._components[component_name]
        assert component_info.status.get() == ComponentStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_collect_performance_metrics(self, manager):
        """测试性能指标收集"""
        # 模拟一些性能数据
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('threading.active_count', return_value=5):
            
            mock_memory.return_value.percent = 60.0
            
            # 收集性能指标
            metrics = await manager._collect_performance_metrics()
            
            # 验证指标
            assert metrics.cpu_usage == 50.0
            assert metrics.memory_usage == 60.0
            assert metrics.active_threads == 5
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization_high_cpu(self, manager):
        """测试高CPU使用率的自适应优化"""
        # 设置初始并发数
        original_max_concurrent = manager.config.max_concurrent_requests
        
        # 创建高CPU使用率的指标
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=90.0,  # 超过阈值80%
            memory_usage=50.0,
            active_threads=10,
            active_connections=5,
            request_throughput=100.0,
            avg_response_time=200.0,
            error_rate=1.0,
            plugin_cache_hit_rate=95.0
        )
        
        # 模拟请求处理器
        manager._request_processor = Mock()
        
        # 重置优化时间以允许优化
        manager._last_optimization = 0
        
        # 执行自适应优化
        await manager._adaptive_optimization(metrics)
        
        # 验证优化历史被记录
        assert len(manager._optimization_history) > 0
        optimization = manager._optimization_history[-1]
        assert 'reduce_concurrency' in optimization['optimizations']
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization_low_cpu(self, manager):
        """测试低CPU使用率的自适应优化"""
        # 创建低CPU使用率的指标
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=20.0,  # 低于30%
            memory_usage=50.0,
            active_threads=10,
            active_connections=5,
            request_throughput=100.0,
            avg_response_time=200.0,
            error_rate=1.0,
            plugin_cache_hit_rate=95.0
        )
        
        # 模拟请求处理器
        manager._request_processor = Mock()
        
        # 重置优化时间以允许优化
        manager._last_optimization = 0
        
        # 执行自适应优化
        await manager._adaptive_optimization(metrics)
        
        # 验证优化历史被记录
        assert len(manager._optimization_history) > 0
        optimization = manager._optimization_history[-1]
        assert 'increase_concurrency' in optimization['optimizations']
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization_high_memory(self, manager):
        """测试高内存使用率的自适应优化"""
        # 创建高内存使用率的指标
        # 内存阈值计算: (2048 / 1024 * 100) = 200%，所以我们需要超过200%
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=50.0,
            memory_usage=250.0,  # 超过阈值200%的高内存使用率
            active_threads=10,
            active_connections=5,
            request_throughput=100.0,
            avg_response_time=200.0,
            error_rate=1.0,
            plugin_cache_hit_rate=95.0
        )
        
        # 模拟内存管理器
        mock_memory_manager = AsyncMock()
        manager._memory_manager = mock_memory_manager
        
        # 重置优化时间以允许优化
        manager._last_optimization = 0
        
        # 模拟垃圾回收
        with patch('gc.collect') as mock_gc:
            # 执行自适应优化
            await manager._adaptive_optimization(metrics)
            
            # 验证内存清理被调用
            mock_memory_manager.cleanup.assert_called_once()
            mock_gc.assert_called_once()
            
            # 验证优化历史被记录
            assert len(manager._optimization_history) > 0
            optimization = manager._optimization_history[-1]
            assert 'memory_cleanup' in optimization['optimizations']
            assert 'force_gc' in optimization['optimizations']
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization_high_response_time(self, manager):
        """测试高响应时间的自适应优化"""
        # 创建高响应时间的指标
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=50.0,
            memory_usage=50.0,
            active_threads=10,
            active_connections=5,
            request_throughput=100.0,
            avg_response_time=600.0,  # 超过阈值300ms
            error_rate=1.0,
            plugin_cache_hit_rate=95.0
        )
        
        # 模拟连接池
        manager._connection_pool = Mock()
        
        # 重置优化时间以允许优化
        manager._last_optimization = 0
        
        # 执行自适应优化
        await manager._adaptive_optimization(metrics)
        
        # 验证优化历史被记录
        assert len(manager._optimization_history) > 0
        optimization = manager._optimization_history[-1]
        assert 'optimize_connection_pool' in optimization['optimizations']
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization_high_error_rate(self, manager):
        """测试高错误率的自适应优化"""
        # 创建高错误率的指标
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=50.0,
            memory_usage=50.0,
            active_threads=10,
            active_connections=5,
            request_throughput=100.0,
            avg_response_time=200.0,
            error_rate=5.0,  # 超过阈值2%
            plugin_cache_hit_rate=95.0
        )
        
        # 重置优化时间以允许优化
        manager._last_optimization = 0
        
        # 执行自适应优化
        await manager._adaptive_optimization(metrics)
        
        # 验证优化历史被记录
        assert len(manager._optimization_history) > 0
        optimization = manager._optimization_history[-1]
        assert 'enable_degradation' in optimization['optimizations']
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization_rate_limiting(self, manager):
        """测试自适应优化的频率限制"""
        # 创建指标
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=90.0,  # 超过阈值
            memory_usage=50.0,
            active_threads=10,
            active_connections=5,
            request_throughput=100.0,
            avg_response_time=200.0,
            error_rate=1.0,
            plugin_cache_hit_rate=95.0
        )
        
        # 设置最近的优化时间
        manager._last_optimization = time.time()
        
        # 执行自适应优化
        await manager._adaptive_optimization(metrics)
        
        # 验证没有新的优化被记录（由于频率限制）
        assert len(manager._optimization_history) == 0
    
    @pytest.mark.asyncio
    async def test_health_check_component_recovery(self, manager):
        """测试健康检查和组件恢复"""
        # 注册一个组件
        mock_component = Mock()
        mock_component.get_statistics = Mock(return_value={'status': 'healthy'})
        
        await manager._register_component(
            "test_component",
            mock_component,
            ComponentStatus.RUNNING
        )
        
        # 模拟组件失败
        component_info = manager._components["test_component"]
        component_info.status.set(ComponentStatus.ERROR)
        component_info.error_count.set(1)
        component_info.last_error = "Test error"
        
        # 执行健康检查
        await manager._check_component_health()
        
        # 验证组件恢复尝试
        assert component_info.recovery_attempts.get() > 0
    
    @pytest.mark.asyncio
    async def test_recover_component_success(self, manager):
        """测试组件恢复成功"""
        # 创建一个可恢复的组件
        mock_component = AsyncMock()
        mock_component.start = AsyncMock()
        
        await manager._register_component(
            "test_component",
            mock_component,
            ComponentStatus.ERROR
        )
        
        # 获取组件信息
        component_info = manager._components["test_component"]
        
        # 执行组件恢复
        await manager._recover_component("test_component", component_info)
        
        # 验证组件状态
        assert component_info.status.get() == ComponentStatus.RUNNING
        assert component_info.start_time is not None
    
    @pytest.mark.asyncio
    async def test_recover_component_failure(self, manager):
        """测试组件恢复失败"""
        # 创建一个无法恢复的组件
        mock_component = AsyncMock()
        mock_component.start = AsyncMock(side_effect=Exception("Recovery failed"))
        
        await manager._register_component(
            "test_component",
            mock_component,
            ComponentStatus.ERROR
        )
        
        # 获取组件信息
        component_info = manager._components["test_component"]
        
        # 执行组件恢复
        await manager._recover_component("test_component", component_info)
        
        # 验证组件状态仍然是错误
        assert component_info.status.get() == ComponentStatus.ERROR
        assert component_info.error_count.get() > 0
        assert "Recovery failed" in component_info.last_error
    
    @pytest.mark.asyncio
    async def test_submit_request(self, manager):
        """测试提交请求"""
        # 模拟请求处理器
        mock_processor = AsyncMock()
        mock_processor.submit_request = AsyncMock(return_value="response")
        manager._request_processor = mock_processor
        
        # 提交请求
        result = await manager.submit_request("GET", "http://example.com")
        
        # 验证请求被提交
        mock_processor.submit_request.assert_called_once_with("GET", "http://example.com")
        assert result == "response"
    
    @pytest.mark.asyncio
    async def test_submit_request_no_processor(self, manager):
        """测试没有请求处理器时提交请求"""
        # 确保没有请求处理器
        manager._request_processor = None
        
        # 提交请求应该抛出异常
        with pytest.raises(RuntimeError, match="异步请求处理器未启用"):
            await manager.submit_request("GET", "http://example.com")
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_with_processor(self, manager):
        """测试使用请求处理器创建聊天完成"""
        # 模拟插件管理器和插件
        mock_plugin_manager = Mock()
        mock_plugin = AsyncMock()
        mock_plugin.create_async = AsyncMock(return_value="chat_response")
        mock_plugin_manager.get_plugin_for_model = Mock(return_value=mock_plugin)
        manager._plugin_manager = mock_plugin_manager
        
        # 模拟请求处理器
        mock_processor = Mock()
        manager._request_processor = mock_processor
        
        # 创建聊天完成
        result = await manager.create_chat_completion(
            model="test_model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # 验证插件被调用
        mock_plugin_manager.get_plugin_for_model.assert_called_once_with("test_model")
        mock_plugin.create_async.assert_called_once_with(
            model="test_model",
            messages=[{"role": "user", "content": "Hello"}],
            request_processor=mock_processor
        )
        assert result == "chat_response"
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_without_processor(self, manager):
        """测试没有请求处理器时创建聊天完成"""
        # 模拟插件管理器和插件
        mock_plugin_manager = Mock()
        mock_plugin = AsyncMock()
        mock_plugin.create_async = AsyncMock(return_value="chat_response")
        mock_plugin_manager.get_plugin_for_model = Mock(return_value=mock_plugin)
        manager._plugin_manager = mock_plugin_manager
        
        # 确保没有请求处理器
        manager._request_processor = None
        
        # 创建聊天完成
        result = await manager.create_chat_completion(
            model="test_model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # 验证插件被调用（不带请求处理器）
        mock_plugin.create_async.assert_called_once_with(
            model="test_model",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert result == "chat_response"
    
    @pytest.mark.asyncio
    async def test_create_chat_completion_no_plugin(self, manager):
        """测试没有找到插件时创建聊天完成"""
        # 模拟插件管理器返回None
        mock_plugin_manager = Mock()
        mock_plugin_manager.get_plugin_for_model = Mock(return_value=None)
        manager._plugin_manager = mock_plugin_manager
        
        # 创建聊天完成应该抛出异常
        with pytest.raises(ValueError, match="未找到模型 test_model 的插件"):
            await manager.create_chat_completion(
                model="test_model",
                messages=[{"role": "user", "content": "Hello"}]
            )
    
    def test_get_plugin(self, manager):
        """测试获取插件"""
        # 模拟插件管理器
        mock_plugin_manager = Mock()
        mock_plugin = Mock()
        mock_plugin_manager.get_plugin_for_model = Mock(return_value=mock_plugin)
        manager._plugin_manager = mock_plugin_manager
        
        # 获取插件
        result = manager.get_plugin("test_model")
        
        # 验证插件被返回
        mock_plugin_manager.get_plugin_for_model.assert_called_once_with("test_model")
        assert result == mock_plugin
    
    def test_get_plugin_no_manager(self, manager):
        """测试没有插件管理器时获取插件"""
        # 确保没有插件管理器
        manager._plugin_manager = None
        
        # 获取插件应该抛出异常
        with pytest.raises(RuntimeError, match="无锁插件管理器未启用"):
            manager.get_plugin("test_model")
    
    def test_get_statistics_with_components(self, manager):
        """测试获取包含组件的统计信息"""
        # 注册一个组件
        mock_component = Mock()
        mock_component.get_statistics = Mock(return_value={'test_stat': 'value'})
        
        # 手动注册组件（同步方式）
        component_info = ComponentInfo(
            name="test_component",
            status=AtomicReference(ComponentStatus.RUNNING),
            instance=mock_component,
            start_time=time.time()
        )
        manager._components["test_component"] = component_info
        
        # 添加性能历史
        metrics = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=50.0,
            memory_usage=60.0,
            active_threads=10,
            active_connections=5,
            request_throughput=100.0,
            avg_response_time=200.0,
            error_rate=1.0,
            plugin_cache_hit_rate=95.0
        )
        manager._performance_history.append(metrics)
        
        # 获取统计信息
        stats = manager.get_statistics()
        
        # 验证组件统计
        assert 'test_component' in stats['components']
        component_stats = stats['components']['test_component']
        assert component_stats['status'] == ComponentStatus.RUNNING.value
        assert component_stats['error_count'] == 0
        assert component_stats['recovery_attempts'] == 0
        assert 'details' in component_stats
        assert component_stats['details']['test_stat'] == 'value'
        
        # 验证性能统计
        assert stats['performance']['cpu_usage'] == 50.0
        assert stats['performance']['memory_usage'] == 60.0
        assert stats['performance']['request_throughput'] == 100.0
    
    def test_get_statistics_with_trends(self, manager):
        """测试获取包含趋势的统计信息"""
        # 添加两个性能指标以计算趋势
        metrics1 = PerformanceMetrics(
            timestamp=time.time() - 60,
            cpu_usage=40.0,
            memory_usage=50.0,
            active_threads=8,
            active_connections=3,
            request_throughput=80.0,
            avg_response_time=250.0,
            error_rate=0.5,
            plugin_cache_hit_rate=90.0
        )
        
        metrics2 = PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=50.0,
            memory_usage=60.0,
            active_threads=10,
            active_connections=5,
            request_throughput=100.0,
            avg_response_time=200.0,
            error_rate=1.0,
            plugin_cache_hit_rate=95.0
        )
        
        manager._performance_history = [metrics1, metrics2]
        
        # 获取统计信息
        stats = manager.get_statistics()
        
        # 验证趋势计算
        assert 'trends' in stats['performance']
        trends = stats['performance']['trends']
        assert trends['cpu_usage_trend'] == 10.0  # 50 - 40
        assert trends['memory_usage_trend'] == 10.0  # 60 - 50
        assert trends['throughput_trend'] == 20.0  # 100 - 80
        assert trends['response_time_trend'] == -50.0  # 200 - 250
    
    def test_get_performance_history(self, manager):
        """测试获取性能历史"""
        current_time = time.time()
        
        # 添加一些性能指标
        metrics1 = PerformanceMetrics(
            timestamp=current_time - 600,  # 10分钟前
            cpu_usage=40.0,
            memory_usage=50.0,
            active_threads=8,
            active_connections=3,
            request_throughput=80.0,
            avg_response_time=250.0,
            error_rate=0.5,
            plugin_cache_hit_rate=90.0
        )
        
        metrics2 = PerformanceMetrics(
            timestamp=current_time - 60,  # 1分钟前
            cpu_usage=50.0,
            memory_usage=60.0,
            active_threads=10,
            active_connections=5,
            request_throughput=100.0,
            avg_response_time=200.0,
            error_rate=1.0,
            plugin_cache_hit_rate=95.0
        )
        
        manager._performance_history.extend([metrics1, metrics2])
        
        # 获取最近5分钟的历史
        history = manager.get_performance_history(duration=300.0)
        
        # 验证只返回最近5分钟的数据
        assert len(history) == 1
        assert history[0] == metrics2
        
        # 获取最近15分钟的历史
        history = manager.get_performance_history(duration=900.0)
        
        # 验证返回所有数据
        assert len(history) == 2
        assert history[0] == metrics1
        assert history[1] == metrics2


class TestConcurrencyManagerIntegration:
    """并发管理器集成测试"""
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_execution(self):
        """测试监控循环执行"""
        config = ConcurrencyConfig(
            enable_adaptive_optimization=True,
            health_check_interval=0.05  # 很短的间隔用于测试
        )
        
        manager = ConcurrencyManager(config)
        
        # 模拟组件
        with patch('harborai.core.optimizations.memory_manager.get_memory_manager') as mock_get_memory, \
             patch('harborai.core.optimizations.concurrency_manager.get_connection_pool') as mock_get_pool, \
             patch('harborai.core.optimizations.concurrency_manager.get_request_processor') as mock_get_processor, \
             patch('harborai.core.optimizations.concurrency_manager.get_lockfree_plugin_manager') as mock_get_plugin, \
             patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory:
            
            # 设置模拟返回值
            mock_memory_manager = AsyncMock()
            mock_pool = Mock()
            mock_processor = Mock()
            mock_plugin = Mock()
            
            mock_get_memory.return_value = mock_memory_manager
            mock_get_pool.return_value = mock_pool
            mock_get_processor.return_value = mock_processor
            mock_get_plugin.return_value = mock_plugin
            
            mock_memory.return_value.percent = 60.0
            
            # 设置统计信息
            mock_pool.get_statistics.return_value = {'active_connections': 2}
            mock_processor.get_statistics.return_value = {
                'performance': {'throughput': 50.0, 'avg_response_time': 100.0},
                'total_requests': 100,
                'failed_requests': 1
            }
            mock_plugin.get_statistics.return_value = {'cache_hits': 90, 'cache_misses': 10}
            
            try:
                # 启动管理器
                await manager.start()
                assert manager._status.get() == ComponentStatus.RUNNING
                
                # 等待监控循环运行几次
                await asyncio.sleep(0.2)
                
                # 验证性能历史被收集
                assert len(manager._performance_history) > 0
                
                # 验证最新的性能指标
                latest_metrics = manager._performance_history[-1]
                assert latest_metrics.cpu_usage == 50.0
                assert latest_metrics.memory_usage == 60.0
                
            finally:
                # 停止管理器
                await manager.stop()
                assert manager._status.get() == ComponentStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_health_check_loop_execution(self):
        """测试健康检查循环执行"""
        config = ConcurrencyConfig(
            enable_health_check=True,
            health_check_interval=0.05  # 很短的间隔用于测试
        )
        
        manager = ConcurrencyManager(config)
        
        # 模拟组件
        with patch('harborai.core.optimizations.memory_manager.get_memory_manager') as mock_get_memory:
            mock_memory_manager = AsyncMock()
            mock_get_memory.return_value = mock_memory_manager
            
            try:
                # 启动管理器
                await manager.start()
                assert manager._status.get() == ComponentStatus.RUNNING
                
                # 注册一个健康的组件
                mock_component = Mock()
                mock_component.get_statistics = Mock(return_value={'status': 'healthy'})
                
                await manager._register_component(
                    "test_component",
                    mock_component,
                    ComponentStatus.RUNNING
                )
                
                # 等待健康检查运行几次
                await asyncio.sleep(0.2)
                
                # 验证组件仍然健康
                component_info = manager._components["test_component"]
                assert component_info.status.get() == ComponentStatus.RUNNING
                
            finally:
                # 停止管理器
                await manager.stop()
                assert manager._status.get() == ComponentStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_full_lifecycle_with_monitoring(self):
        """测试完整生命周期与监控"""
        config = ConcurrencyConfig(
            max_concurrent_requests=5,
            enable_adaptive_optimization=True,
            enable_health_check=True,
            health_check_interval=0.05
        )
        
        manager = ConcurrencyManager(config)
        
        # 模拟所有组件
        with patch('harborai.core.optimizations.memory_manager.get_memory_manager') as mock_get_memory, \
             patch('harborai.core.optimizations.concurrency_manager.get_connection_pool') as mock_get_pool, \
             patch('harborai.core.optimizations.concurrency_manager.get_request_processor') as mock_get_processor, \
             patch('harborai.core.optimizations.concurrency_manager.get_lockfree_plugin_manager') as mock_get_plugin, \
             patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_memory:
            
            # 设置模拟返回值
            mock_memory_manager = AsyncMock()
            mock_pool = Mock()
            mock_processor = Mock()
            mock_plugin = Mock()
            
            mock_get_memory.return_value = mock_memory_manager
            mock_get_pool.return_value = mock_pool
            mock_get_processor.return_value = mock_processor
            mock_get_plugin.return_value = mock_plugin
            
            mock_memory.return_value.percent = 60.0
            
            # 设置统计信息
            mock_pool.get_statistics.return_value = {'active_connections': 2}
            mock_processor.get_statistics.return_value = {
                'performance': {'throughput': 50.0, 'avg_response_time': 100.0},
                'total_requests': 100,
                'failed_requests': 1
            }
            mock_plugin.get_statistics.return_value = {'cache_hits': 90, 'cache_misses': 10}
            
            try:
                # 启动管理器
                await manager.start()
                assert manager._status.get() == ComponentStatus.RUNNING
                
                # 等待一小段时间让监控任务运行
                await asyncio.sleep(0.1)
                
                # 检查统计信息
                stats = manager.get_statistics()
                assert stats['manager_status'] == ComponentStatus.RUNNING.value
                assert len(stats['components']) > 0
                
                # 验证组件已注册
                assert 'memory_manager' in manager._components
                assert 'connection_pool' in manager._components
                assert 'request_processor' in manager._components
                assert 'plugin_manager' in manager._components
                
            finally:
                # 停止管理器
                await manager.stop()
                assert manager._status.get() == ComponentStatus.STOPPED


class TestGlobalConcurrencyManager:
    """测试全局并发管理器"""
    
    @pytest.mark.asyncio
    async def test_get_concurrency_manager_singleton(self):
        """测试获取单例并发管理器"""
        # 重置全局管理器
        await reset_concurrency_manager()
        
        # 模拟组件
        with patch('harborai.core.optimizations.memory_manager.get_memory_manager') as mock_get_memory:
            mock_memory = AsyncMock()
            mock_get_memory.return_value = mock_memory
            
            # 第一次获取
            manager1 = await get_concurrency_manager()
            assert manager1 is not None
            assert manager1._status.get() == ComponentStatus.RUNNING
            
            # 第二次获取应该返回同一个实例
            manager2 = await get_concurrency_manager()
            assert manager2 is manager1
            
            # 清理
            await reset_concurrency_manager()
    
    @pytest.mark.asyncio
    async def test_get_concurrency_manager_with_config(self):
        """测试使用配置获取并发管理器"""
        # 重置全局管理器
        await reset_concurrency_manager()
        
        config = ConcurrencyConfig(max_concurrent_requests=20)
        
        # 模拟组件
        with patch('harborai.core.optimizations.memory_manager.get_memory_manager') as mock_get_memory:
            mock_memory = AsyncMock()
            mock_get_memory.return_value = mock_memory
            
            # 获取管理器
            manager = await get_concurrency_manager(config)
            assert manager is not None
            assert manager.config.max_concurrent_requests == 20
            
            # 清理
            await reset_concurrency_manager()
    
    @pytest.mark.asyncio
    async def test_reset_concurrency_manager(self):
        """测试重置并发管理器"""
        # 模拟组件
        with patch('harborai.core.optimizations.memory_manager.get_memory_manager') as mock_get_memory:
            mock_memory = AsyncMock()
            mock_get_memory.return_value = mock_memory
            
            # 获取管理器
            manager = await get_concurrency_manager()
            assert manager is not None
            assert manager._status.get() == ComponentStatus.RUNNING
            
            # 重置管理器
            await reset_concurrency_manager()
            
            # 再次获取应该是新的实例
            new_manager = await get_concurrency_manager()
            assert new_manager is not manager
            
            # 清理
            await reset_concurrency_manager()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])