#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发管理器

统一管理所有并发优化组件，包括无锁插件管理器、优化连接池、异步请求处理器等。
提供统一的并发优化接口和性能监控。

设计原则：
1. 统一管理并发组件
2. 协调组件间的交互
3. 提供统一的配置接口
4. 实时性能监控
5. 自适应优化策略
6. 故障恢复和降级
7. 资源管理和清理

技术特性：
- 组件生命周期管理
- 统一配置管理
- 性能监控和调优
- 自适应负载均衡
- 故障检测和恢复
- 资源池管理
- 并发控制策略
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import weakref
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc

from .lockfree_plugin_manager import LockFreePluginManager, get_lockfree_plugin_manager, AtomicInteger, AtomicReference
from .optimized_connection_pool import OptimizedConnectionPool, get_connection_pool, PoolConfig
from .async_request_processor import AsyncRequestProcessor, get_request_processor, RequestConfig, RequestPriority
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class ConcurrencyMode(Enum):
    """并发模式枚举"""
    CONSERVATIVE = "conservative"    # 保守模式：稳定优先
    BALANCED = "balanced"           # 平衡模式：性能与稳定平衡
    AGGRESSIVE = "aggressive"       # 激进模式：性能优先
    ADAPTIVE = "adaptive"           # 自适应模式：根据负载动态调整


class ComponentStatus(Enum):
    """组件状态枚举"""
    STOPPED = "stopped"             # 已停止
    STARTING = "starting"           # 启动中
    RUNNING = "running"             # 运行中
    STOPPING = "stopping"          # 停止中
    ERROR = "error"                 # 错误状态
    DEGRADED = "degraded"           # 降级运行


@dataclass
class ConcurrencyConfig:
    """并发配置"""
    max_concurrent_requests: int = 100  # 提升到100个并发请求
    connection_pool_size: int = 50      # 提升连接池大小到50
    request_timeout: float = 15.0       # 减少超时时间到15秒
    enable_adaptive_optimization: bool = True
    enable_health_check: bool = True
    health_check_interval: float = 30.0  # 减少健康检查间隔到30秒
    memory_threshold_mb: int = 2048      # 提升内存阈值到2GB
    cpu_threshold_percent: float = 85.0  # 提升CPU阈值到85%
    response_time_threshold_ms: float = 500.0  # 减少响应时间阈值到500ms
    error_rate_threshold_percent: float = 3.0   # 减少错误率阈值到3%
    max_recovery_attempts: int = 5       # 增加恢复尝试次数


@dataclass
class ComponentInfo:
    """组件信息"""
    name: str
    status: 'AtomicReference'
    instance: Any
    start_time: Optional[float] = None
    error_count: 'AtomicInteger' = field(default_factory=lambda: AtomicInteger(0))
    last_error: Optional[str] = None
    recovery_attempts: 'AtomicInteger' = field(default_factory=lambda: AtomicInteger(0))


@dataclass
class PerformanceMetrics:
    """性能指标"""
    timestamp: float
    cpu_usage: float                    # CPU使用率
    memory_usage: float                 # 内存使用率
    active_threads: int                 # 活跃线程数
    active_connections: int             # 活跃连接数
    request_throughput: float           # 请求吞吐量
    avg_response_time: float            # 平均响应时间
    error_rate: float                   # 错误率
    plugin_cache_hit_rate: float        # 插件缓存命中率


class ConcurrencyManager:
    """并发管理器
    
    统一管理所有并发优化组件，提供协调、监控、自适应优化等功能。
    
    主要功能：
    1. 组件生命周期管理
    2. 统一配置和协调
    3. 性能监控和调优
    4. 自适应负载均衡
    5. 故障检测和恢复
    6. 资源管理和清理
    """
    
    def __init__(self, config: Optional[ConcurrencyConfig] = None):
        """初始化并发管理器
        
        Args:
            config: 并发配置
        """
        self.config = config or ConcurrencyConfig()
        
        # 组件管理
        self._components: Dict[str, ComponentInfo] = {}
        self._component_lock = threading.RLock()
        
        # 核心组件实例
        self._plugin_manager: Optional[LockFreePluginManager] = None
        self._connection_pool: Optional[OptimizedConnectionPool] = None
        self._request_processor: Optional[AsyncRequestProcessor] = None
        self._memory_manager: Optional[MemoryManager] = None
        
        # 线程池
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        
        # 监控和统计
        self._performance_history: List[PerformanceMetrics] = []
        self._performance_lock = threading.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        # 状态管理
        self._status = AtomicReference(ComponentStatus.STOPPED)
        self._running = False
        self._start_time: Optional[float] = None
        
        # 自适应优化
        self._optimization_history: List[Dict[str, Any]] = []
        self._last_optimization: float = 0
        
        logger.info("ConcurrencyManager初始化完成，最大并发请求: %d", self.config.max_concurrent_requests)
    
    async def start(self):
        """启动并发管理器"""
        if self._running:
            return
        
        try:
            self._status.set(ComponentStatus.STARTING)
            self._running = True
            self._start_time = time.time()
            
            # 初始化线程池
            self._thread_pool = ThreadPoolExecutor(
                max_workers=min(self.config.max_concurrent_requests, 64),  # 提升最大工作线程数
                thread_name_prefix="concurrency-"
            )
            
            # 启动核心组件
            await self._start_components()
            
            # 启动监控任务
            if self.config.enable_adaptive_optimization:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            # 启动健康检查任务
            if self.config.enable_health_check:
                self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            self._status.set(ComponentStatus.RUNNING)
            logger.info("ConcurrencyManager已启动")
            
        except Exception as e:
            self._status.set(ComponentStatus.ERROR)
            logger.error("ConcurrencyManager启动失败: %s", str(e))
            await self.stop()
            raise
    
    async def stop(self):
        """停止并发管理器"""
        if not self._running:
            return
        
        try:
            self._status.set(ComponentStatus.STOPPING)
            self._running = False
            
            # 停止监控任务
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            # 停止核心组件
            await self._stop_components()
            
            # 关闭线程池
            if self._thread_pool:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
            
            self._status.set(ComponentStatus.STOPPED)
            logger.info("ConcurrencyManager已停止")
            
        except Exception as e:
            self._status.set(ComponentStatus.ERROR)
            logger.error("ConcurrencyManager停止失败: %s", str(e))
    
    async def _start_components(self):
        """启动核心组件"""
        # 启动内存管理器
        try:
            from .memory_manager import get_memory_manager
            self._memory_manager = await get_memory_manager()
            await self._register_component(
                "memory_manager", 
                self._memory_manager, 
                ComponentStatus.RUNNING
            )
            logger.info("内存管理器已启动")
        except Exception as e:
            logger.error("启动内存管理器失败: %s", str(e))
        
        # 启动连接池
        try:
            pool_config = PoolConfig(
                max_size=self.config.connection_pool_size,
                connection_timeout=self.config.request_timeout
            )
            self._connection_pool = await get_connection_pool(pool_config)
            await self._register_component(
                "connection_pool", 
                self._connection_pool, 
                ComponentStatus.RUNNING
            )
            logger.info("连接池已启动")
        except Exception as e:
            logger.error("启动连接池失败: %s", str(e))
        
        # 启动异步请求处理器
        try:
            request_config = RequestConfig(
                timeout=self.config.request_timeout
            )
            self._request_processor = await get_request_processor(request_config)
            await self._register_component(
                "request_processor", 
                self._request_processor, 
                ComponentStatus.RUNNING
            )
            logger.info("异步请求处理器已启动")
        except Exception as e:
            logger.error("启动异步请求处理器失败: %s", str(e))
        
        # 启动无锁插件管理器
        try:
            self._plugin_manager = await get_lockfree_plugin_manager()
            await self._register_component(
                "plugin_manager", 
                self._plugin_manager, 
                ComponentStatus.RUNNING
            )
            logger.info("无锁插件管理器已启动")
        except Exception as e:
            logger.error("启动无锁插件管理器失败: %s", str(e))
    
    async def _stop_components(self):
        """停止核心组件"""
        components_to_stop = list(self._components.keys())
        
        for component_name in components_to_stop:
            try:
                await self._stop_component(component_name)
            except Exception as e:
                logger.error("停止组件失败 %s: %s", component_name, str(e))
    
    async def _register_component(self, name: str, instance: Any, status: ComponentStatus):
        """注册组件
        
        Args:
            name: 组件名称
            instance: 组件实例
            status: 组件状态
        """
        with self._component_lock:
            component_info = ComponentInfo(
                name=name,
                status=AtomicReference(status),
                instance=instance,
                start_time=time.time() if status == ComponentStatus.RUNNING else None
            )
            self._components[name] = component_info
            logger.debug("注册组件: %s", name)
    
    async def _stop_component(self, name: str):
        """停止组件
        
        Args:
            name: 组件名称
        """
        with self._component_lock:
            if name not in self._components:
                return
            
            component_info = self._components[name]
            component_info.status.set(ComponentStatus.STOPPING)
            
            try:
                instance = component_info.instance
                
                # 调用组件的停止方法
                if hasattr(instance, 'stop') and callable(instance.stop):
                    if asyncio.iscoroutinefunction(instance.stop):
                        await instance.stop()
                    else:
                        instance.stop()
                elif hasattr(instance, 'shutdown') and callable(instance.shutdown):
                    if asyncio.iscoroutinefunction(instance.shutdown):
                        await instance.shutdown()
                    else:
                        instance.shutdown()
                
                component_info.status.set(ComponentStatus.STOPPED)
                logger.debug("停止组件: %s", name)
                
            except Exception as e:
                component_info.status.set(ComponentStatus.ERROR)
                component_info.error_count.increment()
                component_info.last_error = str(e)
                logger.error("停止组件失败 %s: %s", name, str(e))
                raise
    
    async def _monitoring_loop(self):
        """监控循环"""
        logger.debug("启动性能监控")
        
        while self._running:
            try:
                # 收集性能指标
                metrics = await self._collect_performance_metrics()
                
                # 记录性能历史
                with self._performance_lock:
                    self._performance_history.append(metrics)
                    
                    # 保持历史记录在合理范围内
                    max_history = 60  # 保持60个历史记录
                    if len(self._performance_history) > max_history:
                        self._performance_history = self._performance_history[-max_history:]
                
                # 自适应优化
                if self.config.enable_adaptive_optimization:
                    await self._adaptive_optimization(metrics)
                
                await asyncio.sleep(5.0)  # 每5秒监控一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("监控循环异常: %s", str(e))
                await asyncio.sleep(1.0)
        
        logger.debug("性能监控已停止")
    
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """收集性能指标
        
        Returns:
            性能指标
        """
        # 系统指标
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        
        # 线程指标
        active_threads = threading.active_count()
        
        # 连接池指标
        active_connections = 0
        if self._connection_pool:
            pool_stats = self._connection_pool.get_statistics()
            active_connections = pool_stats.get('active_connections', 0)
        
        # 请求处理器指标
        request_throughput = 0.0
        avg_response_time = 0.0
        error_rate = 0.0
        if self._request_processor:
            processor_stats = self._request_processor.get_statistics()
            performance = processor_stats.get('performance', {})
            request_throughput = performance.get('throughput', 0.0)
            avg_response_time = performance.get('avg_response_time', 0.0)
            
            total_requests = processor_stats.get('total_requests', 0)
            failed_requests = processor_stats.get('failed_requests', 0)
            error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        # 插件管理器指标
        plugin_cache_hit_rate = 0.0
        if self._plugin_manager:
            plugin_stats = self._plugin_manager.get_statistics()
            cache_hits = plugin_stats.get('cache_hits', 0)
            cache_misses = plugin_stats.get('cache_misses', 0)
            total_cache_requests = cache_hits + cache_misses
            plugin_cache_hit_rate = (cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_threads=active_threads,
            active_connections=active_connections,
            request_throughput=request_throughput,
            avg_response_time=avg_response_time,
            error_rate=error_rate,
            plugin_cache_hit_rate=plugin_cache_hit_rate
        )
    
    async def _adaptive_optimization(self, metrics: PerformanceMetrics):
        """自适应优化
        
        Args:
            metrics: 性能指标
        """
        current_time = time.time()
        
        # 避免过于频繁的优化
        if current_time - self._last_optimization < 30.0:
            return
        
        try:
            optimizations = []
            
            # CPU使用率优化
            if metrics.cpu_usage > self.config.cpu_threshold_percent:
                # CPU使用率过高，减少并发度
                if self._request_processor:
                    # 可以考虑减少工作线程数或增加批处理延迟
                    optimizations.append("reduce_concurrency")
            elif metrics.cpu_usage < 30.0:
                # CPU使用率较低，可以增加并发度
                if self._request_processor:
                    optimizations.append("increase_concurrency")
            
            # 内存使用率优化
            if metrics.memory_usage > (self.config.memory_threshold_mb / 1024 * 100):
                # 内存使用率过高，触发清理
                if self._memory_manager:
                    await self._memory_manager.cleanup()
                    optimizations.append("memory_cleanup")
                
                # 强制垃圾回收
                gc.collect()
                optimizations.append("force_gc")
            
            # 响应时间优化
            if metrics.avg_response_time > self.config.response_time_threshold_ms:
                # 响应时间过长，优化连接池
                if self._connection_pool:
                    # 可以考虑增加连接池大小
                    optimizations.append("optimize_connection_pool")
            
            # 错误率优化
            if metrics.error_rate > self.config.error_rate_threshold_percent:
                # 错误率过高，启用降级策略
                optimizations.append("enable_degradation")
            
            # 记录优化历史
            if optimizations:
                optimization_record = {
                    'timestamp': current_time,
                    'metrics': metrics,
                    'optimizations': optimizations
                }
                self._optimization_history.append(optimization_record)
                
                # 保持优化历史在合理范围内
                if len(self._optimization_history) > 100:
                    self._optimization_history = self._optimization_history[-100:]
                
                logger.info("执行自适应优化: %s", optimizations)
                self._last_optimization = current_time
        
        except Exception as e:
            logger.error("自适应优化失败: %s", str(e))
    
    async def _health_check_loop(self):
        """健康检查循环"""
        logger.debug("启动健康检查")
        
        while self._running:
            try:
                await self._check_component_health()
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("健康检查循环异常: %s", str(e))
                await asyncio.sleep(1.0)
        
        logger.debug("健康检查已停止")
    
    async def _check_component_health(self):
        """检查组件健康状态"""
        with self._component_lock:
            for name, component_info in self._components.items():
                try:
                    status = component_info.status.get()
                    
                    if status == ComponentStatus.ERROR:
                        # 组件处于错误状态，尝试恢复
                        recovery_attempts = component_info.recovery_attempts.get()
                        
                        if recovery_attempts < self.config.max_recovery_attempts:
                            logger.warning("尝试恢复组件: %s (第%d次)", name, recovery_attempts + 1)
                            await self._recover_component(name, component_info)
                        else:
                            logger.error("组件恢复失败，已达到最大尝试次数: %s", name)
                            component_info.status.set(ComponentStatus.DEGRADED)
                    
                    elif status == ComponentStatus.RUNNING:
                        # 检查组件是否正常工作
                        if hasattr(component_info.instance, 'health_check'):
                            try:
                                health_result = component_info.instance.health_check()
                                if asyncio.iscoroutine(health_result):
                                    health_result = await health_result
                                
                                if not health_result:
                                    logger.warning("组件健康检查失败: %s", name)
                                    component_info.status.set(ComponentStatus.ERROR)
                                    component_info.error_count.increment()
                                    component_info.last_error = "Health check failed"
                            
                            except Exception as e:
                                logger.error("组件健康检查异常 %s: %s", name, str(e))
                                component_info.status.set(ComponentStatus.ERROR)
                                component_info.error_count.increment()
                                component_info.last_error = str(e)
                
                except Exception as e:
                    logger.error("检查组件健康状态失败 %s: %s", name, str(e))
    
    async def _recover_component(self, name: str, component_info: ComponentInfo):
        """恢复组件
        
        Args:
            name: 组件名称
            component_info: 组件信息
        """
        try:
            component_info.recovery_attempts.increment()
            component_info.status.set(ComponentStatus.STARTING)
            
            # 重新启动组件
            instance = component_info.instance
            
            # 先停止组件
            if hasattr(instance, 'stop') and callable(instance.stop):
                if asyncio.iscoroutinefunction(instance.stop):
                    await instance.stop()
                else:
                    instance.stop()
            
            # 等待一段时间
            await asyncio.sleep(1.0)
            
            # 重新启动组件
            if hasattr(instance, 'start') and callable(instance.start):
                if asyncio.iscoroutinefunction(instance.start):
                    await instance.start()
                else:
                    instance.start()
            
            component_info.status.set(ComponentStatus.RUNNING)
            component_info.start_time = time.time()
            logger.info("组件恢复成功: %s", name)
            
        except Exception as e:
            component_info.status.set(ComponentStatus.ERROR)
            component_info.error_count.increment()
            component_info.last_error = str(e)
            logger.error("组件恢复失败 %s: %s", name, str(e))
    
    async def submit_request(self, method: str, url: str, **kwargs) -> Any:
        """提交异步请求
        
        Args:
            method: HTTP方法
            url: 请求URL
            **kwargs: 其他请求参数
            
        Returns:
            请求结果
        """
        if not self._request_processor:
            raise RuntimeError("异步请求处理器未启用")
        
        return await self._request_processor.submit_request(method, url, **kwargs)
    
    async def create_chat_completion(self, model: str, messages: list, **kwargs) -> Any:
        """创建聊天完成（并发优化版本）
        
        Args:
            model: 模型名称
            messages: 消息列表
            **kwargs: 其他参数
            
        Returns:
            聊天完成结果
        """
        # 获取插件
        plugin = self.get_plugin(model)
        if not plugin:
            raise ValueError(f"未找到模型 {model} 的插件")
        
        # 使用异步请求处理器
        if self._request_processor:
            return await plugin.create_async(
                model=model,
                messages=messages,
                request_processor=self._request_processor,
                **kwargs
            )
        else:
            # 回退到传统方式
            return await plugin.create_async(
                model=model,
                messages=messages,
                **kwargs
            )
    
    def get_plugin(self, model: str) -> Any:
        """获取插件实例
        
        Args:
            model: 模型名称
            
        Returns:
            插件实例
        """
        if not self._plugin_manager:
            raise RuntimeError("无锁插件管理器未启用")
        
        return self._plugin_manager.get_plugin_for_model(model)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "manager_status": self._status.get().value,
            "uptime": time.time() - self._start_time if self._start_time else 0,
            "config": {
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "connection_pool_size": self.config.connection_pool_size,
                "request_timeout": self.config.request_timeout,
                "enable_adaptive_optimization": self.config.enable_adaptive_optimization,
                "enable_health_check": self.config.enable_health_check,
            },
            "components": {},
            "performance": {},
            "optimization_history": len(self._optimization_history),
        }
        
        # 组件统计
        with self._component_lock:
            for name, component_info in self._components.items():
                stats["components"][name] = {
                    "status": component_info.status.get().value,
                    "error_count": component_info.error_count.get(),
                    "recovery_attempts": component_info.recovery_attempts.get(),
                    "last_error": component_info.last_error,
                    "uptime": time.time() - component_info.start_time if component_info.start_time else 0,
                }
                
                # 获取组件详细统计
                if hasattr(component_info.instance, 'get_statistics'):
                    try:
                        component_stats = component_info.instance.get_statistics()
                        stats["components"][name]["details"] = component_stats
                    except Exception as e:
                        logger.debug("获取组件统计失败 %s: %s", name, str(e))
        
        # 性能统计
        with self._performance_lock:
            if self._performance_history:
                latest_metrics = self._performance_history[-1]
                stats["performance"] = {
                    "cpu_usage": latest_metrics.cpu_usage,
                    "memory_usage": latest_metrics.memory_usage,
                    "active_threads": latest_metrics.active_threads,
                    "active_connections": latest_metrics.active_connections,
                    "request_throughput": latest_metrics.request_throughput,
                    "avg_response_time": latest_metrics.avg_response_time,
                    "error_rate": latest_metrics.error_rate,
                    "plugin_cache_hit_rate": latest_metrics.plugin_cache_hit_rate,
                }
                
                # 计算趋势
                if len(self._performance_history) >= 2:
                    prev_metrics = self._performance_history[-2]
                    stats["performance"]["trends"] = {
                        "cpu_usage_trend": latest_metrics.cpu_usage - prev_metrics.cpu_usage,
                        "memory_usage_trend": latest_metrics.memory_usage - prev_metrics.memory_usage,
                        "throughput_trend": latest_metrics.request_throughput - prev_metrics.request_throughput,
                        "response_time_trend": latest_metrics.avg_response_time - prev_metrics.avg_response_time,
                    }
        
        return stats
    
    def get_performance_history(self, duration: float = 300.0) -> List[PerformanceMetrics]:
        """获取性能历史
        
        Args:
            duration: 历史时长（秒）
            
        Returns:
            性能指标列表
        """
        current_time = time.time()
        cutoff_time = current_time - duration
        
        with self._performance_lock:
            return [
                metrics for metrics in self._performance_history
                if metrics.timestamp >= cutoff_time
            ]


# 全局并发管理器实例
_global_concurrency_manager: Optional[ConcurrencyManager] = None
_manager_ref = AtomicReference(None)


async def get_concurrency_manager(config: Optional[ConcurrencyConfig] = None) -> ConcurrencyManager:
    """获取全局并发管理器实例
    
    Args:
        config: 并发配置
        
    Returns:
        并发管理器实例
    """
    manager = _manager_ref.get()
    
    if manager is None:
        # 创建新管理器
        new_manager = ConcurrencyManager(config)
        
        # 使用CAS操作设置全局实例
        if _manager_ref.compare_and_swap(None, new_manager):
            await new_manager.start()
            return new_manager
        else:
            # 其他协程已经创建了实例
            await new_manager.stop()  # 清理未使用的实例
            return _manager_ref.get()
    
    return manager


async def reset_concurrency_manager():
    """重置全局并发管理器
    
    主要用于测试场景。
    """
    manager = _manager_ref.get()
    
    if manager is not None:
        await manager.stop()
        _manager_ref.set(None)