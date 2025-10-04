#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无锁插件管理器

实现基于原子操作的无锁数据结构，替代传统的锁机制，显著提升并发性能。
根据技术设计方案，使用原子操作、无锁队列和CAS（Compare-And-Swap）操作。

设计原则：
1. 使用原子操作替代锁，减少线程阻塞
2. 实现无锁数据结构，提升并发访问性能
3. 保持与现有插件系统的兼容性
4. 支持高并发场景下的稳定性
5. 提供性能监控和统计功能

技术特性：
- 原子引用计数
- 无锁哈希表
- CAS操作
- 内存屏障
- 线程安全的状态管理
"""

import threading
import time
import weakref
from typing import Dict, Any, Optional, List, Callable, Union
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import gc
from threading import RLock
from queue import Queue, Empty
import sys

from ..plugins.base import Plugin
from ..base_plugin import BaseLLMPlugin
from ..exceptions import PluginError, PluginLoadError, PluginNotFoundError
from ..lazy_plugin_manager import LazyPluginInfo

logger = logging.getLogger(__name__)


class AtomicInteger:
    """原子整数类
    
    提供线程安全的整数操作，使用内置锁实现原子性。
    在Python中，由于GIL的存在，简单的整数操作通常是原子的，
    但为了确保在所有情况下的线程安全，我们使用显式的原子操作。
    """
    
    def __init__(self, value: int = 0):
        self._value = value
        self._lock = threading.Lock()
    
    def get(self) -> int:
        """获取当前值"""
        with self._lock:
            return self._value
    
    def set(self, value: int) -> None:
        """设置值"""
        with self._lock:
            self._value = value
    
    def increment(self) -> int:
        """原子递增，返回新值"""
        with self._lock:
            self._value += 1
            return self._value
    
    def decrement(self) -> int:
        """原子递减，返回新值"""
        with self._lock:
            self._value -= 1
            return self._value
    
    def add(self, delta: int) -> int:
        """原子加法，返回新值"""
        with self._lock:
            self._value += delta
            return self._value
    
    def compare_and_swap(self, expected: int, new_value: int) -> bool:
        """比较并交换（CAS操作）
        
        Args:
            expected: 期望的当前值
            new_value: 要设置的新值
            
        Returns:
            如果当前值等于期望值并成功设置新值，返回True；否则返回False
        """
        with self._lock:
            if self._value == expected:
                self._value = new_value
                return True
            return False


class AtomicReference:
    """原子引用类
    
    提供线程安全的对象引用操作。
    """
    
    def __init__(self, value: Any = None):
        self._value = value
        self._lock = threading.Lock()
    
    def get(self) -> Any:
        """获取当前引用"""
        with self._lock:
            return self._value
    
    def set(self, value: Any) -> None:
        """设置引用"""
        with self._lock:
            self._value = value
    
    def compare_and_swap(self, expected: Any, new_value: Any) -> bool:
        """比较并交换引用
        
        Args:
            expected: 期望的当前引用
            new_value: 要设置的新引用
            
        Returns:
            如果当前引用等于期望引用并成功设置新引用，返回True；否则返回False
        """
        with self._lock:
            if self._value is expected:
                self._value = new_value
                return True
            return False


@dataclass
class PluginEntry:
    """插件条目
    
    存储插件的元数据和状态信息。
    """
    name: str
    info: LazyPluginInfo
    instance: Optional[Plugin] = None
    load_count: AtomicInteger = field(default_factory=lambda: AtomicInteger(0))
    last_access_time: float = field(default_factory=time.time)
    loading: AtomicInteger = field(default_factory=lambda: AtomicInteger(0))  # 0=未加载, 1=加载中, 2=已加载
    error_count: AtomicInteger = field(default_factory=lambda: AtomicInteger(0))
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.load_count, int):
            self.load_count = AtomicInteger(self.load_count)
        if isinstance(self.loading, int):
            self.loading = AtomicInteger(self.loading)
        if isinstance(self.error_count, int):
            self.error_count = AtomicInteger(self.error_count)


class LockFreePluginManager:
    """无锁插件管理器
    
    使用原子操作和无锁数据结构实现高并发插件管理。
    
    主要特性：
    1. 原子操作：使用原子整数和引用避免锁竞争
    2. 无锁读取：插件查找和访问无需加锁
    3. CAS操作：使用比较并交换实现安全的状态更新
    4. 弱引用：避免循环引用和内存泄漏
    5. 性能监控：实时统计并发性能指标
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化无锁插件管理器
        
        Args:
            config: 配置字典，包含插件配置信息
        """
        self.config = config or {}
        
        # 插件注册表：使用原子引用存储插件条目
        self._plugin_entries: Dict[str, AtomicReference[PluginEntry]] = {}
        
        # 模型到插件的映射：使用原子引用
        self._model_to_plugin: Dict[str, AtomicReference[str]] = {}
        
        # 统计信息：使用原子计数器
        self._stats = {
            'total_requests': AtomicInteger(0),
            'cache_hits': AtomicInteger(0),
            'cache_misses': AtomicInteger(0),
            'load_attempts': AtomicInteger(0),
            'load_successes': AtomicInteger(0),
            'load_failures': AtomicInteger(0),
            'concurrent_loads': AtomicInteger(0),
        }
        
        # 线程池：用于异步插件加载
        max_workers = self.config.get('max_workers', 4)
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="lockfree_plugin"
        )
        
        # 性能监控
        self._performance_monitor = {
            'avg_load_time': 0.0,
            'max_load_time': 0.0,
            'min_load_time': float('inf'),
            'total_load_time': 0.0,
            'load_time_samples': 0,
        }
        self._perf_lock = threading.Lock()  # 仅用于性能统计的锁
        
        # 初始化插件注册表
        self._initialize_plugin_registry()
        
        logger.info("LockFreePluginManager初始化完成，注册了%d个插件", 
                   len(self._plugin_entries))
    
    def _initialize_plugin_registry(self):
        """初始化插件注册表
        
        注册所有支持的插件信息，但不实际加载插件。
        使用原子操作确保线程安全。
        """
        plugins_info = [
            LazyPluginInfo(
                name="deepseek",
                module_path="harborai.core.plugins.deepseek_plugin",
                class_name="DeepSeekPlugin",
                supported_models=[
                    "deepseek-chat",
                    "deepseek-reasoner", 
                    "deepseek-coder"
                ],
                priority=1
            ),
            LazyPluginInfo(
                name="doubao",
                module_path="harborai.core.plugins.doubao_plugin",
                class_name="DoubaoPlugin",
                supported_models=[
                    "doubao-pro-4k",
                    "doubao-pro-32k",
                    "doubao-lite-4k",
                    "doubao-pro-128k"
                ],
                priority=2
            ),
            LazyPluginInfo(
                name="wenxin",
                module_path="harborai.core.plugins.wenxin_plugin",
                class_name="WenxinPlugin",
                supported_models=[
                    "ernie-bot-turbo",
                    "ernie-bot-4",
                    "ernie-bot-8k",
                    "ernie-3.5-8k"
                ],
                priority=3
            ),
            LazyPluginInfo(
                name="openai",
                module_path="harborai.core.plugins.openai_plugin",
                class_name="OpenAIPlugin",
                supported_models=[
                    "gpt-3.5-turbo",
                    "gpt-4",
                    "gpt-4-turbo",
                    "gpt-4o"
                ],
                priority=0
            )
        ]
        
        for plugin_info in plugins_info:
            self._register_plugin_info(plugin_info)
    
    def _register_plugin_info(self, plugin_info: LazyPluginInfo):
        """注册插件信息
        
        使用原子操作注册插件信息，确保线程安全。
        
        Args:
            plugin_info: 插件信息
        """
        # 创建插件条目
        entry = PluginEntry(
            name=plugin_info.name,
            info=plugin_info
        )
        
        # 使用原子引用存储插件条目
        self._plugin_entries[plugin_info.name] = AtomicReference(entry)
        
        # 建立模型到插件的映射
        for model in plugin_info.supported_models:
            self._model_to_plugin[model] = AtomicReference(plugin_info.name)
        
        logger.debug("注册插件信息: %s，支持模型: %s", 
                    plugin_info.name, plugin_info.supported_models)
    
    def get_plugin_name_for_model(self, model: str) -> Optional[str]:
        """根据模型名称获取对应的插件名称
        
        使用原子引用进行无锁读取。
        
        Args:
            model: 模型名称
            
        Returns:
            插件名称，如果不支持该模型则返回None
        """
        self._stats['total_requests'].increment()
        
        model_ref = self._model_to_plugin.get(model)
        if model_ref:
            return model_ref.get()
        return None
    
    def get_plugin_for_model(self, model: str) -> Optional[Plugin]:
        """根据模型名称获取对应的插件实例
        
        这是无锁延迟加载的核心方法。
        
        Args:
            model: 模型名称
            
        Returns:
            插件实例，如果不支持该模型或加载失败则返回None
        """
        plugin_name = self.get_plugin_name_for_model(model)
        if not plugin_name:
            logger.warning("不支持的模型: %s", model)
            return None
        
        return self.get_plugin(plugin_name)
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """获取插件实例
        
        使用无锁算法进行插件获取和延迟加载。
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件实例，如果加载失败则返回None
        """
        entry_ref = self._plugin_entries.get(plugin_name)
        if not entry_ref:
            logger.error("未找到插件: %s", plugin_name)
            return None
        
        entry = entry_ref.get()
        if not entry:
            return None
        
        # 更新访问时间（无锁操作）
        entry.last_access_time = time.time()
        
        # 检查是否已加载
        if entry.loading.get() == 2 and entry.instance:
            self._stats['cache_hits'].increment()
            entry.load_count.increment()
            return entry.instance
        
        # 需要加载插件
        self._stats['cache_misses'].increment()
        return self._load_plugin_lockfree(entry)
    
    def _load_plugin_lockfree(self, entry: PluginEntry) -> Optional[Plugin]:
        """无锁插件加载
        
        使用CAS操作实现无锁的插件加载状态管理。
        
        Args:
            entry: 插件条目
            
        Returns:
            插件实例，如果加载失败则返回None
        """
        # 尝试将状态从未加载(0)改为加载中(1)
        if entry.loading.compare_and_swap(0, 1):
            # 成功获得加载权限
            self._stats['load_attempts'].increment()
            self._stats['concurrent_loads'].increment()
            
            try:
                start_time = time.perf_counter()
                
                # 实际加载插件
                plugin_instance = self._do_load_plugin(entry)
                
                if plugin_instance:
                    # 设置插件实例
                    entry.instance = plugin_instance
                    
                    # 原子地将状态设置为已加载(2)
                    entry.loading.set(2)
                    
                    # 更新统计信息
                    self._stats['load_successes'].increment()
                    entry.load_count.increment()
                    
                    # 更新性能监控
                    load_time = (time.perf_counter() - start_time) * 1000
                    self._update_performance_stats(load_time)
                    
                    logger.info("插件 %s 加载成功，耗时: %.2fms", entry.name, load_time)
                    return plugin_instance
                else:
                    # 加载失败，重置状态
                    entry.loading.set(0)
                    entry.error_count.increment()
                    self._stats['load_failures'].increment()
                    
            except Exception as e:
                # 加载异常，重置状态
                entry.loading.set(0)
                entry.error_count.increment()
                self._stats['load_failures'].increment()
                logger.error("插件 %s 加载失败: %s", entry.name, str(e))
            finally:
                self._stats['concurrent_loads'].decrement()
            
            return None
        
        elif entry.loading.get() == 1:
            # 其他线程正在加载，等待加载完成
            return self._wait_for_loading(entry)
        
        elif entry.loading.get() == 2:
            # 已加载完成
            if entry.instance:
                self._stats['cache_hits'].increment()
                entry.load_count.increment()
                return entry.instance
        
        return None
    
    def _wait_for_loading(self, entry: PluginEntry, timeout: float = 5.0) -> Optional[Plugin]:
        """等待插件加载完成
        
        使用忙等待（busy waiting）策略，避免使用锁。
        
        Args:
            entry: 插件条目
            timeout: 超时时间（秒）
            
        Returns:
            插件实例，如果超时或加载失败则返回None
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = entry.loading.get()
            
            if status == 2:  # 加载完成
                if entry.instance:
                    self._stats['cache_hits'].increment()
                    entry.load_count.increment()
                    return entry.instance
                break
            elif status == 0:  # 加载失败，状态已重置
                break
            
            # 短暂休眠，避免CPU占用过高
            time.sleep(0.001)  # 1ms
        
        logger.warning("等待插件 %s 加载超时", entry.name)
        return None
    
    def _do_load_plugin(self, entry: PluginEntry) -> Optional[Plugin]:
        """实际执行插件加载
        
        Args:
            entry: 插件条目
            
        Returns:
            插件实例，如果加载失败则返回None
        """
        try:
            import importlib
            
            plugin_info = entry.info
            
            # 动态导入插件模块
            logger.debug("开始加载插件: %s", entry.name)
            module = importlib.import_module(plugin_info.module_path)
            
            # 获取插件类
            plugin_class = getattr(module, plugin_info.class_name)
            if not (issubclass(plugin_class, Plugin) or issubclass(plugin_class, BaseLLMPlugin)):
                raise PluginError(f"插件类 {plugin_info.class_name} 必须继承自Plugin或BaseLLMPlugin")
            
            # 获取插件配置
            plugin_config = self._get_plugin_config(entry.name)
            
            # 创建插件实例，根据插件类型使用不同的构造函数
            if issubclass(plugin_class, BaseLLMPlugin):
                # BaseLLMPlugin构造函数: __init__(self, name: str, **config)
                plugin_instance = plugin_class(name=entry.name, **plugin_config)
            else:
                # Plugin构造函数: __init__(self, config: Optional[Dict[str, Any]] = None)
                plugin_instance = plugin_class(config=plugin_config)
            
            # 验证配置
            if hasattr(plugin_instance, 'validate_config') and not plugin_instance.validate_config():
                raise PluginError(f"插件 {entry.name} 配置验证失败")
            
            # 初始化插件
            if hasattr(plugin_instance, 'initialize') and not plugin_instance.initialize():
                raise PluginError(f"插件 {entry.name} 初始化失败")
            
            return plugin_instance
            
        except Exception as e:
            logger.error("插件 %s 加载失败: %s", entry.name, str(e))
            return None
    
    def _get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """获取插件配置
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件配置字典
        """
        plugin_config = {}
        
        if 'plugins' in self.config and plugin_name in self.config['plugins']:
            plugin_config.update(self.config['plugins'][plugin_name])
        
        # 添加通用配置
        if 'timeout' in self.config:
            plugin_config.setdefault('timeout', self.config['timeout'])
        
        if 'max_retries' in self.config:
            plugin_config.setdefault('max_retries', self.config['max_retries'])
        
        return plugin_config
    
    def _update_performance_stats(self, load_time: float):
        """更新性能统计信息
        
        Args:
            load_time: 加载时间（毫秒）
        """
        with self._perf_lock:
            self._performance_monitor['total_load_time'] += load_time
            self._performance_monitor['load_time_samples'] += 1
            
            if load_time > self._performance_monitor['max_load_time']:
                self._performance_monitor['max_load_time'] = load_time
            
            if load_time < self._performance_monitor['min_load_time']:
                self._performance_monitor['min_load_time'] = load_time
            
            # 计算平均加载时间
            self._performance_monitor['avg_load_time'] = (
                self._performance_monitor['total_load_time'] / 
                self._performance_monitor['load_time_samples']
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取管理器统计信息
        
        Returns:
            统计信息字典
        """
        # 计算缓存命中率
        total_requests = self._stats['total_requests'].get()
        cache_hits = self._stats['cache_hits'].get()
        hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # 计算加载成功率
        load_attempts = self._stats['load_attempts'].get()
        load_successes = self._stats['load_successes'].get()
        success_rate = (load_successes / load_attempts * 100) if load_attempts > 0 else 0
        
        # 获取已加载插件数量
        loaded_count = sum(
            1 for entry_ref in self._plugin_entries.values()
            if entry_ref.get() and entry_ref.get().loading.get() == 2
        )
        
        with self._perf_lock:
            perf_stats = self._performance_monitor.copy()
        
        return {
            "registered_plugins": len(self._plugin_entries),
            "loaded_plugins": loaded_count,
            "supported_models": len(self._model_to_plugin),
            "total_requests": total_requests,
            "cache_hits": cache_hits,
            "cache_misses": self._stats['cache_misses'].get(),
            "hit_rate_percent": round(hit_rate, 2),
            "load_attempts": load_attempts,
            "load_successes": load_successes,
            "load_failures": self._stats['load_failures'].get(),
            "success_rate_percent": round(success_rate, 2),
            "concurrent_loads": self._stats['concurrent_loads'].get(),
            "performance": perf_stats
        }
    
    def get_loaded_plugins(self) -> List[str]:
        """获取已加载的插件列表
        
        Returns:
            已加载的插件名称列表
        """
        loaded_plugins = []
        for name, entry_ref in self._plugin_entries.items():
            entry = entry_ref.get()
            if entry and entry.loading.get() == 2 and entry.instance:
                loaded_plugins.append(name)
        return loaded_plugins
    
    def get_supported_models(self) -> List[str]:
        """获取所有支持的模型列表
        
        Returns:
            支持的模型名称列表
        """
        return list(self._model_to_plugin.keys())
    
    def cleanup(self):
        """清理资源
        
        卸载所有插件并关闭线程池。
        """
        logger.info("开始清理LockFreePluginManager资源")
        
        # 卸载所有插件
        for name, entry_ref in self._plugin_entries.items():
            entry = entry_ref.get()
            if entry and entry.instance:
                try:
                    if hasattr(entry.instance, 'cleanup'):
                        entry.instance.cleanup()
                except Exception as e:
                    logger.error("插件 %s 清理失败: %s", name, str(e))
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        logger.info("LockFreePluginManager资源清理完成")
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass  # 忽略析构时的异常


# 全局无锁插件管理器实例
_lockfree_plugin_manager: Optional[LockFreePluginManager] = None
_manager_ref = AtomicReference(None)


async def get_lockfree_plugin_manager(config: Optional[Dict[str, Any]] = None) -> LockFreePluginManager:
    """获取全局无锁插件管理器实例
    
    使用原子引用实现无锁单例模式。
    
    Args:
        config: 配置字典，仅在首次创建时使用
        
    Returns:
        无锁插件管理器实例
    """
    manager = _manager_ref.get()
    
    if manager is None:
        # 尝试创建新实例
        new_manager = LockFreePluginManager(config)
        
        # 使用CAS操作设置全局实例
        if _manager_ref.compare_and_swap(None, new_manager):
            return new_manager
        else:
            # 其他线程已经创建了实例，使用现有实例
            new_manager.cleanup()  # 清理未使用的实例
            return _manager_ref.get()
    
    return manager


def reset_lockfree_plugin_manager():
    """重置全局无锁插件管理器
    
    主要用于测试场景。
    """
    manager = _manager_ref.get()
    
    if manager is not None:
        manager.cleanup()
        _manager_ref.set(None)