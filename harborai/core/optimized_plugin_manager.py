"""优化的插件管理器

实现预加载、缓存和性能优化的插件管理系统。
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import weakref

from harborai.config.settings import get_settings
from harborai.config.performance import get_performance_config
from .base_plugin import BaseLLMPlugin as BasePlugin
from .exceptions import PluginError, ValidationError, ConfigurationError
from .retry import retry, RetryConfig, FixedBackoff
from .background_tasks import get_background_processor
from .cache_manager import get_cache_manager

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """插件状态枚举"""
    UNLOADED = "unloaded"
    PRELOADING = "preloading"
    PRELOADED = "preloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class OptimizedPluginInfo:
    """优化的插件信息类"""
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    plugin_class: Optional[Type[BasePlugin]] = None
    module_path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: PluginStatus = PluginStatus.UNLOADED
    instance: Optional[BasePlugin] = None
    load_time: Optional[float] = None
    error_info: Optional[str] = None
    
    # 性能优化字段
    preload_priority: int = 0  # 预加载优先级
    lazy_load: bool = False  # 是否延迟加载
    cache_enabled: bool = True  # 是否启用缓存
    last_used: Optional[float] = None  # 最后使用时间
    usage_count: int = 0  # 使用次数
    avg_execution_time: float = 0.0  # 平均执行时间
    
    def update_usage_stats(self, execution_time: float):
        """更新使用统计"""
        self.last_used = time.time()
        self.usage_count += 1
        
        # 计算平均执行时间
        if self.avg_execution_time == 0.0:
            self.avg_execution_time = execution_time
        else:
            # 使用指数移动平均
            alpha = 0.1
            self.avg_execution_time = (alpha * execution_time + 
                                     (1 - alpha) * self.avg_execution_time)


class PluginCache:
    """插件缓存管理器"""
    
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._lock = threading.RLock()
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            if key not in self._cache:
                return None
                
            # 检查是否过期
            if time.time() - self._timestamps[key] > self.ttl:
                del self._cache[key]
                del self._timestamps[key]
                return None
                
            return self._cache[key]
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        with self._lock:
            # 检查缓存大小限制
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
    
    def _evict_lru(self) -> None:
        """淘汰最近最少使用的条目"""
        if not self._cache:
            return
            
        # 找到最旧的条目
        oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
        del self._cache[oldest_key]
        del self._timestamps[oldest_key]
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()


class OptimizedPluginManager:
    """优化的插件管理器
    
    实现预加载、缓存和性能优化功能。
    """
    
    def __init__(
        self,
        plugin_dirs: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        auto_load: bool = True,
        max_workers: int = 4,
        enable_preload: bool = None,
        enable_cache: bool = None,
        cache_size: int = 100
    ):
        """初始化优化的插件管理器"""
        # 从性能配置获取插件配置
        perf_config = get_performance_config()
        plugin_config = perf_config.get_plugin_config()
        
        self.plugin_dirs = plugin_dirs or []
        self.config = config or {}
        self.auto_load = auto_load
        self.max_workers = max_workers
        self.enable_preload = enable_preload if enable_preload is not None else plugin_config.get('enable_preload', True)
        self.enable_cache = enable_cache if enable_cache is not None else plugin_config.get('enable_cache', True)
        
        # 插件注册表
        self._plugins: Dict[str, OptimizedPluginInfo] = {}
        self._plugin_instances: Dict[str, BasePlugin] = {}
        self._plugin_hooks: Dict[str, List[Callable]] = {}
        
        # 性能优化组件
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="plugin")
        self._plugin_cache = PluginCache(max_size=cache_size) if enable_cache else None
        self._preload_queue: asyncio.Queue = asyncio.Queue()
        self._preload_task: Optional[asyncio.Task] = None
        
        # 统计信息
        self._stats = {
            'total_loads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'preload_count': 0,
            'avg_load_time': 0.0
        }
        
        # 弱引用管理，避免内存泄漏
        self._weak_refs: Set[weakref.ref] = set()
        
        # 从配置获取插件目录
        settings = get_settings()
        plugin_config_settings = settings.get_plugin_config()
        if not self.plugin_dirs:
            self.plugin_dirs = plugin_config_settings.get('plugin_dirs', ['plugins'])
        
        # 初始化
        self._init_default_plugin_dirs()
        
        if auto_load:
            asyncio.create_task(self._async_discover_and_load())
    
    def _init_default_plugin_dirs(self):
        """初始化默认插件目录"""
        current_dir = Path(__file__).parent
        plugins_dir = current_dir / "plugins"
        if plugins_dir.exists():
            self.plugin_dirs.append(str(plugins_dir))
        
        project_root = current_dir.parent.parent
        project_plugins_dir = project_root / "plugins"
        if project_plugins_dir.exists():
            self.plugin_dirs.append(str(project_plugins_dir))
    
    async def start_preload_worker(self) -> None:
        """启动预加载工作协程"""
        if not self.enable_preload or self._preload_task is not None:
            return
            
        self._preload_task = asyncio.create_task(self._preload_worker())
        logger.info("插件预加载工作协程已启动")
    
    async def stop_preload_worker(self) -> None:
        """停止预加载工作协程"""
        if self._preload_task is not None:
            self._preload_task.cancel()
            try:
                await self._preload_task
            except asyncio.CancelledError:
                pass
            self._preload_task = None
            logger.info("插件预加载工作协程已停止")
    
    async def _preload_worker(self) -> None:
        """预加载工作协程"""
        while True:
            try:
                # 获取预加载任务
                plugin_name = await self._preload_queue.get()
                
                if plugin_name in self._plugins:
                    plugin_info = self._plugins[plugin_name]
                    if plugin_info.status == PluginStatus.UNLOADED:
                        await self._preload_plugin(plugin_info)
                
                self._preload_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"预加载工作协程发生错误: {e}")
    
    async def _preload_plugin(self, plugin_info: OptimizedPluginInfo) -> None:
        """预加载单个插件"""
        try:
            plugin_info.status = PluginStatus.PRELOADING
            
            # 在后台线程中加载模块
            loop = asyncio.get_event_loop()
            module = await loop.run_in_executor(
                self._executor,
                importlib.import_module,
                plugin_info.module_path
            )
            
            # 查找插件类
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    plugin_class = obj
                    break
            
            if plugin_class:
                plugin_info.plugin_class = plugin_class
                plugin_info.status = PluginStatus.PRELOADED
                self._stats['preload_count'] += 1
                logger.debug(f"预加载插件成功: {plugin_info.name}")
            else:
                plugin_info.status = PluginStatus.ERROR
                plugin_info.error_info = "未找到有效的插件类"
                
        except Exception as e:
            plugin_info.status = PluginStatus.ERROR
            plugin_info.error_info = str(e)
            logger.error(f"预加载插件失败 {plugin_info.name}: {e}")
    
    @lru_cache(maxsize=128)
    def _get_cached_plugin_modules(self, plugin_dir: str) -> List[str]:
        """缓存的插件模块发现"""
        discovered_plugins = []
        
        if not os.path.exists(plugin_dir):
            return discovered_plugins
        
        for root, dirs, files in os.walk(plugin_dir):
            for file in files:
                if file.endswith('.py') and not file.startswith('__'):
                    module_path = os.path.join(root, file)
                    relative_path = os.path.relpath(module_path, plugin_dir)
                    module_name = relative_path[:-3].replace(os.sep, '.')
                    discovered_plugins.append(module_name)
        
        return discovered_plugins
    
    def discover_plugins(self) -> List[str]:
        """发现插件（带缓存）"""
        all_plugins = []
        
        for plugin_dir in self.plugin_dirs:
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            plugins = self._get_cached_plugin_modules(plugin_dir)
            all_plugins.extend(plugins)
        
        logger.info(f"发现 {len(all_plugins)} 个插件模块")
        return all_plugins
    
    async def _async_discover_and_load(self) -> None:
        """异步发现和加载插件"""
        try:
            # 启动预加载工作协程
            await self.start_preload_worker()
            
            # 发现插件
            discovered_plugins = self.discover_plugins()
            
            # 创建插件信息
            for module_name in discovered_plugins:
                if module_name not in self._plugins:
                    plugin_info = OptimizedPluginInfo(
                        name=module_name,
                        module_path=module_name
                    )
                    self._plugins[module_name] = plugin_info
                    
                    # 添加到预加载队列
                    if self.enable_preload:
                        await self._preload_queue.put(module_name)
            
            logger.info(f"已发现并准备预加载 {len(discovered_plugins)} 个插件")
            
        except Exception as e:
            logger.error(f"异步发现和加载插件失败: {e}")
    
    async def load_plugin_async(
        self, 
        plugin_name: str, 
        plugin_config: Optional[Dict[str, Any]] = None,
        force_reload: bool = False
    ) -> bool:
        """异步加载插件"""
        start_time = time.time()
        
        try:
            # 检查缓存
            if self.enable_cache and not force_reload:
                cached_instance = self._plugin_cache.get(plugin_name)
                if cached_instance is not None:
                    self._plugin_instances[plugin_name] = cached_instance
                    self._stats['cache_hits'] += 1
                    logger.debug(f"从缓存加载插件: {plugin_name}")
                    return True
                else:
                    self._stats['cache_misses'] += 1
            
            if plugin_name not in self._plugins:
                raise PluginError(f"插件未发现: {plugin_name}")
            
            plugin_info = self._plugins[plugin_name]
            
            # 如果已经加载，直接返回
            if plugin_info.status == PluginStatus.LOADED and not force_reload:
                return True
            
            plugin_info.status = PluginStatus.LOADING
            
            # 如果插件已预加载，直接使用
            if plugin_info.status == PluginStatus.PRELOADED and plugin_info.plugin_class:
                plugin_class = plugin_info.plugin_class
            else:
                # 动态加载
                loop = asyncio.get_event_loop()
                module = await loop.run_in_executor(
                    self._executor,
                    importlib.import_module,
                    plugin_info.module_path
                )
                
                plugin_class = None
                for name, obj in inspect.getmembers(module):
                    if (inspect.isclass(obj) and 
                        issubclass(obj, BasePlugin) and 
                        obj != BasePlugin):
                        plugin_class = obj
                        break
                
                if not plugin_class:
                    raise PluginError(f"未找到有效的插件类: {plugin_name}")
                
                plugin_info.plugin_class = plugin_class
            
            # 更新插件信息
            plugin_info.config.update(plugin_config or {})
            
            # 检查依赖
            await self._check_dependencies_async(plugin_info)
            
            # 创建插件实例
            instance = plugin_class(config=plugin_info.config)
            
            # 初始化插件
            if hasattr(instance, 'initialize'):
                if asyncio.iscoroutinefunction(instance.initialize):
                    await instance.initialize()
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self._executor,
                        instance.initialize
                    )
            
            # 注册插件
            plugin_info.instance = instance
            plugin_info.status = PluginStatus.LOADED
            plugin_info.load_time = time.time()
            
            self._plugin_instances[plugin_name] = instance
            
            # 缓存插件实例
            if self.enable_cache:
                self._plugin_cache.set(plugin_name, instance)
            
            # 注册钩子
            self._register_plugin_hooks(instance)
            
            # 更新统计信息
            load_time = time.time() - start_time
            self._stats['total_loads'] += 1
            if self._stats['avg_load_time'] == 0.0:
                self._stats['avg_load_time'] = load_time
            else:
                alpha = 0.1
                self._stats['avg_load_time'] = (alpha * load_time + 
                                              (1 - alpha) * self._stats['avg_load_time'])
            
            logger.info(f"成功加载插件: {plugin_name}，耗时: {load_time:.3f}s")
            return True
            
        except Exception as e:
            error_msg = f"加载插件失败 {plugin_name}: {str(e)}"
            logger.error(error_msg)
            
            if plugin_name in self._plugins:
                self._plugins[plugin_name].status = PluginStatus.ERROR
                self._plugins[plugin_name].error_info = error_msg
            
            raise PluginError(error_msg) from e
    
    async def _check_dependencies_async(self, plugin_info: OptimizedPluginInfo):
        """异步检查插件依赖"""
        for dependency in plugin_info.dependencies:
            if dependency not in self._plugins:
                # 尝试自动加载依赖
                await self.load_plugin_async(dependency)
            
            if dependency not in self._plugins:
                raise PluginError(
                    f"插件 {plugin_info.name} 依赖 {dependency}，但无法加载"
                )
            
            dep_plugin = self._plugins[dependency]
            if dep_plugin.status != PluginStatus.LOADED:
                raise PluginError(
                    f"插件 {plugin_info.name} 依赖 {dependency}，"
                    f"但其状态为: {dep_plugin.status.value}"
                )
    
    def _register_plugin_hooks(self, instance: BasePlugin):
        """注册插件钩子"""
        for method_name in dir(instance):
            if method_name.startswith('on_'):
                method = getattr(instance, method_name)
                if callable(method):
                    hook_name = method_name[3:]
                    if hook_name not in self._plugin_hooks:
                        self._plugin_hooks[hook_name] = []
                    self._plugin_hooks[hook_name].append(method)
    
    async def call_plugin_with_stats(
        self,
        plugin_name: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """带统计信息的插件调用"""
        start_time = time.time()
        
        try:
            # 确保插件已加载
            if plugin_name not in self._plugin_instances:
                await self.load_plugin_async(plugin_name)
            
            plugin = self._plugin_instances[plugin_name]
            plugin_info = self._plugins[plugin_name]
            
            if not hasattr(plugin, method_name):
                raise PluginError(
                    f"方法 {method_name} 在插件 {plugin_name} 中不存在"
                )
            
            method = getattr(plugin, method_name)
            
            # 执行方法
            if asyncio.iscoroutinefunction(method):
                result = await method(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: method(*args, **kwargs)
                )
            
            # 更新统计信息
            execution_time = time.time() - start_time
            plugin_info.update_usage_stats(execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"调用插件方法失败 {plugin_name}.{method_name}: {e}")
            raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        plugin_stats = {}
        for name, info in self._plugins.items():
            plugin_stats[name] = {
                'status': info.status.value,
                'usage_count': info.usage_count,
                'avg_execution_time': info.avg_execution_time,
                'last_used': info.last_used
            }
        
        return {
            'global_stats': self._stats,
            'plugin_stats': plugin_stats,
            'cache_stats': {
                'enabled': self.enable_cache,
                'size': len(self._plugin_cache._cache) if self._plugin_cache else 0
            },
            'preload_stats': {
                'enabled': self.enable_preload,
                'queue_size': self._preload_queue.qsize() if self._preload_queue else 0
            }
        }
    
    async def cleanup(self):
        """清理资源"""
        # 停止预加载工作协程
        await self.stop_preload_worker()
        
        # 卸载所有插件
        plugin_names = list(self._plugins.keys())
        for plugin_name in plugin_names:
            await self._unload_plugin_async(plugin_name)
        
        # 清理缓存
        if self._plugin_cache:
            self._plugin_cache.clear()
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        # 清理弱引用
        self._weak_refs.clear()
        
        logger.info("优化插件管理器已清理")
    
    async def _unload_plugin_async(self, plugin_name: str) -> bool:
        """异步卸载插件"""
        if plugin_name not in self._plugins:
            return False
        
        try:
            plugin_info = self._plugins[plugin_name]
            instance = plugin_info.instance
            
            if instance and hasattr(instance, 'cleanup'):
                if asyncio.iscoroutinefunction(instance.cleanup):
                    await instance.cleanup()
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self._executor,
                        instance.cleanup
                    )
            
            # 移除注册
            del self._plugins[plugin_name]
            if plugin_name in self._plugin_instances:
                del self._plugin_instances[plugin_name]
            
            # 清理缓存
            if self._plugin_cache:
                self._plugin_cache._cache.pop(plugin_name, None)
                self._plugin_cache._timestamps.pop(plugin_name, None)
            
            logger.info(f"成功卸载插件: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"卸载插件失败 {plugin_name}: {e}")
            return False


# 全局优化插件管理器实例
_optimized_plugin_manager: Optional[OptimizedPluginManager] = None


def get_optimized_plugin_manager() -> OptimizedPluginManager:
    """获取全局优化插件管理器实例"""
    global _optimized_plugin_manager
    if _optimized_plugin_manager is None:
        _optimized_plugin_manager = OptimizedPluginManager()
    return _optimized_plugin_manager


async def start_optimized_plugin_manager() -> None:
    """启动全局优化插件管理器"""
    manager = get_optimized_plugin_manager()
    await manager.start_preload_worker()


async def stop_optimized_plugin_manager() -> None:
    """停止全局优化插件管理器"""
    global _optimized_plugin_manager
    if _optimized_plugin_manager is not None:
        await _optimized_plugin_manager.cleanup()
        _optimized_plugin_manager = None