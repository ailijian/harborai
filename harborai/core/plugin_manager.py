#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 插件管理器模块

负责插件的加载、管理、生命周期控制和调用分发。
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from .base_plugin import BasePlugin
from .exceptions import (
    PluginError,
    ValidationError,
    ConfigurationError,
    HarborAIError
)
from .retry import retry, RetryConfig, RetryStrategy

logger = logging.getLogger(__name__)


class PluginStatus(Enum):
    """插件状态枚举"""
    UNLOADED = "unloaded"  # 未加载
    LOADING = "loading"    # 加载中
    LOADED = "loaded"      # 已加载
    ACTIVE = "active"      # 活跃状态
    INACTIVE = "inactive"  # 非活跃状态
    ERROR = "error"        # 错误状态
    DISABLED = "disabled"  # 已禁用


@dataclass
class PluginInfo:
    """插件信息类
    
    Args:
        name: 插件名称
        version: 插件版本
        description: 插件描述
        author: 插件作者
        plugin_class: 插件类
        module_path: 模块路径
        config: 插件配置
        dependencies: 插件依赖
        status: 插件状态
        instance: 插件实例
        load_time: 加载时间
        error_info: 错误信息
    """
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
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "module_path": self.module_path,
            "config": self.config,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "load_time": self.load_time,
            "error_info": self.error_info
        }


class PluginManager:
    """插件管理器
    
    负责插件的发现、加载、管理和调用。
    """
    
    def __init__(
        self,
        plugin_dirs: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        auto_load: bool = True,
        max_workers: int = 4
    ):
        """初始化插件管理器
        
        Args:
            plugin_dirs: 插件目录列表
            config: 全局配置
            auto_load: 是否自动加载插件
            max_workers: 最大工作线程数
        """
        self.plugin_dirs = plugin_dirs or []
        self.config = config or {}
        self.auto_load = auto_load
        self.max_workers = max_workers
        
        # 插件注册表
        self._plugins: Dict[str, PluginInfo] = {}
        self._plugin_instances: Dict[str, BasePlugin] = {}
        self._plugin_hooks: Dict[str, List[Callable]] = {}
        
        # 线程池
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 初始化默认插件目录
        self._init_default_plugin_dirs()
        
        # 自动加载插件
        if auto_load:
            self.discover_and_load_plugins()
    
    def _init_default_plugin_dirs(self):
        """初始化默认插件目录"""
        # 添加当前包的plugins目录
        current_dir = Path(__file__).parent
        plugins_dir = current_dir / "plugins"
        if plugins_dir.exists():
            self.plugin_dirs.append(str(plugins_dir))
        
        # 添加项目根目录的plugins目录
        project_root = current_dir.parent.parent
        project_plugins_dir = project_root / "plugins"
        if project_plugins_dir.exists():
            self.plugin_dirs.append(str(project_plugins_dir))
    
    def discover_plugins(self) -> List[str]:
        """发现插件
        
        Returns:
            发现的插件模块路径列表
        """
        discovered_plugins = []
        
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                logger.warning(f"Plugin directory not found: {plugin_dir}")
                continue
            
            # 添加到Python路径
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)
            
            # 扫描Python文件
            for root, dirs, files in os.walk(plugin_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        module_path = os.path.join(root, file)
                        relative_path = os.path.relpath(module_path, plugin_dir)
                        module_name = relative_path[:-3].replace(os.sep, '.')
                        discovered_plugins.append(module_name)
        
        logger.info(f"Discovered {len(discovered_plugins)} plugin modules")
        return discovered_plugins
    
    @retry(
        max_attempts=3,
        strategy=RetryStrategy.FIXED,
        base_delay=0.5,
        retryable_exceptions=(ImportError, AttributeError)
    )
    def load_plugin(self, module_name: str, plugin_config: Optional[Dict[str, Any]] = None) -> bool:
        """加载单个插件
        
        Args:
            module_name: 模块名称
            plugin_config: 插件配置
        
        Returns:
            是否加载成功
        """
        try:
            logger.info(f"Loading plugin: {module_name}")
            
            # 导入模块
            module = importlib.import_module(module_name)
            
            # 查找插件类
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    plugin_class = obj
                    break
            
            if not plugin_class:
                raise PluginError(f"No valid plugin class found in module: {module_name}")
            
            # 获取插件信息
            plugin_info = self._extract_plugin_info(plugin_class, module_name)
            plugin_info.plugin_class = plugin_class
            plugin_info.config.update(plugin_config or {})
            plugin_info.status = PluginStatus.LOADING
            
            # 检查依赖
            self._check_dependencies(plugin_info)
            
            # 创建插件实例
            instance = plugin_class(config=plugin_info.config)
            
            # 初始化插件
            if hasattr(instance, 'initialize'):
                if asyncio.iscoroutinefunction(instance.initialize):
                    # 异步初始化
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(instance.initialize())
                else:
                    # 同步初始化
                    instance.initialize()
            
            # 注册插件
            plugin_info.instance = instance
            plugin_info.status = PluginStatus.LOADED
            plugin_info.load_time = time.time()
            
            self._plugins[plugin_info.name] = plugin_info
            self._plugin_instances[plugin_info.name] = instance
            
            # 注册钩子
            self._register_plugin_hooks(instance)
            
            logger.info(f"Successfully loaded plugin: {plugin_info.name}")
            return True
            
        except Exception as e:
            error_msg = f"Failed to load plugin {module_name}: {str(e)}"
            logger.error(error_msg)
            
            # 记录错误信息
            if module_name in self._plugins:
                self._plugins[module_name].status = PluginStatus.ERROR
                self._plugins[module_name].error_info = error_msg
            
            raise PluginError(error_msg) from e
    
    def _extract_plugin_info(self, plugin_class: Type[BasePlugin], module_name: str) -> PluginInfo:
        """提取插件信息
        
        Args:
            plugin_class: 插件类
            module_name: 模块名称
        
        Returns:
            插件信息
        """
        # 从类属性获取信息
        name = getattr(plugin_class, 'name', module_name)
        version = getattr(plugin_class, 'version', '1.0.0')
        description = getattr(plugin_class, 'description', '')
        author = getattr(plugin_class, 'author', '')
        dependencies = getattr(plugin_class, 'dependencies', [])
        
        return PluginInfo(
            name=name,
            version=version,
            description=description,
            author=author,
            module_path=module_name,
            dependencies=dependencies
        )
    
    def _check_dependencies(self, plugin_info: PluginInfo):
        """检查插件依赖
        
        Args:
            plugin_info: 插件信息
        
        Raises:
            PluginError: 依赖检查失败
        """
        for dependency in plugin_info.dependencies:
            if dependency not in self._plugins:
                raise PluginError(
                    f"Plugin {plugin_info.name} depends on {dependency}, "
                    f"but it is not loaded"
                )
            
            dep_plugin = self._plugins[dependency]
            if dep_plugin.status != PluginStatus.LOADED:
                raise PluginError(
                    f"Plugin {plugin_info.name} depends on {dependency}, "
                    f"but it is in status: {dep_plugin.status.value}"
                )
    
    def _register_plugin_hooks(self, instance: BasePlugin):
        """注册插件钩子
        
        Args:
            instance: 插件实例
        """
        # 查找钩子方法
        for method_name in dir(instance):
            if method_name.startswith('on_'):
                method = getattr(instance, method_name)
                if callable(method):
                    hook_name = method_name[3:]  # 移除'on_'前缀
                    if hook_name not in self._plugin_hooks:
                        self._plugin_hooks[hook_name] = []
                    self._plugin_hooks[hook_name].append(method)
    
    def discover_and_load_plugins(self) -> Dict[str, bool]:
        """发现并加载所有插件
        
        Returns:
            插件加载结果字典
        """
        discovered_plugins = self.discover_plugins()
        load_results = {}
        
        for module_name in discovered_plugins:
            try:
                success = self.load_plugin(module_name)
                load_results[module_name] = success
            except Exception as e:
                logger.error(f"Failed to load plugin {module_name}: {e}")
                load_results[module_name] = False
        
        return load_results
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件
        
        Args:
            plugin_name: 插件名称
        
        Returns:
            是否卸载成功
        """
        if plugin_name not in self._plugins:
            logger.warning(f"Plugin not found: {plugin_name}")
            return False
        
        try:
            plugin_info = self._plugins[plugin_name]
            instance = plugin_info.instance
            
            # 调用清理方法
            if instance and hasattr(instance, 'cleanup'):
                if asyncio.iscoroutinefunction(instance.cleanup):
                    loop = asyncio.get_event_loop()
                    loop.run_until_complete(instance.cleanup())
                else:
                    instance.cleanup()
            
            # 移除钩子
            self._unregister_plugin_hooks(instance)
            
            # 移除注册
            del self._plugins[plugin_name]
            if plugin_name in self._plugin_instances:
                del self._plugin_instances[plugin_name]
            
            logger.info(f"Successfully unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def _unregister_plugin_hooks(self, instance: BasePlugin):
        """注销插件钩子
        
        Args:
            instance: 插件实例
        """
        for hook_name, hooks in self._plugin_hooks.items():
            # 移除该插件的所有钩子
            self._plugin_hooks[hook_name] = [
                hook for hook in hooks 
                if not (hasattr(hook, '__self__') and hook.__self__ == instance)
            ]
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """获取插件实例
        
        Args:
            plugin_name: 插件名称
        
        Returns:
            插件实例
        """
        return self._plugin_instances.get(plugin_name)
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """获取插件信息
        
        Args:
            plugin_name: 插件名称
        
        Returns:
            插件信息
        """
        return self._plugins.get(plugin_name)
    
    def list_plugins(self, status_filter: Optional[PluginStatus] = None) -> List[PluginInfo]:
        """列出插件
        
        Args:
            status_filter: 状态过滤器
        
        Returns:
            插件信息列表
        """
        plugins = list(self._plugins.values())
        
        if status_filter:
            plugins = [p for p in plugins if p.status == status_filter]
        
        return plugins
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """启用插件
        
        Args:
            plugin_name: 插件名称
        
        Returns:
            是否启用成功
        """
        if plugin_name not in self._plugins:
            return False
        
        plugin_info = self._plugins[plugin_name]
        if plugin_info.status == PluginStatus.LOADED:
            plugin_info.status = PluginStatus.ACTIVE
            logger.info(f"Enabled plugin: {plugin_name}")
            return True
        
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """禁用插件
        
        Args:
            plugin_name: 插件名称
        
        Returns:
            是否禁用成功
        """
        if plugin_name not in self._plugins:
            return False
        
        plugin_info = self._plugins[plugin_name]
        if plugin_info.status == PluginStatus.ACTIVE:
            plugin_info.status = PluginStatus.INACTIVE
            logger.info(f"Disabled plugin: {plugin_name}")
            return True
        
        return False
    
    async def call_plugin_async(
        self,
        plugin_name: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """异步调用插件方法
        
        Args:
            plugin_name: 插件名称
            method_name: 方法名称
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            方法执行结果
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise PluginError(f"Plugin not found: {plugin_name}")
        
        if not hasattr(plugin, method_name):
            raise PluginError(
                f"Method {method_name} not found in plugin {plugin_name}"
            )
        
        method = getattr(plugin, method_name)
        
        if asyncio.iscoroutinefunction(method):
            return await method(*args, **kwargs)
        else:
            # 在线程池中执行同步方法
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor, 
                lambda: method(*args, **kwargs)
            )
    
    def call_plugin(
        self,
        plugin_name: str,
        method_name: str,
        *args,
        **kwargs
    ) -> Any:
        """同步调用插件方法
        
        Args:
            plugin_name: 插件名称
            method_name: 方法名称
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            方法执行结果
        """
        plugin = self.get_plugin(plugin_name)
        if not plugin:
            raise PluginError(f"Plugin not found: {plugin_name}")
        
        if not hasattr(plugin, method_name):
            raise PluginError(
                f"Method {method_name} not found in plugin {plugin_name}"
            )
        
        method = getattr(plugin, method_name)
        return method(*args, **kwargs)
    
    async def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """触发钩子
        
        Args:
            hook_name: 钩子名称
            *args: 位置参数
            **kwargs: 关键字参数
        
        Returns:
            所有钩子的执行结果列表
        """
        if hook_name not in self._plugin_hooks:
            return []
        
        results = []
        hooks = self._plugin_hooks[hook_name]
        
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    result = await hook(*args, **kwargs)
                else:
                    result = hook(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hook {hook_name}: {e}")
                results.append(None)
        
        return results
    
    def get_plugin_stats(self) -> Dict[str, Any]:
        """获取插件统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            "total_plugins": len(self._plugins),
            "loaded_plugins": len([p for p in self._plugins.values() if p.status == PluginStatus.LOADED]),
            "active_plugins": len([p for p in self._plugins.values() if p.status == PluginStatus.ACTIVE]),
            "error_plugins": len([p for p in self._plugins.values() if p.status == PluginStatus.ERROR]),
            "plugin_dirs": self.plugin_dirs,
            "hooks_count": {name: len(hooks) for name, hooks in self._plugin_hooks.items()}
        }
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        # 卸载所有插件
        plugin_names = list(self._plugins.keys())
        for plugin_name in plugin_names:
            self.unload_plugin(plugin_name)
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        logger.info("Plugin manager cleaned up")


# 全局插件管理器实例
_global_plugin_manager: Optional[PluginManager] = None


def get_plugin_manager() -> PluginManager:
    """获取全局插件管理器实例
    
    Returns:
        插件管理器实例
    """
    global _global_plugin_manager
    if _global_plugin_manager is None:
        _global_plugin_manager = PluginManager()
    return _global_plugin_manager


def set_plugin_manager(manager: PluginManager):
    """设置全局插件管理器实例
    
    Args:
        manager: 插件管理器实例
    """
    global _global_plugin_manager
    _global_plugin_manager = manager