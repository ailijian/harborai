#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 插件管理器

负责插件的注册、发现、加载和管理。
"""

import os
import importlib
import inspect
from typing import Dict, List, Optional, Type, Any
from pathlib import Path
import logging

from .base import Plugin, PluginInfo
from ..exceptions import PluginError, PluginLoadError, PluginNotFoundError, PluginConfigError

logger = logging.getLogger(__name__)


class PluginRegistry:
    """插件注册表
    
    管理所有已注册的插件。
    """
    
    def __init__(self):
        self._plugins: Dict[str, Type[Plugin]] = {}
        self._instances: Dict[str, Plugin] = {}
        self._model_mapping: Dict[str, str] = {}  # model -> plugin_name
    
    def register(self, plugin_class: Type[Plugin], plugin_name: Optional[str] = None) -> bool:
        """注册插件类
        
        Args:
            plugin_class: 插件类
            plugin_name: 插件名称，如果不提供则使用类名
            
        Returns:
            bool: 注册是否成功
        """
        try:
            if not issubclass(plugin_class, Plugin):
                raise PluginError(f"Plugin class {plugin_class.__name__} must inherit from Plugin")
            
            name = plugin_name or plugin_class.__name__
            
            # 检查是否已注册
            if name in self._plugins:
                logger.warning(f"Plugin {name} is already registered, overwriting")
            
            self._plugins[name] = plugin_class
            
            # 创建临时实例以获取插件信息
            temp_instance = plugin_class()
            info = temp_instance.info
            
            # 更新模型映射
            for model in info.supported_models:
                if model in self._model_mapping:
                    logger.warning(f"Model {model} is already mapped to plugin {self._model_mapping[model]}, overwriting with {name}")
                self._model_mapping[model] = name
            
            logger.info(f"Plugin {name} registered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
            raise PluginError(f"Failed to register plugin: {e}")
    
    def unregister(self, plugin_name: str) -> bool:
        """注销插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 注销是否成功
        """
        try:
            if plugin_name not in self._plugins:
                logger.warning(f"Plugin {plugin_name} is not registered")
                return False
            
            # 清理实例
            if plugin_name in self._instances:
                instance = self._instances[plugin_name]
                instance.cleanup()
                del self._instances[plugin_name]
            
            # 清理模型映射
            models_to_remove = [model for model, name in self._model_mapping.items() if name == plugin_name]
            for model in models_to_remove:
                del self._model_mapping[model]
            
            # 移除插件类
            del self._plugins[plugin_name]
            
            logger.info(f"Plugin {plugin_name} unregistered successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister plugin {plugin_name}: {e}")
            return False
    
    def get_plugin_class(self, plugin_name: str) -> Optional[Type[Plugin]]:
        """获取插件类
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件类或None
        """
        return self._plugins.get(plugin_name)
    
    def get_plugin_for_model(self, model: str) -> Optional[str]:
        """根据模型名称获取对应的插件名称
        
        Args:
            model: 模型名称
            
        Returns:
            插件名称或None
        """
        return self._model_mapping.get(model)
    
    def list_plugins(self) -> List[str]:
        """列出所有已注册的插件
        
        Returns:
            插件名称列表
        """
        return list(self._plugins.keys())
    
    def list_models(self) -> List[str]:
        """列出所有支持的模型
        
        Returns:
            模型名称列表
        """
        return list(self._model_mapping.keys())
    
    def get_plugin_info(self, plugin_name: str) -> Optional[PluginInfo]:
        """获取插件信息
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件信息或None
        """
        plugin_class = self.get_plugin_class(plugin_name)
        if plugin_class:
            try:
                temp_instance = plugin_class()
                return temp_instance.info
            except Exception as e:
                logger.error(f"Failed to get info for plugin {plugin_name}: {e}")
        return None


class PluginManager:
    """插件管理器
    
    负责插件的发现、加载、初始化和生命周期管理。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.registry = PluginRegistry()
        self._instances: Dict[str, Plugin] = {}
    
    def discover_plugins(self, plugin_dir: Optional[str] = None) -> List[str]:
        """发现插件
        
        Args:
            plugin_dir: 插件目录路径，如果不提供则使用默认目录
            
        Returns:
            发现的插件名称列表
        """
        if plugin_dir is None:
            # 使用当前模块所在目录
            current_dir = Path(__file__).parent
            plugin_dir = str(current_dir)
        
        discovered = []
        
        try:
            plugin_path = Path(plugin_dir)
            if not plugin_path.exists():
                logger.warning(f"Plugin directory {plugin_dir} does not exist")
                return discovered
            
            # 扫描Python文件
            for file_path in plugin_path.glob("*_plugin.py"):
                if file_path.is_file() and not file_path.name.startswith('__'):
                    module_name = file_path.stem
                    try:
                        # 动态导入模块
                        spec = importlib.util.spec_from_file_location(module_name, file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            
                            # 查找插件类
                            for name, obj in inspect.getmembers(module, inspect.isclass):
                                if (issubclass(obj, Plugin) and 
                                    obj != Plugin and 
                                    not inspect.isabstract(obj)):
                                    
                                    plugin_name = name
                                    self.registry.register(obj, plugin_name)
                                    discovered.append(plugin_name)
                                    logger.info(f"Discovered plugin: {plugin_name}")
                                    break
                    
                    except Exception as e:
                        logger.error(f"Failed to load plugin from {file_path}: {e}")
            
            logger.info(f"Discovered {len(discovered)} plugins")
            return discovered
            
        except Exception as e:
            logger.error(f"Failed to discover plugins: {e}")
            raise PluginError(f"Plugin discovery failed: {e}")
    
    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> Plugin:
        """加载并初始化插件
        
        Args:
            plugin_name: 插件名称
            config: 插件配置
            
        Returns:
            插件实例
        """
        try:
            # 检查是否已加载
            if plugin_name in self._instances:
                return self._instances[plugin_name]
            
            # 获取插件类
            plugin_class = self.registry.get_plugin_class(plugin_name)
            if not plugin_class:
                raise PluginNotFoundError(f"Plugin {plugin_name} not found")
            
            # 合并配置
            plugin_config = {}
            if self.config.get('plugins', {}).get(plugin_name):
                plugin_config.update(self.config['plugins'][plugin_name])
            if config:
                plugin_config.update(config)
            
            # 创建实例
            instance = plugin_class(plugin_config)
            
            # 验证配置
            if not instance.validate_config():
                raise PluginConfigError(f"Invalid configuration for plugin {plugin_name}")
            
            # 初始化
            if not instance.initialize():
                raise PluginLoadError(f"Failed to initialize plugin {plugin_name}")
            
            instance._set_initialized(True)
            self._instances[plugin_name] = instance
            
            logger.info(f"Plugin {plugin_name} loaded successfully")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_name}: {e}")
            if isinstance(e, (PluginNotFoundError, PluginConfigError, PluginLoadError)):
                raise
            else:
                raise PluginLoadError(f"Failed to load plugin {plugin_name}: {e}")
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """获取已加载的插件实例
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件实例或None
        """
        return self._instances.get(plugin_name)
    
    def get_plugin_for_model(self, model: str) -> Optional[Plugin]:
        """根据模型名称获取对应的插件实例
        
        Args:
            model: 模型名称
            
        Returns:
            插件实例或None
        """
        plugin_name = self.registry.get_plugin_for_model(model)
        if plugin_name:
            # 如果插件未加载，尝试加载
            if plugin_name not in self._instances:
                try:
                    return self.load_plugin(plugin_name)
                except Exception as e:
                    logger.error(f"Failed to load plugin {plugin_name} for model {model}: {e}")
                    return None
            return self._instances.get(plugin_name)
        return None
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 卸载是否成功
        """
        try:
            if plugin_name in self._instances:
                instance = self._instances[plugin_name]
                instance.cleanup()
                del self._instances[plugin_name]
                logger.info(f"Plugin {plugin_name} unloaded successfully")
                return True
            else:
                logger.warning(f"Plugin {plugin_name} is not loaded")
                return False
        except Exception as e:
            logger.error(f"Failed to unload plugin {plugin_name}: {e}")
            return False
    
    def reload_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """重新加载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            重新加载的插件实例或None
        """
        try:
            # 先卸载
            self.unload_plugin(plugin_name)
            
            # 重新加载
            return self.load_plugin(plugin_name)
            
        except Exception as e:
            logger.error(f"Failed to reload plugin {plugin_name}: {e}")
            return None
    
    def list_loaded_plugins(self) -> List[str]:
        """列出已加载的插件
        
        Returns:
            已加载的插件名称列表
        """
        return list(self._instances.keys())
    
    def is_loaded(self, plugin_name: str) -> bool:
        """检查插件是否已加载
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 插件是否已加载
        """
        return plugin_name in self._instances
    
    def resolve_dependencies(self, plugin_name: str) -> List[str]:
        """解析插件依赖
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            依赖的插件名称列表
        """
        # 获取插件类
        plugin_class = self.registry.get_plugin_class(plugin_name)
        if not plugin_class:
            return []
        
        # 检查插件是否有依赖属性
        if hasattr(plugin_class, 'dependencies'):
            return plugin_class.dependencies
        
        # 默认无依赖
        return []
    
    def initialize_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """初始化插件
        
        Args:
            plugin_name: 插件名称
            config: 插件配置
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            plugin = self.load_plugin(plugin_name, config)
            return plugin is not None
        except Exception as e:
            logger.error(f"Failed to initialize plugin {plugin_name}: {e}")
            return False
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """启用插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 启用是否成功
        """
        try:
            if plugin_name not in self._instances:
                # 如果插件未加载，先加载
                plugin = self.load_plugin(plugin_name)
                if not plugin:
                    return False
            
            plugin = self._instances[plugin_name]
            if hasattr(plugin, 'enable'):
                return plugin.enable()
            
            # 如果插件没有enable方法，认为默认已启用
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable plugin {plugin_name}: {e}")
            return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """禁用插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            bool: 禁用是否成功
        """
        try:
            if plugin_name not in self._instances:
                logger.warning(f"Plugin {plugin_name} is not loaded")
                return False
            
            plugin = self._instances[plugin_name]
            if hasattr(plugin, 'disable'):
                return plugin.disable()
            
            # 如果插件没有disable方法，认为默认已禁用
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable plugin {plugin_name}: {e}")
            return False
    
    def shutdown(self):
        """关闭插件管理器，清理所有插件"""
        try:
            for plugin_name in list(self._instances.keys()):
                self.unload_plugin(plugin_name)
            logger.info("Plugin manager shutdown completed")
        except Exception as e:
            logger.error(f"Error during plugin manager shutdown: {e}")