#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
延迟插件管理器

实现插件的延迟加载机制，显著提升初始化性能。
根据技术设计方案，支持DeepSeek、豆包、文心一言等模型厂商的延迟加载。

设计原则：
1. 插件在首次使用时才被加载和初始化
2. 加载后的插件被缓存以提高后续访问性能
3. 保持与现有插件系统的兼容性
4. 支持异步加载和错误处理
"""

import importlib
import threading
import time
from typing import Dict, Any, Optional, List, Type
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .plugins.base import Plugin, BaseLLMPlugin
from .exceptions import PluginError, PluginLoadError, PluginNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class LazyPluginInfo:
    """延迟插件信息
    
    包含插件的基本信息，用于延迟加载决策。
    """
    name: str
    module_path: str
    class_name: str
    supported_models: List[str]
    priority: int = 0  # 加载优先级，数字越小优先级越高
    
    
class LazyPluginManager:
    """延迟插件管理器
    
    实现插件的延迟加载机制，在首次使用时才加载插件。
    
    主要特性：
    1. 快速初始化：初始化时不加载任何插件
    2. 按需加载：插件在首次使用时才被加载
    3. 缓存机制：加载后的插件被缓存以提高性能
    4. 线程安全：支持多线程环境下的安全访问
    5. 错误处理：优雅处理插件加载失败的情况
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化延迟插件管理器
        
        Args:
            config: 配置字典，包含插件配置信息
        """
        self.config = config or {}
        
        # 插件注册表：存储插件的基本信息，不实际加载插件
        self._plugin_registry: Dict[str, LazyPluginInfo] = {}
        
        # 已加载的插件实例缓存
        self._loaded_plugins: Dict[str, Plugin] = {}
        
        # 模型到插件的映射
        self._model_to_plugin: Dict[str, str] = {}
        
        # 线程锁，确保线程安全
        self._lock = threading.RLock()
        
        # 线程池，用于异步加载
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="lazy_plugin")
        
        # 初始化插件注册表
        self._initialize_plugin_registry()
        
        logger.info("LazyPluginManager初始化完成，注册了%d个插件", len(self._plugin_registry))
    
    def _initialize_plugin_registry(self):
        """初始化插件注册表
        
        注册所有支持的插件信息，但不实际加载插件。
        这个过程应该非常快速，只是建立映射关系。
        """
        # 注册DeepSeek插件
        deepseek_info = LazyPluginInfo(
            name="deepseek",
            module_path="harborai.core.plugins.deepseek_plugin",
            class_name="DeepSeekPlugin",
            supported_models=[
                "deepseek-chat",
                "deepseek-reasoner",
                "deepseek-coder"
            ],
            priority=1
        )
        self._register_plugin_info(deepseek_info)
        
        # 注册豆包插件
        doubao_info = LazyPluginInfo(
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
        )
        self._register_plugin_info(doubao_info)
        
        # 注册文心一言插件
        wenxin_info = LazyPluginInfo(
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
        )
        self._register_plugin_info(wenxin_info)
        
        # 注册OpenAI插件（用于兼容性测试）
        openai_info = LazyPluginInfo(
            name="openai",
            module_path="harborai.core.plugins.openai_plugin",
            class_name="OpenAIPlugin",
            supported_models=[
                "gpt-3.5-turbo",
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o"
            ],
            priority=0  # 最高优先级
        )
        self._register_plugin_info(openai_info)
    
    def _register_plugin_info(self, plugin_info: LazyPluginInfo):
        """注册插件信息
        
        Args:
            plugin_info: 插件信息
        """
        self._plugin_registry[plugin_info.name] = plugin_info
        
        # 建立模型到插件的映射
        for model in plugin_info.supported_models:
            self._model_to_plugin[model] = plugin_info.name
        
        logger.debug("注册插件信息: %s，支持模型: %s", 
                    plugin_info.name, plugin_info.supported_models)
    
    def get_plugin_name_for_model(self, model: str) -> Optional[str]:
        """根据模型名称获取对应的插件名称
        
        Args:
            model: 模型名称
            
        Returns:
            插件名称，如果不支持该模型则返回None
        """
        return self._model_to_plugin.get(model)
    
    def get_plugin_for_model(self, model: str) -> Optional[Plugin]:
        """根据模型名称获取对应的插件实例
        
        这是延迟加载的核心方法。插件在首次调用时才被加载。
        
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
        
        如果插件未加载，则进行延迟加载。
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件实例，如果加载失败则返回None
        """
        # 检查缓存
        if plugin_name in self._loaded_plugins:
            return self._loaded_plugins[plugin_name]
        
        # 延迟加载插件
        return self._load_plugin(plugin_name)
    
    def _load_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """延迟加载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件实例，如果加载失败则返回None
        """
        with self._lock:
            # 双重检查锁定模式
            if plugin_name in self._loaded_plugins:
                return self._loaded_plugins[plugin_name]
            
            # 获取插件信息
            plugin_info = self._plugin_registry.get(plugin_name)
            if not plugin_info:
                logger.error("未找到插件信息: %s", plugin_name)
                return None
            
            try:
                start_time = time.perf_counter()
                
                # 动态导入插件模块
                logger.debug("开始加载插件: %s", plugin_name)
                module = importlib.import_module(plugin_info.module_path)
                
                # 获取插件类
                plugin_class = getattr(module, plugin_info.class_name)
                if not issubclass(plugin_class, Plugin):
                    raise PluginError(f"插件类 {plugin_info.class_name} 必须继承自Plugin")
                
                # 获取插件配置
                plugin_config = self._get_plugin_config(plugin_name)
                
                # 创建插件实例
                plugin_instance = plugin_class(name=plugin_name, **plugin_config)
                
                # 验证配置
                if hasattr(plugin_instance, 'validate_config') and not plugin_instance.validate_config():
                    raise PluginError(f"插件 {plugin_name} 配置验证失败")
                
                # 初始化插件
                if hasattr(plugin_instance, 'initialize') and not plugin_instance.initialize():
                    raise PluginError(f"插件 {plugin_name} 初始化失败")
                
                # 缓存插件实例
                self._loaded_plugins[plugin_name] = plugin_instance
                
                load_time = (time.perf_counter() - start_time) * 1000
                logger.info("插件 %s 加载成功，耗时: %.2fms", plugin_name, load_time)
                
                return plugin_instance
                
            except Exception as e:
                logger.error("插件 %s 加载失败: %s", plugin_name, str(e))
                return None
    
    def _get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """获取插件配置
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件配置字典
        """
        # 从全局配置中获取插件特定配置
        plugin_config = {}
        
        if 'plugins' in self.config and plugin_name in self.config['plugins']:
            plugin_config.update(self.config['plugins'][plugin_name])
        
        # 添加通用配置
        if 'timeout' in self.config:
            plugin_config.setdefault('timeout', self.config['timeout'])
        
        if 'max_retries' in self.config:
            plugin_config.setdefault('max_retries', self.config['max_retries'])
        
        return plugin_config
    
    def preload_plugin(self, plugin_name: str) -> bool:
        """预加载插件
        
        在某些情况下，可能需要提前加载插件以减少首次使用的延迟。
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            是否加载成功
        """
        plugin = self.get_plugin(plugin_name)
        return plugin is not None
    
    def preload_plugins_for_models(self, models: List[str]) -> Dict[str, bool]:
        """为指定模型预加载插件
        
        Args:
            models: 模型名称列表
            
        Returns:
            每个模型对应插件的加载结果
        """
        results = {}
        for model in models:
            plugin_name = self.get_plugin_name_for_model(model)
            if plugin_name:
                results[model] = self.preload_plugin(plugin_name)
            else:
                results[model] = False
        return results
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            是否卸载成功
        """
        with self._lock:
            if plugin_name in self._loaded_plugins:
                try:
                    plugin = self._loaded_plugins[plugin_name]
                    if hasattr(plugin, 'cleanup'):
                        plugin.cleanup()
                    del self._loaded_plugins[plugin_name]
                    logger.info("插件 %s 卸载成功", plugin_name)
                    return True
                except Exception as e:
                    logger.error("插件 %s 卸载失败: %s", plugin_name, str(e))
                    return False
            return True
    
    def get_loaded_plugins(self) -> List[str]:
        """获取已加载的插件列表
        
        Returns:
            已加载的插件名称列表
        """
        return list(self._loaded_plugins.keys())
    
    def get_supported_models(self) -> List[str]:
        """获取所有支持的模型列表
        
        Returns:
            支持的模型名称列表
        """
        return list(self._model_to_plugin.keys())
    
    def get_plugin_info(self, plugin_name: str) -> Optional[LazyPluginInfo]:
        """获取插件信息
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            插件信息，如果不存在则返回None
        """
        return self._plugin_registry.get(plugin_name)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取管理器统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "registered_plugins": len(self._plugin_registry),
            "loaded_plugins": len(self._loaded_plugins),
            "supported_models": len(self._model_to_plugin),
            "loaded_plugin_names": list(self._loaded_plugins.keys()),
            "load_ratio": len(self._loaded_plugins) / len(self._plugin_registry) if self._plugin_registry else 0
        }
    
    def cleanup(self):
        """清理资源
        
        卸载所有插件并关闭线程池。
        """
        logger.info("开始清理LazyPluginManager资源")
        
        # 卸载所有插件
        for plugin_name in list(self._loaded_plugins.keys()):
            self.unload_plugin(plugin_name)
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        logger.info("LazyPluginManager资源清理完成")
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass  # 忽略析构时的异常


# 全局延迟插件管理器实例
_lazy_plugin_manager: Optional[LazyPluginManager] = None
_manager_lock = threading.Lock()


def get_lazy_plugin_manager(config: Optional[Dict[str, Any]] = None) -> LazyPluginManager:
    """获取全局延迟插件管理器实例
    
    使用单例模式确保全局只有一个管理器实例。
    
    Args:
        config: 配置字典，仅在首次创建时使用
        
    Returns:
        延迟插件管理器实例
    """
    global _lazy_plugin_manager
    
    if _lazy_plugin_manager is None:
        with _manager_lock:
            if _lazy_plugin_manager is None:
                _lazy_plugin_manager = LazyPluginManager(config)
    
    return _lazy_plugin_manager


def reset_lazy_plugin_manager():
    """重置全局延迟插件管理器
    
    主要用于测试场景。
    """
    global _lazy_plugin_manager
    
    with _manager_lock:
        if _lazy_plugin_manager is not None:
            _lazy_plugin_manager.cleanup()
            _lazy_plugin_manager = None