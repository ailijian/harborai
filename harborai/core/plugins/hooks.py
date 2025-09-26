#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 插件钩子系统

提供插件生命周期管理和事件处理机制。
"""

import logging
from enum import Enum
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class HookType(Enum):
    """钩子类型枚举"""
    BEFORE = "before"
    AFTER = "after"
    ERROR = "error"
    BEFORE_INIT = "before_init"
    AFTER_INIT = "after_init"
    BEFORE_EXECUTE = "before_execute"
    AFTER_EXECUTE = "after_execute"
    ON_ERROR = "on_error"
    ON_SUCCESS = "on_success"
    BEFORE_CLEANUP = "before_cleanup"
    AFTER_CLEANUP = "after_cleanup"


@dataclass
class HookContext:
    """钩子上下文
    
    包含钩子执行时的相关信息。
    """
    plugin_name: str
    hook_type: HookType
    data: Dict[str, Any]
    timestamp: float
    

class PluginHook(ABC):
    """插件钩子基类
    
    定义插件钩子的基本接口。
    """
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority  # 优先级，数值越大优先级越高
        self.enabled = True
    
    @abstractmethod
    def execute(self, context: HookContext) -> bool:
        """执行钩子
        
        Args:
            context: 钩子上下文
            
        Returns:
            bool: 是否继续执行后续钩子
        """
        pass
    
    def enable(self):
        """启用钩子"""
        self.enabled = True
    
    def disable(self):
        """禁用钩子"""
        self.enabled = False


class FunctionHook(PluginHook):
    """函数钩子
    
    将普通函数包装为钩子。
    """
    
    def __init__(self, name: str, func: Callable[[HookContext], bool], priority: int = 0):
        super().__init__(name, priority)
        self.func = func
    
    def execute(self, context: HookContext) -> bool:
        """执行函数钩子"""
        try:
            if self.enabled:
                return self.func(context)
            return True
        except Exception as e:
            logger.error(f"Hook {self.name} execution failed: {e}")
            return False


class HookManager:
    """钩子管理器
    
    管理插件的钩子注册和执行。
    """
    
    def __init__(self):
        self._hooks: Dict[HookType, List[PluginHook]] = {}
        for hook_type in HookType:
            self._hooks[hook_type] = []
    
    def register_hook(self, hook_type: HookType, hook: PluginHook) -> bool:
        """注册钩子
        
        Args:
            hook_type: 钩子类型
            hook: 钩子实例
            
        Returns:
            bool: 注册是否成功
        """
        try:
            if hook_type not in self._hooks:
                self._hooks[hook_type] = []
            
            # 检查是否已存在同名钩子
            existing_names = [h.name for h in self._hooks[hook_type]]
            if hook.name in existing_names:
                logger.warning(f"Hook {hook.name} already exists for {hook_type.value}, replacing")
                self.unregister_hook(hook_type, hook.name)
            
            # 插入钩子并按优先级排序
            self._hooks[hook_type].append(hook)
            self._hooks[hook_type].sort(key=lambda h: h.priority, reverse=True)
            
            logger.info(f"Hook {hook.name} registered for {hook_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register hook {hook.name}: {e}")
            return False
    
    def unregister_hook(self, hook_type: HookType, hook_name: str) -> bool:
        """注销钩子
        
        Args:
            hook_type: 钩子类型
            hook_name: 钩子名称
            
        Returns:
            bool: 注销是否成功
        """
        try:
            if hook_type in self._hooks:
                hooks = self._hooks[hook_type]
                for i, hook in enumerate(hooks):
                    if hook.name == hook_name:
                        del hooks[i]
                        logger.info(f"Hook {hook_name} unregistered from {hook_type.value}")
                        return True
            
            logger.warning(f"Hook {hook_name} not found for {hook_type.value}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to unregister hook {hook_name}: {e}")
            return False
    
    def execute_hooks(self, hook_type: HookType, context: HookContext) -> bool:
        """执行指定类型的所有钩子
        
        Args:
            hook_type: 钩子类型
            context: 钩子上下文
            
        Returns:
            bool: 是否所有钩子都成功执行
        """
        try:
            if hook_type not in self._hooks:
                return True
            
            hooks = self._hooks[hook_type]
            for hook in hooks:
                if hook.enabled:
                    try:
                        if not hook.execute(context):
                            logger.warning(f"Hook {hook.name} returned False, stopping hook chain")
                            return False
                    except Exception as e:
                        logger.error(f"Hook {hook.name} execution failed: {e}")
                        # 根据钩子类型决定是否继续
                        if hook_type == HookType.ON_ERROR:
                            continue  # 错误处理钩子失败时继续执行其他钩子
                        else:
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute hooks for {hook_type.value}: {e}")
            return False
    
    def get_hooks(self, hook_type: HookType) -> List[PluginHook]:
        """获取指定类型的钩子列表
        
        Args:
            hook_type: 钩子类型
            
        Returns:
            钩子列表
        """
        return self._hooks.get(hook_type, [])
    
    def clear_hooks(self, hook_type: Optional[HookType] = None):
        """清除钩子
        
        Args:
            hook_type: 钩子类型，如果为None则清除所有钩子
        """
        try:
            if hook_type is None:
                for ht in HookType:
                    self._hooks[ht].clear()
                logger.info("All hooks cleared")
            else:
                if hook_type in self._hooks:
                    self._hooks[hook_type].clear()
                    logger.info(f"Hooks for {hook_type.value} cleared")
        except Exception as e:
            logger.error(f"Failed to clear hooks: {e}")
    
    def list_hooks(self) -> Dict[str, List[str]]:
        """列出所有钩子
        
        Returns:
            钩子类型到钩子名称列表的映射
        """
        result = {}
        for hook_type, hooks in self._hooks.items():
            result[hook_type.value] = [hook.name for hook in hooks]
        return result


# 全局钩子管理器实例
_global_hook_manager = HookManager()


def get_hook_manager() -> HookManager:
    """获取全局钩子管理器
    
    Returns:
        全局钩子管理器实例
    """
    return _global_hook_manager


def register_hook(hook_type: HookType, hook: PluginHook) -> bool:
    """注册钩子到全局管理器
    
    Args:
        hook_type: 钩子类型
        hook: 钩子实例
        
    Returns:
        bool: 注册是否成功
    """
    return _global_hook_manager.register_hook(hook_type, hook)


def register_function_hook(hook_type: HookType, name: str, func: Callable[[HookContext], bool], priority: int = 0) -> bool:
    """注册函数钩子到全局管理器
    
    Args:
        hook_type: 钩子类型
        name: 钩子名称
        func: 钩子函数
        priority: 优先级
        
    Returns:
        bool: 注册是否成功
    """
    hook = FunctionHook(name, func, priority)
    return register_hook(hook_type, hook)


def execute_hooks(hook_type: HookType, plugin_name: str, data: Dict[str, Any]) -> bool:
    """执行钩子
    
    Args:
        hook_type: 钩子类型
        plugin_name: 插件名称
        data: 钩子数据
        
    Returns:
        bool: 是否所有钩子都成功执行
    """
    import time
    context = HookContext(
        plugin_name=plugin_name,
        hook_type=hook_type,
        data=data,
        timestamp=time.time()
    )
    return _global_hook_manager.execute_hooks(hook_type, context)