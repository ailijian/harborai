#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
客户端管理器

负责插件注册、模型路由、降级策略等核心功能。
动态扫描插件目录，管理多个 LLM 厂商的插件。
"""

import importlib
import pkgutil
from typing import Dict, List, Optional, Type, Union, Any
from pathlib import Path

from .base_plugin import BaseLLMPlugin, ModelInfo, ChatCompletion, ChatCompletionChunk, ChatMessage
from ..utils.exceptions import ModelNotFoundError, PluginError
from ..utils.logger import get_logger
from ..utils.tracer import get_current_trace_id
from ..config.settings import get_settings


class ClientManager:
    """客户端管理器"""
    
    def __init__(self, client_config: Optional[Dict[str, Any]] = None):
        self.plugins: Dict[str, BaseLLMPlugin] = {}
        self.model_to_plugin: Dict[str, str] = {}
        self.logger = get_logger("harborai.client_manager")
        self.settings = get_settings()
        self.client_config = client_config or {}
        
        # 自动加载插件
        self._load_plugins()
    
    def _load_plugins(self) -> None:
        """自动加载插件"""
        trace_id = get_current_trace_id()
        self.logger.info(f"Loading plugins [trace_id={trace_id}]")
        
        for plugin_dir in self.settings.plugin_directories:
            try:
                self._scan_plugin_directory(plugin_dir)
            except Exception as e:
                self.logger.error(
                    "Failed to load plugins from directory",
                    trace_id=get_current_trace_id(),
                    directory=plugin_dir,
                    error=str(e)
                )
        
        self.logger.info(
            f"Plugin loading completed [trace_id={get_current_trace_id()}] loaded_plugins={list(self.plugins.keys())} total_models={len(self.model_to_plugin)}"
        )
    
    def _scan_plugin_directory(self, plugin_dir: str) -> None:
        """扫描插件目录"""
        try:
            # 导入插件包
            package = importlib.import_module(plugin_dir)
            
            # 扫描包中的模块
            for importer, modname, ispkg in pkgutil.iter_modules(
                package.__path__, package.__name__ + "."
            ):
                if not ispkg and modname.endswith('_plugin'):
                    try:
                        self._load_plugin_module(modname)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load plugin module [trace_id={get_current_trace_id()}] module={modname} error={str(e)}"
                        )
        except ImportError as e:
            self.logger.warning(
                f"Plugin directory not found [trace_id={get_current_trace_id()}] directory={plugin_dir} error={str(e)}"
            )
    
    def _load_plugin_module(self, module_name: str) -> None:
        """加载插件模块"""
        module = importlib.import_module(module_name)
        
        # 查找插件类
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            
            if (
                isinstance(attr, type) and
                issubclass(attr, BaseLLMPlugin) and
                attr != BaseLLMPlugin
            ):
                # 创建插件实例
                plugin_name = attr_name.lower().replace('plugin', '')
                plugin_config = self.settings.get_plugin_config(plugin_name)
                
                try:
                    # 合并客户端配置和插件配置
                    merged_config = plugin_config.copy()
                    
                    # 如果客户端提供了api_key，使用它
                    if 'api_key' in self.client_config:
                        merged_config['api_key'] = self.client_config['api_key']
                    
                    # 如果客户端提供了base_url，使用它
                    if 'base_url' in self.client_config:
                        merged_config['base_url'] = self.client_config['base_url']
                    
                    plugin_instance = attr(plugin_name, **merged_config)
                    self.register_plugin(plugin_instance)
                    
                    self.logger.info(
                        f"Plugin loaded successfully [trace_id={get_current_trace_id()}] plugin={plugin_name} module={module_name} supported_models={[m.id for m in plugin_instance.supported_models]}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to instantiate plugin [trace_id={get_current_trace_id()}] plugin={plugin_name} error={str(e)}"
                    )
    
    def register_plugin(self, plugin: BaseLLMPlugin) -> None:
        """注册插件"""
        if plugin.name in self.plugins:
            self.logger.warning(
                f"Plugin already registered, replacing [trace_id={get_current_trace_id()}] plugin={plugin.name}"
            )
        
        self.plugins[plugin.name] = plugin
        
        # 注册模型映射
        for model_info in plugin.supported_models:
            if model_info.id in self.model_to_plugin:
                existing_plugin = self.model_to_plugin[model_info.id]
                self.logger.warning(
                    f"Model already registered to another plugin [trace_id={get_current_trace_id()}] model={model_info.id} existing_plugin={existing_plugin} new_plugin={plugin.name}"
                )
            
            self.model_to_plugin[model_info.id] = plugin.name
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """注销插件"""
        if plugin_name not in self.plugins:
            raise PluginError(plugin_name, "Plugin not found")
        
        plugin = self.plugins[plugin_name]
        
        # 移除模型映射
        models_to_remove = []
        for model_name, mapped_plugin in self.model_to_plugin.items():
            if mapped_plugin == plugin_name:
                models_to_remove.append(model_name)
        
        for model_name in models_to_remove:
            del self.model_to_plugin[model_name]
        
        # 移除插件
        del self.plugins[plugin_name]
        
        self.logger.info(
            f"Plugin unregistered [trace_id={get_current_trace_id()}] plugin={plugin_name} removed_models={models_to_remove}"
        )
    
    def get_plugin_for_model(self, model_name: str) -> BaseLLMPlugin:
        """获取模型对应的插件"""
        # 检查模型映射
        if model_name in self.model_to_plugin:
            plugin_name = self.model_to_plugin[model_name]
            return self.plugins[plugin_name]
        
        # 检查模型映射配置
        if model_name in self.settings.model_mappings:
            mapped_model = self.settings.model_mappings[model_name]
            return self.get_plugin_for_model(mapped_model)
        
        raise ModelNotFoundError(
            model_name,
            trace_id=get_current_trace_id(),
            details={
                "available_models": list(self.model_to_plugin.keys()),
                "available_plugins": list(self.plugins.keys())
            }
        )
    
    def get_available_models(self) -> List[ModelInfo]:
        """获取所有可用模型"""
        models = []
        for plugin in self.plugins.values():
            models.extend(plugin.supported_models)
        return models
    
    def get_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """获取插件信息"""
        info = {}
        for plugin_name, plugin in self.plugins.items():
            info[plugin_name] = {
                "name": plugin.name,
                "config": plugin.config,
                "supported_models": [m.id for m in plugin.supported_models],
                "model_count": len(plugin.supported_models)
            }
        return info
    
    async def chat_completion_with_fallback(
        self,
        model: str,
        messages: List[ChatMessage],
        fallback: Optional[List[str]] = None,
        structured_provider: Optional[str] = None,
        **kwargs
    ) -> Union[ChatCompletion, Any]:
        """带降级策略的聊天完成"""
        models_to_try = [model]
        if fallback:
            models_to_try.extend(fallback)
        
        last_exception = None
        
        for attempt_model in models_to_try:
            try:
                plugin = self.get_plugin_for_model(attempt_model)
                
                self.logger.info(
                    f"Attempting chat completion [trace_id={get_current_trace_id()}] model={attempt_model} plugin={plugin.name} is_fallback={attempt_model != model}"
                )
                
                # 更新模型参数
                kwargs_copy = kwargs.copy()
                if structured_provider is not None:
                    kwargs_copy['structured_provider'] = structured_provider
                
                # 为推理模型过滤参数和处理消息
                from ..core.models import filter_parameters_for_model, is_reasoning_model
                
                # 过滤不支持的参数
                kwargs_copy = filter_parameters_for_model(attempt_model, kwargs_copy)
                
                # 处理推理模型的system消息
                processed_messages = messages
                if is_reasoning_model(attempt_model):
                    processed_messages = self._process_messages_for_reasoning_model(messages)
                
                return await plugin.chat_completion_async(
                    attempt_model, processed_messages, **kwargs_copy
                )
                
            except Exception as e:
                last_exception = e
                
                self.logger.warning(
                    f"Chat completion failed, trying next model [trace_id={get_current_trace_id()}] model={attempt_model} error={str(e)} remaining_models={len(models_to_try) - models_to_try.index(attempt_model) - 1}"
                )
                
                # 如果是最后一个模型，抛出异常
                if attempt_model == models_to_try[-1]:
                    self.logger.error(
                        f"All fallback models exhausted [trace_id={get_current_trace_id()}] attempted_models={models_to_try} final_error={str(e)}"
                    )
                    raise
        
        # 理论上不会到达这里
        if last_exception:
            raise last_exception
    
    def _process_messages_for_reasoning_model(self, messages: List[ChatMessage]) -> List[ChatMessage]:
        """处理推理模型的消息，转换system消息"""
        processed_messages = []
        
        for message in messages:
            if message.role == "system":
                # 将system消息转换为user消息
                if message.content:
                    # 如果已经有user消息，将system内容合并到第一个user消息中
                    user_messages = [msg for msg in messages if msg.role == "user"]
                    if user_messages:
                        # 跳过system消息，稍后合并
                        continue
                    else:
                        # 转换为user消息
                        from ..core.base_plugin import ChatMessage
                        processed_messages.append(ChatMessage(
                            role="user",
                            content=f"请按照以下指导原则回答：{message.content}\n\n现在请回答用户的问题。"
                        ))
            else:
                processed_messages.append(message)
        
        # 如果有system消息需要合并到第一个user消息
        system_messages = [msg for msg in messages if msg.role == "system"]
        if system_messages and processed_messages:
            first_user_idx = None
            for i, msg in enumerate(processed_messages):
                if msg.role == "user":
                    first_user_idx = i
                    break
            
            if first_user_idx is not None:
                # 合并system内容到第一个user消息
                system_content = "\n".join([msg.content for msg in system_messages if msg.content])
                original_user_content = processed_messages[first_user_idx].content
                
                from ..core.base_plugin import ChatMessage
                processed_messages[first_user_idx] = ChatMessage(
                    role="user",
                    content=f"请按照以下指导原则回答：{system_content}\n\n{original_user_content}"
                )
        
        return processed_messages
    
    def chat_completion_sync_with_fallback(
        self,
        model: str,
        messages: List[ChatMessage],
        fallback: Optional[List[str]] = None,
        structured_provider: Optional[str] = None,
        **kwargs
    ) -> Union[ChatCompletion, Any]:
        """同步版本的带降级策略的聊天完成"""
        models_to_try = [model]
        if fallback:
            models_to_try.extend(fallback)
        
        last_exception = None
        
        for attempt_model in models_to_try:
            try:
                plugin = self.get_plugin_for_model(attempt_model)
                
                self.logger.info(
                    "Attempting chat completion (sync)",
                    trace_id=get_current_trace_id(),
                    model=attempt_model,
                    plugin=plugin.name,
                    is_fallback=attempt_model != model
                )
                
                # 更新模型参数
                kwargs_copy = kwargs.copy()
                if structured_provider is not None:
                    kwargs_copy['structured_provider'] = structured_provider
                
                # 为推理模型过滤参数和处理消息
                from ..core.models import filter_parameters_for_model, is_reasoning_model
                
                # 过滤不支持的参数
                kwargs_copy = filter_parameters_for_model(attempt_model, kwargs_copy)
                
                # 处理推理模型的system消息
                processed_messages = messages
                if is_reasoning_model(attempt_model):
                    processed_messages = self._process_messages_for_reasoning_model(messages)
                
                return plugin.chat_completion(
                    attempt_model, processed_messages, **kwargs_copy
                )
                
            except Exception as e:
                last_exception = e
                
                self.logger.warning(
                    "Chat completion failed (sync), trying next model",
                    trace_id=get_current_trace_id(),
                    model=attempt_model,
                    error=str(e),
                    remaining_models=len(models_to_try) - models_to_try.index(attempt_model) - 1
                )
                
                # 如果是最后一个模型，抛出异常
                if attempt_model == models_to_try[-1]:
                    self.logger.error(
                        "All fallback models exhausted (sync)",
                        trace_id=get_current_trace_id(),
                        attempted_models=models_to_try,
                        final_error=str(e)
                    )
                    raise
        
        # 理论上不会到达这里
        if last_exception:
            raise last_exception