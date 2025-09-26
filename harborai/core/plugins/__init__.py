#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 插件模块

包含各种LLM厂商的插件实现。
"""

from .base import Plugin, PluginInfo, BaseLLMPlugin, BasePlugin
from .manager import PluginManager, PluginRegistry
from .hooks import PluginHook, HookType, HookManager, FunctionHook
from .openai_plugin import OpenAIPlugin
from .deepseek_plugin import DeepSeekPlugin
from .doubao_plugin import DoubaoPlugin
from .wenxin_plugin import WenxinPlugin

__all__ = [
    'Plugin',
    'PluginInfo',
    'BaseLLMPlugin',
    'BasePlugin',
    'PluginManager',
    'PluginRegistry',
    'PluginHook',
    'HookType',
    'HookManager',
    'FunctionHook',
    'OpenAIPlugin',
    'DeepSeekPlugin', 
    'DoubaoPlugin',
    'WenxinPlugin'
]