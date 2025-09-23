#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 插件模块

包含各个 LLM 厂商的插件实现。
"""

from .openai_plugin import OpenAIPlugin
from .deepseek_plugin import DeepSeekPlugin
from .doubao_plugin import DoubaoPlugin
from .wenxin_plugin import WenxinPlugin

__all__ = [
    "OpenAIPlugin",
    "DeepSeekPlugin", 
    "DoubaoPlugin",
    "WenxinPlugin"
]