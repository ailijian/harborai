#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI - 世界级多模型统一客户端

提供与 OpenAI SDK 几乎一致的开发体验，兼具灵活性、可靠性与可观测性。
支持推理模型、结构化输出、异步日志、容错降级等生产级功能。
"""

__version__ = "1.0.0-beta.6"
__author__ = "HarborAI Team"
__email__ = "team@harborai.com"
__description__ = "世界级多模型统一客户端，提供与 OpenAI SDK 几乎一致的开发体验"

# 导入主要的客户端类
from .api.client import HarborAI
from .utils.exceptions import (
    HarborAIError,
    APIError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ModelNotFoundError,
    PluginError,
)

# 导出公共接口
__all__ = [
    "HarborAI",
    "HarborAIError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "TimeoutError",
    "ModelNotFoundError",
    "PluginError",
    "__version__",
]