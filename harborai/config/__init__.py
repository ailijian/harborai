#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置模块

管理 HarborAI 的全局配置，包括默认设置、环境变量、插件配置等。
"""

from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]