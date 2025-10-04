#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化模块

提供内存使用优化的核心组件，包括：
- MemoryOptimizedCache: 智能缓存管理
- ObjectPool: 对象池技术
- MemoryManager: 统一内存管理
- WeakReference: 弱引用机制

根据HarborAI SDK性能优化技术设计方案第二阶段要求实现。
"""

# 逐步导入已实现的组件
try:
    from .memory_optimized_cache import MemoryOptimizedCache
    __all__ = ['MemoryOptimizedCache']
except ImportError:
    __all__ = []

try:
    from .object_pool import ObjectPool
    __all__.append('ObjectPool')
except ImportError:
    pass

try:
    from .memory_manager import MemoryManager
    __all__.append('MemoryManager')
except ImportError:
    pass