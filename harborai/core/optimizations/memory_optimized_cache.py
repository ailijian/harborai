#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化缓存模块

实现智能缓存管理，使用LRU策略和定期清理机制，
根据HarborAI SDK性能优化技术设计方案要求。

设计目标：
1. 减少内存占用
2. 提供高效的LRU淘汰策略
3. 支持定期清理和手动清理
4. 线程安全
5. 内存使用监控

技术实现：
- 使用OrderedDict实现LRU
- 定期清理过期项
- 内存使用统计
- 线程安全锁机制
"""

import time
import threading
import weakref
from collections import OrderedDict
from typing import Any, Optional, Dict, Callable
import logging

from ...utils.logger import get_logger

logger = get_logger(__name__)


class MemoryOptimizedCache:
    """内存优化的缓存管理器
    
    使用LRU策略和定期清理机制，优化内存使用。
    
    特性：
    1. LRU淘汰策略：最少使用的项目优先被淘汰
    2. 定期清理：自动清理过期和长时间未访问的项目
    3. 内存监控：跟踪缓存的内存使用情况
    4. 线程安全：支持多线程并发访问
    5. 弱引用支持：避免循环引用导致的内存泄漏
    
    Assumptions:
    - A1: 缓存项的访问遵循局部性原理，最近访问的项目更可能再次被访问
    - A2: 系统有足够的内存来维护指定大小的缓存
    - A3: 缓存的键都是可哈希的对象
    - A4: 缓存值的序列化/反序列化开销可以接受
    - A5: 定期清理的开销不会显著影响性能
    
    验证方法：
    - 运行test_memory_optimization.py中的TestMemoryOptimizedCache测试
    - 监控内存使用情况，确保不超过预期阈值
    - 验证LRU策略的正确性
    - 检查线程安全性
    
    回滚计划：
    - 如果内存使用超标，可以回退到简单的dict缓存
    - 如果性能下降，可以禁用定期清理功能
    - 如果出现线程安全问题，可以使用更粗粒度的锁
    """
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_seconds: Optional[float] = None,
                 cleanup_interval: float = 300.0,
                 enable_weak_refs: bool = True):
        """初始化内存优化缓存
        
        Args:
            max_size: 最大缓存项数量
            ttl_seconds: 缓存项生存时间（秒），None表示不过期
            cleanup_interval: 清理间隔（秒）
            enable_weak_refs: 是否启用弱引用支持
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._cleanup_interval = cleanup_interval
        self._enable_weak_refs = enable_weak_refs
        
        # 核心数据结构
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._access_times: Dict[str, float] = {}
        self._creation_times: Dict[str, float] = {}
        
        # 弱引用支持
        if enable_weak_refs:
            self._weak_refs: Dict[str, weakref.ref] = {}
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 统计信息
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        # 定期清理
        self._cleanup_timer: Optional[threading.Timer] = None
        self._start_cleanup_timer()
        
        logger.debug("MemoryOptimizedCache初始化完成，max_size=%d, ttl=%s", 
                    max_size, ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或已过期则返回None
        """
        with self._lock:
            # 检查是否存在
            if key not in self._cache:
                self._misses += 1
                return None
            
            # 检查是否过期
            if self._is_expired(key):
                self._remove_item(key)
                self._misses += 1
                return None
            
            # 更新访问时间
            self._access_times[key] = time.time()
            
            # 移动到末尾（LRU策略）
            value = self._cache.pop(key)
            self._cache[key] = value
            
            self._hits += 1
            return value
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存项
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        with self._lock:
            current_time = time.time()
            
            # 如果已存在，更新值
            if key in self._cache:
                self._cache.pop(key)
                self._cache[key] = value
                self._access_times[key] = current_time
                self._creation_times[key] = current_time
                return
            
            # 检查是否需要淘汰
            if len(self._cache) >= self._max_size:
                self._evict_lru()
            
            # 添加新项
            self._cache[key] = value
            self._access_times[key] = current_time
            self._creation_times[key] = current_time
            
            # 弱引用支持
            if self._enable_weak_refs and hasattr(value, '__weakref__'):
                try:
                    self._weak_refs[key] = weakref.ref(value, 
                                                      lambda ref: self._on_weak_ref_deleted(key))
                except TypeError:
                    # 某些对象不支持弱引用
                    pass
    
    def delete(self, key: str) -> bool:
        """删除缓存项
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功删除
        """
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                return True
            return False
    
    def clear(self) -> None:
        """清空所有缓存项"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._creation_times.clear()
            if self._enable_weak_refs:
                self._weak_refs.clear()
            
            logger.debug("缓存已清空")
    
    def size(self) -> int:
        """获取当前缓存项数量"""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            包含命中率、大小等信息的字典
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions,
                'ttl_seconds': self._ttl_seconds
            }
    
    def cleanup_expired(self) -> int:
        """清理过期项
        
        Returns:
            清理的项目数量
        """
        if self._ttl_seconds is None:
            return 0
        
        with self._lock:
            expired_keys = []
            current_time = time.time()
            
            for key in list(self._cache.keys()):
                if self._is_expired(key, current_time):
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_item(key)
            
            if expired_keys:
                logger.debug("清理了%d个过期缓存项", len(expired_keys))
            
            return len(expired_keys)
    
    def _is_expired(self, key: str, current_time: Optional[float] = None) -> bool:
        """检查缓存项是否过期
        
        Args:
            key: 缓存键
            current_time: 当前时间，None表示使用当前时间
            
        Returns:
            是否过期
        """
        if self._ttl_seconds is None:
            return False
        
        if current_time is None:
            current_time = time.time()
        
        creation_time = self._creation_times.get(key, 0)
        return (current_time - creation_time) > self._ttl_seconds
    
    def _evict_lru(self) -> None:
        """淘汰最少使用的缓存项"""
        if not self._cache:
            return
        
        # OrderedDict的第一个项是最少使用的
        lru_key = next(iter(self._cache))
        self._remove_item(lru_key)
        self._evictions += 1
        
        logger.debug("淘汰LRU缓存项: %s", lru_key)
    
    def _remove_item(self, key: str) -> None:
        """移除缓存项及其相关数据
        
        Args:
            key: 要移除的缓存键
        """
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._creation_times.pop(key, None)
        if self._enable_weak_refs:
            self._weak_refs.pop(key, None)
    
    def _on_weak_ref_deleted(self, key: str) -> None:
        """弱引用对象被删除时的回调
        
        Args:
            key: 对应的缓存键
        """
        with self._lock:
            if key in self._cache:
                self._remove_item(key)
                logger.debug("弱引用对象被删除，移除缓存项: %s", key)
    
    def _start_cleanup_timer(self) -> None:
        """启动定期清理定时器"""
        if self._cleanup_interval > 0:
            self._cleanup_timer = threading.Timer(
                self._cleanup_interval, 
                self._periodic_cleanup
            )
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()
    
    def _periodic_cleanup(self) -> None:
        """定期清理任务"""
        try:
            expired_count = self.cleanup_expired()
            if expired_count > 0:
                logger.debug("定期清理完成，清理了%d个过期项", expired_count)
        except Exception as e:
            logger.error("定期清理失败: %s", str(e))
        finally:
            # 重新启动定时器
            self._start_cleanup_timer()
    
    def __del__(self):
        """析构函数，清理定时器"""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
    
    def __len__(self) -> int:
        """返回缓存项数量"""
        return self.size()
    
    def __contains__(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._cache and not self._is_expired(key)