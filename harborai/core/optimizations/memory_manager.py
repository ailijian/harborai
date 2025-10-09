#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存管理器模块

统一管理内存优化组件，包括缓存、对象池和弱引用机制，
根据HarborAI SDK性能优化技术设计方案要求。

设计目标：
1. 统一管理所有内存优化组件
2. 提供简化的API接口
3. 自动化内存监控和清理
4. 配置化的优化策略
5. 性能指标收集和报告

技术实现：
- 集成MemoryOptimizedCache和ObjectPool
- 实现弱引用管理
- 提供统一的配置接口
- 自动化的内存监控
"""

import threading
import time
import weakref
from typing import Any, Dict, Optional, Type, Callable, List
import logging
import gc

from .memory_optimized_cache import MemoryOptimizedCache
from .object_pool import ObjectPool, ObjectPoolManager
from ...utils.logger import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """内存管理器
    
    统一管理内存优化组件，提供简化的API和自动化管理。
    
    特性：
    1. 统一接口：提供统一的缓存和对象池接口
    2. 自动管理：自动清理和优化内存使用
    3. 监控报告：实时监控内存使用和性能指标
    4. 配置化：支持灵活的配置选项
    5. 弱引用：自动管理对象生命周期
    
    Assumptions:
    - A1: 应用程序有明确的内存使用模式，可以通过配置优化
    - A2: 缓存和对象池的使用是互补的，不会产生冲突
    - A3: 弱引用机制不会影响正常的对象生命周期
    - A4: 自动清理的开销是可接受的
    - A5: 内存监控不会显著影响性能
    
    验证方法：
    - 运行test_memory_optimization.py中的TestMemoryManager测试
    - 监控整体内存使用情况
    - 验证各组件的协调工作
    - 检查自动清理的效果
    
    回滚计划：
    - 如果统一管理导致性能问题，可以回退到独立使用各组件
    - 如果自动清理过于频繁，可以调整清理策略
    - 如果弱引用导致问题，可以禁用弱引用功能
    """
    
    def __init__(self,
                 cache_size: int = 1000,
                 cache_ttl: Optional[float] = None,
                 object_pool_size: int = 100,
                 enable_weak_references: bool = True,
                 auto_cleanup_interval: float = 300.0,
                 memory_threshold_mb: float = 100.0):
        """初始化内存管理器
        
        Args:
            cache_size: 缓存最大大小
            cache_ttl: 缓存项生存时间（秒）
            object_pool_size: 对象池最大大小
            enable_weak_references: 是否启用弱引用
            auto_cleanup_interval: 自动清理间隔（秒）
            memory_threshold_mb: 内存阈值（MB），超过时触发清理
        """
        # 配置验证和修正
        self._cache_size = max(1, int(cache_size)) if isinstance(cache_size, (int, float)) and cache_size > 0 else 1000
        self._cache_ttl = cache_ttl if isinstance(cache_ttl, (int, float, type(None))) else None
        self._object_pool_size = max(1, int(object_pool_size)) if isinstance(object_pool_size, (int, float)) and object_pool_size > 0 else 100
        self._enable_weak_references = bool(enable_weak_references) if isinstance(enable_weak_references, bool) else True
        self._auto_cleanup_interval = max(1.0, float(auto_cleanup_interval)) if isinstance(auto_cleanup_interval, (int, float)) and auto_cleanup_interval > 0 else 300.0
        self._memory_threshold_mb = max(1.0, float(memory_threshold_mb)) if isinstance(memory_threshold_mb, (int, float)) and memory_threshold_mb > 0 else 100.0
        
        # 记录配置修正
        if cache_size != self._cache_size:
            logger.warning("无效的cache_size配置 %s，使用默认值 %d", cache_size, self._cache_size)
        if object_pool_size != self._object_pool_size:
            logger.warning("无效的object_pool_size配置 %s，使用默认值 %d", object_pool_size, self._object_pool_size)
        
        # 初始化组件（使用修正后的值）
        self._cache = MemoryOptimizedCache(
            max_size=self._cache_size,
            ttl_seconds=cache_ttl,
            cleanup_interval=auto_cleanup_interval,
            enable_weak_refs=enable_weak_references
        )
        
        self._object_pool_manager = ObjectPoolManager()
        
        # 创建默认对象池（使用修正后的值）
        self._default_object_pool = self._object_pool_manager.create_pool(
            name='default',
            object_type=object,
            max_size=self._object_pool_size
        )
        
        # 弱引用管理
        if enable_weak_references:
            self._weak_refs: Dict[str, weakref.ref] = {}
            self._weak_refs_lock = threading.RLock()
        
        # 统计和监控
        self._stats = {
            'start_time': time.time(),
            'cache_hits': 0,
            'cache_misses': 0,
            'objects_created': 0,
            'objects_reused': 0,
            'cleanup_count': 0,
            'memory_warnings': 0
        }
        self._stats_lock = threading.RLock()
        
        # 自动清理
        self._cleanup_timer: Optional[threading.Timer] = None
        self._start_auto_cleanup()
        
        logger.debug("MemoryManager初始化完成，cache_size=%d, pool_size=%d", 
                    cache_size, object_pool_size)


    @property
    def cache(self) -> MemoryOptimizedCache:
        """获取缓存实例"""
        return self._cache
    
    @property
    def object_pool(self) -> ObjectPool:
        """获取默认对象池实例"""
        return self._default_object_pool
    
    def create_object_pool(self,
                          name: str,
                          object_type: Type,
                          max_size: Optional[int] = None,
                          factory_func: Optional[Callable] = None,
                          reset_func: Optional[Callable] = None,
                          cleanup_func: Optional[Callable] = None) -> ObjectPool:
        """创建对象池
        
        Args:
            name: 池名称
            object_type: 对象类型
            max_size: 最大大小，None使用默认值
            factory_func: 工厂函数
            reset_func: 重置函数
            cleanup_func: 清理函数
            
        Returns:
            创建的对象池
        """
        if max_size is None:
            max_size = self._object_pool_size
        
        return self._object_pool_manager.create_pool(
            name=name,
            object_type=object_type,
            max_size=max_size,
            factory_func=factory_func,
            reset_func=reset_func,
            cleanup_func=cleanup_func
        )
    
    def get_pooled_object(self, pool_name: str) -> Optional[Any]:
        """从对象池获取对象
        
        Args:
            pool_name: 池名称
            
        Returns:
            对象实例，如果池不存在则返回None
        """
        try:
            obj = self._object_pool_manager.acquire_object(pool_name)
            if obj is not None:
                with self._stats_lock:
                    pool = self._object_pool_manager.get_pool(pool_name)
                    if pool and pool._reused_count > 0:
                        self._stats['objects_reused'] += 1
                    else:
                        self._stats['objects_created'] += 1
            return obj
        except Exception as e:
            logger.error("从对象池'%s'获取对象时发生错误: %s", pool_name, e)
            return None
    
    def release_pooled_object(self, pool_name: str, obj: Any) -> bool:
        """释放对象到对象池
        
        Args:
            pool_name: 池名称
            obj: 要释放的对象
            
        Returns:
            是否成功释放
        """
        return self._object_pool_manager.release_object(pool_name, obj)
    
    def add_weak_reference(self, key: str, obj: Any, 
                          callback: Optional[Callable] = None) -> bool:
        """添加弱引用
        
        Args:
            key: 引用键
            obj: 要引用的对象
            callback: 对象被删除时的回调函数
            
        Returns:
            是否成功添加
        """
        if not self._enable_weak_references:
            return False
        
        try:
            with self._weak_refs_lock:
                if callback:
                    ref = weakref.ref(obj, callback)
                else:
                    ref = weakref.ref(obj, lambda ref: self._on_weak_ref_deleted(key))
                
                self._weak_refs[key] = ref
                logger.debug("添加弱引用: %s", key)
                return True
        except TypeError:
            # 对象不支持弱引用
            logger.warning("对象不支持弱引用: %s", key)
            return False
    
    def get_weak_reference(self, key: str) -> Optional[Any]:
        """获取弱引用对象
        
        Args:
            key: 引用键
            
        Returns:
            引用的对象，如果已被删除则返回None
        """
        if not self._enable_weak_references:
            return None
        
        with self._weak_refs_lock:
            ref = self._weak_refs.get(key)
            if ref:
                obj = ref()
                if obj is None:
                    # 对象已被删除，清理引用
                    del self._weak_refs[key]
                return obj
            return None
    
    def remove_weak_reference(self, key: str) -> bool:
        """移除弱引用
        
        Args:
            key: 引用键
            
        Returns:
            是否成功移除
        """
        if not self._enable_weak_references:
            return False
        
        with self._weak_refs_lock:
            if key in self._weak_refs:
                del self._weak_refs[key]
                logger.debug("移除弱引用: %s", key)
                return True
            return False
    
    def cleanup(self, force_clear: bool = False) -> Dict[str, int]:
        """执行内存清理
        
        Args:
            force_clear: 是否强制清空所有缓存和对象池
        
        Returns:
            清理统计信息
        """
        cleanup_stats = {
            'cache_expired': 0,
            'cache_cleared': 0,
            'pools_shrunk': 0,
            'pools_cleared': 0,
            'weak_refs_cleaned': 0
        }
        
        if force_clear:
            # 强制清空缓存
            cache_size_before = self._cache.size()
            self._cache.clear()
            cleanup_stats['cache_cleared'] = cache_size_before
            
            # 清空所有对象池
            for pool_name, pool in self._object_pool_manager._pools.items():
                pool_size_before = pool.size()
                pool.clear()
                cleanup_stats['pools_cleared'] += pool_size_before
        else:
            # 清理过期缓存
            cleanup_stats['cache_expired'] = self._cache.cleanup_expired()
            
            # 收缩对象池
            for pool_name, pool in self._object_pool_manager._pools.items():
                if pool.size() > pool._max_size // 2:
                    shrunk = pool.shrink()
                    if shrunk > 0:
                        cleanup_stats['pools_shrunk'] += shrunk
        
        # 清理弱引用 - 多轮清理确保彻底
        if self._enable_weak_references:
            # 多轮清理，确保所有死弱引用被清理
            for cleanup_round in range(3):
                with self._weak_refs_lock:
                    dead_refs = []
                    for key, ref in self._weak_refs.items():
                        if ref() is None:
                            dead_refs.append(key)
                    
                    for key in dead_refs:
                        del self._weak_refs[key]
                        cleanup_stats['weak_refs_cleaned'] += 1
                
                # 每轮清理后进行垃圾回收
                gc.collect()
                
                # 如果没有死弱引用，提前退出
                if not dead_refs:
                    break
        
        # 最终强制垃圾回收
        gc.collect()
        
        with self._stats_lock:
            self._stats['cleanup_count'] += 1
        
        logger.debug("内存清理完成: %s", cleanup_stats)
        return cleanup_stats
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息
        
        Returns:
            包含各种内存统计的字典
        """
        with self._stats_lock:
            cache_stats = self._cache.get_stats()
            pool_stats = self._object_pool_manager.get_all_stats()
            
            uptime = time.time() - self._stats['start_time']
            
            stats = {
                'uptime_seconds': uptime,
                'cache': cache_stats,
                'object_pools': pool_stats,
                'weak_references_count': len(self._weak_refs) if self._enable_weak_references else 0,
                'total_cache_hits': self._stats['cache_hits'],
                'total_cache_misses': self._stats['cache_misses'],
                'total_objects_created': self._stats['objects_created'],
                'total_objects_reused': self._stats['objects_reused'],
                'cleanup_count': self._stats['cleanup_count'],
                'memory_warnings': self._stats['memory_warnings']
            }
            
            # 计算复用率
            total_objects = self._stats['objects_created'] + self._stats['objects_reused']
            if total_objects > 0:
                stats['object_reuse_rate'] = self._stats['objects_reused'] / total_objects
            else:
                stats['object_reuse_rate'] = 0.0
            
            return stats
    
    def check_memory_usage(self) -> bool:
        """检查内存使用情况
        
        Returns:
            是否超过阈值
        """
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self._memory_threshold_mb:
                with self._stats_lock:
                    self._stats['memory_warnings'] += 1
                
                logger.warning("内存使用超过阈值: %.2fMB > %.2fMB", 
                             memory_mb, self._memory_threshold_mb)
                return True
            
            return False
        except ImportError:
            logger.warning("psutil未安装，无法监控内存使用")
            return False
    
    def _on_weak_ref_deleted(self, key: str) -> None:
        """弱引用对象被删除时的回调
        
        Args:
            key: 引用键
        """
        with self._weak_refs_lock:
            if key in self._weak_refs:
                del self._weak_refs[key]
                logger.debug("弱引用对象被删除: %s", key)
    
    def _start_auto_cleanup(self) -> None:
        """启动自动清理定时器"""
        if self._auto_cleanup_interval > 0:
            self._cleanup_timer = threading.Timer(
                self._auto_cleanup_interval,
                self._auto_cleanup_task
            )
            self._cleanup_timer.daemon = True
            self._cleanup_timer.start()
    
    def _auto_cleanup_task(self) -> None:
        """自动清理任务"""
        try:
            # 检查内存使用
            memory_exceeded = self.check_memory_usage()
            
            # 如果内存超标或到了清理时间，执行清理
            if memory_exceeded or True:  # 定期清理
                self.cleanup()
            
        except Exception as e:
            logger.error("自动清理任务失败: %s", str(e))
        finally:
            # 重新启动定时器
            self._start_auto_cleanup()
    
    def shutdown(self) -> None:
        """关闭内存管理器"""
        # 停止自动清理
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        # 清理所有资源
        self._cache.clear()
        self._object_pool_manager.clear_all()
        
        if self._enable_weak_references:
            with self._weak_refs_lock:
                self._weak_refs.clear()
        
        logger.debug("MemoryManager已关闭")
    
    def __del__(self):
        """析构函数"""
        try:
            self.shutdown()
        except Exception:
            pass


# 全局内存管理器实例
_global_memory_manager: Optional[MemoryManager] = None


async def get_memory_manager(config: Optional[Dict[str, Any]] = None) -> MemoryManager:
    """获取全局内存管理器实例
    
    Args:
        config: 内存管理器配置
        
    Returns:
        内存管理器实例
    """
    global _global_memory_manager
    
    if _global_memory_manager is None:
        if config:
            _global_memory_manager = MemoryManager(**config)
        else:
            _global_memory_manager = MemoryManager()
    
    return _global_memory_manager