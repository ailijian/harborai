#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对象池模块

实现对象池技术，通过复用对象减少GC压力，
根据HarborAI SDK性能优化技术设计方案要求。

设计目标：
1. 减少对象创建和销毁的开销
2. 降低垃圾回收压力
3. 提高内存使用效率
4. 支持多种对象类型
5. 线程安全

技术实现：
- 使用队列管理空闲对象
- 支持对象重置机制
- 自动扩容和收缩
- 线程安全的获取和释放
"""

import threading
import weakref
from collections import deque
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Generic
import logging

from ...utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ObjectPool(Generic[T]):
    """对象池管理器
    
    通过复用对象减少GC压力和内存分配开销。
    
    特性：
    1. 对象复用：避免重复创建和销毁对象
    2. 自动管理：自动扩容和收缩池大小
    3. 重置机制：支持对象重置以便复用
    4. 线程安全：支持多线程并发访问
    5. 类型安全：使用泛型确保类型安全
    
    Assumptions:
    - A1: 对象创建成本相对较高，复用能带来性能提升
    - A2: 对象支持重置操作，可以安全复用
    - A3: 池中对象的生命周期由池管理，不会被外部长期持有
    - A4: 对象类型有默认构造函数或提供了工厂函数
    - A5: 多线程环境下需要线程安全保证
    
    验证方法：
    - 运行test_memory_optimization.py中的TestObjectPool测试
    - 监控GC频率和内存分配
    - 验证对象复用的正确性
    - 检查线程安全性
    
    回滚计划：
    - 如果对象复用导致状态污染，可以禁用重置机制
    - 如果内存使用过高，可以减小池大小或禁用对象池
    - 如果出现线程安全问题，可以使用更粗粒度的锁
    """
    
    def __init__(self, 
                 object_type: Type[T],
                 max_size: int = 100,
                 factory_func: Optional[Callable[[], T]] = None,
                 reset_func: Optional[Callable[[T], None]] = None,
                 cleanup_func: Optional[Callable[[T], None]] = None):
        """初始化对象池
        
        Args:
            object_type: 对象类型
            max_size: 池的最大大小
            factory_func: 对象工厂函数，None表示使用默认构造函数
            reset_func: 对象重置函数，None表示不重置
            cleanup_func: 对象清理函数，None表示不清理
        """
        self._object_type = object_type
        self._max_size = max_size
        self._factory_func = factory_func or (lambda: object_type())
        self._reset_func = reset_func
        self._cleanup_func = cleanup_func
        
        # 对象池
        self._pool: deque[T] = deque()
        # 使用对象id来跟踪活跃对象，避免弱引用和哈希问题
        self._active_object_ids: set[int] = set()
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 统计信息
        self._created_count = 0
        self._acquired_count = 0
        self._released_count = 0
        self._reused_count = 0
        
        logger.debug("ObjectPool初始化完成，类型=%s, max_size=%d", 
                    object_type.__name__, max_size)
    
    def acquire(self) -> T:
        """获取对象
        
        Returns:
            池中的对象或新创建的对象
        """
        with self._lock:
            # 尝试从池中获取
            if self._pool:
                obj = self._pool.popleft()
                self._reused_count += 1
                logger.debug("从池中复用对象，类型=%s", self._object_type.__name__)
            else:
                # 创建新对象
                obj = self._factory_func()
                self._created_count += 1
                logger.debug("创建新对象，类型=%s", self._object_type.__name__)
            
            # 添加到活跃对象集合
            self._active_object_ids.add(id(obj))
            self._acquired_count += 1
            
            return obj
    
    def release(self, obj: T) -> None:
        """释放对象回池中
        
        Args:
            obj: 要释放的对象
        """
        if obj is None:
            return
        
        with self._lock:
            # 检查对象是否属于此池
            obj_id = id(obj)
            if obj_id not in self._active_object_ids:
                logger.warning("尝试释放不属于此池的对象，类型=%s", 
                             self._object_type.__name__)
                return
            
            # 从活跃对象中移除
            self._active_object_ids.discard(obj_id)
            
            # 检查池是否已满
            if len(self._pool) >= self._max_size:
                # 池已满，清理对象
                if self._cleanup_func:
                    try:
                        self._cleanup_func(obj)
                    except Exception as e:
                        logger.error("对象清理失败: %s", str(e))
                logger.debug("池已满，丢弃对象，类型=%s", self._object_type.__name__)
                return
            
            # 重置对象状态
            if self._reset_func:
                try:
                    self._reset_func(obj)
                except Exception as e:
                    logger.error("对象重置失败: %s", str(e))
                    # 重置失败，不放回池中
                    if self._cleanup_func:
                        try:
                            self._cleanup_func(obj)
                        except Exception:
                            pass
                    return
            
            # 放回池中
            self._pool.append(obj)
            self._released_count += 1
            logger.debug("对象已释放回池，类型=%s", self._object_type.__name__)
    
    def clear(self) -> None:
        """清空对象池"""
        with self._lock:
            # 清理池中的对象
            if self._cleanup_func:
                while self._pool:
                    obj = self._pool.popleft()
                    try:
                        self._cleanup_func(obj)
                    except Exception as e:
                        logger.error("对象清理失败: %s", str(e))
            else:
                self._pool.clear()
            
            logger.debug("对象池已清空，类型=%s", self._object_type.__name__)
    
    def size(self) -> int:
        """获取池中空闲对象数量"""
        with self._lock:
            return len(self._pool)
    
    def active_count(self) -> int:
        """获取活跃对象数量"""
        with self._lock:
            return len(self._active_object_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取对象池统计信息
        
        Returns:
            包含各种统计信息的字典
        """
        with self._lock:
            reuse_rate = (self._reused_count / self._acquired_count 
                         if self._acquired_count > 0 else 0.0)
            
            return {
                'object_type': self._object_type.__name__,
                'max_size': self._max_size,
                'pool_size': len(self._pool),
                'active_count': len(self._active_object_ids),
                'created_count': self._created_count,
                'acquired_count': self._acquired_count,
                'released_count': self._released_count,
                'reused_count': self._reused_count,
                'reuse_rate': reuse_rate
            }
    
    def shrink(self, target_size: Optional[int] = None) -> int:
        """收缩对象池大小
        
        Args:
            target_size: 目标大小，None表示收缩到一半
            
        Returns:
            实际移除的对象数量
        """
        with self._lock:
            if target_size is None:
                target_size = len(self._pool) // 2
            
            removed_count = 0
            while len(self._pool) > target_size and self._pool:
                obj = self._pool.pop()
                if self._cleanup_func:
                    try:
                        self._cleanup_func(obj)
                    except Exception as e:
                        logger.error("对象清理失败: %s", str(e))
                removed_count += 1
            
            if removed_count > 0:
                logger.debug("对象池收缩，移除%d个对象，类型=%s", 
                           removed_count, self._object_type.__name__)
            
            return removed_count
    
    def __len__(self) -> int:
        """返回池中对象数量"""
        return self.size()
    
    def __del__(self):
        """析构函数，清理资源"""
        try:
            self.clear()
        except Exception:
            pass


class ObjectPoolManager:
    """对象池管理器
    
    管理多个不同类型的对象池。
    """
    
    def __init__(self):
        """初始化对象池管理器"""
        self._pools: Dict[str, ObjectPool] = {}
        self._lock = threading.RLock()
        
        logger.debug("ObjectPoolManager初始化完成")
    
    def create_pool(self, 
                   name: str,
                   object_type: Type[T],
                   max_size: int = 100,
                   factory_func: Optional[Callable[[], T]] = None,
                   reset_func: Optional[Callable[[T], None]] = None,
                   cleanup_func: Optional[Callable[[T], None]] = None) -> ObjectPool[T]:
        """创建对象池
        
        Args:
            name: 池名称
            object_type: 对象类型
            max_size: 最大大小
            factory_func: 工厂函数
            reset_func: 重置函数
            cleanup_func: 清理函数
            
        Returns:
            创建的对象池
        """
        with self._lock:
            if name in self._pools:
                raise ValueError(f"对象池'{name}'已存在")
            
            pool = ObjectPool(
                object_type=object_type,
                max_size=max_size,
                factory_func=factory_func,
                reset_func=reset_func,
                cleanup_func=cleanup_func
            )
            
            self._pools[name] = pool
            logger.debug("创建对象池'%s'，类型=%s", name, object_type.__name__)
            
            return pool
    
    def get_pool(self, name: str) -> Optional[ObjectPool]:
        """获取对象池
        
        Args:
            name: 池名称
            
        Returns:
            对象池，如果不存在则返回None
        """
        with self._lock:
            return self._pools.get(name)
    
    def acquire_object(self, pool_name: str) -> Optional[Any]:
        """从指定池获取对象
        
        Args:
            pool_name: 池名称
            
        Returns:
            对象，如果池不存在则返回None
        """
        pool = self.get_pool(pool_name)
        if pool is not None:
            try:
                return pool.acquire()
            except Exception as e:
                logger.error("从池'%s'获取对象时发生错误: %s", pool_name, e)
                return None
        return None
    
    def release_object(self, pool_name: str, obj: Any) -> bool:
        """释放对象到指定池
        
        Args:
            pool_name: 池名称
            obj: 要释放的对象
            
        Returns:
            是否成功释放
        """
        pool = self.get_pool(pool_name)
        if pool is not None:
            pool.release(obj)
            return True
        return False
    
    def clear_all(self) -> None:
        """清空所有对象池"""
        with self._lock:
            for pool in self._pools.values():
                pool.clear()
            logger.debug("所有对象池已清空")
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有对象池的统计信息
        
        Returns:
            包含所有池统计信息的字典
        """
        with self._lock:
            return {name: pool.get_stats() 
                   for name, pool in self._pools.items()}
    
    def remove_pool(self, name: str) -> bool:
        """移除对象池
        
        Args:
            name: 池名称
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if name in self._pools:
                pool = self._pools.pop(name)
                pool.clear()
                logger.debug("移除对象池'%s'", name)
                return True
            return False