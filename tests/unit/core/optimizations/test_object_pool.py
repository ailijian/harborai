#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对象池模块测试

测试ObjectPool和ObjectPoolManager的功能。
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch
from typing import Any

from harborai.core.optimizations.object_pool import ObjectPool, ObjectPoolManager


class TestObject:
    """测试用对象"""
    
    def __init__(self, value: int = 0):
        self.value = value
        self.reset_called = False
        self.cleanup_called = False
    
    def reset(self):
        """重置对象"""
        self.value = 0
        self.reset_called = True
    
    def cleanup(self):
        """清理对象"""
        self.cleanup_called = True


class TestObjectPool:
    """ObjectPool测试类"""
    
    def test_init_basic(self):
        """测试基本初始化"""
        pool = ObjectPool(TestObject, max_size=10)
        
        assert pool._object_type == TestObject
        assert pool._max_size == 10
        assert len(pool._pool) == 0
        assert len(pool._active_object_ids) == 0
        assert pool._created_count == 0
        assert pool._acquired_count == 0
        assert pool._released_count == 0
        assert pool._reused_count == 0
    
    def test_init_with_functions(self):
        """测试带函数的初始化"""
        factory_func = lambda: TestObject(42)
        reset_func = lambda obj: obj.reset()
        cleanup_func = lambda obj: obj.cleanup()
        
        pool = ObjectPool(
            TestObject,
            max_size=5,
            factory_func=factory_func,
            reset_func=reset_func,
            cleanup_func=cleanup_func
        )
        
        assert pool._max_size == 5
        assert pool._factory_func == factory_func
        assert pool._reset_func == reset_func
        assert pool._cleanup_func == cleanup_func
    
    def test_acquire_new_object(self):
        """测试获取新对象"""
        pool = ObjectPool(TestObject)
        
        obj = pool.acquire()
        
        assert isinstance(obj, TestObject)
        assert obj.value == 0
        assert pool._created_count == 1
        assert pool._acquired_count == 1
        assert pool._reused_count == 0
        assert len(pool._active_object_ids) == 1
        assert id(obj) in pool._active_object_ids
    
    def test_acquire_with_factory_func(self):
        """测试使用工厂函数获取对象"""
        factory_func = lambda: TestObject(99)
        pool = ObjectPool(TestObject, factory_func=factory_func)
        
        obj = pool.acquire()
        
        assert obj.value == 99
        assert pool._created_count == 1
    
    def test_release_object(self):
        """测试释放对象"""
        pool = ObjectPool(TestObject)
        obj = pool.acquire()
        
        pool.release(obj)
        
        assert pool._released_count == 1
        assert len(pool._pool) == 1
        assert len(pool._active_object_ids) == 0
        assert id(obj) not in pool._active_object_ids
    
    def test_release_with_reset_func(self):
        """测试带重置函数的释放"""
        reset_func = lambda obj: obj.reset()
        pool = ObjectPool(TestObject, reset_func=reset_func)
        obj = pool.acquire()
        obj.value = 123
        
        pool.release(obj)
        
        assert obj.reset_called
        assert obj.value == 0
        assert len(pool._pool) == 1
    
    def test_release_reset_failure(self):
        """测试重置失败的情况"""
        def failing_reset(obj):
            raise ValueError("Reset failed")
        
        pool = ObjectPool(TestObject, reset_func=failing_reset)
        obj = pool.acquire()
        
        with patch('harborai.core.optimizations.object_pool.logger') as mock_logger:
            pool.release(obj)
            mock_logger.error.assert_called()
        
        # 重置失败，对象不应该放回池中
        assert len(pool._pool) == 0
        assert len(pool._active_object_ids) == 0
    
    def test_release_none_object(self):
        """测试释放None对象"""
        pool = ObjectPool(TestObject)
        
        pool.release(None)
        
        # 应该没有任何变化
        assert pool._released_count == 0
        assert len(pool._pool) == 0
    
    def test_release_foreign_object(self):
        """测试释放不属于池的对象"""
        pool = ObjectPool(TestObject)
        foreign_obj = TestObject()
        
        with patch('harborai.core.optimizations.object_pool.logger') as mock_logger:
            pool.release(foreign_obj)
            mock_logger.warning.assert_called()
        
        assert pool._released_count == 0
        assert len(pool._pool) == 0
    
    def test_acquire_reuse_object(self):
        """测试对象复用"""
        pool = ObjectPool(TestObject)
        
        # 获取并释放对象
        obj1 = pool.acquire()
        original_id = id(obj1)
        pool.release(obj1)
        
        # 再次获取应该复用同一个对象
        obj2 = pool.acquire()
        
        assert id(obj2) == original_id
        assert pool._created_count == 1
        assert pool._acquired_count == 2
        assert pool._reused_count == 1
    
    def test_pool_max_size_limit(self):
        """测试池大小限制"""
        pool = ObjectPool(TestObject, max_size=2)
        
        # 创建3个对象
        obj1 = pool.acquire()
        obj2 = pool.acquire()
        obj3 = pool.acquire()
        
        # 释放所有对象
        pool.release(obj1)
        pool.release(obj2)
        pool.release(obj3)  # 这个应该被丢弃，不计入released_count
        
        assert len(pool._pool) == 2  # 只保留2个
        assert pool._released_count == 2  # 只有2个被实际释放到池中
    
    def test_pool_max_size_with_cleanup(self):
        """测试池大小限制时的清理"""
        cleanup_func = lambda obj: obj.cleanup()
        pool = ObjectPool(TestObject, max_size=1, cleanup_func=cleanup_func)
        
        obj1 = pool.acquire()
        obj2 = pool.acquire()
        
        pool.release(obj1)
        pool.release(obj2)  # 应该被清理
        
        assert len(pool._pool) == 1
        assert obj2.cleanup_called
    
    def test_cleanup_failure(self):
        """测试清理失败的情况"""
        def failing_cleanup(obj):
            raise ValueError("Cleanup failed")
        
        pool = ObjectPool(TestObject, max_size=1, cleanup_func=failing_cleanup)
        obj1 = pool.acquire()
        obj2 = pool.acquire()
        
        pool.release(obj1)
        
        with patch('harborai.core.optimizations.object_pool.logger') as mock_logger:
            pool.release(obj2)
            mock_logger.error.assert_called()
    
    def test_clear_pool(self):
        """测试清空池"""
        pool = ObjectPool(TestObject)
        
        # 添加一些对象到池中
        obj1 = pool.acquire()
        obj2 = pool.acquire()
        pool.release(obj1)
        pool.release(obj2)
        
        assert len(pool._pool) == 2
        
        pool.clear()
        
        assert len(pool._pool) == 0
    
    def test_clear_with_cleanup(self):
        """测试带清理的清空池"""
        cleanup_func = lambda obj: obj.cleanup()
        pool = ObjectPool(TestObject, cleanup_func=cleanup_func)
        
        obj = pool.acquire()
        pool.release(obj)
        
        pool.clear()
        
        assert obj.cleanup_called
        assert len(pool._pool) == 0
    
    def test_clear_cleanup_failure(self):
        """测试清空时清理失败"""
        def failing_cleanup(obj):
            raise ValueError("Cleanup failed")
        
        pool = ObjectPool(TestObject, cleanup_func=failing_cleanup)
        obj = pool.acquire()
        pool.release(obj)
        
        with patch('harborai.core.optimizations.object_pool.logger') as mock_logger:
            pool.clear()
            mock_logger.error.assert_called()
    
    def test_size_and_active_count(self):
        """测试大小和活跃计数"""
        pool = ObjectPool(TestObject)
        
        assert pool.size() == 0
        assert pool.active_count() == 0
        
        obj1 = pool.acquire()
        obj2 = pool.acquire()
        
        assert pool.size() == 0
        assert pool.active_count() == 2
        
        pool.release(obj1)
        
        assert pool.size() == 1
        assert pool.active_count() == 1
        
        pool.release(obj2)
        
        assert pool.size() == 2
        assert pool.active_count() == 0
    
    def test_get_stats(self):
        """测试获取统计信息"""
        pool = ObjectPool(TestObject, max_size=10)
        
        # 初始统计
        stats = pool.get_stats()
        assert stats['object_type'] == 'TestObject'
        assert stats['max_size'] == 10
        assert stats['pool_size'] == 0
        assert stats['active_count'] == 0
        assert stats['created_count'] == 0
        assert stats['acquired_count'] == 0
        assert stats['released_count'] == 0
        assert stats['reused_count'] == 0
        assert stats['reuse_rate'] == 0.0
        
        # 使用后的统计
        obj1 = pool.acquire()
        obj2 = pool.acquire()
        pool.release(obj1)
        obj3 = pool.acquire()  # 复用obj1
        
        stats = pool.get_stats()
        assert stats['created_count'] == 2
        assert stats['acquired_count'] == 3
        assert stats['released_count'] == 1
        assert stats['reused_count'] == 1
        assert stats['reuse_rate'] == 1/3
    
    def test_shrink_pool(self):
        """测试收缩池"""
        pool = ObjectPool(TestObject)
        
        # 添加4个对象到池中
        objects = [pool.acquire() for _ in range(4)]
        for obj in objects:
            pool.release(obj)
        
        assert len(pool._pool) == 4
        
        # 收缩到2个
        removed = pool.shrink(2)
        
        assert removed == 2
        assert len(pool._pool) == 2
    
    def test_shrink_pool_default(self):
        """测试默认收缩池（收缩到一半）"""
        pool = ObjectPool(TestObject)
        
        # 添加4个对象到池中
        objects = [pool.acquire() for _ in range(4)]
        for obj in objects:
            pool.release(obj)
        
        assert len(pool._pool) == 4
        
        # 默认收缩到一半
        removed = pool.shrink()
        
        assert removed == 2
        assert len(pool._pool) == 2
    
    def test_shrink_with_cleanup(self):
        """测试带清理的收缩"""
        cleanup_func = lambda obj: obj.cleanup()
        pool = ObjectPool(TestObject, cleanup_func=cleanup_func)
        
        objects = [pool.acquire() for _ in range(3)]
        for obj in objects:
            pool.release(obj)
        
        removed = pool.shrink(1)
        
        assert removed == 2
        assert len(pool._pool) == 1
        # 检查被移除的对象是否被清理
        assert sum(obj.cleanup_called for obj in objects) >= 2
    
    def test_len_method(self):
        """测试__len__方法"""
        pool = ObjectPool(TestObject)
        
        assert len(pool) == 0
        
        obj = pool.acquire()
        pool.release(obj)
        
        assert len(pool) == 1
    
    def test_del_method(self):
        """测试析构方法"""
        pool = ObjectPool(TestObject)
        obj = pool.acquire()
        pool.release(obj)
        
        # 模拟析构
        with patch.object(pool, 'clear') as mock_clear:
            pool.__del__()
            mock_clear.assert_called_once()
    
    def test_thread_safety(self):
        """测试线程安全"""
        pool = ObjectPool(TestObject, max_size=10)
        results = []
        errors = []
        
        def worker():
            try:
                for _ in range(10):
                    obj = pool.acquire()
                    time.sleep(0.001)  # 模拟使用
                    pool.release(obj)
                    results.append(1)
            except Exception as e:
                errors.append(e)
        
        # 启动多个线程
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 50
        assert pool.active_count() == 0


class TestObjectPoolManager:
    """ObjectPoolManager测试类"""
    
    def test_init(self):
        """测试初始化"""
        manager = ObjectPoolManager()
        
        assert len(manager._pools) == 0
    
    def test_create_pool(self):
        """测试创建池"""
        manager = ObjectPoolManager()
        
        pool = manager.create_pool("test_pool", TestObject, max_size=5)
        
        assert isinstance(pool, ObjectPool)
        assert pool._max_size == 5
        assert "test_pool" in manager._pools
        assert manager._pools["test_pool"] is pool
    
    def test_create_duplicate_pool(self):
        """测试创建重复池"""
        manager = ObjectPoolManager()
        manager.create_pool("test_pool", TestObject)
        
        with pytest.raises(ValueError, match="对象池'test_pool'已存在"):
            manager.create_pool("test_pool", TestObject)
    
    def test_get_pool(self):
        """测试获取池"""
        manager = ObjectPoolManager()
        pool = manager.create_pool("test_pool", TestObject)
        
        retrieved_pool = manager.get_pool("test_pool")
        
        assert retrieved_pool is pool
    
    def test_get_nonexistent_pool(self):
        """测试获取不存在的池"""
        manager = ObjectPoolManager()
        
        pool = manager.get_pool("nonexistent")
        
        assert pool is None
    
    def test_acquire_object(self):
        """测试获取对象"""
        manager = ObjectPoolManager()
        manager.create_pool("test_pool", TestObject)
        
        obj = manager.acquire_object("test_pool")
        
        assert isinstance(obj, TestObject)
    
    def test_acquire_object_nonexistent_pool(self):
        """测试从不存在的池获取对象"""
        manager = ObjectPoolManager()
        
        obj = manager.acquire_object("nonexistent")
        
        assert obj is None
    
    def test_acquire_object_with_error(self):
        """测试获取对象时发生错误"""
        manager = ObjectPoolManager()
        pool = manager.create_pool("test_pool", TestObject)
        
        with patch.object(pool, 'acquire', side_effect=Exception("Test error")):
            with patch('harborai.core.optimizations.object_pool.logger') as mock_logger:
                obj = manager.acquire_object("test_pool")
                assert obj is None
                mock_logger.error.assert_called()
    
    def test_release_object(self):
        """测试释放对象"""
        manager = ObjectPoolManager()
        manager.create_pool("test_pool", TestObject)
        obj = manager.acquire_object("test_pool")
        
        result = manager.release_object("test_pool", obj)
        
        assert result is True
    
    def test_release_object_nonexistent_pool(self):
        """测试释放对象到不存在的池"""
        manager = ObjectPoolManager()
        obj = TestObject()
        
        result = manager.release_object("nonexistent", obj)
        
        assert result is False
    
    def test_clear_all(self):
        """测试清空所有池"""
        manager = ObjectPoolManager()
        pool1 = manager.create_pool("pool1", TestObject)
        pool2 = manager.create_pool("pool2", TestObject)
        
        # 添加一些对象
        obj1 = pool1.acquire()
        obj2 = pool2.acquire()
        pool1.release(obj1)
        pool2.release(obj2)
        
        assert pool1.size() == 1
        assert pool2.size() == 1
        
        manager.clear_all()
        
        assert pool1.size() == 0
        assert pool2.size() == 0
    
    def test_get_all_stats(self):
        """测试获取所有统计信息"""
        manager = ObjectPoolManager()
        manager.create_pool("pool1", TestObject)
        manager.create_pool("pool2", TestObject)
        
        stats = manager.get_all_stats()
        
        assert "pool1" in stats
        assert "pool2" in stats
        assert stats["pool1"]["object_type"] == "TestObject"
        assert stats["pool2"]["object_type"] == "TestObject"
    
    def test_remove_pool(self):
        """测试移除池"""
        manager = ObjectPoolManager()
        pool = manager.create_pool("test_pool", TestObject)
        obj = pool.acquire()
        pool.release(obj)
        
        assert pool.size() == 1
        
        result = manager.remove_pool("test_pool")
        
        assert result is True
        assert "test_pool" not in manager._pools
        assert pool.size() == 0  # 应该被清空
    
    def test_remove_nonexistent_pool(self):
        """测试移除不存在的池"""
        manager = ObjectPoolManager()
        
        result = manager.remove_pool("nonexistent")
        
        assert result is False