#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化缓存模块测试

测试MemoryOptimizedCache类的各种功能，包括：
1. 基本缓存操作（get/set/delete）
2. LRU淘汰策略
3. TTL过期机制
4. 定期清理功能
5. 弱引用支持
6. 线程安全性
7. 内存统计
"""

import pytest
import time
import threading
import weakref
from unittest.mock import patch, MagicMock
from collections import OrderedDict

from harborai.core.optimizations.memory_optimized_cache import MemoryOptimizedCache


class TestMemoryOptimizedCache:
    """内存优化缓存测试类"""
    
    def test_init_default_parameters(self):
        """测试默认参数初始化"""
        cache = MemoryOptimizedCache()
        
        assert cache._max_size == 1000
        assert cache._ttl_seconds is None
        assert cache._cleanup_interval == 300.0
        assert cache._enable_weak_refs is True
        assert isinstance(cache._cache, OrderedDict)
        assert isinstance(cache._access_times, dict)
        assert isinstance(cache._creation_times, dict)
        assert hasattr(cache, '_weak_refs')
        assert cache._hits == 0
        assert cache._misses == 0
        assert cache._evictions == 0
    
    def test_init_custom_parameters(self):
        """测试自定义参数初始化"""
        cache = MemoryOptimizedCache(
            max_size=500,
            ttl_seconds=60.0,
            cleanup_interval=120.0,
            enable_weak_refs=False
        )
        
        assert cache._max_size == 500
        assert cache._ttl_seconds == 60.0
        assert cache._cleanup_interval == 120.0
        assert cache._enable_weak_refs is False
        assert not hasattr(cache, '_weak_refs')
    
    def test_basic_set_and_get(self):
        """测试基本的设置和获取操作"""
        cache = MemoryOptimizedCache()
        
        # 设置值
        cache.set("key1", "value1")
        cache.set("key2", {"data": "value2"})
        
        # 获取值
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == {"data": "value2"}
        assert cache.get("nonexistent") is None
        
        # 检查统计
        assert cache._hits == 2
        assert cache._misses == 1
    
    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        cache = MemoryOptimizedCache()
        
        # 获取不存在的键应该返回None
        result = cache.get("nonexistent")
        assert result is None
        assert cache._misses == 1
    
    def test_contains_operation(self):
        """测试包含操作"""
        cache = MemoryOptimizedCache()
        
        cache.set("key1", "value1")
        
        assert "key1" in cache
        assert "nonexistent" not in cache
    
    def test_delete_operation(self):
        """测试删除操作"""
        cache = MemoryOptimizedCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 删除存在的键
        result = cache.delete("key1")
        assert result is True
        assert "key1" not in cache
        assert cache.get("key1") is None
        
        # 删除不存在的键
        result = cache.delete("nonexistent")
        assert result is False
    
    def test_lru_eviction_policy(self):
        """测试LRU淘汰策略"""
        cache = MemoryOptimizedCache(max_size=3)
        
        # 添加项目直到达到最大容量
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        assert len(cache._cache) == 3
        assert cache._evictions == 0
        
        # 添加第四个项目，应该淘汰最旧的
        cache.set("key4", "value4")
        
        assert len(cache._cache) == 3
        assert cache._evictions == 1
        assert "key1" not in cache  # 最旧的应该被淘汰
        assert "key2" in cache
        assert "key3" in cache
        assert "key4" in cache
    
    def test_lru_access_updates_order(self):
        """测试访问更新LRU顺序"""
        cache = MemoryOptimizedCache(max_size=3)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # 访问key1，使其成为最近使用的
        cache.get("key1")
        
        # 添加新项目，key2应该被淘汰（因为key1被访问过）
        cache.set("key4", "value4")
        
        assert "key1" in cache  # 被访问过，不应该被淘汰
        assert "key2" not in cache  # 最旧且未被访问，应该被淘汰
        assert "key3" in cache
        assert "key4" in cache
    
    def test_ttl_expiration(self):
        """测试TTL过期机制"""
        cache = MemoryOptimizedCache(ttl_seconds=0.1)  # 100ms TTL
        
        cache.set("key1", "value1")
        
        # 立即获取应该成功
        assert cache.get("key1") == "value1"
        
        # 等待过期
        time.sleep(0.15)
        
        # 过期后获取应该返回None
        assert cache.get("key1") is None
        assert cache._misses == 1
    
    def test_clear_operation(self):
        """测试清空操作"""
        cache = MemoryOptimizedCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert len(cache._cache) == 2
        
        cache.clear()
        
        assert len(cache._cache) == 0
        assert len(cache._access_times) == 0
        assert len(cache._creation_times) == 0
        assert cache.get("key1") is None
    
    def test_size_and_len(self):
        """测试大小获取"""
        cache = MemoryOptimizedCache()
        
        assert cache.size() == 0
        assert len(cache) == 0
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        assert cache.size() == 2
        assert len(cache) == 2
    
    def test_cache_internal_structure(self):
        """测试缓存内部结构"""
        cache = MemoryOptimizedCache()
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 检查内部数据结构
        assert "key1" in cache._cache
        assert "key2" in cache._cache
        assert cache._cache["key1"] == "value1"
        assert cache._cache["key2"] == "value2"
    
    def test_statistics(self):
        """测试统计信息"""
        cache = MemoryOptimizedCache()
        
        # 初始统计
        stats = cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['evictions'] == 0
        assert stats['size'] == 0
        assert stats['hit_rate'] == 0.0
        
        # 添加一些操作
        cache.set("key1", "value1")
        cache.get("key1")  # hit
        cache.get("key2")  # miss
        
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['size'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_cleanup_expired_items(self):
        """测试清理过期项目"""
        cache = MemoryOptimizedCache(ttl_seconds=0.1)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 等待过期
        time.sleep(0.15)
        
        # 手动清理
        cleaned = cache.cleanup_expired()
        
        assert cleaned == 2
        assert len(cache._cache) == 0
    
    def test_weak_references_enabled(self):
        """测试启用弱引用"""
        cache = MemoryOptimizedCache(enable_weak_refs=True)
        
        # 创建一个支持弱引用的对象
        class TestObj:
            def __init__(self, data):
                self.data = data
        
        obj = TestObj("test")
        cache.set("key1", obj)
        
        # 检查弱引用是否被创建（如果对象支持弱引用）
        assert cache.get("key1") == obj
        
        # 删除原始引用
        del obj
        
        # 缓存中的对象应该仍然可以访问
        cached_obj = cache.get("key1")
        assert cached_obj is not None
    
    def test_weak_references_disabled(self):
        """测试禁用弱引用"""
        cache = MemoryOptimizedCache(enable_weak_refs=False)
        
        obj = {"data": "test"}
        cache.set("key1", obj)
        
        assert not hasattr(cache, '_weak_refs')
        assert cache.get("key1") == obj
    
    def test_thread_safety(self):
        """测试线程安全性"""
        cache = MemoryOptimizedCache(max_size=100)
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(50):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"thread_{thread_id}_value_{i}"
                    cache.set(key, value)
                    retrieved = cache.get(key)
                    if retrieved == value:
                        results.append(True)
                    else:
                        results.append(False)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证没有错误且所有操作成功
        assert len(errors) == 0
        assert all(results)
        assert len(results) == 250  # 5 threads * 50 operations
    
    def test_memory_pressure_handling(self):
        """测试内存压力处理"""
        cache = MemoryOptimizedCache(max_size=5)
        
        # 添加超过最大容量的项目
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")
        
        # 缓存大小应该保持在最大容量
        assert len(cache._cache) == 5
        assert cache._evictions == 5
        
        # 最新的项目应该仍在缓存中
        for i in range(5, 10):
            assert cache.get(f"key_{i}") == f"value_{i}"
    
    def test_update_existing_key(self):
        """测试更新现有键"""
        cache = MemoryOptimizedCache()
        
        cache.set("key1", "value1")
        original_time = cache._creation_times["key1"]
        
        # 稍等一下确保时间戳不同
        time.sleep(0.01)
        
        # 更新值
        cache.set("key1", "new_value1")
        
        assert cache.get("key1") == "new_value1"
        # 创建时间应该更新
        assert cache._creation_times["key1"] > original_time
    
    @patch('time.time')
    def test_cleanup_with_mocked_time(self, mock_time):
        """测试使用模拟时间的清理功能"""
        # 设置初始时间
        mock_time.return_value = 1000.0
        
        cache = MemoryOptimizedCache(ttl_seconds=60.0)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 模拟时间前进，使项目过期
        mock_time.return_value = 1070.0  # 70秒后
        
        cleaned = cache.cleanup_expired()
        
        assert cleaned == 2
        assert len(cache._cache) == 0
    
    def test_edge_cases(self):
        """测试边界情况"""
        cache = MemoryOptimizedCache(max_size=1)
        
        # 测试空键
        cache.set("", "empty_key_value")
        assert cache.get("") == "empty_key_value"
        
        # 测试None值
        cache.set("none_key", None)
        assert cache.get("none_key") is None
        assert "none_key" in cache
        
        # 测试大对象
        large_obj = {"data": "x" * 1000}
        cache.set("large", large_obj)
        assert cache.get("large") == large_obj
    
    def test_cleanup_timer_functionality(self):
        """测试清理定时器功能"""
        # 测试禁用定时器
        cache = MemoryOptimizedCache(cleanup_interval=0)
        assert cache._cleanup_timer is None
        
        # 测试启用定时器
        cache2 = MemoryOptimizedCache(cleanup_interval=1.0)
        assert cache2._cleanup_timer is not None
        
        # 清理
        cache2.__del__()
    
    def test_no_ttl_cleanup(self):
        """测试没有TTL时的清理"""
        cache = MemoryOptimizedCache(ttl_seconds=None)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 没有TTL时清理应该返回0
        cleaned = cache.cleanup_expired()
        assert cleaned == 0
        assert len(cache._cache) == 2
    
    def test_weak_ref_callback(self):
        """测试弱引用回调"""
        cache = MemoryOptimizedCache(enable_weak_refs=True)
        
        class TestObj:
            def __init__(self, data):
                self.data = data
        
        obj = TestObj("test")
        cache.set("key1", obj)
        
        # 手动调用弱引用回调
        cache._on_weak_ref_deleted("key1")
        
        # 键应该被移除
        assert "key1" not in cache._cache
    
    def test_periodic_cleanup_error_handling(self):
        """测试定期清理的错误处理"""
        cache = MemoryOptimizedCache(cleanup_interval=0.1)
        
        # 模拟清理过程中的错误
        with patch.object(cache, 'cleanup_expired', side_effect=Exception("Test error")):
            # 调用定期清理
            cache._periodic_cleanup()
            
        # 缓存应该仍然可用
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # 清理
        cache.__del__()
    
    def test_weak_ref_unsupported_object(self):
        """测试不支持弱引用的对象"""
        cache = MemoryOptimizedCache(enable_weak_refs=True)
        
        # 某些内置类型不支持弱引用
        cache.set("int_key", 42)
        cache.set("str_key", "string")
        
        assert cache.get("int_key") == 42
        assert cache.get("str_key") == "string"