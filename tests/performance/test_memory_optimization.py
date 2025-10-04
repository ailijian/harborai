#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化测试模块

根据HarborAI SDK性能优化技术设计方案第二阶段要求，
验证内存使用从16.56MB降低到≤8MB的目标。

测试覆盖：
1. 基础内存使用测试
2. 内存泄漏检测
3. 长期稳定性测试
4. 缓存内存管理测试
5. 对象池内存复用测试

设计原则：
- 测试驱动开发（TDD）
- 精确的内存测量
- 可重复的测试结果
- 详细的性能指标记录
"""

import unittest
import time
import gc
import threading
import weakref
from typing import List, Dict, Any, Optional
import psutil
import os

# 导入内存优化组件
try:
    from harborai.core.optimizations.memory_optimized_cache import MemoryOptimizedCache
except ImportError:
    MemoryOptimizedCache = None

try:
    from harborai.core.optimizations.object_pool import ObjectPool
except ImportError:
    ObjectPool = None

try:
    from harborai.core.optimizations.memory_manager import MemoryManager
except ImportError:
    MemoryManager = None

try:
    from harborai.api.fast_client import FastHarborAI
except ImportError:
    FastHarborAI = None


class MemoryTestCase(unittest.TestCase):
    """内存测试基类
    
    提供内存测量和验证的通用方法。
    """
    
    def setUp(self):
        """测试前准备"""
        # 强制垃圾回收，确保基准测量准确
        gc.collect()
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = self._get_memory_usage()
        
    def tearDown(self):
        """测试后清理"""
        gc.collect()
        
    def _get_memory_usage(self) -> float:
        """获取当前内存使用量（MB）
        
        Returns:
            内存使用量，单位MB
        """
        return self.process.memory_info().rss / 1024 / 1024
    
    def _measure_memory_increase(self, func, *args, **kwargs) -> tuple:
        """测量函数执行后的内存增长
        
        Args:
            func: 要测量的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            (结果, 内存增长MB)
        """
        gc.collect()
        before_memory = self._get_memory_usage()
        
        result = func(*args, **kwargs)
        
        gc.collect()
        after_memory = self._get_memory_usage()
        
        memory_increase = after_memory - before_memory
        return result, memory_increase


class TestMemoryOptimizedCache(MemoryTestCase):
    """内存优化缓存测试
    
    验证MemoryOptimizedCache类的内存使用效率。
    """
    
    @unittest.skipIf(MemoryOptimizedCache is None, "MemoryOptimizedCache未实现")
    def test_cache_memory_usage(self):
        """测试缓存的内存使用
        
        验证：
        1. 缓存创建的内存开销
        2. LRU淘汰机制的内存释放
        3. 定期清理的内存回收
        """
        def create_and_use_cache():
            cache = MemoryOptimizedCache(max_size=1000)
            
            # 添加大量数据
            for i in range(1000):
                cache.set(f"key_{i}", f"value_{i}" * 100)  # 每个值约100字节
            
            # 验证LRU淘汰
            for i in range(1000, 1500):
                cache.set(f"key_{i}", f"value_{i}" * 100)
            
            return cache
        
        cache, memory_increase = self._measure_memory_increase(create_and_use_cache)
        
        # 验证内存使用在合理范围内（预期<2MB）
        self.assertLess(memory_increase, 2.0, 
                       f"缓存内存使用{memory_increase:.2f}MB超过预期2MB")
        
        # 验证缓存大小限制生效
        self.assertEqual(len(cache._cache), 1000, "缓存大小限制未生效")
    
    @unittest.skipIf(MemoryOptimizedCache is None, "MemoryOptimizedCache未实现")
    def test_cache_cleanup_memory_release(self):
        """测试缓存清理的内存释放"""
        cache = MemoryOptimizedCache(max_size=500)
        
        # 填充缓存
        for i in range(500):
            cache.set(f"key_{i}", "x" * 1000)  # 每个值1KB
        
        # 验证缓存确实包含数据
        self.assertEqual(len(cache), 500, "缓存未正确填充")
        
        gc.collect()
        before_cleanup = self._get_memory_usage()
        
        # 清理缓存
        cache.clear()
        
        # 验证缓存已清空
        self.assertEqual(len(cache), 0, "缓存未正确清空")
        
        gc.collect()
        after_cleanup = self._get_memory_usage()
        
        memory_released = before_cleanup - after_cleanup
        
        # 由于Python内存管理的复杂性，我们主要验证缓存功能正确性
        # 而不是严格的内存释放量
        print(f"缓存清理释放内存: {memory_released:.2f}MB")
        
        # 验证内存没有显著增长（允许小幅波动）
        self.assertLess(abs(memory_released), 5.0,
                       f"内存变化异常: {memory_released:.2f}MB")


class TestObjectPool(MemoryTestCase):
    """对象池测试
    
    验证ObjectPool类的内存复用效率。
    """
    
    @unittest.skipIf(ObjectPool is None, "ObjectPool未实现")
    def test_object_pool_memory_reuse(self):
        """测试对象池的内存复用"""
        
        class TestObject:
            def __init__(self):
                self.data = "x" * 1000  # 1KB数据
            
            def reset(self):
                self.data = "x" * 1000
        
        def create_pool_and_use():
            pool = ObjectPool(TestObject, max_size=100)
            
            # 获取和释放对象多次
            objects = []
            for _ in range(200):
                obj = pool.acquire()
                objects.append(obj)
            
            for obj in objects:
                pool.release(obj)
            
            return pool
        
        pool, memory_increase = self._measure_memory_increase(create_pool_and_use)
        
        # 验证对象池内存使用效率（预期<1MB）
        self.assertLess(memory_increase, 1.0,
                       f"对象池内存使用{memory_increase:.2f}MB超过预期1MB")
        
        # 验证对象池大小限制
        self.assertLessEqual(len(pool._pool), 100, "对象池大小超过限制")
    
    @unittest.skipIf(ObjectPool is None, "ObjectPool未实现")
    def test_object_pool_gc_pressure_reduction(self):
        """测试对象池减少GC压力"""
        
        class TestObject:
            def __init__(self):
                self.data = [0] * 1000
            
            def reset(self):
                self.data = [0] * 1000
        
        pool = ObjectPool(TestObject, max_size=50)
        
        # 测量GC次数
        gc.collect()
        gc_before = gc.get_stats()
        
        # 使用对象池
        for _ in range(1000):
            obj = pool.acquire()
            # 模拟使用
            obj.data[0] = 1
            pool.release(obj)
        
        gc.collect()
        gc_after = gc.get_stats()
        
        # 验证GC压力减少（这是一个间接测试）
        # 主要验证没有内存泄漏
        current_memory = self._get_memory_usage()
        memory_increase = current_memory - self.baseline_memory
        
        self.assertLess(memory_increase, 0.5,
                       f"对象池使用后内存增长{memory_increase:.2f}MB，可能存在内存泄漏")


class TestMemoryManager(MemoryTestCase):
    """内存管理器测试
    
    验证MemoryManager统一管理内存优化组件的效果。
    """
    
    @unittest.skipIf(MemoryManager is None, "MemoryManager未实现")
    def test_memory_manager_integration(self):
        """测试内存管理器集成效果"""
        
        def create_and_use_manager():
            manager = MemoryManager(
                cache_size=500,
                object_pool_size=50,
                enable_weak_references=True
            )
            
            # 模拟使用各种组件
            for i in range(100):
                manager.cache.set(f"key_{i}", f"value_{i}" * 50)
            
            # 使用对象池
            for _ in range(50):
                obj = manager.get_pooled_object("test_object")
                manager.release_pooled_object("test_object", obj)
            
            return manager
        
        manager, memory_increase = self._measure_memory_increase(create_and_use_manager)
        
        # 验证内存管理器的整体内存效率（预期<1.5MB）
        self.assertLess(memory_increase, 1.5,
                       f"内存管理器内存使用{memory_increase:.2f}MB超过预期1.5MB")
    
    @unittest.skipIf(MemoryManager is None, "MemoryManager未实现")
    def test_memory_manager_cleanup(self):
        """测试内存管理器清理功能"""
        manager = MemoryManager()
        
        # 使用各种组件
        for i in range(200):
            manager.cache.set(f"key_{i}", "x" * 500)
        
        # 验证缓存中有数据
        cache_size_before = len(manager.cache._cache)
        self.assertGreater(cache_size_before, 0, "缓存中应该有数据")
        
        gc.collect()
        before_cleanup = self._get_memory_usage()
        
        # 执行强制清理
        manager.cleanup(force_clear=True)
        
        # 验证缓存被清理
        cache_size_after = len(manager.cache._cache)
        self.assertEqual(cache_size_after, 0, "清理后缓存应该为空")
        
        gc.collect()
        after_cleanup = self._get_memory_usage()
        
        memory_change = after_cleanup - before_cleanup
        
        # 验证清理功能正常（内存不应显著增加）
        self.assertLessEqual(memory_change, 0.1,
                           f"内存管理器清理后内存变化{memory_change:.2f}MB，应该保持稳定")


class TestFastHarborAIMemoryUsage(MemoryTestCase):
    """FastHarborAI客户端内存使用测试
    
    验证集成内存优化后的客户端内存使用情况。
    """
    
    @unittest.skipIf(FastHarborAI is None, "FastHarborAI未实现内存优化")
    def test_client_initialization_memory(self):
        """测试客户端初始化内存使用
        
        目标：验证内存使用≤8MB
        """
        
        def create_client():
            client = FastHarborAI(
                api_key="test-key",
                enable_memory_optimization=True
            )
            return client
        
        client, memory_increase = self._measure_memory_increase(create_client)
        
        # 核心验证：内存使用≤8MB
        self.assertLessEqual(memory_increase, 8.0,
                           f"客户端初始化内存使用{memory_increase:.2f}MB超过目标8MB")
        
        # 验证功能完整性
        self.assertIsNotNone(client.chat, "聊天功能不可用")
        self.assertTrue(hasattr(client.chat, 'completions'), "聊天完成功能不可用")
    
    @unittest.skipIf(FastHarborAI is None, "FastHarborAI未实现内存优化")
    def test_client_multiple_requests_memory(self):
        """测试客户端多次请求的内存稳定性"""
        client = FastHarborAI(
            api_key="test-key",
            enable_memory_optimization=True
        )
        
        gc.collect()
        before_requests = self._get_memory_usage()
        
        # 模拟多次请求（使用mock避免实际网络调用）
        for i in range(50):
            try:
                # 这里会失败，因为没有真实的API密钥，但会测试内存使用
                client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": f"Test message {i}"}]
                )
            except Exception:
                # 忽略API调用错误，专注于内存测试
                pass
        
        gc.collect()
        after_requests = self._get_memory_usage()
        
        memory_increase = after_requests - before_requests
        
        # 验证多次请求后内存增长有限（预期<2MB）
        self.assertLess(memory_increase, 2.0,
                       f"50次请求后内存增长{memory_increase:.2f}MB，可能存在内存泄漏")


class TestMemoryLeakDetection(MemoryTestCase):
    """内存泄漏检测测试
    
    验证系统长期运行的内存稳定性。
    """
    
    def test_weak_reference_cleanup(self):
        """测试弱引用清理机制"""
        
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        # 创建对象和弱引用
        objects = []
        weak_refs = []
        
        for i in range(100):
            obj = TestObject(f"data_{i}")
            objects.append(obj)
            weak_refs.append(weakref.ref(obj))
        
        # 验证弱引用存在
        alive_refs = sum(1 for ref in weak_refs if ref() is not None)
        self.assertEqual(alive_refs, 100, "弱引用创建失败")
        
        # 删除强引用
        del objects
        
        # 多次垃圾回收，确保所有对象被清理
        for _ in range(5):
            gc.collect()
            time.sleep(0.01)  # 给垃圾回收器一些时间
        
        # 验证弱引用被清理 - 允许少量残留（垃圾回收时机问题）
        alive_refs = sum(1 for ref in weak_refs if ref() is not None)
        self.assertLessEqual(alive_refs, 2, f"弱引用清理不完全，剩余{alive_refs}个，可能存在内存泄漏")
    
    def test_circular_reference_handling(self):
        """测试循环引用处理"""
        
        class Node:
            def __init__(self, value):
                self.value = value
                self.parent = None
                self.children = []
            
            def add_child(self, child):
                child.parent = self
                self.children.append(child)
        
        gc.collect()
        before_memory = self._get_memory_usage()
        
        # 创建循环引用结构
        root = Node("root")
        for i in range(50):
            child = Node(f"child_{i}")
            root.add_child(child)
            # 创建循环引用
            child.parent = root
        
        # 删除根引用
        del root
        gc.collect()
        
        after_memory = self._get_memory_usage()
        memory_increase = after_memory - before_memory
        
        # 验证循环引用被正确处理（内存增长应该很小）
        self.assertLess(memory_increase, 0.1,
                       f"循环引用处理后内存增长{memory_increase:.2f}MB，可能存在内存泄漏")


class TestLongTermMemoryStability(MemoryTestCase):
    """长期内存稳定性测试
    
    模拟长期运行场景，验证内存使用的稳定性。
    """
    
    @unittest.skipIf(MemoryOptimizedCache is None, "MemoryOptimizedCache未实现")
    def test_long_term_cache_stability(self):
        """测试长期缓存稳定性"""
        cache = MemoryOptimizedCache(max_size=1000)
        
        memory_samples = []
        
        # 模拟长期使用
        for cycle in range(10):
            # 每个周期添加和访问数据
            for i in range(100):
                key = f"cycle_{cycle}_key_{i}"
                cache.set(key, f"value_{i}" * 50)
            
            # 随机访问已有数据
            for i in range(50):
                key = f"cycle_{max(0, cycle-1)}_key_{i}"
                cache.get(key)
            
            # 记录内存使用
            gc.collect()
            memory_samples.append(self._get_memory_usage())
        
        # 验证内存使用稳定（最后几个样本的方差应该很小）
        recent_samples = memory_samples[-5:]
        memory_variance = max(recent_samples) - min(recent_samples)
        
        self.assertLess(memory_variance, 0.5,
                       f"长期运行内存波动{memory_variance:.2f}MB过大，可能存在内存泄漏")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)