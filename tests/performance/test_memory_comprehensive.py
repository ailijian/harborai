#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合内存测试模块

合并所有内存相关测试，包括：
1. 内存使用基准测试
2. 内存泄漏检测
3. 内存优化效果验证
4. 长期稳定性测试
5. 缓存内存管理测试
6. 对象池内存复用测试

根据HarborAI SDK性能优化技术设计方案第二阶段要求，
验证内存使用从16.56MB降低到≤8MB的目标。

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
import psutil
import os
import pytest
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch

# 尝试导入优化相关模块
try:
    from harborai.api.fast_client import FastHarborAI
except ImportError:
    FastHarborAI = None

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


class MemoryBenchmark:
    """内存使用基准测试类"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.baseline_memory = None
        self.target_memory_mb = 8.0  # 目标内存使用量
        self.original_memory_mb = 16.56  # 原始内存使用量
    
    def get_memory_usage_mb(self) -> float:
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def measure_baseline(self):
        """测量基线内存使用"""
        gc.collect()
        time.sleep(0.1)
        self.baseline_memory = self.get_memory_usage_mb()
        return self.baseline_memory
    
    def measure_with_optimization(self) -> Dict[str, Any]:
        """测量启用内存优化后的内存使用"""
        if not FastHarborAI:
            pytest.skip("FastHarborAI 不可用")
            
        # 强制垃圾回收
        gc.collect()
        
        # 记录初始内存
        initial_memory = self.get_memory_usage_mb()
        
        # 创建启用内存优化的客户端
        config = {
            "memory_optimization": {
                "cache_size": 1000,
                "object_pool_size": 100,
                "enable_weak_references": True,
                "auto_cleanup_interval": 1.0,
                "memory_threshold_mb": 50.0
            }
        }
        
        client = FastHarborAI(config=config, enable_memory_optimization=True)
        
        try:
            # 执行一些典型操作来模拟实际使用
            for i in range(100):
                # 缓存操作
                cache_key = f"benchmark_key_{i}"
                if hasattr(client, '_memory_manager'):
                    client._memory_manager.cache.set(cache_key, f"benchmark_value_{i}")
                    client._memory_manager.cache.get(cache_key)
                    
                    # 对象池操作
                    obj = client._memory_manager.get_pooled_object('default')
                    if obj:
                        client._memory_manager.release_pooled_object('default', obj)
                    
                    # 弱引用操作
                    test_obj = {"benchmark_data": f"data_{i}"}
                    ref_key = f"benchmark_ref_{i}"
                    client._memory_manager.add_weak_reference(ref_key, test_obj)
            
            # 等待一段时间让系统稳定
            time.sleep(0.5)
            
            # 记录使用后的内存
            after_usage_memory = self.get_memory_usage_mb()
            
            # 执行清理
            if hasattr(client, '_memory_manager'):
                client._memory_manager.cleanup()
            gc.collect()
            time.sleep(0.2)
            
            # 记录清理后的内存
            after_cleanup_memory = self.get_memory_usage_mb()
            
            # 获取内存统计
            memory_stats = {}
            if hasattr(client, 'get_memory_stats'):
                memory_stats = client.get_memory_stats()
            
            return {
                "initial_memory_mb": initial_memory,
                "after_usage_memory_mb": after_usage_memory,
                "after_cleanup_memory_mb": after_cleanup_memory,
                "memory_increase_mb": after_usage_memory - initial_memory,
                "memory_stats": memory_stats,
                "client": client
            }
            
        finally:
            # 清理资源
            if hasattr(client, 'cleanup'):
                client.cleanup()


class MemoryLeakDetector:
    """内存泄漏检测器"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = None
        self.memory_samples = []
        self.leak_threshold_mb = 5.0  # 内存泄漏阈值
    
    def start_monitoring(self):
        """开始监控内存使用"""
        gc.collect()
        time.sleep(0.1)
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        self.memory_samples = [self.initial_memory]
    
    def sample_memory(self):
        """采样当前内存使用"""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.memory_samples.append(current_memory)
        return current_memory
    
    def detect_leak(self) -> Dict[str, Any]:
        """检测内存泄漏"""
        if len(self.memory_samples) < 2:
            return {"has_leak": False, "reason": "样本不足"}
        
        # 计算内存增长趋势
        memory_growth = self.memory_samples[-1] - self.memory_samples[0]
        average_growth = memory_growth / len(self.memory_samples)
        
        # 检查是否超过阈值
        has_leak = memory_growth > self.leak_threshold_mb
        
        return {
            "has_leak": has_leak,
            "initial_memory_mb": self.initial_memory,
            "final_memory_mb": self.memory_samples[-1],
            "total_growth_mb": memory_growth,
            "average_growth_mb": average_growth,
            "threshold_mb": self.leak_threshold_mb,
            "samples_count": len(self.memory_samples),
            "memory_samples": self.memory_samples
        }


@pytest.mark.performance
@pytest.mark.memory
class TestMemoryComprehensive:
    """综合内存测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.benchmark = MemoryBenchmark()
        self.leak_detector = MemoryLeakDetector()
        gc.collect()
    
    def teardown_method(self):
        """测试后清理"""
        gc.collect()
    
    @pytest.mark.benchmark
    def test_memory_optimization_effectiveness(self):
        """测试内存优化效果"""
        if not FastHarborAI:
            pytest.skip("FastHarborAI 不可用")
        
        # 测量基线内存
        baseline = self.benchmark.measure_baseline()
        
        # 测量优化后的内存使用
        optimized_result = self.benchmark.measure_with_optimization()
        
        # 验证内存优化效果
        memory_increase = optimized_result["memory_increase_mb"]
        
        # 断言：内存增长应该在合理范围内
        assert memory_increase < self.benchmark.target_memory_mb, \
            f"内存增长 {memory_increase:.2f}MB 超过目标 {self.benchmark.target_memory_mb}MB"
        
        # 断言：清理后内存应该接近初始值
        cleanup_memory = optimized_result["after_cleanup_memory_mb"]
        initial_memory = optimized_result["initial_memory_mb"]
        cleanup_diff = abs(cleanup_memory - initial_memory)
        
        assert cleanup_diff < 2.0, \
            f"清理后内存差异 {cleanup_diff:.2f}MB 过大"
        
        print(f"内存优化测试通过:")
        print(f"  基线内存: {baseline:.2f}MB")
        print(f"  初始内存: {initial_memory:.2f}MB")
        print(f"  使用后内存: {optimized_result['after_usage_memory_mb']:.2f}MB")
        print(f"  清理后内存: {cleanup_memory:.2f}MB")
        print(f"  内存增长: {memory_increase:.2f}MB")
    
    @pytest.mark.leak_detection
    def test_memory_leak_detection(self):
        """测试内存泄漏检测"""
        if not FastHarborAI:
            pytest.skip("FastHarborAI 不可用")
        
        self.leak_detector.start_monitoring()
        
        # 模拟多次操作
        for iteration in range(10):
            # 创建客户端
            client = FastHarborAI()
            
            # 执行一些操作
            for i in range(50):
                # 模拟缓存操作
                test_data = {"iteration": iteration, "index": i, "data": "x" * 100}
                
            # 采样内存
            self.leak_detector.sample_memory()
            
            # 清理
            if hasattr(client, 'cleanup'):
                client.cleanup()
            del client
            gc.collect()
        
        # 检测泄漏
        leak_result = self.leak_detector.detect_leak()
        
        # 断言：不应该有内存泄漏
        assert not leak_result["has_leak"], \
            f"检测到内存泄漏: {leak_result['total_growth_mb']:.2f}MB"
        
        print(f"内存泄漏检测通过:")
        print(f"  总内存增长: {leak_result['total_growth_mb']:.2f}MB")
        print(f"  平均增长: {leak_result['average_growth_mb']:.2f}MB")
    
    @pytest.mark.stability
    def test_memory_stability_under_load(self):
        """测试负载下的内存稳定性"""
        if not FastHarborAI:
            pytest.skip("FastHarborAI 不可用")
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_samples = []
        
        # 模拟高负载操作
        for batch in range(5):
            batch_clients = []
            
            # 创建多个客户端
            for i in range(10):
                client = FastHarborAI()
                batch_clients.append(client)
                
                # 执行操作
                for j in range(20):
                    test_data = {"batch": batch, "client": i, "op": j}
            
            # 采样内存
            current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            # 清理批次
            for client in batch_clients:
                if hasattr(client, 'cleanup'):
                    client.cleanup()
            del batch_clients
            gc.collect()
            time.sleep(0.1)
        
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_growth = final_memory - initial_memory
        
        # 断言：内存增长应该在合理范围内
        assert memory_growth < 10.0, \
            f"负载测试后内存增长 {memory_growth:.2f}MB 过大"
        
        print(f"内存稳定性测试通过:")
        print(f"  初始内存: {initial_memory:.2f}MB")
        print(f"  最终内存: {final_memory:.2f}MB")
        print(f"  内存增长: {memory_growth:.2f}MB")
        print(f"  内存样本: {[f'{m:.1f}' for m in memory_samples]}")
    
    @pytest.mark.cache_memory
    def test_cache_memory_management(self):
        """测试缓存内存管理"""
        if not MemoryOptimizedCache:
            pytest.skip("MemoryOptimizedCache 不可用")
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # 创建缓存
        cache = MemoryOptimizedCache(max_size=1000, max_memory_mb=5.0)
        
        # 填充缓存
        for i in range(2000):  # 超过最大大小
            key = f"cache_key_{i}"
            value = {"data": "x" * 1000, "index": i}  # 每个值约1KB
            cache.set(key, value)
        
        # 检查内存使用
        after_fill_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = after_fill_memory - initial_memory
        
        # 断言：内存增长应该受到限制
        assert memory_increase < 10.0, \
            f"缓存内存增长 {memory_increase:.2f}MB 超过预期"
        
        # 清理缓存
        cache.clear()
        gc.collect()
        
        after_clear_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        cleanup_diff = abs(after_clear_memory - initial_memory)
        
        # 断言：清理后内存应该接近初始值
        assert cleanup_diff < 2.0, \
            f"缓存清理后内存差异 {cleanup_diff:.2f}MB 过大"
        
        print(f"缓存内存管理测试通过:")
        print(f"  初始内存: {initial_memory:.2f}MB")
        print(f"  填充后内存: {after_fill_memory:.2f}MB")
        print(f"  清理后内存: {after_clear_memory:.2f}MB")
    
    @pytest.mark.object_pool
    def test_object_pool_memory_reuse(self):
        """测试对象池内存复用"""
        if not ObjectPool:
            pytest.skip("ObjectPool 不可用")
        
        initial_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        # 创建对象池
        pool = ObjectPool(max_size=100)
        
        # 测试对象复用
        objects = []
        for i in range(200):  # 超过池大小
            obj = pool.get_object('test_type')
            if obj is None:
                obj = {"created": True, "index": i}
            objects.append(obj)
        
        # 归还对象
        for obj in objects:
            pool.return_object('test_type', obj)
        
        # 再次获取对象（应该复用）
        reused_objects = []
        for i in range(100):
            obj = pool.get_object('test_type')
            reused_objects.append(obj)
        
        # 检查内存使用
        after_reuse_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_increase = after_reuse_memory - initial_memory
        
        # 断言：对象复用应该控制内存增长
        assert memory_increase < 5.0, \
            f"对象池内存增长 {memory_increase:.2f}MB 超过预期"
        
        print(f"对象池内存复用测试通过:")
        print(f"  初始内存: {initial_memory:.2f}MB")
        print(f"  复用后内存: {after_reuse_memory:.2f}MB")
        print(f"  内存增长: {memory_increase:.2f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])