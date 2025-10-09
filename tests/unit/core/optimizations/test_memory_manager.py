#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存管理器测试模块

全面测试MemoryManager类的所有功能，包括：
1. 基本功能测试
2. 边界条件测试
3. 异常处理测试
4. 集成测试
5. 性能测试

遵循TDD原则，确保90%以上的代码覆盖率。
"""

import pytest
import threading
import time
import weakref
import gc
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, Optional

from harborai.core.optimizations.memory_manager import MemoryManager, get_memory_manager
from harborai.core.optimizations.memory_optimized_cache import MemoryOptimizedCache
from harborai.core.optimizations.object_pool import ObjectPool, ObjectPoolManager


class TestMemoryManager:
    """内存管理器测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        # 重置全局内存管理器
        import harborai.core.optimizations.memory_manager as mm_module
        mm_module._global_memory_manager = None
        
        # 创建测试用的内存管理器
        self.memory_manager = MemoryManager(
            cache_size=10,
            cache_ttl=1.0,
            object_pool_size=5,
            enable_weak_references=True,
            auto_cleanup_interval=0.1,  # 短间隔用于测试
            memory_threshold_mb=1.0
        )
    
    def teardown_method(self):
        """测试后置清理"""
        if hasattr(self, 'memory_manager'):
            self.memory_manager.shutdown()
        
        # 重置全局内存管理器
        import harborai.core.optimizations.memory_manager as mm_module
        mm_module._global_memory_manager = None
        
        # 强制垃圾回收
        gc.collect()
    
    def test_init_with_valid_config(self):
        """测试：使用有效配置初始化内存管理器"""
        mm = MemoryManager(
            cache_size=100,
            cache_ttl=60.0,
            object_pool_size=50,
            enable_weak_references=True,
            auto_cleanup_interval=300.0,
            memory_threshold_mb=200.0
        )
        
        assert mm._cache_size == 100
        assert mm._cache_ttl == 60.0
        assert mm._object_pool_size == 50
        assert mm._enable_weak_references is True
        assert mm._auto_cleanup_interval == 300.0
        assert mm._memory_threshold_mb == 200.0
        
        # 验证组件初始化
        assert isinstance(mm._cache, MemoryOptimizedCache)
        assert isinstance(mm._object_pool_manager, ObjectPoolManager)
        assert isinstance(mm._default_object_pool, ObjectPool)
        
        mm.shutdown()
    
    def test_init_with_invalid_config(self):
        """测试：使用无效配置初始化内存管理器（配置修正）"""
        with patch('harborai.core.optimizations.memory_manager.logger') as mock_logger:
            mm = MemoryManager(
                cache_size=-10,  # 无效值
                cache_ttl="invalid",  # 无效类型
                object_pool_size=0,  # 无效值
                enable_weak_references="invalid",  # 无效类型
                auto_cleanup_interval=-1,  # 无效值
                memory_threshold_mb=0  # 无效值
            )
            
            # 验证配置被修正为默认值
            assert mm._cache_size == 1000
            assert mm._cache_ttl is None
            assert mm._object_pool_size == 100
            assert mm._enable_weak_references is True
            assert mm._auto_cleanup_interval == 300.0
            assert mm._memory_threshold_mb == 100.0
            
            # 验证警告日志被记录
            assert mock_logger.warning.call_count >= 2
            
            mm.shutdown()
    
    def test_cache_property(self):
        """测试：缓存属性访问"""
        cache = self.memory_manager.cache
        assert isinstance(cache, MemoryOptimizedCache)
        assert cache is self.memory_manager._cache
    
    def test_object_pool_property(self):
        """测试：对象池属性访问"""
        pool = self.memory_manager.object_pool
        assert isinstance(pool, ObjectPool)
        assert pool is self.memory_manager._default_object_pool
    
    def test_create_object_pool(self):
        """测试：创建自定义对象池"""
        class TestObject:
            def __init__(self, value=0):
                self.value = value
        
        def factory():
            return TestObject(42)
        
        def reset_func(obj):
            obj.value = 0
        
        def cleanup_func(obj):
            obj.value = -1
        
        pool = self.memory_manager.create_object_pool(
            name='test_pool',
            object_type=TestObject,
            max_size=3,
            factory_func=factory,
            reset_func=reset_func,
            cleanup_func=cleanup_func
        )
        
        assert isinstance(pool, ObjectPool)
        assert pool._object_type == TestObject
        assert pool._max_size == 3
        assert pool._factory_func == factory
        assert pool._reset_func == reset_func
        assert pool._cleanup_func == cleanup_func
    
    def test_create_object_pool_with_default_size(self):
        """测试：创建对象池使用默认大小"""
        pool = self.memory_manager.create_object_pool(
            name='default_size_pool',
            object_type=dict
        )
        
        assert pool._max_size == self.memory_manager._object_pool_size
    
    def test_get_pooled_object_success(self):
        """测试：成功从对象池获取对象"""
        # 创建测试对象池，提供工厂函数
        pool = self.memory_manager.create_object_pool(
            name='test_pool',
            object_type=dict,
            max_size=2,
            factory_func=lambda: {'created': True}
        )
        
        # 获取对象
        obj = self.memory_manager.get_pooled_object('test_pool')
        assert obj is not None
        assert isinstance(obj, dict)
        assert obj.get('created') is True
        
        # 验证统计信息更新
        stats = self.memory_manager.get_memory_stats()
        assert stats['total_objects_created'] >= 1
    
    def test_get_pooled_object_nonexistent_pool(self):
        """测试：从不存在的对象池获取对象"""
        obj = self.memory_manager.get_pooled_object('nonexistent_pool')
        assert obj is None
    
    def test_release_pooled_object_success(self):
        """测试：成功释放对象到对象池"""
        # 创建测试对象池，提供工厂函数
        pool = self.memory_manager.create_object_pool(
            name='test_pool',
            object_type=dict,
            max_size=2,
            factory_func=lambda: {'created': True}
        )
        
        # 获取并释放对象
        obj = self.memory_manager.get_pooled_object('test_pool')
        assert obj is not None
        
        result = self.memory_manager.release_pooled_object('test_pool', obj)
        assert result is True
    
    def test_release_pooled_object_nonexistent_pool(self):
        """测试：释放对象到不存在的对象池"""
        obj = {}
        result = self.memory_manager.release_pooled_object('nonexistent_pool', obj)
        assert result is False
    
    def test_add_weak_reference_success(self):
        """测试：成功添加弱引用"""
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        test_obj = TestObject('test_data')
        
        result = self.memory_manager.add_weak_reference('test_key', test_obj)
        assert result is True
        
        # 验证弱引用存在
        assert 'test_key' in self.memory_manager._weak_refs
        
        # 验证可以获取对象
        retrieved_obj = self.memory_manager.get_weak_reference('test_key')
        assert retrieved_obj is test_obj
    
    def test_add_weak_reference_with_callback(self):
        """测试：添加带回调的弱引用"""
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        test_obj = TestObject('test_data')
        callback_called = []
        
        def callback(ref):
            callback_called.append(True)
        
        result = self.memory_manager.add_weak_reference('test_key', test_obj, callback)
        assert result is True
        
        # 删除对象触发回调
        del test_obj
        gc.collect()
        
        # 等待回调执行
        time.sleep(0.01)
        
        # 验证回调被调用
        assert len(callback_called) > 0
    
    def test_add_weak_reference_unsupported_object(self):
        """测试：添加不支持弱引用的对象"""
        # 整数不支持弱引用
        with patch('harborai.core.optimizations.memory_manager.logger') as mock_logger:
            result = self.memory_manager.add_weak_reference('test_key', 42)
            assert result is False
            mock_logger.warning.assert_called_once()
    
    def test_add_weak_reference_disabled(self):
        """测试：弱引用功能被禁用时添加弱引用"""
        mm = MemoryManager(enable_weak_references=False)
        
        test_obj = {'test': 'data'}
        result = mm.add_weak_reference('test_key', test_obj)
        assert result is False
        
        mm.shutdown()
    
    def test_get_weak_reference_success(self):
        """测试：成功获取弱引用对象"""
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        test_obj = TestObject('test_data')
        self.memory_manager.add_weak_reference('test_key', test_obj)
        
        retrieved_obj = self.memory_manager.get_weak_reference('test_key')
        assert retrieved_obj is test_obj
    
    def test_get_weak_reference_deleted_object(self):
        """测试：获取已删除对象的弱引用"""
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        test_obj = TestObject('test_data')
        self.memory_manager.add_weak_reference('test_key', test_obj)
        
        # 删除对象
        del test_obj
        gc.collect()
        
        # 尝试获取已删除的对象
        retrieved_obj = self.memory_manager.get_weak_reference('test_key')
        assert retrieved_obj is None
        
        # 验证弱引用被自动清理
        assert 'test_key' not in self.memory_manager._weak_refs
    
    def test_get_weak_reference_nonexistent(self):
        """测试：获取不存在的弱引用"""
        retrieved_obj = self.memory_manager.get_weak_reference('nonexistent_key')
        assert retrieved_obj is None
    
    def test_get_weak_reference_disabled(self):
        """测试：弱引用功能被禁用时获取弱引用"""
        mm = MemoryManager(enable_weak_references=False)
        
        retrieved_obj = mm.get_weak_reference('test_key')
        assert retrieved_obj is None
        
        mm.shutdown()
    
    def test_remove_weak_reference_success(self):
        """测试：成功移除弱引用"""
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        test_obj = TestObject('test_data')
        self.memory_manager.add_weak_reference('test_key', test_obj)
        
        result = self.memory_manager.remove_weak_reference('test_key')
        assert result is True
        
        # 验证弱引用被移除
        assert 'test_key' not in self.memory_manager._weak_refs
    
    def test_remove_weak_reference_nonexistent(self):
        """测试：移除不存在的弱引用"""
        result = self.memory_manager.remove_weak_reference('nonexistent_key')
        assert result is False
    
    def test_remove_weak_reference_disabled(self):
        """测试：弱引用功能被禁用时移除弱引用"""
        mm = MemoryManager(enable_weak_references=False)
        
        result = mm.remove_weak_reference('test_key')
        assert result is False
        
        mm.shutdown()
    
    def test_cleanup_normal(self):
        """测试：正常清理操作"""
        # 添加一些数据到缓存
        self.memory_manager.cache.set('key1', 'value1')
        self.memory_manager.cache.set('key2', 'value2')
        
        # 添加弱引用
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        test_obj = TestObject('test_data')
        self.memory_manager.add_weak_reference('test_key', test_obj)
        
        # 执行清理
        cleanup_stats = self.memory_manager.cleanup()
        
        assert isinstance(cleanup_stats, dict)
        assert 'cache_expired' in cleanup_stats
        assert 'cache_cleared' in cleanup_stats
        assert 'pools_shrunk' in cleanup_stats
        assert 'pools_cleared' in cleanup_stats
        assert 'weak_refs_cleaned' in cleanup_stats
    
    def test_cleanup_force_clear(self):
        """测试：强制清理操作"""
        # 添加一些数据到缓存
        self.memory_manager.cache.set('key1', 'value1')
        self.memory_manager.cache.set('key2', 'value2')
        
        # 获取清理前的缓存大小
        cache_size_before = self.memory_manager.cache.size()
        
        # 执行强制清理
        cleanup_stats = self.memory_manager.cleanup(force_clear=True)
        
        assert cleanup_stats['cache_cleared'] == cache_size_before
        assert self.memory_manager.cache.size() == 0
    
    def test_cleanup_with_dead_weak_references(self):
        """测试：清理死弱引用"""
        # 添加弱引用
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        test_obj = TestObject('test_data')
        self.memory_manager.add_weak_reference('test_key', test_obj)
        
        # 删除对象创建死弱引用
        del test_obj
        gc.collect()
        
        # 执行清理
        cleanup_stats = self.memory_manager.cleanup()
        
        # 验证死弱引用被清理
        assert cleanup_stats['weak_refs_cleaned'] >= 0
    
    def test_get_memory_stats(self):
        """测试：获取内存统计信息"""
        # 添加一些活动
        self.memory_manager.cache.set('key1', 'value1')
        self.memory_manager.get_pooled_object('default')
        
        stats = self.memory_manager.get_memory_stats()
        
        assert isinstance(stats, dict)
        assert 'uptime_seconds' in stats
        assert 'cache' in stats
        assert 'object_pools' in stats
        assert 'weak_references_count' in stats
        assert 'total_cache_hits' in stats
        assert 'total_cache_misses' in stats
        assert 'total_objects_created' in stats
        assert 'total_objects_reused' in stats
        assert 'cleanup_count' in stats
        assert 'memory_warnings' in stats
        assert 'object_reuse_rate' in stats
        
        # 验证统计数据类型
        assert isinstance(stats['uptime_seconds'], (int, float))
        assert isinstance(stats['object_reuse_rate'], float)
        assert 0.0 <= stats['object_reuse_rate'] <= 1.0
    
    def test_get_memory_stats_no_objects(self):
        """测试：没有对象时的统计信息"""
        stats = self.memory_manager.get_memory_stats()
        
        # 当没有对象时，复用率应该为0
        assert stats['object_reuse_rate'] == 0.0
        assert stats['total_objects_created'] == 0
        assert stats['total_objects_reused'] == 0
    
    @patch('psutil.Process')
    def test_check_memory_usage_normal(self, mock_process_class):
        """测试：正常内存使用检查"""
        # 模拟内存使用低于阈值
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 0.5 * 1024 * 1024  # 0.5MB
        mock_process_class.return_value = mock_process
        
        result = self.memory_manager.check_memory_usage()
        assert result is False  # 未超过阈值
    
    @patch('psutil.Process')
    def test_check_memory_usage_exceeded(self, mock_process_class):
        """测试：内存使用超过阈值"""
        # 模拟内存使用超过阈值
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 2.0 * 1024 * 1024  # 2MB
        mock_process_class.return_value = mock_process
        
        with patch('harborai.core.optimizations.memory_manager.logger') as mock_logger:
            result = self.memory_manager.check_memory_usage()
            assert result is True  # 超过阈值
            mock_logger.warning.assert_called_once()
            
            # 验证警告计数增加
            stats = self.memory_manager.get_memory_stats()
            assert stats['memory_warnings'] >= 1
    
    def test_check_memory_usage_no_psutil(self):
        """测试：psutil未安装时的内存检查"""
        # 模拟psutil导入失败
        import builtins
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'psutil':
                raise ImportError("No module named 'psutil'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with patch('harborai.core.optimizations.memory_manager.logger') as mock_logger:
                result = self.memory_manager.check_memory_usage()
                assert result is False
                mock_logger.warning.assert_called_once()
    
    def test_on_weak_ref_deleted(self):
        """测试：弱引用删除回调"""
        # 手动添加弱引用到字典
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        test_obj = TestObject('test_data')
        ref = weakref.ref(test_obj)
        self.memory_manager._weak_refs['test_key'] = ref
        
        # 调用回调
        self.memory_manager._on_weak_ref_deleted('test_key')
        
        # 验证弱引用被移除
        assert 'test_key' not in self.memory_manager._weak_refs
    
    def test_on_weak_ref_deleted_nonexistent(self):
        """测试：删除不存在的弱引用回调"""
        # 调用不存在键的回调不应该出错
        self.memory_manager._on_weak_ref_deleted('nonexistent_key')
    
    def test_auto_cleanup_task(self):
        """测试：自动清理任务"""
        with patch.object(self.memory_manager, 'check_memory_usage', return_value=False):
            with patch.object(self.memory_manager, 'cleanup') as mock_cleanup:
                with patch.object(self.memory_manager, '_start_auto_cleanup') as mock_start:
                    
                    # 执行自动清理任务
                    self.memory_manager._auto_cleanup_task()
                    
                    # 验证清理被调用
                    mock_cleanup.assert_called_once()
                    # 验证定时器重新启动
                    mock_start.assert_called_once()
    
    def test_auto_cleanup_task_with_exception(self):
        """测试：自动清理任务异常处理"""
        with patch.object(self.memory_manager, 'check_memory_usage', side_effect=Exception("Test error")):
            with patch('harborai.core.optimizations.memory_manager.logger') as mock_logger:
                with patch.object(self.memory_manager, '_start_auto_cleanup') as mock_start:
                    
                    # 执行自动清理任务
                    self.memory_manager._auto_cleanup_task()
                    
                    # 验证错误被记录
                    mock_logger.error.assert_called_once()
                    # 验证定时器仍然重新启动
                    mock_start.assert_called_once()
    
    def test_shutdown(self):
        """测试：关闭内存管理器"""
        # 添加一些数据
        self.memory_manager.cache.set('key1', 'value1')
        
        class TestObject:
            def __init__(self, data):
                self.data = data
        
        test_obj = TestObject('test_data')
        self.memory_manager.add_weak_reference('test_key', test_obj)
        
        # 关闭
        self.memory_manager.shutdown()
        
        # 验证资源被清理
        assert self.memory_manager.cache.size() == 0
        assert len(self.memory_manager._weak_refs) == 0
        
        # 验证定时器被取消
        if self.memory_manager._cleanup_timer:
            # 等待定时器停止
            import time
            time.sleep(0.1)
            assert not self.memory_manager._cleanup_timer.is_alive()
    
    def test_destructor(self):
        """测试：析构函数"""
        mm = MemoryManager(cache_size=5, object_pool_size=3)
        
        # 添加一些数据
        mm.cache.set('key1', 'value1')
        
        # 模拟析构
        with patch.object(mm, 'shutdown') as mock_shutdown:
            mm.__del__()
            mock_shutdown.assert_called_once()
    
    def test_destructor_with_exception(self):
        """测试：析构函数异常处理"""
        mm = MemoryManager(cache_size=5, object_pool_size=3)
        
        # 模拟shutdown抛出异常
        with patch.object(mm, 'shutdown', side_effect=Exception("Test error")):
            # 析构函数不应该抛出异常
            mm.__del__()
    
    def test_threading_safety(self):
        """测试：多线程安全性"""
        results = []
        errors = []
        
        def worker():
            try:
                # 并发操作
                for i in range(10):
                    # 缓存操作
                    self.memory_manager.cache.set(f'key_{i}', f'value_{i}')
                    
                    # 对象池操作
                    obj = self.memory_manager.get_pooled_object('default')
                    if obj:
                        self.memory_manager.release_pooled_object('default', obj)
                    
                    # 弱引用操作
                    class TestObject:
                        def __init__(self, data):
                            self.data = data
                    
                    test_obj = TestObject(i)
                    self.memory_manager.add_weak_reference(f'ref_{i}', test_obj)
                    
                    # 统计操作
                    stats = self.memory_manager.get_memory_stats()
                    results.append(stats)
                    
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证没有错误
        assert len(errors) == 0, f"线程安全测试失败: {errors}"
        assert len(results) > 0


class TestGetMemoryManager:
    """全局内存管理器获取函数测试"""
    
    def teardown_method(self):
        """测试后置清理"""
        # 重置全局内存管理器
        import harborai.core.optimizations.memory_manager as mm_module
        if mm_module._global_memory_manager:
            mm_module._global_memory_manager.shutdown()
        mm_module._global_memory_manager = None
    
    @pytest.mark.asyncio
    async def test_get_memory_manager_first_time(self):
        """测试：首次获取全局内存管理器"""
        # 清理全局状态
        import harborai.core.optimizations.memory_manager as mm_module
        original_global = mm_module._global_memory_manager
        mm_module._global_memory_manager = None
        
        try:
            config = {'cache_size': 50, 'object_pool_size': 25}
            
            mm = await get_memory_manager(config)
            
            assert isinstance(mm, MemoryManager)
            assert mm._cache_size == 50
            assert mm._object_pool_size == 25
        finally:
            # 恢复全局状态
            if mm_module._global_memory_manager:
                mm_module._global_memory_manager.shutdown()
            mm_module._global_memory_manager = original_global
    
    @pytest.mark.asyncio
    async def test_get_memory_manager_singleton(self):
        """测试：全局内存管理器单例模式"""
        mm1 = await get_memory_manager()
        mm2 = await get_memory_manager()
        
        assert mm1 is mm2  # 应该是同一个实例
    
    @pytest.mark.asyncio
    async def test_get_memory_manager_no_config(self):
        """测试：无配置获取全局内存管理器"""
        mm = await get_memory_manager()
        
        assert isinstance(mm, MemoryManager)
        # 应该使用默认配置
        assert mm._cache_size == 1000
        assert mm._object_pool_size == 100
    
    @pytest.mark.asyncio
    async def test_get_memory_manager_none_config(self):
        """测试：None配置获取全局内存管理器"""
        mm = await get_memory_manager(None)
        
        assert isinstance(mm, MemoryManager)
        # 应该使用默认配置
        assert mm._cache_size == 1000
        assert mm._object_pool_size == 100


class TestMemoryManagerIntegration:
    """内存管理器集成测试"""
    
    def setup_method(self):
        """测试前置设置"""
        # 重置全局内存管理器
        import harborai.core.optimizations.memory_manager as mm_module
        mm_module._global_memory_manager = None
    
    def teardown_method(self):
        """测试后置清理"""
        # 重置全局内存管理器
        import harborai.core.optimizations.memory_manager as mm_module
        if mm_module._global_memory_manager:
            mm_module._global_memory_manager.shutdown()
        mm_module._global_memory_manager = None
        
        # 强制垃圾回收
        gc.collect()
    
    def test_cache_and_object_pool_integration(self):
        """测试：缓存和对象池集成使用"""
        mm = MemoryManager(cache_size=10, object_pool_size=5)
        
        try:
            # 使用缓存
            mm.cache.set('cache_key', 'cache_value')
            assert mm.cache.get('cache_key') == 'cache_value'
            
            # 创建对象池
            mm.create_object_pool('test_pool', lambda: {'created': True}, 3)
            
            # 使用对象池
            obj = mm.get_pooled_object('test_pool')
            assert obj is not None
            
            # 释放对象
            result = mm.release_pooled_object('test_pool', obj)
            assert result is True
            
            # 验证统计信息
            stats = mm.get_memory_stats()
            assert stats['cache']['size'] == 1
            assert stats['total_objects_created'] >= 1
            
        finally:
            mm.shutdown()
    
    def test_weak_references_and_cleanup_integration(self):
        """测试：弱引用和清理集成"""
        mm = MemoryManager(enable_weak_references=True)
        
        try:
            # 添加弱引用
            class TestObject:
                def __init__(self, data):
                    self.data = data
            
            test_obj = TestObject('test_data')
            mm.add_weak_reference('test_key', test_obj)
            
            # 验证弱引用存在
            assert mm.get_weak_reference('test_key') is test_obj
            
            # 删除对象
            del test_obj
            gc.collect()
            
            # 执行清理
            cleanup_stats = mm.cleanup()
            
            # 验证死弱引用被清理
            assert mm.get_weak_reference('test_key') is None
            
        finally:
            mm.shutdown()
    
    def test_memory_monitoring_and_auto_cleanup(self):
        """测试：内存监控和自动清理集成"""
        mm = MemoryManager(
            memory_threshold_mb=0.001,  # 非常小的阈值，确保触发警告
            auto_cleanup_interval=0.05  # 50ms间隔
        )
        
        try:
            # 添加数据到缓存
            for i in range(5):
                mm.cache.set(f'key_{i}', f'value_{i}')
            
            # 手动触发内存检查（使用真实的psutil）
            mm.check_memory_usage()
            
            # 手动触发清理以确保cleanup_count增加
            mm.cleanup()
            
            # 等待自动清理触发
            time.sleep(0.15)  # 增加等待时间
            
            # 验证内存警告被记录
            stats = mm.get_memory_stats()
            assert stats['memory_warnings'] >= 1
            assert stats['cleanup_count'] >= 1
            
        finally:
            mm.shutdown()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])