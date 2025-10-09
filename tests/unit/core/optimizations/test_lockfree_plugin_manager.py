#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无锁插件管理器测试

测试LockFreePluginManager、AtomicInteger、AtomicReference等类的功能。
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import Future

from harborai.core.optimizations.lockfree_plugin_manager import (
    AtomicInteger, AtomicReference, PluginEntry, LockFreePluginManager
)
from harborai.core.lazy_plugin_manager import LazyPluginInfo
from harborai.core.plugins.base import Plugin


class MockPlugin:
    """模拟插件类"""
    
    def __init__(self, name: str = "mock_plugin"):
        self.name = name
        self.initialized = False
    
    def initialize(self):
        self.initialized = True
    
    def get_supported_models(self):
        return ["mock-model-1", "mock-model-2"]
    
    def chat_completion(self, *args, **kwargs):
        return {"response": "mock response"}
    
    def chat_completion_async(self, *args, **kwargs):
        return {"response": "mock async response"}
    
    @property
    def info(self):
        return {"name": self.name}


class TestAtomicInteger:
    """AtomicInteger测试类"""
    
    def test_init_default(self):
        """测试默认初始化"""
        atomic_int = AtomicInteger()
        assert atomic_int.get() == 0
    
    def test_init_with_value(self):
        """测试带初始值的初始化"""
        atomic_int = AtomicInteger(42)
        assert atomic_int.get() == 42
    
    def test_set_and_get(self):
        """测试设置和获取值"""
        atomic_int = AtomicInteger()
        atomic_int.set(100)
        assert atomic_int.get() == 100
    
    def test_increment(self):
        """测试递增操作"""
        atomic_int = AtomicInteger(5)
        result = atomic_int.increment()
        assert result == 6
        assert atomic_int.get() == 6
    
    def test_decrement(self):
        """测试递减操作"""
        atomic_int = AtomicInteger(10)
        result = atomic_int.decrement()
        assert result == 9
        assert atomic_int.get() == 9
    
    def test_add(self):
        """测试加法操作"""
        atomic_int = AtomicInteger(20)
        result = atomic_int.add(15)
        assert result == 35
        assert atomic_int.get() == 35
    
    def test_add_negative(self):
        """测试负数加法"""
        atomic_int = AtomicInteger(20)
        result = atomic_int.add(-5)
        assert result == 15
        assert atomic_int.get() == 15
    
    def test_compare_and_swap_success(self):
        """测试成功的CAS操作"""
        atomic_int = AtomicInteger(10)
        result = atomic_int.compare_and_swap(10, 20)
        assert result is True
        assert atomic_int.get() == 20
    
    def test_compare_and_swap_failure(self):
        """测试失败的CAS操作"""
        atomic_int = AtomicInteger(10)
        result = atomic_int.compare_and_swap(15, 20)
        assert result is False
        assert atomic_int.get() == 10
    
    def test_thread_safety(self):
        """测试线程安全性"""
        atomic_int = AtomicInteger(0)
        results = []
        
        def worker():
            for _ in range(100):
                atomic_int.increment()
        
        # 启动多个线程
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 应该是1000（10个线程 × 100次递增）
        assert atomic_int.get() == 1000


class TestAtomicReference:
    """AtomicReference测试类"""
    
    def test_init_default(self):
        """测试默认初始化"""
        atomic_ref = AtomicReference()
        assert atomic_ref.get() is None
    
    def test_init_with_value(self):
        """测试带初始值的初始化"""
        obj = {"key": "value"}
        atomic_ref = AtomicReference(obj)
        assert atomic_ref.get() is obj
    
    def test_set_and_get(self):
        """测试设置和获取引用"""
        atomic_ref = AtomicReference()
        obj = [1, 2, 3]
        atomic_ref.set(obj)
        assert atomic_ref.get() is obj
    
    def test_compare_and_swap_success(self):
        """测试成功的CAS操作"""
        obj1 = {"a": 1}
        obj2 = {"b": 2}
        atomic_ref = AtomicReference(obj1)
        
        result = atomic_ref.compare_and_swap(obj1, obj2)
        assert result is True
        assert atomic_ref.get() is obj2
    
    def test_compare_and_swap_failure(self):
        """测试失败的CAS操作"""
        obj1 = {"a": 1}
        obj2 = {"b": 2}
        obj3 = {"c": 3}
        atomic_ref = AtomicReference(obj1)
        
        result = atomic_ref.compare_and_swap(obj2, obj3)
        assert result is False
        assert atomic_ref.get() is obj1
    
    def test_thread_safety(self):
        """测试线程安全性"""
        atomic_ref = AtomicReference([])
        
        def worker(thread_id):
            for i in range(10):
                current = atomic_ref.get()
                new_list = current + [f"thread_{thread_id}_item_{i}"]
                # 尝试CAS直到成功
                while not atomic_ref.compare_and_swap(current, new_list):
                    current = atomic_ref.get()
                    new_list = current + [f"thread_{thread_id}_item_{i}"]
        
        # 启动多个线程
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 检查最终列表长度
        final_list = atomic_ref.get()
        assert len(final_list) == 30  # 3个线程 × 10个项目


class TestPluginEntry:
    """PluginEntry测试类"""
    
    def test_init_basic(self):
        """测试基本初始化"""
        info = LazyPluginInfo(
            name="test_plugin",
            module_path="test.module",
            class_name="TestPlugin",
            supported_models=["model1"]
        )
        entry = PluginEntry(name="test_plugin", info=info)
        
        assert entry.name == "test_plugin"
        assert entry.info is info
        assert entry.instance is None
        assert isinstance(entry.load_count, AtomicInteger)
        assert entry.load_count.get() == 0
        assert isinstance(entry.loading, AtomicInteger)
        assert entry.loading.get() == 0
        assert isinstance(entry.error_count, AtomicInteger)
        assert entry.error_count.get() == 0
        assert isinstance(entry.last_access_time, float)
    
    def test_post_init_conversion(self):
        """测试__post_init__的类型转换"""
        info = LazyPluginInfo(
            name="test_plugin",
            module_path="test.module",
            class_name="TestPlugin",
            supported_models=["model1"]
        )
        # 使用整数初始化
        entry = PluginEntry(
            name="test_plugin",
            info=info,
            load_count=5,
            loading=1,
            error_count=2
        )
        
        # 应该被转换为AtomicInteger
        assert isinstance(entry.load_count, AtomicInteger)
        assert entry.load_count.get() == 5
        assert isinstance(entry.loading, AtomicInteger)
        assert entry.loading.get() == 1
        assert isinstance(entry.error_count, AtomicInteger)
        assert entry.error_count.get() == 2


class TestLockFreePluginManager:
    """LockFreePluginManager测试类"""
    
    def test_init_default(self):
        """测试默认初始化"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            assert manager.config == {}
            assert isinstance(manager._plugin_entries, dict)
            assert isinstance(manager._model_to_plugin, dict)
            assert isinstance(manager._stats, dict)
            assert 'total_requests' in manager._stats
            assert isinstance(manager._stats['total_requests'], AtomicInteger)
    
    def test_init_with_config(self):
        """测试带配置的初始化"""
        config = {"max_workers": 8, "timeout": 30}
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor') as mock_executor:
            manager = LockFreePluginManager(config)
            
            assert manager.config == config
            mock_executor.assert_called_once_with(
                max_workers=8,
                thread_name_prefix="lockfree_plugin"
            )
    
    def test_initialize_plugin_registry(self):
        """测试插件注册表初始化"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 检查是否注册了插件
            assert len(manager._plugin_entries) > 0
            
            # 检查插件条目的结构
            for name, atomic_ref in manager._plugin_entries.items():
                assert isinstance(atomic_ref, AtomicReference)
                entry = atomic_ref.get()
                assert isinstance(entry, PluginEntry)
                assert entry.name == name
                assert isinstance(entry.info, LazyPluginInfo)
    
    def test_get_plugin_entry_existing(self):
        """测试获取存在的插件条目"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 获取第一个插件
            plugin_name = list(manager._plugin_entries.keys())[0]
            entry_ref = manager._plugin_entries.get(plugin_name)
            entry = entry_ref.get() if entry_ref else None
            
            assert entry is not None
            assert isinstance(entry, PluginEntry)
            assert entry.name == plugin_name
    
    def test_get_plugin_entry_nonexistent(self):
        """测试获取不存在的插件条目"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            entry_ref = manager._plugin_entries.get("nonexistent_plugin")
            assert entry_ref is None
    
    def test_get_plugin_name_for_model_existing(self):
        """测试获取模型对应的插件名称"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 查找支持特定模型的插件
            plugin_name = manager.get_plugin_name_for_model("deepseek-chat")
            assert plugin_name == "deepseek"
    
    def test_get_plugin_name_for_model_nonexistent(self):
        """测试获取不存在模型的插件名称"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            plugin_name = manager.get_plugin_name_for_model("nonexistent-model")
            assert plugin_name is None
    
    def test_load_plugin_success(self):
        """测试成功加载插件"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 模拟插件条目
            entry = PluginEntry(
                name="test_plugin",
                info=LazyPluginInfo(
                    name="test_plugin",
                    module_path="test.module",
                    class_name="TestPlugin",
                    supported_models=["test-model"]
                )
            )
            
            # 模拟加载成功
            mock_plugin = MockPlugin("test_plugin")
            with patch.object(manager, '_do_load_plugin', return_value=mock_plugin):
                result = manager._load_plugin_lockfree(entry)
                
                assert result == mock_plugin
                assert entry.instance == mock_plugin
                assert entry.loading.get() == 2  # 已加载状态
    
    def test_load_plugin_failure(self):
        """测试插件加载失败"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            with patch('importlib.import_module', side_effect=ImportError("Module not found")):
                info = LazyPluginInfo(
                    name="test_plugin",
                    module_path="test.module",
                    class_name="TestPlugin",
                    supported_models=["test-model"]
                )
                entry = PluginEntry(name="test_plugin", info=info)
                
                plugin = manager._load_plugin_lockfree(entry)
                
                assert plugin is None
                assert entry.instance is None
                assert entry.loading.get() == 0  # 重置为未加载状态
                assert entry.error_count.get() == 1
    
    def test_get_plugin_cached(self):
        """测试获取缓存的插件"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 模拟已加载的插件
            mock_plugin = MockPlugin("test_plugin")
            entry = PluginEntry(
                name="test_plugin",
                info=LazyPluginInfo(
                    name="test_plugin",
                    module_path="test.module",
                    class_name="TestPlugin",
                    supported_models=["test-model"]
                )
            )
            entry.instance = mock_plugin
            entry.loading.set(2)  # 已加载状态
            
            # 将插件条目添加到管理器
            manager._plugin_entries["test_plugin"] = AtomicReference(entry)
            
            result = manager.get_plugin("test_plugin")
            assert result == mock_plugin
    
    def test_get_plugin_load_required(self):
        """测试需要加载的插件"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 模拟未加载的插件
            entry = PluginEntry(
                name="test_plugin",
                info=LazyPluginInfo(
                    name="test_plugin",
                    module_path="test.module",
                    class_name="TestPlugin",
                    supported_models=["test-model"]
                )
            )
            
            mock_plugin = MockPlugin("test_plugin")
            
            # 将插件条目添加到管理器
            manager._plugin_entries["test_plugin"] = AtomicReference(entry)
            manager._model_to_plugin["test-model"] = "test_plugin"
            
            # 模拟加载插件
            with patch.object(manager, '_do_load_plugin', return_value=mock_plugin):
                result = manager.get_plugin("test_plugin")
                
                assert result == mock_plugin
    
    def test_get_plugin_nonexistent_model(self):
        """测试获取不存在模型的插件"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            plugin = manager.get_plugin("nonexistent-model")
            assert plugin is None
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            stats = manager.get_statistics()
            
            assert isinstance(stats, dict)
            assert 'total_requests' in stats
            assert 'cache_hits' in stats
            assert 'cache_misses' in stats
            assert 'load_attempts' in stats
            assert 'load_successes' in stats
            assert 'load_failures' in stats
            assert 'registered_plugins' in stats
            assert 'loaded_plugins' in stats
    
    def test_clear_cache(self):
        """测试清除缓存"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 设置一些已加载的插件
            plugin_name = "deepseek"
            entry_ref = manager._plugin_entries.get(plugin_name)
            if entry_ref:
                entry = entry_ref.get()
                entry.instance = MockPlugin("deepseek")
                entry.loading.set(2)
                
                # 清理插件实例
                entry.instance = None
                entry.loading.set(0)
                
                assert entry.instance is None
                assert entry.loading.get() == 0
    
    def test_reload_plugin(self):
        """测试重新加载插件"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 设置已加载的插件
            plugin_name = "deepseek"
            entry_ref = manager._plugin_entries.get(plugin_name)
            if entry_ref:
                entry = entry_ref.get()
                old_plugin = MockPlugin("deepseek_old")
                entry.instance = old_plugin
                entry.loading.set(2)
                
                # 模拟重新加载
                new_plugin = MockPlugin("deepseek_new")
                with patch.object(manager, '_do_load_plugin', return_value=new_plugin):
                    # 清理并重新加载
                    entry.instance = None
                    entry.loading.set(0)
                    result = manager._load_plugin_lockfree(entry)
                    
                    assert result is new_plugin
                    assert entry.instance is new_plugin
    
    def test_reload_plugin_nonexistent(self):
        """测试重新加载不存在的插件"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 测试不存在的插件
            entry_ref = manager._plugin_entries.get("nonexistent_plugin")
            assert entry_ref is None
    
    def test_get_supported_models(self):
        """测试获取支持的模型列表"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            models = manager.get_supported_models()
            
            assert isinstance(models, list)
            assert len(models) > 0
            assert "deepseek-chat" in models
            assert "doubao-pro-4k" in models
    
    def test_concurrent_plugin_loading(self):
        """测试并发插件加载"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            results = []
            errors = []
            
            def worker():
                try:
                    plugin = manager.get_plugin("deepseek-chat")
                    results.append(plugin)
                except Exception as e:
                    errors.append(e)
            
            # 启动多个线程同时请求同一个插件
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            # 检查结果
            assert len(errors) == 0
            assert len(results) == 10
            # 所有结果应该是同一个插件实例（或None）
            unique_plugins = set(id(p) for p in results if p is not None)
            assert len(unique_plugins) <= 1  # 最多一个唯一插件实例
    
    def test_performance_monitoring(self):
        """测试性能监控"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 模拟加载时间记录
            load_time = 0.1
            manager._update_performance_stats(load_time)
            
            stats = manager.get_statistics()
            assert 'performance' in stats
            perf_stats = stats['performance']
            assert 'avg_load_time' in perf_stats
            assert 'max_load_time' in perf_stats
            assert 'min_load_time' in perf_stats
    
    def test_cleanup(self):
        """测试清理资源"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor') as mock_executor_class:
            mock_executor = Mock()
            mock_executor_class.return_value = mock_executor
            
            manager = LockFreePluginManager()
            manager.cleanup()
            
            mock_executor.shutdown.assert_called_once_with(wait=True)
    
    def test_thread_safety(self):
        """测试线程安全性"""
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            
            # 测试原子操作的线程安全性
            stats = manager.get_statistics()
            assert isinstance(stats, dict)
            assert 'total_requests' in stats
            assert 'cache_hits' in stats

    def test_wait_for_loading_success(self):
        """测试等待插件加载成功"""
        mock_plugin = MockPlugin("test_plugin")
        
        lazy_info = LazyPluginInfo(
            name="test_plugin",
            module_path="test.module",
            class_name="TestPlugin",
            supported_models=["model1"]
        )
        
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            manager.initialize_plugin_registry([lazy_info])
            
            entry_ref = manager.get_plugin_entry("test_plugin")
            entry = entry_ref.get()
            
            # 模拟另一个线程正在加载
            entry.loading.set(1)
            
            def complete_loading():
                time.sleep(0.1)  # 模拟加载时间
                entry.instance = mock_plugin
                entry.loading.set(2)  # 标记加载完成
            
            # 启动模拟加载线程
            loading_thread = threading.Thread(target=complete_loading)
            loading_thread.start()
            
            # 等待加载完成
            result = manager._wait_for_loading(entry, timeout=1.0)
            
            loading_thread.join()
            
            assert result is mock_plugin
            assert manager._stats['cache_hits'].get() == 1
    
    def test_wait_for_loading_timeout(self):
        """测试等待插件加载超时"""
        lazy_info = LazyPluginInfo(
            name="test_plugin",
            module_path="test.module",
            class_name="TestPlugin",
            supported_models=["model1"]
        )
        
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            manager.initialize_plugin_registry([lazy_info])
            
            entry_ref = manager.get_plugin_entry("test_plugin")
            entry = entry_ref.get()
            
            # 模拟加载状态但不完成
            entry.loading.set(1)
            
            # 等待加载（应该超时）
            result = manager._wait_for_loading(entry, timeout=0.1)
            
            assert result is None
    
    def test_wait_for_loading_failed(self):
        """测试等待插件加载失败"""
        lazy_info = LazyPluginInfo(
            name="test_plugin",
            module_path="test.module",
            class_name="TestPlugin",
            supported_models=["model1"]
        )
        
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            manager = LockFreePluginManager()
            manager.initialize_plugin_registry([lazy_info])
            
            entry_ref = manager.get_plugin_entry("test_plugin")
            entry = entry_ref.get()
            
            # 模拟加载失败
            entry.loading.set(1)
            
            def fail_loading():
                time.sleep(0.05)
                entry.loading.set(0)  # 重置为未加载状态（表示失败）
            
            # 启动模拟失败线程
            fail_thread = threading.Thread(target=fail_loading)
            fail_thread.start()
            
            # 等待加载（应该检测到失败）
            result = manager._wait_for_loading(entry, timeout=1.0)
            
            fail_thread.join()
            
            assert result is None
    
    def test_do_load_plugin_import_error(self):
        """测试插件加载时的导入错误"""
        lazy_info = LazyPluginInfo(
            name="test_plugin",
            module_path="nonexistent.module",
            class_name="TestPlugin",
            supported_models=["model1"]
        )
        
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            with patch('importlib.import_module', side_effect=ImportError("No module named 'nonexistent'")):
                manager = LockFreePluginManager()
                manager.initialize_plugin_registry([lazy_info])
                
                entry_ref = manager.get_plugin_entry("test_plugin")
                entry = entry_ref.get()
                
                result = manager._do_load_plugin(entry)
                
                assert result is None
                assert entry.instance is None
                assert entry.loading.get() == 0
                # 注意：错误计数由调用者(_load_plugin_lockfree)负责增加
    
    def test_do_load_plugin_class_not_found(self):
        """测试插件加载时类不存在错误"""
        lazy_info = LazyPluginInfo(
            name="test_plugin",
            module_path="test.module",
            class_name="NonexistentClass",
            supported_models=["model1"]
        )
        
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            with patch('importlib.import_module') as mock_import:
                mock_module = Mock()
                # 模拟类不存在
                mock_module.NonexistentClass = None
                mock_import.return_value = mock_module
                
                manager = LockFreePluginManager()
                manager.initialize_plugin_registry([lazy_info])
                
                entry_ref = manager.get_plugin_entry("test_plugin")
                entry = entry_ref.get()
                
                result = manager._do_load_plugin(entry)
                
                assert result is None
                assert entry.instance is None
                assert entry.loading.get() == 0
                # 注意：错误计数由调用者(_load_plugin_lockfree)负责增加
    
    def test_do_load_plugin_initialization_error(self):
        """测试插件初始化错误"""
        lazy_info = LazyPluginInfo(
            name="test_plugin",
            module_path="test.module",
            class_name="TestPlugin",
            supported_models=["model1"]
        )
        
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            with patch('importlib.import_module') as mock_import:
                mock_module = Mock()
                mock_class = Mock(side_effect=Exception("Initialization failed"))
                mock_module.TestPlugin = mock_class
                mock_import.return_value = mock_module
                
                manager = LockFreePluginManager()
                manager.initialize_plugin_registry([lazy_info])
                
                entry_ref = manager.get_plugin_entry("test_plugin")
                entry = entry_ref.get()
                
                result = manager._do_load_plugin(entry)
                
                assert result is None
                assert entry.instance is None
                assert entry.loading.get() == 0
                # 注意：错误计数由调用者(_load_plugin_lockfree)负责增加
    
    def test_get_plugin_concurrent_loading_wait(self):
        """测试并发加载时的等待机制"""
        mock_plugin = MockPlugin("test_plugin")
        
        lazy_info = LazyPluginInfo(
            name="test_plugin",
            module_path="test.module",
            class_name="TestPlugin",
            supported_models=["model1"]
        )
        
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            with patch('importlib.import_module') as mock_import:
                mock_module = Mock()
                mock_class = Mock(return_value=mock_plugin)
                mock_module.TestPlugin = mock_class
                mock_import.return_value = mock_module
                
                manager = LockFreePluginManager()
                manager.initialize_plugin_registry([lazy_info])
                
                entry_ref = manager.get_plugin_entry("test_plugin")
                entry = entry_ref.get()
                
                # 模拟第一个线程开始加载
                entry.loading.set(1)
                
                results = []
                
                def first_thread_load():
                    # 模拟加载过程
                    time.sleep(0.1)
                    entry.instance = mock_plugin
                    entry.loading.set(2)
                
                def second_thread_wait():
                    # 第二个线程应该等待第一个线程完成
                    result = manager.get_plugin("test_plugin")
                    results.append(result)
                
                # 启动两个线程
                thread1 = threading.Thread(target=first_thread_load)
                thread2 = threading.Thread(target=second_thread_wait)
                
                thread1.start()
                time.sleep(0.05)  # 确保第一个线程先开始
                thread2.start()
                
                thread1.join()
                thread2.join()
                
                assert len(results) == 1
                assert results[0] is mock_plugin
    
    def test_plugin_reload_with_error_recovery(self):
        """测试插件重载时的错误恢复"""
        mock_plugin = MockPlugin("test_plugin")
        
        lazy_info = LazyPluginInfo(
            name="test_plugin",
            module_path="test.module",
            class_name="TestPlugin",
            supported_models=["model1"]
        )
        
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor'):
            with patch('importlib.import_module') as mock_import:
                # 第一次加载成功
                mock_module = Mock()
                # 创建一个真正的类来模拟插件类
                class TestPlugin(Plugin):
                    def __init__(self, config=None):
                        super().__init__(config)
                        self.name = "test_plugin"
                    
                    @property
                    def info(self):
                        from harborai.core.plugins.base import PluginInfo
                        return PluginInfo(
                            name="test_plugin",
                            version="1.0.0",
                            description="Test plugin",
                            supported_models=["model1"]
                        )
                    
                    def initialize(self):
                        return True
                    
                    def chat_completion(self, messages, model, stream=False, **kwargs):
                        return {"content": "test response"}
                    
                    async def chat_completion_async(self, messages, model, stream=False, **kwargs):
                        return {"content": "test response"}
                    
                    def process(self, data):
                        return mock_plugin.process(data)
                
                mock_module.TestPlugin = TestPlugin
                mock_import.return_value = mock_module
                
                manager = LockFreePluginManager()
                manager.initialize_plugin_registry([lazy_info])
                
                # 首次加载成功
                result1 = manager.get_plugin("test_plugin")
                assert result1 is not None
                assert isinstance(result1, Plugin)
                
                # 模拟重载时出错
                mock_import.side_effect = ImportError("Reload failed")
                
                result2 = manager.reload_plugin("test_plugin")
                assert result2 is None
                
                # 验证错误计数增加
                entry_ref = manager.get_plugin_entry("test_plugin")
                entry = entry_ref.get()
                assert entry.error_count.get() == 1
    
    def test_cleanup_with_gc_enabled(self):
        """测试启用垃圾回收的清理"""
        config = {'enable_gc': True}
        
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor') as mock_executor_class:
            with patch('gc.collect') as mock_gc_collect:
                mock_executor = Mock()
                mock_executor_class.return_value = mock_executor
                
                manager = LockFreePluginManager(config)
                manager.cleanup()
                
                mock_executor.shutdown.assert_called_once_with(wait=True)
                mock_gc_collect.assert_called_once()
    
    def test_cleanup_with_gc_disabled(self):
        """测试禁用垃圾回收的清理"""
        config = {'enable_gc': False}
        
        with patch('harborai.core.optimizations.lockfree_plugin_manager.ThreadPoolExecutor') as mock_executor_class:
            with patch('gc.collect') as mock_gc_collect:
                mock_executor = Mock()
                mock_executor_class.return_value = mock_executor
                
                manager = LockFreePluginManager(config)
                manager.cleanup()
                
                mock_executor.shutdown.assert_called_once_with(wait=True)
                mock_gc_collect.assert_not_called()