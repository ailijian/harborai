#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数缓存层单元测试
"""

import time
import json
import threading
import unittest
from unittest.mock import patch, MagicMock

from harborai.core.parameter_cache import (
    ParameterCacheEntry,
    SchemaCache,
    ConfigCache,
    ParameterCacheManager,
    get_parameter_cache_manager,
    create_parameter_cache_manager
)


class TestParameterCacheEntry(unittest.TestCase):
    """测试参数缓存条目"""
    
    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        entry = ParameterCacheEntry(
            key="test_key",
            value={"test": "data"},
            created_at=time.time(),
            ttl=3600,
            cache_type="test"
        )
        
        self.assertEqual(entry.key, "test_key")
        self.assertEqual(entry.value, {"test": "data"})
        self.assertEqual(entry.ttl, 3600)
        self.assertEqual(entry.cache_type, "test")
        self.assertEqual(entry.access_count, 0)
        self.assertIsNotNone(entry.last_accessed)
    
    def test_cache_entry_expiration(self):
        """测试缓存条目过期检查"""
        # 未过期的条目
        entry = ParameterCacheEntry(
            key="test_key",
            value={"test": "data"},
            created_at=time.time(),
            ttl=3600
        )
        self.assertFalse(entry.is_expired)
        
        # 过期的条目
        expired_entry = ParameterCacheEntry(
            key="expired_key",
            value={"test": "data"},
            created_at=time.time() - 7200,  # 2小时前
            ttl=3600  # 1小时TTL
        )
        self.assertTrue(expired_entry.is_expired)
        
        # 永不过期的条目
        permanent_entry = ParameterCacheEntry(
            key="permanent_key",
            value={"test": "data"},
            created_at=time.time() - 7200,
            ttl=0  # 永不过期
        )
        self.assertFalse(permanent_entry.is_expired)
    
    def test_cache_entry_touch(self):
        """测试缓存条目访问更新"""
        entry = ParameterCacheEntry(
            key="test_key",
            value={"test": "data"},
            created_at=time.time(),
            ttl=3600
        )
        
        initial_access_count = entry.access_count
        initial_last_accessed = entry.last_accessed
        
        time.sleep(0.01)  # 确保时间差异
        entry.touch()
        
        self.assertEqual(entry.access_count, initial_access_count + 1)
        self.assertGreater(entry.last_accessed, initial_last_accessed)


class TestSchemaCache(unittest.TestCase):
    """测试Schema缓存"""
    
    def setUp(self):
        """设置测试环境"""
        self.cache = SchemaCache(max_size=10, default_ttl=3600)
        self.test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        self.converted_schema = {
            "name": ("str", True),
            "age": ("int", False)
        }
    
    def test_schema_cache_key_generation(self):
        """测试Schema缓存键生成"""
        key1 = self.cache._generate_key(self.test_schema)
        key2 = self.cache._generate_key(self.test_schema)
        
        # 相同Schema应生成相同键
        self.assertEqual(key1, key2)
        
        # 不同Schema应生成不同键
        different_schema = {
            "type": "object",
            "properties": {
                "email": {"type": "string"}
            }
        }
        key3 = self.cache._generate_key(different_schema)
        self.assertNotEqual(key1, key3)
    
    def test_schema_cache_normalization(self):
        """测试Schema标准化"""
        schema_with_metadata = {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Test Schema",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        normalized = self.cache._normalize_schema(schema_with_metadata)
        
        # 元数据字段应被移除
        self.assertNotIn("$schema", normalized)
        self.assertNotIn("title", normalized)
        self.assertIn("type", normalized)
        self.assertIn("properties", normalized)
    
    def test_schema_cache_set_and_get(self):
        """测试Schema缓存设置和获取"""
        # 初始状态应为空
        result = self.cache.get_converted_schema(self.test_schema)
        self.assertIsNone(result)
        
        # 设置缓存
        self.cache.set_converted_schema(self.test_schema, self.converted_schema)
        
        # 获取缓存
        result = self.cache.get_converted_schema(self.test_schema)
        self.assertEqual(result, self.converted_schema)
    
    def test_schema_cache_expiration(self):
        """测试Schema缓存过期"""
        # 设置短TTL的缓存
        self.cache.set_converted_schema(self.test_schema, self.converted_schema, ttl=1)
        
        # 立即获取应该成功
        result = self.cache.get_converted_schema(self.test_schema)
        self.assertEqual(result, self.converted_schema)
        
        # 等待过期
        time.sleep(1.1)
        
        # 过期后获取应该返回None
        result = self.cache.get_converted_schema(self.test_schema)
        self.assertIsNone(result)
    
    def test_schema_cache_lru_eviction(self):
        """测试Schema缓存LRU淘汰"""
        # 填满缓存
        for i in range(self.cache.max_size):
            schema = {"type": "string", "pattern": f"test_{i}"}
            converted = {"pattern": f"test_{i}"}
            self.cache.set_converted_schema(schema, converted)
        
        # 添加一个新的条目，应该触发LRU淘汰
        new_schema = {"type": "string", "pattern": "new_test"}
        new_converted = {"pattern": "new_test"}
        self.cache.set_converted_schema(new_schema, new_converted)
        
        # 缓存大小应该保持在最大值
        self.assertEqual(len(self.cache._cache), self.cache.max_size)
        
        # 新条目应该存在
        result = self.cache.get_converted_schema(new_schema)
        self.assertEqual(result, new_converted)
    
    def test_schema_cache_clear_expired(self):
        """测试清理过期Schema缓存"""
        # 添加一些缓存条目，部分过期
        self.cache.set_converted_schema(self.test_schema, self.converted_schema, ttl=3600)
        
        expired_schema = {"type": "string", "pattern": "expired"}
        self.cache.set_converted_schema(expired_schema, {"pattern": "expired"}, ttl=1)
        
        time.sleep(1.1)
        
        # 清理过期条目
        expired_count = self.cache.clear_expired()
        self.assertEqual(expired_count, 1)
        
        # 未过期的条目应该仍然存在
        result = self.cache.get_converted_schema(self.test_schema)
        self.assertEqual(result, self.converted_schema)
    
    def test_schema_cache_stats(self):
        """测试Schema缓存统计"""
        stats = self.cache.get_stats()
        
        self.assertIn('size', stats)
        self.assertIn('max_size', stats)
        self.assertIn('total_access', stats)
        self.assertIn('hit_rate', stats)
        self.assertIn('cache_type', stats)
        self.assertEqual(stats['cache_type'], 'schema')


class TestConfigCache(unittest.TestCase):
    """测试配置缓存"""
    
    def setUp(self):
        """设置测试环境"""
        self.cache = ConfigCache(max_size=10, default_ttl=1800)
        self.test_config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        self.processed_config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000,
            "processed": True
        }
    
    def test_config_cache_key_generation(self):
        """测试配置缓存键生成"""
        key1 = self.cache._generate_key(self.test_config)
        key2 = self.cache._generate_key(self.test_config)
        
        # 相同配置应生成相同键
        self.assertEqual(key1, key2)
        
        # 添加不影响配置的字段，应生成相同键
        config_with_metadata = self.test_config.copy()
        config_with_metadata.update({
            "trace_id": "test_trace",
            "timestamp": time.time(),
            "user_id": "test_user"
        })
        key3 = self.cache._generate_key(config_with_metadata)
        self.assertEqual(key1, key3)
        
        # 不同配置应生成不同键
        different_config = {"model": "gpt-4", "temperature": 0.5}
        key4 = self.cache._generate_key(different_config)
        self.assertNotEqual(key1, key4)
    
    def test_config_cache_set_and_get(self):
        """测试配置缓存设置和获取"""
        # 初始状态应为空
        result = self.cache.get_config(self.test_config)
        self.assertIsNone(result)
        
        # 设置缓存
        self.cache.set_config(self.test_config, self.processed_config)
        
        # 获取缓存
        result = self.cache.get_config(self.test_config)
        self.assertEqual(result, self.processed_config)
    
    def test_config_cache_expiration(self):
        """测试配置缓存过期"""
        # 设置短TTL的缓存
        self.cache.set_config(self.test_config, self.processed_config, ttl=1)
        
        # 立即获取应该成功
        result = self.cache.get_config(self.test_config)
        self.assertEqual(result, self.processed_config)
        
        # 等待过期
        time.sleep(1.1)
        
        # 过期后获取应该返回None
        result = self.cache.get_config(self.test_config)
        self.assertIsNone(result)
    
    def test_config_cache_stats(self):
        """测试配置缓存统计"""
        stats = self.cache.get_stats()
        
        self.assertIn('size', stats)
        self.assertIn('max_size', stats)
        self.assertIn('total_access', stats)
        self.assertIn('hit_rate', stats)
        self.assertIn('cache_type', stats)
        self.assertEqual(stats['cache_type'], 'config')


class TestParameterCacheManager(unittest.TestCase):
    """测试参数缓存管理器"""
    
    def setUp(self):
        """设置测试环境"""
        self.manager = ParameterCacheManager()
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        self.assertIsInstance(self.manager.schema_cache, SchemaCache)
        self.assertIsInstance(self.manager.config_cache, ConfigCache)
        self.assertGreater(self.manager._cleanup_interval, 0)
    
    def test_manager_custom_config(self):
        """测试自定义配置的管理器"""
        schema_config = {"max_size": 500, "default_ttl": 7200}
        config_config = {"max_size": 200, "default_ttl": 900}
        
        manager = ParameterCacheManager(
            schema_cache_config=schema_config,
            config_cache_config=config_config
        )
        
        self.assertEqual(manager.schema_cache.max_size, 500)
        self.assertEqual(manager.schema_cache.default_ttl, 7200)
        self.assertEqual(manager.config_cache.max_size, 200)
        self.assertEqual(manager.config_cache.default_ttl, 900)
    
    def test_manager_cleanup_expired(self):
        """测试管理器清理过期缓存"""
        # 添加一些缓存条目
        test_schema = {"type": "string"}
        test_config = {"model": "test"}
        
        self.manager.schema_cache.set_converted_schema(test_schema, {"type": "str"}, ttl=1)
        self.manager.config_cache.set_config(test_config, {"model": "test", "processed": True}, ttl=1)
        
        time.sleep(1.1)
        
        # 清理过期条目
        result = self.manager.cleanup_expired()
        
        self.assertIn('schema_expired', result)
        self.assertIn('config_expired', result)
    
    def test_manager_comprehensive_stats(self):
        """测试管理器综合统计"""
        stats = self.manager.get_comprehensive_stats()
        
        self.assertIn('schema_cache', stats)
        self.assertIn('config_cache', stats)
        self.assertIn('last_cleanup', stats)
        self.assertIn('cleanup_interval', stats)
    
    def test_manager_clear_all_caches(self):
        """测试清空所有缓存"""
        # 添加一些缓存条目
        test_schema = {"type": "string"}
        test_config = {"model": "test"}
        
        self.manager.schema_cache.set_converted_schema(test_schema, {"type": "str"})
        self.manager.config_cache.set_config(test_config, {"model": "test", "processed": True})
        
        # 确认缓存不为空
        self.assertGreater(len(self.manager.schema_cache._cache), 0)
        self.assertGreater(len(self.manager.config_cache._cache), 0)
        
        # 清空所有缓存
        self.manager.clear_all_caches()
        
        # 确认缓存已清空
        self.assertEqual(len(self.manager.schema_cache._cache), 0)
        self.assertEqual(len(self.manager.config_cache._cache), 0)
    
    def test_manager_set_cleanup_interval(self):
        """测试设置清理间隔"""
        # 设置正常间隔
        self.manager.set_cleanup_interval(600)
        self.assertEqual(self.manager._cleanup_interval, 600)
        
        # 设置过小的间隔，应该被限制为最小值
        self.manager.set_cleanup_interval(30)
        self.assertEqual(self.manager._cleanup_interval, 60)


class TestGlobalFunctions(unittest.TestCase):
    """测试全局函数"""
    
    def test_get_parameter_cache_manager(self):
        """测试获取全局参数缓存管理器"""
        manager1 = get_parameter_cache_manager()
        manager2 = get_parameter_cache_manager()
        
        # 应该返回同一个实例（单例模式）
        self.assertIs(manager1, manager2)
        self.assertIsInstance(manager1, ParameterCacheManager)
    
    def test_create_parameter_cache_manager(self):
        """测试创建参数缓存管理器"""
        schema_config = {"max_size": 100}
        config_config = {"max_size": 50}
        
        manager = create_parameter_cache_manager(
            schema_cache_config=schema_config,
            config_cache_config=config_config
        )
        
        self.assertIsInstance(manager, ParameterCacheManager)
        self.assertEqual(manager.schema_cache.max_size, 100)
        self.assertEqual(manager.config_cache.max_size, 50)


class TestThreadSafety(unittest.TestCase):
    """测试线程安全性"""
    
    def test_schema_cache_thread_safety(self):
        """测试Schema缓存线程安全"""
        cache = SchemaCache(max_size=100)
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(10):
                    schema = {"type": "string", "pattern": f"thread_{thread_id}_{i}"}
                    converted = {"pattern": f"thread_{thread_id}_{i}"}
                    
                    cache.set_converted_schema(schema, converted)
                    result = cache.get_converted_schema(schema)
                    results.append((thread_id, i, result == converted))
            except Exception as e:
                errors.append((thread_id, str(e)))
        
        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 检查结果
        self.assertEqual(len(errors), 0, f"线程安全测试出现错误: {errors}")
        self.assertEqual(len(results), 50)  # 5个线程 * 10次操作
        
        # 所有操作都应该成功
        for thread_id, i, success in results:
            self.assertTrue(success, f"线程 {thread_id} 操作 {i} 失败")


if __name__ == '__main__':
    unittest.main()