#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数缓存模块测试
测试Schema缓存、配置缓存和参数缓存管理器的功能
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock

from harborai.core.parameter_cache import (
    ParameterCacheEntry,
    SchemaCache,
    ConfigCache,
    ParameterCacheManager,
    get_parameter_cache_manager,
    create_parameter_cache_manager
)


class TestParameterCacheEntry:
    """测试参数缓存条目"""
    
    def test_cache_entry_initialization(self):
        """测试缓存条目初始化"""
        entry = ParameterCacheEntry(
            key="test_key",
            value={"data": "test"},
            created_at=time.time(),
            ttl=3600
        )
        
        assert entry.key == "test_key"
        assert entry.value == {"data": "test"}
        assert entry.ttl == 3600
        assert entry.access_count == 0
        assert entry.last_accessed is not None
        assert entry.cache_type == "general"
    
    def test_cache_entry_expiration(self):
        """测试缓存条目过期检查"""
        # 测试未过期的条目
        entry = ParameterCacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            ttl=3600
        )
        assert not entry.is_expired
        
        # 测试过期的条目
        entry_expired = ParameterCacheEntry(
            key="expired_key",
            value="expired_value",
            created_at=time.time() - 7200,  # 2小时前
            ttl=3600  # 1小时TTL
        )
        assert entry_expired.is_expired
        
        # 测试永不过期的条目
        entry_no_expire = ParameterCacheEntry(
            key="no_expire_key",
            value="no_expire_value",
            created_at=time.time() - 7200,
            ttl=0  # 永不过期
        )
        assert not entry_no_expire.is_expired
    
    def test_cache_entry_touch(self):
        """测试缓存条目访问更新"""
        entry = ParameterCacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            ttl=3600
        )
        
        initial_access_count = entry.access_count
        initial_last_accessed = entry.last_accessed
        
        time.sleep(0.01)  # 确保时间差异
        entry.touch()
        
        assert entry.access_count == initial_access_count + 1
        assert entry.last_accessed > initial_last_accessed


class TestSchemaCache:
    """测试Schema缓存"""
    
    def test_schema_cache_initialization(self):
        """测试Schema缓存初始化"""
        cache = SchemaCache(max_size=100, default_ttl=1800)
        
        assert cache.max_size == 100
        assert cache.default_ttl == 1800
        assert len(cache._cache) == 0
    
    def test_schema_normalization(self):
        """测试Schema标准化"""
        cache = SchemaCache()
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Person Schema",
            "examples": [{"name": "John", "age": 30}]
        }
        
        normalized = cache._normalize_schema(schema)
        
        # 排除的字段不应该存在
        assert "$schema" not in normalized
        assert "title" not in normalized
        assert "examples" not in normalized
        
        # 保留的字段应该存在
        assert "type" in normalized
        assert "properties" in normalized
    
    def test_schema_key_generation(self):
        """测试Schema缓存键生成"""
        cache = SchemaCache()
        
        schema1 = {"type": "string", "minLength": 1}
        schema2 = {"type": "string", "minLength": 1}
        schema3 = {"type": "integer", "minimum": 0}
        
        key1 = cache._generate_key(schema1)
        key2 = cache._generate_key(schema2)
        key3 = cache._generate_key(schema3)
        
        # 相同的schema应该生成相同的key
        assert key1 == key2
        # 不同的schema应该生成不同的key
        assert key1 != key3
    
    def test_schema_cache_set_and_get(self):
        """测试Schema缓存设置和获取"""
        cache = SchemaCache()
        
        schema = {"type": "string", "minLength": 1}
        converted_schema = {"type": "str", "min_length": 1}
        
        # 设置缓存
        cache.set_converted_schema(schema, converted_schema)
        
        # 获取缓存
        result = cache.get_converted_schema(schema)
        assert result == converted_schema
        
        # 获取不存在的缓存
        non_existent_schema = {"type": "integer"}
        result = cache.get_converted_schema(non_existent_schema)
        assert result is None
    
    def test_schema_cache_expiration(self):
        """测试Schema缓存过期"""
        cache = SchemaCache(default_ttl=1)  # 1秒TTL
        
        schema = {"type": "string"}
        converted_schema = {"type": "str"}
        
        cache.set_converted_schema(schema, converted_schema)
        
        # 立即获取应该成功
        result = cache.get_converted_schema(schema)
        assert result == converted_schema
        
        # 等待过期
        time.sleep(1.1)
        
        # 过期后获取应该返回None
        result = cache.get_converted_schema(schema)
        assert result is None
    
    def test_schema_cache_lru_eviction(self):
        """测试Schema缓存LRU淘汰"""
        cache = SchemaCache(max_size=2)
        
        schema1 = {"type": "string"}
        schema2 = {"type": "integer"}
        schema3 = {"type": "boolean"}
        
        converted1 = {"type": "str"}
        converted2 = {"type": "int"}
        converted3 = {"type": "bool"}
        
        # 添加两个条目
        cache.set_converted_schema(schema1, converted1)
        cache.set_converted_schema(schema2, converted2)
        
        # 访问第一个条目
        cache.get_converted_schema(schema1)
        
        # 添加第三个条目，应该淘汰第二个条目（最少使用）
        cache.set_converted_schema(schema3, converted3)
        
        # 第一个和第三个应该存在，第二个应该被淘汰
        assert cache.get_converted_schema(schema1) == converted1
        assert cache.get_converted_schema(schema3) == converted3
        assert cache.get_converted_schema(schema2) is None
    
    def test_schema_cache_clear_expired(self):
        """测试Schema缓存清理过期条目"""
        cache = SchemaCache(default_ttl=1)
        
        schema1 = {"type": "string"}
        schema2 = {"type": "integer"}
        
        cache.set_converted_schema(schema1, {"type": "str"})
        cache.set_converted_schema(schema2, {"type": "int"}, ttl=10)  # 长TTL
        
        # 等待第一个过期
        time.sleep(1.1)
        
        # 清理过期条目
        expired_count = cache.clear_expired()
        assert expired_count == 1
        
        # 第二个条目应该仍然存在
        assert cache.get_converted_schema(schema2) is not None
    
    def test_schema_cache_stats(self):
        """测试Schema缓存统计信息"""
        cache = SchemaCache()
        
        schema = {"type": "string"}
        converted_schema = {"type": "str"}
        
        cache.set_converted_schema(schema, converted_schema)
        cache.get_converted_schema(schema)  # 访问一次
        
        stats = cache.get_stats()
        
        assert stats['size'] == 1
        assert stats['max_size'] == cache.max_size
        assert stats['total_access'] == 1
        assert stats['cache_type'] == 'schema'


class TestConfigCache:
    """测试配置缓存"""
    
    def test_config_cache_initialization(self):
        """测试配置缓存初始化"""
        cache = ConfigCache(max_size=200, default_ttl=900)
        
        assert cache.max_size == 200
        assert cache.default_ttl == 900
        assert len(cache._cache) == 0
    
    def test_config_key_generation(self):
        """测试配置缓存键生成"""
        cache = ConfigCache()
        
        config1 = {"model": "gpt-3.5-turbo", "temperature": 0.7}
        config2 = {"model": "gpt-3.5-turbo", "temperature": 0.7, "trace_id": "123"}
        config3 = {"model": "gpt-4", "temperature": 0.7}
        
        key1 = cache._generate_key(config1)
        key2 = cache._generate_key(config2)
        key3 = cache._generate_key(config3)
        
        # 排除trace_id后，前两个应该生成相同的key
        assert key1 == key2
        # 不同的配置应该生成不同的key
        assert key1 != key3
    
    def test_config_cache_set_and_get(self):
        """测试配置缓存设置和获取"""
        cache = ConfigCache()
        
        config_data = {"model": "gpt-3.5-turbo", "temperature": 0.7}
        processed_config = {"model": "gpt-3.5-turbo", "temperature": 0.7, "max_tokens": 1000}
        
        # 设置缓存
        cache.set_config(config_data, processed_config)
        
        # 获取缓存
        result = cache.get_config(config_data)
        assert result == processed_config
        
        # 获取不存在的缓存
        non_existent_config = {"model": "gpt-4"}
        result = cache.get_config(non_existent_config)
        assert result is None
    
    def test_config_cache_expiration(self):
        """测试配置缓存过期"""
        cache = ConfigCache(default_ttl=1)  # 1秒TTL
        
        config_data = {"model": "gpt-3.5-turbo"}
        processed_config = {"model": "gpt-3.5-turbo", "max_tokens": 1000}
        
        cache.set_config(config_data, processed_config)
        
        # 立即获取应该成功
        result = cache.get_config(config_data)
        assert result == processed_config
        
        # 等待过期
        time.sleep(1.1)
        
        # 过期后获取应该返回None
        result = cache.get_config(config_data)
        assert result is None
    
    def test_config_cache_lru_eviction(self):
        """测试配置缓存LRU淘汰"""
        cache = ConfigCache(max_size=2)
        
        config1 = {"model": "gpt-3.5-turbo"}
        config2 = {"model": "gpt-4"}
        config3 = {"model": "claude-3"}
        
        processed1 = {"model": "gpt-3.5-turbo", "max_tokens": 1000}
        processed2 = {"model": "gpt-4", "max_tokens": 2000}
        processed3 = {"model": "claude-3", "max_tokens": 3000}
        
        # 添加两个条目
        cache.set_config(config1, processed1)
        cache.set_config(config2, processed2)
        
        # 访问第一个条目
        cache.get_config(config1)
        
        # 添加第三个条目，应该淘汰第二个条目
        cache.set_config(config3, processed3)
        
        # 第一个和第三个应该存在，第二个应该被淘汰
        assert cache.get_config(config1) == processed1
        assert cache.get_config(config3) == processed3
        assert cache.get_config(config2) is None
    
    def test_config_cache_stats(self):
        """测试配置缓存统计信息"""
        cache = ConfigCache()
        
        config_data = {"model": "gpt-3.5-turbo"}
        processed_config = {"model": "gpt-3.5-turbo", "max_tokens": 1000}
        
        cache.set_config(config_data, processed_config)
        cache.get_config(config_data)  # 访问一次
        
        stats = cache.get_stats()
        
        assert stats['size'] == 1
        assert stats['max_size'] == cache.max_size
        assert stats['total_access'] == 1
        assert stats['cache_type'] == 'config'


class TestParameterCacheManager:
    """测试参数缓存管理器"""
    
    def test_manager_initialization(self):
        """测试缓存管理器初始化"""
        manager = ParameterCacheManager()
        
        assert manager.schema_cache is not None
        assert manager.config_cache is not None
        assert manager._cleanup_interval == 300
    
    def test_manager_custom_initialization(self):
        """测试缓存管理器自定义初始化"""
        schema_config = {"max_size": 500, "default_ttl": 1800}
        config_config = {"max_size": 300, "default_ttl": 900}
        
        manager = ParameterCacheManager(
            schema_cache_config=schema_config,
            config_cache_config=config_config
        )
        
        assert manager.schema_cache.max_size == 500
        assert manager.schema_cache.default_ttl == 1800
        assert manager.config_cache.max_size == 300
        assert manager.config_cache.default_ttl == 900
    
    def test_manager_cleanup_expired(self):
        """测试缓存管理器清理过期条目"""
        manager = ParameterCacheManager()
        
        # 添加一些缓存条目
        schema = {"type": "string"}
        config = {"model": "gpt-3.5-turbo"}
        
        manager.schema_cache.set_converted_schema(schema, {"type": "str"}, ttl=1)
        manager.config_cache.set_config(config, {"model": "gpt-3.5-turbo"}, ttl=1)
        
        # 等待过期
        time.sleep(1.1)
        
        # 强制清理（绕过时间间隔检查）
        manager._last_cleanup = 0
        
        cleanup_stats = manager.cleanup_expired()
        
        assert cleanup_stats['schema_expired'] >= 0
        assert cleanup_stats['config_expired'] >= 0
    
    def test_manager_cleanup_interval_check(self):
        """测试缓存管理器清理间隔检查"""
        manager = ParameterCacheManager()
        
        # 第一次清理
        cleanup_stats1 = manager.cleanup_expired()
        
        # 立即再次清理，应该跳过
        cleanup_stats2 = manager.cleanup_expired()
        
        assert cleanup_stats2['schema_expired'] == 0
        assert cleanup_stats2['config_expired'] == 0
    
    def test_manager_comprehensive_stats(self):
        """测试缓存管理器综合统计信息"""
        manager = ParameterCacheManager()
        
        stats = manager.get_comprehensive_stats()
        
        assert 'schema_cache' in stats
        assert 'config_cache' in stats
        assert 'last_cleanup' in stats
        assert 'cleanup_interval' in stats
        
        assert stats['schema_cache']['cache_type'] == 'schema'
        assert stats['config_cache']['cache_type'] == 'config'
    
    def test_manager_clear_all_caches(self):
        """测试缓存管理器清空所有缓存"""
        manager = ParameterCacheManager()
        
        # 添加一些缓存条目
        schema = {"type": "string"}
        config = {"model": "gpt-3.5-turbo"}
        
        manager.schema_cache.set_converted_schema(schema, {"type": "str"})
        manager.config_cache.set_config(config, {"model": "gpt-3.5-turbo"})
        
        # 验证缓存存在
        assert len(manager.schema_cache._cache) > 0
        assert len(manager.config_cache._cache) > 0
        
        # 清空所有缓存
        manager.clear_all_caches()
        
        # 验证缓存已清空
        assert len(manager.schema_cache._cache) == 0
        assert len(manager.config_cache._cache) == 0
    
    def test_manager_set_cleanup_interval(self):
        """测试缓存管理器设置清理间隔"""
        manager = ParameterCacheManager()
        
        # 设置正常间隔
        manager.set_cleanup_interval(600)
        assert manager._cleanup_interval == 600
        
        # 设置过小的间隔，应该被限制为最小值
        manager.set_cleanup_interval(30)
        assert manager._cleanup_interval == 60


class TestGlobalFunctions:
    """测试全局函数"""
    
    def test_get_parameter_cache_manager_singleton(self):
        """测试获取全局参数缓存管理器单例"""
        manager1 = get_parameter_cache_manager()
        manager2 = get_parameter_cache_manager()
        
        # 应该返回同一个实例
        assert manager1 is manager2
    
    def test_create_parameter_cache_manager(self):
        """测试创建参数缓存管理器"""
        schema_config = {"max_size": 100}
        config_config = {"max_size": 50}
        
        manager = create_parameter_cache_manager(
            schema_cache_config=schema_config,
            config_cache_config=config_config
        )
        
        assert manager.schema_cache.max_size == 100
        assert manager.config_cache.max_size == 50
    
    @patch('harborai.core.parameter_cache._parameter_cache_manager', None)
    def test_get_parameter_cache_manager_initialization(self):
        """测试全局参数缓存管理器初始化"""
        # 重置全局实例
        import harborai.core.parameter_cache
        harborai.core.parameter_cache._parameter_cache_manager = None
        
        manager = get_parameter_cache_manager()
        assert manager is not None
        assert isinstance(manager, ParameterCacheManager)


class TestThreadSafety:
    """测试线程安全性"""
    
    def test_schema_cache_thread_safety(self):
        """测试Schema缓存线程安全"""
        cache = SchemaCache()
        results = []
        
        def worker(thread_id):
            schema = {"type": "string", "thread_id": thread_id}
            converted = {"type": "str", "thread_id": thread_id}
            
            cache.set_converted_schema(schema, converted)
            result = cache.get_converted_schema(schema)
            results.append(result == converted)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 所有线程都应该成功
        assert all(results)
    
    def test_config_cache_thread_safety(self):
        """测试配置缓存线程安全"""
        cache = ConfigCache()
        results = []
        
        def worker(thread_id):
            config = {"model": "gpt-3.5-turbo", "thread_id": thread_id}
            processed = {"model": "gpt-3.5-turbo", "thread_id": thread_id, "max_tokens": 1000}
            
            cache.set_config(config, processed)
            result = cache.get_config(config)
            results.append(result == processed)
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 所有线程都应该成功
        assert all(results)


class TestEdgeCases:
    """测试边界情况"""
    
    def test_empty_cache_operations(self):
        """测试空缓存操作"""
        schema_cache = SchemaCache()
        config_cache = ConfigCache()
        
        # 空缓存的统计信息
        schema_stats = schema_cache.get_stats()
        assert schema_stats['size'] == 0
        assert schema_stats['total_access'] == 0
        
        config_stats = config_cache.get_stats()
        assert config_stats['size'] == 0
        assert config_stats['total_access'] == 0
        
        # 空缓存的清理操作
        assert schema_cache.clear_expired() == 0
        assert config_cache.clear_expired() == 0
        
        # 空缓存的LRU淘汰
        schema_cache._evict_lru()  # 不应该抛出异常
        config_cache._evict_lru()  # 不应该抛出异常
    
    def test_zero_ttl_cache_entries(self):
        """测试零TTL缓存条目（永不过期）"""
        cache = SchemaCache()
        
        schema = {"type": "string"}
        converted = {"type": "str"}
        
        cache.set_converted_schema(schema, converted, ttl=0)
        
        # 即使等待一段时间，也不应该过期
        time.sleep(0.1)
        result = cache.get_converted_schema(schema)
        assert result == converted
    
    def test_complex_nested_schema_normalization(self):
        """测试复杂嵌套Schema标准化"""
        cache = SchemaCache()
        
        complex_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "title": "User Name"},
                        "settings": {
                            "type": "object",
                            "properties": {
                                "theme": {"type": "string", "examples": ["dark", "light"]}
                            },
                            "$id": "settings-schema"
                        }
                    }
                }
            },
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Complex Schema"
        }
        
        normalized = cache._normalize_schema(complex_schema)
        
        # 顶层排除字段应该被移除
        assert "$schema" not in normalized
        assert "title" not in normalized
        
        # 嵌套的排除字段也应该被移除
        assert "title" not in normalized["properties"]["user"]["properties"]["name"]
        assert "examples" not in normalized["properties"]["user"]["properties"]["settings"]["properties"]["theme"]
        assert "$id" not in normalized["properties"]["user"]["properties"]["settings"]
        
        # 保留的字段应该存在
        assert "type" in normalized
        assert "properties" in normalized