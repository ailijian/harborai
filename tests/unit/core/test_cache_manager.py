"""缓存管理器测试

测试Token缓存、响应缓存和缓存管理器的功能。
"""

import asyncio
import json
import pytest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed

from harborai.core.cache_manager import (
    CacheEntry,
    TokenCache,
    ResponseCache,
    CacheManager,
    get_cache_manager,
    start_cache_manager,
    stop_cache_manager,
    _cache_manager
)


class TestCacheEntry:
    """测试缓存条目"""
    
    def test_cache_entry_creation(self):
        """测试缓存条目创建"""
        now = datetime.now()
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=now,
            ttl=300
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.created_at == now
        assert entry.ttl == 300
        assert entry.access_count == 0
        assert entry.last_accessed == now
    
    def test_cache_entry_expiration(self):
        """测试缓存条目过期检查"""
        # 测试未过期
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=datetime.now(),
            ttl=300
        )
        assert not entry.is_expired
        
        # 测试已过期
        past_time = datetime.now() - timedelta(seconds=400)
        expired_entry = CacheEntry(
            key="test",
            value="value",
            created_at=past_time,
            ttl=300
        )
        assert expired_entry.is_expired
        
        # 测试永不过期
        never_expire = CacheEntry(
            key="test",
            value="value",
            created_at=past_time,
            ttl=0
        )
        assert not never_expire.is_expired
    
    def test_cache_entry_touch(self):
        """测试缓存条目访问更新"""
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=datetime.now(),
            ttl=300
        )
        
        original_access_count = entry.access_count
        original_last_accessed = entry.last_accessed
        
        time.sleep(0.01)  # 确保时间差异
        entry.touch()
        
        assert entry.access_count == original_access_count + 1
        assert entry.last_accessed > original_last_accessed


class TestTokenCache:
    """测试Token缓存"""
    
    def test_token_cache_creation(self):
        """测试Token缓存创建"""
        cache = TokenCache(max_size=5000, default_ttl=1800)
        
        assert cache.max_size == 5000
        assert cache.default_ttl == 1800
        assert len(cache._cache) == 0
    
    def test_generate_key(self):
        """测试缓存键生成"""
        cache = TokenCache()
        
        key1 = cache._generate_key("hello world", "gpt-3.5-turbo")
        key2 = cache._generate_key("hello world", "gpt-3.5-turbo")
        key3 = cache._generate_key("hello world", "gpt-4")
        key4 = cache._generate_key("hello", "gpt-3.5-turbo")
        
        # 相同输入应生成相同键
        assert key1 == key2
        # 不同模型应生成不同键
        assert key1 != key3
        # 不同文本应生成不同键
        assert key1 != key4
    
    def test_token_count_operations(self):
        """测试Token计数操作"""
        cache = TokenCache()
        
        # 测试缓存未命中
        count = cache.get_token_count("hello", "gpt-3.5-turbo")
        assert count is None
        
        # 测试设置和获取
        cache.set_token_count("hello", "gpt-3.5-turbo", 5)
        count = cache.get_token_count("hello", "gpt-3.5-turbo")
        assert count == 5
        
        # 测试不同模型
        cache.set_token_count("hello", "gpt-4", 6)
        count_gpt4 = cache.get_token_count("hello", "gpt-4")
        count_gpt35 = cache.get_token_count("hello", "gpt-3.5-turbo")
        
        assert count_gpt4 == 6
        assert count_gpt35 == 5
    
    def test_token_cache_ttl(self):
        """测试Token缓存TTL"""
        cache = TokenCache(default_ttl=1)
        
        # 设置短TTL的缓存
        cache.set_token_count("test", "model", 10, ttl=1)
        
        # 立即获取应该成功
        count = cache.get_token_count("test", "model")
        assert count == 10
        
        # 等待过期
        time.sleep(1.1)
        
        # 过期后应该返回None
        count = cache.get_token_count("test", "model")
        assert count is None
    
    def test_token_cache_lru_eviction(self):
        """测试Token缓存LRU淘汰"""
        cache = TokenCache(max_size=2)
        
        # 添加两个条目
        cache.set_token_count("text1", "model", 10)
        cache.set_token_count("text2", "model", 20)
        
        # 访问第一个条目
        cache.get_token_count("text1", "model")
        
        # 添加第三个条目，应该淘汰text2
        cache.set_token_count("text3", "model", 30)
        
        assert cache.get_token_count("text1", "model") == 10
        assert cache.get_token_count("text2", "model") is None
        assert cache.get_token_count("text3", "model") == 30
    
    def test_clear_expired(self):
        """测试清理过期条目"""
        cache = TokenCache()
        
        # 添加正常条目
        cache.set_token_count("normal", "model", 10, ttl=3600)
        
        # 添加过期条目
        cache.set_token_count("expired", "model", 20, ttl=1)
        time.sleep(1.1)
        
        # 清理前应该有2个条目
        assert len(cache._cache) == 2
        
        # 清理过期条目
        expired_count = cache.clear_expired()
        
        assert expired_count == 1
        assert len(cache._cache) == 1
        assert cache.get_token_count("normal", "model") == 10
    
    def test_get_stats(self):
        """测试获取统计信息"""
        cache = TokenCache(max_size=100)
        
        # 空缓存统计
        stats = cache.get_stats()
        assert stats['size'] == 0
        assert stats['max_size'] == 100
        assert stats['total_access'] == 0
        assert stats['hit_rate'] == 0.0
        
        # 添加并访问条目
        cache.set_token_count("test1", "model", 10)
        cache.set_token_count("test2", "model", 20)
        
        cache.get_token_count("test1", "model")
        cache.get_token_count("test1", "model")
        cache.get_token_count("test2", "model")
        
        stats = cache.get_stats()
        assert stats['size'] == 2
        assert stats['total_access'] == 3
        assert stats['hit_rate'] > 0
    
    def test_token_cache_thread_safety(self):
        """测试Token缓存线程安全"""
        cache = TokenCache(max_size=1000)
        results = []
        
        def worker(thread_id):
            """工作线程函数"""
            for i in range(100):
                text = f"thread_{thread_id}_text_{i}"
                cache.set_token_count(text, "model", i)
                count = cache.get_token_count(text, "model")
                results.append(count == i)
        
        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有操作都成功
        assert all(results)
        assert len(cache._cache) == 500  # 5个线程 * 100个条目


class TestResponseCache:
    """测试响应缓存"""
    
    def test_response_cache_creation(self):
        """测试响应缓存创建"""
        cache = ResponseCache(max_size=500, default_ttl=600)
        
        assert cache.max_size == 500
        assert cache.default_ttl == 600
        assert len(cache._cache) == 0
    
    def test_generate_key_consistency(self):
        """测试请求键生成一致性"""
        cache = ResponseCache()
        
        request1 = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "trace_id": "123",  # 应被排除
            "timestamp": "2024-01-01",  # 应被排除
            "user_id": "user1"  # 应被排除
        }
        
        request2 = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.7,
            "trace_id": "456",  # 不同但应被排除
            "timestamp": "2024-01-02",  # 不同但应被排除
            "user_id": "user2"  # 不同但应被排除
        }
        
        request3 = {
            "temperature": 0.7,
            "model": "gpt-3.5-turbo",  # 顺序不同
            "messages": [{"role": "user", "content": "hello"}]
        }
        
        key1 = cache._generate_key(request1)
        key2 = cache._generate_key(request2)
        key3 = cache._generate_key(request3)
        
        # 排除无关字段后应该生成相同键
        assert key1 == key2 == key3
    
    def test_response_operations(self):
        """测试响应缓存操作"""
        cache = ResponseCache()
        
        request_data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "hello"}]
        }
        
        response_data = {
            "choices": [{"message": {"content": "Hello! How can I help you?"}}],
            "usage": {"total_tokens": 20}
        }
        
        # 测试缓存未命中
        cached_response = cache.get_response(request_data)
        assert cached_response is None
        
        # 设置缓存
        cache.set_response(request_data, response_data)
        
        # 测试缓存命中
        cached_response = cache.get_response(request_data)
        assert cached_response == response_data
    
    def test_response_cache_ttl(self):
        """测试响应缓存TTL"""
        cache = ResponseCache(default_ttl=1)
        
        request_data = {"model": "test"}
        response_data = {"result": "test"}
        
        # 设置短TTL缓存
        cache.set_response(request_data, response_data, ttl=1)
        
        # 立即获取应该成功
        cached = cache.get_response(request_data)
        assert cached == response_data
        
        # 等待过期
        time.sleep(1.1)
        
        # 过期后应该返回None
        cached = cache.get_response(request_data)
        assert cached is None
    
    def test_response_cache_lru_eviction(self):
        """测试响应缓存LRU淘汰"""
        cache = ResponseCache(max_size=2)
        
        request1 = {"model": "test1"}
        request2 = {"model": "test2"}
        request3 = {"model": "test3"}
        
        response1 = {"result": "1"}
        response2 = {"result": "2"}
        response3 = {"result": "3"}
        
        # 添加两个条目
        cache.set_response(request1, response1)
        cache.set_response(request2, response2)
        
        # 访问第一个条目
        cache.get_response(request1)
        
        # 添加第三个条目，应该淘汰request2
        cache.set_response(request3, response3)
        
        assert cache.get_response(request1) == response1
        assert cache.get_response(request2) is None
        assert cache.get_response(request3) == response3
    
    def test_response_cache_thread_safety(self):
        """测试响应缓存线程安全"""
        cache = ResponseCache(max_size=1000)
        results = []
        
        def worker(thread_id):
            """工作线程函数"""
            for i in range(50):
                request = {"model": f"thread_{thread_id}", "index": i}
                response = {"result": f"response_{thread_id}_{i}"}
                
                cache.set_response(request, response)
                cached = cache.get_response(request)
                results.append(cached == response)
        
        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有操作都成功
        assert all(results)
        assert len(cache._cache) == 250  # 5个线程 * 50个条目


class TestCacheManager:
    """测试缓存管理器"""
    
    def test_cache_manager_creation(self):
        """测试缓存管理器创建"""
        # 默认配置
        manager = CacheManager()
        assert manager.token_cache.max_size == 10000
        assert manager.token_cache.default_ttl == 3600
        assert manager.response_cache.max_size == 1000
        assert manager.response_cache.default_ttl == 300
        
        # 自定义配置
        token_config = {"max_size": 5000, "default_ttl": 1800}
        response_config = {"max_size": 500, "default_ttl": 150}
        
        manager = CacheManager(token_config, response_config)
        assert manager.token_cache.max_size == 5000
        assert manager.token_cache.default_ttl == 1800
        assert manager.response_cache.max_size == 500
        assert manager.response_cache.default_ttl == 150
    
    @pytest.mark.asyncio
    async def test_cleanup_task_lifecycle(self):
        """测试清理任务生命周期"""
        manager = CacheManager()
        
        # 初始状态
        assert manager._cleanup_task is None
        
        # 启动清理任务
        await manager.start_cleanup_task()
        assert manager._cleanup_task is not None
        assert not manager._cleanup_task.done()
        
        # 重复启动应该无效
        task_before = manager._cleanup_task
        await manager.start_cleanup_task()
        assert manager._cleanup_task is task_before
        
        # 停止清理任务
        await manager.stop_cleanup_task()
        assert manager._cleanup_task is None
    
    @pytest.mark.asyncio
    async def test_periodic_cleanup(self):
        """测试定期清理功能"""
        manager = CacheManager()
        manager._cleanup_interval = 0.1  # 设置短间隔用于测试
        
        # 添加一些过期条目
        manager.token_cache.set_token_count("test1", "model", 10, ttl=1)
        manager.response_cache.set_response({"test": "1"}, {"result": "1"}, ttl=1)
        
        # 等待过期
        time.sleep(1.1)
        
        # 启动清理任务
        await manager.start_cleanup_task()
        
        # 等待清理执行
        await asyncio.sleep(0.2)
        
        # 验证过期条目被清理
        assert len(manager.token_cache._cache) == 0
        assert len(manager.response_cache._cache) == 0
        
        await manager.stop_cleanup_task()
    
    @pytest.mark.asyncio
    async def test_cleanup_task_error_handling(self):
        """测试清理任务错误处理"""
        manager = CacheManager()
        manager._cleanup_interval = 0.1
        
        # Mock清理方法抛出异常
        with patch.object(manager.token_cache, 'clear_expired', side_effect=Exception("Test error")):
            await manager.start_cleanup_task()
            
            # 等待一段时间，确保任务继续运行
            await asyncio.sleep(0.3)
            
            # 任务应该仍在运行
            assert not manager._cleanup_task.done()
            
            await manager.stop_cleanup_task()
    
    def test_comprehensive_stats(self):
        """测试综合统计信息"""
        manager = CacheManager()
        
        # 添加一些数据
        manager.token_cache.set_token_count("test", "model", 10)
        manager.response_cache.set_response({"test": "data"}, {"result": "data"})
        
        stats = manager.get_comprehensive_stats()
        
        assert 'token_cache' in stats
        assert 'response_cache' in stats
        assert 'cleanup_task_running' in stats
        
        assert stats['token_cache']['size'] == 1
        assert stats['response_cache']['size'] == 1
        assert stats['cleanup_task_running'] is False
    
    def test_clear_all_caches(self):
        """测试清空所有缓存"""
        manager = CacheManager()
        
        # 添加数据
        manager.token_cache.set_token_count("test1", "model", 10)
        manager.token_cache.set_token_count("test2", "model", 20)
        manager.response_cache.set_response({"test": "1"}, {"result": "1"})
        manager.response_cache.set_response({"test": "2"}, {"result": "2"})
        
        # 验证数据存在
        assert len(manager.token_cache._cache) == 2
        assert len(manager.response_cache._cache) == 2
        
        # 清空所有缓存
        manager.clear_all_caches()
        
        # 验证缓存已清空
        assert len(manager.token_cache._cache) == 0
        assert len(manager.response_cache._cache) == 0


class TestGlobalCacheManager:
    """测试全局缓存管理器"""
    
    def setup_method(self):
        """每个测试前重置全局状态"""
        global _cache_manager
        _cache_manager = None
    
    def test_get_cache_manager_singleton(self):
        """测试全局缓存管理器单例"""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, CacheManager)
    
    @pytest.mark.asyncio
    async def test_global_cache_lifecycle(self):
        """测试全局缓存生命周期"""
        # 启动全局缓存管理器
        await start_cache_manager()
        
        manager = get_cache_manager()
        assert manager._cleanup_task is not None
        
        # 停止全局缓存管理器
        await stop_cache_manager()
        
        global _cache_manager
        assert _cache_manager is None


class TestConcurrentAccess:
    """测试并发访问场景"""
    
    def test_concurrent_token_cache_access(self):
        """测试Token缓存并发访问"""
        cache = TokenCache(max_size=1000)
        results = []
        errors = []
        
        def worker(worker_id):
            """工作线程"""
            try:
                for i in range(100):
                    text = f"worker_{worker_id}_text_{i}"
                    model = f"model_{i % 3}"  # 使用3个不同模型
                    
                    # 设置缓存
                    cache.set_token_count(text, model, i)
                    
                    # 获取缓存
                    count = cache.get_token_count(text, model)
                    results.append(count == i)
                    
                    # 随机清理过期条目
                    if i % 20 == 0:
                        cache.clear_expired()
                        
            except Exception as e:
                errors.append(e)
        
        # 启动多个工作线程
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            
            # 等待所有任务完成
            for future in as_completed(futures):
                future.result()
        
        # 验证结果
        assert len(errors) == 0, f"发生错误: {errors}"
        assert all(results), "部分操作失败"
    
    def test_concurrent_response_cache_access(self):
        """测试响应缓存并发访问"""
        cache = ResponseCache(max_size=1000)
        results = []
        errors = []
        
        def worker(worker_id):
            """工作线程"""
            try:
                for i in range(50):
                    request = {
                        "model": f"model_{i % 3}",
                        "worker": worker_id,
                        "index": i
                    }
                    response = {
                        "result": f"response_{worker_id}_{i}",
                        "tokens": i
                    }
                    
                    # 设置缓存
                    cache.set_response(request, response)
                    
                    # 获取缓存
                    cached = cache.get_response(request)
                    results.append(cached == response)
                    
                    # 随机清理过期条目
                    if i % 15 == 0:
                        cache.clear_expired()
                        
            except Exception as e:
                errors.append(e)
        
        # 启动多个工作线程
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(8)]
            
            # 等待所有任务完成
            for future in as_completed(futures):
                future.result()
        
        # 验证结果
        assert len(errors) == 0, f"发生错误: {errors}"
        assert all(results), "部分操作失败"
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_manager_operations(self):
        """测试缓存管理器并发操作"""
        manager = CacheManager()
        results = []
        errors = []
        
        async def async_worker(worker_id):
            """异步工作函数"""
            try:
                for i in range(30):
                    # Token缓存操作
                    text = f"async_worker_{worker_id}_text_{i}"
                    manager.token_cache.set_token_count(text, "model", i)
                    
                    # 响应缓存操作
                    request = {"worker": worker_id, "index": i}
                    response = {"result": f"async_result_{worker_id}_{i}"}
                    manager.response_cache.set_response(request, response)
                    
                    # 验证数据
                    token_count = manager.token_cache.get_token_count(text, "model")
                    cached_response = manager.response_cache.get_response(request)
                    
                    results.append(token_count == i)
                    results.append(cached_response == response)
                    
                    # 获取统计信息
                    if i % 10 == 0:
                        stats = manager.get_comprehensive_stats()
                        results.append(isinstance(stats, dict))
                        
                    await asyncio.sleep(0.001)  # 让出控制权
                    
            except Exception as e:
                errors.append(e)
        
        # 启动多个异步任务
        tasks = [async_worker(i) for i in range(5)]
        await asyncio.gather(*tasks)
        
        # 验证结果
        assert len(errors) == 0, f"发生错误: {errors}"
        assert all(results), "部分操作失败"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])