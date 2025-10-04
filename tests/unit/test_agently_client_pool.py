#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agently客户端池管理器单元测试
验证客户端复用机制的正确性和性能
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from harborai.core.agently_client_pool import (
    AgentlyClientPool,
    AgentlyClientConfig,
    CachedAgentlyClient,
    get_agently_client_pool,
    create_agently_client_config
)


class TestAgentlyClientConfig:
    """测试Agently客户端配置"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = AgentlyClientConfig(
            provider="OpenAICompatible",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key="sk-test123456789",
            temperature=0.1,
            max_tokens=1000
        )
        
        assert config.provider == "OpenAICompatible"
        assert config.base_url == "https://api.deepseek.com"
        assert config.model == "deepseek-chat"
        assert config.api_key == "sk-test123456789"
        assert config.temperature == 0.1
        assert config.max_tokens == 1000
        assert config.model_type == "chat"  # 默认值
    
    def test_cache_key_generation(self):
        """测试缓存键生成"""
        config1 = AgentlyClientConfig(
            provider="OpenAICompatible",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key="sk-test123456789"
        )
        
        config2 = AgentlyClientConfig(
            provider="OpenAICompatible",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key="sk-test123456789"
        )
        
        # 相同配置应该生成相同的缓存键
        assert config1.to_cache_key() == config2.to_cache_key()
        
        # 不同配置应该生成不同的缓存键
        config3 = AgentlyClientConfig(
            provider="OpenAICompatible",
            base_url="https://api.deepseek.com",
            model="deepseek-chat-different",
            api_key="sk-test123456789"
        )
        
        assert config1.to_cache_key() != config3.to_cache_key()
    
    def test_cache_key_api_key_privacy(self):
        """测试缓存键中API密钥的隐私保护"""
        config = AgentlyClientConfig(
            provider="OpenAICompatible",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key="sk-very-secret-key-12345"
        )
        
        cache_key = config.to_cache_key()
        # 缓存键不应包含完整的API密钥
        assert "sk-very-secret-key-12345" not in cache_key


class TestCachedAgentlyClient:
    """测试缓存的Agently客户端"""
    
    def test_cached_client_creation(self):
        """测试缓存客户端创建"""
        mock_client = Mock()
        config = AgentlyClientConfig(
            provider="OpenAICompatible",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key="sk-test123456789"
        )
        
        cached_client = CachedAgentlyClient(
            client=mock_client,
            config=config,
            created_at=time.time(),
            last_used=time.time()
        )
        
        assert cached_client.client == mock_client
        assert cached_client.config == config
        assert cached_client.use_count == 0
    
    def test_mark_used(self):
        """测试标记使用"""
        mock_client = Mock()
        config = AgentlyClientConfig(
            provider="OpenAICompatible",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key="sk-test123456789"
        )
        
        cached_client = CachedAgentlyClient(
            client=mock_client,
            config=config,
            created_at=time.time(),
            last_used=time.time()
        )
        
        initial_use_count = cached_client.use_count
        initial_last_used = cached_client.last_used
        
        time.sleep(0.01)  # 确保时间差异
        cached_client.mark_used()
        
        assert cached_client.use_count == initial_use_count + 1
        assert cached_client.last_used > initial_last_used


class TestAgentlyClientPool:
    """测试Agently客户端池"""
    
    def setup_method(self):
        """测试前设置"""
        # 重置单例实例
        AgentlyClientPool._instance = None
        self.pool = AgentlyClientPool()
    
    def teardown_method(self):
        """测试后清理"""
        if hasattr(self, 'pool'):
            self.pool.clear_pool()
        AgentlyClientPool._instance = None
    
    def test_singleton_pattern(self):
        """测试单例模式"""
        pool1 = AgentlyClientPool()
        pool2 = AgentlyClientPool()
        
        assert pool1 is pool2
    
    def test_client_creation_and_caching(self):
        """测试客户端创建和缓存"""
        with patch('agently.Agently') as mock_agently_class:
            # 模拟Agently
            mock_agent = Mock()
            mock_agently_class.create_agent.return_value = mock_agent
            
            config = AgentlyClientConfig(
                provider="OpenAICompatible",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                api_key="sk-test123456789"
            )
            
            # 第一次获取客户端
            client1 = self.pool.get_client(config)
            assert client1 == mock_agent
            assert self.pool._stats["cache_misses"] == 1
            assert self.pool._stats["cache_hits"] == 0
            assert self.pool._stats["clients_created"] == 1
            
            # 第二次获取相同配置的客户端（应该命中缓存）
            client2 = self.pool.get_client(config)
            assert client2 == mock_agent
            assert client1 is client2
            assert self.pool._stats["cache_misses"] == 1
            assert self.pool._stats["cache_hits"] == 1
            assert self.pool._stats["clients_created"] == 1
    
    def test_different_configs_create_different_clients(self):
        """测试不同配置创建不同客户端"""
        with patch('agently.Agently') as mock_agently_class:
            mock_agent1 = Mock()
            mock_agent2 = Mock()
            mock_agently_class.create_agent.side_effect = [mock_agent1, mock_agent2]
            
            config1 = AgentlyClientConfig(
                provider="OpenAICompatible",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                api_key="sk-test123456789"
            )
            
            config2 = AgentlyClientConfig(
                provider="OpenAICompatible",
                base_url="https://api.deepseek.com",
                model="deepseek-chat-different",
                api_key="sk-test123456789"
            )
            
            client1 = self.pool.get_client(config1)
            client2 = self.pool.get_client(config2)
            
            assert client1 != client2
            assert self.pool._stats["clients_created"] == 2
            assert len(self.pool._clients) == 2
    
    def test_client_expiration(self):
        """测试客户端过期"""
        with patch('agently.Agently') as mock_agently_class:
            mock_agent = Mock()
            mock_agently_class.create_agent.return_value = mock_agent
            
            # 设置很短的TTL用于测试
            self.pool.set_pool_config(ttl=0.1)
            
            config = AgentlyClientConfig(
                provider="OpenAICompatible",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                api_key="sk-test123456789"
            )
            
            # 获取客户端
            client1 = self.pool.get_client(config)
            assert len(self.pool._clients) == 1
            
            # 等待过期
            time.sleep(0.2)
            
            # 再次获取应该创建新客户端
            client2 = self.pool.get_client(config)
            assert self.pool._stats["clients_created"] == 2
    
    def test_pool_size_limit(self):
        """测试池大小限制"""
        with patch('agently.Agently') as mock_agently_class:
            mock_agently_class.create_agent.return_value = Mock()
            
            # 设置小的池大小
            self.pool.set_pool_config(max_size=2)
            
            configs = []
            for i in range(3):
                config = AgentlyClientConfig(
                    provider="OpenAICompatible",
                    base_url="https://api.deepseek.com",
                    model=f"model-{i}",
                    api_key="sk-test123456789"
                )
                configs.append(config)
                self.pool.get_client(config)
            
            # 池大小应该被限制
            assert len(self.pool._clients) <= 2
            assert self.pool._stats["clients_cleaned"] > 0
    
    def test_thread_safety(self):
        """测试线程安全"""
        with patch('agently.Agently') as mock_agently_class:
            mock_agently_class.create_agent.return_value = Mock()
            
            config = AgentlyClientConfig(
                provider="OpenAICompatible",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                api_key="sk-test123456789"
            )
            
            results = []
            errors = []
            
            def worker():
                try:
                    client = self.pool.get_client(config)
                    results.append(client)
                except Exception as e:
                    errors.append(e)
            
            # 创建多个线程同时获取客户端
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=worker)
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 检查结果
            assert len(errors) == 0
            assert len(results) == 10
            # 所有结果应该是同一个客户端实例（缓存命中）
            assert all(client is results[0] for client in results)
    
    def test_get_stats(self):
        """测试获取统计信息"""
        stats = self.pool.get_stats()
        
        assert "cache_hits" in stats
        assert "cache_misses" in stats
        assert "clients_created" in stats
        assert "clients_cleaned" in stats
        assert "total_requests" in stats
        assert "pool_size" in stats
        assert "cache_hit_rate" in stats
        assert "clients_info" in stats
    
    def test_client_context_manager(self):
        """测试客户端上下文管理器"""
        with patch('agently.Agently') as mock_agently_class:
            mock_agent = Mock()
            mock_agently_class.create_agent.return_value = mock_agent
            
            config = AgentlyClientConfig(
                provider="OpenAICompatible",
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                api_key="sk-test123456789"
            )
            
            with self.pool.get_client_context(config) as client:
                assert client == mock_agent
    
    def test_clear_pool(self):
        """测试清空池"""
        # 先添加一些模拟客户端
        self.pool._clients["test1"] = Mock()
        self.pool._clients["test2"] = Mock()
        
        assert len(self.pool._clients) == 2
        
        self.pool.clear_pool()
        
        assert len(self.pool._clients) == 0
    
    def test_set_pool_config(self):
        """测试设置池配置"""
        self.pool.set_pool_config(max_size=20, ttl=7200, cleanup_interval=600)
        
        assert self.pool._max_pool_size == 20
        assert self.pool._client_ttl == 7200
        assert self.pool._cleanup_interval == 600


class TestGlobalFunctions:
    """测试全局函数"""
    
    def test_get_agently_client_pool(self):
        """测试获取全局客户端池"""
        pool1 = get_agently_client_pool()
        pool2 = get_agently_client_pool()
        
        assert pool1 is pool2
        assert isinstance(pool1, AgentlyClientPool)
    
    def test_create_agently_client_config(self):
        """测试创建客户端配置"""
        config = create_agently_client_config(
            provider="OpenAICompatible",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            api_key="sk-test123456789",
            temperature=0.1,
            max_tokens=1000
        )
        
        assert isinstance(config, AgentlyClientConfig)
        assert config.provider == "OpenAICompatible"
        assert config.base_url == "https://api.deepseek.com"
        assert config.model == "deepseek-chat"
        assert config.api_key == "sk-test123456789"
        assert config.temperature == 0.1
        assert config.max_tokens == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])