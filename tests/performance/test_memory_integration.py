"""
内存优化集成测试

测试内存优化组件与FastHarborAI客户端的集成
"""

import pytest
import time
import gc
import psutil
import os
from unittest.mock import Mock, patch

from harborai.api.fast_client import FastHarborAI
from harborai.core.optimizations.memory_manager import MemoryManager


class TestMemoryIntegration:
    """内存优化集成测试类"""
    
    def setup_method(self):
        """测试前设置"""
        gc.collect()
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def teardown_method(self):
        """测试后清理"""
        gc.collect()
    
    def test_fast_harborai_memory_optimization_enabled(self):
        """测试FastHarborAI启用内存优化"""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'enable_memory_optimization': True,
            'memory_optimization': {
                'cache_size': 100,
                'object_pool_size': 50,
                'enable_weak_references': True,
                'memory_threshold': 100.0,
                'cleanup_interval': 60
            }
        }
        
        client = FastHarborAI(config=config)
        
        # 验证内存管理器已初始化
        assert hasattr(client, '_memory_manager')
        assert client._memory_manager is not None
        assert isinstance(client._memory_manager, MemoryManager)
        
        # 验证内存管理器配置
        assert client._memory_manager.cache._max_size == 100
        assert client._memory_manager.object_pool._max_size == 50
        
        # 清理
        client.cleanup()
    
    def test_fast_harborai_memory_optimization_disabled(self):
        """测试FastHarborAI禁用内存优化"""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'enable_memory_optimization': False
        }
        
        client = FastHarborAI(config=config)
        
        # 验证内存管理器未初始化
        assert client._memory_manager is None
        
        # 清理
        client.cleanup()
    
    def test_memory_stats_integration(self):
        """测试内存统计集成"""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'enable_memory_optimization': True
        }
        
        client = FastHarborAI(config=config)
        
        # 获取内存统计
        stats = client.get_memory_stats()
        
        # 验证统计信息结构
        assert 'cache' in stats
        assert 'object_pools' in stats
        assert 'weak_references_count' in stats
        assert 'system_memory' in stats
        
        # 验证缓存统计
        cache_stats = stats['cache']
        assert cache_stats['max_size'] == 1000
        assert cache_stats['size'] >= 0
        
        # 验证对象池统计
        pool_stats = stats['object_pools']
        assert 'default' in pool_stats
        assert pool_stats['default']['max_size'] == 100
        
        # 清理
        client.cleanup()
    
    def test_memory_cleanup_integration(self):
        """测试内存清理集成"""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'enable_memory_optimization': True
        }
        
        client = FastHarborAI(config=config)
        
        # 添加一些缓存数据
        cache = client._memory_manager.cache
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")
        
        # 验证缓存有数据
        assert cache.size() > 0
        
        # 执行内存清理
        client.cleanup_memory(force_clear=True)
        
        # 验证缓存已清空
        assert cache.size() == 0
        
        # 清理
        client.cleanup()
    
    def test_chat_completions_cache_integration(self):
        """测试聊天完成缓存集成"""
        config = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'enable_memory_optimization': True,
            'enable_caching': True
        }
        
        client = FastHarborAI(config=config)
        
        # 模拟插件管理器
        with patch.object(client.chat.completions, '_lazy_manager') as mock_manager:
            mock_plugin = Mock()
            mock_plugin.chat_completion.return_value = {'id': 'test', 'choices': []}
            mock_manager.get_plugin_for_model.return_value = mock_plugin
            
            # 验证缓存管理器使用内存管理器的缓存
            client.chat.completions._ensure_initialized()
            
            # 验证缓存管理器是内存管理器的缓存
            assert client.chat.completions._cache_manager is client._memory_manager.cache
        
        # 清理
        client.cleanup()
    
    def test_memory_usage_optimization(self):
        """测试内存使用优化效果"""
        # 测试不启用内存优化的情况
        config_no_opt = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'enable_memory_optimization': False
        }
        
        # 记录初始内存
        gc.collect()
        initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # 创建多个客户端实例（模拟内存压力）
        clients_no_opt = []
        for i in range(5):
            client = FastHarborAI(config=config_no_opt)
            clients_no_opt.append(client)
        
        # 记录无优化时的内存
        gc.collect()
        memory_no_opt = self.process.memory_info().rss / 1024 / 1024
        
        # 清理无优化的客户端
        for client in clients_no_opt:
            client.cleanup()
        del clients_no_opt
        gc.collect()
        
        # 测试启用内存优化的情况
        config_with_opt = {
            'api_key': 'test_key',
            'base_url': 'https://api.test.com',
            'enable_memory_optimization': True,
            'memory_optimization': {
                'cache_size': 50,
                'object_pool_size': 25
            }
        }
        
        # 创建相同数量的客户端实例
        clients_with_opt = []
        for i in range(5):
            client = FastHarborAI(config=config_with_opt)
            clients_with_opt.append(client)
        
        # 记录有优化时的内存
        gc.collect()
        memory_with_opt = self.process.memory_info().rss / 1024 / 1024
        
        # 验证内存优化效果（允许一定的测量误差）
        memory_increase_no_opt = memory_no_opt - initial_memory
        memory_increase_with_opt = memory_with_opt - initial_memory
        
        # 内存优化应该减少内存使用（至少不显著增加）
        assert memory_increase_with_opt <= memory_increase_no_opt + 2.0  # 允许2MB误差
        
        # 清理有优化的客户端
        for client in clients_with_opt:
            client.cleanup()
        del clients_with_opt
        gc.collect()
    
    def test_error_handling_memory_optimization(self):
        """测试内存优化错误处理"""
        # 使用无效配置
        invalid_config = {
            "memory_optimization": {
                "cache_size": -1,
                "object_pool_size": "invalid"
            }
        }
        
        # 应该能够处理无效配置并使用默认值
        client = FastHarborAI(config=invalid_config, enable_memory_optimization=True)
        
        # 验证内存管理器已初始化（使用默认值）
        assert client._memory_manager is not None
        
        # 验证配置被修正为有效值
        stats = client.get_memory_stats()
        assert stats is not None
        assert stats['cache']['max_size'] > 0  # 应该使用默认值而不是-1
        assert stats['cache']['max_size'] == 1000  # 默认值


if __name__ == '__main__':
    pytest.main([__file__, '-v'])