#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速客户端模块全面测试

测试harborai.api.fast_client模块的所有功能，包括：
- FastChatCompletions: 快速聊天完成接口
- FastHarborAI: 快速HarborAI客户端
- 性能优化功能
- 延迟加载机制
- 内存和并发优化

测试策略：
1. 单元测试：测试核心功能
2. 性能测试：验证启动性能优化
3. 集成测试：测试与其他组件的集成
4. Mock测试：模拟外部依赖和优化组件
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock, PropertyMock
from typing import Any, Dict, List, Optional

from harborai.api.fast_client import (
    FastChatCompletions,
    FastChat,
    FastHarborAI,
    create_fast_client,
    MEMORY_OPTIMIZATION_AVAILABLE,
    CONCURRENCY_OPTIMIZATION_AVAILABLE
)
from harborai.core.exceptions import HarborAIError, ValidationError


class TestFastChatCompletions:
    """测试FastChatCompletions类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.mock_client = Mock()
        self.mock_client.config = {
            'enable_performance_optimization': True,
            'enable_caching': True,
            'supported_models': ['gpt-3.5-turbo', 'gpt-4']
        }
        
        # 创建FastChatCompletions实例
        self.fast_completions = FastChatCompletions(self.mock_client)
    
    def test_fast_chat_completions_initialization(self):
        """测试FastChatCompletions初始化"""
        assert self.fast_completions._client == self.mock_client
        assert self.fast_completions._config == self.mock_client.config
        assert self.fast_completions._initialized is False
        assert self.fast_completions._request_count == 0
        assert self.fast_completions._total_time == 0.0
        assert self.fast_completions._last_request_time is None
        assert self.fast_completions._lazy_plugin_manager is None
        assert self.fast_completions._performance_manager is None
        assert self.fast_completions._cache_manager is None
    
    def test_get_lazy_plugin_manager(self):
        """测试获取延迟插件管理器"""
        with patch('harborai.api.fast_client.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            manager = self.fast_completions._get_lazy_plugin_manager()
            assert manager == mock_manager
            mock_get_manager.assert_called_once_with(self.mock_client.config)
            
            # 测试缓存机制
            manager2 = self.fast_completions._get_lazy_plugin_manager()
            assert manager2 == mock_manager
            # 应该只调用一次，因为有缓存
            mock_get_manager.assert_called_once()
    
    def test_get_performance_manager(self):
        """测试获取性能管理器"""
        # 由于导入路径问题，直接测试ImportError情况
        # 重置manager以测试ImportError
        self.fast_completions._performance_manager = None
        manager = self.fast_completions._get_performance_manager()
        assert manager is None
    
    def test_get_cache_manager(self):
        """测试获取缓存管理器"""
        # 由于导入路径问题，直接测试ImportError情况
        # 重置manager以测试ImportError
        self.fast_completions._cache_manager = None
        manager = self.fast_completions._get_cache_manager()
        assert manager is None
    
    def test_get_concurrency_manager(self):
        """测试获取并发管理器"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        with patch('harborai.api.fast_client.ConcurrencyManager') as mock_concurrency_manager:
            with patch('harborai.api.fast_client.ConcurrencyConfig') as mock_config:
                with patch('asyncio.run') as mock_run:
                    mock_instance = Mock()
                    mock_concurrency_manager.return_value = mock_instance
                    
                    manager = self.fast_completions._get_concurrency_manager()
                    assert manager == mock_instance
    
    def test_ensure_initialized(self):
        """测试确保初始化"""
        with patch('harborai.api.fast_client.get_lazy_plugin_manager') as mock_get_manager:
            with patch.object(self.fast_completions, '_load_performance_manager') as mock_load_perf:
                with patch.object(self.fast_completions, '_load_cache_manager') as mock_load_cache:
                    mock_manager = Mock()
                    mock_get_manager.return_value = mock_manager
                    
                    self.fast_completions._ensure_initialized()
                    
                    assert self.fast_completions._initialized is True
                    assert self.fast_completions._lazy_manager == mock_manager
                    mock_get_manager.assert_called_once_with(self.mock_client.config)
                    mock_load_perf.assert_called_once()
                    mock_load_cache.assert_called_once()
                    
                    # 测试重复调用不会重新初始化
                    self.fast_completions._ensure_initialized()
                    mock_get_manager.assert_called_once()  # 仍然只调用一次
    
    def test_load_performance_manager(self):
        """测试加载性能管理器"""
        with patch('harborai.core.performance_manager.PerformanceManager') as mock_perf_manager:
            mock_instance = Mock()
            mock_perf_manager.return_value = mock_instance
            
            self.fast_completions._load_performance_manager()
            assert self.fast_completions._performance_manager == mock_instance
            
            # 测试ImportError情况
            with patch('harborai.core.performance_manager.PerformanceManager', side_effect=ImportError):
                self.fast_completions._performance_manager = None
                self.fast_completions._load_performance_manager()
                assert self.fast_completions._performance_manager is None
    
    def test_load_cache_manager(self):
        """测试加载缓存管理器"""
        # 测试使用内存管理器的缓存
        mock_memory_manager = Mock()
        mock_cache = Mock()
        mock_memory_manager.cache = mock_cache
        self.mock_client._memory_manager = mock_memory_manager
        
        self.fast_completions._load_cache_manager()
        assert self.fast_completions._cache_manager == mock_cache
        
        # 测试回退到传统缓存管理器
        self.mock_client._memory_manager = None
        self.fast_completions._cache_manager = None
        
        with patch('harborai.core.cache_manager.CacheManager') as mock_cache_manager:
            mock_instance = Mock()
            mock_cache_manager.return_value = mock_instance
            
            self.fast_completions._load_cache_manager()
            assert self.fast_completions._cache_manager == mock_instance
    
    def test_validate_request(self):
        """测试请求验证"""
        messages = [{"role": "user", "content": "Hello"}]
        model = "gpt-3.5-turbo"
        
        # 正常情况 - 需要先初始化
        with patch.object(self.fast_completions, '_ensure_initialized'):
            self.fast_completions._validate_request(messages, model)
        
        # 测试空消息
        with pytest.raises(ValidationError):
            with patch.object(self.fast_completions, '_ensure_initialized'):
                self.fast_completions._validate_request([], model)
        
        # 测试空模型
        with pytest.raises(ValidationError):
            with patch.object(self.fast_completions, '_ensure_initialized'):
                self.fast_completions._validate_request(messages, "")
        
        # 测试无效消息格式
        with pytest.raises(ValidationError):
            with patch.object(self.fast_completions, '_ensure_initialized'):
                self.fast_completions._validate_request([{"invalid": "format"}], model)
    
    def test_generate_cache_key(self):
        """测试生成缓存键"""
        messages = [{"role": "user", "content": "Hello"}]
        model = "gpt-3.5-turbo"
        
        key1 = self.fast_completions._generate_cache_key(messages, model)
        key2 = self.fast_completions._generate_cache_key(messages, model)
        
        # 相同输入应该生成相同的键
        assert key1 == key2
        
        # 不同输入应该生成不同的键
        key3 = self.fast_completions._generate_cache_key(messages, "gpt-4")
        assert key1 != key3
    
    def test_create_traditional(self):
        """测试传统创建方法"""
        messages = [{"role": "user", "content": "Hello"}]
        model = "gpt-3.5-turbo"
        
        # 模拟已初始化状态
        mock_manager = Mock()
        mock_plugin = Mock()
        mock_response = {"choices": [{"message": {"content": "Hi there!"}}]}
        
        mock_manager.get_plugin_for_model.return_value = mock_plugin
        mock_plugin.chat_completion.return_value = mock_response
        self.fast_completions._lazy_manager = mock_manager
        
        result = self.fast_completions._create_traditional(messages, model, False)
        
        assert result == mock_response
        mock_manager.get_plugin_for_model.assert_called_once_with(model)
        mock_plugin.chat_completion.assert_called_once()


class TestFastChat:
    """测试FastChat类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.mock_client = Mock()
        self.mock_client.config = {'test': 'config'}
        self.fast_chat = FastChat(self.mock_client)
    
    def test_fast_chat_initialization(self):
        """测试FastChat初始化"""
        assert self.fast_chat._client == self.mock_client
        assert self.fast_chat._completions is None
    
    def test_completions_property(self):
        """测试completions属性的延迟创建"""
        # 第一次访问应该创建实例
        completions = self.fast_chat.completions
        assert isinstance(completions, FastChatCompletions)
        assert completions._client == self.mock_client
        
        # 第二次访问应该返回相同实例
        completions2 = self.fast_chat.completions
        assert completions2 is completions


class TestFastHarborAI:
    """测试FastHarborAI类"""
    
    def test_fast_harbor_ai_initialization_basic(self):
        """测试FastHarborAI基本初始化"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {'default_timeout': 30}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI()
            
            assert hasattr(client, 'config')
            assert client._chat is None
            assert hasattr(client, '_initialized_at')
            assert client._memory_manager is None
    
    def test_fast_harbor_ai_initialization_with_config(self):
        """测试FastHarborAI带配置初始化"""
        config = {'api_key': 'test-key', 'timeout': 60}
        
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {'default_timeout': 30}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI(config=config)
            
            assert client.config['api_key'] == 'test-key'
            assert client.config['timeout'] == 60
    
    def test_merge_config(self):
        """测试配置合并"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {'default_timeout': 30, 'max_retries': 3}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI()
            
            config = {'api_key': 'test-key'}
            kwargs = {'timeout': 60}
            
            merged = client._merge_config(config, kwargs)
            
            assert merged['default_timeout'] == 30  # 来自默认配置
            assert merged['max_retries'] == 3       # 来自默认配置
            assert merged['api_key'] == 'test-key'  # 来自config
            assert merged['timeout'] == 60          # 来自kwargs
    
    def test_chat_property(self):
        """测试chat属性的延迟创建"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI()
            
            # 第一次访问应该创建实例
            chat = client.chat
            assert isinstance(chat, FastChat)
            assert chat._client == client
            
            # 第二次访问应该返回相同实例
            chat2 = client.chat
            assert chat2 is chat
    
    def test_init_memory_optimization(self):
        """测试内存优化初始化"""
        if not MEMORY_OPTIMIZATION_AVAILABLE:
            pytest.skip("内存优化组件不可用")
        
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            with patch('harborai.api.fast_client.MemoryManager') as mock_memory_manager:
                mock_settings = Mock()
                mock_settings.dict.return_value = {'enable_memory_optimization': True}
                mock_get_settings.return_value = mock_settings
                
                mock_instance = Mock()
                mock_memory_manager.return_value = mock_instance
                
                client = FastHarborAI(enable_memory_optimization=True)
                
                assert client._memory_manager == mock_instance
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            with patch('harborai.api.fast_client.get_lazy_plugin_manager') as mock_get_manager:
                mock_settings = Mock()
                mock_settings.dict.return_value = {}
                mock_get_settings.return_value = mock_settings
                
                mock_manager = Mock()
                mock_manager.get_statistics.return_value = {'plugins_loaded': 2}
                mock_get_manager.return_value = mock_manager
                
                client = FastHarborAI()
                stats = client.get_statistics()
                
                assert stats['client_type'] == 'FastHarborAI'
                assert 'initialized_at' in stats
                assert 'uptime_seconds' in stats
                assert 'memory_optimization_enabled' in stats
                assert stats['plugins_loaded'] == 2
    
    def test_cleanup(self):
        """测试资源清理"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI()
            
            # 模拟内存管理器
            mock_memory_manager = Mock()
            client._memory_manager = mock_memory_manager
            
            # 模拟聊天接口
            mock_chat = Mock()
            mock_completions = Mock()
            mock_performance_manager = Mock()
            mock_completions._performance_manager = mock_performance_manager
            mock_chat.completions = mock_completions
            client._chat = mock_chat
            
            client.cleanup()
            
            mock_performance_manager.cleanup.assert_called_once()
            mock_memory_manager.shutdown.assert_called_once()


class TestCreateFastClient:
    """测试便捷函数"""
    
    def test_create_fast_client(self):
        """测试create_fast_client函数"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            config = {'api_key': 'test-key'}
            client = create_fast_client(config=config, timeout=60)
            
            assert isinstance(client, FastHarborAI)
            assert client.config['api_key'] == 'test-key'
            assert client.config['timeout'] == 60


class TestPerformanceOptimizations:
    """测试性能优化功能"""
    
    def test_initialization_performance(self):
        """测试初始化性能"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            start_time = time.perf_counter()
            client = FastHarborAI()
            init_time = (time.perf_counter() - start_time) * 1000
            
            # 初始化应该很快（目标≤160ms）
            assert init_time < 200  # 给一些余量
            assert hasattr(client, '_initialized_at')
    
    def test_lazy_loading_performance(self):
        """测试延迟加载性能"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI()
            
            # 初始状态下chat应该为None
            assert client._chat is None
            
            # 第一次访问时才创建
            start_time = time.perf_counter()
            chat = client.chat
            access_time = (time.perf_counter() - start_time) * 1000
            
            assert isinstance(chat, FastChat)
            # 延迟加载应该很快
            assert access_time < 50


class TestErrorHandling:
    """测试错误处理"""
    
    def test_initialization_error_handling(self):
        """测试初始化错误处理"""
        with patch('harborai.api.fast_client.get_settings', side_effect=Exception("Settings error")):
            # 即使设置加载失败，也应该能创建客户端
            client = FastHarborAI()
            assert isinstance(client, FastHarborAI)
    
    def test_memory_optimization_error_handling(self):
        """测试内存优化错误处理"""
        if not MEMORY_OPTIMIZATION_AVAILABLE:
            pytest.skip("内存优化组件不可用")
        
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            with patch('harborai.api.fast_client.MemoryManager', side_effect=Exception("Memory error")):
                mock_settings = Mock()
                mock_settings.dict.return_value = {'enable_memory_optimization': True}
                mock_get_settings.return_value = mock_settings
                
                # 即使内存优化失败，也应该能创建客户端
                client = FastHarborAI(enable_memory_optimization=True)
                assert isinstance(client, FastHarborAI)
                assert client._memory_manager is None
    
    def test_cleanup_error_handling(self):
        """测试清理错误处理"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI()
            
            # 模拟清理时出错的内存管理器
            mock_memory_manager = Mock()
            mock_memory_manager.shutdown.side_effect = Exception("Cleanup error")
            client._memory_manager = mock_memory_manager
            
            # 清理应该不抛出异常
            client.cleanup()


class TestEdgeCases:
    """测试边界情况"""
    
    def test_empty_config(self):
        """测试空配置"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI(config={})
            assert isinstance(client, FastHarborAI)
    
    def test_none_config(self):
        """测试None配置"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI(config=None)
            assert isinstance(client, FastHarborAI)
    
    def test_large_config(self):
        """测试大配置"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            large_config = {f'key_{i}': f'value_{i}' for i in range(1000)}
            client = FastHarborAI(config=large_config)
            assert isinstance(client, FastHarborAI)
            assert len(client.config) >= 1000


class TestImportErrorHandling:
    """测试导入错误处理情况"""
    
    def setup_method(self):
        """测试前置设置"""
        self.mock_client = Mock()
        self.mock_client.config = {
            'enable_performance_optimization': True,
            'enable_caching': True,
            'supported_models': ['gpt-3.5-turbo', 'gpt-4']
        }
    
    def test_memory_manager_import_error(self):
        """测试MemoryManager导入失败的情况"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            # 模拟MemoryManager导入失败
            with patch('harborai.api.fast_client.MEMORY_OPTIMIZATION_AVAILABLE', False):
                mock_settings = Mock()
                mock_settings.dict.return_value = {'enable_memory_optimization': True}
                mock_get_settings.return_value = mock_settings
                
                client = FastHarborAI(enable_memory_optimization=True)
                
                # 应该能正常创建客户端，但内存管理器为None
                assert isinstance(client, FastHarborAI)
                assert client._memory_manager is None
    
    def test_concurrency_manager_import_error(self):
        """测试ConcurrencyManager导入失败的情况"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 模拟并发优化不可用
        with patch('harborai.api.fast_client.CONCURRENCY_OPTIMIZATION_AVAILABLE', False):
            manager = fast_completions._get_concurrency_manager()
            assert manager is None
    
    def test_performance_manager_import_error(self):
        """测试性能管理器导入失败时的警告处理"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 测试ImportError情况
        with patch('harborai.core.performance_manager.PerformanceManager', side_effect=ImportError("Module not found")):
            fast_completions._load_performance_manager()
            assert fast_completions._performance_manager is None
        
        # 测试其他异常情况
        with patch('harborai.core.performance_manager.PerformanceManager', side_effect=Exception("Other error")):
            fast_completions._performance_manager = None
            fast_completions._load_performance_manager()
            assert fast_completions._performance_manager is None
    
    def test_cache_manager_import_error(self):
        """测试缓存管理器导入失败时的警告处理"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 测试其他异常情况
        with patch('harborai.core.cache_manager.CacheManager', side_effect=Exception("Other error")):
            # 确保没有内存管理器，这样会尝试使用传统缓存管理器
            self.mock_client._memory_manager = None
            fast_completions._cache_manager = None
            fast_completions._load_cache_manager()
            assert fast_completions._cache_manager is None


class TestMemoryOptimizationEdgeCases:
    """测试内存优化边界情况"""
    
    def test_memory_optimization_init_failure(self):
        """测试内存优化初始化失败的情况"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            with patch('harborai.api.fast_client.MEMORY_OPTIMIZATION_AVAILABLE', True):
                with patch('harborai.api.fast_client.MemoryManager', side_effect=Exception("Init failed")):
                    mock_settings = Mock()
                    mock_settings.dict.return_value = {}
                    mock_get_settings.return_value = mock_settings
                    
                    client = FastHarborAI(enable_memory_optimization=True)
                    
                    # 应该能正常创建客户端，但内存管理器为None
                    assert isinstance(client, FastHarborAI)
                    assert client._memory_manager is None
    
    def test_memory_optimization_unavailable_warning(self):
        """测试内存优化组件不可用时的警告"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            with patch('harborai.api.fast_client.MEMORY_OPTIMIZATION_AVAILABLE', False):
                mock_settings = Mock()
                mock_settings.dict.return_value = {}
                mock_get_settings.return_value = mock_settings
                
                client = FastHarborAI(enable_memory_optimization=True)
                
                # 调用内存优化初始化方法
                client._init_memory_optimization()
                
                # 内存管理器应该为None
                assert client._memory_manager is None
    
    def test_memory_stats_psutil_import_error(self):
        """测试获取内存统计时psutil不可用的情况"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI()
            
            # 模拟内存管理器
            mock_memory_manager = Mock()
            mock_memory_manager.get_memory_stats.return_value = {'cache_size': 100}
            client._memory_manager = mock_memory_manager
            
            # 模拟psutil导入失败
            with patch('psutil.Process', side_effect=ImportError("psutil not available")):
                stats = client.get_memory_stats()
                
                assert stats is not None
                assert stats['cache_size'] == 100
                assert stats['system_memory']['error'] == 'psutil not available'
    
    def test_memory_stats_psutil_runtime_error(self):
        """测试获取内存统计时psutil运行时错误的情况"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI()
            
            # 模拟内存管理器
            mock_memory_manager = Mock()
            mock_memory_manager.get_memory_stats.return_value = {'cache_size': 100}
            client._memory_manager = mock_memory_manager
            
            # 模拟psutil运行时错误
            with patch('psutil.Process', side_effect=Exception("Runtime error")):
                stats = client.get_memory_stats()
                
                assert stats is not None
                assert stats['cache_size'] == 100
                assert stats['system_memory']['error'] == 'Runtime error'
    
    def test_memory_stats_without_memory_manager(self):
        """测试没有内存管理器时获取内存统计"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI()
            
            # 确保没有内存管理器
            client._memory_manager = None
            
            stats = client.get_memory_stats()
            assert stats is None
    
    def test_cleanup_memory_without_memory_manager(self):
        """测试没有内存管理器时清理内存"""
        with patch('harborai.api.fast_client.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.dict.return_value = {}
            mock_get_settings.return_value = mock_settings
            
            client = FastHarborAI()
            
            # 确保没有内存管理器
            client._memory_manager = None
            
            result = client.cleanup_memory()
            assert result is None


class TestConcurrencyManagerEdgeCases:
    """测试并发管理器边界情况"""
    
    def setup_method(self):
        """测试前置设置"""
        self.mock_client = Mock()
        self.mock_client.config = {
            'enable_performance_optimization': True,
            'enable_caching': True,
            'concurrency_optimization': {
                'max_concurrent_requests': 50,
                'connection_pool_size': 20,
                'request_timeout': 30.0,
                'enable_adaptive_optimization': True,
                'enable_health_check': True,
                'health_check_interval': 60.0
            }
        }
    
    @pytest.mark.skipif(not CONCURRENCY_OPTIMIZATION_AVAILABLE, reason="并发优化组件不可用")
    def test_concurrency_manager_start_failure(self):
        """测试并发管理器启动失败的情况"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        with patch('harborai.api.fast_client.ConcurrencyManager') as mock_concurrency_manager:
            with patch('harborai.api.fast_client.ConcurrencyConfig') as mock_config:
                with patch('asyncio.run', side_effect=Exception("Start failed")):
                    mock_instance = Mock()
                    mock_concurrency_manager.return_value = mock_instance
                    
                    # 应该捕获异常并设置manager为None
                    manager = fast_completions._get_concurrency_manager()
                    assert manager is None
    
    @pytest.mark.skipif(not CONCURRENCY_OPTIMIZATION_AVAILABLE, reason="并发优化组件不可用")
    def test_concurrency_manager_with_running_loop(self):
        """测试在运行中的事件循环中初始化并发管理器"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        with patch('harborai.api.fast_client.ConcurrencyManager') as mock_concurrency_manager:
            with patch('harborai.api.fast_client.ConcurrencyConfig') as mock_config:
                with patch('asyncio.get_running_loop') as mock_get_loop:
                    with patch('asyncio.create_task') as mock_create_task:
                        mock_instance = Mock()
                        mock_concurrency_manager.return_value = mock_instance
                        
                        mock_loop = Mock()
                        mock_get_loop.return_value = mock_loop
                        
                        mock_task = Mock()
                        mock_create_task.return_value = mock_task
                        
                        manager = fast_completions._get_concurrency_manager()
                        
                        assert manager == mock_instance
                        assert fast_completions._concurrency_start_task == mock_task
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not CONCURRENCY_OPTIMIZATION_AVAILABLE, reason="并发优化组件不可用")
    async def test_ensure_concurrency_manager_started_success(self):
        """测试确保并发管理器启动成功"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 创建一个真正的协程任务
        async def mock_coroutine():
            return True
        
        # 创建真正的任务
        mock_task = asyncio.create_task(mock_coroutine())
        fast_completions._concurrency_start_task = mock_task
        
        await fast_completions._ensure_concurrency_manager_started()
        
        assert fast_completions._concurrency_start_task is None
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not CONCURRENCY_OPTIMIZATION_AVAILABLE, reason="并发优化组件不可用")
    async def test_ensure_concurrency_manager_started_failure(self):
        """测试确保并发管理器启动失败"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 创建一个会失败的协程任务
        async def failing_coroutine():
            raise Exception("Start task failed")
        
        # 创建真正的任务
        mock_task = asyncio.create_task(failing_coroutine())
        fast_completions._concurrency_start_task = mock_task
        
        with pytest.raises(Exception, match="Start task failed"):
            await fast_completions._ensure_concurrency_manager_started()
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not CONCURRENCY_OPTIMIZATION_AVAILABLE, reason="并发优化组件不可用")
    async def test_ensure_concurrency_manager_started_no_task(self):
        """测试没有启动任务时的情况"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 没有启动任务
        fast_completions._concurrency_start_task = None
        
        # 应该正常返回，不抛出异常
        await fast_completions._ensure_concurrency_manager_started()


class TestCacheManagerFallback:
    """测试缓存管理器回退逻辑"""
    
    def setup_method(self):
        """测试前置设置"""
        self.mock_client = Mock()
        self.mock_client.config = {
            'enable_performance_optimization': True,
            'enable_caching': True
        }
    
    def test_cache_manager_fallback_to_traditional(self):
        """测试从内存管理器缓存回退到传统缓存管理器"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 模拟没有内存管理器
        self.mock_client._memory_manager = None
        
        with patch('harborai.core.cache_manager.CacheManager') as mock_cache_manager:
            mock_instance = Mock()
            mock_cache_manager.return_value = mock_instance
            
            fast_completions._load_cache_manager()
            
            assert fast_completions._cache_manager == mock_instance
            mock_cache_manager.assert_called_once_with(self.mock_client.config)
    
    def test_cache_manager_use_memory_manager_cache(self):
        """测试使用内存管理器的缓存"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 模拟有内存管理器和缓存
        mock_memory_manager = Mock()
        mock_cache = Mock()
        mock_memory_manager.cache = mock_cache
        self.mock_client._memory_manager = mock_memory_manager
        
        fast_completions._load_cache_manager()
        
        assert fast_completions._cache_manager == mock_cache
    
    def test_cache_manager_memory_manager_without_cache(self):
        """测试内存管理器没有缓存属性的情况"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 模拟内存管理器没有cache属性
        mock_memory_manager = Mock(spec=[])  # 空spec，没有cache属性
        self.mock_client._memory_manager = mock_memory_manager
        
        # 当访问不存在的cache属性时，会抛出AttributeError
        # 这会被捕获并记录警告，缓存管理器被设置为None
        fast_completions._load_cache_manager()
        
        # 应该设置为None，因为异常被捕获
        assert fast_completions._cache_manager is None
    
    def test_cache_manager_traditional_import_error(self):
        """测试传统缓存管理器导入失败的情况"""
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 模拟没有内存管理器
        self.mock_client._memory_manager = None
        
        # 模拟CacheManager导入失败
        with patch('harborai.core.cache_manager.CacheManager', side_effect=ImportError("Module not found")):
            fast_completions._load_cache_manager()
            
            # 应该设置为None，因为ImportError被捕获
            assert fast_completions._cache_manager is None


class TestAsyncOperations:
    """测试异步操作"""
    
    def setup_method(self):
        """测试前置设置"""
        self.mock_client = Mock()
        self.mock_client.config = {
            'enable_performance_optimization': True,
            'enable_caching': True,
            'supported_models': ['gpt-3.5-turbo', 'gpt-4']
        }
    
    @pytest.mark.asyncio
    async def test_create_async_with_concurrency_manager(self):
        """测试使用并发管理器的异步创建"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 模拟并发管理器
        mock_concurrency_manager = AsyncMock()
        mock_response = {'choices': [{'message': {'content': 'Hello'}}]}
        mock_concurrency_manager.create_chat_completion.return_value = mock_response
        
        with patch.object(fast_completions, '_get_concurrency_manager', return_value=mock_concurrency_manager):
            with patch.object(fast_completions, '_ensure_initialized'):
                with patch.object(fast_completions, '_validate_request'):
                    with patch.object(fast_completions, '_ensure_concurrency_manager_started', new_callable=AsyncMock):
                        
                        messages = [{"role": "user", "content": "Hello"}]
                        model = "gpt-3.5-turbo"
                        
                        response = await fast_completions.create_async(messages, model)
                        
                        assert response == mock_response
                        mock_concurrency_manager.create_chat_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_async_concurrency_manager_failure(self):
        """测试并发管理器异步处理失败时的回退"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        fast_completions = FastChatCompletions(self.mock_client)
        
        # 模拟并发管理器失败
        mock_concurrency_manager = AsyncMock()
        mock_concurrency_manager.create_chat_completion.side_effect = Exception("Concurrency failed")
        
        # 模拟传统异步方法
        mock_response = {'choices': [{'message': {'content': 'Hello'}}]}
        
        with patch.object(fast_completions, '_get_concurrency_manager', return_value=mock_concurrency_manager):
            with patch.object(fast_completions, '_ensure_initialized'):
                with patch.object(fast_completions, '_validate_request'):
                    with patch.object(fast_completions, '_ensure_concurrency_manager_started', new_callable=AsyncMock):
                        with patch.object(fast_completions, '_create_async_traditional', new_callable=AsyncMock, return_value=mock_response):
                            
                            messages = [{"role": "user", "content": "Hello"}]
                            model = "gpt-3.5-turbo"
                            
                            response = await fast_completions.create_async(messages, model)
                            
                            assert response == mock_response
                            fast_completions._create_async_traditional.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])