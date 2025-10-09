#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastClient 覆盖率改进测试

专门针对fast_client.py未覆盖代码路径的测试用例，
目标是将覆盖率从64%提升到80%+。

重点测试：
1. 并发管理器的实际使用路径
2. 缓存机制的完整工作流程
3. 异步方法的完整测试
4. 错误处理和回退机制
5. 内存优化功能
6. 边界条件和异常情况
"""

import asyncio
import pytest
import time
import json
import hashlib
from unittest.mock import Mock, patch, MagicMock, AsyncMock, PropertyMock, call
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


class TestFastChatCompletionsCoverageImprovement:
    """FastChatCompletions覆盖率改进测试"""
    
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
        self.fast_completions = FastChatCompletions(self.mock_client)
    
    def test_get_concurrency_manager_with_available_optimization(self):
        """测试并发管理器获取（优化可用时）"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        with patch('harborai.api.fast_client.ConcurrencyManager') as mock_cm_class, \
             patch('harborai.api.fast_client.ConcurrencyConfig') as mock_config_class, \
             patch('asyncio.get_running_loop') as mock_get_loop, \
             patch('asyncio.run') as mock_run:
            
            # 模拟配置
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            
            # 模拟并发管理器
            mock_cm = Mock()
            mock_cm_class.return_value = mock_cm
            
            # 模拟没有运行中的事件循环
            mock_get_loop.side_effect = RuntimeError("No running event loop")
            
            # 模拟start方法
            async def mock_start():
                pass
            mock_cm.start = AsyncMock(side_effect=mock_start)
            
            # 调用方法
            manager = self.fast_completions._get_concurrency_manager()
            
            # 验证结果
            assert manager == mock_cm
            mock_config_class.assert_called_once()
            mock_cm_class.assert_called_once_with(mock_config)
            mock_run.assert_called_once()
    
    def test_get_concurrency_manager_with_running_loop(self):
        """测试并发管理器获取（有运行中的事件循环）"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        with patch('harborai.api.fast_client.ConcurrencyManager') as mock_cm_class, \
             patch('harborai.api.fast_client.ConcurrencyConfig') as mock_config_class, \
             patch('asyncio.get_running_loop') as mock_get_loop, \
             patch('asyncio.create_task') as mock_create_task:
            
            # 模拟配置
            mock_config = Mock()
            mock_config_class.return_value = mock_config
            
            # 模拟并发管理器
            mock_cm = Mock()
            mock_cm_class.return_value = mock_cm
            
            # 模拟有运行中的事件循环
            mock_loop = Mock()
            mock_get_loop.return_value = mock_loop
            
            # 模拟任务
            mock_task = Mock()
            mock_create_task.return_value = mock_task
            
            # 模拟start方法
            async def mock_start():
                pass
            mock_cm.start = AsyncMock(side_effect=mock_start)
            
            # 调用方法
            manager = self.fast_completions._get_concurrency_manager()
            
            # 验证结果
            assert manager == mock_cm
            assert self.fast_completions._concurrency_start_task == mock_task
            mock_create_task.assert_called_once()
    
    def test_get_concurrency_manager_initialization_error(self):
        """测试并发管理器初始化错误"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        with patch('harborai.api.fast_client.ConcurrencyManager') as mock_cm_class, \
             patch('harborai.api.fast_client.ConcurrencyConfig') as mock_config_class:
            
            # 模拟初始化错误
            mock_config_class.side_effect = Exception("初始化失败")
            
            # 调用方法
            manager = self.fast_completions._get_concurrency_manager()
            
            # 验证结果
            assert manager is None
    
    def test_ensure_initialized_with_performance_and_cache(self):
        """测试确保初始化（启用性能和缓存优化）"""
        with patch.object(self.fast_completions, '_load_performance_manager') as mock_load_perf, \
             patch.object(self.fast_completions, '_load_cache_manager') as mock_load_cache, \
             patch('harborai.api.fast_client.get_lazy_plugin_manager') as mock_get_manager:
            
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            # 调用方法
            self.fast_completions._ensure_initialized()
            
            # 验证结果
            assert self.fast_completions._initialized is True
            assert self.fast_completions._lazy_manager == mock_manager
            mock_load_perf.assert_called_once()
            mock_load_cache.assert_called_once()
    
    def test_ensure_initialized_disabled_optimizations(self):
        """测试确保初始化（禁用优化）"""
        # 修改配置
        self.fast_completions._client.config = {
            'enable_performance_optimization': False,
            'enable_caching': False
        }
        
        with patch.object(self.fast_completions, '_load_performance_manager') as mock_load_perf, \
             patch.object(self.fast_completions, '_load_cache_manager') as mock_load_cache, \
             patch('harborai.api.fast_client.get_lazy_plugin_manager') as mock_get_manager:
            
            mock_manager = Mock()
            mock_get_manager.return_value = mock_manager
            
            # 调用方法
            self.fast_completions._ensure_initialized()
            
            # 验证结果
            assert self.fast_completions._initialized is True
            mock_load_perf.assert_not_called()
            mock_load_cache.assert_not_called()
    
    def test_ensure_initialized_error(self):
        """测试确保初始化错误"""
        with patch('harborai.api.fast_client.get_lazy_plugin_manager') as mock_get_manager:
            mock_get_manager.side_effect = Exception("初始化失败")
            
            # 调用方法并验证异常
            with pytest.raises(HarborAIError, match="组件初始化失败"):
                self.fast_completions._ensure_initialized()
    
    def test_load_cache_manager_with_memory_manager(self):
        """测试加载缓存管理器（使用内存管理器）"""
        # 模拟客户端有内存管理器
        mock_memory_manager = Mock()
        mock_cache = Mock()
        mock_memory_manager.cache = mock_cache
        self.fast_completions._client._memory_manager = mock_memory_manager
        
        # 调用方法
        self.fast_completions._load_cache_manager()
        
        # 验证结果
        assert self.fast_completions._cache_manager == mock_cache
    
    def test_load_cache_manager_fallback(self):
        """测试加载缓存管理器（回退到传统方式）"""
        # 确保客户端没有内存管理器
        self.fast_completions._client._memory_manager = None
        
        with patch('harborai.core.cache_manager.CacheManager') as mock_cache_class:
            mock_cache = Mock()
            mock_cache_class.return_value = mock_cache
            
            # 调用方法
            self.fast_completions._load_cache_manager()
            
            # 验证结果
            assert self.fast_completions._cache_manager == mock_cache
            mock_cache_class.assert_called_once_with(self.fast_completions._client.config)
    
    def test_load_cache_manager_import_error(self):
        """测试加载缓存管理器导入错误"""
        # 确保客户端没有内存管理器
        self.fast_completions._client._memory_manager = None
        
        with patch('harborai.core.cache_manager.CacheManager', side_effect=ImportError("模块不存在")):
            # 调用方法
            self.fast_completions._load_cache_manager()
            
            # 验证结果
            assert self.fast_completions._cache_manager is None
    
    def test_load_cache_manager_general_error(self):
        """测试加载缓存管理器一般错误"""
        # 确保客户端没有内存管理器
        self.fast_completions._client._memory_manager = None
        
        with patch('harborai.core.cache_manager.CacheManager', side_effect=Exception("加载失败")):
            # 调用方法
            self.fast_completions._load_cache_manager()
            
            # 验证结果
            assert self.fast_completions._cache_manager is None
    
    @pytest.mark.asyncio
    async def test_ensure_concurrency_manager_started_success(self):
        """测试确保并发管理器启动成功"""
        # 模拟启动任务
        async def mock_task():
            pass
        
        task = asyncio.create_task(mock_task())
        self.fast_completions._concurrency_start_task = task
        
        # 调用方法
        await self.fast_completions._ensure_concurrency_manager_started()
        
        # 验证结果
        assert self.fast_completions._concurrency_start_task is None
    
    @pytest.mark.asyncio
    async def test_ensure_concurrency_manager_started_error(self):
        """测试确保并发管理器启动错误"""
        # 模拟启动任务失败
        async def mock_task():
            raise Exception("启动失败")
        
        task = asyncio.create_task(mock_task())
        self.fast_completions._concurrency_start_task = task
        
        # 调用方法并验证异常
        with pytest.raises(Exception, match="启动失败"):
            await self.fast_completions._ensure_concurrency_manager_started()
    
    @pytest.mark.asyncio
    async def test_ensure_concurrency_manager_started_no_task(self):
        """测试确保并发管理器启动（无任务）"""
        # 没有启动任务
        self.fast_completions._concurrency_start_task = None
        
        # 调用方法（应该不做任何事）
        await self.fast_completions._ensure_concurrency_manager_started()
        
        # 验证结果（无异常即可）
        assert True
    
    def test_create_with_concurrency_manager_success(self):
        """测试使用并发管理器创建聊天完成（成功）"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        # 模拟并发管理器
        mock_cm = Mock()
        mock_response = {"choices": [{"message": {"content": "测试响应"}}]}
        
        async def mock_create_completion(**kwargs):
            return mock_response
        
        mock_cm.create_chat_completion = AsyncMock(side_effect=mock_create_completion)
        
        with patch.object(self.fast_completions, '_ensure_initialized'), \
             patch.object(self.fast_completions, '_validate_request'), \
             patch.object(self.fast_completions, '_get_concurrency_manager', return_value=mock_cm), \
             patch('asyncio.run') as mock_run:
            
            mock_run.return_value = mock_response
            
            messages = [{"role": "user", "content": "测试"}]
            model = "gpt-3.5-turbo"
            
            # 调用方法
            result = self.fast_completions.create(messages, model, stream=False)
            
            # 验证结果
            assert result == mock_response
            mock_run.assert_called_once()
    
    def test_create_with_concurrency_manager_error_fallback(self):
        """测试并发管理器错误回退到传统方式"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        # 模拟并发管理器
        mock_cm = Mock()
        mock_cm.create_chat_completion = AsyncMock(side_effect=Exception("并发管理器错误"))
        
        with patch.object(self.fast_completions, '_ensure_initialized'), \
             patch.object(self.fast_completions, '_validate_request'), \
             patch.object(self.fast_completions, '_get_concurrency_manager', return_value=mock_cm), \
             patch.object(self.fast_completions, '_create_traditional') as mock_traditional, \
             patch('asyncio.run', side_effect=Exception("并发管理器错误")):
            
            mock_traditional.return_value = {"fallback": "response"}
            
            messages = [{"role": "user", "content": "测试"}]
            model = "gpt-3.5-turbo"
            
            # 调用方法
            result = self.fast_completions.create(messages, model, stream=False)
            
            # 验证结果
            assert result == {"fallback": "response"}
            mock_traditional.assert_called_once_with(messages, model, False)
    
    def test_create_traditional_with_cache_hit(self):
        """测试传统方式创建（缓存命中）"""
        # 模拟缓存管理器
        mock_cache = Mock()
        cached_response = {"cached": "response"}
        mock_cache.get.return_value = cached_response
        self.fast_completions._cache_manager = mock_cache
        
        # 模拟延迟管理器
        mock_lazy_manager = Mock()
        mock_plugin = Mock()
        mock_lazy_manager.get_plugin_for_model.return_value = mock_plugin
        self.fast_completions._lazy_manager = mock_lazy_manager
        
        with patch.object(self.fast_completions, '_generate_cache_key', return_value="test_key"), \
             patch.object(self.fast_completions, '_format_messages', return_value=[]):
            
            messages = [{"role": "user", "content": "测试"}]
            model = "gpt-3.5-turbo"
            
            # 调用方法
            result = self.fast_completions._create_traditional(messages, model, stream=False)
            
            # 验证结果
            assert result == cached_response
            mock_cache.get.assert_called_once_with("test_key")
            # 插件不应该被调用
            mock_plugin.chat_completion.assert_not_called()
    
    def test_create_traditional_with_cache_miss_and_performance_tracking(self):
        """测试传统方式创建（缓存未命中，性能跟踪）"""
        # 模拟缓存管理器
        mock_cache = Mock()
        mock_cache.get.return_value = None  # 缓存未命中
        self.fast_completions._cache_manager = mock_cache
        
        # 模拟性能管理器
        mock_perf = Mock()
        self.fast_completions._performance_manager = mock_perf
        
        # 模拟延迟管理器和插件
        mock_lazy_manager = Mock()
        mock_plugin = Mock()
        plugin_response = {"plugin": "response"}
        mock_plugin.chat_completion.return_value = plugin_response
        mock_lazy_manager.get_plugin_for_model.return_value = mock_plugin
        self.fast_completions._lazy_manager = mock_lazy_manager
        
        with patch.object(self.fast_completions, '_generate_cache_key', return_value="test_key"), \
             patch.object(self.fast_completions, '_format_messages', return_value=[{"role": "user", "content": "测试"}]):
            
            messages = [{"role": "user", "content": "测试"}]
            model = "gpt-3.5-turbo"
            
            # 调用方法
            result = self.fast_completions._create_traditional(messages, model, stream=False)
            
            # 验证结果
            assert result == plugin_response
            mock_cache.get.assert_called_once_with("test_key")
            mock_cache.set.assert_called_once_with("test_key", plugin_response)
            mock_perf.record_request.assert_called_once()
            mock_plugin.chat_completion.assert_called_once()
    
    def test_create_traditional_unsupported_model(self):
        """测试传统方式创建（不支持的模型）"""
        # 模拟延迟管理器
        mock_lazy_manager = Mock()
        mock_lazy_manager.get_plugin_for_model.return_value = None  # 不支持的模型
        self.fast_completions._lazy_manager = mock_lazy_manager
        
        messages = [{"role": "user", "content": "测试"}]
        model = "unsupported-model"
        
        # 调用方法并验证异常
        with pytest.raises(ValidationError, match="不支持的模型"):
            self.fast_completions._create_traditional(messages, model, stream=False)
    
    def test_create_traditional_plugin_error(self):
        """测试传统方式创建（插件错误）"""
        # 模拟延迟管理器和插件
        mock_lazy_manager = Mock()
        mock_plugin = Mock()
        mock_plugin.chat_completion.side_effect = Exception("插件错误")
        mock_lazy_manager.get_plugin_for_model.return_value = mock_plugin
        self.fast_completions._lazy_manager = mock_lazy_manager
        
        with patch.object(self.fast_completions, '_format_messages', return_value=[]):
            
            messages = [{"role": "user", "content": "测试"}]
            model = "gpt-3.5-turbo"
            
            # 调用方法并验证异常
            with pytest.raises(Exception, match="插件错误"):
                self.fast_completions._create_traditional(messages, model, stream=False)
    
    @pytest.mark.asyncio
    async def test_create_async_with_concurrency_manager_success(self):
        """测试异步创建（并发管理器成功）"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        # 模拟并发管理器
        mock_cm = Mock()
        mock_response = {"choices": [{"message": {"content": "异步测试响应"}}]}
        mock_cm.create_chat_completion = AsyncMock(return_value=mock_response)
        
        with patch.object(self.fast_completions, '_ensure_initialized'), \
             patch.object(self.fast_completions, '_validate_request'), \
             patch.object(self.fast_completions, '_get_concurrency_manager', return_value=mock_cm), \
             patch.object(self.fast_completions, '_ensure_concurrency_manager_started', new_callable=AsyncMock):
            
            messages = [{"role": "user", "content": "异步测试"}]
            model = "gpt-3.5-turbo"
            
            # 调用方法
            result = await self.fast_completions.create_async(messages, model, stream=False)
            
            # 验证结果
            assert result == mock_response
            mock_cm.create_chat_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_async_with_concurrency_manager_error_fallback(self):
        """测试异步创建（并发管理器错误回退）"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        # 模拟并发管理器
        mock_cm = Mock()
        mock_cm.create_chat_completion = AsyncMock(side_effect=Exception("异步并发错误"))
        
        with patch.object(self.fast_completions, '_ensure_initialized'), \
             patch.object(self.fast_completions, '_validate_request'), \
             patch.object(self.fast_completions, '_get_concurrency_manager', return_value=mock_cm), \
             patch.object(self.fast_completions, '_ensure_concurrency_manager_started', new_callable=AsyncMock), \
             patch.object(self.fast_completions, '_create_async_traditional', new_callable=AsyncMock) as mock_traditional:
            
            mock_traditional.return_value = {"async_fallback": "response"}
            
            messages = [{"role": "user", "content": "异步测试"}]
            model = "gpt-3.5-turbo"
            
            # 调用方法
            result = await self.fast_completions.create_async(messages, model, stream=False)
            
            # 验证结果
            assert result == {"async_fallback": "response"}
            mock_traditional.assert_called_once_with(messages, model, False)
    
    @pytest.mark.asyncio
    async def test_create_async_traditional_success(self):
        """测试异步传统方式创建（成功）"""
        # 模拟延迟管理器和插件
        mock_lazy_manager = Mock()
        mock_plugin = Mock()
        plugin_response = {"async_plugin": "response"}
        mock_plugin.chat_completion_async = AsyncMock(return_value=plugin_response)
        mock_lazy_manager.get_plugin_for_model.return_value = mock_plugin
        self.fast_completions._lazy_manager = mock_lazy_manager
        
        with patch.object(self.fast_completions, '_format_messages', return_value=[{"role": "user", "content": "异步测试"}]):
            
            messages = [{"role": "user", "content": "异步测试"}]
            model = "gpt-3.5-turbo"
            
            # 调用方法
            result = await self.fast_completions._create_async_traditional(messages, model, stream=False)
            
            # 验证结果
            assert result == plugin_response
            mock_plugin.chat_completion_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_async_traditional_unsupported_model(self):
        """测试异步传统方式创建（不支持的模型）"""
        # 模拟延迟管理器
        mock_lazy_manager = Mock()
        mock_lazy_manager.get_plugin_for_model.return_value = None  # 不支持的模型
        self.fast_completions._lazy_manager = mock_lazy_manager
        
        messages = [{"role": "user", "content": "异步测试"}]
        model = "unsupported-async-model"
        
        # 调用方法并验证异常
        with pytest.raises(ValidationError, match="不支持的模型"):
            await self.fast_completions._create_async_traditional(messages, model, stream=False)
    
    @pytest.mark.asyncio
    async def test_create_async_traditional_plugin_error(self):
        """测试异步传统方式创建（插件错误）"""
        # 模拟延迟管理器和插件
        mock_lazy_manager = Mock()
        mock_plugin = Mock()
        mock_plugin.chat_completion_async = AsyncMock(side_effect=Exception("异步插件错误"))
        mock_lazy_manager.get_plugin_for_model.return_value = mock_plugin
        self.fast_completions._lazy_manager = mock_lazy_manager
        
        with patch.object(self.fast_completions, '_format_messages', return_value=[]):
            
            messages = [{"role": "user", "content": "异步测试"}]
            model = "gpt-3.5-turbo"
            
            # 调用方法并验证异常
            with pytest.raises(Exception, match="异步插件错误"):
                await self.fast_completions._create_async_traditional(messages, model, stream=False)
    
    def test_validate_request_temperature_boundary(self):
        """测试请求验证（temperature边界值）"""
        messages = [{"role": "user", "content": "测试"}]
        model = "gpt-3.5-turbo"
        
        # 测试有效的temperature值
        self.fast_completions._validate_request(messages, model, temperature=0.0)
        self.fast_completions._validate_request(messages, model, temperature=1.0)
        self.fast_completions._validate_request(messages, model, temperature=2.0)
        
        # 测试无效的temperature值
        with pytest.raises(ValidationError, match="temperature必须在0-2之间"):
            self.fast_completions._validate_request(messages, model, temperature=-0.1)
        
        with pytest.raises(ValidationError, match="temperature必须在0-2之间"):
            self.fast_completions._validate_request(messages, model, temperature=2.1)
    
    def test_validate_request_max_tokens_boundary(self):
        """测试请求验证（max_tokens边界值）"""
        messages = [{"role": "user", "content": "测试"}]
        model = "gpt-3.5-turbo"
        
        # 测试有效的max_tokens值
        self.fast_completions._validate_request(messages, model, max_tokens=1)
        self.fast_completions._validate_request(messages, model, max_tokens=1000)
        
        # 测试无效的max_tokens值
        with pytest.raises(ValidationError, match="max_tokens必须大于0"):
            self.fast_completions._validate_request(messages, model, max_tokens=0)
        
        with pytest.raises(ValidationError, match="max_tokens必须大于0"):
            self.fast_completions._validate_request(messages, model, max_tokens=-1)
    
    def test_format_messages_with_additional_fields(self):
        """测试消息格式化（包含额外字段）"""
        messages = [
            {
                "role": "user",
                "content": "测试消息",
                "name": "test_user",
                "timestamp": "2023-01-01T00:00:00Z"
            },
            {
                "role": "assistant",
                "content": "测试响应",
                "function_call": {"name": "test_function", "arguments": "{}"}
            }
        ]
        
        # 调用方法
        formatted = self.fast_completions._format_messages(messages)
        
        # 验证结果
        assert len(formatted) == 2
        
        # 验证第一条消息
        assert formatted[0]["role"] == "user"
        assert formatted[0]["content"] == "测试消息"
        assert formatted[0]["name"] == "test_user"
        assert formatted[0]["timestamp"] == "2023-01-01T00:00:00Z"
        
        # 验证第二条消息
        assert formatted[1]["role"] == "assistant"
        assert formatted[1]["content"] == "测试响应"
        assert formatted[1]["function_call"] == {"name": "test_function", "arguments": "{}"}
    
    def test_generate_cache_key_consistency(self):
        """测试缓存键生成的一致性"""
        messages = [{"role": "user", "content": "测试"}]
        model = "gpt-3.5-turbo"
        kwargs = {"temperature": 0.7, "max_tokens": 100}
        
        # 多次生成应该得到相同的键
        key1 = self.fast_completions._generate_cache_key(messages, model, **kwargs)
        key2 = self.fast_completions._generate_cache_key(messages, model, **kwargs)
        
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5哈希长度
    
    def test_generate_cache_key_different_inputs(self):
        """测试缓存键生成（不同输入）"""
        messages1 = [{"role": "user", "content": "测试1"}]
        messages2 = [{"role": "user", "content": "测试2"}]
        model = "gpt-3.5-turbo"
        
        key1 = self.fast_completions._generate_cache_key(messages1, model)
        key2 = self.fast_completions._generate_cache_key(messages2, model)
        
        assert key1 != key2
    
    def test_generate_cache_key_excludes_stream(self):
        """测试缓存键生成（排除stream参数）"""
        messages = [{"role": "user", "content": "测试"}]
        model = "gpt-3.5-turbo"
        
        # stream参数应该被排除
        key1 = self.fast_completions._generate_cache_key(messages, model, stream=True, temperature=0.7)
        key2 = self.fast_completions._generate_cache_key(messages, model, stream=False, temperature=0.7)
        
        assert key1 == key2


class TestFastHarborAICoverageImprovement:
    """FastHarborAI覆盖率改进测试"""
    
    def test_merge_config_with_get_settings_error(self):
        """测试配置合并（get_settings错误）"""
        with patch('harborai.api.fast_client.get_settings', side_effect=Exception("设置获取失败")):
            client = FastHarborAI()
            
            # 应该使用空的默认配置
            assert isinstance(client.config, dict)
    
    def test_init_memory_optimization_unavailable(self):
        """测试内存优化初始化（组件不可用）"""
        with patch('harborai.api.fast_client.MEMORY_OPTIMIZATION_AVAILABLE', False):
            config = {'enable_memory_optimization': True}
            client = FastHarborAI(config=config)
            
            # 内存管理器应该为None
            assert client._memory_manager is None
    
    def test_init_memory_optimization_with_custom_config(self):
        """测试内存优化初始化（自定义配置）"""
        if not MEMORY_OPTIMIZATION_AVAILABLE:
            pytest.skip("内存优化组件不可用")
        
        config = {
            'enable_memory_optimization': True,
            'memory_optimization': {
                'cache_size': 2000,
                'object_pool_size': 200,
                'enable_weak_references': False,
                'memory_threshold': 100.0,
                'cleanup_interval': 600
            }
        }
        
        with patch('harborai.api.fast_client.MemoryManager') as mock_mm_class:
            mock_mm = Mock()
            mock_mm_class.return_value = mock_mm
            
            client = FastHarborAI(config=config)
            
            # 验证内存管理器被正确初始化
            assert client._memory_manager == mock_mm
            mock_mm_class.assert_called_once_with(
                cache_size=2000,
                object_pool_size=200,
                enable_weak_references=False,
                memory_threshold_mb=100.0,
                auto_cleanup_interval=600
            )
    
    def test_init_memory_optimization_error(self):
        """测试内存优化初始化错误"""
        if not MEMORY_OPTIMIZATION_AVAILABLE:
            pytest.skip("内存优化组件不可用")
        
        config = {'enable_memory_optimization': True}
        
        with patch('harborai.api.fast_client.MemoryManager', side_effect=Exception("内存管理器初始化失败")):
            client = FastHarborAI(config=config)
            
            # 内存管理器应该为None
            assert client._memory_manager is None
    
    def test_preload_model_success(self):
        """测试模型预加载（成功）"""
        with patch('harborai.api.fast_client.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_plugin_name_for_model.return_value = "openai_plugin"
            mock_manager.preload_plugin.return_value = True
            mock_get_manager.return_value = mock_manager
            
            client = FastHarborAI()
            result = client.preload_model("gpt-3.5-turbo")
            
            assert result is True
            mock_manager.get_plugin_name_for_model.assert_called_once_with("gpt-3.5-turbo")
            mock_manager.preload_plugin.assert_called_once_with("openai_plugin")
    
    def test_preload_model_no_plugin(self):
        """测试模型预加载（无对应插件）"""
        with patch('harborai.api.fast_client.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_plugin_name_for_model.return_value = None
            mock_get_manager.return_value = mock_manager
            
            client = FastHarborAI()
            result = client.preload_model("unsupported-model")
            
            assert result is False
            mock_manager.preload_plugin.assert_not_called()
    
    def test_get_statistics_with_memory_manager(self):
        """测试获取统计信息（包含内存管理器）"""
        with patch('harborai.api.fast_client.get_lazy_plugin_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_statistics.return_value = {"plugins_loaded": 2}
            mock_get_manager.return_value = mock_manager
            
            # 模拟内存管理器
            mock_memory_manager = Mock()
            mock_memory_manager.get_memory_stats.return_value = {"cache_hits": 100}
            
            client = FastHarborAI()
            client._memory_manager = mock_memory_manager
            
            stats = client.get_statistics()
            
            assert stats["plugins_loaded"] == 2
            assert stats["client_type"] == "FastHarborAI"
            assert stats["memory_optimization_enabled"] is True
            assert stats["memory_stats"] == {"cache_hits": 100}
            assert "initialized_at" in stats
            assert "uptime_seconds" in stats
    
    def test_get_memory_stats_with_psutil(self):
        """测试获取内存统计（包含psutil）"""
        # 模拟内存管理器
        mock_memory_manager = Mock()
        mock_memory_manager.get_memory_stats.return_value = {"cache_hits": 100}
        
        # 模拟psutil
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100MB
        mock_memory_info.vms = 200 * 1024 * 1024  # 200MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 5.0
        
        with patch('psutil.Process', return_value=mock_process), \
             patch('os.getpid', return_value=12345):
            
            client = FastHarborAI()
            client._memory_manager = mock_memory_manager
            
            stats = client.get_memory_stats()
            
            assert stats["cache_hits"] == 100
            assert stats["system_memory"]["rss_mb"] == 100.0
            assert stats["system_memory"]["vms_mb"] == 200.0
            assert stats["system_memory"]["percent"] == 5.0
    
    def test_get_memory_stats_psutil_import_error(self):
        """测试获取内存统计（psutil导入错误）"""
        # 模拟内存管理器
        mock_memory_manager = Mock()
        mock_memory_manager.get_memory_stats.return_value = {"cache_hits": 100}
        
        with patch('psutil.Process', side_effect=ImportError("psutil not available")):
            client = FastHarborAI()
            client._memory_manager = mock_memory_manager
            
            stats = client.get_memory_stats()
            
            assert stats["cache_hits"] == 100
            assert stats["system_memory"]["error"] == "psutil not available"
    
    def test_get_memory_stats_psutil_general_error(self):
        """测试获取内存统计（psutil一般错误）"""
        # 模拟内存管理器
        mock_memory_manager = Mock()
        mock_memory_manager.get_memory_stats.return_value = {"cache_hits": 100}
        
        with patch('psutil.Process', side_effect=Exception("psutil error")):
            client = FastHarborAI()
            client._memory_manager = mock_memory_manager
            
            stats = client.get_memory_stats()
            
            assert stats["cache_hits"] == 100
            assert stats["system_memory"]["error"] == "psutil error"
    
    def test_get_memory_stats_no_memory_manager(self):
        """测试获取内存统计（无内存管理器）"""
        client = FastHarborAI()
        client._memory_manager = None
        
        stats = client.get_memory_stats()
        
        assert stats is None
    
    def test_cleanup_memory_with_manager(self):
        """测试内存清理（有内存管理器）"""
        # 模拟内存管理器
        mock_memory_manager = Mock()
        mock_memory_manager.cleanup.return_value = {"cleaned_items": 50}
        
        client = FastHarborAI()
        client._memory_manager = mock_memory_manager
        
        result = client.cleanup_memory(force_clear=True)
        
        assert result == {"cleaned_items": 50}
        mock_memory_manager.cleanup.assert_called_once_with(force_clear=True)
    
    def test_cleanup_memory_no_manager(self):
        """测试内存清理（无内存管理器）"""
        client = FastHarborAI()
        client._memory_manager = None
        
        result = client.cleanup_memory()
        
        assert result is None
    
    def test_cleanup_with_performance_manager(self):
        """测试客户端清理（包含性能管理器）"""
        # 模拟聊天接口和性能管理器
        mock_chat = Mock()
        mock_completions = Mock()
        mock_perf_manager = Mock()
        mock_completions._performance_manager = mock_perf_manager
        mock_chat.completions = mock_completions
        
        # 模拟内存管理器
        mock_memory_manager = Mock()
        
        client = FastHarborAI()
        client._chat = mock_chat
        client._memory_manager = mock_memory_manager
        
        # 调用清理
        client.cleanup()
        
        # 验证清理调用
        mock_perf_manager.cleanup.assert_called_once()
        mock_memory_manager.shutdown.assert_called_once()
    
    def test_cleanup_memory_manager_error(self):
        """测试客户端清理（内存管理器错误）"""
        # 模拟内存管理器错误
        mock_memory_manager = Mock()
        mock_memory_manager.shutdown.side_effect = Exception("清理失败")
        
        client = FastHarborAI()
        client._memory_manager = mock_memory_manager
        
        # 调用清理（不应该抛出异常）
        client.cleanup()
        
        # 验证清理被调用
        mock_memory_manager.shutdown.assert_called_once()
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with patch.object(FastHarborAI, 'cleanup') as mock_cleanup:
            with FastHarborAI() as client:
                assert isinstance(client, FastHarborAI)
            
            # 验证清理被调用
            mock_cleanup.assert_called_once()
    
    def test_destructor(self):
        """测试析构函数"""
        with patch.object(FastHarborAI, 'cleanup') as mock_cleanup:
            client = FastHarborAI()
            del client
            
            # 验证清理被调用
            mock_cleanup.assert_called_once()
    
    def test_destructor_with_exception(self):
        """测试析构函数（清理异常）"""
        with patch.object(FastHarborAI, 'cleanup', side_effect=Exception("清理异常")):
            client = FastHarborAI()
            # 析构时不应该抛出异常
            del client


class TestCreateFastClientFunction:
    """测试create_fast_client便捷函数"""
    
    def test_create_fast_client_with_config(self):
        """测试创建快速客户端（带配置）"""
        config = {"test_key": "test_value"}
        kwargs = {"extra_key": "extra_value"}
        
        client = create_fast_client(config=config, **kwargs)
        
        assert isinstance(client, FastHarborAI)
        assert client.config["test_key"] == "test_value"
        assert client.config["extra_key"] == "extra_value"
    
    def test_create_fast_client_no_config(self):
        """测试创建快速客户端（无配置）"""
        client = create_fast_client()
        
        assert isinstance(client, FastHarborAI)
        assert isinstance(client.config, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])