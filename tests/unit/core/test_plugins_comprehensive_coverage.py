#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 插件模块全面测试覆盖

测试目标：提高插件模块的测试覆盖率到90%以上
包括：
- DeepSeek插件测试
- OpenAI插件测试  
- Doubao插件测试
- Wenxin插件测试
- 插件基础功能测试

遵循VIBE编码规范，使用TDD方法。
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional

from harborai.core.base_plugin import BaseLLMPlugin, ModelInfo
from harborai.core.plugins.deepseek_plugin import DeepSeekPlugin
from harborai.core.plugins.openai_plugin import OpenAIPlugin
from harborai.core.plugins.doubao_plugin import DoubaoPlugin
from harborai.core.plugins.wenxin_plugin import WenxinPlugin
from harborai.core.plugins.base import Plugin, PluginInfo
from harborai.core.exceptions import PluginError, PluginLoadError, PluginConfigError


class TestDeepSeekPlugin:
    """DeepSeek插件测试"""
    
    def test_deepseek_plugin_initialization(self):
        """测试DeepSeek插件初始化"""
        plugin = DeepSeekPlugin()
        
        assert plugin.name == "deepseek"
        assert hasattr(plugin, 'supported_models')
        assert hasattr(plugin, 'api_key')
        assert hasattr(plugin, 'base_url')
        assert plugin.base_url == "https://api.deepseek.com"
        assert plugin.timeout == 60
        assert plugin.max_retries == 3
    
    def test_deepseek_plugin_with_config(self):
        """测试DeepSeek插件配置初始化"""
        config = {
            'api_key': 'test_api_key',
            'base_url': 'https://custom.deepseek.com',
            'timeout': 30,
            'max_retries': 5
        }
        
        plugin = DeepSeekPlugin(**config)
        
        assert plugin.api_key == 'test_api_key'
        assert plugin.base_url == 'https://custom.deepseek.com'
        assert plugin.timeout == 30
        assert plugin.max_retries == 5
    
    def test_deepseek_plugin_supported_models(self):
        """测试DeepSeek插件支持的模型"""
        plugin = DeepSeekPlugin()
        models = plugin.supported_models
        
        assert isinstance(models, list)
        assert len(models) >= 2
        
        # 检查特定模型
        model_ids = [model.id for model in models]
        assert 'deepseek-chat' in model_ids
        assert 'deepseek-reasoner' in model_ids
        
        # 检查模型信息
        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.provider == "deepseek"
            assert model.max_tokens == 32768
            assert model.supports_streaming is True
    
    def test_deepseek_plugin_supports_model(self):
        """测试DeepSeek插件模型支持检查"""
        plugin = DeepSeekPlugin()
        
        assert plugin.supports_model('deepseek-chat') is True
        assert plugin.supports_model('deepseek-reasoner') is True
        assert plugin.supports_model('nonexistent-model') is False
    
    def test_deepseek_plugin_get_model_info(self):
        """测试DeepSeek插件获取模型信息"""
        plugin = DeepSeekPlugin()
        
        model_info = plugin.get_model_info('deepseek-chat')
        assert model_info is not None
        assert model_info.id == 'deepseek-chat'
        assert model_info.name == 'DeepSeek Chat'
        
        # 测试不存在的模型
        model_info = plugin.get_model_info('nonexistent-model')
        assert model_info is None
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_deepseek_plugin_chat_completion_async(self, mock_client):
        """测试DeepSeek插件异步聊天完成"""
        plugin = DeepSeekPlugin(api_key='test_key')
        
        # 模拟HTTP响应
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Hello from DeepSeek!',
                    'role': 'assistant'
                }
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 5,
                'total_tokens': 15
            }
        }
        mock_response.status_code = 200
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        from harborai.core.base_plugin import ChatMessage
        messages = [ChatMessage(role='user', content='Hello')]
        
        # 测试异步聊天完成
        response = await plugin.chat_completion_async(
            model='deepseek-chat',
            messages=messages
        )
        
        assert response is not None


class TestOpenAIPlugin:
    """OpenAI插件测试"""
    
    def test_openai_plugin_initialization(self):
        """测试OpenAI插件初始化"""
        plugin = OpenAIPlugin(name='openai', api_key='sk-test123')
        
        assert plugin.name == "openai"
        assert hasattr(plugin, 'supported_models')
        assert hasattr(plugin, 'api_key')
        assert hasattr(plugin, 'base_url')
    
    def test_openai_plugin_with_config(self):
        """测试OpenAI插件配置初始化"""
        config = {
            'api_key': 'sk-test123',
            'base_url': 'https://api.openai.com/v1',
            'organization': 'org-test'
        }
        
        plugin = OpenAIPlugin(**config)
        
        assert plugin.api_key == 'sk-test123'
        assert plugin.base_url == 'https://api.openai.com/v1'
    
    def test_openai_plugin_supported_models(self):
        """测试OpenAI插件支持的模型"""
        plugin = OpenAIPlugin(name='openai', api_key='sk-test123')
        models = plugin.supported_models
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # 检查模型信息
        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.provider == "openai"
    
    def test_openai_plugin_supports_model(self):
        """测试OpenAI插件模型支持检查"""
        plugin = OpenAIPlugin(name='openai', api_key='sk-test123')
        
        # 测试常见的OpenAI模型
        assert plugin.supports_model('gpt-4o') is True
        assert plugin.supports_model('gpt-4o-mini') is True
        assert plugin.supports_model('nonexistent-model') is False
    
    @patch('harborai.core.plugins.openai_plugin.AsyncOpenAI')
    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @pytest.mark.asyncio
    async def test_openai_plugin_chat_completion_async(self, mock_openai_sync, mock_openai_async):
        """测试OpenAI插件异步聊天完成"""
        # 创建完整的模拟响应对象
        mock_message = Mock()
        mock_message.role = 'assistant'
        mock_message.content = 'Hello from OpenAI!'
        mock_message.tool_calls = None  # 没有工具调用
        
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = 'stop'
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        
        mock_response = Mock()
        mock_response.id = 'chatcmpl-test123'
        mock_response.object = 'chat.completion'
        mock_response.created = 1234567890
        mock_response.model = 'gpt-4o'
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        # 模拟同步和异步客户端
        mock_sync_client = Mock()
        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create.return_value = mock_response
        
        mock_openai_sync.return_value = mock_sync_client
        mock_openai_async.return_value = mock_async_client
        
        plugin = OpenAIPlugin(api_key='sk-test123')
        
        from harborai.core.base_plugin import ChatMessage
        messages = [ChatMessage(role='user', content='Hello')]
        
        # 测试异步聊天完成
        response = await plugin.chat_completion_async(
            model='gpt-4o',  # 使用支持的模型
            messages=messages
        )
        
        assert response is not None
        assert response.choices[0].message.content == 'Hello from OpenAI!'
        assert response.usage.prompt_tokens == 10


class TestDoubaoPlugin:
    """Doubao插件测试"""
    
    def test_doubao_plugin_initialization(self):
        """测试Doubao插件初始化"""
        plugin = DoubaoPlugin(name='doubao', api_key='test_key')
        
        assert plugin.name == "doubao"
        assert hasattr(plugin, 'supported_models')
        assert hasattr(plugin, 'api_key')
        assert hasattr(plugin, 'base_url')
    
    def test_doubao_plugin_with_config(self):
        """测试Doubao插件配置初始化"""
        config = {
            'api_key': 'test_api_key',
            'base_url': 'https://ark.cn-beijing.volces.com/api/v3',
            'timeout': 30
        }
        
        plugin = DoubaoPlugin(**config)
        
        assert plugin.api_key == 'test_api_key'
        assert plugin.base_url == 'https://ark.cn-beijing.volces.com/api/v3'
    
    def test_doubao_plugin_supported_models(self):
        """测试Doubao插件支持的模型"""
        plugin = DoubaoPlugin(name='doubao', api_key='test_key')
        models = plugin.supported_models
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # 检查模型信息
        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.provider == "doubao"
    
    def test_doubao_plugin_supports_model(self):
        """测试Doubao插件模型支持检查"""
        plugin = DoubaoPlugin(name='doubao', api_key='test_key')
        
        # 测试常见的Doubao模型
        model_ids = [model.id for model in plugin.supported_models]
        if model_ids:
            assert plugin.supports_model(model_ids[0]) is True
        
        assert plugin.supports_model('nonexistent-model') is False
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_doubao_plugin_chat_completion_async(self, mock_client):
        """测试Doubao插件异步聊天完成"""
        plugin = DoubaoPlugin(api_key='test_key')
        
        # 模拟HTTP响应
        mock_response = Mock()
        mock_response.json.return_value = {
            'choices': [{
                'message': {
                    'content': 'Hello from Doubao!',
                    'role': 'assistant'
                }
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 5,
                'total_tokens': 15
            }
        }
        mock_response.status_code = 200
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance
        
        messages = [{'role': 'user', 'content': 'Hello'}]
        
        # 测试异步聊天完成 - 使用支持的模型
        response = await plugin.chat_completion_async(
            model='doubao-1-5-pro-32k-character-250715',  # 使用实际支持的模型
            messages=messages
        )
        
        assert response is not None


class TestWenxinPlugin:
    """Wenxin插件测试"""
    
    def test_wenxin_plugin_initialization(self):
        """测试Wenxin插件初始化"""
        plugin = WenxinPlugin(name='wenxin', api_key='test_key')
        assert plugin is not None
        assert hasattr(plugin, 'supported_models')
    
    def test_wenxin_plugin_with_config(self):
        """测试Wenxin插件配置"""
        config = {
            'api_key': 'test_key',
            'timeout': 30
        }
        plugin = WenxinPlugin(name='wenxin', **config)
        assert plugin.api_key == 'test_key'
        assert plugin.timeout == 30
    
    def test_wenxin_plugin_supported_models(self):
        """测试Wenxin插件支持的模型"""
        plugin = WenxinPlugin(name='wenxin', api_key='test_key')
        models = plugin.supported_models
        
        assert isinstance(models, list)
        assert len(models) > 0
        
        # 检查模型信息
        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.provider == "wenxin"
    
    def test_wenxin_plugin_supports_model(self):
        """测试Wenxin插件模型支持检查"""
        plugin = WenxinPlugin(name='wenxin', api_key='test_key')
        
        # 测试常见的Wenxin模型
        model_ids = [model.id for model in plugin.supported_models]
        if model_ids:
            assert plugin.supports_model(model_ids[0]) is True
        
        assert plugin.supports_model('nonexistent-model') is False
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_wenxin_plugin_chat_completion_async(self, mock_client):
        """测试Wenxin插件异步聊天完成"""
        # 修复初始化参数 - 添加必需的name参数
        plugin = WenxinPlugin(name='wenxin', api_key='test_key', secret_key='test_secret')
        
        # 模拟访问令牌和聊天响应
        mock_token_response = Mock()
        mock_token_response.json.return_value = {
            'access_token': 'test_token',
            'expires_in': 3600
        }
        mock_token_response.status_code = 200
        
        mock_chat_response = Mock()
        mock_chat_response.json.return_value = {
            'result': 'Hello from Wenxin!',
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 5,
                'total_tokens': 15
            }
        }
        mock_chat_response.status_code = 200
        
        mock_client_instance = AsyncMock()
        mock_client_instance.post.side_effect = [mock_token_response, mock_chat_response]
        mock_client.return_value = mock_client_instance
        
        messages = [{'role': 'user', 'content': 'Hello'}]
        
        # 测试异步聊天完成
        response = await plugin.chat_completion_async(
            model='ernie-3.5-8k',
            messages=messages
        )
        
        assert response is not None


# BaseLLMPlugin是抽象类，无法直接实例化，因此跳过这些测试


class TestPluginInfo:
    """插件信息测试"""
    
    def test_plugin_info_creation(self):
        """测试插件信息创建"""
        info = PluginInfo(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin description",
            supported_models=["model1", "model2"],
            author="Test Author"
        )
        
        assert info.name == "test_plugin"
        assert info.version == "1.0.0"
        assert info.description == "Test plugin description"
        assert info.author == "Test Author"
        assert info.supported_models == ["model1", "model2"]
        assert info.dependencies == []
    
    def test_plugin_info_with_dependencies(self):
        """测试插件信息包含依赖"""
        info = PluginInfo(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            supported_models=["model1"],
            dependencies=["dep1", "dep2"]
        )
        
        assert info.dependencies == ["dep1", "dep2"]


class TestPluginErrorHandling:
    """插件错误处理测试"""
    
    def test_plugin_load_error(self):
        """测试插件加载错误"""
        error = PluginLoadError("Failed to load plugin", plugin_name="test_plugin")
        
        assert str(error) == "Failed to load plugin"
        assert error.plugin_name == "test_plugin"
        assert isinstance(error, PluginError)
    
    def test_plugin_config_error(self):
        """测试插件配置错误"""
        error = PluginConfigError("Invalid configuration", plugin_name="test_plugin")
        
        assert str(error) == "Invalid configuration"
        assert error.plugin_name == "test_plugin"
        assert isinstance(error, PluginError)


class TestPluginIntegration:
    """插件集成测试"""
    
    def test_plugin_lifecycle_management(self):
        """测试插件生命周期管理"""
        plugin = DeepSeekPlugin(api_key='test_key')
        
        # 验证初始状态
        assert plugin.name == "deepseek"
        assert plugin.api_key == 'test_key'
        
        # 验证模型支持
        assert len(plugin.supported_models) > 0
        assert plugin.supports_model('deepseek-chat') is True
    
    @pytest.mark.asyncio
    async def test_plugin_async_operations(self):
        """测试插件异步操作"""
        plugin = DeepSeekPlugin(api_key='test_key')
        
        # 测试异步方法存在
        assert hasattr(plugin, 'chat_completion_async')
        assert asyncio.iscoroutinefunction(plugin.chat_completion_async)
        
        # 测试其他插件的异步方法
        openai_plugin = OpenAIPlugin(api_key='sk-test')
        assert hasattr(openai_plugin, 'chat_completion_async')
        assert asyncio.iscoroutinefunction(openai_plugin.chat_completion_async)


# BaseLLMPlugin是抽象类，无法直接实例化，因此跳过这些测试