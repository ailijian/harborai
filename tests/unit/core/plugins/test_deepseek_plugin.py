#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek插件测试用例

测试目标：
- 基本功能测试：初始化、模型支持、客户端创建
- 边界条件测试：无效参数、空消息、错误配置
- 异常处理测试：网络错误、API错误、JSON解析错误
- 异步操作测试：异步聊天完成、流式响应
- 集成测试：完整的请求-响应流程
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List

from harborai.core.plugins.deepseek_plugin import DeepSeekPlugin
from harborai.core.base_plugin import ChatMessage, ChatCompletion, ChatCompletionChunk, ModelInfo
from harborai.utils.exceptions import PluginError, ValidationError


class TestDeepSeekPluginInitialization:
    """测试DeepSeek插件初始化功能"""
    
    def test_init_with_default_config(self):
        """测试使用默认配置初始化插件"""
        plugin = DeepSeekPlugin()
        
        assert plugin.name == "deepseek"
        assert plugin.base_url == "https://api.deepseek.com"
        assert plugin.timeout == 90  # 修正：默认值是90，不是60
        assert plugin.max_retries == 3
        assert plugin.api_key is None
        assert len(plugin._supported_models) == 2
    
    def test_init_with_custom_config(self):
        """测试使用自定义配置初始化插件"""
        config = {
            "api_key": "test-key-123",
            "base_url": "https://custom.deepseek.com",
            "timeout": 30,
            "max_retries": 5
        }
        
        plugin = DeepSeekPlugin(name="custom-deepseek", **config)
        
        assert plugin.name == "custom-deepseek"
        assert plugin.api_key == "test-key-123"
        assert plugin.base_url == "https://custom.deepseek.com"
        assert plugin.timeout == 30
        assert plugin.max_retries == 5
    
    def test_supported_models_configuration(self):
        """测试支持的模型配置"""
        plugin = DeepSeekPlugin()
        
        # 验证支持的模型
        models = plugin.supported_models
        assert len(models) == 2
        
        # 验证deepseek-chat模型
        chat_model = next((m for m in models if m.id == "deepseek-chat"), None)
        assert chat_model is not None
        assert chat_model.name == "DeepSeek Chat"
        assert chat_model.provider == "deepseek"
        assert chat_model.max_tokens == 32768
        assert chat_model.supports_streaming is True
        assert chat_model.supports_thinking is False
        assert chat_model.supports_structured_output is True
        
        # 验证deepseek-reasoner模型
        reasoner_model = next((m for m in models if m.id == "deepseek-reasoner"), None)
        assert reasoner_model is not None
        assert reasoner_model.name == "DeepSeek R1"
        assert reasoner_model.supports_thinking is True
    
    def test_is_thinking_model(self):
        """测试推理模型判断功能"""
        plugin = DeepSeekPlugin()
        
        assert plugin.is_thinking_model("deepseek-reasoner") is True
        assert plugin.is_thinking_model("deepseek-chat") is False
        assert plugin.is_thinking_model("unknown-model") is False


class TestDeepSeekPluginClientManagement:
    """测试DeepSeek插件客户端管理功能"""
    
    @patch('httpx.Client')
    def test_get_sync_client_creation(self, mock_httpx_client):
        """测试同步客户端创建"""
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        client = plugin._get_client()
        
        # 验证httpx.Client被调用
        mock_httpx_client.assert_called_once()
        
        # 验证返回的是mock客户端
        assert client == mock_client
    
    @patch('httpx.AsyncClient')
    def test_get_async_client_creation(self, mock_httpx_async_client):
        """测试异步客户端创建"""
        mock_client = AsyncMock()
        mock_httpx_async_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        client = plugin._get_async_client()
        
        # 验证httpx.AsyncClient被调用
        mock_httpx_async_client.assert_called_once()
        
        # 验证返回的是mock客户端
        assert client == mock_client
    
    def test_client_creation_without_httpx(self):
        """测试在没有httpx的情况下创建客户端"""
        plugin = DeepSeekPlugin(api_key="test-key")
        
        # 模拟httpx导入失败
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'httpx':
                    raise ImportError("No module named 'httpx'")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            with pytest.raises(PluginError, match="httpx not installed"):
                plugin._get_client()
            
            with pytest.raises(PluginError, match="httpx not installed"):
                plugin._get_async_client()
    
    @patch('httpx.Client')
    def test_client_singleton_behavior(self, mock_httpx_client):
        """测试客户端单例行为"""
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        
        # 多次调用应返回同一个客户端实例
        client1 = plugin._get_client()
        client2 = plugin._get_client()
        
        assert client1 == client2
        assert mock_httpx_client.call_count == 1


class TestDeepSeekPluginValidation:
    """测试DeepSeek插件请求验证功能"""
    
    def test_validate_request_success(self):
        """测试有效请求验证"""
        plugin = DeepSeekPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 应该不抛出异常
        plugin._validate_request("deepseek-chat", messages)
    
    def test_validate_request_unsupported_model(self):
        """测试不支持的模型验证"""
        plugin = DeepSeekPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError, match="Model unknown-model is not supported"):
            plugin._validate_request("unknown-model", messages)
    
    def test_validate_request_empty_messages(self):
        """测试空消息验证"""
        plugin = DeepSeekPlugin(api_key="test-key")
        
        with pytest.raises(ValidationError, match="Messages cannot be empty"):
            plugin._validate_request("deepseek-chat", [])
    
    def test_validate_request_missing_api_key(self):
        """测试缺少API密钥验证"""
        plugin = DeepSeekPlugin()  # 没有API密钥
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError, match="DeepSeek API key is required"):
            plugin._validate_request("deepseek-chat", messages)
    
    def test_validate_request_invalid_temperature(self):
        """测试无效温度参数验证"""
        plugin = DeepSeekPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            plugin._validate_request("deepseek-chat", messages, temperature=3.0)
        
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            plugin._validate_request("deepseek-chat", messages, temperature=-1.0)
    
    def test_validate_request_invalid_max_tokens(self):
        """测试无效最大令牌数验证"""
        plugin = DeepSeekPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError, match="max_tokens must be positive"):
            plugin._validate_request("deepseek-chat", messages, max_tokens=0)
        
        with pytest.raises(ValidationError, match="max_tokens must be positive"):
            plugin._validate_request("deepseek-chat", messages, max_tokens=-100)


class TestDeepSeekPluginRequestPreparation:
    """测试DeepSeek插件请求准备功能"""
    
    def test_prepare_basic_request(self):
        """测试基本请求准备"""
        plugin = DeepSeekPlugin()
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!")
        ]
        
        request_data = plugin._prepare_deepseek_request("deepseek-chat", messages)
        
        assert request_data["model"] == "deepseek-chat"
        assert len(request_data["messages"]) == 2
        assert request_data["messages"][0]["role"] == "user"
        assert request_data["messages"][0]["content"] == "Hello"
        assert request_data["messages"][1]["role"] == "assistant"
        assert request_data["messages"][1]["content"] == "Hi there!"
    
    def test_prepare_request_with_optional_params(self):
        """测试包含可选参数的请求准备"""
        plugin = DeepSeekPlugin()
        messages = [ChatMessage(role="user", content="Hello")]
        
        kwargs = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1000,
            "stop": ["\\n"],
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2
        }
        
        request_data = plugin._prepare_deepseek_request("deepseek-chat", messages, **kwargs)
        
        assert request_data["temperature"] == 0.7
        assert request_data["top_p"] == 0.9
        assert request_data["max_tokens"] == 1000
        assert request_data["stop"] == ["\\n"]
        assert request_data["frequency_penalty"] == 0.1
        assert request_data["presence_penalty"] == 0.2
    
    def test_prepare_request_with_tools(self):
        """测试包含工具的请求准备"""
        plugin = DeepSeekPlugin()
        messages = [ChatMessage(role="user", content="Hello")]
        
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        tool_choice = "auto"
        
        request_data = plugin._prepare_deepseek_request(
            "deepseek-chat", messages, tools=tools, tool_choice=tool_choice
        )
        
        assert request_data["tools"] == tools
        assert request_data["tool_choice"] == tool_choice
    
    def test_prepare_request_with_stream(self):
        """测试流式请求准备"""
        plugin = DeepSeekPlugin()
        messages = [ChatMessage(role="user", content="Hello")]
        
        request_data = plugin._prepare_deepseek_request(
            "deepseek-chat", messages, stream=True
        )
        
        assert request_data["stream"] is True
    
    def test_prepare_request_with_json_schema_response_format(self):
        """测试JSON Schema响应格式请求准备"""
        plugin = DeepSeekPlugin()
        messages = [ChatMessage(role="user", content="Generate data")]
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        }
        
        request_data = plugin._prepare_deepseek_request(
            "deepseek-chat", messages, response_format=response_format
        )
        
        assert request_data["response_format"]["type"] == "json_object"
        # 验证JSON关键词被添加到最后一条用户消息中
        last_user_msg = request_data["messages"][-1]
        assert "json" in last_user_msg["content"].lower()
    
    def test_prepare_request_with_json_object_response_format(self):
        """测试JSON对象响应格式请求准备"""
        plugin = DeepSeekPlugin()
        messages = [ChatMessage(role="user", content="Generate data")]
        
        response_format = {"type": "json_object"}
        
        request_data = plugin._prepare_deepseek_request(
            "deepseek-chat", messages, response_format=response_format
        )
        
        assert request_data["response_format"]["type"] == "json_object"
        # 验证JSON关键词被添加到最后一条用户消息中
        last_user_msg = request_data["messages"][-1]
        assert "json" in last_user_msg["content"].lower()
    
    def test_ensure_json_keyword_in_prompt(self):
        """测试确保prompt中包含JSON关键词"""
        plugin = DeepSeekPlugin()
        
        # 测试没有JSON关键词的情况
        messages = [{"role": "user", "content": "Generate some data"}]
        plugin._ensure_json_keyword_in_prompt(messages)
        
        assert "json" in messages[0]["content"].lower()
        assert "Return only raw JSON" in messages[0]["content"]
        
        # 测试已有JSON关键词的情况
        messages_with_json = [{"role": "user", "content": "Generate JSON data"}]
        original_content = messages_with_json[0]["content"]
        plugin._ensure_json_keyword_in_prompt(messages_with_json)
        
        # 内容不应该被修改
        assert messages_with_json[0]["content"] == original_content


class TestDeepSeekPluginResponseConversion:
    """测试DeepSeek插件响应转换功能"""
    
    def test_convert_to_harbor_response(self):
        """测试转换为Harbor响应格式"""
        plugin = DeepSeekPlugin()
        
        deepseek_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        
        harbor_response = plugin._convert_to_harbor_response(deepseek_response, "deepseek-chat")
        
        assert harbor_response.id == "chatcmpl-123"
        assert harbor_response.object == "chat.completion"
        assert harbor_response.created == 1677652288
        assert harbor_response.model == "deepseek-chat"
        assert len(harbor_response.choices) == 1
        
        choice = harbor_response.choices[0]
        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content == "Hello! How can I help you today?"
        assert choice.finish_reason == "stop"
        
        assert harbor_response.usage.prompt_tokens == 10
        assert harbor_response.usage.completion_tokens == 20
        assert harbor_response.usage.total_tokens == 30
    
    def test_convert_to_harbor_response_with_reasoning(self):
        """测试转换包含推理内容的响应"""
        plugin = DeepSeekPlugin()
        
        deepseek_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 42.",
                    "reasoning_content": "Let me think about this step by step..."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        
        harbor_response = plugin._convert_to_harbor_response(deepseek_response, "deepseek-reasoner")
        
        choice = harbor_response.choices[0]
        assert choice.message.reasoning_content == "Let me think about this step by step..."
    
    def test_convert_to_harbor_chunk(self):
        """测试转换为Harbor流式响应格式"""
        plugin = DeepSeekPlugin()
        
        deepseek_chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello"
                },
                "finish_reason": None
            }]
        }
        
        harbor_chunk = plugin._convert_to_harbor_chunk(deepseek_chunk, "deepseek-chat")
        
        assert harbor_chunk.id == "chatcmpl-123"
        assert harbor_chunk.object == "chat.completion.chunk"
        assert harbor_chunk.created == 1677652288
        assert harbor_chunk.model == "deepseek-chat"
        assert len(harbor_chunk.choices) == 1
        
        choice = harbor_chunk.choices[0]
        assert choice.index == 0
        assert choice.delta.role == "assistant"
        assert choice.delta.content == "Hello"
        assert choice.finish_reason is None


class TestDeepSeekPluginThinkingContent:
    """测试DeepSeek插件思考内容提取功能"""
    
    def test_extract_thinking_content_from_reasoning_field(self):
        """测试从reasoning字段提取思考内容"""
        plugin = DeepSeekPlugin()
        
        response = {
            "reasoning": "This is my thinking process...",
            "choices": []
        }
        
        thinking = plugin._extract_thinking_content(response)
        assert thinking == "This is my thinking process..."
    
    def test_extract_thinking_content_from_thinking_field(self):
        """测试从thinking字段提取思考内容"""
        plugin = DeepSeekPlugin()
        
        response = {
            "thinking": "Let me analyze this...",
            "choices": []
        }
        
        thinking = plugin._extract_thinking_content(response)
        assert thinking == "Let me analyze this..."
    
    def test_extract_thinking_content_from_choices(self):
        """测试从choices中提取思考内容"""
        plugin = DeepSeekPlugin()
        
        response = {
            "choices": [{
                "message": {
                    "reasoning_content": "Step by step analysis..."
                }
            }]
        }
        
        thinking = plugin._extract_thinking_content(response)
        assert thinking == "Step by step analysis..."
    
    def test_extract_thinking_content_not_found(self):
        """测试未找到思考内容的情况"""
        plugin = DeepSeekPlugin()
        
        response = {
            "choices": [{
                "message": {
                    "content": "Just a regular response"
                }
            }]
        }
        
        thinking = plugin._extract_thinking_content(response)
        assert thinking is None
    
    def test_extract_thinking_content_invalid_response(self):
        """测试无效响应格式"""
        plugin = DeepSeekPlugin()
        
        # 非字典类型
        thinking = plugin._extract_thinking_content("invalid response")
        assert thinking is None
        
        # None类型
        thinking = plugin._extract_thinking_content(None)
        assert thinking is None


class TestDeepSeekPluginChatCompletion:
    """测试DeepSeek插件聊天完成功能"""
    
    @patch('httpx.Client')
    def test_chat_completion_success(self, mock_httpx_client):
        """测试成功的聊天完成"""
        # 设置mock响应
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response = plugin.chat_completion("deepseek-chat", messages)
        
        assert isinstance(response, ChatCompletion)
        assert response.id == "chatcmpl-123"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello! How can I help you?"
        
        # 验证请求被正确发送
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/v1/chat/completions"
        
        request_data = call_args[1]["json"]
        assert request_data["model"] == "deepseek-chat"
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["content"] == "Hello"
    
    @patch('httpx.Client')
    def test_chat_completion_api_error(self, mock_httpx_client):
        """测试API错误处理"""
        # 设置mock错误响应
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status.side_effect = Exception("API Error")
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 应该抛出PluginError异常
        with pytest.raises(PluginError, match="DeepSeek API 请求失败"):
            plugin.chat_completion("deepseek-chat", messages)
    
    @patch('httpx.Client')
    def test_chat_completion_with_stream(self, mock_httpx_client):
        """测试流式聊天完成"""
        # 设置mock流式响应
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}',
            b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"choices":[{"index":0,"delta":{"content":" there!"},"finish_reason":null}]}',
            b'data: [DONE]'
        ]
        mock_response.raise_for_status = Mock()
        
        # 设置上下文管理器
        mock_stream_context = Mock()
        mock_stream_context.__enter__ = Mock(return_value=mock_response)
        mock_stream_context.__exit__ = Mock(return_value=None)
        
        mock_client = Mock()
        mock_client.stream.return_value = mock_stream_context
        mock_httpx_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_generator = plugin.chat_completion("deepseek-chat", messages, stream=True)
        
        # 收集所有流式响应
        chunks = list(response_generator)
        
        assert len(chunks) == 2  # 不包括[DONE]
        assert isinstance(chunks[0], ChatCompletionChunk)
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " there!"


class TestDeepSeekPluginAsyncChatCompletion:
    """测试DeepSeek插件异步聊天完成功能"""
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_success(self):
        """测试成功的异步聊天完成"""
        with patch('httpx.AsyncClient') as mock_async_client_class:
            # 设置mock响应数据
            mock_response_data = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you today?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21
                }
            }
            
            # 创建mock响应对象
            mock_response = Mock()
            mock_response.json = Mock(return_value=mock_response_data)
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 200
            
            # 创建mock客户端
            mock_client = Mock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_async_client_class.return_value = mock_client
            
            plugin = DeepSeekPlugin(api_key="test-key")
            messages = [ChatMessage(role="user", content="Hello")]
            
            response = await plugin.chat_completion_async("deepseek-chat", messages)
            
            assert isinstance(response, ChatCompletion)
            assert response.choices[0].message.content == "Hello! How can I help you today?"
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_stream(self):
        """测试异步流式聊天完成"""
        with patch('httpx.AsyncClient') as mock_async_client_class:
            # 设置mock异步流式响应
            async def mock_aiter_lines():
                lines = [
                    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"choices":[{"index":0,"delta":{"role":"assistant","content":"Async"},"finish_reason":null}]}',
                    b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"choices":[{"index":0,"delta":{"content":" stream!"},"finish_reason":null}]}',
                    b'data: [DONE]'
                ]
                for line in lines:
                    yield line
            
            mock_response = Mock()
            mock_response.aiter_lines = mock_aiter_lines
            mock_response.raise_for_status = Mock()
            mock_response.status_code = 200
            
            # 设置异步上下文管理器
            mock_stream_context = Mock()
            mock_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
            mock_stream_context.__aexit__ = AsyncMock(return_value=None)
            
            mock_client = Mock()
            mock_client.stream = Mock(return_value=mock_stream_context)
            mock_async_client_class.return_value = mock_client
            
            plugin = DeepSeekPlugin(api_key="test-key")
            messages = [ChatMessage(role="user", content="Hello")]
            
            response_generator = await plugin.chat_completion_async("deepseek-chat", messages, stream=True)
            
            # 收集所有异步流式响应
            chunks = []
            async for chunk in response_generator:
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert isinstance(chunks[0], ChatCompletionChunk)
            assert chunks[0].choices[0].delta.content == "Async"
            assert chunks[1].choices[0].delta.content == " stream!"


class TestDeepSeekPluginNativeStructuredOutput:
    """测试DeepSeek插件原生结构化输出功能"""
    
    @patch('httpx.Client')
    def test_native_structured_output_success(self, mock_httpx_client):
        """测试成功的原生结构化输出"""
        # 设置mock响应，返回有效JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"name": "John", "age": 30}'
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Generate user data")]
        
        response_format = {"type": "json_schema", "json_schema": {"name": "user"}}
        response = plugin._handle_native_structured_output(
            "deepseek-chat", messages, stream=False, response_format=response_format
        )
        
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == '{"name": "John", "age": 30}'
        assert response.choices[0].message.parsed == {"name": "John", "age": 30}
    
    @patch('httpx.Client')
    def test_native_structured_output_invalid_json(self, mock_httpx_client):
        """测试原生结构化输出返回无效JSON的处理"""
        # 设置mock响应，返回无效JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": 'Invalid JSON content'
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        }
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_httpx_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Generate user data")]
        
        response_format = {"type": "json_schema", "json_schema": {"name": "user"}}
        response = plugin._handle_native_structured_output(
            "deepseek-chat", messages, stream=False, response_format=response_format
        )
        
        # 应该返回原始响应而不是抛出错误
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == 'Invalid JSON content'
        assert response.choices[0].message.parsed is None


class TestDeepSeekPluginResourceManagement:
    """测试DeepSeek插件资源管理功能"""
    
    @patch('httpx.Client')
    def test_close_sync_client(self, mock_httpx_client):
        """测试关闭同步客户端"""
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        
        # 创建客户端
        plugin._get_client()
        assert plugin._client is not None
        
        # 关闭客户端
        plugin.close()
        mock_client.close.assert_called_once()
        assert plugin._client is None
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_aclose_async_client(self, mock_httpx_async_client):
        """测试关闭异步客户端"""
        mock_client = AsyncMock()
        mock_httpx_async_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        
        # 创建异步客户端
        plugin._get_async_client()
        assert plugin._async_client is not None
        
        # 关闭异步客户端
        await plugin.aclose()
        mock_client.aclose.assert_called_once()
        assert plugin._async_client is None
    
    @patch('httpx.Client')
    def test_destructor_cleanup(self, mock_httpx_client):
        """测试析构函数资源清理"""
        mock_client = Mock()
        mock_httpx_client.return_value = mock_client
        
        plugin = DeepSeekPlugin(api_key="test-key")
        plugin._get_client()  # 创建客户端
        
        # 模拟析构函数调用
        plugin.__del__()
        
        # 验证close被调用
        mock_client.close.assert_called_once()


class TestDeepSeekPluginEdgeCases:
    """测试DeepSeek插件边界情况"""
    
    def test_empty_api_key_handling(self):
        """测试空API密钥处理"""
        plugin = DeepSeekPlugin(api_key="")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError, match="DeepSeek API key is required"):
            plugin._validate_request("deepseek-chat", messages)
    
    def test_none_api_key_handling(self):
        """测试None API密钥处理"""
        plugin = DeepSeekPlugin(api_key=None)
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError, match="DeepSeek API key is required"):
            plugin._validate_request("deepseek-chat", messages)
    
    def test_message_with_all_fields(self):
        """测试包含所有字段的消息处理"""
        plugin = DeepSeekPlugin()
        
        messages = [ChatMessage(
            role="user",
            content="Hello",
            name="test_user",
            tool_calls=[{"id": "call_123", "type": "function"}],
            tool_call_id="call_123"
        )]
        
        request_data = plugin._prepare_deepseek_request("deepseek-chat", messages)
        
        message = request_data["messages"][0]
        assert message["role"] == "user"
        assert message["content"] == "Hello"
        assert message["name"] == "test_user"
        assert message["tool_calls"] == [{"id": "call_123", "type": "function"}]
        assert message["tool_call_id"] == "call_123"
    
    def test_multiple_user_messages_json_keyword(self):
        """测试多条用户消息时JSON关键词添加"""
        plugin = DeepSeekPlugin()
        
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second message"}
        ]
        
        plugin._ensure_json_keyword_in_prompt(messages)
        
        # 只有最后一条用户消息应该被修改
        assert "json" not in messages[0]["content"].lower()
        assert messages[1]["content"] == "Response"  # 助手消息不变
        assert "json" in messages[2]["content"].lower()
    
    def test_response_with_missing_fields(self):
        """测试缺少字段的响应处理"""
        plugin = DeepSeekPlugin()
        
        # 最小响应格式
        minimal_response = {
            "choices": [{
                "message": {
                    "content": "Hello"
                }
            }]
        }
        
        harbor_response = plugin._convert_to_harbor_response(minimal_response, "deepseek-chat")
        
        assert harbor_response.id == ""
        assert harbor_response.object == "chat.completion"
        assert harbor_response.created == 0
        assert harbor_response.model == "deepseek-chat"
        assert len(harbor_response.choices) == 1
        assert harbor_response.choices[0].message.content == "Hello"
        assert harbor_response.usage.total_tokens == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])