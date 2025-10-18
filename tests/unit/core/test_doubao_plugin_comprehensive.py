"""
豆包插件comprehensive测试
---
summary: 全面测试豆包插件的所有功能，包括初始化、聊天完成、流式响应、结构化输出等
coverage_target: 90%+
test_categories:
  - 插件初始化和配置
  - 模型支持检查
  - 同步聊天完成
  - 异步聊天完成
  - 流式响应处理
  - 结构化输出
  - 错误处理
  - 客户端管理
---
"""

import pytest
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# 确保导入doubao_plugin模块以便覆盖率统计
import harborai.core.plugins.doubao_plugin
from harborai.core.plugins.doubao_plugin import DoubaoPlugin
from harborai.core.base_plugin import ChatMessage, ModelInfo
from harborai.utils.exceptions import PluginError, ValidationError


class TestDoubaoPluginInitialization:
    """测试豆包插件初始化和配置"""
    
    def test_basic_initialization(self):
        """测试基本初始化"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        assert plugin.name == "doubao"
        assert plugin.api_key == "test_key"
        assert plugin.base_url == "https://ark.cn-beijing.volces.com/api/v3"
        assert plugin.timeout == 60
        assert plugin.max_retries == 3
    
    def test_initialization_with_custom_config(self):
        """测试自定义配置初始化"""
        config = {
            "api_key": "custom_key",
            "base_url": "https://custom.api.com",
            "timeout": 30,
            "max_retries": 5
        }
        
        plugin = DoubaoPlugin(name="doubao", **config)
        
        assert plugin.api_key == "custom_key"
        assert plugin.base_url == "https://custom.api.com"
        assert plugin.timeout == 30
        assert plugin.max_retries == 5
    
    def test_initialization_without_api_key(self):
        """测试缺少API密钥的初始化"""
        with pytest.raises(PluginError) as exc_info:
            DoubaoPlugin(name="doubao")
        
        assert "API key is required" in str(exc_info.value)
    
    def test_supported_models(self):
        """测试支持的模型列表"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        models = plugin.supported_models
        
        assert isinstance(models, list)
        assert len(models) >= 2  # 至少有两个模型
        
        # 检查模型信息结构
        for model in models:
            assert isinstance(model, ModelInfo)
            assert model.provider == "doubao"
            assert model.id is not None
            assert model.name is not None
            assert isinstance(model.max_tokens, int)
            assert isinstance(model.supports_streaming, bool)
            assert isinstance(model.supports_structured_output, bool)
    
    def test_supports_model(self):
        """测试模型支持检查"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 测试支持的模型
        assert plugin.supports_model("doubao-1-5-pro-32k-character-250715") is True
        assert plugin.supports_model("doubao-seed-1-6-250615") is True
        
        # 测试不支持的模型
        assert plugin.supports_model("nonexistent-model") is False
        assert plugin.supports_model("gpt-4") is False
    
    def test_is_thinking_model(self):
        """测试推理模型判断"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 测试推理模型
        assert plugin.is_thinking_model("doubao-seed-1-6-250615") is True
        
        # 测试非推理模型
        assert plugin.is_thinking_model("doubao-1-5-pro-32k-character-250715") is False
        assert plugin.is_thinking_model("nonexistent-model") is False


class TestDoubaoPluginValidation:
    """测试豆包插件请求验证"""
    
    def test_validate_request_success(self):
        """测试有效请求验证"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 应该不抛出异常
        plugin._validate_request("doubao-1-5-pro-32k-character-250715", messages)
    
    def test_validate_request_unsupported_model(self):
        """测试不支持的模型验证"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError) as exc_info:
            plugin._validate_request("unsupported-model", messages)
        
        assert "not supported" in str(exc_info.value)
    
    def test_validate_request_empty_messages(self):
        """测试空消息验证"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        with pytest.raises(ValidationError) as exc_info:
            plugin._validate_request("doubao-1-5-pro-32k-character-250715", [])
        
        assert "cannot be empty" in str(exc_info.value)
    
    def test_validate_request_invalid_temperature(self):
        """测试无效温度参数验证"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError) as exc_info:
            plugin._validate_request("doubao-1-5-pro-32k-character-250715", messages, temperature=3.0)
        
        assert "Temperature must be between 0 and 2" in str(exc_info.value)
    
    def test_validate_request_invalid_max_tokens(self):
        """测试无效max_tokens参数验证"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError) as exc_info:
            plugin._validate_request("doubao-1-5-pro-32k-character-250715", messages, max_tokens=-1)
        
        assert "max_tokens must be positive" in str(exc_info.value)


class TestDoubaoPluginRequestPreparation:
    """测试豆包插件请求准备"""
    
    def test_prepare_basic_request(self):
        """测试基本请求准备"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        request_data = plugin._prepare_doubao_request("doubao-1-5-pro-32k-character-250715", messages)
        
        assert request_data["model"] == "doubao-1-5-pro-32k-character-250715"
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert request_data["messages"][0]["content"] == "Hello"
    
    def test_prepare_request_with_optional_params(self):
        """测试包含可选参数的请求准备"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        kwargs = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "stream": True
        }
        
        request_data = plugin._prepare_doubao_request("doubao-1-5-pro-32k-character-250715", messages, **kwargs)
        
        assert request_data["temperature"] == 0.7
        assert request_data["max_tokens"] == 100
        assert request_data["top_p"] == 0.9
        assert request_data["stream"] is True
    
    def test_prepare_request_with_structured_output(self):
        """测试结构化输出请求准备"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
            }
        }
        
        kwargs = {
            "response_format": response_format,
            "structured_provider": "native"
        }
        
        request_data = plugin._prepare_doubao_request("doubao-1-5-pro-32k-character-250715", messages, **kwargs)
        
        assert "response_format" in request_data
        assert request_data["response_format"] == response_format
    
    def test_prepare_request_with_complex_messages(self):
        """测试复杂消息格式的请求准备"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [
            ChatMessage(role="user", content="Hello", name="user1"),
            ChatMessage(role="assistant", content="Hi", tool_calls=[{"id": "call_1", "type": "function"}])
        ]
        
        request_data = plugin._prepare_doubao_request("doubao-1-5-pro-32k-character-250715", messages)
        
        assert len(request_data["messages"]) == 2
        assert request_data["messages"][0]["name"] == "user1"
        assert request_data["messages"][1]["tool_calls"] == [{"id": "call_1", "type": "function"}]


class TestDoubaoPluginResponseConversion:
    """测试豆包插件响应转换"""
    
    def test_convert_basic_response(self):
        """测试基本响应转换"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello from Doubao!"
                },
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            },
            "id": "test_id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "doubao-1-5-pro-32k-character-250715"
        }
        
        harbor_response = plugin._convert_to_harbor_response(response_data, "doubao-1-5-pro-32k-character-250715")
        
        assert harbor_response.id == "test_id"
        assert harbor_response.object == "chat.completion"
        assert harbor_response.created == 1234567890
        assert harbor_response.model == "doubao-1-5-pro-32k-character-250715"
        assert len(harbor_response.choices) == 1
        assert harbor_response.choices[0].message.content == "Hello from Doubao!"
        assert harbor_response.usage.total_tokens == 15
    
    def test_convert_response_with_thinking_content(self):
        """测试包含思考内容的响应转换"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning_content": "Let me think about this..."
                },
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "reasoning": "This is the thinking process"
        }
        
        harbor_response = plugin._convert_to_harbor_response(response_data, "doubao-seed-1-6-250615")
        
        assert harbor_response.choices[0].message.reasoning_content == "This is the thinking process"
    
    def test_convert_chunk_response(self):
        """测试流式响应块转换"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        chunk_data = {
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": "Hello"
                },
                "index": 0,
                "finish_reason": None
            }],
            "id": "chunk_id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "doubao-1-5-pro-32k-character-250715"
        }
        
        harbor_chunk = plugin._convert_to_harbor_chunk(chunk_data, "doubao-1-5-pro-32k-character-250715")
        
        assert harbor_chunk.id == "chunk_id"
        assert harbor_chunk.object == "chat.completion.chunk"
        assert harbor_chunk.choices[0].delta.content == "Hello"


class TestDoubaoPluginChatCompletion:
    """测试豆包插件聊天完成功能"""
    
    @patch('httpx.Client')
    def test_chat_completion_success(self, mock_client_class):
        """测试成功的聊天完成"""
        # 设置mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "id": "test_id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "doubao-1-5-pro-32k-character-250715"
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response = plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages)
        
        assert response.choices[0].message.content == "Hello!"
        assert response.usage.total_tokens == 15
        mock_client.post.assert_called_once()
    
    @patch('httpx.Client')
    def test_chat_completion_with_structured_output(self, mock_client_class):
        """测试结构化输出的聊天完成"""
        # 设置mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": '{"answer": "test"}'},
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "id": "test_id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "doubao-1-5-pro-32k-character-250715"
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"]
                }
            }
        }
        
        response = plugin.chat_completion(
            "doubao-1-5-pro-32k-character-250715", 
            messages, 
            response_format=response_format,
            structured_provider="native"
        )
        
        assert response.parsed == {"answer": "test"}
        assert response.choices[0].message.parsed == {"answer": "test"}
    
    @patch('httpx.Client')
    def test_chat_completion_error_handling(self, mock_client_class):
        """测试聊天完成错误处理"""
        # 设置mock抛出异常
        mock_client = Mock()
        mock_client.post.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response = plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages)
        
        # 应该返回错误响应而不是抛出异常
        assert response is not None
        assert "error" in response.choices[0].message.content.lower()


class TestDoubaoPluginAsyncChatCompletion:
    """测试豆包插件异步聊天完成功能"""
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_chat_completion_success(self, mock_client_class):
        """测试成功的异步聊天完成"""
        # 设置mock
        mock_client = AsyncMock()
        mock_response = Mock()  # 使用普通Mock而不是AsyncMock
        mock_response.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello async!"},
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "id": "test_id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "doubao-1-5-pro-32k-character-250715"
        }
        mock_response.raise_for_status.return_value = None
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response = await plugin.chat_completion_async("doubao-1-5-pro-32k-character-250715", messages)
        
        assert response.choices[0].message.content == "Hello async!"
        assert response.usage.total_tokens == 15
        mock_client.post.assert_called_once()


class TestDoubaoPluginStreamingResponse:
    """测试豆包插件流式响应处理"""
    
    @patch('httpx.Client')
    def test_streaming_response_handling(self, mock_client_class):
        """测试流式响应处理"""
        # 创建mock流式响应
        mock_client = Mock()
        mock_response = Mock()
        
        # 模拟流式数据
        stream_data = [
            b'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"},"index":0}],"id":"chunk1","object":"chat.completion.chunk","created":1234567890,"model":"doubao-1-5-pro-32k-character-250715"}\n\n',
            b'data: {"choices":[{"delta":{"content":" world"},"index":0}],"id":"chunk2","object":"chat.completion.chunk","created":1234567890,"model":"doubao-1-5-pro-32k-character-250715"}\n\n',
            b'data: [DONE]\n\n'
        ]
        
        mock_response.iter_lines.return_value = stream_data
        mock_response.raise_for_status.return_value = None
        
        # 设置上下文管理器支持
        mock_stream_context = Mock()
        mock_stream_context.__enter__ = Mock(return_value=mock_response)
        mock_stream_context.__exit__ = Mock(return_value=None)
        mock_client.stream.return_value = mock_stream_context
        
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_generator = plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages, stream=True)
        chunks = list(response_generator)
        
        assert len(chunks) == 2  # 不包括[DONE]
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world"
    
    @patch('httpx.Client')
    def test_streaming_response_with_string_lines(self, mock_client_class):
        """测试字符串格式的流式响应处理"""
        # 创建mock流式响应
        mock_client = Mock()
        mock_response = Mock()
        
        # 模拟字符串格式的流式数据
        stream_data = [
            'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"},"index":0}],"id":"chunk1","object":"chat.completion.chunk","created":1234567890,"model":"doubao-1-5-pro-32k-character-250715"}',
            'data: {"choices":[{"delta":{"content":" world"},"index":0}],"id":"chunk2","object":"chat.completion.chunk","created":1234567890,"model":"doubao-1-5-pro-32k-character-250715"}',
            'data: [DONE]'
        ]
        
        mock_response.iter_lines.return_value = stream_data
        mock_response.raise_for_status.return_value = None
        
        # 设置上下文管理器支持
        mock_stream_context = Mock()
        mock_stream_context.__enter__ = Mock(return_value=mock_response)
        mock_stream_context.__exit__ = Mock(return_value=None)
        mock_client.stream.return_value = mock_stream_context
        
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_generator = plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages, stream=True)
        chunks = list(response_generator)
        
        assert len(chunks) == 2  # 不包括[DONE]
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world"
    
    @patch('httpx.Client')
    def test_streaming_response_with_invalid_json(self, mock_client_class):
        """测试包含无效JSON的流式响应处理"""
        # 创建mock流式响应
        mock_client = Mock()
        mock_response = Mock()
        
        # 模拟包含无效JSON的流式数据
        stream_data = [
            b'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"},"index":0}],"id":"chunk1","object":"chat.completion.chunk","created":1234567890,"model":"doubao-1-5-pro-32k-character-250715"}\n\n',
            b'data: invalid json\n\n',  # 无效JSON，应该被跳过
            b'data: {"choices":[{"delta":{"content":" world"},"index":0}],"id":"chunk2","object":"chat.completion.chunk","created":1234567890,"model":"doubao-1-5-pro-32k-character-250715"}\n\n',
            b'data: [DONE]\n\n'
        ]
        
        mock_response.iter_lines.return_value = stream_data
        mock_response.raise_for_status.return_value = None
        
        # 设置上下文管理器支持
        mock_stream_context = Mock()
        mock_stream_context.__enter__ = Mock(return_value=mock_response)
        mock_stream_context.__exit__ = Mock(return_value=None)
        mock_client.stream.return_value = mock_stream_context
        
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_generator = plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages, stream=True)
        chunks = list(response_generator)
        
        assert len(chunks) == 2  # 无效JSON被跳过
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world"
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_streaming_response_handling(self, mock_client_class):
        """测试异步流式响应处理"""
        # 创建mock异步流式响应
        mock_client = AsyncMock()
        mock_response = AsyncMock()  # 使用AsyncMock而不是普通Mock
        
        # 模拟异步流式数据
        async def mock_aiter_lines():
            stream_data = [
                b'data: {"choices":[{"delta":{"role":"assistant","content":"Hello"},"index":0}],"id":"chunk1","object":"chat.completion.chunk","created":1234567890,"model":"doubao-1-5-pro-32k-character-250715"}\n\n',
                b'data: {"choices":[{"delta":{"content":" async"},"index":0}],"id":"chunk2","object":"chat.completion.chunk","created":1234567890,"model":"doubao-1-5-pro-32k-character-250715"}\n\n',
                b'data: [DONE]\n\n'
            ]
            for line in stream_data:
                yield line
        
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = AsyncMock()
        
        # 设置异步上下文管理器支持
        mock_async_stream_context = AsyncMock()
        mock_async_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_async_stream_context.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = Mock(return_value=mock_async_stream_context)
        
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_generator = await plugin.chat_completion_async("doubao-1-5-pro-32k-character-250715", messages, stream=True)
        chunks = []
        async for chunk in response_generator:
            chunks.append(chunk)
        
        assert len(chunks) == 2  # 不包括[DONE]
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " async"
    
    @patch('httpx.Client')
    def test_streaming_response_with_thinking_model(self, mock_client_class):
        """测试推理模型的流式响应处理"""
        # 创建mock流式响应
        mock_client = Mock()
        mock_response = Mock()
        
        # 模拟包含推理内容的流式数据
        stream_data = [
            b'data: {"choices":[{"delta":{"role":"assistant","reasoning_content":"Let me think..."},"index":0}],"id":"chunk1","object":"chat.completion.chunk","created":1234567890,"model":"doubao-seed-1-6-250615"}\n\n',
            b'data: {"choices":[{"delta":{"content":"Hello"},"index":0}],"id":"chunk2","object":"chat.completion.chunk","created":1234567890,"model":"doubao-seed-1-6-250615"}\n\n',
            b'data: [DONE]\n\n'
        ]
        
        mock_response.iter_lines.return_value = stream_data
        mock_response.raise_for_status.return_value = None
        
        # 设置上下文管理器支持
        mock_stream_context = Mock()
        mock_stream_context.__enter__ = Mock(return_value=mock_response)
        mock_stream_context.__exit__ = Mock(return_value=None)
        mock_client.stream.return_value = mock_stream_context
        
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_generator = plugin.chat_completion("doubao-seed-1-6-250615", messages, stream=True)
        chunks = list(response_generator)
        
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.reasoning_content == "Let me think..."
        assert chunks[1].choices[0].delta.content == "Hello"


class TestDoubaoPluginErrorHandling:
    """测试豆包插件错误处理"""
    
    @patch('httpx.Client')
    def test_http_error_handling(self, mock_client_class):
        """测试HTTP错误处理"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "HTTP Error", request=Mock(), response=Mock()
        )
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(httpx.HTTPStatusError):
            list(plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages))
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_http_error_handling(self, mock_client_class):
        """测试异步HTTP错误处理"""
        mock_client = AsyncMock()
        mock_response = Mock()  # 使用普通Mock而不是AsyncMock
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "HTTP Error", request=Mock(), response=Mock()
        )
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(httpx.HTTPStatusError):
            await plugin.chat_completion_async("doubao-1-5-pro-32k-character-250715", messages)
    
    @patch('httpx.Client')
    def test_connection_error_handling(self, mock_client_class):
        """测试连接错误处理"""
        mock_client = Mock()
        mock_client.post.side_effect = httpx.ConnectError("Connection failed")
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(httpx.ConnectError):
            list(plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages))
    
    @patch('httpx.Client')
    def test_timeout_error_handling(self, mock_client_class):
        """测试超时错误处理"""
        mock_client = Mock()
        mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(httpx.TimeoutException):
            list(plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages))
    
    @patch('httpx.Client')
    def test_invalid_response_format(self, mock_client_class):
        """测试无效响应格式处理"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"invalid": "response"}  # 缺少必要字段
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 应该能处理无效响应格式而不崩溃
        response = plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages)
        result = list(response)
        assert len(result) == 1  # 应该返回一个响应


class TestDoubaoPluginValidation:
    """测试豆包插件参数验证"""
    
    def test_empty_messages_validation(self):
        """测试空消息列表验证"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        with pytest.raises(ValidationError, match="Messages cannot be empty"):
            plugin.chat_completion("doubao-1-5-pro-32k-character-250715", [])
    
    def test_missing_api_key_validation(self):
        """测试缺少API密钥验证"""
        # API密钥验证在初始化时就会抛出错误
        with pytest.raises(PluginError, match="API key is required"):
            DoubaoPlugin(name="doubao", api_key="")
    
    def test_temperature_range_validation(self):
        """测试温度参数范围验证"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 测试温度过高
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages, temperature=3.0)
        
        # 测试温度过低
        with pytest.raises(ValidationError, match="Temperature must be between 0 and 2"):
            plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages, temperature=-1.0)
    
    def test_max_tokens_validation(self):
        """测试最大令牌数验证"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError, match="max_tokens must be positive"):
             plugin.chat_completion("doubao-1-5-pro-32k-character-250715", messages, max_tokens=-1)


class TestDoubaoPluginThinkingContent:
    """测试豆包插件思考内容提取"""
    
    @patch('httpx.Client')
    def test_extract_thinking_content_from_reasoning(self, mock_client_class):
        """测试从reasoning字段提取思考内容"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "id": "test_id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "doubao-seed-1-6-250615",
            "reasoning": "Let me think about this...",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world"
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response = plugin.chat_completion("doubao-seed-1-6-250615", messages)
        
        assert response.choices[0].message.reasoning_content == "Let me think about this..."
    
    @patch('httpx.Client')
    def test_extract_thinking_content_from_message(self, mock_client_class):
        """测试从消息中的reasoning_content字段提取思考内容"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "id": "test_id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "doubao-seed-1-6-250615",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello world",
                    "reasoning_content": "I need to respond politely..."
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response = plugin.chat_completion("doubao-seed-1-6-250615", messages)
        
        assert response.choices[0].message.reasoning_content == "I need to respond politely..."


class TestDoubaoPluginClientManagement:
    """测试豆包插件客户端管理"""
    
    @patch('httpx.Client')
    def test_get_client(self, mock_client_class):
        """测试获取同步客户端"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        client1 = plugin._get_client()
        client2 = plugin._get_client()
        
        # 应该返回同一个客户端实例
        assert client1 is client2
        mock_client_class.assert_called_once()
    
    def test_client_with_custom_base_url(self):
        """测试自定义base_url的客户端"""
        custom_url = "https://custom.api.com"
        plugin = DoubaoPlugin(name="doubao", api_key="test_key", base_url=custom_url)
        assert plugin.base_url == custom_url
    
    def test_client_close(self):
        """测试客户端关闭"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 模拟客户端存在
        mock_client = Mock()
        plugin._client = mock_client
        
        plugin.close()
        
        mock_client.close.assert_called_once()
        assert plugin._client is None
    
    @pytest.mark.asyncio
    async def test_async_client_close(self):
        """测试异步客户端关闭"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 模拟异步客户端存在
        mock_async_client = AsyncMock()
        plugin._async_client = mock_async_client
        
        await plugin.aclose()
        
        mock_async_client.aclose.assert_called_once()
        assert plugin._async_client is None
    
    def test_destructor(self):
        """测试析构函数"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 模拟客户端存在
        mock_client = Mock()
        plugin._client = mock_client
        
        # 调用析构函数
        plugin.__del__()
        
        mock_client.close.assert_called_once()
    
    @patch('httpx.AsyncClient')
    def test_get_async_client(self, mock_client_class):
        """测试获取异步客户端"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        client1 = plugin._get_async_client()
        client2 = plugin._get_async_client()
        
        # 应该返回同一个客户端实例
        assert client1 is client2
        mock_client_class.assert_called_once()
    
    def test_close_clients(self):
        """测试关闭客户端"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 设置mock客户端
        mock_client = Mock()
        plugin._client = mock_client
        
        plugin.close()
        
        mock_client.close.assert_called_once()
        assert plugin._client is None
    
    @pytest.mark.asyncio
    async def test_aclose_clients(self):
        """测试异步关闭客户端"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 设置mock异步客户端
        mock_async_client = AsyncMock()
        plugin._async_client = mock_async_client
        
        await plugin.aclose()
        
        mock_async_client.aclose.assert_called_once()
        assert plugin._async_client is None


class TestDoubaoPluginThinkingContent:
    """测试豆包插件思考内容提取"""
    
    def test_extract_thinking_content_from_reasoning(self):
        """测试从reasoning字段提取思考内容"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        response = {"reasoning": "This is my thinking process"}
        
        thinking_content = plugin._extract_thinking_content(response)
        
        assert thinking_content == "This is my thinking process"
    
    def test_extract_thinking_content_from_thinking(self):
        """测试从thinking字段提取思考内容"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        response = {"thinking": "This is my thinking process"}
        
        thinking_content = plugin._extract_thinking_content(response)
        
        assert thinking_content == "This is my thinking process"
    
    def test_extract_thinking_content_from_choices(self):
        """测试从choices中提取思考内容"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        response = {
            "choices": [{
                "message": {
                    "reasoning_content": "This is reasoning content"
                }
            }]
        }
        
        thinking_content = plugin._extract_thinking_content(response)
        
        assert thinking_content == "This is reasoning content"
    
    def test_extract_thinking_content_none(self):
        """测试无思考内容的情况"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        response = {"choices": [{"message": {"content": "Normal response"}}]}
        
        thinking_content = plugin._extract_thinking_content(response)
        
        assert thinking_content is None


class TestDoubaoPluginErrorHandling:
    """测试豆包插件错误处理"""
    
    def test_missing_httpx_dependency(self):
        """测试缺少httpx依赖的错误处理"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'httpx'")):
            with pytest.raises(PluginError) as exc_info:
                plugin._get_client()
            
            assert "httpx not installed" in str(exc_info.value)
    
    def test_missing_httpx_dependency_async(self):
        """测试缺少httpx依赖的异步错误处理"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'httpx'")):
            with pytest.raises(PluginError) as exc_info:
                plugin._get_async_client()
            
            assert "httpx not installed" in str(exc_info.value)
    
    def test_structured_output_invalid_json(self):
        """测试结构化输出无效JSON的错误处理"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        with patch.object(plugin, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"role": "assistant", "content": "invalid json {"},
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            messages = [ChatMessage(role="user", content="Hello")]
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": {"type": "object"}}
            }
            
            result = plugin._handle_native_structured_output(
                "doubao-1-5-pro-32k-character-250715", 
                messages, 
                response_format=response_format
            )
            
            # 检查返回的是错误响应
            assert result.choices[0].message.content is not None
            assert "不是有效JSON" in result.choices[0].message.content
    
    def test_structured_output_missing_required_fields(self):
        """测试结构化输出缺少必需字段的错误处理"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        with patch.object(plugin, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"role": "assistant", "content": '{"wrong_field": "value"}'},
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            messages = [ChatMessage(role="user", content="Hello")]
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "test", 
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"]
                    }
                }
            }
            
            result = plugin._handle_native_structured_output(
                "doubao-1-5-pro-32k-character-250715", 
                messages, 
                response_format=response_format
            )
            
            # 检查返回的是错误响应
            assert result.choices[0].message.content is not None
            assert "缺少必需字段" in result.choices[0].message.content
    
    def test_structured_output_no_content(self):
        """测试结构化输出无内容的错误处理"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        with patch.object(plugin, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"role": "assistant", "content": None},
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_get_client.return_value = mock_client
            
            messages = [ChatMessage(role="user", content="Hello")]
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": {"type": "object"}}
            }
            
            result = plugin._handle_native_structured_output(
                "doubao-1-5-pro-32k-character-250715", 
                messages, 
                response_format=response_format
            )
            
            # 检查返回的是错误响应
            assert result.choices[0].message.content is not None
            assert "未返回有效的响应内容" in result.choices[0].message.content
    
    @pytest.mark.asyncio
    async def test_async_structured_output_invalid_json(self):
        """测试异步结构化输出无效JSON的错误处理"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        with patch.object(plugin, '_get_async_client') as mock_get_async_client:
            mock_client = AsyncMock()
            mock_response = Mock()  # 使用普通Mock而不是AsyncMock
            mock_response.json.return_value = {
                "choices": [{
                    "message": {"role": "assistant", "content": "invalid json {"},
                    "index": 0,
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }
            mock_response.raise_for_status.return_value = None
            mock_client.post.return_value = mock_response
            mock_get_async_client.return_value = mock_client
            
            messages = [ChatMessage(role="user", content="Hello")]
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": {"type": "object"}}
            }
            
            # 现在期望返回错误响应而不是抛出异常
            response = await plugin._handle_native_structured_output_async(
                "doubao-1-5-pro-32k-character-250715", 
                messages, 
                response_format=response_format
            )
            
            # 验证返回的是错误响应
            assert response.choices[0].message.content.startswith("Error:")
            assert "不是有效JSON" in response.choices[0].message.content
    
    def test_general_exception_handling(self):
        """测试一般异常处理"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        with patch.object(plugin, '_get_client', side_effect=Exception("Network error")):
            messages = [ChatMessage(role="user", content="Hello")]
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": {"type": "object"}}
            }
            
            with pytest.raises(PluginError) as exc_info:
                plugin._handle_native_structured_output(
                    "doubao-1-5-pro-32k-character-250715", 
                    messages, 
                    response_format=response_format
                )
            
            assert "原生结构化输出失败" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_async_general_exception_handling(self):
        """测试异步一般异常处理"""
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        with patch.object(plugin, '_get_async_client', side_effect=Exception("Network error")):
            messages = [ChatMessage(role="user", content="Hello")]
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": {"type": "object"}}
            }
            
            # 现在期望返回错误响应而不是抛出异常
            result = await plugin._handle_native_structured_output_async(
                "doubao-1-5-pro-32k-character-250715", 
                messages, 
                response_format=response_format
            )
            
            # 检查返回的错误响应
            assert result.choices[0].message.content is not None
            assert "Error:" in result.choices[0].message.content
            assert "Network error" in result.choices[0].message.content


class TestDoubaoPluginEdgeCases:
    """测试豆包插件的边界情况和未覆盖路径"""
    
    def test_httpx_import_error_sync(self):
        """
        测试同步客户端httpx导入错误处理
        ---
        summary: 模拟httpx模块不可用时的错误处理
        target_line: 111
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        # 确保客户端为None，强制重新创建
        plugin._client = None
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 模拟httpx导入错误 - 直接在_get_client方法中patch
        def mock_get_client():
            raise PluginError("doubao", "httpx not installed. Please install it to use Doubao plugin.")
        
        with patch.object(plugin, '_get_client', side_effect=mock_get_client):
            result = plugin.chat_completion(messages=messages, model="doubao-1-5-pro-32k-character-250715")
            
            # 检查返回的错误响应
            assert result.choices[0].message.content is not None
            assert "httpx not installed" in result.choices[0].message.content
    
    @pytest.mark.asyncio
    async def test_httpx_import_error_async(self):
        """
        测试异步客户端httpx导入错误处理
        ---
        summary: 模拟httpx模块不可用时的异步错误处理
        target_line: 119
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        # 确保异步客户端为None，强制重新创建
        plugin._async_client = None
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 模拟httpx导入错误 - 直接在_get_async_client方法中patch
        def mock_get_async_client():
            raise PluginError("doubao", "httpx not installed. Please install it to use Doubao plugin.")
        
        with patch.object(plugin, '_get_async_client', side_effect=mock_get_async_client):
            result = await plugin.chat_completion_async(messages=messages, model="doubao-1-5-pro-32k-character-250715")
            
            # 检查返回的错误响应
            assert result.choices[0].message.content is not None
            assert "httpx not installed" in result.choices[0].message.content
    
    def test_validation_edge_cases(self):
        """
        测试参数验证的边界情况
        ---
        summary: 测试温度和最大token数的边界值验证
        target_lines: 132-146
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 测试温度边界值
        with pytest.raises(ValidationError):
            plugin._validate_request(messages, "doubao-1-5-pro-32k-character-250715", temperature=-0.1)
        
        with pytest.raises(ValidationError):
            plugin._validate_request(messages, "doubao-1-5-pro-32k-character-250715", temperature=2.1)
        
        # 测试最大token边界值
        with pytest.raises(ValidationError):
            plugin._validate_request(messages, "doubao-1-5-pro-32k-character-250715", max_tokens=0)
        
        with pytest.raises(ValidationError):
            plugin._validate_request(messages, "doubao-1-5-pro-32k-character-250715", max_tokens=32769)
    
    def test_extract_thinking_content_edge_cases(self):
        """
        测试思考内容提取的边界情况
        ---
        summary: 测试各种响应格式下的思考内容提取
        target_line: 162
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 测试空响应
        assert plugin._extract_thinking_content({}) is None
        
        # 测试只有reasoning字段
        response_with_reasoning = {
            "reasoning": "这是推理内容"
        }
        assert plugin._extract_thinking_content(response_with_reasoning) == "这是推理内容"
        
        # 测试choices中的reasoning_content字段
        response_with_reasoning_content = {
            "choices": [{
                "message": {
                    "reasoning_content": "这是推理内容"
                }
            }]
        }
        assert plugin._extract_thinking_content(response_with_reasoning_content) == "这是推理内容"
        
        # 测试无有效内容
        response_empty = {
            "choices": [{
                "message": {}
            }]
        }
        assert plugin._extract_thinking_content(response_empty) is None

    @patch('httpx.Client')
    def test_native_structured_output_success(self, mock_client_class):
        """
        测试原生结构化输出成功场景
        ---
        summary: 测试豆包API的原生结构化输出功能
        target_lines: 347, 448-473
        ---
        """
        # 设置mock响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"name": "张三", "age": 25}',
                    "role": "assistant"
                }
            }],
            "usage": {"total_tokens": 100}
        }
        mock_response.status_code = 200
        
        # 设置上下文管理器支持
        mock_stream_context = Mock()
        mock_stream_context.__enter__ = Mock(return_value=mock_response)
        mock_stream_context.__exit__ = Mock(return_value=None)
        
        mock_client = Mock()
        mock_client.stream.return_value = mock_stream_context
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="生成一个人员信息")]
        
        # 定义结构化输出schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"]
                }
            }
        }
        
        result = plugin.chat_completion(
            messages=messages,
            model="doubao-1-5-pro-32k-character-250715",
            response_format=response_format,
            structured_provider='native'
        )
        
        assert result.choices[0].message.content == '{"name": "张三", "age": 25}'
        assert result.parsed == {"name": "张三", "age": 25}

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_native_structured_output_success(self, mock_client_class):
        """
        测试异步原生结构化输出成功场景
        ---
        summary: 测试豆包API的异步原生结构化输出功能
        target_lines: 375-377, 480-482
        ---
        """
        # 设置mock响应
        mock_response = Mock()  # 使用普通Mock而不是AsyncMock
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"task": "完成", "status": "success"}',
                    "role": "assistant"
                }
            }],
            "usage": {"total_tokens": 80}
        }
        mock_response.status_code = 200
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="执行任务")]
        
        # 定义结构化输出schema
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "task_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "status": {"type": "string"}
                    },
                    "required": ["task", "status"]
                }
            }
        }
        
        result = await plugin.chat_completion_async(
            messages=messages,
            model="doubao-1-5-pro-32k-character-250715",
            response_format=response_format,
            structured_provider='native'
        )
        
        assert result.choices[0].message.content == '{"task": "完成", "status": "success"}'
        assert result.parsed == {"task": "完成", "status": "success"}

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_native_structured_output_error(self, mock_client_class):
        """
        测试异步原生结构化输出错误处理
        ---
        summary: 测试异步原生结构化输出的错误处理路径
        target_lines: 385-391
        ---
        """
        # 设置mock响应 - 无效JSON
        mock_response = Mock()  # 使用普通Mock而不是AsyncMock
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"invalid": json, "missing": quote}',  # 明确无效的JSON
                    "role": "assistant"
                }
            }],
            "usage": {"total_tokens": 50}
        }
        mock_response.status_code = 200
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="生成JSON")]
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"]
                }
            }
        }
        
        result = await plugin.chat_completion_async(
            messages=messages,
            model="doubao-1-5-pro-32k-character-250715",
            response_format=response_format,
            structured_provider='native'
        )
        
        # 检查返回的错误响应
        assert result.choices[0].message.content is not None
        assert "Error:" in result.choices[0].message.content
        assert "豆包返回的内容不是有效JSON" in result.choices[0].message.content

    @patch('httpx.Client')
    def test_sync_exception_handling(self, mock_client_class):
        """
        测试同步聊天完成的异常处理
        ---
        summary: 测试同步方法中的通用异常处理
        target_line: 313
        ---
        """
        # 设置mock抛出异常
        mock_client = Mock()
        mock_client.post.side_effect = Exception("意外错误")
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        result = plugin.chat_completion(messages=messages, model="doubao-1-5-pro-32k-character-250715")
        
        # 检查返回的错误响应
        assert result.choices[0].message.content is not None
        assert "Error:" in result.choices[0].message.content
        assert "意外错误" in result.choices[0].message.content

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_exception_handling(self, mock_client_class):
        """
        测试异步聊天完成的异常处理
        ---
        summary: 测试异步方法中的通用异常处理
        target_lines: 429, 511
        ---
        """
        # 设置mock抛出异常
        mock_client = AsyncMock()
        mock_client.post.side_effect = Exception("异步意外错误")
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        result = await plugin.chat_completion_async(messages=messages, model="doubao-1-5-pro-32k-character-250715")
        
        # 检查返回的错误响应
        assert result.choices[0].message.content is not None
        assert "Error:" in result.choices[0].message.content
        assert "异步意外错误" in result.choices[0].message.content

    @patch('httpx.Client')
    def test_streaming_response_edge_cases(self, mock_client_class):
        """
        测试流式响应的边界情况
        ---
        summary: 测试流式响应处理中的特殊情况
        target_lines: 530, 538-542, 546, 553, 566, 571, 573
        ---
        """
        # 模拟包含[DONE]标记的流式响应
        stream_data = [
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":" World"}}]}\n\n',
            'data: [DONE]\n\n'  # 结束标记
        ]
        
        mock_response = Mock()
        mock_response.iter_lines.return_value = stream_data
        mock_response.status_code = 200
        
        # 设置同步流式上下文管理器
        mock_stream_context = Mock()
        mock_stream_context.__enter__ = Mock(return_value=mock_response)
        mock_stream_context.__exit__ = Mock(return_value=None)
        
        mock_client = Mock()
        mock_client.stream = Mock(return_value=mock_stream_context)
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        result = plugin.chat_completion(
            messages=messages,
            model="doubao-1-5-pro-32k-character-250715",
            stream=True
        )
        
        # 收集流式响应的内容
        chunks = list(result)
        assert len(chunks) == 2  # 应该有两个chunk
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " World"

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_streaming_response_edge_cases(self, mock_client_class):
        """
        测试异步流式响应的边界情况
        ---
        summary: 测试异步流式响应处理中的特殊情况
        target_lines: 581-582, 592
        ---
        """
        # 创建mock异步流式响应
        mock_client = AsyncMock()
        mock_response = AsyncMock()
        
        # 模拟异步流式数据
        async def mock_aiter_lines():
            stream_data = [
                'data: {"choices":[{"delta":{"content":"Async"}}]}',
                'data: {"choices":[{"delta":{"content":" Response"}}]}',
                'data: [DONE]'
            ]
            for line in stream_data:
                yield line
        
        mock_response.aiter_lines = mock_aiter_lines
        mock_response.raise_for_status = AsyncMock()
        
        # 设置异步上下文管理器支持
        mock_async_stream_context = AsyncMock()
        mock_async_stream_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_async_stream_context.__aexit__ = AsyncMock(return_value=None)
        mock_client.stream = Mock(return_value=mock_async_stream_context)
        
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        result = await plugin.chat_completion_async(
            messages=messages,
            model="doubao-1-5-pro-32k-character-250715",
            stream=True
        )
        
        # 验证流式响应内容 - 收集所有块
        chunks = []
        async for chunk in result:
            chunks.append(chunk)
        
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Async"
        assert chunks[1].choices[0].delta.content == " Response"

    @patch('httpx.Client')
    def test_native_structured_output_invalid_json(self, mock_client_class):
        """
        测试原生结构化输出的无效JSON处理
        ---
        summary: 测试原生结构化输出遇到无效JSON时的错误处理
        target_lines: 448-473
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 模拟返回无效JSON的响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"name": "张三", "age":}',  # 无效JSON
                    "role": "assistant"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        messages = [ChatMessage(role="user", content="Hello")]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}}
        }
        
        # 现在期望返回错误响应而不是抛出异常
        result = plugin._handle_native_structured_output(
            "doubao-1-5-pro-32k-character-250715", 
            messages, 
            response_format=response_format
        )
        
        # 检查返回的错误响应
        assert result.choices[0].message.content is not None
        assert "Error:" in result.choices[0].message.content
        assert "豆包返回的内容不是有效JSON" in result.choices[0].message.content

    @patch('httpx.Client')
    def test_native_structured_output_schema_validation_error(self, mock_client_class):
        """
        测试原生结构化输出的schema验证错误
        ---
        summary: 测试原生结构化输出schema验证失败的处理
        target_lines: 448-473
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 模拟返回不符合schema的JSON响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"name": "张三"}',  # 缺少required字段age
                    "role": "assistant"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        messages = [ChatMessage(role="user", content="Hello")]
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"]
                }
            }
        }
        
        result = plugin._handle_native_structured_output(
            "doubao-1-5-pro-32k-character-250715", 
            messages, 
            response_format=response_format
        )
        
        # 验证返回的是错误响应
        assert "Error:" in result.choices[0].message.content
        assert "豆包返回的JSON缺少必需字段" in result.choices[0].message.content


class TestDoubaoPluginAdditionalCoverage:
    """测试豆包插件的额外覆盖率用例"""

    def test_missing_api_key_validation_in_init(self):
        """
        测试初始化时的API密钥验证
        ---
        summary: 测试当API密钥为空时的初始化错误处理
        target_line: 33
        ---
        """
        with pytest.raises(PluginError) as exc_info:
            DoubaoPlugin(name="doubao", api_key="")  # 空API密钥
        
        assert "API key is required" in str(exc_info.value)

    def test_extract_thinking_content_from_thinking_field(self):
        """
        测试从thinking_content字段提取思考内容
        ---
        summary: 测试从响应的thinking_content字段提取思考内容
        target_line: 145
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        response_data = {
            "choices": [{
                "message": {
                    "thinking_content": "这是思考内容",
                    "content": "这是回复内容"
                }
            }]
        }
        
        thinking_content = plugin._extract_thinking_content(response_data)
        assert thinking_content == "这是思考内容"

    def test_prepare_request_with_tool_call_id(self):
        """
        测试准备请求时包含tool_call_id
        ---
        summary: 测试消息包含tool_call_id时的请求准备
        target_line: 162
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi", tool_call_id="call_123")
        ]
        
        request_data = plugin._prepare_doubao_request("doubao-1-5-pro-32k-character-250715", messages)
        
        assert len(request_data["messages"]) == 2
        assert request_data["messages"][1]["tool_call_id"] == "call_123"

    def test_prepare_request_with_tool_calls(self):
        """
        测试准备请求时包含tool_calls
        ---
        summary: 测试消息包含tool_calls时的请求准备
        target_line: 160
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        tool_calls = [{"id": "call_123", "type": "function", "function": {"name": "test_func"}}]
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi", tool_calls=tool_calls)
        ]
        
        request_data = plugin._prepare_doubao_request("doubao-1-5-pro-32k-character-250715", messages)
        
        assert len(request_data["messages"]) == 2
        assert request_data["messages"][1]["tool_calls"] == tool_calls

    def test_prepare_request_with_message_name(self):
        """
        测试准备请求时包含消息名称
        ---
        summary: 测试消息包含name字段时的请求准备
        target_line: 159
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        messages = [
            ChatMessage(role="user", content="Hello", name="user1"),
            ChatMessage(role="assistant", content="Hi", name="assistant1")
        ]
        
        request_data = plugin._prepare_doubao_request("doubao-1-5-pro-32k-character-250715", messages)
        
        assert len(request_data["messages"]) == 2
        assert request_data["messages"][0]["name"] == "user1"
        assert request_data["messages"][1]["name"] == "assistant1"

    @patch('httpx.Client')
    def test_chat_completion_with_structured_output_handling(self, mock_client_class):
        """
        测试聊天完成时的结构化输出处理
        ---
        summary: 测试聊天完成时调用handle_structured_output的路径
        target_line: 313
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 模拟响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Hello response",
                    "role": "assistant"
                }
            }],
            "usage": {"total_tokens": 10}
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        # 模拟handle_structured_output方法
        with patch.object(plugin, 'handle_structured_output') as mock_handle:
            mock_handle.return_value = Mock()  # 返回处理后的响应
            
            messages = [ChatMessage(role="user", content="Hello")]
            response_format = {"type": "json_object"}
            
            result = plugin.chat_completion(
                "doubao-1-5-pro-32k-character-250715", 
                messages, 
                response_format=response_format
            )
            
            # 验证handle_structured_output被调用
            mock_handle.assert_called_once()

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_chat_completion_with_structured_output_handling(self, mock_client_class):
        """
        测试异步聊天完成时的结构化输出处理
        ---
        summary: 测试异步聊天完成时调用handle_structured_output的路径
        target_line: 347
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 模拟异步响应
        mock_response = Mock()  # 使用普通Mock而不是AsyncMock
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Hello response",
                    "role": "assistant"
                }
            }],
            "usage": {"total_tokens": 10}
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        messages = [ChatMessage(role="user", content="Hello")]
        response_format = {"type": "json_object"}
        
        # 直接调用异步方法，不模拟handle_structured_output
        result = await plugin.chat_completion_async(
            "doubao-1-5-pro-32k-character-250715", 
            messages, 
            response_format=response_format
        )
        
        # 验证响应格式
        assert result is not None

    @patch('httpx.Client')
    def test_native_structured_output_success_without_schema(self, mock_client_class):
        """
        测试原生结构化输出成功（无schema验证）
        ---
        summary: 测试原生结构化输出在没有schema验证时的成功路径
        target_lines: 385-391
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 模拟返回有效JSON的响应（无schema验证）
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"name": "张三", "age": 25}',
                    "role": "assistant"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        messages = [ChatMessage(role="user", content="Hello")]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}}  # 无required字段
        }
        
        result = plugin._handle_native_structured_output(
            "doubao-1-5-pro-32k-character-250715", 
            messages, 
            response_format=response_format
        )
        
        assert result.parsed == {"name": "张三", "age": 25}
        assert result.choices[0].message.parsed == {"name": "张三", "age": 25}

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_native_structured_output_success_without_schema(self, mock_client_class):
        """
        测试异步原生结构化输出成功（无schema验证）
        ---
        summary: 测试异步原生结构化输出在没有schema验证时的成功路径
        target_lines: 462-473
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 模拟返回有效JSON的异步响应（无schema验证）
        mock_response = Mock()  # 使用普通Mock而不是AsyncMock
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": '{"name": "张三", "age": 25}',
                    "role": "assistant"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        messages = [ChatMessage(role="user", content="Hello")]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}}  # 无required字段
        }
        
        result = await plugin._handle_native_structured_output_async(
            "doubao-1-5-pro-32k-character-250715", 
            messages, 
            response_format=response_format
        )
        
        assert result.parsed == {"name": "张三", "age": 25}
        assert result.choices[0].message.parsed == {"name": "张三", "age": 25}

    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_native_structured_output_no_content_error(self, mock_client_class):
        """
        测试异步原生结构化输出无内容错误
        ---
        summary: 测试异步原生结构化输出当没有返回内容时的错误处理
        target_lines: 480-482
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        
        # 模拟返回空内容的异步响应
        mock_response = Mock()  # 使用普通Mock而不是AsyncMock
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "",  # 空内容
                    "role": "assistant"
                }
            }]
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        messages = [ChatMessage(role="user", content="Hello")]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}}
        }
        
        result = await plugin._handle_native_structured_output_async(
            "doubao-1-5-pro-32k-character-250715", 
            messages, 
            response_format=response_format
        )
        
        # 验证返回的是错误响应
        assert "Error:" in result.choices[0].message.content
        assert "豆包未返回有效的响应内容" in result.choices[0].message.content

    def test_temperature_validation_edge_cases(self):
        """
        测试温度参数验证的边界情况
        ---
        summary: 测试温度参数的边界值验证
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 测试边界值
        plugin._validate_request("doubao-1-5-pro-32k-character-250715", messages, temperature=0)
        plugin._validate_request("doubao-1-5-pro-32k-character-250715", messages, temperature=2)
        
        # 测试超出范围的值
        with pytest.raises(ValidationError) as exc_info:
            plugin._validate_request("doubao-1-5-pro-32k-character-250715", messages, temperature=-0.1)
        assert "Temperature must be between 0 and 2" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            plugin._validate_request("doubao-1-5-pro-32k-character-250715", messages, temperature=2.1)
        assert "Temperature must be between 0 and 2" in str(exc_info.value)

    def test_max_tokens_validation_edge_cases(self):
        """
        测试max_tokens参数验证的边界情况
        ---
        summary: 测试max_tokens参数的边界值验证
        ---
        """
        plugin = DoubaoPlugin(name="doubao", api_key="test_key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 测试边界值
        plugin._validate_request("doubao-1-5-pro-32k-character-250715", messages, max_tokens=1)
        
        # 测试无效值
        with pytest.raises(ValidationError) as exc_info:
            plugin._validate_request("doubao-1-5-pro-32k-character-250715", messages, max_tokens=0)
        assert "max_tokens must be positive" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            plugin._validate_request("doubao-1-5-pro-32k-character-250715", messages, max_tokens=-1)
        assert "max_tokens must be positive" in str(exc_info.value)