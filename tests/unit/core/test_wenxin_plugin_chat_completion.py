"""
文心一言插件聊天完成功能测试。

测试覆盖：
- 同步聊天完成
- 异步聊天完成
- 流式响应处理
- 错误处理
- 响应转换
"""

import pytest
import json
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any

from harborai.core.plugins.wenxin_plugin import WenxinPlugin
from harborai.core.base_plugin import ChatMessage, ChatCompletion, ChatCompletionChunk
from harborai.utils.exceptions import PluginError, ValidationError


class TestWenxinPluginChatCompletion:
    """测试文心一言插件聊天完成功能。"""
    
    @patch('httpx.Client')
    def test_chat_completion_success(self, mock_client_class):
        """测试同步聊天完成成功。"""
        # 设置mock响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "ernie-3.5-8k",
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
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        result = plugin.chat_completion("ernie-3.5-8k", messages)
        
        assert isinstance(result, ChatCompletion)
        assert result.model == "ernie-3.5-8k"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello! How can I help you?"
        assert result.usage.total_tokens == 18
        
        # 验证请求调用
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "/chat/completions"
        request_data = call_args[1]["json"]
        assert request_data["model"] == "ernie-3.5-8k"
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert request_data["messages"][0]["content"] == "Hello"
    
    @patch('httpx.Client')
    def test_chat_completion_with_system_message(self, mock_client_class):
        """测试包含系统消息的聊天完成。"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "ernie-3.5-8k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I'm a helpful assistant."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25
            }
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Who are you?")
        ]
        
        result = plugin.chat_completion("ernie-3.5-8k", messages)
        
        assert isinstance(result, ChatCompletion)
        
        # 验证系统消息被合并到用户消息中
        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert "You are a helpful assistant." in request_data["messages"][0]["content"]
        assert "Who are you?" in request_data["messages"][0]["content"]
    
    @patch('httpx.Client')
    def test_chat_completion_with_parameters(self, mock_client_class):
        """测试带参数的聊天完成。"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "ernie-3.5-8k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Response with parameters"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 4,
                "total_tokens": 19
            }
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Test")]
        
        result = plugin.chat_completion(
            "ernie-3.5-8k", 
            messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["END"]
        )
        
        assert isinstance(result, ChatCompletion)
        
        # 验证参数传递
        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["temperature"] == 0.7
        assert request_data["top_p"] == 0.9
        assert request_data["max_tokens"] == 100
        assert request_data["stop"] == ["END"]
    
    @patch('httpx.Client')
    def test_chat_completion_api_error(self, mock_client_class):
        """测试API错误处理。"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "error_code": 18,
            "error_msg": "Open api daily request limit reached"
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # API错误会被转换为错误响应而不是抛出异常
        result = plugin.chat_completion("ernie-3.5-8k", messages)
        assert isinstance(result, ChatCompletion)
        assert "error" in result.choices[0].message.content.lower()
    
    @patch('httpx.Client')
    def test_chat_completion_http_error(self, mock_client_class):
        """测试HTTP错误处理。"""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 500 Error")
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        result = plugin.chat_completion("ernie-3.5-8k", messages)
        
        # 应该返回错误响应而不是抛出异常
        assert isinstance(result, ChatCompletion)
        assert "error" in result.choices[0].message.content.lower()


class TestWenxinPluginAsyncChatCompletion:
    """测试文心一言插件异步聊天完成功能。"""
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_chat_completion_async_success(self, mock_async_client_class):
        """测试异步聊天完成成功。"""
        # 设置mock响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "ernie-3.5-8k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Async response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12
            }
        }
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_async_client_class.return_value = mock_client
        
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello async")]
        
        result = await plugin.chat_completion_async("ernie-3.5-8k", messages)
        
        assert isinstance(result, ChatCompletion)
        assert result.model == "ernie-3.5-8k"
        assert result.choices[0].message.content == "Async response"
        assert result.usage.total_tokens == 12
        
        # 验证异步请求调用
        mock_client.post.assert_called_once()
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_chat_completion_async_error(self, mock_async_client_class):
        """测试异步聊天完成错误处理。"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "error_code": 17,
            "error_msg": "Daily limit exceeded"
        }
        mock_response.raise_for_status = Mock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_async_client_class.return_value = mock_client
        
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # API错误会被转换为错误响应而不是抛出异常
        result = await plugin.chat_completion_async("ernie-3.5-8k", messages)
        assert isinstance(result, ChatCompletion)
        assert "error" in result.choices[0].message.content.lower()


class TestWenxinPluginStreamResponse:
    """测试文心一言插件流式响应处理。"""
    
    @patch('httpx.Client')
    def test_chat_completion_stream(self, mock_client_class):
        """测试流式聊天完成。"""
        # 模拟流式响应数据
        stream_data = [
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"ernie-3.5-8k","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"ernie-3.5-8k","choices":[{"index":0,"delta":{"content":" there"},"finish_reason":null}]}',
            'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,"model":"ernie-3.5-8k","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
            'data: [DONE]'
        ]
        
        mock_response = Mock()
        mock_response.iter_lines.return_value = stream_data
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        result = plugin.chat_completion("ernie-3.5-8k", messages, stream=True)
        
        # 收集流式响应
        chunks = list(result)
        
        assert len(chunks) == 3  # 不包括[DONE]
        assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " there"
        assert chunks[2].choices[0].finish_reason == "stop"
        
        # 验证流式请求
        call_args = mock_client.post.call_args
        request_data = call_args[1]["json"]
        assert request_data["stream"] is True


class TestWenxinPluginResponseConversion:
    """测试文心一言插件响应转换功能。"""
    
    def test_convert_to_harbor_response(self):
        """测试响应转换。"""
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        
        wenxin_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "ernie-3.5-8k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7
            }
        }
        
        messages = [ChatMessage(role="user", content="Test")]
        result = plugin._convert_to_harbor_response(wenxin_response, "ernie-3.5-8k", messages)
        
        assert isinstance(result, ChatCompletion)
        assert result.id == "chatcmpl-test"
        assert result.model == "ernie-3.5-8k"
        assert result.choices[0].message.content == "Test response"
        assert result.usage.total_tokens == 7
    
    def test_convert_to_harbor_response_with_thinking(self):
        """测试包含推理内容的响应转换。"""
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        
        wenxin_response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "ernie-x1-turbo-32k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning_content": "Let me think about this..."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        
        messages = [ChatMessage(role="user", content="Think about this")]
        result = plugin._convert_to_harbor_response(wenxin_response, "ernie-x1-turbo-32k", messages)
        
        assert isinstance(result, ChatCompletion)
        assert result.choices[0].message.content == "Final answer"
        # 推理内容应该在reasoning_content字段中
        assert hasattr(result.choices[0].message, 'reasoning_content')
        assert result.choices[0].message.reasoning_content == "Let me think about this..."
    
    def test_convert_to_harbor_chunk(self):
        """测试流式响应块转换。"""
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        
        chunk_data = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "ernie-3.5-8k",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello"
                },
                "finish_reason": None
            }]
        }
        
        result = plugin._convert_to_harbor_chunk(chunk_data, "ernie-3.5-8k")
        
        assert isinstance(result, ChatCompletionChunk)
        assert result.id == "chatcmpl-test"
        assert result.model == "ernie-3.5-8k"
        assert result.choices[0].delta.content == "Hello"
        assert result.choices[0].finish_reason is None


class TestWenxinPluginRequestPreparation:
    """测试文心一言插件请求准备功能。"""
    
    def test_prepare_wenxin_request_basic(self):
        """测试基本请求准备。"""
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        request_data = plugin._prepare_wenxin_request("ernie-3.5-8k", messages)
        
        assert request_data["model"] == "ernie-3.5-8k"
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert request_data["messages"][0]["content"] == "Hello"
    
    def test_prepare_wenxin_request_with_system(self):
        """测试包含系统消息的请求准备。"""
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello")
        ]
        
        request_data = plugin._prepare_wenxin_request("ernie-3.5-8k", messages)
        
        # 系统消息应该被合并到第一个用户消息中
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert "You are helpful" in request_data["messages"][0]["content"]
        assert "Hello" in request_data["messages"][0]["content"]
    
    def test_prepare_wenxin_request_with_parameters(self):
        """测试包含参数的请求准备。"""
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        request_data = plugin._prepare_wenxin_request(
            "ernie-3.5-8k", 
            messages,
            temperature=0.8,
            top_p=0.95,
            max_tokens=200,
            stop=["STOP"],
            stream=True
        )
        
        assert request_data["temperature"] == 0.8
        assert request_data["top_p"] == 0.95
        assert request_data["max_tokens"] == 200
        assert request_data["stop"] == ["STOP"]
        assert request_data["stream"] is True
    
    def test_get_model_endpoint(self):
        """测试获取模型端点。"""
        plugin = WenxinPlugin(name="wenxin", api_key="test-key")
        
        endpoint = plugin._get_model_endpoint("ernie-3.5-8k")
        assert endpoint == "/chat/completions"
        
        endpoint = plugin._get_model_endpoint("ernie-4.0-turbo-8k")
        assert endpoint == "/chat/completions"