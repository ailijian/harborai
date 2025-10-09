#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI插件全面测试

测试目标：提升OpenAI插件覆盖率到80%+
包括：
- 错误处理测试
- 请求参数准备测试
- 响应转换测试
- 流式响应测试
- 结构化输出测试
- 异步操作测试

遵循VIBE编码规范，使用TDD方法。
"""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional

import openai
import httpx
from openai.types.chat import ChatCompletion as OpenAIChatCompletion, ChatCompletionChunk as OpenAIChatCompletionChunk

from harborai.core.plugins.openai_plugin import OpenAIPlugin
from harborai.core.base_plugin import ChatMessage, ChatCompletion, ChatCompletionChunk, ModelInfo
from harborai.utils.exceptions import APIError, AuthenticationError, RateLimitError, TimeoutError


class TestOpenAIPluginErrorHandling:
    """测试OpenAI插件错误处理"""
    
    def test_handle_authentication_error(self):
        """测试认证错误处理"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建模拟响应
        mock_response = Mock()
        mock_response.status_code = 401
        
        # 创建OpenAI认证错误
        openai_error = openai.AuthenticationError("Invalid API key", response=mock_response, body=None)
        
        # 测试错误转换
        harbor_error = plugin._handle_openai_error(openai_error)
        
        assert isinstance(harbor_error, AuthenticationError)
        assert "OpenAI authentication failed" in str(harbor_error)
        assert harbor_error.details["original_error"] == "Invalid API key"
    
    def test_handle_rate_limit_error(self):
        """测试速率限制错误处理"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建模拟响应
        mock_response = Mock()
        mock_response.status_code = 429
        
        # 创建OpenAI速率限制错误
        openai_error = openai.RateLimitError("Rate limit exceeded", response=mock_response, body=None)
        
        # 测试错误转换
        harbor_error = plugin._handle_openai_error(openai_error)
        
        assert isinstance(harbor_error, RateLimitError)
        assert "OpenAI rate limit exceeded" in str(harbor_error)
        assert harbor_error.details["original_error"] == "Rate limit exceeded"
    
    def test_handle_timeout_error(self):
        """测试超时错误处理"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建OpenAI超时错误
        openai_error = openai.APITimeoutError("Request timed out.")
        
        # 测试错误转换
        harbor_error = plugin._handle_openai_error(openai_error)
        
        assert isinstance(harbor_error, TimeoutError)
        assert "OpenAI API timeout" in str(harbor_error)
        assert harbor_error.details["original_error"] == "Request timed out."
    
    def test_handle_api_error(self):
        """测试API错误处理"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建模拟请求
        mock_request = Mock()
        mock_request.url = "https://api.openai.com/v1/chat/completions"
        
        # 创建OpenAI API错误
        openai_error = openai.APIError("Server error", request=mock_request, body=None)
        
        # 测试错误转换
        harbor_error = plugin._handle_openai_error(openai_error)
        
        assert isinstance(harbor_error, APIError)
        assert "OpenAI API error" in str(harbor_error)
        assert harbor_error.details["original_error"] == "Server error"
    
    def test_handle_unknown_error(self):
        """测试未知错误处理"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建未知错误
        unknown_error = ValueError("Unknown error")
        
        # 测试错误转换
        harbor_error = plugin._handle_openai_error(unknown_error)
        
        # 未知错误应该原样返回
        assert harbor_error is unknown_error


class TestOpenAIPluginRequestPreparation:
    """测试OpenAI插件请求参数准备"""
    
    def test_prepare_basic_request(self):
        """测试基础请求参数准备"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there!")
        ]
        
        request_params = plugin._prepare_openai_request("gpt-4o", messages)
        
        assert request_params["model"] == "gpt-4o"
        assert len(request_params["messages"]) == 2
        assert request_params["messages"][0]["role"] == "user"
        assert request_params["messages"][0]["content"] == "Hello"
        assert request_params["messages"][1]["role"] == "assistant"
        assert request_params["messages"][1]["content"] == "Hi there!"
    
    def test_prepare_request_with_message_attributes(self):
        """测试包含消息属性的请求参数准备"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        messages = [
            ChatMessage(
                role="user", 
                content="Hello",
                name="test_user",
                function_call={"name": "test_func", "arguments": "{}"},
                tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "test"}}]
            )
        ]
        
        request_params = plugin._prepare_openai_request("gpt-4o", messages)
        
        message = request_params["messages"][0]
        assert message["name"] == "test_user"
        assert message["function_call"]["name"] == "test_func"
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["id"] == "call_1"
    
    def test_prepare_request_with_supported_params(self):
        """测试包含支持参数的请求准备"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        kwargs = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2,
            "stop": ["\\n"],
            "seed": 42,
            "user": "test_user"
        }
        
        request_params = plugin._prepare_openai_request("gpt-4o", messages, **kwargs)
        
        assert request_params["temperature"] == 0.7
        assert request_params["max_tokens"] == 100
        assert request_params["top_p"] == 0.9
        assert request_params["frequency_penalty"] == 0.1
        assert request_params["presence_penalty"] == 0.2
        assert request_params["stop"] == ["\\n"]
        assert request_params["seed"] == 42
        assert request_params["user"] == "test_user"
    
    def test_prepare_request_with_json_schema_response_format(self):
        """测试包含JSON schema响应格式的请求准备"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        messages = [ChatMessage(role="user", content="Generate JSON")]
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        
        request_params = plugin._prepare_openai_request(
            "gpt-4o", 
            messages, 
            response_format=response_format
        )
        
        assert request_params["response_format"] == response_format
    
    def test_prepare_request_filters_none_values(self):
        """测试请求准备过滤None值"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        kwargs = {
            "temperature": 0.7,
            "max_tokens": None,  # 应该被过滤
            "top_p": 0.9,
            "stop": None  # 应该被过滤
        }
        
        request_params = plugin._prepare_openai_request("gpt-4o", messages, **kwargs)
        
        assert request_params["temperature"] == 0.7
        assert request_params["top_p"] == 0.9
        assert "max_tokens" not in request_params
        assert "stop" not in request_params


class TestOpenAIPluginResponseConversion:
    """测试OpenAI插件响应转换"""
    
    def test_convert_basic_response(self):
        """测试基础响应转换"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建模拟OpenAI响应
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Hello from OpenAI!"
        mock_message.tool_calls = None
        mock_message.function_call = None
        
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        
        mock_response = Mock()
        mock_response.id = "chatcmpl-test123"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4o"
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        # 测试转换
        harbor_response = plugin._convert_to_harbor_response(mock_response)
        
        assert isinstance(harbor_response, ChatCompletion)
        assert harbor_response.id == "chatcmpl-test123"
        assert harbor_response.object == "chat.completion"
        assert harbor_response.created == 1234567890
        assert harbor_response.model == "gpt-4o"
        assert len(harbor_response.choices) == 1
        assert harbor_response.choices[0].message.content == "Hello from OpenAI!"
        assert harbor_response.usage.prompt_tokens == 10
        assert harbor_response.usage.completion_tokens == 5
        assert harbor_response.usage.total_tokens == 15

    @patch('harborai.core.plugins.openai_plugin.AsyncOpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    @pytest.mark.asyncio
    async def test_chat_completion_async_error_handling(self, mock_get_logger, mock_async_openai_class):
        """
        测试异步聊天完成的错误处理
        
        覆盖未测试的代码行：400-407 (异步错误处理)
        """
        # 配置模拟客户端抛出异常
        mock_client = AsyncMock()
        mock_request = Mock()
        mock_request.url = "https://api.openai.com/v1/chat/completions"
        mock_client.chat.completions.create.side_effect = openai.APIError(
            message="API Error",
            request=mock_request,
            body=None
        )
        mock_async_openai_class.return_value = mock_client
        
        # 配置模拟logger
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        plugin = OpenAIPlugin(api_key="test-key")
        
        messages = [
            ChatMessage(role="user", content="Hello")
        ]
        
        # 测试异步错误处理
        with pytest.raises(APIError):
            await plugin.chat_completion_async("gpt-4o", messages)
        
        # 验证错误处理被调用
        mock_client.chat.completions.create.assert_called_once()
        mock_logger.error.assert_called()

    def test_convert_response_with_tool_calls(self):
        """
        测试包含工具调用的响应转换
        
        覆盖未测试的代码行：262-272 (tool_calls处理)
        """
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建模拟工具调用
        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_function = Mock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"location": "Beijing"}'
        mock_tool_call.function = mock_function
        
        # 创建模拟message with tool_calls
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]
        mock_message.function_call = None
        
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 30
        
        mock_response = Mock()
        mock_response.id = "chatcmpl-tool123"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4"
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        # 转换响应
        harbor_response = plugin._convert_to_harbor_response(mock_response)
        
        # 验证工具调用转换
        assert len(harbor_response.choices) == 1
        choice = harbor_response.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) == 1
        
        tool_call = choice.message.tool_calls[0]
        assert tool_call["id"] == "call_123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert tool_call["function"]["arguments"] == '{"location": "Beijing"}'

    def test_convert_response_with_reasoning_content(self):
        """
        测试包含推理内容的响应转换 (o1模型)
        
        覆盖未测试的代码行：275-278 (reasoning_content处理)
        """
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建模拟message with reasoning_content
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Final answer"
        mock_message.reasoning_content = "Let me think step by step..."
        mock_message.tool_calls = None
        mock_message.function_call = None
        
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 25
        mock_usage.total_tokens = 40
        
        mock_response = Mock()
        mock_response.id = "chatcmpl-o1-123"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "o1-preview"
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        # 转换响应
        harbor_response = plugin._convert_to_harbor_response(mock_response)
        
        # 验证推理内容转换
        assert len(harbor_response.choices) == 1
        choice = harbor_response.choices[0]
        assert choice.message.content == "Final answer"
        assert hasattr(choice.message, 'reasoning_content')
        assert choice.message.reasoning_content == "Let me think step by step..."

    def test_convert_response_with_system_fingerprint(self):
        """
        测试包含系统指纹的响应转换
        
        覆盖未测试的代码行：304-307 (system_fingerprint处理)
        """
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建模拟message
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Hello!"
        mock_message.tool_calls = None
        mock_message.function_call = None
        
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 3
        mock_usage.total_tokens = 8
        
        # 创建带system_fingerprint的响应
        mock_response = Mock()
        mock_response.id = "chatcmpl-123"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4"
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.system_fingerprint = "fp_44709d6fcb"
        
        # 转换响应
        harbor_response = plugin._convert_to_harbor_response(mock_response)
        
        # 验证系统指纹转换
        assert hasattr(harbor_response, 'system_fingerprint')
        assert harbor_response.system_fingerprint == "fp_44709d6fcb"


class TestOpenAIPluginStreamHandling:
    """测试OpenAI插件流式处理"""
    
    def test_convert_chunk_to_harbor_format(self):
        """测试chunk转换为Harbor格式"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建模拟OpenAI chunk
        mock_delta = Mock()
        mock_delta.role = "assistant"
        mock_delta.content = "Hello"
        mock_delta.tool_calls = None
        mock_delta.function_call = None
        
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.delta = mock_delta
        mock_choice.finish_reason = None
        
        mock_chunk = Mock()
        mock_chunk.id = "chatcmpl-test123"
        mock_chunk.object = "chat.completion.chunk"
        mock_chunk.created = 1234567890
        mock_chunk.model = "gpt-4o"
        mock_chunk.choices = [mock_choice]
        
        # 测试转换
        harbor_chunk = plugin._convert_chunk_to_harbor_format(mock_chunk)
        
        assert isinstance(harbor_chunk, ChatCompletionChunk)
        assert harbor_chunk.id == "chatcmpl-test123"
        assert harbor_chunk.object == "chat.completion.chunk"
        assert harbor_chunk.created == 1234567890
        assert harbor_chunk.model == "gpt-4o"
        assert len(harbor_chunk.choices) == 1
        assert harbor_chunk.choices[0].delta.content == "Hello"


class TestOpenAIPluginChatCompletion:
    """测试OpenAI插件聊天完成功能"""
    
    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    def test_chat_completion_success(self, mock_openai_class):
        """测试成功的聊天完成"""
        # 创建模拟响应
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Hello from OpenAI!"
        mock_message.tool_calls = None
        mock_message.function_call = None
        
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        
        mock_response = Mock()
        mock_response.id = "chatcmpl-test123"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4o"
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        # 配置模拟客户端
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 测试聊天完成
        response = plugin.chat_completion("gpt-4o", messages)
        
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Hello from OpenAI!"
        
        # 验证调用参数
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4o"
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["content"] == "Hello"
    
    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    def test_chat_completion_with_error(self, mock_get_logger, mock_openai_class):
        """测试聊天完成时的错误处理"""
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 创建模拟响应
        mock_response = Mock()
        mock_response.status_code = 401
        
        # 配置模拟客户端抛出错误
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.AuthenticationError("Invalid API key", response=mock_response, body=None)
        mock_openai_class.return_value = mock_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 测试错误处理
        with pytest.raises(AuthenticationError):
            plugin.chat_completion("gpt-4o", messages)
    
    @patch('harborai.core.plugins.openai_plugin.AsyncOpenAI')
    @pytest.mark.asyncio
    async def test_chat_completion_async_success(self, mock_async_openai_class):
        """测试成功的异步聊天完成"""
        # 创建模拟响应
        mock_message = Mock()
        mock_message.role = "assistant"
        mock_message.content = "Hello from OpenAI async!"
        mock_message.tool_calls = None
        mock_message.function_call = None
        
        mock_choice = Mock()
        mock_choice.index = 0
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        
        mock_response = Mock()
        mock_response.id = "chatcmpl-test123"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4o"
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        # 配置模拟异步客户端
        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create.return_value = mock_response
        mock_async_openai_class.return_value = mock_async_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 测试异步聊天完成
        response = await plugin.chat_completion_async("gpt-4o", messages)
        
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == "Hello from OpenAI async!"
        
        # 验证调用参数
        mock_async_client.chat.completions.create.assert_called_once()
        call_args = mock_async_client.chat.completions.create.call_args[1]
        assert call_args["model"] == "gpt-4o"
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["content"] == "Hello"


class TestOpenAIPluginStreamingAndStructured:
    """测试OpenAI插件流式处理和结构化输出"""
    
    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    def test_chat_completion_streaming(self, mock_get_logger, mock_openai_class):
        """测试流式聊天完成"""
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        # 创建模拟流式响应
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta = Mock()
        mock_chunk1.choices[0].delta.content = "Hello"
        mock_chunk1.choices[0].delta.role = "assistant"
        mock_chunk1.choices[0].delta.tool_calls = None  # 添加tool_calls属性
        mock_chunk1.choices[0].index = 0  # 添加index属性
        mock_chunk1.choices[0].finish_reason = None
        mock_chunk1.id = "chunk-1"
        mock_chunk1.object = "chat.completion.chunk"  # 添加object属性
        mock_chunk1.created = 1234567890
        mock_chunk1.model = "gpt-4o"
        mock_chunk1.usage = None
        
        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta = Mock()
        mock_chunk2.choices[0].delta.content = " World"
        mock_chunk2.choices[0].delta.role = None
        mock_chunk2.choices[0].delta.tool_calls = None  # 添加tool_calls属性
        mock_chunk2.choices[0].index = 0  # 添加index属性
        mock_chunk2.choices[0].finish_reason = "stop"
        mock_chunk2.id = "chunk-2"
        mock_chunk2.object = "chat.completion.chunk"  # 添加object属性
        mock_chunk2.created = 1234567890
        mock_chunk2.model = "gpt-4o"
        mock_chunk2.usage = Mock()
        mock_chunk2.usage.prompt_tokens = 10
        mock_chunk2.usage.completion_tokens = 5
        mock_chunk2.usage.total_tokens = 15
        
        # 创建一个可迭代的流对象
        class MockStream:
            def __init__(self, chunks):
                self.chunks = chunks
                # 添加choices属性以兼容测试
                self.choices = None
            
            def __iter__(self):
                return iter(self.chunks)
        
        # 配置模拟客户端
        mock_client = Mock()
        # 创建流对象，在流式模式下返回MockStream，非流式模式下返回普通响应
        def create_side_effect(**request_params):
            print(f"create_side_effect called with request_params: {request_params}")
            if request_params.get('stream', False):
                mock_stream = MockStream([mock_chunk1, mock_chunk2])
                print(f"Returning MockStream: {mock_stream}")
                return mock_stream
            else:
                # 返回一个普通的响应对象用于非流式调用
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = "Hello World"
                mock_response.choices[0].message.role = "assistant"
                mock_response.choices[0].message.tool_calls = None
                mock_response.choices[0].index = 0
                mock_response.choices[0].finish_reason = "stop"
                mock_response.id = "response-1"
                mock_response.object = "chat.completion"
                mock_response.created = 1234567890
                mock_response.model = "gpt-4o"
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 5
                mock_response.usage.total_tokens = 15
                return mock_response
        
        mock_client.chat.completions.create.side_effect = create_side_effect
        mock_openai_class.return_value = mock_client
        
        # 添加调试信息
        print(f"Mock client setup complete")
        print(f"Mock side_effect: {mock_client.chat.completions.create.side_effect}")
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 测试流式聊天完成
        response_generator = plugin.chat_completion("gpt-4o", messages, stream=True)
        chunks = list(response_generator)
        
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " World"
        assert chunks[1].choices[0].finish_reason == "stop"
    
    @patch('harborai.core.plugins.openai_plugin.AsyncOpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    @pytest.mark.asyncio
    async def test_chat_completion_async_streaming(self, mock_get_logger, mock_async_openai_class):
        """测试异步流式聊天完成"""
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        # 创建模拟异步流式响应
        class MockAsyncStream:
            def __init__(self):
                self.chunks = [
                    self._create_mock_chunk("Async chunk", "async-chunk-1")
                ]
                self.index = 0
                # 添加choices属性以兼容测试
                self.choices = None
            
            def _create_mock_chunk(self, content, chunk_id):
                mock_chunk = Mock()
                mock_chunk.choices = [Mock()]
                mock_chunk.choices[0].delta = Mock()
                mock_chunk.choices[0].delta.content = content
                mock_chunk.choices[0].delta.role = None
                mock_chunk.choices[0].delta.tool_calls = None  # 添加tool_calls属性
                mock_chunk.choices[0].index = 0  # 添加index属性
                mock_chunk.choices[0].finish_reason = None
                mock_chunk.id = chunk_id
                mock_chunk.object = "chat.completion.chunk"  # 添加object属性
                mock_chunk.created = 1234567890
                mock_chunk.model = "gpt-4o"
                mock_chunk.usage = None
                return mock_chunk
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk
        
        # 配置模拟异步客户端
        mock_async_client = AsyncMock()
        
        # 创建异步side_effect函数
        async def async_create_side_effect(**request_params):
            if request_params.get('stream', False):
                return MockAsyncStream()
            else:
                # 返回一个普通的响应对象用于非流式调用
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = "Async Hello World"
                mock_response.choices[0].message.role = "assistant"
                mock_response.choices[0].message.tool_calls = None
                mock_response.choices[0].index = 0
                mock_response.choices[0].finish_reason = "stop"
                mock_response.id = "async-response-1"
                mock_response.object = "chat.completion"
                mock_response.created = 1234567890
                mock_response.model = "gpt-4o"
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 5
                mock_response.usage.total_tokens = 15
                return mock_response
        
        # 使用 AsyncMock 来模拟异步方法
        mock_async_client.chat.completions.create = AsyncMock(side_effect=async_create_side_effect)
        mock_async_openai_class.return_value = mock_async_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 测试异步流式聊天完成
        response_generator = await plugin.chat_completion_async("gpt-4o", messages, stream=True)
        chunks = []
        async for chunk in response_generator:
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Async chunk"
    
    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    @patch('harborai.api.structured.default_handler')
    def test_chat_completion_with_structured_output(self, mock_default_handler, mock_get_logger, mock_openai_class):
        """测试结构化输出"""
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 模拟结构化输出处理器
        def mock_parse_streaming_response(content_stream, schema, provider='agently', **kwargs):
            # 消费content_stream以模拟实际处理
            content_list = list(content_stream)
            # 返回解析结果
            yield {"name": "John", "age": 30}
        
        mock_default_handler.parse_streaming_response.side_effect = mock_parse_streaming_response
        mock_default_handler._parse_with_native.return_value = {"name": "John", "age": 30}
        # 创建模拟响应
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = '{"name": "John", "age": 30}'
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.tool_calls = None  # 添加tool_calls属性
        mock_response.choices[0].index = 0  # 添加index属性
        mock_response.choices[0].finish_reason = "stop"
        mock_response.id = "response-1"
        mock_response.object = "chat.completion"  # 添加object属性
        mock_response.created = 1234567890
        mock_response.model = "gpt-4o"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 20
        mock_response.usage.completion_tokens = 10
        mock_response.usage.total_tokens = 30
        
        # 配置模拟客户端
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Generate a person")]
        
        # 测试结构化输出
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        }
        
        response = plugin.chat_completion("gpt-4o", messages, response_format=response_format)
        
        assert isinstance(response, ChatCompletion)
        assert response.choices[0].message.content == '{"name": "John", "age": 30}'
        
        # 验证调用参数包含response_format
        call_args = mock_client.chat.completions.create.call_args[1]
        assert "response_format" in call_args

    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    def test_streaming_structured_output_success(self, mock_get_logger, mock_openai_class):
        """
        测试流式结构化输出成功场景
        
        覆盖未测试的代码行：438-483 (_handle_streaming_structured_output)
        """
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 创建模拟流式chunk
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta = Mock()
        mock_chunk1.choices[0].delta.content = '{"name":'
        mock_chunk1.choices[0].delta.role = "assistant"
        mock_chunk1.choices[0].delta.tool_calls = None
        mock_chunk1.choices[0].index = 0
        mock_chunk1.choices[0].finish_reason = None
        mock_chunk1.id = "chunk-1"
        mock_chunk1.object = "chat.completion.chunk"
        mock_chunk1.created = 1234567890
        mock_chunk1.model = "gpt-4o"
        mock_chunk1.usage = None
        
        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta = Mock()
        mock_chunk2.choices[0].delta.content = ' "John", "age": 30}'
        mock_chunk2.choices[0].delta.role = None
        mock_chunk2.choices[0].delta.tool_calls = None
        mock_chunk2.choices[0].index = 0
        mock_chunk2.choices[0].finish_reason = "stop"
        mock_chunk2.id = "chunk-2"
        mock_chunk2.object = "chat.completion.chunk"
        mock_chunk2.created = 1234567890
        mock_chunk2.model = "gpt-4o"
        mock_chunk2.usage = Mock()
        mock_chunk2.usage.prompt_tokens = 10
        mock_chunk2.usage.completion_tokens = 5
        mock_chunk2.usage.total_tokens = 15
        
        # 创建一个可迭代的流对象（支持多次迭代）
        class MockStream:
            def __init__(self, chunks):
                self.chunks = chunks
            
            def __iter__(self):
                # 每次迭代都返回一个新的迭代器，支持多次迭代
                return iter(self.chunks.copy())
        
        # 配置模拟客户端
        mock_client = Mock()
        def create_side_effect(**request_params):
            print(f"create_side_effect called with request_params: {request_params}")
            if request_params.get('stream', False):
                mock_stream = MockStream([mock_chunk1, mock_chunk2])
                print(f"Returning MockStream: {mock_stream}")
                print(f"MockStream chunks count: {len(mock_stream.chunks)}")
                return mock_stream
            else:
                # 返回普通响应
                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message = Mock()
                mock_response.choices[0].message.content = '{"name": "John", "age": 30}'
                mock_response.choices[0].message.role = "assistant"
                mock_response.choices[0].message.tool_calls = None
                mock_response.choices[0].index = 0
                mock_response.choices[0].finish_reason = "stop"
                mock_response.id = "response-1"
                mock_response.object = "chat.completion"
                mock_response.created = 1234567890
                mock_response.model = "gpt-4o"
                mock_response.usage = Mock()
                mock_response.usage.prompt_tokens = 10
                mock_response.usage.completion_tokens = 5
                mock_response.usage.total_tokens = 15
                return mock_response
        
        mock_client.chat.completions.create.side_effect = create_side_effect
        mock_openai_class.return_value = mock_client
        
        # 模拟结构化输出处理器的导入
        with patch('harborai.api.structured.default_handler') as mock_default_handler:
            def mock_parse_streaming_response(content_stream, schema, provider='agently', **kwargs):
                print(f"Mock parse_streaming_response called with schema: {schema}, provider: {provider}")
                # 消费content_stream以模拟实际处理
                content_list = list(content_stream)
                print(f"Content list: {content_list}")
                # 返回解析结果
                yield {"name": "John", "age": 30}
            
            mock_default_handler.parse_streaming_response.side_effect = mock_parse_streaming_response
            
            plugin = OpenAIPlugin(api_key="test-key")
            messages = [ChatMessage(role="user", content="Generate user data")]
            
            # JSON schema
            json_schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
            
            # 测试流式结构化输出
            response_format = {"type": "json_schema", "json_schema": {"schema": json_schema}}
            print(f"Response format: {response_format}")
            
            # 添加调试信息
            print(f"Mock client: {mock_client}")
            print(f"Mock client.chat.completions.create: {mock_client.chat.completions.create}")
            print(f"Mock side_effect: {mock_client.chat.completions.create.side_effect}")
            
            response_generator = plugin.chat_completion(
                "gpt-4o", 
                messages, 
                stream=True,
                response_format=response_format
            )
            
            # 收集所有chunks
            chunks = list(response_generator)
            
            # 验证结果
            assert len(chunks) >= 1
            # 验证chunk的基本结构
            for chunk in chunks:
                assert hasattr(chunk, 'choices')
                assert len(chunk.choices) > 0
                assert hasattr(chunk.choices[0], 'delta')

    @patch('harborai.core.plugins.openai_plugin.AsyncOpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    @patch('harborai.api.structured.default_handler')
    @pytest.mark.asyncio
    async def test_async_streaming_structured_output_success(self, mock_default_handler, mock_get_logger, mock_async_openai_class):
        """
        测试异步流式结构化输出成功场景
        
        覆盖未测试的代码行：487-532 (_handle_async_streaming_structured_output)
        """
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 模拟异步流式响应
        class MockAsyncStream:
            def __init__(self):
                self.index = 0
                self.chunks = [
                    self._create_chunk('{"name":', None),
                    self._create_chunk(' "John", "age": 30}', "stop")
                ]
                self.index = 0
            
            def _create_chunk(self, content, finish_reason):
                chunk = Mock()
                chunk.choices = [Mock()]
                chunk.choices[0].delta = Mock()
                chunk.choices[0].delta.content = content
                chunk.choices[0].delta.tool_calls = None  # 确保tool_calls为None而不是Mock
                chunk.choices[0].index = 0
                chunk.choices[0].finish_reason = finish_reason
                chunk_id = getattr(self, 'index', 0)
                chunk.id = f"async-chunk-{chunk_id}"
                chunk.object = "chat.completion.chunk"
                chunk.created = 1234567890
                chunk.model = "gpt-4o"
                if hasattr(self, 'index'):
                    self.index += 1
                return chunk
            
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk
        
        # 配置模拟客户端
        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create.return_value = MockAsyncStream()
        mock_async_openai_class.return_value = mock_async_client
        
        # 配置结构化输出处理器
        mock_default_handler._parse_with_native.return_value = {"name": "John", "age": 30}
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Generate user data")]
        
        # JSON schema
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        # 测试异步流式结构化输出
        response_generator = await plugin.chat_completion_async(
            "gpt-4o", 
            messages, 
            stream=True,
            response_format={"type": "json_schema", "json_schema": {"schema": json_schema}}
        )
        
        chunks = []
        async for chunk in response_generator:
            chunks.append(chunk)
        
        # 验证结果
        assert len(chunks) >= 1
        # 验证最后一个chunk包含完整的结构化数据
        final_chunk = chunks[-1]
        assert hasattr(final_chunk.choices[0].delta, 'content')

    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    @patch('harborai.api.structured.default_handler')
    def test_streaming_structured_output_error_fallback(self, mock_default_handler, mock_get_logger, mock_openai_class):
        """
        测试流式结构化输出错误回退场景
        
        覆盖未测试的代码行：438-483 (错误处理分支)
        """
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 模拟流式响应
        class MockStream:
            def __iter__(self):
                chunk = Mock()
                chunk.choices = [Mock()]
                chunk.choices[0].delta = Mock()
                chunk.choices[0].delta.content = "Invalid JSON content"
                chunk.choices[0].delta.tool_calls = None  # 确保tool_calls为None而不是Mock
                chunk.choices[0].index = 0
                chunk.choices[0].finish_reason = "stop"
                chunk.id = "chunk-1"
                chunk.object = "chat.completion.chunk"
                chunk.created = 1234567890
                chunk.model = "gpt-4o"
                return iter([chunk])
        
        # 配置模拟客户端
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = MockStream()
        mock_openai_class.return_value = mock_client
        
        # 配置结构化输出处理器抛出异常
        mock_default_handler.parse_streaming_response.side_effect = Exception("Parse error")
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Generate user data")]
        
        # JSON schema
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        # 测试流式结构化输出错误回退
        response_generator = plugin.chat_completion(
            "gpt-4o", 
            messages, 
            stream=True,
            response_format={"type": "json_schema", "json_schema": {"schema": json_schema}}
        )
        
        chunks = list(response_generator)
        
        # 验证回退到正常流式输出
        assert len(chunks) >= 1
        assert chunks[0].choices[0].delta.content == "Invalid JSON content"
        
        # 验证错误日志被记录
        mock_logger.error.assert_called()

    @patch('harborai.core.plugins.openai_plugin.AsyncOpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    @patch('harborai.api.structured.default_handler')
    @pytest.mark.asyncio
    async def test_async_non_streaming_structured_output(self, mock_default_handler, mock_get_logger, mock_async_openai_class):
        """
        测试异步非流式结构化输出处理
        
        覆盖未测试的代码行：391-392 (异步结构化输出处理)
        """
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 模拟OpenAI响应
        mock_response = Mock()
        mock_response.id = "chatcmpl-test"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4o"
        mock_response.choices = [Mock()]
        mock_response.choices[0].index = 0
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = '{"name": "John", "age": 30}'
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.system_fingerprint = "test-fingerprint"
        
        # 配置模拟客户端
        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create.return_value = mock_response
        mock_async_openai_class.return_value = mock_async_client
        
        # 配置结构化输出处理器
        mock_default_handler.handle_structured_output.return_value = Mock()
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="Generate user data")]
        
        # JSON schema
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        # 测试异步非流式结构化输出
        response = await plugin.chat_completion_async(
            "gpt-4o", 
            messages, 
            stream=False,
            response_format={"type": "json_schema", "json_schema": {"schema": json_schema}}
        )
        
        # 验证响应
        assert response is not None
        assert response.id == "chatcmpl-test"
        assert response.model == "gpt-4o"
        
        # 验证结构化输出处理器被调用 - 在异步非流式情况下，调用的是plugin自身的handle_structured_output方法
        # 这个方法内部会调用default_handler，但我们需要mock plugin的方法
        with patch.object(plugin, 'handle_structured_output') as mock_handle_structured:
            mock_handle_structured.return_value = response
            
            # 重新执行测试
            response2 = await plugin.chat_completion_async(
                "gpt-4o", 
                messages, 
                stream=False,
                response_format={"type": "json_schema", "json_schema": {"schema": json_schema}}
            )
            
            # 验证调用
            mock_handle_structured.assert_called_once()

    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    def test_streaming_with_tool_calls(self, mock_get_logger, mock_openai_class):
        """
        测试流式响应中的工具调用处理
        
        覆盖未测试的代码行：542-544 (工具调用处理)
        """
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 模拟流式响应，包含工具调用
        class MockStreamWithToolCalls:
            def __iter__(self):
                # 第一个chunk包含工具调用
                chunk1 = Mock()
                chunk1.choices = [Mock()]
                chunk1.choices[0].delta = Mock()
                chunk1.choices[0].delta.content = None
                chunk1.choices[0].delta.tool_calls = [Mock()]
                chunk1.choices[0].delta.tool_calls[0].id = "call_123"
                chunk1.choices[0].delta.tool_calls[0].type = "function"
                chunk1.choices[0].delta.tool_calls[0].function = Mock()
                chunk1.choices[0].delta.tool_calls[0].function.name = "get_weather"
                chunk1.choices[0].delta.tool_calls[0].function.arguments = '{"location": "Beijing"}'
                chunk1.choices[0].index = 0
                chunk1.choices[0].finish_reason = None
                chunk1.id = "chunk-1"
                chunk1.object = "chat.completion.chunk"
                chunk1.created = 1234567890
                chunk1.model = "gpt-4o"
                
                # 第二个chunk结束
                chunk2 = Mock()
                chunk2.choices = [Mock()]
                chunk2.choices[0].delta = Mock()
                chunk2.choices[0].delta.content = None
                chunk2.choices[0].delta.tool_calls = None
                chunk2.choices[0].index = 0
                chunk2.choices[0].finish_reason = "tool_calls"
                chunk2.id = "chunk-2"
                chunk2.object = "chat.completion.chunk"
                chunk2.created = 1234567890
                chunk2.model = "gpt-4o"
                
                return iter([chunk1, chunk2])
        
        # 配置模拟客户端
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = MockStreamWithToolCalls()
        mock_openai_class.return_value = mock_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="What's the weather in Beijing?")]
        
        # 测试流式响应中的工具调用
        response_generator = plugin.chat_completion(
            "gpt-4o", 
            messages, 
            stream=True
        )
        
        chunks = list(response_generator)
        
        # 验证工具调用被正确处理
        assert len(chunks) >= 1
        
        # 检查第一个chunk是否包含工具调用
        first_chunk = chunks[0]
        assert first_chunk.choices[0].delta.tool_calls is not None
        assert len(first_chunk.choices[0].delta.tool_calls) == 1
        assert first_chunk.choices[0].delta.tool_calls[0]["id"] == "call_123"
        assert first_chunk.choices[0].delta.tool_calls[0]["type"] == "function"
        assert first_chunk.choices[0].delta.tool_calls[0]["function"]["name"] == "get_weather"

    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    def test_supported_models_property(self, mock_get_logger, mock_openai_class):
        """
        测试supported_models属性
        
        覆盖未测试的代码行：112 (supported_models属性)
        """
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 获取支持的模型列表
        models = plugin.supported_models
        
        # 验证模型列表
        assert len(models) > 0
        
        # 验证包含预期的模型
        model_ids = [model.id for model in models]
        assert "gpt-4o" in model_ids
        assert "gpt-4o-mini" in model_ids
        assert "o1-preview" in model_ids
        assert "o1-mini" in model_ids
        
        # 验证模型属性
        gpt4o_model = next(model for model in models if model.id == "gpt-4o")
        assert gpt4o_model.provider == "openai"
        assert gpt4o_model.supports_structured_output == True
        assert gpt4o_model.supports_thinking == False
        assert gpt4o_model.context_window == 128000


class TestOpenAIPluginClientManagement:
    """测试OpenAI插件客户端管理"""
    
    def test_close_sync_client(self):
        """测试关闭同步客户端"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建模拟客户端
        mock_client = Mock()
        plugin.client = mock_client
        
        # 测试关闭
        plugin.close()
        
        mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_aclose_async_client(self):
        """测试关闭异步客户端"""
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建模拟异步客户端
        mock_async_client = AsyncMock()
        plugin.async_client = mock_async_client
        
        # 测试异步关闭
        await plugin.aclose()
        
        mock_async_client.aclose.assert_called_once()
    
    def test_close_without_client(self):
        """测试在没有客户端时关闭"""
        plugin = OpenAIPlugin(api_key="test-key")
        plugin.client = None
        
        # 应该不会抛出异常
        plugin.close()
    
    @pytest.mark.asyncio
    async def test_aclose_without_async_client(self):
        """测试在没有异步客户端时关闭"""
        plugin = OpenAIPlugin(api_key="test-key")
        plugin.async_client = None
        
        # 应该不会抛出异常
        await plugin.aclose()


class TestOpenAIPluginEdgeCases:
    """测试边界情况和错误处理路径"""

    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    def test_response_format_edge_cases(self, mock_get_logger, mock_openai_class):
        """
        测试response_format的边界情况处理
        覆盖 openai_plugin.py 第241-245行
        """
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 配置模拟客户端
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="测试消息")]
        
        # 测试1: response_format不是字典 - 会被直接传递
        request_params = plugin._prepare_openai_request(
            "gpt-4o",
            messages,
            response_format="invalid_format"
        )
        assert request_params["response_format"] == "invalid_format"
        
        # 测试2: response_format是字典但没有type字段 - 会被直接传递
        request_params = plugin._prepare_openai_request(
            "gpt-4o",
            messages,
            response_format={"schema": {}}
        )
        assert request_params["response_format"] == {"schema": {}}
        
        # 测试3: response_format有type但不是json_schema - 会被直接传递
        request_params = plugin._prepare_openai_request(
            "gpt-4o",
            messages,
            response_format={"type": "text"}
        )
        assert request_params["response_format"] == {"type": "text"}
        
        # 测试4: 正确的json_schema格式
        response_format = {"type": "json_schema", "json_schema": {"schema": {}}}
        request_params = plugin._prepare_openai_request(
            "gpt-4o",
            messages,
            response_format=response_format
        )
        assert request_params["response_format"] == response_format

    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    def test_convert_response_without_reasoning_content(self, mock_get_logger, mock_openai_class):
        """
        测试没有reasoning_content的响应转换
        覆盖 openai_plugin.py 第275-278行的else分支
        """
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 配置模拟客户端
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建没有reasoning_content的模拟响应
        mock_response = Mock()
        mock_response.id = "chatcmpl-test"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4o"
        mock_response.choices = [Mock()]
        mock_response.choices[0].index = 0
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "这是回答"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        
        # 确保没有reasoning_content属性
        if hasattr(mock_response.choices[0].message, 'reasoning_content'):
            delattr(mock_response.choices[0].message, 'reasoning_content')
        
        # 转换响应
        harbor_response = plugin._convert_to_harbor_response(mock_response)
        
        # 验证响应正常转换，没有reasoning_content
        assert harbor_response.choices[0].message.content == "这是回答"
        assert not hasattr(harbor_response.choices[0].message, 'reasoning_content') or harbor_response.choices[0].message.reasoning_content is None

    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    def test_convert_response_without_system_fingerprint(self, mock_get_logger, mock_openai_class):
        """
        测试没有system_fingerprint的响应转换
        覆盖 openai_plugin.py 第304-307行的else分支
        """
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 配置模拟客户端
        mock_client = Mock()
        mock_openai_class.return_value = mock_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        
        # 创建没有system_fingerprint的模拟响应
        mock_response = Mock()
        mock_response.id = "chatcmpl-test"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4o"
        mock_response.choices = [Mock()]
        mock_response.choices[0].index = 0
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.role = "assistant"
        mock_response.choices[0].message.content = "测试内容"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        
        # 确保没有system_fingerprint属性
        if hasattr(mock_response, 'system_fingerprint'):
            delattr(mock_response, 'system_fingerprint')
        
        # 转换响应
        harbor_response = plugin._convert_to_harbor_response(mock_response)
        
        # 验证响应正常转换，没有system_fingerprint
        assert harbor_response.id == "chatcmpl-test"
        assert not hasattr(harbor_response, 'system_fingerprint') or harbor_response.system_fingerprint is None

    @patch('harborai.core.plugins.openai_plugin.OpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    def test_network_error_handling(self, mock_get_logger, mock_openai_class):
        """
        测试网络错误处理
        """
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 配置模拟客户端抛出OpenAI API错误
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = openai.APIConnectionError(request=Mock())
        mock_openai_class.return_value = mock_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="测试消息")]
        
        # 测试网络错误处理
        with pytest.raises(APIError):
            plugin.chat_completion("gpt-4o", messages)

    @patch('harborai.core.plugins.openai_plugin.AsyncOpenAI')
    @patch('harborai.core.plugins.openai_plugin.get_logger')
    @pytest.mark.asyncio
    async def test_async_network_error_handling(self, mock_get_logger, mock_async_openai_class):
        """
        测试异步网络错误处理
        """
        # 模拟日志记录器
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        # 配置模拟客户端抛出OpenAI API错误
        mock_async_client = AsyncMock()
        mock_async_client.chat.completions.create.side_effect = openai.APIConnectionError(request=Mock())
        mock_async_openai_class.return_value = mock_async_client
        
        plugin = OpenAIPlugin(api_key="test-key")
        messages = [ChatMessage(role="user", content="测试消息")]
        
        # 测试异步网络错误处理
        with pytest.raises(APIError):
            await plugin.chat_completion_async("gpt-4o", messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])