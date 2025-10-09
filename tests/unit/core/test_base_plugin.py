#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_plugin.py 模块的全面测试用例

测试覆盖：
1. 数据类的创建和验证
2. BaseLLMPlugin的基础功能
3. 抽象方法的实现验证
4. 请求验证和准备
5. 结构化输出处理
6. 错误处理和日志记录
7. 边界条件和异常情况

遵循VIBE编码规范：
- TDD测试驱动开发
- 详细的中文注释
- 覆盖正常、异常、边界条件
- 测试独立性和可重复性
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Generator

from harborai.core.base_plugin import (
    ModelInfo,
    ChatMessage,
    ChatChoiceDelta,
    ChatChoice,
    Usage,
    ChatCompletion,
    ChatCompletionChunk,
    BaseLLMPlugin
)
from harborai.utils.exceptions import PluginError, ModelNotFoundError


class TestModelInfo:
    """测试ModelInfo数据类
    
    功能：验证模型信息数据类的创建、属性访问和不可变性
    参数：模型ID、名称、提供商等基础信息
    返回：ModelInfo实例
    异常：无（数据类创建不会抛出异常）
    边界：测试必需字段和可选字段的组合
    假设：frozen=True确保不可变性
    """
    
    def test_model_info_creation_with_required_fields(self):
        """测试使用必需字段创建ModelInfo"""
        model_info = ModelInfo(
            id="gpt-4",
            name="GPT-4",
            provider="openai"
        )
        
        assert model_info.id == "gpt-4"
        assert model_info.name == "GPT-4"
        assert model_info.provider == "openai"
        assert model_info.supports_streaming is True  # 默认值
        assert model_info.supports_structured_output is False  # 默认值
        assert model_info.supports_thinking is False  # 默认值
        assert model_info.max_tokens is None  # 默认值
        assert model_info.context_window is None  # 默认值
        assert model_info.description is None  # 默认值
    
    def test_model_info_creation_with_all_fields(self):
        """测试使用所有字段创建ModelInfo"""
        model_info = ModelInfo(
            id="gpt-4-turbo",
            name="GPT-4 Turbo",
            provider="openai",
            supports_streaming=True,
            supports_structured_output=True,
            supports_thinking=False,
            max_tokens=4096,
            context_window=128000,
            description="Advanced GPT-4 model with turbo speed"
        )
        
        assert model_info.id == "gpt-4-turbo"
        assert model_info.name == "GPT-4 Turbo"
        assert model_info.provider == "openai"
        assert model_info.supports_streaming is True
        assert model_info.supports_structured_output is True
        assert model_info.supports_thinking is False
        assert model_info.max_tokens == 4096
        assert model_info.context_window == 128000
        assert model_info.description == "Advanced GPT-4 model with turbo speed"
    
    def test_model_info_immutability(self):
        """测试ModelInfo的不可变性（frozen=True）"""
        model_info = ModelInfo(
            id="test-model",
            name="Test Model",
            provider="test"
        )
        
        # 尝试修改字段应该抛出异常
        with pytest.raises(AttributeError):
            model_info.id = "modified-id"
        
        with pytest.raises(AttributeError):
            model_info.supports_streaming = False


class TestChatMessage:
    """测试ChatMessage数据类
    
    功能：验证聊天消息数据类的创建和属性访问
    参数：角色、内容、可选的工具调用等
    返回：ChatMessage实例
    异常：无（数据类创建不会抛出异常）
    边界：测试必需字段和各种可选字段组合
    假设：支持多种消息类型（用户、助手、系统、工具）
    """
    
    def test_chat_message_basic_creation(self):
        """测试基础ChatMessage创建"""
        message = ChatMessage(
            role="user",
            content="Hello, world!"
        )
        
        assert message.role == "user"
        assert message.content == "Hello, world!"
        assert message.name is None
        assert message.function_call is None
        assert message.tool_calls is None
        assert message.tool_call_id is None
        assert message.reasoning_content is None
        assert message.parsed is None
    
    def test_chat_message_with_all_fields(self):
        """测试包含所有字段的ChatMessage创建"""
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test"}}]
        function_call = {"name": "test_function", "arguments": "{}"}
        
        message = ChatMessage(
            role="assistant",
            content="I'll help you with that.",
            name="assistant_bot",
            function_call=function_call,
            tool_calls=tool_calls,
            tool_call_id="call_123",
            reasoning_content="Let me think about this...",
            parsed={"result": "success"}
        )
        
        assert message.role == "assistant"
        assert message.content == "I'll help you with that."
        assert message.name == "assistant_bot"
        assert message.function_call == function_call
        assert message.tool_calls == tool_calls
        assert message.tool_call_id == "call_123"
        assert message.reasoning_content == "Let me think about this..."
        assert message.parsed == {"result": "success"}
    
    def test_chat_message_different_roles(self):
        """测试不同角色的ChatMessage"""
        roles = ["user", "assistant", "system", "tool"]
        
        for role in roles:
            message = ChatMessage(role=role, content=f"Content for {role}")
            assert message.role == role
            assert message.content == f"Content for {role}"


class TestChatChoiceDelta:
    """测试ChatChoiceDelta数据类
    
    功能：验证流式响应增量数据的创建
    参数：可选的角色、内容、工具调用等
    返回：ChatChoiceDelta实例
    异常：无
    边界：测试所有字段都为可选的情况
    假设：用于流式响应的增量更新
    """
    
    def test_chat_choice_delta_empty(self):
        """测试空的ChatChoiceDelta创建"""
        delta = ChatChoiceDelta()
        
        assert delta.role is None
        assert delta.content is None
        assert delta.tool_calls is None
        assert delta.reasoning_content is None
    
    def test_chat_choice_delta_with_content(self):
        """测试包含内容的ChatChoiceDelta"""
        delta = ChatChoiceDelta(
            role="assistant",
            content="Hello",
            reasoning_content="Thinking..."
        )
        
        assert delta.role == "assistant"
        assert delta.content == "Hello"
        assert delta.reasoning_content == "Thinking..."
    
    def test_chat_choice_delta_with_tool_calls(self):
        """测试包含工具调用的ChatChoiceDelta"""
        tool_calls = [{"id": "call_1", "type": "function"}]
        delta = ChatChoiceDelta(tool_calls=tool_calls)
        
        assert delta.tool_calls == tool_calls


class TestChatChoice:
    """测试ChatChoice数据类
    
    功能：验证聊天选择数据的创建
    参数：索引、消息或增量、完成原因
    返回：ChatChoice实例
    异常：无
    边界：测试消息和增量的互斥性
    假设：用于表示模型的响应选择
    """
    
    def test_chat_choice_with_message(self):
        """测试包含完整消息的ChatChoice"""
        message = ChatMessage(role="assistant", content="Hello")
        choice = ChatChoice(
            index=0,
            message=message,
            finish_reason="stop"
        )
        
        assert choice.index == 0
        assert choice.message == message
        assert choice.delta is None
        assert choice.finish_reason == "stop"
    
    def test_chat_choice_with_delta(self):
        """测试包含增量的ChatChoice（流式响应）"""
        delta = ChatChoiceDelta(content="Hello")
        choice = ChatChoice(
            index=0,
            delta=delta,
            finish_reason=None
        )
        
        assert choice.index == 0
        assert choice.message is None
        assert choice.delta == delta
        assert choice.finish_reason is None


class TestUsage:
    """测试Usage数据类
    
    功能：验证使用统计数据的创建
    参数：提示词令牌数、完成令牌数、总令牌数
    返回：Usage实例
    异常：无
    边界：测试零值和大数值
    假设：用于跟踪API使用情况
    """
    
    def test_usage_creation(self):
        """测试Usage数据类创建"""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
    
    def test_usage_zero_values(self):
        """测试零值Usage"""
        usage = Usage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
        
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0


class TestChatCompletion:
    """测试ChatCompletion数据类
    
    功能：验证聊天完成响应数据的创建
    参数：ID、对象类型、创建时间、模型、选择列表等
    返回：ChatCompletion实例
    异常：无
    边界：测试必需字段和可选字段组合
    假设：表示完整的API响应
    """
    
    def test_chat_completion_basic(self):
        """测试基础ChatCompletion创建"""
        choice = ChatChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hello"),
            finish_reason="stop"
        )
        
        completion = ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[choice]
        )
        
        assert completion.id == "chatcmpl-123"
        assert completion.object == "chat.completion"
        assert completion.created == 1234567890
        assert completion.model == "gpt-4"
        assert len(completion.choices) == 1
        assert completion.choices[0] == choice
        assert completion.usage is None
        assert completion.system_fingerprint is None
        assert completion.parsed is None
    
    def test_chat_completion_with_usage(self):
        """测试包含使用统计的ChatCompletion"""
        choice = ChatChoice(
            index=0,
            message=ChatMessage(role="assistant", content="Hello"),
            finish_reason="stop"
        )
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        
        completion = ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-4",
            choices=[choice],
            usage=usage,
            system_fingerprint="fp_123",
            parsed={"result": "success"}
        )
        
        assert completion.usage == usage
        assert completion.system_fingerprint == "fp_123"
        assert completion.parsed == {"result": "success"}


class TestChatCompletionChunk:
    """测试ChatCompletionChunk数据类
    
    功能：验证流式响应块数据的创建
    参数：ID、对象类型、创建时间、模型、选择列表
    返回：ChatCompletionChunk实例
    异常：无
    边界：测试流式响应的各种状态
    假设：表示流式响应的单个块
    """
    
    def test_chat_completion_chunk_creation(self):
        """测试ChatCompletionChunk创建"""
        choice = ChatChoice(
            index=0,
            delta=ChatChoiceDelta(content="Hello"),
            finish_reason=None
        )
        
        chunk = ChatCompletionChunk(
            id="chatcmpl-123",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4",
            choices=[choice]
        )
        
        assert chunk.id == "chatcmpl-123"
        assert chunk.object == "chat.completion.chunk"
        assert chunk.created == 1234567890
        assert chunk.model == "gpt-4"
        assert len(chunk.choices) == 1
        assert chunk.choices[0] == choice
        assert chunk.system_fingerprint is None


# 创建测试用的具体插件实现
class TestLLMPlugin(BaseLLMPlugin):
    """测试用的具体LLM插件实现
    
    功能：提供BaseLLMPlugin的具体实现用于测试
    参数：继承自BaseLLMPlugin的所有参数
    返回：TestLLMPlugin实例
    异常：根据测试需要可能抛出各种异常
    边界：支持各种测试场景
    假设：仅用于测试目的
    """
    
    def __init__(self, name: str = "test", **config):
        super().__init__(name, **config)
        self._supported_models = [
            ModelInfo(
                id="test-model",
                name="Test Model",
                provider="test",
                supports_streaming=True,
                supports_structured_output=True,
                max_tokens=4096
            ),
            ModelInfo(
                id="test-model-2",
                name="Test Model 2",
                provider="test",
                supports_streaming=False,
                supports_structured_output=False
            )
        ]
    
    def chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """同步聊天完成实现"""
        if stream:
            return self._generate_stream_response(model, messages, **kwargs)
        else:
            return self._generate_completion_response(model, messages, **kwargs)
    
    async def chat_completion_async(
        self,
        model: str,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """异步聊天完成实现"""
        if stream:
            return self._generate_async_stream_response(model, messages, **kwargs)
        else:
            return self._generate_completion_response(model, messages, **kwargs)
    
    def _generate_completion_response(self, model: str, messages: List[ChatMessage], **kwargs):
        """生成完成响应"""
        return ChatCompletion(
            id="test-completion-123",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="Test response",
                        reasoning_content="Test reasoning" if kwargs.get("include_reasoning") else None
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )
    
    def _generate_stream_response(self, model: str, messages: List[ChatMessage], **kwargs):
        """生成流式响应"""
        chunks = [
            ChatCompletionChunk(
                id="test-chunk-123",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[
                    ChatChoice(
                        index=0,
                        delta=ChatChoiceDelta(role="assistant"),
                        finish_reason=None
                    )
                ]
            ),
            ChatCompletionChunk(
                id="test-chunk-123",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[
                    ChatChoice(
                        index=0,
                        delta=ChatChoiceDelta(content="Test"),
                        finish_reason=None
                    )
                ]
            ),
            ChatCompletionChunk(
                id="test-chunk-123",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[
                    ChatChoice(
                        index=0,
                        delta=ChatChoiceDelta(content=" response"),
                        finish_reason="stop"
                    )
                ]
            )
        ]
        
        for chunk in chunks:
            yield chunk
    
    async def _generate_async_stream_response(self, model: str, messages: List[ChatMessage], **kwargs):
        """生成异步流式响应"""
        for chunk in self._generate_stream_response(model, messages, **kwargs):
            yield chunk


class TestBaseLLMPlugin:
    """测试BaseLLMPlugin基类
    
    功能：验证LLM插件基类的所有功能
    参数：插件名称、配置等
    返回：根据方法不同返回不同类型
    异常：PluginError、ModelNotFoundError等
    边界：测试各种边界条件和错误情况
    假设：作为所有LLM插件的基类
    """
    
    @pytest.fixture
    def plugin(self):
        """创建测试插件实例"""
        return TestLLMPlugin("test-plugin", api_key="test-key", base_url="https://test.com")
    
    @pytest.fixture
    def sample_messages(self):
        """创建示例消息列表"""
        return [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello, how are you?")
        ]
    
    def test_plugin_initialization(self, plugin):
        """测试插件初始化"""
        assert plugin.name == "test-plugin"
        assert plugin.config["api_key"] == "test-key"
        assert plugin.config["base_url"] == "https://test.com"
        assert len(plugin.supported_models) == 2
        assert plugin.logger is not None
    
    def test_supported_models_property(self, plugin):
        """测试支持的模型列表属性"""
        models = plugin.supported_models
        assert len(models) == 2
        assert models[0].id == "test-model"
        assert models[1].id == "test-model-2"
    
    def test_supports_model_true(self, plugin):
        """测试支持的模型检查（正面情况）"""
        assert plugin.supports_model("test-model") is True
        assert plugin.supports_model("test-model-2") is True
    
    def test_supports_model_false(self, plugin):
        """测试不支持的模型检查（负面情况）"""
        assert plugin.supports_model("unsupported-model") is False
        assert plugin.supports_model("") is False
        assert plugin.supports_model("gpt-4") is False
    
    def test_get_model_info_existing(self, plugin):
        """测试获取存在的模型信息"""
        model_info = plugin.get_model_info("test-model")
        assert model_info is not None
        assert model_info.id == "test-model"
        assert model_info.name == "Test Model"
        assert model_info.provider == "test"
        assert model_info.supports_streaming is True
        assert model_info.supports_structured_output is True
    
    def test_get_model_info_nonexistent(self, plugin):
        """测试获取不存在的模型信息"""
        model_info = plugin.get_model_info("nonexistent-model")
        assert model_info is None
    
    def test_extract_reasoning_content_completion(self, plugin):
        """测试从ChatCompletion提取推理内容"""
        # 测试包含推理内容的响应
        response_with_reasoning = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="Response",
                        reasoning_content="This is my reasoning"
                    ),
                    finish_reason="stop"
                )
            ]
        )
        
        reasoning = plugin.extract_reasoning_content(response_with_reasoning)
        assert reasoning == "This is my reasoning"
        
        # 测试不包含推理内容的响应
        response_without_reasoning = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Response"),
                    finish_reason="stop"
                )
            ]
        )
        
        reasoning = plugin.extract_reasoning_content(response_without_reasoning)
        assert reasoning is None
    
    def test_extract_reasoning_content_chunk(self, plugin):
        """测试从ChatCompletionChunk提取推理内容"""
        # 测试包含推理内容的块
        chunk_with_reasoning = ChatCompletionChunk(
            id="test-123",
            object="chat.completion.chunk",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    delta=ChatChoiceDelta(
                        content="Response",
                        reasoning_content="Chunk reasoning"
                    ),
                    finish_reason=None
                )
            ]
        )
        
        reasoning = plugin.extract_reasoning_content(chunk_with_reasoning)
        assert reasoning == "Chunk reasoning"
        
        # 测试不包含推理内容的块
        chunk_without_reasoning = ChatCompletionChunk(
            id="test-123",
            object="chat.completion.chunk",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    delta=ChatChoiceDelta(content="Response"),
                    finish_reason=None
                )
            ]
        )
        
        reasoning = plugin.extract_reasoning_content(chunk_without_reasoning)
        assert reasoning is None
    
    def test_extract_reasoning_content_empty_choices(self, plugin):
        """测试从空选择列表提取推理内容"""
        response_empty_choices = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[]
        )
        
        reasoning = plugin.extract_reasoning_content(response_empty_choices)
        assert reasoning is None
    
    def test_validate_request_success(self, plugin, sample_messages):
        """测试请求验证成功情况"""
        # 正常情况下不应抛出异常
        plugin.validate_request("test-model", sample_messages)
    
    def test_validate_request_empty_messages(self, plugin):
        """测试空消息列表验证"""
        with pytest.raises(PluginError) as exc_info:
            plugin.validate_request("test-model", [])
        
        assert "Messages cannot be empty" in str(exc_info.value)
    
    def test_validate_request_unsupported_model(self, plugin, sample_messages):
        """测试不支持的模型验证"""
        with pytest.raises(ModelNotFoundError):
            plugin.validate_request("unsupported-model", sample_messages)
    
    def test_validate_request_invalid_message_type(self, plugin):
        """测试无效消息类型验证"""
        invalid_messages = [{"role": "user", "content": "Hello"}]  # 字典而非ChatMessage
        
        with pytest.raises(PluginError) as exc_info:
            plugin.validate_request("test-model", invalid_messages)
        
        assert "must be a ChatMessage instance" in str(exc_info.value)
    
    def test_validate_request_missing_role(self, plugin):
        """测试缺少角色字段验证"""
        invalid_message = ChatMessage(role="", content="Hello")
        
        with pytest.raises(PluginError) as exc_info:
            plugin.validate_request("test-model", [invalid_message])
        
        assert "missing 'role' field" in str(exc_info.value)
    
    def test_validate_request_missing_content(self, plugin):
        """测试缺少内容字段验证"""
        invalid_message = ChatMessage(role="user", content="")
        
        with pytest.raises(PluginError) as exc_info:
            plugin.validate_request("test-model", [invalid_message])
        
        assert "missing 'content' field" in str(exc_info.value)
    
    def test_prepare_request_basic(self, plugin, sample_messages):
        """测试基础请求准备"""
        request_data = plugin.prepare_request("test-model", sample_messages)
        
        assert request_data["model"] == "test-model"
        assert len(request_data["messages"]) == 2
        assert request_data["messages"][0]["role"] == "system"
        assert request_data["messages"][0]["content"] == "You are a helpful assistant."
        assert request_data["messages"][1]["role"] == "user"
        assert request_data["messages"][1]["content"] == "Hello, how are you?"
    
    def test_prepare_request_with_kwargs(self, plugin, sample_messages):
        """测试包含额外参数的请求准备"""
        request_data = plugin.prepare_request(
            "test-model",
            sample_messages,
            temperature=0.7,
            max_tokens=100,
            stream=True,
            none_value=None  # 应该被过滤掉
        )
        
        assert request_data["model"] == "test-model"
        assert request_data["temperature"] == 0.7
        assert request_data["max_tokens"] == 100
        assert request_data["stream"] is True
        assert "none_value" not in request_data  # None值应该被过滤
    
    def test_prepare_request_with_complex_message(self, plugin):
        """测试包含复杂字段的消息准备"""
        complex_message = ChatMessage(
            role="assistant",
            content="I'll help you.",
            name="assistant_bot",
            function_call={"name": "test_func", "arguments": "{}"},
            tool_calls=[{"id": "call_1", "type": "function"}]
        )
        
        request_data = plugin.prepare_request("test-model", [complex_message])
        
        message_dict = request_data["messages"][0]
        assert message_dict["role"] == "assistant"
        assert message_dict["content"] == "I'll help you."
        assert message_dict["name"] == "assistant_bot"
        assert message_dict["function_call"] == {"name": "test_func", "arguments": "{}"}
        assert message_dict["tool_calls"] == [{"id": "call_1", "type": "function"}]
    
    @patch('harborai.core.base_plugin.get_current_trace_id')
    def test_handle_structured_output_no_format(self, mock_trace_id, plugin):
        """测试无结构化格式的处理"""
        mock_trace_id.return_value = "test-trace-123"
        
        response = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Hello"),
                    finish_reason="stop"
                )
            ]
        )
        
        # 无response_format
        result = plugin.handle_structured_output(response)
        assert result == response
        assert result.choices[0].message.parsed is None
        
        # response_format类型不是json_schema
        result = plugin.handle_structured_output(
            response,
            response_format={"type": "text"}
        )
        assert result == response
        assert result.choices[0].message.parsed is None
    
    @patch('harborai.core.base_plugin.get_current_trace_id')
    def test_handle_structured_output_with_agently(self, mock_trace_id, plugin):
        """测试使用Agently的结构化输出处理"""
        mock_trace_id.return_value = "test-trace-123"
        
        response = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content='{"name": "John", "age": 30}'),
                    finish_reason="stop"
                )
            ]
        )
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        }
        
        original_messages = [
            ChatMessage(role="user", content="Generate a person object")
        ]
        
        # Mock Agently解析
        with patch.object(plugin, '_parse_with_agently') as mock_agently:
            mock_agently.return_value = {"name": "John", "age": 30}
            
            result = plugin.handle_structured_output(
                response,
                response_format=response_format,
                structured_provider="agently",
                model="test-model",
                original_messages=original_messages
            )
            
            assert result.choices[0].message.parsed == {"name": "John", "age": 30}
            mock_agently.assert_called_once()
    
    @patch('harborai.core.base_plugin.get_current_trace_id')
    def test_handle_structured_output_with_native(self, mock_trace_id, plugin):
        """测试使用原生方式的结构化输出处理"""
        mock_trace_id.return_value = "test-trace-123"
        
        response = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content='{"name": "John", "age": 30}'),
                    finish_reason="stop"
                )
            ]
        )
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        }
        
        # Mock原生解析
        with patch.object(plugin, '_parse_with_native') as mock_native:
            mock_native.return_value = {"name": "John", "age": 30}
            
            result = plugin.handle_structured_output(
                response,
                response_format=response_format,
                structured_provider="native"
            )
            
            assert result.choices[0].message.parsed == {"name": "John", "age": 30}
            mock_native.assert_called_once()
    
    @patch('harborai.core.base_plugin.get_current_trace_id')
    def test_handle_structured_output_error_handling(self, mock_trace_id, plugin):
        """测试结构化输出错误处理"""
        mock_trace_id.return_value = "test-trace-123"
        
        response = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Invalid JSON"),
                    finish_reason="stop"
                )
            ]
        )
        
        response_format = {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}}
        }
        
        # Mock解析失败
        with patch.object(plugin, '_parse_with_native') as mock_native:
            mock_native.side_effect = Exception("Parse error")
            
            result = plugin.handle_structured_output(
                response,
                response_format=response_format
            )
            
            # 错误情况下应该返回原始响应，parsed为None
            assert result == response
            assert result.choices[0].message.parsed is None
    
    def test_create_error_response(self, plugin):
        """测试创建错误响应"""
        error = Exception("Test error")
        
        error_response = plugin.create_error_response(
            error,
            "test-model",
            "error-123"
        )
        
        assert error_response.id == "error-123"
        assert error_response.object == "chat.completion"
        assert error_response.model == "test-model"
        assert len(error_response.choices) == 1
        assert error_response.choices[0].message.role == "assistant"
        assert "Error: Test error" in error_response.choices[0].message.content
        assert error_response.choices[0].finish_reason == "error"
        assert error_response.usage.prompt_tokens == 0
        assert error_response.usage.completion_tokens == 0
        assert error_response.usage.total_tokens == 0
    
    def test_create_error_response_auto_id(self, plugin):
        """测试自动生成ID的错误响应"""
        error = Exception("Test error")
        
        error_response = plugin.create_error_response(error, "test-model")
        
        assert error_response.id.startswith("error_")
        assert error_response.model == "test-model"
    
    @patch('harborai.core.base_plugin.get_current_trace_id')
    def test_log_request(self, mock_trace_id, plugin, sample_messages):
        """测试请求日志记录"""
        mock_trace_id.return_value = "test-trace-123"
        
        with patch.object(plugin.logger, 'info') as mock_log:
            plugin.log_request(
                "test-model",
                sample_messages,
                stream=True,
                response_format={"type": "json_schema"}
            )
            
            mock_log.assert_called_once_with(
                "Plugin request started",
                extra={
                    "trace_id": "test-trace-123",
                    "plugin": "test-plugin",
                    "model": "test-model",
                    "message_count": 2,
                    "stream": True,
                    "structured_output": True
                }
            )
    
    @patch('harborai.core.base_plugin.get_current_trace_id')
    def test_log_response(self, mock_trace_id, plugin):
        """测试响应日志记录"""
        mock_trace_id.return_value = "test-trace-123"
        
        response = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="Response",
                        reasoning_content="Reasoning"
                    ),
                    finish_reason="stop"
                )
            ]
        )
        
        with patch.object(plugin.logger, 'info') as mock_log:
            plugin.log_response(response, 150.5)
            
            mock_log.assert_called_once_with(
                "Plugin request completed",
                extra={
                    "trace_id": "test-trace-123",
                    "plugin": "test-plugin",
                    "latency_ms": 150.5,
                    "reasoning_content_present": True,
                    "response_type": "ChatCompletion"
                }
            )
    
    def test_chat_completion_sync(self, plugin, sample_messages):
        """测试同步聊天完成"""
        response = plugin.chat_completion("test-model", sample_messages)
        
        assert isinstance(response, ChatCompletion)
        assert response.model == "test-model"
        assert response.choices[0].message.content == "Test response"
    
    def test_chat_completion_sync_stream(self, plugin, sample_messages):
        """测试同步流式聊天完成"""
        response_generator = plugin.chat_completion("test-model", sample_messages, stream=True)
        
        chunks = list(response_generator)
        assert len(chunks) == 3
        assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)
        assert chunks[0].choices[0].delta.role == "assistant"
        assert chunks[1].choices[0].delta.content == "Test"
        assert chunks[2].choices[0].delta.content == " response"
    
    @pytest.mark.asyncio
    async def test_chat_completion_async(self, plugin, sample_messages):
        """测试异步聊天完成"""
        response = await plugin.chat_completion_async("test-model", sample_messages)
        
        assert isinstance(response, ChatCompletion)
        assert response.model == "test-model"
        assert response.choices[0].message.content == "Test response"
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_stream(self, plugin, sample_messages):
        """测试异步流式聊天完成"""
        response_generator = await plugin.chat_completion_async("test-model", sample_messages, stream=True)
        
        chunks = []
        async for chunk in response_generator:
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert all(isinstance(chunk, ChatCompletionChunk) for chunk in chunks)


class TestBaseLLMPluginEdgeCases:
    """测试BaseLLMPlugin的边界情况和异常处理
    
    功能：验证各种边界条件和异常情况的处理
    参数：各种异常输入和边界值
    返回：根据情况返回错误或异常
    异常：各种预期的异常类型
    边界：空值、None、异常大小等
    假设：插件应该优雅地处理所有边界情况
    """
    
    @pytest.fixture
    def plugin(self):
        """创建测试插件实例"""
        return TestLLMPlugin("edge-test")
    
    def test_extract_reasoning_content_with_exception(self, plugin):
        """测试提取推理内容时的异常处理"""
        # 创建一个会导致异常的ChatCompletion响应对象
        from harborai.core.base_plugin import ChatCompletion, ChatChoice, ChatMessage
        
        class BadMessage:
            @property
            def reasoning_content(self):
                raise Exception("Simulated error")
        
        class BadChoice:
            def __init__(self):
                self.message = BadMessage()
        
        # 创建一个看起来正常但会在访问reasoning_content时出错的响应
        bad_response = ChatCompletion(
            id="test",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[BadChoice()]
        )
        
        with patch.object(plugin.logger, 'warning') as mock_warning, \
             patch('harborai.core.base_plugin.get_current_trace_id') as mock_trace:
            mock_trace.return_value = "test-trace-123"
            result = plugin.extract_reasoning_content(bad_response)
            assert result is None
            # 验证warning被调用，参数包含消息和关键字参数
            mock_warning.assert_called_once_with(
                "Failed to extract reasoning content",
                trace_id="test-trace-123",
                error="Simulated error"
            )
    
    def test_validate_request_with_none_values(self, plugin):
        """测试包含None值的请求验证"""
        # 测试None消息列表
        with pytest.raises(PluginError):
            plugin.validate_request("test-model", None)
        
        # 测试包含None的消息
        messages_with_none = [
            ChatMessage(role="user", content="Hello"),
            None
        ]
        
        with pytest.raises(PluginError):
            plugin.validate_request("test-model", messages_with_none)
    
    def test_parse_with_agently_import_error(self, plugin):
        """测试Agently库不可用时的处理"""
        response_format = {
            "json_schema": {
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        }
        
        with patch('harborai.core.base_plugin.get_current_trace_id') as mock_trace, \
             patch.object(plugin.logger, 'warning') as mock_warning:
            mock_trace.return_value = "test-trace-123"
            
            # 模拟ImportError
            with patch('builtins.__import__', side_effect=ImportError("No module named 'agently'")):
                with pytest.raises(Exception, match="Agently library not available"):
                    plugin._parse_with_agently("test input", response_format)
                
                mock_warning.assert_called_once()
    
    def test_parse_with_agently_api_error(self, plugin):
        """测试Agently API错误处理"""
        response_format = {
            "json_schema": {
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        }
        
        with patch('harborai.core.base_plugin.get_current_trace_id') as mock_trace, \
             patch.object(plugin.logger, 'error') as mock_error:
            mock_trace.return_value = "test-trace-123"
            
            # 模拟API错误
            mock_handler = Mock()
            mock_handler._parse_with_agently.side_effect = Exception("API key error")
            
            with patch('harborai.api.structured.default_handler', mock_handler):
                with pytest.raises(Exception, match="API key error"):
                    plugin._parse_with_agently("test input", response_format)
                
                mock_error.assert_called_once()
    
    def test_parse_with_native_success(self, plugin):
        """测试原生解析成功"""
        response_format = {
            "json_schema": {
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        }
        content = '{"name": "test"}'
        
        mock_handler = Mock()
        mock_handler._parse_with_native.return_value = {"name": "test"}
        
        with patch('harborai.api.structured.default_handler', mock_handler):
            result = plugin._parse_with_native(content, response_format)
            assert result == {"name": "test"}
    
    def test_parse_with_native_fallback_to_json(self, plugin):
        """测试原生解析失败时回退到JSON解析"""
        response_format = {
            "json_schema": {
                "schema": {"type": "object", "properties": {"name": {"type": "string"}}}
            }
        }
        content = '{"name": "test"}'
        
        with patch('harborai.core.base_plugin.get_current_trace_id') as mock_trace, \
             patch.object(plugin.logger, 'error') as mock_error:
            mock_trace.return_value = "test-trace-123"
            
            # 模拟原生解析失败
            mock_handler = Mock()
            mock_handler._parse_with_native.side_effect = Exception("Parse error")
            
            with patch('harborai.api.structured.default_handler', mock_handler):
                result = plugin._parse_with_native(content, response_format)
                assert result == {"name": "test"}  # JSON解析成功
                
                mock_error.assert_called_once()
    
    def test_prepare_request_empty_messages(self, plugin):
        """测试准备空消息列表的请求"""
        request_data = plugin.prepare_request("test-model", [])
        
        assert request_data["model"] == "test-model"
        assert request_data["messages"] == []
    
    def test_handle_structured_output_no_choices(self, plugin):
        """测试处理没有选择的结构化输出"""
        response = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[]
        )
        
        response_format = {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}}
        }
        
        # 应该不会抛出异常，返回原始响应
        result = plugin.handle_structured_output(response, response_format)
        assert result == response
    
    def test_handle_structured_output_no_message(self, plugin):
        """测试处理没有消息的结构化输出"""
        response = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=None,  # 没有消息
                    finish_reason="stop"
                )
            ]
        )
        
        response_format = {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object"}}
        }
        
        # 应该不会抛出异常
        result = plugin.handle_structured_output(response, response_format)
        assert result == response
    
    def test_plugin_with_empty_config(self):
        """测试使用空配置创建插件"""
        plugin = TestLLMPlugin("empty-config")
        
        assert plugin.name == "empty-config"
        assert plugin.config == {}
        assert len(plugin.supported_models) == 2
    
    def test_plugin_with_none_config(self):
        """测试使用None配置创建插件"""
        plugin = TestLLMPlugin("none-config", **{})
        
        assert plugin.name == "none-config"
        assert plugin.config == {}
    
    def test_supports_model_with_special_characters(self, plugin):
        """测试包含特殊字符的模型ID"""
        # 添加包含特殊字符的模型
        special_model = ModelInfo(
            id="test-model@v1.0",
            name="Test Model v1.0",
            provider="test"
        )
        plugin._supported_models.append(special_model)
        
        assert plugin.supports_model("test-model@v1.0") is True
        assert plugin.supports_model("test-model@v2.0") is False
    
    def test_get_model_info_case_sensitivity(self, plugin):
        """测试模型信息获取的大小写敏感性"""
        # 应该区分大小写
        assert plugin.get_model_info("test-model") is not None
        assert plugin.get_model_info("Test-Model") is None
        assert plugin.get_model_info("TEST-MODEL") is None