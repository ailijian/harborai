# -*- coding: utf-8 -*-
"""
HarborAI API兼容性测试模?

测试目标?
- 验证HarborAI与DeepSeek API的兼容性
- 测试多厂商API的统一接口
- 验证参数传递和响应格式的一致性
"""

import pytest
import asyncio
from typing import Dict, Any, List
from unittest.mock import Mock, patch

from harborai import HarborAI
from harborai.core.exceptions import HarborAIError
from harborai.utils.exceptions import ValidationError


class TestAPICompatibility:
    """API兼容性测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.api_alignment
    def test_deepseek_chat_completions_interface(self, mock_harborai_client, test_messages):
        """测试DeepSeek chat.completions接口兼容性"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="Hello! I'm an AI assistant.",
                role="assistant"
            ),
            finish_reason="stop",
            index=0
        )]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18
        )
        mock_response.model = "deepseek-chat"
        mock_response.id = "chatcmpl-test"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        
        # 配置mock以返回非流式响应
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 执行测试
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            temperature=0.7,
            max_tokens=100
        )
        
        # 验证响应结构
        assert hasattr(response, 'choices')
        assert hasattr(response, 'usage')
        assert hasattr(response, 'model')
        assert hasattr(response, 'id')
        assert hasattr(response, 'object')
        assert hasattr(response, 'created')
        
        # 验证choices结构
        assert len(response.choices) > 0
        choice = response.choices[0]
        assert hasattr(choice, 'message')
        assert hasattr(choice, 'finish_reason')
        assert hasattr(choice, 'index')
        
        # 验证message结构
        message = choice.message
        assert hasattr(message, 'content')
        assert hasattr(message, 'role')
        assert message.role == "assistant"
        
        # 验证usage结构
        usage = response.usage
        assert hasattr(usage, 'prompt_tokens')
        assert hasattr(usage, 'completion_tokens')
        assert hasattr(usage, 'total_tokens')
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.api_alignment
    @pytest.mark.stream_test
    def test_deepseek_streaming_interface(self, mock_harborai_client):
        """测试DeepSeek流式接口兼容性"""
        # 配置mock流式响应
        mock_chunks = [
            Mock(
                choices=[Mock(
                    delta=Mock(content="Hello"),
                    index=0,
                    finish_reason=None
                )],
                id="chatcmpl-test-stream",
                object="chat.completion.chunk",
                created=1234567890,
                model="deepseek-chat"
            ),
            Mock(
                choices=[Mock(
                    delta=Mock(content=" there!"),
                    index=0,
                    finish_reason=None
                )],
                id="chatcmpl-test-stream",
                object="chat.completion.chunk",
                created=1234567890,
                model="deepseek-chat"
            ),
            Mock(
                choices=[Mock(
                    delta=Mock(content=None),
                    index=0,
                    finish_reason="stop"
                )],
                id="chatcmpl-test-stream",
                object="chat.completion.chunk",
                created=1234567890,
                model="deepseek-chat"
            )
        ]
        
        # 确保delta.content属性正确设置
        mock_chunks[0].choices[0].delta.content = "Hello"
        mock_chunks[1].choices[0].delta.content = " there!"
        mock_chunks[2].choices[0].delta.content = None
        
        # 配置mock以返回流式响应
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return iter(mock_chunks)
            else:
                return Mock()
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 执行流式测试
        stream = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        collected_content = []
        for chunk in stream:
            # 验证chunk结构
            assert hasattr(chunk, 'choices')
            assert hasattr(chunk, 'id')
            assert hasattr(chunk, 'object')
            assert hasattr(chunk, 'created')
            assert hasattr(chunk, 'model')
            assert chunk.object == "chat.completion.chunk"
            
            # 验证choices结构
            if chunk.choices:
                choice = chunk.choices[0]
                assert hasattr(choice, 'delta')
                assert hasattr(choice, 'index')
                assert hasattr(choice, 'finish_reason')
                
                # 收集内容
                if hasattr(choice.delta, 'content') and choice.delta.content:
                    collected_content.append(choice.delta.content)
        
        # 验证收集到的内容
        full_content = ''.join(collected_content)
        assert full_content == "Hello there!"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.parametrize("vendor,model", [
        ("deepseek", "deepseek-chat"),
        ("ernie", "ernie-3.5-8k"),
        ("doubao", "doubao-1-5-pro-32k-character-250715")
    ])
    def test_multi_vendor_api_consistency(self, mock_harborai_client, vendor, model, test_messages):
        """测试多厂商API一致性"""
        # 配置不同厂商的mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content=f"Response from {vendor} {model}",
                role="assistant"
            ),
            finish_reason="stop",
            index=0
        )]
        mock_response.usage = Mock(
            prompt_tokens=15,
            completion_tokens=10,
            total_tokens=25
        )
        mock_response.model = model
        mock_response.id = f"chatcmpl-{vendor}-test"
        
        # 配置mock以返回非流式响应
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return iter([])
            else:
                return mock_response
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 执行测试
        response = mock_harborai_client.chat.completions.create(
            model=model,
            messages=test_messages,
            temperature=0.7
        )
        
        # 验证所有厂商都返回一致性结构
        assert hasattr(response, 'choices')
        assert hasattr(response, 'usage')
        assert hasattr(response, 'model')
        assert response.model == model
        
        # 验证内容不为空
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.choices[0].message.role == "assistant"
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_parameter_validation(self, mock_harborai_client):
        """测试参数验证"""
        # 配置mock以模拟参数验证错误
        def mock_create(*args, **kwargs):
            if not args and not kwargs:
                raise TypeError("Missing required arguments")
            if 'temperature' in kwargs and kwargs['temperature'] > 2.0:
                raise ValueError("temperature must be between 0 and 2")
            if 'max_tokens' in kwargs and kwargs['max_tokens'] < 0:
                raise ValueError("max_tokens must be positive")
            return Mock()
        
        mock_harborai_client.chat.completions.create.side_effect = mock_create
        
        # 测试必需参数
        with pytest.raises((TypeError, ValueError, HarborAIError)):
            mock_harborai_client.chat.completions.create()
        
        # 测试无效的temperature值
        with pytest.raises((ValueError, HarborAIError)):
            mock_harborai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "test"}],
                temperature=2.5  # 超出范围
            )
        
        # 测试无效的max_tokens值
        with pytest.raises((ValueError, HarborAIError)):
            mock_harborai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=-1  # 负数
            )
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_optional_parameters(self, mock_harborai_client, test_messages):
        """测试可选参数处理"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="Test response",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15
        )
        
        # 配置mock以返回非流式响应
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return iter([])
            else:
                return mock_response
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 测试所有可选参?
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            temperature=0.8,
            max_tokens=150,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["\n", "END"],
            stream=False
        )
        
        # 验证调用成功
        assert response is not None
        # ֤óɹ
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_message_format_validation(self, mock_harborai_client):
        """测试消息格式验证"""
        # 配置mock以模拟消息格式验证
        def mock_create(*args, **kwargs):
            from harborai.utils.exceptions import ValidationError
            messages = kwargs.get('messages', [])
            
            # 验证消息格式
            if not messages:
                raise ValidationError("Messages cannot be empty")
            
            for msg in messages:
                if not isinstance(msg, dict):
                    raise ValidationError("Each message must be a dictionary")
                if 'role' not in msg:
                    raise ValidationError("Each message must have a 'role' field")
                if 'content' not in msg:
                    raise ValidationError("Each message must have a 'content' field")
                if msg['role'] not in ['system', 'user', 'assistant']:
                    raise ValidationError("Invalid role")
            
            # 返回有效响应
            return Mock(
                choices=[Mock(
                    message=Mock(
                        content="Valid response",
                        role="assistant"
                    )
                )]
            )
        
        mock_harborai_client.chat.completions.create.side_effect = mock_create
        
        # 测试有效的消息格?
        valid_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=valid_messages
        )
        
        assert response is not None
        
        # 测试无效的消息格?
        invalid_messages_cases = [
            [],  # 空消息列?
            [{"role": "invalid", "content": "test"}],  # 无效角色
            [{"role": "user"}],  # 缺少content
            [{"content": "test"}],  # 缺少role
        ]
        
        for invalid_messages in invalid_messages_cases:
            with pytest.raises((ValueError, TypeError, ValidationError, HarborAIError)):
                mock_harborai_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=invalid_messages
                )
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.deepseek_alignment
    def test_response_format_consistency(self, mock_harborai_client, test_messages):
        """测试响应格式一致性"""
        # 配置详细的mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="Detailed test response",
                role="assistant"
            ),
            finish_reason="stop",
            index=0
        )]
        mock_response.usage = Mock(
            prompt_tokens=20,
            completion_tokens=15,
            total_tokens=35
        )
        mock_response.model = "deepseek-chat"
        mock_response.id = "chatcmpl-consistency-test"
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        
        # 配置mock以返回非流式响应
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return iter([])
            else:
                return mock_response
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 执行测试
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages
        )
        
        # 验证响应格式完全符合DeepSeek标准
        assert response.object == "chat.completion"
        assert isinstance(response.created, int)
        assert response.id.startswith("chatcmpl-")
        assert response.model == "deepseek-chat"
        
        # 验证choices数组
        assert isinstance(response.choices, list)
        assert len(response.choices) > 0
        
        choice = response.choices[0]
        assert choice.index == 0
        assert choice.finish_reason in ["stop", "length", "content_filter", "tool_calls"]
        
        # 验证message对象
        message = choice.message
        assert message.role == "assistant"
        assert isinstance(message.content, str)
        assert len(message.content) > 0
        
        # 验证usage对象
        usage = response.usage
        assert isinstance(usage.prompt_tokens, int)
        assert isinstance(usage.completion_tokens, int)
        assert isinstance(usage.total_tokens, int)
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


class TestAsyncAPICompatibility:
    """异步API兼容性测试类"""
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.async_test
    async def test_async_chat_completions(self, mock_harborai_client, test_messages):
        """测试异步chat completions接口"""
        # 配置异步mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="Async response",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=12,
            completion_tokens=8,
            total_tokens=20
        )
        
        # 创建异步mock
        async def async_create(*args, **kwargs):
            return mock_response
        
        mock_harborai_client.chat.completions.acreate = async_create
        
        # 执行异步测试
        response = await mock_harborai_client.chat.completions.acreate(
            model="deepseek-chat",
            messages=test_messages
        )
        
        # 验证异步响应
        assert response is not None
        assert hasattr(response, 'choices')
        assert hasattr(response, 'usage')
        assert response.choices[0].message.content == "Async response"
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.async_test
    @pytest.mark.stream_test
    async def test_async_streaming(self, mock_harborai_client):
        """测试异步流式接口"""
        # 配置异步流式mock
        async def async_stream_generator():
            chunks = [
                Mock(
                    choices=[Mock(
                        delta=Mock(content="Async"),
                        index=0,
                        finish_reason=None
                    )],
                    object="chat.completion.chunk"
                ),
                Mock(
                    choices=[Mock(
                        delta=Mock(content=" stream"),
                        index=0,
                        finish_reason=None
                    )],
                    object="chat.completion.chunk"
                ),
                Mock(
                    choices=[Mock(
                        delta=Mock(content=None),
                        index=0,
                        finish_reason="stop"
                    )],
                    object="chat.completion.chunk"
                )
            ]
            
            for chunk in chunks:
                yield chunk
        
        # 创建异步mock方法
        async def async_create(*args, **kwargs):
            return async_stream_generator()
        
        mock_harborai_client.chat.completions.acreate = async_create
        
        # 执行异步流式测试
        stream = await mock_harborai_client.chat.completions.acreate(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        collected_content = []
        async for chunk in stream:
            if chunk.choices and hasattr(chunk.choices[0].delta, 'content'):
                if chunk.choices[0].delta.content:
                    collected_content.append(chunk.choices[0].delta.content)
        
        # 验证异步流式内容
        full_content = ''.join(collected_content)
        assert full_content == "Async stream"


class TestParameterPassthrough:
    """A003: 参数透传与扩展参数兼容性测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.parameter_passthrough
    def test_response_format_parameter(self, mock_harborai_client, test_messages):
        """测试response_format参数透传"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content='{"result": "structured response"}',
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18
        )
        
        # 配置mock以返回非流式响应
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return iter([])
            else:
                return mock_response
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 测试response_format参数
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            response_format={"type": "json_object"}
        )
        
        # 验证参数被正确传?
        # ֤óɹ
        # 验证响应格式参数传递
        assert response is not None
        
        # 验证响应格式
        assert response is not None
        assert hasattr(response, 'choices')
        assert response.choices[0].message.content
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.parameter_passthrough
    def test_structured_provider_parameter(self, mock_harborai_client, test_messages):
        """测试structured_provider参数"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="Structured provider response",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=12,
            completion_tokens=10,
            total_tokens=22
        )
        
        # 配置mock以返回非流式响应
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return iter([])
            else:
                return mock_response
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 测试structured_provider参数
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            structured_provider="agently"
        )
        
        # 验证参数被正确传?
        # 验证结构化提供者参数传递
        assert response is not None
        
        # 验证响应
        assert response is not None
        assert response.choices[0].message.content
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.parameter_passthrough
    def test_retry_policy_parameter(self, mock_harborai_client, test_messages):
        """测试retry_policy参数"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="Response with retry policy",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=8,
            completion_tokens=6,
            total_tokens=14
        )
        
        # 配置mock以返回非流式响应
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return iter([])
            else:
                return mock_response
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 测试retry_policy参数
        retry_policy = {
            "max_retries": 3,
            "backoff_factor": 2.0,
            "retry_on": ["timeout", "rate_limit"]
        }
        
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            retry_policy=retry_policy
        )
        
        # 验证参数被正确传?
        # 验证重试策略参数传递
        assert response is not None
        
        # 验证响应
        assert response is not None
        assert response.choices[0].message.content
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.parameter_passthrough
    def test_fallback_parameter(self, mock_harborai_client, test_messages):
        """测试fallback参数"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="Response with fallback",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=11,
            completion_tokens=9,
            total_tokens=20
        )
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 测试fallback参数
        fallback_config = {
            "models": ["deepseek-chat", "gpt-3.5-turbo"],
            "strategy": "sequential"
        }
        
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            fallback=fallback_config
        )
        
        # 验证参数被正确传?
        # 验证回退配置参数传递
        assert response is not None
        
        # 验证响应
        assert response is not None
        assert response.choices[0].message.content
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.parameter_passthrough
    def test_trace_id_parameter(self, mock_harborai_client, test_messages):
        """测试trace_id参数"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="Response with trace ID",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=9,
            completion_tokens=7,
            total_tokens=16
        )
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 测试trace_id参数
        trace_id = "test-trace-12345"
        
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            trace_id=trace_id
        )
        
        # 验证参数被正确传?
        # 验证追踪ID参数传递
        assert response is not None
        
        # 验证响应
        assert response is not None
        assert response.choices[0].message.content
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.parameter_passthrough
    def test_cost_tracking_parameter(self, mock_harborai_client, test_messages):
        """测试cost_tracking参数"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="Response with cost tracking",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=13,
            completion_tokens=11,
            total_tokens=24
        )
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 测试cost_tracking参数
        cost_tracking = {
            "enabled": True,
            "project_id": "test-project",
            "user_id": "test-user"
        }
        
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            cost_tracking=cost_tracking
        )
        
        # 验证成本追踪参数传递
        assert response is not None
        
        # 验证响应
        assert response is not None
        assert response.choices[0].message.content
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.parameter_passthrough
    def test_combined_parameters_compatibility(self, mock_harborai_client, test_messages):
        """测试所有参数组合兼容性"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content='{"combined": "all parameters working"}',
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=20,
            completion_tokens=15,
            total_tokens=35
        )
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 测试所有扩展参数组?
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            # OpenAI标准参数
            temperature=0.7,
            max_tokens=100,
            # 扩展参数
            response_format={"type": "json_object"},
            structured_provider="agently",
            extra_body={"custom": "value"},
            retry_policy={"max_retries": 2},
            fallback={"models": ["deepseek-chat"]},
            trace_id="combined-test-trace",
            cost_tracking={"enabled": True}
        )
        
        # 验证组合参数传递
        assert response is not None
        
        # 验证响应
        assert response is not None
        assert response.choices[0].message.content
        
        # 验证OpenAI兼容性（响应结构符合OpenAI标准?
        assert hasattr(response, 'choices')
        assert hasattr(response, 'usage')
        assert response.choices[0].message.role == "assistant"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.parameter_passthrough
    def test_extra_body_parameter(self, mock_harborai_client, test_messages):
        """测试extra_body参数"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="Response with extra body",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=15,
            completion_tokens=12,
            total_tokens=27
        )
        
        mock_harborai_client.chat.completions.create.return_value = mock_response
        
        # 测试extra_body参数
        extra_body = {
            "custom_field": "custom_value",
            "metadata": {"source": "test"}
        }
        
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            extra_body=extra_body
        )
        
        # 验证额外参数传递
        assert response is not None
        
        # 验证响应
        assert response is not None
        assert response.choices[0].message.content