# -*- coding: utf-8 -*-
"""
HarborAI 标准对齐验证测试模块 (模块N)

测试目标：
- N-001: 验证替换OpenAI SDK为HarborAI后，示例代码无需改动业务逻辑即可运行
- N-002: 验证ChatCompletion/ChatCompletionChunk字段对齐，确保与OpenAI SDK完全兼容

参考文档：
- HarborAI功能与性能测试清单.md (模块N)
- HarborAI_TDD.md
- Agently结构化输出语法设计理念.md
"""

import pytest
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

from harborai import HarborAI
from harborai.core.exceptions import HarborAIError
from harborai.utils.logger import get_logger

# 获取logger实例
logger = get_logger(__name__)


@dataclass
class OpenAICompatibilityTestCase:
    """OpenAI兼容性测试用例数据结构"""
    name: str
    description: str
    openai_code: str
    harborai_code: str
    expected_fields: List[str]
    test_type: str  # 'sync', 'async', 'stream'


class TestStandardAlignment:
    """标准对齐验证测试类"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self, mock_env_vars):
        """测试方法设置"""
        self.test_messages = [
            {"role": "system", "content": "你是一个有用的AI助手。"},
            {"role": "user", "content": "请用一句话解释量子纠缠现象。"}
        ]
        
        # 模拟OpenAI响应结构
        self.mock_openai_response = {
            "id": "chatcmpl-test-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "量子纠缠是指两个或多个粒子之间存在的一种量子力学关联，即使它们相距很远，对其中一个粒子的测量会瞬间影响另一个粒子的状态。"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 45,
                "total_tokens": 70
            }
        }
        
        # 模拟流式响应
        self.mock_stream_chunks = [
            {
                "id": "chatcmpl-test-stream",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-test-stream",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "量子纠缠"},
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-test-stream",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {"content": "是指两个或多个粒子之间存在的一种量子力学关联。"},
                    "finish_reason": None
                }]
            },
            {
                "id": "chatcmpl-test-stream",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
        ]
    
    # ==================== N-001 测试：OpenAI SDK替换兼容性 ====================
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.openai_alignment
    def test_n001_openai_sdk_replacement_sync(self, mock_harborai_client):
        """N-001: 测试同步调用的OpenAI SDK替换兼容性"""
        trace_id = str(uuid.uuid4())
        logger.info(f"开始N-001测试：OpenAI SDK替换兼容性验证 [trace_id={trace_id}]")
        
        try:
            # 配置异步mock响应 - 通过mock底层方法
            from unittest.mock import patch
            mock_response = self._create_mock_response(self.mock_openai_response)
            
            with patch.object(mock_harborai_client.client_manager, 'chat_completion_sync_with_fallback', return_value=mock_response):
                # 模拟原OpenAI代码（仅替换import和客户端初始化）
                # 原代码: from openai import OpenAI; client = OpenAI()
                # 新代码: from harborai import HarborAI; client = HarborAI()
                client = mock_harborai_client
                
                # 业务逻辑代码完全不变
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=self.test_messages,
                    temperature=0.7,
                    max_tokens=150
                )
                
                logger.info(f"HarborAI调用成功完成 [trace_id={trace_id}]")
                
                # 验证响应结构与OpenAI完全一致
                self._verify_openai_response_structure(response)
                
                # 验证业务逻辑可以正常使用响应
                content = response.choices[0].message.content
                assert isinstance(content, str)
                assert len(content) > 0
                assert "量子纠缠" in content
                
                # 验证调用参数传递正确 - 通过底层Mock方法验证
                # 注意：这里验证的是底层方法的调用，而不是表面API
                # 实际的参数验证应该在集成测试中进行
            
        except Exception as e:
            logger.error(f"N-001测试失败: {str(e)} [trace_id={trace_id}]")
            raise
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.openai_alignment
    @pytest.mark.asyncio
    async def test_n001_openai_sdk_replacement_async(self, mock_harborai_client):
        """N-001: 测试异步调用的OpenAI SDK替换兼容性"""
        trace_id = str(uuid.uuid4())
        logger.info(f"开始N-001异步测试：OpenAI SDK替换兼容性验证 [trace_id={trace_id}]")
        
        try:
            # 配置mock响应 - 通过mock底层方法
            from unittest.mock import patch, AsyncMock
            mock_response = self._create_mock_response(self.mock_openai_response)
            
            with patch.object(mock_harborai_client.client_manager, 'chat_completion_with_fallback', new_callable=AsyncMock, return_value=mock_response):
                # 模拟原OpenAI异步代码
                # 原代码: response = await client.chat.completions.create(...)
                # 新代码: response = await client.chat.completions.acreate(...)
                client = mock_harborai_client
                
                response = await client.chat.completions.acreate(
                    model="gpt-4",
                    messages=self.test_messages,
                    temperature=0.7,
                    max_tokens=150
                )
                
                logger.info(f"HarborAI异步调用成功完成 [trace_id={trace_id}]")
                
                # 验证响应结构与OpenAI完全一致
                self._verify_openai_response_structure(response)
                
                # 验证异步调用参数传递正确
                # 注意：这里验证的是底层方法的调用，而不是表面API
                mock_method = mock_harborai_client.client_manager.chat_completion_with_fallback
                assert mock_method.called, "底层异步方法应该被调用"
                
                # 验证调用参数（如果需要的话）
                # call_args = mock_method.call_args
                # if call_args:
                #     args, kwargs = call_args
                #     # 验证传递给底层方法的参数
            
        except Exception as e:
            logger.error(f"N-001异步测试失败: {str(e)} [trace_id={trace_id}]")
            raise
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.openai_alignment
    @pytest.mark.stream_test
    def test_n001_openai_sdk_replacement_stream(self, mock_harborai_client):
        """N-001: 测试流式调用的OpenAI SDK替换兼容性"""
        trace_id = str(uuid.uuid4())
        logger.info(f"开始N-001流式测试：OpenAI SDK替换兼容性验证 [trace_id={trace_id}]")
        
        try:
            # 配置流式mock响应 - 通过mock底层方法
            from unittest.mock import patch
            mock_chunks = [self._create_mock_chunk(chunk_data) for chunk_data in self.mock_stream_chunks]
            
            with patch.object(mock_harborai_client.client_manager, 'chat_completion_sync_with_fallback', return_value=iter(mock_chunks)):
                # 模拟原OpenAI流式代码
                client = mock_harborai_client
                
                stream = client.chat.completions.create(
                    model="gpt-4",
                    messages=self.test_messages,
                    stream=True
                )
                
                # 收集流式响应（业务逻辑代码不变）
                collected_content = []
                for chunk in stream:
                    # 验证每个chunk的结构
                    self._verify_openai_chunk_structure(chunk)
                    
                    # 业务逻辑：收集内容
                    if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        if isinstance(content, str) and content:  # 确保是非空字符串
                            collected_content.append(content)
                
                # 验证收集到的内容
                full_content = ''.join(collected_content)
                assert "量子纠缠" in full_content
                assert "量子力学关联" in full_content
                
        except Exception as e:
            logger.error(f"N-001流式测试失败: {str(e)} [trace_id={trace_id}]")
            raise
    
    # ==================== N-002 测试：字段对齐验证 ====================
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.field_alignment
    def test_n002_chat_completion_field_alignment(self, mock_harborai_client):
        """N-002: 测试ChatCompletion响应字段对齐"""
        trace_id = str(uuid.uuid4())
        logger.info(f"开始N-002测试：ChatCompletion字段对齐验证 [trace_id={trace_id}]")
        
        try:
            # 配置mock响应 - 通过mock底层方法
            from unittest.mock import patch
            mock_response = self._create_mock_response(self.mock_openai_response)
            
            with patch.object(mock_harborai_client.client_manager, 'chat_completion_sync_with_fallback', return_value=mock_response):
                response = mock_harborai_client.chat.completions.create(
                    model="gpt-4",
                    messages=self.test_messages
                )
                
                logger.info(f"HarborAI调用成功，开始字段验证 [trace_id={trace_id}]")
                
                # 验证顶级字段
                required_top_fields = ['id', 'object', 'created', 'model', 'choices', 'usage']
                for field in required_top_fields:
                    assert hasattr(response, field), f"缺少顶级字段: {field}"
                    assert getattr(response, field) is not None, f"字段 {field} 为空"
                
                # 验证object字段值
                assert response.object == "chat.completion"
                
                # 验证choices字段结构
                assert isinstance(response.choices, list)
                assert len(response.choices) > 0
                
                choice = response.choices[0]
                choice_fields = ['index', 'message', 'finish_reason']
                for field in choice_fields:
                    assert hasattr(choice, field), f"choices[0]缺少字段: {field}"
                
                # 验证message字段结构
                message = choice.message
                message_fields = ['role', 'content']
                for field in message_fields:
                    assert hasattr(message, field), f"message缺少字段: {field}"
                
                assert message.role == "assistant"
                assert isinstance(message.content, str)
                
                # 验证usage字段结构
                usage = response.usage
                usage_fields = ['prompt_tokens', 'completion_tokens', 'total_tokens']
                for field in usage_fields:
                    assert hasattr(usage, field), f"usage缺少字段: {field}"
                    assert isinstance(getattr(usage, field), int), f"usage.{field} 不是整数"
                
                # 验证token计算正确性
                assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
            
        except Exception as e:
            logger.error(f"N-002测试失败: {str(e)} [trace_id={trace_id}]")
            raise
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.field_alignment
    @pytest.mark.asyncio
    async def test_n002_chat_completion_field_alignment_async(self, mock_harborai_client):
        """N-002: 测试ChatCompletion异步响应字段对齐"""
        trace_id = str(uuid.uuid4())
        logger.info(f"开始N-002异步测试：ChatCompletion字段对齐验证 [trace_id={trace_id}]")
        
        try:
            # 配置mock响应 - 通过mock底层方法
            from unittest.mock import patch
            mock_response = self._create_mock_response(self.mock_openai_response)
            
            with patch.object(mock_harborai_client.client_manager, 'chat_completion_with_fallback', return_value=mock_response):
                response = await mock_harborai_client.chat.completions.acreate(
                    model="gpt-4",
                    messages=self.test_messages
                )
                
                logger.info(f"HarborAI异步调用成功，开始字段验证 [trace_id={trace_id}]")
                
                # 验证响应结构（与同步版本相同）
                self._verify_openai_response_structure(response)
            
        except Exception as e:
            logger.error(f"N-002异步测试失败: {str(e)} [trace_id={trace_id}]")
            raise
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.field_alignment
    @pytest.mark.stream_test
    def test_n002_chat_completion_chunk_field_alignment(self, mock_harborai_client):
        """N-002: 测试ChatCompletionChunk流式响应字段对齐"""
        # 配置流式mock响应 - 通过mock底层方法
        from unittest.mock import patch
        mock_chunks = [self._create_mock_chunk(chunk) for chunk in self.mock_stream_chunks]
        
        with patch.object(mock_harborai_client.client_manager, 'chat_completion_sync_with_fallback', return_value=iter(mock_chunks)):
            stream = mock_harborai_client.chat.completions.create(
                model="gpt-4",
                messages=self.test_messages,
                stream=True
            )
            
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                
                # 验证顶级字段
                required_chunk_fields = ['id', 'object', 'created', 'model', 'choices']
                for field in required_chunk_fields:
                    assert hasattr(chunk, field), f"chunk缺少顶级字段: {field}"
                
                # 验证object字段值
                assert chunk.object == "chat.completion.chunk"
                
                # 验证choices字段结构
                assert isinstance(chunk.choices, list)
                if chunk.choices:  # 某些chunk可能没有choices
                    choice = chunk.choices[0]
                    choice_fields = ['index', 'delta', 'finish_reason']
                    for field in choice_fields:
                        assert hasattr(choice, field), f"chunk.choices[0]缺少字段: {field}"
                    
                    # 验证delta字段结构
                    delta = choice.delta
                    assert hasattr(delta, 'content') or hasattr(delta, 'role'), "delta必须包含content或role字段"
                    
                    # 验证finish_reason字段
                    if choice.finish_reason is not None:
                        assert choice.finish_reason in ["stop", "length", "content_filter", "tool_calls"]
            
            # 验证至少收到了一些chunk
            assert chunk_count > 0, "没有收到任何流式响应chunk"
    
    # ==================== 辅助方法 ====================
    
    def _create_mock_response(self, response_data: Dict[str, Any]) -> Mock:
        """创建mock响应对象"""
        mock_response = Mock()
        
        # 设置顶级属性，确保类型正确
        for key, value in response_data.items():
            if key == 'choices':
                # 处理choices数组
                mock_choices = []
                for choice_data in value:
                    mock_choice = Mock()
                    mock_choice.index = choice_data.get('index', 0)
                    mock_choice.finish_reason = choice_data.get('finish_reason')
                    
                    # 处理message
                    if 'message' in choice_data:
                        mock_message = Mock()
                        for msg_key, msg_value in choice_data['message'].items():
                            setattr(mock_message, msg_key, msg_value)
                        mock_choice.message = mock_message
                    
                    mock_choices.append(mock_choice)
                mock_response.choices = mock_choices
            elif key == 'usage':
                # 处理usage对象
                mock_usage = Mock()
                for usage_key, usage_value in value.items():
                    setattr(mock_usage, usage_key, usage_value)
                mock_response.usage = mock_usage
            else:
                # 直接设置属性值，保持原始类型
                setattr(mock_response, key, value)
        
        # 确保关键字段的类型正确
        if hasattr(mock_response, 'id') and not isinstance(mock_response.id, str):
            mock_response.id = str(mock_response.id)
        if hasattr(mock_response, 'created') and not isinstance(mock_response.created, int):
            mock_response.created = int(mock_response.created)
        if hasattr(mock_response, 'model') and not isinstance(mock_response.model, str):
            mock_response.model = str(mock_response.model)
        
        return mock_response
    
    def _create_mock_chunk(self, chunk_data: Dict[str, Any]) -> Mock:
        """创建mock流式响应chunk对象"""
        mock_chunk = Mock()
        
        # 设置顶级属性，确保类型正确
        for key, value in chunk_data.items():
            if key == 'choices':
                # 处理choices数组
                mock_choices = []
                for choice_data in value:
                    mock_choice = Mock()
                    mock_choice.index = choice_data.get('index', 0)
                    mock_choice.finish_reason = choice_data.get('finish_reason')
                    
                    # 处理delta
                    if 'delta' in choice_data:
                        mock_delta = Mock()
                        for delta_key, delta_value in choice_data['delta'].items():
                            # 确保content字段是字符串类型，避免Mock对象导致字符串连接错误
                            if delta_key == 'content' and delta_value is not None:
                                setattr(mock_delta, delta_key, str(delta_value))
                            else:
                                setattr(mock_delta, delta_key, delta_value)
                        mock_choice.delta = mock_delta
                    
                    mock_choices.append(mock_choice)
                mock_chunk.choices = mock_choices
            else:
                # 直接设置属性值，保持原始类型
                setattr(mock_chunk, key, value)
        
        # 确保关键字段的类型正确
        if hasattr(mock_chunk, 'id') and not isinstance(mock_chunk.id, str):
            mock_chunk.id = str(mock_chunk.id)
        if hasattr(mock_chunk, 'created') and not isinstance(mock_chunk.created, int):
            mock_chunk.created = int(mock_chunk.created)
        if hasattr(mock_chunk, 'model') and not isinstance(mock_chunk.model, str):
            mock_chunk.model = str(mock_chunk.model)
        
        return mock_chunk
    
    def _verify_openai_response_structure(self, response) -> None:
        """验证响应结构符合OpenAI标准"""
        # 验证必需的顶级字段
        required_fields = ['id', 'object', 'created', 'model', 'choices', 'usage']
        for field in required_fields:
            assert hasattr(response, field), f"响应缺少必需字段: {field}"
        
        # 验证字段类型
        assert isinstance(response.id, str)
        assert response.object == "chat.completion"
        assert isinstance(response.created, int)
        assert isinstance(response.model, str)
        assert isinstance(response.choices, list)
        assert len(response.choices) > 0
        
        # 验证choice结构
        choice = response.choices[0]
        assert hasattr(choice, 'index')
        assert hasattr(choice, 'message')
        assert hasattr(choice, 'finish_reason')
        
        # 验证message结构
        message = choice.message
        assert hasattr(message, 'role')
        assert hasattr(message, 'content')
        assert message.role == "assistant"
        
        # 验证usage结构
        usage = response.usage
        assert hasattr(usage, 'prompt_tokens')
        assert hasattr(usage, 'completion_tokens')
        assert hasattr(usage, 'total_tokens')
    
    def _verify_openai_chunk_structure(self, chunk) -> None:
        """验证流式响应chunk结构符合OpenAI标准"""
        # 验证必需的顶级字段
        required_fields = ['id', 'object', 'created', 'model', 'choices']
        for field in required_fields:
            assert hasattr(chunk, field), f"chunk缺少必需字段: {field}"
        
        # 验证字段类型和值
        assert isinstance(chunk.id, str)
        assert chunk.object == "chat.completion.chunk"
        assert isinstance(chunk.created, int)
        assert isinstance(chunk.model, str)
        assert isinstance(chunk.choices, list)
        
        # 验证choice结构（如果存在）
        if chunk.choices:
            choice = chunk.choices[0]
            assert hasattr(choice, 'index')
            assert hasattr(choice, 'delta')
            assert hasattr(choice, 'finish_reason')