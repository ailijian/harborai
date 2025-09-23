#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
客户端测试

测试 HarborAI 主客户端的功能。
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from harborai import HarborAI
from harborai.utils.exceptions import ModelNotFoundError, ValidationError


class TestHarborAIClient:
    """HarborAI 客户端测试"""
    
    def test_client_initialization(self, harbor_client):
        """测试客户端初始化"""
        assert harbor_client is not None
        assert harbor_client.chat is not None
        assert harbor_client.chat.completions is not None
        assert harbor_client.client_manager is not None
    
    def test_get_available_models(self, harbor_client):
        """测试获取可用模型"""
        models = harbor_client.get_available_models()
        assert isinstance(models, list)
        assert "mock-model" in models
    
    def test_get_plugin_info(self, harbor_client):
        """测试获取插件信息"""
        info = harbor_client.get_plugin_info()
        assert isinstance(info, dict)
        assert "mock" in info
        assert info["mock"]["model_count"] == 1
    
    def test_chat_completion_sync(self, harbor_client, sample_messages):
        """测试同步聊天完成"""
        response = harbor_client.chat.completions.create(
            messages=sample_messages,
            model="mock-model"
        )
        
        assert response is not None
        assert response["id"] == "mock-response-id"
        assert response["model"] == "mock-model"
        assert len(response["choices"]) == 1
        assert response["choices"][0]["message"]["content"] == "Mock response"
    
    @pytest.mark.asyncio
    async def test_chat_completion_async(self, harbor_client, sample_messages):
        """测试异步聊天完成"""
        response = await harbor_client.chat.completions.acreate(
            messages=sample_messages,
            model="mock-model"
        )
        
        assert response is not None
        assert response["id"] == "mock-response-id"
        assert response["model"] == "mock-model"
        assert len(response["choices"]) == 1
        assert response["choices"][0]["message"]["content"] == "Mock response"
    
    def test_chat_completion_with_parameters(self, harbor_client, sample_messages):
        """测试带参数的聊天完成"""
        response = harbor_client.chat.completions.create(
            messages=sample_messages,
            model="mock-model",
            temperature=0.7,
            max_tokens=100,
            top_p=0.9
        )
        
        assert response is not None
        assert response["model"] == "mock-model"
    
    def test_chat_completion_with_structured_output(
        self, 
        harbor_client, 
        sample_messages, 
        sample_structured_schema
    ):
        """测试结构化输出"""
        response = harbor_client.chat.completions.create(
            messages=sample_messages,
            model="mock-model",
            response_format=sample_structured_schema
        )
        
        assert response is not None
        assert response["model"] == "mock-model"
    
    def test_invalid_model_error(self, harbor_client, sample_messages):
        """测试无效模型错误"""
        with pytest.raises(ModelNotFoundError):
            harbor_client.chat.completions.create(
                messages=sample_messages,
                model="invalid-model"
            )
    
    def test_context_manager_sync(self, test_settings, mock_client_manager, monkeypatch):
        """测试同步上下文管理器"""
        monkeypatch.setattr("harborai.config.settings.get_settings", lambda: test_settings)
        
        with HarborAI() as client:
            client.client_manager = mock_client_manager
            models = client.get_available_models()
            assert "mock-model" in models
    
    @pytest.mark.asyncio
    async def test_context_manager_async(self, test_settings, mock_client_manager, monkeypatch):
        """测试异步上下文管理器"""
        monkeypatch.setattr("harborai.config.settings.get_settings", lambda: test_settings)
        
        async with HarborAI() as client:
            client.client_manager = mock_client_manager
            models = client.get_available_models()
            assert "mock-model" in models


class TestChatCompletions:
    """聊天完成接口测试"""
    
    def test_create_with_all_parameters(self, harbor_client, sample_messages):
        """测试使用所有参数创建聊天完成"""
        response = harbor_client.chat.completions.create(
            messages=sample_messages,
            model="mock-model",
            frequency_penalty=0.5,
            logit_bias={"50256": -100},
            logprobs=True,
            top_logprobs=5,
            max_tokens=150,
            n=1,
            presence_penalty=0.6,
            seed=42,
            stop=["\n", "END"],
            temperature=0.8,
            top_p=0.95,
            user="test-user"
        )
        
        assert response is not None
        assert response["model"] == "mock-model"
    
    @pytest.mark.asyncio
    async def test_acreate_with_all_parameters(self, harbor_client, sample_messages):
        """测试异步版本使用所有参数创建聊天完成"""
        response = await harbor_client.chat.completions.acreate(
            messages=sample_messages,
            model="mock-model",
            frequency_penalty=0.5,
            max_tokens=150,
            temperature=0.8,
            top_p=0.95
        )
        
        assert response is not None
        assert response["model"] == "mock-model"
    
    def test_create_with_tools(self, harbor_client, sample_messages):
        """测试使用工具的聊天完成"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        response = harbor_client.chat.completions.create(
            messages=sample_messages,
            model="mock-model",
            tools=tools,
            tool_choice="auto"
        )
        
        assert response is not None
        assert response["model"] == "mock-model"
    
    def test_create_with_extra_body(self, harbor_client, sample_messages):
        """测试使用 extra_body 参数"""
        response = harbor_client.chat.completions.create(
            messages=sample_messages,
            model="mock-model",
            extra_body={"thinking": {"type": "enabled"}}
        )
        
        assert response is not None
        assert response["model"] == "mock-model"
    
    @patch('harborai.utils.logger.APICallLogger.log_request')
    @patch('harborai.utils.logger.APICallLogger.log_response')
    def test_logging_integration(self, mock_log_response, mock_log_request, harbor_client, sample_messages):
        """测试日志集成"""
        response = harbor_client.chat.completions.create(
            messages=sample_messages,
            model="mock-model"
        )
        
        assert response is not None
        # 验证日志方法被调用
        mock_log_request.assert_called_once()
        mock_log_response.assert_called_once()


class TestErrorHandling:
    """错误处理测试"""
    
    def test_model_not_found_error(self, harbor_client, sample_messages):
        """测试模型未找到错误"""
        with pytest.raises(ModelNotFoundError) as exc_info:
            harbor_client.chat.completions.create(
                messages=sample_messages,
                model="nonexistent-model"
            )
        
        assert "nonexistent-model" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_async_model_not_found_error(self, harbor_client, sample_messages):
        """测试异步模型未找到错误"""
        with pytest.raises(ModelNotFoundError) as exc_info:
            await harbor_client.chat.completions.acreate(
                messages=sample_messages,
                model="nonexistent-model"
            )
        
        assert "nonexistent-model" in str(exc_info.value)
    
    def test_empty_messages_validation(self, harbor_client):
        """测试空消息验证"""
        with pytest.raises(ValidationError):
            harbor_client.chat.completions.create(
                messages=[],
                model="mock-model"
            )
    
    def test_invalid_message_format(self, harbor_client):
        """测试无效消息格式"""
        invalid_messages = [
            {"role": "invalid", "content": "test"}
        ]
        
        with pytest.raises(ValidationError):
            harbor_client.chat.completions.create(
                messages=invalid_messages,
                model="mock-model"
            )