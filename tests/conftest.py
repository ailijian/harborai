#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest 配置文件

提供测试夹具和配置。
"""

import os
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Generator, AsyncGenerator

from harborai import HarborAI
from harborai.config.settings import Settings, get_settings
from harborai.core.client_manager import ClientManager
from harborai.core.base_plugin import BaseLLMPlugin, ModelInfo


@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """测试配置"""
    return Settings(
        # 禁用数据库
        enable_database=False,
        
        # 测试日志配置
        log_level="DEBUG",
        log_to_console=True,
        log_to_database=False,
        
        # 测试超时配置
        default_timeout=10,
        max_retries=1,
        
        # 测试插件目录
        plugin_directories=["tests.mocks.plugins"]
    )


@pytest.fixture
def mock_plugin():
    """模拟插件"""
    class MockPlugin(BaseLLMPlugin):
        def __init__(self, name: str = "mock", **config):
            super().__init__(name, **config)
        
        @property
        def supported_models(self):
            return [
                ModelInfo(
                    id="mock-model",
                    name="Mock Model",
                    provider="mock",
                    supports_thinking=True,
                    supports_structured_output=True,
                    max_tokens=4096,
                    context_window=8192
                )
            ]
        
        def chat_completion(self, model, messages, **kwargs):
            return {
                "id": "mock-response-id",
                "object": "chat.completion",
                "created": 1234567890,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Mock response"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15
                }
            }
        
        async def chat_completion_async(self, model, messages, **kwargs):
            return self.chat_completion(model, messages, **kwargs)
    
    return MockPlugin()


@pytest.fixture
def mock_client_manager(mock_plugin):
    """模拟客户端管理器"""
    manager = ClientManager()
    manager.register_plugin(mock_plugin)
    return manager


@pytest.fixture
def harbor_client(test_settings, mock_client_manager, monkeypatch):
    """HarborAI 测试客户端"""
    # 使用测试配置
    monkeypatch.setattr("harborai.config.settings.get_settings", lambda: test_settings)
    
    # 创建客户端
    client = HarborAI()
    
    # 替换客户端管理器
    client.client_manager = mock_client_manager
    
    # 重新初始化聊天接口以使用新的客户端管理器
    from harborai.api.client import ChatCompletions
    client.chat.completions = ChatCompletions(mock_client_manager)
    
    return client


@pytest.fixture
def sample_messages():
    """示例消息"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]


@pytest.fixture
def sample_structured_schema():
    """示例结构化输出模式"""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The response message"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0 and 1"
                    }
                },
                "required": ["message", "confidence"]
            }
        }
    }


@pytest.fixture(autouse=True)
def cleanup_database():
    """清理数据库连接"""
    yield
    # 测试后清理 - 暂时禁用数据库清理
    pass


@pytest.fixture
def mock_openai_response():
    """模拟 OpenAI 响应"""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! I'm doing well, thank you for asking."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 12,
            "total_tokens": 32
        }
    }


@pytest.fixture
def mock_thinking_response():
    """模拟思考模型响应"""
    return {
        "id": "chatcmpl-o1-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "o1-preview",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Based on my analysis, the answer is 42."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150
        },
        "reasoning_content": "Let me think about this step by step. The question asks about the meaning of life..."
    }


class AsyncContextManager:
    """异步上下文管理器辅助类"""
    
    def __init__(self, async_func):
        self.async_func = async_func
    
    async def __aenter__(self):
        return await self.async_func()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """异步上下文管理器工厂"""
    return AsyncContextManager