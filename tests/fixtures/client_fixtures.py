# -*- coding: utf-8 -*-
"""
客户端夹具模块
提供各种客户端实例的测试夹具
"""

import pytest
import os
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock, patch
import aiohttp
import httpx
from pathlib import Path
import json
from harborai import HarborAI
from harborai.core.plugins.deepseek_plugin import DeepSeekPlugin
from harborai.core.plugins.wenxin_plugin import WenxinPlugin
from harborai.core.plugins.doubao_plugin import DoubaoPlugin
from harborai.utils.exceptions import HarborAIError

# HarborAI 模块已在上方正确导入


@pytest.fixture(scope='session')
def base_client_config() -> Dict[str, Any]:
    """基础客户端配置"""
    return {
        'api_key': 'test_api_key_12345',
        'base_url': 'https://api.test.harborai.com',
        'timeout': 30.0,
        'max_retries': 3,
        'retry_delay': 1.0,
        'enable_logging': True,
        'log_level': 'DEBUG'
    }


@pytest.fixture(scope='session')
def deepseek_config(base_client_config) -> Dict[str, Any]:
    """DeepSeek客户端配置"""
    config = base_client_config.copy()
    config.update({
        'provider': 'deepseek',
        'model': 'deepseek-chat',
        'base_url': 'https://api.deepseek.com/v1',
        'api_key': 'sk-deepseek-test-key'
    })
    return config


@pytest.fixture(scope='session')
def ernie_config(base_client_config) -> Dict[str, Any]:
    """ERNIE客户端配置"""
    config = base_client_config.copy()
    config.update({
        'provider': 'ernie',
        'model': 'ernie-4.0-turbo-8k',
        'base_url': 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat',
        'api_key': 'ernie-test-key',
        'secret_key': 'ernie-test-secret'
    })
    return config


@pytest.fixture(scope='session')
def doubao_config(base_client_config) -> Dict[str, Any]:
    """Doubao客户端配置"""
    config = base_client_config.copy()
    config.update({
        'provider': 'doubao',
        'model': 'doubao-1-5-pro-32k-character-250715',
        'base_url': 'https://ark.cn-beijing.volces.com/api/v3',
        'api_key': 'doubao-test-key'
    })
    return config


@pytest.fixture(scope='session')
def reasoning_model_config(base_client_config) -> Dict[str, Any]:
    """推理模型配置"""
    config = base_client_config.copy()
    config.update({
        'provider': 'deepseek',
        'model': 'deepseek-reasoner',
        'enable_reasoning': True,
        'reasoning_effort': 'medium'
    })
    return config


# 多厂商客户端参数化夹具
@pytest.fixture(params=[
    "deepseek",
    "ernie", 
    "doubao"
])
def multi_vendor_client(request, base_client_config):
    """多厂商客户端参数化夹具"""
    vendor = request.param
    
    # 根据厂商配置不同的参数
    vendor_configs = {
        "deepseek": {
            "api_key": os.getenv("DEEPSEEK_API_KEY", "test-deepseek-key"),
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "provider": "deepseek"
        },
        "ernie": {
            "api_key": os.getenv("WENXIN_API_KEY", "test-ernie-key"),
            "base_url": "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat",
            "model": "ernie-4.0-turbo-8k",
            "provider": "ernie"
        },
        "doubao": {
            "api_key": os.getenv("DOUBAO_API_KEY", "test-doubao-key"),
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "model": "doubao-1-5-pro-32k-character-250715",
            "provider": "doubao"
        }
    }
    
    config = {**base_client_config, **vendor_configs[vendor]}
    
    # 如果启用真实API测试，创建真实客户端
    if os.getenv("ENABLE_REAL_API_TESTS", "false").lower() == "true":
        try:
            return HarborAI(**config)
        except Exception as e:
            pytest.skip(f"Failed to create {vendor} client: {e}")
    else:
        # 否则返回Mock客户端
        client = Mock(spec=HarborAI)
        client.config = config
        client.vendor = vendor
        return client


# 推理模型夹具
@pytest.fixture(params=[
    "deepseek-reasoner",
    "ernie-x1-turbo-32k"
])
def reasoning_models(request):
    """推理模型参数化夹具"""
    return request.param


@pytest.fixture(params=[
    "deepseek-chat",
    "ernie-3.5-8k",
    "ernie-4.0-turbo-8k",
    "doubao-1-5-pro-32k-character-250715",
    "doubao-seed-1-6-250615"
])
def non_reasoning_models(request):
    """非推理模型参数化夹具"""
    return request.param


@pytest.fixture(scope='function')
def mock_client(base_client_config) -> Mock:
    """Mock客户端夹具"""
    client = Mock()
    client.config = base_client_config
    
    # 创建嵌套的 Mock 结构
    client.chat = Mock()
    client.chat.completions = Mock()
    
    # 配置常用方法的返回值
    client.chat.completions.create.return_value = {
        'id': 'test-completion-id',
        'object': 'chat.completion',
        'created': 1234567890,
        'model': 'test-model',
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': 'This is a test response'
            },
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': 10,
            'completion_tokens': 5,
            'total_tokens': 15
        }
    }
    
    return client


@pytest.fixture(scope='function')
def mock_async_client(base_client_config) -> AsyncMock:
    """异步Mock客户端夹具"""
    client = AsyncMock()
    client.config = base_client_config
    
    # 创建嵌套的 Mock 结构
    client.chat = AsyncMock()
    client.chat.completions = AsyncMock()
    
    # 配置异步方法的返回值
    async def mock_create_completion(*args, **kwargs):
        return {
            'id': 'test-async-completion-id',
            'object': 'chat.completion',
            'created': 1234567890,
            'model': 'test-model',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': 'This is a test async response'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 10,
                'completion_tokens': 5,
                'total_tokens': 15
            }
        }
    
    client.chat.completions.create = mock_create_completion
    
    return client


@pytest.fixture
def real_client(base_client_config) -> Optional[HarborAI]:
    """真实客户端夹具（仅在启用真实API测试时创建）"""
    if not os.getenv("ENABLE_REAL_API_TESTS", "false").lower() == "true":
        pytest.skip("Real API tests are disabled")
    
    try:
        client = HarborAI(**base_client_config)
        return client
    except Exception as e:
        pytest.skip(f"Failed to create real client: {e}")


@pytest.fixture
def real_async_client(base_client_config) -> Optional[HarborAI]:
    """真实异步客户端夹具（仅在启用真实API测试时创建）"""
    if not os.getenv("ENABLE_REAL_API_TESTS", "false").lower() == "true":
        pytest.skip("Real API tests are disabled")
    
    try:
        client = HarborAI(**base_client_config)
        return client
    except Exception as e:
        pytest.skip(f"Failed to create real async client: {e}")


@pytest.fixture(scope='function')
def deepseek_client(deepseek_config) -> Mock:
    """DeepSeek客户端夹具"""
    client = Mock(spec=HarborAIClient)
    client.config = deepseek_config
    
    # DeepSeek特定的响应格式
    client.chat.completions.create.return_value = {
        'id': 'deepseek-completion-id',
        'object': 'chat.completion',
        'created': 1234567890,
        'model': 'deepseek-chat',
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': 'DeepSeek response content'
            },
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': 15,
            'completion_tokens': 8,
            'total_tokens': 23
        }
    }
    
    return client


@pytest.fixture(scope='function')
def reasoning_client(reasoning_model_config) -> Mock:
    """推理模型客户端夹具"""
    client = Mock(spec=HarborAIClient)
    client.config = reasoning_model_config
    
    # 推理模型特定的响应格式（包含reasoning_content）
    client.chat.completions.create.return_value = {
        'id': 'reasoning-completion-id',
        'object': 'chat.completion',
        'created': 1234567890,
        'model': 'deepseek-reasoner',
        'choices': [{
            'index': 0,
            'message': {
                'role': 'assistant',
                'content': 'Final reasoning result',
                'reasoning_content': 'This is the step-by-step reasoning process...'
            },
            'finish_reason': 'stop'
        }],
        'usage': {
            'prompt_tokens': 20,
            'completion_tokens': 50,
            'total_tokens': 70,
            'reasoning_tokens': 30
        }
    }
    
    return client


@pytest.fixture(scope='function')
def stream_client(base_client_config) -> Mock:
    """流式客户端夹具"""
    client = Mock(spec=HarborAIClient)
    client.config = base_client_config
    
    # 模拟流式响应
    def mock_stream_response():
        chunks = [
            {
                'id': 'stream-chunk-1',
                'object': 'chat.completion.chunk',
                'created': 1234567890,
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant', 'content': 'Hello'},
                    'finish_reason': None
                }]
            },
            {
                'id': 'stream-chunk-2',
                'object': 'chat.completion.chunk',
                'created': 1234567891,
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'delta': {'content': ' world!'},
                    'finish_reason': None
                }]
            },
            {
                'id': 'stream-chunk-3',
                'object': 'chat.completion.chunk',
                'created': 1234567892,
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop'
                }]
            }
        ]
        return iter(chunks)
    
    client.chat.completions.create.return_value = mock_stream_response()
    
    return client


@pytest.fixture(scope='function')
def http_session() -> Generator[aiohttp.ClientSession, None, None]:
    """HTTP会话夹具"""
    session = aiohttp.ClientSession()
    yield session
    asyncio.create_task(session.close())


@pytest.fixture(scope='function')
def httpx_client() -> Generator[httpx.Client, None, None]:
    """HTTPX客户端夹具"""
    with httpx.Client() as client:
        yield client


@pytest.fixture(scope='function')
async def httpx_async_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """异步HTTPX客户端夹具"""
    async with httpx.AsyncClient() as client:
        yield client


@pytest.fixture(scope='function')
def client_with_retry(base_client_config) -> Mock:
    """带重试机制的客户端夹具"""
    client = Mock(spec=HarborAIClient)
    config = base_client_config.copy()
    config.update({
        'max_retries': 5,
        'retry_delay': 0.1,
        'backoff_factor': 2.0
    })
    client.config = config
    
    return client


@pytest.fixture(scope='function')
def client_with_custom_headers(base_client_config) -> Mock:
    """带自定义请求头的客户端夹具"""
    client = Mock(spec=HarborAIClient)
    config = base_client_config.copy()
    config.update({
        'default_headers': {
            'User-Agent': 'HarborAI-Test/1.0',
            'X-Test-Mode': 'true',
            'X-Request-ID': 'test-request-123'
        }
    })
    client.config = config
    
    return client


@pytest.fixture(scope='function')
def multi_provider_clients(deepseek_config, ernie_config, doubao_config) -> Dict[str, Mock]:
    """多提供商客户端夹具"""
    clients = {}
    
    for provider, config in [('deepseek', deepseek_config), ('ernie', ernie_config), ('doubao', doubao_config)]:
        client = Mock(spec=HarborAIClient)
        client.config = config
        client.provider = provider
        clients[provider] = client
    
    return clients


@pytest.fixture(scope='function')
def client_factory():
    """客户端工厂夹具"""
    def create_client(provider: str = 'default', **kwargs) -> Mock:
        client = Mock(spec=HarborAIClient)
        client.config = kwargs
        client.provider = provider
        return client
    
    return create_client