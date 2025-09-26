#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局测试配置

提供全局的测试配置、夹具和钩子函数
"""

import os
import sys
import pytest
import asyncio
from typing import Dict, Any, Generator
from unittest.mock import Mock, AsyncMock

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from harborai import HarborAI
from harborai.utils.exceptions import HarborAIError

# 导入所有夹具模块
from tests.fixtures.client_fixtures import *
from tests.fixtures.data_fixtures import *
from tests.fixtures.mock_fixtures import *
from tests.fixtures.performance_fixtures import *


# ==================== 测试环境配置 ====================

def pytest_configure(config):
    """pytest 配置钩子"""
    # 设置测试环境变量
    os.environ.setdefault("HARBORAI_ENV", "test")
    os.environ.setdefault("HARBORAI_LOG_LEVEL", "DEBUG")
    os.environ.setdefault("HARBORAI_CACHE_ENABLED", "false")
    
    # 禁用真实API测试（除非明确启用）
    if not os.getenv("ENABLE_REAL_API_TESTS"):
        os.environ["ENABLE_REAL_API_TESTS"] = "false"
    
    # 设置测试报告目录
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)


def pytest_collection_modifyitems(config, items):
    """修改测试收集项"""
    # 为真实API测试添加跳过标记
    skip_real_api = pytest.mark.skip(reason="需要设置 ENABLE_REAL_API_TESTS=true 环境变量")
    
    for item in items:
        # 跳过真实API测试
        if "real_api" in item.keywords and not os.getenv("ENABLE_REAL_API_TESTS", "").lower() == "true":
            item.add_marker(skip_real_api)
        
        # 为慢速测试添加标记
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.timeout(60))
        
        # 为性能测试添加标记
        if "performance" in item.keywords:
            item.add_marker(pytest.mark.timeout(120))


def pytest_runtest_setup(item):
    """测试运行前设置"""
    # 检查真实API测试的前置条件
    if "real_api" in item.keywords:
        if not os.getenv("ENABLE_REAL_API_TESTS", "").lower() == "true":
            pytest.skip("真实API测试被禁用")
        
        # 检查必要的API密钥
        required_keys = []
        if "deepseek" in item.keywords:
            required_keys.append("DEEPSEEK_API_KEY")
        if "ernie" in item.keywords:
            required_keys.append("WENXIN_API_KEY")
        if "doubao" in item.keywords:
            required_keys.append("DOUBAO_API_KEY")
        
        for key in required_keys:
            if not os.getenv(key):
                pytest.skip(f"缺少必要的环境变量: {key}")


# ==================== 全局夹具 ====================

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_config() -> Dict[str, Any]:
    """测试配置"""
    return {
        "timeout": 30,
        "max_retries": 3,
        "enable_real_api": os.getenv("ENABLE_REAL_API_TESTS", "").lower() == "true",
        "log_level": "DEBUG",
        "cache_enabled": False,
        "performance_monitoring": True,
        "vendors": {
            "deepseek": {
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                "models": ["deepseek-chat", "deepseek-r1"]
            },
            "ernie": {
                "api_key": os.getenv("WENXIN_API_KEY"),
                "base_url": os.getenv("WENXIN_BASE_URL", "https://qianfan.baidubce.com/v2"),
                "models": ["ernie-3.5-8k", "ernie-4.0-turbo-8k", "ernie-x1-turbo-32k"]
            },
            "doubao": {
                "api_key": os.getenv("DOUBAO_API_KEY"),
                "base_url": os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
                "models": ["doubao-1-5-pro-32k-character-250715", "doubao-seed-1-6-250615"]
            }
        }
    }


@pytest.fixture
def mock_env_vars(monkeypatch):
    """模拟环境变量"""
    test_vars = {
        "HARBORAI_ENV": "test",
        "HARBORAI_LOG_LEVEL": "DEBUG",
        "HARBORAI_CACHE_ENABLED": "false",
        "DEEPSEEK_API_KEY": "test-deepseek-key",
        "WENXIN_API_KEY": "test-wenxin-key",
        "DOUBAO_API_KEY": "test-doubao-key"
    }
    
    for key, value in test_vars.items():
        monkeypatch.setenv(key, value)
    
    return test_vars


@pytest.fixture
def clean_env(monkeypatch):
    """清理环境变量"""
    # 删除可能影响测试的环境变量
    env_vars_to_clean = [
        "DEEPSEEK_API_KEY",
        "WENXIN_API_KEY",
        "DOUBAO_API_KEY",
        "HARBORAI_CACHE_ENABLED",
        "HARBORAI_LOG_LEVEL"
    ]
    
    for var in env_vars_to_clean:
        monkeypatch.delenv(var, raising=False)


# ==================== 测试数据夹具 ====================

@pytest.fixture
def mock_harborai_client(monkeypatch):
    """Mock HarborAI客户端夹具 - 使用真实客户端但mock底层API调用"""
    from unittest.mock import Mock, patch
    from harborai import HarborAI
    
    # 设置测试环境变量
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("WENXIN_API_KEY", "test-key")
    monkeypatch.setenv("DOUBAO_API_KEY", "test-key")
    
    # 创建真实的HarborAI客户端
    client = HarborAI()
    
    # Mock底层的HTTP请求，但保留参数处理逻辑
    with patch.object(client.client_manager, 'chat_completion_sync_with_fallback') as mock_completion:
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="This is a synchronous response.",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=17,
            total_tokens=27
        )
        mock_completion.return_value = mock_response
        
        # 保存原始的create方法以便测试可以检查调用参数
        original_create = client.chat.completions.create
        
        def create_wrapper(*args, **kwargs):
            # 调用原始方法（包含参数过滤逻辑）
            result = original_create(*args, **kwargs)
            # 保存调用参数供测试检查
            create_wrapper.call_args = (args, kwargs)
            return result
        
        client.chat.completions.create = create_wrapper
        
        yield client

@pytest.fixture
def mock_harborai_async_client(monkeypatch):
    """Mock HarborAI异步客户端夹具"""
    from unittest.mock import Mock, patch, AsyncMock
    from harborai import HarborAI
    
    # 设置测试环境变量
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("WENXIN_API_KEY", "test-key")
    monkeypatch.setenv("DOUBAO_API_KEY", "test-key")
    
    # 创建真实的HarborAI客户端
    client = HarborAI()
    
    # Mock底层的异步HTTP请求
    with patch.object(client.client_manager, 'chat_completion_with_fallback') as mock_completion:
        # 配置mock异步响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="异步推理模型的深度分析结果...",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=10,
            completion_tokens=17,
            total_tokens=27
        )
        
        async def async_response(*args, **kwargs):
            return mock_response
        
        mock_completion.side_effect = async_response
        
        # 保存原始的acreate方法
        original_acreate = client.chat.completions.acreate
        
        async def acreate_wrapper(*args, **kwargs):
            # 调用原始方法（包含参数过滤逻辑）
            result = await original_acreate(*args, **kwargs)
            # 保存调用参数供测试检查
            acreate_wrapper.call_args = (args, kwargs)
            return result
        
        client.chat.completions.acreate = acreate_wrapper
        
        yield client

@pytest.fixture
def sample_messages():
    """示例消息"""
    return [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ]


@pytest.fixture
def complex_messages():
    """复杂消息"""
    return [
        {"role": "system", "content": "你是一个专业的数学老师，擅长解释复杂的数学概念。"},
        {"role": "user", "content": "请解释什么是微积分，并给出一个实际应用的例子。"},
        {"role": "assistant", "content": "微积分是数学的一个重要分支，主要研究函数的变化率和累积量。它包括微分和积分两个主要部分。"},
        {"role": "user", "content": "能详细解释一下导数的概念吗？"}
    ]


# reasoning_test_messages 夹具已在 data_fixtures.py 中定义


# ==================== 性能测试夹具 ====================

@pytest.fixture
def performance_config():
    """性能测试配置"""
    return {
        "max_response_time": 2.0,  # 最大响应时间（秒）
        "max_memory_usage": 100.0,  # 最大内存使用（MB）
        "max_cpu_usage": 80.0,  # 最大CPU使用率（%）
        "min_throughput": 1.0,  # 最小吞吐量（req/s）
        "max_error_rate": 0.05,  # 最大错误率（5%）
        "monitoring_interval": 0.1  # 监控间隔（秒）
    }


# ==================== 错误处理夹具 ====================

@pytest.fixture
def mock_api_errors():
    """模拟API错误"""
    return {
        "rate_limit": HarborAIError("Rate limit exceeded", status_code=429),
        "auth_error": HarborAIError("Invalid API key", status_code=401),
        "not_found": HarborAIError("Model not found", status_code=404),
        "server_error": HarborAIError("Internal server error", status_code=500),
        "timeout": HarborAIError("Request timeout", status_code=408),
        "bad_request": HarborAIError("Bad request", status_code=400)
    }


# ==================== 测试工具函数 ====================

def assert_valid_response(response: Dict[str, Any]):
    """验证响应格式"""
    assert "id" in response
    assert "object" in response
    assert "model" in response
    assert "choices" in response
    assert "usage" in response
    
    # 验证choices
    assert len(response["choices"]) > 0
    choice = response["choices"][0]
    assert "index" in choice
    assert "message" in choice
    assert "finish_reason" in choice
    
    # 验证message
    message = choice["message"]
    assert "role" in message
    assert "content" in message
    
    # 验证usage
    usage = response["usage"]
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


def assert_valid_stream_chunk(chunk: Dict[str, Any]):
    """验证流式响应块格式"""
    assert "id" in chunk
    assert "object" in chunk
    assert "model" in chunk
    assert "choices" in chunk
    
    if len(chunk["choices"]) > 0:
        choice = chunk["choices"][0]
        assert "index" in choice
        assert "delta" in choice


def assert_reasoning_response(response: Dict[str, Any]):
    """验证推理响应格式"""
    assert_valid_response(response)
    
    # 验证推理特定字段
    message = response["choices"][0]["message"]
    if "reasoning_content" in message:
        assert message["reasoning_content"] is not None
        assert len(message["reasoning_content"].strip()) > 0
    
    # 验证推理token统计
    usage = response["usage"]
    if "reasoning_tokens" in usage:
        assert usage["reasoning_tokens"] >= 0


# ==================== 测试标记辅助函数 ====================

def requires_real_api(vendor: str = None):
    """需要真实API的测试装饰器"""
    def decorator(func):
        marks = [pytest.mark.real_api]
        if vendor:
            marks.append(getattr(pytest.mark, vendor))
        
        for mark in marks:
            func = mark(func)
        
        return func
    return decorator


def slow_test(timeout: int = 60):
    """慢速测试装饰器"""
    def decorator(func):
        func = pytest.mark.slow(func)
        func = pytest.mark.timeout(timeout)(func)
        return func
    return decorator


def performance_test(max_time: float = 2.0):
    """性能测试装饰器"""
    def decorator(func):
        func = pytest.mark.performance(func)
        func = pytest.mark.timeout(int(max_time * 2))(func)
        return func
    return decorator