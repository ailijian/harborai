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
import json
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
                "models": ["deepseek-chat", "deepseek-reasoner"]
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
    """Mock HarborAI客户端夹具"""
    from unittest.mock import Mock, patch, AsyncMock
    from harborai import HarborAI
    
    # 设置测试环境变量
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("WENXIN_API_KEY", "test-key")
    monkeypatch.setenv("DOUBAO_API_KEY", "test-key")
    
    # 创建Mock客户端
    mock_client = Mock(spec=HarborAI)
    
    # 设置chat.completions的Mock结构
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    
    # 创建默认的mock响应
    default_response = Mock()
    default_response.id = "chatcmpl-test-123"
    default_response.object = "chat.completion"
    default_response.created = 1234567890
    default_response.model = "gpt-4"
    default_response.choices = [Mock(
        index=0,
        message=Mock(
            role="assistant",
            content="量子纠缠是指两个或多个粒子之间存在的一种量子力学关联，即使它们相距很远，对其中一个粒子的测量会瞬间影响另一个粒子的状态。"
        ),
        finish_reason="stop"
    )]
    default_response.usage = Mock(
        prompt_tokens=25,
        completion_tokens=45,
        total_tokens=70
    )
    
    # 保存原始的create方法引用
    original_create = None
    
    # 创建流式响应chunks
    stream_chunks = [
        Mock(
            id="chatcmpl-test-stream",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4",
            choices=[Mock(
                index=0,
                delta=Mock(role="assistant", content=""),
                finish_reason=None
            )]
        ),
        Mock(
            id="chatcmpl-test-stream",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4",
            choices=[Mock(
                index=0,
                delta=Mock(content="量子纠缠"),
                finish_reason=None
            )]
        ),
        Mock(
            id="chatcmpl-test-stream",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4",
            choices=[Mock(
                index=0,
                delta=Mock(content="是指两个或多个粒子之间存在的一种量子力学关联。"),
                finish_reason=None
            )]
        ),
        Mock(
            id="chatcmpl-test-stream",
            object="chat.completion.chunk",
            created=1234567890,
            model="gpt-4",
            choices=[Mock(
                index=0,
                delta=Mock(),
                finish_reason="stop"
            )]
        )
    ]
    
    # 设置同步方法 - 根据参数决定返回流式还是普通响应
    def create_response(*args, **kwargs):
        # 检查是否是结构化输出请求
        response_format = kwargs.get('response_format')
        if response_format and response_format.get('type') == 'json_object':
            # 根据不同的测试场景返回相应的JSON数据
            messages = kwargs.get('messages', [])
            user_content = ""
            system_content = ""
            for msg in messages:
                if msg.get('role') == 'user':
                    user_content = msg.get('content', '')
                elif msg.get('role') == 'system':
                    system_content = msg.get('content', '')
            
            # 根据消息内容确定返回的JSON结构
            if "人员信息" in user_content or "Person模型" in system_content:
                json_content = {
                    "name": "张三",
                    "age": 25,
                    "email": "zhangsan@example.com",
                    "phone": "+86-13800138000",
                    "address": {
                        "street": "中关村大街1号",
                        "city": "北京",
                        "state": "北京市",
                        "zip_code": "100080",
                        "country": "China"
                    },
                    "is_active": True
                }
            elif "任务" in user_content or "Task" in system_content:
                json_content = {
                    "id": "task-001",
                    "title": "测试任务",
                    "description": "这是一个测试任务",
                    "status": "pending",
                    "priority": "medium",
                    "assignee": {
                        "name": "李四",
                        "age": 30,
                        "email": "lisi@example.com",
                        "is_active": True
                    },
                    "due_date": "2024-12-31",
                    "created_at": "2024-01-01T00:00:00",
                    "tags": ["测试", "开发"],
                    "metadata": {"priority_score": 5}
                }
            elif "项目" in user_content or "Project" in system_content:
                json_content = {
                    "project_name": "测试项目",
                    "total_tasks": 10,
                    "completed_tasks": 7,
                    "completion_rate": 0.7,
                    "team_members": [
                        {
                            "name": "张三",
                            "age": 25,
                            "email": "zhangsan@example.com",
                            "is_active": True
                        }
                    ],
                    "active_tasks": [
                        {
                            "id": "task-001",
                            "title": "活跃任务",
                            "status": "in_progress",
                            "priority": "high",
                            "created_at": "2024-01-01T00:00:00",
                            "tags": [],
                            "metadata": {}
                        }
                    ],
                    "project_metadata": {"budget": 100000, "active": True}
                }
            else:
                # 默认简单JSON结构
                json_content = {
                    "name": "张三",
                    "age": 25,
                    "is_student": False
                }
            
            # 创建结构化输出响应
            structured_response = Mock()
            structured_response.id = "chatcmpl-structured-123"
            structured_response.object = "chat.completion"
            structured_response.created = 1234567890
            structured_response.model = kwargs.get('model', 'deepseek-chat')
            structured_response.choices = [Mock(
                index=0,
                message=Mock(
                    role="assistant",
                    content=json.dumps(json_content, ensure_ascii=False, indent=2)
                ),
                finish_reason="stop"
            )]
            structured_response.usage = Mock(
                prompt_tokens=50,
                completion_tokens=100,
                total_tokens=150
            )
            return structured_response
        
        # 对于推理模型，返回包含reasoning_content的响应（推理模型不支持流式）
        if any(model in kwargs.get('model', '') for model in ['deepseek-reasoner', 'deepseek-reasoner', 'o1-preview', 'o1-mini']):
            # 检查是否是不支持的模型测试
            if kwargs.get('model') == "unsupported-reasoning-model":
                from harborai.exceptions import ModelNotSupportedError
                raise ModelNotSupportedError(f"Model {kwargs.get('model')} is not supported")
            
            reasoning_response = Mock()
            reasoning_response.id = "chatcmpl-reasoning-123"
            reasoning_response.object = "chat.completion"
            reasoning_response.created = 1234567890
            reasoning_response.model = kwargs.get('model', 'deepseek-reasoner')
            
            # 根据不同的测试场景返回不同的响应
            messages = kwargs.get('messages', [])
            user_content = ""
            if messages:
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_content = msg.get('content', '')
                        break
            
            # 复杂问题解决测试
            if "分布式缓存系统" in user_content or "complex_problem_solving" in str(kwargs):
                content = """
        这是一个复杂的多步骤问题，让我逐步分析：
        
        **第一步：问题理解**
        首先需要明确问题的核心要求和约束条件...
        
        **第二步：方案分析**
        考虑以下几种可能的解决方案：
        1. 方案A：优点是...，缺点是...
        2. 方案B：优点是...，缺点是...
        3. 方案C：优点是...，缺点是...
        
        **第三步：最优解选择**
        综合考虑各种因素，我认为方案B是最优的，因为...
        
        **第四步：实施建议**
        具体的实施步骤如下：
        1. 准备阶段：...
        2. 执行阶段：...
        3. 验证阶段：...
        
        **结论**
        基于以上分析，最终建议是...
        """
                completion_tokens = 1200
            # Token效率测试
            elif "token_efficiency" in str(kwargs) or "效率" in user_content:
                content = "这是一个高效的推理响应，包含了深度思考和逻辑推理过程。通过量子纠缠原理分析，我们可以得出精确的结论。"
                completion_tokens = 800
            # 系统消息转换测试 - 使用默认响应
            elif any(msg.get('role') == 'system' for msg in messages):
                content = "让我思考一下这个问题。经过分析，我认为这是一个很好的问题。通过推理，我可以得出以下结论：这是一个测试响应。"
                completion_tokens = 20
            # 医疗AI分析测试
            elif "医疗" in user_content or "人工智能" in user_content:
                content = """
        这是一个需要深入分析的复杂问题，让我从多个维度来考虑：
        
        **应用前景分析：**
        1. 诊断辅助：AI可以通过图像识别技术...
        2. 药物研发：机器学习算法能够...
        3. 个性化治疗：基于大数据分析...
        
        **技术挑战：**
        1. 数据质量和隐私保护
        2. 算法可解释性
        3. 监管合规性
        
        **综合评估：**
        考虑到技术发展趋势和实际应用需求...
        """
                completion_tokens = 800
            else:
                # 默认推理响应
                content = "让我思考一下这个问题。经过分析，我认为这是一个很好的问题。通过推理，我可以得出以下结论：这是一个测试响应。"
                completion_tokens = 20
            
            reasoning_response.choices = [Mock(
                index=0,
                message=Mock(
                    role="assistant",
                    content=content,
                    reasoning_content="这是推理过程的详细内容..."
                ),
                finish_reason="stop"
            )]
            reasoning_response.usage = Mock(
                prompt_tokens=10,
                completion_tokens=completion_tokens,
                total_tokens=10 + completion_tokens
            )
            return reasoning_response
        
        if kwargs.get('stream', False):
            return iter(stream_chunks)
        return default_response
    
    # 设置chat.completions.create的mock行为
    def create_side_effect(*args, **kwargs):
        # 检查是否已经设置了return_value（测试用例内部设置）
        if hasattr(mock_client.chat.completions.create, 'return_value') and mock_client.chat.completions.create.return_value is not None:
            # 如果测试用例已经设置了return_value，直接返回它
            return mock_client.chat.completions.create.return_value
        
        # 如果是流式请求
        if kwargs.get('stream', False):
            return iter(stream_chunks)
        
        # 根据模型和消息内容创建响应
        model = kwargs.get('model', 'test-model')
        messages = kwargs.get('messages', [])
        
        return create_response(model, messages, kwargs)
    
    mock_client.chat.completions.create = Mock(side_effect=create_side_effect)
    
    # 设置异步方法
    async def async_create(*args, **kwargs):
        # 调用底层方法以便测试可以验证调用
        if hasattr(mock_client, 'client_manager') and hasattr(mock_client.client_manager, 'chat_completion_with_fallback'):
            result = await mock_client.client_manager.chat_completion_with_fallback(*args, **kwargs)
            if result:
                return result
        
        if kwargs.get('stream', False):
            # 异步流式响应
            async def async_stream():
                for chunk in stream_chunks:
                    yield chunk
            return async_stream()
        return default_response
    
    mock_client.chat.completions.acreate = AsyncMock(side_effect=async_create)
    
    # 添加client_manager属性
    mock_client.client_manager = Mock()
    
    # 为fallback测试配置特殊的side_effect
    call_count = 0
    def fallback_side_effect(*args, **kwargs):
        nonlocal call_count
        model = kwargs.get('model', '')
        
        # 检查特定的错误测试场景
        if model == 'deepseek-reasoner-ultra':
            from harborai.exceptions import ModelNotSupportedError
            raise ModelNotSupportedError("Model 'deepseek-reasoner-ultra' not supported")
        elif 'timeout' in str(kwargs) or (model == 'deepseek-reasoner' and kwargs.get('timeout')):
            from harborai.exceptions import NetworkError
            raise NetworkError("Connection timeout")
        elif model == 'deepseek-reasoner' and any('速率限制' in str(msg.get('content', '')) for msg in kwargs.get('messages', []) if isinstance(msg, dict)):
            from harborai.exceptions import RateLimitError
            raise RateLimitError("Rate limit exceeded for deepseek-reasoner")
        elif model == 'deepseek-reasoner' and any('测试降级' in str(msg.get('content', '')) for msg in kwargs.get('messages', []) if isinstance(msg, dict)):
            # fallback测试场景：第一次调用推理模型失败
            call_count += 1
            if call_count == 1:
                from harborai.exceptions import ModelNotSupportedError
                raise ModelNotSupportedError("Model 'deepseek-reasoner' not supported")
        elif model == 'deepseek-chat':
            # 降级到常规模型成功
            fallback_response = Mock()
            fallback_response.choices = [Mock(
                message=Mock(
                    content="使用常规模型的响应",
                    role="assistant"
                ),
                finish_reason="stop"
            )]
            fallback_response.usage = Mock(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
            return fallback_response
        else:
            return default_response
    
    mock_client.client_manager.chat_completion_sync_with_fallback = Mock(side_effect=fallback_side_effect)
    mock_client.client_manager.chat_completion_with_fallback = AsyncMock(return_value=default_response)
    
    yield mock_client

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