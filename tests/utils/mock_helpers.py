# -*- coding: utf-8 -*-
"""
Mock辅助函数模块

功能：提供各种Mock辅助功能，包括API响应Mock、错误模拟、流式响应Mock等
作者：HarborAI测试团队
创建时间：2024
"""

import json
import time
import random
import asyncio
from typing import Dict, List, Any, Optional, Union, Iterator, AsyncIterator, Callable
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
import logging


@dataclass
class MockResponse:
    """模拟响应数据类
    
    功能：标准化模拟响应格式
    参数：
        content: 响应内容
        status_code: HTTP状态码
        headers: 响应头
        delay: 模拟延迟（秒）
        error: 错误信息
    """
    content: Any = None
    status_code: int = 200
    headers: Dict[str, str] = field(default_factory=dict)
    delay: float = 0.0
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """字典式访问方法"""
        # 优先检查content中的键，然后才检查属性
        if isinstance(self.content, dict) and key in self.content:
            return self.content[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            return default
    
    def __getitem__(self, key: str) -> Any:
        """支持[]访问"""
        # 优先检查content中的键，然后才检查属性
        if isinstance(self.content, dict) and key in self.content:
            return self.content[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            raise KeyError(f"Key '{key}' not found")


@dataclass
class StreamChunk:
    """流式响应块数据类
    
    功能：表示流式响应的单个数据块
    参数：
        data: 块数据
        chunk_type: 块类型（data, error, done）
        timestamp: 时间戳
        metadata: 元数据
    """
    data: Any
    chunk_type: str = "data"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class APIResponseMocker:
    """API响应Mock器
    
    功能：模拟各种API响应场景
    假设：API响应遵循标准格式
    不确定点：某些API可能有特殊响应格式
    验证方法：pytest tests/test_mock_helpers.py::TestAPIResponseMocker
    """
    
    def __init__(self, default_delay: float = 0.1):
        """初始化Mock器
        
        参数：
            default_delay: 默认响应延迟
        """
        self.default_delay = default_delay
        self.response_templates = self._init_response_templates()
        self.call_history = []
    
    def _init_response_templates(self) -> Dict[str, Dict]:
        """初始化响应模板"""
        return {
            'chat_completion': {
                'id': 'chatcmpl-{id}',
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': 'deepseek-chat',
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': '{content}'
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 20,
                    'total_tokens': 30
                }
            },
            'embedding': {
                'object': 'list',
                'data': [{
                    'object': 'embedding',
                    'embedding': [],
                    'index': 0
                }],
                'model': 'text-embedding-ada-002',
                'usage': {
                    'prompt_tokens': 8,
                    'total_tokens': 8
                }
            },
            'error_response': {
                'error': {
                    'message': '{message}',
                    'type': '{error_type}',
                    'code': '{error_code}'
                }
            }
        }
    
    def create_chat_completion_response(
        self, 
        content: str = "这是一个模拟的AI响应",
        model: str = "deepseek-chat",
        **kwargs
    ) -> MockResponse:
        """创建聊天完成响应
        
        功能：生成标准的聊天完成API响应
        参数：
            content: 响应内容
            model: 模型名称
            **kwargs: 额外参数
        返回：MockResponse对象
        """
        template = self.response_templates['chat_completion'].copy()
        template['id'] = template['id'].format(id=self._generate_id())
        template['model'] = model
        template['choices'][0]['message']['content'] = content
        
        # 处理额外参数
        if 'finish_reason' in kwargs:
            template['choices'][0]['finish_reason'] = kwargs['finish_reason']
        if 'usage' in kwargs:
            template['usage'].update(kwargs['usage'])
        
        return MockResponse(
            content=template,
            status_code=200,
            delay=kwargs.get('delay', self.default_delay)
        )
    
    def create_embedding_response(
        self, 
        embeddings: List[List[float]] = None,
        model: str = "deepseek-embedding",
        **kwargs
    ) -> MockResponse:
        """创建嵌入响应
        
        功能：生成标准的嵌入API响应
        参数：
            embeddings: 嵌入向量列表
            model: 模型名称
            **kwargs: 额外参数
        返回：MockResponse对象
        """
        if embeddings is None:
            # 生成随机嵌入向量
            dimension = kwargs.get('dimension', 1536)
            embeddings = [[random.uniform(-1, 1) for _ in range(dimension)]]
        
        template = self.response_templates['embedding'].copy()
        template['model'] = model
        template['data'][0]['embedding'] = embeddings[0]
        
        return MockResponse(
            content=template,
            status_code=200,
            delay=kwargs.get('delay', self.default_delay)
        )
    
    def create_error_response(
        self,
        error_type: str = "invalid_request_error",
        message: str = "Invalid request",
        error_code: str = "400",
        status_code: int = 400,
        **kwargs
    ) -> MockResponse:
        """创建错误响应
        
        功能：生成标准的错误API响应
        参数：
            error_type: 错误类型
            message: 错误消息
            error_code: 错误代码
            status_code: HTTP状态码
            **kwargs: 额外参数
        返回：MockResponse对象
        """
        template = self.response_templates['error_response'].copy()
        template['error']['message'] = message
        template['error']['type'] = error_type
        template['error']['code'] = error_code
        
        return MockResponse(
            content=template,
            status_code=status_code,
            delay=kwargs.get('delay', self.default_delay),
            error=Exception(message)
        )
    
    def create_streaming_response(
        self,
        content: str = "这是一个流式响应",
        chunk_size: int = 10,
        chunk_delay: float = 0.1,
        **kwargs
    ) -> Iterator[StreamChunk]:
        """创建流式响应
        
        功能：生成流式API响应
        参数：
            content: 完整内容
            chunk_size: 每个块的大小
            chunk_delay: 块之间的延迟
            **kwargs: 额外参数
        返回：StreamChunk迭代器
        """
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        for i, chunk_content in enumerate(chunks):
            if chunk_delay > 0:
                time.sleep(chunk_delay)
            
            chunk_data = {
                'id': f'chatcmpl-{self._generate_id()}',
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': kwargs.get('model', 'deepseek-chat'),
                'choices': [{
                    'index': 0,
                    'delta': {
                        'content': chunk_content
                    },
                    'finish_reason': None if i < len(chunks) - 1 else 'stop'
                }]
            }
            
            yield StreamChunk(
                data=chunk_data,
                chunk_type="data",
                metadata={'chunk_index': i, 'total_chunks': len(chunks)}
            )
        
        # 发送结束标记
        yield StreamChunk(
            data={'type': 'done'},
            chunk_type="done",
            metadata={'total_chunks': len(chunks)}
        )
    
    async def create_async_streaming_response(
        self,
        content: str = "这是一个异步流式响应",
        chunk_size: int = 10,
        chunk_delay: float = 0.1,
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """创建异步流式响应
        
        功能：生成异步流式API响应
        参数：
            content: 完整内容
            chunk_size: 每个块的大小
            chunk_delay: 块之间的延迟
            **kwargs: 额外参数
        返回：StreamChunk异步迭代器
        """
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        for i, chunk_content in enumerate(chunks):
            if chunk_delay > 0:
                await asyncio.sleep(chunk_delay)
            
            chunk_data = {
                'id': f'chatcmpl-{self._generate_id()}',
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': kwargs.get('model', 'deepseek-chat'),
                'choices': [{
                    'index': 0,
                    'delta': {
                        'content': chunk_content
                    },
                    'finish_reason': None if i < len(chunks) - 1 else 'stop'
                }]
            }
            
            yield StreamChunk(
                data=chunk_data,
                chunk_type="data",
                metadata={'chunk_index': i, 'total_chunks': len(chunks)}
            )
        
        # 发送结束标记
        yield StreamChunk(
            data={'type': 'done'},
            chunk_type="done",
            metadata={'total_chunks': len(chunks)}
        )
    
    def _generate_id(self, length: int = 8) -> str:
        """生成随机ID"""
        import string
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    def record_call(self, method: str, args: tuple, kwargs: dict, response: MockResponse):
        """记录调用历史"""
        self.call_history.append({
            'method': method,
            'args': args,
            'kwargs': kwargs,
            'response': response,
            'timestamp': time.time()
        })
    
    def get_call_history(self) -> List[Dict]:
        """获取调用历史"""
        return self.call_history.copy()
    
    def clear_call_history(self):
        """清空调用历史"""
        self.call_history.clear()


class ErrorSimulator:
    """错误模拟器
    
    功能：模拟各种错误场景
    假设：错误类型遵循标准分类
    不确定点：某些错误可能需要特殊处理
    验证方法：pytest tests/test_mock_helpers.py::TestErrorSimulator
    """
    
    def __init__(self):
        """初始化错误模拟器"""
        self.error_scenarios = self._init_error_scenarios()
    
    def _init_error_scenarios(self) -> Dict[str, Dict]:
        """初始化错误场景"""
        return {
            'authentication_error': {
                'status_code': 401,
                'error_type': 'authentication_error',
                'message': 'Invalid API key provided',
                'exception_class': 'AuthenticationError'
            },
            'rate_limit_error': {
                'status_code': 429,
                'error_type': 'rate_limit_error',
                'message': 'Rate limit exceeded',
                'exception_class': 'RateLimitError',
                'retry_after': 60
            },
            'invalid_request_error': {
                'status_code': 400,
                'error_type': 'invalid_request_error',
                'message': 'Invalid request parameters',
                'exception_class': 'InvalidRequestError'
            },
            'server_error': {
                'status_code': 500,
                'error_type': 'server_error',
                'message': 'Internal server error',
                'exception_class': 'ServerError'
            },
            'timeout_error': {
                'status_code': 408,
                'error_type': 'timeout_error',
                'message': 'Request timeout',
                'exception_class': 'TimeoutError'
            },
            'network_error': {
                'status_code': 0,
                'error_type': 'network_error',
                'message': 'Network connection failed',
                'exception_class': 'NetworkError'
            }
        }
    
    def simulate_error(
        self, 
        error_type: str, 
        custom_message: str = None,
        **kwargs
    ) -> Exception:
        """模拟特定类型的错误
        
        功能：根据错误类型生成对应的异常
        参数：
            error_type: 错误类型
            custom_message: 自定义错误消息
            **kwargs: 额外参数
        返回：Exception对象
        """
        if error_type not in self.error_scenarios:
            raise ValueError(f"不支持的错误类型：{error_type}")
        
        scenario = self.error_scenarios[error_type]
        message = custom_message or scenario['message']
        
        # 创建异常对象
        exception = Exception(message)
        exception.error_type = error_type
        exception.status_code = scenario['status_code']
        
        # 添加额外属性
        for key, value in kwargs.items():
            setattr(exception, key, value)
        
        if 'retry_after' in scenario:
            exception.retry_after = scenario['retry_after']
        
        return exception
    
    def simulate_intermittent_error(
        self, 
        error_type: str, 
        failure_rate: float = 0.3,
        **kwargs
    ) -> Callable:
        """模拟间歇性错误
        
        功能：创建一个函数，按指定概率抛出错误
        参数：
            error_type: 错误类型
            failure_rate: 失败率（0-1）
            **kwargs: 额外参数
        返回：可能抛出错误的函数
        """
        def error_function(*args, **func_kwargs):
            if random.random() < failure_rate:
                raise self.simulate_error(error_type, **kwargs)
            return "成功响应"
        
        return error_function
    
    def simulate_cascading_errors(
        self, 
        error_sequence: List[str],
        **kwargs
    ) -> Iterator[Exception]:
        """模拟级联错误
        
        功能：按顺序生成一系列错误
        参数：
            error_sequence: 错误类型序列
            **kwargs: 额外参数
        返回：Exception迭代器
        """
        for error_type in error_sequence:
            yield self.simulate_error(error_type, **kwargs)
    
    def create_authentication_error(self, message: str = None) -> MockResponse:
        """创建认证错误响应
        
        功能：创建认证相关的错误响应
        参数：
            message: 自定义错误消息
        返回：认证错误响应
        边界条件：处理空消息
        假设：错误消息格式有效
        不确定点：某些认证错误可能需要特殊处理
        验证方法：pytest tests/test_mock_helpers.py::TestErrorSimulator::test_create_authentication_error
        """
        scenario = self.error_scenarios['authentication_error']
        error_content = {
            'error': {
                'message': message or scenario['message'],
                'type': scenario['error_type'],
                'code': str(scenario['status_code'])
            }
        }
        return MockResponse(
            content=error_content,
            status_code=scenario['status_code']
        )
    
    def create_rate_limit_error(self, message: str = None, retry_after: int = 60) -> MockResponse:
        """创建速率限制错误响应
        
        功能：创建速率限制相关的错误响应
        参数：
            message: 自定义错误消息
            retry_after: 重试等待时间（秒）
        返回：速率限制错误响应
        边界条件：处理空消息和无效重试时间
        假设：错误消息格式有效
        不确定点：某些速率限制错误可能需要特殊处理
        验证方法：pytest tests/test_mock_helpers.py::TestErrorSimulator::test_create_rate_limit_error
        """
        scenario = self.error_scenarios['rate_limit_error']
        error_content = {
            'error': {
                'message': message or scenario['message'],
                'type': scenario['error_type'],
                'code': str(scenario['status_code']),
                'retry_after': retry_after
            }
        }
        return MockResponse(
            content=error_content,
            status_code=scenario['status_code'],
            headers={'Retry-After': str(retry_after)}
        )


class HarborAIMocker:
    """HarborAI客户端Mock器
    
    功能：专门用于Mock HarborAI客户端的各种方法
    假设：HarborAI客户端接口稳定
    不确定点：某些方法可能有特殊行为
    验证方法：pytest tests/test_mock_helpers.py::TestHarborAIMocker
    """
    
    def __init__(self):
        """初始化HarborAI Mock器"""
        self.response_mocker = APIResponseMocker()
        self.error_simulator = ErrorSimulator()
        self.mock_client = None
    
    def create_mock_client(self, **config) -> Mock:
        """创建Mock客户端
        
        功能：创建完整的HarborAI客户端Mock
        参数：
            **config: 配置参数
        返回：Mock客户端对象
        """
        mock_client = Mock()
        
        # Mock chat.completions.create方法
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = Mock()
        
        # 设置默认返回值
        default_response = self.response_mocker.create_chat_completion_response()
        mock_client.chat.completions.create.return_value = default_response.content
        
        # Mock embeddings.create方法
        mock_client.embeddings = Mock()
        mock_client.embeddings.create = Mock()
        
        default_embedding_response = self.response_mocker.create_embedding_response()
        mock_client.embeddings.create.return_value = default_embedding_response.content
        
        # Mock其他方法
        mock_client.models = Mock()
        mock_client.models.list = Mock()
        mock_client.models.list.return_value = {
            'object': 'list',
            'data': [
                {'id': 'deepseek-chat', 'object': 'model'},
                {'id': 'ernie-4.0-8k', 'object': 'model'},
                {'id': 'doubao-pro-32k', 'object': 'model'}
            ]
        }
        
        self.mock_client = mock_client
        return mock_client
    
    def setup_chat_completion_mock(
        self, 
        mock_client: Mock,
        responses: List[Union[str, MockResponse, Exception]] = None,
        **kwargs
    ):
        """设置聊天完成Mock
        
        功能：配置聊天完成方法的Mock行为
        参数：
            mock_client: Mock客户端
            responses: 响应列表（字符串、MockResponse或异常）
            **kwargs: 额外参数
        """
        if responses is None:
            responses = ["这是一个默认的Mock响应"]
        
        def side_effect(*args, **call_kwargs):
            # 记录调用
            self.response_mocker.record_call('chat.completions.create', args, call_kwargs, None)
            
            # 处理响应
            if len(responses) == 1:
                response = responses[0]
            else:
                # 循环使用响应列表
                call_count = mock_client.chat.completions.create.call_count
                response = responses[call_count % len(responses)]
            
            if isinstance(response, Exception):
                raise response
            elif isinstance(response, MockResponse):
                if response.delay > 0:
                    time.sleep(response.delay)
                if response.error:
                    raise response.error
                return response.content
            else:
                # 字符串响应
                mock_response = self.response_mocker.create_chat_completion_response(
                    content=response, **kwargs
                )
                if mock_response.delay > 0:
                    time.sleep(mock_response.delay)
                return mock_response.content
        
        mock_client.chat.completions.create.side_effect = side_effect
    
    def setup_streaming_mock(
        self, 
        mock_client: Mock,
        content: str = "这是一个流式Mock响应",
        **kwargs
    ):
        """设置流式响应Mock
        
        功能：配置流式响应的Mock行为
        参数：
            mock_client: Mock客户端
            content: 流式内容
            **kwargs: 额外参数
        """
        def streaming_side_effect(*args, **call_kwargs):
            # 检查是否请求流式响应
            if call_kwargs.get('stream', False):
                return self.response_mocker.create_streaming_response(
                    content=content, **kwargs
                )
            else:
                # 非流式响应
                mock_response = self.response_mocker.create_chat_completion_response(
                    content=content, **kwargs
                )
                return mock_response.content
        
        mock_client.chat.completions.create.side_effect = streaming_side_effect
    
    def setup_error_scenarios(
        self, 
        mock_client: Mock,
        error_scenarios: Dict[str, Any]
    ):
        """设置错误场景
        
        功能：配置各种错误场景的Mock行为
        参数：
            mock_client: Mock客户端
            error_scenarios: 错误场景配置
        """
        original_side_effect = mock_client.chat.completions.create.side_effect
        
        def error_side_effect(*args, **call_kwargs):
            # 检查是否触发错误场景
            for trigger, error_config in error_scenarios.items():
                if self._should_trigger_error(trigger, args, call_kwargs):
                    error = self.error_simulator.simulate_error(
                        error_config['type'], 
                        error_config.get('message'),
                        **error_config.get('kwargs', {})
                    )
                    raise error
            
            # 如果没有触发错误，执行原始行为
            if original_side_effect:
                return original_side_effect(*args, **call_kwargs)
            else:
                return self.response_mocker.create_chat_completion_response().content
        
        mock_client.chat.completions.create.side_effect = error_side_effect
    
    def _should_trigger_error(self, trigger: str, args: tuple, kwargs: dict) -> bool:
        """判断是否应该触发错误"""
        if trigger == 'invalid_api_key':
            # 检查API密钥
            return kwargs.get('api_key', '').startswith('invalid')
        elif trigger == 'rate_limit':
            # 模拟速率限制（简单实现）
            return random.random() < 0.1  # 10%概率触发
        elif trigger == 'large_request':
            # 检查请求大小
            messages = kwargs.get('messages', [])
            total_length = sum(len(str(msg)) for msg in messages)
            return total_length > 10000
        else:
            return False


@contextmanager
def mock_harborai_client(**config):
    """HarborAI客户端Mock上下文管理器
    
    功能：提供临时的HarborAI客户端Mock
    参数：
        **config: Mock配置
    返回：Mock客户端
    边界条件：确保Mock在退出时清理
    假设：Mock配置有效
    不确定点：某些配置可能影响全局状态
    验证方法：with mock_harborai_client() as client: ...
    """
    mocker = HarborAIMocker()
    mock_client = mocker.create_mock_client(**config)
    
    try:
        yield mock_client
    finally:
        # 清理Mock状态
        if hasattr(mock_client, 'reset_mock'):
            mock_client.reset_mock()


@asynccontextmanager
async def async_mock_harborai_client(**config):
    """异步HarborAI客户端Mock上下文管理器
    
    功能：提供临时的异步HarborAI客户端Mock
    参数：
        **config: Mock配置
    返回：异步Mock客户端
    边界条件：确保异步Mock在退出时清理
    假设：异步Mock配置有效
    不确定点：某些异步操作可能需要特殊处理
    验证方法：async with async_mock_harborai_client() as client: ...
    """
    mocker = HarborAIMocker()
    mock_client = mocker.create_mock_client(**config)
    
    # 将同步方法转换为异步
    original_create = mock_client.chat.completions.create
    
    async def async_create(*args, **kwargs):
        # 模拟异步延迟
        await asyncio.sleep(config.get('async_delay', 0.1))
        return original_create(*args, **kwargs)
    
    mock_client.chat.completions.create = async_create
    
    try:
        yield mock_client
    finally:
        # 清理异步Mock状态
        if hasattr(mock_client, 'reset_mock'):
            mock_client.reset_mock()


def create_mock_response_factory(response_type: str = 'chat_completion'):
    """创建Mock响应工厂函数
    
    功能：创建特定类型的响应工厂
    参数：
        response_type: 响应类型
    返回：响应工厂函数
    边界条件：处理无效的响应类型
    假设：响应类型参数有效
    不确定点：某些响应类型可能需要特殊处理
    验证方法：factory = create_mock_response_factory('chat_completion')
    """
    mocker = APIResponseMocker()
    
    def factory(**kwargs):
        if response_type == 'chat_completion':
            return mocker.create_chat_completion_response(**kwargs)
        elif response_type == 'embedding':
            return mocker.create_embedding_response(**kwargs)
        elif response_type == 'error':
            return mocker.create_error_response(**kwargs)
        else:
            raise ValueError(f"不支持的响应类型：{response_type}")
    
    return factory


def patch_harborai_method(method_path: str, mock_response: Any = None):
    """装饰器：临时替换HarborAI方法
    
    功能：使用装饰器临时Mock HarborAI方法
    参数：
        method_path: 方法路径（如'harborai.Client.chat.completions.create'）
        mock_response: Mock响应
    返回：装饰器函数
    边界条件：确保方法路径有效
    假设：方法路径存在
    不确定点：某些方法可能有复杂的调用链
    验证方法：@patch_harborai_method('harborai.Client.chat.completions.create')
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with patch(method_path) as mock_method:
                if mock_response is not None:
                    mock_method.return_value = mock_response
                else:
                    # 使用默认Mock响应
                    mocker = APIResponseMocker()
                    default_response = mocker.create_chat_completion_response()
                    mock_method.return_value = default_response.content
                
                return func(*args, **kwargs)
        return wrapper
    return decorator


class MockMetricsCollector:
    """Mock指标收集器
    
    功能：收集Mock调用的指标数据
    假设：指标数据格式标准
    不确定点：某些指标可能需要特殊计算
    验证方法：pytest tests/test_mock_helpers.py::TestMockMetricsCollector
    """
    
    def __init__(self):
        """初始化指标收集器"""
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_response_time': 0.0,
            'response_times': [],
            'error_types': {},
            'call_history': []
        }
    
    def record_call(
        self, 
        method: str, 
        success: bool, 
        response_time: float,
        error_type: str = None
    ):
        """记录调用指标
        
        功能：记录单次调用的指标数据
        参数：
            method: 调用方法
            success: 是否成功
            response_time: 响应时间
            error_type: 错误类型（如果失败）
        """
        self.metrics['total_calls'] += 1
        self.metrics['response_times'].append(response_time)
        
        if success:
            self.metrics['successful_calls'] += 1
        else:
            self.metrics['failed_calls'] += 1
            if error_type:
                self.metrics['error_types'][error_type] = \
                    self.metrics['error_types'].get(error_type, 0) + 1
        
        # 更新平均响应时间
        self.metrics['average_response_time'] = \
            sum(self.metrics['response_times']) / len(self.metrics['response_times'])
        
        # 记录调用历史
        self.metrics['call_history'].append({
            'method': method,
            'success': success,
            'response_time': response_time,
            'error_type': error_type,
            'timestamp': time.time()
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标摘要"""
        metrics = self.metrics.copy()
        
        # 计算成功率
        if metrics['total_calls'] > 0:
            metrics['success_rate'] = metrics['successful_calls'] / metrics['total_calls']
        else:
            metrics['success_rate'] = 0.0
        
        # 计算响应时间统计
        if metrics['response_times']:
            metrics['min_response_time'] = min(metrics['response_times'])
            metrics['max_response_time'] = max(metrics['response_times'])
            
            # 计算百分位数
            sorted_times = sorted(metrics['response_times'])
            n = len(sorted_times)
            metrics['p50_response_time'] = sorted_times[int(n * 0.5)]
            metrics['p95_response_time'] = sorted_times[int(n * 0.95)]
            metrics['p99_response_time'] = sorted_times[int(n * 0.99)]
        
        return metrics
    
    def reset_metrics(self):
        """重置指标"""
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_response_time': 0.0,
            'response_times': [],
            'error_types': {},
            'call_history': []
        }


# 全局Mock指标收集器实例
mock_metrics_collector = MockMetricsCollector()