# -*- coding: utf-8 -*-
"""
Mock夹具模块
提供各种Mock对象和响应的测试夹具
"""

import pytest
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import time
import random

# 条件导入可选依赖
try:
    import responses
except ImportError:
    responses = None

try:
    import aioresponses
except ImportError:
    aioresponses = None

# 条件导入 harborai 模块
try:
    from harborai.core.base_plugin import ChatCompletion, ChatCompletionChunk
except ImportError:
    # 如果模块不存在，创建简单的替代类
    class ChatCompletion:
        pass
    class ChatCompletionChunk:
        pass

try:
    from harborai.utils.exceptions import HarborAIError, APIError
except ImportError:
    # 如果异常类不存在，创建简单的替代类
    class HarborAIError(Exception):
        pass
    class APIError(Exception):
        pass


@pytest.fixture(scope='function')
def mock_http_responses():
    """HTTP响应Mock夹具"""
    if responses is None:
        pytest.skip("responses 库未安装")
    
    with responses.RequestsMock() as rsps:
        # 基础聊天完成响应
        rsps.add(
            responses.POST,
            'https://api.test.harborai.com/v1/chat/completions',
            json={
                'id': 'chatcmpl-test123',
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': '这是一个测试响应。'
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': 20,
                    'completion_tokens': 10,
                    'total_tokens': 30
                }
            },
            status=200
        )
        
        # 推理模型响应
        rsps.add(
            responses.POST,
            'https://api.deepseek.com/v1/chat/completions',
            json={
                'id': 'chatcmpl-reasoning123',
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': 'deepseek-reasoner',
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': '基于推理分析，答案是...',
                        'reasoning_content': '首先分析问题的核心要素...然后考虑各种可能性...最终得出结论...'
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': 30,
                    'completion_tokens': 25,
                    'total_tokens': 55,
                    'reasoning_tokens': 20
                }
            },
            status=200
        )
        
        # 错误响应
        rsps.add(
            responses.POST,
            'https://api.test.harborai.com/v1/chat/completions',
            json={
                'error': {
                    'message': 'Invalid API key',
                    'type': 'invalid_request_error',
                    'code': 'invalid_api_key'
                }
            },
            status=401
        )
        
        yield rsps


@pytest.fixture(scope='function')
def mock_async_http_responses():
    """异步HTTP响应Mock夹具"""
    if aioresponses is None:
        pytest.skip("aioresponses 库未安装")
    
    with aioresponses.aioresponses() as m:
        # 基础异步响应
        m.post(
            'https://api.test.harborai.com/v1/chat/completions',
            payload={
                'id': 'chatcmpl-async123',
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': '这是一个异步测试响应。'
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': 25,
                    'completion_tokens': 12,
                    'total_tokens': 37
                }
            },
            status=200
        )
        
        # 推理模型异步响应
        m.post(
            'https://api.deepseek.com/v1/chat/completions',
            payload={
                'id': 'chatcmpl-async-reasoning123',
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': 'deepseek-reasoner',
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': '异步推理结果...',
                        'reasoning_content': '异步推理过程的详细分析...'
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': 35,
                    'completion_tokens': 28,
                    'total_tokens': 63,
                    'reasoning_tokens': 22
                }
            },
            status=200
        )
        
        yield m


@pytest.fixture(scope='function')
def mock_stream_response():
    """流式响应Mock夹具"""
    def create_stream_chunks():
        chunks = [
            {
                'id': 'chatcmpl-stream1',
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'delta': {'role': 'assistant', 'content': '你好'},
                    'finish_reason': None
                }]
            },
            {
                'id': 'chatcmpl-stream2',
                'object': 'chat.completion.chunk',
                'created': int(time.time()) + 1,
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'delta': {'content': '，我是'},
                    'finish_reason': None
                }]
            },
            {
                'id': 'chatcmpl-stream3',
                'object': 'chat.completion.chunk',
                'created': int(time.time()) + 2,
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'delta': {'content': 'AI助手。'},
                    'finish_reason': None
                }]
            },
            {
                'id': 'chatcmpl-stream4',
                'object': 'chat.completion.chunk',
                'created': int(time.time()) + 3,
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop'
                }]
            }
        ]
        
        for chunk in chunks:
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    
    return create_stream_chunks


@pytest.fixture(scope='function')
def mock_reasoning_stream_response():
    """推理模型流式响应Mock夹具"""
    def create_reasoning_stream_chunks():
        chunks = [
            {
                'id': 'chatcmpl-reasoning-stream1',
                'object': 'chat.completion.chunk',
                'created': int(time.time()),
                'model': 'deepseek-reasoner',
                'choices': [{
                    'index': 0,
                    'delta': {
                        'role': 'assistant',
                        'content': '',
                        'reasoning_content': '开始分析问题...'
                    },
                    'finish_reason': None
                }]
            },
            {
                'id': 'chatcmpl-reasoning-stream2',
                'object': 'chat.completion.chunk',
                'created': int(time.time()) + 1,
                'model': 'deepseek-reasoner',
                'choices': [{
                    'index': 0,
                    'delta': {
                        'reasoning_content': '考虑各种可能性...'
                    },
                    'finish_reason': None
                }]
            },
            {
                'id': 'chatcmpl-reasoning-stream3',
                'object': 'chat.completion.chunk',
                'created': int(time.time()) + 2,
                'model': 'deepseek-reasoner',
                'choices': [{
                    'index': 0,
                    'delta': {
                        'content': '基于分析，'
                    },
                    'finish_reason': None
                }]
            },
            {
                'id': 'chatcmpl-reasoning-stream4',
                'object': 'chat.completion.chunk',
                'created': int(time.time()) + 3,
                'model': 'deepseek-reasoner',
                'choices': [{
                    'index': 0,
                    'delta': {
                        'content': '答案是...'
                    },
                    'finish_reason': None
                }]
            },
            {
                'id': 'chatcmpl-reasoning-stream5',
                'object': 'chat.completion.chunk',
                'created': int(time.time()) + 4,
                'model': 'deepseek-reasoner',
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop'
                }]
            }
        ]
        
        for chunk in chunks:
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    
    return create_reasoning_stream_chunks


@pytest.fixture(scope='function')
def mock_error_responses():
    """错误响应Mock夹具"""
    error_responses = {
        'invalid_api_key': {
            'error': {
                'message': 'Invalid API key provided',
                'type': 'invalid_request_error',
                'code': 'invalid_api_key'
            }
        },
        'rate_limit': {
            'error': {
                'message': 'Rate limit exceeded',
                'type': 'rate_limit_error',
                'code': 'rate_limit_exceeded'
            }
        },
        'model_not_found': {
            'error': {
                'message': 'Model not found',
                'type': 'invalid_request_error',
                'code': 'model_not_found'
            }
        },
        'server_error': {
            'error': {
                'message': 'Internal server error',
                'type': 'server_error',
                'code': 'internal_error'
            }
        },
        'timeout': {
            'error': {
                'message': 'Request timeout',
                'type': 'timeout_error',
                'code': 'request_timeout'
            }
        }
    }
    
    return error_responses


@pytest.fixture
def mock_reasoning_response():
    """模拟推理模型响应夹具"""
    return {
        "id": "chatcmpl-reasoning-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "deepseek-r1",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "经过深入思考，我认为这个问题的答案是...",
                "reasoning_content": "让我仔细分析这个问题：\n1. 首先考虑问题的背景\n2. 然后分析可能的解决方案\n3. 最后得出结论"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 100,
            "total_tokens": 150,
            "reasoning_tokens": 75
        }
    }


@pytest.fixture
def mock_reasoning_stream_chunks():
    """模拟推理模型流式响应块夹具"""
    chunks = [
        # 推理阶段开始
        {
            "id": "chatcmpl-reasoning-stream-123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "deepseek-r1",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "reasoning_content": "让我思考一下这个问题..."
                },
                "finish_reason": None
            }]
        },
        # 推理过程中
        {
            "id": "chatcmpl-reasoning-stream-123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "deepseek-r1",
            "choices": [{
                "index": 0,
                "delta": {
                    "reasoning_content": "首先分析问题的核心要素..."
                },
                "finish_reason": None
            }]
        },
        # 推理结束，开始回答
        {
            "id": "chatcmpl-reasoning-stream-123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "deepseek-r1",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "基于我的分析，"
                },
                "finish_reason": None
            }]
        },
        # 继续回答
        {
            "id": "chatcmpl-reasoning-stream-123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "deepseek-r1",
            "choices": [{
                "index": 0,
                "delta": {
                    "content": "答案是..."
                },
                "finish_reason": None
            }]
        },
        # 结束
        {
            "id": "chatcmpl-reasoning-stream-123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "deepseek-r1",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
    ]
    return chunks


@pytest.fixture(scope='function')
def mock_performance_data():
    """性能数据Mock夹具"""
    def generate_performance_metrics():
        return {
            'response_time': random.uniform(0.5, 3.0),
            'memory_usage': random.randint(50, 200) * 1024 * 1024,  # 50-200MB
            'cpu_usage': random.uniform(10.0, 80.0),
            'throughput': random.uniform(20.0, 100.0),
            'concurrent_requests': random.randint(1, 150),
            'error_rate': random.uniform(0.0, 0.05),
            'timestamp': datetime.now().isoformat()
        }
    
    return generate_performance_metrics


@pytest.fixture(scope='function')
def mock_database():
    """数据库Mock夹具"""
    class MockDatabase:
        def __init__(self):
            self.data = {}
            self.call_logs = []
        
        def insert(self, table: str, data: Dict[str, Any]) -> str:
            if table not in self.data:
                self.data[table] = []
            
            record_id = f"{table}_{len(self.data[table]) + 1}"
            record = {'id': record_id, **data}
            self.data[table].append(record)
            
            self.call_logs.append({
                'operation': 'insert',
                'table': table,
                'data': data,
                'timestamp': datetime.now()
            })
            
            return record_id
        
        def select(self, table: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
            if table not in self.data:
                return []
            
            records = self.data[table]
            
            if filters:
                filtered_records = []
                for record in records:
                    match = True
                    for key, value in filters.items():
                        if record.get(key) != value:
                            match = False
                            break
                    if match:
                        filtered_records.append(record)
                records = filtered_records
            
            self.call_logs.append({
                'operation': 'select',
                'table': table,
                'filters': filters,
                'result_count': len(records),
                'timestamp': datetime.now()
            })
            
            return records
        
        def update(self, table: str, record_id: str, data: Dict[str, Any]) -> bool:
            if table not in self.data:
                return False
            
            for record in self.data[table]:
                if record['id'] == record_id:
                    record.update(data)
                    self.call_logs.append({
                        'operation': 'update',
                        'table': table,
                        'record_id': record_id,
                        'data': data,
                        'timestamp': datetime.now()
                    })
                    return True
            
            return False
        
        def delete(self, table: str, record_id: str) -> bool:
            if table not in self.data:
                return False
            
            for i, record in enumerate(self.data[table]):
                if record['id'] == record_id:
                    del self.data[table][i]
                    self.call_logs.append({
                        'operation': 'delete',
                        'table': table,
                        'record_id': record_id,
                        'timestamp': datetime.now()
                    })
                    return True
            
            return False
        
        def get_call_logs(self) -> List[Dict[str, Any]]:
            return self.call_logs.copy()
        
        def clear(self):
            self.data.clear()
            self.call_logs.clear()
    
    return MockDatabase()


@pytest.fixture(scope='function')
def mock_cache():
    """缓存Mock夹具"""
    class MockCache:
        def __init__(self):
            self.data = {}
            self.access_logs = []
        
        def get(self, key: str) -> Optional[Any]:
            value = self.data.get(key)
            self.access_logs.append({
                'operation': 'get',
                'key': key,
                'hit': value is not None,
                'timestamp': datetime.now()
            })
            return value
        
        def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
            self.data[key] = {
                'value': value,
                'ttl': ttl,
                'created_at': datetime.now()
            }
            self.access_logs.append({
                'operation': 'set',
                'key': key,
                'ttl': ttl,
                'timestamp': datetime.now()
            })
        
        def delete(self, key: str) -> bool:
            existed = key in self.data
            if existed:
                del self.data[key]
            
            self.access_logs.append({
                'operation': 'delete',
                'key': key,
                'existed': existed,
                'timestamp': datetime.now()
            })
            
            return existed
        
        def clear(self) -> None:
            self.data.clear()
            self.access_logs.append({
                'operation': 'clear',
                'timestamp': datetime.now()
            })
        
        def get_stats(self) -> Dict[str, Any]:
            total_operations = len(self.access_logs)
            get_operations = [log for log in self.access_logs if log['operation'] == 'get']
            hits = [log for log in get_operations if log['hit']]
            
            return {
                'total_operations': total_operations,
                'get_operations': len(get_operations),
                'cache_hits': len(hits),
                'cache_misses': len(get_operations) - len(hits),
                'hit_rate': len(hits) / len(get_operations) if get_operations else 0.0
            }
    
    return MockCache()


@pytest.fixture(scope='function')
def mock_logger():
    """日志Mock夹具"""
    class MockLogger:
        def __init__(self):
            self.logs = []
        
        def debug(self, message: str, **kwargs):
            self._log('DEBUG', message, kwargs)
        
        def info(self, message: str, **kwargs):
            self._log('INFO', message, kwargs)
        
        def warning(self, message: str, **kwargs):
            self._log('WARNING', message, kwargs)
        
        def error(self, message: str, **kwargs):
            self._log('ERROR', message, kwargs)
        
        def critical(self, message: str, **kwargs):
            self._log('CRITICAL', message, kwargs)
        
        def _log(self, level: str, message: str, kwargs: Dict[str, Any]):
            self.logs.append({
                'level': level,
                'message': message,
                'kwargs': kwargs,
                'timestamp': datetime.now()
            })
        
        def get_logs(self, level: Optional[str] = None) -> List[Dict[str, Any]]:
            if level:
                return [log for log in self.logs if log['level'] == level]
            return self.logs.copy()
        
        def clear(self):
            self.logs.clear()
    
    return MockLogger()


@pytest.fixture(scope='function')
def mock_metrics_collector():
    """指标收集器Mock夹具"""
    class MockMetricsCollector:
        def __init__(self):
            self.metrics = {}
            self.counters = {}
            self.histograms = {}
        
        def increment_counter(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
            key = f"{name}:{tags or {}}"
            self.counters[key] = self.counters.get(key, 0) + value
        
        def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
            key = f"{name}:{tags or {}}"
            if key not in self.histograms:
                self.histograms[key] = []
            self.histograms[key].append(value)
        
        def set_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
            key = f"{name}:{tags or {}}"
            self.metrics[key] = value
        
        def get_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
            key = f"{name}:{tags or {}}"
            return self.counters.get(key, 0.0)
        
        def get_histogram_stats(self, name: str, tags: Optional[Dict[str, str]] = None) -> Dict[str, float]:
            key = f"{name}:{tags or {}}"
            values = self.histograms.get(key, [])
            
            if not values:
                return {}
            
            sorted_values = sorted(values)
            count = len(values)
            
            return {
                'count': count,
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / count,
                'p50': sorted_values[int(count * 0.5)],
                'p95': sorted_values[int(count * 0.95)],
                'p99': sorted_values[int(count * 0.99)]
            }
        
        def get_gauge(self, name: str, tags: Optional[Dict[str, str]] = None) -> Optional[float]:
            key = f"{name}:{tags or {}}"
            return self.metrics.get(key)
        
        def clear(self):
            self.metrics.clear()
            self.counters.clear()
            self.histograms.clear()
    
    return MockMetricsCollector()


@pytest.fixture(scope='function')
def mock_time():
    """时间Mock夹具"""
    class MockTime:
        def __init__(self):
            self.current_time = time.time()
        
        def time(self) -> float:
            return self.current_time
        
        def sleep(self, seconds: float) -> None:
            self.current_time += seconds
        
        def advance(self, seconds: float) -> None:
            self.current_time += seconds
        
        def set_time(self, timestamp: float) -> None:
            self.current_time = timestamp
    
    mock_time_obj = MockTime()
    
    with patch('time.time', side_effect=mock_time_obj.time), \
         patch('time.sleep', side_effect=mock_time_obj.sleep):
        yield mock_time_obj