# -*- coding: utf-8 -*-
"""
数据夹具模块
提供测试数据相关的夹具
"""

import pytest
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from datetime import datetime, timedelta
import random
import string
from faker import Faker
from harborai.core.base_plugin import ChatMessage, ChatCompletion, ChatCompletionChunk
from harborai.utils.exceptions import HarborAIError, APIError, RateLimitError

# 初始化Faker
fake = Faker(['zh_CN', 'en_US'])


@pytest.fixture(scope='session')
def test_messages() -> List[Dict[str, str]]:
    """基础测试消息数据"""
    return [
        {
            'role': 'system',
            'content': '你是一个有用的AI助手，请用中文回答问题。'
        },
        {
            'role': 'user',
            'content': '请介绍一下人工智能的发展历史。'
        },
        {
            'role': 'assistant',
            'content': '人工智能的发展可以追溯到20世纪50年代...'
        },
        {
            'role': 'user',
            'content': '能详细说说深度学习吗？'
        }
    ]


@pytest.fixture(scope='session')
def reasoning_test_messages() -> List[Dict[str, str]]:
    """推理模型测试消息"""
    return [
        {
            'role': 'system',
            'content': '你是一个逻辑推理专家，请详细展示你的推理过程。'
        },
        {
            'role': 'user',
            'content': '如果所有的猫都是动物，而所有的动物都需要食物，那么所有的猫都需要食物吗？请详细推理。'
        }
    ]


@pytest.fixture
def complex_reasoning_messages():
    """复杂推理问题测试消息夹具"""
    return [
        {
            "role": "system",
            "content": "你是一个逻辑推理专家，需要通过严密的逻辑分析来解决复杂问题。"
        },
        {
            "role": "user",
            "content": "有三个盒子，每个盒子里都有两个球。第一个盒子里有两个白球，第二个盒子里有两个黑球，第三个盒子里有一个白球和一个黑球。现在随机选择一个盒子，从中随机取出一个球，发现是白球。请问这个白球来自第三个盒子的概率是多少？请详细展示推理过程。"
        }
    ]


@pytest.fixture
def reasoning_model_test_cases():
    """推理模型测试用例夹具"""
    return [
        {
            "name": "simple_math_reasoning",
            "messages": [
                {"role": "user", "content": "如果一个数加上它的一半等于15，这个数是多少？请展示推理过程。"}
            ],
            "expected_reasoning_keywords": ["设", "方程", "解", "验证"],
            "expected_answer_keywords": ["10"]
        },
        {
            "name": "logical_reasoning",
            "messages": [
                {"role": "user", "content": "所有的猫都是动物，所有的动物都需要食物，因此所有的猫都需要食物。这个推理是否正确？"}
            ],
            "expected_reasoning_keywords": ["三段论", "大前提", "小前提", "结论"],
            "expected_answer_keywords": ["正确", "有效"]
        },
        {
            "name": "probability_reasoning",
            "messages": [
                {"role": "user", "content": "抛掷一枚公平硬币3次，至少出现一次正面的概率是多少？"}
            ],
            "expected_reasoning_keywords": ["互补事件", "独立", "概率"],
            "expected_answer_keywords": ["7/8", "0.875"]
        }
    ]


@pytest.fixture(scope='session')
def structured_output_messages() -> List[Dict[str, Any]]:
    """结构化输出测试消息"""
    return [
        {
            'role': 'system',
            'content': '请按照指定的JSON格式返回结果。'
        },
        {
            'role': 'user',
            'content': '分析以下文本的情感："今天天气真好，我很开心！"',
            'response_format': {
                'type': 'json_object',
                'schema': {
                    'type': 'object',
                    'properties': {
                        'sentiment': {'type': 'string', 'enum': ['positive', 'negative', 'neutral']},
                        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                        'keywords': {'type': 'array', 'items': {'type': 'string'}}
                    },
                    'required': ['sentiment', 'confidence']
                }
            }
        }
    ]





@pytest.fixture(scope='session')
def performance_test_messages() -> List[Dict[str, str]]:
    """性能测试消息（不同长度）"""
    return [
        {
            'role': 'user',
            'content': '你好'  # 短消息
        },
        {
            'role': 'user',
            'content': '请详细介绍一下机器学习的基本概念，包括监督学习、无监督学习和强化学习的区别。'  # 中等长度
        },
        {
            'role': 'user',
            'content': '请写一篇关于人工智能在医疗领域应用的详细报告，包括现状分析、技术挑战、应用案例、未来发展趋势等方面，要求内容详实、逻辑清晰、结构完整。' * 5  # 长消息
        }
    ]


@pytest.fixture(scope='session')
def error_test_cases() -> List[Dict[str, Any]]:
    """错误测试用例"""
    return [
        {
            'name': 'empty_messages',
            'messages': [],
            'expected_error': 'ValidationError'
        },
        {
            'name': 'invalid_role',
            'messages': [{'role': 'invalid', 'content': 'test'}],
            'expected_error': 'ValidationError'
        },
        {
            'name': 'missing_content',
            'messages': [{'role': 'user'}],
            'expected_error': 'ValidationError'
        },
        {
            'name': 'invalid_model',
            'messages': [{'role': 'user', 'content': 'test'}],
            'model': 'non-existent-model',
            'expected_error': 'ModelNotFoundError'
        }
    ]


@pytest.fixture(scope='session')
def mock_responses() -> Dict[str, Any]:
    """Mock响应数据"""
    return {
        'chat_completion': {
            'id': 'chatcmpl-test123',
            'object': 'chat.completion',
            'created': 1234567890,
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
        'reasoning_completion': {
            'id': 'chatcmpl-reasoning123',
            'object': 'chat.completion',
            'created': 1234567890,
            'model': 'deepseek-reasoner',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': '让我思考一下这个问题。经过仔细分析和推理，我认为答案是正确的。通过逻辑推导，可以得出结论。',
                    'reasoning_content': '首先，我需要分析问题的核心要素。然后考虑各种可能性和逻辑关系。接下来进行推理过程：1）建立前提条件，2）应用逻辑规则，3）得出中间结论，4）验证结果的合理性。最终通过严密的推理得出结论。'
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 30,
                'completion_tokens': 20,
                'total_tokens': 50,
                'reasoning_tokens': 15
            }
        },
        'structured_completion': {
            'id': 'chatcmpl-structured123',
            'object': 'chat.completion',
            'created': 1234567890,
            'model': 'test-model',
            'choices': [{
                'index': 0,
                'message': {
                    'role': 'assistant',
                    'content': json.dumps({
                        'sentiment': 'positive',
                        'confidence': 0.95,
                        'keywords': ['天气', '开心', '好']
                    }, ensure_ascii=False)
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 25,
                'completion_tokens': 15,
                'total_tokens': 40
            }
        },
        'stream_chunks': [
            {
                'id': 'chatcmpl-stream1',
                'object': 'chat.completion.chunk',
                'created': 1234567890,
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
                'created': 1234567891,
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
                'created': 1234567892,
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
                'created': 1234567893,
                'model': 'test-model',
                'choices': [{
                    'index': 0,
                    'delta': {},
                    'finish_reason': 'stop'
                }]
            }
        ]
    }


@pytest.fixture(scope='session')
def json_schemas() -> Dict[str, Dict[str, Any]]:
    """JSON Schema定义"""
    return {
        'chat_completion_schema': {
            'type': 'object',
            'properties': {
                'id': {'type': 'string'},
                'object': {'type': 'string', 'enum': ['chat.completion']},
                'created': {'type': 'integer'},
                'model': {'type': 'string'},
                'choices': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'index': {'type': 'integer'},
                            'message': {
                                'type': 'object',
                                'properties': {
                                    'role': {'type': 'string'},
                                    'content': {'type': 'string'}
                                },
                                'required': ['role', 'content']
                            },
                            'finish_reason': {'type': 'string'}
                        },
                        'required': ['index', 'message', 'finish_reason']
                    }
                },
                'usage': {
                    'type': 'object',
                    'properties': {
                        'prompt_tokens': {'type': 'integer'},
                        'completion_tokens': {'type': 'integer'},
                        'total_tokens': {'type': 'integer'}
                    },
                    'required': ['prompt_tokens', 'completion_tokens', 'total_tokens']
                }
            },
            'required': ['id', 'object', 'created', 'model', 'choices', 'usage']
        },
        'reasoning_completion_schema': {
            'type': 'object',
            'properties': {
                'id': {'type': 'string'},
                'object': {'type': 'string', 'enum': ['chat.completion']},
                'created': {'type': 'integer'},
                'model': {'type': 'string'},
                'choices': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'index': {'type': 'integer'},
                            'message': {
                                'type': 'object',
                                'properties': {
                                    'role': {'type': 'string'},
                                    'content': {'type': 'string'},
                                    'reasoning_content': {'type': 'string'}
                                },
                                'required': ['role', 'content']
                            },
                            'finish_reason': {'type': 'string'}
                        },
                        'required': ['index', 'message', 'finish_reason']
                    }
                },
                'usage': {
                    'type': 'object',
                    'properties': {
                        'prompt_tokens': {'type': 'integer'},
                        'completion_tokens': {'type': 'integer'},
                        'total_tokens': {'type': 'integer'},
                        'reasoning_tokens': {'type': 'integer'}
                    },
                    'required': ['prompt_tokens', 'completion_tokens', 'total_tokens']
                }
            },
            'required': ['id', 'object', 'created', 'model', 'choices', 'usage']
        }
    }


@pytest.fixture(scope='function')
def random_messages(request) -> List[Dict[str, str]]:
    """随机生成的测试消息"""
    count = getattr(request, 'param', 3)
    messages = []
    
    # 添加系统消息
    messages.append({
        'role': 'system',
        'content': fake.sentence()
    })
    
    # 添加用户和助手消息
    for i in range(count):
        messages.append({
            'role': 'user',
            'content': fake.text(max_nb_chars=200)
        })
        if i < count - 1:  # 最后一条不添加助手回复
            messages.append({
                'role': 'assistant',
                'content': fake.text(max_nb_chars=300)
            })
    
    return messages


@pytest.fixture(scope='function')
def large_message_dataset() -> List[Dict[str, str]]:
    """大型消息数据集（用于压力测试）"""
    messages = []
    
    for i in range(100):
        messages.append({
            'role': 'user',
            'content': f"测试消息 {i}: {fake.text(max_nb_chars=500)}"
        })
    
    return messages


@pytest.fixture(scope='session')
def model_configs() -> Dict[str, Dict[str, Any]]:
    """模型配置数据"""
    return {
        'deepseek': {
            'model': 'deepseek-chat',
            'max_tokens': 4096,
            'temperature': 0.7,
            'top_p': 0.9,
            'frequency_penalty': 0.0,
            'presence_penalty': 0.0
        },
        'deepseek_reasoner': {
            'model': 'deepseek-reasoner',
            'max_tokens': 8192,
            'temperature': 0.3,
            'reasoning_effort': 'medium'
        },
        'ernie': {
            'model': 'ernie-bot-turbo',
            'max_tokens': 2048,
            'temperature': 0.8,
            'top_p': 0.95
        },
        'doubao': {
            'model': 'doubao-pro-32k',
            'max_tokens': 32768,
            'temperature': 0.6,
            'top_p': 0.85
        }
    }


@pytest.fixture(scope='session')
def performance_baselines() -> Dict[str, Dict[str, float]]:
    """性能基线数据"""
    return {
        'response_time': {
            'p50': 1.0,
            'p95': 2.0,
            'p99': 3.0,
            'max': 5.0
        },
        'throughput': {
            'min': 10.0,
            'target': 50.0,
            'max': 100.0
        },
        'memory_usage': {
            'baseline': 50 * 1024 * 1024,  # 50MB
            'warning': 100 * 1024 * 1024,  # 100MB
            'critical': 200 * 1024 * 1024  # 200MB
        },
        'cpu_usage': {
            'baseline': 20.0,
            'warning': 50.0,
            'critical': 80.0
        }
    }


@pytest.fixture(scope='function')
def test_data_generator():
    """测试数据生成器"""
    class TestDataGenerator:
        @staticmethod
        def generate_message(role: str = 'user', length: str = 'medium') -> Dict[str, str]:
            """生成单条消息"""
            length_map = {
                'short': 50,
                'medium': 200,
                'long': 1000
            }
            max_chars = length_map.get(length, 200)
            
            return {
                'role': role,
                'content': fake.text(max_nb_chars=max_chars)
            }
        
        @staticmethod
        def generate_conversation(turns: int = 3) -> List[Dict[str, str]]:
            """生成对话"""
            messages = [{
                'role': 'system',
                'content': '你是一个有用的AI助手。'
            }]
            
            for i in range(turns):
                messages.append({
                    'role': 'user',
                    'content': fake.question()
                })
                messages.append({
                    'role': 'assistant',
                    'content': fake.text(max_nb_chars=300)
                })
            
            return messages
        
        @staticmethod
        def generate_api_key() -> str:
            """生成测试API密钥"""
            return f"sk-test-{''.join(random.choices(string.ascii_letters + string.digits, k=32))}"
        
        @staticmethod
        def generate_request_id() -> str:
            """生成请求ID"""
            return f"req-{''.join(random.choices(string.ascii_letters + string.digits, k=16))}"
    
    return TestDataGenerator()


@pytest.fixture(scope='session')
def cost_tracking_data() -> Dict[str, Any]:
    """成本追踪测试数据"""
    return {
        'pricing': {
            'deepseek-chat': {
                'input_price_per_1k': 0.0014,
                'output_price_per_1k': 0.0028
            },
            'deepseek-reasoner': {
                'input_price_per_1k': 0.0055,
                'output_price_per_1k': 0.022,
                'reasoning_price_per_1k': 0.055
            },
            'ernie-bot-turbo': {
                'input_price_per_1k': 0.008,
                'output_price_per_1k': 0.012
            }
        },
        'usage_samples': [
            {
                'model': 'deepseek-chat',
                'prompt_tokens': 100,
                'completion_tokens': 50,
                'total_tokens': 150
            },
            {
                'model': 'deepseek-reasoner',
                'prompt_tokens': 200,
                'completion_tokens': 100,
                'total_tokens': 300,
                'reasoning_tokens': 80
            }
        ]
    }


@pytest.fixture(scope='session')
def json_schema_simple() -> Dict[str, Any]:
    """简单的JSON Schema用于结构化输出测试"""
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "姓名"
            },
            "age": {
                "type": "integer",
                "description": "年龄",
                "minimum": 0,
                "maximum": 150
            }
        },
        "required": ["name", "age"],
        "additionalProperties": False
    }


@pytest.fixture(scope='function')
def temp_test_data(tmp_path) -> Path:
    """临时测试数据目录"""
    # 创建临时测试数据文件
    test_data = {
        'messages': [
            {'role': 'user', 'content': '测试消息'}
        ],
        'config': {
            'model': 'test-model',
            'temperature': 0.7
        }
    }
    
    data_file = tmp_path / 'test_data.json'
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    return tmp_path