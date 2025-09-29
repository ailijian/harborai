# HarborAI 测试项目技术设计方案

## 1. 项目概述

### 1.1 设计目标

本文档为 HarborAI 项目设计一个完整的测试项目，基于 `HarborAI功能与性能测试清单.md` 中定义的测试需求，构建一个全面、可靠、高效的测试体系。

### 1.2 核心要求

* **全面覆盖**：覆盖功能测试（A-Q模块）和性能测试（O模块）的所有测试项

* **多厂商支持**：支持 DeepSeek、ERNIE、Doubao 等多厂商 API 测试

* **推理模型支持**：专门测试推理模型（deepseek-reasoner、ernie-x1-turbo-32k、doubao-seed-1-6-250615）的特殊功能

* **环境适配**：完全适配 Windows 11 + PowerShell 环境

* **框架统一**：使用 pytest 作为主要测试框架

* **并发支持**：支持同步、异步、并发测试场景

* **性能基准**：包含完整的性能基准测试和监控

* **容器化**：支持 Docker 化的 PostgreSQL 测试环境

* **安全性**：包含测试数据脱敏和安全性验证

### 1.3 技术栈选择

* **测试框架**：pytest + pytest-asyncio + pytest-benchmark

* **性能测试**：locust + psutil + memory\_profiler

* **Mock 工具**：pytest-mock + responses + aioresponses

* **数据库**：PostgreSQL (Docker) + psycopg2

* **监控工具**：prometheus\_client + grafana

* **报告生成**：pytest-html + allure-pytest

* **CI/CD**：GitHub Actions + Docker

## 2. 测试项目整体架构设计

### 2.1 架构层次

```mermaid
graph TD
    A[测试执行层] --> B[测试框架层]
    B --> C[测试工具层]
    C --> D[基础设施层]
    
    subgraph "测试执行层"
        A1[功能测试套件]
        A2[性能测试套件]
        A3[集成测试套件]
        A4[端到端测试套件]
    end
    
    subgraph "测试框架层"
        B1[pytest 核心]
        B2[异步测试支持]
        B3[性能测试框架]
        B4[Mock 框架]
    end
    
    subgraph "测试工具层"
        C1[测试数据管理]
        C2[环境配置管理]
        C3[报告生成工具]
        C4[监控工具]
    end
    
    subgraph "基础设施层"
        D1[Docker 环境]
        D2[数据库服务]
        D3[CI/CD 流水线]
        D4[监控服务]
    end
```

### 2.2 目录结构设计

```
tests/
├── conftest.py                 # pytest 全局配置
├── pytest.ini                  # pytest 配置文件
├── requirements-test.txt        # 测试依赖
├── docker-compose.yml          # Docker 环境配置
├── .env.test                   # 测试环境变量
├── README.md                   # 测试项目说明
│
├── fixtures/                   # 测试夹具
│   ├── __init__.py
│   ├── client_fixtures.py      # 客户端夹具
│   ├── data_fixtures.py        # 测试数据夹具
│   ├── mock_fixtures.py        # Mock 夹具
│   └── performance_fixtures.py # 性能测试夹具
│
├── utils/                      # 测试工具
│   ├── __init__.py
│   ├── test_helpers.py         # 测试辅助函数
│   ├── mock_helpers.py         # Mock 辅助函数
│   ├── data_generators.py      # 测试数据生成器
│   ├── performance_utils.py    # 性能测试工具
│   ├── security_utils.py       # 安全测试工具
│   └── report_utils.py         # 报告生成工具
│
├── data/                       # 测试数据
│   ├── schemas/                # JSON Schema 文件
│   ├── mock_responses/         # Mock 响应数据
│   ├── test_cases/             # 测试用例数据
│   └── performance_baselines/  # 性能基准数据
│
├── functional/                 # 功能测试
│   ├── __init__.py
│   ├── test_a_api_compatibility.py      # 模块A: API兼容性
│   ├── test_b_sync_async_stream.py      # 模块B: 同步异步流式
│   ├── test_c_structured_output.py      # 模块C: 结构化输出
│   ├── test_d_reasoning_models.py       # 模块D: 推理模型
│   ├── test_e_plugin_system.py          # 模块E: 插件系统
│   ├── test_f_exception_retry.py        # 模块F: 异常重试
│   ├── test_g_fallback_strategy.py      # 模块G: 降级策略
│   ├── test_h_observability.py          # 模块H: 可观测性
│   ├── test_i_cost_tracking.py          # 模块I: 成本统计
│   ├── test_j_persistence.py            # 模块J: 持久化
│   ├── test_k_configuration.py          # 模块K: 配置管理
│   ├── test_l_cli.py                    # 模块L: CLI
│   ├── test_m_security.py               # 模块M: 安全合规
│   ├── test_n_standard_alignment.py     # 模块N: 标准格式对齐
│   ├── test_p_error_robustness.py       # 模块P: 错误健壮性
│   ├── test_native_vs_agently.py      # 原生Schema vs Agently对比测试
│   └── test_resource_monitoring.py    # 资源监控测试
│
├── performance/                # 性能测试
│   ├── __init__.py
│   ├── test_basic_performance.py      # 基础性能测试
│   ├── test_concurrent_performance.py # 并发性能测试
│   ├── test_stress_testing.py         # 压力测试
│   ├── test_resource_monitoring.py    # 资源监控测试
│   ├── test_streaming_performance.py  # 流式性能测试
│   ├── benchmarks/                    # 性能基准测试
│   │   ├── benchmark_wrapper_overhead.py
│   │   ├── benchmark_concurrent_load.py
│   │   ├── benchmark_memory_usage.py
│   │   └── benchmark_vendor_comparison.py
│   └── load_tests/                    # 负载测试
│       ├── locustfile.py
│       └── load_test_scenarios.py
│
├── integration/                # 集成测试
│   ├── __init__.py
│   ├── test_end_to_end.py             # 端到端测试
│   ├── test_multi_vendor.py           # 多厂商集成测试
│   ├── test_database_integration.py   # 数据库集成测试
│   └── test_docker_environment.py     # Docker环境测试
│
├── security/                   # 安全测试
│   ├── __init__.py
│   ├── test_data_sanitization.py      # 数据脱敏测试
│   ├── test_api_key_security.py       # API密钥安全测试
│   └── test_input_validation.py       # 输入验证测试
│
├── reports/                    # 测试报告
│   ├── html/                          # HTML报告
│   ├── allure/                        # Allure报告
│   ├── performance/                   # 性能报告
│   └── coverage/                      # 覆盖率报告
│
└── scripts/                    # 测试脚本
    ├── setup_test_env.ps1             # 测试环境设置脚本
    ├── run_all_tests.ps1              # 运行所有测试
    ├── run_performance_tests.ps1      # 运行性能测试
    ├── generate_reports.ps1           # 生成测试报告
    └── cleanup_test_env.ps1           # 清理测试环境
```

## 3. 功能测试框架设计

### 3.1 测试框架核心组件

#### 3.1.1 pytest 配置 (pytest.ini)

```ini
[tool:pytest]
minversion = 6.0
addopts = 
    -ra
    -q
    --strict-markers
    --strict-config
    --cov=harborai
    --cov-report=html:reports/coverage/html
    --cov-report=xml:reports/coverage/coverage.xml
    --html=reports/html/report.html
    --self-contained-html
    --alluredir=reports/allure
    --tb=short
    --maxfail=5
    -p no:warnings

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

markers =
    unit: 单元测试
    integration: 集成测试
    performance: 性能测试
    security: 安全测试
    slow: 慢速测试
    critical: 关键测试
    p0: P0优先级测试
    p1: P1优先级测试
    p2: P2优先级测试
    p3: P3优先级测试
    deepseek: DeepSeek相关测试
    ernie: ERNIE相关测试
    doubao: Doubao相关测试
    reasoning: 推理模型测试
    non_reasoning: 非推理模型测试
    async_test: 异步测试
    stream_test: 流式测试
    mock: Mock测试（默认执行）
    real_api: 真实API测试（需手动开启）
    standard_alignment: 标准格式对齐测试
    reasoning_content: 推理内容检测测试
    structured_output: 结构化输出测试
    native_schema: 原生Schema测试
    agently_schema: Agently Schema测试
    resource_monitoring: 资源监控测试
    fallback_strategy: 降级策略测试
    cost_tracking: 成本统计测试

filterwarnings =
    ignore::UserWarning
    ignore::DeprecationWarning

asyncio_mode = auto
```

#### 3.1.2 全局配置 (conftest.py)

```python
"""
全局测试配置和夹具定义
"""
import os
import sys
import pytest
import asyncio
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from harborai import HarborAI
from tests.fixtures.client_fixtures import *
from tests.fixtures.data_fixtures import *
from tests.fixtures.mock_fixtures import *
from tests.fixtures.performance_fixtures import *
from tests.utils.test_helpers import TestConfig

# 测试配置
TEST_CONFIG = TestConfig()

def pytest_configure(config):
    """pytest配置钩子"""
    # 设置测试环境变量
    os.environ['HARBORAI_TEST_MODE'] = 'true'
    os.environ['HARBORAI_LOG_LEVEL'] = 'DEBUG'
    
    # 创建报告目录
    os.makedirs('reports/html', exist_ok=True)
    os.makedirs('reports/allure', exist_ok=True)
    os.makedirs('reports/performance', exist_ok=True)
    os.makedirs('reports/coverage', exist_ok=True)

def pytest_collection_modifyitems(config, items):
    """修改测试项收集"""
    # 为异步测试添加标记
    for item in items:
        if 'async' in item.name or 'await' in str(item.function):
            item.add_marker(pytest.mark.async_test)
        
        # 根据文件路径添加模块标记
        if 'performance' in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif 'security' in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif 'integration' in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config() -> TestConfig:
    """测试配置夹具"""
    return TEST_CONFIG

@pytest.fixture(autouse=True)
def setup_test_environment(test_config):
    """自动设置测试环境"""
    # 测试前设置
    original_env = os.environ.copy()
    
    # 设置测试专用环境变量
    test_env = {
        'HARBORAI_TEST_MODE': 'true',
        'HARBORAI_LOG_LEVEL': 'DEBUG',
        'HARBORAI_ENABLE_LOGGING': 'false',  # 默认关闭日志避免干扰
    }
    
    for key, value in test_env.items():
        os.environ[key] = value
    
    yield
    
    # 测试后清理
    os.environ.clear()
    os.environ.update(original_env)
```

### 3.2 测试夹具设计

#### 3.2.1 客户端夹具 (fixtures/client\_fixtures.py)

```python
"""
客户端相关测试夹具
"""
import pytest
from typing import Dict, Any
from unittest.mock import Mock, patch

from harborai import HarborAI
from tests.utils.test_helpers import get_test_credentials

@pytest.fixture
def mock_harborai_client():
    """Mock HarborAI客户端"""
    with patch('harborai.HarborAI') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        # 配置默认的mock响应
        mock_instance.chat.completions.create.return_value = Mock(
            choices=[Mock(
                message=Mock(
                    content="测试响应",
                    role="assistant"
                )
            )],
            usage=Mock(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            ),
            model="test-model",
            id="test-id"
        )
        
        yield mock_instance

@pytest.fixture
def real_harborai_client():
    """真实HarborAI客户端（用于集成测试）- 双重门控"""
    # 检查环境变量门控
    if not os.getenv('ENABLE_REAL_API_TESTS', 'false').lower() == 'true':
        pytest.skip("真实API测试未启用，设置ENABLE_REAL_API_TESTS=true启用")
    
    credentials = get_test_credentials()
    if not credentials:
        pytest.skip("缺少真实API凭证")
    
    client = HarborAI(
        api_key=credentials.get('api_key'),
        base_url=credentials.get('base_url')
    )
    
    yield client

@pytest.fixture
def mock_reasoning_response():
    """Mock推理模型响应 - 包含reasoning_content"""
    return Mock(
        choices=[Mock(
            message=Mock(
                content="基于以上分析，我认为...",
                role="assistant",
                reasoning_content="让我仔细思考这个问题。首先，我需要分析..."
            ),
            finish_reason="stop",
            index=0
        )],
        usage=Mock(
            prompt_tokens=20,
            completion_tokens=50,
            total_tokens=70
        ),
        model="deepseek-reasoner",
        id="chatcmpl-test-reasoning",
        object="chat.completion",
        created=1234567890
    )

@pytest.fixture
def mock_reasoning_stream_chunks():
    """Mock推理模型流式响应Chunks - 包含思考和答案片段"""
    return [
        # 思考片段
        Mock(
            choices=[Mock(
                delta=Mock(
                    reasoning_content="让我思考一下这个问题..."
                ),
                index=0,
                finish_reason=None
            )],
            id="chatcmpl-test-stream",
            object="chat.completion.chunk",
            created=1234567890,
            model="deepseek-reasoner"
        ),
        # 答案片段
        Mock(
            choices=[Mock(
                delta=Mock(
                    content="基于分析，答案是"
                ),
                index=0,
                finish_reason=None
            )],
            id="chatcmpl-test-stream",
            object="chat.completion.chunk",
            created=1234567890,
            model="deepseek-reasoner"
        ),
        # 结束片段
        Mock(
            choices=[Mock(
                delta=Mock(),
                index=0,
                finish_reason="stop"
            )],
            id="chatcmpl-test-stream",
            object="chat.completion.chunk",
            created=1234567890,
            model="deepseek-reasoner"
        )
    ]

@pytest.fixture(params=[
    {'vendor': 'deepseek', 'model': 'deepseek-chat', 'is_reasoning': False},
    {'vendor': 'deepseek', 'model': 'deepseek-reasoner', 'is_reasoning': True},
    {'vendor': 'ernie', 'model': 'ernie-3.5-8k', 'is_reasoning': False},
    {'vendor': 'ernie', 'model': 'ernie-4.0-turbo-8k', 'is_reasoning': False},
    {'vendor': 'ernie', 'model': 'ernie-x1-turbo-32k', 'is_reasoning': True},
    {'vendor': 'doubao', 'model': 'doubao-1-5-pro-32k-character-250715', 'is_reasoning': False},
    {'vendor': 'doubao', 'model': 'doubao-seed-1-6-250615', 'is_reasoning': True}
])
def multi_vendor_client(request):
    """多厂商客户端参数化夹具"""
    vendor_config = request.param
    credentials = get_test_credentials(vendor_config['vendor'])
    
    if not credentials:
        pytest.skip(f"缺少{vendor_config['vendor']}凭证")
    
    client = HarborAI(
        api_key=credentials.get('api_key'),
        base_url=credentials.get('base_url')
    )
    
    yield {
        'client': client,
        'vendor': vendor_config['vendor'],
        'model': vendor_config['model']
    }
```

#### 3.2.2 数据夹具 (fixtures/data\_fixtures.py)

```python
"""
测试数据相关夹具
"""
import pytest
import json
from typing import Dict, List, Any
from pathlib import Path

@pytest.fixture
def test_messages():
    """标准测试消息"""
    return [
        {"role": "user", "content": "你好，请介绍一下你自己。"}
    ]

@pytest.fixture
def complex_test_messages():
    """复杂测试消息（多轮对话）"""
    return [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "请解释什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是人工智能的一个分支..."},
        {"role": "user", "content": "能给个具体例子吗？"}
    ]

@pytest.fixture
def json_schema_simple():
    """简单JSON Schema"""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0}
        },
        "required": ["name", "age"]
    }

@pytest.fixture
def json_schema_complex():
    """复杂JSON Schema"""
    return {
        "type": "object",
        "properties": {
            "user_info": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                    "preferences": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "email"]
            },
            "metadata": {
                "type": "object",
                "additionalProperties": True
            }
        },
        "required": ["user_info"]
    }

@pytest.fixture
def performance_test_data():
    """性能测试数据"""
    return {
        "short_message": "Hello",
        "medium_message": "请详细解释" * 50,
        "long_message": "这是一个很长的测试消息" * 200,
        "concurrent_levels": [1, 10, 50, 100, 500],
        "test_iterations": 1000
    }

@pytest.fixture
def load_test_schemas():
    """加载测试Schema文件"""
    schema_dir = Path(__file__).parent.parent / "data" / "schemas"
    schemas = {}
    
    for schema_file in schema_dir.glob("*.json"):
        with open(schema_file, 'r', encoding='utf-8') as f:
            schemas[schema_file.stem] = json.load(f)
    
    return schemas

@pytest.fixture
def performance_test_models():
    """性能测试模型配置"""
    return [
        {'vendor': 'deepseek', 'model': 'deepseek-chat', 'expected_latency': 1.5, 'is_reasoning': False},
        {'vendor': 'deepseek', 'model': 'deepseek-reasoner', 'expected_latency': 3.0, 'is_reasoning': True},
        {'vendor': 'ernie', 'model': 'ernie-3.5-8k', 'expected_latency': 2.0, 'is_reasoning': False},
        {'vendor': 'ernie', 'model': 'ernie-4.0-turbo-8k', 'expected_latency': 1.8, 'is_reasoning': False},
        {'vendor': 'ernie', 'model': 'ernie-x1-turbo-32k', 'expected_latency': 3.5, 'is_reasoning': True},
        {'vendor': 'doubao', 'model': 'doubao-1-5-pro-32k-character-250715', 'expected_latency': 2.2, 'is_reasoning': False},
        {'vendor': 'doubao', 'model': 'doubao-seed-1-6-250615', 'expected_latency': 3.2, 'is_reasoning': True}
    ]

@pytest.fixture
def reasoning_models():
    """推理模型配置"""
    return [
        {'vendor': 'deepseek', 'model': 'deepseek-reasoner', 'supports_reasoning': True},
        {'vendor': 'ernie', 'model': 'ernie-x1-turbo-32k', 'supports_reasoning': True},
        {'vendor': 'doubao', 'model': 'doubao-seed-1-6-250615', 'supports_reasoning': True}
    ]

@pytest.fixture
def non_reasoning_models():
    """非推理模型配置"""
    return [
        {'vendor': 'deepseek', 'model': 'deepseek-chat', 'supports_reasoning': False},
        {'vendor': 'ernie', 'model': 'ernie-3.5-8k', 'supports_reasoning': False},
        {'vendor': 'ernie', 'model': 'ernie-4.0-turbo-8k', 'supports_reasoning': False},
        {'vendor': 'doubao', 'model': 'doubao-1-5-pro-32k-character-250715', 'supports_reasoning': False}
    ]
```

### 3.3 测试工具设计

#### 3.3.1 测试辅助函数 (utils/test\_helpers.py)

```python
"""
测试辅助函数
"""
import os
import time
import json
import asyncio
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TestConfig:
    """测试配置类 - 硬断言性能阈值"""
    test_timeout: int = 30
    # 硬断言性能阈值
    wrapper_overhead_threshold_ms: float = 1.0  # 封装开销 <1ms (Mock环境)
    concurrent_success_rate_threshold: float = 0.999  # 高并发成功率 >99.9%
    ttfb_threshold_ms: float = 100.0  # 首字节时间 <100ms (Mock环境)
    chunk_interval_threshold_ms: float = 50.0  # Chunk间隔 <50ms (Mock环境)
    memory_growth_threshold_mb: float = 10.0  # 内存增长 <10MB
    concurrent_limit: int = 100
    retry_attempts: int = 3
    
    def __post_init__(self):
        # 从环境变量加载配置
        self.test_timeout = int(os.getenv('TEST_TIMEOUT', self.test_timeout))
        self.wrapper_overhead_threshold_ms = float(os.getenv('WRAPPER_OVERHEAD_THRESHOLD_MS', self.wrapper_overhead_threshold_ms))
        self.concurrent_success_rate_threshold = float(os.getenv('CONCURRENT_SUCCESS_RATE_THRESHOLD', self.concurrent_success_rate_threshold))
        self.ttfb_threshold_ms = float(os.getenv('TTFB_THRESHOLD_MS', self.ttfb_threshold_ms))
        self.chunk_interval_threshold_ms = float(os.getenv('CHUNK_INTERVAL_THRESHOLD_MS', self.chunk_interval_threshold_ms))
        self.memory_growth_threshold_mb = float(os.getenv('MEMORY_GROWTH_THRESHOLD_MB', self.memory_growth_threshold_mb))
        self.concurrent_limit = int(os.getenv('CONCURRENT_LIMIT', self.concurrent_limit))
        self.retry_attempts = int(os.getenv('RETRY_ATTEMPTS', self.retry_attempts))

def get_test_credentials(vendor: str = 'deepseek') -> Optional[Dict[str, str]]:
    """获取测试凭证 - 从.env文件动态加载，无硬编码默认值"""
    # 加载.env文件
    from dotenv import load_dotenv
    load_dotenv()
    
    vendor_configs = {
        'deepseek': {
            'api_key': os.getenv('DEEPSEEK_API_KEY'),
            'base_url': os.getenv('DEEPSEEK_BASE_URL')
        },
        'ernie': {
            'api_key': os.getenv('WENXIN_API_KEY'),  # 使用.env中的WENXIN_API_KEY
            'base_url': os.getenv('WENXIN_BASE_URL')  # 使用.env中的WENXIN_BASE_URL
        },
        'doubao': {
            'api_key': os.getenv('DOUBAO_API_KEY'),
            'base_url': os.getenv('DOUBAO_BASE_URL')
        }
    }
    
    config = vendor_configs.get(vendor, {})
    # 严格检查：必须有API密钥和基础URL
    if not config.get('api_key') or not config.get('base_url'):
        return None
    
    return config

def measure_execution_time(func):
    """测量函数执行时间装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # 将执行时间添加到结果中
        if hasattr(result, '__dict__'):
            result.execution_time = execution_time
        
        return result
    return wrapper

async def measure_async_execution_time(coro):
    """测量异步函数执行时间"""
    start_time = time.perf_counter()
    result = await coro
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    if hasattr(result, '__dict__'):
        result.execution_time = execution_time
    
    return result

def validate_standard_response_format(response) -> bool:
    """验证响应格式是否符合标准格式 - 字段级严格对齐"""
    required_fields = ['choices', 'usage', 'model', 'id', 'object', 'created']
    
    for field in required_fields:
        if not hasattr(response, field):
            return False
    
    # 验证choices结构
    if not response.choices or len(response.choices) == 0:
        return False
    
    choice = response.choices[0]
    # 严格验证choice字段
    choice_required_fields = ['index', 'message', 'finish_reason']
    for field in choice_required_fields:
        if not hasattr(choice, field):
            return False
    
    # 验证message结构
    message = choice.message
    message_required_fields = ['role', 'content']
    for field in message_required_fields:
        if not hasattr(message, field):
            return False
    
    # 验证usage结构
    usage = response.usage
    usage_required_fields = ['prompt_tokens', 'completion_tokens', 'total_tokens']
    for field in usage_required_fields:
        if not hasattr(usage, field):
            return False
    
    # 验证finish_reason有效值
    valid_finish_reasons = ['stop', 'length', 'function_call', 'tool_calls', 'content_filter']
    if choice.finish_reason not in valid_finish_reasons:
        return False
    
    return True

def validate_standard_chunk_format(chunk) -> bool:
    """验证流式响应Chunk格式是否符合标准格式"""
    required_fields = ['id', 'object', 'created', 'model', 'choices']
    
    for field in required_fields:
        if not hasattr(chunk, field):
            return False
    
    # 验证object类型
    if chunk.object != 'chat.completion.chunk':
        return False
    
    # 验证choices结构
    if not chunk.choices or len(chunk.choices) == 0:
        return False
    
    choice = chunk.choices[0]
    choice_required_fields = ['index', 'delta']
    for field in choice_required_fields:
        if not hasattr(choice, field):
            return False
    
    return True

def validate_reasoning_content_detection(response, is_reasoning_model: bool) -> bool:
    """验证推理模型的reasoning_content自动检测"""
    if not is_reasoning_model:
        # 非推理模型不应该有reasoning_content
        return not hasattr(response.choices[0].message, 'reasoning_content')
    
    # 推理模型应该有reasoning_content字段
    message = response.choices[0].message
    if not hasattr(message, 'reasoning_content'):
        return False
    
    # reasoning_content应该是字符串且非空
    reasoning_content = message.reasoning_content
    return isinstance(reasoning_content, str) and len(reasoning_content.strip()) > 0

def validate_structured_output_format(response, expected_schema: Dict[str, Any]) -> bool:
    """验证结构化输出格式"""
    if not hasattr(response.choices[0].message, 'parsed'):
        return False
    
    parsed_content = response.choices[0].message.parsed
    
    # 使用jsonschema验证
    try:
        import jsonschema
        jsonschema.validate(parsed_content, expected_schema)
        return True
    except jsonschema.ValidationError:
        return False
    except Exception:
        return False
    
    # 验证usage结构
    usage_fields = ['prompt_tokens', 'completion_tokens', 'total_tokens']
    for field in usage_fields:
        if not hasattr(response.usage, field):
            return False
    
    return True

def sanitize_api_key(text: str) -> str:
    """脱敏API密钥"""
    import re
    # 匹配常见的API密钥格式
    patterns = [
        r'sk-[a-zA-Z0-9]{48}',  # 标准格式
        r'[a-zA-Z0-9]{32,}',    # 通用长字符串
    ]
    
    for pattern in patterns:
        text = re.sub(pattern, '***REDACTED***', text)
    
    return text

def load_test_data(file_name: str) -> Any:
    """加载测试数据文件"""
    data_dir = Path(__file__).parent.parent / "data" / "test_cases"
    file_path = data_dir / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(f"测试数据文件不存在: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_name.endswith('.json'):
            return json.load(f)
        else:
            return f.read()

class AsyncTestHelper:
    """异步测试辅助类"""
    
    @staticmethod
    async def run_concurrent_tasks(tasks: List, max_concurrent: int = 10):
        """运行并发任务"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_task(task):
            async with semaphore:
                return await task
        
        bounded_tasks = [bounded_task(task) for task in tasks]
        return await asyncio.gather(*bounded_tasks, return_exceptions=True)
    
    @staticmethod
    async def measure_concurrent_performance(task_func, concurrent_levels: List[int], iterations: int = 100):
        """测量并发性能"""
        results = {}
        
        for level in concurrent_levels:
            tasks = [task_func() for _ in range(iterations)]
            
            start_time = time.perf_counter()
            results_list = await AsyncTestHelper.run_concurrent_tasks(tasks, level)
            end_time = time.perf_counter()
            
            # 统计成功和失败
            successes = sum(1 for r in results_list if not isinstance(r, Exception))
            failures = iterations - successes
            
            results[level] = {
                'total_time': end_time - start_time,
                'avg_time': (end_time - start_time) / iterations,
                'success_rate': successes / iterations,
                'qps': iterations / (end_time - start_time),
                'successes': successes,
                'failures': failures
            }
        
        return results
```

## 4. 性能测试框架设计

### 4.1 性能测试架构

#### 4.1.1 性能测试基础类 (utils/performance\_utils.py)

```python
"""
性能测试工具类
"""
import time
import psutil
import asyncio
import statistics
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
from memory_profiler import profile

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    response_times: List[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    timeout_count: int = 0
    start_time: float = 0
    end_time: float = 0
    
    @property
    def total_requests(self) -> int:
        return self.success_count + self.error_count + self.timeout_count
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.success_count / self.total_requests
    
    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.error_count / self.total_requests
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def p50_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(0.99 * len(sorted_times))
        return sorted_times[index] if index < len(sorted_times) else sorted_times[-1]
    
    @property
    def qps(self) -> float:
        duration = self.end_time - self.start_time
        if duration <= 0:
            return 0.0
        return self.total_requests / duration
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_requests': self.total_requests,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'timeout_count': self.timeout_count,
            'success_rate': self.success_rate,
            'error_rate': self.error_rate,
            'avg_response_time': self.avg_response_time,
            'p50_response_time': self.p50_response_time,
            'p95_response_time': self.p95_response_time,
            'p99_response_time': self.p99_response_time,
            'qps': self.qps,
            'duration': self.end_time - self.start_time
        }

@dataclass
class ResourceMetrics:
    """资源使用指标"""
    cpu_percent: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)
    network_io: List[Dict[str, int]] = field(default_factory=list)
    file_handles: List[int] = field(default_factory=list)
    thread_count: List[int] = field(default_factory=list)
    
    @property
    def avg_cpu_percent(self) -> float:
        return statistics.mean(self.cpu_percent) if self.cpu_percent else 0.0
    
    @property
    def max_memory_mb(self) -> float:
        return max(self.memory_mb) if self.memory_mb else 0.0
    
    @property
    def avg_memory_mb(self) -> float:
        return statistics.mean(self.memory_mb) if self.memory_mb else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'avg_cpu_percent': self.avg_cpu_percent,
            'max_cpu_percent': max(self.cpu_percent) if self.cpu_percent else 0.0,
            'avg_memory_mb': self.avg_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'max_file_handles': max(self.file_handles) if self.file_handles else 0,
            'max_thread_count': max(self.thread_count) if self.thread_count else 0
        }

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.resource_metrics = ResourceMetrics()
        self._monitoring = False
        self._monitor_thread = None
        self._process = psutil.Process()
    
    def start_monitoring(self):
        """开始监控"""
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_resources(self):
        """监控资源使用"""
        while self._monitoring:
            try:
                # CPU使用率
                cpu_percent = self._process.cpu_percent()
                self.resource_metrics.cpu_percent.append(cpu_percent)
                
                # 内存使用
                memory_info = self._process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.resource_metrics.memory_mb.append(memory_mb)
                
                # 网络IO
                try:
                    net_io = self._process.io_counters()
                    self.resource_metrics.network_io.append({
                        'read_bytes': net_io.read_bytes,
                        'write_bytes': net_io.write_bytes
                    })
                except (psutil.AccessDenied, AttributeError):
                    pass
                
                # 文件句柄数
                try:
                    file_handles = self._process.num_fds() if hasattr(self._process, 'num_fds') else len(self._process.open_files())
                    self.resource_metrics.file_handles.append(file_handles)
                except (psutil.AccessDenied, AttributeError):
                    pass
                
                # 线程数
                thread_count = self._process.num_threads()
                self.resource_metrics.thread_count.append(thread_count)
                
            except psutil.NoSuchProcess:
                break
            
            time.sleep(self.interval)

class PerformanceTester:
    """性能测试器"""
    
    def __init__(self, monitor_resources: bool = True):
        self.monitor_resources = monitor_resources
        self.monitor = PerformanceMonitor() if monitor_resources else None
    
    async def run_async_performance_test(
        self,
        test_func: Callable,
        concurrent_levels: List[int],
        iterations_per_level: int = 100,
        timeout: float = 30.0
    ) -> Dict[int, Dict[str, Any]]:
        """运行异步性能测试"""
        results = {}
        
        for level in concurrent_levels:
            print(f"测试并发级别: {level}")
            
            if self.monitor:
                self.monitor.start_monitoring()
            
            metrics = PerformanceMetrics()
            metrics.start_time = time.perf_counter()
            
            # 创建并发任务
            semaphore = asyncio.Semaphore(level)
            
            async def bounded_test():
                async with semaphore:
                    start = time.perf_counter()
                    try:
                        result = await asyncio.wait_for(test_func(), timeout=timeout)
                        end = time.perf_counter()
                        metrics.response_times.append(end - start)
                        metrics.success_count += 1
                        return result
                    except asyncio.TimeoutError:
                        metrics.timeout_count += 1
                        raise
                    except Exception as e:
                        metrics.error_count += 1
                        raise
            
            # 执行测试
            tasks = [bounded_test() for _ in range(iterations_per_level)]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            metrics.end_time = time.perf_counter()
            
            if self.monitor:
                self.monitor.stop_monitoring()
                resource_data = self.monitor.resource_metrics.to_dict()
            else:
                resource_data = {}
            
            results[level] = {
                'performance': metrics.to_dict(),
                'resources': resource_data
            }
            
            # 重置监控器
            if self.monitor:
                self.monitor = PerformanceMonitor()
        
        return results
    
    def run_sync_performance_test(
        self,
        test_func: Callable,
        concurrent_levels: List[int],
        iterations_per_level: int = 100,
        timeout: float = 30.0
    ) -> Dict[int, Dict[str, Any]]:
        """运行同步性能测试"""
        results = {}
        
        for level in concurrent_levels:
            print(f"测试并发级别: {level}")
            
            if self.monitor:
                self.monitor.start_monitoring()
            
            metrics = PerformanceMetrics()
            metrics.start_time = time.perf_counter()
            
            def single_test():
                start = time.perf_counter()
                try:
                    result = test_func()
                    end = time.perf_counter()
                    metrics.response_times.append(end - start)
                    metrics.success_count += 1
                    return result
                except Exception as e:
                    metrics.error_count += 1
                    raise
            
            # 使用线程池执行并发测试
            with ThreadPoolExecutor(max_workers=level) as executor:
                futures = [executor.submit(single_test) for _ in range(iterations_per_level)]
                
                for future in futures:
                    try:
                        future.result(timeout=timeout)
                    except Exception:
                        pass
            
            metrics.end_time = time.perf_counter()
            
            if self.monitor:
                self.monitor.stop_monitoring()
                resource_data = self.monitor.resource_metrics.to_dict()
            else:
                resource_data = {}
            
            results[level] = {
                'performance': metrics.to_dict(),
                'resources': resource_data
            }
            
            # 重置监控器
            if self.monitor:
                self.monitor = PerformanceMonitor()
        
        return results

class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_file = baseline_file
        self.baseline_data = self._load_baseline() if baseline_file else {}
    
    def _load_baseline(self) -> Dict[str, Any]:
        """加载基准数据"""
        try:
            import json
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
    
    def save_baseline(self, data: Dict[str, Any]):
        """保存基准数据"""
        if self.baseline_file:
            import json
            with open(self.baseline_file, 'w') as f:
                json.dump(data, f, indent=2)
    
    def compare_with_baseline(self, current_data: Dict[str, Any], tolerance: float = 0.1) -> Dict[str, Any]:
        """与基准数据比较"""
        comparison = {
            'passed': True,
            'details': {}
        }
        
        for metric, current_value in current_data.items():
            if metric in self.baseline_data:
                baseline_value = self.baseline_data[metric]
                if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
                    # 计算变化百分比
                    change_percent = (current_value - baseline_value) / baseline_value if baseline_value != 0 else 0
                    
                    comparison['details'][metric] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'change_percent': change_percent,
                        'passed': abs(change_percent) <= tolerance
                    }
                    
                    if abs(change_percent) > tolerance:
                        comparison['passed'] = False
        
        return comparison
```

### 4.2 性能测试用例设计

#### 4.2.1 基础性能测试 (performance/test\_basic\_performance.py)

````python
"""
基础性能测试
"""
import pytest
import asyncio
import time
from typing import Dict, Any

from tests.utils.performance_utils import PerformanceTester, BenchmarkRunner
from tests.utils.test_helpers import get_test_credentials
from harborai import HarborAI

class TestBasicPerformance:
    """基础性能测试类"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """测试设置"""
        self.performance_tester = PerformanceTester(monitor_resources=True)
        self.benchmark_runner = BenchmarkRunner('tests/data/performance_baselines/basic_performance.json')
        
        # 获取测试凭证
        self.credentials = get_test_credentials('deepseek')
        if not self.credentials:
            pytest.skip("缺少DeepSeek测试凭证")
        
        self.client = HarborAI(
            api_key=self.credentials['api_key'],
            base_url=self.credentials['base_url']
        )
    
    @pytest.mark.performance
    @pytest.mark.p0
    def test_wrapper_overhead_benchmark(self):
        """测试封装开销基准 - O-001"""
        # 使用对应厂商的原生SDK进行对比
        # 这里使用标准的HTTP客户端作为基准
        import requests
        
        test_message = [{"role": "user", "content": "Hello"}]
        iterations = 100
        
        # 预热
        for _ in range(10):
            try:
                self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=test_message
                )
            except Exception:
                pass
        
        # 测试基准性能（使用HTTP直接调用）
        native_times = []
        test_model = "deepseek-chat"  # 使用实际可用的模型
        
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                # 使用requests直接调用API作为基准
                response = requests.post(
                    f"{self.credentials['base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.credentials['api_key']}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": test_model,
                        "messages": test_message
                    }
                )
                end = time.perf_counter()
                if response.status_code == 200:
                    native_times.append(end - start)
            except Exception:
                pass
        
        # 测试HarborAI
        harbor_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            try:
                self.client.chat.completions.create(
                    model=test_model,
                    messages=test_message
                )
                end = time.perf_counter()
                harbor_times.append(end - start)
            except Exception:
                pass
        
        # 计算开销
        if native_times and harbor_times:
            native_avg = sum(native_times) / len(native_times)
            harbor_avg = sum(harbor_times) / len(harbor_times)
            overhead = harbor_avg - native_avg
            
            print(f"基准HTTP调用平均时间: {native_avg*1000:.2f}ms")
            print(f"HarborAI平均时间: {harbor_avg*1000:.2f}ms")
            print(f"封装开销: {overhead*1000:.2f}ms")
            
            # 验证开销小于1ms
            assert overhead < 0.001, f"封装开销 {overhead*1000:.2f}ms 超过1ms阈值"
            
            # 保存基准数据
            baseline_data = {
                'baseline_avg_time': native_avg,
                'harbor_avg_time': harbor_avg,
                'wrapper_overhead': overhead,
                'test_model': test_model
            }
            self.benchmark_runner.save_baseline(baseline_data)
    
    @pytest.mark.performance
    @pytest.mark.p0
    @pytest.mark.asyncio
    async def test_concurrent_success_rate(self):
        """测试高并发成功率 - O-002"""
        concurrent_levels = [10, 50, 100]
        iterations_per_level = 100
        
        async def single_request():
            """单个请求"""
            return await self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "测试消息"}]
            )
        
        results = await self.performance_tester.run_async_performance_test(
            single_request,
            concurrent_levels,
            iterations_per_level,
            timeout=30.0
        )
        
        # 验证成功率
        for level, result in results.items():
            performance = result['performance']
            success_rate = performance['success_rate']
            
            print(f"并发级别 {level}: 成功率 {success_rate:.4f} ({success_rate*100:.2f}%)")
            print(f"  QPS: {performance['qps']:.2f}")
            print(f"  P95响应时间: {performance['p95_response_time']*1000:.2f}ms")
            
            # 验证成功率大于99.9%
            assert success_rate > 0.999, f"并发级别{level}成功率{success_rate:.4f}低于99.9%阈值"
            
            # 验证P95响应时间小于2秒
            assert performance['p95_response_time'] < 2.0, f"P95响应时间{performance['p95_response_time']:.2f}s超过2s阈值"
        
        # 保存性能数据
        self.benchmark_runner.save_baseline({
            'concurrent_performance': results
        })

#### 4.2.2 推理模型专项测试 (functional/test_reasoning_models.py)
```python
"""
推理模型专项测试 - 模块D
"""
import pytest
import asyncio
from typing import Dict, Any

from tests.utils.test_helpers import get_test_credentials
from harborai import HarborAI

class TestReasoningModels:
    """推理模型测试类"""
    
    @pytest.mark.parametrize("model_config", [
        {"model": "deepseek-reasoner", "vendor": "deepseek"},
        {"model": "ernie-x1-turbo-32k", "vendor": "ernie"},
        {"model": "doubao-seed-1-6-250615", "vendor": "doubao"}
    ])
    @pytest.mark.p0
    def test_reasoning_model_basic(self, model_config):
        """测试推理模型基础功能 - D-001"""
        credentials = get_test_credentials(model_config['vendor'])
        if not credentials:
            pytest.skip(f"缺少{model_config['vendor']}测试凭证")
        
        client = HarborAI(
            api_key=credentials['api_key'],
            base_url=credentials['base_url']
        )
        
        model = model_config['model']
        
        # 测试推理模型的基础调用
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "请分析一下为什么1+1=2，详细说明推理过程"}
            ]
        )
        
        # 验证响应格式
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        
        # 验证推理内容（如果支持）
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = response.choices[0].message.reasoning_content
            if reasoning_content:
                assert len(reasoning_content) > 0
                print(f"推理过程: {reasoning_content[:200]}...")
        
        # 验证token使用
        assert response.usage.total_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
    
    @pytest.mark.parametrize("model_config", [
        {"model": "deepseek-reasoner", "vendor": "deepseek"},
        {"model": "ernie-x1-turbo-32k", "vendor": "ernie"},
        {"model": "doubao-seed-1-6-250615", "vendor": "doubao"}
    ])
    @pytest.mark.p0
    @pytest.mark.asyncio
    async def test_reasoning_model_streaming(self, model_config):
        """测试推理模型流式输出 - D-002"""
        credentials = get_test_credentials(model_config['vendor'])
        if not credentials:
            pytest.skip(f"缺少{model_config['vendor']}测试凭证")
        
        client = HarborAI(
            api_key=credentials['api_key'],
            base_url=credentials['base_url']
        )
        
        model = model_config['model']
        
        # 测试流式调用
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "请逐步分析如何解决一个复杂的数学问题"}
            ],
            stream=True
        )
        
        content_chunks = []
        reasoning_chunks = []
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                # 收集内容
                if hasattr(delta, 'content') and delta.content:
                    content_chunks.append(delta.content)
                
                # 收集推理内容
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_chunks.append(delta.reasoning_content)
        
        # 验证流式输出
        assert len(content_chunks) > 0, "未收到内容chunks"
        
        full_content = ''.join(content_chunks)
        assert len(full_content) > 0, "完整内容为空"
        
        # 如果有推理内容，验证推理流
        if reasoning_chunks:
            full_reasoning = ''.join(reasoning_chunks)
            assert len(full_reasoning) > 0, "推理内容为空"
            print(f"推理流长度: {len(reasoning_chunks)} chunks")
    
    @pytest.mark.parametrize("model_config", [
        {"model": "deepseek-reasoner", "vendor": "deepseek"},
        {"model": "ernie-x1-turbo-32k", "vendor": "ernie"},
        {"model": "doubao-seed-1-6-250615", "vendor": "doubao"}
    ])
    @pytest.mark.p1
    def test_reasoning_model_complex_task(self, model_config):
        """测试推理模型复杂任务处理 - D-003"""
        credentials = get_test_credentials(model_config['vendor'])
        if not credentials:
            pytest.skip(f"缺少{model_config['vendor']}测试凭证")
        
        client = HarborAI(
            api_key=credentials['api_key'],
            base_url=credentials['base_url']
        )
        
        model = model_config['model']
        
        # 复杂推理任务
        complex_prompt = """
        请分析以下逻辑推理问题：
        
        有三个人：Alice、Bob、Charlie
        - Alice说："Bob在说谎"
        - Bob说："Charlie在说谎"
        - Charlie说："Alice和Bob都在说谎"
        
        请分析谁在说真话，谁在说谎，并详细说明推理过程。
        """
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": complex_prompt}]
        )
        
        content = response.choices[0].message.content
        assert content is not None
        assert len(content) > 100, "复杂任务回答过短"
        
        # 检查是否包含推理关键词
        reasoning_keywords = ["分析", "推理", "因为", "所以", "假设", "结论"]
        found_keywords = [kw for kw in reasoning_keywords if kw in content]
        assert len(found_keywords) >= 2, f"回答缺少推理关键词，仅找到: {found_keywords}"
        
        # 验证推理内容（如果支持）
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = response.choices[0].message.reasoning_content
            if reasoning_content:
                assert len(reasoning_content) > 50, "推理过程过短"
                print(f"推理过程长度: {len(reasoning_content)} 字符")
    
    @pytest.mark.parametrize("model_config", [
        {"model": "deepseek-chat", "vendor": "deepseek"},
        {"model": "ernie-3.5-8k", "vendor": "ernie"},
        {"model": "ernie-4.0-turbo-8k", "vendor": "ernie"},
        {"model": "doubao-1-5-pro-32k-character-250715", "vendor": "doubao"}
    ])
    @pytest.mark.p1
    def test_non_reasoning_model_behavior(self, model_config):
        """测试非推理模型行为 - D-004"""
        credentials = get_test_credentials(model_config['vendor'])
        if not credentials:
            pytest.skip(f"缺少{model_config['vendor']}测试凭证")
        
        client = HarborAI(
            api_key=credentials['api_key'],
            base_url=credentials['base_url']
        )
        
        model = model_config['model']
        
        # 测试非推理模型
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "请简单介绍一下人工智能"}
            ]
        )
        
        # 验证基础响应
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        
        # 验证非推理模型不应该有reasoning_content
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_content = response.choices[0].message.reasoning_content
            # 非推理模型的reasoning_content应该为None或空
            assert reasoning_content is None or reasoning_content == "", \
                f"非推理模型不应该有推理内容: {reasoning_content}"
        
        # 验证token使用
        assert response.usage.total_tokens > 0
    
    @pytest.mark.p1
    def test_reasoning_vs_non_reasoning_comparison(self, reasoning_models, non_reasoning_models):
        """对比推理模型和非推理模型的行为差异 - D-005"""
        test_prompt = "请分析为什么天空是蓝色的，包括详细的科学原理"
        
        reasoning_results = []
        non_reasoning_results = []
        
        # 测试推理模型
        for model_config in reasoning_models[:1]:  # 只测试第一个推理模型以节省时间
            credentials = get_test_credentials(model_config['vendor'])
            if not credentials:
                continue
            
            client = HarborAI(
                api_key=credentials['api_key'],
                base_url=credentials['base_url']
            )
            
            try:
                response = client.chat.completions.create(
                    model=model_config['model'],
                    messages=[{"role": "user", "content": test_prompt}]
                )
                
                result = {
                    'model': model_config['model'],
                    'content_length': len(response.choices[0].message.content or ""),
                    'has_reasoning': hasattr(response.choices[0].message, 'reasoning_content') and 
                                   response.choices[0].message.reasoning_content is not None,
                    'reasoning_length': len(response.choices[0].message.reasoning_content or "") 
                                      if hasattr(response.choices[0].message, 'reasoning_content') else 0,
                    'total_tokens': response.usage.total_tokens
                }
                reasoning_results.append(result)
            except Exception as e:
                print(f"推理模型 {model_config['model']} 测试失败: {e}")
        
        # 测试非推理模型
        for model_config in non_reasoning_models[:1]:  # 只测试第一个非推理模型
            credentials = get_test_credentials(model_config['vendor'])
            if not credentials:
                continue
            
            client = HarborAI(
                api_key=credentials['api_key'],
                base_url=credentials['base_url']
            )
            
            try:
                response = client.chat.completions.create(
                    model=model_config['model'],
                    messages=[{"role": "user", "content": test_prompt}]
                )
                
                result = {
                    'model': model_config['model'],
                    'content_length': len(response.choices[0].message.content or ""),
                    'has_reasoning': hasattr(response.choices[0].message, 'reasoning_content') and 
                                   response.choices[0].message.reasoning_content is not None,
                    'reasoning_length': len(response.choices[0].message.reasoning_content or "") 
                                      if hasattr(response.choices[0].message, 'reasoning_content') else 0,
                    'total_tokens': response.usage.total_tokens
                }
                non_reasoning_results.append(result)
            except Exception as e:
                print(f"非推理模型 {model_config['model']} 测试失败: {e}")
        
        # 分析结果
        print("\n=== 推理模型 vs 非推理模型对比 ===")
        
        for result in reasoning_results:
            print(f"推理模型 {result['model']}:")
            print(f"  内容长度: {result['content_length']} 字符")
            print(f"  有推理内容: {result['has_reasoning']}")
            print(f"  推理内容长度: {result['reasoning_length']} 字符")
            print(f"  总token数: {result['total_tokens']}")
        
        for result in non_reasoning_results:
            print(f"非推理模型 {result['model']}:")
            print(f"  内容长度: {result['content_length']} 字符")
            print(f"  有推理内容: {result['has_reasoning']}")
            print(f"  推理内容长度: {result['reasoning_length']} 字符")
            print(f"  总token数: {result['total_tokens']}")
        
        # 验证推理模型和非推理模型的行为差异
        if reasoning_results and non_reasoning_results:
            reasoning_result = reasoning_results[0]
            non_reasoning_result = non_reasoning_results[0]
            
            # 推理模型应该有更详细的分析（通常更长的回答或推理内容）
            assert reasoning_result['content_length'] > 0
            assert non_reasoning_result['content_length'] > 0
            
            print(f"\n对比结果:")
            print(f"推理模型内容长度: {reasoning_result['content_length']}")
            print(f"非推理模型内容长度: {non_reasoning_result['content_length']}")
````

```
@pytest.mark.performance
@pytest.mark.p1
def test_memory_leak_detection(self):
    """测试内存泄露检测 - O-003"""
    import psutil
    import gc
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # 执行大量请求
    for i in range(1000):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"测试消息 {i}"}]
            )
            # 确保响应被处理
            _ = response.choices[0].message.content
        except Exception:
            pass
        
        # 每100次请求检查一次内存
        if i % 100 == 0:
            gc.collect()  # 强制垃圾回收
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = current_memory - initial_memory
            
            print(f"请求 {i}: 内存使用 {current_memory:.2f}MB, 增长 {memory_growth:.2f}MB")
            
            # 验证内存增长不超过50MB
            assert memory_growth < 50, f"内存增长{memory_growth:.2f}MB超过50MB阈值"
    
    # 最终内存检查
    gc.collect()
    final_memory = process.memory_info().rss / 1024 / 1024
    total_growth = final_memory - initial_memory
    
    print(f"总内存增长: {total_growth:.2f}MB")
    assert total_growth < 100, f"总内存增长{total_growth:.2f}MB超过100MB阈值"

@pytest.mark.performance
@pytest.mark.p1
@pytest.mark.asyncio
async def test_streaming_performance(self):
    """测试流式调用性能 - O-004"""
    ttfb_times = []  # Time To First Byte
    chunk_intervals = []
    
    for _ in range(10):  # 测试10次流式调用
        start_time = time.perf_counter()
        first_chunk_time = None
        last_chunk_time = start_time
        
        try:
            stream = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "请写一个100字的故事"}],
                stream=True
            )
            
            chunk_count = 0
            for chunk in stream:
                current_time = time.perf_counter()
                
                if first_chunk_time is None:
                    first_chunk_time = current_time
                    ttfb = first_chunk_time - start_time
                    ttfb_times.append(ttfb)
                else:
                    interval = current_time - last_chunk_time
                    chunk_intervals.append(interval)
                
                last_chunk_time = current_time
                chunk_count += 1
            
            print(f"流式调用完成: {chunk_count}个chunk")
            
        except Exception as e:
            print(f"流式调用失败: {e}")
    
    if ttfb_times:
        avg_ttfb = sum(ttfb_times) / len(ttfb_times)
        print(f"平均TTFB: {avg_ttfb*1000:.2f}ms")
        
        # 验证TTFB小于500ms
        assert avg_ttfb < 0.5, f"平均TTFB {avg_ttfb*1000:.2f}ms超过500ms阈值"
    
    if chunk_intervals:
        avg_interval = sum(chunk_intervals) / len(chunk_intervals)
        print(f"平均chunk间隔: {avg_interval*1000:.2f}ms")
        
        # 验证chunk间隔小于100ms
        assert avg_interval < 0.1, f"平均chunk间隔{avg_interval*1000:.2f}ms超过100ms阈值"
```

````

## 5. 测试环境配置与依赖管理

### 5.1 依赖管理 (requirements-test.txt)
```txt
# 核心测试框架
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-mock>=3.10.0
pytest-cov>=4.0.0
pytest-html>=3.1.0
pytest-benchmark>=4.0.0
pytest-xdist>=3.0.0

# 性能测试
locust>=2.14.0
psutil>=5.9.0
memory-profiler>=0.60.0
line-profiler>=4.0.0

# Mock工具
responses>=0.23.0
aioresponses>=0.7.4
requests-mock>=1.10.0

# 数据库测试
psycopg2-binary>=2.9.5
sqlalchemy>=2.0.0
alembic>=1.9.0

# 报告生成
allure-pytest>=2.12.0
jinja2>=3.1.0
markdown>=3.4.0

# 监控工具
prometheus-client>=0.16.0
grafana-api>=1.0.3

# 工具库
pydantic>=1.10.0
pyyaml>=6.0
click>=8.1.0
rich>=13.0.0
tabulate>=0.9.0

# 开发工具
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0

# 容器化
docker>=6.0.0
docker-compose>=1.29.0

# 网络工具
aiohttp>=3.8.0
httpx>=0.24.0
websockets>=11.0.0
````

### 5.2 Docker环境配置 (docker-compose.yml)

```yaml
version: '3.8'

services:
  # PostgreSQL数据库
  postgres:
    image: postgres:15-alpine
    container_name: harborai-test-postgres
    environment:
      POSTGRES_DB: harborai_test
      POSTGRES_USER: test_user
      POSTGRES_PASSWORD: test_password
      POSTGRES_HOST_AUTH_METHOD: trust
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init_db.sql:/docker-entrypoint-initdb.d/init_db.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U test_user -d harborai_test"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - test-network

  # Redis缓存（可选）
  redis:
    image: redis:7-alpine
    container_name: harborai-test-redis
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - test-network

  # Grafana监控（可选）
  grafana:
    image: grafana/grafana:latest
    container_name: harborai-test-grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./config/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./config/grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - test-network

  # Prometheus监控（可选）
  prometheus:
    image: prom/prometheus:latest
    container_name: harborai-test-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./config/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    networks:
      - test-network

volumes:
  postgres_data:
  redis_data:
  grafana_data:
  prometheus_data:

networks:
  test-network:
    driver: bridge
```

### 5.3 测试环境变量配置 (.env.test)

```env
# 测试模式
HARBORAI_TEST_MODE=true
HARBORAI_LOG_LEVEL=DEBUG
HARBORAI_ENABLE_LOGGING=false

# 数据库配置
DATABASE_URL=postgresql://test_user:test_password@localhost:5432/harborai_test
REDIS_URL=redis://localhost:6379/0

# API凭证（测试用）
DEEPSEEK_API_KEY=sk-d996b310528f44ffb1d7bf5b23b5313b
DEEPSEEK_BASE_URL=https://api.deepseek.com
ERNIE_API_KEY=bce-v3/ALTAK-zOkbkPd1PRzUP9Vlnv4iT/70d795c8c18da64a63a357228ff0c66c9bf3032a
ERNIE_BASE_URL=https://qianfan.baidubce.com/v2
DOUBAO_API_KEY=4ed46be9-4eb4-45f1-8576-d2fc3d115026
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3

# 性能测试配置
TEST_TIMEOUT=30
PERFORMANCE_THRESHOLD=1.0
CONCURRENT_LIMIT=100
RETRY_ATTEMPTS=3

# 监控配置
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
MONITORING_ENABLED=true
```

### 5.4 数据库初始化脚本 (scripts/init\_db.sql)

```sql
-- 创建测试数据库和用户
CREATE DATABASE harborai_test;
CREATE USER test_user WITH PASSWORD 'test_password';
GRANT ALL PRIVILEGES ON DATABASE harborai_test TO test_user;

-- 切换到测试数据库
\c harborai_test;

-- 创建测试表
CREATE TABLE IF NOT EXISTS test_requests (
    id SERIAL PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    vendor VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    messages JSONB NOT NULL,
    response JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_time FLOAT,
    status VARCHAR(20) DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    test_name VARCHAR(255) NOT NULL,
    concurrent_level INTEGER NOT NULL,
    success_rate FLOAT NOT NULL,
    avg_response_time FLOAT NOT NULL,
    p95_response_time FLOAT NOT NULL,
    qps FLOAT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS test_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    test_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'running',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB
);

-- 创建索引
CREATE INDEX idx_test_requests_vendor ON test_requests(vendor);
CREATE INDEX idx_test_requests_created_at ON test_requests(created_at);
CREATE INDEX idx_performance_metrics_test_name ON performance_metrics(test_name);
CREATE INDEX idx_test_sessions_session_id ON test_sessions(session_id);
```

## 6. 测试数据管理策略

### 6.1 测试数据分类

#### 6.1.1 静态测试数据

* **Schema文件**: JSON Schema定义文件

* **Mock响应**: 预定义的API响应数据

* **测试用例**: 标准化的测试输入数据

* **性能基准**: 历史性能数据基准

#### 6.1.2 动态测试数据

* **随机生成**: 使用Faker库生成随机测试数据

* **参数化数据**: pytest参数化测试数据

* **环境相关**: 根据测试环境动态调整的数据

### 6.2 数据生成器设计 (utils/data\_generators.py)

```python
"""
测试数据生成器
"""
import random
import string
import json
from typing import Dict, List, Any, Optional
from faker import Faker
from datetime import datetime, timedelta

fake = Faker(['zh_CN', 'en_US'])

class TestDataGenerator:
    """测试数据生成器"""
    
    @staticmethod
    def generate_chat_messages(count: int = 1, complexity: str = 'simple') -> List[Dict[str, str]]:
        """生成聊天消息"""
        messages = []
        
        if complexity == 'simple':
            for _ in range(count):
                messages.append({
                    "role": "user",
                    "content": fake.sentence()
                })
        elif complexity == 'complex':
            # 添加系统消息
            messages.append({
                "role": "system",
                "content": "你是一个有用的AI助手。"
            })
            
            # 添加多轮对话
            for i in range(count):
                messages.append({
                    "role": "user",
                    "content": fake.text(max_nb_chars=200)
                })
                if i < count - 1:  # 不是最后一条
                    messages.append({
                        "role": "assistant",
                        "content": fake.text(max_nb_chars=300)
                    })
        
        return messages
    
    @staticmethod
    def generate_json_schema(complexity: str = 'simple') -> Dict[str, Any]:
        """生成JSON Schema"""
        if complexity == 'simple':
            return {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 0},
                    "email": {"type": "string", "format": "email"}
                },
                "required": ["name"]
            }
        elif complexity == 'complex':
            return {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "profile": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "bio": {"type": "string"},
                                    "tags": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "preferences": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "category": {"type": "string"},
                                "value": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["user"]
            }
    
    @staticmethod
    def generate_api_key(vendor: str = 'deepseek') -> str:
        """生成测试用API密钥"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    
    @staticmethod
    def generate_performance_data(concurrent_levels: List[int]) -> Dict[str, Any]:
        """生成性能测试数据"""
        data = {}
        
        for level in concurrent_levels:
            # 模拟性能数据（响应时间随并发增加而增加）
            base_time = 0.1  # 100ms基础时间
            response_time = base_time + (level * 0.001)  # 每增加1并发增加1ms
            
            data[level] = {
                'avg_response_time': response_time,
                'p95_response_time': response_time * 1.5,
                'success_rate': max(0.95, 1.0 - (level * 0.0001)),
                'qps': min(1000, level * 10)
            }
        
        return data
```

### 6.3 数据脱敏工具 (utils/security\_utils.py)

```python
"""
安全和数据脱敏工具
"""
import re
import hashlib
from typing import Any, Dict, List, Union

class DataSanitizer:
    """数据脱敏器"""
    
    # 敏感数据模式
    SENSITIVE_PATTERNS = {
        'api_key': [
            r'[a-zA-Z0-9]{32,64}',  # 通用API Key
        ],
        'email': [
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        ],
        'phone': [
            r'\+?1?\d{9,15}',
            r'\d{3}-\d{3}-\d{4}',
            r'\(\d{3}\)\s*\d{3}-\d{4}'
        ],
        'credit_card': [
            r'\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}'
        ]
    }
    
    @classmethod
    def sanitize_text(cls, text: str, mask_char: str = '*') -> str:
        """脱敏文本中的敏感信息"""
        if not isinstance(text, str):
            return text
        
        sanitized = text
        
        for category, patterns in cls.SENSITIVE_PATTERNS.items():
            for pattern in patterns:
                # 保留前后几个字符，中间用mask_char替换
                def replace_func(match):
                    matched = match.group(0)
                    if len(matched) <= 6:
                        return mask_char * len(matched)
                    else:
                        return matched[:2] + mask_char * (len(matched) - 4) + matched[-2:]
                
                sanitized = re.sub(pattern, replace_func, sanitized)
        
        return sanitized
    
    @classmethod
    def sanitize_dict(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """脱敏字典中的敏感信息"""
        if not isinstance(data, dict):
            return data
        
        sanitized = {}
        
        for key, value in data.items():
            # 检查键名是否包含敏感信息
            if any(sensitive in key.lower() for sensitive in ['key', 'token', 'password', 'secret']):
                if isinstance(value, str) and len(value) > 6:
                    sanitized[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                else:
                    sanitized[key] = '***REDACTED***'
            elif isinstance(value, str):
                sanitized[key] = cls.sanitize_text(value)
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = cls.sanitize_list(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    @classmethod
    def sanitize_list(cls, data: List[Any]) -> List[Any]:
        """脱敏列表中的敏感信息"""
        if not isinstance(data, list):
            return data
        
        sanitized = []
        
        for item in data:
            if isinstance(item, str):
                sanitized.append(cls.sanitize_text(item))
            elif isinstance(item, dict):
                sanitized.append(cls.sanitize_dict(item))
            elif isinstance(item, list):
                sanitized.append(cls.sanitize_list(item))
            else:
                sanitized.append(item)
        
        return sanitized
    
    @classmethod
    def hash_sensitive_data(cls, data: str, algorithm: str = 'sha256') -> str:
        """对敏感数据进行哈希处理"""
        if not isinstance(data, str):
            return str(data)
        
        hash_func = getattr(hashlib, algorithm, hashlib.sha256)
        return hash_func(data.encode()).hexdigest()[:16]  # 只取前16位

class SecurityValidator:
    """安全验证器"""
    
    @staticmethod
    def validate_api_key_format(api_key: str, vendor: str = 'deepseek') -> bool:
        """验证API密钥格式"""
        if not api_key:
            return False
        
        patterns = {
            'deepseek': r'^[a-zA-Z0-9]{32,}$',
            'doubao': r'^[a-zA-Z0-9]{32,}$',
            'ernie': r'^[a-zA-Z0-9]{32,}$'
        }
        
        pattern = patterns.get(vendor, r'^[a-zA-Z0-9]{16,}$')
        return bool(re.match(pattern, api_key))
    
    @staticmethod
    def check_for_secrets_in_logs(log_content: str) -> List[str]:
        """检查日志中是否包含敏感信息"""
        secrets_found = []
        
        # 检查API密钥
        api_key_patterns = [
            r'[a-zA-Z0-9]{32,64}'
        ]
        
        for pattern in api_key_patterns:
            matches = re.findall(pattern, log_content)
            if matches:
                secrets_found.extend([f"API Key: {match[:8]}..." for match in matches])
        
        return secrets_found
    
    @staticmethod
    def validate_input_safety(user_input: str) -> Dict[str, Any]:
        """验证用户输入安全性"""
        issues = []
        
        # 检查SQL注入
        sql_patterns = [
            r"(?i)(union|select|insert|update|delete|drop|create|alter)\s",
            r"(?i)(or|and)\s+\d+\s*=\s*\d+",
            r"(?i)\';\s*(drop|delete|update)"
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, user_input):
                issues.append("Potential SQL injection detected")
                break
        
        # 检查XSS
        xss_patterns = [
            r"<script[^>]*>.*?</script>",
            r"javascript:\s*",
            r"on\w+\s*=\s*["'][^"']*["']"
        ]
        
        for pattern in xss_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                issues.append("Potential XSS detected")
                break
        
        # 检查过长输入
        if len(user_input) > 10000:
            issues.append("Input too long")
        
        return {
            'is_safe': len(issues) == 0,
            'issues': issues,
            'sanitized_input': DataSanitizer.sanitize_text(user_input)
        }
```

## 7. CI/CD集成方案

### 7.1 GitHub Actions工作流 (.github/workflows/test.yml)

```yaml
name: HarborAI Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # 每天凌晨2点运行完整测试
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.9'
  NODE_VERSION: '18'

jobs:
  # 代码质量检查
  code-quality:
    runs-on: windows-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install black isort flake8 mypy
        pip install -r requirements-test.txt
    
    - name: Code formatting check
      run: |
        black --check tests/
        isort --check-only tests/
    
    - name: Linting
      run: flake8 tests/
    
    - name: Type checking
      run: mypy tests/ --ignore-missing-imports

  # 单元测试
  unit-tests:
    runs-on: windows-latest
    needs: code-quality
    
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        pip install -e .
    
    - name: Run unit tests
      run: |
        pytest tests/functional/ -v \
          --cov=harborai \
          --cov-report=xml \
          --cov-report=html \
          --html=reports/html/unit-tests.html \
          --self-contained-html \
          -m "unit and not slow"
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  # 集成测试
  integration-tests:
    runs-on: windows-latest
    needs: unit-tests
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_USER: test_user
          POSTGRES_DB: harborai_test
        options: >
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        pip install -e .
    
    - name: Set up test database
      run: |
        $env:DATABASE_URL="postgresql://test_user:test_password@localhost:5432/harborai_test"
        python tests/scripts/setup_test_db.py
    
    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://test_user:test_password@localhost:5432/harborai_test
        HARBORAI_TEST_MODE: true
      run: |
        pytest tests/integration/ -v \
          --html=reports/html/integration-tests.html \
          --self-contained-html \
          -m "integration"

  # 性能测试
  performance-tests:
    runs-on: windows-latest
    needs: integration-tests
    if: github.event_name == 'schedule' || contains(github.event.head_commit.message, '[perf-test]')
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        pip install -e .
    
    - name: Run performance tests
      env:
        DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
        ERNIE_API_KEY: ${{ secrets.ERNIE_API_KEY }}
        DOUBAO_API_KEY: ${{ secrets.DOUBAO_API_KEY }}
        PERFORMANCE_THRESHOLD: 1.0
      run: |
        pytest tests/performance/ -v \
          --benchmark-only \
          --benchmark-json=reports/benchmark.json \
          --html=reports/html/performance-tests.html \
          --self-contained-html \
          -m "performance and p0"
    
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: reports/

  # 安全测试
  security-tests:
    runs-on: windows-latest
    needs: unit-tests
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        pip install bandit safety
        pip install -e .
    
    - name: Run security tests
      run: |
        pytest tests/security/ -v \
          --html=reports/html/security-tests.html \
          --self-contained-html \
          -m "security"
    
    - name: Run Bandit security scan
      run: bandit -r harborai/ -f json -o reports/bandit-report.json
    
    - name: Check dependencies for vulnerabilities
      run: safety check --json --output reports/safety-report.json

  # 生成测试报告
  generate-reports:
    runs-on: windows-latest
    needs: [unit-tests, integration-tests, security-tests]
    if: always()
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Download all artifacts
      uses: actions/download-artifact@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Generate consolidated report
      run: |
        python tests/scripts/generate_consolidated_report.py
    
    - name: Upload test reports
      uses: actions/upload-artifact@v3
      with:
        name: test-reports
        path: reports/
        retention-days: 30
    
    - name: Comment PR with test results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const path = 'reports/summary.md';
          if (fs.existsSync(path)) {
            const summary = fs.readFileSync(path, 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: summary
            });
           }
```

### 7.2 PowerShell自动化脚本

#### 7.2.1 测试环境设置脚本 (scripts/setup\_test\_env.ps1)

```powershell
# HarborAI测试环境设置脚本
# 适用于Windows 11 + PowerShell

param(
    [switch]$SkipDocker,
    [switch]$SkipDatabase,
    [string]$PythonVersion = "3.9"
)

# 设置错误处理
$ErrorActionPreference = "Stop"

Write-Host "=== HarborAI测试环境设置 ===" -ForegroundColor Green

# 检查Python版本
Write-Host "检查Python环境..." -ForegroundColor Yellow
try {
    $pythonCmd = Get-Command python -ErrorAction Stop
    $currentVersion = python --version 2>&1
    Write-Host "当前Python版本: $currentVersion" -ForegroundColor Cyan
} catch {
    Write-Error "Python未安装或不在PATH中，请先安装Python $PythonVersion"
    exit 1
}

# 创建虚拟环境
Write-Host "创建Python虚拟环境..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "虚拟环境已存在，跳过创建" -ForegroundColor Cyan
} else {
    python -m venv venv
    Write-Host "虚拟环境创建完成" -ForegroundColor Green
}

# 激活虚拟环境
Write-Host "激活虚拟环境..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# 升级pip
Write-Host "升级pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# 安装测试依赖
Write-Host "安装测试依赖..." -ForegroundColor Yellow
if (Test-Path "tests\requirements-test.txt") {
    pip install -r tests\requirements-test.txt
} else {
    Write-Warning "tests\requirements-test.txt不存在，跳过依赖安装"
}

# 安装项目
Write-Host "安装HarborAI项目..." -ForegroundColor Yellow
pip install -e .

# 设置环境变量
Write-Host "设置测试环境变量..." -ForegroundColor Yellow
if (Test-Path "tests\.env.test") {
    Get-Content "tests\.env.test" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            $name = $matches[1]
            $value = $matches[2]
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
            Write-Host "设置环境变量: $name" -ForegroundColor Cyan
        }
    }
}

# 启动Docker服务（可选）
if (-not $SkipDocker) {
    Write-Host "检查Docker环境..." -ForegroundColor Yellow
    try {
        $dockerCmd = Get-Command docker -ErrorAction Stop
        $dockerVersion = docker --version
        Write-Host "Docker版本: $dockerVersion" -ForegroundColor Cyan
        
        # 启动测试服务
        if (Test-Path "tests\docker-compose.yml") {
            Write-Host "启动Docker测试服务..." -ForegroundColor Yellow
            Set-Location tests
            docker-compose up -d
            Set-Location ..
            
            # 等待服务启动
            Write-Host "等待服务启动..." -ForegroundColor Yellow
            Start-Sleep -Seconds 10
            
            # 检查服务状态
            Set-Location tests
            docker-compose ps
            Set-Location ..
        }
    } catch {
        Write-Warning "Docker未安装或未启动，跳过Docker服务"
    }
}

# 初始化数据库（可选）
if (-not $SkipDatabase) {
    Write-Host "初始化测试数据库..." -ForegroundColor Yellow
    if (Test-Path "tests\scripts\setup_test_db.py") {
        python tests\scripts\setup_test_db.py
    } else {
        Write-Warning "数据库初始化脚本不存在，跳过数据库设置"
    }
}

# 创建报告目录
Write-Host "创建报告目录..." -ForegroundColor Yellow
$reportDirs = @(
    "tests\reports\html",
    "tests\reports\allure",
    "tests\reports\performance",
    "tests\reports\coverage"
)

foreach ($dir in $reportDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "创建目录: $dir" -ForegroundColor Cyan
    }
}

# 验证安装
Write-Host "验证测试环境..." -ForegroundColor Yellow
try {
    python -c "import pytest; import harborai; print('环境验证成功')"
    Write-Host "测试环境设置完成！" -ForegroundColor Green
} catch {
    Write-Error "环境验证失败，请检查安装"
    exit 1
}

Write-Host "\n=== 环境设置完成 ===" -ForegroundColor Green
Write-Host "使用以下命令运行测试:" -ForegroundColor Cyan
Write-Host "  .\tests\scripts\run_all_tests.ps1" -ForegroundColor White
Write-Host "  .\tests\scripts\run_performance_tests.ps1" -ForegroundColor White
```

#### 7.2.2 运行所有测试脚本 (scripts/run\_all\_tests.ps1)

```powershell
# 运行所有测试的PowerShell脚本

param(
    [string]$TestType = "all",  # all, unit, integration, performance, security
    [string]$Priority = "all",  # all, p0, p1, p2, p3
    [switch]$Verbose,
    [switch]$Coverage,
    [switch]$GenerateReport
)

$ErrorActionPreference = "Stop"

Write-Host "=== HarborAI测试执行 ===" -ForegroundColor Green
Write-Host "测试类型: $TestType" -ForegroundColor Cyan
Write-Host "优先级: $Priority" -ForegroundColor Cyan

# 激活虚拟环境
if (Test-Path "venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
    Write-Host "虚拟环境已激活" -ForegroundColor Green
}

# 设置测试参数
$pytestArgs = @()

# 添加详细输出
if ($Verbose) {
    $pytestArgs += "-v"
}

# 添加覆盖率
if ($Coverage) {
    $pytestArgs += "--cov=harborai"
    $pytestArgs += "--cov-report=html:tests/reports/coverage/html"
    $pytestArgs += "--cov-report=xml:tests/reports/coverage/coverage.xml"
}

# 添加HTML报告
if ($GenerateReport) {
    $pytestArgs += "--html=tests/reports/html/test-report.html"
    $pytestArgs += "--self-contained-html"
}

# 根据测试类型设置路径和标记
switch ($TestType.ToLower()) {
    "unit" {
        $pytestArgs += "tests/functional/"
        $pytestArgs += "-m", "unit"
    }
    "integration" {
        $pytestArgs += "tests/integration/"
        $pytestArgs += "-m", "integration"
    }
    "performance" {
        $pytestArgs += "tests/performance/"
        $pytestArgs += "-m", "performance"
    }
    "security" {
        $pytestArgs += "tests/security/"
        $pytestArgs += "-m", "security"
    }
    "all" {
        $pytestArgs += "tests/"
    }
    default {
        Write-Error "未知的测试类型: $TestType"
        exit 1
    }
}

# 根据优先级设置标记
if ($Priority -ne "all") {
    if ($pytestArgs -contains "-m") {
        $markIndex = $pytestArgs.IndexOf("-m") + 1
        $pytestArgs[$markIndex] += " and $Priority"
    } else {
        $pytestArgs += "-m", $Priority
    }
}

# 执行测试
Write-Host "执行测试命令: pytest $($pytestArgs -join ' ')" -ForegroundColor Yellow

try {
    $startTime = Get-Date
    
    # 运行pytest
    & pytest @pytestArgs
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    
    Write-Host "\n=== 测试完成 ===" -ForegroundColor Green
    Write-Host "执行时间: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
    
    # 显示报告位置
    if ($GenerateReport) {
        $reportPath = Resolve-Path "tests/reports/html/test-report.html"
        Write-Host "HTML报告: $reportPath" -ForegroundColor Cyan
    }
    
    if ($Coverage) {
        $coveragePath = Resolve-Path "tests/reports/coverage/html/index.html"
        Write-Host "覆盖率报告: $coveragePath" -ForegroundColor Cyan
    }
    
} catch {
    Write-Error "测试执行失败: $($_.Exception.Message)"
    exit 1
}
```

#### 7.2.3 性能测试脚本 (scripts/run\_performance\_tests.ps1)

```powershell
# 性能测试专用脚本

param(
    [int]$MaxConcurrent = 100,
    [int]$Iterations = 1000,
    [string]$Vendor = "deepseek",
    [switch]$Benchmark,
    [switch]$LoadTest
)

$ErrorActionPreference = "Stop"

Write-Host "=== HarborAI性能测试 ===" -ForegroundColor Green

# 激活虚拟环境
if (Test-Path "venv\Scripts\Activate.ps1") {
    & ".\venv\Scripts\Activate.ps1"
}

# 设置性能测试环境变量
[Environment]::SetEnvironmentVariable("CONCURRENT_LIMIT", $MaxConcurrent, "Process")
[Environment]::SetEnvironmentVariable("PERFORMANCE_ITERATIONS", $Iterations, "Process")
[Environment]::SetEnvironmentVariable("TEST_VENDOR", $Vendor, "Process")

# 检查API凭证
$apiKeyVar = "$($Vendor.ToUpper())_API_KEY"
$apiKey = [Environment]::GetEnvironmentVariable($apiKeyVar)
if (-not $apiKey) {
    Write-Warning "缺少$apiKeyVar环境变量，某些测试可能会跳过"
    Write-Host "支持的厂商: deepseek, ernie, doubao" -ForegroundColor Cyan
}

# 运行基准测试
if ($Benchmark) {
    Write-Host "运行基准测试..." -ForegroundColor Yellow
    pytest tests/performance/benchmarks/ -v `
        --benchmark-only `
        --benchmark-json=tests/reports/performance/benchmark.json `
        --html=tests/reports/html/benchmark-report.html `
        --self-contained-html
}

# 运行负载测试
if ($LoadTest) {
    Write-Host "运行负载测试..." -ForegroundColor Yellow
    
    # 检查Locust是否安装
    try {
        $locustCmd = Get-Command locust -ErrorAction Stop
        
        # 启动Locust负载测试
        Set-Location tests/performance/load_tests
        
        Write-Host "启动Locust Web界面..." -ForegroundColor Cyan
        Write-Host "访问 http://localhost:8089 查看负载测试界面" -ForegroundColor Yellow
        
        locust -f locustfile.py --host=http://localhost:8000
        
        Set-Location ..\..\..\n    } catch {
        Write-Warning "Locust未安装，跳过负载测试"
    }
}

# 运行标准性能测试
Write-Host "运行性能测试套件..." -ForegroundColor Yellow
pytest tests/performance/ -v `
    --html=tests/reports/html/performance-report.html `
    --self-contained-html `
    -m "performance and p0"

Write-Host "\n=== 性能测试完成 ===" -ForegroundColor Green

# 显示报告
$performanceReport = "tests/reports/html/performance-report.html"
if (Test-Path $performanceReport) {
    $reportPath = Resolve-Path $performanceReport
    Write-Host "性能测试报告: $reportPath" -ForegroundColor Cyan
}

$benchmarkReport = "tests/reports/performance/benchmark.json"
if (Test-Path $benchmarkReport) {
    Write-Host "基准测试数据: $(Resolve-Path $benchmarkReport)" -ForegroundColor Cyan
}
```

## 8. 测试报告与监控方案

### 8.1 测试报告生成

#### 8.1.1 报告生成工具 (utils/report\_utils.py)

```python
"""
测试报告生成工具
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import jinja2
from dataclasses import dataclass

@dataclass
class TestSummary:
    """测试摘要数据类"""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    execution_time: float = 0.0
    coverage_percent: float = 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / self.total_tests) * 100

class ReportGenerator:
    """测试报告生成器"""
    
    def __init__(self, template_dir: str = "tests/templates"):
        self.template_dir = Path(template_dir)
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    def generate_summary_report(self, test_results: Dict[str, Any], output_path: str):
        """生成测试摘要报告"""
        summary = self._extract_summary(test_results)
        
        # 生成Markdown报告
        markdown_content = self._generate_markdown_summary(summary, test_results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
    
    def generate_performance_report(self, performance_data: Dict[str, Any], output_path: str):
        """生成性能测试报告"""
        template = self.env.get_template('performance_report.html')
        
        html_content = template.render(
            performance_data=performance_data,
            generated_at=datetime.now().isoformat(),
            title="HarborAI性能测试报告"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def generate_security_report(self, security_results: Dict[str, Any], output_path: str):
        """生成安全测试报告"""
        template = self.env.get_template('security_report.html')
        
        html_content = template.render(
            security_results=security_results,
            generated_at=datetime.now().isoformat(),
            title="HarborAI安全测试报告"
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _extract_summary(self, test_results: Dict[str, Any]) -> TestSummary:
        """提取测试摘要"""
        summary = TestSummary()
        
        # 从pytest结果中提取数据
        if 'summary' in test_results:
            pytest_summary = test_results['summary']
            summary.total_tests = pytest_summary.get('total', 0)
            summary.passed_tests = pytest_summary.get('passed', 0)
            summary.failed_tests = pytest_summary.get('failed', 0)
            summary.skipped_tests = pytest_summary.get('skipped', 0)
            summary.error_tests = pytest_summary.get('error', 0)
        
        # 从覆盖率报告中提取数据
        if 'coverage' in test_results:
            summary.coverage_percent = test_results['coverage'].get('percent', 0.0)
        
        return summary
    
    def _generate_markdown_summary(self, summary: TestSummary, test_results: Dict[str, Any]) -> str:
        """生成Markdown格式的摘要"""
        markdown = f"""
# HarborAI测试报告摘要

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 测试概览

| 指标 | 数值 |
|------|------|
| 总测试数 | {summary.total_tests} |
| 通过 | {summary.passed_tests} |
| 失败 | {summary.failed_tests} |
| 跳过 | {summary.skipped_tests} |
| 错误 | {summary.error_tests} |
| 成功率 | {summary.success_rate:.2f}% |
| 代码覆盖率 | {summary.coverage_percent:.2f}% |

## 测试状态

{'✅ 所有测试通过' if summary.failed_tests == 0 and summary.error_tests == 0 else '❌ 存在失败的测试'}

## 详细报告

- [HTML测试报告](html/test-report.html)
- [覆盖率报告](coverage/html/index.html)
- [性能测试报告](html/performance-report.html)
"""
        
        # 添加失败测试详情
        if summary.failed_tests > 0 or summary.error_tests > 0:
            markdown += "\n## 失败测试详情\n\n"
            if 'failures' in test_results:
                for failure in test_results['failures']:
                    markdown += f"- **{failure['test_name']}**: {failure['error_message']}\n"
        
        return markdown

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, output_dir: str = "tests/reports/metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def collect_test_metrics(self, test_session_id: str) -> Dict[str, Any]:
        """收集测试指标"""
        metrics = {
            'session_id': test_session_id,
            'timestamp': datetime.now().isoformat(),
            'environment': self._get_environment_info(),
            'test_results': self._collect_test_results(),
            'performance_metrics': self._collect_performance_metrics(),
            'coverage_metrics': self._collect_coverage_metrics()
        }
        
        # 保存指标数据
        metrics_file = self.output_dir / f"metrics_{test_session_id}.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        return metrics
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """获取环境信息"""
        import platform
        import sys
        
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0],
            'processor': platform.processor()
        }
    
    def _collect_test_results(self) -> Dict[str, Any]:
        """收集测试结果"""
        # 从pytest缓存或报告文件中读取结果
        results_file = Path("tests/reports/.pytest_cache/v/cache/lastfailed")
        if results_file.exists():
            with open(results_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """收集性能指标"""
        benchmark_file = Path("tests/reports/performance/benchmark.json")
        if benchmark_file.exists():
            with open(benchmark_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _collect_coverage_metrics(self) -> Dict[str, Any]:
        """收集覆盖率指标"""
        coverage_file = Path("tests/reports/coverage/coverage.xml")
        if coverage_file.exists():
            # 解析XML覆盖率报告
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            coverage_data = {
                'line_rate': float(root.get('line-rate', 0)) * 100,
                'branch_rate': float(root.get('branch-rate', 0)) * 100,
                'lines_covered': int(root.get('lines-covered', 0)),
                'lines_valid': int(root.get('lines-valid', 0))
            }
            
            return coverage_data
        
        return {}
```

### 8.2 监控集成

#### 8.2.1 Prometheus指标导出 (utils/prometheus\_exporter.py)

```python
"""
Prometheus指标导出器
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, Any
import time
import threading

class TestMetricsExporter:
    """测试指标导出器"""
    
    def __init__(self, port: int = 8000):
        self.port = port
        
        # 定义指标
        self.test_counter = Counter(
            'harborai_tests_total',
            'Total number of tests executed',
            ['test_type', 'status', 'priority']
        )
        
        self.test_duration = Histogram(
            'harborai_test_duration_seconds',
            'Test execution duration in seconds',
            ['test_type', 'test_name']
        )
        
        self.api_request_duration = Histogram(
            'harborai_api_request_duration_seconds',
            'API request duration in seconds',
            ['vendor', 'model', 'request_type']
        )
        
        self.api_request_counter = Counter(
            'harborai_api_requests_total',
            'Total number of API requests',
            ['vendor', 'model', 'status']
        )
        
        self.concurrent_requests = Gauge(
            'harborai_concurrent_requests',
            'Current number of concurrent requests'
        )
        
        self.memory_usage = Gauge(
            'harborai_memory_usage_bytes',
            'Memory usage in bytes'
        )
        
        self.cpu_usage = Gauge(
            'harborai_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        # 启动HTTP服务器
        self._start_server()
    
    def _start_server(self):
        """启动Prometheus HTTP服务器"""
        def start_server():
            start_http_server(self.port)
        
        server_thread = threading.Thread(target=start_server, daemon=True)
        server_thread.start()
    
    def record_test_execution(self, test_type: str, test_name: str, status: str, 
                            duration: float, priority: str = 'p1'):
        """记录测试执行指标"""
        self.test_counter.labels(
            test_type=test_type,
            status=status,
            priority=priority
        ).inc()
        
        self.test_duration.labels(
            test_type=test_type,
            test_name=test_name
        ).observe(duration)
    
    def record_api_request(self, vendor: str, model: str, request_type: str,
                          duration: float, status: str):
        """记录API请求指标"""
        self.api_request_counter.labels(
            vendor=vendor,
            model=model,
            status=status
        ).inc()
        
        self.api_request_duration.labels(
            vendor=vendor,
            model=model,
            request_type=request_type
        ).observe(duration)
    
    def update_system_metrics(self, memory_bytes: int, cpu_percent: float,
                            concurrent_count: int):
        """更新系统指标"""
        self.memory_usage.set(memory_bytes)
        self.cpu_usage.set(cpu_percent)
        self.concurrent_requests.set(concurrent_count)

# 全局指标导出器实例
metrics_exporter = None

def get_metrics_exporter() -> TestMetricsExporter:
    """获取指标导出器实例"""
    global metrics_exporter
    if metrics_exporter is None:
        port = int(os.getenv('PROMETHEUS_PORT', 8000))
        metrics_exporter = TestMetricsExporter(port)
    return metrics_exporter
```

## 8. 具体测试用例实现

### 8.1 标准格式对齐测试用例 (tests/test\_standard\_alignment.py)

```python
import pytest
import json
from typing import Dict, Any, List
from harborai import HarborAI
from tests.utils.test_helpers import validate_standard_response_format, validate_standard_chunk_format

@pytest.mark.standard_alignment
@pytest.mark.mock
class TestStandardAlignment:
    """标准API格式对齐测试"""
    
    def test_chat_completion_response_format(self, mock_harborai_client):
        """测试ChatCompletion响应格式对齐"""
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # 验证响应格式完全符合标准
        assert validate_standard_response_format(response)
        
        # 验证必需字段
        assert hasattr(response, 'id')
        assert hasattr(response, 'object')
        assert hasattr(response, 'created')
        assert hasattr(response, 'model')
        assert hasattr(response, 'choices')
        assert hasattr(response, 'usage')
        
        # 验证choices结构
        assert len(response.choices) > 0
        choice = response.choices[0]
        assert hasattr(choice, 'index')
        assert hasattr(choice, 'message')
        assert hasattr(choice, 'finish_reason')
        
        # 验证message结构
        message = choice.message
        assert hasattr(message, 'role')
        assert hasattr(message, 'content')
        assert message.role in ['assistant', 'user', 'system']
        
        # 验证usage结构
        usage = response.usage
        assert hasattr(usage, 'prompt_tokens')
        assert hasattr(usage, 'completion_tokens')
        assert hasattr(usage, 'total_tokens')
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    
    def test_streaming_chunk_format(self, mock_harborai_client):
        """测试流式响应Chunk格式对齐"""
        stream = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True
        )
        
        chunks = list(stream)
        assert len(chunks) > 0
        
        for chunk in chunks:
            # 验证每个chunk格式符合标准
            assert validate_standard_chunk_format(chunk)
            
            # 验证基础字段
            assert hasattr(chunk, 'id')
            assert hasattr(chunk, 'object')
            assert chunk.object == 'chat.completion.chunk'
            assert hasattr(chunk, 'created')
            assert hasattr(chunk, 'model')
            assert hasattr(chunk, 'choices')
            
            # 验证choices结构
            if chunk.choices:
                choice = chunk.choices[0]
                assert hasattr(choice, 'index')
                assert hasattr(choice, 'delta')
                assert hasattr(choice, 'finish_reason')
    
    def test_finish_reason_values(self, mock_harborai_client):
        """测试finish_reason值的标准化"""
        # 测试正常完成
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert response.choices[0].finish_reason in ['stop', 'length', 'content_filter', 'tool_calls']
        
        # 测试长度限制
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Very long prompt..."}],
            max_tokens=1
        )
        assert response.choices[0].finish_reason == 'length'
    
    def test_error_code_mapping(self, mock_harborai_client):
        """测试错误码到异常的映射"""
        from harborai.exceptions import AuthenticationError, RateLimitError, APIError
        
        # 测试认证错误
        with pytest.raises(AuthenticationError):
            mock_harborai_client._simulate_error('authentication_error')
        
        # 测试限流错误
        with pytest.raises(RateLimitError):
            mock_harborai_client._simulate_error('rate_limit_error')
        
        # 测试API错误
        with pytest.raises(APIError):
            mock_harborai_client._simulate_error('api_error')
    
    def test_response_format_strict_mode(self, mock_harborai_client):
        """测试response_format严格模式"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        }
        
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Generate a person"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": schema,
                    "strict": True
                }
            }
        )
        
        # 验证返回的JSON严格符合schema
        content = response.choices[0].message.content
        parsed_content = json.loads(content)
        assert "name" in parsed_content
        assert "age" in parsed_content
        assert isinstance(parsed_content["name"], str)
        assert isinstance(parsed_content["age"], int)
    
    def test_parsed_response_format(self, mock_harborai_client):
        """测试parsed响应格式"""
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Generate JSON"}],
            response_format={"type": "json_object"}
        )
        
        # 验证parsed字段存在且可解析
        if hasattr(response.choices[0].message, 'parsed'):
            parsed = response.choices[0].message.parsed
            assert parsed is not None
            assert isinstance(parsed, dict)
```

### 8.2 推理模型测试用例 (tests/test\_reasoning\_models.py)

```python
import pytest
import asyncio
from typing import List, Dict, Any
from harborai import HarborAI
from tests.utils.test_helpers import validate_reasoning_content_detection

@pytest.mark.reasoning_content
@pytest.mark.mock
class TestReasoningModels:
    """推理模型测试"""
    
    def test_reasoning_content_detection(self, mock_reasoning_response):
        """测试reasoning_content自动检测"""
        response = mock_reasoning_response
        
        # 验证reasoning_content字段存在
        assert hasattr(response.choices[0].message, 'reasoning_content')
        reasoning_content = response.choices[0].message.reasoning_content
        
        # 验证reasoning_content内容
        assert reasoning_content is not None
        assert len(reasoning_content) > 0
        assert validate_reasoning_content_detection(reasoning_content)
    
    def test_reasoning_stream_delta_assertions(self, mock_reasoning_stream_chunks):
        """测试流式推理模型delta思考/答案片段断言"""
        chunks = mock_reasoning_stream_chunks
        
        reasoning_chunks = []
        content_chunks = []
        
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                # 收集reasoning_content片段
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_chunks.append(delta.reasoning_content)
                
                # 收集content片段
                if hasattr(delta, 'content') and delta.content:
                    content_chunks.append(delta.content)
        
        # 断言思考片段和答案片段都存在
        assert len(reasoning_chunks) > 0, "应该包含思考过程片段"
        assert len(content_chunks) > 0, "应该包含答案片段"
        
        # 验证思考过程的连续性
        full_reasoning = ''.join(reasoning_chunks)
        assert len(full_reasoning) > 0
        
        # 验证答案的连续性
        full_content = ''.join(content_chunks)
        assert len(full_content) > 0
    
    def test_reasoning_model_mock_scenarios(self, mock_harborai_client):
        """测试推理模型Mock场景"""
        # 场景1: 简单推理任务
        response = mock_harborai_client.chat.completions.create(
            model="o1-preview",
            messages=[{"role": "user", "content": "解决这个数学问题: 2+2=?"}]
        )
        
        assert response.choices[0].message.reasoning_content is not None
        assert "2+2" in response.choices[0].message.reasoning_content
        assert "4" in response.choices[0].message.content
        
        # 场景2: 复杂推理任务
        response = mock_harborai_client.chat.completions.create(
            model="o1-preview",
            messages=[{"role": "user", "content": "分析这个逻辑问题的解决方案"}]
        )
        
        reasoning = response.choices[0].message.reasoning_content
        assert len(reasoning) > 100  # 复杂推理应该有更长的思考过程
        assert any(keyword in reasoning for keyword in ["分析", "逻辑", "步骤", "因为", "所以"])
    
    @pytest.mark.asyncio
    async def test_async_reasoning_stream(self, mock_harborai_client):
        """测试异步推理模型流式响应"""
        stream = await mock_harborai_client.chat.completions.create(
            model="o1-preview",
            messages=[{"role": "user", "content": "思考这个问题"}],
            stream=True
        )
        
        reasoning_parts = []
        content_parts = []
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta
                
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_parts.append(delta.reasoning_content)
                
                if hasattr(delta, 'content') and delta.content:
                    content_parts.append(delta.content)
        
        # 验证异步流式推理的完整性
        assert len(reasoning_parts) > 0
        assert len(content_parts) > 0
        
        full_reasoning = ''.join(reasoning_parts)
        full_content = ''.join(content_parts)
        
        assert len(full_reasoning) > 0
        assert len(full_content) > 0
```

### 8.3 原生Schema对比测试 (tests/test\_native\_vs\_agently.py)

```python
import pytest
import json
from typing import Dict, Any
from harborai import HarborAI

@pytest.mark.native_schema
@pytest.mark.agently_schema
@pytest.mark.mock
class TestNativeVsAgentlySchema:
    """原生Schema与Agently对比测试"""
    
    def test_ernie_native_vs_agently_consistency(self, mock_harborai_client):
        """测试ERNIE原生schema与Agently的一致性"""
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["summary", "keywords"]
        }
        
        # 使用原生structured_provider
        native_response = mock_harborai_client.chat.completions.create(
            model="ernie-bot-turbo",
            messages=[{"role": "user", "content": "总结这篇文章"}],
            response_format={"type": "json_schema", "json_schema": {"schema": schema}},
            structured_provider="native"
        )
        
        # 使用默认Agently
        agently_response = mock_harborai_client.chat.completions.create(
            model="ernie-bot-turbo",
            messages=[{"role": "user", "content": "总结这篇文章"}],
            response_format={"type": "json_schema", "json_schema": {"schema": schema}}
        )
        
        # 解析响应内容
        native_content = json.loads(native_response.choices[0].message.content)
        agently_content = json.loads(agently_response.choices[0].message.content)
        
        # 验证结构一致性
        assert set(native_content.keys()) == set(agently_content.keys())
        assert "summary" in native_content
        assert "keywords" in native_content
        assert isinstance(native_content["keywords"], list)
        
        # 验证类型一致性
        assert type(native_content["summary"]) == type(agently_content["summary"])
        assert type(native_content["keywords"]) == type(agently_content["keywords"])
    
    def test_doubao_native_vs_agently_differences(self, mock_harborai_client):
        """测试Doubao原生schema与Agently的差异性"""
        complex_schema = {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        # 原生实现
        native_response = mock_harborai_client.chat.completions.create(
            model="doubao-pro-32k",
            messages=[{"role": "user", "content": "分析这段文本"}],
            response_format={"type": "json_schema", "json_schema": {"schema": complex_schema}},
            structured_provider="native"
        )
        
        # Agently实现
        agently_response = mock_harborai_client.chat.completions.create(
            model="doubao-pro-32k",
            messages=[{"role": "user", "content": "分析这段文本"}],
            response_format={"type": "json_schema", "json_schema": {"schema": complex_schema}}
        )
        
        native_content = json.loads(native_response.choices[0].message.content)
        agently_content = json.loads(agently_response.choices[0].message.content)
        
        # 记录差异性指标
        native_processing_time = getattr(native_response, '_processing_time', 0)
        agently_processing_time = getattr(agently_response, '_processing_time', 0)
        
        # 验证功能完整性（两种方式都应该工作）
        assert "analysis" in native_content
        assert "analysis" in agently_content
        
        # 记录性能差异（用于后续分析）
        performance_diff = abs(native_processing_time - agently_processing_time)
        assert performance_diff < 1000  # 性能差异不应超过1秒
    
    def test_schema_validation_strictness(self, mock_harborai_client):
        """测试schema验证严格性对比"""
        strict_schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer", "minimum": 1, "maximum": 100}
            },
            "required": ["count"],
            "additionalProperties": False
        }
        
        # 测试原生严格模式
        native_response = mock_harborai_client.chat.completions.create(
            model="ernie-bot-turbo",
            messages=[{"role": "user", "content": "给我一个数字"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "schema": strict_schema,
                    "strict": True
                }
            },
            structured_provider="native"
        )
        
        # 测试Agently模式
        agently_response = mock_harborai_client.chat.completions.create(
            model="ernie-bot-turbo",
            messages=[{"role": "user", "content": "给我一个数字"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": strict_schema}
            }
        )
        
        native_content = json.loads(native_response.choices[0].message.content)
        agently_content = json.loads(agently_response.choices[0].message.content)
        
        # 验证严格性
        assert "count" in native_content
        assert "count" in agently_content
        assert 1 <= native_content["count"] <= 100
        assert 1 <= agently_content["count"] <= 100
        
        # 验证额外属性限制
         assert len(native_content.keys()) == 1  # 严格模式不应有额外属性
```

### 8.4 资源监控测试 (tests/test\_resource\_monitoring.py)

```python
import pytest
import psutil
import time
import threading
from typing import Dict, Any
from harborai import HarborAI
from tests.utils.performance_utils import ResourceMetrics, PerformanceMonitor

@pytest.mark.resource_monitoring
@pytest.mark.mock
class TestResourceMonitoring:
    """资源监控测试"""
    
    def test_windows_handles_monitoring(self, mock_harborai_client):
        """测试Windows句柄数监控"""
        initial_handles = psutil.Process().num_handles()
        
        # 执行多次API调用
        for i in range(10):
            response = mock_harborai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"Test {i}"}]
            )
        
        final_handles = psutil.Process().num_handles()
        handle_growth = final_handles - initial_handles
        
        # 断言句柄增长在合理范围内
        assert handle_growth < 50, f"句柄增长过多: {handle_growth}"
    
    def test_memory_growth_assertion(self, mock_harborai_client):
        """测试内存增长断言"""
        import gc
        gc.collect()  # 强制垃圾回收
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 执行大量API调用
        responses = []
        for i in range(100):
            response = mock_harborai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"Large test {i} with more content"}]
            )
            responses.append(response)
        
        gc.collect()  # 再次垃圾回收
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        # 断言内存增长小于10MB
        assert memory_growth < 10, f"内存增长超过阈值: {memory_growth:.2f}MB"
    
    def test_concurrent_resource_monitoring(self, mock_harborai_client):
        """测试并发场景下的资源监控"""
        monitor = PerformanceMonitor()
        
        def worker_thread(thread_id: int):
            """工作线程函数"""
            for i in range(5):
                response = mock_harborai_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": f"Thread {thread_id} request {i}"}]
                )
        
        # 启动多个并发线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 监控资源使用
        max_memory = 0
        max_handles = 0
        
        while any(t.is_alive() for t in threads):
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            current_handles = psutil.Process().num_handles()
            
            max_memory = max(max_memory, current_memory)
            max_handles = max(max_handles, current_handles)
            
            time.sleep(0.1)
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证资源使用在合理范围内
        assert max_memory < 500, f"并发时内存使用过高: {max_memory:.2f}MB"
        assert max_handles < 1000, f"并发时句柄数过多: {max_handles}"
    
    def test_resource_cleanup_after_errors(self, mock_harborai_client):
        """测试错误后的资源清理"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        initial_handles = psutil.Process().num_handles()
        
        # 模拟错误场景
        error_count = 0
        for i in range(20):
            try:
                if i % 3 == 0:  # 每3次触发一次错误
                    mock_harborai_client._simulate_error('rate_limit_error')
                else:
                    response = mock_harborai_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": f"Test {i}"}]
                    )
            except Exception:
                error_count += 1
        
        import gc
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        final_handles = psutil.Process().num_handles()
        
        memory_growth = final_memory - initial_memory
        handle_growth = final_handles - initial_handles
        
        # 验证即使有错误，资源也能正确清理
        assert memory_growth < 5, f"错误后内存未正确清理: {memory_growth:.2f}MB"
        assert handle_growth < 20, f"错误后句柄未正确清理: {handle_growth}"
        assert error_count > 0, "应该有错误发生"
```

### 8.5 降级策略测试 (tests/test\_fallback\_strategy.py)

```python
import pytest
import time
from unittest.mock import patch, MagicMock
from harborai import HarborAI
from harborai.exceptions import (
    TimeoutError, RateLimitError, QuotaExceededError, 
    AuthenticationError, APIError
)

@pytest.mark.fallback_strategy
@pytest.mark.mock
class TestFallbackStrategy:
    """降级策略测试"""
    
    def test_timeout_fallback_trigger(self, mock_harborai_client):
        """测试超时触发降级"""
        fallback_triggered = False
        fallback_model = None
        
        def mock_fallback_handler(original_model, error, fallback_models):
            nonlocal fallback_triggered, fallback_model
            fallback_triggered = True
            fallback_model = fallback_models[0] if fallback_models else None
            return fallback_model
        
        # 配置降级策略
        mock_harborai_client.configure_fallback(
            fallback_models=["deepseek-chat", "ernie-3.5-8k"],
            fallback_handler=mock_fallback_handler
        )
        
        # 模拟超时错误
        with patch.object(mock_harborai_client, '_make_request') as mock_request:
            mock_request.side_effect = TimeoutError("Request timeout")
            
            try:
                response = mock_harborai_client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": "Test"}],
                    timeout=1.0
                )
            except TimeoutError:
                pass
        
        # 验证降级被触发
        assert fallback_triggered, "超时时应该触发降级"
        assert fallback_model == "deepseek-chat", "应该降级到第一个备选模型"
    
    def test_rate_limit_fallback_order(self, mock_harborai_client):
        """测试限流时的降级顺序"""
        fallback_sequence = []
        
        def track_fallback(original_model, error, fallback_models):
            fallback_sequence.append((original_model, fallback_models[0] if fallback_models else None))
            return fallback_models[0] if fallback_models else None
        
        # 配置多级降级
        mock_harborai_client.configure_fallback(
            fallback_models=["deepseek-chat", "ernie-3.5-8k", "doubao-1-5-pro-32k-character-250715"],
            fallback_handler=track_fallback
        )
        
        # 模拟连续限流错误
        with patch.object(mock_harborai_client, '_make_request') as mock_request:
            mock_request.side_effect = [
                RateLimitError("Rate limit exceeded"),
                RateLimitError("Rate limit exceeded"),
                RateLimitError("Rate limit exceeded")
            ]
            
            for i in range(3):
                try:
                    mock_harborai_client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[{"role": "user", "content": f"Test {i}"}]
                    )
                except RateLimitError:
                    pass
        
        # 验证降级顺序
        assert len(fallback_sequence) == 3
        assert fallback_sequence[0] == ("deepseek-reasoner", "deepseek-chat")
        assert fallback_sequence[1] == ("deepseek-reasoner", "deepseek-chat")
        assert fallback_sequence[2] == ("deepseek-reasoner", "deepseek-chat")
    
    def test_quota_exceeded_fallback(self, mock_harborai_client):
        """测试配额不足触发降级"""
        metrics_reported = []
        
        def mock_metrics_reporter(metric_name, value, tags):
            metrics_reported.append((metric_name, value, tags))
        
        mock_harborai_client.set_metrics_reporter(mock_metrics_reporter)
        
        # 模拟配额不足
        with patch.object(mock_harborai_client, '_make_request') as mock_request:
            mock_request.side_effect = QuotaExceededError("Quota exceeded")
            
            try:
                response = mock_harborai_client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": "Test"}]
                )
            except QuotaExceededError:
                pass
        
        # 验证指标上报
        quota_metrics = [m for m in metrics_reported if 'quota' in m[0]]
        assert len(quota_metrics) > 0, "应该上报配额相关指标"
    
    def test_authentication_failure_no_fallback(self, mock_harborai_client):
        """测试鉴权失败不触发降级"""
        fallback_triggered = False
        
        def track_fallback(original_model, error, fallback_models):
            nonlocal fallback_triggered
            fallback_triggered = True
            return fallback_models[0] if fallback_models else None
        
        mock_harborai_client.configure_fallback(
            fallback_models=["deepseek-chat"],
            fallback_handler=track_fallback
        )
        
        # 模拟认证错误
        with patch.object(mock_harborai_client, '_make_request') as mock_request:
            mock_request.side_effect = AuthenticationError("Invalid API key")
            
            with pytest.raises(AuthenticationError):
                mock_harborai_client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": "Test"}]
                )
        
        # 验证鉴权错误不触发降级
        assert not fallback_triggered, "鉴权错误不应该触发降级"
    
    def test_fallback_metrics_reporting(self, mock_harborai_client):
        """测试降级指标上报"""
        reported_metrics = []
        
        def capture_metrics(metric_name, value, tags):
            reported_metrics.append({
                'name': metric_name,
                'value': value,
                'tags': tags
            })
        
        mock_harborai_client.set_metrics_reporter(capture_metrics)
        
        # 配置降级并触发
        with patch.object(mock_harborai_client, '_make_request') as mock_request:
            mock_request.side_effect = [TimeoutError("Timeout"), MagicMock()]  # 第一次超时，第二次成功
            
            response = mock_harborai_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": "Test"}],
                fallback_models=["deepseek-chat"]
            )
        
        # 验证指标上报
        fallback_metrics = [m for m in reported_metrics if 'fallback' in m['name']]
        assert len(fallback_metrics) > 0, "应该上报降级指标"
        
        # 验证指标包含必要信息
        for metric in fallback_metrics:
            assert 'original_model' in metric['tags']
            assert 'fallback_model' in metric['tags']
            assert 'error_type' in metric['tags']
```

### 8.6 成本统计测试 (tests/test\_cost\_tracking.py)

```python
import pytest
import threading
import time
from decimal import Decimal
from typing import Dict, List
from harborai import HarborAI
from harborai.cost_tracker import CostTracker, VendorPricing

@pytest.mark.cost_tracking
@pytest.mark.mock
class TestCostTracking:
    """成本统计测试"""
    
    def test_multi_vendor_pricing_parameters(self, mock_harborai_client):
        """测试多厂商价格参数化"""
        # 配置各厂商价格
        pricing_config = {
            'deepseek': {
                'deepseek-chat': {'input': Decimal('0.0014'), 'output': Decimal('0.0028')},
                'deepseek-reasoner': {'input': Decimal('0.0014'), 'output': Decimal('0.0028')}
            },
            'doubao': {
                'doubao-1-5-pro-32k-character-250715': {'input': Decimal('0.0008'), 'output': Decimal('0.002')},
                'doubao-seed-1-6-250615': {'input': Decimal('0.0008'), 'output': Decimal('0.002')}
            },
            'ernie': {
                'ernie-3.5-8k': {'input': Decimal('0.0008'), 'output': Decimal('0.002')},
                'ernie-4.0-turbo-8k': {'input': Decimal('0.0008'), 'output': Decimal('0.002')},
                'ernie-x1-turbo-32k': {'input': Decimal('0.0008'), 'output': Decimal('0.002')}
            }
        }
        
        cost_tracker = CostTracker(pricing_config)
        
        # 测试不同厂商的成本计算
        test_cases = [
            ('deepseek', 'deepseek-chat', 2000, 1000, Decimal('0.0056')),  # 0.0014*2 + 0.0028*1
            ('doubao', 'doubao-1-5-pro-32k-character-250715', 1500, 800, Decimal('0.0028')),  # 0.0008*1.5 + 0.002*0.8
            ('ernie', 'ernie-3.5-8k', 1000, 500, Decimal('0.0018')),  # 0.0008*1 + 0.002*0.5
        ]
        
        for vendor, model, input_tokens, output_tokens, expected_cost in test_cases:
            cost = cost_tracker.calculate_cost(
                vendor=vendor,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            assert abs(cost - expected_cost) < Decimal('0.0001'), f"{vendor}/{model} 成本计算错误"
    
    def test_total_cost_assertion(self, mock_harborai_client):
        """测试total_cost断言"""
        cost_tracker = CostTracker()
        
        # 执行多次API调用
        total_expected_cost = Decimal('0')
        
        for i in range(10):
            response = mock_harborai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"Test message {i}"}]
            )
            
            # 模拟token使用情况
            input_tokens = 10 + i
            output_tokens = 20 + i
            
            cost = cost_tracker.track_request(
                vendor="deepseek",
                model="deepseek-chat",
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
            total_expected_cost += cost
        
        # 验证总成本
        actual_total_cost = cost_tracker.get_total_cost()
        assert abs(actual_total_cost - total_expected_cost) < Decimal('0.0001')
        assert actual_total_cost > Decimal('0'), "总成本应该大于0"
    
    def test_concurrent_cost_tracking_no_duplication(self, mock_harborai_client):
        """测试并发下不重复统计成本"""
        cost_tracker = CostTracker()
        request_costs = []
        lock = threading.Lock()
        
        def worker_thread(thread_id: int):
            """工作线程"""
            thread_costs = []
            
            for i in range(5):
                response = mock_harborai_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": f"Thread {thread_id} message {i}"}]
                )
                
                # 每个请求的唯一标识
                request_id = f"thread_{thread_id}_request_{i}"
                
                cost = cost_tracker.track_request(
                    vendor="deepseek",
                    model="deepseek-chat",
                    input_tokens=15,
                    output_tokens=25,
                    request_id=request_id
                )
                thread_costs.append((request_id, cost))
            
            with lock:
                request_costs.extend(thread_costs)
        
        # 启动多个并发线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证没有重复统计
        request_ids = [req[0] for req in request_costs]
        assert len(request_ids) == len(set(request_ids)), "不应该有重复的请求ID"
        
        # 验证总成本正确
        expected_total = sum(req[1] for req in request_costs)
        actual_total = cost_tracker.get_total_cost()
        assert abs(actual_total - expected_total) < Decimal('0.0001')
    
    def test_cost_data_desensitization(self, mock_harborai_client):
        """测试成本数据脱敏"""
        cost_tracker = CostTracker(enable_desensitization=True)
        
        # 执行API调用
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[{"role": "user", "content": "Sensitive business data: API key abc123"}]
        )
        
        # 记录成本并获取脱敏数据
        cost_tracker.track_request(
            vendor="deepseek",
            model="deepseek-reasoner",
            input_tokens=50,
            output_tokens=30,
            request_content="Sensitive business data: API key abc123"
        )
        
        # 获取脱敏后的成本记录
        cost_records = cost_tracker.get_desensitized_records()
        
        # 验证敏感信息被脱敏
        for record in cost_records:
            assert "abc123" not in str(record), "API密钥应该被脱敏"
            assert "***" in record.get('content', ''), "敏感内容应该被脱敏"
            assert record['cost'] > 0, "成本信息应该保留"
    
    def test_cost_database_storage_verification(self, mock_harborai_client, mock_database):
        """测试成本数据落库验证"""
        cost_tracker = CostTracker(database=mock_database)
        
        # 执行多次API调用
        for i in range(5):
            response = mock_harborai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"Test {i}"}]
            )
            
            cost_tracker.track_request(
                vendor="deepseek",
                model="deepseek-chat",
                input_tokens=10,
                output_tokens=15,
                request_id=f"req_{i}"
            )
        
        # 强制刷新到数据库
        cost_tracker.flush_to_database()
        
        # 验证数据库中的记录
        db_records = mock_database.query("SELECT * FROM cost_tracking")
        assert len(db_records) == 5, "应该有5条成本记录"
        
        # 验证记录完整性
        for record in db_records:
            assert record['vendor'] == 'deepseek'
            assert record['model'] == 'deepseek-chat'
            assert record['cost'] > 0
            assert record['input_tokens'] == 10
            assert record['output_tokens'] == 15
            assert 'timestamp' in record
    
    def test_cost_aggregation_by_vendor(self, mock_harborai_client):
        """测试按厂商聚合成本统计"""
        cost_tracker = CostTracker()
        
        # 模拟不同厂商的API调用
        vendor_calls = [
            ('deepseek', 'deepseek-reasoner', 100, 50),
            ('deepseek', 'deepseek-chat', 200, 100),
            ('ernie', 'ernie-3.5-8k', 300, 150),
            ('doubao', 'doubao-1-5-pro-32k-character-250715', 150, 75)
        ]
        
        for vendor, model, input_tokens, output_tokens in vendor_calls:
            cost_tracker.track_request(
                vendor=vendor,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )
        
        # 获取按厂商聚合的成本
        vendor_costs = cost_tracker.get_costs_by_vendor()
        
        # 验证聚合结果
        assert 'deepseek' in vendor_costs
        assert 'ernie' in vendor_costs
        assert 'doubao' in vendor_costs
        
        # 验证DeepSeek成本是两个模型的总和
        deepseek_total = vendor_costs['deepseek']
        assert deepseek_total > 0
        
        # 验证总成本等于各厂商成本之和
        total_by_vendor = sum(vendor_costs.values())
        total_overall = cost_tracker.get_total_cost()
        assert abs(total_by_vendor - total_overall) < Decimal('0.0001')
```

## 9. 项目实施计划

### 9.1 实施阶段

#### 阶段一：基础框架搭建（1-2周）

1. **环境配置**

   * 设置Python虚拟环境

   * 安装测试依赖

   * 配置Docker环境

   * 设置数据库

2. **核心框架**

   * 实现pytest配置

   * 创建基础夹具

   * 实现测试工具类

   * 设置CI/CD流水线

#### 阶段二：功能测试实现（2-3周）

1. **P0优先级测试**

   * API兼容性测试（模块A）

   * 同步异步流式测试（模块B）

   * 结构化输出测试（模块C）

2. **P1优先级测试**

   * 推理模型测试（模块D）

   * 插件系统测试（模块E）

   * 异常重试测试（模块F）

#### 阶段三：性能测试实现（1-2周）

1. **基础性能测试**

   * 封装开销测试

   * 并发性能测试

   * 资源监控测试

2. **高级性能测试**

   * 流式性能测试

   * 压力测试

   * 基准测试

#### 阶段四：集成与优化（1周）

1. **集成测试**

   * 端到端测试

   * 多厂商集成测试

   * 安全测试

2. **报告与监控**

   * 测试报告生成

   * 监控集成

   * 文档完善

### 9.2 质量保证

#### 代码质量

* 代码覆盖率 > 90%

* 所有P0测试必须通过

* 性能测试符合基准要求

* 安全测试无高危漏洞

#### 文档质量

* 完整的API文档

* 详细的使用说明

* 故障排除指南

* 最佳实践文档

### 9.3 维护计划

#### 日常维护

* 每日自动化测试执行

* 性能指标监控

* 测试结果分析

* 问题跟踪处理

#### 定期维护

* 每周测试用例审查

* 每月性能基准更新

* 每季度测试策略评估

* 年度测试框架升级

## 10. 总结

本技术设计方案为HarborAI项目提供了一个全面、可靠、高效的测试体系。方案涵盖了：

### 10.1 核心优势

1. **全面覆盖**：覆盖功能测试、性能测试、安全测试、集成测试等各个方面
2. **技术先进**：采用现代化的测试框架和工具，支持异步、并发、流式测试
3. **环境适配**：完全适配Windows 11 + PowerShell环境，提供便捷的自动化脚本
4. **可扩展性**：模块化设计，易于扩展和维护
5. **监控完善**：集成Prometheus监控和丰富的测试报告

### 10.2 技术特色

1. **多厂商支持**：支持DeepSeek、Doubao、Wenxin等多个厂商的API测试
2. **性能基准**：建立完整的性能基准体系，支持性能回归检测
3. **安全保障**：包含数据脱敏、安全验证等安全测试功能
4. **CI/CD集成**：完整的GitHub Actions工作流，支持自动化测试和部署
5. **容器化支持**：Docker化的测试环境，确保环境一致性

### 10.3 预期效果

1. **质量提升**：通过全面的测试覆盖，显著提升HarborAI的代码质量和稳定性
2. **效率提升**：自动化测试流程，减少手工测试工作量，提升开发效率
3. **风险控制**：及时发现和修复问题，降低生产环境风险
4. **性能保障**：持续的性能监控，确保系统性能符合预期
5. **安全合规**：完善的安全测试，确保系统安全合规

通过实施本技术方案，HarborAI项目将建立起一个现代化、专业化的测试体系，为项目的长期发展提供坚实的质量保障。
