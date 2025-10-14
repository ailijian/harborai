# HarborAI 开发指南

本文档为 HarborAI 项目的完整开发指南，包含环境搭建、开发流程、代码规范和最佳实践。

## 📋 目录

- [开发环境搭建](#开发环境搭建)
- [项目结构](#项目结构)
- [开发流程](#开发流程)
- [代码规范](#代码规范)
- [测试指南](#测试指南)
- [调试技巧](#调试技巧)
- [性能优化开发](#性能优化开发)
- [贡献指南](#贡献指南)

## 开发环境搭建

### 🛠️ 系统要求

- **Python**: 3.8+ (推荐 3.11+)
- **操作系统**: Windows 10+, macOS 10.15+, Ubuntu 18.04+
- **内存**: 最低 8GB，推荐 16GB+
- **存储**: 至少 2GB 可用空间

### 📦 环境安装

#### 1. 克隆项目

```bash
git clone https://github.com/your-org/harborai.git
cd harborai
```

#### 2. 创建虚拟环境

```bash
# 使用 venv
python -m venv venv

# Windows 激活
venv\Scripts\activate

# macOS/Linux 激活
source venv/bin/activate
```

#### 3. 安装依赖

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 或者使用 requirements
pip install -r requirements-dev.txt
```

#### 4. 配置开发工具

```bash
# 安装 pre-commit 钩子
pre-commit install

# 配置 Git 钩子
git config core.hooksPath .githooks
```

### 🔧 IDE 配置

#### VS Code 配置

创建 `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=88"],
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ]
}
```

#### PyCharm 配置

1. 设置 Python 解释器为虚拟环境
2. 配置代码格式化工具为 Black
3. 启用 MyPy 类型检查
4. 配置测试运行器为 pytest

## 项目结构

### 📁 目录结构

```
harborai/
├── harborai/                 # 主要源代码
│   ├── __init__.py          # 包初始化
│   ├── client.py            # 主客户端
│   ├── api/                 # API 相关模块
│   │   ├── __init__.py
│   │   ├── base.py          # 基础 API 类
│   │   ├── chat.py          # 聊天 API
│   │   ├── fast_client.py   # 快速客户端
│   │   └── streaming.py     # 流式处理
│   ├── core/                # 核心功能
│   │   ├── __init__.py
│   │   ├── config.py        # 配置管理
│   │   ├── cache.py         # 缓存系统
│   │   ├── memory.py        # 内存管理
│   │   └── monitoring.py    # 监控系统
│   ├── utils/               # 工具函数
│   │   ├── __init__.py
│   │   ├── helpers.py       # 辅助函数
│   │   ├── validators.py    # 验证器
│   │   └── exceptions.py    # 异常定义
│   └── plugins/             # 插件系统
│       ├── __init__.py
│       ├── base.py          # 插件基类
│       └── performance.py   # 性能插件
├── tests/                   # 测试代码
│   ├── unit/                # 单元测试
│   ├── integration/         # 集成测试
│   ├── performance/         # 性能测试
│   └── fixtures/            # 测试数据
├── docs/                    # 文档
├── examples/                # 示例代码
├── scripts/                 # 构建脚本
├── .github/                 # GitHub 配置
├── requirements.txt         # 生产依赖
├── requirements-dev.txt     # 开发依赖
├── setup.py                 # 包配置
├── pyproject.toml          # 项目配置
└── README.md               # 项目说明
```

### 🏗️ 架构设计原则

#### 1. 模块化设计

```python
"""
模块化设计示例
每个模块都有明确的职责和接口
"""

# harborai/api/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseAPI(ABC):
    """API 基类，定义通用接口"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
    
    @abstractmethod
    async def call(self, **kwargs) -> Any:
        """抽象方法：API 调用"""
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """抽象方法：参数验证"""
        pass
```

#### 2. 依赖注入

```python
"""
依赖注入容器
"""
from typing import Dict, Type, Any, Callable

class DIContainer:
    """简单的依赖注入容器"""
    
    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable] = {}
    
    def register(self, name: str, service: Any):
        """注册服务实例"""
        self._services[name] = service
    
    def register_factory(self, name: str, factory: Callable):
        """注册服务工厂"""
        self._factories[name] = factory
    
    def get(self, name: str) -> Any:
        """获取服务"""
        if name in self._services:
            return self._services[name]
        
        if name in self._factories:
            service = self._factories[name]()
            self._services[name] = service
            return service
        
        raise ValueError(f"Service '{name}' not found")

# 使用示例
container = DIContainer()
container.register_factory('cache', lambda: CacheManager())
container.register_factory('metrics', lambda: PerformanceMetrics())
```

#### 3. 插件系统

```python
"""
插件系统设计
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Plugin(ABC):
    """插件基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """插件名称"""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]):
        """初始化插件"""
        pass
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """处理数据"""
        pass

class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self.plugins: List[Plugin] = []
    
    def register(self, plugin: Plugin):
        """注册插件"""
        self.plugins.append(plugin)
    
    async def process_all(self, data: Any) -> Any:
        """通过所有插件处理数据"""
        result = data
        for plugin in self.plugins:
            result = await plugin.process(result)
        return result
```

## 开发流程

### 🔄 Git 工作流

#### 1. 分支策略

```bash
# 主分支
main          # 生产环境代码
develop       # 开发环境代码

# 功能分支
feature/xxx   # 新功能开发
bugfix/xxx    # Bug 修复
hotfix/xxx    # 紧急修复
release/xxx   # 发布准备
```

#### 2. 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```bash
# 格式
<type>(<scope>): <subject>

# 示例
feat(api): 添加结构化输出支持
fix(cache): 修复缓存键冲突问题
docs(readme): 更新安装说明
perf(client): 优化初始化性能
test(unit): 添加缓存管理器测试
refactor(core): 重构配置管理模块
```

#### 3. 开发流程

```bash
# 1. 创建功能分支
git checkout -b feature/new-feature

# 2. 开发和提交
git add .
git commit -m "feat(api): 添加新功能"

# 3. 推送分支
git push origin feature/new-feature

# 4. 创建 Pull Request
# 在 GitHub/GitLab 上创建 PR

# 5. 代码审查和合并
# 通过审查后合并到 develop 分支
```

### 🧪 测试驱动开发 (TDD)

#### 1. TDD 流程

```python
"""
TDD 开发示例：实现缓存功能
"""

# 第一步：编写失败的测试
def test_cache_set_and_get():
    """测试缓存设置和获取"""
    cache = CacheManager()
    
    # 设置缓存
    cache.set("key1", "value1")
    
    # 获取缓存
    result = cache.get("key1")
    
    # 断言
    assert result == "value1"

# 第二步：编写最小实现
class CacheManager:
    def __init__(self):
        self._cache = {}
    
    def set(self, key: str, value: Any):
        self._cache[key] = value
    
    def get(self, key: str) -> Any:
        return self._cache.get(key)

# 第三步：重构优化
class CacheManager:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache = OrderedDict()
    
    def set(self, key: str, value: Any):
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
        
        self._cache[key] = value
    
    def get(self, key: str) -> Any:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None
```

#### 2. 测试分层

```python
"""
测试分层示例
"""

# 单元测试：测试单个函数/类
class TestCacheManager:
    def test_set_get(self):
        cache = CacheManager()
        cache.set("key", "value")
        assert cache.get("key") == "value"
    
    def test_lru_eviction(self):
        cache = CacheManager(max_size=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # 应该淘汰 key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

# 集成测试：测试模块间交互
class TestAPIIntegration:
    async def test_api_with_cache(self):
        cache = CacheManager()
        api = ChatAPI(cache=cache)
        
        # 第一次调用
        result1 = await api.chat_completion(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        # 第二次调用（应该命中缓存）
        result2 = await api.chat_completion(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}]
        )
        
        assert result1 == result2
        assert api.cache_hit_count == 1

# 端到端测试：测试完整流程
class TestE2E:
    async def test_complete_workflow(self):
        client = HarborAI(api_key="test-key")
        
        # 测试完整的聊天流程
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "你好"}
            ]
        )
        
        assert response.choices[0].message.content
        assert response.usage.total_tokens > 0
```

## 代码规范

### 📝 编码标准

#### 1. Python 代码风格

```python
"""
Python 代码风格示例
遵循 PEP 8 和项目特定规范
"""

from typing import Dict, List, Optional, Union, Any
import asyncio
import logging

# 类型注解
def process_messages(
    messages: List[Dict[str, str]], 
    model: str = "deepseek-chat",
    temperature: float = 0.7,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    处理消息列表
    
    Args:
        messages: 消息列表，每个消息包含 role 和 content
        model: 使用的模型名称
        temperature: 温度参数，控制随机性
        max_tokens: 最大令牌数，None 表示不限制
    
    Returns:
        处理结果字典，包含响应和元数据
    
    Raises:
        ValueError: 当消息格式不正确时
        APIError: 当 API 调用失败时
    """
    # 参数验证
    if not messages:
        raise ValueError("消息列表不能为空")
    
    # 处理逻辑
    result = {
        "response": "处理结果",
        "metadata": {
            "model": model,
            "temperature": temperature,
            "token_count": len(str(messages))
        }
    }
    
    return result

# 类定义
class APIClient:
    """API 客户端类"""
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "https://api.deepseek.com",
        timeout: float = 30.0
    ):
        """
        初始化 API 客户端
        
        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._session = None
        
        # 配置日志
        self.logger = logging.getLogger(__name__)
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._initialize_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self._cleanup_session()
    
    async def _initialize_session(self):
        """初始化会话"""
        import httpx
        
        self._session = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
    
    async def _cleanup_session(self):
        """清理会话"""
        if self._session:
            await self._session.aclose()
            self._session = None
```

#### 2. 文档字符串规范

```python
"""
文档字符串规范示例
使用 Google 风格的 docstring
"""

def calculate_performance_score(
    latency: float,
    throughput: float,
    memory_usage: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    计算性能评分
    
    根据延迟、吞吐量和内存使用情况计算综合性能评分。
    评分范围为 0-100，分数越高表示性能越好。
    
    Args:
        latency: 平均延迟时间（毫秒）
        throughput: 吞吐量（请求/秒）
        memory_usage: 内存使用量（MB）
        weights: 各指标权重，默认为 {"latency": 0.4, "throughput": 0.4, "memory": 0.2}
    
    Returns:
        性能评分（0-100）
    
    Raises:
        ValueError: 当输入参数无效时
        
    Example:
        >>> score = calculate_performance_score(
        ...     latency=100.0,
        ...     throughput=50.0,
        ...     memory_usage=200.0
        ... )
        >>> print(f"性能评分: {score:.1f}")
        性能评分: 75.2
        
    Note:
        - 延迟越低评分越高
        - 吞吐量越高评分越高
        - 内存使用越低评分越高
    """
    # 默认权重
    if weights is None:
        weights = {"latency": 0.4, "throughput": 0.4, "memory": 0.2}
    
    # 参数验证
    if latency < 0 or throughput < 0 or memory_usage < 0:
        raise ValueError("所有参数必须为非负数")
    
    # 计算各项评分（0-100）
    latency_score = max(0, 100 - latency / 10)  # 延迟越低分数越高
    throughput_score = min(100, throughput * 2)  # 吞吐量越高分数越高
    memory_score = max(0, 100 - memory_usage / 10)  # 内存越低分数越高
    
    # 加权平均
    total_score = (
        latency_score * weights["latency"] +
        throughput_score * weights["throughput"] +
        memory_score * weights["memory"]
    )
    
    return round(total_score, 2)
```

#### 3. 错误处理

```python
"""
错误处理最佳实践
"""

# 自定义异常
class HarborAIError(Exception):
    """HarborAI 基础异常类"""
    pass

class APIError(HarborAIError):
    """API 调用异常"""
    
    def __init__(self, message: str, status_code: int = None, response: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

class ConfigurationError(HarborAIError):
    """配置错误异常"""
    pass

class ValidationError(HarborAIError):
    """验证错误异常"""
    pass

# 错误处理装饰器
def handle_api_errors(func):
    """API 错误处理装饰器"""
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            raise APIError(
                f"API 请求失败: {e.response.status_code}",
                status_code=e.response.status_code,
                response=e.response.text
            )
        except httpx.TimeoutException:
            raise APIError("API 请求超时")
        except Exception as e:
            raise HarborAIError(f"未知错误: {str(e)}")
    
    return wrapper

# 使用示例
@handle_api_errors
async def make_api_request(url: str, data: dict) -> dict:
    """发起 API 请求"""
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        response.raise_for_status()
        return response.json()
```

## 测试指南

### 🧪 测试框架配置

#### 1. pytest 配置

创建 `pytest.ini`:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=harborai
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: 单元测试
    integration: 集成测试
    e2e: 端到端测试
    slow: 慢速测试
    performance: 性能测试
```

#### 2. 测试配置

创建 `conftest.py`:

```python
"""
pytest 配置和 fixtures
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from harborai import HarborAI
from harborai.api.fast_client import FastHarborAI

@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_api_key():
    """模拟 API 密钥"""
    return "test-api-key-12345"

@pytest.fixture
def harbor_client(mock_api_key):
    """创建 HarborAI 客户端"""
    return HarborAI(api_key=mock_api_key)

@pytest.fixture
def fast_harbor_client(mock_api_key):
    """创建 FastHarborAI 客户端"""
    return FastHarborAI(
        api_key=mock_api_key,
        performance_mode="fast"
    )

@pytest.fixture
def mock_http_response():
    """模拟 HTTP 响应"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "这是一个测试响应"
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    return mock_response

@pytest.fixture
async def async_mock_session():
    """异步模拟会话"""
    session = AsyncMock()
    session.post.return_value.__aenter__.return_value.json.return_value = {
        "choices": [{"message": {"content": "测试响应"}}]
    }
    return session
```

#### 3. 测试示例

```python
"""
测试示例
"""
import pytest
from unittest.mock import patch, AsyncMock
from harborai.core.cache import CacheManager
from harborai.api.chat import ChatAPI

class TestCacheManager:
    """缓存管理器测试"""
    
    def test_cache_initialization(self):
        """测试缓存初始化"""
        cache = CacheManager(max_size=100)
        assert cache.max_size == 100
        assert len(cache._cache) == 0
    
    def test_cache_set_get(self):
        """测试缓存设置和获取"""
        cache = CacheManager()
        
        # 设置缓存
        cache.set("test_key", "test_value")
        
        # 获取缓存
        result = cache.get("test_key")
        assert result == "test_value"
    
    def test_cache_lru_eviction(self):
        """测试 LRU 淘汰策略"""
        cache = CacheManager(max_size=2)
        
        # 添加两个项目
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # 添加第三个项目，应该淘汰第一个
        cache.set("key3", "value3")
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"

class TestChatAPI:
    """聊天 API 测试"""
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, mock_api_key, async_mock_session):
        """测试聊天完成成功场景"""
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value = async_mock_session
            
            api = ChatAPI(api_key=mock_api_key)
            
            response = await api.chat_completion(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            assert response is not None
            assert "choices" in response
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_cache(self, mock_api_key):
        """测试带缓存的聊天完成"""
        cache = CacheManager()
        api = ChatAPI(api_key=mock_api_key, cache=cache)
        
        # 模拟 API 响应
        mock_response = {
            "choices": [{"message": {"content": "缓存测试响应"}}]
        }
        
        with patch.object(api, '_make_request', return_value=mock_response):
            # 第一次调用
            response1 = await api.chat_completion(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "测试"}]
            )
            
            # 第二次调用（应该命中缓存）
            response2 = await api.chat_completion(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "测试"}]
            )
            
            assert response1 == response2
            assert api._make_request.call_count == 1  # 只调用一次 API

@pytest.mark.integration
class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_client_with_cache_integration(self, mock_api_key):
        """测试客户端与缓存集成"""
        client = FastHarborAI(
            api_key=mock_api_key,
            enable_cache=True
        )
        
        # 模拟 API 调用
        with patch.object(client, '_make_api_request') as mock_request:
            mock_request.return_value = {
                "choices": [{"message": {"content": "集成测试响应"}}]
            }
            
            # 执行多次相同请求
            for _ in range(3):
                await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": "集成测试"}]
                )
            
            # 验证只调用了一次 API
            assert mock_request.call_count == 1

@pytest.mark.performance
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self, fast_harbor_client):
        """测试并发请求性能"""
        import time
        import asyncio
        
        # 创建并发任务
        tasks = []
        for i in range(10):
            task = fast_harbor_client.mock_chat_completion(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"测试 {i}"}]
            )
            tasks.append(task)
        
        # 执行并发测试
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # 验证结果
        assert len(results) == 10
        assert all(result is not None for result in results)
        
        # 性能断言
        total_time = end_time - start_time
        assert total_time < 5.0  # 应该在 5 秒内完成
        
        print(f"并发测试完成时间: {total_time:.2f}s")
```

## 调试技巧

### 🐛 调试工具

#### 1. 日志配置

```python
"""
日志配置
"""
import logging
import sys
from pathlib import Path

def setup_logging(
    level: str = "INFO",
    log_file: str = None,
    format_string: str = None
):
    """
    配置日志系统
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，None 表示只输出到控制台
        format_string: 自定义格式字符串
    """
    if format_string is None:
        format_string = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(message)s"
        )
    
    # 配置根日志器
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=[]
    )
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    logging.getLogger().addHandler(console_handler)
    
    # 添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(format_string))
        logging.getLogger().addHandler(file_handler)

# 使用示例
setup_logging(
    level="DEBUG",
    log_file="logs/harborai.log"
)

logger = logging.getLogger(__name__)
logger.info("日志系统已配置")
```

#### 2. 调试装饰器

```python
"""
调试装饰器
"""
import functools
import time
import logging
from typing import Any, Callable

def debug_performance(func: Callable) -> Callable:
    """性能调试装饰器"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(func.__module__)
        
        logger.debug(f"开始执行 {func.__name__}")
        logger.debug(f"参数: args={args}, kwargs={kwargs}")
        
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(f"执行完成 {func.__name__}, 耗时: {execution_time:.3f}s")
            logger.debug(f"返回值类型: {type(result)}")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"执行失败 {func.__name__}, 耗时: {execution_time:.3f}s, 错误: {e}")
            raise
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        logger = logging.getLogger(func.__module__)
        
        logger.debug(f"开始执行 {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(f"执行完成 {func.__name__}, 耗时: {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"执行失败 {func.__name__}, 耗时: {execution_time:.3f}s, 错误: {e}")
            raise
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

# 使用示例
@debug_performance
async def api_call_with_debug(model: str, messages: list):
    """带调试的 API 调用"""
    # 模拟 API 调用
    await asyncio.sleep(0.1)
    return {"response": "调试测试"}
```

#### 3. 内存调试

```python
"""
内存调试工具
"""
import psutil
import gc
import tracemalloc
from typing import Dict, Any

class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self):
        self.snapshots = []
        self.process = psutil.Process()
    
    def start_tracing(self):
        """开始内存追踪"""
        tracemalloc.start()
        self.take_snapshot("start")
    
    def take_snapshot(self, label: str):
        """拍摄内存快照"""
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            self.snapshots.append({
                'label': label,
                'snapshot': snapshot,
                'memory_mb': memory_mb,
                'timestamp': time.time()
            })
    
    def compare_snapshots(self, start_label: str, end_label: str):
        """比较内存快照"""
        start_snap = None
        end_snap = None
        
        for snap in self.snapshots:
            if snap['label'] == start_label:
                start_snap = snap
            elif snap['label'] == end_label:
                end_snap = snap
        
        if not start_snap or not end_snap:
            print("找不到指定的快照")
            return
        
        # 比较快照
        top_stats = end_snap['snapshot'].compare_to(
            start_snap['snapshot'], 'lineno'
        )
        
        print(f"内存变化分析 ({start_label} -> {end_label}):")
        print(f"总内存变化: {end_snap['memory_mb'] - start_snap['memory_mb']:.1f}MB")
        print("\n前10个内存增长最多的位置:")
        
        for stat in top_stats[:10]:
            print(f"  {stat}")
    
    def get_current_memory_usage(self) -> Dict[str, Any]:
        """获取当前内存使用情况"""
        memory_info = self.process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': self.process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }

# 使用示例
profiler = MemoryProfiler()
profiler.start_tracing()

# 执行一些操作
profiler.take_snapshot("before_operation")

# 模拟内存密集操作
data = [i for i in range(100000)]

profiler.take_snapshot("after_operation")
profiler.compare_snapshots("before_operation", "after_operation")
```

## 性能优化开发

### ⚡ 性能优化策略

#### 1. 异步编程最佳实践

```python
"""
异步编程最佳实践
"""
import asyncio
import aiohttp
from typing import List, Dict, Any

class AsyncBatchProcessor:
    """异步批处理器"""
    
    def __init__(self, max_concurrent: int = 10, batch_size: int = 50):
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_batch(self, items: List[Any]) -> List[Any]:
        """批量处理项目"""
        # 分批处理
        batches = [
            items[i:i + self.batch_size] 
            for i in range(0, len(items), self.batch_size)
        ]
        
        # 并发处理批次
        tasks = [
            self._process_single_batch(batch) 
            for batch in batches
        ]
        
        batch_results = await asyncio.gather(*tasks)
        
        # 合并结果
        results = []
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    async def _process_single_batch(self, batch: List[Any]) -> List[Any]:
        """处理单个批次"""
        async with self.semaphore:
            # 并发处理批次内的项目
            tasks = [self._process_item(item) for item in batch]
            return await asyncio.gather(*tasks)
    
    async def _process_item(self, item: Any) -> Any:
        """处理单个项目"""
        # 模拟异步处理
        await asyncio.sleep(0.01)
        return f"processed_{item}"

# 使用示例
async def main():
    processor = AsyncBatchProcessor(max_concurrent=5, batch_size=20)
    items = list(range(1000))
    
    start_time = time.time()
    results = await processor.process_batch(items)
    end_time = time.time()
    
    print(f"处理 {len(items)} 个项目，耗时: {end_time - start_time:.2f}s")
```

#### 2. 缓存优化策略

```python
"""
高级缓存优化策略
"""
import hashlib
import pickle
import time
from typing import Any, Optional, Callable, Dict
from functools import wraps

class SmartCache:
    """智能缓存系统"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 创建可序列化的参数表示
        serializable_args = []
        for arg in args:
            if hasattr(arg, '__dict__'):
                serializable_args.append(str(arg.__dict__))
            else:
                serializable_args.append(str(arg))
        
        key_data = {
            'func': func_name,
            'args': serializable_args,
            'kwargs': sorted(kwargs.items())
        }
        
        key_str = pickle.dumps(key_data)
        return hashlib.md5(key_str).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self.cache:
            self.miss_count += 1
            return None
        
        entry = self.cache[key]
        
        # 检查 TTL
        if time.time() - entry['timestamp'] > entry['ttl']:
            del self.cache[key]
            self.access_times.pop(key, None)
            self.miss_count += 1
            return None
        
        # 更新访问时间
        self.access_times[key] = time.time()
        self.hit_count += 1
        
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: int = None):
        """设置缓存值"""
        if ttl is None:
            ttl = self.default_ttl
        
        # 如果缓存已满，删除最久未访问的项目
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()
        
        self.cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'ttl': ttl
        }
        self.access_times[key] = time.time()
    
    def _evict_lru(self):
        """淘汰最久未使用的项目"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times, key=self.access_times.get)
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }

def cached(ttl: int = 3600, cache_instance: SmartCache = None):
    """缓存装饰器"""
    if cache_instance is None:
        cache_instance = SmartCache()
    
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = cache_instance._generate_key(
                func.__name__, args, kwargs
            )
            
            # 尝试从缓存获取
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 缓存结果
            cache_instance.set(cache_key, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache_key = cache_instance._generate_key(
                func.__name__, args, kwargs
            )
            
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result, ttl)
            
            return result
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# 使用示例
cache = SmartCache(max_size=500, default_ttl=1800)

@cached(ttl=3600, cache_instance=cache)
async def expensive_api_call(model: str, prompt: str) -> dict:
    """昂贵的 API 调用"""
    # 模拟 API 调用
    await asyncio.sleep(1)
    return {
        "response": f"Response for {prompt} using {model}",
        "timestamp": time.time()
    }
```

## 贡献指南

### 🤝 贡献流程

#### 1. 准备工作

```bash
# Fork 项目到你的 GitHub 账户
# 克隆你的 fork
git clone https://github.com/your-username/harborai.git
cd harborai

# 添加上游仓库
git remote add upstream https://github.com/original-org/harborai.git

# 创建开发分支
git checkout -b feature/your-feature-name
```

#### 2. 开发规范

- **代码风格**: 遵循 PEP 8 和项目代码规范
- **测试覆盖**: 新功能必须包含测试，覆盖率不低于 80%
- **文档更新**: 更新相关文档和示例
- **提交信息**: 使用 Conventional Commits 格式

#### 3. 提交检查清单

- [ ] 代码通过所有测试
- [ ] 代码风格检查通过
- [ ] 类型检查通过
- [ ] 文档已更新
- [ ] 示例代码可运行
- [ ] 性能测试通过（如适用）

#### 4. Pull Request 模板

```markdown
## 变更描述
简要描述这个 PR 的目的和变更内容。

## 变更类型
- [ ] Bug 修复
- [ ] 新功能
- [ ] 性能优化
- [ ] 文档更新
- [ ] 重构
- [ ] 其他

## 测试
- [ ] 添加了新的测试
- [ ] 所有测试通过
- [ ] 手动测试通过

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 自我审查了代码
- [ ] 添加了必要的注释
- [ ] 更新了相关文档
- [ ] 没有引入新的警告

## 相关 Issue
关联的 Issue 编号（如果有）

## 截图
如果有 UI 变更，请提供截图

## 其他说明
任何其他需要说明的内容
```

---

**开发指南版本**: v1.0.0 | **更新日期**: 2025-01-25 | **下次更新**: 2025-02-25