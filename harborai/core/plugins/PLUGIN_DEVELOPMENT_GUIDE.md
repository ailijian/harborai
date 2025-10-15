# HarborAI 模型插件开发指南

> **版本**: 1.0  
> **更新时间**: 2024年12月  
> **适用范围**: HarborAI 插件系统  

## 目录

1. [概述](#概述)
2. [插件系统架构](#插件系统架构)
3. [快速开始](#快速开始)
4. [基础插件类详解](#基础插件类详解)
5. [完整插件实现示例](#完整插件实现示例)
6. [高级特性](#高级特性)
7. [测试指南](#测试指南)
8. [最佳实践](#最佳实践)
9. [常见问题解答](#常见问题解答)
10. [附录](#附录)

---

## 概述

HarborAI 插件系统是一个灵活、可扩展的架构，允许开发者轻松集成各种大语言模型（LLM）厂商的API。本指南将帮助您快速理解插件系统的设计理念，并指导您实现自己的模型插件。

### 核心特性

- **统一接口**: 所有插件都遵循相同的接口规范，确保一致性
- **异步支持**: 完整的同步和异步调用支持
- **流式输出**: 支持实时流式响应
- **推理模型**: 支持具有思考过程的推理模型
- **结构化输出**: 支持JSON Schema约束的结构化响应
- **错误处理**: 完善的错误处理和重试机制
- **可观测性**: 内置日志记录和链路追踪

### 支持的模型类型

1. **标准聊天模型**: 如 GPT-4、Claude 等
2. **推理模型**: 如 OpenAI o1、DeepSeek R1 等，支持思考过程输出
3. **结构化输出模型**: 支持按照 JSON Schema 格式化输出

---

## 插件系统架构

### 核心组件

```
harborai/core/plugins/
├── base.py                 # 旧版插件基类（兼容性保留）
├── base_plugin.py          # 新版插件基类
├── manager.py              # 插件管理器
├── hooks.py                # 插件钩子系统
├── http_config.py          # HTTP配置
├── openai_plugin.py        # OpenAI插件实现
├── deepseek_plugin.py      # DeepSeek插件实现
├── wenxin_plugin.py        # 文心一言插件实现
└── doubao_plugin.py        # 豆包插件实现
```

### 类层次结构

```
BaseLLMPlugin (抽象基类)
├── OpenAIPlugin
├── DeepSeekPlugin
├── WenxinPlugin
└── DoubaoPlugin
```

### 数据模型

```python
@dataclass
class ModelInfo:
    """模型信息"""
    id: str                           # 模型ID
    name: str                         # 显示名称
    provider: str                     # 提供商
    supports_streaming: bool = True   # 是否支持流式
    supports_structured_output: bool = False  # 是否支持结构化输出
    supports_thinking: bool = False   # 是否支持推理思考
    max_tokens: Optional[int] = None  # 最大token数
    context_window: Optional[int] = None  # 上下文窗口大小

@dataclass
class ChatMessage:
    """聊天消息"""
    role: str                         # 角色: system/user/assistant
    content: str                      # 消息内容
    name: Optional[str] = None        # 发送者名称
    reasoning_content: Optional[str] = None  # 推理内容
    parsed: Optional[Any] = None      # 结构化输出解析结果
```

---

## 快速开始

### 1. 创建基础插件文件

创建一个新的插件文件，命名格式为 `{provider}_plugin.py`：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{Provider} 插件实现
"""

from typing import List, Union, AsyncGenerator, Generator
from ..base_plugin import BaseLLMPlugin, ModelInfo, ChatMessage, ChatCompletion, ChatCompletionChunk

class {Provider}Plugin(BaseLLMPlugin):
    """
    {Provider} 插件实现
    """
    
    def __init__(self, name: str = "{provider}", **config):
        super().__init__(name, **config)
        
        # 初始化配置
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.{provider}.com")
        
        # 定义支持的模型
        self._supported_models = [
            ModelInfo(
                id="model-name",
                name="Model Display Name",
                provider="{provider}",
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=False,
                max_tokens=4096
            )
        ]
    
    def chat_completion(self, model: str, messages: List[ChatMessage], 
                       stream: bool = False, **kwargs) -> Union[ChatCompletion, Generator]:
        """同步聊天完成实现"""
        # TODO: 实现同步调用逻辑
        pass
    
    async def chat_completion_async(self, model: str, messages: List[ChatMessage], 
                                   stream: bool = False, **kwargs) -> Union[ChatCompletion, AsyncGenerator]:
        """异步聊天完成实现"""
        # TODO: 实现异步调用逻辑
        pass
```

### 2. 注册插件

在 `__init__.py` 中注册您的插件：

```python
from .{provider}_plugin import {Provider}Plugin

__all__ = [
    # ... 其他插件
    "{Provider}Plugin",
]
```

### 3. 配置插件

在应用配置中添加插件配置：

```python
config = {
    "plugins": {
        "{provider}": {
            "api_key": "your-api-key",
            "base_url": "https://api.{provider}.com",
            "timeout": 30,
            "max_retries": 3
        }
    }
}
```

---

## 基础插件类详解

### BaseLLMPlugin 类

`BaseLLMPlugin` 是所有LLM插件的基类，提供了标准化的接口和通用功能。

#### 核心属性

```python
class BaseLLMPlugin(ABC):
    def __init__(self, name: str, **config):
        self.name = name                    # 插件名称
        self.config = config               # 插件配置
        self.logger = get_logger(f"harborai.plugin.{name}")  # 日志记录器
        self._supported_models: List[ModelInfo] = []  # 支持的模型列表
```

#### 必须实现的抽象方法

##### 1. chat_completion (同步方法)

```python
@abstractmethod
def chat_completion(
    self,
    model: str,
    messages: List[ChatMessage],
    stream: bool = False,
    **kwargs
) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
    """
    同步聊天完成接口
    
    Args:
        model: 模型名称
        messages: 消息列表
        stream: 是否流式输出
        **kwargs: 其他参数（temperature, max_tokens等）
    
    Returns:
        非流式: ChatCompletion 对象
        流式: Generator[ChatCompletionChunk, None, None]
    """
    pass
```

##### 2. chat_completion_async (异步方法)

```python
@abstractmethod
async def chat_completion_async(
    self,
    model: str,
    messages: List[ChatMessage],
    stream: bool = False,
    **kwargs
) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
    """
    异步聊天完成接口
    
    Args:
        model: 模型名称
        messages: 消息列表
        stream: 是否流式输出
        **kwargs: 其他参数
    
    Returns:
        非流式: ChatCompletion 对象
        流式: AsyncGenerator[ChatCompletionChunk, None]
    """
    pass
```

#### 内置辅助方法

##### 1. 模型管理

```python
def supports_model(self, model_id: str) -> bool:
    """检查是否支持指定模型"""
    
def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
    """获取模型信息"""
```

##### 2. 请求验证

```python
def validate_request(self, model: str, messages: List[ChatMessage], **kwargs) -> None:
    """验证请求参数，抛出 PluginError 或 ValidationError"""
```

##### 3. 请求准备

```python
def prepare_request(self, model: str, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
    """将 ChatMessage 转换为 API 请求格式"""
```

##### 4. 结构化输出处理

```python
def handle_structured_output(
    self,
    response: ChatCompletion,
    response_format: Optional[Dict[str, Any]] = None,
    structured_provider: str = "agently",
    model: Optional[str] = None,
    original_messages: Optional[List[ChatMessage]] = None
) -> ChatCompletion:
    """处理结构化输出，支持 Agently 和原生解析"""
```

##### 5. 推理内容提取

```python
def extract_reasoning_content(self, response: Union[ChatCompletion, ChatCompletionChunk]) -> Optional[str]:
    """提取推理模型的思考过程"""
```

##### 6. 日志记录

```python
def log_request(self, model: str, messages: List[ChatMessage], **kwargs) -> None:
    """记录请求日志"""

def log_response(self, response: Union[ChatCompletion, ChatCompletionChunk], latency_ms: float) -> None:
    """记录响应日志"""
```

##### 7. 错误处理

```python
def create_error_response(self, error: Exception, model: str, request_id: Optional[str] = None) -> ChatCompletion:
    """创建标准化的错误响应"""
```

---

## 完整插件实现示例

以下是一个完整的插件实现示例，展示了如何实现一个支持所有特性的LLM插件：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例插件实现 - 展示完整的插件开发流程
"""

import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator

from ..base_plugin import (
    BaseLLMPlugin, ModelInfo, ChatMessage, ChatCompletion, 
    ChatCompletionChunk, ChatChoice, ChatChoiceDelta, Usage
)
from ...utils.logger import get_logger
from ...utils.exceptions import PluginError, ValidationError, TimeoutError
from ...utils.tracer import get_current_trace_id

logger = get_logger(__name__)


class ExamplePlugin(BaseLLMPlugin):
    """示例插件实现"""
    
    def __init__(self, name: str = "example", **config):
        """
        初始化示例插件
        
        Args:
            name: 插件名称
            **config: 配置参数
                - api_key: API密钥
                - base_url: API基础URL
                - timeout: 请求超时时间（秒）
                - max_retries: 最大重试次数
        """
        super().__init__(name, **config)
        
        # 从配置中获取参数
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.example.com")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        
        # 定义支持的模型列表
        self._supported_models = [
            ModelInfo(
                id="example-chat",
                name="Example Chat Model",
                provider="example",
                supports_streaming=True,
                supports_structured_output=True,
                supports_thinking=False,
                max_tokens=4096,
                context_window=8192
            ),
            ModelInfo(
                id="example-reasoning",
                name="Example Reasoning Model",
                provider="example",
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=True,
                max_tokens=8192,
                context_window=16384
            )
        ]
        
        # 初始化HTTP客户端
        self._client = None
        self._async_client = None
        
        self.logger.info(f"Example plugin initialized with base_url: {self.base_url}")
    
    def _get_client(self):
        """获取同步HTTP客户端"""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": f"HarborAI-{self.name}/1.0"
                    }
                )
            except ImportError:
                raise PluginError(self.name, "httpx library is required")
        return self._client
    
    def _get_async_client(self):
        """获取异步HTTP客户端"""
        if self._async_client is None:
            try:
                import httpx
                self._async_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "User-Agent": f"HarborAI-{self.name}/1.0"
                    }
                )
            except ImportError:
                raise PluginError(self.name, "httpx library is required")
        return self._async_client
    
    def _validate_config(self) -> None:
        """验证插件配置"""
        if not self.api_key:
            raise ValidationError("API key is required")
        
        if not self.base_url:
            raise ValidationError("Base URL is required")
    
    def _is_thinking_model(self, model: str) -> bool:
        """判断是否为推理模型"""
        model_info = self.get_model_info(model)
        return model_info.supports_thinking if model_info else False
    
    def _prepare_request_data(self, model: str, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """准备API请求数据"""
        # 使用基类的 prepare_request 方法
        request_data = self.prepare_request(model, messages, **kwargs)
        
        # 添加插件特定的参数处理
        if "temperature" in kwargs:
            request_data["temperature"] = max(0.0, min(2.0, kwargs["temperature"]))
        
        if "max_tokens" in kwargs:
            request_data["max_tokens"] = max(1, kwargs["max_tokens"])
        
        return request_data
    
    def _parse_response(self, response_data: Dict[str, Any], model: str) -> ChatCompletion:
        """解析API响应为标准格式"""
        choices = []
        
        for i, choice_data in enumerate(response_data.get("choices", [])):
            message_data = choice_data.get("message", {})
            
            # 创建消息对象
            message = ChatMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content", ""),
                reasoning_content=message_data.get("reasoning_content")  # 推理内容
            )
            
            choice = ChatChoice(
                index=i,
                message=message,
                finish_reason=choice_data.get("finish_reason")
            )
            choices.append(choice)
        
        # 解析使用统计
        usage_data = response_data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        return ChatCompletion(
            id=response_data.get("id", f"example_{int(time.time())}"),
            object="chat.completion",
            created=response_data.get("created", int(time.time())),
            model=model,
            choices=choices,
            usage=usage
        )
    
    def _parse_stream_chunk(self, chunk_data: Dict[str, Any], model: str) -> ChatCompletionChunk:
        """解析流式响应块"""
        choices = []
        
        for i, choice_data in enumerate(chunk_data.get("choices", [])):
            delta_data = choice_data.get("delta", {})
            
            delta = ChatChoiceDelta(
                role=delta_data.get("role"),
                content=delta_data.get("content"),
                reasoning_content=delta_data.get("reasoning_content")  # 推理内容
            )
            
            choice = ChatChoice(
                index=i,
                delta=delta,
                finish_reason=choice_data.get("finish_reason")
            )
            choices.append(choice)
        
        return ChatCompletionChunk(
            id=chunk_data.get("id", f"example_{int(time.time())}"),
            object="chat.completion.chunk",
            created=chunk_data.get("created", int(time.time())),
            model=model,
            choices=choices
        )
    
    def chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """同步聊天完成实现"""
        start_time = time.time()
        
        try:
            # 验证配置和请求
            self._validate_config()
            self.validate_request(model, messages, **kwargs)
            
            # 记录请求日志
            self.log_request(model, messages, **kwargs)
            
            # 准备请求数据
            request_data = self._prepare_request_data(model, messages, **kwargs)
            request_data["stream"] = stream
            
            # 发送请求
            client = self._get_client()
            
            if stream:
                return self._handle_stream_response(client, request_data, model)
            else:
                response = client.post("/chat/completions", json=request_data)
                response.raise_for_status()
                
                # 解析响应
                result = self._parse_response(response.json(), model)
                
                # 处理结构化输出
                if kwargs.get("response_format"):
                    result = self.handle_structured_output(
                        result,
                        kwargs.get("response_format"),
                        kwargs.get("structured_provider", "agently"),
                        model,
                        messages
                    )
                
                # 记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(result, latency_ms)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Chat completion failed: {str(e)}", 
                            extra={"trace_id": get_current_trace_id()})
            return self.create_error_response(e, model)
    
    def _handle_stream_response(self, client, request_data: Dict[str, Any], model: str) -> Generator[ChatCompletionChunk, None, None]:
        """处理流式响应"""
        try:
            with client.stream("POST", "/chat/completions", json=request_data) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # 移除 "data: " 前缀
                        
                        if data.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            chunk = self._parse_stream_chunk(chunk_data, model)
                            yield chunk
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            self.logger.error(f"Stream processing failed: {str(e)}")
            # 生成错误块
            error_chunk = ChatCompletionChunk(
                id=f"error_{int(time.time())}",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[
                    ChatChoice(
                        index=0,
                        delta=ChatChoiceDelta(content=f"Error: {str(e)}"),
                        finish_reason="error"
                    )
                ]
            )
            yield error_chunk
    
    async def chat_completion_async(
        self,
        model: str,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """异步聊天完成实现"""
        start_time = time.time()
        
        try:
            # 验证配置和请求
            self._validate_config()
            self.validate_request(model, messages, **kwargs)
            
            # 记录请求日志
            self.log_request(model, messages, **kwargs)
            
            # 准备请求数据
            request_data = self._prepare_request_data(model, messages, **kwargs)
            request_data["stream"] = stream
            
            # 发送异步请求
            client = self._get_async_client()
            
            if stream:
                return self._handle_async_stream_response(client, request_data, model)
            else:
                response = await client.post("/chat/completions", json=request_data)
                response.raise_for_status()
                
                # 解析响应
                result = self._parse_response(response.json(), model)
                
                # 处理结构化输出
                if kwargs.get("response_format"):
                    result = self.handle_structured_output(
                        result,
                        kwargs.get("response_format"),
                        kwargs.get("structured_provider", "agently"),
                        model,
                        messages
                    )
                
                # 记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(result, latency_ms)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Async chat completion failed: {str(e)}", 
                            extra={"trace_id": get_current_trace_id()})
            return self.create_error_response(e, model)
    
    async def _handle_async_stream_response(self, client, request_data: Dict[str, Any], model: str) -> AsyncGenerator[ChatCompletionChunk, None]:
        """处理异步流式响应"""
        try:
            async with client.stream("POST", "/chat/completions", json=request_data) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # 移除 "data: " 前缀
                        
                        if data.strip() == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(data)
                            chunk = self._parse_stream_chunk(chunk_data, model)
                            yield chunk
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            self.logger.error(f"Async stream processing failed: {str(e)}")
            # 生成错误块
            error_chunk = ChatCompletionChunk(
                id=f"error_{int(time.time())}",
                object="chat.completion.chunk",
                created=int(time.time()),
                model=model,
                choices=[
                    ChatChoice(
                        index=0,
                        delta=ChatChoiceDelta(content=f"Error: {str(e)}"),
                        finish_reason="error"
                    )
                ]
            )
            yield error_chunk
    
    def __del__(self):
        """清理资源"""
        if self._client:
            self._client.close()
        if self._async_client:
            asyncio.create_task(self._async_client.aclose())
```

---

## 高级特性

### 1. 推理模型支持

推理模型（如 OpenAI o1、DeepSeek R1）具有思考过程，需要特殊处理：

```python
def _is_thinking_model(self, model: str) -> bool:
    """判断是否为推理模型"""
    model_info = self.get_model_info(model)
    return model_info.supports_thinking if model_info else False

def _extract_reasoning_content(self, response_data: Dict[str, Any]) -> Optional[str]:
    """提取推理内容"""
    # 检查不同可能的字段位置
    reasoning_fields = ['reasoning_content', 'thinking', 'reasoning']
    
    for field in reasoning_fields:
        if field in response_data:
            return response_data[field]
    
    # 检查嵌套结构
    if 'choices' in response_data and response_data['choices']:
        message = response_data['choices'][0].get('message', {})
        for field in reasoning_fields:
            if field in message:
                return message[field]
    
    return None
```

### 2. 结构化输出支持

结构化输出允许模型按照指定的 JSON Schema 格式返回数据：

```python
def _handle_structured_output_request(self, request_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """处理结构化输出请求"""
    response_format = kwargs.get("response_format")
    
    if response_format and response_format.get("type") == "json_schema":
        # 添加结构化输出参数到请求中
        request_data["response_format"] = response_format
        
        # 某些模型可能需要在系统消息中添加格式说明
        schema = response_format.get("json_schema", {}).get("schema", {})
        format_instruction = f"Please respond in JSON format according to this schema: {json.dumps(schema)}"
        
        # 在消息中添加格式指令（如果需要）
        messages = request_data.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] += f"\n\n{format_instruction}"
        else:
            messages.insert(0, {"role": "system", "content": format_instruction})
    
    return request_data
```

### 3. 错误处理和重试

实现健壮的错误处理和重试机制：

```python
def _make_request_with_retry(self, client, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """带重试的请求发送"""
    last_exception = None
    
    for attempt in range(self.max_retries + 1):
        try:
            response = client.post(endpoint, json=data)
            
            # 处理不同的HTTP状态码
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # 速率限制
                wait_time = 2 ** attempt  # 指数退避
                self.logger.warning(f"Rate limited, waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)
                continue
            elif response.status_code in [500, 502, 503, 504]:  # 服务器错误
                if attempt < self.max_retries:
                    wait_time = 1 * (attempt + 1)
                    self.logger.warning(f"Server error {response.status_code}, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
            
            # 其他错误直接抛出
            response.raise_for_status()
            
        except Exception as e:
            last_exception = e
            if attempt < self.max_retries:
                wait_time = 1 * (attempt + 1)
                self.logger.warning(f"Request failed: {str(e)}, retrying in {wait_time}s")
                time.sleep(wait_time)
            else:
                break
    
    # 所有重试都失败
    raise last_exception or Exception("All retries failed")
```

### 4. 流式响应处理

正确处理 Server-Sent Events (SSE) 格式的流式响应：

```python
def _parse_sse_line(self, line: str) -> Optional[Dict[str, Any]]:
    """解析SSE格式的行"""
    if not line.startswith("data: "):
        return None
    
    data = line[6:].strip()  # 移除 "data: " 前缀
    
    if data == "[DONE]":
        return {"done": True}
    
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        self.logger.warning(f"Failed to parse SSE data: {data}")
        return None

def _process_stream_chunk(self, chunk_data: Dict[str, Any], model: str) -> Optional[ChatCompletionChunk]:
    """处理单个流式数据块"""
    if chunk_data.get("done"):
        return None
    
    # 解析并返回标准化的块
    return self._parse_stream_chunk(chunk_data, model)
```

---

## 测试指南

### 1. 单元测试

为您的插件创建全面的单元测试：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例插件单元测试
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from harborai.core.plugins.example_plugin import ExamplePlugin
from harborai.core.base_plugin import ChatMessage, ModelInfo
from harborai.utils.exceptions import PluginError, ValidationError


class TestExamplePlugin:
    """示例插件测试类"""
    
    @pytest.fixture
    def plugin_config(self):
        """插件配置fixture"""
        return {
            "api_key": "test-api-key",
            "base_url": "https://api.example.com",
            "timeout": 30,
            "max_retries": 3
        }
    
    @pytest.fixture
    def plugin(self, plugin_config):
        """插件实例fixture"""
        return ExamplePlugin(**plugin_config)
    
    @pytest.fixture
    def sample_messages(self):
        """示例消息fixture"""
        return [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="Hello, how are you?")
        ]
    
    def test_plugin_initialization(self, plugin, plugin_config):
        """测试插件初始化"""
        assert plugin.name == "example"
        assert plugin.api_key == plugin_config["api_key"]
        assert plugin.base_url == plugin_config["base_url"]
        assert len(plugin.supported_models) > 0
    
    def test_model_support(self, plugin):
        """测试模型支持检查"""
        assert plugin.supports_model("example-chat")
        assert plugin.supports_model("example-reasoning")
        assert not plugin.supports_model("non-existent-model")
    
    def test_model_info(self, plugin):
        """测试模型信息获取"""
        model_info = plugin.get_model_info("example-chat")
        assert model_info is not None
        assert model_info.id == "example-chat"
        assert model_info.provider == "example"
        assert model_info.supports_streaming is True
    
    def test_request_validation_success(self, plugin, sample_messages):
        """测试请求验证成功"""
        # 应该不抛出异常
        plugin.validate_request("example-chat", sample_messages)
    
    def test_request_validation_empty_messages(self, plugin):
        """测试空消息列表验证"""
        with pytest.raises(PluginError):
            plugin.validate_request("example-chat", [])
    
    def test_request_validation_unsupported_model(self, plugin, sample_messages):
        """测试不支持的模型验证"""
        with pytest.raises(Exception):  # ModelNotFoundError
            plugin.validate_request("unsupported-model", sample_messages)
    
    def test_config_validation_missing_api_key(self):
        """测试缺少API密钥的配置验证"""
        plugin = ExamplePlugin(base_url="https://api.example.com")
        with pytest.raises(ValidationError):
            plugin._validate_config()
    
    @patch('httpx.Client')
    def test_sync_chat_completion_success(self, mock_client_class, plugin, sample_messages):
        """测试同步聊天完成成功"""
        # 模拟HTTP响应
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "test-response-id",
            "object": "chat.completion",
            "created": 1234567890,
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
                "completion_tokens": 15,
                "total_tokens": 35
            }
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        # 执行测试
        result = plugin.chat_completion("example-chat", sample_messages)
        
        # 验证结果
        assert result.id == "test-response-id"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello! I'm doing well, thank you for asking."
        assert result.usage.total_tokens == 35
    
    @patch('httpx.AsyncClient')
    @pytest.mark.asyncio
    async def test_async_chat_completion_success(self, mock_client_class, plugin, sample_messages):
        """测试异步聊天完成成功"""
        # 模拟异步HTTP响应
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "id": "test-async-response-id",
            "object": "chat.completion",
            "created": 1234567890,
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
                "completion_tokens": 15,
                "total_tokens": 35
            }
        }
        mock_response.raise_for_status.return_value = None
        
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        # 执行测试
        result = await plugin.chat_completion_async("example-chat", sample_messages)
        
        # 验证结果
        assert result.id == "test-async-response-id"
        assert len(result.choices) == 1
        assert result.choices[0].message.content == "Hello! I'm doing well, thank you for asking."
    
    @patch('httpx.Client')
    def test_stream_chat_completion(self, mock_client_class, plugin, sample_messages):
        """测试流式聊天完成"""
        # 模拟流式响应
        mock_response = Mock()
        mock_response.iter_lines.return_value = [
            "data: " + json.dumps({
                "id": "test-stream-id",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None
                    }
                ]
            }),
            "data: " + json.dumps({
                "id": "test-stream-id",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "! How are you?"},
                        "finish_reason": "stop"
                    }
                ]
            }),
            "data: [DONE]"
        ]
        mock_response.raise_for_status.return_value = None
        
        mock_client = Mock()
        mock_client.stream.return_value.__enter__.return_value = mock_response
        mock_client_class.return_value = mock_client
        
        # 执行测试
        chunks = list(plugin.chat_completion("example-chat", sample_messages, stream=True))
        
        # 验证结果
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == "! How are you?"
        assert chunks[1].choices[0].finish_reason == "stop"
    
    def test_reasoning_content_extraction(self, plugin):
        """测试推理内容提取"""
        # 测试从响应中提取推理内容
        response_data = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42.",
                        "reasoning_content": "Let me think about this step by step..."
                    }
                }
            ]
        }
        
        reasoning = plugin._extract_reasoning_content(response_data)
        assert reasoning == "Let me think about this step by step..."
    
    def test_error_response_creation(self, plugin):
        """测试错误响应创建"""
        error = Exception("Test error")
        error_response = plugin.create_error_response(error, "example-chat")
        
        assert error_response.model == "example-chat"
        assert len(error_response.choices) == 1
        assert "Error: Test error" in error_response.choices[0].message.content
        assert error_response.choices[0].finish_reason == "error"


# 集成测试
class TestExamplePluginIntegration:
    """示例插件集成测试"""
    
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv("EXAMPLE_API_KEY"), reason="API key not provided")
    def test_real_api_call(self):
        """测试真实API调用（需要真实API密钥）"""
        plugin = ExamplePlugin(
            api_key=os.getenv("EXAMPLE_API_KEY"),
            base_url=os.getenv("EXAMPLE_BASE_URL", "https://api.example.com")
        )
        
        messages = [
            ChatMessage(role="user", content="Say hello in one word.")
        ]
        
        result = plugin.chat_completion("example-chat", messages)
        
        assert result is not None
        assert len(result.choices) > 0
        assert result.choices[0].message.content.strip()
```

### 2. 性能测试

创建性能基准测试：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
示例插件性能测试
"""

import time
import asyncio
import pytest
from concurrent.futures import ThreadPoolExecutor

from harborai.core.plugins.example_plugin import ExamplePlugin
from harborai.core.base_plugin import ChatMessage


class TestExamplePluginPerformance:
    """示例插件性能测试"""
    
    @pytest.fixture
    def plugin(self):
        """插件实例"""
        return ExamplePlugin(
            api_key="test-key",
            base_url="https://api.example.com"
        )
    
    @pytest.fixture
    def sample_messages(self):
        """示例消息"""
        return [
            ChatMessage(role="user", content="Hello, how are you?")
        ]
    
    @pytest.mark.performance
    def test_concurrent_requests(self, plugin, sample_messages):
        """测试并发请求性能"""
        num_requests = 10
        
        def make_request():
            start_time = time.time()
            try:
                result = plugin.chat_completion("example-chat", sample_messages)
                return time.time() - start_time, True
            except Exception as e:
                return time.time() - start_time, False
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in futures]
        
        total_time = time.time() - start_time
        latencies = [r[0] for r in results]
        success_count = sum(1 for r in results if r[1])
        
        # 性能断言
        assert success_count >= num_requests * 0.9  # 至少90%成功率
        assert max(latencies) < 30.0  # 最大延迟不超过30秒
        assert sum(latencies) / len(latencies) < 10.0  # 平均延迟不超过10秒
        
        print(f"Concurrent requests: {num_requests}")
        print(f"Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average latency: {sum(latencies)/len(latencies):.2f}s")
        print(f"Max latency: {max(latencies):.2f}s")
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_async_concurrent_requests(self, plugin, sample_messages):
        """测试异步并发请求性能"""
        num_requests = 20
        
        async def make_async_request():
            start_time = time.time()
            try:
                result = await plugin.chat_completion_async("example-chat", sample_messages)
                return time.time() - start_time, True
            except Exception as e:
                return time.time() - start_time, False
        
        start_time = time.time()
        
        tasks = [make_async_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        latencies = [r[0] for r in results]
        success_count = sum(1 for r in results if r[1])
        
        # 性能断言
        assert success_count >= num_requests * 0.9  # 至少90%成功率
        assert max(latencies) < 30.0  # 最大延迟不超过30秒
        assert total_time < num_requests * 2  # 总时间应该明显少于串行执行
        
        print(f"Async concurrent requests: {num_requests}")
        print(f"Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.1f}%)")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average latency: {sum(latencies)/len(latencies):.2f}s")
```

### 3. 运行测试

创建测试配置文件 `pytest.ini`：

```ini
[tool:pytest]
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
    performance: marks tests as performance tests (deselect with '-m "not performance"')
    slow: marks tests as slow (deselect with '-m "not slow"')

testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# 测试输出配置
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
    --color=yes

# 异步测试支持
asyncio_mode = auto
```

运行测试命令：

```bash
# 运行所有测试
pytest

# 只运行单元测试（排除集成测试）
pytest -m "not integration"

# 运行性能测试
pytest -m performance

# 运行特定插件的测试
pytest tests/test_example_plugin.py

# 生成覆盖率报告
pytest --cov=harborai.core.plugins --cov-report=html
```

---

## 最佳实践

### 1. 代码组织

#### 文件结构

```
your_plugin/
├── __init__.py
├── plugin.py              # 主插件实现
├── models.py              # 数据模型定义
├── exceptions.py          # 插件特定异常
├── utils.py               # 工具函数
└── tests/
    ├── __init__.py
    ├── test_plugin.py     # 单元测试
    ├── test_integration.py # 集成测试
    └── test_performance.py # 性能测试
```

#### 命名规范

- **插件类名**: `{Provider}Plugin` (如 `OpenAIPlugin`)
- **文件名**: `{provider}_plugin.py` (如 `openai_plugin.py`)
- **模型ID**: 使用厂商官方的模型标识符
- **配置键**: 使用小写下划线格式 (如 `api_key`, `base_url`)

### 2. 错误处理

#### 异常层次结构

```python
from harborai.utils.exceptions import PluginError, ValidationError, TimeoutError

class ExamplePluginError(PluginError):
    """示例插件基础异常"""
    pass

class ExampleAuthenticationError(ExamplePluginError):
    """认证错误"""
    pass

class ExampleRateLimitError(ExamplePluginError):
    """速率限制错误"""
    pass
```

#### 错误处理模式

```python
def _handle_api_error(self, response) -> None:
    """统一的API错误处理"""
    if response.status_code == 401:
        raise ExampleAuthenticationError("Invalid API key")
    elif response.status_code == 429:
        raise ExampleRateLimitError("Rate limit exceeded")
    elif response.status_code >= 500:
        raise ExamplePluginError(f"Server error: {response.status_code}")
    else:
        response.raise_for_status()
```

### 3. 配置管理

#### 配置验证

```python
def _validate_config(self) -> None:
    """验证插件配置"""
    required_fields = ["api_key"]
    
    for field in required_fields:
        if not getattr(self, field):
            raise ValidationError(f"Missing required configuration: {field}")
    
    # 验证URL格式
    if self.base_url and not self.base_url.startswith(("http://", "https://")):
        raise ValidationError("base_url must be a valid HTTP/HTTPS URL")
    
    # 验证数值范围
    if self.timeout <= 0:
        raise ValidationError("timeout must be positive")
```

#### 环境变量支持

```python
def __init__(self, name: str = "example", **config):
    super().__init__(name, **config)
    
    # 支持从环境变量读取配置
    import os
    self.api_key = config.get("api_key") or os.getenv("EXAMPLE_API_KEY")
    self.base_url = config.get("base_url") or os.getenv("EXAMPLE_BASE_URL", "https://api.example.com")
    self.timeout = int(config.get("timeout") or os.getenv("EXAMPLE_TIMEOUT", "30"))
```

### 4. 日志记录

#### 结构化日志

```python
def log_request(self, model: str, messages: List[ChatMessage], **kwargs) -> None:
    """记录请求日志"""
    self.logger.info(
        "Plugin request started",
        extra={
            "trace_id": get_current_trace_id(),
            "plugin": self.name,
            "model": model,
            "message_count": len(messages),
            "stream": kwargs.get('stream', False),
            "temperature": kwargs.get('temperature'),
            "max_tokens": kwargs.get('max_tokens'),
            "structured_output": bool(kwargs.get('response_format'))
        }
    )

def log_response(self, response: ChatCompletion, latency_ms: float) -> None:
    """记录响应日志"""
    usage = response.usage
    
    self.logger.info(
        "Plugin request completed",
        extra={
            "trace_id": get_current_trace_id(),
            "plugin": self.name,
            "latency_ms": latency_ms,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "reasoning_content_present": bool(self.extract_reasoning_content(response))
        }
    )
```

### 5. 性能优化

#### 连接池管理

```python
def _get_client(self):
    """获取HTTP客户端（带连接池优化）"""
    if self._client is None:
        import httpx
        
        # 优化连接池设置
        limits = httpx.Limits(
            max_keepalive_connections=20,  # 保持连接数
            max_connections=100,           # 最大连接数
            keepalive_expiry=30.0         # 连接保持时间
        )
        
        # 超时配置
        timeout = httpx.Timeout(
            connect=5.0,    # 连接超时
            read=30.0,      # 读取超时
            write=10.0,     # 写入超时
            pool=5.0        # 连接池超时
        )
        
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            limits=limits,
            headers=self._get_default_headers(),
            # 启用HTTP/2（如果服务器支持）
            http2=True
        )
    
    return self._client
```

#### 请求优化

```python
def _optimize_request_data(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """优化请求数据"""
    # 移除空值参数
    optimized = {k: v for k, v in request_data.items() if v is not None}
    
    # 压缩长消息（如果支持）
    if len(str(optimized.get("messages", ""))) > 10000:
        # 可以考虑消息压缩或截断策略
        pass
    
    return optimized
```

### 6. 安全考虑

#### API密钥保护

```python
def _mask_api_key(self, api_key: str) -> str:
    """遮蔽API密钥用于日志记录"""
    if not api_key or len(api_key) < 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"

def log_config(self) -> None:
    """记录配置信息（安全）"""
    self.logger.info(
        "Plugin configuration",
        extra={
            "plugin": self.name,
            "base_url": self.base_url,
            "api_key": self._mask_api_key(self.api_key),
            "timeout": self.timeout
        }
    )
```

#### 输入验证

```python
def _sanitize_input(self, content: str) -> str:
    """清理输入内容"""
    # 移除潜在的恶意内容
    import re
    
    # 限制长度
    if len(content) > 100000:  # 100KB限制
        raise ValidationError("Input content too long")
    
    # 移除控制字符
    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
    
    return content
```

---

## 常见问题解答

### Q1: 如何处理不同厂商的消息格式差异？

**A**: 使用基类的 `prepare_request` 方法进行标准化转换，然后在插件中进行厂商特定的格式调整：

```python
def _prepare_vendor_request(self, model: str, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
    """准备厂商特定的请求格式"""
    # 先使用基类方法进行标准转换
    request_data = self.prepare_request(model, messages, **kwargs)
    
    # 进行厂商特定的调整
    vendor_messages = []
    for msg in request_data["messages"]:
        if msg["role"] == "system":
            # 某些厂商可能需要特殊处理system消息
            vendor_messages.append({
                "role": "user",
                "content": f"[System]: {msg['content']}"
            })
        else:
            vendor_messages.append(msg)
    
    request_data["messages"] = vendor_messages
    return request_data
```

### Q2: 如何实现推理模型的思考过程提取？

**A**: 推理模型通常在响应中包含 `reasoning_content` 字段，需要在解析响应时特别处理：

```python
def _parse_reasoning_response(self, response_data: Dict[str, Any], model: str) -> ChatCompletion:
    """解析包含推理内容的响应"""
    choices = []
    
    for choice_data in response_data.get("choices", []):
        message_data = choice_data.get("message", {})
        
        # 提取推理内容
        reasoning_content = message_data.get("reasoning_content")
        
        message = ChatMessage(
            role=message_data.get("role", "assistant"),
            content=message_data.get("content", ""),
            reasoning_content=reasoning_content  # 保存推理内容
        )
        
        choice = ChatChoice(
            index=choice_data.get("index", 0),
            message=message,
            finish_reason=choice_data.get("finish_reason")
        )
        choices.append(choice)
    
    return ChatCompletion(
        id=response_data.get("id"),
        object="chat.completion",
        created=response_data.get("created"),
        model=model,
        choices=choices,
        usage=self._parse_usage(response_data.get("usage", {}))
    )
```

### Q3: 如何处理流式响应中的错误？

**A**: 在流式响应中，错误可能出现在任何时候，需要优雅地处理：

```python
def _handle_stream_response(self, response, model: str) -> Generator[ChatCompletionChunk, None, None]:
    """处理流式响应，包含错误处理"""
    try:
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            
            data = line[6:].strip()
            
            if data == "[DONE]":
                break
            
            try:
                chunk_data = json.loads(data)
                
                # 检查是否包含错误信息
                if "error" in chunk_data:
                    error_chunk = self._create_error_chunk(
                        chunk_data["error"], model
                    )
                    yield error_chunk
                    break
                
                chunk = self._parse_stream_chunk(chunk_data, model)
                yield chunk
                
            except json.JSONDecodeError as e:
                # JSON解析错误，生成错误块
                error_chunk = self._create_error_chunk(
                    f"JSON decode error: {str(e)}", model
                )
                yield error_chunk
                break
                
    except Exception as e:
        # 连接或其他错误
        error_chunk = self._create_error_chunk(
            f"Stream error: {str(e)}", model
        )
        yield error_chunk

def _create_error_chunk(self, error_message: str, model: str) -> ChatCompletionChunk:
    """创建错误块"""
    return ChatCompletionChunk(
        id=f"error_{int(time.time())}",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[
            ChatChoice(
                index=0,
                delta=ChatChoiceDelta(
                    role="assistant",
                    content=f"Error: {error_message}"
                ),
                finish_reason="error"
            )
        ]
    )
```

### Q4: 如何实现结构化输出的验证？

**A**: 结构化输出需要根据 JSON Schema 进行验证：

```python
def _validate_structured_output(self, content: str, schema: Dict[str, Any]) -> Any:
    """验证结构化输出"""
    try:
        import json
        import jsonschema
        
        # 解析JSON
        parsed_data = json.loads(content)
        
        # 使用JSON Schema验证
        jsonschema.validate(parsed_data, schema)
        
        return parsed_data
        
    except json.JSONDecodeError as e:
        raise PluginError(self.name, f"Invalid JSON in structured output: {str(e)}")
    except jsonschema.ValidationError as e:
        raise PluginError(self.name, f"Structured output validation failed: {str(e)}")
    except ImportError:
        # 如果没有jsonschema库，进行基础验证
        self.logger.warning("jsonschema library not available, skipping validation")
        return json.loads(content)
```

### Q5: 如何处理API速率限制？

**A**: 实现指数退避重试机制和速率限制检测：

```python
import time
import random
from typing import Optional

def _handle_rate_limit(self, response, attempt: int) -> Optional[float]:
    """处理速率限制，返回建议的等待时间"""
    if response.status_code == 429:
        # 检查Retry-After头
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        
        # 使用指数退避
        base_delay = 2 ** attempt
        jitter = random.uniform(0.1, 0.3)  # 添加抖动避免雷群效应
        return base_delay + jitter
    
    return None

def _make_request_with_rate_limit(self, client, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """带速率限制处理的请求"""
    for attempt in range(self.max_retries + 1):
        try:
            response = client.post(endpoint, json=data)
            
            if response.status_code == 200:
                return response.json()
            
            # 处理速率限制
            wait_time = self._handle_rate_limit(response, attempt)
            if wait_time and attempt < self.max_retries:
                self.logger.warning(f"Rate limited, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            
        except Exception as e:
            if attempt == self.max_retries:
                raise
            
            wait_time = 2 ** attempt
            self.logger.warning(f"Request failed, retrying in {wait_time}s: {str(e)}")
            time.sleep(wait_time)
    
    raise Exception("All retries exhausted")
```

### Q6: 如何添加自定义的模型参数？

**A**: 在插件中扩展参数处理逻辑：

```python
def _prepare_custom_parameters(self, **kwargs) -> Dict[str, Any]:
    """处理自定义参数"""
    custom_params = {}
    
    # 标准参数映射
    param_mapping = {
        "temperature": "temperature",
        "max_tokens": "max_tokens", 
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty"
    }
    
    # 厂商特定参数
    vendor_params = {
        "repetition_penalty": "repetition_penalty",  # 某些厂商使用这个而不是frequency_penalty
        "do_sample": "do_sample",                    # 是否采样
        "num_beams": "num_beams"                     # beam search
    }
    
    # 处理标准参数
    for param, api_param in param_mapping.items():
        if param in kwargs:
            custom_params[api_param] = kwargs[param]
    
    # 处理厂商特定参数
    for param, api_param in vendor_params.items():
        if param in kwargs:
            custom_params[api_param] = kwargs[param]
    
    return custom_params
```

### Q7: 如何实现插件的健康检查？

**A**: 添加健康检查方法：

```python
def health_check(self) -> Dict[str, Any]:
    """插件健康检查"""
    health_status = {
        "plugin": self.name,
        "status": "unknown",
        "timestamp": int(time.time()),
        "details": {}
    }
    
    try:
        # 检查配置
        self._validate_config()
        health_status["details"]["config"] = "valid"
        
        # 检查网络连接
        client = self._get_client()
        response = client.get("/health", timeout=5)  # 假设有健康检查端点
        
        if response.status_code == 200:
            health_status["status"] = "healthy"
            health_status["details"]["api"] = "accessible"
        else:
            health_status["status"] = "degraded"
            health_status["details"]["api"] = f"status_code_{response.status_code}"
            
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["details"]["error"] = str(e)
    
    return health_status

async def health_check_async(self) -> Dict[str, Any]:
    """异步健康检查"""
    # 类似的实现，但使用异步客户端
    pass
```

---

## 附录

### A. 完整的插件模板

创建新插件时可以使用的完整模板：

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{Provider} 插件模板
"""

import json
import time
import asyncio
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator

from ..base_plugin import (
    BaseLLMPlugin, ModelInfo, ChatMessage, ChatCompletion, 
    ChatCompletionChunk, ChatChoice, ChatChoiceDelta, Usage
)
from ...utils.logger import get_logger
from ...utils.exceptions import PluginError, ValidationError
from ...utils.tracer import get_current_trace_id

logger = get_logger(__name__)


class {Provider}Plugin(BaseLLMPlugin):
    """
    {Provider} 插件实现
    
    支持的功能：
    - 同步和异步聊天完成
    - 流式响应
    - 结构化输出（如果厂商支持）
    - 推理模型（如果厂商支持）
    """
    
    def __init__(self, name: str = "{provider}", **config):
        super().__init__(name, **config)
        
        # 配置参数
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.{provider}.com")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        
        # 支持的模型列表
        self._supported_models = [
            ModelInfo(
                id="{provider}-model-1",
                name="{Provider} Model 1",
                provider="{provider}",
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=False,
                max_tokens=4096,
                context_window=8192
            ),
            # 添加更多模型...
        ]
        
        # HTTP客户端
        self._client = None
        self._async_client = None
        
        self.logger.info(f"{Provider} plugin initialized")
    
    def _get_client(self):
        """获取同步HTTP客户端"""
        if self._client is None:
            import httpx
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._get_headers()
            )
        return self._client
    
    def _get_async_client(self):
        """获取异步HTTP客户端"""
        if self._async_client is None:
            import httpx
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=self._get_headers()
            )
        return self._async_client
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"HarborAI-{self.name}/1.0"
        }
    
    def _validate_config(self) -> None:
        """验证配置"""
        if not self.api_key:
            raise ValidationError("API key is required")
    
    def chat_completion(self, model: str, messages: List[ChatMessage], 
                       stream: bool = False, **kwargs) -> Union[ChatCompletion, Generator]:
        """同步聊天完成"""
        # TODO: 实现同步调用逻辑
        pass
    
    async def chat_completion_async(self, model: str, messages: List[ChatMessage], 
                                   stream: bool = False, **kwargs) -> Union[ChatCompletion, AsyncGenerator]:
        """异步聊天完成"""
        # TODO: 实现异步调用逻辑
        pass
```

### B. 测试模板

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
{Provider} 插件测试模板
"""

import pytest
from unittest.mock import Mock, patch

from harborai.core.plugins.{provider}_plugin import {Provider}Plugin
from harborai.core.base_plugin import ChatMessage


class Test{Provider}Plugin:
    """插件测试类"""
    
    @pytest.fixture
    def plugin(self):
        return {Provider}Plugin(
            api_key="test-key",
            base_url="https://api.{provider}.com"
        )
    
    @pytest.fixture
    def sample_messages(self):
        return [
            ChatMessage(role="user", content="Hello")
        ]
    
    def test_initialization(self, plugin):
        """测试初始化"""
        assert plugin.name == "{provider}"
        assert len(plugin.supported_models) > 0
    
    def test_model_support(self, plugin):
        """测试模型支持"""
        assert plugin.supports_model("{provider}-model-1")
    
    # 添加更多测试...
```

### C. 配置示例

```yaml
# config.yaml
plugins:
  {provider}:
    api_key: "${PROVIDER_API_KEY}"
    base_url: "https://api.{provider}.com"
    timeout: 30
    max_retries: 3
    
    # 插件特定配置
    custom_param: "value"
```

### D. 部署检查清单

在部署新插件前，请确保：

- [ ] 所有必需的依赖已安装
- [ ] API密钥和配置正确设置
- [ ] 单元测试全部通过
- [ ] 集成测试通过（如果有真实API密钥）
- [ ] 性能测试满足要求
- [ ] 错误处理覆盖所有已知场景
- [ ] 日志记录完整且不泄露敏感信息
- [ ] 文档更新完整
- [ ] 代码审查通过

### E. 故障排查指南

#### 常见问题及解决方案

1. **认证失败**
   - 检查API密钥是否正确
   - 确认API密钥权限
   - 验证请求头格式

2. **连接超时**
   - 检查网络连接
   - 调整超时设置
   - 验证base_url正确性

3. **响应解析错误**
   - 检查API响应格式变化
   - 验证JSON解析逻辑
   - 添加更多错误处理

4. **流式响应中断**
   - 检查SSE格式处理
   - 验证连接保持逻辑
   - 添加重连机制

#### 调试技巧

```python
# 启用详细日志
import logging
logging.getLogger("harborai.plugin.{provider}").setLevel(logging.DEBUG)

# 添加请求/响应日志
def debug_request(self, request_data):
    self.logger.debug(f"Request: {json.dumps(request_data, indent=2)}")

def debug_response(self, response_data):
    self.logger.debug(f"Response: {json.dumps(response_data, indent=2)}")
```

---

## 结语

本指南涵盖了 HarborAI 插件开发的所有重要方面。通过遵循这些最佳实践和示例，您应该能够快速开发出高质量、可靠的模型插件。

如果您在开发过程中遇到问题，请：

1. 查阅现有插件的实现作为参考
2. 运行相关的测试用例
3. 检查日志输出
4. 参考本指南的故障排查部分

祝您开发愉快！

---

**文档版本**: 1.0  
**最后更新**: 2024年12月  
**维护者**: HarborAI 开发团队