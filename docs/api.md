# HarborAI API 详细文档

本文档提供 HarborAI 的完整 API 接口说明，包括所有支持的功能和详细的使用示例。

## 📋 目录

- [基础 API](#基础-api)
- [聊天完成 API](#聊天完成-api)
- [结构化输出](#结构化输出)
- [推理模型](#推理模型)
- [流式响应](#流式响应)
- [异步调用](#异步调用)
- [错误处理](#错误处理)
- [性能优化 API](#性能优化-api)

## 基础 API

### 客户端初始化

HarborAI 提供与 OpenAI SDK 完全兼容的 API 接口：

```python
from harborai import HarborAI

# 基础初始化
client = HarborAI(
    api_key="your-api-key",
    base_url="https://api.deepseek.com/v1"  # 可选，默认为 OpenAI
)

# 高性能初始化
from harborai.api.fast_client import FastHarborAI

fast_client = FastHarborAI(
    api_key="your-api-key",
    performance_mode="fast",  # fast, balanced, full
    enable_memory_optimization=True
)
```

### 支持的模型

HarborAI 支持多个 AI 服务提供商的模型：

| 提供商 | 模型名称 | 特性 | 推荐用途 |
|--------|----------|------|----------|
| **DeepSeek** | `deepseek-chat` | 高性价比、中文友好 | 通用对话、代码生成 |
| **DeepSeek** | `deepseek-reasoner` | 推理能力强 | 复杂推理、数学问题 |
| **百度千帆** | `ernie-x1-turbo-32k` | 长上下文、中文优化 | 长文档处理 |
| **豆包** | `doubao-1-6` | 推理模型 | 逻辑推理、分析 |
| **OpenAI** | `gpt-4o` | 多模态、高质量 | 复杂任务、创意写作 |

## 聊天完成 API

### 基础聊天

```python
# 同步调用
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "你好，请介绍一下自己。"}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

### 多轮对话

```python
# 维护对话历史
conversation = [
    {"role": "system", "content": "你是一个Python编程专家。"}
]

# 第一轮对话
conversation.append({"role": "user", "content": "如何创建一个列表？"})
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=conversation
)
conversation.append({"role": "assistant", "content": response.choices[0].message.content})

# 第二轮对话
conversation.append({"role": "user", "content": "如何向列表添加元素？"})
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=conversation
)
```

## 结构化输出

HarborAI 支持两种结构化输出方式：

### 1. JSON Schema 方式

```python
# 定义 JSON Schema
person_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "人员姓名"},
        "age": {"type": "integer", "description": "年龄"},
        "profession": {"type": "string", "description": "职业"},
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "技能列表"
        }
    },
    "required": ["name", "age", "profession"]
}

# 使用结构化输出
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "提取信息：张三，30岁，软件工程师，擅长Python和JavaScript"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "schema": person_schema
        }
    },
    structured_provider="agently"  # 可选："agently" 或 "native"
)

# 解析结果
import json
result = json.loads(response.choices[0].message.content)
print(f"姓名: {result['name']}")
print(f"年龄: {result['age']}")
print(f"职业: {result['profession']}")
```

### 2. Pydantic 模型方式

```python
from pydantic import BaseModel
from typing import List

# 定义 Pydantic 模型
class PersonInfo(BaseModel):
    """人员信息模型"""
    name: str  # 姓名
    age: int   # 年龄
    profession: str  # 职业
    skills: List[str] = []  # 技能列表

# 使用 Pydantic 模型
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "提取信息：李四，25岁，数据科学家，擅长机器学习和数据分析"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "schema": PersonInfo.model_json_schema()
        }
    }
)

# 直接解析为 Pydantic 对象
person = PersonInfo.model_validate_json(response.choices[0].message.content)
print(f"姓名: {person.name}")
print(f"年龄: {person.age}")
print(f"技能: {', '.join(person.skills)}")
```

## 推理模型

推理模型支持显示思考过程，适合复杂的逻辑推理任务：

```python
# 使用推理模型
response = client.chat.completions.create(
    model="deepseek-reasoner",  # 或 "doubao-1-6"
    messages=[
        {"role": "user", "content": "解方程：2x + 5 = 13，请详细说明解题步骤"}
    ]
)

# 推理模型会返回思考过程
print("思考过程:")
print(response.choices[0].message.content)
```

### 复杂推理示例

```python
# 数学问题推理
math_problem = """
一个班级有30名学生，其中60%是女生。
如果新转来5名男生，那么女生占总人数的百分比是多少？
请详细计算并说明每一步。
"""

response = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {"role": "user", "content": math_problem}
    ],
    temperature=0.1  # 降低随机性，提高推理准确性
)

print(response.choices[0].message.content)
```

## 流式响应

流式响应适合需要实时显示生成内容的场景：

### 同步流式响应

```python
# 同步流式调用
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "请写一个关于人工智能的短文"}
    ],
    stream=True
)

print("AI 正在生成内容:")
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")
```

### 异步流式响应

```python
import asyncio

async def async_stream_chat():
    """异步流式聊天示例"""
    response = await client.chat.completions.acreate(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "请介绍一下机器学习的基本概念"}
        ],
        stream=True
    )
    
    print("AI 正在生成内容:")
    async for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")

# 运行异步函数
asyncio.run(async_stream_chat())
```

## 异步调用

HarborAI 提供完整的异步支持，适合高并发场景：

### 基础异步调用

```python
import asyncio

async def async_chat_example():
    """异步聊天示例"""
    response = await client.chat.completions.acreate(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "什么是异步编程？"}
        ]
    )
    return response.choices[0].message.content

# 运行异步函数
result = asyncio.run(async_chat_example())
print(result)
```

### 并发处理多个请求

```python
import asyncio

async def batch_process():
    """批量处理多个请求"""
    questions = [
        "什么是Python？",
        "什么是机器学习？",
        "什么是深度学习？",
        "什么是自然语言处理？"
    ]
    
    # 创建并发任务
    tasks = []
    for question in questions:
        task = client.chat.completions.acreate(
            model="deepseek-chat",
            messages=[{"role": "user", "content": question}]
        )
        tasks.append(task)
    
    # 等待所有任务完成
    responses = await asyncio.gather(*tasks)
    
    # 处理结果
    for i, response in enumerate(responses):
        print(f"问题 {i+1}: {questions[i]}")
        print(f"回答: {response.choices[0].message.content}")
        print("-" * 50)

# 运行批量处理
asyncio.run(batch_process())
```

## 错误处理

HarborAI 提供完善的错误处理机制：

### 基础错误处理

```python
from harborai.core.exceptions import (
    HarborAIError,
    APIError,
    RateLimitError,
    AuthenticationError
)

try:
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "Hello, world!"}
        ]
    )
    print(response.choices[0].message.content)
    
except AuthenticationError as e:
    print(f"认证错误: {e}")
except RateLimitError as e:
    print(f"请求频率限制: {e}")
except APIError as e:
    print(f"API 错误: {e}")
except HarborAIError as e:
    print(f"HarborAI 错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 重试机制

```python
from harborai.core.retry import RetryConfig

# 配置重试策略
retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0
)

# 使用重试配置
client = HarborAI(
    api_key="your-api-key",
    retry_config=retry_config
)
```

## 性能优化 API

### FastHarborAI 客户端

```python
from harborai.api.fast_client import FastHarborAI

# 创建高性能客户端
fast_client = FastHarborAI(
    api_key="your-api-key",
    performance_mode="fast",  # 性能模式
    enable_memory_optimization=True,  # 启用内存优化
    enable_lazy_loading=True,  # 启用延迟加载
    memory_optimization={
        'cache_size': 2000,
        'object_pool_size': 200,
        'memory_threshold_mb': 100.0,
        'auto_cleanup_interval': 600
    }
)
```

### 性能监控

```python
# 获取性能统计
if hasattr(fast_client, 'get_memory_stats'):
    stats = fast_client.get_memory_stats()
    if stats:
        print(f"缓存命中率: {stats['cache']['hit_rate']:.1%}")
        print(f"内存使用: {stats['system_memory']['rss_mb']:.1f}MB")
        print(f"请求总数: {stats['requests']['total']}")

# 手动清理内存
if hasattr(fast_client, 'cleanup_memory'):
    fast_client.cleanup_memory(force_clear=True)
```

### 性能模式对比

| 模式 | 成本跟踪 | 详细日志 | 监控 | 链路追踪 | 适用场景 |
|------|----------|----------|------|----------|----------|
| **FAST** | ❌ | ❌ | ❌ | ❌ | 高并发生产环境 |
| **BALANCED** | ✅ | ❌ | ✅ | ❌ | 一般生产环境 |
| **FULL** | ✅ | ✅ | ✅ | ✅ | 开发调试环境 |

## 最佳实践

### 1. 模型选择建议

```python
# 根据任务类型选择合适的模型
def choose_model(task_type: str) -> str:
    """根据任务类型选择最适合的模型"""
    model_mapping = {
        "chat": "deepseek-chat",           # 通用对话
        "reasoning": "deepseek-reasoner",   # 复杂推理
        "long_context": "ernie-x1-turbo-32k",  # 长文档处理
        "creative": "gpt-4o",              # 创意写作
        "code": "deepseek-chat"            # 代码生成
    }
    return model_mapping.get(task_type, "deepseek-chat")
```

### 2. 错误重试策略

```python
import time
from typing import Optional

async def robust_chat_call(
    client: HarborAI,
    messages: list,
    model: str = "deepseek-chat",
    max_retries: int = 3
) -> Optional[str]:
    """健壮的聊天调用，包含重试机制"""
    
    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.acreate(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
            
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                await asyncio.sleep(wait_time)
                continue
            raise
            
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(1)
                continue
            raise
    
    return None
```

### 3. 内存优化使用

```python
# 大批量处理时的内存优化
async def process_large_batch(questions: list, batch_size: int = 10):
    """分批处理大量请求，避免内存溢出"""
    
    results = []
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        
        # 处理当前批次
        batch_tasks = [
            client.chat.completions.acreate(
                model="deepseek-chat",
                messages=[{"role": "user", "content": q}]
            )
            for q in batch
        ]
        
        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)
        
        # 清理内存（如果使用 FastHarborAI）
        if hasattr(client, 'cleanup_memory'):
            client.cleanup_memory()
        
        # 短暂休息，避免请求过于频繁
        await asyncio.sleep(0.1)
    
    return results
```

## 常见问题

### Q: 如何选择合适的性能模式？

**A**: 根据您的使用场景选择：
- **生产环境高并发**: 使用 `FAST` 模式
- **一般生产环境**: 使用 `BALANCED` 模式  
- **开发调试**: 使用 `FULL` 模式

### Q: 结构化输出失败怎么办？

**A**: 尝试以下解决方案：
1. 检查 JSON Schema 格式是否正确
2. 使用 `structured_provider="agently"` 提高成功率
3. 在 prompt 中明确要求返回 JSON 格式
4. 降低 `temperature` 参数提高一致性

### Q: 如何处理长文本？

**A**: 
1. 使用支持长上下文的模型如 `ernie-x1-turbo-32k`
2. 将长文本分段处理
3. 使用流式响应避免超时

---

**更多 API 详情请参考**: [HarborAI GitHub](https://github.com/ailijian/harborai) | [示例代码](../examples/)