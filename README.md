# HarborAI

> 世界级多模型统一客户端，提供与 OpenAI SDK 几乎一致的开发体验

## 🚀 特性

- **与 OpenAI SDK 完全一致的调用体验**：无缝迁移，零学习成本
- **插件化架构**：轻松支持多个 LLM 厂商（OpenAI、DeepSeek、Doubao、Wenxin 等）
- **思考模型支持**：原生支持 DeepSeek-R1 等思考模型的 reasoning_content 字段
- **结构化输出**：默认使用 Agently，支持厂商原生 schema，支持流式结构化输出
- **生产级可观测性**：全链路 Trace ID、异步日志、PostgreSQL 存储
- **智能容错降级**：自动重试、模型降级、厂商降级

## 📦 安装

```bash
pip install harborai

# 安装 PostgreSQL 支持（可选）
pip install harborai[postgres]

# 开发环境安装
pip install harborai[dev]
```

## 🔧 快速开始

### 基础调用

```python
import os
from harborai import HarborAI

client = HarborAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

messages = [
    {"role": "user", "content": "用一句话解释量子纠缠"}
]

# 极简的原生调用
resp = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)
print(resp.choices[0].message.content)
```

### 思考模型调用

```python
# 调用思考模型（如 deepseek-r1）
resp = client.chat.completions.create(
    model="deepseek-r1",
    messages=messages
)
print(resp.choices[0].message.content)
if hasattr(resp.choices[0].message, 'reasoning_content'):
    print(resp.choices[0].message.reasoning_content)  # 思考过程
```

### 结构化输出

```python
# 结构化输出调用（默认使用 Agently）
json_resp = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "BookInfo",
            "schema": {
                "type": "object",
                "properties": {
                    "book_title": {"type": "string"}
                },
                "required": ["book_title"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)
```

## 🏗️ 架构设计

```
/harborai
├── __init__.py
├── config/
│   └── settings.py
├── core/
│   ├── base_plugin.py
│   ├── client_manager.py
│   └── plugins/
│       ├── openai_plugin.py
│       ├── deepseek_plugin.py
│       ├── doubao_plugin.py
│       └── wenxin_plugin.py
├── api/
│   ├── client.py
│   ├── decorators.py
│   └── structured.py
├── utils/
│   ├── logger.py
│   ├── exceptions.py
│   ├── retry.py
│   └── tracer.py
├── storage/
│   ├── postgres_logger.py
│   └── lifecycle.py
└── cli/
    └── main.py
```

## 📄 许可证

MIT License