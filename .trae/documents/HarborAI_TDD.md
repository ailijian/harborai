# HarborAI 技术设计文档 (TDD)

## 🎉 重要更新：性能优化重大突破

**最新成果**（2025-09-30）：

* ✅ **性能超越**：HarborAI FAST模式比Agently基准快 **18%**（0.82x性能比率）

* ✅ **TDD验证**：所有性能目标100%达成（3/3通过）

* ✅ **响应时间**：平均1.32秒（目标2.0秒）

* ✅ **客户端池命中率**：94.4%（超越90%目标）

**关键技术突破**：

* 快速结构化输出路径：专门为FAST模式优化的处理路径

* 客户端池优化：94.4%命中率，显著减少连接开销

* Schema缓存系统：8.2%性能提升，LRU淘汰机制

* 配置缓存机制：减少重复计算，提升处理效率

详细信息请参考：[性能优化完整报告](./结构化输出性能优化建议.md) 和 [优化成果总结](./HarborAI_性能优化成果总结.md)

***

## 一、项目总览

**项目名称**：HarborAI
**定位**：一个世界级的多模型统一客户端，提供与 OpenAI SDK 几乎一致的开发体验，兼具灵活性、可靠性与可观测性。
**目标**：

* 降低接入多模型生态的学习成本。

* 支持推理模型（如 deepseek-reasoner）与非推理模型调用。

* 提供生产级的日志、容错、降级、持久化能力。

* 支持结构化输出（默认Agently/可选原生schema/流式结构化）。

***

## 二、架构设计

### 2.1 核心设计理念

1. **与 OpenAI 完全一致的调用体验**

   * HarborAI 的入口类 `HarborAI` 与 OpenAI 的 `OpenAI` 保持一致。

   * 所有调用方式均复用 `.chat.completions.create()` 接口。

   * 只在参数上扩展（如 `structured_provider`、`retry_policy`），降低迁移成本。

2. **插件化架构**

   * 每个模型厂商（OpenAI, DeepSeek, Doubao, Wenxin…）作为独立插件。

   * 插件继承 `BaseLLMPlugin`，注册到 `ClientManager`。

3. **推理模型与非推理模型支持**

   * SDK 层动态检测响应中是否包含 `reasoning_content` 字段，无需预先定义模型类型。

   * 所有插件统一处理思考和非思考模式，根据实际响应内容自动适配。

   * **自动兼容模型内置思考模式**：当模型内置自动切换思考/非思考模式时，SDK 会自动适配并在响应中提供思考过程（reasoning\_content）。

4. **可观测性与异步日志**

   * 全链路 Trace ID。

   * 异步日志写入，避免阻塞核心调用。

   * 记录指标：成功率、延迟、token 使用量、调用成本。

   * 支持日志脱敏与生命周期管理（PostgreSQL + Docker 部署）。

5. **结构化输出支持**

   * **默认使用 Agently**：提供强大的流式结构化输出能力。

   * 支持厂商原生 schema 输出（可通过参数指定）。

   * HarborAI 通过 `response_format` 参数启用结构化输出，通过 `structured_provider` 参数选择解析方式（"agently" 或 "native"）。

6. **容错与降级策略**

   * 自动重试、标准化异常。

   * 开发者可一键设置：

     * 模型降级（gpt-4 → gpt-3.5）。

     * 厂商降级（DeepSeek API → OpenAI API）。

***

### 2.2 模块结构

```
/harborai
├── __init__.py
├── config/
│   └── settings.py
│
├── core/
│   ├── base_plugin.py
│   ├── client_manager.py
│   ├── plugins/
│   │   ├── openai_plugin.py
│   │   ├── deepseek_plugin.py
│   │   ├── doubao_plugin.py
│   │   └── wenxin_plugin.py
│
├── api/
│   ├── client.py          # HarborAI 主入口，统一接口
│   ├── decorators.py
│   └── structured.py      # 结构化输出支持
│
├── utils/
│   ├── logger.py
│   ├── exceptions.py
│   ├── retry.py
│   └── tracer.py
│
├── storage/
│   ├── postgres_logger.py
│   └── lifecycle.py
│
├── cli/
│   └── main.py
│
├── docs/
└── examples/
```

***

## 三、接口设计

### 3.1 SDK 调用方式（与 OpenAI 一致）

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

# 1. 极简的原生调用
resp = client.chat.completions.create(
    model="gpt-4",
    messages=messages
)
print(resp.choices[0].message.content)

# 2. 结构化输出调用（默认使用 Agently）
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

# 3. 结构化输出调用（指定使用厂商原生 schema）
native_resp = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "TitleInfo",
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"}
                },
                "required": ["title"],
                "additionalProperties": False
            },
            "strict": True
        }
    },
    structured_provider="native"
)

# 4. 流式调用
for chunk in client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    stream=True
):
    print(chunk)
```

***

### 3.2 推理模型支持

```python
# 调用推理模型（如 deepseek-reasoner）
resp = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages
)
print(resp.choices[0].message.content)
if hasattr(resp.choices[0].message, 'reasoning_content'):
    print(resp.choices[0].message.reasoning_content)  # 思考过程（如果有）

# 推理模型的流式调用
for chunk in client.chat.completions.create(
    model="deepseek-reasoner",
    messages=messages,
    stream=True
):
    if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
        print(f"思考: {chunk.choices[0].delta.reasoning_content}")
    if chunk.choices[0].delta.content:
        print(f"回答: {chunk.choices[0].delta.content}")

# 推理模型的结构化输出
thinking_resp = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[{"role": "user", "content": "分析一下量子计算的优势和挑战"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "QuantumAnalysis",
            "schema": {
                "type": "object",
                "properties": {
                    "advantages": {"type": "string"},
                    "challenges": {"type": "string"}
                },
                "required": ["advantages", "challenges"],
                "additionalProperties": False
            },
            "strict": True
        }
    }
)
print(thinking_resp.parsed)  # 结构化结果
if hasattr(thinking_resp.choices[0].message, 'reasoning_content'):
    print(thinking_resp.choices[0].message.reasoning_content)  # 思考过程
```

***

### 3.3 参数扩展

| 参数名                   | 类型         | 说明                                                         |
| --------------------- | ---------- | ---------------------------------------------------------- |
| `response_format`     | dict       | 启用结构化输出，格式：{"type": "json\_schema", "json\_schema": {...}} |
| `structured_provider` | str        | 结构化输出提供者："agently"（默认）或 "native"                           |
| `extra_body`          | dict       | 思考模式开关，火山引擎：{"thinking": {"type": "enabled/disabled"}}     |
| `retry_policy`        | dict       | 配置重试策略（次数、指数退避等）                                           |
| `fallback`            | list\[str] | 降级模型列表                                                     |
| `trace_id`            | str        | 自定义 Trace ID                                               |
| `cost_tracking`       | bool       | 是否统计调用成本                                                   |

***

## 四、核心模块设计

### 4.1 插件基类 `BaseLLMPlugin`

```python
class BaseLLMPlugin(ABC):
    name: str

    @abstractmethod
    def chat_completion(self, messages, stream=False, **kwargs):
        pass

    @abstractmethod
    async def chat_completion_async(self, messages, stream=False, **kwargs):
        pass

    def extract_reasoning_content(self, response):
        """提取思考过程，动态检测reasoning_content字段"""
        return None
```

***

### 4.2 ClientManager

* 动态扫描 `plugins/` 目录，注册插件。

* 根据 `model` 名称找到对应厂商插件。

* 若配置了 `fallback`，在调用失败时尝试降级。

* 动态检测响应中是否包含reasoning\_content字段，并相应处理响应格式。

***

### 4.3 API 层

#### `HarborAI` 主入口

* 负责统一参数校验。

* 路由到正确的插件。

* 应用装饰器（Trace + 日志 + 重试）。

* 处理结构化输出提供者选择（Agently vs Native）。

* 对外暴露 OpenAI 风格接口：

  * `client.chat.completions.create(...)`

***

### 4.4 日志与存储

* **异步日志管道**：调用完成后将日志推入异步队列，由独立 worker 持久化到 PostgreSQL。

* **存储字段**：

  * trace\_id

  * model\_name

  * request / response

  * latency / tokens / cost

  * success / failure

  * reasoning\_content\_present（是否包含思考过程）

  * structured\_provider（使用的结构化输出提供者）

* **生命周期管理**：

  * 短期数据：7天自动清理。

  * 关键日志：永久保存。

***

## 五、容错与降级

* **重试策略**：

  * 默认 3 次，指数退避。

  * 可自定义。

* **降级策略**：

  * fallback=\["gpt-4", "gpt-3.5"]

  * 当 `gpt-4` 超时或配额不足时自动切换。

***

## 六、扩展性

* 插件化 → 轻松支持新厂商。

* 日志模块可替换（如接入 ELK、Prometheus）。

* 存储可切换为 MySQL、MongoDB。

* SDK 接口保持稳定，方便社区贡献。

***

## 七、推理模型的定义（HarborAI 版本）

* **推理模型 (Reasoner Model)**
  指原生支持生成"思考过程"与"最终答案"的模型。
  例如：

  * **deepseek-reasoner**（先输出 reasoning\_content，再输出最终结果）。

  * OpenAI 官方 SDK 已支持这类模型调用。

  * **Agently** 库扩展了对推理模型的支持，特别是**流式结构化输出**。

* **非推理模型 (Standard Model)**
  普通大模型，不会单独输出思考过程。
  例如：GPT-4、GPT-3.5、文心一言等。

* **自动兼容模式**
  当模型内置自动切换推理/非推理模式时，HarborAI 会：

  * 自动检测响应中是否包含推理过程

  * 在响应对象中提供 `reasoning_content` 或 `thinking`字段

  * 保持与 OpenAI 格式的兼容性

HarborAI 会在 SDK 层根据 `model_type` 自动处理调用逻辑，开发者无需额外区分。

***

