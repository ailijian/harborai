# **PRD 文档：HarborAI**

## 1. 项目背景

随着大语言模型（LLM）在生产和研发中的广泛应用，开发者常常面临以下痛点：

* 各厂商 API 风格不同，迁移成本高。
* 模型调用形式多样（思考 / 非思考、流式 / 非流式、同步 / 异步），开发者需要手动适配。
* 结构化输出方式分散：部分厂商支持原生 schema，部分只能通过第三方工具实现。
* 缺乏统一的日志、监控、成本追踪体系。
* 异常与重试机制缺失，生产环境易受限流或 API 错误影响。
* 多模型冗余配置，不具备自动降级与容灾能力。
* 调用日志缺乏持久化方案，难以进行成本审计与长期分析。

**HarborAI** 的目标是成为 **世界级的多模型客户端标准**，通过插件化架构统一接口，增强可观测性与容错能力，为开发者带来近乎原生的开发体验。

---

## 2. 产品目标

1. **统一 API 标准**

   * 所有 LLM 调用对齐 **OpenAI SDK 风格**，降低学习成本。
   * 保持与原生接口几乎一致，开发者可无缝迁移现有代码。

2. **灵活的调用模式**

   * 支持 **推理模型（例如deepseek-reasoner）与非推理模型（例如GPT-4）**。
   * 同时支持 **同步 / 异步**、**流式 / 非流式**调用。
   * **自动兼容模型内置思考模式**：SDK 会动态检测响应中是否包含思考过程（reasoning_content），无需预先定义模型类型，实现真正的自动适配。

3. **强大的结构化输出**

   * **默认使用 Agently**：提供强大的流式结构化输出能力。
   * 支持标准 JSON Schema 定义，统一使用 Python dict 格式定义 schema。
   * 通过 `structured_provider` 参数在 "agently"（默认）和 "native" 之间切换解析方式。
   * **与 OpenAI 完全一致**：使用 `response_format` 参数，格式为 `{"type": "json_schema", "json_schema": {"name": "schema_name", "schema": {...}, "strict": true}}`。

4. **可观测性与异步日志**

   * **异步日志记录**，不阻塞主线程调用。
   * 全链路追踪：Trace ID、延迟、调用成功率、token 使用量。
   * 成本可视化：支持配置不同模型厂商的 API 价格，统计调用费用。
   * **日志脱敏**：屏蔽 API Key、用户敏感数据。

5. **重试与容错机制**

   * 标准化异常类型（网络、限流、认证、超时）。
   * 内置 **指数退避 + jitter** 重试机制。
   * 自动捕捉错误并分类上报，提升生产可用性。

6. **智能降级策略**

   * 一键设置模型调用降级规则：

     * 高峰期切换备用模型（DeepSeek → OpenAI → Anthropic）。
     * 在同一模型的不同云厂商间切换。

7. **持久化与长期存储**

   * 内置 **Docker 化日志存储方案**，支持 PostgreSQL。
   * 可配置日志生命周期：

     * 部分数据（如 trace）定期清理。
     * 关键数据（如成本审计）永久保存。

---

## 3. 用户画像 & 场景

### 用户画像

* **个人开发者**：希望快速尝试不同模型，无需反复修改调用代码。
* **研究人员**：需要对比多模型效果，统一评测 pipeline。
* **企业团队**：需要在生产环境接入多个模型，保障稳定性与成本可控。

### 使用场景

1. **快速模型切换**：一个配置文件即可在 DeepSeek、OpenAI、Anthropic 之间切换。
2. **流式输出应用**：聊天机器人可使用流式响应提升交互体验。
3. **成本审计与监控**：团队按月统计各模型调用费用，优化云资源开销。
4. **高峰期容灾**：主力模型限流时，自动切换到备用厂商。
5. **安全合规**：日志记录敏感信息自动脱敏，符合企业安全标准。
6. **推理模型应用**：研究人员可以观察模型的思考过程，分析推理链路。

---

## 4. 功能需求

### 4.1 核心功能

* [x] **统一 OpenAI 标准 API**（几乎无感迁移）
* [x] **同步 / 异步 / 流式调用**
* [x] **结构化输出（标准 JSON Schema 定义 + 默认 Agently 解析 + 可选厂商原生解析）**
* [x] **推理模型支持（自动兼容内置思考模式切换）**
* [x] **异步日志系统（trace_id、调用链、延迟、token 消耗）**
* [x] **成本统计（基于厂商 API 价格配置）**
* [x] **日志脱敏**
* [x] **OpenAI 命名空间一致性**：公开接口采用 `client.chat.completions.create(...)` 路径与参数命名，构造函数 `HarborAI(api_key, base_url)` 与 OpenAI Python SDK v1 对齐，返回结构对齐 `ChatCompletion` / `ChatCompletionChunk`。
* [x] **结构化输出一体化**：在 `create(...)` 中通过 `response_format` 参数启用结构化输出，使用标准 JSON Schema 定义格式，默认使用 Agently 解析，可通过 `structured_provider` 参数选择解析方式。
* [x] **流式一体化**：`create(stream=True)` 返回迭代器，按块输出，与 OpenAI 的 `ChatCompletionChunk` 结构字段保持一致，便于无缝替换。

### 4.2 高级功能

* [ ] **重试机制**（指数退避 + jitter）
* [ ] **异常标准化**（AuthError, RateLimitError, TimeoutError 等）
* [ ] **降级策略**（自定义 fallback 顺序，支持云厂商切换）
* [ ] **持久化存储**（PostgreSQL + Docker Compose）
* [ ] **日志生命周期管理**（自动清理 + 永久存档分类）

---

## 5. 非功能需求

* **性能**：调用封装开销 < 1ms。
* **兼容性**：Python 3.9+，支持 asyncio。
* **稳定性**：核心功能调用在高并发下保持 >99.9% 成功率。
* **可扩展性**：插件可通过 entry-point 独立发布。
* **安全性**：支持日志脱敏 & 可选日志禁用。

---

## 6. 系统架构

### 6.1 模块结构

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

---

## 7. 功能优先级

| 优先级 | 功能                                      |
| --- | --------------------------------------- |
| P0  | 统一 API / 插件机制 / 同步调用 / 异步日志             |
| P0  | 异常标准化 / 日志脱敏 / 成本统计                     |
| P0  | 推理模型支持 / 自动兼容内置思考模式                    |
| P1  | 流式调用 / 结构化输出切换（默认 Agently + 可选 Native） |
| P1  | 重试机制 / 容错策略                             |
| P2  | 降级策略 / 插件 entry-points                  |
| P3  | 持久化存储 / 日志生命周期管理                        |

---

## 8. 开发里程碑

### Milestone 1（核心可用）

* 统一 API（OpenAI 风格）
* DeepSeek 插件（支持推理模型）
* 同步 & 异步调用
* 异步日志 & trace\_id
* 成本统计 & 脱敏
* 结构化输出（默认 Agently）

### Milestone 2

* 流式调用支持
* 结构化输出提供者选择（Agently / Native）
* 推理模型自动兼容
* 异常标准化 & 重试机制

### Milestone 3

* 降级策略（多模型容灾）
* 插件 entry-point 机制
* Docker + PostgreSQL 日志存储

### Milestone 4

* 日志生命周期管理
* 文档完善 & PyPI 发布
* 社区贡献规范

---

## 9. 风险与对策

| 风险              | 对策                     |
| --------------- | ---------------------- |
| 各厂商 API 差异大     | 插件层做适配，统一 Schema       |
| 高并发下日志阻塞        | 采用异步写入 & 队列缓冲          |
| 降级策略复杂度高        | 提供默认策略 + 插件化扩展         |
| 成本计算误差          | 允许用户自定义 API 单价         |
| 推理模型格式不统一       | 插件层标准化思考过程输出格式         |
| Agently 依赖风险     | 提供原生 schema 作为备选方案     |

---

## 10. 未来扩展

* **多语言 SDK**：Node.js、Go
* **可观测性增强**：Prometheus + OpenTelemetry
* **插件市场**：类似 HuggingFace Hub 的 LLM 插件集市
* **智能调度**：根据价格 / 延迟 / 成功率自动选择最优模型
* **思考过程分析**：提供思考链路可视化工具

---

## 11. 使用体验与示例（OpenAI 一致）

以下示例与 OpenAI Python SDK v1 的使用方式保持一致，开发者可"零迁移"替换：

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

try:
    # 1) 极简的原生调用（与 OpenAI 完全一致）
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    print(resp.choices[0].message.content)

    # 2) 结构化输出（默认使用 Agently）
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
    print(json_resp.parsed)

    # 3) 指定使用厂商原生 schema
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
    print(native_resp.parsed)

    # 4) 极简的流式调用（统一走 create + stream=True）
    for chunk in client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        stream=True
    ):
        print(chunk.choices[0].delta.content, end="")

    # 5) 推理模型调用
    thinking_resp = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "分析量子计算的优势"}]
    )
    print(thinking_resp.choices[0].message.content)  # 最终答案
    if hasattr(thinking_resp.choices[0].message, 'reasoning_content'):
        print(thinking_resp.choices[0].message.reasoning_content)  # 思考过程

    # 6) 推理模型的流式调用
    for chunk in client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": "解释相对论"}],
        stream=True
    ):
        if hasattr(chunk.choices[0].delta, "reasoning_content") and chunk.choices[0].delta.reasoning_content:
            print(f"思考: {chunk.choices[0].delta.reasoning_content}")
        if chunk.choices[0].delta.content:
            print(f"回答: {chunk.choices[0].delta.content}")

except Exception as e:
    print("调用失败：", e)
```

**OpenAI 标准结构化输出调用方式确认**：

```python
from openai import OpenAI
import json, os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 定义标准 JSON Schema
schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "confidence": {"type": "number"}
    },
    "required": ["name", "age", "confidence"],
    "additionalProperties": False
}

# OpenAI 标准调用
resp = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "张三25岁，请按JSON返回"}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "Person", "schema": schema, "strict": True}
    },
    temperature=0
)

# 解析结果
person = json.loads(resp.choices[0].message.content)
print(person)  # {'name': '张三', 'age': 25, 'confidence': 0.98}
```

**HarborAI 标准调用方式（与 OpenAI 完全一致）**：

```python
# HarborAI 标准版本 - 与 OpenAI 完全一致的调用方式
resp = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[{"role": "user", "content": "张三25岁，请按JSON返回"}],
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "Person", "schema": schema, "strict": True}
    }
)
print(resp.parsed)  # 自动解析的结构化结果
```

验收标准（Milestone 1 必达）：

- 替换 OpenAI SDK 为 HarborAI，以上示例无需改动业务逻辑即可运行；
- `resp` 的字段对齐 OpenAI `ChatCompletion`（含 `choices`, `usage` 等）；
- 流式 `chunk` 字段对齐 `ChatCompletionChunk`；
- 当传入 `response_format` 时返回 JSON 化结构，使用标准 JSON Schema 定义，默认使用 Agently 解析；
- 推理模型自动提供思考过程输出，兼容模型内置的思考/非思考模式切换。

---

✅ **总结**
本项目核心价值是 **打造统一、灵活、生产可用的多模型客户端标准**。
相较于现有 SDK，HarborAI 的独特优势是：

* **统一 OpenAI 风格**（几乎原生体验）。
* **灵活调用模式（推理 / 非推理，流式 / 非流式，同步 / 异步）**。
* **智能结构化输出（默认 Agently + 可选 Native）& 降级容灾**。
* **推理模型自动兼容**，支持模型内置推理模式切换。
* **可观测、成本统计、持久化日志**，真正满足企业级需求。

---