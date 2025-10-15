# HarborAI 日志查看功能使用指南

## 概述

HarborAI 日志查看工具已经过优化，现在支持更强大的日志类型过滤和分析功能。这些改进解决了用户对双重日志记录（请求和响应分离）的困惑，并提供了更好的用户体验。

## 背景说明

### 为什么会有两条记录？

HarborAI 的日志系统设计为每个 API 请求生成两条记录：

1. **请求记录** (`type: "request"`) - 记录请求信息，但 `success`、`latency`、`tokens`、`cost` 等字段为 `null`
2. **响应记录** (`type: "response"`) - 记录响应信息，包含完整的 `success`、`latency`、`tokens`、`cost` 等数据

这种设计是**有意为之的功能特性**，不是 bug，目的是：
- 分离关注点：请求和响应数据分开记录
- 便于调试：可以单独分析请求或响应
- 支持异步处理：请求可以立即记录，响应稍后记录

## 新功能特性

### 1. 布局模式

HarborAI 日志查看工具现在支持两种布局模式，通过 `--layout` 参数控制：

#### Classic 布局（经典布局）
- **默认布局模式**，保持传统的表格显示方式
- **Trace ID 作为首列**，便于快速识别和追踪
- 显示所有日志类型（REQ 和 RES），解决了之前只显示响应记录的问题
- 适合快速浏览和基本分析

```bash
# 使用经典布局（默认）
python view_logs.py --layout classic --limit 10

# 经典布局显示特定类型
python view_logs.py --layout classic --type request --limit 5
```

#### Enhanced 布局（增强布局）
- **基于 trace_id 的智能配对显示**，自动匹配请求和响应
- **双时间列设计**：分别显示请求时间和响应时间
- **自动计算耗时**：显示从请求到响应的完整耗时（毫秒）
- 提供更直观的请求-响应流程视图

```bash
# 使用增强布局
python view_logs.py --layout enhanced --limit 5

# 增强布局配合提供商过滤
python view_logs.py --layout enhanced --provider bytedance --limit 3
```

### 2. Trace ID 优化

为了提高可读性和用户体验，我们对 `trace_id` 进行了重要优化：

- **前缀优化**：从 `harborai_` 简化为 `hb_`
- **长度减少**：从 31 字符减少到 25 字符
- **格式规范**：`hb_{timestamp}_{random_part}`
- **显示优化**：在表格中自动截断显示，保持界面整洁

```bash
# 新的 trace_id 格式示例
hb_1703123456789_a1b2c3d4  # 25字符，更简洁易读
```

### 3. 默认行为改进

我们修复了一个重要的用户体验问题：

- **`--type` 参数默认值**：从 `response` 改为 `all`
- **经典布局修复**：现在默认显示所有日志类型（REQ 和 RES）
- **向后兼容**：所有现有功能保持不变，只是改善了默认体验

这意味着用户现在可以：
- 直接运行 `python view_logs.py` 查看完整的请求-响应流程
- 无需额外参数即可看到 REQ 和 RES 两种类型的记录
- 更直观地理解 API 调用的完整生命周期

### 4. 日志类型过滤

使用 `--type` 参数可以选择查看不同类型的日志：

```bash
# 显示所有日志（请求 + 响应）- 现在是默认行为
python view_logs.py --type all

# 仅显示请求日志
python view_logs.py --type request

# 仅显示响应日志
python view_logs.py --type response

# 配对显示请求-响应日志
python view_logs.py --type paired
```

### 5. 视觉区分

不同类型的日志现在有不同的视觉标识：

- 📤 **REQ** - 请求日志（蓝色）
- 📥 **RES** - 响应日志（绿色）
- ❓ **UNK** - 未知类型（黄色）

### 6. 配对显示功能

两种方式启用配对显示：

```bash
# 方式1：使用 --type paired
python view_logs.py --type paired

# 方式2：使用专用参数
python view_logs.py --show-request-response-pairs
```

配对显示会：
- 按 `trace_id` 匹配请求和响应
- 按时间顺序显示配对的日志
- 如果找不到配对，显示所有相关日志

### 7. 日志类型统计

使用 `--stats` 参数时，现在会显示额外的日志类型统计：

```bash
python view_logs.py --stats --days 7
```

输出包含：
- 传统的模型统计表格
- 新的日志类型分布统计（REQUEST/RESPONSE/UNKNOWN 的数量和占比）

### 8. JSON 格式支持

所有新功能都支持 JSON 格式输出：

```bash
# JSON 格式的统计信息
python view_logs.py --stats --format json

# JSON 格式的日志列表
python view_logs.py --type all --format json
```

## 使用示例

### 基本用法

```bash
# 查看帮助
python view_logs.py --help

# 查看最近的日志（默认显示所有类型）
python view_logs.py --limit 10

# 查看最近3天的所有日志，限制10条
python view_logs.py --days 3 --limit 10

# 查看特定模型的请求日志
python view_logs.py --type request --model "doubao-1-5-pro" --limit 5
```

### 布局模式用法

```bash
# 使用经典布局（默认）- 快速浏览所有日志
python view_logs.py --layout classic --limit 10

# 使用增强布局 - 智能配对显示，包含耗时信息
python view_logs.py --layout enhanced --limit 5

# 增强布局配合特定提供商过滤
python view_logs.py --layout enhanced --provider bytedance --limit 3

# 经典布局显示特定类型
python view_logs.py --layout classic --type request --limit 5
```

### 高级用法

```bash
# 配对显示特定提供商的日志
python view_logs.py --type paired --provider bytedance --limit 6

# 获取详细统计信息
python view_logs.py --stats --days 30

# 导出JSON格式的统计数据
python view_logs.py --stats --format json --days 7 > stats.json

# 结合布局和过滤的复合查询
python view_logs.py --layout enhanced --provider openai --model "gpt-4" --limit 3
```

### 调试场景

```bash
# 快速查看最近的所有日志（默认行为）
python view_logs.py --limit 10

# 使用经典布局查看请求，检查输入参数
python view_logs.py --layout classic --type request --limit 5

# 使用经典布局查看响应，检查输出结果
python view_logs.py --layout classic --type response --limit 5

# 使用增强布局查看配对日志，包含完整耗时信息
python view_logs.py --layout enhanced --limit 4

# 传统配对查看方式（仍然支持）
python view_logs.py --type paired --limit 4

# 调试特定模型的性能问题
python view_logs.py --layout enhanced --model "gpt-4" --limit 3

# 分析特定提供商的响应时间
python view_logs.py --layout enhanced --provider openai --days 1
```

## 模型成本配置说明

### 当前成本配置机制

HarborAI 的模型成本配置采用**硬编码**方式，所有模型价格都预定义在代码中，这确保了价格的一致性和稳定性。

#### 1. 实际成本配置位置

模型价格配置位于 `harborai/core/pricing.py` 文件中的 `PricingCalculator.MODEL_PRICING` 字典：

```python
# 模型价格配置（每1K tokens的价格，单位：人民币）
MODEL_PRICING: Dict[str, ModelPricing] = {
    # DeepSeek模型
    "deepseek-chat": ModelPricing(input_price=0.002, output_price=0.003),
    "deepseek-reasoner": ModelPricing(input_price=0.002, output_price=0.003),
    
    # 百度文心模型（Ernie）
    "ernie-3.5-8k": ModelPricing(input_price=0.0008, output_price=0.0032),
    "ernie-4.0-turbo-8k": ModelPricing(input_price=0.0008, output_price=0.0032),
    "ernie-x1-turbo-32k": ModelPricing(input_price=0.0008, output_price=0.0032),
    
    # 字节跳动豆包模型（Doubao）
    "doubao-1-5-pro-32k-character-250715": ModelPricing(input_price=0.0008, output_price=0.002),
    "doubao-seed-1-6-250615": ModelPricing(input_price=0.0008, output_price=0.002),
    
    # OpenAI模型价格（按汇率1美元=7.2人民币转换）
    "gpt-3.5-turbo": ModelPricing(input_price=0.0108, output_price=0.0144),
    "gpt-4": ModelPricing(input_price=0.216, output_price=0.432),
    "gpt-4-turbo": ModelPricing(input_price=0.072, output_price=0.216),
    "gpt-4o": ModelPricing(input_price=0.036, output_price=0.108),
    "gpt-4o-mini": ModelPricing(input_price=0.00015, output_price=0.0006),
}
```

#### 2. 环境变量控制

成本追踪功能通过环境变量控制：

- **`HARBORAI_COST_TRACKING=true`** - 启用成本追踪功能
- **`HARBORAI_FAST_PATH_SKIP_COST=false`** - 快速路径是否跳过成本计算


#### 3. 查看支持的模型和价格

使用 `PricingCalculator` 类可以查看所有支持的模型和价格：

```python
from harborai.core.pricing import PricingCalculator

# 创建价格计算器实例
calculator = PricingCalculator()

# 查看所有支持的模型
supported_models = calculator.list_supported_models()
print("支持的模型:", supported_models)

# 查看特定模型的价格
pricing = calculator.get_pricing("deepseek-chat")
if pricing:
    print(f"DeepSeek Chat 价格:")
    print(f"  输入: {pricing.input_price} 元/1K tokens")
    print(f"  输出: {pricing.output_price} 元/1K tokens")

# 计算成本示例
from harborai.core.cost_tracking import TokenUsage
token_usage = TokenUsage(input_tokens=1000, output_tokens=500)
cost = calculator.calculate_cost("deepseek", "deepseek-chat", token_usage)
print(f"成本计算结果: {cost.total_cost} 元")
```

#### 4. 自定义模型价格

如果需要自定义模型价格，有两种方法：

**方法1：使用 `add_model_pricing()` 方法（推荐）**

```python
from harborai.core.pricing import PricingCalculator, ModelPricing

calculator = PricingCalculator()

# 添加自定义模型价格
calculator.add_model_pricing(
    model_name="custom-model",
    pricing=ModelPricing(input_price=0.001, output_price=0.002)
)

# 验证添加成功
pricing = calculator.get_pricing("custom-model")
print(f"自定义模型价格: 输入={pricing.input_price}, 输出={pricing.output_price}")
```

**方法2：直接修改 `pricing.py` 文件**

在 `harborai/core/pricing.py` 文件的 `MODEL_PRICING` 字典中添加新的模型配置：

```python
MODEL_PRICING: Dict[str, ModelPricing] = {
    # 现有模型...
    
    # 添加新的自定义模型
    "my-custom-model": ModelPricing(input_price=0.001, output_price=0.002),
}
```

#### 5. 成本追踪在日志中的显示

当启用成本追踪时，日志中会显示详细的成本信息：

```json
{
  "trace_id": "hb_1703123456789_a1b2c3d4",
  "type": "response",
  "model": "deepseek-chat",
  "provider": "deepseek",
  "tokens": {
    "input": 150,
    "output": 75,
    "total": 225
  },
  "cost": {
    "input_cost": 0.0003,
    "output_cost": 0.000225,
    "total_cost": 0.000525,
    "currency": "RMB"
  }
}
```

#### 6. 最佳实践建议

1. **保持默认配置**：对于大多数用户，使用预定义的价格配置即可，价格相对稳定且准确
2. **自定义价格时机**：仅在以下情况下考虑自定义价格：
   - 使用了系统不支持的新模型
   - 有特殊的价格协议或折扣
   - 需要进行成本分析和预算控制
3. **价格更新策略**：定期检查官方价格变动，及时更新 `pricing.py` 中的配置
4. **测试验证**：修改价格配置后，运行相关测试确保成本计算正确

#### 7. 故障排除

**问题：日志中看不到成本信息**
- 检查 `HARBORAI_COST_TRACKING=true` 是否设置
- 确认模型名称在 `MODEL_PRICING` 字典中存在
- 验证 API 响应包含 token 使用信息

**问题：成本计算不准确**
- 检查 `pricing.py` 中的价格配置是否为最新
- 确认使用的模型名称与配置中的键名完全匹配
- 验证 token 计数是否正确

**问题：自定义模型价格不生效**
- 确保模型名称拼写正确
- 检查是否在正确的位置添加了价格配置
- 重启应用以确保配置生效

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--layout` | choice | `classic` | 表格布局模式：`classic`/`enhanced` |
| `--type` | choice | `all` | 日志类型过滤：`all`/`request`/`response`/`paired` |
| `--show-request-response-pairs` | flag | - | 配对显示（等同于 `--type paired`） |
| `--days` | int | 7 | 查看最近几天的日志 |
| `--limit` | int | 50 | 限制显示条数 |
| `--model` | string | - | 过滤特定模型 |
| `--provider` | string | - | 过滤特定提供商 |
| `--source` | choice | `auto` | 日志源：`auto`/`postgres`/`file` |
| `--format` | choice | `table` | 输出格式：`table`/`json` |
| `--stats` | flag | - | 显示统计信息而不是日志列表 |

## 演示脚本

运行演示脚本查看所有功能：

```bash
python demo_log_features.py
```

这个脚本会逐步演示所有新功能，包括：
- **布局模式**：经典布局和增强布局的对比展示
- **trace_id 优化**：新的 `hb_` 前缀格式展示
- **默认行为改进**：显示所有日志类型的默认行为
- **日志类型过滤**：不同类型的日志过滤
- **视觉区分效果**：REQ/RES 类型的视觉标识
- **配对显示功能**：智能配对和耗时计算
- **统计信息展示**：包含布局模式的统计
- **JSON 格式输出**：所有功能的 JSON 格式支持

## 技术实现

### 布局模式实现

#### Classic 布局
- **表格结构优化**：trace_id 作为首列，便于快速识别
- **默认行为修复**：修改 `--type` 参数默认值从 `response` 到 `all`
- **兼容性保持**：保持原有表格格式，确保向后兼容

#### Enhanced 布局
- **智能配对算法**：基于 `trace_id` 自动匹配请求和响应记录
- **双时间列设计**：分别显示请求时间和响应时间
- **耗时计算**：自动计算请求到响应的完整耗时（毫秒级精度）
- **数据聚合**：将配对的记录合并为单行显示

### Trace ID 优化实现

- **前缀简化**：从 `harborai_` 优化为 `hb_`，减少 9 个字符
- **长度控制**：总长度从 31 字符减少到 25 字符
- **格式标准化**：`hb_{timestamp}_{random_part}` 格式
- **显示截断**：在表格中智能截断显示，保持界面整洁

### 文件日志解析器改进

- 修改 `query_api_logs` 方法支持 `log_type` 参数
- 实现 `_create_paired_logs` 方法处理配对逻辑
- 按 `trace_id` 分组并匹配请求-响应对
- 添加布局模式支持和耗时计算

### PostgreSQL 客户端改进

- 动态调整 SQL 查询字段
- 模拟请求-响应记录生成
- 保持与文件日志解析器的一致性
- 支持新的布局模式和 trace_id 格式

### 视图层改进

- 添加日志类型列和视觉标识
- 实现日志类型统计功能
- 支持 Rich 库的彩色输出和简单文本输出
- 新增布局模式选择和渲染逻辑

## 故障排除

### 常见问题

1. **看不到请求日志**
   - 现在默认显示所有类型，如果仍看不到请求日志，检查日志文件是否包含请求记录
   - 使用 `--layout classic --type request` 专门查看请求日志

2. **增强布局显示不完整**
   - 某些请求可能没有对应的响应记录，导致配对失败
   - 使用 `--layout classic --type all` 查看所有记录进行诊断
   - 检查 `trace_id` 格式是否正确（应为 `hb_` 前缀）

3. **布局模式切换无效果**
   - 确保使用正确的参数：`--layout classic` 或 `--layout enhanced`
   - 检查是否有足够的数据进行配对显示（增强布局需要配对数据）

4. **trace_id 显示异常**
   - 新格式为 `hb_` 前缀，如果看到旧格式可能是历史数据
   - 使用 `--days 1` 查看最新的日志记录

5. **统计信息不准确**
   - 确保日志文件格式正确
   - 检查 `trace_id` 字段是否存在且格式正确

### 调试技巧

```bash
# 检查原始日志文件
Get-ChildItem -Path "./logs" -Filter "*.jsonl" | Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { Get-Content $_.FullName | Select-Object -First 10 }

# 对比两种布局模式的显示效果
python view_logs.py --layout classic --limit 5
python view_logs.py --layout enhanced --limit 5

# 检查 trace_id 格式和配对情况
python view_logs.py --layout classic --type all --limit 10

# 查看所有日志类型分布
python view_logs.py --stats --days 30

# 验证增强布局的耗时计算
python view_logs.py --layout enhanced --provider openai --limit 3

# 导出详细数据进行分析
python view_logs.py --type all --format json --days 7 > all_logs.json

# 检查特定 trace_id 的完整流程
python view_logs.py --layout enhanced --days 1 | grep "hb_"
```

## 总结

这些优化显著改善了 HarborAI 日志系统的用户体验：

### 主要改进成果

1. **双布局模式** - 提供经典和增强两种布局，满足不同场景需求
   - **Classic 布局**：快速浏览，trace_id 首列显示
   - **Enhanced 布局**：智能配对，包含耗时分析

2. **trace_id 优化** - 从 `harborai_` 简化为 `hb_`，提高可读性
   - 长度减少 24%（从 31 字符到 25 字符）
   - 界面更整洁，用户体验更佳

3. **默认行为改进** - 修复了只显示响应记录的问题
   - 现在默认显示所有日志类型（REQ + RES）
   - 用户无需额外参数即可看到完整流程

4. **解决了用户困惑** - 通过类型过滤和视觉区分，用户现在可以清楚地理解双重日志记录的设计

5. **提供了灵活性** - 用户可以根据需要选择布局模式和日志类型

6. **增强了分析能力** - 配对显示、耗时计算和类型统计提供了更深入的洞察

7. **保持了兼容性** - 所有现有功能继续正常工作，向后兼容

### 用户价值

- **提升效率**：通过布局模式快速定位问题
- **改善体验**：简化的 trace_id 和直观的界面
- **增强洞察**：耗时分析和智能配对帮助性能优化
- **降低门槛**：默认显示所有类型，新用户更容易上手

通过这些改进，HarborAI 的日志系统现在提供了更好的可观测性、更强的分析能力和更优的用户体验。

---

## Trace ID 查询功能

### 概述

HarborAI 日志查看工具现在支持强大的 `trace_id` 查询功能，允许用户按特定的 `trace_id` 查询相关的所有日志记录。这个功能支持从多种数据源查询日志，并提供友好的输出格式。

### 🔍 多数据源支持
- **PostgreSQL 数据库查询**：优先从 PostgreSQL 数据库查询日志
- **文件日志查询**：自动降级到文件日志系统
- **自动降级机制**：当 PostgreSQL 不可用时，自动使用文件日志

### 📊 灵活的输出格式
- **表格格式**：美观的表格显示（支持 Rich 库美化）
- **JSON 格式**：完整的结构化数据输出
- **详细信息**：显示日志的详细字段和统计信息

### 🛠️ 实用功能
- **trace_id 验证**：自动验证 trace_id 格式
- **最近日志列表**：列出最近的可用 trace_id
- **错误处理**：友好的错误提示和建议
- **多天查询**：支持查询指定天数范围内的日志

## Trace ID 查询命令行选项

| 选项 | 描述 | 默认值 |
|------|------|--------|
| `--trace-id` | 要查询的 trace_id | - |
| `--list-recent-trace-ids` | 列出最近的 trace_id | `false` |
| `--validate-trace-id` | 验证 trace_id 格式 | - |
| `--format` | 输出格式：`table` 或 `json` | `table` |
| `--days` | 查询最近几天的数据 | `1` |
| `--limit` | 限制返回的 trace_id 数量 | `20` |

## Trace ID 查询使用示例

### 1. 查询指定 trace_id
```bash
# 基本查询
python view_logs.py --trace-id hb_1760458760039_257c901f

# JSON 格式输出
python view_logs.py --trace-id hb_1760458760039_257c901f --format json
```

**输出示例：**
```
🔍 查询 trace_id: hb_1760458760039_257c901f
📊 数据源: file
📝 找到 3 条日志记录

          Trace ID 查询结果
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┩
┃ 序号 ┃ 时间                ┃ 模型                                ┃ 来源   ┃ 类型     ┃ 状态      ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━┩
│ 1    │ 2025-10-14 16:19:31 │ N/A                                 │ N/A    │ response │ ⏳ 处理中 │
│ 2    │ 2025-10-14 16:19:32 │ doubao-1-5-pro-32k-character-250715 │ N/A    │ request  │ ⏳ 处理中 │
│ 3    │ 2025-10-14 16:19:34 │ N/A                                 │ N/A    │ response │ ⏳ 处理中 │
└──────┴─────────────────────┴─────────────────────────────────────┴────────┴──────────┴───────────┘

================================================================================
📋 第一条记录详细信息:
----------------------------------------
trace_id: hb_1760458760039_257c901f
timestamp: 2025-10-14T16:19:31.560025
type: response
success: True
latency: 0.0
tokens: {'prompt_tokens': 107, 'completion_tokens': 507, 'total_tokens': 614}
cost: None
reasoning_content_present: True
error: None
response_summary: {'content': '关于"doubao-seed-1-6-250615"模型...', 'content_length': 152, ...}
source: file
file_name: harborai_20251015_001931.jsonl
line_number: 1
```

### 2. 列出最近的 trace_id
```bash
# 列出最近1天的 trace_id
python view_logs.py --list-recent-trace-ids

# 列出最近3天的 trace_id
python view_logs.py --list-recent-trace-ids --days 3

# 限制返回数量
python view_logs.py --list-recent-trace-ids --limit 10
```

**输出示例：**
```
🔍 最近的 Trace ID 列表
📊 数据源: both
📝 找到 1 个 trace_id
============================================================
          最近的 Trace ID
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 序号 ┃ Trace ID                  ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1    │ hb_1760458760039_257c901f │
└──────┴───────────────────────────┘

💡 使用方法:
   python view_logs.py --trace-id <trace_id>
   例如: python view_logs.py --trace-id hb_1760458760039_257c901f
```

### 3. 验证 trace_id 格式
```bash
# 验证有效的 trace_id
python view_logs.py --validate-trace-id hb_1760458760039_257c901f

# 验证无效的 trace_id
python view_logs.py --validate-trace-id invalid_trace_id
```

**输出示例：**
```
✅ trace_id 'hb_1760458760039_257c901f' 格式正确

❌ trace_id 'invalid_trace_id' 格式无效
💡 正确格式: hb_{timestamp}_{random}
   例如: hb_1760458760039_257c901f
```

### 4. 错误处理示例
```bash
# 查询不存在的 trace_id
python view_logs.py --trace-id hb_1234567890123_abcd1234
```

**输出示例：**
```
❌ 未找到 trace_id 'hb_1234567890123_abcd1234' 的日志记录

💡 提示:
   1. 检查 trace_id 是否正确
   2. 尝试使用 --list-recent-trace-ids 查看可用的 trace_id
   3. 调整 --days 参数扩大搜索范围
```

## Trace ID 格式

HarborAI 的 trace_id 遵循以下格式：
```
hb_{timestamp}_{random}
```

- `hb_`：固定前缀
- `{timestamp}`：13位时间戳
- `{random}`：8位随机字符串

**示例：** `hb_1760458760039_257c901f`

## 数据源配置

### PostgreSQL 配置
通过环境变量配置 PostgreSQL 连接：

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=harborai
export POSTGRES_USER=harborai
export POSTGRES_PASSWORD=your_password
```

### 文件日志配置
脚本会自动查找以下目录中的日志文件：
1. `./logs/`
2. `./harborai_logs/`
3. `~/.harborai/logs/`
4. `/tmp/harborai_logs/`

支持的文件格式：
- `.log` 文件
- `.jsonl` 文件

## Trace ID 查询输出字段说明

### 表格输出字段
| 字段 | 描述 |
|------|------|
| 序号 | 日志记录的序号 |
| 时间 | 日志记录的时间戳 |
| 模型 | 使用的AI模型名称 |
| 来源 | 数据来源（PostgreSQL/文件） |
| 类型 | 日志类型（request/response） |
| 状态 | 请求状态（✓成功/✗失败/⏳处理中） |

### JSON 输出字段
- `trace_id`：追踪ID
- `timestamp`：时间戳
- `type`：日志类型
- `model`：模型名称
- `messages`：请求消息
- `parameters`：请求参数
- `success`：是否成功
- `latency`：延迟时间
- `tokens`：token统计
- `cost`：成本信息
- `error`：错误信息
- `response_summary`：响应摘要
- `source`：数据源
- `file_name`：文件名（文件日志）
- `line_number`：行号（文件日志）

## 常见问题

### Q: 为什么显示 "PostgreSQL 连接失败"？
A: 需要设置 PostgreSQL 环境变量，特别是 `POSTGRES_PASSWORD`。如果不使用 PostgreSQL，脚本会自动降级到文件日志。

### Q: 为什么找不到日志文件？
A: 确保日志目录存在且包含 `.jsonl` 或 `.log` 格式的日志文件。检查以下目录：
- `./logs/`
- `./harborai_logs/`
- `~/.harborai/logs/`

### Q: trace_id 格式不正确怎么办？
A: HarborAI 的 trace_id 格式为 `hb_{timestamp}_{random}`。使用 `--validate-trace-id` 验证格式，或使用 `--list-recent-trace-ids` 查看可用的 trace_id。

### Q: 如何获取更多天的数据？
A: 使用 `--days` 参数：
```bash
python view_logs.py --list-recent-trace-ids --days 7
```

### Q: 如何获取完整的 JSON 数据？
A: 使用 `--format json` 参数：
```bash
python view_logs.py --trace-id <trace_id> --format json
```

### Q: 如何同时使用 trace_id 查询和其他过滤条件？
A: trace_id 查询是独立功能，不与其他过滤条件（如 `--model`、`--provider` 等）组合使用。如果指定了 `--trace-id`，其他过滤条件将被忽略。

## 技术实现

### 核心功能集成
trace_id 查询功能已完全集成到 `LogViewer` 类中：
- **初始化**：自动检测和配置数据源
- **查询逻辑**：优先 PostgreSQL，降级到文件日志
- **输出格式化**：支持表格和 JSON 格式
- **错误处理**：友好的错误提示和建议

### 数据源优先级
1. PostgreSQL 数据库（如果配置正确）
2. 文件日志系统（自动降级）

### 依赖检测
- 自动检测 `psycopg2` 是否可用
- 自动检测 `rich` 是否可用（用于美化输出）
- 优雅降级到基本功能

## 完整命令参考

```bash
# 基本日志查看（显示所有类型）
python view_logs.py

# 按 trace_id 查询
python view_logs.py --trace-id hb_1760458760039_257c901f

# 列出最近的 trace_id
python view_logs.py --list-recent-trace-ids

# 验证 trace_id 格式
python view_logs.py --validate-trace-id hb_1760458760039_257c901f

# 组合使用（扩大搜索范围）
python view_logs.py --list-recent-trace-ids --days 7 --limit 50

# JSON 格式输出
python view_logs.py --trace-id hb_1760458760039_257c901f --format json

# 传统日志查看功能（保持不变）
python view_logs.py --days 3 --model gpt-4 --provider openai --limit 10
python view_logs.py --layout enhanced --show-request-response-pairs
python view_logs.py --stats --days 7
```

通过这些 trace_id 查询功能，HarborAI 的日志系统现在提供了更精确的日志追踪能力，帮助用户快速定位和分析特定请求的完整生命周期。