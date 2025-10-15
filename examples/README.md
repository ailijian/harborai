# HarborAI 应用案例集合

本目录包含了HarborAI客户端的完整应用案例，展示了从基础功能到复杂场景的实际使用方法。所有案例都遵循HarborAI的OpenAI兼容接口设计，体现其统一API、插件化架构、可观测性等核心特性。

## 📁 目录结构

```
examples/
├── README.md                    # 本文档
├── .env.example                 # 环境变量配置模板
├── requirements.txt             # 依赖包列表
├── basic/                       # 基础功能使用案例
│   ├── README.md
│   ├── simple_chat.py          # 简单聊天调用
│   ├── async_calls.py          # 异步调用示例
│   ├── streaming_output.py     # 流式输出示例
│   └── reasoning_models.py     # 推理模型调用
├── intermediate/                # 中级功能集成案例
│   ├── README.md
│   ├── structured_output.py    # 结构化输出（Agently vs Native）
│   ├── multi_model_switch.py   # 多模型切换
│   ├── cost_tracking.py        # 成本追踪
│   └── logging_monitoring.py   # 日志监控
├── advanced/                    # 高级功能组合应用案例
│   ├── README.md
│   ├── retry_fallback.py       # 容错与重试机制
│   ├── degradation_strategy.py # 降级策略
│   ├── batch_processing.py     # 批量处理
│   └── performance_optimization.py # 性能优化
└── scenarios/                   # 跨场景综合应用案例
    ├── README.md
    ├── chatbot/                 # 聊天机器人
    ├── content_generator/       # 内容生成系统
    ├── data_analyst/           # 数据分析助手
    └── enterprise_integration/ # 企业级应用集成
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 复制环境变量模板
cp .env.example .env

# 编辑环境变量，填入你的API密钥（推荐使用 DeepSeek）
# DEEPSEEK_API_KEY=your_deepseek_api_key  # 推荐：获取地址 https://platform.deepseek.com/api_keys
# OPENAI_API_KEY=your_openai_api_key
# ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 2. 运行基础案例

```bash
# 简单聊天调用
python basic/simple_chat.py

# 异步调用示例
python basic/async_calls.py

# 流式输出示例
python basic/streaming_output.py
```

## 📚 案例分类说明

### 🔰 基础功能使用案例 (basic/)

适合初学者，展示HarborAI的核心功能：
- **简单聊天调用**: 最基本的模型调用方式
- **异步调用**: 提升并发性能的异步调用
- **流式输出**: 实时响应的流式调用
- **推理模型**: 支持思考过程的推理模型调用

### 🔧 中级功能集成案例 (intermediate/)

展示HarborAI的特色功能：
- **结构化输出**: Agently vs Native两种解析方式
- **多模型切换**: 在不同模型间无缝切换
- **成本追踪**: 实时监控API调用成本
- **日志监控**: 全链路日志记录与分析，包含真实模型调用演示

### ⚡ 高级功能组合应用案例 (advanced/)

展示生产级特性：
- **容错重试**: 指数退避重试机制
- **降级策略**: 自动模型/厂商降级
- **批量处理**: 高效的批量调用处理
- **性能优化**: 缓存、连接池等优化技术
- **日志分析**: 高级日志分析工具，支持性能监控、错误模式识别和交互式浏览

### 🎯 跨场景综合应用案例 (scenarios/)

真实业务场景的完整解决方案：
- **聊天机器人**: 智能对话系统
- **内容生成系统**: 自动化内容创作
- **数据分析助手**: 智能数据洞察
- **企业级应用集成**: 生产环境部署方案

## 🔑 核心特性展示

### OpenAI 兼容接口
所有案例都使用与OpenAI SDK完全一致的调用方式：
```python
from harborai import HarborAI

client = HarborAI(api_key="your_api_key", base_url="your_base_url")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 插件化架构
支持多个模型厂商的无缝切换（推荐使用 DeepSeek）：
- **DeepSeek** (deepseek-chat, deepseek-reasoner) - 推荐
- OpenAI (gpt-4, gpt-3.5-turbo)
- Anthropic (claude-3)
- 豆包 (doubao-pro)
- 文心一言 (ernie-4.0-turbo)

### 可观测性
全面的监控和日志功能：
- Trace ID 全链路追踪
- 成本统计和预算控制
- 性能指标监控
- 异步日志记录

## 📊 日志存储与查看功能

HarborAI 提供了完整的日志存储和查看解决方案，支持多层级的日志管理：

### 🔰 基础日志功能 (basic/log_basics.py)
- **日志生成**: 创建测试日志数据，模拟真实API调用
- **日志查看**: 展示日志文件位置和内容结构
- **基础命令**: 演示 `view_logs.py` 的基本使用方法

```bash
# 运行基础日志演示
python basic/log_basics.py

# 可选参数
python basic/log_basics.py --demo-type full    # 完整演示
python basic/log_basics.py --demo-type create  # 仅创建日志
python basic/log_basics.py --demo-type view    # 仅查看日志
```

### 🔧 中级日志监控 (intermediate/logging_monitoring.py)
- **真实模型调用**: 使用真实API进行模型调用并记录日志
- **结构化日志**: SQLite数据库存储，支持复杂查询
- **性能监控**: 响应时间、成功率、错误率统计
- **实时告警**: 基于规则的告警系统
- **监控面板**: 可视化监控数据展示

```bash
# 运行完整日志监控演示
python intermediate/logging_monitoring.py

# 可选参数
python intermediate/logging_monitoring.py --test-only     # 仅测试模型调用
python intermediate/logging_monitoring.py --monitor-only  # 仅运行监控功能
python intermediate/logging_monitoring.py --model deepseek-chat  # 测试特定模型
```

### ⚡ 高级日志分析 (advanced/log_analysis.py)
- **多维度分析**: 性能趋势、错误模式、使用模式分析
- **交互式浏览**: 命令行交互式日志浏览器
- **自定义报告**: 生成详细的分析报告
- **数据导出**: 支持JSON格式的数据导出
- **统计可视化**: 丰富的统计信息展示

```bash
# 交互式日志浏览器
python advanced/log_analysis.py --interactive

# 运行演示套件
python advanced/log_analysis.py --demo

# 生成性能分析报告
python advanced/log_analysis.py --performance 7

# 生成错误分析报告
python advanced/log_analysis.py --errors 7

# 导出综合报告
python advanced/log_analysis.py --report my_report.json --days 7
```

### 📋 日志查看命令参考

HarborAI 提供了强大的 `view_logs.py` 工具，支持多种查看方式：

```bash
# 基础查看
python view_logs.py                           # 查看所有日志
python view_logs.py --limit 10                # 限制显示条数
python view_logs.py --days 7                  # 查看最近7天

# 按类型过滤
python view_logs.py --type all                # 所有类型（默认）
python view_logs.py --type request            # 仅请求日志
python view_logs.py --type response           # 仅响应日志
python view_logs.py --type paired             # 配对显示请求-响应

# 按模型过滤
python view_logs.py --model deepseek-chat     # 特定模型
python view_logs.py --model-pattern "deepseek*"  # 模型名称模式

# 统计信息
python view_logs.py --stats                   # 显示统计信息
python view_logs.py --stats --format json     # JSON格式统计

# 输出格式
python view_logs.py --format table            # 表格格式（默认）
python view_logs.py --format json             # JSON格式
python view_logs.py --format csv              # CSV格式
```

## 📖 使用指南

### 环境变量配置
```bash
# 必需的API密钥（推荐使用 DeepSeek）
DEEPSEEK_API_KEY=sk-xxx  # 推荐：获取地址 https://platform.deepseek.com/api_keys
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 其他服务提供商（可选）
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx

# 可选的基础URL
OPENAI_BASE_URL=https://api.openai.com/v1

# 日志配置
HARBORAI_LOG_LEVEL=INFO
HARBORAI_ENABLE_COST_TRACKING=true
HARBORAI_ENABLE_ASYNC_LOGGING=true
```

### 错误处理
所有案例都包含完整的错误处理：
```python
try:
    response = client.chat.completions.create(...)
except Exception as e:
    print(f"调用失败: {e}")
    # 具体的错误处理逻辑
```

## 🤝 贡献指南

欢迎提交新的应用案例！请确保：
1. 代码风格符合项目规范
2. 包含完整的文档说明
3. 提供预期输出示例
4. 说明实际应用价值

## 📄 许可证

本项目遵循 MIT 许可证。详见 [LICENSE](../LICENSE) 文件。

---

**注意**: 运行案例前请确保已正确配置API密钥，并且网络连接正常。某些案例可能需要特定的模型访问权限。