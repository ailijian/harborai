# HarborAI 高级功能示例

本目录包含 HarborAI 的高级功能演示，展示了如何使用 HarborAI 的核心特性来构建健壮、高性能的 AI 应用。

## 🎯 核心设计原则

所有示例都严格遵循 HarborAI 的设计规范：

### 1. 统一 OpenAI 风格接口
- 使用标准的 `client.chat.completions.create(...)` 调用方式
- 兼容 OpenAI API 的参数和响应格式
- 支持同步和异步调用模式

### 2. 内置容错机制
- **重试策略**：通过 `retry_policy` 参数配置
- **降级策略**：通过 `fallback` 参数配置多模型降级
- **超时控制**：通过 `timeout` 参数设置请求超时

### 3. 结构化输出支持
- 使用 `response_format` 参数定义输出格式
- 默认支持 Agently 语法解析
- 自动处理 JSON Schema 验证

### 4. 推理模型支持
- 自动检测和处理 `reasoning_content` 字段
- 支持 o1 系列等推理模型的特殊处理
- 透明的推理过程展示

### 5. 流式调用支持
- 通过 `stream=True` 启用流式响应
- 支持实时内容生成和显示
- 兼容结构化输出和推理模型

## 📁 示例文件说明

### 🛡️ fault_tolerance.py - 容错与重试机制
**核心功能**：
- 演示 HarborAI 内置的重试机制
- 展示网络错误、限流等异常的自动处理
- 支持结构化输出和推理模型的容错

**应用价值**：
- 提高应用的稳定性和可靠性
- 减少因网络波动导致的调用失败
- 自动处理 API 限流和临时故障

**关键特性**：
```python
# 使用内置重试机制
response = await client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
    retry_policy={
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 5.0
    },
    timeout=30.0
)
```

### 🔄 fallback_strategy.py - 降级策略
**核心功能**：
- 演示多模型降级策略
- 支持跨厂商的模型切换
- 智能选择最佳可用模型

**应用价值**：
- 确保服务的高可用性
- 优化成本和性能平衡
- 应对单一模型的服务中断

**关键特性**：
```python
# 配置降级模型列表
response = await client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}],
    fallback=["ernie-3.5-8k", "doubao-1-5-pro-32k"],
    retry_policy={"max_attempts": 2}
)
```

### ⚡ batch_processing.py - 批量处理优化
**核心功能**：
- 演示高效的批量请求处理
- 使用原生异步支持提升并发性能
- 支持批量结构化输出和流式处理

**应用价值**：
- 大幅提升批量任务的处理效率
- 优化资源利用率和响应时间
- 支持大规模数据处理场景

**关键特性**：
```python
# 原生异步批量处理
async def process_batch(tasks):
    async with asyncio.TaskGroup() as tg:
        tasks = [

### 📊 log_analysis.py - 高级日志分析工具
**核心功能**：
- 多维度日志分析（性能、错误、使用模式）
- 交互式日志浏览器
- 自定义报告生成和数据导出
- 统计可视化和趋势分析

**应用价值**：
- 深入了解系统运行状况
- 快速定位性能瓶颈和错误模式
- 支持数据驱动的优化决策
- 提供完整的可观测性解决方案

**关键特性**：
```python
# 交互式日志浏览
python log_analysis.py --interactive

# 生成性能分析报告
python log_analysis.py --performance 7

# 导出综合分析报告
python log_analysis.py --report analysis.json --days 7
```

**功能模块**：
- **性能趋势分析**: 响应时间统计、P95分位数、模型性能对比
- **错误模式识别**: 错误类型分布、成功率统计、常见错误分析
- **使用模式分析**: 峰值时间识别、模型流行度、请求大小分布
- **交互式浏览**: 命令行界面、实时查询、关键词搜索
- **报告生成**: JSON格式导出、自定义时间范围、多维度统计
            tg.create_task(client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": task}],
                timeout=30.0
            ))
            for task in tasks
        ]
    return [task.result() for task in tasks]
```

### 🚀 performance_optimization.py - 性能优化
**核心功能**：
- 演示 HarborAI 的性能优化特性
- 展示缓存、连接池等优化技术
- 提供性能监控和分析工具

**应用价值**：
- 显著提升应用响应速度
- 降低 API 调用成本
- 优化用户体验

**关键特性**：
```python
# 启用缓存和性能优化
client = HarborAI(
    api_key="your_api_key",
    enable_cache=True,
    cache_ttl=3600,
    connection_pool_size=10
)
```

### ⚙️ config_helper.py - 配置管理
**核心功能**：
- 演示多供应商配置管理
- 支持环境变量和动态配置
- 提供配置验证和最佳实践

**应用价值**：
- 简化多环境部署配置
- 提高配置安全性
- 支持动态配置切换

**关键特性**：
```python
# 智能配置管理
config_manager = ConfigManager()
client, config = config_manager.create_client()

# 环境感知配置
current_env = os.getenv("HARBORAI_ENV", "development")
```

## 🚀 快速开始

### 1. 环境准备
```bash
# 安装依赖
pip install harborai

# 设置环境变量
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export ERNIE_API_KEY="your_ernie_api_key"
export DOUBAO_API_KEY="your_doubao_api_key"
```

### 2. 运行示例
```bash
# 设置 PYTHONPATH（如果使用本地源码）
export PYTHONPATH="/path/to/harborai"

# 运行容错示例
python fault_tolerance.py

# 运行降级策略示例
python fallback_strategy.py

# 运行批量处理示例
python batch_processing.py

# 运行性能优化示例
python performance_optimization.py

# 运行配置管理示例
python config_helper.py
```

## 📋 代码结构

每个示例文件都遵循统一的结构：

```python
#!/usr/bin/env python3
"""
示例说明文档
- 功能描述
- 应用场景
- 核心价值
"""

import asyncio
from harborai import HarborAI  # 统一导入方式

async def demo_basic_feature():
    """基础功能演示"""
    client = HarborAI(api_key="your_api_key")
    
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "你好"}],
        # HarborAI 特有参数
        retry_policy={"max_attempts": 3},
        fallback=["ernie-3.5-8k"],
        timeout=30.0
    )
    
    return response

async def demo_structured_output():
    """结构化输出演示"""
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "分析这个问题"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "confidence": {"type": "number"}
                    }
                }
            }
        }
    )

async def demo_reasoning_model():
    """推理模型演示"""
    response = await client.chat.completions.create(
        model="deepseek-reasoner",  # 推理模型
        messages=[{"role": "user", "content": "复杂推理问题"}]
    )
    
    # 自动处理 reasoning_content
    if response.choices[0].message.reasoning_content:
        print("推理过程:", response.choices[0].message.reasoning_content)

async def demo_streaming():
    """流式调用演示"""
    stream = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "长文本生成"}],
        stream=True
    )
    
    async for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔧 配置示例

### .env 配置文件
```bash
# HarborAI 环境配置
HARBORAI_ENV=production

# DeepSeek 配置
DEEPSEEK_API_KEY=your_deepseek_api_key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 百度文心一言配置
ERNIE_API_KEY=your_ernie_api_key
ERNIE_BASE_URL=https://aip.baidubce.com

# 字节跳动豆包配置
DOUBAO_API_KEY=your_doubao_api_key
DOUBAO_BASE_URL=https://ark.cn-beijing.volces.com

# OpenAI 配置（可选）
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
```

## 📊 预期输出

运行示例后，您将看到：

### 容错机制输出
```
🛡️ 演示基础重试机制
==================================================
✅ 基础重试测试成功
   响应: 你好！我是一个AI助手...
   Token使用: 25

🔧 演示结构化输出的重试
==================================================
✅ 结构化输出重试成功
   分析结果: {'summary': '这是一个测试分析', 'confidence': 0.95}
```

### 降级策略输出
```
🔄 演示基础降级策略
==================================================
🎯 主模型: deepseek-chat
🔄 降级模型: ['ernie-3.5-8k', 'doubao-1-5-pro-32k']
✅ 降级策略测试成功
   响应: 云计算是一种通过互联网提供计算服务的模式...
```

### 批量处理输出
```
⚡ 演示基础批量处理
==================================================
📊 批量处理统计:
   任务数量: 5
   总耗时: 3.45秒
   平均耗时: 0.69秒/任务
   并发效率: 85%
```

## 💡 最佳实践

### 1. 错误处理
```python
try:
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "你好"}],
        retry_policy={"max_attempts": 3},
        timeout=30.0
    )
except Exception as e:
    logger.error(f"API 调用失败: {e}")
```

### 2. 配置管理
```python
# 使用环境变量
api_key = os.getenv("DEEPSEEK_API_KEY")
base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

client = HarborAI(api_key=api_key, base_url=base_url)
```

### 3. 性能优化
```python
# 启用缓存和连接池
client = HarborAI(
    api_key="your_api_key",
    enable_cache=True,
    cache_ttl=3600,
    connection_pool_size=10
)
```

### 4. 监控和日志
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 记录关键操作
logger.info(f"API 调用成功，Token 使用: {response.usage.total_tokens}")
```

## 🔍 故障排查

### 常见问题

1. **导入错误**
   ```bash
   ModuleNotFoundError: No module named 'harborai'
   ```
   **解决方案**：设置 PYTHONPATH 或安装 harborai 包

2. **API Key 未配置**
   ```bash
   ❌ 缺少环境变量: DEEPSEEK_API_KEY
   ```
   **解决方案**：设置相应的环境变量

3. **网络连接问题**
   ```bash
   ❌ 连接测试: 失败 - Connection timeout
   ```
   **解决方案**：检查网络连接和 API 地址配置

### 调试技巧

1. **启用详细日志**
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查配置状态**
   ```python
   config_manager = ConfigManager()
   config_manager.print_configuration_status()
   ```

3. **测试单个功能**
   ```python
   # 只运行特定演示函数
   await demo_basic_retry()
   ```

## 📚 相关文档

- [HarborAI 基础示例](../basic/README.md)
- [HarborAI 中级示例](../intermediate/README.md)
- [HarborAI 产品需求文档](../../.trae/documents/HarborAI_PRD.md)
- [HarborAI 技术设计文档](../../.trae/documents/HarborAI_TD.md)

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这些示例：

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](../../LICENSE) 文件。