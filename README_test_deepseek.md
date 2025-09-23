# DeepSeek 结构化输出测试脚本使用说明

## 概述

`test_deepseek_structured.py` 是一个综合测试脚本，用于验证 HarborAI 框架中 DeepSeek 模型的结构化输出功能。

## 功能特性

### 测试内容
1. **非流式结构化输出测试**
   - 原生方式（使用 DeepSeek API 的 response_format）
   - Agently 方式（使用 Agently 框架进行后处理）

2. **流式结构化输出测试**
   - 原生流式输出
   - Agently 流式输出

3. **多种复杂度的 JSON Schema**
   - 简单 Schema：基本字段（姓名、年龄）
   - 中等 Schema：嵌套对象和数组
   - 复杂 Schema：深层嵌套和复杂类型

4. **性能对比和结果分析**
   - 执行时间对比
   - 成功率统计
   - 结果一致性验证

5. **错误处理测试**
   - 无效 JSON Schema 处理
   - 无效 JSON 响应处理

## 配置要求

### 环境变量配置

测试脚本会自动从项目根目录的 `.env` 文件中读取 DeepSeek API 密钥。请确保在 `.env` 文件中配置：

```env
DEEPSEEK_API_KEY=sk-your-actual-deepseek-api-key-here
```

**备用配置方法：**

**Windows 系统：**
```cmd
set DEEPSEEK_API_KEY=your_actual_deepseek_api_key_here
```

**Linux/Mac 系统：**
```bash
export DEEPSEEK_API_KEY=your_actual_deepseek_api_key_here
```

**注意：** 如果未配置 API 密钥，测试将使用模拟数据进行功能验证。

## 使用方法

### 基本运行
```bash
python test_deepseek_structured.py
```

### 测试输出示例

```
🚀 开始DeepSeek结构化输出功能测试
⏰ 测试开始时间: 2025-09-24 00:38:18
⚠️  未配置DEEPSEEK_API_KEY，将使用模拟数据进行测试

================================================================================
================================== 非流式结构化输出测试 ==================================
================================================================================

🧪 测试Schema: simple
📝 提示词: 请生成一个虚构人物的基本信息，包括姓名和年龄。

📋 测试: simple_非流式_原生
🔧 方法: native
📊 状态: ✅ 成功
⏱️  耗时: 0.02秒
📄 结果: {
  "name": "张三",
  "age": 25
}
```

## 测试结果解读

### 成功指标
- ✅ **成功**：测试通过，返回了符合 Schema 的结构化数据
- ❌ **失败**：测试失败，通常由于 API 配置问题或网络错误

### 性能指标
- **耗时**：单次测试的执行时间
- **成功率**：该方法的整体成功率
- **结果一致性**：原生方式和 Agently 方式的输出是否一致

### 常见测试结果

1. **API 密钥未配置**
   ```
   ❌ 失败的测试:
      simple_流式_Agently: [STRUCTURED_OUTPUT_ERROR] DEEPSEEK_API_KEY not configured
   ```
   - 解决方案：配置 DEEPSEEK_API_KEY 环境变量

2. **网络连接问题**
   ```
   ❌ 失败的测试:
      simple_非流式_原生: Connection timeout
   ```
   - 解决方案：检查网络连接和 API 服务状态

3. **Schema 验证错误**
   ```
   ✅ 正确处理无效Schema: [STRUCTURED_OUTPUT_ERROR] Failed to parse response
   ```
   - 这是正常的错误处理测试，表明框架正确处理了无效输入

## 扩展使用

### 添加自定义测试用例

可以在脚本中的 `test_schemas` 字典中添加新的 JSON Schema：

```python
self.test_schemas["custom"] = {
    "type": "object",
    "properties": {
        "custom_field": {"type": "string"}
    },
    "required": ["custom_field"]
}
```

### 修改测试提示词

在 `test_prompts` 字典中添加对应的提示词：

```python
self.test_prompts["custom"] = "请生成自定义数据"
```

## 故障排除

### 常见问题

1. **导入错误**
   ```
   ImportError: cannot import name 'HarborAI' from 'harborai.api.client'
   ```
   - 确保 HarborAI 框架已正确安装
   - 检查 Python 路径配置

2. **Agently 相关错误**
   ```
   [STRUCTURED_OUTPUT_ERROR] DEEPSEEK_API_KEY not configured
   ```
   - 配置正确的 API 密钥
   - 确保 Agently 框架已安装

3. **JSON Schema 验证失败**
   - 检查 Schema 格式是否正确
   - 确保必需字段已定义

## 技术细节

### 测试架构

- **DeepSeekStructuredTester**: 主测试类
- **模拟数据生成**: 当 API 密钥未配置时使用
- **异步支持**: 支持异步 API 调用测试
- **错误处理**: 完整的异常捕获和处理机制

### 依赖项

- `harborai`: HarborAI 框架
- `asyncio`: 异步编程支持
- `json`: JSON 数据处理
- `time`: 性能计时
- `structlog`: 结构化日志

## 贡献

如需添加新的测试用例或改进测试脚本，请：

1. 遵循现有的代码风格
2. 添加适当的错误处理
3. 更新相关文档
4. 确保测试的可重复性

---

**注意**: 此测试脚本仅用于验证 HarborAI 框架的结构化输出功能，不应用于生产环境的性能基准测试。