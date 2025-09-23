# Agently HTTP 401 错误修复报告

## 问题描述

在使用 Agently 进行结构化输出时，遇到 HTTP 401 Unauthorized 错误。错误表现为：
- Agently 配置了 DeepSeek API 密钥，但仍然向 OpenAI 端点发送请求
- 请求日志显示：`HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 401 Unauthorized"`
- Agently 解析失败后回退到原生解析方式

## 根因分析

### 1. 配置方式错误

**问题代码**（修复前）：
```python
# 错误的配置方式
agent.set_settings("model.OpenAICompatible", {
    "base_url": base_url,
    "model": model_name,
    "model_type": "OpenAI",
    "api_key": api_key,
    "request_options": {
        "temperature": 0.1,
    }
})
```

**问题分析**：
- 使用了 `agent.set_settings("model.OpenAICompatible", ...)` 的实例级配置
- 配置格式不符合 Agently 的标准规范
- `model_type` 设置为 "OpenAI" 而不是 "chat"
- API 密钥直接传递而不是通过 `auth` 对象

### 2. 配置未生效

由于配置方式错误，Agently 内部仍然使用默认的 OpenAI 配置，导致：
- 请求发送到 `https://api.openai.com/v1/chat/completions`
- 使用错误的认证信息
- 返回 HTTP 401 错误

## 修复方案

### 1. 采用正确的全局配置方式

**修复代码**：
```python
def _configure_agently_model(self, agent):
    """配置Agently使用HarborAI的模型设置。"""
    try:
        import os
        import agently
        
        # 获取DeepSeek API密钥
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise StructuredOutputError("DEEPSEEK_API_KEY not configured")
        
        # 配置DeepSeek模型
        base_url = "https://api.deepseek.com/v1"
        model_name = "deepseek-chat"
        
        # 使用正确的Agently配置方式
        openai_compatible_config = {
            "base_url": base_url,
            "model": model_name,
            "model_type": "chat",
            "auth": {"api_key": api_key},
            "request_options": {
                "temperature": 0.1,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            }
        }
        
        # 设置全局Agently配置
        agently.Agently.set_settings("OpenAICompatible", openai_compatible_config)
        
        self.logger.info(f"Agently configured with DeepSeek model: {model_name}")
        
    except Exception as e:
        self.logger.error(f"Failed to configure Agently model: {e}")
        if "DEEPSEEK_API_KEY" in str(e):
            raise
        else:
            raise StructuredOutputError(f"Failed to configure Agently model: {e}")
```

### 2. 关键修复点

1. **全局配置**：使用 `agently.Agently.set_settings()` 而不是实例级配置
2. **配置格式**：采用标准的 OpenAI 兼容配置格式
3. **认证方式**：使用 `"auth": {"api_key": api_key}` 格式
4. **模型类型**：设置为 `"chat"` 而不是 `"OpenAI"`
5. **完整参数**：添加完整的 `request_options` 配置

## 验证结果

### 修复前
```
HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 401 Unauthorized"
```

### 修复后
```
HTTP Request: POST https://api.deepseek.com/v1/chat/completions "HTTP/1.1 200 OK"
```

### 测试结果

运行 `test_deepseek_structured.py` 测试脚本：

```
📊 总体统计:
   总测试数: 12
   成功: 12 (100.0%)
   失败: 0 (0.0%)

📈 方法性能对比:
   native:
     成功率: 100.0%
     平均耗时: 0.021秒
   agently:
     成功率: 100.0%
     平均耗时: 5.402秒
   native_streaming:
     成功率: 100.0%
     平均耗时: 1.176秒
   agently_streaming:
     成功率: 100.0%
     平均耗时: 1.174秒
```

## 总结

1. **问题已完全解决**：HTTP 401 错误不再出现
2. **功能正常**：Agently 的流式和非流式结构化输出均正常工作
3. **性能表现**：虽然 Agently 非流式输出耗时较长，但功能完全正常
4. **配置标准化**：采用了正确的 Agently 配置方式，符合官方规范

## 经验教训

1. **配置方式很重要**：不同的配置方式会导致完全不同的行为
2. **全局 vs 实例配置**：某些情况下需要使用全局配置才能生效
3. **配置格式标准化**：严格按照官方文档的配置格式进行设置
4. **深度排查**：表面的错误信息可能掩盖了更深层的配置问题

## 相关文件

- 修复文件：`harborai/api/structured.py`
- 测试脚本：`test_deepseek_structured.py`
- 配置文件：`.env`（包含 DEEPSEEK_API_KEY）

---

**修复时间**：2025-09-24  
**修复状态**：✅ 完成  
**验证状态**：✅ 通过所有测试