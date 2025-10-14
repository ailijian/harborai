#!/usr/bin/env python3
"""
容错与重试机制演示

这个示例展示了 HarborAI 的内置容错与重试机制，包括：
1. 内置重试策略配置
2. 异常处理与分类
3. 超时控制
4. 结构化输出的容错
5. 推理模型的容错处理

场景：
- 网络不稳定、API服务偶发故障的生产环境
- 需要自动恢复和容错保障的关键业务
- 提升系统稳定性和可靠性

价值：
- 使用 HarborAI 内置的重试机制，无需自己实现
- 统一的异常处理和错误分类
- 生产环境必备的容错保障
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import os

# 正确的 HarborAI 导入方式
from harborai import HarborAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 从环境变量获取配置
def get_client_config():
    """获取客户端配置"""
    return {
        'api_key': os.getenv('DEEPSEEK_API_KEY'),
        'base_url': os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    }

async def demo_basic_retry():
    """演示基本的重试机制"""
    print("\n🔄 演示基本重试机制")
    print("=" * 50)
    
    config = get_client_config()
    if not config['api_key']:
        print("❌ 请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    # 创建 HarborAI 客户端
    client = HarborAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    
    messages = [
        {"role": "user", "content": "请简单介绍一下人工智能"}
    ]
    
    try:
        # 使用 HarborAI 内置的重试机制
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            retry_policy={
                "max_attempts": 3,
                "base_delay": 1.0,
                "max_delay": 10.0,
                "exponential_base": 2.0,
                "jitter": True
            },
            timeout=30.0
        )
        
        print(f"✅ 调用成功")
        print(f"📝 响应: {response.choices[0].message.content[:100]}...")
        print(f"🔢 Token 使用: {response.usage.total_tokens if response.usage else 'N/A'}")
        
    except Exception as e:
        print(f"❌ 调用失败: {e}")

async def demo_structured_output_retry():
    """演示结构化输出的重试机制"""
    print("\n📊 演示结构化输出重试")
    print("=" * 50)
    
    config = get_client_config()
    if not config['api_key']:
        print("❌ 请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    client = HarborAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    
    messages = [
        {"role": "user", "content": "分析一下苹果公司的优势和挑战"}
    ]
    
    # 定义结构化输出 schema
    schema = {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "advantages": {
                "type": "array",
                "items": {"type": "string"}
            },
            "challenges": {
                "type": "array", 
                "items": {"type": "string"}
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["company", "advantages", "challenges", "confidence"],
        "additionalProperties": False
    }
    
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "CompanyAnalysis",
                    "schema": schema,
                    "strict": True
                }
            },
            retry_policy={
                "max_attempts": 3,
                "base_delay": 1.0,
                "max_delay": 15.0
            },
            timeout=45.0
        )
        
        print(f"✅ 结构化输出成功")
        print(f"📊 解析结果: {response.parsed}")
        
    except Exception as e:
        print(f"❌ 结构化输出失败: {e}")

async def demo_reasoning_model_retry():
    """演示推理模型的重试机制"""
    print("\n🧠 演示推理模型重试")
    print("=" * 50)
    
    config = get_client_config()
    if not config['api_key']:
        print("❌ 请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    client = HarborAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    
    messages = [
        {"role": "user", "content": "请分析量子计算对现代密码学的影响"}
    ]
    
    try:
        response = await client.chat.completions.create(
            model="deepseek-reasoner",
            messages=messages,
            retry_policy={
                "max_attempts": 2,  # 推理模型通常响应较慢，减少重试次数
                "base_delay": 2.0,
                "max_delay": 30.0
            },
            timeout=120.0  # 推理模型需要更长的超时时间
        )
        
        print(f"✅ 推理模型调用成功")
        print(f"💭 最终答案: {response.choices[0].message.content[:150]}...")
        
        # 检查是否有思考过程
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning = response.choices[0].message.reasoning_content
            print(f"🤔 思考过程: {reasoning[:100] if reasoning else 'N/A'}...")
        
    except Exception as e:
        print(f"❌ 推理模型调用失败: {e}")

async def demo_stream_with_retry():
    """演示流式调用的重试机制"""
    print("\n🌊 演示流式调用重试")
    print("=" * 50)
    
    config = get_client_config()
    if not config['api_key']:
        print("❌ 请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    client = HarborAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    
    messages = [
        {"role": "user", "content": "请详细解释机器学习的基本概念"}
    ]
    
    try:
        print("📡 开始流式响应:")
        
        stream = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=True,
            retry_policy={
                "max_attempts": 2,
                "base_delay": 1.0,
                "max_delay": 8.0
            },
            timeout=60.0
        )
        
        content_parts = []
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                content_parts.append(content)
                print(content, end="", flush=True)
        
        print(f"\n✅ 流式调用完成，共接收 {len(content_parts)} 个片段")
        
    except Exception as e:
        print(f"❌ 流式调用失败: {e}")

async def demo_error_handling():
    """演示错误处理和分类"""
    print("\n⚠️ 演示错误处理")
    print("=" * 50)
    
    config = get_client_config()
    client = HarborAI(
        api_key="invalid_key",  # 故意使用无效的 API Key
        base_url=config.get('base_url', 'https://api.deepseek.com')
    )
    
    messages = [
        {"role": "user", "content": "测试错误处理"}
    ]
    
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            retry_policy={
                "max_attempts": 2,
                "base_delay": 0.5,
                "max_delay": 2.0
            },
            timeout=10.0
        )
        
        print("❌ 预期应该失败，但却成功了")
        
    except Exception as e:
        print(f"✅ 正确捕获错误: {type(e).__name__}")
        print(f"📝 错误信息: {str(e)[:100]}...")

async def demo_timeout_handling():
    """演示超时处理"""
    print("\n⏰ 演示超时处理")
    print("=" * 50)
    
    config = get_client_config()
    if not config['api_key']:
        print("❌ 请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    client = HarborAI(
        api_key=config['api_key'],
        base_url=config['base_url']
    )
    
    messages = [
        {"role": "user", "content": "请写一篇关于人工智能发展历史的详细文章，包含所有重要里程碑"}
    ]
    
    try:
        start_time = time.time()
        
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            retry_policy={
                "max_attempts": 1,  # 只尝试一次，专注测试超时
                "base_delay": 1.0
            },
            timeout=5.0  # 设置很短的超时时间
        )
        
        elapsed = time.time() - start_time
        print(f"✅ 调用完成，耗时: {elapsed:.2f}秒")
        print(f"📝 响应长度: {len(response.choices[0].message.content)} 字符")
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"⏰ 超时或其他错误，耗时: {elapsed:.2f}秒")
        print(f"📝 错误类型: {type(e).__name__}")

async def main():
    """主函数"""
    print("🛡️ HarborAI 容错与重试机制演示")
    print("=" * 60)
    
    # 检查环境变量
    if not os.getenv('DEEPSEEK_API_KEY'):
        print("⚠️ 警告: 未设置 DEEPSEEK_API_KEY 环境变量")
        print("部分演示可能无法正常运行")
    
    demos = [
        ("基本重试机制", demo_basic_retry),
        ("结构化输出重试", demo_structured_output_retry),
        ("推理模型重试", demo_reasoning_model_retry),
        ("流式调用重试", demo_stream_with_retry),
        ("错误处理", demo_error_handling),
        ("超时处理", demo_timeout_handling)
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
            await asyncio.sleep(1)  # 避免请求过于频繁
        except Exception as e:
            print(f"❌ {name} 演示失败: {e}")
    
    print("\n🎉 容错与重试机制演示完成！")
    print("\n💡 关键要点:")
    print("1. 使用 retry_policy 参数配置重试策略")
    print("2. 设置合适的 timeout 值")
    print("3. HarborAI 自动处理网络错误、限流等常见问题")
    print("4. 支持结构化输出和推理模型的容错")
    print("5. 流式调用也支持重试机制")

if __name__ == "__main__":
    asyncio.run(main())