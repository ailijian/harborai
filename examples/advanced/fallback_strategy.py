#!/usr/bin/env python3
"""
降级策略演示

这个示例展示了 HarborAI 的内置降级策略功能，包括：
1. 内置多层级降级策略
2. 自动故障转移
3. 模型间切换
4. 结构化输出的降级
5. 推理模型的降级处理

场景：
- 主要AI服务不可用或性能下降时自动切换到备用方案
- 确保服务连续性和可用性
- 在成本和性能之间找到平衡

价值：
- 使用 HarborAI 内置的降级机制，无需自己实现
- 确保服务连续性和可用性
- 优化用户体验，避免服务中断
- 智能选择最优服务
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
def get_client_configs():
    """获取多个客户端配置"""
    return {
        'deepseek': {
            'api_key': os.getenv('DEEPSEEK_API_KEY'),
            'base_url': os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        },
        'ernie': {
            'api_key': os.getenv('ERNIE_API_KEY'),
            'base_url': os.getenv('ERNIE_BASE_URL', 'https://aip.baidubce.com')
        },
        'doubao': {
            'api_key': os.getenv('DOUBAO_API_KEY'),
            'base_url': os.getenv('DOUBAO_BASE_URL', 'https://ark.cn-beijing.volces.com')
        }
    }

def get_primary_client():
    """获取主要客户端"""
    configs = get_client_configs()
    
    # 优先使用 DeepSeek
    if configs['deepseek']['api_key']:
        return HarborAI(
            api_key=configs['deepseek']['api_key'],
            base_url=configs['deepseek']['base_url']
        ), "deepseek-chat"
    
    # 其次使用 Ernie
    if configs['ernie']['api_key']:
        return HarborAI(
            api_key=configs['ernie']['api_key'],
            base_url=configs['ernie']['base_url']
        ), "ernie-3.5-8k"
    
    # 最后使用 Doubao
    if configs['doubao']['api_key']:
        return HarborAI(
            api_key=configs['doubao']['api_key'],
            base_url=configs['doubao']['base_url']
        ), "doubao-1-5-pro-32k-character-250715"
    
    return None, None

async def demo_basic_fallback():
    """演示基本的降级策略"""
    print("\n🔄 演示基本降级策略")
    print("=" * 50)
    
    client, primary_model = get_primary_client()
    if not client:
        print("❌ 请至少设置一个 API Key (DEEPSEEK_API_KEY, ERNIE_API_KEY, 或 DOUBAO_API_KEY)")
        return
    
    messages = [
        {"role": "user", "content": "请简单介绍一下机器学习"}
    ]
    
    # 定义降级模型列表
    fallback_models = ["deepseek-chat", "ernie-3.5-8k", "doubao-1-5-pro-32k-character-250715"]
    
    try:
        # 使用 HarborAI 内置的降级机制
        response = await client.chat.completions.create(
            model=primary_model,
            messages=messages,
            fallback=fallback_models,  # 内置降级策略
            retry_policy={
                "max_attempts": 2,
                "base_delay": 1.0,
                "max_delay": 5.0
            },
            timeout=30.0
        )
        
        print(f"✅ 调用成功")
        print(f"🎯 使用模型: {primary_model}")
        print(f"📝 响应: {response.choices[0].message.content[:100]}...")
        print(f"🔢 Token 使用: {response.usage.total_tokens if response.usage else 'N/A'}")
        
    except Exception as e:
        print(f"❌ 所有降级选项都失败: {e}")

async def demo_structured_output_fallback():
    """演示结构化输出的降级策略"""
    print("\n📊 演示结构化输出降级")
    print("=" * 50)
    
    client, primary_model = get_primary_client()
    if not client:
        print("❌ 请至少设置一个 API Key")
        return
    
    messages = [
        {"role": "user", "content": "分析一下特斯拉公司的商业模式"}
    ]
    
    # 定义结构化输出 schema
    schema = {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "business_model": {"type": "string"},
            "revenue_streams": {
                "type": "array",
                "items": {"type": "string"}
            },
            "competitive_advantages": {
                "type": "array",
                "items": {"type": "string"}
            },
            "risks": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["company", "business_model", "revenue_streams"],
        "additionalProperties": False
    }
    
    try:
        response = await client.chat.completions.create(
            model=primary_model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "BusinessAnalysis",
                    "schema": schema,
                    "strict": True
                }
            },
            fallback=["deepseek-chat", "ernie-3.5-8k"],  # 降级策略
            retry_policy={
                "max_attempts": 2,
                "base_delay": 1.0,
                "max_delay": 10.0
            },
            timeout=45.0
        )
        
        print(f"✅ 结构化输出成功")
        print(f"🎯 使用模型: {primary_model}")
        print(f"📊 解析结果: {response.parsed}")
        
    except Exception as e:
        print(f"❌ 结构化输出降级失败: {e}")

async def demo_reasoning_model_fallback():
    """演示推理模型的降级策略"""
    print("\n🧠 演示推理模型降级")
    print("=" * 50)
    
    client, _ = get_primary_client()
    if not client:
        print("❌ 请至少设置一个 API Key")
        return
    
    messages = [
        {"role": "user", "content": "请分析区块链技术的发展前景和挑战"}
    ]
    
    try:
        # 尝试使用推理模型，失败时降级到普通模型
        response = await client.chat.completions.create(
            model="deepseek-reasoner",  # 主要使用推理模型
            messages=messages,
            fallback=["deepseek-chat", "ernie-3.5-8k"],  # 降级到普通模型
            retry_policy={
                "max_attempts": 2,
                "base_delay": 2.0,
                "max_delay": 15.0
            },
            timeout=90.0
        )
        
        print(f"✅ 推理模型调用成功")
        print(f"💭 最终答案: {response.choices[0].message.content[:150]}...")
        
        # 检查是否有思考过程
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning = response.choices[0].message.reasoning_content
            print(f"🤔 思考过程: {reasoning[:100] if reasoning else 'N/A'}...")
        else:
            print("💡 使用了普通模型（无思考过程）")
        
    except Exception as e:
        print(f"❌ 推理模型降级失败: {e}")

async def demo_stream_fallback():
    """演示流式调用的降级策略"""
    print("\n🌊 演示流式调用降级")
    print("=" * 50)
    
    client, primary_model = get_primary_client()
    if not client:
        print("❌ 请至少设置一个 API Key")
        return
    
    messages = [
        {"role": "user", "content": "请详细解释深度学习的工作原理"}
    ]
    
    try:
        print("📡 开始流式响应:")
        
        stream = await client.chat.completions.create(
            model=primary_model,
            messages=messages,
            stream=True,
            fallback=["deepseek-chat", "ernie-3.5-8k"],  # 流式调用也支持降级
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
        print(f"🎯 使用模型: {primary_model}")
        
    except Exception as e:
        print(f"❌ 流式调用降级失败: {e}")

async def demo_cost_aware_fallback():
    """演示成本感知的降级策略"""
    print("\n💰 演示成本感知降级")
    print("=" * 50)
    
    client, _ = get_primary_client()
    if not client:
        print("❌ 请至少设置一个 API Key")
        return
    
    messages = [
        {"role": "user", "content": "请写一首关于春天的诗"}
    ]
    
    # 按成本从高到低排列的模型（示例）
    cost_ordered_models = [
        "ernie-4.0-turbo-8k",  # 高性能高成本
        "deepseek-chat",       # 中等性能中等成本
        "ernie-3.5-8k"         # 基础性能低成本
    ]
    
    try:
        start_time = time.time()
        
        response = await client.chat.completions.create(
            model=cost_ordered_models[0],  # 优先使用高性能模型
            messages=messages,
            fallback=cost_ordered_models[1:],  # 按成本降级
            retry_policy={
                "max_attempts": 2,
                "base_delay": 1.0,
                "max_delay": 8.0
            },
            timeout=30.0,
            cost_tracking=True  # 启用成本追踪
        )
        
        elapsed = time.time() - start_time
        
        print(f"✅ 成本感知调用成功")
        print(f"⏱️ 响应时间: {elapsed:.2f}秒")
        print(f"📝 响应: {response.choices[0].message.content[:100]}...")
        print(f"🔢 Token 使用: {response.usage.total_tokens if response.usage else 'N/A'}")
        
    except Exception as e:
        print(f"❌ 成本感知降级失败: {e}")

async def demo_intelligent_fallback():
    """演示智能降级策略"""
    print("\n🤖 演示智能降级策略")
    print("=" * 50)
    
    client, primary_model = get_primary_client()
    if not client:
        print("❌ 请至少设置一个 API Key")
        return
    
    # 不同类型的任务使用不同的降级策略
    tasks = [
        {
            "name": "创意写作",
            "messages": [{"role": "user", "content": "写一个科幻小说的开头"}],
            "fallback": ["deepseek-chat", "ernie-4.0-turbo-8k"],  # 创意任务优先高质量模型
            "timeout": 45.0
        },
        {
            "name": "简单问答",
            "messages": [{"role": "user", "content": "什么是人工智能？"}],
            "fallback": ["ernie-3.5-8k", "deepseek-chat"],  # 简单任务优先低成本模型
            "timeout": 20.0
        },
        {
            "name": "技术分析",
            "messages": [{"role": "user", "content": "分析Python和Java的性能差异"}],
            "fallback": ["deepseek-chat", "ernie-4.0-turbo-8k"],  # 技术任务需要专业模型
            "timeout": 60.0
        }
    ]
    
    for task in tasks:
        print(f"\n🎯 任务: {task['name']}")
        try:
            start_time = time.time()
            
            response = await client.chat.completions.create(
                model=primary_model,
                messages=task['messages'],
                fallback=task['fallback'],
                retry_policy={
                    "max_attempts": 2,
                    "base_delay": 1.0,
                    "max_delay": 10.0
                },
                timeout=task['timeout']
            )
            
            elapsed = time.time() - start_time
            
            print(f"   ✅ 成功 - 耗时: {elapsed:.2f}秒")
            print(f"   📝 响应: {response.choices[0].message.content[:80]}...")
            
        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:50]}...")

async def demo_fallback_with_different_providers():
    """演示跨厂商降级策略"""
    print("\n🌐 演示跨厂商降级")
    print("=" * 50)
    
    configs = get_client_configs()
    available_providers = []
    
    # 检查可用的厂商
    for provider, config in configs.items():
        if config['api_key']:
            available_providers.append(provider)
    
    if len(available_providers) < 2:
        print("❌ 需要至少配置两个厂商的 API Key 才能演示跨厂商降级")
        print("请设置 DEEPSEEK_API_KEY, ERNIE_API_KEY, 或 DOUBAO_API_KEY")
        return
    
    print(f"🔍 检测到可用厂商: {', '.join(available_providers)}")
    
    # 使用第一个可用的厂商作为主要客户端
    primary_config = configs[available_providers[0]]
    client = HarborAI(
        api_key=primary_config['api_key'],
        base_url=primary_config['base_url']
    )
    
    messages = [
        {"role": "user", "content": "请解释云计算的基本概念"}
    ]
    
    # 构建跨厂商降级策略
    provider_models = {
        'deepseek': 'deepseek-chat',
        'ernie': 'ernie-3.5-8k',
        'doubao': 'doubao-1-5-pro-32k-character-250715'
    }
    
    fallback_models = [provider_models[provider] for provider in available_providers[1:]]
    
    try:
        response = await client.chat.completions.create(
            model=provider_models[available_providers[0]],
            messages=messages,
            fallback=fallback_models,  # 跨厂商降级
            retry_policy={
                "max_attempts": 2,
                "base_delay": 1.0,
                "max_delay": 8.0
            },
            timeout=30.0
        )
        
        print(f"✅ 跨厂商调用成功")
        print(f"🎯 主要厂商: {available_providers[0]}")
        print(f"🔄 降级选项: {', '.join(available_providers[1:])}")
        print(f"📝 响应: {response.choices[0].message.content[:100]}...")
        
    except Exception as e:
        print(f"❌ 跨厂商降级失败: {e}")

async def main():
    """主函数"""
    print("🔄 HarborAI 降级策略演示")
    print("=" * 60)
    
    # 检查环境变量
    configs = get_client_configs()
    available_keys = [k for k, v in configs.items() if v['api_key']]
    
    if not available_keys:
        print("⚠️ 警告: 未设置任何 API Key")
        print("请设置 DEEPSEEK_API_KEY, ERNIE_API_KEY, 或 DOUBAO_API_KEY")
        return
    
    print(f"🔍 检测到可用配置: {', '.join(available_keys)}")
    
    demos = [
        ("基本降级策略", demo_basic_fallback),
        ("结构化输出降级", demo_structured_output_fallback),
        ("推理模型降级", demo_reasoning_model_fallback),
        ("流式调用降级", demo_stream_fallback),
        ("成本感知降级", demo_cost_aware_fallback),
        ("智能降级策略", demo_intelligent_fallback),
        ("跨厂商降级", demo_fallback_with_different_providers)
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
            await asyncio.sleep(1)  # 避免请求过于频繁
        except Exception as e:
            print(f"❌ {name} 演示失败: {e}")
    
    print("\n🎉 降级策略演示完成！")
    print("\n💡 关键要点:")
    print("1. 使用 fallback 参数配置降级模型列表")
    print("2. 支持结构化输出和推理模型的降级")
    print("3. 流式调用也支持降级机制")
    print("4. 可以根据任务类型选择不同的降级策略")
    print("5. 支持跨厂商的降级策略")
    print("6. 结合 retry_policy 实现更强的容错能力")

if __name__ == "__main__":
    asyncio.run(main())