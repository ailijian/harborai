#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用最新 Agently 版本的正确配置方法
基于 Agently 4.0.3.1 的官方示例
"""

import json
import traceback
import asyncio
from agently import Agently

def test_agently_structured_output():
    """
    测试 Agently 结构化输出的正确方法
    使用最新版本的配置方式
    """
    print("🚀 测试 Agently 结构化输出")
    print("使用最新版本 4.0.3.1 的配置方式")
    
    # 测试输入
    test_input = "今天天气真好，我很开心"
    
    # 期望的 JSON schema
    output_schema = {
        "sentiment": (str, "情感分析结果：positive, negative, neutral"),
        "confidence": (float, "置信度，0-1之间的数值")
    }
    
    print(f"📝 测试输入: {test_input}")
    print(f"📋 期望输出格式: {output_schema}")
    
    try:
        # 使用最新的全局配置方式
        print("\n" + "="*60)
        print("🔧 配置 Agently 全局设置")
        print("="*60)
        
        Agently.set_settings(
            "OpenAICompatible",
            {
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "model": "ep-20250509161856-ntmhj",
                "model_type": "chat",
                "auth": "6c39786b-2758-4dc3-8b88-a3e8b60d96b3",
            },
        )
        
        print("✅ 全局配置完成")
        
        # 验证配置
        print("\n🔍 验证配置:")
        settings = Agently.settings
        openai_settings = settings.get("plugins.ModelRequester.OpenAICompatible")
        if openai_settings:
            print(f"  - base_url: {openai_settings.get('base_url')}")
            print(f"  - model: {openai_settings.get('model')}")
            print(f"  - model_type: {openai_settings.get('model_type')}")
            print(f"  - auth: {openai_settings.get('auth')[:10]}..." if openai_settings.get('auth') else "  - auth: None")
        
        # 方法1：使用 .input().output().start() 的方式
        print("\n" + "="*60)
        print("🧪 方法1: 使用 .input().output().start()")
        print("="*60)
        
        agent1 = Agently.create_agent()
        
        print("📞 执行 API 调用...")
        result1 = (
            agent1
            .input(f"请分析以下文本的情感: {test_input}")
            .output(output_schema)
            .start()
        )
        
        print(f"✅ 方法1 结果: {result1}")
        print(f"📊 结果类型: {type(result1)}")
        
        # 方法2：使用 .set_request_prompt() 的方式
        print("\n" + "="*60)
        print("🧪 方法2: 使用 .set_request_prompt()")
        print("="*60)
        
        agent2 = Agently.create_agent()
        
        # 设置输入
        agent2.set_request_prompt("input", f"请分析以下文本的情感: {test_input}")
        
        # 设置输出格式
        agent2.set_request_prompt("output", output_schema)
        
        print("📞 执行 API 调用...")
        result2 = agent2.start()
        
        print(f"✅ 方法2 结果: {result2}")
        print(f"📊 结果类型: {type(result2)}")
        
        # 方法3：简单调用测试
        print("\n" + "="*60)
        print("🧪 方法3: 简单调用测试")
        print("="*60)
        
        agent3 = Agently.create_agent()
        
        print("📞 执行简单 API 调用...")
        result3 = agent3.input("你好，请回复一句话").start()
        
        print(f"✅ 方法3 结果: {result3}")
        print(f"📊 结果类型: {type(result3)}")
        
        # 总结
        print("\n" + "="*60)
        print("📋 测试总结")
        print("="*60)
        
        print("方法对比:")
        print(f"  - 方法1 (.input().output().start()): {'✅ 成功' if result1 else '❌ 失败'}")
        print(f"  - 方法2 (.set_request_prompt()): {'✅ 成功' if result2 else '❌ 失败'}")
        print(f"  - 方法3 (简单调用): {'✅ 成功' if result3 else '❌ 失败'}")
        
        # 验证结构化输出
        if result1:
            print(f"\n🎯 结构化输出验证 (方法1):")
            if isinstance(result1, dict):
                if 'sentiment' in result1 and 'confidence' in result1:
                    print(f"  ✅ 包含必需字段: sentiment={result1['sentiment']}, confidence={result1['confidence']}")
                else:
                    print(f"  ⚠️ 缺少必需字段: {result1}")
            else:
                print(f"  ⚠️ 非字典格式: {result1}")
        
        if result2:
            print(f"\n🎯 结构化输出验证 (方法2):")
            if isinstance(result2, dict):
                if 'sentiment' in result2 and 'confidence' in result2:
                    print(f"  ✅ 包含必需字段: sentiment={result2['sentiment']}, confidence={result2['confidence']}")
                else:
                    print(f"  ⚠️ 缺少必需字段: {result2}")
            else:
                print(f"  ⚠️ 非字典格式: {result2}")
        
        return result1, result2, result3
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return None, None, None

async def test_agently_async():
    """
    测试 Agently 异步调用
    """
    print("\n" + "="*60)
    print("🧪 异步调用测试")
    print("="*60)
    
    try:
        agent = Agently.create_agent()
        
        result = await agent.input("请简单介绍一下Python").start_async()
        
        print(f"✅ 异步调用结果: {result}")
        return result
        
    except Exception as e:
        print(f"❌ 异步调用失败: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🎯 开始测试 Agently 结构化输出")
    
    # 同步测试
    result1, result2, result3 = test_agently_structured_output()
    
    # 异步测试
    print("\n" + "="*80)
    print("🔄 开始异步测试")
    print("="*80)
    
    async_result = asyncio.run(test_agently_async())
    
    print("\n" + "="*80)
    print("🏁 所有测试完成")
    print("="*80)
    
    print("最终结果:")
    print(f"  - 同步方法1: {'✅' if result1 else '❌'}")
    print(f"  - 同步方法2: {'✅' if result2 else '❌'}")
    print(f"  - 同步方法3: {'✅' if result3 else '❌'}")
    print(f"  - 异步方法: {'✅' if async_result else '❌'}")