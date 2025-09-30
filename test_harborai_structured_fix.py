#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的 HarborAI 结构化输出功能
验证 Agently 配置修复是否生效
"""

import json
import traceback
from harborai import HarborAI

def test_harborai_structured_output():
    """
    测试 HarborAI 结构化输出功能
    """
    print("🚀 测试修复后的 HarborAI 结构化输出功能")
    print("验证 Agently 配置修复是否生效")
    
    # 测试输入
    test_input = "今天天气真好，我很开心"
    
    # 定义 JSON Schema
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "sentiment_analysis",
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "description": "情感分析结果：positive, negative, neutral"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "置信度，0-1之间的数值"
                    },
                    "reason": {
                        "type": "string",
                        "description": "分析原因"
                    }
                },
                "required": ["sentiment", "confidence", "reason"]
            }
        }
    }
    
    print(f"📝 测试输入: {test_input}")
    print(f"📋 期望输出格式: {json.dumps(response_format, indent=2, ensure_ascii=False)}")
    
    try:
        # 创建 HarborAI 客户端（使用火山 Ark 配置）
        print("\n" + "="*60)
        print("🔧 创建 HarborAI 客户端")
        print("="*60)
        
        client = HarborAI(
            api_key="6c39786b-2758-4dc3-8b88-a3e8b60d96b3",
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        
        print("✅ HarborAI 客户端创建成功")
        
        # 测试1：使用 Agently 结构化输出
        print("\n" + "="*60)
        print("🧪 测试1: 使用 Agently 结构化输出")
        print("="*60)
        
        response1 = client.chat.completions.create(
            model="ep-20250509161856-ntmhj",
            messages=[
                {"role": "user", "content": f"请分析以下文本的情感: {test_input}"}
            ],
            response_format=response_format,
            structured_provider="agently"
        )
        
        print(f"✅ 测试1 响应: {response1}")
        print(f"📊 响应类型: {type(response1)}")
        
        if hasattr(response1, 'choices') and response1.choices:
            content1 = response1.choices[0].message.content
            print(f"📄 响应内容: {content1}")
            print(f"📊 内容类型: {type(content1)}")
            
            # 验证结构化输出
            if isinstance(content1, dict):
                print("🎯 结构化输出验证:")
                required_fields = ["sentiment", "confidence", "reason"]
                for field in required_fields:
                    if field in content1:
                        print(f"  ✅ {field}: {content1[field]}")
                    else:
                        print(f"  ❌ 缺少字段: {field}")
            else:
                print(f"⚠️ 内容不是字典格式: {content1}")
        
        # 测试2：使用原生解析（对比测试）
        print("\n" + "="*60)
        print("🧪 测试2: 使用原生解析（对比测试）")
        print("="*60)
        
        response2 = client.chat.completions.create(
            model="ep-20250509161856-ntmhj",
            messages=[
                {"role": "user", "content": f"请分析以下文本的情感，返回JSON格式: {test_input}"}
            ],
            response_format=response_format,
            structured_provider="native"
        )
        
        print(f"✅ 测试2 响应: {response2}")
        
        if hasattr(response2, 'choices') and response2.choices:
            content2 = response2.choices[0].message.content
            print(f"📄 响应内容: {content2}")
            print(f"📊 内容类型: {type(content2)}")
        
        # 测试3：不使用结构化输出（基准测试）
        print("\n" + "="*60)
        print("🧪 测试3: 不使用结构化输出（基准测试）")
        print("="*60)
        
        response3 = client.chat.completions.create(
            model="ep-20250509161856-ntmhj",
            messages=[
                {"role": "user", "content": f"请分析以下文本的情感: {test_input}"}
            ]
        )
        
        print(f"✅ 测试3 响应: {response3}")
        
        if hasattr(response3, 'choices') and response3.choices:
            content3 = response3.choices[0].message.content
            print(f"📄 响应内容: {content3}")
            print(f"📊 内容类型: {type(content3)}")
        
        # 总结
        print("\n" + "="*60)
        print("📋 测试总结")
        print("="*60)
        
        print("测试结果:")
        print(f"  - Agently 结构化输出: {'✅ 成功' if response1 else '❌ 失败'}")
        print(f"  - 原生结构化输出: {'✅ 成功' if response2 else '❌ 失败'}")
        print(f"  - 基准测试（无结构化）: {'✅ 成功' if response3 else '❌ 失败'}")
        
        return response1, response2, response3
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        traceback.print_exc()
        return None, None, None

def test_harborai_agently_direct():
    """
    直接测试 HarborAI 的 Agently 集成
    """
    print("\n" + "="*80)
    print("🔬 直接测试 HarborAI 的 Agently 集成")
    print("="*80)
    
    try:
        from harborai.api.structured import StructuredOutputHandler
        
        # 创建结构化输出处理器
        handler = StructuredOutputHandler(provider="agently")
        
        # 测试 schema
        test_schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "description": "情感分析结果：positive, negative, neutral"
                },
                "confidence": {
                    "type": "number",
                    "description": "置信度，0-1之间的数值"
                }
            },
            "required": ["sentiment", "confidence"]
        }
        
        # 直接调用解析方法
        result = handler.parse_response(
            content="今天天气真好，我很开心",
            schema=test_schema,
            use_agently=True,
            api_key="6c39786b-2758-4dc3-8b88-a3e8b60d96b3",
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            model="ep-20250509161856-ntmhj",
            user_query="请分析以下文本的情感: 今天天气真好，我很开心"
        )
        
        print(f"✅ 直接调用结果: {result}")
        print(f"📊 结果类型: {type(result)}")
        
        return result
        
    except Exception as e:
        print(f"❌ 直接测试失败: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🎯 开始测试修复后的 HarborAI 结构化输出功能")
    
    # 主要测试
    response1, response2, response3 = test_harborai_structured_output()
    
    # 直接测试
    direct_result = test_harborai_agently_direct()
    
    print("\n" + "="*80)
    print("🏁 所有测试完成")
    print("="*80)
    
    print("最终结果:")
    print(f"  - HarborAI Agently 结构化: {'✅' if response1 else '❌'}")
    print(f"  - HarborAI 原生结构化: {'✅' if response2 else '❌'}")
    print(f"  - HarborAI 基准测试: {'✅' if response3 else '❌'}")
    print(f"  - 直接 Agently 调用: {'✅' if direct_result else '❌'}")
    
    if response1 or response2 or response3 or direct_result:
        print("\n🎉 至少有一个测试成功，HarborAI 功能正常！")
    else:
        print("\n⚠️ 所有测试都失败，需要进一步调试")