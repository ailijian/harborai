#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终测试：使用正确的模型名称测试 HarborAI 结构化输出功能
"""

import json
import traceback
from harborai import HarborAI

def test_harborai_with_correct_model():
    """
    使用正确的模型名称测试 HarborAI 结构化输出功能
    """
    print("🎯 使用正确的模型名称测试 HarborAI 结构化输出功能")
    
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
    
    # 测试不同的 doubao 模型
    models_to_test = [
        "doubao-1-5-pro-32k-character-250715",  # 非推理模型
        "doubao-seed-1-6-250615"  # 推理模型
    ]
    
    for model_name in models_to_test:
        print(f"\n" + "="*80)
        print(f"🧪 测试模型: {model_name}")
        print("="*80)
        
        try:
            # 创建 HarborAI 客户端
            client = HarborAI(
                api_key="6c39786b-2758-4dc3-8b88-a3e8b60d96b3",
                base_url="https://ark.cn-beijing.volces.com/api/v3"
            )
            
            print(f"✅ HarborAI 客户端创建成功，使用模型: {model_name}")
            
            # 测试1：使用 Agently 结构化输出
            print(f"\n🔧 测试1: 使用 Agently 结构化输出 - {model_name}")
            
            response1 = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": f"请分析以下文本的情感: {test_input}"}
                ],
                response_format=response_format,
                structured_provider="agently"
            )
            
            print(f"✅ 测试1 响应成功")
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
            
            # 测试2：使用原生解析
            print(f"\n🔧 测试2: 使用原生解析 - {model_name}")
            
            response2 = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": f"请分析以下文本的情感，返回JSON格式: {test_input}"}
                ],
                response_format=response_format,
                structured_provider="native"
            )
            
            print(f"✅ 测试2 响应成功")
            
            if hasattr(response2, 'choices') and response2.choices:
                content2 = response2.choices[0].message.content
                print(f"📄 响应内容: {content2}")
                print(f"📊 内容类型: {type(content2)}")
            
            # 测试3：不使用结构化输出
            print(f"\n🔧 测试3: 不使用结构化输出 - {model_name}")
            
            response3 = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": f"请分析以下文本的情感: {test_input}"}
                ]
            )
            
            print(f"✅ 测试3 响应成功")
            
            if hasattr(response3, 'choices') and response3.choices:
                content3 = response3.choices[0].message.content
                print(f"📄 响应内容: {content3}")
                print(f"📊 内容类型: {type(content3)}")
            
            # 模型测试总结
            print(f"\n📋 模型 {model_name} 测试总结:")
            print(f"  - Agently 结构化输出: {'✅ 成功' if response1 else '❌ 失败'}")
            print(f"  - 原生结构化输出: {'✅ 成功' if response2 else '❌ 失败'}")
            print(f"  - 基准测试（无结构化）: {'✅ 成功' if response3 else '❌ 失败'}")
            
        except Exception as e:
            print(f"❌ 模型 {model_name} 测试失败: {e}")
            traceback.print_exc()

def test_agently_direct_with_correct_config():
    """
    直接测试 Agently 配置（使用正确的模型名称）
    """
    print("\n" + "="*80)
    print("🔬 直接测试 Agently 配置（使用正确的模型名称）")
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
        
        # 测试不同模型
        models_to_test = [
            "doubao-1-5-pro-32k-character-250715",
            "doubao-seed-1-6-250615"
        ]
        
        for model_name in models_to_test:
            print(f"\n🧪 直接测试模型: {model_name}")
            
            result = handler.parse_response(
                content="今天天气真好，我很开心",
                schema=test_schema,
                use_agently=True,
                api_key="6c39786b-2758-4dc3-8b88-a3e8b60d96b3",
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                model=model_name,
                user_query="请分析以下文本的情感: 今天天气真好，我很开心"
            )
            
            print(f"✅ 模型 {model_name} 直接调用结果: {result}")
            print(f"📊 结果类型: {type(result)}")
        
    except Exception as e:
        print(f"❌ 直接测试失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print("🎯 开始最终测试：使用正确的模型名称")
    
    # 主要测试
    test_harborai_with_correct_model()
    
    # 直接测试
    test_agently_direct_with_correct_config()
    
    print("\n" + "="*80)
    print("🏁 最终测试完成")
    print("="*80)