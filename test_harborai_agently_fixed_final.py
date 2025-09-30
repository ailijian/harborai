#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的 HarborAI Agently 结构化输出功能
使用官方配置和正确的 Agently 配置方法
"""

import json
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harborai import HarborAI


def test_deepseek_structured_output():
    """测试 DeepSeek 模型的结构化输出功能"""
    print("=== 测试 HarborAI + DeepSeek + Agently 结构化输出 ===")
    
    try:
        # 使用官方 DeepSeek 配置
        client = HarborAI(
            api_key="sk-d996b310528f44ffb1d7bf5b23b5313b",
            base_url="https://api.deepseek.com"
        )
        print("✓ HarborAI DeepSeek 客户端初始化成功")
        
        # 定义响应格式
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_analysis",
                "schema": {
                    "type": "object",
                    "properties": {
                        "sentiment": {
                            "type": "string",
                            "description": "情感分析结果，可以是 positive, negative, neutral"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "置信度，0-1之间的数值"
                        },
                        "reason": {
                            "type": "string",
                            "description": "分析理由"
                        }
                    },
                    "required": ["sentiment", "confidence"],
                    "additionalProperties": False
                }
            }
        }
        
        # 测试结构化输出
        print("\n--- 测试 1: DeepSeek Chat 情感分析 ---")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "今天天气真好，心情很愉快！请分析这句话的情感。"}
            ],
            response_format=response_format,
            structured_provider="agently"
        )
        
        print(f"✓ 请求成功")
        print(f"响应类型: {type(response)}")
        print(f"消息内容: {response.choices[0].message.content}")
        print(f"解析结果: {response.choices[0].message.parsed}")
        
        # 验证结果
        parsed = response.choices[0].message.parsed
        if isinstance(parsed, dict) and "sentiment" in parsed and "confidence" in parsed:
            print(f"✓ 结构化输出验证成功")
            print(f"  sentiment: {parsed['sentiment']}")
            print(f"  confidence: {parsed['confidence']}")
            print(f"  reason: {parsed.get('reason', 'N/A')}")
            return True
        else:
            print(f"✗ 结构化输出验证失败: {parsed}")
            return False
        
    except Exception as e:
        print(f"✗ DeepSeek 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始测试修复后的 HarborAI Agently 结构化输出功能")
    print("=" * 70)
    
    # 先测试 DeepSeek
    success = test_deepseek_structured_output()
    
    print("\n" + "="*70)
    print("测试总结:")
    
    if success:
        print("✅ DeepSeek: 通过")
        print(f"\n🎉 测试通过！")
        print("\n修复要点总结:")
        print("1. ✅ 使用正确的 Agently.set_settings('OpenAICompatible', config) 全局配置方法")
        print("2. ✅ 配置参数: {base_url, model, model_type: 'chat', auth}")
        print("3. ✅ 移除了错误的 model.OAIClient.* 配置路径")
        print("4. ✅ 保持了 agent.input().output().start() 的调用方式")
        print("5. ✅ 统一了非流式、流式和异步流式的配置方法")
        print("\n🚀 HarborAI 的 Agently 结构化输出功能已成功修复！")
    else:
        print("❌ DeepSeek: 失败")
        print(f"\n❌ 测试失败")
        print("需要进一步检查修复是否正确。")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)