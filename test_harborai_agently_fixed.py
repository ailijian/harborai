#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的 HarborAI Agently 结构化输出功能
"""

import json
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harborai import HarborAI


def test_harborai_structured_output():
    """测试 HarborAI 的结构化输出功能"""
    print("=== 测试 HarborAI 结构化输出功能 ===")
    
    try:
        # 初始化 HarborAI 客户端
        client = HarborAI(
            api_key="6c39786b-2758-4dc3-8b88-a3e8b60d96b3",
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        print("✓ HarborAI 客户端初始化成功")
        
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
                        }
                    },
                    "required": ["sentiment", "confidence"],
                    "additionalProperties": False
                }
            }
        }
        
        # 测试结构化输出
        print("\n--- 测试 1: 简单情感分析 ---")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "今天天气真好，心情很愉快！"}
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
        else:
            print(f"✗ 结构化输出验证失败: {parsed}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始测试修复后的 HarborAI Agently 结构化输出功能")
    print("=" * 60)
    
    success = test_harborai_structured_output()
    
    print("\n" + "="*60)
    print("测试总结:")
    
    if success:
        print("✅ 所有测试通过！HarborAI Agently 结构化输出功能正常工作。")
        print("\n修复要点:")
        print("1. 使用正确的 Agently.set_settings() 配置方法")
        print("2. 使用 model.OAIClient.* 配置路径")
        print("3. 移除了硬编码的火山Ark配置")
        print("4. 保持了 agent.input().output().start() 的调用方式")
    else:
        print("❌ 测试失败，需要进一步调试。")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)