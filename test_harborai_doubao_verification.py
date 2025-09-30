#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证修复后的 HarborAI 豆包 Agently 结构化输出功能
"""

import json
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harborai import HarborAI


def test_doubao_structured_output():
    """测试豆包模型的结构化输出功能"""
    print("=== 测试 HarborAI + 豆包 + Agently 结构化输出 ===")
    
    try:
        # 使用官方豆包配置
        client = HarborAI(
            api_key="4ed46be9-4eb4-45f1-8576-d2fc3d115026",
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        print("✓ HarborAI 豆包客户端初始化成功")
        
        # 定义响应格式
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "simple_qa",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "简单回答"
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "推理过程"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "回答的置信度"
                        }
                    },
                    "required": ["answer"],
                    "additionalProperties": False
                }
            }
        }
        
        # 测试结构化输出
        print("\n--- 测试: 豆包 1.5 Pro 32K 问答 ---")
        response = client.chat.completions.create(
            model="doubao-1-5-pro-32k-character-250715",
            messages=[
                {"role": "user", "content": "请简单回答：什么是人工智能？"}
            ],
            response_format=response_format,
            structured_provider="agently"
        )
        
        print(f"✓ 请求成功")
        print(f"消息内容: {response.choices[0].message.content}")
        print(f"解析结果: {response.choices[0].message.parsed}")
        
        # 验证结果
        parsed = response.choices[0].message.parsed
        if isinstance(parsed, dict) and "answer" in parsed:
            print(f"✓ 结构化输出验证成功")
            print(f"  answer: {parsed['answer']}")
            print(f"  reasoning: {parsed.get('reasoning', 'N/A')}")
            print(f"  confidence: {parsed.get('confidence', 'N/A')}")
            return True
        else:
            print(f"✗ 结构化输出验证失败: {parsed}")
            return False
        
    except Exception as e:
        print(f"✗ 豆包测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("验证修复后的 HarborAI 豆包 Agently 结构化输出功能")
    print("=" * 70)
    
    success = test_doubao_structured_output()
    
    print("\n" + "="*70)
    print("验证结果:")
    
    if success:
        print("✅ 豆包测试通过")
        print("\n🎉 豆包模型的 Agently 结构化输出功能验证成功！")
    else:
        print("❌ 豆包测试失败")
        print("\n⚠️  豆包模型可能需要进一步调试。")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)