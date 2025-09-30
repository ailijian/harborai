#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试豆包模型的 Agently 结构化输出功能
"""

import json
import traceback
from agently import Agently


def test_doubao_agently_direct():
    """直接测试豆包模型的 Agently 结构化输出"""
    print("=== 直接测试豆包 + Agently 结构化输出 ===")
    
    try:
        print("\n🔧 配置 Agently 豆包设置")
        print("="*60)
        
        # 配置豆包模型
        Agently.set_settings(
            "OpenAICompatible",
            {
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "model": "doubao-1-5-pro-32k-character-250715",
                "model_type": "chat",
                "auth": "4ed46be9-4eb4-45f1-8576-d2fc3d115026",
            },
        )
        
        print("✅ 豆包全局配置完成")
        
        # 创建 agent
        agent = Agently.create_agent()
        print("✅ Agent 创建成功")
        
        # 定义输出格式
        output_format = {
            "answer": ("String", "简单回答"),
            "reasoning": ("String", "推理过程"),
            "confidence": ("Number", "回答的置信度")
        }
        
        print("\n📝 测试结构化输出")
        print("="*60)
        
        # 测试方法1: .input().output().start()
        print("\n方法1: .input().output().start()")
        try:
            result1 = (
                agent
                .input("请简单回答：什么是人工智能？")
                .output(output_format)
                .start()
            )
            print(f"✓ 方法1成功: {json.dumps(result1, ensure_ascii=False, indent=2)}")
            
            # 验证结果
            if isinstance(result1, dict) and "answer" in result1:
                print("✅ 方法1结构化输出验证成功")
                return True
            else:
                print(f"❌ 方法1结构化输出验证失败: {result1}")
                return False
                
        except Exception as e:
            print(f"❌ 方法1失败: {e}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ 豆包 Agently 测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("直接测试豆包模型的 Agently 结构化输出功能")
    print("=" * 70)
    
    success = test_doubao_agently_direct()
    
    print("\n" + "="*70)
    print("测试结果:")
    
    if success:
        print("✅ 豆包 Agently 直接测试通过")
        print("\n🎉 豆包模型的 Agently 结构化输出功能正常！")
    else:
        print("❌ 豆包 Agently 直接测试失败")
        print("\n⚠️  豆包模型可能不支持 Agently 或需要特殊配置。")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)