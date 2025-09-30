#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试豆包 Agently 配置问题
"""

import json
import traceback
from agently import Agently


def debug_doubao_agently():
    """调试豆包 Agently 配置"""
    print("=== 调试豆包 Agently 配置 ===")
    
    # 豆包配置
    config = {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-1-5-pro-32k-character-250715",
        "model_type": "chat",
        "auth": "4ed46be9-4eb4-45f1-8576-d2fc3d115026",
    }
    
    print(f"配置: {config}")
    
    try:
        # 设置全局配置
        Agently.set_settings("OpenAICompatible", config)
        print("✅ 全局配置设置成功")
        
        # 创建 agent
        agent = Agently.create_agent()
        print("✅ Agent 创建成功")
        
        # 测试简单输出
        print("\n--- 测试1: 简单文本输出 ---")
        try:
            result1 = agent.input("请说'你好'").start()
            print(f"✓ 简单输出成功: {result1}")
        except Exception as e:
            print(f"❌ 简单输出失败: {e}")
            traceback.print_exc()
        
        # 测试结构化输出
        print("\n--- 测试2: 结构化输出 ---")
        try:
            output_format = {
                "answer": ("String", "简单回答"),
                "confidence": ("Number", "置信度")
            }
            
            result2 = (
                agent
                .input("什么是AI？")
                .output(output_format)
                .start()
            )
            print(f"✓ 结构化输出成功: {json.dumps(result2, ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"❌ 结构化输出失败: {e}")
            traceback.print_exc()
        
        # 测试不同的配置方式
        print("\n--- 测试3: 重新配置测试 ---")
        try:
            # 清除之前的配置（如果有这样的方法）
            agent2 = Agently.create_agent()
            
            # 尝试使用不同的配置方式
            result3 = (
                agent2
                .input("请回答：1+1等于几？")
                .output({"result": ("String", "计算结果")})
                .start()
            )
            print(f"✓ 重新配置测试成功: {json.dumps(result3, ensure_ascii=False, indent=2)}")
        except Exception as e:
            print(f"❌ 重新配置测试失败: {e}")
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 配置失败: {e}")
        traceback.print_exc()


def test_different_models():
    """测试不同的豆包模型"""
    print("\n=== 测试不同的豆包模型 ===")
    
    models = [
        "doubao-1-5-pro-32k-character-250715",
        "doubao-seed-1-6-250615"
    ]
    
    for model in models:
        print(f"\n--- 测试模型: {model} ---")
        try:
            config = {
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "model": model,
                "model_type": "chat",
                "auth": "4ed46be9-4eb4-45f1-8576-d2fc3d115026",
            }
            
            Agently.set_settings("OpenAICompatible", config)
            agent = Agently.create_agent()
            
            result = (
                agent
                .input("请简单回答：什么是机器学习？")
                .output({"answer": ("String", "回答")})
                .start()
            )
            
            print(f"✅ 模型 {model} 测试成功: {json.dumps(result, ensure_ascii=False)}")
            
        except Exception as e:
            print(f"❌ 模型 {model} 测试失败: {e}")


def main():
    """主函数"""
    debug_doubao_agently()
    test_different_models()


if __name__ == "__main__":
    main()