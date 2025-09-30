#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试 Agently 结构化输出功能
确认 Agently 的正确使用方法
"""

import json
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import Agently
    print("✓ Agently 导入成功")
except ImportError as e:
    print(f"✗ Agently 导入失败: {e}")
    sys.exit(1)


def test_agently_deepseek():
    """测试 Agently 直接调用 DeepSeek 的结构化输出"""
    print("\n=== 测试 Agently + DeepSeek 结构化输出 ===")
    
    try:
        # 配置 Agently 使用 DeepSeek
        Agently.set_settings(
            "OpenAICompatible",
            {
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-chat",
                "model_type": "chat",
                "auth": "sk-d996b310528f44ffb1d7bf5b23b5313b",
            },
        )
        print("✓ Agently DeepSeek 配置完成")
        
        # 创建 agent
        agent = Agently.create_agent()
        print("✓ Agent 创建成功")
        
        # 定义输出格式
        output_schema = {
            "sentiment": ("str", "情感分析结果，可以是 positive, negative, neutral"),
            "confidence": ("float", "置信度，0-1之间的数值"),
            "reason": ("str", "分析理由")
        }
        
        # 执行结构化输出
        print("\n--- 执行情感分析任务 ---")
        result = (
            agent
            .input("今天天气真好，心情很愉快！请分析这句话的情感。")
            .output(output_schema)
            .start()
        )
        
        print(f"✓ 请求成功")
        print(f"结果类型: {type(result)}")
        print(f"结果内容: {result}")
        
        # 验证结果
        if isinstance(result, dict):
            if "sentiment" in result and "confidence" in result:
                print(f"✓ 结构化输出验证成功")
                print(f"  sentiment: {result.get('sentiment')}")
                print(f"  confidence: {result.get('confidence')}")
                print(f"  reason: {result.get('reason', 'N/A')}")
                return True
            else:
                print(f"✗ 结果缺少必要字段: {result}")
                return False
        else:
            print(f"✗ 结果不是字典类型: {result}")
            return False
        
    except Exception as e:
        print(f"✗ DeepSeek 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始测试 Agently 直接调用的结构化输出功能")
    print("=" * 70)
    
    # 先测试 DeepSeek
    success = test_agently_deepseek()
    
    print("\n" + "="*70)
    print("测试总结:")
    
    if success:
        print("✅ DeepSeek: 通过")
        print(f"\n🎉 测试通过！")
        print("\nAgently 正确使用方法总结:")
        print("1. 使用 Agently.set_settings('OpenAICompatible', config) 进行全局配置")
        print("2. 配置参数: {base_url, model, model_type: 'chat', auth}")
        print("3. 使用 Agently.create_agent() 创建 agent")
        print("4. 使用 agent.input().output(schema).start() 执行结构化输出")
        print("5. 输出格式: {field: (type, description)}")
    else:
        print("❌ DeepSeek: 失败")
        print(f"\n⚠️  测试失败")
        print("需要检查 Agently 配置和网络连接。")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)