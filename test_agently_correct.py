#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正确的 Agently 使用方法测试脚本
"""

import json
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import Agently
    print("✓ Agently 导入成功")
    print(f"Agently 类型: {type(Agently)}")
    print(f"Agently 可用方法: {[method for method in dir(Agently) if not method.startswith('_')]}")
except ImportError as e:
    print(f"✗ Agently 导入失败: {e}")
    sys.exit(1)


def test_agently_configuration():
    """测试 Agently 配置"""
    print("\n=== 测试 Agently 配置 ===")
    
    try:
        # 配置 Agently
        Agently.set_settings("model.OAIClient.base_url", "https://ark.cn-beijing.volces.com/api/v3")
        Agently.set_settings("model.OAIClient.api_key", "6c39786b-2758-4dc3-8b88-a3e8b60d96b3")
        Agently.set_settings("model.OAIClient.model", "ep-20250509161856-ntmhj")
        print("✓ Agently 配置成功")
        
        # 创建 agent
        agent = Agently.create_agent()
        print("✓ Agent 创建成功")
        print(f"Agent 类型: {type(agent)}")
        print(f"Agent 可用方法: {[method for method in dir(agent) if not method.startswith('_')]}")
        
        return agent
        
    except Exception as e:
        print(f"✗ Agently 配置失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_simple_structured_output(agent):
    """测试简单的结构化输出"""
    print("\n=== 测试简单结构化输出 ===")
    
    if not agent:
        print("✗ Agent 未初始化，跳过测试")
        return False
    
    try:
        # 定义输出格式
        output_format = {
            "sentiment": ("String", "情感分析结果，可以是 positive, negative, neutral"),
            "confidence": ("Number", "置信度，0-1之间的数值")
        }
        
        # 执行结构化输出
        result = agent.input("今天天气真好，心情很愉快！").output(output_format).start()
        
        print(f"✓ 结构化输出成功")
        print(f"结果类型: {type(result)}")
        print(f"结果内容: {result}")
        
        # 验证结果结构
        if isinstance(result, dict):
            if "sentiment" in result and "confidence" in result:
                print("✓ 结果包含所需字段")
                print(f"  sentiment: {result['sentiment']}")
                print(f"  confidence: {result['confidence']}")
                return True
            else:
                print(f"✗ 结果缺少必需字段: {list(result.keys())}")
        else:
            print(f"✗ 结果不是字典类型: {type(result)}")
        
        return False
        
    except Exception as e:
        print(f"✗ 结构化输出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("开始 Agently 正确使用方法测试")
    print("=" * 50)
    
    # 测试基本配置
    agent = test_agently_configuration()
    
    if not agent:
        print("\n✗ 基本配置失败，无法继续测试")
        return False
    
    # 测试简单结构化输出
    success = test_simple_structured_output(agent)
    
    # 输出测试总结
    print("\n" + "="*50)
    print("测试总结:")
    
    if success:
        print("✓ 简单结构化输出: 通过")
        print("\n🎉 测试通过！Agently 配置和使用方法正确。")
        print("\n正确的 Agently 使用方法:")
        print("1. 使用 Agently.set_settings() 进行全局配置")
        print("2. 使用 Agently.create_agent() 创建 agent")
        print("3. 使用 agent.input().output().start() 进行结构化输出")
    else:
        print("✗ 简单结构化输出: 失败")
        print("\n❌ 测试失败，需要检查配置或使用方法。")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)