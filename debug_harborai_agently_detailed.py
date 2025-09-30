#!/usr/bin/env python3
"""
详细调试 HarborAI 中豆包模型的 Agently 结构化输出问题
"""

import os
import sys
import json
import logging
from typing import Dict, Any

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ 环境变量已加载")
except ImportError:
    print("⚠ python-dotenv未安装，直接使用环境变量")

from harborai import HarborAI

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_harborai_doubao_agently():
    """测试 HarborAI 中豆包模型的 Agently 结构化输出"""
    
    print("=== 详细调试 HarborAI 豆包 Agently 结构化输出 ===")
    
    # 初始化 HarborAI 客户端
    client = HarborAI()
    
    # 测试消息
    messages = [
        {"role": "user", "content": "什么是人工智能？请用中文回答。"}
    ]
    
    # 结构化输出格式
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "ai_explanation",
            "schema": {
                "type": "object",
                "properties": {
                    "definition": {
                        "type": "string",
                        "description": "人工智能的定义"
                    },
                    "key_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "人工智能的主要应用领域"
                    },
                    "importance": {
                        "type": "string",
                        "description": "人工智能的重要性"
                    }
                },
                "required": ["definition", "key_areas", "importance"],
                "additionalProperties": False
            }
        }
    }
    
    try:
        print("\n--- 测试豆包 1.5 Pro 32K Agently 结构化输出 ---")
        
        # 使用 Agently 作为结构化提供者
        response = client.chat.completions.create(
            model="doubao-1-5-pro-32k-character-250715",
            messages=messages,
            response_format=response_format,
            structured_provider="agently",  # 明确指定使用 Agently
            temperature=0.1
        )
        
        print(f"✓ 请求成功")
        print(f"消息内容: {response.choices[0].message.content}")
        
        # 检查结构化输出
        if hasattr(response.choices[0].message, 'parsed') and response.choices[0].message.parsed:
            parsed_content = response.choices[0].message.parsed
            print(f"✓ 结构化输出成功: {json.dumps(parsed_content, ensure_ascii=False, indent=2)}")
            
            # 验证结构
            if isinstance(parsed_content, dict):
                required_fields = ["definition", "key_areas", "importance"]
                missing_fields = [field for field in required_fields if field not in parsed_content]
                
                if not missing_fields:
                    print("✓ 结构化输出验证成功")
                    return True
                else:
                    print(f"✗ 缺少必需字段: {missing_fields}")
                    return False
            else:
                print(f"✗ 结构化输出格式错误: {type(parsed_content)}")
                return False
        else:
            print("✗ 结构化输出解析失败: parsed 字段为空")
            return False
            
    except Exception as e:
        print(f"✗ 豆包测试失败: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return False

def main():
    """主函数"""
    success = test_harborai_doubao_agently()
    
    print("\n" + "="*60)
    print("验证结果:")
    if success:
        print("✅ 豆包 Agently 结构化输出测试成功")
    else:
        print("❌ 豆包 Agently 结构化输出测试失败")
        print("\n⚠️  豆包模型的 Agently 配置可能需要进一步调试。")

if __name__ == "__main__":
    main()