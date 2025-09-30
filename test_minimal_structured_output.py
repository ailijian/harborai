#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小的HarborAI结构化输出测试
验证deepseek-chat模型的agently结构化输出功能
"""

import json
import sys
import os
import time

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# 加载环境变量
try:
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✓ 已加载环境变量文件: {env_path}")
    else:
        print(f"⚠ 环境变量文件不存在: {env_path}")
except ImportError:
    print("⚠ python-dotenv未安装，直接使用环境变量")

from harborai import HarborAI

def test_minimal_structured_output():
    """测试最小的结构化输出功能"""
    print("🧪 测试最小的HarborAI结构化输出功能")
    print("=" * 60)
    
    try:
        # 创建HarborAI客户端
        client = HarborAI()
        print("✓ HarborAI客户端创建成功")
        
        # 定义简单的JSON schema
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "情感倾向分析"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "置信度分数"
                }
            },
            "required": ["sentiment", "confidence"]
        }
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_analysis",
                "schema": schema,
                "strict": True
            }
        }
        
        print("✓ JSON schema定义完成")
        print(f"📋 Schema: {json.dumps(schema, indent=2, ensure_ascii=False)}")
        
        # 测试文本
        test_text = "今天天气很好，心情不错"
        print(f"📝 测试文本: {test_text}")
        
        # 调用结构化输出（默认使用agently）
        print("\n🔄 开始调用结构化输出...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": f"请分析这段文本的情感：'{test_text}'"
                }
            ],
            response_format=response_format,
            temperature=0.1,
            max_tokens=500
        )
        
        end_time = time.time()
        print(f"✅ 调用完成，耗时: {end_time - start_time:.2f}秒")
        
        # 输出详细的调试信息
        print("\n🔍 详细调试信息:")
        print(f"   响应对象类型: {type(response)}")
        print(f"   响应对象: {response}")
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            print(f"   Choice对象: {choice}")
            
            if hasattr(choice, 'message'):
                message = choice.message
                print(f"   Message对象: {message}")
                print(f"   Message内容: {message.content}")
                
                if hasattr(message, 'parsed'):
                    parsed_data = message.parsed
                    print(f"   解析后的结构化数据: {parsed_data}")
                    print(f"   解析数据类型: {type(parsed_data)}")
                    
                    # 验证结构化数据
                    if parsed_data:
                        print("\n✅ 结构化输出验证:")
                        print(f"   sentiment: {parsed_data.get('sentiment', 'N/A')}")
                        print(f"   confidence: {parsed_data.get('confidence', 'N/A')}")
                        
                        # 验证必需字段
                        assert 'sentiment' in parsed_data, "缺少sentiment字段"
                        assert 'confidence' in parsed_data, "缺少confidence字段"
                        assert parsed_data['sentiment'] in ['positive', 'negative', 'neutral'], "sentiment值不在允许范围内"
                        assert 0 <= parsed_data['confidence'] <= 100, "confidence值不在0-100范围内"
                        
                        print("✅ 所有验证通过")
                        return True
                    else:
                        print("❌ 解析后的结构化数据为空")
                        return False
                else:
                    print("❌ Message对象没有parsed字段")
                    return False
            else:
                print("❌ Choice对象没有message字段")
                return False
        else:
            print("❌ 响应对象没有choices字段或choices为空")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_minimal_structured_output()
    if success:
        print("\n🎉 最小结构化输出测试通过")
        print("✅ HarborAI结构化输出功能正常工作")
        print("✅ 默认使用agently作为structured_provider")
        print("✅ response.choices[0].message.parsed字段包含正确的结构化数据")
        sys.exit(0)
    else:
        print("\n🚨 最小结构化输出测试失败")
        sys.exit(1)