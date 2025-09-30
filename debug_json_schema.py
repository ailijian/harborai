#!/usr/bin/env python3
"""
调试 JSON Schema 结构化输出问题
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.DEBUG)

# 加载环境变量
from dotenv import load_dotenv

env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"已加载环境变量文件: {env_path}")

# 导入 HarborAI
from harborai import HarborAI

def debug_json_schema():
    """调试 JSON Schema 问题"""
    print("🔍 调试 JSON Schema 结构化输出")
    print("="*50)
    
    try:
        # 初始化 HarborAI
        client = HarborAI()
        
        # 简单的 JSON Schema
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["sentiment", "confidence"]
        }
        
        # 创建 response_format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_analysis",
                "schema": schema,
                "strict": True
            }
        }
        
        print(f"📋 Schema: {json.dumps(schema, ensure_ascii=False, indent=2)}")
        print(f"📋 Response Format: {json.dumps(response_format, ensure_ascii=False, indent=2)}")
        
        # 发送请求
        messages = [
            {"role": "user", "content": "请分析这段文本的情感并返回JSON格式：'今天天气真好，我很开心！'"}
        ]
        
        print("📤 发送请求...")
        print(f"📤 Messages: {json.dumps(messages, ensure_ascii=False, indent=2)}")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format=response_format,
            temperature=0.1
        )
        
        print("✅ 请求成功")
        print(f"📥 响应类型: {type(response)}")
        print(f"📥 响应对象: {response}")
        
        # 解析响应
        content = response.choices[0].message.content
        print(f"📄 原始内容: {content}")
        print(f"📄 内容类型: {type(content)}")
        print(f"📄 内容长度: {len(content) if content else 'None'}")
        
        if content:
            # 尝试解析 JSON
            try:
                parsed_data = json.loads(content)
                print(f"✅ JSON 解析成功: {json.dumps(parsed_data, ensure_ascii=False, indent=2)}")
                
                # 验证数据
                if "sentiment" in parsed_data and "confidence" in parsed_data:
                    print("✅ 所有必需字段都存在")
                    print(f"   情感: {parsed_data['sentiment']}")
                    print(f"   置信度: {parsed_data['confidence']}")
                    return True
                else:
                    print("❌ 缺少必需字段")
                    missing = []
                    if "sentiment" not in parsed_data:
                        missing.append("sentiment")
                    if "confidence" not in parsed_data:
                        missing.append("confidence")
                    print(f"   缺少字段: {missing}")
                    return False
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON 解析失败: {e}")
                print(f"   尝试解析的内容: '{content}'")
                return False
        else:
            print("❌ 响应内容为空")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_json_schema()
    print(f"\n🏁 调试结果: {'成功' if success else '失败'}")