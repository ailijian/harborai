#!/usr/bin/env python3
"""
测试 DeepSeek 是否支持 JSON Schema 原生结构化输出
"""

import os
import sys
import json
import asyncio
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harborai import HarborAI

# 加载环境变量
load_dotenv()

def test_deepseek_json_schema():
    """测试 DeepSeek 的 JSON Schema 支持"""
    
    # 初始化客户端
    client = HarborAI()
    
    # 定义简单的用户信息 JSON Schema
    user_schema = {
        "name": "user_info",
        "description": "用户基本信息",
        "schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "用户姓名"
                },
                "age": {
                    "type": "integer",
                    "description": "用户年龄",
                    "minimum": 0,
                    "maximum": 150
                },
                "email": {
                    "type": "string",
                    "description": "用户邮箱地址",
                    "format": "email"
                },
                "city": {
                    "type": "string",
                    "description": "用户所在城市"
                }
            },
            "required": ["name", "age", "email"],
            "additionalProperties": False
        }
    }
    
    # 测试的 DeepSeek 模型
    deepseek_models = [
        "deepseek-chat",
        "deepseek-reasoner"
    ]
    
    print("=" * 80)
    print("测试 DeepSeek 模型的 JSON Schema 原生结构化输出支持")
    print("=" * 80)
    
    for model in deepseek_models:
        print(f"\n🧪 测试模型: {model}")
        print("-" * 50)
        
        # 测试原生结构化输出 (structured_provider='native')
        print("📋 测试原生 JSON Schema 支持...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user", 
                        "content": "请生成一个虚拟用户的基本信息，包括姓名、年龄、邮箱和城市。用户是一个25岁的软件工程师，住在北京。"
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": user_schema
                },
                structured_provider="native",  # 使用原生结构化输出
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            print(f"✅ 原生结构化输出成功")
            print(f"📄 响应内容: {content}")
            
            # 验证返回的内容是否为有效 JSON
            try:
                parsed_json = json.loads(content)
                print(f"✅ JSON 解析成功: {json.dumps(parsed_json, ensure_ascii=False, indent=2)}")
                
                # 验证必需字段
                required_fields = ["name", "age", "email"]
                missing_fields = [field for field in required_fields if field not in parsed_json]
                if missing_fields:
                    print(f"⚠️  缺少必需字段: {missing_fields}")
                else:
                    print(f"✅ 所有必需字段都存在")
                    
                # 验证数据类型
                if isinstance(parsed_json.get("name"), str):
                    print(f"✅ name 字段类型正确: {parsed_json['name']}")
                else:
                    print(f"❌ name 字段类型错误")
                    
                if isinstance(parsed_json.get("age"), int):
                    print(f"✅ age 字段类型正确: {parsed_json['age']}")
                else:
                    print(f"❌ age 字段类型错误")
                    
                if isinstance(parsed_json.get("email"), str) and "@" in parsed_json.get("email", ""):
                    print(f"✅ email 字段格式正确: {parsed_json['email']}")
                else:
                    print(f"❌ email 字段格式错误")
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON 解析失败: {e}")
                
        except Exception as e:
            print(f"❌ 原生结构化输出失败: {e}")
            print(f"   错误类型: {type(e).__name__}")
            
        # 测试传统方式 (structured_provider='agently')
        print("\n📋 测试传统 Agently 后处理方式...")
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user", 
                        "content": "请生成一个虚拟用户的基本信息，包括姓名、年龄、邮箱和城市。用户是一个30岁的设计师，住在上海。请以JSON格式返回。"
                    }
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": user_schema
                },
                structured_provider="agently",  # 使用 Agently 后处理
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            print(f"✅ Agently 后处理成功")
            print(f"📄 响应内容: {content}")
            
            # 验证返回的内容是否为有效 JSON
            try:
                parsed_json = json.loads(content)
                print(f"✅ JSON 解析成功: {json.dumps(parsed_json, ensure_ascii=False, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON 解析失败: {e}")
                
        except Exception as e:
            print(f"❌ Agently 后处理失败: {e}")
            print(f"   错误类型: {type(e).__name__}")
            
        print("\n" + "=" * 50)

if __name__ == "__main__":
    test_deepseek_json_schema()