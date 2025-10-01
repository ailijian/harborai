#!/usr/bin/env python3
"""
调试架构测试问题
"""

import os
import sys
import json
import time
from dotenv import load_dotenv

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harborai import HarborAI

# 加载环境变量
load_dotenv()

def debug_current_approach():
    """调试当前方案"""
    print("🔍 调试当前方案：DeepSeek json_object + Agently 后处理")
    
    client = HarborAI()
    
    user_schema = {
        "name": "user_info",
        "description": "用户基本信息",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "用户姓名"},
                "age": {"type": "integer", "description": "用户年龄"},
                "email": {"type": "string", "description": "用户邮箱地址"}
            },
            "required": ["name", "age", "email"],
            "additionalProperties": False
        }
    }
    
    prompt = "请生成一个虚拟用户的基本信息，包括姓名、年龄、邮箱。用户是一个28岁的程序员。"
    
    try:
        print("📤 发送请求...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": user_schema
            },
            structured_provider="agently",
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        print(f"📄 响应内容: {content}")
        print(f"📄 响应类型: {type(content)}")
        
        # 验证 JSON
        try:
            parsed_json = json.loads(content)
            print(f"✅ JSON 解析成功:")
            print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

def debug_pure_agently():
    """调试纯 Agently 方案"""
    print("\n🔍 调试纯 Agently 方案")
    
    client = HarborAI()
    
    user_schema = {
        "name": "user_info",
        "description": "用户基本信息",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "用户姓名"},
                "age": {"type": "integer", "description": "用户年龄"},
                "email": {"type": "string", "description": "用户邮箱地址"}
            },
            "required": ["name", "age", "email"],
            "additionalProperties": False
        }
    }
    
    prompt = "请生成一个虚拟用户的基本信息，包括姓名、年龄、邮箱。用户是一个28岁的程序员。"
    
    try:
        print("📤 发送请求...")
        # 不使用 json_object，让 Agently 完全处理
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": user_schema
            },
            # 不指定 structured_provider，让它使用默认的 agently
            temperature=0.3
        )
        
        content = response.choices[0].message.content
        print(f"📄 响应内容: {content}")
        print(f"📄 响应类型: {type(content)}")
        
        # 验证 JSON
        try:
            parsed_json = json.loads(content)
            print(f"✅ JSON 解析成功:")
            print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 请求失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("调试 DeepSeek 架构测试问题")
    print("=" * 60)
    
    result1 = debug_current_approach()
    result2 = debug_pure_agently()
    
    print(f"\n📊 调试结果:")
    print(f"   当前方案成功: {result1}")
    print(f"   纯 Agently 方案成功: {result2}")