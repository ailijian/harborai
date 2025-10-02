#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实际测试 Agently 结构化输出的准确性
使用真实的 API 调用来验证不同 schema 描述的效果
"""

import json
import sys
import os
import time

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harborai import HarborAI

def load_env_file():
    """加载.env文件中的环境变量"""
    from pathlib import Path
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

def test_original_schema():
    """测试原始的 schema"""
    load_env_file()
    
    client = HarborAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    
    # 原始 schema
    original_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "编程语言名称"
            },
            "category": {
                "type": "string",
                "description": "编程语言类别"
            },
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "主要特性列表"
            },
            "difficulty": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
                "description": "学习难度"
            }
        },
        "required": ["name", "category", "features", "difficulty"]
    }
    
    print("🔍 测试原始 Schema:")
    print(json.dumps(original_schema, indent=2, ensure_ascii=False))
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "请介绍Python编程语言的基本信息"}
            ],
            max_tokens=200,
            structured_provider="agently",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "programming_language_info",
                    "schema": original_schema
                }
            }
        )
        
        print("\n✅ 原始 Schema 结果:")
        result = json.loads(response.choices[0].message.content)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"字段列表: {list(result.keys())}")
        
        return result
        
    except Exception as e:
        print(f"❌ 原始 Schema 测试失败: {e}")
        return None

def test_improved_schema():
    """测试改进的 schema，描述更加明确"""
    load_env_file()
    
    client = HarborAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    
    # 改进的 schema，描述更加明确
    improved_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "编程语言的确切名称，必须使用字段名'name'，例如：Python"
            },
            "category": {
                "type": "string",
                "description": "编程语言的分类类型，必须使用字段名'category'，例如：解释型高级编程语言"
            },
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "编程语言的主要特性列表，必须使用字段名'features'"
            },
            "difficulty": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
                "description": "学习难度等级，必须使用字段名'difficulty'，值必须是：easy、medium或hard"
            }
        },
        "required": ["name", "category", "features", "difficulty"]
    }
    
    print("\n" + "="*60)
    print("🔍 测试改进的 Schema:")
    print(json.dumps(improved_schema, indent=2, ensure_ascii=False))
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "请介绍Python编程语言的基本信息"}
            ],
            max_tokens=200,
            structured_provider="agently",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "programming_language_info",
                    "schema": improved_schema
                }
            }
        )
        
        print("\n✅ 改进 Schema 结果:")
        result = json.loads(response.choices[0].message.content)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"字段列表: {list(result.keys())}")
        
        return result
        
    except Exception as e:
        print(f"❌ 改进 Schema 测试失败: {e}")
        return None

def test_strict_schema():
    """测试严格的 schema，强调字段名的重要性"""
    load_env_file()
    
    client = HarborAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    
    # 严格的 schema，强调字段名
    strict_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "【重要】必须使用字段名'name'，不能使用其他名称如language_name。编程语言的名称，如：Python"
            },
            "category": {
                "type": "string",
                "description": "【重要】必须使用字段名'category'，不能使用其他名称。编程语言的类别"
            },
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "【重要】必须使用字段名'features'，不能使用其他名称。编程语言的主要特性列表"
            },
            "difficulty": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
                "description": "【重要】必须使用字段名'difficulty'，值必须是easy、medium或hard之一"
            }
        },
        "required": ["name", "category", "features", "difficulty"],
        "additionalProperties": False
    }
    
    print("\n" + "="*60)
    print("🔍 测试严格的 Schema:")
    print(json.dumps(strict_schema, indent=2, ensure_ascii=False))
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "请严格按照指定的JSON格式介绍Python编程语言，必须使用指定的字段名"}
            ],
            max_tokens=200,
            structured_provider="agently",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "programming_language_info",
                    "schema": strict_schema
                }
            }
        )
        
        print("\n✅ 严格 Schema 结果:")
        result = json.loads(response.choices[0].message.content)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"字段列表: {list(result.keys())}")
        
        return result
        
    except Exception as e:
        print(f"❌ 严格 Schema 测试失败: {e}")
        return None

def analyze_results(original, improved, strict):
    """分析测试结果"""
    print("\n" + "="*60)
    print("📊 结果分析:")
    
    schemas = [
        ("原始 Schema", original),
        ("改进 Schema", improved), 
        ("严格 Schema", strict)
    ]
    
    for name, result in schemas:
        if result:
            print(f"\n{name}:")
            print(f"  - 字段数量: {len(result)}")
            print(f"  - 包含 'name' 字段: {'name' in result}")
            print(f"  - 包含 'category' 字段: {'category' in result}")
            print(f"  - 包含 'features' 字段: {'features' in result}")
            print(f"  - 包含 'difficulty' 字段: {'difficulty' in result}")
            print(f"  - 所有字段: {list(result.keys())}")
        else:
            print(f"\n{name}: 测试失败")

if __name__ == "__main__":
    print("🚀 开始实际测试 Agently 结构化输出的准确性")
    
    # 测试三种不同的 schema
    original_result = test_original_schema()
    time.sleep(2)  # 避免请求过于频繁
    
    improved_result = test_improved_schema()
    time.sleep(2)
    
    strict_result = test_strict_schema()
    
    # 分析结果
    analyze_results(original_result, improved_result, strict_result)