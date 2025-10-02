#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试 Agently 结构化输出转换过程
检查 JSON Schema 转换为 Agently 元组语法时是否存在参数偏差
"""

import json
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harborai.api.structured import StructuredOutputHandler

def test_schema_conversion():
    """测试 JSON Schema 到 Agently 格式的转换"""
    
    # 测试中使用的原始 schema
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
    
    print("🔍 原始 JSON Schema:")
    print(json.dumps(original_schema, indent=2, ensure_ascii=False))
    
    # 创建结构化输出处理器
    handler = StructuredOutputHandler()
    
    # 转换为 Agently 格式
    schema_wrapper = {
        "json_schema": {
            "schema": original_schema
        }
    }
    
    agently_format = handler._convert_json_schema_to_agently_output(schema_wrapper)
    
    print("\n🔄 转换后的 Agently 格式:")
    print(json.dumps(agently_format, indent=2, ensure_ascii=False))
    
    # 分析转换结果
    print("\n📊 转换分析:")
    for field_name, field_def in agently_format.items():
        if isinstance(field_def, tuple):
            field_type, field_desc = field_def
            print(f"  - {field_name}: ({field_type}, '{field_desc}')")
        elif isinstance(field_def, list):
            print(f"  - {field_name}: {field_def} (数组类型)")
        else:
            print(f"  - {field_name}: {field_def} (其他类型)")
    
    return agently_format

def test_improved_schema():
    """测试改进的 schema，使描述更加明确"""
    
    # 改进的 schema，描述更加明确和具体
    improved_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "编程语言的名称，例如：Python、Java、JavaScript等"
            },
            "category": {
                "type": "string", 
                "description": "编程语言的类别，例如：解释型语言、编译型语言、脚本语言等"
            },
            "features": {
                "type": "array",
                "items": {"type": "string"},
                "description": "编程语言的主要特性和优点列表"
            },
            "difficulty": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
                "description": "学习难度等级，必须是以下值之一：easy（简单）、medium（中等）、hard（困难）"
            }
        },
        "required": ["name", "category", "features", "difficulty"]
    }
    
    print("\n" + "="*60)
    print("🔍 改进的 JSON Schema:")
    print(json.dumps(improved_schema, indent=2, ensure_ascii=False))
    
    # 创建结构化输出处理器
    handler = StructuredOutputHandler()
    
    # 转换为 Agently 格式
    schema_wrapper = {
        "json_schema": {
            "schema": improved_schema
        }
    }
    
    agently_format = handler._convert_json_schema_to_agently_output(schema_wrapper)
    
    print("\n🔄 改进后转换的 Agently 格式:")
    print(json.dumps(agently_format, indent=2, ensure_ascii=False))
    
    # 分析转换结果
    print("\n📊 改进后转换分析:")
    for field_name, field_def in agently_format.items():
        if isinstance(field_def, tuple):
            field_type, field_desc = field_def
            print(f"  - {field_name}: ({field_type}, '{field_desc}')")
        elif isinstance(field_def, list):
            print(f"  - {field_name}: {field_def} (数组类型)")
        else:
            print(f"  - {field_name}: {field_def} (其他类型)")
    
    return agently_format

def test_agently_direct_format():
    """测试直接使用 Agently 原生格式"""
    
    # 直接使用 Agently 原生格式
    agently_native = {
        "name": ("str", "编程语言的确切名称，如Python"),
        "category": ("str", "编程语言类别，如解释型高级编程语言"),
        "features": [("str", "编程语言的一个主要特性")],
        "difficulty": ("str", "学习难度：easy/medium/hard")
    }
    
    print("\n" + "="*60)
    print("🎯 Agently 原生格式:")
    print(json.dumps(agently_native, indent=2, ensure_ascii=False))
    
    return agently_native

if __name__ == "__main__":
    print("🚀 开始调试 Agently 结构化输出转换过程")
    
    # 测试原始转换
    original_agently = test_schema_conversion()
    
    # 测试改进的转换
    improved_agently = test_improved_schema()
    
    # 测试原生格式
    native_agently = test_agently_direct_format()
    
    print("\n" + "="*60)
    print("📋 总结对比:")
    print("原始转换和改进转换的主要区别在于描述的详细程度")
    print("Agently 原生格式提供了最直接的控制方式")