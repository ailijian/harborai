#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试JSON Schema到Agently语法的转换映射
验证转换是否符合Agently结构化输出语法设计理念
"""

import json
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

from harborai.api.structured import StructuredOutputHandler

def test_schema_conversion():
    """测试JSON Schema到Agently格式的转换"""
    print("🧪 测试JSON Schema到Agently语法的转换映射")
    print("=" * 60)
    
    handler = StructuredOutputHandler()
    
    # 测试用例1：简单对象
    print("\n📋 测试用例1：简单对象")
    simple_schema = {
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
                "maximum": 1,
                "description": "置信度分数"
            }
        },
        "required": ["sentiment", "confidence"]
    }
    
    print(f"原始JSON Schema:")
    print(json.dumps(simple_schema, ensure_ascii=False, indent=2))
    
    agently_format = handler._convert_json_schema_to_agently_output({
        "json_schema": {"schema": simple_schema}
    })
    
    print(f"\n转换后的Agently格式:")
    print(json.dumps(agently_format, ensure_ascii=False, indent=2))
    
    # 验证转换结果
    assert "sentiment" in agently_format
    assert "confidence" in agently_format
    assert agently_format["sentiment"] == ("str", "情感倾向分析")
    assert agently_format["confidence"] == ("int", "置信度分数")  # number转为int
    print("✅ 简单对象转换正确")
    
    # 测试用例2：复杂嵌套对象
    print("\n📋 测试用例2：复杂嵌套对象")
    complex_schema = {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "对输入内容的分析结果"
            },
            "keywords": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "关键词列表"
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "检测到的语言"
                    },
                    "word_count": {
                        "type": "integer",
                        "description": "词汇数量"
                    }
                },
                "required": ["language", "word_count"]
            }
        },
        "required": ["analysis", "keywords", "metadata"]
    }
    
    print(f"原始JSON Schema:")
    print(json.dumps(complex_schema, ensure_ascii=False, indent=2))
    
    agently_format = handler._convert_json_schema_to_agently_output({
        "json_schema": {"schema": complex_schema}
    })
    
    print(f"\n转换后的Agently格式:")
    print(json.dumps(agently_format, ensure_ascii=False, indent=2))
    
    # 验证转换结果
    assert "analysis" in agently_format
    assert "keywords" in agently_format
    assert "metadata" in agently_format
    
    # 验证字符串字段
    assert agently_format["analysis"] == ("str", "对输入内容的分析结果")
    
    # 验证数组字段
    assert isinstance(agently_format["keywords"], list)
    assert agently_format["keywords"] == [("str", "关键词列表")]
    
    # 验证嵌套对象
    assert isinstance(agently_format["metadata"], dict)
    assert "language" in agently_format["metadata"]
    assert "word_count" in agently_format["metadata"]
    assert agently_format["metadata"]["language"] == ("str", "检测到的语言")
    assert agently_format["metadata"]["word_count"] == ("int", "词汇数量")
    
    print("✅ 复杂嵌套对象转换正确")
    
    # 测试用例3：对象数组
    print("\n📋 测试用例3：对象数组")
    array_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "项目名称"
                        },
                        "value": {
                            "type": "number",
                            "description": "项目值"
                        }
                    },
                    "required": ["name", "value"]
                },
                "description": "项目列表"
            }
        },
        "required": ["items"]
    }
    
    print(f"原始JSON Schema:")
    print(json.dumps(array_schema, ensure_ascii=False, indent=2))
    
    agently_format = handler._convert_json_schema_to_agently_output({
        "json_schema": {"schema": array_schema}
    })
    
    print(f"\n转换后的Agently格式:")
    print(json.dumps(agently_format, ensure_ascii=False, indent=2))
    
    # 验证转换结果
    assert "items" in agently_format
    assert isinstance(agently_format["items"], list)
    assert len(agently_format["items"]) == 1
    
    # 验证数组中的对象结构
    array_item = agently_format["items"][0]
    assert isinstance(array_item, dict)
    assert "name" in array_item
    assert "value" in array_item
    assert array_item["name"] == ("str", "项目名称")
    assert array_item["value"] == ("int", "项目值")
    
    print("✅ 对象数组转换正确")
    
    print("\n🎉 所有转换测试通过！")
    print("JSON Schema到Agently语法的转换映射实现正确")
    
    return True

if __name__ == "__main__":
    try:
        test_schema_conversion()
        print("\n✅ 测试完成，转换映射功能正常")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)