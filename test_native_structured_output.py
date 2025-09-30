#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试通过扩展参数指定native结构化输出的功能
验证HarborAI默认使用agently，但可以通过structured_provider参数指定native
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

def test_default_agently_provider():
    """测试默认使用agently作为结构化输出提供者"""
    print("🔄 测试默认agently结构化输出")
    
    client = HarborAI()
    
    # 创建测试schema
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
                "maximum": 1,
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
    
    start_time = time.time()
    
    # 不指定structured_provider，应该默认使用agently
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "请分析这段文本的情感：'今天天气很好，心情不错'"
            }
        ],
        response_format=response_format,
        # 不指定structured_provider，测试默认行为
        temperature=0.1,
        max_tokens=500
    )
    
    end_time = time.time()
    
    # 验证响应
    assert response is not None, "默认agently调用响应为空"
    assert hasattr(response.choices[0].message, 'parsed'), "默认agently调用缺少parsed字段"
    
    parsed_data = response.choices[0].message.parsed
    assert parsed_data is not None, "默认agently调用解析结果为空"
    assert "sentiment" in parsed_data, "默认agently调用缺少sentiment字段"
    assert "confidence" in parsed_data, "默认agently调用缺少confidence字段"
    
    print(f"✅ 默认agently调用成功，耗时: {end_time - start_time:.2f}秒")
    print(f"   解析结果: {parsed_data}")
    
    return parsed_data

def test_explicit_agently_provider():
    """测试显式指定agently作为结构化输出提供者"""
    print("🔄 测试显式指定agently结构化输出")
    
    client = HarborAI()
    
    # 创建测试schema
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
                "maximum": 1,
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
    
    start_time = time.time()
    
    # 显式指定structured_provider为agently
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "请分析这段文本的情感：'今天天气很好，心情不错'"
            }
        ],
        response_format=response_format,
        structured_provider="agently",  # 显式指定agently
        temperature=0.1,
        max_tokens=500
    )
    
    end_time = time.time()
    
    # 验证响应
    assert response is not None, "显式agently调用响应为空"
    assert hasattr(response.choices[0].message, 'parsed'), "显式agently调用缺少parsed字段"
    
    parsed_data = response.choices[0].message.parsed
    assert parsed_data is not None, "显式agently调用解析结果为空"
    assert "sentiment" in parsed_data, "显式agently调用缺少sentiment字段"
    assert "confidence" in parsed_data, "显式agently调用缺少confidence字段"
    
    print(f"✅ 显式agently调用成功，耗时: {end_time - start_time:.2f}秒")
    print(f"   解析结果: {parsed_data}")
    
    return parsed_data

def test_native_provider():
    """测试指定native作为结构化输出提供者"""
    print("🔄 测试native结构化输出")
    
    client = HarborAI()
    
    # 创建测试schema
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
                "maximum": 1,
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
    
    start_time = time.time()
    
    # 指定structured_provider为native
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "请分析这段文本的情感：'今天天气很好，心情不错'。请严格按照JSON格式返回结果，包含sentiment和confidence字段。"
            }
        ],
        response_format=response_format,
        structured_provider="native",  # 指定native
        temperature=0.1,
        max_tokens=500
    )
    
    end_time = time.time()
    
    # 添加调试信息
    print(f"🔍 Native模式调试信息:")
    print(f"   响应对象: {response}")
    print(f"   响应类型: {type(response)}")
    if hasattr(response, 'choices') and response.choices:
        print(f"   choices[0]: {response.choices[0]}")
        print(f"   message: {response.choices[0].message}")
        print(f"   content: {response.choices[0].message.content}")
        if hasattr(response.choices[0].message, 'parsed'):
            print(f"   parsed: {response.choices[0].message.parsed}")
        else:
            print("   ❌ 没有parsed字段")
    
    # 验证响应
    assert response is not None, "native调用响应为空"
    assert hasattr(response.choices[0].message, 'parsed'), "native调用缺少parsed字段"
    
    parsed_data = response.choices[0].message.parsed
    assert parsed_data is not None, "native调用解析结果为空"
    assert "sentiment" in parsed_data, "native调用缺少sentiment字段"
    assert "confidence" in parsed_data, "native调用缺少confidence字段"
    
    print(f"✅ native调用成功，耗时: {end_time - start_time:.2f}秒")
    print(f"   解析结果: {parsed_data}")
    
    return parsed_data

def test_native_structured_output():
    """测试native结构化输出功能"""
    print("🧪 测试通过扩展参数指定native结构化输出的功能")
    print("=" * 60)
    
    try:
        # 测试默认agently
        print("\n1️⃣ 测试默认行为（应该使用agently）")
        default_result = test_default_agently_provider()
        
        # 测试显式agently
        print("\n2️⃣ 测试显式指定agently")
        explicit_agently_result = test_explicit_agently_provider()
        
        # 测试native
        print("\n3️⃣ 测试指定native")
        native_result = test_native_provider()
        
        # 比较结果
        print("\n📊 结果比较:")
        print(f"默认结果（agently）: {default_result}")
        print(f"显式agently结果: {explicit_agently_result}")
        print(f"native结果: {native_result}")
        
        # 验证结果结构一致性
        assert type(default_result) == type(explicit_agently_result) == type(native_result), "结果类型不一致"
        assert set(default_result.keys()) == set(explicit_agently_result.keys()) == set(native_result.keys()), "结果字段不一致"
        
        print("✅ 所有结构化输出方式都正常工作")
        print("✅ 默认使用agently，可通过structured_provider参数指定native")
        print("✅ 结果结构一致，字段完整")
        
        return True
        
    except Exception as e:
        print(f"❌ native结构化输出测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_native_structured_output()
    if success:
        print("\n🎉 native结构化输出功能验证通过")
        print("✅ HarborAI默认使用agently结构化输出")
        print("✅ 开发者可以通过structured_provider参数指定native结构化输出")
        sys.exit(0)
    else:
        print("\n🚨 native结构化输出功能测试失败")
        sys.exit(1)