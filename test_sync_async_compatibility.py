#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试agently同步异步调用与HarborAI客户端的兼容性
验证同步和异步方法都能正确处理结构化输出
"""

import asyncio
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

def test_sync_structured_output():
    """测试同步结构化输出"""
    print("🔄 测试同步结构化输出")
    
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
    
    # 同步调用
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "请分析这段文本的情感：'今天天气很好，心情不错'"
            }
        ],
        response_format=response_format,
        structured_provider="agently",
        temperature=0.1,
        max_tokens=500
    )
    
    end_time = time.time()
    
    # 验证响应
    assert response is not None, "同步调用响应为空"
    assert hasattr(response.choices[0].message, 'parsed'), "同步调用缺少parsed字段"
    
    parsed_data = response.choices[0].message.parsed
    assert parsed_data is not None, "同步调用解析结果为空"
    assert "sentiment" in parsed_data, "同步调用缺少sentiment字段"
    assert "confidence" in parsed_data, "同步调用缺少confidence字段"
    
    print(f"✅ 同步调用成功，耗时: {end_time - start_time:.2f}秒")
    print(f"   解析结果: {parsed_data}")
    
    return parsed_data

async def test_async_structured_output():
    """测试异步结构化输出"""
    print("🔄 测试异步结构化输出")
    
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
    
    # 异步调用
    response = await client.chat.completions.acreate(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "请分析这段文本的情感：'今天天气很好，心情不错'"
            }
        ],
        response_format=response_format,
        structured_provider="agently",
        temperature=0.1,
        max_tokens=500
    )
    
    end_time = time.time()
    
    # 验证响应
    assert response is not None, "异步调用响应为空"
    assert hasattr(response.choices[0].message, 'parsed'), "异步调用缺少parsed字段"
    
    parsed_data = response.choices[0].message.parsed
    assert parsed_data is not None, "异步调用解析结果为空"
    assert "sentiment" in parsed_data, "异步调用缺少sentiment字段"
    assert "confidence" in parsed_data, "异步调用缺少confidence字段"
    
    print(f"✅ 异步调用成功，耗时: {end_time - start_time:.2f}秒")
    print(f"   解析结果: {parsed_data}")
    
    return parsed_data

def test_compatibility():
    """测试同步异步兼容性"""
    print("🧪 测试agently同步异步调用与HarborAI客户端的兼容性")
    print("=" * 60)
    
    try:
        # 测试同步调用
        sync_result = test_sync_structured_output()
        
        # 测试异步调用
        async_result = asyncio.run(test_async_structured_output())
        
        # 比较结果结构
        print("\n📊 结果比较:")
        print(f"同步结果: {sync_result}")
        print(f"异步结果: {async_result}")
        
        # 验证结果结构一致性
        assert type(sync_result) == type(async_result), "同步异步结果类型不一致"
        assert set(sync_result.keys()) == set(async_result.keys()), "同步异步结果字段不一致"
        
        print("✅ 同步异步调用兼容性验证通过")
        print("✅ 结果结构一致，字段完整")
        
        return True
        
    except Exception as e:
        print(f"❌ 兼容性测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_compatibility()
    if success:
        print("\n🎉 agently同步异步调用与HarborAI客户端完全兼容")
        sys.exit(0)
    else:
        print("\n🚨 兼容性测试失败，需要进一步调试")
        sys.exit(1)