#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试agently流式结构化输出功能
验证HarborAI的流式结构化输出是否正确实现
"""

import json
import sys
import os
import time
import asyncio

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

def test_sync_streaming_structured_output():
    """测试同步流式结构化输出"""
    print("🔄 测试同步流式结构化输出")
    
    client = HarborAI()
    
    # 创建测试schema
    schema = {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "详细的文本分析"
            },
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
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "关键词列表"
            }
        },
        "required": ["analysis", "sentiment", "confidence", "keywords"]
    }
    
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "text_analysis",
            "schema": schema,
            "strict": True
        }
    }
    
    start_time = time.time()
    
    # 使用流式调用
    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "请详细分析这段文本：'今天天气很好，心情不错，准备出去散步，享受美好的阳光。'"
            }
        ],
        response_format=response_format,
        structured_provider="agently",  # 使用agently进行流式结构化输出
        stream=True,
        temperature=0.1,
        max_tokens=1000
    )
    
    print("🔍 流式响应数据:")
    chunks_received = 0
    final_result = None
    
    for chunk in stream:
        chunks_received += 1
        print(f"   Chunk {chunks_received}: {chunk}")
        
        # 检查是否有parsed字段
        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'parsed'):
            parsed_data = chunk.choices[0].delta.parsed
            if parsed_data:
                print(f"   📊 解析数据: {parsed_data}")
                final_result = parsed_data
        
        # 检查是否有完整的message.parsed
        if hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'parsed'):
            parsed_data = chunk.choices[0].message.parsed
            if parsed_data:
                print(f"   📊 完整解析数据: {parsed_data}")
                final_result = parsed_data
    
    end_time = time.time()
    
    print(f"✅ 同步流式调用完成，耗时: {end_time - start_time:.2f}秒")
    print(f"   接收到 {chunks_received} 个数据块")
    print(f"   最终结果: {final_result}")
    
    # 验证结果
    assert final_result is not None, "流式调用没有返回解析结果"
    assert "analysis" in final_result, "缺少analysis字段"
    assert "sentiment" in final_result, "缺少sentiment字段"
    assert "confidence" in final_result, "缺少confidence字段"
    assert "keywords" in final_result, "缺少keywords字段"
    
    return final_result

async def test_async_streaming_structured_output():
    """测试异步流式结构化输出"""
    print("🔄 测试异步流式结构化输出")
    
    client = HarborAI()
    
    # 创建测试schema
    schema = {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "详细的文本分析"
            },
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
            },
            "keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "关键词列表"
            }
        },
        "required": ["analysis", "sentiment", "confidence", "keywords"]
    }
    
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "text_analysis",
            "schema": schema,
            "strict": True
        }
    }
    
    start_time = time.time()
    
    # 使用异步流式调用
    stream = await client.chat.completions.acreate(
        model="deepseek-chat",
        messages=[
            {
                "role": "user",
                "content": "请详细分析这段文本：'今天天气很好，心情不错，准备出去散步，享受美好的阳光。'"
            }
        ],
        response_format=response_format,
        structured_provider="agently",  # 使用agently进行流式结构化输出
        stream=True,
        temperature=0.1,
        max_tokens=1000
    )
    
    print("🔍 异步流式响应数据:")
    chunks_received = 0
    final_result = None
    
    async for chunk in stream:
        chunks_received += 1
        print(f"   Async Chunk {chunks_received}: {chunk}")
        
        # 检查是否有parsed字段
        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'parsed'):
            parsed_data = chunk.choices[0].delta.parsed
            if parsed_data:
                print(f"   📊 异步解析数据: {parsed_data}")
                final_result = parsed_data
        
        # 检查是否有完整的message.parsed
        if hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'parsed'):
            parsed_data = chunk.choices[0].message.parsed
            if parsed_data:
                print(f"   📊 异步完整解析数据: {parsed_data}")
                final_result = parsed_data
    
    end_time = time.time()
    
    print(f"✅ 异步流式调用完成，耗时: {end_time - start_time:.2f}秒")
    print(f"   接收到 {chunks_received} 个数据块")
    print(f"   最终结果: {final_result}")
    
    # 验证结果
    assert final_result is not None, "异步流式调用没有返回解析结果"
    assert "analysis" in final_result, "缺少analysis字段"
    assert "sentiment" in final_result, "缺少sentiment字段"
    assert "confidence" in final_result, "缺少confidence字段"
    assert "keywords" in final_result, "缺少keywords字段"
    
    return final_result

def test_streaming_vs_non_streaming():
    """比较流式和非流式结构化输出的结果"""
    print("🔄 测试流式vs非流式结构化输出")
    
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
    
    test_message = "今天天气很好，心情不错"
    
    # 非流式调用
    print("   📝 非流式调用...")
    start_time = time.time()
    non_stream_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": f"请分析这段文本的情感：'{test_message}'"}],
        response_format=response_format,
        structured_provider="agently",
        stream=False,
        temperature=0.1,
        max_tokens=500
    )
    non_stream_time = time.time() - start_time
    non_stream_result = non_stream_response.choices[0].message.parsed
    
    # 流式调用
    print("   🌊 流式调用...")
    start_time = time.time()
    stream_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": f"请分析这段文本的情感：'{test_message}'"}],
        response_format=response_format,
        structured_provider="agently",
        stream=True,
        temperature=0.1,
        max_tokens=500
    )
    
    stream_result = None
    for chunk in stream_response:
        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'parsed'):
            parsed_data = chunk.choices[0].delta.parsed
            if parsed_data:
                stream_result = parsed_data
        if hasattr(chunk.choices[0], 'message') and hasattr(chunk.choices[0].message, 'parsed'):
            parsed_data = chunk.choices[0].message.parsed
            if parsed_data:
                stream_result = parsed_data
    
    stream_time = time.time() - start_time
    
    print(f"📊 结果比较:")
    print(f"   非流式结果: {non_stream_result} (耗时: {non_stream_time:.2f}秒)")
    print(f"   流式结果: {stream_result} (耗时: {stream_time:.2f}秒)")
    
    # 验证结果结构一致性
    assert type(non_stream_result) == type(stream_result), "结果类型不一致"
    assert set(non_stream_result.keys()) == set(stream_result.keys()), "结果字段不一致"
    
    return non_stream_result, stream_result

def test_sync_streaming_only():
    """仅测试同步流式结构化输出"""
    print("🧪 测试同步流式结构化输出功能")
    print("=" * 60)
    
    try:
        # 测试同步流式
        print("\n1️⃣ 测试同步流式结构化输出")
        sync_result = test_sync_streaming_structured_output()
        
        # 测试流式vs非流式
        print("\n2️⃣ 测试流式vs非流式结构化输出")
        non_stream_result, stream_result = test_streaming_vs_non_streaming()
        
        print("\n📊 同步测试结果:")
        print(f"同步流式结果: {sync_result}")
        print(f"非流式结果: {non_stream_result}")
        print(f"流式结果: {stream_result}")
        
        print("✅ 同步流式结构化输出测试通过")
        print("✅ 流式和非流式结果结构一致")
        
        return True
        
    except Exception as e:
        print(f"❌ 同步流式结构化输出测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_streaming_only():
    """仅测试异步流式结构化输出"""
    print("🧪 测试异步流式结构化输出功能")
    print("=" * 60)
    
    try:
        # 测试异步流式
        print("\n1️⃣ 测试异步流式结构化输出")
        async_result = await test_async_streaming_structured_output()
        
        print("\n📊 异步测试结果:")
        print(f"异步流式结果: {async_result}")
        
        print("✅ 异步流式结构化输出测试通过")
        
        return True
        
    except Exception as e:
        print(f"❌ 异步流式结构化输出测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 先测试同步功能
    print("🔄 开始同步流式测试...")
    sync_success = test_sync_streaming_only()
    
    # 再测试异步功能
    print("\n🔄 开始异步流式测试...")
    async_success = asyncio.run(test_async_streaming_only())
    
    if sync_success and async_success:
        print("\n🎉 agently流式结构化输出功能验证通过")
        print("✅ 支持同步和异步流式结构化输出")
        print("✅ 流式输出与非流式输出结果一致")
        sys.exit(0)
    else:
        print("\n🚨 agently流式结构化输出功能测试失败")
        sys.exit(1)