#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
专门测试豆包模型的结构化输出功能
重点验证Native和Agently两种解析方式
"""

import os
import sys
import json
import time
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# 加载环境变量
try:
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"已加载环境变量文件: {env_path}")
    else:
        print(f"环境变量文件不存在: {env_path}")
except ImportError:
    print("python-dotenv未安装，直接使用环境变量")

from harborai import HarborAI

def create_sentiment_analysis_schema() -> Dict[str, Any]:
    """创建情感分析的JSON Schema定义"""
    return {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "description": "情感倾向分析结果",
                "enum": ["positive", "negative", "neutral"]
            },
            "confidence": {
                "type": "number",
                "description": "置信度分数，范围0-1",
                "minimum": 0,
                "maximum": 1
            }
        },
        "required": ["sentiment", "confidence"],
        "additionalProperties": False
    }

def test_doubao_model(model_name: str, model_display_name: str, is_reasoning: bool = False):
    """测试单个豆包模型的结构化输出功能"""
    print(f"\n{'='*80}")
    print(f"测试豆包模型: {model_display_name} ({model_name})")
    print(f"模型类型: {'推理模型' if is_reasoning else '非推理模型'}")
    print(f"{'='*80}")
    
    # 创建HarborAI客户端
    client = HarborAI()
    
    # 创建schema
    schema = create_sentiment_analysis_schema()
    test_content = "今天天气真好"
    
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "SentimentAnalysis",
            "schema": schema,
            "strict": True
        }
    }
    
    messages = [
        {
            "role": "system",
            "content": "你是一个情感分析专家。请分析用户输入的文本情感，并以JSON格式返回结果。输出格式示例：{\"sentiment\": \"positive\", \"confidence\": 0.95}"
        },
        {
            "role": "user", 
            "content": f"分析这句话的情感：{test_content}"
        }
    ]
    
    # 测试1: Native结构化输出
    print(f"\n🔍 测试1: Native结构化输出 (structured_provider='native')")
    try:
        start_time = time.time()
        native_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format=response_format,
            structured_provider="native",
            temperature=0.1,
            max_tokens=500
        )
        native_time = time.time() - start_time
        
        # 验证响应结构
        assert native_response is not None, "Native响应为空"
        assert hasattr(native_response, 'choices'), "Native响应缺少choices字段"
        assert len(native_response.choices) > 0, "Native响应choices为空"
        
        choice = native_response.choices[0]
        message = choice.message
        
        # 验证parsed字段
        assert hasattr(message, 'parsed'), "Native响应缺少parsed字段"
        native_result = message.parsed
        assert native_result is not None, "Native解析结果为空"
        
        # 验证数据结构
        assert "sentiment" in native_result, "Native结果缺少sentiment字段"
        assert "confidence" in native_result, "Native结果缺少confidence字段"
        assert isinstance(native_result["sentiment"], str), "sentiment字段类型错误"
        assert isinstance(native_result["confidence"], (int, float)), "confidence字段类型错误"
        assert native_result["sentiment"] in ["positive", "negative", "neutral"], f"sentiment值不合法: {native_result['sentiment']}"
        assert 0 <= native_result["confidence"] <= 1, f"confidence值超出范围: {native_result['confidence']}"
        
        print(f"✅ Native解析成功")
        print(f"   延迟: {round(native_time * 1000, 2)}ms")
        print(f"   解析结果: {native_result}")
        if hasattr(message, 'content') and message.content:
            print(f"   原始内容: {message.content}")
        if is_reasoning and hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"   思考过程: {message.reasoning_content[:200]}...")
            
    except Exception as e:
        print(f"❌ Native解析失败: {str(e)}")
        native_result = None
        native_time = 0
    
    # 测试2: Agently结构化输出
    print(f"\n🔍 测试2: Agently结构化输出 (structured_provider='agently')")
    try:
        start_time = time.time()
        agently_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format=response_format,
            structured_provider="agently",
            temperature=0.1,
            max_tokens=500
        )
        agently_time = time.time() - start_time
        
        # 验证响应结构
        assert agently_response is not None, "Agently响应为空"
        assert hasattr(agently_response, 'choices'), "Agently响应缺少choices字段"
        assert len(agently_response.choices) > 0, "Agently响应choices为空"
        
        choice = agently_response.choices[0]
        message = choice.message
        
        # 验证parsed字段
        assert hasattr(message, 'parsed'), "Agently响应缺少parsed字段"
        agently_result = message.parsed
        assert agently_result is not None, "Agently解析结果为空"
        
        # 验证数据结构
        assert "sentiment" in agently_result, "Agently结果缺少sentiment字段"
        assert "confidence" in agently_result, "Agently结果缺少confidence字段"
        assert isinstance(agently_result["sentiment"], str), "sentiment字段类型错误"
        assert isinstance(agently_result["confidence"], (int, float)), "confidence字段类型错误"
        assert agently_result["sentiment"] in ["positive", "negative", "neutral"], f"sentiment值不合法: {agently_result['sentiment']}"
        assert 0 <= agently_result["confidence"] <= 1, f"confidence值超出范围: {agently_result['confidence']}"
        
        print(f"✅ Agently解析成功")
        print(f"   延迟: {round(agently_time * 1000, 2)}ms")
        print(f"   解析结果: {agently_result}")
        if hasattr(message, 'content') and message.content:
            print(f"   原始内容: {message.content}")
        if is_reasoning and hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"   思考过程: {message.reasoning_content[:200]}...")
            
    except Exception as e:
        print(f"❌ Agently解析失败: {str(e)}")
        agently_result = None
        agently_time = 0
    
    # 对比分析
    print(f"\n📊 对比分析:")
    if native_result and agently_result:
        print(f"   Native结果:  {native_result} (耗时: {round(native_time * 1000, 2)}ms)")
        print(f"   Agently结果: {agently_result} (耗时: {round(agently_time * 1000, 2)}ms)")
        
        # 性能对比
        if native_time > 0 and agently_time > 0:
            speed_ratio = agently_time / native_time
            print(f"   性能对比: Agently耗时是Native的 {round(speed_ratio, 2)}倍")
        
        # 结果一致性检查
        sentiment_match = native_result["sentiment"] == agently_result["sentiment"]
        confidence_diff = abs(native_result["confidence"] - agently_result["confidence"])
        
        print(f"   情感一致性: {'✅ 一致' if sentiment_match else '❌ 不一致'}")
        print(f"   置信度差异: {round(confidence_diff, 3)}")
        
        return True
    elif native_result:
        print(f"   仅Native成功: {native_result}")
        return False
    elif agently_result:
        print(f"   仅Agently成功: {agently_result}")
        return False
    else:
        print(f"   ❌ 两种方式都失败")
        return False

def main():
    """主测试函数"""
    print("豆包模型结构化输出专项测试")
    print("="*80)
    
    # 检查环境变量
    if not os.getenv("DOUBAO_API_KEY") or not os.getenv("DOUBAO_BASE_URL"):
        print("❌ 缺少豆包API配置，请检查DOUBAO_API_KEY和DOUBAO_BASE_URL环境变量")
        return
    
    print("✅ 环境变量检查通过")
    
    # 豆包模型列表
    doubao_models = [
        {
            "model": "doubao-1-5-pro-32k-character-250715",
            "name": "豆包 1.5 Pro 32K",
            "is_reasoning": False
        },
        {
            "model": "doubao-seed-1-6-250615", 
            "name": "豆包 Seed 1.6",
            "is_reasoning": True
        }
    ]
    
    success_count = 0
    total_count = len(doubao_models)
    
    # 测试每个豆包模型
    for model_config in doubao_models:
        try:
            success = test_doubao_model(
                model_config["model"],
                model_config["name"],
                model_config["is_reasoning"]
            )
            if success:
                success_count += 1
        except Exception as e:
            print(f"❌ 模型 {model_config['name']} 测试异常: {str(e)}")
    
    # 测试总结
    print(f"\n{'='*80}")
    print(f"豆包模型结构化输出测试总结:")
    print(f"   总计模型: {total_count}")
    print(f"   成功测试: {success_count}")
    print(f"   失败测试: {total_count - success_count}")
    print(f"   成功率: {round(success_count / total_count * 100, 1)}%")
    
    if success_count == total_count:
        print("🎉 所有豆包模型测试通过！Native和Agently结构化输出功能正常工作")
    else:
        print("⚠️  部分豆包模型测试失败，请检查失败的模型配置")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main()