#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端测试用例008：Native结构化输出功能测试

基于HarborAI端到端测试方案.md第366-415行的内容，验证指定使用厂商原生schema的结构化输出功能。
该测试用例专注于验证单个特定功能，确保测试用例简洁明确且可独立执行。

测试目标：
1. 验证structured_provider="native"参数生效
2. 验证所有7个模型的原生结构化输出能力
3. 测试response.parsed字段和数据结构正确性
4. 验证与Agently解析结果的对比
5. 确保性能和准确性符合预期

测试场景：情感分析 - 分析"今天天气真好"的情感倾向和置信度
"""

import os
import sys
import json
import time
import pytest
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
from harborai.utils.logger import get_logger

logger = get_logger(__name__)

@pytest.fixture(scope='session')
def client() -> HarborAI:
    """创建HarborAI客户端实例"""
    return HarborAI()

def create_sentiment_analysis_schema() -> Dict[str, Any]:
    """创建情感分析的JSON Schema定义。
    
    根据测试方案E2E-008，定义情感分析的结构化输出schema。
    包含sentiment（情感倾向）和confidence（置信度）两个字段。
    """
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

def get_test_models() -> List[Dict[str, str]]:
    """获取要测试的模型列表。
    
    根据测试可用模型列表.md，返回所有7个模型的配置信息。
    """
    return [
        # DeepSeek模型
        {"model": "deepseek-chat", "provider": "deepseek", "name": "DeepSeek Chat", "is_reasoning": False},
        {"model": "deepseek-reasoner", "provider": "deepseek", "name": "DeepSeek Reasoner", "is_reasoning": True},
        
        # 文心一言模型
        {"model": "ernie-3.5-8k", "provider": "ernie", "name": "文心一言 3.5", "is_reasoning": False},
        {"model": "ernie-4.0-turbo-8k", "provider": "ernie", "name": "文心一言 4.0 Turbo", "is_reasoning": False},
        {"model": "ernie-x1-turbo-32k", "provider": "ernie", "name": "文心一言 X1 Turbo", "is_reasoning": True},
        
        # 豆包模型 - 仅测试支持原生结构化输出的1.6版本
        {"model": "doubao-seed-1-6-250615", "provider": "doubao", "name": "豆包 Seed 1.6", "is_reasoning": True}
    ]

def validate_sentiment_result(parsed_data: Dict[str, Any], model: str) -> None:
    """验证情感分析结果的正确性。
    
    Args:
        parsed_data: 解析后的结构化数据
        model: 模型名称
    """
    # 验证必填字段存在
    assert "sentiment" in parsed_data, f"模型 {model} 输出缺少sentiment字段"
    assert "confidence" in parsed_data, f"模型 {model} 输出缺少confidence字段"
    
    # 验证数据类型
    assert isinstance(parsed_data["sentiment"], str), f"模型 {model} sentiment字段类型错误"
    assert isinstance(parsed_data["confidence"], (int, float)), f"模型 {model} confidence字段类型错误"
    
    # 验证数据范围
    assert parsed_data["sentiment"] in ["positive", "negative", "neutral"], \
        f"模型 {model} sentiment值不在允许范围内: {parsed_data['sentiment']}"
    assert 0 <= parsed_data["confidence"] <= 1, \
        f"模型 {model} confidence值超出范围[0,1]: {parsed_data['confidence']}"

@pytest.mark.parametrize("model_config", get_test_models())
def test_native_structured_output_with_model(client: HarborAI, model_config: Dict[str, str]):
    """测试单个模型的Native结构化输出功能。
    
    Args:
        client: HarborAI客户端实例
        model_config: 模型配置信息
    """
    schema = create_sentiment_analysis_schema()
    model = model_config["model"]
    provider = model_config["provider"]
    name = model_config["name"]
    is_reasoning = model_config["is_reasoning"]
    
    print(f"\n测试模型: {name} ({model}) - {'推理模型' if is_reasoning else '非推理模型'}")
    
    try:
        # 创建response_format，使用Native结构化输出
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "SentimentAnalysis",
                "schema": schema,
                "strict": True
            }
        }
        
        # 发送测试请求 - 情感分析场景
        test_content = "今天天气真好"
        start_time = time.time()
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个情感分析专家。请分析用户输入的文本情感，并以JSON格式返回结果。输出格式示例：{\"sentiment\": \"positive\", \"confidence\": 0.95}"
                },
                {
                    "role": "user", 
                    "content": f"分析这句话的情感：{test_content}"
                }
            ],
            response_format=response_format,
            structured_provider="native",  # 指定使用原生解析
            temperature=0.1,
            max_tokens=500
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证基础响应结构
        assert response is not None, f"模型 {model} 未返回响应"
        assert hasattr(response, 'choices'), f"模型 {model} 响应缺少choices字段"
        assert len(response.choices) > 0, f"模型 {model} choices为空"
        
        choice = response.choices[0]
        message = choice.message
        
        # 验证Native结构化输出解析结果
        assert hasattr(message, 'parsed'), f"模型 {model} 响应缺少parsed字段"
        parsed_data = message.parsed
        assert parsed_data is not None, f"模型 {model} Native结构化输出解析失败"
        
        # 验证情感分析结果
        validate_sentiment_result(parsed_data, model)
        
        # 输出测试结果
        print(f"  Native解析成功 - 延迟: {round(execution_time * 1000, 2)}ms")
        print(f"  情感分析结果: {parsed_data}")
        
        # 如果是推理模型，检查是否包含思考过程
        if is_reasoning and hasattr(message, 'reasoning_content'):
            reasoning = message.reasoning_content
            if reasoning:
                print(f"  思考过程: {reasoning[:100]}...")
        
        # 验证原始内容也存在
        if hasattr(message, 'content') and message.content:
            print(f"  原始回答: {message.content[:100]}...")
            
    except Exception as e:
        print(f"  测试失败 - {str(e)}")
        raise AssertionError(f"模型 {model} Native结构化输出测试失败: {str(e)}") from e

def test_native_vs_agently_comparison(client: HarborAI):
    """对比Native和Agently结构化输出的差异。
    
    使用同一个模型分别测试Native和Agently解析，对比结果差异。
    """
    print(f"\n对比测试: Native vs Agently 结构化输出")
    
    schema = create_sentiment_analysis_schema()
    test_model = "deepseek-chat"  # 使用稳定的非推理模型进行对比
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
    
    try:
        # 测试Native解析
        print("  测试Native解析...")
        start_time = time.time()
        native_response = client.chat.completions.create(
            model=test_model,
            messages=messages,
            response_format=response_format,
            structured_provider="native",
            temperature=0.1
        )
        native_time = time.time() - start_time
        native_result = native_response.choices[0].message.parsed
        
        # 测试Agently解析
        print("  测试Agently解析...")
        agently_response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "SentimentAnalysis",
                "schema": schema,
                "strict": True
            }
        }
        
        start_time = time.time()
        agently_response = client.chat.completions.create(
            model=test_model,
            messages=messages,
            response_format=agently_response_format,
            structured_provider="agently",
            temperature=0.1
        )
        agently_time = time.time() - start_time
        agently_result = agently_response.choices[0].message.parsed
        
        # 对比结果
        print(f"  Native结果: {native_result} (耗时: {round(native_time * 1000, 2)}ms)")
        print(f"  Agently结果: {agently_result} (耗时: {round(agently_time * 1000, 2)}ms)")
        
        # 验证两种解析都成功
        validate_sentiment_result(native_result, f"{test_model}-native")
        validate_sentiment_result(agently_result, f"{test_model}-agently")
        
        print("  对比测试完成，两种解析方式都正常工作")
        
    except Exception as e:
        print(f"  对比测试失败: {str(e)}")
        raise

def main():
    """主测试函数，可直接运行。"""
    print("开始Native结构化输出功能测试")
    print("=" * 80)
    
    # 检查环境变量
    required_env_vars = [
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
        "DOUBAO_API_KEY", "DOUBAO_BASE_URL", 
        "WENXIN_API_KEY", "WENXIN_BASE_URL"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"缺少环境变量: {missing_vars}")
        print("请检查.env文件配置")
        return
    
    print("环境变量检查通过")
    
    # 创建客户端
    client = HarborAI()
    
    # 执行所有模型的测试
    models = get_test_models()
    success_count = 0
    total_count = len(models)
    
    for model_config in models:
        try:
            test_native_structured_output_with_model(client, model_config)
            success_count += 1
        except Exception as e:
            print(f"模型 {model_config['model']} 测试失败: {str(e)}")
    
    # 执行对比测试
    try:
        test_native_vs_agently_comparison(client)
    except Exception as e:
        print(f"对比测试失败: {str(e)}")
    
    # 输出测试总结
    print("\n" + "=" * 80)
    print(f"测试总结:")
    print(f"   总计模型: {total_count}")
    print(f"   成功测试: {success_count}")
    print(f"   失败测试: {total_count - success_count}")
    print(f"   成功率: {round(success_count / total_count * 100, 1)}%")
    
    if success_count == total_count:
        print("所有测试通过！Native结构化输出功能正常工作")
    else:
        print("部分测试失败，请检查失败的模型配置")

if __name__ == "__main__":
    main()