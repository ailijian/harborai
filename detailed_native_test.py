#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细的原生结构化输出测试脚本
专门用于观察文心一言、豆包、DeepSeek的原生结构化输出能力和性能
"""

import os
import sys
import json
import time
import traceback
from typing import Dict, Any, List

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

def test_single_model(client: HarborAI, model: str, name: str, test_type: str = "native"):
    """测试单个模型的结构化输出"""
    schema = create_sentiment_analysis_schema()
    
    print(f"\n{'='*60}")
    print(f"测试模型: {name} ({model})")
    print(f"测试类型: {test_type}")
    
    try:
        # 创建response_format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "SentimentAnalysis",
                "schema": schema,
                "strict": True
            }
        }
        
        # 发送测试请求
        test_content = "今天天气真好"
        start_time = time.time()
        
        kwargs = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个情感分析专家。请分析用户输入的文本情感，并以JSON格式返回结果。"
                },
                {
                    "role": "user", 
                    "content": f"分析这句话的情感：{test_content}"
                }
            ],
            "response_format": response_format,
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        # 根据测试类型设置structured_provider
        if test_type == "native":
            kwargs["structured_provider"] = "native"
        elif test_type == "agently":
            kwargs["structured_provider"] = "agently"
        
        print(f"发送请求参数: {json.dumps({k: v for k, v in kwargs.items() if k != 'messages'}, indent=2, ensure_ascii=False)}")
        
        response = client.chat.completions.create(**kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"响应时间: {round(execution_time * 1000, 2)}ms")
        
        # 验证响应
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message
            
            print(f"响应对象类型: {type(response)}")
            print(f"Choice对象类型: {type(choice)}")
            print(f"Message对象类型: {type(message)}")
            print(f"Message属性: {dir(message)}")
            
            # 检查parsed字段
            if hasattr(message, 'parsed'):
                parsed_data = message.parsed
                print(f"Parsed字段存在: {parsed_data}")
                print(f"Parsed类型: {type(parsed_data)}")
                
                if parsed_data is not None:
                    print(f"✅ {test_type}解析成功")
                    print(f"   延迟: {round(execution_time * 1000, 2)}ms")
                    print(f"   结果: {parsed_data}")
                    
                    # 验证数据结构
                    if isinstance(parsed_data, dict):
                        if "sentiment" in parsed_data and "confidence" in parsed_data:
                            sentiment = parsed_data["sentiment"]
                            confidence = parsed_data["confidence"]
                            if sentiment in ["positive", "negative", "neutral"] and 0 <= confidence <= 1:
                                print(f"   ✅ 数据格式验证通过")
                                return {
                                    "success": True,
                                    "time": execution_time * 1000,
                                    "result": parsed_data,
                                    "model": model,
                                    "test_type": test_type
                                }
                            else:
                                print(f"   ❌ 数据值验证失败: sentiment={sentiment}, confidence={confidence}")
                        else:
                            print(f"   ❌ 缺少必要字段: {list(parsed_data.keys())}")
                    else:
                        print(f"   ❌ 返回数据不是字典格式: {type(parsed_data)}")
                else:
                    print(f"❌ {test_type}解析失败 - parsed字段为None")
            else:
                print(f"❌ {test_type}解析失败 - 无parsed字段")
            
            # 检查content字段
            if hasattr(message, 'content'):
                content = message.content
                print(f"Content字段: {content[:200] if content else 'None'}...")
            
        else:
            print(f"❌ 响应无效: response={response}")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        print(f"错误详情: {traceback.format_exc()}")
        
        return {
            "success": False,
            "time": 0,
            "result": None,
            "model": model,
            "test_type": test_type,
            "error": str(e)
        }
        
    return {
        "success": False,
        "time": 0,
        "result": None,
        "model": model,
        "test_type": test_type,
        "error": "Unknown error"
    }

def main():
    """主测试函数"""
    print("开始详细原生结构化输出测试")
    print("="*80)
    
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
    
    # 测试模型列表
    test_models = [
        ("deepseek-chat", "DeepSeek Chat"),
        ("ernie-3.5-8k", "文心一言 3.5"),
        ("doubao-1-5-pro-32k-character-250715", "豆包 1.5 Pro 32K")
    ]
    
    results = []
    
    # 测试每个模型
    for model, name in test_models:
        result = test_single_model(client, model, name, "native")
        results.append(result)
    
    # 测试DeepSeek的Agently对比
    print(f"\n{'='*80}")
    print("DeepSeek Native vs Agently 对比测试")
    print(f"{'='*80}")
    
    agently_result = test_single_model(client, "deepseek-chat", "DeepSeek Chat", "agently")
    
    # 生成测试报告
    print(f"\n{'='*80}")
    print("详细测试报告")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n模型: {result['model']}")
        print(f"成功: {'✅' if result['success'] else '❌'}")
        if result['success']:
            print(f"延迟: {round(result['time'], 2)}ms")
            print(f"结果: {result['result']}")
        else:
            print(f"错误: {result.get('error', 'Unknown')}")
    
    # Native vs Agently 对比
    deepseek_native = next((r for r in results if r["model"] == "deepseek-chat"), None)
    if deepseek_native and agently_result:
        print(f"\nDeepSeek 性能对比:")
        print(f"Native成功: {'✅' if deepseek_native['success'] else '❌'}")
        print(f"Agently成功: {'✅' if agently_result['success'] else '❌'}")
        
        if deepseek_native['success'] and agently_result['success']:
            native_time = deepseek_native['time']
            agently_time = agently_result['time']
            speedup = agently_time / native_time
            print(f"Native延迟: {round(native_time, 2)}ms")
            print(f"Agently延迟: {round(agently_time, 2)}ms")
            print(f"性能提升: {round(speedup, 2)}x")

if __name__ == "__main__":
    main()