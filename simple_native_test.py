#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的原生结构化输出测试脚本
专门用于观察文心一言、豆包、DeepSeek的原生结构化输出能力和性能
"""

import os
import sys
import json
import time
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

def get_test_models() -> List[Dict[str, str]]:
    """获取要测试的模型列表"""
    return [
        # DeepSeek模型
        {"model": "deepseek-chat", "provider": "deepseek", "name": "DeepSeek Chat", "is_reasoning": False},
        {"model": "deepseek-reasoner", "provider": "deepseek", "name": "DeepSeek Reasoner", "is_reasoning": True},
        
        # 文心一言模型
        {"model": "ernie-3.5-8k", "provider": "ernie", "name": "文心一言 3.5", "is_reasoning": False},
        {"model": "ernie-4.0-turbo-8k", "provider": "ernie", "name": "文心一言 4.0 Turbo", "is_reasoning": False},
        {"model": "ernie-x1-turbo-32k", "provider": "ernie", "name": "文心一言 X1 Turbo", "is_reasoning": True},
        
        # 豆包模型
        {"model": "doubao-1-5-pro-32k-character-250715", "provider": "doubao", "name": "豆包 1.5 Pro 32K", "is_reasoning": False},
        {"model": "doubao-seed-1-6-250615", "provider": "doubao", "name": "豆包 Seed 1.6", "is_reasoning": True}
    ]

def test_single_model(client: HarborAI, model_config: Dict[str, str], test_type: str = "native"):
    """测试单个模型的结构化输出"""
    schema = create_sentiment_analysis_schema()
    model = model_config["model"]
    name = model_config["name"]
    is_reasoning = model_config["is_reasoning"]
    
    print(f"\n{'='*60}")
    print(f"测试模型: {name} ({model})")
    print(f"模型类型: {'推理模型' if is_reasoning else '非推理模型'}")
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
        
        response = client.chat.completions.create(**kwargs)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证响应
        if response and hasattr(response, 'choices') and len(response.choices) > 0:
            choice = response.choices[0]
            message = choice.message
            
            # 检查parsed字段
            if hasattr(message, 'parsed') and message.parsed is not None:
                parsed_data = message.parsed
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
                            print(f"   ❌ 数据值验证失败")
                    else:
                        print(f"   ❌ 缺少必要字段")
                else:
                    print(f"   ❌ 返回数据不是字典格式")
            else:
                print(f"❌ {test_type}解析失败 - 无parsed字段或为None")
                if hasattr(message, 'content'):
                    print(f"   原始内容: {message.content[:200]}...")
        else:
            print(f"❌ 响应无效")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        
    return {
        "success": False,
        "time": 0,
        "result": None,
        "model": model,
        "test_type": test_type,
        "error": str(e) if 'e' in locals() else "Unknown error"
    }

def main():
    """主测试函数"""
    print("开始原生结构化输出性能测试")
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
    
    # 测试结果统计
    results = {
        "native": [],
        "agently": []
    }
    
    models = get_test_models()
    
    # 测试所有模型的Native结构化输出
    print(f"\n{'='*80}")
    print("第一阶段：测试Native结构化输出")
    print(f"{'='*80}")
    
    for model_config in models:
        result = test_single_model(client, model_config, "native")
        results["native"].append(result)
    
    # 测试DeepSeek的Agently对比
    print(f"\n{'='*80}")
    print("第二阶段：DeepSeek Native vs Agently 对比测试")
    print(f"{'='*80}")
    
    deepseek_model = {"model": "deepseek-chat", "provider": "deepseek", "name": "DeepSeek Chat", "is_reasoning": False}
    agently_result = test_single_model(client, deepseek_model, "agently")
    results["agently"].append(agently_result)
    
    # 生成测试报告
    print(f"\n{'='*80}")
    print("测试报告")
    print(f"{'='*80}")
    
    # 按厂商分组统计
    providers = {
        "DeepSeek": [],
        "文心一言": [],
        "豆包": []
    }
    
    for result in results["native"]:
        model = result["model"]
        if "deepseek" in model:
            providers["DeepSeek"].append(result)
        elif "ernie" in model:
            providers["文心一言"].append(result)
        elif "doubao" in model:
            providers["豆包"].append(result)
    
    # 输出各厂商统计
    for provider, provider_results in providers.items():
        print(f"\n{provider} 原生结构化输出表现:")
        success_count = sum(1 for r in provider_results if r["success"])
        total_count = len(provider_results)
        avg_time = sum(r["time"] for r in provider_results if r["success"]) / max(success_count, 1)
        
        print(f"  成功率: {success_count}/{total_count} ({round(success_count/total_count*100, 1)}%)")
        if success_count > 0:
            print(f"  平均延迟: {round(avg_time, 2)}ms")
        
        for result in provider_results:
            status = "✅" if result["success"] else "❌"
            model_name = result["model"]
            if result["success"]:
                print(f"    {status} {model_name}: {round(result['time'], 2)}ms")
            else:
                error = result.get("error", "Unknown error")
                print(f"    {status} {model_name}: {error[:50]}...")
    
    # Native vs Agently 对比
    if results["agently"] and results["agently"][0]["success"]:
        native_deepseek = next((r for r in results["native"] if r["model"] == "deepseek-chat"), None)
        agently_deepseek = results["agently"][0]
        
        if native_deepseek and native_deepseek["success"]:
            print(f"\nDeepSeek Native vs Agently 性能对比:")
            print(f"  Native:  {round(native_deepseek['time'], 2)}ms")
            print(f"  Agently: {round(agently_deepseek['time'], 2)}ms")
            speedup = agently_deepseek['time'] / native_deepseek['time']
            print(f"  性能提升: {round(speedup, 2)}x ({round((speedup-1)*100, 1)}%)")
    
    # 总体统计
    total_native_success = sum(1 for r in results["native"] if r["success"])
    total_native_count = len(results["native"])
    
    print(f"\n总体统计:")
    print(f"  Native结构化输出成功率: {total_native_success}/{total_native_count} ({round(total_native_success/total_native_count*100, 1)}%)")
    
    if total_native_success > 0:
        avg_native_time = sum(r["time"] for r in results["native"] if r["success"]) / total_native_success
        print(f"  Native平均延迟: {round(avg_native_time, 2)}ms")

if __name__ == "__main__":
    main()