#!/usr/bin/env python3
"""
完整的原生结构化输出测试脚本
测试所有7个模型的原生结构化输出能力和性能
"""

import os
import sys
import time
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harborai import HarborAI

def test_model_native_structured_output(client: HarborAI, model: str) -> Dict[str, Any]:
    """测试单个模型的原生结构化输出"""
    
    # 定义JSON Schema
    schema = {
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
    
    prompt = "请分析这句话的情感倾向：'今天天气真好，心情很愉快！'"
    
    result = {
        "model": model,
        "success": False,
        "native_result": None,
        "native_time": 0,
        "agently_result": None,
        "agently_time": 0,
        "error": None
    }
    
    try:
        # 测试原生结构化输出
        print(f"\n测试模型: {model}")
        print("=" * 50)
        
        # Native测试
        start_time = time.time()
        try:
            native_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": {"name": "sentiment_analysis", "schema": schema, "strict": True}},
                structured_provider="native"
            )
            native_time = (time.time() - start_time) * 1000
            
            if hasattr(native_response.choices[0].message, 'parsed') and native_response.choices[0].message.parsed:
                result["native_result"] = native_response.choices[0].message.parsed
                result["native_time"] = native_time
                print(f"✓ Native结果: {result['native_result']} (耗时: {native_time:.2f}ms)")
            else:
                print(f"✗ Native结果: 解析失败，content: {native_response.choices[0].message.content}")
                result["error"] = "Native解析失败"
                
        except Exception as e:
            native_time = (time.time() - start_time) * 1000
            result["native_time"] = native_time
            print(f"✗ Native测试失败: {str(e)} (耗时: {native_time:.2f}ms)")
            result["error"] = str(e)
        
        # Agently测试
        start_time = time.time()
        try:
            agently_response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": {"name": "sentiment_analysis", "schema": schema, "strict": True}},
                structured_provider="agently"
            )
            agently_time = (time.time() - start_time) * 1000
            
            if hasattr(agently_response.choices[0].message, 'parsed') and agently_response.choices[0].message.parsed:
                result["agently_result"] = agently_response.choices[0].message.parsed
                result["agently_time"] = agently_time
                print(f"✓ Agently结果: {result['agently_result']} (耗时: {agently_time:.2f}ms)")
                
                # 如果Native成功，标记为成功
                if result["native_result"]:
                    result["success"] = True
                    print("✓ 对比测试完成，两种解析方式都正常工作")
                else:
                    print("⚠ 仅Agently成功，Native失败")
            else:
                print(f"✗ Agently结果: 解析失败，content: {agently_response.choices[0].message.content}")
                
        except Exception as e:
            agently_time = (time.time() - start_time) * 1000
            result["agently_time"] = agently_time
            print(f"✗ Agently测试失败: {str(e)} (耗时: {agently_time:.2f}ms)")
            
    except Exception as e:
        print(f"✗ 模型 {model} 测试完全失败: {str(e)}")
        result["error"] = str(e)
    
    return result

def main():
    """主测试函数"""
    print("开始Native结构化输出功能测试")
    print("=" * 80)
    
    # 检查环境变量
    required_env_vars = [
        'DEEPSEEK_API_KEY', 'WENXIN_API_KEY', 
        'DOUBAO_API_KEY'
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"缺少环境变量: {missing_vars}")
        return
    
    print("环境变量检查通过")
    
    # 初始化客户端
    client = HarborAI()
    
    # 测试模型列表
    models = [
        "deepseek-chat",
        "deepseek-reasoner", 
        "ernie-3.5-8k",
        "ernie-4.0-turbo-8k",
        "ernie-x1-turbo-32k",
        "doubao-1-5-pro-32k-character-250715",
        "doubao-seed-1-6-250615"
    ]
    
    results = []
    
    # 测试每个模型
    for model in models:
        result = test_model_native_structured_output(client, model)
        results.append(result)
    
    # 生成测试报告
    print("\n" + "=" * 80)
    print("测试总结:")
    
    successful_models = [r for r in results if r["success"]]
    failed_models = [r for r in results if not r["success"]]
    
    print(f"   总计模型: {len(results)}")
    print(f"   成功测试: {len(successful_models)}")
    print(f"   失败测试: {len(failed_models)}")
    print(f"   成功率: {len(successful_models)/len(results)*100:.1f}%")
    
    # 按厂商分组统计
    print("\n按厂商分组统计:")
    
    # DeepSeek
    deepseek_results = [r for r in results if r["model"].startswith("deepseek")]
    deepseek_success = [r for r in deepseek_results if r["success"]]
    print(f"DeepSeek: {len(deepseek_success)}/{len(deepseek_results)} 成功")
    if deepseek_success:
        avg_native_time = sum(r["native_time"] for r in deepseek_success) / len(deepseek_success)
        avg_agently_time = sum(r["agently_time"] for r in deepseek_success) / len(deepseek_success)
        print(f"  - 平均Native响应时间: {avg_native_time:.2f}ms")
        print(f"  - 平均Agently响应时间: {avg_agently_time:.2f}ms")
    
    # 文心一言
    wenxin_results = [r for r in results if r["model"].startswith("ernie")]
    wenxin_success = [r for r in wenxin_results if r["success"]]
    print(f"文心一言: {len(wenxin_success)}/{len(wenxin_results)} 成功")
    if wenxin_success:
        avg_native_time = sum(r["native_time"] for r in wenxin_success) / len(wenxin_success)
        avg_agently_time = sum(r["agently_time"] for r in wenxin_success) / len(wenxin_success)
        print(f"  - 平均Native响应时间: {avg_native_time:.2f}ms")
        print(f"  - 平均Agently响应时间: {avg_agently_time:.2f}ms")
    
    # 豆包
    doubao_results = [r for r in results if r["model"].startswith("doubao")]
    doubao_success = [r for r in doubao_results if r["success"]]
    print(f"豆包: {len(doubao_success)}/{len(doubao_results)} 成功")
    if doubao_success:
        avg_native_time = sum(r["native_time"] for r in doubao_success) / len(doubao_success)
        avg_agently_time = sum(r["agently_time"] for r in doubao_success) / len(doubao_success)
        print(f"  - 平均Native响应时间: {avg_native_time:.2f}ms")
        print(f"  - 平均Agently响应时间: {avg_agently_time:.2f}ms")
    
    # 失败模型详情
    if failed_models:
        print(f"\n失败模型详情:")
        for result in failed_models:
            print(f"  - {result['model']}: {result['error']}")
    
    # 性能对比
    print(f"\n性能对比 (Native vs Agently):")
    for result in successful_models:
        native_time = result["native_time"]
        agently_time = result["agently_time"]
        speedup = agently_time / native_time if native_time > 0 else 0
        print(f"  {result['model']}: Native {native_time:.0f}ms vs Agently {agently_time:.0f}ms (Native快 {speedup:.1f}x)")

if __name__ == "__main__":
    main()