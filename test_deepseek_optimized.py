#!/usr/bin/env python3
"""
测试优化后的DeepSeek原生结构化输出性能
验证直接使用json_object能力，移除Agently后处理的效果
"""

import os
import sys
import time
import json
import statistics
from typing import Dict, Any, List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from harborai.core.plugins.deepseek_plugin import DeepSeekPlugin
from harborai.core.base_plugin import ChatMessage

def test_optimized_deepseek_structured_output():
    """测试优化后的DeepSeek原生结构化输出"""
    
    # 初始化DeepSeek插件
    plugin = DeepSeekPlugin(
        api_key=os.getenv("DEEPSEEK_API_KEY", "your-api-key-here"),
        base_url="https://api.deepseek.com"
    )
    
    # 测试用例
    test_cases = [
        {
            "name": "简单用户信息",
            "messages": [
                ChatMessage(role="user", content="请生成一个用户信息，包含姓名、年龄、邮箱")
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "user_info",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "email": {"type": "string"}
                        },
                        "required": ["name", "age", "email"]
                    }
                }
            }
        },
        {
            "name": "复杂用户信息",
            "messages": [
                ChatMessage(role="user", content="请生成一个详细的用户档案，包含个人信息、地址、技能等")
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "user_profile",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "personal_info": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "age": {"type": "integer"},
                                    "email": {"type": "string"},
                                    "phone": {"type": "string"}
                                }
                            },
                            "address": {
                                "type": "object",
                                "properties": {
                                    "street": {"type": "string"},
                                    "city": {"type": "string"},
                                    "country": {"type": "string"}
                                }
                            },
                            "skills": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    }
                }
            }
        }
    ]
    
    # 测试场景
    scenarios = [
        {
            "name": "优化后的原生结构化输出",
            "structured_provider": "native",
            "description": "直接使用DeepSeek json_object，无Agently后处理"
        },
        {
            "name": "传统Agently后处理",
            "structured_provider": "agently", 
            "description": "DeepSeek json_object + Agently后处理"
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"测试场景: {scenario['name']}")
        print(f"描述: {scenario['description']}")
        print(f"{'='*60}")
        
        scenario_results = {
            "success_count": 0,
            "total_count": 0,
            "response_times": [],
            "json_validity": [],
            "errors": []
        }
        
        for test_case in test_cases:
            print(f"\n测试用例: {test_case['name']}")
            
            try:
                start_time = time.time()
                
                # 执行测试
                response = plugin.chat_completion(
                    model="deepseek-chat",
                    messages=test_case["messages"],
                    response_format=test_case["response_format"],
                    structured_provider=scenario["structured_provider"]
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                scenario_results["total_count"] += 1
                scenario_results["response_times"].append(response_time)
                
                # 检查响应
                if response and response.choices and response.choices[0].message:
                    content = response.choices[0].message.content
                    
                    # 验证JSON有效性
                    try:
                        json.loads(content)
                        scenario_results["json_validity"].append(True)
                        scenario_results["success_count"] += 1
                        
                        print(f"  ✅ 成功 - 响应时间: {response_time:.3f}s")
                        print(f"  📄 内容长度: {len(content)} 字符")
                        
                        # 检查是否有parsed字段（原生结构化输出特有）
                        if hasattr(response, 'parsed') and response.parsed:
                            print(f"  🎯 包含parsed字段: {type(response.parsed)}")
                        
                    except json.JSONDecodeError as e:
                        scenario_results["json_validity"].append(False)
                        scenario_results["errors"].append(f"JSON解析错误: {e}")
                        print(f"  ❌ JSON无效 - 响应时间: {response_time:.3f}s")
                        print(f"  📄 内容: {content[:200]}...")
                        
                else:
                    scenario_results["json_validity"].append(False)
                    scenario_results["errors"].append("响应为空或无效")
                    print(f"  ❌ 响应无效 - 响应时间: {response_time:.3f}s")
                    
            except Exception as e:
                scenario_results["total_count"] += 1
                scenario_results["errors"].append(str(e))
                print(f"  💥 异常: {e}")
        
        results[scenario["name"]] = scenario_results
    
    # 生成性能对比报告
    print(f"\n{'='*80}")
    print("性能对比报告")
    print(f"{'='*80}")
    
    for scenario_name, result in results.items():
        print(f"\n📊 {scenario_name}:")
        print(f"  成功率: {result['success_count']}/{result['total_count']} ({result['success_count']/result['total_count']*100:.1f}%)")
        
        if result["response_times"]:
            avg_time = statistics.mean(result["response_times"])
            min_time = min(result["response_times"])
            max_time = max(result["response_times"])
            print(f"  平均响应时间: {avg_time:.3f}s")
            print(f"  最快响应时间: {min_time:.3f}s")
            print(f"  最慢响应时间: {max_time:.3f}s")
        
        json_valid_count = sum(result["json_validity"])
        json_total_count = len(result["json_validity"])
        if json_total_count > 0:
            print(f"  JSON有效率: {json_valid_count}/{json_total_count} ({json_valid_count/json_total_count*100:.1f}%)")
        
        if result["errors"]:
            print(f"  错误数量: {len(result['errors'])}")
            for error in result["errors"][:3]:  # 只显示前3个错误
                print(f"    - {error}")
    
    # 性能对比
    if len(results) >= 2:
        scenario_names = list(results.keys())
        native_result = results[scenario_names[0]]
        agently_result = results[scenario_names[1]]
        
        if native_result["response_times"] and agently_result["response_times"]:
            native_avg = statistics.mean(native_result["response_times"])
            agently_avg = statistics.mean(agently_result["response_times"])
            
            print(f"\n🚀 性能提升:")
            if native_avg < agently_avg:
                improvement = ((agently_avg - native_avg) / agently_avg) * 100
                print(f"  原生结构化输出比传统方式快 {improvement:.1f}%")
                print(f"  时间节省: {agently_avg - native_avg:.3f}s")
            else:
                degradation = ((native_avg - agently_avg) / native_avg) * 100
                print(f"  原生结构化输出比传统方式慢 {degradation:.1f}%")
    
    print(f"\n{'='*80}")
    print("测试完成")
    print(f"{'='*80}")

if __name__ == "__main__":
    test_optimized_deepseek_structured_output()