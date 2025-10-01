#!/usr/bin/env python3
"""
DeepSeek 结构化输出架构方案性能对比测试
对比三种方案：
1. 当前方案：DeepSeek json_object + Agently 后处理
2. 纯 Agently 方案：直接使用 Agently 结构化输出
3. 直接 json_object 方案：仅使用 DeepSeek json_object，不做 schema 验证
"""

import os
import sys
import json
import time
import statistics
from dotenv import load_dotenv
import httpx

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harborai import HarborAI

# 加载环境变量
load_dotenv()

def test_current_approach(client, schema, prompt, iterations=3):
    """测试当前方案：DeepSeek json_object + Agently 后处理"""
    print("🔄 测试当前方案：DeepSeek json_object + Agently 后处理")
    
    results = []
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": schema
                },
                structured_provider="agently",  # 使用 Agently 后处理
                temperature=0.3
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            content = response.choices[0].message.content
            
            # 验证 JSON
            try:
                parsed_json = json.loads(content)
                results.append({
                    "iteration": i + 1,
                    "success": True,
                    "response_time": response_time,
                    "json_valid": True,
                    "content": content,
                    "parsed_data": parsed_json
                })
                print(f"  ✅ 迭代 {i+1}: {response_time:.3f}s - JSON有效")
                
            except json.JSONDecodeError as e:
                results.append({
                    "iteration": i + 1,
                    "success": True,
                    "response_time": response_time,
                    "json_valid": False,
                    "error": str(e)
                })
                print(f"  ❌ 迭代 {i+1}: {response_time:.3f}s - JSON无效: {e}")
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            results.append({
                "iteration": i + 1,
                "success": False,
                "response_time": response_time,
                "error": str(e)
            })
            print(f"  ❌ 迭代 {i+1}: {response_time:.3f}s - 请求失败: {e}")
    
    return results

def test_pure_agently(client, schema, prompt, iterations=3):
    """测试纯 Agently 方案：直接使用 Agently 结构化输出"""
    print("🔄 测试纯 Agently 方案：直接使用 Agently 结构化输出")
    
    results = []
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            # 使用 text 格式，让 Agently 完全处理结构化输出
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": schema
                },
                structured_provider="agently",  # 纯 Agently 处理
                temperature=0.3
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            content = response.choices[0].message.content
            
            # 验证 JSON
            try:
                parsed_json = json.loads(content)
                results.append({
                    "iteration": i + 1,
                    "success": True,
                    "response_time": response_time,
                    "json_valid": True,
                    "content": content,
                    "parsed_data": parsed_json
                })
                print(f"  ✅ 迭代 {i+1}: {response_time:.3f}s - JSON有效")
                
            except json.JSONDecodeError as e:
                results.append({
                    "iteration": i + 1,
                    "success": True,
                    "response_time": response_time,
                    "json_valid": False,
                    "error": str(e)
                })
                print(f"  ❌ 迭代 {i+1}: {response_time:.3f}s - JSON无效: {e}")
                
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            results.append({
                "iteration": i + 1,
                "success": False,
                "response_time": response_time,
                "error": str(e)
            })
            print(f"  ❌ 迭代 {i+1}: {response_time:.3f}s - 请求失败: {e}")
    
    return results

def test_direct_json_object(prompt, iterations=3):
    """测试直接 json_object 方案：仅使用 DeepSeek json_object，不做 schema 验证"""
    print("🔄 测试直接 json_object 方案：仅使用 DeepSeek json_object")
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 未找到 DEEPSEEK_API_KEY 环境变量")
        return []
    
    results = []
    
    for i in range(iterations):
        start_time = time.time()
        
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
                "temperature": 0.3
            }
            
            with httpx.Client() as client:
                response = client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # 验证 JSON
                    try:
                        parsed_json = json.loads(content)
                        results.append({
                            "iteration": i + 1,
                            "success": True,
                            "response_time": response_time,
                            "json_valid": True,
                            "content": content,
                            "parsed_data": parsed_json
                        })
                        print(f"  ✅ 迭代 {i+1}: {response_time:.3f}s - JSON有效")
                        
                    except json.JSONDecodeError as e:
                        results.append({
                            "iteration": i + 1,
                            "success": True,
                            "response_time": response_time,
                            "json_valid": False,
                            "error": str(e)
                        })
                        print(f"  ❌ 迭代 {i+1}: {response_time:.3f}s - JSON无效: {e}")
                        
                else:
                    results.append({
                        "iteration": i + 1,
                        "success": False,
                        "response_time": response_time,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    })
                    print(f"  ❌ 迭代 {i+1}: {response_time:.3f}s - HTTP错误: {response.status_code}")
                    
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            results.append({
                "iteration": i + 1,
                "success": False,
                "response_time": response_time,
                "error": str(e)
            })
            print(f"  ❌ 迭代 {i+1}: {response_time:.3f}s - 请求异常: {e}")
    
    return results

def analyze_results(results, approach_name):
    """分析测试结果"""
    if not results:
        return {
            "approach": approach_name,
            "total_tests": 0,
            "success_rate": 0,
            "json_valid_rate": 0,
            "avg_response_time": 0,
            "min_response_time": 0,
            "max_response_time": 0,
            "std_response_time": 0
        }
    
    total_tests = len(results)
    successful_requests = sum(1 for r in results if r["success"])
    valid_json_count = sum(1 for r in results if r.get("json_valid", False))
    
    response_times = [r["response_time"] for r in results]
    avg_response_time = statistics.mean(response_times)
    min_response_time = min(response_times)
    max_response_time = max(response_times)
    std_response_time = statistics.stdev(response_times) if len(response_times) > 1 else 0
    
    return {
        "approach": approach_name,
        "total_tests": total_tests,
        "success_rate": successful_requests / total_tests * 100,
        "json_valid_rate": valid_json_count / total_tests * 100,
        "avg_response_time": avg_response_time,
        "min_response_time": min_response_time,
        "max_response_time": max_response_time,
        "std_response_time": std_response_time
    }

def main():
    """主测试函数"""
    print("=" * 80)
    print("DeepSeek 结构化输出架构方案性能对比测试")
    print("=" * 80)
    
    # 初始化客户端
    client = HarborAI()
    
    # 定义测试用的 JSON Schema
    user_schema = {
        "name": "user_info",
        "description": "用户基本信息",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "用户姓名"},
                "age": {"type": "integer", "description": "用户年龄", "minimum": 0, "maximum": 150},
                "email": {"type": "string", "description": "用户邮箱地址", "format": "email"}
            },
            "required": ["name", "age", "email"],
            "additionalProperties": False
        }
    }
    
    prompt = "请生成一个虚拟用户的基本信息，包括姓名、年龄、邮箱。用户是一个28岁的程序员。"
    
    iterations = 5  # 每种方案测试5次
    
    print(f"\n📋 测试配置:")
    print(f"   - 测试迭代次数: {iterations}")
    print(f"   - 测试模型: deepseek-chat")
    print(f"   - 测试用例: 用户信息生成")
    
    # 测试三种方案
    print(f"\n" + "=" * 60)
    current_results = test_current_approach(client, user_schema, prompt, iterations)
    
    print(f"\n" + "=" * 60)
    agently_results = test_pure_agently(client, user_schema, prompt, iterations)
    
    print(f"\n" + "=" * 60)
    direct_results = test_direct_json_object(prompt, iterations)
    
    # 分析结果
    print(f"\n" + "=" * 80)
    print("性能分析结果")
    print("=" * 80)
    
    current_analysis = analyze_results(current_results, "当前方案 (json_object + Agently)")
    agently_analysis = analyze_results(agently_results, "纯 Agently 方案")
    direct_analysis = analyze_results(direct_results, "直接 json_object 方案")
    
    analyses = [current_analysis, agently_analysis, direct_analysis]
    
    # 打印对比表格
    print(f"\n📊 性能对比表:")
    print("-" * 100)
    print(f"{'方案':<25} {'成功率':<10} {'JSON有效率':<12} {'平均响应时间':<12} {'最小时间':<10} {'最大时间':<10} {'标准差':<10}")
    print("-" * 100)
    
    for analysis in analyses:
        print(f"{analysis['approach']:<25} "
              f"{analysis['success_rate']:<10.1f}% "
              f"{analysis['json_valid_rate']:<12.1f}% "
              f"{analysis['avg_response_time']:<12.3f}s "
              f"{analysis['min_response_time']:<10.3f}s "
              f"{analysis['max_response_time']:<10.3f}s "
              f"{analysis['std_response_time']:<10.3f}s")
    
    print("-" * 100)
    
    # 性能排名
    print(f"\n🏆 性能排名:")
    
    # 按平均响应时间排序
    sorted_by_time = sorted(analyses, key=lambda x: x['avg_response_time'])
    print(f"\n⏱️  响应时间排名 (越小越好):")
    for i, analysis in enumerate(sorted_by_time, 1):
        print(f"   {i}. {analysis['approach']}: {analysis['avg_response_time']:.3f}s")
    
    # 按JSON有效率排序
    sorted_by_validity = sorted(analyses, key=lambda x: x['json_valid_rate'], reverse=True)
    print(f"\n✅ JSON有效率排名 (越高越好):")
    for i, analysis in enumerate(sorted_by_validity, 1):
        print(f"   {i}. {analysis['approach']}: {analysis['json_valid_rate']:.1f}%")
    
    # 综合评分 (响应时间权重0.6，有效率权重0.4)
    for analysis in analyses:
        time_score = (10 - analysis['avg_response_time']) * 0.6  # 时间越短分数越高
        validity_score = analysis['json_valid_rate'] / 10 * 0.4  # 有效率越高分数越高
        analysis['composite_score'] = max(0, time_score + validity_score)
    
    sorted_by_composite = sorted(analyses, key=lambda x: x['composite_score'], reverse=True)
    print(f"\n🎯 综合评分排名 (时间60% + 有效率40%):")
    for i, analysis in enumerate(sorted_by_composite, 1):
        print(f"   {i}. {analysis['approach']}: {analysis['composite_score']:.2f}分")
    
    return analyses

if __name__ == "__main__":
    main()