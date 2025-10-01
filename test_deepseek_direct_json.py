#!/usr/bin/env python3
"""
测试 DeepSeek json_object 模式的直接输出能力
验证是否能直接返回有效的 JSON 数据，无需额外处理
"""

import os
import sys
import json
import time
from dotenv import load_dotenv
import httpx

# 加载环境变量
load_dotenv()

def test_deepseek_direct_json():
    """直接测试 DeepSeek API 的 json_object 输出能力"""
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 未找到 DEEPSEEK_API_KEY 环境变量")
        return
    
    print("=" * 80)
    print("测试 DeepSeek json_object 模式的直接输出能力")
    print("=" * 80)
    
    # 测试用例：要求返回用户信息的 JSON
    test_cases = [
        {
            "name": "简单用户信息",
            "prompt": "请生成一个虚拟用户的基本信息，包括姓名、年龄、邮箱。用户是一个28岁的程序员。请直接返回JSON格式，包含name、age、email字段。",
        },
        {
            "name": "复杂用户信息",
            "prompt": "请生成一个用户的详细信息JSON，包含：姓名(name)、年龄(age)、邮箱(email)、职业(profession)、技能列表(skills)、地址信息(address包含city和country)。用户是一个30岁的前端开发工程师。",
        },
        {
            "name": "数组数据",
            "prompt": "请生成3个用户的信息列表，每个用户包含name、age、email字段。返回JSON数组格式。",
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 测试用例 {i}: {test_case['name']}")
        print("-" * 50)
        
        # 直接调用 DeepSeek API
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "user",
                    "content": test_case["prompt"]
                }
            ],
            "response_format": {
                "type": "json_object"
            },
            "temperature": 0.3
        }
        
        start_time = time.time()
        
        try:
            with httpx.Client() as client:
                response = client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                print(f"📊 响应时间: {response_time:.3f}s")
                print(f"📊 HTTP状态码: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    print(f"📄 原始响应内容:")
                    print(content)
                    print()
                    
                    # 验证是否为有效 JSON
                    try:
                        parsed_json = json.loads(content)
                        print(f"✅ JSON 解析成功")
                        print(f"📋 解析后的数据:")
                        print(json.dumps(parsed_json, ensure_ascii=False, indent=2))
                        
                        # 记录结果
                        results.append({
                            "test_case": test_case["name"],
                            "success": True,
                            "response_time": response_time,
                            "json_valid": True,
                            "content": content,
                            "parsed_data": parsed_json
                        })
                        
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON 解析失败: {e}")
                        results.append({
                            "test_case": test_case["name"],
                            "success": True,
                            "response_time": response_time,
                            "json_valid": False,
                            "content": content,
                            "error": str(e)
                        })
                        
                else:
                    print(f"❌ API 请求失败: {response.status_code}")
                    print(f"错误信息: {response.text}")
                    results.append({
                        "test_case": test_case["name"],
                        "success": False,
                        "response_time": response_time,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    })
                    
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            print(f"❌ 请求异常: {e}")
            results.append({
                "test_case": test_case["name"],
                "success": False,
                "response_time": response_time,
                "error": str(e)
            })
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    total_tests = len(results)
    successful_requests = sum(1 for r in results if r["success"])
    valid_json_count = sum(1 for r in results if r.get("json_valid", False))
    avg_response_time = sum(r["response_time"] for r in results) / total_tests
    
    print(f"📊 总测试数: {total_tests}")
    print(f"📊 成功请求数: {successful_requests} ({successful_requests/total_tests*100:.1f}%)")
    print(f"📊 有效JSON数: {valid_json_count} ({valid_json_count/total_tests*100:.1f}%)")
    print(f"📊 平均响应时间: {avg_response_time:.3f}s")
    
    # 详细结果
    print(f"\n📋 详细结果:")
    for result in results:
        status = "✅" if result["success"] and result.get("json_valid", False) else "❌"
        print(f"{status} {result['test_case']}: {result['response_time']:.3f}s")
        if not result["success"] or not result.get("json_valid", False):
            print(f"   错误: {result.get('error', 'JSON解析失败')}")
    
    return results

if __name__ == "__main__":
    test_deepseek_direct_json()