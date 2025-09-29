#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理模型流式输出调试脚本
用于检查推理模型的实际API响应格式
"""

import os
import json
import httpx
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_deepseek_reasoning_stream():
    """测试DeepSeek推理模型的流式输出"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY 未设置")
        return
    
    print("🔍 测试DeepSeek推理模型流式输出...")
    
    # 准备请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "user", "content": "请解释一下什么是机器学习？"}
        ],
        "stream": True,
        "max_tokens": 100
    }
    
    try:
        with httpx.stream(
            "POST",
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30.0
        ) as response:
            print(f"响应状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"错误响应: {response.text}")
                return
            
            chunk_count = 0
            reasoning_chunks = []
            content_chunks = []
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        print("\n✅ 流式响应结束")
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        chunk_count += 1
                        
                        print(f"\n--- Chunk {chunk_count} ---")
                        print(f"完整chunk数据: {json.dumps(chunk_data, ensure_ascii=False, indent=2)}")
                        
                        # 检查choices结构
                        if "choices" in chunk_data:
                            for i, choice in enumerate(chunk_data["choices"]):
                                print(f"Choice {i}:")
                                if "delta" in choice:
                                    delta = choice["delta"]
                                    print(f"  Delta: {json.dumps(delta, ensure_ascii=False, indent=4)}")
                                    
                                    # 检查reasoning_content
                                    if "reasoning_content" in delta and delta["reasoning_content"]:
                                        reasoning_content = delta["reasoning_content"]
                                        reasoning_chunks.append(reasoning_content)
                                        print(f"  🧠 推理内容: {reasoning_content[:100]}...")
                                    
                                    # 检查content
                                    if "content" in delta and delta["content"]:
                                        content = delta["content"]
                                        content_chunks.append(content)
                                        print(f"  💬 回答内容: {content[:100]}...")
                        
                        # 限制输出数量
                        if chunk_count >= 10:
                            print("\n⚠️ 已输出前10个chunk，停止调试")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}, 原始数据: {data_str}")
                        continue
            
            print(f"\n📊 统计信息:")
            print(f"总chunk数量: {chunk_count}")
            print(f"推理内容chunks: {len(reasoning_chunks)}")
            print(f"回答内容chunks: {len(content_chunks)}")
            
            if reasoning_chunks:
                print(f"\n🧠 完整推理过程:")
                print(''.join(reasoning_chunks))
            
            if content_chunks:
                print(f"\n💬 完整回答内容:")
                print(''.join(content_chunks))
                
    except Exception as e:
        print(f"请求错误: {e}")

def test_wenxin_reasoning_stream():
    """测试文心一言推理模型的流式输出"""
    api_key = os.getenv("WENXIN_API_KEY")
    if not api_key:
        print("❌ WENXIN_API_KEY 未设置")
        return
    
    print("\n🔍 测试文心一言推理模型流式输出...")
    
    # 准备请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "ernie-x1-turbo-32k",
        "messages": [
            {"role": "user", "content": "请解释一下什么是机器学习？"}
        ],
        "stream": True,
        "max_tokens": 100
    }
    
    try:
        with httpx.stream(
            "POST",
            "https://qianfan.baidubce.com/v2/chat/completions",
            headers=headers,
            json=data,
            timeout=30.0
        ) as response:
            print(f"响应状态码: {response.status_code}")
            print(f"响应头: {dict(response.headers)}")
            
            if response.status_code != 200:
                print(f"错误响应: {response.text}")
                return
            
            chunk_count = 0
            reasoning_chunks = []
            content_chunks = []
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        print("\n✅ 流式响应结束")
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        chunk_count += 1
                        
                        print(f"\n--- Chunk {chunk_count} ---")
                        print(f"完整chunk数据: {json.dumps(chunk_data, ensure_ascii=False, indent=2)}")
                        
                        # 检查choices结构
                        if "choices" in chunk_data:
                            for i, choice in enumerate(chunk_data["choices"]):
                                print(f"Choice {i}:")
                                if "delta" in choice:
                                    delta = choice["delta"]
                                    print(f"  Delta: {json.dumps(delta, ensure_ascii=False, indent=4)}")
                                    
                                    # 检查reasoning_content
                                    if "reasoning_content" in delta and delta["reasoning_content"]:
                                        reasoning_content = delta["reasoning_content"]
                                        reasoning_chunks.append(reasoning_content)
                                        print(f"  🧠 推理内容: {reasoning_content[:100]}...")
                                    
                                    # 检查content
                                    if "content" in delta and delta["content"]:
                                        content = delta["content"]
                                        content_chunks.append(content)
                                        print(f"  💬 回答内容: {content[:100]}...")
                        
                        # 限制输出数量
                        if chunk_count >= 10:
                            print("\n⚠️ 已输出前10个chunk，停止调试")
                            break
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}, 原始数据: {data_str}")
                        continue
            
            print(f"\n📊 统计信息:")
            print(f"总chunk数量: {chunk_count}")
            print(f"推理内容chunks: {len(reasoning_chunks)}")
            print(f"回答内容chunks: {len(content_chunks)}")
            
            if reasoning_chunks:
                print(f"\n🧠 完整推理过程:")
                print(''.join(reasoning_chunks))
            
            if content_chunks:
                print(f"\n💬 完整回答内容:")
                print(''.join(content_chunks))
                
    except Exception as e:
        print(f"请求错误: {e}")

if __name__ == "__main__":
    print("🔍 调试推理模型流式输出问题")
    print("检查环境变量...")
    
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    wenxin_key = os.getenv("WENXIN_API_KEY")
    
    print(f"DEEPSEEK_API_KEY: {'✅ 已设置' if deepseek_key else '❌ 未设置'}")
    print(f"WENXIN_API_KEY: {'✅ 已设置' if wenxin_key else '❌ 未设置'}")
    
    if deepseek_key:
        test_deepseek_reasoning_stream()
    
    if wenxin_key:
        test_wenxin_reasoning_stream()
    
    print("\n🔍 调试完成")