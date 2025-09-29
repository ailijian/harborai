#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整推理模型流式输出调试脚本
用于检查推理模型的完整流式响应过程
"""

import os
import json
import httpx
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def test_complete_reasoning_stream():
    """测试完整的推理模型流式输出"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ DEEPSEEK_API_KEY 未设置")
        return
    
    print("🔍 测试完整的DeepSeek推理模型流式输出...")
    
    # 准备请求
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "user", "content": "请解释什么是机器学习？"}
        ],
        "stream": True,
        "max_tokens": 500  # 增加token数量以获取完整响应
    }
    
    try:
        with httpx.stream(
            "POST",
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60.0  # 增加超时时间
        ) as response:
            print(f"响应状态码: {response.status_code}")
            
            if response.status_code != 200:
                print(f"错误响应: {response.text}")
                return
            
            chunk_count = 0
            reasoning_chunks = []
            content_chunks = []
            reasoning_phase = True
            content_phase = False
            
            print("\n🧠 开始接收推理过程...")
            
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        print("\n✅ 流式响应结束")
                        break
                    
                    try:
                        chunk_data = json.loads(data_str)
                        chunk_count += 1
                        
                        # 检查choices结构
                        if "choices" in chunk_data:
                            for choice in chunk_data["choices"]:
                                if "delta" in choice:
                                    delta = choice["delta"]
                                    
                                    # 检查reasoning_content
                                    if "reasoning_content" in delta and delta["reasoning_content"]:
                                        reasoning_content = delta["reasoning_content"]
                                        reasoning_chunks.append(reasoning_content)
                                        if reasoning_phase:
                                            print(f"🧠 {reasoning_content}", end="", flush=True)
                                    
                                    # 检查content
                                    if "content" in delta and delta["content"]:
                                        content = delta["content"]
                                        content_chunks.append(content)
                                        if not content_phase:
                                            print("\n\n💬 开始接收最终答案...")
                                            content_phase = True
                                            reasoning_phase = False
                                        print(f"💬 {content}", end="", flush=True)
                                    
                                    # 检查finish_reason
                                    if "finish_reason" in choice and choice["finish_reason"]:
                                        print(f"\n\n🏁 完成原因: {choice['finish_reason']}")
                        
                        # 每50个chunk显示一次进度
                        if chunk_count % 50 == 0:
                            print(f"\n[进度: {chunk_count} chunks]")
                            
                    except json.JSONDecodeError as e:
                        print(f"\nJSON解析错误: {e}, 原始数据: {data_str[:100]}...")
                        continue
            
            print(f"\n\n📊 统计信息:")
            print(f"总chunk数量: {chunk_count}")
            print(f"推理内容chunks: {len(reasoning_chunks)}")
            print(f"回答内容chunks: {len(content_chunks)}")
            
            if reasoning_chunks:
                full_reasoning = ''.join(reasoning_chunks)
                print(f"\n🧠 完整推理过程 ({len(full_reasoning)} 字符):")
                print("=" * 50)
                print(full_reasoning)
                print("=" * 50)
            
            if content_chunks:
                full_content = ''.join(content_chunks)
                print(f"\n💬 完整回答内容 ({len(full_content)} 字符):")
                print("-" * 50)
                print(full_content)
                print("-" * 50)
            
            # 分析流式输出模式
            print(f"\n📈 流式输出模式分析:")
            print(f"推理阶段chunks: {len(reasoning_chunks)}")
            print(f"回答阶段chunks: {len(content_chunks)}")
            
            if reasoning_chunks and content_chunks:
                print("✅ 推理模型正确支持两阶段流式输出")
            elif reasoning_chunks and not content_chunks:
                print("⚠️ 只有推理阶段，缺少回答阶段")
            elif not reasoning_chunks and content_chunks:
                print("⚠️ 只有回答阶段，缺少推理阶段")
            else:
                print("❌ 没有检测到有效的流式内容")
                
    except Exception as e:
        print(f"请求错误: {e}")

if __name__ == "__main__":
    print("🔍 完整推理模型流式输出调试")
    test_complete_reasoning_stream()
    print("\n🔍 调试完成")