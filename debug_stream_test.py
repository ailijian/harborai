#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试推理模型流式输出问题的测试脚本
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

# 检查环境变量
required_vars = ["DEEPSEEK_API_KEY", "DOUBAO_API_KEY", "WENXIN_API_KEY"]
for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {value[:10]}...{value[-4:] if len(value) > 14 else value}")
    else:
        print(f"❌ 缺少环境变量: {var}")

from harborai.api.client import HarborAI

def debug_stream_response(model_name: str):
    """调试特定模型的流式响应"""
    print(f"\n=== 调试模型: {model_name} ===")
    
    client = HarborAI()
    messages = [{"role": "user", "content": "请简单介绍一下人工智能"}]
    
    try:
        print("发送流式请求...")
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True
        )
        
        print(f"响应类型: {type(response)}")
        print(f"响应对象: {response}")
        
        # 尝试迭代响应
        if hasattr(response, '__iter__'):
            print("响应是可迭代的，开始迭代...")
            chunk_count = 0
            for chunk in response:
                chunk_count += 1
                print(f"Chunk {chunk_count}: {type(chunk)} - {chunk}")
                if chunk_count >= 3:  # 只显示前3个chunk
                    print("...（省略更多chunk）")
                    break
        else:
            print("响应不是可迭代的！")
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 测试有问题的推理模型
    debug_stream_response("deepseek-reasoner")
    debug_stream_response("ernie-x1-turbo-32k")