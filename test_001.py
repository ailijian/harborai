import os
import sys
import time
from dotenv import load_dotenv
from harborai import HarborAI
from openai import OpenAI

# 加载.env文件
load_dotenv()

def test_basic_chat_completion():
    """测试基础对话完成功能"""
    vendor = "deepseek"
    
    # 开始计时
    start_time = time.perf_counter()
    print(f"\n=== 测试基础对话完成功能 ===")
    print(f"开始时间: {time.strftime('%H:%M:%S', time.localtime())}")
    
    client = HarborAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    
    # 基础对话测试
    messages = [
        {"role": "user", "content": "用一句话解释量子纠缠"}
    ]
    
    # 测试第一个可用模型
    model = "deepseek-chat"
    
    try:
        # 记录请求开始时间
        request_start_time = time.perf_counter()
        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        # 记录首token时间（非流式为完整响应时间）
        first_token_time = time.perf_counter()
        first_token_latency = (first_token_time - request_start_time) * 1000
        
        print(f"✓ 完整请求结果： {response}")
        print(f"  回答内容: {response.choices[0].message.content}")
        print(f"  首token时间: {first_token_latency:.2f}ms")
        
    except Exception as e:
        print(f"{vendor} 模型 {model} 调用失败: {str(e)}")
    
    # 计算总调用时间
    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000
    print(f"  总调用时间: {total_time:.2f}ms")
    print(f"结束时间: {time.strftime('%H:%M:%S', time.localtime())}")

def test_streaming_chat_completion():
    """测试流式对话完成功能"""
    vendor = "deepseek"
    
    # 开始计时
    start_time = time.perf_counter()
    print(f"\n=== 测试流式对话完成功能 ===")
    print(f"开始时间: {time.strftime('%H:%M:%S', time.localtime())}")
    
    client = HarborAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    
    # 流式对话测试
    messages = [
        {"role": "user", "content": "请用三句话介绍人工智能的发展历程"}
    ]
    
    # 测试流式输出
    model = "deepseek-chat"
    
    try:
        print(f"模型: {model}")
        print(f"问题: {messages[0]['content']}")
        print(f"回答: ", end="", flush=True)
        
        # 记录请求开始时间
        request_start_time = time.perf_counter()
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        # 处理流式响应
        full_content = ""
        first_token_received = False
        first_token_time = None
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                # 记录首token时间
                if not first_token_received:
                    first_token_time = time.perf_counter()
                    first_token_latency = (first_token_time - request_start_time) * 1000
                    first_token_received = True
                
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                sys.stdout.flush()  # 强制刷新输出缓冲区
                full_content += content
                time.sleep(0.05)  # 添加小延迟以显示流式效果
        
        print(f"\n\n✓ 流式输出测试完成")
        print(f"  完整回答长度: {len(full_content)} 字符")
        if first_token_received:
            print(f"  首token时间: {first_token_latency:.2f}ms")
        
    except Exception as e:
        print(f"\n{vendor} 模型 {model} 流式调用失败: {str(e)}")
    
    # 计算总调用时间
    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000
    print(f"  总调用时间: {total_time:.2f}ms")
    print(f"结束时间: {time.strftime('%H:%M:%S', time.localtime())}")

def test_basic_chat_completion_openai():
    """测试基础对话完成功能"""
    vendor = "deepseek"
    
    # 开始计时
    start_time = time.perf_counter()
    print(f"\n=== 测试OpenAI基础对话完成功能 ===")
    print(f"开始时间: {time.strftime('%H:%M:%S', time.localtime())}")
    
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    
    # 基础对话测试
    messages = [
        {"role": "user", "content": "用一句话解释量子纠缠"}
    ]
    
    # 测试第一个可用模型
    model = "deepseek-chat"
    
    try:
        # 记录请求开始时间
        request_start_time = time.perf_counter()
        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        # 记录首token时间（非流式为完整响应时间）
        first_token_time = time.perf_counter()
        first_token_latency = (first_token_time - request_start_time) * 1000
        
        print(f"✓ openai完整请求结果： {response}")
        print(f"  openai回答内容: {response.choices[0].message.content}")
        print(f"  首token时间: {first_token_latency:.2f}ms")
        
    except Exception as e:
        print(f"openai{vendor} 模型 {model} 调用失败: {str(e)}")
    
    # 计算总调用时间
    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000
    print(f"  总调用时间: {total_time:.2f}ms")
    print(f"结束时间: {time.strftime('%H:%M:%S', time.localtime())}")

def test_streaming_chat_completion_openai():
    """测试流式对话完成功能"""
    vendor = "deepseek"
    
    # 开始计时
    start_time = time.perf_counter()
    print(f"\n=== 测试OpenAI流式对话完成功能 ===")
    print(f"开始时间: {time.strftime('%H:%M:%S', time.localtime())}")
    
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
    
    # 流式对话测试
    messages = [
        {"role": "user", "content": "请用三句话介绍人工智能的发展历程"}
    ]
    
    # 测试流式输出
    model = "deepseek-chat"
    
    try:
        print(f"openai模型: {model}")
        print(f"openai问题: {messages[0]['content']}")
        print(f"openai回答: ", end="", flush=True)
        
        # 记录请求开始时间
        request_start_time = time.perf_counter()
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )
        
        # 处理流式响应
        full_content = ""
        first_token_received = False
        first_token_time = None
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                # 记录首token时间
                if not first_token_received:
                    first_token_time = time.perf_counter()
                    first_token_latency = (first_token_time - request_start_time) * 1000
                    first_token_received = True
                
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                sys.stdout.flush()  # 强制刷新输出缓冲区
                full_content += content
                time.sleep(0.05)  # 添加小延迟以显示流式效果
        
        print(f"\n\n✓ openai流式输出测试完成")
        print(f"  openai完整回答长度: {len(full_content)} 字符")
        if first_token_received:
            print(f"  首token时间: {first_token_latency:.2f}ms")
        
    except Exception as e:
        print(f"\nopenai{vendor} 模型 {model} 流式调用失败: {str(e)}")
    
    # 计算总调用时间
    end_time = time.perf_counter()
    total_time = (end_time - start_time) * 1000
    print(f"  总调用时间: {total_time:.2f}ms")
    print(f"结束时间: {time.strftime('%H:%M:%S', time.localtime())}")


if __name__ == "__main__":
    test_basic_chat_completion()
    test_streaming_chat_completion()
    test_basic_chat_completion_openai()
    test_streaming_chat_completion_openai()