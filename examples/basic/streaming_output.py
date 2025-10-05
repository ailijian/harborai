#!/usr/bin/env python3
"""
HarborAI 流式输出示例

这个示例展示了如何使用HarborAI的流式输出功能，
实现实时响应显示，提升用户体验。

场景描述:
- 流式响应处理
- 实时内容显示
- 打字机效果实现

应用价值:
- 提升用户体验
- 降低感知延迟
- 支持长文本生成
"""

import os
import time
import asyncio
import sys
from typing import Iterator, AsyncIterator, Dict, Any
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

try:
    from harborai import HarborAI
except ImportError:
    print("❌ 请先安装 HarborAI: pip install harborai")
    exit(1)


def create_client() -> HarborAI:
    """
    创建HarborAI客户端
    
    Returns:
        HarborAI: 配置好的客户端实例
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    
    if not api_key:
        raise ValueError("请在环境变量中设置 DEEPSEEK_API_KEY")
    
    return HarborAI(
        api_key=api_key,
        base_url=base_url
    )


def stream_chat_sync(client: HarborAI, question: str, model: str = "deepseek-chat") -> Dict[str, Any]:
    """
    同步流式聊天
    
    Args:
        client: HarborAI客户端
        question: 用户问题
        model: 使用的模型名称
        
    Returns:
        Dict: 包含完整响应和统计信息的字典
    """
    print(f"\n🤖 AI正在思考: {question}")
    print("💭 回答: ", end="", flush=True)
    
    start_time = time.time()
    full_response = ""
    chunk_count = 0
    
    try:
        # 创建流式请求
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=1000,
            stream=True  # 启用流式输出
        )
        
        # 处理流式响应
        for chunk in stream:
            chunk_count += 1
            
            # 检查是否有内容
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                
                # 实时显示内容（打字机效果）
                print(content, end="", flush=True)
                
                # 添加小延迟以模拟打字机效果
                time.sleep(0.02)
        
        elapsed_time = time.time() - start_time
        print(f"\n\n✅ 流式响应完成")
        print(f"📊 统计信息:")
        print(f"   - 响应时间: {elapsed_time:.2f}秒")
        print(f"   - 数据块数: {chunk_count}")
        print(f"   - 响应长度: {len(full_response)}字符")
        
        return {
            "success": True,
            "question": question,
            "answer": full_response,
            "elapsed_time": elapsed_time,
            "chunk_count": chunk_count,
            "model": model
        }
        
    except Exception as e:
        print(f"\n❌ 流式调用失败: {e}")
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "model": model
        }


async def stream_chat_async(client: HarborAI, question: str, model: str = "deepseek-chat") -> Dict[str, Any]:
    """
    异步流式聊天
    
    Args:
        client: HarborAI客户端
        question: 用户问题
        model: 使用的模型名称
        
    Returns:
        Dict: 包含完整响应和统计信息的字典
    """
    print(f"\n🤖 AI正在异步思考: {question}")
    print("💭 回答: ", end="", flush=True)
    
    start_time = time.time()
    full_response = ""
    chunk_count = 0
    
    try:
        # 创建异步流式请求
        stream = await client.chat.completions.acreate(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=1000,
            stream=True  # 启用流式输出
        )
        
        # 处理异步流式响应
        async for chunk in stream:
            chunk_count += 1
            
            # 检查是否有内容
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                
                # 实时显示内容
                print(content, end="", flush=True)
                
                # 异步延迟
                await asyncio.sleep(0.02)
        
        elapsed_time = time.time() - start_time
        print(f"\n\n✅ 异步流式响应完成")
        print(f"📊 统计信息:")
        print(f"   - 响应时间: {elapsed_time:.2f}秒")
        print(f"   - 数据块数: {chunk_count}")
        print(f"   - 响应长度: {len(full_response)}字符")
        
        return {
            "success": True,
            "question": question,
            "answer": full_response,
            "elapsed_time": elapsed_time,
            "chunk_count": chunk_count,
            "model": model
        }
        
    except Exception as e:
        print(f"\n❌ 异步流式调用失败: {e}")
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "model": model
        }


def stream_with_progress(client: HarborAI, question: str, model: str = "deepseek-chat") -> Dict[str, Any]:
    """
    带进度指示的流式输出
    
    Args:
        client: HarborAI客户端
        question: 用户问题
        model: 使用的模型名称
        
    Returns:
        Dict: 包含完整响应和统计信息的字典
    """
    print(f"\n🎯 带进度的流式响应: {question}")
    
    start_time = time.time()
    full_response = ""
    chunk_count = 0
    progress_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    progress_index = 0
    
    try:
        # 显示初始进度
        print("🔄 正在生成回答... ", end="", flush=True)
        
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=1000,
            stream=True
        )
        
        # 清除进度指示器
        print("\r" + " " * 50 + "\r", end="", flush=True)
        print("💭 回答: ", end="", flush=True)
        
        for chunk in stream:
            chunk_count += 1
            
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                print(content, end="", flush=True)
            
            # 更新进度指示器（在没有内容时显示）
            else:
                progress_char = progress_chars[progress_index % len(progress_chars)]
                print(f"\r💭 回答: {full_response}{progress_char}", end="", flush=True)
                progress_index += 1
                time.sleep(0.1)
        
        elapsed_time = time.time() - start_time
        print(f"\n\n✅ 带进度的流式响应完成")
        print(f"📊 统计信息:")
        print(f"   - 响应时间: {elapsed_time:.2f}秒")
        print(f"   - 数据块数: {chunk_count}")
        print(f"   - 响应长度: {len(full_response)}字符")
        
        return {
            "success": True,
            "question": question,
            "answer": full_response,
            "elapsed_time": elapsed_time,
            "chunk_count": chunk_count,
            "model": model
        }
        
    except Exception as e:
        print(f"\n❌ 带进度的流式调用失败: {e}")
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "model": model
        }


def compare_streaming_vs_normal(client: HarborAI, question: str, model: str = "deepseek-chat"):
    """
    对比流式输出与普通输出
    
    Args:
        client: HarborAI客户端
        question: 用户问题
        model: 使用的模型名称
    """
    print("\n" + "="*60)
    print("⚡ 流式输出 vs 普通输出对比")
    print("="*60)
    
    # 普通输出
    print("\n🐌 普通输出模式:")
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=1000,
            stream=False
        )
        
        normal_time = time.time() - start_time
        answer = response.choices[0].message.content
        
        print(f"⏱️  等待时间: {normal_time:.2f}秒")
        print(f"💭 完整回答: {answer[:200]}...")
        
    except Exception as e:
        print(f"❌ 普通调用失败: {e}")
        return
    
    # 流式输出
    print(f"\n🚀 流式输出模式:")
    stream_result = stream_chat_sync(client, question, model)
    
    if stream_result["success"]:
        # 对比分析
        print(f"\n📊 对比分析:")
        print(f"   普通模式总时间: {normal_time:.2f}秒")
        print(f"   流式模式总时间: {stream_result['elapsed_time']:.2f}秒")
        print(f"   流式数据块数: {stream_result['chunk_count']}")
        print(f"   用户体验提升: 实时反馈 vs 等待{normal_time:.1f}秒")


async def multiple_concurrent_streams(client: HarborAI, questions: list, model: str = "deepseek-chat"):
    """
    多个并发流式请求
    
    Args:
        client: HarborAI客户端
        questions: 问题列表
        model: 使用的模型名称
    """
    print("\n" + "="*60)
    print("🔄 多个并发流式请求示例")
    print("="*60)
    
    async def stream_with_id(question: str, stream_id: int):
        """带ID的流式处理"""
        print(f"\n🎯 流 {stream_id}: {question}")
        print(f"💭 回答 {stream_id}: ", end="", flush=True)
        
        try:
            stream = await client.chat.completions.acreate(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.7,
                max_tokens=500,
                stream=True
            )
            
            full_response = ""
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    print(content, end="", flush=True)
                    await asyncio.sleep(0.01)
            
            print(f"\n✅ 流 {stream_id} 完成")
            return {"id": stream_id, "question": question, "answer": full_response}
            
        except Exception as e:
            print(f"\n❌ 流 {stream_id} 失败: {e}")
            return {"id": stream_id, "question": question, "error": str(e)}
    
    # 创建并发任务
    tasks = [stream_with_id(q, i+1) for i, q in enumerate(questions)]
    
    # 并发执行
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    print(f"\n🎉 所有并发流完成，总耗时: {total_time:.2f}秒")
    return results


def interactive_streaming_chat(client: HarborAI, model: str = "deepseek-chat"):
    """
    交互式流式聊天
    
    Args:
        client: HarborAI客户端
        model: 使用的模型名称
    """
    print("\n" + "="*60)
    print("💬 交互式流式聊天")
    print("="*60)
    print("💡 输入 'quit' 或 'exit' 退出聊天")
    print("💡 输入 'clear' 清空对话历史")
    
    conversation_history = []
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n👤 您: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            
            if user_input.lower() in ['clear', '清空']:
                conversation_history = []
                print("🗑️  对话历史已清空")
                continue
            
            if not user_input:
                continue
            
            # 添加用户消息到历史
            conversation_history.append({"role": "user", "content": user_input})
            
            # 流式响应
            print("🤖 AI: ", end="", flush=True)
            
            stream = client.chat.completions.create(
                model=model,
                messages=conversation_history,
                temperature=0.7,
                max_tokens=1000,
                stream=True
            )
            
            ai_response = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    ai_response += content
                    print(content, end="", flush=True)
                    time.sleep(0.02)
            
            # 添加AI响应到历史
            conversation_history.append({"role": "assistant", "content": ai_response})
            
            # 限制历史长度
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            
        except KeyboardInterrupt:
            print("\n\n👋 聊天被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 聊天出错: {e}")


async def main():
    """主函数"""
    print("="*60)
    print("🌊 HarborAI 流式输出示例")
    print("="*60)
    
    try:
        # 创建客户端
        client = create_client()
        print("✅ HarborAI 客户端初始化成功")
        
        # 测试问题
        test_questions = [
            "请详细解释什么是人工智能，包括其发展历程和主要应用领域。",
            "编写一个Python函数来计算斐波那契数列的前n项。",
            "描述机器学习中监督学习和无监督学习的区别。"
        ]
        
        # 1. 同步流式输出
        print("\n🔹 1. 同步流式输出示例")
        stream_chat_sync(client, test_questions[0])
        
        # 2. 异步流式输出
        print("\n🔹 2. 异步流式输出示例")
        await stream_chat_async(client, test_questions[1])
        
        # 3. 带进度的流式输出
        print("\n🔹 3. 带进度指示的流式输出")
        stream_with_progress(client, test_questions[2])
        
        # 4. 流式 vs 普通输出对比
        print("\n🔹 4. 流式输出与普通输出对比")
        compare_streaming_vs_normal(client, "什么是深度学习？")
        
        # 5. 多个并发流式请求
        print("\n🔹 5. 多个并发流式请求")
        concurrent_questions = [
            "什么是云计算？",
            "解释区块链技术",
            "Python的优势是什么？"
        ]
        await multiple_concurrent_streams(client, concurrent_questions)
        
        # 6. 交互式流式聊天
        print("\n🔹 6. 交互式流式聊天")
        choice = input("是否开始交互式聊天？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_streaming_chat(client)
        
        print(f"\n🎉 所有流式输出示例执行完成！")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        print("\n💡 请检查:")
        print("1. 是否正确配置了环境变量")
        print("2. 网络连接是否正常")
        print("3. API密钥是否有效")


if __name__ == "__main__":
    asyncio.run(main())