#!/usr/bin/env python3
"""
HarborAI 异步调用示例

这个示例展示了如何使用HarborAI进行异步调用，
提升并发性能，适合需要同时处理多个请求的场景。

场景描述:
- 异步/等待语法使用
- 并发请求处理
- 性能对比展示

应用价值:
- 提升应用响应速度
- 优化资源利用率
- 支持高并发场景
"""

import os
import time
import asyncio
from typing import List, Dict, Any
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


async def async_chat_single(client: HarborAI, question: str, model: str = "deepseek-chat") -> Dict[str, Any]:
    """
    单个异步聊天调用
    
    Args:
        client: HarborAI客户端
        question: 用户问题
        model: 使用的模型名称
        
    Returns:
        Dict: 包含响应和统计信息的字典
    """
    start_time = time.time()
    
    try:
        # 异步调用模型
        response = await client.chat.completions.acreate(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=500
        )
        
        elapsed_time = time.time() - start_time
        answer = response.choices[0].message.content
        usage = response.usage
        
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "usage": usage,
            "elapsed_time": elapsed_time,
            "model": model
        }
        
    except Exception as e:
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "model": model
        }


def sync_chat_single(client: HarborAI, question: str, model: str = "deepseek-chat") -> Dict[str, Any]:
    """
    单个同步聊天调用
    
    Args:
        client: HarborAI客户端
        question: 用户问题
        model: 使用的模型名称
        
    Returns:
        Dict: 包含响应和统计信息的字典
    """
    start_time = time.time()
    
    try:
        # 同步调用模型
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=500
        )
        
        elapsed_time = time.time() - start_time
        answer = response.choices[0].message.content
        usage = response.usage
        
        return {
            "success": True,
            "question": question,
            "answer": answer,
            "usage": usage,
            "elapsed_time": elapsed_time,
            "model": model
        }
        
    except Exception as e:
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "model": model
        }


async def async_batch_processing(client: HarborAI, questions: List[str], model: str = "deepseek-chat") -> List[Dict[str, Any]]:
    """
    异步批量处理
    
    Args:
        client: HarborAI客户端
        questions: 问题列表
        model: 使用的模型名称
        
    Returns:
        List[Dict]: 所有响应结果
    """
    print(f"\n🚀 开始异步批量处理 {len(questions)} 个请求...")
    start_time = time.time()
    
    # 创建异步任务
    tasks = [async_chat_single(client, question, model) for question in questions]
    
    # 并发执行所有任务
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # 处理结果
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "success": False,
                "question": questions[i],
                "error": str(result),
                "model": model
            })
        else:
            processed_results.append(result)
    
    print(f"✅ 异步批量处理完成，总耗时: {total_time:.2f}秒")
    return processed_results


def sync_batch_processing(client: HarborAI, questions: List[str], model: str = "deepseek-chat") -> List[Dict[str, Any]]:
    """
    同步批量处理
    
    Args:
        client: HarborAI客户端
        questions: 问题列表
        model: 使用的模型名称
        
    Returns:
        List[Dict]: 所有响应结果
    """
    print(f"\n🐌 开始同步批量处理 {len(questions)} 个请求...")
    start_time = time.time()
    
    results = []
    for question in questions:
        result = sync_chat_single(client, question, model)
        results.append(result)
    
    total_time = time.time() - start_time
    print(f"✅ 同步批量处理完成，总耗时: {total_time:.2f}秒")
    return results


def print_results(results: List[Dict[str, Any]], title: str):
    """
    打印结果统计
    
    Args:
        results: 结果列表
        title: 标题
    """
    print(f"\n📊 {title}")
    print("-" * 50)
    
    successful_results = [r for r in results if r["success"]]
    failed_results = [r for r in results if not r["success"]]
    
    if successful_results:
        total_time = sum(r["elapsed_time"] for r in successful_results)
        avg_time = total_time / len(successful_results)
        total_tokens = sum(r["usage"].total_tokens for r in successful_results)
        
        print(f"✅ 成功: {len(successful_results)}/{len(results)}")
        print(f"⏱️  平均耗时: {avg_time:.2f}秒")
        print(f"🎯 总tokens: {total_tokens}")
        
        # 显示前3个结果
        for i, result in enumerate(successful_results[:3]):
            print(f"\n📝 结果 {i+1}:")
            print(f"   问题: {result['question'][:50]}...")
            print(f"   回答: {result['answer'][:100]}...")
            print(f"   耗时: {result['elapsed_time']:.2f}秒")
    
    if failed_results:
        print(f"\n❌ 失败: {len(failed_results)}")
        for result in failed_results:
            print(f"   {result['question'][:50]}... -> {result['error']}")


async def performance_comparison(client: HarborAI):
    """
    性能对比测试
    
    Args:
        client: HarborAI客户端
    """
    print("\n" + "="*60)
    print("⚡ 异步 vs 同步性能对比测试")
    print("="*60)
    
    # 测试问题
    test_questions = [
        "什么是机器学习？",
        "解释一下深度学习的基本概念",
        "Python有哪些优势？",
        "什么是云计算？",
        "区块链技术的应用场景有哪些？"
    ]
    
    print(f"📋 测试问题数量: {len(test_questions)}")
    
    # 同步处理
    sync_results = sync_batch_processing(client, test_questions)
    sync_total_time = sum(r["elapsed_time"] for r in sync_results if r["success"])
    
    # 异步处理
    async_results = await async_batch_processing(client, test_questions)
    async_total_time = sum(r["elapsed_time"] for r in async_results if r["success"])
    
    # 打印结果
    print_results(sync_results, "同步处理结果")
    print_results(async_results, "异步处理结果")
    
    # 性能对比
    print(f"\n🏆 性能对比总结:")
    print("-" * 30)
    print(f"同步总耗时: {sync_total_time:.2f}秒")
    print(f"异步总耗时: {async_total_time:.2f}秒")
    
    if async_total_time > 0:
        improvement = ((sync_total_time - async_total_time) / async_total_time) * 100
        print(f"性能提升: {improvement:.1f}%")
    
    return sync_results, async_results


async def concurrent_different_models(client: HarborAI):
    """
    并发调用不同模型
    
    Args:
        client: HarborAI客户端
    """
    print("\n" + "="*60)
    print("🔄 并发调用不同模型示例")
    print("="*60)
    
    question = "请简单介绍一下人工智能的发展历程"
    
    # 准备不同的客户端和模型
    tasks = []
    
    # DeepSeek模型
    tasks.append(async_chat_single(client, question, "deepseek-chat"))
    
    # 如果配置了其他模型，也可以并发调用
    if os.getenv("OPENAI_API_KEY"):
        openai_client = HarborAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )
        tasks.append(async_chat_single(openai_client, question, "gpt-3.5-turbo"))
    
    # 并发执行
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    print(f"✅ 并发调用完成，总耗时: {total_time:.2f}秒")
    
    # 显示结果
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"\n❌ 模型 {i+1} 调用失败: {result}")
        elif result["success"]:
            print(f"\n✅ 模型: {result['model']}")
            print(f"   耗时: {result['elapsed_time']:.2f}秒")
            print(f"   回答: {result['answer'][:200]}...")
        else:
            print(f"\n❌ 模型: {result['model']} 调用失败: {result['error']}")


async def rate_limited_requests(client: HarborAI, questions: List[str], rate_limit: int = 2):
    """
    限流异步请求示例
    
    Args:
        client: HarborAI客户端
        questions: 问题列表
        rate_limit: 每秒最大请求数
    """
    print(f"\n🚦 限流异步请求示例 (每秒最多 {rate_limit} 个请求)")
    print("-" * 50)
    
    semaphore = asyncio.Semaphore(rate_limit)
    
    async def rate_limited_call(question: str):
        async with semaphore:
            result = await async_chat_single(client, question)
            await asyncio.sleep(1 / rate_limit)  # 控制请求频率
            return result
    
    start_time = time.time()
    tasks = [rate_limited_call(q) for q in questions]
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    print(f"✅ 限流请求完成，总耗时: {total_time:.2f}秒")
    print(f"📊 平均请求间隔: {total_time/len(questions):.2f}秒")
    
    return results


async def main():
    """主函数"""
    print("="*60)
    print("🚀 HarborAI 异步调用示例")
    print("="*60)
    
    try:
        # 创建客户端
        client = create_client()
        print("✅ HarborAI 客户端初始化成功")
        
        # 性能对比测试
        await performance_comparison(client)
        
        # 并发调用不同模型
        await concurrent_different_models(client)
        
        # 限流请求示例
        rate_limit_questions = [
            "什么是异步编程？",
            "Python asyncio的优势是什么？",
            "如何优化API调用性能？"
        ]
        await rate_limited_requests(client, rate_limit_questions)
        
        print(f"\n🎉 所有异步调用示例执行完成！")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        print("\n💡 请检查:")
        print("1. 是否正确配置了环境变量")
        print("2. 网络连接是否正常")
        print("3. API密钥是否有效")


if __name__ == "__main__":
    asyncio.run(main())