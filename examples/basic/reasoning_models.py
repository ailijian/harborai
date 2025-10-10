#!/usr/bin/env python3
"""
HarborAI 推理模型调用示例

这个示例展示了如何使用HarborAI调用推理模型，
包括DeepSeek-R1等专门用于复杂推理任务的模型。

场景描述:
- 推理模型调用
- 复杂问题解决
- 思维链展示

应用价值:
- 解决复杂逻辑问题
- 数学计算和证明
- 多步骤推理任务
"""

import os
import time
import asyncio
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加本地源码路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from harborai import HarborAI
except ImportError:
    print("❌ 无法导入 HarborAI，请检查路径配置")
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


def reasoning_call(client: HarborAI, question: str, model: str = "deepseek-reasoner") -> Dict[str, Any]:
    """
    推理模型调用
    
    Args:
        client: HarborAI客户端
        question: 推理问题
        model: 推理模型名称
        
    Returns:
        Dict: 包含推理过程和结果的字典
    """
    print(f"\n🧠 推理问题: {question}")
    print("🔍 正在进行深度推理...")
    
    start_time = time.time()
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user", 
                    "content": question
                }
            ],
            temperature=0.1,  # 推理任务使用较低温度
            max_tokens=2000,
            # 推理模型特定参数
            reasoning_effort="medium"  # 推理强度: low, medium, high
        )
        
        elapsed_time = time.time() - start_time
        
        # 提取推理内容
        reasoning_content = response.choices[0].message.reasoning if hasattr(response.choices[0].message, 'reasoning') else None
        final_answer = response.choices[0].message.content
        usage = response.usage
        
        result = {
            "success": True,
            "question": question,
            "reasoning_process": reasoning_content,
            "final_answer": final_answer,
            "usage": usage,
            "elapsed_time": elapsed_time,
            "model": model
        }
        
        # 显示结果
        print(f"✅ 推理完成 (耗时: {elapsed_time:.2f}秒)")
        
        if reasoning_content:
            print(f"\n🤔 推理过程:")
            print("-" * 50)
            print(reasoning_content[:500] + "..." if len(reasoning_content) > 500 else reasoning_content)
        
        print(f"\n💡 最终答案:")
        print("-" * 50)
        print(final_answer)
        
        print(f"\n📊 使用统计:")
        print(f"   - 输入tokens: {usage.prompt_tokens}")
        print(f"   - 输出tokens: {usage.completion_tokens}")
        print(f"   - 总tokens: {usage.total_tokens}")
        
        return result
        
    except Exception as e:
        print(f"❌ 推理调用失败: {e}")
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "model": model
        }


async def async_reasoning_call(client: HarborAI, question: str, model: str = "deepseek-reasoner") -> Dict[str, Any]:
    """
    异步推理模型调用
    
    Args:
        client: HarborAI客户端
        question: 推理问题
        model: 推理模型名称
        
    Returns:
        Dict: 包含推理过程和结果的字典
    """
    print(f"\n🧠 异步推理问题: {question}")
    print("🔍 正在进行异步深度推理...")
    
    start_time = time.time()
    
    try:
        response = await client.chat.completions.acreate(
            model=model,
            messages=[
                {
                    "role": "user", 
                    "content": question
                }
            ],
            temperature=0.1,
            max_tokens=2000,
            reasoning_effort="high"  # 异步调用可以使用更高的推理强度
        )
        
        elapsed_time = time.time() - start_time
        
        reasoning_content = response.choices[0].message.reasoning if hasattr(response.choices[0].message, 'reasoning') else None
        final_answer = response.choices[0].message.content
        usage = response.usage
        
        result = {
            "success": True,
            "question": question,
            "reasoning_process": reasoning_content,
            "final_answer": final_answer,
            "usage": usage,
            "elapsed_time": elapsed_time,
            "model": model
        }
        
        print(f"✅ 异步推理完成 (耗时: {elapsed_time:.2f}秒)")
        
        if reasoning_content:
            print(f"\n🤔 推理过程:")
            print("-" * 50)
            print(reasoning_content[:500] + "..." if len(reasoning_content) > 500 else reasoning_content)
        
        print(f"\n💡 最终答案:")
        print("-" * 50)
        print(final_answer)
        
        return result
        
    except Exception as e:
        print(f"❌ 异步推理调用失败: {e}")
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "model": model
        }


def streaming_reasoning(client: HarborAI, question: str, model: str = "deepseek-reasoner") -> Dict[str, Any]:
    """
    流式推理模型调用
    
    Args:
        client: HarborAI客户端
        question: 推理问题
        model: 推理模型名称
        
    Returns:
        Dict: 包含推理过程和结果的字典
    """
    print(f"\n🧠 流式推理问题: {question}")
    print("🔍 正在进行流式推理...")
    
    start_time = time.time()
    full_reasoning = ""
    full_answer = ""
    chunk_count = 0
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user", 
                    "content": question
                }
            ],
            temperature=0.1,
            max_tokens=2000,
            stream=True,
            reasoning_effort="medium"
        )
        
        print("\n🤔 推理过程:")
        print("-" * 50)
        
        for chunk in stream:
            chunk_count += 1
            
            # 处理推理过程
            if hasattr(chunk.choices[0].delta, 'reasoning') and chunk.choices[0].delta.reasoning:
                reasoning_content = chunk.choices[0].delta.reasoning
                full_reasoning += reasoning_content
                print(reasoning_content, end="", flush=True)
                time.sleep(0.02)
            
            # 处理最终答案
            elif chunk.choices[0].delta.content:
                if not full_answer:  # 第一次输出答案时显示标题
                    print(f"\n\n💡 最终答案:")
                    print("-" * 50)
                
                content = chunk.choices[0].delta.content
                full_answer += content
                print(content, end="", flush=True)
                time.sleep(0.02)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n\n✅ 流式推理完成")
        print(f"📊 统计信息:")
        print(f"   - 推理时间: {elapsed_time:.2f}秒")
        print(f"   - 数据块数: {chunk_count}")
        print(f"   - 推理长度: {len(full_reasoning)}字符")
        print(f"   - 答案长度: {len(full_answer)}字符")
        
        return {
            "success": True,
            "question": question,
            "reasoning_process": full_reasoning,
            "final_answer": full_answer,
            "elapsed_time": elapsed_time,
            "chunk_count": chunk_count,
            "model": model
        }
        
    except Exception as e:
        print(f"❌ 流式推理调用失败: {e}")
        return {
            "success": False,
            "question": question,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
            "model": model
        }


def compare_reasoning_efforts(client: HarborAI, question: str, model: str = "deepseek-reasoner"):
    """
    对比不同推理强度
    
    Args:
        client: HarborAI客户端
        question: 推理问题
        model: 推理模型名称
    """
    print("\n" + "="*60)
    print("⚖️  不同推理强度对比")
    print("="*60)
    
    efforts = ["low", "medium", "high"]
    results = []
    
    for effort in efforts:
        print(f"\n🎯 推理强度: {effort}")
        print("-" * 30)
        
        start_time = time.time()
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.1,
                max_tokens=1500,
                reasoning_effort=effort
            )
            
            elapsed_time = time.time() - start_time
            reasoning_content = response.choices[0].message.reasoning if hasattr(response.choices[0].message, 'reasoning') else ""
            final_answer = response.choices[0].message.content
            usage = response.usage
            
            result = {
                "effort": effort,
                "elapsed_time": elapsed_time,
                "reasoning_length": len(reasoning_content),
                "answer_length": len(final_answer),
                "total_tokens": usage.total_tokens,
                "reasoning_content": reasoning_content[:200] + "..." if len(reasoning_content) > 200 else reasoning_content,
                "final_answer": final_answer[:200] + "..." if len(final_answer) > 200 else final_answer
            }
            
            results.append(result)
            
            print(f"⏱️  耗时: {elapsed_time:.2f}秒")
            print(f"📝 推理长度: {len(reasoning_content)}字符")
            print(f"💬 答案长度: {len(final_answer)}字符")
            print(f"🎯 总tokens: {usage.total_tokens}")
            print(f"💡 答案预览: {final_answer[:100]}...")
            
        except Exception as e:
            print(f"❌ 推理强度 {effort} 失败: {e}")
            results.append({
                "effort": effort,
                "error": str(e)
            })
    
    # 对比总结
    print(f"\n📊 推理强度对比总结:")
    print("-" * 40)
    successful_results = [r for r in results if "error" not in r]
    
    if successful_results:
        for result in successful_results:
            print(f"{result['effort']:>6}: {result['elapsed_time']:>6.2f}秒, "
                  f"{result['total_tokens']:>4}tokens, "
                  f"推理{result['reasoning_length']:>4}字符")
    
    return results


async def batch_reasoning_problems(client: HarborAI, problems: List[Dict[str, str]], model: str = "deepseek-reasoner"):
    """
    批量处理推理问题
    
    Args:
        client: HarborAI客户端
        problems: 问题列表，每个问题包含category和question
        model: 推理模型名称
    """
    print("\n" + "="*60)
    print("📚 批量推理问题处理")
    print("="*60)
    
    async def solve_problem(problem: Dict[str, str], index: int):
        """解决单个问题"""
        category = problem["category"]
        question = problem["question"]
        
        print(f"\n🔢 问题 {index+1} ({category}): {question[:50]}...")
        
        try:
            response = await client.chat.completions.acreate(
                model=model,
                messages=[{"role": "user", "content": question}],
                temperature=0.1,
                max_tokens=1500,
                reasoning_effort="medium"
            )
            
            reasoning_content = response.choices[0].message.reasoning if hasattr(response.choices[0].message, 'reasoning') else ""
            final_answer = response.choices[0].message.content
            
            print(f"✅ 问题 {index+1} 解决完成")
            print(f"   答案: {final_answer[:100]}...")
            
            return {
                "index": index,
                "category": category,
                "question": question,
                "reasoning": reasoning_content,
                "answer": final_answer,
                "success": True
            }
            
        except Exception as e:
            print(f"❌ 问题 {index+1} 解决失败: {e}")
            return {
                "index": index,
                "category": category,
                "question": question,
                "error": str(e),
                "success": False
            }
    
    # 并发处理所有问题
    start_time = time.time()
    tasks = [solve_problem(problem, i) for i, problem in enumerate(problems)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # 统计结果
    successful = [r for r in results if isinstance(r, dict) and r.get("success")]
    failed = [r for r in results if isinstance(r, dict) and not r.get("success")]
    
    print(f"\n📊 批量处理完成:")
    print(f"   总耗时: {total_time:.2f}秒")
    print(f"   成功: {len(successful)}/{len(problems)}")
    print(f"   失败: {len(failed)}")
    
    # 按类别统计
    if successful:
        categories = {}
        for result in successful:
            cat = result["category"]
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        print(f"\n📈 按类别统计:")
        for cat, count in categories.items():
            print(f"   {cat}: {count}个问题")
    
    return results


def interactive_reasoning_session(client: HarborAI, model: str = "deepseek-reasoner"):
    """
    交互式推理会话
    
    Args:
        client: HarborAI客户端
        model: 推理模型名称
    """
    print("\n" + "="*60)
    print("🧠 交互式推理会话")
    print("="*60)
    print("💡 输入复杂的推理问题，AI将展示详细的思考过程")
    print("💡 输入 'quit' 或 'exit' 退出")
    print("💡 输入 'effort low/medium/high' 调整推理强度")
    
    current_effort = "medium"
    
    while True:
        try:
            user_input = input(f"\n🤔 您的推理问题 (当前推理强度: {current_effort}): ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("👋 推理会话结束！")
                break
            
            if user_input.startswith('effort '):
                new_effort = user_input.split(' ', 1)[1].strip()
                if new_effort in ['low', 'medium', 'high']:
                    current_effort = new_effort
                    print(f"✅ 推理强度已调整为: {current_effort}")
                else:
                    print("❌ 推理强度必须是: low, medium, high")
                continue
            
            if not user_input:
                continue
            
            # 进行推理
            reasoning_call(client, user_input, model)
            
        except KeyboardInterrupt:
            print("\n\n👋 推理会话被中断！")
            break
        except Exception as e:
            print(f"\n❌ 推理出错: {e}")


async def main():
    """主函数"""
    print("="*60)
    print("🧠 HarborAI 推理模型调用示例")
    print("="*60)
    
    try:
        # 创建客户端
        client = create_client()
        print("✅ HarborAI 客户端初始化成功")
        
        # 测试推理问题
        reasoning_problems = [
            "一个农夫有17只羊，除了9只以外都死了，请问农夫还有几只羊？请详细解释你的推理过程。",
            "如果今天是星期三，那么100天后是星期几？请展示计算步骤。",
            "有3个开关控制3盏灯，你在另一个房间看不到灯，只能进入房间一次，如何确定哪个开关控制哪盏灯？",
            "一个数列：2, 6, 12, 20, 30, ?，请找出规律并计算下一个数。"
        ]
        
        # 1. 基础推理调用
        print("\n🔹 1. 基础推理调用示例")
        reasoning_call(client, reasoning_problems[0])
        
        # 2. 异步推理调用
        print("\n🔹 2. 异步推理调用示例")
        await async_reasoning_call(client, reasoning_problems[1])
        
        # 3. 流式推理调用
        print("\n🔹 3. 流式推理调用示例")
        streaming_reasoning(client, reasoning_problems[2])
        
        # 4. 不同推理强度对比
        print("\n🔹 4. 不同推理强度对比")
        compare_reasoning_efforts(client, reasoning_problems[3])
        
        # 5. 批量推理问题
        print("\n🔹 5. 批量推理问题处理")
        batch_problems = [
            {"category": "逻辑推理", "question": "所有的猫都是动物，所有的动物都需要食物，因此所有的猫都需要食物。这个推理是否正确？"},
            {"category": "数学计算", "question": "计算 (2^10 * 3^5) / (2^7 * 3^3) 的值"},
            {"category": "概率问题", "question": "抛硬币3次，至少出现一次正面的概率是多少？"},
            {"category": "几何问题", "question": "一个圆的半径是5cm，求其面积和周长"}
        ]
        await batch_reasoning_problems(client, batch_problems)
        
        # 6. 交互式推理会话
        print("\n🔹 6. 交互式推理会话")
        choice = input("是否开始交互式推理会话？(y/n): ").strip().lower()
        if choice in ['y', 'yes', '是']:
            interactive_reasoning_session(client)
        
        print(f"\n🎉 所有推理模型示例执行完成！")
        
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        print("\n💡 请检查:")
        print("1. 是否正确配置了环境变量")
        print("2. 网络连接是否正常")
        print("3. API密钥是否有效")
        print("4. 是否有推理模型的访问权限")


if __name__ == "__main__":
    asyncio.run(main())