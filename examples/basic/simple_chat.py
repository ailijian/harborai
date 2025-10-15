#!/usr/bin/env python3
"""
HarborAI 简单聊天调用示例

这个示例展示了如何使用HarborAI进行最基本的模型调用，
与OpenAI SDK的使用方式完全一致。

场景描述:
- 基础的问答对话
- 展示OpenAI兼容接口
- 基础错误处理和统计信息

应用价值:
- 快速验证API连接
- 学习基础调用语法
- 测试不同模型的响应质量
"""

import os
import time
from typing import Dict, Any
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


def simple_chat_example(client: HarborAI, question: str, model: str = "deepseek-chat") -> Dict[str, Any]:
    """
    简单聊天调用示例
    
    Args:
        client: HarborAI客户端
        question: 用户问题
        model: 使用的模型名称
        
    Returns:
        Dict: 包含响应和统计信息的字典
    """
    print(f"\n🤖 正在调用模型: {model}")
    print(f"❓ 用户问题: {question}")
    
    # 构建消息
    messages = [
        {"role": "user", "content": question}
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    try:
        # 调用模型 - 与OpenAI SDK完全一致的语法
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        # 提取响应内容
        answer = response.choices[0].message.content
        usage = response.usage
        
        # 打印响应
        print(f"\n💬 模型回答:")
        print(f"{answer}")
        
        # 打印统计信息
        print(f"\n📊 调用统计:")
        print(f"- 模型: {model}")
        print(f"- 耗时: {elapsed_time:.2f}秒")
        print(f"- 输入tokens: {usage.prompt_tokens}")
        print(f"- 输出tokens: {usage.completion_tokens}")
        print(f"- 总tokens: {usage.total_tokens}")
        
        # 估算成本 (假设价格)
        estimated_cost = (usage.prompt_tokens * 0.0001 + usage.completion_tokens * 0.0002) / 1000
        print(f"- 估算成本: ¥{estimated_cost:.6f}")
        
        return {
            "success": True,
            "answer": answer,
            "usage": usage,
            "elapsed_time": elapsed_time,
            "estimated_cost": estimated_cost
        }
        
    except Exception as e:
        print(f"\n❌ 调用失败: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": time.time() - start_time
        }


def multi_model_comparison(client: HarborAI, question: str):
    """
    多模型对比示例 - 对比 DeepSeek 不同模型
    
    Args:
        client: HarborAI客户端
        question: 测试问题
    """
    print("\n" + "="*60)
    print("🔄 DeepSeek 多模型对比测试")
    print("="*60)
    
    # 测试不同的 DeepSeek 模型
    models = ["deepseek-chat", "deepseek-reasoner"]
    
    results = []
    
    for model in models:
        try:
            # 统一使用 DeepSeek 客户端
            test_client = client
                
            result = simple_chat_example(test_client, question, model)
            results.append({"model": model, **result})
            
        except Exception as e:
            print(f"❌ 模型 {model} 测试失败: {e}")
            results.append({"model": model, "success": False, "error": str(e)})
    
    # 打印对比结果
    print(f"\n📈 模型对比总结:")
    print("-" * 50)
    for result in results:
        if result["success"]:
            print(f"✅ {result['model']}: {result['elapsed_time']:.2f}s, "
                  f"{result['usage'].total_tokens} tokens, "
                  f"¥{result['estimated_cost']:.6f}")
        else:
            print(f"❌ {result['model']}: 调用失败")


def interactive_chat(client: HarborAI):
    """
    交互式聊天示例
    
    Args:
        client: HarborAI客户端
    """
    print("\n" + "="*60)
    print("💬 交互式聊天模式 (输入 'quit' 退出)")
    print("="*60)
    
    while True:
        try:
            question = input("\n👤 你: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                print("👋 再见!")
                break
                
            if not question:
                continue
                
            simple_chat_example(client, question)
            
        except KeyboardInterrupt:
            print("\n👋 再见!")
            break
        except Exception as e:
            print(f"❌ 发生错误: {e}")


def main():
    """主函数"""
    print("="*60)
    print("🚀 HarborAI 简单聊天调用示例")
    print("="*60)
    
    try:
        # 创建客户端
        client = create_client()
        print("✅ HarborAI 客户端初始化成功")
        
        # 示例问题
        test_questions = [
            "什么是人工智能？",
            "请用一句话解释量子计算",
            "Python和JavaScript的主要区别是什么？"
        ]
        
        # 运行基础示例
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 示例 {i}/{len(test_questions)}")
            print("-" * 40)
            simple_chat_example(client, question)
        
        # 多模型对比
        multi_model_comparison(client, "请简单介绍一下机器学习")
        
        # 交互式聊天 (可选)
        user_input = input("\n🤔 是否进入交互式聊天模式？(y/n): ").strip().lower()
        if user_input in ['y', 'yes', '是']:
            interactive_chat(client)
            
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        print("\n💡 请检查:")
        print("1. 是否正确配置了环境变量")
        print("2. 网络连接是否正常")
        print("3. API密钥是否有效")


if __name__ == "__main__":
    main()