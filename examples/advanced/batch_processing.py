#!/usr/bin/env python3
"""
批处理优化演示

这个示例展示了 HarborAI 的批处理优化功能，包括：
1. 原生异步批处理
2. 并发控制和限流
3. 结构化输出的批处理
4. 推理模型的批处理
5. 流式批处理
6. 错误处理和重试

场景：
- 需要处理大量文本数据
- 批量生成结构化内容
- 并发调用多个AI服务
- 优化处理速度和资源使用

价值：
- 使用 HarborAI 原生异步支持，性能更优
- 智能并发控制，避免API限流
- 统一的错误处理和重试机制
- 支持多种输出格式的批处理
"""

import asyncio
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import json
import os

# 正确的 HarborAI 导入方式
from harborai import HarborAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_client():
    """获取 HarborAI 客户端"""
    # 优先使用 DeepSeek
    if os.getenv('DEEPSEEK_API_KEY'):
        return HarborAI(
            api_key=os.getenv('DEEPSEEK_API_KEY'),
            base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        ), "deepseek-chat"
    
    # 其次使用 Ernie
    if os.getenv('ERNIE_API_KEY'):
        return HarborAI(
            api_key=os.getenv('ERNIE_API_KEY'),
            base_url=os.getenv('ERNIE_BASE_URL', 'https://aip.baidubce.com')
        ), "ernie-3.5-8k"
    
    # 最后使用 Doubao
    if os.getenv('DOUBAO_API_KEY'):
        return HarborAI(
            api_key=os.getenv('DOUBAO_API_KEY'),
            base_url=os.getenv('DOUBAO_BASE_URL', 'https://ark.cn-beijing.volces.com')
        ), "doubao-1-5-pro-32k-character-250715"
    
    return None, None

class BatchProcessor:
    """批处理器"""
    
    def __init__(self, max_concurrent: int = 5, delay_between_batches: float = 1.0):
        self.client, self.model = get_client()
        if not self.client:
            raise ValueError("请至少设置一个 API Key (DEEPSEEK_API_KEY, ERNIE_API_KEY, 或 DOUBAO_API_KEY)")
        
        self.max_concurrent = max_concurrent
        self.delay_between_batches = delay_between_batches
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        self.start_time = None
    
    async def process_single_request(self, messages: List[Dict], **kwargs) -> Tuple[bool, Any, str]:
        """处理单个请求"""
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    fallback=["deepseek-chat", "ernie-3.5-8k"],
                    retry_policy={
                        "max_attempts": 2,
                        "base_delay": 1.0,
                        "max_delay": 5.0
                    },
                    timeout=30.0,
                    **kwargs
                )
                
                self.successful_requests += 1
                if response.usage:
                    self.total_tokens += response.usage.total_tokens
                
                return True, response, ""
                
            except Exception as e:
                self.failed_requests += 1
                error_msg = str(e)
                logger.warning(f"请求失败: {error_msg}")
                return False, None, error_msg
            finally:
                self.total_requests += 1
    
    async def process_batch(self, batch_data: List[Dict], **kwargs) -> List[Dict]:
        """处理一批请求"""
        if self.start_time is None:
            self.start_time = time.time()
        
        # 创建异步任务
        tasks = []
        for item in batch_data:
            messages = item.get('messages', [])
            task = self.process_single_request(messages, **kwargs)
            tasks.append(task)
        
        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        processed_results = []
        for i, (item, result) in enumerate(zip(batch_data, results)):
            if isinstance(result, Exception):
                processed_results.append({
                    'index': i,
                    'input': item,
                    'success': False,
                    'error': str(result),
                    'response': None
                })
            else:
                success, response, error = result
                processed_results.append({
                    'index': i,
                    'input': item,
                    'success': success,
                    'error': error,
                    'response': response
                })
        
        return processed_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.successful_requests / max(self.total_requests, 1),
            'total_tokens': self.total_tokens,
            'elapsed_time': elapsed,
            'requests_per_second': self.total_requests / max(elapsed, 1),
            'tokens_per_second': self.total_tokens / max(elapsed, 1)
        }

async def demo_basic_batch_processing():
    """演示基本批处理"""
    print("\n🔄 演示基本批处理")
    print("=" * 50)
    
    # 准备批处理数据
    batch_data = [
        {'messages': [{'role': 'user', 'content': '什么是人工智能？'}]},
        {'messages': [{'role': 'user', 'content': '解释机器学习的概念'}]},
        {'messages': [{'role': 'user', 'content': '深度学习有什么特点？'}]},
        {'messages': [{'role': 'user', 'content': '自然语言处理的应用'}]},
        {'messages': [{'role': 'user', 'content': '计算机视觉技术介绍'}]}
    ]
    
    processor = BatchProcessor(max_concurrent=3)
    
    print(f"📝 处理 {len(batch_data)} 个请求...")
    start_time = time.time()
    
    results = await processor.process_batch(batch_data)
    
    elapsed = time.time() - start_time
    
    # 显示结果
    print(f"\n✅ 批处理完成，耗时: {elapsed:.2f}秒")
    
    for result in results:
        if result['success']:
            content = result['response'].choices[0].message.content[:50] if result['response'] else "无内容"
            print(f"   ✅ 请求 {result['index'] + 1}: {content}...")
        else:
            print(f"   ❌ 请求 {result['index'] + 1}: {result['error']}")
    
    # 显示统计信息
    stats = processor.get_statistics()
    print(f"\n📊 统计信息:")
    print(f"   成功率: {stats['success_rate']:.1%}")
    print(f"   总Token: {stats['total_tokens']}")
    print(f"   请求/秒: {stats['requests_per_second']:.2f}")

async def demo_structured_batch_processing():
    """演示结构化输出的批处理"""
    print("\n📊 演示结构化输出批处理")
    print("=" * 50)
    
    # 定义结构化输出 schema
    schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "summary": {"type": "string"},
            "key_points": {
                "type": "array",
                "items": {"type": "string"}
            },
            "difficulty": {
                "type": "string",
                "enum": ["初级", "中级", "高级"]
            }
        },
        "required": ["topic", "summary", "key_points", "difficulty"],
        "additionalProperties": False
    }
    
    # 准备批处理数据
    topics = [
        "Python编程基础",
        "数据结构与算法",
        "Web开发框架",
        "数据库设计",
        "云计算架构"
    ]
    
    batch_data = []
    for topic in topics:
        batch_data.append({
            'messages': [
                {'role': 'user', 'content': f'请分析"{topic}"这个技术主题'}
            ]
        })
    
    processor = BatchProcessor(max_concurrent=2)
    
    print(f"📝 处理 {len(batch_data)} 个结构化输出请求...")
    
    results = await processor.process_batch(
        batch_data,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "TopicAnalysis",
                "schema": schema,
                "strict": True
            }
        }
    )
    
    # 显示结果
    print(f"\n✅ 结构化批处理完成")
    
    for result in results:
        if result['success'] and result['response']:
            parsed = result['response'].parsed
            if parsed:
                print(f"\n   📋 主题: {parsed.get('topic', 'N/A')}")
                print(f"      难度: {parsed.get('difficulty', 'N/A')}")
                print(f"      要点: {len(parsed.get('key_points', []))} 个")
            else:
                print(f"   ⚠️ 请求 {result['index'] + 1}: 解析失败")
        else:
            print(f"   ❌ 请求 {result['index'] + 1}: {result['error']}")

async def demo_reasoning_batch_processing():
    """演示推理模型的批处理"""
    print("\n🧠 演示推理模型批处理")
    print("=" * 50)
    
    # 准备需要深度思考的问题
    complex_questions = [
        "如何设计一个高可用的分布式系统？",
        "人工智能对就业市场的长期影响是什么？",
        "区块链技术在金融领域的应用前景如何？"
    ]
    
    batch_data = []
    for question in complex_questions:
        batch_data.append({
            'messages': [
                {'role': 'user', 'content': question}
            ]
        })
    
    # 创建支持推理模型的处理器
    processor = BatchProcessor(max_concurrent=2)
    
    # 尝试使用推理模型
    if os.getenv('DEEPSEEK_API_KEY'):
        processor.model = "deepseek-reasoner"
    
    print(f"📝 处理 {len(batch_data)} 个复杂推理问题...")
    
    results = await processor.process_batch(batch_data)
    
    # 显示结果
    print(f"\n✅ 推理批处理完成")
    
    for i, result in enumerate(results):
        if result['success'] and result['response']:
            response = result['response']
            content = response.choices[0].message.content[:100] if response.choices else "无内容"
            
            print(f"\n   🤔 问题 {i + 1}: {complex_questions[i]}")
            print(f"      答案: {content}...")
            
            # 检查是否有思考过程
            if hasattr(response.choices[0].message, 'reasoning_content'):
                reasoning = response.choices[0].message.reasoning_content
                if reasoning:
                    print(f"      思考: {reasoning[:80]}...")
                else:
                    print("      思考: 使用了普通模型")
            else:
                print("      思考: 无思考过程记录")
        else:
            print(f"   ❌ 问题 {i + 1}: {result['error']}")

async def demo_stream_batch_processing():
    """演示流式批处理"""
    print("\n🌊 演示流式批处理")
    print("=" * 50)
    
    client, model = get_client()
    if not client:
        print("❌ 请至少设置一个 API Key")
        return
    
    # 准备需要长回答的问题
    questions = [
        "详细解释深度学习的工作原理",
        "分析云计算的发展趋势和挑战",
        "介绍区块链技术的核心概念"
    ]
    
    print(f"📝 开始 {len(questions)} 个流式请求...")
    
    async def process_stream_request(question: str, index: int):
        """处理单个流式请求"""
        print(f"\n🌊 流式请求 {index + 1}: {question}")
        
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{'role': 'user', 'content': question}],
                stream=True,
                fallback=["deepseek-chat", "ernie-3.5-8k"],
                retry_policy={
                    "max_attempts": 2,
                    "base_delay": 1.0,
                    "max_delay": 5.0
                },
                timeout=60.0
            )
            
            content_parts = []
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    content_parts.append(content)
                    print(content, end="", flush=True)
            
            print(f"\n   ✅ 完成，共 {len(content_parts)} 个片段")
            return True, len(content_parts)
            
        except Exception as e:
            print(f"\n   ❌ 失败: {e}")
            return False, 0
    
    # 并发处理流式请求
    tasks = [
        process_stream_request(question, i) 
        for i, question in enumerate(questions)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 统计结果
    successful = sum(1 for result in results if isinstance(result, tuple) and result[0])
    total_chunks = sum(result[1] for result in results if isinstance(result, tuple) and result[0])
    
    print(f"\n📊 流式批处理统计:")
    print(f"   成功: {successful}/{len(questions)}")
    print(f"   总片段: {total_chunks}")

async def demo_large_scale_batch():
    """演示大规模批处理"""
    print("\n🚀 演示大规模批处理")
    print("=" * 50)
    
    # 生成大量测试数据
    batch_size = 20
    questions = [
        f"请简单介绍第{i+1}个人工智能概念" 
        for i in range(batch_size)
    ]
    
    batch_data = [
        {'messages': [{'role': 'user', 'content': question}]}
        for question in questions
    ]
    
    # 使用更高的并发数
    processor = BatchProcessor(max_concurrent=8, delay_between_batches=0.5)
    
    print(f"📝 处理 {len(batch_data)} 个大规模请求...")
    start_time = time.time()
    
    # 分批处理
    chunk_size = 10
    all_results = []
    
    for i in range(0, len(batch_data), chunk_size):
        chunk = batch_data[i:i + chunk_size]
        print(f"   处理批次 {i//chunk_size + 1}/{(len(batch_data) + chunk_size - 1)//chunk_size}")
        
        chunk_results = await processor.process_batch(chunk)
        all_results.extend(chunk_results)
        
        # 批次间延迟
        if i + chunk_size < len(batch_data):
            await asyncio.sleep(processor.delay_between_batches)
    
    elapsed = time.time() - start_time
    
    # 显示统计
    stats = processor.get_statistics()
    
    print(f"\n✅ 大规模批处理完成")
    print(f"📊 性能统计:")
    print(f"   总请求: {stats['total_requests']}")
    print(f"   成功率: {stats['success_rate']:.1%}")
    print(f"   总耗时: {elapsed:.2f}秒")
    print(f"   平均QPS: {stats['requests_per_second']:.2f}")
    print(f"   总Token: {stats['total_tokens']}")
    print(f"   Token/秒: {stats['tokens_per_second']:.2f}")

async def demo_error_handling_batch():
    """演示错误处理和重试"""
    print("\n🛡️ 演示错误处理和重试")
    print("=" * 50)
    
    # 准备包含可能失败的请求
    batch_data = [
        {'messages': [{'role': 'user', 'content': '正常请求：什么是AI？'}]},
        {'messages': [{'role': 'user', 'content': '超长请求：' + 'x' * 10000}]},  # 可能失败
        {'messages': [{'role': 'user', 'content': '正常请求：机器学习是什么？'}]},
        {'messages': [{'role': 'user', 'content': ''}]},  # 空请求，可能失败
        {'messages': [{'role': 'user', 'content': '正常请求：深度学习的应用'}]}
    ]
    
    processor = BatchProcessor(max_concurrent=3)
    
    print(f"📝 处理 {len(batch_data)} 个包含错误的请求...")
    
    results = await processor.process_batch(batch_data)
    
    # 分析结果
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n📊 错误处理结果:")
    print(f"   成功: {len(successful_results)}")
    print(f"   失败: {len(failed_results)}")
    
    if failed_results:
        print(f"   失败详情:")
        for result in failed_results:
            print(f"     - {result.request_id}: {result.error}")

async def demo_mixed_format_batch():
    """演示混合格式批处理"""
    print("\n🎭 演示混合格式批处理")
    print("=" * 50)
    
    client, model = get_client()
    if not client:
        print("❌ 请至少设置一个 API Key")
        return
    
    # 定义不同类型的请求
    requests = [
        {
            'type': 'normal',
            'messages': [{'role': 'user', 'content': '什么是云计算？'}],
            'params': {}
        },
        {
            'type': 'structured',
            'messages': [{'role': 'user', 'content': '分析Python编程语言'}],
            'params': {
                'response_format': {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "LanguageAnalysis",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "language": {"type": "string"},
                                "strengths": {"type": "array", "items": {"type": "string"}},
                                "use_cases": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["language", "strengths", "use_cases"]
                        }
                    }
                }
            }
        },
        {
            'type': 'stream',
            'messages': [{'role': 'user', 'content': '详细解释区块链技术'}],
            'params': {'stream': True}
        }
    ]
    
    print(f"📝 处理 {len(requests)} 个混合格式请求...")
    
    async def process_mixed_request(request: Dict, index: int):
        """处理混合格式请求"""
        try:
            if request['type'] == 'stream':
                print(f"\n🌊 流式请求 {index + 1}:")
                
                stream = await client.chat.completions.create(
                    model=model,
                    messages=request['messages'],
                    fallback=["deepseek-chat", "ernie-3.5-8k"],
                    **request['params']
                )
                
                content_parts = []
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        content_parts.append(content)
                        print(content, end="", flush=True)
                
                print(f"\n   ✅ 流式完成，{len(content_parts)} 片段")
                return {'type': 'stream', 'success': True, 'chunks': len(content_parts)}
            
            else:
                response = await client.chat.completions.create(
                    model=model,
                    messages=request['messages'],
                    fallback=["deepseek-chat", "ernie-3.5-8k"],
                    retry_policy={
                        "max_attempts": 2,
                        "base_delay": 1.0,
                        "max_delay": 5.0
                    },
                    **request['params']
                )
                
                if request['type'] == 'structured':
                    return {
                        'type': 'structured', 
                        'success': True, 
                        'parsed': response.parsed
                    }
                else:
                    return {
                        'type': 'normal', 
                        'success': True, 
                        'content': response.choices[0].message.content[:100]
                    }
        
        except Exception as e:
            return {'type': request['type'], 'success': False, 'error': str(e)}
    
    # 并发处理
    tasks = [
        process_mixed_request(request, i) 
        for i, request in enumerate(requests)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 显示结果
    print(f"\n📊 混合格式批处理结果:")
    for i, result in enumerate(results):
        if isinstance(result, dict) and result['success']:
            if result['type'] == 'normal':
                print(f"   ✅ 普通请求 {i + 1}: {result['content']}...")
            elif result['type'] == 'structured':
                print(f"   ✅ 结构化请求 {i + 1}: {result['parsed']}")
            elif result['type'] == 'stream':
                print(f"   ✅ 流式请求 {i + 1}: {result['chunks']} 片段")
        else:
            error = result.get('error', str(result)) if isinstance(result, dict) else str(result)
            print(f"   ❌ 请求 {i + 1}: {error[:50]}...")

async def main():
    """主函数"""
    print("🚀 HarborAI 批处理优化演示")
    print("=" * 60)
    
    # 检查环境变量
    client, model = get_client()
    if not client:
        print("⚠️ 警告: 未设置任何 API Key")
        print("请设置 DEEPSEEK_API_KEY, ERNIE_API_KEY, 或 DOUBAO_API_KEY")
        return
    
    print(f"🔍 使用模型: {model}")
    
    demos = [
        ("基本批处理", demo_basic_batch_processing),
        ("结构化输出批处理", demo_structured_batch_processing),
        ("推理模型批处理", demo_reasoning_batch_processing),
        ("流式批处理", demo_stream_batch_processing),
        ("大规模批处理", demo_large_scale_batch),
        ("错误处理和重试", demo_error_handling_batch),
        ("混合格式批处理", demo_mixed_format_batch)
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
            await asyncio.sleep(1)  # 避免请求过于频繁
        except Exception as e:
            print(f"❌ {name} 演示失败: {e}")
    
    print("\n🎉 批处理演示完成！")
    print("\n💡 关键要点:")
    print("1. 使用原生异步支持，避免 asyncio.to_thread")
    print("2. 通过 Semaphore 控制并发数，避免API限流")
    print("3. 支持普通、结构化、流式等多种格式的批处理")
    print("4. 内置重试和降级机制，提高成功率")
    print("5. 详细的统计信息，便于性能监控")
    print("6. 灵活的错误处理，确保批处理的健壮性")

if __name__ == "__main__":
    asyncio.run(main())