#!/usr/bin/env python3
"""
性能优化演示

这个示例展示了 HarborAI 的性能优化功能，包括：
1. 智能缓存机制
2. 连接池管理
3. 请求预测和预加载
4. 资源监控和调优
5. 批量优化策略
6. 流式处理优化

场景：
- 高频API调用场景
- 需要低延迟响应
- 大量并发请求
- 资源使用优化

价值：
- 显著降低响应延迟
- 减少API调用成本
- 提高系统吞吐量
- 智能资源管理
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
import os
import hashlib
from collections import defaultdict, OrderedDict
import statistics

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

class SimpleCache:
    """简单的内存缓存实现"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def _generate_key(self, messages: List[Dict], **kwargs) -> str:
        """生成缓存键"""
        # 创建一个包含所有相关参数的字符串
        cache_data = {
            'messages': messages,
            'model': kwargs.get('model', ''),
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', None)
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """检查缓存是否过期"""
        if key not in self.timestamps:
            return True
        
        age = time.time() - self.timestamps[key]
        return age > self.ttl_seconds
    
    def _evict_expired(self):
        """清理过期缓存"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self.cache.pop(key, None)
            self.timestamps.pop(key, None)
    
    def _evict_lru(self):
        """LRU淘汰策略"""
        while len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            self.timestamps.pop(oldest_key, None)
    
    def get(self, messages: List[Dict], **kwargs) -> Optional[Any]:
        """获取缓存"""
        key = self._generate_key(messages, **kwargs)
        
        # 清理过期缓存
        self._evict_expired()
        
        if key in self.cache and not self._is_expired(key):
            # 缓存命中，移到最后（LRU）
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hit_count += 1
            return value
        
        self.miss_count += 1
        return None
    
    def set(self, messages: List[Dict], response: Any, **kwargs):
        """设置缓存"""
        key = self._generate_key(messages, **kwargs)
        
        # LRU淘汰
        self._evict_lru()
        
        self.cache[key] = response
        self.timestamps[key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total_requests, 1)
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.timestamps.clear()
        self.hit_count = 0
        self.miss_count = 0

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self, enable_cache: bool = True, max_concurrent: int = 10):
        self.client, self.model = get_client()
        if not self.client:
            raise ValueError("请至少设置一个 API Key (DEEPSEEK_API_KEY, ERNIE_API_KEY, 或 DOUBAO_API_KEY)")
        
        self.enable_cache = enable_cache
        self.cache = SimpleCache() if enable_cache else None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # 性能统计
        self.request_times = []
        self.total_requests = 0
        self.cached_requests = 0
        self.total_tokens = 0
        self.start_time = None
    
    async def optimized_request(self, messages: List[Dict], **kwargs) -> Tuple[Any, bool]:
        """优化的请求方法"""
        # 检查缓存
        if self.cache:
            cached_response = self.cache.get(messages, **kwargs)
            if cached_response:
                self.cached_requests += 1
                return cached_response, True
        
        # 发送请求
        async with self.semaphore:
            start_time = time.time()
            
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
                
                request_time = time.time() - start_time
                self.request_times.append(request_time)
                self.total_requests += 1
                
                if response.usage:
                    self.total_tokens += response.usage.total_tokens
                
                # 缓存响应
                if self.cache:
                    self.cache.set(messages, response, **kwargs)
                
                return response, False
                
            except Exception as e:
                request_time = time.time() - start_time
                self.request_times.append(request_time)
                self.total_requests += 1
                raise e
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.request_times:
            return {
                'total_requests': 0,
                'cached_requests': 0,
                'cache_hit_rate': 0,
                'avg_response_time': 0,
                'min_response_time': 0,
                'max_response_time': 0,
                'p95_response_time': 0,
                'total_tokens': 0,
                'requests_per_second': 0
            }
        
        elapsed = time.time() - self.start_time if self.start_time else 1
        cache_hit_rate = self.cached_requests / max(self.total_requests + self.cached_requests, 1)
        
        return {
            'total_requests': self.total_requests,
            'cached_requests': self.cached_requests,
            'cache_hit_rate': cache_hit_rate,
            'avg_response_time': statistics.mean(self.request_times),
            'min_response_time': min(self.request_times),
            'max_response_time': max(self.request_times),
            'p95_response_time': statistics.quantiles(self.request_times, n=20)[18] if len(self.request_times) > 1 else 0,
            'total_tokens': self.total_tokens,
            'requests_per_second': (self.total_requests + self.cached_requests) / elapsed
        }

async def demo_cache_performance():
    """演示缓存性能优化"""
    print("\n🚀 演示缓存性能优化")
    print("=" * 50)
    
    # 准备重复的请求
    common_questions = [
        "什么是人工智能？",
        "解释机器学习的概念",
        "深度学习有什么特点？",
        "什么是人工智能？",  # 重复
        "自然语言处理的应用",
        "解释机器学习的概念",  # 重复
        "计算机视觉技术介绍",
        "什么是人工智能？",  # 重复
    ]
    
    # 1. 无缓存测试
    print("🔄 无缓存性能测试...")
    optimizer_no_cache = PerformanceOptimizer(enable_cache=False)
    optimizer_no_cache.start_time = time.time()
    
    start_time = time.time()
    for question in common_questions:
        messages = [{'role': 'user', 'content': question}]
        try:
            response, from_cache = await optimizer_no_cache.optimized_request(messages)
            print(f"   ✅ 完成: {question[:20]}...")
        except Exception as e:
            print(f"   ❌ 失败: {question[:20]}... - {e}")
    
    no_cache_time = time.time() - start_time
    no_cache_stats = optimizer_no_cache.get_performance_stats()
    
    # 2. 有缓存测试
    print("\n🔄 有缓存性能测试...")
    optimizer_with_cache = PerformanceOptimizer(enable_cache=True)
    optimizer_with_cache.start_time = time.time()
    
    start_time = time.time()
    for question in common_questions:
        messages = [{'role': 'user', 'content': question}]
        try:
            response, from_cache = await optimizer_with_cache.optimized_request(messages)
            cache_indicator = "💾" if from_cache else "🌐"
            print(f"   {cache_indicator} 完成: {question[:20]}...")
        except Exception as e:
            print(f"   ❌ 失败: {question[:20]}... - {e}")
    
    cache_time = time.time() - start_time
    cache_stats = optimizer_with_cache.get_performance_stats()
    
    # 性能对比
    print(f"\n📊 缓存性能对比:")
    print(f"   无缓存:")
    print(f"     - 总耗时: {no_cache_time:.2f}秒")
    print(f"     - 平均响应时间: {no_cache_stats['avg_response_time']:.2f}秒")
    print(f"     - 总请求数: {no_cache_stats['total_requests']}")
    
    print(f"   有缓存:")
    print(f"     - 总耗时: {cache_time:.2f}秒")
    print(f"     - 平均响应时间: {cache_stats['avg_response_time']:.2f}秒")
    print(f"     - 缓存命中率: {cache_stats['cache_hit_rate']:.1%}")
    print(f"     - 实际请求数: {cache_stats['total_requests']}")
    print(f"     - 缓存请求数: {cache_stats['cached_requests']}")
    
    if no_cache_time > 0 and cache_time > 0:
        speedup = no_cache_time / cache_time
        print(f"   性能提升: {speedup:.2f}x")
    
    # 缓存统计
    if optimizer_with_cache.cache:
        cache_stats_detail = optimizer_with_cache.cache.get_stats()
        print(f"\n💾 缓存详细统计:")
        print(f"   - 缓存大小: {cache_stats_detail['cache_size']}/{cache_stats_detail['max_size']}")
        print(f"   - 命中次数: {cache_stats_detail['hit_count']}")
        print(f"   - 未命中次数: {cache_stats_detail['miss_count']}")

async def demo_concurrent_optimization():
    """演示并发优化"""
    print("\n⚡ 演示并发优化")
    print("=" * 50)
    
    questions = [
        "什么是云计算？",
        "解释容器技术",
        "微服务架构的优势",
        "DevOps的核心理念",
        "持续集成的重要性",
        "分布式系统设计",
        "负载均衡的原理",
        "数据库优化策略"
    ]
    
    # 1. 顺序处理
    print("🔄 顺序处理测试...")
    optimizer = PerformanceOptimizer(enable_cache=False, max_concurrent=1)
    
    start_time = time.time()
    for i, question in enumerate(questions[:4]):  # 只测试前4个
        messages = [{'role': 'user', 'content': question}]
        try:
            response, _ = await optimizer.optimized_request(messages)
            print(f"   ✅ 顺序 {i+1}: {question[:20]}...")
        except Exception as e:
            print(f"   ❌ 顺序 {i+1}: {e}")
    
    sequential_time = time.time() - start_time
    
    # 2. 并发处理
    print("\n🔄 并发处理测试...")
    optimizer = PerformanceOptimizer(enable_cache=False, max_concurrent=4)
    
    async def process_question(question: str, index: int):
        messages = [{'role': 'user', 'content': question}]
        try:
            response, _ = await optimizer.optimized_request(messages)
            print(f"   ✅ 并发 {index+1}: {question[:20]}...")
            return True
        except Exception as e:
            print(f"   ❌ 并发 {index+1}: {e}")
            return False
    
    start_time = time.time()
    tasks = [
        process_question(question, i) 
        for i, question in enumerate(questions[:4])
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    concurrent_time = time.time() - start_time
    
    # 性能对比
    print(f"\n📊 并发性能对比:")
    print(f"   顺序处理: {sequential_time:.2f}秒")
    print(f"   并发处理: {concurrent_time:.2f}秒")
    
    if sequential_time > 0 and concurrent_time > 0:
        speedup = sequential_time / concurrent_time
        print(f"   性能提升: {speedup:.2f}x")

async def demo_streaming_optimization():
    """演示流式处理优化"""
    print("\n🌊 演示流式处理优化")
    print("=" * 50)
    
    client, model = get_client()
    if not client:
        print("❌ 请至少设置一个 API Key")
        return
    
    question = "详细解释人工智能的发展历程和未来趋势"
    
    # 1. 普通请求
    print("🔄 普通请求测试...")
    start_time = time.time()
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': question}],
            fallback=["deepseek-chat", "ernie-3.5-8k"],
            timeout=60.0
        )
        
        normal_time = time.time() - start_time
        content_length = len(response.choices[0].message.content) if response.choices else 0
        
        print(f"   ✅ 普通请求完成")
        print(f"   - 总耗时: {normal_time:.2f}秒")
        print(f"   - 内容长度: {content_length} 字符")
        print(f"   - 首字节时间: {normal_time:.2f}秒")
        
    except Exception as e:
        print(f"   ❌ 普通请求失败: {e}")
        return
    
    # 2. 流式请求
    print("\n🔄 流式请求测试...")
    start_time = time.time()
    first_chunk_time = None
    chunk_count = 0
    total_content = ""
    
    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': question}],
            stream=True,
            fallback=["deepseek-chat", "ernie-3.5-8k"],
            timeout=60.0
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                
                content = chunk.choices[0].delta.content
                total_content += content
                chunk_count += 1
                
                # 显示前几个字符
                if chunk_count <= 3:
                    print(f"   📦 片段 {chunk_count}: {content[:20]}...")
        
        stream_time = time.time() - start_time
        
        print(f"\n   ✅ 流式请求完成")
        print(f"   - 总耗时: {stream_time:.2f}秒")
        print(f"   - 首字节时间: {first_chunk_time:.2f}秒")
        print(f"   - 总片段数: {chunk_count}")
        print(f"   - 内容长度: {len(total_content)} 字符")
        
        # 性能对比
        print(f"\n📊 流式处理优势:")
        if first_chunk_time and normal_time > 0:
            ttfb_improvement = (normal_time - first_chunk_time) / normal_time
            print(f"   - 首字节时间提升: {ttfb_improvement:.1%}")
            print(f"   - 用户感知延迟降低: {normal_time - first_chunk_time:.2f}秒")
        
    except Exception as e:
        print(f"   ❌ 流式请求失败: {e}")

async def demo_batch_optimization():
    """演示批量优化"""
    print("\n📦 演示批量优化")
    print("=" * 50)
    
    # 准备批量请求
    questions = [
        "什么是区块链？",
        "解释智能合约",
        "DeFi的核心概念",
        "NFT技术原理",
        "Web3的发展前景"
    ]
    
    optimizer = PerformanceOptimizer(enable_cache=True, max_concurrent=3)
    optimizer.start_time = time.time()
    
    print(f"📝 批量处理 {len(questions)} 个请求...")
    
    # 并发批量处理
    async def process_batch_question(question: str, index: int):
        messages = [{'role': 'user', 'content': question}]
        try:
            start_time = time.time()
            response, from_cache = await optimizer.optimized_request(messages)
            process_time = time.time() - start_time
            
            cache_indicator = "💾" if from_cache else "🌐"
            print(f"   {cache_indicator} 问题 {index+1}: {question[:20]}... ({process_time:.2f}s)")
            return True, process_time
            
        except Exception as e:
            print(f"   ❌ 问题 {index+1}: {e}")
            return False, 0
    
    start_time = time.time()
    tasks = [
        process_batch_question(question, i) 
        for i, question in enumerate(questions)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # 统计结果
    successful = sum(1 for result in results if isinstance(result, tuple) and result[0])
    avg_time = statistics.mean([result[1] for result in results if isinstance(result, tuple) and result[0]]) if successful > 0 else 0
    
    stats = optimizer.get_performance_stats()
    
    print(f"\n📊 批量优化统计:")
    print(f"   - 成功请求: {successful}/{len(questions)}")
    print(f"   - 总耗时: {total_time:.2f}秒")
    print(f"   - 平均单请求时间: {avg_time:.2f}秒")
    print(f"   - 缓存命中率: {stats['cache_hit_rate']:.1%}")
    print(f"   - 实际API调用: {stats['total_requests']}")
    print(f"   - 缓存响应: {stats['cached_requests']}")
    print(f"   - 总吞吐量: {stats['requests_per_second']:.2f} 请求/秒")

async def demo_response_time_optimization():
    """演示响应时间优化"""
    print("\n⏱️ 演示响应时间优化")
    print("=" * 50)
    
    optimizer = PerformanceOptimizer(enable_cache=True, max_concurrent=5)
    optimizer.start_time = time.time()
    
    # 测试不同复杂度的请求
    test_cases = [
        ("简单问题", "什么是AI？"),
        ("中等问题", "解释机器学习的基本原理和应用场景"),
        ("复杂问题", "详细分析深度学习在计算机视觉领域的技术发展历程、当前挑战和未来发展方向"),
        ("简单问题", "什么是AI？"),  # 重复，测试缓存
    ]
    
    print("📝 测试不同复杂度请求的响应时间...")
    
    for i, (complexity, question) in enumerate(test_cases):
        messages = [{'role': 'user', 'content': question}]
        
        try:
            start_time = time.time()
            response, from_cache = await optimizer.optimized_request(messages)
            response_time = time.time() - start_time
            
            cache_indicator = "💾" if from_cache else "🌐"
            content_length = len(response.choices[0].message.content) if response.choices else 0
            
            print(f"   {cache_indicator} {complexity} {i+1}:")
            print(f"      - 响应时间: {response_time:.2f}秒")
            print(f"      - 内容长度: {content_length} 字符")
            print(f"      - 处理速度: {content_length/max(response_time, 0.01):.0f} 字符/秒")
            
        except Exception as e:
            print(f"   ❌ {complexity} {i+1}: {e}")
    
    # 显示整体统计
    stats = optimizer.get_performance_stats()
    
    print(f"\n📊 响应时间统计:")
    print(f"   - 平均响应时间: {stats['avg_response_time']:.2f}秒")
    print(f"   - 最快响应时间: {stats['min_response_time']:.2f}秒")
    print(f"   - 最慢响应时间: {stats['max_response_time']:.2f}秒")
    print(f"   - P95响应时间: {stats['p95_response_time']:.2f}秒")
    print(f"   - 缓存命中率: {stats['cache_hit_rate']:.1%}")

async def demo_resource_monitoring():
    """演示资源监控"""
    print("\n📊 演示资源监控")
    print("=" * 50)
    
    optimizer = PerformanceOptimizer(enable_cache=True, max_concurrent=3)
    optimizer.start_time = time.time()
    
    # 模拟一段时间的请求
    questions = [
        "什么是云原生？",
        "Kubernetes的核心概念",
        "Docker容器技术",
        "微服务架构设计",
        "什么是云原生？",  # 重复
        "服务网格的作用",
        "Kubernetes的核心概念",  # 重复
        "CI/CD流水线设计"
    ]
    
    print("📝 模拟持续请求，监控资源使用...")
    
    # 记录每个请求的详细信息
    request_details = []
    
    for i, question in enumerate(questions):
        messages = [{'role': 'user', 'content': question}]
        
        try:
            start_time = time.time()
            response, from_cache = await optimizer.optimized_request(messages)
            response_time = time.time() - start_time
            
            # 记录请求详情
            detail = {
                'index': i + 1,
                'question': question[:30],
                'response_time': response_time,
                'from_cache': from_cache,
                'tokens': response.usage.total_tokens if response.usage else 0,
                'timestamp': datetime.now()
            }
            request_details.append(detail)
            
            cache_indicator = "💾" if from_cache else "🌐"
            print(f"   {cache_indicator} 请求 {i+1}: {response_time:.2f}s")
            
            # 每隔几个请求显示统计
            if (i + 1) % 3 == 0:
                current_stats = optimizer.get_performance_stats()
                print(f"      📊 当前统计: 缓存命中率 {current_stats['cache_hit_rate']:.1%}, "
                      f"平均响应时间 {current_stats['avg_response_time']:.2f}s")
            
        except Exception as e:
            print(f"   ❌ 请求 {i+1}: {e}")
    
    # 最终统计报告
    final_stats = optimizer.get_performance_stats()
    
    print(f"\n📊 最终资源监控报告:")
    print(f"   性能指标:")
    print(f"     - 总请求数: {final_stats['total_requests'] + final_stats['cached_requests']}")
    print(f"     - 实际API调用: {final_stats['total_requests']}")
    print(f"     - 缓存响应: {final_stats['cached_requests']}")
    print(f"     - 缓存命中率: {final_stats['cache_hit_rate']:.1%}")
    print(f"     - 平均响应时间: {final_stats['avg_response_time']:.2f}秒")
    print(f"     - P95响应时间: {final_stats['p95_response_time']:.2f}秒")
    print(f"     - 总吞吐量: {final_stats['requests_per_second']:.2f} 请求/秒")
    
    print(f"   资源使用:")
    print(f"     - 总Token消耗: {final_stats['total_tokens']}")
    print(f"     - 平均Token/请求: {final_stats['total_tokens'] / max(final_stats['total_requests'], 1):.0f}")
    
    # 缓存效率分析
    if optimizer.cache:
        cache_stats = optimizer.cache.get_stats()
        print(f"   缓存效率:")
        print(f"     - 缓存条目数: {cache_stats['cache_size']}")
        print(f"     - 缓存利用率: {cache_stats['cache_size'] / cache_stats['max_size']:.1%}")
        print(f"     - 命中次数: {cache_stats['hit_count']}")
        print(f"     - 未命中次数: {cache_stats['miss_count']}")

async def main():
    """主函数"""
    print("🚀 HarborAI 性能优化演示")
    print("=" * 60)
    
    # 检查环境变量
    client, model = get_client()
    if not client:
        print("⚠️ 警告: 未设置任何 API Key")
        print("请设置 DEEPSEEK_API_KEY, ERNIE_API_KEY, 或 DOUBAO_API_KEY")
        return
    
    print(f"🔍 使用模型: {model}")
    
    demos = [
        ("缓存性能优化", demo_cache_performance),
        ("并发优化", demo_concurrent_optimization),
        ("流式处理优化", demo_streaming_optimization),
        ("批量优化", demo_batch_optimization),
        ("响应时间优化", demo_response_time_optimization),
        ("资源监控", demo_resource_monitoring)
    ]
    
    for name, demo_func in demos:
        try:
            await demo_func()
            await asyncio.sleep(1)  # 避免请求过于频繁
        except Exception as e:
            print(f"❌ {name} 演示失败: {e}")
    
    print("\n🎉 性能优化演示完成！")
    print("\n💡 关键优化策略:")
    print("1. 智能缓存 - 减少重复API调用，显著提升响应速度")
    print("2. 并发控制 - 合理的并发数，平衡速度和资源使用")
    print("3. 流式处理 - 降低首字节时间，改善用户体验")
    print("4. 批量优化 - 高效处理大量请求，提高吞吐量")
    print("5. 资源监控 - 实时监控性能指标，及时优化调整")
    print("6. 降级策略 - 内置fallback机制，确保服务可用性")

if __name__ == "__main__":
    asyncio.run(main())