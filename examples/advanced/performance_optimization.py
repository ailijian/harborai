#!/usr/bin/env python3
"""
性能优化演示

这个示例展示了 HarborAI 的性能优化技术，包括：
1. 智能缓存策略
2. 连接池管理
3. 请求预测与预加载
4. 性能基准测试
5. 资源监控与调优

场景：
- 高并发、大流量的生产环境
- 需要快速响应的实时应用
- 资源敏感的成本优化场景

价值：
- 显著提升响应速度和用户体验
- 减少资源消耗和运营成本
- 提高系统并发处理能力
- 优化API调用效率
"""

import asyncio
import time
import hashlib
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import OrderedDict, defaultdict

# 导入配置助手
from config_helper import get_model_configs, get_primary_model_config, print_available_models

# 导入 HarborAI
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from harborai import HarborAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """缓存策略"""
    LRU = "lru"          # 最近最少使用
    LFU = "lfu"          # 最少使用频率
    TTL = "ttl"          # 时间过期
    ADAPTIVE = "adaptive" # 自适应

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_at).total_seconds() > self.ttl
    
    def touch(self):
        """更新访问信息"""
        self.last_accessed = datetime.now()
        self.access_count += 1

class IntelligentCache:
    """智能缓存"""
    
    def __init__(self, 
                 max_size: int = 1000,
                 default_ttl: float = 3600,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        self._lock = threading.RLock()
    
    def _generate_key(self, messages: List[Dict], model: str, **kwargs) -> str:
        """生成缓存键"""
        # 创建一个包含所有相关参数的字符串
        cache_data = {
            "messages": messages,
            "model": model,
            **{k: v for k, v in kwargs.items() if k not in ["stream", "timeout"]}
        }
        
        # 使用JSON序列化并计算哈希
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, messages: List[Dict], model: str, **kwargs) -> Optional[Any]:
        """获取缓存"""
        key = self._generate_key(messages, model, **kwargs)
        
        with self._lock:
            self.stats["total_requests"] += 1
            
            if key in self.cache:
                entry = self.cache[key]
                
                # 检查是否过期
                if entry.is_expired():
                    del self.cache[key]
                    self.stats["misses"] += 1
                    return None
                
                # 更新访问信息
                entry.touch()
                
                # LRU策略：移动到末尾
                if self.strategy in [CacheStrategy.LRU, CacheStrategy.ADAPTIVE]:
                    self.cache.move_to_end(key)
                
                self.stats["hits"] += 1
                logger.debug(f"Cache hit for key: {key[:8]}...")
                return entry.value
            
            self.stats["misses"] += 1
            return None
    
    def put(self, messages: List[Dict], model: str, response: Any, **kwargs):
        """存储缓存"""
        key = self._generate_key(messages, model, **kwargs)
        
        with self._lock:
            # 如果缓存已满，执行淘汰策略
            if len(self.cache) >= self.max_size:
                self._evict()
            
            # 创建缓存条目
            entry = CacheEntry(
                key=key,
                value=response,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                ttl=self.default_ttl
            )
            
            self.cache[key] = entry
            logger.debug(f"Cached response for key: {key[:8]}...")
    
    def _evict(self):
        """淘汰缓存条目"""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # 删除最久未使用的
            self.cache.popitem(last=False)
        elif self.strategy == CacheStrategy.LFU:
            # 删除使用频率最低的
            min_key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            del self.cache[min_key]
        elif self.strategy == CacheStrategy.TTL:
            # 删除最早过期的
            now = datetime.now()
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                del self.cache[expired_keys[0]]
            else:
                self.cache.popitem(last=False)
        else:  # ADAPTIVE
            # 自适应策略：结合LRU和LFU
            now = datetime.now()
            
            # 首先删除过期的
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                del self.cache[expired_keys[0]]
            else:
                # 计算综合分数（时间 + 频率）
                scores = {}
                for key, entry in self.cache.items():
                    time_score = (now - entry.last_accessed).total_seconds()
                    freq_score = 1.0 / (entry.access_count + 1)
                    scores[key] = time_score + freq_score * 100
                
                # 删除分数最高的（最久未使用且频率最低）
                worst_key = max(scores.keys(), key=lambda k: scores[k])
                del self.cache[worst_key]
        
        self.stats["evictions"] += 1
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        if self.stats["total_requests"] == 0:
            return 0.0
        return self.stats["hits"] / self.stats["total_requests"]
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        return {
            **self.stats,
            "hit_rate": self.get_hit_rate(),
            "cache_size": len(self.cache),
            "max_size": self.max_size
        }
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self.cache.clear()
            self.stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_requests": 0
            }

class RequestPredictor:
    """请求预测器"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.request_history: List[Dict] = []
        self.pattern_cache: Dict[str, List[str]] = {}
        
    def record_request(self, messages: List[Dict], model: str):
        """记录请求"""
        request_info = {
            "timestamp": datetime.now(),
            "messages": messages,
            "model": model,
            "content_hash": self._hash_content(messages)
        }
        
        self.request_history.append(request_info)
        
        # 保持历史记录在合理范围内
        if len(self.request_history) > self.history_size:
            self.request_history = self.request_history[-self.history_size:]
    
    def _hash_content(self, messages: List[Dict]) -> str:
        """计算内容哈希"""
        content = " ".join([msg.get("content", "") for msg in messages])
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    def predict_next_requests(self, current_messages: List[Dict], limit: int = 5) -> List[Dict]:
        """预测下一个可能的请求"""
        if len(self.request_history) < 2:
            return []
        
        current_hash = self._hash_content(current_messages)
        
        # 查找相似的历史请求
        similar_requests = []
        for i, request in enumerate(self.request_history[:-1]):
            if request["content_hash"] == current_hash:
                # 找到相似请求，预测下一个请求
                next_request = self.request_history[i + 1]
                similar_requests.append(next_request)
        
        # 返回最常见的后续请求
        if similar_requests:
            # 按频率排序
            hash_counts = defaultdict(int)
            hash_to_request = {}
            
            for req in similar_requests:
                req_hash = req["content_hash"]
                hash_counts[req_hash] += 1
                hash_to_request[req_hash] = req
            
            # 返回最频繁的请求
            sorted_hashes = sorted(hash_counts.keys(), key=lambda h: hash_counts[h], reverse=True)
            return [hash_to_request[h] for h in sorted_hashes[:limit]]
        
        return []

class PerformanceOptimizedClient:
    """性能优化客户端"""
    
    def __init__(self,
                 model_name: Optional[str] = None,
                 cache_size: int = 1000,
                 cache_ttl: float = 3600,
                 enable_prediction: bool = True):
        
        # 基础客户端
        self.client = HarborAI()
        self.model_name = model_name or get_primary_model_config().model
        
        # 性能优化组件
        self.cache = IntelligentCache(max_size=cache_size, default_ttl=cache_ttl)
        self.predictor = RequestPredictor() if enable_prediction else None
        
        # 性能统计
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_response_time": 0.0,
            "prediction_hits": 0,
            "preloaded_requests": 0
        }
        
        # 预加载任务
        self.preload_tasks: Dict[str, asyncio.Task] = {}
    
    async def chat_completion(self, 
                            messages: List[Dict], 
                            model: Optional[str] = None,
                            use_cache: bool = True,
                            enable_preload: bool = True,
                            **kwargs) -> Any:
        """优化的聊天完成"""
        # 使用传入的模型或默认模型
        model_to_use = model or self.model_name
        
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # 1. 尝试从缓存获取
        if use_cache:
            cached_response = self.cache.get(messages, model_to_use, **kwargs)
            if cached_response:
                self.stats["cache_hits"] += 1
                response_time = time.time() - start_time
                self.stats["total_response_time"] += response_time
                
                logger.debug(f"Cache hit - Response time: {response_time:.3f}s")
                
                # 记录请求用于预测
                if self.predictor:
                    self.predictor.record_request(messages, model_to_use)
                
                # 触发预加载
                if enable_preload:
                    await self._trigger_preload(messages, model_to_use, **kwargs)
                
                return cached_response
        
        # 2. 发送实际请求
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model_to_use,
                messages=messages,
                timeout=90.0,  # 使用90秒超时
                **kwargs
            )
            
            response_time = time.time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["cache_misses"] += 1
            
            logger.debug(f"API call - Response time: {response_time:.3f}s")
            
            # 3. 存储到缓存
            if use_cache:
                self.cache.put(messages, model_to_use, response, **kwargs)
            
            # 4. 记录请求用于预测
            if self.predictor:
                self.predictor.record_request(messages, model_to_use)
            
            # 5. 触发预加载
            if enable_preload:
                await self._trigger_preload(messages, model_to_use, **kwargs)
            
            return response
            
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise e
    
    async def _trigger_preload(self, messages: List[Dict], model: str, **kwargs):
        """触发预加载"""
        if not self.predictor:
            return
        
        # 预测下一个可能的请求
        predicted_requests = self.predictor.predict_next_requests(messages, limit=3)
        
        for predicted in predicted_requests:
            pred_messages = predicted["messages"]
            pred_model = predicted.get("model", model)
            
            # 检查是否已经在缓存中
            if self.cache.get(pred_messages, pred_model, **kwargs):
                continue
            
            # 检查是否已经在预加载
            pred_key = self.cache._generate_key(pred_messages, pred_model, **kwargs)
            if pred_key in self.preload_tasks:
                continue
            
            # 启动预加载任务
            task = asyncio.create_task(self._preload_request(pred_messages, pred_model, pred_key, **kwargs))
            self.preload_tasks[pred_key] = task
    
    async def _preload_request(self, messages: List[Dict], model: str, key: str, **kwargs):
        """预加载请求"""
        try:
            logger.debug(f"Preloading request: {key[:8]}...")
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model,
                messages=messages,
                timeout=90.0,
                **kwargs
            )
            
            # 存储到缓存
            self.cache.put(messages, model, response, **kwargs)
            self.stats["preloaded_requests"] += 1
            
            logger.debug(f"Preload completed: {key[:8]}...")
            
        except Exception as e:
            logger.warning(f"Preload failed: {str(e)}")
        finally:
            # 清理任务
            if key in self.preload_tasks:
                del self.preload_tasks[key]
    
    def get_average_response_time(self) -> float:
        """获取平均响应时间"""
        if self.stats["total_requests"] == 0:
            return 0.0
        return self.stats["total_response_time"] / self.stats["total_requests"]
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        cache_stats = self.cache.get_stats()
        
        return {
            "requests": self.stats,
            "cache": cache_stats,
            "performance": {
                "average_response_time": self.get_average_response_time(),
                "cache_hit_rate": cache_stats["hit_rate"],
                "preload_efficiency": self.stats["preloaded_requests"] / max(self.stats["total_requests"], 1)
            }
        }

class PerformanceBenchmark:
    """性能基准测试"""
    
    def __init__(self):
        self.results: List[Dict] = []
    
    async def run_benchmark(self, 
                          client: PerformanceOptimizedClient,
                          test_messages: List[List[Dict]],
                          iterations: int = 10) -> Dict:
        """运行基准测试"""
        print(f"🔄 开始性能基准测试 ({iterations} 次迭代)")
        
        start_time = time.time()
        response_times = []
        
        for i in range(iterations):
            for j, messages in enumerate(test_messages):
                iter_start = time.time()
                
                try:
                    await client.chat_completion(messages)
                    iter_time = time.time() - iter_start
                    response_times.append(iter_time)
                    
                    print(f"   迭代 {i+1}/{iterations}, 消息 {j+1}/{len(test_messages)}: {iter_time:.3f}s")
                    
                except Exception as e:
                    print(f"   ❌ 迭代 {i+1}/{iterations}, 消息 {j+1} 失败: {str(e)}")
        
        total_time = time.time() - start_time
        
        # 计算统计数据
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            # 计算百分位数
            sorted_times = sorted(response_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p50 = p95 = p99 = 0
        
        # 获取客户端性能统计
        perf_stats = client.get_performance_stats()
        
        benchmark_result = {
            "test_config": {
                "iterations": iterations,
                "total_messages": len(test_messages),
                "total_requests": iterations * len(test_messages)
            },
            "timing": {
                "total_time": total_time,
                "average_response_time": avg_response_time,
                "min_response_time": min_response_time,
                "max_response_time": max_response_time,
                "p50_response_time": p50,
                "p95_response_time": p95,
                "p99_response_time": p99
            },
            "performance": perf_stats,
            "throughput": {
                "requests_per_second": len(response_times) / total_time if total_time > 0 else 0
            }
        }
        
        self.results.append(benchmark_result)
        return benchmark_result

# 演示函数
async def demo_intelligent_cache():
    """演示智能缓存"""
    print("\n🧠 智能缓存演示")
    print("=" * 50)
    
    # 创建优化客户端
    client = PerformanceOptimizedClient(cache_size=100, cache_ttl=300)
    
    # 测试消息
    test_messages = [
        [{"role": "user", "content": "什么是人工智能？"}],
        [{"role": "user", "content": "解释机器学习"}],
        [{"role": "user", "content": "什么是深度学习？"}],
        [{"role": "user", "content": "什么是人工智能？"}],  # 重复请求
    ]
    
    print("🔄 发送测试请求...")
    
    for i, messages in enumerate(test_messages):
        start_time = time.time()
        
        try:
            response = await client.chat_completion(messages)
            response_time = time.time() - start_time
            
            cache_stats = client.cache.get_stats()
            print(f"   请求 {i+1}: {response_time:.3f}s (缓存命中率: {cache_stats['hit_rate']:.1%})")
            
        except Exception as e:
            print(f"   ❌ 请求 {i+1} 失败: {str(e)}")
    
    # 显示缓存统计
    cache_stats = client.cache.get_stats()
    print(f"\n📊 缓存统计:")
    print(f"   总请求数: {cache_stats['total_requests']}")
    print(f"   缓存命中: {cache_stats['hits']}")
    print(f"   缓存未命中: {cache_stats['misses']}")
    print(f"   命中率: {cache_stats['hit_rate']:.1%}")
    print(f"   缓存大小: {cache_stats['cache_size']}/{cache_stats['max_size']}")

async def demo_request_prediction():
    """演示请求预测"""
    print("\n🔮 请求预测演示")
    print("=" * 50)
    
    client = PerformanceOptimizedClient(enable_prediction=True)
    
    # 模拟用户对话模式
    conversation_patterns = [
        [{"role": "user", "content": "你好"}],
        [{"role": "user", "content": "我想了解AI"}],
        [{"role": "user", "content": "谢谢"}],
        [{"role": "user", "content": "你好"}],  # 重复模式开始
        [{"role": "user", "content": "我想了解机器学习"}],  # 类似但不同的后续
    ]
    
    print("🔄 建立对话模式...")
    
    for i, messages in enumerate(conversation_patterns):
        try:
            start_time = time.time()
            response = await client.chat_completion(messages, enable_preload=True)
            response_time = time.time() - start_time
            
            print(f"   对话 {i+1}: {response_time:.3f}s")
            
            # 显示预测结果
            if client.predictor:
                predictions = client.predictor.predict_next_requests(messages, limit=2)
                if predictions:
                    print(f"     预测下一步: {len(predictions)} 个可能的请求")
            
        except Exception as e:
            print(f"   ❌ 对话 {i+1} 失败: {str(e)}")
        
        await asyncio.sleep(0.5)  # 模拟用户思考时间
    
    # 显示预加载统计
    perf_stats = client.get_performance_stats()
    print(f"\n📊 预测统计:")
    print(f"   预加载请求数: {client.stats['preloaded_requests']}")
    print(f"   预加载效率: {perf_stats['performance']['preload_efficiency']:.1%}")

async def demo_performance_comparison():
    """演示性能对比"""
    print("\n⚡ 性能对比演示")
    print("=" * 50)
    
    # 普通客户端
    normal_client = HarborAI()
    
    # 优化客户端
    optimized_client = PerformanceOptimizedClient(
        cache_size=50,
        cache_ttl=300,
        enable_prediction=True
    )
    
    # 测试消息
    test_messages = [
        [{"role": "user", "content": "简单测试1"}],
        [{"role": "user", "content": "简单测试2"}],
        [{"role": "user", "content": "简单测试1"}],  # 重复
        [{"role": "user", "content": "简单测试3"}],
    ]
    
    # 测试普通客户端
    print("🔄 测试普通客户端...")
    normal_times = []
    normal_start = time.time()
    
    for i, messages in enumerate(test_messages):
        try:
            start_time = time.time()
            await asyncio.to_thread(
                normal_client.chat.completions.create,
                model=optimized_client.model_name,
                messages=messages
            )
            response_time = time.time() - start_time
            normal_times.append(response_time)
            print(f"   请求 {i+1}: {response_time:.3f}s")
        except Exception as e:
            print(f"   ❌ 请求 {i+1} 失败: {str(e)}")
    
    normal_total = time.time() - normal_start
    
    # 测试优化客户端
    print("\n🔄 测试优化客户端...")
    optimized_times = []
    optimized_start = time.time()
    
    for i, messages in enumerate(test_messages):
        try:
            start_time = time.time()
            await optimized_client.chat_completion(messages)
            response_time = time.time() - start_time
            optimized_times.append(response_time)
            print(f"   请求 {i+1}: {response_time:.3f}s")
        except Exception as e:
            print(f"   ❌ 请求 {i+1} 失败: {str(e)}")
    
    optimized_total = time.time() - optimized_start
    
    # 性能对比
    if normal_times and optimized_times:
        avg_normal = sum(normal_times) / len(normal_times)
        avg_optimized = sum(optimized_times) / len(optimized_times)
        
        print(f"\n📊 性能对比:")
        print(f"   普通客户端:")
        print(f"     平均响应时间: {avg_normal:.3f}s")
        print(f"     总时间: {normal_total:.3f}s")
        
        print(f"   优化客户端:")
        print(f"     平均响应时间: {avg_optimized:.3f}s")
        print(f"     总时间: {optimized_total:.3f}s")
        
        if avg_normal > 0:
            improvement = ((avg_normal - avg_optimized) / avg_normal * 100)
            print(f"     性能提升: {improvement:.1f}%")
        
        # 显示优化统计
        perf_stats = optimized_client.get_performance_stats()
        print(f"     缓存命中率: {perf_stats['cache']['hit_rate']:.1%}")

async def demo_benchmark_test():
    """演示基准测试"""
    print("\n📊 基准测试演示")
    print("=" * 50)
    
    # 创建基准测试器
    benchmark = PerformanceBenchmark()
    
    # 创建优化客户端
    client = PerformanceOptimizedClient(
        cache_size=100,
        cache_ttl=600,
        enable_prediction=True
    )
    
    # 测试消息集
    test_messages = [
        [{"role": "user", "content": "测试消息1"}],
        [{"role": "user", "content": "测试消息2"}],
        [{"role": "user", "content": "测试消息3"}],
    ]
    
    # 运行基准测试
    result = await benchmark.run_benchmark(
        client=client,
        test_messages=test_messages,
        iterations=3  # 减少迭代次数以节省时间
    )
    
    # 显示结果
    print(f"\n📊 基准测试结果:")
    print(f"   总请求数: {result['test_config']['total_requests']}")
    print(f"   总时间: {result['timing']['total_time']:.3f}s")
    print(f"   平均响应时间: {result['timing']['average_response_time']:.3f}s")
    print(f"   P95响应时间: {result['timing']['p95_response_time']:.3f}s")
    print(f"   吞吐量: {result['throughput']['requests_per_second']:.1f} req/s")
    print(f"   缓存命中率: {result['performance']['cache']['hit_rate']:.1%}")

async def demo_cache_strategies():
    """演示不同缓存策略"""
    print("\n🎯 缓存策略对比演示")
    print("=" * 50)
    
    strategies = [
        ("LRU", CacheStrategy.LRU),
        ("LFU", CacheStrategy.LFU),
        ("自适应", CacheStrategy.ADAPTIVE)
    ]
    
    test_messages = [
        [{"role": "user", "content": f"测试消息{i}"}] for i in range(1, 6)
    ]
    
    for strategy_name, strategy in strategies:
        print(f"\n🔄 测试 {strategy_name} 策略:")
        
        # 创建使用特定策略的缓存
        cache = IntelligentCache(max_size=3, strategy=strategy)
        client = PerformanceOptimizedClient(cache_size=3)
        client.cache = cache
        
        # 发送请求以填充缓存
        for i, messages in enumerate(test_messages):
            try:
                await client.chat_completion(messages)
                stats = cache.get_stats()
                print(f"   请求 {i+1}: 缓存大小 {stats['cache_size']}, 命中率 {stats['hit_rate']:.1%}")
            except Exception as e:
                print(f"   ❌ 请求 {i+1} 失败: {str(e)}")
        
        # 显示最终统计
        final_stats = cache.get_stats()
        print(f"   最终命中率: {final_stats['hit_rate']:.1%}")
        print(f"   淘汰次数: {final_stats['evictions']}")

async def main():
    """主演示函数"""
    print("⚡ HarborAI 性能优化演示")
    print("=" * 60)
    
    # 显示可用模型配置
    print_available_models()
    
    try:
        # 智能缓存演示
        await demo_intelligent_cache()
        
        # 请求预测演示
        await demo_request_prediction()
        
        # 性能对比演示
        await demo_performance_comparison()
        
        # 基准测试演示
        await demo_benchmark_test()
        
        # 缓存策略对比演示
        await demo_cache_strategies()
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境建议:")
        print("   1. 根据业务特点调整缓存大小和TTL")
        print("   2. 监控缓存命中率并优化策略")
        print("   3. 使用请求预测减少延迟")
        print("   4. 定期进行性能基准测试")
        print("   5. 结合业务指标优化性能参数")
        print("   6. 使用90秒超时配置应对网络延迟")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())