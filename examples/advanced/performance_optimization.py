#!/usr/bin/env python3
"""
HarborAI 性能优化演示

场景描述:
在高并发、大流量的生产环境中，通过智能缓存、连接池管理、请求预测等
多种优化技术，显著提升系统响应速度和资源利用效率。

应用价值:
- 显著提升响应速度和用户体验
- 减少资源消耗和运营成本
- 提高系统并发处理能力
- 优化API调用效率

核心功能:
1. 智能缓存策略
2. 连接池管理
3. 请求预测与预加载
4. 性能基准测试
5. 资源监控与调优
"""

import asyncio
import time
import hashlib
import pickle
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from collections import OrderedDict, defaultdict
import psutil
import aiohttp
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

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
    
    def get(self, messages: List[Dict], model: str, **kwargs) -> Optional[ChatCompletion]:
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
    
    def put(self, messages: List[Dict], model: str, response: ChatCompletion, **kwargs):
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

class ConnectionPool:
    """连接池管理"""
    
    def __init__(self, 
                 max_connections: int = 20,
                 max_keepalive_connections: int = 10,
                 keepalive_expiry: float = 30.0):
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.keepalive_expiry = keepalive_expiry
        
        # 连接池统计
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "reused_connections": 0,
            "connection_errors": 0
        }
        
        # 创建连接器
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=max_connections,
            keepalive_timeout=keepalive_expiry,
            enable_cleanup_closed=True
        )
    
    async def get_session(self) -> aiohttp.ClientSession:
        """获取HTTP会话"""
        self.stats["active_connections"] += 1
        
        return aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    def get_stats(self) -> Dict:
        """获取连接池统计"""
        return {
            **self.stats,
            "max_connections": self.max_connections,
            "connector_stats": {
                "total_connections": len(self.connector._conns),
                "available_connections": sum(len(conns) for conns in self.connector._conns.values())
            }
        }
    
    async def close(self):
        """关闭连接池"""
        await self.connector.close()

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
                 api_key: str,
                 base_url: str = "https://api.harborai.com/v1",
                 cache_size: int = 1000,
                 cache_ttl: float = 3600,
                 max_connections: int = 20,
                 enable_prediction: bool = True):
        
        # 基础客户端
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        
        # 性能优化组件
        self.cache = IntelligentCache(max_size=cache_size, default_ttl=cache_ttl)
        self.connection_pool = ConnectionPool(max_connections=max_connections)
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
                            model: str = "deepseek-chat",
                            use_cache: bool = True,
                            enable_preload: bool = True,
                            **kwargs) -> ChatCompletion:
        """优化的聊天完成"""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # 1. 尝试从缓存获取
        if use_cache:
            cached_response = self.cache.get(messages, model, **kwargs)
            if cached_response:
                self.stats["cache_hits"] += 1
                response_time = time.time() - start_time
                self.stats["total_response_time"] += response_time
                
                logger.debug(f"Cache hit - Response time: {response_time:.3f}s")
                
                # 记录请求用于预测
                if self.predictor:
                    self.predictor.record_request(messages, model)
                
                # 触发预加载
                if enable_preload:
                    await self._trigger_preload(messages, model, **kwargs)
                
                return cached_response
        
        # 2. 发送实际请求
        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            
            response_time = time.time() - start_time
            self.stats["total_response_time"] += response_time
            self.stats["cache_misses"] += 1
            
            logger.debug(f"API call - Response time: {response_time:.3f}s")
            
            # 3. 存储到缓存
            if use_cache:
                self.cache.put(messages, model, response, **kwargs)
            
            # 4. 记录请求用于预测
            if self.predictor:
                self.predictor.record_request(messages, model)
            
            # 5. 触发预加载
            if enable_preload:
                await self._trigger_preload(messages, model, **kwargs)
            
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
            
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
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
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        avg_response_time = 0.0
        if self.stats["total_requests"] > 0:
            avg_response_time = self.stats["total_response_time"] / self.stats["total_requests"]
        
        return {
            "requests": self.stats,
            "cache": self.cache.get_stats(),
            "connection_pool": self.connection_pool.get_stats(),
            "average_response_time": avg_response_time,
            "active_preload_tasks": len(self.preload_tasks)
        }
    
    async def warmup_cache(self, warmup_requests: List[Tuple[List[Dict], str]]):
        """预热缓存"""
        logger.info(f"Warming up cache with {len(warmup_requests)} requests...")
        
        tasks = []
        for messages, model in warmup_requests:
            task = asyncio.create_task(self.chat_completion(messages, model, enable_preload=False))
            tasks.append(task)
        
        # 并发执行预热请求
        await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Cache warmup completed")
    
    async def close(self):
        """关闭客户端"""
        # 取消所有预加载任务
        for task in self.preload_tasks.values():
            task.cancel()
        
        # 等待任务完成
        if self.preload_tasks:
            await asyncio.gather(*self.preload_tasks.values(), return_exceptions=True)
        
        # 关闭连接池
        await self.connection_pool.close()

# 演示函数
async def demo_cache_performance():
    """演示缓存性能"""
    print("\n🚀 缓存性能演示")
    print("=" * 50)
    
    # 创建优化客户端
    client = PerformanceOptimizedClient(
        api_key="your-api-key-here",
        cache_size=100,
        cache_ttl=300  # 5分钟
    )
    
    # 测试请求
    test_requests = [
        [{"role": "user", "content": "什么是人工智能？"}],
        [{"role": "user", "content": "解释机器学习"}],
        [{"role": "user", "content": "什么是深度学习？"}],
        [{"role": "user", "content": "什么是人工智能？"}],  # 重复请求
        [{"role": "user", "content": "解释机器学习"}],     # 重复请求
    ]
    
    print("🔄 执行测试请求...")
    
    for i, messages in enumerate(test_requests):
        try:
            start_time = time.time()
            response = await client.chat_completion(messages)
            end_time = time.time()
            
            print(f"✅ 请求 {i+1}: {end_time - start_time:.3f}s")
            
        except Exception as e:
            print(f"❌ 请求 {i+1} 失败: {str(e)}")
    
    # 显示性能统计
    stats = client.get_performance_stats()
    print(f"\n📊 缓存性能统计:")
    print(f"   - 总请求数: {stats['requests']['total_requests']}")
    print(f"   - 缓存命中: {stats['requests']['cache_hits']}")
    print(f"   - 缓存未命中: {stats['requests']['cache_misses']}")
    print(f"   - 缓存命中率: {stats['cache']['hit_rate']:.1%}")
    print(f"   - 平均响应时间: {stats['average_response_time']:.3f}s")
    
    await client.close()

async def demo_connection_pool():
    """演示连接池优化"""
    print("\n🔗 连接池优化演示")
    print("=" * 50)
    
    # 创建不同配置的客户端进行对比
    
    # 1. 无连接池优化的客户端
    normal_client = AsyncOpenAI(api_key="your-api-key-here", base_url="https://api.harborai.com/v1")
    
    # 2. 连接池优化的客户端
    optimized_client = PerformanceOptimizedClient(
        api_key="your-api-key-here",
        max_connections=10,
        cache_size=0  # 禁用缓存以测试纯连接池性能
    )
    
    test_messages = [{"role": "user", "content": f"测试请求 {i}"} for i in range(5)]
    
    # 测试普通客户端
    print("🔄 测试普通客户端...")
    normal_start = time.time()
    normal_tasks = []
    
    for messages in test_messages:
        task = asyncio.create_task(normal_client.chat.completions.create(
            model="deepseek-chat",
            messages=[messages],
            max_tokens=50
        ))
        normal_tasks.append(task)
    
    try:
        await asyncio.gather(*normal_tasks)
        normal_time = time.time() - normal_start
        print(f"✅ 普通客户端完成时间: {normal_time:.3f}s")
    except Exception as e:
        print(f"❌ 普通客户端测试失败: {str(e)}")
        normal_time = float('inf')
    
    # 测试优化客户端
    print("🔄 测试连接池优化客户端...")
    optimized_start = time.time()
    optimized_tasks = []
    
    for messages in test_messages:
        task = asyncio.create_task(optimized_client.chat_completion(
            [messages],
            model="deepseek-chat",
            max_tokens=50,
            use_cache=False
        ))
        optimized_tasks.append(task)
    
    try:
        await asyncio.gather(*optimized_tasks)
        optimized_time = time.time() - optimized_start
        print(f"✅ 优化客户端完成时间: {optimized_time:.3f}s")
        
        # 显示连接池统计
        stats = optimized_client.get_performance_stats()
        print(f"📊 连接池统计:")
        print(f"   - 最大连接数: {stats['connection_pool']['max_connections']}")
        print(f"   - 活跃连接数: {stats['connection_pool']['active_connections']}")
        
        if normal_time != float('inf') and optimized_time > 0:
            improvement = (normal_time - optimized_time) / normal_time * 100
            print(f"   - 性能提升: {improvement:.1f}%")
        
    except Exception as e:
        print(f"❌ 优化客户端测试失败: {str(e)}")
    
    await optimized_client.close()

async def demo_request_prediction():
    """演示请求预测"""
    print("\n🧠 请求预测演示")
    print("=" * 50)
    
    client = PerformanceOptimizedClient(
        api_key="your-api-key-here",
        enable_prediction=True
    )
    
    # 模拟用户对话模式
    conversation_patterns = [
        # 模式1：AI基础问题序列
        [
            [{"role": "user", "content": "什么是人工智能？"}],
            [{"role": "user", "content": "AI有哪些应用领域？"}],
            [{"role": "user", "content": "AI的发展前景如何？"}]
        ],
        # 模式2：技术深入序列
        [
            [{"role": "user", "content": "什么是机器学习？"}],
            [{"role": "user", "content": "监督学习和无监督学习的区别？"}],
            [{"role": "user", "content": "如何选择机器学习算法？"}]
        ]
    ]
    
    # 训练预测模型
    print("🔄 训练预测模型...")
    for pattern in conversation_patterns:
        for messages in pattern:
            try:
                await client.chat_completion(messages, enable_preload=False)
                await asyncio.sleep(0.1)  # 模拟用户思考时间
            except Exception as e:
                print(f"❌ 训练请求失败: {str(e)}")
    
    # 测试预测效果
    print("\n🔄 测试预测效果...")
    
    # 发送第一个请求，应该触发预测和预加载
    test_messages = [{"role": "user", "content": "什么是人工智能？"}]
    
    start_time = time.time()
    response1 = await client.chat_completion(test_messages)
    time1 = time.time() - start_time
    print(f"✅ 第一个请求: {time1:.3f}s")
    
    # 等待预加载完成
    await asyncio.sleep(2)
    
    # 发送预测的下一个请求
    predicted_messages = [{"role": "user", "content": "AI有哪些应用领域？"}]
    
    start_time = time.time()
    response2 = await client.chat_completion(predicted_messages)
    time2 = time.time() - start_time
    print(f"✅ 预测请求: {time2:.3f}s")
    
    # 显示预测统计
    stats = client.get_performance_stats()
    print(f"\n📊 预测统计:")
    print(f"   - 预加载请求数: {stats['requests']['preloaded_requests']}")
    print(f"   - 活跃预加载任务: {stats['active_preload_tasks']}")
    print(f"   - 缓存命中率: {stats['cache']['hit_rate']:.1%}")
    
    if time2 < time1:
        speedup = (time1 - time2) / time1 * 100
        print(f"   - 预测加速: {speedup:.1f}%")
    
    await client.close()

async def demo_comprehensive_optimization():
    """演示综合优化效果"""
    print("\n⚡ 综合优化效果演示")
    print("=" * 50)
    
    # 创建不同配置的客户端
    clients = {
        "基础客户端": AsyncOpenAI(api_key="your-api-key-here", base_url="https://api.harborai.com/v1"),
        "缓存优化": PerformanceOptimizedClient(
            api_key="your-api-key-here",
            cache_size=100,
            max_connections=5,
            enable_prediction=False
        ),
        "全面优化": PerformanceOptimizedClient(
            api_key="your-api-key-here",
            cache_size=100,
            max_connections=10,
            enable_prediction=True
        )
    }
    
    # 测试场景：重复请求和相关请求
    test_scenarios = [
        [{"role": "user", "content": "什么是人工智能？"}],
        [{"role": "user", "content": "什么是机器学习？"}],
        [{"role": "user", "content": "什么是人工智能？"}],  # 重复
        [{"role": "user", "content": "AI的应用有哪些？"}],
        [{"role": "user", "content": "什么是机器学习？"}],  # 重复
    ]
    
    results = {}
    
    for client_name, client in clients.items():
        print(f"\n🔄 测试 {client_name}...")
        
        start_time = time.time()
        successful_requests = 0
        
        for i, messages in enumerate(test_scenarios):
            try:
                if client_name == "基础客户端":
                    await client.chat.completions.create(
                        model="deepseek-chat",
                        messages=messages,
                        max_tokens=100
                    )
                else:
                    await client.chat_completion(
                        messages,
                        model="deepseek-chat",
                        max_tokens=100
                    )
                successful_requests += 1
                
            except Exception as e:
                print(f"❌ 请求 {i+1} 失败: {str(e)}")
        
        total_time = time.time() - start_time
        
        results[client_name] = {
            "total_time": total_time,
            "successful_requests": successful_requests,
            "avg_time_per_request": total_time / successful_requests if successful_requests > 0 else 0
        }
        
        print(f"✅ {client_name} 完成:")
        print(f"   - 总时间: {total_time:.3f}s")
        print(f"   - 成功请求: {successful_requests}")
        print(f"   - 平均时间: {results[client_name]['avg_time_per_request']:.3f}s/req")
        
        # 显示优化客户端的详细统计
        if hasattr(client, 'get_performance_stats'):
            stats = client.get_performance_stats()
            print(f"   - 缓存命中率: {stats['cache']['hit_rate']:.1%}")
            if stats['requests']['preloaded_requests'] > 0:
                print(f"   - 预加载请求: {stats['requests']['preloaded_requests']}")
    
    # 性能对比
    print(f"\n📊 性能对比:")
    baseline = results["基础客户端"]["avg_time_per_request"]
    
    for client_name, result in results.items():
        if client_name != "基础客户端" and baseline > 0:
            improvement = (baseline - result["avg_time_per_request"]) / baseline * 100
            print(f"   - {client_name}: {improvement:.1f}% 提升")
    
    # 关闭优化客户端
    for client_name, client in clients.items():
        if hasattr(client, 'close'):
            await client.close()

async def demo_memory_optimization():
    """演示内存优化"""
    print("\n🧠 内存优化演示")
    print("=" * 50)
    
    # 监控内存使用
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    initial_memory = get_memory_usage()
    print(f"📊 初始内存使用: {initial_memory:.1f}MB")
    
    # 创建大缓存客户端
    client = PerformanceOptimizedClient(
        api_key="your-api-key-here",
        cache_size=500,  # 较大的缓存
        cache_ttl=3600
    )
    
    # 生成大量不同的请求
    print("🔄 生成大量缓存数据...")
    for i in range(50):
        messages = [{"role": "user", "content": f"测试问题 {i}: 请解释概念{i}"}]
        try:
            await client.chat_completion(messages, max_tokens=50)
        except Exception as e:
            print(f"❌ 请求 {i} 失败: {str(e)}")
        
        # 每10个请求检查一次内存
        if i % 10 == 0:
            current_memory = get_memory_usage()
            print(f"   请求 {i}: 内存使用 {current_memory:.1f}MB (+{current_memory - initial_memory:.1f}MB)")
    
    # 最终内存统计
    final_memory = get_memory_usage()
    cache_stats = client.get_performance_stats()["cache"]
    
    print(f"\n📊 内存优化统计:")
    print(f"   - 初始内存: {initial_memory:.1f}MB")
    print(f"   - 最终内存: {final_memory:.1f}MB")
    print(f"   - 内存增长: {final_memory - initial_memory:.1f}MB")
    print(f"   - 缓存大小: {cache_stats['cache_size']}")
    print(f"   - 缓存命中率: {cache_stats['hit_rate']:.1%}")
    print(f"   - 缓存淘汰次数: {cache_stats['evictions']}")
    
    # 清理缓存并检查内存释放
    client.cache.clear()
    await asyncio.sleep(1)  # 等待垃圾回收
    
    cleared_memory = get_memory_usage()
    print(f"   - 清理后内存: {cleared_memory:.1f}MB")
    print(f"   - 释放内存: {final_memory - cleared_memory:.1f}MB")
    
    await client.close()

async def main():
    """主演示函数"""
    print("⚡ HarborAI 性能优化演示")
    print("=" * 60)
    
    try:
        # 缓存性能演示
        await demo_cache_performance()
        
        # 连接池优化演示
        await demo_connection_pool()
        
        # 请求预测演示
        await demo_request_prediction()
        
        # 综合优化效果演示
        await demo_comprehensive_optimization()
        
        # 内存优化演示
        await demo_memory_optimization()
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境建议:")
        print("   1. 根据业务特点调整缓存策略和大小")
        print("   2. 监控缓存命中率和内存使用情况")
        print("   3. 合理配置连接池参数")
        print("   4. 利用请求模式进行智能预加载")
        print("   5. 定期清理过期缓存和监控性能指标")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())