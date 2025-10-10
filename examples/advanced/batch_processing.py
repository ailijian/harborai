#!/usr/bin/env python3
"""
HarborAI 批量处理优化演示

场景描述:
在需要处理大量AI请求的场景中，如批量文档分析、大规模数据处理等，
通过智能批量聚合、并发控制等技术，显著提升处理效率和资源利用率。

应用价值:
- 大幅提升处理效率和吞吐量
- 降低API调用成本
- 优化资源利用率
- 支持大规模数据处理场景

核心功能:
1. 智能批量聚合
2. 并发控制与限流
3. 内存管理优化
4. 进度追踪与监控
5. 结果分发机制
"""

import asyncio
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from harborai import HarborAI
from harborai.types.chat import ChatCompletion

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchStatus(Enum):
    """批次状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ProcessingMode(Enum):
    """处理模式"""
    SEQUENTIAL = "sequential"    # 顺序处理
    CONCURRENT = "concurrent"    # 并发处理
    BATCH = "batch"             # 批量处理

@dataclass
class BatchConfig:
    """批量处理配置"""
    batch_size: int = 10
    max_concurrent_batches: int = 5
    max_wait_time: float = 5.0
    memory_limit_mb: int = 1024
    retry_attempts: int = 3
    timeout_per_request: float = 30.0

@dataclass
class RequestItem:
    """请求项"""
    id: str
    messages: List[Dict]
    model: str = "deepseek-chat"
    kwargs: Dict = field(default_factory=dict)
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    
@dataclass
class BatchResult:
    """批次结果"""
    batch_id: str
    status: BatchStatus
    items: List[RequestItem]
    results: List[Optional[ChatCompletion]] = field(default_factory=list)
    errors: List[Optional[Exception]] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_cost: float = 0.0
    
    def get_processing_time(self) -> float:
        """获取处理时间"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if not self.results:
            return 0.0
        successful = sum(1 for result in self.results if result is not None)
        return successful / len(self.results)

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, limit_mb: int = 1024):
        self.limit_mb = limit_mb
        self.limit_bytes = limit_mb * 1024 * 1024
        
    def get_memory_usage(self) -> Dict:
        """获取内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    
    def is_memory_available(self, estimated_usage_mb: float = 0) -> bool:
        """检查内存是否可用"""
        current_usage = self.get_memory_usage()
        projected_usage = current_usage["rss_mb"] + estimated_usage_mb
        return projected_usage < self.limit_mb
    
    def wait_for_memory(self, required_mb: float = 100, timeout: float = 60.0):
        """等待内存可用"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_memory_available(required_mb):
                return True
            
            logger.info(f"Waiting for memory... Current usage: {self.get_memory_usage()['rss_mb']:.1f}MB")
            time.sleep(1.0)
        
        return False

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: str = "https://api.deepseek.com/v1",
                 config: Optional[BatchConfig] = None):
        self.client = HarborAI(api_key=api_key, base_url=base_url)
        self.config = config or BatchConfig()
        self.memory_monitor = MemoryMonitor(self.config.memory_limit_mb)
        
        # 批次管理
        self.pending_requests: List[RequestItem] = []
        self.active_batches: Dict[str, BatchResult] = {}
        self.completed_batches: List[BatchResult] = []
        
        # 统计信息
        self.total_processed = 0
        self.total_cost = 0.0
        self.start_time = datetime.now()
        
        # 并发控制
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)
        self.batch_counter = 0
        
    def add_request(self, messages: List[Dict], model: str = "deepseek-chat", priority: int = 0, **kwargs) -> str:
        """添加请求到队列"""
        request_id = f"req_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        request_item = RequestItem(
            id=request_id,
            messages=messages,
            model=model,
            priority=priority,
            kwargs=kwargs
        )
        
        self.pending_requests.append(request_item)
        logger.info(f"Added request {request_id} to queue (queue size: {len(self.pending_requests)})")
        
        return request_id
    
    def _create_batch(self) -> Optional[BatchResult]:
        """创建批次"""
        if not self.pending_requests:
            return None
        
        # 按优先级排序
        self.pending_requests.sort(key=lambda x: x.priority, reverse=True)
        
        # 取出一批请求
        batch_size = min(self.config.batch_size, len(self.pending_requests))
        batch_items = self.pending_requests[:batch_size]
        self.pending_requests = self.pending_requests[batch_size:]
        
        # 创建批次
        self.batch_counter += 1
        batch_id = f"batch_{self.batch_counter}_{int(time.time())}"
        
        batch = BatchResult(
            batch_id=batch_id,
            status=BatchStatus.PENDING,
            items=batch_items,
            results=[None] * len(batch_items),
            errors=[None] * len(batch_items)
        )
        
        logger.info(f"Created batch {batch_id} with {len(batch_items)} items")
        return batch
    
    async def _process_batch(self, batch: BatchResult) -> BatchResult:
        """处理单个批次"""
        async with self.semaphore:
            batch.status = BatchStatus.PROCESSING
            batch.start_time = datetime.now()
            
            logger.info(f"Processing batch {batch.batch_id}...")
            
            # 检查内存
            if not self.memory_monitor.is_memory_available(100):  # 预估100MB
                logger.warning("Memory limit reached, waiting...")
                if not self.memory_monitor.wait_for_memory(100, 30.0):
                    batch.status = BatchStatus.FAILED
                    return batch
            
            # 并发处理批次中的所有请求
            tasks = []
            for i, item in enumerate(batch.items):
                task = self._process_single_request(item, i, batch.batch_id)
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            total_cost = 0.0
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    batch.errors[i] = result
                    logger.error(f"Request {batch.items[i].id} failed: {str(result)}")
                else:
                    batch.results[i] = result
                    # 估算成本
                    if result and hasattr(result, 'usage') and result.usage:
                        total_cost += result.usage.total_tokens * 0.0001  # 简化成本计算
            
            batch.total_cost = total_cost
            batch.end_time = datetime.now()
            batch.status = BatchStatus.COMPLETED
            
            # 更新统计
            self.total_processed += len(batch.items)
            self.total_cost += total_cost
            
            logger.info(f"Batch {batch.batch_id} completed in {batch.get_processing_time():.2f}s")
            logger.info(f"Success rate: {batch.get_success_rate():.1%}, Cost: ${batch.total_cost:.6f}")
            
            return batch
    
    async def _process_single_request(self, item: RequestItem, index: int, batch_id: str) -> ChatCompletion:
        """处理单个请求"""
        for attempt in range(self.config.retry_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=item.model,
                    messages=item.messages,
                    timeout=self.config.timeout_per_request,
                    **item.kwargs
                )
                
                logger.debug(f"Request {item.id} in batch {batch_id} completed (attempt {attempt + 1})")
                return response
                
            except Exception as e:
                logger.warning(f"Request {item.id} attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.config.retry_attempts - 1:
                    raise e
                await asyncio.sleep(2 ** attempt)  # 指数退避
    
    async def process_all(self, progress_callback: Optional[Callable] = None) -> List[BatchResult]:
        """处理所有待处理请求"""
        logger.info(f"Starting batch processing of {len(self.pending_requests)} requests...")
        
        all_batches = []
        
        while self.pending_requests or self.active_batches:
            # 创建新批次
            while len(self.active_batches) < self.config.max_concurrent_batches and self.pending_requests:
                batch = self._create_batch()
                if batch:
                    self.active_batches[batch.batch_id] = batch
                    
                    # 启动批次处理任务
                    task = asyncio.create_task(self._process_batch(batch))
                    
                    # 添加完成回调
                    def batch_completed(batch_id):
                        def callback(task):
                            completed_batch = task.result()
                            if batch_id in self.active_batches:
                                del self.active_batches[batch_id]
                            self.completed_batches.append(completed_batch)
                            all_batches.append(completed_batch)
                            
                            if progress_callback:
                                progress_callback(completed_batch)
                        return callback
                    
                    task.add_done_callback(batch_completed(batch.batch_id))
            
            # 等待一段时间
            await asyncio.sleep(0.1)
        
        # 等待所有活跃批次完成
        while self.active_batches:
            await asyncio.sleep(0.1)
        
        logger.info(f"All batches completed. Total processed: {self.total_processed}")
        return all_batches
    
    def get_statistics(self) -> Dict:
        """获取处理统计"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "total_processed": self.total_processed,
            "total_cost": self.total_cost,
            "total_time": total_time,
            "throughput": self.total_processed / total_time if total_time > 0 else 0,
            "average_cost_per_request": self.total_cost / self.total_processed if self.total_processed > 0 else 0,
            "completed_batches": len(self.completed_batches),
            "active_batches": len(self.active_batches),
            "pending_requests": len(self.pending_requests),
            "memory_usage": self.memory_monitor.get_memory_usage()
        }

# 演示函数
async def demo_basic_batch_processing():
    """演示基础批量处理"""
    print("\n📦 基础批量处理演示")
    print("=" * 50)
    
    # 创建批量处理器
    processor = BatchProcessor(
        api_key="your-deepseek-key",
        config=BatchConfig(
            batch_size=5,
            max_concurrent_batches=2,
            max_wait_time=3.0
        )
    )
    
    # 添加测试请求
    test_questions = [
        "什么是人工智能？",
        "解释机器学习的基本概念",
        "深度学习有哪些应用？",
        "什么是自然语言处理？",
        "计算机视觉的主要技术有哪些？",
        "强化学习的原理是什么？",
        "神经网络是如何工作的？",
        "什么是大语言模型？",
        "AI在医疗领域有哪些应用？",
        "自动驾驶技术的核心是什么？"
    ]
    
    # 添加请求到处理器
    request_ids = []
    for i, question in enumerate(test_questions):
        messages = [{"role": "user", "content": question}]
        request_id = processor.add_request(messages, priority=random.randint(1, 5))
        request_ids.append(request_id)
    
    print(f"✅ 已添加 {len(request_ids)} 个请求到处理队列")
    
    # 进度回调函数
    def progress_callback(batch: BatchResult):
        print(f"🔄 批次 {batch.batch_id} 完成:")
        print(f"   - 处理时间: {batch.get_processing_time():.2f}s")
        print(f"   - 成功率: {batch.get_success_rate():.1%}")
        print(f"   - 成本: ${batch.total_cost:.6f}")
    
    # 开始处理
    start_time = time.time()
    batches = await processor.process_all(progress_callback)
    end_time = time.time()
    
    # 显示统计信息
    stats = processor.get_statistics()
    print(f"\n📊 处理统计:")
    print(f"   - 总处理数量: {stats['total_processed']}")
    print(f"   - 总处理时间: {end_time - start_time:.2f}s")
    print(f"   - 吞吐量: {stats['throughput']:.2f} req/s")
    print(f"   - 总成本: ${stats['total_cost']:.6f}")
    print(f"   - 平均成本: ${stats['average_cost_per_request']:.6f}")
    print(f"   - 内存使用: {stats['memory_usage']['rss_mb']:.1f}MB")

async def demo_performance_comparison():
    """演示性能对比"""
    print("\n⚡ 性能对比演示")
    print("=" * 50)
    
    # 准备测试数据
    test_questions = [
        f"请解释概念{i}: 这是一个测试问题，用于性能对比。" 
        for i in range(20)
    ]
    
    # 1. 顺序处理
    print("🔄 顺序处理测试...")
    client = HarborAI(api_key="your-deepseek-key", base_url="https://api.deepseek.com/v1")
    
    sequential_start = time.time()
    sequential_results = []
    
    for question in test_questions[:5]:  # 只测试5个请求
        try:
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": question}],
                max_tokens=100
            )
            sequential_results.append(response)
        except Exception as e:
            print(f"❌ 顺序处理失败: {str(e)}")
            sequential_results.append(None)
    
    sequential_time = time.time() - sequential_start
    sequential_success = sum(1 for r in sequential_results if r is not None)
    
    # 2. 批量处理
    print("🔄 批量处理测试...")
    processor = BatchProcessor(
        api_key="your-deepseek-key",
        config=BatchConfig(
            batch_size=5,
            max_concurrent_batches=3
        )
    )
    
    # 添加相同的请求
    for question in test_questions[:5]:
        messages = [{"role": "user", "content": question}]
        processor.add_request(messages, max_tokens=100)
    
    batch_start = time.time()
    batches = await processor.process_all()
    batch_time = time.time() - batch_start
    
    batch_success = sum(batch.get_success_rate() * len(batch.items) for batch in batches)
    
    # 性能对比
    print(f"\n📊 性能对比结果:")
    print(f"   顺序处理:")
    print(f"     - 处理时间: {sequential_time:.2f}s")
    print(f"     - 成功数量: {sequential_success}")
    print(f"     - 吞吐量: {sequential_success/sequential_time:.2f} req/s")
    
    print(f"   批量处理:")
    print(f"     - 处理时间: {batch_time:.2f}s")
    print(f"     - 成功数量: {int(batch_success)}")
    print(f"     - 吞吐量: {batch_success/batch_time:.2f} req/s")
    
    if sequential_time > 0 and batch_time > 0:
        speedup = sequential_time / batch_time
        print(f"   性能提升: {speedup:.2f}x")

async def demo_memory_management():
    """演示内存管理"""
    print("\n🧠 内存管理演示")
    print("=" * 50)
    
    # 创建内存监控器
    memory_monitor = MemoryMonitor(limit_mb=512)  # 设置较低的限制用于演示
    
    # 显示初始内存状态
    initial_memory = memory_monitor.get_memory_usage()
    print(f"📊 初始内存状态:")
    print(f"   - RSS: {initial_memory['rss_mb']:.1f}MB")
    print(f"   - 可用内存: {initial_memory['available_mb']:.1f}MB")
    print(f"   - 内存限制: {memory_monitor.limit_mb}MB")
    
    # 创建批量处理器（较小的批次大小）
    processor = BatchProcessor(
        api_key="your-deepseek-key",
        config=BatchConfig(
            batch_size=3,
            max_concurrent_batches=2,
            memory_limit_mb=512
        )
    )
    
    # 添加一些请求
    for i in range(10):
        messages = [{"role": "user", "content": f"测试请求 {i+1}"}]
        processor.add_request(messages)
    
    # 监控内存使用情况
    def memory_progress_callback(batch: BatchResult):
        current_memory = memory_monitor.get_memory_usage()
        print(f"🔄 批次 {batch.batch_id} 完成 - 内存使用: {current_memory['rss_mb']:.1f}MB")
    
    # 处理请求
    await processor.process_all(memory_progress_callback)
    
    # 显示最终内存状态
    final_memory = memory_monitor.get_memory_usage()
    print(f"\n📊 最终内存状态:")
    print(f"   - RSS: {final_memory['rss_mb']:.1f}MB")
    print(f"   - 内存增长: {final_memory['rss_mb'] - initial_memory['rss_mb']:.1f}MB")

async def demo_priority_processing():
    """演示优先级处理"""
    print("\n🎯 优先级处理演示")
    print("=" * 50)
    
    processor = BatchProcessor(
        api_key="your-deepseek-key",
        config=BatchConfig(batch_size=3, max_concurrent_batches=1)
    )
    
    # 添加不同优先级的请求
    requests_data = [
        ("紧急问题", "这是一个紧急问题，需要立即处理", 10),
        ("普通问题1", "这是一个普通问题", 5),
        ("低优先级问题", "这是一个低优先级问题", 1),
        ("高优先级问题", "这是一个高优先级问题", 8),
        ("普通问题2", "这是另一个普通问题", 5),
        ("最高优先级", "这是最高优先级问题", 15)
    ]
    
    # 随机顺序添加请求
    import random
    random.shuffle(requests_data)
    
    for name, content, priority in requests_data:
        messages = [{"role": "user", "content": content}]
        request_id = processor.add_request(messages, priority=priority)
        print(f"📝 添加请求: {name} (优先级: {priority})")
    
    # 处理请求并观察处理顺序
    def priority_progress_callback(batch: BatchResult):
        print(f"\n🔄 处理批次 {batch.batch_id}:")
        for item in batch.items:
            print(f"   - 请求内容: {item.messages[0]['content'][:20]}... (优先级: {item.priority})")
    
    await processor.process_all(priority_progress_callback)

async def demo_large_scale_processing():
    """演示大规模处理"""
    print("\n🚀 大规模处理演示")
    print("=" * 50)
    
    # 创建大规模处理器
    processor = BatchProcessor(
        api_key="your-deepseek-key",
        config=BatchConfig(
            batch_size=10,
            max_concurrent_batches=5,
            memory_limit_mb=1024
        )
    )
    
    # 生成大量测试数据
    print("📝 生成测试数据...")
    categories = ["科技", "医疗", "教育", "金融", "环境"]
    
    for i in range(50):  # 生成50个请求
        category = random.choice(categories)
        content = f"请分析{category}领域的发展趋势和挑战 - 问题{i+1}"
        messages = [{"role": "user", "content": content}]
        processor.add_request(messages, priority=random.randint(1, 10))
    
    print(f"✅ 已生成 50 个测试请求")
    
    # 处理统计
    processed_count = 0
    total_cost = 0.0
    
    def large_scale_progress_callback(batch: BatchResult):
        nonlocal processed_count, total_cost
        processed_count += len(batch.items)
        total_cost += batch.total_cost
        
        progress = processed_count / 50 * 100
        print(f"🔄 进度: {progress:.1f}% ({processed_count}/50) - 批次成功率: {batch.get_success_rate():.1%}")
    
    # 开始大规模处理
    start_time = time.time()
    batches = await processor.process_all(large_scale_progress_callback)
    end_time = time.time()
    
    # 最终统计
    stats = processor.get_statistics()
    print(f"\n📊 大规模处理统计:")
    print(f"   - 总处理时间: {end_time - start_time:.2f}s")
    print(f"   - 平均吞吐量: {stats['throughput']:.2f} req/s")
    print(f"   - 总成本: ${stats['total_cost']:.6f}")
    print(f"   - 内存峰值: {stats['memory_usage']['rss_mb']:.1f}MB")
    print(f"   - 批次数量: {len(batches)}")
    
    # 成功率分析
    overall_success_rate = sum(batch.get_success_rate() * len(batch.items) for batch in batches) / 50
    print(f"   - 整体成功率: {overall_success_rate:.1%}")

async def main():
    """主演示函数"""
    print("📦 HarborAI 批量处理优化演示")
    print("=" * 60)
    
    try:
        # 基础批量处理演示
        await demo_basic_batch_processing()
        
        # 性能对比演示
        await demo_performance_comparison()
        
        # 内存管理演示
        await demo_memory_management()
        
        # 优先级处理演示
        await demo_priority_processing()
        
        # 大规模处理演示
        await demo_large_scale_processing()
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境建议:")
        print("   1. 根据系统资源调整批次大小和并发数")
        print("   2. 实施内存监控和限制机制")
        print("   3. 使用优先级队列处理重要请求")
        print("   4. 监控处理性能和成本效益")
        print("   5. 实现故障恢复和重试机制")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())