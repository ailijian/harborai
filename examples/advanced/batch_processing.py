#!/usr/bin/env python3
"""
批量处理优化演示

这个示例展示了 HarborAI 的批量处理优化功能，包括：
1. 批量请求聚合
2. 并发控制
3. 内存优化
4. 进度跟踪
5. 结果分发

场景：
- 大量文本需要批量处理（翻译、摘要、分析等）
- 需要控制并发数量避免API限制
- 需要监控内存使用避免OOM
- 需要实时跟踪处理进度

价值：
- 提高处理效率（批量+并发）
- 降低API调用成本
- 提供可靠的错误恢复机制
- 实时监控和进度反馈
"""

import asyncio
import time
import psutil
import json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Callable, Union
from concurrent.futures import ThreadPoolExecutor
import logging

# 导入配置助手
from config_helper import get_primary_model_config, get_fallback_models, print_available_models

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

class BatchStatus(Enum):
    """批次状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingMode(Enum):
    """处理模式"""
    SEQUENTIAL = "sequential"  # 顺序处理
    CONCURRENT = "concurrent"  # 并发处理
    ADAPTIVE = "adaptive"      # 自适应处理

@dataclass
class BatchConfig:
    """批量处理配置"""
    batch_size: int = 10           # 批次大小
    max_concurrent: int = 5        # 最大并发数
    memory_limit_mb: int = 1024    # 内存限制（MB）
    timeout_seconds: int = 90      # 请求超时时间
    retry_attempts: int = 3        # 重试次数
    processing_mode: ProcessingMode = ProcessingMode.CONCURRENT
    enable_progress_callback: bool = True

@dataclass
class RequestItem:
    """请求项"""
    id: str
    prompt: str
    model: Optional[str] = None
    temperature: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class BatchResult:
    """批次结果"""
    request_id: str
    success: bool
    response: Optional[str] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, limit_mb: int = 1024):
        self.limit_mb = limit_mb
        self.process = psutil.Process()
    
    def get_memory_usage_mb(self) -> float:
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def is_memory_available(self, required_mb: float = 100) -> bool:
        """检查是否有足够内存"""
        current_usage = self.get_memory_usage_mb()
        return (current_usage + required_mb) <= self.limit_mb
    
    def get_memory_stats(self) -> Dict[str, float]:
        """获取内存统计"""
        current = self.get_memory_usage_mb()
        return {
            "current_mb": current,
            "limit_mb": self.limit_mb,
            "usage_percent": (current / self.limit_mb) * 100,
            "available_mb": self.limit_mb - current
        }

class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.memory_limit_mb)
        self.results: List[BatchResult] = []
        self.failed_requests: List[RequestItem] = []
        self.processing_stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "failed_requests": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "memory_peak_mb": 0.0
        }
        
        # 初始化 HarborAI 客户端
        model_config = get_primary_model_config()
        if not model_config:
            raise ValueError("没有找到可用的模型配置，请检查环境变量设置")
        
        self.client = HarborAI()
        self.primary_model = model_config.model
        self.fallback_models = get_fallback_models()
        
        logger.info(f"✅ 批量处理器初始化完成")
        logger.info(f"   主要模型: {self.primary_model}")
        logger.info(f"   降级模型: {', '.join(self.fallback_models[1:]) if len(self.fallback_models) > 1 else '无'}")
        logger.info(f"   批次大小: {config.batch_size}")
        logger.info(f"   最大并发: {config.max_concurrent}")
        logger.info(f"   内存限制: {config.memory_limit_mb}MB")
    
    def add_request(self, request: RequestItem) -> None:
        """添加请求到处理队列"""
        if not request.model:
            request.model = self.primary_model
        
        self.processing_stats["total_requests"] += 1
        logger.debug(f"添加请求: {request.id}")
    
    def _create_batch(self, requests: List[RequestItem]) -> List[List[RequestItem]]:
        """创建批次"""
        batches = []
        for i in range(0, len(requests), self.config.batch_size):
            batch = requests[i:i + self.config.batch_size]
            batches.append(batch)
        
        logger.info(f"创建了 {len(batches)} 个批次，总共 {len(requests)} 个请求")
        return batches
    
    async def _process_batch(self, batch: List[RequestItem], batch_index: int) -> List[BatchResult]:
        """处理单个批次"""
        logger.info(f"开始处理批次 {batch_index + 1}，包含 {len(batch)} 个请求")
        
        # 检查内存
        if not self.memory_monitor.is_memory_available():
            logger.warning(f"内存不足，跳过批次 {batch_index + 1}")
            return [
                BatchResult(
                    request_id=req.id,
                    success=False,
                    error="内存不足",
                    processing_time=0.0
                ) for req in batch
            ]
        
        # 并发处理批次中的请求
        if self.config.processing_mode == ProcessingMode.CONCURRENT:
            semaphore = asyncio.Semaphore(self.config.max_concurrent)
            tasks = [
                self._process_single_request(request, semaphore)
                for request in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # 顺序处理
            results = []
            for request in batch:
                result = await self._process_single_request(request)
                results.append(result)
        
        # 处理异常结果
        batch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                batch_results.append(BatchResult(
                    request_id=batch[i].id,
                    success=False,
                    error=str(result),
                    processing_time=0.0
                ))
            else:
                batch_results.append(result)
        
        # 更新内存峰值
        current_memory = self.memory_monitor.get_memory_usage_mb()
        if current_memory > self.processing_stats["memory_peak_mb"]:
            self.processing_stats["memory_peak_mb"] = current_memory
        
        logger.info(f"批次 {batch_index + 1} 处理完成")
        return batch_results
    
    async def _process_single_request(self, request: RequestItem, semaphore: Optional[asyncio.Semaphore] = None) -> BatchResult:
        """处理单个请求"""
        if semaphore:
            async with semaphore:
                return await self._do_process_request(request)
        else:
            return await self._do_process_request(request)
    
    async def _do_process_request(self, request: RequestItem) -> BatchResult:
        """执行单个请求处理"""
        start_time = time.time()
        
        try:
            # 构建请求参数
            request_params = {
                "model": request.model or self.primary_model,
                "messages": [{"role": "user", "content": request.prompt}],
                "temperature": request.temperature or 0.7,
                "timeout": self.config.timeout_seconds
            }
            
            # 发送请求（在线程池中运行同步方法）
            response = await asyncio.to_thread(
                self.client.chat.completions.create, 
                **request_params
            )
            
            processing_time = time.time() - start_time
            
            # 提取响应内容
            content = response.choices[0].message.content if response.choices else "无响应内容"
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') and response.usage else 0
            
            result = BatchResult(
                request_id=request.id,
                success=True,
                response=content,
                processing_time=processing_time,
                model_used=request_params["model"],
                tokens_used=tokens_used
            )
            
            self.processing_stats["completed_requests"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            
            logger.debug(f"请求 {request.id} 处理成功，耗时 {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            result = BatchResult(
                request_id=request.id,
                success=False,
                error=error_msg,
                processing_time=processing_time,
                model_used=request.model
            )
            
            self.processing_stats["failed_requests"] += 1
            self.failed_requests.append(request)
            
            logger.error(f"请求 {request.id} 处理失败: {error_msg}")
            return result
    
    async def process_all(self, requests: List[RequestItem], 
                         progress_callback: Optional[Callable[[int, int], None]] = None) -> List[BatchResult]:
        """处理所有请求"""
        logger.info(f"开始批量处理 {len(requests)} 个请求")
        start_time = time.time()
        
        # 创建批次
        batches = self._create_batch(requests)
        all_results = []
        
        # 处理每个批次
        for i, batch in enumerate(batches):
            batch_results = await self._process_batch(batch, i)
            all_results.extend(batch_results)
            
            # 进度回调
            if progress_callback and self.config.enable_progress_callback:
                completed = (i + 1) * self.config.batch_size
                total = len(requests)
                progress_callback(min(completed, total), total)
        
        # 更新统计信息
        total_time = time.time() - start_time
        if self.processing_stats["completed_requests"] > 0:
            self.processing_stats["average_processing_time"] = (
                self.processing_stats["total_processing_time"] / 
                self.processing_stats["completed_requests"]
            )
        
        logger.info(f"批量处理完成，总耗时 {total_time:.2f}s")
        logger.info(f"成功: {self.processing_stats['completed_requests']}, "
                   f"失败: {self.processing_stats['failed_requests']}")
        
        self.results = all_results
        return all_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        memory_stats = self.memory_monitor.get_memory_stats()
        
        return {
            "processing_stats": self.processing_stats,
            "memory_stats": memory_stats,
            "success_rate": (
                self.processing_stats["completed_requests"] / 
                max(self.processing_stats["total_requests"], 1)
            ) * 100,
            "failed_requests_count": len(self.failed_requests)
        }

# 演示函数
async def demo_basic_batch_processing():
    """演示基础批量处理"""
    print("\n📦 基础批量处理演示")
    print("=" * 50)
    
    # 创建批量处理器
    config = BatchConfig(
        batch_size=5,
        max_concurrent=3,
        timeout_seconds=90,
        memory_limit_mb=512
    )
    processor = BatchProcessor(config)
    
    # 准备测试请求
    test_requests = [
        RequestItem(id=f"req_{i+1}", prompt=f"用一句话解释{topic}（不超过20字）")
        for i, topic in enumerate([
            "人工智能", "机器学习", "深度学习", "自然语言处理", 
            "计算机视觉", "强化学习", "神经网络", "大语言模型"
        ])
    ]
    
    print(f"✅ 准备处理 {len(test_requests)} 个请求")
    
    # 进度回调函数
    def progress_callback(completed: int, total: int):
        progress = completed / total * 100
        print(f"🔄 处理进度: {progress:.1f}% ({completed}/{total})")
    
    # 开始处理
    start_time = time.time()
    results = await processor.process_all(test_requests, progress_callback)
    end_time = time.time()
    
    # 显示结果
    print(f"\n📊 处理结果:")
    print(f"   - 总处理时间: {end_time - start_time:.2f}s")
    print(f"   - 成功请求: {sum(1 for r in results if r.success)}")
    print(f"   - 失败请求: {sum(1 for r in results if not r.success)}")
    
    # 显示统计信息
    stats = processor.get_statistics()
    print(f"   - 成功率: {stats['success_rate']:.1f}%")
    print(f"   - 平均处理时间: {stats['processing_stats']['average_processing_time']:.2f}s")
    print(f"   - 内存峰值: {stats['memory_stats']['current_mb']:.1f}MB")

async def demo_concurrent_vs_sequential():
    """演示并发与顺序处理的性能对比"""
    print("\n⚡ 并发 vs 顺序处理对比")
    print("=" * 50)
    
    # 准备测试数据
    test_requests = [
        RequestItem(id=f"req_{i+1}", prompt=f"简单回答：什么是概念{i+1}？")
        for i in range(6)
    ]
    
    # 1. 顺序处理
    print("🔄 顺序处理测试...")
    sequential_config = BatchConfig(
        batch_size=1,
        max_concurrent=1,
        processing_mode=ProcessingMode.SEQUENTIAL
    )
    sequential_processor = BatchProcessor(sequential_config)
    
    sequential_start = time.time()
    sequential_results = await sequential_processor.process_all(test_requests[:3])
    sequential_time = time.time() - sequential_start
    
    # 2. 并发处理
    print("🔄 并发处理测试...")
    concurrent_config = BatchConfig(
        batch_size=3,
        max_concurrent=3,
        processing_mode=ProcessingMode.CONCURRENT
    )
    concurrent_processor = BatchProcessor(concurrent_config)
    
    concurrent_start = time.time()
    concurrent_results = await concurrent_processor.process_all(test_requests[:3])
    concurrent_time = time.time() - concurrent_start
    
    # 性能对比
    print(f"\n📊 性能对比结果:")
    print(f"   顺序处理:")
    print(f"     - 处理时间: {sequential_time:.2f}s")
    print(f"     - 成功数量: {sum(1 for r in sequential_results if r.success)}")
    
    print(f"   并发处理:")
    print(f"     - 处理时间: {concurrent_time:.2f}s")
    print(f"     - 成功数量: {sum(1 for r in concurrent_results if r.success)}")
    
    if sequential_time > 0 and concurrent_time > 0:
        speedup = sequential_time / concurrent_time
        print(f"   性能提升: {speedup:.2f}x")

async def demo_memory_monitoring():
    """演示内存监控"""
    print("\n🧠 内存监控演示")
    print("=" * 50)
    
    # 创建内存监控器
    memory_monitor = MemoryMonitor(limit_mb=256)
    
    # 显示初始内存状态
    initial_stats = memory_monitor.get_memory_stats()
    print(f"📊 初始内存状态:")
    print(f"   - 当前使用: {initial_stats['current_mb']:.1f}MB")
    print(f"   - 内存限制: {initial_stats['limit_mb']:.1f}MB")
    print(f"   - 使用率: {initial_stats['usage_percent']:.1f}%")
    
    # 创建处理器
    config = BatchConfig(
        batch_size=3,
        max_concurrent=2,
        memory_limit_mb=256
    )
    processor = BatchProcessor(config)
    
    # 添加请求
    test_requests = [
        RequestItem(id=f"req_{i+1}", prompt=f"简短回答问题{i+1}")
        for i in range(6)
    ]
    
    # 处理并监控内存
    await processor.process_all(test_requests)
    
    # 显示最终内存状态
    final_stats = processor.get_statistics()['memory_stats']
    print(f"\n📊 最终内存状态:")
    print(f"   - 当前使用: {final_stats['current_mb']:.1f}MB")
    print(f"   - 内存峰值: {processor.processing_stats['memory_peak_mb']:.1f}MB")

async def demo_error_handling():
    """演示错误处理"""
    print("\n🛡️ 错误处理演示")
    print("=" * 50)
    
    config = BatchConfig(
        batch_size=3,
        max_concurrent=2,
        retry_attempts=2
    )
    processor = BatchProcessor(config)
    
    # 混合正常和可能出错的请求
    test_requests = [
        RequestItem(id="normal_1", prompt="什么是AI？"),
        RequestItem(id="normal_2", prompt="什么是机器学习？"),
        RequestItem(id="invalid_model", prompt="测试请求", model="invalid-model-name"),
        RequestItem(id="normal_3", prompt="什么是深度学习？"),
    ]
    
    results = await processor.process_all(test_requests)
    
    # 分析结果
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"📊 错误处理结果:")
    print(f"   - 成功请求: {len(successful)}")
    print(f"   - 失败请求: {len(failed)}")
    
    if failed:
        print(f"   失败详情:")
        for result in failed:
            print(f"     - {result.request_id}: {result.error}")

async def main():
    """主演示函数"""
    print("📦 HarborAI 批量处理优化演示")
    print("=" * 60)
    
    # 显示可用模型配置
    print_available_models()
    
    try:
        # 基础批量处理演示
        await demo_basic_batch_processing()
        
        # 并发 vs 顺序处理对比
        await demo_concurrent_vs_sequential()
        
        # 内存监控演示
        await demo_memory_monitoring()
        
        # 错误处理演示
        await demo_error_handling()
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境建议:")
        print("   1. 根据系统资源调整批次大小和并发数")
        print("   2. 实施内存监控和限制机制")
        print("   3. 实现完善的错误处理和重试机制")
        print("   4. 监控处理性能和成本效益")
        print("   5. 使用适当的超时配置（当前90秒）")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())