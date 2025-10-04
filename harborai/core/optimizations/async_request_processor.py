#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步请求处理器

实现全面异步化IO操作，支持批量处理、请求合并、流式处理和智能重试。
根据技术设计方案，提供高性能的异步请求处理能力。

设计原则：
1. 全异步IO操作，避免阻塞
2. 批量请求处理，提升吞吐量
3. 请求合并和去重，减少网络开销
4. 流式处理，支持大数据传输
5. 智能重试和错误处理
6. 请求优先级和调度
7. 性能监控和限流

技术特性：
- 异步请求队列
- 批量处理
- 请求合并
- 流式处理
- 智能重试
- 优先级调度
- 限流控制
- 性能监控
"""

import asyncio
import aiohttp
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum, IntEnum
import logging
from urllib.parse import urljoin
import weakref

from .lockfree_plugin_manager import AtomicInteger, AtomicReference
from .optimized_connection_pool import OptimizedConnectionPool, get_connection_pool

logger = logging.getLogger(__name__)


class RequestPriority(IntEnum):
    """请求优先级枚举"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


class RequestStatus(Enum):
    """请求状态枚举"""
    PENDING = "pending"         # 待处理
    PROCESSING = "processing"   # 处理中
    COMPLETED = "completed"     # 已完成
    FAILED = "failed"          # 失败
    CANCELLED = "cancelled"     # 已取消
    RETRYING = "retrying"      # 重试中


@dataclass
class RequestConfig:
    """请求配置"""
    timeout: float = 15.0                    # 减少请求超时到15秒
    max_retries: int = 5                     # 增加最大重试次数到5
    retry_delay: float = 0.5                 # 减少重试延迟到0.5秒
    retry_backoff: float = 1.5               # 减少重试退避倍数到1.5
    enable_compression: bool = True          # 启用压缩
    enable_keepalive: bool = True            # 启用长连接
    max_redirects: int = 5                   # 减少最大重定向次数到5
    chunk_size: int = 16384                  # 增加流式传输块大小到16KB
    enable_request_merging: bool = True      # 启用请求合并
    merge_window: float = 0.05               # 减少请求合并窗口到50ms
    max_batch_size: int = 20                 # 增加最大批处理大小到20
    enable_rate_limiting: bool = True        # 启用限流
    rate_limit_requests: int = 200           # 增加限流请求数到200
    rate_limit_window: float = 60.0          # 限流窗口（秒）


@dataclass
class AsyncRequest:
    """异步请求"""
    id: str
    method: str
    url: str
    headers: Optional[Dict[str, str]] = None
    data: Optional[Union[str, bytes, Dict[str, Any]]] = None
    params: Optional[Dict[str, Any]] = None
    priority: RequestPriority = RequestPriority.NORMAL
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)
    status: 'AtomicReference' = field(default_factory=lambda: AtomicReference(RequestStatus.PENDING))
    retry_count: 'AtomicInteger' = field(default_factory=lambda: AtomicInteger(0))
    future: Optional[asyncio.Future] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.status, RequestStatus):
            self.status = AtomicReference(self.status)
        if isinstance(self.retry_count, int):
            self.retry_count = AtomicInteger(self.retry_count)
        
        # 生成请求哈希用于去重
        self.hash = self._generate_hash()
    
    def _generate_hash(self) -> str:
        """生成请求哈希"""
        content = f"{self.method}:{self.url}:{self.headers}:{self.data}:{self.params}"
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class AsyncResponse:
    """异步响应"""
    request_id: str
    status_code: int
    headers: Dict[str, str]
    data: Union[str, bytes, Dict[str, Any]]
    response_time: float
    success: bool
    error: Optional[str] = None


class RateLimiter:
    """限流器"""
    
    def __init__(self, max_requests: int, window: float):
        """初始化限流器
        
        Args:
            max_requests: 最大请求数
            window: 时间窗口（秒）
        """
        self.max_requests = max_requests
        self.window = window
        self.requests = deque()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """获取请求许可
        
        Returns:
            是否获得许可
        """
        async with self._lock:
            current_time = time.time()
            
            # 清理过期请求
            while self.requests and current_time - self.requests[0] > self.window:
                self.requests.popleft()
            
            # 检查是否超过限制
            if len(self.requests) >= self.max_requests:
                return False
            
            # 记录请求时间
            self.requests.append(current_time)
            return True
    
    async def wait_for_permit(self, timeout: float = 10.0) -> bool:
        """等待请求许可
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            是否获得许可
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self.acquire():
                return True
            await asyncio.sleep(0.1)
        
        return False


class AsyncRequestProcessor:
    """异步请求处理器
    
    实现全面异步化的请求处理，支持批量处理、请求合并、流式处理等高级功能。
    
    主要特性：
    1. 异步请求队列：支持优先级调度
    2. 批量处理：提升吞吐量
    3. 请求合并：减少重复请求
    4. 流式处理：支持大数据传输
    5. 智能重试：指数退避重试策略
    6. 限流控制：防止过载
    7. 性能监控：实时统计
    """
    
    def __init__(self, config: Optional[RequestConfig] = None, 
                 connection_pool: Optional[OptimizedConnectionPool] = None):
        """初始化异步请求处理器
        
        Args:
            config: 请求配置
            connection_pool: 连接池
        """
        self.config = config or RequestConfig()
        self.connection_pool = connection_pool
        
        # 请求队列：按优先级分组
        self._request_queues: Dict[RequestPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in RequestPriority
        }
        
        # 请求映射：用于查找和去重
        self._pending_requests: Dict[str, AsyncRequest] = {}
        self._request_hashes: Dict[str, str] = {}  # hash -> request_id
        
        # 批处理缓冲区
        self._batch_buffer: List[AsyncRequest] = []
        self._batch_lock = asyncio.Lock()
        
        # 统计信息
        self._stats = {
            'total_requests': AtomicInteger(0),
            'completed_requests': AtomicInteger(0),
            'failed_requests': AtomicInteger(0),
            'retried_requests': AtomicInteger(0),
            'merged_requests': AtomicInteger(0),
            'batched_requests': AtomicInteger(0),
            'rate_limited_requests': AtomicInteger(0),
            'active_requests': AtomicInteger(0),
        }
        
        # 性能监控
        self._performance = {
            'avg_response_time': 0.0,
            'max_response_time': 0.0,
            'min_response_time': float('inf'),
            'total_response_time': 0.0,
            'response_time_samples': 0,
            'throughput': 0.0,  # 请求/秒
        }
        self._perf_lock = asyncio.Lock()
        
        # 限流器
        if self.config.enable_rate_limiting:
            self._rate_limiter = RateLimiter(
                self.config.rate_limit_requests,
                self.config.rate_limit_window
            )
        else:
            self._rate_limiter = None
        
        # 处理任务
        self._processor_tasks: List[asyncio.Task] = []
        self._batch_processor_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("AsyncRequestProcessor初始化完成，配置: %s", self.config)
    
    async def start(self, num_workers: int = 8):
        """启动请求处理器
        
        Args:
            num_workers: 工作线程数（默认提升到8个）
        """
        if self._running:
            return
        
        self._running = True
        
        # 获取连接池
        if not self.connection_pool:
            self.connection_pool = await get_connection_pool()
        
        # 启动处理任务
        for i in range(num_workers):
            task = asyncio.create_task(self._request_processor_loop(f"worker-{i}"))
            self._processor_tasks.append(task)
        
        # 启动批处理任务
        if self.config.max_batch_size > 1:
            self._batch_processor_task = asyncio.create_task(self._batch_processor_loop())
        
        logger.info("AsyncRequestProcessor已启动，工作线程数: %d", num_workers)
    
    async def stop(self):
        """停止请求处理器"""
        if not self._running:
            return
        
        self._running = False
        
        # 停止处理任务
        for task in self._processor_tasks:
            task.cancel()
        
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self._processor_tasks, return_exceptions=True)
        if self._batch_processor_task:
            try:
                await self._batch_processor_task
            except asyncio.CancelledError:
                pass
        
        # 取消所有待处理请求
        for request in self._pending_requests.values():
            if request.future and not request.future.done():
                request.future.cancel()
        
        self._pending_requests.clear()
        self._request_hashes.clear()
        
        logger.info("AsyncRequestProcessor已停止")
    
    async def submit_request(self, method: str, url: str, 
                           headers: Optional[Dict[str, str]] = None,
                           data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
                           params: Optional[Dict[str, Any]] = None,
                           priority: RequestPriority = RequestPriority.NORMAL,
                           timeout: Optional[float] = None,
                           max_retries: Optional[int] = None,
                           callback: Optional[Callable] = None) -> asyncio.Future[AsyncResponse]:
        """提交异步请求
        
        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            data: 请求数据
            params: 查询参数
            priority: 请求优先级
            timeout: 超时时间
            max_retries: 最大重试次数
            callback: 回调函数
            
        Returns:
            异步响应Future
        """
        # 创建请求
        request = AsyncRequest(
            id=f"req_{int(time.time() * 1000000)}_{id(self)}",
            method=method.upper(),
            url=url,
            headers=headers or {},
            data=data,
            params=params,
            priority=priority,
            timeout=timeout or self.config.timeout,
            max_retries=max_retries or self.config.max_retries,
            callback=callback,
        )
        
        # 创建Future
        request.future = asyncio.Future()
        
        # 检查请求合并
        if self.config.enable_request_merging:
            existing_request_id = self._request_hashes.get(request.hash)
            if existing_request_id and existing_request_id in self._pending_requests:
                # 合并请求
                existing_request = self._pending_requests[existing_request_id]
                self._stats['merged_requests'].increment()
                logger.debug("合并重复请求: %s", request.hash)
                return existing_request.future
        
        # 记录请求
        self._pending_requests[request.id] = request
        self._request_hashes[request.hash] = request.id
        self._stats['total_requests'].increment()
        
        # 添加到队列
        await self._request_queues[priority].put(request)
        
        logger.debug("提交请求: %s %s (优先级: %s)", method, url, priority.name)
        return request.future
    
    async def submit_batch_requests(self, requests: List[Dict[str, Any]]) -> List[asyncio.Future[AsyncResponse]]:
        """批量提交请求
        
        Args:
            requests: 请求列表
            
        Returns:
            异步响应Future列表
        """
        futures = []
        
        for req_data in requests:
            future = await self.submit_request(**req_data)
            futures.append(future)
        
        return futures
    
    async def _request_processor_loop(self, worker_name: str):
        """请求处理循环
        
        Args:
            worker_name: 工作线程名称
        """
        logger.debug("启动请求处理器: %s", worker_name)
        
        while self._running:
            try:
                # 按优先级处理请求
                request = await self._get_next_request()
                if request:
                    await self._process_request(request)
                else:
                    # 没有请求，短暂休眠
                    await asyncio.sleep(0.01)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("请求处理器 %s 异常: %s", worker_name, str(e))
                await asyncio.sleep(0.1)
        
        logger.debug("请求处理器 %s 已停止", worker_name)
    
    async def _get_next_request(self) -> Optional[AsyncRequest]:
        """获取下一个请求
        
        按优先级顺序获取请求。
        
        Returns:
            下一个请求，如果没有则返回None
        """
        # 按优先级从高到低检查队列
        for priority in sorted(RequestPriority, reverse=True):
            queue = self._request_queues[priority]
            try:
                request = queue.get_nowait()
                return request
            except asyncio.QueueEmpty:
                continue
        
        return None
    
    async def _process_request(self, request: AsyncRequest):
        """处理单个请求
        
        Args:
            request: 异步请求
        """
        try:
            # 检查限流
            if self._rate_limiter:
                if not await self._rate_limiter.wait_for_permit():
                    self._stats['rate_limited_requests'].increment()
                    await self._complete_request(request, None, "Rate limit exceeded")
                    return
            
            # 更新状态
            request.status.set(RequestStatus.PROCESSING)
            self._stats['active_requests'].increment()
            
            # 执行请求
            start_time = time.perf_counter()
            response = await self._execute_request(request)
            response_time = (time.perf_counter() - start_time) * 1000
            
            # 更新性能统计
            await self._update_performance_stats(response_time)
            
            # 完成请求
            await self._complete_request(request, response)
            
        except Exception as e:
            logger.error("处理请求失败 %s: %s", request.id, str(e))
            await self._handle_request_error(request, str(e))
        finally:
            self._stats['active_requests'].decrement()
    
    async def _execute_request(self, request: AsyncRequest) -> AsyncResponse:
        """执行HTTP请求
        
        Args:
            request: 异步请求
            
        Returns:
            异步响应
        """
        session = await self.connection_pool.get_session(request.url)
        if not session:
            raise Exception("无法获取连接会话")
        
        try:
            # 准备请求参数
            kwargs = {
                'timeout': aiohttp.ClientTimeout(total=request.timeout),
                'headers': request.headers,
            }
            
            if request.params:
                kwargs['params'] = request.params
            
            if request.data:
                if isinstance(request.data, dict):
                    kwargs['json'] = request.data
                else:
                    kwargs['data'] = request.data
            
            # 执行请求
            async with session.request(request.method, request.url, **kwargs) as response:
                # 读取响应数据
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    data = await response.text()
                
                # 创建响应对象
                async_response = AsyncResponse(
                    request_id=request.id,
                    status_code=response.status,
                    headers=dict(response.headers),
                    data=data,
                    response_time=time.time() - request.created_at,
                    success=200 <= response.status < 400,
                )
                
                # 归还连接
                await self.connection_pool.return_session(
                    request.url, session, async_response.success
                )
                
                return async_response
                
        except Exception as e:
            # 归还连接（标记为失败）
            await self.connection_pool.return_session(request.url, session, False)
            raise e
    
    async def _handle_request_error(self, request: AsyncRequest, error: str):
        """处理请求错误
        
        Args:
            request: 异步请求
            error: 错误信息
        """
        retry_count = request.retry_count.get()
        max_retries = request.max_retries or self.config.max_retries
        
        if retry_count < max_retries:
            # 重试请求
            request.retry_count.increment()
            request.status.set(RequestStatus.RETRYING)
            self._stats['retried_requests'].increment()
            
            # 计算重试延迟（指数退避）
            delay = self.config.retry_delay * (self.config.retry_backoff ** retry_count)
            
            logger.debug("重试请求 %s，第 %d 次重试，延迟 %.2f 秒", 
                        request.id, retry_count + 1, delay)
            
            # 延迟后重新加入队列
            await asyncio.sleep(delay)
            await self._request_queues[request.priority].put(request)
        else:
            # 重试次数用尽，标记为失败
            await self._complete_request(request, None, error)
    
    async def _complete_request(self, request: AsyncRequest, 
                              response: Optional[AsyncResponse], 
                              error: Optional[str] = None):
        """完成请求处理
        
        Args:
            request: 异步请求
            response: 异步响应
            error: 错误信息
        """
        try:
            # 更新状态
            if response and response.success:
                request.status.set(RequestStatus.COMPLETED)
                self._stats['completed_requests'].increment()
            else:
                request.status.set(RequestStatus.FAILED)
                self._stats['failed_requests'].increment()
            
            # 清理请求记录
            if request.id in self._pending_requests:
                del self._pending_requests[request.id]
            
            if request.hash in self._request_hashes:
                del self._request_hashes[request.hash]
            
            # 设置Future结果
            if request.future and not request.future.done():
                if response:
                    request.future.set_result(response)
                else:
                    request.future.set_exception(Exception(error or "Request failed"))
            
            # 调用回调函数
            if request.callback:
                try:
                    if asyncio.iscoroutinefunction(request.callback):
                        await request.callback(response, error)
                    else:
                        request.callback(response, error)
                except Exception as e:
                    logger.error("回调函数执行失败: %s", str(e))
            
        except Exception as e:
            logger.error("完成请求处理失败 %s: %s", request.id, str(e))
    
    async def _batch_processor_loop(self):
        """批处理循环"""
        logger.debug("启动批处理器")
        
        while self._running:
            try:
                await asyncio.sleep(self.config.merge_window)
                await self._process_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("批处理器异常: %s", str(e))
                await asyncio.sleep(0.1)
        
        logger.debug("批处理器已停止")
    
    async def _process_batch(self):
        """处理批次"""
        async with self._batch_lock:
            if not self._batch_buffer:
                return
            
            batch = self._batch_buffer[:self.config.max_batch_size]
            self._batch_buffer = self._batch_buffer[self.config.max_batch_size:]
            
            if len(batch) > 1:
                self._stats['batched_requests'].add(len(batch))
                logger.debug("批处理 %d 个请求", len(batch))
                
                # 并发处理批次中的请求
                tasks = [self._process_request(request) for request in batch]
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _update_performance_stats(self, response_time: float):
        """更新性能统计
        
        Args:
            response_time: 响应时间（毫秒）
        """
        async with self._perf_lock:
            self._performance['total_response_time'] += response_time
            self._performance['response_time_samples'] += 1
            
            if response_time > self._performance['max_response_time']:
                self._performance['max_response_time'] = response_time
            
            if response_time < self._performance['min_response_time']:
                self._performance['min_response_time'] = response_time
            
            # 计算平均响应时间
            self._performance['avg_response_time'] = (
                self._performance['total_response_time'] / 
                self._performance['response_time_samples']
            )
            
            # 计算吞吐量（最近1分钟）
            current_time = time.time()
            if not hasattr(self, '_throughput_window'):
                self._throughput_window = deque()
            
            self._throughput_window.append(current_time)
            
            # 清理1分钟前的记录
            while (self._throughput_window and 
                   current_time - self._throughput_window[0] > 60.0):
                self._throughput_window.popleft()
            
            self._performance['throughput'] = len(self._throughput_window) / 60.0
    
    async def stream_request(self, method: str, url: str,
                           headers: Optional[Dict[str, str]] = None,
                           data: Optional[Union[str, bytes]] = None,
                           params: Optional[Dict[str, Any]] = None,
                           chunk_size: Optional[int] = None) -> AsyncGenerator[bytes, None]:
        """流式请求处理
        
        Args:
            method: HTTP方法
            url: 请求URL
            headers: 请求头
            data: 请求数据
            params: 查询参数
            chunk_size: 块大小
            
        Yields:
            数据块
        """
        session = await self.connection_pool.get_session(url)
        if not session:
            raise Exception("无法获取连接会话")
        
        try:
            kwargs = {
                'headers': headers or {},
                'timeout': aiohttp.ClientTimeout(total=self.config.timeout),
            }
            
            if params:
                kwargs['params'] = params
            if data:
                kwargs['data'] = data
            
            async with session.request(method, url, **kwargs) as response:
                if response.status >= 400:
                    await self.connection_pool.return_session(url, session, False)
                    raise Exception(f"HTTP {response.status}: {response.reason}")
                
                chunk_size = chunk_size or self.config.chunk_size
                
                async for chunk in response.content.iter_chunked(chunk_size):
                    yield chunk
                
                await self.connection_pool.return_session(url, session, True)
                
        except Exception as e:
            await self.connection_pool.return_session(url, session, False)
            raise e
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取处理器统计信息
        
        Returns:
            统计信息字典
        """
        total_requests = self._stats['total_requests'].get()
        completed_requests = self._stats['completed_requests'].get()
        success_rate = (completed_requests / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_requests": total_requests,
            "completed_requests": completed_requests,
            "failed_requests": self._stats['failed_requests'].get(),
            "retried_requests": self._stats['retried_requests'].get(),
            "merged_requests": self._stats['merged_requests'].get(),
            "batched_requests": self._stats['batched_requests'].get(),
            "rate_limited_requests": self._stats['rate_limited_requests'].get(),
            "active_requests": self._stats['active_requests'].get(),
            "pending_requests": len(self._pending_requests),
            "success_rate_percent": round(success_rate, 2),
            "performance": dict(self._performance),
            "queue_sizes": {
                priority.name: queue.qsize() 
                for priority, queue in self._request_queues.items()
            },
            "config": {
                "timeout": self.config.timeout,
                "max_retries": self.config.max_retries,
                "max_batch_size": self.config.max_batch_size,
                "enable_request_merging": self.config.enable_request_merging,
                "enable_rate_limiting": self.config.enable_rate_limiting,
            }
        }


# 全局请求处理器实例
_global_request_processor: Optional[AsyncRequestProcessor] = None
_processor_ref = AtomicReference(None)


async def get_request_processor(config: Optional[RequestConfig] = None) -> AsyncRequestProcessor:
    """获取全局请求处理器实例
    
    Args:
        config: 请求配置
        
    Returns:
        请求处理器实例
    """
    processor = _processor_ref.get()
    
    if processor is None:
        # 创建新处理器
        new_processor = AsyncRequestProcessor(config)
        
        # 使用CAS操作设置全局实例
        if _processor_ref.compare_and_swap(None, new_processor):
            await new_processor.start()
            return new_processor
        else:
            # 其他协程已经创建了实例
            await new_processor.stop()  # 清理未使用的实例
            return _processor_ref.get()
    
    return processor


async def reset_request_processor():
    """重置全局请求处理器
    
    主要用于测试场景。
    """
    processor = _processor_ref.get()
    
    if processor is not None:
        await processor.stop()
        _processor_ref.set(None)