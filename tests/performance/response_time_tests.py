"""
响应时间测试模块

该模块提供全面的响应时间测试功能，支持：
- 同步API响应时间测试
- 异步API响应时间测试
- 流式响应时间测试
- 响应时间分布分析
- 百分位数统计
- 响应时间回归检测

作者：HarborAI性能测试团队
创建时间：2024年
"""

import asyncio
import time
import statistics
from typing import Dict, List, Optional, Any, Callable, Union, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import json
import aiohttp
import requests
from pathlib import Path

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ResponseTimeMetrics:
    """响应时间指标"""
    test_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times: List[float]
    error_messages: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def average_response_time(self) -> float:
        """平均响应时间"""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def median_response_time(self) -> float:
        """中位数响应时间"""
        if not self.response_times:
            return 0.0
        return statistics.median(self.response_times)
    
    @property
    def min_response_time(self) -> float:
        """最小响应时间"""
        if not self.response_times:
            return 0.0
        return min(self.response_times)
    
    @property
    def max_response_time(self) -> float:
        """最大响应时间"""
        if not self.response_times:
            return 0.0
        return max(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        """95百分位响应时间"""
        if not self.response_times:
            return 0.0
        return self._percentile(self.response_times, 95)
    
    @property
    def p99_response_time(self) -> float:
        """99百分位响应时间"""
        if not self.response_times:
            return 0.0
        return self._percentile(self.response_times, 99)
    
    @property
    def standard_deviation(self) -> float:
        """标准差"""
        if len(self.response_times) < 2:
            return 0.0
        return statistics.stdev(self.response_times)
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'test_name': self.test_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration': str(self.end_time - self.start_time),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.success_rate,
            'average_response_time': self.average_response_time,
            'median_response_time': self.median_response_time,
            'min_response_time': self.min_response_time,
            'max_response_time': self.max_response_time,
            'p95_response_time': self.p95_response_time,
            'p99_response_time': self.p99_response_time,
            'standard_deviation': self.standard_deviation,
            'error_messages': self.error_messages
        }


@dataclass
class StreamingMetrics:
    """流式响应指标"""
    test_name: str
    start_time: datetime
    end_time: datetime
    total_streams: int
    successful_streams: int
    failed_streams: int
    first_token_times: List[float]  # 首个token响应时间
    total_stream_times: List[float]  # 完整流响应时间
    token_counts: List[int]  # 每个流的token数量
    tokens_per_second: List[float]  # 每秒token数
    error_messages: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_streams == 0:
            return 0.0
        return self.successful_streams / self.total_streams
    
    @property
    def average_first_token_time(self) -> float:
        """平均首token时间"""
        if not self.first_token_times:
            return 0.0
        return statistics.mean(self.first_token_times)
    
    @property
    def average_total_time(self) -> float:
        """平均总时间"""
        if not self.total_stream_times:
            return 0.0
        return statistics.mean(self.total_stream_times)
    
    @property
    def average_tokens_per_second(self) -> float:
        """平均每秒token数"""
        if not self.tokens_per_second:
            return 0.0
        return statistics.mean(self.tokens_per_second)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'test_name': self.test_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration': str(self.end_time - self.start_time),
            'total_streams': self.total_streams,
            'successful_streams': self.successful_streams,
            'failed_streams': self.failed_streams,
            'success_rate': self.success_rate,
            'average_first_token_time': self.average_first_token_time,
            'average_total_time': self.average_total_time,
            'average_tokens_per_second': self.average_tokens_per_second,
            'p95_first_token_time': self._percentile(self.first_token_times, 95),
            'p99_first_token_time': self._percentile(self.first_token_times, 99),
            'error_messages': self.error_messages
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))


class ResponseTimeTimer:
    """响应时间计时器上下文管理器"""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_time(self) -> float:
        """获取经过的时间（秒）"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time


class ResponseTimeTester:
    """
    响应时间测试器
    
    功能特性：
    - 同步API响应时间测试
    - 异步API响应时间测试
    - 流式响应时间测试
    - 批量并发测试
    - 响应时间分析和统计
    """
    
    def __init__(self, timeout: float = 30.0, max_workers: int = 10):
        """
        初始化响应时间测试器
        
        参数:
            timeout: 请求超时时间（秒）
            max_workers: 最大并发工作线程数
        """
        self.timeout = timeout
        self.max_workers = max_workers
        self.session = requests.Session()
        
        # 配置请求会话
        self.session.headers.update({
            'User-Agent': 'HarborAI-Performance-Tester/1.0'
        })
        
        logger.info(f"响应时间测试器初始化完成，超时: {timeout}s，最大工作线程: {max_workers}")
    
    def test_sync_api(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        num_requests: int = 100,
        test_name: str = "同步API测试"
    ) -> ResponseTimeMetrics:
        """
        测试同步API响应时间
        
        参数:
            url: API端点URL
            method: HTTP方法
            headers: 请求头
            data: 请求数据
            params: 查询参数
            num_requests: 请求次数
            test_name: 测试名称
        
        返回:
            响应时间指标
        """
        logger.info(f"开始同步API测试: {test_name}, URL: {url}, 请求数: {num_requests}")
        
        start_time = datetime.now()
        response_times = []
        error_messages = []
        successful_requests = 0
        failed_requests = 0
        
        # 准备请求参数
        request_kwargs = {
            'timeout': self.timeout,
            'headers': headers or {},
            'params': params
        }
        
        if data is not None:
            if method.upper() in ['POST', 'PUT', 'PATCH']:
                if isinstance(data, dict):
                    request_kwargs['json'] = data
                else:
                    request_kwargs['data'] = data
        
        # 执行请求
        for i in range(num_requests):
            try:
                with ResponseTimeTimer() as timer:
                    response = self.session.request(method, url, **request_kwargs)
                    response.raise_for_status()
                
                response_times.append(timer.elapsed_time)
                successful_requests += 1
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"已完成 {i + 1}/{num_requests} 个请求")
                    
            except Exception as e:
                failed_requests += 1
                error_messages.append(f"请求 {i + 1} 失败: {str(e)}")
                logger.warning(f"请求 {i + 1} 失败: {e}")
        
        end_time = datetime.now()
        
        metrics = ResponseTimeMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_requests=num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            response_times=response_times,
            error_messages=error_messages
        )
        
        logger.info(f"同步API测试完成: 成功率 {metrics.success_rate:.1%}, "
                   f"平均响应时间 {metrics.average_response_time:.3f}s")
        
        return metrics
    
    async def test_async_api(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        num_requests: int = 100,
        concurrent_requests: int = 10,
        test_name: str = "异步API测试"
    ) -> ResponseTimeMetrics:
        """
        测试异步API响应时间
        
        参数:
            url: API端点URL
            method: HTTP方法
            headers: 请求头
            data: 请求数据
            params: 查询参数
            num_requests: 总请求次数
            concurrent_requests: 并发请求数
            test_name: 测试名称
        
        返回:
            响应时间指标
        """
        logger.info(f"开始异步API测试: {test_name}, URL: {url}, "
                   f"请求数: {num_requests}, 并发数: {concurrent_requests}")
        
        start_time = datetime.now()
        response_times = []
        error_messages = []
        successful_requests = 0
        failed_requests = 0
        
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def make_request(session: aiohttp.ClientSession, request_id: int) -> None:
            """执行单个异步请求"""
            nonlocal successful_requests, failed_requests
            
            async with semaphore:
                try:
                    # 准备请求参数
                    request_kwargs = {
                        'timeout': aiohttp.ClientTimeout(total=self.timeout),
                        'headers': headers or {},
                        'params': params
                    }
                    
                    if data is not None and method.upper() in ['POST', 'PUT', 'PATCH']:
                        if isinstance(data, dict):
                            request_kwargs['json'] = data
                        else:
                            request_kwargs['data'] = data
                    
                    with ResponseTimeTimer() as timer:
                        async with session.request(method, url, **request_kwargs) as response:
                            response.raise_for_status()
                            await response.read()  # 确保完全读取响应
                    
                    response_times.append(timer.elapsed_time)
                    successful_requests += 1
                    
                except Exception as e:
                    failed_requests += 1
                    error_messages.append(f"请求 {request_id} 失败: {str(e)}")
                    logger.warning(f"异步请求 {request_id} 失败: {e}")
        
        # 执行所有异步请求
        async with aiohttp.ClientSession() as session:
            tasks = [make_request(session, i + 1) for i in range(num_requests)]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        
        metrics = ResponseTimeMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_requests=num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            response_times=response_times,
            error_messages=error_messages
        )
        
        logger.info(f"异步API测试完成: 成功率 {metrics.success_rate:.1%}, "
                   f"平均响应时间 {metrics.average_response_time:.3f}s")
        
        return metrics
    
    async def test_streaming_api(
        self,
        url: str,
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        num_streams: int = 10,
        test_name: str = "流式API测试",
        token_parser: Optional[Callable[[str], int]] = None
    ) -> StreamingMetrics:
        """
        测试流式API响应时间
        
        参数:
            url: API端点URL
            method: HTTP方法
            headers: 请求头
            data: 请求数据
            params: 查询参数
            num_streams: 流测试次数
            test_name: 测试名称
            token_parser: token解析函数
        
        返回:
            流式响应指标
        """
        logger.info(f"开始流式API测试: {test_name}, URL: {url}, 流数: {num_streams}")
        
        start_time = datetime.now()
        first_token_times = []
        total_stream_times = []
        token_counts = []
        tokens_per_second = []
        error_messages = []
        successful_streams = 0
        failed_streams = 0
        
        async def test_single_stream(session: aiohttp.ClientSession, stream_id: int) -> None:
            """测试单个流"""
            nonlocal successful_streams, failed_streams
            
            try:
                # 准备请求参数
                request_kwargs = {
                    'timeout': aiohttp.ClientTimeout(total=self.timeout),
                    'headers': headers or {},
                    'params': params
                }
                
                if data is not None:
                    if isinstance(data, dict):
                        request_kwargs['json'] = data
                    else:
                        request_kwargs['data'] = data
                
                stream_start = time.perf_counter()
                first_token_time = None
                token_count = 0
                
                async with session.request(method, url, **request_kwargs) as response:
                    response.raise_for_status()
                    
                    async for chunk in response.content.iter_chunked(1024):
                        if chunk:
                            # 记录首个token时间
                            if first_token_time is None:
                                first_token_time = time.perf_counter() - stream_start
                            
                            # 解析token数量
                            if token_parser:
                                try:
                                    chunk_tokens = token_parser(chunk.decode('utf-8', errors='ignore'))
                                    token_count += chunk_tokens
                                except Exception:
                                    token_count += 1  # 默认每个chunk算1个token
                            else:
                                token_count += 1
                
                stream_end = time.perf_counter()
                total_time = stream_end - stream_start
                
                # 记录指标
                if first_token_time is not None:
                    first_token_times.append(first_token_time)
                total_stream_times.append(total_time)
                token_counts.append(token_count)
                
                if total_time > 0 and token_count > 0:
                    tokens_per_second.append(token_count / total_time)
                
                successful_streams += 1
                
            except Exception as e:
                failed_streams += 1
                error_messages.append(f"流 {stream_id} 失败: {str(e)}")
                logger.warning(f"流式请求 {stream_id} 失败: {e}")
        
        # 执行所有流测试
        async with aiohttp.ClientSession() as session:
            tasks = [test_single_stream(session, i + 1) for i in range(num_streams)]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        
        metrics = StreamingMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_streams=num_streams,
            successful_streams=successful_streams,
            failed_streams=failed_streams,
            first_token_times=first_token_times,
            total_stream_times=total_stream_times,
            token_counts=token_counts,
            tokens_per_second=tokens_per_second,
            error_messages=error_messages
        )
        
        logger.info(f"流式API测试完成: 成功率 {metrics.success_rate:.1%}, "
                   f"平均首token时间 {metrics.average_first_token_time:.3f}s, "
                   f"平均token/秒 {metrics.average_tokens_per_second:.1f}")
        
        return metrics
    
    async def test_async_api_response_time(
        self,
        url: str = "https://httpbin.org/delay/0.1",
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        num_requests: int = 100,
        concurrent_requests: int = 10,
        test_name: str = "异步API响应时间测试"
    ) -> ResponseTimeMetrics:
        """
        异步API响应时间专项测试
        
        专门用于测试异步API的响应时间性能，包含内存优化策略。
        
        参数:
            url: API端点URL，默认使用httpbin测试端点
            method: HTTP方法
            headers: 请求头
            data: 请求数据
            params: 查询参数
            num_requests: 总请求次数
            concurrent_requests: 并发请求数
            test_name: 测试名称
        
        返回:
            ResponseTimeMetrics: 响应时间指标对象
        
        内存优化特性:
            - 使用生成器减少内存占用
            - 实时清理临时数据
            - 内存使用监控和警告
        """
        import gc
        import psutil
        
        logger.info(f"开始异步API响应时间测试: {test_name}, URL: {url}, "
                   f"请求数: {num_requests}, 并发数: {concurrent_requests}")
        
        # 内存监控 - 记录初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"测试开始时内存使用: {initial_memory:.2f} MB")
        
        start_time = datetime.now()
        response_times = []
        error_messages = []
        successful_requests = 0
        failed_requests = 0
        
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def make_request_with_memory_optimization(session: aiohttp.ClientSession, request_id: int) -> Optional[float]:
            """执行单个异步请求，包含内存优化"""
            nonlocal successful_requests, failed_requests
            
            async with semaphore:
                try:
                    # 准备请求参数
                    request_kwargs = {
                        'timeout': aiohttp.ClientTimeout(total=self.timeout),
                        'headers': headers or {},
                        'params': params
                    }
                    
                    if data is not None and method.upper() in ['POST', 'PUT', 'PATCH']:
                        if isinstance(data, dict):
                            request_kwargs['json'] = data
                        else:
                            request_kwargs['data'] = data
                    
                    with ResponseTimeTimer() as timer:
                        async with session.request(method, url, **request_kwargs) as response:
                            response.raise_for_status()
                            # 读取响应但不存储大量数据
                            await response.read()
                    
                    successful_requests += 1
                    return timer.elapsed_time
                    
                except Exception as e:
                    failed_requests += 1
                    error_msg = f"请求 {request_id} 失败: {str(e)}"
                    error_messages.append(error_msg)
                    logger.warning(error_msg)
                    return None
        
        # 分批处理请求以控制内存使用
        batch_size = min(50, num_requests)  # 每批最多50个请求
        
        async with aiohttp.ClientSession() as session:
            for batch_start in range(0, num_requests, batch_size):
                batch_end = min(batch_start + batch_size, num_requests)
                batch_tasks = [
                    make_request_with_memory_optimization(session, i + 1) 
                    for i in range(batch_start, batch_end)
                ]
                
                # 执行当前批次
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # 收集有效的响应时间
                for result in batch_results:
                    if isinstance(result, float):
                        response_times.append(result)
                
                # 内存清理
                del batch_results
                del batch_tasks
                gc.collect()
                
                # 内存监控
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                if current_memory > initial_memory * 2:  # 内存使用超过初始值2倍时警告
                    logger.warning(f"内存使用过高: {current_memory:.2f} MB (初始: {initial_memory:.2f} MB)")
                
                # 批次间短暂休息，避免过度占用资源
                if batch_end < num_requests:
                    await asyncio.sleep(0.01)
        
        end_time = datetime.now()
        
        # 最终内存检查
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        logger.info(f"测试结束时内存使用: {final_memory:.2f} MB (增加: {memory_increase:.2f} MB)")
        
        metrics = ResponseTimeMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_requests=num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            response_times=response_times,
            error_messages=error_messages
        )
        
        logger.info(f"异步API响应时间测试完成: 成功率 {metrics.success_rate:.1%}, "
                   f"平均响应时间 {metrics.average_response_time:.3f}s, "
                   f"内存增长 {memory_increase:.2f} MB")
        
        return metrics
    
    def test_concurrent_response_time(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        num_requests: int = 100,
        concurrent_users: int = 10,
        test_name: str = "并发响应时间测试"
    ) -> ResponseTimeMetrics:
        """
        测试并发场景下的响应时间
        
        参数:
            url: API端点URL
            method: HTTP方法
            headers: 请求头
            data: 请求数据
            params: 查询参数
            num_requests: 总请求次数
            concurrent_users: 并发用户数
            test_name: 测试名称
        
        返回:
            响应时间指标
        """
        logger.info(f"开始并发响应时间测试: {test_name}, URL: {url}, "
                   f"请求数: {num_requests}, 并发用户: {concurrent_users}")
        
        start_time = datetime.now()
        response_times = []
        error_messages = []
        successful_requests = 0
        failed_requests = 0
        
        def make_request(request_id: int) -> None:
            """执行单个请求"""
            nonlocal successful_requests, failed_requests
            
            try:
                # 准备请求参数
                request_kwargs = {
                    'timeout': self.timeout,
                    'headers': headers or {},
                    'params': params
                }
                
                if data is not None and method.upper() in ['POST', 'PUT', 'PATCH']:
                    if isinstance(data, dict):
                        request_kwargs['json'] = data
                    else:
                        request_kwargs['data'] = data
                
                with ResponseTimeTimer() as timer:
                    response = self.session.request(method, url, **request_kwargs)
                    response.raise_for_status()
                
                response_times.append(timer.elapsed_time)
                successful_requests += 1
                
            except Exception as e:
                failed_requests += 1
                error_messages.append(f"请求 {request_id} 失败: {str(e)}")
                logger.warning(f"并发请求 {request_id} 失败: {e}")
        
        # 使用线程池执行并发请求
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_request, i + 1) for i in range(num_requests)]
            
            # 等待所有请求完成
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"并发请求执行异常: {e}")
        
        end_time = datetime.now()
        
        metrics = ResponseTimeMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_requests=num_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            response_times=response_times,
            error_messages=error_messages
        )
        
        logger.info(f"并发响应时间测试完成: 成功率 {metrics.success_rate:.1%}, "
                   f"平均响应时间 {metrics.average_response_time:.3f}s")
        
        return metrics
    
    def analyze_response_time_distribution(
        self,
        metrics: ResponseTimeMetrics,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        分析响应时间分布
        
        参数:
            metrics: 响应时间指标
            output_file: 输出文件路径
        
        返回:
            分析结果
        """
        if not metrics.response_times:
            return {}
        
        # 计算分布统计
        response_times = sorted(metrics.response_times)
        
        # 分桶统计
        buckets = {
            '< 100ms': 0,
            '100ms - 500ms': 0,
            '500ms - 1s': 0,
            '1s - 2s': 0,
            '2s - 5s': 0,
            '> 5s': 0
        }
        
        for rt in response_times:
            if rt < 0.1:
                buckets['< 100ms'] += 1
            elif rt < 0.5:
                buckets['100ms - 500ms'] += 1
            elif rt < 1.0:
                buckets['500ms - 1s'] += 1
            elif rt < 2.0:
                buckets['1s - 2s'] += 1
            elif rt < 5.0:
                buckets['2s - 5s'] += 1
            else:
                buckets['> 5s'] += 1
        
        # 百分位数分析
        percentiles = {}
        for p in [50, 75, 90, 95, 99, 99.9]:
            percentiles[f'p{p}'] = metrics._percentile(response_times, p)
        
        analysis = {
            'test_name': metrics.test_name,
            'total_requests': metrics.total_requests,
            'successful_requests': metrics.successful_requests,
            'success_rate': metrics.success_rate,
            'statistics': {
                'mean': metrics.average_response_time,
                'median': metrics.median_response_time,
                'min': metrics.min_response_time,
                'max': metrics.max_response_time,
                'std_dev': metrics.standard_deviation
            },
            'percentiles': percentiles,
            'distribution_buckets': buckets,
            'performance_grade': self._grade_performance(metrics)
        }
        
        # 保存分析结果
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            logger.info(f"响应时间分析结果已保存到: {output_path}")
        
        return analysis
    
    def _grade_performance(self, metrics: ResponseTimeMetrics) -> str:
        """
        评估性能等级
        
        参数:
            metrics: 响应时间指标
        
        返回:
            性能等级 (A/B/C/D/F)
        """
        if metrics.success_rate < 0.95:
            return 'F'  # 成功率过低
        
        avg_time = metrics.average_response_time
        p95_time = metrics.p95_response_time
        
        if avg_time <= 0.1 and p95_time <= 0.2:
            return 'A'  # 优秀
        elif avg_time <= 0.3 and p95_time <= 0.5:
            return 'B'  # 良好
        elif avg_time <= 0.5 and p95_time <= 1.0:
            return 'C'  # 一般
        elif avg_time <= 1.0 and p95_time <= 2.0:
            return 'D'  # 较差
        else:
            return 'F'  # 很差
    
    def test_api_response_time(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        num_requests: int = 100,
        test_name: str = "API响应时间测试"
    ) -> ResponseTimeMetrics:
        """
        测试API响应时间的便捷方法
        
        Args:
            url: 请求URL
            method: HTTP方法
            headers: 请求头
            data: 请求数据
            params: 请求参数
            num_requests: 请求次数
            test_name: 测试名称
            
        Returns:
            ResponseTimeMetrics: 响应时间指标
        """
        return self.test_sync_api(
            url=url,
            method=method,
            headers=headers,
            data=data,
            params=params,
            num_requests=num_requests,
            test_name=test_name
        )

    def close(self):
        """关闭测试器"""
        self.session.close()
        logger.info("响应时间测试器已关闭")


# 便捷函数
def test_api_response_time(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Any] = None,
    num_requests: int = 100,
    test_name: str = "API响应时间测试"
) -> ResponseTimeMetrics:
    """
    快速测试API响应时间
    
    参数:
        url: API端点URL
        method: HTTP方法
        headers: 请求头
        data: 请求数据
        num_requests: 请求次数
        test_name: 测试名称
    
    返回:
        响应时间指标
    """
    tester = ResponseTimeTester()
    try:
        return tester.test_sync_api(
            url=url,
            method=method,
            headers=headers,
            data=data,
            num_requests=num_requests,
            test_name=test_name
        )
    finally:
        tester.close()


async def test_async_api_response_time(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Any] = None,
    num_requests: int = 100,
    concurrent_requests: int = 10,
    test_name: str = "异步API响应时间测试"
) -> ResponseTimeMetrics:
    """
    快速测试异步API响应时间
    
    参数:
        url: API端点URL
        method: HTTP方法
        headers: 请求头
        data: 请求数据
        num_requests: 总请求次数
        concurrent_requests: 并发请求数
        test_name: 测试名称
    
    返回:
        响应时间指标
    """
    tester = ResponseTimeTester()
    try:
        return await tester.test_async_api(
            url=url,
            method=method,
            headers=headers,
            data=data,
            num_requests=num_requests,
            concurrent_requests=concurrent_requests,
            test_name=test_name
        )
    finally:
        tester.close()


if __name__ == "__main__":
    # 示例使用
    import asyncio
    
    async def main():
        # 创建测试器
        tester = ResponseTimeTester(timeout=10.0, max_workers=5)
        
        try:
            # 测试同步API
            sync_metrics = tester.test_sync_api(
                url="https://httpbin.org/delay/0.1",
                num_requests=50,
                test_name="HTTPBin延迟测试"
            )
            
            print("同步API测试结果:")
            print(f"成功率: {sync_metrics.success_rate:.1%}")
            print(f"平均响应时间: {sync_metrics.average_response_time:.3f}s")
            print(f"P95响应时间: {sync_metrics.p95_response_time:.3f}s")
            
            # 测试异步API
            async_metrics = await tester.test_async_api(
                url="https://httpbin.org/delay/0.1",
                num_requests=50,
                concurrent_requests=10,
                test_name="HTTPBin异步测试"
            )
            
            print("\n异步API测试结果:")
            print(f"成功率: {async_metrics.success_rate:.1%}")
            print(f"平均响应时间: {async_metrics.average_response_time:.3f}s")
            print(f"P95响应时间: {async_metrics.p95_response_time:.3f}s")
            
            # 分析响应时间分布
            analysis = tester.analyze_response_time_distribution(
                sync_metrics,
                output_file="response_time_analysis.json"
            )
            
            print(f"\n性能等级: {analysis['performance_grade']}")
            
        finally:
            tester.close()
    
    # 运行示例
    asyncio.run(main())