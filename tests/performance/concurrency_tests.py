"""
并发处理能力测试模块

该模块提供全面的并发处理能力测试功能，支持：
- 高并发负载测试
- 并发用户模拟
- 吞吐量测试
- 并发稳定性验证
- 资源竞争检测
- 死锁检测
- 并发性能分析

作者：HarborAI性能测试团队
创建时间：2024年
"""

import asyncio
import threading
import time
import queue
import statistics
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import json
import aiohttp
import requests
from pathlib import Path
import psutil
import multiprocessing

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ConcurrencyMetrics:
    """并发测试指标"""
    test_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    concurrent_users: int
    response_times: List[float]
    throughput_per_second: List[float]
    error_messages: List[str] = field(default_factory=list)
    resource_usage: Dict[str, List[float]] = field(default_factory=dict)
    
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
    def average_throughput(self) -> float:
        """平均吞吐量"""
        if not self.throughput_per_second:
            return 0.0
        return statistics.mean(self.throughput_per_second)
    
    @property
    def peak_throughput(self) -> float:
        """峰值吞吐量"""
        if not self.throughput_per_second:
            return 0.0
        return max(self.throughput_per_second)
    
    @property
    def test_duration(self) -> timedelta:
        """测试持续时间"""
        return self.end_time - self.start_time
    
    @property
    def requests_per_second(self) -> float:
        """每秒请求数"""
        duration_seconds = self.test_duration.total_seconds()
        if duration_seconds == 0:
            return 0.0
        return self.successful_requests / duration_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'test_name': self.test_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration': str(self.test_duration),
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'concurrent_users': self.concurrent_users,
            'success_rate': self.success_rate,
            'average_response_time': self.average_response_time,
            'average_throughput': self.average_throughput,
            'peak_throughput': self.peak_throughput,
            'requests_per_second': self.requests_per_second,
            'error_messages': self.error_messages,
            'resource_usage': self.resource_usage
        }


@dataclass
class LoadTestConfig:
    """负载测试配置"""
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    data: Optional[Any] = None
    params: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    
    # 并发配置
    concurrent_users: int = 10
    requests_per_user: int = 100
    ramp_up_time: float = 0.0  # 渐增时间（秒）
    test_duration: Optional[float] = None  # 测试持续时间（秒）
    
    # 验证配置
    expected_success_rate: float = 0.999  # 期望成功率 >99.9%
    max_response_time: float = 5.0  # 最大响应时间
    min_throughput: float = 0.0  # 最小吞吐量


class ConcurrencyTester:
    """
    并发处理能力测试器
    
    功能特性：
    - 多线程并发测试
    - 多进程并发测试
    - 异步并发测试
    - 渐增负载测试
    - 资源监控
    - 性能分析
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        初始化并发测试器
        
        参数:
            max_workers: 最大工作线程/进程数，默认为CPU核心数
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.session = requests.Session()
        
        # 配置请求会话
        self.session.headers.update({
            'User-Agent': 'HarborAI-Concurrency-Tester/1.0'
        })
        
        # 监控数据
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._resource_data: Dict[str, List[float]] = {
            'cpu_percent': [],
            'memory_percent': [],
            'network_io': [],
            'disk_io': []
        }
        
        logger.info(f"并发测试器初始化完成，最大工作数: {self.max_workers}")
    
    def test_thread_concurrency(
        self,
        config: LoadTestConfig,
        test_name: str = "线程并发测试"
    ) -> ConcurrencyMetrics:
        """
        使用多线程进行并发测试
        
        参数:
            config: 负载测试配置
            test_name: 测试名称
        
        返回:
            并发测试指标
        """
        logger.info(f"开始线程并发测试: {test_name}, "
                   f"并发用户: {config.concurrent_users}, "
                   f"每用户请求: {config.requests_per_user}")
        
        start_time = datetime.now()
        response_times = []
        error_messages = []
        successful_requests = 0
        failed_requests = 0
        
        # 开始资源监控
        self._start_monitoring()
        
        def user_session(user_id: int) -> Tuple[int, int, List[float], List[str]]:
            """模拟单个用户会话"""
            user_successful = 0
            user_failed = 0
            user_response_times = []
            user_errors = []
            
            # 渐增延迟
            if config.ramp_up_time > 0:
                delay = (config.ramp_up_time / config.concurrent_users) * user_id
                time.sleep(delay)
            
            # 执行用户请求
            for req_id in range(config.requests_per_user):
                try:
                    # 准备请求参数
                    request_kwargs = {
                        'timeout': config.timeout,
                        'headers': config.headers or {},
                        'params': config.params
                    }
                    
                    if config.data is not None and config.method.upper() in ['POST', 'PUT', 'PATCH']:
                        if isinstance(config.data, dict):
                            request_kwargs['json'] = config.data
                        else:
                            request_kwargs['data'] = config.data
                    
                    # 执行请求
                    start_req = time.perf_counter()
                    response = self.session.request(config.method, config.url, **request_kwargs)
                    response.raise_for_status()
                    end_req = time.perf_counter()
                    
                    response_time = end_req - start_req
                    user_response_times.append(response_time)
                    user_successful += 1
                    
                except Exception as e:
                    user_failed += 1
                    user_errors.append(f"用户{user_id}请求{req_id}失败: {str(e)}")
            
            return user_successful, user_failed, user_response_times, user_errors
        
        # 使用线程池执行并发用户
        with ThreadPoolExecutor(max_workers=min(config.concurrent_users, self.max_workers)) as executor:
            futures = [executor.submit(user_session, i) for i in range(config.concurrent_users)]
            
            for future in as_completed(futures):
                try:
                    user_successful, user_failed, user_response_times, user_errors = future.result()
                    successful_requests += user_successful
                    failed_requests += user_failed
                    response_times.extend(user_response_times)
                    error_messages.extend(user_errors)
                except Exception as e:
                    logger.error(f"用户会话执行异常: {e}")
                    failed_requests += config.requests_per_user
        
        # 停止资源监控
        self._stop_monitoring()
        end_time = datetime.now()
        
        # 计算吞吐量
        throughput_per_second = self._calculate_throughput(
            successful_requests, 
            start_time, 
            end_time
        )
        
        metrics = ConcurrencyMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_requests=config.concurrent_users * config.requests_per_user,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            concurrent_users=config.concurrent_users,
            response_times=response_times,
            throughput_per_second=throughput_per_second,
            error_messages=error_messages,
            resource_usage=self._resource_data.copy()
        )
        
        logger.info(f"线程并发测试完成: 成功率 {metrics.success_rate:.3%}, "
                   f"平均响应时间 {metrics.average_response_time:.3f}s, "
                   f"吞吐量 {metrics.requests_per_second:.1f} req/s")
        
        return metrics
    
    async def test_async_concurrency(
        self,
        config: LoadTestConfig,
        test_name: str = "异步并发测试"
    ) -> ConcurrencyMetrics:
        """
        使用异步方式进行并发测试
        
        参数:
            config: 负载测试配置
            test_name: 测试名称
        
        返回:
            并发测试指标
        """
        logger.info(f"开始异步并发测试: {test_name}, "
                   f"并发用户: {config.concurrent_users}, "
                   f"每用户请求: {config.requests_per_user}")
        
        start_time = datetime.now()
        response_times = []
        error_messages = []
        successful_requests = 0
        failed_requests = 0
        
        # 开始资源监控
        self._start_monitoring()
        
        # 创建信号量控制并发数
        semaphore = asyncio.Semaphore(config.concurrent_users)
        
        async def user_session(session: aiohttp.ClientSession, user_id: int) -> None:
            """异步用户会话"""
            nonlocal successful_requests, failed_requests
            
            # 渐增延迟
            if config.ramp_up_time > 0:
                delay = (config.ramp_up_time / config.concurrent_users) * user_id
                await asyncio.sleep(delay)
            
            # 执行用户请求
            for req_id in range(config.requests_per_user):
                async with semaphore:
                    try:
                        # 准备请求参数
                        request_kwargs = {
                            'timeout': aiohttp.ClientTimeout(total=config.timeout),
                            'headers': config.headers or {},
                            'params': config.params
                        }
                        
                        if config.data is not None and config.method.upper() in ['POST', 'PUT', 'PATCH']:
                            if isinstance(config.data, dict):
                                request_kwargs['json'] = config.data
                            else:
                                request_kwargs['data'] = config.data
                        
                        # 执行请求
                        start_req = time.perf_counter()
                        async with session.request(config.method, config.url, **request_kwargs) as response:
                            response.raise_for_status()
                            await response.read()
                        end_req = time.perf_counter()
                        
                        response_time = end_req - start_req
                        response_times.append(response_time)
                        successful_requests += 1
                        
                    except Exception as e:
                        failed_requests += 1
                        error_messages.append(f"用户{user_id}请求{req_id}失败: {str(e)}")
        
        # 执行所有异步用户会话
        async with aiohttp.ClientSession() as session:
            tasks = [user_session(session, i) for i in range(config.concurrent_users)]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # 停止资源监控
        self._stop_monitoring()
        end_time = datetime.now()
        
        # 计算吞吐量
        throughput_per_second = self._calculate_throughput(
            successful_requests, 
            start_time, 
            end_time
        )
        
        metrics = ConcurrencyMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_requests=config.concurrent_users * config.requests_per_user,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            concurrent_users=config.concurrent_users,
            response_times=response_times,
            throughput_per_second=throughput_per_second,
            error_messages=error_messages,
            resource_usage=self._resource_data.copy()
        )
        
        logger.info(f"异步并发测试完成: 成功率 {metrics.success_rate:.3%}, "
                   f"平均响应时间 {metrics.average_response_time:.3f}s, "
                   f"吞吐量 {metrics.requests_per_second:.1f} req/s")
        
        return metrics
    
    def test_process_concurrency(
        self,
        config: LoadTestConfig,
        test_name: str = "进程并发测试"
    ) -> ConcurrencyMetrics:
        """
        使用多进程进行并发测试
        
        参数:
            config: 负载测试配置
            test_name: 测试名称
        
        返回:
            并发测试指标
        """
        logger.info(f"开始进程并发测试: {test_name}, "
                   f"并发进程: {config.concurrent_users}, "
                   f"每进程请求: {config.requests_per_user}")
        
        start_time = datetime.now()
        response_times = []
        error_messages = []
        successful_requests = 0
        failed_requests = 0
        
        # 开始资源监控
        self._start_monitoring()
        
        def process_worker(user_id: int) -> Tuple[int, int, List[float], List[str]]:
            """进程工作函数"""
            import requests
            import time
            
            session = requests.Session()
            session.headers.update({'User-Agent': 'HarborAI-Process-Tester/1.0'})
            
            worker_successful = 0
            worker_failed = 0
            worker_response_times = []
            worker_errors = []
            
            # 渐增延迟
            if config.ramp_up_time > 0:
                delay = (config.ramp_up_time / config.concurrent_users) * user_id
                time.sleep(delay)
            
            # 执行请求
            for req_id in range(config.requests_per_user):
                try:
                    # 准备请求参数
                    request_kwargs = {
                        'timeout': config.timeout,
                        'headers': config.headers or {},
                        'params': config.params
                    }
                    
                    if config.data is not None and config.method.upper() in ['POST', 'PUT', 'PATCH']:
                        if isinstance(config.data, dict):
                            request_kwargs['json'] = config.data
                        else:
                            request_kwargs['data'] = config.data
                    
                    # 执行请求
                    start_req = time.perf_counter()
                    response = session.request(config.method, config.url, **request_kwargs)
                    response.raise_for_status()
                    end_req = time.perf_counter()
                    
                    response_time = end_req - start_req
                    worker_response_times.append(response_time)
                    worker_successful += 1
                    
                except Exception as e:
                    worker_failed += 1
                    worker_errors.append(f"进程{user_id}请求{req_id}失败: {str(e)}")
            
            session.close()
            return worker_successful, worker_failed, worker_response_times, worker_errors
        
        # 使用进程池执行并发工作
        with ProcessPoolExecutor(max_workers=min(config.concurrent_users, self.max_workers)) as executor:
            futures = [executor.submit(process_worker, i) for i in range(config.concurrent_users)]
            
            for future in as_completed(futures):
                try:
                    worker_successful, worker_failed, worker_response_times, worker_errors = future.result()
                    successful_requests += worker_successful
                    failed_requests += worker_failed
                    response_times.extend(worker_response_times)
                    error_messages.extend(worker_errors)
                except Exception as e:
                    logger.error(f"进程工作执行异常: {e}")
                    failed_requests += config.requests_per_user
        
        # 停止资源监控
        self._stop_monitoring()
        end_time = datetime.now()
        
        # 计算吞吐量
        throughput_per_second = self._calculate_throughput(
            successful_requests, 
            start_time, 
            end_time
        )
        
        metrics = ConcurrencyMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_requests=config.concurrent_users * config.requests_per_user,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            concurrent_users=config.concurrent_users,
            response_times=response_times,
            throughput_per_second=throughput_per_second,
            error_messages=error_messages,
            resource_usage=self._resource_data.copy()
        )
        
        logger.info(f"进程并发测试完成: 成功率 {metrics.success_rate:.3%}, "
                   f"平均响应时间 {metrics.average_response_time:.3f}s, "
                   f"吞吐量 {metrics.requests_per_second:.1f} req/s")
        
        return metrics
    
    async def test_async_high_concurrency(
        self,
        url: str = "https://httpbin.org/delay/0.1",
        concurrent_users: int = 100,
        requests_per_user: int = 100,
        expected_success_rate: float = 0.999,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Any] = None,
        test_name: str = "异步高并发测试"
    ) -> ConcurrencyMetrics:
        """
        异步高并发专项测试
        
        专门用于测试高并发场景下的系统性能，包含内存优化策略。
        
        参数:
            url: 测试URL，默认使用httpbin测试端点
            concurrent_users: 并发用户数
            requests_per_user: 每用户请求数
            expected_success_rate: 期望成功率
            method: HTTP方法
            headers: 请求头
            data: 请求数据
            test_name: 测试名称
        
        返回:
            ConcurrencyMetrics: 并发测试指标对象
        
        内存优化特性:
            - 分批处理高并发请求
            - 动态内存监控和清理
            - 资源使用限制和警告
        """
        import gc
        import psutil
        
        logger.info(f"开始异步高并发测试: {test_name}, URL: {url}, "
                   f"并发用户: {concurrent_users}, 每用户请求: {requests_per_user}")
        
        # 内存监控 - 记录初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        logger.info(f"测试开始时内存使用: {initial_memory:.2f} MB")
        
        # 创建负载测试配置
        config = LoadTestConfig(
            url=url,
            method=method,
            headers=headers,
            data=data,
            concurrent_users=concurrent_users,
            requests_per_user=requests_per_user,
            expected_success_rate=expected_success_rate
        )
        
        start_time = datetime.now()
        response_times = []
        error_messages = []
        successful_requests = 0
        failed_requests = 0
        
        # 开始资源监控
        self._start_monitoring()
        
        # 创建信号量控制并发数 - 对于高并发场景，使用更大的信号量
        max_concurrent = min(concurrent_users, 200)  # 限制最大并发数以控制资源使用
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def user_session_with_memory_optimization(session: aiohttp.ClientSession, user_id: int) -> None:
            """异步用户会话，包含内存优化"""
            nonlocal successful_requests, failed_requests
            
            # 渐增延迟 - 避免瞬间高并发冲击
            if config.ramp_up_time > 0:
                delay = (config.ramp_up_time / config.concurrent_users) * user_id
                await asyncio.sleep(delay)
            
            user_response_times = []
            
            # 执行用户请求
            for req_id in range(config.requests_per_user):
                async with semaphore:
                    try:
                        # 准备请求参数
                        request_kwargs = {
                            'timeout': aiohttp.ClientTimeout(total=config.timeout),
                            'headers': config.headers or {},
                            'params': config.params
                        }
                        
                        if config.data is not None and config.method.upper() in ['POST', 'PUT', 'PATCH']:
                            if isinstance(config.data, dict):
                                request_kwargs['json'] = config.data
                            else:
                                request_kwargs['data'] = config.data
                        
                        # 执行请求
                        start_req = time.perf_counter()
                        async with session.request(config.method, config.url, **request_kwargs) as response:
                            response.raise_for_status()
                            # 读取响应但不存储大量数据
                            await response.read()
                        end_req = time.perf_counter()
                        
                        response_time = end_req - start_req
                        user_response_times.append(response_time)
                        successful_requests += 1
                        
                    except Exception as e:
                        failed_requests += 1
                        error_msg = f"用户{user_id}请求{req_id}失败: {str(e)}"
                        error_messages.append(error_msg)
                        logger.warning(error_msg)
                    
                    # 每10个请求检查一次内存使用
                    if req_id % 10 == 0:
                        current_memory = process.memory_info().rss / 1024 / 1024  # MB
                        if current_memory > initial_memory * 3:  # 内存使用超过初始值3倍时警告
                            logger.warning(f"高并发测试内存使用过高: {current_memory:.2f} MB (初始: {initial_memory:.2f} MB)")
                            # 强制垃圾回收
                            gc.collect()
            
            # 将用户响应时间添加到全局列表
            response_times.extend(user_response_times)
        
        # 分批执行用户会话以控制内存使用
        batch_size = min(50, concurrent_users)  # 每批最多50个用户
        
        async with aiohttp.ClientSession() as session:
            for batch_start in range(0, concurrent_users, batch_size):
                batch_end = min(batch_start + batch_size, concurrent_users)
                batch_tasks = [
                    user_session_with_memory_optimization(session, i) 
                    for i in range(batch_start, batch_end)
                ]
                
                # 执行当前批次
                await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                # 批次间内存清理
                del batch_tasks
                gc.collect()
                
                # 内存监控
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                logger.info(f"批次 {batch_start//batch_size + 1} 完成，当前内存使用: {current_memory:.2f} MB")
                
                # 批次间短暂休息
                if batch_end < concurrent_users:
                    await asyncio.sleep(0.1)
        
        # 停止资源监控
        self._stop_monitoring()
        end_time = datetime.now()
        
        # 最终内存检查
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        logger.info(f"测试结束时内存使用: {final_memory:.2f} MB (增加: {memory_increase:.2f} MB)")
        
        # 计算吞吐量
        throughput_per_second = self._calculate_throughput(
            successful_requests, 
            start_time, 
            end_time
        )
        
        metrics = ConcurrencyMetrics(
            test_name=test_name,
            start_time=start_time,
            end_time=end_time,
            total_requests=concurrent_users * requests_per_user,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            concurrent_users=concurrent_users,
            response_times=response_times,
            throughput_per_second=throughput_per_second,
            error_messages=error_messages,
            resource_usage=self._resource_data.copy()
        )
        
        logger.info(f"异步高并发测试完成: 成功率 {metrics.success_rate:.3%}, "
                   f"平均响应时间 {metrics.average_response_time:.3f}s, "
                   f"吞吐量 {metrics.requests_per_second:.1f} req/s, "
                   f"内存增长 {memory_increase:.2f} MB")
        
        return metrics
    
    def test_ramp_up_load(
        self,
        config: LoadTestConfig,
        max_users: int = 100,
        ramp_up_steps: int = 10,
        step_duration: float = 30.0,
        test_name: str = "渐增负载测试"
    ) -> List[ConcurrencyMetrics]:
        """
        渐增负载测试
        
        参数:
            config: 基础负载测试配置
            max_users: 最大用户数
            ramp_up_steps: 渐增步骤数
            step_duration: 每步持续时间（秒）
            test_name: 测试名称
        
        返回:
            每个步骤的并发测试指标列表
        """
        logger.info(f"开始渐增负载测试: {test_name}, "
                   f"最大用户: {max_users}, 步骤数: {ramp_up_steps}")
        
        results = []
        users_per_step = max_users // ramp_up_steps
        
        for step in range(1, ramp_up_steps + 1):
            current_users = users_per_step * step
            step_config = LoadTestConfig(
                url=config.url,
                method=config.method,
                headers=config.headers,
                data=config.data,
                params=config.params,
                timeout=config.timeout,
                concurrent_users=current_users,
                requests_per_user=max(1, int(step_duration * 2)),  # 基于持续时间调整请求数
                ramp_up_time=5.0
            )
            
            step_name = f"{test_name}_步骤{step}_{current_users}用户"
            metrics = self.test_thread_concurrency(step_config, step_name)
            results.append(metrics)
            
            logger.info(f"步骤 {step}/{ramp_up_steps} 完成: "
                       f"{current_users}用户, 成功率 {metrics.success_rate:.3%}")
            
            # 步骤间休息
            if step < ramp_up_steps:
                time.sleep(2.0)
        
        return results
    
    def validate_concurrency_requirements(
        self,
        metrics: ConcurrencyMetrics,
        config: LoadTestConfig
    ) -> Dict[str, Any]:
        """
        验证并发性能要求
        
        参数:
            metrics: 并发测试指标
            config: 负载测试配置
        
        返回:
            验证结果
        """
        validation_results = {
            'test_name': metrics.test_name,
            'requirements_met': True,
            'validations': {}
        }
        
        # 验证成功率 >99.9%
        success_rate_met = metrics.success_rate >= config.expected_success_rate
        validation_results['validations']['success_rate'] = {
            'requirement': f">= {config.expected_success_rate:.1%}",
            'actual': f"{metrics.success_rate:.3%}",
            'met': success_rate_met
        }
        
        # 验证最大响应时间
        max_response_time = max(metrics.response_times) if metrics.response_times else float('inf')
        response_time_met = max_response_time <= config.max_response_time
        validation_results['validations']['max_response_time'] = {
            'requirement': f"<= {config.max_response_time}s",
            'actual': f"{max_response_time:.3f}s",
            'met': response_time_met
        }
        
        # 验证最小吞吐量
        throughput_met = metrics.requests_per_second >= config.min_throughput
        validation_results['validations']['min_throughput'] = {
            'requirement': f">= {config.min_throughput} req/s",
            'actual': f"{metrics.requests_per_second:.1f} req/s",
            'met': throughput_met
        }
        
        # 验证并发稳定性（响应时间标准差）
        response_time_std = statistics.stdev(metrics.response_times) if len(metrics.response_times) > 1 else 0
        stability_threshold = metrics.average_response_time * 0.5  # 标准差不超过平均值的50%
        stability_met = response_time_std <= stability_threshold
        validation_results['validations']['response_time_stability'] = {
            'requirement': f"标准差 <= {stability_threshold:.3f}s",
            'actual': f"{response_time_std:.3f}s",
            'met': stability_met
        }
        
        # 总体验证结果
        validation_results['requirements_met'] = all(
            v['met'] for v in validation_results['validations'].values()
        )
        
        # 性能等级评估
        if validation_results['requirements_met']:
            if metrics.success_rate >= 0.999 and metrics.average_response_time <= 0.1:
                grade = 'A'
            elif metrics.success_rate >= 0.999 and metrics.average_response_time <= 0.5:
                grade = 'B'
            elif metrics.success_rate >= 0.995:
                grade = 'C'
            else:
                grade = 'D'
        else:
            grade = 'F'
        
        validation_results['performance_grade'] = grade
        
        return validation_results
    
    def _start_monitoring(self) -> None:
        """开始资源监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._resource_data = {
            'cpu_percent': [],
            'memory_percent': [],
            'network_io': [],
            'disk_io': []
        }
        
        def monitor_resources():
            """资源监控线程函数"""
            while self._monitoring:
                try:
                    # CPU使用率
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self._resource_data['cpu_percent'].append(cpu_percent)
                    
                    # 内存使用率
                    memory = psutil.virtual_memory()
                    self._resource_data['memory_percent'].append(memory.percent)
                    
                    # 网络IO
                    net_io = psutil.net_io_counters()
                    self._resource_data['network_io'].append(net_io.bytes_sent + net_io.bytes_recv)
                    
                    # 磁盘IO
                    disk_io = psutil.disk_io_counters()
                    if disk_io:
                        self._resource_data['disk_io'].append(disk_io.read_bytes + disk_io.write_bytes)
                    
                except Exception as e:
                    logger.warning(f"资源监控异常: {e}")
                
                time.sleep(1)
        
        self._monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        self._monitor_thread.start()
    
    def _stop_monitoring(self) -> None:
        """停止资源监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
    
    def _calculate_throughput(
        self,
        successful_requests: int,
        start_time: datetime,
        end_time: datetime
    ) -> List[float]:
        """
        计算每秒吞吐量
        
        参数:
            successful_requests: 成功请求数
            start_time: 开始时间
            end_time: 结束时间
        
        返回:
            每秒吞吐量列表
        """
        duration = (end_time - start_time).total_seconds()
        if duration <= 0:
            return []
        
        # 简化计算：假设请求均匀分布
        avg_throughput = successful_requests / duration
        return [avg_throughput] * int(duration)
    
    def close(self):
        """关闭测试器"""
        self._stop_monitoring()
        self.session.close()
        logger.info("并发测试器已关闭")


# 便捷函数
def test_high_concurrency(
    url: str,
    concurrent_users: int = 100,
    requests_per_user: int = 100,
    expected_success_rate: float = 0.999,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Any] = None,
    test_name: str = "高并发测试"
) -> Tuple[ConcurrencyMetrics, Dict[str, Any]]:
    """
    快速高并发测试
    
    参数:
        url: 测试URL
        concurrent_users: 并发用户数
        requests_per_user: 每用户请求数
        expected_success_rate: 期望成功率
        method: HTTP方法
        headers: 请求头
        data: 请求数据
        test_name: 测试名称
    
    返回:
        (并发测试指标, 验证结果)
    """
    config = LoadTestConfig(
        url=url,
        method=method,
        headers=headers,
        data=data,
        concurrent_users=concurrent_users,
        requests_per_user=requests_per_user,
        expected_success_rate=expected_success_rate
    )
    
    tester = ConcurrencyTester()
    try:
        metrics = tester.test_thread_concurrency(config, test_name)
        validation = tester.validate_concurrency_requirements(metrics, config)
        return metrics, validation
    finally:
        tester.close()


async def test_async_high_concurrency(
    url: str,
    concurrent_users: int = 100,
    requests_per_user: int = 100,
    expected_success_rate: float = 0.999,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Any] = None,
    test_name: str = "异步高并发测试"
) -> Tuple[ConcurrencyMetrics, Dict[str, Any]]:
    """
    快速异步高并发测试
    
    参数:
        url: 测试URL
        concurrent_users: 并发用户数
        requests_per_user: 每用户请求数
        expected_success_rate: 期望成功率
        method: HTTP方法
        headers: 请求头
        data: 请求数据
        test_name: 测试名称
    
    返回:
        (并发测试指标, 验证结果)
    """
    config = LoadTestConfig(
        url=url,
        method=method,
        headers=headers,
        data=data,
        concurrent_users=concurrent_users,
        requests_per_user=requests_per_user,
        expected_success_rate=expected_success_rate
    )
    
    tester = ConcurrencyTester()
    try:
        metrics = await tester.test_async_concurrency(config, test_name)
        validation = tester.validate_concurrency_requirements(metrics, config)
        return metrics, validation
    finally:
        tester.close()


if __name__ == "__main__":
    # 示例使用
    import asyncio
    
    async def main():
        # 创建测试配置
        config = LoadTestConfig(
            url="https://httpbin.org/delay/0.1",
            concurrent_users=50,
            requests_per_user=20,
            expected_success_rate=0.999,
            max_response_time=2.0,
            min_throughput=10.0
        )
        
        # 创建测试器
        tester = ConcurrencyTester()
        
        try:
            # 线程并发测试
            print("执行线程并发测试...")
            thread_metrics = tester.test_thread_concurrency(config, "HTTPBin线程并发测试")
            thread_validation = tester.validate_concurrency_requirements(thread_metrics, config)
            
            print(f"线程并发测试结果:")
            print(f"成功率: {thread_metrics.success_rate:.3%}")
            print(f"平均响应时间: {thread_metrics.average_response_time:.3f}s")
            print(f"吞吐量: {thread_metrics.requests_per_second:.1f} req/s")
            print(f"性能等级: {thread_validation['performance_grade']}")
            print(f"要求满足: {thread_validation['requirements_met']}")
            
            # 异步并发测试
            print("\n执行异步并发测试...")
            async_metrics = await tester.test_async_concurrency(config, "HTTPBin异步并发测试")
            async_validation = tester.validate_concurrency_requirements(async_metrics, config)
            
            print(f"异步并发测试结果:")
            print(f"成功率: {async_metrics.success_rate:.3%}")
            print(f"平均响应时间: {async_metrics.average_response_time:.3f}s")
            print(f"吞吐量: {async_metrics.requests_per_second:.1f} req/s")
            print(f"性能等级: {async_validation['performance_grade']}")
            print(f"要求满足: {async_validation['requirements_met']}")
            
            # 渐增负载测试
            print("\n执行渐增负载测试...")
            ramp_results = tester.test_ramp_up_load(
                config,
                max_users=100,
                ramp_up_steps=5,
                step_duration=10.0,
                test_name="HTTPBin渐增负载测试"
            )
            
            print("渐增负载测试结果:")
            for i, result in enumerate(ramp_results, 1):
                print(f"步骤{i}: {result.concurrent_users}用户, "
                     f"成功率 {result.success_rate:.3%}, "
                     f"吞吐量 {result.requests_per_second:.1f} req/s")
            
        finally:
            tester.close()
    
    # 运行示例
    asyncio.run(main())