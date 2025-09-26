# -*- coding: utf-8 -*-
"""
流式性能测试模块

本模块实现了HarborAI项目的流式性能测试，包括：
- 流式响应延迟测试
- 流式吞吐量测试
- 流式数据完整性测试
- 流式并发性能测试
- 流式错误处理测试
- 流式资源使用测试

作者: HarborAI Team
版本: 1.0.0
创建时间: 2024-01-20
"""

import asyncio
import json
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, AsyncGenerator, Callable, Any, NamedTuple
from unittest.mock import Mock, AsyncMock, patch
import pytest
import statistics
from datetime import datetime, timedelta
import concurrent.futures
import queue
import uuid

from tests.performance import PERFORMANCE_CONFIG, SUPPORTED_VENDORS


class StreamChunk(NamedTuple):
    """
    流式数据块
    
    表示流式响应中的单个数据块
    """
    timestamp: float
    sequence: int
    content: str
    chunk_size: int
    is_final: bool = False
    metadata: Optional[Dict] = None


@dataclass
class StreamingMetrics:
    """
    流式性能指标数据类
    
    记录流式性能测试中的各项指标
    """
    # 基础指标
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # 延迟指标
    first_chunk_latencies: List[float] = field(default_factory=list)
    chunk_intervals: List[float] = field(default_factory=list)
    total_response_times: List[float] = field(default_factory=list)
    
    # 吞吐量指标
    chunks_per_second: List[float] = field(default_factory=list)
    bytes_per_second: List[float] = field(default_factory=list)
    tokens_per_second: List[float] = field(default_factory=list)
    
    # 数据完整性指标
    total_chunks: int = 0
    lost_chunks: int = 0
    duplicate_chunks: int = 0
    out_of_order_chunks: int = 0
    
    # 资源使用指标
    peak_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    connection_count: int = 0
    
    # 错误统计
    error_types: Dict[str, int] = field(default_factory=dict)
    timeout_count: int = 0
    
    def calculate_summary_metrics(self) -> Dict[str, float]:
        """
        计算汇总指标
        
        返回:
            Dict[str, float]: 汇总指标字典
        """
        summary = {}
        
        # 成功率
        if self.total_requests > 0:
            summary['success_rate'] = self.successful_requests / self.total_requests * 100
        else:
            summary['success_rate'] = 0.0
        
        # 延迟统计
        if self.first_chunk_latencies:
            summary['avg_first_chunk_latency'] = statistics.mean(self.first_chunk_latencies)
            summary['p95_first_chunk_latency'] = statistics.quantiles(self.first_chunk_latencies, n=20)[18]  # 95th percentile
            summary['p99_first_chunk_latency'] = statistics.quantiles(self.first_chunk_latencies, n=100)[98]  # 99th percentile
        
        if self.chunk_intervals:
            summary['avg_chunk_interval'] = statistics.mean(self.chunk_intervals)
            summary['max_chunk_interval'] = max(self.chunk_intervals)
        
        if self.total_response_times:
            summary['avg_total_response_time'] = statistics.mean(self.total_response_times)
            summary['p95_total_response_time'] = statistics.quantiles(self.total_response_times, n=20)[18]
        
        # 吞吐量统计
        if self.chunks_per_second:
            summary['avg_chunks_per_second'] = statistics.mean(self.chunks_per_second)
            summary['peak_chunks_per_second'] = max(self.chunks_per_second)
        
        if self.bytes_per_second:
            summary['avg_bytes_per_second'] = statistics.mean(self.bytes_per_second)
            summary['peak_bytes_per_second'] = max(self.bytes_per_second)
        
        if self.tokens_per_second:
            summary['avg_tokens_per_second'] = statistics.mean(self.tokens_per_second)
            summary['peak_tokens_per_second'] = max(self.tokens_per_second)
        
        # 数据完整性
        if self.total_chunks > 0:
            summary['chunk_loss_rate'] = self.lost_chunks / self.total_chunks * 100
            summary['chunk_duplicate_rate'] = self.duplicate_chunks / self.total_chunks * 100
            summary['chunk_disorder_rate'] = self.out_of_order_chunks / self.total_chunks * 100
        
        return summary


class MockStreamingAPI:
    """
    模拟流式API
    
    用于测试的模拟流式API实现
    """
    
    def __init__(self, 
                 chunk_count: int = 10,
                 chunk_size: int = 100,
                 chunk_interval: float = 0.1,
                 error_rate: float = 0.0,
                 timeout_rate: float = 0.0):
        self.chunk_count = chunk_count
        self.chunk_size = chunk_size
        self.chunk_interval = chunk_interval
        self.error_rate = error_rate
        self.timeout_rate = timeout_rate
        self.request_count = 0
    
    async def stream_response(self, request_id: str) -> AsyncGenerator[StreamChunk, None]:
        """
        生成模拟流式响应
        
        参数:
            request_id: 请求ID
        
        生成:
            StreamChunk: 流式数据块
        """
        self.request_count += 1
        
        # 模拟超时
        if self.timeout_rate > 0 and (self.request_count % int(1/self.timeout_rate)) == 0:
            await asyncio.sleep(10)  # 模拟超时
            return
        
        # 模拟错误
        if self.error_rate > 0 and (self.request_count % int(1/self.error_rate)) == 0:
            raise Exception(f"模拟API错误 - 请求ID: {request_id}")
        
        # 生成流式数据块
        for i in range(self.chunk_count):
            await asyncio.sleep(self.chunk_interval)
            
            content = f"数据块 {i+1}/{self.chunk_count} - 请求 {request_id} - " + "x" * self.chunk_size
            
            chunk = StreamChunk(
                timestamp=time.time(),
                sequence=i,
                content=content,
                chunk_size=len(content),
                is_final=(i == self.chunk_count - 1),
                metadata={'request_id': request_id, 'chunk_id': f"{request_id}_{i}"}
            )
            
            yield chunk
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取API统计信息
        
        返回:
            Dict[str, Any]: 统计信息
        """
        return {
            'total_requests': self.request_count,
            'chunk_count': self.chunk_count,
            'chunk_size': self.chunk_size,
            'chunk_interval': self.chunk_interval,
            'error_rate': self.error_rate,
            'timeout_rate': self.timeout_rate
        }


class StreamingPerformanceTest:
    """
    流式性能测试执行器
    
    提供各种流式性能测试方法
    """
    
    def __init__(self):
        self.config = PERFORMANCE_CONFIG['streaming']
        self.metrics = StreamingMetrics()
        self.active_connections = set()
        self._lock = threading.Lock()
    
    async def single_stream_test(self, 
                                api: MockStreamingAPI,
                                request_id: str,
                                timeout: float = 30.0) -> Dict[str, Any]:
        """
        单个流式请求测试
        
        参数:
            api: 模拟API实例
            request_id: 请求ID
            timeout: 超时时间
        
        返回:
            Dict[str, Any]: 测试结果
        """
        start_time = time.time()
        first_chunk_time = None
        chunks_received = []
        last_chunk_time = start_time
        total_bytes = 0
        
        try:
            with self._lock:
                self.active_connections.add(request_id)
                self.metrics.connection_count = len(self.active_connections)
            
            async with asyncio.timeout(timeout):
                async for chunk in api.stream_response(request_id):
                    current_time = time.time()
                    
                    # 记录第一个数据块延迟
                    if first_chunk_time is None:
                        first_chunk_time = current_time
                        first_chunk_latency = first_chunk_time - start_time
                        with self._lock:
                            self.metrics.first_chunk_latencies.append(first_chunk_latency)
                    
                    # 记录数据块间隔
                    if chunks_received:
                        interval = current_time - last_chunk_time
                        with self._lock:
                            self.metrics.chunk_intervals.append(interval)
                    
                    chunks_received.append(chunk)
                    total_bytes += chunk.chunk_size
                    last_chunk_time = current_time
                    
                    with self._lock:
                        self.metrics.total_chunks += 1
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 计算吞吐量指标
            if total_time > 0:
                chunks_per_sec = len(chunks_received) / total_time
                bytes_per_sec = total_bytes / total_time
                # 估算tokens（假设平均每个字符0.75个token）
                estimated_tokens = total_bytes * 0.75
                tokens_per_sec = estimated_tokens / total_time
                
                with self._lock:
                    self.metrics.chunks_per_second.append(chunks_per_sec)
                    self.metrics.bytes_per_second.append(bytes_per_sec)
                    self.metrics.tokens_per_second.append(tokens_per_sec)
                    self.metrics.total_response_times.append(total_time)
                    self.metrics.successful_requests += 1
            
            return {
                'success': True,
                'request_id': request_id,
                'total_time': total_time,
                'first_chunk_latency': first_chunk_latency if first_chunk_time else None,
                'chunks_count': len(chunks_received),
                'total_bytes': total_bytes,
                'chunks_per_second': chunks_per_sec if total_time > 0 else 0,
                'bytes_per_second': bytes_per_sec if total_time > 0 else 0
            }
        
        except asyncio.TimeoutError:
            with self._lock:
                self.metrics.timeout_count += 1
                self.metrics.failed_requests += 1
                self.metrics.error_types['timeout'] = self.metrics.error_types.get('timeout', 0) + 1
            
            return {
                'success': False,
                'request_id': request_id,
                'error': 'timeout',
                'chunks_received': len(chunks_received)
            }
        
        except Exception as e:
            error_type = type(e).__name__
            with self._lock:
                self.metrics.failed_requests += 1
                self.metrics.error_types[error_type] = self.metrics.error_types.get(error_type, 0) + 1
            
            return {
                'success': False,
                'request_id': request_id,
                'error': str(e),
                'error_type': error_type,
                'chunks_received': len(chunks_received)
            }
        
        finally:
            with self._lock:
                self.active_connections.discard(request_id)
                self.metrics.connection_count = len(self.active_connections)
                self.metrics.total_requests += 1
    
    async def concurrent_stream_test(self,
                                   api: MockStreamingAPI,
                                   concurrent_requests: int,
                                   timeout: float = 30.0) -> List[Dict[str, Any]]:
        """
        并发流式请求测试
        
        参数:
            api: 模拟API实例
            concurrent_requests: 并发请求数
            timeout: 超时时间
        
        返回:
            List[Dict[str, Any]]: 测试结果列表
        """
        tasks = []
        
        for i in range(concurrent_requests):
            request_id = f"concurrent_req_{i}_{uuid.uuid4().hex[:8]}"
            task = asyncio.create_task(
                self.single_stream_test(api, request_id, timeout)
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'request_id': f"concurrent_req_{i}",
                    'error': str(result),
                    'error_type': type(result).__name__
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def validate_stream_integrity(self, chunks: List[StreamChunk]) -> Dict[str, Any]:
        """
        验证流式数据完整性
        
        参数:
            chunks: 数据块列表
        
        返回:
            Dict[str, Any]: 验证结果
        """
        if not chunks:
            return {
                'valid': False,
                'issues': ['没有接收到数据块'],
                'lost_chunks': 0,
                'duplicate_chunks': 0,
                'out_of_order_chunks': 0
            }
        
        issues = []
        lost_chunks = 0
        duplicate_chunks = 0
        out_of_order_chunks = 0
        
        # 检查序列号
        sequences = [chunk.sequence for chunk in chunks]
        expected_sequences = list(range(len(chunks)))
        
        # 检查丢失的数据块
        missing_sequences = set(expected_sequences) - set(sequences)
        lost_chunks = len(missing_sequences)
        if missing_sequences:
            issues.append(f"丢失数据块: {sorted(missing_sequences)}")
        
        # 检查重复的数据块
        seen_sequences = set()
        for seq in sequences:
            if seq in seen_sequences:
                duplicate_chunks += 1
            seen_sequences.add(seq)
        
        if duplicate_chunks > 0:
            issues.append(f"重复数据块数量: {duplicate_chunks}")
        
        # 检查顺序
        for i in range(1, len(chunks)):
            if chunks[i].sequence < chunks[i-1].sequence:
                out_of_order_chunks += 1
        
        if out_of_order_chunks > 0:
            issues.append(f"乱序数据块数量: {out_of_order_chunks}")
        
        # 检查最后一个数据块标记
        if chunks and not chunks[-1].is_final:
            issues.append("最后一个数据块未标记为final")
        
        # 更新全局指标
        with self._lock:
            self.metrics.lost_chunks += lost_chunks
            self.metrics.duplicate_chunks += duplicate_chunks
            self.metrics.out_of_order_chunks += out_of_order_chunks
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'lost_chunks': lost_chunks,
            'duplicate_chunks': duplicate_chunks,
            'out_of_order_chunks': out_of_order_chunks,
            'total_chunks': len(chunks)
        }
    
    def reset_metrics(self):
        """重置性能指标"""
        self.metrics = StreamingMetrics()
        self.active_connections.clear()


class TestStreamingPerformance:
    """
    流式性能测试类
    
    包含各种流式性能测试场景
    """
    
    def setup_method(self):
        """测试方法设置"""
        self.streaming_test = StreamingPerformanceTest()
        self.config = PERFORMANCE_CONFIG['streaming']
    
    def teardown_method(self):
        """测试方法清理"""
        self.streaming_test.reset_metrics()
    
    def _print_streaming_summary(self, metrics: StreamingMetrics):
        """打印流式性能测试摘要"""
        summary = metrics.calculate_summary_metrics()
        
        print(f"\n=== 流式性能测试结果 ===")
        print(f"总请求数: {metrics.total_requests}")
        print(f"成功请求数: {metrics.successful_requests}")
        print(f"失败请求数: {metrics.failed_requests}")
        print(f"成功率: {summary.get('success_rate', 0):.1f}%")
        
        print(f"\n延迟指标:")
        if 'avg_first_chunk_latency' in summary:
            print(f"  首块平均延迟: {summary['avg_first_chunk_latency']*1000:.1f}ms")
            print(f"  首块P95延迟: {summary['p95_first_chunk_latency']*1000:.1f}ms")
            print(f"  首块P99延迟: {summary['p99_first_chunk_latency']*1000:.1f}ms")
        
        if 'avg_chunk_interval' in summary:
            print(f"  平均块间隔: {summary['avg_chunk_interval']*1000:.1f}ms")
            print(f"  最大块间隔: {summary['max_chunk_interval']*1000:.1f}ms")
        
        if 'avg_total_response_time' in summary:
            print(f"  平均总响应时间: {summary['avg_total_response_time']:.2f}s")
            print(f"  P95总响应时间: {summary['p95_total_response_time']:.2f}s")
        
        print(f"\n吞吐量指标:")
        if 'avg_chunks_per_second' in summary:
            print(f"  平均块/秒: {summary['avg_chunks_per_second']:.1f}")
            print(f"  峰值块/秒: {summary['peak_chunks_per_second']:.1f}")
        
        if 'avg_bytes_per_second' in summary:
            print(f"  平均字节/秒: {summary['avg_bytes_per_second']:.0f}")
            print(f"  峰值字节/秒: {summary['peak_bytes_per_second']:.0f}")
        
        if 'avg_tokens_per_second' in summary:
            print(f"  平均tokens/秒: {summary['avg_tokens_per_second']:.1f}")
            print(f"  峰值tokens/秒: {summary['peak_tokens_per_second']:.1f}")
        
        print(f"\n数据完整性:")
        print(f"  总数据块: {metrics.total_chunks}")
        if 'chunk_loss_rate' in summary:
            print(f"  丢失率: {summary['chunk_loss_rate']:.2f}%")
            print(f"  重复率: {summary['chunk_duplicate_rate']:.2f}%")
            print(f"  乱序率: {summary['chunk_disorder_rate']:.2f}%")
        
        if metrics.error_types:
            print(f"\n错误统计:")
            for error_type, count in metrics.error_types.items():
                print(f"  {error_type}: {count}")
        
        if metrics.timeout_count > 0:
            print(f"  超时次数: {metrics.timeout_count}")
    
    @pytest.mark.performance
    @pytest.mark.streaming
    @pytest.mark.asyncio
    async def test_single_stream_latency(self):
        """
        单流延迟测试
        
        测试单个流式请求的延迟性能
        """
        # 配置模拟API
        api = MockStreamingAPI(
            chunk_count=20,
            chunk_size=50,
            chunk_interval=0.05  # 50ms间隔
        )
        
        # 执行单个流式请求
        result = await self.streaming_test.single_stream_test(
            api, "single_stream_test", timeout=10.0
        )
        
        metrics = self.streaming_test.metrics
        self._print_streaming_summary(metrics)
        
        # 延迟性能断言
        assert result['success'], f"流式请求失败: {result.get('error', 'Unknown error')}"
        assert result['chunks_count'] == 20
        assert result['first_chunk_latency'] <= self.config['max_first_chunk_latency']
        assert result['total_time'] <= self.config['max_total_response_time']
        
        # 检查指标
        assert len(metrics.first_chunk_latencies) == 1
        assert metrics.first_chunk_latencies[0] <= self.config['max_first_chunk_latency']
        assert len(metrics.chunk_intervals) == 19  # 20个块，19个间隔
    
    @pytest.mark.performance
    @pytest.mark.streaming
    @pytest.mark.asyncio
    async def test_streaming_throughput(self):
        """
        流式吞吐量测试
        
        测试流式响应的吞吐量性能
        """
        # 配置高吞吐量模拟API
        api = MockStreamingAPI(
            chunk_count=50,
            chunk_size=200,
            chunk_interval=0.02  # 20ms间隔，高吞吐量
        )
        
        # 执行流式请求
        result = await self.streaming_test.single_stream_test(
            api, "throughput_test", timeout=15.0
        )
        
        metrics = self.streaming_test.metrics
        self._print_streaming_summary(metrics)
        
        # 吞吐量性能断言
        assert result['success'], f"流式请求失败: {result.get('error', 'Unknown error')}"
        assert result['chunks_per_second'] >= self.config['min_chunks_per_second']
        assert result['bytes_per_second'] >= self.config['min_bytes_per_second']
        
        # 检查吞吐量指标
        summary = metrics.calculate_summary_metrics()
        assert summary['avg_chunks_per_second'] >= self.config['min_chunks_per_second']
        assert summary['avg_bytes_per_second'] >= self.config['min_bytes_per_second']
    
    @pytest.mark.performance
    @pytest.mark.streaming
    @pytest.mark.asyncio
    async def test_concurrent_streaming(self):
        """
        并发流式测试
        
        测试多个并发流式请求的性能
        """
        concurrent_count = 5
        
        # 配置模拟API
        api = MockStreamingAPI(
            chunk_count=15,
            chunk_size=100,
            chunk_interval=0.1
        )
        
        # 执行并发流式请求
        start_time = time.time()
        results = await self.streaming_test.concurrent_stream_test(
            api, concurrent_count, timeout=20.0
        )
        end_time = time.time()
        
        metrics = self.streaming_test.metrics
        self._print_streaming_summary(metrics)
        
        # 并发性能断言
        successful_results = [r for r in results if r.get('success', False)]
        assert len(successful_results) >= concurrent_count * 0.8  # 至少80%成功
        
        # 检查并发性能
        total_time = end_time - start_time
        assert total_time <= self.config['max_concurrent_response_time']
        
        # 检查指标
        assert metrics.total_requests == concurrent_count
        assert metrics.successful_requests >= concurrent_count * 0.8
        
        summary = metrics.calculate_summary_metrics()
        assert summary['success_rate'] >= 80.0
    
    @pytest.mark.performance
    @pytest.mark.streaming
    @pytest.mark.asyncio
    async def test_stream_data_integrity(self):
        """
        流式数据完整性测试
        
        测试流式数据的完整性和正确性
        """
        # 配置模拟API
        api = MockStreamingAPI(
            chunk_count=30,
            chunk_size=80,
            chunk_interval=0.03
        )
        
        # 收集流式数据
        chunks = []
        request_id = "integrity_test"
        
        try:
            async for chunk in api.stream_response(request_id):
                chunks.append(chunk)
        except Exception as e:
            pytest.fail(f"流式数据接收失败: {e}")
        
        # 验证数据完整性
        integrity_result = self.streaming_test.validate_stream_integrity(chunks)
        
        print(f"\n=== 数据完整性验证结果 ===")
        print(f"验证通过: {integrity_result['valid']}")
        print(f"总数据块: {integrity_result['total_chunks']}")
        print(f"丢失数据块: {integrity_result['lost_chunks']}")
        print(f"重复数据块: {integrity_result['duplicate_chunks']}")
        print(f"乱序数据块: {integrity_result['out_of_order_chunks']}")
        
        if integrity_result['issues']:
            print(f"发现问题:")
            for issue in integrity_result['issues']:
                print(f"  - {issue}")
        
        # 数据完整性断言
        assert integrity_result['valid'], f"数据完整性验证失败: {integrity_result['issues']}"
        assert integrity_result['total_chunks'] == 30
        assert integrity_result['lost_chunks'] == 0
        assert integrity_result['duplicate_chunks'] == 0
        assert integrity_result['out_of_order_chunks'] == 0
        
        # 检查数据块内容
        assert len(chunks) == 30
        assert chunks[0].sequence == 0
        assert chunks[-1].sequence == 29
        assert chunks[-1].is_final
    
    @pytest.mark.performance
    @pytest.mark.streaming
    @pytest.mark.asyncio
    async def test_streaming_error_handling(self):
        """
        流式错误处理测试
        
        测试流式请求的错误处理能力
        """
        # 配置有错误的模拟API
        api = MockStreamingAPI(
            chunk_count=20,
            chunk_size=100,
            chunk_interval=0.05,
            error_rate=0.3  # 30%错误率
        )
        
        # 执行多个请求以触发错误
        concurrent_count = 10
        results = await self.streaming_test.concurrent_stream_test(
            api, concurrent_count, timeout=15.0
        )
        
        metrics = self.streaming_test.metrics
        self._print_streaming_summary(metrics)
        
        # 错误处理断言
        failed_results = [r for r in results if not r.get('success', True)]
        assert len(failed_results) > 0  # 应该有一些失败的请求
        
        # 检查错误统计
        assert metrics.failed_requests > 0
        assert len(metrics.error_types) > 0
        
        # 检查错误类型
        for result in failed_results:
            assert 'error' in result
            assert result['error'] is not None
    
    @pytest.mark.performance
    @pytest.mark.streaming
    @pytest.mark.asyncio
    async def test_streaming_timeout_handling(self):
        """
        流式超时处理测试
        
        测试流式请求的超时处理
        """
        # 配置有超时的模拟API
        api = MockStreamingAPI(
            chunk_count=10,
            chunk_size=100,
            chunk_interval=0.1,
            timeout_rate=0.5  # 50%超时率
        )
        
        # 执行请求，设置较短的超时时间
        concurrent_count = 6
        results = await self.streaming_test.concurrent_stream_test(
            api, concurrent_count, timeout=3.0  # 3秒超时
        )
        
        metrics = self.streaming_test.metrics
        self._print_streaming_summary(metrics)
        
        # 超时处理断言
        timeout_results = [r for r in results if r.get('error') == 'timeout']
        assert len(timeout_results) > 0  # 应该有超时的请求
        
        # 检查超时统计
        assert metrics.timeout_count > 0
        assert 'timeout' in metrics.error_types
    
    @pytest.mark.performance
    @pytest.mark.streaming
    @pytest.mark.parametrize("chunk_size", [50, 200, 500, 1000])
    @pytest.mark.asyncio
    async def test_variable_chunk_size_performance(self, chunk_size: int):
        """
        可变数据块大小性能测试
        
        测试不同数据块大小对性能的影响
        
        参数:
            chunk_size: 数据块大小（字节）
        """
        # 配置模拟API
        api = MockStreamingAPI(
            chunk_count=25,
            chunk_size=chunk_size,
            chunk_interval=0.04
        )
        
        # 执行流式请求
        result = await self.streaming_test.single_stream_test(
            api, f"chunk_size_{chunk_size}_test", timeout=15.0
        )
        
        metrics = self.streaming_test.metrics
        
        print(f"\n数据块大小 {chunk_size} 字节的性能:")
        self._print_streaming_summary(metrics)
        
        # 可变数据块大小性能断言
        assert result['success'], f"流式请求失败: {result.get('error', 'Unknown error')}"
        assert result['total_bytes'] == chunk_size * 25
        
        # 根据数据块大小调整性能期望
        if chunk_size <= 100:
            # 小数据块应该有更高的块/秒
            assert result['chunks_per_second'] >= self.config['min_chunks_per_second']
        else:
            # 大数据块应该有更高的字节/秒
            assert result['bytes_per_second'] >= self.config['min_bytes_per_second']
    
    @pytest.mark.performance
    @pytest.mark.streaming
    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_streaming_benchmark(self, benchmark):
        """
        流式性能基准测试
        
        使用pytest-benchmark测试流式性能
        """
        async def streaming_benchmark():
            # 配置标准模拟API
            api = MockStreamingAPI(
                chunk_count=20,
                chunk_size=150,
                chunk_interval=0.05
            )
            
            # 执行并发流式请求
            concurrent_count = 3
            results = await self.streaming_test.concurrent_stream_test(
                api, concurrent_count, timeout=15.0
            )
            
            metrics = self.streaming_test.metrics
            summary = metrics.calculate_summary_metrics()
            
            return {
                'total_requests': metrics.total_requests,
                'successful_requests': metrics.successful_requests,
                'success_rate': summary.get('success_rate', 0),
                'avg_first_chunk_latency': summary.get('avg_first_chunk_latency', 0),
                'avg_chunks_per_second': summary.get('avg_chunks_per_second', 0),
                'avg_bytes_per_second': summary.get('avg_bytes_per_second', 0),
                'total_chunks': metrics.total_chunks
            }
        
        # 运行基准测试
        result = await benchmark(streaming_benchmark)
        
        # 基准测试断言
        assert result['total_requests'] > 0
        assert result['success_rate'] >= 90.0
        assert result['avg_first_chunk_latency'] <= self.config['max_first_chunk_latency']
        assert result['avg_chunks_per_second'] >= self.config['min_chunks_per_second']
        
        print(f"\n流式性能基准结果:")
        print(f"总请求数: {result['total_requests']}")
        print(f"成功率: {result['success_rate']:.1f}%")
        print(f"首块延迟: {result['avg_first_chunk_latency']*1000:.1f}ms")
        print(f"块/秒: {result['avg_chunks_per_second']:.1f}")
        print(f"字节/秒: {result['avg_bytes_per_second']:.0f}")
        print(f"总数据块: {result['total_chunks']}")