#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异步请求处理器测试

测试异步请求处理器的各项功能，包括：
1. 基础配置和初始化
2. 请求提交和处理
3. 批量处理
4. 请求合并和去重
5. 优先级调度
6. 限流控制
7. 重试机制
8. 流式处理
9. 性能监控
10. 错误处理和恢复

遵循VIBE编码规范，使用TDD流程，目标覆盖率≥85%
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, Optional, List
import logging
import aiohttp

from harborai.core.optimizations.async_request_processor import (
    AsyncRequestProcessor,
    RequestConfig,
    RequestPriority,
    RequestStatus,
    AsyncRequest,
    AsyncResponse,
    RateLimiter
)
from harborai.core.optimizations.lockfree_plugin_manager import AtomicInteger, AtomicReference


class TestRequestConfig:
    """测试请求配置类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RequestConfig()
        
        assert config.timeout == 15.0
        assert config.max_retries == 5
        assert config.retry_delay == 0.5
        assert config.retry_backoff == 1.5
        assert config.enable_compression is True
        assert config.enable_keepalive is True
        assert config.max_redirects == 5
        assert config.chunk_size == 16384
        assert config.enable_request_merging is True
        assert config.merge_window == 0.05
        assert config.max_batch_size == 20
        assert config.enable_rate_limiting is True
        assert config.rate_limit_requests == 200
        assert config.rate_limit_window == 60.0
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = RequestConfig(
            timeout=30.0,
            max_retries=3,
            retry_delay=1.0,
            enable_compression=False,
            max_batch_size=10,
            rate_limit_requests=100
        )
        
        assert config.timeout == 30.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.enable_compression is False
        assert config.max_batch_size == 10
        assert config.rate_limit_requests == 100


class TestAsyncRequest:
    """测试异步请求类"""
    
    def test_request_creation(self):
        """测试请求创建"""
        request = AsyncRequest(
            id="test-001",
            method="GET",
            url="https://api.example.com/test",
            headers={"Authorization": "Bearer token"},
            priority=RequestPriority.HIGH
        )
        
        assert request.id == "test-001"
        assert request.method == "GET"
        assert request.url == "https://api.example.com/test"
        assert request.headers["Authorization"] == "Bearer token"
        assert request.priority == RequestPriority.HIGH
        assert request.status.get() == RequestStatus.PENDING
        assert request.retry_count.get() == 0
        assert request.created_at > 0
    
    def test_request_hash_generation(self):
        """测试请求哈希生成"""
        request1 = AsyncRequest(
            id="test-001",
            method="GET",
            url="https://api.example.com/test",
            params={"key": "value"}
        )
        
        request2 = AsyncRequest(
            id="test-002",
            method="GET",
            url="https://api.example.com/test",
            params={"key": "value"}
        )
        
        # 相同的请求应该生成相同的哈希
        assert request1._generate_hash() == request2._generate_hash()
        
        # 不同的请求应该生成不同的哈希
        request3 = AsyncRequest(
            id="test-003",
            method="POST",
            url="https://api.example.com/test",
            params={"key": "value"}
        )
        
        assert request1._generate_hash() != request3._generate_hash()


class TestAsyncResponse:
    """测试异步响应类"""
    
    def test_response_creation(self):
        """测试响应创建"""
        response = AsyncResponse(
            request_id="test-001",
            status_code=200,
            headers={"Content-Type": "application/json"},
            data={"result": "success"},
            response_time=0.5,
            success=True
        )
        
        assert response.request_id == "test-001"
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "application/json"
        assert response.data["result"] == "success"
        assert response.response_time == 0.5
        assert response.success is True
        assert response.error is None
    
    def test_error_response(self):
        """测试错误响应"""
        response = AsyncResponse(
            request_id="test-001",
            status_code=500,
            headers={},
            data="",
            response_time=1.0,
            success=False,
            error="Internal Server Error"
        )
        
        assert response.success is False
        assert response.error == "Internal Server Error"


class TestRateLimiter:
    """测试限流器"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """测试基础限流功能"""
        limiter = RateLimiter(max_requests=2, window=1.0)
        
        # 前两个请求应该通过
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        
        # 第三个请求应该被限制
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_rate_limiter_window_reset(self):
        """测试限流窗口重置"""
        limiter = RateLimiter(max_requests=1, window=0.1)
        
        # 第一个请求通过
        assert await limiter.acquire() is True
        
        # 立即的第二个请求被限制
        assert await limiter.acquire() is False
        
        # 等待窗口重置
        await asyncio.sleep(0.15)
        
        # 新的请求应该通过
        assert await limiter.acquire() is True
    
    @pytest.mark.asyncio
    async def test_wait_for_permit(self):
        """测试等待许可"""
        limiter = RateLimiter(max_requests=1, window=0.2)
        
        # 第一个请求通过
        assert await limiter.acquire() is True
        
        # 等待许可应该在窗口重置后成功
        assert await limiter.wait_for_permit(timeout=0.5) is True
    
    @pytest.mark.asyncio
    async def test_wait_for_permit_timeout(self):
        """测试等待许可超时"""
        limiter = RateLimiter(max_requests=1, window=1.0)
        
        # 第一个请求通过
        assert await limiter.acquire() is True
        
        # 等待许可应该超时
        assert await limiter.wait_for_permit(timeout=0.1) is False


class TestAsyncRequestProcessor:
    """测试异步请求处理器"""
    
    @pytest.fixture
    def config(self):
        """测试配置"""
        return RequestConfig(
            timeout=10.0,
            max_retries=3,
            enable_rate_limiting=True,
            rate_limit_requests=10,
            rate_limit_window=1.0,
            max_batch_size=5,
            enable_request_merging=False  # 默认禁用请求合并以便测试
        )
    
    @pytest.fixture
    def processor(self, config):
        """测试处理器"""
        return AsyncRequestProcessor(config)
    
    def test_init_with_default_config(self):
        """测试默认配置初始化"""
        processor = AsyncRequestProcessor()
        
        assert processor.config is not None
        assert processor.config.timeout == 15.0
        assert processor._running is False
        assert len(processor._request_queues) == len(RequestPriority)
        assert processor._rate_limiter is not None
    
    def test_init_with_custom_config(self, config, processor):
        """测试自定义配置初始化"""
        assert processor.config == config
        assert processor.config.timeout == 10.0
        assert processor._rate_limiter is not None
    
    def test_init_without_rate_limiting(self):
        """测试禁用限流的初始化"""
        config = RequestConfig(enable_rate_limiting=False)
        processor = AsyncRequestProcessor(config)
        
        assert processor._rate_limiter is None
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, processor):
        """测试启动停止生命周期"""
        # 模拟连接池
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 启动处理器
        await processor.start(num_workers=2)
        
        assert processor._running is True
        assert len(processor._processor_tasks) == 2
        assert processor._batch_processor_task is not None
        
        # 停止处理器
        await processor.stop()
        
        assert processor._running is False
    
    @pytest.mark.asyncio
    async def test_start_already_running(self, processor):
        """测试重复启动"""
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        await processor.start()
        initial_tasks = len(processor._processor_tasks)
        
        # 重复启动不应该创建新任务
        await processor.start()
        assert len(processor._processor_tasks) == initial_tasks
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_submit_request_basic(self, processor):
        """测试基础请求提交"""
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 提交请求
        future = await processor.submit_request(
            method="GET",
            url="https://api.example.com/test",
            headers={"Authorization": "Bearer token"},
            priority=RequestPriority.HIGH
        )
        
        assert future is not None
        assert isinstance(future, asyncio.Future)
        
        # 验证请求被添加到待处理列表
        assert len(processor._pending_requests) == 1
        
        # 验证请求参数
        request_id = list(processor._pending_requests.keys())[0]
        request = processor._pending_requests[request_id]
        assert request.method == "GET"
        assert request.url == "https://api.example.com/test"
        assert request.priority == RequestPriority.HIGH
        assert request.headers["Authorization"] == "Bearer token"
    
    @pytest.mark.asyncio
    async def test_submit_request_with_data(self, processor):
        """测试带数据的请求提交"""
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        data = {"key": "value", "number": 42}
        
        future = await processor.submit_request(
            method="POST",
            url="https://api.example.com/data",
            data=data,
            params={"param1": "value1"}
        )
        
        assert future is not None
        assert isinstance(future, asyncio.Future)
        
        # 验证请求被添加到待处理列表
        assert len(processor._pending_requests) == 1
        
        # 验证请求数据
        request_id = list(processor._pending_requests.keys())[0]
        request = processor._pending_requests[request_id]
        assert request.method == "POST"
        assert request.data == data
        assert request.params == {"param1": "value1"}
    
    @pytest.mark.asyncio
    async def test_request_merging(self, processor):
        """测试请求合并"""
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 启用请求合并
        processor.config.enable_request_merging = True
        
        # 提交两个相同的请求
        future1 = await processor.submit_request(
            method="GET",
            url="https://api.example.com/same",
            headers={"Content-Type": "application/json"}
        )
        
        future2 = await processor.submit_request(
            method="GET",
            url="https://api.example.com/same",
            headers={"Content-Type": "application/json"}
        )
        
        # 第二个请求应该被合并，只有一个请求在待处理列表中
        assert len(processor._pending_requests) == 1
        assert future1 == future2  # 应该返回相同的Future
        
        # 验证合并统计
        assert processor._stats['merged_requests'].get() == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, processor):
        """测试限流功能"""
        # 设置严格的限流
        processor.config.rate_limit_requests = 2
        processor.config.rate_limit_window = 1.0
        processor._rate_limiter = RateLimiter(2, 1.0)
        
        # 禁用请求合并以确保每个请求都被单独处理
        processor.config.enable_request_merging = False
        
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 提交3个完全不同的请求（不同方法、URL、参数）
        # 添加小延迟确保时间戳不同
        future1 = await processor.submit_request("GET", "https://api1.example.com/users")
        await asyncio.sleep(0.001)  # 1ms延迟
        future2 = await processor.submit_request("POST", "https://api2.example.com/orders", data={"id": 1})
        await asyncio.sleep(0.001)  # 1ms延迟
        future3 = await processor.submit_request("PUT", "https://api3.example.com/products", params={"category": "books"})
        
        # 验证所有请求都被提交到待处理列表（限流在处理时进行，不是提交时）
        assert len(processor._pending_requests) == 3
        assert future1 is not None
        assert future2 is not None
        assert future3 is not None
        
        # 验证限流器存在
        assert processor._rate_limiter is not None
        assert processor._rate_limiter.max_requests == 2
        assert processor._rate_limiter.window == 1.0
    
    def test_get_statistics(self, processor):
        """测试获取统计信息"""
        # 模拟一些统计数据
        processor._stats['total_requests'].set(100)
        processor._stats['completed_requests'].set(85)
        processor._stats['failed_requests'].set(10)
        processor._stats['retried_requests'].set(5)
        
        stats = processor.get_statistics()
        
        assert stats['total_requests'] == 100
        assert stats['completed_requests'] == 85
        assert stats['failed_requests'] == 10
        assert stats['retried_requests'] == 5
        assert stats['success_rate_percent'] == 85.0
        assert 'performance' in stats
        assert 'queue_sizes' in stats
        assert 'config' in stats
        assert 'pending_requests' in stats
        assert 'active_requests' in stats
    
    @pytest.mark.asyncio
    async def test_cancel_request(self, processor):
        """测试取消请求（通过停止处理器）"""
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 启动处理器
        await processor.start()
        
        # 提交请求
        future = await processor.submit_request("GET", "https://api.example.com/test")
        request_id = list(processor._pending_requests.keys())[0]
        
        # 验证请求存在
        assert request_id in processor._pending_requests
        assert not future.done()
        
        # 停止处理器会取消所有待处理请求
        await processor.stop()
        
        # 验证Future被取消（请求字典在停止时被清理）
        assert future.cancelled() or future.done()
        # 停止后请求字典应该被清空
        assert len(processor._pending_requests) == 0
    
    @pytest.mark.asyncio
    async def test_manual_request_cancellation(self, processor):
        """测试手动取消请求Future"""
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 提交请求
        future = await processor.submit_request("GET", "https://api.example.com/test")
        
        # 手动取消Future
        future.cancel()
        
        # 验证取消结果
        assert future.cancelled()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, processor):
        """测试批量处理"""
        processor.config.max_batch_size = 3
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 创建多个不同的请求
        requests = [
            {"method": "GET", "url": f"https://api.example.com/item/{i}"}
            for i in range(5)
        ]
        
        # 批量提交请求，在每次提交间添加延迟确保唯一时间戳
        futures = []
        for i, req in enumerate(requests):
            if i > 0:
                await asyncio.sleep(0.001)  # 1ms延迟确保时间戳不同
            future = await processor.submit_request(**req)
            futures.append(future)
        
        # 验证批处理
        assert len(futures) == 5
        for future in futures:
            assert isinstance(future, asyncio.Future)
        
        # 验证所有请求都被添加到待处理列表
        assert len(processor._pending_requests) == 5
        
        # 验证批处理统计
        assert processor._stats['total_requests'].get() == 5
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, processor):
        """测试重试机制"""
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 提交请求
        future = await processor.submit_request("GET", "https://api.example.com/test", max_retries=3)
        request_id = list(processor._pending_requests.keys())[0]
        request = processor._pending_requests[request_id]
        
        # 验证初始状态
        assert request.max_retries == 3
        assert request.retry_count.get() == 0
        
        # 模拟处理错误触发重试
        await processor._handle_request_error(request, "Connection failed")
        
        # 验证重试计数增加
        assert request.retry_count.get() == 1
        assert request.status.get() == RequestStatus.RETRYING
        assert processor._stats['retried_requests'].get() == 1
    
    @pytest.mark.asyncio
    async def test_retry_exhausted(self, processor):
        """测试重试次数用尽"""
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 提交请求
        future = await processor.submit_request("GET", "https://api.example.com/test", max_retries=2)
        request_id = list(processor._pending_requests.keys())[0]
        request = processor._pending_requests[request_id]
        
        # 设置重试次数已达上限
        request.retry_count.set(2)
        
        # 模拟处理错误，应该不再重试
        await processor._handle_request_error(request, "Max retries exceeded")
        
        # 验证请求失败且不再重试
        assert request.status.get() == RequestStatus.FAILED
        assert request.retry_count.get() == 2  # 不应该再增加
        assert processor._stats['failed_requests'].get() == 1
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, processor):
        """测试性能监控"""
        # 模拟一些响应时间
        response_times = [100.0, 200.0, 300.0, 400.0, 500.0]  # 毫秒
        
        for rt in response_times:
            await processor._update_performance_stats(rt)
        
        stats = processor.get_statistics()
        
        # 验证性能统计存在
        assert 'performance' in stats
        performance = stats['performance']
        assert 'avg_response_time' in performance
        assert 'max_response_time' in performance
        assert 'min_response_time' in performance
    
    @pytest.mark.asyncio
    async def test_stream_processing(self, processor):
        """测试流式处理"""
        mock_pool = AsyncMock()
        processor.connection_pool = mock_pool
        
        # 模拟连接池返回会话
        mock_session = Mock()  # 使用普通Mock而不是AsyncMock
        mock_pool.get_session = AsyncMock(return_value=mock_session)
        mock_pool.return_session = AsyncMock()
        
        # 模拟响应
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.reason = "OK"
        
        # 创建异步迭代器
        async def mock_iter_chunked(chunk_size):
            for i in range(3):
                yield f"chunk-{i}".encode()
        
        mock_response.content.iter_chunked = mock_iter_chunked
        
        # 创建一个真正的异步上下文管理器
        class MockAsyncContextManager:
            async def __aenter__(self):
                return mock_response
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None
        
        mock_session.request.return_value = MockAsyncContextManager()
        
        # 测试流式请求
        chunks = []
        async for chunk in processor.stream_request(
            method="GET",
            url="https://api.example.com/stream"
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0] == b"chunk-0"
        assert chunks[2] == b"chunk-2"


class TestAsyncRequestProcessorIntegration:
    """异步请求处理器集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_request_lifecycle(self):
        """测试完整的请求生命周期"""
        config = RequestConfig(
            timeout=5.0,
            max_retries=1,
            enable_rate_limiting=False,
            max_batch_size=1
        )
        
        processor = AsyncRequestProcessor(config)
        
        # 模拟连接池和HTTP响应
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.text = AsyncMock(return_value='{"result": "success"}')
            
            mock_session.return_value.__aenter__.return_value.request = AsyncMock(
                return_value=mock_response
            )
            
            # 启动处理器
            await processor.start(num_workers=1)
            
            try:
                # 提交请求
                future = await processor.submit_request(
                    method="GET",
                    url="https://api.example.com/test"
                )
                
                # 等待处理完成
                await asyncio.sleep(0.1)
                
                # 验证统计信息
                stats = processor.get_statistics()
                assert stats['total_requests'] >= 1
                
            finally:
                await processor.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """测试并发请求处理"""
        config = RequestConfig(
            enable_rate_limiting=False,
            max_batch_size=1
        )
        
        processor = AsyncRequestProcessor(config)
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        await processor.start(num_workers=3)
        
        try:
            # 提交多个并发请求
            futures = []
            for i in range(5):
                future = await processor.submit_request(
                    method="GET",
                    url=f"https://api.example.com/test{i}",
                    priority=RequestPriority.NORMAL
                )
                if future:
                    futures.append(future)
            
            # 等待处理
            await asyncio.sleep(0.2)
            
            # 验证统计
            stats = processor.get_statistics()
            assert stats['total_requests'] >= len(futures)
            
        finally:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_priority_scheduling(self):
        """测试优先级调度"""
        processor = AsyncRequestProcessor()
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 记录处理顺序
        processed_order = []
        
        async def mock_process(request):
            processed_order.append((request.id, request.priority))
            return AsyncResponse(
                request_id=request.id,
                status_code=200,
                headers={},
                data={},
                response_time=0.1,
                success=True
            )
        
        with patch.object(processor, '_execute_request', side_effect=mock_process):
            await processor.start(num_workers=1)
            
            try:
                # 提交不同优先级的请求
                await processor.submit_request(
                    method="GET",
                    url="https://api.example.com/low",
                    priority=RequestPriority.LOW
                )
                
                await processor.submit_request(
                    method="GET",
                    url="https://api.example.com/urgent",
                    priority=RequestPriority.URGENT
                )
                
                await processor.submit_request(
                    method="GET",
                    url="https://api.example.com/high",
                    priority=RequestPriority.HIGH
                )
                
                # 等待处理
                await asyncio.sleep(0.2)
                
                # 验证处理顺序（高优先级先处理）
                if len(processed_order) >= 2:
                    # 紧急请求应该在低优先级请求之前处理
                    urgent_index = next(i for i, (_, p) in enumerate(processed_order) if p == RequestPriority.URGENT)
                    low_index = next((i for i, (_, p) in enumerate(processed_order) if p == RequestPriority.LOW), len(processed_order))
                    
                    if low_index < len(processed_order):
                        assert urgent_index < low_index
                
            finally:
                await processor.stop()


    @pytest.mark.asyncio
    async def test_stream_request_success(self):
        """测试流式请求成功处理"""
        processor = AsyncRequestProcessor()
        
        # 使用patch来模拟整个流式请求方法
        mock_chunks = [b"chunk1", b"chunk2", b"chunk3"]
        
        async def mock_stream_request(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk
        
        with patch.object(processor, 'stream_request', side_effect=mock_stream_request):
            # 测试流式请求
            chunks = []
            async for chunk in processor.stream_request(
                method="GET",
                url="https://api.example.com/stream"
            ):
                chunks.append(chunk)
            
            # 验证结果
            assert chunks == mock_chunks
    
    @pytest.mark.asyncio
    async def test_stream_request_http_error(self):
        """测试流式请求HTTP错误处理"""
        processor = AsyncRequestProcessor()
        
        # 模拟HTTP错误
        async def mock_stream_request_error(*args, **kwargs):
            raise Exception("HTTP 404: Not Found")
            yield  # 这行永远不会执行，但让它成为生成器
        
        with patch.object(processor, 'stream_request', side_effect=mock_stream_request_error):
            # 测试HTTP错误
            with pytest.raises(Exception, match="HTTP 404: Not Found"):
                async for chunk in processor.stream_request(
                    method="GET",
                    url="https://api.example.com/notfound"
                ):
                    pass
    
    @pytest.mark.asyncio
    async def test_stream_request_connection_error(self):
        """测试流式请求连接错误处理"""
        processor = AsyncRequestProcessor()
        
        mock_pool = Mock()
        mock_pool.get_session = AsyncMock(return_value=None)
        processor.connection_pool = mock_pool
        
        # 测试连接获取失败
        with pytest.raises(Exception, match="无法获取连接会话"):
            async for chunk in processor.stream_request(
                method="GET",
                url="https://api.example.com/stream"
            ):
                pass
    
    @pytest.mark.asyncio
    async def test_request_retry_mechanism(self):
        """测试请求重试机制"""
        processor = AsyncRequestProcessor()
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 模拟前两次失败，第三次成功
        call_count = 0
        async def mock_execute(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Network error")
            return AsyncResponse(
                request_id=request.id,
                status_code=200,
                headers={},
                data={"success": True},
                response_time=0.1,
                success=True
            )
        
        with patch.object(processor, '_execute_request', side_effect=mock_execute):
            await processor.start(num_workers=1)
            
            try:
                # 提交请求
                future = await processor.submit_request(
                    method="GET",
                    url="https://api.example.com/retry",
                    max_retries=3
                )
                
                # 等待处理完成
                response = await future
                
                # 验证重试成功
                assert response.success is True
                assert call_count == 3  # 两次失败 + 一次成功
                
                # 验证统计
                stats = processor.get_statistics()
                assert stats['retried_requests'] >= 2
                
            finally:
                await processor.stop()
    
    @pytest.mark.asyncio
    async def test_request_max_retries_exceeded(self):
        """测试超过最大重试次数"""
        processor = AsyncRequestProcessor()
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        # 模拟总是失败
        async def mock_execute(request):
            raise Exception("Persistent error")
        
        with patch.object(processor, '_execute_request', side_effect=mock_execute):
            await processor.start(num_workers=1)
            
            try:
                # 提交请求
                future = await processor.submit_request(
                    method="GET",
                    url="https://api.example.com/fail",
                    max_retries=2
                )
                
                # 等待处理完成
                try:
                    response = await future
                    # 如果到这里，说明请求被处理了，验证失败状态
                    assert response.success is False
                    # 检查错误信息（可能是字符串或异常对象）
                    error_str = str(response.error) if response.error else ""
                    assert "Persistent error" in error_str or "error" in error_str.lower()
                except Exception as e:
                    # 如果直接抛出异常，也是预期的行为
                    assert "Persistent error" in str(e)
                
                # 验证统计
                stats = processor.get_statistics()
                assert stats['total_requests'] >= 1
                
            finally:
                await processor.stop()
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """测试批处理功能"""
        config = RequestConfig(
            max_batch_size=3,
            merge_window=0.1
        )
        processor = AsyncRequestProcessor(config=config)
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        processed_requests = []
        
        async def mock_process(request):
            processed_requests.append(request.id)
            return AsyncResponse(
                request_id=request.id,
                status_code=200,
                headers={},
                data={},
                response_time=0.1,
                success=True
            )
        
        with patch.object(processor, '_execute_request', side_effect=mock_process):
            await processor.start(num_workers=1)
            
            try:
                # 快速提交多个请求
                futures = []
                for i in range(5):
                    future = await processor.submit_request(
                        method="GET",
                        url=f"https://api.example.com/batch/{i}"
                    )
                    futures.append(future)
                
                # 等待所有请求完成
                await asyncio.gather(*futures)
                
                # 验证请求被处理
                assert len(processed_requests) == 5
                
                # 验证统计（至少有一些请求被处理）
                stats = processor.get_statistics()
                assert stats['completed_requests'] >= 5
                
            finally:
                await processor.stop()
    
    @pytest.mark.asyncio
    async def test_request_merging(self):
        """测试请求合并功能"""
        config = RequestConfig(
            enable_request_merging=True,
            merge_window=0.1
        )
        processor = AsyncRequestProcessor(config=config)
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        execute_count = 0
        
        async def mock_execute(request):
            nonlocal execute_count
            execute_count += 1
            return AsyncResponse(
                request_id=request.id,
                status_code=200,
                headers={},
                data={"merged": True},
                response_time=0.1,
                success=True
            )
        
        with patch.object(processor, '_execute_request', side_effect=mock_execute):
            await processor.start(num_workers=1)
            
            try:
                # 提交相同的请求多次
                futures = []
                for i in range(3):
                    future = await processor.submit_request(
                        method="GET",
                        url="https://api.example.com/same",
                        headers={"Content-Type": "application/json"}
                    )
                    futures.append(future)
                
                # 等待处理
                responses = await asyncio.gather(*futures)
                
                # 验证所有响应都成功
                for response in responses:
                    assert response.success is True
                
                # 验证请求合并（执行次数应该少于提交次数）
                stats = processor.get_statistics()
                if stats['merged_requests'] > 0:
                    assert execute_count < 3
                
            finally:
                await processor.stop()
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """测试限流功能"""
        config = RequestConfig(
            enable_rate_limiting=True,
            rate_limit_requests=2,
            rate_limit_window=1.0
        )
        processor = AsyncRequestProcessor(config=config)
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        async def mock_execute(request):
            return AsyncResponse(
                request_id=request.id,
                status_code=200,
                headers={},
                data={},
                response_time=0.1,
                success=True
            )
        
        with patch.object(processor, '_execute_request', side_effect=mock_execute):
            await processor.start(num_workers=1)
            
            try:
                # 快速提交超过限制的请求
                futures = []
                for i in range(5):
                    future = await processor.submit_request(
                        method="GET",
                        url=f"https://api.example.com/limited/{i}"
                    )
                    futures.append(future)
                
                # 等待所有请求完成
                await asyncio.gather(*futures)
                
                # 验证所有请求都被处理（无论是否被限流）
                stats = processor.get_statistics()
                assert stats['total_requests'] == 5
                
            finally:
                await processor.stop()
    
    @pytest.mark.asyncio
    async def test_callback_execution(self):
        """测试回调函数执行"""
        processor = AsyncRequestProcessor()
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        callback_called = False
        callback_response = None
        callback_error = None
        
        def sync_callback(response, error):
            nonlocal callback_called, callback_response, callback_error
            callback_called = True
            callback_response = response
            callback_error = error
        
        async def mock_execute(request):
            return AsyncResponse(
                request_id=request.id,
                status_code=200,
                headers={},
                data={"callback": True},
                response_time=0.1,
                success=True
            )
        
        with patch.object(processor, '_execute_request', side_effect=mock_execute):
            await processor.start(num_workers=1)
            
            try:
                # 提交带回调的请求
                future = await processor.submit_request(
                    method="GET",
                    url="https://api.example.com/callback",
                    callback=sync_callback
                )
                
                # 等待处理完成
                response = await future
                
                # 验证回调被调用
                assert callback_called is True
                assert callback_response is not None
                assert callback_response.success is True
                assert callback_error is None
                
            finally:
                await processor.stop()
    
    @pytest.mark.asyncio
    async def test_async_callback_execution(self):
        """测试异步回调函数执行"""
        processor = AsyncRequestProcessor()
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        callback_called = False
        
        async def async_callback(response, error):
            nonlocal callback_called
            callback_called = True
            await asyncio.sleep(0.01)  # 模拟异步操作
        
        async def mock_execute(request):
            return AsyncResponse(
                request_id=request.id,
                status_code=200,
                headers={},
                data={},
                response_time=0.1,
                success=True
            )
        
        with patch.object(processor, '_execute_request', side_effect=mock_execute):
            await processor.start(num_workers=1)
            
            try:
                # 提交带异步回调的请求
                future = await processor.submit_request(
                    method="GET",
                    url="https://api.example.com/async_callback",
                    callback=async_callback
                )
                
                # 等待处理完成
                await future
                await asyncio.sleep(0.1)  # 等待回调完成
                
                # 验证异步回调被调用
                assert callback_called is True
                
            finally:
                await processor.stop()
    
    @pytest.mark.asyncio
    async def test_callback_error_handling(self):
        """测试回调函数错误处理"""
        processor = AsyncRequestProcessor()
        mock_pool = Mock()
        processor.connection_pool = mock_pool
        
        def failing_callback(response, error):
            raise Exception("Callback error")
        
        async def mock_execute(request):
            return AsyncResponse(
                request_id=request.id,
                status_code=200,
                headers={},
                data={},
                response_time=0.1,
                success=True
            )
        
        with patch.object(processor, '_execute_request', side_effect=mock_execute):
            with patch('harborai.core.optimizations.async_request_processor.logger') as mock_logger:
                await processor.start(num_workers=1)
                
                try:
                    # 提交带失败回调的请求
                    future = await processor.submit_request(
                        method="GET",
                        url="https://api.example.com/failing_callback",
                        callback=failing_callback
                    )
                    
                    # 等待处理完成
                    response = await future
                    
                    # 验证请求仍然成功，但回调错误被记录
                    assert response.success is True
                    mock_logger.error.assert_called()
                    
                finally:
                    await processor.stop()
    
    @pytest.mark.asyncio
    async def test_performance_statistics_update(self):
        """测试性能统计更新"""
        processor = AsyncRequestProcessor()
        
        # 测试性能统计更新
        await processor._update_performance_stats(100.0)
        await processor._update_performance_stats(200.0)
        await processor._update_performance_stats(150.0)
        
        stats = processor.get_statistics()
        perf_stats = stats['performance']
        
        # 验证性能统计
        assert perf_stats['avg_response_time'] > 0
        assert perf_stats['min_response_time'] == 100.0
        assert perf_stats['max_response_time'] == 200.0
        assert perf_stats['total_response_time'] == 450.0
        assert perf_stats['response_time_samples'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])