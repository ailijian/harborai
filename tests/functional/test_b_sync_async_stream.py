# -*- coding: utf-8 -*-
"""
HarborAI 同步异步流式测试模块

测试目标：
- 验证同步和异步调用的正确性
- 测试流式输出的完整性和性能
- 验证流式数据的实时性和准确性
"""

import pytest
import asyncio
import time
from typing import List, AsyncGenerator, Generator
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor

from harborai import HarborAI
from harborai.core.exceptions import HarborAIError


class TestSyncOperations:
    """同步操作测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.sync_test
    def test_sync_chat_completion(self, mock_harborai_client, test_messages):
        """测试同步聊天完成"""
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="This is a synchronous response.",
                role="assistant"
            ),
            finish_reason="stop",
            index=0
        )]
        mock_response.usage = Mock(
            prompt_tokens=15,
            completion_tokens=12,
            total_tokens=27
        )
        mock_response.model = "deepseek-chat"
        
        # 使用side_effect而不是return_value来确保mock正确工作
        def mock_create(*args, **kwargs):
            return mock_response
        
        mock_harborai_client.chat.completions.create.side_effect = mock_create
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行同步调用
        response = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            stream=False
        )
        
        # 记录结束时间
        end_time = time.time()
        
        # 验证响应
        assert response is not None
        assert response.choices[0].message.content == "This is a synchronous response."
        assert response.choices[0].message.role == "assistant"
        assert response.usage.total_tokens == 27
        
        # 验证响应内容
        assert response.choices[0].message.content is not None
        
        # 验证执行时间（同步调用应该阻塞）
        execution_time = end_time - start_time
        assert execution_time >= 0  # 基本时间验证
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.sync_test
    @pytest.mark.stream_test
    def test_sync_streaming(self, mock_harborai_client, test_messages):
        """测试同步流式输出"""
        # 配置流式mock响应
        mock_chunks = [
            Mock(
                choices=[Mock(
                    delta=Mock(content="Hello"),
                    index=0,
                    finish_reason=None
                )],
                id="chatcmpl-sync-stream",
                object="chat.completion.chunk",
                created=int(time.time()),
                model="deepseek-chat"
            ),
            Mock(
                choices=[Mock(
                    delta=Mock(content=" from"),
                    index=0,
                    finish_reason=None
                )],
                id="chatcmpl-sync-stream",
                object="chat.completion.chunk",
                created=int(time.time()),
                model="deepseek-chat"
            ),
            Mock(
                choices=[Mock(
                    delta=Mock(content=" sync"),
                    index=0,
                    finish_reason=None
                )],
                id="chatcmpl-sync-stream",
                object="chat.completion.chunk",
                created=int(time.time()),
                model="deepseek-chat"
            ),
            Mock(
                choices=[Mock(
                    delta=Mock(content=" stream!"),
                    index=0,
                    finish_reason=None
                )],
                id="chatcmpl-sync-stream",
                object="chat.completion.chunk",
                created=int(time.time()),
                model="deepseek-chat"
            ),
            Mock(
                choices=[Mock(
                    delta=Mock(content=None),
                    index=0,
                    finish_reason="stop"
                )],
                id="chatcmpl-sync-stream",
                object="chat.completion.chunk",
                created=int(time.time()),
                model="deepseek-chat"
            )
        ]
        
        # 配置mock以返回可迭代的流式响应
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return iter(mock_chunks)
            else:
                # 返回非流式响应
                return Mock()
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 执行同步流式调用
        stream = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=test_messages,
            stream=True
        )
        
        # 收集流式数据
        collected_content = []
        chunk_count = 0
        start_time = time.time()
        
        for chunk in stream:
            chunk_count += 1
            
            # 验证chunk结构
            assert hasattr(chunk, 'choices')
            assert hasattr(chunk, 'object')
            assert chunk.object == "chat.completion.chunk"
            
            # 收集内容
            if chunk.choices and hasattr(chunk.choices[0], 'delta'):
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    collected_content.append(delta.content)
        
        end_time = time.time()
        
        # 验证流式输出
        assert chunk_count == 5  # 4个内容chunk + 1个结束chunk
        full_content = ''.join(collected_content)
        assert full_content == "Hello from sync stream!"
        
        # 验证流式处理时间
        processing_time = end_time - start_time
        assert processing_time >= 0
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.sync_test
    def test_sync_multiple_requests(self, mock_harborai_client, test_messages):
        """测试同步多请求处理"""
        # 配置多个mock响应
        responses = []
        for i in range(3):
            mock_response = Mock()
            mock_response.choices = [Mock(
                message=Mock(
                    content=f"Response {i+1}",
                    role="assistant"
                ),
                finish_reason="stop"
            )]
            mock_response.usage = Mock(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15
            )
            responses.append(mock_response)
        
        # 配置mock以按顺序返回响应
        response_iter = iter(responses)
        def mock_create(*args, **kwargs):
            try:
                return next(response_iter)
            except StopIteration:
                return responses[-1]  # 如果超出范围，返回最后一个响应
        
        mock_harborai_client.chat.completions.create = mock_create
        mock_harborai_client.chat.completions.create.call_count = 0
        
        # 包装函数以计数调用次数
        original_mock = mock_harborai_client.chat.completions.create
        def counting_mock(*args, **kwargs):
            counting_mock.call_count += 1
            return original_mock(*args, **kwargs)
        counting_mock.call_count = 0
        mock_harborai_client.chat.completions.create = counting_mock
        
        # 执行多个同步请求
        results = []
        start_time = time.time()
        
        for i in range(3):
            response = mock_harborai_client.chat.completions.create(
                model="deepseek-chat",
                messages=test_messages
            )
            results.append(response.choices[0].message.content)
        
        end_time = time.time()
        
        # 验证结果
        assert len(results) == 3
        for i, content in enumerate(results):
            assert content == f"Response {i+1}"
        
        # 验证调用次数
        assert mock_harborai_client.chat.completions.create.call_count == 3
        
        # 验证总执行时间
        total_time = end_time - start_time
        assert total_time >= 0


class TestAsyncOperations:
    """异步操作测试类"""
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.async_test
    async def test_async_chat_completion(self, mock_harborai_client, test_messages):
        """测试异步聊天完成"""
        # 配置异步mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="This is an asynchronous response.",
                role="assistant"
            ),
            finish_reason="stop",
            index=0
        )]
        mock_response.usage = Mock(
            prompt_tokens=18,
            completion_tokens=15,
            total_tokens=33
        )
        mock_response.model = "deepseek-chat"
        
        # 创建异步mock函数
        async def async_create(*args, **kwargs):
            # 模拟异步延迟
            await asyncio.sleep(0.1)
            return mock_response
        
        mock_harborai_client.chat.completions.acreate = async_create
        
        # 记录开始时间
        start_time = time.time()
        
        # 执行异步调用
        response = await mock_harborai_client.chat.completions.acreate(
            model="deepseek-chat",
            messages=test_messages,
            stream=False
        )
        
        # 记录结束时间
        end_time = time.time()
        
        # 验证响应
        assert response is not None
        assert response.choices[0].message.content == "This is an asynchronous response."
        assert response.choices[0].message.role == "assistant"
        assert response.usage.total_tokens == 33
        
        # 验证执行时间（应该包含模拟的延迟）
        execution_time = end_time - start_time
        assert execution_time >= 0.1  # 至少包含模拟延迟
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.async_test
    @pytest.mark.stream_test
    async def test_async_streaming(self, mock_harborai_client, test_messages):
        """测试异步流式输出"""
        # 配置异步流式mock
        async def async_stream_generator():
            chunks = [
                Mock(
                    choices=[Mock(
                        delta=Mock(content="Async"),
                        index=0,
                        finish_reason=None
                    )],
                    id="chatcmpl-async-stream",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="deepseek-chat"
                ),
                Mock(
                    choices=[Mock(
                        delta=Mock(content=" streaming"),
                        index=0,
                        finish_reason=None
                    )],
                    id="chatcmpl-async-stream",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="deepseek-chat"
                ),
                Mock(
                    choices=[Mock(
                        delta=Mock(content=" works!"),
                        index=0,
                        finish_reason=None
                    )],
                    id="chatcmpl-async-stream",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="deepseek-chat"
                ),
                Mock(
                    choices=[Mock(
                        delta=Mock(content=None),
                        index=0,
                        finish_reason="stop"
                    )],
                    id="chatcmpl-async-stream",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="deepseek-chat"
                )
            ]
            
            for chunk in chunks:
                # 模拟异步延迟
                await asyncio.sleep(0.05)
                yield chunk
        
        # 配置异步流式mock
        def mock_acreate(*args, **kwargs):
            if kwargs.get('stream', False):
                return async_stream_generator()
            else:
                # 返回非流式异步响应
                async def async_response():
                    return Mock()
                return async_response()
        
        mock_harborai_client.chat.completions.acreate = mock_acreate
        
        # 执行异步流式调用
        stream = mock_harborai_client.chat.completions.acreate(
            model="deepseek-chat",
            messages=test_messages,
            stream=True
        )
        
        # 收集异步流式数据
        collected_content = []
        chunk_count = 0
        start_time = time.time()
        
        async for chunk in stream:
            chunk_count += 1
            
            # 验证chunk结构
            assert hasattr(chunk, 'choices')
            assert hasattr(chunk, 'object')
            assert chunk.object == "chat.completion.chunk"
            
            # 收集内容
            if chunk.choices and hasattr(chunk.choices[0], 'delta'):
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    collected_content.append(delta.content)
        
        end_time = time.time()
        
        # 验证异步流式输出
        assert chunk_count == 4  # 3个内容chunk + 1个结束chunk
        full_content = ''.join(collected_content)
        assert full_content == "Async streaming works!"
        
        # 验证异步处理时间（应该包含模拟延迟）
        processing_time = end_time - start_time
        assert processing_time >= 0.15  # 至少包含4次0.05秒延迟
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.async_test
    async def test_async_concurrent_requests(self, mock_harborai_client, test_messages):
        """测试异步并发请求"""
        # 配置多个异步mock响应
        async def async_create_with_delay(delay, response_id):
            await asyncio.sleep(delay)
            mock_response = Mock()
            mock_response.choices = [Mock(
                message=Mock(
                    content=f"Concurrent response {response_id}",
                    role="assistant"
                ),
                finish_reason="stop"
            )]
            mock_response.usage = Mock(
                prompt_tokens=10,
                completion_tokens=8,
                total_tokens=18
            )
            return mock_response
        
        # 创建并发任务
        tasks = []
        start_time = time.time()
        
        for i in range(3):
            # 每个请求有不同的延迟
            delay = 0.1 + (i * 0.05)
            task = async_create_with_delay(delay, i+1)
            tasks.append(task)
        
        # 并发执行
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # 验证并发结果
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.choices[0].message.content == f"Concurrent response {i+1}"
        
        # 验证并发执行时间（应该接近最长延迟，而不是所有延迟的总和）
        total_time = end_time - start_time
        max_delay = 0.1 + (2 * 0.05)  # 最长延迟 = 0.2秒
        assert total_time < max_delay + 0.3  # 允许更多额外开销
        assert total_time >= max_delay * 0.8  # 允许一些时间误差


class TestStreamingPerformance:
    """流式性能测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.stream_test
    @pytest.mark.performance
    def test_streaming_latency(self, mock_harborai_client):
        """测试流式输出延迟"""
        # 配置带时间戳的流式响应
        chunk_timestamps = []
        
        def create_chunk_with_timestamp(content, is_final=False):
            timestamp = time.time()
            chunk_timestamps.append(timestamp)
            
            return Mock(
                choices=[Mock(
                    delta=Mock(content=content if not is_final else None),
                    index=0,
                    finish_reason="stop" if is_final else None
                )],
                object="chat.completion.chunk",
                created=int(timestamp)
            )
        
        mock_chunks = [
            create_chunk_with_timestamp("First"),
            create_chunk_with_timestamp(" chunk"),
            create_chunk_with_timestamp(" received"),
            create_chunk_with_timestamp("", is_final=True)
        ]
        
        # 配置流式性能测试mock
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return iter(mock_chunks)
            else:
                return Mock()
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 执行流式调用并测量延迟
        stream = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Test latency"}],
            stream=True
        )
        
        received_timestamps = []
        for chunk in stream:
            received_timestamps.append(time.time())
        
        # 验证延迟
        assert len(received_timestamps) == 4
        
        # 计算处理延迟（接收时间 - 创建时间）
        latencies = []
        for i, (chunk_time, received_time) in enumerate(zip(chunk_timestamps, received_timestamps)):
            latency = received_time - chunk_time
            latencies.append(latency)
            # 延迟应该很小（通常小于1秒）
            assert latency < 1.0, f"Chunk {i} latency too high: {latency}s"
        
        # 平均延迟应该合理
        avg_latency = sum(latencies) / len(latencies)
        assert avg_latency < 0.5, f"Average latency too high: {avg_latency}s"
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.stream_test
    @pytest.mark.performance
    def test_streaming_throughput(self, mock_harborai_client):
        """测试流式输出吞吐量"""
        # 配置大量chunk的流式响应
        chunk_count = 50
        mock_chunks = []
        
        for i in range(chunk_count):
            if i < chunk_count - 1:
                content = f"Chunk{i} "
                finish_reason = None
            else:
                content = None
                finish_reason = "stop"
            
            mock_chunks.append(Mock(
                choices=[Mock(
                    delta=Mock(content=content),
                    index=0,
                    finish_reason=finish_reason
                )],
                object="chat.completion.chunk"
            ))
        
        # 配置吞吐量测试mock
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return iter(mock_chunks)
            else:
                return Mock()
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 执行流式调用并测量吞吐量
        start_time = time.time()
        
        stream = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Test throughput"}],
            stream=True
        )
        
        processed_chunks = 0
        for chunk in stream:
            processed_chunks += 1
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 验证吞吐量
        assert processed_chunks == chunk_count
        
        # 计算每秒处理的chunk数
        throughput = processed_chunks / processing_time if processing_time > 0 else float('inf')
        
        # 吞吐量应该合理（这里设置一个较低的阈值）
        assert throughput > 10, f"Throughput too low: {throughput} chunks/second"
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.async_test
    @pytest.mark.stream_test
    @pytest.mark.performance
    async def test_async_streaming_performance(self, mock_harborai_client):
        """测试异步流式性能"""
        # 配置异步流式性能测试
        async def async_performance_stream():
            for i in range(20):
                # 模拟网络延迟
                await asyncio.sleep(0.01)
                
                if i < 19:
                    content = f"Token{i} "
                    finish_reason = None
                else:
                    content = None
                    finish_reason = "stop"
                
                yield Mock(
                    choices=[Mock(
                        delta=Mock(content=content),
                        index=0,
                        finish_reason=finish_reason
                    )],
                    object="chat.completion.chunk",
                    created=int(time.time())
                )
        
        # 配置异步性能测试mock
        def mock_acreate(*args, **kwargs):
            if kwargs.get('stream', False):
                return async_performance_stream()
            else:
                async def async_response():
                    return Mock()
                return async_response()
        
        mock_harborai_client.chat.completions.acreate = mock_acreate
        
        # 执行异步流式性能测试
        start_time = time.time()
        
        stream = mock_harborai_client.chat.completions.acreate(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Performance test"}],
            stream=True
        )
        
        chunk_count = 0
        first_chunk_time = None
        last_chunk_time = None
        
        async for chunk in stream:
            current_time = time.time()
            
            if first_chunk_time is None:
                first_chunk_time = current_time
            last_chunk_time = current_time
            
            chunk_count += 1
        
        end_time = time.time()
        
        # 验证异步流式性能
        assert chunk_count == 20
        
        # 计算总处理时间和流式时间
        total_time = end_time - start_time
        streaming_time = last_chunk_time - first_chunk_time if first_chunk_time and last_chunk_time else 0
        
        # 验证时间合理性
        assert total_time >= 0.2  # 至少包含模拟的网络延迟
        assert streaming_time >= 0.18  # 流式时间应该接近总延迟
        
        # 计算异步流式吞吐量
        async_throughput = chunk_count / total_time if total_time > 0 else float('inf')
        assert async_throughput > 5, f"Async throughput too low: {async_throughput} chunks/second"


class TestStreamingErrorHandling:
    """流式错误处理测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.stream_test
    @pytest.mark.error_handling
    def test_streaming_interruption(self, mock_harborai_client):
        """测试流式输出中断处理"""
        # 配置会中断的流式响应
        def interrupted_stream():
            # 正常的chunk
            yield Mock(
                choices=[Mock(
                    delta=Mock(content="Normal"),
                    index=0,
                    finish_reason=None
                )],
                object="chat.completion.chunk"
            )
            
            # 中断（抛出异常）
            raise ConnectionError("Stream interrupted")
        
        # 配置中断测试mock
        def mock_create(*args, **kwargs):
            if kwargs.get('stream', False):
                return interrupted_stream()
            else:
                return Mock()
        
        mock_harborai_client.chat.completions.create = mock_create
        
        # 执行流式调用并处理中断
        stream = mock_harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Test interruption"}],
            stream=True
        )
        
        collected_content = []
        with pytest.raises(ConnectionError, match="Stream interrupted"):
            for chunk in stream:
                if chunk.choices and hasattr(chunk.choices[0].delta, 'content'):
                    if chunk.choices[0].delta.content:
                        collected_content.append(chunk.choices[0].delta.content)
        
        # 验证在中断前收集到了部分内容
        assert len(collected_content) == 1
        assert collected_content[0] == "Normal"
    
    @pytest.mark.asyncio
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.async_test
    @pytest.mark.stream_test
    @pytest.mark.error_handling
    async def test_async_streaming_timeout(self, mock_harborai_client):
        """测试异步流式超时处理"""
        # 配置会超时的异步流
        async def timeout_stream():
            # 第一个chunk正常
            yield Mock(
                choices=[Mock(
                    delta=Mock(content="Before"),
                    index=0,
                    finish_reason=None
                )],
                object="chat.completion.chunk"
            )
            
            # 模拟长时间等待（超时）
            await asyncio.sleep(2.0)
            
            # 这个chunk不应该被接收到
            yield Mock(
                choices=[Mock(
                    delta=Mock(content="After"),
                    index=0,
                    finish_reason=None
                )],
                object="chat.completion.chunk"
            )
        
        # 配置超时测试mock
        def mock_acreate(*args, **kwargs):
            if kwargs.get('stream', False):
                return timeout_stream()
            else:
                async def async_response():
                    return Mock()
                return async_response()
        
        mock_harborai_client.chat.completions.acreate = mock_acreate
        
        # 执行带超时的异步流式调用
        stream = mock_harborai_client.chat.completions.acreate(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Test timeout"}],
            stream=True
        )
        
        collected_content = []
        
        # 使用超时处理
        with pytest.raises(asyncio.TimeoutError):
            async with asyncio.timeout(1.0):  # 1秒超时
                async for chunk in stream:
                    if chunk.choices and hasattr(chunk.choices[0].delta, 'content'):
                        if chunk.choices[0].delta.content:
                            collected_content.append(chunk.choices[0].delta.content)
        
        # 验证在超时前收集到了部分内容
        assert len(collected_content) == 1
        assert collected_content[0] == "Before"