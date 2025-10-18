#!/usr/bin/env python3
"""
追踪系统集成测试

测试追踪系统各组件的集成功能：
- DualTraceIDManager与TracingDataCollector集成
- OpenTelemetry与HarborAI追踪系统集成
- 日志系统与追踪系统集成
- 端到端追踪流程测试
- 性能和可靠性测试

作者: HarborAI团队
创建时间: 2025-01-15
版本: v1.0.0
"""

import pytest
import asyncio
import time
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

import asyncpg
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

from harborai.core.tracing.dual_trace_manager import (
    DualTraceIDManager,
    DualTraceContext,
    TracingRecord
)
from harborai.core.tracing.data_collector import (
    TracingDataCollector,
    CollectorStatus,
    CollectorStatistics
)
from harborai.core.logging.enhanced_fallback_logger import EnhancedFallbackLogger
from harborai.core.logging.enhanced_file_logger import EnhancedFileSystemLogger
from harborai.core.logging.optimized_postgresql_logger import OptimizedPostgreSQLLogger
from harborai.utils.exceptions import TracingError, DatabaseError


class TestTracingSystemIntegration:
    """追踪系统集成测试"""
    
    @pytest.fixture
    def mock_db_config(self):
        """模拟数据库配置"""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "test_harborai",
            "user": "test_user",
            "password": "test_password"
        }
    
    @pytest.fixture
    def trace_manager(self):
        """创建追踪管理器"""
        return DualTraceIDManager(
            service_name="integration-test-service",
            service_version="1.0.0",
            environment="test"
        )
    
    @pytest.fixture
    def data_collector(self, mock_db_config):
        """创建数据收集器"""
        return TracingDataCollector(
            db_config=mock_db_config,
            batch_size=5,
            flush_interval=0.1,
            max_queue_size=50,
            max_retries=2
        )
    
    @pytest.fixture
    def temp_log_dir(self):
        """创建临时日志目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.mark.asyncio
    async def test_trace_manager_collector_integration(self, trace_manager, data_collector):
        """测试追踪管理器与数据收集器集成"""
        # 模拟数据库操作
        mock_connection = AsyncMock()
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.close = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool):
            with patch('harborai.core.tracing.dual_trace_manager.trace.get_tracer') as mock_get_tracer:
                # 设置OpenTelemetry模拟
                mock_tracer, mock_span = self._setup_mock_tracer(mock_get_tracer)
                
                # 启动数据收集器
                await data_collector.start()
                
                try:
                    # 1. 创建追踪上下文
                    context = trace_manager.create_trace_context(
                        hb_trace_id="integration_test_001",
                        operation_name="ai.chat.completion"
                    )
                    
                    # 2. 添加标签和日志
                    trace_manager.add_tag_to_context(
                        "integration_test_001", 
                        "model", 
                        "gpt-4"
                    )
                    trace_manager.add_tag_to_context(
                        "integration_test_001", 
                        "provider", 
                        "openai"
                    )
                    trace_manager.add_log_to_context(
                        "integration_test_001",
                        "info",
                        "Request processing started",
                        {"request_id": "req_001"}
                    )
                    
                    # 3. 模拟处理时间
                    await asyncio.sleep(0.05)
                    
                    # 4. 添加更多日志
                    trace_manager.add_log_to_context(
                        "integration_test_001",
                        "debug",
                        "Token processing completed",
                        {"tokens": 150}
                    )
                    
                    # 5. 完成追踪
                    finished_context = trace_manager.finish_trace_context(
                        "integration_test_001",
                        success=True
                    )
                    
                    # 6. 创建追踪记录并发送到收集器
                    tracing_record = TracingRecord.from_context(finished_context)
                    data_collector.add_record(tracing_record)
                    
                    # 7. 等待处理
                    await asyncio.sleep(0.2)
                    
                    # 验证结果
                    assert finished_context.status == "completed"
                    assert finished_context.tags["model"] == "gpt-4"
                    assert len(finished_context.logs) == 2
                    
                    # 验证数据收集器状态
                    status = data_collector.get_status()
                    assert status.processed_count > 0
                    
                    # 验证数据库调用
                    mock_connection.executemany.assert_called()
                    
                finally:
                    await data_collector.stop()
    
    @pytest.mark.asyncio
    async def test_enhanced_logger_tracing_integration(self, trace_manager, temp_log_dir):
        """测试增强日志器与追踪系统集成"""
        # 创建增强文件日志器
        file_logger = EnhancedFileSystemLogger(
            log_directory=temp_log_dir,
            max_file_size=1024*1024,  # 1MB
            max_files=5,
            compression_enabled=False,  # 测试时禁用压缩
            trace_manager=trace_manager
        )
        
        with patch('harborai.core.tracing.dual_trace_manager.trace.get_tracer') as mock_get_tracer:
            # 设置OpenTelemetry模拟
            mock_tracer, mock_span = self._setup_mock_tracer(mock_get_tracer)
            
            try:
                # 启动日志器
                await file_logger.start()
                
                # 1. 使用追踪功能记录请求
                request_data = {
                    "model": "gpt-4",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.7
                }
                
                trace_id = await file_logger.log_request_with_tracing(
                    operation_name="ai.chat.completion",
                    request_data=request_data,
                    api_tags={"provider": "openai", "model": "gpt-4"},
                    internal_tags={"service": "harborai", "version": "2.0.0"}
                )
                
                # 2. 模拟处理时间
                await asyncio.sleep(0.05)
                
                # 3. 记录响应
                response_data = {
                    "choices": [{"message": {"content": "Hello! How can I help you?"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
                }
                
                await file_logger.log_response_with_tracing(
                    trace_id=trace_id,
                    response_data=response_data,
                    success=True,
                    latency_ms=45.5,
                    cost_breakdown={"input_cost": 0.001, "output_cost": 0.002, "total_cost": 0.003}
                )
                
                # 4. 等待日志写入
                await asyncio.sleep(0.2)
                
                # 验证追踪上下文
                completed_contexts = trace_manager.get_completed_contexts()
                assert len(completed_contexts) == 1
                
                completed_context = list(completed_contexts.values())[0]
                assert completed_context.status == "completed"
                assert completed_context.tags["provider"] == "openai"
                assert completed_context.tags["model"] == "gpt-4"
                
                # 验证日志文件
                log_files = file_logger.get_log_files()
                assert len(log_files) > 0
                
                # 读取日志内容
                logs = await file_logger.read_logs(trace_id=trace_id)
                assert len(logs) >= 2  # 至少有请求和响应日志
                
                # 验证日志内容包含追踪信息
                request_log = next(log for log in logs if log.get("type") == "request")
                assert request_log["trace_id"] == trace_id
                assert "otel_trace_id" in request_log
                
                response_log = next(log for log in logs if log.get("type") == "response")
                assert response_log["trace_id"] == trace_id
                assert response_log["success"] is True
                assert response_log["latency_ms"] == 45.5
                
            finally:
                await file_logger.stop()
    
    @pytest.mark.asyncio
    async def test_fallback_logger_tracing_integration(self, trace_manager, temp_log_dir, mock_db_config):
        """测试回退日志器与追踪系统集成"""
        # 创建增强回退日志器
        fallback_logger = EnhancedFallbackLogger(
            postgres_config=mock_db_config,
            file_config={
                "log_directory": temp_log_dir,
                "max_file_size": 1024*1024,
                "max_files": 5,
                "compression_enabled": False
            },
            trace_manager=trace_manager
        )
        
        # 模拟PostgreSQL连接失败，强制使用文件回退
        with patch('asyncpg.create_pool') as mock_create_pool:
            mock_create_pool.side_effect = Exception("PostgreSQL unavailable")
            
            with patch('harborai.core.tracing.dual_trace_manager.trace.get_tracer') as mock_get_tracer:
                # 设置OpenTelemetry模拟
                mock_tracer, mock_span = self._setup_mock_tracer(mock_get_tracer)
                
                try:
                    # 启动回退日志器
                    await fallback_logger.start()
                    
                    # 等待状态稳定
                    await asyncio.sleep(0.1)
                    
                    # 验证已切换到文件回退模式
                    status = fallback_logger.get_status()
                    assert status["current_logger"] == "file"
                    assert status["postgres_status"] == "failed"
                    
                    # 1. 使用追踪功能记录请求
                    request_data = {
                        "model": "claude-3",
                        "messages": [{"role": "user", "content": "Explain quantum computing"}],
                        "max_tokens": 500
                    }
                    
                    trace_id = await fallback_logger.log_request_with_tracing(
                        operation_name="ai.chat.completion",
                        request_data=request_data,
                        api_tags={"provider": "anthropic", "model": "claude-3"},
                        internal_tags={"service": "harborai", "fallback": "true"}
                    )
                    
                    # 2. 模拟处理时间
                    await asyncio.sleep(0.05)
                    
                    # 3. 记录响应
                    response_data = {
                        "content": [{"text": "Quantum computing is..."}],
                        "usage": {"input_tokens": 15, "output_tokens": 120}
                    }
                    
                    await fallback_logger.log_response_with_tracing(
                        trace_id=trace_id,
                        response_data=response_data,
                        success=True,
                        latency_ms=89.2,
                        cost_breakdown={"input_cost": 0.002, "output_cost": 0.008, "total_cost": 0.010}
                    )
                    
                    # 4. 等待处理
                    await asyncio.sleep(0.2)
                    
                    # 验证追踪上下文
                    completed_contexts = trace_manager.get_completed_contexts()
                    assert len(completed_contexts) == 1
                    
                    completed_context = list(completed_contexts.values())[0]
                    assert completed_context.status == "completed"
                    assert completed_context.tags["provider"] == "anthropic"
                    assert completed_context.tags["fallback"] == "true"
                    
                    # 验证文件日志器记录了数据
                    file_logger = fallback_logger.file_logger
                    logs = await file_logger.read_logs(trace_id=trace_id)
                    assert len(logs) >= 2
                    
                    # 验证日志包含回退信息
                    request_log = next(log for log in logs if log.get("type") == "request")
                    assert request_log["trace_id"] == trace_id
                    assert "fallback_mode" in request_log or "logger_type" in request_log
                    
                finally:
                    await fallback_logger.stop()
    
    @pytest.mark.asyncio
    async def test_end_to_end_tracing_flow(self, trace_manager, data_collector, temp_log_dir):
        """测试端到端追踪流程"""
        # 创建完整的追踪系统
        file_logger = EnhancedFileSystemLogger(
            log_directory=temp_log_dir,
            max_file_size=1024*1024,
            max_files=5,
            compression_enabled=False,
            trace_manager=trace_manager
        )
        
        # 模拟数据库操作
        mock_connection = AsyncMock()
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.close = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool):
            with patch('harborai.core.tracing.dual_trace_manager.trace.get_tracer') as mock_get_tracer:
                # 设置OpenTelemetry模拟
                mock_tracer, mock_span = self._setup_mock_tracer(mock_get_tracer)
                
                try:
                    # 启动所有组件
                    await data_collector.start()
                    await file_logger.start()
                    
                    # 模拟完整的AI请求处理流程
                    requests = [
                        {
                            "operation": "ai.chat.completion",
                            "model": "gpt-4",
                            "provider": "openai",
                            "messages": [{"role": "user", "content": "What is AI?"}]
                        },
                        {
                            "operation": "ai.text.embedding",
                            "model": "text-embedding-ada-002",
                            "provider": "openai",
                            "input": "Machine learning is a subset of AI"
                        },
                        {
                            "operation": "ai.image.generation",
                            "model": "dall-e-3",
                            "provider": "openai",
                            "prompt": "A futuristic robot"
                        }
                    ]
                    
                    trace_ids = []
                    
                    # 处理每个请求
                    for i, request in enumerate(requests):
                        # 1. 开始追踪
                        trace_id = await file_logger.log_request_with_tracing(
                            operation_name=request["operation"],
                            request_data=request,
                            api_tags={
                                "provider": request["provider"],
                                "model": request["model"],
                                "request_index": str(i)
                            },
                            internal_tags={
                                "service": "harborai",
                                "version": "2.0.0",
                                "environment": "test"
                            }
                        )
                        trace_ids.append(trace_id)
                        
                        # 2. 模拟处理时间
                        processing_time = 0.02 + (i * 0.01)  # 递增处理时间
                        await asyncio.sleep(processing_time)
                        
                        # 3. 模拟响应
                        if request["operation"] == "ai.chat.completion":
                            response_data = {
                                "choices": [{"message": {"content": "AI is artificial intelligence..."}}],
                                "usage": {"prompt_tokens": 12, "completion_tokens": 25, "total_tokens": 37}
                            }
                            cost = 0.005
                        elif request["operation"] == "ai.text.embedding":
                            response_data = {
                                "data": [{"embedding": [0.1, 0.2, 0.3]}],
                                "usage": {"prompt_tokens": 8, "total_tokens": 8}
                            }
                            cost = 0.001
                        else:  # image generation
                            response_data = {
                                "data": [{"url": "https://example.com/image.png"}],
                                "usage": {"prompt_tokens": 5}
                            }
                            cost = 0.040
                        
                        # 4. 记录响应
                        await file_logger.log_response_with_tracing(
                            trace_id=trace_id,
                            response_data=response_data,
                            success=True,
                            latency_ms=(processing_time * 1000) + 10,
                            cost_breakdown={
                                "input_cost": cost * 0.3,
                                "output_cost": cost * 0.7,
                                "total_cost": cost
                            }
                        )
                        
                        # 5. 将追踪记录发送到收集器
                        completed_contexts = trace_manager.get_completed_contexts()
                        if trace_id in completed_contexts:
                            tracing_record = TracingRecord.from_context(completed_contexts[trace_id])
                            data_collector.add_record(tracing_record)
                    
                    # 等待所有处理完成
                    await asyncio.sleep(0.5)
                    
                    # 验证追踪管理器状态
                    stats = trace_manager.get_statistics()
                    assert stats["completed_contexts"] == 3
                    assert stats["total_contexts"] == 3
                    
                    # 验证数据收集器状态
                    collector_status = data_collector.get_status()
                    assert collector_status.processed_count == 3
                    
                    # 验证文件日志
                    for trace_id in trace_ids:
                        logs = await file_logger.read_logs(trace_id=trace_id)
                        assert len(logs) >= 2  # 请求和响应日志
                        
                        request_log = next(log for log in logs if log.get("type") == "request")
                        response_log = next(log for log in logs if log.get("type") == "response")
                        
                        assert request_log["trace_id"] == trace_id
                        assert response_log["trace_id"] == trace_id
                        assert response_log["success"] is True
                    
                    # 验证数据库调用
                    assert mock_connection.executemany.call_count >= 1
                    
                    # 验证追踪记录内容
                    call_args = mock_connection.executemany.call_args_list
                    for call in call_args:
                        sql, records = call[0]
                        assert "INSERT INTO tracing_info" in sql
                        assert len(records) > 0
                        
                        for record in records:
                            assert record[0] in trace_ids  # hb_trace_id
                            assert len(record[1]) == 32  # otel_trace_id
                            assert record[5] == "completed"  # status
                    
                finally:
                    await data_collector.stop()
                    await file_logger.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, trace_manager, data_collector):
        """测试错误处理集成"""
        # 模拟数据库操作（部分失败）
        mock_connection = AsyncMock()
        call_count = 0
        
        def mock_executemany(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # 第二次调用失败
                raise Exception("Database connection lost")
            return None
        
        mock_connection.executemany.side_effect = mock_executemany
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.close = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool):
            with patch('harborai.core.tracing.dual_trace_manager.trace.get_tracer') as mock_get_tracer:
                # 设置OpenTelemetry模拟
                mock_tracer, mock_span = self._setup_mock_tracer(mock_get_tracer)
                
                try:
                    await data_collector.start()
                    
                    # 创建多个追踪上下文，其中一些会失败
                    trace_contexts = []
                    for i in range(5):
                        context = trace_manager.create_trace_context(
                            hb_trace_id=f"error_test_{i}",
                            operation_name=f"ai.test.operation.{i}"
                        )
                        
                        # 添加标签
                        trace_manager.add_tag_to_context(
                            f"error_test_{i}",
                            "test_index",
                            str(i)
                        )
                        
                        # 模拟一些操作失败
                        if i == 2 or i == 4:
                            # 以错误状态完成
                            finished_context = trace_manager.finish_trace_context(
                                f"error_test_{i}",
                                success=False,
                                error=f"Simulated error for test {i}"
                            )
                        else:
                            # 正常完成
                            finished_context = trace_manager.finish_trace_context(
                                f"error_test_{i}",
                                success=True
                            )
                        
                        trace_contexts.append(finished_context)
                        
                        # 发送到收集器
                        tracing_record = TracingRecord.from_context(finished_context)
                        data_collector.add_record(tracing_record)
                    
                    # 等待处理
                    await asyncio.sleep(0.5)
                    
                    # 验证错误处理
                    collector_status = data_collector.get_status()
                    assert collector_status.error_count > 0  # 应该有数据库错误
                    assert collector_status.processed_count > 0  # 应该有成功处理的记录
                    
                    # 验证追踪上下文状态
                    completed_contexts = trace_manager.get_completed_contexts()
                    assert len(completed_contexts) == 5
                    
                    # 验证错误状态的上下文
                    error_contexts = [ctx for ctx in completed_contexts.values() if ctx.status == "error"]
                    success_contexts = [ctx for ctx in completed_contexts.values() if ctx.status == "completed"]
                    
                    assert len(error_contexts) == 2  # error_test_2 和 error_test_4
                    assert len(success_contexts) == 3
                    
                    # 验证错误信息
                    for error_ctx in error_contexts:
                        assert error_ctx.error is not None
                        assert "Simulated error" in error_ctx.error
                    
                finally:
                    await data_collector.stop()
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, trace_manager, data_collector):
        """测试性能监控集成"""
        # 模拟数据库操作
        mock_connection = AsyncMock()
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.close = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool):
            with patch('harborai.core.tracing.dual_trace_manager.trace.get_tracer') as mock_get_tracer:
                # 设置OpenTelemetry模拟
                mock_tracer, mock_span = self._setup_mock_tracer(mock_get_tracer)
                
                try:
                    await data_collector.start()
                    
                    # 记录开始时间
                    start_time = time.time()
                    
                    # 创建大量追踪上下文以测试性能
                    trace_count = 50
                    for i in range(trace_count):
                        context = trace_manager.create_trace_context(
                            hb_trace_id=f"perf_test_{i}",
                            operation_name="ai.performance.test"
                        )
                        
                        # 添加标签和日志
                        trace_manager.add_tag_to_context(
                            f"perf_test_{i}",
                            "batch_index",
                            str(i // 10)
                        )
                        trace_manager.add_log_to_context(
                            f"perf_test_{i}",
                            "info",
                            f"Processing item {i}"
                        )
                        
                        # 模拟短暂处理时间
                        await asyncio.sleep(0.001)
                        
                        # 完成追踪
                        finished_context = trace_manager.finish_trace_context(
                            f"perf_test_{i}",
                            success=True
                        )
                        
                        # 发送到收集器
                        tracing_record = TracingRecord.from_context(finished_context)
                        data_collector.add_record(tracing_record)
                    
                    # 等待所有处理完成
                    await asyncio.sleep(1.0)
                    
                    # 记录结束时间
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    # 验证性能指标
                    collector_stats = data_collector.get_statistics()
                    trace_stats = trace_manager.get_statistics()
                    
                    # 验证处理速度
                    records_per_second = collector_stats.records_per_second
                    assert records_per_second > 0
                    
                    # 验证平均处理时间
                    avg_processing_time = collector_stats.average_processing_time_ms
                    assert avg_processing_time > 0
                    assert avg_processing_time < 100  # 应该小于100ms
                    
                    # 验证所有记录都被处理
                    assert collector_stats.successful_records == trace_count
                    assert trace_stats["completed_contexts"] == trace_count
                    
                    # 验证总处理时间合理
                    assert total_time < 10.0  # 应该在10秒内完成
                    
                    # 验证内存使用（队列应该被清空）
                    collector_status = data_collector.get_status()
                    assert collector_status.queue_size == 0
                    
                    print(f"性能测试结果:")
                    print(f"  总记录数: {trace_count}")
                    print(f"  总时间: {total_time:.2f}秒")
                    print(f"  处理速度: {records_per_second:.2f} 记录/秒")
                    print(f"  平均处理时间: {avg_processing_time:.2f}ms")
                    print(f"  成功率: {collector_stats.successful_records / trace_count * 100:.1f}%")
                    
                finally:
                    await data_collector.stop()
    
    def _setup_mock_tracer(self, mock_get_tracer):
        """设置模拟tracer的辅助方法"""
        mock_tracer = Mock()
        mock_span = Mock()
        
        # 设置span上下文
        mock_span_context = Mock()
        mock_span_context.trace_id = 0x0123456789abcdef0123456789abcdef
        mock_span_context.span_id = 0x0123456789abcdef
        mock_span.get_span_context.return_value = mock_span_context
        
        # 设置span的上下文管理器行为
        mock_span.__enter__ = Mock(return_value=mock_span)
        mock_span.__exit__ = Mock(return_value=None)
        
        # 设置tracer返回span
        mock_tracer.start_as_current_span.return_value = mock_span
        mock_get_tracer.return_value = mock_tracer
        
        return mock_tracer, mock_span


class TestTracingSystemReliability:
    """追踪系统可靠性测试"""
    
    @pytest.fixture
    def mock_db_config(self):
        """模拟数据库配置"""
        return {
            "host": "localhost",
            "port": 5432,
            "database": "test_harborai",
            "user": "test_user",
            "password": "test_password"
        }
    
    @pytest.mark.asyncio
    async def test_system_recovery_after_failure(self, mock_db_config):
        """测试系统故障后恢复"""
        trace_manager = DualTraceIDManager(
            service_name="reliability-test",
            service_version="1.0.0",
            environment="test"
        )
        
        data_collector = TracingDataCollector(
            db_config=mock_db_config,
            batch_size=5,
            flush_interval=0.1,
            max_queue_size=20,
            max_retries=3
        )
        
        # 模拟数据库连接（先失败后恢复）
        mock_connection = AsyncMock()
        failure_count = 0
        
        def mock_executemany(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 3:  # 前3次失败
                raise Exception("Database temporarily unavailable")
            return None  # 之后成功
        
        mock_connection.executemany.side_effect = mock_executemany
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.close = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool):
            with patch('harborai.core.tracing.dual_trace_manager.trace.get_tracer') as mock_get_tracer:
                # 设置OpenTelemetry模拟
                mock_tracer = Mock()
                mock_span = Mock()
                mock_span_context = Mock()
                mock_span_context.trace_id = 0x0123456789abcdef0123456789abcdef
                mock_span_context.span_id = 0x0123456789abcdef
                mock_span.get_span_context.return_value = mock_span_context
                mock_span.__enter__ = Mock(return_value=mock_span)
                mock_span.__exit__ = Mock(return_value=None)
                mock_tracer.start_as_current_span.return_value = mock_span
                mock_get_tracer.return_value = mock_tracer
                
                try:
                    await data_collector.start()
                    
                    # 在故障期间添加记录
                    for i in range(10):
                        context = trace_manager.create_trace_context(
                            hb_trace_id=f"recovery_test_{i}",
                            operation_name="ai.recovery.test"
                        )
                        
                        finished_context = trace_manager.finish_trace_context(
                            f"recovery_test_{i}",
                            success=True
                        )
                        
                        tracing_record = TracingRecord.from_context(finished_context)
                        data_collector.add_record(tracing_record)
                        
                        # 短暂等待
                        await asyncio.sleep(0.01)
                    
                    # 等待恢复和处理
                    await asyncio.sleep(1.0)
                    
                    # 验证系统恢复
                    collector_status = data_collector.get_status()
                    assert collector_status.processed_count > 0  # 应该有成功处理的记录
                    assert collector_status.error_count > 0  # 应该有错误记录
                    
                    # 验证重试机制工作
                    assert mock_connection.executemany.call_count > 3  # 应该有重试
                    
                    # 验证最终数据一致性
                    trace_stats = trace_manager.get_statistics()
                    assert trace_stats["completed_contexts"] == 10
                    
                finally:
                    await data_collector.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_stress_test(self, mock_db_config):
        """测试并发压力"""
        import threading
        
        trace_manager = DualTraceIDManager(
            service_name="stress-test",
            service_version="1.0.0",
            environment="test"
        )
        
        data_collector = TracingDataCollector(
            db_config=mock_db_config,
            batch_size=10,
            flush_interval=0.05,
            max_queue_size=200,
            max_retries=2
        )
        
        # 模拟数据库操作
        mock_connection = AsyncMock()
        mock_pool = Mock()
        mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        mock_pool.close = AsyncMock()
        
        with patch('asyncpg.create_pool', return_value=mock_pool):
            with patch('harborai.core.tracing.dual_trace_manager.trace.get_tracer') as mock_get_tracer:
                # 设置OpenTelemetry模拟
                mock_tracer = Mock()
                mock_span = Mock()
                mock_span_context = Mock()
                mock_span_context.trace_id = 0x0123456789abcdef0123456789abcdef
                mock_span_context.span_id = 0x0123456789abcdef
                mock_span.get_span_context.return_value = mock_span_context
                mock_span.__enter__ = Mock(return_value=mock_span)
                mock_span.__exit__ = Mock(return_value=None)
                mock_tracer.start_as_current_span.return_value = mock_span
                mock_get_tracer.return_value = mock_tracer
                
                try:
                    await data_collector.start()
                    
                    # 并发创建追踪上下文
                    def create_traces(thread_id, count):
                        for i in range(count):
                            try:
                                # 创建上下文（不使用OpenTelemetry以避免线程问题）
                                hb_trace_id = f"stress_test_{thread_id}_{i}"
                                context = DualTraceContext(
                                    hb_trace_id=hb_trace_id,
                                    otel_trace_id=f"{thread_id:016x}{i:016x}",
                                    span_id=f"{thread_id:08x}{i:08x}",
                                    operation_name=f"ai.stress.test.{thread_id}",
                                    service_name="stress-test"
                                )
                                
                                trace_manager.active_contexts[hb_trace_id] = context
                                
                                # 添加标签
                                trace_manager.add_tag_to_context(
                                    hb_trace_id,
                                    "thread_id",
                                    str(thread_id)
                                )
                                
                                # 完成上下文
                                finished_context = trace_manager.finish_trace_context(
                                    hb_trace_id,
                                    success=True
                                )
                                
                                # 发送到收集器
                                tracing_record = TracingRecord.from_context(finished_context)
                                data_collector.add_record(tracing_record)
                                
                            except Exception as e:
                                print(f"Error in thread {thread_id}, iteration {i}: {e}")
                    
                    # 创建多个线程
                    threads = []
                    thread_count = 10
                    traces_per_thread = 20
                    
                    for thread_id in range(thread_count):
                        thread = threading.Thread(
                            target=create_traces,
                            args=(thread_id, traces_per_thread)
                        )
                        threads.append(thread)
                    
                    # 启动所有线程
                    start_time = time.time()
                    for thread in threads:
                        thread.start()
                    
                    # 等待所有线程完成
                    for thread in threads:
                        thread.join()
                    
                    # 等待处理完成
                    await asyncio.sleep(2.0)
                    end_time = time.time()
                    
                    # 验证结果
                    total_expected = thread_count * traces_per_thread
                    
                    collector_stats = data_collector.get_statistics()
                    trace_stats = trace_manager.get_statistics()
                    
                    # 验证数据完整性
                    assert trace_stats["completed_contexts"] == total_expected
                    assert collector_stats.total_records >= total_expected * 0.9  # 允许10%的容错
                    
                    # 验证性能
                    total_time = end_time - start_time
                    throughput = total_expected / total_time
                    
                    print(f"并发压力测试结果:")
                    print(f"  线程数: {thread_count}")
                    print(f"  每线程追踪数: {traces_per_thread}")
                    print(f"  总追踪数: {total_expected}")
                    print(f"  总时间: {total_time:.2f}秒")
                    print(f"  吞吐量: {throughput:.2f} 追踪/秒")
                    print(f"  成功处理: {collector_stats.successful_records}")
                    print(f"  失败处理: {collector_stats.failed_records}")
                    
                    # 性能断言
                    assert throughput > 50  # 至少50追踪/秒
                    assert collector_stats.failed_records < total_expected * 0.1  # 失败率小于10%
                    
                finally:
                    await data_collector.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])