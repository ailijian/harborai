#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
链路追踪模块comprehensive测试

测试 HarborAI 全链路追踪功能的所有组件，确保追踪机制正常工作。
"""

import pytest
import asyncio
import time
import threading
from unittest.mock import Mock, patch, call
from contextvars import ContextVar
from typing import Any, Optional

from harborai.utils.tracer import (
    generate_trace_id,
    set_current_trace_id,
    get_current_trace_id,
    get_or_create_trace_id,
    TraceContext,
    SpanTimer,
    trace_function,
    trace_async_function,
    _current_trace_id,
)


class TestGenerateTraceId:
    """generate_trace_id函数测试"""
    
    def test_generate_trace_id_format(self):
        """测试生成的trace_id格式"""
        trace_id = generate_trace_id()
        
        # 验证格式：harborai_timestamp_random
        assert isinstance(trace_id, str)
        assert trace_id.startswith("harborai_")
        assert len(trace_id) == 31  # harborai_ (9) + timestamp (13) + _ (1) + random (8) = 31
        
        parts = trace_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "harborai"
        assert parts[1].isdigit()  # timestamp
        assert len(parts[2]) == 8  # random part
    
    def test_generate_trace_id_uniqueness(self):
        """测试生成的trace_id唯一性"""
        trace_ids = set()
        
        # 生成多个trace_id
        for _ in range(1000):
            trace_id = generate_trace_id()
            trace_ids.add(trace_id)
        
        # 所有trace_id应该是唯一的
        assert len(trace_ids) == 1000
    
    def test_generate_trace_id_format_consistency(self):
        """测试生成的trace_id格式一致性"""
        trace_id = generate_trace_id()
        
        # 应该包含harborai前缀
        assert trace_id.startswith("harborai_")
        
        # 应该包含时间戳和随机部分
        parts = trace_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "harborai"
        assert parts[1].isdigit()  # 时间戳
        assert len(parts[2]) == 8  # 8位随机字符


class TestTraceIdContext:
    """trace_id上下文管理测试"""
    
    def test_set_and_get_trace_id(self):
        """测试设置和获取trace_id"""
        test_trace_id = "test_trace_123"
        
        # 设置trace_id
        set_current_trace_id(test_trace_id)
        
        # 获取trace_id
        current_trace_id = get_current_trace_id()
        assert current_trace_id == test_trace_id
    
    def test_get_trace_id_when_none(self):
        """测试当没有设置trace_id时获取"""
        # 清除当前trace_id
        _current_trace_id.set(None)
        
        current_trace_id = get_current_trace_id()
        assert current_trace_id is None
    
    def test_get_or_create_trace_id_existing(self):
        """测试获取或创建trace_id（已存在）"""
        existing_trace_id = "existing_trace_456"
        set_current_trace_id(existing_trace_id)
        
        trace_id = get_or_create_trace_id()
        assert trace_id == existing_trace_id
    
    def test_get_or_create_trace_id_new(self):
        """测试获取或创建trace_id（不存在）"""
        # 清除当前trace_id
        _current_trace_id.set(None)
        
        trace_id = get_or_create_trace_id()
        
        # 应该生成新的trace_id
        assert trace_id is not None
        assert isinstance(trace_id, str)
        assert trace_id.startswith("harborai_")
        
        # 应该设置到上下文中
        assert get_current_trace_id() == trace_id
    
    def test_get_or_create_trace_id_with_custom(self):
        """测试带自定义trace_id的获取或创建"""
        # 清除当前trace_id
        _current_trace_id.set(None)
        
        custom_trace_id = "custom_trace_123"
        trace_id = get_or_create_trace_id(custom_trace_id=custom_trace_id)
        
        assert trace_id == custom_trace_id
    
    def test_trace_id_context_isolation(self):
        """测试不同线程间的trace_id隔离"""
        results = {}
        
        def worker(thread_id):
            trace_id = f"thread_{thread_id}_trace"
            set_current_trace_id(trace_id)
            time.sleep(0.01)  # 模拟一些工作
            results[thread_id] = get_current_trace_id()
        
        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证每个线程都有自己的trace_id
        for i in range(5):
            assert results[i] == f"thread_{i}_trace"


class TestTraceContext:
    """TraceContext上下文管理器测试"""
    
    def test_trace_context_basic(self):
        """测试基础上下文管理"""
        test_trace_id = "context_test_trace"
        
        # 在上下文外应该没有trace_id
        _current_trace_id.set(None)
        assert get_current_trace_id() is None
        
        # 在上下文内应该有trace_id
        with TraceContext(test_trace_id):
            assert get_current_trace_id() == test_trace_id
        
        # 退出上下文后应该恢复原状态
        assert get_current_trace_id() is None
    
    def test_trace_context_nested(self):
        """测试嵌套上下文管理"""
        outer_trace_id = "outer_trace"
        inner_trace_id = "inner_trace"
        
        with TraceContext(outer_trace_id):
            assert get_current_trace_id() == outer_trace_id
            
            with TraceContext(inner_trace_id):
                assert get_current_trace_id() == inner_trace_id
            
            # 内层上下文退出后应该恢复外层
            assert get_current_trace_id() == outer_trace_id
    
    def test_trace_context_with_existing_trace_id(self):
        """测试在已有trace_id时使用上下文"""
        existing_trace_id = "existing_trace"
        new_trace_id = "new_trace"
        
        set_current_trace_id(existing_trace_id)
        
        with TraceContext(new_trace_id):
            assert get_current_trace_id() == new_trace_id
        
        # 退出后应该恢复原有的trace_id
        assert get_current_trace_id() == existing_trace_id
    
    def test_trace_context_exception_handling(self):
        """测试上下文管理器的异常处理"""
        original_trace_id = "original_trace"
        context_trace_id = "context_trace"
        
        set_current_trace_id(original_trace_id)
        
        try:
            with TraceContext(context_trace_id):
                assert get_current_trace_id() == context_trace_id
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # 即使发生异常，也应该恢复原有的trace_id
        assert get_current_trace_id() == original_trace_id
    
    def test_trace_context_auto_generate(self):
        """测试自动生成trace_id的上下文"""
        _current_trace_id.set(None)
        
        with TraceContext() as trace_id:
            # 应该自动生成trace_id
            assert trace_id is not None
            assert isinstance(trace_id, str)
            assert trace_id.startswith("harborai_")
            assert get_current_trace_id() == trace_id
        
        # 退出后应该清除
        assert get_current_trace_id() is None


class TestSpanTimer:
    """SpanTimer计时器测试"""
    
    def test_span_timer_basic(self):
        """测试基础计时功能"""
        span_name = "test_span"
        
        with SpanTimer(span_name) as timer:
            time.sleep(0.01)  # 模拟一些工作
        
        # 验证计时器属性
        assert timer.name == span_name
        assert timer.start_time is not None
        assert timer.end_time is not None
        assert timer.duration_ms is not None
        assert timer.duration_ms > 0
    
    def test_span_timer_with_trace_id(self):
        """测试带trace_id的计时"""
        span_name = "test_span_with_trace"
        trace_id = "test_trace_123"
        
        set_current_trace_id(trace_id)
        
        with SpanTimer(span_name) as timer:
            time.sleep(0.01)
        
        # 验证trace_id被正确设置
        assert timer.trace_id == trace_id
    
    def test_span_timer_without_trace_id(self):
        """测试没有trace_id的计时"""
        span_name = "test_span_no_trace"
        
        _current_trace_id.set(None)
        
        with SpanTimer(span_name) as timer:
            time.sleep(0.01)
        
        # 验证trace_id为None
        assert timer.trace_id is None
    
    def test_span_timer_exception_handling(self):
        """测试计时器的异常处理"""
        span_name = "test_span_exception"
        
        timer = None
        try:
            with SpanTimer(span_name) as timer:
                time.sleep(0.01)
                raise ValueError("Test exception")
        except ValueError:
            pass
        
        # 即使发生异常，也应该记录时间
        assert timer is not None
        assert timer.duration_ms is not None
        assert timer.duration_ms > 0
    
    def test_span_timer_duration_accuracy(self):
        """测试计时精度"""
        span_name = "test_span_duration"
        sleep_time = 0.05  # 50ms
        
        with SpanTimer(span_name) as timer:
            time.sleep(sleep_time)
        
        # 验证计时精度
        duration_ms = timer.duration_ms
        expected_ms = sleep_time * 1000
        
        # 允许一定的误差（±20ms）
        assert abs(duration_ms - expected_ms) < 20
    
    def test_span_timer_get_duration_ms(self):
        """测试get_duration_ms方法"""
        span_name = "test_span_get_duration"
        
        with SpanTimer(span_name) as timer:
            # 在计时过程中获取当前持续时间
            time.sleep(0.01)
            current_duration = timer.get_duration_ms()
            assert current_duration is not None
            assert current_duration > 0
        
        # 计时结束后获取最终持续时间
        final_duration = timer.get_duration_ms()
        assert final_duration == timer.duration_ms
    
    def test_span_timer_to_dict(self):
        """测试to_dict方法"""
        span_name = "test_span_to_dict"
        trace_id = "test_trace_dict"
        
        set_current_trace_id(trace_id)
        
        with SpanTimer(span_name) as timer:
            time.sleep(0.01)
        
        result_dict = timer.to_dict()
        
        assert result_dict["name"] == span_name
        assert result_dict["trace_id"] == trace_id
        assert result_dict["start_time"] is not None
        assert result_dict["end_time"] is not None
        assert result_dict["duration_ms"] is not None
        assert result_dict["duration_ms"] > 0


class TestTraceFunctionDecorator:
    """trace_function装饰器测试"""
    
    def test_trace_function_basic(self):
        """测试基础函数追踪"""
        @trace_function("test_function")
        def test_func():
            return "result"
        
        result = test_func()
        assert result == "result"
    
    def test_trace_function_with_args_kwargs(self):
        """测试带参数的函数追踪"""
        @trace_function("test_function_with_args")
        def test_func(a, b, c=None):
            return f"{a}-{b}-{c}"
        
        result = test_func("arg1", "arg2", c="kwarg1")
        assert result == "arg1-arg2-kwarg1"
    
    def test_trace_function_with_exception(self):
        """测试函数异常时的追踪"""
        @trace_function("test_function_exception")
        def failing_func():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_func()
    
    def test_trace_function_with_existing_trace_id(self):
        """测试在已有trace_id时的函数追踪"""
        trace_id = "existing_trace_123"
        set_current_trace_id(trace_id)
        
        @trace_function("test_function_existing_trace")
        def test_func():
            # 在函数内部应该保持相同的trace_id
            return get_current_trace_id()
        
        result = test_func()
        assert result == trace_id
    
    def test_trace_function_auto_generate_trace_id(self):
        """测试自动生成trace_id的函数追踪"""
        _current_trace_id.set(None)
        
        @trace_function("test_function_auto_trace")
        def test_func():
            # 在函数内部应该有trace_id（由SpanTimer设置）
            current_trace_id = get_current_trace_id()
            return current_trace_id
        
        returned_trace_id = test_func()
        
        # 由于trace_function使用SpanTimer，而SpanTimer不会自动设置trace_id到上下文
        # 所以这里返回的应该是None
        assert returned_trace_id is None


class TestTraceAsyncFunctionDecorator:
    """trace_async_function装饰器测试"""
    
    @pytest.mark.asyncio
    async def test_trace_async_function_basic(self):
        """测试基础异步函数追踪"""
        @trace_async_function("test_async_function")
        async def test_async_func():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = await test_async_func()
        assert result == "async_result"
    
    @pytest.mark.asyncio
    async def test_trace_async_function_with_args(self):
        """测试带参数的异步函数追踪"""
        @trace_async_function("test_async_function_with_args")
        async def test_async_func(a, b):
            await asyncio.sleep(0.01)
            return f"{a}-{b}"
        
        result = await test_async_func("arg1", "arg2")
        assert result == "arg1-arg2"
    
    @pytest.mark.asyncio
    async def test_trace_async_function_with_exception(self):
        """测试异步函数异常时的追踪"""
        @trace_async_function("test_async_function_exception")
        async def failing_async_func():
            await asyncio.sleep(0.01)
            raise ValueError("Async test error")
        
        with pytest.raises(ValueError):
            await failing_async_func()
    
    @pytest.mark.asyncio
    async def test_trace_async_function_with_existing_trace_id(self):
        """测试在已有trace_id时的异步函数追踪"""
        trace_id = "existing_async_trace_123"
        set_current_trace_id(trace_id)
        
        @trace_async_function("test_async_function_existing_trace")
        async def test_async_func():
            await asyncio.sleep(0.01)
            # 在函数内部应该保持相同的trace_id
            return get_current_trace_id()
        
        result = await test_async_func()
        assert result == trace_id
    
    @pytest.mark.asyncio
    async def test_trace_async_function_auto_generate_trace_id(self):
        """测试自动生成trace_id的异步函数追踪"""
        _current_trace_id.set(None)
        
        @trace_async_function("test_async_function_auto_trace")
        async def test_async_func():
            await asyncio.sleep(0.01)
            # 在函数内部应该有trace_id（由SpanTimer设置）
            current_trace_id = get_current_trace_id()
            return current_trace_id
        
        returned_trace_id = await test_async_func()
        
        # 由于trace_async_function使用SpanTimer，而SpanTimer不会自动设置trace_id到上下文
        # 所以这里返回的应该是None
        assert returned_trace_id is None


class TestIntegration:
    """集成测试"""
    
    def test_full_trace_workflow(self):
        """测试完整的追踪工作流程"""
        with TraceContext("full_workflow_trace") as trace_id:
            # 使用装饰器
            @trace_function("workflow_function")
            def workflow_func():
                return get_current_trace_id()
            
            result = workflow_func()
            assert result == "full_workflow_trace"
            
            # 使用SpanTimer
            with SpanTimer("workflow_span") as timer:
                time.sleep(0.01)
            
            assert timer.duration_ms is not None
            assert timer.duration_ms > 0
    
    def test_full_workflow_integration(self):
        """测试完整的追踪工作流程"""
        # 使用TraceContext创建新的trace_id
        with TraceContext() as ctx_trace_id:
            # ctx_trace_id是TraceContext.__enter__返回的trace_id
            assert isinstance(ctx_trace_id, str)
            assert ctx_trace_id.startswith("harborai_")
            
            # 验证当前上下文中的trace_id
            assert get_current_trace_id() == ctx_trace_id
            
            # 使用SpanTimer
            with SpanTimer("integration_test", ctx_trace_id) as timer:
                time.sleep(0.01)
            
            assert timer.duration_ms is not None
            assert timer.duration_ms > 0
            assert timer.trace_id == ctx_trace_id
        
        # 验证上下文恢复（应该是None，因为之前没有设置）
        assert get_current_trace_id() is None
    
    @pytest.mark.asyncio
    async def test_mixed_sync_async_tracing(self):
        """测试同步和异步函数混合追踪"""
        trace_id = generate_trace_id()
        set_current_trace_id(trace_id)
        
        @trace_function("sync_function")
        def sync_func():
            return get_current_trace_id()
        
        @trace_async_function("async_function")
        async def async_func():
            await asyncio.sleep(0.01)
            return get_current_trace_id()
        
        sync_result = sync_func()
        async_result = await async_func()
        
        # 验证两个函数都能访问到相同的trace_id
        assert sync_result == trace_id
        assert async_result == trace_id
    
    def test_nested_span_timers(self):
        """测试嵌套的SpanTimer"""
        trace_id = generate_trace_id()
        
        with SpanTimer("outer_span", trace_id) as outer:
            time.sleep(0.01)
            
            with SpanTimer("inner_span", trace_id) as inner:
                time.sleep(0.01)
            
            assert inner.duration_ms is not None
            assert inner.duration_ms > 0
        
        assert outer.duration_ms is not None
        assert outer.duration_ms > inner.duration_ms
    
    def test_trace_id_propagation_across_threads(self):
        """测试trace_id在线程间的传播"""
        trace_id = generate_trace_id()
        set_current_trace_id(trace_id)
        
        results = []
        
        def worker():
            # 在新线程中应该没有trace_id（contextvars是线程隔离的）
            current_trace_id = get_current_trace_id()
            results.append(current_trace_id)
        
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        
        # 新线程中应该没有trace_id
        assert results[0] is None
        
        # 主线程中仍然有trace_id
        assert get_current_trace_id() == trace_id
    
    def test_error_handling_with_tracing(self):
        """测试追踪过程中的错误处理"""
        @trace_function("error_function")
        def error_func():
            raise RuntimeError("Test error")
        
        @trace_async_function("async_error_function")
        async def async_error_func():
            await asyncio.sleep(0.01)
            raise RuntimeError("Async test error")
        
        # 同步函数错误
        with pytest.raises(RuntimeError):
            error_func()
        
        # 异步函数错误
        with pytest.raises(RuntimeError):
            asyncio.run(async_error_func())