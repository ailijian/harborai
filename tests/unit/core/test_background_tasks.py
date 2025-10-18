"""后台任务处理器测试

功能：测试后台任务处理器的所有功能
参数：任务函数、优先级、重试次数等
返回：根据方法不同返回不同类型
异常：asyncio.QueueFull、asyncio.TimeoutError等
边界：测试队列满、超时、重试等边界条件
假设：异步环境下的任务处理
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import Any, Dict

from harborai.core.background_tasks import (
    BackgroundTask,
    BackgroundTaskProcessor,
    get_background_processor,
    start_background_processor,
    stop_background_processor,
    submit_background_task
)


class TestBackgroundTask:
    """测试BackgroundTask数据类
    
    功能：验证后台任务数据类的创建和属性
    参数：任务ID、函数、参数、优先级等
    返回：BackgroundTask实例
    异常：无特殊异常
    边界：测试默认值和自动生成时间
    假设：数据类正确初始化
    """
    
    def test_background_task_creation(self):
        """测试后台任务创建"""
        def dummy_func():
            pass
            
        task = BackgroundTask(
            task_id="test-task",
            func=dummy_func,
            args=(1, 2),
            kwargs={"key": "value"},
            priority=5,
            max_retries=3
        )
        
        assert task.task_id == "test-task"
        assert task.func == dummy_func
        assert task.args == (1, 2)
        assert task.kwargs == {"key": "value"}
        assert task.priority == 5
        assert task.max_retries == 3
        assert task.retry_count == 0
        assert isinstance(task.created_at, datetime)
    
    def test_background_task_default_values(self):
        """测试后台任务默认值"""
        def dummy_func():
            pass
            
        task = BackgroundTask(
            task_id="test-task",
            func=dummy_func,
            args=(),
            kwargs={}
        )
        
        assert task.priority == 0
        assert task.max_retries == 3
        assert task.retry_count == 0
        assert task.created_at is not None
    
    def test_background_task_auto_created_at(self):
        """测试自动生成创建时间"""
        def dummy_func():
            pass
            
        before = datetime.now()
        task = BackgroundTask(
            task_id="test-task",
            func=dummy_func,
            args=(),
            kwargs={}
        )
        after = datetime.now()
        
        assert before <= task.created_at <= after


class TestBackgroundTaskProcessor:
    """测试BackgroundTaskProcessor类
    
    功能：验证后台任务处理器的所有功能
    参数：最大工作线程数、队列大小等
    返回：根据方法不同返回不同类型
    异常：asyncio.QueueFull、asyncio.TimeoutError等
    边界：测试启动停止、队列满、超时等情况
    假设：异步环境下正确运行
    """
    
    @pytest.fixture
    async def processor(self):
        """创建测试用的后台任务处理器"""
        processor = BackgroundTaskProcessor(max_workers=1, max_queue_size=5)  # 简化配置
        yield processor
        # 强制清理：确保处理器停止
        try:
            await asyncio.wait_for(processor.stop(timeout=0.5), timeout=1.0)
        except asyncio.TimeoutError:
            # 强制清理
            processor._running = False
            for worker in processor._workers:
                if not worker.cancelled():
                    worker.cancel()
            # 清空队列
            while not processor._task_queue.empty():
                try:
                    processor._task_queue.get_nowait()
                    processor._task_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            processor._workers.clear()
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self, processor):
        """测试处理器初始化"""
        assert processor.max_workers == 1
        assert processor.max_queue_size == 5
        assert not processor._running
        assert len(processor._workers) == 0
        assert processor._stats['total_tasks'] == 0
        assert processor._stats['completed_tasks'] == 0
        assert processor._stats['failed_tasks'] == 0
        assert processor._stats['retried_tasks'] == 0
    
    @pytest.mark.asyncio
    async def test_processor_start_stop(self, processor):
        """测试处理器启动和停止"""
        # 测试启动
        await processor.start()
        assert processor._running
        assert len(processor._workers) == 1
        
        # 测试停止
        await asyncio.wait_for(processor.stop(timeout=0.5), timeout=1.0)
        assert not processor._running
        assert len(processor._workers) == 0
        
        # 测试重复停止（应该无效果）
        await processor.stop()
        assert not processor._running
    
    @pytest.mark.asyncio
    async def test_submit_task_success(self, processor):
        """测试成功提交任务"""
        await processor.start()
        
        def dummy_func():
            return "success"
        
        result = await processor.submit_task(dummy_func, task_id="test-task")
        assert result is True
        assert processor._stats['total_tasks'] == 1
        
        # 等待任务完成
        completed = await asyncio.wait_for(
            processor.wait_for_completion(timeout=0.5), 
            timeout=1.0
        )
        assert completed is True
    
    @pytest.mark.asyncio
    async def test_submit_task_not_running(self, processor):
        """测试处理器未启动时提交任务"""
        def dummy_func():
            pass
        
        result = await processor.submit_task(dummy_func)
        assert result is False
        assert processor._stats['total_tasks'] == 0
    
    @pytest.mark.asyncio
    async def test_submit_task_auto_id(self, processor):
        """测试自动生成任务ID"""
        await processor.start()
        
        def dummy_func():
            return "auto_id_test"
        
        result = await processor.submit_task(dummy_func)
        assert result is True
        assert processor._stats['total_tasks'] == 1
        
        # 等待任务完成
        completed = await asyncio.wait_for(
            processor.wait_for_completion(timeout=0.5), 
            timeout=1.0
        )
        assert completed is True
    
    @pytest.mark.asyncio
    async def test_submit_task_queue_full(self, processor):
        """测试队列满时提交任务"""
        # 创建一个小队列的处理器
        small_processor = BackgroundTaskProcessor(max_workers=1, max_queue_size=1)
        
        try:
            # 不启动处理器，只设置运行状态
            small_processor._running = True
            
            def simple_task():
                return "test"
            
            # 填满队列
            result1 = await small_processor.submit_task(simple_task, task_id="task1")
            assert result1 is True
            
            # 队列满了，第二个任务应该失败
            result2 = await small_processor.submit_task(simple_task, task_id="task2")
            assert result2 is False
            
        finally:
            # 清理
            small_processor._running = False
            while not small_processor._task_queue.empty():
                try:
                    small_processor._task_queue.get_nowait()
                    small_processor._task_queue.task_done()
                except asyncio.QueueEmpty:
                    break
    
    @pytest.mark.asyncio
    async def test_execute_sync_function(self, processor):
        """测试执行同步函数"""
        await processor.start()
        
        result_container = []
        
        def sync_func(value):
            result_container.append(value)
            return value
        
        await processor.submit_task(sync_func, 42)
        
        # 等待任务完成
        completed = await asyncio.wait_for(
            processor.wait_for_completion(timeout=0.5), 
            timeout=1.0
        )
        assert completed is True
        assert len(result_container) == 1
        assert result_container[0] == 42
        assert processor._stats['completed_tasks'] == 1
    
    @pytest.mark.asyncio
    async def test_execute_async_function(self, processor):
        """测试执行异步函数"""
        await processor.start()
        
        result_container = []
        
        async def async_func(value):
            await asyncio.sleep(0.01)  # 模拟异步操作
            result_container.append(value)
            return value
        
        await processor.submit_task(async_func, 42)
        
        # 等待任务完成
        completed = await asyncio.wait_for(
            processor.wait_for_completion(timeout=0.5), 
            timeout=1.0
        )
        assert completed is True
        assert len(result_container) == 1
        assert result_container[0] == 42
        assert processor._stats['completed_tasks'] == 1
    
    @pytest.mark.asyncio
    async def test_task_failure_and_retry(self, processor):
        """测试任务失败和重试"""
        await processor.start()
        
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # 第一次失败，第二次成功
                raise Exception("Simulated failure")
            return "success"
        
        await processor.submit_task(failing_func, max_retries=1)  # 最多重试1次
        
        # 等待任务完成（包括重试）
        completed = await asyncio.wait_for(
            processor.wait_for_completion(timeout=0.5), 
            timeout=1.0
        )
        assert completed is True
        assert call_count == 2  # 原始调用失败 + 1次重试成功
        assert processor._stats['completed_tasks'] == 1
        assert processor._stats['retried_tasks'] == 1
    
    @pytest.mark.asyncio
    async def test_task_max_retries_exceeded(self, processor):
        """测试超过最大重试次数"""
        await processor.start()
        
        call_count = 0
        
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")
        
        await processor.submit_task(always_failing_func, max_retries=1)
        
        # 等待任务完成（包括重试）
        completed = await asyncio.wait_for(
            processor.wait_for_completion(timeout=0.5), 
            timeout=1.0
        )
        assert completed is True  # 任务完成（虽然失败）
        assert call_count == 2  # 原始调用 + 1次重试
        assert processor._stats['failed_tasks'] == 1
        assert processor._stats['retried_tasks'] == 1
    
    @pytest.mark.asyncio
    async def test_task_priority_ordering(self, processor):
        """测试任务优先级排序"""
        await processor.start()
        
        execution_order = []
        
        def task_func(task_name):
            execution_order.append(task_name)
        
        # 提交不同优先级的任务（简化为两个任务）
        await processor.submit_task(task_func, "low", priority=1)
        await processor.submit_task(task_func, "high", priority=10)
        
        # 等待所有任务完成
        completed = await asyncio.wait_for(
            processor.wait_for_completion(timeout=0.5), 
            timeout=1.0
        )
        assert completed is True
        
        # 高优先级任务应该先执行
        assert execution_order[0] == "high"
        assert execution_order[1] == "low"
    
    @pytest.mark.asyncio
    async def test_get_stats(self, processor):
        """测试获取统计信息"""
        stats = processor.get_stats()
        
        expected_keys = [
            'total_tasks', 'completed_tasks', 'failed_tasks', 'retried_tasks',
            'queue_size', 'running', 'workers'
        ]
        
        for key in expected_keys:
            assert key in stats
        
        assert stats['running'] is False
        assert stats['workers'] == 0
        assert stats['queue_size'] == 0
    
    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self, processor):
        """测试等待完成超时"""
        await processor.start()
        
        async def slow_task():
            await asyncio.sleep(0.1)  # 短暂延迟
        
        await processor.submit_task(slow_task)
        
        # 短超时应该返回False
        result = await asyncio.wait_for(
            processor.wait_for_completion(timeout=0.01), 
            timeout=0.5
        )
        assert result is False
        
        # 等待任务实际完成
        completed = await asyncio.wait_for(
            processor.wait_for_completion(timeout=0.5), 
            timeout=1.0
        )
        assert completed is True
    
    @pytest.mark.asyncio
    async def test_wait_for_completion_success(self, processor):
        """测试等待完成成功"""
        await processor.start()
        
        def quick_task():
            return "done"
        
        await processor.submit_task(quick_task)
        
        # 足够的超时应该返回True
        result = await asyncio.wait_for(
            processor.wait_for_completion(timeout=0.5), 
            timeout=1.0
        )
        assert result is True


class TestGlobalFunctions:
    """测试全局函数"""
    
    def test_get_background_processor_singleton(self):
        """测试全局处理器单例模式"""
        processor1 = get_background_processor()
        processor2 = get_background_processor()
        
        assert processor1 is processor2
        assert isinstance(processor1, BackgroundTaskProcessor)
    
    @pytest.mark.asyncio
    async def test_submit_background_task(self):
        """测试提交后台任务便捷函数"""
        result_container = []
        
        def test_func(value):
            result_container.append(value)
        
        # 启动处理器
        await start_background_processor()
        
        try:
            # 提交任务
            result = await submit_background_task(test_func, 42, task_id="test-global")
            assert result is True
            
            # 等待任务完成
            processor = get_background_processor()
            completed = await asyncio.wait_for(
                processor.wait_for_completion(timeout=0.5), 
                timeout=1.0
            )
            assert completed is True
            assert len(result_container) == 1
            assert result_container[0] == 42
            
        finally:
            # 清理
            await stop_background_processor()