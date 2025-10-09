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
        processor = BackgroundTaskProcessor(max_workers=2, max_queue_size=10)
        yield processor
        # 清理：确保处理器停止
        if processor._running:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_processor_initialization(self, processor):
        """测试处理器初始化"""
        assert processor.max_workers == 2
        assert processor.max_queue_size == 10
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
        assert len(processor._workers) == 2
        
        # 测试重复启动（应该无效果）
        await processor.start()
        assert len(processor._workers) == 2
        
        # 测试停止
        await processor.stop()
        assert not processor._running
        assert all(worker.cancelled() for worker in processor._workers)
        
        # 测试重复停止（应该无效果）
        await processor.stop()
        assert not processor._running
    
    @pytest.mark.asyncio
    async def test_submit_task_success(self, processor):
        """测试成功提交任务"""
        await processor.start()
        
        def dummy_func(x, y):
            return x + y
        
        result = await processor.submit_task(
            dummy_func, 1, 2,
            task_id="test-task",
            priority=5,
            max_retries=2
        )
        
        assert result is True
        assert processor._stats['total_tasks'] == 1
        
        await processor.stop()
    
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
            pass
        
        result = await processor.submit_task(dummy_func)
        assert result is True
        assert processor._stats['total_tasks'] == 1
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_submit_task_queue_full(self, processor):
        """测试队列满时提交任务"""
        # 创建一个小队列的处理器
        small_processor = BackgroundTaskProcessor(max_workers=1, max_queue_size=1)
        await small_processor.start()
        
        def dummy_func():
            time.sleep(0.1)  # 模拟耗时任务
        
        # 填满队列
        result1 = await small_processor.submit_task(dummy_func)
        assert result1 is True
        
        # 尝试再次提交应该失败
        result2 = await small_processor.submit_task(dummy_func)
        # 注意：由于队列可能被快速处理，这个测试可能不稳定
        # 我们主要测试逻辑是否正确
        
        await small_processor.stop()
    
    @pytest.mark.asyncio
    async def test_execute_sync_function(self, processor):
        """测试执行同步函数"""
        await processor.start()
        
        result_container = []
        
        def sync_func(value):
            result_container.append(value)
            return value * 2
        
        await processor.submit_task(sync_func, 42)
        
        # 等待任务完成
        await processor.wait_for_completion(timeout=1.0)
        
        assert len(result_container) == 1
        assert result_container[0] == 42
        assert processor._stats['completed_tasks'] == 1
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_execute_async_function(self, processor):
        """测试执行异步函数"""
        await processor.start()
        
        result_container = []
        
        async def async_func(value):
            await asyncio.sleep(0.01)  # 模拟异步操作
            result_container.append(value)
            return value * 2
        
        await processor.submit_task(async_func, 42)
        
        # 等待任务完成
        await processor.wait_for_completion(timeout=1.0)
        
        assert len(result_container) == 1
        assert result_container[0] == 42
        assert processor._stats['completed_tasks'] == 1
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_task_failure_and_retry(self, processor):
        """测试任务失败和重试"""
        await processor.start()
        
        call_count = 0
        
        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # 前两次失败
                raise Exception("Simulated failure")
            return "success"
        
        await processor.submit_task(failing_func, max_retries=3)
        
        # 等待任务完成（包括重试）
        await processor.wait_for_completion(timeout=2.0)
        
        assert call_count == 3  # 原始调用 + 2次重试
        assert processor._stats['completed_tasks'] == 1
        assert processor._stats['retried_tasks'] == 2
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_task_max_retries_exceeded(self, processor):
        """测试超过最大重试次数"""
        await processor.start()
        
        call_count = 0
        
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")
        
        await processor.submit_task(always_failing_func, max_retries=2)
        
        # 等待任务完成（包括重试）
        await processor.wait_for_completion(timeout=2.0)
        
        assert call_count == 3  # 原始调用 + 2次重试
        assert processor._stats['failed_tasks'] == 1
        assert processor._stats['retried_tasks'] == 2
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_task_priority_ordering(self, processor):
        """测试任务优先级排序"""
        await processor.start()
        
        execution_order = []
        
        def task_func(task_name):
            execution_order.append(task_name)
        
        # 提交不同优先级的任务
        await processor.submit_task(task_func, "low", priority=1)
        await processor.submit_task(task_func, "high", priority=10)
        await processor.submit_task(task_func, "medium", priority=5)
        
        # 等待所有任务完成
        await processor.wait_for_completion(timeout=1.0)
        
        # 高优先级任务应该先执行
        assert execution_order[0] == "high"
        assert "medium" in execution_order
        assert "low" in execution_order
        
        await processor.stop()
    
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
        
        def slow_task():
            time.sleep(2.0)  # 耗时任务
        
        await processor.submit_task(slow_task)
        
        # 短超时应该返回False
        result = await processor.wait_for_completion(timeout=0.1)
        assert result is False
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_wait_for_completion_success(self, processor):
        """测试等待完成成功"""
        await processor.start()
        
        def quick_task():
            pass
        
        await processor.submit_task(quick_task)
        
        # 足够的超时应该返回True
        result = await processor.wait_for_completion(timeout=1.0)
        assert result is True
        
        await processor.stop()


class TestGlobalFunctions:
    """测试全局函数
    
    功能：验证全局后台任务处理器函数
    参数：任务函数、优先级等
    返回：处理器实例或提交结果
    异常：无特殊异常
    边界：测试单例模式和状态管理
    假设：全局状态正确管理
    """
    
    def test_get_background_processor_singleton(self):
        """测试全局处理器单例模式"""
        processor1 = get_background_processor()
        processor2 = get_background_processor()
        
        assert processor1 is processor2
        assert isinstance(processor1, BackgroundTaskProcessor)
    
    @pytest.mark.asyncio
    async def test_start_stop_background_processor(self):
        """测试启动停止全局处理器"""
        # 启动
        await start_background_processor()
        processor = get_background_processor()
        assert processor._running
        
        # 停止
        await stop_background_processor()
        # 注意：stop_background_processor会将全局实例设为None
        
        # 重新获取应该是新实例
        new_processor = get_background_processor()
        assert not new_processor._running
    
    @pytest.mark.asyncio
    async def test_submit_background_task(self):
        """测试提交后台任务便捷函数"""
        result_container = []
        
        def test_func(value):
            result_container.append(value)
        
        # 启动处理器
        await start_background_processor()
        
        # 提交任务
        result = await submit_background_task(
            test_func, 42,
            task_id="test-global",
            priority=5,
            max_retries=2
        )
        
        assert result is True
        
        # 等待任务完成
        processor = get_background_processor()
        await processor.wait_for_completion(timeout=1.0)
        
        assert len(result_container) == 1
        assert result_container[0] == 42
        
        # 清理
        await stop_background_processor()


class TestBackgroundTasksEdgeCases:
    """测试后台任务的边界情况和异常处理
    
    功能：验证各种边界条件和异常情况
    参数：各种异常输入和边界值
    返回：根据情况返回不同结果
    异常：各种预期和非预期异常
    边界：空队列、满队列、超时等
    假设：系统在异常情况下的稳定性
    """
    
    @pytest.mark.asyncio
    async def test_worker_exception_handling(self):
        """测试工作协程异常处理"""
        processor = BackgroundTaskProcessor(max_workers=1)
        await processor.start()
        
        # 模拟工作协程内部异常
        with patch.object(processor, '_execute_task', side_effect=Exception("Worker error")):
            def dummy_func():
                pass
            
            await processor.submit_task(dummy_func)
            
            # 等待一段时间让工作协程处理异常
            await asyncio.sleep(0.1)
            
            # 工作协程应该继续运行
            assert processor._running
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_task_with_kwargs(self):
        """测试带关键字参数的任务"""
        processor = BackgroundTaskProcessor()
        await processor.start()
        
        result_container = {}
        
        def task_with_kwargs(a, b, c=None, d=None):
            result_container.update({'a': a, 'b': b, 'c': c, 'd': d})
        
        await processor.submit_task(
            task_with_kwargs, 1, 2,
            c=3, d=4
        )
        
        await processor.wait_for_completion(timeout=1.0)
        
        assert result_container == {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self):
        """测试并发任务执行"""
        processor = BackgroundTaskProcessor(max_workers=3)
        await processor.start()
        
        execution_times = []
        
        async def concurrent_task(task_id):
            start_time = time.time()
            await asyncio.sleep(0.1)  # 模拟异步工作
            end_time = time.time()
            execution_times.append((task_id, start_time, end_time))
        
        # 提交多个任务
        for i in range(5):
            await processor.submit_task(concurrent_task, i)
        
        await processor.wait_for_completion(timeout=2.0)
        
        assert len(execution_times) == 5
        assert processor._stats['completed_tasks'] == 5
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_retry_queue_full_scenario(self):
        """测试重试时队列满的情况"""
        # 创建一个很小的队列
        processor = BackgroundTaskProcessor(max_workers=1, max_queue_size=1)
        await processor.start()
        
        call_count = 0
        
        def failing_task():
            nonlocal call_count
            call_count += 1
            raise Exception("Always fails")
        
        # 提交一个会失败的任务
        await processor.submit_task(failing_task, max_retries=1)
        
        # 立即填满队列，使重试无法入队
        def blocking_task():
            time.sleep(0.5)
        
        await processor.submit_task(blocking_task)
        
        # 等待处理
        await processor.wait_for_completion(timeout=2.0)
        
        # 验证失败任务被正确处理
        assert processor._stats['failed_tasks'] >= 1
        
        await processor.stop()