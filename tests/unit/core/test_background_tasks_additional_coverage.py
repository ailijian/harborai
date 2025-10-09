"""后台任务处理器额外覆盖率测试

功能：补充测试后台任务处理器的未覆盖代码路径
参数：各种边界条件和异常情况
返回：根据测试方法不同返回不同类型
异常：测试各种异常处理路径
边界：测试队列满、超时、重试等边界条件
假设：异步环境下的任务处理正确性
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
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


class TestBackgroundTaskProcessorAdditionalCoverage:
    """测试BackgroundTaskProcessor的额外覆盖率
    
    功能：验证未覆盖的代码路径
    参数：各种边界条件参数
    返回：根据方法不同返回不同类型
    异常：测试异常处理路径
    边界：测试各种边界条件
    假设：异步环境下正确运行
    """
    
    @pytest.fixture
    async def processor(self):
        """创建测试用的后台任务处理器"""
        processor = BackgroundTaskProcessor(max_workers=1, max_queue_size=5)
        yield processor
        # 清理：确保处理器停止
        if processor._running:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_execute_task_sync_function(self, processor):
        """测试执行同步函数任务
        
        功能：验证同步函数在线程池中的执行
        参数：同步函数和参数
        返回：无返回值（测试执行过程）
        异常：无特殊异常
        边界：测试同步函数执行路径
        假设：线程池正确执行同步函数
        """
        await processor.start()
        
        # 创建一个同步函数
        def sync_func(x, y):
            return x + y
        
        task = BackgroundTask(
            task_id="sync-task",
            func=sync_func,
            args=(1, 2),
            kwargs={}
        )
        
        # 执行任务
        await processor._execute_task(task, "test-worker")
        
        # 验证统计信息
        assert processor._stats['completed_tasks'] == 1
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_execute_task_async_function(self, processor):
        """测试执行异步函数任务
        
        功能：验证异步函数的直接执行
        参数：异步函数和参数
        返回：无返回值（测试执行过程）
        异常：无特殊异常
        边界：测试异步函数执行路径
        假设：异步函数正确执行
        """
        await processor.start()
        
        # 创建一个异步函数
        async def async_func(x, y):
            await asyncio.sleep(0.01)  # 模拟异步操作
            return x + y
        
        task = BackgroundTask(
            task_id="async-task",
            func=async_func,
            args=(3, 4),
            kwargs={}
        )
        
        # 执行任务
        await processor._execute_task(task, "test-worker")
        
        # 验证统计信息
        assert processor._stats['completed_tasks'] == 1
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_execute_task_with_exception_and_retry(self, processor):
        """测试任务执行异常和重试机制
        
        功能：验证任务执行失败时的重试逻辑
        参数：会抛出异常的函数
        返回：无返回值（测试重试过程）
        异常：测试异常处理和重试
        边界：测试重试次数限制
        假设：重试机制正确工作
        """
        await processor.start()
        
        # 创建一个会抛出异常的函数
        call_count = 0
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Test error {call_count}")
        
        task = BackgroundTask(
            task_id="failing-task",
            func=failing_func,
            args=(),
            kwargs={},
            max_retries=2
        )
        
        # 执行任务（第一次）
        await processor._execute_task(task, "test-worker")
        
        # 验证任务被重新提交到队列
        assert processor._task_queue.qsize() == 1
        assert processor._stats['retried_tasks'] == 1
        assert task.retry_count == 1
        
        # 再次执行任务（第二次重试）
        _, retry_task = await processor._task_queue.get()
        await processor._execute_task(retry_task, "test-worker")
        
        # 验证再次重试
        assert processor._task_queue.qsize() == 1
        assert processor._stats['retried_tasks'] == 2
        assert retry_task.retry_count == 2
        
        # 第三次执行（达到重试上限）
        _, final_task = await processor._task_queue.get()
        await processor._execute_task(final_task, "test-worker")
        
        # 验证任务最终失败
        assert processor._stats['failed_tasks'] == 1
        assert final_task.retry_count == 2  # 不再增加
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_execute_task_retry_queue_full(self, processor):
        """测试重试时队列满的情况
        
        功能：验证重试时队列满的处理逻辑
        参数：会抛出异常的函数和小队列
        返回：无返回值（测试队列满处理）
        异常：测试队列满异常处理
        边界：测试队列容量限制
        假设：队列满时正确处理重试失败
        """
        # 创建一个小队列的处理器
        small_processor = BackgroundTaskProcessor(max_workers=1, max_queue_size=1)
        await small_processor.start()
        
        # 先填满队列
        def dummy_func():
            time.sleep(0.1)  # 阻塞一段时间
        
        await small_processor.submit_task(dummy_func, task_id="blocking-task")
        
        # 创建一个会失败的任务
        def failing_func():
            raise ValueError("Test error")
        
        task = BackgroundTask(
            task_id="failing-task",
            func=failing_func,
            args=(),
            kwargs={},
            max_retries=1
        )
        
        # 执行失败的任务，重试时队列应该满了
        await small_processor._execute_task(task, "test-worker")
        
        # 验证任务直接标记为失败（因为重试队列满）
        assert small_processor._stats['failed_tasks'] == 1
        
        await small_processor.stop()
    
    @pytest.mark.asyncio
    async def test_worker_timeout_handling(self, processor):
        """测试工作协程的超时处理
        
        功能：验证工作协程在获取任务超时时的处理
        参数：空队列情况
        返回：无返回值（测试超时处理）
        异常：测试超时异常处理
        边界：测试队列空时的超时
        假设：超时时正确继续循环
        """
        await processor.start()
        
        # 模拟工作协程运行一小段时间然后停止
        await asyncio.sleep(0.1)  # 让工作协程运行
        
        # 停止处理器
        await processor.stop()
        
        # 验证工作协程正常停止
        assert not processor._running
        assert all(worker.cancelled() for worker in processor._workers)
    
    @pytest.mark.asyncio
    async def test_worker_exception_handling(self, processor):
        """测试工作协程的异常处理
        
        功能：验证工作协程在处理任务时发生异常的处理
        参数：会导致工作协程异常的情况
        返回：无返回值（测试异常处理）
        异常：测试工作协程异常处理
        边界：测试工作协程异常恢复
        假设：工作协程异常后能继续工作
        """
        await processor.start()
        
        # 使用mock来模拟_execute_task抛出异常
        original_execute = processor._execute_task
        call_count = 0
        
        async def mock_execute_task(task, worker_name):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated worker error")
            return await original_execute(task, worker_name)
        
        processor._execute_task = mock_execute_task
        
        # 提交一个任务
        def simple_func():
            return "success"
        
        await processor.submit_task(simple_func, task_id="test-task")
        
        # 等待任务处理
        await asyncio.sleep(0.2)
        
        # 验证工作协程仍在运行（异常被捕获）
        assert processor._running
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_get_stats(self, processor):
        """测试获取统计信息
        
        功能：验证统计信息的正确性
        参数：处理器状态
        返回：统计信息字典
        异常：无特殊异常
        边界：测试各种状态下的统计
        假设：统计信息准确反映状态
        """
        # 测试未启动状态
        stats = processor.get_stats()
        assert stats['running'] is False
        assert stats['workers'] == 0
        assert stats['queue_size'] == 0
        assert stats['total_tasks'] == 0
        
        # 启动处理器
        await processor.start()
        
        # 提交一些任务
        def dummy_func():
            pass
        
        await processor.submit_task(dummy_func, task_id="task1")
        await processor.submit_task(dummy_func, task_id="task2")
        
        # 获取统计信息
        stats = processor.get_stats()
        assert stats['running'] is True
        assert stats['workers'] == 1
        assert stats['total_tasks'] == 2
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_wait_for_completion_success(self, processor):
        """测试等待任务完成（成功情况）
        
        功能：验证等待所有任务完成的功能
        参数：超时时间
        返回：是否在超时前完成
        异常：无特殊异常
        边界：测试正常完成情况
        假设：任务能在超时前完成
        """
        await processor.start()
        
        # 提交一个快速任务
        def quick_func():
            return "done"
        
        await processor.submit_task(quick_func, task_id="quick-task")
        
        # 等待完成
        result = await processor.wait_for_completion(timeout=1.0)
        assert result is True
        
        await processor.stop()
    
    @pytest.mark.asyncio
    async def test_wait_for_completion_timeout(self, processor):
        """测试等待任务完成（超时情况）
        
        功能：验证等待任务完成时的超时处理
        参数：很短的超时时间
        返回：超时时返回False
        异常：测试超时异常处理
        边界：测试超时边界
        假设：超时时正确返回False
        """
        await processor.start()
        
        # 提交一个慢任务
        def slow_func():
            time.sleep(0.5)
            return "done"
        
        await processor.submit_task(slow_func, task_id="slow-task")
        
        # 等待完成（超时）
        result = await processor.wait_for_completion(timeout=0.1)
        assert result is False
        
        await processor.stop()


class TestGlobalFunctions:
    """测试全局函数
    
    功能：验证全局后台任务处理器函数
    参数：各种全局函数参数
    返回：根据函数不同返回不同类型
    异常：测试全局函数异常处理
    边界：测试全局状态管理
    假设：全局函数正确管理处理器实例
    """
    
    @pytest.mark.asyncio
    async def test_global_processor_lifecycle(self):
        """测试全局处理器生命周期
        
        功能：验证全局处理器的创建、启动、停止
        参数：无
        返回：无返回值（测试生命周期）
        异常：无特殊异常
        边界：测试全局状态管理
        假设：全局处理器正确管理
        """
        # 获取全局处理器
        processor1 = get_background_processor()
        processor2 = get_background_processor()
        
        # 验证单例模式
        assert processor1 is processor2
        
        # 启动全局处理器
        await start_background_processor()
        assert processor1._running
        
        # 停止全局处理器
        await stop_background_processor()
        
        # 验证处理器被重置
        processor3 = get_background_processor()
        assert processor3 is not processor1
    
    @pytest.mark.asyncio
    async def test_submit_background_task_function(self):
        """测试全局提交任务函数
        
        功能：验证全局提交任务函数的功能
        参数：任务函数和参数
        返回：是否成功提交
        异常：无特殊异常
        边界：测试全局任务提交
        假设：全局函数正确提交任务
        """
        # 启动全局处理器
        await start_background_processor()
        
        # 提交任务
        def test_func(x):
            return x * 2
        
        result = await submit_background_task(
            test_func, 5,
            task_id="global-task",
            priority=1,
            max_retries=2
        )
        
        assert result is True
        
        # 获取处理器并验证统计
        processor = get_background_processor()
        assert processor._stats['total_tasks'] == 1
        
        # 清理
        await stop_background_processor()