"""后台任务处理器简单测试

功能：简单测试后台任务处理器的基本功能，避免复杂异步操作
参数：基本的任务参数
返回：根据测试方法不同返回不同类型
异常：测试基本异常处理
边界：测试基本边界条件
假设：基本功能正确性
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from datetime import datetime

from harborai.core.background_tasks import (
    BackgroundTask,
    BackgroundTaskProcessor,
    get_background_processor,
    start_background_processor,
    stop_background_processor,
    submit_background_task
)


class TestBackgroundTaskSimple:
    """测试BackgroundTask的简单功能
    
    功能：验证后台任务数据类的基本功能
    参数：基本参数
    返回：BackgroundTask实例
    异常：无特殊异常
    边界：测试基本边界
    假设：数据类正确工作
    """
    
    def test_background_task_comparison(self):
        """测试后台任务比较功能
        
        功能：验证任务比较方法
        参数：两个任务实例
        返回：比较结果
        异常：无特殊异常
        边界：测试比较边界
        假设：比较方法正确实现
        """
        def dummy_func():
            pass
        
        # 创建两个任务，第二个稍晚创建
        task1 = BackgroundTask(
            task_id="task1",
            func=dummy_func,
            args=(),
            kwargs={}
        )
        
        time.sleep(0.001)  # 确保时间差异
        
        task2 = BackgroundTask(
            task_id="task2", 
            func=dummy_func,
            args=(),
            kwargs={}
        )
        
        # 验证比较
        assert task1 < task2
        assert not (task2 < task1)
        
        # 测试与非BackgroundTask对象比较
        assert task1.__lt__("not a task") == NotImplemented


class TestBackgroundTaskProcessorSimple:
    """测试BackgroundTaskProcessor的简单功能
    
    功能：验证后台任务处理器的基本功能
    参数：基本参数
    返回：根据方法不同返回不同类型
    异常：测试基本异常处理
    边界：测试基本边界条件
    假设：处理器基本功能正确
    """
    
    def test_processor_get_stats_basic(self):
        """测试获取基本统计信息
        
        功能：验证统计信息获取
        参数：处理器实例
        返回：统计信息字典
        异常：无特殊异常
        边界：测试基本状态
        假设：统计信息正确
        """
        processor = BackgroundTaskProcessor(max_workers=2, max_queue_size=10)
        
        stats = processor.get_stats()
        
        assert stats['total_tasks'] == 0
        assert stats['completed_tasks'] == 0
        assert stats['failed_tasks'] == 0
        assert stats['retried_tasks'] == 0
        assert stats['queue_size'] == 0
        assert stats['running'] is False
        assert stats['workers'] == 0
    
    @pytest.mark.asyncio
    async def test_processor_submit_task_not_running(self):
        """测试处理器未启动时提交任务
        
        功能：验证未启动状态下的任务提交
        参数：任务函数
        返回：False（提交失败）
        异常：无特殊异常
        边界：测试未启动状态
        假设：未启动时正确拒绝任务
        """
        processor = BackgroundTaskProcessor()
        
        def dummy_func():
            pass
        
        result = await processor.submit_task(dummy_func)
        assert result is False
        assert processor._stats['total_tasks'] == 0
    
    @pytest.mark.asyncio
    async def test_processor_start_stop_basic(self):
        """测试处理器基本启动停止
        
        功能：验证处理器启动停止功能
        参数：无
        返回：无返回值
        异常：无特殊异常
        边界：测试启动停止状态
        假设：启动停止正确工作
        """
        processor = BackgroundTaskProcessor(max_workers=1)
        
        # 测试启动
        await processor.start()
        assert processor._running is True
        assert len(processor._workers) == 1
        
        # 测试重复启动（应该无效果）
        await processor.start()
        assert len(processor._workers) == 1
        
        # 测试停止
        await processor.stop()
        assert processor._running is False
        
        # 测试重复停止（应该无效果）
        await processor.stop()
        assert processor._running is False
    
    @pytest.mark.asyncio
    async def test_processor_submit_task_with_auto_id(self):
        """测试自动生成任务ID
        
        功能：验证任务ID自动生成
        参数：无任务ID的任务
        返回：True（提交成功）
        异常：无特殊异常
        边界：测试ID自动生成
        假设：ID自动生成正确
        """
        processor = BackgroundTaskProcessor(max_workers=1)
        await processor.start()
        
        def dummy_func():
            pass
        
        result = await processor.submit_task(dummy_func)
        assert result is True
        assert processor._stats['total_tasks'] == 1
        
        await processor.stop()


class TestGlobalFunctionsSimple:
    """测试全局函数的简单功能
    
    功能：验证全局函数的基本功能
    参数：基本参数
    返回：根据函数不同返回不同类型
    异常：测试基本异常处理
    边界：测试基本边界条件
    假设：全局函数正确工作
    """
    
    def test_get_background_processor_singleton(self):
        """测试全局处理器单例模式
        
        功能：验证全局处理器单例
        参数：无
        返回：处理器实例
        异常：无特殊异常
        边界：测试单例模式
        假设：单例模式正确实现
        """
        # 重置全局处理器
        import harborai.core.background_tasks
        harborai.core.background_tasks._background_processor = None
        
        processor1 = get_background_processor()
        processor2 = get_background_processor()
        
        assert processor1 is processor2
        assert isinstance(processor1, BackgroundTaskProcessor)
    
    @pytest.mark.asyncio
    async def test_global_processor_lifecycle_basic(self):
        """测试全局处理器基本生命周期
        
        功能：验证全局处理器生命周期管理
        参数：无
        返回：无返回值
        异常：无特殊异常
        边界：测试生命周期管理
        假设：生命周期管理正确
        """
        # 重置全局处理器
        import harborai.core.background_tasks
        harborai.core.background_tasks._background_processor = None
        
        # 启动全局处理器
        await start_background_processor()
        processor = get_background_processor()
        assert processor._running is True
        
        # 停止全局处理器
        await stop_background_processor()
        
        # 验证处理器被重置
        new_processor = get_background_processor()
        assert new_processor is not processor
    
    @pytest.mark.asyncio
    async def test_submit_background_task_basic(self):
        """测试全局任务提交基本功能
        
        功能：验证全局任务提交
        参数：任务函数和参数
        返回：True（提交成功）
        异常：无特殊异常
        边界：测试基本任务提交
        假设：全局任务提交正确
        """
        # 重置全局处理器
        import harborai.core.background_tasks
        harborai.core.background_tasks._background_processor = None
        
        # 启动全局处理器
        await start_background_processor()
        
        def test_func(x):
            return x * 2
        
        result = await submit_background_task(
            test_func, 5,
            task_id="global-task",
            priority=1,
            max_retries=2
        )
        
        assert result is True
        
        processor = get_background_processor()
        assert processor._stats['total_tasks'] == 1
        
        # 清理
        await stop_background_processor()