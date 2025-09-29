"""后台任务处理器

处理非关键路径的异步操作，如日志记录、统计更新等。
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import time

logger = logging.getLogger(__name__)


@dataclass
class BackgroundTask:
    """后台任务数据类"""
    task_id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0  # 优先级，数字越大优先级越高
    created_at: datetime = None
    max_retries: int = 3
    retry_count: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class BackgroundTaskProcessor:
    """后台任务处理器
    
    异步处理非关键路径的操作，避免阻塞主请求流程。
    """
    
    def __init__(self, max_workers: int = 4, max_queue_size: int = 1000):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bg_task")
        self._workers: List[asyncio.Task] = []
        self._running = False
        self._stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'retried_tasks': 0
        }
        
    async def start(self) -> None:
        """启动后台任务处理器"""
        if self._running:
            return
            
        self._running = True
        
        # 启动工作协程
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self._workers.append(worker)
            
        logger.info(f"后台任务处理器已启动，工作线程数: {self.max_workers}")
        
    async def stop(self) -> None:
        """停止后台任务处理器"""
        if not self._running:
            return
            
        self._running = False
        
        # 等待所有任务完成
        await self._task_queue.join()
        
        # 取消所有工作协程
        for worker in self._workers:
            worker.cancel()
            
        # 等待工作协程结束
        await asyncio.gather(*self._workers, return_exceptions=True)
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        logger.info("后台任务处理器已停止")
        
    async def submit_task(
        self,
        func: Callable,
        *args,
        task_id: Optional[str] = None,
        priority: int = 0,
        max_retries: int = 3,
        **kwargs
    ) -> bool:
        """提交后台任务
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            task_id: 任务ID，如果为None则自动生成
            priority: 优先级，数字越大优先级越高
            max_retries: 最大重试次数
            **kwargs: 函数关键字参数
            
        Returns:
            bool: 是否成功提交任务
        """
        if not self._running:
            logger.warning("后台任务处理器未启动，无法提交任务")
            return False
            
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
            
        task = BackgroundTask(
            task_id=task_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries
        )
        
        try:
            # 使用负优先级，因为PriorityQueue是最小堆
            await self._task_queue.put((-priority, task))
            self._stats['total_tasks'] += 1
            return True
        except asyncio.QueueFull:
            logger.warning(f"后台任务队列已满，丢弃任务: {task_id}")
            return False
            
    async def _worker(self, worker_name: str) -> None:
        """工作协程"""
        logger.debug(f"后台任务工作协程 {worker_name} 已启动")
        
        while self._running:
            try:
                # 获取任务，超时1秒
                try:
                    priority, task = await asyncio.wait_for(
                        self._task_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                    
                # 执行任务
                await self._execute_task(task, worker_name)
                
                # 标记任务完成
                self._task_queue.task_done()
                
            except asyncio.CancelledError:
                logger.debug(f"后台任务工作协程 {worker_name} 被取消")
                break
            except Exception as e:
                logger.error(f"后台任务工作协程 {worker_name} 发生错误: {e}")
                
        logger.debug(f"后台任务工作协程 {worker_name} 已停止")
        
    async def _execute_task(self, task: BackgroundTask, worker_name: str) -> None:
        """执行单个任务"""
        start_time = time.time()
        
        try:
            logger.debug(f"[{worker_name}] 开始执行任务: {task.task_id}")
            
            # 判断是否为异步函数
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    lambda: task.func(*task.args, **task.kwargs)
                )
                
            duration = time.time() - start_time
            logger.debug(f"[{worker_name}] 任务 {task.task_id} 执行成功，耗时: {duration:.3f}s")
            self._stats['completed_tasks'] += 1
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"[{worker_name}] 任务 {task.task_id} 执行失败: {e}，耗时: {duration:.3f}s")
            
            # 检查是否需要重试
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.info(f"[{worker_name}] 重试任务 {task.task_id}，第 {task.retry_count} 次重试")
                
                # 重新提交任务
                try:
                    await self._task_queue.put((-task.priority, task))
                    self._stats['retried_tasks'] += 1
                except asyncio.QueueFull:
                    logger.warning(f"重试任务 {task.task_id} 时队列已满")
                    self._stats['failed_tasks'] += 1
            else:
                logger.error(f"任务 {task.task_id} 重试次数已达上限，放弃执行")
                self._stats['failed_tasks'] += 1
                
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            'queue_size': self._task_queue.qsize(),
            'running': self._running,
            'workers': len(self._workers)
        }
        
    async def wait_for_completion(self, timeout: Optional[float] = None) -> bool:
        """等待所有任务完成
        
        Args:
            timeout: 超时时间（秒），None表示无限等待
            
        Returns:
            bool: 是否在超时前完成所有任务
        """
        try:
            await asyncio.wait_for(self._task_queue.join(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False


# 全局后台任务处理器实例
_background_processor: Optional[BackgroundTaskProcessor] = None


def get_background_processor() -> BackgroundTaskProcessor:
    """获取全局后台任务处理器实例"""
    global _background_processor
    if _background_processor is None:
        _background_processor = BackgroundTaskProcessor()
    return _background_processor


async def start_background_processor() -> None:
    """启动全局后台任务处理器"""
    processor = get_background_processor()
    await processor.start()


async def stop_background_processor() -> None:
    """停止全局后台任务处理器"""
    global _background_processor
    if _background_processor is not None:
        await _background_processor.stop()
        _background_processor = None


async def submit_background_task(
    func: Callable,
    *args,
    task_id: Optional[str] = None,
    priority: int = 0,
    max_retries: int = 3,
    **kwargs
) -> bool:
    """提交后台任务的便捷函数"""
    processor = get_background_processor()
    return await processor.submit_task(
        func, *args,
        task_id=task_id,
        priority=priority,
        max_retries=max_retries,
        **kwargs
    )