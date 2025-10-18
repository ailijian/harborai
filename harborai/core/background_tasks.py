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
    
    def __lt__(self, other):
        """支持优先级队列比较"""
        if not isinstance(other, BackgroundTask):
            return NotImplemented
        # 按创建时间排序，确保相同优先级的任务按FIFO顺序执行
        return self.created_at < other.created_at


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
        
    async def stop(self, timeout: float = 5.0) -> None:
        """停止后台任务处理器
        
        Args:
            timeout: 等待任务完成的超时时间（秒）
        """
        if not self._running:
            return
            
        self._running = False
        logger.info("正在停止后台任务处理器...")
        
        # 第一步：尝试优雅停止 - 等待当前任务完成
        try:
            await asyncio.wait_for(self._task_queue.join(), timeout=timeout)
            logger.info("所有任务已完成")
        except asyncio.TimeoutError:
            logger.warning(f"等待任务完成超时（{timeout}秒），开始强制停止")
        
        # 第二步：强制停止 - 取消所有工作协程
        for worker in self._workers:
            if not worker.cancelled():
                worker.cancel()
        
        # 第三步：等待工作协程结束
        if self._workers:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._workers, return_exceptions=True),
                    timeout=2.0
                )
            except asyncio.TimeoutError:
                logger.warning("等待工作协程结束超时，强制清理")
        
        # 第四步：清空队列并平衡计数器
        cleared_count = 0
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                self._task_queue.task_done()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        
        if cleared_count > 0:
            logger.info(f"清理了 {cleared_count} 个未完成的任务")
        
        # 第五步：清理资源
        self._workers.clear()
        self._executor.shutdown(wait=False)
        
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
            # 使用 put_nowait 来立即检查队列是否满
            self._task_queue.put_nowait((-priority, task))
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
                    logger.debug(f"[{worker_name}] 获取到任务: {task.task_id}, 重试次数: {task.retry_count}")
                except asyncio.TimeoutError:
                    continue
                    
                # 执行任务并处理重试逻辑
                try:
                    await self._execute_task_with_retry(task, worker_name)
                except Exception as e:
                    logger.error(f"[{worker_name}] 执行任务 {task.task_id} 时发生未处理异常: {e}")
                    self._stats['failed_tasks'] += 1
                
                # 总是调用task_done()，因为我们从队列中取出了一个任务
                self._task_queue.task_done()
                
            except asyncio.CancelledError:
                logger.debug(f"后台任务工作协程 {worker_name} 被取消")
                break
            except Exception as e:
                logger.error(f"后台任务工作协程 {worker_name} 发生错误: {e}")
                
        logger.debug(f"后台任务工作协程 {worker_name} 已停止")
        
    async def _execute_task_with_retry(self, task: BackgroundTask, worker_name: str) -> None:
        """执行任务并处理重试逻辑"""
        max_attempts = task.max_retries + 1  # 原始执行 + 重试次数
        
        for attempt in range(max_attempts):
            start_time = time.time()
            
            try:
                logger.debug(f"[{worker_name}] 开始执行任务: {task.task_id} (第{attempt + 1}次尝试)")
                
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
                
                # 如果有重试，记录重试统计
                if attempt > 0:
                    self._stats['retried_tasks'] += attempt
                    
                return  # 任务成功完成
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"[{worker_name}] 任务 {task.task_id} 执行失败: {e}，耗时: {duration:.3f}s")
                
                # 如果还有重试机会，继续下一次尝试
                if attempt < max_attempts - 1:
                    logger.info(f"[{worker_name}] 重试任务 {task.task_id}，第 {attempt + 1} 次重试")
                    # 短暂延迟后重试
                    await asyncio.sleep(0.1 * (attempt + 1))  # 递增延迟
                else:
                    # 重试次数用尽，任务最终失败
                    logger.error(f"任务 {task.task_id} 重试次数已达上限，放弃执行")
                    self._stats['failed_tasks'] += 1
                    if attempt > 0:
                        self._stats['retried_tasks'] += attempt
                
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            'queue_size': self._task_queue.qsize(),
            'running': self._running,
            'workers': len(self._workers)
        }
        
    async def wait_for_completion(self, timeout: float = 30.0) -> bool:
        """等待所有任务完成
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否在超时前完成所有任务
        """
        if not self._running:
            logger.debug("处理器未运行，直接返回完成状态")
            return True
        
        # 如果队列为空，直接返回
        if self._task_queue.empty():
            logger.debug("任务队列为空，直接返回完成状态")
            return True
            
        try:
            await asyncio.wait_for(self._task_queue.join(), timeout=timeout)
            logger.debug("所有任务已完成")
            return True
        except asyncio.TimeoutError:
            logger.warning(f"等待任务完成超时（{timeout}秒），当前队列大小: {self._task_queue.qsize()}")
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