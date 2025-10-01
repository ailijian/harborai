"""异步成本追踪模块

提供非阻塞的成本追踪功能，通过后台任务处理成本计算和记录。
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging

from .cost_tracking import (
    TokenUsage, CostBreakdown, ApiCall, CostTracker,
    TokenCounter, PricingCalculator, BudgetManager
)
from decimal import Decimal
import uuid
from ..config.settings import get_settings
from ..config.performance import get_performance_config

logger = logging.getLogger(__name__)


class AsyncCostTracker:
    """异步成本追踪器
    
    通过后台任务异步处理成本计算，避免阻塞主请求流程。
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.perf_config = get_performance_config()
        self._sync_tracker = CostTracker()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cost_tracker")
        self._pending_calls: List[Dict[str, Any]] = []
        self._batch_size = 10
        self._batch_timeout = 5.0  # 秒
        self._last_batch_time = time.time()
        self._processing_lock = asyncio.Lock()
        
    async def track_api_call_async(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        cost: float,
        duration: float,
        success: bool = True,
        user_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        **metadata
    ) -> None:
        """异步追踪API调用
        
        Args:
            model: 模型名称
            provider: 提供商
            input_tokens: 输入token数
            output_tokens: 输出token数
            cost: 成本
            duration: 持续时间
            success: 是否成功
            user_id: 用户ID
            trace_id: 追踪ID
            **metadata: 其他元数据
        """
        call_data = {
            'model': model,
            'provider': provider,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost,
            'duration': duration,
            'success': success,
            'user_id': user_id,
            'trace_id': trace_id,
            'timestamp': datetime.now(),
            'metadata': metadata
        }
        
        # 检查性能配置是否启用成本追踪
        middleware_config = self.perf_config.get_middleware_config()
        if not middleware_config.get('cost_tracking_middleware', True):
            return
            
        # 如果启用快速路径且跳过成本追踪，直接返回
        if self.perf_config.should_use_fast_path(model=model):
            # 检查是否应该跳过成本追踪
            if self.settings.fast_path_skip_cost_tracking:
                return
            
        # 添加到待处理队列
        async with self._processing_lock:
            self._pending_calls.append(call_data)
            
            # 检查是否需要批量处理
            current_time = time.time()
            should_process = (
                len(self._pending_calls) >= self._batch_size or
                current_time - self._last_batch_time >= self._batch_timeout
            )
            
            if should_process:
                await self._process_batch()
                
    async def _process_batch(self) -> None:
        """批量处理待处理的API调用"""
        if not self._pending_calls:
            return
            
        # 复制并清空待处理列表
        calls_to_process = self._pending_calls.copy()
        self._pending_calls.clear()
        self._last_batch_time = time.time()
        
        # 在后台线程中处理
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                self._executor,
                self._process_calls_sync,
                calls_to_process
            )
        except Exception as e:
            logger.error(f"批量处理成本追踪失败: {e}")
            # 如果处理失败，将调用重新加入队列
            async with self._processing_lock:
                self._pending_calls.extend(calls_to_process)
                
    def _process_calls_sync(self, calls: List[Dict[str, Any]]) -> None:
        """同步处理API调用列表"""
        for call_data in calls:
            try:
                # 创建TokenUsage对象
                input_tokens = call_data.get('input_tokens', 0)
                output_tokens = call_data.get('output_tokens', 0)
                token_usage = TokenUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens
                )
                
                # 创建CostBreakdown对象
                cost = call_data.get('cost', 0.0)  # 如果没有cost字段，默认为0.0
                cost_breakdown = CostBreakdown(
                    input_cost=Decimal(str(cost * 0.6)),  # 假设60%为输入成本
                    output_cost=Decimal(str(cost * 0.4)),  # 假设40%为输出成本
                    currency="USD"
                )
                
                # 创建ApiCall对象
                api_call = ApiCall(
                    id=str(uuid.uuid4()),
                    timestamp=call_data.get('timestamp', datetime.now()),
                    provider=call_data.get('provider', 'unknown'),
                    model=call_data.get('model', 'unknown'),
                    endpoint="/chat/completions",  # 默认端点
                    token_usage=token_usage,
                    cost_breakdown=cost_breakdown,
                    request_size=1024,  # 默认请求大小
                    response_size=512,  # 默认响应大小
                    duration=call_data.get('duration', 0.0),
                    status="success" if call_data.get('success', True) else "error",
                    user_id=call_data.get('user_id', 'unknown'),
                    tags=call_data.get('metadata', {})
                )
                
                # 使用同步追踪器记录
                self._sync_tracker.api_calls.append(api_call)
                self._sync_tracker._update_cost_stats(api_call)
                
            except Exception as e:
                logger.error(f"处理单个API调用失败: {e}")
                
    async def get_cost_summary(self) -> Dict[str, Any]:
        """获取成本摘要"""
        # 确保所有待处理的调用都被处理
        await self.flush_pending()
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._sync_tracker.get_cost_summary
        )
        
    async def flush_pending(self) -> None:
        """刷新所有待处理的调用"""
        async with self._processing_lock:
            if self._pending_calls:
                await self._process_batch()
                
    def track_sync(self, trace_id: str, function_name: str, **cost_info):
        """同步版本的成本追踪方法
        
        Args:
            trace_id: 追踪ID
            function_name: 函数名称
            **cost_info: 成本信息（包含model、input_tokens、output_tokens等）
        """
        try:
            # 将同步调用转换为异步调用
            call_info = {
                'trace_id': trace_id,
                'function_name': function_name,
                'timestamp': time.time(),
                **cost_info
            }
            
            # 直接添加到待处理队列
            self._pending_calls.append(call_info)
            
            # 如果队列满了，触发批处理
            if len(self._pending_calls) >= self._batch_size:
                # 在同步环境中，我们使用线程池来处理异步操作
                import threading
                import asyncio
                
                def run_async_batch():
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self._process_batch())
                        loop.close()
                    except Exception as e:
                        logger.warning(f"同步成本追踪批处理失败: {e}")
                
                # 在后台线程中运行异步批处理
                thread = threading.Thread(target=run_async_batch, daemon=True)
                thread.start()
                
        except Exception as e:
            logger.warning(f"同步成本追踪失败: {e}")

    async def close(self) -> None:
        """关闭异步成本追踪器"""
        # 处理所有待处理的调用
        await self.flush_pending()
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
    def __del__(self):
        """析构函数"""
        try:
            self._executor.shutdown(wait=False)
        except:
            pass


# 全局异步成本追踪器实例
_async_cost_tracker: Optional[AsyncCostTracker] = None


def get_async_cost_tracker() -> AsyncCostTracker:
    """获取全局异步成本追踪器实例"""
    global _async_cost_tracker
    if _async_cost_tracker is None:
        _async_cost_tracker = AsyncCostTracker()
    return _async_cost_tracker


async def cleanup_async_cost_tracker() -> None:
    """清理全局异步成本追踪器"""
    global _async_cost_tracker
    if _async_cost_tracker is not None:
        await _async_cost_tracker.close()
        _async_cost_tracker = None