"""性能管理器

统一管理所有性能优化组件的生命周期和配置。
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from ..config.settings import get_settings
from .async_cost_tracking import get_async_cost_tracker, cleanup_async_cost_tracker
from .background_tasks import (
    get_background_processor, 
    start_background_processor, 
    stop_background_processor
)
from .cache_manager import (
    get_cache_manager, 
    start_cache_manager, 
    stop_cache_manager
)
from .optimized_plugin_manager import (
    get_optimized_plugin_manager,
    start_optimized_plugin_manager,
    stop_optimized_plugin_manager
)

logger = logging.getLogger(__name__)


@dataclass
class PerformanceStats:
    """性能统计信息"""
    startup_time: float = 0.0
    total_requests: int = 0
    avg_response_time: float = 0.0
    cache_hit_rate: float = 0.0
    error_rate: float = 0.0
    cost_tracking_stats: Dict[str, Any] = field(default_factory=dict)
    plugin_stats: Dict[str, Any] = field(default_factory=dict)
    background_task_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'startup_time': self.startup_time,
            'total_requests': self.total_requests,
            'avg_response_time': self.avg_response_time,
            'cache_hit_rate': self.cache_hit_rate,
            'error_rate': self.error_rate,
            'cost_tracking_stats': self.cost_tracking_stats,
            'plugin_stats': self.plugin_stats,
            'background_task_stats': self.background_task_stats
        }


class PerformanceManager:
    """性能管理器
    
    统一管理所有性能优化组件。
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._initialized = False
        self._startup_time = 0.0
        self._stats = PerformanceStats()
        
        # 组件引用
        self._cost_tracker = None
        self._background_processor = None
        self._cache_manager = None
        self._plugin_manager = None
    
    async def initialize(self) -> None:
        """初始化性能管理器"""
        if self._initialized:
            logger.warning("性能管理器已经初始化")
            return
        
        start_time = time.time()
        logger.info("开始初始化性能管理器...")
        
        try:
            # 初始化各个组件
            await self._initialize_components()
            
            # 记录启动时间
            self._startup_time = time.time() - start_time
            self._stats.startup_time = self._startup_time
            
            self._initialized = True
            logger.info(f"性能管理器初始化完成，耗时: {self._startup_time:.3f}s")
            
        except Exception as e:
            logger.error(f"性能管理器初始化失败: {e}")
            await self.cleanup()
            raise
    
    async def _initialize_components(self) -> None:
        """初始化各个组件"""
        initialization_tasks = []
        
        # 异步成本追踪器
        if self.settings.enable_async_decorators:
            logger.debug("初始化异步成本追踪器...")
            self._cost_tracker = get_async_cost_tracker()
        
        # 后台任务处理器
        logger.debug("初始化后台任务处理器...")
        initialization_tasks.append(start_background_processor())
        
        # 缓存管理器
        if self.settings.enable_token_cache:
            logger.debug("初始化缓存管理器...")
            initialization_tasks.append(start_cache_manager())
        
        # 优化插件管理器
        logger.debug("初始化优化插件管理器...")
        initialization_tasks.append(start_optimized_plugin_manager())
        
        # 并发初始化
        if initialization_tasks:
            await asyncio.gather(*initialization_tasks, return_exceptions=True)
        
        # 获取组件引用
        self._background_processor = get_background_processor()
        if self.settings.enable_token_cache:
            self._cache_manager = get_cache_manager()
        self._plugin_manager = get_optimized_plugin_manager()
    
    async def cleanup(self) -> None:
        """清理资源"""
        if not self._initialized:
            return
        
        logger.info("开始清理性能管理器...")
        
        cleanup_tasks = []
        
        # 清理各个组件
        if self._cost_tracker:
            cleanup_tasks.append(cleanup_async_cost_tracker())
        
        cleanup_tasks.append(stop_background_processor())
        
        if self._cache_manager:
            cleanup_tasks.append(stop_cache_manager())
        
        if self._plugin_manager:
            cleanup_tasks.append(stop_optimized_plugin_manager())
        
        # 并发清理
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        self._initialized = False
        logger.info("性能管理器清理完成")
    
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._initialized
    
    def get_startup_time(self) -> float:
        """获取启动时间"""
        return self._startup_time
    
    async def get_performance_stats(self) -> PerformanceStats:
        """获取性能统计信息"""
        if not self._initialized:
            return self._stats
        
        try:
            # 更新统计信息
            await self._update_stats()
            return self._stats
            
        except Exception as e:
            logger.error(f"获取性能统计失败: {e}")
            return self._stats
    
    async def _update_stats(self) -> None:
        """更新统计信息"""
        # 成本追踪统计
        if self._cost_tracker:
            try:
                cost_summary = await self._cost_tracker.get_cost_summary()
                self._stats.cost_tracking_stats = cost_summary
            except Exception as e:
                logger.warning(f"获取成本追踪统计失败: {e}")
        
        # 插件统计
        if self._plugin_manager:
            try:
                plugin_stats = self._plugin_manager.get_performance_stats()
                self._stats.plugin_stats = plugin_stats
            except Exception as e:
                logger.warning(f"获取插件统计失败: {e}")
        
        # 后台任务统计
        if self._background_processor:
            try:
                bg_stats = self._background_processor.get_stats()
                self._stats.background_task_stats = bg_stats
            except Exception as e:
                logger.warning(f"获取后台任务统计失败: {e}")
        
        # 缓存统计
        if self._cache_manager:
            try:
                cache_stats = self._cache_manager.get_stats()
                # 计算缓存命中率
                total_requests = cache_stats.get('total_requests', 0)
                cache_hits = cache_stats.get('cache_hits', 0)
                if total_requests > 0:
                    self._stats.cache_hit_rate = cache_hits / total_requests
            except Exception as e:
                logger.warning(f"获取缓存统计失败: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'status': 'healthy',
            'initialized': self._initialized,
            'startup_time': self._startup_time,
            'components': {}
        }
        
        if not self._initialized:
            health_status['status'] = 'not_initialized'
            return health_status
        
        try:
            # 检查各个组件状态
            components = health_status['components']
            
            # 成本追踪器
            if self._cost_tracker:
                components['cost_tracker'] = 'healthy'
            
            # 后台任务处理器
            if self._background_processor:
                bg_stats = self._background_processor.get_stats()
                components['background_processor'] = {
                    'status': 'healthy',
                    'queue_size': bg_stats.get('queue_size', 0),
                    'processed_tasks': bg_stats.get('processed_tasks', 0)
                }
            
            # 缓存管理器
            if self._cache_manager:
                cache_stats = self._cache_manager.get_stats()
                components['cache_manager'] = {
                    'status': 'healthy',
                    'cache_size': cache_stats.get('cache_size', 0)
                }
            
            # 插件管理器
            if self._plugin_manager:
                plugin_stats = self._plugin_manager.get_performance_stats()
                components['plugin_manager'] = {
                    'status': 'healthy',
                    'loaded_plugins': len(plugin_stats.get('plugin_stats', {}))
                }
            
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
            logger.error(f"健康检查失败: {e}")
        
        return health_status
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """执行性能优化"""
        if not self._initialized:
            return {'status': 'error', 'message': '性能管理器未初始化'}
        
        optimization_results = {
            'status': 'success',
            'optimizations': []
        }
        
        try:
            # 清理缓存
            if self._cache_manager:
                await self._cache_manager.cleanup_expired()
                optimization_results['optimizations'].append('cache_cleanup')
            
            # 刷新待处理的成本追踪
            if self._cost_tracker:
                await self._cost_tracker.flush_pending()
                optimization_results['optimizations'].append('cost_tracking_flush')
            
            # 优化后台任务队列
            if self._background_processor:
                # 这里可以添加队列优化逻辑
                optimization_results['optimizations'].append('background_task_optimization')
            
            logger.info(f"性能优化完成: {optimization_results['optimizations']}")
            
        except Exception as e:
            optimization_results['status'] = 'error'
            optimization_results['error'] = str(e)
            logger.error(f"性能优化失败: {e}")
        
        return optimization_results


# 全局性能管理器实例
_performance_manager: Optional[PerformanceManager] = None


def get_performance_manager() -> PerformanceManager:
    """获取全局性能管理器实例"""
    global _performance_manager
    if _performance_manager is None:
        _performance_manager = PerformanceManager()
    return _performance_manager


async def initialize_performance_manager() -> None:
    """初始化全局性能管理器"""
    manager = get_performance_manager()
    await manager.initialize()


async def cleanup_performance_manager() -> None:
    """清理全局性能管理器"""
    global _performance_manager
    if _performance_manager is not None:
        await _performance_manager.cleanup()
        _performance_manager = None


async def get_system_performance_stats() -> Dict[str, Any]:
    """获取系统性能统计"""
    manager = get_performance_manager()
    if manager.is_initialized():
        stats = await manager.get_performance_stats()
        return stats.to_dict()
    else:
        return {'status': 'not_initialized'}


async def perform_system_health_check() -> Dict[str, Any]:
    """执行系统健康检查"""
    manager = get_performance_manager()
    return await manager.health_check()


async def optimize_system_performance() -> Dict[str, Any]:
    """优化系统性能"""
    manager = get_performance_manager()
    return await manager.optimize_performance()