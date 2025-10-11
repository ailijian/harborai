#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能模式配置和功能开关

根据性能分析报告的建议，提供不同的性能模式和功能开关配置。
"""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass


class PerformanceMode(Enum):
    """
    性能模式枚举
    
    根据性能分析报告的建议，提供三种性能模式：
    - FAST: 最小功能，最快速度（预期改善 2000-3000ms）
    - BALANCED: 平衡功能和性能（默认模式）
    - FULL: 完整功能，包含所有监控和追踪
    """
    FAST = "fast"          # 最小功能，最快速度
    BALANCED = "balanced"  # 平衡功能和性能
    FULL = "full"          # 完整功能


@dataclass
class FeatureFlags:
    """
    功能开关配置
    
    允许用户选择性启用/禁用功能，根据性能分析报告优化：
    - 成本追踪系统（影响：600-1000ms）
    - 监控系统（影响：300-600ms）
    - 数据库日志（影响：200-500ms）
    - 分布式追踪（影响：100-200ms）
    """
    # 核心功能开关
    enable_cost_tracking: bool = True
    enable_prometheus_metrics: bool = True
    enable_opentelemetry: bool = True
    enable_postgres_logging: bool = True
    enable_detailed_logging: bool = True
    
    # 性能优化开关
    enable_fast_path: bool = True
    enable_async_decorators: bool = True
    enable_plugin_preload: bool = True
    enable_response_cache: bool = True
    enable_token_cache: bool = True
    
    # 后台任务开关
    enable_background_tasks: bool = True
    enable_async_cost_tracking: bool = True
    enable_batch_processing: bool = True
    
    # 调试和开发开关
    enable_debug_mode: bool = False
    enable_performance_profiling: bool = False
    enable_memory_monitoring: bool = False


class PerformanceConfig:
    """
    性能配置管理器
    
    根据性能模式自动配置功能开关，实现性能分析报告中的优化建议。
    """
    
    def __init__(self, mode: Optional[PerformanceMode] = None):
        # 延迟导入避免循环导入
        from .settings import get_settings
        self.settings = get_settings()
        self.mode = mode or PerformanceMode(self.settings.performance_mode)
        self.feature_flags = self._create_feature_flags()
    
    def _create_feature_flags(self) -> FeatureFlags:
        """
        根据性能模式创建功能开关配置
        
        实现性能分析报告中的优化建议：
        - FAST模式：禁用非关键功能，预期改善2000-3000ms
        - BALANCED模式：平衡功能和性能
        - FULL模式：启用所有功能
        """
        if self.mode == PerformanceMode.FAST:
            return FeatureFlags(
                # 核心功能 - 最小化
                enable_cost_tracking=False,  # 禁用成本追踪（节省600-1000ms）
                enable_prometheus_metrics=False,  # 禁用Prometheus（节省50-150ms）
                enable_opentelemetry=False,  # 禁用分布式追踪（节省100-200ms）
                enable_postgres_logging=False,  # 禁用数据库日志（节省100-300ms）
                enable_detailed_logging=False,  # 禁用详细日志
                
                # 性能优化 - 全部启用
                enable_fast_path=True,
                enable_async_decorators=True,
                enable_plugin_preload=True,
                enable_response_cache=True,
                enable_token_cache=True,
                
                # 后台任务 - 启用异步处理
                enable_background_tasks=True,
                enable_async_cost_tracking=True,
                enable_batch_processing=True,
                
                # 调试功能 - 全部禁用
                enable_debug_mode=False,
                enable_performance_profiling=False,
                enable_memory_monitoring=False
            )
        
        elif self.mode == PerformanceMode.BALANCED:
            return FeatureFlags(
                # 核心功能 - 选择性启用
                enable_cost_tracking=self.settings.enable_cost_tracking,
                enable_prometheus_metrics=True,
                enable_opentelemetry=True,
                enable_postgres_logging=self.settings.enable_postgres_logging,
                enable_detailed_logging=False,  # 禁用详细日志以提升性能
                
                # 性能优化 - 全部启用
                enable_fast_path=self.settings.enable_fast_path,
                enable_async_decorators=self.settings.enable_async_decorators,
                enable_plugin_preload=self.settings.enable_plugin_preload,
                enable_response_cache=self.settings.enable_response_cache,
                enable_token_cache=self.settings.enable_token_cache,
                
                # 后台任务 - 全部启用
                enable_background_tasks=self.settings.enable_background_tasks,
                enable_async_cost_tracking=True,
                enable_batch_processing=True,
                
                # 调试功能 - 根据debug模式决定
                enable_debug_mode=self.settings.debug,
                enable_performance_profiling=False,
                enable_memory_monitoring=False
            )
        
        else:  # PerformanceMode.FULL
            return FeatureFlags(
                # 核心功能 - 全部启用
                enable_cost_tracking=True,
                enable_prometheus_metrics=True,
                enable_opentelemetry=True,
                enable_postgres_logging=True,
                enable_detailed_logging=True,
                
                # 性能优化 - 全部启用
                enable_fast_path=self.settings.enable_fast_path,
                enable_async_decorators=self.settings.enable_async_decorators,
                enable_plugin_preload=self.settings.enable_plugin_preload,
                enable_response_cache=self.settings.enable_response_cache,
                enable_token_cache=self.settings.enable_token_cache,
                
                # 后台任务 - 全部启用
                enable_background_tasks=self.settings.enable_background_tasks,
                enable_async_cost_tracking=True,
                enable_batch_processing=True,
                
                # 调试功能 - 根据debug模式决定
                enable_debug_mode=self.settings.debug,
                enable_performance_profiling=self.settings.debug,
                enable_memory_monitoring=self.settings.debug
            )
    
    def get_decorator_config(self) -> Dict[str, bool]:
        """
        获取装饰器配置
        
        根据功能开关决定哪些装饰器应该启用，
        实现性能分析报告中的装饰器优化建议（影响：800-1200ms）。
        """
        return {
            "cost_tracking": self.feature_flags.enable_cost_tracking,
            "postgres_logging": self.feature_flags.enable_postgres_logging,
            "detailed_tracing": self.feature_flags.enable_opentelemetry,
            "async_decorators": self.feature_flags.enable_async_decorators,
            "prometheus_metrics": self.feature_flags.enable_prometheus_metrics
        }
    
    def get_middleware_config(self) -> Dict[str, bool]:
        """
        获取中间件配置
        
        根据功能开关决定哪些中间件应该启用，
        实现快速路径绕过复杂中间件的优化。
        """
        return {
            "fast_path": self.feature_flags.enable_fast_path,
            "cost_tracking_middleware": self.feature_flags.enable_cost_tracking,
            "logging_middleware": self.feature_flags.enable_detailed_logging,
            "metrics_middleware": self.feature_flags.enable_prometheus_metrics,
            "tracing_middleware": self.feature_flags.enable_opentelemetry
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """
        获取缓存配置
        
        实现性能分析报告中的缓存策略优化建议。
        """
        return {
            "response_cache": {
                "enabled": self.feature_flags.enable_response_cache,
                "ttl": self.settings.response_cache_ttl,
                "cleanup_interval": self.settings.cache_cleanup_interval
            },
            "token_cache": {
                "enabled": self.feature_flags.enable_token_cache,
                "ttl": self.settings.token_cache_ttl,
                "cleanup_interval": self.settings.cache_cleanup_interval
            }
        }
    
    def get_background_task_config(self) -> Dict[str, Any]:
        """
        获取后台任务配置
        
        实现性能分析报告中的异步化处理建议。
        """
        return {
            "enabled": self.feature_flags.enable_background_tasks,
            "workers": self.settings.background_task_workers,
            "async_cost_tracking": self.feature_flags.enable_async_cost_tracking,
            "batch_processing": self.feature_flags.enable_batch_processing
        }
    
    def get_plugin_config(self) -> Dict[str, Any]:
        """
        获取插件配置
        
        实现性能分析报告中的插件预加载和缓存优化建议。
        """
        return {
            "preload": self.feature_flags.enable_plugin_preload,
            "cache_size": self.settings.plugin_cache_size,
            "directories": self.settings.plugin_directories
        }
    
    def should_use_fast_path(self, model: str, max_tokens: Optional[int] = None, **kwargs) -> bool:
        """
        判断是否应该使用快速路径
        
        实现性能分析报告中的快速路径优化建议，
        为简单请求绕过复杂的中间件处理链路。
        
        快速路径和性能模式是正交的：
        - feature_flags.enable_fast_path 控制快速路径的总开关
        - 性能模式控制功能的启用程度，但不强制禁用快速路径
        - 只有在 FAST 模式下才强制启用快速路径
        """
        if not self.feature_flags.enable_fast_path:
            return False
        
        # FAST 模式：强制启用快速路径（最大化性能）
        if self.mode == PerformanceMode.FAST:
            return True
        
        # BALANCED 和 FULL 模式：委托给 settings 的判断逻辑
        # FULL 模式不再强制禁用快速路径，允许用户通过 HARBORAI_FAST_PATH 控制
        return self.settings.is_fast_path_enabled(model, max_tokens)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能配置摘要
        
        用于调试和监控当前的性能配置状态。
        """
        return {
            "mode": self.mode.value,
            "feature_flags": {
                "cost_tracking": self.feature_flags.enable_cost_tracking,
                "prometheus_metrics": self.feature_flags.enable_prometheus_metrics,
                "opentelemetry": self.feature_flags.enable_opentelemetry,
                "postgres_logging": self.feature_flags.enable_postgres_logging,
                "fast_path": self.feature_flags.enable_fast_path,
                "async_decorators": self.feature_flags.enable_async_decorators,
                "background_tasks": self.feature_flags.enable_background_tasks
            },
            "expected_improvements": self._get_expected_improvements()
        }
    
    def _get_expected_improvements(self) -> Dict[str, str]:
        """
        获取预期的性能改善
        
        基于性能分析报告的数据。
        """
        if self.mode == PerformanceMode.FAST:
            return {
                "first_token_time": "减少 2000-3000ms",
                "total_improvement": "70-80%",
                "resource_usage": "减少 30-50%"
            }
        elif self.mode == PerformanceMode.BALANCED:
            return {
                "first_token_time": "减少 1000-2000ms",
                "total_improvement": "40-60%",
                "resource_usage": "减少 20-30%"
            }
        else:
            return {
                "first_token_time": "减少 500-1000ms",
                "total_improvement": "20-30%",
                "resource_usage": "减少 10-20%"
            }


# 全局性能配置实例
_performance_config: Optional[PerformanceConfig] = None


def get_performance_config() -> PerformanceConfig:
    """
    获取全局性能配置实例（单例模式）
    """
    global _performance_config
    if _performance_config is None:
        _performance_config = PerformanceConfig()
    return _performance_config


def reset_performance_config(mode: Optional[PerformanceMode] = None) -> PerformanceConfig:
    """
    重置性能配置
    
    用于测试或动态切换性能模式。
    """
    global _performance_config
    _performance_config = PerformanceConfig(mode)
    return _performance_config