#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastHarborAI 快速客户端

实现最小化初始化的HarborAI客户端，显著提升启动性能。
根据技术设计方案，目标是将初始化时间降低到≤160ms。

设计原则：
1. 最小化初始化：只初始化必要的组件
2. 延迟加载：核心组件在首次使用时才被加载
3. 兼容性：保持与现有API的完全兼容
4. 性能优先：优化关键路径的性能
"""

import time
import threading
from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Generator
import logging

from ..core.lazy_plugin_manager import get_lazy_plugin_manager
from ..core.exceptions import HarborAIError, ValidationError
from ..config.settings import get_settings
from ..utils.logger import get_logger

# 内存优化组件（可选导入）
try:
    from ..core.optimizations.memory_manager import MemoryManager
    MEMORY_OPTIMIZATION_AVAILABLE = True
except ImportError:
    MemoryManager = None
    MEMORY_OPTIMIZATION_AVAILABLE = False

# 并发优化组件（可选导入）
try:
    from ..core.optimizations.concurrency_manager import ConcurrencyManager, ConcurrencyConfig, get_concurrency_manager
    from ..core.optimizations.async_request_processor import RequestPriority
    CONCURRENCY_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ConcurrencyManager = None
    ConcurrencyConfig = None
    get_concurrency_manager = None
    RequestPriority = None
    CONCURRENCY_OPTIMIZATION_AVAILABLE = False

logger = get_logger(__name__)


class FastChatCompletions:
    """快速聊天完成接口
    
    设计原则：
    1. 最小化初始化开销
    2. 延迟加载核心组件
    3. 保持与标准接口的兼容性
    4. 性能优先
    """
    
    def __init__(self, client: 'FastHarborAI'):
        """初始化聊天完成接口
        
        Args:
            client: FastHarborAI客户端实例
        """
        self._client = client
        self._config = client.config
        
        # 延迟初始化的组件
        self._lazy_plugin_manager = None
        self._performance_manager = None
        self._cache_manager = None
        self._concurrency_manager = None
        self._concurrency_start_task = None
        
        # 初始化状态
        self._initialized = False
        self._lock = threading.RLock()
        
        # 性能统计
        self._request_count = 0
        self._total_time = 0.0
        self._last_request_time = None
        
        logger.debug("FastChatCompletions初始化完成")
    
    def _get_lazy_plugin_manager(self):
        """获取延迟插件管理器"""
        if self._lazy_plugin_manager is None:
            self._lazy_plugin_manager = get_lazy_plugin_manager(self._config)
        return self._lazy_plugin_manager
    
    def _get_performance_manager(self):
        """获取性能管理器"""
        if self._performance_manager is None:
            try:
                from ..core.performance import PerformanceManager
                self._performance_manager = PerformanceManager()
            except ImportError:
                logger.warning("性能管理器不可用")
                self._performance_manager = None
        return self._performance_manager
    
    def _get_cache_manager(self):
        """获取缓存管理器"""
        if self._cache_manager is None:
            try:
                from ..core.cache import CacheManager
                cache_config = self._config.get('cache', {})
                self._cache_manager = CacheManager(**cache_config)
            except ImportError:
                logger.warning("缓存管理器不可用")
                self._cache_manager = None
        return self._cache_manager
    
    def _get_concurrency_manager(self):
        """获取并发管理器"""
        if self._concurrency_manager is None and CONCURRENCY_OPTIMIZATION_AVAILABLE:
            try:
                # 从配置中获取并发优化参数
                concurrency_config = self._config.get('concurrency_optimization', {})
                config = ConcurrencyConfig(
                    max_concurrent_requests=concurrency_config.get('max_concurrent_requests', 50),
                    connection_pool_size=concurrency_config.get('connection_pool_size', 20),
                    request_timeout=concurrency_config.get('request_timeout', 30.0),
                    enable_adaptive_optimization=concurrency_config.get('enable_adaptive_optimization', True),
                    enable_health_check=concurrency_config.get('enable_health_check', True),
                    health_check_interval=concurrency_config.get('health_check_interval', 60.0)
                )
                # 创建并发管理器实例
                self._concurrency_manager = ConcurrencyManager(config)
                
                # 同步启动并发管理器，确保完全启动
                import asyncio
                try:
                    # 检查是否有运行中的事件循环
                    loop = asyncio.get_running_loop()
                    # 如果有运行中的事件循环，创建任务并等待完成
                    task = asyncio.create_task(self._concurrency_manager.start())
                    # 注意：这里不能直接await，因为我们在同步方法中
                    # 将启动任务存储，稍后在异步方法中检查
                    self._concurrency_start_task = task
                except RuntimeError:
                    # 没有运行中的事件循环，使用run同步等待启动完成
                    asyncio.run(self._concurrency_manager.start())
                    self._concurrency_start_task = None
                
                logger.info("并发管理器初始化成功")
            except Exception as e:
                logger.error("并发管理器初始化失败: %s", str(e))
                self._concurrency_manager = None
        return self._concurrency_manager
    
    def _ensure_initialized(self):
        """确保组件已初始化
        
        延迟初始化核心组件，只在首次使用时加载。
        """
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            try:
                start_time = time.perf_counter()
                
                # 获取延迟插件管理器
                self._lazy_manager = get_lazy_plugin_manager(self._client.config)
                
                # 延迟加载性能管理器
                if self._client.config.get('enable_performance_optimization', True):
                    self._load_performance_manager()
                
                # 延迟加载缓存管理器
                if self._client.config.get('enable_caching', True):
                    self._load_cache_manager()
                
                self._initialized = True
                
                init_time = (time.perf_counter() - start_time) * 1000
                logger.debug("FastChatCompletions组件初始化完成，耗时: %.2fms", init_time)
                
            except Exception as e:
                logger.error("FastChatCompletions组件初始化失败: %s", str(e))
                raise HarborAIError(f"组件初始化失败: {e}")
    
    def _load_performance_manager(self):
        """延迟加载性能管理器"""
        try:
            from ..core.performance_manager import PerformanceManager
            self._performance_manager = PerformanceManager(self._client.config)
        except ImportError:
            logger.warning("性能管理器不可用，跳过加载")
        except Exception as e:
            logger.warning("性能管理器加载失败: %s", str(e))
    
    def _load_cache_manager(self):
        """延迟加载缓存管理器"""
        try:
            # 优先使用内存管理器的缓存
            if hasattr(self._client, '_memory_manager') and self._client._memory_manager:
                self._cache_manager = self._client._memory_manager.cache
                logger.debug("使用内存管理器的缓存")
            else:
                # 回退到传统缓存管理器
                from ..core.cache_manager import CacheManager
                self._cache_manager = CacheManager(self._client.config)
                logger.debug("使用传统缓存管理器")
        except ImportError:
            logger.warning("缓存管理器不可用，跳过加载")
        except Exception as e:
            logger.warning("缓存管理器加载失败: %s", str(e))
    
    async def _ensure_concurrency_manager_started(self):
        """确保并发管理器完全启动"""
        if hasattr(self, '_concurrency_start_task') and self._concurrency_start_task:
            try:
                # 等待启动任务完成
                await self._concurrency_start_task
                self._concurrency_start_task = None
                logger.debug("并发管理器启动完成")
            except Exception as e:
                logger.error("等待并发管理器启动失败: %s", str(e))
                raise
    
    def create(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """创建聊天完成
        
        Args:
            messages: 消息列表
            model: 模型名称
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            响应数据或流式生成器
        """
        # 延迟初始化
        self._ensure_initialized()
        
        # 参数验证
        self._validate_request(messages, model, **kwargs)
        
        # 尝试使用并发管理器
        concurrency_manager = self._get_concurrency_manager()
        if concurrency_manager and not stream:
            try:
                # 使用并发管理器处理请求
                priority = kwargs.get('priority', RequestPriority.NORMAL if RequestPriority else 'normal')
                # 使用asyncio.run来运行异步方法
                import asyncio
                response = asyncio.run(concurrency_manager.create_chat_completion(
                    model=model,
                    messages=messages,
                    stream=stream,
                    priority=priority,
                    **kwargs
                ))
                return response
            except Exception as e:
                logger.warning("并发管理器处理失败，回退到传统方式: %s", str(e))
        
        # 传统方式处理
        return self._create_traditional(messages, model, stream, **kwargs)
    
    def _create_traditional(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """传统方式创建聊天完成"""
        # 获取对应的插件
        plugin = self._lazy_manager.get_plugin_for_model(model)
        if not plugin:
            raise ValidationError(f"不支持的模型: {model}")
        
        try:
            # 检查缓存
            cache_key = None
            if self._cache_manager and not stream:
                cache_key = self._generate_cache_key(messages, model, **kwargs)
                cached_response = self._cache_manager.get(cache_key)
                if cached_response:
                    logger.debug("使用缓存响应，模型: %s", model)
                    return cached_response
            
            # 调用插件
            start_time = time.perf_counter()
            
            # 转换消息格式
            formatted_messages = self._format_messages(messages)
            
            # 调用插件的聊天完成方法
            response = plugin.chat_completion(
                messages=formatted_messages,
                model=model,
                stream=stream,
                **kwargs
            )
            
            # 记录性能指标
            if self._performance_manager:
                response_time = (time.perf_counter() - start_time) * 1000
                self._performance_manager.record_request(model, response_time)
            
            # 缓存响应
            if self._cache_manager and cache_key and not stream:
                self._cache_manager.set(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error("聊天完成请求失败，模型: %s，错误: %s", model, str(e))
            raise
    
    async def create_async(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """异步创建聊天完成
        
        Args:
            messages: 消息列表
            model: 模型名称
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            响应数据或异步流式生成器
        """
        # 延迟初始化
        self._ensure_initialized()
        
        # 参数验证
        self._validate_request(messages, model, **kwargs)
        
        # 尝试使用并发管理器
        concurrency_manager = self._get_concurrency_manager()
        if concurrency_manager:
            try:
                # 确保并发管理器完全启动
                await self._ensure_concurrency_manager_started()
                
                # 使用并发管理器处理异步请求
                priority = kwargs.get('priority', RequestPriority.NORMAL if RequestPriority else 'normal')
                response = await concurrency_manager.create_chat_completion(
                    model=model,
                    messages=messages,
                    stream=stream,
                    priority=priority,
                    **kwargs
                )
                return response
            except Exception as e:
                logger.warning("并发管理器异步处理失败，回退到传统方式: %s", str(e))
        
        # 传统异步方式处理
        return await self._create_async_traditional(messages, model, stream, **kwargs)
    
    async def _create_async_traditional(
        self,
        messages: List[Dict[str, str]],
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """传统异步方式创建聊天完成"""
        # 获取对应的插件
        plugin = self._lazy_manager.get_plugin_for_model(model)
        if not plugin:
            raise ValidationError(f"不支持的模型: {model}")
        
        try:
            # 转换消息格式
            formatted_messages = self._format_messages(messages)
            
            # 调用插件的异步聊天完成方法
            response = await plugin.chat_completion_async(
                messages=formatted_messages,
                model=model,
                stream=stream,
                **kwargs
            )
            
            return response
            
        except Exception as e:
            logger.error("异步聊天完成请求失败，模型: %s，错误: %s", model, str(e))
            raise
    
    def _validate_request(self, messages: List[Dict[str, str]], model: str, **kwargs):
        """验证请求参数
        
        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他参数
        """
        if not messages:
            raise ValidationError("消息列表不能为空")
        
        if not model:
            raise ValidationError("模型名称不能为空")
        
        # 验证消息格式
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValidationError(f"消息 {i} 必须是字典格式")
            
            if 'role' not in message:
                raise ValidationError(f"消息 {i} 缺少role字段")
            
            if 'content' not in message:
                raise ValidationError(f"消息 {i} 缺少content字段")
        
        # 验证可选参数
        temperature = kwargs.get('temperature')
        if temperature is not None and not (0 <= temperature <= 2):
            raise ValidationError("temperature必须在0-2之间")
        
        max_tokens = kwargs.get('max_tokens')
        if max_tokens is not None and max_tokens <= 0:
            raise ValidationError("max_tokens必须大于0")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """格式化消息
        
        Args:
            messages: 原始消息列表
            
        Returns:
            格式化后的消息列表
        """
        # 简单的消息格式化，确保兼容性
        formatted = []
        for message in messages:
            formatted_message = {
                'role': message['role'],
                'content': message['content']
            }
            
            # 保留其他字段
            for key, value in message.items():
                if key not in ['role', 'content']:
                    formatted_message[key] = value
            
            formatted.append(formatted_message)
        
        return formatted
    
    def _generate_cache_key(self, messages: List[Dict[str, str]], model: str, **kwargs) -> str:
        """生成缓存键
        
        Args:
            messages: 消息列表
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            缓存键
        """
        import hashlib
        import json
        
        # 构建缓存数据
        cache_data = {
            'messages': messages,
            'model': model,
            'kwargs': {k: v for k, v in kwargs.items() if k not in ['stream']}
        }
        
        # 生成哈希
        cache_str = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(cache_str.encode('utf-8')).hexdigest()


class FastChat:
    """快速聊天接口
    
    提供与标准Chat接口兼容的API。
    """
    
    def __init__(self, client: 'FastHarborAI'):
        """初始化快速聊天接口
        
        Args:
            client: FastHarborAI客户端实例
        """
        self._client = client
        self._completions = None
    
    @property
    def completions(self) -> FastChatCompletions:
        """获取聊天完成接口
        
        延迟创建聊天完成接口实例。
        
        Returns:
            FastChatCompletions实例
        """
        if self._completions is None:
            self._completions = FastChatCompletions(self._client)
        return self._completions


class FastHarborAI:
    """FastHarborAI 快速客户端
    
    实现最小化初始化的HarborAI客户端，显著提升启动性能。
    
    主要特性：
    1. 快速初始化：只加载必要的配置和基础组件
    2. 延迟加载：核心功能在首次使用时才被加载
    3. 完全兼容：与现有HarborAI API完全兼容
    4. 性能优化：优化关键路径的执行效率
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """初始化FastHarborAI客户端
        
        Args:
            config: 配置字典
            **kwargs: 其他配置参数
        """
        start_time = time.perf_counter()
        
        # 合并配置
        self.config = self._merge_config(config, kwargs)
        
        # 初始化基础组件
        self._chat = None
        self._initialized_at = time.time()
        self._memory_manager = None
        
        # 初始化内存优化（如果启用）
        if self.config.get('enable_memory_optimization', False):
            self._init_memory_optimization()
        
        # 设置日志级别
        log_level = self.config.get('log_level', 'INFO')
        logging.getLogger('harborai').setLevel(getattr(logging, log_level.upper()))
        
        init_time = (time.perf_counter() - start_time) * 1000
        logger.info("FastHarborAI客户端初始化完成，耗时: %.2fms", init_time)
    
    def _merge_config(self, config: Optional[Dict[str, Any]], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """合并配置
        
        Args:
            config: 配置字典
            kwargs: 关键字参数
            
        Returns:
            合并后的配置
        """
        # 获取默认设置
        try:
            default_config = get_settings().dict()
        except Exception:
            default_config = {}
        
        # 合并配置
        merged_config = {}
        merged_config.update(default_config)
        
        if config:
            merged_config.update(config)
        
        merged_config.update(kwargs)
        
        return merged_config
    
    def _init_memory_optimization(self):
        """初始化内存优化组件"""
        if not MEMORY_OPTIMIZATION_AVAILABLE:
            logger.warning("内存优化组件不可用，跳过初始化")
            return
        
        try:
            # 从配置中获取内存优化参数
            memory_opt_config = self.config.get('memory_optimization', {})
            memory_config = {
                'cache_size': memory_opt_config.get('cache_size', self.config.get('memory_cache_size', 1000)),
                'object_pool_size': memory_opt_config.get('object_pool_size', self.config.get('memory_object_pool_size', 100)),
                'enable_weak_references': memory_opt_config.get('enable_weak_references', self.config.get('memory_enable_weak_refs', True)),
                'memory_threshold_mb': memory_opt_config.get('memory_threshold', self.config.get('memory_threshold_mb', 50.0)),
                'auto_cleanup_interval': memory_opt_config.get('cleanup_interval', self.config.get('memory_cleanup_interval', 300))  # 5分钟
            }
            
            self._memory_manager = MemoryManager(**memory_config)
            logger.info("内存优化组件初始化成功")
            
        except Exception as e:
            logger.error("内存优化组件初始化失败: %s", str(e))
            self._memory_manager = None
    
    @property
    def chat(self) -> FastChat:
        """获取聊天接口
        
        延迟创建聊天接口实例。
        
        Returns:
            FastChat实例
        """
        if self._chat is None:
            self._chat = FastChat(self)
        return self._chat
    
    @property
    def completions(self) -> FastChatCompletions:
        """获取聊天完成接口（兼容性属性）
        
        Returns:
            FastChatCompletions实例
        """
        return self.chat.completions
    
    def get_supported_models(self) -> List[str]:
        """获取支持的模型列表
        
        Returns:
            支持的模型名称列表
        """
        lazy_manager = get_lazy_plugin_manager(self.config)
        return lazy_manager.get_supported_models()
    
    def preload_model(self, model: str) -> bool:
        """预加载模型对应的插件
        
        Args:
            model: 模型名称
            
        Returns:
            是否预加载成功
        """
        lazy_manager = get_lazy_plugin_manager(self.config)
        plugin_name = lazy_manager.get_plugin_name_for_model(model)
        if plugin_name:
            return lazy_manager.preload_plugin(plugin_name)
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取客户端统计信息
        
        Returns:
            统计信息字典
        """
        lazy_manager = get_lazy_plugin_manager(self.config)
        stats = lazy_manager.get_statistics()
        
        stats.update({
            'client_type': 'FastHarborAI',
            'initialized_at': self._initialized_at,
            'uptime_seconds': time.time() - self._initialized_at,
            'memory_optimization_enabled': self._memory_manager is not None
        })
        
        # 添加内存统计信息
        if self._memory_manager:
            stats['memory_stats'] = self._memory_manager.get_memory_stats()
        
        return stats
    
    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """获取内存统计信息
        
        Returns:
            内存统计信息，如果未启用内存优化则返回None
        """
        if self._memory_manager:
            stats = self._memory_manager.get_memory_stats()
            
            # 添加系统内存信息
            try:
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                stats['system_memory'] = {
                    'rss_mb': memory_info.rss / 1024 / 1024,
                    'vms_mb': memory_info.vms / 1024 / 1024,
                    'percent': process.memory_percent()
                }
            except ImportError:
                stats['system_memory'] = {'error': 'psutil not available'}
            except Exception as e:
                stats['system_memory'] = {'error': str(e)}
            
            return stats
        return None
    
    def cleanup_memory(self, force_clear: bool = False) -> Optional[Dict[str, int]]:
        """执行内存清理
        
        Args:
            force_clear: 是否强制清空所有缓存
            
        Returns:
            清理统计信息，如果未启用内存优化则返回None
        """
        if self._memory_manager:
            return self._memory_manager.cleanup(force_clear=force_clear)
        return None
    
    def cleanup(self):
        """清理客户端资源"""
        logger.info("开始清理FastHarborAI客户端资源")
        
        # 清理聊天接口
        if self._chat and hasattr(self._chat.completions, '_performance_manager'):
            if self._chat.completions._performance_manager:
                self._chat.completions._performance_manager.cleanup()
        
        # 清理内存管理器
        if self._memory_manager:
            try:
                self._memory_manager.shutdown()
                logger.info("内存管理器清理完成")
            except Exception as e:
                logger.error("内存管理器清理失败: %s", str(e))
        
        logger.info("FastHarborAI客户端资源清理完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except Exception:
            pass  # 忽略析构时的异常


# 便捷函数
def create_fast_client(config: Optional[Dict[str, Any]] = None, **kwargs) -> FastHarborAI:
    """创建FastHarborAI客户端的便捷函数
    
    Args:
        config: 配置字典
        **kwargs: 其他配置参数
        
    Returns:
        FastHarborAI客户端实例
    """
    return FastHarborAI(config=config, **kwargs)