#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agently客户端池管理器
实现单例模式的客户端复用机制，避免重复创建客户端实例

优化策略：
1. 单例模式确保全局唯一的客户端池
2. 基于配置参数的客户端缓存
3. 线程安全的客户端获取和释放
4. 自动清理和资源管理
"""

import threading
import time
import hashlib
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from harborai.utils.logger import get_logger
from harborai.config.performance import PerformanceConfig

logger = get_logger(__name__)


@dataclass
class AgentlyClientConfig:
    """Agently客户端配置"""
    provider: str
    base_url: str
    model: str
    api_key: str
    model_type: str = "chat"
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    
    def to_cache_key(self) -> str:
        """生成缓存键"""
        # 只使用关键配置参数生成缓存键
        key_data = {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key": self.api_key[:10] + "...",  # 只使用API key前缀避免泄露
            "model_type": self.model_type
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()


@dataclass
class CachedAgentlyClient:
    """缓存的Agently客户端"""
    client: Any  # Agently Agent实例
    config: AgentlyClientConfig
    created_at: float
    last_used: float
    use_count: int = 0
    
    def mark_used(self):
        """标记客户端被使用"""
        self.last_used = time.time()
        self.use_count += 1


class AgentlyClientPool:
    """Agently客户端池管理器
    
    实现单例模式的客户端复用机制：
    - 基于配置参数缓存客户端实例
    - 线程安全的客户端获取和释放
    - 自动清理过期客户端
    - 性能监控和统计
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化客户端池"""
        if self._initialized:
            return
            
        self._clients: Dict[str, CachedAgentlyClient] = {}
        self._client_lock = threading.RLock()
        self._performance_config = PerformanceConfig()
        
        # 配置参数
        self._max_pool_size = 10  # 最大池大小
        self._client_ttl = 3600   # 客户端生存时间（秒）
        self._cleanup_interval = 300  # 清理间隔（秒）
        self._last_cleanup = time.time()
        
        # 统计信息
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "clients_created": 0,
            "clients_cleaned": 0,
            "total_requests": 0
        }
        
        self._initialized = True
        logger.info("🏊 Agently客户端池初始化完成")
    
    def _map_provider_to_agently(self, provider: str) -> str:
        """将HarborAI的provider名称映射到Agently支持的provider名称
        
        Args:
            provider: HarborAI的provider名称
            
        Returns:
            Agently支持的provider名称
        """
        # 大多数OpenAI兼容的API都使用OpenAICompatible
        provider_mapping = {
            "deepseek": "OpenAICompatible",
            "doubao": "OpenAICompatible", 
            "wenxin": "OpenAICompatible",
            "openai": "OpenAI",
            "anthropic": "Claude",
            "gemini": "Gemini"
        }
        
        mapped_provider = provider_mapping.get(provider.lower(), "OpenAICompatible")
        logger.debug(f"Provider映射: {provider} -> {mapped_provider}")
        return mapped_provider
    
    def get_client(self, config: AgentlyClientConfig) -> Any:
        """获取Agently客户端
        
        Args:
            config: 客户端配置
            
        Returns:
            Agently Agent实例
        """
        with self._client_lock:
            self._stats["total_requests"] += 1
            
            # 生成缓存键
            cache_key = config.to_cache_key()
            
            # 检查缓存
            if cache_key in self._clients:
                cached_client = self._clients[cache_key]
                
                # 检查客户端是否过期
                if time.time() - cached_client.created_at < self._client_ttl:
                    cached_client.mark_used()
                    self._stats["cache_hits"] += 1
                    logger.debug(f"🎯 客户端缓存命中: {cache_key[:8]}...")
                    return cached_client.client
                else:
                    # 客户端过期，删除
                    del self._clients[cache_key]
                    logger.debug(f"⏰ 客户端过期删除: {cache_key[:8]}...")
            
            # 缓存未命中，创建新客户端
            self._stats["cache_misses"] += 1
            client = self._create_agently_client(config)
            
            # 缓存新客户端
            cached_client = CachedAgentlyClient(
                client=client,
                config=config,
                created_at=time.time(),
                last_used=time.time(),
                use_count=1
            )
            
            # 检查池大小限制
            if len(self._clients) >= self._max_pool_size:
                self._evict_oldest_client()
            
            self._clients[cache_key] = cached_client
            self._stats["clients_created"] += 1
            
            logger.debug(f"🆕 创建新Agently客户端: {cache_key[:8]}...")
            
            # 定期清理
            self._maybe_cleanup()
            
            return client
    
    def _create_agently_client(self, config: AgentlyClientConfig) -> Any:
        """创建Agently客户端实例
        
        Args:
            config: 客户端配置
            
        Returns:
            Agently Agent实例
        """
        try:
            from Agently.agently import Agently
            
            # 设置Agently配置
            settings = {
                "base_url": config.base_url,
                "model": config.model,
                "model_type": config.model_type,
                "auth": config.api_key,
            }
            
            # 添加可选参数
            if config.temperature is not None:
                settings["temperature"] = config.temperature
            if config.max_tokens is not None:
                settings["max_tokens"] = config.max_tokens
            
            # 配置Agently - 将provider映射到Agently支持的格式
            agently_provider = self._map_provider_to_agently(config.provider)
            Agently.set_settings(agently_provider, settings)
            
            # 创建agent
            agent = Agently.create_agent()
            
            logger.debug(f"✅ Agently客户端创建成功: {config.provider}/{config.model}")
            return agent
            
        except Exception as e:
            logger.error(f"❌ 创建Agently客户端失败: {e}")
            raise
    
    def _evict_oldest_client(self):
        """驱逐最旧的客户端"""
        if not self._clients:
            return
            
        # 找到最旧的客户端
        oldest_key = min(self._clients.keys(), 
                        key=lambda k: self._clients[k].last_used)
        
        del self._clients[oldest_key]
        self._stats["clients_cleaned"] += 1
        logger.debug(f"🗑️ 驱逐最旧客户端: {oldest_key[:8]}...")
    
    def _maybe_cleanup(self):
        """可能执行清理操作"""
        now = time.time()
        if now - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired_clients()
            self._last_cleanup = now
    
    def _cleanup_expired_clients(self):
        """清理过期的客户端"""
        now = time.time()
        expired_keys = []
        
        for key, cached_client in self._clients.items():
            if now - cached_client.created_at > self._client_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._clients[key]
            self._stats["clients_cleaned"] += 1
        
        if expired_keys:
            logger.debug(f"🧹 清理过期客户端: {len(expired_keys)}个")
    
    @contextmanager
    def get_client_context(self, config: AgentlyClientConfig):
        """获取客户端的上下文管理器
        
        Args:
            config: 客户端配置
            
        Yields:
            Agently Agent实例
        """
        client = self.get_client(config)
        try:
            yield client
        finally:
            # 这里可以添加客户端释放逻辑
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取池统计信息
        
        Returns:
            统计信息字典
        """
        with self._client_lock:
            stats = self._stats.copy()
            stats.update({
                "pool_size": len(self._clients),
                "cache_hit_rate": (
                    self._stats["cache_hits"] / max(1, self._stats["total_requests"])
                ) * 100,
                "clients_info": [
                    {
                        "cache_key": key[:8] + "...",
                        "provider": client.config.provider,
                        "model": client.config.model,
                        "created_at": client.created_at,
                        "last_used": client.last_used,
                        "use_count": client.use_count,
                        "age_seconds": time.time() - client.created_at
                    }
                    for key, client in self._clients.items()
                ]
            })
            return stats
    
    def clear_pool(self):
        """清空客户端池"""
        with self._client_lock:
            cleared_count = len(self._clients)
            self._clients.clear()
            logger.info(f"🧹 客户端池已清空: {cleared_count}个客户端")
    
    def set_pool_config(self, max_size: int = None, ttl: int = None, 
                       cleanup_interval: int = None):
        """设置池配置参数
        
        Args:
            max_size: 最大池大小
            ttl: 客户端生存时间（秒）
            cleanup_interval: 清理间隔（秒）
        """
        with self._client_lock:
            if max_size is not None:
                self._max_pool_size = max_size
            if ttl is not None:
                self._client_ttl = ttl
            if cleanup_interval is not None:
                self._cleanup_interval = cleanup_interval
            
            logger.info(f"🔧 客户端池配置更新: max_size={self._max_pool_size}, "
                       f"ttl={self._client_ttl}, cleanup_interval={self._cleanup_interval}")


# 全局客户端池实例
_global_client_pool = None
_global_pool_lock = threading.Lock()


def get_agently_client_pool() -> AgentlyClientPool:
    """获取全局Agently客户端池实例
    
    Returns:
        AgentlyClientPool实例
    """
    global _global_client_pool
    
    if _global_client_pool is None:
        with _global_pool_lock:
            if _global_client_pool is None:
                _global_client_pool = AgentlyClientPool()
    
    return _global_client_pool


def create_agently_client_config(provider: str, base_url: str, model: str, 
                                api_key: str, **kwargs) -> AgentlyClientConfig:
    """创建Agently客户端配置
    
    Args:
        provider: 提供商名称
        base_url: API基础URL
        model: 模型名称
        api_key: API密钥
        **kwargs: 其他配置参数
        
    Returns:
        AgentlyClientConfig实例
    """
    return AgentlyClientConfig(
        provider=provider,
        base_url=base_url,
        model=model,
        api_key=api_key,
        **kwargs
    )