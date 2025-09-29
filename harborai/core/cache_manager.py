"""缓存管理器

提供Token计数、响应结果等的缓存功能，提升性能。
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    ttl: int  # 生存时间（秒）
    access_count: int = 0
    last_accessed: datetime = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl <= 0:
            return False  # 永不过期
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    def touch(self) -> None:
        """更新访问时间和计数"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class TokenCache:
    """Token计数缓存
    
    缓存模型的Token计数结果，避免重复计算。
    """
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
    def _generate_key(self, text: str, model: str) -> str:
        """生成缓存键"""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get_token_count(self, text: str, model: str) -> Optional[int]:
        """获取Token计数"""
        key = self._generate_key(text, model)
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
                
            if entry.is_expired:
                del self._cache[key]
                return None
                
            entry.touch()
            return entry.value
    
    def set_token_count(self, text: str, model: str, count: int, ttl: Optional[int] = None) -> None:
        """设置Token计数"""
        key = self._generate_key(text, model)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # 检查缓存大小限制
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            entry = CacheEntry(
                key=key,
                value=count,
                created_at=datetime.now(),
                ttl=ttl
            )
            self._cache[key] = entry
    
    def _evict_lru(self) -> None:
        """淘汰最近最少使用的条目"""
        if not self._cache:
            return
            
        # 找到最近最少访问的条目
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].last_accessed, self._cache[k].access_count)
        )
        del self._cache[lru_key]
    
    def clear_expired(self) -> int:
        """清理过期条目"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
                
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_access = sum(entry.access_count for entry in self._cache.values())
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_access': total_access,
                'hit_rate': 0.0 if total_access == 0 else len(self._cache) / total_access
            }


class ResponseCache:
    """响应结果缓存
    
    缓存API响应结果，对于相同的请求可以直接返回缓存结果。
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        
    def _generate_key(self, request_data: Dict[str, Any]) -> str:
        """生成请求缓存键"""
        # 排除不影响结果的字段
        cache_data = {k: v for k, v in request_data.items() 
                     if k not in ['trace_id', 'timestamp', 'user_id']}
        
        # 确保字典键的顺序一致
        sorted_data = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(sorted_data.encode('utf-8')).hexdigest()
    
    def get_response(self, request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取缓存的响应"""
        key = self._generate_key(request_data)
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
                
            if entry.is_expired:
                del self._cache[key]
                return None
                
            entry.touch()
            return entry.value
    
    def set_response(
        self, 
        request_data: Dict[str, Any], 
        response_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """设置响应缓存"""
        key = self._generate_key(request_data)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # 检查缓存大小限制
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            entry = CacheEntry(
                key=key,
                value=response_data,
                created_at=datetime.now(),
                ttl=ttl
            )
            self._cache[key] = entry
    
    def _evict_lru(self) -> None:
        """淘汰最近最少使用的条目"""
        if not self._cache:
            return
            
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (self._cache[k].last_accessed, self._cache[k].access_count)
        )
        del self._cache[lru_key]
    
    def clear_expired(self) -> int:
        """清理过期条目"""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
                
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_access = sum(entry.access_count for entry in self._cache.values())
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_access': total_access,
                'hit_rate': 0.0 if total_access == 0 else len(self._cache) / total_access
            }


class CacheManager:
    """缓存管理器
    
    统一管理各种缓存，提供清理和统计功能。
    """
    
    def __init__(self, token_cache_config: Optional[Dict] = None, response_cache_config: Optional[Dict] = None):
        token_config = token_cache_config or {}
        response_config = response_cache_config or {}
        
        self.token_cache = TokenCache(
            max_size=token_config.get('max_size', 10000),
            default_ttl=token_config.get('default_ttl', 3600)
        )
        
        self.response_cache = ResponseCache(
            max_size=response_config.get('max_size', 1000),
            default_ttl=response_config.get('default_ttl', 300)
        )
        
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 300  # 5分钟清理一次
        
    async def start_cleanup_task(self) -> None:
        """启动定期清理任务"""
        if self._cleanup_task is not None:
            return
            
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info("缓存清理任务已启动")
    
    async def stop_cleanup_task(self) -> None:
        """停止定期清理任务"""
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("缓存清理任务已停止")
    
    async def _periodic_cleanup(self) -> None:
        """定期清理过期缓存"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                
                token_expired = self.token_cache.clear_expired()
                response_expired = self.response_cache.clear_expired()
                
                if token_expired > 0 or response_expired > 0:
                    logger.debug(f"清理过期缓存: Token缓存 {token_expired} 条，响应缓存 {response_expired} 条")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"缓存清理任务发生错误: {e}")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合缓存统计信息"""
        return {
            'token_cache': self.token_cache.get_stats(),
            'response_cache': self.response_cache.get_stats(),
            'cleanup_task_running': self._cleanup_task is not None and not self._cleanup_task.done()
        }
    
    def clear_all_caches(self) -> None:
        """清空所有缓存"""
        with self.token_cache._lock:
            self.token_cache._cache.clear()
            
        with self.response_cache._lock:
            self.response_cache._cache.clear()
            
        logger.info("所有缓存已清空")


# 全局缓存管理器实例
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """获取全局缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


async def start_cache_manager() -> None:
    """启动全局缓存管理器"""
    manager = get_cache_manager()
    await manager.start_cleanup_task()


async def stop_cache_manager() -> None:
    """停止全局缓存管理器"""
    global _cache_manager
    if _cache_manager is not None:
        await _cache_manager.stop_cleanup_task()
        _cache_manager = None