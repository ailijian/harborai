#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数缓存层
提供Schema转换结果和配置参数的缓存机制，减少重复计算开销
"""

import hashlib
import json
import time
import threading
from typing import Any, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from ..utils.logger import get_logger
from .unified_decorators import fast_trace

logger = get_logger(__name__)


@dataclass
class ParameterCacheEntry:
    """参数缓存条目"""
    key: str
    value: Any
    created_at: float
    ttl: int  # 生存时间（秒）
    access_count: int = 0
    last_accessed: float = None
    cache_type: str = "general"  # 缓存类型：schema, config, general
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = self.created_at
    
    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl <= 0:
            return False  # 永不过期
        return time.time() > self.created_at + self.ttl
    
    def touch(self) -> None:
        """更新访问时间和计数"""
        self.last_accessed = time.time()
        self.access_count += 1


class SchemaCache:
    """Schema转换结果缓存
    
    缓存JSON Schema到Agently格式的转换结果，避免重复转换计算
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """初始化Schema缓存
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认生存时间（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, ParameterCacheEntry] = {}
        self._lock = threading.RLock()
        
    @fast_trace
    def _generate_key(self, schema: Dict[str, Any]) -> str:
        """生成Schema缓存键
        
        Args:
            schema: JSON Schema定义
            
        Returns:
            缓存键字符串
        """
        # 确保字典键的顺序一致，并排除不影响转换结果的字段
        cache_schema = self._normalize_schema(schema)
        schema_str = json.dumps(cache_schema, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(schema_str.encode('utf-8')).hexdigest()
    
    def _normalize_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """标准化Schema，移除不影响转换的字段
        
        Args:
            schema: 原始Schema
            
        Returns:
            标准化后的Schema
        """
        # 排除不影响转换结果的字段
        excluded_fields = {'$schema', 'title', 'examples', '$id'}
        
        def clean_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(d, dict):
                return d
            
            cleaned = {}
            for k, v in d.items():
                if k not in excluded_fields:
                    if isinstance(v, dict):
                        cleaned[k] = clean_dict(v)
                    elif isinstance(v, list):
                        cleaned[k] = [clean_dict(item) if isinstance(item, dict) else item for item in v]
                    else:
                        cleaned[k] = v
            return cleaned
        
        return clean_dict(schema)
    
    @fast_trace
    def get_converted_schema(self, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取缓存的Schema转换结果
        
        Args:
            schema: JSON Schema定义
            
        Returns:
            转换后的Agently格式Schema，如果未缓存则返回None
        """
        key = self._generate_key(schema)
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
                
            if entry.is_expired:
                del self._cache[key]
                return None
                
            entry.touch()
            return entry.value
    
    @fast_trace
    def set_converted_schema(
        self, 
        schema: Dict[str, Any], 
        converted_schema: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """设置Schema转换结果缓存
        
        Args:
            schema: 原始JSON Schema
            converted_schema: 转换后的Agently格式Schema
            ttl: 生存时间（秒），如果为None则使用默认值
        """
        key = self._generate_key(schema)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # 检查缓存大小限制
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            entry = ParameterCacheEntry(
                key=key,
                value=converted_schema,
                created_at=time.time(),
                ttl=ttl,
                cache_type="schema"
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
        """清理过期条目
        
        Returns:
            清理的条目数量
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
                
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        with self._lock:
            total_access = sum(entry.access_count for entry in self._cache.values())
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_access': total_access,
                'hit_rate': 0.0 if total_access == 0 else len(self._cache) / total_access,
                'cache_type': 'schema'
            }


class ConfigCache:
    """配置参数缓存
    
    缓存客户端配置和请求参数的组合结果，避免重复配置计算
    """
    
    def __init__(self, max_size: int = 500, default_ttl: int = 1800):
        """初始化配置缓存
        
        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认生存时间（秒）
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, ParameterCacheEntry] = {}
        self._lock = threading.RLock()
    
    @fast_trace
    def _generate_key(self, config_data: Dict[str, Any]) -> str:
        """生成配置缓存键
        
        Args:
            config_data: 配置数据
            
        Returns:
            缓存键字符串
        """
        # 排除不影响配置的字段
        cache_data = {k: v for k, v in config_data.items() 
                     if k not in ['trace_id', 'timestamp', 'user_id', 'request_id']}
        
        # 确保字典键的顺序一致
        sorted_data = json.dumps(cache_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(sorted_data.encode('utf-8')).hexdigest()
    
    @fast_trace
    def get_config(self, config_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """获取缓存的配置
        
        Args:
            config_data: 配置数据
            
        Returns:
            缓存的配置结果，如果未缓存则返回None
        """
        key = self._generate_key(config_data)
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
                
            if entry.is_expired:
                del self._cache[key]
                return None
                
            entry.touch()
            return entry.value
    
    @fast_trace
    def set_config(
        self, 
        config_data: Dict[str, Any], 
        processed_config: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> None:
        """设置配置缓存
        
        Args:
            config_data: 原始配置数据
            processed_config: 处理后的配置
            ttl: 生存时间（秒），如果为None则使用默认值
        """
        key = self._generate_key(config_data)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # 检查缓存大小限制
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            entry = ParameterCacheEntry(
                key=key,
                value=processed_config,
                created_at=time.time(),
                ttl=ttl,
                cache_type="config"
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
        """清理过期条目
        
        Returns:
            清理的条目数量
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
                
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        Returns:
            缓存统计信息字典
        """
        with self._lock:
            total_access = sum(entry.access_count for entry in self._cache.values())
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'total_access': total_access,
                'hit_rate': 0.0 if total_access == 0 else len(self._cache) / total_access,
                'cache_type': 'config'
            }


class ParameterCacheManager:
    """参数缓存管理器
    
    统一管理Schema缓存和配置缓存，提供清理和统计功能
    """
    
    def __init__(
        self, 
        schema_cache_config: Optional[Dict] = None, 
        config_cache_config: Optional[Dict] = None
    ):
        """初始化参数缓存管理器
        
        Args:
            schema_cache_config: Schema缓存配置
            config_cache_config: 配置缓存配置
        """
        schema_config = schema_cache_config or {}
        config_config = config_cache_config or {}
        
        self.schema_cache = SchemaCache(
            max_size=schema_config.get('max_size', 1000),
            default_ttl=schema_config.get('default_ttl', 3600)
        )
        
        self.config_cache = ConfigCache(
            max_size=config_config.get('max_size', 500),
            default_ttl=config_config.get('default_ttl', 1800)
        )
        
        self._cleanup_interval = 300  # 5分钟清理一次
        self._last_cleanup = time.time()
    
    @fast_trace
    def cleanup_expired(self) -> Dict[str, int]:
        """清理过期缓存
        
        Returns:
            清理统计信息
        """
        current_time = time.time()
        
        # 检查是否需要清理
        if current_time - self._last_cleanup < self._cleanup_interval:
            return {'schema_expired': 0, 'config_expired': 0}
        
        schema_expired = self.schema_cache.clear_expired()
        config_expired = self.config_cache.clear_expired()
        
        self._last_cleanup = current_time
        
        if schema_expired > 0 or config_expired > 0:
            logger.debug(f"清理过期参数缓存: Schema缓存 {schema_expired} 条，配置缓存 {config_expired} 条")
        
        return {
            'schema_expired': schema_expired,
            'config_expired': config_expired
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合缓存统计信息
        
        Returns:
            综合统计信息字典
        """
        return {
            'schema_cache': self.schema_cache.get_stats(),
            'config_cache': self.config_cache.get_stats(),
            'last_cleanup': self._last_cleanup,
            'cleanup_interval': self._cleanup_interval
        }
    
    def clear_all_caches(self) -> None:
        """清空所有缓存"""
        with self.schema_cache._lock:
            self.schema_cache._cache.clear()
            
        with self.config_cache._lock:
            self.config_cache._cache.clear()
            
        logger.info("所有参数缓存已清空")
    
    def set_cleanup_interval(self, interval: int) -> None:
        """设置清理间隔
        
        Args:
            interval: 清理间隔（秒）
        """
        self._cleanup_interval = max(60, interval)  # 最小1分钟


# 全局参数缓存管理器实例
_parameter_cache_manager: Optional[ParameterCacheManager] = None


def get_parameter_cache_manager() -> ParameterCacheManager:
    """获取全局参数缓存管理器实例
    
    Returns:
        全局参数缓存管理器实例
    """
    global _parameter_cache_manager
    if _parameter_cache_manager is None:
        _parameter_cache_manager = ParameterCacheManager()
    return _parameter_cache_manager


def create_parameter_cache_manager(
    schema_cache_config: Optional[Dict] = None,
    config_cache_config: Optional[Dict] = None
) -> ParameterCacheManager:
    """创建参数缓存管理器实例
    
    Args:
        schema_cache_config: Schema缓存配置
        config_cache_config: 配置缓存配置
        
    Returns:
        参数缓存管理器实例
    """
    return ParameterCacheManager(
        schema_cache_config=schema_cache_config,
        config_cache_config=config_cache_config
    )