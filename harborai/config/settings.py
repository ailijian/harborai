#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置管理

定义 HarborAI 的全局配置，包括默认设置、环境变量处理、插件配置等。
"""

import os
from typing import Dict, List, Optional, Any
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from .performance import PerformanceMode, get_performance_config


class Settings(BaseSettings):
    """
    HarborAI 全局配置类
    
    支持从环境变量加载配置，环境变量前缀为 HARBORAI_
    """
    model_config = SettingsConfigDict(
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="HARBORAI_"
    )
    
    # 基础配置
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # API 配置
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    default_timeout: int = Field(default=60, alias="HARBORAI_TIMEOUT", gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_delay: float = Field(default=1.0, ge=0)
    
    # 结构化输出配置
    default_structured_provider: str = Field(default="agently", alias="HARBORAI_STRUCTURED_PROVIDER")
    
    # 数据库配置（PostgreSQL）
    postgres_url: Optional[str] = Field(default=None)
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_user: str = Field(default="harborai")
    postgres_password: str = Field(default="")
    postgres_database: str = Field(default="harborai")
    
    # 日志配置
    enable_async_logging: bool = Field(default=True, alias="HARBORAI_ASYNC_LOGGING")
    log_retention_days: int = Field(default=7)
    
    # 插件配置
    plugin_directories: List[str] = Field(default_factory=lambda: ["harborai.core.plugins"])
    
    # 成本追踪配置
    enable_cost_tracking: bool = Field(default=True, alias="HARBORAI_COST_TRACKING")
    
    # 性能优化配置
    performance_mode: str = Field(default="full", alias="HARBORAI_PERFORMANCE_MODE")  # fast, balanced, full
    enable_fast_path: bool = Field(default=True, alias="HARBORAI_FAST_PATH")
    enable_async_decorators: bool = Field(default=True, alias="HARBORAI_ASYNC_DECORATORS")
    enable_postgres_logging: bool = Field(default=True, alias="HARBORAI_POSTGRES_LOGGING")
    enable_detailed_tracing: bool = Field(default=True, alias="HARBORAI_DETAILED_TRACING")
    
    # 快速路径配置
    fast_path_models: List[str] = Field(default_factory=lambda: ["gpt-3.5-turbo", "gpt-4o-mini"], alias="HARBORAI_FAST_PATH_MODELS")
    fast_path_max_tokens: int = Field(default=1000, alias="HARBORAI_FAST_PATH_MAX_TOKENS")
    fast_path_skip_cost_tracking: bool = Field(default=False, alias="HARBORAI_FAST_PATH_SKIP_COST")
    
    # 缓存配置
    enable_token_cache: bool = Field(default=True, alias="HARBORAI_TOKEN_CACHE")
    token_cache_ttl: int = Field(default=300, alias="HARBORAI_TOKEN_CACHE_TTL")  # 5分钟
    enable_response_cache: bool = Field(default=True, alias="HARBORAI_RESPONSE_CACHE")
    response_cache_ttl: int = Field(default=600, alias="HARBORAI_RESPONSE_CACHE_TTL")  # 10分钟
    cache_cleanup_interval: int = Field(default=300, alias="HARBORAI_CACHE_CLEANUP_INTERVAL")  # 5分钟
    
    # 性能管理器配置
    enable_performance_manager: bool = Field(default=True, alias="HARBORAI_PERFORMANCE_MANAGER")
    enable_background_tasks: bool = Field(default=True, alias="HARBORAI_BACKGROUND_TASKS")
    background_task_workers: int = Field(default=2, alias="HARBORAI_BACKGROUND_WORKERS")
    enable_plugin_preload: bool = Field(default=True, alias="HARBORAI_PLUGIN_PRELOAD")
    plugin_cache_size: int = Field(default=100, alias="HARBORAI_PLUGIN_CACHE_SIZE")
    
    # 模型映射配置
    model_mappings: Dict[str, str] = Field(default_factory=dict)
    
    def get_postgres_url(self) -> Optional[str]:
        """获取 PostgreSQL 连接 URL"""
        if self.postgres_url:
            return self.postgres_url
        
        if self.postgres_password:
            return (
                f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
            )
        return None
    
    def get_plugin_config(self, plugin_name: str) -> Dict[str, Any]:
        """获取特定插件的配置"""
        # 从环境变量中读取插件特定配置
        config = {}
        prefix = f"{plugin_name.upper()}_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                config[config_key] = value
        
        return config
    
    def is_fast_path_enabled(self, model: str, max_tokens: Optional[int] = None) -> bool:
        """判断是否应该使用快速路径"""
        if not self.enable_fast_path:
            return False
        
        # 检查性能模式
        if self.performance_mode == "fast":
            return True
        elif self.performance_mode == "full":
            return False
        
        # balanced 模式下的判断逻辑
        if model in self.fast_path_models:
            if max_tokens is None or max_tokens <= self.fast_path_max_tokens:
                return True
        
        return False
    
    def get_decorator_config(self) -> Dict[str, bool]:
        """获取装饰器启用配置"""
        return {
            "cost_tracking": self.enable_cost_tracking and not (self.performance_mode == "fast" and self.fast_path_skip_cost_tracking),
            "postgres_logging": self.enable_postgres_logging and self.performance_mode != "fast",
            "detailed_tracing": self.enable_detailed_tracing,
            "async_decorators": self.enable_async_decorators
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能管理器配置"""
        return {
            "enabled": self.enable_performance_manager,
            "background_tasks": {
                "enabled": self.enable_background_tasks,
                "workers": self.background_task_workers
            },
            "cache": {
                "token_cache": self.enable_token_cache,
                "token_cache_ttl": self.token_cache_ttl,
                "response_cache": self.enable_response_cache,
                "response_cache_ttl": self.response_cache_ttl,
                "cleanup_interval": self.cache_cleanup_interval
            },
            "plugins": {
                "preload": self.enable_plugin_preload,
                "cache_size": self.plugin_cache_size
            }
        }
    
    def get_current_performance_config(self):
        """
        获取当前性能配置实例
        
        Returns:
            PerformanceConfig: 当前性能配置实例
        """
        return get_performance_config()
    
    def set_performance_mode(self, mode: str) -> None:
        """
        设置性能模式并重置性能配置
        
        Args:
            mode: 性能模式 ('fast', 'balanced', 'full')
        """
        from .performance import reset_performance_config
        self.performance_mode = mode
        reset_performance_config(PerformanceMode(mode))


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置实例（单例模式）"""
    return Settings()