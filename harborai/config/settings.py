#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置管理

定义 HarborAI 的全局配置，包括默认设置、环境变量处理、插件配置等。
"""

import os
import json
from typing import Dict, List, Optional, Any, Union, Annotated
from pydantic import Field, field_validator, BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from .performance import PerformanceMode, get_performance_config


def parse_comma_separated_list(v: Union[str, List[str]]) -> List[str]:
    """解析逗号分隔的字符串为列表"""
    if isinstance(v, str):
        return [item.strip() for item in v.split(',') if item.strip()]
    elif isinstance(v, list):
        return v
    else:
        return []


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
        env_prefix="HARBORAI_",
        env_parse_none_str="None",
        env_parse_enums=False,
        env_nested_delimiter=None
    )
    
    # 基础配置
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # API 配置
    api_key: Optional[str] = Field(default=None)
    base_url: Optional[str] = Field(default=None)
    default_timeout: int = Field(default=60, alias="HARBORAI_TIMEOUT", gt=0)
    request_timeout: int = Field(default=90, alias="REQUEST_TIMEOUT", gt=0)
    connect_timeout: int = Field(default=30, alias="CONNECT_TIMEOUT", gt=0)
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
    file_log_directory: str = Field(default="./logs", alias="HARBORAI_FILE_LOG_DIR")
    
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
    fast_path_max_tokens: Optional[int] = Field(default=None, alias="HARBORAI_FAST_PATH_MAX_TOKENS")  # None表示无限制，由模型厂商控制
    fast_path_skip_cost_tracking: bool = Field(default=False, alias="HARBORAI_FAST_PATH_SKIP_COST")
    
    @property
    def fast_path_models(self) -> List[str]:
        """获取快速路径模型列表"""
        env_value = os.environ.get("HARBORAI_FAST_PATH_MODELS", "")
        if env_value:
            return parse_comma_separated_list(env_value)
        return ["gpt-3.5-turbo", "gpt-4o-mini"]
    
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
    
    # 降级策略配置
    enable_fallback: bool = Field(default=True, alias="HARBORAI_ENABLE_FALLBACK")
    # 临时注释掉 fallback_models 字段以解决解析问题
    # fallback_models: List[str] = Field(
    #     default_factory=lambda: ["deepseek-chat", "ernie-4.0-turbo-8k", "doubao-1-5-pro-32k-character-250715"], 
    #     alias="HARBORAI_FALLBACK_MODELS"
    # )
    
    # @field_validator('fallback_models', mode='before')
    # @classmethod
    # def parse_fallback_models(cls, v):
    #     """解析降级模型列表"""
    #     return parse_comma_separated_list(v)
    
    @property
    def fallback_models(self) -> List[str]:
        """获取降级模型列表"""
        env_value = os.environ.get("HARBORAI_FALLBACK_MODELS", "")
        if env_value:
            return parse_comma_separated_list(env_value)
        return ["deepseek-chat", "ernie-4.0-turbo-8k", "doubao-1-5-pro-32k-character-250715"]

    
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
        """
        判断是否应该使用快速路径
        
        快速路径和性能模式是正交的：
        - enable_fast_path 控制快速路径的总开关
        - performance_mode 控制功能的启用程度，但不影响快速路径
        - FAST 模式强制启用快速路径（忽略 enable_fast_path 设置）
        """
        # FAST 模式：强制启用快速路径（最大化性能），忽略 enable_fast_path 设置
        if self.performance_mode == "fast":
            return True
        
        # BALANCED 和 FULL 模式：根据 enable_fast_path 和模型/token 限制判断
        if not self.enable_fast_path:
            return False
        
        # FULL 模式不再强制禁用快速路径，允许用户通过 HARBORAI_FAST_PATH 控制
        if model in self.fast_path_models:
            # 如果 fast_path_max_tokens 为 None，表示无限制，允许快速路径
            if self.fast_path_max_tokens is None:
                return True
            # 如果用户未指定 max_tokens 或者指定的值在限制范围内，允许快速路径
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
    """获取全局配置实例"""
    # 显式加载 .env 文件
    from dotenv import load_dotenv
    load_dotenv()
    return Settings()