#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目配置管理

定义 HarborAI 的全局配置，包括默认设置、环境变量处理、插件配置等。
"""

import os
from typing import Dict, List, Optional, Any
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """
    HarborAI 全局配置类
    
    支持从环境变量加载配置，环境变量前缀为 HARBORAI_
    """
    model_config = {
        "extra": "allow",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False
    }
    
    # 基础配置
    debug: bool = Field(default=False, env="HARBORAI_DEBUG")
    log_level: str = Field(default="INFO", env="HARBORAI_LOG_LEVEL")
    
    # API 配置
    api_key: Optional[str] = Field(default=None, env="HARBORAI_API_KEY")
    base_url: Optional[str] = Field(default=None, env="HARBORAI_BASE_URL")
    default_timeout: int = Field(default=60, env="HARBORAI_TIMEOUT")
    max_retries: int = Field(default=3, env="HARBORAI_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="HARBORAI_RETRY_DELAY")
    
    # 结构化输出配置
    default_structured_provider: str = Field(default="agently", env="HARBORAI_STRUCTURED_PROVIDER")
    
    # 数据库配置（PostgreSQL）
    postgres_url: Optional[str] = Field(default=None, env="HARBORAI_POSTGRES_URL")
    postgres_host: str = Field(default="localhost", env="HARBORAI_POSTGRES_HOST")
    postgres_port: int = Field(default=5432, env="HARBORAI_POSTGRES_PORT")
    postgres_user: str = Field(default="harborai", env="HARBORAI_POSTGRES_USER")
    postgres_password: str = Field(default="", env="HARBORAI_POSTGRES_PASSWORD")
    postgres_database: str = Field(default="harborai", env="HARBORAI_POSTGRES_DATABASE")
    
    # 日志配置
    enable_async_logging: bool = Field(default=True, env="HARBORAI_ASYNC_LOGGING")
    log_retention_days: int = Field(default=7, env="HARBORAI_LOG_RETENTION_DAYS")
    
    # 插件配置
    plugin_directories: List[str] = Field(default_factory=lambda: ["harborai.core.plugins"])
    
    # 成本追踪配置
    enable_cost_tracking: bool = Field(default=True, env="HARBORAI_COST_TRACKING")
    
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


@lru_cache()
def get_settings() -> Settings:
    """获取全局配置实例（单例模式）"""
    return Settings()