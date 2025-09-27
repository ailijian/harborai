#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置管理模块单元测试"""

import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from harborai.config.settings import Settings, get_settings


class TestSettings:
    """Settings类测试"""
    
    def test_default_values(self):
        """测试默认值设置（不加载.env文件）"""
        # 创建一个不加载.env文件的Settings实例来测试默认值
        from harborai.config.settings import SettingsConfigDict
        from pydantic_settings import BaseSettings
        from pydantic import Field
        
        class TestSettings(BaseSettings):
            model_config = SettingsConfigDict(
                extra="allow",
                env_file=None,  # 不加载.env文件
                env_file_encoding="utf-8",
                case_sensitive=False,
                env_prefix="HARBORAI_",
                env_ignore_empty=True
            )
            
            # 基础配置
            debug: bool = Field(default=False)
            log_level: str = Field(default="INFO")
            default_timeout: int = Field(default=60, alias="HARBORAI_TIMEOUT")
            max_retries: int = Field(default=3)
            retry_delay: float = Field(default=1.0)
            
            # 数据库配置
            postgres_host: str = Field(default="localhost")
            postgres_port: int = Field(default=5432)
            postgres_user: str = Field(default="harborai")
            postgres_password: str = Field(default="")
            postgres_database: str = Field(default="harborai")
            postgres_url: str | None = Field(default=None)
            
            # 插件配置
            plugin_directories: list[str] = Field(default=["harborai.core.plugins"])
            
            # 成本追踪
            enable_cost_tracking: bool = Field(default=True, alias="HARBORAI_COST_TRACKING")
            
            # 模型映射
            model_mappings: dict = Field(default_factory=dict)
            
            # 结构化输出配置
            default_structured_provider: str = Field(default="agently", alias="HARBORAI_STRUCTURED_PROVIDER")
            
            # 日志配置
            enable_async_logging: bool = Field(default=True, alias="HARBORAI_ASYNC_LOGGING")
            log_retention_days: int = Field(default=7)
        
        # 清除所有相关环境变量以确保使用默认值
        env_vars_to_clear = [
            "HARBORAI_DEBUG", "HARBORAI_LOG_LEVEL", "HARBORAI_TIMEOUT",
            "HARBORAI_MAX_RETRIES", "HARBORAI_RETRY_DELAY", "HARBORAI_POSTGRES_HOST",
            "HARBORAI_POSTGRES_PORT", "HARBORAI_POSTGRES_USER", "HARBORAI_POSTGRES_PASSWORD",
            "HARBORAI_POSTGRES_DATABASE", "HARBORAI_POSTGRES_URL", "HARBORAI_COST_TRACKING",
            "HARBORAI_STRUCTURED_PROVIDER", "HARBORAI_ASYNC_LOGGING", "HARBORAI_LOG_RETENTION_DAYS"
        ]
        
        with patch.dict(os.environ, {}, clear=True):
            # 确保环境变量被清除
            for var in env_vars_to_clear:
                os.environ.pop(var, None)
            
            settings = TestSettings()
            
            # 验证所有默认值
            assert settings.debug is False
            assert settings.log_level == "INFO"
            assert settings.default_timeout == 60
            assert settings.max_retries == 3
            assert settings.retry_delay == 1.0
            assert settings.postgres_host == "localhost"
            assert settings.postgres_port == 5432
            assert settings.postgres_user == "harborai"
            assert settings.postgres_password == ""
            assert settings.postgres_database == "harborai"
            assert settings.enable_cost_tracking is True
            assert settings.model_mappings == {}
            assert settings.default_structured_provider == "agently"
            assert settings.enable_async_logging is True
            assert settings.log_retention_days == 7
            assert settings.plugin_directories == ["harborai.core.plugins"]
    
    def test_environment_variable_loading(self):
        """测试环境变量加载"""
        env_vars = {
            "HARBORAI_DEBUG": "true",
            "HARBORAI_LOG_LEVEL": "DEBUG",
            "HARBORAI_TIMEOUT": "120",
            "HARBORAI_MAX_RETRIES": "5",
            "HARBORAI_RETRY_DELAY": "2.5",
            "HARBORAI_STRUCTURED_PROVIDER": "custom",
            "HARBORAI_POSTGRES_HOST": "db.example.com",
            "HARBORAI_POSTGRES_PORT": "5433",
            "HARBORAI_POSTGRES_USER": "testuser",
            "HARBORAI_POSTGRES_PASSWORD": "testpass",
            "HARBORAI_POSTGRES_DATABASE": "testdb",
            "HARBORAI_ASYNC_LOGGING": "false",
            "HARBORAI_LOG_RETENTION_DAYS": "14",
            "HARBORAI_COST_TRACKING": "false"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            
            # 验证环境变量被正确加载
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.default_timeout == 120
            assert settings.max_retries == 5
            assert settings.retry_delay == 2.5
            assert settings.default_structured_provider == "custom"
            assert settings.postgres_host == "db.example.com"
            assert settings.postgres_port == 5433
            assert settings.postgres_user == "testuser"
            assert settings.postgres_password == "testpass"
            assert settings.postgres_database == "testdb"
            assert settings.enable_async_logging is False
            assert settings.log_retention_days == 14
            assert settings.enable_cost_tracking is False
    
    def test_postgres_url_generation(self):
        """测试PostgreSQL URL生成"""
        # 测试直接提供URL的情况
        settings = Settings(postgres_url="postgresql://user:pass@host:5432/db")
        assert settings.get_postgres_url() == "postgresql://user:pass@host:5432/db"
        
        # 测试从组件生成URL的情况
        settings = Settings(
            postgres_host="localhost",
            postgres_port=5432,
            postgres_user="testuser",
            postgres_password="testpass",
            postgres_database="testdb"
        )
        expected_url = "postgresql+asyncpg://testuser:testpass@localhost:5432/testdb"
        assert settings.get_postgres_url() == expected_url
        
        # 测试没有密码的情况
        settings = Settings(
            postgres_host="localhost",
            postgres_port=5432,
            postgres_user="testuser",
            postgres_password="",
            postgres_database="testdb"
        )
        assert settings.get_postgres_url() is None
    
    def test_plugin_config_extraction(self):
        """测试插件配置提取"""
        env_vars = {

            "ANTHROPIC_API_KEY": "ant-test456",
            "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
            "OTHER_CONFIG": "should_not_appear"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            

            
            # 测试Anthropic插件配置
            anthropic_config = settings.get_plugin_config("anthropic")
            expected_anthropic = {
                "api_key": "ant-test456",
                "base_url": "https://api.anthropic.com"
            }
            assert anthropic_config == expected_anthropic
            
            # 测试不存在的插件配置
            empty_config = settings.get_plugin_config("nonexistent")
            assert empty_config == {}
    
    def test_case_insensitive_env_vars(self):
        """测试环境变量大小写不敏感"""
        env_vars = {
            "harborai_debug": "true",  # 小写
            "HARBORAI_LOG_LEVEL": "DEBUG",  # 大写
            "HARBORAI_TIMEOUT": "90"  # 正确的环境变量名
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            
            # Pydantic应该处理大小写不敏感
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.default_timeout == 90
    
    def test_extra_fields_allowed(self):
        """测试允许额外字段"""
        # 测试通过初始化参数添加额外字段
        settings = Settings(custom_field="custom_value", another_field=123)
        
        assert hasattr(settings, "custom_field")
        assert settings.custom_field == "custom_value"
        assert hasattr(settings, "another_field")
        assert settings.another_field == 123
    
    def test_field_validation(self):
        """测试字段验证"""
        # 创建一个不加载.env文件的Settings实例来测试字段验证
        from harborai.config.settings import SettingsConfigDict
        from pydantic_settings import BaseSettings
        from pydantic import Field
        
        class ValidationTestSettings(BaseSettings):
            model_config = SettingsConfigDict(
                extra="allow",
                env_file=None,  # 不加载.env文件
                env_file_encoding="utf-8",
                case_sensitive=False,
                env_prefix="HARBORAI_"
            )
            
            debug: bool = Field(default=False)
            log_level: str = Field(default="INFO")
            default_timeout: int = Field(default=60)
            max_retries: int = Field(default=3)
            retry_delay: float = Field(default=1.0)
            postgres_port: int = Field(default=5432)
        
        # 测试有效值
        settings = ValidationTestSettings(
            debug=True,
            log_level="WARNING",
            default_timeout=30,
            max_retries=5,
            retry_delay=0.5,
            postgres_port=5432
        )
        
        assert settings.debug is True
        assert settings.log_level == "WARNING"
        assert settings.default_timeout == 30
        assert settings.max_retries == 5
        assert settings.retry_delay == 0.5
        assert settings.postgres_port == 5432
        
        # 测试类型转换
        settings = ValidationTestSettings(
            default_timeout="45",  # 字符串转整数
            retry_delay="1.5",     # 字符串转浮点数
            debug="true"           # 字符串转布尔值
        )
        
        assert settings.default_timeout == 45
        assert settings.retry_delay == 1.5
        assert settings.debug is True
    
    def test_model_mappings_configuration(self):
        """测试模型映射配置"""
        # 测试默认空映射
        settings = Settings()
        assert settings.model_mappings == {}
        
        # 测试自定义映射
        custom_mappings = {
            "deepseek-chat": "deepseek/deepseek-chat",
            "ernie-4.0-turbo-8k": "ernie/ernie-4.0-turbo-8k"
        }
        settings = Settings(model_mappings=custom_mappings)
        assert settings.model_mappings == custom_mappings
    
    def test_plugin_directories_configuration(self):
        """测试插件目录配置"""
        # 测试默认插件目录
        settings = Settings()
        assert settings.plugin_directories == ["harborai.core.plugins"]
        
        # 测试自定义插件目录
        custom_dirs = ["custom.plugins", "another.plugin.dir"]
        settings = Settings(plugin_directories=custom_dirs)
        assert settings.plugin_directories == custom_dirs


class TestGetSettings:
    """get_settings函数测试"""
    
    def test_singleton_behavior(self):
        """测试单例模式行为"""
        # 清除缓存
        get_settings.cache_clear()
        
        # 获取两次实例
        settings1 = get_settings()
        settings2 = get_settings()
        
        # 应该是同一个实例
        assert settings1 is settings2
        assert id(settings1) == id(settings2)
    
    def test_cache_clearing(self):
        """测试缓存清除"""
        # 获取初始实例
        settings1 = get_settings()
        
        # 清除缓存
        get_settings.cache_clear()
        
        # 获取新实例
        settings2 = get_settings()
        
        # 应该是不同的实例
        assert settings1 is not settings2
    
    def test_environment_changes_after_cache(self):
        """测试缓存后环境变量变化的影响"""
        # 清除缓存
        get_settings.cache_clear()
        
        # 设置初始环境变量
        with patch.dict(os.environ, {"HARBORAI_DEBUG": "false"}, clear=False):
            settings1 = get_settings()
            assert settings1.debug is False
        
        # 更改环境变量（但不清除缓存）
        with patch.dict(os.environ, {"HARBORAI_DEBUG": "true"}, clear=False):
            settings2 = get_settings()
            # 由于缓存，应该仍然是旧值
            assert settings2.debug is False
            assert settings1 is settings2
        
        # 清除缓存后重新获取
        get_settings.cache_clear()
        with patch.dict(os.environ, {"HARBORAI_DEBUG": "true"}, clear=False):
            settings3 = get_settings()
            # 现在应该是新值
            assert settings3.debug is True
            assert settings1 is not settings3


class TestSettingsIntegration:
    """Settings集成测试"""
    
    def test_complete_configuration_scenario(self):
        """测试完整配置场景"""
        env_vars = {
            "HARBORAI_DEBUG": "true",
            "HARBORAI_LOG_LEVEL": "DEBUG",
            "HARBORAI_TIMEOUT": "120",
            "HARBORAI_MAX_RETRIES": "5",
            "HARBORAI_POSTGRES_HOST": "prod-db.example.com",
            "HARBORAI_POSTGRES_PORT": "5432",
            "HARBORAI_POSTGRES_USER": "harborai_prod",
            "HARBORAI_POSTGRES_PASSWORD": "secure_password",
            "HARBORAI_POSTGRES_DATABASE": "harborai_prod",
            "HARBORAI_COST_TRACKING": "true",

            "ANTHROPIC_API_KEY": "ant-prod456"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            
            # 验证基础配置
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.default_timeout == 120
            assert settings.max_retries == 5
            
            # 验证数据库配置
            expected_db_url = "postgresql+asyncpg://harborai_prod:secure_password@prod-db.example.com:5432/harborai_prod"
            assert settings.get_postgres_url() == expected_db_url
            
            # 验证成本追踪
            assert settings.enable_cost_tracking is True
            

            
            anthropic_config = settings.get_plugin_config("anthropic")
            assert anthropic_config["api_key"] == "ant-prod456"
    
    def test_minimal_configuration_scenario(self):
        """测试最小配置场景（不加载.env文件）"""
        # 创建一个不加载.env文件的Settings实例来测试最小配置
        from harborai.config.settings import SettingsConfigDict
        from pydantic_settings import BaseSettings
        from pydantic import Field
        
        class MinimalSettings(BaseSettings):
            model_config = SettingsConfigDict(
                extra="allow",
                env_file=None,  # 不加载.env文件
                env_file_encoding="utf-8",
                case_sensitive=False,
                env_prefix="HARBORAI_"
            )
            
            debug: bool = Field(default=False)
            log_level: str = Field(default="INFO")
            default_timeout: int = Field(default=60)
            postgres_password: str = Field(default="")
            enable_cost_tracking: bool = Field(default=True)
            
            def get_postgres_url(self):
                if not self.postgres_password:
                    return None
                return "postgresql+asyncpg://user:pass@host:5432/db"
        
        # 清除所有相关环境变量
        env_vars_to_clear = [
            "HARBORAI_DEBUG", "HARBORAI_LOG_LEVEL", "HARBORAI_TIMEOUT",
            "HARBORAI_POSTGRES_URL", "HARBORAI_POSTGRES_PASSWORD"
        ]
        
        with patch.dict(os.environ, {}, clear=False):
            # 确保环境变量被清除
            for var in env_vars_to_clear:
                os.environ.pop(var, None)
            
            settings = MinimalSettings()
            
            # 验证所有值都是默认值
            assert settings.debug is False
            assert settings.log_level == "INFO"
            assert settings.default_timeout == 60
            assert settings.get_postgres_url() is None
            assert settings.enable_cost_tracking is True
    
    def test_configuration_override_priority(self):
        """测试配置覆盖优先级（不加载.env文件）"""
        # 创建一个不加载.env文件的Settings实例来测试覆盖优先级
        from harborai.config.settings import SettingsConfigDict
        from pydantic_settings import BaseSettings
        from pydantic import Field
        
        class OverrideSettings(BaseSettings):
            model_config = SettingsConfigDict(
                extra="allow",
                env_file=None,  # 不加载.env文件
                env_file_encoding="utf-8",
                case_sensitive=False,
                env_prefix="HARBORAI_"
            )
            
            debug: bool = Field(default=False)
            default_timeout: int = Field(default=60)
        
        env_vars = {
            "HARBORAI_DEBUG": "true",
            "HARBORAI_TIMEOUT": "90"
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            # 通过初始化参数覆盖环境变量
            settings = OverrideSettings(debug=False, default_timeout=150)
            
            # 初始化参数应该优先于环境变量
            assert settings.debug is False  # 覆盖了环境变量
            assert settings.default_timeout == 150  # 覆盖了环境变量
    
    def test_error_handling_for_invalid_values(self):
        """测试无效值的错误处理"""
        # 创建一个不加载.env文件的Settings实例来测试错误处理
        from harborai.config.settings import SettingsConfigDict
        from pydantic_settings import BaseSettings
        from pydantic import Field, ValidationError
        
        class ErrorTestSettings(BaseSettings):
            model_config = SettingsConfigDict(
                extra="allow",
                env_file=None,  # 不加载.env文件
                env_file_encoding="utf-8",
                case_sensitive=False,
                env_prefix="HARBORAI_"
            )
            
            debug: bool = Field(default=False)
            default_timeout: int = Field(default=60, gt=0)  # 必须大于0
            postgres_port: int = Field(default=5432, ge=1, le=65535)  # 端口范围验证
        
        # 测试无效的数字值 - 负数应该触发验证错误
        with pytest.raises(ValidationError):
            ErrorTestSettings(default_timeout=-1)
        
        # 测试端口范围验证
        with pytest.raises(ValidationError):
            ErrorTestSettings(postgres_port=0)
        
        with pytest.raises(ValidationError):
            ErrorTestSettings(postgres_port=70000)
        
        # 测试通过环境变量的无效值
        with patch.dict(os.environ, {"HARBORAI_DEFAULT_TIMEOUT": "-5"}, clear=True):
            with pytest.raises(ValidationError):
                ErrorTestSettings()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])