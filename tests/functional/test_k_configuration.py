# -*- coding: utf-8 -*-
"""
HarborAI 配置管理测试模块

本模块测试配置管理系统的各项功能，包括：
- 配置加载和解析
- 配置验证和校验
- 配置热重载
- 环境变量处理
- 配置文件格式支持
- 配置优先级处理
- 配置缓存机制
- 配置安全性

作者: HarborAI Team
创建时间: 2024-01-20
"""

import pytest
import os
import json
import yaml
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from unittest.mock import Mock, patch, MagicMock
from decimal import Decimal
from datetime import datetime, timedelta
import threading
import configparser
import toml


# 模拟配置数据类
@dataclass
class ModelConfig:
    """模型配置类"""
    name: str
    provider: str
    api_key: str
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    retry_count: int = 3
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    """提供商配置类"""
    name: str
    api_key: str
    base_url: str
    models: List[str] = field(default_factory=list)
    rate_limit: int = 100
    priority: int = 1
    enabled: bool = True
    features: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """系统配置类"""
    debug: bool = False
    log_level: str = "INFO"
    cache_enabled: bool = True
    cache_ttl: int = 3600
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    retry_enabled: bool = True
    fallback_enabled: bool = True
    monitoring_enabled: bool = True
    cost_tracking_enabled: bool = True


@dataclass
class HarborAIConfig:
    """HarborAI主配置类"""
    system: SystemConfig
    providers: Dict[str, ProviderConfig] = field(default_factory=dict)
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    version: str = "1.0.0"
    last_updated: Optional[datetime] = None


# 模拟配置管理器
class MockConfigManager:
    """模拟配置管理器"""
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = config_dir or tempfile.mkdtemp()
        self.config: Optional[HarborAIConfig] = None
        self.watchers: List[callable] = []
        self.cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        self.file_watchers: Dict[str, threading.Thread] = {}
        self.validation_rules: Dict[str, callable] = {}
        self._lock = threading.Lock()
    
    def load_config(self, config_path: str) -> bool:
        """加载配置文件"""
        try:
            if not os.path.exists(config_path):
                return False
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    data = json.load(f)
                elif config_path.endswith(('.yml', '.yaml')):
                    data = yaml.safe_load(f)
                elif config_path.endswith('.toml'):
                    data = toml.load(f)
                else:
                    return False
            
            # 解析配置数据
            system_config = SystemConfig(**data.get('system', {}))
            
            providers = {}
            for name, provider_data in data.get('providers', {}).items():
                providers[name] = ProviderConfig(name=name, **provider_data)
            
            models = {}
            for name, model_data in data.get('models', {}).items():
                models[name] = ModelConfig(name=name, **model_data)
            
            self.config = HarborAIConfig(
                system=system_config,
                providers=providers,
                models=models,
                version=data.get('version', '1.0.0'),
                last_updated=datetime.now()
            )
            
            return True
            
        except Exception as e:
            print(f"配置加载失败: {e}")
            return False
    
    def validate_config(self, config: HarborAIConfig) -> List[str]:
        """验证配置"""
        errors = []
        
        # 验证系统配置
        if config.system.log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            errors.append("无效的日志级别")
        
        if config.system.cache_ttl <= 0:
            errors.append("缓存TTL必须大于0")
        
        if config.system.max_concurrent_requests <= 0:
            errors.append("最大并发请求数必须大于0")
        
        # 验证提供商配置
        for name, provider in config.providers.items():
            if not provider.api_key:
                errors.append(f"提供商 {name} 缺少API密钥")
            
            if not provider.base_url:
                errors.append(f"提供商 {name} 缺少基础URL")
            
            if provider.rate_limit <= 0:
                errors.append(f"提供商 {name} 速率限制必须大于0")
        
        # 验证模型配置
        for name, model in config.models.items():
            if not model.provider:
                errors.append(f"模型 {name} 缺少提供商")
            
            if model.provider not in config.providers:
                errors.append(f"模型 {name} 的提供商 {model.provider} 不存在")
            
            if model.max_tokens <= 0:
                errors.append(f"模型 {name} 最大token数必须大于0")
            
            if not 0 <= model.temperature <= 2:
                errors.append(f"模型 {name} 温度值必须在0-2之间")
        
        return errors
    
    def save_config(self, config_path: str) -> bool:
        """保存配置文件"""
        try:
            if not self.config:
                return False
            
            # 转换为字典
            data = {
                'version': self.config.version,
                'system': {
                    'debug': self.config.system.debug,
                    'log_level': self.config.system.log_level,
                    'cache_enabled': self.config.system.cache_enabled,
                    'cache_ttl': self.config.system.cache_ttl,
                    'max_concurrent_requests': self.config.system.max_concurrent_requests,
                    'request_timeout': self.config.system.request_timeout,
                    'retry_enabled': self.config.system.retry_enabled,
                    'fallback_enabled': self.config.system.fallback_enabled,
                    'monitoring_enabled': self.config.system.monitoring_enabled,
                    'cost_tracking_enabled': self.config.system.cost_tracking_enabled
                },
                'providers': {},
                'models': {}
            }
            
            # 添加提供商配置
            for name, provider in self.config.providers.items():
                data['providers'][name] = {
                    'api_key': provider.api_key,
                    'base_url': provider.base_url,
                    'models': provider.models,
                    'rate_limit': provider.rate_limit,
                    'priority': provider.priority,
                    'enabled': provider.enabled,
                    'features': provider.features
                }
            
            # 添加模型配置
            for name, model in self.config.models.items():
                data['models'][name] = {
                    'provider': model.provider,
                    'api_key': model.api_key,
                    'base_url': model.base_url,
                    'max_tokens': model.max_tokens,
                    'temperature': model.temperature,
                    'timeout': model.timeout,
                    'retry_count': model.retry_count,
                    'enabled': model.enabled,
                    'metadata': model.metadata
                }
            
            # 保存文件
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.endswith('.json'):
                    json.dump(data, f, indent=2, ensure_ascii=False)
                elif config_path.endswith(('.yml', '.yaml')):
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
                elif config_path.endswith('.toml'):
                    toml.dump(data, f)
                else:
                    return False
            
            return True
            
        except Exception as e:
            print(f"配置保存失败: {e}")
            return False
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """获取配置值"""
        if not self.config:
            return default
        
        # 检查缓存
        if key_path in self.cache:
            if key_path in self.cache_ttl and datetime.now() < self.cache_ttl[key_path]:
                return self.cache[key_path]
        
        # 解析键路径
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                if hasattr(value, key):
                    value = getattr(value, key)
                elif isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            # 缓存结果
            self.cache[key_path] = value
            self.cache_ttl[key_path] = datetime.now() + timedelta(seconds=300)  # 5分钟缓存
            
            return value
            
        except Exception:
            return default
    
    def set_config_value(self, key_path: str, value: Any) -> bool:
        """设置配置值"""
        if not self.config:
            return False
        
        try:
            keys = key_path.split('.')
            target = self.config
            
            # 导航到目标对象
            for key in keys[:-1]:
                if hasattr(target, key):
                    target = getattr(target, key)
                elif isinstance(target, dict) and key in target:
                    target = target[key]
                else:
                    return False
            
            # 设置值
            final_key = keys[-1]
            if hasattr(target, final_key):
                setattr(target, final_key, value)
            elif isinstance(target, dict):
                target[final_key] = value
            else:
                return False
            
            # 清除相关缓存
            keys_to_remove = [k for k in self.cache.keys() if k.startswith(key_path)]
            for k in keys_to_remove:
                del self.cache[k]
                if k in self.cache_ttl:
                    del self.cache_ttl[k]
            
            # 通知观察者
            self._notify_watchers(key_path, value)
            
            return True
            
        except Exception as e:
            print(f"设置配置值失败: {e}")
            return False
    
    def watch_config(self, callback: callable) -> str:
        """监听配置变化"""
        watcher_id = f"watcher_{len(self.watchers)}"
        self.watchers.append((watcher_id, callback))
        return watcher_id
    
    def unwatch_config(self, watcher_id: str) -> bool:
        """取消监听配置变化"""
        for i, (w_id, _) in enumerate(self.watchers):
            if w_id == watcher_id:
                del self.watchers[i]
                return True
        return False
    
    def _notify_watchers(self, key_path: str, value: Any):
        """通知配置观察者"""
        for watcher_id, callback in self.watchers:
            try:
                callback(key_path, value)
            except Exception as e:
                print(f"配置观察者 {watcher_id} 回调失败: {e}")
    
    def reload_config(self, config_path: str) -> bool:
        """重新加载配置"""
        old_config = self.config
        success = self.load_config(config_path)
        
        if success and old_config:
            # 比较配置变化
            self._detect_config_changes(old_config, self.config)
        
        return success
    
    def _detect_config_changes(self, old_config: HarborAIConfig, new_config: HarborAIConfig):
        """检测配置变化"""
        # 简化的变化检测
        if old_config.system.debug != new_config.system.debug:
            self._notify_watchers('system.debug', new_config.system.debug)
        
        if old_config.system.log_level != new_config.system.log_level:
            self._notify_watchers('system.log_level', new_config.system.log_level)
    
    def clear_cache(self):
        """清除配置缓存"""
        with self._lock:
            self.cache.clear()
            self.cache_ttl.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            now = datetime.now()
            valid_entries = sum(1 for ttl in self.cache_ttl.values() if ttl > now)
            
            return {
                'total_entries': len(self.cache),
                'valid_entries': valid_entries,
                'expired_entries': len(self.cache) - valid_entries,
                'cache_hit_ratio': 0.85  # 模拟值
            }


# 模拟环境变量管理器
class MockEnvironmentManager:
    """模拟环境变量管理器"""
    
    def __init__(self):
        self.env_vars: Dict[str, str] = {}
        self.original_env: Dict[str, str] = {}
    
    def set_env_var(self, key: str, value: str, persist: bool = False):
        """设置环境变量"""
        if key not in self.original_env:
            self.original_env[key] = os.environ.get(key)
        
        self.env_vars[key] = value
        os.environ[key] = value
        
        if persist:
            # 模拟持久化到配置文件
            pass
    
    def get_env_var(self, key: str, default: str = None) -> str:
        """获取环境变量"""
        return self.env_vars.get(key, os.environ.get(key, default))
    
    def remove_env_var(self, key: str):
        """移除环境变量"""
        if key in self.env_vars:
            del self.env_vars[key]
        
        if key in os.environ:
            del os.environ[key]
    
    def restore_env_vars(self):
        """恢复原始环境变量"""
        for key, original_value in self.original_env.items():
            if original_value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = original_value
        
        self.env_vars.clear()
        self.original_env.clear()
    
    def load_env_file(self, env_file_path: str) -> bool:
        """加载.env文件"""
        try:
            if not os.path.exists(env_file_path):
                return False
            
            # 尝试多种编码方式
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(env_file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"无法解码文件 {env_file_path}")
                return False
            
            # 解析内容
            for line in content.splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    self.set_env_var(key, value)
            
            return True
            
        except Exception as e:
            print(f"加载环境文件失败: {e}")
            return False


class TestConfigurationLoading:
    """配置加载测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.configuration
    def test_json_config_loading(self):
        """测试JSON配置文件加载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = MockConfigManager(temp_dir)
            
            # 创建JSON配置文件
            config_data = {
                "version": "1.0.0",
                "system": {
                    "debug": True,
                    "log_level": "DEBUG",
                    "cache_enabled": True,
                    "cache_ttl": 3600
                },
                "providers": {
                    "deepseek": {
                "api_key": "sk-test-key",
                "base_url": "https://api.deepseek.com/v1",
                        "rate_limit": 100,
                        "enabled": True
                    }
                },
                "models": {
                    "deepseek-chat": {
                "provider": "deepseek",
                        "api_key": "sk-test-key",
                        "max_tokens": 4096,
                        "temperature": 0.7
                    }
                }
            }
            
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(config_data, f)
            
            # 加载配置
            success = config_manager.load_config(config_path)
            assert success == True
            
            # 验证配置
            config = config_manager.config
            assert config is not None
            assert config.version == "1.0.0"
            assert config.system.debug == True
            assert config.system.log_level == "DEBUG"
            assert "deepseek" in config.providers
        assert "deepseek-chat" in config.models
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.configuration
    def test_yaml_config_loading(self):
        """测试YAML配置文件加载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = MockConfigManager(temp_dir)
            
            # 创建YAML配置文件
            yaml_content = """
version: "1.0.0"
system:
  debug: false
  log_level: "INFO"
  cache_enabled: true
  cache_ttl: 7200
providers:
  ernie:
    api_key: "test-ernie-key"
    base_url: "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
    rate_limit: 50
    enabled: true
models:
  ernie-3.5-8k:
    provider: "ernie"
    api_key: "test-ernie-key"
    max_tokens: 8192
    temperature: 0.5
"""
            
            config_path = os.path.join(temp_dir, "config.yaml")
            with open(config_path, 'w') as f:
                f.write(yaml_content)
            
            # 加载配置
            success = config_manager.load_config(config_path)
            assert success == True
            
            # 验证配置
            config = config_manager.config
            assert config.system.debug == False
            assert config.system.cache_ttl == 7200
            assert "ernie" in config.providers
            assert config.providers["ernie"].rate_limit == 50
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_invalid_config_loading(self):
        """测试无效配置文件加载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = MockConfigManager(temp_dir)
            
            # 测试不存在的文件
            success = config_manager.load_config("/nonexistent/config.json")
            assert success == False
            
            # 测试无效JSON
            invalid_json_path = os.path.join(temp_dir, "invalid.json")
            with open(invalid_json_path, 'w') as f:
                f.write("{ invalid json }")
            
            success = config_manager.load_config(invalid_json_path)
            assert success == False
            
            # 测试不支持的格式
            unsupported_path = os.path.join(temp_dir, "config.txt")
            with open(unsupported_path, 'w') as f:
                f.write("some text")
            
            success = config_manager.load_config(unsupported_path)
            assert success == False
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_config_file_formats(self):
        """测试多种配置文件格式"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = MockConfigManager(temp_dir)
            
            base_config = {
                "version": "1.0.0",
                "system": {
                    "debug": True,
                    "log_level": "INFO"
                }
            }
            
            # 测试JSON格式
            json_path = os.path.join(temp_dir, "config.json")
            with open(json_path, 'w') as f:
                json.dump(base_config, f)
            
            assert config_manager.load_config(json_path) == True
            assert config_manager.config.version == "1.0.0"
            
            # 测试YAML格式
            yaml_path = os.path.join(temp_dir, "config.yml")
            with open(yaml_path, 'w') as f:
                yaml.dump(base_config, f)
            
            assert config_manager.load_config(yaml_path) == True
            assert config_manager.config.system.debug == True


class TestConfigurationValidation:
    """配置验证测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.configuration
    def test_valid_config_validation(self):
        """测试有效配置验证"""
        config_manager = MockConfigManager()
        
        # 创建有效配置
        system_config = SystemConfig(
            debug=False,
            log_level="INFO",
            cache_enabled=True,
            cache_ttl=3600,
            max_concurrent_requests=10
        )
        
        provider_config = ProviderConfig(
            name="deepseek",
            api_key="sk-test-key",
            base_url="https://api.deepseek.com/v1",
            rate_limit=100
        )
        
        model_config = ModelConfig(
            name="deepseek-chat",
            provider="deepseek",
            api_key="sk-test-key",
            max_tokens=4096,
            temperature=0.7
        )
        
        config = HarborAIConfig(
            system=system_config,
            providers={"deepseek": provider_config},
            models={"deepseek-chat": model_config}
        )
        
        # 验证配置
        errors = config_manager.validate_config(config)
        assert len(errors) == 0
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_invalid_system_config_validation(self):
        """测试无效系统配置验证"""
        config_manager = MockConfigManager()
        
        # 创建无效系统配置
        system_config = SystemConfig(
            debug=False,
            log_level="INVALID",  # 无效日志级别
            cache_enabled=True,
            cache_ttl=-1,  # 无效TTL
            max_concurrent_requests=0  # 无效并发数
        )
        
        config = HarborAIConfig(system=system_config)
        
        # 验证配置
        errors = config_manager.validate_config(config)
        assert len(errors) >= 3
        assert any("无效的日志级别" in error for error in errors)
        assert any("缓存TTL必须大于0" in error for error in errors)
        assert any("最大并发请求数必须大于0" in error for error in errors)
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_invalid_provider_config_validation(self):
        """测试无效提供商配置验证"""
        config_manager = MockConfigManager()
        
        # 创建无效提供商配置
        provider_config = ProviderConfig(
            name="test_provider",
            api_key="",  # 空API密钥
            base_url="",  # 空URL
            rate_limit=-1  # 无效速率限制
        )
        
        system_config = SystemConfig()
        config = HarborAIConfig(
            system=system_config,
            providers={"test_provider": provider_config}
        )
        
        # 验证配置
        errors = config_manager.validate_config(config)
        assert len(errors) >= 3
        assert any("缺少API密钥" in error for error in errors)
        assert any("缺少基础URL" in error for error in errors)
        assert any("速率限制必须大于0" in error for error in errors)
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_invalid_model_config_validation(self):
        """测试无效模型配置验证"""
        config_manager = MockConfigManager()
        
        # 创建无效模型配置
        model_config = ModelConfig(
            name="test_model",
            provider="nonexistent_provider",  # 不存在的提供商
            api_key="sk-test",
            max_tokens=-1,  # 无效token数
            temperature=3.0  # 无效温度值
        )
        
        system_config = SystemConfig()
        config = HarborAIConfig(
            system=system_config,
            models={"test_model": model_config}
        )
        
        # 验证配置
        errors = config_manager.validate_config(config)
        assert len(errors) >= 3
        assert any("提供商 nonexistent_provider 不存在" in error for error in errors)
        assert any("最大token数必须大于0" in error for error in errors)
        assert any("温度值必须在0-2之间" in error for error in errors)


class TestConfigurationHotReload:
    """配置热重载测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_config_hot_reload(self):
        """测试配置热重载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = MockConfigManager(temp_dir)
            
            # 创建初始配置
            initial_config = {
                "version": "1.0.0",
                "system": {
                    "debug": False,
                    "log_level": "INFO"
                }
            }
            
            config_path = os.path.join(temp_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(initial_config, f)
            
            # 加载初始配置
            config_manager.load_config(config_path)
            assert config_manager.config.system.debug == False
            assert config_manager.config.system.log_level == "INFO"
            
            # 修改配置文件
            updated_config = {
                "version": "1.0.1",
                "system": {
                    "debug": True,
                    "log_level": "DEBUG"
                }
            }
            
            with open(config_path, 'w') as f:
                json.dump(updated_config, f)
            
            # 重新加载配置
            success = config_manager.reload_config(config_path)
            assert success == True
            assert config_manager.config.version == "1.0.1"
            assert config_manager.config.system.debug == True
            assert config_manager.config.system.log_level == "DEBUG"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_config_change_notification(self):
        """测试配置变化通知"""
        config_manager = MockConfigManager()
        
        # 设置初始配置
        system_config = SystemConfig(debug=False, log_level="INFO")
        config = HarborAIConfig(system=system_config)
        config_manager.config = config
        
        # 注册配置观察者
        changes = []
        
        def config_watcher(key_path: str, value: Any):
            changes.append((key_path, value))
        
        watcher_id = config_manager.watch_config(config_watcher)
        
        # 修改配置
        config_manager.set_config_value("system.debug", True)
        config_manager.set_config_value("system.log_level", "DEBUG")
        
        # 验证通知
        assert len(changes) == 2
        assert ("system.debug", True) in changes
        assert ("system.log_level", "DEBUG") in changes
        
        # 取消观察者
        success = config_manager.unwatch_config(watcher_id)
        assert success == True
        
        # 再次修改配置，应该没有新通知
        changes.clear()
        config_manager.set_config_value("system.debug", False)
        assert len(changes) == 0
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.configuration
    def test_config_watcher_error_handling(self):
        """测试配置观察者错误处理"""
        config_manager = MockConfigManager()
        
        # 设置初始配置
        system_config = SystemConfig(debug=False)
        config = HarborAIConfig(system=system_config)
        config_manager.config = config
        
        # 注册会抛出异常的观察者
        def failing_watcher(key_path: str, value: Any):
            raise Exception("观察者异常")
        
        config_manager.watch_config(failing_watcher)
        
        # 修改配置，不应该因为观察者异常而失败
        success = config_manager.set_config_value("system.debug", True)
        assert success == True
        assert config_manager.config.system.debug == True


class TestEnvironmentVariables:
    """环境变量测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.configuration
    def test_environment_variable_management(self):
        """测试环境变量管理"""
        env_manager = MockEnvironmentManager()
        
        # 设置环境变量
        env_manager.set_env_var("HARBORAI_API_KEY", "test-key-123")
        env_manager.set_env_var("HARBORAI_DEBUG", "true")
        
        # 获取环境变量
        api_key = env_manager.get_env_var("HARBORAI_API_KEY")
        debug_mode = env_manager.get_env_var("HARBORAI_DEBUG")
        
        assert api_key == "test-key-123"
        assert debug_mode == "true"
        
        # 测试默认值
        unknown_var = env_manager.get_env_var("UNKNOWN_VAR", "default_value")
        assert unknown_var == "default_value"
        
        # 移除环境变量
        env_manager.remove_env_var("HARBORAI_API_KEY")
        removed_var = env_manager.get_env_var("HARBORAI_API_KEY")
        assert removed_var is None
        
        # 恢复环境变量
        env_manager.restore_env_vars()
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_env_file_loading(self):
        """测试.env文件加载"""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_manager = MockEnvironmentManager()
            
            # 创建.env文件
            env_content = """
# HarborAI配置
HARBORAI_API_KEY=sk-test-env-key
HARBORAI_BASE_URL=https://api.example.com
HARBORAI_DEBUG=true
HARBORAI_LOG_LEVEL=DEBUG

# 注释行应该被忽略
# IGNORED_VAR=ignored

# 空行也应该被忽略

HARBORAI_TIMEOUT=30
"""
            
            env_file_path = os.path.join(temp_dir, ".env")
            with open(env_file_path, 'w') as f:
                f.write(env_content)
            
            # 加载.env文件
            success = env_manager.load_env_file(env_file_path)
            assert success == True
            
            # 验证环境变量
            assert env_manager.get_env_var("HARBORAI_API_KEY") == "sk-test-env-key"
            assert env_manager.get_env_var("HARBORAI_BASE_URL") == "https://api.example.com"
            assert env_manager.get_env_var("HARBORAI_DEBUG") == "true"
            assert env_manager.get_env_var("HARBORAI_LOG_LEVEL") == "DEBUG"
            assert env_manager.get_env_var("HARBORAI_TIMEOUT") == "30"
            
            # 验证注释和空行被忽略
            assert env_manager.get_env_var("IGNORED_VAR") is None
            
            # 恢复环境变量
            env_manager.restore_env_vars()
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_env_file_error_handling(self):
        """测试.env文件错误处理"""
        env_manager = MockEnvironmentManager()
        
        # 测试不存在的文件
        success = env_manager.load_env_file("/nonexistent/.env")
        assert success == False
        
        # 测试格式错误的文件
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_env_path = os.path.join(temp_dir, "invalid.env")
            with open(invalid_env_path, 'w') as f:
                f.write("INVALID_LINE_WITHOUT_EQUALS\n")
                f.write("VALID_VAR=valid_value\n")
            
            # 应该能够加载有效的行，忽略无效的行
            success = env_manager.load_env_file(invalid_env_path)
            assert success == True
            assert env_manager.get_env_var("VALID_VAR") == "valid_value"
            
            env_manager.restore_env_vars()


class TestConfigurationCaching:
    """配置缓存测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_config_value_caching(self):
        """测试配置值缓存"""
        config_manager = MockConfigManager()
        
        # 设置配置
        system_config = SystemConfig(debug=True, log_level="DEBUG")
        config = HarborAIConfig(system=system_config)
        config_manager.config = config
        
        # 第一次获取配置值（应该被缓存）
        debug_value1 = config_manager.get_config_value("system.debug")
        assert debug_value1 == True
        
        # 第二次获取相同配置值（应该从缓存获取）
        debug_value2 = config_manager.get_config_value("system.debug")
        assert debug_value2 == True
        
        # 验证缓存中有该值
        assert "system.debug" in config_manager.cache
        
        # 修改配置值（应该清除相关缓存）
        config_manager.set_config_value("system.debug", False)
        
        # 验证缓存被清除
        assert "system.debug" not in config_manager.cache
        
        # 重新获取配置值
        debug_value3 = config_manager.get_config_value("system.debug")
        assert debug_value3 == False
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_cache_expiration(self):
        """测试缓存过期"""
        config_manager = MockConfigManager()
        
        # 设置配置
        system_config = SystemConfig(debug=True)
        config = HarborAIConfig(system=system_config)
        config_manager.config = config
        
        # 获取配置值
        debug_value = config_manager.get_config_value("system.debug")
        assert debug_value == True
        
        # 手动设置缓存过期时间为过去
        config_manager.cache_ttl["system.debug"] = datetime.now() - timedelta(seconds=1)
        
        # 再次获取配置值（应该重新计算，因为缓存已过期）
        debug_value2 = config_manager.get_config_value("system.debug")
        assert debug_value2 == True
        
        # 验证缓存被更新
        assert config_manager.cache_ttl["system.debug"] > datetime.now()
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.configuration
    def test_cache_statistics(self):
        """测试缓存统计"""
        config_manager = MockConfigManager()
        
        # 设置配置
        system_config = SystemConfig(debug=True, log_level="INFO")
        config = HarborAIConfig(system=system_config)
        config_manager.config = config
        
        # 添加一些缓存条目
        config_manager.get_config_value("system.debug")
        config_manager.get_config_value("system.log_level")
        
        # 手动设置一个过期的缓存条目
        config_manager.cache["expired_key"] = "expired_value"
        config_manager.cache_ttl["expired_key"] = datetime.now() - timedelta(seconds=1)
        
        # 获取缓存统计
        stats = config_manager.get_cache_stats()
        
        assert stats["total_entries"] == 3
        assert stats["valid_entries"] == 2
        assert stats["expired_entries"] == 1
        assert "cache_hit_ratio" in stats
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.configuration
    def test_cache_clearing(self):
        """测试缓存清除"""
        config_manager = MockConfigManager()
        
        # 设置配置
        system_config = SystemConfig(debug=True, log_level="INFO")
        config = HarborAIConfig(system=system_config)
        config_manager.config = config
        
        # 添加缓存条目
        config_manager.get_config_value("system.debug")
        config_manager.get_config_value("system.log_level")
        
        # 验证缓存不为空
        assert len(config_manager.cache) > 0
        assert len(config_manager.cache_ttl) > 0
        
        # 清除缓存
        config_manager.clear_cache()
        
        # 验证缓存被清除
        assert len(config_manager.cache) == 0
        assert len(config_manager.cache_ttl) == 0


class TestConfigurationPersistence:
    """配置持久化测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_config_saving(self):
        """测试配置保存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = MockConfigManager(temp_dir)
            
            # 创建配置
            system_config = SystemConfig(
                debug=True,
                log_level="DEBUG",
                cache_enabled=True,
                cache_ttl=7200
            )
            
            provider_config = ProviderConfig(
                name="deepseek",
                api_key="sk-test-save",
                base_url="https://api.deepseek.com/v1",
                rate_limit=150,
                enabled=True
            )
            
            model_config = ModelConfig(
                name="deepseek-chat",
                provider="deepseek",
                api_key="sk-test-save",
                max_tokens=8192,
                temperature=0.8
            )
            
            config = HarborAIConfig(
                system=system_config,
                providers={"deepseek": provider_config},
                models={"deepseek-chat": model_config},
                version="1.2.0"
            )
            
            config_manager.config = config
            
            # 保存配置
            config_path = os.path.join(temp_dir, "saved_config.json")
            success = config_manager.save_config(config_path)
            assert success == True
            
            # 验证文件存在
            assert os.path.exists(config_path)
            
            # 重新加载并验证
            new_config_manager = MockConfigManager(temp_dir)
            load_success = new_config_manager.load_config(config_path)
            assert load_success == True
            
            loaded_config = new_config_manager.config
            assert loaded_config.version == "1.2.0"
            assert loaded_config.system.debug == True
            assert loaded_config.system.cache_ttl == 7200
            assert "deepseek" in loaded_config.providers
            assert loaded_config.providers["deepseek"].rate_limit == 150
            assert "deepseek-chat" in loaded_config.models
            assert loaded_config.models["deepseek-chat"].max_tokens == 8192
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_config_saving_different_formats(self):
        """测试不同格式的配置保存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = MockConfigManager(temp_dir)
            
            # 创建简单配置
            system_config = SystemConfig(debug=False, log_level="INFO")
            config = HarborAIConfig(system=system_config, version="1.0.0")
            config_manager.config = config
            
            # 测试JSON格式保存
            json_path = os.path.join(temp_dir, "config.json")
            success = config_manager.save_config(json_path)
            assert success == True
            assert os.path.exists(json_path)
            
            # 测试YAML格式保存
            yaml_path = os.path.join(temp_dir, "config.yaml")
            success = config_manager.save_config(yaml_path)
            assert success == True
            assert os.path.exists(yaml_path)
            
            # 测试不支持的格式
            txt_path = os.path.join(temp_dir, "config.txt")
            success = config_manager.save_config(txt_path)
            assert success == False
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.configuration
    def test_config_save_without_config(self):
        """测试没有配置时的保存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = MockConfigManager(temp_dir)
            
            # 尝试保存空配置
            config_path = os.path.join(temp_dir, "empty_config.json")
            success = config_manager.save_config(config_path)
            assert success == False


class TestConfigurationSecurity:
    """配置安全性测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_sensitive_data_handling(self):
        """测试敏感数据处理"""
        config_manager = MockConfigManager()
        
        # 创建包含敏感数据的配置
        provider_config = ProviderConfig(
            name="deepseek",
            api_key="sk-very-secret-key-12345",
            base_url="https://api.deepseek.com/v1"
        )
        
        system_config = SystemConfig()
        config = HarborAIConfig(
            system=system_config,
            providers={"deepseek": provider_config}
        )
        
        config_manager.config = config
        
        # 获取API密钥（应该正常返回）
        api_key = config_manager.get_config_value("providers.deepseek.api_key")
        assert api_key == "sk-very-secret-key-12345"
        
        # 验证敏感数据不会被意外暴露在日志中
        # 这里可以添加日志检查逻辑
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.configuration
    def test_config_file_permissions(self):
        """测试配置文件权限"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_manager = MockConfigManager(temp_dir)
            
            # 创建包含敏感信息的配置
            system_config = SystemConfig()
            provider_config = ProviderConfig(
                name="test",
                api_key="secret-key",
                base_url="https://api.test.com"
            )
            
            config = HarborAIConfig(
                system=system_config,
                providers={"test": provider_config}
            )
            
            config_manager.config = config
            
            # 保存配置文件
            config_path = os.path.join(temp_dir, "secure_config.json")
            success = config_manager.save_config(config_path)
            assert success == True
            
            # 验证文件存在
            assert os.path.exists(config_path)
            
            # 在实际实现中，这里应该检查文件权限
            # 确保只有所有者可以读写配置文件
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.configuration
    def test_config_validation_security(self):
        """测试配置验证安全性"""
        config_manager = MockConfigManager()
        
        # 测试潜在的安全风险配置
        system_config = SystemConfig(
            debug=True,  # 生产环境不应该开启debug
            log_level="DEBUG"  # 可能暴露敏感信息
        )
        
        provider_config = ProviderConfig(
            name="test",
            api_key="weak-key",  # 弱密钥
            base_url="http://insecure.com"  # 不安全的HTTP连接
        )
        
        config = HarborAIConfig(
            system=system_config,
            providers={"test": provider_config}
        )
        
        # 验证配置（在实际实现中应该包含安全检查）
        errors = config_manager.validate_config(config)
        
        # 基本验证应该通过，但可以添加安全警告
        # 在实际实现中，可以添加安全相关的验证规则


if __name__ == "__main__":
    # 运行测试的示例
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "configuration"
    ])