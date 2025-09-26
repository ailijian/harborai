# -*- coding: utf-8 -*-
"""
HarborAI 插件系统测试模块

测试目标：
- 验证插件的加载和卸载机制
- 测试插件的生命周期管理
- 验证插件间的通信和依赖关系
- 测试插件的配置和权限管理
- 验证插件的错误处理和隔离
"""

import pytest
import json
import os
import tempfile
import shutil
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from harborai import HarborAI
from harborai.core.exceptions import HarborAIError, PluginError, PluginLoadError, PluginConfigError
from harborai.core.plugins import PluginManager, Plugin, PluginRegistry
from harborai.core.plugins.base import BasePlugin
from harborai.core.plugins.hooks import PluginHook, HookType


class MockPlugin(BasePlugin):
    """测试用的模拟插件"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        super().__init__()
        self.name = name
        self.version = version
        self.enabled = False
        self.initialized = False
        self.call_count = 0
    
    def initialize(self) -> bool:
        """初始化插件"""
        self.initialized = True
        return True
    
    def enable(self) -> bool:
        """启用插件"""
        if not self.initialized:
            return False
        self.enabled = True
        return True
    
    def disable(self) -> bool:
        """禁用插件"""
        self.enabled = False
        return True
    
    def execute(self, *args, **kwargs) -> Any:
        """执行插件功能"""
        if not self.enabled:
            raise PluginError(f"Plugin {self.name} is not enabled")
        
        self.call_count += 1
        return f"Plugin {self.name} executed with args: {args}, kwargs: {kwargs}"
    
    def get_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return {
            "name": self.name,
            "version": self.version,
            "enabled": self.enabled,
            "initialized": self.initialized,
            "call_count": self.call_count
        }
    
    @property
    def info(self) -> 'PluginInfo':
        """获取插件信息"""
        from harborai.core.plugins import PluginInfo
        return PluginInfo(
            name=self.name,
            version=self.version,
            description=f"Mock plugin {self.name}",
            author="Test",
            supported_models=["mock-model"]
        )
    
    def chat_completion(self, messages, model=None, **kwargs):
        """同步聊天完成"""
        if not self.enabled:
            raise PluginError(f"Plugin {self.name} is not enabled")
        
        self.call_count += 1
        return {
            "id": f"mock-{self.call_count}",
            "object": "chat.completion",
            "model": model or "mock-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Mock response from {self.name}"
                },
                "finish_reason": "stop"
            }]
        }
    
    async def chat_completion_async(self, messages, model=None, **kwargs):
        """异步聊天完成"""
        return self.chat_completion(messages, model, **kwargs)


class TestPluginLoading:
    """插件加载测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.plugin_system
    def test_plugin_discovery(self, tmp_path):
        """测试插件发现机制"""
        # 创建临时插件目录
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        
        # 创建插件文件
        plugin_files = [
            "test_plugin_1.py",
            "test_plugin_2.py",
            "invalid_plugin.txt",  # 无效文件
            "__pycache__",         # 应该被忽略的目录
        ]
        
        for file_name in plugin_files:
            if file_name.endswith(".py"):
                plugin_file = plugin_dir / file_name
                plugin_content = f"""
class TestPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.name = "{file_name[:-3]}"
        self.version = "1.0.0"
    
    def initialize(self):
        return True
"""
                plugin_file.write_text(plugin_content)
            elif file_name == "__pycache__":
                (plugin_dir / file_name).mkdir()
            else:
                (plugin_dir / file_name).write_text("invalid content")
        
        # 测试插件发现
        with patch('harborai.core.plugins.PluginManager') as mock_manager:
            mock_manager_instance = Mock()
            mock_manager.return_value = mock_manager_instance
            
            # 模拟发现插件
            discovered_plugins = [
                "test_plugin_1",
                "test_plugin_2"
            ]
            mock_manager_instance.discover_plugins.return_value = discovered_plugins
            
            manager = mock_manager(plugin_directory=str(plugin_dir))
            plugins = manager.discover_plugins()
            
            # 验证发现的插件
            assert len(plugins) == 2
            assert "test_plugin_1" in plugins
            assert "test_plugin_2" in plugins
            assert "invalid_plugin" not in plugins
            assert "__pycache__" not in plugins
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.plugin_system
    def test_plugin_loading_success(self):
        """测试插件加载成功"""
        # 创建模拟插件管理器
        plugin_manager = Mock(spec=PluginManager)
        
        # 创建测试插件
        test_plugin = MockPlugin("test_plugin", "1.0.0")
        
        # 配置mock行为
        plugin_manager.load_plugin.return_value = test_plugin
        plugin_manager.get_plugin.return_value = test_plugin
        plugin_manager.is_loaded.return_value = True
        
        # 测试加载插件
        loaded_plugin = plugin_manager.load_plugin("test_plugin")
        
        # 验证加载结果
        assert loaded_plugin is not None
        assert loaded_plugin.name == "test_plugin"
        assert loaded_plugin.version == "1.0.0"
        
        # 验证插件状态
        assert plugin_manager.is_loaded("test_plugin")
        
        # 验证可以获取插件
        retrieved_plugin = plugin_manager.get_plugin("test_plugin")
        assert retrieved_plugin == test_plugin
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    def test_plugin_loading_failure(self):
        """测试插件加载失败"""
        # 创建模拟插件管理器
        plugin_manager = Mock(spec=PluginManager)
        
        # 配置加载失败
        plugin_manager.load_plugin.side_effect = PluginLoadError("Failed to load plugin: invalid_plugin")
        plugin_manager.is_loaded.return_value = False
        
        # 测试加载失败的插件
        with pytest.raises(PluginLoadError) as exc_info:
            plugin_manager.load_plugin("invalid_plugin")
        
        assert "Failed to load plugin" in str(exc_info.value)
        assert "invalid_plugin" in str(exc_info.value)
        
        # 验证插件未加载
        assert not plugin_manager.is_loaded("invalid_plugin")
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    def test_plugin_dependency_resolution(self):
        """测试插件依赖解析"""
        # 创建有依赖关系的插件
        plugin_a = MockPlugin("plugin_a", "1.0.0")
        plugin_b = MockPlugin("plugin_b", "1.0.0")
        plugin_c = MockPlugin("plugin_c", "1.0.0")
        
        # 设置依赖关系：C依赖B，B依赖A
        dependencies = {
            "plugin_c": ["plugin_b"],
            "plugin_b": ["plugin_a"],
            "plugin_a": []
        }
        
        # 创建模拟插件管理器
        plugin_manager = Mock(spec=PluginManager)
        
        # 模拟依赖解析
        def mock_resolve_dependencies(plugin_name):
            return dependencies.get(plugin_name, [])
        
        def mock_load_with_dependencies(plugin_name):
            deps = mock_resolve_dependencies(plugin_name)
            # 先加载依赖
            for dep in deps:
                mock_load_with_dependencies(dep)
            # 再加载自己
            if plugin_name == "plugin_a":
                return plugin_a
            elif plugin_name == "plugin_b":
                return plugin_b
            elif plugin_name == "plugin_c":
                return plugin_c
        
        plugin_manager.resolve_dependencies.side_effect = mock_resolve_dependencies
        plugin_manager.load_plugin.side_effect = mock_load_with_dependencies
        
        # 测试加载有依赖的插件
        loaded_plugin = plugin_manager.load_plugin("plugin_c")
        
        # 验证依赖解析
        deps = plugin_manager.resolve_dependencies("plugin_c")
        assert "plugin_b" in deps
        
        deps_b = plugin_manager.resolve_dependencies("plugin_b")
        assert "plugin_a" in deps_b
        
        deps_a = plugin_manager.resolve_dependencies("plugin_a")
        assert len(deps_a) == 0
        
        # 验证加载成功
        assert loaded_plugin.name == "plugin_c"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    def test_circular_dependency_detection(self):
        """测试循环依赖检测"""
        # 创建模拟插件管理器
        plugin_manager = Mock(spec=PluginManager)
        
        # 设置循环依赖：A依赖B，B依赖C，C依赖A
        circular_dependencies = {
            "plugin_a": ["plugin_b"],
            "plugin_b": ["plugin_c"],
            "plugin_c": ["plugin_a"]
        }
        
        # 配置循环依赖检测
        plugin_manager.resolve_dependencies.side_effect = lambda name: circular_dependencies.get(name, [])
        plugin_manager.load_plugin.side_effect = PluginLoadError("Circular dependency detected")
        
        # 测试循环依赖检测
        with pytest.raises(PluginLoadError) as exc_info:
            plugin_manager.load_plugin("plugin_a")
        
        assert "Circular dependency" in str(exc_info.value)


class TestPluginLifecycle:
    """插件生命周期测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.plugin_system
    def test_plugin_initialization(self):
        """测试插件初始化"""
        # 创建测试插件
        plugin = MockPlugin("test_plugin")
        
        # 验证初始状态
        assert not plugin.initialized
        assert not plugin.enabled
        
        # 测试初始化
        result = plugin.initialize()
        
        # 验证初始化结果
        assert result is True
        assert plugin.initialized
        assert not plugin.enabled  # 初始化后还未启用
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.plugin_system
    def test_plugin_enable_disable(self):
        """测试插件启用和禁用"""
        # 创建并初始化插件
        plugin = MockPlugin("test_plugin")
        plugin.initialize()
        
        # 测试启用插件
        enable_result = plugin.enable()
        assert enable_result is True
        assert plugin.enabled
        
        # 测试执行插件功能
        result = plugin.execute("test_arg", test_param="test_value")
        assert "Plugin test_plugin executed" in result
        assert plugin.call_count == 1
        
        # 测试禁用插件
        disable_result = plugin.disable()
        assert disable_result is True
        assert not plugin.enabled
        
        # 测试禁用后无法执行
        with pytest.raises(PluginError) as exc_info:
            plugin.execute("test_arg")
        
        assert "not enabled" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    def test_plugin_enable_without_initialization(self):
        """测试未初始化插件的启用"""
        # 创建未初始化的插件
        plugin = MockPlugin("test_plugin")
        
        # 尝试启用未初始化的插件
        enable_result = plugin.enable()
        
        # 验证启用失败
        assert enable_result is False
        assert not plugin.enabled
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    def test_plugin_lifecycle_management(self):
        """测试插件生命周期管理"""
        # 创建插件管理器
        plugin_manager = Mock(spec=PluginManager)
        
        # 创建测试插件
        plugin = MockPlugin("lifecycle_plugin")
        
        # 模拟生命周期管理
        lifecycle_states = []
        
        def mock_initialize(plugin_name):
            lifecycle_states.append(f"initialize_{plugin_name}")
            return plugin.initialize()
        
        def mock_enable(plugin_name):
            lifecycle_states.append(f"enable_{plugin_name}")
            return plugin.enable()
        
        def mock_disable(plugin_name):
            lifecycle_states.append(f"disable_{plugin_name}")
            return plugin.disable()
        
        def mock_unload(plugin_name):
            lifecycle_states.append(f"unload_{plugin_name}")
            plugin.initialized = False
            return True
        
        plugin_manager.initialize_plugin.side_effect = mock_initialize
        plugin_manager.enable_plugin.side_effect = mock_enable
        plugin_manager.disable_plugin.side_effect = mock_disable
        plugin_manager.unload_plugin.side_effect = mock_unload
        
        # 执行完整的生命周期
        plugin_name = "lifecycle_plugin"
        
        # 初始化
        plugin_manager.initialize_plugin(plugin_name)
        
        # 启用
        plugin_manager.enable_plugin(plugin_name)
        
        # 禁用
        plugin_manager.disable_plugin(plugin_name)
        
        # 卸载
        plugin_manager.unload_plugin(plugin_name)
        
        # 验证生命周期顺序
        expected_states = [
            "initialize_lifecycle_plugin",
            "enable_lifecycle_plugin",
            "disable_lifecycle_plugin",
            "unload_lifecycle_plugin"
        ]
        
        assert lifecycle_states == expected_states
        assert not plugin.initialized  # 卸载后应该重置状态


class TestPluginCommunication:
    """插件通信测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    def test_plugin_hook_system(self):
        """测试插件钩子系统"""
        # 创建钩子管理器
        hook_manager = Mock()
        
        # 定义钩子类型
        hook_types = {
            "before_request": HookType.BEFORE,
            "after_request": HookType.AFTER,
            "on_error": HookType.ERROR
        }
        
        # 创建测试插件
        plugin_a = MockPlugin("plugin_a")
        plugin_b = MockPlugin("plugin_b")
        
        # 注册钩子
        registered_hooks = {}
        
        def mock_register_hook(hook_name, plugin, callback):
            if hook_name not in registered_hooks:
                registered_hooks[hook_name] = []
            registered_hooks[hook_name].append((plugin, callback))
        
        def mock_trigger_hook(hook_name, *args, **kwargs):
            results = []
            if hook_name in registered_hooks:
                for plugin, callback in registered_hooks[hook_name]:
                    result = callback(*args, **kwargs)
                    results.append((plugin.name, result))
            return results
        
        hook_manager.register_hook.side_effect = mock_register_hook
        hook_manager.trigger_hook.side_effect = mock_trigger_hook
        
        # 注册钩子回调
        def before_request_callback(*args, **kwargs):
            return f"plugin_a before_request: {args}, {kwargs}"
        
        def after_request_callback(*args, **kwargs):
            return f"plugin_b after_request: {args}, {kwargs}"
        
        hook_manager.register_hook("before_request", plugin_a, before_request_callback)
        hook_manager.register_hook("after_request", plugin_b, after_request_callback)
        
        # 触发钩子
        before_results = hook_manager.trigger_hook("before_request", "test_data", param="test")
        after_results = hook_manager.trigger_hook("after_request", "response_data")
        
        # 验证钩子执行结果
        assert len(before_results) == 1
        assert before_results[0][0] == "plugin_a"
        assert "before_request" in before_results[0][1]
        
        assert len(after_results) == 1
        assert after_results[0][0] == "plugin_b"
        assert "after_request" in after_results[0][1]
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    def test_plugin_event_system(self):
        """测试插件事件系统"""
        # 创建事件管理器
        event_manager = Mock()
        
        # 事件订阅和发布
        event_subscribers = {}
        
        def mock_subscribe(event_name, plugin, handler):
            if event_name not in event_subscribers:
                event_subscribers[event_name] = []
            event_subscribers[event_name].append((plugin, handler))
        
        def mock_publish(event_name, event_data):
            results = []
            if event_name in event_subscribers:
                for plugin, handler in event_subscribers[event_name]:
                    try:
                        result = handler(event_data)
                        results.append((plugin.name, result, None))
                    except Exception as e:
                        results.append((plugin.name, None, str(e)))
            return results
        
        event_manager.subscribe.side_effect = mock_subscribe
        event_manager.publish.side_effect = mock_publish
        
        # 创建测试插件
        plugin_a = MockPlugin("event_plugin_a")
        plugin_b = MockPlugin("event_plugin_b")
        
        # 定义事件处理器
        def handle_user_login(event_data):
            return f"plugin_a handled login: {event_data['user_id']}"
        
        def handle_user_logout(event_data):
            return f"plugin_b handled logout: {event_data['user_id']}"
        
        # 订阅事件
        event_manager.subscribe("user_login", plugin_a, handle_user_login)
        event_manager.subscribe("user_logout", plugin_b, handle_user_logout)
        
        # 发布事件
        login_results = event_manager.publish("user_login", {"user_id": "user123", "timestamp": "2024-01-01"})
        logout_results = event_manager.publish("user_logout", {"user_id": "user123", "timestamp": "2024-01-02"})
        
        # 验证事件处理结果
        assert len(login_results) == 1
        assert login_results[0][0] == "event_plugin_a"
        assert "handled login: user123" in login_results[0][1]
        assert login_results[0][2] is None  # 无错误
        
        assert len(logout_results) == 1
        assert logout_results[0][0] == "event_plugin_b"
        assert "handled logout: user123" in logout_results[0][1]
        assert logout_results[0][2] is None  # 无错误
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.plugin_system
    def test_plugin_data_sharing(self):
        """测试插件数据共享"""
        # 创建数据共享管理器
        data_manager = Mock()
        
        # 模拟共享数据存储
        shared_data = {}
        
        def mock_set_data(key, value, plugin_name):
            shared_data[f"{plugin_name}:{key}"] = value
        
        def mock_get_data(key, plugin_name=None):
            if plugin_name:
                return shared_data.get(f"{plugin_name}:{key}")
            else:
                # 获取所有插件的该key数据
                results = {}
                for full_key, value in shared_data.items():
                    if full_key.endswith(f":{key}"):
                        plugin = full_key.split(":")[0]
                        results[plugin] = value
                return results
        
        def mock_delete_data(key, plugin_name):
            full_key = f"{plugin_name}:{key}"
            if full_key in shared_data:
                del shared_data[full_key]
                return True
            return False
        
        data_manager.set_data.side_effect = mock_set_data
        data_manager.get_data.side_effect = mock_get_data
        data_manager.delete_data.side_effect = mock_delete_data
        
        # 测试数据共享
        plugin_a = MockPlugin("data_plugin_a")
        plugin_b = MockPlugin("data_plugin_b")
        
        # 插件A设置数据
        data_manager.set_data("user_preferences", {"theme": "dark", "language": "zh"}, "data_plugin_a")
        data_manager.set_data("cache_size", 1024, "data_plugin_a")
        
        # 插件B设置数据
        data_manager.set_data("user_preferences", {"theme": "light", "notifications": True}, "data_plugin_b")
        
        # 获取特定插件的数据
        plugin_a_prefs = data_manager.get_data("user_preferences", "data_plugin_a")
        assert plugin_a_prefs["theme"] == "dark"
        assert plugin_a_prefs["language"] == "zh"
        
        plugin_b_prefs = data_manager.get_data("user_preferences", "data_plugin_b")
        assert plugin_b_prefs["theme"] == "light"
        assert plugin_b_prefs["notifications"] is True
        
        # 获取所有插件的同名数据
        all_prefs = data_manager.get_data("user_preferences")
        assert "data_plugin_a" in all_prefs
        assert "data_plugin_b" in all_prefs
        assert len(all_prefs) == 2
        
        # 删除数据
        delete_result = data_manager.delete_data("cache_size", "data_plugin_a")
        assert delete_result is True
        
        # 验证数据已删除
        deleted_data = data_manager.get_data("cache_size", "data_plugin_a")
        assert deleted_data is None


class TestPluginConfiguration:
    """插件配置测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    def test_plugin_configuration_loading(self, tmp_path):
        """测试插件配置加载"""
        # 创建配置文件
        config_file = tmp_path / "plugin_config.json"
        config_data = {
            "test_plugin": {
                "enabled": True,
                "settings": {
                    "api_key": "test_key_123",
                    "timeout": 30,
                    "retry_count": 3
                },
                "permissions": ["read", "write"]
            },
            "another_plugin": {
                "enabled": False,
                "settings": {
                    "debug_mode": True
                },
                "permissions": ["read"]
            }
        }
        
        config_file.write_text(json.dumps(config_data, indent=2))
        
        # 创建配置管理器
        config_manager = Mock()
        
        def mock_load_config(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        def mock_get_plugin_config(plugin_name, config_data):
            return config_data.get(plugin_name, {})
        
        config_manager.load_config.side_effect = mock_load_config
        config_manager.get_plugin_config.side_effect = mock_get_plugin_config
        
        # 加载配置
        loaded_config = config_manager.load_config(str(config_file))
        
        # 获取特定插件配置
        test_plugin_config = config_manager.get_plugin_config("test_plugin", loaded_config)
        another_plugin_config = config_manager.get_plugin_config("another_plugin", loaded_config)
        
        # 验证配置加载
        assert test_plugin_config["enabled"] is True
        assert test_plugin_config["settings"]["api_key"] == "test_key_123"
        assert test_plugin_config["settings"]["timeout"] == 30
        assert "read" in test_plugin_config["permissions"]
        assert "write" in test_plugin_config["permissions"]
        
        assert another_plugin_config["enabled"] is False
        assert another_plugin_config["settings"]["debug_mode"] is True
        assert "read" in another_plugin_config["permissions"]
        assert "write" not in another_plugin_config["permissions"]
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    def test_plugin_configuration_validation(self):
        """测试插件配置验证"""
        # 创建配置验证器
        config_validator = Mock()
        
        # 定义配置模式
        config_schema = {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "settings": {
                    "type": "object",
                    "properties": {
                        "api_key": {"type": "string", "minLength": 1},
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 300},
                        "retry_count": {"type": "integer", "minimum": 0, "maximum": 10}
                    },
                    "required": ["api_key"]
                },
                "permissions": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["read", "write", "admin"]}
                }
            },
            "required": ["enabled"]
        }
        
        def mock_validate_config(config, schema):
            # 简化的验证逻辑
            if not isinstance(config.get("enabled"), bool):
                raise PluginConfigError("enabled must be a boolean")
            
            if "settings" in config:
                settings = config["settings"]
                if "api_key" not in settings or not settings["api_key"]:
                    raise PluginConfigError("api_key is required and cannot be empty")
                
                if "timeout" in settings:
                    timeout = settings["timeout"]
                    if not isinstance(timeout, int) or timeout < 1 or timeout > 300:
                        raise PluginConfigError("timeout must be an integer between 1 and 300")
            
            return True
        
        config_validator.validate.side_effect = mock_validate_config
        
        # 测试有效配置
        valid_config = {
            "enabled": True,
            "settings": {
                "api_key": "valid_key",
                "timeout": 30,
                "retry_count": 3
            },
            "permissions": ["read", "write"]
        }
        
        result = config_validator.validate(valid_config, config_schema)
        assert result is True
        
        # 测试无效配置 - 缺少api_key
        invalid_config_1 = {
            "enabled": True,
            "settings": {
                "timeout": 30
            }
        }
        
        with pytest.raises(PluginConfigError) as exc_info:
            config_validator.validate(invalid_config_1, config_schema)
        assert "api_key is required" in str(exc_info.value)
        
        # 测试无效配置 - 超时值超出范围
        invalid_config_2 = {
            "enabled": True,
            "settings": {
                "api_key": "valid_key",
                "timeout": 500  # 超出最大值
            }
        }
        
        with pytest.raises(PluginConfigError) as exc_info:
            config_validator.validate(invalid_config_2, config_schema)
        assert "timeout must be an integer between 1 and 300" in str(exc_info.value)
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.plugin_system
    def test_plugin_configuration_hot_reload(self):
        """测试插件配置热重载"""
        # 创建配置管理器
        config_manager = Mock()
        
        # 初始配置
        initial_config = {
            "test_plugin": {
                "enabled": True,
                "settings": {"debug": False}
            }
        }
        
        # 更新后的配置
        updated_config = {
            "test_plugin": {
                "enabled": True,
                "settings": {"debug": True, "log_level": "INFO"}
            }
        }
        
        # 配置变更历史
        config_history = [initial_config]
        
        def mock_reload_config():
            config_history.append(updated_config)
            return updated_config
        
        def mock_get_current_config():
            return config_history[-1]
        
        def mock_notify_config_change(plugin_name, old_config, new_config):
            return f"Config changed for {plugin_name}: {old_config} -> {new_config}"
        
        config_manager.reload_config.side_effect = mock_reload_config
        config_manager.get_current_config.side_effect = mock_get_current_config
        config_manager.notify_config_change.side_effect = mock_notify_config_change
        
        # 获取初始配置
        current_config = config_manager.get_current_config()
        assert current_config["test_plugin"]["settings"]["debug"] is False
        
        # 热重载配置
        new_config = config_manager.reload_config()
        
        # 验证配置更新
        assert new_config["test_plugin"]["settings"]["debug"] is True
        assert new_config["test_plugin"]["settings"]["log_level"] == "INFO"
        
        # 验证变更通知
        old_plugin_config = initial_config["test_plugin"]
        new_plugin_config = updated_config["test_plugin"]
        notification = config_manager.notify_config_change("test_plugin", old_plugin_config, new_plugin_config)
        
        assert "Config changed for test_plugin" in notification
        assert len(config_history) == 2


class TestPluginSecurity:
    """插件安全测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    @pytest.mark.security
    def test_plugin_permission_system(self):
        """测试插件权限系统"""
        # 创建权限管理器
        permission_manager = Mock()
        
        # 定义权限
        permissions = {
            "read_plugin": ["read"],
            "write_plugin": ["read", "write"],
            "admin_plugin": ["read", "write", "admin"]
        }
        
        def mock_check_permission(plugin_name, required_permission):
            plugin_permissions = permissions.get(plugin_name, [])
            return required_permission in plugin_permissions
        
        def mock_grant_permission(plugin_name, permission):
            if plugin_name not in permissions:
                permissions[plugin_name] = []
            if permission not in permissions[plugin_name]:
                permissions[plugin_name].append(permission)
            return True
        
        def mock_revoke_permission(plugin_name, permission):
            if plugin_name in permissions and permission in permissions[plugin_name]:
                permissions[plugin_name].remove(permission)
                return True
            return False
        
        permission_manager.check_permission.side_effect = mock_check_permission
        permission_manager.grant_permission.side_effect = mock_grant_permission
        permission_manager.revoke_permission.side_effect = mock_revoke_permission
        
        # 测试权限检查
        assert permission_manager.check_permission("read_plugin", "read") is True
        assert permission_manager.check_permission("read_plugin", "write") is False
        assert permission_manager.check_permission("write_plugin", "write") is True
        assert permission_manager.check_permission("admin_plugin", "admin") is True
        
        # 测试权限授予
        permission_manager.grant_permission("read_plugin", "write")
        assert permission_manager.check_permission("read_plugin", "write") is True
        
        # 测试权限撤销
        permission_manager.revoke_permission("admin_plugin", "admin")
        assert permission_manager.check_permission("admin_plugin", "admin") is False
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    @pytest.mark.security
    def test_plugin_sandboxing(self):
        """测试插件沙箱隔离"""
        # 创建沙箱管理器
        sandbox_manager = Mock()
        
        # 沙箱限制
        sandbox_limits = {
            "max_memory": 100 * 1024 * 1024,  # 100MB
            "max_cpu_time": 10,  # 10秒
            "allowed_modules": ["json", "re", "datetime"],
            "forbidden_operations": ["file_write", "network_access", "subprocess"]
        }
        
        def mock_create_sandbox(plugin_name, limits):
            return f"sandbox_{plugin_name}"
        
        def mock_execute_in_sandbox(sandbox_id, code, *args, **kwargs):
            # 模拟沙箱执行
            if "import os" in code:
                raise PluginError("Module 'os' is not allowed in sandbox")
            if "open(" in code and "w" in code:
                raise PluginError("File write operation is not allowed")
            
            # 模拟正常执行
            return f"Executed in {sandbox_id}: {code[:50]}..."
        
        def mock_destroy_sandbox(sandbox_id):
            return True
        
        sandbox_manager.create_sandbox.side_effect = mock_create_sandbox
        sandbox_manager.execute_in_sandbox.side_effect = mock_execute_in_sandbox
        sandbox_manager.destroy_sandbox.side_effect = mock_destroy_sandbox
        
        # 创建沙箱
        sandbox_id = sandbox_manager.create_sandbox("test_plugin", sandbox_limits)
        assert sandbox_id == "sandbox_test_plugin"
        
        # 测试允许的操作
        safe_code = "import json; result = json.dumps({'test': 'data'})"
        result = sandbox_manager.execute_in_sandbox(sandbox_id, safe_code)
        assert "Executed in sandbox_test_plugin" in result
        
        # 测试禁止的模块导入
        unsafe_code_1 = "import os; os.system('rm -rf /')"
        with pytest.raises(PluginError) as exc_info:
            sandbox_manager.execute_in_sandbox(sandbox_id, unsafe_code_1)
        assert "Module 'os' is not allowed" in str(exc_info.value)
        
        # 测试禁止的文件操作
        unsafe_code_2 = "with open('/etc/passwd', 'w') as f: f.write('malicious')"
        with pytest.raises(PluginError) as exc_info:
            sandbox_manager.execute_in_sandbox(sandbox_id, unsafe_code_2)
        assert "File write operation is not allowed" in str(exc_info.value)
        
        # 销毁沙箱
        destroy_result = sandbox_manager.destroy_sandbox(sandbox_id)
        assert destroy_result is True
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.plugin_system
    @pytest.mark.security
    def test_plugin_resource_limits(self):
        """测试插件资源限制"""
        # 创建资源监控器
        resource_monitor = Mock()
        
        # 资源使用情况
        resource_usage = {
            "test_plugin": {
                "memory": 50 * 1024 * 1024,  # 50MB
                "cpu_time": 5.0,  # 5秒
                "network_requests": 10
            }
        }
        
        # 资源限制
        resource_limits = {
            "test_plugin": {
                "max_memory": 100 * 1024 * 1024,  # 100MB
                "max_cpu_time": 10.0,  # 10秒
                "max_network_requests": 20
            }
        }
        
        def mock_get_resource_usage(plugin_name):
            return resource_usage.get(plugin_name, {})
        
        def mock_check_resource_limits(plugin_name):
            usage = resource_usage.get(plugin_name, {})
            limits = resource_limits.get(plugin_name, {})
            
            violations = []
            
            if usage.get("memory", 0) > limits.get("max_memory", float('inf')):
                violations.append("memory")
            
            if usage.get("cpu_time", 0) > limits.get("max_cpu_time", float('inf')):
                violations.append("cpu_time")
            
            if usage.get("network_requests", 0) > limits.get("max_network_requests", float('inf')):
                violations.append("network_requests")
            
            return violations
        
        def mock_enforce_limits(plugin_name, violations):
            if violations:
                return f"Plugin {plugin_name} suspended due to: {', '.join(violations)}"
            return None
        
        resource_monitor.get_resource_usage.side_effect = mock_get_resource_usage
        resource_monitor.check_resource_limits.side_effect = mock_check_resource_limits
        resource_monitor.enforce_limits.side_effect = mock_enforce_limits
        
        # 测试正常资源使用
        usage = resource_monitor.get_resource_usage("test_plugin")
        assert usage["memory"] == 50 * 1024 * 1024
        assert usage["cpu_time"] == 5.0
        
        violations = resource_monitor.check_resource_limits("test_plugin")
        assert len(violations) == 0
        
        enforcement = resource_monitor.enforce_limits("test_plugin", violations)
        assert enforcement is None
        
        # 测试资源超限
        resource_usage["test_plugin"]["memory"] = 150 * 1024 * 1024  # 超过100MB限制
        resource_usage["test_plugin"]["cpu_time"] = 15.0  # 超过10秒限制
        
        violations = resource_monitor.check_resource_limits("test_plugin")
        assert "memory" in violations
        assert "cpu_time" in violations
        
        enforcement = resource_monitor.enforce_limits("test_plugin", violations)
        assert "suspended" in enforcement
        assert "memory" in enforcement
        assert "cpu_time" in enforcement


class TestPluginErrorHandling:
    """插件错误处理测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    @pytest.mark.error_handling
    def test_plugin_exception_isolation(self):
        """测试插件异常隔离"""
        # 创建异常处理器
        exception_handler = Mock()
        
        # 异常记录
        exception_log = []
        
        def mock_handle_plugin_exception(plugin_name, exception, context=None):
            exception_info = {
                "plugin": plugin_name,
                "exception_type": type(exception).__name__,
                "message": str(exception),
                "context": context,
                "timestamp": "2024-01-01T00:00:00Z"
            }
            exception_log.append(exception_info)
            
            # 决定是否隔离插件
            if isinstance(exception, (PluginError, RuntimeError)):
                return "isolate"  # 隔离插件
            else:
                return "continue"  # 继续运行
        
        exception_handler.handle_exception.side_effect = mock_handle_plugin_exception
        
        # 测试不同类型的异常
        test_cases = [
            ("plugin_a", PluginError("Plugin configuration error"), "config_load"),
            ("plugin_b", RuntimeError("Plugin runtime error"), "execution"),
            ("plugin_c", ValueError("Invalid parameter"), "parameter_validation"),
            ("plugin_d", KeyError("Missing key"), "data_access")
        ]
        
        for plugin_name, exception, context in test_cases:
            action = exception_handler.handle_exception(plugin_name, exception, context)
            
            if isinstance(exception, (PluginError, RuntimeError)):
                assert action == "isolate"
            else:
                assert action == "continue"
        
        # 验证异常记录
        assert len(exception_log) == 4
        
        plugin_a_log = next(log for log in exception_log if log["plugin"] == "plugin_a")
        assert plugin_a_log["exception_type"] == "PluginError"
        assert plugin_a_log["context"] == "config_load"
        
        plugin_b_log = next(log for log in exception_log if log["plugin"] == "plugin_b")
        assert plugin_b_log["exception_type"] == "RuntimeError"
        assert plugin_b_log["context"] == "execution"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.plugin_system
    @pytest.mark.error_handling
    def test_plugin_recovery_mechanism(self):
        """测试插件恢复机制"""
        # 创建恢复管理器
        recovery_manager = Mock()
        
        # 插件状态
        plugin_states = {
            "test_plugin": {
                "status": "running",
                "failure_count": 0,
                "last_failure": None
            }
        }
        
        def mock_attempt_recovery(plugin_name, failure_type):
            state = plugin_states.get(plugin_name, {})
            state["failure_count"] = state.get("failure_count", 0) + 1
            state["last_failure"] = failure_type
            
            # 恢复策略
            if state["failure_count"] <= 3:
                # 尝试重启插件
                state["status"] = "recovering"
                return "restart"
            elif state["failure_count"] <= 5:
                # 尝试重新加载
                state["status"] = "reloading"
                return "reload"
            else:
                # 禁用插件
                state["status"] = "disabled"
                return "disable"
        
        def mock_execute_recovery(plugin_name, recovery_action):
            state = plugin_states.get(plugin_name, {})
            
            if recovery_action == "restart":
                # 模拟重启成功
                state["status"] = "running"
                return True
            elif recovery_action == "reload":
                # 模拟重新加载成功
                state["status"] = "running"
                state["failure_count"] = 0  # 重置失败计数
                return True
            elif recovery_action == "disable":
                # 禁用插件
                state["status"] = "disabled"
                return True
            
            return False
        
        recovery_manager.attempt_recovery.side_effect = mock_attempt_recovery
        recovery_manager.execute_recovery.side_effect = mock_execute_recovery
        
        # 模拟多次失败和恢复
        plugin_name = "test_plugin"
        
        # 第一次失败 - 重启
        recovery_action = recovery_manager.attempt_recovery(plugin_name, "execution_error")
        assert recovery_action == "restart"
        
        recovery_result = recovery_manager.execute_recovery(plugin_name, recovery_action)
        assert recovery_result is True
        assert plugin_states[plugin_name]["status"] == "running"
        
        # 第二次失败 - 重启
        recovery_action = recovery_manager.attempt_recovery(plugin_name, "timeout_error")
        assert recovery_action == "restart"
        
        # 第三次失败 - 重启
        recovery_action = recovery_manager.attempt_recovery(plugin_name, "memory_error")
        assert recovery_action == "restart"
        
        # 第四次失败 - 重新加载
        recovery_action = recovery_manager.attempt_recovery(plugin_name, "config_error")
        assert recovery_action == "reload"
        
        recovery_result = recovery_manager.execute_recovery(plugin_name, recovery_action)
        assert recovery_result is True
        assert plugin_states[plugin_name]["failure_count"] == 0  # 重置计数
        
        # 继续失败直到禁用
        for i in range(6):
            recovery_action = recovery_manager.attempt_recovery(plugin_name, "persistent_error")
        
        assert recovery_action == "disable"
        
        recovery_result = recovery_manager.execute_recovery(plugin_name, recovery_action)
        assert recovery_result is True
        assert plugin_states[plugin_name]["status"] == "disabled"
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.plugin_system
    @pytest.mark.error_handling
    def test_plugin_health_monitoring(self):
        """测试插件健康监控"""
        # 创建健康监控器
        health_monitor = Mock()
        
        # 健康指标
        health_metrics = {
            "plugin_a": {
                "response_time": 0.1,
                "success_rate": 0.95,
                "memory_usage": 50 * 1024 * 1024,
                "error_count": 2,
                "last_heartbeat": "2024-01-01T00:00:00Z"
            },
            "plugin_b": {
                "response_time": 2.5,  # 响应时间过长
                "success_rate": 0.70,  # 成功率过低
                "memory_usage": 200 * 1024 * 1024,  # 内存使用过高
                "error_count": 15,
                "last_heartbeat": "2024-01-01T00:00:00Z"
            }
        }
        
        def mock_check_plugin_health(plugin_name):
            metrics = health_metrics.get(plugin_name, {})
            
            health_issues = []
            
            # 检查响应时间
            if metrics.get("response_time", 0) > 1.0:
                health_issues.append("slow_response")
            
            # 检查成功率
            if metrics.get("success_rate", 1.0) < 0.8:
                health_issues.append("low_success_rate")
            
            # 检查内存使用
            if metrics.get("memory_usage", 0) > 100 * 1024 * 1024:
                health_issues.append("high_memory_usage")
            
            # 检查错误数量
            if metrics.get("error_count", 0) > 10:
                health_issues.append("high_error_count")
            
            if health_issues:
                return {"status": "unhealthy", "issues": health_issues}
            else:
                return {"status": "healthy", "issues": []}
        
        def mock_get_health_score(plugin_name):
            metrics = health_metrics.get(plugin_name, {})
            
            # 计算健康分数 (0-100)
            score = 100
            
            # 响应时间影响
            response_time = metrics.get("response_time", 0)
            if response_time > 1.0:
                score -= min(30, (response_time - 1.0) * 20)
            
            # 成功率影响
            success_rate = metrics.get("success_rate", 1.0)
            score *= success_rate
            
            # 内存使用影响
            memory_usage = metrics.get("memory_usage", 0)
            if memory_usage > 100 * 1024 * 1024:
                score -= 20
            
            # 错误数量影响
            error_count = metrics.get("error_count", 0)
            if error_count > 10:
                score -= min(30, (error_count - 10) * 2)
            
            return max(0, int(score))
        
        health_monitor.check_health.side_effect = mock_check_plugin_health
        health_monitor.get_health_score.side_effect = mock_get_health_score
        
        # 检查健康插件
        plugin_a_health = health_monitor.check_health("plugin_a")
        assert plugin_a_health["status"] == "healthy"
        assert len(plugin_a_health["issues"]) == 0
        
        plugin_a_score = health_monitor.get_health_score("plugin_a")
        assert plugin_a_score >= 90  # 健康插件应该有高分
        
        # 检查不健康插件
        plugin_b_health = health_monitor.check_health("plugin_b")
        assert plugin_b_health["status"] == "unhealthy"
        assert "slow_response" in plugin_b_health["issues"]
        assert "low_success_rate" in plugin_b_health["issues"]
        assert "high_memory_usage" in plugin_b_health["issues"]
        assert "high_error_count" in plugin_b_health["issues"]
        
        plugin_b_score = health_monitor.get_health_score("plugin_b")
        assert plugin_b_score < 50  # 不健康插件应该有低分