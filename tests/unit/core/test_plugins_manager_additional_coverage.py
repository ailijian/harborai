#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 harborai.core.plugins.manager 模块的额外覆盖率

目标：补充缺失的测试覆盖率，达到90%以上
"""

import pytest
import importlib.util
import inspect
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Union, Generator, AsyncGenerator

from harborai.core.plugins.manager import PluginRegistry, PluginManager
from harborai.core.plugins.base import Plugin, PluginInfo
from harborai.core.exceptions import (
    PluginError, 
    PluginLoadError, 
    PluginNotFoundError, 
    PluginConfigError
)


class MockPlugin(Plugin):
    """模拟插件类用于测试"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._enabled = True
    
    def initialize(self):
        return True
    
    def cleanup(self):
        pass
    
    def get_supported_models(self):
        return ["mock-model-1", "mock-model-2"]
    
    @property
    def info(self):
        return PluginInfo(
            name="MockPlugin",
            version="1.0.0",
            description="Mock plugin for testing",
            author="Test",
            supported_models=self.get_supported_models()
        )
    
    def chat_completion(self, messages, model, **kwargs):
        return {"choices": [{"message": {"content": "Mock response"}}]}
    
    async def chat_completion_async(self, messages, model, **kwargs):
        return {"choices": [{"message": {"content": "Mock async response"}}]}
    
    def enable(self):
        self._enabled = True
        return True
    
    def disable(self):
        self._enabled = False
        return True


class TestPluginManagerAdditionalCoverage:
    """测试PluginManager的额外覆盖率"""
    
    def test_discover_plugins_spec_none(self, caplog):
        """测试spec为None的情况"""
        manager = PluginManager()
        
        with patch('harborai.core.plugins.manager.Path') as mock_path:
            # 创建模拟的文件路径对象
            mock_file = Mock()
            mock_file.is_file.return_value = True
            mock_file.name = "test_plugin.py"
            mock_file.stem = "test_plugin"
            
            # 模拟插件目录存在且包含插件文件
            mock_plugin_path = Mock()
            mock_plugin_path.exists.return_value = True
            mock_plugin_path.glob.return_value = [mock_file]
            mock_path.return_value = mock_plugin_path
            
            # 模拟spec为None
            with patch('harborai.core.plugins.manager.importlib.util.spec_from_file_location', return_value=None):
                discovered = manager.discover_plugins("/test/path")
                
                assert discovered == []
    
    def test_discover_plugins_spec_no_loader(self, caplog):
        """测试spec.loader为None的情况"""
        manager = PluginManager()
        
        with patch('harborai.core.plugins.manager.Path') as mock_path:
            # 创建模拟的文件路径对象
            mock_file = Mock()
            mock_file.is_file.return_value = True
            mock_file.name = "test_plugin.py"
            mock_file.stem = "test_plugin"
            
            # 模拟插件目录存在且包含插件文件
            mock_plugin_path = Mock()
            mock_plugin_path.exists.return_value = True
            mock_plugin_path.glob.return_value = [mock_file]
            mock_path.return_value = mock_plugin_path
            
            # 模拟spec.loader为None
            mock_spec = Mock()
            mock_spec.loader = None
            with patch('harborai.core.plugins.manager.importlib.util.spec_from_file_location', return_value=mock_spec):
                discovered = manager.discover_plugins("/test/path")
                
                assert discovered == []
    
    def test_discover_plugins_no_plugin_class_found(self, caplog):
        """测试模块中没有找到插件类的情况"""
        manager = PluginManager()
        
        with patch('harborai.core.plugins.manager.Path') as mock_path:
            # 创建模拟的文件路径对象
            mock_file = Mock()
            mock_file.is_file.return_value = True
            mock_file.name = "test_plugin.py"
            mock_file.stem = "test_plugin"
            
            # 模拟插件目录存在且包含插件文件
            mock_plugin_path = Mock()
            mock_plugin_path.exists.return_value = True
            mock_plugin_path.glob.return_value = [mock_file]
            mock_path.return_value = mock_plugin_path
            
            # 模拟模块加载成功但没有插件类
            mock_spec = Mock()
            mock_loader = Mock()
            mock_spec.loader = mock_loader
            mock_module = Mock()
            
            with patch('harborai.core.plugins.manager.importlib.util.spec_from_file_location', return_value=mock_spec):
                with patch('harborai.core.plugins.manager.importlib.util.module_from_spec', return_value=mock_module):
                    with patch('harborai.core.plugins.manager.inspect.getmembers', return_value=[]):
                        discovered = manager.discover_plugins("/test/path")
                        
                        assert discovered == []
    
    def test_load_plugin_already_loaded(self):
        """测试加载已经加载的插件"""
        manager = PluginManager()
        
        # 注册并加载插件
        manager.registry.register(MockPlugin, "test_plugin")
        first_instance = manager.load_plugin("test_plugin")
        
        # 再次加载应该返回同一个实例
        second_instance = manager.load_plugin("test_plugin")
        
        assert first_instance is second_instance
    
    def test_load_plugin_with_config(self):
        """测试使用配置加载插件"""
        manager = PluginManager()
        config = {"key": "value"}
        
        # 注册插件
        manager.registry.register(MockPlugin, "test_plugin")
        
        # 使用配置加载插件
        instance = manager.load_plugin("test_plugin", config)
        
        assert instance is not None
        assert isinstance(instance, MockPlugin)
    
    def test_unload_plugin_with_cleanup_exception(self, caplog):
        """测试卸载插件时cleanup抛出异常"""
        manager = PluginManager()
        
        # 创建一个cleanup会抛异常的插件
        class FailingCleanupPlugin(MockPlugin):
            def cleanup(self):
                raise Exception("Cleanup failed")
        
        # 注册并加载插件
        manager.registry.register(FailingCleanupPlugin, "failing_plugin")
        manager.load_plugin("failing_plugin")
        
        # 卸载插件应该返回False
        result = manager.unload_plugin("failing_plugin")
        
        assert result is False
        assert "Failed to unload plugin failing_plugin" in caplog.text
    
    def test_reload_plugin_success(self):
        """测试成功重新加载插件"""
        manager = PluginManager()
        
        # 注册并加载插件
        manager.registry.register(MockPlugin, "test_plugin")
        first_instance = manager.load_plugin("test_plugin")
        
        # 重新加载插件
        second_instance = manager.reload_plugin("test_plugin")
        
        assert second_instance is not None
        assert isinstance(second_instance, MockPlugin)
        # 应该是不同的实例
        assert first_instance is not second_instance
    
    def test_reload_plugin_failure(self, caplog):
        """测试重新加载插件失败"""
        manager = PluginManager()
        
        # 注册插件但不加载
        manager.registry.register(MockPlugin, "test_plugin")
        
        # 模拟load_plugin失败
        with patch.object(manager, 'load_plugin', side_effect=PluginLoadError("Load failed")):
            result = manager.reload_plugin("test_plugin")
            
            assert result is None
            assert "Failed to reload plugin test_plugin" in caplog.text
    
    def test_is_loaded(self):
        """测试检查插件是否已加载"""
        manager = PluginManager()
        
        # 注册插件
        manager.registry.register(MockPlugin, "test_plugin")
        
        # 未加载时应该返回False
        assert manager.is_loaded("test_plugin") is False
        
        # 加载后应该返回True
        manager.load_plugin("test_plugin")
        assert manager.is_loaded("test_plugin") is True
    
    def test_resolve_dependencies_with_dependencies(self):
        """测试解析有依赖的插件"""
        manager = PluginManager()
        
        # 创建有依赖的插件
        class DependentPlugin(MockPlugin):
            dependencies = ["dependency1", "dependency2"]
        
        # 注册插件
        manager.registry.register(DependentPlugin, "dependent_plugin")
        
        # 解析依赖
        deps = manager.resolve_dependencies("dependent_plugin")
        
        assert deps == ["dependency1", "dependency2"]
    
    def test_resolve_dependencies_no_dependencies(self):
        """测试解析无依赖的插件"""
        manager = PluginManager()
        
        # 注册插件
        manager.registry.register(MockPlugin, "test_plugin")
        
        # 解析依赖
        deps = manager.resolve_dependencies("test_plugin")
        
        assert deps == []
    
    def test_resolve_dependencies_plugin_not_found(self):
        """测试解析不存在插件的依赖"""
        manager = PluginManager()
        
        # 解析不存在插件的依赖
        deps = manager.resolve_dependencies("nonexistent")
        
        assert deps == []
    
    def test_initialize_plugin_success(self):
        """测试成功初始化插件"""
        manager = PluginManager()
        
        # 注册插件
        manager.registry.register(MockPlugin, "test_plugin")
        
        # 初始化插件
        result = manager.initialize_plugin("test_plugin")
        
        assert result is True
    
    def test_initialize_plugin_failure(self, caplog):
        """测试初始化插件失败"""
        manager = PluginManager()
        
        # 模拟load_plugin失败
        with patch.object(manager, 'load_plugin', side_effect=PluginLoadError("Load failed")):
            result = manager.initialize_plugin("test_plugin")
            
            assert result is False
            assert "Failed to initialize plugin test_plugin" in caplog.text
    
    def test_enable_plugin_not_loaded(self):
        """测试启用未加载的插件"""
        manager = PluginManager()
        
        # 注册插件
        manager.registry.register(MockPlugin, "test_plugin")
        
        # 启用未加载的插件应该自动加载
        result = manager.enable_plugin("test_plugin")
        
        assert result is True
        assert manager.is_loaded("test_plugin") is True
    
    def test_enable_plugin_already_loaded(self):
        """测试启用已加载的插件"""
        manager = PluginManager()
        
        # 注册并加载插件
        manager.registry.register(MockPlugin, "test_plugin")
        manager.load_plugin("test_plugin")
        
        # 启用插件
        result = manager.enable_plugin("test_plugin")
        
        assert result is True
    
    def test_enable_plugin_no_enable_method(self):
        """测试启用没有enable方法的插件"""
        manager = PluginManager()
        
        # 创建没有enable方法的插件
        class NoEnablePlugin(MockPlugin):
            pass  # 不包含enable方法
        
        # 注册并加载插件
        manager.registry.register(NoEnablePlugin, "test_plugin")
        manager.load_plugin("test_plugin")
        
        # 启用插件应该返回True（默认已启用）
        result = manager.enable_plugin("test_plugin")
        
        assert result is True
    
    def test_enable_plugin_load_failure(self, caplog):
        """测试启用插件时加载失败"""
        manager = PluginManager()
        
        # 模拟load_plugin失败
        with patch.object(manager, 'load_plugin', return_value=None):
            result = manager.enable_plugin("test_plugin")
            
            assert result is False
    
    def test_enable_plugin_exception(self, caplog):
        """测试启用插件时发生异常"""
        manager = PluginManager()
        
        # 注册并加载插件
        manager.registry.register(MockPlugin, "test_plugin")
        manager.load_plugin("test_plugin")
        
        # 模拟enable方法抛异常
        plugin = manager.get_plugin("test_plugin")
        plugin.enable = Mock(side_effect=Exception("Enable failed"))
        
        result = manager.enable_plugin("test_plugin")
        
        assert result is False
        assert "Failed to enable plugin test_plugin" in caplog.text
    
    def test_disable_plugin_success(self):
        """测试成功禁用插件"""
        manager = PluginManager()
        
        # 注册并加载插件
        manager.registry.register(MockPlugin, "test_plugin")
        manager.load_plugin("test_plugin")
        
        # 禁用插件
        result = manager.disable_plugin("test_plugin")
        
        assert result is True
    
    def test_disable_plugin_not_loaded(self, caplog):
        """测试禁用未加载的插件"""
        manager = PluginManager()
        
        # 禁用未加载的插件
        result = manager.disable_plugin("test_plugin")
        
        assert result is False
        assert "Plugin test_plugin is not loaded" in caplog.text
    
    def test_disable_plugin_no_disable_method(self):
        """测试禁用没有disable方法的插件"""
        manager = PluginManager()
        
        # 创建没有disable方法的插件
        class NoDisablePlugin(MockPlugin):
            pass  # 不包含disable方法
        
        # 注册并加载插件
        manager.registry.register(NoDisablePlugin, "test_plugin")
        manager.load_plugin("test_plugin")
        
        # 禁用插件应该返回True（默认已禁用）
        result = manager.disable_plugin("test_plugin")
        
        assert result is True
    
    def test_disable_plugin_exception(self, caplog):
        """测试禁用插件时发生异常"""
        manager = PluginManager()
        
        # 注册并加载插件
        manager.registry.register(MockPlugin, "test_plugin")
        manager.load_plugin("test_plugin")
        
        # 模拟disable方法抛异常
        plugin = manager.get_plugin("test_plugin")
        plugin.disable = Mock(side_effect=Exception("Disable failed"))
        
        result = manager.disable_plugin("test_plugin")
        
        assert result is False
        assert "Failed to disable plugin test_plugin" in caplog.text
    
    def test_shutdown_success(self):
        """测试成功关闭插件管理器"""
        manager = PluginManager()
        
        # 注册并加载多个插件
        manager.registry.register(MockPlugin, "plugin1")
        manager.registry.register(MockPlugin, "plugin2")
        manager.load_plugin("plugin1")
        manager.load_plugin("plugin2")
        
        # 关闭插件管理器
        manager.shutdown()
        
        # 所有插件应该被卸载
        assert len(manager.list_loaded_plugins()) == 0
    
    def test_shutdown_with_exception(self, caplog):
        """测试关闭插件管理器时发生异常"""
        manager = PluginManager()
        
        # 注册并加载插件
        manager.registry.register(MockPlugin, "test_plugin")
        manager.load_plugin("test_plugin")
        
        # 模拟unload_plugin抛异常
        with patch.object(manager, 'unload_plugin', side_effect=Exception("Unload failed")):
            manager.shutdown()
            
            assert "Error during plugin manager shutdown" in caplog.text
    
    def test_get_plugin_for_model_load_exception(self, caplog):
        """测试获取模型插件时加载失败"""
        manager = PluginManager()
        
        # 注册插件但不加载
        manager.registry.register(MockPlugin, "test_plugin")
        
        # 模拟load_plugin抛异常
        with patch.object(manager, 'load_plugin', side_effect=PluginLoadError("Load failed")):
            plugin = manager.get_plugin_for_model("mock-model-1")
            
            assert plugin is None
            assert "Failed to load plugin test_plugin for model mock-model-1" in caplog.text