#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClientManager 覆盖率增强测试

专门针对提升 ClientManager 代码覆盖率的测试用例。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from harborai.core.client_manager import ClientManager
from harborai.core.base_plugin import ChatMessage
from harborai.config.settings import Settings


@pytest.fixture
def mock_settings():
    """模拟设置对象"""
    settings = Mock(spec=Settings)
    settings.plugin_directories = ["harborai.plugins"]
    settings.get_plugin_config.return_value = {"api_key": "test_key"}
    settings.model_mappings = {}
    return settings


class TestClientManagerCoverage:
    """ClientManager 覆盖率增强测试"""

    def test_plugin_directory_scan_exception_handling(self, mock_settings):
        """测试插件目录扫描异常处理"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings), \
             patch('harborai.core.client_manager.importlib.import_module') as mock_import:
            
            # 模拟导入插件包时抛出异常
            mock_import.side_effect = ImportError("插件包导入失败")
            
            # 创建ClientManager实例
            manager = ClientManager()
            
            # 验证异常被捕获，插件字典为空
            assert len(manager.plugins) == 0

    def test_plugin_module_load_exception_handling(self, mock_settings):
        """测试插件模块加载异常处理"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings), \
             patch('harborai.core.client_manager.importlib.import_module') as mock_import, \
             patch('harborai.core.client_manager.pkgutil.iter_modules') as mock_iter:
            
            # 模拟包导入成功
            mock_package = Mock()
            mock_package.__path__ = ["/fake/path"]
            mock_package.__name__ = "harborai.plugins"
            mock_import.side_effect = [mock_package, ImportError("模块加载失败")]
            
            # 模拟找到插件模块
            mock_iter.return_value = [(None, "harborai.plugins.test_plugin", False)]
            
            # 创建ClientManager实例
            manager = ClientManager()
            
            # 验证异常被捕获，插件字典为空
            assert len(manager.plugins) == 0

    def test_lazy_loading_mode_get_plugin_info(self, mock_settings):
        """测试延迟加载模式下的get_plugin_info方法"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings), \
             patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            
            # 模拟LazyPluginManager实例
            mock_lazy_manager = Mock()
            mock_lazy_manager.get_plugin_info.return_value = {"test": "info"}
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            # 创建延迟加载模式的ClientManager
            manager = ClientManager(lazy_loading=True)
            
            # 调用get_plugin_info
            result = manager.get_plugin_info()
            
            # 验证调用了LazyPluginManager的方法
            mock_lazy_manager.get_plugin_info.assert_called_once()
            assert result == {"test": "info"}

    def test_preload_plugin_lazy_mode(self, mock_settings):
        """测试延迟加载模式下的preload_plugin方法"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings), \
             patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            
            # 模拟LazyPluginManager实例
            mock_lazy_manager = Mock()
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            # 创建延迟加载模式的ClientManager
            manager = ClientManager(lazy_loading=True)
            
            # 调用preload_plugin
            manager.preload_plugin("test_plugin")
            
            # 验证调用了LazyPluginManager的方法
            mock_lazy_manager.preload_plugin.assert_called_once_with("test_plugin")

    def test_preload_plugin_non_lazy_mode(self, mock_settings):
        """测试非延迟加载模式下的preload_plugin方法警告"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            
            # 创建非延迟加载模式的ClientManager
            manager = ClientManager(lazy_loading=False)
            
            # 调用preload_plugin应该记录警告
            with patch.object(manager.logger, 'warning') as mock_warning:
                manager.preload_plugin("test_plugin")
                mock_warning.assert_called_once()

    def test_preload_model_lazy_mode(self, mock_settings):
        """测试延迟加载模式下的preload_model方法"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings), \
             patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            
            # 模拟LazyPluginManager实例
            mock_lazy_manager = Mock()
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            # 创建延迟加载模式的ClientManager
            manager = ClientManager(lazy_loading=True)
            
            # 调用preload_model
            manager.preload_model("test_model")
            
            # 验证调用了LazyPluginManager的方法
            mock_lazy_manager.preload_model.assert_called_once_with("test_model")

    def test_preload_model_non_lazy_mode(self, mock_settings):
        """测试非延迟加载模式下的preload_model方法警告"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            
            # 创建非延迟加载模式的ClientManager
            manager = ClientManager(lazy_loading=False)
            
            # 调用preload_model应该记录警告
            with patch.object(manager.logger, 'warning') as mock_warning:
                manager.preload_model("test_model")
                mock_warning.assert_called_once()

    def test_process_messages_for_reasoning_model_system_merge(self, mock_settings):
        """测试推理模型的系统消息合并逻辑"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            
            # 创建ClientManager实例
            manager = ClientManager()
            
            # 创建测试消息
            messages = [
                ChatMessage(role="system", content="系统消息"),
                ChatMessage(role="user", content="用户消息")
            ]
            
            # 调用_process_messages_for_reasoning_model
            result = manager._process_messages_for_reasoning_model(messages)
            
            # 验证系统消息被合并到用户消息中
            assert len(result) == 1
            assert result[0].role == "user"
            assert "系统消息" in result[0].content
            assert "用户消息" in result[0].content

    def test_get_available_models_lazy_mode(self, mock_settings):
        """测试延迟加载模式下的get_available_models方法"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings), \
             patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            
            # 模拟LazyPluginManager实例
            mock_lazy_manager = Mock()
            # 模拟get_supported_models方法返回模型名称列表
            mock_lazy_manager.get_supported_models.return_value = ["test_model"]
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            # 创建延迟加载模式的ClientManager
            manager = ClientManager(lazy_loading=True)
            
            # 调用get_available_models
            result = manager.get_available_models()
            
            # 验证调用了LazyPluginManager的get_supported_models方法
            mock_lazy_manager.get_supported_models.assert_called_once()
            # 验证返回的是ModelInfo对象列表
            assert len(result) == 1
            assert result[0].id == "test_model"
            assert result[0].name == "test_model"

    def test_get_loading_statistics_lazy_mode(self, mock_settings):
        """测试延迟加载模式下的get_loading_statistics方法"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings), \
             patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            
            # 模拟LazyPluginManager实例
            mock_lazy_manager = Mock()
            mock_stats = {"mode": "lazy", "loaded_plugins": 0}
            mock_lazy_manager.get_statistics.return_value = mock_stats
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            # 创建延迟加载模式的ClientManager
            manager = ClientManager(lazy_loading=True)
            
            # 调用get_loading_statistics
            result = manager.get_loading_statistics()
            
            # 验证调用了LazyPluginManager的方法
            mock_lazy_manager.get_statistics.assert_called_once()
            assert result == mock_stats

    def test_get_loading_statistics_traditional_mode(self, mock_settings):
        """测试传统模式下的get_loading_statistics方法"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            
            # 创建传统模式的ClientManager
            manager = ClientManager(lazy_loading=False)
            
            # 调用get_loading_statistics
            result = manager.get_loading_statistics()
            
            # 验证返回传统模式的统计信息
            assert result["mode"] == "traditional"
            assert "loaded_plugins" in result
            assert "total_models" in result
            assert "plugin_names" in result

    def test_get_plugin_for_model_with_model_mapping(self, mock_settings):
        """测试通过模型映射获取插件"""
        # 设置模型映射
        mock_settings.model_mappings = {"mapped_model": "real_model"}
        
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            
            # 创建ClientManager实例
            manager = ClientManager()
            
            # 创建模拟插件
            mock_plugin = Mock()
            manager.plugins = {"test_plugin": mock_plugin}
            manager.model_to_plugin = {"real_model": "test_plugin"}
            
            # 调用get_plugin_for_model使用映射的模型
            result = manager.get_plugin_for_model("mapped_model")
            
            # 验证返回正确的插件
            assert result == mock_plugin

    def test_get_plugin_for_model_lazy_fallback(self, mock_settings):
        """测试延迟加载模式下的插件获取回退逻辑"""
        from harborai.utils.exceptions import ModelNotFoundError
        
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings), \
             patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            
            # 模拟LazyPluginManager实例
            mock_lazy_manager = Mock()
            mock_lazy_manager.get_plugin_for_model.side_effect = ModelNotFoundError("test_model")
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            # 创建延迟加载模式的ClientManager
            manager = ClientManager(lazy_loading=True)
            
            # 创建模拟插件用于回退
            mock_plugin = Mock()
            manager.plugins = {"test_plugin": mock_plugin}
            manager.model_to_plugin = {"test_model": "test_plugin"}
            
            # 调用get_plugin_for_model
            result = manager.get_plugin_for_model("test_model")
            
            # 验证先尝试LazyPluginManager，然后回退到传统方式
            mock_lazy_manager.get_plugin_for_model.assert_called_once_with("test_model")
            assert result == mock_plugin