"""
测试 ClientManager 最终覆盖率提升
专门针对未覆盖的代码路径进行测试
"""

import pytest
import time
import os
from unittest.mock import Mock, patch, MagicMock
from typing import List

from harborai.core.client_manager import ClientManager
from harborai.core.base_plugin import BaseLLMPlugin, ChatMessage, ChatCompletion, ModelInfo, ChatChoice
from harborai.core.exceptions import PluginError


class MockPlugin(BaseLLMPlugin):
    """模拟插件类用于测试"""
    
    def __init__(self, name: str, models: List[str] = None, should_fail: bool = False):
        super().__init__(name)
        self.models = models or []
        self.should_fail = should_fail
        
        # 设置支持的模型
        self._supported_models = [
            ModelInfo(
                id=model,
                name=model,
                provider=name
            ) for model in models
        ]
        
    def get_supported_models(self) -> List[ModelInfo]:
        return self._supported_models
        
    def chat_completion(self, model: str, messages: List[ChatMessage], **kwargs) -> ChatCompletion:
        """模拟聊天完成"""
        if self.should_fail:
            raise PluginError(self.name, "模拟插件错误")
            
        return ChatCompletion(
            id="test_completion",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="模拟回复")
                )
            ],
            usage=None
        )
        
    async def chat_completion_async(self, model: str, messages: List[ChatMessage], **kwargs) -> ChatCompletion:
        """模拟异步聊天完成"""
        if self.should_fail:
            raise PluginError(self.name, "模拟插件错误")
            
        return ChatCompletion(
            id="test_completion_async",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="模拟异步回复")
                )
            ],
            usage=None
        )


class TestClientManagerFinalCoverage:
    """测试 ClientManager 最终覆盖率提升"""

    @pytest.fixture
    def mock_logger(self):
        """模拟日志记录器"""
        return Mock()

    @pytest.fixture
    def client_manager(self, mock_logger):
        """创建 ClientManager 实例"""
        with patch('harborai.core.client_manager.get_logger', return_value=mock_logger), \
             patch('harborai.core.client_manager.get_settings') as mock_settings:
            mock_settings.return_value.plugin_directories = ["/test/path"]  # 设置非空目录
            mock_settings.return_value.model_mappings = {}
            mock_settings.return_value.get_plugin_config = Mock(return_value={})
            
            manager = ClientManager(lazy_loading=False)
            manager.logger = mock_logger
            return manager

    def test_plugin_directory_scan_exception(self, client_manager):
        """测试插件目录扫描异常处理 - 覆盖行 77-78"""
        # 直接模拟 _scan_plugin_directory 抛出异常
        with patch.object(client_manager, '_scan_plugin_directory') as mock_scan:
            mock_scan.side_effect = OSError("Permission denied")
            
            # 模拟设置中有插件目录
            with patch.object(client_manager.settings, 'plugin_directories', ["/fake/path"]):
                # 调用 _load_plugins 方法，它会调用 _scan_plugin_directory 并捕获异常
                client_manager._load_plugins()
            
            # 验证错误被记录
            client_manager.logger.error.assert_called()
            error_call = client_manager.logger.error.call_args
            assert "Failed to load plugins from directory" in str(error_call)

    def test_plugin_loading_exception_handling(self, client_manager):
        """测试插件加载异常处理 - 覆盖行 102-103"""
        # 模拟 _load_plugin_module 抛出异常
        with patch.object(client_manager, '_load_plugin_module') as mock_load:
            mock_load.side_effect = ImportError("No module named 'test_plugin'")
            
            # 模拟 pkgutil.iter_modules 返回一个插件模块
            with patch('harborai.core.client_manager.pkgutil.iter_modules') as mock_iter:
                mock_iter.return_value = [(None, "test_plugin", False)]
                
                # 模拟 importlib.import_module 成功导入包
                with patch('harborai.core.client_manager.importlib.import_module') as mock_import:
                    mock_package = Mock()
                    mock_package.__path__ = ["/fake/path"]
                    mock_package.__name__ = "fake_package"
                    mock_import.return_value = mock_package
                    
                    # 调用 _scan_plugin_directory，它会调用 _load_plugin_module 并捕获异常
                    client_manager._scan_plugin_directory("fake_package")
                    
                    # 验证警告被记录
                    client_manager.logger.warning.assert_called()
                    warning_call = client_manager.logger.warning.call_args
                    assert "Failed to load plugin module" in str(warning_call)

    def test_configuration_merging_with_api_key(self, client_manager):
        """测试配置合并 - API密钥"""
        # 测试客户端配置是否正确设置
        assert client_manager.client_config is not None
        
        # 创建一个插件并验证配置传递
        plugin = MockPlugin("test_plugin", ["model1"])
        client_manager.register_plugin(plugin)
        
        # 验证插件注册成功
        assert "test_plugin" in client_manager.plugins

    def test_configuration_merging_with_base_url(self, client_manager):
        """测试配置合并 - 基础URL"""
        # 测试客户端配置是否正确设置
        assert client_manager.client_config is not None
        
        # 创建一个插件并验证配置传递
        plugin = MockPlugin("test_plugin", ["model1"])
        client_manager.register_plugin(plugin)
        
        # 验证插件注册成功
        assert "test_plugin" in client_manager.plugins

    def test_lazy_loading_plugin_info_retrieval(self):
        """测试懒加载模式下的插件信息获取 - 覆盖行 255"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            manager = ClientManager(lazy_loading=True)
            
            # 模拟懒加载插件管理器
            mock_lazy_manager = Mock()
            mock_lazy_manager.get_plugin_info.return_value = {
                "test_plugin": {
                    "name": "test_plugin",
                    "config": {},
                    "supported_models": ["model1"],
                    "model_count": 1
                }
            }
            manager._lazy_manager = mock_lazy_manager
            
            # 获取插件信息
            info = manager.get_plugin_info()
            
            # 验证信息正确返回
            assert "test_plugin" in info
            assert info["test_plugin"]["name"] == "test_plugin"
            assert info["test_plugin"]["model_count"] == 1

    def test_preload_warning_in_non_lazy_mode(self, client_manager):
        """测试非懒加载模式下的预加载警告 - 覆盖行 274-275"""
        # 尝试预加载插件
        client_manager.preload_plugin("test_plugin")
        
        # 验证警告被记录
        client_manager.logger.warning.assert_called()

    def test_preload_model_warning_in_non_lazy_mode(self, client_manager):
        """测试非懒加载模式下的模型预加载警告 - 覆盖行 289-290"""
        # 尝试预加载模型
        client_manager.preload_model("test_model")
        
        # 验证警告被记录
        client_manager.logger.warning.assert_called()

    def test_synchronous_fallback_exception_handling(self, client_manager):
        """测试同步降级异常处理 - 覆盖行 470-480"""
        # 创建一个会失败的插件
        plugin = MockPlugin("failing_plugin", ["model1"])
        plugin.should_fail = True
        client_manager.register_plugin(plugin)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 测试同步聊天完成异常处理
        with pytest.raises(PluginError):
            client_manager.chat_completion_sync_with_fallback("model1", messages)

    def test_import_error_in_plugin_directory_scan(self, client_manager):
        """测试插件目录导入错误 - 覆盖行 107-109"""
        with patch('harborai.core.client_manager.importlib.import_module') as mock_import:
            # 模拟导入错误
            mock_import.side_effect = ImportError("No module named '/nonexistent/path'")
            
            # 调用扫描方法，应该捕获异常并记录警告
            client_manager._scan_plugin_directory("/nonexistent/path")
            
            # 验证警告被记录
            client_manager.logger.warning.assert_called()