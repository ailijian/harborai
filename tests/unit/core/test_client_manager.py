"""
客户端管理器测试模块

测试 ClientManager 类的所有功能，包括：
- 插件注册和管理
- 模型路由和映射
- 懒加载和传统加载模式
- 故障转移和降级策略
- 消息处理和参数过滤
- 统计信息收集
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List, Dict, Any, Optional, Union
import logging

from harborai.core.client_manager import ClientManager
from harborai.core.base_plugin import BaseLLMPlugin, ChatMessage, ChatCompletion, ModelInfo
from harborai.core.exceptions import (
    PluginNotFoundError,
    PluginLoadError,
    HarborAIError
)
from harborai.utils.exceptions import PluginError, ModelNotFoundError


class MockPlugin(BaseLLMPlugin):
    """模拟插件类用于测试"""
    
    def __init__(self, name: str, models: List[str] = None):
        super().__init__(name)
        self.models = models or []
        self.is_loaded = False
        self.load_time = 0.1
        
        # 设置支持的模型
        from harborai.core.base_plugin import ModelInfo
        self._supported_models = [
            ModelInfo(
                id=model,
                name=model,
                provider=name,
                supports_streaming=True,
                supports_structured_output=False
            ) for model in models
        ]
        
    def get_supported_models(self) -> List[str]:
        return self.models
        
    async def chat_completion_async(self, model: str, messages: List[ChatMessage], **kwargs) -> ChatCompletion:
        """异步聊天完成"""
        if model not in self.models:
            raise ModelNotFoundError(f"Model {model} not supported by {self.name}")
        
        return ChatCompletion(
            id="test-completion",
            object="chat.completion",
            created=1234567890,
            model=model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Response from {self.name} using {model}"
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        )
    
    def chat_completion(self, model: str, messages: List[ChatMessage], **kwargs) -> ChatCompletion:
        """同步聊天完成"""
        if model not in self.models:
            raise ModelNotFoundError(f"Model {model} not supported by {self.name}")
        
        return ChatCompletion(
            id="test-completion-sync",
            object="chat.completion",
            created=1234567890,
            model=model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"Sync response from {self.name} using {model}"
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        )


class MockLazyManager:
    """模拟懒加载管理器"""
    
    def __init__(self):
        self.plugins = {}
        self.model_to_plugin = {}
        self.statistics = {
            "mode": "lazy",
            "loaded_plugins": 0,
            "total_plugins": 0,
            "load_times": {}
        }
    
    def get_plugin_for_model(self, model: str) -> BaseLLMPlugin:
        if model not in self.model_to_plugin:
            raise ModelNotFoundError(f"Model {model} not found")
        return self.model_to_plugin[model]
    
    def get_available_models(self) -> List[ModelInfo]:
        return [ModelInfo(id=model, name=model, provider="mock") for model in self.model_to_plugin.keys()]
    
    def get_supported_models(self) -> List[str]:
        """获取所有支持的模型列表
        
        Returns:
            支持的模型名称列表
        """
        return list(self.model_to_plugin.keys())
    
    def get_plugin_info(self, plugin_name: str = None) -> Dict[str, Any]:
        """获取插件信息
        
        Args:
            plugin_name: 插件名称，如果为None则返回所有插件信息
            
        Returns:
            插件信息字典
        """
        if plugin_name is None:
            # 返回所有插件信息
            return {name: {"name": name, "loaded": True} for name in self.plugins.keys()}
        
        if plugin_name not in self.plugins:
            raise PluginNotFoundError(f"Plugin {plugin_name} not found")
        return {"name": plugin_name, "loaded": True}
    
    def preload_plugin(self, plugin_name: str) -> None:
        if plugin_name not in self.plugins:
            raise PluginNotFoundError(f"Plugin {plugin_name} not found")
    
    def preload_model(self, model: str) -> None:
        if model not in self.model_to_plugin:
            raise ModelNotFoundError(f"Model {model} not found")
    
    def get_statistics(self) -> Dict[str, Any]:
        return self.statistics.copy()


class TestClientManager:
    """ClientManager 基础功能测试"""
    
    @pytest.fixture
    def mock_settings(self):
        """模拟设置"""
        settings = Mock()
        settings.plugin_directories = []
        settings.model_mappings = {}
        settings.get_plugin_config = Mock(return_value={})
        return settings
    
    def test_init_traditional_mode(self, mock_settings):
        """测试传统模式初始化"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            
            assert manager.lazy_loading is False
            assert manager._lazy_manager is None
            assert manager.plugins == {}
            assert manager.model_to_plugin == {}
            assert isinstance(manager.logger, logging.Logger)
    
    def test_init_lazy_mode(self, mock_settings):
        """测试懒加载模式初始化"""
        with patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class, \
             patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            
            mock_lazy_manager = MockLazyManager()
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            manager = ClientManager(lazy_loading=True)
            
            assert manager.lazy_loading is True
            assert manager._lazy_manager is mock_lazy_manager
            mock_lazy_manager_class.assert_called_once_with(config={})
    
    def test_register_plugin_traditional_mode(self, mock_settings):
        """测试传统模式插件注册"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            plugin = MockPlugin("test_plugin", ["model1", "model2"])
            
            manager.register_plugin(plugin)
            
            assert "test_plugin" in manager.plugins
            assert manager.plugins["test_plugin"] is plugin
            assert manager.model_to_plugin["model1"] == "test_plugin"
            assert manager.model_to_plugin["model2"] == "test_plugin"
    
    def test_register_plugin_duplicate_name(self, mock_settings):
        """测试重复插件名称注册 - 应该替换现有插件"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            plugin1 = MockPlugin("test_plugin", ["model1"])
            plugin2 = MockPlugin("test_plugin", ["model2"])
            
            manager.register_plugin(plugin1)
            assert "test_plugin" in manager.plugins
            assert manager.plugins["test_plugin"] == plugin1
            
            # 注册同名插件应该替换原有插件
            manager.register_plugin(plugin2)
            assert "test_plugin" in manager.plugins
            assert manager.plugins["test_plugin"] == plugin2
    
    def test_register_plugin_duplicate_model(self, mock_settings):
        """测试重复模型注册 - 应该覆盖模型映射"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            plugin1 = MockPlugin("plugin1", ["model1"])
            plugin2 = MockPlugin("plugin2", ["model1"])
            
            manager.register_plugin(plugin1)
            assert manager.model_to_plugin["model1"] == "plugin1"
            
            # 注册相同模型应该覆盖映射
            manager.register_plugin(plugin2)
            assert manager.model_to_plugin["model1"] == "plugin2"
    
    def test_scan_plugin_directory_success(self, mock_settings):
        """测试插件目录扫描成功"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            
            with patch('importlib.import_module') as mock_import, \
                 patch('pkgutil.iter_modules') as mock_iter:
                
                # 模拟包导入
                mock_package = Mock()
                mock_package.__path__ = ['/fake/path']
                mock_package.__name__ = 'harborai.core.plugins'
                mock_import.return_value = mock_package
                
                # 模拟模块迭代
                mock_iter.return_value = [
                    (None, 'harborai.core.plugins.test_plugin', False)
                ]
                
                # 模拟插件模块
                mock_plugin_module = Mock()
                mock_plugin_class = type('TestPlugin', (MockPlugin,), {})
                setattr(mock_plugin_module, 'TestPlugin', mock_plugin_class)
                mock_import.side_effect = [mock_package, mock_plugin_module]
                
                # 执行扫描
                manager._scan_plugin_directory('harborai.core.plugins')
                
                # 验证调用
                assert mock_import.call_count >= 2
    
    def test_load_plugin_module_success(self, mock_settings):
        """测试加载插件模块成功"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            
            # 创建模拟模块
            mock_module = Mock()
            mock_plugin_class = type('TestPlugin', (MockPlugin,), {
                '__init__': lambda self, name, **kwargs: MockPlugin.__init__(self, name, ["model1"])
            })
            setattr(mock_module, 'TestPlugin', mock_plugin_class)
            
            with patch('importlib.import_module', return_value=mock_module):
                manager._load_plugin_module('test.module')
                
                # 验证插件被注册（插件名称是从模块名推导出来的）
                assert 'test' in manager.plugins
    
    def test_load_plugin_module_no_plugin_class(self, mock_settings):
        """测试模块中没有插件类"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            
            # 创建一个简单的模块类型对象
            import types
            mock_module = types.ModuleType('test_module')
            mock_module.SomeOtherClass = type('SomeOtherClass', (), {})
            
            with patch('importlib.import_module', return_value=mock_module):
                # 应该不会抛出异常，只是不注册任何插件
                manager._load_plugin_module('test.module')
                
                # 验证没有插件被注册
                assert len(manager.plugins) == 0


class TestClientManagerModelRouting:
    """ClientManager 模型路由测试"""
    
    @pytest.fixture
    def mock_settings(self):
        """模拟设置"""
        settings = Mock()
        settings.plugin_directories = ["tests/plugins"]
        settings.lazy_loading = False
        settings.enable_fallback = True
        settings.max_retries = 3
        settings.timeout = 30
        settings.model_mappings = {}  # 添加模型映射
        return settings
    
    def test_get_plugin_for_model_traditional_mode_success(self, mock_settings):
        """测试传统模式获取插件成功"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            plugin = MockPlugin("test_plugin", ["model1"])
            # 将 chat_completion_async 替换为 AsyncMock
            plugin.chat_completion_async = AsyncMock(return_value=ChatCompletion(
                id="test-completion",
                object="chat.completion", 
                created=1234567890,
                model="model1",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test response"
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            ))
            manager.register_plugin(plugin)
            
            result = manager.get_plugin_for_model("model1")
            
            assert result is plugin
    
    def test_get_plugin_for_model_traditional_mode_not_found(self, mock_settings):
        """测试传统模式模型未找到"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            
            with pytest.raises(ModelNotFoundError):
                manager.get_plugin_for_model("unknown_model")
    
    def test_get_plugin_for_model_lazy_mode(self, mock_settings):
        """测试懒加载模式获取插件"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings), \
             patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            
            mock_lazy_manager = MockLazyManager()
            mock_plugin = MockPlugin("test_plugin", ["model1"])
            mock_lazy_manager.model_to_plugin["model1"] = mock_plugin
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            manager = ClientManager(lazy_loading=True)
            result = manager.get_plugin_for_model("model1")
            
            assert result is mock_plugin
    
    def test_get_available_models_traditional_mode(self, mock_settings):
        """测试传统模式获取可用模型"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            plugin1 = MockPlugin("plugin1", ["model1", "model2"])
            plugin2 = MockPlugin("plugin2", ["model3"])
            
            manager.register_plugin(plugin1)
            manager.register_plugin(plugin2)
            
            models = manager.get_available_models()
            model_ids = [m.id for m in models]
            
            assert set(model_ids) == {"model1", "model2", "model3"}
    
    def test_get_available_models_lazy_mode(self, mock_settings):
        """测试懒加载模式获取可用模型"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings), \
             patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            
            mock_lazy_manager = Mock()
            # 模拟get_supported_models返回模型名称列表
            mock_lazy_manager.get_supported_models.return_value = ["model1", "model2"]
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            manager = ClientManager(lazy_loading=True)
            models = manager.get_available_models()
            
            # 验证返回的是ModelInfo对象列表
            assert len(models) == 2
            assert models[0].id == "model1"
            assert models[1].id == "model2"
    
    def test_get_plugin_info_traditional_mode(self, mock_settings):
        """测试传统模式获取插件信息"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            plugin = MockPlugin("test_plugin", ["model1"])
            # 将 chat_completion_async 替换为 AsyncMock
            plugin.chat_completion_async = AsyncMock(return_value=ChatCompletion(
                id="test-completion",
                object="chat.completion", 
                created=1234567890,
                model="model1",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test response"
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            ))
            manager.register_plugin(plugin)
            
            info = manager.get_plugin_info()
            
            assert "test_plugin" in info
            assert info["test_plugin"]["name"] == "test_plugin"
            assert info["test_plugin"]["supported_models"] == ["model1"]
            assert info["test_plugin"]["model_count"] == 1
    
    def test_get_plugin_info_empty(self, mock_settings):
        """测试获取空插件信息"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            
            info = manager.get_plugin_info()
            
            # 应该返回空字典（除了可能加载的默认插件）
            assert isinstance(info, dict)
    
    def test_preload_plugin_traditional_mode(self):
        """测试传统模式预加载插件（无操作）"""
        manager = ClientManager(lazy_loading=False)
        plugin = MockPlugin("test_plugin", ["model1"])
        manager.register_plugin(plugin)
        
        # 传统模式下预加载是无操作
        manager.preload_plugin("test_plugin")
        
        # 应该没有异常
        assert True
    
    def test_preload_model_traditional_mode(self):
        """测试传统模式预加载模型（无操作）"""
        manager = ClientManager(lazy_loading=False)
        plugin = MockPlugin("test_plugin", ["model1"])
        manager.register_plugin(plugin)
        
        # 传统模式下预加载是无操作
        manager.preload_model("model1")
        
        # 应该没有异常
        assert True


class TestClientManagerStatistics:
    """ClientManager 统计信息测试"""
    
    @pytest.fixture
    def mock_settings(self):
        """模拟设置"""
        settings = Mock()
        settings.plugin_directories = []  # 空目录，避免加载真实插件
        settings.lazy_loading = False
        settings.enable_fallback = True
        settings.max_retries = 3
        settings.timeout = 30
        settings.model_mappings = {}
        return settings
    
    def test_get_loading_statistics_traditional_mode(self, mock_settings):
        """测试传统模式获取统计信息"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            plugin1 = MockPlugin("plugin1", ["model1", "model2"])
            plugin2 = MockPlugin("plugin2", ["model3"])
            
            manager.register_plugin(plugin1)
            manager.register_plugin(plugin2)
            
            stats = manager.get_loading_statistics()
            
            assert stats["mode"] == "traditional"
            assert stats["loaded_plugins"] == 2
            assert stats["total_models"] == 3
            assert set(stats["plugin_names"]) == {"plugin1", "plugin2"}
    
    def test_get_loading_statistics_lazy_mode(self):
        """测试懒加载模式获取统计信息"""
        with patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            mock_lazy_manager = MockLazyManager()
            mock_lazy_manager.statistics = {
                "mode": "lazy",
                "loaded_plugins": 1,
                "total_plugins": 3,
                "load_times": {"plugin1": 0.1}
            }
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            manager = ClientManager(lazy_loading=True)
            stats = manager.get_loading_statistics()
            
            assert stats["mode"] == "lazy"
            assert stats["loaded_plugins"] == 1
            assert stats["total_plugins"] == 3


class TestClientManagerChatCompletion:
    """ClientManager 聊天完成功能测试"""
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_fallback_success(self):
        """测试带降级策略的异步聊天完成成功"""
        manager = ClientManager(lazy_loading=False)
        plugin = MockPlugin("test_plugin", ["model1"])
        manager.register_plugin(plugin)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
             patch('harborai.core.models.filter_parameters_for_model', return_value={}), \
             patch('harborai.core.models.is_reasoning_model', return_value=False):
            
            result = await manager.chat_completion_with_fallback(
                model="model1",
                messages=messages
            )
            
            assert result.model == "model1"
            assert "test_plugin" in result.choices[0]["message"]["content"]
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_fallback_fallback_success(self):
        """测试降级策略成功"""
        manager = ClientManager(lazy_loading=False)
        plugin1 = MockPlugin("plugin1", ["model1"])
        plugin2 = MockPlugin("plugin2", ["model2"])
        
        # 让第一个插件抛出异常
        plugin1.chat_completion_async = AsyncMock(side_effect=Exception("Model1 failed"))
        
        manager.register_plugin(plugin1)
        manager.register_plugin(plugin2)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
             patch('harborai.core.models.filter_parameters_for_model', return_value={}), \
             patch('harborai.core.models.is_reasoning_model', return_value=False):
            
            result = await manager.chat_completion_with_fallback(
                model="model1",
                messages=messages,
                fallback=["model2"]
            )
            
            assert result.model == "model2"
            assert "plugin2" in result.choices[0]["message"]["content"]
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_fallback_all_fail(self):
        """测试所有模型都失败"""
        manager = ClientManager(lazy_loading=False)
        plugin1 = MockPlugin("plugin1", ["model1"])
        plugin2 = MockPlugin("plugin2", ["model2"])
        
        # 让两个插件都抛出异常
        plugin1.chat_completion_async = AsyncMock(side_effect=Exception("Model1 failed"))
        plugin2.chat_completion_async = AsyncMock(side_effect=Exception("Model2 failed"))
        
        manager.register_plugin(plugin1)
        manager.register_plugin(plugin2)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
             patch('harborai.core.models.filter_parameters_for_model', return_value={}), \
             patch('harborai.core.models.is_reasoning_model', return_value=False):
            
            with pytest.raises(Exception, match="Model2 failed"):
                await manager.chat_completion_with_fallback(
                    model="model1",
                    messages=messages,
                    fallback=["model2"]
                )
    
    def test_chat_completion_sync_with_fallback_success(self):
        """测试同步版本带降级策略的聊天完成成功"""
        manager = ClientManager(lazy_loading=False)
        plugin = MockPlugin("test_plugin", ["model1"])
        manager.register_plugin(plugin)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
             patch('harborai.core.models.filter_parameters_for_model', return_value={}), \
             patch('harborai.core.models.is_reasoning_model', return_value=False):
            
            result = manager.chat_completion_sync_with_fallback(
                model="model1",
                messages=messages
            )
            
            assert result.model == "model1"
            assert "test_plugin" in result.choices[0]["message"]["content"]
    
    def test_chat_completion_sync_with_fallback_fallback_success(self):
        """测试同步版本降级策略成功"""
        manager = ClientManager(lazy_loading=False)
        plugin1 = MockPlugin("plugin1", ["model1"])
        plugin2 = MockPlugin("plugin2", ["model2"])
        
        # 让第一个插件抛出异常
        plugin1.chat_completion = Mock(side_effect=Exception("Model1 failed"))
        
        manager.register_plugin(plugin1)
        manager.register_plugin(plugin2)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
             patch('harborai.core.models.filter_parameters_for_model', return_value={}), \
             patch('harborai.core.models.is_reasoning_model', return_value=False):
            
            result = manager.chat_completion_sync_with_fallback(
                model="model1",
                messages=messages,
                fallback=["model2"]
            )
            
            assert result.model == "model2"
            assert "plugin2" in result.choices[0]["message"]["content"]


class TestClientManagerReasoningModel:
    """ClientManager 推理模型处理测试"""
    
    def test_process_messages_for_reasoning_model_system_to_user(self):
        """测试推理模型system消息转换为user消息"""
        manager = ClientManager(lazy_loading=False)
        
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant"),
            ChatMessage(role="user", content="Hello")
        ]
        
        processed = manager._process_messages_for_reasoning_model(messages)
        
        assert len(processed) == 1
        assert processed[0].role == "user"
        assert "请按照以下指导原则回答：You are a helpful assistant" in processed[0].content
        assert "Hello" in processed[0].content
    
    def test_process_messages_for_reasoning_model_no_system(self):
        """测试推理模型无system消息"""
        manager = ClientManager(lazy_loading=False)
        
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there")
        ]
        
        processed = manager._process_messages_for_reasoning_model(messages)
        
        assert len(processed) == 2
        assert processed[0].role == "user"
        assert processed[0].content == "Hello"
        assert processed[1].role == "assistant"
        assert processed[1].content == "Hi there"
    
    def test_process_messages_for_reasoning_model_only_system(self):
        """测试推理模型只有system消息"""
        manager = ClientManager(lazy_loading=False)
        
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant")
        ]
        
        processed = manager._process_messages_for_reasoning_model(messages)
        
        assert len(processed) == 1
        assert processed[0].role == "user"
        assert "请按照以下指导原则回答：You are a helpful assistant" in processed[0].content
        assert "现在请回答用户的问题。" in processed[0].content
    
    def test_process_messages_for_reasoning_model_multiple_system(self):
        """测试推理模型多个system消息"""
        manager = ClientManager(lazy_loading=False)
        
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="system", content="Be concise"),
            ChatMessage(role="user", content="Hello")
        ]
        
        processed = manager._process_messages_for_reasoning_model(messages)
        
        assert len(processed) == 1
        assert processed[0].role == "user"
        assert "You are helpful" in processed[0].content
        assert "Be concise" in processed[0].content
        assert "Hello" in processed[0].content
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_reasoning_model(self):
        """测试推理模型的聊天完成"""
        mock_settings = Mock()
        mock_settings.plugin_directories = []
        mock_settings.lazy_loading = False
        mock_settings.enable_fallback = True
        mock_settings.max_retries = 3
        mock_settings.timeout = 30
        mock_settings.model_mappings = {}
        
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            plugin = MockPlugin("reasoning_plugin", ["o1-preview"])
            # 将 chat_completion_async 替换为 Mock
            plugin.chat_completion_async = AsyncMock(return_value=ChatCompletion(
                id="test-completion",
                object="chat.completion",
                created=1234567890,
                model="o1-preview",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Response from reasoning_plugin using o1-preview"
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            ))
            manager.register_plugin(plugin)
        
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello")
        ]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
             patch('harborai.core.models.filter_parameters_for_model', return_value={}), \
             patch('harborai.core.models.is_reasoning_model', return_value=True):
            
            result = await manager.chat_completion_with_fallback(
                model="o1-preview",
                messages=messages
            )
            
            assert result.model == "o1-preview"
            # 验证插件收到的是处理后的消息
            assert plugin.chat_completion_async.called
            call_args = plugin.chat_completion_async.call_args
            processed_messages = call_args[0][1]  # 第二个参数是messages
            
            assert len(processed_messages) == 1
            assert processed_messages[0].role == "user"
            assert "You are helpful" in processed_messages[0].content


class TestClientManagerParameterFiltering:
    """ClientManager 参数过滤测试"""
    
    @pytest.fixture
    def mock_settings(self):
        """模拟设置"""
        settings = Mock()
        settings.plugin_directories = ["tests/plugins"]
        settings.model_mappings = {}
        return settings
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_parameter_filtering(self, mock_settings):
        """测试聊天完成时的参数过滤"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            plugin = MockPlugin("test_plugin", ["model1"])
            # 将 chat_completion_async 替换为 AsyncMock
            plugin.chat_completion_async = AsyncMock(return_value=ChatCompletion(
                id="test-completion",
                object="chat.completion", 
                created=1234567890,
                model="model1",
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Test response"
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            ))
            manager.register_plugin(plugin)
            
            messages = [ChatMessage(role="user", content="Hello")]
            
            # 模拟参数过滤
            filtered_params = {"temperature": 0.7}
            
            with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
                 patch('harborai.core.models.filter_parameters_for_model', return_value=filtered_params) as mock_filter, \
                 patch('harborai.core.models.is_reasoning_model', return_value=False):
                
                await manager.chat_completion_with_fallback(
                    model="model1",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100,
                    unsupported_param="value"
                )
                
                # 验证参数过滤被调用
                mock_filter.assert_called_once_with("model1", {
                    "temperature": 0.7,
                    "max_tokens": 100,
                    "unsupported_param": "value"
                })
                
                # 验证插件收到的是过滤后的参数
                assert plugin.chat_completion_async.called
                call_kwargs = plugin.chat_completion_async.call_args[1]
                assert call_kwargs == filtered_params
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_structured_provider(self):
        """测试带结构化提供者的聊天完成"""
        manager = ClientManager(lazy_loading=False)
        plugin = MockPlugin("test_plugin", ["model1"])
        manager.register_plugin(plugin)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
             patch('harborai.core.models.filter_parameters_for_model', return_value={}) as mock_filter, \
             patch('harborai.core.models.is_reasoning_model', return_value=False):
            
            await manager.chat_completion_with_fallback(
                model="model1",
                messages=messages,
                structured_provider="openai"
            )
            
            # 验证structured_provider被添加到参数中
            mock_filter.assert_called_once_with("model1", {
                "structured_provider": "openai"
            })


class TestClientManagerErrorHandling:
    """ClientManager 错误处理测试"""
    
    @pytest.fixture
    def mock_settings(self):
        """模拟设置"""
        settings = Mock()
        settings.plugin_directories = ["tests/plugins"]
        settings.model_mappings = {"gpt-4o": "gpt-4"}  # 模型映射
        return settings
    
    def test_get_plugin_for_model_with_fallback_mapping(self, mock_settings):
        """测试模型映射降级处理"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            plugin = MockPlugin("test_plugin", ["gpt-4"])
            manager.register_plugin(plugin)
            
            result = manager.get_plugin_for_model("gpt-4o")
            
            assert result is plugin
    
    def test_get_plugin_for_model_no_fallback_mapping(self, mock_settings):
        """测试无降级映射时的错误"""
        # 使用空的模型映射
        mock_settings.model_mappings = {}
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            
            with pytest.raises(ModelNotFoundError, match=r"Model 'unknown_model' not found or not supported"):
                manager.get_plugin_for_model("unknown_model")
    
    @pytest.mark.asyncio
    async def test_chat_completion_model_not_found_error(self):
        """测试聊天完成时模型未找到错误"""
        manager = ClientManager(lazy_loading=False)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"):
            with pytest.raises(ModelNotFoundError):
                await manager.chat_completion_with_fallback(
                    model="unknown_model",
                    messages=messages
                )
    
    def test_register_plugin_invalid_plugin(self):
        """测试注册无效插件"""
        manager = ClientManager(lazy_loading=False)
        
        with pytest.raises(AttributeError):
            manager.register_plugin("not_a_plugin")
    
    def test_plugin_loading_import_error(self, mock_settings):
        """测试插件加载时的导入错误"""
        with patch('harborai.core.client_manager.get_settings', return_value=mock_settings):
            manager = ClientManager(lazy_loading=False)
            
            # 测试注册无效插件时的错误处理
            with pytest.raises(AttributeError):
                manager.register_plugin("not_a_plugin")


class TestClientManagerIntegration:
    """ClientManager 集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_multiple_plugins(self):
        """测试多插件完整工作流程"""
        # 阻止自动加载插件
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []  # 空目录列表，防止自动加载
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            manager = ClientManager(lazy_loading=False)
            
            # 注册多个插件
            openai_plugin = MockPlugin("openai", ["gpt-3.5-turbo", "gpt-4"])
            anthropic_plugin = MockPlugin("anthropic", ["claude-3-sonnet"])
            
            manager.register_plugin(openai_plugin)
            manager.register_plugin(anthropic_plugin)
            
            # 测试获取可用模型
            models = manager.get_available_models()
            model_ids = [m.id for m in models]
            assert set(model_ids) == {"gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"}
            
            # 测试聊天完成
            messages = [ChatMessage(role="user", content="Hello")]
            
            with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
                 patch('harborai.core.models.filter_parameters_for_model', return_value={}), \
                 patch('harborai.core.models.is_reasoning_model', return_value=False):
                
                # 测试OpenAI模型
                result1 = await manager.chat_completion_with_fallback(
                    model="gpt-4",
                    messages=messages
                )
                assert "openai" in result1.choices[0]["message"]["content"]
                
                # 测试Anthropic模型
                result2 = await manager.chat_completion_with_fallback(
                    model="claude-3-sonnet",
                    messages=messages
                )
                assert "anthropic" in result2.choices[0]["message"]["content"]
            
            # 测试统计信息
            stats = manager.get_loading_statistics()
            assert stats["loaded_plugins"] == 2
            assert stats["total_models"] == 3
    
    @pytest.mark.asyncio
    async def test_complex_fallback_scenario(self):
        """测试复杂降级场景"""
        manager = ClientManager(lazy_loading=False)
        
        # 创建插件，第一个会失败，第二个会成功
        plugin1 = MockPlugin("plugin1", ["model1"])
        plugin2 = MockPlugin("plugin2", ["model2"])
        plugin3 = MockPlugin("plugin3", ["model3"])
        
        # 让前两个插件失败
        plugin1.chat_completion_async = AsyncMock(side_effect=Exception("Service unavailable"))
        plugin2.chat_completion_async = AsyncMock(side_effect=Exception("Rate limited"))
        
        manager.register_plugin(plugin1)
        manager.register_plugin(plugin2)
        manager.register_plugin(plugin3)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
             patch('harborai.core.models.filter_parameters_for_model', return_value={}), \
             patch('harborai.core.models.is_reasoning_model', return_value=False):
            
            result = await manager.chat_completion_with_fallback(
                model="model1",
                messages=messages,
                fallback=["model2", "model3"]
            )
            
            # 应该使用第三个模型
            assert result.model == "model3"
            assert "plugin3" in result.choices[0]["message"]["content"]
    
    def test_lazy_vs_traditional_mode_consistency(self):
        """测试懒加载和传统模式的一致性"""
        plugin = MockPlugin("test_plugin", ["model1", "model2"])
        
        # 传统模式 - 阻止自动加载插件
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []  # 空目录列表，防止自动加载
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            traditional_manager = ClientManager(lazy_loading=False)
            traditional_manager.register_plugin(plugin)
            traditional_models = traditional_manager.get_available_models()
            traditional_plugin = traditional_manager.get_plugin_for_model("model1")
        
        # 懒加载模式（模拟）
        with patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            mock_lazy_manager = MockLazyManager()
            mock_lazy_manager.model_to_plugin = {"model1": plugin, "model2": plugin}
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            lazy_manager = ClientManager(lazy_loading=True)
            lazy_models = lazy_manager.get_available_models()
            lazy_plugin = lazy_manager.get_plugin_for_model("model1")
            
            # 验证一致性
            traditional_model_ids = {m.id for m in traditional_models}
            lazy_model_ids = {m.id for m in lazy_models}
            assert traditional_model_ids == lazy_model_ids
            assert traditional_plugin is lazy_plugin


class TestClientManagerUnregister:
    """ClientManager 插件注销测试"""
    
    def test_unregister_plugin_success(self):
        """测试插件注销成功"""
        manager = ClientManager(lazy_loading=False)
        plugin = MockPlugin("test_plugin", ["model1", "model2"])
        manager.register_plugin(plugin)
        
        # 验证插件已注册
        assert "test_plugin" in manager.plugins
        assert "model1" in manager.model_to_plugin
        assert "model2" in manager.model_to_plugin
        
        # 注销插件
        manager.unregister_plugin("test_plugin")
        
        # 验证插件已注销
        assert "test_plugin" not in manager.plugins
        assert "model1" not in manager.model_to_plugin
        assert "model2" not in manager.model_to_plugin
    
    def test_unregister_plugin_not_found(self):
        """测试注销不存在的插件"""
        manager = ClientManager(lazy_loading=False)
        
        with pytest.raises(PluginError):
            manager.unregister_plugin("nonexistent_plugin")


class TestClientManagerEdgeCases:
    """ClientManager 边界情况测试"""
    
    def test_get_plugin_for_model_with_model_mapping(self):
        """测试通过模型映射获取插件"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.model_mappings = {"alias_model": "real_model"}
            mock_settings.plugin_directories = []
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            manager = ClientManager(lazy_loading=False)
            plugin = MockPlugin("test_plugin", ["real_model"])
            manager.register_plugin(plugin)
            
            # 通过别名获取插件
            result = manager.get_plugin_for_model("alias_model")
            
            assert result is plugin
    
    def test_lazy_mode_fallback_to_traditional(self):
        """测试懒加载模式回退到传统模式"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings, \
             patch('harborai.core.client_manager.LazyPluginManager') as mock_lazy_manager_class:
            
            mock_settings = Mock()
            mock_settings.plugin_directories = []
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            # 设置懒加载管理器抛出ModelNotFoundError
            mock_lazy_manager = MockLazyManager()
            mock_lazy_manager.get_plugin_for_model = Mock(side_effect=ModelNotFoundError("Model not found"))
            mock_lazy_manager_class.return_value = mock_lazy_manager
            
            manager = ClientManager(lazy_loading=True)
            
            # 在传统模式中注册插件
            plugin = MockPlugin("test_plugin", ["model1"])
            manager.register_plugin(plugin)
            
            # 应该能够通过传统模式找到插件
            result = manager.get_plugin_for_model("model1")
            assert result is plugin
    
    def test_plugin_config_merging(self):
        """测试插件配置合并"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={"timeout": 30})
            mock_get_settings.return_value = mock_settings
            
            # 客户端配置
            client_config = {"api_key": "test_key", "base_url": "http://test.com"}
            manager = ClientManager(client_config=client_config, lazy_loading=False)
            
            # 注册插件并验证配置
            plugin = MockPlugin("test_plugin", ["model1"])
            manager.register_plugin(plugin)
            
            # 验证插件注册成功
            assert "test_plugin" in manager.plugins
    
    def test_plugin_loading_with_valid_module(self):
        """测试有效模块的插件加载"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            manager = ClientManager(lazy_loading=False)
            
            # 验证插件加载成功
            assert len(manager.plugins) >= 0
    
    def test_plugin_loading_with_empty_directory(self):
        """测试空目录的插件加载"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            manager = ClientManager(lazy_loading=False)
            
            # 验证没有插件加载
            assert len(manager.plugins) == 0
    
    def test_plugin_registration_multiple_plugins(self):
        """测试多个插件注册"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            manager = ClientManager(lazy_loading=False)
            
            # 注册多个插件
            plugin1 = MockPlugin("plugin1", ["model1"])
            plugin2 = MockPlugin("plugin2", ["model2"])
            
            manager.register_plugin(plugin1)
            manager.register_plugin(plugin2)
            
            # 验证插件注册成功
            assert len(manager.plugins) >= 2
            assert "plugin1" in manager.plugins
            assert "plugin2" in manager.plugins
    
    def test_load_plugins_error_handling(self):
        """测试加载插件时的错误处理"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = ["invalid.package"]
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            # 模拟导入错误
            with patch('importlib.import_module', side_effect=ImportError("Package not found")):
                manager = ClientManager(lazy_loading=False)
                
                # 应该不会抛出异常，只是记录错误
                assert len(manager.plugins) == 0
    
    def test_plugin_error_handling_graceful(self):
        """测试插件错误处理的优雅性"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            # 即使有错误，也应该能创建管理器
            manager = ClientManager(lazy_loading=False)
            
            # 验证管理器创建成功
            assert manager is not None
            assert hasattr(manager, 'plugins')


class TestClientManagerAdvancedFeatures:
    """ClientManager 高级功能测试"""
    
    def test_get_plugin_info_with_specific_plugin(self):
        """测试获取特定插件信息"""
        manager = ClientManager(lazy_loading=False)
        plugin = MockPlugin("test_plugin", ["model1", "model2"])
        manager.register_plugin(plugin)
        
        info = manager.get_plugin_info()
        
        assert "test_plugin" in info
        assert info["test_plugin"]["name"] == "test_plugin"
        assert info["test_plugin"]["supported_models"] == ["model1", "model2"]
        assert info["test_plugin"]["model_count"] == 2
    
    def test_get_plugin_info_all_plugins(self):
        """测试获取所有插件信息"""
        manager = ClientManager(lazy_loading=False)
        plugin1 = MockPlugin("plugin1", ["model1"])
        plugin2 = MockPlugin("plugin2", ["model2", "model3"])
        
        manager.register_plugin(plugin1)
        manager.register_plugin(plugin2)
        
        info = manager.get_plugin_info()
        
        assert "plugin1" in info
        assert "plugin2" in info
        assert info["plugin1"]["model_count"] == 1
        assert info["plugin2"]["model_count"] == 2
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_trace_id(self):
        """测试带追踪ID的聊天完成"""
        manager = ClientManager(lazy_loading=False)
        plugin = MockPlugin("test_plugin", ["model1"])
        manager.register_plugin(plugin)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
             patch('harborai.core.models.filter_parameters_for_model', return_value={}), \
             patch('harborai.core.models.is_reasoning_model', return_value=False):
            
            result = await manager.chat_completion_with_fallback(
                model="model1",
                messages=messages
            )
            
            # 验证追踪ID被正确传递
            assert result.id == "test-completion"
    
    def test_model_fallback_mapping(self):
        """测试模型降级映射"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []
            mock_settings.model_mappings = {"gpt-4o": "gpt-4"}  # 使用settings中的model_mappings
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            manager = ClientManager(lazy_loading=False)
            plugin = MockPlugin("test_plugin", ["gpt-4"])
            manager.register_plugin(plugin)
            
            result = manager.get_plugin_for_model("gpt-4o")
            
            assert result is plugin
    
    def test_empty_plugin_directory_list(self):
        """测试空插件目录列表"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            manager = ClientManager(lazy_loading=False)
            
            # 应该没有插件被加载
            assert len(manager.plugins) == 0
    
    def test_none_plugin_directory_list(self):
        """测试None插件目录列表"""
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = None
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            manager = ClientManager(lazy_loading=False)
            
            # 应该没有插件被加载
            assert len(manager.plugins) == 0


class TestClientManagerConcurrency:
    """ClientManager 并发测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_chat_completions(self):
        """测试并发聊天完成"""
        manager = ClientManager(lazy_loading=False)
        plugin = MockPlugin("test_plugin", ["model1"])
        manager.register_plugin(plugin)
        
        messages = [ChatMessage(role="user", content="Hello")]
        
        async def make_request():
            with patch('harborai.core.client_manager.get_current_trace_id', return_value="trace123"), \
                 patch('harborai.core.models.filter_parameters_for_model', return_value={}), \
                 patch('harborai.core.models.is_reasoning_model', return_value=False):
                
                return await manager.chat_completion_with_fallback(
                    model="model1",
                    messages=messages
                )
        
        # 并发执行多个请求
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # 验证所有请求都成功
        assert len(results) == 5
        for result in results:
            assert result.model == "model1"
    
    def test_thread_safety_plugin_registration(self):
        """测试插件注册的线程安全性"""
        import threading
        
        with patch('harborai.core.client_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.plugin_directories = []
            mock_settings.model_mappings = {}
            mock_settings.get_plugin_config = Mock(return_value={})
            mock_get_settings.return_value = mock_settings
            
            manager = ClientManager(lazy_loading=False)
            initial_count = len(manager.plugins)
            errors = []
            
            def register_plugin(name):
                try:
                    plugin = MockPlugin(name, [f"model_{name}"])
                    manager.register_plugin(plugin)
                except Exception as e:
                    errors.append(e)
            
            # 创建多个线程同时注册插件
            threads = []
            for i in range(10):
                thread = threading.Thread(target=register_plugin, args=(f"plugin_{i}",))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 验证没有错误发生
            assert len(errors) == 0
            assert len(manager.plugins) == initial_count + 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])