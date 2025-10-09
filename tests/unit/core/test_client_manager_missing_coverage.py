"""
client_manager.py 缺失覆盖率测试用例

目标：补充缺失的测试用例以达到90%覆盖率
主要测试：
- 插件加载异常处理
- 模型回退机制
- 错误处理边界条件
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from harborai.core.client_manager import ClientManager
from harborai.core.base_plugin import BaseLLMPlugin, ChatMessage, ChatCompletion, ModelInfo, ChatChoice
from harborai.utils.exceptions import PluginError, ModelNotFoundError
from typing import List
import time


class MockPlugin(BaseLLMPlugin):
    """模拟插件用于测试"""
    
    def __init__(self, name: str, models: list, should_fail: bool = False):
        super().__init__(name)
        self.models = models
        self.should_fail = should_fail
        # 设置支持的模型
        self._supported_models = [ModelInfo(id=model, name=model, provider=self.name) for model in models]
    
    def get_supported_models(self) -> List[ModelInfo]:
        return self._supported_models
    
    def chat_completion(self, model: str, messages: List[ChatMessage], **kwargs) -> ChatCompletion:
        """模拟聊天完成"""
        if self.should_fail:
            raise PluginError(self.name, "模拟插件错误")
        return ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content="测试响应"))],
            usage=None
        )
    
    async def chat_completion_async(self, model: str, messages: List[ChatMessage], **kwargs) -> ChatCompletion:
        """模拟异步聊天完成"""
        if self.should_fail:
            raise PluginError(self.name, "模拟插件错误")
        return ChatCompletion(
            id="test-id",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[ChatChoice(index=0, message=ChatMessage(role="assistant", content="测试响应"))],
            usage=None
        )


class TestClientManagerMissingCoverage:
    """测试 ClientManager 缺失的覆盖率"""

    @pytest.fixture
    def mock_logger(self):
        """模拟日志记录器"""
        return Mock()

    @pytest.fixture
    def client_manager(self, mock_logger):
        """创建 ClientManager 实例"""
        with patch('harborai.core.client_manager.get_logger', return_value=mock_logger), \
             patch('harborai.core.client_manager.get_settings') as mock_settings:
            mock_settings.return_value.plugin_directories = []
            mock_settings.return_value.model_mappings = {}
            mock_settings.return_value.get_plugin_config = Mock(return_value={})
            
            manager = ClientManager(lazy_loading=False)
            manager.logger = mock_logger
            return manager

    def test_plugin_directory_import_error(self, client_manager):
        """测试插件目录导入错误 - 覆盖行 107-109"""
        with patch('harborai.core.client_manager.importlib.import_module') as mock_import:
            # 模拟导入包失败
            mock_import.side_effect = ImportError("No module named '/fake/path'")
            
            # 调用扫描方法，应该捕获异常并记录警告
            client_manager._scan_plugin_directory("/fake/path")
            
            # 验证警告被记录
            client_manager.logger.warning.assert_called()
            warning_call = client_manager.logger.warning.call_args[0][0]
            assert "Plugin directory not found" in warning_call

    def test_plugin_module_loading_exception(self, client_manager):
        """测试插件模块加载异常处理"""
        # 模拟logger.warning方法以避免关键字参数问题
        with patch.object(client_manager.logger, 'warning') as mock_warning:
            # 模拟插件模块加载失败
            with patch('importlib.import_module') as mock_import:
                mock_import.side_effect = ImportError("模块加载失败")
                
                # 调用 _scan_plugin_directory，这会触发ImportError
                client_manager._scan_plugin_directory("/fake/path")
            
            # 验证警告被记录
            mock_warning.assert_called()
            # 检查调用参数中是否包含预期的警告消息
            call_args = mock_warning.call_args
            if call_args:
                # call_args[0] 是位置参数，call_args[1] 是关键字参数
                args, kwargs = call_args
                # 检查第一个参数（消息）
                assert "Plugin directory not found" in args[0]

    def test_fallback_models_exhausted_sync(self, client_manager):
        """测试同步模式下所有fallback模型都失败"""
        # 注册一个会失败的插件
        failing_plugin = MockPlugin("test_plugin", ["test-model"], should_fail=True)
        client_manager.register_plugin(failing_plugin)
        
        # 测试当所有模型都失败时抛出异常
        with pytest.raises(PluginError, match="模拟插件错误"):
            client_manager.chat_completion_sync_with_fallback(
                model="test-model",
                messages=[ChatMessage(role="user", content="test")]
            )

    def test_model_fallback_warning_sync(self, client_manager):
        """测试同步模型回退警告"""
        # 注册两个插件，第一个失败，第二个成功
        failing_plugin = MockPlugin("failing_plugin", ["model1"], should_fail=True)
        success_plugin = MockPlugin("success_plugin", ["model2"], should_fail=False)
        
        client_manager.register_plugin(failing_plugin)
        client_manager.register_plugin(success_plugin)
        
        messages = [ChatMessage(role="user", content="test")]
        
        with patch('harborai.core.client_manager.get_current_trace_id', return_value="test-trace"):
            result = client_manager.chat_completion_sync_with_fallback(
                model="model1",
                messages=messages,
                fallback=["model2"]
            )
        
        # 验证成功返回结果
        assert result.model == "model2"
        
        # 验证警告日志被记录
        client_manager.logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_async_fallback_models_exhausted(self, client_manager):
        """测试异步模式下所有fallback模型都失败"""
        # 注册一个会失败的插件
        failing_plugin = MockPlugin("test_plugin", ["test-model"], should_fail=True)
        client_manager.register_plugin(failing_plugin)
        
        # 测试当所有模型都失败时抛出异常
        with pytest.raises(PluginError, match="模拟插件错误"):
            await client_manager.chat_completion_with_fallback(
                model="test-model",
                messages=[ChatMessage(role="user", content="test")]
            )

    def test_reasoning_model_message_processing(self, client_manager):
        """测试推理模型消息处理"""
        # 注册插件
        plugin = MockPlugin("test_plugin", ["o1-preview"])
        client_manager.register_plugin(plugin)
        
        messages = [
            ChatMessage(role="system", content="You are helpful"),
            ChatMessage(role="user", content="Hello")
        ]
        
        with patch('harborai.core.models.is_reasoning_model', return_value=True):
            with patch('harborai.core.client_manager.get_current_trace_id', return_value="test-trace"):
                processed = client_manager._process_messages_for_reasoning_model(messages)
                
                # 推理模型应该只有一个user消息
                assert len(processed) == 1
                assert processed[0].role == "user"
                assert "请按照以下指导原则回答：You are helpful" in processed[0].content
                assert "Hello" in processed[0].content

    def test_get_plugin_for_model_not_found(self, client_manager):
        """测试获取不存在模型的插件"""
        with pytest.raises(ModelNotFoundError):
            client_manager.get_plugin_for_model("nonexistent_model")

    def test_unregister_nonexistent_plugin(self, client_manager):
        """测试注销不存在的插件"""
        with pytest.raises(PluginError):
            client_manager.unregister_plugin("nonexistent_plugin")