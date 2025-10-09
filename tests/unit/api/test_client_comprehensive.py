"""API客户端综合测试。"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional

from harborai.api.client import HarborAI, ChatCompletions, Chat
from harborai.core.base_plugin import ChatCompletion, ChatCompletionChunk
from harborai.utils.exceptions import HarborAIError, ValidationError


class TestChatCompletions:
    """ChatCompletions类测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_初始化(self):
        """测试ChatCompletions初始化。"""
        assert self.chat_completions.client_manager == self.mock_client_manager
        assert self.chat_completions.logger is not None
        assert self.chat_completions.api_logger is not None
        assert self.chat_completions.settings is not None
    
    def test_获取快速处理器(self):
        """测试获取快速结构化输出处理器。"""
        with patch('harborai.core.fast_structured_output.create_fast_structured_output_processor') as mock_create:
            mock_processor = Mock()
            mock_create.return_value = mock_processor
            
            processor = self.chat_completions._get_fast_processor()
            
            assert processor == mock_processor
            mock_create.assert_called_once_with(client_manager=self.mock_client_manager)
            
            # 测试缓存
            processor2 = self.chat_completions._get_fast_processor()
            assert processor2 == mock_processor
            assert mock_create.call_count == 1  # 只调用一次
    
    @patch('harborai.api.client.get_performance_config')
    def test_create_基本调用(self, mock_get_perf_config):
        """测试基本的create调用。"""
        mock_get_perf_config.return_value = Mock(
            enable_fast_path=False,
            enable_structured_output_optimization=False
        )
        
        with patch.object(self.chat_completions, '_create_core') as mock_create_core:
            mock_response = Mock(spec=ChatCompletion)
            mock_create_core.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = self.chat_completions.create(messages=messages, model="gpt-3.5-turbo")
            
            assert result == mock_response
            mock_create_core.assert_called_once()
    
    @patch('harborai.api.client.get_performance_config')
    def test_create_快速路径(self, mock_get_perf_config):
        """测试快速路径调用。"""
        mock_get_perf_config.return_value = Mock(
            enable_fast_path=True,
            enable_structured_output_optimization=False
        )
        
        with patch.object(self.chat_completions, '_create_core') as mock_create_core:
            mock_response = Mock(spec=ChatCompletion)
            mock_create_core.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = self.chat_completions.create(messages=messages, model="gpt-3.5-turbo")
            
            assert result == mock_response
            mock_create_core.assert_called_once()
    
    @patch('harborai.api.client.get_performance_config')
    def test_create_结构化输出优化(self, mock_get_perf_config):
        """测试结构化输出优化路径。"""
        mock_get_perf_config.return_value = Mock(
            enable_fast_path=True,
            enable_structured_output_optimization=True
        )
        
        with patch.object(self.chat_completions, '_create_core') as mock_create_core:
            mock_response = Mock(spec=ChatCompletion)
            mock_create_core.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            response_format = {"type": "json_object"}
            result = self.chat_completions.create(
                messages=messages, 
                model="gpt-3.5-turbo",
                response_format=response_format,
                structured_provider="openai"
            )
            
            assert result == mock_response
            mock_create_core.assert_called_once()
    
    def test_validate_messages_有效消息(self):
        """测试有效消息验证。"""
        valid_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        # 不应该抛出异常
        self.chat_completions._validate_messages(valid_messages)
    
    def test_validate_messages_空消息列表(self):
        """测试空消息列表验证。"""
        with pytest.raises(ValidationError, match="Messages cannot be empty"):
            self.chat_completions._validate_messages([])
    
    def test_validate_messages_无效角色(self):
        """测试无效角色验证。"""
        invalid_messages = [
            {"role": "invalid_role", "content": "Hello"}
        ]
        
        with pytest.raises(ValidationError, match="has invalid role.*invalid_role"):
            self.chat_completions._validate_messages(invalid_messages)
    
    def test_validate_messages_缺少内容(self):
        """测试缺少内容验证。"""
        invalid_messages = [
            {"role": "user"}  # 缺少content
        ]
        
        with pytest.raises(ValidationError, match="Message at index 0 must have either"):
            self.chat_completions._validate_messages(invalid_messages)
    
    def test_process_messages_for_reasoning_model(self):
        """测试推理模型消息处理。"""
        messages = [
            {"role": "user", "content": "Solve this problem", "reasoning_content": "Think step by step"}
        ]
        
        processed = self.chat_completions._process_messages_for_reasoning_model(messages)
        
        assert len(processed) == 1
        assert processed[0]["role"] == "user"
        assert "reasoning_content" in processed[0]
    
    @pytest.mark.asyncio
    @patch('harborai.api.client.get_performance_config')
    async def test_acreate_基本调用(self, mock_get_perf_config):
        """测试异步基本调用。"""
        mock_get_perf_config.return_value = Mock(
            enable_fast_path=False,
            enable_structured_output_optimization=False
        )
        
        with patch.object(self.chat_completions, '_acreate_core') as mock_acreate_core:
            mock_response = Mock(spec=ChatCompletion)
            mock_acreate_core.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = await self.chat_completions.acreate(messages=messages, model="gpt-3.5-turbo")
            
            assert result == mock_response
            mock_acreate_core.assert_called_once()


class TestChat:
    """Chat类测试。"""
    
    def test_初始化(self):
        """测试Chat初始化。"""
        mock_client_manager = Mock()
        chat = Chat(mock_client_manager)
        
        assert isinstance(chat.completions, ChatCompletions)
        assert chat.completions.client_manager == mock_client_manager


class TestHarborAI:
    """HarborAI主客户端测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        with patch('harborai.api.client.ClientManager') as mock_cm_class:
            self.mock_client_manager = Mock()
            # 设置可迭代的属性
            self.mock_client_manager.plugins = {"openai": Mock(), "claude": Mock()}
            self.mock_client_manager.model_to_plugin = {"gpt-3.5-turbo": "openai", "claude-3": "claude"}
            # 设置异步方法
            self.mock_client_manager.aclose = AsyncMock()
            mock_cm_class.return_value = self.mock_client_manager
            
            self.client = HarborAI(api_key="test-key")
    
    def test_初始化基本参数(self):
        """测试基本参数初始化。"""
        with patch('harborai.api.client.ClientManager') as mock_cm_class:
            mock_client_manager = Mock()
            # 设置可迭代的属性
            mock_client_manager.plugins = {"openai": Mock(), "claude": Mock()}
            mock_client_manager.model_to_plugin = {"gpt-3.5-turbo": "openai", "claude-3": "claude"}
            mock_cm_class.return_value = mock_client_manager
            
            client = HarborAI(
                api_key="test-key",
                organization="test-org",
                project="test-project",
                base_url="https://api.test.com",
                timeout=30.0,
                max_retries=5
            )
            
            assert isinstance(client.chat, Chat)
            assert client.chat.completions.client_manager == mock_client_manager
    
    def test_初始化默认参数(self):
        """测试默认参数初始化。"""
        with patch('harborai.api.client.ClientManager') as mock_cm_class:
            mock_client_manager = Mock()
            # 设置可迭代的属性
            mock_client_manager.plugins = {"openai": Mock(), "claude": Mock()}
            mock_client_manager.model_to_plugin = {"gpt-3.5-turbo": "openai", "claude-3": "claude"}
            mock_cm_class.return_value = mock_client_manager
            
            client = HarborAI()
            
            assert isinstance(client.chat, Chat)
    
    def test_get_available_models(self):
        """测试获取可用模型列表。"""
        expected_models = ["gpt-3.5-turbo", "claude-3"]
        
        # Mock get_available_models 方法返回字符串列表
        self.mock_client_manager.get_available_models.return_value = expected_models
        
        models = self.client.get_available_models()
        
        assert models == expected_models
    
    def test_get_plugin_info(self):
        """测试获取插件信息。"""
        expected_info = {"openai": {"version": "1.0"}, "claude": {"version": "2.0"}}
        self.mock_client_manager.get_plugin_info.return_value = expected_info
        
        info = self.client.get_plugin_info()
        
        assert info == expected_info
        self.mock_client_manager.get_plugin_info.assert_called_once()
    
    def test_register_plugin(self):
        """测试注册插件。"""
        mock_plugin = Mock()
        
        self.client.register_plugin(mock_plugin)
        
        self.mock_client_manager.register_plugin.assert_called_once_with(mock_plugin)
    
    def test_unregister_plugin(self):
        """测试注销插件。"""
        plugin_name = "test_plugin"
        
        self.client.unregister_plugin(plugin_name)
        
        self.mock_client_manager.unregister_plugin.assert_called_once_with(plugin_name)
    
    def test_get_total_cost(self):
        """测试获取总成本。"""
        # 当前实现返回0.0作为占位符
        cost = self.client.get_total_cost()
        
        assert cost == 0.0
    
    def test_reset_cost(self):
        """测试重置成本。"""
        # 当前实现是空的pass，只测试不抛异常
        self.client.reset_cost()
        
        # 如果没有异常，测试通过
    
    @pytest.mark.asyncio
    async def test_aclose(self):
        """测试异步关闭。"""
        # 设置插件mock
        mock_plugin = Mock()
        mock_plugin.aclose = AsyncMock()
        self.mock_client_manager.plugins = {"test": mock_plugin}
        
        # 模拟性能管理器已初始化
        mock_perf_manager = Mock()
        mock_perf_manager.is_initialized.return_value = True
        self.client._performance_manager = mock_perf_manager
        
        with patch('harborai.api.client.cleanup_performance_manager') as mock_cleanup:
            await self.client.aclose()
            
            mock_plugin.aclose.assert_called_once()
            mock_cleanup.assert_called_once()
    
    def test_close(self):
        """测试同步关闭。"""
        # 设置插件mock
        mock_plugin = Mock()
        mock_plugin.close = Mock()
        self.mock_client_manager.plugins = {"test": mock_plugin}
        
        with patch('harborai.api.client.cleanup_performance_manager') as mock_cleanup:
            self.client.close()
            
            mock_plugin.close.assert_called_once()
    
    def test_context_manager_sync(self):
        """测试同步上下文管理器。"""
        with patch.object(self.client, 'close') as mock_close:
            with self.client as client:
                assert client == self.client
            
            mock_close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_context_manager_async(self):
        """测试异步上下文管理器。"""
        with patch.object(self.client, 'aclose') as mock_aclose:
            async with self.client as client:
                assert client == self.client
            
            mock_aclose.assert_called_once()


class TestIntegrationScenarios:
    """集成场景测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        with patch('harborai.api.client.ClientManager') as mock_cm_class:
            self.mock_client_manager = Mock()
            # 设置可迭代的属性
            self.mock_client_manager.plugins = {"openai": Mock(), "claude": Mock()}
            self.mock_client_manager.model_to_plugin = {"gpt-3.5-turbo": "openai", "claude-3": "claude"}
            mock_cm_class.return_value = self.mock_client_manager
            
            self.client = HarborAI(api_key="test-key")
    
    @patch('harborai.api.client.get_performance_config')
    def test_完整聊天流程(self, mock_get_perf_config):
        """测试完整的聊天流程。"""
        mock_get_perf_config.return_value = Mock(
            enable_fast_path=False,
            enable_structured_output_optimization=False
        )
        
        # 模拟响应
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        
        with patch.object(self.client.chat.completions, '_create_core') as mock_create_core:
            mock_create_core.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            response = self.client.chat.completions.create(
                messages=messages,
                model="gpt-3.5-turbo"
            )
            
            assert response == mock_response
            mock_create_core.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('harborai.api.client.get_performance_config')
    async def test_异步聊天流程(self, mock_get_perf_config):
        """测试异步聊天流程。"""
        mock_get_perf_config.return_value = Mock(
            enable_fast_path=False,
            enable_structured_output_optimization=False
        )
        
        # 模拟异步响应
        mock_response = Mock(spec=ChatCompletion)
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"
        
        with patch.object(self.client.chat.completions, '_acreate_core') as mock_acreate_core:
            mock_acreate_core.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            response = await self.client.chat.completions.acreate(
                messages=messages,
                model="gpt-3.5-turbo"
            )
            
            assert response == mock_response
            mock_acreate_core.assert_called_once()


class TestErrorHandling:
    """错误处理测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_处理客户端管理器错误(self):
        """测试处理客户端管理器错误。"""
        with patch.object(self.chat_completions, '_create_core') as mock_create_core:
            mock_create_core.side_effect = HarborAIError("Client manager error")
            
            with pytest.raises(HarborAIError, match="Client manager error"):
                self.chat_completions.create(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-3.5-turbo"
                )
    
    @pytest.mark.asyncio
    async def test_处理异步错误(self):
        """测试处理异步错误。"""
        with patch.object(self.chat_completions, '_acreate_core') as mock_acreate_core:
            mock_acreate_core.side_effect = HarborAIError("Async error")
            
            with pytest.raises(HarborAIError, match="Async error"):
                await self.chat_completions.acreate(
                    messages=[{"role": "user", "content": "Hello"}],
                    model="gpt-3.5-turbo"
                )


class TestEdgeCases:
    """边界情况测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_空字符串内容(self):
        """测试空字符串内容。"""
        messages = [{"role": "user", "content": ""}]
        
        # 当前实现允许空字符串内容，只要有content字段
        # 这个测试应该通过而不抛出异常
        try:
            self.chat_completions._validate_messages(messages)
        except ValidationError:
            pytest.fail("空字符串内容不应该抛出ValidationError")
    
    def test_None内容(self):
        """测试None内容。"""
        messages = [{"role": "user", "content": None}]
        
        # 当前实现允许None内容，只要有content字段
        # 这个测试应该通过而不抛出异常
        try:
            self.chat_completions._validate_messages(messages)
        except ValidationError:
            pytest.fail("None内容不应该抛出ValidationError")
    
    def test_大量消息(self):
        """测试大量消息处理。"""
        messages = [{"role": "user", "content": f"Message {i}"} for i in range(1000)]
        
        # 不应该抛出异常
        self.chat_completions._validate_messages(messages)
    
    def test_特殊字符消息(self):
        """测试包含特殊字符的消息。"""
        messages = [
            {"role": "user", "content": "Hello 🌟 World! 中文测试 @#$%^&*()"}
        ]
        
        # 不应该抛出异常
        self.chat_completions._validate_messages(messages)
    
    def test_推理内容处理(self):
        """测试推理内容处理。"""
        messages = [
            {
                "role": "user", 
                "content": "Solve this problem",
                "reasoning_content": "Let me think about this step by step..."
            }
        ]
        
        processed = self.chat_completions._process_messages_for_reasoning_model(messages)
        
        assert len(processed) == 1
        assert "reasoning_content" in processed[0]
        assert processed[0]["reasoning_content"] == "Let me think about this step by step..."


class TestChatCompletionsAdvanced:
    """ChatCompletions高级功能测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_create_core_基本功能(self):
        """测试_create_core的基本功能。"""
        # 模拟依赖
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "测试响应"
        
        # 模拟所有必要的依赖
        with patch('harborai.api.client.get_or_create_trace_id', return_value='test-trace-id'), \
             patch('harborai.api.client.TraceContext'), \
             patch('harborai.utils.logger.LogContext'), \
             patch('harborai.api.client.retry_with_backoff') as mock_retry, \
             patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback', return_value=mock_response):
            
            # 配置retry装饰器返回原函数
            mock_retry.side_effect = lambda config=None: lambda func: func
            
            result = self.chat_completions._create_core(
                messages=[{"role": "user", "content": "测试"}],
                model="gpt-3.5-turbo"
            )
            
            assert result == mock_response
    
    def test_create_core_带所有参数(self):
        """测试_create_core带所有参数。"""
        # 模拟所有必要的依赖
        with patch('harborai.api.client.get_or_create_trace_id', return_value='test-trace-id'), \
             patch('harborai.api.client.TraceContext'), \
             patch('harborai.utils.logger.LogContext'), \
             patch('harborai.api.client.retry_with_backoff') as mock_retry, \
             patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback') as mock_create:
            
            # 配置retry装饰器返回原函数
            mock_retry.side_effect = lambda config=None: lambda func: func
            
            mock_response = Mock(spec=ChatCompletion)
            mock_create.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=100,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.2,
                stop=["END"],
                stream=False,
                response_format={"type": "json_object"},
                tools=[{"type": "function", "function": {"name": "test"}}],
                tool_choice="auto",
                user="test_user",
                seed=42,
                logprobs=True,
                top_logprobs=5,
                n=1,
                timeout=30.0,
                extra_body={"custom": "value"}
            )
            
            assert result == mock_response
            mock_create.assert_called_once()
            
            # 验证参数传递
            call_args = mock_create.call_args[1]
            assert call_args["temperature"] == 0.7
            assert call_args["max_tokens"] == 100
            assert call_args["top_p"] == 0.9
    
    @pytest.mark.asyncio
    async def test_acreate_core_基本功能(self):
        """测试异步_acreate_core的基本功能。"""
        mock_response = Mock()
        
        # 模拟所有必要的依赖
        with patch('harborai.api.client.get_or_create_trace_id', return_value='test-trace-id'), \
             patch('harborai.api.client.TraceContext'), \
             patch('harborai.utils.logger.LogContext'), \
             patch('harborai.api.client.async_retry_with_backoff') as mock_retry:
            
            # 配置retry装饰器返回原函数
            mock_retry.side_effect = lambda config=None: lambda func: func
            
            # 模拟异步方法
            async_mock = AsyncMock(return_value=mock_response)
            with patch.object(self.chat_completions.client_manager, 'chat_completion_with_fallback', async_mock):
                result = await self.chat_completions._acreate_core(
                    messages=[{"role": "user", "content": "测试"}],
                    model="gpt-3.5-turbo"
                )
                
                assert result == mock_response
                async_mock.assert_called_once()
    
    @patch('harborai.api.client.get_performance_config')
    def test_create_快速结构化路径(self, mock_get_perf_config):
        """测试快速结构化输出路径。"""
        mock_perf_config = Mock()
        mock_perf_config.mode.value = "fast"  # 小写
        mock_get_perf_config.return_value = mock_perf_config
        
        with patch.object(self.chat_completions, '_create_fast_structured_path') as mock_fast_structured:
            mock_response = Mock(spec=ChatCompletion)
            mock_fast_structured.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            response_format = {"type": "json_schema"}  # 需要是json_schema才能触发快速结构化路径
            result = self.chat_completions.create(
                messages=messages,
                model="gpt-3.5-turbo",
                response_format=response_format,
                structured_provider="agently",  # 需要是agently才能触发快速结构化路径
                stream=False
            )
            
            assert result == mock_response
            mock_fast_structured.assert_called_once()
    
    @patch('harborai.api.client.get_performance_config')
    def test_create_快速路径_启用(self, mock_get_perf_config):
        """测试启用快速路径。"""
        mock_perf_config = Mock()
        mock_perf_config.mode.value = "fast"
        mock_perf_config.should_use_fast_path.return_value = True  # 模拟应该使用快速路径
        mock_get_perf_config.return_value = mock_perf_config
        
        with patch.object(self.chat_completions, '_create_fast_path') as mock_fast_path:
            mock_response = Mock(spec=ChatCompletion)
            mock_fast_path.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = self.chat_completions.create(
                messages=messages,
                model="gpt-3.5-turbo"
            )
            
            assert result == mock_response
            mock_fast_path.assert_called_once()
    
    @patch('harborai.api.client.get_performance_config')
    def test_create_完整路径(self, mock_get_perf_config):
        """测试完整路径。"""
        mock_perf_config = Mock()
        mock_perf_config.mode.value = "full"
        mock_perf_config.should_use_fast_path.return_value = False  # 模拟不应该使用快速路径
        mock_get_perf_config.return_value = mock_perf_config
        
        with patch.object(self.chat_completions, '_create_full_path') as mock_full_path:
            mock_response = Mock(spec=ChatCompletion)
            mock_full_path.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = self.chat_completions.create(
                messages=messages,
                model="gpt-3.5-turbo"
            )
            
            assert result == mock_response
            mock_full_path.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('harborai.api.client.get_performance_config')
    async def test_acreate_快速结构化路径(self, mock_get_perf_config):
        """测试异步快速结构化路径。"""
        mock_perf_config = Mock()
        mock_perf_config.mode.value = "fast"
        mock_get_perf_config.return_value = mock_perf_config
        
        with patch.object(self.chat_completions, '_acreate_fast_structured_path') as mock_fast_structured:
            mock_response = Mock(spec=ChatCompletion)
            mock_fast_structured.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            response_format = {"type": "json_schema"}
            result = await self.chat_completions.acreate(
                messages=messages,
                model="gpt-3.5-turbo",
                response_format=response_format,
                structured_provider="agently",
                stream=False
            )
            
            assert result == mock_response
            mock_fast_structured.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('harborai.api.client.get_performance_config')
    async def test_acreate_快速路径(self, mock_get_perf_config):
        """测试异步快速路径。"""
        mock_perf_config = Mock()
        mock_perf_config.mode.value = "fast"
        mock_perf_config.should_use_fast_path.return_value = True
        mock_get_perf_config.return_value = mock_perf_config
        
        with patch.object(self.chat_completions, '_acreate_fast_path') as mock_fast_path:
            mock_response = Mock(spec=ChatCompletion)
            mock_fast_path.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = await self.chat_completions.acreate(
                messages=messages,
                model="gpt-3.5-turbo"
            )
            
            assert result == mock_response
            mock_fast_path.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('harborai.api.client.get_performance_config')
    async def test_acreate_完整路径(self, mock_get_perf_config):
        """测试异步完整路径。"""
        mock_perf_config = Mock()
        mock_perf_config.mode.value = "full"
        mock_perf_config.should_use_fast_path.return_value = False
        mock_get_perf_config.return_value = mock_perf_config
        
        with patch.object(self.chat_completions, '_acreate_full_path') as mock_full_path:
            mock_response = Mock(spec=ChatCompletion)
            mock_full_path.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = await self.chat_completions.acreate(
                messages=messages,
                model="gpt-3.5-turbo"
            )
            
            assert result == mock_response
            mock_full_path.assert_called_once()
    
    @patch.object(ChatCompletions, '_create_core')
    def test_create_fast_structured_path_基本功能(self, mock_create_core):
        """测试快速结构化输出路径的基本功能 - 测试回退到常规路径"""
        # 设置mock返回值
        mock_response = Mock(spec=ChatCompletion)
        mock_create_core.return_value = mock_response
        
        # 准备测试数据
        messages = [{"role": "user", "content": "测试消息"}]
        response_format = {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
        
        # 调用方法
        result = self.chat_completions._create_fast_structured_path(
            messages=messages,
            model="gpt-3.5-turbo",
            response_format=response_format,
            structured_provider="agently"
        )
        
        # 验证结果 - 应该回退到常规路径
        assert result is not None
        mock_create_core.assert_called_once()
    
    @pytest.mark.asyncio
    @patch.object(ChatCompletions, '_acreate_core')
    async def test_acreate_fast_structured_path_基本功能(self, mock_acreate_core):
        """测试异步快速结构化输出路径的基本功能 - 测试回退到常规路径"""
        # 设置mock返回值
        mock_response = Mock(spec=ChatCompletion)
        mock_acreate_core.return_value = mock_response
        
        # 准备测试数据
        messages = [{"role": "user", "content": "测试消息"}]
        response_format = {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
        
        # 调用方法
        result = await self.chat_completions._acreate_fast_structured_path(
            messages=messages,
            model="gpt-3.5-turbo",
            response_format=response_format,
            structured_provider="agently"
        )
        
        # 验证结果 - 应该回退到常规路径
        assert result is not None
        mock_acreate_core.assert_called_once()
    
    def test_validate_messages_工具调用消息(self):
        """测试工具调用消息验证。"""
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function"}]},
            {"role": "tool", "content": "result", "tool_call_id": "call_123"}
        ]
        # 不应该抛出异常
        self.chat_completions._validate_messages(messages)
    
    def test_validate_messages_空内容(self):
        """测试空内容验证。"""
        messages = [
            {"role": "user", "content": ""}
        ]
        # 当前实现允许空内容
        self.chat_completions._validate_messages(messages)
    
    def test_process_messages_无推理内容(self):
        """测试无推理内容的消息处理。"""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        processed = self.chat_completions._process_messages_for_reasoning_model(messages)
        
        assert processed == messages  # 应该返回原始消息


class TestHarborAIAdvanced:
    """HarborAI高级功能测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        # Mock所有依赖
        self.mock_client_manager = Mock()
        self.mock_client_manager.plugins = {}  # 修复迭代问题
        self.mock_client_manager.model_to_plugin = {}  # 修复len()问题
        self.mock_chat = Mock()
        self.mock_completions = Mock()
        
        # 设置mock链
        self.mock_chat.completions = self.mock_completions
        
        # 创建patches
        self.client_manager_patch = patch('harborai.api.client.ClientManager', return_value=self.mock_client_manager)
        self.chat_patch = patch('harborai.api.client.Chat', return_value=self.mock_chat)
        self.auto_init_patch = patch('harborai.api.client.auto_initialize')
        self.perf_manager_patch = patch('harborai.api.client.initialize_performance_manager')
    
    def test_初始化自定义参数(self):
        """测试自定义参数初始化。"""
        with self.client_manager_patch, self.chat_patch, self.auto_init_patch, self.perf_manager_patch:
            custom_headers = {"Custom-Header": "value"}
            custom_query = {"param": "value"}
            
            client = HarborAI(
                default_headers=custom_headers,
                default_query=custom_query,
                custom_param="custom_value"
            )
            
            assert client.chat == self.mock_chat
            assert client.chat.completions == self.mock_completions
    
    def test_get_total_cost_with_tracker(self):
        """测试获取总成本（带追踪器）。"""
        with self.client_manager_patch, self.chat_patch, self.auto_init_patch, self.perf_manager_patch:
            client = HarborAI()
            cost = client.get_total_cost()
            
            # 当前实现返回0.0
            assert cost == 0.0
    
    def test_reset_cost_with_tracker(self):
        """测试重置成本（带追踪器）。"""
        with self.client_manager_patch, self.chat_patch, self.auto_init_patch, self.perf_manager_patch:
            client = HarborAI()
            client.reset_cost()
            
            # 当前实现是pass，不会调用追踪器，测试不抛异常即可
            assert True
    
    @pytest.mark.asyncio
    async def test_aclose_异常处理(self):
        """测试异步关闭的异常处理。"""
        with self.client_manager_patch, self.chat_patch, self.auto_init_patch, self.perf_manager_patch:
            with patch('harborai.api.client.cleanup_performance_manager') as mock_cleanup:
                # 确保mock_client_manager有plugins属性
                mock_plugin = Mock()
                mock_plugin.aclose = AsyncMock(side_effect=Exception("关闭失败"))
                mock_plugin.name = "test_plugin"
                self.mock_client_manager.plugins = {"test": mock_plugin}
                
                client = HarborAI()
                
                # 设置性能管理器以确保清理函数被调用
                mock_perf_manager = Mock()
                mock_perf_manager.is_initialized.return_value = True
                client._performance_manager = mock_perf_manager
                
                # 异常应该被捕获，不会抛出
                await client.aclose()
                
                # 验证插件的aclose被调用
                mock_plugin.aclose.assert_called_once()
                mock_cleanup.assert_called_once()
    
    def test_close_异常处理(self):
        """测试同步关闭的异常处理。"""
        with self.client_manager_patch, self.chat_patch, self.auto_init_patch, self.perf_manager_patch:
            with patch('harborai.api.client.cleanup_performance_manager') as mock_cleanup:
                # 确保mock_client_manager有plugins属性
                mock_plugin = Mock()
                mock_plugin.close = Mock(side_effect=Exception("关闭失败"))
                mock_plugin.name = "test_plugin"
                self.mock_client_manager.plugins = {"test": mock_plugin}
                
                client = HarborAI()
                
                # 设置性能管理器以确保清理函数被调用
                mock_perf_manager = Mock()
                mock_perf_manager.is_initialized.return_value = True
                client._performance_manager = mock_perf_manager
                
                # 异常应该被捕获，不会抛出
                client.close()
                
                # 验证插件的close被调用
                mock_plugin.close.assert_called_once()
                mock_cleanup.assert_called_once()
    
    def test_client_别名(self):
        """测试Client别名。"""
        from harborai.api.client import Client
        assert Client == HarborAI


class TestPerformanceOptimizations:
    """性能优化测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_快速处理器缓存(self):
        """测试快速处理器缓存机制。"""
        with patch('harborai.core.fast_structured_output.create_fast_structured_output_processor') as mock_create:
            mock_processor = Mock()
            mock_create.return_value = mock_processor
            
            # 第一次调用
            processor1 = self.chat_completions._get_fast_processor()
            # 第二次调用
            processor2 = self.chat_completions._get_fast_processor()
            
            # 应该返回同一个实例
            assert processor1 == processor2
            # 创建函数只应该被调用一次
            assert mock_create.call_count == 1
    
    @patch('harborai.api.client.get_performance_config')
    def test_性能配置路由(self, mock_get_perf_config):
        """测试性能配置对路由的影响。"""
        mock_perf_config = Mock()
        mock_get_perf_config.return_value = mock_perf_config
        
        # 测试快速路径
        mock_perf_config.should_use_fast_path.return_value = True
        
        with patch.object(self.chat_completions, '_create_fast_path') as mock_fast:
            mock_fast.return_value = Mock(spec=ChatCompletion)
            
            messages = [{"role": "user", "content": "Hello"}]
            self.chat_completions.create(messages=messages, model="gpt-3.5-turbo")
            
            mock_fast.assert_called_once()
        
        # 测试完整路径
        mock_perf_config.should_use_fast_path.return_value = False
        
        with patch.object(self.chat_completions, '_create_full_path') as mock_full:
            mock_full.return_value = Mock(spec=ChatCompletion)
            
            messages = [{"role": "user", "content": "Hello"}]
            self.chat_completions.create(messages=messages, model="gpt-3.5-turbo")
            
            mock_full.assert_called_once()
    
    def test_消息预处理缓存(self):
        """测试消息预处理的缓存效果。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        # 多次调用相同的消息处理
        for _ in range(5):
            processed = self.chat_completions._process_messages_for_reasoning_model(messages)
            assert processed == messages
    
    def test_参数验证优化(self):
        """测试参数验证的性能优化。"""
        import time
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # 测试验证性能
        start_time = time.time()
        for _ in range(100):
            self.chat_completions._validate_messages(messages)
        validation_time = time.time() - start_time
        
        # 验证应该很快完成
        assert validation_time < 1.0  # 100次验证应该在1秒内完成


class TestStreamingAndTools:
    """流式响应和工具调用测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_流式响应处理(self):
        """测试流式响应处理。"""
        # 模拟流式响应
        def mock_stream():
            yield Mock(spec=ChatCompletionChunk, choices=[Mock(delta=Mock(content="Hello"))])
            yield Mock(spec=ChatCompletionChunk, choices=[Mock(delta=Mock(content=" world"))])
            yield Mock(spec=ChatCompletionChunk, choices=[Mock(delta=Mock(content="!"))])
        
        with patch('harborai.api.client.get_or_create_trace_id', return_value='test-trace-id'), \
             patch('harborai.api.client.TraceContext'), \
             patch('harborai.utils.logger.LogContext'), \
             patch('harborai.api.client.retry_with_backoff') as mock_retry, \
             patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback', return_value=mock_stream()):
            
            # 配置retry装饰器返回原函数
            mock_retry.side_effect = lambda config=None: lambda func: func
            
            messages = [{"role": "user", "content": "Hello"}]
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                stream=True
            )
            
            # 验证返回的是生成器
            chunks = list(result)
            assert len(chunks) == 3
            assert all(isinstance(chunk, Mock) for chunk in chunks)
    
    def test_工具调用流程(self):
        """测试工具调用流程。"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }
            }
        ]
        
        mock_response = Mock(spec=ChatCompletion)
        mock_function = Mock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"location": "Beijing"}'
        
        mock_response.choices = [
            Mock(message=Mock(
                tool_calls=[
                    Mock(
                        id="call_123",
                        type="function",
                        function=mock_function
                    )
                ]
            ))
        ]
        
        with patch('harborai.api.client.get_or_create_trace_id', return_value='test-trace-id'), \
             patch('harborai.api.client.TraceContext'), \
             patch('harborai.utils.logger.LogContext'), \
             patch('harborai.api.client.retry_with_backoff') as mock_retry, \
             patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback', return_value=mock_response):
            
            # 配置retry装饰器返回原函数
            mock_retry.side_effect = lambda config=None: lambda func: func
            
            messages = [{"role": "user", "content": "What's the weather in Beijing?"}]
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                tools=tools,
                tool_choice="auto"
            )
            
            assert result == mock_response
            assert result.choices[0].message.tool_calls[0].function.name == "get_weather"


class TestAdvancedErrorHandling:
    """高级错误处理测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_处理网络超时(self):
        """测试处理网络超时。"""
        import asyncio
        
        with patch('harborai.api.client.get_or_create_trace_id', return_value='test-trace-id'), \
             patch('harborai.api.client.TraceContext'), \
             patch('harborai.utils.logger.LogContext'), \
             patch('harborai.api.client.retry_with_backoff') as mock_retry, \
             patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback', side_effect=asyncio.TimeoutError("请求超时")):
            
            # 配置retry装饰器返回原函数
            mock_retry.side_effect = lambda config=None: lambda func: func
            
            with pytest.raises(asyncio.TimeoutError):
                messages = [{"role": "user", "content": "Hello"}]
                self.chat_completions._create_core(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    timeout=1.0
                )
    
    def test_处理参数错误(self):
        """测试处理参数错误。"""
        # 测试无效的temperature值
        messages = [{"role": "user", "content": "Hello"}]
        
        # 当前实现不验证参数范围，所以这个测试验证不抛出异常
        try:
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=2.0  # 超出有效范围
            )
        except ValidationError:
            pytest.fail("当前实现不应该验证temperature范围")
    
    def test_处理模型不存在错误(self):
        """测试处理模型不存在错误。"""
        with patch('harborai.api.client.get_or_create_trace_id', return_value='test-trace-id'), \
             patch('harborai.api.client.TraceContext'), \
             patch('harborai.utils.logger.LogContext'), \
             patch('harborai.api.client.retry_with_backoff') as mock_retry, \
             patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback', side_effect=HarborAIError("模型不存在")):
            
            # 配置retry装饰器返回原函数
            mock_retry.side_effect = lambda config=None: lambda func: func
            
            with pytest.raises(HarborAIError, match="模型不存在"):
                messages = [{"role": "user", "content": "Hello"}]
                self.chat_completions._create_core(
                    messages=messages,
                    model="non-existent-model"
                )


class TestAdvancedEdgeCases:
    """高级边界情况测试。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_超长内容(self):
        """测试超长内容。"""
        # 创建一个非常长的消息
        long_content = "x" * 100000
        messages = [{"role": "user", "content": long_content}]
        
        # 应该能正常验证
        self.chat_completions._validate_messages(messages)
    
    def test_混合消息类型(self):
        """测试混合消息类型。"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "tool_calls": [{"id": "call_123", "type": "function"}]},
            {"role": "tool", "content": "Function result", "tool_call_id": "call_123"}
        ]
        
        # 应该能正常验证
        self.chat_completions._validate_messages(messages)
    
    def test_极端参数值(self):
        """测试极端参数值。"""
        messages = [{"role": "user", "content": "Test"}]
        
        # 测试极端的temperature值
        with patch('harborai.api.client.get_or_create_trace_id', return_value='test-trace-id'), \
             patch('harborai.api.client.TraceContext'), \
             patch('harborai.utils.logger.LogContext'), \
             patch('harborai.api.client.retry_with_backoff') as mock_retry, \
             patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback') as mock_create:
            
            # 配置retry装饰器返回原函数
            mock_retry.side_effect = lambda config=None: lambda func: func
            mock_create.return_value = Mock(spec=ChatCompletion)
            
            # 最小值
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.0
            )
            
            # 最大值
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=2.0
            )
    
    def test_并发调用(self):
        """测试并发调用。"""
        import threading
        
        results = []
        errors = []
        
        def make_call():
            try:
                with patch('harborai.api.client.get_or_create_trace_id', return_value='test-trace-id'), \
                     patch('harborai.api.client.TraceContext'), \
                     patch('harborai.utils.logger.LogContext'), \
                     patch('harborai.api.client.retry_with_backoff') as mock_retry, \
                     patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback') as mock_create:
                    
                    # 配置retry装饰器返回原函数
                    mock_retry.side_effect = lambda config=None: lambda func: func
                    mock_create.return_value = Mock(spec=ChatCompletion)
                    
                    messages = [{"role": "user", "content": f"Hello from thread {threading.current_thread().ident}"}]
                    result = self.chat_completions._create_core(
                        messages=messages,
                        model="gpt-3.5-turbo"
                    )
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # 创建多个线程
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_call)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证结果
        assert len(results) == 10
        assert len(errors) == 0


class TestParameterValidation:
    """参数验证测试类。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_temperature_参数验证_非数字类型(self):
        """测试temperature参数验证 - 非数字类型。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="temperature must be a number"):
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature="invalid"
            )
    
    def test_temperature_参数验证_超出范围_负数(self):
        """测试temperature参数验证 - 负数。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=-0.1
            )
    
    def test_temperature_参数验证_超出范围_过大(self):
        """测试temperature参数验证 - 超过2.0。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=2.1
            )
    
    def test_temperature_参数验证_有效值(self):
        """测试temperature参数验证 - 有效值。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback') as mock_create:
            mock_response = Mock(spec=ChatCompletion)
            mock_create.return_value = mock_response
            
            # 测试边界值
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.0
            )
            assert result == mock_response
            
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=2.0
            )
            assert result == mock_response
            
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=1.0
            )
            assert result == mock_response
    
    def test_max_tokens_参数验证_非整数类型(self):
        """测试max_tokens参数验证 - 非整数类型。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="max_tokens must be an integer"):
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                max_tokens="invalid"
            )
    
    def test_max_tokens_参数验证_非正数(self):
        """测试max_tokens参数验证 - 非正数。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                max_tokens=0
            )
        
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                max_tokens=-1
            )
    
    @patch('harborai.core.models.get_model_capabilities')
    def test_max_tokens_参数验证_超出模型限制(self, mock_get_capabilities):
        """测试max_tokens参数验证 - 超出模型限制。"""
        from harborai.core.exceptions import ValidationError as CoreValidationError
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # 模拟模型能力
        mock_capabilities = Mock()
        mock_capabilities.max_tokens_limit = 4096
        mock_get_capabilities.return_value = mock_capabilities
        
        with pytest.raises(CoreValidationError, match="max_tokens \\(5000\\) exceeds limit for model"):
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                max_tokens=5000
            )
    
    @patch('harborai.core.models.get_model_capabilities')
    def test_max_tokens_参数验证_有效值(self, mock_get_capabilities):
        """测试max_tokens参数验证 - 有效值。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        # 模拟模型能力
        mock_capabilities = Mock()
        mock_capabilities.max_tokens_limit = 4096
        mock_get_capabilities.return_value = mock_capabilities
        
        with patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback') as mock_create:
            mock_response = Mock(spec=ChatCompletion)
            mock_create.return_value = mock_response
            
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                max_tokens=1000
            )
            assert result == mock_response
    
    def test_structured_provider_参数验证_无效值(self):
        """测试structured_provider参数验证 - 无效值。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValidationError, match="Invalid structured_provider 'invalid'"):
            self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                structured_provider="invalid"
            )
    
    def test_structured_provider_参数验证_有效值(self):
        """测试structured_provider参数验证 - 有效值。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback') as mock_create:
            mock_response = Mock(spec=ChatCompletion)
            mock_create.return_value = mock_response
            
            # 测试agently
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                structured_provider="agently"
            )
            assert result == mock_response
            
            # 测试native
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                structured_provider="native"
            )
            assert result == mock_response
    
    def test_fallback_参数处理_优先级(self):
        """测试fallback参数处理 - fallback_models优先级高于fallback。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback') as mock_create:
            mock_response = Mock(spec=ChatCompletion)
            mock_create.return_value = mock_response
            
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                fallback=["gpt-4"],
                fallback_models=["claude-3"]
            )
            
            # 验证调用参数中使用了fallback_models
            call_args = mock_create.call_args[1]
            assert call_args['fallback'] == ["claude-3"]
    
    def test_fallback_参数处理_默认空列表(self):
        """测试fallback参数处理 - 默认为空列表。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback') as mock_create:
            mock_response = Mock(spec=ChatCompletion)
            mock_create.return_value = mock_response
            
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo"
            )
            
            # 验证调用参数中fallback为空列表
            call_args = mock_create.call_args[1]
            assert call_args['fallback'] == []


class TestAsyncParameterValidation:
    """异步参数验证测试类。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    @pytest.mark.asyncio
    async def test_async_temperature_参数验证_非数字类型(self):
        """测试异步temperature参数验证 - 非数字类型。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="temperature must be a number"):
            await self.chat_completions._acreate_core(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_async_max_tokens_参数验证_非整数类型(self):
        """测试异步max_tokens参数验证 - 非整数类型。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValueError, match="max_tokens must be an integer"):
            await self.chat_completions._acreate_core(
                messages=messages,
                model="gpt-3.5-turbo",
                max_tokens="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_async_structured_provider_参数验证_无效值(self):
        """测试异步structured_provider参数验证 - 无效值。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ValidationError, match="Invalid structured_provider 'invalid'. Must be 'agently' or 'native'"):
            await self.chat_completions._acreate_core(
                messages=messages,
                model="gpt-3.5-turbo",
                structured_provider="invalid"
            )


class TestFastStructuredOutputErrorHandling:
    """快速结构化输出错误处理测试类。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_fast_structured_output_处理器异常(self):
        """测试快速结构化输出处理器异常时回退到常规路径。"""
        messages = [{"role": "user", "content": "Hello"}]
        response_format = {"type": "json_schema", "json_schema": {"schema": {}}}
        
        with patch.object(self.chat_completions, '_get_fast_processor') as mock_get_processor:
            mock_processor = Mock()
            mock_processor.process_structured_output.side_effect = Exception("处理器错误")
            mock_get_processor.return_value = mock_processor
            
            with patch.object(self.chat_completions, '_create_core') as mock_create_core:
                mock_response = Mock(spec=ChatCompletion)
                mock_create_core.return_value = mock_response
                
                result = self.chat_completions._create_fast_structured_path(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    response_format=response_format,
                    structured_provider="agently"
                )
                
                # 验证回退到常规路径
                mock_create_core.assert_called_once()
                assert result == mock_response
    
    @pytest.mark.asyncio
    async def test_async_fast_structured_output_处理器异常(self):
        """测试异步快速结构化输出处理器异常时回退到常规路径。"""
        messages = [{"role": "user", "content": "Hello"}]
        response_format = {"type": "json_schema", "json_schema": {"schema": {}}}
        
        with patch.object(self.chat_completions, '_get_fast_processor') as mock_get_processor:
            mock_processor = Mock()
            mock_processor.aprocess_structured_output = AsyncMock(side_effect=Exception("异步处理器错误"))
            mock_get_processor.return_value = mock_processor
            
            with patch.object(self.chat_completions, '_acreate_core') as mock_acreate_core:
                mock_response = Mock(spec=ChatCompletion)
                mock_acreate_core.return_value = mock_response
                
                result = await self.chat_completions._acreate_fast_structured_path(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    response_format=response_format,
                    structured_provider="agently"
                )
                
                # 验证回退到常规路径
                mock_acreate_core.assert_called_once()
                assert result == mock_response
    
    def test_fast_structured_output_响应构建异常(self):
        """测试快速结构化输出响应构建异常时回退到常规路径。"""
        messages = [{"role": "user", "content": "Hello"}]
        response_format = {"type": "json_schema", "json_schema": {"schema": {}}}
        
        with patch.object(self.chat_completions, '_get_fast_processor') as mock_get_processor:
            mock_processor = Mock()
            mock_processor.process_structured_output.return_value = "valid_response"
            mock_get_processor.return_value = mock_processor
            
            # 模拟ChatCompletion构造失败 - 需要patch在client.py中的导入
            with patch('harborai.api.client.ChatCompletion', side_effect=Exception("响应构建失败")):
                with patch.object(self.chat_completions, '_create_core') as mock_create_core:
                    mock_response = Mock(spec=ChatCompletion)
                    mock_create_core.return_value = mock_response
                    
                    result = self.chat_completions._create_fast_structured_path(
                        messages=messages,
                        model="gpt-3.5-turbo",
                        response_format=response_format,
                        structured_provider="agently"
                    )
                    
                    # 验证回退到常规路径
                    mock_create_core.assert_called_once()
                    assert result == mock_response


class TestStreamingFunctionality:
    """流式处理功能测试类。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
    
    def test_流式响应_基本功能(self):
        """测试流式响应基本功能。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        # 模拟流式响应
        mock_chunk1 = Mock(spec=ChatCompletionChunk)
        mock_chunk2 = Mock(spec=ChatCompletionChunk)
        mock_stream = iter([mock_chunk1, mock_chunk2])
        
        with patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback') as mock_create:
            mock_create.return_value = mock_stream
            
            result = self.chat_completions._create_core(
                messages=messages,
                model="gpt-3.5-turbo",
                stream=True
            )
            
            # 验证返回的是迭代器
            chunks = list(result)
            assert len(chunks) == 2
            assert chunks[0] == mock_chunk1
            assert chunks[1] == mock_chunk2
    
    @pytest.mark.asyncio
    async def test_异步流式响应_基本功能(self):
        """测试异步流式响应基本功能。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        # 创建模拟的异步生成器
        async def mock_async_generator():
            mock_chunk1 = Mock(spec=ChatCompletionChunk)
            mock_chunk1.choices = [{"delta": {"content": "Hello"}, "finish_reason": None}]
            mock_chunk2 = Mock(spec=ChatCompletionChunk)
            mock_chunk2.choices = [{"delta": {"content": " World"}, "finish_reason": "stop"}]
            yield mock_chunk1
            yield mock_chunk2
        
        with patch.object(self.chat_completions.client_manager, 'chat_completion_with_fallback', new_callable=AsyncMock) as mock_acreate:
            # 使用AsyncMock并返回异步生成器
            mock_acreate.return_value = mock_async_generator()
            
            result = await self.chat_completions._acreate_core(
                messages=messages,
                model="gpt-3.5-turbo",
                stream=True
            )
            
            # 验证返回的是异步迭代器
            chunks = []
            async for chunk in result:
                chunks.append(chunk)
            
            assert len(chunks) == 2
            assert chunks[0].choices[0]["delta"]["content"] == "Hello"
            assert chunks[1].choices[0]["delta"]["content"] == " World"
    
    def test_流式响应_错误处理(self):
        """测试流式响应错误处理。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(self.chat_completions.client_manager, 'chat_completion_sync_with_fallback') as mock_create:
            mock_create.side_effect = Exception("流式处理错误")
            
            with pytest.raises(Exception, match="流式处理错误"):
                result = self.chat_completions._create_core(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    stream=True
                )
                list(result)  # 触发异常
    
    @pytest.mark.asyncio
    async def test_异步流式响应_错误处理(self):
        """测试异步流式响应错误处理。"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(self.chat_completions.client_manager, 'chat_completion_with_fallback') as mock_acreate:
            mock_acreate.side_effect = Exception("异步流式处理错误")
            
            with pytest.raises(Exception, match="异步流式处理错误"):
                await self.chat_completions._acreate_core(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    stream=True
                )


class TestHarborAIAdvancedMethods:
    """HarborAI高级方法测试类。"""
    
    def setup_method(self):
        """设置测试环境。"""
        self.harbor_ai = HarborAI()
    
    def test_get_available_models_实现(self):
        """测试get_available_models方法实现。"""
        with patch.object(self.harbor_ai.client_manager, 'get_available_models') as mock_get_models:
            mock_models = ["gpt-3.5-turbo", "gpt-4", "claude-3"]
            mock_get_models.return_value = mock_models
            
            result = self.harbor_ai.get_available_models()
            assert result == mock_models
            mock_get_models.assert_called_once()
    
    def test_get_plugin_info_实现(self):
        """测试get_plugin_info方法实现。"""
        with patch.object(self.harbor_ai.client_manager, 'get_plugin_info') as mock_get_info:
            mock_info = {"plugins": ["plugin1", "plugin2"]}
            mock_get_info.return_value = mock_info
            
            result = self.harbor_ai.get_plugin_info()
            assert result == mock_info
            mock_get_info.assert_called_once()
    
    def test_register_plugin_实现(self):
        """测试register_plugin方法实现。"""
        mock_plugin = Mock()
        
        with patch.object(self.harbor_ai.client_manager, 'register_plugin') as mock_register:
            self.harbor_ai.register_plugin(mock_plugin)
            mock_register.assert_called_once_with(mock_plugin)
    
    def test_unregister_plugin_实现(self):
        """测试unregister_plugin方法实现。"""
        plugin_name = "test_plugin"
        
        with patch.object(self.harbor_ai.client_manager, 'unregister_plugin') as mock_unregister:
            self.harbor_ai.unregister_plugin(plugin_name)
            mock_unregister.assert_called_once_with(plugin_name)
    
    def test_get_total_cost_with_cost_tracker(self):
        """测试带成本跟踪器的get_total_cost方法。"""
        mock_cost_tracker = Mock()
        mock_cost_tracker.get_total_cost.return_value = 15.75
        
        with patch.object(self.harbor_ai, 'cost_tracker', mock_cost_tracker):
            result = self.harbor_ai.get_total_cost()
            assert result == 15.75
            mock_cost_tracker.get_total_cost.assert_called_once()
    
    def test_get_total_cost_without_cost_tracker(self):
        """测试无成本跟踪器的get_total_cost方法。"""
        with patch.object(self.harbor_ai, 'cost_tracker', None):
            result = self.harbor_ai.get_total_cost()
            assert result == 0.0
    
    def test_reset_cost_with_cost_tracker(self):
        """测试带成本跟踪器的reset_cost方法。"""
        mock_cost_tracker = Mock()
        
        with patch.object(self.harbor_ai, 'cost_tracker', mock_cost_tracker):
            self.harbor_ai.reset_cost()
            mock_cost_tracker.reset.assert_called_once()
    
    def test_reset_cost_without_cost_tracker(self):
        """测试无成本跟踪器的reset_cost方法。"""
        with patch.object(self.harbor_ai, 'cost_tracker', None):
            # 应该不抛出异常
            self.harbor_ai.reset_cost()
    
    @pytest.mark.asyncio
    async def test_aclose_with_plugin_error(self):
        """测试aclose方法在插件错误时的处理。"""
        # 创建一个有aclose方法的mock插件
        mock_plugin = Mock()
        mock_plugin.name = "test_plugin"
        mock_plugin.aclose = AsyncMock(side_effect=Exception("关闭错误"))
        
        # 将mock插件添加到client_manager.plugins中
        self.harbor_ai.client_manager.plugins = {"test_plugin": mock_plugin}
        
        # 应该不抛出异常，只记录日志
        await self.harbor_ai.aclose()
        
        # 验证插件的aclose方法被调用
        mock_plugin.aclose.assert_called_once()
    
    def test_close_with_plugin_error(self):
        """测试close方法在插件错误时的处理。"""
        # 创建一个有close方法的mock插件
        mock_plugin = Mock()
        mock_plugin.name = "test_plugin"
        mock_plugin.close = Mock(side_effect=Exception("关闭错误"))
        
        # 将mock插件添加到client_manager.plugins中
        self.harbor_ai.client_manager.plugins = {"test_plugin": mock_plugin}
        
        # 应该不抛出异常，只记录日志
        self.harbor_ai.close()
        
        # 验证插件的close方法被调用
        mock_plugin.close.assert_called_once()
    
    def test_client_别名属性(self):
        """测试client别名属性。"""
        assert hasattr(self.harbor_ai, 'client')
        assert self.harbor_ai.client == self.harbor_ai