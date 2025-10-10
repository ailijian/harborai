"""
测试用例专门用于提升client.py的覆盖率
重点覆盖未测试的代码路径和边界条件
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, List, Any, Optional

from harborai.api.client import HarborAI, ChatCompletions, Chat
from harborai.core.base_plugin import ChatCompletion, ChatCompletionChunk
from harborai.utils.exceptions import HarborAIError, ValidationError


class TestChatCompletionsCoverageImprovement:
    """提升ChatCompletions类的测试覆盖率"""
    
    def setup_method(self):
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
        self.chat_completions.api_logger = Mock()
    
    @patch('harborai.api.client.get_performance_config')
    def test_create_fast_structured_path_no_user_input(self, mock_get_perf_config):
        """测试快速结构化路径在没有用户输入时的回退"""
        mock_get_perf_config.return_value = Mock(
            enable_fast_path=True,
            enable_structured_output_optimization=True
        )
        
        # 没有用户消息的消息列表
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"}
        ]
        
        response_format = {"type": "json_object"}
        
        with patch.object(self.chat_completions, '_create_core') as mock_create_core:
            mock_create_core.return_value = Mock()
            
            self.chat_completions._create_fast_structured_path(
                messages=messages,
                model="gpt-3.5-turbo",
                response_format=response_format,
                structured_provider="openai"
            )
            
            # 验证回退到常规路径
            mock_create_core.assert_called_once()
    
    @patch('harborai.api.client.get_performance_config')
    def test_create_fast_structured_path_exception_fallback(self, mock_get_perf_config):
        """测试_create_fast_structured_path方法在异常时回退到_create_core"""
        # Mock性能配置
        mock_get_perf_config.return_value = {'enable_fast_structured_output': True}
        
        # Mock快速处理器抛出异常
        mock_processor = Mock()
        mock_processor.process_structured_output.side_effect = Exception("处理失败")
        
        # Mock api_logger
        self.chat_completions.api_logger = Mock()
        
        with patch.object(self.chat_completions, '_get_fast_processor', return_value=mock_processor):
            with patch.object(self.chat_completions, '_create_core', new_callable=Mock) as mock_create_core:
                mock_create_core.return_value = Mock(spec=ChatCompletion)
                
                messages = [{"role": "user", "content": "test"}]
                response_format = {"type": "json_object"}
                
                result = self.chat_completions._create_fast_structured_path(
                    messages=messages,
                    model="test-model",
                    response_format=response_format,
                    structured_provider="test"
                )
                
                # 验证回退到_create_core
                mock_create_core.assert_called_once()
                # 验证错误日志被记录
                self.chat_completions.api_logger.log_error.assert_called_once()
                assert result is not None
    
    @pytest.mark.asyncio
    @patch('harborai.api.client.get_performance_config')
    async def test_acreate_fast_structured_path_no_user_input(self, mock_get_perf_config):
        """测试异步快速结构化路径在没有用户输入时的回退"""
        mock_get_perf_config.return_value = Mock(
            enable_fast_path=True,
            enable_structured_output_optimization=True
        )
        
        # 没有用户消息的消息列表
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "How can I help you?"}
        ]
        
        response_format = {"type": "json_object"}
        
        with patch.object(self.chat_completions, '_acreate_core', new_callable=AsyncMock) as mock_acreate_core:
            mock_acreate_core.return_value = Mock()
            
            await self.chat_completions._acreate_fast_structured_path(
                messages=messages,
                model="gpt-3.5-turbo",
                response_format=response_format,
                structured_provider="openai"
            )
            
            # 验证回退到常规路径
            mock_acreate_core.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('harborai.api.client.get_performance_config')
    async def test_acreate_fast_structured_path_exception_fallback(self, mock_get_perf_config):
        """测试异步快速结构化路径异常时的回退"""
        mock_get_perf_config.return_value = Mock(
            enable_fast_path=True,
            enable_structured_output_optimization=True
        )
        
        messages = [{"role": "user", "content": "Hello"}]
        response_format = {"type": "json_object"}
        
        # Mock快速处理器抛出异常
        with patch.object(self.chat_completions, '_get_fast_processor') as mock_get_processor:
            mock_processor = Mock()
            mock_processor.process_structured_output = Mock(side_effect=Exception("异步处理失败"))
            mock_get_processor.return_value = mock_processor
            
            with patch.object(self.chat_completions, '_acreate_core', new_callable=AsyncMock) as mock_acreate_core:
                mock_acreate_core.return_value = Mock()
                
                await self.chat_completions._acreate_fast_structured_path(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    response_format=response_format,
                    structured_provider="openai"
                )
                
                # 验证回退到常规路径
                mock_acreate_core.assert_called_once()
                # 注意：异步版本没有记录错误日志，只是简单回退
    
    def test_create_core_with_stream_true(self):
        """测试_create_core方法在stream=True时的行为"""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock流式响应
        mock_stream_response = [
            Mock(spec=ChatCompletionChunk),
            Mock(spec=ChatCompletionChunk)
        ]
        
        self.mock_client_manager.chat_completion_sync_with_fallback.return_value = mock_stream_response
        
        result = self.chat_completions._create_core(
            messages=messages,
            model="gpt-3.5-turbo",
            stream=True
        )
        
        # 验证返回流式响应
        assert result == mock_stream_response
        self.mock_client_manager.chat_completion_sync_with_fallback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_acreate_core_with_stream_true(self):
        """测试_acreate_core方法在stream=True时的行为"""
        # 创建一个真正的异步生成器函数
        async def mock_stream():
            yield ChatCompletionChunk(id="test", choices=[], created=123, model="test")
        
        # 使用AsyncMock并设置side_effect为异步生成器函数
        self.mock_client_manager.chat_completion_with_fallback = AsyncMock(side_effect=lambda *args, **kwargs: mock_stream())
        
        # Mock api_logger的异步方法
        self.chat_completions.api_logger = Mock()
        self.chat_completions.api_logger.alog_request = AsyncMock()
        self.chat_completions.api_logger.alog_response = AsyncMock()
        self.chat_completions.api_logger.alog_error = AsyncMock()
        
        messages = [{"role": "user", "content": "test"}]
        result = await self.chat_completions._acreate_core(
            messages=messages,
            model="test-model",
            stream=True
        )
        
        # 验证返回的是异步迭代器
        assert hasattr(result, '__aiter__')
        self.mock_client_manager.chat_completion_with_fallback.assert_called_once()
    
    def test_validate_messages_with_tool_calls(self):
        """测试包含工具调用的消息验证"""
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}
                    }
                ]
            },
            {
                "role": "tool",
                "content": "Sunny, 25°C",
                "tool_call_id": "call_123"
            }
        ]
        
        # 应该不抛出异常
        self.chat_completions._validate_messages(messages)
    
    def test_validate_messages_empty_content_with_tool_calls(self):
        """测试空内容但有工具调用的消息验证"""
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}
                    }
                ]
            }
        ]
        
        # 应该不抛出异常，因为有tool_calls
        self.chat_completions._validate_messages(messages)
    
    def test_process_messages_for_reasoning_model_no_reasoning(self):
        """测试推理模型消息处理，但没有推理内容"""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        result = self.chat_completions._process_messages_for_reasoning_model(messages)
        
        # 应该返回原始消息
        assert result == messages


class TestHarborAICoverageImprovement:
    """提升HarborAI类的测试覆盖率"""
    
    def setup_method(self):
        with patch('harborai.api.client.ClientManager'):
            self.harbor_ai = HarborAI()
    
    def test_get_available_models_with_model_info_objects(self):
        """测试get_available_models返回ModelInfo对象的情况"""
        from harborai.core.base_plugin import ModelInfo
        
        # Mock返回ModelInfo对象列表
        mock_model_infos = [
            ModelInfo(id="gpt-3.5-turbo", name="GPT-3.5 Turbo", provider="openai", max_tokens=4096),
            ModelInfo(id="gpt-4", name="GPT-4", provider="openai", max_tokens=8192)
        ]
        
        self.harbor_ai.client_manager.get_available_models.return_value = mock_model_infos
        
        result = self.harbor_ai.get_available_models()
        
        # 应该返回模型ID列表
        assert result == ["gpt-3.5-turbo", "gpt-4"]
    
    def test_get_available_models_with_string_list(self):
        """测试get_available_models返回字符串列表的情况"""
        # Mock返回字符串列表
        mock_model_names = ["gpt-3.5-turbo", "claude-3", "gemini-pro"]
        
        self.harbor_ai.client_manager.get_available_models.return_value = mock_model_names
        
        result = self.harbor_ai.get_available_models()
        
        # 应该直接返回字符串列表
        assert result == mock_model_names
    
    @pytest.mark.asyncio
    async def test_aclose_with_performance_manager_error(self):
        """测试aclose方法在性能管理器清理错误时的处理"""
        # Mock性能管理器
        mock_perf_manager = Mock()
        mock_perf_manager.is_initialized.return_value = True
        self.harbor_ai._performance_manager = mock_perf_manager
        
        # Mock cleanup_performance_manager抛出异常
        with patch('harborai.api.client.cleanup_performance_manager') as mock_cleanup:
            mock_cleanup.side_effect = Exception("清理失败")
            
            # Mock logger
            self.harbor_ai.logger = Mock()
            
            # 应该不抛出异常，只记录警告
            await self.harbor_ai.aclose()
            
            # 验证警告日志被记录
            self.harbor_ai.logger.warning.assert_called()
    
    def test_close_with_performance_manager_error(self):
        """测试close方法在性能管理器清理时出错的情况"""
        # 模拟性能管理器已初始化
        mock_perf_manager = Mock()
        mock_perf_manager.is_initialized.return_value = True
        self.harbor_ai._performance_manager = mock_perf_manager
        
        # 模拟事件循环错误
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_running.return_value = False
            mock_loop.run_until_complete.side_effect = Exception("Event loop error")
            mock_get_loop.return_value = mock_loop
            
            # Mock logger
            self.harbor_ai.logger = Mock()
            
            # 调用close方法，不应该抛出异常
            self.harbor_ai.close()
            
            # 验证警告日志被记录
            self.harbor_ai.logger.warning.assert_called()
    
    def test_close_with_running_event_loop(self):
        """测试close方法在事件循环运行时的处理"""
        # Mock性能管理器
        mock_perf_manager = Mock()
        mock_perf_manager.is_initialized.return_value = True
        self.harbor_ai._performance_manager = mock_perf_manager
        
        # Mock asyncio
        with patch('asyncio.get_event_loop') as mock_get_loop:
            with patch('asyncio.create_task') as mock_create_task:
                mock_loop = Mock()
                mock_loop.is_running.return_value = True
                mock_get_loop.return_value = mock_loop
                
                # Mock logger
                self.harbor_ai.logger = Mock()
                
                self.harbor_ai.close()
                
                # 验证create_task被调用
                mock_create_task.assert_called_once()
    
    def test_close_with_stopped_event_loop(self):
        """测试close方法在事件循环未运行时的处理"""
        # Mock性能管理器
        mock_perf_manager = Mock()
        mock_perf_manager.is_initialized.return_value = True
        self.harbor_ai._performance_manager = mock_perf_manager
        
        # Mock asyncio
        with patch('asyncio.get_event_loop') as mock_get_loop:
            mock_loop = Mock()
            mock_loop.is_running.return_value = False
            mock_get_loop.return_value = mock_loop
            
            # Mock logger
            self.harbor_ai.logger = Mock()
            
            self.harbor_ai.close()
            
            # 验证run_until_complete被调用
            mock_loop.run_until_complete.assert_called_once()


class TestEdgeCasesAndErrorHandling:
    """测试边界情况和错误处理"""
    
    def setup_method(self):
        self.mock_client_manager = Mock()
        self.chat_completions = ChatCompletions(self.mock_client_manager)
        self.chat_completions.api_logger = Mock()
    
    def test_create_with_all_optional_parameters(self):
        """测试create方法使用所有可选参数"""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock响应
        mock_response = Mock(spec=ChatCompletion)
        self.mock_client_manager.chat_completion_sync_with_fallback.return_value = mock_response
        
        result = self.chat_completions.create(
            messages=messages,
            model="gpt-3.5-turbo",
            frequency_penalty=0.5,
            function_call="auto",
            functions=[{"name": "test_func", "description": "Test function"}],
            logit_bias={"token_id": 0.5},
            logprobs=True,
            top_logprobs=5,
            max_tokens=1000,
            n=2,
            presence_penalty=0.3,
            response_format={"type": "text"},
            seed=42,
            stop=["\\n"],
            stream=False,
            structured_provider="agently",
            temperature=0.7,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "test_tool"}}],
            top_p=0.9,
            user="test_user",
            extra_body={"custom_param": "value"},
            timeout=30.0,
            fallback=["gpt-4"],
            fallback_models=["claude-3"],
            retry_policy={"max_retries": 3}
        )
        
        assert result == mock_response
        self.mock_client_manager.chat_completion_sync_with_fallback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_acreate_with_all_optional_parameters(self):
        """测试acreate方法使用所有可选参数"""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock返回值
        mock_response = Mock(spec=ChatCompletion)
        self.mock_client_manager.chat_completion_with_fallback = AsyncMock(return_value=mock_response)
        
        # Mock api_logger的异步方法
        self.chat_completions.api_logger = Mock()
        self.chat_completions.api_logger.alog_request = AsyncMock()
        self.chat_completions.api_logger.alog_response = AsyncMock()
        self.chat_completions.api_logger.alog_error = AsyncMock()
        
        result = await self.chat_completions.acreate(
            messages=messages,
            model="gpt-3.5-turbo",
            frequency_penalty=0.5,
            function_call="auto",
            functions=[{"name": "test_func", "description": "test"}],
            logit_bias={"token": 0.1},
            logprobs=True,
            top_logprobs=5,
            max_tokens=100,
            n=1,
            presence_penalty=0.3,
            response_format={"type": "json_object"},
            seed=42,
            stop=["stop"],
            stream=False,
            structured_provider="agently",
            temperature=0.7,
            tool_choice="auto",
            tools=[{"type": "function", "function": {"name": "test"}}],
            top_p=0.9,
            user="test_user",
            extra_body={"custom": "value"},
            timeout=30.0,
            fallback=["gpt-4"],
            fallback_models=["gpt-4"],
            retry_policy={"max_retries": 3}
        )
        
        assert result == mock_response
        self.mock_client_manager.chat_completion_with_fallback.assert_called_once()