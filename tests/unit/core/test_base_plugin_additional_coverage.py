#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_plugin.py 模块的额外测试覆盖用例

专门针对未覆盖的代码行进行测试，特别是：
- ImportError异常处理（第262-264行）
- 其他边界条件和异常情况

遵循VIBE编码规范：
- TDD测试驱动开发
- 详细的中文注释
- 覆盖异常和边界条件
- 测试独立性和可重复性
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from harborai.core.base_plugin import (
    ChatMessage,
    ChatChoice,
    ChatCompletion,
    BaseLLMPlugin
)


class TestBaseLLMPlugin:
    """测试BaseLLMPlugin的额外覆盖用例
    
    专门测试未被现有测试覆盖的代码路径
    """
    
    @pytest.fixture
    def plugin(self):
        """创建测试用的插件实例"""
        class TestPlugin(BaseLLMPlugin):
            def __init__(self):
                super().__init__("test-plugin")
            
            def get_available_models(self):
                return []
            
            def chat_completion(self, model, messages, **kwargs):
                return ChatCompletion(
                    id="test-123",
                    object="chat.completion",
                    created=int(time.time()),
                    model=model,
                    choices=[
                        ChatChoice(
                            index=0,
                            message=ChatMessage(role="assistant", content="Test response"),
                            finish_reason="stop"
                        )
                    ]
                )
            
            async def chat_completion_async(self, model, messages, **kwargs):
                return self.chat_completion(model, messages, **kwargs)
        
        return TestPlugin()
    
    @patch('harborai.core.base_plugin.get_current_trace_id')
    def test_parse_with_agently_import_error(self, mock_trace_id, plugin):
        """测试_parse_with_agently方法的ImportError异常处理
        
        功能：验证当Agently库不可用时的异常处理
        参数：用户输入、响应格式、模型名称
        返回：抛出Exception异常
        异常：ImportError -> Exception("Agently library not available...")
        边界：模拟Agently库导入失败的情况
        假设：ImportError会被捕获并转换为特定异常
        """
        mock_trace_id.return_value = "test-trace-123"
        
        user_input = "测试用户输入"
        response_format = {
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        model = "test-model"
        
        # 模拟ImportError - 直接patch导入语句
        import sys
        original_import = __builtins__['__import__']
        
        def mock_import(name, *args, **kwargs):
            if name == 'harborai.api.structured' or 'structured' in name:
                raise ImportError("No module named 'structured'")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(Exception) as exc_info:
                plugin._parse_with_agently(user_input, response_format, model)
            
            # 验证异常消息
            assert "Agently library not available and cannot fallback to native parsing with user input" in str(exc_info.value)
    
    @patch('harborai.core.base_plugin.get_current_trace_id')
    def test_parse_with_agently_api_error(self, mock_trace_id, plugin):
        """测试_parse_with_agently方法的API错误处理
        
        功能：验证当Agently API调用失败时的异常处理
        参数：用户输入、响应格式、模型名称
        返回：抛出原始异常
        异常：API错误直接抛出，不回退
        边界：模拟API密钥错误或其他API异常
        假设：非ImportError的异常会直接抛出
        """
        mock_trace_id.return_value = "test-trace-123"
        
        user_input = "测试用户输入"
        response_format = {
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        model = "test-model"
        
        # 模拟API错误
        api_error = Exception("API key is invalid")
        
        with patch('harborai.api.structured.default_handler._parse_with_agently', side_effect=api_error):
            with pytest.raises(Exception) as exc_info:
                plugin._parse_with_agently(user_input, response_format, model)
            
            # 验证抛出的是原始异常
            assert str(exc_info.value) == "API key is invalid"
    
    @patch('harborai.core.base_plugin.get_current_trace_id')
    def test_parse_with_native_json_fallback(self, mock_trace_id, plugin):
        """测试_parse_with_native方法的JSON回退处理
        
        功能：验证当结构化处理失败时回退到JSON解析
        参数：内容字符串、响应格式
        返回：JSON解析结果
        异常：结构化处理失败时使用json.loads回退
        边界：模拟结构化处理器失败的情况
        假设：最后的回退是直接JSON解析
        """
        mock_trace_id.return_value = "test-trace-123"
        
        content = '{"name": "John", "age": 30}'
        response_format = {
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            }
        }
        
        # 模拟结构化处理器失败
        with patch('harborai.api.structured.default_handler._parse_with_native', side_effect=Exception("Structured parsing failed")):
            result = plugin._parse_with_native(content, response_format)
            
            # 验证回退到JSON解析成功
            assert result == {"name": "John", "age": 30}
    
    @patch('harborai.core.base_plugin.get_current_trace_id')
    def test_handle_structured_output_agently_import_error_flow(self, mock_trace_id, plugin):
        """测试handle_structured_output中Agently ImportError的完整流程
        
        功能：验证结构化输出处理中ImportError的完整处理流程
        参数：响应对象、响应格式、结构化提供商
        返回：原始响应（异常被捕获）
        异常：ImportError被捕获，记录警告日志
        边界：测试Agently不可用时的完整处理流程
        假设：异常被捕获后返回原始响应
        """
        mock_trace_id.return_value = "test-trace-123"
        
        response = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content='{"name": "John"}'),
                    finish_reason="stop"
                )
            ]
        )
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        
        original_messages = [
            ChatMessage(role="user", content="请返回JSON格式的用户信息")
        ]
        
        # 模拟_parse_with_agently抛出ImportError异常
        with patch.object(plugin, '_parse_with_agently') as mock_agently:
            mock_agently.side_effect = Exception("Agently library not available and cannot fallback to native parsing with user input")
            
            with patch.object(plugin.logger, 'warning') as mock_warning:
                result = plugin.handle_structured_output(
                    response,
                    response_format=response_format,
                    structured_provider="agently",
                    original_messages=original_messages,
                    model="test-model"
                )
                
                # 验证返回原始响应
                assert result == response
                # 验证parsed属性未被设置（因为异常被捕获）
                assert not hasattr(result.choices[0].message, 'parsed') or result.choices[0].message.parsed is None
                
                # 验证警告日志记录
                mock_warning.assert_called_once()
                warning_call = mock_warning.call_args
                assert "Failed to parse structured output" in warning_call[0][0]
                assert warning_call[1]["extra"]["trace_id"] == "test-trace-123"
                assert warning_call[1]["extra"]["provider"] == "agently"
                assert "Agently library not available" in warning_call[1]["extra"]["error"]
    
    def test_handle_structured_output_no_original_messages_agently(self, plugin):
        """测试handle_structured_output在没有原始消息时使用Agently的处理
        
        功能：验证当使用Agently但没有原始消息时的回退处理
        参数：响应对象、响应格式、结构化提供商（无原始消息）
        返回：使用原生解析的结果
        异常：无
        边界：测试Agently模式下缺少原始消息的情况
        假设：没有原始消息时会回退到原生解析
        """
        response = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content='{"name": "John"}'),
                    finish_reason="stop"
                )
            ]
        )
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        
        # 模拟原生解析成功
        with patch.object(plugin, '_parse_with_native') as mock_native:
            mock_native.return_value = {"name": "John"}
            
            result = plugin.handle_structured_output(
                response,
                response_format=response_format,
                structured_provider="agently",
                original_messages=None  # 没有原始消息
            )
            
            # 验证调用了原生解析
            mock_native.assert_called_once_with(
                '{"name": "John"}',
                response_format
            )
            
            # 验证结果
            assert result.choices[0].message.parsed == {"name": "John"}
    
    def test_handle_structured_output_empty_original_messages_agently(self, plugin):
        """测试handle_structured_output在原始消息为空时使用Agently的处理
        
        功能：验证当使用Agently但原始消息为空列表时的回退处理
        参数：响应对象、响应格式、结构化提供商（空消息列表）
        返回：使用原生解析的结果
        异常：无
        边界：测试Agently模式下原始消息为空列表的情况
        假设：空消息列表时会回退到原生解析
        """
        response = ChatCompletion(
            id="test-123",
            object="chat.completion",
            created=int(time.time()),
            model="test-model",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content='{"name": "John"}'),
                    finish_reason="stop"
                )
            ]
        )
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"}
                    }
                }
            }
        }
        
        # 模拟原生解析成功
        with patch.object(plugin, '_parse_with_native') as mock_native:
            mock_native.return_value = {"name": "John"}
            
            result = plugin.handle_structured_output(
                response,
                response_format=response_format,
                structured_provider="agently",
                original_messages=[]  # 空消息列表
            )
            
            # 验证调用了原生解析
            mock_native.assert_called_once_with(
                '{"name": "John"}',
                response_format
            )
            
            # 验证结果
            assert result.choices[0].message.parsed == {"name": "John"}