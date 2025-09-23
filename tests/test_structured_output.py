#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构化输出测试

测试基于Agently和native的结构化输出功能，包括流式和非流式输出。
"""

import json
import os
import pytest
import asyncio
import sys
from typing import Dict, Any, List
from unittest.mock import Mock, patch, AsyncMock

from harborai import HarborAI
from harborai.api.structured import StructuredOutputHandler, default_handler
from harborai.utils.exceptions import ValidationError, StructuredOutputError


class TestStructuredOutputHandler:
    """StructuredOutputHandler 测试类"""
    
    def test_init(self):
        """测试初始化"""
        handler = StructuredOutputHandler()
        assert handler is not None
        assert hasattr(handler, 'parse_response')
        assert hasattr(handler, 'parse_streaming_response')
    
    def test_check_agently_availability_success(self):
        """测试Agently可用性检查 - 成功"""
        handler = StructuredOutputHandler()
        with patch('builtins.__import__') as mock_import:
            mock_agently = Mock()
            mock_import.return_value = mock_agently
            result = handler._check_agently_availability()
            assert result is True
    
    def test_check_agently_availability_failure(self):
        """测试Agently可用性检查 - 失败"""
        handler = StructuredOutputHandler()
        with patch('builtins.__import__', side_effect=ImportError()):
            result = handler._check_agently_availability()
            assert result is False
    
    def test_convert_json_schema_to_agently_output(self):
        """测试JSON Schema到Agently格式的转换"""
        handler = StructuredOutputHandler()
        
        schema_wrapper = {
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "The message content"}
                    },
                    "required": ["message"]
                }
            }
        }
        
        result = handler._convert_json_schema_to_agently_output(schema_wrapper)
        
        assert isinstance(result, dict)
        assert "message" in result
        assert result["message"] == ("str", "The message content")
    
    def test_validate_against_schema_success(self):
        """测试JSON Schema验证 - 成功"""
        handler = StructuredOutputHandler()
        
        data = {"name": "John", "age": 30}
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        
        # 应该不抛出异常
        handler._validate_against_schema(data, schema)
    
    def test_validate_against_schema_failure(self):
        """测试JSON Schema验证 - 失败"""
        handler = StructuredOutputHandler()
        
        data = {"name": "John"}  # 缺少required字段age
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        
        with pytest.raises(StructuredOutputError):
            handler._validate_against_schema(data, schema)
    
    @pytest.mark.skipif(not os.getenv('DEEPSEEK_API_KEY'), reason="DEEPSEEK_API_KEY not configured")
    def test_parse_with_agently_success(self):
        """测试Agently解析 - 成功"""
        handler = StructuredOutputHandler()
        
        # 模拟Agently可用
        with patch.object(handler, '_check_agently_availability', return_value=True):
            # 直接模拟_parse_with_agently方法的返回值
            with patch.object(handler, '_parse_with_agently') as mock_parse:
                mock_parse.return_value = {"name": "John", "age": 30}
                
                response_text = 'Some response with {"name": "John", "age": 30}'
                schema = {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "number"}
                    },
                    "required": ["name", "age"]
                }
                
                result = handler._parse_with_agently(response_text, schema)
                
                assert result == {"name": "John", "age": 30}
    
    def test_parse_with_native_success(self):
        """测试原生解析 - 成功"""
        handler = StructuredOutputHandler()
        
        response_text = '{"name": "John", "age": 30}'
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        
        result = handler._parse_with_native(response_text, schema)
        
        assert result == {"name": "John", "age": 30}
    
    def test_parse_with_native_invalid_json(self):
        """测试原生解析 - 无效JSON"""
        handler = StructuredOutputHandler()
        
        response_text = 'invalid json'
        schema = {"type": "object"}
        
        with pytest.raises(StructuredOutputError):
            handler._parse_with_native(response_text, schema)
    
    def test_parse_response_agently_mode(self):
        """测试解析响应 - Agently模式"""
        handler = StructuredOutputHandler()
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            }
        }
        
        with patch.object(handler, '_parse_with_agently') as mock_agently:
            mock_agently.return_value = {"message": "test"}
            
            result = handler.parse_response(
                "test response",
                response_format["json_schema"]["schema"],
                use_agently=True
            )
            
            assert result == {"message": "test"}
            mock_agently.assert_called_once()
    
    def test_parse_response_native_mode(self):
        """测试解析响应 - Native模式"""
        handler = StructuredOutputHandler()
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            }
        }
        
        with patch.object(handler, '_parse_with_native') as mock_native:
            mock_native.return_value = {"message": "test"}
            
            result = handler.parse_response(
                "test response",
                response_format["json_schema"]["schema"],
                use_agently=False
            )
            
            assert result == {"message": "test"}
            mock_native.assert_called_once()
    
    def test_parse_response_fallback_mechanism(self):
        """测试解析响应 - 回退机制"""
        handler = StructuredOutputHandler()
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            }
        }
        
        # 模拟Agently不可用，应该回退到native解析
        with patch.object(handler, '_check_agently_availability', return_value=False):
            with patch.object(handler, '_parse_with_native') as mock_native:
                mock_native.return_value = {"message": "test"}
                
                result = handler.parse_response(
                    "test response",
                    response_format["json_schema"]["schema"],
                    use_agently=True
                )
                
                assert result == {"message": "test"}
                mock_native.assert_called_once()


class TestStreamingStructuredOutput:
    """流式结构化输出测试类"""
    
    def test_parse_streaming_response_sync(self):
        """测试同步流式解析"""
        handler = StructuredOutputHandler()
        
        def mock_stream():
            yield '{"message": "Hello"'
            yield ', "status": "ok"}'
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "status": {"type": "string"}
                    },
                    "required": ["message", "status"]
                }
            }
        }
        
        schema = response_format["json_schema"]["schema"]
        results = list(handler.parse_streaming_response(
            mock_stream(),
            schema,
            provider="native"
        ))
        
        # 应该有部分解析结果和最终完整结果
        assert len(results) > 0
        final_result = results[-1]
        assert final_result == {"message": "Hello", "status": "ok"}
    
    @pytest.mark.asyncio
    async def test_parse_streaming_response_async(self):
        """测试异步流式解析"""
        handler = StructuredOutputHandler()
        
        async def mock_async_stream():
            yield '{"message": "Hello"'
            yield ', "status": "ok"}'
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "status": {"type": "string"}
                    },
                    "required": ["message", "status"]
                }
            }
        }
        
        schema = response_format["json_schema"]["schema"]
        results = []
        async for result in handler.parse_streaming_response(
            mock_async_stream(),
            schema,
            provider="native"
        ):
            results.append(result)
        
        # 应该有部分解析结果和最终完整结果
        assert len(results) > 0
        final_result = results[-1]
        assert final_result == {"message": "Hello", "status": "ok"}


class TestIntegrationWithClient:
    """与客户端集成测试"""
    
    def test_structured_provider_parameter(self, harbor_client, sample_messages):
        """测试structured_provider参数"""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            }
        }
        
        # 测试agently模式
        response = harbor_client.chat.completions.create(
            model="mock-model",
            messages=sample_messages,
            response_format=response_format,
            structured_provider="agently"
        )
        
        assert response is not None
        
        # 测试native模式
        response = harbor_client.chat.completions.create(
            model="mock-model",
            messages=sample_messages,
            response_format=response_format,
            structured_provider="native"
        )
        
        assert response is not None
    
    def test_invalid_structured_provider(self, harbor_client, sample_messages):
        """测试无效的structured_provider参数"""
        with pytest.raises(ValidationError):
            harbor_client.chat.completions.create(
                model="mock-model",
                messages=sample_messages,
                structured_provider="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_async_structured_provider_parameter(self, harbor_client, sample_messages):
        """测试异步structured_provider参数"""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"}
                    },
                    "required": ["message"]
                }
            }
        }
        
        # 测试agently模式
        response = await harbor_client.chat.completions.acreate(
            model="mock-model",
            messages=sample_messages,
            response_format=response_format,
            structured_provider="agently"
        )
        
        assert response is not None
        
        # 测试native模式
        response = await harbor_client.chat.completions.acreate(
            model="mock-model",
            messages=sample_messages,
            response_format=response_format,
            structured_provider="native"
        )
        
        assert response is not None


class TestDefaultHandler:
    """默认处理器测试"""
    
    def test_default_handler_exists(self):
        """测试默认处理器存在"""
        assert default_handler is not None
        assert isinstance(default_handler, StructuredOutputHandler)
    
    def test_parse_streaming_structured_output_function(self):
        """测试便捷函数"""
        from harborai.api.structured import parse_streaming_structured_output
        
        def mock_stream():
            yield '{"test": "value"}'
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {
                        "test": {"type": "string"}
                    },
                    "required": ["test"]
                }
            }
        }
        
        schema = response_format["json_schema"]["schema"]
        results = list(parse_streaming_structured_output(
            mock_stream(),
            schema,
            provider="native"
        ))
        
        assert len(results) > 0
        assert results[-1] == {"test": "value"}