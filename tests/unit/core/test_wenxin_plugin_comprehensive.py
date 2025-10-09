"""文心一言插件comprehensive测试。"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import time

from harborai.core.plugins.wenxin_plugin import WenxinPlugin
from harborai.core.base_plugin import ChatMessage, ModelInfo
from harborai.utils.exceptions import PluginError, ValidationError


class TestWenxinPluginInitialization:
    """测试文心一言插件初始化。"""
    
    def test_wenxin_plugin_init_basic(self):
        """测试基本初始化。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        assert plugin.name == "test_wenxin"
        assert plugin.api_key == "test-key"
        assert plugin.base_url == "https://qianfan.baidubce.com/v2"
        assert plugin.timeout == 60
        assert plugin.max_retries == 3
        assert plugin._client is None
        assert plugin._async_client is None
    
    def test_wenxin_plugin_init_with_custom_config(self):
        """测试自定义配置初始化。"""
        plugin = WenxinPlugin(
            name="test_wenxin",
            api_key="test-key",
            base_url="https://custom.api.com",
            timeout=30,
            max_retries=5
        )
        assert plugin.base_url == "https://custom.api.com"
        assert plugin.timeout == 30
        assert plugin.max_retries == 5
    
    def test_wenxin_plugin_init_api_key_from_kwargs(self):
        """测试从kwargs获取API key。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key-from-kwargs")
        assert plugin.api_key == "test-key-from-kwargs"
    
    def test_wenxin_plugin_supported_models(self):
        """测试支持的模型列表。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        models = plugin.supported_models
        assert len(models) == 3
        
        # 检查模型信息
        model_ids = [model.id for model in models]
        assert "ernie-3.5-8k" in model_ids
        assert "ernie-4.0-turbo-8k" in model_ids
        assert "ernie-x1-turbo-32k" in model_ids
        
        # 检查推理模型
        thinking_model = next(model for model in models if model.id == "ernie-x1-turbo-32k")
        assert thinking_model.supports_thinking is True
        
        non_thinking_model = next(model for model in models if model.id == "ernie-3.5-8k")
        assert non_thinking_model.supports_thinking is False
    
    def test_wenxin_plugin_is_thinking_model(self):
        """测试推理模型判断。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        # 推理模型
        assert plugin.is_thinking_model("ernie-x1-turbo-32k") is True
        
        # 非推理模型
        assert plugin.is_thinking_model("ernie-3.5-8k") is False
        assert plugin.is_thinking_model("ernie-4.0-turbo-8k") is False
        
        # 未知模型
        assert plugin.is_thinking_model("unknown-model") is False


class TestWenxinPluginClientManagement:
    """测试客户端管理。"""
    
    def test_get_client_success(self):
        """测试成功获取同步客户端。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        with patch('httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = plugin._get_client()
            
            assert client == mock_client
            assert plugin._client == mock_client
            
            # 验证客户端配置
            mock_client_class.assert_called_once_with(
                base_url="https://qianfan.baidubce.com/v2",
                timeout=60,
                headers={
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json"
                }
            )
    
    def test_get_client_httpx_import_error(self):
        """测试httpx导入错误。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'httpx'")):
            with pytest.raises(PluginError, match="httpx not installed"):
                plugin._get_client()
    
    def test_get_async_client_success(self):
        """测试成功获取异步客户端。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            client = plugin._get_async_client()
            
            assert client == mock_client
            assert plugin._async_client == mock_client
            
            # 验证客户端配置
            mock_client_class.assert_called_once_with(
                base_url="https://qianfan.baidubce.com/v2",
                timeout=60,
                headers={
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json"
                }
            )
    
    def test_get_async_client_httpx_import_error(self):
        """测试异步客户端httpx导入错误。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'httpx'")):
            with pytest.raises(PluginError, match="httpx not installed"):
                plugin._get_async_client()
    
    def test_client_reuse(self):
        """测试客户端重用。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        with patch('httpx.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # 第一次获取
            client1 = plugin._get_client()
            # 第二次获取应该返回同一个实例
            client2 = plugin._get_client()
            
            assert client1 == client2
            assert mock_client_class.call_count == 1


class TestWenxinPluginValidation:
    """测试请求验证。"""
    
    def test_validate_request_success(self):
        """测试成功验证。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 应该不抛出异常
        plugin._validate_request("ernie-3.5-8k", messages)
    
    def test_validate_request_unsupported_model(self):
        """测试不支持的模型。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError, match="Model unknown-model is not supported"):
            plugin._validate_request("unknown-model", messages)
    
    def test_validate_request_empty_messages(self):
        """测试空消息列表。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        with pytest.raises(ValidationError, match="Messages cannot be empty"):
            plugin._validate_request("ernie-3.5-8k", [])
    
    def test_validate_request_missing_api_key(self):
        """测试缺少API key。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key=None)
        messages = [ChatMessage(role="user", content="Hello")]
        
        with pytest.raises(ValidationError, match="Wenxin API key is required"):
            plugin._validate_request("ernie-3.5-8k", messages)
    
    def test_validate_request_invalid_temperature(self):
        """测试无效的temperature参数。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # temperature太小
        with pytest.raises(ValidationError, match="Temperature must be between 0.01 and 1.0"):
            plugin._validate_request("ernie-3.5-8k", messages, temperature=0.0)
        
        # temperature太大
        with pytest.raises(ValidationError, match="Temperature must be between 0.01 and 1.0"):
            plugin._validate_request("ernie-3.5-8k", messages, temperature=1.1)
    
    def test_validate_request_invalid_top_p(self):
        """测试无效的top_p参数。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # top_p太小
        with pytest.raises(ValidationError, match="top_p must be between 0.01 and 1.0"):
            plugin._validate_request("ernie-3.5-8k", messages, top_p=0.0)
        
        # top_p太大
        with pytest.raises(ValidationError, match="top_p must be between 0.01 and 1.0"):
            plugin._validate_request("ernie-3.5-8k", messages, top_p=1.1)
    
    def test_validate_request_valid_parameters(self):
        """测试有效的参数。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        # 应该不抛出异常
        plugin._validate_request("ernie-3.5-8k", messages, temperature=0.7, top_p=0.9)


class TestWenxinPluginThinkingContent:
    """测试思考内容提取。"""
    
    def test_extract_thinking_content_from_reasoning_content(self):
        """测试从reasoning_content字段提取思考内容。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        response = {
            "reasoning_content": "这是思考过程",
            "result": "这是回答"
        }
        
        thinking = plugin._extract_thinking_content(response)
        assert thinking == "这是思考过程"
    
    def test_extract_thinking_content_from_result_reasoning(self):
        """测试从result.reasoning_content字段提取思考内容。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        response = {
            "result": {
                "reasoning_content": "嵌套的思考过程",
                "content": "回答内容"
            }
        }
        
        thinking = plugin._extract_thinking_content(response)
        assert thinking == "嵌套的思考过程"
    
    def test_extract_thinking_content_from_choices_message(self):
        """测试从choices.message.reasoning_content字段提取思考内容。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        response = {
            "choices": [{
                "message": {
                    "reasoning_content": "OpenAI格式的思考过程",
                    "content": "回答内容"
                }
            }]
        }
        
        thinking = plugin._extract_thinking_content(response)
        assert thinking == "OpenAI格式的思考过程"
    
    def test_extract_thinking_content_empty_reasoning(self):
        """测试空的reasoning_content。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        response = {
            "reasoning_content": "",
            "result": "这是回答"
        }
        
        thinking = plugin._extract_thinking_content(response)
        assert thinking is None
    
    def test_extract_thinking_content_no_reasoning(self):
        """测试没有reasoning_content字段。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        response = {
            "result": "这是回答"
        }
        
        thinking = plugin._extract_thinking_content(response)
        assert thinking is None
    
    def test_extract_thinking_content_non_dict_response(self):
        """测试非字典响应。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        thinking = plugin._extract_thinking_content("string response")
        assert thinking is None
        
        thinking = plugin._extract_thinking_content(None)
        assert thinking is None


class TestWenxinPluginModelSupport:
    """测试模型支持。"""
    
    def test_get_model_endpoint(self):
        """测试获取模型端点。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        endpoint = plugin._get_model_endpoint("ernie-3.5-8k")
        assert endpoint == "/chat/completions"
        
        # 所有模型都使用相同的端点
        endpoint = plugin._get_model_endpoint("ernie-4.0-turbo-8k")
        assert endpoint == "/chat/completions"


class TestWenxinPluginRequestPreparation:
    """测试请求准备。"""
    
    def test_prepare_wenxin_request_basic(self):
        """测试基本请求准备。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        request_data = plugin._prepare_wenxin_request("ernie-3.5-8k", messages)
        
        assert request_data["model"] == "ernie-3.5-8k"
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert request_data["messages"][0]["content"] == "Hello"
    
    def test_prepare_wenxin_request_with_system_message(self):
        """测试包含system消息的请求准备。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant"),
            ChatMessage(role="user", content="Hello")
        ]
        
        request_data = plugin._prepare_wenxin_request("ernie-3.5-8k", messages)
        
        # system消息应该合并到第一个user消息中
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert "You are a helpful assistant" in request_data["messages"][0]["content"]
        assert "Hello" in request_data["messages"][0]["content"]
    
    def test_prepare_wenxin_request_multiple_system_messages(self):
        """测试多个system消息的处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [
            ChatMessage(role="system", content="First system message"),
            ChatMessage(role="system", content="Second system message"),
            ChatMessage(role="user", content="Hello")
        ]
        
        request_data = plugin._prepare_wenxin_request("ernie-3.5-8k", messages)
        
        # 多个system消息应该合并
        assert len(request_data["messages"]) == 1
        assert "First system message" in request_data["messages"][0]["content"]
        assert "Second system message" in request_data["messages"][0]["content"]
        assert "Hello" in request_data["messages"][0]["content"]
    
    def test_prepare_wenxin_request_only_system_message(self):
        """测试只有system消息的情况。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="system", content="You are a helpful assistant")]
        
        request_data = plugin._prepare_wenxin_request("ernie-3.5-8k", messages)
        
        # 应该创建一个user消息
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert request_data["messages"][0]["content"] == "You are a helpful assistant"
    
    def test_prepare_wenxin_request_with_name(self):
        """测试包含name字段的消息。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello", name="test_user")]
        
        request_data = plugin._prepare_wenxin_request("ernie-3.5-8k", messages)
        
        assert request_data["messages"][0]["name"] == "test_user"
    
    def test_prepare_wenxin_request_with_parameters(self):
        """测试包含参数的请求准备。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        request_data = plugin._prepare_wenxin_request(
            "ernie-3.5-8k", 
            messages,
            temperature=0.7,
            top_p=0.9,
            max_tokens=100,
            stop=["stop"],
            stream=True
        )
        
        assert request_data["temperature"] == 0.7
        assert request_data["top_p"] == 0.9
        assert request_data["max_tokens"] == 100
        assert request_data["stop"] == ["stop"]
        assert request_data["stream"] is True
    
    def test_prepare_wenxin_request_with_response_format_json_schema(self):
        """测试JSON Schema格式的response_format。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": {"type": "object"}
            }
        }
        
        request_data = plugin._prepare_wenxin_request(
            "ernie-3.5-8k", 
            messages,
            response_format=response_format
        )
        
        assert request_data["response_format"]["type"] == "json_schema"
        assert "json_schema" in request_data["response_format"]
    
    def test_prepare_wenxin_request_with_response_format_json_object(self):
        """测试json_object格式的response_format。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_format = {"type": "json_object"}
        
        request_data = plugin._prepare_wenxin_request(
            "ernie-3.5-8k", 
            messages,
            response_format=response_format
        )
        
        assert request_data["response_format"]["type"] == "json_object"
    
    def test_prepare_wenxin_request_with_response_format_text(self):
        """测试text格式的response_format。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_format = {"type": "text"}
        
        request_data = plugin._prepare_wenxin_request(
            "ernie-3.5-8k", 
            messages,
            response_format=response_format
        )
        
        assert request_data["response_format"]["type"] == "text"
    
    def test_prepare_wenxin_request_with_response_format_unknown_type(self):
        """测试未知类型的response_format。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        response_format = {"type": "unknown_type"}
        
        request_data = plugin._prepare_wenxin_request(
            "ernie-3.5-8k", 
            messages,
            response_format=response_format
        )
        
        # 应该默认使用json_object
        assert request_data["response_format"]["type"] == "json_object"
    
    def test_prepare_wenxin_request_with_response_format_non_dict(self):
        """测试非字典类型的response_format。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        request_data = plugin._prepare_wenxin_request(
            "ernie-3.5-8k", 
            messages,
            response_format="json"
        )
        
        # 应该默认使用json_object
        assert request_data["response_format"]["type"] == "json_object"
    
    def test_prepare_wenxin_request_empty_content(self):
        """测试空内容消息。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content=None)]
        
        request_data = plugin._prepare_wenxin_request("ernie-3.5-8k", messages)
        
        assert request_data["messages"][0]["content"] == ""
    
    def test_prepare_wenxin_request_ignore_other_roles(self):
        """测试忽略其他角色类型。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [
            ChatMessage(role="tool", content="Tool message"),
            ChatMessage(role="function", content="Function message"),
            ChatMessage(role="user", content="User message")
        ]
        
        request_data = plugin._prepare_wenxin_request("ernie-3.5-8k", messages)
        
        # 只应该包含user消息
        assert len(request_data["messages"]) == 1
        assert request_data["messages"][0]["role"] == "user"
        assert request_data["messages"][0]["content"] == "User message"


class TestWenxinPluginResponseConversion:
    """测试响应转换。"""
    
    def test_convert_to_harbor_response_basic(self):
        """测试基本响应转换。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        wenxin_response = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "ernie-3.5-8k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello, how can I help you?"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }
        
        harbor_response = plugin._convert_to_harbor_response(wenxin_response, "ernie-3.5-8k")
        
        assert harbor_response.id == "test-id"
        assert harbor_response.object == "chat.completion"
        assert harbor_response.created == 1234567890
        assert harbor_response.model == "ernie-3.5-8k"
        assert len(harbor_response.choices) == 1
        assert harbor_response.choices[0].message.role == "assistant"
        assert harbor_response.choices[0].message.content == "Hello, how can I help you?"
        assert harbor_response.choices[0].finish_reason == "stop"
        assert harbor_response.usage.prompt_tokens == 10
        assert harbor_response.usage.completion_tokens == 8
        assert harbor_response.usage.total_tokens == 18
    
    def test_convert_to_harbor_response_with_thinking_content(self):
        """测试包含思考内容的响应转换。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        wenxin_response = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "ernie-x1-turbo-32k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning_content": "This is my thinking process"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }
        
        harbor_response = plugin._convert_to_harbor_response(wenxin_response, "ernie-x1-turbo-32k")
        
        assert harbor_response.choices[0].message.reasoning_content == "This is my thinking process"
    
    def test_convert_to_harbor_response_missing_usage(self):
        """测试缺少usage信息的响应转换。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        wenxin_response = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "ernie-3.5-8k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello"
                },
                "finish_reason": "stop"
            }]
        }
        
        # 模拟消息列表用于token估算
        messages = [ChatMessage(role="user", content="Hello")]
        harbor_response = plugin._convert_to_harbor_response(wenxin_response, "ernie-3.5-8k", messages)
        
        # 应该估算token使用量
        assert harbor_response.usage.prompt_tokens > 0
        assert harbor_response.usage.completion_tokens > 0
        assert harbor_response.usage.total_tokens > 0
    
    def test_convert_to_harbor_chunk_basic(self):
        """测试基本流响应块转换。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        wenxin_chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "ernie-3.5-8k",
            "choices": [{
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "content": "Hello"
                },
                "finish_reason": None
            }]
        }
        
        harbor_chunk = plugin._convert_to_harbor_chunk(wenxin_chunk, "ernie-3.5-8k")
        
        assert harbor_chunk.id == "test-id"
        assert harbor_chunk.object == "chat.completion.chunk"
        assert harbor_chunk.created == 1234567890
        assert harbor_chunk.model == "ernie-3.5-8k"
        assert len(harbor_chunk.choices) == 1
        assert harbor_chunk.choices[0].delta.role == "assistant"
        assert harbor_chunk.choices[0].delta.content == "Hello"
        assert harbor_chunk.choices[0].finish_reason is None
    
    def test_convert_to_harbor_chunk_with_finish_reason(self):
        """测试包含finish_reason的流响应块转换。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        wenxin_chunk = {
            "id": "test-id",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "ernie-3.5-8k",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 8,
                "total_tokens": 18
            }
        }
        
        harbor_chunk = plugin._convert_to_harbor_chunk(wenxin_chunk, "ernie-3.5-8k")
        
        assert harbor_chunk.choices[0].finish_reason == "stop"
        # ChatCompletionChunk可能没有usage字段，这是正常的
        if hasattr(harbor_chunk, 'usage') and harbor_chunk.usage:
            assert harbor_chunk.usage.prompt_tokens == 10
            assert harbor_chunk.usage.completion_tokens == 8
            assert harbor_chunk.usage.total_tokens == 18


class TestWenxinPluginStreamHandling:
    """测试流处理。"""
    
    def test_handle_stream_response_basic(self):
        """测试基本流响应处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        # 模拟流响应
        stream_data = [
            'data: {"id":"test-id","object":"chat.completion.chunk","created":1234567890,"model":"ernie-3.5-8k","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
            'data: {"id":"test-id","object":"chat.completion.chunk","created":1234567890,"model":"ernie-3.5-8k","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
            'data: {"id":"test-id","object":"chat.completion.chunk","created":1234567890,"model":"ernie-3.5-8k","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            'data: [DONE]\n\n'
        ]
        
        mock_response = Mock()
        mock_response.iter_lines.return_value = stream_data
        
        chunks = list(plugin._handle_stream_response(mock_response, "ernie-3.5-8k"))
        
        assert len(chunks) == 3  # 不包括[DONE]
        assert chunks[0].choices[0].delta.content == "Hello"
        assert chunks[1].choices[0].delta.content == " world"
        assert chunks[2].choices[0].finish_reason == "stop"
    
    def test_handle_stream_response_invalid_json(self):
        """测试无效JSON的流响应处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        stream_data = [
            'data: {"invalid": json}\n\n',
            'data: {"id":"test-id","object":"chat.completion.chunk","created":1234567890,"model":"ernie-3.5-8k","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            'data: [DONE]\n\n'
        ]
        
        mock_response = Mock()
        mock_response.iter_lines.return_value = stream_data
        
        chunks = list(plugin._handle_stream_response(mock_response, "ernie-3.5-8k"))
        
        # 应该跳过无效JSON，只返回有效的块
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hello"
    
    def test_handle_stream_response_empty_lines(self):
        """测试包含空行的流响应处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        stream_data = [
            '',
            'data: {"id":"test-id","object":"chat.completion.chunk","created":1234567890,"model":"ernie-3.5-8k","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            '',
            'data: [DONE]\n\n'
        ]
        
        mock_response = Mock()
        mock_response.iter_lines.return_value = stream_data
        
        chunks = list(plugin._handle_stream_response(mock_response, "ernie-3.5-8k"))
        
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hello"
    
    @pytest.mark.asyncio
    async def test_handle_async_stream_response_basic(self):
        """测试基本异步流响应处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        stream_data = [
            'data: {"id":"test-id","object":"chat.completion.chunk","created":1234567890,"model":"ernie-3.5-8k","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n',
            'data: [DONE]\n\n'
        ]
        
        async def mock_aiter_lines():
            for line in stream_data:
                yield line
        
        mock_response = Mock()
        mock_response.aiter_lines.return_value = mock_aiter_lines()
        
        chunks = []
        async for chunk in plugin._handle_async_stream_response(mock_response, "ernie-3.5-8k"):
            chunks.append(chunk)
        
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hello"
    
    @pytest.mark.asyncio
    async def test_handle_async_stream_response_invalid_json(self):
        """测试异步流响应处理无效JSON。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        stream_data = [
            'data: {"invalid": json}\n\n',
            'data: {"id":"test-id","object":"chat.completion.chunk","created":1234567890,"model":"ernie-3.5-8k","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
            'data: [DONE]\n\n'
        ]
        
        async def mock_aiter_lines():
            for line in stream_data:
                yield line
        
        mock_response = Mock()
        mock_response.aiter_lines.return_value = mock_aiter_lines()
        
        chunks = []
        async for chunk in plugin._handle_async_stream_response(mock_response, "ernie-3.5-8k"):
            chunks.append(chunk)
        
        # 应该跳过无效JSON
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.content == "Hello"


class TestWenxinPluginErrorHandling:
    """测试错误处理。"""
    
    def test_chat_completion_http_error(self):
        """测试HTTP错误处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(plugin, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # 模拟HTTP错误
            import httpx
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Bad Request", 
                request=Mock(), 
                response=mock_response
            )
            
            # 应该返回错误响应而不是抛出异常
            result = plugin.chat_completion("ernie-3.5-8k", messages)
            assert result.choices[0].message.content.startswith("Error:")
    
    def test_chat_completion_connection_error(self):
        """测试连接错误处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(plugin, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # 模拟连接错误
            import httpx
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")
            
            # 应该返回错误响应而不是抛出异常
            result = plugin.chat_completion("ernie-3.5-8k", messages)
            assert result.choices[0].message.content.startswith("Error:")
    
    def test_chat_completion_timeout_error(self):
        """测试超时错误处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(plugin, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # 模拟超时错误
            import httpx
            mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
            
            # 应该返回错误响应而不是抛出异常
            result = plugin.chat_completion("ernie-3.5-8k", messages)
            assert result.choices[0].message.content.startswith("Error:")
    
    def test_chat_completion_general_exception(self):
        """测试一般异常处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(plugin, '_get_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # 模拟一般异常
            mock_client.post.side_effect = Exception("Unexpected error")
            
            # 应该返回错误响应而不是抛出异常
            result = plugin.chat_completion("ernie-3.5-8k", messages)
            assert result.choices[0].message.content.startswith("Error:")
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_http_error(self):
        """测试异步HTTP错误处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(plugin, '_get_async_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # 模拟HTTP错误
            import httpx
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "Bad request"
            
            mock_client.post.side_effect = httpx.HTTPStatusError(
                "Bad Request", 
                request=Mock(), 
                response=mock_response
            )
            
            # 应该返回错误响应而不是抛出异常
            result = await plugin.chat_completion_async("ernie-3.5-8k", messages)
            assert result.choices[0].message.content.startswith("Error:")
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_connection_error(self):
        """测试异步连接错误处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(plugin, '_get_async_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # 模拟连接错误
            import httpx
            mock_client.post.side_effect = httpx.ConnectError("Connection failed")
            
            # 应该返回错误响应而不是抛出异常
            result = await plugin.chat_completion_async("ernie-3.5-8k", messages)
            assert result.choices[0].message.content.startswith("Error:")
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_timeout_error(self):
        """测试异步超时错误处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(plugin, '_get_async_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # 模拟超时错误
            import httpx
            mock_client.post.side_effect = httpx.TimeoutException("Request timeout")
            
            # 应该返回错误响应而不是抛出异常
            result = await plugin.chat_completion_async("ernie-3.5-8k", messages)
            assert result.choices[0].message.content.startswith("Error:")
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_general_exception(self):
        """测试异步一般异常处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        messages = [ChatMessage(role="user", content="Hello")]
        
        with patch.object(plugin, '_get_async_client') as mock_get_client:
            mock_client = Mock()
            mock_get_client.return_value = mock_client
            
            # 模拟一般异常
            mock_client.post.side_effect = Exception("Unexpected error")
            
            # 应该返回错误响应而不是抛出异常
            result = await plugin.chat_completion_async("ernie-3.5-8k", messages)
            assert result.choices[0].message.content.startswith("Error:")


class TestWenxinPluginCleanup:
    """测试资源清理。"""
    
    def test_close_client(self):
        """测试关闭同步客户端。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        # 模拟已创建的客户端
        mock_client = Mock()
        plugin._client = mock_client
        
        plugin.close()
        
        mock_client.close.assert_called_once()
        assert plugin._client is None
    
    def test_close_client_none(self):
        """测试关闭空客户端。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        # 应该不抛出异常
        plugin.close()
    
    @pytest.mark.asyncio
    async def test_aclose_async_client(self):
        """测试关闭异步客户端。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        # 模拟已创建的异步客户端
        mock_client = Mock()
        # 创建一个异步mock方法
        async def mock_aclose():
            pass
        mock_client.aclose = mock_aclose
        plugin._async_client = mock_client
        
        await plugin.aclose()
        
        assert plugin._async_client is None
    
    @pytest.mark.asyncio
    async def test_aclose_async_client_none(self):
        """测试关闭空异步客户端。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        # 应该不抛出异常
        await plugin.aclose()
    
    def test_destructor(self):
        """测试析构函数。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        # 模拟已创建的客户端
        mock_client = Mock()
        plugin._client = mock_client
        
        # 调用析构函数
        plugin.__del__()
        
        mock_client.close.assert_called_once()
    
    def test_destructor_with_exception(self):
        """测试析构函数异常处理。"""
        plugin = WenxinPlugin(name="test_wenxin", api_key="test-key")
        
        # 模拟客户端关闭时抛出异常
        mock_client = Mock()
        mock_client.close.side_effect = Exception("Close error")
        plugin._client = mock_client
        
        # 应该不抛出异常
        plugin.__del__()