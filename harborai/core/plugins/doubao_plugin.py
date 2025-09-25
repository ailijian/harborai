"""豆包插件实现。"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator

from ..base_plugin import BaseLLMPlugin, ModelInfo, ChatMessage, ChatCompletion, ChatCompletionChunk
from ...utils.logger import get_logger
from ...utils.exceptions import PluginError, ValidationError

logger = get_logger(__name__)


class DoubaoPlugin(BaseLLMPlugin):
    """豆包插件实现。"""
    
    def __init__(self, name: str = "doubao", **kwargs):
        """初始化豆包插件。
        
        Args:
            name: 插件名称
            **kwargs: 配置参数，包括api_key, base_url等
        """
        super().__init__(name, **kwargs)
        self.api_key = kwargs.get("api_key")
        self.base_url = kwargs.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")
        self.timeout = kwargs.get("timeout", 60)
        self.max_retries = kwargs.get("max_retries", 3)
        self.config = kwargs
        
        if not self.api_key:
            raise PluginError("doubao", "API key is required")
        
        # 初始化HTTP客户端
        self._client = None
        self._async_client = None
        
        # 支持的模型列表
        self._supported_models = [
            ModelInfo(
                id="doubao-1-5-pro-32k-character-250715",
                name="doubao-1-5-pro-32k-character-250715",
                provider="doubao",
                max_tokens=32768,
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=False
            ),
            ModelInfo(
                id="doubao-seed-1-6-250615",
                name="doubao-seed-1-6-250615",
                provider="doubao",
                max_tokens=32768,
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=True  # 1.6版本支持思考
            )
        ]
    
    @property
    def supported_models(self) -> List[ModelInfo]:
        """获取支持的模型列表。"""
        return self._supported_models
    
    def _get_client(self):
        """获取同步HTTP客户端。"""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
            except ImportError:
                raise PluginError("doubao", "httpx not installed. Please install it to use Doubao plugin.")
        return self._client
    
    def _get_async_client(self):
        """获取异步HTTP客户端。"""
        if self._async_client is None:
            try:
                import httpx
                self._async_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
            except ImportError:
                raise PluginError("doubao", "httpx not installed. Please install it to use Doubao plugin.")
        return self._async_client
    
    def _validate_request(self, model: str, messages: List[ChatMessage], **kwargs) -> None:
        """验证请求参数。"""
        # 检查模型是否支持
        if not self.supports_model(model):
            raise ValidationError(f"Model {model} is not supported by Doubao plugin")
        
        # 检查消息格式
        if not messages:
            raise ValidationError("Messages cannot be empty")
        
        # 检查API密钥
        if not self.api_key:
            raise ValidationError("Doubao API key is required")
        
        # 检查参数范围
        temperature = kwargs.get("temperature")
        if temperature is not None and not (0 <= temperature <= 2):
            raise ValidationError("Temperature must be between 0 and 2")
        
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None and max_tokens <= 0:
            raise ValidationError("max_tokens must be positive")
    
    def _extract_thinking_content(self, response: Any) -> Optional[str]:
        """提取思考内容（豆包1.6版本支持推理模型）。"""
        if isinstance(response, dict):
            # 检查是否有思考内容字段
            if 'reasoning' in response:
                return response['reasoning']
            if 'thinking' in response:
                return response['thinking']
            # 检查choices中的思考内容
            choices = response.get('choices', [])
            if choices and len(choices) > 0:
                message = choices[0].get('message', {})
                if 'reasoning_content' in message:
                    return message['reasoning_content']
                if 'thinking_content' in message:
                    return message['thinking_content']
        return None
    
    def _prepare_doubao_request(self, model: str, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """准备豆包API请求。"""
        # 转换消息格式
        doubao_messages = []
        for msg in messages:
            doubao_msg = {
                "role": msg.role,
                "content": msg.content
            }
            if msg.name:
                doubao_msg["name"] = msg.name
            if msg.tool_calls:
                doubao_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                doubao_msg["tool_call_id"] = msg.tool_call_id
            doubao_messages.append(doubao_msg)
        
        # 构建请求参数
        request_data = {
            "model": model,
            "messages": doubao_messages
        }
        
        # 添加可选参数
        optional_params = [
            "temperature", "top_p", "max_tokens", "stop", 
            "frequency_penalty", "presence_penalty", "tools", "tool_choice"
        ]
        
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                request_data[param] = kwargs[param]
        
        # 处理流式参数
        if kwargs.get("stream", False):
            request_data["stream"] = True
        
        return request_data
    
    def _convert_to_harbor_response(self, response_data: Dict[str, Any], model: str) -> ChatCompletion:
        """将豆包响应转换为Harbor格式。"""
        from ..base_plugin import ChatChoice, Usage
        
        choices = []
        for choice_data in response_data.get("choices", []):
            message_data = choice_data.get("message", {})
            
            # 提取思考内容
            reasoning_content = self._extract_thinking_content(response_data)
            
            message = ChatMessage(
                role=message_data.get("role", "assistant"),
                content=message_data.get("content"),
                name=message_data.get("name"),
                tool_calls=message_data.get("tool_calls"),
                reasoning_content=reasoning_content
            )
            
            choice = ChatChoice(
                index=choice_data.get("index", 0),
                message=message,
                finish_reason=choice_data.get("finish_reason")
            )
            choices.append(choice)
        
        # 处理使用统计
        usage_data = response_data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        return ChatCompletion(
            id=response_data.get("id", ""),
            object=response_data.get("object", "chat.completion"),
            created=response_data.get("created", 0),
            model=model,
            choices=choices,
            usage=usage
        )
    
    def _convert_to_harbor_chunk(self, chunk_data: Dict[str, Any], model: str) -> ChatCompletionChunk:
        """将豆包流式响应转换为Harbor格式。"""
        from ..base_plugin import ChatChoiceDelta, ChatChoice
        
        choices = []
        for choice_data in chunk_data.get("choices", []):
            delta_data = choice_data.get("delta", {})
            
            delta = ChatChoiceDelta(
                role=delta_data.get("role"),
                content=delta_data.get("content"),
                tool_calls=delta_data.get("tool_calls")
            )
            
            choice = ChatChoice(
                index=choice_data.get("index", 0),
                delta=delta,
                finish_reason=choice_data.get("finish_reason")
            )
            choices.append(choice)
        
        return ChatCompletionChunk(
            id=chunk_data.get("id", ""),
            object=chunk_data.get("object", "chat.completion.chunk"),
            created=chunk_data.get("created", 0),
            model=model,
            choices=choices
        )
    
    def chat_completion(self, 
                       model: str, 
                       messages: List[ChatMessage], 
                       stream: bool = False,
                       **kwargs) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """同步聊天完成。"""
        # 验证请求
        self._validate_request(model, messages, **kwargs)
        
        # 记录请求日志
        self.log_request(model, messages, **kwargs)
        
        try:
            # 准备请求
            request_data = self._prepare_doubao_request(model, messages, stream=stream, **kwargs)
            
            # 发送请求
            client = self._get_client()
            response = client.post("/chat/completions", json=request_data)
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response, model)
            else:
                start_time = time.time()
                response_data = response.json()
                harbor_response = self._convert_to_harbor_response(response_data, model)
                
                # 处理结构化输出
                response_format = kwargs.get('response_format')
                if response_format:
                    structured_provider = kwargs.get('structured_provider', 'agently')
                    harbor_response = self.handle_structured_output(harbor_response, response_format, structured_provider)
                
                # 记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(harbor_response, latency_ms)
                
                return harbor_response
                
        except Exception as e:
            logger.error(f"Doubao API error: {e}")
            error_response = self.create_error_response(str(e), model)
            self.log_response(error_response, 0)
            return error_response
    
    async def chat_completion_async(self, 
                                   model: str, 
                                   messages: List[ChatMessage], 
                                   stream: bool = False,
                                   **kwargs) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """异步聊天完成。"""
        # 验证请求
        self._validate_request(model, messages, **kwargs)
        
        # 记录请求日志
        self.log_request(model, messages, **kwargs)
        
        try:
            # 准备请求
            request_data = self._prepare_doubao_request(model, messages, stream=stream, **kwargs)
            
            # 发送请求
            client = self._get_async_client()
            response = await client.post("/chat/completions", json=request_data)
            response.raise_for_status()
            
            if stream:
                return self._handle_async_stream_response(response, model)
            else:
                start_time = time.time()
                response_data = response.json()
                harbor_response = self._convert_to_harbor_response(response_data, model)
                
                # 处理结构化输出
                response_format = kwargs.get('response_format')
                if response_format:
                    structured_provider = kwargs.get('structured_provider', 'agently')
                    harbor_response = self.handle_structured_output(harbor_response, response_format, structured_provider)
                
                # 记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(harbor_response, latency_ms)
                
                return harbor_response
                
        except Exception as e:
            logger.error(f"Doubao API error: {e}")
            error_response = self.create_error_response(str(e), model)
            self.log_response(error_response, 0)
            return error_response
    
    def _handle_stream_response(self, response, model: str) -> Generator[ChatCompletionChunk, None, None]:
        """处理同步流式响应。"""
        for line in response.iter_lines():
            if line.startswith(b"data: "):
                data = line[6:].decode('utf-8').strip()
                if data == "[DONE]":
                    break
                
                try:
                    chunk_data = json.loads(data)
                    yield self._convert_to_harbor_chunk(chunk_data, model)
                except json.JSONDecodeError:
                    continue
    
    async def _handle_async_stream_response(self, response, model: str) -> AsyncGenerator[ChatCompletionChunk, None]:
        """处理异步流式响应。"""
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                
                try:
                    chunk_data = json.loads(data)
                    yield self._convert_to_harbor_chunk(chunk_data, model)
                except json.JSONDecodeError:
                    continue
    
    def close(self):
        """关闭同步客户端。"""
        if self._client:
            self._client.close()
            self._client = None
    
    async def aclose(self):
        """关闭异步客户端。"""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
    
    def __del__(self):
        """析构函数，确保资源清理。"""
        try:
            self.close()
        except:
            pass