"""文心一言插件实现。"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator

from ..base_plugin import BaseLLMPlugin, ModelInfo, ChatMessage, ChatCompletion, ChatCompletionChunk
from ...utils.logger import get_logger
from ...utils.exceptions import PluginError, ValidationError

logger = get_logger(__name__)


class WenxinPlugin(BaseLLMPlugin):
    """文心一言插件实现。"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: Optional[str] = None, **kwargs):
        """初始化文心一言插件。
        
        Args:
            api_key: 百度API Key
            secret_key: 百度Secret Key
            base_url: API基础URL，默认为百度官方API
            **kwargs: 其他配置参数
        """
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url or "https://aip.baidubce.com"
        self.timeout = kwargs.get("timeout", 60)
        self.max_retries = kwargs.get("max_retries", 3)
        
        # 访问令牌缓存
        self._access_token = None
        self._token_expires_at = 0
        
        # 初始化HTTP客户端
        self._client = None
        self._async_client = None
        
        # 支持的模型列表
        self._supported_models = [
            ModelInfo(
                id="ernie-bot",
                name="文心一言",
                provider="wenxin",
                max_tokens=2048,
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=False
            ),
            ModelInfo(
                id="ernie-bot-turbo",
                name="文心一言 Turbo",
                provider="wenxin",
                max_tokens=2048,
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=False
            ),
            ModelInfo(
                id="ernie-bot-4",
                name="文心一言 4.0",
                provider="wenxin",
                max_tokens=4096,
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=False
            ),
            ModelInfo(
                id="ernie-3.5-8k",
                name="ERNIE 3.5 8K",
                provider="wenxin",
                max_tokens=8192,
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=False
            ),
            ModelInfo(
                id="ernie-3.5-128k",
                name="ERNIE 3.5 128K",
                provider="wenxin",
                max_tokens=131072,
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=False
            ),
            ModelInfo(
                id="ernie-lite-8k",
                name="ERNIE Lite 8K",
                provider="wenxin",
                max_tokens=8192,
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=False
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
                    timeout=self.timeout
                )
            except ImportError:
                raise PluginError("httpx not installed. Please install it to use Wenxin plugin.")
        return self._client
    
    def _get_async_client(self):
        """获取异步HTTP客户端。"""
        if self._async_client is None:
            try:
                import httpx
                self._async_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout
                )
            except ImportError:
                raise PluginError("httpx not installed. Please install it to use Wenxin plugin.")
        return self._async_client
    
    def _get_access_token(self) -> str:
        """获取访问令牌。"""
        import time
        
        # 检查令牌是否过期
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token
        
        # 获取新令牌
        client = self._get_client()
        response = client.post(
            "/oauth/2.0/token",
            params={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key
            }
        )
        response.raise_for_status()
        
        token_data = response.json()
        self._access_token = token_data["access_token"]
        self._token_expires_at = time.time() + token_data.get("expires_in", 3600) - 60  # 提前1分钟过期
        
        return self._access_token
    
    async def _get_access_token_async(self) -> str:
        """异步获取访问令牌。"""
        import time
        
        # 检查令牌是否过期
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token
        
        # 获取新令牌
        client = self._get_async_client()
        response = await client.post(
            "/oauth/2.0/token",
            params={
                "grant_type": "client_credentials",
                "client_id": self.api_key,
                "client_secret": self.secret_key
            }
        )
        response.raise_for_status()
        
        token_data = response.json()
        self._access_token = token_data["access_token"]
        self._token_expires_at = time.time() + token_data.get("expires_in", 3600) - 60  # 提前1分钟过期
        
        return self._access_token
    
    def _validate_request(self, model: str, messages: List[ChatMessage], **kwargs) -> None:
        """验证请求参数。"""
        # 检查模型是否支持
        if not self.supports_model(model):
            raise ValidationError(f"Model {model} is not supported by Wenxin plugin")
        
        # 检查消息格式
        if not messages:
            raise ValidationError("Messages cannot be empty")
        
        # 检查API密钥
        if not self.api_key or not self.secret_key:
            raise ValidationError("Wenxin API key and secret key are required")
        
        # 检查参数范围
        temperature = kwargs.get("temperature")
        if temperature is not None and not (0.01 <= temperature <= 1.0):
            raise ValidationError("Temperature must be between 0.01 and 1.0")
        
        top_p = kwargs.get("top_p")
        if top_p is not None and not (0.01 <= top_p <= 1.0):
            raise ValidationError("top_p must be between 0.01 and 1.0")
    
    def _extract_thinking_content(self, response: Any) -> Optional[str]:
        """提取思考内容（文心一言当前不支持思考模型）。"""
        return None
    
    def _get_model_endpoint(self, model: str) -> str:
        """获取模型对应的API端点。"""
        model_endpoints = {
            "ernie-bot": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
            "ernie-bot-turbo": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant",
            "ernie-bot-4": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro",
            "ernie-3.5-8k": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-8k",
            "ernie-3.5-128k": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-128k",
            "ernie-lite-8k": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k"
        }
        return model_endpoints.get(model, "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions")
    
    def _prepare_wenxin_request(self, model: str, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """准备文心一言API请求。"""
        # 转换消息格式
        wenxin_messages = []
        for msg in messages:
            wenxin_msg = {
                "role": msg.role,
                "content": msg.content
            }
            if msg.name:
                wenxin_msg["name"] = msg.name
            wenxin_messages.append(wenxin_msg)
        
        # 构建请求参数
        request_data = {
            "messages": wenxin_messages
        }
        
        # 添加可选参数
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            request_data["temperature"] = kwargs["temperature"]
        
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            request_data["top_p"] = kwargs["top_p"]
        
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            request_data["max_output_tokens"] = kwargs["max_tokens"]
        
        if "stop" in kwargs and kwargs["stop"] is not None:
            request_data["stop"] = kwargs["stop"]
        
        # 处理流式参数
        if kwargs.get("stream", False):
            request_data["stream"] = True
        
        return request_data
    
    def _convert_to_harbor_response(self, response_data: Dict[str, Any], model: str) -> ChatCompletion:
        """将文心一言响应转换为Harbor格式。"""
        from ..base_plugin import ChatChoice, Usage
        
        # 文心一言的响应格式
        content = response_data.get("result", "")
        
        message = ChatMessage(
            role="assistant",
            content=content,
            reasoning_content=None  # 文心一言不支持思考内容
        )
        
        choice = ChatChoice(
            index=0,
            message=message,
            finish_reason=response_data.get("finish_reason", "stop")
        )
        
        # 处理使用统计
        usage_data = response_data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0)
        )
        
        return ChatCompletion(
            id=response_data.get("id", ""),
            object="chat.completion",
            created=response_data.get("created", 0),
            model=model,
            choices=[choice],
            usage=usage
        )
    
    def _convert_to_harbor_chunk(self, chunk_data: Dict[str, Any], model: str) -> ChatCompletionChunk:
        """将文心一言流式响应转换为Harbor格式。"""
        from ..base_plugin import ChatChoiceDelta, ChatChoice
        
        content = chunk_data.get("result", "")
        
        delta = ChatChoiceDelta(
            role="assistant" if content else None,
            content=content
        )
        
        choice = ChatChoice(
            index=0,
            delta=delta,
            finish_reason=chunk_data.get("finish_reason")
        )
        
        return ChatCompletionChunk(
            id=chunk_data.get("id", ""),
            object="chat.completion.chunk",
            created=chunk_data.get("created", 0),
            model=model,
            choices=[choice]
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
            # 获取访问令牌
            access_token = self._get_access_token()
            
            # 准备请求
            request_data = self._prepare_wenxin_request(model, messages, stream=stream, **kwargs)
            endpoint = self._get_model_endpoint(model)
            
            # 发送请求
            client = self._get_client()
            response = client.post(
                endpoint,
                params={"access_token": access_token},
                json=request_data
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response, model)
            else:
                start_time = time.time()
                response_data = response.json()
                
                # 检查错误
                if "error_code" in response_data:
                    raise PluginError(f"Wenxin API error: {response_data.get('error_msg', 'Unknown error')}")
                
                harbor_response = self._convert_to_harbor_response(response_data, model)
                
                # 处理结构化输出
                response_format = kwargs.get('response_format')
                if response_format:
                    harbor_response = self.handle_structured_output(harbor_response, response_format)
                
                # 记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(harbor_response, latency_ms)
                
                return harbor_response
                
        except Exception as e:
            logger.error(f"Wenxin API error: {e}")
            error_response = self.create_error_response(str(e), model)
            self.log_response(error_response, success=False)
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
            # 获取访问令牌
            access_token = await self._get_access_token_async()
            
            # 准备请求
            request_data = self._prepare_wenxin_request(model, messages, stream=stream, **kwargs)
            endpoint = self._get_model_endpoint(model)
            
            # 发送请求
            client = self._get_async_client()
            response = await client.post(
                endpoint,
                params={"access_token": access_token},
                json=request_data
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_async_stream_response(response, model)
            else:
                start_time = time.time()
                response_data = response.json()
                
                # 检查错误
                if "error_code" in response_data:
                    raise PluginError(f"Wenxin API error: {response_data.get('error_msg', 'Unknown error')}")
                
                harbor_response = self._convert_to_harbor_response(response_data, model)
                
                # 处理结构化输出
                response_format = kwargs.get('response_format')
                if response_format:
                    harbor_response = self.handle_structured_output(harbor_response, response_format)
                
                # 记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(harbor_response, latency_ms)
                
                return harbor_response
                
        except Exception as e:
            logger.error(f"Wenxin API error: {e}")
            error_response = self.create_error_response(str(e), model)
            self.log_response(error_response, success=False)
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
                    if "error_code" not in chunk_data:
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
                    if "error_code" not in chunk_data:
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