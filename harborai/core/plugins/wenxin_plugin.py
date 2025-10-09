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
    
    def __init__(self, name: str, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        """初始化文心一言插件。
        
        Args:
            name: 插件名称
            api_key: 百度API Key
            base_url: API基础URL，默认为百度官方API
            **kwargs: 其他配置参数
        """
        super().__init__(name, **kwargs)
        
        # 从kwargs中获取api_key（如果没有直接传递）
        self.api_key = api_key or kwargs.get('api_key')
        
        # 统一使用OpenAI标准调用方式
        self.base_url = base_url or "https://qianfan.baidubce.com/v2"
            
        self.timeout = kwargs.get("timeout", 60)
        self.max_retries = kwargs.get("max_retries", 3)
        
        # 初始化HTTP客户端
        self._client = None
        self._async_client = None
        
        # 支持的模型列表
        self._supported_models = [
            ModelInfo(
                id="ernie-3.5-8k",
                name="文心一言3.5",
                provider="wenxin",
                max_tokens=2048,
                supports_streaming=True,
                supports_structured_output=True,
                supports_thinking=False
            ),
            ModelInfo(
                id="ernie-4.0-turbo-8k",
                name="文心一言 4.0 Turbo",
                provider="wenxin",
                max_tokens=2048,
                supports_streaming=True,
                supports_structured_output=True,
                supports_thinking=False
            ),
            # 保持向后兼容性
            ModelInfo(
                id="ernie-x1-turbo-32k",
                name="文心一言 x1 turbo 推理模型 (别名)",
                provider="wenxin",
                max_tokens=32768,
                supports_streaming=True,
                supports_structured_output=False,
                supports_thinking=True
            )
        ]
    
    @property
    def supported_models(self) -> List[ModelInfo]:
        """获取支持的模型列表。"""
        return self._supported_models
    
    def is_thinking_model(self, model: str) -> bool:
        """判断是否为推理模型。"""
        # 根据_supported_models中的supports_thinking属性判断
        for model_info in self._supported_models:
            if model_info.id == model:
                return model_info.supports_thinking
        return False
    
    def _get_client(self):
        """获取同步HTTP客户端。"""
        if self._client is None:
            try:
                import httpx
                # 百度千帆v2 API的API Key格式为 bce-v3/ALTAK-***/***
                # 直接使用Bearer + API Key的格式
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    headers=headers
                )
            except ImportError:
                raise PluginError(self.name, "httpx not installed. Please install it to use Wenxin plugin.")
        return self._client
    
    def _get_async_client(self):
        """获取异步HTTP客户端。"""
        if self._async_client is None:
            try:
                import httpx
                # 百度千帆v2 API的API Key格式为 bce-v3/ALTAK-***/***
                # 直接使用Bearer + API Key的格式
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                self._async_client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    headers=headers
                )
            except ImportError:
                raise PluginError(self.name, "httpx not installed. Please install it to use Wenxin plugin.")
        return self._async_client
    

    
    def _validate_request(self, model: str, messages: List[ChatMessage], **kwargs) -> None:
        """验证请求参数。"""
        # 检查模型是否支持
        if not self.supports_model(model):
            raise ValidationError(f"Model {model} is not supported by Wenxin plugin")
        
        # 检查消息格式
        if not messages:
            raise ValidationError("Messages cannot be empty")
        
        # 检查API密钥
        if not self.api_key:
            raise ValidationError("Wenxin API key is required")
        
        # 移除传统OAuth2.0认证方式，统一使用OpenAI标准调用方式
        
        # 检查参数范围
        temperature = kwargs.get("temperature")
        if temperature is not None and not (0.01 <= temperature <= 1.0):
            raise ValidationError("Temperature must be between 0.01 and 1.0")
        
        top_p = kwargs.get("top_p")
        if top_p is not None and not (0.01 <= top_p <= 1.0):
            raise ValidationError("top_p must be between 0.01 and 1.0")
    
    def _extract_thinking_content(self, response: Any) -> Optional[str]:
        """提取思考内容，根据TDD文档定义，直接从API响应中获取reasoning_content字段。"""
        if isinstance(response, dict):
            # 根据TDD文档，推理模型会在响应中直接提供reasoning_content字段
            if 'reasoning_content' in response and response['reasoning_content']:
                return str(response['reasoning_content'])
            
            # 检查嵌套结构中的reasoning_content
            if 'result' in response and isinstance(response['result'], dict):
                if 'reasoning_content' in response['result'] and response['result']['reasoning_content']:
                    return str(response['result']['reasoning_content'])
            
            # 检查choices结构中的reasoning_content（OpenAI格式）
            if 'choices' in response and response['choices']:
                choice = response['choices'][0]
                if isinstance(choice, dict) and 'message' in choice:
                    message = choice['message']
                    if isinstance(message, dict) and 'reasoning_content' in message:
                        return str(message['reasoning_content'])
        
        return None
    
    def _get_model_endpoint(self, model: str) -> str:
        """获取模型对应的API端点。"""
        # 统一使用OpenAI标准调用方式的端点
        return "/chat/completions"
    
    def _prepare_wenxin_request(self, model: str, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """准备文心一言API请求。"""
        # 转换消息格式，优化system消息处理
        wenxin_messages = []
        system_content = ""
        
        for msg in messages:
            # 处理空内容
            content = msg.content or ""
            
            if msg.role == "system":
                # 收集system消息内容
                if system_content:
                    system_content += "\n" + content
                else:
                    system_content = content
                continue
            elif msg.role in ["user", "assistant"]:
                # 如果是第一个user消息且有system内容，合并system内容
                if msg.role == "user" and system_content and not any(m["role"] == "user" for m in wenxin_messages):
                    content = f"{system_content}\n\n{content}"
                    system_content = ""  # 清空，避免重复添加
                
                wenxin_msg = {
                    "role": msg.role,
                    "content": content
                }
                if msg.name:
                    wenxin_msg["name"] = msg.name
                wenxin_messages.append(wenxin_msg)
            # 忽略其他角色类型（如tool、function等）
        
        # 如果只有system消息而没有其他消息，创建一个user消息
        if system_content and not wenxin_messages:
            wenxin_messages.append({
                "role": "user",
                "content": system_content
            })
        
        # 构建请求参数
        request_data = {
            "messages": wenxin_messages,
            "model": model  # OpenAI标准调用方式需要model参数
        }
        
        # 添加可选参数
        if "temperature" in kwargs and kwargs["temperature"] is not None:
            request_data["temperature"] = kwargs["temperature"]
        
        if "top_p" in kwargs and kwargs["top_p"] is not None:
            request_data["top_p"] = kwargs["top_p"]
        
        if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
            # 在OpenAI标准模式下，文心API使用max_tokens参数
            request_data["max_tokens"] = kwargs["max_tokens"]
        
        if "stop" in kwargs and kwargs["stop"] is not None:
            request_data["stop"] = kwargs["stop"]
        
        # 处理流式参数
        if kwargs.get("stream", False):
            request_data["stream"] = True
        
        # 处理结构化输出参数（response_format）
        response_format = kwargs.get("response_format")
        if response_format:
            # 根据文心大模型官方API格式处理response_format
            if isinstance(response_format, dict):
                if response_format.get("type") == "json_schema":
                    # 文心一言支持完整的json_schema格式，保持原始格式
                    request_data["response_format"] = {
                        "type": "json_schema",
                        "json_schema": response_format.get("json_schema", {})
                    }
                    logger.info(f"文心一言使用JSON Schema结构化输出")
                elif response_format.get("type") in ["json_object", "text"]:
                    # 直接使用文心支持的格式
                    request_data["response_format"] = {
                        "type": response_format["type"]
                    }
                    logger.info(f"文心一言使用{response_format['type']}格式输出")
                else:
                    # 默认使用json_object格式
                    request_data["response_format"] = {
                        "type": "json_object"
                    }
                    logger.info(f"文心一言使用默认json_object格式输出")
            else:
                # 如果response_format不是字典，默认使用json_object
                request_data["response_format"] = {
                    "type": "json_object"
                }
                logger.info(f"文心一言使用默认json_object格式输出")
        
        return request_data
    
    def _convert_to_harbor_response(self, response_data: Dict[str, Any], model: str, messages: List[ChatMessage] = None) -> ChatCompletion:
        """将文心一言响应转换为Harbor格式。"""
        from ..base_plugin import ChatChoice, Usage
        
        # OpenAI标准模式下，文心API返回标准的OpenAI格式
        if "choices" in response_data and len(response_data["choices"]) > 0:
            choice_data = response_data["choices"][0]
            message_data = choice_data.get("message", {})
            content = message_data.get("content", "")
            finish_reason = choice_data.get("finish_reason", "stop")
            # 对于推理模型，直接从API响应中获取reasoning_content
            reasoning_content = message_data.get("reasoning_content") if self.is_thinking_model(model) else None
        else:
            content = response_data.get("result", "")
            finish_reason = response_data.get("finish_reason", "stop")
            reasoning_content = None
        
        message = ChatMessage(
            role="assistant",
            content=content,
            reasoning_content=reasoning_content
        )
        
        choice = ChatChoice(
            index=0,
            message=message,
            finish_reason=finish_reason
        )
        
        # 处理使用统计
        usage_data = response_data.get("usage", {})
        
        # 如果没有usage信息，尝试估算token数量
        if not usage_data or usage_data.get("total_tokens", 0) == 0:
            # 简单估算：中文字符数约等于token数，英文单词数*1.3约等于token数
            prompt_text = " ".join([msg.content or "" for msg in messages]) if messages else ""
            completion_text = content or ""
            
            # 估算prompt tokens
            prompt_tokens = len(prompt_text) if prompt_text else 0
            
            # 估算completion tokens
            completion_tokens = len(completion_text) if completion_text else 0
            
            usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        else:
            usage = Usage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0)
            )
        
        return ChatCompletion(
            id=response_data.get("id", f"chatcmpl-{int(time.time())}"),
            object="chat.completion",
            created=response_data.get("created", int(time.time())),
            model=model,
            choices=[choice],
            usage=usage
        )
    
    def _convert_to_harbor_chunk(self, chunk_data: Dict[str, Any], model: str) -> ChatCompletionChunk:
        """将文心一言流式响应转换为Harbor格式。"""
        from ..base_plugin import ChatChoiceDelta, ChatChoice
        
        # 统一使用OpenAI标准格式处理响应
        if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
            choice_data = chunk_data["choices"][0]
            delta_data = choice_data.get("delta", {})
            content = delta_data.get("content", "")
            finish_reason = choice_data.get("finish_reason")
            # 对于推理模型，处理reasoning_content字段
            reasoning_content = delta_data.get("reasoning_content") if self.is_thinking_model(model) else None
        else:
            content = chunk_data.get("result", "")
            finish_reason = chunk_data.get("finish_reason")
            reasoning_content = None
        
        delta = ChatChoiceDelta(
            role="assistant" if content else None,
            content=content,
            reasoning_content=reasoning_content
        )
        
        choice = ChatChoice(
            index=0,
            delta=delta,
            finish_reason=finish_reason
        )
        
        return ChatCompletionChunk(
            id=chunk_data.get("id", f"chatcmpl-{int(time.time())}"),
            object="chat.completion.chunk",
            created=chunk_data.get("created", int(time.time())),
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
        
        start_time = time.time()
        try:
            # 准备请求
            request_data = self._prepare_wenxin_request(model, messages, stream=stream, **kwargs)
            endpoint = self._get_model_endpoint(model)
            
            # 发送请求（统一使用OpenAI标准调用方式）
            client = self._get_client()
            response = client.post(
                endpoint,
                json=request_data
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response, model)
            else:
                response_data = response.json()
                
                # 检查错误
                if "error_code" in response_data:
                    raise PluginError(self.name, f"Wenxin API error: {response_data.get('error_msg', 'Unknown error')}")
                
                harbor_response = self._convert_to_harbor_response(response_data, model, messages)
                
                # 处理结构化输出
                response_format = kwargs.get('response_format')
                if response_format:
                    structured_provider = kwargs.get('structured_provider', 'agently')
                    harbor_response = self.handle_structured_output(harbor_response, response_format, structured_provider, original_messages=messages, model=model)
                
                # 记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(harbor_response, latency_ms)
                
                return harbor_response
                
        except Exception as e:
            logger.error(f"Wenxin API error: {e}")
            error_response = self.create_error_response(str(e), model)
            latency_ms = (time.time() - start_time) * 1000
            self.log_response(error_response, latency_ms)
            if stream:
                # 流式模式下返回单个错误块的生成器
                from ..base_plugin import ChatChoice, ChatChoiceDelta, ChatCompletionChunk
                error_msg = str(e)
                def error_generator():
                    yield ChatCompletionChunk(
                        id="error",
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=model,
                        choices=[ChatChoice(
                            index=0,
                            delta=ChatChoiceDelta(content=error_msg),
                            finish_reason="error"
                        )]
                    )
                return error_generator()
            else:
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
        
        start_time = time.time()
        try:
            # 准备请求
            request_data = self._prepare_wenxin_request(model, messages, stream=stream, **kwargs)
            endpoint = self._get_model_endpoint(model)
            
            # 发送请求（统一使用OpenAI标准调用方式）
            client = self._get_async_client()
            response = await client.post(
                endpoint,
                json=request_data
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_async_stream_response(response, model)
            else:
                response_data = response.json()
                
                # 检查错误
                if "error_code" in response_data:
                    raise PluginError(self.name, f"Wenxin API error: {response_data.get('error_msg', 'Unknown error')}")
                
                harbor_response = self._convert_to_harbor_response(response_data, model, messages)
                
                # 处理结构化输出
                response_format = kwargs.get('response_format')
                if response_format:
                    structured_provider = kwargs.get('structured_provider', 'agently')
                    harbor_response = self.handle_structured_output(harbor_response, response_format, structured_provider, original_messages=messages, model=model)
                
                # 记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(harbor_response, latency_ms)
                
                return harbor_response
                
        except Exception as e:
            logger.error(f"Wenxin API error: {e}")
            error_response = self.create_error_response(str(e), model)
            latency_ms = (time.time() - start_time) * 1000
            self.log_response(error_response, latency_ms)
            if stream:
                # 流式模式下返回单个错误块的异步生成器
                from ..base_plugin import ChatChoice, ChatChoiceDelta, ChatCompletionChunk
                error_msg = str(e)
                async def error_async_generator():
                    yield ChatCompletionChunk(
                        id="error",
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=model,
                        choices=[ChatChoice(
                            index=0,
                            delta=ChatChoiceDelta(content=error_msg),
                            finish_reason="error"
                        )]
                    )
                return error_async_generator()
            else:
                return error_response
    
    def _handle_stream_response(self, response, model: str) -> Generator[ChatCompletionChunk, None, None]:
        """处理同步流式响应。"""
        for line in response.iter_lines():
            # 处理bytes和str两种情况
            if isinstance(line, bytes):
                line_str = line.decode('utf-8')
            else:
                line_str = line
            
            if line_str.startswith("data: "):
                data = line_str[6:].strip()
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
            # 处理字节和字符串两种情况
            if isinstance(line, bytes):
                line_str = line.decode('utf-8')
            else:
                line_str = line
            
            if line_str.startswith("data: "):
                data = line_str[6:].strip()
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