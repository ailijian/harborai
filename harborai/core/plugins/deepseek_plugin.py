"""DeepSeek插件实现。"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator

from ..base_plugin import BaseLLMPlugin, ModelInfo, ChatMessage, ChatCompletion, ChatCompletionChunk, ChatChoice, Usage
from ...utils.logger import get_logger
from ...utils.exceptions import PluginError, ValidationError
from ...utils.tracer import get_current_trace_id

logger = get_logger(__name__)


class DeepSeekPlugin(BaseLLMPlugin):
    """DeepSeek插件实现。"""
    
    def __init__(self, name: str = "deepseek", **config):
        """初始化DeepSeek插件。
        
        Args:
            name: 插件名称
            **config: 配置参数，包括api_key, base_url等
        """
        super().__init__(name, **config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.deepseek.com")
        self.timeout = config.get("timeout", 60)
        self.max_retries = config.get("max_retries", 3)
        
        # 设置支持的模型列表
        self._supported_models = [
            ModelInfo(
                id="deepseek-chat",
                name="DeepSeek Chat",
                provider="deepseek",
                max_tokens=32768,
                supports_streaming=True,
                supports_thinking=False,
                supports_structured_output=True
            ),
            ModelInfo(
                id="deepseek-reasoner",
                name="DeepSeek R1",
                provider="deepseek",
                max_tokens=32768,
                supports_streaming=True,
                supports_thinking=True,
                supports_structured_output=True
            )
        ]
        
        # 初始化HTTP客户端
        self._client = None
        self._async_client = None
        
        self.logger = get_logger(f"harborai.plugins.{name}")
    
    def is_thinking_model(self, model: str) -> bool:
        """判断是否为推理模型。"""
        for model_info in self._supported_models:
            if model_info.id == model:
                return model_info.supports_thinking
        return False
    
    def _get_client(self):
        """获取同步HTTP客户端。"""
        if self._client is None:
            try:
                import httpx
                # 调试信息：显示配置
                masked_key = f"{self.api_key[:6]}...{self.api_key[-4:]}" if self.api_key and len(self.api_key) > 10 else "无效密钥"
                self.logger.info(f"DeepSeek 插件配置 - Base URL: {self.base_url}, API Key: {masked_key}")
                
                self._client = httpx.Client(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream, application/json"
                    },
                    # 禁用响应缓冲以支持流式传输
                    follow_redirects=True,
                    # 优化流式传输的配置
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                    # 禁用HTTP/2以避免潜在的流式问题
                    http2=False
                )
            except ImportError:
                raise PluginError("deepseek", "httpx not installed. Please install it to use DeepSeek plugin.")
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
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream, application/json"
                    },
                    # 禁用响应缓冲以支持流式传输
                    follow_redirects=True,
                    # 优化流式传输的配置
                    limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
                    # 禁用HTTP/2以避免潜在的流式问题
                    http2=False
                )
            except ImportError:
                raise PluginError("deepseek", "httpx not installed. Please install it to use DeepSeek plugin.")
        return self._async_client
    
    def _validate_request(self, model: str, messages: List[ChatMessage], **kwargs) -> None:
        """验证请求参数。"""
        # 检查模型是否支持
        if not self.supports_model(model):
            raise ValidationError(f"Model {model} is not supported by DeepSeek plugin")
        
        # 检查消息格式
        if not messages:
            raise ValidationError("Messages cannot be empty")
        
        # 检查API密钥
        if not self.api_key:
            raise ValidationError("DeepSeek API key is required")
        
        # 检查参数范围
        temperature = kwargs.get("temperature")
        if temperature is not None and not (0 <= temperature <= 2):
            raise ValidationError("Temperature must be between 0 and 2")
        
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None and max_tokens <= 0:
            raise ValidationError("max_tokens must be positive")
    
    def _extract_thinking_content(self, response: Any) -> Optional[str]:
        """提取思考内容。"""
        if not isinstance(response, dict):
            return None
        
        # 尝试从不同字段提取思考内容
        thinking_fields = ['reasoning', 'thinking', 'reasoning_content', 'thinking_content']
        for field in thinking_fields:
            if field in response and response[field]:
                return response[field]
        
        # 尝试从choices中提取
        choices = response.get('choices', [])
        if choices and len(choices) > 0:
            message = choices[0].get('message', {})
            for field in thinking_fields:
                if field in message and message[field]:
                    return message[field]
        
        return None
    
    def _prepare_deepseek_request(self, model: str, messages: List[ChatMessage], **kwargs) -> Dict[str, Any]:
        """准备DeepSeek API请求。"""
        # 转换消息格式
        deepseek_messages = []
        for msg in messages:
            deepseek_msg = {
                "role": msg.role,
                "content": msg.content
            }
            if msg.name:
                deepseek_msg["name"] = msg.name
            if msg.tool_calls:
                deepseek_msg["tool_calls"] = msg.tool_calls
            if msg.tool_call_id:
                deepseek_msg["tool_call_id"] = msg.tool_call_id
            deepseek_messages.append(deepseek_msg)
        
        # 构建请求参数
        request_data = {
            "model": model,
            "messages": deepseek_messages
        }
        
        # 添加可选参数
        optional_params = [
            "temperature", "top_p", "max_tokens", "stop", 
            "frequency_penalty", "presence_penalty", "tools", "tool_choice"
        ]
        
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                request_data[param] = kwargs[param]
        
        # 处理结构化输出参数（response_format）
        response_format = kwargs.get("response_format")
        
        if response_format:
            # DeepSeek仅使用官方原生json_object能力，不使用Agently后处理
            if isinstance(response_format, dict):
                if response_format.get("type") == "json_schema":
                    # 对于json_schema请求，使用DeepSeek原生json_object格式
                    request_data["response_format"] = {
                        "type": "json_object"
                    }
                    
                    # 确保prompt中包含"json"关键词（DeepSeek API要求）
                    self._ensure_json_keyword_in_prompt(deepseek_messages)
                    logger.info(f"DeepSeek使用原生json_object结构化输出")
                    
                elif response_format.get("type") in ["json_object", "text"]:
                    # 直接使用DeepSeek支持的格式
                    request_data["response_format"] = {
                        "type": response_format["type"]
                    }
                    
                    # 如果是json_object，确保prompt包含"json"
                    if response_format["type"] == "json_object":
                        self._ensure_json_keyword_in_prompt(deepseek_messages)
                    
                    logger.info(f"DeepSeek使用{response_format['type']}格式输出")
                else:
                    # 默认使用json_object格式
                    request_data["response_format"] = {
                        "type": "json_object"
                    }
                    
                    # 确保prompt包含"json"
                    self._ensure_json_keyword_in_prompt(deepseek_messages)
                    logger.info(f"DeepSeek使用默认json_object格式输出")
            else:
                # 如果response_format不是字典，默认使用json_object
                request_data["response_format"] = {
                    "type": "json_object"
                }
                
                # 确保prompt包含"json"
                self._ensure_json_keyword_in_prompt(deepseek_messages)
                logger.info(f"DeepSeek使用默认json_object格式输出")
        
        # 处理流式参数
        if kwargs.get("stream", False):
            request_data["stream"] = True
        
        return request_data
    
    def _ensure_json_keyword_in_prompt(self, deepseek_messages: List[Dict[str, Any]]):
        """确保prompt中包含'json'关键词，这是DeepSeek API使用json_object格式的要求。"""
        # 检查最后一条用户消息是否包含"json"关键词
        for msg in reversed(deepseek_messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and "json" not in content.lower():
                    # 在用户消息末尾添加json格式要求
                    msg["content"] = content + "\n\nReturn only raw JSON with no extra text, markdown, or explanation."
                    logger.info("已在prompt中添加JSON格式要求")
                break
    
    def _convert_to_harbor_response(self, response_data: Dict[str, Any], model: str) -> ChatCompletion:
        """将DeepSeek响应转换为Harbor格式。"""
        from ..base_plugin import ChatChoice, Usage
        
        choices = []
        for choice_data in response_data.get("choices", []):
            message_data = choice_data.get("message", {})
            
            # 对于推理模型，提取思考内容
            reasoning_content = None
            if self.is_thinking_model(model):
                reasoning_content = message_data.get("reasoning_content")
                if not reasoning_content:
                    # 尝试从其他字段提取思考内容
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
        """将DeepSeek流式响应转换为Harbor格式。"""
        from ..base_plugin import ChatChoiceDelta
        
        choices = []
        for choice_data in chunk_data.get("choices", []):
            delta_data = choice_data.get("delta", {})
            
            # 对于推理模型，处理思考内容
            reasoning_content = None
            if self.is_thinking_model(model):
                reasoning_content = delta_data.get("reasoning_content")
            
            # 创建delta消息
            delta = ChatChoiceDelta(
                role=delta_data.get("role"),
                content=delta_data.get("content"),
                tool_calls=delta_data.get("tool_calls"),
                reasoning_content=reasoning_content
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
    
    def _handle_native_structured_output(self, model: str, messages: List[ChatMessage], stream: bool = False, **kwargs):
        """处理DeepSeek原生结构化输出。
        
        直接使用DeepSeek的json_object能力，无需Agently后处理。
        """
        response_format = kwargs.get('response_format')
        
        logger.info(f"使用DeepSeek原生json_object处理模型 {model} 的结构化输出")
        
        try:
            # 设置使用原生结构化输出标志
            kwargs['use_native_structured'] = True
            
            # 准备包含response_format的请求
            request_data = self._prepare_deepseek_request(model, messages, stream=stream, **kwargs)
            
            # 发送请求到标准端点
            client = self._get_client()
            # 构建URL路径 - 检查base_url是否已包含/v1
            if self.base_url.endswith('/v1'):
                url_path = "/chat/completions"
            else:
                url_path = "/v1/chat/completions"
            response = client.post(url_path, json=request_data)
            response.raise_for_status()
            
            if stream:
                return self._handle_stream_response(response, model)
            else:
                import time
                start_time = time.time()
                response_data = response.json()
                harbor_response = self._convert_to_harbor_response(response_data, model)
                
                # 检查是否成功返回结构化输出
                if (harbor_response.choices and 
                    harbor_response.choices[0].message and 
                    harbor_response.choices[0].message.content):
                    
                    content = harbor_response.choices[0].message.content
                    
                    # 验证返回的内容是否为有效JSON
                    try:
                        parsed_json = json.loads(content)
                        logger.info(f"DeepSeek模型 {model} 原生结构化输出成功，返回有效JSON")
                        
                        # 设置parsed字段到message对象上
                        harbor_response.choices[0].message.parsed = parsed_json
                        
                        # 计算延迟并记录响应日志
                        latency_ms = (time.time() - start_time) * 1000
                        self.log_response(harbor_response, latency_ms)
                        
                        return harbor_response
                    except json.JSONDecodeError as e:
                        logger.warning(f"DeepSeek模型 {model} 返回的内容JSON解析失败: {e}")
                        logger.warning(f"返回内容长度: {len(content)}")
                        logger.warning(f"返回内容前500字符: {content[:500]}")
                        logger.warning(f"返回内容后100字符: {content[-100:]}")
                        
                        # 尝试修复常见的JSON问题
                        try:
                            # 尝试去除可能的前后空白字符
                            cleaned_content = content.strip()
                            parsed_json = json.loads(cleaned_content)
                            logger.info(f"DeepSeek模型 {model} JSON清理后解析成功")
                            
                            # 设置parsed字段到message对象上
                            harbor_response.choices[0].message.parsed = parsed_json
                            
                            # 计算延迟并记录响应日志
                            latency_ms = (time.time() - start_time) * 1000
                            self.log_response(harbor_response, latency_ms)
                            
                            return harbor_response
                        except json.JSONDecodeError:
                            # 如果仍然失败，返回原始响应而不是抛出错误
                            logger.warning(f"DeepSeek模型 {model} JSON解析最终失败，返回原始响应")
                            
                            # 确保parsed字段不存在
                            if hasattr(harbor_response.choices[0].message, 'parsed'):
                                delattr(harbor_response.choices[0].message, 'parsed')
                            
                            # 计算延迟并记录响应日志
                            latency_ms = (time.time() - start_time) * 1000
                            self.log_response(harbor_response, latency_ms)
                            
                            return harbor_response
                else:
                    logger.error(f"DeepSeek模型 {model} 返回了无效的响应内容")
                    raise PluginError("deepseek", "DeepSeek返回了无效的响应内容")
                    
        except Exception as e:
            logger.error(f"DeepSeek原生结构化输出处理失败: {str(e)}")
            raise PluginError("deepseek", f"DeepSeek原生结构化输出处理失败: {str(e)}")

    def chat_completion(self, 
                       model: str, 
                       messages: List[ChatMessage], 
                       stream: bool = False,
                       **kwargs) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """同步聊天完成。"""
        import time
        start_time = time.time()
        
        # 验证请求
        self._validate_request(model, messages, **kwargs)
        
        # 记录请求日志
        self.log_request(model, messages, **kwargs)
        
        try:
            # 检查是否使用原生结构化输出
            response_format = kwargs.get('response_format')
            structured_provider = kwargs.get('structured_provider', 'agently')
            use_native_structured = response_format and structured_provider == 'native'
            
            if use_native_structured:
                # DeepSeek原生结构化输出：直接使用json_object，无需Agently后处理
                logger.info(f"使用DeepSeek原生结构化输出处理模型 {model}")
                return self._handle_native_structured_output(model, messages, stream, **kwargs)
            else:
                # 标准请求处理（使用Agently后处理）
                request_data = self._prepare_deepseek_request(model, messages, stream=stream, **kwargs)
                
                # 发送请求
                client = self._get_client()
                
                # 调试信息：显示请求详情
                # 构建完整URL - 检查base_url是否已包含/v1
                if self.base_url.endswith('/v1'):
                    full_url = f"{self.base_url}/chat/completions"
                else:
                    full_url = f"{self.base_url}/v1/chat/completions"
                self.logger.info(f"发送请求到: {full_url}")
                self.logger.info(f"请求模型: {model}")
                
                # 构建URL路径 - 检查base_url是否已包含/v1
                if self.base_url.endswith('/v1'):
                    url_path = "/chat/completions"
                else:
                    url_path = "/v1/chat/completions"
                
                # 为流式请求添加特殊配置
                if stream:
                    # 流式请求需要特殊的headers和配置
                    stream_headers = {
                        "Accept": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"  # 禁用Nginx缓冲
                    }
                    # 使用stream方法进行真正的流式请求
                    return self._handle_stream_request(client, url_path, request_data, stream_headers, model)
                else:
                    response = client.post(url_path, json=request_data)
                
                # 调试信息：显示响应状态
                self.logger.info(f"响应状态码: {response.status_code}")
                if response.status_code != 200:
                    self.logger.error(f"响应内容: {response.text}")
                
                response.raise_for_status()
                
                response_data = response.json()
                harbor_response = self._convert_to_harbor_response(response_data, model)
                
                # 处理结构化输出（使用Agently后处理）
                self.logger.debug(f"DeepSeek插件检查结构化输出: response_format={response_format}")
                if response_format:
                    self.logger.debug(f"DeepSeek插件调用handle_structured_output: structured_provider={structured_provider}")
                    harbor_response = self.handle_structured_output(harbor_response, response_format, structured_provider, model=model, original_messages=messages)
                
                # 计算延迟并记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(harbor_response, latency_ms)
                
                return harbor_response
                
        except Exception as e:
            self.logger.error(f"DeepSeek API error: {e}")
            # 计算延迟
            latency_ms = (time.time() - start_time) * 1000
            # 记录错误信息
            self.logger.error(f"DeepSeek同步请求失败 [trace_id={get_current_trace_id()}] model={model} error={str(e)} latency_ms={latency_ms}")
            # 抛出异常而不是返回错误响应，让上层处理
            raise

    def _handle_stream_request(self, client, url_path: str, request_data: dict, headers: dict, model: str) -> Generator[ChatCompletionChunk, None, None]:
        """处理同步流式请求。"""
        with client.stream("POST", url_path, json=request_data, headers=headers, timeout=None) as response:
            # 调试信息：显示响应状态
            self.logger.info(f"响应状态码: {response.status_code}")
            if response.status_code != 200:
                self.logger.error(f"响应内容: {response.text}")
            
            response.raise_for_status()
            
            for line in response.iter_lines():
                # 处理字节和字符串类型的line
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk_data = json.loads(data)
                        yield self._convert_to_harbor_chunk(chunk_data, model)
                    except json.JSONDecodeError:
                        continue

    async def chat_completion_async(self,
                                   model: str, 
                                   messages: List[ChatMessage], 
                                   stream: bool = False,
                                   **kwargs) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """异步聊天完成。"""
        import time
        start_time = time.time()
        
        # 验证请求
        self._validate_request(model, messages, **kwargs)
        
        # 记录请求日志
        self.log_request(model, messages, **kwargs)
        
        try:
            # 检查是否使用原生结构化输出
            response_format = kwargs.get('response_format')
            structured_provider = kwargs.get('structured_provider', 'agently')
            use_native_structured = response_format and structured_provider == 'native'
            
            if use_native_structured:
                # DeepSeek原生结构化输出：直接使用json_object，无需Agently后处理
                logger.info(f"使用DeepSeek原生结构化输出处理模型 {model}")
                return self._handle_native_structured_output(model, messages, stream, **kwargs)
            else:
                # 标准请求处理（使用Agently后处理）
                request_data = self._prepare_deepseek_request(model, messages, stream=stream, **kwargs)
                
                # 发送请求
                client = self._get_async_client()
                
                # 调试信息：显示请求详情
                # 构建完整URL - 检查base_url是否已包含/v1
                if self.base_url.endswith('/v1'):
                    full_url = f"{self.base_url}/chat/completions"
                else:
                    full_url = f"{self.base_url}/v1/chat/completions"
                self.logger.info(f"发送请求到: {full_url}")
                self.logger.info(f"请求模型: {model}")
                
                # 构建URL路径 - 检查base_url是否已包含/v1
                if self.base_url.endswith('/v1'):
                    url_path = "/chat/completions"
                else:
                    url_path = "/v1/chat/completions"
                
                # 为流式请求添加特殊配置
                if stream:
                    # 流式请求需要特殊的headers和配置
                    stream_headers = {
                        "Accept": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no"  # 禁用Nginx缓冲
                    }
                    # 使用stream方法进行真正的流式请求
                    return self._handle_async_stream_request(client, url_path, request_data, stream_headers, model)
                else:
                    response = await client.post(url_path, json=request_data)
                
                # 调试信息：显示响应状态
                self.logger.info(f"响应状态码: {response.status_code}")
                if response.status_code != 200:
                    self.logger.error(f"响应内容: {response.text}")
                
                response.raise_for_status()
                
                response_data = response.json()
                harbor_response = self._convert_to_harbor_response(response_data, model)
                
                # 处理结构化输出（使用Agently后处理）
                self.logger.debug(f"DeepSeek插件检查结构化输出: response_format={response_format}")
                if response_format:
                    self.logger.debug(f"DeepSeek插件调用handle_structured_output: structured_provider={structured_provider}")
                    harbor_response = self.handle_structured_output(harbor_response, response_format, structured_provider, model=model, original_messages=messages)
                
                # 计算延迟并记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(harbor_response, latency_ms)
                
                return harbor_response
                
        except Exception as e:
            self.logger.error(f"DeepSeek API error: {e}")
            # 计算延迟
            latency_ms = (time.time() - start_time) * 1000
            # 记录错误信息
            self.logger.error(f"DeepSeek异步请求失败 [trace_id={get_current_trace_id()}] model={model} error={str(e)} latency_ms={latency_ms}")
            # 抛出异常而不是返回错误响应，让上层处理
            raise
    
    async def _handle_async_stream_request(self, client, url_path: str, request_data: dict, headers: dict, model: str) -> AsyncGenerator[ChatCompletionChunk, None]:
        """处理异步流式请求。"""
        async with client.stream("POST", url_path, json=request_data, headers=headers, timeout=None) as response:
            # 调试信息：显示响应状态
            self.logger.info(f"响应状态码: {response.status_code}")
            if response.status_code != 200:
                self.logger.error(f"响应内容: {response.aread()}")
            
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                # 处理字节和字符串类型的line
                if isinstance(line, bytes):
                    line = line.decode('utf-8')
                
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    
                    try:
                        chunk_data = json.loads(data)
                        yield self._convert_to_harbor_chunk(chunk_data, model)
                    except json.JSONDecodeError:
                        continue
 
    def _handle_stream_response(self, response, model: str) -> Generator[ChatCompletionChunk, None, None]:
        """处理同步流式响应。"""
        for line in response.iter_lines():
            # 处理字节和字符串类型的line
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            
            if line.startswith("data: "):
                data = line[6:].strip()
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
            # 处理字节和字符串类型的line
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            
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