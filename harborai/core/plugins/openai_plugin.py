#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 插件

实现 OpenAI API 的调用逻辑，支持推理模型和结构化输出。
"""

import json
import time
from typing import Dict, List, Optional, Union, Any, AsyncGenerator, Iterator, Generator
from dataclasses import dataclass, asdict

import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion as OpenAIChatCompletion, ChatCompletionChunk as OpenAIChatCompletionChunk

from ..base_plugin import BaseLLMPlugin, ModelInfo, ChatMessage, ChatChoice, ChatChoiceDelta, Usage, ChatCompletion, ChatCompletionChunk
from ...utils.exceptions import APIError, AuthenticationError, RateLimitError, TimeoutError
from ...utils.logger import get_logger
from ...utils.tracer import get_current_trace_id


class OpenAIPlugin(BaseLLMPlugin):
    """OpenAI 插件"""
    
    def __init__(self, name: str = "openai", **config):
        super().__init__(name, **config)
        
        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            timeout=config.get("timeout", 30),
            max_retries=config.get("max_retries", 3)
        )
        
        self.async_client = AsyncOpenAI(
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            timeout=config.get("timeout", 30),
            max_retries=config.get("max_retries", 3)
        )
        
        self.logger = get_logger(f"harborai.plugins.{name}")
    
    @property
    def supported_models(self) -> List[ModelInfo]:
        """支持的模型列表"""
        return [
            ModelInfo(
                id="gpt-4o",
                name="gpt-4o",
                provider="openai",
                supports_thinking=False,
                supports_structured_output=True,
                max_tokens=128000,
                context_window=128000
            ),
            ModelInfo(
                id="gpt-4o-mini",
                name="gpt-4o-mini",
                provider="openai",
                supports_thinking=False,
                supports_structured_output=True,
                max_tokens=16384,
                context_window=128000
            ),
            ModelInfo(
                id="gpt-4-turbo",
                name="gpt-4-turbo",
                provider="openai",
                supports_thinking=False,
                supports_structured_output=True,
                max_tokens=4096,
                context_window=128000
            ),
            ModelInfo(
                id="gpt-5",
                name="gpt-5",
                provider="openai",
                supports_thinking=False,
                supports_structured_output=True,
                max_tokens=4096,
                context_window=16385
            ),
            ModelInfo(
                id="o1-preview",
                name="o1-preview",
                provider="openai",
                supports_thinking=True,
                supports_structured_output=False,
                max_tokens=32768,
                context_window=128000
            ),
            ModelInfo(
                id="o1-mini",
                name="o1-mini",
                provider="openai",
                supports_thinking=True,
                supports_structured_output=False,
                max_tokens=65536,
                context_window=128000
            )
        ]
    
    def _handle_openai_error(self, error: Exception) -> Exception:
        """处理 OpenAI 错误"""
        if isinstance(error, openai.AuthenticationError):
            return AuthenticationError(
                "OpenAI authentication failed",
                trace_id=get_current_trace_id(),
                details={"original_error": str(error)}
            )
        elif isinstance(error, openai.RateLimitError):
            return RateLimitError(
                "OpenAI rate limit exceeded",
                trace_id=get_current_trace_id(),
                details={"original_error": str(error)}
            )
        elif isinstance(error, openai.APITimeoutError):
            return TimeoutError(
                "OpenAI API timeout",
                trace_id=get_current_trace_id(),
                details={"original_error": str(error)}
            )
        elif isinstance(error, openai.APIError):
            return APIError(
                f"OpenAI API error: {error}",
                trace_id=get_current_trace_id(),
                details={"original_error": str(error)}
            )
        else:
            return error
    
    def _prepare_openai_request(
        self,
        model: str,
        messages: List[ChatMessage],
        **kwargs
    ) -> Dict[str, Any]:
        """准备 OpenAI 请求参数"""
        # 转换ChatMessage为字典格式
        message_dicts = []
        for msg in messages:
            msg_dict = {
                "role": msg.role,
                "content": msg.content
            }
            if msg.name:
                msg_dict["name"] = msg.name
            if msg.function_call:
                msg_dict["function_call"] = msg.function_call
            if msg.tool_calls:
                msg_dict["tool_calls"] = msg.tool_calls
            message_dicts.append(msg_dict)
        
        # 基础参数
        request_params = {
            "model": model,
            "messages": message_dicts
        }
        
        # 添加支持的参数
        supported_params = [
            "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
            "max_tokens", "n", "presence_penalty", "response_format",
            "seed", "stop", "stream", "temperature", "tool_choice",
            "tools", "top_p", "user"
        ]
        
        for param in supported_params:
            if param in kwargs and kwargs[param] is not None:
                request_params[param] = kwargs[param]
        
        # 处理结构化输出
        if "response_format" in kwargs and kwargs["response_format"]:
            response_format = kwargs["response_format"]
            if isinstance(response_format, dict) and "type" in response_format:
                if response_format["type"] == "json_schema":
                    request_params["response_format"] = response_format
        
        return request_params
    
    def _convert_to_harbor_response(
        self,
        openai_response: OpenAIChatCompletion
    ) -> ChatCompletion:
        """转换 OpenAI 响应为 Harbor 格式"""
        choices = []
        for choice in openai_response.choices:
            # 创建消息对象
            message = ChatMessage(
                role=choice.message.role,
                content=choice.message.content
            )
            
            # 添加工具调用
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                message.tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in choice.message.tool_calls
                ]
            
            # 添加思考内容（对于 o1 模型）
            if hasattr(choice.message, 'reasoning_content'):
                message.reasoning_content = choice.message.reasoning_content
            
            harbor_choice = ChatChoice(
                index=choice.index,
                message=message,
                finish_reason=choice.finish_reason
            )
            
            choices.append(harbor_choice)
        
        # 创建使用统计
        usage = Usage(
            prompt_tokens=openai_response.usage.prompt_tokens,
            completion_tokens=openai_response.usage.completion_tokens,
            total_tokens=openai_response.usage.total_tokens
        )
        
        # 构建响应
        response = ChatCompletion(
            id=openai_response.id,
            object=openai_response.object,
            created=openai_response.created,
            model=openai_response.model,
            choices=choices,
            usage=usage
        )
        
        # 添加系统指纹
        if hasattr(openai_response, 'system_fingerprint'):
            response.system_fingerprint = openai_response.system_fingerprint
        
        return response
    
    def chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """同步聊天完成"""
        start_time = time.time()
        try:
            # 验证请求
            self.validate_request(model, messages, **kwargs)
            
            # 记录请求日志
            self.log_request(model, messages, **kwargs)
            
            # 准备请求参数
            request_params = self._prepare_openai_request(model, messages, **kwargs)
            
            # 发送请求
            if kwargs.get("stream", False):
                # 流式响应
                stream = self.client.chat.completions.create(**request_params)
                return self._handle_stream_response(stream, **kwargs)
            else:
                # 非流式响应
                response = self.client.chat.completions.create(**request_params)
                harbor_response = self._convert_to_harbor_response(response)
                
                # 处理结构化输出
                response_format = kwargs.get('response_format')
                if response_format:
                    structured_provider = kwargs.get('structured_provider', 'agently')
                    harbor_response = self.handle_structured_output(harbor_response, response_format, structured_provider)
                
                # 计算耗时并记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(harbor_response, latency_ms)
                
                return harbor_response
                
        except Exception as e:
            self.logger.error(
                "OpenAI chat completion failed",
                trace_id=get_current_trace_id(),
                model=model,
                error=str(e)
            )
            raise self._handle_openai_error(e)
    
    async def chat_completion_async(
        self,
        model: str,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """异步聊天完成"""
        start_time = time.time()
        try:
            # 验证请求
            self.validate_request(model, messages, **kwargs)
            
            # 记录请求日志
            self.log_request(model, messages, **kwargs)
            
            # 准备请求参数
            request_params = self._prepare_openai_request(model, messages, **kwargs)
            
            # 发送请求
            if kwargs.get("stream", False):
                # 流式响应
                stream = await self.async_client.chat.completions.create(**request_params)
                return self._handle_async_stream_response(stream, **kwargs)
            else:
                # 非流式响应
                response = await self.async_client.chat.completions.create(**request_params)
                harbor_response = self._convert_to_harbor_response(response)
                
                # 处理结构化输出
                response_format = kwargs.get('response_format')
                if response_format:
                    structured_provider = kwargs.get('structured_provider', 'agently')
                    harbor_response = self.handle_structured_output(harbor_response, response_format, structured_provider)
                
                # 计算耗时并记录响应日志
                latency_ms = (time.time() - start_time) * 1000
                self.log_response(harbor_response, latency_ms)
                
                return harbor_response
                
        except Exception as e:
            self.logger.error(
                "Async OpenAI chat completion failed",
                trace_id=get_current_trace_id(),
                model=model,
                error=str(e)
            )
            raise self._handle_openai_error(e)
    
    def _handle_stream_response(self, stream, **kwargs) -> Generator[ChatCompletionChunk, None, None]:
        """处理流式响应"""
        response_format = kwargs.get('response_format')
        structured_provider = kwargs.get('structured_provider', 'agently')
        
        if response_format and response_format.get("type") == "json_schema":
            # 流式结构化输出
            return self._handle_streaming_structured_output(stream, response_format, structured_provider)
        else:
            # 普通流式输出
            for chunk in stream:
                yield self._convert_chunk_to_harbor_format(chunk)
    
    async def _handle_async_stream_response(self, stream, **kwargs) -> AsyncGenerator[ChatCompletionChunk, None]:
        """处理异步流式响应"""
        response_format = kwargs.get('response_format')
        structured_provider = kwargs.get('structured_provider', 'agently')
        
        if response_format and response_format.get("type") == "json_schema":
            # 异步流式结构化输出
            async for chunk in self._handle_async_streaming_structured_output(stream, response_format, structured_provider):
                yield chunk
        else:
            # 普通异步流式输出
            async for chunk in stream:
                yield self._convert_chunk_to_harbor_format(chunk)
    
    def _handle_streaming_structured_output(self, stream, response_format: Dict[str, Any], structured_provider: str) -> Generator[ChatCompletionChunk, None, None]:
        """处理流式结构化输出"""
        try:
            from ..api.structured import default_handler
            
            # 创建内容流生成器
            def content_stream():
                for chunk in stream:
                    harbor_chunk = self._convert_chunk_to_harbor_format(chunk)
                    if harbor_chunk.choices and harbor_chunk.choices[0].delta and harbor_chunk.choices[0].delta.content:
                        yield harbor_chunk.choices[0].delta.content
            
            # 使用结构化输出处理器解析流式内容
            schema = response_format.get('json_schema', {})
            use_agently = structured_provider == 'agently'
            parsed_stream = default_handler.parse_streaming_response(
                content_stream(), 
                schema, 
                use_agently, 
                self.api_key, 
                self.base_url, 
                self.model
            )
            
            # 将解析结果转换为ChatCompletionChunk格式
            for parsed_data in parsed_stream:
                yield ChatCompletionChunk(
                    id=f"chatcmpl-{int(time.time())}",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="structured-output",
                    choices=[
                        ChatChoice(
                            index=0,
                            delta=ChatChoiceDelta(
                                role="assistant",
                                content=str(parsed_data)
                            ),
                            finish_reason=None
                        )
                    ]
                )
                
        except Exception as e:
            self.logger.error(f"Streaming structured output failed: {e}")
            # 回退到普通流式输出
            for chunk in stream:
                yield self._convert_chunk_to_harbor_format(chunk)
    
    async def _handle_async_streaming_structured_output(self, stream, response_format: Dict[str, Any], structured_provider: str) -> AsyncGenerator[ChatCompletionChunk, None]:
        """处理异步流式结构化输出"""
        try:
            from ..api.structured import default_handler
            
            # 创建异步内容流生成器
            async def async_content_stream():
                async for chunk in stream:
                    harbor_chunk = self._convert_chunk_to_harbor_format(chunk)
                    if harbor_chunk.choices and harbor_chunk.choices[0].delta and harbor_chunk.choices[0].delta.content:
                        yield harbor_chunk.choices[0].delta.content
            
            # 使用结构化输出处理器解析异步流式内容
            schema = response_format.get('json_schema', {})
            use_agently = structured_provider == 'agently'
            parsed_stream = default_handler.parse_streaming_response(
                async_content_stream(), 
                schema, 
                use_agently, 
                self.api_key, 
                self.base_url, 
                self.model
            )
            
            # 将解析结果转换为ChatCompletionChunk格式
            async for parsed_data in parsed_stream:
                yield ChatCompletionChunk(
                    id=f"chatcmpl-{int(time.time())}",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="structured-output",
                    choices=[
                        ChatChoice(
                            index=0,
                            delta=ChatChoiceDelta(
                                role="assistant",
                                content=str(parsed_data)
                            ),
                            finish_reason=None
                        )
                    ]
                )
                
        except Exception as e:
            self.logger.error(f"Async streaming structured output failed: {e}")
            # 回退到普通异步流式输出
            async for chunk in stream:
                yield self._convert_chunk_to_harbor_format(chunk)
    
    def _convert_chunk_to_harbor_format(self, chunk: OpenAIChatCompletionChunk) -> ChatCompletionChunk:
        """转换流式响应块为 Harbor 格式"""
        choices = []
        
        for choice in chunk.choices:
            delta = choice.delta
            
            # 处理工具调用
            tool_calls = None
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                tool_calls = []
                for tool_call in delta.tool_calls:
                    tool_calls.append({
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    })
            
            # 创建 ChatChoiceDelta 对象作为 delta
            delta_obj = ChatChoiceDelta(
                role=getattr(delta, 'role', None),
                content=getattr(delta, 'content', None),
                tool_calls=tool_calls
            )
            
            # 添加思考内容（对于 o1 模型的流式响应）
            if hasattr(delta, 'reasoning_content'):
                delta_obj.reasoning_content = delta.reasoning_content
            
            # 创建 ChatChoice 对象
            harbor_choice = ChatChoice(
                index=choice.index,
                delta=delta_obj,
                finish_reason=choice.finish_reason
            )
            
            choices.append(harbor_choice)
        
        # 返回 ChatCompletionChunk 对象
        return ChatCompletionChunk(
            id=chunk.id,
            object=chunk.object,
            created=chunk.created,
            model=chunk.model,
            choices=choices,
            system_fingerprint=getattr(chunk, 'system_fingerprint', None)
        )
    
    def close(self) -> None:
        """关闭客户端"""
        if hasattr(self.client, 'close'):
            self.client.close()
    
    async def aclose(self) -> None:
        """异步关闭客户端"""
        if hasattr(self.async_client, 'aclose'):
            await self.async_client.aclose()