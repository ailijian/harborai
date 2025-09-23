#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 主客户端

提供与 OpenAI SDK 一致的调用接口，支持思考模型、结构化输出等功能。
"""

import asyncio
from typing import Dict, List, Optional, Union, Any, AsyncGenerator, Iterator

from ..core.client_manager import ClientManager
from ..core.base_plugin import ChatCompletion, ChatCompletionChunk
from ..utils.exceptions import HarborAIError, ValidationError
from ..utils.logger import get_logger, APICallLogger
from ..utils.tracer import TraceContext, get_or_create_trace_id
from ..utils.retry import async_retry_with_backoff, retry_with_backoff
from ..config.settings import get_settings


class ChatCompletions:
    """聊天完成接口"""
    
    def __init__(self, client_manager: ClientManager):
        self.client_manager = client_manager
        self.logger = get_logger("harborai.chat_completions")
        self.api_logger = APICallLogger()
        self.settings = get_settings()
    
    def create(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        frequency_penalty: Optional[float] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        structured_provider: Optional[str] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """创建聊天完成（同步版本）"""
        # 验证消息
        self._validate_messages(messages)
        
        trace_id = get_or_create_trace_id()
        
        with TraceContext(trace_id):
            # 验证structured_provider参数
            if structured_provider and structured_provider not in ["agently", "native"]:
                raise ValidationError(
                    f"Invalid structured_provider '{structured_provider}'. "
                    "Must be 'agently' or 'native'"
                )
            
            # 构建请求参数
            request_params = {
                "messages": messages,
                "model": model,
                "frequency_penalty": frequency_penalty,
                "function_call": function_call,
                "functions": functions,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
                "max_tokens": max_tokens,
                "n": n,
                "presence_penalty": presence_penalty,
                "response_format": response_format,
                "seed": seed,
                "stop": stop,
                "stream": stream,
                "structured_provider": structured_provider or "agently",
                "temperature": temperature,
                "tool_choice": tool_choice,
                "tools": tools,
                "top_p": top_p,
                "user": user,
                "extra_body": extra_body,
                "timeout": timeout,
                **kwargs
            }
            
            # 移除 None 值
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            try:
                # 记录请求日志
                self.api_logger.log_request(
                    method="POST",
                    url="/chat/completions",
                    params=request_params,
                    trace_id=trace_id
                )
                
                # 使用重试装饰器
                # 转换字典消息为ChatMessage对象
                from ..core.base_plugin import ChatMessage
                chat_messages = [
                    ChatMessage(
                        role=msg["role"],
                        content=msg.get("content"),
                        name=msg.get("name"),
                        function_call=msg.get("function_call"),
                        tool_calls=msg.get("tool_calls"),
                        tool_call_id=msg.get("tool_call_id")
                    )
                    for msg in messages
                ]
                
                @retry_with_backoff()
                def _create_with_retry():
                    return self.client_manager.chat_completion_sync_with_fallback(
                        model=model,
                        messages=chat_messages,
                        **{k: v for k, v in request_params.items() if k not in ['model', 'messages']}
                    )
                
                response = _create_with_retry()
                
                # 记录响应日志
                self.api_logger.log_response(
                    response=response,
                    trace_id=trace_id
                )
                
                return response
                
            except Exception as e:
                # 记录错误日志
                self.api_logger.log_error(
                    error=e,
                    model=model,
                    plugin_name="unknown",
                    latency_ms=0,
                    trace_id=trace_id
                )
                raise e
    
    async def acreate(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        *,
        frequency_penalty: Optional[float] = None,
        function_call: Optional[Union[str, Dict[str, Any]]] = None,
        functions: Optional[List[Dict[str, Any]]] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = None,
        structured_provider: Optional[str] = None,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        top_p: Optional[float] = None,
        user: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """创建聊天完成（异步版本）"""
        # 验证消息
        self._validate_messages(messages)
        
        trace_id = get_or_create_trace_id()
        
        async with TraceContext(trace_id):
            # 验证structured_provider参数
            if structured_provider and structured_provider not in ["agently", "native"]:
                raise ValidationError(
                    f"Invalid structured_provider '{structured_provider}'. "
                    "Must be 'agently' or 'native'"
                )
            
            # 构建请求参数
            request_params = {
                "messages": messages,
                "model": model,
                "frequency_penalty": frequency_penalty,
                "function_call": function_call,
                "functions": functions,
                "logit_bias": logit_bias,
                "logprobs": logprobs,
                "top_logprobs": top_logprobs,
                "max_tokens": max_tokens,
                "n": n,
                "presence_penalty": presence_penalty,
                "response_format": response_format,
                "seed": seed,
                "stop": stop,
                "stream": stream,
                "structured_provider": structured_provider or "agently",
                "temperature": temperature,
                "tool_choice": tool_choice,
                "tools": tools,
                "top_p": top_p,
                "user": user,
                "extra_body": extra_body,
                "timeout": timeout,
                **kwargs
            }
            
            # 移除 None 值
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            try:
                # 记录请求日志
                await self.api_logger.alog_request(
                    method="POST",
                    url="/chat/completions",
                    params=request_params,
                    trace_id=trace_id
                )
                
                # 转换字典消息为ChatMessage对象
                from ..core.base_plugin import ChatMessage
                chat_messages = [
                    ChatMessage(
                        role=msg["role"],
                        content=msg.get("content"),
                        name=msg.get("name"),
                        function_call=msg.get("function_call"),
                        tool_calls=msg.get("tool_calls"),
                        tool_call_id=msg.get("tool_call_id")
                    )
                    for msg in messages
                ]
                
                # 使用重试装饰器
                @async_retry_with_backoff()
                async def _acreate_with_retry():
                    return await self.client_manager.chat_completion_with_fallback(
                        model=model,
                        messages=chat_messages,
                        **{k: v for k, v in request_params.items() if k not in ['model', 'messages']}
                    )
                
                response = await _acreate_with_retry()
                
                # 记录响应日志
                await self.api_logger.alog_response(
                    response=response,
                    trace_id=trace_id
                )
                
                return response
                
            except Exception as e:
                # 记录错误日志
                await self.api_logger.alog_error(
                    error=e,
                    model=model,
                    plugin_name="unknown",
                    latency_ms=0,
                    trace_id=trace_id
                )
                raise e
    
    def _validate_messages(self, messages: List[Dict[str, Any]]) -> None:
        """验证消息格式"""
        if not messages:
            raise ValidationError("Messages cannot be empty")
        
        valid_roles = {"system", "user", "assistant", "tool", "function"}
        
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValidationError(f"Message at index {i} must be a dictionary")
            
            if "role" not in message:
                raise ValidationError(f"Message at index {i} must have a 'role' field")
            
            if message["role"] not in valid_roles:
                raise ValidationError(
                    f"Message at index {i} has invalid role '{message['role']}'. "
                    f"Valid roles are: {', '.join(valid_roles)}"
                )
            
            if "content" not in message and "tool_calls" not in message:
                raise ValidationError(
                    f"Message at index {i} must have either 'content' or 'tool_calls' field"
                )


class Chat:
    """聊天接口"""
    
    def __init__(self, client_manager: ClientManager):
        self.completions = ChatCompletions(client_manager)


class HarborAI:
    """HarborAI 主客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        default_headers: Optional[Dict[str, str]] = None,
        default_query: Optional[Dict[str, str]] = None,
        http_client: Optional[Any] = None,
        **kwargs
    ):
        """初始化 HarborAI 客户端"""
        self.logger = get_logger("harborai.client")
        self.settings = get_settings()
        
        # 存储客户端配置
        self.config = {
            "api_key": api_key,
            "organization": organization,
            "project": project,
            "base_url": base_url,
            "timeout": timeout or self.settings.default_timeout,
            "max_retries": max_retries or self.settings.max_retries,
            "default_headers": default_headers or {},
            "default_query": default_query or {},
            "http_client": http_client,
            **kwargs
        }
        
        # 初始化客户端管理器
        self.client_manager = ClientManager()
        
        # 初始化接口
        self.chat = Chat(self.client_manager)
        
        self.logger.info(
            "HarborAI client initialized",
            trace_id=get_or_create_trace_id(),
            config={
                k: v for k, v in self.config.items() 
                if k not in ['api_key', 'http_client']
            },
            available_plugins=list(self.client_manager.plugins.keys()),
            available_models=len(self.client_manager.model_to_plugin)
        )
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        return list(self.client_manager.model_to_plugin.keys())
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return self.client_manager.get_plugin_info()
    
    def register_plugin(self, plugin) -> None:
        """注册插件"""
        self.client_manager.register_plugin(plugin)
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """注销插件"""
        self.client_manager.unregister_plugin(plugin_name)
    
    async def aclose(self) -> None:
        """异步关闭客户端"""
        # 清理资源
        for plugin in self.client_manager.plugins.values():
            if hasattr(plugin, 'aclose'):
                try:
                    await plugin.aclose()
                except Exception as e:
                    self.logger.warning(
                        "Error closing plugin",
                        trace_id=get_or_create_trace_id(),
                        plugin=plugin.name,
                        error=str(e)
                    )
        
        self.logger.info(
            "HarborAI client closed",
            trace_id=get_or_create_trace_id()
        )
    
    def close(self) -> None:
        """同步关闭客户端"""
        # 清理资源
        for plugin in self.client_manager.plugins.values():
            if hasattr(plugin, 'close'):
                try:
                    plugin.close()
                except Exception as e:
                    self.logger.warning(
                        "Error closing plugin",
                        trace_id=get_or_create_trace_id(),
                        plugin=plugin.name,
                        error=str(e)
                    )
        
        self.logger.info(
            "HarborAI client closed",
            trace_id=get_or_create_trace_id()
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()


# 为了兼容性，提供别名
Client = HarborAI