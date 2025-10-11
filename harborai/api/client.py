#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 主客户端

提供与 OpenAI SDK 一致的调用接口，支持推理模型、结构化输出等功能。
"""

import asyncio
import time
import uuid
from typing import Dict, List, Optional, Union, Any, AsyncGenerator, Iterator
from collections.abc import AsyncIterator

from ..core.client_manager import ClientManager
from ..core.base_plugin import ChatCompletion, ChatCompletionChunk
from ..utils.exceptions import HarborAIError, ValidationError
from ..utils.logger import get_logger, APICallLogger
from ..utils.tracer import TraceContext, get_or_create_trace_id
from ..utils.retry import async_retry_with_backoff, retry_with_backoff
from ..config.settings import get_settings
from ..config.performance import get_performance_config
from ..storage.lifecycle import auto_initialize
from ..core.unified_decorators import smart_decorator, fast_trace, full_trace
from ..core.performance_manager import get_performance_manager, initialize_performance_manager, cleanup_performance_manager

# 导入优化组件
from ..core.agently_client_pool import get_agently_client_pool, create_agently_client_config
from ..core.parameter_cache import get_parameter_cache_manager
from ..core.fast_structured_output import FastStructuredOutputProcessor, get_fast_structured_output_processor


class ChatCompletions:
    """聊天完成接口"""
    
    def __init__(self, client_manager: ClientManager):
        self.client_manager = client_manager
        self.logger = get_logger("harborai.chat_completions")
        self.api_logger = APICallLogger(self.logger)
        self.settings = get_settings()
        
        # 初始化快速结构化输出处理器
        self._fast_processor = None
    
    def _get_fast_processor(self) -> FastStructuredOutputProcessor:
        """获取快速结构化输出处理器实例（延迟初始化）"""
        if self._fast_processor is None:
            from ..core.fast_structured_output import create_fast_structured_output_processor
            self._fast_processor = create_fast_structured_output_processor(client_manager=self.client_manager)
        return self._fast_processor
    
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
        fallback: Optional[List[str]] = None,
        fallback_models: Optional[List[str]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """创建聊天完成（同步版本）"""
        import sys
        sys.stderr.write("=" * 50 + "\n")
        sys.stderr.write("DEBUG: ChatCompletions.create method called\n")
        sys.stderr.write(f"DEBUG: model={model}, messages={messages}\n")
        sys.stderr.write("=" * 50 + "\n")
        sys.stderr.flush()
        
        # 获取性能配置
        perf_config = get_performance_config()
        
        # 检查是否启用FAST模式且有结构化输出
        print(f"🔍 检查快速结构化输出条件:")
        print(f"   - perf_config.mode.value: {perf_config.mode.value}")
        print(f"   - response_format: {response_format}")
        print(f"   - structured_provider: {structured_provider}")
        print(f"   - stream: {stream}")
        
        should_use_fast_structured = (perf_config.mode.value == "fast" and 
            response_format and 
            response_format.get("type") == "json_schema" and
            structured_provider == "agently" and
            not stream)
        
        print(f"   - 应该使用快速结构化输出: {should_use_fast_structured}")
        
        if should_use_fast_structured:  # 流式输出暂不支持快速路径
            print("✅ 使用快速结构化输出路径")
            return self._create_fast_structured_path(
                messages, model, response_format, structured_provider,
                frequency_penalty=frequency_penalty,
                function_call=function_call,
                functions=functions,
                logit_bias=logit_bias,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                max_tokens=max_tokens,
                n=n,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                stream=stream,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=tools,
                top_p=top_p,
                user=user,
                extra_body=extra_body,
                timeout=timeout,
                fallback=fallback,
                fallback_models=fallback_models,
                retry_policy=retry_policy,
                **kwargs
            )
        
        # 检查是否使用快速路径
        if perf_config.should_use_fast_path(model, max_tokens):
            return self._create_fast_path(
                messages=messages, model=model, frequency_penalty=frequency_penalty,
                function_call=function_call, functions=functions, logit_bias=logit_bias,
                logprobs=logprobs, top_logprobs=top_logprobs, max_tokens=max_tokens,
                n=n, presence_penalty=presence_penalty, response_format=response_format,
                seed=seed, stop=stop, stream=stream, structured_provider=structured_provider,
                temperature=temperature, tool_choice=tool_choice, tools=tools,
                top_p=top_p, user=user, extra_body=extra_body, timeout=timeout,
                fallback=fallback, fallback_models=fallback_models,
                retry_policy=retry_policy, **kwargs
            )
        else:
            return self._create_full_path(
                messages=messages, model=model, frequency_penalty=frequency_penalty,
                function_call=function_call, functions=functions, logit_bias=logit_bias,
                logprobs=logprobs, top_logprobs=top_logprobs, max_tokens=max_tokens,
                n=n, presence_penalty=presence_penalty, response_format=response_format,
                seed=seed, stop=stop, stream=stream, structured_provider=structured_provider,
                temperature=temperature, tool_choice=tool_choice, tools=tools,
                top_p=top_p, user=user, extra_body=extra_body, timeout=timeout,
                fallback=fallback, fallback_models=fallback_models,
                retry_policy=retry_policy, **kwargs
            )
    
    @fast_trace
    def _create_fast_structured_path(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_format: Dict[str, Any],
        structured_provider: str,
        **kwargs
    ) -> ChatCompletion:
        """快速结构化输出路径 - 使用优化组件"""
        print("🚀 进入快速结构化输出路径")
        
        # 获取trace_id
        from ..utils.tracer import get_or_create_trace_id
        trace_id = get_or_create_trace_id()
        
        # 提取用户输入
        user_input = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_input = msg["content"]
                break
        
        print(f"📝 提取的用户输入: {user_input}")
        
        if not user_input:
            # 如果没有用户输入，回退到常规路径
            print("❌ 没有用户输入，回退到常规路径")
            return self._create_core(messages, model, response_format=response_format, 
                                   structured_provider=structured_provider, **kwargs)
        
        try:
            # 记录请求日志
            from ..utils.logger import LogContext
            log_context = LogContext(trace_id=trace_id)
            request_params = {
                "model": model,
                "messages": messages,
                "response_format": response_format,
                "structured_provider": structured_provider,
                **kwargs
            }
            self.api_logger.log_request(
                context=log_context,
                request_data=request_params
            )
            
            # 使用快速结构化输出处理器
            print("🔧 获取快速处理器")
            fast_processor = self._get_fast_processor()
            
            # 提取schema
            json_schema = response_format.get('json_schema', {})
            schema = json_schema.get('schema', {})
            print(f"📋 提取的Schema: {schema}")
            
            # 调用快速处理器
            print("⚡ 调用快速处理器")
            start_time = time.time()
            parsed_result = fast_processor.process_structured_output(
                user_query=user_input,
                schema=schema,
                api_key=self.client_manager.client_config.get('api_key'),
                base_url=self.client_manager.client_config.get('base_url'),
                model=model,
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_tokens')
            )
            duration = time.time() - start_time
            print(f"✅ 快速处理器返回结果: {parsed_result}")
            
            # 构造兼容的响应对象
            from ..core.base_plugin import ChatMessage, ChatChoice
            
            response = ChatCompletion(
                id=f"fast-structured-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=str(parsed_result)  # 将结构化结果转为字符串
                        ),
                        finish_reason="stop"
                    )
                ]
            )
            
            # 设置parsed属性
            response.choices[0].message.parsed = parsed_result
            
            # 记录响应日志
            response_data = {
                'status_code': 200,
                'response': response,
                'duration': duration
            }
            self.api_logger.log_response(
                context=log_context,
                response_data=response_data
            )
            
            return response
            
        except Exception as e:
            print(
                f"⚠️ 快速结构化输出处理失败，回退到常规路径: {str(e)}"
            )
            # 记录错误日志
            error_data = {
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
            self.api_logger.log_error(
                context=log_context,
                error_data=error_data
            )
            # 回退到常规路径
            return self._create_core(messages, model, response_format=response_format, 
                                   structured_provider=structured_provider, **kwargs)
    
    @fast_trace
    def _create_fast_path(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """快速路径创建聊天完成"""
        return self._create_core(messages, model, **kwargs)
    
    @full_trace
    def _create_full_path(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """完整路径创建聊天完成"""
        return self._create_core(messages, model, **kwargs)
    
    def _create_core(
        self,
        messages: List[Dict[str, Any]],
        model: str,
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
        fallback: Optional[List[str]] = None,
        fallback_models: Optional[List[str]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[ChatCompletion, Iterator[ChatCompletionChunk]]:
        """核心创建逻辑"""
        import sys
        sys.stderr.write("DEBUG: _create_core method called\n")
        sys.stderr.flush()
        
        # 验证消息
        self._validate_messages(messages)
        
        # 验证temperature参数
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                raise ValueError("temperature must be a number")
            if temperature < 0 or temperature > 2.0:
                raise ValueError("temperature must be between 0 and 2")
        
        # 验证max_tokens参数
        if max_tokens is not None:
            from ..core.models import get_model_capabilities
            from ..core.exceptions import ValidationError as CoreValidationError
            
            if not isinstance(max_tokens, int):
                raise ValueError("max_tokens must be an integer")
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            
            capabilities = get_model_capabilities(model)
            if capabilities and capabilities.max_tokens_limit:
                if max_tokens > capabilities.max_tokens_limit:
                    raise CoreValidationError(
                        f"max_tokens ({max_tokens}) exceeds limit for model {model}: {capabilities.max_tokens_limit}"
                    )
        
        # 处理fallback参数兼容性：fallback_models优先级高于fallback
        if fallback_models is not None:
            fallback = fallback_models
        elif fallback is None:
            # 使用settings中配置的默认降级模型
            from ..config.settings import get_settings
            settings = get_settings()
            if settings.enable_fallback and settings.fallback_models:
                fallback = settings.fallback_models
            else:
                fallback = []
        
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
                "fallback": fallback,
                "retry_policy": retry_policy,
                **kwargs
            }
            
            # 移除 None 值
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            # 对推理模型进行参数过滤和消息处理
            from ..core.models import filter_parameters_for_model, is_reasoning_model
            if is_reasoning_model(model):
                # 过滤不支持的参数
                request_params = filter_parameters_for_model(model, request_params)
                
                # 处理推理模型的system消息
                messages = self._process_messages_for_reasoning_model(messages)
                request_params["messages"] = messages
            
            try:
                # 记录请求日志
                import sys
                sys.stderr.write(f"[DEBUG] 准备调用 api_logger.log_request\n")
                sys.stderr.flush()
                
                from ..utils.logger import LogContext
                log_context = LogContext(trace_id=trace_id)
                
                sys.stderr.write(f"[DEBUG] 调用 api_logger.log_request: {self.api_logger}\n")
                sys.stderr.flush()
                
                self.api_logger.log_request(
                    context=log_context,
                    request_data=request_params
                )
                
                sys.stderr.write(f"[DEBUG] api_logger.log_request 调用完成\n")
                sys.stderr.flush()
                
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
                
                # 配置重试策略
                from ..utils.retry import RetryConfig
                retry_config = None
                if retry_policy:
                    retry_config = RetryConfig(
                        max_attempts=retry_policy.get('max_attempts', 3),
                        base_delay=retry_policy.get('base_delay', 1.0),
                        max_delay=retry_policy.get('max_delay', 60.0),
                        exponential_base=retry_policy.get('exponential_base', 2.0),
                        jitter=retry_policy.get('jitter', True)
                    )
                
                @retry_with_backoff(config=retry_config)
                def _create_with_retry():
                    return self.client_manager.chat_completion_sync_with_fallback(
                        model=model,
                        messages=chat_messages,
                        fallback=fallback,
                        **{k: v for k, v in request_params.items() if k not in ['model', 'messages', 'fallback', 'retry_policy']}
                    )
                
                response = _create_with_retry()
                
                # 记录响应日志
                response_data = {
                    'status_code': 200,
                    'response': response
                }
                self.api_logger.log_response(
                    context=log_context,
                    response_data=response_data
                )
                
                return response
                
            except Exception as e:
                # 记录错误日志
                error_context = LogContext(trace_id=trace_id)
                error_data = {
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
                self.api_logger.log_error(
                    context=error_context,
                    error_data=error_data
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
        fallback: Optional[List[str]] = None,
        fallback_models: Optional[List[str]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """创建聊天完成（异步版本）"""
        # 获取性能配置
        perf_config = get_performance_config()
        
        # 检查是否启用FAST模式且有结构化输出
        if (perf_config.mode.value == "fast" and 
            response_format and 
            response_format.get("type") == "json_schema" and
            structured_provider == "agently" and
            not stream):  # 流式输出暂不支持快速路径
            
            return await self._acreate_fast_structured_path(
                messages, model, response_format, structured_provider,
                frequency_penalty=frequency_penalty,
                function_call=function_call,
                functions=functions,
                logit_bias=logit_bias,
                logprobs=logprobs,
                top_logprobs=top_logprobs,
                max_tokens=max_tokens,
                n=n,
                presence_penalty=presence_penalty,
                seed=seed,
                stop=stop,
                stream=stream,
                temperature=temperature,
                tool_choice=tool_choice,
                tools=tools,
                top_p=top_p,
                user=user,
                extra_body=extra_body,
                timeout=timeout,
                fallback=fallback,
                fallback_models=fallback_models,
                retry_policy=retry_policy,
                **kwargs
            )
        
        # 检查是否使用快速路径
        if perf_config.should_use_fast_path(model, max_tokens):
            return await self._acreate_fast_path(
                messages=messages, model=model, frequency_penalty=frequency_penalty,
                function_call=function_call, functions=functions, logit_bias=logit_bias,
                logprobs=logprobs, top_logprobs=top_logprobs, max_tokens=max_tokens,
                n=n, presence_penalty=presence_penalty, response_format=response_format,
                seed=seed, stop=stop, stream=stream, structured_provider=structured_provider,
                temperature=temperature, tool_choice=tool_choice, tools=tools,
                top_p=top_p, user=user, extra_body=extra_body, timeout=timeout,
                fallback=fallback, fallback_models=fallback_models,
                retry_policy=retry_policy, **kwargs
            )
        else:
            return await self._acreate_full_path(
                messages=messages, model=model, frequency_penalty=frequency_penalty,
                function_call=function_call, functions=functions, logit_bias=logit_bias,
                logprobs=logprobs, top_logprobs=top_logprobs, max_tokens=max_tokens,
                n=n, presence_penalty=presence_penalty, response_format=response_format,
                seed=seed, stop=stop, stream=stream, structured_provider=structured_provider,
                temperature=temperature, tool_choice=tool_choice, tools=tools,
                top_p=top_p, user=user, extra_body=extra_body, timeout=timeout,
                fallback=fallback, fallback_models=fallback_models,
                retry_policy=retry_policy, **kwargs
            )
    
    @fast_trace
    async def _acreate_fast_structured_path(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        response_format: Dict[str, Any],
        structured_provider: str,
        **kwargs
    ) -> ChatCompletion:
        """快速结构化输出路径（异步版本）"""
        try:
            # 提取用户输入
            user_input = None
            for msg in reversed(messages):
                if msg.get("role") == "user" and msg.get("content"):
                    user_input = msg["content"]
                    break
            
            if not user_input:
                # 没有用户输入，回退到常规路径
                return await self._acreate_core(messages, model, **kwargs)
            
            # 获取快速处理器
            fast_processor = self._get_fast_processor()
            
            # 提取schema
            json_schema = response_format.get('json_schema', {})
            schema = json_schema.get('schema', {})
            
            # 使用快速处理器处理结构化输出（同步方法，在异步上下文中运行）
            result = fast_processor.process_structured_output(
                user_query=user_input,
                schema=schema,
                api_key=self.client_manager.client_config.get('api_key'),
                base_url=self.client_manager.client_config.get('base_url'),
                model=model,
                temperature=kwargs.get('temperature', 0.1),
                max_tokens=kwargs.get('max_tokens')
            )
            
            # 构造兼容的响应对象
            from ..core.base_plugin import ChatMessage, ChatChoice
            
            response = ChatCompletion(
                id=f"fast-structured-async-{int(time.time())}",
                object="chat.completion",
                created=int(time.time()),
                model=model,
                choices=[
                    ChatChoice(
                        index=0,
                        message=ChatMessage(
                            role="assistant",
                            content=str(result)  # 将结构化结果转为字符串
                        ),
                        finish_reason="stop"
                    )
                ]
            )
            
            # 设置parsed属性
            response.choices[0].message.parsed = result
            
            return response
            
        except Exception as e:
            # 快速路径失败，回退到常规路径
            return await self._acreate_core(messages, model, **kwargs)

    @fast_trace
    async def _acreate_fast_path(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """快速路径异步创建聊天完成"""
        return await self._acreate_core(messages, model, **kwargs)
    
    @full_trace
    async def _acreate_full_path(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """完整路径异步创建聊天完成"""
        return await self._acreate_core(messages, model, **kwargs)
    
    async def _acreate_core(
        self,
        messages: List[Dict[str, Any]],
        model: str,
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
        fallback: Optional[List[str]] = None,
        fallback_models: Optional[List[str]] = None,
        retry_policy: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """异步核心创建逻辑"""
        # 验证消息
        self._validate_messages(messages)
        
        # 验证temperature参数
        if temperature is not None:
            if not isinstance(temperature, (int, float)):
                raise ValueError("temperature must be a number")
            if temperature < 0 or temperature > 2.0:
                raise ValueError("temperature must be between 0 and 2")
        
        # 验证max_tokens参数
        if max_tokens is not None:
            from ..core.models import get_model_capabilities
            from ..core.exceptions import ValidationError as CoreValidationError
            
            if not isinstance(max_tokens, int):
                raise ValueError("max_tokens must be an integer")
            if max_tokens <= 0:
                raise ValueError("max_tokens must be positive")
            
            capabilities = get_model_capabilities(model)
            if capabilities and capabilities.max_tokens_limit:
                if max_tokens > capabilities.max_tokens_limit:
                    raise CoreValidationError(
                        f"max_tokens ({max_tokens}) exceeds limit for model {model}: {capabilities.max_tokens_limit}"
                    )
        
        # 处理fallback参数兼容性：fallback_models优先级高于fallback
        if fallback_models is not None:
            fallback = fallback_models
        elif fallback is None:
            # 使用settings中配置的默认降级模型
            from ..config.settings import get_settings
            settings = get_settings()
            if settings.enable_fallback and settings.fallback_models:
                fallback = settings.fallback_models
            else:
                fallback = []
        
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
                "fallback": fallback,
                "retry_policy": retry_policy,
                **kwargs
            }
            
            # 移除 None 值
            request_params = {k: v for k, v in request_params.items() if v is not None}
            
            # 对推理模型进行参数过滤和消息处理
            from ..core.models import filter_parameters_for_model, is_reasoning_model
            if is_reasoning_model(model):
                # 过滤不支持的参数
                request_params = filter_parameters_for_model(model, request_params)
                
                # 处理推理模型的system消息
                messages = self._process_messages_for_reasoning_model(messages)
                request_params["messages"] = messages
            
            try:
                # 记录请求日志
                from ..utils.logger import LogContext
                log_context = LogContext(trace_id=trace_id)
                request_data = {
                    "method": "POST",
                    "url": "/chat/completions",
                    "model": model,
                    "messages": messages,
                    **request_params
                }
                await self.api_logger.alog_request(log_context, request_data)
                
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
                
                # 配置重试策略
                from ..utils.retry import RetryConfig
                retry_config = None
                if retry_policy:
                    retry_config = RetryConfig(
                        max_attempts=retry_policy.get('max_attempts', 3),
                        base_delay=retry_policy.get('base_delay', 1.0),
                        max_delay=retry_policy.get('max_delay', 60.0),
                        exponential_base=retry_policy.get('exponential_base', 2.0),
                        jitter=retry_policy.get('jitter', True)
                    )
                
                # 使用重试装饰器
                @async_retry_with_backoff(config=retry_config)
                async def _acreate_with_retry():
                    return await self.client_manager.chat_completion_with_fallback(
                        model=model,
                        messages=chat_messages,
                        fallback=fallback,
                        **{k: v for k, v in request_params.items() if k not in ['model', 'messages', 'fallback', 'retry_policy']}
                    )
                
                response = await _acreate_with_retry()
                
                # 记录响应日志
                response_data = {
                    "response": response,
                    "model": model
                }
                await self.api_logger.alog_response(log_context, response_data)
                
                return response
                
            except Exception as e:
                # 记录错误日志
                error_data = {
                    "error": str(e),
                    "model": model,
                    "plugin_name": "unknown",
                    "latency_ms": 0
                }
                await self.api_logger.alog_error(log_context, error_data)
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
    
    def _process_messages_for_reasoning_model(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """处理推理模型的消息，将system消息合并到user消息中"""
        processed_messages = []
        system_content = ""
        
        # 收集所有system消息的内容
        for message in messages:
            if message.get("role") == "system":
                if message.get("content"):
                    system_content += message["content"] + "\n"
            else:
                processed_messages.append(message)
        
        # 如果有system内容，将其添加到第一个user消息中
        if system_content and processed_messages:
            for i, message in enumerate(processed_messages):
                if message.get("role") == "user":
                    original_content = message.get("content", "")
                    processed_messages[i] = {
                        **message,
                        "content": f"{system_content.strip()}\n\n{original_content}"
                    }
                    break
        
        return processed_messages


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
        # 自动初始化生命周期管理（包括PostgreSQL日志记录器）
        auto_initialize()
        
        self.logger = get_logger("harborai.client")
        self.settings = get_settings()
        
        # 初始化性能管理器（如果启用）
        self._performance_manager = None
        if self.settings.enable_performance_manager:
            self._performance_manager = get_performance_manager()
            # 异步初始化将在第一次使用时进行
        
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
        
        # 初始化客户端管理器，传递客户端配置
        self.client_manager = ClientManager(client_config=self.config)
        
        # 初始化接口
        self.chat = Chat(self.client_manager)
        
        # 初始化成本跟踪器（可选）
        self.cost_tracker = None
        
        trace_id = get_or_create_trace_id()
        config_info = {
            k: v for k, v in self.config.items() 
            if k not in ['api_key', 'http_client']
        }
        self.logger.info(
            f"HarborAI client initialized [trace_id={trace_id}] - "
            f"config: {config_info}, "
            f"available_plugins: {list(self.client_manager.plugins.keys())}, "
            f"available_models: {len(self.client_manager.model_to_plugin)}"
        )
    
    def get_available_models(self) -> List[str]:
        """获取可用模型列表"""
        model_infos = self.client_manager.get_available_models()
        # 如果返回的是ModelInfo对象列表，提取id；如果是字符串列表，直接返回
        if model_infos and hasattr(model_infos[0], 'id'):
            return [model_info.id for model_info in model_infos]
        else:
            return model_infos
    
    def get_plugin_info(self) -> Dict[str, Any]:
        """获取插件信息"""
        return self.client_manager.get_plugin_info()
    
    def register_plugin(self, plugin) -> None:
        """注册插件"""
        self.client_manager.register_plugin(plugin)
    
    def unregister_plugin(self, plugin_name: str) -> None:
        """注销插件"""
        self.client_manager.unregister_plugin(plugin_name)
    
    @property
    def client(self):
        """为了兼容性提供的别名属性"""
        return self
    
    def get_total_cost(self) -> float:
        """获取总成本
        
        Returns:
            总成本
        """
        if self.cost_tracker is not None:
            return self.cost_tracker.get_total_cost()
        return 0.0
    
    def reset_cost(self) -> None:
        """重置成本计数器"""
        if self.cost_tracker is not None:
            self.cost_tracker.reset()
    
    async def aclose(self) -> None:
        """异步关闭客户端"""
        # 清理性能管理器
        if self._performance_manager and self._performance_manager.is_initialized():
            try:
                await cleanup_performance_manager()
            except Exception as e:
                trace_id = get_or_create_trace_id()
                self.logger.warning(
                    f"Error cleaning up performance manager [trace_id={trace_id}] - "
                    f"error: {str(e)}"
                )
        
        # 清理资源
        for plugin in self.client_manager.plugins.values():
            if hasattr(plugin, 'aclose'):
                try:
                    await plugin.aclose()
                except Exception as e:
                    trace_id = get_or_create_trace_id()
                    self.logger.warning(
                        f"Error closing plugin [trace_id={trace_id}] - "
                        f"plugin: {plugin.name}, error: {str(e)}"
                    )
        
        trace_id = get_or_create_trace_id()
        self.logger.info(f"HarborAI client closed [trace_id={trace_id}]")
    
    def close(self) -> None:
        """同步关闭客户端"""
        # 同步清理性能管理器（通过异步运行）
        if self._performance_manager and self._performance_manager.is_initialized():
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果事件循环正在运行，创建任务
                    asyncio.create_task(cleanup_performance_manager())
                else:
                    # 如果事件循环未运行，直接运行
                    loop.run_until_complete(cleanup_performance_manager())
            except Exception as e:
                trace_id = get_or_create_trace_id()
                self.logger.warning(
                    f"Error cleaning up performance manager [trace_id={trace_id}] - "
                    f"error: {str(e)}"
                )
        
        # 清理资源
        for plugin in self.client_manager.plugins.values():
            if hasattr(plugin, 'close'):
                try:
                    plugin.close()
                except Exception as e:
                    trace_id = get_or_create_trace_id()
                    self.logger.warning(
                        f"Error closing plugin [trace_id={trace_id}] - "
                        f"plugin: {plugin.name}, error: {str(e)}"
                    )
        
        trace_id = get_or_create_trace_id()
        self.logger.info(f"HarborAI client closed [trace_id={trace_id}]")
    
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