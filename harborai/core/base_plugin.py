#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件基类

定义所有 LLM 插件的统一接口和基础功能。
支持推理模型、结构化输出、流式调用等特性。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, AsyncGenerator, Generator
from dataclasses import dataclass
import time

from ..utils.exceptions import PluginError, ModelNotFoundError
from ..utils.logger import get_logger
from ..utils.tracer import get_current_trace_id


@dataclass
class ModelInfo:
    """模型信息"""
    id: str  # 模型ID
    name: str  # 模型显示名称
    provider: str
    supports_streaming: bool = True
    supports_structured_output: bool = False
    supports_thinking: bool = False
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    description: Optional[str] = None


@dataclass
class ChatMessage:
    """聊天消息"""
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    reasoning_content: Optional[str] = None  # 推理模型的推理内容
    parsed: Optional[Any] = None  # 结构化输出的解析结果


@dataclass
class ChatChoiceDelta:
    """聊天选择增量数据。"""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning_content: Optional[str] = None  # 推理模型的推理内容

@dataclass
class ChatChoice:
    """聊天选择。"""
    index: int
    message: Optional[ChatMessage] = None
    delta: Optional[ChatChoiceDelta] = None
    finish_reason: Optional[str] = None


@dataclass
class Usage:
    """使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    """聊天完成响应"""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None
    # 结构化输出支持
    parsed: Optional[Any] = None


@dataclass
class ChatCompletionChunk:
    """聊天完成流式响应块"""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatChoice]
    system_fingerprint: Optional[str] = None


class BaseLLMPlugin(ABC):
    """LLM 插件基类"""
    
    def __init__(self, name: str, **config):
        self.name = name
        self.config = config
        self.logger = get_logger(f"harborai.plugin.{name}")
        self._supported_models: List[ModelInfo] = []
    
    @property
    def supported_models(self) -> List[ModelInfo]:
        """获取支持的模型列表"""
        return self._supported_models
    
    def supports_model(self, model_id: str) -> bool:
        """检查是否支持指定模型"""
        return any(model.id == model_id for model in self._supported_models)
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """获取模型信息"""
        for model in self._supported_models:
            if model.id == model_id:
                return model
        return None
    
    @abstractmethod
    def chat_completion(
        self,
        model: str,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        """同步聊天完成"""
        pass
    
    @abstractmethod
    async def chat_completion_async(
        self,
        model: str,
        messages: List[ChatMessage],
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        """异步聊天完成"""
        pass
    
    def extract_reasoning_content(self, response: Union[ChatCompletion, ChatCompletionChunk]) -> Optional[str]:
        """提取思考过程，动态检测 reasoning_content 字段"""
        try:
            if isinstance(response, ChatCompletion):
                # 非流式响应
                if response.choices and response.choices[0].message:
                    message = response.choices[0].message
                    return getattr(message, 'reasoning_content', None)
            elif isinstance(response, ChatCompletionChunk):
                # 流式响应
                if response.choices and response.choices[0].delta:
                    delta = response.choices[0].delta
                    return getattr(delta, 'reasoning_content', None)
        except Exception as e:
            self.logger.warning(
                "Failed to extract reasoning content",
                trace_id=get_current_trace_id(),
                error=str(e)
            )
        return None
    
    def validate_request(
        self,
        model: str,
        messages: List[ChatMessage],
        **kwargs
    ) -> None:
        """验证请求参数"""
        if not messages:
            raise PluginError(self.name, "Messages cannot be empty")
        
        if not self.supports_model(model):
            raise ModelNotFoundError(model, trace_id=get_current_trace_id())
        
        # 验证消息格式
        for i, message in enumerate(messages):
            if not isinstance(message, ChatMessage):
                raise PluginError(self.name, f"Message {i} must be a ChatMessage instance")
            
            if not message.role:
                raise PluginError(self.name, f"Message {i} missing 'role' field")
            
            if not message.content:
                raise PluginError(self.name, f"Message {i} missing 'content' field")
    
    def prepare_request(
        self,
        model: str,
        messages: List[ChatMessage],
        **kwargs
    ) -> Dict[str, Any]:
        """准备请求数据"""
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
        
        # 基础请求数据
        request_data = {
            "model": model,
            "messages": message_dicts,
        }
        
        # 添加其他参数
        for key, value in kwargs.items():
            if value is not None:
                request_data[key] = value
        
        return request_data
    
    def handle_structured_output(
        self,
        response: ChatCompletion,
        response_format: Optional[Dict[str, Any]] = None,
        structured_provider: str = "agently",
        model: Optional[str] = None
    ) -> ChatCompletion:
        """处理结构化输出"""
        if not response_format or response_format.get("type") != "json_schema":
            return response
        
        try:
            if structured_provider == "agently":
                # 使用 Agently 解析
                parsed_content = self._parse_with_agently(
                    response.choices[0].message.content,
                    response_format,
                    model
                )
            else:
                # 使用原生解析
                parsed_content = self._parse_with_native(
                    response.choices[0].message.content,
                    response_format
                )
            
            # 将parsed属性设置到message对象上，而不是response对象上
            response.choices[0].message.parsed = parsed_content
            
        except Exception as e:
            self.logger.warning(
                "Failed to parse structured output",
                extra={
                    "trace_id": get_current_trace_id(),
                    "provider": structured_provider,
                    "error": str(e)
                }
            )
        
        return response
    
    def _parse_with_agently(self, content: str, response_format: Dict[str, Any], model: Optional[str] = None) -> Any:
        """使用 Agently 解析结构化输出"""
        try:
            from ..api.structured import default_handler
            
            # 使用StructuredOutputHandler进行解析
            schema = response_format.get('json_schema', {})
            return default_handler.parse_response(
                content, 
                schema, 
                use_agently=True, 
                api_key=getattr(self, 'api_key', None), 
                base_url=getattr(self, 'base_url', None),
                model=model or getattr(self, 'model', None)
            )
            
        except ImportError as e:
            self.logger.warning(
                "Agently library not available, falling back to native parsing",
                extra={
                    "trace_id": get_current_trace_id(),
                    "error": str(e)
                }
            )
            # Agently库不可用，回退到原生解析
            return self._parse_with_native(content, response_format)
        except Exception as e:
            # API密钥错误或其他错误，不回退，直接抛出
            self.logger.error(
                "Agently parsing failed with error",
                extra={
                    "trace_id": get_current_trace_id(),
                    "error": str(e)
                }
            )
            raise
    
    def _parse_with_native(self, content: str, response_format: Dict[str, Any]) -> Any:
        """使用原生方式解析结构化输出"""
        try:
            from ..api.structured import default_handler
            
            # 使用StructuredOutputHandler进行原生解析
            schema = response_format.get('json_schema', {})
            return default_handler.parse_response(
                content, 
                schema, 
                use_agently=False
            )
            
        except Exception as e:
            self.logger.error(
                "Native parsing failed",
                extra={
                    "trace_id": get_current_trace_id(),
                    "error": str(e)
                }
            )
            # 最后的回退：直接JSON解析
            import json
            return json.loads(content)
    
    def create_error_response(
        self,
        error: Exception,
        model: str,
        request_id: Optional[str] = None
    ) -> ChatCompletion:
        """创建错误响应"""
        return ChatCompletion(
            id=request_id or f"error_{int(time.time())}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=f"Error: {str(error)}"
                    ),
                    finish_reason="error"
                )
            ]
        )
    
    def log_request(
        self,
        model: str,
        messages: List[ChatMessage],
        **kwargs
    ) -> None:
        """记录请求日志"""
        self.logger.info(
            "Plugin request started",
            extra={
                "trace_id": get_current_trace_id(),
                "plugin": self.name,
                "model": model,
                "message_count": len(messages),
                "stream": kwargs.get('stream', False),
                "structured_output": bool(kwargs.get('response_format'))
            }
        )
    
    def log_response(
        self,
        response: Union[ChatCompletion, ChatCompletionChunk],
        latency_ms: float
    ) -> None:
        """记录响应日志"""
        reasoning_present = bool(self.extract_reasoning_content(response))
        
        self.logger.info(
            "Plugin request completed",
            extra={
                "trace_id": get_current_trace_id(),
                "plugin": self.name,
                "latency_ms": latency_ms,
                "reasoning_content_present": reasoning_present,
                "response_type": type(response).__name__
            }
        )