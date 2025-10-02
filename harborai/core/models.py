#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 模型定义模块

定义了HarborAI项目中使用的模型相关类和函数，包括推理模型和模型能力检测。
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import re


class ModelType(Enum):
    """模型类型枚举"""
    REASONING = "reasoning"  # 推理模型
    CHAT = "chat"           # 聊天模型
    COMPLETION = "completion"  # 补全模型
    EMBEDDING = "embedding"   # 嵌入模型


@dataclass
class ModelCapabilities:
    """模型能力定义
    
    定义了模型支持的各种能力和限制。
    
    Args:
        supports_reasoning: 是否支持推理思考
        supports_streaming: 是否支持流式输出
        supports_temperature: 是否支持温度参数
        supports_system_message: 是否支持系统消息
        supports_function_calling: 是否支持函数调用
        supports_structured_output: 是否支持结构化输出
        max_tokens_limit: 最大token限制
        max_context_length: 最大上下文长度
        supported_parameters: 支持的参数列表
        unsupported_parameters: 不支持的参数列表
    """
    supports_reasoning: bool = False
    supports_streaming: bool = True
    supports_temperature: bool = True
    supports_system_message: bool = True
    supports_function_calling: bool = False
    supports_structured_output: bool = False
    max_tokens_limit: int = 4096
    max_context_length: int = 4096
    supported_parameters: Optional[List[str]] = None
    unsupported_parameters: Optional[List[str]] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.supported_parameters is None:
            self.supported_parameters = []
        if self.unsupported_parameters is None:
            self.unsupported_parameters = []


@dataclass
class ReasoningModel:
    """推理模型定义
    
    定义了推理模型的特殊属性和行为。
    
    Args:
        name: 模型名称
        provider: 模型提供商
        capabilities: 模型能力
        reasoning_format: 推理格式（如：思考过程的格式）
        max_reasoning_tokens: 最大推理token数
        supports_chain_of_thought: 是否支持思维链
        requires_special_handling: 是否需要特殊处理
    """
    name: str
    provider: str
    capabilities: ModelCapabilities
    reasoning_format: str = "thinking"
    max_reasoning_tokens: int = 8192
    supports_chain_of_thought: bool = True
    requires_special_handling: bool = True
    
    def is_reasoning_model(self) -> bool:
        """判断是否为推理模型"""
        return self.capabilities.supports_reasoning
    
    def get_filtered_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """获取过滤后的参数
        
        移除推理模型不支持的参数。
        
        Args:
            parameters: 原始参数字典
            
        Returns:
            过滤后的参数字典
        """
        filtered = parameters.copy()
        
        # 移除不支持的参数
        for param in self.capabilities.unsupported_parameters:
            if param in filtered:
                del filtered[param]
        
        return filtered


# 预定义的模型能力配置
MODEL_CAPABILITIES_CONFIG = {
    # DeepSeek 推理模型 - 支持推理过程和结果的流式输出
    "deepseek-reasoner": ModelCapabilities(
        supports_reasoning=True,
        supports_streaming=True,  # 推理模型支持流式输出
        supports_temperature=True,  # 推理模型支持temperature参数
        supports_system_message=False,
        supports_function_calling=False,
        supports_structured_output=False,
        max_tokens_limit=32768,
        max_context_length=32768,
        supported_parameters=["messages", "model", "max_tokens", "temperature", "top_p", "stream"],  # 添加temperature和top_p支持
        unsupported_parameters=["frequency_penalty", "presence_penalty", "functions", "function_call"]  # 移除temperature和top_p
    ),
    
    # DeepSeek 常规模型
    "deepseek-chat": ModelCapabilities(
        supports_reasoning=False,
        supports_streaming=True,
        supports_temperature=True,
        supports_system_message=True,
        supports_function_calling=True,
        supports_structured_output=True,
        max_tokens_limit=4096,
        max_context_length=32768,
        supported_parameters=["messages", "model", "max_tokens", "temperature", "top_p", "stream", "functions", "function_call"],
        unsupported_parameters=[]
    ),

    # 文心一言 X1 Turbo 32K - 推理模型
    "ernie-x1-turbo-32k": ModelCapabilities(
        supports_reasoning=True,
        supports_streaming=True,
        supports_temperature=True,  # 推理模型支持temperature参数
        supports_system_message=True,
        supports_function_calling=False,
        supports_structured_output=False,
        max_tokens_limit=32768,
        max_context_length=32768,
        supported_parameters=["messages", "model", "max_tokens", "temperature", "top_p", "stream"],  # 添加temperature和top_p支持
        unsupported_parameters=["frequency_penalty", "presence_penalty", "functions", "function_call"]  # 移除temperature和top_p
    ),

    # 文心一言 常规模型
    "ernie-3.5-8k": ModelCapabilities(
        supports_reasoning=False,
        supports_streaming=True,
        supports_temperature=True,
        supports_system_message=True,
        supports_function_calling=True,
        supports_structured_output=False,
        max_tokens_limit=8192,
        max_context_length=8192,
        supported_parameters=["messages", "model", "max_tokens", "temperature", "top_p", "stream"],
        unsupported_parameters=["frequency_penalty", "presence_penalty"]
    ),
    "ernie-4.0-turbo-8k": ModelCapabilities(
        supports_reasoning=False,
        supports_streaming=True,
        supports_temperature=True,
        supports_system_message=True,
        supports_function_calling=True,
        supports_structured_output=False,
        max_tokens_limit=8192,
        max_context_length=8192,
        supported_parameters=["messages", "model", "max_tokens", "temperature", "top_p", "stream"],
        unsupported_parameters=["frequency_penalty", "presence_penalty"]
    ),

    # 豆包 推理模型 - doubao-seed-1-6-250615 支持推理过程和结果的流式输出
    "doubao-seed-1-6-250615": ModelCapabilities(
        supports_reasoning=True,  # 修正为推理模型
        supports_streaming=True,  # 推理模型支持流式输出
        supports_temperature=True,
        supports_system_message=True,
        supports_function_calling=False,
        supports_structured_output=True,  # 豆包支持结构化输出
        max_tokens_limit=32768,
        max_context_length=32768,
        supported_parameters=["messages", "model", "max_tokens", "temperature", "top_p", "stream"],
        unsupported_parameters=["frequency_penalty", "presence_penalty", "functions", "function_call"]
    ),
    
    # 豆包 常规模型
    "doubao-1-5-pro-32k-character-250715": ModelCapabilities(
        supports_reasoning=False,
        supports_streaming=True,
        supports_temperature=True,
        supports_system_message=True,
        supports_function_calling=False,
        supports_structured_output=True,  # 豆包支持结构化输出
        max_tokens_limit=4096,
        max_context_length=4096,
        supported_parameters=["messages", "model", "max_tokens", "temperature", "top_p", "stream"],
        unsupported_parameters=["frequency_penalty", "presence_penalty", "functions", "function_call"]
    )
}

# 推理模型名称模式
REASONING_MODEL_PATTERNS = [
    r".*-r\d+.*",        # 如：deepseek-reasoner,
    r".*-reasoner.*",    # 如：deepseek-reasoner
    r".*reasoning.*",    # 如：gpt-4-reasoning
    r".*think.*",        # 如：claude-think
]


def is_reasoning_model(model_name: str) -> bool:
    """判断是否为推理模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        是否为推理模型
    """
    if not model_name:
        return False
    
    # 首先检查预定义配置
    if model_name in MODEL_CAPABILITIES_CONFIG:
        return MODEL_CAPABILITIES_CONFIG[model_name].supports_reasoning
    
    # 使用模式匹配
    for pattern in REASONING_MODEL_PATTERNS:
        if re.match(pattern, model_name, re.IGNORECASE):
            return True
    
    return False


def get_model_capabilities(model_name: str) -> ModelCapabilities:
    """获取模型能力
    
    Args:
        model_name: 模型名称
        
    Returns:
        模型能力对象
    """
    if model_name in MODEL_CAPABILITIES_CONFIG:
        return MODEL_CAPABILITIES_CONFIG[model_name]
    
    # 如果没有预定义配置，返回默认能力
    if is_reasoning_model(model_name):
        # 推理模型的默认能力
        return ModelCapabilities(
            supports_reasoning=True,
            supports_streaming=False,
            supports_temperature=False,
            supports_system_message=False,
            max_tokens_limit=32768,
            unsupported_parameters=["temperature", "top_p", "frequency_penalty", "presence_penalty", "stream"]
        )
    else:
        # 常规模型的默认能力
        return ModelCapabilities(
            supports_reasoning=False,
            supports_streaming=True,
            supports_temperature=True,
            supports_system_message=True,
            max_tokens_limit=4096
        )


def create_reasoning_model(model_name: str, provider: str = "unknown") -> ReasoningModel:
    """创建推理模型实例
    
    Args:
        model_name: 模型名称
        provider: 模型提供商
        
    Returns:
        推理模型实例
    """
    capabilities = get_model_capabilities(model_name)
    
    return ReasoningModel(
        name=model_name,
        provider=provider,
        capabilities=capabilities
    )


def get_supported_models() -> Dict[str, ModelCapabilities]:
    """获取所有支持的模型及其能力
    
    Returns:
        模型名称到能力的映射字典
    """
    return MODEL_CAPABILITIES_CONFIG.copy()


def get_reasoning_models() -> List[str]:
    """获取所有推理模型名称
    
    Returns:
        推理模型名称列表
    """
    return [
        model_name for model_name, capabilities in MODEL_CAPABILITIES_CONFIG.items()
        if capabilities.supports_reasoning
    ]


def filter_parameters_for_model(model_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """为指定模型过滤参数
    
    Args:
        model_name: 模型名称
        parameters: 原始参数字典
        
    Returns:
        过滤后的参数字典
    """
    import logging
    
    capabilities = get_model_capabilities(model_name)
    filtered = parameters.copy()
    
    # 移除不支持的参数并记录警告
    for param in capabilities.unsupported_parameters:
        if param in filtered:
            logging.warning(f"参数 '{param}' 不被模型 '{model_name}' 支持，已自动移除")
            del filtered[param]
    
    return filtered