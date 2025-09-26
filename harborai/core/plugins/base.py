#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 插件基础模块

定义了插件系统的基础类和接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Generator
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """插件信息"""
    name: str
    version: str
    description: str
    supported_models: List[str]
    author: str = ""
    homepage: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class Plugin(ABC):
    """插件基类
    
    所有插件都必须继承此类并实现相应的抽象方法。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._info = None
    
    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """返回插件信息"""
        pass
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化插件
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """聊天完成接口
        
        Args:
            messages: 消息列表
            model: 模型名称
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            响应数据或流式生成器
        """
        pass
    
    @abstractmethod
    async def chat_completion_async(
        self, 
        messages: List[Dict[str, str]], 
        model: str,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """异步聊天完成接口
        
        Args:
            messages: 消息列表
            model: 模型名称
            stream: 是否流式输出
            **kwargs: 其他参数
            
        Returns:
            响应数据或异步流式生成器
        """
        pass
    
    def supports_model(self, model: str) -> bool:
        """检查是否支持指定模型
        
        Args:
            model: 模型名称
            
        Returns:
            bool: 是否支持
        """
        return model in self.info.supported_models
    
    def extract_reasoning_content(self, response: Dict[str, Any]) -> Optional[str]:
        """提取思考过程内容
        
        Args:
            response: API响应
            
        Returns:
            思考过程内容，如果没有则返回None
        """
        # 默认实现，子类可以重写
        if isinstance(response, dict):
            choices = response.get('choices', [])
            if choices and len(choices) > 0:
                message = choices[0].get('message', {})
                return message.get('reasoning_content')
        return None
    
    def validate_config(self) -> bool:
        """验证插件配置
        
        Returns:
            bool: 配置是否有效
        """
        # 默认实现，子类可以重写
        return True
    
    def cleanup(self):
        """清理插件资源"""
        # 默认实现，子类可以重写
        pass
    
    @property
    def is_initialized(self) -> bool:
        """检查插件是否已初始化"""
        return self._initialized
    
    def _set_initialized(self, status: bool):
        """设置初始化状态"""
        self._initialized = status


class BaseLLMPlugin(Plugin):
    """LLM插件基类
    
    为LLM厂商插件提供通用功能。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._api_key = None
        self._base_url = None
        self._headers = {}
    
    def set_api_key(self, api_key: str):
        """设置API密钥"""
        self._api_key = api_key
    
    def set_base_url(self, base_url: str):
        """设置基础URL"""
        self._base_url = base_url
    
    def set_headers(self, headers: Dict[str, str]):
        """设置请求头"""
        self._headers.update(headers)
    
    def format_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """格式化消息
        
        子类可以重写此方法来适配不同厂商的消息格式。
        
        Args:
            messages: 原始消息列表
            
        Returns:
            格式化后的消息列表
        """
        return messages
    
    def format_response(self, response: Any) -> Dict[str, Any]:
        """格式化响应
        
        子类可以重写此方法来统一响应格式。
        
        Args:
            response: 原始响应
            
        Returns:
            格式化后的响应
        """
        return {
            "content": str(response),
            "model": self.info.name,
            "usage": {}
        }


# 为了兼容测试文件，提供BasePlugin别名
BasePlugin = Plugin