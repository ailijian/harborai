#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 厂商管理器

负责管理多个AI厂商的配置、切换和故障转移。
支持DeepSeek、百度ERNIE、字节跳动Doubao等厂商。
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .exceptions import HarborAIError, APIError
from .plugins.deepseek_plugin import DeepSeekPlugin
from .plugins.wenxin_plugin import WenxinPlugin
from .plugins.doubao_plugin import DoubaoPlugin

logger = logging.getLogger(__name__)


class VendorType(Enum):
    """支持的厂商类型"""
    DEEPSEEK = "deepseek"
    ERNIE = "ernie"
    DOUBAO = "doubao"


@dataclass
class VendorConfig:
    """厂商配置数据类"""
    name: str
    api_key: str
    base_url: str
    models: List[str]
    max_tokens: int
    supports_streaming: bool
    supports_function_calling: bool
    rate_limit_rpm: int


class VendorManager:
    """厂商管理器
    
    负责管理多个AI厂商的配置、切换和故障转移。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化厂商管理器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self._vendors: Dict[VendorType, VendorConfig] = {}
        self._plugins: Dict[VendorType, Any] = {}
        self._initialize_vendors()
    
    def _initialize_vendors(self):
        """初始化厂商配置"""
        # DeepSeek 配置
        self._vendors[VendorType.DEEPSEEK] = VendorConfig(
            name="DeepSeek",
            api_key=self.config.get("deepseek_api_key", ""),
            base_url="https://api.deepseek.com",
            models=["deepseek-chat", "deepseek-coder"],
            max_tokens=4096,
            supports_streaming=True,
            supports_function_calling=True,
            rate_limit_rpm=60
        )
        
        # 百度ERNIE 配置
        self._vendors[VendorType.ERNIE] = VendorConfig(
            name="百度ERNIE",
            api_key=self.config.get("ernie_api_key", ""),
            base_url="https://aip.baidubce.com",
            models=["ernie-3.5-8k", "ernie-4.0-8k"],
            max_tokens=8192,
            supports_streaming=True,
            supports_function_calling=False,
            rate_limit_rpm=100
        )
        
        # 字节跳动Doubao 配置
        self._vendors[VendorType.DOUBAO] = VendorConfig(
            name="字节跳动Doubao",
            api_key=self.config.get("doubao_api_key", ""),
            base_url="https://ark.cn-beijing.volces.com",
            models=["doubao-lite-4k", "doubao-pro-4k"],
            max_tokens=4096,
            supports_streaming=True,
            supports_function_calling=True,
            rate_limit_rpm=120
        )
        
        # 初始化插件
        self._initialize_plugins()
    
    def _initialize_plugins(self):
        """初始化厂商插件"""
        try:
            self._plugins[VendorType.DEEPSEEK] = DeepSeekPlugin()
        except Exception as e:
            logger.warning(f"Failed to initialize DeepSeek plugin: {e}")
        
        try:
            self._plugins[VendorType.ERNIE] = WenxinPlugin()
        except Exception as e:
            logger.warning(f"Failed to initialize ERNIE plugin: {e}")
        
        try:
            self._plugins[VendorType.DOUBAO] = DoubaoPlugin()
        except Exception as e:
            logger.warning(f"Failed to initialize Doubao plugin: {e}")
    
    def get_available_vendors(self) -> List[str]:
        """获取可用的厂商列表
        
        Returns:
            厂商名称列表
        """
        return [vendor.value for vendor in self._vendors.keys()]
    
    def get_vendor_config(self, vendor: Union[str, VendorType]) -> Optional[VendorConfig]:
        """获取厂商配置
        
        Args:
            vendor: 厂商类型或名称
            
        Returns:
            厂商配置或None
        """
        if isinstance(vendor, str):
            try:
                vendor = VendorType(vendor)
            except ValueError:
                return None
        
        return self._vendors.get(vendor)
    
    def get_vendor_plugin(self, vendor: Union[str, VendorType]) -> Optional[Any]:
        """获取厂商插件
        
        Args:
            vendor: 厂商类型或名称
            
        Returns:
            厂商插件或None
        """
        if isinstance(vendor, str):
            try:
                vendor = VendorType(vendor)
            except ValueError:
                return None
        
        return self._plugins.get(vendor)
    
    def is_vendor_available(self, vendor: Union[str, VendorType]) -> bool:
        """检查厂商是否可用
        
        Args:
            vendor: 厂商类型或名称
            
        Returns:
            是否可用
        """
        config = self.get_vendor_config(vendor)
        if not config:
            return False
        
        # 检查API密钥是否配置
        return bool(config.api_key)
    
    def get_models_for_vendor(self, vendor: Union[str, VendorType]) -> List[str]:
        """获取厂商支持的模型列表
        
        Args:
            vendor: 厂商类型或名称
            
        Returns:
            模型列表
        """
        config = self.get_vendor_config(vendor)
        if not config:
            return []
        
        return config.models
    
    def get_vendor_for_model(self, model: str) -> Optional[VendorType]:
        """根据模型名称获取对应的厂商
        
        Args:
            model: 模型名称
            
        Returns:
            厂商类型或None
        """
        for vendor, config in self._vendors.items():
            if model in config.models:
                return vendor
        
        return None
    
    def switch_vendor(self, from_vendor: Union[str, VendorType], 
                     to_vendor: Union[str, VendorType]) -> bool:
        """切换厂商
        
        Args:
            from_vendor: 源厂商
            to_vendor: 目标厂商
            
        Returns:
            是否切换成功
        """
        if not self.is_vendor_available(to_vendor):
            logger.error(f"Target vendor {to_vendor} is not available")
            return False
        
        logger.info(f"Switching from {from_vendor} to {to_vendor}")
        return True
    
    def get_failover_sequence(self, primary_vendor: Union[str, VendorType]) -> List[VendorType]:
        """获取故障转移序列
        
        Args:
            primary_vendor: 主要厂商
            
        Returns:
            故障转移序列
        """
        if isinstance(primary_vendor, str):
            try:
                primary_vendor = VendorType(primary_vendor)
            except ValueError:
                return []
        
        # 构建故障转移序列，优先级：DeepSeek -> ERNIE -> Doubao
        sequence = [primary_vendor]
        
        for vendor in [VendorType.DEEPSEEK, VendorType.ERNIE, VendorType.DOUBAO]:
            if vendor != primary_vendor and self.is_vendor_available(vendor):
                sequence.append(vendor)
        
        return sequence
    
    def update_vendor_config(self, vendor: Union[str, VendorType], 
                           config_updates: Dict[str, Any]) -> bool:
        """更新厂商配置
        
        Args:
            vendor: 厂商类型或名称
            config_updates: 配置更新
            
        Returns:
            是否更新成功
        """
        if isinstance(vendor, str):
            try:
                vendor = VendorType(vendor)
            except ValueError:
                return False
        
        if vendor not in self._vendors:
            return False
        
        config = self._vendors[vendor]
        
        # 更新配置字段
        for key, value in config_updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        logger.info(f"Updated config for vendor {vendor.value}")
        return True