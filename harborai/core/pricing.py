#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型价格配置和成本计算模块"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ModelPricing:
    """模型价格配置"""
    input_price: float  # 每1K输入tokens的价格（美元）
    output_price: float  # 每1K输出tokens的价格（美元）


class PricingCalculator:
    """价格计算器"""
    
    # 模型价格配置（每1K tokens的价格，单位：美元）
    MODEL_PRICING: Dict[str, ModelPricing] = {
        "deepseek-chat": ModelPricing(input_price=0.00014, output_price=0.00028),
        "ernie-3.5-8k": ModelPricing(input_price=0.0008, output_price=0.002),
        "ernie-4.0-turbo-8k": ModelPricing(input_price=0.003, output_price=0.009),
        "ernie-x1-turbo-32k": ModelPricing(input_price=0.008, output_price=0.024),
        "doubao-1-5-pro-32k-character-250715": ModelPricing(input_price=0.0008, output_price=0.002),
        "doubao-seed-1-6-250615": ModelPricing(input_price=0.0005, output_price=0.001),
        # OpenAI模型价格
        "gpt-3.5-turbo": ModelPricing(input_price=0.0015, output_price=0.002),
        "gpt-4": ModelPricing(input_price=0.03, output_price=0.06),
        "gpt-4-turbo": ModelPricing(input_price=0.01, output_price=0.03),
        "gpt-4o": ModelPricing(input_price=0.005, output_price=0.015),
        "gpt-4o-mini": ModelPricing(input_price=0.00015, output_price=0.0006),
    }
    
    @classmethod
    def calculate_cost(cls, input_tokens: int, output_tokens: int, model_name: str) -> Optional[float]:
        """
        计算调用成本
        
        Args:
            input_tokens: 输入token数量
            output_tokens: 输出token数量
            model_name: 模型名称
            
        Returns:
            成本（美元），如果无法计算则返回None
        """
        if model_name not in cls.MODEL_PRICING:
            return None
        
        pricing = cls.MODEL_PRICING[model_name]
        input_cost = (input_tokens / 1000) * pricing.input_price
        output_cost = (output_tokens / 1000) * pricing.output_price
        
        return input_cost + output_cost
    
    @classmethod
    def get_model_pricing(cls, model_name: str) -> Optional[ModelPricing]:
        """
        获取模型价格配置
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型价格配置，如果不存在则返回None
        """
        return cls.MODEL_PRICING.get(model_name)
    
    @classmethod
    def add_model_pricing(cls, model_name: str, input_price: float, output_price: float) -> None:
        """
        添加模型价格配置
        
        Args:
            model_name: 模型名称
            input_price: 每1K输入tokens的价格（美元）
            output_price: 每1K输出tokens的价格（美元）
        """
        cls.MODEL_PRICING[model_name] = ModelPricing(
            input_price=input_price,
            output_price=output_price
        )
    
    @classmethod
    def list_supported_models(cls) -> list[str]:
        """
        列出支持价格计算的模型
        
        Returns:
            支持的模型名称列表
        """
        return list(cls.MODEL_PRICING.keys())