#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的成本计算器

基于现有PricingCalculator的增强版本，支持输入输出成本细分和环境变量配置。
根据HarborAI日志系统重构设计方案实现。
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import structlog
from .pricing import PricingCalculator, ModelPricing

logger = structlog.get_logger(__name__)

@dataclass
class EnhancedModelPricing:
    """增强的模型价格配置
    
    扩展原有ModelPricing，添加货币、来源等信息
    
    价格来源类型说明：
    - builtin: 使用系统内置的模型价格配置
    - environment_variable: 从环境变量中加载的价格配置
    - dynamic: 动态设置的价格配置
    - unknown: 未知来源或无法确定价格配置
    """
    input_price_per_1k: float  # 每1K输入tokens的价格
    output_price_per_1k: float  # 每1K输出tokens的价格
    currency: str = "CNY"  # 货币单位
    source: str = "builtin"  # 价格来源：builtin, environment_variable, dynamic, unknown
    last_updated: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now(timezone.utc)

@dataclass
class CostBreakdown:
    """成本细分信息
    
    详细记录输入、输出和总成本信息
    """
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str
    pricing_source: str
    pricing_timestamp: str
    pricing_details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "input_cost": float(f"{self.input_cost:.6f}"),  # 保留6位小数，避免科学计数法
            "output_cost": float(f"{self.output_cost:.6f}"),  # 保留6位小数，避免科学计数法
            "total_cost": float(f"{self.total_cost:.6f}"),  # 保留6位小数，避免科学计数法
            "currency": self.currency,
            "pricing_source": self.pricing_source,
            "pricing_timestamp": self.pricing_timestamp,
            "pricing_details": self.pricing_details
        }

class EnvironmentPricingLoader:
    """环境变量价格加载器
    
    从环境变量中加载模型价格配置
    """
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    def load_model_pricing(self, provider: str, model: str) -> Optional[EnhancedModelPricing]:
        """从环境变量加载模型价格
        
        支持的环境变量格式：
        - OPENAI_GPT4_INPUT_PRICE=0.03
        - OPENAI_GPT4_OUTPUT_PRICE=0.06
        - DEEPSEEK_INPUT_PRICE=0.003
        - DEEPSEEK_OUTPUT_PRICE=0.006
        
        Args:
            provider: 厂商名称
            model: 模型名称
            
        Returns:
            EnhancedModelPricing实例或None
        """
        try:
            # 构建环境变量名称
            provider_upper = provider.upper()
            model_clean = model.replace("-", "_").replace(".", "_").upper()
            
            # 尝试多种环境变量命名格式
            env_patterns = [
                f"{provider_upper}_{model_clean}_INPUT_PRICE",
                f"{provider_upper}_{model_clean}_OUTPUT_PRICE",
                f"{provider_upper}_INPUT_PRICE",
                f"{provider_upper}_OUTPUT_PRICE",
            ]
            
            input_price = None
            output_price = None
            
            # 查找输入价格
            for pattern in [env_patterns[0], env_patterns[2]]:
                if pattern in os.environ:
                    try:
                        input_price = float(os.environ[pattern])
                        break
                    except ValueError:
                        continue
            
            # 查找输出价格
            for pattern in [env_patterns[1], env_patterns[3]]:
                if pattern in os.environ:
                    try:
                        output_price = float(os.environ[pattern])
                        break
                    except ValueError:
                        continue
            
            if input_price is not None and output_price is not None:
                self.logger.info(
                    "从环境变量加载模型价格",
                    provider=provider,
                    model=model,
                    input_price=input_price,
                    output_price=output_price
                )
                
                return EnhancedModelPricing(
                    input_price_per_1k=input_price,
                    output_price_per_1k=output_price,
                    currency=os.environ.get("COST_CURRENCY", "CNY"),
                    source="environment_variable"
                )
            
            return None
            
        except Exception as e:
            self.logger.error("环境变量价格加载失败", provider=provider, model=model, error=str(e))
            return None
    
    def get_cost_tracking_config(self) -> Dict[str, Any]:
        """获取成本跟踪配置
        
        Returns:
            成本跟踪配置字典
        """
        return {
            "enabled": os.environ.get("HARBORAI_COST_TRACKING", "true").lower() == "true",
            "currency": os.environ.get("COST_CURRENCY", "CNY"),
            "retention_days": int(os.environ.get("COST_RETENTION_DAYS", "90"))
        }

class EnhancedPricingCalculator(PricingCalculator):
    """基于现有PricingCalculator的增强版本
    
    支持输入输出成本细分、环境变量配置和动态价格设置
    """
    
    def __init__(self):
        super().__init__()
        self.env_pricing_loader = EnvironmentPricingLoader()
        self.dynamic_pricing: Dict[str, EnhancedModelPricing] = {}
        self.logger = structlog.get_logger(__name__)
        
        # 加载成本跟踪配置
        self.cost_config = self.env_pricing_loader.get_cost_tracking_config()
        
        self.logger.info("增强成本计算器初始化完成", cost_config=self.cost_config)
    
    async def calculate_detailed_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> CostBreakdown:
        """计算详细的成本分解
        
        优先级：动态价格 > 环境变量 > 内置价格
        
        Args:
            provider: 厂商名称
            model: 模型名称
            prompt_tokens: 输入token数量
            completion_tokens: 输出token数量
            
        Returns:
            CostBreakdown实例
        """
        try:
            # 获取模型价格（按优先级）
            pricing = await self._get_model_pricing_with_env(provider, model)
            
            if not pricing:
                self.logger.warning("未找到模型价格配置", provider=provider, model=model)
                return CostBreakdown(
                    input_cost=0.0,
                    output_cost=0.0,
                    total_cost=0.0,
                    currency=self.cost_config["currency"],
                    pricing_source="unknown",
                    pricing_timestamp=datetime.now(timezone.utc).isoformat(),
                    pricing_details={"error": "模型价格配置未找到"}
                )
            
            # 计算输入和输出成本
            input_cost = (prompt_tokens / 1000) * pricing.input_price_per_1k
            output_cost = (completion_tokens / 1000) * pricing.output_price_per_1k
            total_cost = input_cost + output_cost
            
            self.logger.debug(
                "成本计算完成",
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                pricing_source=pricing.source
            )
            
            return CostBreakdown(
                input_cost=input_cost,
                output_cost=output_cost,
                total_cost=total_cost,
                currency=pricing.currency,
                pricing_source=pricing.source,
                pricing_timestamp=datetime.now(timezone.utc).isoformat(),
                pricing_details={
                    "input_price_per_1k": pricing.input_price_per_1k,
                    "output_price_per_1k": pricing.output_price_per_1k,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens
                }
            )
            
        except Exception as e:
            self.logger.error("成本计算失败", provider=provider, model=model, error=str(e))
            return CostBreakdown(
                input_cost=0.0,
                output_cost=0.0,
                total_cost=0.0,
                currency=self.cost_config["currency"],
                pricing_source="unknown",
                pricing_timestamp=datetime.now(timezone.utc).isoformat(),
                pricing_details={"error": str(e)}
            )
    
    async def _get_model_pricing_with_env(self, provider: str, model: str) -> Optional[EnhancedModelPricing]:
        """获取模型价格配置（支持环境变量）
        
        优先级：动态价格 > 环境变量 > 内置价格
        
        Args:
            provider: 厂商名称
            model: 模型名称
            
        Returns:
            EnhancedModelPricing实例或None
        """
        # 1. 检查动态价格
        dynamic_key = f"{provider}:{model}"
        if dynamic_key in self.dynamic_pricing:
            self.logger.debug("使用动态价格配置", provider=provider, model=model)
            return self.dynamic_pricing[dynamic_key]
        
        # 2. 检查环境变量
        env_pricing = self.env_pricing_loader.load_model_pricing(provider, model)
        if env_pricing:
            self.logger.debug("使用环境变量价格配置", provider=provider, model=model)
            return env_pricing
        
        # 3. 使用内置价格
        builtin_pricing = self.get_model_pricing(model)
        if builtin_pricing:
            self.logger.debug("使用内置价格配置", provider=provider, model=model)
            return EnhancedModelPricing(
                input_price_per_1k=builtin_pricing.input_price,
                output_price_per_1k=builtin_pricing.output_price,
                currency=self.cost_config["currency"],
                source="builtin"
            )
        
        return None
    
    def add_dynamic_pricing(
        self,
        provider: str,
        model: str,
        input_price_per_1k: float,
        output_price_per_1k: float,
        currency: str = "CNY"
    ):
        """添加动态价格配置
        
        Args:
            provider: 厂商名称
            model: 模型名称
            input_price_per_1k: 每1K输入tokens的价格
            output_price_per_1k: 每1K输出tokens的价格
            currency: 货币单位
        """
        dynamic_key = f"{provider}:{model}"
        self.dynamic_pricing[dynamic_key] = EnhancedModelPricing(
            input_price_per_1k=input_price_per_1k,
            output_price_per_1k=output_price_per_1k,
            currency=currency,
            source="dynamic"
        )
        
        self.logger.info(
            "添加动态价格配置",
            provider=provider,
            model=model,
            input_price=input_price_per_1k,
            output_price=output_price_per_1k,
            currency=currency
        )
    
    def remove_dynamic_pricing(self, provider: str, model: str):
        """移除动态价格配置
        
        Args:
            provider: 厂商名称
            model: 模型名称
        """
        dynamic_key = f"{provider}:{model}"
        if dynamic_key in self.dynamic_pricing:
            del self.dynamic_pricing[dynamic_key]
            self.logger.info("移除动态价格配置", provider=provider, model=model)
    
    def get_pricing_summary(self) -> Dict[str, Any]:
        """获取价格配置摘要
        
        Returns:
            价格配置摘要字典
        """
        return {
            "cost_tracking_enabled": self.cost_config["enabled"],
            "default_currency": self.cost_config["currency"],
            "retention_days": self.cost_config["retention_days"],
            "builtin_models_count": len(self.MODEL_PRICING),
            "dynamic_models_count": len(self.dynamic_pricing),
            "dynamic_models": list(self.dynamic_pricing.keys())
        }

class CostBreakdownService:
    """成本细分服务
    
    提供成本分析和统计功能
    """
    
    def __init__(self, pricing_calculator: EnhancedPricingCalculator):
        self.pricing_calculator = pricing_calculator
        self.logger = structlog.get_logger(__name__)
    
    async def analyze_cost_efficiency(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> Dict[str, Any]:
        """分析成本效率
        
        Args:
            provider: 厂商名称
            model: 模型名称
            prompt_tokens: 输入token数量
            completion_tokens: 输出token数量
            
        Returns:
            成本效率分析结果
        """
        cost_breakdown = await self.pricing_calculator.calculate_detailed_cost(
            provider, model, prompt_tokens, completion_tokens
        )
        
        total_tokens = prompt_tokens + completion_tokens
        cost_per_token = cost_breakdown.total_cost / total_tokens if total_tokens > 0 else 0
        
        return {
            "cost_breakdown": cost_breakdown.to_dict(),
            "efficiency_metrics": {
                "cost_per_token": round(cost_per_token, 8),
                "cost_per_1k_tokens": round(cost_per_token * 1000, 6),
                "input_output_ratio": completion_tokens / prompt_tokens if prompt_tokens > 0 else 0,
                "total_tokens": total_tokens
            }
        }