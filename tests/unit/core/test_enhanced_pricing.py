# -*- coding: utf-8 -*-
"""
增强定价计算器单元测试

测试EnhancedPricingCalculator、EnhancedModelPricing、CostBreakdown和EnvironmentPricingLoader的功能
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from harborai.core.enhanced_pricing import (
    EnhancedModelPricing,
    CostBreakdown,
    EnvironmentPricingLoader,
    EnhancedPricingCalculator
)
from harborai.core.token_usage import TokenUsage


class TestEnhancedModelPricing:
    """测试EnhancedModelPricing类"""
    
    def test_enhanced_model_pricing_creation(self):
        """测试EnhancedModelPricing创建"""
        pricing = EnhancedModelPricing(
            input_price_per_1k=0.01,
            output_price_per_1k=0.02,
            currency="CNY",
            source="dynamic"
        )
        
        assert pricing.input_price_per_1k == 0.01
        assert pricing.output_price_per_1k == 0.02
        assert pricing.currency == "CNY"
        assert pricing.source == "dynamic"
    
    def test_enhanced_model_pricing_defaults(self):
        """测试EnhancedModelPricing默认值"""
        pricing = EnhancedModelPricing(
            input_price_per_1k=0.01,
            output_price_per_1k=0.02
        )
        
        assert pricing.currency == "CNY"
        assert pricing.source == "builtin"  # 实际默认值


class TestCostBreakdown:
    """测试CostBreakdown类"""
    
    def test_cost_breakdown_creation(self):
        """测试CostBreakdown创建"""
        breakdown = CostBreakdown(
            input_cost=0.05,
            output_cost=0.10,
            total_cost=0.15,
            currency="CNY",
            pricing_source="dynamic",
            pricing_timestamp="2024-01-01T00:00:00Z",
            pricing_details={"test": "data"}
        )
        
        assert breakdown.input_cost == 0.05
        assert breakdown.output_cost == 0.10
        assert breakdown.total_cost == 0.15
        assert breakdown.currency == "CNY"
        assert breakdown.pricing_source == "dynamic"
        assert breakdown.pricing_timestamp == "2024-01-01T00:00:00Z"
        assert breakdown.pricing_details == {"test": "data"}
    
    def test_cost_breakdown_to_dict(self):
        """测试CostBreakdown转换为字典"""
        breakdown = CostBreakdown(
            input_cost=0.05,
            output_cost=0.10,
            total_cost=0.15,
            currency="CNY",
            pricing_source="dynamic",
            pricing_timestamp="2024-01-01T00:00:00Z",
            pricing_details={"test": "data"}
        )
        
        result = breakdown.to_dict()
        expected = {
            "input_cost": 0.05,
            "output_cost": 0.10,
            "total_cost": 0.15,
            "currency": "CNY",
            "pricing_source": "dynamic",
            "pricing_timestamp": "2024-01-01T00:00:00Z",
            "pricing_details": {"test": "data"}
        }
        
        assert result == expected


class TestEnvironmentPricingLoader:
    """测试EnvironmentPricingLoader类"""
    
    def test_load_model_pricing_specific_model(self):
        """测试加载特定模型价格配置"""
        with patch.dict(os.environ, {
            'OPENAI_GPT_4_INPUT_PRICE': '0.03',  # 实际的环境变量格式
            'OPENAI_GPT_4_OUTPUT_PRICE': '0.06',
            'COST_CURRENCY': 'CNY'
        }):
            loader = EnvironmentPricingLoader()
            pricing = loader.load_model_pricing("openai", "gpt-4")
            
            assert pricing is not None
            assert pricing.input_price_per_1k == 0.03
            assert pricing.output_price_per_1k == 0.06
            assert pricing.currency == "CNY"
            assert pricing.source == "environment_variable"  # 实际的source值
    
    def test_load_model_pricing_provider_level(self):
        """测试加载厂商级别价格配置"""
        with patch.dict(os.environ, {
            'OPENAI_INPUT_PRICE': '0.02',
            'OPENAI_OUTPUT_PRICE': '0.04',
            'COST_CURRENCY': 'CNY'
        }):
            loader = EnvironmentPricingLoader()
            pricing = loader.load_model_pricing("openai", "gpt-3.5-turbo")
            
            assert pricing is not None
            assert pricing.input_price_per_1k == 0.02
            assert pricing.output_price_per_1k == 0.04
            assert pricing.currency == "CNY"
            assert pricing.source == "environment_variable"
    
    def test_load_model_pricing_not_found(self):
        """测试未找到价格配置"""
        loader = EnvironmentPricingLoader()
        pricing = loader.load_model_pricing("unknown", "unknown-model")
        
        assert pricing is None
    
    def test_load_model_pricing_invalid_values(self):
        """测试无效价格值"""
        with patch.dict(os.environ, {
            'OPENAI_GPT_4_INPUT_PRICE': 'invalid',
            'OPENAI_GPT_4_OUTPUT_PRICE': '0.06'
        }):
            loader = EnvironmentPricingLoader()
            pricing = loader.load_model_pricing("openai", "gpt-4")
            
            assert pricing is None
    
    def test_get_cost_tracking_config(self):
        """测试获取成本跟踪配置"""
        with patch.dict(os.environ, {
            'HARBORAI_COST_TRACKING': 'true',
            'COST_CURRENCY': 'CNY',
            'COST_RETENTION_DAYS': '30'
        }):
            loader = EnvironmentPricingLoader()
            config = loader.get_cost_tracking_config()
            
            assert config["enabled"] is True
            assert config["currency"] == "CNY"
            assert config["retention_days"] == 30


class TestEnhancedPricingCalculator:
    """测试EnhancedPricingCalculator类"""
    
    def test_enhanced_pricing_calculator_creation(self):
        """测试EnhancedPricingCalculator创建"""
        calculator = EnhancedPricingCalculator()
        
        assert calculator is not None
        assert hasattr(calculator, 'env_pricing_loader')
        assert hasattr(calculator, 'dynamic_pricing')
        assert hasattr(calculator, 'cost_config')
    
    @pytest.mark.asyncio
    async def test_calculate_detailed_cost_builtin_pricing(self):
        """测试使用内置价格计算详细成本"""
        calculator = EnhancedPricingCalculator()
        
        # 模拟内置价格
        with patch.object(calculator, 'get_model_pricing') as mock_get_pricing:
            mock_pricing = MagicMock()
            mock_pricing.input_price = 0.01
            mock_pricing.output_price = 0.02
            mock_get_pricing.return_value = mock_pricing
            
            result = await calculator.calculate_detailed_cost(
                provider="openai",
                model="gpt-3.5-turbo",
                prompt_tokens=1000,
                completion_tokens=500
            )
            
            assert result.input_cost == 0.01  # (1000/1000) * 0.01
            assert result.output_cost == 0.01  # (500/1000) * 0.02
            assert result.total_cost == 0.02
            # 使用实际的默认货币
            assert result.currency in ["CNY", "RMB"]  # 可能是CNY或RMB
            assert result.pricing_source == "builtin"
    
    @pytest.mark.asyncio
    async def test_calculate_detailed_cost_no_pricing(self):
        """测试无价格配置时的成本计算"""
        calculator = EnhancedPricingCalculator()
        
        # 模拟无价格配置
        with patch.object(calculator, 'get_model_pricing', return_value=None):
            with patch.object(calculator.env_pricing_loader, 'load_model_pricing', return_value=None):
                result = await calculator.calculate_detailed_cost(
                    provider="unknown",
                    model="unknown-model",
                    prompt_tokens=1000,
                    completion_tokens=500
                )
                
                assert result.input_cost == 0.0
                assert result.output_cost == 0.0
                assert result.total_cost == 0.0
                # 使用实际的默认货币
                assert result.currency in ["CNY", "RMB"]  # 可能是CNY或RMB
                assert result.pricing_source == "not_found"
    
    def test_add_dynamic_pricing(self):
        """测试添加动态价格配置"""
        calculator = EnhancedPricingCalculator()
        
        calculator.add_dynamic_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            currency="USD"
        )
        
        dynamic_key = "openai:gpt-4"
        assert dynamic_key in calculator.dynamic_pricing
        
        pricing = calculator.dynamic_pricing[dynamic_key]
        assert pricing.input_price_per_1k == 0.03
        assert pricing.output_price_per_1k == 0.06
        assert pricing.currency == "USD"
        assert pricing.source == "dynamic"
    
    def test_remove_dynamic_pricing(self):
        """测试移除动态价格配置"""
        calculator = EnhancedPricingCalculator()
        
        # 先添加
        calculator.add_dynamic_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06
        )
        
        # 再移除
        calculator.remove_dynamic_pricing("openai", "gpt-4")
        
        dynamic_key = "openai:gpt-4"
        assert dynamic_key not in calculator.dynamic_pricing
    
    def test_get_pricing_summary(self):
        """测试获取价格配置摘要"""
        calculator = EnhancedPricingCalculator()
        
        # 添加一些动态价格
        calculator.add_dynamic_pricing("openai", "gpt-4", 0.03, 0.06)
        calculator.add_dynamic_pricing("deepseek", "deepseek-chat", 0.001, 0.002)
        
        summary = calculator.get_pricing_summary()
        
        assert "cost_tracking_enabled" in summary
        assert "default_currency" in summary
        assert "retention_days" in summary
        assert "builtin_models_count" in summary
        assert "dynamic_models_count" in summary
        assert "dynamic_models" in summary
        
        assert summary["dynamic_models_count"] == 2
        assert "openai:gpt-4" in summary["dynamic_models"]
        assert "deepseek:deepseek-chat" in summary["dynamic_models"]
    
    @pytest.mark.asyncio
    async def test_model_pricing_priority(self):
        """测试模型价格优先级：动态 > 环境变量 > 内置"""
        calculator = EnhancedPricingCalculator()
        
        # 设置动态价格
        calculator.add_dynamic_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.05,  # 动态价格
            output_price_per_1k=0.10
        )
        
        # 模拟环境变量价格
        with patch.object(calculator.env_pricing_loader, 'load_model_pricing') as mock_env:
            mock_env.return_value = EnhancedModelPricing(
                input_price_per_1k=0.03,  # 环境变量价格
                output_price_per_1k=0.06,
                source="environment_variable"
            )
            
            # 模拟内置价格
            with patch.object(calculator, 'get_model_pricing') as mock_builtin:
                mock_pricing = MagicMock()
                mock_pricing.input_price = 0.01  # 内置价格
                mock_pricing.output_price = 0.02
                mock_builtin.return_value = mock_pricing
                
                result = await calculator.calculate_detailed_cost(
                    provider="openai",
                    model="gpt-4",
                    prompt_tokens=1000,
                    completion_tokens=1000
                )
                
                # 应该使用动态价格（最高优先级）
                assert result.input_cost == 0.05  # (1000/1000) * 0.05
                assert result.output_cost == 0.10  # (1000/1000) * 0.10
                assert result.pricing_source == "dynamic"