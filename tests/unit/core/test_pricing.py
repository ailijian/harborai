#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""价格计算模块测试"""

import pytest
from harborai.core.pricing import PricingCalculator, ModelPricing


class TestModelPricing:
    """ModelPricing数据类测试"""
    
    def test_model_pricing_creation(self):
        """测试ModelPricing创建"""
        pricing = ModelPricing(input_price=0.001, output_price=0.002)
        assert pricing.input_price == 0.001
        assert pricing.output_price == 0.002
    
    def test_model_pricing_equality(self):
        """测试ModelPricing相等性"""
        pricing1 = ModelPricing(input_price=0.001, output_price=0.002)
        pricing2 = ModelPricing(input_price=0.001, output_price=0.002)
        pricing3 = ModelPricing(input_price=0.002, output_price=0.003)
        
        assert pricing1 == pricing2
        assert pricing1 != pricing3


class TestPricingCalculator:
    """PricingCalculator测试"""
    
    def test_calculate_cost_known_model(self):
        """测试已知模型的成本计算"""
        # 测试deepseek-chat模型
        cost = PricingCalculator.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="deepseek-chat"
        )
        # 1000 tokens * 0.00014 + 500 tokens * 0.00028 = 0.14 + 0.14 = 0.28
        expected_cost = (1000 / 1000) * 0.00014 + (500 / 1000) * 0.00028
        assert cost == expected_cost
    
    def test_calculate_cost_unknown_model(self):
        """测试未知模型的成本计算"""
        cost = PricingCalculator.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="unknown-model"
        )
        assert cost is None
    
    def test_calculate_cost_zero_tokens(self):
        """测试零token的成本计算"""
        cost = PricingCalculator.calculate_cost(
            input_tokens=0,
            output_tokens=0,
            model_name="gpt-3.5-turbo"
        )
        assert cost == 0.0
    
    def test_calculate_cost_large_numbers(self):
        """测试大数量token的成本计算"""
        cost = PricingCalculator.calculate_cost(
            input_tokens=100000,
            output_tokens=50000,
            model_name="gpt-4"
        )
        # 100000/1000 * 0.03 + 50000/1000 * 0.06 = 3.0 + 3.0 = 6.0
        expected_cost = (100000 / 1000) * 0.03 + (50000 / 1000) * 0.06
        assert cost == expected_cost
    
    @pytest.mark.parametrize("model_name,input_tokens,output_tokens", [
        ("gpt-3.5-turbo", 1000, 500),
        ("gpt-4", 2000, 1000),
        ("gpt-4o", 1500, 750),
        ("ernie-3.5-8k", 1200, 600),
        ("doubao-1-5-pro-32k-character-250715", 800, 400),
    ])
    def test_calculate_cost_various_models(self, model_name, input_tokens, output_tokens):
        """测试各种模型的成本计算"""
        cost = PricingCalculator.calculate_cost(input_tokens, output_tokens, model_name)
        assert cost is not None
        assert cost >= 0
    
    def test_get_model_pricing_existing(self):
        """测试获取存在的模型价格配置"""
        pricing = PricingCalculator.get_model_pricing("deepseek-chat")
        assert pricing is not None
        assert pricing.input_price == 0.00014
        assert pricing.output_price == 0.00028
    
    def test_get_model_pricing_non_existing(self):
        """测试获取不存在的模型价格配置"""
        pricing = PricingCalculator.get_model_pricing("non-existing-model")
        assert pricing is None
    
    def test_add_model_pricing(self):
        """测试添加模型价格配置"""
        model_name = "test-model"
        input_price = 0.005
        output_price = 0.01
        
        # 确保模型不存在
        assert PricingCalculator.get_model_pricing(model_name) is None
        
        # 添加模型价格
        PricingCalculator.add_model_pricing(model_name, input_price, output_price)
        
        # 验证添加成功
        pricing = PricingCalculator.get_model_pricing(model_name)
        assert pricing is not None
        assert pricing.input_price == input_price
        assert pricing.output_price == output_price
        
        # 验证可以计算成本
        cost = PricingCalculator.calculate_cost(1000, 500, model_name)
        expected_cost = (1000 / 1000) * input_price + (500 / 1000) * output_price
        assert cost == expected_cost
        
        # 清理：移除测试模型
        del PricingCalculator.MODEL_PRICING[model_name]
    
    def test_add_model_pricing_overwrite(self):
        """测试覆盖现有模型价格配置"""
        model_name = "gpt-3.5-turbo"
        original_pricing = PricingCalculator.get_model_pricing(model_name)
        
        new_input_price = 0.999
        new_output_price = 0.888
        
        # 覆盖价格配置
        PricingCalculator.add_model_pricing(model_name, new_input_price, new_output_price)
        
        # 验证覆盖成功
        pricing = PricingCalculator.get_model_pricing(model_name)
        assert pricing.input_price == new_input_price
        assert pricing.output_price == new_output_price
        
        # 恢复原始价格配置
        PricingCalculator.MODEL_PRICING[model_name] = original_pricing
    
    def test_list_supported_models(self):
        """测试列出支持的模型"""
        models = PricingCalculator.list_supported_models()
        
        # 验证返回类型
        assert isinstance(models, list)
        assert len(models) > 0
        
        # 验证包含已知模型
        expected_models = [
            "deepseek-chat",
            "gpt-3.5-turbo",
            "gpt-4",
            "ernie-3.5-8k"
        ]
        for model in expected_models:
            assert model in models
        
        # 验证所有模型都有价格配置
        for model in models:
            pricing = PricingCalculator.get_model_pricing(model)
            assert pricing is not None
    
    def test_model_pricing_consistency(self):
        """测试模型价格配置的一致性"""
        for model_name in PricingCalculator.list_supported_models():
            pricing = PricingCalculator.get_model_pricing(model_name)
            
            # 验证价格为正数
            assert pricing.input_price > 0
            assert pricing.output_price > 0
            
            # 验证价格合理性（通常输出价格高于输入价格）
            assert pricing.output_price >= pricing.input_price
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试负数token（虽然实际不应该发生）
        cost = PricingCalculator.calculate_cost(-100, -50, "gpt-3.5-turbo")
        assert cost < 0  # 应该返回负成本
        
        # 测试非常小的token数量
        cost = PricingCalculator.calculate_cost(1, 1, "gpt-3.5-turbo")
        assert cost > 0
        assert cost < 0.01  # 应该是很小的成本
    
    def test_precision(self):
        """测试计算精度"""
        # 使用精确的数字进行测试
        cost = PricingCalculator.calculate_cost(1234, 5678, "deepseek-chat")
        
        # 手动计算期望值
        expected_input_cost = (1234 / 1000) * 0.00014
        expected_output_cost = (5678 / 1000) * 0.00028
        expected_total = expected_input_cost + expected_output_cost
        
        # 验证精度
        assert abs(cost - expected_total) < 1e-10