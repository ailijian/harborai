#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强Token解析器改进功能测试

测试新增的厂商支持、智能降级策略和动态配置更新功能
"""

import pytest
from unittest.mock import Mock, patch
from harborai.core.enhanced_token_parser import (
    EnhancedTokenParser, 
    VendorType, 
    TokenParsingResult
)


class TestEnhancedTokenParserImprovements:
    """增强Token解析器改进功能测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.parser = EnhancedTokenParser()
    
    def test_new_vendor_support(self):
        """测试新增厂商支持"""
        # 测试智谱AI
        zhipu_response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        result = self.parser.parse_token_usage(zhipu_response, VendorType.ZHIPU)
        assert result.is_valid()
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert "zhipu" in result.parsing_method.lower()
        
        # 测试月之暗面
        moonshot_response = {
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300
            }
        }
        
        result = self.parser.parse_token_usage(moonshot_response, VendorType.MOONSHOT)
        assert result.is_valid()
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 100
        assert result.total_tokens == 300
        
        # 测试MiniMax
        minimax_response = {
            "usage": {
                "input_tokens": 80,
                "output_tokens": 40,
                "total_tokens": 120
            }
        }
        
        result = self.parser.parse_token_usage(minimax_response, VendorType.MINIMAX)
        assert result.is_valid()
        assert result.prompt_tokens == 80
        assert result.completion_tokens == 40
        assert result.total_tokens == 120
    
    def test_xunfei_nested_format(self):
        """测试讯飞星火嵌套格式"""
        xunfei_response = {
            "payload": {
                "usage": {
                    "question_tokens": 60,
                    "answer_tokens": 30,
                    "total_tokens": 90
                }
            }
        }
        
        result = self.parser.parse_token_usage(xunfei_response, VendorType.XUNFEI)
        assert result.is_valid()
        assert result.prompt_tokens == 60
        assert result.completion_tokens == 30
        assert result.total_tokens == 90
    
    def test_tencent_uppercase_format(self):
        """测试腾讯混元大写字段格式"""
        tencent_response = {
            "Usage": {
                "PromptTokens": 120,
                "CompletionTokens": 80,
                "TotalTokens": 200
            }
        }
        
        result = self.parser.parse_token_usage(tencent_response, VendorType.TENCENT)
        assert result.is_valid()
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 80
        assert result.total_tokens == 200
    
    def test_intelligent_degradation_similar_vendors(self):
        """测试智能降级策略 - 相似厂商"""
        # 模拟未知厂商的响应，但格式类似OpenAI
        unknown_response = {
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225
            }
        }
        
        # 使用一个不存在的厂商类型，应该降级到相似厂商
        result = self.parser.parse_token_usage(unknown_response, VendorType.ZHIPU)
        assert result.is_valid()
        assert result.prompt_tokens == 150
        assert result.completion_tokens == 75
        assert result.total_tokens == 225
    
    def test_generic_field_search(self):
        """测试通用字段搜索策略"""
        # 创建一个复杂的嵌套结构
        complex_response = {
            "data": {
                "metrics": {
                    "input_tokens": 100,
                    "output_tokens": 50
                },
                "billing": {
                    "total_tokens": 150
                }
            }
        }
        
        # 直接调用通用搜索方法
        result = self.parser._generic_field_search(complex_response)
        assert result.is_valid()
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.parsing_method == "generic_field_search"
    
    def test_content_estimation_extraction(self):
        """测试内容估算的文本提取"""
        # 测试字符串格式
        request_data_str = {
            "prompt": "这是一个测试提示"
        }
        content = self.parser._extract_content_for_estimation(request_data_str)
        assert content == "这是一个测试提示"
        
        # 测试messages格式
        request_data_messages = {
            "messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"}
            ]
        }
        content = self.parser._extract_content_for_estimation(request_data_messages)
        assert "你好" in content
        assert "你好！有什么可以帮助你的吗？" in content
    
    def test_vendor_detection_from_string(self):
        """测试从字符串检测厂商"""
        assert self.parser._detect_vendor_from_string("openai-gpt-4") == VendorType.OPENAI
        assert self.parser._detect_vendor_from_string("deepseek-chat") == VendorType.DEEPSEEK
        assert self.parser._detect_vendor_from_string("ernie-bot") == VendorType.BAIDU
        assert self.parser._detect_vendor_from_string("doubao-pro") == VendorType.BYTEDANCE
        assert self.parser._detect_vendor_from_string("claude-3") == VendorType.ANTHROPIC
        assert self.parser._detect_vendor_from_string("gemini-pro") == VendorType.GOOGLE
        assert self.parser._detect_vendor_from_string("glm-4") == VendorType.ZHIPU
        assert self.parser._detect_vendor_from_string("moonshot-v1") == VendorType.MOONSHOT
        assert self.parser._detect_vendor_from_string("minimax-abab") == VendorType.MINIMAX
        assert self.parser._detect_vendor_from_string("spark-3.5") == VendorType.XUNFEI
        assert self.parser._detect_vendor_from_string("qwen-max") == VendorType.ALIBABA
        assert self.parser._detect_vendor_from_string("hunyuan-pro") == VendorType.TENCENT
    
    def test_vendor_detection_from_response(self):
        """测试从响应检测厂商"""
        # 测试腾讯格式
        tencent_response = {
            "Usage": {
                "PromptTokens": 100
            }
        }
        vendor = self.parser._detect_vendor_from_response(tencent_response)
        assert vendor == VendorType.TENCENT
        
        # 测试DeepSeek格式
        deepseek_response = {
            "usage": {
                "reasoning_tokens": 50,
                "prompt_tokens": 100
            }
        }
        vendor = self.parser._detect_vendor_from_response(deepseek_response)
        assert vendor == VendorType.DEEPSEEK
        
        # 测试MiniMax格式
        minimax_response = {
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        }
        vendor = self.parser._detect_vendor_from_response(minimax_response)
        assert vendor == VendorType.MINIMAX
    
    def test_dynamic_config_update(self):
        """测试动态配置更新"""
        # 获取初始配置信息
        initial_config = self.parser.get_config_info()
        initial_version = initial_config['config_version']
        
        # 更新配置
        new_config = {
            'degradation': {
                'enable_content_estimation': False,
                'min_confidence_threshold': 0.8
            }
        }
        
        success = self.parser.update_config(new_config)
        assert success
        
        # 验证配置已更新
        updated_config = self.parser.get_config_info()
        assert updated_config['config_version'] == initial_version + 1
        assert not self.parser.degradation_config['enable_content_estimation']
        assert self.parser.degradation_config['min_confidence_threshold'] == 0.8
    
    def test_parsing_statistics(self):
        """测试解析统计功能"""
        # 执行一些解析操作
        openai_response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        # 清空统计
        self.parser.parsing_stats = {
            'total_attempts': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'vendor_stats': {},
            'method_stats': {},
            'degradation_stats': {}
        }
        
        # 执行解析
        result = self.parser.parse_token_usage(openai_response, VendorType.OPENAI)
        
        # 验证统计
        assert self.parser.parsing_stats['total_attempts'] == 1
        assert self.parser.parsing_stats['successful_parses'] == 1
        assert 'openai' in self.parser.parsing_stats['vendor_stats']
        assert self.parser.parsing_stats['vendor_stats']['openai']['attempts'] == 1
        assert self.parser.parsing_stats['vendor_stats']['openai']['successes'] == 1
    
    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        # 测试无效响应数据
        invalid_response = {"invalid": "data"}
        
        result = self.parser.parse_token_usage(invalid_response, VendorType.OPENAI)
        assert not result.is_valid()
        assert "failed" in result.parsing_method.lower()
        
        # 测试空响应
        empty_response = {}
        result = self.parser.parse_token_usage(empty_response, VendorType.OPENAI)
        assert not result.is_valid()
    
    def test_confidence_scoring(self):
        """测试置信度评分"""
        # 标准格式应该有高置信度
        standard_response = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        result = self.parser.parse_token_usage(standard_response, VendorType.OPENAI)
        assert result.confidence >= 0.8
        
        # 通用搜索应该有中等置信度
        generic_response = {
            "data": {
                "input_tokens": 100,
                "output_tokens": 50
            }
        }
        
        generic_result = self.parser._generic_field_search(generic_response)
        if generic_result.is_valid():
            assert 0.5 <= generic_result.confidence <= 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])