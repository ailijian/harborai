#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强Token解析器测试

测试各厂商特殊格式的Token解析功能，包括降级策略和错误恢复机制。
"""

import pytest
import json
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from harborai.core.enhanced_token_parser import (
    EnhancedTokenParser,
    TokenParsingResult,
    VendorType,
    parse_token_usage,
    get_token_parsing_stats
)


class TestEnhancedTokenParser:
    """增强Token解析器测试类"""
    
    def setup_method(self):
        """测试前设置"""
        self.parser = EnhancedTokenParser()
        self.parser.reset_statistics()
        self.parser.clear_cache()
    
    def test_openai_standard_format(self):
        """测试OpenAI标准格式解析"""
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.OPENAI)
        
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.vendor == VendorType.OPENAI
        assert result.parsing_method == "standard"
        assert result.confidence > 0.8
        assert not result.fallback_used
    
    def test_deepseek_reasoning_tokens(self):
        """测试DeepSeek推理Token格式"""
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "reasoning_tokens": 25,
                "total_tokens": 175
            }
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.DEEPSEEK)
        
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 75  # completion + reasoning
        assert result.total_tokens == 175
        assert result.vendor == VendorType.DEEPSEEK
        assert "deepseek_reasoning" in result.parsing_method
        assert result.confidence > 0.8
    
    def test_baidu_ernie_format(self):
        """测试百度ERNIE格式"""
        response_data = {
            "result": {
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 40,
                    "total_tokens": 120
                }
            }
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.BAIDU)
        
        assert result.prompt_tokens == 80
        assert result.completion_tokens == 40
        assert result.total_tokens == 120
        assert result.vendor == VendorType.BAIDU
        assert "baidu_ernie" in result.parsing_method
        assert result.confidence > 0.8
    
    def test_doubao_answer_tokens(self):
        """测试豆包answer_tokens格式"""
        response_data = {
            "usage": {
                "prompt_tokens": 90,
                "answer_tokens": 45,
                "total_tokens": 135
            }
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.BYTEDANCE)
        
        assert result.prompt_tokens == 90
        assert result.completion_tokens == 45
        assert result.total_tokens == 135
        assert result.vendor == VendorType.BYTEDANCE
        assert "doubao_format" in result.parsing_method
        assert result.confidence > 0.8
    
    def test_claude_input_output_tokens(self):
        """测试Claude input/output tokens格式"""
        response_data = {
            "usage": {
                "input_tokens": 120,
                "output_tokens": 60
            }
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.ANTHROPIC)
        
        assert result.prompt_tokens == 120
        assert result.completion_tokens == 60
        assert result.total_tokens == 180
        assert result.vendor == VendorType.ANTHROPIC
        assert "claude_format" in result.parsing_method
        assert result.confidence > 0.8
    
    def test_gemini_usage_metadata(self):
        """测试Gemini usage_metadata格式"""
        response_data = {
            "usage_metadata": {
                "prompt_token_count": 110,
                "candidates_token_count": 55,
                "total_token_count": 165
            }
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.GOOGLE)
        
        assert result.prompt_tokens == 110
        assert result.completion_tokens == 55
        assert result.total_tokens == 165
        assert result.vendor == VendorType.GOOGLE
        assert "gemini_format" in result.parsing_method
        assert result.confidence > 0.8
    
    def test_alternative_field_mapping(self):
        """测试备选字段映射"""
        response_data = {
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150
            }
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.OPENAI)
        
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert "alternative" in result.parsing_method
        assert result.confidence > 0.8
    
    def test_fallback_extraction(self):
        """测试降级提取策略"""
        response_data = {
            "data": {
                "prompt_tokens": 80,
                "completion_tokens": 40,
                "total_tokens": 120
            }
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.UNKNOWN)
        
        assert result.prompt_tokens == 80
        assert result.completion_tokens == 40
        assert result.total_tokens == 120
        assert result.parsing_method == "fallback_extraction"
        assert result.confidence == 0.7
        assert result.fallback_used
    
    def test_content_estimation(self):
        """测试基于内容的Token估算"""
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": "这是一个测试响应内容，用于验证Token估算功能。"
                    }
                }
            ]
        }
        
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "请生成一个测试响应"
                }
            ]
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.UNKNOWN, request_data=request_data)
        
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
        assert result.total_tokens > 0
        assert result.parsing_method == "content_estimation"
        assert result.confidence == 0.5
        assert result.fallback_used
    
    def test_vendor_detection_from_model(self):
        """测试从模型名称检测厂商"""
        test_cases = [
            ("gpt-4", VendorType.OPENAI),
            ("deepseek-chat", VendorType.DEEPSEEK),
            ("ernie-bot", VendorType.BAIDU),
            ("doubao-pro", VendorType.BYTEDANCE),
            ("claude-3", VendorType.ANTHROPIC),
            ("gemini-pro", VendorType.GOOGLE),
            ("unknown-model", VendorType.UNKNOWN)
        ]
        
        for model, expected_vendor in test_cases:
            detected = self.parser._detect_vendor_from_response({}, model)
            assert detected == expected_vendor, f"模型 {model} 应该检测为 {expected_vendor}"
    
    def test_vendor_detection_from_response_structure(self):
        """测试从响应结构检测厂商"""
        # Gemini格式
        response_data = {"usage_metadata": {}}
        detected = self.parser._detect_vendor_from_response(response_data)
        assert detected == VendorType.GOOGLE
        
        # 百度格式
        response_data = {"result": {"usage": {}}}
        detected = self.parser._detect_vendor_from_response(response_data)
        assert detected == VendorType.BAIDU
    
    def test_token_validation(self):
        """测试Token数据验证"""
        # 有效数据
        assert self.parser._validate_token_data(100, 50, 150)
        assert self.parser._validate_token_data(0, 0, 0)
        
        # 无效数据
        assert not self.parser._validate_token_data(-1, 50, 150)  # 负数
        assert not self.parser._validate_token_data(100, 50, 200)  # 总数不匹配
        assert not self.parser._validate_token_data(1000000, 50, 1000050)  # 过大值
        assert not self.parser._validate_token_data("invalid", 50, 150)  # 非数字
    
    def test_vendor_estimation_ratios(self):
        """测试厂商特定的估算比例"""
        ratios = {
            VendorType.OPENAI: 0.25,
            VendorType.DEEPSEEK: 0.25,
            VendorType.BAIDU: 0.5,
            VendorType.BYTEDANCE: 0.4,
            VendorType.ANTHROPIC: 0.25,
            VendorType.GOOGLE: 0.3,
            VendorType.UNKNOWN: 0.25
        }
        
        for vendor, expected_ratio in ratios.items():
            actual_ratio = self.parser._get_vendor_estimation_ratio(vendor)
            assert actual_ratio == expected_ratio
    
    def test_caching_mechanism(self):
        """测试缓存机制"""
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        # 第一次解析
        result1 = self.parser.parse_token_usage(response_data, VendorType.OPENAI)
        stats1 = self.parser.get_parsing_statistics()
        
        # 第二次解析（应该使用缓存）
        result2 = self.parser.parse_token_usage(response_data, VendorType.OPENAI)
        stats2 = self.parser.get_parsing_statistics()
        
        assert result1.prompt_tokens == result2.prompt_tokens
        assert result1.completion_tokens == result2.completion_tokens
        assert stats2['cache_hits'] > stats1['cache_hits']
    
    def test_statistics_tracking(self):
        """测试统计信息跟踪"""
        # 成功解析
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        self.parser.parse_token_usage(response_data, VendorType.OPENAI)
        
        # 降级解析
        response_data = {"unknown_field": "value"}
        self.parser.parse_token_usage(response_data, VendorType.UNKNOWN)
        
        stats = self.parser.get_parsing_statistics()
        
        assert stats['total_parsed'] == 2
        assert stats['successful_parsed'] == 1
        assert stats['fallback_used'] == 1
        assert stats['success_rate'] == 0.5
        assert stats['fallback_rate'] == 0.5
        assert 'openai' in stats['vendor_stats']
        assert 'unknown' in stats['vendor_stats']
    
    def test_error_handling(self):
        """测试错误处理"""
        # 无效响应数据
        result = self.parser.parse_token_usage(None, VendorType.OPENAI)
        assert result.error_message is not None
        assert result.parsing_method == "error"
        
        # 空响应数据
        result = self.parser.parse_token_usage({}, VendorType.OPENAI)
        assert result.fallback_used
        assert result.confidence < 0.5
    
    def test_special_characters_in_content(self):
        """测试内容中的特殊字符处理"""
        response_data = {
            "choices": [
                {
                    "message": {
                        "content": "测试内容包含特殊字符：\n\t\"引号\"、'单引号'、\\反斜杠、emoji😀"
                    }
                }
            ]
        }
        
        request_data = {
            "messages": [
                {
                    "role": "user", 
                    "content": "生成包含特殊字符的内容"
                }
            ]
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.UNKNOWN, request_data=request_data)
        
        assert result.prompt_tokens > 0
        assert result.completion_tokens > 0
        assert result.parsing_method == "content_estimation"
    
    def test_large_token_counts(self):
        """测试大Token数量处理"""
        response_data = {
            "usage": {
                "prompt_tokens": 50000,
                "completion_tokens": 25000,
                "total_tokens": 75000
            }
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.OPENAI)
        
        assert result.prompt_tokens == 50000
        assert result.completion_tokens == 25000
        assert result.total_tokens == 75000
        assert result.confidence > 0.8
    
    def test_zero_token_counts(self):
        """测试零Token数量处理"""
        response_data = {
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
        
        result = self.parser.parse_token_usage(response_data, VendorType.OPENAI)
        
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0
        assert result.confidence > 0.8


class TestConvenienceFunctions:
    """便捷函数测试类"""
    
    def test_parse_token_usage_function(self):
        """测试parse_token_usage便捷函数"""
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        result = parse_token_usage(response_data, "openai")
        
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.vendor == VendorType.OPENAI
    
    def test_get_token_parsing_stats_function(self):
        """测试get_token_parsing_stats便捷函数"""
        # 先进行一些解析操作
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        parse_token_usage(response_data, "openai")
        
        stats = get_token_parsing_stats()
        
        assert 'total_parsed' in stats
        assert 'successful_parsed' in stats
        assert 'success_rate' in stats
        assert stats['total_parsed'] > 0


class TestIntegrationScenarios:
    """集成场景测试类"""
    
    def test_multi_vendor_batch_processing(self):
        """测试多厂商批量处理"""
        parser = EnhancedTokenParser()
        parser.reset_statistics()
        
        test_cases = [
            # OpenAI格式
            ({
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150
                }
            }, VendorType.OPENAI),
            
            # DeepSeek格式
            ({
                "usage": {
                    "prompt_tokens": 80,
                    "completion_tokens": 40,
                    "reasoning_tokens": 20,
                    "total_tokens": 140
                }
            }, VendorType.DEEPSEEK),
            
            # 百度格式
            ({
                "result": {
                    "usage": {
                        "prompt_tokens": 90,
                        "completion_tokens": 45,
                        "total_tokens": 135
                    }
                }
            }, VendorType.BAIDU),
        ]
        
        results = []
        for response_data, vendor in test_cases:
            result = parser.parse_token_usage(response_data, vendor)
            results.append(result)
        
        # 验证所有解析都成功
        for result in results:
            assert result.confidence > 0.8
            assert not result.fallback_used
            assert result.error_message is None
        
        # 验证统计信息
        stats = parser.get_parsing_statistics()
        assert stats['total_parsed'] == 3
        assert stats['successful_parsed'] == 3
        assert stats['success_rate'] == 1.0
    
    def test_degradation_chain(self):
        """测试降级链处理"""
        parser = EnhancedTokenParser()
        
        # 完全无法识别的响应格式
        response_data = {
            "completely_unknown": {
                "some_field": "some_value"
            }
        }
        
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": "测试请求内容"
                }
            ]
        }
        
        result = parser.parse_token_usage(response_data, VendorType.UNKNOWN, request_data=request_data)
        
        # 应该使用内容估算降级策略
        assert result.fallback_used
        assert result.parsing_method == "content_estimation"
        assert result.confidence == 0.5
        assert result.prompt_tokens > 0
        assert result.completion_tokens >= 0  # 可能为0，因为响应中没有内容
    
    def test_performance_with_large_dataset(self):
        """测试大数据集性能"""
        parser = EnhancedTokenParser()
        parser.reset_statistics()
        
        # 模拟大量解析请求
        response_template = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        import time
        start_time = time.time()
        
        for i in range(1000):
            # 稍微变化数据以避免缓存
            response_data = {
                "usage": {
                    "prompt_tokens": 100 + i % 10,
                    "completion_tokens": 50 + i % 5,
                    "total_tokens": 150 + i % 15
                }
            }
            result = parser.parse_token_usage(response_data, VendorType.OPENAI)
            assert result.confidence > 0.8
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 验证性能（应该在合理时间内完成）
        assert processing_time < 5.0  # 1000次解析应该在5秒内完成
        
        stats = parser.get_parsing_statistics()
        assert stats['total_parsed'] == 1000
        assert stats['successful_parsed'] == 1000
        assert stats['success_rate'] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])