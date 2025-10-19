#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token解析服务单元测试

测试各厂商Token解析器的功能和数据一致性。
根据VIBE规则实现，使用中文注释和文档。
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone

from harborai.core.token_parsing import (
    BaseTokenParser,
    DeepSeekTokenParser,
    OpenAITokenParser,
    DoubaoTokenParser,
    WenxinTokenParser,
    TokenParsingService
)
from harborai.core.token_usage import TokenUsage


class TestBaseTokenParser:
    """BaseTokenParser抽象基类测试"""
    
    def test_base_token_parser_initialization(self):
        """测试基类初始化"""
        # 由于是抽象类，不能直接实例化，需要创建子类
        class TestParser(BaseTokenParser):
            async def parse_tokens(self, response_data, model):
                return TokenUsage(0, 0, 0)
        
        parser = TestParser("test_provider")
        assert parser.provider_name == "test_provider"
        assert parser.logger is not None
    
    def test_extract_usage_data_standard_format(self):
        """测试标准格式的usage数据提取"""
        class TestParser(BaseTokenParser):
            async def parse_tokens(self, response_data, model):
                return TokenUsage(0, 0, 0)
        
        parser = TestParser("test")
        
        # 测试标准usage字段
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        usage_data = parser._extract_usage_data(response_data)
        assert usage_data is not None
        assert usage_data["prompt_tokens"] == 100
        assert usage_data["completion_tokens"] == 50
        assert usage_data["total_tokens"] == 150
    
    def test_extract_usage_data_nested_format(self):
        """测试嵌套格式的usage数据提取"""
        class TestParser(BaseTokenParser):
            async def parse_tokens(self, response_data, model):
                return TokenUsage(0, 0, 0)
        
        parser = TestParser("test")
        
        # 测试嵌套的usage字段
        response_data = {
            "data": {
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 100,
                    "total_tokens": 300
                }
            }
        }
        
        usage_data = parser._extract_usage_data(response_data)
        assert usage_data is not None
        assert usage_data["prompt_tokens"] == 200
    
    def test_extract_usage_data_not_found(self):
        """测试未找到usage数据的情况"""
        class TestParser(BaseTokenParser):
            async def parse_tokens(self, response_data, model):
                return TokenUsage(0, 0, 0)
        
        parser = TestParser("test")
        
        # 测试没有usage字段的响应
        response_data = {
            "result": "success",
            "message": "completed"
        }
        
        usage_data = parser._extract_usage_data(response_data)
        assert usage_data is None


class TestDeepSeekTokenParser:
    """DeepSeek Token解析器测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.parser = DeepSeekTokenParser()
    
    @pytest.mark.asyncio
    async def test_parse_tokens_success(self):
        """测试成功解析DeepSeek Token使用量"""
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        result = await self.parser.parse_tokens(response_data, "deepseek-chat")
        
        assert isinstance(result, TokenUsage)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.parsing_method == "deepseek_direct"
        assert result.confidence == 1.0
        assert result.raw_data == response_data["usage"]
    
    @pytest.mark.asyncio
    async def test_parse_tokens_missing_usage(self):
        """测试缺少usage数据的情况"""
        response_data = {
            "result": "success",
            "message": "completed"
        }
        
        with patch.object(self.parser, 'logger') as mock_logger:
            result = await self.parser.parse_tokens(response_data, "deepseek-chat")
            
            assert isinstance(result, TokenUsage)
            assert result.prompt_tokens == 0
            assert result.completion_tokens == 0
            assert result.total_tokens == 0
            assert result.parsing_method == "fallback_zero"
            assert result.confidence == 0.0
            
            # 验证日志记录
            mock_logger.warning.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_parse_tokens_partial_data(self):
        """测试部分Token数据的情况"""
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50
                # 缺少total_tokens
            }
        }
        
        result = await self.parser.parse_tokens(response_data, "deepseek-chat")
        
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150  # 自动计算
        assert result.parsing_method == "deepseek_direct"
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_parse_tokens_exception_handling(self):
        """测试异常处理"""
        # 模拟异常情况
        response_data = None
        
        with patch.object(self.parser, 'logger') as mock_logger:
            result = await self.parser.parse_tokens(response_data, "deepseek-chat")
            
            assert isinstance(result, TokenUsage)
            assert result.prompt_tokens == 0
            assert result.completion_tokens == 0
            assert result.total_tokens == 0
            assert result.parsing_method == "error_fallback"
            assert result.confidence == 0.0
            
            # 验证错误日志记录
            mock_logger.error.assert_called_once()


class TestOpenAITokenParser:
    """OpenAI Token解析器测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.parser = OpenAITokenParser()
    
    @pytest.mark.asyncio
    async def test_parse_tokens_success(self):
        """测试成功解析OpenAI Token使用量"""
        response_data = {
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300
            }
        }
        
        result = await self.parser.parse_tokens(response_data, "gpt-3.5-turbo")
        
        assert isinstance(result, TokenUsage)
        assert result.prompt_tokens == 200
        assert result.completion_tokens == 100
        assert result.total_tokens == 300
        assert result.parsing_method == "openai_direct"
        assert result.confidence == 1.0
        assert result.raw_data == response_data["usage"]
    
    @pytest.mark.asyncio
    async def test_parse_tokens_missing_usage(self):
        """测试缺少usage数据的情况"""
        response_data = {
            "choices": [{"message": {"content": "Hello"}}]
        }
        
        with patch.object(self.parser, 'logger') as mock_logger:
            result = await self.parser.parse_tokens(response_data, "gpt-3.5-turbo")
            
            assert result.parsing_method == "fallback_zero"
            assert result.confidence == 0.0
            mock_logger.warning.assert_called_once()


class TestDoubaoTokenParser:
    """豆包Token解析器测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.parser = DoubaoTokenParser()
    
    @pytest.mark.asyncio
    async def test_parse_tokens_standard_format(self):
        """测试标准格式的豆包Token解析"""
        response_data = {
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225
            }
        }
        
        result = await self.parser.parse_tokens(response_data, "doubao-pro")
        
        assert result.prompt_tokens == 150
        assert result.completion_tokens == 75
        assert result.total_tokens == 225
        assert result.parsing_method == "doubao_direct"
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_parse_tokens_alternative_field_names(self):
        """测试豆包的替代字段名"""
        response_data = {
            "usage": {
                "input_tokens": 120,
                "output_tokens": 80,
                "total_tokens": 200
            }
        }
        
        result = await self.parser.parse_tokens(response_data, "doubao-pro")
        
        assert result.prompt_tokens == 120  # 从input_tokens映射
        assert result.completion_tokens == 80  # 从output_tokens映射
        assert result.total_tokens == 200
        assert result.parsing_method == "doubao_direct"


class TestWenxinTokenParser:
    """文心一言Token解析器测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.parser = WenxinTokenParser()
    
    @pytest.mark.asyncio
    async def test_parse_tokens_standard_format(self):
        """测试标准格式的文心一言Token解析"""
        response_data = {
            "usage": {
                "prompt_tokens": 180,
                "completion_tokens": 90,
                "total_tokens": 270
            }
        }
        
        result = await self.parser.parse_tokens(response_data, "ernie-3.5")
        
        assert result.prompt_tokens == 180
        assert result.completion_tokens == 90
        assert result.total_tokens == 270
        assert result.parsing_method == "wenxin_direct"
        assert result.confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_parse_tokens_alternative_field_names(self):
        """测试文心一言的替代字段名"""
        response_data = {
            "usage": {
                "prompt_token": 160,
                "completion_token": 85,
                "total_token": 245
            }
        }
        
        result = await self.parser.parse_tokens(response_data, "ernie-4.0")
        
        assert result.prompt_tokens == 160  # 从prompt_token映射
        assert result.completion_tokens == 85  # 从completion_token映射
        assert result.total_tokens == 245  # 从total_token映射
        assert result.parsing_method == "wenxin_direct"


class TestTokenParsingService:
    """Token解析服务测试"""
    
    def setup_method(self):
        """测试前置设置"""
        self.service = TokenParsingService()
    
    def test_initialization(self):
        """测试服务初始化"""
        assert "deepseek" in self.service.provider_parsers
        assert "openai" in self.service.provider_parsers
        assert "doubao" in self.service.provider_parsers
        assert "wenxin" in self.service.provider_parsers
        assert self.service.logger is not None
    
    def test_get_supported_providers(self):
        """测试获取支持的厂商列表"""
        providers = self.service.get_supported_providers()
        
        assert isinstance(providers, list)
        assert "deepseek" in providers
        assert "openai" in providers
        assert "doubao" in providers
        assert "wenxin" in providers
    
    @pytest.mark.asyncio
    async def test_parse_token_usage_success(self):
        """测试成功解析Token使用量"""
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        result = await self.service.parse_token_usage("deepseek", "deepseek-chat", response_data)
        
        assert isinstance(result, TokenUsage)
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.parsing_method == "deepseek_direct"
    
    @pytest.mark.asyncio
    async def test_parse_token_usage_unsupported_provider(self):
        """测试不支持的厂商"""
        response_data = {"usage": {"prompt_tokens": 100}}
        
        with pytest.raises(ValueError, match="不支持的提供商"):
            await self.service.parse_token_usage("unsupported", "model", response_data)
    
    @pytest.mark.asyncio
    async def test_parse_token_usage_case_insensitive(self):
        """测试厂商名称大小写不敏感"""
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        # 测试大写
        result = await self.service.parse_token_usage("DEEPSEEK", "deepseek-chat", response_data)
        assert result.parsing_method == "deepseek_direct"
        
        # 测试混合大小写
        result = await self.service.parse_token_usage("OpenAI", "gpt-3.5-turbo", response_data)
        assert result.parsing_method == "openai_direct"
    
    @pytest.mark.asyncio
    async def test_parse_token_usage_service_exception(self):
        """测试服务级别异常处理"""
        # 模拟解析器抛出异常
        with patch.object(self.service.provider_parsers["deepseek"], "parse_tokens", side_effect=Exception("测试异常")):
            with patch.object(self.service, 'logger') as mock_logger:
                result = await self.service.parse_token_usage("deepseek", "deepseek-chat", {})
                
                assert result.parsing_method == "service_error_fallback"
                assert result.confidence == 0.0
                mock_logger.error.assert_called_once()
    
    def test_add_provider_parser(self):
        """测试添加新的厂商解析器"""
        # 创建自定义解析器
        class CustomParser(BaseTokenParser):
            async def parse_tokens(self, response_data, model):
                return TokenUsage(0, 0, 0, parsing_method="custom")
        
        custom_parser = CustomParser("custom")
        
        with patch.object(self.service, 'logger') as mock_logger:
            self.service.add_provider_parser("custom", custom_parser)
            
            assert "custom" in self.service.provider_parsers
            assert self.service.provider_parsers["custom"] == custom_parser
            mock_logger.info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_parse_token_usage_logging(self):
        """测试解析过程的日志记录"""
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        with patch.object(self.service, 'logger') as mock_logger:
            await self.service.parse_token_usage("deepseek", "deepseek-chat", response_data)
            
            # 验证调试日志和信息日志都被调用
            mock_logger.debug.assert_called_once()
            mock_logger.info.assert_called_once()


class TestTokenParsingIntegration:
    """Token解析集成测试"""
    
    @pytest.mark.asyncio
    async def test_all_providers_integration(self):
        """测试所有厂商解析器的集成"""
        service = TokenParsingService()
        
        # 通用的响应数据格式
        response_data = {
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }
        
        providers = ["deepseek", "openai", "doubao", "wenxin"]
        
        for provider in providers:
            result = await service.parse_token_usage(provider, f"{provider}-model", response_data)
            
            assert isinstance(result, TokenUsage)
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 50
            assert result.total_tokens == 150
            assert result.confidence == 1.0
            assert f"{provider}_direct" in result.parsing_method
    
    @pytest.mark.asyncio
    async def test_error_recovery_chain(self):
        """测试错误恢复链"""
        service = TokenParsingService()
        
        # 测试各种错误情况的恢复
        test_cases = [
            # 空响应
            {},
            # 无效响应
            {"error": "invalid request"},
            # 部分数据
            {"usage": {"prompt_tokens": 100}},
            # None值
            None
        ]
        
        for response_data in test_cases:
            try:
                result = await service.parse_token_usage("deepseek", "deepseek-chat", response_data)
                
                # 所有情况都应该返回有效的TokenUsage对象
                assert isinstance(result, TokenUsage)
                assert result.confidence >= 0.0
                assert result.parsing_method is not None
                
            except Exception as e:
                # 只有不支持的厂商应该抛出异常
                assert "不支持的提供商" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])