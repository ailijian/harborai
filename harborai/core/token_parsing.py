#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token解析服务

实现各厂商专用的Token解析器，从厂商响应中直接解析Token使用量。
根据HarborAI日志系统重构设计方案实现。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import structlog
from .token_usage import TokenUsage

logger = structlog.get_logger(__name__)

class BaseTokenParser(ABC):
    """Token解析器抽象基类
    
    定义各厂商Token解析器的统一接口
    """
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.logger = structlog.get_logger(__name__).bind(provider=provider_name)
    
    @abstractmethod
    async def parse_tokens(self, response_data: Dict[str, Any], model: str) -> TokenUsage:
        """从厂商响应中解析Token使用量
        
        Args:
            response_data: 厂商API响应数据
            model: 模型名称
            
        Returns:
            TokenUsage实例
            
        Raises:
            ValueError: 当响应数据格式不正确时
        """
        pass
    
    def _extract_usage_data(self, response_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """提取usage数据的通用方法
        
        Args:
            response_data: 厂商API响应数据
            
        Returns:
            usage数据字典或None
        """
        # 常见的usage字段位置
        usage_paths = [
            "usage",
            "data.usage", 
            "result.usage",
            "response.usage"
        ]
        
        for path in usage_paths:
            try:
                current = response_data
                for key in path.split('.'):
                    current = current.get(key, {})
                if current and isinstance(current, dict):
                    return current
            except (AttributeError, TypeError):
                continue
        
        return None

class DeepSeekTokenParser(BaseTokenParser):
    """DeepSeek Token解析器
    
    解析DeepSeek API响应中的Token使用量
    """
    
    def __init__(self):
        super().__init__("deepseek")
    
    async def parse_tokens(self, response_data: Dict[str, Any], model: str) -> TokenUsage:
        """解析DeepSeek响应中的Token使用量
        
        DeepSeek响应格式：
        {
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 49,
                "total_tokens": 70
            }
        }
        """
        try:
            usage_data = self._extract_usage_data(response_data)
            
            if not usage_data:
                self.logger.warning("DeepSeek响应中未找到usage数据", response_keys=list(response_data.keys()))
                return TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    parsing_method="fallback_zero",
                    confidence=0.0,
                    raw_data=response_data
                )
            
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
            total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)
            
            self.logger.debug(
                "成功解析DeepSeek Token使用量",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=model
            )
            
            return TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                parsing_method="deepseek_direct",
                confidence=1.0,
                raw_data=usage_data
            )
            
        except Exception as e:
            self.logger.error("DeepSeek Token解析失败", error=str(e), response_data=response_data)
            return TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                parsing_method="error_fallback",
                confidence=0.0,
                raw_data=response_data
            )

class OpenAITokenParser(BaseTokenParser):
    """OpenAI Token解析器
    
    解析OpenAI API响应中的Token使用量
    """
    
    def __init__(self):
        super().__init__("openai")
    
    async def parse_tokens(self, response_data: Dict[str, Any], model: str) -> TokenUsage:
        """解析OpenAI响应中的Token使用量
        
        OpenAI响应格式：
        {
            "usage": {
                "prompt_tokens": 21,
                "completion_tokens": 49,
                "total_tokens": 70
            }
        }
        """
        try:
            usage_data = self._extract_usage_data(response_data)
            
            if not usage_data:
                self.logger.warning("OpenAI响应中未找到usage数据", response_keys=list(response_data.keys()))
                return TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    parsing_method="fallback_zero",
                    confidence=0.0,
                    raw_data=response_data
                )
            
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
            total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)
            
            self.logger.debug(
                "成功解析OpenAI Token使用量",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=model
            )
            
            return TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                parsing_method="openai_direct",
                confidence=1.0,
                raw_data=usage_data
            )
            
        except Exception as e:
            self.logger.error("OpenAI Token解析失败", error=str(e), response_data=response_data)
            return TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                parsing_method="error_fallback",
                confidence=0.0,
                raw_data=response_data
            )

class DoubaoTokenParser(BaseTokenParser):
    """豆包 Token解析器
    
    解析豆包API响应中的Token使用量
    """
    
    def __init__(self):
        super().__init__("doubao")
    
    async def parse_tokens(self, response_data: Dict[str, Any], model: str) -> TokenUsage:
        """解析豆包响应中的Token使用量
        
        豆包响应格式可能有所不同，需要适配
        """
        try:
            usage_data = self._extract_usage_data(response_data)
            
            if not usage_data:
                self.logger.warning("豆包响应中未找到usage数据", response_keys=list(response_data.keys()))
                return TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    parsing_method="fallback_zero",
                    confidence=0.0,
                    raw_data=response_data
                )
            
            # 豆包可能使用不同的字段名，需要适配
            prompt_tokens = usage_data.get("prompt_tokens", usage_data.get("input_tokens", 0))
            completion_tokens = usage_data.get("completion_tokens", usage_data.get("output_tokens", 0))
            total_tokens = usage_data.get("total_tokens", prompt_tokens + completion_tokens)
            
            self.logger.debug(
                "成功解析豆包Token使用量",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=model
            )
            
            return TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                parsing_method="doubao_direct",
                confidence=1.0,
                raw_data=usage_data
            )
            
        except Exception as e:
            self.logger.error("豆包Token解析失败", error=str(e), response_data=response_data)
            return TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                parsing_method="error_fallback",
                confidence=0.0,
                raw_data=response_data
            )

class WenxinTokenParser(BaseTokenParser):
    """文心一言 Token解析器
    
    解析文心一言API响应中的Token使用量
    """
    
    def __init__(self):
        super().__init__("wenxin")
    
    async def parse_tokens(self, response_data: Dict[str, Any], model: str) -> TokenUsage:
        """解析文心一言响应中的Token使用量
        
        文心一言响应格式可能有所不同，需要适配
        """
        try:
            usage_data = self._extract_usage_data(response_data)
            
            if not usage_data:
                self.logger.warning("文心一言响应中未找到usage数据", response_keys=list(response_data.keys()))
                return TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    parsing_method="fallback_zero",
                    confidence=0.0,
                    raw_data=response_data
                )
            
            # 文心一言可能使用不同的字段名，需要适配
            prompt_tokens = usage_data.get("prompt_tokens", usage_data.get("prompt_token", 0))
            completion_tokens = usage_data.get("completion_tokens", usage_data.get("completion_token", 0))
            total_tokens = usage_data.get("total_tokens", usage_data.get("total_token", prompt_tokens + completion_tokens))
            
            self.logger.debug(
                "成功解析文心一言Token使用量",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model=model
            )
            
            return TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                parsing_method="wenxin_direct",
                confidence=1.0,
                raw_data=usage_data
            )
            
        except Exception as e:
            self.logger.error("文心一言Token解析失败", error=str(e), response_data=response_data)
            return TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                parsing_method="error_fallback",
                confidence=0.0,
                raw_data=response_data
            )

class TokenParsingService:
    """Token数据解析服务
    
    统一管理各厂商的Token解析器，从厂商响应中直接解析Token使用量
    """
    
    def __init__(self):
        self.provider_parsers = {
            "deepseek": DeepSeekTokenParser(),
            "openai": OpenAITokenParser(),
            "doubao": DoubaoTokenParser(),
            "wenxin": WenxinTokenParser(),
        }
        self.logger = structlog.get_logger(__name__)
    
    async def parse_token_usage(
        self, 
        provider: str,
        model: str,
        response_data: Dict[str, Any]
    ) -> TokenUsage:
        """从厂商响应中解析Token使用量
        
        Args:
            provider: 厂商名称 (deepseek, openai, doubao, wenxin)
            model: 模型名称
            response_data: 厂商API响应数据
            
        Returns:
            TokenUsage实例
            
        Raises:
            ValueError: 当不支持的厂商时
        """
        parser = self.provider_parsers.get(provider.lower())
        if not parser:
            self.logger.error("不支持的厂商", provider=provider, supported_providers=list(self.provider_parsers.keys()))
            raise ValueError(f"不支持的提供商: {provider}")
        
        self.logger.debug("开始解析Token使用量", provider=provider, model=model)
        
        try:
            token_usage = await parser.parse_tokens(response_data, model)
            
            # 记录解析结果
            self.logger.info(
                "Token解析完成",
                provider=provider,
                model=model,
                prompt_tokens=token_usage.prompt_tokens,
                completion_tokens=token_usage.completion_tokens,
                total_tokens=token_usage.total_tokens,
                parsing_method=token_usage.parsing_method,
                confidence=token_usage.confidence
            )
            
            return token_usage
            
        except Exception as e:
            self.logger.error("Token解析服务异常", provider=provider, model=model, error=str(e))
            # 返回零值Token使用量作为降级处理
            return TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                parsing_method="service_error_fallback",
                confidence=0.0,
                raw_data=response_data
            )
    
    def get_supported_providers(self) -> list:
        """获取支持的厂商列表
        
        Returns:
            支持的厂商名称列表
        """
        return list(self.provider_parsers.keys())
    
    def add_provider_parser(self, provider: str, parser: BaseTokenParser):
        """添加新的厂商解析器
        
        Args:
            provider: 厂商名称
            parser: Token解析器实例
        """
        self.provider_parsers[provider.lower()] = parser
        self.logger.info("添加新的厂商解析器", provider=provider)