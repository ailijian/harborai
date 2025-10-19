#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的Token解析服务

专门处理各厂商特殊格式的Token解析，提供完善的降级策略和错误恢复机制。
支持OpenAI、DeepSeek、百度千帆、豆包、Claude等主流厂商的特殊响应格式。
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import structlog
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VendorType(Enum):
    """厂商类型枚举"""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    BAIDU = "baidu"
    BYTEDANCE = "bytedance"  # 豆包
    ANTHROPIC = "anthropic"  # Claude
    GOOGLE = "google"  # Gemini
    ZHIPU = "zhipu"  # 智谱AI
    MOONSHOT = "moonshot"  # 月之暗面
    MINIMAX = "minimax"  # MiniMax
    XUNFEI = "xunfei"  # 讯飞星火
    ALIBABA = "alibaba"  # 阿里云通义
    TENCENT = "tencent"  # 腾讯混元
    UNKNOWN = "unknown"


@dataclass
class TokenParsingResult:
    """Token解析结果"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    parsing_method: str = "unknown"
    confidence: float = 0.0
    vendor: VendorType = VendorType.UNKNOWN
    raw_usage_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    fallback_used: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def is_valid(self) -> bool:
        """检查解析结果是否有效"""
        return (
            self.total_tokens > 0 and 
            self.prompt_tokens >= 0 and 
            self.completion_tokens >= 0 and
            self.error_message is None
        )


@dataclass
class VendorTokenMapping:
    """厂商Token字段映射配置"""
    usage_key: str  # 使用量数据的主键
    prompt_key: str  # 输入Token字段名
    completion_key: str  # 输出Token字段名
    total_key: str  # 总Token字段名
    alternative_keys: List[Tuple[str, str, str, str]] = field(default_factory=list)  # 备选字段组合
    special_handlers: List[str] = field(default_factory=list)  # 特殊处理方法名
    nested_paths: List[str] = field(default_factory=list)  # 嵌套路径，如 ["data", "usage"]
    custom_extractors: Dict[str, str] = field(default_factory=dict)  # 自定义提取器


class EnhancedTokenParser:
    """增强的Token解析器
    
    功能特性：
    1. 支持多厂商特殊格式解析
    2. 智能降级策略
    3. 置信度评估
    4. 缓存机制
    5. 详细错误报告
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化增强Token解析器
        
        Args:
            config: 配置字典，包含厂商映射和解析选项
        """
        self.config = config or {}
        self.logger = structlog.get_logger(__name__)
        self.vendor_mappings = {}
        self.special_handlers = {}
        self.parsing_stats = {
            'total_attempts': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'vendor_stats': {},
            'method_stats': {},
            'degradation_stats': {}
        }
        
        # 智能降级策略配置
        self.degradation_config = self.config.get('degradation', {
            'enable_content_estimation': True,
            'enable_fallback_vendors': True,
            'min_confidence_threshold': 0.7,
            'max_retry_attempts': 3
        })
        
        # 动态配置更新
        self._config_version = 0
        self._last_config_update = datetime.now().timestamp()
        
        # 设置厂商映射和特殊处理器
        self._setup_vendor_mappings()
        self._setup_special_handlers()
        self._parsing_cache = {}
        self._stats = {
            'total_parsed': 0,
            'successful_parsed': 0,
            'fallback_used': 0,
            'cache_hits': 0,
            'vendor_stats': {}
        }
    
    def _setup_vendor_mappings(self):
        """设置厂商Token字段映射"""
        self.vendor_mappings = {
            VendorType.OPENAI: VendorTokenMapping(
                usage_key="usage",
                prompt_key="prompt_tokens",
                completion_key="completion_tokens", 
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                ]
            ),
            
            VendorType.DEEPSEEK: VendorTokenMapping(
                usage_key="usage",
                prompt_key="prompt_tokens",
                completion_key="completion_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("usage", "input_tokens", "output_tokens", "total_tokens"),
                ],
                special_handlers=["_handle_deepseek_reasoning_tokens"]
            ),
            
            VendorType.BAIDU: VendorTokenMapping(
                usage_key="usage",
                prompt_key="prompt_tokens",
                completion_key="completion_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("result", "prompt_tokens", "completion_tokens", "total_tokens"),
                ],
                special_handlers=["_handle_baidu_ernie_format"]
            ),
            
            VendorType.BYTEDANCE: VendorTokenMapping(
                usage_key="usage",
                prompt_key="prompt_tokens", 
                completion_key="completion_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("usage", "prompt_tokens", "answer_tokens", "total_tokens"),  # 豆包特殊字段
                ],
                special_handlers=["_handle_doubao_format"]
            ),
            
            VendorType.ANTHROPIC: VendorTokenMapping(
                usage_key="usage",
                prompt_key="input_tokens",
                completion_key="output_tokens", 
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("usage", "prompt_tokens", "completion_tokens", "total_tokens"),
                ],
                special_handlers=["_handle_claude_format"]
            ),
            
            VendorType.GOOGLE: VendorTokenMapping(
                usage_key="usage_metadata",
                prompt_key="prompt_token_count",
                completion_key="candidates_token_count",
                total_key="total_token_count",
                alternative_keys=[
                    ("usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("usage", "prompt_tokens", "completion_tokens", "total_tokens"),
                ],
                special_handlers=["_handle_gemini_format"]
            ),
            
            VendorType.ZHIPU: VendorTokenMapping(
                usage_key="usage",
                prompt_key="prompt_tokens",
                completion_key="completion_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("usage", "input_tokens", "output_tokens", "total_tokens"),
                ],
                special_handlers=["_handle_zhipu_format"]
            ),
            
            VendorType.MOONSHOT: VendorTokenMapping(
                usage_key="usage",
                prompt_key="prompt_tokens",
                completion_key="completion_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                ],
                special_handlers=["_handle_moonshot_format", "_handle_moonshot_v1_format"]
            ),
            
            VendorType.MINIMAX: VendorTokenMapping(
                usage_key="usage",
                prompt_key="input_tokens",
                completion_key="output_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("usage", "prompt_tokens", "completion_tokens", "total_tokens"),
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("data", "usage", "input_tokens", "output_tokens"),
                    ("response", "usage", "input_tokens", "output_tokens"),
                ],
                special_handlers=["_handle_minimax_format", "_handle_minimax_abab_format"]
            ),
            
            VendorType.XUNFEI: VendorTokenMapping(
                usage_key="usage",
                prompt_key="prompt_tokens",
                completion_key="completion_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "question_tokens", "answer_tokens", "total_tokens"),
                    ("usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("payload", "usage", "prompt_tokens", "completion_tokens"),
                    ("data", "usage", "question_tokens", "answer_tokens"),
                ],
                nested_paths=["payload", "usage"],
                special_handlers=["_handle_xunfei_format", "_handle_xunfei_spark_format"]
            ),
            
            VendorType.ALIBABA: VendorTokenMapping(
                usage_key="usage",
                prompt_key="input_tokens",
                completion_key="output_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("usage", "prompt_tokens", "completion_tokens", "total_tokens"),
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("data", "usage", "input_tokens", "output_tokens"),
                    ("response", "usage", "input_tokens", "output_tokens"),
                ],
                special_handlers=["_handle_alibaba_format", "_handle_alibaba_qwen_format"]
            ),
            
            VendorType.TENCENT: VendorTokenMapping(
                usage_key="Usage",
                prompt_key="PromptTokens",
                completion_key="CompletionTokens",
                total_key="TotalTokens",
                alternative_keys=[
                    ("usage", "prompt_tokens", "completion_tokens", "total_tokens"),
                    ("Usage", "InputTokens", "OutputTokens", "TotalTokens"),
                    ("data", "Usage", "PromptTokens", "CompletionTokens"),
                    ("response", "Usage", "PromptTokens", "CompletionTokens"),
                ],
                special_handlers=["_handle_tencent_format", "_handle_tencent_hunyuan_format"]
            ),
            
            # 新增厂商支持
            VendorType.STEPFUN: VendorTokenMapping(
                usage_key="usage",
                prompt_key="prompt_tokens",
                completion_key="completion_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("usage", "input_tokens", "output_tokens", "total_tokens"),
                ],
                special_handlers=["_handle_stepfun_format"]
            ),
            
            VendorType.SENSETIME: VendorTokenMapping(
                usage_key="usage",
                prompt_key="prompt_tokens",
                completion_key="completion_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("usage", "input_tokens", "output_tokens", "total_tokens"),
                ],
                special_handlers=["_handle_sensetime_format"]
            ),
            
            VendorType.BAICHUAN: VendorTokenMapping(
                usage_key="usage",
                prompt_key="prompt_tokens",
                completion_key="completion_tokens",
                total_key="total_tokens",
                alternative_keys=[
                    ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
                    ("usage", "input_tokens", "output_tokens", "total_tokens"),
                ],
                special_handlers=["_handle_baichuan_format"]
            )
        }
    
    def _setup_special_handlers(self):
        """设置特殊处理器"""
        self.special_handlers = {
            "_handle_deepseek_reasoning_tokens": self._handle_deepseek_reasoning_tokens,
            "_handle_deepseek_v3_format": self._handle_deepseek_v3_format,
            "_handle_baidu_ernie_format": self._handle_baidu_ernie_format,
            "_handle_baidu_qianfan_format": self._handle_baidu_qianfan_format,
            "_handle_doubao_format": self._handle_doubao_format,
            "_handle_doubao_pro_format": self._handle_doubao_pro_format,
            "_handle_claude_format": self._handle_claude_format,
            "_handle_claude_3_format": self._handle_claude_3_format,
            "_handle_gemini_format": self._handle_gemini_format,
            "_handle_gemini_pro_format": self._handle_gemini_pro_format,
            "_handle_zhipu_format": self._handle_zhipu_format,
            "_handle_zhipu_glm_format": self._handle_zhipu_glm_format,
            "_handle_moonshot_format": self._handle_moonshot_format,
            "_handle_moonshot_v1_format": self._handle_moonshot_v1_format,
            "_handle_minimax_format": self._handle_minimax_format,
            "_handle_minimax_abab_format": self._handle_minimax_abab_format,
            "_handle_xunfei_format": self._handle_xunfei_format,
            "_handle_xunfei_spark_format": self._handle_xunfei_spark_format,
            "_handle_alibaba_format": self._handle_alibaba_format,
            "_handle_alibaba_qwen_format": self._handle_alibaba_qwen_format,
            "_handle_tencent_format": self._handle_tencent_format,
            "_handle_tencent_hunyuan_format": self._handle_tencent_hunyuan_format,
            "_handle_stepfun_format": self._handle_stepfun_format,
            "_handle_sensetime_format": self._handle_sensetime_format,
            "_handle_baichuan_format": self._handle_baichuan_format,
        }
    
    def parse_token_usage(self, 
                         response_data: Dict[str, Any], 
                         vendor: Optional[Union[str, VendorType]] = None,
                         model: Optional[str] = None,
                         request_data: Optional[Dict[str, Any]] = None) -> TokenParsingResult:
        """解析Token使用量
        
        Args:
            response_data: API响应数据
            vendor: 厂商类型
            model: 模型名称
            request_data: 请求数据（用于估算）
            
        Returns:
            Token解析结果
        """
        self.parsing_stats['total_attempts'] += 1
        
        # 确定厂商类型
        if isinstance(vendor, str):
            vendor = self._detect_vendor_from_string(vendor)
        elif vendor is None:
            vendor = self._detect_vendor_from_response(response_data, model)
        
        # 更新厂商统计
        vendor_name = vendor.value
        if vendor_name not in self.parsing_stats['vendor_stats']:
            self.parsing_stats['vendor_stats'][vendor_name] = {'attempts': 0, 'successes': 0}
        self.parsing_stats['vendor_stats'][vendor_name]['attempts'] += 1
        
        try:
            # 尝试标准解析
            result = self._parse_with_vendor_mapping(response_data, vendor)
            if result.is_valid() and result.confidence >= 0.8:
                self.parsing_stats['successful_parses'] += 1
                self.parsing_stats['vendor_stats'][vendor_name]['successes'] += 1
                self._update_method_stats(result.parsing_method, True)
                return result
            
            # 尝试特殊处理器
            if vendor in self.vendor_mappings:
                mapping = self.vendor_mappings[vendor]
                for handler_name in mapping.special_handlers:
                    if handler_name in self.special_handlers:
                        handler_result = self.special_handlers[handler_name](response_data)
                        if handler_result.is_valid() and handler_result.confidence >= 0.7:
                            self.parsing_stats['successful_parses'] += 1
                            self.parsing_stats['vendor_stats'][vendor_name]['successes'] += 1
                            self._update_method_stats(handler_result.parsing_method, True)
                            return handler_result
            
            # 智能降级解析
            content = self._extract_content_for_estimation(request_data)
            degradation_result = self._intelligent_degradation_parse(response_data, vendor, content)
            if degradation_result.is_valid():
                self.parsing_stats['successful_parses'] += 1
                self.parsing_stats['vendor_stats'][vendor_name]['successes'] += 1
                self._update_method_stats(degradation_result.parsing_method, True)
                return degradation_result
            
            # 解析失败
            self.parsing_stats['failed_parses'] += 1
            self._update_method_stats("parsing_failed", False)
            
            return TokenParsingResult(
                parsing_method="all_methods_failed",
                confidence=0.0,
                metadata={'vendor': vendor_name, 'model': model}
            )
            
        except Exception as e:
            self.logger.error("Token解析异常", error=str(e), vendor=vendor_name, model=model)
            self.parsing_stats['failed_parses'] += 1
            self._update_method_stats("parsing_exception", False)
            return TokenParsingResult(
                error_message=str(e),
                vendor=vendor,
                parsing_method="error"
            )
    
    def _parse_standard_format(self, response_data: Dict[str, Any], vendor: VendorType) -> TokenParsingResult:
        """标准格式解析"""
        mapping = self.vendor_mappings.get(vendor)
        if not mapping:
            return TokenParsingResult(vendor=vendor, parsing_method="no_mapping")
        
        # 尝试主要字段映射
        result = self._extract_tokens_with_mapping(response_data, mapping, vendor)
        if result.confidence > 0:
            return result
        
        # 尝试备选字段映射
        for alt_usage_key, alt_prompt_key, alt_completion_key, alt_total_key in mapping.alternative_keys:
            alt_mapping = VendorTokenMapping(
                usage_key=alt_usage_key,
                prompt_key=alt_prompt_key,
                completion_key=alt_completion_key,
                total_key=alt_total_key
            )
            result = self._extract_tokens_with_mapping(response_data, alt_mapping, vendor)
            if result.confidence > 0:
                result.parsing_method = f"alternative_{alt_usage_key}"
                return result
        
        return TokenParsingResult(vendor=vendor, parsing_method="standard_failed")
    
    def _extract_tokens_with_mapping(self, 
                                   response_data: Dict[str, Any], 
                                   mapping: VendorTokenMapping, 
                                   vendor: VendorType) -> TokenParsingResult:
        """使用指定映射提取Token信息"""
        try:
            if mapping.usage_key not in response_data:
                return TokenParsingResult(vendor=vendor)
            
            usage_data = response_data[mapping.usage_key]
            if not isinstance(usage_data, dict):
                return TokenParsingResult(vendor=vendor)
            
            prompt_tokens = usage_data.get(mapping.prompt_key, 0)
            completion_tokens = usage_data.get(mapping.completion_key, 0)
            total_tokens = usage_data.get(mapping.total_key, prompt_tokens + completion_tokens)
            
            # 验证数据合理性
            if not self._validate_token_data(prompt_tokens, completion_tokens, total_tokens):
                return TokenParsingResult(vendor=vendor, parsing_method="validation_failed")
            
            return TokenParsingResult(
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
                total_tokens=int(total_tokens),
                parsing_method="standard",
                confidence=0.9,
                vendor=vendor,
                raw_usage_data=usage_data
            )
            
        except (KeyError, TypeError, ValueError) as e:
            return TokenParsingResult(
                vendor=vendor,
                error_message=str(e),
                parsing_method="extraction_error"
            )
    
    def _parse_with_special_handlers(self, response_data: Dict[str, Any], vendor: VendorType) -> TokenParsingResult:
        """使用特殊处理器解析"""
        mapping = self.vendor_mappings.get(vendor)
        if not mapping or not mapping.special_handlers:
            return TokenParsingResult(vendor=vendor, parsing_method="no_special_handlers")
        
        for handler_name in mapping.special_handlers:
            handler = self.special_handlers.get(handler_name)
            if handler:
                try:
                    result = handler(response_data)
                    if result.confidence > 0:
                        result.vendor = vendor
                        result.parsing_method = f"special_{handler_name}"
                        return result
                except Exception as e:
                    self.logger.warning(f"特殊处理器{handler_name}失败", error=str(e))
        
        return TokenParsingResult(vendor=vendor, parsing_method="special_handlers_failed")
    
    def _fallback_parsing(self, 
                         response_data: Dict[str, Any], 
                         vendor: VendorType,
                         request_data: Optional[Dict[str, Any]] = None) -> TokenParsingResult:
        """降级解析策略"""
        # 策略1：尝试所有可能的字段组合
        common_field_combinations = [
            ("usage", "prompt_tokens", "completion_tokens", "total_tokens"),
            ("token_usage", "input_tokens", "output_tokens", "total_tokens"),
            ("usage", "input_tokens", "output_tokens", "total_tokens"),
            ("result", "prompt_tokens", "completion_tokens", "total_tokens"),
            ("data", "prompt_tokens", "completion_tokens", "total_tokens"),
        ]
        
        for usage_key, prompt_key, completion_key, total_key in common_field_combinations:
            try:
                if usage_key in response_data:
                    usage = response_data[usage_key]
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get(prompt_key, 0)
                        completion_tokens = usage.get(completion_key, 0)
                        total_tokens = usage.get(total_key, prompt_tokens + completion_tokens)
                        
                        if self._validate_token_data(prompt_tokens, completion_tokens, total_tokens):
                            return TokenParsingResult(
                                prompt_tokens=int(prompt_tokens),
                                completion_tokens=int(completion_tokens),
                                total_tokens=int(total_tokens),
                                parsing_method="fallback_extraction",
                                confidence=0.7,
                                vendor=vendor,
                                raw_usage_data=usage
                            )
            except (TypeError, ValueError):
                continue
        
        # 策略2：基于文本内容估算
        if request_data:
            return self._estimate_tokens_from_content(response_data, request_data, vendor)
        
        # 策略3：返回默认值
        return TokenParsingResult(
            prompt_tokens=1,
            completion_tokens=1,
            total_tokens=2,
            parsing_method="default_fallback",
            confidence=0.1,
            vendor=vendor
        )
    
    def _estimate_tokens_from_content(self, 
                                    response_data: Dict[str, Any], 
                                    request_data: Dict[str, Any],
                                    vendor: VendorType) -> TokenParsingResult:
        """基于内容估算Token数量"""
        try:
            # 提取请求文本
            prompt_text = ""
            if 'messages' in request_data:
                for msg in request_data['messages']:
                    if isinstance(msg, dict) and 'content' in msg:
                        prompt_text += str(msg['content']) + " "
            elif 'prompt' in request_data:
                prompt_text = str(request_data['prompt'])
            
            # 提取响应文本
            completion_text = ""
            if 'choices' in response_data:
                for choice in response_data['choices']:
                    if isinstance(choice, dict):
                        if 'message' in choice and 'content' in choice['message']:
                            completion_text += str(choice['message']['content']) + " "
                        elif 'text' in choice:
                            completion_text += str(choice['text']) + " "
            elif 'content' in response_data:
                completion_text = str(response_data['content'])
            elif 'text' in response_data:
                completion_text = str(response_data['text'])
            
            # 估算Token数量（根据厂商调整估算比例）
            estimation_ratio = self._get_vendor_estimation_ratio(vendor)
            prompt_tokens = max(1, int(len(prompt_text) * estimation_ratio))
            completion_tokens = max(1, int(len(completion_text) * estimation_ratio))
            total_tokens = prompt_tokens + completion_tokens
            
            return TokenParsingResult(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                parsing_method="content_estimation",
                confidence=0.5,
                vendor=vendor
            )
            
        except Exception as e:
            self.logger.warning("内容估算失败", error=str(e))
            return TokenParsingResult(
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                parsing_method="estimation_failed",
                confidence=0.1,
                vendor=vendor,
                error_message=str(e)
            )
    
    # ==================== 特殊处理器实现 ====================
    
    def _handle_deepseek_reasoning_tokens(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理DeepSeek推理模型的特殊Token格式"""
        try:
            # DeepSeek推理模型可能有reasoning_tokens字段
            if 'usage' in response_data:
                usage = response_data['usage']
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    reasoning_tokens = usage.get('reasoning_tokens', 0)  # 推理Token
                    total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens + reasoning_tokens)
                    
                    return TokenParsingResult(
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens + reasoning_tokens),  # 将推理Token计入输出
                        total_tokens=int(total_tokens),
                        parsing_method="deepseek_reasoning",
                        confidence=0.9,
                        raw_usage_data=usage
                    )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="deepseek_reasoning_failed")
    
    def _handle_baidu_ernie_format(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理百度ERNIE的特殊格式"""
        try:
            # 百度可能使用result字段
            if 'result' in response_data and 'usage' in response_data['result']:
                usage = response_data['result']['usage']
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
                    
                    return TokenParsingResult(
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                        total_tokens=int(total_tokens),
                        parsing_method="baidu_ernie",
                        confidence=0.9,
                        raw_usage_data=usage
                    )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="baidu_ernie_failed")
    
    def _handle_doubao_format(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理豆包的特殊格式"""
        try:
            # 豆包可能使用answer_tokens而不是completion_tokens
            if 'usage' in response_data:
                usage = response_data['usage']
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    answer_tokens = usage.get('answer_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', answer_tokens)
                    total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
                    
                    return TokenParsingResult(
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                        total_tokens=int(total_tokens),
                        parsing_method="doubao_format",
                        confidence=0.9,
                        raw_usage_data=usage
                    )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="doubao_format_failed")
    
    def _handle_claude_format(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理Claude的特殊格式"""
        try:
            # Claude使用input_tokens和output_tokens
            if 'usage' in response_data:
                usage = response_data['usage']
                if isinstance(usage, dict):
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    total_tokens = input_tokens + output_tokens
                    
                    return TokenParsingResult(
                        prompt_tokens=int(input_tokens),
                        completion_tokens=int(output_tokens),
                        total_tokens=int(total_tokens),
                        parsing_method="claude_format",
                        confidence=0.9,
                        raw_usage_data=usage
                    )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="claude_format_failed")
    
    def _handle_gemini_format(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理Gemini的特殊格式"""
        try:
            # Gemini使用usage_metadata
            if 'usage_metadata' in response_data:
                usage = response_data['usage_metadata']
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_token_count', 0)
                    completion_tokens = usage.get('candidates_token_count', 0)
                    total_tokens = usage.get('total_token_count', prompt_tokens + completion_tokens)
                    
                    return TokenParsingResult(
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                        total_tokens=int(total_tokens),
                        parsing_method="gemini_format",
                        confidence=0.9,
                        raw_usage_data=usage
                    )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="gemini_format_failed")
    
    def _handle_zhipu_format(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理智谱AI的特殊格式"""
        try:
            # 智谱AI标准格式
            if 'usage' in response_data:
                usage = response_data['usage']
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
                    
                    return TokenParsingResult(
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                        total_tokens=int(total_tokens),
                        parsing_method="zhipu_format",
                        confidence=0.9,
                        raw_usage_data=usage
                    )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="zhipu_format_failed")
    
    def _handle_moonshot_format(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理月之暗面的特殊格式"""
        try:
            # 月之暗面标准格式
            if 'usage' in response_data:
                usage = response_data['usage']
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('prompt_tokens', 0)
                    completion_tokens = usage.get('completion_tokens', 0)
                    total_tokens = usage.get('total_tokens', prompt_tokens + completion_tokens)
                    
                    return TokenParsingResult(
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                        total_tokens=int(total_tokens),
                        parsing_method="moonshot_format",
                        confidence=0.9,
                        raw_usage_data=usage
                    )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="moonshot_format_failed")
    
    def _handle_minimax_format(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理MiniMax的特殊格式"""
        try:
            # MiniMax使用input_tokens和output_tokens
            if 'usage' in response_data:
                usage = response_data['usage']
                if isinstance(usage, dict):
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
                    
                    return TokenParsingResult(
                        prompt_tokens=int(input_tokens),
                        completion_tokens=int(output_tokens),
                        total_tokens=int(total_tokens),
                        parsing_method="minimax_format",
                        confidence=0.9,
                        raw_usage_data=usage
                    )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="minimax_format_failed")
    
    def _handle_xunfei_format(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理讯飞星火的特殊格式"""
        try:
            # 讯飞可能使用嵌套的payload.usage结构
            usage_data = response_data
            
            # 尝试嵌套路径
            if 'payload' in response_data and 'usage' in response_data['payload']:
                usage_data = response_data['payload']['usage']
            elif 'usage' in response_data:
                usage_data = response_data['usage']
            
            if isinstance(usage_data, dict):
                # 讯飞可能使用question_tokens和answer_tokens
                question_tokens = usage_data.get('question_tokens', 0)
                answer_tokens = usage_data.get('answer_tokens', 0)
                prompt_tokens = usage_data.get('prompt_tokens', question_tokens)
                completion_tokens = usage_data.get('completion_tokens', answer_tokens)
                total_tokens = usage_data.get('total_tokens', prompt_tokens + completion_tokens)
                
                return TokenParsingResult(
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    total_tokens=int(total_tokens),
                    parsing_method="xunfei_format",
                    confidence=0.9,
                    raw_usage_data=usage_data
                )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="xunfei_format_failed")
    
    def _handle_alibaba_format(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理阿里云通义的特殊格式"""
        try:
            # 阿里云通义使用input_tokens和output_tokens
            if 'usage' in response_data:
                usage = response_data['usage']
                if isinstance(usage, dict):
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)
                    total_tokens = usage.get('total_tokens', input_tokens + output_tokens)
                    
                    return TokenParsingResult(
                        prompt_tokens=int(input_tokens),
                        completion_tokens=int(output_tokens),
                        total_tokens=int(total_tokens),
                        parsing_method="alibaba_format",
                        confidence=0.9,
                        raw_usage_data=usage
                    )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="alibaba_format_failed")
    
    def _handle_tencent_format(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """处理腾讯混元的特殊格式"""
        try:
            # 腾讯混元使用大写字段名
            if 'Usage' in response_data:
                usage = response_data['Usage']
                if isinstance(usage, dict):
                    prompt_tokens = usage.get('PromptTokens', 0)
                    completion_tokens = usage.get('CompletionTokens', 0)
                    total_tokens = usage.get('TotalTokens', prompt_tokens + completion_tokens)
                    
                    return TokenParsingResult(
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                        total_tokens=int(total_tokens),
                        parsing_method="tencent_format",
                        confidence=0.9,
                        raw_usage_data=usage
                    )
        except (KeyError, TypeError, ValueError):
            pass
        
        return TokenParsingResult(parsing_method="tencent_format_failed")
    
    # ==================== 智能降级策略 ====================
    
    def _intelligent_degradation_parse(self, response_data: Dict[str, Any], 
                                     vendor: VendorType, 
                                     content: Optional[str] = None) -> TokenParsingResult:
        """
        智能降级解析策略
        
        Args:
            response_data: 响应数据
            vendor: 厂商类型
            content: 内容文本（用于估算）
            
        Returns:
            TokenParsingResult: 解析结果
        """
        degradation_attempts = []
        
        # 策略1: 尝试相似厂商的解析器
        if self.degradation_config.get('enable_fallback_vendors', True):
            similar_vendors = self._get_similar_vendors(vendor)
            for similar_vendor in similar_vendors:
                try:
                    result = self._parse_with_vendor_mapping(response_data, similar_vendor)
                    if result.is_valid() and result.confidence >= self.degradation_config.get('min_confidence_threshold', 0.7):
                        result.parsing_method = f"degraded_to_{similar_vendor.value}"
                        degradation_attempts.append(('similar_vendor', similar_vendor.value, result))
                        self.parsing_stats['degradation_stats']['similar_vendor_success'] = \
                            self.parsing_stats['degradation_stats'].get('similar_vendor_success', 0) + 1
                        return result
                except Exception as e:
                    self.logger.warning(f"相似厂商{similar_vendor.value}解析失败", error=str(e))
        
        # 策略2: 通用字段搜索
        generic_result = self._generic_field_search(response_data)
        if generic_result.is_valid():
            generic_result.parsing_method = "degraded_generic_search"
            degradation_attempts.append(('generic_search', 'generic', generic_result))
            self.parsing_stats['degradation_stats']['generic_search_success'] = \
                self.parsing_stats['degradation_stats'].get('generic_search_success', 0) + 1
            return generic_result
        
        # 策略3: 内容估算
        if content and self.degradation_config.get('enable_content_estimation', True):
            estimated_result = self._estimate_tokens_from_content(content, vendor)
            if estimated_result.is_valid():
                estimated_result.parsing_method = "degraded_content_estimation"
                degradation_attempts.append(('content_estimation', 'estimated', estimated_result))
                self.parsing_stats['degradation_stats']['content_estimation_success'] = \
                    self.parsing_stats['degradation_stats'].get('content_estimation_success', 0) + 1
                return estimated_result
        
        # 记录降级失败
        self.parsing_stats['degradation_stats']['total_failures'] = \
            self.parsing_stats['degradation_stats'].get('total_failures', 0) + 1
        
        return TokenParsingResult(
            parsing_method="degradation_failed",
            confidence=0.0,
            metadata={'degradation_attempts': degradation_attempts}
        )
    
    def _get_similar_vendors(self, vendor: VendorType) -> List[VendorType]:
        """获取相似厂商列表"""
        # 基于API格式相似性的厂商分组
        vendor_groups = {
            'openai_like': [VendorType.OPENAI, VendorType.DEEPSEEK, VendorType.ZHIPU, VendorType.MOONSHOT],
            'chinese_vendors': [VendorType.BAIDU, VendorType.ALIBABA, VendorType.TENCENT, VendorType.XUNFEI],
            'international': [VendorType.ANTHROPIC, VendorType.GOOGLE, VendorType.OPENAI],
            'bytedance_like': [VendorType.BYTEDANCE, VendorType.MINIMAX]
        }
        
        # 找到当前厂商所在的组
        for group_vendors in vendor_groups.values():
            if vendor in group_vendors:
                # 返回同组的其他厂商
                return [v for v in group_vendors if v != vendor]
        
        # 如果没找到组，返回OpenAI兼容的厂商
        return [VendorType.OPENAI, VendorType.DEEPSEEK]
    
    def _generic_field_search(self, response_data: Dict[str, Any]) -> TokenParsingResult:
        """通用字段搜索策略"""
        try:
            # 常见的token字段名
            token_fields = [
                'tokens', 'token_count', 'token_usage', 'usage',
                'prompt_tokens', 'completion_tokens', 'total_tokens',
                'input_tokens', 'output_tokens',
                'question_tokens', 'answer_tokens'
            ]
            
            found_tokens = {}
            
            def search_recursive(data, path=""):
                """递归搜索token字段"""
                if isinstance(data, dict):
                    for key, value in data.items():
                        current_path = f"{path}.{key}" if path else key
                        
                        # 检查是否是token相关字段
                        if any(field in key.lower() for field in token_fields):
                            if isinstance(value, (int, float)):
                                found_tokens[current_path] = int(value)
                        
                        # 递归搜索
                        if isinstance(value, (dict, list)):
                            search_recursive(value, current_path)
                elif isinstance(data, list):
                    for i, item in enumerate(data):
                        search_recursive(item, f"{path}[{i}]")
            
            search_recursive(response_data)
            
            if found_tokens:
                # 尝试匹配标准字段
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                
                for path, value in found_tokens.items():
                    path_lower = path.lower()
                    if 'prompt' in path_lower or 'input' in path_lower or 'question' in path_lower:
                        prompt_tokens = max(prompt_tokens, value)
                    elif 'completion' in path_lower or 'output' in path_lower or 'answer' in path_lower:
                        completion_tokens = max(completion_tokens, value)
                    elif 'total' in path_lower:
                        total_tokens = max(total_tokens, value)
                
                # 如果没有total_tokens，计算它
                if total_tokens == 0:
                    total_tokens = prompt_tokens + completion_tokens
                
                if total_tokens > 0:
                    return TokenParsingResult(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        parsing_method="generic_field_search",
                        confidence=0.6,
                        metadata={'found_fields': found_tokens}
                    )
        
        except Exception as e:
            self.logger.warning("通用字段搜索失败", error=str(e))
        
        return TokenParsingResult(parsing_method="generic_search_failed")
    
    # ==================== 动态配置更新 ====================
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        动态更新配置
        
        Args:
            new_config: 新的配置字典
            
        Returns:
            bool: 更新是否成功
        """
        try:
            # 备份当前配置
            old_config = self.config.copy()
            old_version = self._config_version
            
            # 更新配置
            self.config.update(new_config)
            self._config_version += 1
            self._last_config_update = datetime.now().timestamp()
            
            # 重新设置映射和处理器
            if 'vendor_mappings' in new_config or 'special_handlers' in new_config:
                self._setup_vendor_mappings()
                self._setup_special_handlers()
            
            # 更新降级配置
            if 'degradation' in new_config:
                self.degradation_config.update(new_config['degradation'])
            
            self.logger.info(
                "配置更新成功",
                old_version=old_version,
                new_version=self._config_version,
                updated_keys=list(new_config.keys())
            )
            
            return True
            
        except Exception as e:
            # 回滚配置
            self.config = old_config
            self.logger.error("配置更新失败，已回滚", error=str(e))
            return False
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取当前配置信息"""
        return {
            'config_version': self._config_version,
            'last_update': self._last_config_update,
            'vendor_count': len(self.vendor_mappings),
            'handler_count': len(self.special_handlers),
            'degradation_config': self.degradation_config.copy()
        }
    
    def _parse_with_vendor_mapping(self, response_data: Dict[str, Any], vendor: VendorType) -> TokenParsingResult:
        """使用厂商映射解析Token"""
        if vendor not in self.vendor_mappings:
            return TokenParsingResult(parsing_method="vendor_not_supported")
        
        mapping = self.vendor_mappings[vendor]
        
        try:
            # 尝试主要路径
            usage_data = response_data.get(mapping.usage_key, {})
            if isinstance(usage_data, dict):
                prompt_tokens = usage_data.get(mapping.prompt_key, 0)
                completion_tokens = usage_data.get(mapping.completion_key, 0)
                total_tokens = usage_data.get(mapping.total_key, prompt_tokens + completion_tokens)
                
                if total_tokens > 0:
                    return TokenParsingResult(
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                        total_tokens=int(total_tokens),
                        parsing_method=f"vendor_mapping_{vendor.value}",
                        confidence=0.9,
                        raw_usage_data=usage_data
                    )
            
            # 尝试替代路径
            for alt_path in mapping.alternative_keys:
                try:
                    data = response_data
                    for key in alt_path[:-3]:  # 导航到usage数据
                        data = data.get(key, {})
                    
                    if isinstance(data, dict):
                        prompt_tokens = data.get(alt_path[-3], 0)
                        completion_tokens = data.get(alt_path[-2], 0)
                        total_tokens = data.get(alt_path[-1], prompt_tokens + completion_tokens)
                        
                        if total_tokens > 0:
                            return TokenParsingResult(
                                prompt_tokens=int(prompt_tokens),
                                completion_tokens=int(completion_tokens),
                                total_tokens=int(total_tokens),
                                parsing_method=f"vendor_mapping_alt_{vendor.value}",
                                confidence=0.8,
                                raw_usage_data=data
                            )
                except (KeyError, TypeError):
                    continue
            
            # 尝试嵌套路径（如果配置了）
            if hasattr(mapping, 'nested_paths') and mapping.nested_paths:
                for nested_path in mapping.nested_paths:
                    try:
                        data = response_data
                        for key in nested_path.split('.'):
                            data = data.get(key, {})
                        
                        if isinstance(data, dict) and any(k in data for k in [mapping.prompt_key, mapping.completion_key, mapping.total_key]):
                            prompt_tokens = data.get(mapping.prompt_key, 0)
                            completion_tokens = data.get(mapping.completion_key, 0)
                            total_tokens = data.get(mapping.total_key, prompt_tokens + completion_tokens)
                            
                            if total_tokens > 0:
                                return TokenParsingResult(
                                    prompt_tokens=int(prompt_tokens),
                                    completion_tokens=int(completion_tokens),
                                    total_tokens=int(total_tokens),
                                    parsing_method=f"vendor_mapping_nested_{vendor.value}",
                                    confidence=0.8,
                                    raw_usage_data=data
                                )
                    except (KeyError, TypeError):
                        continue
        
        except Exception as e:
            self.logger.warning(f"厂商映射解析失败: {vendor.value}", error=str(e))
        
        return TokenParsingResult(parsing_method=f"vendor_mapping_failed_{vendor.value}")
    
    def _extract_content_for_estimation(self, request_data: Optional[Dict[str, Any]]) -> Optional[str]:
        """从请求数据中提取内容用于估算"""
        if not request_data:
            return None
        
        # 常见的内容字段
        content_fields = ['messages', 'prompt', 'input', 'text', 'content']
        
        for field in content_fields:
            if field in request_data:
                content = request_data[field]
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # 处理messages格式
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and 'content' in item:
                            text_parts.append(str(item['content']))
                        elif isinstance(item, str):
                            text_parts.append(item)
                    return ' '.join(text_parts)
        
        return None
    
    def _update_method_stats(self, method: str, success: bool):
        """更新解析方法统计"""
        if method not in self.parsing_stats['method_stats']:
            self.parsing_stats['method_stats'][method] = {'attempts': 0, 'successes': 0}
        
        self.parsing_stats['method_stats'][method]['attempts'] += 1
        if success:
            self.parsing_stats['method_stats'][method]['successes'] += 1
    
    def _detect_vendor_from_string(self, vendor_str: str) -> VendorType:
        """从字符串检测厂商类型"""
        vendor_str = vendor_str.lower()
        
        vendor_mapping = {
            'openai': VendorType.OPENAI,
            'gpt': VendorType.OPENAI,
            'deepseek': VendorType.DEEPSEEK,
            'baidu': VendorType.BAIDU,
            'ernie': VendorType.BAIDU,
            'bytedance': VendorType.BYTEDANCE,
            'doubao': VendorType.BYTEDANCE,
            'anthropic': VendorType.ANTHROPIC,
            'claude': VendorType.ANTHROPIC,
            'google': VendorType.GOOGLE,
            'gemini': VendorType.GOOGLE,
            'zhipu': VendorType.ZHIPU,
            'glm': VendorType.ZHIPU,
            'moonshot': VendorType.MOONSHOT,
            'kimi': VendorType.MOONSHOT,
            'minimax': VendorType.MINIMAX,
            'xunfei': VendorType.XUNFEI,
            'spark': VendorType.XUNFEI,
            'alibaba': VendorType.ALIBABA,
            'qwen': VendorType.ALIBABA,
            'tencent': VendorType.TENCENT,
            'hunyuan': VendorType.TENCENT
        }
        
        for key, vendor_type in vendor_mapping.items():
            if key in vendor_str:
                return vendor_type
        
        return VendorType.UNKNOWN  # 默认
    
    def _detect_vendor_from_response(self, response_data: Dict[str, Any], model: Optional[str] = None) -> VendorType:
        """从响应数据检测厂商类型"""
        # 基于模型名称检测
        if model:
            model_lower = model.lower()
            if 'gpt' in model_lower or 'openai' in model_lower:
                return VendorType.OPENAI
            elif 'deepseek' in model_lower:
                return VendorType.DEEPSEEK
            elif 'ernie' in model_lower or 'baidu' in model_lower:
                return VendorType.BAIDU
            elif 'doubao' in model_lower or 'bytedance' in model_lower:
                return VendorType.BYTEDANCE
            elif 'claude' in model_lower or 'anthropic' in model_lower:
                return VendorType.ANTHROPIC
            elif 'gemini' in model_lower or 'google' in model_lower:
                return VendorType.GOOGLE
            elif 'glm' in model_lower or 'zhipu' in model_lower:
                return VendorType.ZHIPU
            elif 'moonshot' in model_lower or 'kimi' in model_lower:
                return VendorType.MOONSHOT
            elif 'minimax' in model_lower:
                return VendorType.MINIMAX
            elif 'spark' in model_lower or 'xunfei' in model_lower:
                return VendorType.XUNFEI
            elif 'qwen' in model_lower or 'alibaba' in model_lower:
                return VendorType.ALIBABA
            elif 'hunyuan' in model_lower or 'tencent' in model_lower:
                return VendorType.TENCENT
        
        # 基于响应结构检测
        if 'Usage' in response_data and 'PromptTokens' in response_data.get('Usage', {}):
            return VendorType.TENCENT
        elif 'usage' in response_data:
            usage = response_data['usage']
            if isinstance(usage, dict):
                if 'reasoning_tokens' in usage:
                    return VendorType.DEEPSEEK
                elif 'input_tokens' in usage and 'output_tokens' in usage:
                    return VendorType.MINIMAX
        
        return VendorType.UNKNOWN  # 默认
    
    # ==================== 辅助方法 ====================
    
    def _validate_token_data(self, prompt_tokens: Any, completion_tokens: Any, total_tokens: Any) -> bool:
        """验证Token数据的合理性"""
        try:
            p_tokens = int(prompt_tokens)
            c_tokens = int(completion_tokens)
            t_tokens = int(total_tokens)
            
            # 基本合理性检查
            if p_tokens < 0 or c_tokens < 0 or t_tokens < 0:
                return False
            
            # 总数检查（允许一定误差）
            expected_total = p_tokens + c_tokens
            if abs(t_tokens - expected_total) > max(10, expected_total * 0.1):
                return False
            
            # 合理范围检查（避免异常大的值）
            if p_tokens > 1000000 or c_tokens > 1000000 or t_tokens > 2000000:
                return False
            
            return True
            
        except (TypeError, ValueError):
            return False
    
    def _get_vendor_estimation_ratio(self, vendor: VendorType) -> float:
        """获取厂商特定的Token估算比例"""
        ratios = {
            VendorType.OPENAI: 0.25,      # 1 token ≈ 4 chars
            VendorType.DEEPSEEK: 0.25,    # 类似OpenAI
            VendorType.BAIDU: 0.5,        # 中文Token较大
            VendorType.BYTEDANCE: 0.4,    # 中文优化
            VendorType.ANTHROPIC: 0.25,   # 类似OpenAI
            VendorType.GOOGLE: 0.3,       # 稍大一些
            VendorType.UNKNOWN: 0.25      # 默认值
        }
        return ratios.get(vendor, 0.25)
    
    def _generate_cache_key(self, response_data: Dict[str, Any], vendor: Optional[VendorType], model: Optional[str]) -> str:
        """生成缓存键"""
        try:
            # 使用响应数据的关键部分生成哈希
            key_data = {
                'vendor': vendor.value if vendor else 'unknown',
                'model': model or 'unknown',
                'usage_keys': list(response_data.keys()) if isinstance(response_data, dict) else []
            }
            return f"token_parse_{hash(str(key_data))}"
        except:
            return f"token_parse_{hash(str(response_data))}"
    
    def get_parsing_statistics(self) -> Dict[str, Any]:
        """获取解析统计信息"""
        stats = self._stats.copy()
        if stats['total_parsed'] > 0:
            stats['success_rate'] = stats['successful_parsed'] / stats['total_parsed']
            stats['fallback_rate'] = stats['fallback_used'] / stats['total_parsed']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_parsed']
        else:
            stats['success_rate'] = 0.0
            stats['fallback_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def clear_cache(self):
        """清空解析缓存"""
        self._parsing_cache.clear()
        self.logger.info("Token解析缓存已清空")
    
    def reset_statistics(self):
        """重置统计信息"""
        self._stats = {
            'total_parsed': 0,
            'successful_parsed': 0,
            'fallback_used': 0,
            'cache_hits': 0,
            'vendor_stats': {}
        }
        self.logger.info("Token解析统计信息已重置")


# 全局Token解析器实例
global_token_parser = EnhancedTokenParser()


def parse_token_usage(response_data: Dict[str, Any], 
                     vendor: Optional[str] = None,
                     model: Optional[str] = None,
                     request_data: Optional[Dict[str, Any]] = None) -> TokenParsingResult:
    """解析Token使用量的便捷函数"""
    return global_token_parser.parse_token_usage(response_data, vendor, model, request_data)


def get_token_parsing_stats() -> Dict[str, Any]:
    """获取Token解析统计信息的便捷函数"""
    return global_token_parser.get_parsing_statistics()