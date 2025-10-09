#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 模型定义模块测试

测试模型类型、能力、推理模型等功能。
"""

import pytest
from unittest.mock import patch, MagicMock

from harborai.core.models import (
    ModelType, ModelCapabilities, ReasoningModel,
    MODEL_CAPABILITIES_CONFIG, REASONING_MODEL_PATTERNS,
    is_reasoning_model, get_model_capabilities, create_reasoning_model,
    get_supported_models, get_reasoning_models, filter_parameters_for_model
)


class TestModelType:
    """测试模型类型枚举"""
    
    def test_model_type_values(self):
        """测试模型类型枚举值"""
        assert ModelType.REASONING.value == "reasoning"
        assert ModelType.CHAT.value == "chat"
        assert ModelType.COMPLETION.value == "completion"
        assert ModelType.EMBEDDING.value == "embedding"
    
    def test_model_type_members(self):
        """测试模型类型成员"""
        expected_types = {"REASONING", "CHAT", "COMPLETION", "EMBEDDING"}
        actual_types = {member.name for member in ModelType}
        assert actual_types == expected_types


class TestModelCapabilities:
    """测试模型能力类"""
    
    def test_default_initialization(self):
        """测试默认初始化"""
        capabilities = ModelCapabilities()
        
        assert capabilities.supports_reasoning is False
        assert capabilities.supports_streaming is True
        assert capabilities.supports_temperature is True
        assert capabilities.supports_system_message is True
        assert capabilities.supports_function_calling is False
        assert capabilities.supports_structured_output is False
        assert capabilities.max_tokens_limit == 4096
        assert capabilities.max_context_length == 4096
        assert capabilities.supported_parameters == []
        assert capabilities.unsupported_parameters == []
    
    def test_custom_initialization(self):
        """测试自定义初始化"""
        capabilities = ModelCapabilities(
            supports_reasoning=True,
            supports_streaming=False,
            supports_temperature=False,
            supports_system_message=False,
            supports_function_calling=True,
            supports_structured_output=True,
            max_tokens_limit=8192,
            max_context_length=16384,
            supported_parameters=["messages", "model"],
            unsupported_parameters=["temperature", "top_p"]
        )
        
        assert capabilities.supports_reasoning is True
        assert capabilities.supports_streaming is False
        assert capabilities.supports_temperature is False
        assert capabilities.supports_system_message is False
        assert capabilities.supports_function_calling is True
        assert capabilities.supports_structured_output is True
        assert capabilities.max_tokens_limit == 8192
        assert capabilities.max_context_length == 16384
        assert capabilities.supported_parameters == ["messages", "model"]
        assert capabilities.unsupported_parameters == ["temperature", "top_p"]
    
    def test_post_init_with_none_parameters(self):
        """测试__post_init__处理None参数"""
        capabilities = ModelCapabilities(
            supported_parameters=None,
            unsupported_parameters=None
        )
        
        assert capabilities.supported_parameters == []
        assert capabilities.unsupported_parameters == []


class TestReasoningModel:
    """测试推理模型类"""
    
    def test_initialization(self):
        """测试推理模型初始化"""
        capabilities = ModelCapabilities(supports_reasoning=True)
        model = ReasoningModel(
            name="test-reasoner",
            provider="test-provider",
            capabilities=capabilities,
            reasoning_format="thinking",
            max_reasoning_tokens=16384,
            supports_chain_of_thought=True,
            requires_special_handling=True
        )
        
        assert model.name == "test-reasoner"
        assert model.provider == "test-provider"
        assert model.capabilities == capabilities
        assert model.reasoning_format == "thinking"
        assert model.max_reasoning_tokens == 16384
        assert model.supports_chain_of_thought is True
        assert model.requires_special_handling is True
    
    def test_default_values(self):
        """测试默认值"""
        capabilities = ModelCapabilities()
        model = ReasoningModel(
            name="test-model",
            provider="test-provider",
            capabilities=capabilities
        )
        
        assert model.reasoning_format == "thinking"
        assert model.max_reasoning_tokens == 8192
        assert model.supports_chain_of_thought is True
        assert model.requires_special_handling is True
    
    def test_is_reasoning_model_true(self):
        """测试推理模型判断为真"""
        capabilities = ModelCapabilities(supports_reasoning=True)
        model = ReasoningModel(
            name="test-reasoner",
            provider="test-provider",
            capabilities=capabilities
        )
        
        assert model.is_reasoning_model() is True
    
    def test_is_reasoning_model_false(self):
        """测试推理模型判断为假"""
        capabilities = ModelCapabilities(supports_reasoning=False)
        model = ReasoningModel(
            name="test-model",
            provider="test-provider",
            capabilities=capabilities
        )
        
        assert model.is_reasoning_model() is False
    
    def test_get_filtered_parameters(self):
        """测试参数过滤"""
        capabilities = ModelCapabilities(
            unsupported_parameters=["temperature", "top_p", "frequency_penalty"]
        )
        model = ReasoningModel(
            name="test-reasoner",
            provider="test-provider",
            capabilities=capabilities
        )
        
        original_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-reasoner",
            "max_tokens": 1000,
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "stream": True
        }
        
        filtered_params = model.get_filtered_parameters(original_params)
        
        expected_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-reasoner",
            "max_tokens": 1000,
            "stream": True
        }
        
        assert filtered_params == expected_params
        # 确保原始参数未被修改
        assert "temperature" in original_params
    
    def test_get_filtered_parameters_no_unsupported(self):
        """测试无不支持参数的过滤"""
        capabilities = ModelCapabilities(unsupported_parameters=[])
        model = ReasoningModel(
            name="test-model",
            provider="test-provider",
            capabilities=capabilities
        )
        
        original_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
            "temperature": 0.7
        }
        
        filtered_params = model.get_filtered_parameters(original_params)
        
        assert filtered_params == original_params


class TestModelCapabilitiesConfig:
    """测试模型能力配置"""
    
    def test_config_contains_expected_models(self):
        """测试配置包含预期的模型"""
        expected_models = {
            "deepseek-reasoner",
            "deepseek-chat",
            "ernie-x1-turbo-32k",
            "ernie-3.5-8k",
            "ernie-4.0-turbo-8k",
            "doubao-seed-1-6-250615",
            "doubao-1-5-pro-32k-character-250715"
        }
        
        actual_models = set(MODEL_CAPABILITIES_CONFIG.keys())
        assert expected_models.issubset(actual_models)
    
    def test_deepseek_reasoner_config(self):
        """测试DeepSeek推理模型配置"""
        config = MODEL_CAPABILITIES_CONFIG["deepseek-reasoner"]
        
        assert config.supports_reasoning is True
        assert config.supports_streaming is True
        assert config.supports_temperature is True
        assert config.supports_system_message is False
        assert config.supports_function_calling is False
        assert config.supports_structured_output is False
        assert config.max_tokens_limit == 32768
        assert config.max_context_length == 32768
        assert "temperature" in config.supported_parameters
        assert "frequency_penalty" in config.unsupported_parameters
    
    def test_deepseek_chat_config(self):
        """测试DeepSeek聊天模型配置"""
        config = MODEL_CAPABILITIES_CONFIG["deepseek-chat"]
        
        assert config.supports_reasoning is False
        assert config.supports_streaming is True
        assert config.supports_temperature is True
        assert config.supports_system_message is True
        assert config.supports_function_calling is True
        assert config.supports_structured_output is True
        assert config.max_tokens_limit == 4096
        assert config.max_context_length == 32768
        assert "functions" in config.supported_parameters
        assert config.unsupported_parameters == []
    
    def test_ernie_x1_turbo_config(self):
        """测试文心一言X1 Turbo配置"""
        config = MODEL_CAPABILITIES_CONFIG["ernie-x1-turbo-32k"]
        
        assert config.supports_reasoning is True
        assert config.supports_streaming is True
        assert config.supports_temperature is True
        assert config.supports_system_message is True
        assert config.supports_function_calling is False
        assert config.supports_structured_output is False
        assert config.max_tokens_limit == 32768
        assert config.max_context_length == 32768
    
    def test_doubao_seed_config(self):
        """测试豆包推理模型配置"""
        config = MODEL_CAPABILITIES_CONFIG["doubao-seed-1-6-250615"]
        
        assert config.supports_reasoning is True
        assert config.supports_streaming is True
        assert config.supports_temperature is True
        assert config.supports_system_message is True
        assert config.supports_function_calling is False
        assert config.supports_structured_output is True
        assert config.max_tokens_limit == 32768
        assert config.max_context_length == 32768


class TestReasoningModelPatterns:
    """测试推理模型模式"""
    
    def test_reasoning_patterns_exist(self):
        """测试推理模型模式存在"""
        assert len(REASONING_MODEL_PATTERNS) > 0
        
        expected_patterns = [
            r".*-r\d+.*",
            r".*-reasoner.*",
            r".*reasoning.*",
            r".*think.*"
        ]
        
        for pattern in expected_patterns:
            assert pattern in REASONING_MODEL_PATTERNS


class TestIsReasoningModel:
    """测试推理模型判断函数"""
    
    def test_empty_model_name(self):
        """测试空模型名称"""
        assert is_reasoning_model("") is False
        assert is_reasoning_model(None) is False
    
    def test_predefined_reasoning_models(self):
        """测试预定义的推理模型"""
        reasoning_models = [
            "deepseek-reasoner",
            "ernie-x1-turbo-32k",
            "doubao-seed-1-6-250615"
        ]
        
        for model in reasoning_models:
            assert is_reasoning_model(model) is True
    
    def test_predefined_non_reasoning_models(self):
        """测试预定义的非推理模型"""
        non_reasoning_models = [
            "deepseek-chat",
            "ernie-3.5-8k",
            "ernie-4.0-turbo-8k",
            "doubao-1-5-pro-32k-character-250715"
        ]
        
        for model in non_reasoning_models:
            assert is_reasoning_model(model) is False
    
    def test_pattern_matching_reasoning_models(self):
        """测试模式匹配的推理模型"""
        pattern_models = [
            "gpt-4-r1",
            "claude-reasoner",
            "llama-reasoning",
            "gemini-think"
        ]
        
        for model in pattern_models:
            assert is_reasoning_model(model) is True
    
    def test_pattern_matching_non_reasoning_models(self):
        """测试模式匹配的非推理模型"""
        non_pattern_models = [
            "gpt-4",
            "claude-3",
            "llama-2",
            "gemini-pro"
        ]
        
        for model in non_pattern_models:
            assert is_reasoning_model(model) is False


class TestGetModelCapabilities:
    """测试获取模型能力函数"""
    
    def test_predefined_model_capabilities(self):
        """测试预定义模型能力"""
        capabilities = get_model_capabilities("deepseek-reasoner")
        
        assert capabilities.supports_reasoning is True
        assert capabilities.supports_streaming is True
        assert capabilities.max_tokens_limit == 32768
    
    def test_unknown_reasoning_model_capabilities(self):
        """测试未知推理模型能力"""
        with patch('harborai.core.models.is_reasoning_model', return_value=True):
            capabilities = get_model_capabilities("unknown-reasoner")
            
            assert capabilities.supports_reasoning is True
            assert capabilities.supports_streaming is False
            assert capabilities.supports_temperature is False
            assert capabilities.supports_system_message is False
            assert capabilities.max_tokens_limit == 32768
            assert "temperature" in capabilities.unsupported_parameters
    
    def test_unknown_non_reasoning_model_capabilities(self):
        """测试未知非推理模型能力"""
        with patch('harborai.core.models.is_reasoning_model', return_value=False):
            capabilities = get_model_capabilities("unknown-model")
            
            assert capabilities.supports_reasoning is False
            assert capabilities.supports_streaming is True
            assert capabilities.supports_temperature is True
            assert capabilities.supports_system_message is True
            assert capabilities.max_tokens_limit == 4096


class TestCreateReasoningModel:
    """测试创建推理模型函数"""
    
    def test_create_predefined_reasoning_model(self):
        """测试创建预定义推理模型"""
        model = create_reasoning_model("deepseek-reasoner", "deepseek")
        
        assert model.name == "deepseek-reasoner"
        assert model.provider == "deepseek"
        assert model.capabilities.supports_reasoning is True
        assert model.reasoning_format == "thinking"
        assert model.max_reasoning_tokens == 8192
        assert model.supports_chain_of_thought is True
        assert model.requires_special_handling is True
    
    def test_create_unknown_reasoning_model(self):
        """测试创建未知推理模型"""
        with patch('harborai.core.models.is_reasoning_model', return_value=True):
            model = create_reasoning_model("custom-reasoner", "custom")
            
            assert model.name == "custom-reasoner"
            assert model.provider == "custom"
            assert model.capabilities.supports_reasoning is True
    
    def test_create_reasoning_model_default_provider(self):
        """测试创建推理模型默认提供商"""
        model = create_reasoning_model("test-reasoner")
        
        assert model.name == "test-reasoner"
        assert model.provider == "unknown"


class TestGetSupportedModels:
    """测试获取支持的模型函数"""
    
    def test_get_supported_models_returns_copy(self):
        """测试获取支持的模型返回副本"""
        supported_models = get_supported_models()
        
        # 修改返回的字典不应影响原始配置
        original_count = len(MODEL_CAPABILITIES_CONFIG)
        supported_models["test-model"] = ModelCapabilities()
        
        assert len(MODEL_CAPABILITIES_CONFIG) == original_count
        assert "test-model" not in MODEL_CAPABILITIES_CONFIG
    
    def test_get_supported_models_content(self):
        """测试获取支持的模型内容"""
        supported_models = get_supported_models()
        
        assert "deepseek-reasoner" in supported_models
        assert "deepseek-chat" in supported_models
        assert isinstance(supported_models["deepseek-reasoner"], ModelCapabilities)


class TestGetReasoningModels:
    """测试获取推理模型函数"""
    
    def test_get_reasoning_models_returns_list(self):
        """测试获取推理模型返回列表"""
        reasoning_models = get_reasoning_models()
        
        assert isinstance(reasoning_models, list)
        assert len(reasoning_models) > 0
    
    def test_get_reasoning_models_content(self):
        """测试获取推理模型内容"""
        reasoning_models = get_reasoning_models()
        
        expected_reasoning_models = [
            "deepseek-reasoner",
            "ernie-x1-turbo-32k",
            "doubao-seed-1-6-250615"
        ]
        
        for model in expected_reasoning_models:
            assert model in reasoning_models
    
    def test_get_reasoning_models_excludes_non_reasoning(self):
        """测试获取推理模型排除非推理模型"""
        reasoning_models = get_reasoning_models()
        
        non_reasoning_models = [
            "deepseek-chat",
            "ernie-3.5-8k",
            "doubao-1-5-pro-32k-character-250715"
        ]
        
        for model in non_reasoning_models:
            assert model not in reasoning_models


class TestFilterParametersForModel:
    """测试模型参数过滤函数"""
    
    def test_filter_parameters_predefined_model(self):
        """测试预定义模型参数过滤"""
        original_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "deepseek-reasoner",
            "max_tokens": 1000,
            "temperature": 0.7,
            "frequency_penalty": 0.1,
            "stream": True
        }
        
        with patch('logging.warning') as mock_warning:
            filtered_params = filter_parameters_for_model("deepseek-reasoner", original_params)
            
            # 检查不支持的参数被移除
            assert "frequency_penalty" not in filtered_params
            assert "temperature" in filtered_params  # deepseek-reasoner支持temperature
            assert "stream" in filtered_params
            assert "messages" in filtered_params
            assert "model" in filtered_params
            assert "max_tokens" in filtered_params
            
            # 检查警告日志
            mock_warning.assert_called()
    
    def test_filter_parameters_unknown_model(self):
        """测试未知模型参数过滤"""
        original_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "unknown-model",
            "temperature": 0.7
        }
        
        with patch('harborai.core.models.is_reasoning_model', return_value=False):
            filtered_params = filter_parameters_for_model("unknown-model", original_params)
            
            # 未知非推理模型默认支持所有参数
            assert filtered_params == original_params
    
    def test_filter_parameters_preserves_original(self):
        """测试参数过滤保留原始参数"""
        original_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "frequency_penalty": 0.1
        }
        
        with patch('logging.warning'):
            filter_parameters_for_model("deepseek-reasoner", original_params)
            
            # 原始参数应该保持不变
            assert "frequency_penalty" in original_params
    
    def test_filter_parameters_no_unsupported(self):
        """测试无不支持参数的过滤"""
        original_params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "deepseek-chat",
            "temperature": 0.7
        }
        
        with patch('logging.warning') as mock_warning:
            filtered_params = filter_parameters_for_model("deepseek-chat", original_params)
            
            assert filtered_params == original_params
            # 不应该有警告日志
            mock_warning.assert_not_called()


class TestModelIntegration:
    """测试模型功能集成"""
    
    def test_reasoning_model_workflow(self):
        """测试推理模型完整工作流"""
        # 1. 判断是否为推理模型
        model_name = "deepseek-reasoner"
        assert is_reasoning_model(model_name) is True
        
        # 2. 获取模型能力
        capabilities = get_model_capabilities(model_name)
        assert capabilities.supports_reasoning is True
        
        # 3. 创建推理模型实例
        reasoning_model = create_reasoning_model(model_name, "deepseek")
        assert reasoning_model.is_reasoning_model() is True
        
        # 4. 过滤参数
        params = {
            "messages": [{"role": "user", "content": "Think about this"}],
            "model": model_name,
            "temperature": 0.7,
            "frequency_penalty": 0.1
        }
        
        with patch('logging.warning'):
            filtered_params = reasoning_model.get_filtered_parameters(params)
            assert "frequency_penalty" not in filtered_params
            assert "temperature" in filtered_params
    
    def test_non_reasoning_model_workflow(self):
        """测试非推理模型完整工作流"""
        # 1. 判断是否为推理模型
        model_name = "deepseek-chat"
        assert is_reasoning_model(model_name) is False
        
        # 2. 获取模型能力
        capabilities = get_model_capabilities(model_name)
        assert capabilities.supports_reasoning is False
        assert capabilities.supports_function_calling is True
        
        # 3. 过滤参数（非推理模型支持更多参数）
        params = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": model_name,
            "temperature": 0.7,
            "functions": [{"name": "test_func"}]
        }
        
        with patch('logging.warning') as mock_warning:
            filtered_params = filter_parameters_for_model(model_name, params)
            assert filtered_params == params
            mock_warning.assert_not_called()
    
    def test_get_all_reasoning_models_integration(self):
        """测试获取所有推理模型集成"""
        reasoning_models = get_reasoning_models()
        
        # 验证每个返回的模型确实是推理模型
        for model_name in reasoning_models:
            assert is_reasoning_model(model_name) is True
            capabilities = get_model_capabilities(model_name)
            assert capabilities.supports_reasoning is True
    
    def test_model_capabilities_consistency(self):
        """测试模型能力一致性"""
        for model_name, capabilities in MODEL_CAPABILITIES_CONFIG.items():
            # 推理模型应该在推理模型列表中
            if capabilities.supports_reasoning:
                assert model_name in get_reasoning_models()
                assert is_reasoning_model(model_name) is True
            else:
                assert model_name not in get_reasoning_models()