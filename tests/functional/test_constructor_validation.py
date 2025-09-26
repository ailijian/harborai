#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI构造函数验证测试

验证HarborAI构造函数与OpenAI SDK的参数一致性和实例创建功能。
"""

import pytest
import os
from unittest.mock import Mock, patch
from typing import Dict, Any

from harborai import HarborAI
from harborai.utils.exceptions import HarborAIError


class TestHarborAIConstructor:
    """HarborAI构造函数测试类"""
    
    def test_constructor_parameters_consistency(self):
        """测试构造函数参数与OpenAI一致性"""
        # 测试基本参数
        client = HarborAI(
            api_key="test-key",
            base_url="https://api.test.com",
            timeout=30.0,
            max_retries=3
        )
        
        assert client.config["api_key"] == "test-key"
        assert client.config["base_url"] == "https://api.test.com"
        assert client.config["timeout"] == 30.0
        assert client.config["max_retries"] == 3
        
    def test_constructor_optional_parameters(self):
        """测试可选参数"""
        client = HarborAI(
            api_key="test-key",
            organization="test-org",
            project="test-project",
            default_headers={"Custom-Header": "value"},
            default_query={"version": "v1"}
        )
        
        assert client.config["organization"] == "test-org"
        assert client.config["project"] == "test-project"
        assert client.config["default_headers"]["Custom-Header"] == "value"
        assert client.config["default_query"]["version"] == "v1"
        
    def test_constructor_with_minimal_params(self):
        """测试最小参数构造"""
        client = HarborAI()
        
        # 验证默认值
        assert client.config["api_key"] is None
        assert client.config["base_url"] is None
        assert client.config["timeout"] is not None  # 应该有默认值
        assert client.config["max_retries"] is not None  # 应该有默认值
        
    def test_instance_creation_success(self):
        """测试实例创建成功"""
        client = HarborAI(api_key="test-key")
        
        # 验证实例属性
        assert hasattr(client, 'chat')
        assert hasattr(client, 'client_manager')
        assert hasattr(client, 'config')
        assert hasattr(client, 'logger')
        
        # 验证chat接口
        assert hasattr(client.chat, 'completions')
        assert hasattr(client.chat.completions, 'create')
        assert hasattr(client.chat.completions, 'acreate')
        
    def test_client_manager_initialization(self):
        """测试客户端管理器初始化"""
        client = HarborAI(api_key="test-key", base_url="https://api.test.com")
        
        # 验证客户端管理器接收到正确的配置
        assert client.client_manager is not None
        # 客户端管理器应该接收到客户端配置
        # 这里我们验证配置是否正确传递
        
    def test_available_methods(self):
        """测试可用方法"""
        client = HarborAI(api_key="test-key")
        
        # 验证公共方法存在
        assert callable(getattr(client, 'get_available_models', None))
        assert callable(getattr(client, 'get_plugin_info', None))
        assert callable(getattr(client, 'register_plugin', None))
        assert callable(getattr(client, 'unregister_plugin', None))
        assert callable(getattr(client, 'get_total_cost', None))
        assert callable(getattr(client, 'reset_cost', None))
        assert callable(getattr(client, 'close', None))
        assert callable(getattr(client, 'aclose', None))
        
    def test_context_manager_support(self):
        """测试上下文管理器支持"""
        # 测试同步上下文管理器
        with HarborAI(api_key="test-key") as client:
            assert client is not None
            assert hasattr(client, 'chat')
        
    @pytest.mark.asyncio
    async def test_async_context_manager_support(self):
        """测试异步上下文管理器支持"""
        async with HarborAI(api_key="test-key") as client:
            assert client is not None
            assert hasattr(client, 'chat')
            
    def test_constructor_with_kwargs(self):
        """测试额外关键字参数"""
        client = HarborAI(
            api_key="test-key",
            custom_param="custom_value",
            another_param=123
        )
        
        # 验证额外参数被保存
        assert client.config["custom_param"] == "custom_value"
        assert client.config["another_param"] == 123
        
    def test_default_settings_integration(self):
        """测试默认设置集成"""
        client = HarborAI()
        
        # 验证设置对象存在
        assert hasattr(client, 'settings')
        assert client.settings is not None
        
        # 验证默认值来自设置
        assert client.config["timeout"] == client.settings.default_timeout
        assert client.config["max_retries"] == client.settings.max_retries
        
    def test_logger_initialization(self):
        """测试日志记录器初始化"""
        client = HarborAI(api_key="test-key")
        
        # 验证日志记录器存在
        assert hasattr(client, 'logger')
        assert client.logger is not None
        
    def test_parameter_types(self):
        """测试参数类型验证"""
        # 测试正确的参数类型
        client = HarborAI(
            api_key="string",
            timeout=30.0,  # float
            max_retries=3,  # int
            default_headers={"key": "value"},  # dict
            default_query={"param": "value"}  # dict
        )
        
        assert isinstance(client.config["api_key"], str)
        assert isinstance(client.config["timeout"], float)
        assert isinstance(client.config["max_retries"], int)
        assert isinstance(client.config["default_headers"], dict)
        assert isinstance(client.config["default_query"], dict)
        
    def test_openai_compatibility_parameters(self):
        """测试与OpenAI SDK兼容的参数"""
        # 这些是OpenAI SDK支持的主要参数
        openai_compatible_params = {
            "api_key": "test-key",
            "organization": "test-org", 
            "project": "test-project",
            "base_url": "https://api.test.com",
            "timeout": 30.0,
            "max_retries": 3,
            "default_headers": {"Authorization": "Bearer token"},
            "default_query": {"version": "v1"}
        }
        
        # 验证HarborAI可以接受所有这些参数
        client = HarborAI(**openai_compatible_params)
        
        for param, value in openai_compatible_params.items():
            assert client.config[param] == value
            
    def test_internal_routing_setup(self):
        """测试内部路由设置"""
        client = HarborAI(api_key="test-key")
        
        # 验证聊天接口路由
        assert client.chat is not None
        assert client.chat.completions is not None
        
        # 验证客户端管理器连接
        assert client.chat.completions.client_manager is client.client_manager
        
    def test_configuration_loading(self):
        """测试配置加载"""
        test_config = {
            "api_key": "test-key",
            "base_url": "https://custom.api.com",
            "timeout": 45.0,
            "max_retries": 5
        }
        
        client = HarborAI(**test_config)
        
        # 验证配置正确加载
        for key, value in test_config.items():
            assert client.config[key] == value
            
        # 验证配置传递给客户端管理器
        # 客户端管理器应该接收到完整的客户端配置
        assert client.client_manager is not None