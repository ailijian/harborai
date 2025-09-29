# -*- coding: utf-8 -*-
"""
多厂商集成测试

本模块测试 HarborAI 对多个AI厂商的集成支持，包括：
- DeepSeek API 集成
- 百度 ERNIE API 集成
- 字节跳动 Doubao API 集成
- 厂商间切换和兼容性
- 统一接口适配
- 错误处理和降级
"""

import pytest
import asyncio
import time
import json
import os
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from enum import Enum

# 导入 HarborAI 相关模块
try:
    from harborai import HarborAI
    from harborai.core.exceptions import HarborAIError, APIError, RateLimitError
    from harborai.core.plugins.deepseek_plugin import DeepSeekPlugin
    from harborai.core.plugins.wenxin_plugin import WenxinPlugin
    from harborai.core.plugins.doubao_plugin import DoubaoPlugin

    from harborai.core.vendor_manager import VendorManager
except ImportError as e:
    pytest.skip(f"无法导入 HarborAI 多厂商模块: {e}", allow_module_level=True)

from tests.integration import INTEGRATION_TEST_CONFIG, SUPPORTED_VENDORS, TEST_DATA_CONFIG


class VendorType(Enum):
    """支持的厂商类型"""
    DEEPSEEK = "deepseek"
    ERNIE = "ernie"
    DOUBAO = "doubao"


@dataclass
class VendorConfig:
    """厂商配置数据类"""
    name: str
    api_key: str
    base_url: str
    models: List[str]
    max_tokens: int
    supports_streaming: bool
    supports_function_calling: bool
    rate_limit_rpm: int


class TestMultiVendorIntegration:
    """
    多厂商集成测试类
    
    测试不同AI厂商的API集成和兼容性。
    """
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """测试方法设置"""
        self.test_message = TEST_DATA_CONFIG["simple_message"]
        self.test_messages = [{"role": "user", "content": self.test_message}]
        
        # 厂商配置
        self.vendor_configs = {
            VendorType.DEEPSEEK: VendorConfig(
                name="DeepSeek",
                api_key="sk-test-deepseek",
                base_url="https://api.deepseek.com",
                models=["deepseek-chat", "deepseek-reasoner"],
                max_tokens=4096,
                supports_streaming=True,
                supports_function_calling=True,
                rate_limit_rpm=60
            ),
            VendorType.ERNIE: VendorConfig(
                name="ERNIE",
                api_key="test-ernie-token",
                base_url="https://aip.baidubce.com",
                models=["ernie-3.5-8k", "ernie-4.0-turbo-8k"],
                max_tokens=2048,
                supports_streaming=True,
                supports_function_calling=False,
                rate_limit_rpm=100
            ),
            VendorType.DOUBAO: VendorConfig(
                name="Doubao",
                api_key="test-doubao-key",
                base_url="https://ark.cn-beijing.volces.com",
                models=["doubao-seed-1-6-250615", "doubao-1-5-pro-32k-character-250715"],
                max_tokens=8192,
                supports_streaming=True,
                supports_function_calling=True,
                rate_limit_rpm=120
            ),

        }
    
    @pytest.fixture
    def mock_vendor_manager(self):
        """Mock 厂商管理器夹具"""
        with patch('harborai.core.vendor_manager.VendorManager') as mock_manager:
            manager_instance = Mock()
            mock_manager.return_value = manager_instance
            
            # 配置厂商列表
            manager_instance.get_available_vendors.return_value = list(self.vendor_configs.keys())
            manager_instance.get_vendor_config.side_effect = lambda vendor: self.vendor_configs.get(vendor)
            
            yield manager_instance
    
    def create_mock_response(self, vendor: VendorType, content: str = None) -> Mock:
        """创建模拟响应"""
        if content is None:
            content = f"这是来自{self.vendor_configs[vendor].name}的测试响应"
        
        response = Mock()
        response.choices = [Mock(
            message=Mock(
                content=content,
                role="assistant"
            ),
            finish_reason="stop",
            index=0
        )]
        response.usage = Mock(
            prompt_tokens=15,
            completion_tokens=len(content.split()),
            total_tokens=15 + len(content.split())
        )
        response.model = self.vendor_configs[vendor].models[0]
        response.id = f"chatcmpl-{vendor.value}-test"
        response.object = "chat.completion"
        response.created = int(time.time())
        
        return response
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.p0
    def test_deepseek_integration(self):
        """测试 DeepSeek API 集成"""
        with patch('harborai.core.plugins.deepseek_plugin.DeepSeekPlugin') as mock_plugin:
            # 配置 DeepSeek 响应
            mock_instance = Mock()
            mock_plugin.return_value = mock_instance
            
            expected_response = self.create_mock_response(
                VendorType.DEEPSEEK,
                "我是DeepSeek，一个专注于推理和编程的AI助手。"
            )
            mock_instance.chat.completions.create.return_value = expected_response
            
            # 执行测试
            plugin = mock_plugin(api_key="sk-test-deepseek")
            response = plugin.chat.completions.create(
                model="deepseek-chat",
                messages=self.test_messages,
                max_tokens=100
            )
            
            # 验证响应
            assert response is not None
            assert "DeepSeek" in response.choices[0].message.content
            assert response.model == "deepseek-chat"
            
            # 验证API调用
            mock_instance.chat.completions.create.assert_called_once_with(
                model="deepseek-chat",
                messages=self.test_messages,
                max_tokens=100
            )
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.p0
    def test_wenxin_integration(self):
        """测试百度文心一言 API 集成"""
        with patch('harborai.core.plugins.wenxin_plugin.WenxinPlugin') as mock_plugin:
            # 配置文心一言响应
            mock_instance = Mock()
            mock_plugin.return_value = mock_instance
            
            expected_response = self.create_mock_response(
                VendorType.ERNIE,
                "我是文心一言，百度开发的大语言模型。"
            )
            mock_instance.chat.completions.create.return_value = expected_response
            
            # 执行测试
            plugin = mock_plugin(api_key="test-ernie-token")
            response = plugin.chat.completions.create(
                model="ernie-3.5-8k",
                messages=self.test_messages,
                max_tokens=100
            )
            
            # 验证响应
            assert response is not None
            assert "文心一言" in response.choices[0].message.content
            assert response.model == "ernie-3.5-8k"
            
            # 验证API调用
            mock_instance.chat.completions.create.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.p0
    def test_doubao_integration(self):
        """测试字节跳动 Doubao API 集成"""
        with patch('harborai.core.plugins.doubao_plugin.DoubaoPlugin') as mock_plugin:
            # 配置 Doubao 响应
            mock_instance = Mock()
            mock_plugin.return_value = mock_instance
            
            expected_response = self.create_mock_response(
                VendorType.DOUBAO,
                "我是豆包，字节跳动开发的AI助手。"
            )
            mock_instance.chat.completions.create.return_value = expected_response
            
            # 执行测试
            plugin = mock_plugin(api_key="test-doubao-key")
            response = plugin.chat.completions.create(
                model="doubao-seed-1-6-250615",
                messages=self.test_messages,
                max_tokens=100
            )
            
            # 验证响应
            assert response is not None
            assert "豆包" in response.choices[0].message.content
            assert response.model == "doubao-seed-1-6-250615"
            
            # 验证API调用
            mock_instance.chat.completions.create.assert_called_once()
    
    # OpenAI 集成测试已移除，不再支持 OpenAI 模型
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.p1
    def test_vendor_switching(self, mock_vendor_manager):
        """测试厂商切换功能"""
        # 模拟厂商切换场景
        vendor_sequence = [
            VendorType.DEEPSEEK,
            VendorType.ERNIE,
            VendorType.DOUBAO
        ]
        
        responses = []
        
        for vendor in vendor_sequence:
            with patch(f'harborai.vendors.{vendor.value}.{vendor.value.title()}Client') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance
                
                # 配置特定厂商响应
                vendor_response = self.create_mock_response(
                    vendor,
                    f"来自{self.vendor_configs[vendor].name}的响应"
                )
                mock_instance.chat.completions.create.return_value = vendor_response
                
                # 执行API调用
                client = mock_client(api_key=self.vendor_configs[vendor].api_key)
                response = client.chat.completions.create(
                    model=self.vendor_configs[vendor].models[0],
                    messages=self.test_messages
                )
                
                responses.append((vendor, response))
        
        # 验证所有厂商都成功响应
        assert len(responses) == 3
        
        for vendor, response in responses:
            vendor_name = self.vendor_configs[vendor].name
            assert vendor_name in response.choices[0].message.content
            assert response.model == self.vendor_configs[vendor].models[0]
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.p1
    def test_unified_interface_compatibility(self):
        """测试统一接口兼容性"""
        # 测试所有厂商是否都支持统一的接口格式
        common_params = {
            "messages": self.test_messages,
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        for vendor_type in VendorType:
            vendor_config = self.vendor_configs[vendor_type]
            
            with patch(f'harborai.vendors.{vendor_type.value}.{vendor_type.value.title()}Client') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance
                
                # 配置统一格式响应
                unified_response = self.create_mock_response(vendor_type)
                mock_instance.chat.completions.create.return_value = unified_response
                
                # 测试统一接口调用
                client = mock_client(api_key=vendor_config.api_key)
                response = client.chat.completions.create(
                    model=vendor_config.models[0],
                    **common_params
                )
                
                # 验证响应格式一致性
                assert hasattr(response, 'choices')
                assert hasattr(response, 'usage')
                assert hasattr(response, 'model')
                assert hasattr(response, 'id')
                assert len(response.choices) > 0
                assert hasattr(response.choices[0], 'message')
                assert hasattr(response.choices[0].message, 'content')
                assert hasattr(response.choices[0].message, 'role')
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.streaming
    @pytest.mark.p1
    def test_streaming_support_across_vendors(self):
        """测试多厂商流式响应支持"""
        streaming_vendors = [
            vendor for vendor, config in self.vendor_configs.items()
            if config.supports_streaming
        ]
        
        for vendor_type in streaming_vendors:
            with patch(f'harborai.vendors.{vendor_type.value}.{vendor_type.value.title()}Client') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance
                
                # 创建流式响应chunks
                stream_chunks = [
                    Mock(
                        choices=[Mock(
                            delta=Mock(content="流式"),
                            index=0,
                            finish_reason=None
                        )],
                        id=f"chatcmpl-{vendor_type.value}-stream",
                        object="chat.completion.chunk",
                        model=self.vendor_configs[vendor_type].models[0]
                    ),
                    Mock(
                        choices=[Mock(
                            delta=Mock(content="响应测试"),
                            index=0,
                            finish_reason="stop"
                        )],
                        id=f"chatcmpl-{vendor_type.value}-stream",
                        object="chat.completion.chunk",
                        model=self.vendor_configs[vendor_type].models[0]
                    )
                ]
                
                mock_instance.chat.completions.create.return_value = iter(stream_chunks)
                
                # 测试流式调用
                client = mock_client(api_key=self.vendor_configs[vendor_type].api_key)
                stream = client.chat.completions.create(
                    model=self.vendor_configs[vendor_type].models[0],
                    messages=self.test_messages,
                    stream=True
                )
                
                # 收集流式内容
                collected_content = []
                for chunk in stream:
                    if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                        collected_content.append(chunk.choices[0].delta.content)
                
                # 验证流式响应
                assert len(collected_content) == 2
                assert ''.join(collected_content) == "流式响应测试"
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.p1
    def test_function_calling_support(self):
        """测试函数调用支持"""
        function_calling_vendors = [
            vendor for vendor, config in self.vendor_configs.items()
            if config.supports_function_calling
        ]
        
        # 定义测试函数
        test_function = {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称"
                    }
                },
                "required": ["city"]
            }
        }
        
        for vendor_type in function_calling_vendors:
            with patch(f'harborai.vendors.{vendor_type.value}.{vendor_type.value.title()}Client') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance
                
                # 配置函数调用响应
                function_response = Mock()
                function_response.choices = [Mock(
                    message=Mock(
                        content=None,
                        role="assistant",
                        function_call=Mock(
                            name="get_weather",
                            arguments=json.dumps({"city": "北京"})
                        )
                    ),
                    finish_reason="function_call"
                )]
                function_response.usage = Mock(
                    prompt_tokens=25,
                    completion_tokens=10,
                    total_tokens=35
                )
                
                mock_instance.chat.completions.create.return_value = function_response
                
                # 测试函数调用
                client = mock_client(api_key=self.vendor_configs[vendor_type].api_key)
                response = client.chat.completions.create(
                    model=self.vendor_configs[vendor_type].models[0],
                    messages=[{"role": "user", "content": "北京今天天气怎么样？"}],
                    functions=[test_function],
                    function_call="auto"
                )
                
                # 验证函数调用响应
                assert response.choices[0].finish_reason == "function_call"
                assert hasattr(response.choices[0].message, 'function_call')
                assert response.choices[0].message.function_call.name == "get_weather"
                
                # 验证函数参数
                args = json.loads(response.choices[0].message.function_call.arguments)
                assert "city" in args
                assert args["city"] == "北京"
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.p2
    def test_vendor_failover_mechanism(self):
        """测试厂商故障转移机制"""
        # 配置故障转移序列：DeepSeek -> ERNIE -> Doubao
        failover_sequence = [
            VendorType.DEEPSEEK,
            VendorType.ERNIE,
            VendorType.DOUBAO
        ]
        
        # 模拟第一个厂商失败，第二个厂商成功
        with patch('harborai.vendors.deepseek.DeepSeekClient') as mock_deepseek, \
             patch('harborai.vendors.ernie.ErnieClient') as mock_ernie:
            
            # DeepSeek 失败
            mock_deepseek_instance = Mock()
            mock_deepseek.return_value = mock_deepseek_instance
            mock_deepseek_instance.chat.completions.create.side_effect = APIError(
                "DeepSeek API 暂时不可用", status_code=503
            )
            
            # ERNIE 成功
            mock_ernie_instance = Mock()
            mock_ernie.return_value = mock_ernie_instance
            ernie_response = self.create_mock_response(
                VendorType.ERNIE,
                "故障转移成功，这是ERNIE的响应"
            )
            mock_ernie_instance.chat.completions.create.return_value = ernie_response
            
            # 模拟故障转移逻辑
            last_error = None
            successful_response = None
            
            for vendor_type in failover_sequence:
                try:
                    if vendor_type == VendorType.DEEPSEEK:
                        client = mock_deepseek(api_key=self.vendor_configs[vendor_type].api_key)
                    elif vendor_type == VendorType.ERNIE:
                        client = mock_ernie(api_key=self.vendor_configs[vendor_type].api_key)
                    else:
                        continue  # 跳过其他厂商
                    
                    response = client.chat.completions.create(
                        model=self.vendor_configs[vendor_type].models[0],
                        messages=self.test_messages
                    )
                    
                    successful_response = response
                    break
                    
                except APIError as e:
                    last_error = e
                    continue
            
            # 验证故障转移结果
            assert successful_response is not None
            assert "ERNIE" in successful_response.choices[0].message.content
            assert "故障转移成功" in successful_response.choices[0].message.content
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.p2
    def test_rate_limit_handling(self):
        """测试速率限制处理"""
        # 测试不同厂商的速率限制处理
        for vendor_type, config in self.vendor_configs.items():
            with patch(f'harborai.vendors.{vendor_type.value}.{vendor_type.value.title()}Client') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance
                
                # 模拟速率限制错误
                from harborai.core.exceptions import RateLimitError
                mock_instance.chat.completions.create.side_effect = [
                    RateLimitError(
                        f"{config.name} API 速率限制",
                        retry_after=1,
                        rate_limit_rpm=config.rate_limit_rpm
                    ),
                    self.create_mock_response(vendor_type, "速率限制恢复后的响应")
                ]
                
                # 模拟重试逻辑
                client = mock_client(api_key=config.api_key)
                retry_count = 0
                max_retries = 2
                
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(
                            model=config.models[0],
                            messages=self.test_messages
                        )
                        break
                    except RateLimitError as e:
                        retry_count += 1
                        if attempt < max_retries - 1:
                            time.sleep(0.1)  # 短暂延迟
                            continue
                        else:
                            pytest.fail(f"{config.name} 速率限制重试失败")
                
                # 验证重试成功
                assert retry_count == 1
                assert "速率限制恢复" in response.choices[0].message.content
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.p2
    def test_vendor_specific_features(self):
        """测试厂商特定功能"""
        # DeepSeek 特定功能：代码生成优化
        with patch('harborai.vendors.deepseek.DeepSeekClient') as mock_deepseek:
            mock_instance = Mock()
            mock_deepseek.return_value = mock_instance
            
            code_response = self.create_mock_response(
                VendorType.DEEPSEEK,
                "```python\ndef hello_world():\n    print('Hello, World!')\n```"
            )
            mock_instance.chat.completions.create.return_value = code_response
            
            client = mock_deepseek(api_key="sk-test-deepseek")
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": "写一个Hello World程序"}],
                # DeepSeek 特定参数
                code_mode=True,
                language="python"
            )
            
            assert "```python" in response.choices[0].message.content
            assert "def hello_world" in response.choices[0].message.content
        
        # ERNIE 特定功能：中文优化
        with patch('harborai.vendors.ernie.ErnieClient') as mock_ernie:
            mock_instance = Mock()
            mock_ernie.return_value = mock_instance
            
            chinese_response = self.create_mock_response(
                VendorType.ERNIE,
                "这是专门针对中文优化的回复，语言表达更加自然流畅。"
            )
            mock_instance.chat.completions.create.return_value = chinese_response
            
            client = mock_ernie(api_key="test-ernie-token")
            response = client.chat.completions.create(
                model="ernie-3.5-8k",
                messages=[{"role": "user", "content": "请用中文回答问题"}],
                # ERNIE 特定参数
                chinese_optimized=True
            )
            
            assert "中文优化" in response.choices[0].message.content
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.p3
    def test_cross_vendor_consistency(self):
        """测试跨厂商一致性"""
        # 使用相同的输入测试所有厂商，比较输出一致性
        test_prompt = "请简单介绍一下人工智能"
        vendor_responses = {}
        
        for vendor_type in VendorType:
            with patch(f'harborai.vendors.{vendor_type.value}.{vendor_type.value.title()}Client') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance
                
                # 配置一致性测试响应
                consistent_response = self.create_mock_response(
                    vendor_type,
                    f"人工智能是一门综合性学科，{self.vendor_configs[vendor_type].name}版本的回答。"
                )
                mock_instance.chat.completions.create.return_value = consistent_response
                
                client = mock_client(api_key=self.vendor_configs[vendor_type].api_key)
                response = client.chat.completions.create(
                    model=self.vendor_configs[vendor_type].models[0],
                    messages=[{"role": "user", "content": test_prompt}],
                    temperature=0.1  # 低温度确保一致性
                )
                
                vendor_responses[vendor_type] = response.choices[0].message.content
        
        # 验证所有厂商都提供了有效响应
        assert len(vendor_responses) == len(VendorType)
        
        for vendor_type, content in vendor_responses.items():
            assert len(content) > 0
            assert "人工智能" in content
            assert self.vendor_configs[vendor_type].name in content
    
    @pytest.mark.integration
    @pytest.mark.multi_vendor
    @pytest.mark.real_api
    @pytest.mark.p3
    def test_real_multi_vendor_apis(self):
        """真实多厂商API测试（需要真实凭证）"""
        # 检查是否启用真实API测试
        if not os.getenv('ENABLE_REAL_API_TESTS', 'false').lower() == 'true':
            pytest.skip("真实API测试未启用，设置ENABLE_REAL_API_TESTS=true启用")
        
        # 检查各厂商API凭证
        vendor_keys = {
            VendorType.DEEPSEEK: os.getenv('DEEPSEEK_API_KEY'),
            VendorType.ERNIE: os.getenv('ERNIE_API_KEY'),
            VendorType.DOUBAO: os.getenv('DOUBAO_API_KEY')
        }
        
        available_vendors = {
            vendor: key for vendor, key in vendor_keys.items() if key
        }
        
        if not available_vendors:
            pytest.skip("没有可用的厂商API密钥")
        
        # 测试可用的厂商
        for vendor_type, api_key in available_vendors.items():
            try:
                # 根据厂商类型创建插件
                if vendor_type == VendorType.DEEPSEEK:
                    plugin = DeepSeekPlugin(api_key=api_key)
                elif vendor_type == VendorType.ERNIE:
                    plugin = WenxinPlugin(api_key=api_key)
                elif vendor_type == VendorType.DOUBAO:
                    plugin = DoubaoPlugin(api_key=api_key)
                
                # 执行真实API调用
                response = plugin.chat.completions.create(
                    model=self.vendor_configs[vendor_type].models[0],
                    messages=self.test_messages,
                    max_tokens=50
                )
                
                # 验证真实响应
                assert response is not None
                assert len(response.choices) > 0
                assert response.choices[0].message.role == "assistant"
                assert len(response.choices[0].message.content) > 0
                assert response.usage.total_tokens > 0
                
            except Exception as e:
                pytest.fail(f"{self.vendor_configs[vendor_type].name} 真实API调用失败: {e}")


class TestVendorCompatibility:
    """
    厂商兼容性测试类
    
    测试不同厂商间的兼容性和互操作性。
    """
    
    @pytest.mark.integration
    @pytest.mark.compatibility
    @pytest.mark.p2
    def test_parameter_compatibility(self):
        """测试参数兼容性"""
        # 定义通用参数集
        common_params = {
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }
        
        # 测试所有厂商是否支持通用参数
        for vendor_type in VendorType:
            with patch(f'harborai.vendors.{vendor_type.value}.{vendor_type.value.title()}Client') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance
                
                # 配置兼容性响应
                mock_instance.chat.completions.create.return_value = Mock(
                    choices=[Mock(
                        message=Mock(
                            content="参数兼容性测试成功",
                            role="assistant"
                        )
                    )]
                )
                
                # 测试参数兼容性
                client = mock_client(api_key="test-key")
                try:
                    response = client.chat.completions.create(
                        model="test-model",
                        messages=[{"role": "user", "content": "测试"}],
                        **common_params
                    )
                    assert response is not None
                except Exception as e:
                    pytest.fail(f"{vendor_type.value} 参数兼容性测试失败: {e}")
    
    @pytest.mark.integration
    @pytest.mark.compatibility
    @pytest.mark.p2
    def test_response_format_compatibility(self):
        """测试响应格式兼容性"""
        # 验证所有厂商返回的响应格式是否一致
        expected_fields = [
            'choices', 'usage', 'model', 'id', 'object', 'created'
        ]
        
        for vendor_type in VendorType:
            with patch(f'harborai.vendors.{vendor_type.value}.{vendor_type.value.title()}Client') as mock_client:
                mock_instance = Mock()
                mock_client.return_value = mock_instance
                
                # 创建标准格式响应
                standard_response = Mock()
                for field in expected_fields:
                    setattr(standard_response, field, f"test_{field}")
                
                standard_response.choices = [Mock(
                    message=Mock(
                        content="格式兼容性测试",
                        role="assistant"
                    ),
                    finish_reason="stop",
                    index=0
                )]
                standard_response.usage = Mock(
                    prompt_tokens=10,
                    completion_tokens=15,
                    total_tokens=25
                )
                
                mock_instance.chat.completions.create.return_value = standard_response
                
                # 测试响应格式
                client = mock_client(api_key="test-key")
                response = client.chat.completions.create(
                    model="test-model",
                    messages=[{"role": "user", "content": "测试"}]
                )
                
                # 验证响应格式
                for field in expected_fields:
                    assert hasattr(response, field), f"{vendor_type.value} 缺少字段: {field}"
                
                # 验证choices结构
                assert len(response.choices) > 0
                choice = response.choices[0]
                assert hasattr(choice, 'message')
                assert hasattr(choice, 'finish_reason')
                assert hasattr(choice, 'index')
                
                # 验证message结构
                message = choice.message
                assert hasattr(message, 'content')
                assert hasattr(message, 'role')
                
                # 验证usage结构
                usage = response.usage
                assert hasattr(usage, 'prompt_tokens')
                assert hasattr(usage, 'completion_tokens')
                assert hasattr(usage, 'total_tokens')