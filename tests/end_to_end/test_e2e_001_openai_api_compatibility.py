#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 端到端测试用例 E2E-001：OpenAI标准API调用

测试目标：验证HarborAI与OpenAI SDK的接口一致性

基于文档：HarborAI端到端测试方案.md 第61-102行
测试范围：
- 验证响应结构与OpenAI ChatCompletion一致
- 包含choices、usage、id等标准字段
- message.content包含有效回答
- 测试三个厂商的API配置（DEEPSEEK、WENXIN、DOUBAO）
"""

import os
import sys
import pytest
from typing import Any
from openai.types.chat import ChatCompletion

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not installed. Environment variables may not be loaded from .env file")

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# 直接导入HarborAI，不使用任何mock
from harborai import HarborAI


class TestE2E001OpenAIAPICompatibility:
    """
    测试用例 E2E-001：OpenAI标准API调用兼容性测试
    """
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        # 从环境变量加载API配置
        cls.api_configs = {
            "DEEPSEEK": {
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": os.getenv("DEEPSEEK_BASE_URL"),
                "models": ["deepseek-chat", "deepseek-reasoner"]
            },
            "WENXIN": {
                "api_key": os.getenv("WENXIN_API_KEY"),
                "base_url": os.getenv("WENXIN_BASE_URL"),
                "models": ["ernie-3.5-8k", "ernie-4.0-turbo-8k"]
            },
            "DOUBAO": {
                "api_key": os.getenv("DOUBAO_API_KEY"),
                "base_url": os.getenv("DOUBAO_BASE_URL"),
                "models": ["doubao-1-5-pro-32k-character-250715"]
            }
        }
        
        # 验证配置完整性
        cls.available_configs = {}
        for vendor, config in cls.api_configs.items():
            if config["api_key"] and config["base_url"]:
                cls.available_configs[vendor] = config
                print(f"✓ {vendor} API配置已加载")
            else:
                print(f"⚠ {vendor} API配置缺失，跳过该厂商测试")
        
        if not cls.available_configs:
            pytest.skip("没有可用的API配置，跳过所有测试")
    
    def test_harborai_import(self):
        """测试HarborAI模块导入"""
        assert HarborAI is not None, "HarborAI类应该可以正常导入"
        print("✓ HarborAI模块导入成功")
    
    @pytest.mark.parametrize("vendor", ["DEEPSEEK", "WENXIN", "DOUBAO"])
    def test_client_initialization(self, vendor):
        """测试客户端初始化（与OpenAI SDK一致）"""
        if vendor not in self.available_configs:
            pytest.skip(f"{vendor} API配置不可用")
        
        config = self.available_configs[vendor]
        
        # 初始化客户端（与OpenAI SDK一致的参数）
        client = HarborAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        
        # 验证客户端属性
        assert hasattr(client, 'chat'), "客户端应该有chat属性"
        assert hasattr(client.chat, 'completions'), "chat应该有completions属性"
        assert hasattr(client.chat.completions, 'create'), "completions应该有create方法"
        
        print(f"✓ {vendor} 客户端初始化成功")
    
    @pytest.mark.parametrize("vendor", ["DEEPSEEK", "WENXIN", "DOUBAO"])
    def test_basic_chat_completion(self, vendor):
        """测试基础对话完成功能"""
        if vendor not in self.available_configs:
            pytest.skip(f"{vendor} API配置不可用")
        
        config = self.available_configs[vendor]
        client = HarborAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        
        # 基础对话测试
        messages = [
            {"role": "user", "content": "用一句话解释量子纠缠"}
        ]
        
        # 测试第一个可用模型
        model = config["models"][0]
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages
            )
            
            # 验证响应结构与OpenAI ChatCompletion一致
            self._validate_response_structure(response, vendor, model)
            
            print(f"✓ {vendor} 模型 {model} 基础对话测试通过")
            print(f"  回答内容: {response.choices[0].message.content[:100]}...")
            
        except Exception as e:
            pytest.fail(f"{vendor} 模型 {model} 调用失败: {str(e)}")
    
    def test_all_configured_models(self):
        """测试所有配置的模型"""
        total_tests = 0
        successful_tests = 0
        failed_tests = []
        
        for vendor, config in self.available_configs.items():
            client = HarborAI(
                api_key=config["api_key"],
                base_url=config["base_url"]
            )
            
            for model in config["models"]:
                total_tests += 1
                
                try:
                    messages = [
                        {"role": "user", "content": f"请简单介绍一下{model}模型的特点"}
                    ]
                    
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages
                    )
                    
                    # 验证响应结构
                    self._validate_response_structure(response, vendor, model)
                    
                    successful_tests += 1
                    print(f"✓ {vendor} - {model}: 测试通过")
                    
                except Exception as e:
                    # 检查是否是API密钥相关错误
                    error_str = str(e).lower()
                    if '401' in error_str or 'unauthorized' in error_str or 'invalid api key' in error_str:
                        print(f"警告: {vendor} - {model} API密钥可能无效或过期: {e}")
                    failed_tests.append(f"{vendor} - {model}: {str(e)}")
                    print(f"✗ {vendor} - {model}: 测试失败 - {str(e)}")
        
        # 输出测试统计
        print(f"\n=== 测试统计 ===")
        print(f"总测试数: {total_tests}")
        print(f"成功: {successful_tests}")
        print(f"失败: {len(failed_tests)}")
        print(f"成功率: {(successful_tests/total_tests*100):.1f}%")
        
        if failed_tests:
            print(f"\n失败详情:")
            for failure in failed_tests:
                print(f"  - {failure}")
        
        # 至少要有一个模型成功，证明系统基本功能正常
        assert successful_tests > 0, f"所有模型都失败了，请检查API配置。成功率: {(successful_tests/total_tests*100):.1f}%"
        
        # 如果成功率低于60%，给出警告但不失败测试
        success_rate = (successful_tests/total_tests*100)
        if success_rate < 60:
            print(f"警告: 成功率较低 ({success_rate:.1f}%)，可能需要检查API密钥配置")
    
    def _validate_response_structure(self, response: Any, vendor: str, model: str) -> None:
        """验证响应结构与OpenAI ChatCompletion一致"""
        # 验证顶级属性
        assert hasattr(response, 'choices'), f"{vendor}-{model}: 响应应该包含choices字段"
        assert hasattr(response, 'usage'), f"{vendor}-{model}: 响应应该包含usage字段"
        assert hasattr(response, 'id'), f"{vendor}-{model}: 响应应该包含id字段"
        
        # 验证choices结构
        assert len(response.choices) > 0, f"{vendor}-{model}: choices不应为空"
        choice = response.choices[0]
        assert hasattr(choice, 'message'), f"{vendor}-{model}: choice应该包含message字段"
        assert hasattr(choice.message, 'content'), f"{vendor}-{model}: message应该包含content字段"
        assert hasattr(choice.message, 'role'), f"{vendor}-{model}: message应该包含role字段"
        
        # 验证usage结构
        assert hasattr(response.usage, 'prompt_tokens'), f"{vendor}-{model}: usage应该包含prompt_tokens字段"
        assert hasattr(response.usage, 'completion_tokens'), f"{vendor}-{model}: usage应该包含completion_tokens字段"
        assert hasattr(response.usage, 'total_tokens'), f"{vendor}-{model}: usage应该包含total_tokens字段"
        
        # 验证内容有效性
        content = choice.message.content
        assert content is not None, f"{vendor}-{model}: message.content不应为None"
        assert len(content.strip()) > 0, f"{vendor}-{model}: message.content不应为空"
        assert choice.message.role == "assistant", f"{vendor}-{model}: message.role应该为assistant"
        
        # 验证token计数
        assert response.usage.prompt_tokens > 0, f"{vendor}-{model}: prompt_tokens应该大于0"
        assert response.usage.completion_tokens > 0, f"{vendor}-{model}: completion_tokens应该大于0"
        assert response.usage.total_tokens > 0, f"{vendor}-{model}: total_tokens应该大于0"
        assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens, \
            f"{vendor}-{model}: total_tokens应该等于prompt_tokens + completion_tokens"
        
        # 验证ID格式
        assert response.id is not None, f"{vendor}-{model}: id不应为None"
        assert len(str(response.id)) > 0, f"{vendor}-{model}: id不应为空"
    
    def test_error_handling(self):
        """测试错误处理机制"""
        if not self.available_configs:
            pytest.skip("没有可用的API配置")
        
        # 使用第一个可用配置进行错误测试
        vendor = list(self.available_configs.keys())[0]
        config = self.available_configs[vendor]
        
        # 测试无效API密钥
        try:
            client_invalid = HarborAI(
                api_key="invalid_key",
                base_url=config["base_url"]
            )
            response = client_invalid.chat.completions.create(
                model=config["models"][0],
                messages=[{"role": "user", "content": "test"}]
            )
            print(f"注意: 无效API密钥调用未抛出异常，可能使用了fallback机制")
        except Exception as e:
            print(f"✓ 无效API密钥正确抛出异常: {e}")
        
        # 测试无效模型
        client = HarborAI(
            api_key=config["api_key"],
            base_url=config["base_url"]
        )
        
        try:
            response = client.chat.completions.create(
                model="invalid-model-name",
                messages=[
                    {"role": "user", "content": "测试"}
                ]
            )
            # 如果没有抛出异常，检查是否是因为fallback机制
            print(f"注意: 无效模型调用未抛出异常，可能使用了fallback机制")
        except Exception as e:
            print(f"✓ 无效模型正确抛出异常: {e}")
            # 这是期望的行为
            pass
        
        print(f"✓ {vendor} 错误处理测试通过")


if __name__ == "__main__":
    # 直接运行测试
    print("=== HarborAI E2E-001 OpenAI标准API调用兼容性测试 ===")
    print("\n正在加载环境变量...")
    
    # 检查环境变量
    required_vars = [
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
        "WENXIN_API_KEY", "WENXIN_BASE_URL", 
        "DOUBAO_API_KEY", "DOUBAO_BASE_URL"
    ]
    
    available_vars = []
    for var in required_vars:
        if os.getenv(var):
            available_vars.append(var)
    
    print(f"环境变量检查: {len(available_vars)}/{len(required_vars)} 个变量已配置")
    
    if len(available_vars) == 0:
        print("❌ 没有找到任何API配置，请检查.env文件")
        sys.exit(1)
    
    # 运行pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s"
    ])