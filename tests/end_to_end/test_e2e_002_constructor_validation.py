#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI E2E-002 构造函数参数验证测试

测试目标：
- 验证HarborAI构造函数与OpenAI SDK参数对齐
- 测试不同构造方式（api_key、base_url、timeout等参数）
- 验证参数设置正确性
- 确保无异常抛出

测试范围：
- 基础构造函数参数验证
- 参数类型检查
- 默认值验证
- 错误参数处理
- 多厂商API配置兼容性
"""

import os
import sys
import pytest
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from harborai import HarborAI


class TestHarborAIConstructorValidation:
    """HarborAI构造函数参数验证测试类"""
    
    @classmethod
    def setup_class(cls):
        """测试类初始化"""
        # 加载环境变量
        env_path = os.path.join(project_root, '.env')
        print(f"正在加载环境变量文件: {env_path}")
        print(f"环境变量文件是否存在: {os.path.exists(env_path)}")
        
        # 强制加载环境变量
        load_dotenv(env_path, override=True)
        
        # 验证环境变量是否加载成功
        test_vars = ["DEEPSEEK_API_KEY", "WENXIN_API_KEY", "DOUBAO_API_KEY"]
        for var in test_vars:
            value = os.getenv(var)
            if value:
                print(f"✓ {var}: {value[:10]}...")
            else:
                print(f"✗ {var}: 未找到")
        
        # 配置三个厂商的API信息
        cls.vendor_configs = {
            "DEEPSEEK": {
                "api_key": os.getenv("DEEPSEEK_API_KEY"),
                "base_url": os.getenv("DEEPSEEK_BASE_URL"),
                "models": ["deepseek-chat"]
            },
            "WENXIN": {
                "api_key": os.getenv("WENXIN_API_KEY"),
                "base_url": os.getenv("WENXIN_BASE_URL"),
                "models": ["ernie-4.0-8k"]
            },
            "DOUBAO": {
                "api_key": os.getenv("DOUBAO_API_KEY"),
                "base_url": os.getenv("DOUBAO_BASE_URL"),
                "models": ["ep-20241230140956-8xqzx"]
            }
        }
        
        # 过滤出有效配置
        cls.available_configs = {}
        for vendor, config in cls.vendor_configs.items():
            if config["api_key"] and config["base_url"]:
                cls.available_configs[vendor] = config
                print(f"✓ {vendor} API配置已加载")
            else:
                print(f"⚠ {vendor} API配置缺失，跳过相关测试")
        
        if not cls.available_configs:
            pytest.skip("没有可用的API配置，请检查.env文件")
    
    def test_basic_constructor_parameters(self):
        """测试基础构造函数参数"""
        print("\n=== 测试基础构造函数参数 ===")
        
        for vendor, config in self.available_configs.items():
            print(f"\n测试 {vendor} 厂商构造函数...")
            
            # 测试基础参数构造
            try:
                client = HarborAI(
                    api_key=config["api_key"],
                    base_url=config["base_url"]
                )
                
                # 验证客户端对象创建成功
                assert client is not None, f"{vendor}: 客户端对象不应为None"
                assert hasattr(client, 'chat'), f"{vendor}: 客户端应该有chat属性"
                assert hasattr(client.chat, 'completions'), f"{vendor}: chat应该有completions属性"
                
                print(f"  ✓ {vendor} 基础构造函数测试通过")
                
            except Exception as e:
                pytest.fail(f"{vendor} 基础构造函数测试失败: {str(e)}")
    
    def test_constructor_with_timeout(self):
        """测试带超时参数的构造函数"""
        print("\n=== 测试带超时参数的构造函数 ===")
        
        # 使用第一个可用配置进行测试
        vendor = list(self.available_configs.keys())[0]
        config = self.available_configs[vendor]
        
        # 测试不同的超时值
        timeout_values = [10, 30, 60, 120]
        
        for timeout in timeout_values:
            try:
                client = HarborAI(
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                    timeout=timeout
                )
                
                assert client is not None, f"超时值{timeout}s: 客户端对象不应为None"
                print(f"  ✓ 超时参数 {timeout}s 构造测试通过")
                
            except Exception as e:
                pytest.fail(f"超时参数 {timeout}s 构造测试失败: {str(e)}")
    
    def test_constructor_parameter_types(self):
        """测试构造函数参数类型验证"""
        print("\n=== 测试构造函数参数类型验证 ===")
        
        vendor = list(self.available_configs.keys())[0]
        config = self.available_configs[vendor]
        
        # 测试正确的参数类型
        try:
            client = HarborAI(
                api_key=str(config["api_key"]),  # 确保是字符串
                base_url=str(config["base_url"]),  # 确保是字符串
                timeout=30  # 整数
            )
            assert client is not None
            print("  ✓ 正确参数类型构造测试通过")
        except Exception as e:
            pytest.fail(f"正确参数类型构造测试失败: {str(e)}")
        
        # 测试浮点数超时值
        try:
            client = HarborAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
                timeout=30.5  # 浮点数
            )
            assert client is not None
            print("  ✓ 浮点数超时值构造测试通过")
        except Exception as e:
            pytest.fail(f"浮点数超时值构造测试失败: {str(e)}")
    
    def test_constructor_with_invalid_parameters(self):
        """测试无效参数的构造函数"""
        print("\n=== 测试无效参数的构造函数 ===")
        
        # 测试None参数
        try:
            client = HarborAI(
                api_key=None,
                base_url=None
            )
            # 如果没有抛出异常，说明可能有默认处理机制
            print("  ⚠ None参数未抛出异常，可能有默认处理机制")
        except Exception as e:
            print(f"  ✓ None参数正确抛出异常: {type(e).__name__}")
        
        # 测试空字符串参数
        try:
            client = HarborAI(
                api_key="",
                base_url=""
            )
            print("  ⚠ 空字符串参数未抛出异常，可能有默认处理机制")
        except Exception as e:
            print(f"  ✓ 空字符串参数正确抛出异常: {type(e).__name__}")
        
        # 测试负数超时值
        vendor = list(self.available_configs.keys())[0]
        config = self.available_configs[vendor]
        
        try:
            client = HarborAI(
                api_key=config["api_key"],
                base_url=config["base_url"],
                timeout=-10
            )
            print("  ⚠ 负数超时值未抛出异常，可能有默认处理机制")
        except Exception as e:
            print(f"  ✓ 负数超时值正确抛出异常: {type(e).__name__}")
    
    def test_constructor_default_values(self):
        """测试构造函数默认值"""
        print("\n=== 测试构造函数默认值 ===")
        
        vendor = list(self.available_configs.keys())[0]
        config = self.available_configs[vendor]
        
        # 测试仅提供必需参数
        try:
            client = HarborAI(
                api_key=config["api_key"],
                base_url=config["base_url"]
            )
            
            assert client is not None
            print("  ✓ 仅必需参数构造测试通过")
            
            # 尝试获取默认超时值（如果有公开属性的话）
            if hasattr(client, 'timeout'):
                print(f"  ✓ 默认超时值: {client.timeout}")
            elif hasattr(client, '_timeout'):
                print(f"  ✓ 默认超时值: {client._timeout}")
            else:
                print("  ℹ 无法获取超时值属性")
                
        except Exception as e:
            pytest.fail(f"仅必需参数构造测试失败: {str(e)}")
    
    def test_constructor_with_additional_parameters(self):
        """测试带额外参数的构造函数"""
        print("\n=== 测试带额外参数的构造函数 ===")
        
        vendor = list(self.available_configs.keys())[0]
        config = self.available_configs[vendor]
        
        # 测试可能的额外参数
        additional_params = {
            'max_retries': 3,
            'default_headers': {'User-Agent': 'HarborAI-Test'},
            'organization': 'test-org'
        }
        
        for param_name, param_value in additional_params.items():
            try:
                kwargs = {
                    'api_key': config["api_key"],
                    'base_url': config["base_url"],
                    param_name: param_value
                }
                
                client = HarborAI(**kwargs)
                assert client is not None
                print(f"  ✓ 额外参数 {param_name} 构造测试通过")
                
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    print(f"  ℹ 参数 {param_name} 不被支持: {e}")
                else:
                    print(f"  ⚠ 参数 {param_name} 类型错误: {e}")
            except Exception as e:
                print(f"  ⚠ 参数 {param_name} 其他错误: {e}")
    
    def test_all_vendors_constructor_compatibility(self):
        """测试所有厂商的构造函数兼容性"""
        print("\n=== 测试所有厂商的构造函数兼容性 ===")
        
        successful_vendors = []
        failed_vendors = []
        
        for vendor, config in self.available_configs.items():
            try:
                # 测试基础构造
                client = HarborAI(
                    api_key=config["api_key"],
                    base_url=config["base_url"],
                    timeout=30
                )
                
                # 验证客户端基本功能
                assert client is not None
                assert hasattr(client, 'chat')
                assert hasattr(client.chat, 'completions')
                
                successful_vendors.append(vendor)
                print(f"  ✓ {vendor} 构造函数兼容性测试通过")
                
            except Exception as e:
                failed_vendors.append((vendor, str(e)))
                print(f"  ✗ {vendor} 构造函数兼容性测试失败: {e}")
        
        # 输出统计信息
        print(f"\n=== 构造函数兼容性测试统计 ===")
        print(f"成功厂商: {len(successful_vendors)}/{len(self.available_configs)}")
        print(f"成功厂商列表: {', '.join(successful_vendors)}")
        
        if failed_vendors:
            print(f"失败厂商详情:")
            for vendor, error in failed_vendors:
                print(f"  - {vendor}: {error}")
        
        # 至少要有一个厂商成功
        assert len(successful_vendors) > 0, "所有厂商的构造函数都失败了"
    
    def test_constructor_parameter_persistence(self):
        """测试构造函数参数持久性"""
        print("\n=== 测试构造函数参数持久性 ===")
        
        vendor = list(self.available_configs.keys())[0]
        config = self.available_configs[vendor]
        
        # 创建客户端
        client = HarborAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=45
        )
        
        # 检查参数是否正确保存（如果有公开属性）
        param_checks = [
            ('api_key', config["api_key"]),
            ('base_url', config["base_url"]),
            ('timeout', 45)
        ]
        
        for param_name, expected_value in param_checks:
            # 尝试多种可能的属性名
            possible_attrs = [param_name, f'_{param_name}', f'__{param_name}']
            
            found = False
            for attr_name in possible_attrs:
                if hasattr(client, attr_name):
                    actual_value = getattr(client, attr_name)
                    if actual_value == expected_value:
                        print(f"  ✓ 参数 {param_name} 正确保存为 {attr_name}")
                        found = True
                        break
            
            if not found:
                print(f"  ℹ 参数 {param_name} 无法验证（可能为私有属性）")


if __name__ == "__main__":
    # 直接运行测试
    print("=== HarborAI E2E-002 构造函数参数验证测试 ===")
    print("\n正在加载环境变量...")
    
    # 强制加载环境变量
    env_path = os.path.join(project_root, '.env')
    print(f"环境变量文件路径: {env_path}")
    print(f"环境变量文件是否存在: {os.path.exists(env_path)}")
    load_dotenv(env_path, override=True)
    
    # 检查环境变量
    required_vars = [
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
        "WENXIN_API_KEY", "WENXIN_BASE_URL", 
        "DOUBAO_API_KEY", "DOUBAO_BASE_URL"
    ]
    
    available_vars = []
    for var in required_vars:
        value = os.getenv(var)
        if value:
            available_vars.append(var)
            print(f"✓ {var}: {value[:15]}...")
        else:
            print(f"✗ {var}: 未找到")
    
    print(f"\n环境变量检查: {len(available_vars)}/{len(required_vars)} 个变量已配置")
    
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