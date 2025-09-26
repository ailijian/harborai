#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试环境变量测试失败问题
"""

import sys
import os
import tempfile
sys.path.append('.')

from tests.functional.test_k_configuration import MockEnvironmentManager

def debug_env_file_loading():
    """调试.env文件加载功能"""
    print("开始调试环境变量文件加载...")
    
    env_manager = MockEnvironmentManager()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # 创建.env文件内容
        env_content = """# HarborAI配置
HARBORAI_API_KEY=sk-test-env-key
HARBORAI_BASE_URL=https://api.example.com
HARBORAI_DEBUG=true
HARBORAI_LOG_LEVEL=DEBUG

# 注释行应该被忽略
# IGNORED_VAR=ignored

# 空行也应该被忽略

HARBORAI_TIMEOUT=30
"""
        
        env_file_path = os.path.join(temp_dir, ".env")
        print(f"创建.env文件: {env_file_path}")
        
        with open(env_file_path, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print(f"文件内容:\n{env_content}")
        
        # 加载.env文件
        print("\n加载.env文件...")
        success = env_manager.load_env_file(env_file_path)
        print(f"加载结果: {success}")
        
        # 检查环境变量
        print("\n检查环境变量:")
        test_vars = [
            "HARBORAI_API_KEY",
            "HARBORAI_BASE_URL", 
            "HARBORAI_DEBUG",
            "HARBORAI_LOG_LEVEL",
            "HARBORAI_TIMEOUT",
            "IGNORED_VAR"
        ]
        
        for var in test_vars:
            value = env_manager.get_env_var(var)
            print(f"  {var}: {value}")
        
        # 检查内部状态
        print(f"\nenv_manager.env_vars: {env_manager.env_vars}")
        
        # 验证具体的断言
        print("\n验证断言:")
        assertions = [
            ("success == True", success == True),
            ("API_KEY == 'sk-test-env-key'", env_manager.get_env_var("HARBORAI_API_KEY") == "sk-test-env-key"),
            ("BASE_URL == 'https://api.example.com'", env_manager.get_env_var("HARBORAI_BASE_URL") == "https://api.example.com"),
            ("DEBUG == 'true'", env_manager.get_env_var("HARBORAI_DEBUG") == "true"),
            ("LOG_LEVEL == 'DEBUG'", env_manager.get_env_var("HARBORAI_LOG_LEVEL") == "DEBUG"),
            ("TIMEOUT == '30'", env_manager.get_env_var("HARBORAI_TIMEOUT") == "30"),
            ("IGNORED_VAR is None", env_manager.get_env_var("IGNORED_VAR") is None)
        ]
        
        for desc, result in assertions:
            status = "✓" if result else "✗"
            print(f"  {status} {desc}: {result}")
            if not result:
                print(f"    失败的断言: {desc}")

if __name__ == "__main__":
    debug_env_file_loading()