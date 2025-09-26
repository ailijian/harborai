#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试脚本 - 用于调试测试失败问题
"""

import sys
import os
import traceback
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

# 设置环境变量
os.environ['HARBORAI_API_KEY'] = 'test-key'
os.environ['OPENAI_API_KEY'] = 'test-openai-key'

try:
    # 导入测试类
    from tests.functional.test_n_standard_alignment import TestStandardAlignment
    
    print("成功导入测试类")
    
    # 创建测试实例
    test_instance = TestStandardAlignment()
    
    print("创建测试实例成功")
    
    # 手动初始化测试数据（模拟setup_method的功能）
    test_instance.test_messages = [
        {"role": "user", "content": "请用一句话解释量子纠缠现象。"}
    ]
    
    test_instance.mock_openai_response = {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "量子纠缠是指两个或多个粒子之间存在的一种量子力学关联，即使它们相距很远，对其中一个粒子的测量会瞬间影响另一个粒子的状态。"
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 25,
            "completion_tokens": 45,
            "total_tokens": 70
        }
    }
    
    print("测试数据初始化成功")
    
    # 创建mock客户端
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    
    print("创建mock客户端成功")
    
    # 直接调用测试方法
    try:
        test_instance.test_n001_openai_sdk_replacement_sync(mock_client)
        print("测试通过！")
    except Exception as e:
        print(f"测试失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        
except Exception as e:
    print(f"导入或初始化失败: {e}")
    print("详细错误信息:")
    traceback.print_exc()