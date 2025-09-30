#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
探索 Agently API 的测试脚本
"""

import json
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import Agently
    print("✓ Agently 导入成功")
    print(f"Agently 版本: {getattr(Agently, '__version__', '未知')}")
    print(f"Agently 可用属性: {[attr for attr in dir(Agently) if not attr.startswith('_')]}")
except ImportError as e:
    print(f"✗ Agently 导入失败: {e}")
    sys.exit(1)

# 探索 Agently 的 API
print("\n=== 探索 Agently API ===")

# 尝试创建 agent
try:
    agent = Agently.create_agent()
    print("✓ 成功创建 agent")
    print(f"Agent 类型: {type(agent)}")
    print(f"Agent 可用方法: {[method for method in dir(agent) if not method.startswith('_')]}")
except Exception as e:
    print(f"✗ 创建 agent 失败: {e}")
    sys.exit(1)

# 探索配置方法
print("\n=== 探索配置方法 ===")

# 检查是否有全局设置方法
if hasattr(Agently, 'set_settings'):
    print("✓ 找到 Agently.set_settings 方法")
else:
    print("✗ 未找到 Agently.set_settings 方法")

if hasattr(Agently, 'settings'):
    print("✓ 找到 Agently.settings 属性")
else:
    print("✗ 未找到 Agently.settings 属性")

# 检查 agent 的配置方法
config_methods = ['set_settings', 'settings', 'config', 'configure', 'set_config']
for method in config_methods:
    if hasattr(agent, method):
        print(f"✓ Agent 有 {method} 方法")
    else:
        print(f"✗ Agent 没有 {method} 方法")

# 尝试不同的配置方式
print("\n=== 尝试配置方式 ===")

# 方式1: 通过 agent.set_settings
try:
    if hasattr(agent, 'set_settings'):
        agent.set_settings("model.OAIClient.base_url", "https://ark.cn-beijing.volces.com/api/v3")
        agent.set_settings("model.OAIClient.api_key", "6c39786b-2758-4dc3-8b88-a3e8b60d96b3")
        agent.set_settings("model.OAIClient.model", "ep-20250509161856-ntmhj")
        print("✓ 方式1: agent.set_settings 配置成功")
    else:
        print("✗ 方式1: agent.set_settings 不可用")
except Exception as e:
    print(f"✗ 方式1: agent.set_settings 配置失败: {e}")

# 方式2: 链式配置
try:
    agent2 = Agently.create_agent()
    if hasattr(agent2, 'set_settings'):
        agent2 = (agent2
                 .set_settings("model.OAIClient.base_url", "https://ark.cn-beijing.volces.com/api/v3")
                 .set_settings("model.OAIClient.api_key", "6c39786b-2758-4dc3-8b88-a3e8b60d96b3")
                 .set_settings("model.OAIClient.model", "ep-20250509161856-ntmhj"))
        print("✓ 方式2: 链式配置成功")
    else:
        print("✗ 方式2: 链式配置不可用")
except Exception as e:
    print(f"✗ 方式2: 链式配置失败: {e}")

# 方式3: 检查是否有其他配置方法
try:
    # 检查是否有 use_model 方法
    if hasattr(agent, 'use_model'):
        print("✓ 找到 use_model 方法")
        # 尝试使用
        agent3 = Agently.create_agent()
        agent3.use_model("OpenAI", {
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "api_key": "6c39786b-2758-4dc3-8b88-a3e8b60d96b3",
            "model": "ep-20250509161856-ntmhj"
        })
        print("✓ 方式3: use_model 配置成功")
    else:
        print("✗ 未找到 use_model 方法")
except Exception as e:
    print(f"✗ 方式3: use_model 配置失败: {e}")

# 测试简单的结构化输出
print("\n=== 测试结构化输出 ===")

try:
    # 使用已配置的 agent
    output_format = {
        "sentiment": ("String", "情感分析结果"),
        "confidence": ("Number", "置信度")
    }
    
    result = agent.input("今天天气真好！").output(output_format).start()
    print(f"✓ 结构化输出成功: {result}")
    
except Exception as e:
    print(f"✗ 结构化输出失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== API 探索完成 ===")