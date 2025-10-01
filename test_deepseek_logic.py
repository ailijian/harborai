#!/usr/bin/env python3
"""
测试DeepSeek插件的逻辑修改
验证_prepare_deepseek_request和_handle_native_structured_output方法的正确性
"""

import os
import sys
import json
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from harborai.core.plugins.deepseek_plugin import DeepSeekPlugin
from harborai.core.base_plugin import ChatMessage

def test_prepare_deepseek_request_logic():
    """测试_prepare_deepseek_request方法的逻辑"""
    
    print("测试 _prepare_deepseek_request 方法逻辑")
    print("="*50)
    
    # 初始化插件
    plugin = DeepSeekPlugin(
        api_key="test-key",
        base_url="https://api.deepseek.com"
    )
    
    # 测试消息
    messages = [
        ChatMessage(role="user", content="请生成一个用户信息")
    ]
    
    # 测试场景1：原生结构化输出
    print("\n1. 测试原生结构化输出 (structured_provider='native')")
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        }
    }
    
    request_data = plugin._prepare_deepseek_request(
        model="deepseek-chat",
        messages=messages,
        response_format=response_format,
        structured_provider="native"
    )
    
    print(f"  Response format: {request_data.get('response_format')}")
    print(f"  Messages count: {len(request_data.get('messages', []))}")
    
    # 检查是否添加了JSON关键词
    last_message = request_data.get('messages', [])[-1]
    if 'json' in last_message.get('content', '').lower():
        print("  ✅ 已在prompt中添加JSON关键词")
    else:
        print("  ❌ 未在prompt中添加JSON关键词")
    
    print(f"  最后一条消息: {last_message.get('content')}")
    
    # 测试场景2：传统Agently后处理
    print("\n2. 测试传统Agently后处理 (structured_provider='agently')")
    
    # 重新创建消息（避免被修改）
    messages2 = [
        ChatMessage(role="user", content="请生成一个用户信息")
    ]
    
    request_data2 = plugin._prepare_deepseek_request(
        model="deepseek-chat",
        messages=messages2,
        response_format=response_format,
        structured_provider="agently"
    )
    
    print(f"  Response format: {request_data2.get('response_format')}")
    
    # 检查是否未添加JSON关键词
    last_message2 = request_data2.get('messages', [])[-1]
    if 'json' in last_message2.get('content', '').lower():
        print("  ⚠️  在Agently模式下也添加了JSON关键词")
    else:
        print("  ✅ 在Agently模式下未添加JSON关键词")
    
    print(f"  最后一条消息: {last_message2.get('content')}")

def test_ensure_json_keyword_logic():
    """测试_ensure_json_keyword_in_prompt方法的逻辑"""
    
    print("\n\n测试 _ensure_json_keyword_in_prompt 方法逻辑")
    print("="*50)
    
    plugin = DeepSeekPlugin(api_key="test-key")
    
    # 测试场景1：消息中没有JSON关键词
    print("\n1. 测试添加JSON关键词")
    messages1 = [
        {"role": "user", "content": "请生成一个用户信息"}
    ]
    
    plugin._ensure_json_keyword_in_prompt(messages1)
    print(f"  修改后的消息: {messages1[0]['content']}")
    
    if 'json' in messages1[0]['content'].lower():
        print("  ✅ 成功添加JSON关键词")
    else:
        print("  ❌ 未能添加JSON关键词")
    
    # 测试场景2：消息中已有JSON关键词
    print("\n2. 测试已有JSON关键词的情况")
    messages2 = [
        {"role": "user", "content": "请生成一个JSON格式的用户信息"}
    ]
    original_content = messages2[0]['content']
    
    plugin._ensure_json_keyword_in_prompt(messages2)
    print(f"  原始消息: {original_content}")
    print(f"  处理后消息: {messages2[0]['content']}")
    
    if messages2[0]['content'] == original_content:
        print("  ✅ 正确保持原有内容不变")
    else:
        print("  ⚠️  意外修改了已包含JSON关键词的消息")

def test_native_structured_output_logic():
    """测试原生结构化输出的逻辑流程"""
    
    print("\n\n测试原生结构化输出逻辑流程")
    print("="*50)
    
    plugin = DeepSeekPlugin(api_key="test-key")
    
    # 模拟成功的API响应
    mock_response_data = {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"name": "张三", "age": 25, "email": "zhangsan@example.com"}'
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
    }
    
    # 模拟HTTP响应
    mock_http_response = Mock()
    mock_http_response.json.return_value = mock_response_data
    mock_http_response.raise_for_status.return_value = None
    
    # 模拟HTTP客户端
    mock_client = Mock()
    mock_client.post.return_value = mock_http_response
    
    messages = [ChatMessage(role="user", content="请生成用户信息")]
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "user_info",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"}
                }
            }
        }
    }
    
    # 使用patch模拟_get_client方法
    with patch.object(plugin, '_get_client', return_value=mock_client):
        with patch.object(plugin, 'log_response'):
            try:
                result = plugin._handle_native_structured_output(
                    model="deepseek-chat",
                    messages=messages,
                    response_format=response_format
                )
                
                print("  ✅ 原生结构化输出方法执行成功")
                print(f"  响应内容: {result.choices[0].message.content}")
                
                # 检查是否设置了parsed字段
                if hasattr(result, 'parsed') and result.parsed:
                    print(f"  ✅ 成功设置parsed字段: {result.parsed}")
                    print(f"  parsed字段类型: {type(result.parsed)}")
                else:
                    print("  ❌ 未设置parsed字段")
                
                # 验证JSON内容
                try:
                    content_json = json.loads(result.choices[0].message.content)
                    print(f"  ✅ 返回内容是有效JSON: {content_json}")
                except json.JSONDecodeError:
                    print("  ❌ 返回内容不是有效JSON")
                    
            except Exception as e:
                print(f"  ❌ 原生结构化输出方法执行失败: {e}")

def main():
    """主测试函数"""
    print("DeepSeek插件逻辑测试")
    print("="*80)
    
    test_prepare_deepseek_request_logic()
    test_ensure_json_keyword_logic()
    test_native_structured_output_logic()
    
    print("\n" + "="*80)
    print("所有逻辑测试完成")
    print("="*80)

if __name__ == "__main__":
    main()