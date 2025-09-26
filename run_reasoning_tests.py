#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理模型功能完整测试脚本
避免pytest环境问题，直接运行所有测试用例
"""

import os
import sys
import traceback
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_environment():
    """设置测试环境"""
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    os.environ["WENXIN_API_KEY"] = "test-key"
    os.environ["DOUBAO_API_KEY"] = "test-key"

def create_mock_client():
    """创建模拟客户端"""
    from harborai import HarborAI
    
    client = HarborAI()
    
    # Mock底层的client_manager调用
    with patch.object(client.client_manager, 'chat_completion_sync_with_fallback') as mock_completion:
        # 创建一个完整的Mock响应对象
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="这是一个复杂的推理问题，需要仔细分析...",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300
        )
        # 添加model_dump方法
        mock_response.model_dump.return_value = {
            "choices": [{
                "message": {
                    "content": "这是一个复杂的推理问题，需要仔细分析...",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
        }
        mock_completion.return_value = mock_response
        
        return client, mock_completion

def test_deepseek_reasoning_model_detection():
    """测试DeepSeek推理模型检测"""
    print("\n=== 测试DeepSeek推理模型检测 ===")
    
    try:
        client, mock_completion = create_mock_client()
        
        reasoning_models = [
            "deepseek-r1",
            "deepseek-r1-lite",
            "deepseek-reasoner"
        ]
        
        for model_name in reasoning_models:
            print(f"测试模型: {model_name}")
            
            # 执行推理模型请求
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": "请解决这个复杂的数学问题：证明费马大定理"}
                ]
            )
            
            # 验证响应
            assert response is not None
            assert response.choices[0].message.content is not None
            assert len(response.choices[0].message.content) > 0
            
            # 验证调用参数（推理模型应该移除不支持的参数）
            call_args = mock_completion.call_args
            call_kwargs = call_args[1] if len(call_args) > 1 else call_args[0]
            
            assert call_kwargs.get('model') == model_name
            
            # 推理模型不应该包含这些参数
            unsupported_params = ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'stream']
            for param in unsupported_params:
                assert param not in call_kwargs, f"推理模型 {model_name} 不应该包含参数 {param}"
            
            print(f"✓ {model_name} 检测正常")
        
        print("✓ DeepSeek推理模型检测测试通过")
        return True
        
    except Exception as e:
        print(f"✗ DeepSeek推理模型检测测试失败: {e}")
        traceback.print_exc()
        return False

def test_non_reasoning_model_detection():
    """测试非推理模型检测"""
    print("\n=== 测试非推理模型检测 ===")
    
    try:
        client, mock_completion = create_mock_client()
        
        regular_models = [
            "deepseek-chat",
            "ernie-3.5-8k",
            "ernie-4.0-turbo-8k",
            "doubao-pro-4k",
            "doubao-pro-32k"
        ]
        
        for model_name in regular_models:
            print(f"测试模型: {model_name}")
            
            # 执行常规模型请求（包含推理模型不支持的参数）
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": "你好"}
                ],
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            # 验证响应
            assert response is not None
            
            # 验证调用参数（常规模型应该保留所有参数）
            call_args = mock_completion.call_args
            call_kwargs = call_args[1] if len(call_args) > 1 else call_args[0]
            
            assert call_kwargs.get('model') == model_name
            assert call_kwargs.get('temperature') == 0.7
            assert call_kwargs.get('top_p') == 0.9
            assert call_kwargs.get('frequency_penalty') == 0.1
            assert call_kwargs.get('presence_penalty') == 0.1
            
            print(f"✓ {model_name} 检测正常")
        
        print("✓ 非推理模型检测测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 非推理模型检测测试失败: {e}")
        traceback.print_exc()
        return False

def test_parameter_filtering_for_reasoning_models():
    """测试推理模型的参数过滤"""
    print("\n=== 测试推理模型参数过滤 ===")
    
    try:
        client, mock_completion = create_mock_client()
        
        # 尝试使用推理模型不支持的参数
        response = client.chat.completions.create(
            model="deepseek-r1",
            messages=[
                {"role": "system", "content": "你是一个AI助手"},  # 推理模型不支持system消息
                {"role": "user", "content": "解决这个问题"}
            ],
            temperature=0.7,        # 推理模型不支持
            top_p=0.9,             # 推理模型不支持
            frequency_penalty=0.1,  # 推理模型不支持
            presence_penalty=0.1,   # 推理模型不支持
            stream=True,           # 推理模型不支持
            max_tokens=1000
        )
        
        # 验证响应
        assert response is not None
        
        # 检查传递给client_manager的参数
        call_args = mock_completion.call_args
        call_kwargs = call_args[1] if len(call_args) > 1 else call_args[0]
        
        # 应该保留的参数
        assert call_kwargs.get('model') == "deepseek-r1"
        assert call_kwargs.get('max_tokens') == 1000
        
        # 应该被过滤的参数
        filtered_params = ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'stream']
        for param in filtered_params:
            assert param not in call_kwargs, f"参数 {param} 应该被过滤"
        
        # 验证system消息处理
        messages = call_kwargs.get('messages', [])
        system_messages = []
        for msg in messages:
            if hasattr(msg, 'role'):
                if msg.role == 'system':
                    system_messages.append(msg)
            elif isinstance(msg, dict) and msg.get('role') == 'system':
                system_messages.append(msg)
        assert len(system_messages) == 0, "推理模型不应该包含system消息"
        
        # 验证system消息内容被合并到user消息中
        user_messages = []
        for msg in messages:
            if hasattr(msg, 'role'):
                if msg.role == 'user':
                    user_messages.append(msg)
            elif isinstance(msg, dict) and msg.get('role') == 'user':
                user_messages.append(msg)
        
        assert len(user_messages) > 0, "应该至少有一个user消息"
        
        # 检查第一个user消息是否包含原system消息的内容
        first_user_msg = user_messages[0]
        if hasattr(first_user_msg, 'content'):
            content = first_user_msg.content
        else:
            content = first_user_msg.get('content', '')
        
        assert "你是一个AI助手" in content, "system消息内容应该被合并到user消息中"
        
        print("✓ 推理模型参数过滤测试通过")
        print(f"✓ 传递给client_manager的参数: {list(call_kwargs.keys())}")
        print(f"✓ 处理后的消息数量: {len(messages)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 推理模型参数过滤测试失败: {e}")
        traceback.print_exc()
        return False

def test_model_capabilities_detection():
    """测试模型能力检测"""
    print("\n=== 测试模型能力检测 ===")
    
    try:
        from harborai.core.models import is_reasoning_model, filter_parameters_for_model
        
        # 测试推理模型检测
        reasoning_models = ["deepseek-r1", "deepseek-r1-lite", "deepseek-reasoner"]
        for model in reasoning_models:
            assert is_reasoning_model(model) == True, f"{model} 应该被识别为推理模型"
            print(f"✓ {model} 正确识别为推理模型")
        
        # 测试非推理模型检测
        regular_models = ["deepseek-chat", "ernie-3.5-8k", "doubao-pro-4k"]
        for model in regular_models:
            assert is_reasoning_model(model) == False, f"{model} 不应该被识别为推理模型"
            print(f"✓ {model} 正确识别为常规模型")
        
        # 测试参数过滤功能
        test_params = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stream": True,
            "max_tokens": 1000
        }
        
        filtered_params = filter_parameters_for_model("deepseek-r1", test_params)
        
        # 检查过滤结果
        assert "model" in filtered_params
        assert "messages" in filtered_params
        assert "max_tokens" in filtered_params
        
        # 这些参数应该被过滤掉
        filtered_out = ["temperature", "top_p", "frequency_penalty", "presence_penalty", "stream"]
        for param in filtered_out:
            assert param not in filtered_params, f"参数 {param} 应该被过滤"
        
        print("✓ 模型能力检测测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 模型能力检测测试失败: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    print("开始运行HarborAI推理模型功能测试...")
    print("=" * 60)
    
    setup_environment()
    
    tests = [
        ("模型能力检测", test_model_capabilities_detection),
        ("DeepSeek推理模型检测", test_deepseek_reasoning_model_detection),
        ("非推理模型检测", test_non_reasoning_model_detection),
        ("推理模型参数过滤", test_parameter_filtering_for_reasoning_models),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} - 通过")
            else:
                failed += 1
                print(f"❌ {test_name} - 失败")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} - 异常: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试总结: 通过 {passed}/{len(tests)}, 失败 {failed}/{len(tests)}")
    
    if failed == 0:
        print("🎉 所有测试通过！")
        return True
    else:
        print(f"❌ 有 {failed} 个测试失败")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)