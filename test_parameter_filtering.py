#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的参数过滤测试脚本
避免pytest的复杂环境问题
"""

import os
import sys
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_parameter_filtering():
    """测试推理模型的参数过滤功能"""
    print("开始测试参数过滤功能...")
    
    # 设置环境变量
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    os.environ["WENXIN_API_KEY"] = "test-key"
    os.environ["DOUBAO_API_KEY"] = "test-key"
    
    try:
        from harborai import HarborAI
        from harborai.core.models import filter_parameters_for_model, is_reasoning_model
        
        print("✓ 成功导入HarborAI模块")
        
        # 测试模型检测
        assert is_reasoning_model("deepseek-r1") == True
        assert is_reasoning_model("deepseek-chat") == False
        print("✓ 推理模型检测功能正常")
        
        # 测试参数过滤
        original_params = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1,
            "stream": True,
            "max_tokens": 1000
        }
        
        filtered_params = filter_parameters_for_model("deepseek-r1", original_params)
        
        # 检查过滤结果
        assert "model" in filtered_params
        assert "messages" in filtered_params
        assert "max_tokens" in filtered_params
        
        # 这些参数应该被过滤掉
        filtered_out = ["temperature", "top_p", "frequency_penalty", "presence_penalty", "stream"]
        for param in filtered_out:
            assert param not in filtered_params, f"参数 {param} 应该被过滤"
        
        print("✓ 参数过滤功能正常")
        
        # 测试HarborAI客户端的参数过滤
        client = HarborAI()
        
        # Mock底层的client_manager调用
        with patch.object(client.client_manager, 'chat_completion_sync_with_fallback') as mock_completion:
            # 创建一个更完整的Mock响应对象
            mock_response = Mock()
            mock_response.choices = [Mock(
                message=Mock(
                    content="测试响应",
                    role="assistant"
                ),
                finish_reason="stop"
            )]
            mock_response.usage = Mock(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30
            )
            # 添加model_dump方法
            mock_response.model_dump.return_value = {
                "choices": [{
                    "message": {
                        "content": "测试响应",
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            }
            mock_completion.return_value = mock_response
            
            # 调用create方法
            response = client.chat.completions.create(
                model="deepseek-r1",
                messages=[
                    {"role": "system", "content": "你是一个AI助手"},
                    {"role": "user", "content": "解决这个问题"}
                ],
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                stream=True,
                max_tokens=1000
            )
            
            # 检查调用参数
            call_args = mock_completion.call_args
            if call_args:
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
                
                print("✓ HarborAI客户端参数过滤功能正常")
                print(f"✓ 传递给client_manager的参数: {list(call_kwargs.keys())}")
                print(f"✓ 处理后的消息数量: {len(messages)}")
                
                return True
            else:
                print("✗ 未能获取client_manager的调用参数")
                return False
                
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parameter_filtering()
    if success:
        print("\n🎉 所有测试通过！")
        sys.exit(0)
    else:
        print("\n❌ 测试失败！")
        sys.exit(1)