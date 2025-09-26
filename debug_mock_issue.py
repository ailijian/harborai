#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试Mock.keys()错误的脚本
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from unittest.mock import Mock, patch
from harborai import HarborAI

def debug_mock_issue():
    """调试Mock问题"""
    print("开始调试Mock问题...")
    
    # 设置环境变量
    os.environ["DEEPSEEK_API_KEY"] = "test-key"
    os.environ["WENXIN_API_KEY"] = "test-key"
    os.environ["DOUBAO_API_KEY"] = "test-key"
    
    try:
        # 创建HarborAI客户端
        client = HarborAI()
        print("✓ HarborAI客户端创建成功")
        
        # Mock client_manager的方法
        with patch.object(client.client_manager, 'chat_completion_sync_with_fallback') as mock_completion:
            # 配置mock响应
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
            mock_completion.return_value = mock_response
            
            print("✓ Mock配置完成")
            
            # 尝试调用create方法
            try:
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
                    stream=False,
                    max_tokens=1000
                )
                print("✓ create方法调用成功")
                print(f"响应: {response}")
                
                # 检查调用参数
                if mock_completion.call_args:
                    print("✓ mock_completion被调用")
                    call_kwargs = mock_completion.call_args.kwargs
                    print(f"调用参数类型: {type(call_kwargs)}")
                    print(f"调用参数: {call_kwargs}")
                    
                    # 检查参数过滤
                    filtered_params = ['temperature', 'top_p', 'frequency_penalty', 'presence_penalty', 'stream']
                    for param in filtered_params:
                        if param in call_kwargs:
                            print(f"⚠️  参数 {param} 未被过滤")
                        else:
                            print(f"✓ 参数 {param} 已被过滤")
                else:
                    print("❌ mock_completion未被调用")
                    
            except Exception as e:
                print(f"❌ create方法调用失败: {e}")
                import traceback
                traceback.print_exc()
                
    except Exception as e:
        print(f"❌ 调试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_mock_issue()