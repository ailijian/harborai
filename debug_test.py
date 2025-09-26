import os
import sys
sys.path.insert(0, '.')

# 设置环境变量
os.environ['HARBORAI_ENV'] = 'test'
os.environ['HARBORAI_LOG_LEVEL'] = 'DEBUG'
os.environ['HARBORAI_CACHE_ENABLED'] = 'false'
os.environ['OPENAI_API_KEY'] = 'test-key'
os.environ['ANTHROPIC_API_KEY'] = 'test-key'
os.environ['GOOGLE_API_KEY'] = 'test-key'
os.environ['OPEN_ROUTER_API_KEY'] = 'test-key'

import traceback

try:
    from tests.functional.test_n_standard_alignment import TestStandardAlignment
    from harborai.core.base_plugin import ChatMessage
    from unittest.mock import Mock
    
    # 创建测试实例
    test_instance = TestStandardAlignment()
    
    # 模拟monkeypatch
    class MockMonkeypatch:
        def setenv(self, key, value):
            os.environ[key] = value
    
    mock_monkeypatch = MockMonkeypatch()
    
    # 直接初始化测试实例的属性（复制setup_method的逻辑）
    test_instance.test_messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "请用一句话解释量子纠缠现象。"}
    ]
    
    # 模拟OpenAI响应结构
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
    
    # 模拟流式响应
    test_instance.mock_stream_chunks = [
        {
            "id": "chatcmpl-test-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None
            }]
        },
        {
            "id": "chatcmpl-test-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": "量子纠缠"},
                "finish_reason": None
            }]
        },
        {
            "id": "chatcmpl-test-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {"content": "是指两个或多个粒子之间存在的一种量子力学关联。"},
                "finish_reason": None
            }]
        },
        {
            "id": "chatcmpl-test-stream",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
    ]
    
    # 创建mock HarborAI客户端（模拟conftest.py中的fixture）
    from unittest.mock import Mock, patch
    from harborai import HarborAI
    
    # 设置测试环境变量
    mock_monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    mock_monkeypatch.setenv("WENXIN_API_KEY", "test-key")
    mock_monkeypatch.setenv("DOUBAO_API_KEY", "test-key")
    
    # 创建真实的HarborAI客户端
    client = HarborAI()
    
    # Mock底层的HTTP请求
    with patch.object(client.client_manager, 'chat_completion_sync_with_fallback') as mock_completion:
        # 配置mock响应
        mock_response = Mock()
        mock_response.choices = [Mock(
            message=Mock(
                content="量子纠缠是量子力学中的一个重要现象，指两个或多个粒子之间存在一种特殊的量子力学关联...",
                role="assistant"
            ),
            finish_reason="stop"
        )]
        mock_response.usage = Mock(
            prompt_tokens=25,
            completion_tokens=50,
            total_tokens=75
        )
        mock_completion.return_value = mock_response
        
        # 保存原始的create方法
        original_create = client.chat.completions.create
        
        def create_wrapper(*args, **kwargs):
            result = original_create(*args, **kwargs)
            create_wrapper.call_args = (args, kwargs)
            return result
        
        client.chat.completions.create = create_wrapper
        
        print("开始运行测试...")
        
        # 先测试基本调用
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": "Hello, world!"}
                ]
            )
            print(f"响应类型: {type(response)}")
            print(f"响应对象: {response}")
            print(f"响应对象属性: {dir(response)}")
            if hasattr(response, 'id'):
                print(f"response.id 类型: {type(response.id)}")
                print(f"response.id 值: {response.id}")
            else:
                print("response 没有 id 属性")
        except Exception as e:
            print(f"基本调用失败: {e}")
            traceback.print_exc()
        
        # 然后运行测试
        try:
            test_instance.test_n001_openai_sdk_replacement_sync(client)
            print("测试成功完成！")
        except Exception as e:
            print(f"测试失败: {e}")
            traceback.print_exc()
    
except Exception as e:
    print(f"测试失败: {e}")
    print("\n完整错误堆栈:")
    traceback.print_exc()