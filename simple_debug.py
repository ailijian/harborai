import os
import sys
sys.path.insert(0, '.')

# 设置环境变量
os.environ['HARBORAI_ENV'] = 'test'
os.environ['HARBORAI_LOG_LEVEL'] = 'INFO'
os.environ['HARBORAI_CACHE_ENABLED'] = 'false'
os.environ['OPENAI_API_KEY'] = 'test-key'
os.environ['ANTHROPIC_API_KEY'] = 'test-key'
os.environ['GOOGLE_API_KEY'] = 'test-key'
os.environ['OPEN_ROUTER_API_KEY'] = 'test-key'
os.environ['DEEPSEEK_API_KEY'] = 'test-key'
os.environ['WENXIN_API_KEY'] = 'test-key'
os.environ['DOUBAO_API_KEY'] = 'test-key'

print("开始简单调试...")

try:
    from harborai import HarborAI
    from unittest.mock import Mock, patch
    
    print("创建HarborAI客户端...")
    client = HarborAI()
    
    print("设置mock...")
    with patch.object(client.client_manager, 'chat_completion_sync_with_fallback') as mock_completion:
        # 配置mock响应
        mock_response = Mock()
        mock_response.id = "chatcmpl-test-123"  # 确保id是字符串
        mock_response.object = "chat.completion"
        mock_response.created = 1234567890
        mock_response.model = "gpt-4"
        mock_response.choices = [Mock(
            index=0,
            message=Mock(
                content="测试响应",
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
        
        print("调用chat.completions.create...")
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
            
        # 测试其他属性
        for attr in ['object', 'created', 'model', 'choices', 'usage']:
            if hasattr(response, attr):
                value = getattr(response, attr)
                print(f"response.{attr} 类型: {type(value)}, 值: {value}")
            else:
                print(f"response 没有 {attr} 属性")
                
except Exception as e:
    import traceback
    print(f"错误: {e}")
    traceback.print_exc()