#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小结构化输出测试
仅使用deepseek-chat模型进行调试，验证HarborAI的结构化输出功能
"""

import os
import sys
import json
import traceback
import logging

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# 加载环境变量
try:
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"✓ 已加载环境变量文件: {env_path}")
    else:
        print(f"⚠ 环境变量文件不存在: {env_path}")
except ImportError:
    print("⚠ python-dotenv未安装，直接使用环境变量")

from harborai import HarborAI

def test_minimal_structured_output():
    """
    最小结构化输出测试
    """
    print("🚀 开始最小结构化输出测试")
    print("=" * 60)
    
    # 启用调试日志
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 设置HarborAI相关的日志级别
    logging.getLogger('harborai').setLevel(logging.DEBUG)
    logging.getLogger('harborai.api.structured').setLevel(logging.DEBUG)
    logging.getLogger('harborai.core.plugins').setLevel(logging.DEBUG)
    logging.getLogger('harborai.core.plugins.deepseek_plugin').setLevel(logging.DEBUG)
    logging.getLogger('harborai.core.base_plugin').setLevel(logging.DEBUG)
    
    # 检查环境变量
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")
    
    if not deepseek_api_key or not deepseek_base_url:
        print(f"❌ 缺少DeepSeek环境变量:")
        print(f"   DEEPSEEK_API_KEY: {'✓' if deepseek_api_key else '❌'}")
        print(f"   DEEPSEEK_BASE_URL: {'✓' if deepseek_base_url else '❌'}")
        return False
    
    print("✅ 环境变量检查通过")
    print(f"   DEEPSEEK_API_KEY: {deepseek_api_key[:10]}...")
    print(f"   DEEPSEEK_BASE_URL: {deepseek_base_url}")
    
    # 创建简单的JSON schema
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "description": "情感分析结果"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "置信度，0-1之间的数值"
            }
        },
        "required": ["sentiment", "confidence"]
    }
    
    print(f"\n📋 测试Schema: {json.dumps(schema, ensure_ascii=False, indent=2)}")
    
    # 测试文本
    test_text = "今天天气很好，心情不错"
    print(f"\n📝 测试文本: {test_text}")
    
    try:
        # 初始化HarborAI客户端
        print("\n🔧 初始化HarborAI客户端...")
        client = HarborAI()
        print("✅ HarborAI客户端初始化成功")
        
        # 创建response_format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_analysis",
                "schema": schema,
                "strict": True
            }
        }
        
        print(f"\n📤 发送请求...")
        print(f"   模型: deepseek-chat")
        print(f"   structured_provider: agently (默认)")
        print(f"   response_format: {json.dumps(response_format, ensure_ascii=False, indent=2)}")
        
        # 发送请求
        print(f"\n🚀 发送聊天完成请求...")
        print(f"   请求参数:")
        print(f"   - model: deepseek-chat")
        print(f"   - messages: [{{\"role\": \"user\", \"content\": \"{test_text}\"}}]")
        print(f"   - response_format: {response_format}")
        print(f"   - structured_provider: agently")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": f"请分析以下文本的情感: {test_text}"
                }
            ],
            response_format=response_format,
            structured_provider="agently",  # 明确指定使用agently
            temperature=0.1,
            max_tokens=500
        )
        
        print("\n📥 收到响应:")
        print(f"   响应类型: {type(response)}")
        print(f"   响应对象: {response}")
        
        # 验证响应结构
        if not response:
            print("❌ 响应为空")
            return False
            
        if not hasattr(response, 'choices') or not response.choices:
            print("❌ 响应缺少choices字段")
            return False
            
        choice = response.choices[0]
        message = choice.message
        
        print(f"\n🔍 分析响应结构:")
        print(f"   choice类型: {type(choice)}")
        print(f"   message类型: {type(message)}")
        print(f"   message属性: {dir(message)}")
        
        # 检查原始内容
        if hasattr(message, 'content'):
            print(f"   原始内容: {message.content}")
        
        # 检查结构化输出
        if hasattr(message, 'parsed'):
            parsed_data = message.parsed
            print(f"   结构化输出: {parsed_data}")
            print(f"   结构化输出类型: {type(parsed_data)}")
            
            if parsed_data is None:
                print("❌ 结构化输出为None")
                return False
                
            # 验证字段
            if isinstance(parsed_data, dict):
                if "sentiment" in parsed_data and "confidence" in parsed_data:
                    print(f"✅ 结构化输出验证成功:")
                    print(f"   sentiment: {parsed_data['sentiment']}")
                    print(f"   confidence: {parsed_data['confidence']}")
                    return True
                else:
                    print(f"❌ 结构化输出缺少必需字段: {parsed_data}")
                    return False
            else:
                print(f"❌ 结构化输出不是字典格式: {parsed_data}")
                return False
        else:
            print("❌ message对象缺少parsed属性")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_minimal_structured_output()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 最小结构化输出测试成功!")
    else:
        print("🚨 最小结构化输出测试失败!")
    print("=" * 60)
    
    return success

if __name__ == "__main__":
    main()