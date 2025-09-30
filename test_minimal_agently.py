#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最小Agently结构化输出测试
仅使用deepseek-chat模型进行调试

测试目标：
1. 验证HarborAI的Agently结构化输出功能
2. 确认response.choices[0].message.parsed字段正确设置
3. 测试structured_provider="agently"参数
4. 验证response_format参数的正确传递
"""

import os
import sys
import json
import logging
from pathlib import Path



# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量
from dotenv import load_dotenv

# 确保加载.env文件
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"已加载环境变量文件: {env_path}")
else:
    print(f"❌ 环境变量文件不存在: {env_path}")

def test_minimal_agently_structured_output():
    """最小Agently结构化输出测试"""
    print("开始最小Agently结构化输出测试")
    print("=" * 50)
    
    # 检查必要的环境变量
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL")
    
    if not deepseek_api_key:
        print("❌ 缺少DEEPSEEK_API_KEY环境变量")
        return False
    
    if not deepseek_base_url:
        print("❌ 缺少DEEPSEEK_BASE_URL环境变量")
        return False
    
    print(f"DeepSeek API Key: {deepseek_api_key[:10]}...")
    print(f"DeepSeek Base URL: {deepseek_base_url}")
    
    try:
        # 导入HarborAI客户端
        from harborai import HarborAI
        print("HarborAI客户端导入成功")
        
        # 创建客户端实例
        client = HarborAI()
        print("HarborAI客户端实例创建成功")
        
        # 定义简单的JSON Schema
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "情感倾向分析"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "置信度分数，范围0-1之间的小数，例如0.9表示90%置信度"
                }
            },
            "required": ["sentiment", "confidence"]
        }
        
        print("JSON Schema定义完成")
        print(f"  Schema: {json.dumps(schema, ensure_ascii=False, indent=2)}")
        
        # 创建response_format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_analysis",
                "schema": schema,
                "strict": True
            }
        }
        
        print("response_format创建完成")
        
        # 测试用户输入
        test_message = "请分析这段文本的情感：'今天天气很好，心情不错'"
        
        print(f"📝 测试消息: {test_message}")
        print("🔄 发送请求...")
        
        # 发送请求
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": test_message
                }
            ],
            response_format=response_format,
            structured_provider="agently",  # 明确指定使用Agently
            temperature=0.1,
            max_tokens=500
        )
        
        print("请求发送成功")
        
        # 验证响应
        if response is None:
            print("❌ 响应为空")
            return False
        
        print(f"响应类型: {type(response)}")
        
        # 检查响应结构
        if not hasattr(response, 'choices') or not response.choices:
            print("❌ 响应缺少choices字段")
            return False
        
        choice = response.choices[0]
        message = choice.message
        
        print(f"消息内容: {message.content}")
        
        # 检查parsed字段
        if not hasattr(message, 'parsed'):
            print("❌ 消息缺少parsed字段")
            return False
        
        parsed_data = message.parsed
        
        if parsed_data is None:
            print("❌ parsed字段为空")
            return False
        
        print(f"解析数据: {json.dumps(parsed_data, ensure_ascii=False, indent=2)}")
        
        # 验证必需字段
        if "sentiment" not in parsed_data:
            print("❌ 缺少sentiment字段")
            return False
        
        if "confidence" not in parsed_data:
            print("❌ 缺少confidence字段")
            return False
        
        # 验证字段值
        sentiment = parsed_data["sentiment"]
        confidence = parsed_data["confidence"]
        
        if sentiment not in ["positive", "negative", "neutral"]:
            print(f"❌ sentiment值无效: {sentiment}")
            return False
        
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            print(f"❌ confidence值无效: {confidence}")
            return False
        
        print(f"✅ 测试成功!")
        print(f"   情感: {sentiment}")
        print(f"   置信度: {confidence}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        print("详细错误信息:")
        traceback.print_exc()
        return False

def test_agently_availability():
    """测试Agently库的可用性"""
    print("\n检查Agently库可用性")
    print("-" * 30)
    
    try:
        import agently
        print("Agently库导入成功")
        print(f"  版本: {getattr(agently, '__version__', '未知')}")
        
        # 测试创建Agent
        agent = agently.Agently.create_agent()
        print("Agently Agent创建成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ Agently库导入失败: {e}")
        print("请安装Agently库: pip install agently")
        return False
    except Exception as e:
        print(f"❌ Agently测试失败: {e}")
        return False

def main():
    """主函数"""
    print("HarborAI Agently结构化输出最小测试")
    print("=" * 60)
    
    # 检查Agently可用性
    agently_available = test_agently_availability()
    
    if not agently_available:
        print("\nAgently不可用，无法进行结构化输出测试")
        return
    
    # 执行最小测试
    success = test_minimal_agently_structured_output()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 最小测试通过！Agently结构化输出功能正常")
    else:
        print("🚨 最小测试失败！需要检查和修复问题")
    print("=" * 60)

if __name__ == "__main__":
    main()