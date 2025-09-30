#!/usr/bin/env python3
"""
HarborAI 简化结构化输出测试
专注于测试基本的结构化输出功能
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 配置日志
logging.basicConfig(level=logging.INFO)

# 加载环境变量
from dotenv import load_dotenv

env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"已加载环境变量文件: {env_path}")
else:
    print(f"❌ 环境变量文件不存在: {env_path}")

# 导入 HarborAI
from harborai import HarborAI

def test_basic_json_schema():
    """测试基本的 JSON Schema 结构化输出"""
    print("\n" + "="*60)
    print("🧪 测试基本 JSON Schema 结构化输出")
    print("="*60)
    
    try:
        # 初始化 HarborAI
        client = HarborAI()
        
        # 简单的 JSON Schema
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "required": ["sentiment", "confidence"]
        }
        
        # 创建 response_format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_analysis",
                "schema": schema,
                "strict": True
            }
        }
        
        print(f"📋 Schema: {json.dumps(schema, ensure_ascii=False, indent=2)}")
        
        # 发送请求
        messages = [
            {"role": "user", "content": "请分析这段文本的情感并返回JSON格式：'今天天气真好，我很开心！'"}
        ]
        
        print("📤 发送请求...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format=response_format,
            temperature=0.1
        )
        
        print("✅ 请求成功")
        print(f"📥 响应类型: {type(response)}")
        
        # 解析响应
        content = response.choices[0].message.content
        print(f"📄 原始内容: {content}")
        
        # 尝试解析 JSON
        try:
            parsed_data = json.loads(content)
            print(f"✅ JSON 解析成功: {json.dumps(parsed_data, ensure_ascii=False, indent=2)}")
            
            # 验证数据
            if "sentiment" in parsed_data and "confidence" in parsed_data:
                print("✅ 所有必需字段都存在")
                print(f"   情感: {parsed_data['sentiment']}")
                print(f"   置信度: {parsed_data['confidence']}")
                return True
            else:
                print("❌ 缺少必需字段")
                return False
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_structured_prompt():
    """测试简单的结构化提示"""
    print("\n" + "="*60)
    print("🧪 测试简单结构化提示")
    print("="*60)
    
    try:
        # 初始化 HarborAI
        client = HarborAI()
        
        # 发送带结构化要求的请求
        messages = [
            {
                "role": "system", 
                "content": "你是一个情感分析助手。请严格按照JSON格式返回结果，包含sentiment（positive/negative/neutral）和confidence（0-1之间的数字）字段。"
            },
            {
                "role": "user", 
                "content": "分析这段文本的情感：'虽然遇到困难，但我相信能够克服。'"
            }
        ]
        
        print("📤 发送请求...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.1
        )
        
        print("✅ 请求成功")
        print(f"📥 响应类型: {type(response)}")
        
        # 解析响应
        content = response.choices[0].message.content
        print(f"📄 响应内容:\n{content}")
        
        # 尝试从响应中提取JSON
        import re
        json_pattern = r'\{[^{}]*\}'
        json_matches = re.findall(json_pattern, content)
        
        if json_matches:
            for i, json_str in enumerate(json_matches):
                try:
                    parsed_data = json.loads(json_str)
                    print(f"✅ 找到有效JSON ({i+1}): {json.dumps(parsed_data, ensure_ascii=False, indent=2)}")
                    
                    if "sentiment" in parsed_data and "confidence" in parsed_data:
                        print("✅ JSON包含必需字段")
                        return True
                except json.JSONDecodeError:
                    continue
        
        print("❌ 未找到有效的JSON格式")
        return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_structured_output_handler():
    """测试结构化输出处理器"""
    print("\n" + "="*60)
    print("🧪 测试结构化输出处理器")
    print("="*60)
    
    try:
        # 导入结构化输出处理器
        from harborai.api.structured import StructuredOutputHandler
        
        # 创建处理器
        handler = StructuredOutputHandler(provider="agently")
        
        print(f"✅ 结构化输出处理器创建成功")
        print(f"📋 提供者: {handler.provider}")
        print(f"📋 Agently可用: {handler._agently_available}")
        
        # 测试schema转换
        schema = {
            "type": "object",
            "properties": {
                "sentiment": {"type": "string"},
                "confidence": {"type": "number"}
            }
        }
        
        # 测试response_format创建
        from harborai.api.structured import create_response_format
        response_format = create_response_format(schema, "test_schema")
        
        print(f"✅ response_format创建成功: {json.dumps(response_format, ensure_ascii=False, indent=2)}")
        
        return True
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始 HarborAI 简化结构化输出测试")
    print("="*80)
    
    # 检查环境变量
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("❌ 缺少 DEEPSEEK_API_KEY 环境变量")
        return False
    
    print(f"✅ 环境变量检查通过")
    
    # 运行测试
    tests = [
        ("基本 JSON Schema", test_basic_json_schema),
        ("简单结构化提示", test_simple_structured_prompt),
        ("结构化输出处理器", test_structured_output_handler),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🧪 开始测试: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                print(f"✅ {test_name} 测试通过")
            else:
                print(f"❌ {test_name} 测试失败")
        except Exception as e:
            print(f"❌ {test_name} 测试异常: {e}")
            results[test_name] = False
    
    # 总结结果
    print("\n" + "="*80)
    print("📊 测试结果总结")
    print("="*80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n📈 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试都通过！HarborAI 结构化输出功能正常")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步调试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)