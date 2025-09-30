#!/usr/bin/env python3
"""
HarborAI 结构化输出完整测试脚本
测试所有结构化输出方法：JSON Schema、Pydantic、自定义格式
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

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

def test_json_schema_structured_output():
    """测试 JSON Schema 结构化输出"""
    print("\n" + "="*60)
    print("🧪 测试 JSON Schema 结构化输出")
    print("="*60)
    
    try:
        # 初始化 HarborAI
        client = HarborAI()
        
        # 定义 JSON Schema
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
                    "description": "置信度分数，范围0-1之间的小数"
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词列表"
                }
            },
            "required": ["sentiment", "confidence", "keywords"]
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
            {"role": "user", "content": "分析这段文本的情感：'今天阳光明媚，我和朋友们一起去公园野餐，度过了愉快的一天。'"}
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
            if all(key in parsed_data for key in ["sentiment", "confidence", "keywords"]):
                print("✅ 所有必需字段都存在")
                print(f"   情感: {parsed_data['sentiment']}")
                print(f"   置信度: {parsed_data['confidence']}")
                print(f"   关键词: {parsed_data['keywords']}")
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

def test_pydantic_structured_output():
    """测试 Pydantic 结构化输出"""
    print("\n" + "="*60)
    print("🧪 测试 Pydantic 结构化输出")
    print("="*60)
    
    try:
        from pydantic import BaseModel, Field
        from typing import List, Literal
        
        # 定义 Pydantic 模型
        class SentimentAnalysis(BaseModel):
            sentiment: Literal["positive", "negative", "neutral"] = Field(description="情感倾向")
            confidence: float = Field(ge=0, le=1, description="置信度分数，范围0-1")
            keywords: List[str] = Field(description="关键词列表")
            summary: str = Field(description="分析摘要")
        
        # 初始化 HarborAI
        client = HarborAI()
        
        print(f"📋 Pydantic 模型: {SentimentAnalysis.__name__}")
        print(f"📋 模型字段: {list(SentimentAnalysis.model_fields.keys())}")
        
        # 发送请求
        messages = [
            {"role": "user", "content": "分析这段文本的情感：'工作压力很大，最近总是加班到很晚，感觉身心疲惫。'"}
        ]
        
        print("📤 发送请求...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format=SentimentAnalysis,
            temperature=0.1
        )
        
        print("✅ 请求成功")
        print(f"📥 响应类型: {type(response)}")
        
        # 解析响应
        content = response.choices[0].message.content
        print(f"📄 原始内容: {content}")
        
        # 尝试解析为 Pydantic 模型
        try:
            parsed_data = json.loads(content)
            model_instance = SentimentAnalysis(**parsed_data)
            print(f"✅ Pydantic 解析成功:")
            print(f"   情感: {model_instance.sentiment}")
            print(f"   置信度: {model_instance.confidence}")
            print(f"   关键词: {model_instance.keywords}")
            print(f"   摘要: {model_instance.summary}")
            return True
            
        except Exception as e:
            print(f"❌ Pydantic 解析失败: {e}")
            return False
            
    except ImportError:
        print("⚠️ Pydantic 未安装，跳过测试")
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_format_structured_output():
    """测试自定义格式结构化输出"""
    print("\n" + "="*60)
    print("🧪 测试自定义格式结构化输出")
    print("="*60)
    
    try:
        # 初始化 HarborAI
        client = HarborAI()
        
        # 自定义格式
        custom_format = {
            "type": "text",
            "format": "请按照以下格式输出：\n情感：[positive/negative/neutral]\n置信度：[0-1之间的数字]\n理由：[分析理由]"
        }
        
        print(f"📋 自定义格式: {custom_format['format']}")
        
        # 发送请求
        messages = [
            {"role": "user", "content": "分析这段文本的情感：'虽然遇到了一些困难，但我相信通过努力一定能够克服。'"}
        ]
        
        print("📤 发送请求...")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format=custom_format,
            temperature=0.1
        )
        
        print("✅ 请求成功")
        print(f"📥 响应类型: {type(response)}")
        
        # 解析响应
        content = response.choices[0].message.content
        print(f"📄 响应内容:\n{content}")
        
        # 简单验证格式
        if "情感：" in content and "置信度：" in content and "理由：" in content:
            print("✅ 自定义格式输出正确")
            return True
        else:
            print("❌ 自定义格式输出不符合要求")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_agently_direct_call():
    """测试直接 Agently 调用"""
    print("\n" + "="*60)
    print("🧪 测试直接 Agently 调用")
    print("="*60)
    
    try:
        # 导入 structured 模块
        from harborai.api.structured import create_structured_completion
        
        # 定义输出格式
        outputs = {
            "sentiment": ("String", "情感倾向：positive/negative/neutral"),
            "confidence": ("Number", "置信度分数，0-1之间的小数"),
            "analysis": ("String", "详细分析")
        }
        
        print(f"📋 输出格式: {outputs}")
        
        # 测试输入
        user_input = "今天收到了心仪公司的面试邀请，既兴奋又紧张。"
        
        print(f"📤 输入文本: {user_input}")
        print("📤 发送请求...")
        
        # 调用结构化输出
        result = create_structured_completion(
            model="deepseek-chat",
            user_input=user_input,
            outputs=outputs,
            system_prompt="你是一个专业的情感分析师，请分析用户输入文本的情感倾向。"
        )
        
        print("✅ 请求成功")
        print(f"📥 结果类型: {type(result)}")
        print(f"📄 结果内容: {result}")
        
        # 验证结果
        if result and isinstance(result, dict):
            if all(key in result for key in ["sentiment", "confidence", "analysis"]):
                print("✅ 所有字段都存在")
                print(f"   情感: {result['sentiment']}")
                print(f"   置信度: {result['confidence']}")
                print(f"   分析: {result['analysis']}")
                return True
            else:
                print("❌ 缺少必需字段")
                return False
        else:
            print("❌ 结果格式不正确")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始 HarborAI 结构化输出完整测试")
    print("="*80)
    
    # 检查环境变量
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("❌ 缺少 DEEPSEEK_API_KEY 环境变量")
        return False
    
    print(f"✅ 环境变量检查通过")
    
    # 运行所有测试
    tests = [
        ("JSON Schema 结构化输出", test_json_schema_structured_output),
        ("Pydantic 结构化输出", test_pydantic_structured_output),
        ("自定义格式结构化输出", test_custom_format_structured_output),
        ("直接 Agently 调用", test_agently_direct_call),
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