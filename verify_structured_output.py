# -*- coding: utf-8 -*-
"""
验证HarborAI和Agently的结构化输出结果
"""

import os
import json
import sys
from dotenv import load_dotenv

# 设置控制台编码
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

load_dotenv()

def test_harborai_structured_output():
    """测试HarborAI结构化输出"""
    print("🚀 测试HarborAI结构化输出")
    print("="*50)
    
    try:
        from harborai import HarborAI
        
        client = HarborAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        
        schema = {
            "type": "object",
            "properties": {
                "analysis": {"type": "string", "description": "详细的情感分析"},
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"], "description": "情感倾向"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "置信度"},
                "keywords": {"type": "array", "items": {"type": "string"}, "description": "关键词"}
            },
            "required": ["analysis", "sentiment", "confidence", "keywords"]
        }
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "请分析以下文本的情感倾向：'今天天气真好，我心情很愉快，工作也很顺利。'"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "schema": schema,
                    "strict": True
                }
            },
            structured_provider="agently",
            temperature=0.1
        )
        
        print(f"✅ HarborAI调用成功")
        print(f"📝 Response类型: {type(response)}")
        print(f"📝 Message类型: {type(response.choices[0].message)}")
        
        # 检查parsed属性
        if hasattr(response.choices[0].message, 'parsed'):
            print(f"✅ 有parsed属性: {response.choices[0].message.parsed is not None}")
            if response.choices[0].message.parsed:
                result = response.choices[0].message.parsed
                print(f"🎯 结构化结果:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return result
        
        # 检查content
        content = response.choices[0].message.content
        print(f"📝 Content: {content}")
        
        if isinstance(content, str):
            try:
                result = json.loads(content)
                print(f"🎯 从Content解析的结果:")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                return result
            except json.JSONDecodeError as e:
                print(f"❌ Content JSON解析失败: {e}")
        
        return None
        
    except Exception as e:
        print(f"❌ HarborAI测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_agently_structured_output():
    """测试Agently结构化输出"""
    print("\n🤖 测试Agently结构化输出")
    print("="*50)
    
    try:
        from Agently.agently import Agently
        
        # 配置Agently
        Agently.set_settings(
            "OpenAICompatible",
            {
                "base_url": os.getenv("DEEPSEEK_BASE_URL"),
                "model": "deepseek-chat",
                "model_type": "chat",
                "auth": os.getenv("DEEPSEEK_API_KEY"),
            },
        )
        
        agent = Agently.create_agent()
        
        agently_output = {
            "analysis": ("str", "详细的情感分析"),
            "sentiment": ("str", "情感倾向: positive/negative/neutral"),
            "confidence": ("float", "置信度(0-1)"),
            "keywords": (["str"], "关键词列表")
        }
        
        result = (
            agent
            .input("请分析以下文本的情感倾向：'今天天气真好，我心情很愉快，工作也很顺利。'")
            .output(agently_output)
            .start()
        )
        
        print(f"✅ Agently调用成功")
        print(f"🎯 结构化结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
        
    except Exception as e:
        print(f"❌ Agently测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_results(harborai_result, agently_result):
    """对比两种结果"""
    print("\n📊 结果对比")
    print("="*50)
    
    if harborai_result and agently_result:
        print("✅ 两种方式都成功获得结构化输出")
        
        # 检查字段一致性
        harborai_keys = set(harborai_result.keys()) if isinstance(harborai_result, dict) else set()
        agently_keys = set(agently_result.keys()) if isinstance(agently_result, dict) else set()
        
        print(f"📋 HarborAI字段: {sorted(harborai_keys)}")
        print(f"📋 Agently字段: {sorted(agently_keys)}")
        
        common_keys = harborai_keys & agently_keys
        print(f"🔗 共同字段: {sorted(common_keys)}")
        
        if harborai_keys == agently_keys:
            print("✅ 字段结构完全一致")
        else:
            print("⚠️ 字段结构存在差异")
            print(f"   HarborAI独有: {sorted(harborai_keys - agently_keys)}")
            print(f"   Agently独有: {sorted(agently_keys - harborai_keys)}")
        
        # 检查数据类型
        for key in common_keys:
            harborai_type = type(harborai_result[key]).__name__
            agently_type = type(agently_result[key]).__name__
            print(f"🔍 {key}: HarborAI({harborai_type}) vs Agently({agently_type})")
            
    elif harborai_result:
        print("⚠️ 只有HarborAI成功获得结构化输出")
    elif agently_result:
        print("⚠️ 只有Agently成功获得结构化输出")
    else:
        print("❌ 两种方式都未能获得结构化输出")

if __name__ == "__main__":
    print("🔍 验证HarborAI和Agently的结构化输出")
    print("="*80)
    
    harborai_result = test_harborai_structured_output()
    agently_result = test_agently_structured_output()
    compare_results(harborai_result, agently_result)
    
    print("\n🏁 验证完成")