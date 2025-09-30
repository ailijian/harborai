#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端测试用例007：Agently默认结构化输出功能测试

基于HarborAI端到端测试方案.md第301-364行的内容，验证Agently默认结构化输出功能。
该测试用例专注于验证单个特定功能，确保测试用例简洁明确且可独立执行。

测试目标：
1. 验证所有7个模型的结构化输出能力
2. 测试response.parsed字段和数据结构正确性
3. 确保Agently解析器正常工作
4. 验证schema定义和模型循环测试
"""

import os
import sys
import json
import time
import pytest
from typing import Dict, Any, List

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
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
from harborai.utils.logger import get_logger

logger = get_logger(__name__)

@pytest.fixture(scope='session')
def client() -> HarborAI:
    """创建HarborAI客户端实例"""
    return HarborAI()

def create_test_schema() -> Dict[str, Any]:
    """创建测试用的JSON Schema定义。
    
    根据测试方案，定义一个包含多种数据类型的结构化输出schema。
    """
    return {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "对输入内容的分析结果"
            },
            "sentiment": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
                "description": "情感倾向分析"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "置信度分数"
            },
            "keywords": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "关键词列表"
            },
            "metadata": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "description": "检测到的语言"
                    },
                    "word_count": {
                        "type": "integer",
                        "description": "词汇数量"
                    }
                },
                "required": ["language", "word_count"]
            }
        },
        "required": ["analysis", "sentiment", "confidence", "keywords", "metadata"]
    }

def get_test_models() -> List[Dict[str, str]]:
    """获取要测试的模型列表。
    
    根据.env文件中配置的三个模型厂商，返回所有7个模型的配置信息。
    """
    return [
        # DeepSeek模型
        {"model": "deepseek-chat", "provider": "deepseek", "name": "DeepSeek Chat"},
        {"model": "deepseek-reasoner", "provider": "deepseek", "name": "DeepSeek Reasoner"},
        
        # 文心一言模型
        {"model": "ernie-3.5-8k", "provider": "ernie", "name": "文心一言 3.5"},
        {"model": "ernie-4.0-turbo-8k", "provider": "ernie", "name": "文心一言 4.0 Turbo"},
        {"model": "ernie-x1-turbo-32k", "provider": "ernie", "name": "文心一言 X1 Turbo"},
        
        # 豆包模型
        {"model": "doubao-1-5-pro-32k-character-250715", "provider": "doubao", "name": "豆包 1.5 Pro 32K"},
        {"model": "doubao-seed-1-6-250615", "provider": "doubao", "name": "豆包 Seed 1.6"}
    ]

@pytest.mark.parametrize("model_config", get_test_models())
def test_structured_output_with_model(client: HarborAI, model_config: Dict[str, str]):
    """测试单个模型的结构化输出功能。
    
    Args:
        client: HarborAI客户端实例
        model_config: 模型配置信息
    """
    schema = create_test_schema()
    model = model_config["model"]
    provider = model_config["provider"]
    name = model_config["name"]
    
    print(f"\n🧪 测试模型: {name} ({model})")
    
    try:
        # 创建response_format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "analysis_result",
                "schema": schema,
                "strict": True
            }
        }
        
        # 发送测试请求
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "请分析这段文本：'今天天气真不错，阳光明媚，心情很好！我决定去公园散步，享受这美好的一天。'"
                }
            ],
            response_format=response_format,
            structured_provider="agently",  # 使用Agently作为结构化输出提供者
            temperature=0.1,
            max_tokens=1000
        )
        end_time = time.time()
        
        # 验证测试结果
        assert response is not None, f"模型 {model} 未返回响应"
        
        choice = response.choices[0]
        message = choice.message
        parsed_data = message.parsed if hasattr(message, 'parsed') else None
        
        assert parsed_data is not None, f"模型 {model} 结构化输出解析失败"
        assert "analysis" in parsed_data, f"模型 {model} 输出缺少analysis字段"
        assert "sentiment" in parsed_data, f"模型 {model} 输出缺少sentiment字段"
        assert "confidence" in parsed_data, f"模型 {model} 输出缺少confidence字段"
        assert "keywords" in parsed_data, f"模型 {model} 输出缺少keywords字段"
        assert "metadata" in parsed_data, f"模型 {model} 输出缺少metadata字段"
        
        execution_time = end_time - start_time
        print(f"✓ 模型 {model} 测试通过，执行时间: {execution_time:.2f}秒")
        print(f"  解析结果: {parsed_data}")
        
        # 测试成功完成
        print(f"  ✅ 测试成功 - 延迟: {round((end_time - start_time) * 1000, 2)}ms")
            
    except Exception as e:
        print(f"  ❌ 异常 - {str(e)}")
        raise AssertionError(f"模型 {model} 测试失败: {str(e)}") from e

def main():
    """主测试函数。"""
    print("🚀 开始Agently默认结构化输出功能测试")
    print("=" * 60)
    
    # 检查环境变量
    required_env_vars = [
        "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL",
        "DOUBAO_API_KEY", "DOUBAO_BASE_URL", 
        "WENXIN_API_KEY"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ 缺少环境变量: {missing_vars}")
        return
    
    print("✅ 环境变量检查通过")
    
    # 初始化HarborAI客户端
    try:
        client = HarborAI()
        print("✅ HarborAI客户端初始化成功")
    except Exception as e:
        print(f"❌ HarborAI客户端初始化失败: {e}")
        return
    
    # 创建测试schema
    schema = create_test_schema()
    print(f"✅ 测试Schema创建成功，包含{len(schema['properties'])}个字段")
    
    # 获取测试模型列表
    models = get_test_models()
    print(f"✅ 准备测试{len(models)}个模型")
    
    # 执行测试
    results = []
    successful_tests = 0
    
    for model_config in models:
        try:
            test_structured_output_with_model(client, model_config)
            result = {
                "model": model_config["model"],
                "provider": model_config["provider"],
                "name": model_config["name"],
                "success": True,
                "error": None,
                "latency_ms": 0  # 这里需要在实际测试中记录
            }
            successful_tests += 1
        except Exception as e:
            result = {
                "model": model_config["model"],
                "provider": model_config["provider"],
                "name": model_config["name"],
                "success": False,
                "error": str(e),
                "latency_ms": 0
            }
        results.append(result)
    
    # 生成测试报告
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    success_rate = (successful_tests / len(models)) * 100
    print(f"总测试数: {len(models)}")
    print(f"成功数: {successful_tests}")
    print(f"失败数: {len(models) - successful_tests}")
    print(f"成功率: {success_rate:.1f}%")
    
    # 按提供商分组显示结果
    providers = {}
    for result in results:
        provider = result["provider"]
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(result)
    
    for provider, provider_results in providers.items():
        print(f"\n📋 {provider.upper()} 提供商结果:")
        for result in provider_results:
            status = "✅" if result["success"] else "❌"
            print(f"  {status} {result['name']} ({result['model']})")
            if not result["success"]:
                print(f"     错误: {result['error']}")
            else:
                print(f"     延迟: {result['latency_ms']}ms")
    
    # 保存详细结果到JSON文件
    output_file = os.path.join(os.path.dirname(__file__), "agently_structured_output_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "test_name": "Agently默认结构化输出功能测试",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_models": len(models),
            "successful_tests": successful_tests,
            "success_rate": success_rate,
            "schema": schema,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    
    # 测试结论
    if success_rate >= 80:
        print("\n🎉 测试结论: Agently结构化输出功能整体表现良好")
    elif success_rate >= 50:
        print("\n⚠️  测试结论: Agently结构化输出功能部分正常，需要进一步优化")
    else:
        print("\n🚨 测试结论: Agently结构化输出功能存在严重问题，需要紧急修复")

if __name__ == "__main__":
    main()