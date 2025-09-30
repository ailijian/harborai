#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 HarborAI + Agently 性能对比测试

本测试用于验证和对比：
1. HarborAI + Agently 结构化输出的性能
2. 直接使用 Agently 结构化输出的性能

目标：确认 HarborAI 是否直接将用户输入传递给 Agently，而不是先生成后解析
"""

import os
import time
import json
import statistics
from typing import Dict, Any, List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def setup_console_encoding():
    """设置控制台编码为UTF-8（Windows兼容）"""
    import sys
    if sys.platform.startswith('win'):
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except:
            pass

setup_console_encoding()

def get_test_schema() -> Dict[str, Any]:
    """获取测试用的JSON Schema"""
    return {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "对人工智能技术发展趋势的详细分析"
            },
            "trends": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "主要发展趋势列表",
                "minItems": 3,
                "maxItems": 8
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "分析结果的置信度（0-1之间）"
            },
            "keywords": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "关键词列表",
                "minItems": 3,
                "maxItems": 10
            }
        },
        "required": ["analysis", "trends", "confidence", "keywords"],
        "additionalProperties": False
    }

def test_harborai_agently_structured_output(user_input: str, schema: Dict[str, Any]) -> tuple:
    """测试 HarborAI + Agently 结构化输出"""
    print("[INFO] 开始测试 HarborAI + Agently 结构化输出...")
    
    try:
        from harborai import HarborAI
        
        # 创建 HarborAI 客户端
        client = HarborAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        
        # 构建 response_format
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "ai_trend_analysis",
                "schema": schema,
                "strict": True
            }
        }
        
        print(f"[DEBUG] 使用模型: deepseek-chat")
        print(f"[DEBUG] 用户输入: {user_input}")
        print(f"[DEBUG] Schema: {json.dumps(schema, ensure_ascii=False, indent=2)}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 调用 HarborAI 结构化输出
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": user_input}
            ],
            response_format=response_format,
            structured_provider="agently",  # 明确指定使用 Agently
            temperature=0.1
        )
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"[SUCCESS] HarborAI + Agently 调用成功，耗时: {duration:.3f}秒")
        
        # 获取结构化结果
        if hasattr(response.choices[0].message, 'parsed') and response.choices[0].message.parsed:
            result = response.choices[0].message.parsed
            print(f"[DEBUG] 结构化结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
            return duration, result, None
        else:
            error_msg = "未获得结构化输出结果"
            print(f"[ERROR] {error_msg}")
            return duration, None, error_msg
            
    except Exception as e:
        print(f"[ERROR] HarborAI + Agently 测试失败: {e}")
        return 0, None, str(e)

def test_direct_agently_structured_output(user_input: str, schema: Dict[str, Any]) -> tuple:
    """测试直接使用 Agently 结构化输出"""
    print("[INFO] 开始测试直接 Agently 结构化输出...")
    
    try:
        from Agently.agently import Agently
        
        # 配置 Agently
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        model = "deepseek-chat"
        
        print(f"[DEBUG] 配置 Agently: base_url={base_url}, model={model}")
        print(f"[DEBUG] API Key: {api_key[:10] if api_key else None}...")
        
        # 使用 OpenAICompatible 全局配置
        Agently.set_settings(
            "OpenAICompatible",
            {
                "base_url": base_url,
                "model": model,
                "model_type": "chat",
                "auth": api_key,
            },
        )
        
        print("[DEBUG] Agently 全局配置完成")
        
        # 创建 agent
        agent = Agently.create_agent()
        
        # 将 JSON Schema 转换为 Agently output 格式
        agently_output = convert_json_schema_to_agently_output(schema)
        print(f"[DEBUG] Agently output 格式: {agently_output}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 调用 Agently 结构化输出
        result = (
            agent
            .input(user_input)
            .output(agently_output)
            .start()
        )
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"[SUCCESS] 直接 Agently 调用成功，耗时: {duration:.3f}秒")
        print(f"[DEBUG] 结构化结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        return duration, result, None
        
    except Exception as e:
        print(f"[ERROR] 直接 Agently 测试失败: {e}")
        return 0, None, str(e)

def convert_json_schema_to_agently_output(schema: Dict[str, Any]) -> Dict[str, Any]:
    """将 JSON Schema 转换为 Agently output 格式"""
    agently_output = {}
    
    if "properties" in schema:
        for prop_name, prop_def in schema["properties"].items():
            prop_type = prop_def.get("type", "string")
            description = prop_def.get("description", "")
            
            if prop_type == "string":
                agently_output[prop_name] = ("str", description)
            elif prop_type == "number":
                agently_output[prop_name] = ("float", description)
            elif prop_type == "integer":
                agently_output[prop_name] = ("int", description)
            elif prop_type == "boolean":
                agently_output[prop_name] = ("bool", description)
            elif prop_type == "array":
                items_type = prop_def.get("items", {}).get("type", "string")
                if items_type == "string":
                    agently_output[prop_name] = ([("str", "")], description)
                elif items_type == "number":
                    agently_output[prop_name] = ([("float", "")], description)
                elif items_type == "integer":
                    agently_output[prop_name] = ([("int", "")], description)
                else:
                    agently_output[prop_name] = ([("str", "")], description)
            else:
                agently_output[prop_name] = ("str", description)
    
    return agently_output

def run_performance_comparison(iterations: int = 3) -> None:
    """运行性能对比测试"""
    print("="*80)
    print("🚀 HarborAI + Agently 性能对比测试")
    print("="*80)
    
    # 测试参数
    user_input = "请分析人工智能技术的发展趋势"
    schema = get_test_schema()
    
    print(f"[CONFIG] 测试轮数: {iterations}")
    print(f"[CONFIG] 用户输入: {user_input}")
    print(f"[CONFIG] 使用模型: deepseek-chat")
    print()
    
    # 存储测试结果
    harborai_times = []
    agently_times = []
    harborai_results = []
    agently_results = []
    harborai_errors = []
    agently_errors = []
    
    # 进行多轮测试
    for i in range(iterations):
        print(f"第 {i+1}/{iterations} 轮测试")
        print("-" * 60)
        
        # 测试 HarborAI + Agently
        print(f"[ROUND {i+1}] 测试 HarborAI + Agently...")
        harborai_time, harborai_result, harborai_error = test_harborai_agently_structured_output(user_input, schema)
        harborai_times.append(harborai_time)
        harborai_results.append(harborai_result)
        harborai_errors.append(harborai_error)
        
        print()
        
        # 测试直接 Agently
        print(f"[ROUND {i+1}] 测试直接 Agently...")
        agently_time, agently_result, agently_error = test_direct_agently_structured_output(user_input, schema)
        agently_times.append(agently_time)
        agently_results.append(agently_result)
        agently_errors.append(agently_error)
        
        print()
        print(f"[ROUND {i+1}] 本轮对比:")
        print(f"  HarborAI + Agently: {harborai_time:.3f}秒")
        print(f"  直接 Agently:      {agently_time:.3f}秒")
        if harborai_time > 0 and agently_time > 0:
            diff = harborai_time - agently_time
            percent = (diff / agently_time) * 100
            print(f"  时间差异:          {diff:+.3f}秒 ({percent:+.1f}%)")
        print()
    
    # 计算统计数据
    print("="*80)
    print("📊 性能对比统计结果")
    print("="*80)
    
    # 过滤掉失败的测试
    valid_harborai_times = [t for t in harborai_times if t > 0]
    valid_agently_times = [t for t in agently_times if t > 0]
    
    if valid_harborai_times:
        harborai_avg = statistics.mean(valid_harborai_times)
        harborai_min = min(valid_harborai_times)
        harborai_max = max(valid_harborai_times)
        print(f"HarborAI + Agently 耗时:")
        print(f"  平均: {harborai_avg:.3f}秒")
        print(f"  最小: {harborai_min:.3f}秒")
        print(f"  最大: {harborai_max:.3f}秒")
        print(f"  成功率: {len(valid_harborai_times)}/{iterations} ({len(valid_harborai_times)/iterations*100:.1f}%)")
    else:
        print("HarborAI + Agently: 所有测试均失败")
    
    print()
    
    if valid_agently_times:
        agently_avg = statistics.mean(valid_agently_times)
        agently_min = min(valid_agently_times)
        agently_max = max(valid_agently_times)
        print(f"直接 Agently 耗时:")
        print(f"  平均: {agently_avg:.3f}秒")
        print(f"  最小: {agently_min:.3f}秒")
        print(f"  最大: {agently_max:.3f}秒")
        print(f"  成功率: {len(valid_agently_times)}/{iterations} ({len(valid_agently_times)/iterations*100:.1f}%)")
    else:
        print("直接 Agently: 所有测试均失败")
    
    print()
    
    # 性能对比分析
    if valid_harborai_times and valid_agently_times:
        avg_diff = harborai_avg - agently_avg
        avg_percent = (avg_diff / agently_avg) * 100
        
        print("🔍 性能对比分析:")
        print(f"  平均时间差异: {avg_diff:+.3f}秒 ({avg_percent:+.1f}%)")
        
        if abs(avg_percent) < 5:
            print("  结论: 两种方式性能基本相当")
        elif avg_percent > 0:
            print(f"  结论: HarborAI + Agently 比直接 Agently 慢 {avg_percent:.1f}%")
            print("  可能原因: HarborAI 包装层增加了额外开销")
        else:
            print(f"  结论: HarborAI + Agently 比直接 Agently 快 {abs(avg_percent):.1f}%")
            print("  可能原因: HarborAI 可能有优化或缓存机制")
    
    # 错误分析
    harborai_error_count = sum(1 for e in harborai_errors if e is not None)
    agently_error_count = sum(1 for e in agently_errors if e is not None)
    
    if harborai_error_count > 0 or agently_error_count > 0:
        print()
        print("❌ 错误统计:")
        print(f"  HarborAI + Agently 错误: {harborai_error_count}/{iterations}")
        print(f"  直接 Agently 错误:      {agently_error_count}/{iterations}")
        
        if harborai_error_count > 0:
            print("  HarborAI + Agently 错误详情:")
            for i, error in enumerate(harborai_errors):
                if error:
                    print(f"    第{i+1}轮: {error}")
        
        if agently_error_count > 0:
            print("  直接 Agently 错误详情:")
            for i, error in enumerate(agently_errors):
                if error:
                    print(f"    第{i+1}轮: {error}")
    
    # 结果内容对比
    print()
    print("📋 结果内容对比:")
    
    # 找到第一个成功的结果进行对比
    harborai_sample = next((r for r in harborai_results if r is not None), None)
    agently_sample = next((r for r in agently_results if r is not None), None)
    
    if harborai_sample and agently_sample:
        print("  HarborAI + Agently 示例结果:")
        print(f"    分析长度: {len(harborai_sample.get('analysis', ''))}")
        print(f"    趋势数量: {len(harborai_sample.get('trends', []))}")
        print(f"    关键词数量: {len(harborai_sample.get('keywords', []))}")
        print(f"    置信度: {harborai_sample.get('confidence', 'N/A')}")
        
        print("  直接 Agently 示例结果:")
        print(f"    分析长度: {len(agently_sample.get('analysis', ''))}")
        print(f"    趋势数量: {len(agently_sample.get('trends', []))}")
        print(f"    关键词数量: {len(agently_sample.get('keywords', []))}")
        print(f"    置信度: {agently_sample.get('confidence', 'N/A')}")
    
    print()
    print("="*80)
    print("✅ 性能对比测试完成")
    print("="*80)

if __name__ == "__main__":
    print("🎯 开始 HarborAI + Agently 性能对比测试")
    
    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DEEPSEEK_BASE_URL"):
        print("[ERROR] 请确保 .env 文件中配置了 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL")
        exit(1)
    
    # 运行性能对比测试
    run_performance_comparison(iterations=3)