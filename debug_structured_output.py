#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 结构化输出性能调试工具
专门测试结构化输出在不同性能模式下的表现
"""

import os
import sys
import time
import importlib
from typing import Dict, Any, List
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def setup_console_encoding():
    """设置控制台编码"""
    if sys.platform == "win32":
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

def set_performance_mode_env(mode: str) -> None:
    """设置性能模式环境变量"""
    mode_configs = {
        'fast': {
            'HARBORAI_PERFORMANCE_MODE': 'fast',
            'HARBORAI_ENABLE_FAST_PATH': 'true',
            'HARBORAI_ENABLE_COST_TRACKING': 'false',
            'HARBORAI_ENABLE_DETAILED_LOGGING': 'false'
        },
        'balanced': {
            'HARBORAI_PERFORMANCE_MODE': 'balanced',
            'HARBORAI_ENABLE_FAST_PATH': 'false',
            'HARBORAI_ENABLE_COST_TRACKING': 'true',
            'HARBORAI_ENABLE_DETAILED_LOGGING': 'false'
        },
        'full': {
            'HARBORAI_PERFORMANCE_MODE': 'full',
            'HARBORAI_ENABLE_FAST_PATH': 'false',
            'HARBORAI_ENABLE_COST_TRACKING': 'true',
            'HARBORAI_ENABLE_DETAILED_LOGGING': 'true'
        }
    }
    
    config = mode_configs.get(mode, mode_configs['full'])
    for key, value in config.items():
        os.environ[key] = value

def clear_harborai_modules():
    """清除已加载的HarborAI模块"""
    modules_to_remove = [name for name in sys.modules.keys() if name.startswith('harborai')]
    for module_name in modules_to_remove:
        del sys.modules[module_name]
    return len(modules_to_remove)

def test_structured_output_performance(mode: str, rounds: int = 3) -> Dict[str, Any]:
    """测试结构化输出性能"""
    print(f"\n============================================================")
    print(f"🧪 测试 {mode.upper()} 模式结构化输出性能")
    print(f"============================================================")
    
    # 设置环境变量
    set_performance_mode_env(mode)
    
    # 清除并重新加载模块
    cleared = clear_harborai_modules()
    print(f"🔄 重新加载HarborAI模块... (清除了 {cleared} 个模块)")
    
    # 导入HarborAI
    import harborai
    from harborai.config.settings import get_settings
    
    # 验证配置
    settings = get_settings()
    print(f"✅ 当前性能模式: {settings.performance_mode}")
    
    # 定义结构化输出schema
    schema = {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "object",
                "properties": {
                    "main_topic": {"type": "string"},
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["main_topic", "key_points", "sentiment", "confidence"]
            },
            "summary": {"type": "string"},
            "recommendations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string"},
                        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                        "rationale": {"type": "string"}
                    },
                    "required": ["action", "priority", "rationale"]
                }
            }
        },
        "required": ["analysis", "summary", "recommendations"]
    }
    
    # 测试prompt
    prompt = """请分析以下商业场景并提供结构化建议：

一家中型科技公司正在考虑是否要投资开发AI驱动的客户服务系统。当前他们使用传统的人工客服，但面临成本上升和响应时间长的问题。公司有200名员工，年收入5000万元，客服团队占20人。

请分析这个场景并提供详细的建议。"""
    
    times = []
    results = []
    errors = []
    
    for round_num in range(1, rounds + 1):
        print(f"\n🔄 第 {round_num} 轮测试...")
        
        try:
            start_time = time.time()
            
            # 创建HarborAI客户端
            client = harborai.HarborAI()
            
            # 调用结构化输出
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": {"name": "business_analysis", "schema": schema}}
            )
            
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            
            # 检查结果
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                if content and content.strip():
                    results.append(content)
                    print(f"  ✅ 耗时: {elapsed:.3f}秒 - 成功获取结构化输出")
                else:
                    errors.append(f"第{round_num}轮: 空响应")
                    print(f"  ❌ 耗时: {elapsed:.3f}秒 - 空响应")
            else:
                errors.append(f"第{round_num}轮: 无效响应格式")
                print(f"  ❌ 耗时: {elapsed:.3f}秒 - 无效响应格式")
                
        except Exception as e:
            end_time = time.time()
            elapsed = end_time - start_time
            times.append(elapsed)
            errors.append(f"第{round_num}轮: {str(e)}")
            print(f"  ❌ 耗时: {elapsed:.3f}秒 - 错误: {str(e)}")
    
    # 计算统计数据
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        success_rate = (len(results) / rounds) * 100
    else:
        avg_time = min_time = max_time = 0
        success_rate = 0
    
    return {
        'mode': mode,
        'times': times,
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'success_rate': success_rate,
        'results': results,
        'errors': errors
    }

def main():
    """主函数"""
    setup_console_encoding()
    
    print("🔍 HarborAI 结构化输出性能深度调试")
    print("=" * 80)
    print("专门测试复杂结构化输出在不同性能模式下的表现差异")
    
    modes = ['fast', 'balanced', 'full']
    all_results = {}
    
    for mode in modes:
        result = test_structured_output_performance(mode, rounds=3)
        all_results[mode] = result
    
    # 输出对比结果
    print(f"\n" + "=" * 80)
    print("📊 结构化输出性能对比结果")
    print("=" * 80)
    
    for mode in modes:
        result = all_results[mode]
        print(f"\n🎯 {mode.upper()} 模式:")
        print(f"  平均耗时: {result['avg_time']:.3f}秒")
        print(f"  最短耗时: {result['min_time']:.3f}秒")
        print(f"  最长耗时: {result['max_time']:.3f}秒")
        print(f"  成功率: {result['success_rate']:.1f}%")
        if result['errors']:
            print(f"  错误: {len(result['errors'])}个")
            for error in result['errors']:
                print(f"    - {error}")
    
    # 性能差异分析
    print(f"\n🔍 性能差异分析:")
    fast_time = all_results['fast']['avg_time']
    balanced_time = all_results['balanced']['avg_time']
    full_time = all_results['full']['avg_time']
    
    if fast_time > 0:
        balanced_ratio = balanced_time / fast_time
        full_ratio = full_time / fast_time
        print(f"  BALANCED vs FAST: {balanced_ratio:.2f}x ({(balanced_ratio-1)*100:+.1f}%)")
        print(f"  FULL vs FAST: {full_ratio:.2f}x ({(full_ratio-1)*100:+.1f}%)")
    
    if balanced_time > 0 and full_time > 0:
        full_vs_balanced = full_time / balanced_time
        print(f"  FULL vs BALANCED: {full_vs_balanced:.2f}x ({(full_vs_balanced-1)*100:+.1f}%)")
    
    print(f"\n🎯 关键发现:")
    if balanced_time > full_time:
        print(f"  ⚠️  BALANCED模式比FULL模式慢 {(balanced_time/full_time-1)*100:.1f}% - 这是异常的！")
    
    if fast_time > balanced_time or fast_time > full_time:
        print(f"  ⚠️  FAST模式没有显示出性能优势 - 优化可能无效！")

if __name__ == "__main__":
    main()