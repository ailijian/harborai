#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 HarborAI 性能模式对比测试

本测试用于验证和对比：
1. 直接使用 Agently 结构化输出的性能（基准）
2. HarborAI FAST 模式的性能
3. HarborAI BALANCED 模式的性能
4. HarborAI FULL 模式的性能

目标：验证不同性能模式的真实性能差异，确认 README.md 中的性能数据准确性
"""

import os
import sys
import time
import json
import statistics
import importlib
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
try:
    import psutil
except Exception:
    psutil = None

# 加载环境变量
load_dotenv()

def setup_console_encoding():
    """设置控制台编码为UTF-8（Windows兼容）"""
    if sys.platform.startswith('win'):
        try:
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
        except:
            pass

setup_console_encoding()

def set_performance_mode_env(mode: str) -> None:
    """设置性能模式的环境变量"""
    # 清除之前的设置
    env_vars_to_clear = [
        'HARBORAI_PERFORMANCE_MODE',
        'HARBORAI_ENABLE_FAST_PATH',
        'HARBORAI_ENABLE_COST_TRACKING',
        'HARBORAI_ENABLE_DETAILED_LOGGING'
    ]
    
    for var in env_vars_to_clear:
        if var in os.environ:
            del os.environ[var]
    
    if mode == "FAST":
        os.environ['HARBORAI_PERFORMANCE_MODE'] = 'fast'
        os.environ['HARBORAI_ENABLE_FAST_PATH'] = 'true'
        os.environ['HARBORAI_ENABLE_COST_TRACKING'] = 'false'
        os.environ['HARBORAI_ENABLE_DETAILED_LOGGING'] = 'false'
        print("[CONFIG] 设置 FAST 模式环境变量")
    elif mode == "BALANCED":
        os.environ['HARBORAI_PERFORMANCE_MODE'] = 'balanced'
        os.environ['HARBORAI_ENABLE_FAST_PATH'] = 'true'
        os.environ['HARBORAI_ENABLE_COST_TRACKING'] = 'true'
        os.environ['HARBORAI_ENABLE_DETAILED_LOGGING'] = 'false'
        print("[CONFIG] 设置 BALANCED 模式环境变量")
    elif mode == "FULL":
        os.environ['HARBORAI_PERFORMANCE_MODE'] = 'full'
        os.environ['HARBORAI_ENABLE_FAST_PATH'] = 'false'
        os.environ['HARBORAI_ENABLE_COST_TRACKING'] = 'true'
        os.environ['HARBORAI_ENABLE_DETAILED_LOGGING'] = 'true'
        print("[CONFIG] 设置 FULL 模式环境变量")
    else:
        print(f"[WARNING] 未知的性能模式: {mode}")

def reload_harborai_module():
    """重新加载 HarborAI 模块以应用新的环境变量"""
    modules_to_reload = []
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('harborai'):
            modules_to_reload.append(module_name)
    
    # 删除已加载的模块
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    print("[DEBUG] 已重新加载 HarborAI 模块")

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

def test_harborai_with_mode(user_input: str, schema: Dict[str, Any], mode: str) -> Tuple[float, Any, str]:
    """测试指定性能模式下的 HarborAI 结构化输出"""
    print(f"[INFO] 开始测试 HarborAI {mode} 模式结构化输出...")
    
    try:
        # 设置性能模式环境变量
        set_performance_mode_env(mode)
        
        # 重新加载模块以应用新的环境变量
        reload_harborai_module()
        
        # 导入 HarborAI
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
        print(f"[DEBUG] 性能模式: {mode}")
        print(f"[DEBUG] 环境变量检查:")
        print(f"  HARBORAI_PERFORMANCE_MODE: {os.getenv('HARBORAI_PERFORMANCE_MODE', 'None')}")
        print(f"  HARBORAI_ENABLE_FAST_PATH: {os.getenv('HARBORAI_ENABLE_FAST_PATH', 'None')}")
        print(f"  HARBORAI_ENABLE_COST_TRACKING: {os.getenv('HARBORAI_ENABLE_COST_TRACKING', 'None')}")
        print(f"  HARBORAI_ENABLE_DETAILED_LOGGING: {os.getenv('HARBORAI_ENABLE_DETAILED_LOGGING', 'None')}")
        
        # 记录开始时间与资源使用
        start_time = time.time()
        proc = psutil.Process(os.getpid()) if psutil else None
        cpu_start = proc.cpu_times() if proc else None
        mem_start = proc.memory_info().rss if proc else None
        io_start = proc.io_counters() if (proc and hasattr(proc, "io_counters")) else None
        
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
        
        # 记录结束时间与资源使用
        end_time = time.time()
        duration = end_time - start_time
        cpu_end = proc.cpu_times() if proc else None
        mem_end = proc.memory_info().rss if proc else None
        io_end = proc.io_counters() if (proc and hasattr(proc, "io_counters")) else None
        if psutil:
            cpu_user = (cpu_end.user - cpu_start.user) if (cpu_start and cpu_end) else None
            cpu_sys = (cpu_end.system - cpu_start.system) if (cpu_start and cpu_end) else None
            mem_rss_mb = (mem_end / (1024*1024)) if mem_end is not None else None
            io_read = (io_end.read_bytes - io_start.read_bytes) if (io_start and io_end) else None
            io_write = (io_end.write_bytes - io_start.write_bytes) if (io_start and io_end) else None
            print(f"  资源: CPU(user) {cpu_user}s, CPU(sys) {cpu_sys}s, RSS {mem_rss_mb}MB, 读 {io_read}B, 写 {io_write}B")
        
        print(f"[SUCCESS] HarborAI {mode} 模式调用成功，耗时: {duration:.3f}秒")
        
        # 获取结构化结果
        if hasattr(response.choices[0].message, 'parsed') and response.choices[0].message.parsed:
            result = response.choices[0].message.parsed
            return duration, result, None
        else:
            error_msg = "未获得结构化输出结果"
            print(f"[ERROR] {error_msg}")
            return duration, None, error_msg
            
    except Exception as e:
        print(f"[ERROR] HarborAI {mode} 模式测试失败: {e}")
        return 0, None, str(e)

def test_direct_agently_structured_output(user_input: str, schema: Dict[str, Any]) -> Tuple[float, Any, str]:
    """测试直接使用 Agently 结构化输出（基准测试）"""
    print("[INFO] 开始测试直接 Agently 结构化输出（基准）...")
    
    try:
        from Agently.agently import Agently
        
        # 配置 Agently
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        model = "deepseek-chat"
        
        print(f"[DEBUG] 配置 Agently: base_url={base_url}, model={model}")
        
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
        
        # 记录开始时间与资源使用
        start_time = time.time()
        proc = psutil.Process(os.getpid()) if psutil else None
        cpu_start = proc.cpu_times() if proc else None
        mem_start = proc.memory_info().rss if proc else None
        io_start = proc.io_counters() if (proc and hasattr(proc, "io_counters")) else None
        
        # 调用 Agently 结构化输出
        result = (
            agent
            .input(user_input)
            .output(agently_output)
            .start()
        )
        
        # 记录结束时间与资源使用
        end_time = time.time()
        duration = end_time - start_time
        cpu_end = proc.cpu_times() if proc else None
        mem_end = proc.memory_info().rss if proc else None
        io_end = proc.io_counters() if (proc and hasattr(proc, "io_counters")) else None
        if psutil:
            cpu_user = (cpu_end.user - cpu_start.user) if (cpu_start and cpu_end) else None
            cpu_sys = (cpu_end.system - cpu_start.system) if (cpu_start and cpu_end) else None
            mem_rss_mb = (mem_end / (1024*1024)) if mem_end is not None else None
            io_read = (io_end.read_bytes - io_start.read_bytes) if (io_start and io_end) else None
            io_write = (io_end.write_bytes - io_start.write_bytes) if (io_start and io_end) else None
            print(f"  资源: CPU(user) {cpu_user}s, CPU(sys) {cpu_sys}s, RSS {mem_rss_mb}MB, 读 {io_read}B, 写 {io_write}B")
        
        print(f"[SUCCESS] 直接 Agently 调用成功，耗时: {duration:.3f}秒")
        
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
    """运行性能模式对比测试"""
    print("="*80)
    print("🚀 HarborAI 性能模式对比测试")
    print("="*80)
    
    # 测试参数
    user_input = "请分析人工智能技术的发展趋势"
    schema = get_test_schema()
    
    print(f"[CONFIG] 测试轮数: {iterations}")
    print(f"[CONFIG] 用户输入: {user_input}")
    print(f"[CONFIG] 使用模型: deepseek-chat")
    print(f"[CONFIG] 测试模式: 直接 Agently（基准）、HarborAI FAST、HarborAI BALANCED、HarborAI FULL")
    print()
    
    # 存储测试结果
    test_modes = ["Agently", "FAST", "BALANCED", "FULL"]
    results = {mode: {"times": [], "results": [], "errors": []} for mode in test_modes}
    
    # 进行多轮测试
    for i in range(iterations):
        print(f"第 {i+1}/{iterations} 轮测试")
        print("=" * 60)
        
        # 测试直接 Agently（基准）
        print(f"[ROUND {i+1}] 测试直接 Agently（基准）...")
        agently_time, agently_result, agently_error = test_direct_agently_structured_output(user_input, schema)
        results["Agently"]["times"].append(agently_time)
        results["Agently"]["results"].append(agently_result)
        results["Agently"]["errors"].append(agently_error)
        print(f"  耗时: {agently_time:.3f}秒")
        print()
        
        # 测试 HarborAI FAST 模式
        print(f"[ROUND {i+1}] 测试 HarborAI FAST 模式...")
        fast_time, fast_result, fast_error = test_harborai_with_mode(user_input, schema, "FAST")
        results["FAST"]["times"].append(fast_time)
        results["FAST"]["results"].append(fast_result)
        results["FAST"]["errors"].append(fast_error)
        print(f"  耗时: {fast_time:.3f}秒")
        print()
        
        # 测试 HarborAI BALANCED 模式
        print(f"[ROUND {i+1}] 测试 HarborAI BALANCED 模式...")
        balanced_time, balanced_result, balanced_error = test_harborai_with_mode(user_input, schema, "BALANCED")
        results["BALANCED"]["times"].append(balanced_time)
        results["BALANCED"]["results"].append(balanced_result)
        results["BALANCED"]["errors"].append(balanced_error)
        print(f"  耗时: {balanced_time:.3f}秒")
        print()
        
        # 测试 HarborAI FULL 模式
        print(f"[ROUND {i+1}] 测试 HarborAI FULL 模式...")
        full_time, full_result, full_error = test_harborai_with_mode(user_input, schema, "FULL")
        results["FULL"]["times"].append(full_time)
        results["FULL"]["results"].append(full_result)
        results["FULL"]["errors"].append(full_error)
        print(f"  耗时: {full_time:.3f}秒")
        print()
        
        # 本轮对比
        print(f"[ROUND {i+1}] 本轮性能对比:")
        print(f"  直接 Agently（基准）: {agently_time:.3f}秒")
        print(f"  HarborAI FAST:       {fast_time:.3f}秒")
        print(f"  HarborAI BALANCED:   {balanced_time:.3f}秒")
        print(f"  HarborAI FULL:       {full_time:.3f}秒")
        
        if agently_time > 0:
            print(f"  相对性能比率:")
            if fast_time > 0:
                fast_ratio = fast_time / agently_time
                print(f"    FAST vs Agently:     {fast_ratio:.2f}x ({(fast_ratio-1)*100:+.1f}%)")
            if balanced_time > 0:
                balanced_ratio = balanced_time / agently_time
                print(f"    BALANCED vs Agently: {balanced_ratio:.2f}x ({(balanced_ratio-1)*100:+.1f}%)")
            if full_time > 0:
                full_ratio = full_time / agently_time
                print(f"    FULL vs Agently:     {full_ratio:.2f}x ({(full_ratio-1)*100:+.1f}%)")
        print()
    
    # 计算统计数据
    print("="*80)
    print("📊 性能模式对比统计结果")
    print("="*80)
    
    stats = {}
    for mode in test_modes:
        valid_times = [t for t in results[mode]["times"] if t > 0]
        if valid_times:
            stats[mode] = {
                "avg": statistics.mean(valid_times),
                "min": min(valid_times),
                "max": max(valid_times),
                "success_rate": len(valid_times) / iterations * 100
            }
        else:
            stats[mode] = None
    
    # 输出详细统计
    for mode in test_modes:
        if stats[mode]:
            print(f"{mode} 模式:")
            print(f"  平均耗时: {stats[mode]['avg']:.3f}秒")
            print(f"  最小耗时: {stats[mode]['min']:.3f}秒")
            print(f"  最大耗时: {stats[mode]['max']:.3f}秒")
            print(f"  成功率:   {stats[mode]['success_rate']:.1f}%")
            # 吞吐量按 1/平均耗时 估算
            throughput = (1.0 / stats[mode]['avg']) if stats[mode]['avg'] > 0 else 0.0
            print(f"  吞吐量:   {throughput:.3f} 次/秒")
        else:
            print(f"{mode} 模式: 所有测试均失败")
        print()
    
    # 性能对比分析
    if stats["Agently"]:
        baseline_avg = stats["Agently"]["avg"]
        print("🔍 性能对比分析（相对于 Agently 基准）:")
        print(f"  Agently 基准平均耗时: {baseline_avg:.3f}秒")
        print()
        
        for mode in ["FAST", "BALANCED", "FULL"]:
            if stats[mode]:
                mode_avg = stats[mode]["avg"]
                ratio = mode_avg / baseline_avg
                improvement = (1 - ratio) * 100
                
                print(f"  HarborAI {mode} 模式:")
                print(f"    平均耗时: {mode_avg:.3f}秒")
                print(f"    相对性能: {ratio:.2f}x")
                if improvement > 0:
                    print(f"    性能提升: +{improvement:.1f}%")
                else:
                    print(f"    性能下降: {improvement:.1f}%")
                print()
        
        # 验证 README.md 数据
        print("📋 README.md 数据验证:")
        readme_data = {
            "Agently": {"expected": 4.37, "ratio": 1.00},
            "FAST": {"expected": 3.87, "ratio": 0.88},
            "BALANCED": {"expected": 4.47, "ratio": 1.02},
            "FULL": {"expected": 3.92, "ratio": 0.90}
        }
        
        for mode in test_modes:
            if stats[mode]:
                actual_time = stats[mode]["avg"]
                expected_time = readme_data[mode]["expected"]
                expected_ratio = readme_data[mode]["ratio"]
                actual_ratio = actual_time / baseline_avg if mode != "Agently" else 1.0
                
                print(f"  {mode} 模式:")
                print(f"    README 预期: {expected_time:.2f}秒 ({expected_ratio:.2f}x)")
                print(f"    实际测试:   {actual_time:.3f}秒 ({actual_ratio:.2f}x)")
                
                time_diff = abs(actual_time - expected_time)
                ratio_diff = abs(actual_ratio - expected_ratio)
                
                if time_diff < 1.0 and ratio_diff < 0.1:
                    print(f"    验证结果: ✅ 数据基本一致")
                else:
                    print(f"    验证结果: ❌ 数据存在差异")
                    print(f"    时间差异: {time_diff:.3f}秒")
                    print(f"    比率差异: {ratio_diff:.3f}")
                print()
    
    # 错误分析
    print("❌ 错误统计:")
    for mode in test_modes:
        error_count = sum(1 for e in results[mode]["errors"] if e is not None)
        if error_count > 0:
            print(f"  {mode} 模式错误: {error_count}/{iterations}")
            for i, error in enumerate(results[mode]["errors"]):
                if error:
                    print(f"    第{i+1}轮: {error}")
        else:
            print(f"  {mode} 模式: 无错误")
    
    print()
    print("="*80)
    print("✅ HarborAI 性能模式对比测试完成")
    print("="*80)

if __name__ == "__main__":
    print("🎯 开始 HarborAI 性能模式对比测试")
    print("📋 本测试将对比以下四种情况：")
    print("   1. 直接 Agently（基准）")
    print("   2. HarborAI FAST 模式")
    print("   3. HarborAI BALANCED 模式")
    print("   4. HarborAI FULL 模式")
    print()
    
    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DEEPSEEK_BASE_URL"):
        print("[ERROR] 请确保 .env 文件中配置了 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL")
        exit(1)
    
    # 运行性能模式对比测试
    run_performance_comparison(iterations=3)