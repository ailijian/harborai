#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 HarborAI 性能配置调试工具

用于验证性能模式配置是否真正生效
"""

import os
import sys
import time
import importlib
from typing import Dict, Any
from dotenv import load_dotenv

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

def set_and_verify_performance_mode(mode: str) -> None:
    """设置性能模式并验证是否生效"""
    print(f"\n{'='*60}")
    print(f"🔧 测试 {mode.upper()} 模式配置")
    print(f"{'='*60}")
    
    # 清除之前的环境变量
    env_vars = [
        'HARBORAI_PERFORMANCE_MODE',
        'HARBORAI_ENABLE_FAST_PATH',
        'HARBORAI_ENABLE_COST_TRACKING',
        'HARBORAI_ENABLE_DETAILED_LOGGING'
    ]
    
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]
    
    # 设置新的环境变量
    if mode == "fast":
        os.environ['HARBORAI_PERFORMANCE_MODE'] = 'fast'
        os.environ['HARBORAI_ENABLE_FAST_PATH'] = 'true'
        os.environ['HARBORAI_ENABLE_COST_TRACKING'] = 'false'
        os.environ['HARBORAI_ENABLE_DETAILED_LOGGING'] = 'false'
    elif mode == "balanced":
        os.environ['HARBORAI_PERFORMANCE_MODE'] = 'balanced'
        os.environ['HARBORAI_ENABLE_FAST_PATH'] = 'true'
        os.environ['HARBORAI_ENABLE_COST_TRACKING'] = 'true'
        os.environ['HARBORAI_ENABLE_DETAILED_LOGGING'] = 'false'
    elif mode == "full":
        os.environ['HARBORAI_PERFORMANCE_MODE'] = 'full'
        os.environ['HARBORAI_ENABLE_FAST_PATH'] = 'false'
        os.environ['HARBORAI_ENABLE_COST_TRACKING'] = 'true'
        os.environ['HARBORAI_ENABLE_DETAILED_LOGGING'] = 'true'
    
    # 显示环境变量设置
    print("📋 环境变量设置:")
    for var in env_vars:
        value = os.environ.get(var, "未设置")
        print(f"  {var}: {value}")
    
    # 重新加载HarborAI模块
    print("\n🔄 重新加载HarborAI模块...")
    modules_to_reload = []
    for module_name in list(sys.modules.keys()):
        if module_name.startswith('harborai'):
            modules_to_reload.append(module_name)
    
    for module_name in modules_to_reload:
        if module_name in sys.modules:
            del sys.modules[module_name]
    
    print(f"  已清除 {len(modules_to_reload)} 个HarborAI模块")
    
    # 重新导入并检查配置
    try:
        from harborai.config.settings import get_settings
        from harborai.config.performance import get_performance_config
        
        settings = get_settings()
        perf_config = get_performance_config()
        
        print(f"\n✅ HarborAI配置验证:")
        print(f"  设置中的性能模式: {settings.performance_mode}")
        print(f"  性能配置模式: {perf_config.mode.value}")
        
        # 检查功能开关
        feature_flags = perf_config.feature_flags
        print(f"\n🔧 功能开关状态:")
        print(f"  快速路径: {'✅' if feature_flags.enable_fast_path else '❌'}")
        print(f"  成本追踪: {'✅' if feature_flags.enable_cost_tracking else '❌'}")
        print(f"  详细日志: {'✅' if feature_flags.enable_detailed_logging else '❌'}")
        print(f"  性能监控: {'✅' if feature_flags.enable_prometheus_metrics else '❌'}")
        print(f"  分布式追踪: {'✅' if feature_flags.enable_opentelemetry else '❌'}")
        
        # 验证配置是否与环境变量一致
        expected_fast_path = os.environ.get('HARBORAI_ENABLE_FAST_PATH') == 'true'
        expected_cost_tracking = os.environ.get('HARBORAI_ENABLE_COST_TRACKING') == 'true'
        expected_detailed_logging = os.environ.get('HARBORAI_ENABLE_DETAILED_LOGGING') == 'true'
        
        print(f"\n🔍 配置一致性检查:")
        fast_path_match = feature_flags.enable_fast_path == expected_fast_path
        cost_tracking_match = feature_flags.enable_cost_tracking == expected_cost_tracking
        detailed_logging_match = feature_flags.enable_detailed_logging == expected_detailed_logging
        
        print(f"  快速路径: {'✅' if fast_path_match else '❌'} (期望: {expected_fast_path}, 实际: {feature_flags.enable_fast_path})")
        print(f"  成本追踪: {'✅' if cost_tracking_match else '❌'} (期望: {expected_cost_tracking}, 实际: {feature_flags.enable_cost_tracking})")
        print(f"  详细日志: {'✅' if detailed_logging_match else '❌'} (期望: {expected_detailed_logging}, 实际: {feature_flags.enable_detailed_logging})")
        
        if fast_path_match and cost_tracking_match and detailed_logging_match:
            print(f"\n🎉 {mode.upper()} 模式配置完全正确！")
        else:
            print(f"\n⚠️ {mode.upper()} 模式配置存在问题！")
            
    except Exception as e:
        print(f"\n❌ 配置验证失败: {e}")

def test_simple_performance(mode: str) -> float:
    """测试简单性能"""
    try:
        from harborai import HarborAI
        
        client = HarborAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL")
        )
        
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "请简单回答：1+1等于几？"}],
            max_tokens=10,
            temperature=0
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️ {mode.upper()} 模式简单测试:")
        print(f"  耗时: {duration:.3f}秒")
        print(f"  响应: {response.choices[0].message.content.strip()}")
        
        return duration
        
    except Exception as e:
        print(f"\n❌ {mode.upper()} 模式测试失败: {e}")
        return 0

def main():
    """主函数"""
    print("🔍 HarborAI 性能配置深度调试")
    print("="*80)
    
    # 检查环境变量
    if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DEEPSEEK_BASE_URL"):
        print("[ERROR] 请确保 .env 文件中配置了 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL")
        return
    
    modes = ["fast", "balanced", "full"]
    performance_results = {}
    
    for mode in modes:
        set_and_verify_performance_mode(mode)
        duration = test_simple_performance(mode)
        performance_results[mode] = duration
        time.sleep(1)  # 避免API限制
    
    # 性能对比
    print(f"\n{'='*80}")
    print("📊 性能对比结果")
    print(f"{'='*80}")
    
    for mode, duration in performance_results.items():
        if duration > 0:
            print(f"  {mode.upper()} 模式: {duration:.3f}秒")
        else:
            print(f"  {mode.upper()} 模式: 测试失败")
    
    # 分析性能差异
    valid_results = {k: v for k, v in performance_results.items() if v > 0}
    if len(valid_results) >= 2:
        print(f"\n🔍 性能差异分析:")
        baseline = valid_results.get("full", list(valid_results.values())[0])
        for mode, duration in valid_results.items():
            if mode != "full":
                diff = duration - baseline
                percent = (diff / baseline) * 100
                print(f"  {mode.upper()} vs FULL: {diff:+.3f}秒 ({percent:+.1f}%)")

if __name__ == "__main__":
    main()