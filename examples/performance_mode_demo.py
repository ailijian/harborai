#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 性能模式使用示例

演示如何使用HarborAI的三种性能模式：FAST、BALANCED、FULL
"""

import asyncio
import time
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from harborai import HarborAI
from harborai.config.settings import get_settings
from harborai.config.performance import PerformanceMode


def demonstrate_performance_modes():
    """
    演示三种性能模式的配置和使用
    """
    print("🎯 HarborAI 性能模式演示")
    print("=" * 50)
    
    # 获取配置实例
    settings = get_settings()
    
    print(f"📊 当前默认性能模式: {settings.performance_mode}")
    print()
    
    # 演示三种性能模式
    modes = [
        ("fast", "🚀 FAST模式 - 最快速度，最小功能"),
        ("balanced", "⚖️ BALANCED模式 - 平衡性能与功能"),
        ("full", "🔧 FULL模式 - 完整功能，包含所有监控")
    ]
    
    for mode, description in modes:
        print(f"{description}")
        print("-" * 40)
        
        # 设置性能模式
        settings.set_performance_mode(mode)
        
        # 获取当前性能配置
        perf_config = settings.get_current_performance_config()
        feature_flags = perf_config.feature_flags
        
        # 显示功能开关状态
        print(f"  成本追踪: {'✅' if feature_flags.enable_cost_tracking else '❌'}")
        print(f"  详细日志: {'✅' if feature_flags.enable_detailed_logging else '❌'}")
        print(f"  性能监控: {'✅' if feature_flags.enable_prometheus_metrics else '❌'}")
        print(f"  分布式追踪: {'✅' if feature_flags.enable_opentelemetry else '❌'}")
        print(f"  快速路径: {'✅' if feature_flags.enable_fast_path else '❌'}")
        print(f"  响应缓存: {'✅' if feature_flags.enable_response_cache else '❌'}")
        print()


def demonstrate_client_initialization():
    """
    演示如何在客户端初始化时指定性能模式
    """
    print("🔧 客户端初始化性能模式演示")
    print("=" * 50)
    
    # 方法1: 环境变量方式（模拟）
    print("📝 方法1: 通过环境变量设置")
    print("export HARBORAI_PERFORMANCE_MODE=fast")
    print()
    
    # 方法2: 代码中动态设置
    print("📝 方法2: 代码中动态设置")
    print("```python")
    print("from harborai.config import get_settings")
    print("settings = get_settings()")
    print("settings.set_performance_mode('fast')")
    print("```")
    print()
    
    # 方法3: 初始化时指定（模拟，因为需要真实的API密钥）
    print("📝 方法3: 初始化时指定性能模式")
    print("```python")
    print("from harborai import HarborAI")
    print("")
    print("# 同步客户端")
    print("client = HarborAI(")
    print("    api_key='your-api-key',")
    print("    performance_mode='fast'")
    print(")")
    print("")
    print("# 异步客户端")
    print("async_client = HarborAI(")
    print("    api_key='your-api-key',")
    print("    performance_mode='balanced'")
    print(")")
    print("```")
    print()


def demonstrate_performance_comparison():
    """
    演示性能模式对比（模拟）
    """
    print("📊 性能模式对比演示")
    print("=" * 50)
    
    # 模拟不同性能模式的响应时间
    performance_data = {
        "FAST": {
            "avg_response_time": "1.2s",
            "features_enabled": 3,
            "memory_usage": "低",
            "适用场景": "高并发生产环境"
        },
        "BALANCED": {
            "avg_response_time": "1.8s",
            "features_enabled": 6,
            "memory_usage": "中等",
            "适用场景": "一般生产环境"
        },
        "FULL": {
            "avg_response_time": "2.5s",
            "features_enabled": 10,
            "memory_usage": "高",
            "适用场景": "开发和调试环境"
        }
    }
    
    print(f"{'模式':<10} {'响应时间':<12} {'功能数量':<10} {'内存使用':<10} {'适用场景':<15}")
    print("-" * 70)
    
    for mode, data in performance_data.items():
        print(f"{mode:<10} {data['avg_response_time']:<12} {data['features_enabled']:<10} {data['memory_usage']:<10} {data['适用场景']:<15}")
    
    print()
    print("💡 建议:")
    print("  - 生产环境高并发场景: 使用 FAST 模式")
    print("  - 一般生产环境: 使用 BALANCED 模式")
    print("  - 开发调试环境: 使用 FULL 模式")
    print()


def demonstrate_runtime_switching():
    """
    演示运行时性能模式切换
    """
    print("🔄 运行时性能模式切换演示")
    print("=" * 50)
    
    settings = get_settings()
    
    # 记录初始模式
    initial_mode = settings.performance_mode
    print(f"🎯 初始性能模式: {initial_mode}")
    
    # 演示切换过程
    modes_to_test = ["fast", "balanced", "full"]
    
    for mode in modes_to_test:
        print(f"\n🔄 切换到 {mode.upper()} 模式...")
        
        start_time = time.time()
        settings.set_performance_mode(mode)
        switch_time = time.time() - start_time
        
        print(f"✅ 切换完成，耗时: {switch_time:.3f}s")
        print(f"📊 当前模式: {settings.performance_mode}")
        
        # 获取当前配置
        perf_config = settings.get_current_performance_config()
        enabled_features = sum([
            perf_config.feature_flags.enable_cost_tracking,
            perf_config.feature_flags.enable_detailed_logging,
            perf_config.feature_flags.enable_prometheus_metrics,
            perf_config.feature_flags.enable_opentelemetry,
            perf_config.feature_flags.enable_postgres_logging
        ])
        print(f"🔧 启用的核心功能数量: {enabled_features}/5")
    
    # 恢复初始模式
    print(f"\n🔙 恢复到初始模式: {initial_mode}")
    settings.set_performance_mode(initial_mode)
    print(f"✅ 恢复完成，当前模式: {settings.performance_mode}")


def main():
    """
    主演示函数
    """
    print("🌟 HarborAI 性能模式完整演示")
    print("=" * 60)
    print()
    
    try:
        # 1. 演示性能模式配置
        demonstrate_performance_modes()
        print()
        
        # 2. 演示客户端初始化
        demonstrate_client_initialization()
        
        # 3. 演示性能对比
        demonstrate_performance_comparison()
        
        # 4. 演示运行时切换
        demonstrate_runtime_switching()
        
        print("\n🎉 演示完成！")
        print("\n📚 更多信息请参考:")
        print("  - README.md 中的性能模式配置章节")
        print("  - harborai/config/performance.py 源码")
        print("  - 性能测试脚本: comprehensive_performance_test.py")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)