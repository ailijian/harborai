#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试默认性能模式设置

验证HarborAI的默认性能模式是否已正确设置为FULL模式。
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from harborai.config.settings import get_settings
from harborai.config.performance import PerformanceMode, get_performance_config

def test_default_performance_mode():
    """
    测试默认性能模式配置
    """
    print("🔧 测试HarborAI默认性能模式配置...")
    print("=" * 50)
    
    # 获取设置实例
    settings = get_settings()
    
    # 检查默认性能模式
    print(f"📊 默认性能模式: {settings.performance_mode}")
    
    # 验证是否为FULL模式
    if settings.performance_mode == "full":
        print("✅ 默认性能模式已正确设置为FULL模式")
    else:
        print(f"❌ 默认性能模式设置错误，当前为: {settings.performance_mode}")
        return False
    
    # 获取性能配置实例
    perf_config = get_performance_config()
    print(f"🎯 当前性能模式: {perf_config.mode.value}")
    
    # 检查FULL模式的功能开关
    feature_flags = perf_config.feature_flags
    print("\n🔧 FULL模式功能开关状态:")
    print(f"  - 成本追踪: {'✅' if feature_flags.enable_cost_tracking else '❌'}")
    print(f"  - 详细日志: {'✅' if feature_flags.enable_detailed_logging else '❌'}")
    print(f"  - 性能监控: {'✅' if feature_flags.enable_prometheus_metrics else '❌'}")
    print(f"  - 分布式追踪: {'✅' if feature_flags.enable_opentelemetry else '❌'}")
    print(f"  - 数据库日志: {'✅' if feature_flags.enable_postgres_logging else '❌'}")
    print(f"  - 快速路径: {'✅' if feature_flags.enable_fast_path else '❌'}")
    print(f"  - 响应缓存: {'✅' if feature_flags.enable_response_cache else '❌'}")
    print(f"  - 令牌缓存: {'✅' if feature_flags.enable_token_cache else '❌'}")
    
    # 验证FULL模式应该启用的关键功能
    expected_enabled = [
        feature_flags.enable_cost_tracking,
        feature_flags.enable_detailed_logging,
        feature_flags.enable_prometheus_metrics,
        feature_flags.enable_opentelemetry,
        feature_flags.enable_postgres_logging
    ]
    
    if all(expected_enabled):
        print("\n✅ FULL模式的所有关键功能都已正确启用")
        return True
    else:
        print("\n❌ FULL模式的某些关键功能未正确启用")
        return False

def test_performance_mode_switching():
    """
    测试性能模式切换功能
    """
    print("\n🔄 测试性能模式切换功能...")
    print("=" * 50)
    
    settings = get_settings()
    
    # 测试切换到FAST模式
    print("🚀 切换到FAST模式...")
    settings.set_performance_mode("fast")
    
    # 验证切换结果
    if settings.performance_mode == "fast":
        print("✅ 成功切换到FAST模式")
    else:
        print("❌ 切换到FAST模式失败")
        return False
    
    # 切换回FULL模式
    print("🔧 切换回FULL模式...")
    settings.set_performance_mode("full")
    
    if settings.performance_mode == "full":
        print("✅ 成功切换回FULL模式")
        return True
    else:
        print("❌ 切换回FULL模式失败")
        return False

def main():
    """
    主测试函数
    """
    print("🎯 HarborAI 性能模式配置测试")
    print("=" * 60)
    
    # 测试默认性能模式
    test1_passed = test_default_performance_mode()
    
    # 测试性能模式切换
    test2_passed = test_performance_mode_switching()
    
    # 总结测试结果
    print("\n📋 测试结果总结:")
    print("=" * 30)
    print(f"默认性能模式测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"性能模式切换测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！HarborAI性能模式配置正常")
        return True
    else:
        print("\n⚠️  部分测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)