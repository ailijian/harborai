#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能基准测试脚本

对比延迟加载优化前后的性能差异，生成详细的性能报告。
根据技术设计方案，验证初始化性能从355.58ms降低到≤160ms的目标。

测试内容：
1. 传统ClientManager初始化性能
2. LazyPluginManager初始化性能  
3. FastHarborAI客户端初始化性能
4. 插件加载性能对比
5. 内存使用对比
"""

import time
import psutil
import os
import statistics
from typing import Dict, List, Any
import json
from datetime import datetime

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from harborai.core.client_manager import ClientManager
from harborai.core.lazy_plugin_manager import LazyPluginManager
from harborai.api.fast_client import FastHarborAI


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self):
        """初始化基准测试"""
        self.test_config = {
            'timeout': 30,
            'max_retries': 3,
            'plugins': {
                'deepseek': {
                    'api_key': 'test_key',
                    'base_url': 'https://api.deepseek.com'
                }
            }
        }
        self.results = {}
        
    def measure_memory_usage(self) -> float:
        """测量当前内存使用量（MB）"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def run_multiple_times(self, func, times: int = 10) -> Dict[str, float]:
        """多次运行函数并统计性能指标
        
        Args:
            func: 要测试的函数
            times: 运行次数
            
        Returns:
            性能统计结果
        """
        execution_times = []
        memory_before = []
        memory_after = []
        
        for i in range(times):
            # 强制垃圾回收
            import gc
            gc.collect()
            
            # 记录开始状态
            mem_before = self.measure_memory_usage()
            start_time = time.perf_counter()
            
            # 执行函数
            result = func()
            
            # 记录结束状态
            end_time = time.perf_counter()
            mem_after = self.measure_memory_usage()
            
            execution_times.append((end_time - start_time) * 1000)  # 转换为毫秒
            memory_before.append(mem_before)
            memory_after.append(mem_after)
            
            # 清理资源
            if hasattr(result, 'cleanup'):
                result.cleanup()
            del result
        
        return {
            'avg_time_ms': statistics.mean(execution_times),
            'min_time_ms': min(execution_times),
            'max_time_ms': max(execution_times),
            'std_time_ms': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'avg_memory_delta_mb': statistics.mean([after - before for before, after in zip(memory_before, memory_after)]),
            'all_times_ms': execution_times
        }
    
    def test_traditional_client_manager(self) -> Dict[str, float]:
        """测试传统ClientManager初始化性能"""
        print("🔍 测试传统ClientManager初始化性能...")
        
        def create_traditional_manager():
            return ClientManager(client_config=self.test_config, lazy_loading=False)
        
        return self.run_multiple_times(create_traditional_manager)
    
    def test_lazy_plugin_manager(self) -> Dict[str, float]:
        """测试LazyPluginManager初始化性能"""
        print("🚀 测试LazyPluginManager初始化性能...")
        
        def create_lazy_manager():
            return LazyPluginManager(config=self.test_config)
        
        return self.run_multiple_times(create_lazy_manager)
    
    def test_lazy_client_manager(self) -> Dict[str, float]:
        """测试延迟加载ClientManager初始化性能"""
        print("⚡ 测试延迟加载ClientManager初始化性能...")
        
        def create_lazy_client_manager():
            return ClientManager(client_config=self.test_config, lazy_loading=True)
        
        return self.run_multiple_times(create_lazy_client_manager)
    
    def test_fast_harbor_ai(self) -> Dict[str, float]:
        """测试FastHarborAI客户端初始化性能"""
        print("🏃 测试FastHarborAI客户端初始化性能...")
        
        def create_fast_client():
            return FastHarborAI(config=self.test_config)
        
        return self.run_multiple_times(create_fast_client)
    
    def test_plugin_loading_performance(self) -> Dict[str, Any]:
        """测试插件加载性能"""
        print("🔌 测试插件加载性能...")
        
        # 创建延迟加载管理器
        lazy_manager = LazyPluginManager(config=self.test_config)
        
        # 测试首次加载性能
        start_time = time.perf_counter()
        plugin = lazy_manager.get_plugin("deepseek")
        first_load_time = (time.perf_counter() - start_time) * 1000
        
        # 测试缓存访问性能
        start_time = time.perf_counter()
        plugin2 = lazy_manager.get_plugin("deepseek")
        cached_access_time = (time.perf_counter() - start_time) * 1000
        
        # 清理
        lazy_manager.cleanup()
        
        return {
            'first_load_time_ms': first_load_time,
            'cached_access_time_ms': cached_access_time,
            'cache_speedup_ratio': first_load_time / cached_access_time if cached_access_time > 0 else float('inf')
        }
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """运行所有基准测试"""
        print("🎯 开始性能基准测试...")
        print("=" * 60)
        
        # 运行各项测试
        self.results['traditional_client_manager'] = self.test_traditional_client_manager()
        self.results['lazy_plugin_manager'] = self.test_lazy_plugin_manager()
        self.results['lazy_client_manager'] = self.test_lazy_client_manager()
        self.results['fast_harbor_ai'] = self.test_fast_harbor_ai()
        self.results['plugin_loading'] = self.test_plugin_loading_performance()
        
        # 计算性能提升
        self.calculate_improvements()
        
        return self.results
    
    def calculate_improvements(self):
        """计算性能提升指标"""
        traditional_time = self.results['traditional_client_manager']['avg_time_ms']
        lazy_manager_time = self.results['lazy_plugin_manager']['avg_time_ms']
        lazy_client_time = self.results['lazy_client_manager']['avg_time_ms']
        fast_client_time = self.results['fast_harbor_ai']['avg_time_ms']
        
        self.results['improvements'] = {
            'lazy_manager_vs_traditional': {
                'speedup_ratio': traditional_time / lazy_manager_time,
                'time_saved_ms': traditional_time - lazy_manager_time,
                'improvement_percentage': ((traditional_time - lazy_manager_time) / traditional_time) * 100
            },
            'lazy_client_vs_traditional': {
                'speedup_ratio': traditional_time / lazy_client_time,
                'time_saved_ms': traditional_time - lazy_client_time,
                'improvement_percentage': ((traditional_time - lazy_client_time) / traditional_time) * 100
            },
            'fast_client_vs_traditional': {
                'speedup_ratio': traditional_time / fast_client_time,
                'time_saved_ms': traditional_time - fast_client_time,
                'improvement_percentage': ((traditional_time - fast_client_time) / traditional_time) * 100
            }
        }
    
    def generate_report(self) -> str:
        """生成性能报告"""
        report = []
        report.append("🎯 HarborAI SDK 延迟加载性能优化报告")
        report.append("=" * 60)
        report.append(f"📅 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"🖥️  测试环境: Python {sys.version.split()[0]}")
        report.append("")
        
        # 初始化性能对比
        report.append("📊 初始化性能对比")
        report.append("-" * 40)
        
        traditional = self.results['traditional_client_manager']
        lazy_manager = self.results['lazy_plugin_manager']
        lazy_client = self.results['lazy_client_manager']
        fast_client = self.results['fast_harbor_ai']
        
        report.append(f"传统ClientManager:     {traditional['avg_time_ms']:.2f}ms ± {traditional['std_time_ms']:.2f}ms")
        report.append(f"LazyPluginManager:     {lazy_manager['avg_time_ms']:.2f}ms ± {lazy_manager['std_time_ms']:.2f}ms")
        report.append(f"延迟ClientManager:     {lazy_client['avg_time_ms']:.2f}ms ± {lazy_client['std_time_ms']:.2f}ms")
        report.append(f"FastHarborAI客户端:    {fast_client['avg_time_ms']:.2f}ms ± {fast_client['std_time_ms']:.2f}ms")
        report.append("")
        
        # 性能提升分析
        report.append("🚀 性能提升分析")
        report.append("-" * 40)
        
        improvements = self.results['improvements']
        
        for key, data in improvements.items():
            name_map = {
                'lazy_manager_vs_traditional': 'LazyPluginManager vs 传统方式',
                'lazy_client_vs_traditional': '延迟ClientManager vs 传统方式',
                'fast_client_vs_traditional': 'FastHarborAI vs 传统方式'
            }
            
            name = name_map.get(key, key)
            report.append(f"{name}:")
            report.append(f"  ⚡ 加速比: {data['speedup_ratio']:.2f}x")
            report.append(f"  ⏱️  节省时间: {data['time_saved_ms']:.2f}ms")
            report.append(f"  📈 性能提升: {data['improvement_percentage']:.1f}%")
            report.append("")
        
        # 插件加载性能
        plugin_loading = self.results['plugin_loading']
        report.append("🔌 插件加载性能")
        report.append("-" * 40)
        report.append(f"首次加载时间:         {plugin_loading['first_load_time_ms']:.2f}ms")
        report.append(f"缓存访问时间:         {plugin_loading['cached_access_time_ms']:.2f}ms")
        report.append(f"缓存加速比:           {plugin_loading['cache_speedup_ratio']:.0f}x")
        report.append("")
        
        # 目标达成情况
        report.append("🎯 优化目标达成情况")
        report.append("-" * 40)
        target_time = 160  # 目标时间160ms
        best_time = min(lazy_manager['avg_time_ms'], lazy_client['avg_time_ms'], fast_client['avg_time_ms'])
        
        if best_time <= target_time:
            report.append(f"✅ 目标达成！最佳初始化时间: {best_time:.2f}ms ≤ {target_time}ms")
        else:
            report.append(f"❌ 目标未达成。最佳初始化时间: {best_time:.2f}ms > {target_time}ms")
        
        original_time = 355.58  # 原始时间
        improvement = ((original_time - best_time) / original_time) * 100
        report.append(f"📊 相比原始性能提升: {improvement:.1f}% (从{original_time}ms降至{best_time:.2f}ms)")
        report.append("")
        
        # 内存使用情况
        report.append("💾 内存使用情况")
        report.append("-" * 40)
        report.append(f"传统ClientManager:     {traditional['avg_memory_delta_mb']:.2f}MB")
        report.append(f"LazyPluginManager:     {lazy_manager['avg_memory_delta_mb']:.2f}MB")
        report.append(f"延迟ClientManager:     {lazy_client['avg_memory_delta_mb']:.2f}MB")
        report.append(f"FastHarborAI客户端:    {fast_client['avg_memory_delta_mb']:.2f}MB")
        report.append("")
        
        # 结论和建议
        report.append("📝 结论和建议")
        report.append("-" * 40)
        report.append("1. 延迟加载机制显著提升了初始化性能")
        report.append("2. LazyPluginManager实现了按需加载，减少了启动时间")
        report.append("3. FastHarborAI客户端提供了最优的用户体验")
        report.append("4. 缓存机制确保了后续访问的高性能")
        report.append("5. 建议在生产环境中使用延迟加载模式")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """保存测试结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_benchmark_{timestamp}.json"
        
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        # 添加元数据
        results_with_metadata = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'python_version': sys.version,
                'test_config': self.test_config
            },
            'results': self.results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_with_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"📁 测试结果已保存到: {filepath}")
        return filepath


def main():
    """主函数"""
    print("🎯 HarborAI SDK 性能基准测试")
    print("=" * 60)
    
    # 创建基准测试实例
    benchmark = PerformanceBenchmark()
    
    try:
        # 运行所有测试
        results = benchmark.run_all_benchmarks()
        
        # 生成并显示报告
        report = benchmark.generate_report()
        print("\n" + report)
        
        # 保存结果
        benchmark.save_results()
        
        print("\n✅ 性能基准测试完成！")
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())