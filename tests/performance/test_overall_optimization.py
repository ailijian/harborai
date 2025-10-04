"""
HarborAI SDK 整体性能优化验证测试

验证第一阶段（延迟加载）和第二阶段（内存优化）的综合效果
"""

import pytest
import time
import gc
import psutil
import os
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from harborai.api.fast_client import FastHarborAI
from harborai.core.optimizations.memory_manager import MemoryManager


class OverallPerformanceBenchmark:
    """整体性能基准测试类"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage_mb(self) -> float:
        """获取当前内存使用量（MB）"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def measure_initialization_performance(self, enable_optimization: bool = True) -> Dict[str, Any]:
        """测量初始化性能"""
        gc.collect()
        start_memory = self.get_memory_usage_mb()
        start_time = time.time()
        
        # 创建客户端
        client = FastHarborAI(
            api_key="test_key",
            enable_memory_optimization=enable_optimization,
            enable_lazy_loading=enable_optimization
        )
        
        init_time = time.time() - start_time
        init_memory = self.get_memory_usage_mb()
        
        # 模拟一些操作
        with patch('harborai.core.lazy_plugin_manager.LazyPluginManager.get_plugin') as mock_get_plugin:
            mock_plugin = MagicMock()
            mock_plugin.chat_completion.return_value = MagicMock()
            mock_get_plugin.return_value = mock_plugin
            
            # 执行多个操作
            for i in range(10):
                try:
                    client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": f"测试消息 {i}"}]
                    )
                except Exception:
                    pass  # 忽略模拟请求的错误
        
        operation_time = time.time() - start_time - init_time
        final_memory = self.get_memory_usage_mb()
        
        # 获取内存统计
        memory_stats = client.get_memory_stats() if hasattr(client, 'get_memory_stats') else {}
        
        # 清理
        if hasattr(client, 'cleanup'):
            client.cleanup()
        
        gc.collect()
        cleanup_memory = self.get_memory_usage_mb()
        
        return {
            'init_time': init_time,
            'operation_time': operation_time,
            'start_memory_mb': start_memory,
            'init_memory_mb': init_memory,
            'final_memory_mb': final_memory,
            'cleanup_memory_mb': cleanup_memory,
            'memory_increase_mb': final_memory - start_memory,
            'memory_stats': memory_stats,
            'optimization_enabled': enable_optimization
        }


class TestOverallOptimization:
    """整体优化效果测试"""
    
    def setup_method(self):
        """测试前准备"""
        gc.collect()
        self.benchmark = OverallPerformanceBenchmark()
    
    def test_optimization_comparison(self):
        """对比优化前后的性能差异"""
        print("\n=== 整体性能优化对比测试 ===")
        
        # 测试未优化版本
        print("\n1. 测试未优化版本...")
        unoptimized_result = self.benchmark.measure_initialization_performance(enable_optimization=False)
        
        # 等待一段时间确保内存稳定
        time.sleep(1)
        gc.collect()
        
        # 测试优化版本
        print("\n2. 测试优化版本...")
        optimized_result = self.benchmark.measure_initialization_performance(enable_optimization=True)
        
        # 输出结果
        print(f"\n=== 性能对比结果 ===")
        print(f"初始化时间:")
        print(f"  未优化: {unoptimized_result['init_time']:.3f}s")
        print(f"  优化后: {optimized_result['init_time']:.3f}s")
        print(f"  改善: {((unoptimized_result['init_time'] - optimized_result['init_time']) / unoptimized_result['init_time'] * 100):.1f}%")
        
        print(f"\n操作时间:")
        print(f"  未优化: {unoptimized_result['operation_time']:.3f}s")
        print(f"  优化后: {optimized_result['operation_time']:.3f}s")
        
        print(f"\n内存使用:")
        print(f"  未优化增长: {unoptimized_result['memory_increase_mb']:.2f}MB")
        print(f"  优化后增长: {optimized_result['memory_increase_mb']:.2f}MB")
        
        if optimized_result['memory_stats']:
            print(f"\n内存优化统计:")
            stats = optimized_result['memory_stats']
            if 'cache' in stats:
                print(f"  缓存大小: {stats['cache']['size']}")
                print(f"  缓存命中率: {stats['cache']['hit_rate']:.1%}")
            if 'weak_references_count' in stats:
                print(f"  弱引用数量: {stats['weak_references_count']}")
        
        # 验证优化效果
        # 1. 内存增长应该合理
        assert optimized_result['memory_increase_mb'] <= 5.0, \
            f"优化后内存增长过大: {optimized_result['memory_increase_mb']:.2f}MB"
        
        # 2. 初始化时间应该合理
        assert optimized_result['init_time'] <= 1.0, \
            f"优化后初始化时间过长: {optimized_result['init_time']:.3f}s"
        
        print(f"\n✓ 整体优化验证通过")
    
    def test_memory_stability_under_load(self):
        """测试负载下的内存稳定性"""
        print("\n=== 负载下内存稳定性测试 ===")
        
        client = FastHarborAI(
            api_key="test_key",
            enable_memory_optimization=True,
            enable_lazy_loading=True
        )
        
        try:
            initial_memory = self.benchmark.get_memory_usage_mb()
            memory_readings = [initial_memory]
            
            with patch('harborai.core.lazy_plugin_manager.LazyPluginManager.get_plugin') as mock_get_plugin:
                mock_plugin = MagicMock()
                mock_plugin.chat_completion.return_value = MagicMock()
                mock_get_plugin.return_value = mock_plugin
                
                # 执行大量操作
                for batch in range(5):
                    print(f"执行批次 {batch + 1}/5...")
                    
                    for i in range(20):
                        try:
                            client.chat.completions.create(
                                model="gpt-3.5-turbo",
                                messages=[{"role": "user", "content": f"负载测试消息 {batch}_{i}"}]
                            )
                        except Exception:
                            pass
                    
                    # 记录内存使用
                    gc.collect()
                    current_memory = self.benchmark.get_memory_usage_mb()
                    memory_readings.append(current_memory)
                    print(f"  当前内存: {current_memory:.2f}MB")
            
            # 分析内存趋势
            memory_increase = memory_readings[-1] - memory_readings[0]
            max_memory = max(memory_readings)
            min_memory = min(memory_readings)
            
            print(f"\n内存稳定性分析:")
            print(f"  初始内存: {memory_readings[0]:.2f}MB")
            print(f"  最终内存: {memory_readings[-1]:.2f}MB")
            print(f"  总增长: {memory_increase:.2f}MB")
            print(f"  最大内存: {max_memory:.2f}MB")
            print(f"  内存波动: {max_memory - min_memory:.2f}MB")
            
            # 验证内存稳定性
            assert memory_increase <= 10.0, f"内存增长过大: {memory_increase:.2f}MB"
            assert (max_memory - min_memory) <= 15.0, f"内存波动过大: {max_memory - min_memory:.2f}MB"
            
            print(f"✓ 内存稳定性验证通过")
            
        finally:
            if hasattr(client, 'cleanup'):
                client.cleanup()
    
    def test_optimization_components_integration(self):
        """测试优化组件的集成效果"""
        print("\n=== 优化组件集成测试 ===")
        
        client = FastHarborAI(
            api_key="test_key",
            enable_memory_optimization=True,
            enable_lazy_loading=True
        )
        
        try:
            # 验证组件是否正确初始化
            assert hasattr(client, '_memory_manager'), "内存管理器未初始化"
            assert hasattr(client, 'get_memory_stats'), "内存统计方法未找到"
            
            # 获取初始统计
            initial_stats = client.get_memory_stats()
            print(f"初始内存统计: {initial_stats}")
            
            # 执行一些操作
            with patch('harborai.core.lazy_plugin_manager.LazyPluginManager.get_plugin') as mock_get_plugin:
                mock_plugin = MagicMock()
                mock_plugin.chat_completion.return_value = MagicMock()
                mock_get_plugin.return_value = mock_plugin
                
                for i in range(5):
                    try:
                        client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": f"集成测试消息 {i}"}]
                        )
                    except Exception:
                        pass
            
            # 获取最终统计
            final_stats = client.get_memory_stats()
            print(f"最终内存统计: {final_stats}")
            
            # 验证组件工作正常
            assert 'cache' in final_stats, "缓存统计缺失"
            assert final_stats['cache']['size'] >= 0, "缓存大小异常"
            
            print(f"✓ 优化组件集成验证通过")
            
        finally:
            if hasattr(client, 'cleanup'):
                client.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])