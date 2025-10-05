#!/usr/bin/env python3
"""
本地集成测试脚本
验证所有性能测试组件的基本功能，不依赖外部服务
"""

import asyncio
import time
import math
try:
    from .memory_leak_detector import MemoryLeakDetector
    from .resource_utilization_monitor import ResourceUtilizationMonitor
except ImportError:
    from memory_leak_detector import MemoryLeakDetector
    from resource_utilization_monitor import ResourceUtilizationMonitor

# 创建一个简单的执行效率测试器类
class ExecutionEfficiencyTester:
    """简单的执行效率测试器"""
    
    def __init__(self):
        self.test_results = []
        
    async def test_async_performance(self, func, iterations=100):
        """测试异步函数性能"""
        start_time = time.time()
        for _ in range(iterations):
            await func()
        end_time = time.time()
        
        duration = end_time - start_time
        avg_time = duration / iterations
        
        result = {
            'function': func.__name__,
            'iterations': iterations,
            'total_time': duration,
            'avg_time': avg_time,
            'throughput': iterations / duration
        }
        
        self.test_results.append(result)
        return result
        
    def test_sync_performance(self, func, iterations=100):
        """测试同步函数性能"""
        start_time = time.time()
        for _ in range(iterations):
            func()
        end_time = time.time()
        
        duration = end_time - start_time
        avg_time = duration / iterations
        
        result = {
            'function': func.__name__,
            'iterations': iterations,
            'total_time': duration,
            'avg_time': avg_time,
            'throughput': iterations / duration
        }
        
        self.test_results.append(result)
        return result
        
    def get_summary(self):
        """获取测试摘要"""
        if not self.test_results:
            return {'total_tests': 0}
            
        return {
            'total_tests': len(self.test_results),
            'avg_throughput': sum(r['throughput'] for r in self.test_results) / len(self.test_results),
            'total_time': sum(r['total_time'] for r in self.test_results),
            'results': self.test_results
        }

async def run_local_integration_tests():
    """运行本地集成测试"""
    print('=== 开始本地性能测试验证 ===')
    
    # 1. 执行效率测试
    print('\n1. 执行效率测试')
    efficiency_tester = ExecutionEfficiencyTester()
    
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n-1) + fibonacci(n-2)
    
    def quick_sort(arr):
        if len(arr) <= 1:
            return arr
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        return quick_sort(left) + middle + quick_sort(right)
    
    # 测试斐波那契
    metrics1 = efficiency_tester.measure_execution_time(lambda: fibonacci(20))
    print(f'   斐波那契测试: {metrics1.execution_time:.4f}s, 成功: {metrics1.success}')
    
    # 测试排序
    test_data = [3, 6, 8, 10, 1, 2, 1]
    metrics2 = efficiency_tester.measure_execution_time(lambda: quick_sort(test_data))
    print(f'   快速排序测试: {metrics2.execution_time:.4f}s, 成功: {metrics2.success}')
    
    # 2. 内存泄漏检测
    print('\n2. 内存泄漏检测')
    detector = MemoryLeakDetector(monitoring_interval=0.5)
    
    def memory_test():
        data = []
        for i in range(100):
            data.append([j for j in range(10)])
        time.sleep(0.1)
        return len(data)
    
    detector.start_monitoring()
    for _ in range(5):
        memory_test()
        time.sleep(0.2)
    
    analysis = detector._analyze_memory_leak()
    detector.stop_monitoring()
    print(f'   内存泄漏检测: {analysis.is_leak_detected}, 泄漏率: {analysis.leak_rate:.2f} bytes/s')
    
    # 3. 资源监控
    print('\n3. 资源监控')
    monitor = ResourceUtilizationMonitor(monitoring_interval=0.5)
    monitor.start_monitoring()
    
    # 模拟一些负载
    for i in range(1000):
        result = math.sqrt(i * 1000)
    
    time.sleep(2)
    stats = monitor.get_resource_statistics()
    monitor.stop_monitoring()
    
    cpu_peak = stats.get('cpu_stats', {}).get('peak', 0)
    memory_peak = stats.get('memory_stats', {}).get('peak', 0)
    print(f'   CPU峰值: {cpu_peak:.1f}%')
    print(f'   内存峰值: {memory_peak:.1f}%')
    
    print('\n=== 本地性能测试验证完成 ===')
    
    # 返回测试结果摘要
    return {
        'execution_efficiency': {
            'fibonacci_time': metrics1.execution_time,
            'sort_time': metrics2.execution_time,
            'all_success': metrics1.success and metrics2.success
        },
        'memory_leak': {
            'leak_detected': analysis.is_leak_detected,
            'leak_rate': analysis.leak_rate
        },
        'resource_monitoring': {
            'cpu_peak': cpu_peak,
            'memory_peak': memory_peak,
            'snapshots': stats.get('total_snapshots', 0)
        }
    }

if __name__ == "__main__":
    result = asyncio.run(run_local_integration_tests())
    print(f'\n测试结果摘要: {result}')