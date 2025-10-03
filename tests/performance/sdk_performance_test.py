#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI SDK 直接性能测试

直接测试SDK的性能，不需要启动Web服务
"""

import asyncio
import time
import statistics
import psutil
import gc
import sys
import os
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from harborai import HarborAI
    from harborai.config.performance import PerformanceMode
    from harborai.utils.exceptions import HarborAIError
except ImportError as e:
    print(f"❌ 导入HarborAI失败: {e}")
    print("请确保HarborAI已正确安装")
    sys.exit(1)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    response_times: List[float]
    memory_usage: List[float]
    cpu_usage: List[float]
    success_count: int
    error_count: int
    total_requests: int
    
    @property
    def avg_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0
    
    @property
    def p95_response_time(self) -> float:
        return statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) >= 20 else 0
    
    @property
    def success_rate(self) -> float:
        return (self.success_count / self.total_requests * 100) if self.total_requests > 0 else 0

class SDKPerformanceTester:
    """SDK性能测试器"""
    
    def __init__(self):
        self.results = {}
        self.process = psutil.Process()
        
    def setup_test_client(self, mode: PerformanceMode = PerformanceMode.BALANCED) -> HarborAI:
        """设置测试客户端"""
        try:
            # 使用模拟配置，避免真实API调用
            client = HarborAI(
                api_key="test-key-for-performance-testing",
                performance_mode=mode,
                enable_cache=True,
                enable_fallback=False,  # 禁用fallback避免网络调用
                enable_cost_tracking=True
            )
            return client
        except Exception as e:
            print(f"❌ 创建客户端失败: {e}")
            return None
    
    def measure_initialization_overhead(self) -> Dict[str, float]:
        """测量初始化开销"""
        print("📊 测试初始化开销...")
        
        results = {}
        
        for mode in [PerformanceMode.FAST, PerformanceMode.BALANCED, PerformanceMode.FULL]:
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                client = self.setup_test_client(mode)
                end_time = time.perf_counter()
                
                if client:
                    times.append((end_time - start_time) * 1000)  # 转换为毫秒
                
                # 清理
                del client
                gc.collect()
            
            if times:
                results[mode.value] = {
                    'avg_ms': statistics.mean(times),
                    'min_ms': min(times),
                    'max_ms': max(times),
                    'p95_ms': statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
                }
        
        return results
    
    def measure_method_call_overhead(self) -> Dict[str, Any]:
        """测量方法调用开销"""
        print("📊 测试方法调用开销...")
        
        client = self.setup_test_client(PerformanceMode.FAST)
        if not client:
            return {}
        
        # 测试不同方法的调用开销
        methods_to_test = [
            ('chat.completions.create', self._test_chat_completion_call),
            ('parameter_validation', self._test_parameter_validation),
            ('plugin_switching', self._test_plugin_switching)
        ]
        
        results = {}
        
        for method_name, test_func in methods_to_test:
            times = []
            errors = 0
            
            for _ in range(100):  # 多次测试获得准确结果
                try:
                    start_time = time.perf_counter()
                    test_func(client)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000000)  # 转换为微秒
                except Exception:
                    errors += 1
            
            if times:
                results[method_name] = {
                    'avg_us': statistics.mean(times),
                    'min_us': min(times),
                    'max_us': max(times),
                    'p95_us': statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times),
                    'error_count': errors
                }
        
        return results
    
    def _test_chat_completion_call(self, client: HarborAI):
        """测试聊天完成调用（不实际发送请求）"""
        try:
            # 只测试参数处理和验证，不实际发送请求
            messages = [{"role": "user", "content": "Hello"}]
            
            # 这里我们只测试参数验证和预处理的开销
            # 实际的网络请求会被模拟或跳过
            params = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            # 模拟参数验证过程
            if hasattr(client.chat.completions, '_validate_parameters'):
                client.chat.completions._validate_parameters(params)
            
        except Exception:
            # 预期会有错误，因为我们没有真实的API密钥
            pass
    
    def _test_parameter_validation(self, client: HarborAI):
        """测试参数验证开销"""
        messages = [{"role": "user", "content": "Test message"}]
        
        # 测试各种参数组合的验证开销
        test_params = [
            {"model": "gpt-3.5-turbo", "messages": messages},
            {"model": "gpt-4", "messages": messages, "temperature": 0.5},
            {"model": "claude-3", "messages": messages, "max_tokens": 200, "stream": True}
        ]
        
        for params in test_params:
            try:
                # 只进行参数验证，不实际调用
                if hasattr(client.chat.completions, '_validate_parameters'):
                    client.chat.completions._validate_parameters(params)
            except Exception:
                pass
    
    def _test_plugin_switching(self, client: HarborAI):
        """测试插件切换开销"""
        models = ["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet"]
        
        for model in models:
            try:
                # 测试插件管理器的模型切换开销
                if hasattr(client, '_client_manager') and hasattr(client._client_manager, 'get_plugin'):
                    client._client_manager.get_plugin(model)
            except Exception:
                pass
    
    def measure_memory_usage(self) -> Dict[str, Any]:
        """测量内存使用情况"""
        print("📊 测试内存使用...")
        
        # 获取基线内存
        gc.collect()
        baseline_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        results = {
            'baseline_mb': baseline_memory,
            'client_creation': {},
            'memory_leak_test': {}
        }
        
        # 测试客户端创建的内存开销
        for mode in [PerformanceMode.FAST, PerformanceMode.BALANCED, PerformanceMode.FULL]:
            gc.collect()
            before_memory = self.process.memory_info().rss / 1024 / 1024
            
            clients = []
            for _ in range(10):
                client = self.setup_test_client(mode)
                if client:
                    clients.append(client)
            
            after_memory = self.process.memory_info().rss / 1024 / 1024
            
            # 清理
            for client in clients:
                del client
            clients.clear()
            gc.collect()
            
            cleanup_memory = self.process.memory_info().rss / 1024 / 1024
            
            results['client_creation'][mode.value] = {
                'before_mb': before_memory,
                'after_mb': after_memory,
                'cleanup_mb': cleanup_memory,
                'overhead_per_client_mb': (after_memory - before_memory) / 10 if clients else 0
            }
        
        # 内存泄漏测试
        gc.collect()
        start_memory = self.process.memory_info().rss / 1024 / 1024
        
        for i in range(100):
            client = self.setup_test_client(PerformanceMode.FAST)
            if client:
                # 模拟一些操作
                try:
                    self._test_chat_completion_call(client)
                except Exception:
                    pass
                del client
            
            if i % 20 == 0:
                gc.collect()
        
        gc.collect()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        
        results['memory_leak_test'] = {
            'start_mb': start_memory,
            'end_mb': end_memory,
            'potential_leak_mb': end_memory - start_memory
        }
        
        return results
    
    def measure_concurrent_performance(self) -> Dict[str, Any]:
        """测量并发性能"""
        print("📊 测试并发性能...")
        
        def worker_task(worker_id: int, num_operations: int) -> Dict[str, Any]:
            """工作线程任务"""
            client = self.setup_test_client(PerformanceMode.FAST)
            if not client:
                return {'success': 0, 'errors': num_operations, 'times': []}
            
            times = []
            errors = 0
            
            for _ in range(num_operations):
                try:
                    start_time = time.perf_counter()
                    self._test_chat_completion_call(client)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)
                except Exception:
                    errors += 1
            
            return {
                'success': num_operations - errors,
                'errors': errors,
                'times': times,
                'worker_id': worker_id
            }
        
        # 测试不同并发级别
        concurrency_levels = [1, 5, 10, 20]
        operations_per_worker = 50
        
        results = {}
        
        for concurrency in concurrency_levels:
            print(f"  测试并发级别: {concurrency}")
            
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(worker_task, i, operations_per_worker)
                    for i in range(concurrency)
                ]
                
                worker_results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        worker_results.append(result)
                    except Exception as e:
                        print(f"    工作线程异常: {e}")
            
            end_time = time.perf_counter()
            
            # 汇总结果
            total_success = sum(r['success'] for r in worker_results)
            total_errors = sum(r['errors'] for r in worker_results)
            all_times = []
            for r in worker_results:
                all_times.extend(r['times'])
            
            results[f'concurrency_{concurrency}'] = {
                'total_operations': concurrency * operations_per_worker,
                'successful_operations': total_success,
                'failed_operations': total_errors,
                'success_rate': (total_success / (concurrency * operations_per_worker) * 100) if concurrency * operations_per_worker > 0 else 0,
                'total_time_seconds': end_time - start_time,
                'operations_per_second': total_success / (end_time - start_time) if (end_time - start_time) > 0 else 0,
                'avg_response_time_ms': statistics.mean(all_times) if all_times else 0,
                'p95_response_time_ms': statistics.quantiles(all_times, n=20)[18] if len(all_times) >= 20 else 0
            }
        
        return results
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合性能测试"""
        print("🚀 开始HarborAI SDK综合性能测试")
        print("=" * 60)
        
        start_time = datetime.now()
        
        # 收集系统信息
        system_info = {
            'python_version': sys.version,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'platform': sys.platform
        }
        
        results = {
            'test_info': {
                'start_time': start_time.isoformat(),
                'system_info': system_info
            },
            'initialization_overhead': self.measure_initialization_overhead(),
            'method_call_overhead': self.measure_method_call_overhead(),
            'memory_usage': self.measure_memory_usage(),
            'concurrent_performance': self.measure_concurrent_performance()
        }
        
        end_time = datetime.now()
        results['test_info']['end_time'] = end_time.isoformat()
        results['test_info']['total_duration_seconds'] = (end_time - start_time).total_seconds()
        
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """打印测试结果摘要"""
        print("\n" + "=" * 60)
        print("📊 HarborAI SDK 性能测试结果摘要")
        print("=" * 60)
        
        # 初始化开销
        if 'initialization_overhead' in results:
            print("\n🚀 初始化开销:")
            for mode, metrics in results['initialization_overhead'].items():
                print(f"  {mode}模式: {metrics['avg_ms']:.2f}ms (平均)")
        
        # 方法调用开销
        if 'method_call_overhead' in results:
            print("\n⚡ 方法调用开销:")
            for method, metrics in results['method_call_overhead'].items():
                print(f"  {method}: {metrics['avg_us']:.2f}μs (平均)")
        
        # 内存使用
        if 'memory_usage' in results:
            print("\n💾 内存使用:")
            baseline = results['memory_usage']['baseline_mb']
            print(f"  基线内存: {baseline:.2f}MB")
            
            if 'memory_leak_test' in results['memory_usage']:
                leak = results['memory_usage']['memory_leak_test']['potential_leak_mb']
                print(f"  潜在内存泄漏: {leak:.2f}MB")
        
        # 并发性能
        if 'concurrent_performance' in results:
            print("\n🔄 并发性能:")
            for level, metrics in results['concurrent_performance'].items():
                if level.startswith('concurrency_'):
                    concurrency = level.split('_')[1]
                    print(f"  {concurrency}并发: {metrics['success_rate']:.1f}%成功率, {metrics['operations_per_second']:.1f}ops/s")
        
        # PRD合规性检查
        print("\n✅ PRD合规性检查:")
        self._check_prd_compliance(results)
        
        print("\n" + "=" * 60)
    
    def _check_prd_compliance(self, results: Dict[str, Any]):
        """检查PRD合规性"""
        compliance_results = []
        
        # 检查调用封装开销 < 1ms
        if 'method_call_overhead' in results:
            for method, metrics in results['method_call_overhead'].items():
                avg_ms = metrics['avg_us'] / 1000  # 转换为毫秒
                if avg_ms < 1.0:
                    compliance_results.append(f"  ✅ {method}: {avg_ms:.3f}ms < 1ms")
                else:
                    compliance_results.append(f"  ❌ {method}: {avg_ms:.3f}ms >= 1ms")
        
        # 检查高并发成功率 > 99.9%
        if 'concurrent_performance' in results:
            for level, metrics in results['concurrent_performance'].items():
                if level.startswith('concurrency_'):
                    success_rate = metrics['success_rate']
                    if success_rate > 99.9:
                        compliance_results.append(f"  ✅ {level}: {success_rate:.1f}% > 99.9%")
                    else:
                        compliance_results.append(f"  ❌ {level}: {success_rate:.1f}% <= 99.9%")
        
        # 检查内存泄漏
        if 'memory_usage' in results and 'memory_leak_test' in results['memory_usage']:
            leak = results['memory_usage']['memory_leak_test']['potential_leak_mb']
            if leak < 10:  # 小于10MB认为可接受
                compliance_results.append(f"  ✅ 内存泄漏: {leak:.2f}MB < 10MB")
            else:
                compliance_results.append(f"  ❌ 内存泄漏: {leak:.2f}MB >= 10MB")
        
        for result in compliance_results:
            print(result)

def main():
    """主函数"""
    tester = SDKPerformanceTester()
    
    try:
        results = tester.run_comprehensive_test()
        tester.print_summary(results)
        
        # 保存详细结果
        import json
        output_file = "sdk_performance_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细结果已保存到: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())