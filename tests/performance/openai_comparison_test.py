#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI SDK vs OpenAI SDK 性能对比测试

对比HarborAI SDK和OpenAI SDK在相同条件下的性能表现
"""

import asyncio
import time
import statistics
import psutil
import gc
import sys
import os
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from datetime import datetime
import json

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from harborai import HarborAI
    from harborai.config.performance import PerformanceMode
    from harborai.utils.exceptions import HarborAIError
except ImportError as e:
    print(f"❌ 导入HarborAI失败: {e}")
    HarborAI = None

try:
    import openai
    from openai import OpenAI
except ImportError as e:
    print(f"❌ 导入OpenAI失败: {e}")
    OpenAI = None

@dataclass
class ComparisonMetrics:
    """对比指标数据类"""
    sdk_name: str
    initialization_time_ms: float
    method_call_overhead_us: float
    memory_usage_mb: float
    concurrent_throughput_ops_per_sec: float
    success_rate_percent: float
    error_count: int

class SDKComparator:
    """SDK对比测试器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.test_api_key = "test-key-for-comparison"
        
    def create_mock_harborai_client(self) -> Optional[HarborAI]:
        """创建模拟HarborAI客户端"""
        if HarborAI is None:
            return None
        
        try:
            return HarborAI(
                api_key=self.test_api_key,
                performance_mode=PerformanceMode.FAST,
                enable_cache=True,
                enable_fallback=False,
                enable_cost_tracking=False
            )
        except Exception as e:
            print(f"❌ 创建HarborAI客户端失败: {e}")
            return None
    
    def create_mock_openai_client(self) -> Optional[OpenAI]:
        """创建模拟OpenAI客户端"""
        if OpenAI is None:
            return None
        
        try:
            return OpenAI(
                api_key=self.test_api_key,
                timeout=30.0
            )
        except Exception as e:
            print(f"❌ 创建OpenAI客户端失败: {e}")
            return None
    
    def measure_initialization_time(self, sdk_name: str, create_func) -> float:
        """测量初始化时间"""
        times = []
        
        for _ in range(10):
            gc.collect()
            start_time = time.perf_counter()
            
            try:
                client = create_func()
                end_time = time.perf_counter()
                
                if client:
                    times.append((end_time - start_time) * 1000)  # 转换为毫秒
                    del client
            except Exception:
                pass
            
            gc.collect()
        
        return statistics.mean(times) if times else 0
    
    def measure_method_call_overhead(self, sdk_name: str, client) -> float:
        """测量方法调用开销"""
        if client is None:
            return 0
        
        times = []
        
        for _ in range(100):
            try:
                start_time = time.perf_counter()
                
                # 模拟方法调用（不实际发送请求）
                if sdk_name == "HarborAI":
                    self._mock_harborai_call(client)
                elif sdk_name == "OpenAI":
                    self._mock_openai_call(client)
                
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000000)  # 转换为微秒
                
            except Exception:
                pass
        
        return statistics.mean(times) if times else 0
    
    def _mock_harborai_call(self, client):
        """模拟HarborAI调用"""
        try:
            # 只测试参数处理，不实际发送请求
            messages = [{"role": "user", "content": "Hello"}]
            params = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            # 模拟参数验证
            if hasattr(client.chat.completions, '_validate_parameters'):
                client.chat.completions._validate_parameters(params)
        except Exception:
            pass
    
    def _mock_openai_call(self, client):
        """模拟OpenAI调用"""
        try:
            # 只测试参数处理，不实际发送请求
            messages = [{"role": "user", "content": "Hello"}]
            
            # 模拟参数构建（OpenAI SDK的内部处理）
            params = {
                "model": "gpt-3.5-turbo",
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 100
            }
            
            # 这里只是构建参数，不实际调用API
            # 因为我们没有真实的API密钥
            
        except Exception:
            pass
    
    def measure_memory_usage(self, sdk_name: str, create_func) -> float:
        """测量内存使用"""
        gc.collect()
        baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
        clients = []
        try:
            # 创建多个客户端实例
            for _ in range(10):
                client = create_func()
                if client:
                    clients.append(client)
            
            peak_memory = self.process.memory_info().rss / 1024 / 1024
            memory_overhead = peak_memory - baseline_memory
            
        except Exception:
            memory_overhead = 0
        finally:
            # 清理
            for client in clients:
                del client
            clients.clear()
            gc.collect()
        
        return memory_overhead
    
    def measure_concurrent_performance(self, sdk_name: str, create_func) -> Dict[str, float]:
        """测量并发性能"""
        def worker_task(worker_id: int, num_operations: int) -> Dict[str, Any]:
            """工作线程任务"""
            client = create_func()
            if not client:
                return {'success': 0, 'errors': num_operations, 'times': []}
            
            times = []
            errors = 0
            
            for _ in range(num_operations):
                try:
                    start_time = time.perf_counter()
                    
                    if sdk_name == "HarborAI":
                        self._mock_harborai_call(client)
                    elif sdk_name == "OpenAI":
                        self._mock_openai_call(client)
                    
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)
                except Exception:
                    errors += 1
            
            return {
                'success': num_operations - errors,
                'errors': errors,
                'times': times
            }
        
        # 测试并发性能
        concurrency = 10
        operations_per_worker = 50
        
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
                except Exception:
                    pass
        
        end_time = time.perf_counter()
        
        # 汇总结果
        total_success = sum(r['success'] for r in worker_results)
        total_errors = sum(r['errors'] for r in worker_results)
        total_operations = concurrency * operations_per_worker
        
        return {
            'throughput_ops_per_sec': total_success / (end_time - start_time) if (end_time - start_time) > 0 else 0,
            'success_rate': (total_success / total_operations * 100) if total_operations > 0 else 0,
            'error_count': total_errors
        }
    
    def run_comparison_test(self) -> Dict[str, ComparisonMetrics]:
        """运行对比测试"""
        print("🚀 开始HarborAI vs OpenAI SDK性能对比测试")
        print("=" * 60)
        
        results = {}
        
        # 测试HarborAI SDK
        if HarborAI is not None:
            print("📊 测试HarborAI SDK...")
            
            init_time = self.measure_initialization_time("HarborAI", self.create_mock_harborai_client)
            
            harborai_client = self.create_mock_harborai_client()
            call_overhead = self.measure_method_call_overhead("HarborAI", harborai_client)
            
            memory_usage = self.measure_memory_usage("HarborAI", self.create_mock_harborai_client)
            
            concurrent_metrics = self.measure_concurrent_performance("HarborAI", self.create_mock_harborai_client)
            
            results["HarborAI"] = ComparisonMetrics(
                sdk_name="HarborAI",
                initialization_time_ms=init_time,
                method_call_overhead_us=call_overhead,
                memory_usage_mb=memory_usage,
                concurrent_throughput_ops_per_sec=concurrent_metrics['throughput_ops_per_sec'],
                success_rate_percent=concurrent_metrics['success_rate'],
                error_count=concurrent_metrics['error_count']
            )
            
            print(f"  ✅ HarborAI测试完成")
        else:
            print("  ❌ HarborAI SDK不可用")
        
        # 测试OpenAI SDK
        if OpenAI is not None:
            print("📊 测试OpenAI SDK...")
            
            init_time = self.measure_initialization_time("OpenAI", self.create_mock_openai_client)
            
            openai_client = self.create_mock_openai_client()
            call_overhead = self.measure_method_call_overhead("OpenAI", openai_client)
            
            memory_usage = self.measure_memory_usage("OpenAI", self.create_mock_openai_client)
            
            concurrent_metrics = self.measure_concurrent_performance("OpenAI", self.create_mock_openai_client)
            
            results["OpenAI"] = ComparisonMetrics(
                sdk_name="OpenAI",
                initialization_time_ms=init_time,
                method_call_overhead_us=call_overhead,
                memory_usage_mb=memory_usage,
                concurrent_throughput_ops_per_sec=concurrent_metrics['throughput_ops_per_sec'],
                success_rate_percent=concurrent_metrics['success_rate'],
                error_count=concurrent_metrics['error_count']
            )
            
            print(f"  ✅ OpenAI测试完成")
        else:
            print("  ❌ OpenAI SDK不可用")
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, ComparisonMetrics]) -> str:
        """生成对比报告"""
        report = []
        
        report.append("# HarborAI vs OpenAI SDK 性能对比报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        if len(results) < 2:
            report.append("⚠️ 无法进行完整对比，缺少SDK实例")
            return "\n".join(report)
        
        harborai_metrics = results.get("HarborAI")
        openai_metrics = results.get("OpenAI")
        
        if not harborai_metrics or not openai_metrics:
            report.append("⚠️ 无法进行完整对比，缺少测试数据")
            return "\n".join(report)
        
        # 性能对比表格
        report.append("## 性能指标对比")
        report.append("")
        report.append("| 指标 | HarborAI | OpenAI | 优势 |")
        report.append("|------|----------|--------|------|")
        
        # 初始化时间
        harbor_init = harborai_metrics.initialization_time_ms
        openai_init = openai_metrics.initialization_time_ms
        init_winner = "HarborAI" if harbor_init < openai_init else "OpenAI"
        init_improvement = abs(harbor_init - openai_init) / max(harbor_init, openai_init) * 100
        
        report.append(f"| 初始化时间 | {harbor_init:.2f}ms | {openai_init:.2f}ms | {init_winner} ({init_improvement:.1f}%更快) |")
        
        # 方法调用开销
        harbor_call = harborai_metrics.method_call_overhead_us
        openai_call = openai_metrics.method_call_overhead_us
        call_winner = "HarborAI" if harbor_call < openai_call else "OpenAI"
        call_improvement = abs(harbor_call - openai_call) / max(harbor_call, openai_call) * 100
        
        report.append(f"| 方法调用开销 | {harbor_call:.2f}μs | {openai_call:.2f}μs | {call_winner} ({call_improvement:.1f}%更快) |")
        
        # 内存使用
        harbor_mem = harborai_metrics.memory_usage_mb
        openai_mem = openai_metrics.memory_usage_mb
        mem_winner = "HarborAI" if harbor_mem < openai_mem else "OpenAI"
        mem_improvement = abs(harbor_mem - openai_mem) / max(harbor_mem, openai_mem) * 100
        
        report.append(f"| 内存使用 | {harbor_mem:.2f}MB | {openai_mem:.2f}MB | {mem_winner} ({mem_improvement:.1f}%更少) |")
        
        # 并发吞吐量
        harbor_throughput = harborai_metrics.concurrent_throughput_ops_per_sec
        openai_throughput = openai_metrics.concurrent_throughput_ops_per_sec
        throughput_winner = "HarborAI" if harbor_throughput > openai_throughput else "OpenAI"
        throughput_improvement = abs(harbor_throughput - openai_throughput) / max(harbor_throughput, openai_throughput) * 100
        
        report.append(f"| 并发吞吐量 | {harbor_throughput:.1f}ops/s | {openai_throughput:.1f}ops/s | {throughput_winner} ({throughput_improvement:.1f}%更高) |")
        
        # 成功率
        harbor_success = harborai_metrics.success_rate_percent
        openai_success = openai_metrics.success_rate_percent
        success_winner = "HarborAI" if harbor_success > openai_success else "OpenAI"
        
        report.append(f"| 成功率 | {harbor_success:.1f}% | {openai_success:.1f}% | {success_winner} |")
        
        report.append("")
        
        # 综合评分
        report.append("## 综合评分")
        
        # 计算各项得分（0-100分）
        harbor_scores = {
            'initialization': 100 if harbor_init <= openai_init else max(0, 100 - (harbor_init - openai_init) / openai_init * 100),
            'call_overhead': 100 if harbor_call <= openai_call else max(0, 100 - (harbor_call - openai_call) / openai_call * 100),
            'memory': 100 if harbor_mem <= openai_mem else max(0, 100 - (harbor_mem - openai_mem) / openai_mem * 100),
            'throughput': 100 if harbor_throughput >= openai_throughput else max(0, harbor_throughput / openai_throughput * 100),
            'success_rate': harbor_success
        }
        
        openai_scores = {
            'initialization': 100 if openai_init <= harbor_init else max(0, 100 - (openai_init - harbor_init) / harbor_init * 100),
            'call_overhead': 100 if openai_call <= harbor_call else max(0, 100 - (openai_call - harbor_call) / harbor_call * 100),
            'memory': 100 if openai_mem <= harbor_mem else max(0, 100 - (openai_mem - harbor_mem) / harbor_mem * 100),
            'throughput': 100 if openai_throughput >= harbor_throughput else max(0, openai_throughput / harbor_throughput * 100),
            'success_rate': openai_success
        }
        
        harbor_total = sum(harbor_scores.values()) / len(harbor_scores)
        openai_total = sum(openai_scores.values()) / len(openai_scores)
        
        report.append(f"- **HarborAI 综合得分**: {harbor_total:.1f}/100")
        report.append(f"- **OpenAI 综合得分**: {openai_total:.1f}/100")
        report.append("")
        
        # 优势分析
        report.append("## 优势分析")
        
        if harbor_total > openai_total:
            report.append("### 🏆 HarborAI 整体表现更优")
            report.append("**HarborAI的优势：**")
            if harbor_init < openai_init:
                report.append(f"- 初始化速度更快 ({init_improvement:.1f}%)")
            if harbor_call < openai_call:
                report.append(f"- 方法调用开销更低 ({call_improvement:.1f}%)")
            if harbor_mem < openai_mem:
                report.append(f"- 内存使用更少 ({mem_improvement:.1f}%)")
            if harbor_throughput > openai_throughput:
                report.append(f"- 并发处理能力更强 ({throughput_improvement:.1f}%)")
        else:
            report.append("### 🏆 OpenAI 整体表现更优")
            report.append("**OpenAI的优势：**")
            if openai_init < harbor_init:
                report.append(f"- 初始化速度更快 ({init_improvement:.1f}%)")
            if openai_call < harbor_call:
                report.append(f"- 方法调用开销更低 ({call_improvement:.1f}%)")
            if openai_mem < harbor_mem:
                report.append(f"- 内存使用更少 ({mem_improvement:.1f}%)")
            if openai_throughput > harbor_throughput:
                report.append(f"- 并发处理能力更强 ({throughput_improvement:.1f}%)")
        
        report.append("")
        
        # 结论
        report.append("## 结论")
        if harbor_total > openai_total:
            report.append("HarborAI SDK在性能测试中表现优异，在多个关键指标上超越了OpenAI SDK。")
            report.append("这证明了HarborAI的架构优化和性能调优是有效的。")
        else:
            report.append("OpenAI SDK在性能测试中表现更好，HarborAI仍有优化空间。")
            report.append("建议重点关注性能瓶颈的优化。")
        
        return "\n".join(report)
    
    def print_summary(self, results: Dict[str, ComparisonMetrics]):
        """打印对比摘要"""
        print("\n" + "=" * 60)
        print("📊 HarborAI vs OpenAI SDK 性能对比摘要")
        print("=" * 60)
        
        for sdk_name, metrics in results.items():
            print(f"\n🔧 {sdk_name} SDK:")
            print(f"  初始化时间: {metrics.initialization_time_ms:.2f}ms")
            print(f"  方法调用开销: {metrics.method_call_overhead_us:.2f}μs")
            print(f"  内存使用: {metrics.memory_usage_mb:.2f}MB")
            print(f"  并发吞吐量: {metrics.concurrent_throughput_ops_per_sec:.1f}ops/s")
            print(f"  成功率: {metrics.success_rate_percent:.1f}%")
        
        print("\n" + "=" * 60)

def main():
    """主函数"""
    comparator = SDKComparator()
    
    try:
        results = comparator.run_comparison_test()
        
        if not results:
            print("❌ 没有可用的SDK进行测试")
            return 1
        
        comparator.print_summary(results)
        
        # 生成详细报告
        report = comparator.generate_comparison_report(results)
        
        # 保存报告
        report_file = "harborai_vs_openai_comparison_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存JSON数据
        json_data = {
            sdk_name: {
                'initialization_time_ms': metrics.initialization_time_ms,
                'method_call_overhead_us': metrics.method_call_overhead_us,
                'memory_usage_mb': metrics.memory_usage_mb,
                'concurrent_throughput_ops_per_sec': metrics.concurrent_throughput_ops_per_sec,
                'success_rate_percent': metrics.success_rate_percent,
                'error_count': metrics.error_count
            }
            for sdk_name, metrics in results.items()
        }
        
        json_file = "sdk_comparison_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        print(f"📄 JSON数据已保存到: {json_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())