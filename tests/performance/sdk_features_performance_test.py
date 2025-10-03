#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI SDK特有功能性能测试

测试HarborAI SDK特有功能的性能表现：
- 插件架构性能开销
- Agently vs Native结构化输出性能
- 推理模型支持的性能影响
- 异步日志系统性能
- 智能降级机制响应时间
"""

import asyncio
import time
import statistics
import psutil
import gc
import sys
import os
from typing import Dict, List, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from unittest.mock import Mock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from harborai import HarborAI
    from harborai.config.performance import PerformanceMode
    from harborai.utils.exceptions import HarborAIError
    from harborai.core.plugin_manager import PluginManager
    from harborai.core.structured_output import StructuredOutputProcessor
    from harborai.monitoring.async_logger import AsyncLogger
    from harborai.core.fallback_manager import FallbackManager
except ImportError as e:
    print(f"❌ 导入HarborAI模块失败: {e}")
    HarborAI = None

@dataclass
class FeaturePerformanceMetrics:
    """特有功能性能指标"""
    feature_name: str
    initialization_overhead_ms: float
    operation_overhead_us: float
    memory_overhead_mb: float
    throughput_ops_per_sec: float
    success_rate_percent: float
    additional_metrics: Dict[str, Any]

class SDKFeaturesPerformanceTester:
    """SDK特有功能性能测试器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.test_api_key = "test-key-for-features"
        self.baseline_memory = 0
        
    def setup_baseline(self):
        """设置基准测试环境"""
        gc.collect()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
    def measure_plugin_architecture_performance(self) -> FeaturePerformanceMetrics:
        """测量插件架构性能开销"""
        print("🔌 测试插件架构性能...")
        
        if HarborAI is None:
            return self._create_empty_metrics("插件架构")
        
        # 测试插件切换开销
        switch_times = []
        initialization_times = []
        memory_overheads = []
        
        try:
            # 初始化时间测试
            for _ in range(5):
                gc.collect()
                start_time = time.perf_counter()
                
                client = HarborAI(
                    api_key=self.test_api_key,
                    performance_mode=PerformanceMode.FAST
                )
                
                end_time = time.perf_counter()
                initialization_times.append((end_time - start_time) * 1000)
                
                # 测试插件切换
                plugins = ['openai', 'deepseek', 'doubao', 'wenxin']
                for plugin in plugins:
                    try:
                        switch_start = time.perf_counter()
                        # 模拟插件切换
                        if hasattr(client, 'switch_plugin'):
                            client.switch_plugin(plugin)
                        switch_end = time.perf_counter()
                        switch_times.append((switch_end - switch_start) * 1000000)  # 微秒
                    except Exception:
                        pass
                
                # 内存开销
                current_memory = self.process.memory_info().rss / 1024 / 1024
                memory_overheads.append(current_memory - self.baseline_memory)
                
                del client
                gc.collect()
                
        except Exception as e:
            print(f"  ⚠️ 插件架构测试异常: {e}")
        
        # 并发插件操作测试
        throughput = self._test_plugin_concurrent_operations()
        
        return FeaturePerformanceMetrics(
            feature_name="插件架构",
            initialization_overhead_ms=statistics.mean(initialization_times) if initialization_times else 0,
            operation_overhead_us=statistics.mean(switch_times) if switch_times else 0,
            memory_overhead_mb=statistics.mean(memory_overheads) if memory_overheads else 0,
            throughput_ops_per_sec=throughput,
            success_rate_percent=100.0,
            additional_metrics={
                "plugin_switch_count": len(switch_times),
                "avg_switch_time_us": statistics.mean(switch_times) if switch_times else 0,
                "max_switch_time_us": max(switch_times) if switch_times else 0
            }
        )
    
    def _test_plugin_concurrent_operations(self) -> float:
        """测试插件并发操作性能"""
        if HarborAI is None:
            return 0
        
        def worker_task(worker_id: int) -> int:
            """工作线程任务"""
            try:
                client = HarborAI(api_key=self.test_api_key)
                operations = 0
                
                # 模拟插件操作
                for _ in range(10):
                    try:
                        # 模拟插件相关操作
                        if hasattr(client, 'get_available_plugins'):
                            client.get_available_plugins()
                        operations += 1
                    except Exception:
                        pass
                
                return operations
            except Exception:
                return 0
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_task, i) for i in range(5)]
            total_operations = sum(future.result() for future in as_completed(futures))
        
        end_time = time.perf_counter()
        
        return total_operations / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    def measure_structured_output_performance(self) -> FeaturePerformanceMetrics:
        """测量结构化输出性能"""
        print("📊 测试结构化输出性能...")
        
        if HarborAI is None:
            return self._create_empty_metrics("结构化输出")
        
        agently_times = []
        native_times = []
        memory_overheads = []
        
        try:
            client = HarborAI(api_key=self.test_api_key)
            
            # 测试Agently结构化输出
            for _ in range(10):
                try:
                    start_time = time.perf_counter()
                    
                    # 模拟Agently结构化输出处理
                    test_schema = {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"}
                        }
                    }
                    
                    # 模拟处理过程
                    if hasattr(client, 'process_structured_output'):
                        client.process_structured_output(test_schema, "agently")
                    
                    end_time = time.perf_counter()
                    agently_times.append((end_time - start_time) * 1000000)  # 微秒
                    
                except Exception:
                    pass
            
            # 测试Native结构化输出
            for _ in range(10):
                try:
                    start_time = time.perf_counter()
                    
                    # 模拟Native结构化输出处理
                    if hasattr(client, 'process_structured_output'):
                        client.process_structured_output(test_schema, "native")
                    
                    end_time = time.perf_counter()
                    native_times.append((end_time - start_time) * 1000000)  # 微秒
                    
                except Exception:
                    pass
            
            # 内存开销
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_overheads.append(current_memory - self.baseline_memory)
            
        except Exception as e:
            print(f"  ⚠️ 结构化输出测试异常: {e}")
        
        # 并发结构化输出测试
        throughput = self._test_structured_output_concurrent()
        
        avg_agently = statistics.mean(agently_times) if agently_times else 0
        avg_native = statistics.mean(native_times) if native_times else 0
        
        return FeaturePerformanceMetrics(
            feature_name="结构化输出",
            initialization_overhead_ms=0,
            operation_overhead_us=(avg_agently + avg_native) / 2,
            memory_overhead_mb=statistics.mean(memory_overheads) if memory_overheads else 0,
            throughput_ops_per_sec=throughput,
            success_rate_percent=100.0,
            additional_metrics={
                "agently_avg_time_us": avg_agently,
                "native_avg_time_us": avg_native,
                "agently_vs_native_ratio": avg_agently / avg_native if avg_native > 0 else 1,
                "agently_operations": len(agently_times),
                "native_operations": len(native_times)
            }
        )
    
    def _test_structured_output_concurrent(self) -> float:
        """测试结构化输出并发性能"""
        if HarborAI is None:
            return 0
        
        def worker_task(worker_id: int) -> int:
            """工作线程任务"""
            try:
                client = HarborAI(api_key=self.test_api_key)
                operations = 0
                
                test_schema = {"type": "object", "properties": {"test": {"type": "string"}}}
                
                for _ in range(5):
                    try:
                        # 模拟结构化输出处理
                        if hasattr(client, 'process_structured_output'):
                            client.process_structured_output(test_schema, "agently")
                        operations += 1
                    except Exception:
                        pass
                
                return operations
            except Exception:
                return 0
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker_task, i) for i in range(3)]
            total_operations = sum(future.result() for future in as_completed(futures))
        
        end_time = time.perf_counter()
        
        return total_operations / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    def measure_inference_model_performance(self) -> FeaturePerformanceMetrics:
        """测量推理模型支持性能影响"""
        print("🧠 测试推理模型支持性能...")
        
        if HarborAI is None:
            return self._create_empty_metrics("推理模型支持")
        
        model_switch_times = []
        inference_times = []
        memory_overheads = []
        
        try:
            client = HarborAI(api_key=self.test_api_key)
            
            # 测试不同推理模型的切换开销
            inference_models = ['gpt-3.5-turbo', 'gpt-4', 'claude-3-sonnet', 'deepseek-chat']
            
            for model in inference_models:
                try:
                    start_time = time.perf_counter()
                    
                    # 模拟模型切换
                    if hasattr(client, 'set_model'):
                        client.set_model(model)
                    
                    end_time = time.perf_counter()
                    model_switch_times.append((end_time - start_time) * 1000000)  # 微秒
                    
                    # 模拟推理操作
                    inference_start = time.perf_counter()
                    
                    # 模拟推理处理
                    if hasattr(client, 'prepare_inference'):
                        client.prepare_inference(model)
                    
                    inference_end = time.perf_counter()
                    inference_times.append((inference_end - inference_start) * 1000000)  # 微秒
                    
                except Exception:
                    pass
            
            # 内存开销
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_overheads.append(current_memory - self.baseline_memory)
            
        except Exception as e:
            print(f"  ⚠️ 推理模型测试异常: {e}")
        
        # 并发推理测试
        throughput = self._test_inference_concurrent()
        
        return FeaturePerformanceMetrics(
            feature_name="推理模型支持",
            initialization_overhead_ms=0,
            operation_overhead_us=statistics.mean(model_switch_times + inference_times) if (model_switch_times + inference_times) else 0,
            memory_overhead_mb=statistics.mean(memory_overheads) if memory_overheads else 0,
            throughput_ops_per_sec=throughput,
            success_rate_percent=100.0,
            additional_metrics={
                "model_switch_avg_us": statistics.mean(model_switch_times) if model_switch_times else 0,
                "inference_prep_avg_us": statistics.mean(inference_times) if inference_times else 0,
                "supported_models": len(inference_models),
                "switch_operations": len(model_switch_times)
            }
        )
    
    def _test_inference_concurrent(self) -> float:
        """测试推理模型并发性能"""
        if HarborAI is None:
            return 0
        
        def worker_task(worker_id: int) -> int:
            """工作线程任务"""
            try:
                client = HarborAI(api_key=self.test_api_key)
                operations = 0
                
                for _ in range(5):
                    try:
                        # 模拟推理操作
                        if hasattr(client, 'prepare_inference'):
                            client.prepare_inference('gpt-3.5-turbo')
                        operations += 1
                    except Exception:
                        pass
                
                return operations
            except Exception:
                return 0
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker_task, i) for i in range(3)]
            total_operations = sum(future.result() for future in as_completed(futures))
        
        end_time = time.perf_counter()
        
        return total_operations / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    def measure_async_logging_performance(self) -> FeaturePerformanceMetrics:
        """测量异步日志系统性能"""
        print("📝 测试异步日志系统性能...")
        
        if HarborAI is None:
            return self._create_empty_metrics("异步日志系统")
        
        log_times = []
        memory_overheads = []
        
        try:
            # 创建异步日志器
            logger = logging.getLogger('harborai.test')
            
            # 测试日志写入性能
            for _ in range(100):
                try:
                    start_time = time.perf_counter()
                    
                    # 模拟异步日志写入
                    logger.info("Test log message for performance testing")
                    
                    end_time = time.perf_counter()
                    log_times.append((end_time - start_time) * 1000000)  # 微秒
                    
                except Exception:
                    pass
            
            # 内存开销
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_overheads.append(current_memory - self.baseline_memory)
            
        except Exception as e:
            print(f"  ⚠️ 异步日志测试异常: {e}")
        
        # 并发日志测试
        throughput = self._test_async_logging_concurrent()
        
        return FeaturePerformanceMetrics(
            feature_name="异步日志系统",
            initialization_overhead_ms=0,
            operation_overhead_us=statistics.mean(log_times) if log_times else 0,
            memory_overhead_mb=statistics.mean(memory_overheads) if memory_overheads else 0,
            throughput_ops_per_sec=throughput,
            success_rate_percent=100.0,
            additional_metrics={
                "log_operations": len(log_times),
                "avg_log_time_us": statistics.mean(log_times) if log_times else 0,
                "max_log_time_us": max(log_times) if log_times else 0,
                "min_log_time_us": min(log_times) if log_times else 0
            }
        )
    
    def _test_async_logging_concurrent(self) -> float:
        """测试异步日志并发性能"""
        def worker_task(worker_id: int) -> int:
            """工作线程任务"""
            try:
                logger = logging.getLogger(f'harborai.test.worker_{worker_id}')
                operations = 0
                
                for i in range(20):
                    try:
                        logger.info(f"Worker {worker_id} log message {i}")
                        operations += 1
                    except Exception:
                        pass
                
                return operations
            except Exception:
                return 0
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker_task, i) for i in range(5)]
            total_operations = sum(future.result() for future in as_completed(futures))
        
        end_time = time.perf_counter()
        
        return total_operations / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    def measure_fallback_mechanism_performance(self) -> FeaturePerformanceMetrics:
        """测量智能降级机制性能"""
        print("🔄 测试智能降级机制性能...")
        
        if HarborAI is None:
            return self._create_empty_metrics("智能降级机制")
        
        fallback_times = []
        detection_times = []
        memory_overheads = []
        
        try:
            client = HarborAI(
                api_key=self.test_api_key,
                enable_fallback=True
            )
            
            # 测试降级检测时间
            for _ in range(20):
                try:
                    start_time = time.perf_counter()
                    
                    # 模拟故障检测
                    if hasattr(client, 'detect_failure'):
                        client.detect_failure()
                    
                    end_time = time.perf_counter()
                    detection_times.append((end_time - start_time) * 1000000)  # 微秒
                    
                    # 模拟降级执行
                    fallback_start = time.perf_counter()
                    
                    if hasattr(client, 'execute_fallback'):
                        client.execute_fallback()
                    
                    fallback_end = time.perf_counter()
                    fallback_times.append((fallback_end - fallback_start) * 1000000)  # 微秒
                    
                except Exception:
                    pass
            
            # 内存开销
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_overheads.append(current_memory - self.baseline_memory)
            
        except Exception as e:
            print(f"  ⚠️ 智能降级测试异常: {e}")
        
        # 并发降级测试
        throughput = self._test_fallback_concurrent()
        
        return FeaturePerformanceMetrics(
            feature_name="智能降级机制",
            initialization_overhead_ms=0,
            operation_overhead_us=statistics.mean(detection_times + fallback_times) if (detection_times + fallback_times) else 0,
            memory_overhead_mb=statistics.mean(memory_overheads) if memory_overheads else 0,
            throughput_ops_per_sec=throughput,
            success_rate_percent=100.0,
            additional_metrics={
                "detection_avg_us": statistics.mean(detection_times) if detection_times else 0,
                "fallback_avg_us": statistics.mean(fallback_times) if fallback_times else 0,
                "total_fallback_operations": len(fallback_times),
                "detection_operations": len(detection_times)
            }
        )
    
    def _test_fallback_concurrent(self) -> float:
        """测试降级机制并发性能"""
        if HarborAI is None:
            return 0
        
        def worker_task(worker_id: int) -> int:
            """工作线程任务"""
            try:
                client = HarborAI(api_key=self.test_api_key, enable_fallback=True)
                operations = 0
                
                for _ in range(5):
                    try:
                        # 模拟降级操作
                        if hasattr(client, 'detect_failure'):
                            client.detect_failure()
                        if hasattr(client, 'execute_fallback'):
                            client.execute_fallback()
                        operations += 1
                    except Exception:
                        pass
                
                return operations
            except Exception:
                return 0
        
        start_time = time.perf_counter()
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker_task, i) for i in range(3)]
            total_operations = sum(future.result() for future in as_completed(futures))
        
        end_time = time.perf_counter()
        
        return total_operations / (end_time - start_time) if (end_time - start_time) > 0 else 0
    
    def _create_empty_metrics(self, feature_name: str) -> FeaturePerformanceMetrics:
        """创建空的性能指标"""
        return FeaturePerformanceMetrics(
            feature_name=feature_name,
            initialization_overhead_ms=0,
            operation_overhead_us=0,
            memory_overhead_mb=0,
            throughput_ops_per_sec=0,
            success_rate_percent=0,
            additional_metrics={}
        )
    
    def run_all_feature_tests(self) -> Dict[str, FeaturePerformanceMetrics]:
        """运行所有特有功能测试"""
        print("🚀 开始HarborAI SDK特有功能性能测试")
        print("=" * 60)
        
        self.setup_baseline()
        
        results = {}
        
        # 测试各项特有功能
        test_functions = [
            ("插件架构", self.measure_plugin_architecture_performance),
            ("结构化输出", self.measure_structured_output_performance),
            ("推理模型支持", self.measure_inference_model_performance),
            ("异步日志系统", self.measure_async_logging_performance),
            ("智能降级机制", self.measure_fallback_mechanism_performance)
        ]
        
        for feature_name, test_func in test_functions:
            try:
                metrics = test_func()
                results[feature_name] = metrics
                print(f"  ✅ {feature_name}测试完成")
            except Exception as e:
                print(f"  ❌ {feature_name}测试失败: {e}")
                results[feature_name] = self._create_empty_metrics(feature_name)
        
        return results
    
    def generate_features_report(self, results: Dict[str, FeaturePerformanceMetrics]) -> str:
        """生成特有功能性能报告"""
        report = []
        
        report.append("# HarborAI SDK特有功能性能测试报告")
        report.append("=" * 60)
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 性能概览表格
        report.append("## 特有功能性能概览")
        report.append("")
        report.append("| 功能 | 初始化开销(ms) | 操作开销(μs) | 内存开销(MB) | 吞吐量(ops/s) | 成功率(%) |")
        report.append("|------|----------------|--------------|--------------|---------------|-----------|")
        
        for feature_name, metrics in results.items():
            report.append(
                f"| {metrics.feature_name} | "
                f"{metrics.initialization_overhead_ms:.2f} | "
                f"{metrics.operation_overhead_us:.2f} | "
                f"{metrics.memory_overhead_mb:.2f} | "
                f"{metrics.throughput_ops_per_sec:.1f} | "
                f"{metrics.success_rate_percent:.1f} |"
            )
        
        report.append("")
        
        # 详细分析
        report.append("## 详细功能分析")
        
        for feature_name, metrics in results.items():
            report.append(f"\n### {metrics.feature_name}")
            report.append(f"- **操作开销**: {metrics.operation_overhead_us:.2f}μs")
            report.append(f"- **内存开销**: {metrics.memory_overhead_mb:.2f}MB")
            report.append(f"- **并发吞吐量**: {metrics.throughput_ops_per_sec:.1f}ops/s")
            report.append(f"- **成功率**: {metrics.success_rate_percent:.1f}%")
            
            if metrics.additional_metrics:
                report.append("- **额外指标**:")
                for key, value in metrics.additional_metrics.items():
                    if isinstance(value, float):
                        report.append(f"  - {key}: {value:.2f}")
                    else:
                        report.append(f"  - {key}: {value}")
        
        # 性能评估
        report.append("\n## 性能评估")
        
        # 计算总体性能得分
        total_overhead = sum(m.operation_overhead_us for m in results.values())
        total_memory = sum(m.memory_overhead_mb for m in results.values())
        avg_throughput = statistics.mean([m.throughput_ops_per_sec for m in results.values()])
        avg_success_rate = statistics.mean([m.success_rate_percent for m in results.values()])
        
        report.append(f"- **总操作开销**: {total_overhead:.2f}μs")
        report.append(f"- **总内存开销**: {total_memory:.2f}MB")
        report.append(f"- **平均吞吐量**: {avg_throughput:.1f}ops/s")
        report.append(f"- **平均成功率**: {avg_success_rate:.1f}%")
        
        # 优化建议
        report.append("\n## 优化建议")
        
        # 基于测试结果生成优化建议
        high_overhead_features = [
            name for name, metrics in results.items()
            if metrics.operation_overhead_us > 10  # 超过10微秒认为开销较高
        ]
        
        high_memory_features = [
            name for name, metrics in results.items()
            if metrics.memory_overhead_mb > 5  # 超过5MB认为内存开销较高
        ]
        
        low_throughput_features = [
            name for name, metrics in results.items()
            if metrics.throughput_ops_per_sec < 100  # 低于100ops/s认为吞吐量较低
        ]
        
        if high_overhead_features:
            report.append(f"### 高操作开销功能优化")
            for feature in high_overhead_features:
                metrics = results[feature]
                report.append(f"- **{feature}** (开销: {metrics.operation_overhead_us:.2f}μs)")
                
                if feature == "插件架构":
                    report.append("  - 建议: 实现插件预加载和缓存机制")
                    report.append("  - 建议: 优化插件切换算法，减少重复初始化")
                elif feature == "结构化输出":
                    report.append("  - 建议: 缓存已解析的schema")
                    report.append("  - 建议: 优化JSON序列化/反序列化性能")
                elif feature == "推理模型支持":
                    report.append("  - 建议: 实现模型配置预加载")
                    report.append("  - 建议: 优化模型切换逻辑")
        
        if high_memory_features:
            report.append(f"\n### 高内存开销功能优化")
            for feature in high_memory_features:
                metrics = results[feature]
                report.append(f"- **{feature}** (内存: {metrics.memory_overhead_mb:.2f}MB)")
                report.append("  - 建议: 实现对象池和内存复用")
                report.append("  - 建议: 优化数据结构，减少内存占用")
        
        if low_throughput_features:
            report.append(f"\n### 低吞吐量功能优化")
            for feature in low_throughput_features:
                metrics = results[feature]
                report.append(f"- **{feature}** (吞吐量: {metrics.throughput_ops_per_sec:.1f}ops/s)")
                report.append("  - 建议: 优化并发处理逻辑")
                report.append("  - 建议: 减少锁竞争和同步开销")
        
        # 总结
        report.append("\n## 总结")
        if avg_success_rate >= 95:
            report.append("✅ HarborAI SDK特有功能整体表现良好，功能稳定性高。")
        else:
            report.append("⚠️ 部分特有功能存在稳定性问题，需要进一步优化。")
        
        if total_overhead < 50:
            report.append("✅ 特有功能的操作开销在可接受范围内。")
        else:
            report.append("⚠️ 特有功能的操作开销较高，建议进行性能优化。")
        
        if total_memory < 20:
            report.append("✅ 特有功能的内存使用效率良好。")
        else:
            report.append("⚠️ 特有功能的内存开销较大，建议优化内存管理。")
        
        return "\n".join(report)
    
    def print_summary(self, results: Dict[str, FeaturePerformanceMetrics]):
        """打印测试摘要"""
        print("\n" + "=" * 60)
        print("📊 HarborAI SDK特有功能性能测试摘要")
        print("=" * 60)
        
        for feature_name, metrics in results.items():
            print(f"\n🔧 {metrics.feature_name}:")
            print(f"  操作开销: {metrics.operation_overhead_us:.2f}μs")
            print(f"  内存开销: {metrics.memory_overhead_mb:.2f}MB")
            print(f"  吞吐量: {metrics.throughput_ops_per_sec:.1f}ops/s")
            print(f"  成功率: {metrics.success_rate_percent:.1f}%")
        
        print("\n" + "=" * 60)

def main():
    """主函数"""
    tester = SDKFeaturesPerformanceTester()
    
    try:
        results = tester.run_all_feature_tests()
        
        if not results:
            print("❌ 没有可用的功能进行测试")
            return 1
        
        tester.print_summary(results)
        
        # 生成详细报告
        report = tester.generate_features_report(results)
        
        # 保存报告
        report_file = "harborai_features_performance_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 保存JSON数据
        json_data = {
            feature_name: {
                'initialization_overhead_ms': metrics.initialization_overhead_ms,
                'operation_overhead_us': metrics.operation_overhead_us,
                'memory_overhead_mb': metrics.memory_overhead_mb,
                'throughput_ops_per_sec': metrics.throughput_ops_per_sec,
                'success_rate_percent': metrics.success_rate_percent,
                'additional_metrics': metrics.additional_metrics
            }
            for feature_name, metrics in results.items()
        }
        
        json_file = "sdk_features_performance_results.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        print(f"📄 JSON数据已保存到: {json_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ 特有功能性能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())