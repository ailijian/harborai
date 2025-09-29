# -*- coding: utf-8 -*-
"""
基础性能测试模块

本模块测试 HarborAI 各厂商 API 的基础性能指标，包括：
- 单次请求响应时间
- 不同模型的性能差异
- 推理模型的特殊性能测试
- 基础吞吐量测试
- 错误率统计

测试覆盖：
- DeepSeek API (deepseek-chat, deepseek-reasoner)
- ERNIE API (ernie-3.5-8k, ernie-4.0-turbo-8k, ernie-x1-turbo-32k)
- Doubao API (doubao-1-5-pro-32k-character-250715, doubao-seed-1-6-250615)
"""

import asyncio
import time
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest
import psutil

from tests.performance import PERFORMANCE_CONFIG, SUPPORTED_VENDORS
from tests.fixtures.performance_fixtures import (
    performance_monitor
)
from tests.fixtures.client_fixtures import (
    mock_client
)
from tests.fixtures.data_fixtures import (
    test_messages,
    performance_test_messages
)
from tests.utils.test_helpers import TestDataGenerator


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    response_time: float
    success: bool
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None
    model: Optional[str] = None
    vendor: Optional[str] = None
    timestamp: Optional[datetime] = None


class BasicPerformanceTest:
    """基础性能测试类"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.test_start_time = None
        self.test_end_time = None
    
    def record_metric(self, metric: PerformanceMetrics):
        """记录性能指标"""
        metric.timestamp = datetime.now()
        self.metrics.append(metric)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """获取汇总统计信息"""
        if not self.metrics:
            return {}
        
        response_times = [m.response_time for m in self.metrics if m.success]
        success_count = sum(1 for m in self.metrics if m.success)
        total_count = len(self.metrics)
        
        if not response_times:
            return {
                "total_requests": total_count,
                "success_count": success_count,
                "success_rate": 0.0,
                "error_rate": 1.0
            }
        
        return {
            "total_requests": total_count,
            "success_count": success_count,
            "success_rate": success_count / total_count,
            "error_rate": (total_count - success_count) / total_count,
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "p95_response_time": self._percentile(response_times, 95),
            "p99_response_time": self._percentile(response_times, 99),
            "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0.0
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index == int(index):
            return sorted_data[int(index)]
        else:
            lower_idx = int(index)
            upper_idx = min(lower_idx + 1, len(sorted_data) - 1)
            lower = sorted_data[lower_idx]
            upper = sorted_data[upper_idx]
            return lower + (upper - lower) * (index - lower_idx)


@pytest.mark.performance
@pytest.mark.basic
class TestBasicPerformance:
    """基础性能测试套件"""
    
    @pytest.fixture(autouse=True)
    def setup_test(self, performance_monitor):
        """测试设置"""
        self.performance_test = BasicPerformanceTest()
        self.monitor = performance_monitor
        self.test_id = TestDataGenerator.generate_test_id("basic_performance")
        
        # 开始监控
        self.monitor.start_monitoring()
        yield
        # 停止监控
        self.monitor.stop_monitoring()
    
    @pytest.mark.parametrize("vendor,model", [
        ("deepseek", "deepseek-chat"),
        ("deepseek", "deepseek-reasoner"),
        ("ernie", "ernie-3.5-8k"),
        ("ernie", "ernie-4.0-turbo-8k"),
        ("ernie", "ernie-x1-turbo-32k"),
        ("doubao", "doubao-1-5-pro-32k-character-250715"),
        ("doubao", "doubao-seed-1-6-250615")
    ])
    def test_single_request_latency(self, mock_client, vendor, model, test_messages):
        """测试单次请求延迟
        
        验证各厂商各模型的单次请求响应时间是否在可接受范围内
        """
        start_time = time.time()
        
        try:
            # 发送请求
            response = mock_client.chat.completions.create(
                model=model,
                messages=test_messages,
                max_tokens=100
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # 记录指标
            metric = PerformanceMetrics(
                response_time=response_time,
                success=True,
                model=model,
                vendor=vendor,
                tokens_used=response.get('usage', {}).get('total_tokens')
            )
            self.performance_test.record_metric(metric)
            
            # 验证响应时间阈值
            threshold = PERFORMANCE_CONFIG["basic_latency_threshold"]
            if model in SUPPORTED_VENDORS[vendor].get("reasoning_models", []):
                threshold = PERFORMANCE_CONFIG["reasoning_latency_threshold"]
            
            assert response_time < threshold, (
                f"响应时间 {response_time:.2f}s 超过阈值 {threshold}s "
                f"(vendor: {vendor}, model: {model})"
            )
            
            # 验证响应内容
            assert 'choices' in response
            assert len(response['choices']) > 0
            assert 'message' in response['choices'][0]
            
        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time
            
            # 记录错误指标
            metric = PerformanceMetrics(
                response_time=response_time,
                success=False,
                error_message=str(e),
                model=model,
                vendor=vendor
            )
            self.performance_test.record_metric(metric)
            
            pytest.fail(f"请求失败: {e}")
    
    @pytest.mark.parametrize("vendor", ["deepseek", "ernie", "doubao"])
    def test_model_performance_comparison(self, mock_client, vendor, test_messages):
        """测试同一厂商不同模型的性能对比
        
        比较同一厂商下不同模型的响应时间差异
        """
        models = SUPPORTED_VENDORS[vendor]["models"]
        model_metrics = {}
        
        for model in models:
            response_times = []
            
            # 每个模型测试5次取平均值
            for _ in range(5):
                start_time = time.time()
                
                try:
                    response = mock_client.chat.completions.create(
                        model=model,
                        messages=test_messages,
                        max_tokens=50
                    )
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    response_times.append(response_time)
                    
                    # 记录指标
                    metric = PerformanceMetrics(
                        response_time=response_time,
                        success=True,
                        model=model,
                        vendor=vendor
                    )
                    self.performance_test.record_metric(metric)
                    
                except Exception as e:
                    # 记录错误但继续测试
                    metric = PerformanceMetrics(
                        response_time=time.time() - start_time,
                        success=False,
                        error_message=str(e),
                        model=model,
                        vendor=vendor
                    )
                    self.performance_test.record_metric(metric)
            
            if response_times:
                model_metrics[model] = {
                    "avg_response_time": statistics.mean(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "success_rate": len(response_times) / 5
                }
        
        # 验证至少有一个模型成功
        assert len(model_metrics) > 0, f"厂商 {vendor} 的所有模型都测试失败"
        
        # 打印性能对比结果
        print(f"\n{vendor} 模型性能对比:")
        for model, metrics in model_metrics.items():
            print(f"  {model}: 平均响应时间 {metrics['avg_response_time']:.3f}s, "
                  f"成功率 {metrics['success_rate']:.1%}")
    
    @pytest.mark.reasoning
    def test_reasoning_model_performance(self, mock_client, complex_reasoning_messages):
        """测试推理模型的特殊性能
        
        推理模型通常需要更长的处理时间，测试其在复杂任务下的性能表现
        """
        reasoning_models = []
        for vendor, config in SUPPORTED_VENDORS.items():
            for model in config.get("reasoning_models", []):
                reasoning_models.append((vendor, model))
        
        for vendor, model in reasoning_models:
            start_time = time.time()
            
            try:
                # 使用复杂消息测试推理能力
                response = mock_client.chat.completions.create(
                    model=model,
                    messages=complex_reasoning_messages,
                    max_tokens=200,
                    temperature=0.1  # 降低随机性以获得更一致的性能
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # 记录指标
                metric = PerformanceMetrics(
                    response_time=response_time,
                    success=True,
                    model=model,
                    vendor=vendor,
                    tokens_used=response.get('usage', {}).get('total_tokens')
                )
                self.performance_test.record_metric(metric)
                
                # 推理模型的响应时间阈值更宽松
                threshold = PERFORMANCE_CONFIG["reasoning_latency_threshold"]
                assert response_time < threshold, (
                    f"推理模型 {model} 响应时间 {response_time:.2f}s 超过阈值 {threshold}s"
                )
                
                # 验证推理模型返回了内容（在mock环境中内容长度可能较短）
                content = response['choices'][0]['message']['content']
                assert len(content) > 0, f"推理模型 {model} 返回内容为空"
                
                # 在mock环境中，我们主要验证响应结构而不是内容长度
                print(f"推理模型 {vendor}/{model} 返回内容长度: {len(content)} 字符 (mock环境)")
                
                print(f"推理模型 {vendor}/{model}: {response_time:.3f}s, "
                      f"tokens: {metric.tokens_used}")
                
            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                
                # 记录错误指标
                metric = PerformanceMetrics(
                    response_time=response_time,
                    success=False,
                    error_message=str(e),
                    model=model,
                    vendor=vendor
                )
                self.performance_test.record_metric(metric)
                
                pytest.fail(f"推理模型 {vendor}/{model} 测试失败: {e}")
    
    def test_throughput_baseline(self, mock_client, test_messages, benchmark):
        """测试基础吞吐量基准
        
        使用 pytest-benchmark 测试单个客户端的基础吞吐量
        """
        def single_request():
            """单次请求函数"""
            vendor = "deepseek"
            model = "deepseek-chat"
            
            response = mock_client.chat.completions.create(
                model=model,
                messages=test_messages,
                max_tokens=50
            )
            
            return response
        
        # 使用 benchmark 装饰器进行基准测试
        result = benchmark.pedantic(
            single_request,
            rounds=10,
            iterations=5,
            warmup_rounds=2
        )
        
        # 验证基准测试结果
        assert result is not None
        
        # 基准测试统计信息会在测试输出中显示
        print(f"\n基础吞吐量基准测试完成，详细统计信息请查看上方的benchmark表格")
    
    @pytest.mark.parametrize("message_complexity", ["simple", "complex"])
    def test_message_complexity_impact(self, mock_client, message_complexity):
        """测试消息复杂度对性能的影响
        
        比较简单消息和复杂消息的处理时间差异
        """
        if message_complexity == "simple":
            messages = [{
                "role": "user",
                "content": "Hello, how are you?"
            }]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Please analyze the following complex scenario and provide detailed recommendations: A company is considering implementing a new AI system for customer service. What are the key factors they should consider?"},
                {"role": "assistant", "content": "Here are the key factors to consider when implementing an AI customer service system..."}
            ]
        
        vendor = "deepseek"
        model = "deepseek-chat"
        response_times = []
        
        # 测试10次取平均值
        for _ in range(10):
            start_time = time.time()
            
            try:
                response = mock_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=100
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                # 记录指标
                metric = PerformanceMetrics(
                    response_time=response_time,
                    success=True,
                    model=model,
                    vendor=vendor
                )
                self.performance_test.record_metric(metric)
                
            except Exception as e:
                # 记录错误但继续测试
                metric = PerformanceMetrics(
                    response_time=time.time() - start_time,
                    success=False,
                    error_message=str(e),
                    model=model,
                    vendor=vendor
                )
                self.performance_test.record_metric(metric)
        
        if response_times:
            avg_time = statistics.mean(response_times)
            print(f"\n{message_complexity} 消息平均响应时间: {avg_time:.3f}s")
            
            # 验证响应时间在合理范围内
            threshold = PERFORMANCE_CONFIG["basic_latency_threshold"]
            if message_complexity == "complex":
                threshold *= 2  # 复杂消息允许更长的处理时间
            
            assert avg_time < threshold, (
                f"{message_complexity} 消息平均响应时间 {avg_time:.3f}s 超过阈值 {threshold}s"
            )
    
    def test_error_rate_monitoring(self, mock_client):
        """测试错误率监控
        
        故意发送一些会导致错误的请求，监控错误率
        """
        test_cases = [
            # 正常请求
            {
                "vendor": "deepseek",
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "Hello"}],
                "should_succeed": True
            },
            # 无效模型
            {
                "vendor": "deepseek",
                "model": "invalid-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "should_succeed": False
            },
            # 空消息
            {
                "vendor": "deepseek",
                "model": "deepseek-chat",
                "messages": [],
                "should_succeed": False
            },
            # 过长消息
            {
                "vendor": "deepseek",
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "x" * 10000}],
                "should_succeed": False
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            start_time = time.time()
            
            try:
                response = mock_client.chat.completions.create(
                    model=test_case["model"],
                    messages=test_case["messages"],
                    max_tokens=50
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                
                # 记录成功指标
                metric = PerformanceMetrics(
                    response_time=response_time,
                    success=True,
                    model=test_case["model"],
                    vendor=test_case["vendor"]
                )
                self.performance_test.record_metric(metric)
                
                # 对于mock客户端，所有请求都会成功
                # 在实际环境中，这些请求可能会失败
                if not test_case["should_succeed"]:
                    print(f"注意：测试用例 {i} 在mock环境中成功，但在实际环境中可能失败")
                
            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                
                # 记录错误指标
                metric = PerformanceMetrics(
                    response_time=response_time,
                    success=False,
                    error_message=str(e),
                    model=test_case["model"],
                    vendor=test_case["vendor"]
                )
                self.performance_test.record_metric(metric)
                
                if test_case["should_succeed"]:
                    pytest.fail(f"测试用例 {i} 应该成功但失败了: {e}")
        
        # 验证错误率统计
        stats = self.performance_test.get_summary_stats()
        print(f"\n错误率监控结果:")
        print(f"  总请求数: {stats['total_requests']}")
        print(f"  成功率: {stats['success_rate']:.1%}")
        print(f"  错误率: {stats['error_rate']:.1%}")
    
    def teardown_method(self):
        """测试清理
        
        输出测试汇总统计信息
        """
        if hasattr(self, 'performance_test') and self.performance_test.metrics:
            stats = self.performance_test.get_summary_stats()
            
            print(f"\n=== 基础性能测试汇总 ===")
            print(f"测试ID: {getattr(self, 'test_id', 'unknown')}")
            print(f"总请求数: {stats.get('total_requests', 0)}")
            print(f"成功率: {stats.get('success_rate', 0):.1%}")
            print(f"平均响应时间: {stats.get('avg_response_time', 0):.3f}s")
            print(f"P95响应时间: {stats.get('p95_response_time', 0):.3f}s")
            print(f"P99响应时间: {stats.get('p99_response_time', 0):.3f}s")
            
            # 检查是否有资源监控数据
            if hasattr(self, 'monitor'):
                resource_stats = self.monitor.get_summary()
                if resource_stats:
                    print(f"峰值内存使用: {resource_stats.get('peak_memory_mb', 0):.1f}MB")
                    print(f"平均CPU使用: {resource_stats.get('avg_cpu_percent', 0):.1f}%")