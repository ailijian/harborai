#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发性能测试

验证并发优化组件的性能提升效果，目标是将并发吞吐量从505.6ops/s提升到≥1000ops/s。

测试策略：
1. 基准测试：使用传统方式测试当前性能
2. 并发优化测试：使用并发优化组件测试性能
3. 对比分析：验证性能提升效果
4. 压力测试：验证高并发场景下的稳定性

Assumptions:
- A1: 并发优化组件能够正确处理多个并发请求
- A2: 无锁数据结构能够提升并发性能
- A3: 异步连接池能够减少连接开销
- A4: 请求合并和批处理能够提升吞吐量
"""

import asyncio
import time
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
import pytest
import logging

# 测试用的模拟组件
from unittest.mock import Mock, AsyncMock, patch

# 导入被测试的组件
try:
    from harborai.api.fast_client import FastHarborAI, create_fast_client
    from harborai.core.optimizations.concurrency_manager import ConcurrencyManager, ConcurrencyConfig
    from harborai.core.optimizations.lockfree_plugin_manager import LockFreePluginManager
    from harborai.core.optimizations.async_request_processor import AsyncRequestProcessor
    from harborai.core.optimizations.optimized_connection_pool import OptimizedConnectionPool
    CONCURRENCY_OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    CONCURRENCY_OPTIMIZATION_AVAILABLE = False
    print(f"并发优化组件不可用: {e}")

logger = logging.getLogger(__name__)


class MockPlugin:
    """模拟插件，用于性能测试"""
    
    def __init__(self, response_time_ms: float = 100):
        """
        Args:
            response_time_ms: 模拟响应时间（毫秒）
        """
        self.response_time_ms = response_time_ms
        self.call_count = 0
        self.lock = threading.Lock()
    
    def chat_completion(self, messages: List[Dict], model: str, **kwargs) -> Dict[str, Any]:
        """模拟同步聊天完成"""
        with self.lock:
            self.call_count += 1
        
        # 模拟处理时间
        time.sleep(self.response_time_ms / 1000)
        
        return {
            "id": f"chatcmpl-{self.call_count}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"这是第{self.call_count}个响应"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
    
    async def chat_completion_async(self, messages: List[Dict], model: str, **kwargs) -> Dict[str, Any]:
        """模拟异步聊天完成"""
        with self.lock:
            self.call_count += 1
        
        # 模拟异步处理时间
        await asyncio.sleep(self.response_time_ms / 1000)
        
        return {
            "id": f"chatcmpl-async-{self.call_count}",
            "object": "chat.completion",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"这是第{self.call_count}个异步响应"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }


class ConcurrencyPerformanceTester:
    """并发性能测试器"""
    
    def __init__(self):
        self.mock_plugin = MockPlugin(response_time_ms=50)  # 50ms响应时间
        self.test_messages = [
            {"role": "user", "content": "你好，请介绍一下你自己"}
        ]
        self.test_model = "gpt-3.5-turbo"
    
    def setup_mock_plugin_manager(self):
        """设置模拟插件管理器"""
        def mock_get_plugin_for_model(model):
            return self.mock_plugin
        
        def mock_get_plugin_name_for_model(model):
            return "mock_plugin"
        
        # 模拟延迟插件管理器
        mock_manager = Mock()
        mock_manager.get_plugin_for_model = mock_get_plugin_for_model
        mock_manager.get_plugin_name_for_model = mock_get_plugin_name_for_model
        mock_manager.get_supported_models.return_value = [self.test_model]
        
        return mock_manager
    
    def test_traditional_sync_performance(self, num_requests: int = 100, num_threads: int = 10) -> Dict[str, float]:
        """测试传统同步方式的性能
        
        Args:
            num_requests: 总请求数
            num_threads: 线程数
            
        Returns:
            性能统计信息
        """
        logger.info(f"开始传统同步性能测试: {num_requests}个请求, {num_threads}个线程")
        
        # 创建客户端（禁用并发优化）
        config = {
            'enable_memory_optimization': False,
            'concurrency_optimization': {
                'max_concurrent_requests': 1  # 禁用并发优化
            }
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_get_manager.return_value = self.setup_mock_plugin_manager()
            
            client = create_fast_client(config=config)
            
            def make_request():
                """执行单个请求"""
                try:
                    response = client.chat.completions.create(
                        messages=self.test_messages,
                        model=self.test_model
                    )
                    return True, response
                except Exception as e:
                    return False, str(e)
            
            # 执行并发测试
            start_time = time.perf_counter()
            success_count = 0
            error_count = 0
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(make_request) for _ in range(num_requests)]
                
                for future in as_completed(futures):
                    success, result = future.result()
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        logger.error(f"请求失败: {result}")
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # 计算性能指标
            ops_per_second = num_requests / total_time
            avg_response_time = total_time / num_requests * 1000  # 毫秒
            
            stats = {
                'total_requests': num_requests,
                'success_count': success_count,
                'error_count': error_count,
                'total_time_seconds': total_time,
                'ops_per_second': ops_per_second,
                'avg_response_time_ms': avg_response_time,
                'success_rate': success_count / num_requests * 100
            }
            
            logger.info(f"传统同步性能测试结果: {ops_per_second:.2f} ops/s")
            return stats
    
    async def test_concurrency_async_performance(self, num_requests: int = 100, max_concurrent: int = 50) -> Dict[str, float]:
        """测试并发优化异步方式的性能
        
        Args:
            num_requests: 总请求数
            max_concurrent: 最大并发数
            
        Returns:
            性能统计信息
        """
        logger.info(f"开始并发优化异步性能测试: {num_requests}个请求, 最大并发{max_concurrent}")
        
        # 创建客户端（启用并发优化）
        config = {
            'enable_memory_optimization': True,
            'concurrency_optimization': {
                'max_concurrent_requests': max_concurrent,
                'connection_pool_size': 50,
                'request_timeout': 30.0,
                'enable_adaptive_optimization': True,
                'enable_health_check': True
            }
        }
        
        with patch('harborai.core.lazy_plugin_manager.get_lazy_plugin_manager') as mock_get_manager:
            mock_get_manager.return_value = self.setup_mock_plugin_manager()
            
            client = create_fast_client(config=config)
            
            async def make_async_request():
                """执行单个异步请求"""
                try:
                    response = await client.chat.completions.create_async(
                        messages=self.test_messages,
                        model=self.test_model
                    )
                    return True, response
                except Exception as e:
                    return False, str(e)
            
            # 执行异步并发测试
            start_time = time.perf_counter()
            success_count = 0
            error_count = 0
            
            # 使用信号量控制并发数
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def controlled_request():
                async with semaphore:
                    return await make_async_request()
            
            # 创建所有任务
            tasks = [controlled_request() for _ in range(num_requests)]
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                    logger.error(f"异步请求失败: {result}")
                else:
                    success, response = result
                    if success:
                        success_count += 1
                    else:
                        error_count += 1
                        logger.error(f"异步请求失败: {response}")
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # 计算性能指标
            ops_per_second = num_requests / total_time
            avg_response_time = total_time / num_requests * 1000  # 毫秒
            
            stats = {
                'total_requests': num_requests,
                'success_count': success_count,
                'error_count': error_count,
                'total_time_seconds': total_time,
                'ops_per_second': ops_per_second,
                'avg_response_time_ms': avg_response_time,
                'success_rate': success_count / num_requests * 100,
                'max_concurrent': max_concurrent
            }
            
            logger.info(f"并发优化异步性能测试结果: {ops_per_second:.2f} ops/s")
            return stats
    
    def compare_performance(self, traditional_stats: Dict, concurrent_stats: Dict) -> Dict[str, Any]:
        """比较性能结果
        
        Args:
            traditional_stats: 传统方式性能统计
            concurrent_stats: 并发优化性能统计
            
        Returns:
            性能对比结果
        """
        improvement_ratio = concurrent_stats['ops_per_second'] / traditional_stats['ops_per_second']
        response_time_improvement = traditional_stats['avg_response_time_ms'] / concurrent_stats['avg_response_time_ms']
        
        comparison = {
            'traditional_ops_per_second': traditional_stats['ops_per_second'],
            'concurrent_ops_per_second': concurrent_stats['ops_per_second'],
            'improvement_ratio': improvement_ratio,
            'improvement_percentage': (improvement_ratio - 1) * 100,
            'traditional_avg_response_time_ms': traditional_stats['avg_response_time_ms'],
            'concurrent_avg_response_time_ms': concurrent_stats['avg_response_time_ms'],
            'response_time_improvement_ratio': response_time_improvement,
            'target_achieved': concurrent_stats['ops_per_second'] >= 1000.0,
            'baseline_improvement': concurrent_stats['ops_per_second'] >= 505.6 * 1.5  # 至少50%提升
        }
        
        return comparison


# 测试用例
@pytest.mark.asyncio
@pytest.mark.performance
class TestConcurrencyPerformance:
    """并发性能测试用例"""
    
    def setup_method(self):
        """测试前设置"""
        self.tester = ConcurrencyPerformanceTester()
        logging.basicConfig(level=logging.INFO)
    
    def test_traditional_performance_baseline(self):
        """测试传统方式性能基准"""
        stats = self.tester.test_traditional_sync_performance(
            num_requests=100,
            num_threads=10
        )
        
        # 验证基本指标
        assert stats['success_rate'] >= 95.0, f"成功率过低: {stats['success_rate']}%"
        assert stats['ops_per_second'] > 0, "吞吐量必须大于0"
        
        logger.info(f"传统方式基准性能: {stats['ops_per_second']:.2f} ops/s")
    
    async def test_concurrent_performance_optimized(self):
        """测试并发优化性能"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        stats = await self.tester.test_concurrency_async_performance(
            num_requests=200,
            max_concurrent=50
        )
        
        # 验证基本指标
        assert stats['success_rate'] >= 95.0, f"成功率过低: {stats['success_rate']}%"
        assert stats['ops_per_second'] > 0, "吞吐量必须大于0"
        
        logger.info(f"并发优化性能: {stats['ops_per_second']:.2f} ops/s")
    
    async def test_performance_comparison_and_target_validation(self):
        """性能对比测试和目标验证"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        # 执行传统方式测试
        traditional_stats = self.tester.test_traditional_sync_performance(
            num_requests=100,
            num_threads=10
        )
        
        # 执行并发优化测试
        concurrent_stats = await self.tester.test_concurrency_async_performance(
            num_requests=200,
            max_concurrent=50
        )
        
        # 性能对比
        comparison = self.tester.compare_performance(traditional_stats, concurrent_stats)
        
        # 输出详细对比结果
        logger.info("=== 性能对比结果 ===")
        logger.info(f"传统方式: {comparison['traditional_ops_per_second']:.2f} ops/s")
        logger.info(f"并发优化: {comparison['concurrent_ops_per_second']:.2f} ops/s")
        logger.info(f"性能提升: {comparison['improvement_percentage']:.1f}%")
        logger.info(f"响应时间改善: {comparison['response_time_improvement_ratio']:.2f}x")
        logger.info(f"目标达成: {comparison['target_achieved']}")
        logger.info(f"基准提升: {comparison['baseline_improvement']}")
        
        # 验证性能目标
        assert comparison['improvement_ratio'] >= 1.5, f"性能提升不足50%: {comparison['improvement_percentage']:.1f}%"
        assert comparison['concurrent_ops_per_second'] >= 1000.0, f"未达到1000ops/s目标: {comparison['concurrent_ops_per_second']:.2f}"
        assert comparison['baseline_improvement'], f"未达到基准提升要求"
        
        return comparison
    
    async def test_high_concurrency_stress(self):
        """高并发压力测试"""
        if not CONCURRENCY_OPTIMIZATION_AVAILABLE:
            pytest.skip("并发优化组件不可用")
        
        # 高并发压力测试
        stress_stats = await self.tester.test_concurrency_async_performance(
            num_requests=500,
            max_concurrent=100
        )
        
        # 验证高并发下的稳定性
        assert stress_stats['success_rate'] >= 90.0, f"高并发下成功率过低: {stress_stats['success_rate']}%"
        assert stress_stats['ops_per_second'] >= 800.0, f"高并发下性能下降过多: {stress_stats['ops_per_second']:.2f} ops/s"
        
        logger.info(f"高并发压力测试结果: {stress_stats['ops_per_second']:.2f} ops/s, 成功率: {stress_stats['success_rate']:.1f}%")


if __name__ == "__main__":
    """直接运行性能测试"""
    import sys
    
    async def main():
        """主测试函数"""
        tester = ConcurrencyPerformanceTester()
        
        print("=== HarborAI 并发性能测试 ===")
        print()
        
        # 传统方式基准测试
        print("1. 传统方式基准测试...")
        traditional_stats = tester.test_traditional_sync_performance(
            num_requests=100,
            num_threads=10
        )
        print(f"   结果: {traditional_stats['ops_per_second']:.2f} ops/s")
        print()
        
        if CONCURRENCY_OPTIMIZATION_AVAILABLE:
            # 并发优化测试
            print("2. 并发优化测试...")
            concurrent_stats = await tester.test_concurrency_async_performance(
                num_requests=200,
                max_concurrent=50
            )
            print(f"   结果: {concurrent_stats['ops_per_second']:.2f} ops/s")
            print()
            
            # 性能对比
            print("3. 性能对比分析...")
            comparison = tester.compare_performance(traditional_stats, concurrent_stats)
            
            print(f"   传统方式: {comparison['traditional_ops_per_second']:.2f} ops/s")
            print(f"   并发优化: {comparison['concurrent_ops_per_second']:.2f} ops/s")
            print(f"   性能提升: {comparison['improvement_percentage']:.1f}%")
            print(f"   目标达成: {'✓' if comparison['target_achieved'] else '✗'}")
            print()
            
            # 高并发压力测试
            print("4. 高并发压力测试...")
            stress_stats = await tester.test_concurrency_async_performance(
                num_requests=500,
                max_concurrent=100
            )
            print(f"   结果: {stress_stats['ops_per_second']:.2f} ops/s, 成功率: {stress_stats['success_rate']:.1f}%")
            
        else:
            print("并发优化组件不可用，跳过优化测试")
        
        print()
        print("=== 测试完成 ===")
    
    # 运行测试
    asyncio.run(main())