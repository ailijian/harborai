#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""集成测试模块 - 多插件协作和错误恢复测试"""

import asyncio
import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any, List, Optional

from harborai.core.plugin_manager import PluginManager
from harborai.core.exceptions import (
    PluginError,
    PluginNotFoundError,
    PluginLoadError,
    PluginExecutionError,
    RetryableError,
    NonRetryableError,
    RateLimitError,
    DatabaseError
)
from harborai.core.retry import RetryManager, RetryConfig
from harborai.api.decorators import cost_tracking, with_postgres_logging
from harborai.monitoring.prometheus_metrics import PrometheusMetrics


class MockPlugin:
    """模拟插件基类"""
    
    def __init__(self, name: str, should_fail: bool = False, failure_count: int = 0):
        self.name = name
        self.should_fail = should_fail
        self.failure_count = failure_count
        self.call_count = 0
        self.execution_history = []
    
    def process(self, data: Any) -> Any:
        """处理数据"""
        self.call_count += 1
        self.execution_history.append({
            "call_count": self.call_count,
            "input_data": data,
            "timestamp": "mock_timestamp"
        })
        
        if self.should_fail and self.call_count <= self.failure_count:
            raise PluginExecutionError(f"Plugin {self.name} failed", plugin_name=self.name)
        
        return f"Processed by {self.name}: {data}"
    
    async def async_process(self, data: Any) -> Any:
        """异步处理数据"""
        await asyncio.sleep(0.01)  # 模拟异步操作
        return self.process(data)


class MockRetryablePlugin(MockPlugin):
    """模拟可重试失败的插件"""
    
    def process(self, data: Any) -> Any:
        self.call_count += 1
        self.execution_history.append({
            "call_count": self.call_count,
            "input_data": data,
            "timestamp": "mock_timestamp"
        })
        
        if self.should_fail and self.call_count <= self.failure_count:
            raise RateLimitError(f"Rate limited in {self.name}", retry_after=1)
        
        return f"Processed by {self.name}: {data}"


class MockNonRetryablePlugin(MockPlugin):
    """模拟不可重试失败的插件"""
    
    def process(self, data: Any) -> Any:
        self.call_count += 1
        self.execution_history.append({
            "call_count": self.call_count,
            "input_data": data,
            "timestamp": "mock_timestamp"
        })
        
        if self.should_fail:
            raise NonRetryableError(f"Non-retryable error in {self.name}")
        
        return f"Processed by {self.name}: {data}"


class TestMultiPluginCollaboration:
    """多插件协作测试"""
    
    def setup_method(self):
        """测试方法设置"""
        self.plugin_manager = PluginManager()
        self.retry_manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))
    
    def test_sequential_plugin_execution(self):
        """测试顺序插件执行"""
        # 创建多个插件
        plugin1 = MockPlugin("plugin1")
        plugin2 = MockPlugin("plugin2")
        plugin3 = MockPlugin("plugin3")
        
        plugins = [plugin1, plugin2, plugin3]
        
        # 模拟顺序执行
        data = "initial_data"
        for plugin in plugins:
            data = plugin.process(data)
        
        # 验证执行结果
        expected_result = "Processed by plugin3: Processed by plugin2: Processed by plugin1: initial_data"
        assert data == expected_result
        
        # 验证所有插件都被调用
        for plugin in plugins:
            assert plugin.call_count == 1
    
    @pytest.mark.asyncio
    async def test_parallel_plugin_execution(self):
        """测试并行插件执行"""
        # 创建多个插件
        plugins = [
            MockPlugin("plugin1"),
            MockPlugin("plugin2"),
            MockPlugin("plugin3")
        ]
        
        # 并行执行
        data = "test_data"
        tasks = [plugin.async_process(data) for plugin in plugins]
        results = await asyncio.gather(*tasks)
        
        # 验证结果
        expected_results = [
            "Processed by plugin1: test_data",
            "Processed by plugin2: test_data",
            "Processed by plugin3: test_data"
        ]
        assert results == expected_results
        
        # 验证所有插件都被调用
        for plugin in plugins:
            assert plugin.call_count == 1
    
    def test_plugin_chain_with_data_transformation(self):
        """测试插件链数据转换"""
        class TransformPlugin(MockPlugin):
            def __init__(self, name: str, transform_func):
                super().__init__(name)
                self.transform_func = transform_func
            
            def process(self, data: Any) -> Any:
                self.call_count += 1
                return self.transform_func(data)
        
        # 创建转换插件链
        plugins = [
            TransformPlugin("uppercase", lambda x: x.upper()),
            TransformPlugin("add_prefix", lambda x: f"PREFIX_{x}"),
            TransformPlugin("add_suffix", lambda x: f"{x}_SUFFIX")
        ]
        
        # 执行转换链
        data = "hello"
        for plugin in plugins:
            data = plugin.process(data)
        
        assert data == "PREFIX_HELLO_SUFFIX"
    
    def test_plugin_dependency_resolution(self):
        """测试插件依赖解析"""
        class DependentPlugin(MockPlugin):
            def __init__(self, name: str, dependencies: List[str] = None):
                super().__init__(name)
                self.dependencies = dependencies or []
                self.dependency_results = {}
            
            def set_dependency_result(self, dep_name: str, result: Any):
                self.dependency_results[dep_name] = result
            
            def process(self, data: Any) -> Any:
                self.call_count += 1
                # 检查依赖是否满足
                for dep in self.dependencies:
                    if dep not in self.dependency_results:
                        raise PluginExecutionError(f"Dependency {dep} not satisfied", plugin_name=self.name)
                
                # 合并依赖结果
                combined_data = {
                    "input": data,
                    "dependencies": self.dependency_results
                }
                return f"Processed by {self.name}: {combined_data}"
        
        # 创建有依赖关系的插件
        plugin_a = DependentPlugin("plugin_a")
        plugin_b = DependentPlugin("plugin_b", dependencies=["plugin_a"])
        plugin_c = DependentPlugin("plugin_c", dependencies=["plugin_a", "plugin_b"])
        
        # 按依赖顺序执行
        data = "test_data"
        
        result_a = plugin_a.process(data)
        plugin_b.set_dependency_result("plugin_a", result_a)
        
        result_b = plugin_b.process(data)
        plugin_c.set_dependency_result("plugin_a", result_a)
        plugin_c.set_dependency_result("plugin_b", result_b)
        
        result_c = plugin_c.process(data)
        
        # 验证执行顺序和依赖满足
        assert plugin_a.call_count == 1
        assert plugin_b.call_count == 1
        assert plugin_c.call_count == 1
        assert "plugin_a" in str(result_c)
        assert "plugin_b" in str(result_c)
    
    def test_plugin_communication_via_shared_context(self):
        """测试通过共享上下文的插件通信"""
        class ContextAwarePlugin(MockPlugin):
            def __init__(self, name: str, context: Dict[str, Any]):
                super().__init__(name)
                self.context = context
            
            def process(self, data: Any) -> Any:
                self.call_count += 1
                # 从上下文读取数据
                context_data = self.context.get(f"{self.name}_input", "")
                
                # 处理数据
                result = f"Processed by {self.name}: {data} + {context_data}"
                
                # 将结果写入上下文
                self.context[f"{self.name}_output"] = result
                
                return result
        
        # 创建共享上下文
        shared_context = {
            "plugin1_input": "context_data_1",
            "plugin2_input": "context_data_2"
        }
        
        # 创建上下文感知插件
        plugin1 = ContextAwarePlugin("plugin1", shared_context)
        plugin2 = ContextAwarePlugin("plugin2", shared_context)
        
        # 执行插件
        data = "test_data"
        result1 = plugin1.process(data)
        result2 = plugin2.process(data)
        
        # 验证上下文通信
        assert "context_data_1" in result1
        assert "context_data_2" in result2
        assert "plugin1_output" in shared_context
        assert "plugin2_output" in shared_context


class TestErrorRecoveryMechanisms:
    """错误恢复机制测试"""
    
    def setup_method(self):
        """测试方法设置"""
        self.retry_manager = RetryManager(RetryConfig(max_attempts=3, base_delay=0.01))
    
    def test_single_plugin_retry_recovery(self):
        """测试单个插件重试恢复"""
        # 创建会失败2次然后成功的插件
        plugin = MockRetryablePlugin("retryable_plugin", should_fail=True, failure_count=2)
        
        # 使用重试管理器执行
        result = self.retry_manager.execute(plugin.process, "test_data")
        
        # 验证最终成功
        assert result == "Processed by retryable_plugin: test_data"
        assert plugin.call_count == 3  # 失败2次 + 成功1次
    
    def test_plugin_chain_partial_failure_recovery(self):
        """测试插件链部分失败恢复"""
        # 创建插件链：成功 -> 失败(可重试) -> 成功
        plugin1 = MockPlugin("plugin1")
        plugin2 = MockRetryablePlugin("plugin2", should_fail=True, failure_count=1)
        plugin3 = MockPlugin("plugin3")
        
        plugins = [plugin1, plugin2, plugin3]
        
        def execute_plugin_chain(data):
            result = data
            for i, plugin in enumerate(plugins):
                if i == 1:  # 对第二个插件使用重试
                    result = self.retry_manager.execute(plugin.process, result)
                else:
                    result = plugin.process(result)
            return result
        
        result = execute_plugin_chain("test_data")
        
        # 验证链式执行成功
        expected = "Processed by plugin3: Processed by plugin2: Processed by plugin1: test_data"
        assert result == expected
        
        # 验证重试插件被调用了2次（1次失败 + 1次成功）
        assert plugin2.call_count == 2
    
    def test_plugin_fallback_mechanism(self):
        """测试插件回退机制"""
        # 主插件（会失败）
        primary_plugin = MockNonRetryablePlugin("primary", should_fail=True)
        # 备用插件（会成功）
        fallback_plugin = MockPlugin("fallback")
        
        def execute_with_fallback(data):
            try:
                return primary_plugin.process(data)
            except NonRetryableError:
                # 主插件失败，使用备用插件
                return fallback_plugin.process(data)
        
        result = execute_with_fallback("test_data")
        
        # 验证使用了备用插件
        assert result == "Processed by fallback: test_data"
        assert primary_plugin.call_count == 1
        assert fallback_plugin.call_count == 1
    
    def test_circuit_breaker_pattern(self):
        """测试断路器模式"""
        class CircuitBreakerPlugin(MockPlugin):
            def __init__(self, name: str, failure_threshold: int = 3):
                super().__init__(name)
                self.failure_threshold = failure_threshold
                self.failure_count = 0
                self.circuit_open = False
                self.last_failure_time = None
                self.recovery_timeout = 0.1  # 100ms恢复超时
            
            def process(self, data: Any) -> Any:
                import time
                
                # 检查断路器状态
                if self.circuit_open:
                    if (time.time() - self.last_failure_time) > self.recovery_timeout:
                        # 尝试恢复
                        self.circuit_open = False
                        self.failure_count = 0
                    else:
                        raise PluginExecutionError("Circuit breaker is open", plugin_name=self.name)
                
                self.call_count += 1
                
                # 模拟前几次调用失败
                if self.call_count <= 3:
                    self.failure_count += 1
                    if self.failure_count >= self.failure_threshold:
                        self.circuit_open = True
                        self.last_failure_time = time.time()
                    raise PluginExecutionError(f"Simulated failure {self.call_count}", plugin_name=self.name)
                
                # 成功执行
                return f"Processed by {self.name}: {data}"
        
        plugin = CircuitBreakerPlugin("circuit_breaker")
        
        # 前3次调用应该失败并触发断路器
        for i in range(3):
            with pytest.raises(PluginExecutionError):
                plugin.process("test_data")
        
        # 断路器应该已经打开
        assert plugin.circuit_open is True
        
        # 立即调用应该因为断路器打开而失败
        with pytest.raises(PluginExecutionError, match="Circuit breaker is open"):
            plugin.process("test_data")
        
        # 等待恢复超时
        import time
        time.sleep(0.15)
        
        # 现在应该可以成功执行
        result = plugin.process("test_data")
        assert result == "Processed by circuit_breaker: test_data"
    
    @pytest.mark.asyncio
    async def test_async_error_recovery(self):
        """测试异步错误恢复"""
        class AsyncRetryablePlugin:
            def __init__(self, name: str, failure_count: int = 2):
                self.name = name
                self.failure_count = failure_count
                self.call_count = 0
            
            async def async_process(self, data: Any) -> Any:
                await asyncio.sleep(0.01)
                self.call_count += 1
                
                if self.call_count <= self.failure_count:
                    raise DatabaseError(f"Async failure {self.call_count}")
                
                return f"Async processed by {self.name}: {data}"
        
        plugin = AsyncRetryablePlugin("async_plugin")
        
        # 使用异步重试管理器
        result = await self.retry_manager.execute_async(plugin.async_process, "test_data")
        
        assert result == "Async processed by async_plugin: test_data"
        assert plugin.call_count == 3  # 2次失败 + 1次成功
    
    def test_error_aggregation_and_reporting(self):
        """测试错误聚合和报告"""
        class ErrorReportingPlugin(MockPlugin):
            def __init__(self, name: str, error_collector: List[Exception]):
                super().__init__(name)
                self.error_collector = error_collector
            
            def process(self, data: Any) -> Any:
                self.call_count += 1
                try:
                    if self.call_count == 1:
                        raise RateLimitError("Rate limit error")
                    elif self.call_count == 2:
                        raise DatabaseError("Database error")
                    else:
                        return f"Processed by {self.name}: {data}"
                except Exception as e:
                    self.error_collector.append(e)
                    raise
        
        error_collector = []
        plugin = ErrorReportingPlugin("error_reporting", error_collector)
        
        # 执行并收集错误
        result = self.retry_manager.execute(plugin.process, "test_data")
        
        # 验证最终成功
        assert result == "Processed by error_reporting: test_data"
        
        # 验证错误被收集
        assert len(error_collector) == 2
        assert isinstance(error_collector[0], RateLimitError)
        assert isinstance(error_collector[1], DatabaseError)


class TestIntegrationWithMonitoring:
    """与监控系统的集成测试"""
    
    def setup_method(self):
        """测试方法设置"""
        self.token_stats = TokenStatisticsCollector()
        self.prometheus_metrics = PrometheusMetrics()
    
    def test_plugin_execution_with_metrics(self):
        """测试插件执行与指标收集"""
        plugin = MockPlugin("metrics_plugin")
        
        # 模拟带指标收集的插件执行
        def execute_with_metrics(data):
            import time
            start_time = time.time()
            
            try:
                result = plugin.process(data)
                
                # 记录成功指标
                duration = time.time() - start_time
                self.prometheus_metrics.record_api_request(
                    method="plugin_execution",
                    endpoint=f"/plugin/{plugin.name}",
                    status_code=200,
                    duration=duration
                )
                
                # 记录Token使用统计
                self.token_stats.record_usage(
                    model="mock_model",
                    input_tokens=10,
                    output_tokens=20,
                    duration=duration
                )
                
                return result
                
            except Exception as e:
                # 记录错误指标
                duration = time.time() - start_time
                self.prometheus_metrics.record_api_request(
                    method="plugin_execution",
                    endpoint=f"/plugin/{plugin.name}",
                    status_code=500,
                    duration=duration
                )
                raise
        
        result = execute_with_metrics("test_data")
        
        # 验证执行成功
        assert result == "Processed by metrics_plugin: test_data"
        
        # 验证指标被记录（这里只是验证方法被调用，实际指标值需要mock验证）
        assert plugin.call_count == 1
    
    @pytest.mark.asyncio
    async def test_async_plugin_chain_with_monitoring(self):
        """测试异步插件链与监控"""
        plugins = [
            MockPlugin("async_plugin1"),
            MockPlugin("async_plugin2"),
            MockPlugin("async_plugin3")
        ]
        
        async def execute_monitored_chain(data):
            results = []
            
            for plugin in plugins:
                import time
                start_time = time.time()
                
                try:
                    result = await plugin.async_process(data)
                    duration = time.time() - start_time
                    
                    # 记录每个插件的执行指标
                    self.prometheus_metrics.record_api_request(
                        method="async_plugin_execution",
                        endpoint=f"/async_plugin/{plugin.name}",
                        status_code=200,
                        duration=duration
                    )
                    
                    results.append(result)
                    data = result  # 链式传递数据
                    
                except Exception as e:
                    duration = time.time() - start_time
                    self.prometheus_metrics.record_api_request(
                        method="async_plugin_execution",
                        endpoint=f"/async_plugin/{plugin.name}",
                        status_code=500,
                        duration=duration
                    )
                    raise
            
            return results
        
        results = await execute_monitored_chain("initial_data")
        
        # 验证链式执行结果
        assert len(results) == 3
        assert "async_plugin1" in results[0]
        assert "async_plugin2" in results[1]
        assert "async_plugin3" in results[2]
        
        # 验证所有插件都被执行
        for plugin in plugins:
            assert plugin.call_count == 1


class TestRealWorldScenarios:
    """真实世界场景测试"""
    
    def test_data_processing_pipeline(self):
        """测试数据处理管道"""
        class DataValidationPlugin(MockPlugin):
            def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
                self.call_count += 1
                if not isinstance(data, dict) or "content" not in data:
                    raise PluginExecutionError("Invalid data format", plugin_name=self.name)
                return {**data, "validated": True}
        
        class DataEnrichmentPlugin(MockPlugin):
            def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
                self.call_count += 1
                return {**data, "enriched": True, "timestamp": "2024-01-01"}
        
        class DataTransformationPlugin(MockPlugin):
            def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
                self.call_count += 1
                content = data.get("content", "")
                return {**data, "transformed_content": content.upper()}
        
        # 创建数据处理管道
        pipeline = [
            DataValidationPlugin("validator"),
            DataEnrichmentPlugin("enricher"),
            DataTransformationPlugin("transformer")
        ]
        
        # 执行管道
        input_data = {"content": "hello world", "source": "test"}
        result = input_data
        
        for plugin in pipeline:
            result = plugin.process(result)
        
        # 验证管道执行结果
        assert result["validated"] is True
        assert result["enriched"] is True
        assert result["transformed_content"] == "HELLO WORLD"
        assert result["timestamp"] == "2024-01-01"
        assert result["source"] == "test"
        
        # 验证所有插件都被执行
        for plugin in pipeline:
            assert plugin.call_count == 1
    
    def test_api_request_processing_with_retries(self):
        """测试API请求处理与重试"""
        class APIRequestPlugin:
            def __init__(self, name: str):
                self.name = name
                self.call_count = 0
                self.rate_limit_count = 0
            
            def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
                self.call_count += 1
                
                # 模拟前2次请求遇到速率限制
                if self.call_count <= 2:
                    self.rate_limit_count += 1
                    raise RateLimitError(f"Rate limited (attempt {self.call_count})", retry_after=1)
                
                # 第3次请求成功
                return {
                    "status": "success",
                    "data": f"Processed request: {request_data}",
                    "attempts": self.call_count
                }
        
        plugin = APIRequestPlugin("api_processor")
        retry_manager = RetryManager(RetryConfig(max_attempts=5, base_delay=0.01))
        
        request_data = {"user_id": "123", "action": "get_data"}
        
        # 执行带重试的API请求
        result = retry_manager.execute(plugin.process_request, request_data)
        
        # 验证最终成功
        assert result["status"] == "success"
        assert result["attempts"] == 3
        assert plugin.call_count == 3
        assert plugin.rate_limit_count == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_plugin_execution_with_error_handling(self):
        """测试并发插件执行与错误处理"""
        class ConcurrentPlugin:
            def __init__(self, name: str, should_fail: bool = False, delay: float = 0.01):
                self.name = name
                self.should_fail = should_fail
                self.delay = delay
                self.call_count = 0
            
            async def async_process(self, data: Any) -> Dict[str, Any]:
                await asyncio.sleep(self.delay)
                self.call_count += 1
                
                if self.should_fail:
                    raise PluginExecutionError(f"Plugin {self.name} failed", plugin_name=self.name)
                
                return {
                    "plugin": self.name,
                    "result": f"Processed: {data}",
                    "call_count": self.call_count
                }
        
        # 创建混合插件（一些成功，一些失败）
        plugins = [
            ConcurrentPlugin("plugin1", should_fail=False),
            ConcurrentPlugin("plugin2", should_fail=True),
            ConcurrentPlugin("plugin3", should_fail=False),
            ConcurrentPlugin("plugin4", should_fail=True),
            ConcurrentPlugin("plugin5", should_fail=False)
        ]
        
        # 并发执行所有插件
        async def execute_plugin(plugin, data):
            try:
                return await plugin.async_process(data)
            except PluginExecutionError as e:
                return {"error": str(e), "plugin": plugin.name}
        
        tasks = [execute_plugin(plugin, "test_data") for plugin in plugins]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        # 分析结果
        successful_results = [r for r in results if "error" not in r]
        failed_results = [r for r in results if "error" in r]
        
        # 验证结果
        assert len(successful_results) == 3  # plugin1, plugin3, plugin5
        assert len(failed_results) == 2     # plugin2, plugin4
        
        # 验证成功的插件
        successful_plugins = [r["plugin"] for r in successful_results]
        assert "plugin1" in successful_plugins
        assert "plugin3" in successful_plugins
        assert "plugin5" in successful_plugins
        
        # 验证失败的插件
        failed_plugins = [r["plugin"] for r in failed_results]
        assert "plugin2" in failed_plugins
        assert "plugin4" in failed_plugins


if __name__ == "__main__":
    pytest.main([__file__, "-v"])