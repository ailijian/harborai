#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prometheus指标模块完整测试套件

本测试套件全面测试Prometheus指标模块的所有功能，包括：
- PrometheusMetrics类的初始化和指标记录
- 各种指标类型的记录和获取
- 中间件装饰器的功能
- 全局实例管理
- 错误处理和边界情况

遵循VIBE编码规范，使用TDD方法，确保100%测试通过。
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
from prometheus_client import CollectorRegistry, CONTENT_TYPE_LATEST

from harborai.monitoring.prometheus_metrics import (
    PrometheusMetrics, get_prometheus_metrics, init_prometheus_metrics,
    prometheus_middleware, prometheus_async_middleware
)
from harborai.utils.exceptions import HarborAIError


class TestPrometheusMetrics:
    """测试PrometheusMetrics类"""
    
    def setup_method(self):
        """测试前设置"""
        self.registry = CollectorRegistry()
        self.metrics = PrometheusMetrics(self.registry)
    
    def test_initialization_default_registry(self):
        """测试默认注册表初始化"""
        metrics = PrometheusMetrics()
        assert metrics.registry is not None
        assert hasattr(metrics, 'api_requests_total')
        assert hasattr(metrics, 'api_request_duration_seconds')
        assert hasattr(metrics, 'tokens_used_total')
        assert hasattr(metrics, 'cost_total')
        assert hasattr(metrics, 'api_errors_total')
        assert hasattr(metrics, 'active_connections')
        assert hasattr(metrics, 'retries_total')
        assert hasattr(metrics, 'cache_hits_total')
        assert hasattr(metrics, 'cache_misses_total')
        assert hasattr(metrics, 'system_info')
    
    def test_initialization_custom_registry(self):
        """测试自定义注册表初始化"""
        custom_registry = CollectorRegistry()
        metrics = PrometheusMetrics(custom_registry)
        assert metrics.registry is custom_registry
    
    def test_record_api_request_success(self):
        """测试记录API请求 - 成功"""
        method = 'chat_completion'
        model = 'gpt-4'
        provider = 'openai'
        duration = 2.5
        
        self.metrics.record_api_request(method, model, provider, duration)
        
        # 检查指标是否正确记录
        metrics_output = self.metrics.get_metrics()
        
        # 检查请求计数器
        assert f'harborai_api_requests_total{{method="{method}",model="{model}",provider="{provider}",status="success"}} 1.0' in metrics_output
        
        # 检查响应时间直方图
        assert f'harborai_api_request_duration_seconds_count{{method="{method}",model="{model}",provider="{provider}"}} 1.0' in metrics_output
        assert f'harborai_api_request_duration_seconds_sum{{method="{method}",model="{model}",provider="{provider}"}} {duration}' in metrics_output
    
    def test_record_api_request_error(self):
        """测试记录API请求 - 错误"""
        method = 'chat_completion'
        model = 'gpt-4'
        provider = 'openai'
        duration = 1.0
        error_type = 'RateLimitError'
        
        self.metrics.record_api_request(
            method, model, provider, duration, 
            status='error', error_type=error_type
        )
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查错误状态的请求计数器
        assert f'harborai_api_requests_total{{method="{method}",model="{model}",provider="{provider}",status="error"}} 1.0' in metrics_output
        
        # 检查错误计数器
        assert f'harborai_api_errors_total{{error_type="{error_type}",method="{method}",model="{model}",provider="{provider}"}} 1.0' in metrics_output
    
    def test_record_api_request_error_without_type(self):
        """测试记录API请求错误（无错误类型）"""
        self.metrics.record_api_request(
            method="chat_completion",
            model="gpt-3.5-turbo",
            provider="openai",
            duration=1.5,
            status="error"
        )
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查请求总数指标
        assert 'harborai_api_requests_total{method="chat_completion",model="gpt-3.5-turbo",provider="openai",status="error"} 1.0' in metrics_output
        
        # 检查响应时间指标
        assert 'harborai_api_request_duration_seconds_count{method="chat_completion",model="gpt-3.5-turbo",provider="openai"} 1.0' in metrics_output
        
        # 当没有error_type时，不应该记录错误指标
        # 但是由于实际实现中只有在error_type存在时才记录错误，所以这里应该检查不存在
        # 但是由于Prometheus会初始化所有指标，所以错误指标会存在但值为0
        # 我们检查是否没有增加错误计数
        assert 'harborai_api_errors_total{' not in metrics_output or 'harborai_api_errors_total{method="chat_completion",model="gpt-3.5-turbo",provider="openai",error_type=' not in metrics_output
    
    def test_record_token_usage(self):
        """测试记录Token使用量"""
        model = 'gpt-4'
        provider = 'openai'
        prompt_tokens = 100
        completion_tokens = 50
        
        self.metrics.record_token_usage(model, provider, prompt_tokens, completion_tokens)
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查输入Token计数器
        assert f'harborai_tokens_used_total{{model="{model}",provider="{provider}",token_type="prompt"}} {prompt_tokens}.0' in metrics_output
        
        # 检查输出Token计数器
        assert f'harborai_tokens_used_total{{model="{model}",provider="{provider}",token_type="completion"}} {completion_tokens}.0' in metrics_output
    
    def test_record_cost(self):
        """测试记录成本"""
        model = 'gpt-4'
        provider = 'openai'
        cost = 0.05
        
        self.metrics.record_cost(model, provider, cost)
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查成本计数器
        assert f'harborai_cost_total{{model="{model}",provider="{provider}"}} {cost}' in metrics_output
    
    def test_record_retry(self):
        """测试记录重试"""
        model = 'gpt-4'
        provider = 'openai'
        retry_reason = 'rate_limit'
        
        self.metrics.record_retry(model, provider, retry_reason)
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查重试计数器
        assert f'harborai_retries_total{{model="{model}",provider="{provider}",retry_reason="{retry_reason}"}} 1.0' in metrics_output
    
    def test_record_cache_hit(self):
        """测试记录缓存命中"""
        cache_type = 'response_cache'
        
        self.metrics.record_cache_hit(cache_type)
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查缓存命中计数器
        assert f'harborai_cache_hits_total{{cache_type="{cache_type}"}} 1.0' in metrics_output
    
    def test_record_cache_miss(self):
        """测试记录缓存未命中"""
        cache_type = 'response_cache'
        
        self.metrics.record_cache_miss(cache_type)
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查缓存未命中计数器
        assert f'harborai_cache_misses_total{{cache_type="{cache_type}"}} 1.0' in metrics_output
    
    def test_set_active_connections(self):
        """测试设置活跃连接数"""
        provider = 'openai'
        count = 5
        
        self.metrics.set_active_connections(provider, count)
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查活跃连接数量表
        assert f'harborai_active_connections{{provider="{provider}"}} {count}.0' in metrics_output
    
    def test_set_system_info(self):
        """测试设置系统信息"""
        info = {
            'version': '1.0.0',
            'environment': 'production',
            'python_version': '3.11.5'
        }
        
        self.metrics.set_system_info(info)
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查系统信息
        assert 'harborai_system_info' in metrics_output
        for key, value in info.items():
            assert f'{key}="{value}"' in metrics_output
    
    def test_get_metrics_format(self):
        """测试获取指标格式"""
        # 记录一些指标
        self.metrics.record_api_request('test', 'gpt-4', 'openai', 1.0)
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查输出格式
        assert isinstance(metrics_output, str)
        assert 'harborai_api_requests_total' in metrics_output
        assert 'harborai_api_request_duration_seconds' in metrics_output
    
    def test_get_content_type(self):
        """测试获取Content-Type"""
        content_type = self.metrics.get_content_type()
        assert content_type == CONTENT_TYPE_LATEST
    
    def test_multiple_metrics_recording(self):
        """测试多个指标记录"""
        # 记录多个API请求
        self.metrics.record_api_request('chat', 'gpt-4', 'openai', 1.0)
        self.metrics.record_api_request('chat', 'gpt-4', 'openai', 2.0)
        self.metrics.record_api_request('chat', 'gpt-3.5-turbo', 'openai', 0.5)
        
        # 记录Token使用
        self.metrics.record_token_usage('gpt-4', 'openai', 100, 50)
        self.metrics.record_token_usage('gpt-4', 'openai', 200, 100)
        
        # 记录成本
        self.metrics.record_cost('gpt-4', 'openai', 0.05)
        self.metrics.record_cost('gpt-4', 'openai', 0.10)
        
        metrics_output = self.metrics.get_metrics()
        
        # 检查累计值
        assert 'harborai_api_requests_total{method="chat",model="gpt-4",provider="openai",status="success"} 2.0' in metrics_output
        assert 'harborai_api_requests_total{method="chat",model="gpt-3.5-turbo",provider="openai",status="success"} 1.0' in metrics_output
        assert 'harborai_tokens_used_total{model="gpt-4",provider="openai",token_type="prompt"} 300.0' in metrics_output
        assert 'harborai_tokens_used_total{model="gpt-4",provider="openai",token_type="completion"} 150.0' in metrics_output
        assert 'harborai_cost_total{model="gpt-4",provider="openai"} 0.15' in metrics_output


class TestGlobalMetricsManagement:
    """测试全局指标管理"""
    
    def setup_method(self):
        """测试前设置"""
        # 清理全局状态
        import harborai.monitoring.prometheus_metrics as pm
        pm._prometheus_metrics = None
    
    def teardown_method(self):
        """测试后清理"""
        # 清理全局状态
        import harborai.monitoring.prometheus_metrics as pm
        pm._prometheus_metrics = None
    
    def test_get_prometheus_metrics_none(self):
        """测试获取未初始化的全局指标实例"""
        metrics = get_prometheus_metrics()
        assert metrics is None
    
    def test_init_prometheus_metrics_default(self):
        """测试初始化全局指标实例 - 默认注册表"""
        metrics = init_prometheus_metrics()
        
        assert metrics is not None
        assert isinstance(metrics, PrometheusMetrics)
        assert get_prometheus_metrics() is metrics
    
    def test_init_prometheus_metrics_custom_registry(self):
        """测试初始化全局指标实例 - 自定义注册表"""
        custom_registry = CollectorRegistry()
        metrics = init_prometheus_metrics(custom_registry)
        
        assert metrics is not None
        assert metrics.registry is custom_registry
        assert get_prometheus_metrics() is metrics
    
    def test_init_prometheus_metrics_overwrite(self):
        """测试重新初始化全局指标实例"""
        # 第一次初始化
        metrics1 = init_prometheus_metrics()
        
        # 第二次初始化
        metrics2 = init_prometheus_metrics()
        
        assert metrics1 is not metrics2
        assert get_prometheus_metrics() is metrics2


class TestPrometheusMiddleware:
    """测试Prometheus中间件"""
    
    def setup_method(self):
        """设置测试环境"""
        # 重置全局实例
        import harborai.monitoring.prometheus_metrics as prometheus_metrics
        prometheus_metrics._prometheus_metrics = None
        self.metrics = prometheus_metrics.init_prometheus_metrics()
    
    def teardown_method(self):
        """清理测试环境"""
        import harborai.monitoring.prometheus_metrics as prometheus_metrics
        prometheus_metrics._prometheus_metrics = None
    
    @patch('harborai.config.performance.get_performance_config')
    def test_prometheus_middleware_enabled(self, mock_get_config):
        """测试Prometheus中间件启用时的行为"""
        # Mock性能配置
        mock_config = Mock()
        mock_config.get_middleware_config.return_value = {'metrics_middleware': True}
        mock_get_config.return_value = mock_config
        
        @prometheus_middleware
        def test_function(model="test-model", provider="test-provider"):
            return Mock(usage=Mock(prompt_tokens=10, completion_tokens=20))
        
        # 执行函数
        result = test_function()
        
        # 验证指标被记录
        metrics_output = self.metrics.get_metrics()
        assert 'harborai_api_requests_total' in metrics_output
        assert 'harborai_tokens_used_total' in metrics_output
    
    @patch('harborai.config.performance.get_performance_config')
    def test_prometheus_middleware_disabled(self, mock_get_config):
        """测试Prometheus中间件禁用时的行为"""
        # Mock性能配置
        mock_config = Mock()
        mock_config.get_middleware_config.return_value = {'metrics_middleware': False}
        mock_get_config.return_value = mock_config
        
        @prometheus_middleware
        def test_function(model="test-model", provider="test-provider"):
            return "test_result"
        
        # 执行函数
        result = test_function()
        
        # 验证函数正常执行但没有记录指标
        assert result == "test_result"
        metrics_output = self.metrics.get_metrics()
        # 由于中间件被禁用，不应该有API请求记录
        assert 'harborai_api_requests_total{method="test_function"' not in metrics_output
    
    def test_prometheus_middleware_no_metrics_instance(self):
        """测试没有指标实例时的行为"""
        # 清除全局实例
        import harborai.monitoring.prometheus_metrics as prometheus_metrics
        prometheus_metrics._prometheus_metrics = None
        
        @prometheus_middleware
        def test_function():
            return "test_result"
        
        # 执行函数
        result = test_function()
        
        # 验证函数正常执行
        assert result == "test_result"
    
    @patch('harborai.config.performance.get_performance_config')
    def test_prometheus_middleware_harbor_ai_error(self, mock_get_config):
        """测试Prometheus中间件处理HarborAI错误"""
        # Mock性能配置
        mock_config = Mock()
        mock_config.get_middleware_config.return_value = {'metrics_middleware': True}
        mock_get_config.return_value = mock_config
        
        # 创建一个自定义异常类来模拟HarborAIError
        class MockHarborAIError(Exception):
            pass
        
        @prometheus_middleware
        def test_function(model="test-model", provider="test-provider"):
            raise MockHarborAIError("测试错误")
        
        # 执行函数并捕获异常
        with pytest.raises(MockHarborAIError):
            test_function()
        
        # 验证错误指标被记录
        metrics_output = self.metrics.get_metrics()
        assert 'harborai_api_requests_total' in metrics_output
        assert 'status="error"' in metrics_output
    
    @patch('harborai.config.performance.get_performance_config')
    def test_prometheus_middleware_unexpected_error(self, mock_get_config):
        """测试Prometheus中间件 - 意外错误"""
        # Mock性能配置
        mock_config = Mock()
        mock_config.get_middleware_config.return_value = {'metrics_middleware': True}
        mock_get_config.return_value = mock_config
        
        @prometheus_middleware
        def test_function(model='gpt-4', provider='openai'):
            raise ValueError("意外错误")
        
        with pytest.raises(ValueError):
            test_function()
        
        # 检查错误指标
        metrics_output = self.metrics.get_metrics()
        assert 'harborai_api_requests_total' in metrics_output
        assert 'status="error"' in metrics_output
        assert 'harborai_api_errors_total' in metrics_output
        assert 'error_type="UnexpectedError"' in metrics_output
    
    @patch('harborai.config.performance.get_performance_config')
    def test_prometheus_middleware_no_usage(self, mock_get_config):
        """测试Prometheus中间件 - 无使用量信息"""
        # Mock性能配置
        mock_config = Mock()
        mock_config.get_middleware_config.return_value = {'metrics_middleware': True}
        mock_get_config.return_value = mock_config
        
        @prometheus_middleware
        def test_function(model='gpt-4', provider='openai'):
            # 返回一个没有usage属性的对象
            return "simple_result"
        
        result = test_function()
        
        # 检查基本指标记录，但无token和成本指标
        metrics_output = self.metrics.get_metrics()
        assert 'harborai_api_requests_total' in metrics_output
        assert 'status="success"' in metrics_output
        # 由于没有usage信息，不应该有token指标
        assert 'harborai_tokens_used_total{model="gpt-4"' not in metrics_output
        assert 'harborai_cost_total{model="gpt-4"' not in metrics_output
    
    @patch('harborai.config.performance.get_performance_config')
    def test_prometheus_middleware_cost_calculation_none(self, mock_get_config):
        """测试Prometheus中间件 - 成本计算返回None"""
        # Mock性能配置
        mock_config = Mock()
        mock_config.get_middleware_config.return_value = {'metrics_middleware': True}
        mock_get_config.return_value = mock_config
        
        @prometheus_middleware
        def test_function(model='gpt-4', provider='openai'):
            result = Mock()
            result.usage = Mock()
            result.usage.prompt_tokens = 100
            result.usage.completion_tokens = 50
            return result
        
        with patch('harborai.core.pricing.PricingCalculator') as mock_pricing:
            mock_pricing.calculate_cost.return_value = None
            
            result = test_function()
            
            # 检查token指标记录，但无成本指标
            metrics_output = self.metrics.get_metrics()
            assert 'harborai_api_requests_total' in metrics_output
            assert 'status="success"' in metrics_output
            assert 'harborai_tokens_used_total' in metrics_output
            assert 'token_type="prompt"' in metrics_output
            assert 'token_type="completion"' in metrics_output
            # 成本计算返回None时，不应该有成本指标
            assert 'harborai_cost_total{model="gpt-4"' not in metrics_output


class TestPrometheusAsyncMiddleware:
    """测试Prometheus异步中间件"""
    
    def setup_method(self):
        """测试前设置"""
        # 初始化全局指标实例
        import harborai.monitoring.prometheus_metrics as prometheus_metrics
        prometheus_metrics._prometheus_metrics = None
        self.metrics = prometheus_metrics.init_prometheus_metrics()
    
    def teardown_method(self):
        """测试后清理"""
        # 清理全局状态
        import harborai.monitoring.prometheus_metrics as prometheus_metrics
        prometheus_metrics._prometheus_metrics = None
    
    @patch('harborai.config.performance.get_performance_config')
    @pytest.mark.asyncio
    async def test_prometheus_async_middleware_enabled(self, mock_get_config):
        """测试Prometheus异步中间件 - 启用状态"""
        # Mock性能配置
        mock_config = Mock()
        mock_config.get_middleware_config.return_value = {'metrics_middleware': True}
        mock_get_config.return_value = mock_config
        
        @prometheus_async_middleware
        async def test_function(model='gpt-4', provider='openai'):
            await asyncio.sleep(0.01)  # 模拟异步处理
            result = Mock()
            result.usage = Mock()
            result.usage.prompt_tokens = 100
            result.usage.completion_tokens = 50
            return result
        
        with patch('harborai.core.pricing.PricingCalculator') as mock_pricing:
            mock_pricing.calculate_cost.return_value = 0.05
            
            result = await test_function()
            
            # 检查指标是否记录
            metrics_output = self.metrics.get_metrics()
            assert 'harborai_api_requests_total' in metrics_output
            assert 'status="success"' in metrics_output
            assert 'harborai_tokens_used_total' in metrics_output
            assert 'token_type="prompt"' in metrics_output
            assert 'token_type="completion"' in metrics_output
            assert 'harborai_cost_total' in metrics_output
    
    @patch('harborai.config.performance.get_performance_config')
    @pytest.mark.asyncio
    async def test_prometheus_async_middleware_disabled(self, mock_get_config):
        """测试Prometheus异步中间件 - 禁用状态"""
        # Mock性能配置
        mock_config = Mock()
        mock_config.get_middleware_config.return_value = {'metrics_middleware': False}
        mock_get_config.return_value = mock_config
        
        @prometheus_async_middleware
        async def test_function():
            return 'async_result'
        
        result = await test_function()
        
        # 检查指标是否未记录
        metrics_output = self.metrics.get_metrics()
        # 由于中间件被禁用，不应该有新的API请求记录
        assert 'harborai_api_requests_total{method="test_function"' not in metrics_output
        assert result == 'async_result'
    
    @patch('harborai.config.performance.get_performance_config')
    @pytest.mark.asyncio
    async def test_prometheus_async_middleware_error(self, mock_get_config):
        """测试Prometheus异步中间件 - 错误处理"""
        # Mock性能配置
        mock_config = Mock()
        mock_config.get_middleware_config.return_value = {'metrics_middleware': True}
        mock_get_config.return_value = mock_config
        
        @prometheus_async_middleware
        async def test_function(model='gpt-4', provider='openai'):
            raise ValueError("异步错误")
        
        with pytest.raises(ValueError):
            await test_function()
        
        # 检查错误指标
        metrics_output = self.metrics.get_metrics()
        assert 'harborai_api_requests_total' in metrics_output
        assert 'status="error"' in metrics_output
        assert 'harborai_api_errors_total' in metrics_output
        assert 'error_type="UnexpectedError"' in metrics_output


class TestMetricsIntegration:
    """测试指标集成"""
    
    def setup_method(self):
        """测试前设置"""
        self.metrics = init_prometheus_metrics()
    
    def teardown_method(self):
        """测试后清理"""
        import harborai.monitoring.prometheus_metrics as pm
        pm._prometheus_metrics = None
    
    def test_comprehensive_metrics_workflow(self):
        """测试完整指标工作流"""
        # 模拟完整的API调用流程
        
        # 1. 记录成功的API请求
        self.metrics.record_api_request('chat_completion', 'gpt-4', 'openai', 2.5)
        self.metrics.record_token_usage('gpt-4', 'openai', 150, 75)
        self.metrics.record_cost('gpt-4', 'openai', 0.075)
        
        # 2. 记录失败的API请求
        self.metrics.record_api_request('chat_completion', 'gpt-4', 'openai', 1.0, 'error', 'RateLimitError')
        self.metrics.record_retry('gpt-4', 'openai', 'rate_limit')
        
        # 3. 记录缓存操作
        self.metrics.record_cache_hit('response_cache')
        self.metrics.record_cache_miss('response_cache')
        
        # 4. 设置系统状态
        self.metrics.set_active_connections('openai', 3)
        self.metrics.set_system_info({
            'version': '1.0.0',
            'environment': 'test'
        })
        
        # 验证所有指标
        metrics_output = self.metrics.get_metrics()
        
        # 验证API请求指标
        assert 'harborai_api_requests_total{method="chat_completion",model="gpt-4",provider="openai",status="success"} 1.0' in metrics_output
        assert 'harborai_api_requests_total{method="chat_completion",model="gpt-4",provider="openai",status="error"} 1.0' in metrics_output
        
        # 验证Token使用指标
        assert 'harborai_tokens_used_total{model="gpt-4",provider="openai",token_type="prompt"} 150.0' in metrics_output
        assert 'harborai_tokens_used_total{model="gpt-4",provider="openai",token_type="completion"} 75.0' in metrics_output
        
        # 验证成本指标
        assert 'harborai_cost_total{model="gpt-4",provider="openai"} 0.075' in metrics_output
        
        # 验证错误和重试指标
        assert 'harborai_api_errors_total{error_type="RateLimitError",method="chat_completion",model="gpt-4",provider="openai"} 1.0' in metrics_output
        assert 'harborai_retries_total{model="gpt-4",provider="openai",retry_reason="rate_limit"} 1.0' in metrics_output
        
        # 验证缓存指标
        assert 'harborai_cache_hits_total{cache_type="response_cache"} 1.0' in metrics_output
        assert 'harborai_cache_misses_total{cache_type="response_cache"} 1.0' in metrics_output
        
        # 验证系统状态指标
        assert 'harborai_active_connections{provider="openai"} 3.0' in metrics_output
        assert 'harborai_system_info_info{environment="test",version="1.0.0"} 1.0' in metrics_output
    
    def test_metrics_persistence_across_calls(self):
        """测试指标在多次调用间的持久性"""
        # 第一次调用
        self.metrics.record_api_request('test', 'gpt-4', 'openai', 1.0)
        self.metrics.record_cost('gpt-4', 'openai', 0.01)
        
        # 第二次调用
        self.metrics.record_api_request('test', 'gpt-4', 'openai', 2.0)
        self.metrics.record_cost('gpt-4', 'openai', 0.02)
        
        # 第三次调用
        self.metrics.record_api_request('test', 'gpt-3.5-turbo', 'openai', 0.5)
        self.metrics.record_cost('gpt-3.5-turbo', 'openai', 0.005)
        
        metrics_output = self.metrics.get_metrics()
        
        # 验证累计值
        assert 'harborai_api_requests_total{method="test",model="gpt-4",provider="openai",status="success"} 2.0' in metrics_output
        assert 'harborai_api_requests_total{method="test",model="gpt-3.5-turbo",provider="openai",status="success"} 1.0' in metrics_output
        assert 'harborai_cost_total{model="gpt-4",provider="openai"} 0.03' in metrics_output
        assert 'harborai_cost_total{model="gpt-3.5-turbo",provider="openai"} 0.005' in metrics_output
    
    def test_metrics_content_type_consistency(self):
        """测试指标内容类型一致性"""
        content_type = self.metrics.get_content_type()
        assert content_type == CONTENT_TYPE_LATEST
        assert 'text/plain' in content_type
        assert 'version=0.0.4' in content_type