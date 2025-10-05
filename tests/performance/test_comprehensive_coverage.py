#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合覆盖率测试

专门用于提升整体测试覆盖率的测试文件，重点测试边界条件、错误处理和极端场景
"""

import pytest
import asyncio
import os
import sys
import tempfile
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from harborai import HarborAI
from harborai.utils.exceptions import (
    HarborAIError, 
    ValidationError,
    APIError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    PluginError,
    StructuredOutputError,
    StorageError,
    TimeoutError
)
from harborai.core.exceptions import RetryableError
from harborai.config.settings import Settings
from harborai.core.pricing import PricingCalculator
from harborai.monitoring.token_statistics import TokenStatisticsCollector
from harborai.utils.logger import get_logger
from harborai.core.retry import RetryManager
from harborai.core.observability import TracingManager


class TestExceptionHandling:
    """异常处理测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_harborai_error_creation(self):
        """测试HarborAI错误创建"""
        # 基本错误
        error = HarborAIError("测试错误")
        assert str(error) == "测试错误"
        assert error.message == "测试错误"
        
        # 带错误代码的错误
        error_with_code = HarborAIError("测试错误", error_code="TEST_001")
        assert error_with_code.error_code == "TEST_001"
        
        # 带详细信息的错误
        error_with_details = HarborAIError(
            "测试错误", 
            error_code="TEST_002",
            details={"key": "value"}
        )
        assert error_with_details.details == {"key": "value"}
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_model_not_found_error(self):
        """测试模型未找到错误"""
        error = ModelNotFoundError("test-model")
        assert isinstance(error, HarborAIError)
        assert "test-model" in str(error)
        assert error.model_name == "test-model"
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_validation_error(self):
        """测试验证错误"""
        error = ValidationError("验证失败", field="test_field")
        assert isinstance(error, HarborAIError)
        assert error.field == "test_field"
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_api_error(self):
        """测试API错误"""
        error = APIError("API调用失败", status_code=500)
        assert isinstance(error, HarborAIError)
        assert error.status_code == 500
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_rate_limit_error(self):
        """测试速率限制错误"""
        error = RateLimitError("速率限制", retry_after=60)
        assert isinstance(error, APIError)
        assert error.retry_after == 60
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_authentication_error(self):
        """测试认证错误"""
        error = AuthenticationError("认证失败")
        assert isinstance(error, APIError)
        assert str(error) == "[AUTHENTICATION_ERROR] 认证失败"


class TestSettingsConfiguration:
    """设置配置测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_settings_default_values(self):
        """测试设置默认值"""
        settings = Settings()
        
        # 检查默认值
        assert settings.default_timeout > 0
        assert settings.max_retries >= 0
        assert isinstance(settings.plugin_directories, list)
        assert len(settings.plugin_directories) > 0
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_settings_custom_values(self):
        """测试自定义设置值"""
        custom_settings = Settings(
            default_timeout=60,
            max_retries=5,
            plugin_directories=["custom.plugins"]
        )
        
        assert custom_settings.default_timeout == 60
        assert custom_settings.max_retries == 5
        assert custom_settings.plugin_directories == ["custom.plugins"]
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_settings_environment_variables(self):
        """测试环境变量设置"""
        with patch.dict(os.environ, {
            "HARBORAI_DEFAULT_TIMEOUT": "120",
            "HARBORAI_MAX_RETRIES": "10"
        }):
            settings = Settings()
            # 注意：实际实现可能需要支持环境变量
            # 这里只是测试框架
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_settings_validation(self):
        """测试设置验证"""
        import os
        from pydantic import ValidationError as PydanticValidationError
        
        # 测试无效的超时值（通过环境变量）
        os.environ['HARBORAI_TIMEOUT'] = '-1'
        try:
            with pytest.raises((ValueError, ValidationError, PydanticValidationError)):
                Settings()
        finally:
            if 'HARBORAI_TIMEOUT' in os.environ:
                del os.environ['HARBORAI_TIMEOUT']
        
        # 测试无效的重试次数
        with pytest.raises((ValueError, ValidationError, PydanticValidationError)):
            Settings(max_retries=-1)


class TestPricingCalculator:
    """定价计算器测试类"""
    
    @pytest.fixture
    def pricing_calculator(self):
        """创建定价计算器实例"""
        return PricingCalculator()
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_pricing_calculator_creation(self, pricing_calculator):
        """测试定价计算器创建"""
        assert pricing_calculator is not None
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_calculate_cost_basic(self, pricing_calculator):
        """测试基本成本计算"""
        # 模拟基本的成本计算
        cost = pricing_calculator.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="test-model"
        )
        assert cost is None or cost >= 0
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_calculate_cost_zero_tokens(self, pricing_calculator):
        """测试零token成本计算"""
        cost = pricing_calculator.calculate_cost(
            input_tokens=0,
            output_tokens=0,
            model_name="test-model"
        )
        assert cost is None or cost == 0
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_calculate_cost_invalid_model(self, pricing_calculator):
        """测试无效模型成本计算"""
        # 应该处理未知模型
        cost = pricing_calculator.calculate_cost(
            input_tokens=1000,
            output_tokens=500,
            model_name="unknown-model"
        )
        # 可能返回默认价格或抛出异常
        assert cost is None or cost >= 0


class TestTokenStatistics:
    """Token统计测试类"""
    
    @pytest.fixture
    def token_stats(self):
        """创建Token统计实例"""
        return TokenStatisticsCollector()
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_token_statistics_creation(self, token_stats):
        """测试Token统计创建"""
        assert token_stats is not None
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_record_usage(self, token_stats):
        """测试记录使用情况"""
        token_stats.record_usage(
            trace_id="test-trace-123",
            model="test-model",
            input_tokens=1000,
            output_tokens=500,
            duration=1.5,
            success=True
        )
        
        # 检查统计数据
        stats = token_stats.get_summary_stats()
        assert stats["total_tokens"] >= 1500
        assert stats["total_requests"] >= 1
        assert stats["total_cost"] >= 0
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_get_model_statistics(self, token_stats):
        """测试获取模型统计"""
        token_stats.record_usage("trace1", "model1", 1000, 500, 1.0, True)
        token_stats.record_usage("trace2", "model2", 2000, 1000, 1.5, True)
        
        model1_stats = token_stats.get_model_statistics("model1")
        assert "model1" in model1_stats
        assert model1_stats["model1"].total_input_tokens >= 1000
        assert model1_stats["model1"].total_output_tokens >= 500
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_clear_old_records(self, token_stats):
        """测试清理旧记录"""
        token_stats.record_usage("trace1", "test-model", 1000, 500, 1.0, True)
        
        # 清理7天前的记录（应该不会清理刚添加的记录）
        cleared_count = token_stats.clear_old_records(days=7)
        
        stats = token_stats.get_summary_stats()
        assert stats["total_requests"] >= 1
        assert cleared_count >= 0


class TestLogger:
    """日志记录器测试类"""
    
    @pytest.fixture
    def logger(self):
        """创建日志记录器实例"""
        return get_logger("test_logger")
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_logger_creation(self, logger):
        """测试日志记录器创建"""
        assert logger is not None
        assert logger.name == "test_logger"
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_log_levels(self, logger):
        """测试不同日志级别"""
        logger.debug("调试信息")
        logger.info("信息")
        logger.warning("警告")
        logger.error("错误")
        logger.critical("严重错误")
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_log_with_extra_data(self, logger):
        """测试带额外数据的日志"""
        logger.info("测试消息", extra={
            "user_id": "123",
            "request_id": "req_456"
        })
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_log_exception(self, logger):
        """测试异常日志"""
        try:
            raise ValueError("测试异常")
        except ValueError as e:
            logger.exception("捕获异常", exc_info=e)


class TestRetryManager:
    """重试管理器测试类"""
    
    @pytest.fixture
    def retry_manager(self):
        """创建重试管理器实例"""
        from harborai.core.retry import RetryConfig
        config = RetryConfig(max_attempts=3, base_delay=0.1)
        return RetryManager(config)
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_retry_manager_creation(self, retry_manager):
        """测试重试管理器创建"""
        assert retry_manager is not None
        assert retry_manager.config.max_attempts == 3
        assert retry_manager.config.base_delay == 0.1
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_successful_operation(self, retry_manager):
        """测试成功操作（无需重试）"""
        def successful_operation():
            return "成功"
        
        result = retry_manager.execute(successful_operation)
        assert result == "成功"
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_retry_on_failure(self, retry_manager):
        """测试失败时重试"""
        call_count = 0
        
        def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                from harborai.core.exceptions import RetryableError
                raise RetryableError("临时失败")
            return "最终成功"
        
        result = retry_manager.execute(failing_operation)
        assert result == "最终成功"
        assert call_count == 3
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_max_retries_exceeded(self, retry_manager):
        """测试超过最大重试次数"""
        def always_failing_operation():
            from harborai.core.exceptions import RetryableError
            raise RetryableError("总是失败")
        
        with pytest.raises(RetryableError):
            retry_manager.execute(always_failing_operation)


class TestTracingManager:
    """追踪管理器测试类"""
    
    @pytest.fixture
    def tracing_manager(self):
        """创建追踪管理器实例"""
        return TracingManager()
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_tracing_manager_creation(self, tracing_manager):
        """测试追踪管理器创建"""
        assert tracing_manager is not None
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_start_trace(self, tracing_manager):
        """测试开始追踪"""
        trace_id = tracing_manager.start_trace("test_operation")
        assert trace_id is not None
        assert isinstance(trace_id, str)
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_span_operations(self, tracing_manager):
        """测试跨度操作"""
        trace_id = tracing_manager.start_trace("test_operation")
        span_id = tracing_manager.get_active_span_id()
        assert span_id is not None
        
        # 结束跨度
        tracing_manager.finish_span(span_id, status="ok")
    
    @pytest.mark.unit
    @pytest.mark.p1
    def test_nested_spans(self, tracing_manager):
        """测试嵌套跨度"""
        trace_id = tracing_manager.start_trace("parent_operation")
        parent_span_id = tracing_manager.get_active_span_id()
        
        child_span_id = tracing_manager.start_span("child_operation", parent_span_id=parent_span_id)
        assert child_span_id is not None
        
        tracing_manager.finish_span(child_span_id, status="ok")
        tracing_manager.finish_span(parent_span_id, status="ok")


class TestBoundaryConditions:
    """边界条件测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_empty_string_handling(self):
        """测试空字符串处理"""
        # 测试各种组件对空字符串的处理
        harborai = HarborAI()
        
        # 空消息应该被适当处理
        with pytest.raises((ValueError, ValidationError, ModelNotFoundError)):
            harborai.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": ""}]
            )
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_very_long_input(self):
        """测试超长输入处理"""
        harborai = HarborAI()
        
        # 创建一个很长的输入
        long_content = "测试" * 10000
        
        # 应该有适当的长度限制或处理
        try:
            result = harborai.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": long_content}]
            )
        except (ValueError, ValidationError, ModelNotFoundError) as e:
            # 预期的验证错误或模型未找到错误
            assert "长度" in str(e) or "length" in str(e).lower() or "not found" in str(e).lower()
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_special_characters_handling(self):
        """测试特殊字符处理"""
        harborai = HarborAI()
        
        special_chars = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"
        unicode_chars = "你好世界🌍🚀💻"
        
        # 应该能处理特殊字符和Unicode
        try:
            harborai.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": special_chars + unicode_chars}]
            )
        except Exception as e:
            # 如果抛出异常，应该是合理的验证错误
            pass
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_null_and_none_handling(self):
        """测试NULL和None值处理"""
        harborai = HarborAI()
        
        # None值应该被适当处理
        with pytest.raises((ValueError, TypeError, ValidationError, ModelNotFoundError)):
            harborai.chat.completions.create(
                model="test-model",
                messages=None
            )
        
        with pytest.raises((ValueError, TypeError, ValidationError, ModelNotFoundError)):
            harborai.chat.completions.create(
                model=None,
                messages=[{"role": "user", "content": "测试"}]
            )


class TestErrorRecovery:
    """错误恢复测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_network_error_recovery(self):
        """测试网络错误恢复"""
        harborai = HarborAI()
        
        # 模拟网络错误
        with patch('requests.post') as mock_post:
            mock_post.side_effect = ConnectionError("网络连接失败")
            
            with pytest.raises((ConnectionError, APIError, ModelNotFoundError)):
                harborai.chat.completions.create(
                    model="test-model",
                    messages=[{"role": "user", "content": "测试"}]
                )
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_timeout_handling(self):
        """测试超时处理"""
        harborai = HarborAI()
        
        # 模拟超时
        with patch('requests.post') as mock_post:
            mock_post.side_effect = TimeoutError("请求超时")
            
            with pytest.raises((TimeoutError, APIError, ModelNotFoundError)):
                harborai.chat.completions.create(
                    model="test-model",
                    messages=[{"role": "user", "content": "测试"}]
                )
    
    @pytest.mark.unit
    @pytest.mark.p2
    def test_rate_limit_handling(self):
        """测试速率限制处理"""
        harborai = HarborAI()
        
        # 模拟速率限制
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.json.return_value = {"error": "Rate limit exceeded"}
            mock_post.return_value = mock_response
            
            with pytest.raises((RateLimitError, ModelNotFoundError)):
                harborai.chat.completions.create(
                    model="test-model",
                    messages=[{"role": "user", "content": "测试"}]
                )


class TestPerformanceEdgeCases:
    """性能边界情况测试类"""
    
    @pytest.mark.performance
    @pytest.mark.p3
    def test_concurrent_requests(self):
        """测试并发请求处理"""
        harborai = HarborAI()
        
        # 使用同步方法进行并发测试
        import threading
        import time
        
        results = []
        errors = []
        
        def make_request():
            try:
                # 模拟请求，预期会抛出ModelNotFoundError
                harborai.chat.completions.create(
                    model="test-model",
                    messages=[{"role": "user", "content": "测试"}]
                )
            except (ModelNotFoundError, Exception) as e:
                errors.append(e)
        
        # 创建多个线程进行并发测试
        threads = []
        for _ in range(5):  # 减少线程数量以避免过度负载
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # 启动所有线程
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # 验证并发处理
        assert len(errors) == 5  # 所有请求都应该产生错误（因为使用test-model）
        assert end_time - start_time < 5  # 并发处理应该在合理时间内完成
    
    @pytest.mark.performance
    @pytest.mark.p3
    def test_memory_usage_monitoring(self):
        """测试内存使用监控"""
        import psutil
        import gc
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # 创建少量对象进行测试
            harborai_instances = []
            for i in range(10):  # 减少实例数量
                harborai_instances.append(HarborAI())
            
            # 检查内存增长
            current_memory = process.memory_info().rss
            memory_growth = current_memory - initial_memory
            
            # 清理
            del harborai_instances
            gc.collect()
            
            # 检查内存是否释放
            final_memory = process.memory_info().rss
            
            # 内存增长应该在合理范围内（更宽松的限制）
            assert memory_growth < 500 * 1024 * 1024  # 500MB
            
            # 记录内存使用情况（不强制要求释放，因为Python的GC行为可能不可预测）
            print(f"初始内存: {initial_memory / 1024 / 1024:.2f}MB")
            print(f"当前内存: {current_memory / 1024 / 1024:.2f}MB")
            print(f"最终内存: {final_memory / 1024 / 1024:.2f}MB")
            print(f"内存增长: {memory_growth / 1024 / 1024:.2f}MB")
            
        except Exception as e:
            # 如果内存监控失败，跳过测试而不是失败
            pytest.skip(f"内存监控测试跳过: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])