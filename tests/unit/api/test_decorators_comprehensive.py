#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装饰器模块全面测试

测试harborai.api.decorators模块的所有装饰器功能，包括：
- with_trace: 同步函数追踪
- with_async_trace: 异步函数追踪  
- with_logging: 同步函数日志记录
- with_async_logging: 异步函数日志记录
- cost_tracking: 成本追踪
- with_postgres_logging: PostgreSQL日志记录

测试策略：
1. 单元测试：测试每个装饰器的基本功能
2. 边界测试：测试异常情况和边界条件
3. 集成测试：测试装饰器组合使用
4. Mock测试：模拟外部依赖
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Any, Dict

from harborai.api.decorators import (
    with_trace,
    with_async_trace,
    with_logging,
    with_async_logging,
    cost_tracking,
    with_postgres_logging
)
from harborai.utils.exceptions import HarborAIError


class TestWithTrace:
    """测试with_trace装饰器"""
    
    def test_with_trace_basic_functionality(self):
        """测试基本追踪功能"""
        @with_trace
        def sample_function(x: int, y: int, trace_id: str = None) -> int:
            return x + y
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            with patch('harborai.api.decorators.generate_trace_id', return_value='test-trace-123'):
                result = sample_function(1, 2)
                
                assert result == 3
                assert mock_logger.info.call_count == 2  # start and complete
                mock_logger.info.assert_any_call("[test-trace-123] Starting sample_function")
    
    def test_with_trace_with_existing_trace_id(self):
        """测试使用已有trace_id的情况"""
        @with_trace
        def sample_function(x: int, trace_id: str = None) -> int:
            return x * 2
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            result = sample_function(5, trace_id='existing-trace-456')
            
            assert result == 10
            mock_logger.info.assert_any_call("[existing-trace-456] Starting sample_function")
    
    def test_with_trace_exception_handling(self):
        """测试异常处理"""
        @with_trace
        def failing_function(trace_id: str = None):
            raise ValueError("测试异常")
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            with patch('harborai.api.decorators.generate_trace_id', return_value='error-trace-789'):
                with pytest.raises(ValueError, match="测试异常"):
                    failing_function()
                
                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args[0][0]
                assert "error-trace-789" in error_call
                assert "Failed failing_function" in error_call
    
    def test_with_trace_preserves_function_metadata(self):
        """测试装饰器保持函数元数据"""
        @with_trace
        def documented_function(trace_id: str = None):
            """这是一个有文档的函数"""
            return "result"
        
        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "这是一个有文档的函数"


class TestWithAsyncTrace:
    """测试with_async_trace装饰器"""
    
    @pytest.mark.asyncio
    async def test_with_async_trace_basic_functionality(self):
        """测试异步追踪基本功能"""
        @with_async_trace
        async def async_sample_function(x: int, y: int, trace_id: str = None) -> int:
            await asyncio.sleep(0.01)  # 模拟异步操作
            return x + y
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            with patch('harborai.api.decorators.generate_trace_id', return_value='async-trace-123'):
                result = await async_sample_function(3, 4)
                
                assert result == 7
                assert mock_logger.info.call_count == 2
                mock_logger.info.assert_any_call("[async-trace-123] Starting async async_sample_function")
    
    @pytest.mark.asyncio
    async def test_with_async_trace_exception_handling(self):
        """测试异步异常处理"""
        @with_async_trace
        async def async_failing_function(trace_id: str = None):
            await asyncio.sleep(0.01)
            raise RuntimeError("异步测试异常")
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            with patch('harborai.api.decorators.generate_trace_id', return_value='async-error-456'):
                with pytest.raises(RuntimeError, match="异步测试异常"):
                    await async_failing_function()
                
                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args[0][0]
                assert "async-error-456" in error_call
                assert "Failed async async_failing_function" in error_call


class TestWithLogging:
    """测试with_logging装饰器"""
    
    def test_with_logging_basic_functionality(self):
        """测试基本日志记录功能"""
        @with_logging
        def logged_function(value: str) -> str:
            return f"processed: {value}"
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            result = logged_function("test_data")
            
            assert result == "processed: test_data"
            # 验证debug日志被调用
            assert mock_logger.debug.call_count >= 1
            # 验证调用日志包含函数名
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("logged_function" in call for call in debug_calls)
    
    def test_with_logging_exception_handling(self):
        """测试日志记录异常处理"""
        @with_logging
        def failing_logged_function():
            raise HarborAIError("日志测试异常")
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            with pytest.raises(HarborAIError, match="日志测试异常"):
                failing_logged_function()
            
            # 验证警告日志被调用（HarborAIError会触发warning）
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "HarborAI error" in warning_call
            assert "failing_logged_function" in warning_call


class TestWithAsyncLogging:
    """测试with_async_logging装饰器"""
    
    @pytest.mark.asyncio
    async def test_with_async_logging_basic_functionality(self):
        """测试异步日志记录基本功能"""
        @with_async_logging
        async def async_logged_function(data: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"result": data.get("input", "default")}
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            test_data = {"input": "async_test"}
            result = await async_logged_function(test_data)
            
            assert result == {"result": "async_test"}
            # 验证debug日志被调用
            assert mock_logger.debug.call_count >= 1
            # 验证调用日志包含函数名
            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            assert any("async_logged_function" in call for call in debug_calls)
    
    @pytest.mark.asyncio
    async def test_with_async_logging_exception_handling(self):
        """测试异步日志记录异常处理"""
        @with_async_logging
        async def async_failing_logged_function():
            await asyncio.sleep(0.01)
            raise ValueError("异步日志测试异常")
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            with pytest.raises(ValueError, match="异步日志测试异常"):
                await async_failing_logged_function()
            
            # 验证错误日志被调用（ValueError会触发error）
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "Unexpected error" in error_call
            assert "async_failing_logged_function" in error_call


class TestCostTracking:
    """测试cost_tracking装饰器"""
    
    def test_cost_tracking_basic_functionality(self):
        """测试基本成本追踪功能"""
        @cost_tracking
        def cost_tracked_function(model: str = "gpt-3.5-turbo", trace_id: str = None) -> Dict[str, Any]:
            # 创建一个模拟的结果对象，包含usage属性
            result = Mock()
            result.usage = Mock()
            result.usage.prompt_tokens = 10
            result.usage.completion_tokens = 20
            result.usage.total_tokens = 30
            return result
        
        with patch('harborai.api.decorators.PricingCalculator.calculate_cost') as mock_calculate:
            with patch('harborai.api.decorators.record_token_usage') as mock_record:
                mock_calculate.return_value = 0.0015
                
                result = cost_tracked_function()
                
                assert hasattr(result, 'usage')
                mock_calculate.assert_called_once()
                mock_record.assert_called_once()
    
    def test_cost_tracking_with_custom_model(self):
        """测试自定义模型的成本追踪"""
        @cost_tracking
        def custom_model_function(model: str = "gpt-4", trace_id: str = None) -> Dict[str, Any]:
            # 创建一个模拟的结果对象，包含usage属性
            result = Mock()
            result.usage = Mock()
            result.usage.prompt_tokens = 50
            result.usage.completion_tokens = 100
            result.usage.total_tokens = 150
            result.model = model
            return result
        
        with patch('harborai.api.decorators.PricingCalculator.calculate_cost') as mock_calculate:
            with patch('harborai.api.decorators.record_token_usage') as mock_record:
                mock_calculate.return_value = 0.0075
                
                result = custom_model_function(model="gpt-4-turbo")
                
                assert result.model == "gpt-4-turbo"
                mock_calculate.assert_called_once()
    
    def test_cost_tracking_exception_handling(self):
        """测试成本追踪异常处理"""
        @cost_tracking
        def failing_cost_function():
            raise RuntimeError("成本追踪测试异常")
        
        with patch('harborai.api.decorators.PricingCalculator') as mock_calculator:
            with patch('harborai.api.decorators.record_token_usage') as mock_record:
                mock_calc_instance = Mock()
                mock_calculator.return_value = mock_calc_instance
                
                with pytest.raises(RuntimeError, match="成本追踪测试异常"):
                    failing_cost_function()
                
                # 验证即使异常也会记录成本信息
                mock_record.assert_called_once()


class TestWithPostgresLogging:
    """测试with_postgres_logging装饰器"""
    
    def test_with_postgres_logging_sync_function(self):
        """测试同步函数的PostgreSQL日志记录"""
        @with_postgres_logging
        def postgres_logged_sync_function(data: str, trace_id: str = None) -> str:
            return f"sync_processed: {data}"
        
        with patch('harborai.api.decorators.get_postgres_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            result = postgres_logged_sync_function("sync_test")
            
            assert result == "sync_processed: sync_test"
            mock_logger.log_request.assert_called_once()
            mock_logger.log_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_with_postgres_logging_async_function(self):
        """测试异步函数的PostgreSQL日志记录"""
        @with_postgres_logging
        async def postgres_logged_async_function(data: str, trace_id: str = None) -> str:
            await asyncio.sleep(0.01)
            return f"async_processed: {data}"
        
        with patch('harborai.api.decorators.get_postgres_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.log_request = AsyncMock()
            mock_logger.log_response = AsyncMock()
            mock_get_logger.return_value = mock_logger
            
            result = await postgres_logged_async_function("async_test")
            
            assert result == "async_processed: async_test"
            mock_logger.log_request.assert_called_once()
            mock_logger.log_response.assert_called_once()


class TestDecoratorCombinations:
    """测试装饰器组合使用"""
    
    def test_multiple_decorators_combination(self):
        """测试多个装饰器组合使用"""
        @with_trace
        @cost_tracking
        @with_logging
        def multi_decorated_function(value: int, trace_id: str = None) -> Dict[str, Any]:
            # 创建一个模拟的结果对象，包含usage属性
            result = Mock()
            result.result = value * 2
            result.usage = Mock()
            result.usage.prompt_tokens = 5
            result.usage.completion_tokens = 10
            result.usage.total_tokens = 15
            result.model = "test-model"
            return result
        
        with patch('harborai.api.decorators.logger') as mock_trace_logger:
            with patch('harborai.api.decorators.PricingCalculator.calculate_cost') as mock_calculate:
                with patch('harborai.api.decorators.record_token_usage') as mock_record:
                    with patch('harborai.api.decorators.generate_trace_id', return_value='multi-trace-123'):
                        mock_calculate.return_value = 0.001
                        
                        result = multi_decorated_function(5)
                        
                        assert result.result == 10
                        # 验证所有装饰器都被调用
                        mock_trace_logger.debug.assert_called()
                        mock_calculate.assert_called()
                        mock_record.assert_called()
    
    @pytest.mark.asyncio
    async def test_async_decorators_combination(self):
        """测试异步装饰器组合使用"""
        @with_async_trace
        @with_async_logging
        async def async_multi_decorated_function(data: Dict[str, Any], trace_id: str = None) -> Dict[str, Any]:
            await asyncio.sleep(0.01)
            return {"processed": data, "status": "success"}
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            with patch('harborai.api.decorators.generate_trace_id', return_value='async-multi-456'):
                test_data = {"input": "test"}
                result = await async_multi_decorated_function(test_data)
                
                assert result["status"] == "success"
                assert result["processed"] == test_data
                mock_logger.info.assert_called()
                mock_logger.debug.assert_called()


class TestEdgeCases:
    """测试边界情况和异常场景"""
    
    def test_decorator_with_no_arguments(self):
        """测试无参数函数的装饰器"""
        @with_trace
        def no_args_function(trace_id: str = None):
            return "no_args_result"
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            with patch('harborai.api.decorators.generate_trace_id', return_value='no-args-trace'):
                result = no_args_function()
                
                assert result == "no_args_result"
                mock_logger.info.assert_called()
    
    def test_decorator_with_kwargs_only(self):
        """测试仅关键字参数函数的装饰器"""
        @with_trace
        def kwargs_only_function(**kwargs):
            return kwargs
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            with patch('harborai.api.decorators.generate_trace_id', return_value='kwargs-trace'):
                result = kwargs_only_function(a=1, b=2, trace_id='custom-trace')
                
                assert result == {"a": 1, "b": 2, "trace_id": "custom-trace"}
                mock_logger.info.assert_any_call("[custom-trace] Starting kwargs_only_function")
    
    def test_cost_tracking_with_missing_usage(self):
        """测试缺少usage信息的成本追踪"""
        @cost_tracking
        def no_usage_function(trace_id: str = None):
            # 创建一个没有usage属性的结果对象
            result = Mock()
            result.model = "test-model"
            # 故意不设置usage属性
            return result
        
        with patch('harborai.api.decorators.PricingCalculator.calculate_cost') as mock_calculate:
            with patch('harborai.api.decorators.record_token_usage') as mock_record:
                # 设置calculate_cost返回一个实际的数值而不是Mock
                mock_calculate.return_value = 0.001
                
                result = no_usage_function()
                
                assert result.model == "test-model"
                # 验证即使缺少usage也不会崩溃
                mock_record.assert_called()
    
    def test_logging_with_none_result(self):
        """测试返回None的函数日志记录"""
        @with_logging
        def none_result_function(trace_id: str = None):
            return None
        
        with patch('harborai.api.decorators.logger') as mock_logger:
            result = none_result_function()
            
            assert result is None
            mock_logger.debug.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])