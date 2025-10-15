#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
装饰器模块缺失覆盖率测试

专门测试decorators.py中未被现有测试覆盖的代码路径，
目标是将覆盖率从74%提升到80%+。

主要测试场景：
1. cost_tracking装饰器的禁用分支
2. cost_tracking装饰器的异常处理路径
3. with_postgres_logging装饰器的无logger分支
4. with_postgres_logging装饰器的异常处理路径
5. 边界条件和异常情况

遵循VIBE规范：
- 使用中文注释
- TDD流程
- 小步快验
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Any, Dict

from harborai.api.decorators import (
    cost_tracking,
    with_postgres_logging
)
from harborai.utils.exceptions import HarborAIError


class TestCostTrackingMissingCoverage:
    """测试cost_tracking装饰器的未覆盖代码路径"""
    
    def test_cost_tracking_disabled_sync(self):
        """测试同步函数禁用成本追踪的情况（覆盖行80-82）"""
        @cost_tracking
        def sync_function_with_disabled_tracking(value: int, cost_tracking: bool = True) -> int:
            return value * 2
        
        # 测试禁用成本追踪的情况
        with patch('harborai.api.decorators.PricingCalculator.calculate_cost') as mock_calculate:
            with patch('harborai.api.decorators.record_token_usage') as mock_record:
                result = sync_function_with_disabled_tracking(5, cost_tracking=False)
                
                assert result == 10
                # 验证成本追踪相关函数没有被调用
                mock_calculate.assert_not_called()
                mock_record.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cost_tracking_disabled_async(self):
        """测试异步函数禁用成本追踪的情况（覆盖行101-102）"""
        @cost_tracking
        async def async_function_with_disabled_tracking(value: int, cost_tracking: bool = True) -> int:
            await asyncio.sleep(0.01)
            return value * 3
        
        # 测试禁用成本追踪的情况
        with patch('harborai.api.decorators.PricingCalculator.calculate_cost') as mock_calculate:
            with patch('harborai.api.decorators.record_token_usage') as mock_record:
                result = await async_function_with_disabled_tracking(7, cost_tracking=False)
                
                assert result == 21
                # 验证成本追踪相关函数没有被调用
                mock_calculate.assert_not_called()
                mock_record.assert_not_called()
    
    def test_cost_tracking_sync_with_usage_and_cost_calculation(self):
        """测试同步函数成本追踪的完整流程（覆盖行113-176）"""
        @cost_tracking
        def sync_function_with_usage(model: str = "gpt-3.5-turbo", trace_id: str = "test-trace") -> Mock:
            # 创建包含usage信息的结果对象
            result = Mock()
            result.usage = Mock()
            result.usage.prompt_tokens = 100
            result.usage.completion_tokens = 50
            result.usage.total_tokens = 150
            return result
        
        with patch('harborai.api.decorators.PricingCalculator.calculate_cost') as mock_calculate:
            with patch('harborai.api.decorators.record_token_usage') as mock_record:
                with patch('harborai.api.decorators.logger') as mock_logger:
                    # 设置成本计算返回值
                    mock_calculate.return_value = 0.0025
                    
                    result = sync_function_with_usage(model="gpt-4", trace_id="sync-trace-123")
                    
                    # 验证结果
                    assert hasattr(result, 'usage')
                    assert result.usage.prompt_tokens == 100
                    assert result.usage.completion_tokens == 50
                    
                    # 验证成本计算被调用
                    mock_calculate.assert_called_once_with(
                        input_tokens=100,
                        output_tokens=50,
                        model_name="gpt-4"
                    )
                    
                    # 验证token使用记录被调用
                    mock_record.assert_called_once()
                    record_call_args = mock_record.call_args[1]
                    assert record_call_args['trace_id'] == "sync-trace-123"
                    assert record_call_args['model'] == "gpt-4"
                    assert record_call_args['input_tokens'] == 100
                    assert record_call_args['output_tokens'] == 50
                    assert record_call_args['success'] is True
                    
                    # 验证日志记录
                    mock_logger.info.assert_called()
                    log_message = mock_logger.info.call_args[0][0]
                    assert "Cost tracking" in log_message
                    assert "gpt-4" in log_message
                    assert "Cost: ¥0.002500" in log_message
    
    def test_cost_tracking_sync_with_none_cost(self):
        """测试同步函数成本计算返回None的情况"""
        @cost_tracking
        def sync_function_with_none_cost(model: str = "unknown-model") -> Mock:
            result = Mock()
            result.usage = Mock()
            result.usage.prompt_tokens = 10
            result.usage.completion_tokens = 5
            result.usage.total_tokens = 15
            return result
        
        with patch('harborai.api.decorators.PricingCalculator.calculate_cost') as mock_calculate:
            with patch('harborai.api.decorators.record_token_usage') as mock_record:
                with patch('harborai.api.decorators.logger') as mock_logger:
                    # 设置成本计算返回None
                    mock_calculate.return_value = None
                    
                    result = sync_function_with_none_cost()
                    
                    # 验证日志记录包含"Cost: N/A"
                    mock_logger.info.assert_called()
                    log_message = mock_logger.info.call_args[0][0]
                    assert "Cost: N/A" in log_message
    
    def test_cost_tracking_sync_exception_handling(self):
        """测试同步函数成本追踪的异常处理（覆盖行194->224）"""
        @cost_tracking
        def sync_failing_function(**kwargs) -> None:
            raise ValueError("同步成本追踪测试异常")
        
        with patch('harborai.api.decorators.record_token_usage') as mock_record:
            with pytest.raises(ValueError, match="同步成本追踪测试异常"):
                sync_failing_function(model="test-model", trace_id="error-trace")
            
            # 验证失败情况下的token使用记录
            mock_record.assert_called_once()
            record_call_args = mock_record.call_args[1]
            assert record_call_args['trace_id'] == "error-trace"
            assert record_call_args['model'] == "test-model"
            assert record_call_args['input_tokens'] == 0
            assert record_call_args['output_tokens'] == 0
            assert record_call_args['success'] is False
            assert record_call_args['error'] == "同步成本追踪测试异常"
    
    @pytest.mark.asyncio
    async def test_cost_tracking_async_exception_handling(self):
        """测试异步函数成本追踪的异常处理（覆盖行183）"""
        @cost_tracking
        async def async_failing_function(**kwargs) -> None:
            await asyncio.sleep(0.01)
            raise RuntimeError("异步成本追踪测试异常")
        
        with patch('harborai.api.decorators.record_token_usage') as mock_record:
            with pytest.raises(RuntimeError, match="异步成本追踪测试异常"):
                await async_failing_function(model="async-model", trace_id="async-error")
            
            # 验证失败情况下的token使用记录
            mock_record.assert_called_once()
            record_call_args = mock_record.call_args[1]
            assert record_call_args['trace_id'] == "async-error"
            assert record_call_args['model'] == "async-model"
            assert record_call_args['input_tokens'] == 0
            assert record_call_args['output_tokens'] == 0
            assert record_call_args['success'] is False
            assert record_call_args['error'] == "异步成本追踪测试异常"


class TestPostgresLoggingMissingCoverage:
    """测试with_postgres_logging装饰器的未覆盖代码路径"""
    
    def test_postgres_logging_no_logger_sync(self):
        """测试同步函数无PostgreSQL日志记录器的情况（覆盖行252）"""
        @with_postgres_logging
        def sync_function_no_postgres(data: str) -> str:
            return f"processed: {data}"
        
        with patch('harborai.api.decorators.get_postgres_logger') as mock_get_logger:
            # 模拟没有配置PostgreSQL日志记录器
            mock_get_logger.return_value = None
            
            result = sync_function_no_postgres("test_data")
            
            assert result == "processed: test_data"
            # 验证get_postgres_logger被调用但返回None
            mock_get_logger.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_postgres_logging_no_logger_async(self):
        """测试异步函数无PostgreSQL日志记录器的情况"""
        @with_postgres_logging
        async def async_function_no_postgres(data: str) -> str:
            await asyncio.sleep(0.01)
            return f"async_processed: {data}"
        
        with patch('harborai.api.decorators.get_postgres_logger') as mock_get_logger:
            # 模拟没有配置PostgreSQL日志记录器
            mock_get_logger.return_value = None
            
            result = await async_function_no_postgres("async_test")
            
            assert result == "async_processed: async_test"
            # 验证get_postgres_logger被调用但返回None
            mock_get_logger.assert_called_once()
    
    def test_with_postgres_logging_sync_exception_handling(self):
        """测试同步函数PostgreSQL日志记录的异常处理（覆盖行295->305）"""
        @with_postgres_logging
        def sync_failing_function(**kwargs) -> None:
            raise ValueError("同步PostgreSQL日志测试异常")
        
        mock_logger = Mock()
        with patch('harborai.api.decorators.get_postgres_logger', return_value=mock_logger):
            with pytest.raises(ValueError, match="同步PostgreSQL日志测试异常"):
                sync_failing_function(model="test-model", trace_id="error-trace")
            
            # 验证日志记录调用
            mock_logger.log_request.assert_called_once()
            mock_logger.log_response.assert_called_once()
            
            # 验证错误日志记录
            response_call_args = mock_logger.log_response.call_args[1]
            assert response_call_args['trace_id'] == "error-trace"
            assert response_call_args['response'] is None
            assert response_call_args['success'] is False
            assert response_call_args['error'] == "同步PostgreSQL日志测试异常"
    
    @pytest.mark.asyncio
    async def test_postgres_logging_async_exception_handling(self):
        """测试异步函数PostgreSQL日志记录的异常处理（覆盖行270->280）"""
        @with_postgres_logging
        async def async_failing_function(**kwargs) -> None:
            raise ValueError("异步PostgreSQL日志测试异常")
        
        mock_logger = Mock()
        with patch('harborai.api.decorators.get_postgres_logger', return_value=mock_logger):
            with pytest.raises(ValueError, match="异步PostgreSQL日志测试异常"):
                await async_failing_function(model="test-model", trace_id="error-trace")
            
            # 验证日志记录调用
            mock_logger.log_request.assert_called_once()
            mock_logger.log_response.assert_called_once()
            
            # 验证错误日志记录
            response_call_args = mock_logger.log_response.call_args[1]
            assert response_call_args['trace_id'] == "error-trace"
            assert response_call_args['response'] is None
            assert response_call_args['success'] is False
            assert response_call_args['error'] == "异步PostgreSQL日志测试异常"
    
    def test_postgres_logging_sync_with_messages_and_model(self):
        """测试同步函数PostgreSQL日志记录的完整流程"""
        @with_postgres_logging
        def sync_postgres_complete_function(
            messages: list, 
            model: str = "test-model", 
            trace_id: str = None,
            extra_param: str = "extra"
        ) -> dict:
            return {"result": "success", "processed_messages": len(messages)}
        
        with patch('harborai.api.decorators.get_postgres_logger') as mock_get_logger:
            with patch('harborai.api.decorators.generate_trace_id', return_value='complete-trace'):
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                
                test_messages = [{"role": "user", "content": "test"}]
                result = sync_postgres_complete_function(
                    messages=test_messages,
                    model="gpt-4",
                    extra_param="test_extra"
                )
                
                assert result["result"] == "success"
                assert result["processed_messages"] == 1
                
                # 验证请求日志记录
                mock_logger.log_request.assert_called_once()
                request_call_args = mock_logger.log_request.call_args[1]
                assert request_call_args['trace_id'] == 'complete-trace'
                assert request_call_args['model'] == 'gpt-4'
                assert request_call_args['messages'] == test_messages
                assert request_call_args['extra_param'] == 'test_extra'
                
                # 验证成功响应日志记录
                mock_logger.log_response.assert_called_once()
                response_call_args = mock_logger.log_response.call_args[1]
                assert response_call_args['trace_id'] == 'complete-trace'
                assert response_call_args['response'] == result
                assert response_call_args['success'] is True


class TestDecoratorEdgeCases:
    """测试装饰器的边界情况"""
    
    def test_cost_tracking_with_no_usage_attribute(self):
        """测试成本追踪装饰器处理没有usage属性的结果（覆盖行130->140）"""
        @cost_tracking
        def function_without_usage(**kwargs) -> str:
            return "没有usage属性的结果"
        
        with patch('harborai.api.decorators.record_token_usage') as mock_record:
            result = function_without_usage(model="test-model", trace_id="no-usage")
            
            # 验证返回结果
            assert result == "没有usage属性的结果"
            
            # 验证没有调用record_token_usage（因为没有usage属性）
            mock_record.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_async_cost_tracking_with_no_usage_attribute(self):
        """测试异步函数结果对象没有usage属性的情况"""
        @cost_tracking
        async def async_function_without_usage(**kwargs) -> str:
            await asyncio.sleep(0.01)
            return "异步没有usage属性的结果"
        
        with patch('harborai.api.decorators.record_token_usage') as mock_record:
            result = await async_function_without_usage(model="async-test", trace_id="async-no-usage")
            
            # 验证函数正常执行
            assert result == "异步没有usage属性的结果"
            # 验证没有调用record_token_usage（因为没有usage属性）
            mock_record.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])