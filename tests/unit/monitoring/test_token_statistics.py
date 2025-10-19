#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Token统计模块测试"""

import pytest
import json
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from collections import deque

from harborai.monitoring.token_statistics import (
    TokenUsageRecord,
    ModelStatistics,
    TimeWindowStatistics,
    TokenStatisticsCollector,
    get_token_statistics_collector,
    record_token_usage
)


class TestTokenUsageRecord:
    """TokenUsageRecord数据类测试"""
    
    def test_token_usage_record_creation(self):
        """测试TokenUsageRecord创建"""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        record = TokenUsageRecord(
            timestamp=timestamp,
            model_name="gpt-4",
            provider="openai",
            request_id="test_request_123",
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            cost=0.75,
            latency_ms=2500.0,
            success=True,
            error_message=None
        )
        
        assert record.timestamp == timestamp
        assert record.model_name == "gpt-4"
        assert record.provider == "openai"
        assert record.request_id == "test_request_123"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.total_tokens == 150
        assert record.cost == 0.75
        assert record.latency_ms == 2500.0
        assert record.success is True
        assert record.error_message is None
    
    def test_token_usage_record_with_error(self):
        """测试带错误的TokenUsageRecord"""
        timestamp = datetime(2024, 1, 1, 12, 0, 0)
        record = TokenUsageRecord(
            timestamp=timestamp,
            model_name="gpt-4",
            provider="openai",
            request_id="error_request_456",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost=0.0,
            latency_ms=1000.0,
            success=False,
            error_message="API调用失败"
        )
        
        assert record.success is False
        assert record.error_message == "API调用失败"
        assert record.cost == 0.0


class TestTokenStatisticsCollector:
    """TokenStatisticsCollector测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.collector = TokenStatisticsCollector(max_records=1000, cleanup_interval=3600)
    
    def test_collector_initialization(self):
        """测试收集器初始化"""
        assert self.collector.max_records == 1000
        assert self.collector.cleanup_interval == 3600
        assert len(self.collector._records) == 0
        assert len(self.collector._model_stats) == 0
    
    @patch('harborai.core.cost_tracking.PricingCalculator.calculate_cost')
    def test_record_usage_success(self, mock_calculate_cost):
        """测试记录成功使用"""
        mock_calculate_cost.return_value = 0.75
        
        self.collector.record_usage(
            trace_id="test_trace",
            model="gpt-4",
            input_tokens=100,
            output_tokens=50,
            duration=2.5,
            success=True
        )
        
        # 检查记录是否添加
        assert len(self.collector._records) == 1
        record = self.collector._records[0]
        assert record.request_id == "test_trace"
        assert record.model_name == "gpt-4"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.total_tokens == 150
        assert record.cost == 0.75
        assert record.latency_ms == 2500.0
        assert record.success is True
        assert record.error_message is None
        
        # 检查模型统计是否更新
        assert "gpt-4" in self.collector._model_stats
        model_stats = self.collector._model_stats["gpt-4"]
        assert model_stats.total_requests == 1
        assert model_stats.successful_requests == 1
        assert model_stats.failed_requests == 0
        
        mock_calculate_cost.assert_called_once_with(100, 50, "gpt-4")
    
    def test_record_usage_failure(self):
        """测试记录失败使用"""
        self.collector.record_usage(
            trace_id="error_trace",
            model="gpt-4",
            input_tokens=0,
            output_tokens=0,
            duration=1.0,
            success=False,
            error="API调用失败"
        )
        
        # 检查记录是否添加
        assert len(self.collector._records) == 1
        record = self.collector._records[0]
        assert record.request_id == "error_trace"
        assert record.success is False
        assert record.error_message == "API调用失败"
        assert record.cost == 0.0
        
        # 检查模型统计是否更新
        assert "gpt-4" in self.collector._model_stats
        model_stats = self.collector._model_stats["gpt-4"]
        assert model_stats.total_requests == 1
        assert model_stats.successful_requests == 0
        assert model_stats.failed_requests == 1


if __name__ == '__main__':
    pytest.main([__file__])