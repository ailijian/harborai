#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerformanceTestController 单元测试

本模块包含对性能测试控制器的基础单元测试。

作者: HarborAI Team
创建时间: 2024-01-20
遵循: VIBE Coding 规范
"""

import pytest
import asyncio
import time
from unittest.mock import Mock
from datetime import datetime

try:
    from .core_performance_framework import (
        PerformanceTestController,
        PerformanceConfig,
        TestType,
        TestStatus,
        TestMetrics
    )
except ImportError:
    from core_performance_framework import (
        PerformanceTestController,
        PerformanceConfig,
        TestType,
        TestStatus,
        TestMetrics
    )


class TestPerformanceTestControllerUnit:
    """PerformanceTestController 单元测试类"""
    
    @pytest.fixture
    def default_config(self):
        """默认配置fixture"""
        return PerformanceConfig(
            test_duration=10.0,
            warmup_duration=2.0,
            cooldown_duration=1.0,
            max_concurrent_users=5,
            response_time_threshold=1.0,
            cpu_usage_threshold=70.0
        )
    
    @pytest.fixture
    def controller(self, default_config):
        """性能测试控制器fixture"""
        return PerformanceTestController(default_config)
    
    @pytest.fixture
    def mock_test_runner(self):
        """模拟测试运行器fixture"""
        async def mock_runner(**kwargs):
            await asyncio.sleep(0.1)
            return {
                'response_time': 0.5,
                'success_rate': 0.99,
                'throughput': 100.0,
                'error_count': 1
            }
        return mock_runner

    def test_controller_initialization(self, default_config):
        """测试控制器初始化"""
        controller = PerformanceTestController(default_config)
        
        assert controller.config == default_config
        assert controller.results_collector is not None
        assert len(controller.test_runners) == 0
        assert controller.current_test is None
        assert controller.is_running is False
    
    def test_config_validation(self):
        """测试配置验证"""
        # 测试有效配置
        valid_config = PerformanceConfig(test_duration=30.0)
        errors = valid_config.validate()
        assert len(errors) == 0
        
        # 测试无效配置
        invalid_config = PerformanceConfig(
            test_duration=-10.0,
            warmup_duration=-5.0,
            max_concurrent_users=0
        )
        errors = invalid_config.validate()
        assert len(errors) > 0
    
    def test_register_test_runner(self, controller, mock_test_runner):
        """测试注册测试运行器"""
        controller.register_test_runner(TestType.RESPONSE_TIME, mock_test_runner)
        
        assert TestType.RESPONSE_TIME in controller.test_runners
        assert controller.test_runners[TestType.RESPONSE_TIME] == mock_test_runner
    
    @pytest.mark.asyncio
    async def test_run_single_test_success(self, controller, mock_test_runner):
        """测试成功运行单个测试"""
        controller.register_test_runner(TestType.RESPONSE_TIME, mock_test_runner)
        
        metrics = await controller.run_single_test(
            TestType.RESPONSE_TIME, 
            "测试响应时间",
            target_url="http://example.com"
        )
        
        assert isinstance(metrics, TestMetrics)
        assert metrics.test_name == "测试响应时间"
        assert metrics.test_type == TestType.RESPONSE_TIME
        assert metrics.status == TestStatus.COMPLETED
        assert metrics.response_time == 0.5
        assert metrics.success_rate == 0.99
        assert metrics.duration is not None
    
    def test_get_test_status_idle(self, controller):
        """测试获取空闲状态"""
        status = controller.get_test_status()
        
        assert status['is_running'] is False
        assert status['current_test'] is None
        assert status['total_tests_run'] == 0
    
    def test_stop_current_test(self, controller):
        """测试停止当前测试"""
        controller.current_test = {
            'test_name': '测试停止功能',
            'start_time': datetime.now()
        }
        controller.is_running = True
        
        controller.stop_current_test()
        
        assert controller.is_running is False
        assert controller.current_test is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])