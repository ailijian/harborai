#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能配置测试

测试性能模式配置和功能开关的正确性。
"""

import pytest
from unittest.mock import MagicMock, patch
from harborai.config.performance import (
    PerformanceMode,
    FeatureFlags,
    PerformanceConfig,
    get_performance_config,
    reset_performance_config
)
from harborai.config.settings import Settings


class TestPerformanceMode:
    """性能模式枚举测试"""
    
    def test_performance_mode_values(self):
        """测试性能模式枚举值"""
        assert PerformanceMode.FAST.value == "fast"
        assert PerformanceMode.BALANCED.value == "balanced"
        assert PerformanceMode.FULL.value == "full"
    
    def test_performance_mode_from_string(self):
        """测试从字符串创建性能模式"""
        assert PerformanceMode("fast") == PerformanceMode.FAST
        assert PerformanceMode("balanced") == PerformanceMode.BALANCED
        assert PerformanceMode("full") == PerformanceMode.FULL


class TestFeatureFlags:
    """功能开关测试"""
    
    def test_default_feature_flags(self):
        """测试默认功能开关配置"""
        flags = FeatureFlags()
        
        # 核心功能默认启用
        assert flags.enable_cost_tracking is True
        assert flags.enable_prometheus_metrics is True
        assert flags.enable_opentelemetry is True
        assert flags.enable_postgres_logging is True
        
        # 性能优化默认启用
        assert flags.enable_fast_path is True
        assert flags.enable_async_decorators is True
        assert flags.enable_plugin_preload is True
        
        # 调试功能默认禁用
        assert flags.enable_debug_mode is False
        assert flags.enable_performance_profiling is False
    
    def test_custom_feature_flags(self):
        """测试自定义功能开关配置"""
        flags = FeatureFlags(
            enable_cost_tracking=False,
            enable_fast_path=True,
            enable_debug_mode=True
        )
        
        assert flags.enable_cost_tracking is False
        assert flags.enable_fast_path is True
        assert flags.enable_debug_mode is True


class TestPerformanceConfig:
    """性能配置管理器测试"""
    
    @pytest.fixture
    def mock_settings(self):
        """模拟设置"""
        settings = MagicMock(spec=Settings)
        settings.performance_mode = "balanced"
        settings.enable_cost_tracking = True
        settings.enable_postgres_logging = True
        settings.enable_fast_path = True
        settings.enable_async_decorators = True
        settings.enable_plugin_preload = True
        settings.enable_response_cache = True
        settings.enable_token_cache = True
        settings.enable_background_tasks = True
        settings.debug = False
        settings.response_cache_ttl = 600
        settings.token_cache_ttl = 300
        settings.cache_cleanup_interval = 300
        settings.background_task_workers = 2
        settings.plugin_cache_size = 100
        settings.plugin_directories = ["harborai.core.plugins"]
        settings.is_fast_path_enabled = MagicMock(return_value=True)
        return settings
    
    def test_fast_mode_configuration(self, mock_settings):
        """测试FAST模式配置"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig(PerformanceMode.FAST)
            
            # FAST模式应该禁用大部分功能以提升性能
            assert config.feature_flags.enable_cost_tracking is False
            assert config.feature_flags.enable_prometheus_metrics is False
            assert config.feature_flags.enable_opentelemetry is False
            assert config.feature_flags.enable_postgres_logging is False
            assert config.feature_flags.enable_detailed_logging is False
            
            # 但应该启用性能优化功能
            assert config.feature_flags.enable_fast_path is True
            assert config.feature_flags.enable_async_decorators is True
            assert config.feature_flags.enable_background_tasks is True
    
    def test_balanced_mode_configuration(self, mock_settings):
        """测试BALANCED模式配置"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig(PerformanceMode.BALANCED)
            
            # BALANCED模式应该平衡功能和性能
            assert config.feature_flags.enable_cost_tracking is True  # 从settings继承
            assert config.feature_flags.enable_prometheus_metrics is True
            assert config.feature_flags.enable_opentelemetry is True
            assert config.feature_flags.enable_postgres_logging is True  # 从settings继承
            assert config.feature_flags.enable_detailed_logging is False  # 为性能禁用
            
            # 性能优化功能全部启用
            assert config.feature_flags.enable_fast_path is True
            assert config.feature_flags.enable_async_decorators is True
    
    def test_full_mode_configuration(self, mock_settings):
        """测试FULL模式配置"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig(PerformanceMode.FULL)
            
            # FULL模式应该启用所有功能
            assert config.feature_flags.enable_cost_tracking is True
            assert config.feature_flags.enable_prometheus_metrics is True
            assert config.feature_flags.enable_opentelemetry is True
            assert config.feature_flags.enable_postgres_logging is True
            assert config.feature_flags.enable_detailed_logging is True
            
            # 性能优化功能也应该启用
            assert config.feature_flags.enable_fast_path is True
            assert config.feature_flags.enable_async_decorators is True
    
    def test_get_decorator_config(self, mock_settings):
        """测试装饰器配置获取"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            # 测试FAST模式
            config = PerformanceConfig(PerformanceMode.FAST)
            decorator_config = config.get_decorator_config()
            
            assert decorator_config["cost_tracking"] is False
            assert decorator_config["postgres_logging"] is False
            assert decorator_config["detailed_tracing"] is False
            assert decorator_config["async_decorators"] is True
            assert decorator_config["prometheus_metrics"] is False
    
    def test_get_middleware_config(self, mock_settings):
        """测试中间件配置获取"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig(PerformanceMode.BALANCED)
            middleware_config = config.get_middleware_config()
            
            assert "fast_path" in middleware_config
            assert "cost_tracking_middleware" in middleware_config
            assert "logging_middleware" in middleware_config
            assert "metrics_middleware" in middleware_config
            assert "tracing_middleware" in middleware_config
    
    def test_get_cache_config(self, mock_settings):
        """测试缓存配置获取"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig(PerformanceMode.BALANCED)
            cache_config = config.get_cache_config()
            
            assert "response_cache" in cache_config
            assert "token_cache" in cache_config
            assert cache_config["response_cache"]["enabled"] is True
            assert cache_config["token_cache"]["enabled"] is True
            assert cache_config["response_cache"]["ttl"] == 600
            assert cache_config["token_cache"]["ttl"] == 300
    
    def test_get_background_task_config(self, mock_settings):
        """测试后台任务配置获取"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig(PerformanceMode.BALANCED)
            bg_config = config.get_background_task_config()
            
            assert bg_config["enabled"] is True
            assert bg_config["workers"] == 2
            assert bg_config["async_cost_tracking"] is True
            assert bg_config["batch_processing"] is True
    
    def test_get_plugin_config(self, mock_settings):
        """测试插件配置获取"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig(PerformanceMode.BALANCED)
            plugin_config = config.get_plugin_config()
            
            assert plugin_config["preload"] is True
            assert plugin_config["cache_size"] == 100
            assert plugin_config["directories"] == ["harborai.core.plugins"]
    
    def test_should_use_fast_path(self, mock_settings):
        """测试快速路径判断逻辑"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            # FAST模式：总是使用快速路径
            fast_config = PerformanceConfig(PerformanceMode.FAST)
            assert fast_config.should_use_fast_path("gpt-4", 2000) is True
            
            # FULL模式：从不使用快速路径
            full_config = PerformanceConfig(PerformanceMode.FULL)
            assert full_config.should_use_fast_path("gpt-3.5-turbo", 500) is False
            
            # BALANCED模式：根据settings判断
            balanced_config = PerformanceConfig(PerformanceMode.BALANCED)
            assert balanced_config.should_use_fast_path("gpt-3.5-turbo", 500) is True
            mock_settings.is_fast_path_enabled.assert_called_with("gpt-3.5-turbo", 500)
    
    def test_should_use_fast_path_disabled(self, mock_settings):
        """测试快速路径禁用时的行为"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig(PerformanceMode.FAST)
            config.feature_flags.enable_fast_path = False
            
            assert config.should_use_fast_path("gpt-3.5-turbo", 500) is False
    
    def test_get_performance_summary(self, mock_settings):
        """测试性能配置摘要获取"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig(PerformanceMode.FAST)
            summary = config.get_performance_summary()
            
            assert summary["mode"] == "fast"
            assert "feature_flags" in summary
            assert "expected_improvements" in summary
            
            # 验证预期改善信息
            improvements = summary["expected_improvements"]
            assert "first_token_time" in improvements
            assert "total_improvement" in improvements
            assert "resource_usage" in improvements
    
    def test_expected_improvements_by_mode(self, mock_settings):
        """测试不同模式的预期改善"""
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            # FAST模式应该有最大改善
            fast_config = PerformanceConfig(PerformanceMode.FAST)
            fast_improvements = fast_config._get_expected_improvements()
            assert "2000-3000ms" in fast_improvements["first_token_time"]
            assert "70-80%" in fast_improvements["total_improvement"]
            
            # BALANCED模式应该有中等改善
            balanced_config = PerformanceConfig(PerformanceMode.BALANCED)
            balanced_improvements = balanced_config._get_expected_improvements()
            assert "1000-2000ms" in balanced_improvements["first_token_time"]
            assert "40-60%" in balanced_improvements["total_improvement"]
            
            # FULL模式应该有最小改善
            full_config = PerformanceConfig(PerformanceMode.FULL)
            full_improvements = full_config._get_expected_improvements()
            assert "500-1000ms" in full_improvements["first_token_time"]
            assert "20-30%" in full_improvements["total_improvement"]


class TestGlobalPerformanceConfig:
    """全局性能配置测试"""
    
    def test_get_performance_config_singleton(self):
        """测试全局性能配置单例模式"""
        # 重置全局配置
        reset_performance_config()
        
        config1 = get_performance_config()
        config2 = get_performance_config()
        
        # 应该返回同一个实例
        assert config1 is config2
    
    def test_reset_performance_config(self):
        """测试重置性能配置"""
        # 获取初始配置
        config1 = get_performance_config()
        
        # 重置配置
        config2 = reset_performance_config(PerformanceMode.FAST)
        
        # 应该是不同的实例
        assert config1 is not config2
        assert config2.mode == PerformanceMode.FAST
        
        # 再次获取应该返回新的配置
        config3 = get_performance_config()
        assert config2 is config3


class TestPerformanceConfigIntegration:
    """性能配置集成测试"""
    
    def test_performance_mode_from_settings(self):
        """测试从settings读取性能模式"""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.performance_mode = "fast"
        mock_settings.enable_cost_tracking = True
        mock_settings.enable_postgres_logging = True
        mock_settings.enable_fast_path = True
        mock_settings.enable_async_decorators = True
        mock_settings.enable_plugin_preload = True
        mock_settings.enable_response_cache = True
        mock_settings.enable_token_cache = True
        mock_settings.enable_background_tasks = True
        mock_settings.debug = False
        
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig()
            assert config.mode == PerformanceMode.FAST
    
    def test_feature_flags_override_settings(self):
        """测试功能开关覆盖settings配置"""
        mock_settings = MagicMock(spec=Settings)
        mock_settings.performance_mode = "fast"
        mock_settings.enable_cost_tracking = True  # settings中启用
        mock_settings.enable_postgres_logging = True
        mock_settings.enable_fast_path = True
        mock_settings.enable_async_decorators = True
        mock_settings.enable_plugin_preload = True
        mock_settings.enable_response_cache = True
        mock_settings.enable_token_cache = True
        mock_settings.enable_background_tasks = True
        mock_settings.debug = False
        
        with patch('harborai.config.settings.get_settings', return_value=mock_settings):
            config = PerformanceConfig(PerformanceMode.FAST)
            
            # FAST模式应该覆盖settings，禁用成本追踪
            assert config.feature_flags.enable_cost_tracking is False
            
            decorator_config = config.get_decorator_config()
            assert decorator_config["cost_tracking"] is False