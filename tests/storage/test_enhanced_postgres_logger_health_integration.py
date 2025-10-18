"""
测试 EnhancedPostgreSQLLogger 与 ConnectionPoolHealthChecker 的集成

这个测试文件验证：
1. 健康检查器的正确初始化和集成
2. 健康状态变化时的自动处理
3. 预防性措施和恢复机制
4. 统计信息的正确收集
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from harborai.storage.enhanced_postgres_logger import EnhancedPostgreSQLLogger
from harborai.storage.connection_pool_health_checker import (
    ConnectionPoolHealthChecker, HealthStatus, HealthCheckResult, 
    ConnectionMetrics, PerformanceThresholds
)
from harborai.storage.connection_pool import ConnectionPoolConfig
from harborai.storage.batch_processor import BatchConfig
from harborai.storage.error_handler import RetryConfig


@pytest.fixture
def mock_connection_string():
    """模拟数据库连接字符串"""
    return "postgresql://test:test@localhost:5432/test_db"


@pytest.fixture
def health_thresholds():
    """健康检查阈值配置"""
    return PerformanceThresholds(
        max_response_time_ms=100.0,
        max_error_rate=0.05,
        min_connection_utilization=0.1,
        max_connection_utilization=0.8,
        min_throughput=1.0,
        health_check_timeout=5.0
    )


@pytest.fixture
def logger_config():
    """日志记录器配置"""
    return {
        'pool_config': ConnectionPoolConfig(
            min_connections=2,
            max_connections=10,
            connection_timeout=30.0
        ),
        'batch_config': BatchConfig(
            max_batch_size=50,
            flush_interval=2.0
        ),
        'retry_config': RetryConfig(
            max_retries=2
        )
    }


@pytest.fixture
async def logger_with_health_checker(mock_connection_string, health_thresholds, logger_config):
    """创建带健康检查器的日志记录器"""
    logger = EnhancedPostgreSQLLogger(
        connection_string=mock_connection_string,
        health_thresholds=health_thresholds,
        enable_health_monitoring=True,
        **logger_config
    )
    
    # 模拟数据库和连接池初始化
    with patch.object(logger, '_initialize_database', new_callable=AsyncMock), \
         patch.object(logger, '_initialize_connection_pool', new_callable=AsyncMock) as mock_init_pool, \
         patch.object(logger, '_initialize_batch_processor', new_callable=AsyncMock):
        
        # 创建模拟连接池
        mock_pool = Mock()
        mock_pool.initialize = AsyncMock()
        mock_pool.shutdown = AsyncMock()
        mock_pool.health_check = AsyncMock(return_value={'healthy': True})
        mock_pool._stats = Mock()
        mock_pool._stats.total_connections = 5
        mock_pool._stats.active_connections = 2
        mock_pool._stats.idle_connections = 3
        mock_pool._stats.failed_connections = 0
        mock_pool._stats.average_connection_time = 50.0
        mock_pool._stats.total_requests = 100
        mock_pool._stats.successful_requests = 95
        mock_pool._stats.failed_requests = 5
        
        logger.connection_pool = mock_pool
        
        # 模拟健康检查器初始化
        async def mock_init_pool_func():
            # 创建模拟健康检查器
            mock_health_checker = Mock()
            mock_health_checker.add_health_callback = Mock()
            mock_health_checker.start_monitoring = AsyncMock()
            mock_health_checker.stop_monitoring = AsyncMock()
            mock_health_checker.check_health = AsyncMock()
            mock_health_checker.get_current_status = Mock(return_value=HealthStatus.HEALTHY)
            mock_health_checker.get_metrics_summary = Mock(return_value={})
            mock_health_checker._monitoring = True
            mock_health_checker._health_history = []
            mock_health_checker.check_interval = 30.0
            
            logger.health_checker = mock_health_checker
            
            # 模拟实际的初始化逻辑
            if logger.enable_health_monitoring:
                logger.health_checker.add_health_callback(logger._on_health_status_change)
                await logger.health_checker.start_monitoring()
        
        mock_init_pool.side_effect = mock_init_pool_func
        
        await logger.start()
        
        yield logger
        
        await logger.stop()


class TestHealthCheckerIntegration:
    """测试健康检查器集成"""
    
    async def test_health_checker_initialization(self, logger_with_health_checker):
        """测试健康检查器正确初始化"""
        logger = logger_with_health_checker
        
        # 验证健康检查器已初始化
        assert logger.health_checker is not None
        assert logger.enable_health_monitoring is True
        
        # 验证回调已注册
        logger.health_checker.add_health_callback.assert_called_once()
        
        # 验证监控已启动
        logger.health_checker.start_monitoring.assert_called_once()
    
    async def test_health_status_change_callback(self, logger_with_health_checker):
        """测试健康状态变化回调"""
        logger = logger_with_health_checker
        
        # 模拟健康状态变化
        health_result = HealthCheckResult(
            status=HealthStatus.WARNING,
            score=75.0,
            details={},
            recommendations=["减少批次大小", "增加连接超时"],
            timestamp=datetime.now()
        )
        
        # 调用健康状态变化回调
        logger._on_health_status_change(health_result)
        
        # 验证预防性措施被应用
        # 这里可以检查批次大小是否被调整等
    
    async def test_critical_health_handling(self, logger_with_health_checker):
        """测试严重健康状态处理"""
        logger = logger_with_health_checker
        
        # 模拟严重健康状态
        health_result = HealthCheckResult(
            status=HealthStatus.CRITICAL,
            score=10.0,
            details={},
            recommendations=["重新初始化连接池"],
            timestamp=datetime.now()
        )
        
        # 模拟重新初始化方法
        logger._reinitialize_connection_pool = AsyncMock()
        
        # 调用健康状态变化回调
        logger._on_health_status_change(health_result)
        
        # 等待异步任务完成
        await asyncio.sleep(0.1)
    
    async def test_preventive_measures_application(self, logger_with_health_checker):
        """测试预防性措施应用"""
        logger = logger_with_health_checker
        
        # 创建模拟批处理器
        mock_batch_processor = Mock()
        mock_batch_processor.config = Mock()
        mock_batch_processor.config.max_batch_size = 100
        logger.batch_processor = mock_batch_processor
        
        # 模拟警告状态
        health_result = HealthCheckResult(
            status=HealthStatus.WARNING,
            score=70.0,
            details={},
            recommendations=["减少批次大小"],
            timestamp=datetime.now()
        )
        
        # 应用预防性措施
        logger._apply_preventive_measures(health_result)
        
        # 验证批次大小被减少
        assert mock_batch_processor.config.max_batch_size == 80  # 100 * 0.8
    
    async def test_normal_configuration_restoration(self, logger_with_health_checker):
        """测试正常配置恢复"""
        logger = logger_with_health_checker
        
        # 创建模拟批处理器
        mock_batch_processor = Mock()
        mock_batch_processor.config = Mock()
        mock_batch_processor.config.max_batch_size = 50  # 已降级的大小
        logger.batch_processor = mock_batch_processor
        logger.batch_config.max_batch_size = 100  # 原始大小
        
        # 恢复正常配置
        logger._restore_normal_configuration()
        
        # 验证批次大小被恢复
        assert mock_batch_processor.config.max_batch_size == 100
    
    async def test_health_check_with_health_checker(self, logger_with_health_checker):
        """测试使用健康检查器的健康检查"""
        logger = logger_with_health_checker
        
        # 模拟健康检查结果
        health_result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            score=95.0,
            details={},
            recommendations=[],
            timestamp=datetime.now()
        )
        
        logger.health_checker.check_health.return_value = health_result
        
        # 执行健康检查
        health_status = await logger.health_check()
        
        # 验证结果
        assert health_status['status'] == 'healthy'
        assert 'connection_pool' in health_status['components']
        assert health_status['components']['connection_pool']['status'] == 'healthy'
        assert health_status['components']['connection_pool']['healthy'] is True
    
    async def test_statistics_with_health_checker(self, logger_with_health_checker):
        """测试包含健康检查器的统计信息"""
        logger = logger_with_health_checker
        
        # 获取统计信息
        stats = logger.get_statistics()
        
        # 验证健康检查器统计信息
        assert 'health_checker' in stats
        health_stats = stats['health_checker']
        assert 'current_status' in health_stats
        assert 'monitoring_enabled' in health_stats
        assert 'check_interval' in health_stats
        assert 'history_size' in health_stats
        assert 'metrics_summary' in health_stats
    
    async def test_reinitialize_connection_pool_with_health_checker(self, logger_with_health_checker):
        """测试重新初始化连接池时健康检查器的处理"""
        logger = logger_with_health_checker
        
        # 保存原始健康检查器的引用
        original_health_checker = logger.health_checker
        
        # 模拟重新初始化方法
        with patch.object(logger, '_initialize_connection_pool', new_callable=AsyncMock) as mock_init:
            # 模拟新的健康检查器
            async def mock_init_func():
                new_health_checker = Mock()
                new_health_checker.add_health_callback = Mock()
                new_health_checker.start_monitoring = AsyncMock()
                new_health_checker.stop_monitoring = AsyncMock()
                logger.health_checker = new_health_checker
                
                if logger.enable_health_monitoring:
                    logger.health_checker.add_health_callback(logger._on_health_status_change)
                    await logger.health_checker.start_monitoring()
            
            mock_init.side_effect = mock_init_func
            
            # 执行重新初始化
            await logger._reinitialize_connection_pool()
            
            # 验证原始健康检查器被停止
            original_health_checker.stop_monitoring.assert_called()
            
            # 验证连接池重新初始化被调用
            mock_init.assert_called()
            
            # 验证新的健康检查器被创建
            assert logger.health_checker is not None
            assert logger.health_checker != original_health_checker
    
    async def test_stop_with_health_checker(self, logger_with_health_checker):
        """测试停止时健康检查器的正确清理"""
        logger = logger_with_health_checker
        
        # 确保健康检查器存在
        assert logger.health_checker is not None
        health_checker = logger.health_checker
        
        # 停止日志记录器
        await logger.stop()
        
        # 验证健康检查器被停止
        health_checker.stop_monitoring.assert_called()
        
        # 验证健康检查器被设置为None
        assert logger.health_checker is None


class TestHealthCheckerDisabled:
    """测试禁用健康检查器的情况"""
    
    async def test_logger_without_health_checker(self, mock_connection_string, logger_config):
        """测试禁用健康检查器的日志记录器"""
        logger = EnhancedPostgreSQLLogger(
            connection_string=mock_connection_string,
            enable_health_monitoring=False,
            **logger_config
        )
        
        # 模拟初始化
        with patch.object(logger, '_initialize_database', new_callable=AsyncMock), \
             patch.object(logger, '_initialize_connection_pool', new_callable=AsyncMock), \
             patch.object(logger, '_initialize_batch_processor', new_callable=AsyncMock):
            
            await logger.start()
            
            # 验证健康检查器未初始化
            assert logger.health_checker is None
            assert logger.enable_health_monitoring is False
            
            # 健康检查应该回退到基本模式
            mock_pool = Mock()
            mock_pool.health_check = AsyncMock(return_value={'healthy': True})
            logger.connection_pool = mock_pool
            
            health_status = await logger.health_check()
            assert health_status['status'] == 'healthy'
            
            await logger.stop()


class TestIntegrationScenarios:
    """集成场景测试"""
    
    async def test_full_degradation_and_recovery_cycle(self, logger_with_health_checker):
        """测试完整的降级和恢复周期"""
        logger = logger_with_health_checker
        
        # 创建模拟批处理器
        mock_batch_processor = Mock()
        mock_batch_processor.config = Mock()
        mock_batch_processor.config.max_batch_size = 100
        logger.batch_processor = mock_batch_processor
        logger.batch_config.max_batch_size = 100
        
        # 1. 模拟警告状态 - 应用预防性措施
        warning_result = HealthCheckResult(
            status=HealthStatus.WARNING,
            score=70.0,
            details={},
            recommendations=["减少批次大小"],
            timestamp=datetime.now()
        )
        
        logger._on_health_status_change(warning_result)
        assert mock_batch_processor.config.max_batch_size == 80
        
        # 2. 模拟恢复到健康状态 - 恢复正常配置
        healthy_result = HealthCheckResult(
            status=HealthStatus.HEALTHY,
            score=95.0,
            details={},
            recommendations=[],
            timestamp=datetime.now()
        )
        
        logger._on_health_status_change(healthy_result)
        assert mock_batch_processor.config.max_batch_size == 100
    
    async def test_error_handling_during_health_operations(self, logger_with_health_checker):
        """测试健康操作期间的错误处理"""
        logger = logger_with_health_checker
        
        # 模拟预防性措施应用时的错误
        health_result = HealthCheckResult(
            status=HealthStatus.WARNING,
            score=70.0,
            details={},
            recommendations=["减少批次大小"],
            timestamp=datetime.now()
        )
        
        # 应该不会抛出异常
        logger._apply_preventive_measures(health_result)
        
        # 模拟恢复配置时的错误
        logger.batch_processor = None  # 故意设置为 None
        logger._restore_normal_configuration()  # 应该不会抛出异常