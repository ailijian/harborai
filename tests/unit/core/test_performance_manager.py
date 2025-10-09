"""性能管理器测试

测试性能管理器的组件管理、统计收集和优化功能。
"""

import asyncio
import pytest
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, Any

from harborai.core.performance_manager import (
    PerformanceStats,
    PerformanceManager,
    get_performance_manager,
    initialize_performance_manager,
    cleanup_performance_manager,
    get_system_performance_stats,
    perform_system_health_check,
    optimize_system_performance,
    _performance_manager
)


class TestPerformanceStats:
    """测试性能统计信息类"""
    
    def test_performance_stats_creation(self):
        """测试性能统计信息创建"""
        stats = PerformanceStats()
        
        assert stats.startup_time == 0.0
        assert stats.total_requests == 0
        assert stats.avg_response_time == 0.0
        assert stats.cache_hit_rate == 0.0
        assert stats.error_rate == 0.0
        assert stats.cost_tracking_stats == {}
        assert stats.plugin_stats == {}
        assert stats.background_task_stats == {}
    
    def test_performance_stats_with_values(self):
        """测试带值的性能统计信息"""
        stats = PerformanceStats(
            startup_time=1.5,
            total_requests=100,
            avg_response_time=0.25,
            cache_hit_rate=0.85,
            error_rate=0.02,
            cost_tracking_stats={"total_cost": 10.5},
            plugin_stats={"loaded": 3},
            background_task_stats={"processed": 50}
        )
        
        assert stats.startup_time == 1.5
        assert stats.total_requests == 100
        assert stats.avg_response_time == 0.25
        assert stats.cache_hit_rate == 0.85
        assert stats.error_rate == 0.02
        assert stats.cost_tracking_stats == {"total_cost": 10.5}
        assert stats.plugin_stats == {"loaded": 3}
        assert stats.background_task_stats == {"processed": 50}
    
    def test_performance_stats_to_dict(self):
        """测试性能统计信息转换为字典"""
        stats = PerformanceStats(
            startup_time=2.0,
            total_requests=200,
            avg_response_time=0.3,
            cache_hit_rate=0.9,
            error_rate=0.01
        )
        
        result = stats.to_dict()
        
        assert result["startup_time"] == 2.0
        assert result["total_requests"] == 200
        assert result["avg_response_time"] == 0.3
        assert result["cache_hit_rate"] == 0.9
        assert result["error_rate"] == 0.01
        assert result["cost_tracking_stats"] == {}
        assert result["plugin_stats"] == {}
        assert result["background_task_stats"] == {}


class TestPerformanceManager:
    """测试性能管理器"""
    
    def setup_method(self):
        """每个测试前的设置"""
        # 重置全局性能管理器
        global _performance_manager
        _performance_manager = None
    
    @patch('harborai.core.performance_manager.get_settings')
    def test_performance_manager_creation(self, mock_get_settings):
        """测试性能管理器创建"""
        mock_settings = Mock()
        mock_settings.enable_async_decorators = True
        mock_settings.enable_token_cache = True
        mock_get_settings.return_value = mock_settings
        
        manager = PerformanceManager()
        
        assert manager.settings is mock_settings
        assert manager._initialized is False
        assert manager._startup_time == 0.0
        assert isinstance(manager._stats, PerformanceStats)
        assert manager._cost_tracker is None
        assert manager._background_processor is None
        assert manager._cache_manager is None
        assert manager._plugin_manager is None
    
    @patch('harborai.core.performance_manager.get_settings')
    @patch('harborai.core.performance_manager.get_async_cost_tracker')
    @patch('harborai.core.performance_manager.start_background_processor')
    @patch('harborai.core.performance_manager.start_cache_manager')
    @patch('harborai.core.performance_manager.start_optimized_plugin_manager')
    @patch('harborai.core.performance_manager.get_background_processor')
    @patch('harborai.core.performance_manager.get_cache_manager')
    @patch('harborai.core.performance_manager.get_optimized_plugin_manager')
    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        mock_get_plugin_manager,
        mock_get_cache_manager,
        mock_get_background_processor,
        mock_start_plugin_manager,
        mock_start_cache_manager,
        mock_start_background_processor,
        mock_get_cost_tracker,
        mock_get_settings
    ):
        """测试成功初始化"""
        # 设置模拟
        mock_settings = Mock()
        mock_settings.enable_async_decorators = True
        mock_settings.enable_token_cache = True
        mock_get_settings.return_value = mock_settings
        
        mock_cost_tracker = Mock()
        mock_background_processor = Mock()
        mock_cache_manager = Mock()
        mock_plugin_manager = Mock()
        
        mock_get_cost_tracker.return_value = mock_cost_tracker
        mock_get_background_processor.return_value = mock_background_processor
        mock_get_cache_manager.return_value = mock_cache_manager
        mock_get_plugin_manager.return_value = mock_plugin_manager
        
        # 设置异步函数
        mock_start_background_processor.return_value = asyncio.Future()
        mock_start_background_processor.return_value.set_result(None)
        
        mock_start_cache_manager.return_value = asyncio.Future()
        mock_start_cache_manager.return_value.set_result(None)
        
        mock_start_plugin_manager.return_value = asyncio.Future()
        mock_start_plugin_manager.return_value.set_result(None)
        
        manager = PerformanceManager()
        
        # 初始化
        await manager.initialize()
        
        # 验证初始化状态
        assert manager._initialized is True
        assert manager.get_startup_time() >= 0
        assert manager._stats.startup_time >= 0
        assert manager._cost_tracker is mock_cost_tracker
        assert manager._background_processor is mock_background_processor
        assert manager._cache_manager is mock_cache_manager
        assert manager._plugin_manager is mock_plugin_manager
        
        # 验证调用
        mock_get_cost_tracker.assert_called_once()
        mock_start_background_processor.assert_called_once()
        mock_start_cache_manager.assert_called_once()
        mock_start_plugin_manager.assert_called_once()
    
    @patch('harborai.core.performance_manager.get_settings')
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self, mock_get_settings):
        """测试重复初始化"""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings
        
        manager = PerformanceManager()
        manager._initialized = True
        
        # 重复初始化应该直接返回
        await manager.initialize()
        
        assert manager._initialized is True
    
    @patch('harborai.core.performance_manager.get_settings')
    @patch('harborai.core.performance_manager.start_background_processor')
    @pytest.mark.asyncio
    async def test_initialize_failure(self, mock_start_background_processor, mock_get_settings):
        """测试初始化失败"""
        mock_settings = Mock()
        mock_settings.enable_async_decorators = False
        mock_settings.enable_token_cache = False
        mock_get_settings.return_value = mock_settings
        
        # 模拟初始化失败
        mock_start_background_processor.side_effect = Exception("初始化失败")
        
        manager = PerformanceManager()
        
        with patch.object(manager, 'cleanup') as mock_cleanup:
            mock_cleanup.return_value = asyncio.Future()
            mock_cleanup.return_value.set_result(None)
            
            with pytest.raises(Exception):
                await manager.initialize()
            
            mock_cleanup.assert_called_once()
        
        assert manager._initialized is False
    
    @patch('harborai.core.performance_manager.cleanup_async_cost_tracker')
    @patch('harborai.core.performance_manager.stop_background_processor')
    @patch('harborai.core.performance_manager.stop_cache_manager')
    @patch('harborai.core.performance_manager.stop_optimized_plugin_manager')
    @pytest.mark.asyncio
    async def test_cleanup(
        self,
        mock_stop_plugin_manager,
        mock_stop_cache_manager,
        mock_stop_background_processor,
        mock_cleanup_cost_tracker
    ):
        """测试清理资源"""
        # 设置异步函数
        mock_cleanup_cost_tracker.return_value = asyncio.Future()
        mock_cleanup_cost_tracker.return_value.set_result(None)
        
        mock_stop_background_processor.return_value = asyncio.Future()
        mock_stop_background_processor.return_value.set_result(None)
        
        mock_stop_cache_manager.return_value = asyncio.Future()
        mock_stop_cache_manager.return_value.set_result(None)
        
        mock_stop_plugin_manager.return_value = asyncio.Future()
        mock_stop_plugin_manager.return_value.set_result(None)
        
        manager = PerformanceManager()
        manager._initialized = True
        manager._cost_tracker = Mock()
        manager._cache_manager = Mock()
        manager._plugin_manager = Mock()
        
        # 清理
        await manager.cleanup()
        
        # 验证状态
        assert manager._initialized is False
        
        # 验证调用
        mock_cleanup_cost_tracker.assert_called_once()
        mock_stop_background_processor.assert_called_once()
        mock_stop_cache_manager.assert_called_once()
        mock_stop_plugin_manager.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_not_initialized(self):
        """测试清理未初始化的管理器"""
        manager = PerformanceManager()
        
        # 清理未初始化的管理器应该直接返回
        await manager.cleanup()
        
        assert manager._initialized is False
    
    def test_is_initialized(self):
        """测试检查初始化状态"""
        manager = PerformanceManager()
        
        assert manager.is_initialized() is False
        
        manager._initialized = True
        assert manager.is_initialized() is True
    
    def test_get_startup_time(self):
        """测试获取启动时间"""
        manager = PerformanceManager()
        
        assert manager.get_startup_time() == 0.0
        
        manager._startup_time = 1.5
        assert manager.get_startup_time() == 1.5
    
    @pytest.mark.asyncio
    async def test_get_performance_stats_not_initialized(self):
        """测试获取未初始化时的性能统计"""
        manager = PerformanceManager()
        
        stats = await manager.get_performance_stats()
        
        assert isinstance(stats, PerformanceStats)
        assert stats is manager._stats
    
    @pytest.mark.asyncio
    async def test_get_performance_stats_initialized(self):
        """测试获取已初始化时的性能统计"""
        manager = PerformanceManager()
        manager._initialized = True
        
        with patch.object(manager, '_update_stats') as mock_update_stats:
            mock_update_stats.return_value = asyncio.Future()
            mock_update_stats.return_value.set_result(None)
            
            stats = await manager.get_performance_stats()
            
            mock_update_stats.assert_called_once()
            assert isinstance(stats, PerformanceStats)
    
    @pytest.mark.asyncio
    async def test_get_performance_stats_error(self):
        """测试获取性能统计时发生错误"""
        manager = PerformanceManager()
        manager._initialized = True
        
        with patch.object(manager, '_update_stats') as mock_update_stats:
            mock_update_stats.side_effect = Exception("更新统计失败")
            
            stats = await manager.get_performance_stats()
            
            assert isinstance(stats, PerformanceStats)
    
    @pytest.mark.asyncio
    async def test_update_stats(self):
        """测试更新统计信息"""
        manager = PerformanceManager()
        
        # 设置模拟组件
        mock_cost_tracker = AsyncMock()
        mock_cost_tracker.get_cost_summary.return_value = {"total_cost": 10.5}
        
        mock_plugin_manager = Mock()
        mock_plugin_manager.get_performance_stats.return_value = {"loaded": 3}
        
        mock_background_processor = Mock()
        mock_background_processor.get_stats.return_value = {"processed": 50}
        
        mock_cache_manager = Mock()
        mock_cache_manager.get_stats.return_value = {
            "total_requests": 100,
            "cache_hits": 85
        }
        
        manager._cost_tracker = mock_cost_tracker
        manager._plugin_manager = mock_plugin_manager
        manager._background_processor = mock_background_processor
        manager._cache_manager = mock_cache_manager
        
        # 更新统计
        await manager._update_stats()
        
        # 验证统计信息
        assert manager._stats.cost_tracking_stats == {"total_cost": 10.5}
        assert manager._stats.plugin_stats == {"loaded": 3}
        assert manager._stats.background_task_stats == {"processed": 50}
        assert manager._stats.cache_hit_rate == 0.85
    
    @pytest.mark.asyncio
    async def test_update_stats_with_errors(self):
        """测试更新统计信息时发生错误"""
        manager = PerformanceManager()
        
        # 设置会抛出异常的模拟组件
        mock_cost_tracker = AsyncMock()
        mock_cost_tracker.get_cost_summary.side_effect = Exception("成本追踪错误")
        
        mock_plugin_manager = Mock()
        mock_plugin_manager.get_performance_stats.side_effect = Exception("插件错误")
        
        manager._cost_tracker = mock_cost_tracker
        manager._plugin_manager = mock_plugin_manager
        
        # 更新统计应该不抛出异常
        await manager._update_stats()
        
        # 统计信息应该保持默认值
        assert manager._stats.cost_tracking_stats == {}
        assert manager._stats.plugin_stats == {}
    
    @pytest.mark.asyncio
    async def test_health_check_not_initialized(self):
        """测试未初始化时的健康检查"""
        manager = PerformanceManager()
        
        health = await manager.health_check()
        
        assert health["status"] == "not_initialized"
        assert health["initialized"] is False
        assert health["startup_time"] == 0.0
        assert health["components"] == {}
    
    @pytest.mark.asyncio
    async def test_health_check_initialized(self):
        """测试已初始化时的健康检查"""
        manager = PerformanceManager()
        manager._initialized = True
        manager._startup_time = 1.5
        
        # 设置模拟组件
        mock_cost_tracker = Mock()
        mock_background_processor = Mock()
        mock_background_processor.get_stats.return_value = {
            "queue_size": 5,
            "processed_tasks": 100
        }
        mock_cache_manager = Mock()
        mock_cache_manager.get_stats.return_value = {"cache_size": 50}
        mock_plugin_manager = Mock()
        mock_plugin_manager.get_performance_stats.return_value = {
            "plugin_stats": {"plugin1": {}, "plugin2": {}}
        }
        
        manager._cost_tracker = mock_cost_tracker
        manager._background_processor = mock_background_processor
        manager._cache_manager = mock_cache_manager
        manager._plugin_manager = mock_plugin_manager
        
        health = await manager.health_check()
        
        assert health["status"] == "healthy"
        assert health["initialized"] is True
        assert health["startup_time"] == 1.5
        
        components = health["components"]
        assert components["cost_tracker"] == "healthy"
        assert components["background_processor"]["status"] == "healthy"
        assert components["background_processor"]["queue_size"] == 5
        assert components["background_processor"]["processed_tasks"] == 100
        assert components["cache_manager"]["status"] == "healthy"
        assert components["cache_manager"]["cache_size"] == 50
        assert components["plugin_manager"]["status"] == "healthy"
        assert components["plugin_manager"]["loaded_plugins"] == 2
    
    @pytest.mark.asyncio
    async def test_health_check_error(self):
        """测试健康检查时发生错误"""
        manager = PerformanceManager()
        manager._initialized = True
        
        # 设置会抛出异常的组件
        mock_background_processor = Mock()
        mock_background_processor.get_stats.side_effect = Exception("组件错误")
        manager._background_processor = mock_background_processor
        
        health = await manager.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health
    
    @pytest.mark.asyncio
    async def test_optimize_performance_not_initialized(self):
        """测试未初始化时的性能优化"""
        manager = PerformanceManager()
        
        result = await manager.optimize_performance()
        
        assert result["status"] == "error"
        assert result["message"] == "性能管理器未初始化"
    
    @pytest.mark.asyncio
    async def test_optimize_performance_success(self):
        """测试成功的性能优化"""
        manager = PerformanceManager()
        manager._initialized = True
        
        # 设置模拟组件
        mock_cache_manager = AsyncMock()
        mock_cost_tracker = AsyncMock()
        mock_background_processor = Mock()
        
        manager._cache_manager = mock_cache_manager
        manager._cost_tracker = mock_cost_tracker
        manager._background_processor = mock_background_processor
        
        result = await manager.optimize_performance()
        
        assert result["status"] == "success"
        assert "cache_cleanup" in result["optimizations"]
        assert "cost_tracking_flush" in result["optimizations"]
        assert "background_task_optimization" in result["optimizations"]
        
        # 验证调用
        mock_cache_manager.cleanup_expired.assert_called_once()
        mock_cost_tracker.flush_pending.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_optimize_performance_error(self):
        """测试性能优化时发生错误"""
        manager = PerformanceManager()
        manager._initialized = True
        
        # 设置会抛出异常的组件
        mock_cache_manager = AsyncMock()
        mock_cache_manager.cleanup_expired.side_effect = Exception("优化失败")
        manager._cache_manager = mock_cache_manager
        
        result = await manager.optimize_performance()
        
        assert result["status"] == "error"
        assert "error" in result


class TestGlobalFunctions:
    """测试全局函数"""
    
    def setup_method(self):
        """每个测试前重置全局状态"""
        global _performance_manager
        _performance_manager = None
    
    def test_get_performance_manager_singleton(self):
        """测试全局性能管理器单例"""
        manager1 = get_performance_manager()
        manager2 = get_performance_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, PerformanceManager)
    
    @pytest.mark.asyncio
    async def test_initialize_performance_manager(self):
        """测试初始化全局性能管理器"""
        with patch.object(PerformanceManager, 'initialize') as mock_initialize:
            mock_initialize.return_value = asyncio.Future()
            mock_initialize.return_value.set_result(None)
            
            await initialize_performance_manager()
            
            mock_initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_performance_manager(self):
        """测试清理全局性能管理器"""
        # 使用AsyncMock来模拟异步cleanup方法
        mock_manager = AsyncMock()
        
        with patch('harborai.core.performance_manager._performance_manager', mock_manager):
            await cleanup_performance_manager()
            
            mock_manager.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_performance_manager_none(self):
        """测试清理空的全局性能管理器"""
        global _performance_manager
        _performance_manager = None
        
        # 应该不抛出异常
        await cleanup_performance_manager()
        
        assert _performance_manager is None
    
    @pytest.mark.asyncio
    async def test_get_system_performance_stats_initialized(self):
        """测试获取已初始化的系统性能统计"""
        with patch('harborai.core.performance_manager.get_performance_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.is_initialized.return_value = True
            
            mock_stats = Mock()
            mock_stats.to_dict.return_value = {"startup_time": 1.5}
            
            mock_manager.get_performance_stats.return_value = asyncio.Future()
            mock_manager.get_performance_stats.return_value.set_result(mock_stats)
            
            mock_get_manager.return_value = mock_manager
            
            result = await get_system_performance_stats()
            
            assert result == {"startup_time": 1.5}
    
    @pytest.mark.asyncio
    async def test_get_system_performance_stats_not_initialized(self):
        """测试获取未初始化的系统性能统计"""
        with patch('harborai.core.performance_manager.get_performance_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.is_initialized.return_value = False
            mock_get_manager.return_value = mock_manager
            
            result = await get_system_performance_stats()
            
            assert result == {"status": "not_initialized"}
    
    @pytest.mark.asyncio
    async def test_perform_system_health_check(self):
        """测试执行系统健康检查"""
        with patch('harborai.core.performance_manager.get_performance_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.health_check.return_value = asyncio.Future()
            mock_manager.health_check.return_value.set_result({"status": "healthy"})
            mock_get_manager.return_value = mock_manager
            
            result = await perform_system_health_check()
            
            assert result == {"status": "healthy"}
            mock_manager.health_check.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_optimize_system_performance(self):
        """测试优化系统性能"""
        with patch('harborai.core.performance_manager.get_performance_manager') as mock_get_manager:
            mock_manager = Mock()
            mock_manager.optimize_performance.return_value = asyncio.Future()
            mock_manager.optimize_performance.return_value.set_result({"status": "success"})
            mock_get_manager.return_value = mock_manager
            
            result = await optimize_system_performance()
            
            assert result == {"status": "success"}
            mock_manager.optimize_performance.assert_called_once()


class TestPerformanceManagerIntegration:
    """测试性能管理器集成场景"""
    
    def setup_method(self):
        """每个测试前重置全局状态"""
        global _performance_manager
        _performance_manager = None
    
    @patch('harborai.core.performance_manager.get_settings')
    @pytest.mark.asyncio
    async def test_full_lifecycle(self, mock_get_settings):
        """测试完整生命周期"""
        mock_settings = Mock()
        mock_settings.enable_async_decorators = True
        mock_settings.enable_token_cache = True
        mock_get_settings.return_value = mock_settings
        
        # 创建模拟组件
        mock_cost_tracker = AsyncMock()
        mock_cost_tracker.flush_pending.return_value = None
        mock_cost_tracker.get_cost_summary.return_value = {"total_cost": 10.5}
        
        mock_cache_manager = Mock()  # 改为普通Mock
        mock_cache_manager.cleanup_expired = AsyncMock()
        mock_cache_manager.get_stats.return_value = {"cache_size": 50}
        
        mock_background_processor = Mock()
        mock_background_processor.get_stats.return_value = {
            "processed": 100,
            "queue_size": 0,
            "processed_tasks": 100
        }
        
        mock_plugin_manager = Mock()
        mock_plugin_manager.get_performance_stats.return_value = {
            "plugins": 5,
            "plugin_stats": {"plugin1": {}, "plugin2": {}}
        }
        
        with patch('harborai.core.performance_manager.get_async_cost_tracker', return_value=mock_cost_tracker), \
             patch('harborai.core.performance_manager.start_background_processor'), \
             patch('harborai.core.performance_manager.start_cache_manager'), \
             patch('harborai.core.performance_manager.start_optimized_plugin_manager'), \
             patch('harborai.core.performance_manager.get_background_processor', return_value=mock_background_processor), \
             patch('harborai.core.performance_manager.get_cache_manager', return_value=mock_cache_manager), \
             patch('harborai.core.performance_manager.get_optimized_plugin_manager', return_value=mock_plugin_manager), \
             patch('harborai.core.performance_manager.cleanup_async_cost_tracker'), \
             patch('harborai.core.performance_manager.stop_background_processor'), \
             patch('harborai.core.performance_manager.stop_cache_manager'), \
             patch('harborai.core.performance_manager.stop_optimized_plugin_manager'):
            
            # 初始化
            await initialize_performance_manager()
            
            manager = get_performance_manager()
            assert manager.is_initialized() is True
            
            # 获取统计信息
            stats = await get_system_performance_stats()
            assert isinstance(stats, dict)
            
            # 健康检查
            health = await perform_system_health_check()
            assert health["status"] == "healthy"
            
            # 性能优化
            optimization = await optimize_system_performance()
            assert optimization["status"] == "success"
            
            # 清理
            await cleanup_performance_manager()
            
            # 验证清理后状态
            global _performance_manager
            assert _performance_manager is None


class TestPerformanceManagerConcurrency:
    """测试性能管理器并发场景"""
    
    def setup_method(self):
        """每个测试前重置全局状态"""
        global _performance_manager
        _performance_manager = None
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """测试并发操作"""
        manager = PerformanceManager()
        manager._initialized = True
        
        # 设置模拟组件
        mock_cost_tracker = AsyncMock()
        mock_cost_tracker.get_cost_summary.return_value = {"total_cost": 10.5}
        mock_cost_tracker.flush_pending.return_value = None
        
        mock_cache_manager = AsyncMock()
        mock_cache_manager.get_stats.return_value = {"cache_size": 50}
        mock_cache_manager.cleanup_expired.return_value = None
        
        mock_background_processor = Mock()
        mock_background_processor.get_stats.return_value = {"processed": 100}
        
        manager._cost_tracker = mock_cost_tracker
        manager._cache_manager = mock_cache_manager
        manager._background_processor = mock_background_processor
        
        # 并发执行多个操作
        tasks = [
            manager.get_performance_stats(),
            manager.health_check(),
            manager.optimize_performance(),
            manager.get_performance_stats(),
            manager.health_check()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # 验证所有操作都成功完成
        assert len(results) == 5
        assert all(result is not None for result in results)
    
    @pytest.mark.asyncio
    async def test_concurrent_initialization(self):
        """测试并发初始化"""
        with patch('harborai.core.performance_manager.get_settings') as mock_get_settings:
            mock_settings = Mock()
            mock_settings.enable_async_decorators = False
            mock_settings.enable_token_cache = False
            mock_get_settings.return_value = mock_settings
            
            with patch('harborai.core.performance_manager.start_background_processor'), \
                 patch('harborai.core.performance_manager.start_optimized_plugin_manager'), \
                 patch('harborai.core.performance_manager.get_background_processor'), \
                 patch('harborai.core.performance_manager.get_optimized_plugin_manager'):
                
                manager = PerformanceManager()
                
                # 并发初始化
                tasks = [manager.initialize() for _ in range(3)]
                await asyncio.gather(*tasks)
                
                # 应该只初始化一次
                assert manager._initialized is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])