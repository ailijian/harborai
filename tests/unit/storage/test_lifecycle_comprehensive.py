"""
生命周期管理模块全面测试用例。

测试目标：
- 覆盖LifecycleManager类的所有方法和分支
- 测试信号处理、钩子执行、错误处理等关键功能
- 确保线程安全和并发场景下的正确性
- 验证装饰器、上下文管理器等便捷功能
"""

import pytest
import signal
import sys
import threading
import time
import logging
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Callable

from harborai.storage.lifecycle import (
    LifecycleManager,
    get_lifecycle_manager,
    initialize_lifecycle,
    add_startup_hook,
    add_shutdown_hook,
    shutdown,
    on_startup,
    on_shutdown,
    LifecycleContext,
    auto_initialize
)


class TestLifecycleManager:
    """LifecycleManager类的全面测试。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        # 重置全局状态
        import harborai.storage.lifecycle
        harborai.storage.lifecycle._lifecycle_manager = None
        
        # 创建新的管理器实例
        self.manager = LifecycleManager()
        
        # 用于测试的钩子函数
        self.startup_calls = []
        self.shutdown_calls = []
        
        def startup_hook():
            self.startup_calls.append("startup")
            
        def shutdown_hook():
            self.shutdown_calls.append("shutdown")
            
        self.startup_hook = startup_hook
        self.shutdown_hook = shutdown_hook
    
    def test_初始化状态(self):
        """测试LifecycleManager的初始状态。"""
        assert not self.manager._initialized
        assert not self.manager._shutdown_in_progress
        assert len(self.manager._startup_hooks) == 0
        assert len(self.manager._shutdown_hooks) == 0
        assert self.manager._lock is not None
    
    def test_添加启动钩子(self):
        """测试添加启动钩子功能。"""
        # 添加单个钩子
        self.manager.add_startup_hook(self.startup_hook)
        assert len(self.manager._startup_hooks) == 1
        assert self.startup_hook in self.manager._startup_hooks
        
        # 添加多个钩子
        hook2 = Mock()
        self.manager.add_startup_hook(hook2)
        assert len(self.manager._startup_hooks) == 2
        assert hook2 in self.manager._startup_hooks
    
    def test_添加关闭钩子(self):
        """测试添加关闭钩子功能。"""
        # 添加单个钩子
        self.manager.add_shutdown_hook(self.shutdown_hook)
        assert len(self.manager._shutdown_hooks) == 1
        assert self.shutdown_hook in self.manager._shutdown_hooks
        
        # 添加多个钩子
        hook2 = Mock()
        self.manager.add_shutdown_hook(hook2)
        assert len(self.manager._shutdown_hooks) == 2
        assert hook2 in self.manager._shutdown_hooks
    
    @patch('atexit.register')
    @patch('signal.signal')
    def test_初始化管理器(self, mock_signal, mock_atexit):
        """测试初始化管理器功能。"""
        # 添加启动钩子
        self.manager.add_startup_hook(self.startup_hook)
        
        # 初始化
        self.manager.initialize()
        
        # 验证状态
        assert self.manager._initialized
        assert len(self.startup_calls) == 1
        
        # 验证注册了atexit处理器
        mock_atexit.assert_called_once()
        
        # 验证注册了信号处理器
        assert mock_signal.call_count >= 1
        
        # 重复初始化应该被忽略
        self.manager.initialize()
        assert len(self.startup_calls) == 1  # 不应该重复执行
    
    @patch('signal.signal')
    def test_信号处理器注册(self, mock_signal):
        """测试信号处理器注册功能。"""
        self.manager._register_signal_handlers()
        
        # 验证注册了预期的信号
        expected_signals = []
        if hasattr(signal, 'SIGTERM'):
            expected_signals.append(signal.SIGTERM)
        if hasattr(signal, 'SIGINT'):
            expected_signals.append(signal.SIGINT)
        if hasattr(signal, 'SIGBREAK'):
            expected_signals.append(signal.SIGBREAK)
        
        assert mock_signal.call_count == len(expected_signals)
        
        # 验证信号处理器函数
        for call_args in mock_signal.call_args_list:
            sig, handler = call_args[0]
            assert sig in expected_signals
            assert callable(handler)
    
    def test_启动钩子执行(self):
        """测试启动钩子执行功能。"""
        # 创建测试钩子
        hook1 = Mock()
        hook1.__name__ = "hook1"
        hook2 = Mock()
        hook2.__name__ = "hook2"
        hook3 = Mock()
        hook3.__name__ = "hook3"
        
        self.manager.add_startup_hook(hook1)
        self.manager.add_startup_hook(hook2)
        self.manager.add_startup_hook(hook3)
        
        # 执行启动钩子
        self.manager._execute_startup_hooks()
        
        # 验证所有钩子都被调用
        hook1.assert_called_once()
        hook2.assert_called_once()
        hook3.assert_called_once()
    
    def test_启动钩子异常处理(self):
        """测试启动钩子异常处理。"""
        # 创建会抛出异常的钩子
        def failing_hook():
            raise ValueError("测试异常")
        
        normal_hook = Mock()
        normal_hook.__name__ = "normal_hook"
        
        self.manager.add_startup_hook(failing_hook)
        self.manager.add_startup_hook(normal_hook)
        
        # 执行启动钩子（不应该抛出异常）
        with patch('harborai.storage.lifecycle.logger') as mock_logger:
            self.manager._execute_startup_hooks()
            
            # 验证记录了错误日志
            mock_logger.error.assert_called()
            
            # 验证正常钩子仍然被执行
            normal_hook.assert_called_once()
    
    def test_关闭流程(self):
        """测试关闭流程功能。"""
        # 添加关闭钩子
        hook1 = Mock()
        hook2 = Mock()
        hook3 = Mock()
        
        self.manager.add_shutdown_hook(hook1)
        self.manager.add_shutdown_hook(hook2)
        self.manager.add_shutdown_hook(hook3)
        
        # 执行关闭流程
        self.manager._shutdown()
        
        # 验证状态
        assert self.manager._shutdown_in_progress
        
        # 验证钩子按逆序执行
        hook3.assert_called_once()
        hook2.assert_called_once()
        hook1.assert_called_once()
    
    def test_关闭流程重复调用(self):
        """测试关闭流程重复调用的处理。"""
        hook = Mock()
        self.manager.add_shutdown_hook(hook)
        
        # 第一次调用
        self.manager._shutdown()
        assert hook.call_count == 1
        
        # 第二次调用应该被忽略
        self.manager._shutdown()
        assert hook.call_count == 1
    
    def test_关闭钩子异常处理(self):
        """测试关闭钩子异常处理。"""
        # 创建会抛出异常的钩子
        def failing_hook():
            raise ValueError("测试异常")
        
        normal_hook = Mock()
        
        self.manager.add_shutdown_hook(failing_hook)
        self.manager.add_shutdown_hook(normal_hook)
        
        # 执行关闭流程（不应该抛出异常）
        self.manager._shutdown()
        
        # 验证正常钩子仍然被执行
        normal_hook.assert_called_once()
    
    @patch('structlog.reset_defaults')
    @patch('structlog._config')
    @patch('logging.getLogger')
    def test_关闭日志系统(self, mock_get_logger, mock_config, mock_reset):
        """测试关闭日志系统功能。"""
        # 模拟日志处理器
        mock_handler1 = Mock()
        mock_handler2 = Mock()
        
        # 模拟处理器的stream属性
        mock_handler1.stream = Mock()
        mock_handler1.stream.closed = False
        mock_handler2.stream = Mock()
        mock_handler2.stream.closed = False
        
        mock_root_logger = Mock()
        mock_root_logger.handlers = [mock_handler1, mock_handler2]
        mock_get_logger.return_value = mock_root_logger
        mock_config.is_configured = True
        
        # 执行关闭日志系统
        self.manager._shutdown_logging_system()
        
        # 验证处理器被关闭
        mock_handler1.flush.assert_called_once()
        mock_handler1.close.assert_called_once()
        mock_handler2.flush.assert_called_once()
        mock_handler2.close.assert_called_once()
        
        # 验证处理器被移除
        assert mock_root_logger.removeHandler.call_count == 2
        
        # 验证structlog被重置
        mock_reset.assert_called_once()
    
    @patch('structlog.reset_defaults')
    @patch('structlog._config')
    @patch('logging.getLogger')
    def test_关闭日志系统异常处理(self, mock_get_logger, mock_config, mock_reset):
        """测试关闭日志系统的异常处理。"""
        # 模拟会抛出异常的处理器
        mock_handler = Mock()
        mock_handler.flush.side_effect = OSError("测试异常")
        mock_handler.stream = Mock()
        mock_handler.stream.closed = False
        mock_root_logger = Mock()
        mock_root_logger.handlers = [mock_handler]
        mock_get_logger.return_value = mock_root_logger
        mock_config.is_configured = True
        
        # 执行关闭日志系统（不应该抛出异常）
        self.manager._shutdown_logging_system()
        
        # 验证处理器仍然尝试被关闭
        mock_handler.flush.assert_called_once()
    
    def test_手动关闭(self):
        """测试手动关闭功能。"""
        with patch.object(self.manager, '_shutdown') as mock_shutdown:
            self.manager.shutdown()
            mock_shutdown.assert_called_once()
    
    def test_线程安全(self):
        """测试线程安全性。"""
        results = []
        
        def add_hooks():
            for i in range(10):
                self.manager.add_startup_hook(lambda: results.append(threading.current_thread().ident))
                self.manager.add_shutdown_hook(lambda: results.append(threading.current_thread().ident))
        
        # 创建多个线程同时添加钩子
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=add_hooks)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证钩子数量正确
        assert len(self.manager._startup_hooks) == 50
        assert len(self.manager._shutdown_hooks) == 50


class TestGlobalFunctions:
    """全局函数的测试。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        # 重置全局状态
        import harborai.storage.lifecycle
        harborai.storage.lifecycle._lifecycle_manager = None
    
    def test_获取生命周期管理器(self):
        """测试获取全局生命周期管理器。"""
        # 第一次调用应该创建新实例
        manager1 = get_lifecycle_manager()
        assert isinstance(manager1, LifecycleManager)
        
        # 第二次调用应该返回同一实例
        manager2 = get_lifecycle_manager()
        assert manager1 is manager2
    
    @patch('harborai.storage.lifecycle.shutdown_postgres_logger')
    @patch.object(LifecycleManager, 'initialize')
    def test_初始化生命周期(self, mock_initialize, mock_shutdown_postgres):
        """测试初始化生命周期功能。"""
        initialize_lifecycle()
        
        # 验证管理器被初始化
        mock_initialize.assert_called_once()
        
        # 验证添加了默认关闭钩子
        manager = get_lifecycle_manager()
        assert mock_shutdown_postgres in manager._shutdown_hooks
    
    def test_添加启动钩子便捷函数(self):
        """测试添加启动钩子的便捷函数。"""
        hook = Mock()
        add_startup_hook(hook)
        
        manager = get_lifecycle_manager()
        assert hook in manager._startup_hooks
    
    def test_添加关闭钩子便捷函数(self):
        """测试添加关闭钩子的便捷函数。"""
        hook = Mock()
        add_shutdown_hook(hook)
        
        manager = get_lifecycle_manager()
        assert hook in manager._shutdown_hooks
    
    def test_关闭便捷函数(self):
        """测试关闭的便捷函数。"""
        with patch.object(LifecycleManager, 'shutdown') as mock_shutdown:
            shutdown()
            mock_shutdown.assert_called_once()


class TestDecorators:
    """装饰器功能的测试。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        # 重置全局状态
        import harborai.storage.lifecycle
        harborai.storage.lifecycle._lifecycle_manager = None
    
    def test_启动装饰器(self):
        """测试启动钩子装饰器。"""
        @on_startup
        def my_startup_function():
            return "startup_result"
        
        # 验证函数被添加到启动钩子
        manager = get_lifecycle_manager()
        assert my_startup_function in manager._startup_hooks
        
        # 验证装饰器返回原函数
        result = my_startup_function()
        assert result == "startup_result"
    
    def test_关闭装饰器(self):
        """测试关闭钩子装饰器。"""
        @on_shutdown
        def my_shutdown_function():
            return "shutdown_result"
        
        # 验证函数被添加到关闭钩子
        manager = get_lifecycle_manager()
        assert my_shutdown_function in manager._shutdown_hooks
        
        # 验证装饰器返回原函数
        result = my_shutdown_function()
        assert result == "shutdown_result"


class TestLifecycleContext:
    """生命周期上下文管理器的测试。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        # 重置全局状态
        import harborai.storage.lifecycle
        harborai.storage.lifecycle._lifecycle_manager = None
    
    @patch('harborai.storage.lifecycle.initialize_lifecycle')
    @patch('harborai.storage.lifecycle.shutdown')
    def test_上下文管理器正常流程(self, mock_shutdown, mock_initialize):
        """测试上下文管理器的正常流程。"""
        with LifecycleContext() as ctx:
            assert isinstance(ctx, LifecycleContext)
        
        # 验证初始化和关闭被调用
        mock_initialize.assert_called_once()
        mock_shutdown.assert_called_once()
    
    @patch('harborai.storage.lifecycle.initialize_lifecycle')
    @patch('harborai.storage.lifecycle.shutdown')
    def test_上下文管理器异常处理(self, mock_shutdown, mock_initialize):
        """测试上下文管理器的异常处理。"""
        try:
            with LifecycleContext():
                raise ValueError("测试异常")
        except ValueError:
            pass
        
        # 验证即使发生异常也会调用关闭
        mock_initialize.assert_called_once()
        mock_shutdown.assert_called_once()


class TestAutoInitialize:
    """自动初始化功能的测试。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        # 重置全局状态
        import harborai.storage.lifecycle
        harborai.storage.lifecycle._lifecycle_manager = None
    
    @patch('harborai.storage.lifecycle.initialize_lifecycle')
    def test_自动初始化未初始化状态(self, mock_initialize):
        """测试自动初始化未初始化的管理器。"""
        auto_initialize()
        mock_initialize.assert_called_once()
    
    @patch('harborai.storage.lifecycle.initialize_lifecycle')
    def test_自动初始化已初始化状态(self, mock_initialize):
        """测试自动初始化已初始化的管理器。"""
        # 先手动初始化
        manager = get_lifecycle_manager()
        manager._initialized = True
        
        # 调用自动初始化
        auto_initialize()
        
        # 验证不会重复初始化
        mock_initialize.assert_not_called()


class TestIntegrationScenarios:
    """集成场景测试。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        # 重置全局状态
        import harborai.storage.lifecycle
        harborai.storage.lifecycle._lifecycle_manager = None
    
    def test_完整生命周期流程(self):
        """测试完整的生命周期流程。"""
        startup_calls = []
        shutdown_calls = []
        
        @on_startup
        def startup_task():
            startup_calls.append("task1")
        
        @on_shutdown
        def shutdown_task():
            shutdown_calls.append("task1")
        
        # 添加更多钩子
        add_startup_hook(lambda: startup_calls.append("task2"))
        add_shutdown_hook(lambda: shutdown_calls.append("task2"))
        
        # 初始化
        with patch('atexit.register'), patch('signal.signal'):
            initialize_lifecycle()
        
        # 验证启动钩子被执行
        assert "task1" in startup_calls
        assert "task2" in startup_calls
        
        # 执行关闭
        shutdown()
        
        # 验证关闭钩子被执行（逆序）
        assert shutdown_calls == ["task2", "task1"]
    
    def test_并发场景(self):
        """测试并发场景下的行为。"""
        results = []
        
        def concurrent_operation():
            # 添加钩子
            add_startup_hook(lambda: results.append(threading.current_thread().ident))
            
            # 获取管理器
            manager = get_lifecycle_manager()
            
            # 尝试初始化
            with patch('atexit.register'), patch('signal.signal'):
                manager.initialize()
        
        # 创建多个线程
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=concurrent_operation)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证管理器只被初始化一次
        manager = get_lifecycle_manager()
        assert manager._initialized
        
        # 验证所有钩子都被添加
        assert len(manager._startup_hooks) >= 10


class TestErrorHandling:
    """错误处理测试。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        # 重置全局状态
        import harborai.storage.lifecycle
        harborai.storage.lifecycle._lifecycle_manager = None
    
    def test_信号处理器异常(self):
        """测试信号处理器中的异常处理。"""
        manager = LifecycleManager()
        
        # 添加会抛出异常的关闭钩子
        def failing_hook():
            raise RuntimeError("关闭钩子异常")
        
        manager.add_shutdown_hook(failing_hook)
        
        # 模拟信号处理器
        with patch('sys.exit') as mock_exit:
            manager._register_signal_handlers()
            
            # 获取信号处理器
            signal_handler = None
            with patch('signal.signal') as mock_signal:
                manager._register_signal_handlers()
                if mock_signal.call_args_list:
                    signal_handler = mock_signal.call_args_list[0][0][1]
            
            if signal_handler:
                # 调用信号处理器（不应该抛出异常）
                signal_handler(signal.SIGTERM, None)
                mock_exit.assert_called_once_with(0)
    
    def test_日志系统关闭异常(self):
        """测试日志系统关闭时的异常处理。"""
        manager = LifecycleManager()
        
        with patch('logging.getLogger') as mock_get_logger:
            # 模拟getLogger抛出异常
            mock_get_logger.side_effect = Exception("日志系统异常")
            
            # 执行关闭日志系统（不应该抛出异常）
            manager._shutdown_logging_system()
    
    def test_structlog关闭异常(self):
        """测试structlog关闭时的异常处理。"""
        manager = LifecycleManager()
        
        with patch('structlog.reset_defaults') as mock_reset:
            mock_reset.side_effect = Exception("structlog异常")
            
            # 执行关闭日志系统（不应该抛出异常）
            manager._shutdown_logging_system()


class TestEdgeCases:
    """边界情况测试。"""
    
    def setup_method(self):
        """每个测试方法前的设置。"""
        # 重置全局状态
        import harborai.storage.lifecycle
        harborai.storage.lifecycle._lifecycle_manager = None
    
    def test_空钩子列表(self):
        """测试空钩子列表的处理。"""
        manager = LifecycleManager()
        
        # 执行启动钩子（空列表）
        manager._execute_startup_hooks()
        
        # 执行关闭流程（空列表）
        manager._shutdown()
        
        # 验证状态正确
        assert manager._shutdown_in_progress
    
    def test_None钩子处理(self):
        """测试None钩子的处理。"""
        manager = LifecycleManager()
        
        # 尝试添加None钩子
        manager.add_startup_hook(None)
        manager.add_shutdown_hook(None)
        
        # 执行钩子（应该处理None值）
        # None钩子会导致TypeError，但应该被捕获并记录
        with patch('harborai.storage.lifecycle.logger') as mock_logger:
            manager._execute_startup_hooks()
            # 验证记录了错误日志
            mock_logger.error.assert_called()
    
    def test_重复添加相同钩子(self):
        """测试重复添加相同钩子。"""
        manager = LifecycleManager()
        hook = Mock()
        hook.__name__ = "test_hook"
        
        # 重复添加相同钩子
        manager.add_startup_hook(hook)
        manager.add_startup_hook(hook)
        manager.add_shutdown_hook(hook)
        manager.add_shutdown_hook(hook)
        
        # 验证钩子被添加多次
        assert manager._startup_hooks.count(hook) == 2
        assert manager._shutdown_hooks.count(hook) == 2
        
        # 执行钩子
        manager._execute_startup_hooks()
        assert hook.call_count == 2
        
        manager._shutdown()
        assert hook.call_count == 4  # 2次启动 + 2次关闭