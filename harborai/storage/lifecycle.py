"""生命周期管理模块。"""

import atexit
import signal
import sys
import logging
from typing import Callable, List, Optional
from threading import Lock

from ..utils.logger import get_logger
from .postgres_logger import get_postgres_logger, shutdown_postgres_logger

logger = get_logger(__name__)


class LifecycleManager:
    """应用生命周期管理器。"""
    
    def __init__(self):
        self._shutdown_hooks: List[Callable] = []
        self._startup_hooks: List[Callable] = []
        self._lock = Lock()
        self._initialized = False
        self._shutdown_in_progress = False
    
    def add_startup_hook(self, hook: Callable):
        """添加启动钩子。
        
        Args:
            hook: 启动时执行的函数
        """
        with self._lock:
            self._startup_hooks.append(hook)
    
    def add_shutdown_hook(self, hook: Callable):
        """添加关闭钩子。
        
        Args:
            hook: 关闭时执行的函数
        """
        with self._lock:
            self._shutdown_hooks.append(hook)
    
    def initialize(self):
        """初始化生命周期管理器。"""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            # 注册信号处理器
            self._register_signal_handlers()
            
            # 注册atexit处理器
            atexit.register(self._shutdown)
            
            # 执行启动钩子
            self._execute_startup_hooks()
            
            self._initialized = True
            logger.info("Lifecycle manager initialized")
    
    def _register_signal_handlers(self):
        """注册信号处理器。"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            self._shutdown()
            sys.exit(0)
        
        # 注册常见的终止信号
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        
        if hasattr(signal, 'SIGINT'):
            signal.signal(signal.SIGINT, signal_handler)
        
        # Windows特定信号
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)
    
    def _execute_startup_hooks(self):
        """执行启动钩子。"""
        for hook in self._startup_hooks:
            try:
                hook()
                logger.debug(f"Executed startup hook: {hook.__name__}")
            except Exception as e:
                logger.error(f"Error executing startup hook {hook.__name__}: {e}")
    
    def _shutdown(self):
        """执行关闭流程。"""
        if self._shutdown_in_progress:
            return
        
        with self._lock:
            if self._shutdown_in_progress:
                return
            
            self._shutdown_in_progress = True
            
            # 在关闭过程开始时记录日志，但要处理可能的异常
            try:
                logger.info("Starting shutdown process")
            except (ValueError, OSError):
                # 如果日志系统已经关闭，忽略错误
                pass
            
            # 执行关闭钩子（逆序执行）
            for hook in reversed(self._shutdown_hooks):
                try:
                    hook()
                    # 尝试记录调试信息，但如果失败则忽略
                    try:
                        logger.debug(f"Executed shutdown hook: {hook.__name__}")
                    except (ValueError, OSError):
                        pass
                except Exception as e:
                    # 尝试记录错误，但如果失败则忽略
                    try:
                        logger.error(f"Error executing shutdown hook {hook.__name__}: {e}")
                    except (ValueError, OSError):
                        # 如果日志系统已经关闭，使用标准输出
                        print(f"Error executing shutdown hook {hook.__name__}: {e}", file=sys.stderr)
            
            # 关闭标准库日志系统
            self._shutdown_logging_system()
            
            # 最后尝试记录完成信息
            try:
                logger.info("Shutdown process completed")
            except (ValueError, OSError):
                # 如果日志系统已经关闭，使用标准输出
                print("Shutdown process completed", file=sys.stderr)
    
    def _shutdown_logging_system(self):
        """优雅关闭日志系统。"""
        try:
            # 关闭所有日志处理器
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                try:
                    handler.close()
                    root_logger.removeHandler(handler)
                except Exception:
                    pass
            
            # 关闭structlog相关的处理器
            import structlog
            try:
                # 清理structlog的缓存
                structlog.reset_defaults()
            except Exception:
                pass
                
        except Exception:
            # 如果关闭日志系统时出现任何错误，都忽略
            pass
    
    def shutdown(self):
        """手动触发关闭流程。"""
        self._shutdown()


# 全局生命周期管理器实例
_lifecycle_manager: Optional[LifecycleManager] = None


def get_lifecycle_manager() -> LifecycleManager:
    """获取全局生命周期管理器。"""
    global _lifecycle_manager
    
    if _lifecycle_manager is None:
        _lifecycle_manager = LifecycleManager()
    
    return _lifecycle_manager


def initialize_lifecycle():
    """初始化生命周期管理。"""
    manager = get_lifecycle_manager()
    
    # 添加默认的关闭钩子
    manager.add_shutdown_hook(shutdown_postgres_logger)
    
    # 初始化管理器
    manager.initialize()


def add_startup_hook(hook: Callable):
    """添加启动钩子的便捷函数。"""
    get_lifecycle_manager().add_startup_hook(hook)


def add_shutdown_hook(hook: Callable):
    """添加关闭钩子的便捷函数。"""
    get_lifecycle_manager().add_shutdown_hook(hook)


def shutdown():
    """手动触发关闭流程的便捷函数。"""
    get_lifecycle_manager().shutdown()


# 装饰器支持
def on_startup(func: Callable) -> Callable:
    """启动钩子装饰器。
    
    Usage:
        @on_startup
        def my_startup_function():
            print("Application starting...")
    """
    add_startup_hook(func)
    return func


def on_shutdown(func: Callable) -> Callable:
    """关闭钩子装饰器。
    
    Usage:
        @on_shutdown
        def my_shutdown_function():
            print("Application shutting down...")
    """
    add_shutdown_hook(func)
    return func


# 上下文管理器支持
class LifecycleContext:
    """生命周期上下文管理器。
    
    Usage:
        with LifecycleContext():
            # 应用代码
            pass
    """
    
    def __enter__(self):
        initialize_lifecycle()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        shutdown()
        return False


# 自动初始化支持
def auto_initialize():
    """自动初始化生命周期管理（如果尚未初始化）。"""
    manager = get_lifecycle_manager()
    if not manager._initialized:
        initialize_lifecycle()