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
            
            # 不使用日志系统，避免在测试环境中出现I/O错误
            
            # 执行关闭钩子（逆序执行）
            for hook in reversed(self._shutdown_hooks):
                try:
                    hook()
                except Exception:
                    # 忽略关闭钩子中的错误，避免日志系统问题
                    pass
            
            # 关闭标准库日志系统
            self._shutdown_logging_system()
    
    def _shutdown_logging_system(self):
        """优雅关闭日志系统。"""
        try:
            # 首先检查是否已经在关闭过程中
            import sys
            if hasattr(sys, '_getframe'):
                # 检查调用栈中是否已经有关闭操作
                frame = sys._getframe()
                while frame:
                    if 'shutdown' in frame.f_code.co_name.lower():
                        break
                    frame = frame.f_back
            
            # 关闭所有日志处理器
            root_logger = logging.getLogger()
            handlers_to_close = list(root_logger.handlers)
            
            for handler in handlers_to_close:
                try:
                    # 检查处理器是否已经关闭
                    if hasattr(handler, 'stream') and hasattr(handler.stream, 'closed'):
                        if handler.stream.closed:
                            continue
                    
                    # 安全关闭处理器
                    handler.flush()
                    handler.close()
                    root_logger.removeHandler(handler)
                except (ValueError, OSError, AttributeError) as e:
                    # 处理器已经关闭或无效，忽略错误
                    pass
                except Exception as e:
                    # 其他未预期的错误，记录到stderr但继续
                    try:
                        print(f"Warning: Error closing log handler: {e}", file=sys.stderr)
                    except:
                        pass
            
            # 关闭structlog相关的处理器
            try:
                import structlog
                # 检查structlog是否已经配置
                if hasattr(structlog, '_config') and structlog._config.is_configured:
                    # 安全地重置structlog配置
                    structlog.reset_defaults()
            except (ImportError, AttributeError, ValueError, OSError):
                # structlog未安装、未配置或已经关闭，忽略
                pass
            except Exception as e:
                # 其他structlog相关错误
                try:
                    print(f"Warning: Error resetting structlog: {e}", file=sys.stderr)
                except:
                    pass
                
        except Exception as e:
            # 如果关闭日志系统时出现任何错误，都忽略
            try:
                print(f"Warning: Error in shutdown logging system: {e}", file=sys.stderr)
            except:
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