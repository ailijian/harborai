"""存储模块。"""

from .postgres_logger import (
    PostgreSQLLogger,
    get_postgres_logger,
    initialize_postgres_logger,
    shutdown_postgres_logger
)
from .lifecycle import (
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

__all__ = [
    # PostgreSQL日志记录
    "PostgreSQLLogger",
    "get_postgres_logger",
    "initialize_postgres_logger",
    "shutdown_postgres_logger",
    
    # 生命周期管理
    "LifecycleManager",
    "get_lifecycle_manager",
    "initialize_lifecycle",
    "add_startup_hook",
    "add_shutdown_hook",
    "shutdown",
    "on_startup",
    "on_shutdown",
    "LifecycleContext",
    "auto_initialize"
]