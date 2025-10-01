"""
存储模块初始化
"""

from .postgres_logger import PostgreSQLLogger, initialize_postgres_logger, shutdown_postgres_logger
from .file_logger import FileSystemLogger, initialize_file_logger, shutdown_file_logger
from .fallback_logger import FallbackLogger, LoggerState, initialize_fallback_logger, shutdown_fallback_logger, get_fallback_logger
from .lifecycle import LifecycleManager

__all__ = [
    'PostgreSQLLogger',
    'initialize_postgres_logger', 
    'shutdown_postgres_logger',
    'FileSystemLogger',
    'initialize_file_logger',
    'shutdown_file_logger',
    'FallbackLogger',
    'LoggerState',
    'initialize_fallback_logger',
    'shutdown_fallback_logger',
    'get_fallback_logger',
    'LifecycleManager'
]