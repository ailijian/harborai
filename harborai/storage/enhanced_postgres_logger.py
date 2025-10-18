"""
增强的 PostgreSQL 日志记录器
支持连接池、智能批处理和高级错误处理
"""

import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from decimal import Decimal
import traceback

from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Integer, Text, Numeric, Boolean
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool

from ..database.models import APILog, TokenUsageModel, TracingInfoModel
from ..utils.logger import get_logger
from ..utils.exceptions import StorageError
from .connection_pool import ConnectionPool, ConnectionPoolConfig
from .connection_pool_health_checker import ConnectionPoolHealthChecker, PerformanceThresholds, HealthStatus
from .batch_processor import AdaptiveBatchProcessor, BatchConfig, BatchItem, BatchPriority
from .error_handler import EnhancedErrorHandler, RetryConfig, ErrorCategory, ErrorSeverity

logger = get_logger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理datetime和Decimal对象"""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


class EnhancedPostgreSQLLogger:
    """增强的 PostgreSQL 日志记录器"""
    
    def __init__(self, 
                 connection_string: str,
                 batch_size: int = 100,
                 flush_interval: float = 5.0,
                 max_retries: int = 3,
                 pool_config: Optional[ConnectionPoolConfig] = None,
                 batch_config: Optional[BatchConfig] = None,
                 retry_config: Optional[RetryConfig] = None,
                 health_thresholds: Optional[PerformanceThresholds] = None,
                 enable_health_monitoring: bool = True):
        """初始化增强的 PostgreSQL 日志记录器
        
        Args:
            connection_string: 数据库连接字符串
            batch_size: 批处理大小
            flush_interval: 刷新间隔（秒）
            max_retries: 最大重试次数
            pool_config: 连接池配置
            batch_config: 批处理配置
            retry_config: 重试配置
        """
        self.connection_string = connection_string
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        
        # 连接池
        self.pool_config = pool_config or ConnectionPoolConfig()
        self.connection_pool: Optional[ConnectionPool] = None
        
        # 批处理器
        self.batch_config = batch_config or BatchConfig()
        self.batch_processor: Optional[AdaptiveBatchProcessor] = None
        
        # 健康监控
        self.health_thresholds = health_thresholds or PerformanceThresholds()
        self.enable_health_monitoring = enable_health_monitoring
        self.health_checker: Optional[ConnectionPoolHealthChecker] = None
        
        # 错误处理器
        self.retry_config = retry_config or RetryConfig()
        self.error_handler = EnhancedErrorHandler(
            retry_config=self.retry_config,
            fallback_handler=self._fallback_handler,
            alert_callback=self._alert_callback
        )
        
        # 状态管理
        self._running = False
        self._lock = threading.RLock()
        self._stats = {
            'total_logs': 0,
            'successful_logs': 0,
            'failed_logs': 0,
            'total_batches': 0,
            'successful_batches': 0,
            'failed_batches': 0,
            'connection_errors': 0,
            'retry_count': 0,
            'last_flush_time': None,
            'average_batch_size': 0.0,
            'average_flush_time': 0.0
        }
        
        # 数据库相关
        self.engine = None
        self.SessionLocal = None
        self.metadata = None
        
        # 表定义
        self.api_logs_table = None
        self.token_usage_table = None
        self.tracing_info_table = None
        
        logger.info("增强的 PostgreSQL 日志记录器已初始化")
    
    async def start(self):
        """启动日志记录器"""
        if self._running:
            logger.warning("日志记录器已在运行")
            return
        
        try:
            # 初始化数据库连接
            await self._initialize_database()
            
            # 初始化连接池
            await self._initialize_connection_pool()
            
            # 初始化批处理器
            await self._initialize_batch_processor()
            
            self._running = True
            logger.info("增强的 PostgreSQL 日志记录器已启动")
            
        except Exception as e:
            logger.error(f"启动日志记录器失败: {e}")
            await self.stop()
            raise StorageError(f"启动日志记录器失败: {e}") from e
    
    async def stop(self):
        """停止日志记录器"""
        if not self._running:
            return
        
        self._running = False
        
        try:
            # 停止健康检查器
            if self.health_checker:
                await self.health_checker.stop_monitoring()
                self.health_checker = None
                logger.info("健康检查器已停止")
            
            # 停止批处理器
            if self.batch_processor:
                await self.batch_processor.stop()
                self.batch_processor = None
            
            # 关闭连接池
            if self.connection_pool:
                await self.connection_pool.shutdown()
                self.connection_pool = None
            
            # 关闭数据库引擎
            if self.engine:
                self.engine.dispose()
                self.engine = None
            
            logger.info("增强的 PostgreSQL 日志记录器已停止")
            
        except Exception as e:
            logger.error(f"停止日志记录器失败: {e}")
    
    async def _initialize_database(self):
        """初始化数据库连接和表结构"""
        def _create_engine():
            return create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=self.pool_config.min_connections,
                max_overflow=self.pool_config.max_connections - self.pool_config.min_connections,
                pool_timeout=self.pool_config.connection_timeout,
                pool_recycle=self.pool_config.max_lifetime,
                pool_pre_ping=True,
                echo=False
            )
        
        # 创建引擎
        try:
            self.engine = await asyncio.get_event_loop().run_in_executor(
                None, 
                _create_engine
            )
        except Exception as e:
            # 使用错误处理器处理创建引擎失败
            self.engine = self.error_handler.handle_error(
                e,
                _create_engine,
                {'operation': 'create_engine', 'is_critical_operation': True}
            )
        
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.metadata = MetaData()
        
        # 定义表结构
        self._define_tables()
        
        # 创建表
        await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.metadata.create_all(self.engine)
        )
    
    def _define_tables(self):
        """定义数据库表结构"""
        # API 日志表
        self.api_logs_table = Table(
            'api_logs', self.metadata,
            Column('id', String, primary_key=True),
            Column('timestamp', DateTime, nullable=False),
            Column('method', String(10), nullable=False),
            Column('url', Text, nullable=False),
            Column('status_code', Integer),
            Column('response_time', Numeric(10, 3)),
            Column('request_size', Integer),
            Column('response_size', Integer),
            Column('user_id', String),
            Column('session_id', String),
            Column('trace_id', String),
            Column('span_id', String),
            Column('request_headers', Text),
            Column('response_headers', Text),
            Column('request_body', Text),
            Column('response_body', Text),
            Column('error_message', Text),
            Column('api_tags', Text),
            Column('internal_tags', Text),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # Token 使用表
        self.token_usage_table = Table(
            'token_usage', self.metadata,
            Column('id', String, primary_key=True),
            Column('api_log_id', String, nullable=False),
            Column('model_name', String),
            Column('prompt_tokens', Integer),
            Column('completion_tokens', Integer),
            Column('total_tokens', Integer),
            Column('estimated_cost', Numeric(10, 6)),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
        
        # 追踪信息表
        self.tracing_info_table = Table(
            'tracing_info', self.metadata,
            Column('id', String, primary_key=True),
            Column('api_log_id', String, nullable=False),
            Column('trace_id', String, nullable=False),
            Column('span_id', String, nullable=False),
            Column('parent_span_id', String),
            Column('operation_name', String),
            Column('start_time', DateTime),
            Column('end_time', DateTime),
            Column('duration', Numeric(10, 3)),
            Column('status', String),
            Column('tags', Text),
            Column('logs', Text),
            Column('created_at', DateTime, default=datetime.utcnow)
        )
    
    async def _initialize_connection_pool(self):
        """初始化连接池"""
        try:
            self.connection_pool = ConnectionPool(
                connection_string=self.connection_string,
                config=self.pool_config
            )
            await self.connection_pool.initialize()
            logger.info("连接池初始化成功")
            
            # 初始化健康检查器
            if self.enable_health_monitoring:
                self.health_checker = ConnectionPoolHealthChecker(
                    connection_pool=self.connection_pool,
                    thresholds=self.health_thresholds
                )
                # 注册健康状态回调
                self.health_checker.add_health_callback(self._on_health_status_change)
                await self.health_checker.start_monitoring()
                logger.info("连接池健康监控启动成功")
            
        except Exception as e:
            logger.error(f"连接池初始化失败: {e}")
            raise
    
    async def _initialize_batch_processor(self):
        """初始化批处理器"""
        self.batch_processor = AdaptiveBatchProcessor(
            config=self.batch_config,
            process_batch_func=self._process_batch
        )
        await self.batch_processor.start()
    
    async def log_api_call(self, 
                          api_log: APILog,
                          token_usage: Optional[TokenUsageModel] = None,
                          tracing_info: Optional[TracingInfoModel] = None,
                          priority: BatchPriority = BatchPriority.NORMAL):
        """记录 API 调用日志
        
        Args:
            api_log: API 日志模型
            token_usage: Token 使用模型
            tracing_info: 追踪信息模型
            priority: 批处理优先级
        """
        if not self._running:
            raise StorageError("日志记录器未运行")
        
        try:
            # 准备批处理项
            batch_item = BatchItem(
                data={
                    'api_log': api_log,
                    'token_usage': token_usage,
                    'tracing_info': tracing_info
                },
                priority=priority,
                timestamp=datetime.now()
            )
            
            # 添加到批处理器
            await self.batch_processor.add_item(batch_item)
            
            # 更新统计
            with self._lock:
                self._stats['total_logs'] += 1
            
        except Exception as e:
            with self._lock:
                self._stats['failed_logs'] += 1
            
            logger.error(f"记录 API 调用失败: {e}")
            raise StorageError(f"记录 API 调用失败: {e}") from e
    
    async def _process_batch(self, items: List[BatchItem]) -> bool:
        """处理批次数据
        
        Args:
            items: 批处理项列表
            
        Returns:
            bool: 处理是否成功
        """
        if not items:
            return True
        
        start_time = datetime.now()
        
        try:
            # 获取连接
            async with self.connection_pool.get_connection() as conn:
                # 开始事务
                trans = conn.begin()
                
                try:
                    # 批量插入数据
                    await self._batch_insert_data(conn, items)
                    
                    # 提交事务
                    trans.commit()
                    
                    # 更新统计
                    with self._lock:
                        self._stats['successful_logs'] += len(items)
                        self._stats['successful_batches'] += 1
                        self._stats['total_batches'] += 1
                        
                        # 更新平均批次大小
                        total_batches = self._stats['total_batches']
                        old_avg = self._stats['average_batch_size']
                        self._stats['average_batch_size'] = (
                            (old_avg * (total_batches - 1) + len(items)) / total_batches
                        )
                        
                        # 更新平均刷新时间
                        flush_time = (datetime.now() - start_time).total_seconds()
                        old_avg_time = self._stats['average_flush_time']
                        self._stats['average_flush_time'] = (
                            (old_avg_time * (total_batches - 1) + flush_time) / total_batches
                        )
                        
                        self._stats['last_flush_time'] = datetime.now()
                    
                    logger.debug(f"成功处理批次，包含 {len(items)} 条记录")
                    return True
                
                except Exception as e:
                    # 回滚事务
                    trans.rollback()
                    raise e
        
        except Exception as e:
            # 更新统计
            with self._lock:
                self._stats['failed_logs'] += len(items)
                self._stats['failed_batches'] += 1
                self._stats['total_batches'] += 1
            
            logger.error(f"处理批次失败: {e}")
            
            # 使用错误处理器处理错误
            return self.error_handler.handle_error(
                e,
                lambda: False,  # 返回失败
                {
                    'operation': 'batch_process',
                    'batch_size': len(items),
                    'is_critical_operation': False
                }
            )
    
    async def _batch_insert_data(self, conn, items: List[BatchItem]):
        """批量插入数据
        
        Args:
            conn: 数据库连接
            items: 批处理项列表
        """
        api_logs_data = []
        token_usage_data = []
        tracing_info_data = []
        
        # 准备数据
        for item in items:
            data = item.data
            
            # API 日志数据
            if 'api_log' in data and data['api_log']:
                api_log = data['api_log']
                api_logs_data.append({
                    'id': api_log.id,
                    'timestamp': api_log.timestamp,
                    'method': api_log.method,
                    'url': api_log.url,
                    'status_code': api_log.status_code,
                    'response_time': api_log.response_time,
                    'request_size': api_log.request_size,
                    'response_size': api_log.response_size,
                    'user_id': api_log.user_id,
                    'session_id': api_log.session_id,
                    'trace_id': api_log.trace_id,
                    'span_id': api_log.span_id,
                    'request_headers': json.dumps(api_log.request_headers) if api_log.request_headers else None,
                    'response_headers': json.dumps(api_log.response_headers) if api_log.response_headers else None,
                    'request_body': self._sanitize_data(api_log.request_body),
                    'response_body': self._sanitize_data(api_log.response_body),
                    'error_message': api_log.error_message,
                    'api_tags': json.dumps(api_log.api_tags) if api_log.api_tags else None,
                    'internal_tags': json.dumps(api_log.internal_tags) if api_log.internal_tags else None,
                    'created_at': datetime.utcnow()
                })
            
            # Token 使用数据
            if 'token_usage' in data and data['token_usage']:
                token_usage = data['token_usage']
                token_usage_data.append({
                    'id': token_usage.id,
                    'api_log_id': token_usage.api_log_id,
                    'model_name': token_usage.model_name,
                    'prompt_tokens': token_usage.prompt_tokens,
                    'completion_tokens': token_usage.completion_tokens,
                    'total_tokens': token_usage.total_tokens,
                    'estimated_cost': token_usage.estimated_cost,
                    'created_at': datetime.utcnow()
                })
            
            # 追踪信息数据
            if 'tracing_info' in data and data['tracing_info']:
                tracing_info = data['tracing_info']
                tracing_info_data.append({
                    'id': tracing_info.id,
                    'api_log_id': tracing_info.api_log_id,
                    'trace_id': tracing_info.trace_id,
                    'span_id': tracing_info.span_id,
                    'parent_span_id': tracing_info.parent_span_id,
                    'operation_name': tracing_info.operation_name,
                    'start_time': tracing_info.start_time,
                    'end_time': tracing_info.end_time,
                    'duration': tracing_info.duration,
                    'status': tracing_info.status,
                    'tags': json.dumps(tracing_info.tags) if tracing_info.tags else None,
                    'logs': json.dumps(tracing_info.logs) if tracing_info.logs else None,
                    'created_at': datetime.utcnow()
                })
        
        # 批量插入
        if api_logs_data:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: conn.execute(self.api_logs_table.insert(), api_logs_data)
            )
        
        if token_usage_data:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: conn.execute(self.token_usage_table.insert(), token_usage_data)
            )
        
        if tracing_info_data:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: conn.execute(self.tracing_info_table.insert(), tracing_info_data)
            )
    
    def _fallback_handler(self, error_info) -> bool:
        """降级处理函数
        
        Args:
            error_info: 错误信息
            
        Returns:
            bool: 处理结果
        """
        logger.warning(f"执行降级处理: {error_info.category.value}")
        
        # 根据错误类别执行不同的降级策略
        if error_info.category == ErrorCategory.CONNECTION:
            # 连接错误：尝试重新初始化连接池
            try:
                asyncio.create_task(self._reinitialize_connection_pool())
                return True
            except Exception as e:
                logger.error(f"重新初始化连接池失败: {e}")
                return False
        
        elif error_info.category == ErrorCategory.RESOURCE_EXHAUSTED:
            # 资源耗尽：减少批次大小
            if self.batch_processor:
                current_size = self.batch_processor.config.max_batch_size
                new_size = max(10, current_size // 2)
                self.batch_processor.config.max_batch_size = new_size
                logger.info(f"降低批次大小: {current_size} -> {new_size}")
                return True
        
        return False
    
    async def _reinitialize_connection_pool(self):
        """重新初始化连接池"""
        try:
            # 停止健康检查器
            if self.health_checker:
                await self.health_checker.stop_monitoring()
                self.health_checker = None
            
            # 关闭现有连接池
            if self.connection_pool:
                await self.connection_pool.shutdown()
            
            # 重新初始化连接池（包括健康检查器）
            await self._initialize_connection_pool()
            logger.info("连接池重新初始化成功")
            
        except Exception as e:
            logger.error(f"重新初始化连接池失败: {e}")
            raise
    
    def _alert_callback(self, error_info):
        """告警回调函数
        
        Args:
            error_info: 错误信息
        """
        logger.error(f"严重错误告警: {error_info.category.value} - {error_info.exception}")
        
        # 这里可以集成外部告警系统
        # 例如：发送邮件、Slack 通知、钉钉通知等
        
        # 更新连接错误统计
        if error_info.category == ErrorCategory.CONNECTION:
            with self._lock:
                self._stats['connection_errors'] += 1
    
    def _on_health_status_change(self, health_result):
        """健康状态变化回调
        
        Args:
            health_result: 健康检查结果
        """
        status = health_result.status
        logger.info(f"连接池健康状态变化: {status.value}")
        
        # 根据健康状态执行相应的处理
        if status == HealthStatus.CRITICAL:
            logger.error("连接池状态严重，触发紧急处理")
            # 触发连接池重新初始化
            asyncio.create_task(self._handle_critical_health())
            
        elif status == HealthStatus.WARNING:
            logger.warning("连接池状态警告，执行预防性措施")
            # 执行预防性措施，如减少批次大小
            self._apply_preventive_measures(health_result)
            
        elif status == HealthStatus.HEALTHY:
            logger.info("连接池状态恢复正常")
            # 恢复正常配置
            self._restore_normal_configuration()
    
    async def _handle_critical_health(self):
        """处理严重健康状态"""
        try:
            logger.info("开始处理严重健康状态")
            
            # 停止健康监控
            if self.health_checker:
                await self.health_checker.stop_monitoring()
            
            # 重新初始化连接池
            await self._reinitialize_connection_pool()
            
            logger.info("严重健康状态处理完成")
            
        except Exception as e:
            logger.error(f"处理严重健康状态失败: {e}")
    
    def _apply_preventive_measures(self, health_result):
        """应用预防性措施
        
        Args:
            health_result: 健康检查结果
        """
        try:
            # 根据健康检查结果调整配置
            recommendations = health_result.recommendations
            
            for recommendation in recommendations:
                if "减少批次大小" in recommendation:
                    if self.batch_processor:
                        current_size = self.batch_processor.config.max_batch_size
                        new_size = max(10, int(current_size * 0.8))
                        self.batch_processor.config.max_batch_size = new_size
                        logger.info(f"预防性措施：减少批次大小 {current_size} -> {new_size}")
                
                elif "增加连接超时" in recommendation:
                    if self.connection_pool:
                        # 增加连接超时时间
                        logger.info("预防性措施：增加连接超时时间")
                        
        except Exception as e:
            logger.error(f"应用预防性措施失败: {e}")
    
    def _restore_normal_configuration(self):
        """恢复正常配置"""
        try:
            # 恢复批次大小
            if self.batch_processor:
                original_size = self.batch_config.max_batch_size
                current_size = self.batch_processor.config.max_batch_size
                if current_size < original_size:
                    self.batch_processor.config.max_batch_size = original_size
                    logger.info(f"恢复批次大小: {current_size} -> {original_size}")
            
            logger.info("正常配置恢复完成")
            
        except Exception as e:
            logger.error(f"恢复正常配置失败: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取日志记录器统计信息"""
        with self._lock:
            stats = self._stats.copy()
        
        # 添加连接池统计
        if self.connection_pool:
            pool_stats = self.connection_pool.get_stats()
            stats['connection_pool'] = {
                'total_connections': pool_stats.total_connections,
                'active_connections': pool_stats.active_connections,
                'idle_connections': pool_stats.idle_connections,
                'failed_connections': pool_stats.failed_connections,
                'average_connection_time': pool_stats.average_connection_time,
                'total_requests': pool_stats.total_requests,
                'successful_requests': pool_stats.successful_requests,
                'failed_requests': pool_stats.failed_requests
            }
        
        # 添加批处理器统计
        if self.batch_processor:
            batch_stats = self.batch_processor.get_stats()
            stats['batch_processor'] = {
                'total_items': batch_stats.total_items,
                'processed_items': batch_stats.processed_items,
                'failed_items': batch_stats.failed_items,
                'total_batches': batch_stats.total_batches,
                'successful_batches': batch_stats.successful_batches,
                'failed_batches': batch_stats.failed_batches,
                'current_batch_size': batch_stats.current_batch_size,
                'average_batch_size': batch_stats.average_batch_size,
                'average_processing_time': batch_stats.average_processing_time,
                'items_per_second': batch_stats.items_per_second
            }
        
        # 添加错误处理器统计
        error_stats = self.error_handler.get_stats()
        stats['error_handler'] = {
            'total_errors': error_stats.total_errors,
            'errors_by_category': {k.value: v for k, v in error_stats.errors_by_category.items()},
            'errors_by_severity': {k.value: v for k, v in error_stats.errors_by_severity.items()},
            'total_retries': error_stats.total_retries,
            'successful_retries': error_stats.successful_retries,
            'failed_retries': error_stats.failed_retries,
            'average_retry_delay': error_stats.average_retry_delay,
            'error_rate': error_stats.error_rate
        }
        
        # 添加健康检查器统计
        if self.health_checker:
            health_stats = {
                'current_status': self.health_checker.get_current_status().value,
                'monitoring_enabled': self.health_checker._monitoring,
                'check_interval': self.health_checker.check_interval,
                'history_size': len(self.health_checker._health_history),
                'metrics_summary': self.health_checker.get_metrics_summary()
            }
            stats['health_checker'] = health_stats
        
        return stats
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            Dict: 健康状态信息
        """
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        try:
            # 使用健康检查器进行连接池检查
            if self.health_checker:
                pool_health_result = await self.health_checker.check_health()
                health_status['components']['connection_pool'] = {
                    'status': pool_health_result.status.value,
                    'healthy': pool_health_result.status == HealthStatus.HEALTHY,
                    'score': pool_health_result.score,
                    'details': pool_health_result.details,
                    'recommendations': pool_health_result.recommendations,
                    'timestamp': pool_health_result.timestamp.isoformat()
                }
                
                # 根据连接池状态更新整体状态
                if pool_health_result.status == HealthStatus.CRITICAL:
                    health_status['status'] = 'unhealthy'
                elif pool_health_result.status == HealthStatus.WARNING and health_status['status'] == 'healthy':
                    health_status['status'] = 'degraded'
            
            elif self.connection_pool:
                # 回退到基本的连接池检查
                pool_health = await self.connection_pool.health_check()
                health_status['components']['connection_pool'] = pool_health
                
                if not pool_health.get('healthy', False):
                    health_status['status'] = 'unhealthy'
            
            # 检查批处理器
            if self.batch_processor:
                batch_health = {
                    'healthy': self.batch_processor._running,
                    'queue_size': len(self.batch_processor._queue),
                    'current_batch_size': len(self.batch_processor._current_batch)
                }
                health_status['components']['batch_processor'] = batch_health
                
                if not batch_health['healthy']:
                    health_status['status'] = 'unhealthy'
            
            # 检查错误率
            error_summary = self.error_handler.get_error_summary(timedelta(minutes=5))
            if error_summary['error_rate'] > 10:  # 每分钟超过 10 个错误
                if health_status['status'] == 'healthy':
                    health_status['status'] = 'degraded'
                health_status['components']['error_rate'] = {
                    'healthy': False,
                    'error_rate': error_summary['error_rate'],
                    'threshold': 10
                }
            else:
                health_status['components']['error_rate'] = {
                    'healthy': True,
                    'error_rate': error_summary['error_rate'],
                    'threshold': 10
                }
        
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
            logger.error(f"健康检查失败: {e}")
        
        return health_status
    
    def _sanitize_data(self, data: Any) -> Any:
         """脱敏处理敏感数据
         
         Args:
             data: 要处理的数据
             
         Returns:
             脱敏后的数据
         """
         if data is None:
             return None
         
         if isinstance(data, dict):
             sanitized = {}
             sensitive_keys = {"api_key", "authorization", "token", "secret", "password", "key"}
             
             for key, value in data.items():
                 if any(sensitive in key.lower() for sensitive in sensitive_keys):
                     sanitized[key] = "[REDACTED]"
                 else:
                     sanitized[key] = self._sanitize_data(value)
             
             return sanitized
         
         elif isinstance(data, list):
             return [self._sanitize_data(item) for item in data]
         
         elif isinstance(data, str):
             # 对于字符串，可以进行更复杂的脱敏处理
             # 这里简化处理，只处理明显的敏感信息
             if len(data) > 1000:  # 限制长度
                 return data[:1000] + "...[TRUNCATED]"
             return data
         
         else:
             return data


# 全局实例管理
_enhanced_postgres_logger: Optional[EnhancedPostgreSQLLogger] = None
_logger_lock = threading.Lock()


def get_enhanced_postgres_logger() -> Optional[EnhancedPostgreSQLLogger]:
    """获取增强的 PostgreSQL 日志记录器实例"""
    return _enhanced_postgres_logger


def initialize_enhanced_postgres_logger(
    connection_string: str,
    pool_config: Optional[ConnectionPoolConfig] = None,
    batch_config: Optional[BatchConfig] = None,
    retry_config: Optional[RetryConfig] = None
) -> EnhancedPostgreSQLLogger:
    """初始化增强的 PostgreSQL 日志记录器
    
    Args:
        connection_string: 数据库连接字符串
        pool_config: 连接池配置
        batch_config: 批处理配置
        retry_config: 重试配置
        
    Returns:
        EnhancedPostgreSQLLogger: 日志记录器实例
    """
    global _enhanced_postgres_logger
    
    with _logger_lock:
        if _enhanced_postgres_logger is not None:
            logger.warning("增强的 PostgreSQL 日志记录器已存在，将关闭现有实例")
            asyncio.create_task(_enhanced_postgres_logger.stop())
        
        _enhanced_postgres_logger = EnhancedPostgreSQLLogger(
            connection_string=connection_string,
            pool_config=pool_config,
            batch_config=batch_config,
            retry_config=retry_config
        )
        
        return _enhanced_postgres_logger


async def shutdown_enhanced_postgres_logger():
    """关闭增强的 PostgreSQL 日志记录器"""
    global _enhanced_postgres_logger
    
    with _logger_lock:
        if _enhanced_postgres_logger is not None:
            await _enhanced_postgres_logger.stop()
            _enhanced_postgres_logger = None
            logger.info("增强的 PostgreSQL 日志记录器已关闭")