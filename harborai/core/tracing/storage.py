#!/usr/bin/env python3
"""
追踪信息存储模块

提供高效的追踪数据持久化和查询功能，包括：
- 追踪数据的高效存储和检索
- 复杂查询和聚合分析
- 数据压缩和归档
- 性能优化和缓存
- 数据一致性保障

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import hashlib
import gzip
import pickle
from enum import Enum

from sqlalchemy import text, create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool

from .data_collector import TracingRecord


class StorageMode(Enum):
    """存储模式"""
    REAL_TIME = "real_time"      # 实时存储
    BATCH = "batch"              # 批量存储
    COMPRESSED = "compressed"    # 压缩存储
    ARCHIVED = "archived"        # 归档存储


class QueryOptimization(Enum):
    """查询优化策略"""
    INDEX_SCAN = "index_scan"
    FULL_SCAN = "full_scan"
    PARTITION_SCAN = "partition_scan"
    CACHED_RESULT = "cached_result"


@dataclass
class StorageConfig:
    """存储配置"""
    database_url: str
    batch_size: int = 1000
    compression_enabled: bool = True
    cache_size: int = 10000
    archive_after_days: int = 30
    cleanup_after_days: int = 90
    connection_pool_size: int = 20
    max_overflow: int = 30
    query_timeout: int = 30
    enable_partitioning: bool = True


@dataclass
class QueryFilter:
    """查询过滤器"""
    hb_trace_id: Optional[str] = None
    otel_trace_id: Optional[str] = None
    span_id: Optional[str] = None
    operation_name: Optional[str] = None
    service_name: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    status: Optional[str] = None
    start_time_from: Optional[datetime] = None
    start_time_to: Optional[datetime] = None
    duration_min: Optional[float] = None
    duration_max: Optional[float] = None
    has_errors: Optional[bool] = None
    cost_min: Optional[float] = None
    cost_max: Optional[float] = None
    tags: Optional[Dict[str, Any]] = None


@dataclass
class QueryOptions:
    """查询选项"""
    limit: int = 100
    offset: int = 0
    sort_by: str = "start_time"
    sort_order: str = "desc"
    include_logs: bool = True
    include_tags: bool = True
    optimization: QueryOptimization = QueryOptimization.INDEX_SCAN


@dataclass
class StorageMetrics:
    """存储指标"""
    total_records: int = 0
    storage_size_bytes: int = 0
    compression_ratio: float = 0.0
    query_performance_ms: float = 0.0
    cache_hit_rate: float = 0.0
    index_efficiency: float = 0.0
    partition_distribution: Dict[str, int] = None
    
    def __post_init__(self):
        if self.partition_distribution is None:
            self.partition_distribution = {}


class TracingInfoStorage:
    """
    追踪信息存储类
    
    提供高效的追踪数据持久化和查询功能，支持：
    1. 多种存储模式（实时、批量、压缩、归档）
    2. 智能查询优化和缓存
    3. 数据压缩和分区
    4. 性能监控和指标收集
    5. 数据一致性保障
    """
    
    def __init__(self, config: StorageConfig):
        """
        初始化追踪信息存储
        
        参数:
            config: 存储配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 数据库连接
        self.async_engine = None
        self.async_session_factory = None
        
        # 缓存
        self._query_cache: Dict[str, Tuple[datetime, Any]] = {}
        self._cache_ttl = timedelta(minutes=5)
        
        # 批量处理
        self._batch_buffer: List[TracingRecord] = []
        self._batch_lock = asyncio.Lock()
        
        # 性能指标
        self._metrics = StorageMetrics()
        self._last_metrics_update = datetime.now()
        
        # 后台任务
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # 初始化标志
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化存储系统"""
        if self._initialized:
            return
        
        try:
            # 创建数据库引擎
            self.async_engine = create_async_engine(
                self.config.database_url,
                poolclass=QueuePool,
                pool_size=self.config.connection_pool_size,
                max_overflow=self.config.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            
            # 创建会话工厂
            self.async_session_factory = async_sessionmaker(
                self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # 启动后台任务
            await self._start_background_tasks()
            
            self._initialized = True
            self.logger.info("追踪信息存储系统初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化追踪信息存储失败: {e}")
            raise
    
    async def _start_background_tasks(self) -> None:
        """启动后台任务"""
        # 批量处理任务
        batch_task = asyncio.create_task(self._batch_processor())
        self._background_tasks.append(batch_task)
        
        # 缓存清理任务
        cache_task = asyncio.create_task(self._cache_cleaner())
        self._background_tasks.append(cache_task)
        
        # 指标更新任务
        metrics_task = asyncio.create_task(self._metrics_updater())
        self._background_tasks.append(metrics_task)
        
        # 数据归档任务
        archive_task = asyncio.create_task(self._data_archiver())
        self._background_tasks.append(archive_task)
    
    async def store_record(
        self, 
        record: TracingRecord, 
        mode: StorageMode = StorageMode.BATCH
    ) -> bool:
        """
        存储追踪记录
        
        参数:
            record: 追踪记录
            mode: 存储模式
            
        返回:
            bool: 存储是否成功
        """
        try:
            if mode == StorageMode.REAL_TIME:
                return await self._store_real_time(record)
            elif mode == StorageMode.BATCH:
                return await self._store_batch(record)
            elif mode == StorageMode.COMPRESSED:
                return await self._store_compressed(record)
            elif mode == StorageMode.ARCHIVED:
                return await self._store_archived(record)
            else:
                self.logger.warning(f"未知的存储模式: {mode}")
                return False
                
        except Exception as e:
            self.logger.error(f"存储追踪记录失败: {e}")
            return False
    
    async def _store_real_time(self, record: TracingRecord) -> bool:
        """实时存储记录"""
        if not self.async_session_factory:
            return False
        
        async with self.async_session_factory() as session:
            try:
                insert_sql = text("""
                    INSERT INTO tracing_info (
                        hb_trace_id, otel_trace_id, span_id, parent_span_id,
                        operation_name, service_name, start_time, end_time, duration_ms,
                        provider, model, status, error_message,
                        prompt_tokens, completion_tokens, total_tokens, parsing_method, confidence,
                        input_cost, output_cost, total_cost, currency, pricing_source,
                        tags, logs, created_at
                    ) VALUES (
                        :hb_trace_id, :otel_trace_id, :span_id, :parent_span_id,
                        :operation_name, :service_name, :start_time, :end_time, :duration_ms,
                        :provider, :model, :status, :error_message,
                        :prompt_tokens, :completion_tokens, :total_tokens, :parsing_method, :confidence,
                        :input_cost, :output_cost, :total_cost, :currency, :pricing_source,
                        :tags, :logs, :created_at
                    )
                """)
                
                data = self._prepare_record_data(record)
                await session.execute(insert_sql, data)
                await session.commit()
                
                return True
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"实时存储失败: {e}")
                return False
    
    async def _store_batch(self, record: TracingRecord) -> bool:
        """批量存储记录"""
        async with self._batch_lock:
            self._batch_buffer.append(record)
            
            # 如果达到批量大小，立即处理
            if len(self._batch_buffer) >= self.config.batch_size:
                await self._flush_batch()
        
        return True
    
    async def _store_compressed(self, record: TracingRecord) -> bool:
        """压缩存储记录"""
        try:
            # 压缩记录数据
            record_data = asdict(record)
            compressed_data = gzip.compress(pickle.dumps(record_data))
            
            # 存储压缩数据
            if not self.async_session_factory:
                return False
            
            async with self.async_session_factory() as session:
                insert_sql = text("""
                    INSERT INTO tracing_info_compressed (
                        hb_trace_id, compressed_data, original_size, compressed_size, created_at
                    ) VALUES (
                        :hb_trace_id, :compressed_data, :original_size, :compressed_size, :created_at
                    )
                """)
                
                original_size = len(pickle.dumps(record_data))
                compressed_size = len(compressed_data)
                
                data = {
                    "hb_trace_id": record.hb_trace_id,
                    "compressed_data": compressed_data,
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                    "created_at": datetime.now()
                }
                
                await session.execute(insert_sql, data)
                await session.commit()
                
                # 更新压缩比指标
                compression_ratio = compressed_size / original_size if original_size > 0 else 0
                self._metrics.compression_ratio = (
                    self._metrics.compression_ratio * 0.9 + compression_ratio * 0.1
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"压缩存储失败: {e}")
            return False
    
    async def _store_archived(self, record: TracingRecord) -> bool:
        """归档存储记录"""
        try:
            # 归档到专门的归档表
            if not self.async_session_factory:
                return False
            
            async with self.async_session_factory() as session:
                insert_sql = text("""
                    INSERT INTO tracing_info_archive (
                        hb_trace_id, otel_trace_id, span_id, parent_span_id,
                        operation_name, service_name, start_time, end_time, duration_ms,
                        provider, model, status, error_message,
                        prompt_tokens, completion_tokens, total_tokens, parsing_method, confidence,
                        input_cost, output_cost, total_cost, currency, pricing_source,
                        tags, logs, created_at, archived_at
                    ) VALUES (
                        :hb_trace_id, :otel_trace_id, :span_id, :parent_span_id,
                        :operation_name, :service_name, :start_time, :end_time, :duration_ms,
                        :provider, :model, :status, :error_message,
                        :prompt_tokens, :completion_tokens, :total_tokens, :parsing_method, :confidence,
                        :input_cost, :output_cost, :total_cost, :currency, :pricing_source,
                        :tags, :logs, :created_at, :archived_at
                    )
                """)
                
                data = self._prepare_record_data(record)
                data["archived_at"] = datetime.now()
                
                await session.execute(insert_sql, data)
                await session.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"归档存储失败: {e}")
            return False
    
    def _prepare_record_data(self, record: TracingRecord) -> Dict[str, Any]:
        """准备记录数据"""
        return {
            "hb_trace_id": record.hb_trace_id,
            "otel_trace_id": record.otel_trace_id,
            "span_id": record.span_id,
            "parent_span_id": record.parent_span_id,
            "operation_name": record.operation_name,
            "service_name": record.service_name,
            "start_time": record.start_time,
            "end_time": record.end_time,
            "duration_ms": record.duration_ms,
            "provider": record.provider,
            "model": record.model,
            "status": record.status,
            "error_message": record.error_message,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "total_tokens": record.total_tokens,
            "parsing_method": record.parsing_method,
            "confidence": record.confidence,
            "input_cost": record.input_cost,
            "output_cost": record.output_cost,
            "total_cost": record.total_cost,
            "currency": record.currency,
            "pricing_source": record.pricing_source,
            "tags": json.dumps(record.tags) if record.tags else None,
            "logs": json.dumps(record.logs) if record.logs else None,
            "created_at": record.created_at
        }
    
    async def query_records(
        self,
        filter_obj: QueryFilter,
        options: QueryOptions = None
    ) -> List[Dict[str, Any]]:
        """
        查询追踪记录
        
        参数:
            filter_obj: 查询过滤器
            options: 查询选项
            
        返回:
            List[Dict]: 追踪记录列表
        """
        if options is None:
            options = QueryOptions()
        
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(filter_obj, options)
            cached_result = self._get_cached_result(cache_key)
            if cached_result is not None:
                self._metrics.cache_hit_rate = (
                    self._metrics.cache_hit_rate * 0.9 + 1.0 * 0.1
                )
                return cached_result
            
            # 执行查询
            start_time = datetime.now()
            result = await self._execute_query(filter_obj, options)
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # 更新性能指标
            self._metrics.query_performance_ms = (
                self._metrics.query_performance_ms * 0.9 + query_time * 0.1
            )
            self._metrics.cache_hit_rate = (
                self._metrics.cache_hit_rate * 0.9 + 0.0 * 0.1
            )
            
            # 缓存结果
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"查询追踪记录失败: {e}")
            return []
    
    async def _execute_query(
        self,
        filter_obj: QueryFilter,
        options: QueryOptions
    ) -> List[Dict[str, Any]]:
        """执行查询"""
        if not self.async_session_factory:
            return []
        
        async with self.async_session_factory() as session:
            # 构建查询条件
            conditions, params = self._build_query_conditions(filter_obj)
            
            # 构建排序
            sort_clause = self._build_sort_clause(options.sort_by, options.sort_order)
            
            # 构建查询SQL
            base_query = "SELECT * FROM tracing_info"
            if conditions:
                base_query += f" WHERE {' AND '.join(conditions)}"
            
            base_query += f" {sort_clause}"
            base_query += f" LIMIT {options.limit} OFFSET {options.offset}"
            
            query_sql = text(base_query)
            
            # 执行查询
            result = await session.execute(query_sql, params)
            rows = result.fetchall()
            
            # 转换结果
            records = []
            for row in rows:
                record_dict = dict(row._mapping)
                
                # 解析JSON字段
                if not options.include_tags:
                    record_dict.pop("tags", None)
                elif record_dict.get("tags"):
                    try:
                        record_dict["tags"] = json.loads(record_dict["tags"])
                    except:
                        record_dict["tags"] = {}
                
                if not options.include_logs:
                    record_dict.pop("logs", None)
                elif record_dict.get("logs"):
                    try:
                        record_dict["logs"] = json.loads(record_dict["logs"])
                    except:
                        record_dict["logs"] = []
                
                records.append(record_dict)
            
            return records
    
    def _build_query_conditions(self, filter_obj: QueryFilter) -> Tuple[List[str], Dict[str, Any]]:
        """构建查询条件"""
        conditions = []
        params = {}
        
        if filter_obj.hb_trace_id:
            conditions.append("hb_trace_id = :hb_trace_id")
            params["hb_trace_id"] = filter_obj.hb_trace_id
        
        if filter_obj.otel_trace_id:
            conditions.append("otel_trace_id = :otel_trace_id")
            params["otel_trace_id"] = filter_obj.otel_trace_id
        
        if filter_obj.span_id:
            conditions.append("span_id = :span_id")
            params["span_id"] = filter_obj.span_id
        
        if filter_obj.operation_name:
            conditions.append("operation_name = :operation_name")
            params["operation_name"] = filter_obj.operation_name
        
        if filter_obj.service_name:
            conditions.append("service_name = :service_name")
            params["service_name"] = filter_obj.service_name
        
        if filter_obj.provider:
            conditions.append("provider = :provider")
            params["provider"] = filter_obj.provider
        
        if filter_obj.model:
            conditions.append("model = :model")
            params["model"] = filter_obj.model
        
        if filter_obj.status:
            conditions.append("status = :status")
            params["status"] = filter_obj.status
        
        if filter_obj.start_time_from:
            conditions.append("start_time >= :start_time_from")
            params["start_time_from"] = filter_obj.start_time_from
        
        if filter_obj.start_time_to:
            conditions.append("start_time <= :start_time_to")
            params["start_time_to"] = filter_obj.start_time_to
        
        if filter_obj.duration_min is not None:
            conditions.append("duration_ms >= :duration_min")
            params["duration_min"] = filter_obj.duration_min
        
        if filter_obj.duration_max is not None:
            conditions.append("duration_ms <= :duration_max")
            params["duration_max"] = filter_obj.duration_max
        
        if filter_obj.has_errors is not None:
            if filter_obj.has_errors:
                conditions.append("error_message IS NOT NULL")
            else:
                conditions.append("error_message IS NULL")
        
        if filter_obj.cost_min is not None:
            conditions.append("total_cost >= :cost_min")
            params["cost_min"] = filter_obj.cost_min
        
        if filter_obj.cost_max is not None:
            conditions.append("total_cost <= :cost_max")
            params["cost_max"] = filter_obj.cost_max
        
        return conditions, params
    
    def _build_sort_clause(self, sort_by: str, sort_order: str) -> str:
        """构建排序子句"""
        valid_sort_fields = [
            "start_time", "end_time", "duration_ms", "total_cost",
            "total_tokens", "operation_name", "service_name", "provider"
        ]
        
        if sort_by not in valid_sort_fields:
            sort_by = "start_time"
        
        if sort_order.lower() not in ["asc", "desc"]:
            sort_order = "desc"
        
        return f"ORDER BY {sort_by} {sort_order.upper()}"
    
    def _generate_cache_key(self, filter_obj: QueryFilter, options: QueryOptions) -> str:
        """生成缓存键"""
        filter_dict = asdict(filter_obj)
        options_dict = asdict(options)
        
        # 移除None值
        filter_dict = {k: v for k, v in filter_dict.items() if v is not None}
        
        cache_data = {
            "filter": filter_dict,
            "options": options_dict
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """获取缓存结果"""
        if cache_key in self._query_cache:
            cached_time, result = self._query_cache[cache_key]
            if datetime.now() - cached_time < self._cache_ttl:
                return result
            else:
                del self._query_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: List[Dict[str, Any]]) -> None:
        """缓存结果"""
        # 限制缓存大小
        if len(self._query_cache) >= self.config.cache_size:
            # 删除最旧的缓存项
            oldest_key = min(self._query_cache.keys(), 
                           key=lambda k: self._query_cache[k][0])
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = (datetime.now(), result)
    
    async def _flush_batch(self) -> None:
        """刷新批量缓冲区"""
        if not self._batch_buffer or not self.async_session_factory:
            return
        
        records_to_process = self._batch_buffer.copy()
        self._batch_buffer.clear()
        
        async with self.async_session_factory() as session:
            try:
                # 准备批量插入数据
                insert_data = []
                for record in records_to_process:
                    data = self._prepare_record_data(record)
                    insert_data.append(data)
                
                # 执行批量插入
                insert_sql = text("""
                    INSERT INTO tracing_info (
                        hb_trace_id, otel_trace_id, span_id, parent_span_id,
                        operation_name, service_name, start_time, end_time, duration_ms,
                        provider, model, status, error_message,
                        prompt_tokens, completion_tokens, total_tokens, parsing_method, confidence,
                        input_cost, output_cost, total_cost, currency, pricing_source,
                        tags, logs, created_at
                    ) VALUES (
                        :hb_trace_id, :otel_trace_id, :span_id, :parent_span_id,
                        :operation_name, :service_name, :start_time, :end_time, :duration_ms,
                        :provider, :model, :status, :error_message,
                        :prompt_tokens, :completion_tokens, :total_tokens, :parsing_method, :confidence,
                        :input_cost, :output_cost, :total_cost, :currency, :pricing_source,
                        :tags, :logs, :created_at
                    )
                """)
                
                await session.execute(insert_sql, insert_data)
                await session.commit()
                
                self.logger.debug(f"批量存储了 {len(insert_data)} 条追踪记录")
                
            except Exception as e:
                await session.rollback()
                self.logger.error(f"批量存储失败: {e}")
                
                # 将失败的记录重新加入缓冲区
                async with self._batch_lock:
                    self._batch_buffer.extend(records_to_process)
    
    async def _batch_processor(self) -> None:
        """批量处理器后台任务"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5)  # 每5秒检查一次
                
                async with self._batch_lock:
                    if self._batch_buffer:
                        await self._flush_batch()
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"批量处理器错误: {e}")
    
    async def _cache_cleaner(self) -> None:
        """缓存清理器后台任务"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                
                current_time = datetime.now()
                expired_keys = []
                
                for cache_key, (cached_time, _) in self._query_cache.items():
                    if current_time - cached_time > self._cache_ttl:
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    del self._query_cache[key]
                
                if expired_keys:
                    self.logger.debug(f"清理了 {len(expired_keys)} 个过期缓存项")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"缓存清理器错误: {e}")
    
    async def _metrics_updater(self) -> None:
        """指标更新器后台任务"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(30)  # 每30秒更新一次
                await self._update_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"指标更新器错误: {e}")
    
    async def _update_metrics(self) -> None:
        """更新存储指标"""
        if not self.async_session_factory:
            return
        
        try:
            async with self.async_session_factory() as session:
                # 获取记录总数
                count_sql = text("SELECT COUNT(*) as total FROM tracing_info")
                result = await session.execute(count_sql)
                row = result.fetchone()
                self._metrics.total_records = row[0] if row else 0
                
                # 获取存储大小（近似）
                size_sql = text("""
                    SELECT pg_total_relation_size('tracing_info') as size_bytes
                """)
                try:
                    result = await session.execute(size_sql)
                    row = result.fetchone()
                    self._metrics.storage_size_bytes = row[0] if row else 0
                except:
                    # 如果不是PostgreSQL，使用估算
                    self._metrics.storage_size_bytes = self._metrics.total_records * 1024
                
                self._last_metrics_update = datetime.now()
                
        except Exception as e:
            self.logger.error(f"更新指标失败: {e}")
    
    async def _data_archiver(self) -> None:
        """数据归档器后台任务"""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3600)  # 每小时检查一次
                await self._archive_old_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"数据归档器错误: {e}")
    
    async def _archive_old_data(self) -> None:
        """归档旧数据"""
        if not self.async_session_factory:
            return
        
        try:
            archive_cutoff = datetime.now() - timedelta(days=self.config.archive_after_days)
            
            async with self.async_session_factory() as session:
                # 查找需要归档的数据
                select_sql = text("""
                    SELECT * FROM tracing_info 
                    WHERE start_time < :cutoff_time
                    LIMIT 1000
                """)
                
                result = await session.execute(select_sql, {"cutoff_time": archive_cutoff})
                rows = result.fetchall()
                
                if not rows:
                    return
                
                # 插入到归档表
                archive_sql = text("""
                    INSERT INTO tracing_info_archive (
                        hb_trace_id, otel_trace_id, span_id, parent_span_id,
                        operation_name, service_name, start_time, end_time, duration_ms,
                        provider, model, status, error_message,
                        prompt_tokens, completion_tokens, total_tokens, parsing_method, confidence,
                        input_cost, output_cost, total_cost, currency, pricing_source,
                        tags, logs, created_at, archived_at
                    ) VALUES (
                        :hb_trace_id, :otel_trace_id, :span_id, :parent_span_id,
                        :operation_name, :service_name, :start_time, :end_time, :duration_ms,
                        :provider, :model, :status, :error_message,
                        :prompt_tokens, :completion_tokens, :total_tokens, :parsing_method, :confidence,
                        :input_cost, :output_cost, :total_cost, :currency, :pricing_source,
                        :tags, :logs, :created_at, :archived_at
                    )
                """)
                
                archive_data = []
                ids_to_delete = []
                
                for row in rows:
                    row_dict = dict(row._mapping)
                    row_dict["archived_at"] = datetime.now()
                    archive_data.append(row_dict)
                    ids_to_delete.append(row_dict["id"])
                
                # 执行归档
                await session.execute(archive_sql, archive_data)
                
                # 删除原始数据
                delete_sql = text("DELETE FROM tracing_info WHERE id = ANY(:ids)")
                await session.execute(delete_sql, {"ids": ids_to_delete})
                
                await session.commit()
                
                self.logger.info(f"归档了 {len(archive_data)} 条追踪记录")
                
        except Exception as e:
            self.logger.error(f"数据归档失败: {e}")
    
    async def get_storage_metrics(self) -> StorageMetrics:
        """获取存储指标"""
        # 如果指标太旧，更新一次
        if datetime.now() - self._last_metrics_update > timedelta(minutes=5):
            await self._update_metrics()
        
        return self._metrics
    
    async def optimize_storage(self) -> Dict[str, Any]:
        """优化存储性能"""
        if not self.async_session_factory:
            return {"status": "error", "message": "数据库未初始化"}
        
        try:
            async with self.async_session_factory() as session:
                optimization_results = {}
                
                # 分析表统计信息
                analyze_sql = text("ANALYZE tracing_info")
                await session.execute(analyze_sql)
                optimization_results["analyze"] = "completed"
                
                # 重建索引（如果需要）
                reindex_sql = text("REINDEX TABLE tracing_info")
                try:
                    await session.execute(reindex_sql)
                    optimization_results["reindex"] = "completed"
                except:
                    optimization_results["reindex"] = "skipped"
                
                # 清理统计信息
                await session.commit()
                
                return {
                    "status": "success",
                    "optimizations": optimization_results,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"存储优化失败: {e}")
            return {"status": "error", "message": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "status": "healthy",
            "checks": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 检查数据库连接
            if self.async_session_factory:
                async with self.async_session_factory() as session:
                    test_sql = text("SELECT 1")
                    await session.execute(test_sql)
                    health_status["checks"]["database"] = "healthy"
            else:
                health_status["checks"]["database"] = "unhealthy"
                health_status["status"] = "unhealthy"
            
            # 检查批量缓冲区
            buffer_size = len(self._batch_buffer)
            if buffer_size < self.config.batch_size * 2:
                health_status["checks"]["batch_buffer"] = "healthy"
            else:
                health_status["checks"]["batch_buffer"] = "warning"
            
            health_status["checks"]["buffer_size"] = buffer_size
            
            # 检查缓存
            cache_size = len(self._query_cache)
            health_status["checks"]["cache_size"] = cache_size
            health_status["checks"]["cache"] = "healthy"
            
            # 检查后台任务
            active_tasks = sum(1 for task in self._background_tasks if not task.done())
            health_status["checks"]["background_tasks"] = active_tasks
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def shutdown(self) -> None:
        """关闭存储系统"""
        try:
            # 设置关闭事件
            self._shutdown_event.set()
            
            # 刷新剩余的批量数据
            await self._flush_batch()
            
            # 取消后台任务
            for task in self._background_tasks:
                task.cancel()
            
            # 等待任务完成
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # 关闭数据库连接
            if self.async_engine:
                await self.async_engine.dispose()
            
            self.logger.info("追踪信息存储系统已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭存储系统失败: {e}")


# 全局存储实例
_global_storage: Optional[TracingInfoStorage] = None


def get_global_storage() -> Optional[TracingInfoStorage]:
    """获取全局存储实例"""
    return _global_storage


async def setup_global_storage(config: StorageConfig) -> TracingInfoStorage:
    """设置全局存储实例"""
    global _global_storage
    _global_storage = TracingInfoStorage(config)
    await _global_storage.initialize()
    return _global_storage