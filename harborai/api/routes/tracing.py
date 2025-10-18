#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
追踪API路由

提供分布式追踪功能，包括：
- 追踪链路查询
- 跨度(Span)管理
- 性能分析
- 追踪统计
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum

from ...database.manager import DatabaseManager
from ...utils.logger import get_logger
from ..schemas import StandardResponse, PaginatedResponse, ErrorResponse

logger = get_logger("harborai.api.tracing")

tracing_router = APIRouter(prefix="/v1/tracing", tags=["tracing"])


class SpanStatus(str, Enum):
    """跨度状态枚举"""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TracingCreateRequest(BaseModel):
    """创建追踪请求模型"""
    trace_id: str = Field(..., description="追踪ID")
    span_id: str = Field(..., description="跨度ID")
    parent_span_id: Optional[str] = Field(None, description="父跨度ID")
    operation_name: str = Field(..., description="操作名称")
    start_time: datetime = Field(..., description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    duration_ms: Optional[float] = Field(None, ge=0, description="持续时间(毫秒)")
    status: SpanStatus = Field(SpanStatus.OK, description="状态")
    tags: Optional[Dict[str, Any]] = Field(None, description="标签")
    logs: Optional[List[Dict[str, Any]]] = Field(None, description="日志事件")
    baggage: Optional[Dict[str, str]] = Field(None, description="行李数据")
    references: Optional[List[Dict[str, Any]]] = Field(None, description="引用关系")


class TracingUpdateRequest(BaseModel):
    """更新追踪请求模型"""
    end_time: Optional[datetime] = Field(None, description="结束时间")
    duration_ms: Optional[float] = Field(None, ge=0, description="持续时间")
    status: Optional[SpanStatus] = Field(None, description="状态")
    tags: Optional[Dict[str, Any]] = Field(None, description="标签")
    logs: Optional[List[Dict[str, Any]]] = Field(None, description="日志事件")
    baggage: Optional[Dict[str, str]] = Field(None, description="行李数据")


class SpanResponse(BaseModel):
    """跨度响应模型"""
    id: int
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[float]
    status: str
    tags: Optional[Dict[str, Any]]
    logs: Optional[List[Dict[str, Any]]]
    baggage: Optional[Dict[str, str]]
    references: Optional[List[Dict[str, Any]]]
    created_at: datetime
    updated_at: Optional[datetime]
    
    # 关联数据
    api_log: Optional[Dict[str, Any]] = None
    child_spans: Optional[List[Dict[str, Any]]] = None


class TraceResponse(BaseModel):
    """追踪链路响应模型"""
    trace_id: str
    root_span: SpanResponse
    spans: List[SpanResponse]
    total_duration_ms: float
    span_count: int
    error_count: int
    service_count: int
    services: List[str]
    operations: List[str]
    start_time: datetime
    end_time: datetime
    status: str
    
    # 性能指标
    critical_path: List[str]
    bottlenecks: List[Dict[str, Any]]
    error_spans: List[SpanResponse]


class TracingStatistics(BaseModel):
    """追踪统计信息"""
    total_traces: int
    total_spans: int
    average_trace_duration: float
    average_span_duration: float
    error_rate: float
    most_common_operations: List[Dict[str, Any]]
    slowest_operations: List[Dict[str, Any]]
    service_distribution: Dict[str, int]
    time_range: Dict[str, datetime]


async def get_db_manager() -> DatabaseManager:
    """获取数据库管理器依赖"""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    return db_manager


@tracing_router.post("/spans", response_model=StandardResponse[SpanResponse])
async def create_span(
    request: TracingCreateRequest,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    创建新的跨度记录
    
    创建一个新的分布式追踪跨度记录。
    """
    try:
        # 计算持续时间
        duration_ms = request.duration_ms
        if duration_ms is None and request.end_time:
            duration_ms = (request.end_time - request.start_time).total_seconds() * 1000
        
        # 插入跨度记录
        span_data = {
            'trace_id': request.trace_id,
            'span_id': request.span_id,
            'parent_span_id': request.parent_span_id,
            'operation_name': request.operation_name,
            'start_time': request.start_time,
            'end_time': request.end_time,
            'duration_ms': duration_ms,
            'status': request.status.value,
            'tags': request.tags,
            'logs': request.logs,
            'baggage': request.baggage,
            'references': request.references,
            'created_at': datetime.now()
        }
        
        result = await db_manager.execute_query("""
            INSERT INTO tracing_spans (
                trace_id, span_id, parent_span_id, operation_name, start_time,
                end_time, duration_ms, status, tags, logs, baggage, references, created_at
            ) VALUES (
                %(trace_id)s, %(span_id)s, %(parent_span_id)s, %(operation_name)s,
                %(start_time)s, %(end_time)s, %(duration_ms)s, %(status)s,
                %(tags)s, %(logs)s, %(baggage)s, %(references)s, %(created_at)s
            ) RETURNING id
        """, span_data)
        
        span_id_db = result[0]['id']
        
        # 获取完整的跨度记录
        span_record = await _get_span_by_id(db_manager, span_id_db)
        
        return StandardResponse(
            success=True,
            data=span_record,
            message="跨度创建成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"创建跨度失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建跨度失败: {str(e)}")


@tracing_router.get("/spans/{span_db_id}", response_model=StandardResponse[SpanResponse])
async def get_span(
    span_db_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    根据数据库ID获取跨度记录
    
    获取指定数据库ID的跨度记录。
    """
    try:
        span_record = await _get_span_by_id(db_manager, span_db_id)
        
        if not span_record:
            raise HTTPException(status_code=404, detail="跨度记录不存在")
        
        return StandardResponse(
            success=True,
            data=span_record,
            message="获取跨度成功",
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取跨度失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取跨度失败: {str(e)}")


@tracing_router.put("/spans/{span_db_id}", response_model=StandardResponse[SpanResponse])
async def update_span(
    span_db_id: int,
    request: TracingUpdateRequest,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    更新跨度记录
    
    更新指定数据库ID的跨度记录。
    """
    try:
        # 检查跨度是否存在
        existing_span = await _get_span_by_id(db_manager, span_db_id)
        if not existing_span:
            raise HTTPException(status_code=404, detail="跨度记录不存在")
        
        # 构建更新数据
        update_data = {'id': span_db_id, 'updated_at': datetime.now()}
        update_fields = []
        
        if request.end_time is not None:
            update_data['end_time'] = request.end_time
            update_fields.append('end_time = %(end_time)s')
        
        if request.duration_ms is not None:
            update_data['duration_ms'] = request.duration_ms
            update_fields.append('duration_ms = %(duration_ms)s')
        elif request.end_time is not None:
            # 重新计算持续时间
            start_time = existing_span.start_time
            duration_ms = (request.end_time - start_time).total_seconds() * 1000
            update_data['duration_ms'] = duration_ms
            update_fields.append('duration_ms = %(duration_ms)s')
        
        if request.status is not None:
            update_data['status'] = request.status.value
            update_fields.append('status = %(status)s')
        
        if request.tags is not None:
            update_data['tags'] = request.tags
            update_fields.append('tags = %(tags)s')
        
        if request.logs is not None:
            update_data['logs'] = request.logs
            update_fields.append('logs = %(logs)s')
        
        if request.baggage is not None:
            update_data['baggage'] = request.baggage
            update_fields.append('baggage = %(baggage)s')
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="没有提供更新字段")
        
        # 更新跨度记录
        update_fields.append('updated_at = %(updated_at)s')
        update_query = f"""
            UPDATE tracing_spans 
            SET {', '.join(update_fields)}
            WHERE id = %(id)s
        """
        
        await db_manager.execute_query(update_query, update_data)
        
        # 获取更新后的跨度记录
        updated_span = await _get_span_by_id(db_manager, span_db_id)
        
        return StandardResponse(
            success=True,
            data=updated_span,
            message="跨度更新成功",
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新跨度失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新跨度失败: {str(e)}")


@tracing_router.get("/traces/{trace_id}", response_model=StandardResponse[TraceResponse])
async def get_trace(
    trace_id: str,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    根据追踪ID获取完整的追踪链路
    
    获取指定追踪ID的完整链路信息，包括所有跨度和性能分析。
    """
    try:
        # 获取所有跨度
        spans_query = """
            SELECT * FROM tracing_spans 
            WHERE trace_id = %s 
            ORDER BY start_time ASC
        """
        
        spans_result = await db_manager.execute_query(spans_query, (trace_id,))
        
        if not spans_result:
            raise HTTPException(status_code=404, detail="追踪记录不存在")
        
        # 转换跨度数据
        spans = []
        root_span = None
        
        for row in spans_result:
            span = SpanResponse(
                id=row['id'],
                trace_id=row['trace_id'],
                span_id=row['span_id'],
                parent_span_id=row['parent_span_id'],
                operation_name=row['operation_name'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                duration_ms=row['duration_ms'],
                status=row['status'],
                tags=row['tags'],
                logs=row['logs'],
                baggage=row['baggage'],
                references=row['references'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            
            spans.append(span)
            
            # 找到根跨度（没有父跨度的跨度）
            if not row['parent_span_id']:
                root_span = span
        
        if not root_span:
            # 如果没有明确的根跨度，使用最早开始的跨度
            root_span = min(spans, key=lambda s: s.start_time)
        
        # 计算追踪统计信息
        total_duration = 0
        error_count = 0
        services = set()
        operations = set()
        error_spans = []
        
        start_time = min(span.start_time for span in spans)
        end_time = max(span.end_time for span in spans if span.end_time)
        
        if end_time:
            total_duration = (end_time - start_time).total_seconds() * 1000
        
        for span in spans:
            if span.status != 'ok':
                error_count += 1
                error_spans.append(span)
            
            # 从标签中提取服务信息
            if span.tags and 'service' in span.tags:
                services.add(span.tags['service'])
            
            operations.add(span.operation_name)
        
        # 分析关键路径和瓶颈
        critical_path = _analyze_critical_path(spans)
        bottlenecks = _analyze_bottlenecks(spans)
        
        # 确定整体状态
        overall_status = 'error' if error_count > 0 else 'ok'
        
        trace_response = TraceResponse(
            trace_id=trace_id,
            root_span=root_span,
            spans=spans,
            total_duration_ms=total_duration,
            span_count=len(spans),
            error_count=error_count,
            service_count=len(services),
            services=list(services),
            operations=list(operations),
            start_time=start_time,
            end_time=end_time or start_time,
            status=overall_status,
            critical_path=critical_path,
            bottlenecks=bottlenecks,
            error_spans=error_spans
        )
        
        return StandardResponse(
            success=True,
            data=trace_response,
            message="获取追踪链路成功",
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取追踪链路失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取追踪链路失败: {str(e)}")


@tracing_router.get("/traces", response_model=PaginatedResponse[Dict[str, Any]])
async def query_traces(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    operation_name: Optional[str] = Query(None, description="操作名称"),
    service: Optional[str] = Query(None, description="服务名称"),
    status: Optional[str] = Query(None, description="状态"),
    min_duration: Optional[float] = Query(None, ge=0, description="最小持续时间(毫秒)"),
    max_duration: Optional[float] = Query(None, ge=0, description="最大持续时间(毫秒)"),
    has_errors: Optional[bool] = Query(None, description="是否包含错误"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(50, ge=1, le=1000, description="每页大小"),
    sort_by: str = Query("start_time", description="排序字段"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="排序顺序"),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    查询追踪记录
    
    根据各种条件查询追踪记录，支持分页和排序。
    """
    try:
        # 构建查询条件
        where_conditions = []
        query_params = {}
        
        if start_time:
            where_conditions.append("start_time >= %(start_time)s")
            query_params['start_time'] = start_time
        
        if end_time:
            where_conditions.append("start_time <= %(end_time)s")
            query_params['end_time'] = end_time
        
        if operation_name:
            where_conditions.append("operation_name = %(operation_name)s")
            query_params['operation_name'] = operation_name
        
        if service:
            where_conditions.append("tags->>'service' = %(service)s")
            query_params['service'] = service
        
        if status:
            where_conditions.append("status = %(status)s")
            query_params['status'] = status
        
        if min_duration is not None:
            where_conditions.append("duration_ms >= %(min_duration)s")
            query_params['min_duration'] = min_duration
        
        if max_duration is not None:
            where_conditions.append("duration_ms <= %(max_duration)s")
            query_params['max_duration'] = max_duration
        
        if has_errors is not None:
            if has_errors:
                where_conditions.append("status != 'ok'")
            else:
                where_conditions.append("status = 'ok'")
        
        # 构建WHERE子句
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 获取唯一的追踪ID列表
        traces_query = f"""
            SELECT DISTINCT trace_id,
                   MIN(start_time) as trace_start_time,
                   MAX(COALESCE(end_time, start_time)) as trace_end_time,
                   COUNT(*) as span_count,
                   SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) as error_count,
                   AVG(duration_ms) as avg_duration
            FROM tracing_spans
            {where_clause}
            GROUP BY trace_id
            ORDER BY trace_start_time {sort_order.upper()}
            LIMIT %(limit)s OFFSET %(offset)s
        """
        
        # 计算分页
        offset = (page - 1) * page_size
        query_params.update({
            'limit': page_size,
            'offset': offset
        })
        
        traces_result = await db_manager.execute_query(traces_query, query_params)
        
        # 计算总数
        count_query = f"""
            SELECT COUNT(DISTINCT trace_id) as total
            FROM tracing_spans
            {where_clause}
        """
        
        count_result = await db_manager.execute_query(count_query, {k: v for k, v in query_params.items() if k not in ['limit', 'offset']})
        total_count = count_result[0]['total']
        
        # 构建响应数据
        traces = []
        for row in traces_result:
            trace_summary = {
                'trace_id': row['trace_id'],
                'start_time': row['trace_start_time'],
                'end_time': row['trace_end_time'],
                'duration_ms': (row['trace_end_time'] - row['trace_start_time']).total_seconds() * 1000,
                'span_count': row['span_count'],
                'error_count': row['error_count'],
                'avg_span_duration': row['avg_duration'],
                'status': 'error' if row['error_count'] > 0 else 'ok'
            }
            traces.append(trace_summary)
        
        # 计算分页信息
        total_pages = (total_count + page_size - 1) // page_size
        
        return PaginatedResponse(
            success=True,
            data=traces,
            pagination={
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            },
            message="查询追踪记录成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"查询追踪记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询追踪记录失败: {str(e)}")


@tracing_router.get("/statistics", response_model=StandardResponse[TracingStatistics])
async def get_tracing_statistics(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    service: Optional[str] = Query(None, description="服务名称"),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    获取追踪统计信息
    
    获取指定时间范围和条件下的追踪统计信息。
    """
    try:
        # 构建查询条件
        where_conditions = []
        query_params = {}
        
        if start_time:
            where_conditions.append("start_time >= %(start_time)s")
            query_params['start_time'] = start_time
        
        if end_time:
            where_conditions.append("start_time <= %(end_time)s")
            query_params['end_time'] = end_time
        
        if service:
            where_conditions.append("tags->>'service' = %(service)s")
            query_params['service'] = service
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 获取基础统计信息
        basic_stats_query = f"""
            SELECT 
                COUNT(DISTINCT trace_id) as total_traces,
                COUNT(*) as total_spans,
                AVG(duration_ms) as avg_span_duration,
                COUNT(CASE WHEN status != 'ok' THEN 1 END) as error_spans,
                MIN(start_time) as earliest_span,
                MAX(start_time) as latest_span
            FROM tracing_spans
            {where_clause}
        """
        
        basic_result = await db_manager.execute_query(basic_stats_query, query_params)
        basic_stats = basic_result[0]
        
        # 获取追踪级别的平均持续时间
        trace_duration_query = f"""
            SELECT AVG(trace_duration) as avg_trace_duration
            FROM (
                SELECT trace_id,
                       EXTRACT(EPOCH FROM (MAX(COALESCE(end_time, start_time)) - MIN(start_time))) * 1000 as trace_duration
                FROM tracing_spans
                {where_clause}
                GROUP BY trace_id
            ) trace_durations
        """
        
        trace_duration_result = await db_manager.execute_query(trace_duration_query, query_params)
        avg_trace_duration = trace_duration_result[0]['avg_trace_duration'] or 0
        
        # 获取最常见的操作
        common_ops_query = f"""
            SELECT operation_name, COUNT(*) as count
            FROM tracing_spans
            {where_clause}
            GROUP BY operation_name
            ORDER BY count DESC
            LIMIT 10
        """
        
        common_ops_result = await db_manager.execute_query(common_ops_query, query_params)
        most_common_operations = [
            {'operation': row['operation_name'], 'count': row['count']}
            for row in common_ops_result
        ]
        
        # 获取最慢的操作
        slow_ops_query = f"""
            SELECT operation_name, AVG(duration_ms) as avg_duration, COUNT(*) as count
            FROM tracing_spans
            {where_clause}
            AND duration_ms IS NOT NULL
            GROUP BY operation_name
            HAVING COUNT(*) >= 5
            ORDER BY avg_duration DESC
            LIMIT 10
        """
        
        slow_ops_result = await db_manager.execute_query(slow_ops_query, query_params)
        slowest_operations = [
            {
                'operation': row['operation_name'],
                'avg_duration': row['avg_duration'],
                'count': row['count']
            }
            for row in slow_ops_result
        ]
        
        # 获取服务分布
        service_dist_query = f"""
            SELECT tags->>'service' as service, COUNT(*) as count
            FROM tracing_spans
            {where_clause}
            AND tags->>'service' IS NOT NULL
            GROUP BY tags->>'service'
            ORDER BY count DESC
        """
        
        service_dist_result = await db_manager.execute_query(service_dist_query, query_params)
        service_distribution = {
            row['service']: row['count']
            for row in service_dist_result
        }
        
        # 计算错误率
        total_spans = basic_stats['total_spans'] or 0
        error_spans = basic_stats['error_spans'] or 0
        error_rate = error_spans / total_spans if total_spans > 0 else 0
        
        statistics = TracingStatistics(
            total_traces=basic_stats['total_traces'] or 0,
            total_spans=total_spans,
            average_trace_duration=avg_trace_duration,
            average_span_duration=basic_stats['avg_span_duration'] or 0,
            error_rate=error_rate,
            most_common_operations=most_common_operations,
            slowest_operations=slowest_operations,
            service_distribution=service_distribution,
            time_range={
                'start': basic_stats['earliest_span'],
                'end': basic_stats['latest_span']
            }
        )
        
        return StandardResponse(
            success=True,
            data=statistics,
            message="获取追踪统计信息成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取追踪统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取追踪统计信息失败: {str(e)}")


@tracing_router.get("/traces/{trace_id}/performance", response_model=StandardResponse[Dict[str, Any]])
async def analyze_trace_performance(
    trace_id: str,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    分析追踪性能
    
    对指定追踪ID进行详细的性能分析。
    """
    try:
        # 获取追踪的所有跨度
        spans_query = """
            SELECT * FROM tracing_spans 
            WHERE trace_id = %s 
            ORDER BY start_time ASC
        """
        
        spans_result = await db_manager.execute_query(spans_query, (trace_id,))
        
        if not spans_result:
            raise HTTPException(status_code=404, detail="追踪记录不存在")
        
        # 转换为SpanResponse对象
        spans = []
        for row in spans_result:
            span = SpanResponse(
                id=row['id'],
                trace_id=row['trace_id'],
                span_id=row['span_id'],
                parent_span_id=row['parent_span_id'],
                operation_name=row['operation_name'],
                start_time=row['start_time'],
                end_time=row['end_time'],
                duration_ms=row['duration_ms'],
                status=row['status'],
                tags=row['tags'],
                logs=row['logs'],
                baggage=row['baggage'],
                references=row['references'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            spans.append(span)
        
        # 性能分析
        analysis = {
            'trace_id': trace_id,
            'total_spans': len(spans),
            'critical_path': _analyze_critical_path(spans),
            'bottlenecks': _analyze_bottlenecks(spans),
            'parallel_execution': _analyze_parallel_execution(spans),
            'resource_utilization': _analyze_resource_utilization(spans),
            'error_analysis': _analyze_errors(spans),
            'recommendations': _generate_performance_recommendations(spans)
        }
        
        return StandardResponse(
            success=True,
            data=analysis,
            message="性能分析完成",
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"性能分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"性能分析失败: {str(e)}")


async def _get_span_by_id(db_manager: DatabaseManager, span_id: int) -> Optional[SpanResponse]:
    """根据数据库ID获取跨度记录"""
    query = """
        SELECT ts.*, al.model, al.provider, al.total_cost
        FROM tracing_spans ts
        LEFT JOIN tracing_info ti ON ts.trace_id = ti.trace_id AND ts.span_id = ti.span_id
        LEFT JOIN api_logs al ON ti.log_id = al.id
        WHERE ts.id = %s
    """
    
    result = await db_manager.execute_query(query, (span_id,))
    
    if not result:
        return None
    
    row = result[0]
    span_data = SpanResponse(
        id=row['id'],
        trace_id=row['trace_id'],
        span_id=row['span_id'],
        parent_span_id=row['parent_span_id'],
        operation_name=row['operation_name'],
        start_time=row['start_time'],
        end_time=row['end_time'],
        duration_ms=row['duration_ms'],
        status=row['status'],
        tags=row['tags'],
        logs=row['logs'],
        baggage=row['baggage'],
        references=row['references'],
        created_at=row['created_at'],
        updated_at=row['updated_at']
    )
    
    # 添加关联的API日志信息
    if row['model']:
        span_data.api_log = {
            'model': row['model'],
            'provider': row['provider'],
            'total_cost': row['total_cost']
        }
    
    return span_data


def _analyze_critical_path(spans: List[SpanResponse]) -> List[str]:
    """分析关键路径"""
    # 简化的关键路径分析：找到最长的执行路径
    if not spans:
        return []
    
    # 按持续时间排序，取前几个最耗时的操作
    sorted_spans = sorted(spans, key=lambda s: s.duration_ms or 0, reverse=True)
    return [span.operation_name for span in sorted_spans[:5]]


def _analyze_bottlenecks(spans: List[SpanResponse]) -> List[Dict[str, Any]]:
    """分析性能瓶颈"""
    bottlenecks = []
    
    if not spans:
        return bottlenecks
    
    # 找到持续时间超过平均值2倍的跨度
    durations = [span.duration_ms for span in spans if span.duration_ms]
    if durations:
        avg_duration = sum(durations) / len(durations)
        threshold = avg_duration * 2
        
        for span in spans:
            if span.duration_ms and span.duration_ms > threshold:
                bottlenecks.append({
                    'span_id': span.span_id,
                    'operation': span.operation_name,
                    'duration_ms': span.duration_ms,
                    'severity': 'high' if span.duration_ms > threshold * 1.5 else 'medium'
                })
    
    return bottlenecks


def _analyze_parallel_execution(spans: List[SpanResponse]) -> Dict[str, Any]:
    """分析并行执行情况"""
    if not spans:
        return {'parallel_spans': 0, 'sequential_spans': 0, 'parallelism_ratio': 0}
    
    # 简化分析：检查时间重叠的跨度
    parallel_count = 0
    total_spans = len(spans)
    
    for i, span1 in enumerate(spans):
        if not span1.end_time:
            continue
        
        for j, span2 in enumerate(spans[i+1:], i+1):
            if not span2.end_time:
                continue
            
            # 检查时间重叠
            if (span1.start_time < span2.end_time and 
                span2.start_time < span1.end_time):
                parallel_count += 1
                break
    
    return {
        'parallel_spans': parallel_count,
        'sequential_spans': total_spans - parallel_count,
        'parallelism_ratio': parallel_count / total_spans if total_spans > 0 else 0
    }


def _analyze_resource_utilization(spans: List[SpanResponse]) -> Dict[str, Any]:
    """分析资源利用率"""
    services = set()
    operations = {}
    
    for span in spans:
        if span.tags and 'service' in span.tags:
            services.add(span.tags['service'])
        
        if span.operation_name in operations:
            operations[span.operation_name] += 1
        else:
            operations[span.operation_name] = 1
    
    return {
        'unique_services': len(services),
        'operation_distribution': operations,
        'most_used_operation': max(operations.items(), key=lambda x: x[1])[0] if operations else None
    }


def _analyze_errors(spans: List[SpanResponse]) -> Dict[str, Any]:
    """分析错误情况"""
    error_spans = [span for span in spans if span.status != 'ok']
    error_operations = {}
    
    for span in error_spans:
        if span.operation_name in error_operations:
            error_operations[span.operation_name] += 1
        else:
            error_operations[span.operation_name] = 1
    
    return {
        'total_errors': len(error_spans),
        'error_rate': len(error_spans) / len(spans) if spans else 0,
        'error_operations': error_operations,
        'most_error_prone_operation': max(error_operations.items(), key=lambda x: x[1])[0] if error_operations else None
    }


def _generate_performance_recommendations(spans: List[SpanResponse]) -> List[str]:
    """生成性能优化建议"""
    recommendations = []
    
    if not spans:
        return recommendations
    
    # 分析持续时间
    durations = [span.duration_ms for span in spans if span.duration_ms]
    if durations:
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        
        if max_duration > avg_duration * 3:
            recommendations.append("存在异常耗时的操作，建议优化最慢的操作")
        
        if avg_duration > 1000:  # 1秒
            recommendations.append("平均响应时间较长，建议优化整体性能")
    
    # 分析错误率
    error_count = len([span for span in spans if span.status != 'ok'])
    if error_count > 0:
        error_rate = error_count / len(spans)
        if error_rate > 0.1:  # 10%
            recommendations.append("错误率较高，建议检查错误处理逻辑")
    
    # 分析并行度
    parallel_analysis = _analyze_parallel_execution(spans)
    if parallel_analysis['parallelism_ratio'] < 0.3:
        recommendations.append("并行度较低，建议优化并发执行")
    
    return recommendations