#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志API路由

提供日志记录、查询和管理功能，包括：
- API调用日志记录
- 日志查询和过滤
- 批量日志操作
- 日志统计信息
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum

from ...database.manager import DatabaseManager
from ...utils.logger import get_logger
from ...core.models import APILogEntry, TokenUsage, CostInfo
from ..schemas import StandardResponse, PaginatedResponse, ErrorResponse

logger = get_logger("harborai.api.logs")

logs_router = APIRouter(prefix="/v1/logs", tags=["logs"])


class LogLevel(str, Enum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogType(str, Enum):
    """日志类型枚举"""
    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    ALL = "all"


class LogCreateRequest(BaseModel):
    """创建日志请求模型"""
    trace_id: str = Field(..., description="追踪ID")
    span_id: Optional[str] = Field(None, description="跨度ID")
    model: str = Field(..., description="使用的模型")
    provider: Optional[str] = Field(None, description="服务提供商")
    prompt_tokens: int = Field(..., ge=0, description="提示词token数量")
    completion_tokens: int = Field(..., ge=0, description="完成token数量")
    total_tokens: int = Field(..., ge=0, description="总token数量")
    total_cost: float = Field(..., ge=0, description="总成本")
    duration_ms: float = Field(..., ge=0, description="请求持续时间(毫秒)")
    status: str = Field(..., description="请求状态")
    error_message: Optional[str] = Field(None, description="错误信息")
    request_data: Optional[Dict[str, Any]] = Field(None, description="请求数据")
    response_data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class LogUpdateRequest(BaseModel):
    """更新日志请求模型"""
    status: Optional[str] = Field(None, description="更新状态")
    error_message: Optional[str] = Field(None, description="错误信息")
    completion_tokens: Optional[int] = Field(None, ge=0, description="完成token数量")
    total_tokens: Optional[int] = Field(None, ge=0, description="总token数量")
    total_cost: Optional[float] = Field(None, ge=0, description="总成本")
    duration_ms: Optional[float] = Field(None, ge=0, description="请求持续时间")
    response_data: Optional[Dict[str, Any]] = Field(None, description="响应数据")
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")


class LogQueryParams(BaseModel):
    """日志查询参数"""
    trace_id: Optional[str] = Field(None, description="追踪ID")
    model: Optional[str] = Field(None, description="模型名称")
    provider: Optional[str] = Field(None, description="服务提供商")
    status: Optional[str] = Field(None, description="状态")
    start_time: Optional[datetime] = Field(None, description="开始时间")
    end_time: Optional[datetime] = Field(None, description="结束时间")
    min_cost: Optional[float] = Field(None, ge=0, description="最小成本")
    max_cost: Optional[float] = Field(None, ge=0, description="最大成本")
    min_tokens: Optional[int] = Field(None, ge=0, description="最小token数")
    max_tokens: Optional[int] = Field(None, ge=0, description="最大token数")
    log_type: LogType = Field(LogType.ALL, description="日志类型")
    page: int = Field(1, ge=1, description="页码")
    page_size: int = Field(50, ge=1, le=1000, description="每页大小")
    sort_by: str = Field("created_at", description="排序字段")
    sort_order: str = Field("desc", regex="^(asc|desc)$", description="排序顺序")


class LogResponse(BaseModel):
    """日志响应模型"""
    id: int
    trace_id: str
    span_id: Optional[str]
    model: str
    provider: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    total_cost: float
    duration_ms: float
    status: str
    error_message: Optional[str]
    request_data: Optional[Dict[str, Any]]
    response_data: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: Optional[datetime]
    
    # 关联数据
    token_usage: Optional[Dict[str, Any]] = None
    cost_info: Optional[Dict[str, Any]] = None
    tracing_info: Optional[Dict[str, Any]] = None


class LogStatistics(BaseModel):
    """日志统计信息"""
    total_logs: int
    total_tokens: int
    total_cost: float
    average_duration: float
    success_rate: float
    error_rate: float
    models_used: List[str]
    providers_used: List[str]
    time_range: Dict[str, datetime]


async def get_db_manager() -> DatabaseManager:
    """获取数据库管理器依赖"""
    # 这里应该从依赖注入容器获取
    # 暂时创建一个新实例
    db_manager = DatabaseManager()
    await db_manager.initialize()
    return db_manager


@logs_router.post("/", response_model=StandardResponse[LogResponse])
async def create_log(
    request: LogCreateRequest,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    创建新的API调用日志
    
    创建一条新的API调用日志记录，包括token使用情况和成本信息。
    """
    try:
        # 插入主日志记录
        log_data = {
            'trace_id': request.trace_id,
            'span_id': request.span_id,
            'model': request.model,
            'provider': request.provider,
            'prompt_tokens': request.prompt_tokens,
            'completion_tokens': request.completion_tokens,
            'total_tokens': request.total_tokens,
            'total_cost': request.total_cost,
            'duration_ms': request.duration_ms,
            'status': request.status,
            'error_message': request.error_message,
            'request_data': request.request_data,
            'response_data': request.response_data,
            'metadata': request.metadata,
            'created_at': datetime.now()
        }
        
        # 插入API日志
        result = await db_manager.execute_query("""
            INSERT INTO api_logs (
                trace_id, span_id, model, provider, prompt_tokens, completion_tokens,
                total_tokens, total_cost, duration_ms, status, error_message,
                request_data, response_data, metadata, created_at
            ) VALUES (
                %(trace_id)s, %(span_id)s, %(model)s, %(provider)s, %(prompt_tokens)s,
                %(completion_tokens)s, %(total_tokens)s, %(total_cost)s, %(duration_ms)s,
                %(status)s, %(error_message)s, %(request_data)s, %(response_data)s,
                %(metadata)s, %(created_at)s
            ) RETURNING id
        """, log_data)
        
        log_id = result[0]['id']
        
        # 插入token使用记录
        await db_manager.execute_query("""
            INSERT INTO token_usage (log_id, prompt_tokens, completion_tokens, total_tokens)
            VALUES (%(log_id)s, %(prompt_tokens)s, %(completion_tokens)s, %(total_tokens)s)
        """, {
            'log_id': log_id,
            'prompt_tokens': request.prompt_tokens,
            'completion_tokens': request.completion_tokens,
            'total_tokens': request.total_tokens
        })
        
        # 插入成本信息
        prompt_cost = request.total_cost * 0.6  # 假设提示词占60%成本
        completion_cost = request.total_cost * 0.4  # 完成占40%成本
        
        await db_manager.execute_query("""
            INSERT INTO cost_info (log_id, prompt_cost, completion_cost, total_cost)
            VALUES (%(log_id)s, %(prompt_cost)s, %(completion_cost)s, %(total_cost)s)
        """, {
            'log_id': log_id,
            'prompt_cost': prompt_cost,
            'completion_cost': completion_cost,
            'total_cost': request.total_cost
        })
        
        # 获取完整的日志记录
        log_record = await _get_log_by_id(db_manager, log_id)
        
        return StandardResponse(
            success=True,
            data=log_record,
            message="日志创建成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"创建日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建日志失败: {str(e)}")


@logs_router.get("/{log_id}", response_model=StandardResponse[LogResponse])
async def get_log(
    log_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    根据ID获取日志记录
    
    获取指定ID的日志记录，包括关联的token使用和成本信息。
    """
    try:
        log_record = await _get_log_by_id(db_manager, log_id)
        
        if not log_record:
            raise HTTPException(status_code=404, detail="日志记录不存在")
        
        return StandardResponse(
            success=True,
            data=log_record,
            message="获取日志成功",
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取日志失败: {str(e)}")


@logs_router.put("/{log_id}", response_model=StandardResponse[LogResponse])
async def update_log(
    log_id: int,
    request: LogUpdateRequest,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    更新日志记录
    
    更新指定ID的日志记录信息。
    """
    try:
        # 检查日志是否存在
        existing_log = await _get_log_by_id(db_manager, log_id)
        if not existing_log:
            raise HTTPException(status_code=404, detail="日志记录不存在")
        
        # 构建更新数据
        update_data = {'id': log_id, 'updated_at': datetime.now()}
        update_fields = []
        
        if request.status is not None:
            update_data['status'] = request.status
            update_fields.append('status = %(status)s')
        
        if request.error_message is not None:
            update_data['error_message'] = request.error_message
            update_fields.append('error_message = %(error_message)s')
        
        if request.completion_tokens is not None:
            update_data['completion_tokens'] = request.completion_tokens
            update_fields.append('completion_tokens = %(completion_tokens)s')
        
        if request.total_tokens is not None:
            update_data['total_tokens'] = request.total_tokens
            update_fields.append('total_tokens = %(total_tokens)s')
        
        if request.total_cost is not None:
            update_data['total_cost'] = request.total_cost
            update_fields.append('total_cost = %(total_cost)s')
        
        if request.duration_ms is not None:
            update_data['duration_ms'] = request.duration_ms
            update_fields.append('duration_ms = %(duration_ms)s')
        
        if request.response_data is not None:
            update_data['response_data'] = request.response_data
            update_fields.append('response_data = %(response_data)s')
        
        if request.metadata is not None:
            update_data['metadata'] = request.metadata
            update_fields.append('metadata = %(metadata)s')
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="没有提供更新字段")
        
        # 更新主日志记录
        update_fields.append('updated_at = %(updated_at)s')
        update_query = f"""
            UPDATE api_logs 
            SET {', '.join(update_fields)}
            WHERE id = %(id)s
        """
        
        await db_manager.execute_query(update_query, update_data)
        
        # 更新相关表
        if request.completion_tokens is not None or request.total_tokens is not None:
            token_update_data = {'log_id': log_id}
            token_fields = []
            
            if request.completion_tokens is not None:
                token_update_data['completion_tokens'] = request.completion_tokens
                token_fields.append('completion_tokens = %(completion_tokens)s')
            
            if request.total_tokens is not None:
                token_update_data['total_tokens'] = request.total_tokens
                token_fields.append('total_tokens = %(total_tokens)s')
            
            if token_fields:
                await db_manager.execute_query(f"""
                    UPDATE token_usage 
                    SET {', '.join(token_fields)}
                    WHERE log_id = %(log_id)s
                """, token_update_data)
        
        if request.total_cost is not None:
            # 重新计算成本分配
            prompt_cost = request.total_cost * 0.6
            completion_cost = request.total_cost * 0.4
            
            await db_manager.execute_query("""
                UPDATE cost_info 
                SET prompt_cost = %(prompt_cost)s, 
                    completion_cost = %(completion_cost)s, 
                    total_cost = %(total_cost)s
                WHERE log_id = %(log_id)s
            """, {
                'log_id': log_id,
                'prompt_cost': prompt_cost,
                'completion_cost': completion_cost,
                'total_cost': request.total_cost
            })
        
        # 获取更新后的日志记录
        updated_log = await _get_log_by_id(db_manager, log_id)
        
        return StandardResponse(
            success=True,
            data=updated_log,
            message="日志更新成功",
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新日志失败: {str(e)}")


@logs_router.delete("/{log_id}", response_model=StandardResponse[Dict[str, Any]])
async def delete_log(
    log_id: int,
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    删除日志记录
    
    删除指定ID的日志记录及其关联数据。
    """
    try:
        # 检查日志是否存在
        existing_log = await _get_log_by_id(db_manager, log_id)
        if not existing_log:
            raise HTTPException(status_code=404, detail="日志记录不存在")
        
        # 删除关联数据（由于外键约束，需要先删除子表数据）
        await db_manager.execute_query("DELETE FROM tracing_info WHERE log_id = %s", (log_id,))
        await db_manager.execute_query("DELETE FROM cost_info WHERE log_id = %s", (log_id,))
        await db_manager.execute_query("DELETE FROM token_usage WHERE log_id = %s", (log_id,))
        
        # 删除主日志记录
        await db_manager.execute_query("DELETE FROM api_logs WHERE id = %s", (log_id,))
        
        return StandardResponse(
            success=True,
            data={"deleted_log_id": log_id},
            message="日志删除成功",
            timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除日志失败: {str(e)}")


@logs_router.get("/", response_model=PaginatedResponse[LogResponse])
async def query_logs(
    trace_id: Optional[str] = Query(None, description="追踪ID"),
    model: Optional[str] = Query(None, description="模型名称"),
    provider: Optional[str] = Query(None, description="服务提供商"),
    status: Optional[str] = Query(None, description="状态"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    min_cost: Optional[float] = Query(None, ge=0, description="最小成本"),
    max_cost: Optional[float] = Query(None, ge=0, description="最大成本"),
    min_tokens: Optional[int] = Query(None, ge=0, description="最小token数"),
    max_tokens: Optional[int] = Query(None, ge=0, description="最大token数"),
    log_type: LogType = Query(LogType.ALL, description="日志类型"),
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(50, ge=1, le=1000, description="每页大小"),
    sort_by: str = Query("created_at", description="排序字段"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="排序顺序"),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    查询日志记录
    
    根据各种条件查询日志记录，支持分页和排序。
    """
    try:
        # 构建查询条件
        where_conditions = []
        query_params = {}
        
        if trace_id:
            where_conditions.append("al.trace_id = %(trace_id)s")
            query_params['trace_id'] = trace_id
        
        if model:
            where_conditions.append("al.model = %(model)s")
            query_params['model'] = model
        
        if provider:
            where_conditions.append("al.provider = %(provider)s")
            query_params['provider'] = provider
        
        if status:
            where_conditions.append("al.status = %(status)s")
            query_params['status'] = status
        
        if start_time:
            where_conditions.append("al.created_at >= %(start_time)s")
            query_params['start_time'] = start_time
        
        if end_time:
            where_conditions.append("al.created_at <= %(end_time)s")
            query_params['end_time'] = end_time
        
        if min_cost is not None:
            where_conditions.append("al.total_cost >= %(min_cost)s")
            query_params['min_cost'] = min_cost
        
        if max_cost is not None:
            where_conditions.append("al.total_cost <= %(max_cost)s")
            query_params['max_cost'] = max_cost
        
        if min_tokens is not None:
            where_conditions.append("al.total_tokens >= %(min_tokens)s")
            query_params['min_tokens'] = min_tokens
        
        if max_tokens is not None:
            where_conditions.append("al.total_tokens <= %(max_tokens)s")
            query_params['max_tokens'] = max_tokens
        
        # 构建WHERE子句
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 计算总数
        count_query = f"""
            SELECT COUNT(*) as total
            FROM api_logs al
            {where_clause}
        """
        
        count_result = await db_manager.execute_query(count_query, query_params)
        total_count = count_result[0]['total']
        
        # 计算分页
        offset = (page - 1) * page_size
        query_params.update({
            'limit': page_size,
            'offset': offset
        })
        
        # 构建主查询
        main_query = f"""
            SELECT 
                al.*,
                tu.prompt_tokens as tu_prompt_tokens,
                tu.completion_tokens as tu_completion_tokens,
                tu.total_tokens as tu_total_tokens,
                ci.prompt_cost,
                ci.completion_cost,
                ci.total_cost as ci_total_cost,
                ti.parent_span_id,
                ti.operation_name,
                ti.tags,
                ti.logs as trace_logs
            FROM api_logs al
            LEFT JOIN token_usage tu ON al.id = tu.log_id
            LEFT JOIN cost_info ci ON al.id = ci.log_id
            LEFT JOIN tracing_info ti ON al.id = ti.log_id
            {where_clause}
            ORDER BY al.{sort_by} {sort_order.upper()}
            LIMIT %(limit)s OFFSET %(offset)s
        """
        
        results = await db_manager.execute_query(main_query, query_params)
        
        # 转换结果
        logs = []
        for row in results:
            log_data = LogResponse(
                id=row['id'],
                trace_id=row['trace_id'],
                span_id=row['span_id'],
                model=row['model'],
                provider=row['provider'],
                prompt_tokens=row['prompt_tokens'],
                completion_tokens=row['completion_tokens'],
                total_tokens=row['total_tokens'],
                total_cost=row['total_cost'],
                duration_ms=row['duration_ms'],
                status=row['status'],
                error_message=row['error_message'],
                request_data=row['request_data'],
                response_data=row['response_data'],
                metadata=row['metadata'],
                created_at=row['created_at'],
                updated_at=row['updated_at']
            )
            
            # 添加关联数据
            if row['tu_prompt_tokens'] is not None:
                log_data.token_usage = {
                    'prompt_tokens': row['tu_prompt_tokens'],
                    'completion_tokens': row['tu_completion_tokens'],
                    'total_tokens': row['tu_total_tokens']
                }
            
            if row['prompt_cost'] is not None:
                log_data.cost_info = {
                    'prompt_cost': row['prompt_cost'],
                    'completion_cost': row['completion_cost'],
                    'total_cost': row['ci_total_cost']
                }
            
            if row['parent_span_id'] is not None:
                log_data.tracing_info = {
                    'parent_span_id': row['parent_span_id'],
                    'operation_name': row['operation_name'],
                    'tags': row['tags'],
                    'logs': row['trace_logs']
                }
            
            logs.append(log_data)
        
        # 计算分页信息
        total_pages = (total_count + page_size - 1) // page_size
        
        return PaginatedResponse(
            success=True,
            data=logs,
            pagination={
                'page': page,
                'page_size': page_size,
                'total_count': total_count,
                'total_pages': total_pages,
                'has_next': page < total_pages,
                'has_prev': page > 1
            },
            message="查询日志成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"查询日志失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询日志失败: {str(e)}")


@logs_router.get("/statistics", response_model=StandardResponse[LogStatistics])
async def get_log_statistics(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    model: Optional[str] = Query(None, description="模型名称"),
    provider: Optional[str] = Query(None, description="服务提供商"),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    获取日志统计信息
    
    获取指定时间范围和条件下的日志统计信息。
    """
    try:
        # 构建查询条件
        where_conditions = []
        query_params = {}
        
        if start_time:
            where_conditions.append("created_at >= %(start_time)s")
            query_params['start_time'] = start_time
        
        if end_time:
            where_conditions.append("created_at <= %(end_time)s")
            query_params['end_time'] = end_time
        
        if model:
            where_conditions.append("model = %(model)s")
            query_params['model'] = model
        
        if provider:
            where_conditions.append("provider = %(provider)s")
            query_params['provider'] = provider
        
        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 获取统计信息
        stats_query = f"""
            SELECT 
                COUNT(*) as total_logs,
                SUM(total_tokens) as total_tokens,
                SUM(total_cost) as total_cost,
                AVG(duration_ms) as average_duration,
                COUNT(CASE WHEN status = 'success' THEN 1 END) as success_count,
                COUNT(CASE WHEN status != 'success' THEN 1 END) as error_count,
                ARRAY_AGG(DISTINCT model) as models_used,
                ARRAY_AGG(DISTINCT provider) as providers_used,
                MIN(created_at) as earliest_log,
                MAX(created_at) as latest_log
            FROM api_logs
            {where_clause}
        """
        
        result = await db_manager.execute_query(stats_query, query_params)
        stats = result[0]
        
        total_logs = stats['total_logs'] or 0
        success_count = stats['success_count'] or 0
        error_count = stats['error_count'] or 0
        
        statistics = LogStatistics(
            total_logs=total_logs,
            total_tokens=stats['total_tokens'] or 0,
            total_cost=stats['total_cost'] or 0.0,
            average_duration=stats['average_duration'] or 0.0,
            success_rate=success_count / total_logs if total_logs > 0 else 0.0,
            error_rate=error_count / total_logs if total_logs > 0 else 0.0,
            models_used=[m for m in (stats['models_used'] or []) if m],
            providers_used=[p for p in (stats['providers_used'] or []) if p],
            time_range={
                'start': stats['earliest_log'],
                'end': stats['latest_log']
            }
        )
        
        return StandardResponse(
            success=True,
            data=statistics,
            message="获取统计信息成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


async def _get_log_by_id(db_manager: DatabaseManager, log_id: int) -> Optional[LogResponse]:
    """根据ID获取完整的日志记录"""
    query = """
        SELECT 
            al.*,
            tu.prompt_tokens as tu_prompt_tokens,
            tu.completion_tokens as tu_completion_tokens,
            tu.total_tokens as tu_total_tokens,
            ci.prompt_cost,
            ci.completion_cost,
            ci.total_cost as ci_total_cost,
            ti.parent_span_id,
            ti.operation_name,
            ti.tags,
            ti.logs as trace_logs
        FROM api_logs al
        LEFT JOIN token_usage tu ON al.id = tu.log_id
        LEFT JOIN cost_info ci ON al.id = ci.log_id
        LEFT JOIN tracing_info ti ON al.id = ti.log_id
        WHERE al.id = %s
    """
    
    result = await db_manager.execute_query(query, (log_id,))
    
    if not result:
        return None
    
    row = result[0]
    log_data = LogResponse(
        id=row['id'],
        trace_id=row['trace_id'],
        span_id=row['span_id'],
        model=row['model'],
        provider=row['provider'],
        prompt_tokens=row['prompt_tokens'],
        completion_tokens=row['completion_tokens'],
        total_tokens=row['total_tokens'],
        total_cost=row['total_cost'],
        duration_ms=row['duration_ms'],
        status=row['status'],
        error_message=row['error_message'],
        request_data=row['request_data'],
        response_data=row['response_data'],
        metadata=row['metadata'],
        created_at=row['created_at'],
        updated_at=row['updated_at']
    )
    
    # 添加关联数据
    if row['tu_prompt_tokens'] is not None:
        log_data.token_usage = {
            'prompt_tokens': row['tu_prompt_tokens'],
            'completion_tokens': row['tu_completion_tokens'],
            'total_tokens': row['tu_total_tokens']
        }
    
    if row['prompt_cost'] is not None:
        log_data.cost_info = {
            'prompt_cost': row['prompt_cost'],
            'completion_cost': row['completion_cost'],
            'total_cost': row['ci_total_cost']
        }
    
    if row['parent_span_id'] is not None:
        log_data.tracing_info = {
            'parent_span_id': row['parent_span_id'],
            'operation_name': row['operation_name'],
            'tags': row['tags'],
            'logs': row['trace_logs']
        }
    
    return log_data