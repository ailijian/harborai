#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计API路由

提供各种统计信息查询功能，包括：
- Token使用统计
- 成本统计
- 性能统计
- 错误统计
- 趋势分析
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timedelta
from enum import Enum

from ...database.manager import DatabaseManager
from ...utils.logger import get_logger
from ..schemas import StandardResponse, StatisticsResponse, ResponseBuilder, MetaDataBuilder

logger = get_logger("harborai.api.statistics")

statistics_router = APIRouter(prefix="/v1/statistics", tags=["statistics"])


class TimeGranularity(str, Enum):
    """时间粒度枚举"""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class MetricType(str, Enum):
    """指标类型枚举"""
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    PERFORMANCE = "performance"
    ERROR_RATE = "error_rate"
    REQUEST_COUNT = "request_count"
    ALL = "all"


class AggregationType(str, Enum):
    """聚合类型枚举"""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"


class TokenUsageStatistics(BaseModel):
    """Token使用统计"""
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    average_tokens_per_request: float
    peak_usage_hour: Optional[Dict[str, Any]]
    model_distribution: Dict[str, int]
    provider_distribution: Dict[str, int]
    daily_usage: List[Dict[str, Any]]


class CostStatistics(BaseModel):
    """成本统计"""
    total_cost: float
    prompt_cost: float
    completion_cost: float
    average_cost_per_request: float
    cost_by_model: Dict[str, float]
    cost_by_provider: Dict[str, float]
    daily_cost: List[Dict[str, Any]]
    cost_trends: Dict[str, Any]


class PerformanceStatistics(BaseModel):
    """性能统计"""
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    fastest_request: Dict[str, Any]
    slowest_request: Dict[str, Any]
    performance_by_model: Dict[str, Dict[str, float]]
    performance_trends: List[Dict[str, Any]]


class ErrorStatistics(BaseModel):
    """错误统计"""
    total_errors: int
    error_rate: float
    error_by_type: Dict[str, int]
    error_by_model: Dict[str, int]
    error_by_provider: Dict[str, int]
    error_trends: List[Dict[str, Any]]
    most_common_errors: List[Dict[str, Any]]


class OverallStatistics(BaseModel):
    """综合统计"""
    period: Dict[str, datetime]
    total_requests: int
    success_rate: float
    token_usage: TokenUsageStatistics
    cost: CostStatistics
    performance: PerformanceStatistics
    errors: ErrorStatistics
    top_models: List[Dict[str, Any]]
    top_providers: List[Dict[str, Any]]


async def get_db_manager() -> DatabaseManager:
    """获取数据库管理器依赖"""
    db_manager = DatabaseManager()
    await db_manager.initialize()
    return db_manager


@statistics_router.get("/overview", response_model=StandardResponse[OverallStatistics])
async def get_overview_statistics(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    model: Optional[str] = Query(None, description="模型名称"),
    provider: Optional[str] = Query(None, description="服务提供商"),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    获取综合统计概览
    
    获取指定时间范围内的综合统计信息，包括token使用、成本、性能和错误统计。
    """
    try:
        # 设置默认时间范围（最近7天）
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        # 构建查询条件
        where_conditions = ["al.created_at >= %(start_time)s", "al.created_at <= %(end_time)s"]
        query_params = {'start_time': start_time, 'end_time': end_time}
        
        if model:
            where_conditions.append("al.model = %(model)s")
            query_params['model'] = model
        
        if provider:
            where_conditions.append("al.provider = %(provider)s")
            query_params['provider'] = provider
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 获取基础统计
        basic_stats_query = f"""
            SELECT 
                COUNT(*) as total_requests,
                COUNT(CASE WHEN al.status = 'success' THEN 1 END) as success_requests,
                SUM(tu.total_tokens) as total_tokens,
                SUM(tu.prompt_tokens) as total_prompt_tokens,
                SUM(tu.completion_tokens) as total_completion_tokens,
                SUM(ci.total_cost) as total_cost,
                SUM(ci.prompt_cost) as total_prompt_cost,
                SUM(ci.completion_cost) as total_completion_cost,
                AVG(al.duration_ms) as avg_duration,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY al.duration_ms) as median_duration,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY al.duration_ms) as p95_duration,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY al.duration_ms) as p99_duration,
                MIN(al.duration_ms) as min_duration,
                MAX(al.duration_ms) as max_duration
            FROM api_logs al
            LEFT JOIN token_usage tu ON al.id = tu.log_id
            LEFT JOIN cost_info ci ON al.id = ci.log_id
            {where_clause}
        """
        
        basic_result = await db_manager.execute_query(basic_stats_query, query_params)
        basic_stats = basic_result[0]
        
        # Token使用统计
        token_stats = await _get_token_statistics(db_manager, where_clause, query_params)
        
        # 成本统计
        cost_stats = await _get_cost_statistics(db_manager, where_clause, query_params)
        
        # 性能统计
        performance_stats = await _get_performance_statistics(db_manager, where_clause, query_params, basic_stats)
        
        # 错误统计
        error_stats = await _get_error_statistics(db_manager, where_clause, query_params)
        
        # 获取热门模型和提供商
        top_models = await _get_top_models(db_manager, where_clause, query_params)
        top_providers = await _get_top_providers(db_manager, where_clause, query_params)
        
        # 构建综合统计
        total_requests = basic_stats['total_requests'] or 0
        success_requests = basic_stats['success_requests'] or 0
        
        overall_stats = OverallStatistics(
            period={'start': start_time, 'end': end_time},
            total_requests=total_requests,
            success_rate=success_requests / total_requests if total_requests > 0 else 0,
            token_usage=token_stats,
            cost=cost_stats,
            performance=performance_stats,
            errors=error_stats,
            top_models=top_models,
            top_providers=top_providers
        )
        
        return StandardResponse(
            success=True,
            data=overall_stats,
            message="获取综合统计成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取综合统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取综合统计失败: {str(e)}")


@statistics_router.get("/tokens", response_model=StandardResponse[TokenUsageStatistics])
async def get_token_statistics(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    model: Optional[str] = Query(None, description="模型名称"),
    provider: Optional[str] = Query(None, description="服务提供商"),
    granularity: TimeGranularity = Query(TimeGranularity.DAY, description="时间粒度"),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    获取Token使用统计
    
    获取指定时间范围内的Token使用统计信息。
    """
    try:
        # 设置默认时间范围
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        # 构建查询条件
        where_conditions = ["al.created_at >= %(start_time)s", "al.created_at <= %(end_time)s"]
        query_params = {'start_time': start_time, 'end_time': end_time}
        
        if model:
            where_conditions.append("al.model = %(model)s")
            query_params['model'] = model
        
        if provider:
            where_conditions.append("al.provider = %(provider)s")
            query_params['provider'] = provider
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        token_stats = await _get_token_statistics(db_manager, where_clause, query_params, granularity)
        
        return StandardResponse(
            success=True,
            data=token_stats,
            message="获取Token统计成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取Token统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取Token统计失败: {str(e)}")


@statistics_router.get("/costs", response_model=StandardResponse[CostStatistics])
async def get_cost_statistics(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    model: Optional[str] = Query(None, description="模型名称"),
    provider: Optional[str] = Query(None, description="服务提供商"),
    granularity: TimeGranularity = Query(TimeGranularity.DAY, description="时间粒度"),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    获取成本统计
    
    获取指定时间范围内的成本统计信息。
    """
    try:
        # 设置默认时间范围
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        # 构建查询条件
        where_conditions = ["al.created_at >= %(start_time)s", "al.created_at <= %(end_time)s"]
        query_params = {'start_time': start_time, 'end_time': end_time}
        
        if model:
            where_conditions.append("al.model = %(model)s")
            query_params['model'] = model
        
        if provider:
            where_conditions.append("al.provider = %(provider)s")
            query_params['provider'] = provider
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        cost_stats = await _get_cost_statistics(db_manager, where_clause, query_params, granularity)
        
        return StandardResponse(
            success=True,
            data=cost_stats,
            message="获取成本统计成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取成本统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取成本统计失败: {str(e)}")


@statistics_router.get("/performance", response_model=StandardResponse[PerformanceStatistics])
async def get_performance_statistics(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    model: Optional[str] = Query(None, description="模型名称"),
    provider: Optional[str] = Query(None, description="服务提供商"),
    granularity: TimeGranularity = Query(TimeGranularity.DAY, description="时间粒度"),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    获取性能统计
    
    获取指定时间范围内的性能统计信息。
    """
    try:
        # 设置默认时间范围
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        # 构建查询条件
        where_conditions = ["al.created_at >= %(start_time)s", "al.created_at <= %(end_time)s"]
        query_params = {'start_time': start_time, 'end_time': end_time}
        
        if model:
            where_conditions.append("al.model = %(model)s")
            query_params['model'] = model
        
        if provider:
            where_conditions.append("al.provider = %(provider)s")
            query_params['provider'] = provider
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 获取基础性能数据
        basic_perf_query = f"""
            SELECT 
                AVG(duration_ms) as avg_duration,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms) as median_duration,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) as p95_duration,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) as p99_duration,
                MIN(duration_ms) as min_duration,
                MAX(duration_ms) as max_duration
            FROM api_logs al
            {where_clause}
        """
        
        basic_perf_result = await db_manager.execute_query(basic_perf_query, query_params)
        basic_perf = basic_perf_result[0]
        
        performance_stats = await _get_performance_statistics(db_manager, where_clause, query_params, basic_perf, granularity)
        
        return StandardResponse(
            success=True,
            data=performance_stats,
            message="获取性能统计成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取性能统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取性能统计失败: {str(e)}")


@statistics_router.get("/errors", response_model=StandardResponse[ErrorStatistics])
async def get_error_statistics(
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    model: Optional[str] = Query(None, description="模型名称"),
    provider: Optional[str] = Query(None, description="服务提供商"),
    granularity: TimeGranularity = Query(TimeGranularity.DAY, description="时间粒度"),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    获取错误统计
    
    获取指定时间范围内的错误统计信息。
    """
    try:
        # 设置默认时间范围
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=7)
        
        # 构建查询条件
        where_conditions = ["al.created_at >= %(start_time)s", "al.created_at <= %(end_time)s"]
        query_params = {'start_time': start_time, 'end_time': end_time}
        
        if model:
            where_conditions.append("al.model = %(model)s")
            query_params['model'] = model
        
        if provider:
            where_conditions.append("al.provider = %(provider)s")
            query_params['provider'] = provider
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        error_stats = await _get_error_statistics(db_manager, where_clause, query_params, granularity)
        
        return StandardResponse(
            success=True,
            data=error_stats,
            message="获取错误统计成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取错误统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取错误统计失败: {str(e)}")


@statistics_router.get("/trends", response_model=StandardResponse[Dict[str, Any]])
async def get_trend_analysis(
    metric_type: MetricType = Query(MetricType.ALL, description="指标类型"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    granularity: TimeGranularity = Query(TimeGranularity.DAY, description="时间粒度"),
    model: Optional[str] = Query(None, description="模型名称"),
    provider: Optional[str] = Query(None, description="服务提供商"),
    db_manager: DatabaseManager = Depends(get_db_manager)
):
    """
    获取趋势分析
    
    获取指定指标的趋势分析数据。
    """
    try:
        # 设置默认时间范围
        if not end_time:
            end_time = datetime.now()
        if not start_time:
            start_time = end_time - timedelta(days=30)  # 默认30天
        
        # 构建查询条件
        where_conditions = ["al.created_at >= %(start_time)s", "al.created_at <= %(end_time)s"]
        query_params = {'start_time': start_time, 'end_time': end_time}
        
        if model:
            where_conditions.append("al.model = %(model)s")
            query_params['model'] = model
        
        if provider:
            where_conditions.append("al.provider = %(provider)s")
            query_params['provider'] = provider
        
        where_clause = "WHERE " + " AND ".join(where_conditions)
        
        # 根据指标类型获取趋势数据
        trends = {}
        
        if metric_type in [MetricType.TOKEN_USAGE, MetricType.ALL]:
            trends['token_usage'] = await _get_token_trends(db_manager, where_clause, query_params, granularity)
        
        if metric_type in [MetricType.COST, MetricType.ALL]:
            trends['cost'] = await _get_cost_trends(db_manager, where_clause, query_params, granularity)
        
        if metric_type in [MetricType.PERFORMANCE, MetricType.ALL]:
            trends['performance'] = await _get_performance_trends(db_manager, where_clause, query_params, granularity)
        
        if metric_type in [MetricType.ERROR_RATE, MetricType.ALL]:
            trends['error_rate'] = await _get_error_trends(db_manager, where_clause, query_params, granularity)
        
        if metric_type in [MetricType.REQUEST_COUNT, MetricType.ALL]:
            trends['request_count'] = await _get_request_count_trends(db_manager, where_clause, query_params, granularity)
        
        return StandardResponse(
            success=True,
            data={
                'period': {'start': start_time, 'end': end_time},
                'granularity': granularity,
                'trends': trends
            },
            message="获取趋势分析成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取趋势分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取趋势分析失败: {str(e)}")


# 辅助函数
async def _get_token_statistics(
    db_manager: DatabaseManager, 
    where_clause: str, 
    query_params: Dict[str, Any],
    granularity: TimeGranularity = TimeGranularity.DAY
) -> TokenUsageStatistics:
    """获取Token使用统计"""
    
    # 基础Token统计
    token_query = f"""
        SELECT 
            SUM(tu.total_tokens) as total_tokens,
            SUM(tu.prompt_tokens) as total_prompt_tokens,
            SUM(tu.completion_tokens) as total_completion_tokens,
            AVG(tu.total_tokens) as avg_tokens_per_request,
            COUNT(*) as request_count
        FROM api_logs al
        LEFT JOIN token_usage tu ON al.id = tu.log_id
        {where_clause}
    """
    
    token_result = await db_manager.execute_query(token_query, query_params)
    token_stats = token_result[0]
    
    # 模型分布
    model_dist_query = f"""
        SELECT al.model, SUM(tu.total_tokens) as tokens
        FROM api_logs al
        LEFT JOIN token_usage tu ON al.id = tu.log_id
        {where_clause}
        GROUP BY al.model
        ORDER BY tokens DESC
    """
    
    model_dist_result = await db_manager.execute_query(model_dist_query, query_params)
    model_distribution = {row['model']: row['tokens'] or 0 for row in model_dist_result}
    
    # 提供商分布
    provider_dist_query = f"""
        SELECT al.provider, SUM(tu.total_tokens) as tokens
        FROM api_logs al
        LEFT JOIN token_usage tu ON al.id = tu.log_id
        {where_clause}
        AND al.provider IS NOT NULL
        GROUP BY al.provider
        ORDER BY tokens DESC
    """
    
    provider_dist_result = await db_manager.execute_query(provider_dist_query, query_params)
    provider_distribution = {row['provider']: row['tokens'] or 0 for row in provider_dist_result}
    
    # 每日使用量
    daily_usage = await _get_daily_token_usage(db_manager, where_clause, query_params, granularity)
    
    # 峰值使用时间
    peak_usage = await _get_peak_token_usage(db_manager, where_clause, query_params)
    
    return TokenUsageStatistics(
        total_tokens=token_stats['total_tokens'] or 0,
        prompt_tokens=token_stats['total_prompt_tokens'] or 0,
        completion_tokens=token_stats['total_completion_tokens'] or 0,
        average_tokens_per_request=token_stats['avg_tokens_per_request'] or 0,
        peak_usage_hour=peak_usage,
        model_distribution=model_distribution,
        provider_distribution=provider_distribution,
        daily_usage=daily_usage
    )


async def _get_cost_statistics(
    db_manager: DatabaseManager, 
    where_clause: str, 
    query_params: Dict[str, Any],
    granularity: TimeGranularity = TimeGranularity.DAY
) -> CostStatistics:
    """获取成本统计"""
    
    # 基础成本统计
    cost_query = f"""
        SELECT 
            SUM(ci.total_cost) as total_cost,
            SUM(ci.prompt_cost) as total_prompt_cost,
            SUM(ci.completion_cost) as total_completion_cost,
            AVG(ci.total_cost) as avg_cost_per_request
        FROM api_logs al
        LEFT JOIN cost_info ci ON al.id = ci.log_id
        {where_clause}
    """
    
    cost_result = await db_manager.execute_query(cost_query, query_params)
    cost_stats = cost_result[0]
    
    # 按模型的成本分布
    cost_by_model_query = f"""
        SELECT al.model, SUM(ci.total_cost) as cost
        FROM api_logs al
        LEFT JOIN cost_info ci ON al.id = ci.log_id
        {where_clause}
        GROUP BY al.model
        ORDER BY cost DESC
    """
    
    cost_by_model_result = await db_manager.execute_query(cost_by_model_query, query_params)
    cost_by_model = {row['model']: row['cost'] or 0 for row in cost_by_model_result}
    
    # 按提供商的成本分布
    cost_by_provider_query = f"""
        SELECT al.provider, SUM(ci.total_cost) as cost
        FROM api_logs al
        LEFT JOIN cost_info ci ON al.id = ci.log_id
        {where_clause}
        AND al.provider IS NOT NULL
        GROUP BY al.provider
        ORDER BY cost DESC
    """
    
    cost_by_provider_result = await db_manager.execute_query(cost_by_provider_query, query_params)
    cost_by_provider = {row['provider']: row['cost'] or 0 for row in cost_by_provider_result}
    
    # 每日成本
    daily_cost = await _get_daily_cost(db_manager, where_clause, query_params, granularity)
    
    # 成本趋势
    cost_trends = await _get_cost_trends(db_manager, where_clause, query_params, granularity)
    
    return CostStatistics(
        total_cost=cost_stats['total_cost'] or 0,
        prompt_cost=cost_stats['total_prompt_cost'] or 0,
        completion_cost=cost_stats['total_completion_cost'] or 0,
        average_cost_per_request=cost_stats['avg_cost_per_request'] or 0,
        cost_by_model=cost_by_model,
        cost_by_provider=cost_by_provider,
        daily_cost=daily_cost,
        cost_trends=cost_trends
    )


async def _get_performance_statistics(
    db_manager: DatabaseManager, 
    where_clause: str, 
    query_params: Dict[str, Any],
    basic_stats: Dict[str, Any],
    granularity: TimeGranularity = TimeGranularity.DAY
) -> PerformanceStatistics:
    """获取性能统计"""
    
    # 最快和最慢的请求
    fastest_query = f"""
        SELECT al.id, al.trace_id, al.model, al.duration_ms, al.created_at
        FROM api_logs al
        {where_clause}
        ORDER BY al.duration_ms ASC
        LIMIT 1
    """
    
    slowest_query = f"""
        SELECT al.id, al.trace_id, al.model, al.duration_ms, al.created_at
        FROM api_logs al
        {where_clause}
        ORDER BY al.duration_ms DESC
        LIMIT 1
    """
    
    fastest_result = await db_manager.execute_query(fastest_query, query_params)
    slowest_result = await db_manager.execute_query(slowest_query, query_params)
    
    fastest_request = fastest_result[0] if fastest_result else {}
    slowest_request = slowest_result[0] if slowest_result else {}
    
    # 按模型的性能分布
    perf_by_model_query = f"""
        SELECT 
            al.model,
            AVG(al.duration_ms) as avg_duration,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY al.duration_ms) as median_duration,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY al.duration_ms) as p95_duration
        FROM api_logs al
        {where_clause}
        GROUP BY al.model
    """
    
    perf_by_model_result = await db_manager.execute_query(perf_by_model_query, query_params)
    performance_by_model = {}
    for row in perf_by_model_result:
        performance_by_model[row['model']] = {
            'avg_duration': row['avg_duration'] or 0,
            'median_duration': row['median_duration'] or 0,
            'p95_duration': row['p95_duration'] or 0
        }
    
    # 性能趋势
    performance_trends = await _get_performance_trends(db_manager, where_clause, query_params, granularity)
    
    return PerformanceStatistics(
        average_response_time=basic_stats['avg_duration'] or 0,
        median_response_time=basic_stats['median_duration'] or 0,
        p95_response_time=basic_stats['p95_duration'] or 0,
        p99_response_time=basic_stats['p99_duration'] or 0,
        fastest_request=fastest_request,
        slowest_request=slowest_request,
        performance_by_model=performance_by_model,
        performance_trends=performance_trends
    )


async def _get_error_statistics(
    db_manager: DatabaseManager, 
    where_clause: str, 
    query_params: Dict[str, Any],
    granularity: TimeGranularity = TimeGranularity.DAY
) -> ErrorStatistics:
    """获取错误统计"""
    
    # 基础错误统计
    error_query = f"""
        SELECT 
            COUNT(*) as total_requests,
            COUNT(CASE WHEN al.status != 'success' THEN 1 END) as total_errors
        FROM api_logs al
        {where_clause}
    """
    
    error_result = await db_manager.execute_query(error_query, query_params)
    error_stats = error_result[0]
    
    total_requests = error_stats['total_requests'] or 0
    total_errors = error_stats['total_errors'] or 0
    error_rate = total_errors / total_requests if total_requests > 0 else 0
    
    # 按错误类型分布
    error_by_type_query = f"""
        SELECT al.status, COUNT(*) as count
        FROM api_logs al
        {where_clause}
        AND al.status != 'success'
        GROUP BY al.status
        ORDER BY count DESC
    """
    
    error_by_type_result = await db_manager.execute_query(error_by_type_query, query_params)
    error_by_type = {row['status']: row['count'] for row in error_by_type_result}
    
    # 按模型的错误分布
    error_by_model_query = f"""
        SELECT al.model, COUNT(*) as count
        FROM api_logs al
        {where_clause}
        AND al.status != 'success'
        GROUP BY al.model
        ORDER BY count DESC
    """
    
    error_by_model_result = await db_manager.execute_query(error_by_model_query, query_params)
    error_by_model = {row['model']: row['count'] for row in error_by_model_result}
    
    # 按提供商的错误分布
    error_by_provider_query = f"""
        SELECT al.provider, COUNT(*) as count
        FROM api_logs al
        {where_clause}
        AND al.status != 'success'
        AND al.provider IS NOT NULL
        GROUP BY al.provider
        ORDER BY count DESC
    """
    
    error_by_provider_result = await db_manager.execute_query(error_by_provider_query, query_params)
    error_by_provider = {row['provider']: row['count'] for row in error_by_provider_result}
    
    # 最常见的错误
    common_errors_query = f"""
        SELECT al.error_message, COUNT(*) as count
        FROM api_logs al
        {where_clause}
        AND al.status != 'success'
        AND al.error_message IS NOT NULL
        GROUP BY al.error_message
        ORDER BY count DESC
        LIMIT 10
    """
    
    common_errors_result = await db_manager.execute_query(common_errors_query, query_params)
    most_common_errors = [
        {'error_message': row['error_message'], 'count': row['count']}
        for row in common_errors_result
    ]
    
    # 错误趋势
    error_trends = await _get_error_trends(db_manager, where_clause, query_params, granularity)
    
    return ErrorStatistics(
        total_errors=total_errors,
        error_rate=error_rate,
        error_by_type=error_by_type,
        error_by_model=error_by_model,
        error_by_provider=error_by_provider,
        error_trends=error_trends,
        most_common_errors=most_common_errors
    )


async def _get_top_models(db_manager: DatabaseManager, where_clause: str, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """获取热门模型"""
    query = f"""
        SELECT 
            al.model,
            COUNT(*) as request_count,
            SUM(tu.total_tokens) as total_tokens,
            SUM(ci.total_cost) as total_cost,
            AVG(al.duration_ms) as avg_duration
        FROM api_logs al
        LEFT JOIN token_usage tu ON al.id = tu.log_id
        LEFT JOIN cost_info ci ON al.id = ci.log_id
        {where_clause}
        GROUP BY al.model
        ORDER BY request_count DESC
        LIMIT 10
    """
    
    result = await db_manager.execute_query(query, query_params)
    return [
        {
            'model': row['model'],
            'request_count': row['request_count'],
            'total_tokens': row['total_tokens'] or 0,
            'total_cost': row['total_cost'] or 0,
            'avg_duration': row['avg_duration'] or 0
        }
        for row in result
    ]


async def _get_top_providers(db_manager: DatabaseManager, where_clause: str, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """获取热门提供商"""
    query = f"""
        SELECT 
            al.provider,
            COUNT(*) as request_count,
            SUM(tu.total_tokens) as total_tokens,
            SUM(ci.total_cost) as total_cost,
            AVG(al.duration_ms) as avg_duration
        FROM api_logs al
        LEFT JOIN token_usage tu ON al.id = tu.log_id
        LEFT JOIN cost_info ci ON al.id = ci.log_id
        {where_clause}
        AND al.provider IS NOT NULL
        GROUP BY al.provider
        ORDER BY request_count DESC
        LIMIT 10
    """
    
    result = await db_manager.execute_query(query, query_params)
    return [
        {
            'provider': row['provider'],
            'request_count': row['request_count'],
            'total_tokens': row['total_tokens'] or 0,
            'total_cost': row['total_cost'] or 0,
            'avg_duration': row['avg_duration'] or 0
        }
        for row in result
    ]


# 趋势分析辅助函数
async def _get_daily_token_usage(db_manager: DatabaseManager, where_clause: str, query_params: Dict[str, Any], granularity: TimeGranularity) -> List[Dict[str, Any]]:
    """获取每日Token使用量"""
    time_format = _get_time_format(granularity)
    
    query = f"""
        SELECT 
            DATE_TRUNC('{granularity.value}', al.created_at) as period,
            SUM(tu.total_tokens) as tokens
        FROM api_logs al
        LEFT JOIN token_usage tu ON al.id = tu.log_id
        {where_clause}
        GROUP BY period
        ORDER BY period
    """
    
    result = await db_manager.execute_query(query, query_params)
    return [
        {
            'period': row['period'],
            'tokens': row['tokens'] or 0
        }
        for row in result
    ]


async def _get_daily_cost(db_manager: DatabaseManager, where_clause: str, query_params: Dict[str, Any], granularity: TimeGranularity) -> List[Dict[str, Any]]:
    """获取每日成本"""
    query = f"""
        SELECT 
            DATE_TRUNC('{granularity.value}', al.created_at) as period,
            SUM(ci.total_cost) as cost
        FROM api_logs al
        LEFT JOIN cost_info ci ON al.id = ci.log_id
        {where_clause}
        GROUP BY period
        ORDER BY period
    """
    
    result = await db_manager.execute_query(query, query_params)
    return [
        {
            'period': row['period'],
            'cost': row['cost'] or 0
        }
        for row in result
    ]


async def _get_peak_token_usage(db_manager: DatabaseManager, where_clause: str, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """获取峰值Token使用时间"""
    query = f"""
        SELECT 
            DATE_TRUNC('hour', al.created_at) as hour,
            SUM(tu.total_tokens) as tokens
        FROM api_logs al
        LEFT JOIN token_usage tu ON al.id = tu.log_id
        {where_clause}
        GROUP BY hour
        ORDER BY tokens DESC
        LIMIT 1
    """
    
    result = await db_manager.execute_query(query, query_params)
    if result:
        return {
            'hour': result[0]['hour'],
            'tokens': result[0]['tokens'] or 0
        }
    return None


async def _get_token_trends(db_manager: DatabaseManager, where_clause: str, query_params: Dict[str, Any], granularity: TimeGranularity) -> List[Dict[str, Any]]:
    """获取Token使用趋势"""
    return await _get_daily_token_usage(db_manager, where_clause, query_params, granularity)


async def _get_cost_trends(db_manager: DatabaseManager, where_clause: str, query_params: Dict[str, Any], granularity: TimeGranularity) -> List[Dict[str, Any]]:
    """获取成本趋势"""
    return await _get_daily_cost(db_manager, where_clause, query_params, granularity)


async def _get_performance_trends(db_manager: DatabaseManager, where_clause: str, query_params: Dict[str, Any], granularity: TimeGranularity) -> List[Dict[str, Any]]:
    """获取性能趋势"""
    query = f"""
        SELECT 
            DATE_TRUNC('{granularity.value}', al.created_at) as period,
            AVG(al.duration_ms) as avg_duration,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY al.duration_ms) as p95_duration
        FROM api_logs al
        {where_clause}
        GROUP BY period
        ORDER BY period
    """
    
    result = await db_manager.execute_query(query, query_params)
    return [
        {
            'period': row['period'],
            'avg_duration': row['avg_duration'] or 0,
            'p95_duration': row['p95_duration'] or 0
        }
        for row in result
    ]


async def _get_error_trends(db_manager: DatabaseManager, where_clause: str, query_params: Dict[str, Any], granularity: TimeGranularity) -> List[Dict[str, Any]]:
    """获取错误趋势"""
    query = f"""
        SELECT 
            DATE_TRUNC('{granularity.value}', al.created_at) as period,
            COUNT(*) as total_requests,
            COUNT(CASE WHEN al.status != 'success' THEN 1 END) as error_count
        FROM api_logs al
        {where_clause}
        GROUP BY period
        ORDER BY period
    """
    
    result = await db_manager.execute_query(query, query_params)
    return [
        {
            'period': row['period'],
            'total_requests': row['total_requests'],
            'error_count': row['error_count'],
            'error_rate': row['error_count'] / row['total_requests'] if row['total_requests'] > 0 else 0
        }
        for row in result
    ]


async def _get_request_count_trends(db_manager: DatabaseManager, where_clause: str, query_params: Dict[str, Any], granularity: TimeGranularity) -> List[Dict[str, Any]]:
    """获取请求数量趋势"""
    query = f"""
        SELECT 
            DATE_TRUNC('{granularity.value}', al.created_at) as period,
            COUNT(*) as request_count
        FROM api_logs al
        {where_clause}
        GROUP BY period
        ORDER BY period
    """
    
    result = await db_manager.execute_query(query, query_params)
    return [
        {
            'period': row['period'],
            'request_count': row['request_count']
        }
        for row in result
    ]


def _get_time_format(granularity: TimeGranularity) -> str:
    """获取时间格式"""
    formats = {
        TimeGranularity.HOUR: 'YYYY-MM-DD HH24:00:00',
        TimeGranularity.DAY: 'YYYY-MM-DD',
        TimeGranularity.WEEK: 'YYYY-"W"WW',
        TimeGranularity.MONTH: 'YYYY-MM',
        TimeGranularity.YEAR: 'YYYY'
    }
    return formats.get(granularity, 'YYYY-MM-DD')