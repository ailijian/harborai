#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API响应模式定义

定义标准化的API响应格式，包括：
- 标准响应格式
- 分页响应格式
- 错误响应格式
- 元数据结构
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Generic, TypeVar, Union
from datetime import datetime
from enum import Enum

# 泛型类型变量
T = TypeVar('T')


class ResponseStatus(str, Enum):
    """响应状态枚举"""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ErrorCode(str, Enum):
    """错误代码枚举"""
    # 通用错误
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    
    # 数据库错误
    DATABASE_ERROR = "DATABASE_ERROR"
    DATABASE_CONNECTION_ERROR = "DATABASE_CONNECTION_ERROR"
    DATABASE_TIMEOUT = "DATABASE_TIMEOUT"
    
    # 业务逻辑错误
    VALIDATION_ERROR = "VALIDATION_ERROR"
    BUSINESS_RULE_VIOLATION = "BUSINESS_RULE_VIOLATION"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    
    # 外部服务错误
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"
    EXTERNAL_SERVICE_TIMEOUT = "EXTERNAL_SERVICE_TIMEOUT"
    EXTERNAL_SERVICE_UNAVAILABLE = "EXTERNAL_SERVICE_UNAVAILABLE"


class MetaData(BaseModel):
    """元数据结构"""
    request_id: Optional[str] = Field(None, description="请求ID")
    trace_id: Optional[str] = Field(None, description="追踪ID")
    span_id: Optional[str] = Field(None, description="跨度ID")
    user_id: Optional[str] = Field(None, description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    api_version: Optional[str] = Field(None, description="API版本")
    server_time: Optional[datetime] = Field(None, description="服务器时间")
    processing_time_ms: Optional[float] = Field(None, description="处理时间(毫秒)")
    rate_limit: Optional[Dict[str, Any]] = Field(None, description="限流信息")
    cache_info: Optional[Dict[str, Any]] = Field(None, description="缓存信息")


class PaginationInfo(BaseModel):
    """分页信息"""
    page: int = Field(..., ge=1, description="当前页码")
    page_size: int = Field(..., ge=1, le=1000, description="每页大小")
    total_count: int = Field(..., ge=0, description="总记录数")
    total_pages: int = Field(..., ge=0, description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")
    next_page: Optional[int] = Field(None, description="下一页页码")
    prev_page: Optional[int] = Field(None, description="上一页页码")


class ErrorDetail(BaseModel):
    """错误详情"""
    code: ErrorCode = Field(..., description="错误代码")
    message: str = Field(..., description="错误消息")
    field: Optional[str] = Field(None, description="错误字段")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    suggestion: Optional[str] = Field(None, description="解决建议")


class StandardResponse(BaseModel, Generic[T]):
    """标准响应格式"""
    success: bool = Field(..., description="请求是否成功")
    status: ResponseStatus = Field(ResponseStatus.SUCCESS, description="响应状态")
    data: Optional[T] = Field(None, description="响应数据")
    message: str = Field("", description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    meta: Optional[MetaData] = Field(None, description="元数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginatedResponse(BaseModel, Generic[T]):
    """分页响应格式"""
    success: bool = Field(..., description="请求是否成功")
    status: ResponseStatus = Field(ResponseStatus.SUCCESS, description="响应状态")
    data: List[T] = Field(..., description="响应数据列表")
    pagination: PaginationInfo = Field(..., description="分页信息")
    message: str = Field("", description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    meta: Optional[MetaData] = Field(None, description="元数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """错误响应格式"""
    success: bool = Field(False, description="请求是否成功")
    status: ResponseStatus = Field(ResponseStatus.ERROR, description="响应状态")
    error: ErrorDetail = Field(..., description="错误信息")
    message: str = Field("", description="错误消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    meta: Optional[MetaData] = Field(None, description="元数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BatchResponse(BaseModel, Generic[T]):
    """批量操作响应格式"""
    success: bool = Field(..., description="批量操作是否成功")
    status: ResponseStatus = Field(ResponseStatus.SUCCESS, description="响应状态")
    total_count: int = Field(..., ge=0, description="总操作数")
    success_count: int = Field(..., ge=0, description="成功操作数")
    failure_count: int = Field(..., ge=0, description="失败操作数")
    results: List[Union[T, ErrorDetail]] = Field(..., description="操作结果列表")
    message: str = Field("", description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    meta: Optional[MetaData] = Field(None, description="元数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """健康检查响应格式"""
    success: bool = Field(..., description="健康检查是否通过")
    status: ResponseStatus = Field(..., description="健康状态")
    service: str = Field(..., description="服务名称")
    version: str = Field(..., description="服务版本")
    uptime_seconds: float = Field(..., description="运行时间(秒)")
    checks: Dict[str, Dict[str, Any]] = Field(..., description="各项检查结果")
    timestamp: datetime = Field(default_factory=datetime.now, description="检查时间戳")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StatisticsResponse(BaseModel):
    """统计信息响应格式"""
    success: bool = Field(..., description="请求是否成功")
    status: ResponseStatus = Field(ResponseStatus.SUCCESS, description="响应状态")
    period: Dict[str, datetime] = Field(..., description="统计周期")
    metrics: Dict[str, Any] = Field(..., description="统计指标")
    aggregations: Dict[str, Any] = Field(..., description="聚合数据")
    trends: Optional[Dict[str, Any]] = Field(None, description="趋势分析")
    message: str = Field("", description="响应消息")
    timestamp: datetime = Field(default_factory=datetime.now, description="响应时间戳")
    meta: Optional[MetaData] = Field(None, description="元数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


# 响应构建器类
class ResponseBuilder:
    """响应构建器
    
    提供便捷的方法来构建标准化的API响应。
    """
    
    @staticmethod
    def success(
        data: Any = None,
        message: str = "操作成功",
        meta: Optional[MetaData] = None
    ) -> StandardResponse:
        """构建成功响应"""
        return StandardResponse(
            success=True,
            status=ResponseStatus.SUCCESS,
            data=data,
            message=message,
            timestamp=datetime.now(),
            meta=meta
        )
    
    @staticmethod
    def error(
        error_code: ErrorCode,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        meta: Optional[MetaData] = None
    ) -> ErrorResponse:
        """构建错误响应"""
        error_detail = ErrorDetail(
            code=error_code,
            message=message,
            field=field,
            details=details,
            suggestion=suggestion
        )
        
        return ErrorResponse(
            success=False,
            status=ResponseStatus.ERROR,
            error=error_detail,
            message=message,
            timestamp=datetime.now(),
            meta=meta
        )
    
    @staticmethod
    def paginated(
        data: List[Any],
        page: int,
        page_size: int,
        total_count: int,
        message: str = "查询成功",
        meta: Optional[MetaData] = None
    ) -> PaginatedResponse:
        """构建分页响应"""
        total_pages = (total_count + page_size - 1) // page_size
        
        pagination = PaginationInfo(
            page=page,
            page_size=page_size,
            total_count=total_count,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
            next_page=page + 1 if page < total_pages else None,
            prev_page=page - 1 if page > 1 else None
        )
        
        return PaginatedResponse(
            success=True,
            status=ResponseStatus.SUCCESS,
            data=data,
            pagination=pagination,
            message=message,
            timestamp=datetime.now(),
            meta=meta
        )
    
    @staticmethod
    def batch(
        results: List[Union[Any, ErrorDetail]],
        message: str = "批量操作完成",
        meta: Optional[MetaData] = None
    ) -> BatchResponse:
        """构建批量操作响应"""
        total_count = len(results)
        success_count = len([r for r in results if not isinstance(r, ErrorDetail)])
        failure_count = total_count - success_count
        
        return BatchResponse(
            success=failure_count == 0,
            status=ResponseStatus.SUCCESS if failure_count == 0 else ResponseStatus.WARNING,
            total_count=total_count,
            success_count=success_count,
            failure_count=failure_count,
            results=results,
            message=message,
            timestamp=datetime.now(),
            meta=meta
        )
    
    @staticmethod
    def health_check(
        service: str,
        version: str,
        uptime_seconds: float,
        checks: Dict[str, Dict[str, Any]],
        overall_healthy: bool = True
    ) -> HealthCheckResponse:
        """构建健康检查响应"""
        return HealthCheckResponse(
            success=overall_healthy,
            status=ResponseStatus.SUCCESS if overall_healthy else ResponseStatus.ERROR,
            service=service,
            version=version,
            uptime_seconds=uptime_seconds,
            checks=checks,
            timestamp=datetime.now()
        )
    
    @staticmethod
    def statistics(
        period: Dict[str, datetime],
        metrics: Dict[str, Any],
        aggregations: Dict[str, Any],
        trends: Optional[Dict[str, Any]] = None,
        message: str = "统计信息获取成功",
        meta: Optional[MetaData] = None
    ) -> StatisticsResponse:
        """构建统计信息响应"""
        return StatisticsResponse(
            success=True,
            status=ResponseStatus.SUCCESS,
            period=period,
            metrics=metrics,
            aggregations=aggregations,
            trends=trends,
            message=message,
            timestamp=datetime.now(),
            meta=meta
        )


# 元数据构建器
class MetaDataBuilder:
    """元数据构建器"""
    
    @staticmethod
    def create(
        request_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        api_version: str = "v1",
        processing_time_ms: Optional[float] = None,
        rate_limit: Optional[Dict[str, Any]] = None,
        cache_info: Optional[Dict[str, Any]] = None
    ) -> MetaData:
        """创建元数据"""
        return MetaData(
            request_id=request_id,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            session_id=session_id,
            api_version=api_version,
            server_time=datetime.now(),
            processing_time_ms=processing_time_ms,
            rate_limit=rate_limit,
            cache_info=cache_info
        )


# 常用错误响应预定义
class CommonErrors:
    """常用错误响应预定义"""
    
    @staticmethod
    def internal_error(message: str = "内部服务器错误") -> ErrorResponse:
        return ResponseBuilder.error(
            ErrorCode.INTERNAL_ERROR,
            message,
            suggestion="请稍后重试，如果问题持续存在请联系技术支持"
        )
    
    @staticmethod
    def invalid_request(message: str = "请求参数无效", field: Optional[str] = None) -> ErrorResponse:
        return ResponseBuilder.error(
            ErrorCode.INVALID_REQUEST,
            message,
            field=field,
            suggestion="请检查请求参数格式和内容"
        )
    
    @staticmethod
    def not_found(resource: str = "资源") -> ErrorResponse:
        return ResponseBuilder.error(
            ErrorCode.NOT_FOUND,
            f"{resource}不存在",
            suggestion="请检查资源ID是否正确"
        )
    
    @staticmethod
    def unauthorized(message: str = "未授权访问") -> ErrorResponse:
        return ResponseBuilder.error(
            ErrorCode.UNAUTHORIZED,
            message,
            suggestion="请提供有效的认证信息"
        )
    
    @staticmethod
    def forbidden(message: str = "访问被禁止") -> ErrorResponse:
        return ResponseBuilder.error(
            ErrorCode.FORBIDDEN,
            message,
            suggestion="您没有执行此操作的权限"
        )
    
    @staticmethod
    def rate_limited(message: str = "请求频率超限") -> ErrorResponse:
        return ResponseBuilder.error(
            ErrorCode.RATE_LIMITED,
            message,
            suggestion="请降低请求频率后重试"
        )
    
    @staticmethod
    def database_error(message: str = "数据库操作失败") -> ErrorResponse:
        return ResponseBuilder.error(
            ErrorCode.DATABASE_ERROR,
            message,
            suggestion="请稍后重试，如果问题持续存在请联系技术支持"
        )
    
    @staticmethod
    def validation_error(message: str = "数据验证失败", field: Optional[str] = None) -> ErrorResponse:
        return ResponseBuilder.error(
            ErrorCode.VALIDATION_ERROR,
            message,
            field=field,
            suggestion="请检查输入数据的格式和内容"
        )