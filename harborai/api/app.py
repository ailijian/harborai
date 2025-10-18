#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI API 主应用

提供统一的API接口，包括：
- 聊天完成接口（兼容OpenAI API）
- 模型列表接口
- 健康检查接口
- 日志管理接口
- 追踪查询接口
- 统计分析接口
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import uvicorn

from ..utils.logger import get_logger
from ..core.chat_completion import ChatCompletionService
from ..core.model_manager import ModelManager
from .routes import logs_router, tracing_router, statistics_router, health_router
from .schemas import StandardResponse, ErrorResponse, ResponseBuilder

logger = get_logger("harborai.api")

# 创建FastAPI应用
app = FastAPI(
    title="HarborAI API",
    description="HarborAI统一API接口 - 提供聊天完成、日志管理、追踪查询和统计分析功能",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(description="服务状态")
    timestamp: datetime = Field(description="检查时间")
    version: str = Field(description="版本号")


class ChatCompletionRequest(BaseModel):
    """聊天完成请求"""
    model: str = Field(description="模型名称")
    messages: List[Dict[str, Any]] = Field(description="消息列表")
    temperature: Optional[float] = Field(default=0.7, description="温度参数")
    max_tokens: Optional[int] = Field(default=None, description="最大token数")
    stream: Optional[bool] = Field(default=False, description="是否流式响应")
    top_p: Optional[float] = Field(default=1.0, description="top_p参数")
    frequency_penalty: Optional[float] = Field(default=0.0, description="频率惩罚")
    presence_penalty: Optional[float] = Field(default=0.0, description="存在惩罚")
    stop: Optional[Union[str, List[str]]] = Field(default=None, description="停止词")
    trace_id: Optional[str] = Field(default=None, description="追踪ID")
    user_id: Optional[str] = Field(default=None, description="用户ID")


class ChatCompletionResponse(BaseModel):
    """聊天完成响应"""
    id: str = Field(description="响应ID")
    object: str = Field(description="对象类型")
    created: int = Field(description="创建时间戳")
    model: str = Field(description="使用的模型")
    choices: List[Dict[str, Any]] = Field(description="选择列表")
    usage: Dict[str, int] = Field(description="使用统计")
    trace_id: Optional[str] = Field(description="追踪ID")


class ModelInfo(BaseModel):
    """模型信息"""
    id: str = Field(description="模型ID")
    object: str = Field(description="对象类型")
    created: int = Field(description="创建时间戳")
    owned_by: str = Field(description="拥有者")
    capabilities: List[str] = Field(description="模型能力")
    context_length: int = Field(description="上下文长度")


class ModelsResponse(BaseModel):
    """模型列表响应"""
    object: str = Field(description="对象类型")
    data: List[ModelInfo] = Field(description="模型列表")


# 注册路由
app.include_router(logs_router)
app.include_router(tracing_router)
app.include_router(statistics_router)
app.include_router(health_router)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP异常处理器"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            timestamp=datetime.now()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """通用异常处理器"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error_code="INTERNAL_SERVER_ERROR",
            message="服务器内部错误",
            timestamp=datetime.now()
        ).dict()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """基础健康检查接口"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )


@app.get("/", response_model=StandardResponse[Dict[str, Any]])
async def root():
    """根路径接口"""
    return StandardResponse(
        success=True,
        data={
            "service": "HarborAI API",
            "version": "1.0.0",
            "description": "统一的AI服务API接口",
            "endpoints": {
                "docs": "/docs",
                "health": "/health",
                "chat": "/v1/chat/completions",
                "models": "/v1/models",
                "logs": "/v1/logs",
                "tracing": "/v1/tracing",
                "statistics": "/v1/statistics"
            }
        },
        message="欢迎使用HarborAI API",
        timestamp=datetime.now()
    )


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    创建聊天完成
    
    兼容OpenAI API的聊天完成接口，支持追踪和日志记录
    """
    try:
        # 初始化聊天完成服务
        chat_service = ChatCompletionService()
        
        # 处理聊天完成请求
        response = await chat_service.create_completion(
            model=request.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
            top_p=request.top_p,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            stop=request.stop,
            trace_id=request.trace_id,
            user_id=request.user_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"聊天完成请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """
    获取可用模型列表
    
    返回所有可用的模型信息，包括模型能力和配置
    """
    try:
        # 初始化模型管理器
        model_manager = ModelManager()
        
        # 获取模型列表
        models = await model_manager.list_available_models()
        
        # 转换为API响应格式
        model_data = []
        for model in models:
            model_data.append(ModelInfo(
                id=model.get("id", ""),
                object="model",
                created=int(datetime.now().timestamp()),
                owned_by=model.get("provider", "harborai"),
                capabilities=model.get("capabilities", ["chat"]),
                context_length=model.get("context_length", 4096)
            ))
        
        return ModelsResponse(
            object="list",
            data=model_data
        )
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/info", response_model=StandardResponse[Dict[str, Any]])
async def get_api_info():
    """
    获取API信息
    
    返回API的详细信息和配置
    """
    try:
        return StandardResponse(
            success=True,
            data={
                "api_version": "1.0.0",
                "service_name": "HarborAI",
                "supported_features": [
                    "chat_completions",
                    "model_management",
                    "request_logging",
                    "distributed_tracing",
                    "usage_statistics",
                    "health_monitoring"
                ],
                "openai_compatibility": True,
                "rate_limits": {
                    "requests_per_minute": 1000,
                    "tokens_per_minute": 100000
                },
                "supported_models": "Use /v1/models to get the list",
                "documentation": "/docs"
            },
            message="API信息获取成功",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"获取API信息失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_app() -> FastAPI:
    """创建并配置FastAPI应用"""
    return app


if __name__ == "__main__":
    uvicorn.run(
        "harborai.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )