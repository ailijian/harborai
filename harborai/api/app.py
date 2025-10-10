#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI FastAPI 应用

提供 RESTful API 接口，兼容 OpenAI API 格式。
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

from ..config.settings import get_settings
from ..utils.logger import get_logger

logger = get_logger("harborai.api")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    timestamp: str


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """聊天请求"""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    settings = get_settings()
    
    app = FastAPI(
        title="HarborAI API",
        description="高性能AI API代理和管理平台",
        version="1.0.0-beta.6",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # 添加 CORS 中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """健康检查端点"""
        from datetime import datetime
        return HealthResponse(
            status="healthy",
            version="1.0.0-beta.6",
            timestamp=datetime.now().isoformat()
        )
    
    @app.get("/")
    async def root():
        """根端点"""
        return {
            "message": "Welcome to HarborAI API",
            "version": "1.0.0-beta.6",
            "docs": "/docs"
        }
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        """聊天完成端点 (兼容 OpenAI API)"""
        try:
            # 这里应该调用实际的 HarborAI 客户端
            # 目前返回一个简单的响应
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1234567890,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello! This is a test response from HarborAI."
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20
                }
            }
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/v1/models")
    async def list_models():
        """列出可用模型"""
        return {
            "object": "list",
            "data": [
                {
                    "id": "gpt-3.5-turbo",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "harborai"
                },
                {
                    "id": "gpt-4",
                    "object": "model", 
                    "created": 1234567890,
                    "owned_by": "harborai"
                }
            ]
        }
    
    return app


if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)