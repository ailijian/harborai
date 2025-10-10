"""
统一HTTP客户端配置模块

提供优化的HTTP客户端配置，用于所有LLM插件，确保最佳的流式传输性能。
"""

from typing import Dict, Any, Optional
import httpx


class OptimizedHTTPConfig:
    """
    优化的HTTP客户端配置类
    
    统一管理所有LLM插件的HTTP客户端配置，确保：
    - 最佳的流式传输性能
    - 统一的连接池配置
    - 优化的TTFB（首字节时间）
    - 避免HTTP/2相关的流式问题
    """
    
    # 默认配置常量
    DEFAULT_TIMEOUT = 60
    DEFAULT_MAX_CONNECTIONS = 100
    DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 20
    DEFAULT_MAX_RETRIES = 3
    
    @classmethod
    def get_sync_client_config(
        cls,
        base_url: str,
        api_key: str,
        timeout: Optional[int] = None,
        max_connections: Optional[int] = None,
        max_keepalive_connections: Optional[int] = None,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        获取同步HTTP客户端配置
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            timeout: 超时时间（秒）
            max_connections: 最大连接数
            max_keepalive_connections: 最大保持连接数
            additional_headers: 额外的请求头
            
        Returns:
            Dict[str, Any]: httpx.Client的配置参数
        """
        # 基础请求头
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream, application/json"
        }
        
        # 添加额外请求头
        if additional_headers:
            headers.update(additional_headers)
        
        return {
            "base_url": base_url,
            "timeout": timeout or cls.DEFAULT_TIMEOUT,
            "headers": headers,
            "follow_redirects": True,
            # 优化流式传输的连接池配置
            "limits": httpx.Limits(
                max_connections=max_connections or cls.DEFAULT_MAX_CONNECTIONS,
                max_keepalive_connections=max_keepalive_connections or cls.DEFAULT_MAX_KEEPALIVE_CONNECTIONS
            ),
            # 禁用HTTP/2以避免潜在的流式问题
            "http2": False
        }
    
    @classmethod
    def get_async_client_config(
        cls,
        base_url: str,
        api_key: str,
        timeout: Optional[int] = None,
        max_connections: Optional[int] = None,
        max_keepalive_connections: Optional[int] = None,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        获取异步HTTP客户端配置
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            timeout: 超时时间（秒）
            max_connections: 最大连接数
            max_keepalive_connections: 最大保持连接数
            additional_headers: 额外的请求头
            
        Returns:
            Dict[str, Any]: httpx.AsyncClient的配置参数
        """
        # 异步客户端配置与同步客户端相同
        return cls.get_sync_client_config(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            additional_headers=additional_headers
        )
    
    @classmethod
    def get_stream_headers(cls) -> Dict[str, str]:
        """
        获取流式传输专用请求头
        
        Returns:
            Dict[str, str]: 流式传输优化的请求头
        """
        return {
            "Accept": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # 禁用Nginx缓冲，提升TTFB性能
        }
    
    @classmethod
    def create_sync_client(
        cls,
        base_url: str,
        api_key: str,
        timeout: Optional[int] = None,
        max_connections: Optional[int] = None,
        max_keepalive_connections: Optional[int] = None,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> httpx.Client:
        """
        创建优化的同步HTTP客户端
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            timeout: 超时时间（秒）
            max_connections: 最大连接数
            max_keepalive_connections: 最大保持连接数
            additional_headers: 额外的请求头
            
        Returns:
            httpx.Client: 配置优化的同步HTTP客户端
        """
        config = cls.get_sync_client_config(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            additional_headers=additional_headers
        )
        return httpx.Client(**config)
    
    @classmethod
    def create_async_client(
        cls,
        base_url: str,
        api_key: str,
        timeout: Optional[int] = None,
        max_connections: Optional[int] = None,
        max_keepalive_connections: Optional[int] = None,
        additional_headers: Optional[Dict[str, str]] = None
    ) -> httpx.AsyncClient:
        """
        创建优化的异步HTTP客户端
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            timeout: 超时时间（秒）
            max_connections: 最大连接数
            max_keepalive_connections: 最大保持连接数
            additional_headers: 额外的请求头
            
        Returns:
            httpx.AsyncClient: 配置优化的异步HTTP客户端
        """
        config = cls.get_async_client_config(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_connections=max_connections,
            max_keepalive_connections=max_keepalive_connections,
            additional_headers=additional_headers
        )
        return httpx.AsyncClient(**config)


class StreamingOptimizer:
    """
    流式传输优化器
    
    提供流式传输相关的优化配置和工具方法。
    """
    
    @staticmethod
    def get_optimized_stream_config() -> Dict[str, Any]:
        """
        获取优化的流式传输配置
        
        Returns:
            Dict[str, Any]: 流式传输优化配置
        """
        return {
            "chunk_size": 1024,  # 优化的chunk大小
            "decode_unicode": True,  # 启用Unicode解码
            "follow_redirects": True,  # 跟随重定向
        }
    
    @staticmethod
    def should_use_stream_method(provider: str) -> bool:
        """
        判断是否应该使用client.stream()方法
        
        Args:
            provider: 提供商名称
            
        Returns:
            bool: 是否使用stream方法
        """
        # 对于支持的提供商，推荐使用stream方法以获得更好的性能
        supported_providers = {"deepseek", "wenxin", "doubao"}
        return provider.lower() in supported_providers