#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速结构化输出处理器
专为FAST模式设计的轻量级结构化输出处理器，最大化性能优化
"""

import json
import time
import threading
from typing import Any, Dict, Optional, Union, Tuple
from dataclasses import dataclass

from .unified_decorators import fast_trace
from .agently_client_pool import get_agently_client_pool, create_agently_client_config
from .parameter_cache import get_parameter_cache_manager
from ..utils.logger import get_logger
from ..utils.exceptions import StructuredOutputError

logger = get_logger(__name__)


@dataclass
class FastProcessingConfig:
    """快速处理配置"""
    enable_schema_cache: bool = True
    enable_client_pool: bool = True
    enable_config_cache: bool = True
    skip_validation: bool = True  # FAST模式跳过详细验证
    max_retry_attempts: int = 1  # 减少重试次数
    timeout_seconds: float = 10.0  # 更短的超时时间
    use_lightweight_parsing: bool = True  # 使用轻量级解析


class FastStructuredOutputProcessor:
    """快速结构化输出处理器
    
    专为FAST模式设计，通过以下优化策略提升性能：
    1. 客户端池复用 - 避免重复创建Agently客户端
    2. Schema转换缓存 - 缓存JSON Schema到Agently格式的转换结果
    3. 配置参数缓存 - 缓存处理过的配置参数
    4. 轻量级解析 - 跳过非必要的验证和中间件
    5. 快速路径 - 为常见场景提供优化的执行路径
    """
    
    def __init__(self, config: Optional[FastProcessingConfig] = None, client_manager=None):
        """初始化快速结构化输出处理器
        
        Args:
            config: 快速处理配置，如果为None则使用默认配置
            client_manager: 客户端管理器实例
        """
        self.config = config or FastProcessingConfig()
        self.logger = get_logger(__name__)  # 添加logger属性
        self.client_manager = client_manager  # 添加client_manager属性
        self._client_pool = get_agently_client_pool() if self.config.enable_client_pool else None
        self._cache_manager = get_parameter_cache_manager() if (
            self.config.enable_schema_cache or self.config.enable_config_cache
        ) else None
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time': 0.0,
            'client_pool_hits': 0,
            'fast_path_usage': 0
        }
        self._stats_lock = threading.Lock()
    
    @fast_trace
    def process_structured_output(
        self,
        user_query: str,
        schema: Dict[str, Any],
        api_key: str,
        base_url: str,
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """处理结构化输出请求
        
        Args:
            user_query: 用户查询
            schema: JSON Schema定义
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            结构化输出结果
        """
        start_time = time.time()
        
        try:
            # 更新统计信息
            with self._stats_lock:
                self._stats['total_requests'] += 1
            
            # 快速路径检查 - 优先使用缓存的配置和Schema
            if self._can_use_fast_path(schema, api_key, base_url, model):
                result = self._process_fast_path(user_query, schema, api_key, base_url, model, **kwargs)
                with self._stats_lock:
                    self._stats['fast_path_usage'] += 1
                return result
            
            # 标准路径处理
            return self._process_standard_path(user_query, schema, api_key, base_url, model, **kwargs)
            
        except Exception as e:
            logger.error(f"快速结构化输出处理失败: {e}")
            raise StructuredOutputError(f"快速处理失败: {e}")
        finally:
            # 更新平均处理时间
            processing_time = time.time() - start_time
            with self._stats_lock:
                if self._stats['avg_processing_time'] == 0.0:
                    self._stats['avg_processing_time'] = processing_time
                else:
                    # 使用指数移动平均
                    alpha = 0.1
                    self._stats['avg_processing_time'] = (
                        alpha * processing_time + 
                        (1 - alpha) * self._stats['avg_processing_time']
                    )
    
    def _can_use_fast_path(
        self, 
        schema: Dict[str, Any], 
        api_key: str, 
        base_url: str, 
        model: str
    ) -> bool:
        """检查是否可以使用快速路径
        
        Args:
            schema: JSON Schema定义
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            
        Returns:
            是否可以使用快速路径
        """
        if not self.config.enable_schema_cache or not self.config.enable_config_cache:
            return False
        
        if not self._cache_manager:
            return False
        
        # 检查Schema缓存
        cached_schema = self._cache_manager.schema_cache.get_converted_schema(schema)
        if cached_schema is None:
            return False
        
        # 检查配置缓存
        config_data = {
            'api_key_hash': hash(api_key) if api_key else None,
            'base_url': base_url,
            'model': model
        }
        cached_config = self._cache_manager.config_cache.get_config(config_data)
        if cached_config is None:
            return False
        
        return True
    
    @fast_trace
    def _process_fast_path(
        self,
        user_query: str,
        schema: Dict[str, Any],
        api_key: str,
        base_url: str,
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """快速路径处理
        
        使用缓存的Schema转换和配置，最小化处理开销
        
        Args:
            user_query: 用户查询
            schema: JSON Schema定义
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            结构化输出结果
        """
        # 从缓存获取转换后的Schema
        agently_schema = self._cache_manager.schema_cache.get_converted_schema(schema)
        if agently_schema is None:
            raise StructuredOutputError("缓存中未找到转换后的Schema")
        
        # 从缓存获取配置
        config_data = {
            'api_key_hash': hash(api_key) if api_key else None,
            'base_url': base_url,
            'model': model
        }
        cached_config = self._cache_manager.config_cache.get_config(config_data)
        if cached_config is None:
            raise StructuredOutputError("缓存中未找到配置信息")
        
        # 更新缓存命中统计
        with self._stats_lock:
            self._stats['cache_hits'] += 2  # Schema和配置都命中
        
        # 使用客户端池获取Agently客户端
        if self._client_pool:
            # 获取模型对应的provider
            try:
                # 通过client_manager获取模型对应的插件，从而获取provider
                if self.client_manager is None:
                    raise ValueError("client_manager未初始化")
                plugin = self.client_manager.get_plugin_for_model(model)
                if plugin is None:
                    raise ValueError(f"未找到模型{model}对应的插件")
                provider = plugin.name
            except Exception as e:
                self.logger.warning(f"无法获取模型{model}的provider，使用默认值: {str(e)}")
                provider = "unknown"
            
            client_config = create_agently_client_config(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            
            with self._client_pool.get_client_context(client_config) as agently_client:
                with self._stats_lock:
                    self._stats['client_pool_hits'] += 1
                
                # 执行快速结构化输出
                result = self._execute_agently_request(
                    agently_client, 
                    user_query, 
                    agently_schema,
                    lightweight=True
                )
                
                return result
        else:
            # 回退到标准处理
            return self._process_standard_path(user_query, schema, api_key, base_url, model, **kwargs)
    
    @fast_trace
    def _process_standard_path(
        self,
        user_query: str,
        schema: Dict[str, Any],
        api_key: str,
        base_url: str,
        model: str,
        **kwargs
    ) -> Dict[str, Any]:
        """标准路径处理
        
        当快速路径不可用时的标准处理流程
        
        Args:
            user_query: 用户查询
            schema: JSON Schema定义
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            **kwargs: 其他参数
            
        Returns:
            结构化输出结果
        """
        # 转换Schema格式
        agently_schema = self._convert_schema_with_cache(schema)
        
        # 处理配置
        processed_config = self._process_config_with_cache(api_key, base_url, model)
        
        # 使用客户端池或创建新客户端
        if self._client_pool:
            # 获取模型对应的provider
            try:
                if self.client_manager is None:
                    raise ValueError("client_manager未初始化")
                plugin = self.client_manager.get_plugin_for_model(model)
                if plugin is None:
                    raise ValueError(f"未找到模型{model}对应的插件")
                provider = plugin.name
            except Exception as e:
                self.logger.warning(f"无法获取模型{model}的provider，使用默认值: {str(e)}")
                provider = "unknown"
            
            client_config = create_agently_client_config(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            
            with self._client_pool.get_client_context(client_config) as agently_client:
                with self._stats_lock:
                    self._stats['client_pool_hits'] += 1
                
                result = self._execute_agently_request(
                    agently_client, 
                    user_query, 
                    agently_schema,
                    lightweight=self.config.use_lightweight_parsing
                )
                
                return result
        else:
            # 直接创建客户端（性能较低的回退方案）
            return self._create_and_execute(user_query, agently_schema, api_key, base_url, model)
    
    @fast_trace
    def _convert_schema_with_cache(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """使用缓存转换Schema格式
        
        Args:
            schema: JSON Schema定义
            
        Returns:
            转换后的Agently格式Schema
        """
        if self.config.enable_schema_cache and self._cache_manager:
            # 尝试从缓存获取
            cached_result = self._cache_manager.schema_cache.get_converted_schema(schema)
            if cached_result is not None:
                with self._stats_lock:
                    self._stats['cache_hits'] += 1
                return cached_result
        
        # 缓存未命中，执行转换
        with self._stats_lock:
            self._stats['cache_misses'] += 1
        
        converted_schema = self._convert_json_schema_to_agently(schema)
        
        # 存储到缓存
        if self.config.enable_schema_cache and self._cache_manager:
            self._cache_manager.schema_cache.set_converted_schema(schema, converted_schema)
        
        return converted_schema
    
    @fast_trace
    def _process_config_with_cache(self, api_key: str, base_url: str, model: str) -> Dict[str, Any]:
        """使用缓存处理配置
        
        Args:
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            
        Returns:
            处理后的配置
        """
        config_data = {
            'api_key_hash': hash(api_key) if api_key else None,
            'base_url': base_url,
            'model': model
        }
        
        if self.config.enable_config_cache and self._cache_manager:
            # 尝试从缓存获取
            cached_config = self._cache_manager.config_cache.get_config(config_data)
            if cached_config is not None:
                with self._stats_lock:
                    self._stats['cache_hits'] += 1
                return cached_config
        
        # 缓存未命中，处理配置
        with self._stats_lock:
            self._stats['cache_misses'] += 1
        
        processed_config = {
            'api_key': api_key,
            'base_url': base_url,
            'model': model,
            'timeout': self.config.timeout_seconds,
            'max_retries': self.config.max_retry_attempts
        }
        
        # 存储到缓存
        if self.config.enable_config_cache and self._cache_manager:
            self._cache_manager.config_cache.set_config(config_data, processed_config)
        
        return processed_config
    
    def _convert_json_schema_to_agently(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """将JSON Schema转换为Agently格式
        
        轻量级转换实现，专注于性能
        
        Args:
            schema: JSON Schema定义
            
        Returns:
            Agently格式的Schema
        """
        if not isinstance(schema, dict):
            return {"result": ("str", "Generated result")}
        
        schema_type = schema.get("type", "object")
        
        if schema_type == "object":
            return self._convert_object_schema_fast(schema)
        elif schema_type == "array":
            return self._convert_array_schema_fast(schema)
        else:
            return self._convert_primitive_schema_fast(schema)
    
    def _convert_object_schema_fast(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """快速转换object类型Schema"""
        result = {}
        properties = schema.get("properties", {})
        
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            description = prop_schema.get("description", prop_name)
            
            # 简化的类型映射
            if prop_type in ("string", "str"):
                result[prop_name] = ("str", description)
            elif prop_type in ("integer", "number", "int", "float"):
                result[prop_name] = ("int", description)
            elif prop_type in ("boolean", "bool"):
                result[prop_name] = ("bool", description)
            elif prop_type == "object":
                result[prop_name] = self._convert_object_schema_fast(prop_schema)
            elif prop_type == "array":
                # 对于数组类型，直接返回数组格式
                result[prop_name] = self._convert_array_schema_fast(prop_schema)
            else:
                result[prop_name] = ("str", description)
        
        return result
    
    def _convert_array_schema_fast(self, schema: Dict[str, Any]) -> list:
        """快速转换array类型Schema"""
        items_schema = schema.get("items", {"type": "string"})
        description = schema.get("description", "Array item")
        
        if isinstance(items_schema, dict):
            item_type = items_schema.get("type", "string")
            
            if item_type == "object":
                return [self._convert_object_schema_fast(items_schema)]
            elif item_type in ("string", "str"):
                return [("str", description)]
            elif item_type in ("integer", "number", "int", "float"):
                return [("int", description)]
            elif item_type in ("boolean", "bool"):
                return [("bool", description)]
            else:
                return [("str", description)]
        else:
            return [("str", description)]
    
    def _convert_primitive_schema_fast(self, schema: Dict[str, Any]) -> tuple:
        """快速转换基本类型Schema"""
        schema_type = schema.get("type", "string")
        description = schema.get("description", f"{schema_type} value")
        
        if schema_type in ("string", "str"):
            return ("str", description)
        elif schema_type in ("integer", "number", "int", "float"):
            return ("int", description)
        elif schema_type in ("boolean", "bool"):
            return ("bool", description)
        else:
            return ("str", description)
    
    @fast_trace
    def _execute_agently_request(
        self, 
        agently_client, 
        user_query: str, 
        agently_schema: Dict[str, Any],
        lightweight: bool = True
    ) -> Dict[str, Any]:
        """执行Agently请求
        
        Args:
            agently_client: Agently客户端实例
            user_query: 用户查询
            agently_schema: Agently格式的Schema
            lightweight: 是否使用轻量级模式
            
        Returns:
            结构化输出结果
        """
        try:
            # 使用轻量级模式时跳过额外的验证和处理
            if lightweight:
                result = (
                    agently_client
                    .input(user_query)
                    .output(agently_schema)
                    .start()
                )
            else:
                # 标准模式包含更多验证
                result = (
                    agently_client
                    .input(user_query)
                    .output(agently_schema)
                    .start()
                )
            
            if result is None:
                raise StructuredOutputError("Agently返回空结果")
            
            return result
            
        except Exception as e:
            logger.error(f"Agently请求执行失败: {e}")
            raise StructuredOutputError(f"Agently执行失败: {e}")
    
    def _create_and_execute(
        self, 
        user_query: str, 
        agently_schema: Dict[str, Any], 
        api_key: str, 
        base_url: str, 
        model: str
    ) -> Dict[str, Any]:
        """创建客户端并执行请求（回退方案）
        
        Args:
            user_query: 用户查询
            agently_schema: Agently格式的Schema
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            
        Returns:
            结构化输出结果
        """
        try:
            from agently import Agently
            
            # 配置Agently
            Agently.set_settings(
                "OpenAICompatible",
                {
                    "base_url": base_url,
                    "model": model,
                    "model_type": "chat",
                    "auth": api_key,
                },
            )
            
            agent = Agently.create_agent()
            
            result = (
                agent
                .input(user_query)
                .output(agently_schema)
                .start()
            )
            
            if result is None:
                raise StructuredOutputError("Agently返回空结果")
            
            return result
            
        except Exception as e:
            logger.error(f"直接创建Agently客户端失败: {e}")
            raise StructuredOutputError(f"客户端创建失败: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息
        
        Returns:
            性能统计信息字典
        """
        with self._stats_lock:
            stats = self._stats.copy()
        
        # 计算缓存命中率
        total_cache_operations = stats['cache_hits'] + stats['cache_misses']
        cache_hit_rate = (
            stats['cache_hits'] / total_cache_operations 
            if total_cache_operations > 0 else 0.0
        )
        
        # 计算快速路径使用率
        fast_path_rate = (
            stats['fast_path_usage'] / stats['total_requests'] 
            if stats['total_requests'] > 0 else 0.0
        )
        
        return {
            **stats,
            'cache_hit_rate': cache_hit_rate,
            'fast_path_rate': fast_path_rate,
            'config': {
                'enable_schema_cache': self.config.enable_schema_cache,
                'enable_client_pool': self.config.enable_client_pool,
                'enable_config_cache': self.config.enable_config_cache,
                'use_lightweight_parsing': self.config.use_lightweight_parsing
            }
        }
    
    def clear_caches(self) -> None:
        """清空所有缓存"""
        if self._cache_manager:
            self._cache_manager.clear_all_caches()
        
        if self._client_pool:
            self._client_pool.clear_pool()
        
        logger.info("快速结构化输出处理器缓存已清空")


# 全局快速处理器实例
_fast_processor: Optional[FastStructuredOutputProcessor] = None


def get_fast_structured_output_processor() -> FastStructuredOutputProcessor:
    """获取全局快速结构化输出处理器实例
    
    Returns:
        全局快速处理器实例
    """
    global _fast_processor
    if _fast_processor is None:
        _fast_processor = FastStructuredOutputProcessor()
    return _fast_processor


def create_fast_structured_output_processor(
    config: Optional[FastProcessingConfig] = None,
    client_manager=None
) -> FastStructuredOutputProcessor:
    """创建快速结构化输出处理器实例
    
    Args:
        config: 快速处理配置
        client_manager: 客户端管理器实例
        
    Returns:
        快速处理器实例
    """
    return FastStructuredOutputProcessor(config=config, client_manager=client_manager)