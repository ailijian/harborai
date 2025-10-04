#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速结构化输出处理器单元测试
"""

import pytest
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from harborai.core.fast_structured_output import (
    FastStructuredOutputProcessor,
    FastProcessingConfig,
    get_fast_structured_output_processor,
    create_fast_structured_output_processor
)
from harborai.utils.exceptions import StructuredOutputError


class TestFastProcessingConfig:
    """测试快速处理配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = FastProcessingConfig()
        
        assert config.enable_schema_cache is True
        assert config.enable_client_pool is True
        assert config.enable_config_cache is True
        assert config.skip_validation is True
        assert config.max_retry_attempts == 1
        assert config.timeout_seconds == 10.0
        assert config.use_lightweight_parsing is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = FastProcessingConfig(
            enable_schema_cache=False,
            enable_client_pool=False,
            max_retry_attempts=3,
            timeout_seconds=30.0
        )
        
        assert config.enable_schema_cache is False
        assert config.enable_client_pool is False
        assert config.max_retry_attempts == 3
        assert config.timeout_seconds == 30.0


class TestFastStructuredOutputProcessor:
    """测试快速结构化输出处理器"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        config = FastProcessingConfig(
            enable_schema_cache=True,
            enable_client_pool=True,
            enable_config_cache=True
        )
        return FastStructuredOutputProcessor(config)
    
    @pytest.fixture
    def sample_schema(self):
        """示例JSON Schema"""
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "用户姓名"
                },
                "age": {
                    "type": "integer",
                    "description": "用户年龄"
                },
                "active": {
                    "type": "boolean",
                    "description": "是否活跃"
                }
            },
            "required": ["name", "age"]
        }
    
    def test_initialization(self, processor):
        """测试初始化"""
        assert processor.config.enable_schema_cache is True
        assert processor.config.enable_client_pool is True
        assert processor._stats['total_requests'] == 0
        assert processor._stats['cache_hits'] == 0
        assert processor._stats['cache_misses'] == 0
    
    def test_initialization_with_default_config(self):
        """测试使用默认配置初始化"""
        processor = FastStructuredOutputProcessor()
        assert processor.config.enable_schema_cache is True
        assert processor.config.enable_client_pool is True
    
    def test_convert_object_schema_fast(self, processor, sample_schema):
        """测试快速转换object类型Schema"""
        result = processor._convert_object_schema_fast(sample_schema)
        
        expected = {
            "name": ("str", "用户姓名"),
            "age": ("int", "用户年龄"),
            "active": ("bool", "是否活跃")
        }
        
        assert result == expected
    
    def test_convert_array_schema_fast(self, processor):
        """测试快速转换array类型Schema"""
        schema = {
            "type": "array",
            "items": {
                "type": "string",
                "description": "字符串项"
            },
            "description": "字符串数组"
        }
        
        result = processor._convert_array_schema_fast(schema)
        expected = [("str", "字符串数组")]
        
        assert result == expected
    
    def test_convert_array_schema_with_object_items(self, processor):
        """测试转换包含对象项的数组Schema"""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer", "description": "ID"},
                    "name": {"type": "string", "description": "名称"}
                }
            }
        }
        
        result = processor._convert_array_schema_fast(schema)
        expected = [{
            "id": ("int", "ID"),
            "name": ("str", "名称")
        }]
        
        assert result == expected
    
    def test_convert_primitive_schema_fast(self, processor):
        """测试快速转换基本类型Schema"""
        # 测试字符串类型
        string_schema = {"type": "string", "description": "文本"}
        result = processor._convert_primitive_schema_fast(string_schema)
        assert result == ("str", "文本")
        
        # 测试整数类型
        int_schema = {"type": "integer", "description": "数字"}
        result = processor._convert_primitive_schema_fast(int_schema)
        assert result == ("int", "数字")
        
        # 测试布尔类型
        bool_schema = {"type": "boolean", "description": "布尔值"}
        result = processor._convert_primitive_schema_fast(bool_schema)
        assert result == ("bool", "布尔值")
        
        # 测试未知类型
        unknown_schema = {"type": "unknown", "description": "未知类型"}
        result = processor._convert_primitive_schema_fast(unknown_schema)
        assert result == ("str", "未知类型")
    
    def test_convert_json_schema_to_agently(self, processor, sample_schema):
        """测试JSON Schema到Agently格式的转换"""
        result = processor._convert_json_schema_to_agently(sample_schema)
        
        expected = {
            "name": ("str", "用户姓名"),
            "age": ("int", "用户年龄"),
            "active": ("bool", "是否活跃")
        }
        
        assert result == expected
    
    def test_convert_json_schema_invalid_input(self, processor):
        """测试无效输入的Schema转换"""
        # 测试非字典输入
        result = processor._convert_json_schema_to_agently("invalid")
        assert result == {"result": ("str", "Generated result")}
        
        # 测试空字典
        result = processor._convert_json_schema_to_agently({})
        assert result == {}
    
    @patch('harborai.core.fast_structured_output.get_parameter_cache_manager')
    def test_convert_schema_with_cache_hit(self, mock_cache_manager, processor):
        """测试Schema转换缓存命中"""
        # 模拟缓存管理器
        mock_cache = Mock()
        mock_cache.schema_cache.get_converted_schema.return_value = {"cached": "result"}
        mock_cache_manager.return_value = mock_cache
        processor._cache_manager = mock_cache
        
        schema = {"type": "string"}
        result = processor._convert_schema_with_cache(schema)
        
        assert result == {"cached": "result"}
        assert processor._stats['cache_hits'] == 1
        assert processor._stats['cache_misses'] == 0
    
    @patch('harborai.core.fast_structured_output.get_parameter_cache_manager')
    def test_convert_schema_with_cache_miss(self, mock_cache_manager, processor):
        """测试Schema转换缓存未命中"""
        # 模拟缓存管理器
        mock_cache = Mock()
        mock_cache.schema_cache.get_converted_schema.return_value = None
        mock_cache_manager.return_value = mock_cache
        processor._cache_manager = mock_cache
        
        schema = {"type": "string", "description": "测试"}
        result = processor._convert_schema_with_cache(schema)
        
        expected = ("str", "测试")
        assert result == expected
        assert processor._stats['cache_hits'] == 0
        assert processor._stats['cache_misses'] == 1
        
        # 验证缓存存储被调用
        mock_cache.schema_cache.set_converted_schema.assert_called_once_with(schema, expected)
    
    @patch('harborai.core.fast_structured_output.get_parameter_cache_manager')
    def test_process_config_with_cache_hit(self, mock_cache_manager, processor):
        """测试配置处理缓存命中"""
        # 模拟缓存管理器
        mock_cache = Mock()
        mock_cache.config_cache.get_config.return_value = {"cached": "config"}
        mock_cache_manager.return_value = mock_cache
        processor._cache_manager = mock_cache
        
        result = processor._process_config_with_cache("api_key", "base_url", "model")
        
        assert result == {"cached": "config"}
        assert processor._stats['cache_hits'] == 1
        assert processor._stats['cache_misses'] == 0
    
    @patch('harborai.core.fast_structured_output.get_parameter_cache_manager')
    def test_process_config_with_cache_miss(self, mock_cache_manager, processor):
        """测试配置处理缓存未命中"""
        # 模拟缓存管理器
        mock_cache = Mock()
        mock_cache.config_cache.get_config.return_value = None
        mock_cache_manager.return_value = mock_cache
        processor._cache_manager = mock_cache
        
        result = processor._process_config_with_cache("api_key", "base_url", "model")
        
        expected = {
            'api_key': "api_key",
            'base_url': "base_url",
            'model': "model",
            'timeout': processor.config.timeout_seconds,
            'max_retries': processor.config.max_retry_attempts
        }
        
        assert result == expected
        assert processor._stats['cache_hits'] == 0
        assert processor._stats['cache_misses'] == 1
        
        # 验证缓存存储被调用
        config_data = {
            'api_key_hash': hash("api_key"),
            'base_url': "base_url",
            'model': "model"
        }
        mock_cache.config_cache.set_config.assert_called_once_with(config_data, expected)
    
    def test_can_use_fast_path_no_cache(self):
        """测试无缓存时不能使用快速路径"""
        config = FastProcessingConfig(enable_schema_cache=False)
        processor = FastStructuredOutputProcessor(config)
        
        result = processor._can_use_fast_path({}, "api_key", "base_url", "model")
        assert result is False
    
    @patch('harborai.core.fast_structured_output.get_parameter_cache_manager')
    def test_can_use_fast_path_cache_miss(self, mock_cache_manager, processor):
        """测试缓存未命中时不能使用快速路径"""
        # 模拟缓存管理器
        mock_cache = Mock()
        mock_cache.schema_cache.get_converted_schema.return_value = None
        mock_cache_manager.return_value = mock_cache
        processor._cache_manager = mock_cache
        
        result = processor._can_use_fast_path({}, "api_key", "base_url", "model")
        assert result is False
    
    @patch('harborai.core.fast_structured_output.get_parameter_cache_manager')
    def test_can_use_fast_path_success(self, mock_cache_manager, processor):
        """测试可以使用快速路径"""
        # 模拟缓存管理器
        mock_cache = Mock()
        mock_cache.schema_cache.get_converted_schema.return_value = {"cached": "schema"}
        mock_cache.config_cache.get_config.return_value = {"cached": "config"}
        mock_cache_manager.return_value = mock_cache
        processor._cache_manager = mock_cache
        
        result = processor._can_use_fast_path({}, "api_key", "base_url", "model")
        assert result is True
    
    def test_execute_agently_request_lightweight(self, processor):
        """测试轻量级模式执行Agently请求"""
        # 模拟Agently客户端
        mock_client = Mock()
        mock_client.input.return_value = mock_client
        mock_client.output.return_value = mock_client
        mock_client.start.return_value = {"result": "success"}
        
        result = processor._execute_agently_request(
            mock_client, 
            "test query", 
            {"test": "schema"},
            lightweight=True
        )
        
        assert result == {"result": "success"}
        mock_client.input.assert_called_once_with("test query")
        mock_client.output.assert_called_once_with({"test": "schema"})
        mock_client.start.assert_called_once()
    
    def test_execute_agently_request_error(self, processor):
        """测试Agently请求执行错误"""
        # 模拟Agently客户端抛出异常
        mock_client = Mock()
        mock_client.input.side_effect = Exception("Test error")
        
        with pytest.raises(StructuredOutputError, match="Agently执行失败"):
            processor._execute_agently_request(
                mock_client, 
                "test query", 
                {"test": "schema"}
            )
    
    def test_execute_agently_request_none_result(self, processor):
        """测试Agently返回空结果"""
        # 模拟Agently客户端返回None
        mock_client = Mock()
        mock_client.input.return_value = mock_client
        mock_client.output.return_value = mock_client
        mock_client.start.return_value = None
        
        with pytest.raises(StructuredOutputError, match="Agently返回空结果"):
            processor._execute_agently_request(
                mock_client, 
                "test query", 
                {"test": "schema"}
            )
    
    def test_get_performance_stats(self, processor):
        """测试获取性能统计信息"""
        # 模拟一些统计数据
        processor._stats['total_requests'] = 10
        processor._stats['cache_hits'] = 8
        processor._stats['cache_misses'] = 2
        processor._stats['fast_path_usage'] = 6
        processor._stats['avg_processing_time'] = 0.5
        
        stats = processor.get_performance_stats()
        
        assert stats['total_requests'] == 10
        assert stats['cache_hits'] == 8
        assert stats['cache_misses'] == 2
        assert stats['fast_path_usage'] == 6
        assert stats['cache_hit_rate'] == 0.8  # 8/10
        assert stats['fast_path_rate'] == 0.6  # 6/10
        assert stats['avg_processing_time'] == 0.5
        
        # 检查配置信息
        assert 'config' in stats
        assert stats['config']['enable_schema_cache'] is True
        assert stats['config']['enable_client_pool'] is True
    
    def test_get_performance_stats_zero_operations(self, processor):
        """测试零操作时的性能统计"""
        stats = processor.get_performance_stats()
        
        assert stats['cache_hit_rate'] == 0.0
        assert stats['fast_path_rate'] == 0.0
    
    @patch('harborai.core.fast_structured_output.get_parameter_cache_manager')
    @patch('harborai.core.fast_structured_output.get_agently_client_pool')
    def test_clear_caches(self, mock_client_pool, mock_cache_manager, processor):
        """测试清空缓存"""
        # 模拟缓存管理器和客户端池
        mock_cache = Mock()
        mock_pool = Mock()
        mock_cache_manager.return_value = mock_cache
        mock_client_pool.return_value = mock_pool
        
        processor._cache_manager = mock_cache
        processor._client_pool = mock_pool
        
        processor.clear_caches()
        
        mock_cache.clear_all_caches.assert_called_once()
        mock_pool.clear_pool.assert_called_once()


class TestGlobalFunctions:
    """测试全局函数"""
    
    def test_get_fast_structured_output_processor(self):
        """测试获取全局处理器实例"""
        processor1 = get_fast_structured_output_processor()
        processor2 = get_fast_structured_output_processor()
        
        # 应该返回同一个实例
        assert processor1 is processor2
        assert isinstance(processor1, FastStructuredOutputProcessor)
    
    def test_create_fast_structured_output_processor(self):
        """测试创建处理器实例"""
        config = FastProcessingConfig(enable_schema_cache=False)
        processor = create_fast_structured_output_processor(config)
        
        assert isinstance(processor, FastStructuredOutputProcessor)
        assert processor.config.enable_schema_cache is False
    
    def test_create_fast_structured_output_processor_default_config(self):
        """测试使用默认配置创建处理器实例"""
        processor = create_fast_structured_output_processor()
        
        assert isinstance(processor, FastStructuredOutputProcessor)
        assert processor.config.enable_schema_cache is True


class TestComplexSchemas:
    """测试复杂Schema转换"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        return FastStructuredOutputProcessor()
    
    def test_nested_object_schema(self, processor):
        """测试嵌套对象Schema"""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "姓名"},
                        "profile": {
                            "type": "object",
                            "properties": {
                                "age": {"type": "integer", "description": "年龄"},
                                "active": {"type": "boolean", "description": "活跃状态"}
                            }
                        }
                    }
                }
            }
        }
        
        result = processor._convert_json_schema_to_agently(schema)
        
        expected = {
            "user": {
                "name": ("str", "姓名"),
                "profile": {
                    "age": ("int", "年龄"),
                    "active": ("bool", "活跃状态")
                }
            }
        }
        
        assert result == expected
    
    def test_array_of_objects_schema(self, processor):
        """测试对象数组Schema"""
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "用户ID"},
                            "name": {"type": "string", "description": "用户名"}
                        }
                    }
                }
            }
        }
        
        result = processor._convert_json_schema_to_agently(schema)
        
        expected = {
            "users": [{
                "id": ("int", "用户ID"),
                "name": ("str", "用户名")
            }]
        }
        
        assert result == expected
    
    def test_mixed_types_schema(self, processor):
        """测试混合类型Schema"""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "标题"},
                "count": {"type": "number", "description": "数量"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "标签列表"
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created": {"type": "string", "description": "创建时间"},
                        "published": {"type": "boolean", "description": "是否发布"}
                    }
                }
            }
        }
        
        result = processor._convert_json_schema_to_agently(schema)
        
        expected = {
            "title": ("str", "标题"),
            "count": ("int", "数量"),  # number类型映射为int
            "tags": [("str", "标签列表")],
            "metadata": {
                "created": ("str", "创建时间"),
                "published": ("bool", "是否发布")
            }
        }
        
        assert result == expected


class TestThreadSafety:
    """测试线程安全性"""
    
    @pytest.fixture
    def processor(self):
        """创建处理器实例"""
        return FastStructuredOutputProcessor()
    
    def test_stats_thread_safety(self, processor):
        """测试统计信息的线程安全性"""
        def update_stats():
            for _ in range(100):
                with processor._stats_lock:
                    processor._stats['total_requests'] += 1
                    processor._stats['cache_hits'] += 1
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=update_stats)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 验证最终结果
        assert processor._stats['total_requests'] == 1000  # 10 threads * 100 updates
        assert processor._stats['cache_hits'] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])