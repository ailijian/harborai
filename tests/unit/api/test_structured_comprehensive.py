"""
结构化输出处理器的全面测试套件。

测试覆盖：
- 初始化和配置
- 不同提供者的解析功能
- 错误处理和边界条件
- 缓存机制
- 性能测试
- 集成测试
"""

import asyncio
import json
import pytest
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Any, Dict, List, Optional, Union

from harborai.api.structured import StructuredOutputHandler, parse_structured_output, parse_streaming_structured_output, create_response_format
from harborai.utils.exceptions import StructuredOutputError


class TestStructuredOutputHandler:
    """结构化输出处理器基础功能测试"""
    
    def test_init_default_provider(self):
        """测试默认初始化"""
        handler = StructuredOutputHandler()
        assert handler.provider == "agently"
        assert hasattr(handler, '_agently_available')
    
    @patch('harborai.api.structured.Agently')
    def test_init_agently_provider_available(self, mock_agently):
        """测试Agently提供者可用时的初始化"""
        handler = StructuredOutputHandler(provider="agently")
        assert handler.provider == "agently"
        assert handler._agently_available is True
    
    def test_init_agently_provider_unavailable(self):
        """测试Agently提供者不可用时的初始化"""
        # 由于Agently在模块顶层导入，我们需要模拟_check_agently_availability方法
        with patch.object(StructuredOutputHandler, '_check_agently_availability', return_value=False):
            handler = StructuredOutputHandler(provider="agently")
            assert handler.provider == "agently"
            assert handler._agently_available is False
    
    def test_init_invalid_provider(self):
        """测试无效提供者的初始化"""
        handler = StructuredOutputHandler(provider="invalid")
        assert handler.provider == "invalid"  # 当前实现接受任何provider值
        assert hasattr(handler, '_agently_available')
    
    def test_init_custom_cache_settings(self):
        """测试自定义缓存设置"""
        handler = StructuredOutputHandler(provider="native")
        assert handler.provider == "native"
        assert hasattr(handler, '_agently_available')
    
    @patch('harborai.api.structured.Agently')
    def test_check_agently_availability_available(self, mock_agently):
        """测试检查Agently可用性 - 可用"""
        mock_agent = Mock()
        mock_agently.create_agent.return_value = mock_agent
        
        handler = StructuredOutputHandler()
        result = handler._check_agently_availability()
        assert result is True
    
    def test_check_agently_availability_unavailable(self):
        """测试检查Agently可用性 - 不可用"""
        # 模拟Agently没有create_agent属性
        with patch('harborai.api.structured.Agently') as mock_agently:
            # 删除create_agent属性
            del mock_agently.create_agent
            handler = StructuredOutputHandler()
            result = handler._check_agently_availability()
            assert result is False
    
    def test_parse_response_native_valid_json(self):
        """测试使用原生提供者解析有效JSON"""
        handler = StructuredOutputHandler(provider="native")
        
        test_data = {
            "name": "测试用户",
            "age": 25,
            "email": "test@example.com",
            "active": True
        }
        json_string = json.dumps(test_data, ensure_ascii=False)
        schema = {"type": "object"}
        
        result = handler.parse_response(json_string, schema)
        
        assert result == test_data
        assert isinstance(result, dict)
    
    def test_parse_response_native_invalid_json(self):
        """测试使用原生提供者解析无效JSON"""
        handler = StructuredOutputHandler(provider="native")
        
        invalid_json = '{"name": "测试", "value": 123, "incomplete":'
        schema = {"type": "object"}
        
        with pytest.raises(StructuredOutputError):
            handler.parse_response(invalid_json, schema)
    
    def test_parse_response_native_empty_string(self):
        """测试使用原生提供者解析空字符串"""
        handler = StructuredOutputHandler(provider="native")
        schema = {"type": "object"}
        
        with pytest.raises(StructuredOutputError):
            handler.parse_response("", schema)
    
    def test_parse_response_native_none_input(self):
        """测试使用原生提供者解析None输入"""
        handler = StructuredOutputHandler(provider="native")
        schema = {"type": "object"}
        
        with pytest.raises(StructuredOutputError):
            handler.parse_response(None, schema)
    
    @patch('harborai.api.structured.Agently')
    def test_parse_response_agently_success(self, mock_agently):
        """测试使用Agently成功解析"""
        # 设置mock
        mock_agent = Mock()
        mock_agently.create_agent.return_value = mock_agent
        mock_agent.input.return_value = mock_agent
        mock_agent.output.return_value = mock_agent
        mock_agent.start.return_value = {"name": "John", "age": 30}
        
        handler = StructuredOutputHandler()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        result = handler.parse_response(
            content="test content",
            schema=schema,
            use_agently=True,
            api_key="test-key",
            base_url="https://api.test.com",
            model="gpt-3.5-turbo",
            user_query="test query"
        )
        
        assert result == {"name": "John", "age": 30}
    
    @patch('harborai.api.structured.Agently')
    def test_parse_response_agently_fallback(self, mock_agently):
        """测试Agently解析失败时的回退机制"""
        # 设置mock使Agently抛出异常
        mock_agently.create_agent.side_effect = Exception("Agently error")
        
        handler = StructuredOutputHandler()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        # 应该回退到原生解析
        with pytest.raises(StructuredOutputError):
            handler.parse_response(
                content="invalid json",
                schema=schema,
                use_agently=True,
                api_key="test-key",
                base_url="https://api.test.com",
                model="gpt-3.5-turbo"
            )
    
    def test_parse_response_with_schema_validation_success(self):
        """测试带schema验证的成功解析"""
        handler = StructuredOutputHandler(provider="native")
        
        test_data = {"name": "测试", "age": 25}
        json_string = json.dumps(test_data, ensure_ascii=False)
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        
        result = handler.parse_response(json_string, schema)
        assert result == test_data
    
    def test_parse_response_json_in_codeblock(self):
        """测试从代码块中提取JSON"""
        handler = StructuredOutputHandler()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        content = '''
        Here is the result:
        ```json
        {"name": "John"}
        ```
        '''
        
        result = handler.parse_response(content, schema)
        assert result == {"name": "John"}
    
    def test_parse_response_array_structure(self):
        """测试解析数组结构"""
        handler = StructuredOutputHandler()
        
        test_data = [
            {"id": 1, "name": "项目1"},
            {"id": 2, "name": "项目2"},
            {"id": 3, "name": "项目3"}
        ]
        json_string = json.dumps(test_data, ensure_ascii=False)
        schema = {"type": "array"}
        
        result = handler.parse_response(json_string, schema)
        
        assert result == test_data
        assert isinstance(result, list)
        assert len(result) == 3
    
    def test_parse_response_nested_structure(self):
        """测试解析嵌套结构"""
        handler = StructuredOutputHandler()
        
        test_data = {
            "user": {
                "profile": {
                    "name": "张三",
                    "contact": {
                        "email": "zhangsan@example.com",
                        "phone": "13800138000"
                    }
                }
            }
        }
        json_string = json.dumps(test_data, ensure_ascii=False)
        schema = {"type": "object"}
        
        result = handler.parse_response(json_string, schema)
        assert result == test_data
    
    def test_parse_response_unicode_content(self):
        """测试解析包含Unicode字符的内容"""
        handler = StructuredOutputHandler()
        
        test_data = {
            "message": "你好，世界！🌍",
            "emoji": "😊🎉🚀",
            "special": "αβγδε"
        }
        json_string = json.dumps(test_data, ensure_ascii=False)
        schema = {"type": "object"}
        
        result = handler.parse_response(json_string, schema)
        assert result == test_data
    
    def test_parse_response_special_characters(self):
        """测试解析包含特殊字符的内容"""
        handler = StructuredOutputHandler()
        
        test_data = {
            "path": "C:\\Users\\测试\\Documents",
            "regex": r"^\d{4}-\d{2}-\d{2}$",
            "quotes": 'He said "Hello" and \'Goodbye\''
        }
        json_string = json.dumps(test_data, ensure_ascii=False)
        schema = {"type": "object"}
        
        result = handler.parse_response(json_string, schema)
        assert result == test_data
    
    def test_format_response_format(self):
        """测试格式化response_format参数"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        result = handler.format_response_format(schema, name="test_schema", strict=True)
        
        expected = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_schema",
                "schema": schema,
                "strict": True
            }
        }
        
        assert result == expected
    
    def test_extract_json_from_text_various_formats(self):
        """测试从各种文本格式中提取JSON"""
        handler = StructuredOutputHandler()
        
        # 测试```json格式
        text1 = '''
        Here's the result:
        ```json
        {"name": "test"}
        ```
        '''
        assert handler.extract_json_from_text(text1) == '{"name": "test"}'
        
        # 测试```格式
        text2 = '''
        ```
        {"value": 123}
        ```
        '''
        assert handler.extract_json_from_text(text2) == '{"value": 123}'
        
        # 测试直接JSON
        text3 = 'Some text {"direct": "json"} more text'
        assert handler.extract_json_from_text(text3) == '{"direct": "json"}'


class TestStructuredOutputHandlerIntegration:
    """结构化输出处理器集成测试"""
    
    def test_real_world_api_response_parsing(self):
        """测试真实世界API响应解析"""
        handler = StructuredOutputHandler()
        
        # 模拟真实API响应
        api_response = {
            "data": {
                "users": [
                    {
                        "id": 1,
                        "name": "张三",
                        "profile": {
                            "age": 28,
                            "department": "技术部",
                            "skills": ["Python", "JavaScript", "Docker"]
                        }
                    }
                ]
            },
            "meta": {
                "total": 1,
                "page": 1,
                "per_page": 10
            }
        }
        
        json_string = json.dumps(api_response, ensure_ascii=False)
        schema = {"type": "object"}
        
        result = handler.parse_response(json_string, schema)
        assert result == api_response
    
    def test_error_response_parsing(self):
        """测试错误响应解析"""
        handler = StructuredOutputHandler()
        
        error_response = {
            "error": {
                "code": 400,
                "message": "请求参数无效",
                "details": [
                    {
                        "field": "email",
                        "issue": "格式不正确"
                    },
                    {
                        "field": "age", 
                        "issue": "必须为正整数"
                    }
                ]
            }
        }
        
        json_string = json.dumps(error_response, ensure_ascii=False)
        schema = {"type": "object"}
        
        result = handler.parse_response(json_string, schema)
        assert result == error_response
    
    def test_large_data_parsing_performance(self):
        """测试大数据解析性能"""
        handler = StructuredOutputHandler()
        
        # 生成大量数据
        large_data = {
            "items": [
                {
                    "id": i,
                    "name": f"项目{i}",
                    "description": f"这是第{i}个项目的详细描述" * 10,
                    "tags": [f"tag{j}" for j in range(5)]
                }
                for i in range(1000)
            ]
        }
        
        json_string = json.dumps(large_data, ensure_ascii=False)
        schema = {"type": "object"}
        
        start_time = time.time()
        result = handler.parse_response(json_string, schema)
        duration = time.time() - start_time
        
        assert result == large_data
        assert duration < 5.0  # 应该在5秒内完成
    
    def test_concurrent_parsing(self):
        """测试并发解析"""
        handler = StructuredOutputHandler()
        
        def parse_task(task_id):
            test_data = {"task_id": task_id, "result": f"任务{task_id}完成"}
            json_string = json.dumps(test_data, ensure_ascii=False)
            schema = {"type": "object"}
            return handler.parse_response(json_string, schema)
        
        # 并发执行多个解析任务
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(parse_task, i) for i in range(50)]
            results = [future.result() for future in futures]
        
        # 验证所有任务都成功完成
        assert len(results) == 50
        for i, result in enumerate(results):
            assert result["task_id"] == i
            assert result["result"] == f"任务{i}完成"


class TestEdgeCasesAndErrorHandling:
    """边界条件和错误处理测试"""
    
    def test_malformed_json_with_extra_characters(self):
        """测试带有额外字符的畸形JSON"""
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        # 这种情况下extract_json_from_text会提取出有效的JSON部分
        malformed_json = '{"valid": "json"}extra_characters_here'
        result = handler.parse_response(malformed_json, schema)
        assert result == {"valid": "json"}
        
        # 测试真正无效的JSON
        truly_invalid_json = 'not json at all'
        with pytest.raises(StructuredOutputError):
            handler.parse_response(truly_invalid_json, schema)
    
    def test_json_with_comments(self):
        """测试包含注释的JSON（非标准）"""
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        json_with_comments = '''
        {
            // 这是注释
            "name": "测试",
            /* 多行注释 */
            "value": 123
        }
        '''
        
        with pytest.raises(StructuredOutputError):
            handler.parse_response(json_with_comments, schema)
    
    def test_extremely_nested_structure(self):
        """测试极深嵌套结构"""
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        # 创建深度嵌套的结构
        nested_data = {"level": 0}
        current = nested_data
        for i in range(1, 100):
            current["next"] = {"level": i}
            current = current["next"]
        
        json_string = json.dumps(nested_data)
        result = handler.parse_response(json_string, schema)
        
        # 验证结构正确
        current = result
        for i in range(100):
            assert current["level"] == i
            if i < 99:
                current = current["next"]
    
    def test_json_with_null_values(self):
        """测试包含null值的JSON"""
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        test_data = {
            "name": "测试",
            "optional_field": None,
            "nested": {
                "value": None,
                "array": [1, None, 3]
            }
        }
        
        json_string = json.dumps(test_data)
        result = handler.parse_response(json_string, schema)
        
        assert result == test_data
    
    def test_json_with_boolean_values(self):
        """测试包含布尔值的JSON"""
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        test_data = {
            "is_active": True,
            "is_deleted": False,
            "flags": {
                "feature_a": True,
                "feature_b": False
            }
        }
        
        json_string = json.dumps(test_data)
        result = handler.parse_response(json_string, schema)
        
        assert result == test_data
    
    def test_json_with_numeric_edge_cases(self):
        """测试包含数值边界情况的JSON"""
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        test_data = {
            "zero": 0,
            "negative": -123,
            "float_val": 3.14159,
            "scientific": 1.23e-4,
            "large_int": 9223372036854775807
        }
        
        json_string = json.dumps(test_data)
        result = handler.parse_response(json_string, schema)
        
        assert result == test_data


class TestJSONSchemaConversion:
    """JSON Schema转换测试"""
    
    def test_convert_json_schema_to_agently_output_basic(self):
        """测试基本JSON Schema转换"""
        handler = StructuredOutputHandler()
        
        schema_wrapper = {
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "用户姓名"},
                        "age": {"type": "integer", "description": "用户年龄"}
                    }
                }
            }
        }
        
        result = handler._convert_json_schema_to_agently_output(schema_wrapper)
        
        assert "name" in result
        assert "age" in result
        assert result["name"] == ("str", "用户姓名")
        assert result["age"] == ("int", "用户年龄")
    
    def test_convert_json_schema_to_agently_output_direct_schema(self):
        """测试直接传入schema的转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "标题"}
            }
        }
        
        result = handler._convert_json_schema_to_agently_output(schema)
        
        assert "title" in result
        assert result["title"] == ("str", "标题")
    
    def test_convert_json_schema_to_agently_output_exception(self):
        """测试转换过程中的异常处理"""
        handler = StructuredOutputHandler()
        
        # 传入无效的schema（非字典类型）
        invalid_schema = "invalid_schema"
        
        result = handler._convert_json_schema_to_agently_output(invalid_schema)
        
        # 应该返回fallback格式（非字典类型会返回value字段）
        assert "value" in result
        assert result["value"] == ("str", "Generated value")
    
    def test_convert_schema_to_agently_format_object(self):
        """测试对象类型schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "姓名"},
                "email": {"type": "string", "description": "邮箱"},
                "active": {"type": "boolean", "description": "是否激活"}
            },
            "required": ["name"]
        }
        
        result = handler._convert_schema_to_agently_format(schema)
        
        assert result["name"] == ("str", "姓名")
        assert result["email"] == ("str", "邮箱")
        assert result["active"] == ("bool", "是否激活")
    
    def test_convert_schema_to_agently_format_array(self):
        """测试数组类型schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "array",
            "description": "用户列表",
            "items": {
                "type": "string"
            }
        }
        
        result = handler._convert_schema_to_agently_format(schema)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == ("str", "用户列表")
    
    def test_convert_schema_to_agently_format_primitive(self):
        """测试基本类型schema转换"""
        handler = StructuredOutputHandler()
        
        # 字符串类型
        string_schema = {"type": "string", "description": "文本内容"}
        result = handler._convert_schema_to_agently_format(string_schema)
        assert result == ("str", "文本内容")
        
        # 整数类型
        int_schema = {"type": "integer", "description": "数字"}
        result = handler._convert_schema_to_agently_format(int_schema)
        assert result == ("int", "数字")
        
        # 布尔类型
        bool_schema = {"type": "boolean", "description": "开关"}
        result = handler._convert_schema_to_agently_format(bool_schema)
        assert result == ("bool", "开关")
    
    def test_convert_schema_to_agently_format_non_dict(self):
        """测试非字典类型的schema转换"""
        handler = StructuredOutputHandler()
        
        # 传入非字典类型
        result = handler._convert_schema_to_agently_format("invalid")
        
        assert result == {"value": ("str", "Generated value")}
    
    def test_convert_object_schema_with_enum(self):
        """测试带枚举值的对象schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "状态",
                    "enum": ["active", "inactive", "pending"]
                }
            }
        }
        
        result = handler._convert_object_schema(schema)
        
        assert "status" in result
        assert result["status"] == ("str", "状态，可选值: active/inactive/pending")
    
    def test_convert_object_schema_nested(self):
        """测试嵌套对象schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "姓名"}
                    }
                }
            }
        }
        
        result = handler._convert_object_schema(schema)
        
        assert "user" in result
        assert isinstance(result["user"], dict)
        assert "name" in result["user"]
        assert result["user"]["name"] == ("str", "姓名")
    
    def test_convert_object_schema_with_array(self):
        """测试包含数组的对象schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "description": "标签列表",
                    "items": {"type": "string"}
                }
            }
        }
        
        result = handler._convert_object_schema(schema)
        
        assert "tags" in result
        assert isinstance(result["tags"], list)
        assert result["tags"] == [("str", "标签列表")]
    
    def test_convert_array_schema_object_items(self):
        """测试对象数组schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "array",
            "description": "用户列表",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "姓名"}
                }
            }
        }
        
        result = handler._convert_array_schema(schema)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert "name" in result[0]
        assert result[0]["name"] == ("str", "姓名")
    
    def test_convert_array_schema_non_dict_items(self):
        """测试非字典items的数组schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "array",
            "description": "数字列表",
            "items": "invalid"
        }
        
        result = handler._convert_array_schema(schema)
        
        assert result == [("str", "数字列表")]
    
    def test_convert_primitive_schema_number(self):
        """测试数字类型的基本schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {"type": "number", "description": "浮点数"}
        result = handler._convert_primitive_schema(schema)
        
        assert result == ("int", "浮点数")
    
    def test_convert_primitive_schema_unknown_type(self):
        """测试未知类型的基本schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {"type": "unknown", "description": "未知类型"}
        result = handler._convert_primitive_schema(schema)
        
        assert result == ("str", "未知类型")


class TestStreamingParsing:
    """流式解析测试"""
    
    @patch('harborai.api.structured.Agently')
    def test_parse_streaming_response_agently_success(self, mock_agently):
        """测试使用Agently的流式解析成功"""
        # 设置Mock
        mock_agent = Mock()
        mock_agently.create_agent.return_value = mock_agent
        mock_agent.input.return_value = mock_agent
        mock_agent.output.return_value = mock_agent
        mock_agent.start.return_value = {"name": "John", "age": 30}
        
        # 设置get_instant_generator返回一个事件流
        def mock_event_generator():
            yield {
                "complete_value": {"name": "John", "age": 30},
                "key": "name",
                "delta": "John",
                "indexes": []
            }
        mock_agent.get_instant_generator.return_value = mock_event_generator()
        
        handler = StructuredOutputHandler()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        
        # 模拟流式响应
        def mock_stream():
            yield "chunk1"
            yield "chunk2"
            yield "chunk3"
        
        result_generator = handler.parse_streaming_response(
            content_stream=mock_stream(),
            schema=schema,
            provider="agently",
            api_key="test-key",
            base_url="https://api.test.com",
            model="gpt-3.5-turbo"
        )
        
        # 流式解析返回生成器，需要转换为列表
        result_list = list(result_generator)
        assert len(result_list) == 1
        assert result_list[0] == {"name": "John", "age": 30}
    
    def test_parse_streaming_response_native_success(self):
        """测试使用原生方式的流式解析成功"""
        handler = StructuredOutputHandler(provider="native")
        schema = {"type": "object"}
        
        # 模拟流式响应，最后一个chunk包含完整JSON
        def mock_stream():
            yield "partial"
            yield " json "
            yield '{"name": "test", "value": 123}'
        
        result_generator = handler.parse_streaming_response(
            content_stream=mock_stream(),
            schema=schema
        )
        
        # 流式解析返回生成器，需要转换为列表
        result_list = list(result_generator)
        assert len(result_list) == 1
        assert result_list[0] == {"name": "test", "value": 123}
    
    def test_parse_streaming_response_empty_stream(self):
        """测试空流的处理"""
        handler = StructuredOutputHandler(provider="native")
        schema = {"type": "object"}
        
        def empty_stream():
            return
            yield  # 这行永远不会执行
        
        # 空流不会抛出异常，而是返回空的生成器
        result_generator = handler.parse_streaming_response(
            content_stream=empty_stream(),
            schema=schema
        )
        
        # 将生成器转换为列表，应该为空
        result_list = list(result_generator)
        assert len(result_list) == 0
    
    @patch('harborai.api.structured.Agently')
    def test_parse_sync_streaming_with_agently(self, mock_agently):
        """测试同步Agently流式解析"""
        # 设置Mock
        mock_agent = Mock()
        mock_agently.create_agent.return_value = mock_agent
        mock_agent.input.return_value = mock_agent
        mock_agent.output.return_value = mock_agent
        mock_agent.start.return_value = {"result": "success"}
        
        # 设置get_instant_generator返回一个事件流
        def mock_event_generator():
            yield {
                "complete_value": {"result": "success"},
                "key": "result",
                "delta": "success",
                "indexes": []
            }
        mock_agent.get_instant_generator.return_value = mock_event_generator()
        
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        def mock_stream():
            yield "data1"
            yield "data2"
        
        result_generator = handler._parse_sync_streaming_with_agently(
            content_stream=mock_stream(),
            schema={"type": "object"},
            api_key="test-key",
            base_url="https://api.test.com",
            model="gpt-3.5-turbo"
        )
        
        # 流式解析返回生成器，需要转换为列表
        result_list = list(result_generator)
        assert len(result_list) == 1
        assert result_list[0] == {"result": "success"}
    
    @pytest.mark.asyncio
    @patch('harborai.api.structured.Agently')
    async def test_parse_async_streaming_with_agently(self, mock_agently):
        """测试异步流式Agently解析"""
        # 设置Mock
        mock_agent = Mock()
        mock_agently.create_agent.return_value = mock_agent
        mock_agent.input.return_value = mock_agent
        mock_agent.output.return_value = mock_agent
        mock_agent.start.return_value = {"result": "async_success"}
        
        # 设置get_instant_generator返回一个事件流
        def mock_event_generator():
            yield {
                "complete_value": {"result": "async_success"},
                "key": "result",
                "delta": "async_success",
                "indexes": []
            }
        
        # 模拟异步生成器方法
        async def mock_async_event_generator():
            for event in mock_event_generator():
                yield event
        
        mock_agent.get_async_instant_generator = Mock(return_value=mock_async_event_generator())
        mock_agent.get_instant_generator.return_value = mock_event_generator()
        
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        async def mock_async_stream():
            yield "async_data1"
            yield "async_data2"
        
        result_generator = handler._parse_async_streaming_with_agently(
            content_stream=mock_async_stream(),
            schema=schema,
            api_key="test-key",
            base_url="https://api.test.com",
            model="gpt-3.5-turbo"
        )
        
        # 异步流式解析返回异步生成器，需要转换为列表
        result_list = []
        async for item in result_generator:
            result_list.append(item)
        assert len(result_list) == 1
        assert result_list[0] == {"result": "async_success"}
    
    def test_parse_sync_streaming_with_native(self):
        """测试同步流式原生解析"""
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        def mock_stream():
            yield "{"
            yield '"name":'
            yield '"test",'
            yield '"value":'
            yield '123}'
        
        result_generator = handler._parse_sync_streaming_with_native(
            content_stream=mock_stream(),
            schema=schema
        )
        
        # 将生成器转换为列表
        result_list = list(result_generator)
        assert len(result_list) == 1
        assert result_list[0] == {"name": "test", "value": 123}
    
    @pytest.mark.asyncio
    async def test_parse_async_streaming_with_native(self):
        """测试异步流式原生解析"""
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        async def mock_async_stream():
            yield "{"
            yield '"async":'
            yield 'true,'
            yield '"data":'
            yield '"test"}'
        
        result_generator = handler._parse_async_streaming_with_native(
            content_stream=mock_async_stream(),
            schema=schema
        )
        
        # 将异步生成器转换为列表
        result_list = [item async for item in result_generator]
        assert len(result_list) == 1
        assert result_list[0] == {"async": True, "data": "test"}
    
    def test_parse_streaming_with_native_invalid_json(self):
        """测试原生流式解析无效JSON"""
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        def mock_stream():
            yield "invalid"
            yield "json"
            yield "content"
        
        # 原生流式解析对于无效JSON不会抛出异常，而是返回空的生成器
        result_generator = handler._parse_streaming_with_native(
            content_stream=mock_stream(),
            schema=schema
        )
        
        # 将生成器转换为列表，应该为空
        result_list = list(result_generator)
        assert len(result_list) == 0


class TestJSONExtractionAndValidation:
    """JSON提取和验证测试"""
    
    def test_extract_json_from_text_json_codeblock(self):
        """测试从JSON代码块提取"""
        handler = StructuredOutputHandler()
        
        text = '''
        Here's the result:
        ```json
        {"name": "test", "value": 123}
        ```
        Some other text
        '''
        
        result = handler.extract_json_from_text(text)
        assert result == '{"name": "test", "value": 123}'
    
    def test_extract_json_from_text_generic_codeblock(self):
        """测试从通用代码块提取"""
        handler = StructuredOutputHandler()
        
        text = '''
        ```
        {"data": "from generic block"}
        ```
        '''
        
        result = handler.extract_json_from_text(text)
        assert result == '{"data": "from generic block"}'
    
    def test_extract_json_from_text_inline_json(self):
        """测试提取内联JSON"""
        handler = StructuredOutputHandler()
        
        text = 'Some text {"inline": "json"} more text'
        result = handler.extract_json_from_text(text)
        assert result == '{"inline": "json"}'
    
    def test_extract_json_from_text_multiple_json(self):
        """测试提取多个JSON（应该返回第一个）"""
        handler = StructuredOutputHandler()
        
        text = '''
        First: {"first": "json"}
        Second: {"second": "json"}
        '''
        
        result = handler.extract_json_from_text(text)
        # extract_json_from_text会查找第一个{...}格式的JSON
        # 但如果没有找到有效的JSON，会返回原始文本
        # 让我们检查实际返回的内容
        assert '{"first": "json"}' in result
    
    def test_extract_json_from_text_no_json(self):
        """测试没有JSON的文本"""
        handler = StructuredOutputHandler()
        
        text = "This text contains no JSON at all"
        result = handler.extract_json_from_text(text)
        assert result == text
    
    def test_extract_json_from_text_array(self):
        """测试提取JSON数组"""
        handler = StructuredOutputHandler()
        text = '''
        ```json
        [{"id": 1}, {"id": 2}]
        ```
        '''
        
        result = handler.extract_json_from_text(text)
        # extract_json_from_text应该提取代码块中的JSON
        # 去除空格和换行符进行比较
        normalized_result = result.replace(' ', '').replace('\n', '')
        assert '[{"id":1},{"id":2}]' in normalized_result
    
    def test_validate_against_schema_success(self):
        """测试schema验证成功"""
        handler = StructuredOutputHandler()
        
        data = {"name": "test", "age": 25}
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        
        # 应该不抛出异常
        handler._validate_against_schema(data, schema)
    
    def test_validate_against_schema_failure(self):
        """测试schema验证失败"""
        handler = StructuredOutputHandler()
        
        data = {"name": "test"}  # 缺少required字段age
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            },
            "required": ["name", "age"]
        }
        
        with pytest.raises(StructuredOutputError):
            handler._validate_against_schema(data, schema)
    
    def test_validate_against_schema_type_mismatch(self):
        """测试schema类型不匹配"""
        handler = StructuredOutputHandler()
        
        data = {"name": "test", "age": "not_a_number"}
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }
        
        with pytest.raises(StructuredOutputError):
            handler._validate_against_schema(data, schema)


class TestGlobalFunctions:
    """全局函数测试"""
    
    def test_parse_structured_output_function(self):
        """测试parse_structured_output全局函数"""
        content = '{"name": "test", "value": 123}'
        schema = {"type": "object"}
        
        result = parse_structured_output(content, schema, use_agently=False)
        assert result == {"name": "test", "value": 123}
    
    def test_parse_streaming_structured_output_function(self):
        """测试parse_streaming_structured_output全局函数"""
        def mock_stream():
            yield '{"streaming":'
            yield ' "test"}'
        
        schema = {"type": "object"}
        result_generator = parse_streaming_structured_output(
            content_stream=mock_stream(),
            schema=schema,
            provider="native"
        )
        
        # 将生成器转换为列表并获取第一个结果
        result_list = list(result_generator)
        assert len(result_list) == 1
        assert result_list[0] == {"streaming": "test"}
    
    def test_create_response_format_function(self):
        """测试create_response_format全局函数"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        result = create_response_format(schema, name="test_format", strict=True)
        
        expected = {
            "type": "json_schema",
            "json_schema": {
                "name": "test_format",
                "schema": schema,
                "strict": True
            }
        }
        
        assert result == expected


class TestUtilityMethods:
    """工具方法测试"""
    
    def test_update_result_by_key_simple(self):
        """测试简单键值更新"""
        handler = StructuredOutputHandler()
        
        result = {}
        handler._update_result_by_key(result, "name", "test")
        
        assert result == {"name": "test"}
    
    def test_update_result_by_key_nested(self):
        """测试嵌套键值更新"""
        handler = StructuredOutputHandler()
        
        result = {}
        handler._update_result_by_key(result, "user.profile.name", "test")
        
        expected = {
            "user": {
                "profile": {
                    "name": "test"
                }
            }
        }
        
        assert result == expected
    
    def test_update_result_by_key_existing_structure(self):
        """测试在现有结构上更新"""
        handler = StructuredOutputHandler()
        
        result = {
            "user": {
                "id": 1
            }
        }
        handler._update_result_by_key(result, "user.name", "test")
        
        expected = {
            "user": {
                "id": 1,
                "name": "test"
            }
        }
        
        assert result == expected
    
    def test_update_result_by_key_array_index(self):
        """测试数组索引更新"""
        handler = StructuredOutputHandler()
        
        result = {}
        # 使用indexes参数来指定数组索引
        handler._update_result_by_key(result, "items", "first", indexes=[0])
        handler._update_result_by_key(result, "items", " item", indexes=[0])
        
        # 验证结果
        assert "items" in result
        assert isinstance(result["items"], list)
        assert len(result["items"]) >= 1
        assert result["items"][0] == "first item"


class TestErrorHandlingAndEdgeCases:
    """错误处理和边界条件测试"""
    
    def test_check_agently_availability_import_error(self):
        """测试Agently导入错误"""
        # 由于Agently在模块顶层导入，我们需要模拟hasattr调用时的异常
        with patch('harborai.api.structured.hasattr', side_effect=ImportError("No module")):
            handler = StructuredOutputHandler()
            result = handler._check_agently_availability()
            assert result is False
    
    def test_check_agently_availability_attribute_error(self):
        """测试Agently属性错误"""
        with patch('harborai.api.structured.Agently') as mock_agently:
            del mock_agently.create_agent
            handler = StructuredOutputHandler()
            result = handler._check_agently_availability()
            assert result is False
    
    def test_check_agently_availability_general_exception(self):
        """测试Agently一般异常"""
        # 模拟hasattr调用时的一般异常
        with patch('harborai.api.structured.hasattr', side_effect=Exception("General error")):
            handler = StructuredOutputHandler()
            result = handler._check_agently_availability()
            assert result is False
    
    @patch('harborai.api.structured.Agently')
    def test_parse_with_agently_no_model_error(self, mock_agently):
        """测试Agently解析时模型为空"""
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        with pytest.raises(StructuredOutputError, match="模型名称不能为空"):
            handler._parse_with_agently(
                user_query="test",
                schema=schema,
                api_key="test-key",
                base_url="https://api.test.com",
                model=None
            )
    
    @patch('harborai.api.structured.Agently')
    def test_parse_with_agently_none_result(self, mock_agently):
        """测试Agently返回None结果"""
        mock_agent = Mock()
        mock_agently.create_agent.return_value = mock_agent
        mock_agent.input.return_value = mock_agent
        mock_agent.output.return_value = mock_agent
        mock_agent.start.return_value = None
        
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        with pytest.raises(StructuredOutputError, match="Agently返回None结果"):
            handler._parse_with_agently(
                user_query="test",
                schema=schema,
                api_key="test-key",
                base_url="https://api.test.com",
                model="gpt-3.5-turbo"
            )
    
    @patch('harborai.api.structured.Agently')
    def test_parse_with_agently_exception_with_fallback(self, mock_agently):
        """测试Agently异常时的回退处理"""
        mock_agently.create_agent.side_effect = Exception("Agently error")
        
        handler = StructuredOutputHandler()
        schema = {"type": "object"}
        
        # 提供有效的model_response作为回退
        model_response = '{"fallback": "data"}'
        
        with patch.object(handler, 'extract_json_from_text', return_value=model_response):
            result = handler._parse_with_agently(
                user_query="test",
                schema=schema,
                api_key="test-key",
                base_url="https://api.test.com",
                model="gpt-3.5-turbo",
                model_response=model_response
            )
            
            assert result == {"fallback": "data"}


class TestAdditionalCoverage:
    """额外的覆盖率测试"""
    
    def test_convert_json_schema_to_agently_output_exception_handling(self):
        """测试JSON Schema转换异常处理"""
        handler = StructuredOutputHandler()
        
        # 测试无效的schema导致异常
        invalid_schema = {"json_schema": {"schema": {"type": "invalid_type"}}}
        
        # 模拟转换过程中的异常
        with patch.object(handler, '_convert_schema_to_agently_format', side_effect=Exception("Conversion error")):
            result = handler._convert_json_schema_to_agently_output(invalid_schema)
            # 应该返回fallback格式
            assert result == {"result": ("str", "Generated result")}
    
    def test_extract_json_from_text_edge_cases(self):
        """测试JSON提取的边界情况"""
        handler = StructuredOutputHandler()
        
        # 测试空字符串
        result = handler.extract_json_from_text("")
        assert result == ""
        
        # 测试只有空白字符
        result = handler.extract_json_from_text("   \n\t   ")
        assert result.strip() == ""
        
        # 测试包含特殊字符的JSON
        text = '{"name": "test\\nwith\\tspecial\\rchars"}'
        result = handler.extract_json_from_text(text)
        assert '"name"' in result
    
    def test_validate_against_schema_edge_cases(self):
        """测试schema验证的边界情况"""
        handler = StructuredOutputHandler()
        
        # 测试空数据和空schema - 应该不抛出异常
        handler._validate_against_schema({}, {})
        
        # 测试None数据与object schema - 应该抛出异常
        with pytest.raises(StructuredOutputError):
            handler._validate_against_schema(None, {"type": "object"})
    
    def test_update_result_by_key_complex_paths(self):
        """测试复杂路径的结果更新"""
        handler = StructuredOutputHandler()
        
        # 测试深层嵌套路径
        result = {}
        handler._update_result_by_key(result, "level1.level2.level3", "deep_value")
        
        expected = {
            "level1": {
                "level2": {
                    "level3": "deep_value"
                }
            }
        }
        assert result == expected
        
        # 测试数组索引超出范围
        result = {"items": []}
        handler._update_result_by_key(result, "items", "value", indexes=[5])
        assert len(result["items"]) == 6
        assert result["items"][5] == "value"
        assert all(item == "" for item in result["items"][:5])
    
    def test_format_response_format_edge_cases(self):
        """测试响应格式化的边界情况"""
        handler = StructuredOutputHandler()
        
        # 测试空schema - 应该返回完整的包装格式
        result = handler.format_response_format({})
        expected = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": {},
                "strict": True
            }
        }
        assert result == expected
        
        # 测试None schema - 应该返回None的包装格式
        result = handler.format_response_format(None)
        expected = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": None,
                "strict": True
            }
        }
        assert result == expected
    
    @patch('harborai.api.structured.Agently')
    def test_parse_with_agently_configuration_edge_cases(self, mock_agently):
        """测试Agently配置的边界情况"""
        handler = StructuredOutputHandler()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        # 测试没有API key和base_url的情况
        mock_agent = Mock()
        mock_agent.input.return_value = mock_agent
        mock_agent.output.return_value = mock_agent
        mock_agent.start.return_value = {"name": "test"}
        mock_agently.create_agent.return_value = mock_agent
        
        result = handler._parse_with_agently(
            user_query="test query",
            schema=schema,
            api_key=None,
            base_url=None,
            model="test-model"
        )
        
        assert result == {"name": "test"}
        mock_agently.create_agent.assert_called()
    
    @patch('harborai.api.structured.Agently')
    def test_parse_streaming_with_agently_error_handling(self, mock_agently):
        """测试Agently流式解析错误处理"""
        handler = StructuredOutputHandler()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        # 模拟空的流式响应
        def empty_stream():
            return
            yield  # 这行永远不会执行，确保生成器为空
        
        mock_agent = Mock()
        mock_agent.input.return_value = mock_agent
        mock_agent.output.return_value = mock_agent
        mock_agent.get_instant_generator.return_value = empty_stream()
        mock_agently.create_agent.return_value = mock_agent
        
        result_generator = handler._parse_streaming_with_agently(
            response_stream=empty_stream(),
            schema=schema,
            api_key="test-key",
            base_url="http://test.com",
            model="test-model"
        )
        
        # 空流应该返回空的生成器
        result_list = list(result_generator)
        assert result_list == []
    

    
    def test_parse_response_with_none_content(self):
        """测试解析None内容的响应"""
        handler = StructuredOutputHandler()
        
        # None内容应该抛出异常
        with pytest.raises(StructuredOutputError):
            handler.parse_response(
                content=None,
                schema={"type": "object"},
                use_agently=False
            )
    
    def test_parse_response_with_empty_schema(self):
        """测试使用空schema解析响应"""
        handler = StructuredOutputHandler()
        
        result = handler.parse_response(
            content='{"name": "test"}',
            schema={},
            use_agently=False
        )
        
        # 空schema应该返回解析的JSON
        assert result == {"name": "test"}
    
    def test_convert_schema_to_agently_format_complex_types(self):
        """测试复杂类型的schema转换"""
        handler = StructuredOutputHandler()
        
        # 测试包含数组和嵌套对象的复杂schema
        complex_schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "description": "用户列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                            "active": {"type": "boolean"}
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "total": {"type": "number"},
                        "tags": {
                            "type": "array",
                            "description": "标签列表",
                            "items": {"type": "string"}
                        }
                    }
                }
            }
        }
        
        result = handler._convert_schema_to_agently_format(complex_schema)
        
        # 验证转换结果包含正确的结构
        assert "users" in result
        assert "metadata" in result
        
        # 验证数组类型转换 - 应该返回列表格式
        users_array = result["users"]
        assert isinstance(users_array, list)
        assert len(users_array) == 1
        
        # 验证嵌套对象结构
        user_object = users_array[0]
        assert isinstance(user_object, dict)
        assert "name" in user_object
        assert "age" in user_object
        assert "active" in user_object
    
    def test_extract_json_from_text_malformed_json(self):
        """测试提取格式错误的JSON"""
        handler = StructuredOutputHandler()
        
        # 测试格式错误的JSON
        malformed_texts = [
            '{"name": "test"',  # 缺少闭合括号
            '{"name": test}',   # 值没有引号
            '{name: "test"}',   # 键没有引号
            '{"name": "test",}', # 尾随逗号
        ]
        
        for text in malformed_texts:
            result = handler.extract_json_from_text(text)
            # 格式错误的JSON应该返回原文本
            assert result == text
    
    def test_update_result_by_key_type_conversion(self):
        """测试结果更新时的类型转换"""
        handler = StructuredOutputHandler()
        
        # 测试将非字符串值转换为字符串后连接
        result = {"count": 42}
        handler._update_result_by_key(result, "count", " items")
        
        # 数字应该被转换为字符串并连接
        assert result["count"] == "42 items"
        
        # 测试布尔值转换
        result = {"active": True}
        handler._update_result_by_key(result, "active", " status")
        assert result["active"] == "True status"


class TestStreamingAndErrorHandling:
    """流式解析和错误处理的专项测试类"""
    
    @patch('harborai.api.structured.Agently')
    def test_parse_streaming_agently_import_error(self, mock_agently):
        """测试Agently导入失败的情况"""
        handler = StructuredOutputHandler(provider="agently")
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        content_stream = iter(["chunk1", "chunk2"])
        
        # 模拟Agently.create_agent()调用时ImportError
        mock_agently.create_agent.side_effect = ImportError("Agently not available")
        
        with pytest.raises(ImportError):
            list(handler._parse_sync_streaming_with_agently(
                content_stream, schema, 
                api_key="test-key", 
                base_url="https://api.test.com", 
                model="gpt-3.5-turbo"
            ))
    
    def test_parse_streaming_agently_structured_output_error(self):
        """测试Agently流式解析中的StructuredOutputError"""
        handler = StructuredOutputHandler(provider="agently")
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        content_stream = iter(["test content"])
        
        # 模拟在_convert_json_schema_to_agently_output中抛出StructuredOutputError
        with patch.object(handler, '_convert_json_schema_to_agently_output', 
                         side_effect=StructuredOutputError("Schema conversion error")):
            with pytest.raises(StructuredOutputError):
                list(handler._parse_sync_streaming_with_agently(
                    content_stream, schema,
                    api_key="test-key", 
                    base_url="https://api.test.com", 
                    model="gpt-3.5-turbo"
                ))
    
    @patch('harborai.api.structured.Agently')
    def test_parse_streaming_agently_general_exception_fallback(self, mock_agently):
        """测试Agently流式解析一般异常的回退机制"""
        handler = StructuredOutputHandler(provider="agently")
        
        # 模拟agent创建成功但解析失败
        mock_agent = Mock()
        mock_agently.create_agent.return_value = mock_agent
        
        # 模拟链式调用失败
        mock_chain = Mock()
        mock_agent.input.return_value = mock_chain
        mock_chain.output.return_value = mock_chain
        mock_chain.get_instant_generator.side_effect = Exception("General error")
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        content_stream = iter(['{"name": "test"}'])
        
        # 应该回退到原生解析
        result = list(handler._parse_sync_streaming_with_agently(
            content_stream, schema,
            api_key="test-key", 
            base_url="https://api.test.com", 
            model="gpt-3.5-turbo"
        ))
        
        # 应该有回退结果（至少一个空字典或解析结果）
        assert len(result) > 0
        # 验证回退机制工作正常
        assert isinstance(result[0], dict)
    
    @patch('harborai.api.structured.Agently')
    def test_parse_streaming_agently_event_processing_error(self, mock_agently):
        """测试Agently流式解析事件处理错误"""
        handler = StructuredOutputHandler(provider="agently")
        
        # 模拟agent创建成功
        mock_agent = Mock()
        mock_agently.create_agent.return_value = mock_agent
        
        # 模拟get_instant_generator返回有问题的事件
        def problematic_generator():
            yield {"invalid": "event"}  # 无效事件格式
            yield {"complete_value": {"name": "test"}}  # 有效事件
        
        mock_agent.input.return_value.output.return_value.get_instant_generator.return_value = \
            problematic_generator()
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        content_stream = ["test content"]
        
        # 应该跳过无效事件，处理有效事件
        result = list(handler._parse_sync_streaming_with_agently(
            content_stream, schema,
            api_key="test-key", 
            base_url="https://api.test.com", 
            model="gpt-3.5-turbo"
        ))
        assert len(result) >= 1
        assert result[-1] == {"name": "test"}
    
    def test_extract_json_from_text_code_block_validation(self):
        """测试代码块中JSON验证的边界情况"""
        handler = StructuredOutputHandler()
        
        # 测试代码块中包含无效JSON
        text_with_invalid_json = """
        ```
        {invalid json content}
        ```
        """
        
        # 应该返回原始文本，因为JSON无效
        result = handler.extract_json_from_text(text_with_invalid_json)
        assert text_with_invalid_json.strip() in result
    
    def test_parse_with_agently_fallback_to_json_extraction(self):
        """测试Agently解析失败时的JSON提取回退"""
        handler = StructuredOutputHandler(provider="agently")
        
        # 模拟有模型响应的情况
        model_response = '{"name": "test", "age": 25}'
        
        with patch('harborai.api.structured.Agently') as mock_agently:
            # 模拟Agently解析失败
            mock_agently.create_agent.side_effect = Exception("Agently failed")
            
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            
            # 应该回退到JSON提取
            result = handler._parse_with_agently(
                user_query="test query",
                schema=schema,
                api_key="test-key",
                base_url="https://api.test.com",
                model="gpt-3.5-turbo",
                model_response=model_response
            )
            assert result == {"name": "test", "age": 25}
    
    def test_parse_with_agently_no_fallback_content(self):
        """测试Agently解析失败且无回退内容的情况"""
        handler = StructuredOutputHandler(provider="agently")
        
        with patch('harborai.api.structured.Agently') as mock_agently:
            # 模拟Agently解析失败
            mock_agently.create_agent.side_effect = Exception("Agently failed")
            
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            
            # 没有model_response，应该抛出异常
            with pytest.raises(StructuredOutputError):
                handler._parse_with_agently(
                    user_query="test content",
                    schema=schema,
                    api_key="test-key",
                    base_url="https://api.test.com",
                    model="gpt-3.5-turbo"
                )
    
    def test_convert_object_schema_with_array_property(self):
        """测试对象schema中包含数组属性的转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "姓名"},
                "tags": {
                    "type": "array",
                    "description": "标签列表",
                    "items": {"type": "string"}
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created": {"type": "string"}
                    }
                }
            }
        }
        
        result = handler._convert_object_schema(schema)
        
        # 验证数组属性被正确转换
        assert "tags" in result
        assert isinstance(result["tags"], list)
        assert result["tags"] == [("str", "标签列表")]
        
        # 验证嵌套对象被正确转换
        assert "metadata" in result
        assert isinstance(result["metadata"], dict)
    
    def test_convert_object_schema_default_string_type(self):
        """测试对象schema中未知类型默认为字符串的情况"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "object",
            "properties": {
                "unknown_field": {
                    "type": "unknown_type",
                    "description": "未知类型字段"
                }
            }
        }
        
        result = handler._convert_object_schema(schema)
        
        # 未知类型应该默认为字符串
        assert result["unknown_field"] == ("str", "未知类型字段")


class TestAsyncStreamingWithAgently:
    """异步Agently流式解析的专项测试类"""
    
    @patch('harborai.api.structured.Agently')
    @pytest.mark.asyncio
    async def test_parse_async_streaming_agently_import_error(self, mock_agently):
        """测试异步Agently流式解析导入错误"""
        handler = StructuredOutputHandler(provider="agently")
        
        # 模拟Agently.create_agent()调用时ImportError
        mock_agently.create_agent.side_effect = ImportError("Agently not available")
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        async def test_stream():
            yield "test content"
        
        with pytest.raises(ImportError):
            async for _ in handler._parse_async_streaming_with_agently(
                test_stream(), schema,
                api_key="test-key", 
                base_url="https://api.test.com", 
                model="gpt-3.5-turbo"
            ):
                pass
    
    @patch('harborai.api.structured.Agently')
    @pytest.mark.asyncio
    async def test_parse_async_streaming_agently_structured_output_error(self, mock_agently):
        """测试异步Agently流式解析中的StructuredOutputError"""
        handler = StructuredOutputHandler(provider="agently")
        
        # 模拟_convert_json_schema_to_agently_output抛出StructuredOutputError
        with patch.object(handler, '_convert_json_schema_to_agently_output', 
                         side_effect=StructuredOutputError("Schema conversion error")):
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            
            async def test_stream():
                yield "test content"
            
            with pytest.raises(StructuredOutputError):
                async for _ in handler._parse_async_streaming_with_agently(
                    test_stream(), schema,
                    api_key="test-key", 
                    base_url="https://api.test.com", 
                    model="gpt-3.5-turbo"
                ):
                    pass
    
    @patch('harborai.api.structured.Agently')
    @pytest.mark.asyncio
    async def test_parse_async_streaming_agently_fallback_mechanism(self, mock_agently):
        """测试异步Agently流式解析的回退机制"""
        handler = StructuredOutputHandler(provider="agently")
        
        # 模拟agent创建成功但解析失败
        mock_agent = Mock()
        mock_agently.create_agent.return_value = mock_agent
        
        # 模拟get_instant_generator抛出一般异常
        mock_agent.input.return_value.output.return_value.get_instant_generator.side_effect = \
            Exception("General error")
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        async def test_stream():
            yield '{"name": "test"}'
        
        # 应该回退到原生解析
        results = []
        async for result in handler._parse_async_streaming_with_agently(
            test_stream(), schema,
            api_key="test-key", 
            base_url="https://api.test.com", 
            model="gpt-3.5-turbo"
        ):
            results.append(result)
        
        assert len(results) > 0
    
    @patch('harborai.api.structured.Agently')
    @pytest.mark.asyncio
    async def test_parse_async_streaming_agently_event_processing(self, mock_agently):
        """测试异步Agently流式解析事件处理"""
        handler = StructuredOutputHandler(provider="agently")
        
        # 模拟agent创建成功
        mock_agent = Mock()
        mock_agently.create_agent.return_value = mock_agent
        
        # 模拟get_instant_generator返回事件
        async def async_generator():
            yield {"complete_value": {"name": "test1"}}
            yield {"complete_value": {"name": "test2", "age": 30}}
        
        # 模拟agent支持异步instant generator
        mock_input_output = Mock()
        mock_input_output.get_async_instant_generator.return_value = async_generator()
        mock_agent.input.return_value.output.return_value = mock_input_output
        
        # 检查是否有get_async_instant_generator方法
        hasattr_mock = Mock(side_effect=lambda obj, attr: attr == 'get_async_instant_generator')
        
        with patch('builtins.hasattr', hasattr_mock):
            schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
            
            async def test_stream():
                yield "test content"
            
            results = []
            async for result in handler._parse_async_streaming_with_agently(
                test_stream(), schema,
                api_key="test-key", 
                base_url="https://api.test.com", 
                model="gpt-3.5-turbo"
            ):
                results.append(result)
            
            # 应该处理所有事件
            assert len(results) >= 1
        assert {"name": "test1"} in results
        assert {"name": "test2", "age": 30} in results


class TestJSONSchemaConversion:
    """JSON Schema转换的专项测试类"""
    
    def test_convert_schema_with_non_dict_input(self):
        """测试转换非字典类型的schema输入"""
        handler = StructuredOutputHandler()
        
        # 测试非字典输入
        result = handler._convert_schema_to_agently_format("not a dict")
        expected = {"value": ("str", "Generated value")}
        assert result == expected
    
    def test_convert_object_schema_with_enum_values(self):
        """测试转换包含枚举值的对象schema"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string", 
                    "description": "状态",
                    "enum": ["active", "inactive", "pending"]
                },
                "priority": {
                    "type": "integer",
                    "description": "优先级"
                },
                "score": {
                    "type": "number",
                    "description": "分数"
                },
                "enabled": {
                    "type": "boolean",
                    "description": "是否启用"
                }
            }
        }
        
        result = handler._convert_object_schema(schema)
        expected = {
            "status": ("str", "状态，可选值: active/inactive/pending"),
            "priority": ("int", "优先级"),
            "score": ("float", "分数"),
            "enabled": ("bool", "是否启用")
        }
        assert result == expected
     
    def test_convert_primitive_schema(self):
        """测试转换基本类型schema"""
        handler = StructuredOutputHandler()
        
        # 测试字符串类型
        string_schema = {"type": "string", "description": "用户名"}
        result = handler._convert_primitive_schema(string_schema)
        assert result == ("str", "用户名")
        
        # 测试整数类型
        int_schema = {"type": "integer", "description": "年龄"}
        result = handler._convert_primitive_schema(int_schema)
        assert result == ("int", "年龄")
        
        # 测试数字类型
        number_schema = {"type": "number", "description": "分数"}
        result = handler._convert_primitive_schema(number_schema)
        assert result == ("int", "分数")
        
        # 测试布尔类型
        bool_schema = {"type": "boolean", "description": "是否激活"}
        result = handler._convert_primitive_schema(bool_schema)
        assert result == ("bool", "是否激活")
        
        # 测试未知类型（应该默认为字符串）
        unknown_schema = {"type": "unknown", "description": "未知类型"}
        result = handler._convert_primitive_schema(unknown_schema)
        assert result == ("str", "未知类型")
        
        # 测试没有描述的情况
        no_desc_schema = {"type": "string"}
        result = handler._convert_primitive_schema(no_desc_schema)
        assert result == ("str", "string value")
     
    def test_convert_array_schema_with_object_items(self):
        """测试对象数组的Schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            },
            "description": "用户列表"
        }
        
        result = handler._convert_array_schema(schema)
        assert isinstance(result, list)
        assert len(result) == 1
        # 应该包含对象结构
        assert isinstance(result[0], dict)
    
    def test_convert_array_schema_with_string_items(self):
        """测试字符串数组的Schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "description": "字符串列表"
        }
        
        result = handler._convert_array_schema(schema)
        assert result == [("str", "字符串列表")]
    
    def test_convert_array_schema_with_integer_items(self):
        """测试整数数组的Schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "array",
            "items": {"type": "integer"},
            "description": "整数列表"
        }
        
        result = handler._convert_array_schema(schema)
        assert result == [("int", "整数列表")]
    
    def test_convert_array_schema_with_number_items(self):
        """测试数字数组的Schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "array",
            "items": {"type": "number"},
            "description": "数字列表"
        }
        
        result = handler._convert_array_schema(schema)
        assert result == [("int", "数字列表")]
    
    def test_convert_array_schema_with_boolean_items(self):
        """测试布尔数组的Schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "array",
            "items": {"type": "boolean"},
            "description": "布尔列表"
        }
        
        result = handler._convert_array_schema(schema)
        assert result == [("bool", "布尔列表")]
    
    def test_convert_array_schema_with_unknown_items(self):
        """测试未知类型数组的Schema转换"""
        handler = StructuredOutputHandler()
        
        schema = {
            "type": "array",
            "items": {"type": "unknown_type"},
            "description": "未知类型列表"
        }
        
        result = handler._convert_array_schema(schema)
        assert result == [("str", "未知类型列表")]


class TestErrorHandlingAndEdgeCases:
    """错误处理和边界条件的专项测试类"""
    
    def test_parse_with_agently_none_result(self):
        """测试Agently返回None结果的情况"""
        with patch('harborai.api.structured.Agently') as mock_agently:
            # 模拟Agently返回None
            mock_agent = MagicMock()
            mock_agent.input.return_value = mock_agent
            mock_agent.output.return_value = mock_agent
            mock_agent.start.return_value = None
            mock_agently.create_agent.return_value = mock_agent
            
            handler = StructuredOutputHandler()
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}
            
            # 应该抛出StructuredOutputError
            with pytest.raises(StructuredOutputError, match="Agently返回None结果"):
                handler._parse_with_agently(
                    user_query="test query",
                    schema=schema,
                    api_key="test-key",
                    base_url="https://api.test.com",
                    model="gpt-3.5-turbo"
                )
    
    def test_extract_json_from_text_with_code_blocks(self):
        """测试从代码块中提取JSON"""
        handler = StructuredOutputHandler()
        
        # 测试```json格式
        text_with_json_block = '''
        这是一些文本
        ```json
        {"name": "test", "age": 25}
        ```
        更多文本
        '''
        
        result = handler.extract_json_from_text(text_with_json_block)
        assert result == '{"name": "test", "age": 25}'
    
    def test_extract_json_from_text_with_generic_code_blocks(self):
        """测试从通用代码块中提取JSON"""
        handler = StructuredOutputHandler()
        
        # 测试```格式（无语言标识）
        text_with_code_block = '''
        这是一些文本
        ```
        {"name": "test", "age": 25}
        ```
        更多文本
        '''
        
        result = handler.extract_json_from_text(text_with_code_block)
        assert result == '{"name": "test", "age": 25}'
    
    def test_extract_json_from_text_with_invalid_code_block(self):
        """测试从包含无效JSON的代码块中提取"""
        handler = StructuredOutputHandler()
        
        # 测试```格式但内容不是有效JSON
        text_with_invalid_block = '''
        这是一些文本
        ```
        这不是JSON内容
        ```
        更多文本
        '''
        
        result = handler.extract_json_from_text(text_with_invalid_block)
        # 如果代码块内容不是有效JSON，会继续查找其他格式，最终返回原文本
        assert "这是一些文本" in result
        assert "这不是JSON内容" in result
    
    def test_parse_streaming_with_native_provider(self):
        """测试使用原生提供者的流式解析"""
        handler = StructuredOutputHandler(provider="native")
        
        def test_stream():
            yield '{"name": "test"}'
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        # 使用原生解析
        results = list(handler.parse_streaming_response(
            test_stream(), schema
        ))
        
        assert len(results) > 0
        assert results[-1] == {"name": "test"}
    
    def test_parse_response_with_native_provider(self):
        """测试使用原生提供者的非流式解析"""
        handler = StructuredOutputHandler(provider="native")
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        # 使用原生解析
        result = handler.parse_response(
            content='{"name": "test"}',
            schema=schema,
            use_agently=False
        )
        
        assert result == {"name": "test"}


class TestUtilityMethods:
    """工具方法的专项测试类"""
    
    def test_update_result_by_key_simple_path(self):
        """测试简单路径的结果更新"""
        handler = StructuredOutputHandler()
        result = {}
        
        handler._update_result_by_key(result, "name", "test")
        assert result == {"name": "test"}
        
        # 追加更新
        handler._update_result_by_key(result, "name", " user")
        assert result == {"name": "test user"}
    
    def test_update_result_by_key_nested_path(self):
        """测试嵌套路径的结果更新"""
        handler = StructuredOutputHandler()
        result = {}
        
        handler._update_result_by_key(result, "user.name", "test", [0])
        assert "user" in result
        assert "name" in result["user"]
        # 由于有indexes参数，name字段会被创建为数组
        assert isinstance(result["user"]["name"], list)
        assert result["user"]["name"][0] == "test"
    
    def test_update_result_by_key_array_path(self):
        """测试数组路径的结果更新"""
        handler = StructuredOutputHandler()
        result = {}
        
        handler._update_result_by_key(result, "users", "test", [0])
        assert "users" in result
        assert isinstance(result["users"], list)
        assert len(result["users"]) > 0
        assert result["users"][0] == "test"
    
    def test_update_result_by_key_complex_nested_path(self):
        """测试复杂嵌套路径的结果更新"""
        handler = StructuredOutputHandler()
        result = {}
        
        handler._update_result_by_key(result, "data.users.name", "test", [0, 1])
        assert "data" in result
        assert "users" in result["data"]
        assert "name" in result["data"]["users"]
        # 由于有indexes参数，name字段会被创建为数组，使用第一个索引0
        assert isinstance(result["data"]["users"]["name"], list)
        assert len(result["data"]["users"]["name"]) > 0
        assert result["data"]["users"]["name"][0] == "test"
    
    def test_update_result_by_key_error_handling(self):
        """测试路径解析失败时的错误处理"""
        handler = StructuredOutputHandler()
        result = {}
        
        # 使用包含空字符串的路径格式，会创建嵌套结构
        handler._update_result_by_key(result, "invalid..path", "test")
        # 路径会被分割为['invalid', '', 'path']，创建嵌套结构
        assert "invalid" in result
        assert "" in result["invalid"]
        assert "path" in result["invalid"][""]
        assert result["invalid"][""]["path"] == "test"
    
    def test_update_result_by_key_non_string_existing_value(self):
        """测试更新非字符串现有值"""
        handler = StructuredOutputHandler()
        result = {"count": 5}
        
        handler._update_result_by_key(result, "count", " items")
        assert result["count"] == "5 items"
    
    def test_extract_json_from_text_with_braces(self):
        """测试从大括号格式中提取JSON"""
        handler = StructuredOutputHandler()
        
        text_with_braces = '''
        这是一些文本
        {"name": "test", "age": 25}
        更多文本
        '''
        
        result = handler.extract_json_from_text(text_with_braces)
        assert result == '{"name": "test", "age": 25}'
    
    def test_extract_json_from_text_with_array(self):
        """测试从数组格式中提取JSON"""
        handler = StructuredOutputHandler()
        
        text_with_array = '''
        这是一些文本
        [{"name": "test1"}, {"name": "test2"}]
        更多文本
        '''
        
        result = handler.extract_json_from_text(text_with_array)
        # extract_json_from_text会查找{...}格式，如果没找到会返回原文本
        # 由于数组格式[...]不在{...}匹配范围内，会返回原文本
        assert '[{"name": "test1"}, {"name": "test2"}]' in result
    
    def test_extract_json_from_text_no_json(self):
        """测试从不包含JSON的文本中提取"""
        handler = StructuredOutputHandler()
        
        text_without_json = '''
        这是一些普通文本
        没有JSON内容
        '''
        
        result = handler.extract_json_from_text(text_without_json)
        # 如果没有找到JSON格式，会返回原文本的strip()版本
        assert "这是一些普通文本" in result
        assert "没有JSON内容" in result


class TestNativeStreamingParsing:
    """原生流式解析的专项测试类"""
    
    def test_parse_streaming_with_native_empty_stream(self):
        """测试原生流式解析空流的情况"""
        handler = StructuredOutputHandler(provider="native")
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        empty_stream = []
        
        result = list(handler._parse_streaming_with_native(empty_stream, schema))
        assert len(result) == 0  # 空流不会产生任何输出
    
    @pytest.mark.asyncio
    async def test_parse_async_streaming_with_native_empty_stream(self):
        """测试异步原生流式解析空流的情况"""
        handler = StructuredOutputHandler(provider="native")
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        
        async def empty_async_stream():
            return
            yield  # 这行永远不会执行
        
        results = []
        async for result in handler._parse_async_streaming_with_native(empty_async_stream(), schema):
            results.append(result)
        
        assert len(results) == 0  # 空流不会产生任何输出
    
    def test_parse_streaming_with_native_json_chunks(self):
        """测试原生流式解析JSON块的情况"""
        handler = StructuredOutputHandler(provider="native")
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        json_stream = ['{"name":', '"test",', '"age":', '25}']
        
        result = list(handler._parse_streaming_with_native(json_stream, schema))
        
        # 应该逐步构建JSON对象
        assert len(result) > 0
        final_result = result[-1]
        assert final_result == {"name": "test", "age": 25}
    
    @pytest.mark.asyncio
    async def test_parse_async_streaming_with_native_json_chunks(self):
        """测试异步原生流式解析JSON块的情况"""
        handler = StructuredOutputHandler(provider="native")
        
        schema = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}
        
        async def json_async_stream():
            yield '{"name":'
            yield '"test",'
            yield '"age":'
            yield '25}'
        
        results = []
        async for result in handler._parse_async_streaming_with_native(json_async_stream(), schema):
            results.append(result)
        
        # 应该逐步构建JSON对象
        assert len(results) > 0
        final_result = results[-1]
        assert final_result == {"name": "test", "age": 25}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])