"""结构化输出支持模块。

支持基于Agently的流式和非流式结构化输出解析。
"""

from agently import Agently
import json
import jsonschema
import re
from typing import Any, Dict, Optional, Type, Union, Generator, AsyncGenerator

from ..utils.logger import get_logger
from ..utils.exceptions import StructuredOutputError

logger = get_logger(__name__)


class StructuredOutputHandler:
    """结构化输出处理器。"""
    
    def __init__(self, provider: str = "agently"):
        """初始化结构化输出处理器。
        
        Args:
            provider: 解析提供者，"agently" 或 "native"
        """
        self.provider = provider
        self.logger = get_logger(__name__)
        self._agently_available = self._check_agently_availability()
    
    def _check_agently_availability(self) -> bool:
        """检查Agently是否可用。
        
        Returns:
            bool: Agently是否可用
        """
        try:
             # 检查Agently版本是否支持所需功能
             if hasattr(Agently, 'create_agent'):
                 return True
             else:
                 logger.warning("Agently version does not support required features")
                 return False
        except ImportError:
            logger.debug("Agently not installed")
            return False
        except Exception as e:
            logger.warning(f"Agently availability check failed: {e}")
            return False
    

    def _convert_json_schema_to_agently_output(self, schema_wrapper: Dict[str, Any]) -> Dict[str, Any]:
        """将JSON Schema转换为Agently的output格式。
        
        根据Agently结构化输出语法设计理念，将JSON Schema转换为Agently的元组表达式格式。
        
        Args:
            schema_wrapper: 包含JSON Schema的包装器，格式为 {"json_schema": {"schema": actual_schema}}
            
        Returns:
            转换后的Agently output格式字典
        """
        try:
            # 提取实际的schema
            if "json_schema" in schema_wrapper and "schema" in schema_wrapper["json_schema"]:
                schema = schema_wrapper["json_schema"]["schema"]
            else:
                schema = schema_wrapper
            
            return self._convert_schema_to_agently_format(schema)
            
        except Exception as e:
            self.logger.error(f"Failed to convert JSON Schema to Agently format: {e}")
            # 返回一个基本的格式作为fallback
            return {"result": ("str", "Generated result")}
    
    def _convert_schema_to_agently_format(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """递归转换JSON Schema到Agently格式。
        
        Args:
            schema: JSON Schema定义
            
        Returns:
            Agently格式的字典
        """
        if not isinstance(schema, dict):
            return {"value": ("str", "Generated value")}
        
        schema_type = schema.get("type", "object")
        
        if schema_type == "object":
            return self._convert_object_schema(schema)
        elif schema_type == "array":
            return self._convert_array_schema(schema)
        else:
            # 基本类型
            return self._convert_primitive_schema(schema)
    
    def _convert_object_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """转换object类型的JSON Schema到Agently格式。
        
        Args:
            schema: object类型的JSON Schema
            
        Returns:
            Agently格式的字典
        """
        result = {}
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])
        
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "string")
            description = prop_schema.get("description", f"{prop_name} field")
            enum_values = prop_schema.get("enum")
            
            # 根据Agently语法，使用元组表达式
            if prop_type == "string":
                # 如果有enum约束，在描述中包含可选值
                if enum_values:
                    enum_str = "/".join(enum_values)
                    enhanced_description = f"{description}，可选值: {enum_str}"
                    result[prop_name] = ("str", enhanced_description)
                else:
                    result[prop_name] = ("str", description)
            elif prop_type == "integer":
                result[prop_name] = ("int", description)
            elif prop_type == "number":
                result[prop_name] = ("float", description)
            elif prop_type == "boolean":
                result[prop_name] = ("bool", description)
            elif prop_type == "object":
                # 递归处理嵌套对象
                result[prop_name] = self._convert_object_schema(prop_schema)
            elif prop_type == "array":
                # 处理数组类型
                result[prop_name] = self._convert_array_schema(prop_schema)
            else:
                # 默认为字符串类型
                result[prop_name] = ("str", description)
        
        return result
    
    def _convert_array_schema(self, schema: Dict[str, Any]) -> list:
        """转换array类型的JSON Schema到Agently格式。
        
        Args:
            schema: array类型的JSON Schema
            
        Returns:
            Agently格式的列表
        """
        items_schema = schema.get("items", {"type": "string"})
        # 使用数组本身的描述，而不是items的描述
        array_description = schema.get("description", "Array item")
        
        if isinstance(items_schema, dict):
            item_type = items_schema.get("type", "string")
            
            if item_type == "object":
                # 对象数组
                return [self._convert_object_schema(items_schema)]
            elif item_type == "string":
                return [("str", array_description)]
            elif item_type == "integer" or item_type == "number":
                return [("int", array_description)]
            elif item_type == "boolean":
                return [("bool", array_description)]
            else:
                return [("str", array_description)]
        else:
            # 如果items不是字典，默认为字符串数组
            return [("str", array_description)]
    
    def _convert_primitive_schema(self, schema: Dict[str, Any]) -> tuple:
        """转换基本类型的JSON Schema到Agently格式。
        
        Args:
            schema: 基本类型的JSON Schema
            
        Returns:
            Agently格式的元组
        """
        schema_type = schema.get("type", "string")
        description = schema.get("description", f"{schema_type} value")
        
        if schema_type == "string":
            return ("str", description)
        elif schema_type == "integer" or schema_type == "number":
            return ("int", description)
        elif schema_type == "boolean":
            return ("bool", description)
        else:
             return ("str", description)
    

    
    def parse_response(self, content: str, schema: Dict[str, Any], use_agently: bool = False, api_key: str = None, base_url: str = None, model: str = None, user_query: str = None) -> Dict[str, Any]:
        """解析响应内容为结构化数据。
        
        Args:
            content: 要解析的内容（模型响应或用户查询）
            schema: JSON Schema定义
            use_agently: 是否使用Agently进行解析
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            user_query: 用户的原始查询（使用Agently时必需）
            
        Returns:
            解析后的结构化数据
        """
        try:
            self.logger.debug(f"parse_response调用: use_agently={use_agently}, api_key={api_key[:10] if api_key else None}..., base_url={base_url}, model={model}")
            if use_agently:
                try:
                    # 如果使用Agently但没有提供user_query，使用content作为查询
                    query = user_query if user_query else content
                    return self._parse_with_agently(query, schema, api_key, base_url, model, content)
                except ImportError as e:
                    # Agently库不可用，回退到原生解析
                    self.logger.warning(f"Agently library not available, falling back to native: {e}")
                    # 回退到原生解析
                    pass
                except StructuredOutputError as e:
                    # API密钥错误或其他结构化输出错误，不回退，直接抛出
                    self.logger.error(f"Agently parsing failed with StructuredOutputError: {e}")
                    raise
            
            # 使用原生解析
            return self._parse_with_native(content, schema)
            
        except Exception as e:
            self.logger.error(f"All parsing methods failed: {e}")
            raise StructuredOutputError(f"Failed to parse response: {e}")
    

    
    def _parse_with_agently(self, user_query: str, schema: Dict[str, Any], api_key: str, base_url: str, model: str, model_response: str = None) -> Any:
        """
        使用Agently进行结构化解析
        
        根据Agently结构化输出语法设计理念，使用.input().output().start()方法进行非流式结构化输出。
        注意：这里不是解析已有内容，而是让Agently根据原始问题重新生成结构化输出。
        
        Args:
            user_query: 用户的原始查询
            schema: JSON Schema定义
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            model_response: 模型返回的原始内容（可选，用于fallback）
            
        Returns:
            解析后的结构化数据
        """
        try:
            
            self.logger.debug(f"使用Agently生成结构化输出，输入: {user_query[:100]}...")
            
            # 调试：打印传入的参数
            self.logger.debug(f"_parse_with_agently参数: api_key={api_key[:10] if api_key else None}..., base_url={base_url}, model={model}")
            
            # 确保model不为None，但保持原始模型名称
            if not model:
                raise StructuredOutputError("模型名称不能为空")
            
            # 将JSON Schema转换为Agently output格式
            agently_format = self._convert_json_schema_to_agently_output({"json_schema": {"schema": schema}})
            
            self.logger.debug(f"转换后的Agently格式: {agently_format}")
            
            # 使用测试验证的正确配置方法
            if api_key and base_url:
                # 根据测试结果，使用正确的 Agently.set_settings 全局配置方式
                self.logger.debug(f"配置 Agently: base_url={base_url}, model={model}")
                
                # 使用 OpenAICompatible 全局配置（基于测试成功的方法）
                Agently.set_settings(
                    "OpenAICompatible",
                    {
                        "base_url": base_url,
                        "model": model,
                        "model_type": "chat",
                        "auth": api_key,
                    },
                )
                
                self.logger.debug(f"Agently 全局配置完成: URL={base_url}, model={model}")
                
                agent = Agently.create_agent()
            else:
                # 默认配置
                agent = Agently.create_agent()
                self.logger.debug(f"使用默认Agently配置")
            
            self.logger.debug(f"Agently agent创建完成，准备进行结构化输出")
            
            # 根据Agently结构化输出语法设计理念，使用.input().output().start()进行非流式结构化输出
            # user_query是用户的原始问题，而不是模型的响应
            result = (
                agent
                .input(user_query)
                .output(agently_format)
                .start()
            )
            
            self.logger.debug(f"Agently生成结果: {result}")
            
            # 验证结果
            if result is None:
                self.logger.warning("Agently返回None结果，可能是配置或输入问题")
                raise StructuredOutputError("Agently返回None结果，请检查API配置和输入内容")
                
            return result
            
        except StructuredOutputError:
            # 重新抛出已知的结构化输出错误（包括API密钥错误）
            raise
        except Exception as e:
            self.logger.error(f"Agently解析失败: {e}")
            # 如果有模型响应，尝试从中提取JSON作为fallback
            if model_response:
                try:
                    json_content = self.extract_json_from_text(model_response)
                    return json.loads(json_content)
                except Exception as fallback_e:
                    self.logger.debug(f"Fallback解析也失败: {fallback_e}")
            raise StructuredOutputError(f"Agently解析失败: {e}")
    
    def _parse_with_native(self, content: str, schema: Dict[str, Any]) -> Any:
        """使用原生JSON解析结构化输出。"""
        try:
            # 先从文本中提取JSON内容（处理代码块等格式）
            json_content = self.extract_json_from_text(content)
            
            # 尝试解析JSON
            result = json.loads(json_content)
            
            # 基本的schema验证
            self._validate_against_schema(result, schema)
            
            logger.debug(f"Native parsing successful: {result}")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            raise StructuredOutputError(f"Invalid JSON format: {e}")
        except Exception as e:
            logger.error(f"Native parsing failed: {e}")
            raise StructuredOutputError(f"Native parsing failed: {e}")
    

    
    def _validate_against_schema(self, data: Any, schema: Dict[str, Any]) -> None:
        """验证数据是否符合schema。"""
        try:
            jsonschema.validate(data, schema)
        except ImportError:
            logger.warning("jsonschema not available, skipping validation")
        except Exception as e:
            raise StructuredOutputError(f"Schema validation failed: {e}")
    
    def format_response_format(self, 
                              schema: Dict[str, Any], 
                              name: str = "response",
                              strict: bool = True) -> Dict[str, Any]:
        """格式化response_format参数。
        
        Args:
            schema: JSON Schema定义
            name: Schema名称
            strict: 是否启用严格模式
            
        Returns:
            格式化的response_format字典
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": schema,
                "strict": strict
            }
        }
    
    def extract_json_from_text(self, text: str) -> str:
        """从文本中提取JSON内容。
        
        Args:
            text: 包含JSON的文本
            
        Returns:
            提取的JSON字符串
        """
        # 尝试找到JSON代码块
        import re
        
        # 查找```json...```格式
        json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # 查找```...```格式
        code_match = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        if code_match:
            content = code_match.group(1).strip()
            # 检查是否是有效JSON
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError:
                pass
        
        # 查找{...}格式
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            content = brace_match.group(0)
            try:
                json.loads(content)
                return content
            except json.JSONDecodeError:
                pass
        
        # 如果都没找到，返回原文本
        return text.strip()


    def parse_streaming_response(self, 
                               content_stream: Union[Generator[str, None, None], AsyncGenerator[str, None]], 
                               schema: Dict[str, Any],
                               provider: Optional[str] = None,
                               api_key: str = None,
                               base_url: str = None,
                               model: str = None) -> Union[Generator[Any, None, None], AsyncGenerator[Any, None]]:
        """解析流式结构化输出响应。
        
        Args:
            content_stream: 流式响应内容生成器
            schema: JSON Schema定义
            provider: 解析提供者（agently或native）
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            
        Returns:
            解析后的结构化数据流
        """
        provider = provider or self.provider
        
        # 首先尝试使用Agently流式解析
        if provider == "agently":
            try:
                self.logger.debug("Attempting Agently streaming parsing")
                result = self._parse_streaming_with_agently(content_stream, schema, api_key, base_url, model)
                self.logger.debug("Agently streaming parsing successful")
                return result
            except ImportError as e:
                self.logger.warning(f"Agently import failed: {e}, falling back to native streaming parsing")
            except StructuredOutputError as e:
                self.logger.error(f"Agently streaming parsing failed with StructuredOutputError: {e}")
                raise
            except Exception as e:
                self.logger.warning(f"Agently streaming parsing failed: {e}, falling back to native streaming parsing")
        
        # 回退到原生流式解析
        try:
            self.logger.debug("Using native streaming parsing")
            result = self._parse_streaming_with_native(content_stream, schema)
            self.logger.debug("Native streaming parsing successful")
            return result
        except Exception as e:
            self.logger.error(f"Both Agently and native streaming parsing failed: {e}")
            raise StructuredOutputError(f"All streaming parsing methods failed: {e}")
    
    def _parse_streaming_with_agently(self, response_stream, schema: Dict[str, Any], api_key: str = None, base_url: str = None, model: str = None):
        """使用Agently解析流式响应。
        
        Args:
            response_stream: 流式响应对象
            schema: JSON Schema定义
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            
        Returns:
            解析后的结构化数据流（同步或异步生成器）
        """
        # 检查是否为异步生成器
        if hasattr(response_stream, '__aiter__'):
            return self._parse_async_streaming_with_agently(response_stream, schema, api_key, base_url, model)
        else:
            return self._parse_sync_streaming_with_agently(response_stream, schema, api_key, base_url, model)
    
    def _parse_sync_streaming_with_agently(self, 
                                         content_stream: Generator[str, None, None], 
                                         schema: Dict[str, Any],
                                         api_key: str = None,
                                         base_url: str = None,
                                         model: str = None) -> Generator[Any, None, None]:
        """同步解析流式结构化输出（使用Agently）。
        
        根据Agently结构化输出语法设计理念，使用get_instant_generator()方法进行真正的流式解析。
        
        Args:
            content_stream: 流式响应内容生成器
            schema: JSON Schema定义
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            
        Yields:
            部分解析的结构化数据
        """
        try:
            
            # 将JSON Schema转换为Agently格式
            agently_format = self._convert_json_schema_to_agently_output({"json_schema": {"schema": schema}})
            
            # 收集完整内容（注意：Agently需要完整的输入内容）
            full_content = ""
            for chunk in content_stream:
                full_content += chunk
            
            # 创建Agently代理
            agent = Agently.create_agent()
            
            # 为model提供默认值，避免None导致的配置失败
            model = model or "deepseek-chat"
            
            # 使用全局配置方法（与非流式方法保持一致）
            if api_key and base_url:
                # 使用 OpenAICompatible 全局配置（基于测试成功的方法）
                Agently.set_settings(
                    "OpenAICompatible",
                    {
                        "base_url": base_url,
                        "model": model,
                        "model_type": "chat",
                        "auth": api_key,
                    },
                )
                self.logger.debug(f"为流式agent配置全局设置: {model}, base_url: {base_url}")
            else:
                self.logger.warning("流式解析：API密钥或base_url为空，使用默认配置")
            
            # 根据Agently文档，使用get_instant_generator()进行流式解析
            try:
                # 根据Agently结构化输出语法设计理念，使用.input().output().get_instant_generator()进行流式解析
                instant_generator = (
                    agent
                    .input(full_content)
                    .output(agently_format)
                    .get_instant_generator()
                )
                
                # 处理流式事件
                current_result = {}
                for event in instant_generator:
                    try:
                        # 根据文档，事件包含key、indexes、delta、value、complete_value字段
                        if "complete_value" in event and event["complete_value"] is not None:
                            # 优先使用complete_value获取完整状态
                            yield event["complete_value"]
                            self.logger.debug(f"Agently流式事件 - complete_value: {event['complete_value']}")
                        elif "key" in event and "delta" in event:
                            # 处理增量更新
                            key = event["key"]
                            delta = event["delta"]
                            indexes = event.get("indexes", [])
                            
                            # 更新当前结果
                            self._update_result_by_key(current_result, key, delta, indexes)
                            
                            # 产出当前状态的副本
                            yield dict(current_result)
                            self.logger.debug(f"Agently流式事件 - 增量更新: key={key}, delta={delta}")
                        
                    except Exception as e:
                        self.logger.error(f"处理Agently流式事件失败: {e}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"Agently get_instant_generator失败: {e}")
                # 回退到原生解析
                if full_content:
                    try:
                        yield from self._parse_sync_streaming_with_native([full_content], schema)
                    except Exception as fallback_e:
                        self.logger.error(f"原生解析回退也失败: {fallback_e}")
                        yield {}
                else:
                    yield {}
                    
        except ImportError as e:
            self.logger.error(f"Agently导入失败: {e}")
            raise
        except StructuredOutputError:
            # 重新抛出已知的结构化输出错误（包括API密钥错误）
            raise
        except Exception as e:
            self.logger.error(f"Agently流式解析失败: {e}")
            # 异常回退
            if 'full_content' in locals() and full_content:
                try:
                    yield from self._parse_sync_streaming_with_native([full_content], schema)
                except Exception as fallback_e:
                    self.logger.error(f"原生解析回退也失败: {fallback_e}")
                    yield {}
            else:
                yield {}
    
    async def _parse_async_streaming_with_agently(self, 
                                                content_stream: AsyncGenerator[str, None], 
                                                schema: Dict[str, Any],
                                                api_key: str = None,
                                                base_url: str = None,
                                                model: str = None) -> AsyncGenerator[Any, None]:
        """异步解析流式结构化输出（使用Agently）。
        
        根据Agently结构化输出语法设计理念，使用get_instant_generator()方法进行真正的流式解析。
        
        Args:
            content_stream: 异步流式响应内容生成器
            schema: JSON Schema定义
            api_key: API密钥
            base_url: API基础URL
            model: 模型名称
            
        Yields:
            部分解析的结构化数据
        """
        try:
            # 将JSON Schema转换为Agently格式
            agently_format = self._convert_json_schema_to_agently_output({"json_schema": {"schema": schema}})
            
            # 收集完整内容（注意：Agently需要完整的输入内容）
            full_content = ""
            async for chunk in content_stream:
                full_content += chunk
            
            # 创建Agently代理
            agent = Agently.create_agent()
            
            # 为model提供默认值，避免None导致的配置失败
            model = model or "deepseek-chat"
            
            # 使用全局配置方法（与其他方法保持一致）
            if api_key and base_url:
                # 使用 OpenAICompatible 全局配置（基于测试成功的方法）
                Agently.set_settings(
                    "OpenAICompatible",
                    {
                        "base_url": base_url,
                        "model": model,
                        "model_type": "chat",
                        "auth": api_key,
                    },
                )
                self.logger.debug(f"为异步流式agent配置全局设置: {model}, base_url: {base_url}")
            else:
                self.logger.warning("异步流式解析：API密钥或base_url为空，使用默认配置")
            
            # 根据Agently文档，尝试使用异步instant generator，如果不支持则回退到同步方式
            try:
                # 尝试使用异步instant generator（如果Agently支持）
                if hasattr(agent.input(full_content).output(agently_format), 'get_async_instant_generator'):
                    instant_generator = agent.input(full_content).output(agently_format).get_async_instant_generator()
                    
                    # 处理异步流式事件
                    current_result = {}
                    async for event in instant_generator:
                        try:
                            # 根据文档，事件包含key、indexes、delta、value、complete_value字段
                            if "complete_value" in event and event["complete_value"] is not None:
                                # 优先使用complete_value获取完整状态
                                yield event["complete_value"]
                                self.logger.debug(f"Agently异步流式事件 - complete_value: {event['complete_value']}")
                            elif "key" in event and "delta" in event:
                                # 处理增量更新
                                key = event["key"]
                                delta = event["delta"]
                                indexes = event.get("indexes", [])
                                
                                # 更新当前结果
                                self._update_result_by_key(current_result, key, delta, indexes)
                                
                                # 产出当前状态的副本
                                yield dict(current_result)
                                self.logger.debug(f"Agently异步流式事件 - 增量更新: key={key}, delta={delta}")
                                
                        except Exception as e:
                            self.logger.error(f"处理Agently异步流式事件失败: {e}")
                            continue
                else:
                    # 如果不支持异步instant generator，使用同步方式
                    instant_generator = agent.input(full_content).output(agently_format).get_instant_generator()
                    
                    current_result = {}
                    for event in instant_generator:
                        try:
                            if "complete_value" in event and event["complete_value"] is not None:
                                yield event["complete_value"]
                                self.logger.debug(f"Agently同步流式事件 - complete_value: {event['complete_value']}")
                            elif "key" in event and "delta" in event:
                                key = event["key"]
                                delta = event["delta"]
                                indexes = event.get("indexes", [])
                                
                                self._update_result_by_key(current_result, key, delta, indexes)
                                yield dict(current_result)
                                self.logger.debug(f"Agently同步流式事件 - 增量更新: key={key}, delta={delta}")
                                
                        except Exception as e:
                            self.logger.error(f"处理Agently同步流式事件失败: {e}")
                            continue
                    
            except Exception as e:
                self.logger.error(f"Agently get_instant_generator失败: {e}")
                # 回退到原生解析
                if full_content:
                    try:
                        async for result in self._parse_async_streaming_with_native([full_content], schema):
                            yield result
                    except Exception as fallback_e:
                        self.logger.error(f"异步原生解析回退也失败: {fallback_e}")
                        yield {}
                else:
                    yield {}
                    
        except ImportError as e:
            self.logger.error(f"Agently导入失败: {e}")
            raise
        except StructuredOutputError:
            # 重新抛出已知的结构化输出错误（包括API密钥错误）
            raise
        except Exception as e:
            self.logger.error(f"Agently异步流式解析失败: {e}")
            # 异常回退
            if 'full_content' in locals() and full_content:
                try:
                    async for result in self._parse_async_streaming_with_native([full_content], schema):
                        yield result
                except Exception as fallback_e:
                    self.logger.error(f"异步原生解析回退也失败: {fallback_e}")
                    yield {}
            else:
                yield {}
    
    def _parse_streaming_with_native(self, 
                                   content_stream: Union[Generator[str, None, None], AsyncGenerator[str, None]], 
                                   schema: Dict[str, Any]) -> Union[Generator[Any, None, None], AsyncGenerator[Any, None]]:
        """使用原生方式解析流式结构化输出。"""
        # 检查是否为异步生成器
        if hasattr(content_stream, '__aiter__'):
            return self._parse_async_streaming_with_native(content_stream, schema)
        else:
            return self._parse_sync_streaming_with_native(content_stream, schema)
    
    def _parse_sync_streaming_with_native(self, 
                                        content_stream: Generator[str, None, None], 
                                        schema: Dict[str, Any]) -> Generator[Any, None, None]:
        """同步流式原生解析。"""
        accumulated_content = ""
        
        for chunk in content_stream:
            accumulated_content += chunk
            
            # 尝试解析当前累积的内容
            try:
                json_content = self.extract_json_from_text(accumulated_content)
                if json_content:
                    try:
                        parsed_data = json.loads(json_content)
                        self._validate_against_schema(parsed_data, schema)
                        yield parsed_data
                    except (json.JSONDecodeError, StructuredOutputError):
                        continue
            except Exception as e:
                self.logger.debug(f"Native streaming parse attempt failed: {e}")
                continue
    
    async def _parse_async_streaming_with_native(self, 
                                               content_stream: AsyncGenerator[str, None], 
                                               schema: Dict[str, Any]) -> AsyncGenerator[Any, None]:
        """异步流式原生解析。"""
        accumulated_content = ""
        
        async for chunk in content_stream:
            accumulated_content += chunk
            
            # 尝试解析当前累积的内容
            try:
                json_content = self.extract_json_from_text(accumulated_content)
                if json_content:
                    try:
                        parsed_data = json.loads(json_content)
                        self._validate_against_schema(parsed_data, schema)
                        yield parsed_data
                    except (json.JSONDecodeError, StructuredOutputError):
                        continue
            except Exception as e:
                self.logger.debug(f"Native async streaming parse attempt failed: {e}")
                continue
    
    def _update_result_by_key(self, result: Dict[str, Any], key: str, delta: str, indexes: list = None) -> None:
        """根据key路径更新结果字典。
        
        Args:
            result: 要更新的结果字典
            key: 键路径（如'name'或'items'）
            delta: 增量内容
            indexes: 数组索引列表（如[0]表示更新数组第0个元素）
        """
        try:
            # 分割key路径
            key_parts = key.split('.')
            current = result
            
            # 遍历到最后一个key之前
            for part in key_parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            # 处理最后一个key
            final_key = key_parts[-1]
            
            # 如果有indexes参数，说明这是一个数组字段
            if indexes is not None and len(indexes) > 0:
                idx = indexes[0]  # 使用第一个索引
                
                # 确保字段存在且为数组
                if final_key not in current:
                    current[final_key] = []
                elif not isinstance(current[final_key], list):
                    current[final_key] = []
                
                # 确保数组足够长
                while len(current[final_key]) <= idx:
                    current[final_key].append("")
                
                # 更新数组元素
                if isinstance(current[final_key][idx], str):
                    current[final_key][idx] += delta
                else:
                    current[final_key][idx] = str(current[final_key][idx]) + delta
            else:
                # 普通字段
                if final_key not in current:
                    current[final_key] = ""
                
                if isinstance(current[final_key], str):
                    current[final_key] += delta
                else:
                    current[final_key] = str(current[final_key]) + delta
                
        except Exception as e:
            self.logger.debug(f"Failed to update result by key {key}: {e}")
            # 如果路径解析失败，直接设置到根级别
            if key not in result:
                result[key] = ""
            if isinstance(result[key], str):
                result[key] += delta
            else:
                result[key] = str(result[key]) + delta


# 全局实例
default_handler = StructuredOutputHandler()


def parse_structured_output(response_content: str, 
                           schema: Dict[str, Any],
                           use_agently: bool = True) -> Any:
    """解析结构化输出的便捷函数。
    
    Args:
        response_content: 响应内容
        schema: JSON Schema定义
        use_agently: 是否使用Agently解析
        
    Returns:
        解析后的结构化数据
    """
    return default_handler.parse_response(response_content, schema, use_agently)


def parse_streaming_structured_output(content_stream: Union[Generator[str, None, None], AsyncGenerator[str, None]], 
                                     schema: Dict[str, Any], 
                                     provider: str = "agently") -> Union[Generator[Any, None, None], AsyncGenerator[Any, None]]:
    """解析流式结构化输出的便捷函数。"""
    return default_handler.parse_streaming_response(content_stream, schema, provider)


def create_response_format(schema: Dict[str, Any], 
                          name: str = "response",
                          strict: bool = True) -> Dict[str, Any]:
    """创建response_format参数的便捷函数。
    
    Args:
        schema: JSON Schema定义
        name: Schema名称
        strict: 是否启用严格模式
        
    Returns:
        格式化的response_format字典
    """
    return default_handler.format_response_format(schema, name, strict)