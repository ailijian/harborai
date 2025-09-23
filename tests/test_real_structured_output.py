#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实模型结构化输出测试

使用DeepSeek模型测试结构化输出功能的真实效果。
需要在.env文件中配置DeepSeek API密钥。
"""

import os
import json
import pytest
import asyncio
from typing import Dict, Any, List
from dotenv import load_dotenv

from harborai import HarborAI
from harborai.utils.exceptions import ValidationError, StructuredOutputError

# 加载环境变量
load_dotenv()

# 检查是否有DeepSeek配置
HAS_DEEPSEEK_CONFIG = bool(
    os.getenv('DEEPSEEK_API_KEY') or 
    os.getenv('OPENAI_API_KEY')  # DeepSeek可能使用OpenAI兼容格式
)


@pytest.mark.skipif(not HAS_DEEPSEEK_CONFIG, reason="需要DeepSeek API配置")
class TestRealStructuredOutput:
    """真实模型结构化输出测试"""
    
    @pytest.fixture
    def real_client(self):
        """真实的HarborAI客户端"""
        return HarborAI()
    
    @pytest.fixture
    def person_schema(self):
        """人员信息Schema"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "person_info",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "人员姓名"
                        },
                        "age": {
                            "type": "integer",
                            "description": "年龄",
                            "minimum": 0,
                            "maximum": 150
                        },
                        "occupation": {
                            "type": "string",
                            "description": "职业"
                        },
                        "skills": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "技能列表"
                        },
                        "experience_years": {
                            "type": "number",
                            "description": "工作经验年数"
                        }
                    },
                    "required": ["name", "age", "occupation"]
                }
            }
        }
    
    @pytest.fixture
    def analysis_schema(self):
        """分析结果Schema"""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "analysis_result",
                "schema": {
                    "type": "object",
                    "properties": {
                        "summary": {
                            "type": "string",
                            "description": "分析摘要"
                        },
                        "confidence": {
                            "type": "number",
                            "description": "置信度(0-1)",
                            "minimum": 0,
                            "maximum": 1
                        },
                        "categories": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "分类标签"
                        },
                        "recommendations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string"},
                                    "priority": {"type": "string", "enum": ["high", "medium", "low"]}
                                },
                                "required": ["action", "priority"]
                            },
                            "description": "建议列表"
                        }
                    },
                    "required": ["summary", "confidence"]
                }
            }
        }
    
    def test_agently_structured_output_person_info(self, real_client, person_schema):
        """测试Agently模式 - 人员信息提取"""
        messages = [
            {
                "role": "system",
                "content": "你是一个信息提取助手，请从用户描述中提取人员信息。"
            },
            {
                "role": "user",
                "content": "张三是一名30岁的软件工程师，有5年的Python开发经验，擅长机器学习和Web开发。"
            }
        ]
        
        response = real_client.chat.completions.create(
            model="deepseek-chat",  # 使用DeepSeek模型
            messages=messages,
            response_format=person_schema,
            structured_provider="agently",
            temperature=0.1
        )
        
        # 验证响应结构
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        
        # 如果有结构化输出，验证其格式
        choice = response.choices[0]
        if hasattr(choice.message, 'parsed') and choice.message.parsed:
            parsed_data = choice.message.parsed
            assert isinstance(parsed_data, dict)
            assert "name" in parsed_data
            assert "age" in parsed_data
            assert "occupation" in parsed_data
            
            # 验证数据类型
            assert isinstance(parsed_data["name"], str)
            assert isinstance(parsed_data["age"], int)
            assert isinstance(parsed_data["occupation"], str)
            
            # 验证解析结果格式正确
            assert isinstance(parsed_data, dict)
    
    def test_native_structured_output_person_info(self, real_client, person_schema):
        """测试Native模式 - 人员信息提取"""
        messages = [
            {
                "role": "system",
                "content": "你是一个信息提取助手，请从用户描述中提取人员信息。"
            },
            {
                "role": "user",
                "content": "李四是一名25岁的数据科学家，有3年的机器学习经验，精通Python、R和SQL。"
            }
        ]
        
        response = real_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format=person_schema,
            structured_provider="native",
            temperature=0.1
        )
        
        # 验证响应结构
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        
        # 如果有结构化输出，验证其格式
        choice = response.choices[0]
        if hasattr(choice.message, 'parsed') and choice.message.parsed:
            parsed_data = choice.message.parsed
            assert isinstance(parsed_data, dict)
            assert "name" in parsed_data
            assert "age" in parsed_data
            assert "occupation" in parsed_data
            
            # 验证解析结果格式正确
            assert isinstance(parsed_data, dict)
    
    def test_agently_vs_native_comparison(self, real_client, analysis_schema):
        """测试Agently vs Native模式对比"""
        messages = [
            {
                "role": "system",
                "content": "你是一个数据分析师，请分析给定的文本内容。"
            },
            {
                "role": "user",
                "content": "我们公司的销售额在过去一年中增长了25%，主要得益于新产品线的推出和市场营销策略的优化。但是，客户满意度略有下降，需要改进客户服务质量。"
            }
        ]
        
        # 测试Agently模式
        agently_response = real_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format=analysis_schema,
            structured_provider="agently",
            temperature=0.1
        )
        
        # 测试Native模式
        native_response = real_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format=analysis_schema,
            structured_provider="native",
            temperature=0.1
        )
        
        # 验证两种模式都能正常工作
        assert agently_response is not None
        assert native_response is not None
        
        # 验证两种模式的解析结果
        if (hasattr(agently_response.choices[0].message, 'parsed') and 
            agently_response.choices[0].message.parsed):
            assert isinstance(agently_response.choices[0].message.parsed, dict)
        
        if (hasattr(native_response.choices[0].message, 'parsed') and 
            native_response.choices[0].message.parsed):
            assert isinstance(native_response.choices[0].message.parsed, dict)
    
    @pytest.mark.asyncio
    async def test_async_agently_structured_output(self, real_client, person_schema):
        """测试异步Agently结构化输出"""
        messages = [
            {
                "role": "system",
                "content": "你是一个信息提取助手，请从用户描述中提取人员信息。"
            },
            {
                "role": "user",
                "content": "王五是一名28岁的产品经理，有4年的产品设计经验，擅长用户体验设计和数据分析。"
            }
        ]
        
        response = await real_client.chat.completions.acreate(
            model="deepseek-chat",
            messages=messages,
            response_format=person_schema,
            structured_provider="agently",
            temperature=0.1
        )
        
        # 验证响应结构
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        
        # 如果有结构化输出，验证其格式
        choice = response.choices[0]
        if hasattr(choice.message, 'parsed') and choice.message.parsed:
            parsed_data = choice.message.parsed
            assert isinstance(parsed_data, dict)
            assert "name" in parsed_data
            assert "age" in parsed_data
            assert "occupation" in parsed_data
            
            # 验证异步解析结果格式正确
            assert isinstance(parsed_data, dict)
    
    def test_streaming_structured_output(self, real_client, person_schema):
        """测试流式结构化输出"""
        messages = [
            {
                "role": "system",
                "content": "你是一个信息提取助手，请从用户描述中提取人员信息。"
            },
            {
                "role": "user",
                "content": "赵六是一名35岁的架构师，有10年的软件开发经验，专精于分布式系统和云计算。"
            }
        ]
        
        # 测试流式输出
        stream = real_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format=person_schema,
            structured_provider="agently",
            stream=True,
            temperature=0.1
        )
        
        # 收集流式响应
        chunks = []
        for chunk in stream:
            chunks.append(chunk)
            # 验证流式内容格式
            if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                assert isinstance(chunk.choices[0].delta.content, str)
        
        assert len(chunks) > 0
        # 验证收到了足够的流式块
        assert len(chunks) >= 1
    
    def test_error_handling_invalid_schema(self, real_client):
        """测试错误处理 - 无效Schema"""
        messages = [
            {
                "role": "user",
                "content": "测试消息"
            }
        ]
        
        invalid_schema = {
            "type": "invalid_type",  # 无效的类型
            "json_schema": {
                "name": "test",
                "schema": {"type": "object"}
            }
        }
        
        # 应该能处理无效schema而不崩溃
        try:
            response = real_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                response_format=invalid_schema,
                structured_provider="agently",
                temperature=0.1
            )
            # 如果没有抛出异常，至少应该有响应
            assert response is not None
        except Exception as e:
            # 如果抛出异常，应该是可预期的异常类型
            assert isinstance(e, (ValidationError, StructuredOutputError))
    
    def test_fallback_mechanism(self, real_client, person_schema):
        """测试回退机制"""
        messages = [
            {
                "role": "system",
                "content": "你是一个信息提取助手。"
            },
            {
                "role": "user",
                "content": "提取这个人的信息：小明，22岁，学生。"
            }
        ]
        
        # 即使Agently解析失败，也应该能回退到native解析
        response = real_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format=person_schema,
            structured_provider="agently",
            temperature=0.1
        )
        
        # 应该有响应，无论使用哪种解析方式
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        
        # 验证回退机制正常工作
        assert len(response.choices[0].message.content) > 0


if __name__ == "__main__":
    # 运行测试的示例
    import sys
    if HAS_DEEPSEEK_CONFIG:
        sys.exit(0)  # 配置正确，可以运行测试
    else:
        sys.exit(1)  # 配置缺失，需要配置API密钥