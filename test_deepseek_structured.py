#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek 结构化输出功能测试脚本

测试内容：
1. 非流式结构化输出（原生 vs Agently）
2. 流式结构化输出（原生 vs Agently）
3. 不同复杂度的 JSON Schema
4. 性能对比和结果分析
5. 错误处理和异常测试

配置要求：
    需要配置 DEEPSEEK_API_KEY 环境变量：
    - Windows: set DEEPSEEK_API_KEY=your_api_key_here
    - Linux/Mac: export DEEPSEEK_API_KEY=your_api_key_here
    
    如果未配置，将使用模拟数据进行测试。

使用方法：
    python test_deepseek_structured.py
"""

import os
import sys
import json
import time
import asyncio
from typing import Dict, Any, List, Generator, AsyncGenerator
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harborai.api.client import HarborAI
from harborai.api.structured import StructuredOutputHandler, parse_structured_output, parse_streaming_structured_output
from harborai.utils.exceptions import StructuredOutputError
from harborai.utils.logger import get_logger


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    method: str  # 'native' or 'agently'
    success: bool
    duration: float
    result_data: Any
    error_message: str = None
    tokens_used: int = 0


class DeepSeekStructuredTester:
    """DeepSeek结构化输出测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.logger = get_logger(__name__)
        self.client = None
        self.handler = StructuredOutputHandler()
        self.test_results: List[TestResult] = []
        
        # 检查环境变量
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key or self.api_key == 'sk-your-deepseek-api-key-here':
            self.logger.warning("DEEPSEEK_API_KEY未正确配置，将跳过实际API调用测试")
            self.api_key = None
        else:
            self.logger.info(f"已配置DEEPSEEK_API_KEY: {self.api_key[:10]}...")
    
    def setup_client(self) -> bool:
        """设置HarborAI客户端"""
        try:
            if not self.api_key:
                self.logger.warning("API密钥未配置，无法进行实际API调用测试")
                return False
                
            self.client = HarborAI(
                provider="deepseek",
                api_key=self.api_key,
                model="deepseek-chat"
            )
            self.logger.info("HarborAI客户端初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"客户端初始化失败: {e}")
            return False
    
    def get_test_schemas(self) -> Dict[str, Dict[str, Any]]:
        """获取测试用的JSON Schema"""
        return {
            "simple": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "人物姓名"},
                    "age": {"type": "integer", "description": "年龄"}
                },
                "required": ["name", "age"]
            },
            "medium": {
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "用户姓名"},
                            "email": {"type": "string", "description": "邮箱地址"},
                            "preferences": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "用户偏好列表"
                            }
                        },
                        "required": ["name", "email"]
                    },
                    "score": {"type": "number", "description": "评分"}
                },
                "required": ["user", "score"]
            },
            "complex": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "公司名称"},
                            "employees": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer", "description": "员工ID"},
                                        "name": {"type": "string", "description": "员工姓名"},
                                        "department": {"type": "string", "description": "部门"},
                                        "skills": {
                                            "type": "array",
                                            "items": {"type": "string"},
                                            "description": "技能列表"
                                        }
                                    },
                                    "required": ["id", "name", "department"]
                                },
                                "description": "员工列表"
                            }
                        },
                        "required": ["name", "employees"]
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "created_at": {"type": "string", "description": "创建时间"},
                            "version": {"type": "string", "description": "版本号"}
                        }
                    }
                },
                "required": ["company"]
            }
        }
    
    def get_test_prompts(self) -> Dict[str, str]:
        """获取测试提示词"""
        return {
            "simple": "请生成一个虚构人物的基本信息，包括姓名和年龄。",
            "medium": "请生成一个用户档案，包含用户基本信息和评分。",
            "complex": "请生成一个科技公司的组织架构信息，包含公司名称、员工列表和元数据。"
        }
    
    def print_separator(self, title: str):
        """打印分隔线"""
        print("\n" + "="*80)
        print(f" {title} ".center(80, "="))
        print("="*80)
    
    def print_test_result(self, result: TestResult):
        """打印测试结果"""
        status = "✅ 成功" if result.success else "❌ 失败"
        print(f"\n📋 测试: {result.test_name}")
        print(f"🔧 方法: {result.method}")
        print(f"📊 状态: {status}")
        print(f"⏱️  耗时: {result.duration:.2f}秒")
        
        if result.success and result.result_data:
            print(f"📄 结果: {json.dumps(result.result_data, ensure_ascii=False, indent=2)}")
        
        if result.error_message:
            print(f"❗ 错误: {result.error_message}")
        
        if result.tokens_used > 0:
            print(f"🎯 Token使用: {result.tokens_used}")
    
    def simulate_api_response(self, schema_name: str) -> str:
        """模拟API响应（当没有真实API密钥时使用）"""
        mock_responses = {
            "simple": '{"name": "张三", "age": 25}',
            "medium": '{"user": {"name": "李四", "email": "lisi@example.com", "preferences": ["编程", "阅读"]}, "score": 8.5}',
            "complex": '{"company": {"name": "创新科技有限公司", "employees": [{"id": 1, "name": "王五", "department": "研发部", "skills": ["Python", "AI"]}]}, "metadata": {"created_at": "2024-01-15", "version": "1.0"}}'
        }
        return mock_responses.get(schema_name, '{}')
    
    def simulate_streaming_response(self, schema_name: str) -> Generator[str, None, None]:
        """模拟流式API响应"""
        response = self.simulate_api_response(schema_name)
        # 将响应分块返回，模拟流式输出
        chunk_size = max(1, len(response) // 10)
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]
            time.sleep(0.1)  # 模拟网络延迟
    
    def test_non_streaming_native(self, schema_name: str, schema: Dict[str, Any], prompt: str) -> TestResult:
        """测试非流式原生结构化输出"""
        start_time = time.time()
        
        try:
            if self.client:
                # 使用真实API调用
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": f"{schema_name}_response",
                            "schema": schema,
                            "strict": True
                        }
                    }
                )
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
            else:
                # 使用模拟响应
                content = self.simulate_api_response(schema_name)
                tokens_used = 0
            
            # 使用原生方式解析
            result = self.handler.parse_response(content, schema, use_agently=False)
            duration = time.time() - start_time
            
            return TestResult(
                test_name=f"{schema_name}_非流式_原生",
                method="native",
                success=True,
                duration=duration,
                result_data=result,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=f"{schema_name}_非流式_原生",
                method="native",
                success=False,
                duration=duration,
                result_data=None,
                error_message=str(e)
            )
    
    def test_non_streaming_agently(self, schema_name: str, schema: Dict[str, Any], prompt: str) -> TestResult:
        """测试非流式Agently结构化输出"""
        start_time = time.time()
        
        try:
            if self.client:
                # 使用真实API调用（不使用response_format，让Agently处理）
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 0
            else:
                # 使用模拟响应
                content = self.simulate_api_response(schema_name)
                tokens_used = 0
            
            # 使用Agently方式解析
            result = self.handler.parse_response(content, schema, use_agently=True)
            duration = time.time() - start_time
            
            return TestResult(
                test_name=f"{schema_name}_非流式_Agently",
                method="agently",
                success=True,
                duration=duration,
                result_data=result,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=f"{schema_name}_非流式_Agently",
                method="agently",
                success=False,
                duration=duration,
                result_data=None,
                error_message=str(e)
            )
    
    async def run_non_streaming_tests(self):
        """运行所有非流式测试"""
        self.print_separator("非流式结构化输出测试")
        
        schemas = self.get_test_schemas()
        prompts = self.get_test_prompts()
        
        for schema_name, schema in schemas.items():
            prompt = prompts[schema_name]
            
            print(f"\n🧪 测试Schema: {schema_name}")
            print(f"📝 提示词: {prompt}")
            
            # 测试原生方式
            native_result = self.test_non_streaming_native(schema_name, schema, prompt)
            self.test_results.append(native_result)
            self.print_test_result(native_result)
            
            # 测试Agently方式
            agently_result = self.test_non_streaming_agently(schema_name, schema, prompt)
            self.test_results.append(agently_result)
            self.print_test_result(agently_result)
            
            # 对比结果
            if native_result.success and agently_result.success:
                print(f"\n📊 性能对比:")
                print(f"   原生方式: {native_result.duration:.2f}秒")
                print(f"   Agently方式: {agently_result.duration:.2f}秒")
                
                if native_result.result_data == agently_result.result_data:
                    print("   ✅ 结果一致")
                else:
                    print("   ⚠️  结果不一致")
            
            print("-" * 60)
    
    def run_error_tests(self):
        """运行错误处理测试"""
        self.print_separator("错误处理测试")
        
        # 测试无效的JSON Schema
        invalid_schema = {
            "type": "invalid_type",  # 无效类型
            "properties": {
                "name": {"type": "string"}
            }
        }
        
        print("\n🧪 测试无效Schema处理")
        try:
            result = self.handler.parse_response(
                '{"name": "test"}', 
                invalid_schema, 
                use_agently=False
            )
            print("❌ 应该抛出异常但没有")
        except Exception as e:
            print(f"✅ 正确处理无效Schema: {str(e)[:100]}...")
        
        # 测试无效的JSON响应
        valid_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"}
            },
            "required": ["name"]
        }
        
        print("\n🧪 测试无效JSON响应处理")
        try:
            result = self.handler.parse_response(
                "这不是有效的JSON", 
                valid_schema, 
                use_agently=False
            )
            print("❌ 应该抛出异常但没有")
        except Exception as e:
            print(f"✅ 正确处理无效JSON: {str(e)[:100]}...")
    
    def analyze_results(self):
        """分析测试结果"""
        self.print_separator("测试结果分析")
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.success])
        failed_tests = total_tests - successful_tests
        
        print(f"\n📊 总体统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   成功: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        print(f"   失败: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # 按方法分组分析
        method_stats = {}
        for result in self.test_results:
            method = result.method
            if method not in method_stats:
                method_stats[method] = {'total': 0, 'success': 0, 'total_time': 0}
            
            method_stats[method]['total'] += 1
            if result.success:
                method_stats[method]['success'] += 1
                method_stats[method]['total_time'] += result.duration
        
        print(f"\n📈 方法性能对比:")
        for method, stats in method_stats.items():
            success_rate = stats['success'] / stats['total'] * 100
            avg_time = stats['total_time'] / stats['success'] if stats['success'] > 0 else 0
            print(f"   {method}:")
            print(f"     成功率: {success_rate:.1f}%")
            print(f"     平均耗时: {avg_time:.3f}秒")
        
        # 显示失败的测试
        failed_results = [r for r in self.test_results if not r.success]
        if failed_results:
            print(f"\n❌ 失败的测试:")
            for result in failed_results:
                print(f"   {result.test_name}: {result.error_message}")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始DeepSeek结构化输出功能测试")
        print(f"⏰ 测试开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 检查API密钥配置
        if not self.api_key:
            print("⚠️  未配置DEEPSEEK_API_KEY，将使用模拟数据进行测试")
        else:
            print("✅ 已配置DEEPSEEK_API_KEY，将进行真实API调用测试")
        
        try:
            # 运行非流式测试
            await self.run_non_streaming_tests()
            
            # 运行流式测试
            await self.run_streaming_tests()
            
            # 运行错误处理测试
            self.run_error_tests()
            
            # 分析结果
            self.analyze_results()
            
        except Exception as e:
            print(f"\n❌ 测试过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n🏁 测试完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n" + "=" * 80)

    def test_streaming_native(self, schema_name: str, schema: Dict[str, Any], prompt: str) -> TestResult:
        """测试流式原生结构化输出"""
        start_time = time.time()
        
        try:
            if self.client:
                # 使用真实API调用（流式）
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": f"{schema_name}_response",
                            "schema": schema,
                            "strict": True
                        }
                    },
                    stream=True
                )
                
                # 收集流式响应
                def content_generator():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                
                tokens_used = 0  # 流式响应中难以准确计算token
            else:
                # 使用模拟流式响应
                content_generator = lambda: self.simulate_streaming_response(schema_name)
                tokens_used = 0
            
            # 使用原生方式解析流式输出
            results = list(parse_streaming_structured_output(content_generator(), schema, provider="native"))
            final_result = results[-1] if results else {}
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name=f"{schema_name}_流式_原生",
                method="native_streaming",
                success=True,
                duration=duration,
                result_data=final_result,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=f"{schema_name}_流式_原生",
                method="native_streaming",
                success=False,
                duration=duration,
                result_data=None,
                error_message=str(e)
            )
    
    def test_streaming_agently(self, schema_name: str, schema: Dict[str, Any], prompt: str) -> TestResult:
        """测试流式Agently结构化输出"""
        start_time = time.time()
        
        try:
            if self.client:
                # 使用真实API调用（流式，不使用response_format）
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )
                
                # 收集流式响应
                def content_generator():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                
                tokens_used = 0  # 流式响应中难以准确计算token
            else:
                # 使用模拟流式响应
                content_generator = lambda: self.simulate_streaming_response(schema_name)
                tokens_used = 0
            
            # 使用Agently方式解析流式输出
            results = list(parse_streaming_structured_output(content_generator(), schema, provider="agently"))
            final_result = results[-1] if results else {}
            
            duration = time.time() - start_time
            
            return TestResult(
                test_name=f"{schema_name}_流式_Agently",
                method="agently_streaming",
                success=True,
                duration=duration,
                result_data=final_result,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                test_name=f"{schema_name}_流式_Agently",
                method="agently_streaming",
                success=False,
                duration=duration,
                result_data=None,
                error_message=str(e)
            )
    
    async def run_streaming_tests(self):
        """运行所有流式测试"""
        self.print_separator("流式结构化输出测试")
        
        schemas = self.get_test_schemas()
        prompts = self.get_test_prompts()
        
        for schema_name, schema in schemas.items():
            prompt = prompts[schema_name]
            
            print(f"\n🧪 测试Schema: {schema_name} (流式)")
            print(f"📝 提示词: {prompt}")
            
            # 测试原生流式方式
            native_result = self.test_streaming_native(schema_name, schema, prompt)
            self.test_results.append(native_result)
            self.print_test_result(native_result)
            
            # 测试Agently流式方式
            agently_result = self.test_streaming_agently(schema_name, schema, prompt)
            self.test_results.append(agently_result)
            self.print_test_result(agently_result)
            
            # 对比结果
            if native_result.success and agently_result.success:
                print(f"\n📊 流式性能对比:")
                print(f"   原生流式: {native_result.duration:.2f}秒")
                print(f"   Agently流式: {agently_result.duration:.2f}秒")
                
                if native_result.result_data == agently_result.result_data:
                    print("   ✅ 结果一致")
                else:
                    print("   ⚠️  结果不一致")
            
            print("-" * 60)


# 运行测试
if __name__ == "__main__":
    tester = DeepSeekStructuredTester()
    asyncio.run(tester.run_all_tests())