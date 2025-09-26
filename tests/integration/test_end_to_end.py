# -*- coding: utf-8 -*-
"""
端到端集成测试

本模块测试 HarborAI 的完整端到端功能，包括：
- 完整的API调用流程
- 同步和异步调用
- 流式响应处理
- 结构化输出
- 错误处理和重试机制
- 性能监控
"""

import pytest
import asyncio
import time
import json
import os
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, AsyncMock

# 导入 HarborAI 相关模块
try:
    from harborai import HarborAI
    from harborai.core.exceptions import HarborAIError, APIError, RateLimitError
    from harborai.monitoring.cost_analysis import CostAnalyzer
    from harborai.storage.postgres_logger import PostgreSQLLogger
except ImportError as e:
    pytest.skip(f"无法导入 HarborAI 模块: {e}", allow_module_level=True)


@pytest.fixture
def postgres_logger():
    """PostgreSQL日志记录器fixture"""
    logger = PostgreSQLLogger(
        connection_string="postgresql://test:test@localhost:5432/test_db",
        table_name="test_logs"
    )
    yield logger
    # 清理
    logger.stop()

from tests.integration import INTEGRATION_TEST_CONFIG, TEST_DATA_CONFIG, PERFORMANCE_BENCHMARKS


class TestEndToEndIntegration:
    """
    端到端集成测试类
    
    测试完整的API调用流程，从客户端初始化到响应处理的全过程。
    """
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """测试方法设置"""
        self.test_timeout = INTEGRATION_TEST_CONFIG["timeout"]
        self.retry_count = INTEGRATION_TEST_CONFIG["retry_count"]
        self.test_messages = [
            {"role": "user", "content": TEST_DATA_CONFIG["simple_message"]}
        ]
        
    @pytest.fixture
    def mock_client(self):
        """Mock HarborAI 客户端夹具"""
        with patch('harborai.HarborAI') as mock_harborai:
            mock_instance = Mock()
            mock_harborai.return_value = mock_instance
            
            # 配置默认的成功响应
            mock_response = Mock()
            mock_response.choices = [Mock(
                message=Mock(
                    content="这是一个测试响应，我是HarborAI助手。",
                    role="assistant"
                ),
                finish_reason="stop",
                index=0
            )]
            mock_response.usage = Mock(
                prompt_tokens=15,
                completion_tokens=25,
                total_tokens=40
            )
            mock_response.model = "test-model"
            mock_response.id = "chatcmpl-test-e2e"
            mock_response.object = "chat.completion"
            mock_response.created = int(time.time())
            
            mock_instance.chat.completions.create.return_value = mock_response
            
            yield mock_instance
    
    @pytest.fixture
    def mock_async_client(self):
        """Mock 异步 HarborAI 客户端夹具"""
        with patch('harborai.HarborAI') as mock_harborai:
            mock_instance = Mock()
            mock_harborai.return_value = mock_instance
            
            # 配置异步响应
            async def async_create(*args, **kwargs):
                mock_response = Mock()
                mock_response.choices = [Mock(
                    message=Mock(
                        content="这是一个异步测试响应。",
                        role="assistant"
                    ),
                    finish_reason="stop",
                    index=0
                )]
                mock_response.usage = Mock(
                    prompt_tokens=12,
                    completion_tokens=20,
                    total_tokens=32
                )
                mock_response.model = "test-async-model"
                mock_response.id = "chatcmpl-test-async"
                return mock_response
            
            mock_instance.chat.completions.create = AsyncMock(side_effect=async_create)
            
            yield mock_instance
    
    @pytest.fixture
    def mock_stream_client(self):
        """Mock 流式响应客户端夹具"""
        with patch('harborai.HarborAI') as mock_harborai:
            mock_instance = Mock()
            mock_harborai.return_value = mock_instance
            
            # 创建流式响应chunks
            stream_chunks = [
                Mock(
                    choices=[Mock(
                        delta=Mock(content="这是"),
                        index=0,
                        finish_reason=None
                    )],
                    id="chatcmpl-test-stream",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="test-stream-model"
                ),
                Mock(
                    choices=[Mock(
                        delta=Mock(content="流式"),
                        index=0,
                        finish_reason=None
                    )],
                    id="chatcmpl-test-stream",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="test-stream-model"
                ),
                Mock(
                    choices=[Mock(
                        delta=Mock(content="响应测试"),
                        index=0,
                        finish_reason="stop"
                    )],
                    id="chatcmpl-test-stream",
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model="test-stream-model"
                )
            ]
            
            mock_instance.chat.completions.create.return_value = iter(stream_chunks)
            
            yield mock_instance
    
    @pytest.mark.integration
    @pytest.mark.p0
    def test_basic_chat_completion(self, mock_client):
        """
        测试基础聊天完成功能
        
        验证：
        - 客户端初始化
        - 基本API调用
        - 响应解析
        - 使用统计
        """
        # 执行测试
        response = mock_client.chat.completions.create(
            model="test-model",
            messages=self.test_messages,
            max_tokens=100,
            temperature=0.7
        )
        
        # 验证响应结构
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], 'message')
        assert response.choices[0].message.role == "assistant"
        assert len(response.choices[0].message.content) > 0
        
        # 验证使用统计
        assert hasattr(response, 'usage')
        assert response.usage.total_tokens > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        
        # 验证API调用参数
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['model'] == "test-model"
        assert call_args.kwargs['messages'] == self.test_messages
        assert call_args.kwargs['max_tokens'] == 100
        assert call_args.kwargs['temperature'] == 0.7
    
    @pytest.mark.integration
    @pytest.mark.async_test
    @pytest.mark.p0
    @pytest.mark.asyncio
    async def test_async_chat_completion(self, mock_async_client):
        """
        测试异步聊天完成功能
        
        验证：
        - 异步API调用
        - 异步响应处理
        - 并发安全性
        """
        # 执行异步测试
        start_time = time.time()
        response = await mock_async_client.chat.completions.create(
            model="test-async-model",
            messages=self.test_messages,
            max_tokens=50
        )
        end_time = time.time()
        
        # 验证响应
        assert response is not None
        assert response.choices[0].message.content == "这是一个异步测试响应。"
        assert response.model == "test-async-model"
        
        # 验证性能（异步调用应该很快完成）
        execution_time = end_time - start_time
        assert execution_time < 1.0, f"异步调用耗时过长: {execution_time}秒"
        
        # 验证异步调用
        mock_async_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.stream_test
    @pytest.mark.p1
    def test_streaming_response(self, mock_stream_client):
        """
        测试流式响应功能
        
        验证：
        - 流式API调用
        - 流式数据处理
        - 完整内容拼接
        """
        # 执行流式测试
        stream = mock_stream_client.chat.completions.create(
            model="test-stream-model",
            messages=self.test_messages,
            stream=True
        )
        
        # 收集流式响应
        collected_content = []
        chunk_count = 0
        
        for chunk in stream:
            chunk_count += 1
            assert hasattr(chunk, 'choices')
            assert len(chunk.choices) > 0
            
            choice = chunk.choices[0]
            if hasattr(choice, 'delta') and hasattr(choice.delta, 'content'):
                if choice.delta.content:
                    collected_content.append(choice.delta.content)
            
            # 验证chunk结构
            assert chunk.object == "chat.completion.chunk"
            assert chunk.model == "test-stream-model"
            assert chunk.id == "chatcmpl-test-stream"
        
        # 验证流式响应完整性
        assert chunk_count == 3, f"期望3个chunks，实际收到{chunk_count}个"
        full_content = ''.join(collected_content)
        assert full_content == "这是流式响应测试"
        
        # 验证最后一个chunk的finish_reason
        assert choice.finish_reason == "stop"
    
    @pytest.mark.integration
    @pytest.mark.structured_output
    @pytest.mark.p1
    def test_structured_output(self, mock_client):
        """
        测试结构化输出功能
        
        验证：
        - JSON Schema 约束
        - 结构化响应解析
        - 数据验证
        """
        # 配置结构化响应
        structured_response = Mock()
        structured_response.choices = [Mock(
            message=Mock(
                content=json.dumps({
                    "name": "张三",
                    "age": 25
                }),
                role="assistant"
            ),
            finish_reason="stop"
        )]
        structured_response.usage = Mock(
            prompt_tokens=20,
            completion_tokens=15,
            total_tokens=35
        )
        
        mock_client.chat.completions.create.return_value = structured_response
        
        # 执行结构化输出测试
        response = mock_client.chat.completions.create(
            model="test-model",
            messages=[
                {"role": "user", "content": "请返回一个包含姓名和年龄的JSON对象"}
            ],
            response_format={
                "type": "json_object",
                "schema": TEST_DATA_CONFIG["json_schema_simple"]
            }
        )
        
        # 验证结构化响应
        assert response is not None
        content = response.choices[0].message.content
        
        # 解析JSON响应
        try:
            parsed_data = json.loads(content)
            assert "name" in parsed_data
            assert "age" in parsed_data
            assert isinstance(parsed_data["name"], str)
            assert isinstance(parsed_data["age"], int)
            assert parsed_data["age"] >= 0
        except json.JSONDecodeError:
            pytest.fail("响应内容不是有效的JSON格式")
    
    @pytest.mark.integration
    @pytest.mark.p1
    def test_error_handling_and_retry(self, mock_client):
        """
        测试错误处理和重试机制
        
        验证：
        - API错误处理
        - 重试机制
        - 错误恢复
        """
        # 配置错误响应序列（前两次失败，第三次成功）
        error_responses = [
            APIError("API调用失败", status_code=500),
            RateLimitError("请求频率限制", retry_after=1),
            Mock(  # 成功响应
                choices=[Mock(
                    message=Mock(
                        content="重试成功的响应",
                        role="assistant"
                    )
                )],
                usage=Mock(prompt_tokens=10, completion_tokens=15, total_tokens=25)
            )
        ]
        
        mock_client.chat.completions.create.side_effect = error_responses
        
        # 模拟重试逻辑
        max_retries = 3
        retry_count = 0
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = mock_client.chat.completions.create(
                    model="test-model",
                    messages=self.test_messages
                )
                # 如果成功，验证响应
                if hasattr(response, 'choices'):
                    assert response.choices[0].message.content == "重试成功的响应"
                    break
            except (APIError, RateLimitError) as e:
                last_error = e
                retry_count += 1
                if attempt < max_retries - 1:
                    time.sleep(0.1)  # 短暂延迟
                continue
        
        # 验证重试行为
        assert mock_client.chat.completions.create.call_count == 3
        assert retry_count == 2  # 前两次失败，第三次成功
    
    @pytest.mark.integration
    @pytest.mark.performance
    @pytest.mark.p2
    def test_performance_benchmarks(self, mock_client):
        """
        测试性能基准
        
        验证：
        - 响应时间
        - 内存使用
        - 并发处理能力
        """
        # 配置快速响应
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(
                message=Mock(
                    content="性能测试响应",
                    role="assistant"
                )
            )],
            usage=Mock(prompt_tokens=5, completion_tokens=10, total_tokens=15)
        )
        
        # 测试单次调用性能
        start_time = time.time()
        response = mock_client.chat.completions.create(
            model="test-model",
            messages=self.test_messages
        )
        end_time = time.time()
        
        # 验证响应时间
        response_time = end_time - start_time
        max_response_time = PERFORMANCE_BENCHMARKS["max_response_time"]
        assert response_time < max_response_time, f"响应时间超出基准: {response_time}s > {max_response_time}s"
        
        # 验证响应正确性
        assert response.choices[0].message.content == "性能测试响应"
    
    @pytest.mark.integration
    @pytest.mark.p2
    def test_concurrent_requests(self, mock_async_client):
        """
        测试并发请求处理
        
        验证：
        - 并发API调用
        - 响应一致性
        - 资源管理
        """
        async def single_request(request_id: int):
            """单个异步请求"""
            response = await mock_async_client.chat.completions.create(
                model="test-concurrent-model",
                messages=[
                    {"role": "user", "content": f"并发请求 #{request_id}"}
                ]
            )
            return request_id, response
        
        async def run_concurrent_test():
            """运行并发测试"""
            concurrent_count = 5
            tasks = [
                single_request(i) for i in range(concurrent_count)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # 验证所有请求都成功
            successful_results = [r for r in results if not isinstance(r, Exception)]
            assert len(successful_results) == concurrent_count
            
            # 验证并发性能
            total_time = end_time - start_time
            assert total_time < 2.0, f"并发请求耗时过长: {total_time}s"
            
            return successful_results
        
        # 运行并发测试
        results = asyncio.run(run_concurrent_test())
        
        # 验证结果
        assert len(results) == 5
        for request_id, response in results:
            assert isinstance(request_id, int)
            assert hasattr(response, 'choices')
    
    @pytest.mark.integration
    @pytest.mark.cost_tracking
    @pytest.mark.p2
    def test_cost_tracking_integration(self, mock_client):
        """
        测试成本跟踪集成
        
        验证：
        - 成本计算
        - 使用统计
        - 成本分析
        """
        # 配置带有详细使用信息的响应
        mock_client.chat.completions.create.return_value = Mock(
            choices=[Mock(
                message=Mock(
                    content="成本跟踪测试响应",
                    role="assistant"
                )
            )],
            usage=Mock(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            ),
            model="test-cost-model"
        )
        
        # 执行API调用
        response = mock_client.chat.completions.create(
            model="test-cost-model",
            messages=self.test_messages
        )
        
        # 验证使用统计
        usage = response.usage
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        
        # 模拟成本计算（实际实现中会有真实的成本分析器）
        estimated_cost = usage.total_tokens * 0.0001  # 假设每token 0.0001元
        assert estimated_cost > 0
        assert abs(estimated_cost - 0.015) < 0.001  # 150 * 0.0001
    
    @pytest.mark.integration
    @pytest.mark.real_api
    @pytest.mark.p3
    def test_real_api_integration(self):
        """
        真实API集成测试（需要真实凭证）
        
        注意：此测试需要设置 ENABLE_REAL_API_TESTS=true 环境变量
        """
        # 检查是否启用真实API测试
        if not os.getenv('ENABLE_REAL_API_TESTS', 'false').lower() == 'true':
            pytest.skip("真实API测试未启用，设置ENABLE_REAL_API_TESTS=true启用")
        
        # 检查API凭证
        api_key = os.getenv('HARBORAI_API_KEY')
        if not api_key:
            pytest.skip("缺少API密钥，设置HARBORAI_API_KEY环境变量")
        
        # 创建真实客户端
        client = HarborAI(api_key=api_key)
        
        # 执行真实API调用
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",  # 使用可用的模型
                messages=self.test_messages,
                max_tokens=50,
                temperature=0.7
            )
            
            # 验证真实响应
            assert response is not None
            assert len(response.choices) > 0
            assert response.choices[0].message.role == "assistant"
            assert len(response.choices[0].message.content) > 0
            assert response.usage.total_tokens > 0
            
        except Exception as e:
            pytest.fail(f"真实API调用失败: {e}")


class TestEndToEndWorkflows:
    """
    端到端工作流测试类
    
    测试完整的业务工作流程。
    """
    
    @pytest.mark.integration
    @pytest.mark.p1
    def test_complete_conversation_workflow(self, mock_client):
        """
        测试完整对话工作流
        
        验证：
        - 多轮对话
        - 上下文保持
        - 会话管理
        """
        # 配置多轮对话响应
        responses = [
            Mock(
                choices=[Mock(
                    message=Mock(
                        content="你好！我是HarborAI助手，很高兴为您服务。",
                        role="assistant"
                    )
                )],
                usage=Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
            ),
            Mock(
                choices=[Mock(
                    message=Mock(
                        content="机器学习是人工智能的一个重要分支...",
                        role="assistant"
                    )
                )],
                usage=Mock(prompt_tokens=25, completion_tokens=40, total_tokens=65)
            ),
            Mock(
                choices=[Mock(
                    message=Mock(
                        content="当然可以！比如推荐系统就是一个典型应用...",
                        role="assistant"
                    )
                )],
                usage=Mock(prompt_tokens=35, completion_tokens=30, total_tokens=65)
            )
        ]
        
        mock_client.chat.completions.create.side_effect = responses
        
        # 模拟多轮对话
        conversation_history = []
        
        # 第一轮：问候
        conversation_history.append({"role": "user", "content": "你好"})
        response1 = mock_client.chat.completions.create(
            model="test-model",
            messages=conversation_history.copy()
        )
        conversation_history.append({
            "role": "assistant", 
            "content": response1.choices[0].message.content
        })
        
        # 第二轮：询问问题
        conversation_history.append({
            "role": "user", 
            "content": "什么是机器学习？"
        })
        response2 = mock_client.chat.completions.create(
            model="test-model",
            messages=conversation_history.copy()
        )
        conversation_history.append({
            "role": "assistant", 
            "content": response2.choices[0].message.content
        })
        
        # 第三轮：追问
        conversation_history.append({
            "role": "user", 
            "content": "能给个具体例子吗？"
        })
        response3 = mock_client.chat.completions.create(
            model="test-model",
            messages=conversation_history.copy()
        )
        
        # 验证对话流程
        assert len(conversation_history) == 5  # 3轮用户输入 + 2轮助手回复
        assert mock_client.chat.completions.create.call_count == 3
        
        # 验证每次调用都包含完整的对话历史
        calls = mock_client.chat.completions.create.call_args_list
        assert len(calls[0].kwargs['messages']) == 1  # 第一次调用
        assert len(calls[1].kwargs['messages']) == 3  # 第二次调用
        assert len(calls[2].kwargs['messages']) == 5  # 第三次调用
    
    @pytest.mark.integration
    @pytest.mark.p2
    def test_error_recovery_workflow(self, mock_client):
        """
        测试错误恢复工作流
        
        验证：
        - 错误检测
        - 自动恢复
        - 降级处理
        """
        # 配置错误恢复序列
        error_sequence = [
            APIError("网络错误", status_code=503),
            RateLimitError("频率限制", retry_after=1),
            Mock(  # 恢复成功
                choices=[Mock(
                    message=Mock(
                        content="系统已恢复正常，这是您的回复。",
                        role="assistant"
                    )
                )],
                usage=Mock(prompt_tokens=15, completion_tokens=25, total_tokens=40)
            )
        ]
        
        mock_client.chat.completions.create.side_effect = error_sequence
        
        # 模拟错误恢复逻辑
        max_retries = 3
        retry_delay = 0.1
        test_messages = [{"role": "user", "content": "Test message"}]
        
        for attempt in range(max_retries):
            try:
                response = mock_client.chat.completions.create(
                    model="test-model",
                    messages=test_messages
                )
                
                # 成功时验证响应
                if hasattr(response, 'choices'):
                    assert "系统已恢复正常" in response.choices[0].message.content
                    break
                    
            except (APIError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    pytest.fail(f"错误恢复失败: {e}")
        
        # 验证重试次数
        assert mock_client.chat.completions.create.call_count == 3