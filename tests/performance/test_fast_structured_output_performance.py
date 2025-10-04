#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST模式结构化输出性能测试

测试目标：
1. 验证FAST模式结构化输出性能接近直接Agently调用
2. 建立性能基准和目标验证机制
3. 测试各项优化组件的性能表现

遵循TDD原则：先写失败测试，再实现优化
"""

import os
import sys
import time
import json
import pytest
import statistics
from typing import Dict, Any
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 加载环境变量
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    load_dotenv(env_path, override=True)
    print(f"✓ 环境变量已从 {env_path} 加载")
except ImportError:
    print("⚠ python-dotenv未安装，直接使用环境变量")

from harborai import HarborAI
from harborai.config.performance import PerformanceMode, reset_performance_config

@pytest.fixture
def setup_fast_mode():
    """设置FAST模式测试环境"""
    # 确保环境变量已加载
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("缺少DEEPSEEK_API_KEY环境变量")
    
    # 设置FAST模式
    os.environ["HARBORAI_PERFORMANCE_MODE"] = "fast"
    
    # 创建客户端
    client = HarborAI()
    
    return {
        "client": client,
        "model": "deepseek-chat",
        "test_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name", "age"]
        }
    }

class TestFastModeStructuredOutputPerformance:
    """FAST模式结构化输出性能测试"""
    
    @pytest.fixture(autouse=True)
    def setup_fast_mode_auto(self):
        """设置FAST模式"""
        reset_performance_config(PerformanceMode.FAST)
        yield
        # 测试后重置为默认模式
        reset_performance_config()
    
    def test_fast_mode_performance_target(self):
        """
        测试FAST模式性能目标：应接近直接Agently调用
        
        目标：FAST模式性能应在直接Agently调用的1.3倍以内
        这个测试预期会失败，直到我们实现了优化功能
        """
        # 确保环境变量已加载
        if not os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("缺少DEEPSEEK_API_KEY环境变量")
        
        # 测试配置
        test_rounds = 3
        schema = self._create_test_schema()
        user_query = "请分析这段文本的情感：'今天天气真好，心情很愉快！'"
        
        # 测试HarborAI FAST模式
        harborai_times = []
        client = HarborAI()
        
        for i in range(test_rounds):
            print(f"执行HarborAI第{i+1}轮测试...")
            start_time = time.perf_counter()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": user_query}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sentiment_analysis",
                        "schema": schema,
                        "strict": True
                    }
                },
                structured_provider="agently",
                temperature=0.1,
                max_tokens=1000
            )
            end_time = time.perf_counter()
            harborai_times.append(end_time - start_time)
            
            # 验证结果正确性
            assert response is not None
            assert hasattr(response.choices[0].message, 'parsed')
            assert response.choices[0].message.parsed is not None
            print(f"HarborAI第{i+1}轮耗时: {harborai_times[-1]:.4f}秒")
        
        # 测试直接Agently调用
        agently_times = []
        self._configure_agently()
        
        for i in range(test_rounds):
            print(f"执行Agently第{i+1}轮测试...")
            start_time = time.perf_counter()
            agent = self._create_agently_agent()
            agently_format = self._convert_schema_to_agently(schema)
            result = agent.input(user_query).output(agently_format).start()
            end_time = time.perf_counter()
            agently_times.append(end_time - start_time)
            
            # 验证结果正确性
            assert result is not None
            print(f"Agently第{i+1}轮耗时: {agently_times[-1]:.4f}秒")
        
        # 性能对比分析
        harborai_avg = statistics.mean(harborai_times)
        agently_avg = statistics.mean(agently_times)
        performance_ratio = harborai_avg / agently_avg
        
        print(f"\n=== 性能对比结果 ===")
        print(f"HarborAI FAST模式平均时间: {harborai_avg:.4f}秒")
        print(f"直接Agently平均时间: {agently_avg:.4f}秒")
        print(f"性能比率: {performance_ratio:.2f}x")
        print(f"性能差距: {((performance_ratio - 1) * 100):.1f}%")
        
        # 保存性能基准数据
        self._save_performance_baseline(harborai_avg, agently_avg, performance_ratio)
        
        # 性能目标：FAST模式应在直接Agently调用的1.3倍以内
        # 这个断言预期会失败，直到我们实现优化
        assert performance_ratio <= 1.3, f"FAST模式性能未达标：{performance_ratio:.2f}x > 1.3x"
    
    def test_client_reuse_performance_improvement(self):
        """
        测试客户端复用的性能提升效果
        
        这个测试将在实现AgentlyClientPool后通过
        """
        pytest.skip("等待AgentlyClientPool实现")
    
    def test_parameter_cache_effectiveness(self):
        """
        测试参数缓存的有效性
        
        这个测试将在实现ParameterCache后通过
        """
        pytest.skip("等待ParameterCache实现")
    
    def test_fast_structured_handler_performance(self):
        """
        测试快速结构化输出处理器的性能
        
        这个测试将在实现FastStructuredOutputHandler后通过
        """
        pytest.skip("等待FastStructuredOutputHandler实现")
    
    def test_memory_usage_optimization(self):
        """
        测试内存使用优化效果
        
        目标：FAST模式内存使用减少30-50%
        """
        pytest.skip("等待内存优化实现")
    
    def _create_test_schema(self) -> Dict[str, Any]:
        """创建测试用的JSON Schema"""
        return {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "情感分析结果"
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "情感倾向"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "置信度"
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词列表"
                }
            },
            "required": ["analysis", "sentiment", "confidence"]
        }
    
    def _configure_agently(self):
        """配置Agently"""
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        if not api_key:
            pytest.skip("需要设置DEEPSEEK_API_KEY环境变量")
        
        from agently import Agently
        Agently.set_settings(
            "OpenAICompatible",
            {
                "base_url": base_url,
                "model": "deepseek-chat",
                "model_type": "chat",
                "auth": api_key,
            },
        )
    
    def _create_agently_agent(self):
        """创建Agently代理"""
        from agently import Agently
        return Agently.create_agent()
    
    def _convert_schema_to_agently(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """将JSON Schema转换为Agently格式"""
        return {
            "analysis": ("str", "情感分析结果"),
            "sentiment": ("str", "情感倾向: positive/negative/neutral"),
            "confidence": ("float", "置信度(0-1)"),
            "keywords": ("list", "关键词列表")
        }
    
    def _save_performance_baseline(self, harborai_time: float, agently_time: float, ratio: float):
        """保存性能基准数据"""
        baseline_data = {
            "timestamp": time.time(),
            "harborai_avg_time": harborai_time,
            "agently_avg_time": agently_time,
            "performance_ratio": ratio,
            "test_mode": "FAST",
            "optimization_status": "未优化"
        }
        
        baseline_file = os.path.join(project_root, "performance_baseline.json")
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2, ensure_ascii=False)
        
        print(f"性能基准数据已保存到: {baseline_file}")


class TestStructuredOutputCompatibility:
    """结构化输出兼容性测试"""
    
    def test_fast_mode_api_compatibility(self):
        """测试FAST模式API兼容性"""
        # 确保环境变量已加载
        if not os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("缺少DEEPSEEK_API_KEY环境变量")
        
        # 确保FAST模式与现有API完全兼容
        reset_performance_config(PerformanceMode.FAST)
        
        client = HarborAI()
        schema = {
            "type": "object",
            "properties": {
                "result": {"type": "string", "description": "测试结果"}
            },
            "required": ["result"]
        }
        
        # 测试API调用格式保持一致
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "请回答：测试"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "test_response",
                    "schema": schema,
                    "strict": True
                }
            },
            structured_provider="agently"
        )
        
        # 验证响应格式
        assert response is not None
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert hasattr(response.choices[0], 'message')
        assert hasattr(response.choices[0].message, 'parsed')
        
        reset_performance_config()
    
    def test_response_format_consistency(self):
        """测试响应格式一致性"""
        # 确保环境变量已加载
        if not os.getenv("DEEPSEEK_API_KEY"):
            pytest.skip("缺少DEEPSEEK_API_KEY环境变量")
        
        # 确保FAST和FULL模式返回相同格式的响应
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "回答"}
            },
            "required": ["answer"]
        }
        
        messages = [{"role": "user", "content": "请简单回答：你好"}]
        
        # 测试FAST模式
        reset_performance_config(PerformanceMode.FAST)
        client_fast = HarborAI()
        response_fast = client_fast.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": schema, "strict": True}
            },
            structured_provider="agently"
        )
        
        # 测试FULL模式
        reset_performance_config(PerformanceMode.FULL)
        client_full = HarborAI()
        response_full = client_full.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "test", "schema": schema, "strict": True}
            },
            structured_provider="agently"
        )
        
        # 验证响应结构一致性
        assert type(response_fast) == type(response_full)
        assert hasattr(response_fast.choices[0].message, 'parsed')
        assert hasattr(response_full.choices[0].message, 'parsed')
        assert type(response_fast.choices[0].message.parsed) == type(response_full.choices[0].message.parsed)
        
        reset_performance_config()


if __name__ == "__main__":
    # 直接运行性能测试
    test_instance = TestFastModeStructuredOutputPerformance()
    test_instance.setup_fast_mode_auto()
    
    try:
        test_instance.test_fast_mode_performance_target()
    except AssertionError as e:
        print(f"预期的测试失败: {e}")
        print("这是正常的，因为优化功能尚未实现")
    except Exception as e:
        print(f"测试执行错误: {e}")
    finally:
        # 清理
        reset_performance_config()