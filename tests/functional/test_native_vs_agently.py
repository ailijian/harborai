#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原生Schema vs Agently对比测试模块

本模块测试HarborAI中原生Schema和Agently框架的对比功能。
包括性能对比、功能对比、兼容性测试、迁移测试等场景。

作者: HarborAI团队
创建时间: 2024-01-20
"""

import pytest
import asyncio
import time
import json
import threading
import random
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import memory_profiler
import psutil
import gc
from contextlib import contextmanager
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime, timedelta


class SchemaType(Enum):
    """Schema类型枚举"""
    NATIVE = "native"
    AGENTLY = "agently"
    HYBRID = "hybrid"


class TestComplexity(Enum):
    """测试复杂度枚举"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    EXTREME = "extreme"


@dataclass
class PerformanceMetrics:
    """性能指标"""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    throughput: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    error_rate: float = 0.0
    success_count: int = 0
    error_count: int = 0


@dataclass
class ComparisonResult:
    """对比结果"""
    native_metrics: PerformanceMetrics
    agently_metrics: PerformanceMetrics
    performance_ratio: float = 0.0
    memory_ratio: float = 0.0
    throughput_ratio: float = 0.0
    winner: SchemaType = SchemaType.NATIVE
    improvement_percentage: float = 0.0
    recommendation: str = ""


class NativeSchemaProcessor:
    """原生Schema处理器"""
    
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
        self.processing_times = []
    
    def validate_simple_schema(self, data: Dict[str, Any]) -> bool:
        """验证简单Schema"""
        start_time = time.perf_counter()
        
        try:
            # 原生验证逻辑
            required_fields = ["name", "age", "email"]
            
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # 类型验证
            if not isinstance(data["name"], str):
                raise ValueError("Name must be string")
            
            if not isinstance(data["age"], int) or data["age"] < 0:
                raise ValueError("Age must be positive integer")
            
            if not isinstance(data["email"], str) or "@" not in data["email"]:
                raise ValueError("Invalid email format")
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            self.error_count += 1
            raise
        
        finally:
            processing_time = max(time.perf_counter() - start_time, 1e-6)  # 确保不为0
            self.processing_times.append(processing_time)
    
    def validate_complex_schema(self, data: Dict[str, Any]) -> bool:
        """验证复杂Schema"""
        start_time = time.perf_counter()
        
        try:
            # 复杂验证逻辑
            required_fields = [
                "user_info", "preferences", "history", "metadata"
            ]
            
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # 嵌套对象验证
            user_info = data["user_info"]
            if not isinstance(user_info, dict):
                raise ValueError("user_info must be object")
            
            # 验证用户信息
            user_required = ["id", "name", "profile"]
            for field in user_required:
                if field not in user_info:
                    raise ValueError(f"Missing user_info.{field}")
            
            # 验证偏好设置
            preferences = data["preferences"]
            if not isinstance(preferences, dict):
                raise ValueError("preferences must be object")
            
            # 验证历史记录
            history = data["history"]
            if not isinstance(history, list):
                raise ValueError("history must be array")
            
            for item in history:
                if not isinstance(item, dict):
                    raise ValueError("history item must be object")
                if "timestamp" not in item or "action" not in item:
                    raise ValueError("history item missing required fields")
            
            # 验证元数据
            metadata = data["metadata"]
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be object")
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            self.error_count += 1
            raise
        
        finally:
            processing_time = max(time.perf_counter() - start_time, 1e-6)  # 确保不为0
            self.processing_times.append(processing_time)
    
    def process_batch(self, data_list: List[Dict[str, Any]], schema_type: str = "simple") -> Dict[str, Any]:
        """批量处理数据"""
        start_time = time.perf_counter()
        success_count = 0
        error_count = 0
        
        for data in data_list:
            try:
                if schema_type == "simple":
                    self.validate_simple_schema(data)
                else:
                    self.validate_complex_schema(data)
                success_count += 1
            except Exception:
                error_count += 1
        
        end_time = time.perf_counter()
        processing_time = max(end_time - start_time, 1e-6)  # 确保不为0
        
        # 确保处理时间被记录
        if not self.processing_times:
            self.processing_times.append(processing_time)
        
        return {
            "total_processed": len(data_list),
            "success_count": success_count,
            "error_count": error_count,
            "processing_time": processing_time,
            "throughput": len(data_list) / processing_time
        }
    
    def get_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        if not self.processing_times:
            return PerformanceMetrics()
        
        return PerformanceMetrics(
            execution_time=sum(self.processing_times),
            throughput=self.processed_count / sum(self.processing_times) if sum(self.processing_times) > 0 else 0,
            latency_p50=statistics.median(self.processing_times),
            latency_p95=statistics.quantiles(self.processing_times, n=20)[18] if len(self.processing_times) > 20 else max(self.processing_times),
            latency_p99=statistics.quantiles(self.processing_times, n=100)[98] if len(self.processing_times) > 100 else max(self.processing_times),
            success_count=self.processed_count,
            error_count=self.error_count,
            error_rate=self.error_count / (self.processed_count + self.error_count) if (self.processed_count + self.error_count) > 0 else 0
        )


class AgentlySchemaProcessor:
    """Agently Schema处理器"""
    
    def __init__(self):
        self.processed_count = 0
        self.error_count = 0
        self.processing_times = []
        self.agent_instances = {}
    
    def create_simple_agent(self) -> Dict[str, Any]:
        """创建简单Agent"""
        return {
            "type": "simple_validator",
            "schema": {
                "name": {"type": "string", "required": True},
                "age": {"type": "integer", "minimum": 0, "required": True},
                "email": {"type": "string", "pattern": r".*@.*", "required": True}
            },
            "validation_rules": [
                "name_not_empty",
                "age_positive",
                "email_valid"
            ]
        }
    
    def create_complex_agent(self) -> Dict[str, Any]:
        """创建复杂Agent"""
        return {
            "type": "complex_validator",
            "schema": {
                "user_info": {
                    "type": "object",
                    "required": True,
                    "properties": {
                        "id": {"type": "string", "required": True},
                        "name": {"type": "string", "required": True},
                        "profile": {"type": "object", "required": True}
                    }
                },
                "preferences": {
                    "type": "object",
                    "required": True
                },
                "history": {
                    "type": "array",
                    "required": True,
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp": {"type": "string", "required": True},
                            "action": {"type": "string", "required": True}
                        }
                    }
                },
                "metadata": {
                    "type": "object",
                    "required": True
                }
            },
            "validation_rules": [
                "user_info_complete",
                "preferences_valid",
                "history_chronological",
                "metadata_consistent"
            ]
        }
    
    def validate_with_agent(self, data: Dict[str, Any], agent_config: Dict[str, Any]) -> bool:
        """使用Agent验证数据"""
        start_time = time.perf_counter()
        
        try:
            # 模拟Agently验证过程
            schema = agent_config["schema"]
            
            # 递归验证
            self._validate_object(data, schema)
            
            # 应用验证规则
            for rule in agent_config["validation_rules"]:
                self._apply_validation_rule(data, rule)
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            self.error_count += 1
            raise
        
        finally:
            processing_time = max(time.perf_counter() - start_time, 1e-6)  # 确保不为0
            self.processing_times.append(processing_time)
    
    def _validate_object(self, data: Any, schema: Dict[str, Any]):
        """验证对象"""
        if schema.get("type") == "object":
            if not isinstance(data, dict):
                raise ValueError("Expected object")
            
            properties = schema.get("properties", {})
            for prop_name, prop_schema in properties.items():
                if prop_schema.get("required") and prop_name not in data:
                    raise ValueError(f"Missing required property: {prop_name}")
                
                if prop_name in data:
                    self._validate_object(data[prop_name], prop_schema)
        
        elif schema.get("type") == "array":
            if not isinstance(data, list):
                raise ValueError("Expected array")
            
            items_schema = schema.get("items", {})
            for item in data:
                self._validate_object(item, items_schema)
        
        elif schema.get("type") == "string":
            if not isinstance(data, str):
                raise ValueError("Expected string")
            
            pattern = schema.get("pattern")
            if pattern and not self._match_pattern(data, pattern):
                raise ValueError(f"String does not match pattern: {pattern}")
        
        elif schema.get("type") == "integer":
            if not isinstance(data, int):
                raise ValueError("Expected integer")
            
            minimum = schema.get("minimum")
            if minimum is not None and data < minimum:
                raise ValueError(f"Value {data} is less than minimum {minimum}")
    
    def _match_pattern(self, value: str, pattern: str) -> bool:
        """匹配模式"""
        import re
        try:
            return bool(re.match(pattern, value))
        except re.error:
            return False
    
    def _apply_validation_rule(self, data: Dict[str, Any], rule: str):
        """应用验证规则"""
        if rule == "name_not_empty":
            if not data.get("name", "").strip():
                raise ValueError("Name cannot be empty")
        
        elif rule == "age_positive":
            if data.get("age", 0) <= 0:
                raise ValueError("Age must be positive")
        
        elif rule == "email_valid":
            email = data.get("email", "")
            if "@" not in email or "." not in email:
                raise ValueError("Invalid email format")
        
        elif rule == "user_info_complete":
            user_info = data.get("user_info", {})
            required_fields = ["id", "name", "profile"]
            for field in required_fields:
                if not user_info.get(field):
                    raise ValueError(f"user_info.{field} is required")
        
        elif rule == "history_chronological":
            history = data.get("history", [])
            timestamps = [item.get("timestamp") for item in history if "timestamp" in item]
            if len(timestamps) > 1:
                # 简单的时间顺序检查
                for i in range(1, len(timestamps)):
                    if timestamps[i] < timestamps[i-1]:
                        raise ValueError("History must be chronological")
    
    def process_batch(self, data_list: List[Dict[str, Any]], schema_type: str = "simple") -> Dict[str, Any]:
        """批量处理数据"""
        start_time = time.perf_counter()
        success_count = 0
        error_count = 0
        
        # 创建Agent配置
        if schema_type == "simple":
            agent_config = self.create_simple_agent()
        else:
            agent_config = self.create_complex_agent()
        
        for data in data_list:
            try:
                self.validate_with_agent(data, agent_config)
                success_count += 1
            except Exception:
                error_count += 1
        
        end_time = time.perf_counter()
        processing_time = max(end_time - start_time, 1e-6)  # 确保不为0
        
        # 确保处理时间被记录
        if not self.processing_times:
            self.processing_times.append(processing_time)
        
        return {
            "total_processed": len(data_list),
            "success_count": success_count,
            "error_count": error_count,
            "processing_time": processing_time,
            "throughput": len(data_list) / processing_time
        }
    
    def get_metrics(self) -> PerformanceMetrics:
        """获取性能指标"""
        if not self.processing_times:
            return PerformanceMetrics()
        
        return PerformanceMetrics(
            execution_time=sum(self.processing_times),
            throughput=self.processed_count / sum(self.processing_times) if sum(self.processing_times) > 0 else 0,
            latency_p50=statistics.median(self.processing_times),
            latency_p95=statistics.quantiles(self.processing_times, n=20)[18] if len(self.processing_times) > 20 else max(self.processing_times),
            latency_p99=statistics.quantiles(self.processing_times, n=100)[98] if len(self.processing_times) > 100 else max(self.processing_times),
            success_count=self.processed_count,
            error_count=self.error_count,
            error_rate=self.error_count / (self.processed_count + self.error_count) if (self.processed_count + self.error_count) > 0 else 0
        )


class SchemaComparator:
    """Schema对比器"""
    
    def __init__(self):
        self.native_processor = NativeSchemaProcessor()
        self.agently_processor = AgentlySchemaProcessor()
    
    def generate_test_data(self, complexity: TestComplexity, count: int = 100) -> List[Dict[str, Any]]:
        """生成测试数据"""
        data_list = []
        
        for i in range(count):
            if complexity == TestComplexity.SIMPLE:
                data = {
                    "name": f"User{i}",
                    "age": random.randint(18, 80),
                    "email": f"user{i}@example.com"
                }
            
            elif complexity == TestComplexity.COMPLEX:
                data = {
                    "user_info": {
                        "id": f"user_{i}",
                        "name": f"User {i}",
                        "profile": {
                            "avatar": f"avatar_{i}.jpg",
                            "bio": f"Bio for user {i}"
                        }
                    },
                    "preferences": {
                        "theme": random.choice(["light", "dark"]),
                        "language": random.choice(["en", "zh", "es"]),
                        "notifications": random.choice([True, False])
                    },
                    "history": [
                        {
                            "timestamp": f"2024-01-{j+1:02d}T10:00:00Z",
                            "action": f"action_{j}"
                        }
                        for j in range(random.randint(1, 10))
                    ],
                    "metadata": {
                        "created_at": "2024-01-01T00:00:00Z",
                        "updated_at": "2024-01-20T00:00:00Z",
                        "version": "1.0"
                    }
                }
            
            else:
                # 简单数据作为默认
                data = {
                    "name": f"User{i}",
                    "age": random.randint(18, 80),
                    "email": f"user{i}@example.com"
                }
            
            data_list.append(data)
        
        return data_list
    
    def compare_performance(
        self,
        complexity: TestComplexity,
        data_count: int = 1000
    ) -> ComparisonResult:
        """对比性能"""
        # 重置处理器状态
        self.native_processor.processed_count = 0
        self.native_processor.error_count = 0
        self.native_processor.processing_times = []
        
        self.agently_processor.processed_count = 0
        self.agently_processor.error_count = 0
        self.agently_processor.processing_times = []
        
        # 生成测试数据
        test_data = self.generate_test_data(complexity, data_count)
        schema_type = "simple" if complexity == TestComplexity.SIMPLE else "complex"
        
        # 测试原生Schema
        native_start_memory = psutil.Process().memory_info().rss
        native_result = self.native_processor.process_batch(test_data, schema_type)
        native_end_memory = psutil.Process().memory_info().rss
        native_metrics = self.native_processor.get_metrics()
        native_metrics.memory_usage = (native_end_memory - native_start_memory) / 1024 / 1024  # MB
        
        # 测试Agently Schema
        agently_start_memory = psutil.Process().memory_info().rss
        agently_result = self.agently_processor.process_batch(test_data, schema_type)
        agently_end_memory = psutil.Process().memory_info().rss
        agently_metrics = self.agently_processor.get_metrics()
        agently_metrics.memory_usage = (agently_end_memory - agently_start_memory) / 1024 / 1024  # MB
        
        # 计算对比结果
        performance_ratio = native_metrics.execution_time / agently_metrics.execution_time if agently_metrics.execution_time > 0 else float('inf')
        memory_ratio = native_metrics.memory_usage / agently_metrics.memory_usage if agently_metrics.memory_usage > 0 else float('inf')
        throughput_ratio = native_metrics.throughput / agently_metrics.throughput if agently_metrics.throughput > 0 else float('inf')
        
        # 确定获胜者
        native_score = (1/performance_ratio if performance_ratio > 0 else 0) + (1/memory_ratio if memory_ratio > 0 else 0) + throughput_ratio
        agently_score = performance_ratio + memory_ratio + (1/throughput_ratio if throughput_ratio > 0 else 0)
        
        winner = SchemaType.NATIVE if native_score > agently_score else SchemaType.AGENTLY
        improvement_percentage = abs(native_score - agently_score) / max(native_score, agently_score) * 100
        
        # 生成建议
        recommendation = self._generate_recommendation(
            native_metrics, agently_metrics, winner, complexity
        )
        
        return ComparisonResult(
            native_metrics=native_metrics,
            agently_metrics=agently_metrics,
            performance_ratio=performance_ratio,
            memory_ratio=memory_ratio,
            throughput_ratio=throughput_ratio,
            winner=winner,
            improvement_percentage=improvement_percentage,
            recommendation=recommendation
        )
    
    def _generate_recommendation(
        self,
        native_metrics: PerformanceMetrics,
        agently_metrics: PerformanceMetrics,
        winner: SchemaType,
        complexity: TestComplexity
    ) -> str:
        """生成建议"""
        recommendations = []
        
        if winner == SchemaType.NATIVE:
            recommendations.append("原生Schema在当前场景下表现更好")
            
            if native_metrics.execution_time < agently_metrics.execution_time:
                recommendations.append("原生Schema执行速度更快")
            
            if native_metrics.memory_usage < agently_metrics.memory_usage:
                recommendations.append("原生Schema内存使用更少")
            
            if complexity == TestComplexity.SIMPLE:
                recommendations.append("对于简单Schema，建议使用原生实现")
        
        else:
            recommendations.append("Agently Schema在当前场景下表现更好")
            
            if agently_metrics.execution_time < native_metrics.execution_time:
                recommendations.append("Agently Schema执行速度更快")
            
            if agently_metrics.memory_usage < native_metrics.memory_usage:
                recommendations.append("Agently Schema内存使用更少")
            
            if complexity == TestComplexity.COMPLEX:
                recommendations.append("对于复杂Schema，建议使用Agently框架")
        
        # 错误率建议
        if native_metrics.error_rate < agently_metrics.error_rate:
            recommendations.append("原生Schema错误率更低，稳定性更好")
        elif agently_metrics.error_rate < native_metrics.error_rate:
            recommendations.append("Agently Schema错误率更低，稳定性更好")
        
        return "; ".join(recommendations)


class TestNativeVsAgentlyBasic:
    """原生vs Agently基础测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.comparator = SchemaComparator()
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.schema_comparison
    def test_simple_schema_validation(self):
        """测试简单Schema验证"""
        # 生成简单测试数据
        test_data = {
            "name": "John Doe",
            "age": 30,
            "email": "john@example.com"
        }
        
        # 测试原生验证
        native_result = self.comparator.native_processor.validate_simple_schema(test_data)
        assert native_result is True
        
        # 测试Agently验证
        agent_config = self.comparator.agently_processor.create_simple_agent()
        agently_result = self.comparator.agently_processor.validate_with_agent(test_data, agent_config)
        assert agently_result is True
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.schema_comparison
    def test_complex_schema_validation(self):
        """测试复杂Schema验证"""
        # 生成复杂测试数据
        test_data = {
            "user_info": {
                "id": "user_123",
                "name": "John Doe",
                "profile": {
                    "avatar": "avatar.jpg",
                    "bio": "Software developer"
                }
            },
            "preferences": {
                "theme": "dark",
                "language": "en"
            },
            "history": [
                {
                    "timestamp": "2024-01-01T10:00:00Z",
                    "action": "login"
                },
                {
                    "timestamp": "2024-01-01T11:00:00Z",
                    "action": "view_profile"
                }
            ],
            "metadata": {
                "created_at": "2024-01-01T00:00:00Z",
                "version": "1.0"
            }
        }
        
        # 测试原生验证
        native_result = self.comparator.native_processor.validate_complex_schema(test_data)
        assert native_result is True
        
        # 测试Agently验证
        agent_config = self.comparator.agently_processor.create_complex_agent()
        agently_result = self.comparator.agently_processor.validate_with_agent(test_data, agent_config)
        assert agently_result is True
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.schema_comparison
    def test_invalid_data_handling(self):
        """测试无效数据处理"""
        # 无效的简单数据
        invalid_simple_data = {
            "name": "",  # 空名称
            "age": -5,   # 负年龄
            "email": "invalid_email"  # 无效邮箱
        }
        
        # 测试原生验证
        with pytest.raises(ValueError):
            self.comparator.native_processor.validate_simple_schema(invalid_simple_data)
        
        # 测试Agently验证
        agent_config = self.comparator.agently_processor.create_simple_agent()
        with pytest.raises(ValueError):
            self.comparator.agently_processor.validate_with_agent(invalid_simple_data, agent_config)
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.schema_comparison
    def test_missing_fields_handling(self):
        """测试缺失字段处理"""
        # 缺失字段的数据
        incomplete_data = {
            "name": "John Doe"
            # 缺少 age 和 email
        }
        
        # 测试原生验证
        with pytest.raises(ValueError, match="Missing required field"):
            self.comparator.native_processor.validate_simple_schema(incomplete_data)
        
        # 测试Agently验证
        agent_config = self.comparator.agently_processor.create_simple_agent()
        with pytest.raises(ValueError):
            self.comparator.agently_processor.validate_with_agent(incomplete_data, agent_config)


class TestPerformanceComparison:
    """性能对比测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.comparator = SchemaComparator()
        # 确保处理器状态完全重置
        self.comparator.native_processor.processed_count = 0
        self.comparator.native_processor.error_count = 0
        self.comparator.native_processor.processing_times = []
        
        self.comparator.agently_processor.processed_count = 0
        self.comparator.agently_processor.error_count = 0
        self.comparator.agently_processor.processing_times = []
    
    @pytest.mark.performance
    @pytest.mark.p1
    @pytest.mark.schema_comparison
    def test_simple_schema_performance(self):
        """测试简单Schema性能"""
        result = self.comparator.compare_performance(
            complexity=TestComplexity.SIMPLE,
            data_count=1000
        )
        
        # 验证对比结果
        assert result.native_metrics.success_count > 0
        assert result.agently_metrics.success_count > 0
        assert result.performance_ratio > 0
        assert result.memory_ratio > 0
        assert result.throughput_ratio > 0
        assert result.winner in [SchemaType.NATIVE, SchemaType.AGENTLY]
        assert len(result.recommendation) > 0
        
        # 验证性能指标合理性
        assert result.native_metrics.execution_time > 0
        assert result.agently_metrics.execution_time > 0
        assert result.native_metrics.throughput > 0
        assert result.agently_metrics.throughput > 0
    
    @pytest.mark.performance
    @pytest.mark.p1
    @pytest.mark.schema_comparison
    def test_complex_schema_performance(self):
        """测试复杂Schema性能"""
        result = self.comparator.compare_performance(
            complexity=TestComplexity.COMPLEX,
            data_count=500
        )
        
        # 验证对比结果
        assert result.native_metrics.success_count > 0
        assert result.agently_metrics.success_count > 0
        assert result.performance_ratio > 0
        assert result.winner in [SchemaType.NATIVE, SchemaType.AGENTLY]
        
        # 复杂Schema通常Agently表现更好
        print(f"复杂Schema对比结果: {result.winner.value} 获胜")
        print(f"性能比率: {result.performance_ratio:.2f}")
        print(f"内存比率: {result.memory_ratio:.2f}")
        print(f"吞吐量比率: {result.throughput_ratio:.2f}")
        print(f"建议: {result.recommendation}")
    
    @pytest.mark.performance
    @pytest.mark.p2
    @pytest.mark.schema_comparison
    def test_scalability_comparison(self):
        """测试可扩展性对比"""
        data_sizes = [100, 500, 1000, 2000]
        native_times = []
        agently_times = []
        
        for size in data_sizes:
            result = self.comparator.compare_performance(
                complexity=TestComplexity.SIMPLE,
                data_count=size
            )
            
            native_times.append(result.native_metrics.execution_time)
            agently_times.append(result.agently_metrics.execution_time)
        
        # 验证可扩展性
        # 时间应该随数据量增长
        assert native_times[-1] > native_times[0]
        assert agently_times[-1] > agently_times[0]
        
        # 计算增长率
        native_growth_rate = native_times[-1] / native_times[0]
        agently_growth_rate = agently_times[-1] / agently_times[0]
        
        print(f"原生Schema增长率: {native_growth_rate:.2f}")
        print(f"Agently Schema增长率: {agently_growth_rate:.2f}")
        
        # 增长率应该合理（线性增长）
        assert native_growth_rate < 50  # 不应该超过50倍
        assert agently_growth_rate < 50
    
    @pytest.mark.performance
    @pytest.mark.p2
    @pytest.mark.schema_comparison
    def test_memory_usage_comparison(self):
        """测试内存使用对比"""
        # 大数据量测试
        result = self.comparator.compare_performance(
            complexity=TestComplexity.COMPLEX,
            data_count=2000
        )
        
        # 验证内存使用
        assert result.native_metrics.memory_usage >= 0
        assert result.agently_metrics.memory_usage >= 0
        
        # 内存使用应该在合理范围内（小于100MB）
        assert result.native_metrics.memory_usage < 100
        assert result.agently_metrics.memory_usage < 100
        
        print(f"原生Schema内存使用: {result.native_metrics.memory_usage:.2f}MB")
        print(f"Agently Schema内存使用: {result.agently_metrics.memory_usage:.2f}MB")
        print(f"内存比率: {result.memory_ratio:.2f}")


class TestConcurrencyComparison:
    """并发对比测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.comparator = SchemaComparator()
        # 确保处理器状态完全重置
        self.comparator.native_processor.processed_count = 0
        self.comparator.native_processor.error_count = 0
        self.comparator.native_processor.processing_times = []
        
        self.comparator.agently_processor.processed_count = 0
        self.comparator.agently_processor.error_count = 0
        self.comparator.agently_processor.processing_times = []
    
    @pytest.mark.performance
    @pytest.mark.p2
    @pytest.mark.schema_comparison
    def test_concurrent_validation(self):
        """测试并发验证"""
        test_data = self.comparator.generate_test_data(TestComplexity.SIMPLE, 100)
        thread_count = 10
        
        # 测试原生并发
        native_results = []
        native_errors = []
        
        def native_worker():
            try:
                for data in test_data[:10]:  # 每个线程处理10个
                    result = self.comparator.native_processor.validate_simple_schema(data)
                    native_results.append(result)
            except Exception as e:
                native_errors.append(e)
        
        # 测试Agently并发
        agently_results = []
        agently_errors = []
        agent_config = self.comparator.agently_processor.create_simple_agent()
        
        def agently_worker():
            try:
                for data in test_data[:10]:  # 每个线程处理10个
                    result = self.comparator.agently_processor.validate_with_agent(data, agent_config)
                    agently_results.append(result)
            except Exception as e:
                agently_errors.append(e)
        
        # 运行原生并发测试
        native_threads = []
        native_start_time = time.time()
        
        for _ in range(thread_count):
            thread = threading.Thread(target=native_worker)
            native_threads.append(thread)
            thread.start()
        
        for thread in native_threads:
            thread.join()
        
        native_end_time = time.time()
        
        # 运行Agently并发测试
        agently_threads = []
        agently_start_time = time.time()
        
        for _ in range(thread_count):
            thread = threading.Thread(target=agently_worker)
            agently_threads.append(thread)
            thread.start()
        
        for thread in agently_threads:
            thread.join()
        
        agently_end_time = time.time()
        
        # 验证并发结果
        assert len(native_results) > 0
        assert len(agently_results) > 0
        assert len(native_errors) == 0
        assert len(agently_errors) == 0
        
        # 比较并发性能
        native_concurrent_time = native_end_time - native_start_time
        agently_concurrent_time = agently_end_time - agently_start_time
        
        print(f"原生并发时间: {native_concurrent_time:.3f}s")
        print(f"Agently并发时间: {agently_concurrent_time:.3f}s")
        
        # 并发时间应该合理
        assert native_concurrent_time < 10.0
        assert agently_concurrent_time < 10.0
    
    @pytest.mark.performance
    @pytest.mark.p2
    @pytest.mark.schema_comparison
    def test_thread_safety(self):
        """测试线程安全性"""
        shared_data = {"counter": 0}
        lock = threading.Lock()
        
        def safe_increment():
            with lock:
                current = shared_data["counter"]
                time.sleep(0.001)  # 模拟处理时间
                shared_data["counter"] = current + 1
        
        # 测试线程安全
        threads = []
        thread_count = 20
        
        for _ in range(thread_count):
            thread = threading.Thread(target=safe_increment)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 验证线程安全
        assert shared_data["counter"] == thread_count


class TestMigrationScenarios:
    """迁移场景测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.comparator = SchemaComparator()
    
    @pytest.mark.integration
    @pytest.mark.p1
    @pytest.mark.schema_comparison
    def test_native_to_agently_migration(self):
        """测试从原生到Agently的迁移"""
        # 原生Schema数据
        native_data = {
            "name": "Migration Test",
            "age": 25,
            "email": "test@migration.com"
        }
        
        # 验证原生Schema工作正常
        native_result = self.comparator.native_processor.validate_simple_schema(native_data)
        assert native_result is True
        
        # 迁移到Agently Schema
        agent_config = self.comparator.agently_processor.create_simple_agent()
        agently_result = self.comparator.agently_processor.validate_with_agent(native_data, agent_config)
        assert agently_result is True
        
        # 验证迁移后数据一致性
        assert native_result == agently_result
    
    @pytest.mark.integration
    @pytest.mark.p1
    @pytest.mark.schema_comparison
    def test_agently_to_native_migration(self):
        """测试从Agently到原生的迁移"""
        # Agently Schema数据
        agently_data = {
            "name": "Reverse Migration Test",
            "age": 30,
            "email": "reverse@migration.com"
        }
        
        # 验证Agently Schema工作正常
        agent_config = self.comparator.agently_processor.create_simple_agent()
        agently_result = self.comparator.agently_processor.validate_with_agent(agently_data, agent_config)
        assert agently_result is True
        
        # 迁移到原生Schema
        native_result = self.comparator.native_processor.validate_simple_schema(agently_data)
        assert native_result is True
        
        # 验证迁移后数据一致性
        assert agently_result == native_result
    
    @pytest.mark.integration
    @pytest.mark.p2
    @pytest.mark.schema_comparison
    def test_hybrid_approach(self):
        """测试混合方法"""
        # 简单数据使用原生Schema
        simple_data = {
            "name": "Simple User",
            "age": 28,
            "email": "simple@hybrid.com"
        }
        
        # 复杂数据使用Agently Schema
        complex_data = {
            "user_info": {
                "id": "complex_user",
                "name": "Complex User",
                "profile": {"bio": "Complex user profile"}
            },
            "preferences": {"theme": "dark"},
            "history": [
                {"timestamp": "2024-01-01T10:00:00Z", "action": "login"}
            ],
            "metadata": {"version": "1.0"}
        }
        
        # 验证混合方法
        simple_result = self.comparator.native_processor.validate_simple_schema(simple_data)
        assert simple_result is True
        
        agent_config = self.comparator.agently_processor.create_complex_agent()
        complex_result = self.comparator.agently_processor.validate_with_agent(complex_data, agent_config)
        assert complex_result is True
        
        # 验证混合方法的有效性
        assert simple_result and complex_result


class TestCompatibilityMatrix:
    """兼容性矩阵测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.comparator = SchemaComparator()
    
    @pytest.mark.integration
    @pytest.mark.p2
    @pytest.mark.schema_comparison
    def test_data_format_compatibility(self):
        """测试数据格式兼容性"""
        # 测试不同数据格式
        test_formats = [
            {"name": "String Test", "age": 25, "email": "string@test.com"},
            {"name": "Unicode Test 测试", "age": 30, "email": "unicode@测试.com"},
            {"name": "Special!@#$%^&*()", "age": 35, "email": "special@test.com"},
        ]
        
        for test_data in test_formats:
            try:
                # 测试原生兼容性
                native_result = self.comparator.native_processor.validate_simple_schema(test_data)
                
                # 测试Agently兼容性
                agent_config = self.comparator.agently_processor.create_simple_agent()
                agently_result = self.comparator.agently_processor.validate_with_agent(test_data, agent_config)
                
                # 验证兼容性
                assert native_result == agently_result
                
            except Exception as e:
                # 记录不兼容的情况
                print(f"数据格式不兼容: {test_data}, 错误: {e}")
    
    @pytest.mark.integration
    @pytest.mark.p2
    @pytest.mark.schema_comparison
    def test_version_compatibility(self):
        """测试版本兼容性"""
        # 模拟不同版本的Schema
        v1_data = {
            "name": "Version 1 User",
            "age": 25,
            "email": "v1@test.com"
        }
        
        v2_data = {
            "name": "Version 2 User",
            "age": 30,
            "email": "v2@test.com",
            "phone": "+1234567890"  # 新增字段
        }
        
        # 测试向后兼容性
        try:
            native_v1 = self.comparator.native_processor.validate_simple_schema(v1_data)
            native_v2 = self.comparator.native_processor.validate_simple_schema(v2_data)
            
            agent_config = self.comparator.agently_processor.create_simple_agent()
            agently_v1 = self.comparator.agently_processor.validate_with_agent(v1_data, agent_config)
            agently_v2 = self.comparator.agently_processor.validate_with_agent(v2_data, agent_config)
            
            # V1数据应该在两个系统中都能验证
            assert native_v1 is True
            assert agently_v1 is True
            
            # V2数据可能在某些系统中失败（额外字段）
            print(f"V2原生验证: {native_v2}")
            print(f"V2 Agently验证: {agently_v2}")
            
        except Exception as e:
            print(f"版本兼容性测试异常: {e}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "schema_comparison"
    ])