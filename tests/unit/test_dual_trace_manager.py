"""
双追踪ID管理器单元测试

测试DualTraceIDManager和DualTraceContext的功能
"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime
from opentelemetry import trace
from opentelemetry.trace import SpanContext, TraceFlags

from harborai.core.tracing.dual_trace_manager import (
    DualTraceIDManager,
    DualTraceContext
)


class TestDualTraceContext:
    """测试DualTraceContext数据类"""
    
    def test_create_context(self):
        """测试创建双追踪上下文"""
        context = DualTraceContext(
            hb_trace_id="hb_123456789_test_random",
            otel_trace_id="abcdef1234567890",
            span_id="span123",
            correlation_id="corr123",
            session_id="sess123",
            user_id="user123",
            service_name="test-service",
            operation_name="test.operation"
        )
        
        assert context.hb_trace_id == "hb_123456789_test_random"
        assert context.otel_trace_id == "abcdef1234567890"
        assert context.span_id == "span123"
        assert context.correlation_id == "corr123"
        assert context.session_id == "sess123"
        assert context.user_id == "user123"
        assert context.service_name == "test-service"
        assert context.operation_name == "test.operation"
    
    def test_dual_trace_context_fields(self):
        """测试DualTraceContext字段"""
        context = DualTraceContext(
            hb_trace_id="hb_123456789_test_random",
            otel_trace_id="abcdef1234567890",
            span_id="span123",
            correlation_id="corr123",
            session_id="sess123",
            user_id="user123",
            service_name="test-service",
            operation_name="test.operation"
        )
        
        # 验证所有字段都存在且值正确
        assert hasattr(context, 'hb_trace_id')
        assert hasattr(context, 'otel_trace_id')
        assert hasattr(context, 'span_id')
        assert hasattr(context, 'correlation_id')
        assert hasattr(context, 'session_id')
        assert hasattr(context, 'user_id')
        assert hasattr(context, 'service_name')
        assert hasattr(context, 'operation_name')


class TestDualTraceIDManager:
    """测试DualTraceIDManager类"""
    
    @pytest.fixture
    def trace_manager(self):
        """创建测试用的双追踪ID管理器"""
        return DualTraceIDManager(
            hb_prefix="test",
            service_name="test-service"
        )
    
    def test_dual_trace_manager_initialization(self, trace_manager):
        """测试双追踪ID管理器初始化"""
        assert trace_manager.hb_prefix == "test"
        assert trace_manager.service_name == "test-service"
        assert hasattr(trace_manager, '_trace_mapping')
        assert hasattr(trace_manager, '_otel_to_hb_mapping')
        assert len(trace_manager._trace_mapping) == 0
        assert len(trace_manager._otel_to_hb_mapping) == 0
    
    def test_generate_hb_trace_id(self, trace_manager):
        """测试生成HarborAI追踪ID"""
        trace_id = trace_manager.generate_hb_trace_id("test.operation")
        
        assert trace_id.startswith("test_")
        # 验证ID格式：prefix_timestamp_service_hash_operation_hash_random
        parts = trace_id.split("_")
        assert len(parts) == 5  # prefix, timestamp, service_hash, operation_hash, random
        
        # 验证时间戳部分是数字
        assert parts[1].isdigit()
        
        # 验证服务哈希和操作哈希长度
        assert len(parts[2]) == 4  # service_hash
        assert len(parts[3]) == 4  # operation_hash
        assert len(parts[4]) == 8  # random_part
        
        # 测试生成的ID是唯一的
        trace_id2 = trace_manager.generate_hb_trace_id("test.operation")
        assert trace_id != trace_id2
    
    def test_extract_otel_trace_id(self, trace_manager):
        """测试提取OpenTelemetry追踪ID"""
        # 模拟有效的span context
        with patch('opentelemetry.trace.get_current_span') as mock_span:
            mock_span_context = Mock()
            mock_span_context.trace_id = 0x12345678901234567890123456789012
            mock_span.return_value.get_span_context.return_value = mock_span_context
            
            trace_id = trace_manager.extract_otel_trace_id()
            assert trace_id == "12345678901234567890123456789012"
    
    def test_extract_span_id(self, trace_manager):
        """测试提取OpenTelemetry span ID"""
        # 模拟有效的span context
        with patch('opentelemetry.trace.get_current_span') as mock_span:
            mock_span_context = Mock()
            mock_span_context.span_id = 0x1234567890123456
            mock_span.return_value.get_span_context.return_value = mock_span_context
            
            span_id = trace_manager.extract_span_id()
            assert span_id == "1234567890123456"
    
    def test_create_dual_trace_context(self, trace_manager):
        """测试创建双追踪上下文"""
        context = trace_manager.create_dual_trace_context(
            operation_name="test.operation",
            correlation_id="corr123",
            session_id="sess123",
            user_id="user123"
        )
        
        assert isinstance(context, DualTraceContext)
        assert context.hb_trace_id.startswith("test_")
        assert context.correlation_id == "corr123"
        assert context.session_id == "sess123"
        assert context.user_id == "user123"
        assert context.service_name == "test-service"
        assert context.operation_name == "test.operation"
    
    def test_validate_hb_trace_id(self, trace_manager):
        """测试验证HarborAI追踪ID"""
        # 生成一个有效的ID
        valid_id = trace_manager.generate_hb_trace_id("test.operation")
        assert trace_manager.validate_hb_trace_id(valid_id) == True
        
        # 测试无效的ID
        assert trace_manager.validate_hb_trace_id("") == False
        assert trace_manager.validate_hb_trace_id("invalid_id") == False
        assert trace_manager.validate_hb_trace_id("wrong_prefix_123_service_op") == False
        assert trace_manager.validate_hb_trace_id(None) == False
    
    def test_get_cache_stats(self, trace_manager):
        """测试获取缓存统计信息"""
        stats = trace_manager.get_cache_stats()
        
        assert isinstance(stats, dict)
        assert "trace_mapping_size" in stats
        assert "otel_to_hb_mapping_size" in stats
        assert "max_cache_size" in stats
        assert "cache_ttl_seconds" in stats
        assert stats["trace_mapping_size"] == 0
        assert stats["otel_to_hb_mapping_size"] == 0
    
    def test_clear_cache(self, trace_manager):
        """测试清空缓存"""
        # 先创建一些上下文来填充缓存
        context = trace_manager.create_dual_trace_context("test.operation")
        
        # 验证缓存有内容
        stats_before = trace_manager.get_cache_stats()
        assert stats_before["trace_mapping_size"] > 0
        
        # 清空缓存
        trace_manager.clear_cache()
        
        # 验证缓存已清空
        stats_after = trace_manager.get_cache_stats()
        assert stats_after["trace_mapping_size"] == 0
        assert stats_after["otel_to_hb_mapping_size"] == 0


class TestDualTraceManagerIntegration:
    """双追踪ID管理器集成测试"""
    
    @pytest.fixture
    def trace_manager(self):
        """创建测试用的双追踪ID管理器"""
        return DualTraceIDManager(
            hb_prefix="integration",
            service_name="integration-test"
        )
    
    def test_id_generation_uniqueness(self, trace_manager):
        """测试ID生成的唯一性"""
        ids = set()
        for i in range(100):
            trace_id = trace_manager.generate_hb_trace_id(f"test.operation.{i}")
            assert trace_id not in ids
            ids.add(trace_id)
    
    def test_context_creation_and_caching(self, trace_manager):
        """测试上下文创建和缓存"""
        context1 = trace_manager.create_dual_trace_context(
            operation_name="test.operation1",
            correlation_id="corr1"
        )
        context2 = trace_manager.create_dual_trace_context(
            operation_name="test.operation2",
            correlation_id="corr2"
        )
        
        # 验证上下文不同
        assert context1.hb_trace_id != context2.hb_trace_id
        assert context1.correlation_id != context2.correlation_id
        
        # 验证缓存中有记录
        stats = trace_manager.get_cache_stats()
        assert stats["trace_mapping_size"] >= 2
    
    def test_cache_operations(self, trace_manager):
        """测试缓存操作"""
        # 创建多个上下文
        contexts = []
        for i in range(5):
            context = trace_manager.create_dual_trace_context(
                operation_name=f"test.operation.{i}",
                correlation_id=f"corr{i}"
            )
            contexts.append(context)
        
        # 验证缓存统计
        stats = trace_manager.get_cache_stats()
        assert stats["trace_mapping_size"] == 5
        
        # 通过HB ID获取上下文
        for context in contexts:
            cached_context = trace_manager.get_dual_context_by_hb_id(context.hb_trace_id)
            assert cached_context is not None
            assert cached_context.hb_trace_id == context.hb_trace_id
            assert cached_context.correlation_id == context.correlation_id
    
    def test_multiple_context_management(self, trace_manager):
        """测试多个上下文管理"""
        contexts = []
        
        # 创建多个不同的上下文
        for i in range(10):
            context = trace_manager.create_dual_trace_context(
                operation_name=f"operation.{i}",
                correlation_id=f"correlation_{i}",
                session_id=f"session_{i}",
                user_id=f"user_{i}"
            )
            contexts.append(context)
        
        # 验证每个上下文都是唯一的
        hb_ids = [ctx.hb_trace_id for ctx in contexts]
        assert len(set(hb_ids)) == 10  # 所有ID都是唯一的
        
        # 验证可以通过ID检索到正确的上下文
        for original_context in contexts:
            retrieved_context = trace_manager.get_dual_context_by_hb_id(
                original_context.hb_trace_id
            )
            assert retrieved_context is not None
            assert retrieved_context.hb_trace_id == original_context.hb_trace_id
            assert retrieved_context.correlation_id == original_context.correlation_id
            assert retrieved_context.session_id == original_context.session_id
            assert retrieved_context.user_id == original_context.user_id