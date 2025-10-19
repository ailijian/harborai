"""TokenUsage and TokenUsageStats' unit tests.

Test TokenUsage data model's functionality and data consistency.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch

from harborai.core.token_usage import TokenUsage, TokenUsageStats


class TestTokenUsage:
    """TokenUsage data model test."""
    
    def test_token_usage_creation_valid_data(self):
        """Test using valid data to create TokenUsage."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.parsing_method == "direct_extraction"
        assert usage.confidence == 1.0
        assert usage.raw_data == {}
        assert usage.timestamp is not None
    
    def test_token_usage_creation_with_custom_data(self):
        """Test using custom data to create TokenUsage."""
        timestamp = datetime.now(timezone.utc)
        raw_data = {"original_response": {"usage": {"prompt_tokens": 100}}}
        
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            parsing_method="api_response",
            confidence=0.9,
            raw_data=raw_data,
            timestamp=timestamp
        )
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.parsing_method == "api_response"
        assert usage.confidence == 0.9
        assert usage.raw_data == raw_data
        assert usage.timestamp == timestamp
    
    def test_token_usage_auto_calculate_total_consistent(self):
        """Test consistent token data will not be modified."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150  # 正确的总和
        )
        
        assert usage.total_tokens == 150
        assert usage.confidence == 1.0
    
    def test_token_usage_auto_calculate_total_inconsistent(self):
        """Test inconsistent token data will be automatically corrected."""
        with patch('harborai.core.token_usage.logger') as mock_logger:
            usage = TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=200  # 错误的总和
            )
            
            # 应该被修正为正确的总和
            assert usage.total_tokens == 150
            assert usage.confidence == 0.8  # 置信度降低
            
            # 验证日志记录
            mock_logger.warning.assert_called_once()
            mock_logger.info.assert_called_once()
    
    def test_token_usage_data_consistency_validation_valid(self):
        """Test valid data consistency validation."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        assert usage.validate_consistency() is True
        assert usage.is_valid() is True
    
    def test_token_usage_data_consistency_validation_invalid_total(self):
        """Test invalid total consistency validation."""
        # 创建时会自动修正，所以我们需要手动设置错误的值
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # 手动设置错误的total_tokens来测试验证
        usage.total_tokens = 200
        
        assert usage.validate_consistency() is False
    
    def test_token_usage_data_consistency_validation_negative_values(self):
        """Test negative values data validation."""
        usage = TokenUsage(
            prompt_tokens=-10,
            completion_tokens=50,
            total_tokens=40
        )
        
        assert usage.is_valid() is False
    
    def test_token_usage_data_consistency_validation_zero_values(self):
        """Test zero values data validation."""
        usage = TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0
        )
        
        assert usage.is_valid() is True
        assert usage.validate_consistency() is True
    
    def test_token_usage_auto_data_correction_only_total(self):
        """Test auto data correction when only total_tokens has value."""
        with patch('harborai.core.token_usage.logger') as mock_logger:
            usage = TokenUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=100  # 只有总数有值
            )
            
            # 应该保持原有的total_tokens值
            assert usage.total_tokens == 100
            
            # 验证日志记录
            mock_logger.warning.assert_called_once()
            mock_logger.info.assert_called_once()
    
    def test_token_usage_confidence_score_calculation(self):
        """测试置信度评分计算。"""
        # 一致的数据，高置信度
        usage1 = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            confidence=1.0
        )
        assert usage1.get_quality_score() == 1.0
        
        # 不一致的数据（手动设置）
        usage2 = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            confidence=0.8
        )
        usage2.total_tokens = 200  # 手动设置不一致
        assert usage2.get_quality_score() == 0.8 * 0.7  # confidence * consistency_score
    
    def test_token_usage_to_dict(self):
        """测试转换为字典。"""
        timestamp = datetime.now(timezone.utc)
        raw_data = {"test": "data"}
        
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            parsing_method="api_response",
            confidence=0.9,
            raw_data=raw_data,
            timestamp=timestamp
        )
        
        result = usage.to_dict()
        
        assert result["prompt_tokens"] == 100
        assert result["completion_tokens"] == 50
        assert result["total_tokens"] == 150
        assert result["parsing_method"] == "api_response"
        assert result["confidence"] == 0.9
        assert result["raw_data"] == raw_data
        assert result["timestamp"] == timestamp.isoformat()
    
    def test_token_usage_from_dict(self):
        """测试从字典创建TokenUsage。"""
        timestamp = datetime.now(timezone.utc)
        data = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "parsing_method": "api_response",
            "confidence": 0.9,
            "raw_data": {"test": "data"},
            "timestamp": timestamp.isoformat()
        }
        
        usage = TokenUsage.from_dict(data)
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150
        assert usage.parsing_method == "api_response"
        assert usage.confidence == 0.9
        assert usage.raw_data == {"test": "data"}
        assert usage.timestamp == timestamp
    
    def test_token_usage_from_dict_missing_fields(self):
        """测试从缺少字段的字典创建TokenUsage。"""
        data = {
            "prompt_tokens": 100,
            "completion_tokens": 50
        }
        
        usage = TokenUsage.from_dict(data)
        
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150  # 自动计算值 (100 + 50)
        assert usage.parsing_method == "direct_extraction"  # 默认值
        assert usage.confidence == 0.8  # 自动修正后降低的置信度
        assert usage.raw_data == {}  # 默认值
    
    def test_token_usage_data_quality_assessment(self):
        """测试数据质量评估。"""
        # 高质量数据
        high_quality = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            confidence=1.0
        )
        assert high_quality.get_quality_score() == 1.0
        
        # 中等质量数据（置信度降低）
        medium_quality = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            confidence=0.8
        )
        assert medium_quality.get_quality_score() == 0.8
        
        # 低质量数据（不一致）
        low_quality = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            confidence=1.0
        )
        low_quality.total_tokens = 200  # 手动设置不一致
        assert low_quality.get_quality_score() == 0.7  # 1.0 * 0.7


class TestTokenUsageIntegration:
    """TokenUsage集成测试。"""
    
    def test_complete_workflow(self):
        """测试完整的工作流程。"""
        # 1. 创建TokenUsage
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            parsing_method="api_response",
            raw_data={"original": "data"}
        )
        
        # 2. 验证数据
        assert usage.is_valid()
        assert usage.validate_consistency()
        
        # 3. 转换为字典
        data = usage.to_dict()
        
        # 4. 从字典重建
        rebuilt = TokenUsage.from_dict(data)
        
        # 5. 验证重建的对象
        assert rebuilt.prompt_tokens == usage.prompt_tokens
        assert rebuilt.completion_tokens == usage.completion_tokens
        assert rebuilt.total_tokens == usage.total_tokens
        assert rebuilt.parsing_method == usage.parsing_method
        assert rebuilt.confidence == usage.confidence
        assert rebuilt.raw_data == usage.raw_data
    
    def test_error_recovery(self):
        """测试错误恢复。"""
        # 创建有问题的数据
        with patch('harborai.core.token_usage.logger'):
            usage = TokenUsage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=200  # 错误的总数
            )
            
            # 应该自动修正
            assert usage.total_tokens == 150
            assert usage.confidence < 1.0
            
            # 仍然是有效的
            assert usage.is_valid()
    
    def test_batch_processing(self):
        """测试批量处理。"""
        usages = []
        
        # 创建多个TokenUsage实例
        for i in range(10):
            usage = TokenUsage(
                prompt_tokens=100 + i,
                completion_tokens=50 + i,
                total_tokens=150 + 2 * i
            )
            usages.append(usage)
        
        # 验证所有实例
        for usage in usages:
            assert usage.is_valid()
            assert usage.validate_consistency()
        
        # 转换为字典列表
        data_list = [usage.to_dict() for usage in usages]
        
        # 从字典重建
        rebuilt_usages = [TokenUsage.from_dict(data) for data in data_list]
        
        # 验证重建的实例
        assert len(rebuilt_usages) == len(usages)
        for original, rebuilt in zip(usages, rebuilt_usages):
            assert original.prompt_tokens == rebuilt.prompt_tokens
            assert original.completion_tokens == rebuilt.completion_tokens
            assert original.total_tokens == rebuilt.total_tokens


class TestTokenUsageStats:
    """TokenUsageStats测试。"""
    
    def test_token_usage_stats_creation(self):
        """测试TokenUsageStats创建。"""
        stats = TokenUsageStats()
        
        assert stats.total_requests == 0
        assert stats.total_prompt_tokens == 0
        assert stats.total_completion_tokens == 0
        assert stats.total_tokens == 0
        assert stats.avg_prompt_tokens == 0.0
        assert stats.avg_completion_tokens == 0.0
        assert stats.avg_total_tokens == 0.0
        assert stats.date_range_start is None
        assert stats.date_range_end is None
    
    def test_token_usage_stats_creation_with_data(self):
        """测试使用数据创建TokenUsageStats。"""
        start_date = datetime.now(timezone.utc)
        end_date = datetime.now(timezone.utc)
        
        stats = TokenUsageStats(
            total_requests=10,
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_tokens=1500,
            avg_prompt_tokens=100.0,
            avg_completion_tokens=50.0,
            avg_total_tokens=150.0,
            date_range_start=start_date,
            date_range_end=end_date
        )
        
        assert stats.total_requests == 10
        assert stats.total_prompt_tokens == 1000
        assert stats.total_completion_tokens == 500
        assert stats.total_tokens == 1500
        assert stats.avg_prompt_tokens == 100.0
        assert stats.avg_completion_tokens == 50.0
        assert stats.avg_total_tokens == 150.0
        assert stats.date_range_start == start_date
        assert stats.date_range_end == end_date
    
    def test_token_usage_stats_to_dict(self):
        """测试TokenUsageStats转换为字典。"""
        start_date = datetime.now(timezone.utc)
        end_date = datetime.now(timezone.utc)
        
        stats = TokenUsageStats(
            total_requests=10,
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_tokens=1500,
            avg_prompt_tokens=100.0,
            avg_completion_tokens=50.0,
            avg_total_tokens=150.0,
            date_range_start=start_date,
            date_range_end=end_date
        )
        
        result = stats.to_dict()
        
        assert result["total_requests"] == 10
        assert result["total_prompt_tokens"] == 1000
        assert result["total_completion_tokens"] == 500
        assert result["total_tokens"] == 1500
        assert result["avg_prompt_tokens"] == 100.0
        assert result["avg_completion_tokens"] == 50.0
        assert result["avg_total_tokens"] == 150.0
        assert result["date_range_start"] == start_date.isoformat()
        assert result["date_range_end"] == end_date.isoformat()
    
    def test_token_usage_stats_edge_cases(self):
        """测试TokenUsageStats边界情况。"""
        # 空统计
        empty_stats = TokenUsageStats()
        result = empty_stats.to_dict()
        
        assert result["date_range_start"] is None
        assert result["date_range_end"] is None
        
        # 大数值
        large_stats = TokenUsageStats(
            total_requests=1000000,
            total_prompt_tokens=100000000,
            total_completion_tokens=50000000,
            total_tokens=150000000
        )
        
        assert large_stats.total_requests == 1000000
        assert large_stats.total_prompt_tokens == 100000000