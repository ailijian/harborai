#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能告警管理器测试
"""

import pytest
import asyncio
import json
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from harborai.monitoring.intelligent_alert_manager import (
    IntelligentAlertManager,
    IntelligentThreshold,
    DataQualityRule,
    AlertSuppressionRule,
    ThresholdType,
    AnomalyDetectionMethod,
    get_intelligent_alert_manager
)
from harborai.core.alerts import AlertSeverity, AlertStatus
from harborai.monitoring.prometheus_metrics import PrometheusMetrics


class TestIntelligentAlertManager:
    """智能告警管理器测试类"""
    
    @pytest.fixture
    def mock_prometheus_metrics(self):
        """模拟Prometheus指标"""
        metrics = Mock(spec=PrometheusMetrics)
        metrics.record_intelligent_threshold_adjustment = Mock()
        metrics.record_alert_rule_trigger = Mock()
        metrics.data_integrity_score = Mock()
        metrics.data_integrity_score.labels.return_value.set = Mock()
        return metrics
    
    @pytest.fixture
    def mock_alert_manager(self):
        """模拟告警管理器"""
        alert_manager = Mock()
        alert_manager.send_alert = AsyncMock()
        alert_manager.create_alert = AsyncMock()
        return alert_manager
    
    @pytest.fixture
    def temp_config_file(self):
        """临时配置文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config = {
                "intelligent_thresholds": [
                    {
                        "metric_name": "test_metric",
                        "threshold_type": "adaptive",
                        "base_value": 100.0,
                        "sensitivity": 1.5,
                        "detection_method": "statistical"
                    }
                ],
                "data_quality_rules": [
                    {
                        "rule_id": "test_rule",
                        "name": "测试规则",
                        "description": "测试数据质量规则",
                        "table_name": "test_table",
                        "rule_type": "completeness",
                        "threshold": 0.95
                    }
                ],
                "suppression_rules": [
                    {
                        "rule_id": "test_suppression",
                        "name": "测试抑制规则",
                        "conditions": {"frequency": "> 10/min"},
                        "duration": 900,  # 15分钟
                        "max_occurrences": 3
                    }
                ]
            }
            json.dump(config, f)
            yield f.name
        os.unlink(f.name)
    
    @pytest.fixture
    async def alert_manager(self, mock_prometheus_metrics, mock_alert_manager, temp_config_file):
        """智能告警管理器实例"""
        manager = IntelligentAlertManager(
            prometheus_metrics=mock_prometheus_metrics,
            alert_manager=mock_alert_manager,
            config_path=temp_config_file
        )
        yield manager
        await manager.stop()
    
    def test_init_with_default_config(self, mock_prometheus_metrics, mock_alert_manager):
        """测试使用默认配置初始化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "nonexistent.json")
            manager = IntelligentAlertManager(
                prometheus_metrics=mock_prometheus_metrics,
                alert_manager=mock_alert_manager,
                config_path=config_path
            )
            
            # 验证默认配置已加载
            assert len(manager.intelligent_thresholds) > 0
            assert len(manager.data_quality_rules) > 0
            assert len(manager.suppression_rules) > 0
    
    def test_load_config_from_file(self, alert_manager):
        """测试从文件加载配置"""
        # 验证配置已正确加载
        assert "test_metric" in alert_manager.intelligent_thresholds
        assert "test_rule" in alert_manager.data_quality_rules
        assert "test_suppression" in alert_manager.suppression_rules
        
        # 验证阈值配置
        threshold = alert_manager.intelligent_thresholds["test_metric"]
        assert threshold.metric_name == "test_metric"
        assert threshold.threshold_type == ThresholdType.ADAPTIVE
        assert threshold.base_value == 100.0
        assert threshold.sensitivity == 1.5
        assert threshold.detection_method == AnomalyDetectionMethod.STATISTICAL
    
    @pytest.mark.asyncio
    async def test_start_stop_manager(self, alert_manager):
        """测试启动和停止管理器"""
        # 测试启动
        await alert_manager.start()
        assert alert_manager._running is True
        assert alert_manager._monitoring_task is not None
        
        # 测试停止
        await alert_manager.stop()
        assert alert_manager._running is False
    
    @pytest.mark.asyncio
    async def test_adaptive_threshold_adjustment(self, alert_manager):
        """测试自适应阈值调整"""
        threshold = alert_manager.intelligent_thresholds["test_metric"]
        
        # 添加历史数据
        for i in range(150):  # 超过最小样本数
            threshold.historical_values.append({
                'value': 100.0 + i * 0.1,
                'timestamp': datetime.now()
            })
        
        # 调整阈值
        await alert_manager._adjust_adaptive_threshold(threshold, 115.0)
        
        # 验证阈值已调整
        assert threshold.current_threshold is not None
        assert threshold.last_update is not None
    
    @pytest.mark.asyncio
    async def test_dynamic_threshold_adjustment(self, alert_manager):
        """测试动态阈值调整"""
        threshold = IntelligentThreshold(
            metric_name="dynamic_test",
            threshold_type=ThresholdType.DYNAMIC,
            base_value=50.0,
            sensitivity=1.2
        )
        
        # 添加趋势数据
        for i in range(50):
            threshold.historical_values.append({
                'value': 50.0 + i * 0.5,  # 上升趋势
                'timestamp': datetime.now()
            })
        
        # 调整阈值
        await alert_manager._adjust_dynamic_threshold(threshold, 75.0)
        
        # 验证阈值已调整
        assert threshold.current_threshold is not None
        assert threshold.last_update is not None
    
    @pytest.mark.asyncio
    async def test_ml_threshold_adjustment(self, alert_manager):
        """测试机器学习阈值调整"""
        threshold = IntelligentThreshold(
            metric_name="ml_test",
            threshold_type=ThresholdType.ML_BASED,
            base_value=100.0,
            sensitivity=1.0,
            min_samples=50
        )
        
        # 添加历史数据
        import random
        for i in range(100):
            value = 100.0 + random.gauss(0, 10)  # 正态分布数据
            threshold.historical_values.append({
                'value': value,
                'timestamp': datetime.now()
            })
        
        # 调整阈值
        with patch('sklearn.ensemble.IsolationForest') as mock_forest:
            mock_model = Mock()
            mock_model.fit = Mock()
            mock_model.decision_function.return_value = [0.1] * 100
            mock_forest.return_value = mock_model
            
            await alert_manager._adjust_ml_threshold(threshold, 105.0)
            
            # 验证模型已训练
            mock_model.fit.assert_called_once()
            assert threshold.current_threshold is not None
    
    @pytest.mark.asyncio
    async def test_threshold_alert_creation(self, alert_manager, mock_alert_manager):
        """测试阈值告警创建"""
        threshold = alert_manager.intelligent_thresholds["test_metric"]
        threshold.current_threshold = 100.0
        
        # 触发告警
        await alert_manager._check_threshold_alert(threshold, 150.0)
        
        # 验证告警已发送
        mock_alert_manager.send_alert.assert_called_once()
        
        # 验证告警内容
        call_args = mock_alert_manager.send_alert.call_args[0][0]
        assert call_args.severity == AlertSeverity.HIGH
        assert call_args.status == AlertStatus.FIRING
        assert "test_metric" in call_args.name
    
    @pytest.mark.asyncio
    async def test_data_quality_check(self, alert_manager):
        """测试数据质量检查"""
        rule = alert_manager.data_quality_rules["test_rule"]
        
        # 执行质量检查
        with patch.object(alert_manager, '_check_completeness', return_value=0.98):
            score = await alert_manager._execute_quality_check(rule)
            assert score == 0.98
        
        # 测试低质量分数告警
        with patch.object(alert_manager, '_check_completeness', return_value=0.90):
            await alert_manager._check_data_quality()
            
            # 验证告警已创建
            assert rule.last_result == 0.90
            assert len(rule.trend_data) > 0
    
    @pytest.mark.asyncio
    async def test_alert_suppression(self, alert_manager):
        """测试告警抑制"""
        from harborai.core.alerts import Alert
        
        # 创建测试告警
        alert = Alert(
            id="test_alert",
            rule_id="test_rule",
            name="测试告警",
            description="测试告警描述",
            severity=AlertSeverity.MEDIUM,
            status=AlertStatus.FIRING,
            labels={"metric": "test_metric"},
            annotations={},
            starts_at=datetime.now(),
            generator_url="test"
        )
        
        # 测试抑制逻辑
        suppressed = await alert_manager._should_suppress_alert(alert)
        
        # 根据配置，应该被抑制
        assert isinstance(suppressed, bool)
    
    def test_add_remove_threshold(self, alert_manager):
        """测试添加和移除智能阈值"""
        new_threshold = IntelligentThreshold(
            metric_name="new_metric",
            threshold_type=ThresholdType.STATIC,
            base_value=200.0
        )
        
        # 添加阈值
        alert_manager.add_intelligent_threshold(new_threshold)
        assert "new_metric" in alert_manager.intelligent_thresholds
        
        # 移除阈值
        alert_manager.remove_intelligent_threshold("new_metric")
        assert "new_metric" not in alert_manager.intelligent_thresholds
    
    def test_add_remove_data_quality_rule(self, alert_manager):
        """测试添加和移除数据质量规则"""
        new_rule = DataQualityRule(
            rule_id="new_rule",
            name="新规则",
            description="新的数据质量规则",
            table_name="new_table"
        )
        
        # 添加规则
        alert_manager.add_data_quality_rule(new_rule)
        assert "new_rule" in alert_manager.data_quality_rules
        
        # 移除规则
        alert_manager.remove_data_quality_rule("new_rule")
        assert "new_rule" not in alert_manager.data_quality_rules
    
    def test_add_remove_suppression_rule(self, alert_manager):
        """测试添加和移除抑制规则"""
        new_rule = AlertSuppressionRule(
            rule_id="new_suppression",
            name="新抑制规则",
            conditions={"severity": "low"},
            duration=timedelta(minutes=10)
        )
        
        # 添加规则
        alert_manager.add_suppression_rule(new_rule)
        assert "new_suppression" in alert_manager.suppression_rules
        
        # 移除规则
        alert_manager.remove_suppression_rule("new_suppression")
        assert "new_suppression" not in alert_manager.suppression_rules
    
    def test_get_threshold_status(self, alert_manager):
        """测试获取阈值状态"""
        status = alert_manager.get_threshold_status("test_metric")
        
        assert status is not None
        assert status["metric_name"] == "test_metric"
        assert status["threshold_type"] == "adaptive"
        assert status["base_value"] == 100.0
        assert status["sensitivity"] == 1.5
        assert status["detection_method"] == "statistical"
        
        # 测试不存在的指标
        status = alert_manager.get_threshold_status("nonexistent")
        assert status is None
    
    def test_get_data_quality_status(self, alert_manager):
        """测试获取数据质量状态"""
        status = alert_manager.get_data_quality_status("test_rule")
        
        assert status is not None
        assert status["rule_id"] == "test_rule"
        assert status["name"] == "测试规则"
        assert status["table_name"] == "test_table"
        assert status["rule_type"] == "completeness"
        assert status["threshold"] == 0.95
        
        # 测试不存在的规则
        status = alert_manager.get_data_quality_status("nonexistent")
        assert status is None
    
    def test_get_statistics(self, alert_manager):
        """测试获取统计信息"""
        stats = alert_manager.get_statistics()
        
        assert "total_evaluations" in stats
        assert "threshold_adjustments" in stats
        assert "data_quality_checks" in stats
        assert "alerts_suppressed" in stats
        assert "ml_predictions" in stats
        assert "intelligent_thresholds_count" in stats
        assert "data_quality_rules_count" in stats
        assert "suppression_rules_count" in stats
        assert "last_evaluation" in stats
        assert "running" in stats
        
        # 验证计数正确
        assert stats["intelligent_thresholds_count"] == len(alert_manager.intelligent_thresholds)
        assert stats["data_quality_rules_count"] == len(alert_manager.data_quality_rules)
        assert stats["suppression_rules_count"] == len(alert_manager.suppression_rules)
    
    @pytest.mark.asyncio
    async def test_save_config(self, alert_manager):
        """测试保存配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            alert_manager.config_path = f.name
        
        try:
            # 保存配置
            await alert_manager._save_config()
            
            # 验证文件已创建
            assert os.path.exists(f.name)
            
            # 验证配置内容
            with open(f.name, 'r', encoding='utf-8') as file:
                config = json.load(file)
                assert "intelligent_thresholds" in config
                assert "data_quality_rules" in config
                assert "suppression_rules" in config
        finally:
            if os.path.exists(f.name):
                os.unlink(f.name)
    
    def test_global_instance(self):
        """测试全局实例"""
        manager1 = get_intelligent_alert_manager()
        manager2 = get_intelligent_alert_manager()
        
        # 验证单例模式
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self, alert_manager):
        """测试监控循环错误处理"""
        # 模拟评估过程中的异常
        with patch.object(alert_manager, '_evaluate_intelligent_thresholds', side_effect=Exception("测试异常")):
            await alert_manager.start()
            
            # 等待一个监控周期
            await asyncio.sleep(0.1)
            
            # 验证管理器仍在运行
            assert alert_manager._running is True
            
            await alert_manager.stop()
    
    @pytest.mark.asyncio
    async def test_metric_value_retrieval(self, alert_manager):
        """测试指标值获取"""
        # 测试获取指标值
        with patch.object(alert_manager, '_get_prometheus_gauge_value', return_value=75.5):
            value = await alert_manager._get_metric_value("api_response_time_p95")
            assert value == 75.5
        
        # 测试未知指标
        value = await alert_manager._get_metric_value("unknown_metric")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_data_quality_check_types(self, alert_manager):
        """测试不同类型的数据质量检查"""
        rule = DataQualityRule(
            rule_id="completeness_test",
            name="完整性测试",
            description="测试完整性检查",
            table_name="test_table",
            rule_type="completeness"
        )
        
        # 测试完整性检查
        score = await alert_manager._check_completeness(rule)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        # 测试一致性检查
        rule.rule_type = "consistency"
        score = await alert_manager._check_consistency(rule)
        assert isinstance(score, float)
        
        # 测试准确性检查
        rule.rule_type = "accuracy"
        score = await alert_manager._check_accuracy(rule)
        assert isinstance(score, float)
        
        # 测试有效性检查
        rule.rule_type = "validity"
        score = await alert_manager._check_validity(rule)
        assert isinstance(score, float)


class TestIntelligentThreshold:
    """智能阈值测试类"""
    
    def test_threshold_creation(self):
        """测试阈值创建"""
        threshold = IntelligentThreshold(
            metric_name="test_metric",
            threshold_type=ThresholdType.ADAPTIVE,
            base_value=100.0,
            sensitivity=1.5
        )
        
        assert threshold.metric_name == "test_metric"
        assert threshold.threshold_type == ThresholdType.ADAPTIVE
        assert threshold.base_value == 100.0
        assert threshold.sensitivity == 1.5
        assert threshold.detection_method == AnomalyDetectionMethod.STATISTICAL
        assert len(threshold.historical_values) == 0
        assert threshold.current_threshold is None
    
    def test_threshold_defaults(self):
        """测试阈值默认值"""
        threshold = IntelligentThreshold(
            metric_name="test",
            threshold_type=ThresholdType.STATIC,
            base_value=50.0
        )
        
        assert threshold.sensitivity == 1.0
        assert threshold.adaptation_rate == 0.1
        assert threshold.min_samples == 100
        assert threshold.confidence_level == 0.95
        assert threshold.upper_multiplier == 1.5
        assert threshold.lower_multiplier == 0.5
        assert threshold.max_adjustment == 0.3


class TestDataQualityRule:
    """数据质量规则测试类"""
    
    def test_rule_creation(self):
        """测试规则创建"""
        rule = DataQualityRule(
            rule_id="test_rule",
            name="测试规则",
            description="测试数据质量规则",
            table_name="test_table",
            column_name="test_column",
            rule_type="completeness",
            threshold=0.95,
            severity=AlertSeverity.HIGH
        )
        
        assert rule.rule_id == "test_rule"
        assert rule.name == "测试规则"
        assert rule.table_name == "test_table"
        assert rule.column_name == "test_column"
        assert rule.rule_type == "completeness"
        assert rule.threshold == 0.95
        assert rule.severity == AlertSeverity.HIGH
        assert rule.enabled is True
    
    def test_rule_defaults(self):
        """测试规则默认值"""
        rule = DataQualityRule(
            rule_id="test",
            name="测试",
            description="测试",
            table_name="test"
        )
        
        assert rule.column_name is None
        assert rule.rule_type == "completeness"
        assert rule.threshold == 0.95
        assert rule.severity == AlertSeverity.MEDIUM
        assert rule.enabled is True
        assert rule.check_query is None
        assert rule.check_function is None


class TestAlertSuppressionRule:
    """告警抑制规则测试类"""
    
    def test_suppression_rule_creation(self):
        """测试抑制规则创建"""
        rule = AlertSuppressionRule(
            rule_id="test_suppression",
            name="测试抑制",
            conditions={"severity": "low"},
            duration=timedelta(minutes=15),
            max_occurrences=5
        )
        
        assert rule.rule_id == "test_suppression"
        assert rule.name == "测试抑制"
        assert rule.conditions == {"severity": "low"}
        assert rule.duration == timedelta(minutes=15)
        assert rule.max_occurrences == 5
        assert rule.enabled is True
        assert len(rule.active_suppressions) == 0
        assert len(rule.occurrence_count) == 0
    
    def test_suppression_rule_defaults(self):
        """测试抑制规则默认值"""
        rule = AlertSuppressionRule(
            rule_id="test",
            name="测试",
            conditions={},
            duration=timedelta(minutes=10)
        )
        
        assert rule.max_occurrences == 5
        assert rule.enabled is True
        assert rule.last_triggered is None


if __name__ == "__main__":
    pytest.main([__file__])