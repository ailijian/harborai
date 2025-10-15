#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成本分析模块完整测试文件"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

from harborai.monitoring.cost_analysis import (
    CostTrend, BudgetAlert, CostForecast, ModelEfficiency, CostAnalysisReport,
    CostAnalyzer, get_cost_analyzer, generate_daily_report, 
    generate_weekly_report, generate_monthly_report
)
from harborai.monitoring.token_statistics import TokenUsageRecord, ModelStatistics, TimeWindowStats


class TestCostTrend:
    """成本趋势数据类测试"""
    
    def test_cost_trend_creation(self):
        """测试成本趋势创建"""
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        
        trend = CostTrend(
            period="2024-01-01 00:00",
            start_time=start_time,
            end_time=end_time,
            total_cost=10.5,
            total_tokens=1000,
            request_count=5,
            average_cost_per_request=2.1,
            average_cost_per_token=0.0105,
            top_models=[("gpt-4", 8.0), ("gpt-3.5", 2.5)]
        )
        
        assert trend.period == "2024-01-01 00:00"
        assert trend.total_cost == 10.5
        assert trend.total_tokens == 1000
        assert trend.request_count == 5
        assert len(trend.top_models) == 2
        assert trend.top_models[0][0] == "gpt-4"


class TestBudgetAlert:
    """预算预警数据类测试"""
    
    def test_budget_alert_creation(self):
        """测试预算预警创建"""
        timestamp = datetime.now()
        
        alert = BudgetAlert(
            alert_type="warning",
            message="Daily budget usage at 85%",
            current_cost=85.0,
            budget_limit=100.0,
            usage_percentage=0.85,
            period="daily",
            timestamp=timestamp,
            recommendations=["监控剩余预算", "优化API调用"]
        )
        
        assert alert.alert_type == "warning"
        assert alert.current_cost == 85.0
        assert alert.usage_percentage == 0.85
        assert len(alert.recommendations) == 2


class TestCostForecast:
    """成本预测数据类测试"""
    
    def test_cost_forecast_creation(self):
        """测试成本预测创建"""
        forecast = CostForecast(
            period="Day +1",
            predicted_cost=12.5,
            confidence_level=0.8,
            trend_direction="increasing",
            factors=["API调用量增加", "使用更昂贵模型"]
        )
        
        assert forecast.period == "Day +1"
        assert forecast.predicted_cost == 12.5
        assert forecast.confidence_level == 0.8
        assert forecast.trend_direction == "increasing"
        assert len(forecast.factors) == 2


class TestModelEfficiency:
    """模型效率分析数据类测试"""
    
    def test_model_efficiency_creation(self):
        """测试模型效率分析创建"""
        efficiency = ModelEfficiency(
            model_name="gpt-4",
            cost_per_token=0.03,
            average_response_time=2.5,
            success_rate=0.95,
            efficiency_score=0.8,
            usage_frequency=100,
            recommendations=["性能良好", "继续使用"]
        )
        
        assert efficiency.model_name == "gpt-4"
        assert efficiency.cost_per_token == 0.03
        assert efficiency.success_rate == 0.95
        assert efficiency.efficiency_score == 0.8
        assert len(efficiency.recommendations) == 2


class TestCostAnalysisReport:
    """成本分析报告数据类测试"""
    
    def test_cost_analysis_report_creation(self):
        """测试成本分析报告创建"""
        now = datetime.now()
        start_time = now - timedelta(days=7)
        
        report = CostAnalysisReport(
            report_id="test_report_001",
            generated_at=now,
            period_start=start_time,
            period_end=now,
            total_cost=100.0,
            total_tokens=10000,
            total_requests=50,
            cost_trends=[],
            growth_rate=0.1,
            budget_alerts=[],
            forecasts=[],
            model_efficiency=[],
            optimization_recommendations=["优化建议1", "优化建议2"]
        )
        
        assert report.report_id == "test_report_001"
        assert report.total_cost == 100.0
        assert report.growth_rate == 0.1
        assert len(report.optimization_recommendations) == 2


class TestCostAnalyzer:
    """成本分析器类测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.analyzer = CostAnalyzer()
    
    def test_cost_analyzer_initialization(self):
        """测试成本分析器初始化"""
        assert self.analyzer.budget_limits == {}
        assert self.analyzer.alert_thresholds['warning'] == 0.8
        assert self.analyzer.alert_thresholds['critical'] == 0.95
        assert self.analyzer.collector is not None
    
    def test_set_budget_limit(self):
        """测试设置预算限制"""
        self.analyzer.set_budget_limit("daily", 100.0)
        self.analyzer.set_budget_limit("monthly", 3000.0)
        
        assert self.analyzer.budget_limits["daily"] == 100.0
        assert self.analyzer.budget_limits["monthly"] == 3000.0
    
    def test_get_budget_limit(self):
        """测试获取预算限制"""
        self.analyzer.set_budget_limit("weekly", 500.0)
        
        assert self.analyzer.get_budget_limit("weekly") == 500.0
        assert self.analyzer.get_budget_limit("yearly") is None
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_analyze_cost_trends_daily(self, mock_get_collector):
        """测试分析成本趋势 - 日粒度"""
        # 模拟时间窗口统计数据
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        
        start_time = datetime.now() - timedelta(days=1)
        end_time = datetime.now()
        
        mock_window_stats = [
            TimeWindowStats(
                window_start=start_time,
                window_end=end_time,
                total_requests=5,
                total_tokens=1000,
                total_cost=10.0,
                models_used={"gpt-4": 3, "gpt-3.5": 2}
            )
        ]
        
        mock_collector.get_time_window_stats.return_value = mock_window_stats
        
        # 模拟模型统计数据
        mock_model_stats = {
            "gpt-4": Mock(total_cost=8.0),
            "gpt-3.5": Mock(total_cost=2.0)
        }
        mock_collector.get_model_statistics.return_value = mock_model_stats
        
        self.analyzer.collector = mock_collector
        
        trends = self.analyzer.analyze_cost_trends(days=1, granularity="daily")
        
        assert len(trends) == 1
        assert trends[0].total_cost == 10.0
        assert trends[0].total_tokens == 1000
        assert trends[0].request_count == 5
        assert trends[0].average_cost_per_request == 2.0
        assert trends[0].average_cost_per_token == 0.01
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_analyze_cost_trends_empty_data(self, mock_get_collector):
        """测试分析成本趋势 - 空数据"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        mock_collector.get_time_window_stats.return_value = []
        
        self.analyzer.collector = mock_collector
        
        trends = self.analyzer.analyze_cost_trends(days=7)
        
        assert trends == []
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_analyze_cost_trends_exception(self, mock_get_collector):
        """测试分析成本趋势 - 异常情况"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        mock_collector.get_time_window_stats.side_effect = Exception("数据库错误")
        
        self.analyzer.collector = mock_collector
        
        trends = self.analyzer.analyze_cost_trends(days=7)
        
        assert trends == []
    
    def test_check_budget_alerts_no_limits(self):
        """测试检查预算预警 - 无预算限制"""
        alerts = self.analyzer.check_budget_alerts()
        
        assert alerts == []
    
    @patch.object(CostAnalyzer, '_get_current_period_cost')
    def test_check_budget_alerts_warning(self, mock_get_cost):
        """测试检查预算预警 - 警告级别"""
        self.analyzer.set_budget_limit("daily", 100.0)
        mock_get_cost.return_value = 85.0  # 85% 使用率
        
        alerts = self.analyzer.check_budget_alerts()
        
        assert len(alerts) == 1
        assert alerts[0].alert_type == "warning"
        assert alerts[0].usage_percentage == 0.85
        assert alerts[0].period == "daily"
    
    @patch.object(CostAnalyzer, '_get_current_period_cost')
    def test_check_budget_alerts_critical(self, mock_get_cost):
        """测试检查预算预警 - 严重级别"""
        self.analyzer.set_budget_limit("monthly", 1000.0)
        mock_get_cost.return_value = 980.0  # 98% 使用率
        
        alerts = self.analyzer.check_budget_alerts()
        
        assert len(alerts) == 1
        assert alerts[0].alert_type == "critical"
        assert alerts[0].usage_percentage == 0.98
        assert "立即停止非必要的API调用" in alerts[0].recommendations
    
    @patch.object(CostAnalyzer, '_get_current_period_cost')
    def test_check_budget_alerts_exception(self, mock_get_cost):
        """测试检查预算预警 - 异常情况"""
        self.analyzer.set_budget_limit("daily", 100.0)
        mock_get_cost.side_effect = Exception("计算错误")
        
        alerts = self.analyzer.check_budget_alerts()
        
        assert alerts == []
    
    @patch.object(CostAnalyzer, 'analyze_cost_trends')
    def test_forecast_costs_insufficient_data(self, mock_analyze_trends):
        """测试成本预测 - 数据不足"""
        mock_analyze_trends.return_value = []  # 无历史数据
        
        forecasts = self.analyzer.forecast_costs(days_ahead=7)
        
        assert forecasts == []
    
    @patch.object(CostAnalyzer, 'analyze_cost_trends')
    def test_forecast_costs_success(self, mock_analyze_trends):
        """测试成本预测 - 成功预测"""
        # 模拟14天历史数据
        trends = []
        base_time = datetime.now() - timedelta(days=14)
        
        for i in range(14):
            trend = CostTrend(
                period=f"Day {i}",
                start_time=base_time + timedelta(days=i),
                end_time=base_time + timedelta(days=i+1),
                total_cost=10.0 + i * 0.5,  # 递增成本
                total_tokens=1000,
                request_count=5,
                average_cost_per_request=2.0,
                average_cost_per_token=0.01,
                top_models=[]
            )
            trends.append(trend)
        
        mock_analyze_trends.return_value = trends
        
        forecasts = self.analyzer.forecast_costs(days_ahead=3)
        
        assert len(forecasts) == 3
        assert all(f.predicted_cost > 0 for f in forecasts)
        assert all(f.confidence_level > 0 for f in forecasts)
        assert forecasts[0].period == "Day +1"
    
    @patch.object(CostAnalyzer, 'analyze_cost_trends')
    def test_forecast_costs_exception(self, mock_analyze_trends):
        """测试成本预测 - 异常情况"""
        mock_analyze_trends.side_effect = Exception("分析错误")
        
        forecasts = self.analyzer.forecast_costs()
        
        assert forecasts == []
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_analyze_model_efficiency_success(self, mock_get_collector):
        """测试模型效率分析 - 成功分析"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        
        # 模拟模型统计数据
        mock_stats = Mock()
        mock_stats.total_requests = 100
        mock_stats.total_tokens = 10000
        mock_stats.total_cost = 50.0
        mock_stats.success_rate = 0.95
        mock_stats.average_latency = 2.0
        mock_stats.error_rate = 0.05
        
        mock_model_stats = {"gpt-4": mock_stats}
        mock_collector.get_model_statistics.return_value = mock_model_stats
        
        self.analyzer.collector = mock_collector
        
        efficiency_list = self.analyzer.analyze_model_efficiency()
        
        assert len(efficiency_list) == 1
        assert efficiency_list[0].model_name == "gpt-4"
        assert efficiency_list[0].cost_per_token == 0.005
        assert efficiency_list[0].success_rate == 0.95
        assert efficiency_list[0].usage_frequency == 100
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_analyze_model_efficiency_no_requests(self, mock_get_collector):
        """测试模型效率分析 - 无请求数据"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        
        mock_stats = Mock()
        mock_stats.total_requests = 0
        
        mock_model_stats = {"gpt-4": mock_stats}
        mock_collector.get_model_statistics.return_value = mock_model_stats
        
        self.analyzer.collector = mock_collector
        
        efficiency_list = self.analyzer.analyze_model_efficiency()
        
        assert efficiency_list == []
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_analyze_model_efficiency_exception(self, mock_get_collector):
        """测试模型效率分析 - 异常情况"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        mock_collector.get_model_statistics.side_effect = Exception("统计错误")
        
        self.analyzer.collector = mock_collector
        
        efficiency_list = self.analyzer.analyze_model_efficiency()
        
        assert efficiency_list == []
    
    @patch.object(CostAnalyzer, 'analyze_cost_trends')
    @patch.object(CostAnalyzer, 'check_budget_alerts')
    @patch.object(CostAnalyzer, 'forecast_costs')
    @patch.object(CostAnalyzer, 'analyze_model_efficiency')
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_generate_comprehensive_report_success(self, mock_get_collector, 
                                                 mock_analyze_efficiency, mock_forecast,
                                                 mock_check_alerts, mock_analyze_trends):
        """测试生成综合报告 - 成功生成"""
        # 模拟各个组件的返回值
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        mock_collector.get_summary_stats.return_value = {
            'total_cost': 100.0,
            'total_tokens': 10000,
            'total_requests': 50
        }
        
        mock_analyze_trends.return_value = []
        mock_check_alerts.return_value = []
        mock_forecast.return_value = []
        mock_analyze_efficiency.return_value = []
        
        self.analyzer.collector = mock_collector
        
        report = self.analyzer.generate_comprehensive_report(days=7)
        
        assert report.total_cost == 100.0
        assert report.total_tokens == 10000
        assert report.total_requests == 50
        assert isinstance(report.report_id, str)
        assert isinstance(report.generated_at, datetime)
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_generate_comprehensive_report_exception(self, mock_get_collector):
        """测试生成综合报告 - 异常情况"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        mock_collector.get_summary_stats.side_effect = Exception("统计错误")
        
        self.analyzer.collector = mock_collector
        
        with pytest.raises(Exception):
            self.analyzer.generate_comprehensive_report()
    
    def test_export_report_json(self):
        """测试导出JSON格式报告"""
        now = datetime.now()
        report = CostAnalysisReport(
            report_id="test_001",
            generated_at=now,
            period_start=now - timedelta(days=1),
            period_end=now,
            total_cost=50.0,
            total_tokens=5000,
            total_requests=25,
            cost_trends=[],
            growth_rate=0.05,
            budget_alerts=[],
            forecasts=[],
            model_efficiency=[],
            optimization_recommendations=[]
        )
        
        json_output = self.analyzer.export_report(report, "json")
        
        assert isinstance(json_output, str)
        parsed = json.loads(json_output)
        assert parsed["report_id"] == "test_001"
        assert parsed["total_cost"] == 50.0
    
    def test_export_report_html(self):
        """测试导出HTML格式报告"""
        now = datetime.now()
        report = CostAnalysisReport(
            report_id="test_001",
            generated_at=now,
            period_start=now - timedelta(days=1),
            period_end=now,
            total_cost=50.0,
            total_tokens=5000,
            total_requests=25,
            cost_trends=[],
            growth_rate=0.05,
            budget_alerts=[],
            forecasts=[],
            model_efficiency=[],
            optimization_recommendations=["建议1", "建议2"]
        )
        
        html_output = self.analyzer.export_report(report, "html")
        
        assert isinstance(html_output, str)
        assert "<!DOCTYPE html>" in html_output
        assert "test_001" in html_output
        assert "¥50.0000" in html_output
    
    def test_export_report_unsupported_format(self):
        """测试导出不支持的格式"""
        now = datetime.now()
        report = CostAnalysisReport(
            report_id="test_001",
            generated_at=now,
            period_start=now - timedelta(days=1),
            period_end=now,
            total_cost=50.0,
            total_tokens=5000,
            total_requests=25,
            cost_trends=[],
            growth_rate=0.05,
            budget_alerts=[],
            forecasts=[],
            model_efficiency=[],
            optimization_recommendations=[]
        )
        
        with pytest.raises(ValueError, match="Unsupported format"):
            self.analyzer.export_report(report, "xml")
    
    def test_aggregate_to_weekly(self):
        """测试聚合为周统计"""
        daily_stats = list(range(14))  # 14天数据
        
        weekly_stats = self.analyzer._aggregate_to_weekly(daily_stats)
        
        assert len(weekly_stats) == 2  # 14天 / 7 = 2周
        assert weekly_stats == [0, 7]
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_get_model_costs_in_period(self, mock_get_collector):
        """测试获取时间段内模型成本"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        
        mock_stats1 = Mock(total_cost=30.0)
        mock_stats2 = Mock(total_cost=20.0)
        mock_model_stats = {
            "gpt-4": mock_stats1,
            "gpt-3.5": mock_stats2
        }
        mock_collector.get_model_statistics.return_value = mock_model_stats
        
        self.analyzer.collector = mock_collector
        
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        model_costs = self.analyzer._get_model_costs_in_period(start_time, end_time)
        
        assert model_costs["gpt-4"] == 30.0
        assert model_costs["gpt-3.5"] == 20.0
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_get_current_period_cost_daily(self, mock_get_collector):
        """测试获取当前周期成本 - 日周期"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        mock_collector.get_summary_stats.return_value = {'total_cost': 25.0}
        
        self.analyzer.collector = mock_collector
        
        cost = self.analyzer._get_current_period_cost("daily")
        
        assert cost == 25.0
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_get_current_period_cost_weekly(self, mock_get_collector):
        """测试获取当前周期成本 - 周周期"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        mock_collector.get_summary_stats.return_value = {'total_cost': 150.0}
        
        self.analyzer.collector = mock_collector
        
        cost = self.analyzer._get_current_period_cost("weekly")
        
        assert cost == 150.0
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_get_current_period_cost_monthly(self, mock_get_collector):
        """测试获取当前周期成本 - 月周期"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        mock_collector.get_summary_stats.return_value = {'total_cost': 500.0}
        
        self.analyzer.collector = mock_collector
        
        cost = self.analyzer._get_current_period_cost("monthly")
        
        assert cost == 500.0
    
    def test_get_current_period_cost_invalid(self):
        """测试获取当前周期成本 - 无效周期"""
        cost = self.analyzer._get_current_period_cost("yearly")
        
        assert cost == 0.0
    
    def test_generate_budget_recommendations_critical(self):
        """测试生成预算建议 - 严重级别"""
        recommendations = self.analyzer._generate_budget_recommendations(
            "daily", 95.0, 100.0, 0.95
        )
        
        assert "立即停止非必要的API调用" in recommendations
        assert "考虑使用更便宜的模型" in recommendations
        assert "当前daily预算使用率：95.0%" in recommendations
    
    def test_generate_budget_recommendations_warning(self):
        """测试生成预算建议 - 警告级别"""
        recommendations = self.analyzer._generate_budget_recommendations(
            "weekly", 85.0, 100.0, 0.85
        )
        
        assert "监控剩余预算使用情况" in recommendations
        assert "优化API调用频率" in recommendations
        assert "当前weekly预算使用率：85.0%" in recommendations
    
    def test_analyze_cost_factors_increasing(self):
        """测试分析成本影响因素 - 成本增加"""
        trends = [
            CostTrend("Day 1", datetime.now(), datetime.now(), 10.0, 1000, 5, 2.0, 0.01, []),
            CostTrend("Day 2", datetime.now(), datetime.now(), 15.0, 1000, 5, 3.0, 0.015, [])
        ]
        
        factors = self.analyzer._analyze_cost_factors(trends)
        
        assert "API调用量显著增加" in factors
        assert "使用了更昂贵的模型" in factors
    
    def test_analyze_cost_factors_stable(self):
        """测试分析成本影响因素 - 成本稳定"""
        trends = [
            CostTrend("Day 1", datetime.now(), datetime.now(), 10.0, 1000, 5, 2.0, 0.01, []),
            CostTrend("Day 2", datetime.now(), datetime.now(), 10.5, 1000, 5, 2.1, 0.0105, [])
        ]
        
        factors = self.analyzer._analyze_cost_factors(trends)
        
        assert factors == []
    
    def test_generate_model_recommendations_low_efficiency(self):
        """测试生成模型建议 - 低效率"""
        mock_stats = Mock()
        mock_stats.error_rate = 0.15
        mock_stats.average_latency = 6.0
        
        recommendations = self.analyzer._generate_model_recommendations(
            "test-model", mock_stats, 0.002, 0.3
        )
        
        assert "考虑替换为更高效的模型" in recommendations
        assert "检查API调用参数，降低错误率" in recommendations
        assert "成本较高，考虑使用更便宜的替代模型" in recommendations
        assert "响应时间较慢，考虑优化或更换模型" in recommendations
    
    def test_generate_model_recommendations_high_efficiency(self):
        """测试生成模型建议 - 高效率"""
        mock_stats = Mock()
        mock_stats.error_rate = 0.02
        mock_stats.average_latency = 1.5
        
        recommendations = self.analyzer._generate_model_recommendations(
            "test-model", mock_stats, 0.0005, 0.8
        )
        
        assert recommendations == []
    
    def test_calculate_growth_rate_insufficient_data(self):
        """测试计算增长率 - 数据不足"""
        trends = [
            CostTrend("Day 1", datetime.now(), datetime.now(), 10.0, 1000, 5, 2.0, 0.01, [])
        ]
        
        growth_rate = self.analyzer._calculate_growth_rate(trends)
        
        assert growth_rate == 0.0
    
    def test_calculate_growth_rate_positive(self):
        """测试计算增长率 - 正增长"""
        trends = []
        base_time = datetime.now()
        
        # 创建14天数据，前7天平均10，后7天平均15
        for i in range(14):
            cost = 10.0 if i < 7 else 15.0
            trend = CostTrend(f"Day {i}", base_time, base_time, cost, 1000, 5, 2.0, 0.01, [])
            trends.append(trend)
        
        growth_rate = self.analyzer._calculate_growth_rate(trends)
        
        assert growth_rate == 0.5  # (15-10)/10 = 0.5
    
    def test_generate_optimization_recommendations_comprehensive(self):
        """测试生成优化建议 - 综合情况"""
        # 成本快速增长的趋势
        trends = [
            CostTrend("Day 1", datetime.now(), datetime.now(), 10.0, 1000, 5, 2.0, 0.01, []),
            CostTrend("Day 2", datetime.now(), datetime.now(), 20.0, 1000, 5, 4.0, 0.02, [])
        ]
        
        # 低效率模型
        efficiency = [
            ModelEfficiency("low-model", 0.05, 3.0, 0.8, 0.3, 50, [])
        ]
        
        # 严重预算预警
        alerts = [
            BudgetAlert("critical", "Budget exceeded", 100.0, 95.0, 1.05, "daily", datetime.now(), [])
        ]
        
        recommendations = self.analyzer._generate_optimization_recommendations(
            trends, efficiency, alerts
        )
        
        assert "成本增长过快，建议审查API使用策略" in recommendations
        assert "考虑优化或替换低效率模型：low-model" in recommendations
        assert "预算即将耗尽，建议立即采取成本控制措施" in recommendations


class TestGlobalFunctions:
    """全局函数测试"""
    
    def test_get_cost_analyzer(self):
        """测试获取全局成本分析器"""
        analyzer1 = get_cost_analyzer()
        analyzer2 = get_cost_analyzer()
        
        assert analyzer1 is analyzer2  # 应该是同一个实例
        assert isinstance(analyzer1, CostAnalyzer)
    
    @patch.object(CostAnalyzer, 'generate_comprehensive_report')
    def test_generate_daily_report(self, mock_generate_report):
        """测试生成日报"""
        mock_report = Mock()
        mock_generate_report.return_value = mock_report
        
        report = generate_daily_report()
        
        mock_generate_report.assert_called_once_with(days=1)
        assert report is mock_report
    
    @patch.object(CostAnalyzer, 'generate_comprehensive_report')
    def test_generate_weekly_report(self, mock_generate_report):
        """测试生成周报"""
        mock_report = Mock()
        mock_generate_report.return_value = mock_report
        
        report = generate_weekly_report()
        
        mock_generate_report.assert_called_once_with(days=7)
        assert report is mock_report
    
    @patch.object(CostAnalyzer, 'generate_comprehensive_report')
    def test_generate_monthly_report(self, mock_generate_report):
        """测试生成月报"""
        mock_report = Mock()
        mock_generate_report.return_value = mock_report
        
        report = generate_monthly_report()
        
        mock_generate_report.assert_called_once_with(days=30)
        assert report is mock_report


class TestEdgeCases:
    """边界情况测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.analyzer = CostAnalyzer()
    
    def test_zero_cost_analysis(self):
        """测试零成本分析"""
        trends = [
            CostTrend("Day 1", datetime.now(), datetime.now(), 0.0, 0, 0, 0.0, 0.0, [])
        ]
        
        growth_rate = self.analyzer._calculate_growth_rate(trends)
        assert growth_rate == 0.0
        
        factors = self.analyzer._analyze_cost_factors(trends)
        assert factors == []
    
    def test_negative_cost_handling(self):
        """测试负成本处理"""
        # 预测成本可能为负数的情况
        forecasts = self.analyzer.forecast_costs(days_ahead=1)
        
        # 由于没有历史数据，应该返回空列表
        assert forecasts == []
    
    def test_unicode_in_recommendations(self):
        """测试建议中的Unicode字符"""
        recommendations = self.analyzer._generate_budget_recommendations(
            "测试周期", 90.0, 100.0, 0.9
        )
        
        assert any("测试周期" in rec for rec in recommendations)
    
    def test_large_numbers_handling(self):
        """测试大数值处理"""
        # 测试大成本值
        trends = [
            CostTrend("Day 1", datetime.now(), datetime.now(), 1000000.0, 1000000, 1000, 1000.0, 1.0, [])
        ]
        
        factors = self.analyzer._analyze_cost_factors(trends)
        assert isinstance(factors, list)
    
    @patch('harborai.monitoring.cost_analysis.get_token_statistics_collector')
    def test_empty_model_statistics(self, mock_get_collector):
        """测试空模型统计"""
        mock_collector = Mock()
        mock_get_collector.return_value = mock_collector
        mock_collector.get_model_statistics.return_value = {}
        
        self.analyzer.collector = mock_collector
        
        efficiency_list = self.analyzer.analyze_model_efficiency()
        
        assert efficiency_list == []
    
    def test_datetime_serialization(self):
        """测试datetime序列化"""
        now = datetime.now()
        report = CostAnalysisReport(
            report_id="test_datetime",
            generated_at=now,
            period_start=now - timedelta(days=1),
            period_end=now,
            total_cost=10.0,
            total_tokens=1000,
            total_requests=5,
            cost_trends=[],
            growth_rate=0.0,
            budget_alerts=[],
            forecasts=[],
            model_efficiency=[],
            optimization_recommendations=[]
        )
        
        json_output = self.analyzer.export_report(report, "json")
        parsed = json.loads(json_output)
        
        # 验证datetime被正确序列化为ISO格式
        assert "T" in parsed["generated_at"]
        assert isinstance(parsed["generated_at"], str)