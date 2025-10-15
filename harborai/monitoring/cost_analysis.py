#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成本分析报告模块"""

import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics

from .token_statistics import get_token_statistics_collector, TokenUsageRecord
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CostTrend:
    """成本趋势数据类"""
    period: str  # 时间段标识
    start_time: datetime
    end_time: datetime
    total_cost: float
    total_tokens: int
    request_count: int
    average_cost_per_request: float
    average_cost_per_token: float
    top_models: List[Tuple[str, float]]  # (模型名, 成本)


@dataclass
class BudgetAlert:
    """预算预警数据类"""
    alert_type: str  # "warning" 或 "critical"
    message: str
    current_cost: float
    budget_limit: float
    usage_percentage: float
    period: str
    timestamp: datetime
    recommendations: List[str]


@dataclass
class CostForecast:
    """成本预测数据类"""
    period: str
    predicted_cost: float
    confidence_level: float
    trend_direction: str  # "increasing", "decreasing", "stable"
    factors: List[str]  # 影响因素


@dataclass
class ModelEfficiency:
    """模型效率分析数据类"""
    model_name: str
    cost_per_token: float
    average_response_time: float
    success_rate: float
    efficiency_score: float  # 综合效率评分
    usage_frequency: int
    recommendations: List[str]


@dataclass
class CostAnalysisReport:
    """成本分析报告数据类"""
    report_id: str
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    
    # 总体统计
    total_cost: float
    total_tokens: int
    total_requests: int
    
    # 趋势分析
    cost_trends: List[CostTrend]
    growth_rate: float  # 成本增长率
    
    # 预算预警
    budget_alerts: List[BudgetAlert]
    
    # 成本预测
    forecasts: List[CostForecast]
    
    # 模型效率分析
    model_efficiency: List[ModelEfficiency]
    
    # 优化建议
    optimization_recommendations: List[str]


class CostAnalyzer:
    """成本分析器类"""
    
    def __init__(self):
        """
        初始化成本分析器
        """
        self.collector = get_token_statistics_collector()
        self.budget_limits = {}  # 预算限制配置
        self.alert_thresholds = {
            'warning': 0.8,  # 80%预警
            'critical': 0.95  # 95%严重预警
        }
    
    def set_budget_limit(self, period: str, limit: float) -> None:
        """
        设置预算限制
        
        Args:
            period: 时间周期（"daily", "weekly", "monthly"）
            limit: 预算限制金额
        """
        self.budget_limits[period] = limit
        logger.info(f"Set budget limit for {period}: ${limit:.2f}")
    
    def get_budget_limit(self, period: str) -> Optional[float]:
        """
        获取预算限制
        
        Args:
            period: 时间周期
            
        Returns:
            预算限制金额，如果未设置则返回None
        """
        return self.budget_limits.get(period)
    
    def analyze_cost_trends(self, days: int = 30, granularity: str = "daily") -> List[CostTrend]:
        """
        分析成本趋势
        
        Args:
            days: 分析天数
            granularity: 粒度（"hourly", "daily", "weekly"）
            
        Returns:
            成本趋势列表
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # 获取时间窗口统计
            if granularity == "hourly":
                window_stats = self.collector.get_time_window_stats("hour", days * 24)
            elif granularity == "daily":
                window_stats = self.collector.get_time_window_stats("day", days)
            else:  # weekly
                window_stats = self.collector.get_time_window_stats("day", days)
                # 将日统计聚合为周统计
                window_stats = self._aggregate_to_weekly(window_stats)
            
            trends = []
            for stats in window_stats:
                # 获取该时间段内的模型使用情况
                model_costs = self._get_model_costs_in_period(
                    stats.window_start, stats.window_end
                )
                
                # 计算平均成本
                avg_cost_per_request = (
                    stats.total_cost / stats.total_requests 
                    if stats.total_requests > 0 else 0
                )
                avg_cost_per_token = (
                    stats.total_cost / stats.total_tokens 
                    if stats.total_tokens > 0 else 0
                )
                
                # 获取前3个最昂贵的模型
                top_models = sorted(
                    model_costs.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:3]
                
                trend = CostTrend(
                    period=stats.window_start.strftime("%Y-%m-%d %H:%M"),
                    start_time=stats.window_start,
                    end_time=stats.window_end,
                    total_cost=stats.total_cost,
                    total_tokens=stats.total_tokens,
                    request_count=stats.total_requests,
                    average_cost_per_request=avg_cost_per_request,
                    average_cost_per_token=avg_cost_per_token,
                    top_models=top_models
                )
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing cost trends: {e}")
            return []
    
    def check_budget_alerts(self) -> List[BudgetAlert]:
        """
        检查预算预警
        
        Returns:
            预算预警列表
        """
        alerts = []
        
        try:
            for period, limit in self.budget_limits.items():
                current_cost = self._get_current_period_cost(period)
                usage_percentage = current_cost / limit if limit > 0 else 0
                
                alert_type = None
                if usage_percentage >= self.alert_thresholds['critical']:
                    alert_type = "critical"
                elif usage_percentage >= self.alert_thresholds['warning']:
                    alert_type = "warning"
                
                if alert_type:
                    recommendations = self._generate_budget_recommendations(
                        period, current_cost, limit, usage_percentage
                    )
                    
                    alert = BudgetAlert(
                        alert_type=alert_type,
                        message=f"{period.capitalize()} budget usage at {usage_percentage:.1%}",
                        current_cost=current_cost,
                        budget_limit=limit,
                        usage_percentage=usage_percentage,
                        period=period,
                        timestamp=datetime.now(),
                        recommendations=recommendations
                    )
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking budget alerts: {e}")
            return []
    
    def forecast_costs(self, days_ahead: int = 7) -> List[CostForecast]:
        """
        预测未来成本
        
        Args:
            days_ahead: 预测天数
            
        Returns:
            成本预测列表
        """
        try:
            # 获取历史数据进行趋势分析
            historical_trends = self.analyze_cost_trends(days=30, granularity="daily")
            
            if len(historical_trends) < 7:
                logger.warning("Insufficient historical data for forecasting")
                return []
            
            # 计算趋势
            recent_costs = [trend.total_cost for trend in historical_trends[-7:]]
            older_costs = [trend.total_cost for trend in historical_trends[-14:-7]]
            
            recent_avg = statistics.mean(recent_costs) if recent_costs else 0
            older_avg = statistics.mean(older_costs) if older_costs else 0
            
            # 计算增长率
            growth_rate = (
                (recent_avg - older_avg) / older_avg 
                if older_avg > 0 else 0
            )
            
            # 确定趋势方向
            if abs(growth_rate) < 0.05:  # 5%以内认为稳定
                trend_direction = "stable"
            elif growth_rate > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            # 生成预测
            forecasts = []
            base_cost = recent_avg
            
            for i in range(1, days_ahead + 1):
                # 简单线性预测（实际应用中可以使用更复杂的模型）
                predicted_cost = base_cost * (1 + growth_rate * i / 7)
                
                # 计算置信度（基于历史数据的方差）
                if len(recent_costs) > 1:
                    variance = statistics.variance(recent_costs)
                    confidence = max(0.5, 1 - (variance / (recent_avg ** 2)))
                else:
                    confidence = 0.5
                
                # 影响因素分析
                factors = self._analyze_cost_factors(historical_trends)
                
                forecast = CostForecast(
                    period=f"Day +{i}",
                    predicted_cost=max(0, predicted_cost),
                    confidence_level=confidence,
                    trend_direction=trend_direction,
                    factors=factors
                )
                forecasts.append(forecast)
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error forecasting costs: {e}")
            return []
    
    def analyze_model_efficiency(self) -> List[ModelEfficiency]:
        """
        分析模型效率
        
        Returns:
            模型效率分析列表
        """
        try:
            model_stats = self.collector.get_model_statistics()
            efficiency_list = []
            
            for model_name, stats in model_stats.items():
                if stats.total_requests == 0:
                    continue
                
                # 计算每token成本
                cost_per_token = (
                    stats.total_cost / stats.total_tokens 
                    if stats.total_tokens > 0 else 0
                )
                
                # 计算效率评分（综合考虑成本、速度、成功率）
                # 成功率权重40%，速度权重30%，成本权重30%
                success_score = stats.success_rate * 0.4
                
                # 速度评分（假设平均延迟越低越好，以5秒为基准）
                speed_score = max(0, (5 - stats.average_latency) / 5) * 0.3
                
                # 成本评分（假设每token成本越低越好，以0.001为基准）
                cost_score = max(0, (0.001 - cost_per_token) / 0.001) * 0.3
                
                efficiency_score = success_score + speed_score + cost_score
                
                # 生成建议
                recommendations = self._generate_model_recommendations(
                    model_name, stats, cost_per_token, efficiency_score
                )
                
                efficiency = ModelEfficiency(
                    model_name=model_name,
                    cost_per_token=cost_per_token,
                    average_response_time=stats.average_latency,
                    success_rate=stats.success_rate,
                    efficiency_score=efficiency_score,
                    usage_frequency=stats.total_requests,
                    recommendations=recommendations
                )
                efficiency_list.append(efficiency)
            
            # 按效率评分排序
            efficiency_list.sort(key=lambda x: x.efficiency_score, reverse=True)
            
            return efficiency_list
            
        except Exception as e:
            logger.error(f"Error analyzing model efficiency: {e}")
            return []
    
    def generate_comprehensive_report(self, days: int = 30) -> CostAnalysisReport:
        """
        生成综合成本分析报告
        
        Args:
            days: 分析天数
            
        Returns:
            综合成本分析报告
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # 获取总体统计
            summary_stats = self.collector.get_summary_stats()
            
            # 分析成本趋势
            cost_trends = self.analyze_cost_trends(days=days)
            
            # 计算增长率
            growth_rate = self._calculate_growth_rate(cost_trends)
            
            # 检查预算预警
            budget_alerts = self.check_budget_alerts()
            
            # 成本预测
            forecasts = self.forecast_costs()
            
            # 模型效率分析
            model_efficiency = self.analyze_model_efficiency()
            
            # 生成优化建议
            optimization_recommendations = self._generate_optimization_recommendations(
                cost_trends, model_efficiency, budget_alerts
            )
            
            report = CostAnalysisReport(
                report_id=f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                generated_at=datetime.now(),
                period_start=start_time,
                period_end=end_time,
                total_cost=summary_stats.get('total_cost', 0),
                total_tokens=summary_stats.get('total_tokens', 0),
                total_requests=summary_stats.get('total_requests', 0),
                cost_trends=cost_trends,
                growth_rate=growth_rate,
                budget_alerts=budget_alerts,
                forecasts=forecasts,
                model_efficiency=model_efficiency,
                optimization_recommendations=optimization_recommendations
            )
            
            logger.info(f"Generated comprehensive cost analysis report: {report.report_id}")
            return report
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            raise
    
    def export_report(self, report: CostAnalysisReport, format_type: str = "json") -> str:
        """
        导出报告
        
        Args:
            report: 成本分析报告
            format_type: 导出格式（"json" 或 "html"）
            
        Returns:
            导出的报告内容
        """
        try:
            if format_type == "json":
                return self._export_json_report(report)
            elif format_type == "html":
                return self._export_html_report(report)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            raise
    
    def _aggregate_to_weekly(self, daily_stats):
        """将日统计聚合为周统计"""
        # 简化实现，实际应用中需要更复杂的聚合逻辑
        return daily_stats[::7]  # 每7天取一个
    
    def _get_model_costs_in_period(self, start_time: datetime, end_time: datetime) -> Dict[str, float]:
        """获取指定时间段内各模型的成本"""
        # 简化实现，实际应用中需要查询数据库
        model_stats = self.collector.get_model_statistics()
        return {model: stats.total_cost for model, stats in model_stats.items()}
    
    def _get_current_period_cost(self, period: str) -> float:
        """获取当前周期的成本"""
        now = datetime.now()
        
        if period == "daily":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "weekly":
            days_since_monday = now.weekday()
            start_time = now - timedelta(days=days_since_monday)
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "monthly":
            start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return 0.0
        
        # 简化实现，实际应用中需要查询指定时间段的数据
        summary_stats = self.collector.get_summary_stats()
        return summary_stats.get('total_cost', 0)
    
    def _generate_budget_recommendations(self, period: str, current_cost: float, 
                                       limit: float, usage_percentage: float) -> List[str]:
        """生成预算建议"""
        recommendations = []
        
        if usage_percentage >= 0.95:
            recommendations.append("立即停止非必要的API调用")
            recommendations.append("考虑使用更便宜的模型")
        elif usage_percentage >= 0.8:
            recommendations.append("监控剩余预算使用情况")
            recommendations.append("优化API调用频率")
        
        recommendations.append(f"当前{period}预算使用率：{usage_percentage:.1%}")
        
        return recommendations
    
    def _analyze_cost_factors(self, trends: List[CostTrend]) -> List[str]:
        """分析成本影响因素"""
        factors = []
        
        if len(trends) >= 2:
            recent_trend = trends[-1]
            previous_trend = trends[-2]
            
            if recent_trend.total_cost > previous_trend.total_cost * 1.2:
                factors.append("API调用量显著增加")
            
            if recent_trend.average_cost_per_token > previous_trend.average_cost_per_token * 1.1:
                factors.append("使用了更昂贵的模型")
        
        return factors
    
    def _generate_model_recommendations(self, model_name: str, stats, 
                                      cost_per_token: float, efficiency_score: float) -> List[str]:
        """生成模型建议"""
        recommendations = []
        
        if efficiency_score < 0.5:
            recommendations.append("考虑替换为更高效的模型")
        
        if stats.error_rate > 0.1:
            recommendations.append("检查API调用参数，降低错误率")
        
        if cost_per_token > 0.001:
            recommendations.append("成本较高，考虑使用更便宜的替代模型")
        
        if stats.average_latency > 5.0:
            recommendations.append("响应时间较慢，考虑优化或更换模型")
        
        return recommendations
    
    def _calculate_growth_rate(self, trends: List[CostTrend]) -> float:
        """计算成本增长率"""
        if len(trends) < 2:
            return 0.0
        
        recent_costs = [trend.total_cost for trend in trends[-7:]]
        older_costs = [trend.total_cost for trend in trends[-14:-7]] if len(trends) >= 14 else []
        
        if not older_costs:
            return 0.0
        
        recent_avg = statistics.mean(recent_costs)
        older_avg = statistics.mean(older_costs)
        
        return (recent_avg - older_avg) / older_avg if older_avg > 0 else 0.0
    
    def _generate_optimization_recommendations(self, trends: List[CostTrend], 
                                             efficiency: List[ModelEfficiency], 
                                             alerts: List[BudgetAlert]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于趋势的建议
        if trends and len(trends) >= 2:
            if trends[-1].total_cost > trends[-2].total_cost * 1.5:
                recommendations.append("成本增长过快，建议审查API使用策略")
        
        # 基于模型效率的建议
        if efficiency:
            low_efficiency_models = [m for m in efficiency if m.efficiency_score < 0.5]
            if low_efficiency_models:
                recommendations.append(f"考虑优化或替换低效率模型：{', '.join([m.model_name for m in low_efficiency_models])}")
        
        # 基于预算预警的建议
        if alerts:
            critical_alerts = [a for a in alerts if a.alert_type == "critical"]
            if critical_alerts:
                recommendations.append("预算即将耗尽，建议立即采取成本控制措施")
        
        return recommendations
    
    def _export_json_report(self, report: CostAnalysisReport) -> str:
        """导出JSON格式报告"""
        # 转换为可序列化的字典
        report_dict = asdict(report)
        
        # 处理datetime对象
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        return json.dumps(report_dict, indent=2, default=datetime_handler, ensure_ascii=False)
    
    def _export_html_report(self, report: CostAnalysisReport) -> str:
        """导出HTML格式报告"""
        # 简化的HTML模板
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>成本分析报告 - {report.report_id}</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .alert {{ padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; }}
                .critical {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>HarborAI 成本分析报告</h1>
                <p>报告ID: {report.report_id}</p>
                <p>生成时间: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>分析周期: {report.period_start.strftime('%Y-%m-%d')} 至 {report.period_end.strftime('%Y-%m-%d')}</p>
            </div>
            
            <div class="section">
                <h2>总体统计</h2>
                <p>总成本: ¥{report.total_cost:.4f}</p>
                <p>总Token数: {report.total_tokens:,}</p>
                <p>总请求数: {report.total_requests:,}</p>
                <p>成本增长率: {report.growth_rate:.2%}</p>
            </div>
            
            <div class="section">
                <h2>预算预警</h2>
                {''.join([f'<div class="alert {alert.alert_type}"><strong>{alert.alert_type.upper()}:</strong> {alert.message}</div>' for alert in report.budget_alerts])}
            </div>
            
            <div class="section">
                <h2>优化建议</h2>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in report.optimization_recommendations])}
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_template


# 创建全局分析器实例
cost_analyzer = CostAnalyzer()


def get_cost_analyzer() -> CostAnalyzer:
    """
    获取全局成本分析器实例
    
    Returns:
        成本分析器实例
    """
    return cost_analyzer


def generate_daily_report() -> CostAnalysisReport:
    """
    生成日报
    
    Returns:
        日成本分析报告
    """
    return cost_analyzer.generate_comprehensive_report(days=1)


def generate_weekly_report() -> CostAnalysisReport:
    """
    生成周报
    
    Returns:
        周成本分析报告
    """
    return cost_analyzer.generate_comprehensive_report(days=7)


def generate_monthly_report() -> CostAnalysisReport:
    """
    生成月报
    
    Returns:
        月成本分析报告
    """
    return cost_analyzer.generate_comprehensive_report(days=30)