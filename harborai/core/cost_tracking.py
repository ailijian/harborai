#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成本追踪模块

提供完整的成本追踪、预算管理和成本优化功能。
包含Token计数、定价计算、预算管理、成本报告和优化建议等核心功能。
"""

import uuid
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
from enum import Enum

from .pricing import PricingCalculator as BasePricingCalculator, ModelPricing
from ..monitoring.cost_analysis import CostAnalyzer, CostAnalysisReport
from ..utils.logger import get_logger
from .exceptions import HarborAIError

logger = get_logger(__name__)


class BudgetExceededError(HarborAIError):
    """预算超限异常"""
    pass


class TokenType(Enum):
    """Token类型枚举"""
    INPUT = "input"
    OUTPUT = "output"
    TOTAL = "total"


class CostCategory(Enum):
    """成本类别枚举"""
    API_CALLS = "api_calls"
    TOKEN_USAGE = "token_usage"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    COMPUTE = "compute"


class BudgetPeriod(Enum):
    """预算周期枚举"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class TokenUsage:
    """Token使用量"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class CostBreakdown:
    """成本分解"""
    input_cost: Decimal = Decimal('0')
    output_cost: Decimal = Decimal('0')
    total_cost: Decimal = Decimal('0')
    currency: str = "RMB"
    
    def __post_init__(self):
        if self.total_cost == Decimal('0'):
            self.total_cost = self.input_cost + self.output_cost


@dataclass
class ApiCall:
    """API调用记录"""
    id: str
    timestamp: datetime
    provider: str
    model: str
    endpoint: str
    token_usage: TokenUsage
    cost_breakdown: CostBreakdown
    request_size: int  # 字节
    response_size: int  # 字节
    duration: float  # 秒
    status: str  # "success", "error", "timeout"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class Budget:
    """预算配置"""
    id: str
    name: str
    amount: Decimal
    period: BudgetPeriod
    currency: str = "RMB"
    categories: List[CostCategory] = field(default_factory=list)
    providers: List[str] = field(default_factory=list)
    models: List[str] = field(default_factory=list)
    users: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.9])  # 50%, 80%, 90%
    enabled: bool = True


@dataclass
class CostReport:
    """成本报告"""
    period_start: datetime
    period_end: datetime
    total_cost: Decimal
    currency: str
    breakdown_by_provider: Dict[str, Decimal] = field(default_factory=dict)
    breakdown_by_model: Dict[str, Decimal] = field(default_factory=dict)
    breakdown_by_category: Dict[str, Decimal] = field(default_factory=dict)
    breakdown_by_user: Dict[str, Decimal] = field(default_factory=dict)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    api_call_count: int = 0
    average_cost_per_call: Decimal = Decimal('0')
    average_cost_per_token: Decimal = Decimal('0')
    top_expensive_calls: List[ApiCall] = field(default_factory=list)


class TokenCounter:
    """Token计数器"""
    
    def __init__(self):
        self.encoding_cache = {}
        self.model_encodings = {
            "deepseek-chat": "cl100k_base",
            "deepseek-reasoner": "cl100k_base",
            "ernie-3.5-8k": "cl100k_base",
            "ernie-4.0-turbo-8k": "ernie",
            "doubao-1-5-pro-32k-character-250715": "doubao",
            "gemini-pro": "gemini"
        }
    
    def count_tokens(self, text: str, model: str = "deepseek-chat") -> int:
        """计算文本的Token数量"""
        if not text:
            return 0
        
        # 基于模型的不同计算方式
        if model.startswith("gpt"):
            # GPT模型大约4个字符=1个token
            return max(1, len(text) // 4)
        elif model.startswith("ernie"):
            # ERNIE模型大约3.5个字符=1个token
            return max(1, int(len(text) / 3.5))
        elif model.startswith("doubao"):
            # Doubao模型大约3.5个字符=1个token
            return max(1, int(len(text) / 3.5))
        elif model.startswith("gemini"):
            # Gemini模型大约4.5个字符=1个token
            return max(1, int(len(text) / 4.5))
        else:
            # 默认计算方式
            return max(1, len(text) // 4)
    
    def count_message_tokens(self, messages: List[Dict[str, Any]], model: str = "deepseek-chat") -> TokenUsage:
        """计算消息列表的Token使用量"""
        input_tokens = 0
        
        for message in messages:
            # 计算消息内容的tokens
            content = message.get("content", "")
            if isinstance(content, str):
                input_tokens += self.count_tokens(content, model)
            elif isinstance(content, list):
                # 处理多模态内容
                for item in content:
                    if item.get("type") == "text":
                        input_tokens += self.count_tokens(item.get("text", ""), model)
                    elif item.get("type") == "image_url":
                        # 图片token计算（简化）
                        input_tokens += 85  # 基础图片token
            
            # 添加消息格式的额外tokens
            input_tokens += 4  # 每条消息的格式开销
        
        # 添加对话格式的额外tokens
        input_tokens += 2  # 对话开始和结束的tokens
        
        return TokenUsage(input_tokens=input_tokens, output_tokens=0, total_tokens=input_tokens)
    
    def count_response_tokens(self, response: str, model: str = "deepseek-chat") -> TokenUsage:
        """计算响应的Token使用量"""
        output_tokens = self.count_tokens(response, model)
        return TokenUsage(input_tokens=0, output_tokens=output_tokens, total_tokens=output_tokens)
    
    def estimate_tokens(self, text: str, model: str = "deepseek-chat") -> int:
        """估算文本的Token数量（快速估算）"""
        return self.count_tokens(text, model)


class PricingCalculator(BasePricingCalculator):
    """扩展的定价计算器"""
    
    def __init__(self):
        super().__init__()
        # 批量折扣（基于月使用量）
        self.volume_discounts = {
            1000000: 0.05,    # 100万tokens以上5%折扣
            5000000: 0.10,    # 500万tokens以上10%折扣
            10000000: 0.15,   # 1000万tokens以上15%折扣
            50000000: 0.20    # 5000万tokens以上20%折扣
        }
    
    def calculate_cost_breakdown(self, provider: str, model: str, token_usage: TokenUsage) -> CostBreakdown:
        """计算详细成本分解"""
        # 获取模型定价
        pricing = self.get_model_pricing(model)
        if not pricing:
            # 使用默认定价
            pricing = ModelPricing(input_price=0.01, output_price=0.02)
        
        # 计算输入和输出成本（价格是每1000个tokens）
        input_cost = Decimal(str((token_usage.input_tokens / 1000) * pricing.input_price))
        output_cost = Decimal(str((token_usage.output_tokens / 1000) * pricing.output_price))
        
        # 四舍五入到6位小数
        input_cost = input_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
        output_cost = output_cost.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP)
        
        return CostBreakdown(
            input_cost=input_cost,
            output_cost=output_cost,
            currency="RMB"
        )
    
    def calculate_cost_with_discount(self, provider: str, model: str, token_usage: TokenUsage, 
                                   monthly_volume: int = 0) -> CostBreakdown:
        """计算带折扣的成本"""
        base_cost = self.calculate_cost_breakdown(provider, model, token_usage)
        
        # 应用批量折扣
        discount_rate = 0.0
        for volume_threshold, discount in sorted(self.volume_discounts.items(), reverse=True):
            if monthly_volume >= volume_threshold:
                discount_rate = discount
                break
        
        if discount_rate > 0:
            discount_multiplier = Decimal(str(1 - discount_rate))
            base_cost.input_cost *= discount_multiplier
            base_cost.output_cost *= discount_multiplier
            base_cost.total_cost = base_cost.input_cost + base_cost.output_cost
        
        return base_cost
    
    def compare_provider_costs(self, token_usage: TokenUsage, models: List[tuple]) -> Dict[str, CostBreakdown]:
        """比较不同提供商的成本"""
        costs = {}
        
        for provider, model in models:
            cost = self.calculate_cost_breakdown(provider, model, token_usage)
            costs[f"{provider}/{model}"] = cost
        
        return costs
    
    def estimate_monthly_cost(self, daily_usage: TokenUsage, provider: str, model: str) -> Decimal:
        """估算月度成本"""
        monthly_usage = TokenUsage(
            input_tokens=daily_usage.input_tokens * 30,
            output_tokens=daily_usage.output_tokens * 30
        )
        
        cost = self.calculate_cost_with_discount(provider, model, monthly_usage, monthly_usage.total_tokens)
        return cost.total_cost


class BudgetManager:
    """预算管理器"""
    
    def __init__(self):
        self.budgets: Dict[str, Budget] = {}
        self.current_usage: Dict[str, Decimal] = defaultdict(lambda: Decimal('0'))
    
    def create_budget(self, name: str, amount: Decimal, period: BudgetPeriod, 
                     categories: List[CostCategory] = None, 
                     providers: List[str] = None,
                     models: List[str] = None,
                     users: List[str] = None) -> Budget:
        """创建预算"""
        budget_id = str(uuid.uuid4())
        budget = Budget(
            id=budget_id,
            name=name,
            amount=amount,
            period=period,
            categories=categories or [],
            providers=providers or [],
            models=models or [],
            users=users or []
        )
        
        self.budgets[budget_id] = budget
        logger.info(f"Created budget '{name}' with amount ${amount} for {period.value}")
        return budget
    
    def check_budget(self, budget_id: str, cost: Decimal) -> bool:
        """检查预算是否超限"""
        budget = self.budgets.get(budget_id)
        if not budget or not budget.enabled:
            return True
        
        current_usage = self.current_usage[budget_id]
        new_usage = current_usage + cost
        
        if new_usage > budget.amount:
            logger.warning(f"Budget '{budget.name}' exceeded: ${new_usage} > ${budget.amount}")
            return False
        
        # 检查预警阈值
        for threshold in budget.alert_thresholds:
            threshold_amount = budget.amount * Decimal(str(threshold))
            if current_usage <= threshold_amount < new_usage:
                usage_percentage = float(new_usage / budget.amount * 100)
                logger.warning(
                    f"Budget '{budget.name}' alert: {usage_percentage:.1f}% used "
                    f"(${new_usage}/${budget.amount})"
                )
        
        return True
    
    def update_usage(self, budget_id: str, cost: Decimal):
        """更新预算使用量"""
        if budget_id in self.budgets:
            self.current_usage[budget_id] += cost
    
    def get_budget_status(self, budget_id: str) -> Dict[str, Any]:
        """获取预算状态"""
        budget = self.budgets.get(budget_id)
        if not budget:
            return {}
        
        current_usage = self.current_usage[budget_id]
        remaining = budget.amount - current_usage
        usage_percentage = float(current_usage / budget.amount * 100)
        
        return {
            "budget_id": budget_id,
            "name": budget.name,
            "amount": budget.amount,
            "current_usage": current_usage,
            "remaining": remaining,
            "usage_percentage": usage_percentage,
            "period": budget.period.value,
            "enabled": budget.enabled
        }


class CostTracker:
    """成本追踪器"""
    
    def __init__(self):
        self.api_calls: List[ApiCall] = []
        self.token_counter = TokenCounter()
        self.pricing_calculator = PricingCalculator()
        self.budget_manager = BudgetManager()
        self.current_costs: Dict[str, Decimal] = defaultdict(lambda: Decimal('0'))
        self.daily_costs: Dict[str, Decimal] = defaultdict(lambda: Decimal('0'))
        self.monthly_costs: Dict[str, Decimal] = defaultdict(lambda: Decimal('0'))
    
    def track_api_call(self, provider: str, model: str, endpoint: str, 
                      messages: List[Dict[str, Any]], response: str,
                      duration: float, status: str = "success",
                      user_id: str = None, session_id: str = None,
                      tags: Dict[str, str] = None) -> ApiCall:
        """追踪API调用"""
        # 计算token使用量
        input_usage = self.token_counter.count_message_tokens(messages, model)
        output_usage = self.token_counter.count_response_tokens(response, model)
        
        total_usage = TokenUsage(
            input_tokens=input_usage.input_tokens,
            output_tokens=output_usage.output_tokens
        )
        
        # 计算成本
        cost_breakdown = self.pricing_calculator.calculate_cost_breakdown(provider, model, total_usage)
        
        # 创建API调用记录
        api_call = ApiCall(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            endpoint=endpoint,
            token_usage=total_usage,
            cost_breakdown=cost_breakdown,
            request_size=len(json.dumps(messages).encode('utf-8')),
            response_size=len(response.encode('utf-8')),
            duration=duration,
            status=status,
            user_id=user_id,
            session_id=session_id,
            tags=tags or {}
        )
        
        # 记录调用
        self.api_calls.append(api_call)
        
        # 更新成本统计
        self._update_cost_stats(api_call)
        
        return api_call
    
    def _update_cost_stats(self, api_call: ApiCall):
        """更新成本统计"""
        cost = api_call.cost_breakdown.total_cost
        
        # 更新总成本
        self.current_costs["total"] += cost
        self.current_costs[f"provider_{api_call.provider}"] += cost
        self.current_costs[f"model_{api_call.model}"] += cost
        
        if api_call.user_id:
            self.current_costs[f"user_{api_call.user_id}"] += cost
    
    def get_cost_summary(self, period: str = "total") -> Dict[str, Any]:
        """获取成本摘要"""
        if period == "total":
            total_cost = self.current_costs["total"]
            total_calls = len(self.api_calls)
            total_tokens = sum(call.token_usage.total_tokens for call in self.api_calls)
            calls_data = self.api_calls
        else:
            # 根据时间段过滤
            filtered_calls = self._filter_calls_by_period(period)
            total_cost = sum(call.cost_breakdown.total_cost for call in filtered_calls)
            total_calls = len(filtered_calls)
            total_tokens = sum(call.token_usage.total_tokens for call in filtered_calls)
            calls_data = filtered_calls
        
        avg_cost_per_call = total_cost / total_calls if total_calls > 0 else Decimal('0')
        avg_cost_per_token = total_cost / total_tokens if total_tokens > 0 else Decimal('0')
        
        # 将API调用转换为字典格式，便于序列化和测试
        calls_list = []
        for call in calls_data:
            # 处理timestamp字段，可能是datetime对象或float时间戳
            if isinstance(call.timestamp, datetime):
                timestamp_str = call.timestamp.isoformat()
            elif isinstance(call.timestamp, (int, float)):
                timestamp_str = datetime.fromtimestamp(call.timestamp).isoformat()
            else:
                timestamp_str = str(call.timestamp)
                
            calls_list.append({
                "id": call.id,
                "timestamp": timestamp_str,
                "provider": call.provider,
                "model": call.model,
                "endpoint": call.endpoint,
                "total_tokens": call.token_usage.total_tokens,
                "prompt_tokens": call.token_usage.input_tokens,
                "completion_tokens": call.token_usage.output_tokens,
                "total_cost": float(call.cost_breakdown.total_cost),
                "duration": call.duration,
                "status": call.status,
                "user_id": call.user_id
            })
        
        return {
            "total_cost": total_cost,
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "average_cost_per_call": avg_cost_per_call,
            "average_cost_per_token": avg_cost_per_token,
            "currency": "USD",
            "calls": calls_list
        }
    
    def _filter_calls_by_period(self, period: str) -> List[ApiCall]:
        """根据时间段过滤API调用"""
        now = datetime.now()
        
        if period == "today":
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "week":
            start_time = now - timedelta(days=7)
        elif period == "month":
            start_time = now - timedelta(days=30)
        else:
            return self.api_calls
        
        return [call for call in self.api_calls if call.timestamp >= start_time]


class CostReporter:
    """成本报告器"""
    
    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker
        self.cost_analyzer = CostAnalyzer()
    
    def generate_report(self, period_start: datetime, period_end: datetime) -> CostReport:
        """生成成本报告"""
        # 过滤时间段内的API调用
        filtered_calls = [
            call for call in self.cost_tracker.api_calls
            if period_start <= call.timestamp <= period_end
        ]
        
        if not filtered_calls:
            return CostReport(
                period_start=period_start,
                period_end=period_end,
                total_cost=Decimal('0'),
                currency="RMB"
            )
        
        # 计算总成本和统计
        total_cost = sum(call.cost_breakdown.total_cost for call in filtered_calls)
        total_tokens = sum(call.token_usage.total_tokens for call in filtered_calls)
        total_calls = len(filtered_calls)
        
        # 按提供商分解
        breakdown_by_provider = defaultdict(lambda: Decimal('0'))
        for call in filtered_calls:
            breakdown_by_provider[call.provider] += call.cost_breakdown.total_cost
        
        # 按模型分解
        breakdown_by_model = defaultdict(lambda: Decimal('0'))
        for call in filtered_calls:
            breakdown_by_model[call.model] += call.cost_breakdown.total_cost
        
        # 按用户分解
        breakdown_by_user = defaultdict(lambda: Decimal('0'))
        for call in filtered_calls:
            if call.user_id:
                breakdown_by_user[call.user_id] += call.cost_breakdown.total_cost
        
        # 计算平均值
        avg_cost_per_call = total_cost / total_calls if total_calls > 0 else Decimal('0')
        avg_cost_per_token = total_cost / total_tokens if total_tokens > 0 else Decimal('0')
        
        # 获取最昂贵的调用
        top_expensive_calls = sorted(
            filtered_calls,
            key=lambda x: x.cost_breakdown.total_cost,
            reverse=True
        )[:10]
        
        return CostReport(
            period_start=period_start,
            period_end=period_end,
            total_cost=total_cost,
            currency="RMB",
            breakdown_by_provider=dict(breakdown_by_provider),
            breakdown_by_model=dict(breakdown_by_model),
            breakdown_by_user=dict(breakdown_by_user),
            token_usage=TokenUsage(
                input_tokens=sum(call.token_usage.input_tokens for call in filtered_calls),
                output_tokens=sum(call.token_usage.output_tokens for call in filtered_calls),
                total_tokens=total_tokens
            ),
            api_call_count=total_calls,
            average_cost_per_call=avg_cost_per_call,
            average_cost_per_token=avg_cost_per_token,
            top_expensive_calls=top_expensive_calls
        )


class CostOptimizer:
    """成本优化器"""
    
    def __init__(self, cost_tracker: CostTracker):
        self.cost_tracker = cost_tracker
        self.pricing_calculator = cost_tracker.pricing_calculator
    
    def analyze_model_efficiency(self) -> List[Dict[str, Any]]:
        """分析模型效率"""
        model_stats = defaultdict(lambda: {
            'total_cost': Decimal('0'),
            'total_tokens': 0,
            'total_calls': 0,
            'total_duration': 0.0,
            'success_count': 0
        })
        
        for call in self.cost_tracker.api_calls:
            stats = model_stats[call.model]
            stats['total_cost'] += call.cost_breakdown.total_cost
            stats['total_tokens'] += call.token_usage.total_tokens
            stats['total_calls'] += 1
            stats['total_duration'] += call.duration
            if call.status == 'success':
                stats['success_count'] += 1
        
        efficiency_analysis = []
        for model, stats in model_stats.items():
            if stats['total_calls'] > 0:
                cost_per_token = stats['total_cost'] / stats['total_tokens'] if stats['total_tokens'] > 0 else Decimal('0')
                avg_duration = stats['total_duration'] / stats['total_calls']
                success_rate = stats['success_count'] / stats['total_calls']
                
                # 计算效率评分（成本越低、速度越快、成功率越高，评分越高）
                efficiency_score = (
                    (1 / float(cost_per_token) if cost_per_token > 0 else 0) * 0.4 +
                    (1 / avg_duration if avg_duration > 0 else 0) * 0.3 +
                    success_rate * 0.3
                )
                
                efficiency_analysis.append({
                    'model': model,
                    'cost_per_token': cost_per_token,
                    'average_duration': avg_duration,
                    'success_rate': success_rate,
                    'efficiency_score': efficiency_score,
                    'total_calls': stats['total_calls']
                })
        
        return sorted(efficiency_analysis, key=lambda x: x['efficiency_score'], reverse=True)
    
    def suggest_optimizations(self) -> List[str]:
        """提供优化建议"""
        suggestions = []
        
        # 分析模型效率
        efficiency_analysis = self.analyze_model_efficiency()
        
        if len(efficiency_analysis) > 1:
            best_model = efficiency_analysis[0]
            worst_model = efficiency_analysis[-1]
            
            if best_model['efficiency_score'] > worst_model['efficiency_score'] * 1.5:
                suggestions.append(
                    f"建议优先使用 {best_model['model']} 模型，其效率评分为 "
                    f"{best_model['efficiency_score']:.2f}，显著高于 {worst_model['model']} "
                    f"的 {worst_model['efficiency_score']:.2f}"
                )
        
        # 分析成本趋势
        recent_calls = self.cost_tracker._filter_calls_by_period("week")
        if len(recent_calls) > 10:
            avg_cost = sum(call.cost_breakdown.total_cost for call in recent_calls) / len(recent_calls)
            high_cost_calls = [call for call in recent_calls if call.cost_breakdown.total_cost > avg_cost * 2]
            
            if len(high_cost_calls) > len(recent_calls) * 0.1:  # 超过10%的调用成本过高
                suggestions.append(
                    f"发现 {len(high_cost_calls)} 次高成本调用，建议检查输入长度和模型选择"
                )
        
        # 检查失败率
        failed_calls = [call for call in self.cost_tracker.api_calls if call.status != 'success']
        if len(failed_calls) > len(self.cost_tracker.api_calls) * 0.05:  # 失败率超过5%
            suggestions.append(
                f"API调用失败率为 {len(failed_calls)/len(self.cost_tracker.api_calls)*100:.1f}%，"
                "建议检查网络连接和重试机制"
            )
        
        return suggestions