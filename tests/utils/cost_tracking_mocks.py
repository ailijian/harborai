# -*- coding: utf-8 -*-
"""
成本跟踪相关的Mock类

功能：为成本跟踪功能测试提供Mock实现
作者：HarborAI测试团队
创建时间：2024
"""

import time
import random
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

class BudgetPeriod(Enum):
    """预算周期枚举"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


@dataclass
class TokenUsage:
    """Token使用量数据类"""
    input_tokens: int
    output_tokens: int
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostBreakdown:
    """成本分解"""
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str = "RMB"


@dataclass
class ApiCall:
    """API调用记录"""
    timestamp: datetime
    provider: str
    model: str
    endpoint: str
    token_usage: TokenUsage
    cost_breakdown: CostBreakdown
    duration: float
    status: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: Optional[Dict] = None
    metadata: Optional[Dict] = None
    id: Optional[str] = None
    request_size: int = 0
    response_size: int = 0
    
    def __post_init__(self):
        """生成唯一ID"""
        if self.id is None:
            import uuid
            self.id = str(uuid.uuid4())


class MockTokenCounter:
    """Mock Token计数器
    
    功能：模拟Token计数功能
    假设：不同模型有不同的Token计数规则
    验证方法：pytest tests/functional/test_i_cost_tracking.py
    """
    
    def __init__(self):
        """初始化Mock Token计数器"""
        self.model_multipliers = {
            "deepseek-chat": 1.0,
            "deepseek-reasoner": 1.2,
            "ernie-4.0-8k": 1.1,
            "ernie-3.5-8k": 1.0,
            "doubao-pro-4k": 0.9,
            "doubao-lite-4k": 0.8
        }
    
    def count_tokens(self, text: str, model: str) -> int:
        """计算文本的Token数量
        
        参数：
            text: 输入文本
            model: 模型名称
        返回：
            Token数量
        """
        if not text:
            return 0
        
        # 简单的Token计数模拟：基于字符数和模型特性
        base_tokens = max(1, len(text) // 4)  # 大约4个字符一个token
        multiplier = self.model_multipliers.get(model, 1.0)
        
        return int(base_tokens * multiplier)
    
    def estimate_tokens(self, text: str, model: str) -> int:
        """估算文本的Token数量（与count_tokens相同的简化实现）"""
        return self.count_tokens(text, model)
    
    def count_message_tokens(self, messages: List[Dict[str, str]], model: str) -> TokenUsage:
        """计算消息列表的Token数量"""
        input_tokens = 0
        output_tokens = 0
        
        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "")
            
            # 处理多模态内容
            if isinstance(content, list):
                text_content = ""
                image_tokens = 0
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_content += item.get("text", "")
                        elif item.get("type") == "image_url":
                            # 图片通常占用较多tokens，这里模拟一个固定值
                            image_tokens += 85  # 假设每张图片85个tokens
                content = text_content
                content_tokens = self.count_tokens(content, model) + image_tokens
            else:
                content_tokens = self.count_tokens(content, model)
            
            role_tokens = 3  # 角色标识的固定开销
            
            # 根据角色分类tokens
            if role in ["user", "system"]:
                input_tokens += content_tokens + role_tokens
            elif role == "assistant":
                output_tokens += content_tokens + role_tokens
            else:
                # 未知角色默认为输入
                input_tokens += content_tokens + role_tokens
        
        return TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    
    def count_response_tokens(self, response: Dict[str, Any], model: str) -> TokenUsage:
        """计算响应的Token数量"""
        # 从响应中提取内容
        content = ""
        if isinstance(response, dict):
            if "choices" in response and response["choices"]:
                choice = response["choices"][0]
                if "message" in choice:
                    content = choice["message"].get("content", "")
                elif "text" in choice:
                    content = choice["text"]
            elif "content" in response:
                content = response["content"]
        elif isinstance(response, str):
            content = response
        
        output_tokens = self.count_tokens(content, model)
        # 响应通常没有输入tokens（这里是输出）
        return TokenUsage(input_tokens=0, output_tokens=output_tokens)


class MockPricingCalculator:
    """Mock 定价计算器
    
    功能：模拟各厂商模型的定价计算
    假设：定价信息相对稳定
    验证方法：pytest tests/functional/test_i_cost_tracking.py
    """
    
    def __init__(self):
        """初始化Mock定价计算器"""
        # 模拟的定价信息（每1000个token的价格，单位：元）
        self.pricing_data = {
            "deepseek": {
                "deepseek-chat": {"input": 0.0014, "output": 0.0028},
                "deepseek-reasoner": {"input": 0.0055, "output": 0.0280}
            },
            "ernie": {
                "ernie-4.0-8k": {"input": 0.0120, "output": 0.0120},
                "ernie-3.5-8k": {"input": 0.0080, "output": 0.0080}
            },
            "doubao": {
                "doubao-pro-4k": {"input": 0.0008, "output": 0.0020},
                "doubao-lite-4k": {"input": 0.0003, "output": 0.0006}
            }
        }
    
    def get_model_pricing(self, provider: str, model: str) -> Dict[str, float]:
        """获取模型定价信息"""
        return self.pricing_data.get(provider, {}).get(model, {"input": 0.001, "output": 0.002})
    
    def calculate_cost(self, provider: str, model: str, token_usage: TokenUsage) -> CostBreakdown:
        """计算成本
        
        参数：
            provider: 提供商名称
            model: 模型名称
            token_usage: Token使用量
        返回：
            成本分解
        """
        pricing = self.get_model_pricing(provider, model)
        
        # 计算成本（价格是每1000个token）
        input_cost = (token_usage.input_tokens / 1000) * pricing["input"]
        output_cost = (token_usage.output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost
        
        return CostBreakdown(
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total_cost, 6)
        )
    
    def compare_provider_costs(self, token_usage: TokenUsage, models: List[tuple]) -> Dict[str, CostBreakdown]:
        """比较不同提供商的成本"""
        costs = {}
        for provider, model in models:
            cost = self.calculate_cost(provider, model, token_usage)
            costs[f"{provider}/{model}"] = cost
        return costs
    
    def estimate_monthly_cost(self, daily_usage: TokenUsage, provider: str, model: str) -> float:
        """估算月度成本"""
        daily_cost = self.calculate_cost(provider, model, daily_usage)
        # 假设有5%的批量折扣
        monthly_cost = daily_cost.total_cost * 30 * 0.95
        return round(monthly_cost, 6)
    
    def calculate_cost_with_discount(self, provider: str, model: str, token_usage: TokenUsage, 
                                   volume_tier: str = "standard", monthly_volume: Optional[int] = None) -> CostBreakdown:
        """计算带折扣的成本
        
        参数：
            provider: 提供商名称
            model: 模型名称
            token_usage: Token使用量
            volume_tier: 用量等级 ("standard", "high", "enterprise")
        返回：
            成本分解
        """
        base_cost = self.calculate_cost(provider, model, token_usage)
        
        # 根据用量等级应用折扣
        discount_rates = {
            "standard": 0.0,    # 无折扣
            "high": 0.1,        # 10%折扣
            "enterprise": 0.2   # 20%折扣
        }
        
        discount_rate = discount_rates.get(volume_tier, 0.0)
        
        # 如果提供了月度用量，根据用量调整折扣
        if monthly_volume is not None:
            if monthly_volume > 10000000:  # 超过1000万tokens
                discount_rate = max(discount_rate, 0.25)  # 至少25%折扣
            elif monthly_volume > 1000000:  # 超过100万tokens
                discount_rate = max(discount_rate, 0.15)  # 至少15%折扣
            elif monthly_volume > 500000:  # 超过50万tokens
                discount_rate = max(discount_rate, 0.1)   # 至少10%折扣
        
        discount_multiplier = 1.0 - discount_rate
        
        return CostBreakdown(
            input_cost=round(base_cost.input_cost * discount_multiplier, 6),
            output_cost=round(base_cost.output_cost * discount_multiplier, 6),
            total_cost=round(base_cost.total_cost * discount_multiplier, 6),
            currency=base_cost.currency
        )


class MockCostTracker:
    """Mock 成本跟踪器
    
    功能：跟踪API调用成本
    假设：所有API调用都能正确记录
    验证方法：pytest tests/functional/test_i_cost_tracking.py
    """
    
    def __init__(self):
        """初始化Mock成本跟踪器"""
        self.api_calls = []
        self.total_cost = 0.0
        self.token_counter = MockTokenCounter()
        self.pricing_calculator = MockPricingCalculator()
    
    def track_api_call(self, provider: str, model: str, endpoint: str = "/v1/chat/completions", 
                      messages: Optional[list] = None, response: Optional[str] = None,
                      duration: Optional[float] = None, status: str = "success",
                      user_id: Optional[str] = None, session_id: Optional[str] = None,
                      tags: Optional[Dict] = None, input_text: Optional[str] = None, 
                      output_text: Optional[str] = None, metadata: Optional[Dict] = None) -> 'ApiCall':
        """跟踪API调用
        
        参数：
            provider: 提供商
            model: 模型
            endpoint: API端点
            messages: 消息列表
            response: 响应文本
            duration: 调用时长
            status: 调用状态
            user_id: 用户ID
            input_text: 输入文本（向后兼容）
            output_text: 输出文本（向后兼容）
            metadata: 元数据
        返回：
            ApiCall对象
        """
        # 确定输入和输出文本
        if messages and not input_text:
            # 从messages中提取文本
            input_text = ""
            for msg in messages:
                if isinstance(msg.get("content"), str):
                    input_text += msg["content"] + " "
                elif isinstance(msg.get("content"), list):
                    # 处理多模态内容
                    for content in msg["content"]:
                        if content.get("type") == "text":
                            input_text += content.get("text", "") + " "
            input_text = input_text.strip()
        
        if response and not output_text:
            output_text = response
        
        # 使用默认值如果没有提供
        input_text = input_text or "Hello"
        output_text = output_text or "Hi there!"
        
        # 计算token使用量
        if messages:
            # 计算输入tokens
            input_usage = self.token_counter.count_message_tokens(messages, model)
            input_tokens = input_usage.input_tokens
            
            # 计算输出tokens（如果有响应）
            if output_text:
                output_tokens = self.token_counter.count_tokens(output_text, model)
            else:
                output_tokens = 0
                
            token_usage = TokenUsage(input_tokens, output_tokens)
        else:
            input_tokens = self.token_counter.count_tokens(input_text, model)
            output_tokens = self.token_counter.count_tokens(output_text, model)
            token_usage = TokenUsage(input_tokens, output_tokens)
        
        # 计算成本
        cost_breakdown = self.pricing_calculator.calculate_cost(provider, model, token_usage)
        
        # 计算请求和响应大小
        request_size = 0
        response_size = 0
        
        if messages:
            # 计算消息的字节大小
            import json
            request_size = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
        elif input_text:
            request_size = len(input_text.encode('utf-8'))
            
        if output_text:
            response_size = len(output_text.encode('utf-8'))
        elif response:
            # 如果有响应对象，计算其大小
            import json
            response_size = len(json.dumps(response, ensure_ascii=False).encode('utf-8'))

        # 创建ApiCall对象
        api_call = ApiCall(
            timestamp=datetime.now(),
            provider=provider,
            model=model,
            endpoint=endpoint,
            token_usage=token_usage,
            cost_breakdown=cost_breakdown,
            duration=duration or 1.0,
            status=status,
            user_id=user_id,
            session_id=session_id,
            tags=tags or {},
            metadata=metadata or {},
            request_size=request_size,
            response_size=response_size
        )
        
        self.api_calls.append(api_call)
        self.total_cost += cost_breakdown.total_cost
        
        return api_call
    
    def get_token_usage(self, start_time: Optional[datetime] = None, 
                       end_time: Optional[datetime] = None) -> TokenUsage:
        """获取token使用量统计"""
        total_input = 0
        total_output = 0
        
        for call in self.api_calls:
            if start_time and call.timestamp < start_time:
                continue
            if end_time and call.timestamp > end_time:
                continue
            total_input += call.token_usage.input_tokens
            total_output += call.token_usage.output_tokens
        
        return TokenUsage(total_input, total_output)
    
    def get_total_cost(self, start_time: Optional[datetime] = None, 
                      end_time: Optional[datetime] = None, start_date: Optional[datetime] = None,
                      end_date: Optional[datetime] = None) -> float:
        """获取总成本"""
        # 处理参数别名
        if start_date:
            start_time = start_date
        if end_date:
            end_time = end_date
            
        if not start_time and not end_time:
            return self.total_cost
        
        total = 0.0
        for call in self.api_calls:
            if start_time and call.timestamp < start_time:
                continue
            if end_time and call.timestamp > end_time:
                continue
            total += call.cost_breakdown.total_cost
        
        return total
    
    def get_expensive_calls(self, threshold: float = 1.0) -> List[ApiCall]:
        """获取高成本调用"""
        return [call for call in self.api_calls 
                if call.cost_breakdown.total_cost > threshold]
    
    def get_cost_by_provider(self, start_time=None, end_time=None) -> Dict[str, float]:
        """按提供商获取成本统计"""
        provider_costs = {}
        for call in self.api_calls:
            # 时间过滤
            if start_time and call.timestamp < start_time:
                continue
            if end_time and call.timestamp > end_time:
                continue
                
            provider = call.provider
            if provider not in provider_costs:
                provider_costs[provider] = 0.0
            provider_costs[provider] += call.cost_breakdown.total_cost
        return provider_costs
    
    def get_most_expensive_calls(self, limit: int = 10, start_time=None, end_time=None) -> List[ApiCall]:
        """获取最昂贵的调用"""
        filtered_calls = []
        for call in self.api_calls:
            # 时间过滤
            if start_time and call.timestamp < start_time:
                continue
            if end_time and call.timestamp > end_time:
                continue
            filtered_calls.append(call)
        
        # 按成本排序并返回前N个
        sorted_calls = sorted(filtered_calls, 
                            key=lambda x: x.cost_breakdown.total_cost, 
                            reverse=True)
        return sorted_calls[:limit]
    
    def get_cost_by_model(self, start_time=None, end_time=None) -> Dict[str, float]:
        """按模型获取成本统计"""
        model_costs = {}
        for call in self.api_calls:
            # 时间过滤
            if start_time and call.timestamp < start_time:
                continue
            if end_time and call.timestamp > end_time:
                continue
                
            model = call.model
            if model not in model_costs:
                model_costs[model] = 0.0
            model_costs[model] += call.cost_breakdown.total_cost
        return model_costs
    
    def get_cost_by_user(self, start_time=None, end_time=None) -> Dict[str, float]:
        """按用户获取成本统计"""
        user_costs = {}
        for call in self.api_calls:
            # 时间过滤
            if start_time and call.timestamp < start_time:
                continue
            if end_time and call.timestamp > end_time:
                continue
                
            user_id = call.user_id or "anonymous"
            if user_id not in user_costs:
                user_costs[user_id] = 0.0
            user_costs[user_id] += call.cost_breakdown.total_cost
        return user_costs


@dataclass
class Budget:
    """预算数据类"""
    name: str
    amount: float  # 预算金额
    period: BudgetPeriod  # 预算周期
    providers: Optional[List[str]] = None
    models: Optional[List[str]] = None
    currency: str = "RMB"
    enabled: bool = True
    alert_thresholds: List[float] = None
    created_at: datetime = None
    id: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.id is None:
            import uuid
            self.id = str(uuid.uuid4())
        if self.alert_thresholds is None:
            self.alert_thresholds = [0.5, 0.8, 0.9]  # 50%, 80%, 90%


class MockBudgetManager:
    """Mock 预算管理器
    
    功能：管理成本预算和告警
    假设：预算配置有效
    验证方法：pytest tests/functional/test_i_cost_tracking.py
    """
    
    def __init__(self, cost_tracker=None):
        """初始化Mock预算管理器"""
        self.budgets = {}
        self.cost_tracker = cost_tracker or MockCostTracker()
    
    def create_budget(self, name: str, amount: float, period: BudgetPeriod,
                     providers: Optional[List[str]] = None,
                     models: Optional[List[str]] = None,
                     currency: str = "RMB",
                     enabled: bool = True) -> Budget:
        """创建预算"""
        budget = Budget(
            name=name,
            amount=amount,
            period=period,
            providers=providers,
            models=models,
            currency=currency,
            enabled=enabled
        )
        self.budgets[name] = budget
        return budget
    
    def check_budget_status(self, budget_name: str) -> Dict[str, Any]:
        """检查预算状态"""
        if budget_name not in self.budgets:
            raise ValueError(f"Budget '{budget_name}' not found")
        
        budget = self.budgets[budget_name]
        
        # 计算当前周期的使用量
        start_time = self._get_period_start(budget.period)
        current_cost = self.cost_tracker.get_total_cost(start_time)
        
        # 如果有提供商或模型过滤，需要进一步过滤
        if budget.providers or budget.models:
            current_cost = self._get_filtered_cost(budget, start_time)
        
        usage_percentage = (current_cost / budget.amount) * 100 if budget.amount > 0 else 0
        
        return {
            "budget_name": budget_name,
            "limit": budget.amount,
            "current_cost": current_cost,
            "remaining": budget.amount - current_cost,
            "usage_percentage": usage_percentage,
            "status": self._get_budget_status(usage_percentage),
            "period": budget.period,
            "period_start": start_time
        }
    
    def get_budget_alerts(self, budget_name: str) -> List[Dict[str, Any]]:
        """获取预算告警"""
        status = self.check_budget_status(budget_name)
        alerts = []
        
        if status["usage_percentage"] >= 90:
            alerts.append({
                "level": "critical",
                "message": f"预算 '{budget_name}' 使用率已达到 {status['usage_percentage']:.1f}%",
                "timestamp": datetime.now()
            })
        elif status["usage_percentage"] >= 75:
            alerts.append({
                "level": "warning",
                "message": f"预算 '{budget_name}' 使用率已达到 {status['usage_percentage']:.1f}%",
                "timestamp": datetime.now()
            })
        
        return alerts
    
    def _get_period_start(self, period: str) -> datetime:
        """获取周期开始时间"""
        now = datetime.now()
        if period == "daily":
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "weekly":
            days_since_monday = now.weekday()
            return (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "monthly":
            return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _get_filtered_cost(self, budget: Budget, start_time: datetime) -> float:
        """获取过滤后的成本"""
        # 简化实现：返回总成本的一个比例
        total_cost = self.cost_tracker.get_total_cost(start_time)
        return total_cost * 0.8  # 假设80%的成本符合过滤条件
    
    def _get_budget_status(self, usage_percentage: float) -> str:
        """获取预算状态"""
        if usage_percentage >= 100:
            return "exceeded"
        elif usage_percentage >= 90:
            return "critical"
        elif usage_percentage >= 75:
            return "warning"
        else:
            return "normal"


class CostReport:
    """成本报告类
    
    功能：生成成本分析报告
    假设：数据完整性良好
    验证方法：pytest tests/functional/test_i_cost_tracking.py
    """
    
    def __init__(self, cost_tracker: MockCostTracker):
        """初始化成本报告"""
        self.cost_tracker = cost_tracker
    
    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """生成日报告"""
        if date is None:
            date = datetime.now()
        
        start_time = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=1)
        
        return self._generate_report(start_time, end_time, "daily")
    
    def generate_weekly_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """生成周报告"""
        if date is None:
            date = datetime.now()
        
        days_since_monday = date.weekday()
        start_time = (date - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_time = start_time + timedelta(days=7)
        
        return self._generate_report(start_time, end_time, "weekly")
    
    def generate_monthly_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """生成月报告"""
        if date is None:
            date = datetime.now()
        
        start_time = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if date.month == 12:
            end_time = start_time.replace(year=date.year + 1, month=1)
        else:
            end_time = start_time.replace(month=date.month + 1)
        
        return self._generate_report(start_time, end_time, "monthly")
    
    def _generate_report(self, start_time: datetime, end_time: datetime, period: str) -> Dict[str, Any]:
        """生成报告"""
        filtered_calls = self.cost_tracker._filter_calls_by_time(start_time, end_time)
        
        if not filtered_calls:
            return {
                "period": period,
                "start_time": start_time,
                "end_time": end_time,
                "total_cost": 0.0,
                "total_calls": 0,
                "total_tokens": 0,
                "provider_breakdown": {},
                "model_breakdown": {},
                "cost_trend": []
            }
        
        total_cost = sum(call["total_cost"] for call in filtered_calls)
        total_calls = len(filtered_calls)
        total_tokens = sum(call["total_tokens"] for call in filtered_calls)
        
        # 按提供商分组
        provider_breakdown = {}
        for call in filtered_calls:
            provider = call["provider"]
            if provider not in provider_breakdown:
                provider_breakdown[provider] = {"cost": 0.0, "calls": 0, "tokens": 0}
            provider_breakdown[provider]["cost"] += call["total_cost"]
            provider_breakdown[provider]["calls"] += 1
            provider_breakdown[provider]["tokens"] += call["total_tokens"]
        
        # 按模型分组
        model_breakdown = {}
        for call in filtered_calls:
            model = call["model"]
            if model not in model_breakdown:
                model_breakdown[model] = {"cost": 0.0, "calls": 0, "tokens": 0}
            model_breakdown[model]["cost"] += call["total_cost"]
            model_breakdown[model]["calls"] += 1
            model_breakdown[model]["tokens"] += call["total_tokens"]
        
        return {
            "period": period,
            "start_time": start_time,
            "end_time": end_time,
            "total_cost": round(total_cost, 6),
            "total_calls": total_calls,
            "total_tokens": total_tokens,
            "average_cost_per_call": round(total_cost / total_calls, 6) if total_calls > 0 else 0,
            "average_cost_per_token": round(total_cost / total_tokens, 8) if total_tokens > 0 else 0,
            "provider_breakdown": provider_breakdown,
            "model_breakdown": model_breakdown,
            "cost_trend": self._generate_cost_trend(filtered_calls, period)
        }
    
    def _generate_cost_trend(self, calls: List[Dict], period: str) -> List[Dict]:
        """生成成本趋势数据"""
        if not calls:
            return []
        
        # 简化实现：按小时分组
        hourly_costs = {}
        for call in calls:
            hour_key = call["timestamp"].replace(minute=0, second=0, microsecond=0)
            if hour_key not in hourly_costs:
                hourly_costs[hour_key] = 0.0
            hourly_costs[hour_key] += call["total_cost"]
        
        trend = []
        for hour, cost in sorted(hourly_costs.items()):
            trend.append({
                "timestamp": hour,
                "cost": round(cost, 6)
            })
        
        return trend