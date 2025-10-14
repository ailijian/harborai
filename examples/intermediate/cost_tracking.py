#!/usr/bin/env python3
"""
HarborAI 成本追踪示例

这个示例展示了如何在HarborAI中实现详细的成本追踪和分析，
帮助用户监控和优化AI模型的使用成本。

场景描述:
- 实时成本计算
- 成本预算控制
- 使用量分析
- 成本优化建议

应用价值:
- 控制AI使用成本
- 优化资源配置
- 预算管理
- 成本透明化
"""

import os
import json
import time
import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import statistics
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加本地源码路径
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from harborai import HarborAI
except ImportError:
    print("❌ 无法导入 HarborAI，请检查路径配置")
    exit(1)


class CostCategory(Enum):
    """成本分类"""
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    TRANSLATION = "translation"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    REASONING = "reasoning"
    OTHER = "other"


@dataclass
class ModelPricing:
    """模型定价信息"""
    model_name: str
    provider: str
    input_cost_per_1k: float  # 输入token成本
    output_cost_per_1k: float  # 输出token成本
    currency: str = "RMB"
    last_updated: str = None
    
    def __post_init__(self):
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()


@dataclass
class UsageRecord:
    """使用记录"""
    id: str
    timestamp: str
    model_name: str
    category: CostCategory
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    response_time: float
    user_id: str = "default"
    session_id: str = "default"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CostTracker:
    """成本追踪器"""
    
    def __init__(self, db_path: str = "cost_tracking.db"):
        self.db_path = db_path
        self.pricing_data = self._load_pricing_data()
        self.init_database()
        
        # 预算设置
        self.daily_budget = float(os.getenv("DAILY_BUDGET", "10.0"))
        self.monthly_budget = float(os.getenv("MONTHLY_BUDGET", "300.0"))
        self.alert_threshold = float(os.getenv("ALERT_THRESHOLD", "0.8"))  # 80%预警
    
    def _load_pricing_data(self) -> Dict[str, ModelPricing]:
        """加载模型定价数据"""
        pricing = {
            "deepseek-chat": ModelPricing(
                model_name="deepseek-chat",
                provider="DeepSeek",
                input_cost_per_1k=0.0014,
                output_cost_per_1k=0.0028
            ),
            "deepseek-reasoner": ModelPricing(
                model_name="deepseek-reasoner",
                provider="DeepSeek",
                input_cost_per_1k=0.0055,
                output_cost_per_1k=0.0220
            )
        }
        return pricing
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建使用记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS usage_records (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                category TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                input_cost REAL NOT NULL,
                output_cost REAL NOT NULL,
                total_cost REAL NOT NULL,
                response_time REAL NOT NULL,
                user_id TEXT DEFAULT 'default',
                session_id TEXT DEFAULT 'default',
                metadata TEXT
            )
        ''')
        
        # 创建预算表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS budgets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                period_type TEXT NOT NULL,  -- daily, monthly, yearly
                period_start TEXT NOT NULL,
                budget_amount REAL NOT NULL,
                spent_amount REAL DEFAULT 0,
                created_at TEXT NOT NULL
            )
        ''')
        
        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON usage_records(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_name ON usage_records(model_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON usage_records(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON usage_records(user_id)')
        
        conn.commit()
        conn.close()
    
    def calculate_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> Tuple[float, float, float]:
        """
        计算成本
        
        Args:
            model_name: 模型名称
            input_tokens: 输入token数
            output_tokens: 输出token数
            
        Returns:
            Tuple[float, float, float]: (输入成本, 输出成本, 总成本)
        """
        if model_name not in self.pricing_data:
            # 使用默认定价
            input_cost = (input_tokens / 1000) * 0.002
            output_cost = (output_tokens / 1000) * 0.002
        else:
            pricing = self.pricing_data[model_name]
            input_cost = (input_tokens / 1000) * pricing.input_cost_per_1k
            output_cost = (output_tokens / 1000) * pricing.output_cost_per_1k
        
        total_cost = input_cost + output_cost
        return input_cost, output_cost, total_cost
    
    def record_usage(self, record: UsageRecord):
        """记录使用情况"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO usage_records 
            (id, timestamp, model_name, category, input_tokens, output_tokens, 
             total_tokens, input_cost, output_cost, total_cost, response_time, 
             user_id, session_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.id, record.timestamp, record.model_name, record.category.value,
            record.input_tokens, record.output_tokens, record.total_tokens,
            record.input_cost, record.output_cost, record.total_cost,
            record.response_time, record.user_id, record.session_id,
            json.dumps(record.metadata) if record.metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_usage_stats(self, 
                       start_date: str = None, 
                       end_date: str = None,
                       model_name: str = None,
                       category: CostCategory = None,
                       user_id: str = None) -> Dict[str, Any]:
        """
        获取使用统计
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            model_name: 模型名称
            category: 成本分类
            user_id: 用户ID
            
        Returns:
            Dict: 统计信息
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 构建查询条件
        conditions = []
        params = []
        
        if start_date:
            conditions.append("timestamp >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(end_date + " 23:59:59")
        
        if model_name:
            conditions.append("model_name = ?")
            params.append(model_name)
        
        if category:
            conditions.append("category = ?")
            params.append(category.value)
        
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # 基础统计
        cursor.execute(f'''
            SELECT 
                COUNT(*) as total_requests,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(total_cost) as total_cost,
                AVG(total_cost) as avg_cost_per_request,
                AVG(response_time) as avg_response_time,
                MIN(timestamp) as first_request,
                MAX(timestamp) as last_request
            FROM usage_records
            {where_clause}
        ''', params)
        
        basic_stats = cursor.fetchone()
        
        # 按模型统计
        cursor.execute(f'''
            SELECT 
                model_name,
                COUNT(*) as requests,
                SUM(total_cost) as cost,
                SUM(total_tokens) as tokens
            FROM usage_records
            {where_clause}
            GROUP BY model_name
            ORDER BY cost DESC
        ''', params)
        
        model_stats = cursor.fetchall()
        
        # 按分类统计
        cursor.execute(f'''
            SELECT 
                category,
                COUNT(*) as requests,
                SUM(total_cost) as cost,
                SUM(total_tokens) as tokens
            FROM usage_records
            {where_clause}
            GROUP BY category
            ORDER BY cost DESC
        ''', params)
        
        category_stats = cursor.fetchall()
        
        # 按日期统计
        cursor.execute(f'''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as requests,
                SUM(total_cost) as cost,
                SUM(total_tokens) as tokens
            FROM usage_records
            {where_clause}
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
        ''', params)
        
        daily_stats = cursor.fetchall()
        
        conn.close()
        
        return {
            "basic": {
                "total_requests": basic_stats[0] or 0,
                "total_input_tokens": basic_stats[1] or 0,
                "total_output_tokens": basic_stats[2] or 0,
                "total_tokens": basic_stats[3] or 0,
                "total_cost": basic_stats[4] or 0,
                "avg_cost_per_request": basic_stats[5] or 0,
                "avg_response_time": basic_stats[6] or 0,
                "first_request": basic_stats[7],
                "last_request": basic_stats[8]
            },
            "by_model": [
                {
                    "model_name": row[0],
                    "requests": row[1],
                    "cost": row[2],
                    "tokens": row[3]
                } for row in model_stats
            ],
            "by_category": [
                {
                    "category": row[0],
                    "requests": row[1],
                    "cost": row[2],
                    "tokens": row[3]
                } for row in category_stats
            ],
            "daily": [
                {
                    "date": row[0],
                    "requests": row[1],
                    "cost": row[2],
                    "tokens": row[3]
                } for row in daily_stats
            ]
        }
    
    def check_budget_status(self) -> Dict[str, Any]:
        """检查预算状态"""
        today = datetime.now().date()
        month_start = today.replace(day=1)
        
        # 今日使用情况
        daily_stats = self.get_usage_stats(
            start_date=today.isoformat(),
            end_date=today.isoformat()
        )
        daily_spent = daily_stats["basic"]["total_cost"]
        
        # 本月使用情况
        monthly_stats = self.get_usage_stats(
            start_date=month_start.isoformat(),
            end_date=today.isoformat()
        )
        monthly_spent = monthly_stats["basic"]["total_cost"]
        
        # 计算预算状态
        daily_usage_pct = (daily_spent / self.daily_budget) * 100 if self.daily_budget > 0 else 0
        monthly_usage_pct = (monthly_spent / self.monthly_budget) * 100 if self.monthly_budget > 0 else 0
        
        # 预警状态
        daily_alert = daily_usage_pct >= (self.alert_threshold * 100)
        monthly_alert = monthly_usage_pct >= (self.alert_threshold * 100)
        
        return {
            "daily": {
                "budget": self.daily_budget,
                "spent": daily_spent,
                "remaining": max(0, self.daily_budget - daily_spent),
                "usage_percentage": daily_usage_pct,
                "alert": daily_alert
            },
            "monthly": {
                "budget": self.monthly_budget,
                "spent": monthly_spent,
                "remaining": max(0, self.monthly_budget - monthly_spent),
                "usage_percentage": monthly_usage_pct,
                "alert": monthly_alert
            }
        }
    
    def get_cost_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """获取成本优化建议"""
        suggestions = []
        
        # 获取最近30天的统计
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        stats = self.get_usage_stats(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        
        if not stats["by_model"]:
            return suggestions
        
        # 分析高成本模型
        total_cost = stats["basic"]["total_cost"]
        for model_stat in stats["by_model"]:
            model_cost_pct = (model_stat["cost"] / total_cost) * 100 if total_cost > 0 else 0
            
            if model_cost_pct > 50:  # 如果某个模型占用超过50%的成本
                # 查找更便宜的替代方案
                current_model = model_stat["model_name"]
                cheaper_alternatives = []
                
                if current_model in self.pricing_data:
                    current_pricing = self.pricing_data[current_model]
                    for model_name, pricing in self.pricing_data.items():
                        if (pricing.input_cost_per_1k < current_pricing.input_cost_per_1k and
                            pricing.output_cost_per_1k < current_pricing.output_cost_per_1k):
                            cheaper_alternatives.append({
                                "model": model_name,
                                "savings_input": current_pricing.input_cost_per_1k - pricing.input_cost_per_1k,
                                "savings_output": current_pricing.output_cost_per_1k - pricing.output_cost_per_1k
                            })
                
                if cheaper_alternatives:
                    suggestions.append({
                        "type": "model_substitution",
                        "priority": "high",
                        "current_model": current_model,
                        "cost_percentage": model_cost_pct,
                        "alternatives": cheaper_alternatives[:3],  # 只显示前3个
                        "description": f"{current_model} 占用了 {model_cost_pct:.1f}% 的成本，考虑使用更便宜的替代模型"
                    })
        
        # 分析使用模式
        if len(stats["daily"]) >= 7:
            daily_costs = [day["cost"] for day in stats["daily"][:7]]
            avg_daily_cost = statistics.mean(daily_costs)
            
            if avg_daily_cost > self.daily_budget * 0.8:
                suggestions.append({
                    "type": "budget_optimization",
                    "priority": "medium",
                    "avg_daily_cost": avg_daily_cost,
                    "daily_budget": self.daily_budget,
                    "description": f"平均日成本 ${avg_daily_cost:.4f} 接近预算限制，建议优化使用策略"
                })
        
        # 分析token使用效率
        if stats["basic"]["total_requests"] > 0:
            avg_tokens_per_request = stats["basic"]["total_tokens"] / stats["basic"]["total_requests"]
            avg_cost_per_token = stats["basic"]["total_cost"] / stats["basic"]["total_tokens"] if stats["basic"]["total_tokens"] > 0 else 0
            
            if avg_tokens_per_request > 2000:  # 如果平均每次请求超过2000 tokens
                suggestions.append({
                    "type": "token_optimization",
                    "priority": "low",
                    "avg_tokens_per_request": avg_tokens_per_request,
                    "avg_cost_per_token": avg_cost_per_token,
                    "description": f"平均每次请求使用 {avg_tokens_per_request:.0f} tokens，考虑优化提示词长度"
                })
        
        return suggestions


async def track_api_call(tracker: CostTracker, client: HarborAI, model_name: str, 
                        prompt: str, category: CostCategory, **kwargs) -> Tuple[Any, UsageRecord]:
    """
    追踪API调用并记录成本
    
    Args:
        tracker: 成本追踪器
        client: HarborAI客户端
        model_name: 模型名称
        prompt: 提示词
        category: 成本分类
        **kwargs: 其他参数
        
    Returns:
        Tuple: (API响应, 使用记录)
    """
    start_time = time.time()
    record_id = f"{int(time.time() * 1000)}_{hash(prompt) % 10000}"
    
    try:
        response = await client.chat.completions.acreate(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        
        response_time = time.time() - start_time
        usage = response.usage
        
        # 计算成本
        input_cost, output_cost, total_cost = tracker.calculate_cost(
            model_name, usage.prompt_tokens, usage.completion_tokens
        )
        
        # 创建使用记录
        record = UsageRecord(
            id=record_id,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            category=category,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            response_time=response_time,
            metadata={
                "prompt_length": len(prompt),
                "response_length": len(response.choices[0].message.content),
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens")  # 默认无限制，由模型厂商控制
            }
        )
        
        # 记录到数据库
        tracker.record_usage(record)
        
        return response, record
        
    except Exception as e:
        response_time = time.time() - start_time
        
        # 即使失败也记录（成本为0）
        record = UsageRecord(
            id=record_id,
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            category=category,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_cost=0,
            output_cost=0,
            total_cost=0,
            response_time=response_time,
            metadata={
                "error": str(e),
                "prompt_length": len(prompt)
            }
        )
        
        tracker.record_usage(record)
        raise e


async def cost_comparison_demo(tracker: CostTracker):
    """成本对比演示"""
    print(f"\n💰 成本对比演示")
    print("=" * 60)
    
    # 测试不同模型的成本
    test_prompt = "请解释什么是机器学习，并给出一个简单的例子。"
    models_to_test = ["deepseek-chat", "gpt-3.5-turbo", "gpt-4"]
    
    results = []
    
    for model_name in models_to_test:
        if model_name in tracker.pricing_data:
            pricing = tracker.pricing_data[model_name]
            
            # 估算成本（假设输入500 tokens，输出1000 tokens）
            estimated_input_tokens = 500
            estimated_output_tokens = 1000
            
            input_cost, output_cost, total_cost = tracker.calculate_cost(
                model_name, estimated_input_tokens, estimated_output_tokens
            )
            
            results.append({
                "model": model_name,
                "provider": pricing.provider,
                "input_cost_per_1k": pricing.input_cost_per_1k,
                "output_cost_per_1k": pricing.output_cost_per_1k,
                "estimated_cost": total_cost
            })
    
    # 显示对比结果
    print(f"📊 模型成本对比 (估算: 500输入 + 1000输出 tokens):")
    print("-" * 80)
    print(f"{'模型':<20} {'提供商':<12} {'输入成本/1K':<12} {'输出成本/1K':<12} {'预估总成本':<12}")
    print("-" * 80)
    
    for result in sorted(results, key=lambda x: x["estimated_cost"]):
        print(f"{result['model']:<20} {result['provider']:<12} "
              f"${result['input_cost_per_1k']:<11.4f} ${result['output_cost_per_1k']:<11.4f} "
              f"${result['estimated_cost']:<11.4f}")


async def budget_monitoring_demo(tracker: CostTracker):
    """预算监控演示"""
    print(f"\n📊 预算监控演示")
    print("=" * 60)
    
    # 检查预算状态
    budget_status = tracker.check_budget_status()
    
    print(f"📅 今日预算状态:")
    daily = budget_status["daily"]
    print(f"   预算: ${daily['budget']:.2f}")
    print(f"   已用: ${daily['spent']:.4f}")
    print(f"   剩余: ${daily['remaining']:.4f}")
    print(f"   使用率: {daily['usage_percentage']:.1f}%")
    if daily["alert"]:
        print(f"   ⚠️  预警: 已超过预警阈值!")
    else:
        print(f"   ✅ 状态: 正常")
    
    print(f"\n📅 本月预算状态:")
    monthly = budget_status["monthly"]
    print(f"   预算: ${monthly['budget']:.2f}")
    print(f"   已用: ${monthly['spent']:.4f}")
    print(f"   剩余: ${monthly['remaining']:.4f}")
    print(f"   使用率: {monthly['usage_percentage']:.1f}%")
    if monthly["alert"]:
        print(f"   ⚠️  预警: 已超过预警阈值!")
    else:
        print(f"   ✅ 状态: 正常")


async def usage_analytics_demo(tracker: CostTracker):
    """使用分析演示"""
    print(f"\n📈 使用分析演示")
    print("=" * 60)
    
    # 获取最近7天的统计
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)
    
    stats = tracker.get_usage_stats(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat()
    )
    
    basic = stats["basic"]
    
    print(f"📊 基础统计 (最近7天):")
    print(f"   总请求数: {basic['total_requests']}")
    print(f"   总Token数: {basic['total_tokens']:,}")
    print(f"   总成本: ${basic['total_cost']:.4f}")
    print(f"   平均每次请求成本: ${basic['avg_cost_per_request']:.4f}")
    print(f"   平均响应时间: {basic['avg_response_time']:.2f}秒")
    
    if stats["by_model"]:
        print(f"\n📊 按模型统计:")
        print(f"{'模型':<20} {'请求数':<10} {'成本':<12} {'Token数':<12}")
        print("-" * 60)
        for model in stats["by_model"]:
            print(f"{model['model_name']:<20} {model['requests']:<10} "
                  f"${model['cost']:<11.4f} {model['tokens']:<12,}")
    
    if stats["by_category"]:
        print(f"\n📊 按分类统计:")
        print(f"{'分类':<20} {'请求数':<10} {'成本':<12} {'Token数':<12}")
        print("-" * 60)
        for category in stats["by_category"]:
            print(f"{category['category']:<20} {category['requests']:<10} "
                  f"${category['cost']:<11.4f} {category['tokens']:<12,}")


async def optimization_suggestions_demo(tracker: CostTracker):
    """优化建议演示"""
    print(f"\n💡 成本优化建议")
    print("=" * 60)
    
    suggestions = tracker.get_cost_optimization_suggestions()
    
    if not suggestions:
        print("✅ 暂无优化建议，当前使用模式良好！")
        return
    
    for i, suggestion in enumerate(suggestions, 1):
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        emoji = priority_emoji.get(suggestion["priority"], "ℹ️")
        
        print(f"\n{emoji} 建议 {i}: {suggestion['type']}")
        print(f"   优先级: {suggestion['priority']}")
        print(f"   描述: {suggestion['description']}")
        
        if suggestion["type"] == "model_substitution":
            print(f"   当前模型: {suggestion['current_model']}")
            print(f"   成本占比: {suggestion['cost_percentage']:.1f}%")
            print(f"   替代方案:")
            for alt in suggestion["alternatives"]:
                print(f"     - {alt['model']} (节省输入: ${alt['savings_input']:.4f}/1K, "
                      f"输出: ${alt['savings_output']:.4f}/1K)")
        
        elif suggestion["type"] == "budget_optimization":
            print(f"   平均日成本: ${suggestion['avg_daily_cost']:.4f}")
            print(f"   日预算: ${suggestion['daily_budget']:.2f}")
        
        elif suggestion["type"] == "token_optimization":
            print(f"   平均每次请求Token数: {suggestion['avg_tokens_per_request']:.0f}")
            print(f"   平均每Token成本: ${suggestion['avg_cost_per_token']:.6f}")


async def simulate_usage(tracker: CostTracker):
    """模拟一些使用记录用于演示"""
    print(f"\n🎭 模拟使用记录...")
    
    # 模拟一些API调用
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("❌ 需要DEEPSEEK_API_KEY环境变量来模拟调用")
        return
    
    client = HarborAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    test_scenarios = [
        {"prompt": "你好，请介绍一下你自己。", "category": CostCategory.CHAT},
        {"prompt": "请写一个Python函数来计算斐波那契数列。", "category": CostCategory.CODE_GENERATION},
        {"prompt": "请将'Hello World'翻译成中文。", "category": CostCategory.TRANSLATION},
        {"prompt": "分析一下当前AI技术的发展趋势。", "category": CostCategory.ANALYSIS}
    ]
    
    for scenario in test_scenarios:
        try:
            response, record = await track_api_call(
                tracker, client, "deepseek-chat", 
                scenario["prompt"], scenario["category"],
                max_tokens=500, temperature=0.7
            )
            print(f"✅ 记录了 {scenario['category'].value} 调用，成本: ${record.total_cost:.4f}")
        except Exception as e:
            print(f"❌ 调用失败: {e}")


async def main():
    """主函数"""
    print("="*60)
    print("💰 HarborAI 成本追踪示例")
    print("="*60)
    
    # 初始化成本追踪器
    tracker = CostTracker()
    
    print(f"✅ 成本追踪器初始化完成")
    print(f"📊 已加载 {len(tracker.pricing_data)} 个模型的定价信息")
    print(f"💰 日预算: ${tracker.daily_budget:.2f}")
    print(f"💰 月预算: ${tracker.monthly_budget:.2f}")
    print(f"⚠️  预警阈值: {tracker.alert_threshold:.0%}")
    
    # 1. 成本对比演示
    await cost_comparison_demo(tracker)
    
    # 2. 模拟一些使用记录
    await simulate_usage(tracker)
    
    # 3. 预算监控演示
    await budget_monitoring_demo(tracker)
    
    # 4. 使用分析演示
    await usage_analytics_demo(tracker)
    
    # 5. 优化建议演示
    await optimization_suggestions_demo(tracker)
    
    print(f"\n🎉 成本追踪示例执行完成！")
    print(f"📁 数据库文件: {tracker.db_path}")
    print(f"💡 提示: 可以使用SQLite工具查看详细的使用记录")


if __name__ == "__main__":
    asyncio.run(main())