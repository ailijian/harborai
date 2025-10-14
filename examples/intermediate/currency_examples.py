#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 货币单位配置演示案例

本案例展示如何在 HarborAI 中使用不同的货币单位进行成本追踪，
包括 RMB（默认）、USD、EUR 等多种货币的配置和使用方法。

功能演示：
1. 环境变量配置货币单位
2. 客户端初始化时指定货币
3. 成本追踪对象配置货币
4. 实际 API 调用的成本分析
5. 多货币成本对比

作者: HarborAI Team
创建时间: 2024-12-25
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# 导入 HarborAI 相关模块
try:
    from harborai import HarborAI
    from harborai.core.cost_tracking import CostBreakdown, Budget, BudgetPeriod
    from harborai.core.pricing import PricingCalculator
    from harborai.monitoring.opentelemetry_tracer import OpenTelemetryTracer
    from decimal import Decimal
    import uuid
except ImportError as e:
    print(f"❌ 导入 HarborAI 模块失败: {e}")
    print("请确保已正确安装 HarborAI 并且项目路径配置正确")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('currency_examples.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class CurrencyExamplesDemo:
    """货币单位配置演示类"""
    
    def __init__(self):
        """初始化演示类"""
        self.api_key = self._load_api_key()
        self.base_url = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        self.test_message = "请简单介绍一下人工智能的发展历程，大约100字。"
        
        # 支持的货币列表
        self.supported_currencies = ['RMB', 'CNY', 'USD', 'EUR', 'JPY', 'GBP']
        
        # 测试结果存储
        self.test_results = {}
        
    def _load_api_key(self) -> str:
        """从环境变量加载 API 密钥"""
        # 尝试从 examples/.env 文件加载
        env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
        if os.path.exists(env_file):
            from dotenv import load_dotenv
            load_dotenv(env_file)
            
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("❌ 未找到 DEEPSEEK_API_KEY，请检查 examples/.env 文件")
        
        logger.info(f"✅ 成功加载 API 密钥: {api_key[:10]}...")
        return api_key
    
    def demo_1_default_rmb_currency(self):
        """演示1: 默认 RMB 货币配置"""
        print("\n" + "="*60)
        print("📊 演示1: 默认 RMB 货币配置")
        print("="*60)
        
        try:
            # 创建默认配置的客户端（应该使用 RMB）
            client = HarborAI(api_key=self.api_key)
            
            # 创建成本追踪对象（默认货币）
            breakdown = CostBreakdown()
            budget = Budget(
                id=str(uuid.uuid4()),
                name="默认RMB预算",
                amount=Decimal("50.0"),
                period=BudgetPeriod.DAILY
            )  # 50 RMB 预算
            
            print(f"✅ 默认成本分析货币: {breakdown.currency}")
            print(f"✅ 默认预算货币: {budget.currency}")
            print(f"✅ 预算限额: {budget.amount} {budget.currency}")
            
            # 记录测试结果
            self.test_results['default_rmb'] = {
                'breakdown_currency': breakdown.currency,
                'budget_currency': budget.currency,
                'budget_amount': float(budget.amount),
                'status': 'success'
            }
            
            return client, breakdown, budget
            
        except Exception as e:
            logger.error(f"❌ 演示1失败: {e}")
            self.test_results['default_rmb'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def demo_2_environment_variable_config(self):
        """演示2: 环境变量配置货币"""
        print("\n" + "="*60)
        print("💰 演示2: 环境变量配置货币 (USD)")
        print("="*60)
        
        try:
            # 临时设置环境变量
            original_currency = os.getenv('HARBORAI_DEFAULT_CURRENCY')
            os.environ['HARBORAI_DEFAULT_CURRENCY'] = 'USD'
            
            # 创建客户端（应该使用 USD）
            client = HarborAI(api_key=self.api_key)
            
            # 创建成本追踪对象
            breakdown = CostBreakdown()
            budget = Budget(
                id=str(uuid.uuid4()),
                name="USD预算",
                amount=Decimal("10.0"),
                period=BudgetPeriod.DAILY,
                currency="USD"
            )  # 10 USD 预算
            
            print(f"✅ 环境变量配置货币: {os.getenv('HARBORAI_DEFAULT_CURRENCY')}")
            print(f"✅ 成本分析货币: {breakdown.currency}")
            print(f"✅ 预算货币: {budget.currency}")
            print(f"✅ 预算限额: {budget.amount} {budget.currency}")
            
            # 恢复原始环境变量
            if original_currency:
                os.environ['HARBORAI_DEFAULT_CURRENCY'] = original_currency
            else:
                os.environ.pop('HARBORAI_DEFAULT_CURRENCY', None)
            
            # 记录测试结果
            self.test_results['env_usd'] = {
                'breakdown_currency': breakdown.currency,
                'budget_currency': budget.currency,
                'budget_amount': float(budget.amount),
                'status': 'success'
            }
            
            return client, breakdown, budget
            
        except Exception as e:
            logger.error(f"❌ 演示2失败: {e}")
            self.test_results['env_usd'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def demo_3_client_initialization_config(self):
        """演示3: 客户端初始化时指定货币"""
        print("\n" + "="*60)
        print("🏦 演示3: 客户端初始化时指定货币 (EUR)")
        print("="*60)
        
        try:
            # 在客户端初始化时指定货币
            client = HarborAI(
                api_key=self.api_key,
                default_currency="EUR"
            )
            
            # 创建成本追踪对象
            breakdown = CostBreakdown(currency="EUR")
            budget = Budget(
                id=str(uuid.uuid4()),
                name="EUR预算",
                amount=Decimal("8.0"),
                period=BudgetPeriod.DAILY,
                currency="EUR"
            )  # 8 EUR 预算
            
            print(f"✅ 客户端默认货币: EUR")
            print(f"✅ 成本分析货币: {breakdown.currency}")
            print(f"✅ 预算货币: {budget.currency}")
            print(f"✅ 预算限额: {budget.amount} {budget.currency}")
            
            # 记录测试结果
            self.test_results['client_eur'] = {
                'breakdown_currency': breakdown.currency,
                'budget_currency': budget.currency,
                'budget_amount': float(budget.amount),
                'status': 'success'
            }
            
            return client, breakdown, budget
            
        except Exception as e:
            logger.error(f"❌ 演示3失败: {e}")
            self.test_results['client_eur'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def demo_4_actual_api_call_with_cost_tracking(self):
        """演示4: 实际 API 调用与成本追踪"""
        print("\n" + "="*60)
        print("🚀 演示4: 实际 API 调用与成本追踪")
        print("="*60)
        
        try:
            # 使用默认 RMB 配置
            client = HarborAI(api_key=self.api_key)
            
            print(f"📝 测试消息: {self.test_message}")
            print("🔄 正在调用 deepseek-chat API...")
            
            # 进行实际的 API 调用
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": self.test_message}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            print("✅ API 调用成功!")
            print(f"📄 响应内容: {response.choices[0].message.content[:100]}...")
            
            # 获取成本信息（如果可用）
            if hasattr(response, 'usage'):
                print(f"📊 Token 使用情况:")
                print(f"   - 输入 tokens: {response.usage.prompt_tokens}")
                print(f"   - 输出 tokens: {response.usage.completion_tokens}")
                print(f"   - 总计 tokens: {response.usage.total_tokens}")
            
            # 记录测试结果
            self.test_results['api_call'] = {
                'status': 'success',
                'model': 'deepseek-chat',
                'response_length': len(response.choices[0].message.content),
                'usage': response.usage.__dict__ if hasattr(response, 'usage') else None
            }
            
            return response
            
        except Exception as e:
            logger.error(f"❌ 演示4失败: {e}")
            self.test_results['api_call'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def demo_5_multi_currency_comparison(self):
        """演示5: 多货币成本对比"""
        print("\n" + "="*60)
        print("📈 演示5: 多货币成本对比")
        print("="*60)
        
        try:
            # 模拟相同成本在不同货币下的显示
            base_cost_rmb = 1.50  # 基础成本 1.5 RMB
            
            # 汇率（仅用于演示，实际应用中应从汇率API获取）
            exchange_rates = {
                'RMB': 1.0,
                'CNY': 1.0,
                'USD': 0.14,  # 1 RMB ≈ 0.14 USD
                'EUR': 0.13,  # 1 RMB ≈ 0.13 EUR
                'JPY': 20.0,  # 1 RMB ≈ 20 JPY
                'GBP': 0.11   # 1 RMB ≈ 0.11 GBP
            }
            
            print("💱 相同成本在不同货币下的显示:")
            print("-" * 40)
            
            currency_costs = {}
            for currency, rate in exchange_rates.items():
                cost = base_cost_rmb * rate
                breakdown = CostBreakdown(currency=currency)
                
                print(f"   {currency}: {cost:.4f} {currency}")
                currency_costs[currency] = cost
            
            # 记录测试结果
            self.test_results['multi_currency'] = {
                'base_cost_rmb': base_cost_rmb,
                'currency_costs': currency_costs,
                'status': 'success'
            }
            
            return currency_costs
            
        except Exception as e:
            logger.error(f"❌ 演示5失败: {e}")
            self.test_results['multi_currency'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def demo_6_cost_tracking_objects(self):
        """演示6: 成本追踪对象的货币配置"""
        print("\n" + "="*60)
        print("🔧 演示6: 成本追踪对象的货币配置")
        print("="*60)
        
        try:
            print("📋 创建不同货币的成本追踪对象:")
            print("-" * 40)
            
            # 创建不同货币的成本分析对象
            rmb_breakdown = CostBreakdown(currency="RMB")
            usd_breakdown = CostBreakdown(currency="USD")
            eur_breakdown = CostBreakdown(currency="EUR")
            
            print(f"   RMB 成本分析: {rmb_breakdown.currency}")
            print(f"   USD 成本分析: {usd_breakdown.currency}")
            print(f"   EUR 成本分析: {eur_breakdown.currency}")
            
            # 创建不同货币的预算对象
            rmb_budget = Budget(
                id=str(uuid.uuid4()),
                name="RMB预算",
                amount=Decimal("100.0"),
                period=BudgetPeriod.DAILY,
                currency="RMB"
            )
            usd_budget = Budget(
                id=str(uuid.uuid4()),
                name="USD预算",
                amount=Decimal("15.0"),
                period=BudgetPeriod.DAILY,
                currency="USD"
            )
            eur_budget = Budget(
                id=str(uuid.uuid4()),
                name="EUR预算",
                amount=Decimal("13.0"),
                period=BudgetPeriod.DAILY,
                currency="EUR"
            )
            
            print(f"\n💰 不同货币的预算配置:")
            print(f"   RMB 预算: {rmb_budget.amount} {rmb_budget.currency}")
            print(f"   USD 预算: {usd_budget.amount} {usd_budget.currency}")
            print(f"   EUR 预算: {eur_budget.amount} {eur_budget.currency}")
            
            # 记录测试结果
            self.test_results['cost_objects'] = {
                'breakdowns': {
                    'rmb': rmb_breakdown.currency,
                    'usd': usd_breakdown.currency,
                    'eur': eur_breakdown.currency
                },
                'budgets': {
                    'rmb': {'amount': float(rmb_budget.amount), 'currency': rmb_budget.currency},
                    'usd': {'amount': float(usd_budget.amount), 'currency': usd_budget.currency},
                    'eur': {'amount': float(eur_budget.amount), 'currency': eur_budget.currency}
                },
                'status': 'success'
            }
            
            return {
                'breakdowns': [rmb_breakdown, usd_breakdown, eur_breakdown],
                'budgets': [rmb_budget, usd_budget, eur_budget]
            }
            
        except Exception as e:
            logger.error(f"❌ 演示6失败: {e}")
            self.test_results['cost_objects'] = {'status': 'failed', 'error': str(e)}
            raise
    
    def print_test_summary(self):
        """打印测试结果摘要"""
        print("\n" + "="*60)
        print("📊 测试结果摘要")
        print("="*60)
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if result.get('status') == 'success')
        failed_tests = total_tests - successful_tests
        
        print(f"总测试数: {total_tests}")
        print(f"成功: {successful_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"成功率: {(successful_tests/total_tests)*100:.1f}%")
        
        print("\n📋 详细结果:")
        print("-" * 40)
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result.get('status') == 'success' else "❌"
            print(f"{status_icon} {test_name}: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'failed' and 'error' in result:
                print(f"   错误: {result['error']}")
        
        # 保存测试结果到文件
        import json
        results_file = 'currency_examples_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_tests': total_tests,
                    'successful_tests': successful_tests,
                    'failed_tests': failed_tests,
                    'success_rate': (successful_tests/total_tests)*100
                },
                'detailed_results': self.test_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细测试结果已保存到: {results_file}")

def main():
    """主函数"""
    print("🌟 HarborAI 货币单位配置演示案例")
    print("=" * 60)
    print("本案例将演示如何在 HarborAI 中配置和使用不同的货币单位")
    print("包括 RMB（默认）、USD、EUR 等多种货币的配置方法")
    print("=" * 60)
    
    try:
        # 创建演示实例
        demo = CurrencyExamplesDemo()
        
        # 执行各个演示
        print("\n🚀 开始执行演示...")
        
        # 演示1: 默认 RMB 货币配置
        demo.demo_1_default_rmb_currency()
        
        # 演示2: 环境变量配置货币
        demo.demo_2_environment_variable_config()
        
        # 演示3: 客户端初始化时指定货币
        demo.demo_3_client_initialization_config()
        
        # 演示4: 实际 API 调用与成本追踪
        demo.demo_4_actual_api_call_with_cost_tracking()
        
        # 演示5: 多货币成本对比
        demo.demo_5_multi_currency_comparison()
        
        # 演示6: 成本追踪对象的货币配置
        demo.demo_6_cost_tracking_objects()
        
        # 打印测试结果摘要
        demo.print_test_summary()
        
        print("\n🎉 所有演示完成!")
        print("✨ HarborAI 支持灵活的货币配置，满足不同地区和场景的需求")
        
    except Exception as e:
        logger.error(f"❌ 演示执行失败: {e}")
        print(f"\n❌ 演示执行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # 运行主函数
    main()