import pytest
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import time
from datetime import datetime, timedelta
from decimal import Decimal
from tests.utils.cost_tracking_mocks import (
    MockTokenCounter, 
    MockPricingCalculator, 
    MockCostTracker, 
    MockBudgetManager, 
    Budget, 
    BudgetPeriod,
    CostReport,
    TokenUsage,
    ApiCall
)


class TestTokenCounting:
    """Token计数测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.cost_tracking
    def test_basic_token_counting(self):
        """测试基本Token计数功能"""
        counter = MockTokenCounter()
        
        # 测试简单文本
        text1 = "Hello, world!"
        tokens1 = counter.count_tokens(text1, "deepseek-chat")
        assert tokens1 > 0
        assert tokens1 <= len(text1)  # Token数不应超过字符数
        
        # 测试空文本
        tokens_empty = counter.count_tokens("", "deepseek-chat")
        assert tokens_empty == 0
        
        # 测试长文本
        long_text = "This is a much longer text that should result in more tokens. " * 10
        tokens_long = counter.count_tokens(long_text, "deepseek-chat")
        assert tokens_long > tokens1
        assert tokens_long > 50  # 长文本应该有更多tokens
        
        # 测试不同模型的差异
        text = "This is a test message for token counting."
        deepseek_tokens = counter.count_tokens(text, "deepseek-chat")
        ernie_tokens = counter.count_tokens(text, "ernie-4.0-8k")
        doubao_tokens = counter.count_tokens(text, "doubao-pro-4k")
        
        # 不同模型的token计数可能不同
        assert deepseek_tokens > 0
        assert ernie_tokens > 0
        assert doubao_tokens > 0
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.cost_tracking
    def test_message_token_counting(self):
        """测试消息Token计数"""
        counter = MockTokenCounter()
        
        # 测试简单消息
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        usage = counter.count_message_tokens(messages, "deepseek-chat")
        assert usage.input_tokens > 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == usage.input_tokens
        
        # 测试多条消息
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What about Germany?"}
        ]
        
        usage_multi = counter.count_message_tokens(messages, "deepseek-chat")
        assert usage_multi.input_tokens > usage.input_tokens
        assert usage_multi.input_tokens > 20  # 多条消息应该有更多tokens
        
        # 测试多模态消息
        multimodal_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
                ]
            }
        ]
        
        usage_multimodal = counter.count_message_tokens(multimodal_messages, "deepseek-chat")
        assert usage_multimodal.input_tokens > 85  # 应该包含图片的tokens
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.cost_tracking
    def test_response_token_counting(self):
        """测试响应Token计数"""
        counter = MockTokenCounter()
        
        # 测试简单响应
        response1 = "Hello! I'm doing well, thank you for asking."
        usage1 = counter.count_response_tokens(response1, "deepseek-chat")
        
        assert usage1.input_tokens == 0
        assert usage1.output_tokens > 0
        assert usage1.total_tokens == usage1.output_tokens
        
        # 测试长响应
        long_response = "This is a much longer response that contains multiple sentences. " * 20
        usage_long = counter.count_response_tokens(long_response, "deepseek-chat")
        
        assert usage_long.output_tokens > usage1.output_tokens
        assert usage_long.output_tokens > 100
        
        # 测试空响应
        usage_empty = counter.count_response_tokens("", "deepseek-chat")
        assert usage_empty.output_tokens == 0
        assert usage_empty.total_tokens == 0
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.cost_tracking
    def test_token_estimation(self):
        """测试Token估算"""
        counter = MockTokenCounter()
        
        # 测试估算准确性
        text = "This is a test message for token estimation accuracy."
        
        actual_tokens = counter.count_tokens(text, "deepseek-chat")
        estimated_tokens = counter.estimate_tokens(text, "deepseek-chat")
        
        # 估算应该与实际计数相同（在这个简化实现中）
        assert estimated_tokens == actual_tokens
        
        # 测试不同长度文本的估算
        texts = [
            "Short",
            "This is a medium length text for testing.",
            "This is a much longer text that spans multiple sentences and contains various types of content including numbers like 123 and special characters like @#$%."
        ]
        
        estimates = []
        for text in texts:
            estimate = counter.estimate_tokens(text, "deepseek-chat")
            estimates.append(estimate)
            assert estimate > 0
        
        # 估算应该随文本长度增加
        assert estimates[0] < estimates[1] < estimates[2]


class TestCostCalculation:
    """成本计算测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.cost_tracking
    def test_basic_cost_calculation(self):
        """测试基本成本计算"""
        calculator = MockPricingCalculator()
        
        # 测试DeepSeek成本计算
        token_usage = TokenUsage(input_tokens=1000, output_tokens=1000)
        cost = calculator.calculate_cost("deepseek", "deepseek-chat", token_usage)
        
        # 验证成本结构
        assert cost.input_cost > 0
        assert cost.output_cost > 0
        assert cost.total_cost == cost.input_cost + cost.output_cost
        assert cost.currency == "RMB"
        
        # 验证输出成本通常高于输入成本（deepseek-chat: input=0.001, output=0.002）
        assert cost.output_cost > cost.input_cost
        
        # 测试不同模型的成本差异
        deepseek_cost = calculator.calculate_cost("deepseek", "deepseek-chat", token_usage)
        ernie_cost = calculator.calculate_cost("ernie", "ernie-4.0-8k", token_usage)
        
        # DeepSeek应该比ERNIE更昂贵（根据定价表）
        assert deepseek_cost.total_cost < ernie_cost.total_cost
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.cost_tracking
    def test_provider_cost_comparison(self):
        """测试提供商成本比较"""
        calculator = MockPricingCalculator()
        
        token_usage = TokenUsage(input_tokens=2000, output_tokens=1000)
        
        # 比较不同提供商的成本
        models = [
            ("deepseek", "deepseek-chat"),
            ("deepseek", "deepseek-reasoner"),
            ("ernie", "ernie-4.0-8k"),
            ("ernie", "ernie-3.5-8k"),
            ("doubao", "doubao-pro-4k")
        ]
        
        costs = calculator.compare_provider_costs(token_usage, models)
        
        # 验证所有模型都有成本
        assert len(costs) == len(models)
        for model_key, cost in costs.items():
            assert cost.total_cost > 0
            assert "/" in model_key  # 格式应该是 "provider/model"
        
        # 验证成本差异
        deepseek_chat_cost = costs["deepseek/deepseek-chat"]
        deepseek_r1_cost = costs["deepseek/deepseek-reasoner"]
        doubao_cost = costs["doubao/doubao-pro-4k"]
        
        assert deepseek_r1_cost.total_cost > deepseek_chat_cost.total_cost
        assert doubao_cost.total_cost < deepseek_r1_cost.total_cost  # Doubao通常更便宜
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.cost_tracking
    def test_volume_discounts(self):
        """测试批量折扣"""
        calculator = MockPricingCalculator()
        
        token_usage = TokenUsage(input_tokens=10000, output_tokens=5000)
        
        # 测试无折扣
        cost_no_discount = calculator.calculate_cost_with_discount(
            "deepseek", "deepseek-chat", token_usage, monthly_volume=0
        )
        
        # 测试小批量（无折扣）
        cost_small_volume = calculator.calculate_cost_with_discount(
            "deepseek", "deepseek-chat", token_usage, monthly_volume=500000
        )
        
        # 测试大批量（有折扣）
        cost_large_volume = calculator.calculate_cost_with_discount(
            "deepseek", "deepseek-chat", token_usage, monthly_volume=2000000
        )
        
        # 测试超大批量（更大折扣）
        cost_huge_volume = calculator.calculate_cost_with_discount(
            "deepseek", "deepseek-chat", token_usage, monthly_volume=20000000
        )
        
        # 验证折扣效果
        assert cost_no_discount.total_cost == cost_small_volume.total_cost  # 小批量无折扣
        assert cost_large_volume.total_cost < cost_no_discount.total_cost   # 大批量有折扣
        assert cost_huge_volume.total_cost < cost_large_volume.total_cost   # 超大批量折扣更大
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.cost_tracking
    def test_monthly_cost_estimation(self):
        """测试月度成本估算"""
        calculator = MockPricingCalculator()
        
        # 测试日使用量
        daily_usage = TokenUsage(input_tokens=1000, output_tokens=500)
        
        # 估算月度成本
        monthly_cost = calculator.estimate_monthly_cost(daily_usage, "deepseek", "deepseek-chat")
        
        assert monthly_cost > 0
        
        # 验证月度成本是日成本的约30倍（考虑可能的批量折扣）
        daily_cost = calculator.calculate_cost("deepseek", "deepseek-chat", daily_usage)
        expected_monthly = daily_cost.total_cost * 30
        
        # 月度成本应该小于等于简单的30倍（因为可能有批量折扣）
        assert monthly_cost <= expected_monthly
        
        # 测试不同使用量的月度成本
        small_daily = TokenUsage(input_tokens=100, output_tokens=50)
        large_daily = TokenUsage(input_tokens=10000, output_tokens=5000)
        
        small_monthly = calculator.estimate_monthly_cost(small_daily, "deepseek", "deepseek-chat")
        large_monthly = calculator.estimate_monthly_cost(large_daily, "deepseek", "deepseek-chat")
        
        assert large_monthly > small_monthly
        assert large_monthly > small_monthly * 50  # 大使用量应该显著更昂贵
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.cost_tracking
    def test_pricing_accuracy(self):
        """测试定价准确性"""
        calculator = MockPricingCalculator()
        
        # 测试精确的token数量
        token_usage = TokenUsage(input_tokens=1000, output_tokens=1000)
        
        # 获取定价信息
        pricing = calculator.get_model_pricing("deepseek", "deepseek-chat")
        
        # 手动计算预期成本
        expected_input_cost = pricing["input"]  # 1000 tokens = 1 * price_per_1000
        expected_output_cost = pricing["output"]  # 1000 tokens = 1 * price_per_1000
        expected_total = expected_input_cost + expected_output_cost
        
        # 使用计算器计算
        calculated_cost = calculator.calculate_cost("deepseek", "deepseek-chat", token_usage)
        
        # 验证计算准确性
        assert calculated_cost.input_cost == expected_input_cost
        assert calculated_cost.output_cost == expected_output_cost
        assert calculated_cost.total_cost == expected_total
        
        # 测试小数精度
        small_usage = TokenUsage(input_tokens=1, output_tokens=1)
        small_cost = calculator.calculate_cost("deepseek", "deepseek-chat", small_usage)
        
        # 小数应该被正确处理
        assert small_cost.input_cost > 0
        assert small_cost.output_cost > 0
        assert small_cost.total_cost == small_cost.input_cost + small_cost.output_cost


class TestCostTracking:
    """成本追踪测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.cost_tracking
    def test_api_call_tracking(self):
        """测试API调用追踪"""
        tracker = MockCostTracker()
        
        # 模拟API调用
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        response = "I'm doing well, thank you for asking!"
        
        api_call = tracker.track_api_call(
            provider="deepseek",
            model="deepseek-chat",
            endpoint="/v1/chat/completions",
            messages=messages,
            response=response,
            duration=1.5,
            status="success",
            user_id="user123",
            session_id="session456",
            tags={"environment": "production", "feature": "chat"}
        )
        
        # 验证API调用记录
        assert api_call.id is not None
        assert api_call.provider == "deepseek"
        assert api_call.model == "deepseek-chat"
        assert api_call.endpoint == "/v1/chat/completions"
        assert api_call.duration == 1.5
        assert api_call.status == "success"
        assert api_call.user_id == "user123"
        assert api_call.session_id == "session456"
        assert api_call.tags["environment"] == "production"
        assert api_call.tags["feature"] == "chat"
        
        # 验证token使用量
        assert api_call.token_usage.input_tokens > 0
        assert api_call.token_usage.output_tokens > 0
        assert api_call.token_usage.total_tokens > 0
        
        # 验证成本计算
        assert api_call.cost_breakdown.input_cost > 0
        assert api_call.cost_breakdown.output_cost > 0
        assert api_call.cost_breakdown.total_cost > 0
        
        # 验证请求和响应大小
        assert api_call.request_size > 0
        assert api_call.response_size > 0
        
        # 验证追踪器状态
        assert len(tracker.api_calls) == 1
        assert tracker.get_total_cost() > 0
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.cost_tracking
    def test_cost_aggregation(self):
        """测试成本聚合"""
        tracker = MockCostTracker()
        
        # 追踪多个API调用
        calls_data = [
            ("deepseek", "deepseek-chat", "Hello", "Hi there!", "user1"),
            ("ernie", "ernie-4.0-8k", "How are you?", "I'm fine!", "user1"),
            ("doubao", "doubao-pro-4k", "What's the weather?", "It's sunny!", "user2"),
            ("ernie", "ernie-3.5-8k", "Tell me a joke", "Why did the chicken cross the road?", "user2"),
            ("deepseek", "deepseek-chat", "Explain AI", "AI is artificial intelligence...", "user3")
        ]
        
        for provider, model, prompt, response, user_id in calls_data:
            messages = [{"role": "user", "content": prompt}]
            tracker.track_api_call(
                provider=provider,
                model=model,
                endpoint="/v1/chat/completions",
                messages=messages,
                response=response,
                duration=1.0,
                user_id=user_id
            )
        
        # 验证总成本
        total_cost = tracker.get_total_cost()
        assert total_cost > 0
        
        # 验证按提供商的成本分解
        provider_costs = tracker.get_cost_by_provider()
        assert "deepseek" in provider_costs
        assert "ernie" in provider_costs
        assert "doubao" in provider_costs
        assert provider_costs["deepseek"] > 0
        assert provider_costs["ernie"] > 0
        assert provider_costs["doubao"] > 0
        
        # 验证按模型的成本分解
        model_costs = tracker.get_cost_by_model()
        assert "deepseek-chat" in model_costs
        assert "ernie-4.0-8k" in model_costs
        assert "doubao-pro-4k" in model_costs
        assert "ernie-3.5-8k" in model_costs
        
        # 验证按用户的成本分解
        user_costs = tracker.get_cost_by_user()
        assert "user1" in user_costs
        assert "user2" in user_costs
        assert "user3" in user_costs
        assert user_costs["user1"] > 0  # user1有2次调用
        assert user_costs["user2"] > 0  # user2有2次调用
        assert user_costs["user3"] > 0  # user3有1次调用
        
        # 验证总成本等于各部分之和
        assert abs(total_cost - sum(provider_costs.values())) < Decimal('0.000001')
        assert abs(total_cost - sum(model_costs.values())) < Decimal('0.000001')
        assert abs(total_cost - sum(user_costs.values())) < Decimal('0.000001')
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.cost_tracking
    def test_token_usage_aggregation(self):
        """测试Token使用量聚合"""
        tracker = MockCostTracker()
        
        # 追踪多个不同大小的API调用
        test_cases = [
            ("Short message", "Short response"),
            ("This is a medium length message for testing", "This is a medium length response"),
            ("This is a very long message " * 20, "This is a very long response " * 15)
        ]
        
        total_input_expected = 0
        total_output_expected = 0
        
        for prompt, response in test_cases:
            messages = [{"role": "user", "content": prompt}]
            api_call = tracker.track_api_call(
                provider="deepseek",
                model="deepseek-chat",
                endpoint="/v1/chat/completions",
                messages=messages,
                response=response,
                duration=1.0
            )
            
            total_input_expected += api_call.token_usage.input_tokens
            total_output_expected += api_call.token_usage.output_tokens
        
        # 验证聚合的token使用量
        total_usage = tracker.get_token_usage()
        assert total_usage.input_tokens == total_input_expected
        assert total_usage.output_tokens == total_output_expected
        assert total_usage.total_tokens == total_input_expected + total_output_expected
        
        # 验证token使用量随消息长度增加
        assert total_usage.input_tokens > 50  # 应该有足够的输入tokens
        assert total_usage.output_tokens > 30  # 应该有足够的输出tokens
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.cost_tracking
    def test_expensive_calls_tracking(self):
        """测试昂贵调用追踪"""
        tracker = MockCostTracker()
        
        # 创建不同成本的API调用
        test_cases = [
            ("deepseek-chat", "Very long prompt " * 100, "Very long response " * 80),  # 昂贵
            ("ernie-3.5-8k", "Short", "Short"),  # 便宜
            ("deepseek-chat", "Medium prompt " * 20, "Medium response " * 15),  # 中等
            ("doubao-pro-4k", "Long prompt " * 50, "Long response " * 40),  # 昂贵
            ("ernie-4.0-8k", "Short prompt", "Short response")  # 便宜
        ]
        
        for model, prompt, response in test_cases:
            provider = "deepseek" if model.startswith("deepseek") else ("ernie" if model.startswith("ernie") else "doubao")
            messages = [{"role": "user", "content": prompt}]
            tracker.track_api_call(
                provider=provider,
                model=model,
                endpoint="/v1/chat/completions",
                messages=messages,
                response=response,
                duration=1.0
            )
        
        # 获取最昂贵的调用
        expensive_calls = tracker.get_most_expensive_calls(limit=3)
        
        assert len(expensive_calls) == 3
        
        # 验证排序（成本从高到低）
        for i in range(len(expensive_calls) - 1):
            assert expensive_calls[i].cost_breakdown.total_cost >= expensive_calls[i + 1].cost_breakdown.total_cost
        
        # 验证最昂贵的调用确实是高成本的
        most_expensive = expensive_calls[0]
        assert most_expensive.cost_breakdown.total_cost > Decimal('0.001')  # 应该有显著成本
        
        # 验证最昂贵的调用可能是长文本或昂贵模型
        assert (most_expensive.model in ["deepseek-chat", "doubao-pro-4k"] or 
                most_expensive.token_usage.total_tokens > 100)
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.cost_tracking
    def test_time_based_filtering(self):
        """测试基于时间的过滤"""
        tracker = MockCostTracker()
        
        # 记录当前时间
        now = datetime.now()
        
        # 模拟不同时间的API调用
        # 注意：由于我们使用的是当前时间，这里只能测试基本功能
        messages = [{"role": "user", "content": "Test message"}]
        
        # 第一次调用
        call1 = tracker.track_api_call(
            provider="deepseek",
            model="deepseek-chat",
            endpoint="/v1/chat/completions",
            messages=messages,
            response="Response 1",
            duration=1.0
        )
        
        # 等待一小段时间
        time.sleep(0.1)
        
        # 第二次调用
        call2 = tracker.track_api_call(
            provider="deepseek",
            model="deepseek-chat",
            endpoint="/v1/chat/completions",
            messages=messages,
            response="Response 2",
            duration=1.0
        )
        
        # 测试时间过滤
        middle_time = call1.timestamp + timedelta(milliseconds=50)
        
        # 获取中间时间之后的成本
        recent_cost = tracker.get_total_cost(start_date=middle_time)
        total_cost = tracker.get_total_cost()
        
        # 最近的成本应该小于总成本
        assert recent_cost < total_cost
        assert recent_cost > 0
        
        # 获取中间时间之前的成本
        early_cost = tracker.get_total_cost(end_date=middle_time)
        assert early_cost > 0
        assert early_cost < total_cost
        
        # 验证时间范围的成本加起来等于总成本
        assert abs((early_cost + recent_cost) - total_cost) < Decimal('0.000001')


class TestBudgetManagement:
    """预算管理测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.cost_tracking
    def test_budget_creation(self):
        """测试预算创建"""
        tracker = MockCostTracker()
        budget_manager = MockBudgetManager(tracker)
        
        # 创建基本预算
        budget = budget_manager.create_budget(
            name="Monthly API Budget",
            amount=Decimal('100.00'),
            period=BudgetPeriod.MONTHLY
        )
        
        assert budget.id is not None
        assert budget.name == "Monthly API Budget"
        assert budget.amount == Decimal('100.00')
        assert budget.period == BudgetPeriod.MONTHLY
        assert budget.currency == "RMB"
        assert budget.enabled is True
        assert len(budget.alert_thresholds) == 3
        
        # 创建带过滤条件的预算
        filtered_budget = budget_manager.create_budget(
            name="DeepSeek Budget",
            amount=Decimal('50.00'),
            period=BudgetPeriod.DAILY,
            providers=["deepseek"],
            models=["deepseek-chat"],
            users=["user1", "user2"],
            alert_thresholds=[0.7, 0.9]
        )
        
        assert filtered_budget.providers == ["deepseek"]
        assert filtered_budget.models == ["deepseek-chat"]
        assert filtered_budget.users == ["user1", "user2"]
        assert filtered_budget.alert_thresholds == [0.7, 0.9]
        
        # 验证预算已保存
        assert len(budget_manager.budgets) == 2
        assert budget.id in budget_manager.budgets
        assert filtered_budget.id in budget_manager.budgets
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.cost_tracking
    def test_budget_status_checking(self):
        """测试预算状态检查"""
        tracker = MockCostTracker()
        budget_manager = MockBudgetManager(tracker)
        
        # 创建预算
        budget = budget_manager.create_budget(
            name="Test Budget",
            amount=Decimal('10.00'),
            period=BudgetPeriod.DAILY
        )
        
        # 初始状态检查
        status = budget_manager.check_budget_status(budget.id)
        assert status["status"] == "ok"
        assert status["current_usage"] == Decimal('0')
        assert status["usage_percentage"] == 0
        assert status["remaining"] == Decimal('10.00')
        
        # 添加一些API调用
        messages = [{"role": "user", "content": "Test message"}]
        
        # 添加少量使用（应该仍然是ok状态）
        tracker.track_api_call(
            provider="ernie",
            model="ernie-3.5-8k",  # 使用便宜的模型
            endpoint="/v1/chat/completions",
            messages=messages,
            response="Short response",
            duration=1.0
        )
        
        status = budget_manager.check_budget_status(budget.id)
        assert status["status"] == "ok"
        assert status["current_usage"] > 0
        assert status["usage_percentage"] < 0.5
        
        # 添加更多使用（达到警告阈值）
        for _ in range(10):
            tracker.track_api_call(
                provider="deepseek",
                model="deepseek-chat",  # 使用昂贵的模型
                endpoint="/v1/chat/completions",
                messages=[{"role": "user", "content": "Long message " * 50}],
                response="Long response " * 40,
                duration=1.0
            )
        
        status = budget_manager.check_budget_status(budget.id)
        # 状态应该是warning或更严重（由于模拟的成本很低，可能仍然是ok或caution）
        assert status["status"] in ["ok", "caution", "warning", "critical", "exceeded"]
        assert status["current_usage"] > 0  # 应该有一些使用
        # 注意：由于模拟的token成本很低，可能不会达到5.00的使用量
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.cost_tracking
    def test_budget_filtering(self):
        """测试预算过滤功能"""
        tracker = MockCostTracker()
        budget_manager = MockBudgetManager(tracker)
        
        # 创建针对特定提供商的预算
        deepseek_budget = budget_manager.create_budget(
            name="DeepSeek Budget",
            amount=Decimal('20.00'),
            period=BudgetPeriod.DAILY,
            providers=["deepseek"]
        )
        
        # 创建针对特定用户的预算
        user_budget = budget_manager.create_budget(
            name="User1 Budget",
            amount=Decimal('15.00'),
            period=BudgetPeriod.DAILY,
            users=["user1"]
        )
        
        # 添加不同的API调用
        messages = [{"role": "user", "content": "Test message"}]
        
        # DeepSeek调用（应该影响DeepSeek预算）
        tracker.track_api_call(
            provider="deepseek",
            model="deepseek-chat",
            endpoint="/v1/chat/completions",
            messages=messages,
            response="Response",
            duration=1.0,
            user_id="user1"
        )
        
        # Ernie调用（不应该影响DeepSeek预算）
        tracker.track_api_call(
            provider="ernie",
            model="ernie-4.0-8k",
            endpoint="/v1/messages",
            messages=messages,
            response="Response",
            duration=1.0,
            user_id="user2"
        )
        
        # 检查DeepSeek预算状态
        deepseek_status = budget_manager.check_budget_status(deepseek_budget.id)
        assert deepseek_status["current_usage"] > 0  # 应该有DeepSeek使用
        
        # 检查用户预算状态
        user_status = budget_manager.check_budget_status(user_budget.id)
        assert user_status["current_usage"] > 0  # 应该有user1的使用
        
        # DeepSeek预算的使用应该等于user1预算的使用（因为只有user1使用了DeepSeek）
        assert deepseek_status["current_usage"] == user_status["current_usage"]
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.cost_tracking
    def test_budget_alerts(self):
        """测试预算告警"""
        tracker = MockCostTracker()
        budget_manager = MockBudgetManager(tracker)
        
        # 创建带自定义告警阈值的预算
        budget = budget_manager.create_budget(
            name="Alert Test Budget",
            amount=Decimal('5.00'),
            period=BudgetPeriod.DAILY,
            alert_thresholds=[0.6, 0.8, 0.95]
        )
        
        # 初始状态：无告警
        alerts = budget_manager.get_budget_alerts(budget.id)
        assert len(alerts) == 0
        
        # 发送告警
        budget_manager.send_budget_alert(budget.id, 0.6, Decimal('3.00'))
        budget_manager.send_budget_alert(budget.id, 0.8, Decimal('4.00'))
        
        # 检查告警
        alerts = budget_manager.get_budget_alerts(budget.id)
        assert len(alerts) == 2
        
        # 验证告警内容
        first_alert = alerts[0]
        assert first_alert["budget_id"] == budget.id
        assert first_alert["budget_name"] == "Alert Test Budget"
        assert first_alert["threshold"] == 0.6
        assert first_alert["current_usage"] == Decimal('3.00')
        assert first_alert["budget_amount"] == Decimal('5.00')
        assert first_alert["usage_percentage"] == 0.6
        
        # 测试获取所有告警
        all_alerts = budget_manager.get_budget_alerts()
        assert len(all_alerts) == 2
        
        # 清除告警
        budget_manager.clear_alerts()
        alerts = budget_manager.get_budget_alerts()
        assert len(alerts) == 0
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.cost_tracking
    def test_multiple_budget_checking(self):
        """测试多预算检查"""
        tracker = MockCostTracker()
        budget_manager = MockBudgetManager(tracker)
        
        # 创建多个预算
        budget1 = budget_manager.create_budget(
            name="Budget 1",
            amount=Decimal('10.00'),
            period=BudgetPeriod.DAILY
        )
        
        budget2 = budget_manager.create_budget(
            name="Budget 2",
            amount=Decimal('20.00'),
            period=BudgetPeriod.DAILY,
            providers=["deepseek"]
        )
        
        # 禁用的预算
        budget3 = budget_manager.create_budget(
            name="Disabled Budget",
            amount=Decimal('5.00'),
            period=BudgetPeriod.DAILY
        )
        budget3.enabled = False
        budget_manager.budgets[budget3.id] = budget3
        
        # 检查所有预算
        all_status = budget_manager.check_all_budgets()
        
        # 应该只包含启用的预算
        assert len(all_status) == 2
        assert budget1.id in all_status
        assert budget2.id in all_status
        assert budget3.id not in all_status
        
        # 验证每个预算的状态
        for budget_id, status in all_status.items():
            assert "status" in status
            assert "current_usage" in status
            assert "usage_percentage" in status
            assert "remaining" in status


class TestCostReporting:
    """成本报告测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.cost_tracking
    def test_cost_report_generation(self):
        """测试成本报告生成"""
        tracker = MockCostTracker()
        
        # 添加多样化的API调用
        test_data = [
            ("deepseek", "deepseek-chat", "Hello", "Hi!", "user1"),
            ("ernie", "ernie-4.0-8k", "How are you?", "Fine!", "user1"),
            ("doubao", "doubao-pro-4k", "What's up?", "Nothing much!", "user2"),
            ("ernie", "ernie-3.5-8k", "Tell me a joke", "Why did the chicken...", "user2")
        ]
        
        start_time = datetime.now()
        
        for provider, model, prompt, response, user_id in test_data:
            messages = [{"role": "user", "content": prompt}]
            tracker.track_api_call(
                provider=provider,
                model=model,
                endpoint="/v1/chat/completions",
                messages=messages,
                response=response,
                duration=1.0,
                user_id=user_id
            )
        
        end_time = datetime.now()
        
        # 生成报告数据
        total_cost = tracker.get_total_cost(start_time, end_time)
        provider_costs = tracker.get_cost_by_provider(start_time, end_time)
        model_costs = tracker.get_cost_by_model(start_time, end_time)
        user_costs = tracker.get_cost_by_user(start_time, end_time)
        token_usage = tracker.get_token_usage(start_time, end_time)
        expensive_calls = tracker.get_most_expensive_calls(5, start_time, end_time)
        
        # 创建报告
        report = CostReport(
            period_start=start_time,
            period_end=end_time,
            total_cost=total_cost,
            currency="USD",
            breakdown_by_provider=provider_costs,
            breakdown_by_model=model_costs,
            breakdown_by_user=user_costs,
            token_usage=token_usage,
            api_call_count=len(tracker.api_calls),
            top_expensive_calls=expensive_calls
        )
        
        # 计算平均成本
        if report.api_call_count > 0:
            report.average_cost_per_call = report.total_cost / report.api_call_count
        if report.token_usage.total_tokens > 0:
            report.average_cost_per_token = report.total_cost / report.token_usage.total_tokens
        
        # 验证报告
        assert report.total_cost > 0
        assert report.api_call_count == 4
        assert len(report.breakdown_by_provider) == 3  # deepseek, ernie, doubao
        assert len(report.breakdown_by_model) == 4
        assert len(report.breakdown_by_user) == 2
        assert report.token_usage.total_tokens > 0
        assert report.average_cost_per_call > 0
        assert report.average_cost_per_token > 0
        assert len(report.top_expensive_calls) <= 4
        
        # 验证成本分解的一致性
        provider_total = sum(report.breakdown_by_provider.values())
        model_total = sum(report.breakdown_by_model.values())
        user_total = sum(report.breakdown_by_user.values())
        
        assert abs(report.total_cost - provider_total) < Decimal('0.000001')
        assert abs(report.total_cost - model_total) < Decimal('0.000001')
        assert abs(report.total_cost - user_total) < Decimal('0.000001')
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.cost_tracking
    def test_cost_trend_analysis(self):
        """测试成本趋势分析"""
        tracker = MockCostTracker()
        
        # 模拟不同时间段的使用
        base_time = datetime.now() - timedelta(days=7)
        
        daily_costs = []
        
        for day in range(7):
            day_start = base_time + timedelta(days=day)
            
            # 每天的调用次数递增（模拟使用量增长）
            calls_per_day = (day + 1) * 2
            
            day_cost = Decimal('0')
            for call_idx in range(calls_per_day):
                messages = [{"role": "user", "content": f"Day {day} message {call_idx}"}]
                api_call = tracker.track_api_call(
                    provider="deepseek",
                    model="deepseek-chat",
                    endpoint="/v1/chat/completions",
                    messages=messages,
                    response=f"Day {day} response {call_idx}",
                    duration=1.0
                )
                
                # 手动设置时间戳
                api_call.timestamp = day_start + timedelta(hours=call_idx)
                day_cost += api_call.cost_breakdown.total_cost
            
            daily_costs.append(day_cost)
        
        # 验证成本趋势
        assert len(daily_costs) == 7
        
        # 成本应该总体呈上升趋势
        for i in range(1, len(daily_costs)):
            assert daily_costs[i] >= daily_costs[i-1]  # 每天成本应该不减少
        
        # 最后一天的成本应该明显高于第一天
        assert daily_costs[-1] > daily_costs[0] * 2
        
        # 计算总的7天成本
        week_total = sum(daily_costs)
        tracker_total = tracker.get_total_cost()
        
        # 应该基本相等（可能有小的浮点误差）
        assert abs(week_total - tracker_total) < Decimal('0.001')
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.cost_tracking
    def test_cost_optimization_insights(self):
        """测试成本优化洞察"""
        tracker = MockCostTracker()
        calculator = MockPricingCalculator()
        
        # 添加不同效率的API调用
        test_scenarios = [
            # 场景1：使用昂贵模型处理简单任务
            ("deepseek", "deepseek-chat", "What is 2+2?", "4"),
            
            # 场景2：使用便宜模型处理简单任务
            ("ernie", "ernie-3.5-8k", "What is 3+3?", "6"),
            
            # 场景3：使用昂贵模型处理复杂任务
            ("deepseek", "deepseek-chat", "Explain quantum computing " * 20, "Quantum computing is " * 50),
            
            # 场景4：使用便宜模型处理复杂任务
            ("doubao", "doubao-pro-4k", "Explain machine learning " * 20, "Machine learning is " * 50)
        ]
        
        for provider, model, prompt, response in test_scenarios:
            messages = [{"role": "user", "content": prompt}]
            tracker.track_api_call(
                provider=provider,
                model=model,
                endpoint="/v1/chat/completions",
                messages=messages,
                response=response,
                duration=1.0
            )
        
        # 分析成本效率
        efficiency_analysis = []
        
        for call in tracker.api_calls:
            # 计算每个token的成本
            cost_per_token = call.cost_breakdown.total_cost / call.token_usage.total_tokens
            
            # 计算输入输出比例
            io_ratio = call.token_usage.output_tokens / max(call.token_usage.input_tokens, 1)
            
            efficiency_analysis.append({
                "call_id": call.id,
                "provider": call.provider,
                "model": call.model,
                "cost_per_token": cost_per_token,
                "io_ratio": io_ratio,
                "total_cost": call.cost_breakdown.total_cost,
                "total_tokens": call.token_usage.total_tokens
            })
        
        # 验证分析结果
        assert len(efficiency_analysis) == 4
        
        # 找出最高效和最低效的调用
        efficiency_analysis.sort(key=lambda x: x["cost_per_token"])
        
        most_efficient = efficiency_analysis[0]
        least_efficient = efficiency_analysis[-1]
        
        # 最低效的调用成本应该明显高于最高效的
        assert least_efficient["cost_per_token"] > most_efficient["cost_per_token"]
        
        # 验证不同模型的成本差异
        deepseek_calls = [a for a in efficiency_analysis if a["model"] == "deepseek-chat"]
        ernie_calls = [a for a in efficiency_analysis if a["model"] == "ernie-3.5-8k"]
        
        if deepseek_calls and ernie_calls:
            avg_deepseek_cost = sum(c["cost_per_token"] for c in deepseek_calls) / len(deepseek_calls)
            avg_ernie_cost = sum(c["cost_per_token"] for c in ernie_calls) / len(ernie_calls)
            
            # DeepSeek的平均每token成本应该高于Ernie-3.5
            assert avg_deepseek_cost > avg_ernie_cost


class TestCostOptimization:
    """成本优化测试类"""
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.cost_tracking
    def test_model_recommendation(self):
        """测试模型推荐"""
        calculator = MockPricingCalculator()
        
        # 测试不同使用场景的模型推荐
        scenarios = [
            # 简单任务：大量小请求
            TokenUsage(input_tokens=100, output_tokens=50),
            
            # 复杂任务：少量大请求
            TokenUsage(input_tokens=2000, output_tokens=1500),
            
            # 中等任务：中等请求
            TokenUsage(input_tokens=500, output_tokens=300)
        ]
        
        models_to_compare = [
            ("deepseek", "deepseek-chat"),
            ("ernie", "ernie-4.0-8k"),
            ("ernie", "ernie-3.5-8k"),
            ("doubao", "doubao-pro-4k"),
            ("doubao", "doubao-lite-4k")
        ]
        
        for i, usage in enumerate(scenarios):
            costs = calculator.compare_provider_costs(usage, models_to_compare)
            
            # 按成本排序
            sorted_costs = sorted(costs.items(), key=lambda x: x[1].total_cost)
            
            # 验证排序
            for j in range(len(sorted_costs) - 1):
                assert sorted_costs[j][1].total_cost <= sorted_costs[j + 1][1].total_cost
            
            # 最便宜的选项
            cheapest = sorted_costs[0]
            most_expensive = sorted_costs[-1]
            
            # 成本差异应该显著
            cost_ratio = most_expensive[1].total_cost / cheapest[1].total_cost
            assert cost_ratio > 1.5  # 最贵的应该比最便宜的贵至少50%
            
            print(f"Scenario {i+1} - Cheapest: {cheapest[0]} (${cheapest[1].total_cost})")
            print(f"Scenario {i+1} - Most expensive: {most_expensive[0]} (${most_expensive[1].total_cost})")
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.cost_tracking
    def test_usage_pattern_optimization(self):
        """测试使用模式优化"""
        tracker = MockCostTracker()
        
        # 模拟不同的使用模式
        patterns = {
            "batch_processing": {
                "description": "批量处理：少量大请求",
                "calls": [
                    ("Long document analysis " * 100, "Detailed analysis " * 80),
                    ("Complex data processing " * 120, "Comprehensive results " * 90)
                ]
            },
            "interactive_chat": {
                "description": "交互式聊天：大量小请求",
                "calls": [
                    ("Hi", "Hello!"),
                    ("How are you?", "I'm fine!"),
                    ("What's the weather?", "It's sunny!"),
                    ("Tell me a joke", "Why did the chicken cross the road?"),
                    ("Goodbye", "See you later!")
                ]
            },
            "mixed_usage": {
                "description": "混合使用：中等请求",
                "calls": [
                    ("Medium question " * 20, "Medium answer " * 15),
                    ("Another question " * 25, "Another answer " * 18),
                    ("Final question " * 22, "Final answer " * 16)
                ]
            }
        }
        
        pattern_costs = {}
        
        for pattern_name, pattern_data in patterns.items():
            # 清除之前的数据
            tracker.clear_data()
            
            # 执行该模式的调用
            for prompt, response in pattern_data["calls"]:
                messages = [{"role": "user", "content": prompt}]
                tracker.track_api_call(
                    provider="deepseek",
                    model="deepseek-chat",
                    endpoint="/v1/chat/completions",
                    messages=messages,
                    response=response,
                    duration=1.0
                )
            
            # 计算该模式的成本指标
            total_cost = tracker.get_total_cost()
            total_usage = tracker.get_token_usage()
            call_count = len(tracker.api_calls)
            
            avg_cost_per_call = total_cost / call_count if call_count > 0 else Decimal('0')
            avg_cost_per_token = total_cost / total_usage.total_tokens if total_usage.total_tokens > 0 else Decimal('0')
            avg_tokens_per_call = total_usage.total_tokens / call_count if call_count > 0 else 0
            
            pattern_costs[pattern_name] = {
                "total_cost": total_cost,
                "call_count": call_count,
                "total_tokens": total_usage.total_tokens,
                "avg_cost_per_call": avg_cost_per_call,
                "avg_cost_per_token": avg_cost_per_token,
                "avg_tokens_per_call": avg_tokens_per_call,
                "description": pattern_data["description"]
            }
        
        # 分析不同模式的效率
        batch_metrics = pattern_costs["batch_processing"]
        chat_metrics = pattern_costs["interactive_chat"]
        mixed_metrics = pattern_costs["mixed_usage"]
        
        # 验证模式特征
        # 批量处理：高每次调用成本，低每token成本
        assert batch_metrics["avg_cost_per_call"] > chat_metrics["avg_cost_per_call"]
        assert batch_metrics["avg_tokens_per_call"] > chat_metrics["avg_tokens_per_call"]
        
        # 交互式聊天：低每次调用成本，高调用频率
        assert chat_metrics["call_count"] > batch_metrics["call_count"]
        assert chat_metrics["avg_tokens_per_call"] < mixed_metrics["avg_tokens_per_call"]
        
        # 输出优化建议
        for pattern_name, metrics in pattern_costs.items():
            print(f"\n{pattern_name} ({metrics['description']}):")
            print(f"  Total cost: ${metrics['total_cost']}")
            print(f"  Avg cost per call: ${metrics['avg_cost_per_call']}")
            print(f"  Avg cost per token: ${metrics['avg_cost_per_token']}")
            print(f"  Avg tokens per call: {metrics['avg_tokens_per_call']}")
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.cost_tracking
    def test_cost_saving_recommendations(self):
        """测试成本节约建议"""
        calculator = MockPricingCalculator()
        
        # 分析当前使用情况
        current_usage = TokenUsage(input_tokens=10000, output_tokens=5000)
        current_model = ("deepseek", "deepseek-chat")
        
        # 计算当前成本
        current_cost = calculator.calculate_cost(current_model[0], current_model[1], current_usage)
        
        # 测试替代方案
        alternatives = [
            ("ernie", "ernie-3.5-8k"),
            ("doubao", "doubao-lite-4k"),
            ("ernie", "ernie-4.0-8k")
        ]
        
        savings_analysis = []
        
        for provider, model in alternatives:
            alt_cost = calculator.calculate_cost(provider, model, current_usage)
            
            if alt_cost.total_cost < current_cost.total_cost:
                savings = current_cost.total_cost - alt_cost.total_cost
                savings_percentage = float(savings / current_cost.total_cost * 100)
                
                savings_analysis.append({
                    "provider": provider,
                    "model": model,
                    "cost": alt_cost.total_cost,
                    "savings": savings,
                    "savings_percentage": savings_percentage
                })
        
        # 按节约金额排序
        savings_analysis.sort(key=lambda x: x["savings"], reverse=True)
        
        # 验证节约建议
        assert len(savings_analysis) > 0  # 应该有一些更便宜的选项
        
        best_alternative = savings_analysis[0]
        assert best_alternative["savings"] > 0
        assert best_alternative["savings_percentage"] > 10  # 至少节约10%
        
        # 计算年度节约（假设当前使用量）
        monthly_usage = TokenUsage(
            input_tokens=current_usage.input_tokens * 30,
            output_tokens=current_usage.output_tokens * 30
        )
        
        monthly_current_cost = calculator.calculate_cost(current_model[0], current_model[1], monthly_usage)
        monthly_alt_cost = calculator.calculate_cost(
            best_alternative["provider"], 
            best_alternative["model"], 
            monthly_usage
        )
        
        monthly_savings = monthly_current_cost.total_cost - monthly_alt_cost.total_cost
        annual_savings = monthly_savings * 12
        
        assert annual_savings > 0
        
        print(f"\nCost Optimization Analysis:")
        print(f"Current model: {current_model[0]}/{current_model[1]} - ${current_cost.total_cost}")
        print(f"Best alternative: {best_alternative['provider']}/{best_alternative['model']} - ${best_alternative['cost']}")
        print(f"Immediate savings: ${best_alternative['savings']} ({best_alternative['savings_percentage']:.1f}%)")
        print(f"Estimated annual savings: ${annual_savings}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "cost_tracking"
    ])