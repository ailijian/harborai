#!/usr/bin/env python3
"""
HarborAI 多模型切换示例

这个示例展示了如何在HarborAI中动态切换不同的AI模型，
根据任务类型、成本考虑或性能需求选择最适合的模型。

场景描述:
- 动态模型选择
- 性能对比分析
- 智能路由策略
- 成本效益优化

应用价值:
- 优化成本效益
- 提升任务适配性
- 实现智能负载均衡
- 支持多厂商模型
"""

import os
import time
import asyncio
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
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


class TaskType(Enum):
    """任务类型枚举"""
    CHAT = "chat"
    CODE = "code"
    REASONING = "reasoning"
    TRANSLATION = "translation"
    CREATIVE = "creative"
    ANALYSIS = "analysis"


class ModelProvider(Enum):
    """模型提供商枚举"""
    DEEPSEEK = "deepseek"


@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    provider: ModelProvider
    api_key_env: str
    base_url_env: str
    default_base_url: str
    cost_per_1k_tokens: float
    max_tokens: int
    strengths: List[TaskType]
    description: str


@dataclass
class ModelPerformance:
    """模型性能指标"""
    model_name: str
    response_time: float
    token_count: int
    cost: float
    quality_score: Optional[float] = None
    success: bool = True
    error: Optional[str] = None


class ModelRouter:
    """智能模型路由器"""
    
    def __init__(self):
        self.models = {
            "deepseek-chat": ModelConfig(
                name="deepseek-chat",
                provider=ModelProvider.DEEPSEEK,
                api_key_env="DEEPSEEK_API_KEY",
                base_url_env="DEEPSEEK_BASE_URL",
                default_base_url="https://api.deepseek.com",
                cost_per_1k_tokens=0.0014,
                max_tokens=4096,
                strengths=[TaskType.CHAT, TaskType.CODE, TaskType.CREATIVE, TaskType.ANALYSIS, TaskType.TRANSLATION],
                description="DeepSeek通用模型，适用于对话、代码生成、内容创作等多种任务"
            ),
            "deepseek-reasoner": ModelConfig(
                name="deepseek-reasoner",
                provider=ModelProvider.DEEPSEEK,
                api_key_env="DEEPSEEK_API_KEY",
                base_url_env="DEEPSEEK_BASE_URL",
                default_base_url="https://api.deepseek.com",
                cost_per_1k_tokens=0.0055,
                max_tokens=4096,
                strengths=[TaskType.REASONING],
                description="DeepSeek推理专用模型，适用于复杂推理、数学计算、逻辑分析"
            )
        }
        
        self.clients = {}
        self.performance_history = []
    
    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        available = []
        for model_name, config in self.models.items():
            api_key = os.getenv(config.api_key_env)
            if api_key:
                available.append(model_name)
        return available
    
    def get_client(self, model_name: str) -> Optional[HarborAI]:
        """获取指定模型的客户端"""
        if model_name not in self.models:
            return None
        
        if model_name in self.clients:
            return self.clients[model_name]
        
        config = self.models[model_name]
        api_key = os.getenv(config.api_key_env)
        base_url = os.getenv(config.base_url_env, config.default_base_url)
        
        if not api_key:
            return None
        
        client = HarborAI(api_key=api_key, base_url=base_url)
        self.clients[model_name] = client
        return client
    
    def recommend_model(self, task_type: TaskType, budget_priority: bool = False) -> str:
        """
        根据任务类型推荐模型
        
        Args:
            task_type: 任务类型
            budget_priority: 是否优先考虑成本
            
        Returns:
            str: 推荐的模型名称
        """
        available_models = self.get_available_models()
        suitable_models = []
        
        for model_name in available_models:
            config = self.models[model_name]
            if task_type in config.strengths:
                suitable_models.append((model_name, config))
        
        if not suitable_models:
            # 如果没有专门适合的模型，返回通用模型
            for model_name in available_models:
                config = self.models[model_name]
                if TaskType.CHAT in config.strengths:
                    return model_name
            return available_models[0] if available_models else None
        
        if budget_priority:
            # 按成本排序
            suitable_models.sort(key=lambda x: x[1].cost_per_1k_tokens)
        else:
            # 按能力排序（这里简化为按成本倒序，实际应该有更复杂的评分）
            suitable_models.sort(key=lambda x: x[1].cost_per_1k_tokens, reverse=True)
        
        return suitable_models[0][0]
    
    def record_performance(self, performance: ModelPerformance):
        """记录模型性能"""
        self.performance_history.append(performance)
    
    def get_performance_stats(self, model_name: str = None) -> Dict[str, Any]:
        """获取性能统计"""
        if model_name:
            records = [p for p in self.performance_history if p.model_name == model_name]
        else:
            records = self.performance_history
        
        if not records:
            return {}
        
        successful_records = [r for r in records if r.success]
        
        if not successful_records:
            return {"total_calls": len(records), "success_rate": 0}
        
        return {
            "total_calls": len(records),
            "successful_calls": len(successful_records),
            "success_rate": len(successful_records) / len(records),
            "avg_response_time": statistics.mean(r.response_time for r in successful_records),
            "avg_token_count": statistics.mean(r.token_count for r in successful_records),
            "avg_cost": statistics.mean(r.cost for r in successful_records),
            "total_cost": sum(r.cost for r in successful_records)
        }


async def call_model_with_metrics(router: ModelRouter, model_name: str, prompt: str, **kwargs) -> ModelPerformance:
    """
    调用模型并记录性能指标
    
    Args:
        router: 模型路由器
        model_name: 模型名称
        prompt: 提示词
        **kwargs: 其他参数
        
    Returns:
        ModelPerformance: 性能指标
    """
    client = router.get_client(model_name)
    if not client:
        return ModelPerformance(
            model_name=model_name,
            response_time=0,
            token_count=0,
            cost=0,
            success=False,
            error="模型客户端不可用"
        )
    
    config = router.models[model_name]
    start_time = time.time()
    
    try:
        response = await client.chat.completions.acreate(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens")  # 默认无限制，由模型厂商控制
        )
        
        elapsed_time = time.time() - start_time
        usage = response.usage
        cost = (usage.total_tokens / 1000) * config.cost_per_1k_tokens
        
        performance = ModelPerformance(
            model_name=model_name,
            response_time=elapsed_time,
            token_count=usage.total_tokens,
            cost=cost,
            success=True
        )
        
        router.record_performance(performance)
        return performance
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        performance = ModelPerformance(
            model_name=model_name,
            response_time=elapsed_time,
            token_count=0,
            cost=0,
            success=False,
            error=str(e)
        )
        
        router.record_performance(performance)
        return performance


async def compare_models_performance(router: ModelRouter, prompt: str, models: List[str] = None) -> Dict[str, ModelPerformance]:
    """
    对比多个模型的性能
    
    Args:
        router: 模型路由器
        prompt: 测试提示
        models: 要对比的模型列表
        
    Returns:
        Dict: 各模型的性能指标
    """
    if models is None:
        models = router.get_available_models()
    
    print(f"\n⚖️  对比模型性能")
    print(f"🎯 测试提示: {prompt[:100]}...")
    print(f"📋 测试模型: {', '.join(models)}")
    
    # 并发调用所有模型
    tasks = [call_model_with_metrics(router, model, prompt) for model in models]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 整理结果
    performance_results = {}
    for i, result in enumerate(results):
        model_name = models[i]
        if isinstance(result, Exception):
            performance_results[model_name] = ModelPerformance(
                model_name=model_name,
                response_time=0,
                token_count=0,
                cost=0,
                success=False,
                error=str(result)
            )
        else:
            performance_results[model_name] = result
    
    # 显示对比结果
    print(f"\n📊 性能对比结果:")
    print("-" * 80)
    print(f"{'模型名称':<20} {'响应时间':<10} {'Token数':<10} {'成本':<10} {'状态':<10}")
    print("-" * 80)
    
    for model_name, perf in performance_results.items():
        if perf.success:
            print(f"{model_name:<20} {perf.response_time:<10.2f} {perf.token_count:<10} ${perf.cost:<9.4f} {'✅成功':<10}")
        else:
            print(f"{model_name:<20} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'❌失败':<10}")
    
    return performance_results


async def intelligent_model_routing(router: ModelRouter, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    智能模型路由
    
    Args:
        router: 模型路由器
        tasks: 任务列表，每个任务包含prompt和task_type
        
    Returns:
        List: 处理结果
    """
    print(f"\n🧠 智能模型路由")
    print(f"📋 处理 {len(tasks)} 个任务")
    
    results = []
    
    for i, task in enumerate(tasks):
        prompt = task["prompt"]
        task_type = task["task_type"]
        budget_priority = task.get("budget_priority", False)
        
        # 推荐模型
        recommended_model = router.recommend_model(task_type, budget_priority)
        
        if not recommended_model:
            results.append({
                "task_index": i,
                "task_type": task_type.value,
                "prompt": prompt[:50] + "...",
                "recommended_model": None,
                "success": False,
                "error": "没有可用的模型"
            })
            continue
        
        print(f"\n🎯 任务 {i+1}: {task_type.value}")
        print(f"   推荐模型: {recommended_model}")
        print(f"   预算优先: {budget_priority}")
        
        # 调用推荐的模型
        performance = await call_model_with_metrics(router, recommended_model, prompt)
        
        result = {
            "task_index": i,
            "task_type": task_type.value,
            "prompt": prompt[:50] + "...",
            "recommended_model": recommended_model,
            "performance": performance,
            "success": performance.success
        }
        
        if performance.success:
            print(f"   ✅ 完成 - 耗时: {performance.response_time:.2f}s, 成本: ${performance.cost:.4f}")
        else:
            print(f"   ❌ 失败 - {performance.error}")
            result["error"] = performance.error
        
        results.append(result)
    
    return results


async def load_balancing_demo(router: ModelRouter, prompt: str, num_requests: int = 10):
    """
    负载均衡演示
    
    Args:
        router: 模型路由器
        prompt: 测试提示
        num_requests: 请求数量
    """
    print(f"\n⚖️  负载均衡演示")
    print(f"📋 发送 {num_requests} 个并发请求")
    
    available_models = router.get_available_models()
    if len(available_models) < 2:
        print("❌ 需要至少2个可用模型进行负载均衡演示")
        return
    
    # 轮询分配请求到不同模型
    tasks = []
    for i in range(num_requests):
        model = available_models[i % len(available_models)]
        tasks.append(call_model_with_metrics(router, model, f"{prompt} (请求 {i+1})"))
    
    start_time = time.time()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    total_time = time.time() - start_time
    
    # 统计结果
    model_stats = {}
    for i, result in enumerate(results):
        model = available_models[i % len(available_models)]
        if model not in model_stats:
            model_stats[model] = {"success": 0, "failed": 0, "total_time": 0, "total_cost": 0}
        
        if isinstance(result, Exception) or not result.success:
            model_stats[model]["failed"] += 1
        else:
            model_stats[model]["success"] += 1
            model_stats[model]["total_time"] += result.response_time
            model_stats[model]["total_cost"] += result.cost
    
    print(f"\n📊 负载均衡结果 (总耗时: {total_time:.2f}秒):")
    print("-" * 60)
    print(f"{'模型':<20} {'成功':<8} {'失败':<8} {'平均耗时':<12} {'总成本':<10}")
    print("-" * 60)
    
    for model, stats in model_stats.items():
        avg_time = stats["total_time"] / stats["success"] if stats["success"] > 0 else 0
        print(f"{model:<20} {stats['success']:<8} {stats['failed']:<8} {avg_time:<12.2f} ${stats['total_cost']:<9.4f}")


def show_model_capabilities(router: ModelRouter):
    """显示模型能力对比"""
    print(f"\n📋 模型能力对比")
    print("=" * 80)
    
    available_models = router.get_available_models()
    
    print(f"{'模型名称':<20} {'提供商':<12} {'成本/1K':<10} {'擅长领域':<30} {'描述'}")
    print("-" * 80)
    
    for model_name in available_models:
        config = router.models[model_name]
        strengths = ", ".join([t.value for t in config.strengths])
        print(f"{model_name:<20} {config.provider.value:<12} ${config.cost_per_1k_tokens:<9.4f} {strengths:<30} {config.description}")


async def adaptive_model_selection(router: ModelRouter, prompts: List[str]):
    """
    自适应模型选择演示
    
    Args:
        router: 模型路由器
        prompts: 测试提示列表
    """
    print(f"\n🎯 自适应模型选择演示")
    print("=" * 60)
    
    # 定义不同类型的任务
    task_configs = [
        {"task_type": TaskType.CHAT, "budget_priority": True},
        {"task_type": TaskType.CODE, "budget_priority": False},
        {"task_type": TaskType.REASONING, "budget_priority": False},
        {"task_type": TaskType.CREATIVE, "budget_priority": True},
    ]
    
    for i, prompt in enumerate(prompts):
        if i >= len(task_configs):
            break
        
        config = task_configs[i]
        task_type = config["task_type"]
        budget_priority = config["budget_priority"]
        
        print(f"\n📝 任务 {i+1}: {task_type.value}")
        print(f"   提示: {prompt[:80]}...")
        print(f"   预算优先: {budget_priority}")
        
        # 获取推荐模型
        recommended = router.recommend_model(task_type, budget_priority)
        print(f"   推荐模型: {recommended}")
        
        if recommended:
            # 执行任务
            performance = await call_model_with_metrics(router, recommended, prompt)
            if performance.success:
                print(f"   ✅ 执行成功 - 耗时: {performance.response_time:.2f}s, 成本: ${performance.cost:.4f}")
            else:
                print(f"   ❌ 执行失败: {performance.error}")


async def main():
    """主函数"""
    print("="*60)
    print("🔄 HarborAI 多模型切换示例")
    print("="*60)
    
    # 初始化路由器
    router = ModelRouter()
    available_models = router.get_available_models()
    
    if not available_models:
        print("❌ 没有可用的模型，请检查环境变量配置")
        print("💡 需要配置的环境变量:")
        for model_name, config in router.models.items():
            print(f"   - {config.api_key_env}: {config.description}")
        return
    
    print(f"✅ 发现 {len(available_models)} 个可用模型: {', '.join(available_models)}")
    
    # 显示模型能力
    show_model_capabilities(router)
    
    # 测试提示
    test_prompts = [
        "你好，请介绍一下你自己。",
        "请写一个Python函数来计算斐波那契数列。",
        "如果一个农夫有17只羊，除了9只以外都死了，请问农夫还有几只羊？",
        "写一首关于春天的诗。"
    ]
    
    # 1. 模型性能对比
    print(f"\n🔹 1. 模型性能对比")
    await compare_models_performance(router, test_prompts[0], available_models[:3])
    
    # 2. 智能模型路由
    print(f"\n🔹 2. 智能模型路由")
    routing_tasks = [
        {"prompt": test_prompts[0], "task_type": TaskType.CHAT, "budget_priority": True},
        {"prompt": test_prompts[1], "task_type": TaskType.CODE, "budget_priority": False},
        {"prompt": test_prompts[2], "task_type": TaskType.REASONING, "budget_priority": False},
        {"prompt": test_prompts[3], "task_type": TaskType.CREATIVE, "budget_priority": True}
    ]
    await intelligent_model_routing(router, routing_tasks)
    
    # 3. 负载均衡演示
    if len(available_models) >= 2:
        print(f"\n🔹 3. 负载均衡演示")
        await load_balancing_demo(router, "请简单介绍一下人工智能。", 6)
    
    # 4. 自适应模型选择
    print(f"\n🔹 4. 自适应模型选择")
    await adaptive_model_selection(router, test_prompts)
    
    # 5. 性能统计报告
    print(f"\n🔹 5. 性能统计报告")
    print("=" * 60)
    
    overall_stats = router.get_performance_stats()
    if overall_stats:
        print(f"📊 总体统计:")
        print(f"   总调用次数: {overall_stats['total_calls']}")
        print(f"   成功率: {overall_stats['success_rate']:.1%}")
        print(f"   平均响应时间: {overall_stats['avg_response_time']:.2f}秒")
        print(f"   平均Token数: {overall_stats['avg_token_count']:.0f}")
        print(f"   总成本: ${overall_stats['total_cost']:.4f}")
        
        print(f"\n📈 各模型统计:")
        for model in available_models:
            model_stats = router.get_performance_stats(model)
            if model_stats and model_stats['total_calls'] > 0:
                print(f"   {model}:")
                print(f"     调用次数: {model_stats['total_calls']}")
                print(f"     成功率: {model_stats['success_rate']:.1%}")
                if model_stats['successful_calls'] > 0:
                    print(f"     平均耗时: {model_stats['avg_response_time']:.2f}秒")
                    print(f"     总成本: ${model_stats['total_cost']:.4f}")
    
    print(f"\n🎉 多模型切换示例执行完成！")


if __name__ == "__main__":
    asyncio.run(main())