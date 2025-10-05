#!/usr/bin/env python3
"""
HarborAI 降级策略演示

场景描述:
在生产环境中，当主要AI服务不可用或性能下降时，需要自动切换到备用方案，
确保服务连续性。本示例展示多层级降级策略、自动故障转移等机制。

应用价值:
- 确保服务连续性和可用性
- 优化用户体验，避免服务中断
- 降低服务中断风险
- 在成本和性能之间找到平衡

核心功能:
1. 多层级降级策略
2. 服务健康监控
3. 自动故障转移
4. 性能监控与告警
5. 优雅降级处理
"""

import asyncio
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceTier(Enum):
    """服务层级"""
    PRIMARY = "primary"      # 主要服务
    SECONDARY = "secondary"  # 次要服务
    FALLBACK = "fallback"    # 降级服务
    EMERGENCY = "emergency"  # 紧急服务

class ServiceStatus(Enum):
    """服务状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"

@dataclass
class ServiceConfig:
    """服务配置"""
    name: str
    model: str
    tier: ServiceTier
    api_key: str
    base_url: str = "https://api.harborai.com/v1"
    max_tokens: int = 1000
    temperature: float = 0.7
    cost_per_token: float = 0.0001
    expected_latency: float = 2.0
    quality_score: float = 1.0

@dataclass
class ServiceMetrics:
    """服务指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    total_cost: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error_time: Optional[datetime] = None
    consecutive_failures: int = 0
    
    def add_success(self, latency: float, cost: float):
        """添加成功记录"""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency += latency
        self.total_cost += cost
        self.last_request_time = datetime.now()
        self.consecutive_failures = 0
    
    def add_failure(self):
        """添加失败记录"""
        self.total_requests += 1
        self.failed_requests += 1
        self.last_error_time = datetime.now()
        self.consecutive_failures += 1
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    def get_average_latency(self) -> float:
        """获取平均延迟"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency / self.successful_requests
    
    def get_average_cost(self) -> float:
        """获取平均成本"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_cost / self.successful_requests

class ServiceHealthChecker:
    """服务健康检查器"""
    
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        self.metrics = ServiceMetrics()
        self.status = ServiceStatus.HEALTHY
        
    async def health_check(self) -> bool:
        """执行健康检查"""
        try:
            start_time = time.time()
            
            # 发送简单的健康检查请求
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=10,
                timeout=5.0
            )
            
            latency = time.time() - start_time
            cost = len(response.choices[0].message.content) * self.config.cost_per_token
            
            self.metrics.add_success(latency, cost)
            self._update_status()
            
            logger.info(f"Health check passed for {self.config.name} - Latency: {latency:.2f}s")
            return True
            
        except Exception as e:
            self.metrics.add_failure()
            self._update_status()
            
            logger.warning(f"Health check failed for {self.config.name}: {str(e)}")
            return False
    
    def _update_status(self):
        """更新服务状态"""
        success_rate = self.metrics.get_success_rate()
        avg_latency = self.metrics.get_average_latency()
        
        if self.metrics.consecutive_failures >= 3:
            self.status = ServiceStatus.OFFLINE
        elif success_rate < 0.5:
            self.status = ServiceStatus.UNHEALTHY
        elif success_rate < 0.8 or avg_latency > self.config.expected_latency * 2:
            self.status = ServiceStatus.DEGRADED
        else:
            self.status = ServiceStatus.HEALTHY
    
    async def make_request(self, messages: List[Dict], **kwargs) -> ChatCompletion:
        """发送请求"""
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs
            )
            
            latency = time.time() - start_time
            cost = response.usage.total_tokens * self.config.cost_per_token if response.usage else 0
            
            self.metrics.add_success(latency, cost)
            self._update_status()
            
            return response
            
        except Exception as e:
            self.metrics.add_failure()
            self._update_status()
            raise e

class FallbackStrategy:
    """降级策略"""
    
    def __init__(self, services: List[ServiceConfig]):
        self.services = {config.name: ServiceHealthChecker(config) for config in services}
        self.service_order = sorted(services, key=lambda x: x.tier.value)
        self.current_service = None
        self.fallback_history: List[Dict] = []
        
    async def initialize(self):
        """初始化服务"""
        logger.info("Initializing fallback strategy...")
        
        # 对所有服务进行健康检查
        for service_name, checker in self.services.items():
            await checker.health_check()
        
        # 选择最佳服务
        self.current_service = self._select_best_service()
        logger.info(f"Selected primary service: {self.current_service}")
    
    def _select_best_service(self) -> Optional[str]:
        """选择最佳服务"""
        # 按优先级排序，选择健康的服务
        for config in self.service_order:
            checker = self.services[config.name]
            if checker.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
                return config.name
        
        # 如果没有健康的服务，选择最不差的
        available_services = [(name, checker) for name, checker in self.services.items()]
        if available_services:
            best_service = max(available_services, key=lambda x: x[1].metrics.get_success_rate())
            return best_service[0]
        
        return None
    
    async def make_request(self, messages: List[Dict], **kwargs) -> ChatCompletion:
        """发送请求（带降级策略）"""
        original_service = self.current_service
        
        # 尝试所有可用服务
        for config in self.service_order:
            service_name = config.name
            checker = self.services[service_name]
            
            # 跳过离线服务
            if checker.status == ServiceStatus.OFFLINE:
                continue
            
            try:
                logger.info(f"Attempting request with service: {service_name}")
                response = await checker.make_request(messages, **kwargs)
                
                # 如果使用了降级服务，记录降级事件
                if service_name != original_service:
                    self._record_fallback(original_service, service_name, "success")
                    self.current_service = service_name
                
                return response
                
            except Exception as e:
                logger.warning(f"Request failed with {service_name}: {str(e)}")
                
                # 记录降级事件
                if service_name == original_service:
                    self._record_fallback(original_service, None, "failure")
                
                continue
        
        # 所有服务都失败了
        raise Exception("All services are unavailable")
    
    def _record_fallback(self, from_service: Optional[str], to_service: Optional[str], reason: str):
        """记录降级事件"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "from_service": from_service,
            "to_service": to_service,
            "reason": reason
        }
        self.fallback_history.append(event)
        
        # 保持历史记录在合理范围内
        if len(self.fallback_history) > 100:
            self.fallback_history = self.fallback_history[-100:]
        
        logger.info(f"Fallback event: {from_service} -> {to_service} ({reason})")
    
    async def periodic_health_check(self, interval: float = 30.0):
        """定期健康检查"""
        while True:
            try:
                logger.info("Performing periodic health checks...")
                
                # 检查所有服务健康状态
                for service_name, checker in self.services.items():
                    await checker.health_check()
                
                # 重新选择最佳服务
                best_service = self._select_best_service()
                if best_service and best_service != self.current_service:
                    logger.info(f"Switching to better service: {self.current_service} -> {best_service}")
                    self.current_service = best_service
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
                await asyncio.sleep(interval)
    
    def get_service_status(self) -> Dict:
        """获取服务状态"""
        status = {}
        for service_name, checker in self.services.items():
            config = next(c for c in self.service_order if c.name == service_name)
            status[service_name] = {
                "tier": config.tier.value,
                "status": checker.status.value,
                "success_rate": checker.metrics.get_success_rate(),
                "average_latency": checker.metrics.get_average_latency(),
                "average_cost": checker.metrics.get_average_cost(),
                "total_requests": checker.metrics.total_requests,
                "consecutive_failures": checker.metrics.consecutive_failures
            }
        return status
    
    def get_fallback_summary(self) -> Dict:
        """获取降级摘要"""
        if not self.fallback_history:
            return {"total_fallbacks": 0, "recent_fallbacks": []}
        
        recent_fallbacks = self.fallback_history[-10:]  # 最近10次降级
        
        return {
            "total_fallbacks": len(self.fallback_history),
            "recent_fallbacks": recent_fallbacks,
            "current_service": self.current_service,
            "fallback_rate": len([e for e in recent_fallbacks if e["reason"] == "failure"]) / max(len(recent_fallbacks), 1)
        }

# 演示函数
async def demo_basic_fallback():
    """演示基础降级功能"""
    print("\n🔄 基础降级演示")
    print("=" * 50)
    
    # 配置多个服务
    services = [
        ServiceConfig(
            name="primary-gpt4",
            model="gpt-4",
            tier=ServiceTier.PRIMARY,
            api_key="your-openai-key",
            cost_per_token=0.03,
            expected_latency=3.0,
            quality_score=1.0
        ),
        ServiceConfig(
            name="secondary-gpt35",
            model="gpt-3.5-turbo",
            tier=ServiceTier.SECONDARY,
            api_key="your-openai-key",
            cost_per_token=0.002,
            expected_latency=2.0,
            quality_score=0.8
        ),
        ServiceConfig(
            name="fallback-deepseek",
            model="deepseek-chat",
            tier=ServiceTier.FALLBACK,
            api_key="your-deepseek-key",
            cost_per_token=0.0001,
            expected_latency=1.5,
            quality_score=0.7
        )
    ]
    
    # 创建降级策略
    strategy = FallbackStrategy(services)
    await strategy.initialize()
    
    # 测试请求
    test_messages = [
        {"role": "user", "content": "什么是机器学习？请简要说明。"}
    ]
    
    try:
        response = await strategy.make_request(test_messages)
        print(f"✅ 请求成功")
        print(f"🎯 使用服务: {strategy.current_service}")
        print(f"📝 回复: {response.choices[0].message.content[:100]}...")
        
        # 显示服务状态
        status = strategy.get_service_status()
        print(f"\n📊 服务状态:")
        for service_name, info in status.items():
            print(f"   - {service_name}: {info['status']} (成功率: {info['success_rate']:.1%})")
        
    except Exception as e:
        print(f"❌ 所有服务都不可用: {str(e)}")

async def demo_service_monitoring():
    """演示服务监控"""
    print("\n📊 服务监控演示")
    print("=" * 50)
    
    # 配置服务（使用可用的API密钥）
    services = [
        ServiceConfig(
            name="primary-deepseek",
            model="deepseek-chat",
            tier=ServiceTier.PRIMARY,
            api_key="your-deepseek-key",
            cost_per_token=0.0001,
            expected_latency=2.0
        ),
        ServiceConfig(
            name="fallback-deepseek-r1",
            model="deepseek-r1",
            tier=ServiceTier.FALLBACK,
            api_key="your-deepseek-key",
            cost_per_token=0.0002,
            expected_latency=3.0
        )
    ]
    
    strategy = FallbackStrategy(services)
    await strategy.initialize()
    
    # 启动健康监控（后台任务）
    monitoring_task = asyncio.create_task(strategy.periodic_health_check(interval=5.0))
    
    try:
        # 模拟一些请求
        test_messages = [
            {"role": "user", "content": "测试请求1"},
            {"role": "user", "content": "测试请求2"},
            {"role": "user", "content": "测试请求3"}
        ]
        
        print("🔄 发送测试请求...")
        for i, messages in enumerate([test_messages[0]]):  # 只发送一个请求用于演示
            try:
                response = await strategy.make_request([messages])
                print(f"✅ 请求 {i+1} 成功 - 服务: {strategy.current_service}")
            except Exception as e:
                print(f"❌ 请求 {i+1} 失败: {str(e)}")
            
            await asyncio.sleep(2)
        
        # 等待一段时间让监控运行
        await asyncio.sleep(10)
        
        # 停止监控
        monitoring_task.cancel()
        
        # 显示监控结果
        status = strategy.get_service_status()
        print(f"\n📊 最终服务状态:")
        for service_name, info in status.items():
            print(f"   - {service_name}:")
            print(f"     状态: {info['status']}")
            print(f"     成功率: {info['success_rate']:.1%}")
            print(f"     平均延迟: {info['average_latency']:.2f}s")
            print(f"     平均成本: ${info['average_cost']:.6f}")
        
        # 显示降级摘要
        fallback_summary = strategy.get_fallback_summary()
        print(f"\n🔄 降级摘要:")
        print(f"   - 总降级次数: {fallback_summary['total_fallbacks']}")
        print(f"   - 当前服务: {fallback_summary['current_service']}")
        print(f"   - 降级率: {fallback_summary['fallback_rate']:.1%}")
        
    except asyncio.CancelledError:
        pass

async def demo_cost_optimization():
    """演示成本优化"""
    print("\n💰 成本优化演示")
    print("=" * 50)
    
    # 配置不同成本的服务
    services = [
        ServiceConfig(
            name="premium-service",
            model="gpt-4",
            tier=ServiceTier.PRIMARY,
            api_key="your-openai-key",
            cost_per_token=0.03,
            quality_score=1.0
        ),
        ServiceConfig(
            name="standard-service",
            model="gpt-3.5-turbo",
            tier=ServiceTier.SECONDARY,
            api_key="your-openai-key",
            cost_per_token=0.002,
            quality_score=0.8
        ),
        ServiceConfig(
            name="budget-service",
            model="deepseek-chat",
            tier=ServiceTier.FALLBACK,
            api_key="your-deepseek-key",
            cost_per_token=0.0001,
            quality_score=0.7
        )
    ]
    
    strategy = FallbackStrategy(services)
    await strategy.initialize()
    
    # 模拟不同类型的请求
    request_types = [
        ("简单问答", "什么是AI？"),
        ("复杂分析", "请详细分析人工智能在医疗领域的应用前景，包括技术挑战和伦理考虑。"),
        ("创意写作", "写一个关于未来城市的科幻短故事。")
    ]
    
    total_cost = 0.0
    results = []
    
    for request_type, content in request_types:
        try:
            messages = [{"role": "user", "content": content}]
            
            start_time = time.time()
            response = await strategy.make_request(messages)
            end_time = time.time()
            
            # 估算成本
            service_config = next(s for s in services if s.name == strategy.current_service)
            estimated_cost = len(response.choices[0].message.content) * service_config.cost_per_token
            total_cost += estimated_cost
            
            result = {
                "type": request_type,
                "service": strategy.current_service,
                "latency": end_time - start_time,
                "cost": estimated_cost,
                "quality": service_config.quality_score
            }
            results.append(result)
            
            print(f"✅ {request_type}")
            print(f"   服务: {strategy.current_service}")
            print(f"   延迟: {result['latency']:.2f}s")
            print(f"   成本: ${result['cost']:.6f}")
            print(f"   质量分数: {result['quality']}")
            
        except Exception as e:
            print(f"❌ {request_type} 失败: {str(e)}")
    
    # 成本分析
    print(f"\n💰 成本分析:")
    print(f"   - 总成本: ${total_cost:.6f}")
    print(f"   - 平均成本: ${total_cost/len(results):.6f}")
    
    # 如果全部使用最贵服务的成本对比
    premium_cost = sum(len(r["type"]) * 0.03 for r in results)  # 简化计算
    savings = premium_cost - total_cost
    print(f"   - 如果全用高端服务: ${premium_cost:.6f}")
    print(f"   - 节省成本: ${savings:.6f} ({savings/premium_cost*100:.1f}%)")

async def demo_intelligent_routing():
    """演示智能路由"""
    print("\n🧠 智能路由演示")
    print("=" * 50)
    
    class IntelligentFallbackStrategy(FallbackStrategy):
        """智能降级策略"""
        
        def _select_service_for_request(self, messages: List[Dict]) -> str:
            """根据请求内容选择最适合的服务"""
            content = " ".join([msg.get("content", "") for msg in messages])
            content_length = len(content)
            
            # 根据内容复杂度选择服务
            if content_length > 500 or "分析" in content or "详细" in content:
                # 复杂请求，优先使用高质量服务
                for config in self.service_order:
                    checker = self.services[config.name]
                    if checker.status == ServiceStatus.HEALTHY and config.quality_score >= 0.9:
                        return config.name
            elif content_length < 100:
                # 简单请求，优先使用经济服务
                for config in reversed(self.service_order):
                    checker = self.services[config.name]
                    if checker.status == ServiceStatus.HEALTHY:
                        return config.name
            
            # 默认选择
            return self._select_best_service()
        
        async def make_request(self, messages: List[Dict], **kwargs) -> ChatCompletion:
            """智能路由请求"""
            # 选择最适合的服务
            selected_service = self._select_service_for_request(messages)
            
            if selected_service:
                try:
                    checker = self.services[selected_service]
                    response = await checker.make_request(messages, **kwargs)
                    logger.info(f"Intelligent routing selected: {selected_service}")
                    return response
                except Exception as e:
                    logger.warning(f"Selected service {selected_service} failed, falling back...")
            
            # 如果选择的服务失败，使用标准降级策略
            return await super().make_request(messages, **kwargs)
    
    # 配置服务
    services = [
        ServiceConfig(
            name="high-quality",
            model="gpt-4",
            tier=ServiceTier.PRIMARY,
            api_key="your-openai-key",
            cost_per_token=0.03,
            quality_score=1.0
        ),
        ServiceConfig(
            name="balanced",
            model="gpt-3.5-turbo",
            tier=ServiceTier.SECONDARY,
            api_key="your-openai-key",
            cost_per_token=0.002,
            quality_score=0.8
        ),
        ServiceConfig(
            name="economical",
            model="deepseek-chat",
            tier=ServiceTier.FALLBACK,
            api_key="your-deepseek-key",
            cost_per_token=0.0001,
            quality_score=0.7
        )
    ]
    
    strategy = IntelligentFallbackStrategy(services)
    await strategy.initialize()
    
    # 测试不同类型的请求
    test_cases = [
        ("简单问答", "你好"),
        ("复杂分析", "请详细分析人工智能在金融领域的应用，包括风险管理、算法交易、客户服务等方面的技术实现和商业价值。"),
        ("中等复杂度", "解释一下机器学习的基本概念")
    ]
    
    for case_type, content in test_cases:
        try:
            messages = [{"role": "user", "content": content}]
            response = await strategy.make_request(messages)
            
            print(f"✅ {case_type}")
            print(f"   内容长度: {len(content)} 字符")
            print(f"   选择服务: {strategy.current_service}")
            print(f"   回复长度: {len(response.choices[0].message.content)} 字符")
            
        except Exception as e:
            print(f"❌ {case_type} 失败: {str(e)}")

async def main():
    """主演示函数"""
    print("🔄 HarborAI 降级策略演示")
    print("=" * 60)
    
    try:
        # 基础降级演示
        await demo_basic_fallback()
        
        # 服务监控演示
        await demo_service_monitoring()
        
        # 成本优化演示
        await demo_cost_optimization()
        
        # 智能路由演示
        await demo_intelligent_routing()
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境建议:")
        print("   1. 配置多个不同层级的服务")
        print("   2. 设置合理的健康检查间隔")
        print("   3. 监控降级事件和成本变化")
        print("   4. 根据业务需求调整路由策略")
        print("   5. 建立降级事件的告警机制")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())