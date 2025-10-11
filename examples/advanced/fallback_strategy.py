#!/usr/bin/env python3
"""
降级策略演示

这个示例展示了 HarborAI 的降级策略功能，包括：
1. 多层级降级策略
2. 服务健康监控
3. 自动故障转移
4. 性能监控与告警
5. 优雅降级处理

场景：
- 主要AI服务不可用或性能下降时自动切换到备用方案
- 确保服务连续性和可用性
- 在成本和性能之间找到平衡

价值：
- 确保服务连续性和可用性
- 优化用户体验，避免服务中断
- 降低服务中断风险
- 智能选择最优服务
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# 导入配置助手
from config_helper import get_model_configs, get_primary_model_config, print_available_models

# 导入 HarborAI
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from harborai import HarborAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServiceTier(Enum):
    """服务层级"""
    PRIMARY = 1      # 主要服务
    SECONDARY = 2    # 次要服务
    FALLBACK = 3     # 降级服务
    EMERGENCY = 4    # 紧急服务

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
    vendor: str
    tier: ServiceTier
    api_key: str
    base_url: str
    temperature: float = 0.7
    cost_per_token: float = 0.0001
    expected_latency: float = 3.0
    quality_score: float = 1.0
    timeout: int = 90

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
        self.client = HarborAI()
        self.metrics = ServiceMetrics()
        self.status = ServiceStatus.HEALTHY
        
    async def health_check(self) -> bool:
        """执行健康检查"""
        try:
            start_time = time.time()
            
            # 发送简单的健康检查请求
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.model,
                messages=[{"role": "user", "content": "简单回答：你好"}],
                timeout=10
            )
            
            latency = time.time() - start_time
            
            # 估算成本
            content_length = len(response.choices[0].message.content) if response.choices else 0
            cost = content_length * self.config.cost_per_token
            
            self.metrics.add_success(latency, cost)
            self._update_status()
            
            logger.info(f"✅ {self.config.name} 健康检查通过 - 延迟: {latency:.2f}s")
            return True
            
        except Exception as e:
            self.metrics.add_failure()
            self._update_status()
            
            logger.warning(f"❌ {self.config.name} 健康检查失败: {str(e)}")
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
    
    async def make_request(self, messages: List[Dict], **kwargs) -> Any:
        """发送请求"""
        start_time = time.time()
        
        try:
            # 构建请求参数
            request_params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "timeout": self.config.timeout,
                **kwargs
            }
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create, 
                **request_params
            )
            
            latency = time.time() - start_time
            
            # 估算成本
            content_length = len(response.choices[0].message.content) if response.choices else 0
            cost = content_length * self.config.cost_per_token
            
            self.metrics.add_success(latency, cost)
            self._update_status()
            
            return response
            
        except Exception as e:
            self.metrics.add_failure()
            self._update_status()
            raise e

class FallbackStrategy:
    """降级策略"""
    
    def __init__(self):
        self.services: Dict[str, ServiceHealthChecker] = {}
        self.service_order: List[ServiceConfig] = []
        self.current_service: Optional[str] = None
        self.fallback_history: List[Dict] = []
        
        # 初始化服务配置
        self._initialize_services()
        
    def _initialize_services(self):
        """初始化服务配置"""
        model_configs = get_model_configs()
        
        if not model_configs:
            raise ValueError("没有找到可用的模型配置，请检查环境变量设置")
        
        # 为每个模型创建服务配置
        tier_mapping = {
            'deepseek': ServiceTier.PRIMARY,
            'ernie': ServiceTier.SECONDARY,
            'doubao': ServiceTier.FALLBACK
        }
        
        for i, model_config in enumerate(model_configs):
            tier = tier_mapping.get(model_config.vendor, ServiceTier.EMERGENCY)
            
            service_config = ServiceConfig(
                name=f"{model_config.vendor}_{model_config.model}",
                model=model_config.model,
                vendor=model_config.vendor,
                tier=tier,
                api_key=model_config.api_key,
                base_url=model_config.base_url,
                expected_latency=2.0 if model_config.vendor == 'deepseek' else 3.0,
                quality_score=1.0 if not model_config.is_reasoning else 1.2
            )
            
            self.services[service_config.name] = ServiceHealthChecker(service_config)
            self.service_order.append(service_config)
        
        # 按层级排序
        self.service_order.sort(key=lambda x: x.tier.value)
        
        logger.info(f"✅ 初始化了 {len(self.services)} 个服务")
        for config in self.service_order:
            logger.info(f"   - {config.name} (层级: {config.tier.name})")
        
    async def initialize(self):
        """初始化服务健康检查"""
        logger.info("🔍 开始服务健康检查...")
        
        # 对所有服务进行健康检查
        health_results = {}
        for service_name, checker in self.services.items():
            health_results[service_name] = await checker.health_check()
        
        # 选择最佳服务
        self.current_service = self._select_best_service()
        
        if self.current_service:
            logger.info(f"🎯 选择主要服务: {self.current_service}")
        else:
            logger.warning("⚠️ 没有找到可用的服务")
    
    def _select_best_service(self) -> Optional[str]:
        """选择最佳服务"""
        # 按优先级排序，选择健康的服务
        for config in self.service_order:
            checker = self.services[config.name]
            if checker.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]:
                return config.name
        
        # 如果没有健康的服务，选择成功率最高的
        available_services = [(name, checker) for name, checker in self.services.items()]
        if available_services:
            best_service = max(available_services, key=lambda x: x[1].metrics.get_success_rate())
            return best_service[0]
        
        return None
    
    async def make_request(self, messages: List[Dict], **kwargs) -> Any:
        """发送请求（带降级策略）"""
        original_service = self.current_service
        
        # 尝试所有可用服务
        for config in self.service_order:
            service_name = config.name
            checker = self.services[service_name]
            
            # 跳过离线服务
            if checker.status == ServiceStatus.OFFLINE:
                logger.debug(f"⏭️ 跳过离线服务: {service_name}")
                continue
            
            try:
                logger.info(f"🔄 尝试使用服务: {service_name}")
                response = await checker.make_request(messages, **kwargs)
                
                # 如果使用了降级服务，记录降级事件
                if service_name != original_service:
                    self._record_fallback(original_service, service_name, "success")
                    self.current_service = service_name
                    logger.info(f"🔄 降级到服务: {service_name}")
                
                return response
                
            except Exception as e:
                logger.warning(f"❌ 服务 {service_name} 请求失败: {str(e)}")
                
                # 记录降级事件
                if service_name == original_service:
                    self._record_fallback(original_service, None, "failure")
                
                continue
        
        # 所有服务都失败了
        raise Exception("所有服务都不可用")
    
    def _record_fallback(self, from_service: Optional[str], to_service: Optional[str], reason: str):
        """记录降级事件"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "from_service": from_service,
            "to_service": to_service,
            "reason": reason
        }
        self.fallback_history.append(event)
        logger.info(f"📝 记录降级事件: {from_service} -> {to_service} ({reason})")
    
    def get_service_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        status = {}
        for name, checker in self.services.items():
            status[name] = {
                "status": checker.status.value,
                "metrics": {
                    "total_requests": checker.metrics.total_requests,
                    "success_rate": checker.metrics.get_success_rate(),
                    "average_latency": checker.metrics.get_average_latency(),
                    "consecutive_failures": checker.metrics.consecutive_failures
                }
            }
        return status
    
    def get_fallback_history(self) -> List[Dict]:
        """获取降级历史"""
        return self.fallback_history.copy()

# 演示函数
async def demo_basic_fallback():
    """演示基础降级策略"""
    print("\n🔄 基础降级策略演示")
    print("=" * 50)
    
    # 创建降级策略
    strategy = FallbackStrategy()
    await strategy.initialize()
    
    # 测试请求
    test_messages = [
        [{"role": "user", "content": "什么是人工智能？"}],
        [{"role": "user", "content": "解释机器学习的概念"}],
        [{"role": "user", "content": "深度学习有什么特点？"}]
    ]
    
    print(f"\n📝 发送 {len(test_messages)} 个测试请求...")
    
    for i, messages in enumerate(test_messages, 1):
        try:
            print(f"\n🔄 请求 {i}: {messages[0]['content']}")
            response = await strategy.make_request(messages)
            
            if response and response.choices:
                content = response.choices[0].message.content
                print(f"✅ 响应: {content[:100]}...")
            else:
                print("✅ 请求成功，但无响应内容")
                
        except Exception as e:
            print(f"❌ 请求失败: {str(e)}")
    
    # 显示服务状态
    print(f"\n📊 服务状态:")
    status = strategy.get_service_status()
    for service_name, service_status in status.items():
        print(f"   {service_name}:")
        print(f"     状态: {service_status['status']}")
        print(f"     成功率: {service_status['metrics']['success_rate']:.1%}")
        print(f"     平均延迟: {service_status['metrics']['average_latency']:.2f}s")

async def demo_service_failure_simulation():
    """演示服务故障模拟"""
    print("\n🚨 服务故障模拟演示")
    print("=" * 50)
    
    strategy = FallbackStrategy()
    await strategy.initialize()
    
    # 模拟主服务故障
    if strategy.current_service:
        current_checker = strategy.services[strategy.current_service]
        print(f"🎯 当前主服务: {strategy.current_service}")
        
        # 人为增加失败次数来模拟故障
        for _ in range(3):
            current_checker.metrics.add_failure()
        current_checker._update_status()
        
        print(f"💥 模拟 {strategy.current_service} 服务故障")
        print(f"   状态变更为: {current_checker.status.value}")
    
    # 测试降级
    test_message = [{"role": "user", "content": "在服务故障情况下，这个请求应该自动降级"}]
    
    try:
        print(f"\n🔄 发送测试请求...")
        response = await strategy.make_request(test_message)
        
        if response and response.choices:
            print(f"✅ 降级成功，当前服务: {strategy.current_service}")
            print(f"   响应: {response.choices[0].message.content[:100]}...")
        
    except Exception as e:
        print(f"❌ 降级失败: {str(e)}")
    
    # 显示降级历史
    history = strategy.get_fallback_history()
    if history:
        print(f"\n📝 降级历史:")
        for event in history:
            print(f"   {event['timestamp']}: {event['from_service']} -> {event['to_service']} ({event['reason']})")

async def demo_performance_monitoring():
    """演示性能监控"""
    print("\n📊 性能监控演示")
    print("=" * 50)
    
    strategy = FallbackStrategy()
    await strategy.initialize()
    
    # 发送多个请求来收集性能数据
    test_requests = [
        "什么是云计算？",
        "解释区块链技术",
        "人工智能的应用领域",
        "机器学习算法分类",
        "深度学习的优势"
    ]
    
    print(f"📝 发送 {len(test_requests)} 个请求收集性能数据...")
    
    for i, prompt in enumerate(test_requests, 1):
        try:
            messages = [{"role": "user", "content": prompt}]
            start_time = time.time()
            
            response = await strategy.make_request(messages)
            
            elapsed = time.time() - start_time
            print(f"   请求 {i}: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"   请求 {i}: 失败 - {str(e)}")
        
        # 短暂延迟
        await asyncio.sleep(0.5)
    
    # 显示详细性能统计
    print(f"\n📊 详细性能统计:")
    status = strategy.get_service_status()
    
    for service_name, service_status in status.items():
        metrics = service_status['metrics']
        print(f"\n   {service_name}:")
        print(f"     总请求数: {metrics['total_requests']}")
        print(f"     成功率: {metrics['success_rate']:.1%}")
        print(f"     平均延迟: {metrics['average_latency']:.2f}s")
        print(f"     连续失败: {metrics['consecutive_failures']}")

async def demo_adaptive_strategy():
    """演示自适应策略"""
    print("\n🧠 自适应策略演示")
    print("=" * 50)
    
    strategy = FallbackStrategy()
    await strategy.initialize()
    
    print("🔄 测试自适应服务选择...")
    
    # 模拟不同类型的请求
    request_types = [
        ("简单问答", "什么是AI？"),
        ("复杂分析", "分析人工智能对未来社会的影响，包括技术、经济、伦理等多个维度"),
        ("创意生成", "写一首关于科技发展的诗"),
        ("代码解释", "解释Python中的装饰器概念"),
        ("翻译任务", "将'Hello World'翻译成中文")
    ]
    
    for request_type, prompt in request_types:
        print(f"\n📝 {request_type}: {prompt}")
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await strategy.make_request(messages)
            
            if response and response.choices:
                print(f"✅ 使用服务: {strategy.current_service}")
                print(f"   响应长度: {len(response.choices[0].message.content)} 字符")
            
        except Exception as e:
            print(f"❌ 请求失败: {str(e)}")
        
        await asyncio.sleep(1)

async def main():
    """主演示函数"""
    print("🔄 HarborAI 降级策略演示")
    print("=" * 60)
    
    # 显示可用模型配置
    print_available_models()
    
    try:
        # 基础降级策略演示
        await demo_basic_fallback()
        
        # 服务故障模拟演示
        await demo_service_failure_simulation()
        
        # 性能监控演示
        await demo_performance_monitoring()
        
        # 自适应策略演示
        await demo_adaptive_strategy()
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境建议:")
        print("   1. 实施实时健康检查和监控")
        print("   2. 配置合理的降级阈值")
        print("   3. 建立告警机制")
        print("   4. 定期评估服务性能")
        print("   5. 实现智能路由策略")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())