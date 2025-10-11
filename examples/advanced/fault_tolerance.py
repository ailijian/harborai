#!/usr/bin/env python3
"""
容错与重试机制演示

这个示例展示了 HarborAI 的容错与重试机制，包括：
1. 指数退避重试策略
2. 断路器模式实现
3. 请求超时处理
4. 错误分类与统计
5. 健康检查机制

场景：
- 网络不稳定、API服务偶发故障的生产环境
- 需要自动恢复和容错保障的关键业务
- 提升系统稳定性和可靠性

价值：
- 提升系统稳定性和可靠性
- 减少因临时故障导致的失败
- 自动恢复机制，减少人工干预
- 生产环境必备的容错保障
"""

import asyncio
import time
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
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

class ErrorType(Enum):
    """错误类型枚举"""
    NETWORK_ERROR = "network_error"
    API_ERROR = "api_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    UNKNOWN_ERROR = "unknown_error"

class CircuitState(Enum):
    """断路器状态"""
    CLOSED = "closed"      # 正常状态
    OPEN = "open"          # 断开状态
    HALF_OPEN = "half_open"  # 半开状态

@dataclass
class RetryConfig:
    """重试配置"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    timeout: float = 90.0

@dataclass
class CircuitBreakerConfig:
    """断路器配置"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3

@dataclass
class ErrorStats:
    """错误统计"""
    total_requests: int = 0
    total_failures: int = 0
    error_counts: Dict[ErrorType, int] = field(default_factory=dict)
    last_error_time: Optional[datetime] = None
    
    def add_error(self, error_type: ErrorType):
        """添加错误记录"""
        self.total_failures += 1
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.last_error_time = datetime.now()
    
    def add_success(self):
        """添加成功记录"""
        self.total_requests += 1
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_requests == 0:
            return 0.0
        return (self.total_requests - self.total_failures) / self.total_requests

class CircuitBreaker:
    """断路器实现"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        
    def can_execute(self) -> bool:
        """检查是否可以执行请求"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """记录成功"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
    
    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        if self.last_failure_time is None:
            return True
        return (datetime.now() - self.last_failure_time).total_seconds() >= self.config.recovery_timeout

class FaultTolerantClient:
    """容错客户端"""
    
    def __init__(self, 
                 model_name: Optional[str] = None,
                 retry_config: Optional[RetryConfig] = None,
                 circuit_config: Optional[CircuitBreakerConfig] = None):
        self.client = HarborAI()
        self.model_name = model_name or get_primary_model_config().model
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(circuit_config or CircuitBreakerConfig())
        self.error_stats = ErrorStats()
        
    def _classify_error(self, error: Exception) -> ErrorType:
        """分类错误类型"""
        error_str = str(error).lower()
        if "timeout" in error_str or "read operation timed out" in error_str:
            return ErrorType.TIMEOUT_ERROR
        elif "rate limit" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT_ERROR
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK_ERROR
        elif "api" in error_str or "400" in error_str or "500" in error_str:
            return ErrorType.API_ERROR
        else:
            return ErrorType.UNKNOWN_ERROR
    
    def _calculate_delay(self, attempt: int) -> float:
        """计算重试延迟"""
        delay = self.retry_config.base_delay * (self.retry_config.exponential_base ** (attempt - 1))
        delay = min(delay, self.retry_config.max_delay)
        
        if self.retry_config.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    async def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """带重试的执行函数"""
        last_error = None
        
        for attempt in range(1, self.retry_config.max_attempts + 1):
            # 检查断路器状态
            if not self.circuit_breaker.can_execute():
                raise Exception("断路器已打开 - 服务不可用")
            
            try:
                # 执行请求
                result = await func(*args, **kwargs)
                
                # 记录成功
                self.circuit_breaker.record_success()
                self.error_stats.add_success()
                
                return result
                
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                
                # 记录错误
                self.circuit_breaker.record_failure()
                self.error_stats.add_error(error_type)
                
                logger.warning(f"尝试 {attempt} 失败: {error_type.value} - {str(e)}")
                
                # 如果是最后一次尝试，直接抛出异常
                if attempt == self.retry_config.max_attempts:
                    break
                
                # 计算延迟并等待
                delay = self._calculate_delay(attempt)
                logger.info(f"将在 {delay:.2f} 秒后重试...")
                await asyncio.sleep(delay)
        
        # 所有重试都失败了
        raise last_error
    
    async def chat_completion(self, messages: List[Dict], model: Optional[str] = None, **kwargs) -> Any:
        """容错的聊天完成"""
        # 使用传入的模型或默认模型
        model_to_use = model or self.model_name
        
        async def _chat():
            return await asyncio.to_thread(
                self.client.chat.completions.create,
                model=model_to_use,
                messages=messages,
                timeout=self.retry_config.timeout,
                **kwargs
            )
        
        return await self._execute_with_retry(_chat)
    
    def get_health_status(self) -> Dict:
        """获取健康状态"""
        return {
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "success_rate": self.error_stats.get_success_rate(),
            "total_requests": self.error_stats.total_requests,
            "total_failures": self.error_stats.total_failures,
            "error_breakdown": {k.value: v for k, v in self.error_stats.error_counts.items()},
            "last_error_time": self.error_stats.last_error_time.isoformat() if self.error_stats.last_error_time else None
        }

class HealthChecker:
    """健康检查器"""
    
    def __init__(self, client: FaultTolerantClient, check_interval: float = 30.0):
        self.client = client
        self.check_interval = check_interval
        self.is_running = False
        self.health_history: List[Dict] = []
    
    async def start_monitoring(self):
        """开始健康监控"""
        self.is_running = True
        logger.info("健康监控已启动")
        
        while self.is_running:
            try:
                # 执行健康检查
                start_time = time.time()
                await self.client.chat_completion([
                    {"role": "user", "content": "健康检查"}
                ])
                response_time = time.time() - start_time
                
                # 记录健康状态
                health_status = self.client.get_health_status()
                health_status["response_time"] = response_time
                health_status["timestamp"] = datetime.now().isoformat()
                
                self.health_history.append(health_status)
                
                # 保持历史记录在合理范围内
                if len(self.health_history) > 100:
                    self.health_history = self.health_history[-100:]
                
                logger.info(f"健康检查完成 - 响应时间: {response_time:.2f}s")
                
            except Exception as e:
                logger.error(f"健康检查失败: {str(e)}")
            
            await asyncio.sleep(self.check_interval)
    
    def stop_monitoring(self):
        """停止健康监控"""
        self.is_running = False
        logger.info("健康监控已停止")
    
    def get_health_summary(self) -> Dict:
        """获取健康摘要"""
        if not self.health_history:
            return {"status": "no_data"}
        
        recent_checks = self.health_history[-10:]  # 最近10次检查
        avg_response_time = sum(check.get("response_time", 0) for check in recent_checks) / len(recent_checks)
        avg_success_rate = sum(check.get("success_rate", 0) for check in recent_checks) / len(recent_checks)
        
        return {
            "status": "healthy" if avg_success_rate > 0.9 else "degraded" if avg_success_rate > 0.5 else "unhealthy",
            "average_response_time": avg_response_time,
            "average_success_rate": avg_success_rate,
            "total_checks": len(self.health_history),
            "circuit_breaker_state": recent_checks[-1].get("circuit_breaker_state", "unknown")
        }

# 演示函数
async def demo_basic_retry():
    """演示基础重试功能"""
    print("\n🔄 基础重试演示")
    print("=" * 50)
    
    # 创建容错客户端
    client = FaultTolerantClient(
        retry_config=RetryConfig(max_attempts=3, base_delay=0.5)
    )
    
    test_messages = [
        {"role": "user", "content": "用一句话解释人工智能"}
    ]
    
    try:
        start_time = time.time()
        response = await client.chat_completion(test_messages)
        end_time = time.time()
        
        print(f"✅ 请求成功")
        print(f"⏱️  响应时间: {end_time - start_time:.2f}s")
        print(f"📝 回复: {response.choices[0].message.content[:100]}...")
        
        # 显示健康状态
        health = client.get_health_status()
        print(f"📊 成功率: {health['success_rate']:.1%}")
        print(f"🔧 断路器状态: {health['circuit_breaker_state']}")
        
    except Exception as e:
        print(f"❌ 请求失败: {str(e)}")

async def demo_circuit_breaker():
    """演示断路器功能"""
    print("\n🔧 断路器演示")
    print("=" * 50)
    
    # 创建容错客户端（较低的失败阈值用于演示）
    client = FaultTolerantClient(
        circuit_config=CircuitBreakerConfig(failure_threshold=2, recovery_timeout=5.0)
    )
    
    test_messages = [
        {"role": "user", "content": "测试断路器"}
    ]
    
    print("🔄 发送正常请求测试断路器...")
    
    # 发送几个正常请求
    for i in range(3):
        try:
            print(f"🔄 尝试请求 {i+1}")
            response = await client.chat_completion(test_messages)
            health = client.get_health_status()
            print(f"✅ 请求成功 - 断路器状态: {health['circuit_breaker_state']}")
        except Exception as e:
            health = client.get_health_status()
            print(f"❌ 请求失败: {str(e)[:50]}...")
            print(f"🔧 断路器状态: {health['circuit_breaker_state']}")
            
            if health['circuit_breaker_state'] == 'open':
                print("⚠️  断路器已打开，停止请求")
                break
        
        await asyncio.sleep(1)
    
    # 显示最终状态
    health = client.get_health_status()
    print(f"\n📊 最终状态:")
    print(f"   断路器状态: {health['circuit_breaker_state']}")
    print(f"   成功率: {health['success_rate']:.1%}")
    print(f"   总请求数: {health['total_requests']}")

async def demo_error_classification():
    """演示错误分类和统计"""
    print("\n📊 错误分类演示")
    print("=" * 50)
    
    client = FaultTolerantClient(
        retry_config=RetryConfig(max_attempts=2)
    )
    
    # 模拟不同类型的错误统计
    print("📝 模拟错误分类统计...")
    
    # 手动添加一些错误统计用于演示
    client.error_stats.add_error(ErrorType.TIMEOUT_ERROR)
    client.error_stats.add_error(ErrorType.RATE_LIMIT_ERROR)
    client.error_stats.add_error(ErrorType.API_ERROR)
    client.error_stats.add_error(ErrorType.NETWORK_ERROR)
    
    # 添加一些成功记录
    for _ in range(5):
        client.error_stats.add_success()
    
    # 显示错误统计
    health = client.get_health_status()
    print(f"\n📊 错误统计:")
    print(f"   总请求数: {health['total_requests']}")
    print(f"   总失败数: {health['total_failures']}")
    print(f"   成功率: {health['success_rate']:.1%}")
    print(f"   错误分类:")
    for error_type, count in health['error_breakdown'].items():
        print(f"     - {error_type}: {count}次")

async def demo_health_monitoring():
    """演示健康监控"""
    print("\n🏥 健康监控演示")
    print("=" * 50)
    
    client = FaultTolerantClient(
        retry_config=RetryConfig(max_attempts=2)
    )
    
    health_checker = HealthChecker(client, check_interval=2.0)
    
    # 启动健康监控（后台任务）
    monitoring_task = asyncio.create_task(health_checker.start_monitoring())
    
    try:
        # 运行一段时间的监控
        print("🔄 开始健康监控（8秒）...")
        await asyncio.sleep(8)
        
        # 停止监控
        health_checker.stop_monitoring()
        monitoring_task.cancel()
        
        # 显示健康摘要
        summary = health_checker.get_health_summary()
        print(f"\n📊 健康摘要:")
        print(f"   状态: {summary.get('status', 'unknown')}")
        print(f"   平均响应时间: {summary.get('average_response_time', 0):.2f}s")
        print(f"   平均成功率: {summary.get('average_success_rate', 0):.1%}")
        print(f"   总检查次数: {summary.get('total_checks', 0)}")
        print(f"   断路器状态: {summary.get('circuit_breaker_state', 'unknown')}")
        
    except asyncio.CancelledError:
        pass

async def demo_performance_comparison():
    """演示性能对比"""
    print("\n⚡ 性能对比演示")
    print("=" * 50)
    
    # 普通客户端
    normal_client = HarborAI()
    
    # 容错客户端
    fault_tolerant_client = FaultTolerantClient(
        retry_config=RetryConfig(max_attempts=2, base_delay=0.1)
    )
    
    test_messages = [
        {"role": "user", "content": "简单测试"}
    ]
    
    # 测试普通客户端
    print("🔄 测试普通客户端...")
    normal_times = []
    for i in range(3):
        try:
            start_time = time.time()
            await asyncio.to_thread(
                normal_client.chat.completions.create,
                model=fault_tolerant_client.model_name,
                messages=test_messages
            )
            end_time = time.time()
            normal_times.append(end_time - start_time)
            print(f"   请求 {i+1}: {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"❌ 普通客户端请求 {i+1} 失败: {str(e)[:50]}...")
    
    # 测试容错客户端
    print("\n🔄 测试容错客户端...")
    fault_tolerant_times = []
    for i in range(3):
        try:
            start_time = time.time()
            await fault_tolerant_client.chat_completion(test_messages)
            end_time = time.time()
            fault_tolerant_times.append(end_time - start_time)
            print(f"   请求 {i+1}: {end_time - start_time:.2f}s")
        except Exception as e:
            print(f"❌ 容错客户端请求 {i+1} 失败: {str(e)[:50]}...")
    
    # 性能对比
    if normal_times and fault_tolerant_times:
        avg_normal = sum(normal_times) / len(normal_times)
        avg_fault_tolerant = sum(fault_tolerant_times) / len(fault_tolerant_times)
        
        print(f"\n📊 性能对比:")
        print(f"   普通客户端平均响应时间: {avg_normal:.2f}s")
        print(f"   容错客户端平均响应时间: {avg_fault_tolerant:.2f}s")
        
        if avg_normal > 0:
            overhead = ((avg_fault_tolerant - avg_normal) / avg_normal * 100)
            print(f"   性能开销: {overhead:.1f}%")
    
    # 显示容错统计
    health = fault_tolerant_client.get_health_status()
    print(f"   容错客户端成功率: {health['success_rate']:.1%}")

async def demo_advanced_retry_strategies():
    """演示高级重试策略"""
    print("\n🚀 高级重试策略演示")
    print("=" * 50)
    
    # 不同的重试配置
    strategies = [
        ("快速重试", RetryConfig(max_attempts=3, base_delay=0.1, max_delay=1.0)),
        ("标准重试", RetryConfig(max_attempts=3, base_delay=1.0, max_delay=10.0)),
        ("保守重试", RetryConfig(max_attempts=5, base_delay=2.0, max_delay=30.0))
    ]
    
    test_messages = [{"role": "user", "content": "测试重试策略"}]
    
    for strategy_name, config in strategies:
        print(f"\n🔄 测试 {strategy_name}:")
        print(f"   最大尝试次数: {config.max_attempts}")
        print(f"   基础延迟: {config.base_delay}s")
        print(f"   最大延迟: {config.max_delay}s")
        
        client = FaultTolerantClient(retry_config=config)
        
        try:
            start_time = time.time()
            response = await client.chat_completion(test_messages)
            end_time = time.time()
            
            print(f"   ✅ 成功 - 总时间: {end_time - start_time:.2f}s")
            
        except Exception as e:
            print(f"   ❌ 失败: {str(e)[:50]}...")
        
        # 显示健康状态
        health = client.get_health_status()
        print(f"   📊 成功率: {health['success_rate']:.1%}")

async def main():
    """主演示函数"""
    print("🔄 HarborAI 容错与重试机制演示")
    print("=" * 60)
    
    # 显示可用模型配置
    print_available_models()
    
    try:
        # 基础重试演示
        await demo_basic_retry()
        
        # 断路器演示
        await demo_circuit_breaker()
        
        # 错误分类演示
        await demo_error_classification()
        
        # 健康监控演示
        await demo_health_monitoring()
        
        # 性能对比演示
        await demo_performance_comparison()
        
        # 高级重试策略演示
        await demo_advanced_retry_strategies()
        
        print("\n✅ 所有演示完成！")
        print("\n💡 生产环境建议:")
        print("   1. 根据实际网络环境调整重试参数")
        print("   2. 设置合适的断路器阈值")
        print("   3. 集成监控和告警系统")
        print("   4. 定期分析错误统计数据")
        print("   5. 建立故障恢复流程")
        print("   6. 使用90秒超时配置应对网络延迟")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 运行演示
    asyncio.run(main())