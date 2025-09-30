#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 结构化输出性能基准测试
专门测试FAST模式下的性能优化效果

测试目标：
1. 验证FAST模式下的性能接近直接Agently调用
2. 测试客户端池、Schema缓存等优化组件的效果
3. 建立性能基准数据
4. 识别性能瓶颈并提供优化建议

遵循TDD原则：
- 先写失败测试（性能目标）
- 实现优化
- 验证性能提升
"""

import os
import sys
import time
import json
import statistics
import tracemalloc
import psutil
import threading
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# 设置控制台编码
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# 导入依赖
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置FAST模式环境变量（必须在导入HarborAI之前设置）
os.environ["HARBORAI_PERFORMANCE_MODE"] = "fast"
os.environ["HARBORAI_ENABLE_FAST_PATH"] = "true"
os.environ["HARBORAI_ENABLE_CLIENT_POOL"] = "true"
os.environ["HARBORAI_ENABLE_SCHEMA_CACHE"] = "true"
os.environ["HARBORAI_ENABLE_CONFIG_CACHE"] = "true"

print(f"🚀 设置性能模式: FAST")
print(f"🚀 启用所有优化组件")

# 导入测试库
from agently import Agently
from harborai import HarborAI

# 清除缓存以确保环境变量生效
from harborai.config.settings import get_settings
from harborai.config.performance import reset_performance_config, PerformanceMode

get_settings.cache_clear()
reset_performance_config(PerformanceMode.FAST)


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    test_name: str
    avg_duration: float
    min_duration: float
    max_duration: float
    std_deviation: float
    success_rate: float
    memory_usage: float
    cpu_usage: float
    iterations: int
    cache_hit_rate: Optional[float] = None
    client_pool_hit_rate: Optional[float] = None


class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.results: List[BenchmarkResult] = []
        
        # 测试配置
        self.test_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "用户姓名"},
                "age": {"type": "integer", "description": "用户年龄"},
                "email": {"type": "string", "description": "用户邮箱"},
                "skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "技能列表"
                }
            },
            "required": ["name", "age", "email"]
        }
        
        self.test_query = "请生成一个软件工程师的个人信息，包括姓名、年龄、邮箱和技能列表"
        
        # 性能目标（基于TDD原则的失败测试）
        self.performance_targets = {
            "harborai_fast_avg_duration": 2.0,  # 目标：平均2秒内完成
            "harborai_vs_agently_ratio": 1.2,   # 目标：不超过直接Agently调用的1.2倍
            "cache_hit_rate_after_warmup": 0.8,  # 目标：预热后80%缓存命中率
            "client_pool_hit_rate": 0.9,         # 目标：90%客户端池命中率
        }
    
    def get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_cpu_percent(self) -> float:
        """获取CPU使用率"""
        try:
            return self.process.cpu_percent()
        except:
            return 0.0
    
    @contextmanager
    def monitor_performance(self):
        """性能监控上下文管理器"""
        tracemalloc.start()
        start_memory = self.get_memory_usage()
        start_cpu = self.get_cpu_percent()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self.get_memory_usage()
            end_cpu = self.get_cpu_percent()
            
            if tracemalloc.is_tracing():
                tracemalloc.stop()
    
    def test_agently_direct(self, iterations: int = 5) -> BenchmarkResult:
        """测试直接Agently调用性能（作为基准）"""
        print(f"\n🔍 测试直接Agently调用性能 (iterations={iterations})")
        
        durations = []
        success_count = 0
        memory_usage = 0
        cpu_usage = 0
        
        # 配置Agently
        Agently.set_settings(
            "OpenAICompatible",
            {
                "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                "model": "doubao-1-5-pro-32k-character-250715",
                "model_type": "chat",
                "auth": os.getenv("DOUBAO_API_KEY"),
            },
        )
        
        # 转换Schema为Agently格式
        agently_output = {
            "output": {
                "name": ("str", "用户姓名"),
                "age": ("int", "用户年龄"),
                "email": ("str", "用户邮箱"),
                "skills": ("[str]", "技能列表")
            }
        }
        
        for i in range(iterations):
            print(f"  执行第 {i+1}/{iterations} 次测试...")
            
            with self.monitor_performance():
                start_time = time.time()
                
                try:
                    agent = Agently.create_agent()
                    result = (
                        agent
                        .input(self.test_query)
                        .output(agently_output)
                        .start()
                    )
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    durations.append(duration)
                    success_count += 1
                    
                    memory_usage += self.get_memory_usage()
                    cpu_usage += self.get_cpu_percent()
                    
                    print(f"    ✅ 成功，耗时: {duration:.2f}s")
                    
                except Exception as e:
                    print(f"    ❌ 失败: {e}")
        
        if not durations:
            raise RuntimeError("所有Agently测试都失败了")
        
        return BenchmarkResult(
            test_name="Agently直接调用",
            avg_duration=statistics.mean(durations),
            min_duration=min(durations),
            max_duration=max(durations),
            std_deviation=statistics.stdev(durations) if len(durations) > 1 else 0,
            success_rate=success_count / iterations,
            memory_usage=memory_usage / iterations,
            cpu_usage=cpu_usage / iterations,
            iterations=iterations
        )
    
    def test_harborai_fast_mode(self, iterations: int = 5) -> BenchmarkResult:
        """测试HarborAI FAST模式性能"""
        print(f"\n🚀 测试HarborAI FAST模式性能 (iterations={iterations})")
        
        durations = []
        success_count = 0
        memory_usage = 0
        cpu_usage = 0
        
        # 创建HarborAI客户端
        client = HarborAI(
            api_key=os.getenv("DOUBAO_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )
        
        for i in range(iterations):
            print(f"  执行第 {i+1}/{iterations} 次测试...")
            
            with self.monitor_performance():
                start_time = time.time()
                
                try:
                    response = client.chat.completions.create(
                        model="doubao-1-5-pro-32k-character-250715",
                        messages=[
                            {"role": "user", "content": self.test_query}
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {
                                "name": "user_info",
                                "schema": self.test_schema
                            }
                        },
                        structured_provider="agently",
                        temperature=0.1
                    )
                    
                    end_time = time.time()
                    duration = end_time - start_time
                    durations.append(duration)
                    success_count += 1
                    
                    memory_usage += self.get_memory_usage()
                    cpu_usage += self.get_cpu_percent()
                    
                    print(f"    ✅ 成功，耗时: {duration:.2f}s")
                    
                except Exception as e:
                    print(f"    ❌ 失败: {e}")
        
        if not durations:
            raise RuntimeError("所有HarborAI测试都失败了")
        
        # 尝试获取缓存统计信息
        cache_hit_rate = None
        client_pool_hit_rate = None
        
        try:
            from harborai.core.parameter_cache import get_parameter_cache_manager
            from harborai.core.agently_client_pool import get_agently_client_pool
            
            cache_manager = get_parameter_cache_manager()
            if cache_manager:
                schema_stats = cache_manager.schema_cache.get_stats()
                if schema_stats.get('total_requests', 0) > 0:
                    cache_hit_rate = schema_stats.get('cache_hits', 0) / schema_stats.get('total_requests', 1)
            
            client_pool = get_agently_client_pool()
            if client_pool:
                pool_stats = client_pool.get_stats()
                if pool_stats.get('total_requests', 0) > 0:
                    client_pool_hit_rate = pool_stats.get('cache_hits', 0) / pool_stats.get('total_requests', 1)
        except Exception as e:
            print(f"    ⚠️ 无法获取缓存统计信息: {e}")
        
        return BenchmarkResult(
            test_name="HarborAI FAST模式",
            avg_duration=statistics.mean(durations),
            min_duration=min(durations),
            max_duration=max(durations),
            std_deviation=statistics.stdev(durations) if len(durations) > 1 else 0,
            success_rate=success_count / iterations,
            memory_usage=memory_usage / iterations,
            cpu_usage=cpu_usage / iterations,
            iterations=iterations,
            cache_hit_rate=cache_hit_rate,
            client_pool_hit_rate=client_pool_hit_rate
        )
    
    def test_cache_warmup_effect(self, warmup_iterations: int = 3, test_iterations: int = 5) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """测试缓存预热效果"""
        print(f"\n🔥 测试缓存预热效果")
        print(f"  预热轮次: {warmup_iterations}, 测试轮次: {test_iterations}")
        
        # 冷启动测试
        cold_result = self.test_harborai_fast_mode(iterations=test_iterations)
        cold_result.test_name = "HarborAI FAST模式 (冷启动)"
        
        # 预热
        print(f"\n🔥 执行缓存预热...")
        self.test_harborai_fast_mode(iterations=warmup_iterations)
        
        # 预热后测试
        warm_result = self.test_harborai_fast_mode(iterations=test_iterations)
        warm_result.test_name = "HarborAI FAST模式 (预热后)"
        
        return cold_result, warm_result
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """运行完整的性能基准测试"""
        print("🎯 开始HarborAI结构化输出性能基准测试")
        print("=" * 60)
        
        results = {}
        
        try:
            # 1. 测试直接Agently调用（基准）
            agently_result = self.test_agently_direct(iterations=5)
            results['agently_baseline'] = agently_result
            self.results.append(agently_result)
            
            # 2. 测试HarborAI FAST模式
            harborai_result = self.test_harborai_fast_mode(iterations=5)
            results['harborai_fast'] = harborai_result
            self.results.append(harborai_result)
            
            # 3. 测试缓存预热效果
            cold_result, warm_result = self.test_cache_warmup_effect(warmup_iterations=3, test_iterations=5)
            results['harborai_cold'] = cold_result
            results['harborai_warm'] = warm_result
            self.results.extend([cold_result, warm_result])
            
            # 4. 性能分析
            analysis = self.analyze_performance(results)
            results['analysis'] = analysis
            
            # 5. TDD验证（检查是否达到性能目标）
            tdd_results = self.verify_performance_targets(results)
            results['tdd_verification'] = tdd_results
            
            return results
            
        except Exception as e:
            print(f"❌ 基准测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
    
    def analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能结果"""
        print(f"\n📊 性能分析")
        print("-" * 40)
        
        analysis = {}
        
        # 基本性能对比
        agently_baseline = results.get('agently_baseline')
        harborai_fast = results.get('harborai_fast')
        
        if agently_baseline and harborai_fast:
            performance_ratio = harborai_fast.avg_duration / agently_baseline.avg_duration
            analysis['performance_ratio'] = performance_ratio
            
            print(f"📈 性能对比:")
            print(f"  Agently基准: {agently_baseline.avg_duration:.2f}s")
            print(f"  HarborAI FAST: {harborai_fast.avg_duration:.2f}s")
            print(f"  性能比率: {performance_ratio:.2f}x")
            
            if performance_ratio <= 1.2:
                print(f"  ✅ 性能优秀 (≤1.2x)")
            elif performance_ratio <= 1.5:
                print(f"  ⚠️ 性能可接受 (≤1.5x)")
            else:
                print(f"  ❌ 性能需要优化 (>1.5x)")
        
        # 缓存效果分析
        harborai_cold = results.get('harborai_cold')
        harborai_warm = results.get('harborai_warm')
        
        if harborai_cold and harborai_warm:
            cache_improvement = (harborai_cold.avg_duration - harborai_warm.avg_duration) / harborai_cold.avg_duration
            analysis['cache_improvement'] = cache_improvement
            
            print(f"\n🔥 缓存效果:")
            print(f"  冷启动: {harborai_cold.avg_duration:.2f}s")
            print(f"  预热后: {harborai_warm.avg_duration:.2f}s")
            print(f"  性能提升: {cache_improvement*100:.1f}%")
            
            if harborai_warm.cache_hit_rate:
                print(f"  缓存命中率: {harborai_warm.cache_hit_rate*100:.1f}%")
            
            if harborai_warm.client_pool_hit_rate:
                print(f"  客户端池命中率: {harborai_warm.client_pool_hit_rate*100:.1f}%")
        
        return analysis
    
    def verify_performance_targets(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """验证性能目标（TDD原则）"""
        print(f"\n🎯 TDD性能目标验证")
        print("-" * 40)
        
        verification = {}
        passed_tests = 0
        total_tests = 0
        
        harborai_fast = results.get('harborai_fast')
        harborai_warm = results.get('harborai_warm')
        analysis = results.get('analysis', {})
        
        # 测试1: HarborAI FAST模式平均响应时间
        total_tests += 1
        target = self.performance_targets['harborai_fast_avg_duration']
        actual = harborai_fast.avg_duration if harborai_fast else float('inf')
        passed = actual <= target
        
        verification['avg_duration_test'] = {
            'target': f"≤{target}s",
            'actual': f"{actual:.2f}s",
            'passed': passed
        }
        
        if passed:
            passed_tests += 1
            print(f"  ✅ 平均响应时间: {actual:.2f}s ≤ {target}s")
        else:
            print(f"  ❌ 平均响应时间: {actual:.2f}s > {target}s")
        
        # 测试2: HarborAI vs Agently性能比率
        total_tests += 1
        target = self.performance_targets['harborai_vs_agently_ratio']
        actual = analysis.get('performance_ratio', float('inf'))
        passed = actual <= target
        
        verification['performance_ratio_test'] = {
            'target': f"≤{target}x",
            'actual': f"{actual:.2f}x",
            'passed': passed
        }
        
        if passed:
            passed_tests += 1
            print(f"  ✅ 性能比率: {actual:.2f}x ≤ {target}x")
        else:
            print(f"  ❌ 性能比率: {actual:.2f}x > {target}x")
        
        # 测试3: 缓存命中率
        if harborai_warm and harborai_warm.cache_hit_rate is not None:
            total_tests += 1
            target = self.performance_targets['cache_hit_rate_after_warmup']
            actual = harborai_warm.cache_hit_rate
            passed = actual >= target
            
            verification['cache_hit_rate_test'] = {
                'target': f"≥{target*100:.0f}%",
                'actual': f"{actual*100:.1f}%",
                'passed': passed
            }
            
            if passed:
                passed_tests += 1
                print(f"  ✅ 缓存命中率: {actual*100:.1f}% ≥ {target*100:.0f}%")
            else:
                print(f"  ❌ 缓存命中率: {actual*100:.1f}% < {target*100:.0f}%")
        
        # 测试4: 客户端池命中率
        if harborai_warm and harborai_warm.client_pool_hit_rate is not None:
            total_tests += 1
            target = self.performance_targets['client_pool_hit_rate']
            actual = harborai_warm.client_pool_hit_rate
            passed = actual >= target
            
            verification['client_pool_hit_rate_test'] = {
                'target': f"≥{target*100:.0f}%",
                'actual': f"{actual*100:.1f}%",
                'passed': passed
            }
            
            if passed:
                passed_tests += 1
                print(f"  ✅ 客户端池命中率: {actual*100:.1f}% ≥ {target*100:.0f}%")
            else:
                print(f"  ❌ 客户端池命中率: {actual*100:.1f}% < {target*100:.0f}%")
        
        # 总体结果
        verification['summary'] = {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'overall_passed': passed_tests == total_tests
        }
        
        print(f"\n📊 TDD验证结果: {passed_tests}/{total_tests} 通过")
        if passed_tests == total_tests:
            print(f"  🎉 所有性能目标达成！")
        else:
            print(f"  ⚠️ 需要进一步优化")
        
        return verification
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成性能报告"""
        report = []
        report.append("# HarborAI 结构化输出性能基准测试报告")
        report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 测试概述
        report.append("## 测试概述")
        report.append("本次测试验证HarborAI FAST模式下结构化输出的性能优化效果")
        report.append("")
        
        # 测试结果
        report.append("## 测试结果")
        for name, result in results.items():
            if isinstance(result, BenchmarkResult):
                report.append(f"### {result.test_name}")
                report.append(f"- 平均耗时: {result.avg_duration:.2f}s")
                report.append(f"- 最小耗时: {result.min_duration:.2f}s")
                report.append(f"- 最大耗时: {result.max_duration:.2f}s")
                report.append(f"- 标准差: {result.std_deviation:.2f}s")
                report.append(f"- 成功率: {result.success_rate*100:.1f}%")
                report.append(f"- 内存使用: {result.memory_usage:.1f}MB")
                if result.cache_hit_rate is not None:
                    report.append(f"- 缓存命中率: {result.cache_hit_rate*100:.1f}%")
                if result.client_pool_hit_rate is not None:
                    report.append(f"- 客户端池命中率: {result.client_pool_hit_rate*100:.1f}%")
                report.append("")
        
        # 性能分析
        if 'analysis' in results:
            analysis = results['analysis']
            report.append("## 性能分析")
            if 'performance_ratio' in analysis:
                report.append(f"- HarborAI vs Agently性能比率: {analysis['performance_ratio']:.2f}x")
            if 'cache_improvement' in analysis:
                report.append(f"- 缓存预热性能提升: {analysis['cache_improvement']*100:.1f}%")
            report.append("")
        
        # TDD验证结果
        if 'tdd_verification' in results:
            verification = results['tdd_verification']
            report.append("## TDD性能目标验证")
            summary = verification.get('summary', {})
            report.append(f"- 通过测试: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}")
            report.append(f"- 成功率: {summary.get('success_rate', 0)*100:.1f}%")
            report.append(f"- 整体结果: {'✅ 通过' if summary.get('overall_passed', False) else '❌ 需要优化'}")
            report.append("")
        
        return "\n".join(report)


def main():
    """主函数"""
    print("🎯 HarborAI 结构化输出性能基准测试")
    print("=" * 60)
    
    # 检查环境变量
    if not os.getenv("DOUBAO_API_KEY"):
        print("❌ 错误: 未设置 DOUBAO_API_KEY 环境变量")
        return
    
    # 创建基准测试器
    benchmark = PerformanceBenchmark()
    
    try:
        # 运行基准测试
        results = benchmark.run_full_benchmark()
        
        # 生成报告
        report = benchmark.generate_report(results)
        
        # 保存报告
        report_file = "performance_benchmark_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 报告已保存到: {report_file}")
        
        # 保存详细结果
        results_file = "performance_benchmark_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # 转换BenchmarkResult对象为字典
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, BenchmarkResult):
                    serializable_results[key] = asdict(value)
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"📊 详细结果已保存到: {results_file}")
        
        # 返回TDD验证结果
        tdd_verification = results.get('tdd_verification', {})
        summary = tdd_verification.get('summary', {})
        
        if summary.get('overall_passed', False):
            print(f"\n🎉 所有性能目标达成！")
            return 0
        else:
            print(f"\n⚠️ 部分性能目标未达成，需要进一步优化")
            return 1
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)