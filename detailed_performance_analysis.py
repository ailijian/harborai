# -*- coding: utf-8 -*-
"""
详细性能分析测试 - HarborAI vs 直接Agently调用
目标：观测每个步骤的性能开销，确保样本一致性，分析重构必要性
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
from functools import wraps
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
from Agently.agently import Agently
from harborai import HarborAI

# 加载环境变量
load_env_result = load_dotenv()
print(f"🔧 环境变量加载结果: {load_env_result}")

# 设置FAST模式环境变量（必须在导入HarborAI之前设置）
os.environ["HARBORAI_PERFORMANCE_MODE"] = "fast"
os.environ["HARBORAI_ENABLE_FAST_PATH"] = "true"
print(f"🚀 设置性能模式: FAST")
print(f"🚀 启用快速路径: true")

# 清除settings缓存以确保环境变量生效
from harborai.config.settings import get_settings
from harborai.config.performance import reset_performance_config, PerformanceMode

# 清除缓存
get_settings.cache_clear()
print(f"🔄 清除settings缓存")

# 重置性能配置
reset_performance_config(PerformanceMode.FAST)
print(f"🔄 重置性能配置为FAST模式")

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    step_name: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_delta: float
    cpu_percent: float
    thread_count: int
    additional_info: Dict[str, Any] = None

    def __post_init__(self):
        if self.additional_info is None:
            self.additional_info = {}

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """开始内存监控"""
        tracemalloc.start()
        
    def stop_monitoring(self):
        """停止内存监控"""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
    
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
    
    def get_thread_count(self) -> int:
        """获取线程数"""
        try:
            return self.process.num_threads()
        except:
            return 0
    
    @contextmanager
    def monitor_step(self, step_name: str, additional_info: Dict[str, Any] = None):
        """监控单个步骤的性能"""
        print(f"📊 开始监控步骤: {step_name}")
        
        # 记录开始状态
        start_time = time.perf_counter()
        memory_before = self.get_memory_usage()
        cpu_before = self.get_cpu_percent()
        thread_count = self.get_thread_count()
        
        try:
            yield
        finally:
            # 记录结束状态
            end_time = time.perf_counter()
            memory_after = self.get_memory_usage()
            duration = end_time - start_time
            memory_delta = memory_after - memory_before
            
            # 创建性能指标
            metrics = PerformanceMetrics(
                step_name=step_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_delta=memory_delta,
                cpu_percent=cpu_before,
                thread_count=thread_count,
                additional_info=additional_info or {}
            )
            
            self.metrics.append(metrics)
            
            print(f"✅ 步骤完成: {step_name}")
            print(f"   ⏱️  耗时: {duration:.4f}秒")
            print(f"   💾 内存变化: {memory_delta:+.2f}MB ({memory_before:.2f} → {memory_after:.2f})")
            print(f"   🖥️  CPU: {cpu_before:.1f}%")
            print(f"   🧵 线程数: {thread_count}")
            if additional_info:
                print(f"   📝 额外信息: {additional_info}")
            print()

def performance_decorator(step_name: str):
    """性能监控装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = getattr(wrapper, '_monitor', None)
            if monitor:
                with monitor.monitor_step(step_name, {"function": func.__name__, "args_count": len(args), "kwargs_count": len(kwargs)}):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

class TestConfiguration:
    """测试配置类 - 确保样本一致性"""
    
    def __init__(self):
        # 从环境变量获取配置
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        
        if not self.deepseek_api_key:
            raise ValueError("❌ 未找到 DEEPSEEK_API_KEY 环境变量")
        
        print(f"🔑 DeepSeek API Key: {self.deepseek_api_key[:10]}...")
        print(f"🌐 DeepSeek Base URL: {self.deepseek_base_url}")
        
        # 统一的测试参数
        self.model_name = "deepseek-chat"
        self.temperature = 0.1  # 低温度确保结果一致性
        self.max_tokens = 1000
        self.test_rounds = 5
        
        # 统一的测试输入
        self.test_prompt = "请分析以下文本的情感倾向：'今天天气真好，我心情很愉快，工作也很顺利。'"
        
        # 统一的JSON Schema
        self.json_schema = {
            "type": "object",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "详细的情感分析"
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "情感倾向分类"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "置信度分数"
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "关键词列表"
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "processed_at": {"type": "string"},
                        "model_used": {"type": "string"}
                    }
                }
            },
            "required": ["analysis", "sentiment", "confidence", "keywords"]
        }
        
        print(f"📋 测试配置:")
        print(f"   🤖 模型: {self.model_name}")
        print(f"   🌡️  温度: {self.temperature}")
        print(f"   📝 最大tokens: {self.max_tokens}")
        print(f"   🔄 测试轮数: {self.test_rounds}")
        print(f"   💬 测试提示: {self.test_prompt}")
        print()

class HarborAITester:
    """HarborAI测试器 - 带详细性能监控"""
    
    def __init__(self, config: TestConfiguration, monitor: PerformanceMonitor):
        self.config = config
        self.monitor = monitor
        self.client = None
        
        # 为装饰器设置监控器
        self._setup_decorators()
    
    def _setup_decorators(self):
        """设置装饰器的监控器"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_monitor'):
                attr._monitor = self.monitor
    
    @performance_decorator("HarborAI-客户端创建")
    def create_client(self):
        """创建HarborAI客户端 - 配置为FAST模式"""
        print("🚀 创建HarborAI客户端（FAST模式）...")
        
        self.client = HarborAI(
            api_key=self.config.deepseek_api_key,
            base_url=self.config.deepseek_base_url,
            model=self.config.model_name,
            performance_mode="fast"  # 显式设置FAST模式
        )
        
        print(f"✅ HarborAI客户端创建完成（FAST模式）")
        print(f"🚀 性能模式: FAST - 启用快速路径优化")
        return self.client
    
    @performance_decorator("HarborAI-参数准备")
    def prepare_parameters(self):
        """准备调用参数"""
        print("📋 准备HarborAI调用参数...")
        
        params = {
            "model": self.config.model_name,
            "messages": [{"role": "user", "content": self.config.test_prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "schema": self.config.json_schema,
                    "strict": True
                }
            },
            "structured_provider": "agently",  # 关键：指定使用Agently
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
        
        print(f"✅ 参数准备完成: {len(params)} 个参数")
        print(f"🔧 使用结构化提供者: agently")
        print(f"📋 Schema名称: sentiment_analysis")
        return params
    
    @performance_decorator("HarborAI-结构化输出调用")
    def call_structured_output(self, params):
        """调用HarborAI的结构化输出"""
        print("🎯 调用HarborAI结构化输出...")
        print(f"📋 调用参数: {list(params.keys())}")
        print(f"🚀 性能模式: {params.get('structured_provider', 'unknown')}")
        print(f"📝 响应格式: {params.get('response_format', {}).get('type', 'unknown')}")
        
        # 检查性能配置
        from harborai.config.performance import get_performance_config
        perf_config = get_performance_config()
        print(f"🔧 性能配置模式: {perf_config.mode.value}")
        print(f"🚀 快速路径启用: {perf_config.feature_flags.enable_fast_path}")
        
        # 检查快速路径条件
        response_format = params.get('response_format', {})
        structured_provider = params.get('structured_provider')
        stream = params.get('stream', False)
        
        fast_structured_conditions = [
            f"FAST模式: {perf_config.mode.value == 'fast'}",
            f"有结构化输出: {response_format and response_format.get('type') == 'json_schema'}",
            f"使用agently: {structured_provider == 'agently'}",
            f"非流式: {not stream}"
        ]
        print(f"🔍 快速结构化路径条件: {fast_structured_conditions}")
        
        response = self.client.chat.completions.create(**params)
        
        print(f"✅ HarborAI调用完成")
        print(f"🔍 响应ID: {getattr(response, 'id', 'unknown')}")
        print(f"🎯 是否快速路径: {'fast-structured' in getattr(response, 'id', '')}")
        return response
    
    @performance_decorator("HarborAI-响应解析")
    def parse_response(self, response):
        """解析响应"""
        print("🔍 解析HarborAI响应...")
        
        # 检查是否有parsed结果（结构化输出）
        if hasattr(response.choices[0].message, 'parsed') and response.choices[0].message.parsed:
            parsed_content = response.choices[0].message.parsed
            print(f"✅ 获得结构化输出结果: {len(str(parsed_content))} 字符")
            print(f"🎯 结构化结果类型: {type(parsed_content)}")
            print(f"📝 结构化结果预览: {json.dumps(parsed_content, ensure_ascii=False, indent=2)[:200]}...")
            return parsed_content
        else:
            # 回退到content解析
            content = response.choices[0].message.content
            print(f"⚠️ 未获得结构化输出，尝试解析content: {content[:100]}...")
            
            if isinstance(content, str):
                try:
                    parsed_content = json.loads(content)
                    print(f"✅ Content JSON解析成功: {len(str(parsed_content))} 字符")
                    return parsed_content
                except json.JSONDecodeError as e:
                    print(f"❌ Content JSON解析失败: {e}")
                    raise ValueError(f"无法解析响应内容为JSON: {e}")
            else:
                print(f"✅ Content直接返回: {len(str(content))} 字符")
                return content
    
    def run_single_test(self) -> Tuple[Dict[str, Any], float]:
        """运行单次测试"""
        print("🧪 开始HarborAI单次测试...")
        
        start_time = time.perf_counter()
        
        try:
            # 创建客户端
            client = self.create_client()
            
            # 准备参数
            params = self.prepare_parameters()
            
            # 调用API
            response = self.call_structured_output(params)
            
            # 解析响应
            result = self.parse_response(response)
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            print(f"✅ HarborAI单次测试完成，总耗时: {total_time:.4f}秒")
            return result, total_time
            
        except Exception as e:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"❌ HarborAI测试失败: {str(e)}")
            print(f"❌ 错误类型: {type(e).__name__}")
            import traceback
            print(f"❌ 详细错误信息:")
            traceback.print_exc()
            return None, total_time

class AgentlyTester:
    """Agently测试器 - 带详细性能监控"""
    
    def __init__(self, config: TestConfiguration, monitor: PerformanceMonitor):
        self.config = config
        self.monitor = monitor
        self.agent = None
        
        # 为装饰器设置监控器
        self._setup_decorators()
    
    def _setup_decorators(self):
        """设置装饰器的监控器"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_monitor'):
                attr._monitor = self.monitor
    
    @performance_decorator("Agently-全局配置")
    def configure_agently(self):
        """配置Agently全局设置"""
        print("⚙️ 配置Agently全局设置...")
        
        # 使用正确的Agently配置方式
        Agently.set_settings(
            "OpenAICompatible",
            {
                "base_url": self.config.deepseek_base_url,
                "model": self.config.model_name,
                "model_type": "chat",
                "auth": self.config.deepseek_api_key,
            },
        )
        
        print(f"✅ Agently全局配置完成")
    
    @performance_decorator("Agently-Agent创建")
    def create_agent(self):
        """创建Agently Agent"""
        print("🤖 创建Agently Agent...")
        
        # 使用正确的Agently创建方式
        self.agent = Agently.create_agent()
        
        print(f"✅ Agently Agent创建完成")
        return self.agent
    
    @performance_decorator("Agently-Schema转换")
    def convert_schema_to_agently(self):
        """将JSON Schema转换为Agently格式"""
        print("🔄 转换JSON Schema为Agently格式...")
        
        # 转换JSON Schema为Agently输出格式
        agently_output = {
            "analysis": ("str", "详细的情感分析"),
            "sentiment": ("str", "情感倾向分类: positive/negative/neutral"),
            "confidence": ("float", "置信度分数(0-1)"),
            "keywords": (["str"], "关键词列表"),
            "metadata": ({
                "processed_at": ("str", "处理时间"),
                "model_used": ("str", "使用的模型")
            }, "元数据信息")
        }
        
        print(f"✅ Schema转换完成: {len(agently_output)} 个字段")
        return agently_output
    
    @performance_decorator("Agently-输出格式设置")
    def set_output_format(self, agently_output):
        """设置输出格式"""
        print("📝 设置Agently输出格式...")
        
        self.agent.output(agently_output)
        
        print(f"✅ 输出格式设置完成")
    
    @performance_decorator("Agently-结构化输出调用")
    def call_structured_output(self):
        """调用结构化输出"""
        print("🎯 调用Agently结构化输出...")
        
        result = self.agent.input(self.config.test_prompt).start()
        
        print(f"✅ Agently调用完成")
        return result
    
    def run_single_test(self) -> Tuple[Dict[str, Any], float]:
        """运行单次测试"""
        print("🧪 开始Agently单次测试...")
        
        start_time = time.perf_counter()
        
        try:
            # 配置Agently
            self.configure_agently()
            
            # 创建Agent
            agent = self.create_agent()
            
            # 转换Schema
            agently_output = self.convert_schema_to_agently()
            
            # 设置输出格式
            self.set_output_format(agently_output)
            
            # 调用API
            result = self.call_structured_output()
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            print(f"✅ Agently单次测试完成，总耗时: {total_time:.4f}秒")
            return result, total_time
            
        except Exception as e:
            end_time = time.perf_counter()
            total_time = end_time - start_time
            print(f"❌ Agently测试失败: {str(e)}")
            print(f"❌ 错误类型: {type(e).__name__}")
            import traceback
            print(f"❌ 详细错误信息:")
            traceback.print_exc()
            return None, total_time

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.harborai_results = []
        self.agently_results = []
        self.harborai_metrics = []
        self.agently_metrics = []
    
    def analyze_step_performance(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """分析步骤性能"""
        if not metrics:
            return {}
        
        # 按步骤分组
        step_groups = {}
        for metric in metrics:
            if metric.step_name not in step_groups:
                step_groups[metric.step_name] = []
            step_groups[metric.step_name].append(metric)
        
        # 分析每个步骤
        step_analysis = {}
        for step_name, step_metrics in step_groups.items():
            durations = [m.duration for m in step_metrics]
            memory_deltas = [m.memory_delta for m in step_metrics]
            
            step_analysis[step_name] = {
                "count": len(step_metrics),
                "total_duration": sum(durations),
                "avg_duration": statistics.mean(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
                "total_memory_delta": sum(memory_deltas),
                "avg_memory_delta": statistics.mean(memory_deltas),
                "percentage": (sum(durations) / sum(m.duration for m in metrics)) * 100
            }
        
        return step_analysis
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        print("📊 生成详细性能报告...")
        
        # 计算总体统计
        harborai_times = [t for _, t in self.harborai_results if t > 0]
        agently_times = [t for _, t in self.agently_results if t > 0]
        
        harborai_success_count = len([r for r, _ in self.harborai_results if r is not None])
        agently_success_count = len([r for r, _ in self.agently_results if r is not None])
        
        # 分析步骤性能
        harborai_step_analysis = self.analyze_step_performance(self.harborai_metrics)
        agently_step_analysis = self.analyze_step_performance(self.agently_metrics)
        
        report = {
            "test_summary": {
                "test_time": datetime.now().isoformat(),
                "total_rounds": len(self.harborai_results),
                "harborai_success_rate": harborai_success_count / len(self.harborai_results) * 100,
                "agently_success_rate": agently_success_count / len(self.agently_results) * 100
            },
            "harborai_performance": {
                "total_time": sum(harborai_times),
                "avg_time": statistics.mean(harborai_times) if harborai_times else 0,
                "min_time": min(harborai_times) if harborai_times else 0,
                "max_time": max(harborai_times) if harborai_times else 0,
                "std_time": statistics.stdev(harborai_times) if len(harborai_times) > 1 else 0,
                "step_analysis": harborai_step_analysis
            },
            "agently_performance": {
                "total_time": sum(agently_times),
                "avg_time": statistics.mean(agently_times) if agently_times else 0,
                "min_time": min(agently_times) if agently_times else 0,
                "max_time": max(agently_times) if agently_times else 0,
                "std_time": statistics.stdev(agently_times) if len(agently_times) > 1 else 0,
                "step_analysis": agently_step_analysis
            }
        }
        
        # 计算性能对比
        if harborai_times and agently_times:
            harborai_avg = statistics.mean(harborai_times)
            agently_avg = statistics.mean(agently_times)
            
            report["performance_comparison"] = {
                "harborai_avg_time": harborai_avg,
                "agently_avg_time": agently_avg,
                "time_difference": harborai_avg - agently_avg,
                "harborai_slower_by_factor": harborai_avg / agently_avg if agently_avg > 0 else 0,
                "harborai_slower_by_percentage": ((harborai_avg - agently_avg) / agently_avg * 100) if agently_avg > 0 else 0
            }
        
        return report
    
    def identify_bottlenecks(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 分析HarborAI步骤瓶颈
        harborai_steps = report.get("harborai_performance", {}).get("step_analysis", {})
        for step_name, step_data in harborai_steps.items():
            if step_data["percentage"] > 20:  # 占用超过20%时间的步骤
                bottlenecks.append({
                    "type": "HarborAI步骤瓶颈",
                    "step": step_name,
                    "percentage": step_data["percentage"],
                    "avg_duration": step_data["avg_duration"],
                    "severity": "高" if step_data["percentage"] > 50 else "中"
                })
        
        # 分析Agently步骤瓶颈
        agently_steps = report.get("agently_performance", {}).get("step_analysis", {})
        for step_name, step_data in agently_steps.items():
            if step_data["percentage"] > 20:
                bottlenecks.append({
                    "type": "Agently步骤瓶颈",
                    "step": step_name,
                    "percentage": step_data["percentage"],
                    "avg_duration": step_data["avg_duration"],
                    "severity": "高" if step_data["percentage"] > 50 else "中"
                })
        
        return bottlenecks
    
    def generate_optimization_recommendations(self, report: Dict[str, Any], bottlenecks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []
        
        # 基于性能对比的建议
        comparison = report.get("performance_comparison", {})
        if comparison.get("harborai_slower_by_factor", 0) > 2:
            recommendations.append({
                "priority": "高",
                "category": "架构优化",
                "title": "考虑重构HarborAI的Agently集成",
                "description": f"HarborAI比直接Agently慢{comparison.get('harborai_slower_by_factor', 0):.1f}倍，建议重构",
                "specific_actions": [
                    "减少中间层调用开销",
                    "优化参数传递流程",
                    "缓存重复配置操作",
                    "考虑直接使用Agently API"
                ]
            })
        
        # 基于瓶颈的建议
        for bottleneck in bottlenecks:
            if "HarborAI" in bottleneck["type"]:
                if "客户端创建" in bottleneck["step"]:
                    recommendations.append({
                        "priority": "中",
                        "category": "客户端优化",
                        "title": "优化HarborAI客户端创建",
                        "description": f"客户端创建占用{bottleneck['percentage']:.1f}%时间",
                        "specific_actions": [
                            "实现客户端连接池",
                            "缓存客户端实例",
                            "延迟初始化非必要组件"
                        ]
                    })
                elif "结构化输出调用" in bottleneck["step"]:
                    recommendations.append({
                        "priority": "高",
                        "category": "API调用优化",
                        "title": "优化结构化输出调用",
                        "description": f"API调用占用{bottleneck['percentage']:.1f}%时间",
                        "specific_actions": [
                            "减少不必要的参数处理",
                            "优化JSON Schema转换",
                            "实现请求批处理"
                        ]
                    })
        
        # 通用优化建议
        recommendations.append({
            "priority": "中",
            "category": "监控优化",
            "title": "添加生产环境性能监控",
            "description": "建立持续的性能监控体系",
            "specific_actions": [
                "添加关键步骤的耗时监控",
                "设置性能告警阈值",
                "定期进行性能基准测试",
                "建立性能回归检测"
            ]
        })
        
        return recommendations

def main():
    """主函数"""
    print("🚀 开始详细性能分析测试")
    print("=" * 80)
    
    # 初始化组件
    config = TestConfiguration()
    monitor = PerformanceMonitor()
    analyzer = PerformanceAnalyzer()
    
    monitor.start_monitoring()
    
    try:
        # 运行多轮测试
        print(f"🔄 开始 {config.test_rounds} 轮性能对比测试")
        print("=" * 80)
        
        for round_num in range(1, config.test_rounds + 1):
            print(f"\n🏁 第 {round_num}/{config.test_rounds} 轮测试")
            print("-" * 40)
            
            # 测试HarborAI
            print("\n🌊 测试HarborAI + Agently")
            harborai_monitor = PerformanceMonitor()
            harborai_monitor.start_monitoring()
            
            harborai_tester = HarborAITester(config, harborai_monitor)
            harborai_result, harborai_time = harborai_tester.run_single_test()
            
            analyzer.harborai_results.append((harborai_result, harborai_time))
            analyzer.harborai_metrics.extend(harborai_monitor.metrics)
            harborai_monitor.stop_monitoring()
            
            # 测试直接Agently
            print("\n⚡ 测试直接Agently")
            agently_monitor = PerformanceMonitor()
            agently_monitor.start_monitoring()
            
            agently_tester = AgentlyTester(config, agently_monitor)
            agently_result, agently_time = agently_tester.run_single_test()
            
            analyzer.agently_results.append((agently_result, agently_time))
            analyzer.agently_metrics.extend(agently_monitor.metrics)
            agently_monitor.stop_monitoring()
            
            print(f"\n📊 第{round_num}轮结果:")
            print(f"   HarborAI: {harborai_time:.4f}秒 {'✅' if harborai_result else '❌'}")
            print(f"   Agently:  {agently_time:.4f}秒 {'✅' if agently_result else '❌'}")
            
            if harborai_time > 0 and agently_time > 0:
                factor = harborai_time / agently_time
                print(f"   性能差异: HarborAI比Agently慢 {factor:.2f}倍")
        
        # 生成详细报告
        print("\n" + "=" * 80)
        print("📊 生成详细性能分析报告")
        print("=" * 80)
        
        report = analyzer.generate_performance_report()
        bottlenecks = analyzer.identify_bottlenecks(report)
        recommendations = analyzer.generate_optimization_recommendations(report, bottlenecks)
        
        # 输出报告
        print("\n📈 测试总结:")
        summary = report["test_summary"]
        print(f"   测试时间: {summary['test_time']}")
        print(f"   总测试轮数: {summary['total_rounds']}")
        print(f"   HarborAI成功率: {summary['harborai_success_rate']:.1f}%")
        print(f"   Agently成功率: {summary['agently_success_rate']:.1f}%")
        
        print("\n⏱️ 性能对比:")
        if "performance_comparison" in report:
            comp = report["performance_comparison"]
            print(f"   HarborAI平均耗时: {comp['harborai_avg_time']:.4f}秒")
            print(f"   Agently平均耗时: {comp['agently_avg_time']:.4f}秒")
            print(f"   时间差异: {comp['time_difference']:.4f}秒")
            print(f"   HarborAI慢倍数: {comp['harborai_slower_by_factor']:.2f}倍")
            print(f"   HarborAI慢百分比: {comp['harborai_slower_by_percentage']:.1f}%")
        
        print("\n🔍 HarborAI步骤分析:")
        harborai_steps = report["harborai_performance"]["step_analysis"]
        for step_name, step_data in harborai_steps.items():
            print(f"   {step_name}:")
            print(f"     平均耗时: {step_data['avg_duration']:.4f}秒")
            print(f"     时间占比: {step_data['percentage']:.1f}%")
            print(f"     内存变化: {step_data['avg_memory_delta']:+.2f}MB")
        
        print("\n🔍 Agently步骤分析:")
        agently_steps = report["agently_performance"]["step_analysis"]
        for step_name, step_data in agently_steps.items():
            print(f"   {step_name}:")
            print(f"     平均耗时: {step_data['avg_duration']:.4f}秒")
            print(f"     时间占比: {step_data['percentage']:.1f}%")
            print(f"     内存变化: {step_data['avg_memory_delta']:+.2f}MB")
        
        print("\n🚨 性能瓶颈识别:")
        if bottlenecks:
            for bottleneck in bottlenecks:
                print(f"   [{bottleneck['severity']}] {bottleneck['type']}")
                print(f"     步骤: {bottleneck['step']}")
                print(f"     时间占比: {bottleneck['percentage']:.1f}%")
                print(f"     平均耗时: {bottleneck['avg_duration']:.4f}秒")
        else:
            print("   未发现明显性能瓶颈")
        
        print("\n💡 优化建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. [{rec['priority']}] {rec['title']}")
            print(f"      类别: {rec['category']}")
            print(f"      描述: {rec['description']}")
            print(f"      具体行动:")
            for action in rec['specific_actions']:
                print(f"        - {action}")
            print()
        
        # 保存详细报告
        report_data = {
            "report": report,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "raw_metrics": {
                "harborai": [asdict(m) for m in analyzer.harborai_metrics],
                "agently": [asdict(m) for m in analyzer.agently_metrics]
            }
        }
        
        with open("detailed_performance_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print(f"📄 详细报告已保存到: detailed_performance_report.json")
        
        # 重构建议总结
        print("\n" + "=" * 80)
        print("🔧 重构建议总结")
        print("=" * 80)
        
        if "performance_comparison" in report:
            factor = report["performance_comparison"]["harborai_slower_by_factor"]
            if factor > 3:
                print("❗ 强烈建议重构HarborAI的Agently集成:")
                print("   1. 性能差异过大，需要架构级优化")
                print("   2. 考虑直接使用Agently API而非包装层")
                print("   3. 如需保留包装层，需大幅优化调用链")
            elif factor > 2:
                print("⚠️ 建议优化HarborAI的Agently集成:")
                print("   1. 优化关键步骤的性能瓶颈")
                print("   2. 减少不必要的中间处理")
                print("   3. 考虑缓存和连接池优化")
            else:
                print("✅ HarborAI性能可接受，建议进行微调优化")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        monitor.stop_monitoring()
        print("\n🏁 详细性能分析测试完成")

if __name__ == "__main__":
    main()