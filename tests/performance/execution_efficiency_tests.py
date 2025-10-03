"""
执行效率测试模块

该模块提供API调用开销、插件加载时间等关键指标的测试功能，支持：
- API调用延迟测试
- 插件加载性能测试
- 函数执行时间分析
- 代码热点识别
- 执行路径优化建议
- 性能回归检测

作者：HarborAI性能测试团队
创建时间：2024年
"""

import time
import sys
import gc
import cProfile
import pstats
import io
import functools
import threading
import asyncio
import inspect
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import logging
from pathlib import Path
import json
import traceback

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """执行指标数据结构"""
    function_name: str
    execution_time: float  # 执行时间（秒）
    cpu_time: float  # CPU时间（秒）
    memory_before: int  # 执行前内存（bytes）
    memory_after: int  # 执行后内存（bytes）
    memory_peak: int  # 峰值内存（bytes）
    call_count: int  # 调用次数
    timestamp: datetime
    success: bool  # 是否成功执行
    error_message: Optional[str] = None
    
    @property
    def average_execution_time(self) -> float:
        """平均执行时间（对于单次执行，等于执行时间）"""
        return self.execution_time
    
    @property
    def performance_grade(self) -> str:
        """性能等级评估"""
        if not self.success:
            return 'F'
        
        # 基于执行时间评估性能等级
        if self.execution_time < 0.001:  # < 1ms
            return 'A+'
        elif self.execution_time < 0.01:  # < 10ms
            return 'A'
        elif self.execution_time < 0.1:  # < 100ms
            return 'B'
        elif self.execution_time < 1.0:  # < 1s
            return 'C'
        else:  # >= 1s
            return 'F'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'function_name': self.function_name,
            'execution_time': self.execution_time,
            'cpu_time': self.cpu_time,
            'memory_before': self.memory_before,
            'memory_after': self.memory_after,
            'memory_peak': self.memory_peak,
            'call_count': self.call_count,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class APICallMetrics:
    """API调用指标数据结构"""
    api_name: str
    method: str  # GET, POST, etc.
    url: str
    request_time: float  # 请求时间（秒）
    response_time: float  # 响应时间（秒）
    total_time: float  # 总时间（秒）
    status_code: int
    request_size: int  # 请求大小（bytes）
    response_size: int  # 响应大小（bytes）
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'api_name': self.api_name,
            'method': self.method,
            'url': self.url,
            'request_time': self.request_time,
            'response_time': self.response_time,
            'total_time': self.total_time,
            'status_code': self.status_code,
            'request_size': self.request_size,
            'response_size': self.response_size,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class PluginLoadMetrics:
    """插件加载指标数据结构"""
    plugin_name: str
    plugin_path: str
    load_time: float  # 加载时间（秒）
    initialization_time: float  # 初始化时间（秒）
    memory_usage: int  # 内存使用（bytes）
    dependencies_count: int  # 依赖数量
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'plugin_name': self.plugin_name,
            'plugin_path': self.plugin_path,
            'load_time': self.load_time,
            'initialization_time': self.initialization_time,
            'memory_usage': self.memory_usage,
            'dependencies_count': self.dependencies_count,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class PerformanceProfile:
    """性能分析结果"""
    function_name: str
    total_calls: int
    total_time: float
    cumulative_time: float
    average_time: float
    hotspots: List[Dict[str, Any]]  # 热点函数
    call_graph: Dict[str, Any]  # 调用图
    
    @property
    def peak_memory_mb(self) -> float:
        """峰值内存使用（MB）"""
        # 从热点数据中提取内存信息
        if self.hotspots and isinstance(self.hotspots[0], dict):
            memory_peak = self.hotspots[0].get('memory_peak', 0)
            return memory_peak / (1024 * 1024)  # 转换为MB
        return 0.0
    
    @property
    def efficiency_grade(self) -> str:
        """效率等级评估"""
        if self.average_time < 0.001:  # < 1ms
            return 'A+'
        elif self.average_time < 0.01:  # < 10ms
            return 'A'
        elif self.average_time < 0.1:  # < 100ms
            return 'B'
        elif self.average_time < 1.0:  # < 1s
            return 'C'
        else:  # >= 1s
            return 'F'
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'function_name': self.function_name,
            'total_calls': self.total_calls,
            'total_time': self.total_time,
            'cumulative_time': self.cumulative_time,
            'average_time': self.average_time,
            'hotspots': self.hotspots,
            'call_graph': self.call_graph
        }


class ExecutionTimer:
    """执行时间计时器"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
    
    def get_elapsed_time(self) -> float:
        """获取经过的时间"""
        if self.elapsed_time is not None:
            return self.elapsed_time
        elif self.start_time is not None:
            return time.perf_counter() - self.start_time
        else:
            return 0.0


class MemoryProfiler:
    """内存分析器"""
    
    def __init__(self):
        self.memory_before = 0
        self.memory_after = 0
        self.memory_peak = 0
        self._monitoring = False
        self._monitor_thread = None
    
    def __enter__(self):
        import psutil
        process = psutil.Process()
        self.memory_before = process.memory_info().rss
        self.memory_peak = self.memory_before
        
        # 启动内存监控线程
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitor_thread.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
        
        import psutil
        process = psutil.Process()
        self.memory_after = process.memory_info().rss
    
    def _monitor_memory(self):
        """监控内存使用"""
        import psutil
        process = psutil.Process()
        
        while self._monitoring:
            try:
                current_memory = process.memory_info().rss
                self.memory_peak = max(self.memory_peak, current_memory)
                time.sleep(0.01)  # 10ms间隔
            except Exception:
                break
    
    def get_memory_usage(self) -> Tuple[int, int, int]:
        """获取内存使用情况"""
        return self.memory_before, self.memory_after, self.memory_peak
    
    def get_profile(self) -> 'PerformanceProfile':
        """获取内存使用分析结果"""
        return PerformanceProfile(
            function_name="memory_profile",
            total_calls=1,
            total_time=0.0,
            cumulative_time=0.0,
            average_time=0.0,
            hotspots=[{
                'memory_before': self.memory_before,
                'memory_after': self.memory_after,
                'memory_peak': self.memory_peak,
                'memory_delta': self.memory_after - self.memory_before
            }],
            call_graph={}
        )


class ExecutionEfficiencyTester:
    """
    执行效率测试器
    
    功能特性：
    - 函数执行时间测量
    - API调用性能测试
    - 插件加载性能测试
    - 性能分析和热点识别
    - 性能回归检测
    """
    
    def __init__(self, max_records: int = 10000):
        """
        初始化执行效率测试器
        
        参数:
            max_records: 最大记录数量
        """
        self.max_records = max_records
        
        # 指标存储
        self.execution_metrics: deque = deque(maxlen=max_records)
        self.api_metrics: deque = deque(maxlen=max_records)
        self.plugin_metrics: deque = deque(maxlen=max_records)
        
        # 性能基线
        self.performance_baselines: Dict[str, Dict[str, float]] = {}
        
        # 统计信息
        self.total_tests = 0
        self.failed_tests = 0
        
        logger.info("执行效率测试器初始化完成")
    
    def _get_memory_usage(self) -> int:
        """获取当前内存使用量"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except Exception:
            return 0

    def measure_execution_time(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> ExecutionMetrics:
        """
        测量函数执行时间和资源使用情况
        
        参数:
            func: 要测试的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        返回:
            ExecutionMetrics: 执行指标数据
        """
        logger.info(f"开始测量函数执行效率: {func.__name__}")
        
        # 垃圾回收，确保内存测量准确
        gc.collect()
        
        # 记录执行前状态
        memory_before = self._get_memory_usage()
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        
        success = True
        error_message = None
        
        try:
            # 使用内存分析器监控峰值内存
            with MemoryProfiler() as memory_profiler:
                result = func(*args, **kwargs)
            
            # 记录执行后状态
            end_time = time.perf_counter()
            end_cpu_time = time.process_time()
            memory_after = self._get_memory_usage()
            
            # 获取峰值内存
            _, peak_memory, _ = memory_profiler.get_memory_usage()
            
        except Exception as e:
            end_time = time.perf_counter()
            end_cpu_time = time.process_time()
            memory_after = self._get_memory_usage()
            peak_memory = memory_before
            success = False
            error_message = str(e)
            logger.error(f"函数执行失败: {error_message}")
        
        # 创建执行指标
        metrics = ExecutionMetrics(
            function_name=func.__name__,
            execution_time=end_time - start_time,
            cpu_time=end_cpu_time - start_cpu_time,
            memory_before=memory_before,
            memory_after=memory_after,
            memory_peak=peak_memory,
            call_count=1,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        
        # 记录指标
        self.execution_metrics.append(metrics)
        
        logger.info(f"函数 {func.__name__} 执行完成，耗时: {metrics.execution_time:.4f}s")
        return metrics

    def measure_function_execution(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> ExecutionMetrics:
        """
        测量函数执行效率的别名方法
        
        这是measure_execution_time的别名，为了保持向后兼容性
        """
        return self.measure_execution_time(func, *args, **kwargs)

    def measure_api_call(
        self,
        api_call_func: Callable,
        api_name: str,
        method: str = "GET",
        url: str = "",
        *args,
        **kwargs
    ) -> APICallMetrics:
        """
        测量API调用性能
        
        参数:
            api_call_func: API调用函数
            api_name: API名称
            method: HTTP方法
            url: API URL
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        返回:
            API调用指标
        """
        request_start = time.perf_counter()
        
        try:
            # 执行API调用
            response = api_call_func(*args, **kwargs)
            
            response_time = time.perf_counter()
            total_time = response_time - request_start
            
            # 尝试获取响应信息
            status_code = getattr(response, 'status_code', 200)
            request_size = len(str(kwargs).encode('utf-8')) if kwargs else 0
            response_size = len(str(response).encode('utf-8')) if response else 0
            
            success = True
            error_message = None
            
        except Exception as e:
            response_time = time.perf_counter()
            total_time = response_time - request_start
            
            status_code = 0
            request_size = 0
            response_size = 0
            success = False
            error_message = str(e)
        
        # 创建API调用指标
        metrics = APICallMetrics(
            api_name=api_name,
            method=method,
            url=url,
            request_time=0.0,  # 简化实现
            response_time=total_time,
            total_time=total_time,
            status_code=status_code,
            request_size=request_size,
            response_size=response_size,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        
        # 存储指标
        self.api_metrics.append(metrics)
        self.total_tests += 1
        if not success:
            self.failed_tests += 1
        
        logger.debug(f"API {api_name} 调用时间: {metrics.total_time:.4f}s")
        
        return metrics
    
    def measure_plugin_load(
        self,
        plugin_loader_func: Callable,
        plugin_name: str,
        plugin_path: str,
        *args,
        **kwargs
    ) -> PluginLoadMetrics:
        """
        测量插件加载性能
        
        参数:
            plugin_loader_func: 插件加载函数
            plugin_name: 插件名称
            plugin_path: 插件路径
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        返回:
            插件加载指标
        """
        import psutil
        process = psutil.Process()
        memory_before = process.memory_info().rss
        
        load_start = time.perf_counter()
        
        try:
            # 执行插件加载
            plugin = plugin_loader_func(*args, **kwargs)
            
            load_time = time.perf_counter() - load_start
            
            # 测量初始化时间
            init_start = time.perf_counter()
            if hasattr(plugin, 'initialize'):
                plugin.initialize()
            init_time = time.perf_counter() - init_start
            
            # 计算内存使用
            memory_after = process.memory_info().rss
            memory_usage = memory_after - memory_before
            
            # 计算依赖数量（简化实现）
            dependencies_count = len(getattr(plugin, '__dict__', {}))
            
            success = True
            error_message = None
            
        except Exception as e:
            load_time = time.perf_counter() - load_start
            init_time = 0.0
            memory_usage = 0
            dependencies_count = 0
            success = False
            error_message = str(e)
        
        # 创建插件加载指标
        metrics = PluginLoadMetrics(
            plugin_name=plugin_name,
            plugin_path=plugin_path,
            load_time=load_time,
            initialization_time=init_time,
            memory_usage=memory_usage,
            dependencies_count=dependencies_count,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message
        )
        
        # 存储指标
        self.plugin_metrics.append(metrics)
        self.total_tests += 1
        if not success:
            self.failed_tests += 1
        
        logger.debug(f"插件 {plugin_name} 加载时间: {metrics.load_time:.4f}s")
        
        return metrics
    
    def profile_function(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> PerformanceProfile:
        """
        对函数进行性能分析
        
        参数:
            func: 要分析的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        返回:
            性能分析结果
        """
        function_name = getattr(func, '__name__', str(func))
        
        # 创建性能分析器
        profiler = cProfile.Profile()
        
        try:
            # 开始分析
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            # 获取统计信息
            stats_stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stats_stream)
            stats.sort_stats('cumulative')
            
            # 分析热点函数
            hotspots = []
            for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
                filename, line_number, func_name = func_info
                hotspots.append({
                    'function': func_name,
                    'filename': filename,
                    'line_number': line_number,
                    'call_count': cc,
                    'total_time': tt,
                    'cumulative_time': ct,
                    'average_time': tt / cc if cc > 0 else 0
                })
            
            # 按累计时间排序
            hotspots.sort(key=lambda x: x['cumulative_time'], reverse=True)
            hotspots = hotspots[:10]  # 取前10个热点
            
            # 构建调用图（简化版）
            call_graph = {}
            for func_info, (cc, nc, tt, ct, callers) in stats.stats.items():
                filename, line_number, func_name = func_info
                call_graph[func_name] = {
                    'call_count': cc,
                    'total_time': tt,
                    'callers': list(callers.keys()) if callers else []
                }
            
            # 计算总体统计
            total_calls = sum(cc for (cc, nc, tt, ct, callers) in stats.stats.values())
            total_time = sum(tt for (cc, nc, tt, ct, callers) in stats.stats.values())
            cumulative_time = sum(ct for (cc, nc, tt, ct, callers) in stats.stats.values())
            average_time = total_time / total_calls if total_calls > 0 else 0
            
        except Exception as e:
            logger.error(f"性能分析失败: {e}")
            hotspots = []
            call_graph = {}
            total_calls = 0
            total_time = 0.0
            cumulative_time = 0.0
            average_time = 0.0
        
        return PerformanceProfile(
            function_name=function_name,
            total_calls=total_calls,
            total_time=total_time,
            cumulative_time=cumulative_time,
            average_time=average_time,
            hotspots=hotspots,
            call_graph=call_graph
        )
    
    def profile_memory(self, name: str = ""):
        """
        内存分析上下文管理器
        
        参数:
            name: 分析名称
        
        返回:
            MemoryProfiler: 内存分析器实例
        """
        return MemoryProfiler()
    
    def benchmark_function(
        self,
        func: Callable,
        iterations: int = 100,
        warmup_iterations: int = 10,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        对函数进行基准测试
        
        参数:
            func: 要测试的函数
            iterations: 测试迭代次数
            warmup_iterations: 预热迭代次数
            *args: 函数参数
            **kwargs: 函数关键字参数
        
        返回:
            基准测试结果
        """
        function_name = getattr(func, '__name__', str(func))
        
        # 预热
        for _ in range(warmup_iterations):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # 强制垃圾回收
        gc.collect()
        
        # 执行基准测试
        execution_times = []
        successful_runs = 0
        
        for i in range(iterations):
            metrics = self.measure_execution_time(func, *args, **kwargs)
            if metrics.success:
                execution_times.append(metrics.execution_time)
                successful_runs += 1
        
        if not execution_times:
            return {
                'function_name': function_name,
                'iterations': iterations,
                'successful_runs': 0,
                'success_rate': 0.0,
                'error': '所有测试都失败了'
            }
        
        # 计算统计信息
        min_time = min(execution_times)
        max_time = max(execution_times)
        mean_time = statistics.mean(execution_times)
        median_time = statistics.median(execution_times)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
        
        # 计算百分位数
        sorted_times = sorted(execution_times)
        p95_time = sorted_times[int(0.95 * len(sorted_times))]
        p99_time = sorted_times[int(0.99 * len(sorted_times))]
        
        return {
            'function_name': function_name,
            'iterations': iterations,
            'successful_runs': successful_runs,
            'success_rate': successful_runs / iterations,
            'min_time': min_time,
            'max_time': max_time,
            'mean_time': mean_time,
            'median_time': median_time,
            'std_time': std_time,
            'p95_time': p95_time,
            'p99_time': p99_time,
            'throughput': successful_runs / sum(execution_times) if execution_times else 0
        }
    
    def set_performance_baseline(
        self,
        function_name: str,
        baseline_metrics: Dict[str, float]
    ) -> None:
        """设置性能基线"""
        self.performance_baselines[function_name] = baseline_metrics
        logger.info(f"为函数 {function_name} 设置性能基线")
    
    def check_performance_regression(
        self,
        function_name: str,
        current_metrics: Dict[str, float],
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """
        检查性能回归
        
        参数:
            function_name: 函数名称
            current_metrics: 当前指标
            tolerance: 容忍度（10%）
        
        返回:
            回归检查结果
        """
        if function_name not in self.performance_baselines:
            return {
                'has_baseline': False,
                'message': f'函数 {function_name} 没有性能基线'
            }
        
        baseline = self.performance_baselines[function_name]
        regressions = []
        improvements = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline:
                baseline_value = baseline[metric_name]
                change_ratio = (current_value - baseline_value) / baseline_value
                
                if change_ratio > tolerance:
                    regressions.append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_ratio': change_ratio,
                        'change_percent': change_ratio * 100
                    })
                elif change_ratio < -tolerance:
                    improvements.append({
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_ratio': change_ratio,
                        'change_percent': change_ratio * 100
                    })
        
        return {
            'has_baseline': True,
            'has_regression': len(regressions) > 0,
            'regressions': regressions,
            'improvements': improvements,
            'tolerance': tolerance
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        if not self.execution_metrics:
            return {}
        
        # 按函数分组统计
        function_stats = defaultdict(list)
        for metrics in self.execution_metrics:
            function_stats[metrics.function_name].append(metrics.execution_time)
        
        # 计算统计信息
        stats = {}
        for func_name, times in function_stats.items():
            stats[func_name] = {
                'call_count': len(times),
                'total_time': sum(times),
                'average_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_time': statistics.stdev(times) if len(times) > 1 else 0.0
            }
        
        return {
            'total_tests': self.total_tests,
            'failed_tests': self.failed_tests,
            'success_rate': (self.total_tests - self.failed_tests) / self.total_tests if self.total_tests > 0 else 0,
            'function_statistics': stats
        }
    
    def export_metrics(self, filepath: str) -> None:
        """导出指标数据"""
        data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_tests': self.total_tests,
                'failed_tests': self.failed_tests
            },
            'execution_metrics': [m.to_dict() for m in self.execution_metrics],
            'api_metrics': [m.to_dict() for m in self.api_metrics],
            'plugin_metrics': [m.to_dict() for m in self.plugin_metrics],
            'performance_baselines': self.performance_baselines
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"执行效率指标已导出到: {filepath}")


# 装饰器
def measure_performance(tester: Optional[ExecutionEfficiencyTester] = None):
    """性能测量装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal tester
            if tester is None:
                tester = ExecutionEfficiencyTester()
            
            return tester.measure_execution_time(func, *args, **kwargs)
        
        return wrapper
    return decorator


# 便捷函数
def quick_benchmark(
    func: Callable,
    iterations: int = 100,
    *args,
    **kwargs
) -> Dict[str, Any]:
    """
    快速基准测试函数
    
    参数:
        func: 要测试的函数
        iterations: 测试迭代次数
        *args: 函数参数
        **kwargs: 函数关键字参数
    
    返回:
        基准测试结果
    """
    tester = ExecutionEfficiencyTester()
    return tester.benchmark_function(func, iterations, *args, **kwargs)


if __name__ == "__main__":
    # 示例使用
    def example_function(n: int = 1000):
        """示例函数"""
        return sum(i * i for i in range(n))
    
    # 创建测试器
    tester = ExecutionEfficiencyTester()
    
    # 测量执行时间
    metrics = tester.measure_execution_time(example_function, 1000)
    print(f"执行指标: {metrics.to_dict()}")
    
    # 基准测试
    benchmark_result = tester.benchmark_function(example_function, iterations=50, n=1000)
    print(f"基准测试结果: {json.dumps(benchmark_result, indent=2, ensure_ascii=False)}")
    
    # 性能分析
    profile = tester.profile_function(example_function, 1000)
    print(f"性能分析: {profile.to_dict()}")