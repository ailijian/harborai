# -*- coding: utf-8 -*-
"""
性能测试夹具模块
提供性能测试相关的夹具和工具
"""

import pytest
import time
import psutil
import threading
import asyncio
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import statistics
import json
import os
from pathlib import Path
from harborai import HarborAI
from harborai.utils.exceptions import HarborAIError


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    response_time: float
    memory_usage: int  # bytes
    cpu_usage: float  # percentage
    throughput: float  # requests per second
    error_rate: float  # percentage
    concurrent_requests: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'response_time': self.response_time,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
            'concurrent_requests': self.concurrent_requests,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class PerformanceThresholds:
    """性能阈值配置"""
    max_response_time: float = 2.0  # 秒
    max_memory_usage: int = 500 * 1024 * 1024  # 500MB
    max_cpu_usage: float = 80.0  # 百分比
    min_throughput: float = 50.0  # 每秒请求数
    max_error_rate: float = 0.01  # 1%
    max_concurrent_requests: int = 100
    
    def validate(self, metrics: PerformanceMetrics) -> Dict[str, bool]:
        """验证性能指标是否符合阈值"""
        return {
            'response_time': metrics.response_time <= self.max_response_time,
            'memory_usage': metrics.memory_usage <= self.max_memory_usage,
            'cpu_usage': metrics.cpu_usage <= self.max_cpu_usage,
            'throughput': metrics.throughput >= self.min_throughput,
            'error_rate': metrics.error_rate <= self.max_error_rate,
            'concurrent_requests': metrics.concurrent_requests <= self.max_concurrent_requests
        }
    
    def get_violations(self, metrics: PerformanceMetrics) -> List[str]:
        """获取违反阈值的指标列表"""
        violations = []
        validation_results = self.validate(metrics)
        
        for metric, passed in validation_results.items():
            if not passed:
                violations.append(metric)
        
        return violations


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.process = psutil.Process()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
    
    def start_monitoring(self, interval: float = 0.1):
        """开始性能监控"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self.start_time = time.time()
        
        def monitor():
            while self._monitoring:
                try:
                    metrics = self._collect_metrics()
                    with self._lock:
                        self.metrics_history.append(metrics)
                    time.sleep(interval)
                except Exception:
                    # 忽略监控过程中的错误
                    pass
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """停止性能监控"""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self.end_time = time.time()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集当前性能指标"""
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        return PerformanceMetrics(
            response_time=0.0,  # 将在具体测试中设置
            memory_usage=memory_info.rss,
            cpu_usage=cpu_percent,
            throughput=0.0,  # 将在具体测试中计算
            error_rate=0.0,  # 将在具体测试中计算
            concurrent_requests=0  # 将在具体测试中设置
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能监控摘要"""
        if not self.metrics_history:
            return {}
        
        response_times = [m.response_time for m in self.metrics_history if m.response_time > 0]
        memory_usages = [m.memory_usage for m in self.metrics_history]
        cpu_usages = [m.cpu_usage for m in self.metrics_history if m.cpu_usage > 0]
        throughputs = [m.throughput for m in self.metrics_history if m.throughput > 0]
        
        summary = {
            'duration': self.end_time - self.start_time if self.end_time and self.start_time else 0,
            'total_samples': len(self.metrics_history)
        }
        
        if response_times:
            summary['response_time'] = {
                'min': min(response_times),
                'max': max(response_times),
                'mean': statistics.mean(response_times),
                'median': statistics.median(response_times),
                'p95': statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times),
                'p99': statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times)
            }
        
        if memory_usages:
            summary['memory_usage'] = {
                'min': min(memory_usages),
                'max': max(memory_usages),
                'mean': statistics.mean(memory_usages),
                'median': statistics.median(memory_usages)
            }
        
        if cpu_usages:
            summary['cpu_usage'] = {
                'min': min(cpu_usages),
                'max': max(cpu_usages),
                'mean': statistics.mean(cpu_usages),
                'median': statistics.median(cpu_usages)
            }
        
        if throughputs:
            summary['throughput'] = {
                'min': min(throughputs),
                'max': max(throughputs),
                'mean': statistics.mean(throughputs),
                'median': statistics.median(throughputs)
            }
        
        return summary
    
    def clear(self):
        """清除监控数据"""
        with self._lock:
            self.metrics_history.clear()
        self.start_time = None
        self.end_time = None


class LoadTestRunner:
    """负载测试运行器"""
    
    def __init__(self, target_function: Callable, max_workers: int = 10):
        self.target_function = target_function
        self.max_workers = max_workers
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Exception] = []
    
    async def run_async_load_test(
        self,
        duration: float,
        requests_per_second: float,
        **kwargs
    ) -> Dict[str, Any]:
        """运行异步负载测试"""
        start_time = time.time()
        end_time = start_time + duration
        interval = 1.0 / requests_per_second
        
        tasks = []
        request_count = 0
        
        while time.time() < end_time:
            if len(tasks) < self.max_workers:
                task = asyncio.create_task(self._execute_async_request(request_count, **kwargs))
                tasks.append(task)
                request_count += 1
                
                # 控制请求频率
                await asyncio.sleep(interval)
            else:
                # 等待一些任务完成
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)
                
                for task in done:
                    try:
                        result = await task
                        self.results.append(result)
                    except Exception as e:
                        self.errors.append(e)
        
        # 等待剩余任务完成
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    self.errors.append(result)
                else:
                    self.results.append(result)
        
        return self._calculate_load_test_summary()
    
    def run_sync_load_test(
        self,
        duration: float,
        requests_per_second: float,
        **kwargs
    ) -> Dict[str, Any]:
        """运行同步负载测试"""
        import concurrent.futures
        
        start_time = time.time()
        end_time = start_time + duration
        interval = 1.0 / requests_per_second
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            request_count = 0
            
            while time.time() < end_time:
                if len(futures) < self.max_workers:
                    future = executor.submit(self._execute_sync_request, request_count, **kwargs)
                    futures.append(future)
                    request_count += 1
                    
                    # 控制请求频率
                    time.sleep(interval)
                else:
                    # 等待一些任务完成
                    done_futures = []
                    for future in futures[:]:
                        if future.done():
                            done_futures.append(future)
                            futures.remove(future)
                    
                    for future in done_futures:
                        try:
                            result = future.result()
                            self.results.append(result)
                        except Exception as e:
                            self.errors.append(e)
            
            # 等待剩余任务完成
            for future in futures:
                try:
                    result = future.result(timeout=10.0)
                    self.results.append(result)
                except Exception as e:
                    self.errors.append(e)
        
        return self._calculate_load_test_summary()
    
    async def _execute_async_request(self, request_id: int, **kwargs) -> Dict[str, Any]:
        """执行异步请求"""
        start_time = time.time()
        
        try:
            if asyncio.iscoroutinefunction(self.target_function):
                result = await self.target_function(**kwargs)
            else:
                result = self.target_function(**kwargs)
            
            end_time = time.time()
            
            return {
                'request_id': request_id,
                'success': True,
                'response_time': end_time - start_time,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            end_time = time.time()
            
            return {
                'request_id': request_id,
                'success': False,
                'response_time': end_time - start_time,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_sync_request(self, request_id: int, **kwargs) -> Dict[str, Any]:
        """执行同步请求"""
        start_time = time.time()
        
        try:
            result = self.target_function(**kwargs)
            end_time = time.time()
            
            return {
                'request_id': request_id,
                'success': True,
                'response_time': end_time - start_time,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            end_time = time.time()
            
            return {
                'request_id': request_id,
                'success': False,
                'response_time': end_time - start_time,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _calculate_load_test_summary(self) -> Dict[str, Any]:
        """计算负载测试摘要"""
        total_requests = len(self.results)
        successful_requests = len([r for r in self.results if r['success']])
        failed_requests = total_requests - successful_requests
        
        if total_requests == 0:
            return {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'error_rate': 0.0,
                'average_response_time': 0.0
            }
        
        response_times = [r['response_time'] for r in self.results]
        successful_response_times = [r['response_time'] for r in self.results if r['success']]
        
        summary = {
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'error_rate': failed_requests / total_requests,
            'average_response_time': statistics.mean(response_times) if response_times else 0.0
        }
        
        if successful_response_times:
            summary.update({
                'min_response_time': min(successful_response_times),
                'max_response_time': max(successful_response_times),
                'median_response_time': statistics.median(successful_response_times),
                'p95_response_time': statistics.quantiles(successful_response_times, n=20)[18] if len(successful_response_times) >= 20 else max(successful_response_times),
                'p99_response_time': statistics.quantiles(successful_response_times, n=100)[98] if len(successful_response_times) >= 100 else max(successful_response_times)
            })
        
        return summary
    
    def clear(self):
        """清除测试结果"""
        self.results.clear()
        self.errors.clear()


@pytest.fixture(scope='session')
def performance_thresholds():
    """性能阈值配置夹具"""
    return PerformanceThresholds()


@pytest.fixture(scope='function')
def performance_monitor():
    """性能监控器夹具"""
    monitor = PerformanceMonitor()
    yield monitor
    monitor.stop_monitoring()
    monitor.clear()


@pytest.fixture(scope='function')
def load_test_runner():
    """负载测试运行器夹具工厂"""
    def create_runner(target_function: Callable, max_workers: int = 10) -> LoadTestRunner:
        return LoadTestRunner(target_function, max_workers)
    
    return create_runner


@pytest.fixture(scope='function')
def performance_baseline_loader():
    """性能基线加载器夹具"""
    def load_baseline(test_name: str) -> Optional[Dict[str, Any]]:
        baseline_file = Path(f"tests/data/performance_baselines/{test_name}.json")
        
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return None
        
        return None
    
    return load_baseline


@pytest.fixture(scope='function')
def performance_baseline_saver():
    """性能基线保存器夹具"""
    def save_baseline(test_name: str, baseline_data: Dict[str, Any]) -> bool:
        baseline_dir = Path("tests/data/performance_baselines")
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        baseline_file = baseline_dir / f"{test_name}.json"
        
        try:
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(baseline_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
    
    return save_baseline


@pytest.fixture(scope='function')
def performance_comparator():
    """性能比较器夹具"""
    def compare_with_baseline(
        current_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        tolerance: float = 0.1
    ) -> Dict[str, Any]:
        """与基线性能进行比较"""
        comparison_result = {
            'passed': True,
            'details': {},
            'summary': ''
        }
        
        metrics_to_compare = [
            'average_response_time',
            'p95_response_time',
            'p99_response_time',
            'error_rate',
            'throughput'
        ]
        
        violations = []
        
        for metric in metrics_to_compare:
            if metric in current_metrics and metric in baseline_metrics:
                current_value = current_metrics[metric]
                baseline_value = baseline_metrics[metric]
                
                if baseline_value == 0:
                    continue
                
                # 对于响应时间和错误率，当前值应该不超过基线值的(1+tolerance)
                # 对于吞吐量，当前值应该不低于基线值的(1-tolerance)
                if metric in ['average_response_time', 'p95_response_time', 'p99_response_time', 'error_rate']:
                    threshold = baseline_value * (1 + tolerance)
                    passed = current_value <= threshold
                    comparison_result['details'][metric] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'threshold': threshold,
                        'passed': passed,
                        'change_percent': ((current_value - baseline_value) / baseline_value) * 100
                    }
                elif metric == 'throughput':
                    threshold = baseline_value * (1 - tolerance)
                    passed = current_value >= threshold
                    comparison_result['details'][metric] = {
                        'current': current_value,
                        'baseline': baseline_value,
                        'threshold': threshold,
                        'passed': passed,
                        'change_percent': ((current_value - baseline_value) / baseline_value) * 100
                    }
                
                if not passed:
                    violations.append(metric)
                    comparison_result['passed'] = False
        
        if violations:
            comparison_result['summary'] = f"性能回归检测失败，违反指标: {', '.join(violations)}"
        else:
            comparison_result['summary'] = "性能回归检测通过"
        
        return comparison_result
    
    return compare_with_baseline


@pytest.fixture(scope='function')
def memory_profiler():
    """内存分析器夹具"""
    try:
        from memory_profiler import profile
        
        class MemoryProfiler:
            def __init__(self):
                self.profiles = []
            
            def profile_function(self, func: Callable) -> Callable:
                """装饰器：分析函数内存使用"""
                def wrapper(*args, **kwargs):
                    import io
                    import sys
                    from contextlib import redirect_stdout
                    
                    # 捕获内存分析输出
                    f = io.StringIO()
                    with redirect_stdout(f):
                        result = profile(func)(*args, **kwargs)
                    
                    profile_output = f.getvalue()
                    self.profiles.append({
                        'function': func.__name__,
                        'profile': profile_output,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    return result
                
                return wrapper
            
            def get_profiles(self) -> List[Dict[str, Any]]:
                return self.profiles.copy()
            
            def clear(self):
                self.profiles.clear()
        
        profiler = MemoryProfiler()
        yield profiler
        profiler.clear()
    
    except ImportError:
        # 如果memory_profiler未安装，提供一个空的实现
        class DummyMemoryProfiler:
            def profile_function(self, func: Callable) -> Callable:
                return func
            
            def get_profiles(self) -> List[Dict[str, Any]]:
                return []
            
            def clear(self):
                pass
        
        yield DummyMemoryProfiler()


@pytest.fixture(scope='function')
def performance_report_generator():
    """性能报告生成器夹具"""
    def generate_report(
        test_name: str,
        metrics: Dict[str, Any],
        thresholds: PerformanceThresholds,
        baseline_comparison: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成性能测试报告"""
        report_lines = [
            f"# 性能测试报告: {test_name}",
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 测试指标"
        ]
        
        # 添加基本指标
        for key, value in metrics.items():
            if isinstance(value, dict):
                report_lines.append(f"### {key}")
                for sub_key, sub_value in value.items():
                    report_lines.append(f"- {sub_key}: {sub_value}")
            else:
                report_lines.append(f"- {key}: {value}")
        
        # 添加阈值验证结果
        if hasattr(metrics, 'response_time'):
            mock_metrics = PerformanceMetrics(
                response_time=metrics.get('average_response_time', 0),
                memory_usage=metrics.get('memory_usage', {}).get('max', 0),
                cpu_usage=metrics.get('cpu_usage', {}).get('max', 0),
                throughput=metrics.get('throughput', {}).get('mean', 0),
                error_rate=metrics.get('error_rate', 0),
                concurrent_requests=metrics.get('concurrent_requests', 0)
            )
            
            violations = thresholds.get_violations(mock_metrics)
            
            report_lines.extend([
                "",
                "## 阈值验证"
            ])
            
            if violations:
                report_lines.append("❌ 阈值验证失败")
                for violation in violations:
                    report_lines.append(f"- {violation}: 超出阈值")
            else:
                report_lines.append("✅ 所有指标均在阈值范围内")
        
        # 添加基线比较结果
        if baseline_comparison:
            report_lines.extend([
                "",
                "## 基线比较",
                f"结果: {'✅ 通过' if baseline_comparison['passed'] else '❌ 失败'}",
                f"摘要: {baseline_comparison['summary']}"
            ])
            
            if baseline_comparison['details']:
                report_lines.append("### 详细比较")
                for metric, details in baseline_comparison['details'].items():
                    status = '✅' if details['passed'] else '❌'
                    change = details['change_percent']
                    report_lines.append(
                        f"- {metric}: {status} 当前值={details['current']:.4f}, "
                        f"基线值={details['baseline']:.4f}, 变化={change:+.2f}%"
                    )
        
        return "\n".join(report_lines)
    
    return generate_report