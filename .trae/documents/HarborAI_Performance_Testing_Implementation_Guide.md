# HarborAI SDK 性能测试实施指南

## 1. 测试环境准备

### 1.1 依赖安装
```bash
# 安装性能测试依赖
pip install pytest pytest-benchmark pytest-asyncio
pip install memory-profiler line-profiler
pip install psutil aiohttp
pip install matplotlib seaborn  # 用于生成性能图表
pip install locust  # 用于负载测试
```

### 1.2 测试目录结构
```
tests/performance/
├── __init__.py
├── conftest.py                 # pytest配置
├── test_response_time.py       # 响应时间测试
├── test_concurrency.py         # 并发测试
├── test_resource_usage.py      # 资源使用测试
├── test_stability.py           # 稳定性测试
├── test_comparison.py          # 对比测试
├── utils/
│   ├── __init__.py
│   ├── metrics_collector.py    # 指标收集器
│   ├── load_generator.py       # 负载生成器
│   ├── resource_monitor.py     # 资源监控器
│   └── report_generator.py     # 报告生成器
└── data/
    ├── test_messages.json      # 测试消息数据
    └── test_schemas.json       # 测试Schema数据
```

### 1.3 测试配置文件
```yaml
# tests/performance/config.yaml
test_config:
  # 基础配置
  base_url: "https://api.deepseek.com"
  api_key: "${DEEPSEEK_API_KEY}"
  
  # 测试参数
  test_duration: 300  # 5分钟
  warmup_duration: 30  # 30秒预热
  
  # 并发配置
  max_concurrent_users: 1000
  ramp_up_time: 60
  
  # 模型配置
  test_models:
    - "deepseek-chat"
    - "deepseek-reasoner"
  
  # 性能阈值
  performance_thresholds:
    max_response_time: 1000  # ms
    min_success_rate: 99.9   # %
    max_memory_growth: 5     # %
    max_cpu_usage: 80        # %
```

## 2. 核心测试工具类

### 2.1 指标收集器
```python
# tests/performance/utils/metrics_collector.py
import time
import psutil
import threading
from typing import Dict, List, Any
from dataclasses import dataclass, field
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    response_times: List[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    error_details: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    network_io: List[Dict[str, float]] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)

class MetricsCollector:
    """性能指标收集器"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.start_time = None
        self.monitoring = False
        self.monitor_thread = None
        self.process = psutil.Process()
    
    def start_monitoring(self):
        """开始监控系统资源"""
        self.monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_resources)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_resources(self):
        """监控系统资源使用情况"""
        while self.monitoring:
            try:
                # 收集内存使用
                memory_info = self.process.memory_info()
                self.metrics.memory_usage.append(memory_info.rss / 1024 / 1024)  # MB
                
                # 收集CPU使用
                cpu_percent = self.process.cpu_percent()
                self.metrics.cpu_usage.append(cpu_percent)
                
                # 收集网络I/O
                net_io = psutil.net_io_counters()
                self.metrics.network_io.append({
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                })
                
                # 记录时间戳
                self.metrics.timestamps.append(time.time() - self.start_time)
                
                time.sleep(1)  # 每秒收集一次
            except Exception as e:
                print(f"资源监控错误: {e}")
    
    def record_request(self, response_time: float, success: bool, error_type: str = None):
        """记录单次请求结果"""
        self.metrics.response_times.append(response_time)
        if success:
            self.metrics.success_count += 1
        else:
            self.metrics.error_count += 1
            if error_type:
                self.metrics.error_details[error_type] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        response_times = self.metrics.response_times
        if not response_times:
            return {}
        
        response_times.sort()
        total_requests = len(response_times)
        
        return {
            'total_requests': total_requests,
            'success_count': self.metrics.success_count,
            'error_count': self.metrics.error_count,
            'success_rate': (self.metrics.success_count / total_requests) * 100 if total_requests > 0 else 0,
            'response_time': {
                'mean': sum(response_times) / len(response_times),
                'median': response_times[len(response_times) // 2],
                'p95': response_times[int(len(response_times) * 0.95)],
                'p99': response_times[int(len(response_times) * 0.99)],
                'min': min(response_times),
                'max': max(response_times)
            },
            'resource_usage': {
                'memory': {
                    'avg': sum(self.metrics.memory_usage) / len(self.metrics.memory_usage) if self.metrics.memory_usage else 0,
                    'peak': max(self.metrics.memory_usage) if self.metrics.memory_usage else 0
                },
                'cpu': {
                    'avg': sum(self.metrics.cpu_usage) / len(self.metrics.cpu_usage) if self.metrics.cpu_usage else 0,
                    'peak': max(self.metrics.cpu_usage) if self.metrics.cpu_usage else 0
                }
            },
            'error_details': dict(self.metrics.error_details)
        }
```

### 2.2 负载生成器
```python
# tests/performance/utils/load_generator.py
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor

class LoadGenerator:
    """负载生成器"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            headers={'Authorization': f'Bearer {self.api_key}'},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()
    
    async def single_request(self, model: str, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """发送单个请求"""
        start_time = time.time()
        try:
            payload = {
                'model': model,
                'messages': messages,
                **kwargs
            }
            
            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                response_data = await response.json()
                response_time = (time.time() - start_time) * 1000  # ms
                
                return {
                    'success': response.status == 200,
                    'response_time': response_time,
                    'status_code': response.status,
                    'data': response_data,
                    'error_type': None if response.status == 200 else f"HTTP_{response.status}"
                }
        except asyncio.TimeoutError:
            return {
                'success': False,
                'response_time': (time.time() - start_time) * 1000,
                'status_code': 0,
                'data': None,
                'error_type': 'TIMEOUT'
            }
        except Exception as e:
            return {
                'success': False,
                'response_time': (time.time() - start_time) * 1000,
                'status_code': 0,
                'data': None,
                'error_type': type(e).__name__
            }
    
    async def concurrent_requests(self, 
                                concurrent_users: int,
                                requests_per_user: int,
                                model: str,
                                messages: List[Dict],
                                **kwargs) -> List[Dict[str, Any]]:
        """并发请求测试"""
        tasks = []
        for _ in range(concurrent_users):
            for _ in range(requests_per_user):
                task = asyncio.create_task(
                    self.single_request(model, messages, **kwargs)
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'response_time': 0,
                    'status_code': 0,
                    'data': None,
                    'error_type': type(result).__name__
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def ramp_up_test(self,
                          max_users: int,
                          ramp_up_time: int,
                          test_duration: int,
                          model: str,
                          messages: List[Dict],
                          **kwargs) -> List[Dict[str, Any]]:
        """渐进式负载测试"""
        results = []
        start_time = time.time()
        
        # 计算每秒增加的用户数
        users_per_second = max_users / ramp_up_time
        current_users = 0
        
        while time.time() - start_time < test_duration:
            # 计算当前应该有多少用户
            elapsed = time.time() - start_time
            if elapsed < ramp_up_time:
                target_users = int(elapsed * users_per_second)
            else:
                target_users = max_users
            
            # 增加用户
            new_users = target_users - current_users
            if new_users > 0:
                tasks = []
                for _ in range(new_users):
                    task = asyncio.create_task(
                        self.single_request(model, messages, **kwargs)
                    )
                    tasks.append(task)
                
                # 等待这批请求完成
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)
                current_users = target_users
            
            await asyncio.sleep(1)  # 每秒检查一次
        
        return results
```

## 3. 具体测试实现

### 3.1 响应时间测试
```python
# tests/performance/test_response_time.py
import pytest
import asyncio
import time
from harborai import HarborAI
from openai import OpenAI
from .utils.metrics_collector import MetricsCollector
from .utils.load_generator import LoadGenerator

class TestResponseTime:
    """响应时间测试类"""
    
    @pytest.fixture
    def harborai_client(self):
        """HarborAI客户端fixture"""
        return HarborAI(
            api_key="test_key",
            base_url="https://api.deepseek.com"
        )
    
    @pytest.fixture
    def openai_client(self):
        """OpenAI客户端fixture"""
        return OpenAI(
            api_key="test_key",
            base_url="https://api.openai.com/v1"
        )
    
    @pytest.fixture
    def test_messages(self):
        """测试消息fixture"""
        return [
            {"role": "user", "content": "简单测试问题"},
            {"role": "user", "content": "中等长度的测试问题，包含更多上下文信息"},
            {"role": "user", "content": "复杂的长文本测试问题" * 20}
        ]
    
    def test_sync_api_response_time(self, harborai_client, test_messages, benchmark):
        """测试同步API响应时间"""
        def sync_call():
            return harborai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[test_messages[0]]
            )
        
        # 使用pytest-benchmark进行基准测试
        result = benchmark(sync_call)
        
        # 验证响应时间要求
        assert benchmark.stats['mean'] < 1.0, f"平均响应时间 {benchmark.stats['mean']:.3f}s 超过1秒阈值"
        assert result is not None, "响应结果不能为空"
    
    @pytest.mark.asyncio
    async def test_async_api_response_time(self, harborai_client, test_messages):
        """测试异步API响应时间"""
        collector = MetricsCollector()
        collector.start_monitoring()
        
        try:
            # 测试100个异步请求
            tasks = []
            for i in range(100):
                async def async_call():
                    start_time = time.time()
                    try:
                        result = await harborai_client.chat.completions.acreate(
                            model="deepseek-chat",
                            messages=[test_messages[i % len(test_messages)]]
                        )
                        response_time = (time.time() - start_time) * 1000
                        collector.record_request(response_time, True)
                        return result
                    except Exception as e:
                        response_time = (time.time() - start_time) * 1000
                        collector.record_request(response_time, False, type(e).__name__)
                        raise
                
                tasks.append(async_call())
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
        finally:
            collector.stop_monitoring()
        
        # 分析结果
        summary = collector.get_summary()
        
        # 验证性能要求
        assert summary['success_rate'] >= 99.0, f"成功率 {summary['success_rate']:.1f}% 低于99%"
        assert summary['response_time']['mean'] < 1000, f"平均响应时间 {summary['response_time']['mean']:.1f}ms 超过1秒"
        assert summary['response_time']['p95'] < 2000, f"P95响应时间 {summary['response_time']['p95']:.1f}ms 超过2秒"
    
    def test_streaming_response_time(self, harborai_client, test_messages):
        """测试流式响应时间"""
        collector = MetricsCollector()
        
        # 测试首字节时间(TTFB)
        start_time = time.time()
        first_chunk_time = None
        total_chunks = 0
        
        stream = harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[test_messages[1]],
            stream=True
        )
        
        for chunk in stream:
            if first_chunk_time is None:
                first_chunk_time = (time.time() - start_time) * 1000
            total_chunks += 1
        
        total_time = (time.time() - start_time) * 1000
        
        # 验证流式性能
        assert first_chunk_time < 500, f"首字节时间 {first_chunk_time:.1f}ms 超过500ms"
        assert total_chunks > 0, "流式响应应该包含多个chunk"
        assert total_time < 10000, f"总完成时间 {total_time:.1f}ms 超过10秒"
    
    def test_structured_output_response_time(self, harborai_client, test_messages):
        """测试结构化输出响应时间"""
        schema = {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "summary": {"type": "string"},
                "confidence": {"type": "number"}
            },
            "required": ["title", "summary", "confidence"]
        }
        
        # 测试Agently结构化输出
        start_time = time.time()
        agently_result = harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[test_messages[0]],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "TestSchema", "schema": schema}
            },
            structured_provider="agently"
        )
        agently_time = (time.time() - start_time) * 1000
        
        # 测试Native结构化输出
        start_time = time.time()
        native_result = harborai_client.chat.completions.create(
            model="deepseek-chat",
            messages=[test_messages[0]],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "TestSchema", "schema": schema}
            },
            structured_provider="native"
        )
        native_time = (time.time() - start_time) * 1000
        
        # 验证结构化输出性能
        assert agently_time < 2000, f"Agently结构化输出时间 {agently_time:.1f}ms 超过2秒"
        assert native_time < 2000, f"Native结构化输出时间 {native_time:.1f}ms 超过2秒"
        assert agently_result.parsed is not None, "Agently结构化结果不能为空"
        assert native_result.parsed is not None, "Native结构化结果不能为空"
    
    def test_reasoning_model_response_time(self, harborai_client, test_messages):
        """测试推理模型响应时间"""
        start_time = time.time()
        
        result = harborai_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[test_messages[0]]
        )
        
        response_time = (time.time() - start_time) * 1000
        
        # 验证推理模型性能
        assert response_time < 5000, f"推理模型响应时间 {response_time:.1f}ms 超过5秒"
        assert result.choices[0].message.content is not None, "推理模型应该返回内容"
        
        # 检查是否包含思考过程
        if hasattr(result.choices[0].message, 'reasoning_content'):
            assert result.choices[0].message.reasoning_content is not None, "推理模型应该包含思考过程"
```

### 3.2 并发测试
```python
# tests/performance/test_concurrency.py
import pytest
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from .utils.metrics_collector import MetricsCollector
from .utils.load_generator import LoadGenerator

class TestConcurrency:
    """并发性能测试类"""
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stability(self):
        """高并发稳定性测试"""
        collector = MetricsCollector()
        collector.start_monitoring()
        
        try:
            async with LoadGenerator("https://api.deepseek.com", "test_key") as generator:
                # 1000并发用户，每人发送1个请求
                results = await generator.concurrent_requests(
                    concurrent_users=1000,
                    requests_per_user=1,
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": "并发测试"}]
                )
                
                # 统计结果
                for result in results:
                    collector.record_request(
                        result['response_time'],
                        result['success'],
                        result.get('error_type')
                    )
        
        finally:
            collector.stop_monitoring()
        
        summary = collector.get_summary()
        
        # 验证并发性能要求
        assert summary['success_rate'] >= 99.9, f"高并发成功率 {summary['success_rate']:.1f}% 低于99.9%"
        assert summary['response_time']['p95'] < 3000, f"高并发P95响应时间 {summary['response_time']['p95']:.1f}ms 超过3秒"
        assert summary['resource_usage']['memory']['peak'] < 1000, f"内存峰值 {summary['resource_usage']['memory']['peak']:.1f}MB 过高"
    
    @pytest.mark.asyncio
    async def test_ramp_up_load(self):
        """渐进式负载测试"""
        collector = MetricsCollector()
        collector.start_monitoring()
        
        try:
            async with LoadGenerator("https://api.deepseek.com", "test_key") as generator:
                # 60秒内从0增加到500用户，然后维持300秒
                results = await generator.ramp_up_test(
                    max_users=500,
                    ramp_up_time=60,
                    test_duration=360,  # 6分钟
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": "负载测试"}]
                )
                
                # 统计结果
                for result in results:
                    if isinstance(result, dict):
                        collector.record_request(
                            result['response_time'],
                            result['success'],
                            result.get('error_type')
                        )
        
        finally:
            collector.stop_monitoring()
        
        summary = collector.get_summary()
        
        # 验证负载测试要求
        assert summary['success_rate'] >= 99.5, f"负载测试成功率 {summary['success_rate']:.1f}% 低于99.5%"
        assert len(summary['error_details']) <= 3, f"错误类型过多: {summary['error_details']}"
    
    def test_thread_pool_concurrency(self, harborai_client):
        """线程池并发测试"""
        collector = MetricsCollector()
        collector.start_monitoring()
        
        def single_request():
            start_time = time.time()
            try:
                result = harborai_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": "线程池测试"}]
                )
                response_time = (time.time() - start_time) * 1000
                collector.record_request(response_time, True)
                return result
            except Exception as e:
                response_time = (time.time() - start_time) * 1000
                collector.record_request(response_time, False, type(e).__name__)
                raise
        
        try:
            # 使用线程池执行100个并发请求
            with ThreadPoolExecutor(max_workers=50) as executor:
                futures = [executor.submit(single_request) for _ in range(100)]
                
                # 等待所有请求完成
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"请求失败: {e}")
        
        finally:
            collector.stop_monitoring()
        
        summary = collector.get_summary()
        
        # 验证线程池并发性能
        assert summary['success_rate'] >= 95.0, f"线程池并发成功率 {summary['success_rate']:.1f}% 低于95%"
        assert summary['response_time']['mean'] < 2000, f"线程池并发平均响应时间 {summary['response_time']['mean']:.1f}ms 超过2秒"
    
    @pytest.mark.asyncio
    async def test_mixed_model_concurrency(self):
        """混合模型并发测试"""
        collector = MetricsCollector()
        collector.start_monitoring()
        
        models = ["deepseek-chat", "deepseek-reasoner"]
        
        try:
            async with LoadGenerator("https://api.deepseek.com", "test_key") as generator:
                tasks = []
                
                # 为每个模型创建并发请求
                for model in models:
                    for _ in range(50):  # 每个模型50个请求
                        task = asyncio.create_task(
                            generator.single_request(
                                model=model,
                                messages=[{"role": "user", "content": f"测试{model}"}]
                            )
                        )
                        tasks.append((model, task))
                
                # 等待所有请求完成
                for model, task in tasks:
                    try:
                        result = await task
                        collector.record_request(
                            result['response_time'],
                            result['success'],
                            result.get('error_type')
                        )
                    except Exception as e:
                        collector.record_request(0, False, type(e).__name__)
        
        finally:
            collector.stop_monitoring()
        
        summary = collector.get_summary()
        
        # 验证混合模型并发性能
        assert summary['success_rate'] >= 98.0, f"混合模型并发成功率 {summary['success_rate']:.1f}% 低于98%"
        assert summary['resource_usage']['cpu']['peak'] < 90, f"CPU峰值使用率 {summary['resource_usage']['cpu']['peak']:.1f}% 过高"
```

### 3.3 资源使用测试
```python
# tests/performance/test_resource_usage.py
import pytest
import time
import gc
import tracemalloc
from memory_profiler import profile
from .utils.metrics_collector import MetricsCollector

class TestResourceUsage:
    """资源使用测试类"""
    
    def test_memory_usage_efficiency(self, harborai_client):
        """内存使用效率测试"""
        # 启动内存跟踪
        tracemalloc.start()
        
        # 记录基线内存
        gc.collect()
        baseline_snapshot = tracemalloc.take_snapshot()
        baseline_memory = sum(stat.size for stat in baseline_snapshot.statistics('filename'))
        
        # 执行大量请求
        for i in range(100):
            try:
                result = harborai_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": f"内存测试 {i}"}]
                )
                # 确保结果被使用
                _ = result.choices[0].message.content
            except Exception as e:
                print(f"请求 {i} 失败: {e}")
        
        # 强制垃圾回收
        gc.collect()
        
        # 测量最终内存
        final_snapshot = tracemalloc.take_snapshot()
        final_memory = sum(stat.size for stat in final_snapshot.statistics('filename'))
        
        # 计算内存增长
        memory_growth = (final_memory - baseline_memory) / baseline_memory * 100
        
        tracemalloc.stop()
        
        # 验证内存使用要求
        assert memory_growth < 10, f"内存增长 {memory_growth:.1f}% 超过10%阈值"
        
        # 检查内存泄漏
        top_stats = final_snapshot.compare_to(baseline_snapshot, 'lineno')
        significant_leaks = [stat for stat in top_stats[:10] if stat.size_diff > 1024 * 1024]  # 1MB
        assert len(significant_leaks) == 0, f"检测到显著内存泄漏: {significant_leaks}"
    
    def test_cpu_usage_efficiency(self, harborai_client):
        """CPU使用效率测试"""
        collector = MetricsCollector()
        collector.start_monitoring()
        
        try:
            # 执行CPU密集型操作
            start_time = time.time()
            request_count = 0
            
            while time.time() - start_time < 60:  # 运行1分钟
                try:
                    result = harborai_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[{"role": "user", "content": "CPU测试"}]
                    )
                    request_count += 1
                except Exception as e:
                    print(f"CPU测试请求失败: {e}")
                
                time.sleep(0.1)  # 短暂休息
        
        finally:
            collector.stop_monitoring()
        
        summary = collector.get_summary()
        
        # 验证CPU使用效率
        avg_cpu = summary['resource_usage']['cpu']['avg']
        peak_cpu = summary['resource_usage']['cpu']['peak']
        
        assert avg_cpu < 50, f"平均CPU使用率 {avg_cpu:.1f}% 过高"
        assert peak_cpu < 80, f"峰值CPU使用率 {peak_cpu:.1f}% 过高"
        assert request_count > 100, f"1分钟内完成的请求数 {request_count} 过少"
    
    @profile
    def test_memory_profile_detailed(self, harborai_client):
        """详细内存分析测试"""
        # 这个测试会生成详细的内存使用报告
        results = []
        
        for i in range(50):
            result = harborai_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": f"详细内存测试 {i}"}]
            )
            results.append(result)
            
            # 每10个请求清理一次
            if i % 10 == 0:
                gc.collect()
        
        # 验证结果
        assert len(results) == 50, "应该完成50个请求"
        assert all(r.choices[0].message.content for r in results), "所有请求都应该有响应内容"
    
    def test_network_resource_efficiency(self, harborai_client):
        """网络资源效率测试"""
        collector = MetricsCollector()
        collector.start_monitoring()
        
        try:
            # 测试不同大小的请求
            test_cases = [
                {"role": "user", "content": "短"},
                {"role": "user", "content": "中等长度的测试内容" * 10},
                {"role": "user", "content": "长文本测试内容" * 100}
            ]
            
            for i, message in enumerate(test_cases * 20):  # 每种类型20次
                try:
                    result = harborai_client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[message]
                    )
                    # 确保响应被处理
                    _ = len(result.choices[0].message.content)
                except Exception as e:
                    print(f"网络测试请求 {i} 失败: {e}")
        
        finally:
            collector.stop_monitoring()
        
        # 分析网络I/O效率
        if collector.metrics.network_io:
            initial_io = collector.metrics.network_io[0]
            final_io = collector.metrics.network_io[-1]
            
            bytes_sent = final_io['bytes_sent'] - initial_io['bytes_sent']
            bytes_recv = final_io['bytes_recv'] - initial_io['bytes_recv']
            
            # 验证网络效率
            assert bytes_sent > 0, "应该有数据发送"
            assert bytes_recv > 0, "应该有数据接收"
            
            # 计算网络效率指标
            total_bytes = bytes_sent + bytes_recv
            test_duration = collector.metrics.timestamps[-1] if collector.metrics.timestamps else 1
            throughput = total_bytes / test_duration / 1024  # KB/s
            
            print(f"网络吞吐量: {throughput:.2f} KB/s")
            print(f"发送数据: {bytes_sent / 1024:.2f} KB")
            print(f"接收数据: {bytes_recv / 1024:.2f} KB")
```

## 4. 测试执行脚本

### 4.1 自动化测试脚本
```bash
#!/bin/bash
# run_performance_tests.sh

echo "开始HarborAI SDK性能测试..."

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export DEEPSEEK_API_KEY="your_api_key_here"

# 创建测试结果目录
mkdir -p test_results/performance
cd test_results/performance

# 运行基础性能测试
echo "1. 运行响应时间测试..."
python -m pytest ../../tests/performance/test_response_time.py -v --benchmark-json=response_time_benchmark.json

# 运行并发测试
echo "2. 运行并发性能测试..."
python -m pytest ../../tests/performance/test_concurrency.py -v --tb=short

# 运行资源使用测试
echo "3. 运行资源使用测试..."
python -m pytest ../../tests/performance/test_resource_usage.py -v --tb=short

# 运行稳定性测试
echo "4. 运行稳定性测试..."
python -m pytest ../../tests/performance/test_stability.py -v --tb=short

# 生成性能报告
echo "5. 生成性能测试报告..."
python ../../tests/performance/utils/report_generator.py

echo "性能测试完成！结果保存在 test_results/performance/ 目录中"
```

### 4.2 持续集成配置
```yaml
# .github/workflows/performance_test.yml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # 每天凌晨2点运行
  workflow_dispatch:  # 手动触发

jobs:
  performance-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-benchmark pytest-asyncio
        pip install memory-profiler psutil
    
    - name: Run performance tests
      env:
        DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
      run: |
        chmod +x scripts/run_performance_tests.sh
        ./scripts/run_performance_tests.sh
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      with:
        name: performance-test-results
        path: test_results/performance/
    
    - name: Comment PR with results
      if: github.event_name == 'pull_request'
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const path = 'test_results/performance/summary.json';
          if (fs.existsSync(path)) {
            const results = JSON.parse(fs.readFileSync(path, 'utf8'));
            const comment = `## 性能测试结果
            
            - 平均响应时间: ${results.avg_response_time}ms
            - 成功率: ${results.success_rate}%
            - 内存使用: ${results.memory_usage}MB
            - CPU使用率: ${results.cpu_usage}%
            
            详细报告请查看 Artifacts。`;
            
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: comment
            });
          }
```

## 5. 性能监控和告警

### 5.1 实时监控脚本
```python
# scripts/performance_monitor.py
import time
import json
import psutil
import requests
from datetime import datetime
from typing import Dict, Any

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config_file: str = "monitor_config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.thresholds = self.config['thresholds']
        self.alert_webhook = self.config.get('alert_webhook')
    
    def collect_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'network_io': dict(psutil.net_io_counters()._asdict()),
            'process_count': len(psutil.pids())
        }
    
    def check_thresholds(self, metrics: Dict[str, Any]) -> list:
        """检查阈值违规"""
        violations = []
        
        if metrics['cpu_percent'] > self.thresholds['cpu_max']:
            violations.append(f"CPU使用率过高: {metrics['cpu_percent']:.1f}%")
        
        if metrics['memory_percent'] > self.thresholds['memory_max']:
            violations.append(f"内存使用率过高: {metrics['memory_percent']:.1f}%")
        
        if metrics['disk_usage'] > self.thresholds['disk_max']:
            violations.append(f"磁盘使用率过高: {metrics['disk_usage']:.1f}%")
        
        return violations
    
    def send_alert(self, violations: list):
        """发送告警"""
        if not violations or not self.alert_webhook:
            return
        
        alert_message = {
            'text': f"HarborAI性能告警:\n" + "\n".join(violations),
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            response = requests.post(self.alert_webhook, json=alert_message)
            response.raise_for_status()
        except Exception as e:
            print(f"发送告警失败: {e}")
    
    def run_monitoring(self, duration: int = 3600):
        """运行监控"""
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                metrics = self.collect_metrics()
                violations = self.check_thresholds(metrics)
                
                if violations:
                    self.send_alert(violations)
                
                # 记录指标到文件
                with open('performance_metrics.jsonl', 'a') as f:
                    f.write(json.dumps(metrics) + '\n')
                
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                print(f"监控错误: {e}")
                time.sleep(10)

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.run_monitoring()
```

### 5.2 监控配置文件
```json
{
  "thresholds": {
    "cpu_max": 80,
    "memory_max": 85,
    "disk_max": 90,
    "response_time_max": 2000,
    "error_rate_max": 5
  },
  "alert_webhook": "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
  "monitoring_interval": 60,
  "retention_days": 30
}
```

## 6. 测试结果分析工具

### 6.1 报告生成器
```python
# tests/performance/utils/report_generator.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List

class PerformanceReportGenerator:
    """性能测试报告生成器"""
    
    def __init__(self, results_dir: str = "test_results/performance"):
        self.results_dir = results_dir
        self.report_data = {}
    
    def load_test_results(self):
        """加载测试结果"""
        # 加载各种测试结果文件
        test_files = [
            'response_time_results.json',
            'concurrency_results.json',
            'resource_usage_results.json',
            'stability_results.json'
        ]
        
        for file_name in test_files:
            try:
                with open(f"{self.results_dir}/{file_name}", 'r') as f:
                    test_name = file_name.replace('_results.json', '')
                    self.report_data[test_name] = json.load(f)
            except FileNotFoundError:
                print(f"警告: 未找到测试结果文件 {file_name}")
    
    def generate_charts(self):
        """生成性能图表"""
        # 响应时间分布图
        if 'response_time' in self.report_data:
            self._create_response_time_chart()
        
        # 并发性能图
        if 'concurrency' in self.report_data:
            self._create_concurrency_chart()
        
        # 资源使用图
        if 'resource_usage' in self.report_data:
            self._create_resource_usage_chart()
    
    def _create_response_time_chart(self):
        """创建响应时间图表"""
        data = self.report_data['response_time']
        
        plt.figure(figsize=(12, 8))
        
        # 响应时间分布直方图
        plt.subplot(2, 2, 1)
        plt.hist(data.get('response_times', []), bins=50, alpha=0.7)
        plt.title('响应时间分布')
        plt.xlabel('响应时间 (ms)')
        plt.ylabel('频次')
        
        # 响应时间百分位图
        plt.subplot(2, 2, 2)
        percentiles = ['p50', 'p95', 'p99']
        values = [data.get('response_time', {}).get(p, 0) for p in percentiles]
        plt.bar(percentiles, values)
        plt.title('响应时间百分位')
        plt.ylabel('响应时间 (ms)')
        
        # 成功率饼图
        plt.subplot(2, 2, 3)
        success_count = data.get('success_count', 0)
        error_count = data.get('error_count', 0)
        plt.pie([success_count, error_count], labels=['成功', '失败'], autopct='%1.1f%%')
        plt.title('请求成功率')
        
        # 错误类型分布
        plt.subplot(2, 2, 4)
        error_details = data.get('error_details', {})
        if error_details:
            plt.bar(error_details.keys(), error_details.values())
            plt.title('错误类型分布')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/response_time_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_concurrency_chart(self):
        """创建并发性能图表"""
        data = self.report_data['concurrency']
        
        plt.figure(figsize=(12, 6))
        
        # 并发用户数 vs 响应时间
        plt.subplot(1, 2, 1)
        concurrent_users = data.get('concurrent_users', [])
        avg_response_times = data.get('avg_response_times', [])
        plt.plot(concurrent_users, avg_response_times, marker='o')
        plt.title('并发用户数 vs 平均响应时间')
        plt.xlabel('并发用户数')
        plt.ylabel('平均响应时间 (ms)')
        
        # 吞吐量图
        plt.subplot(1, 2, 2)
        throughput = data.get('throughput', [])
        plt.plot(concurrent_users, throughput, marker='s', color='green')
        plt.title('并发用户数 vs 吞吐量')
        plt.xlabel('并发用户数')
        plt.ylabel('吞吐量 (RPS)')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/concurrency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_resource_usage_chart(self):
        """创建资源使用图表"""
        data = self.report_data['resource_usage']
        
        plt.figure(figsize=(12, 8))
        
        # 内存使用趋势
        plt.subplot(2, 2, 1)
        timestamps = data.get('timestamps', [])
        memory_usage = data.get('memory_usage', [])
        plt.plot(timestamps, memory_usage)
        plt.title('内存使用趋势')
        plt.xlabel('时间 (s)')
        plt.ylabel('内存使用 (MB)')
        
        # CPU使用趋势
        plt.subplot(2, 2, 2)
        cpu_usage = data.get('cpu_usage', [])
        plt.plot(timestamps, cpu_usage, color='red')
        plt.title('CPU使用趋势')
        plt.xlabel('时间 (s)')
        plt.ylabel('CPU使用率 (%)')
        
        # 网络I/O
        plt.subplot(2, 2, 3)
        network_sent = data.get('network_sent', [])
        network_recv = data.get('network_recv', [])
        plt.plot(timestamps, network_sent, label='发送', alpha=0.7)
        plt.plot(timestamps, network_recv, label='接收', alpha=0.7)
        plt.title('网络I/O')
        plt.xlabel('时间 (s)')
        plt.ylabel('字节数')
        plt.legend()
        
        # 资源使用汇总
        plt.subplot(2, 2, 4)
        resources = ['内存峰值', 'CPU峰值', '网络总量']
        values = [
            max(memory_usage) if memory_usage else 0,
            max(cpu_usage) if cpu_usage else 0,
            sum(network_sent) + sum(network_recv) if network_sent and network_recv else 0
        ]
        plt.bar(resources, values)
        plt.title('资源使用汇总')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/resource_usage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_html_report(self):
        """生成HTML报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>HarborAI SDK 性能测试报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
                .chart { text-align: center; margin: 20px 0; }
                .success { color: green; }
                .warning { color: orange; }
                .error { color: red; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>HarborAI SDK 性能测试报告</h1>
                <p>生成时间: {timestamp}</p>
            </div>
            
            <div class="section">
                <h2>测试摘要</h2>
                {summary_content}
            </div>
            
            <div class="section">
                <h2>性能图表</h2>
                {charts_content}
            </div>
            
            <div class="section">
                <h2>详细结果</h2>
                {details_content}
            </div>
        </body>
        </html>
        """
        
        # 生成摘要内容
        summary_content = self._generate_summary_html()
        
        # 生成图表内容
        charts_content = """
        <div class="chart">
            <h3>响应时间分析</h3>
            <img src="response_time_analysis.png" alt="响应时间分析">
        </div>
        <div class="chart">
            <h3>并发性能分析</h3>
            <img src="concurrency_analysis.png" alt="并发性能分析">
        </div>
        <div class="chart">
            <h3>资源使用分析</h3>
            <img src="resource_usage_analysis.png" alt="资源使用分析">
        </div>
        """
        
        # 生成详细内容
        details_content = self._generate_details_html()
        
        # 填充模板
        html_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            summary_content=summary_content,
            charts_content=charts_content,
            details_content=details_content
        )
        
        # 保存HTML报告
        with open(f"{self.results_dir}/performance_report.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_summary_html(self) -> str:
        """生成摘要HTML"""
        summary_items = []
        
        # 响应时间摘要
        if 'response_time' in self.report_data:
            rt_data = self.report_data['response_time']
            success_rate = rt_data.get('success_rate', 0)
            avg_time = rt_data.get('response_time', {}).get('mean', 0)
            
            status_class = 'success' if success_rate >= 99 else 'warning' if success_rate >= 95 else 'error'
            summary_items.append(f'<div class="metric {status_class}">成功率: {success_rate:.1f}%</div>')
            
            status_class = 'success' if avg_time <= 1000 else 'warning' if avg_time <= 2000 else 'error'
            summary_items.append(f'<div class="metric {status_class}">平均响应时间: {avg_time:.1f}ms</div>')
        
        # 并发性能摘要
        if 'concurrency' in self.report_data:
            conc_data = self.report_data['concurrency']
            max_throughput = max(conc_data.get('throughput', [0]))
            summary_items.append(f'<div class="metric">最大吞吐量: {max_throughput:.1f} RPS</div>')
        
        # 资源使用摘要
        if 'resource_usage' in self.report_data:
            res_data = self.report_data['resource_usage']
            peak_memory = max(res_data.get('memory_usage', [0]))
            peak_cpu = max(res_data.get('cpu_usage', [0]))
            
            status_class = 'success' if peak_memory <= 500 else 'warning' if peak_memory <= 1000 else 'error'
            summary_items.append(f'<div class="metric {status_class}">内存峰值: {peak_memory:.1f}MB</div>')
            
            status_class = 'success' if peak_cpu <= 50 else 'warning' if peak_cpu <= 80 else 'error'
            summary_items.append(f'<div class="metric {status_class}">CPU峰值: {peak_cpu:.1f}%</div>')
        
        return '\n'.join(summary_items)
    
    def _generate_details_html(self) -> str:
        """生成详细结果HTML"""
        details = []
        
        for test_name, test_data in self.report_data.items():
            details.append(f"<h3>{test_name.replace('_', ' ').title()}</h3>")
            details.append("<pre>")
            details.append(json.dumps(test_data, indent=2, ensure_ascii=False))
            details.append("</pre>")
        
        return '\n'.join(details)
    
    def generate_report(self):
        """生成完整报告"""
        print("加载测试结果...")
        self.load_test_results()
        
        print("生成性能图表...")
        self.generate_charts()
        
        print("生成HTML报告...")
        self.generate_html_report()
        
        print(f"报告生成完成，保存在 {self.results_dir}/performance_report.html")

if __name__ == "__main__":
    generator = PerformanceReportGenerator()
    generator.generate_report()
```

这个实施指南提供了完整的性能测试框架，包括：

1. **测试环境准备**：依赖安装、目录结构、配置文件
2. **核心测试工具**：指标收集器、负载生成器等
3. **具体测试实现**：响应时间、并发、资源使用等测试
4. **自动化脚本**：测试执行、持续集成配置
5. **监控告警**：实时监控、阈值检查、告警通知
6. **结果分析**：图表生成、HTML报告、详细分析

通过这个框架，可以对HarborAI SDK进行全面的性能测试和评估，确保其满足企业级应用的性能要求。

---

**文档版本**: v1.0  
**创建日期**: 2025年1月3日  
**负责人**: HarborAI开发团队