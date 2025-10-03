# HarborAI 性能测试框架

## 概述

HarborAI性能测试框架是一个全面的性能测试解决方案，专为HarborAI项目设计。该框架提供了多维度的性能测试能力，包括响应时间、并发处理、内存泄漏检测、资源利用率监控和执行效率测试。

## 核心组件

### 1. 核心框架 (`core_performance_framework.py`)
- **PerformanceTestController**: 主控制器，协调各类性能测试
- **ResultsCollector**: 测试结果收集和聚合
- **PerformanceConfig**: 性能测试配置管理

### 2. 内存泄漏检测 (`memory_leak_detector.py`)
- **MemoryLeakDetector**: 长期内存监控和泄漏检测
- 支持连续监控、趋势分析和优化建议

### 3. 资源利用率监控 (`resource_utilization_monitor.py`)
- **ResourceUtilizationMonitor**: 全面的系统资源监控
- 监控CPU、内存、磁盘、网络、GPU等资源使用情况

### 4. 执行效率测试 (`execution_efficiency_tests.py`)
- **ExecutionEfficiencyTester**: 函数执行效率测试
- 支持性能分析、基准测试和回归检查

### 5. 响应时间测试 (`response_time_tests.py`)
- **ResponseTimeTester**: API响应时间测试
- 支持同步、异步、流式和并发响应时间测试

### 6. 并发处理能力测试 (`concurrency_tests.py`)
- **ConcurrencyTester**: 并发处理能力测试
- 支持多线程、多进程和异步并发测试

### 7. 性能报告生成 (`performance_report_generator.py`)
- **PerformanceReportGenerator**: 生成HTML和JSON格式的性能报告
- 支持多种图表类型和详细的性能分析

### 8. 集成测试 (`test_integration.py`)
- **IntegratedPerformanceTestSuite**: 集成测试套件
- 协调所有组件进行综合性能测试

## 快速开始

### 1. 运行完整性能测试

```bash
# 运行所有性能测试
python tests/performance/run_performance_tests.py --all --output ./reports

# 运行快速集成测试
python tests/performance/run_performance_tests.py --quick-test --output ./reports
```

### 2. 运行特定测试模块

```bash
# 响应时间测试
python tests/performance/run_performance_tests.py --module response_time --url https://api.example.com

# 并发处理能力测试
python tests/performance/run_performance_tests.py --module concurrency --users 100 --requests 50

# 内存泄漏检测
python tests/performance/run_performance_tests.py --module memory_leak --duration 300

# 资源利用率监控
python tests/performance/run_performance_tests.py --module resource_monitoring --duration 120

# 执行效率测试
python tests/performance/run_performance_tests.py --module execution_efficiency
```

### 3. 运行集成测试

```bash
# 完整集成测试
python tests/performance/run_performance_tests.py --integration --duration 300 --users 50 --output ./reports
```

## 编程接口使用

### 1. 响应时间测试

```python
from tests.performance.response_time_tests import test_api_response_time, test_async_api_response_time

# 同步API测试
metrics = test_api_response_time(
    "https://api.example.com/endpoint",
    num_requests=100,
    test_name="API响应时间测试"
)
print(f"平均响应时间: {metrics.average_response_time:.3f}s")
print(f"性能等级: {metrics.performance_grade}")

# 异步API测试
import asyncio
async def test_async():
    metrics = await test_async_api_response_time(
        "https://api.example.com/endpoint",
        num_requests=100,
        test_name="异步API响应时间测试"
    )
    print(f"平均响应时间: {metrics.average_response_time:.3f}s")

asyncio.run(test_async())
```

### 2. 并发处理能力测试

```python
from tests.performance.concurrency_tests import test_high_concurrency, test_async_high_concurrency

# 线程并发测试
metrics, validation = test_high_concurrency(
    "https://api.example.com/endpoint",
    concurrent_users=50,
    requests_per_user=100,
    test_name="高并发测试"
)
print(f"成功率: {metrics.success_rate:.3%}")
print(f"吞吐量: {metrics.requests_per_second:.2f} req/s")
print(f"性能要求满足: {validation['requirements_met']}")

# 异步并发测试
import asyncio
async def test_async_concurrency():
    metrics, validation = await test_async_high_concurrency(
        "https://api.example.com/endpoint",
        concurrent_users=50,
        requests_per_user=100,
        test_name="异步高并发测试"
    )
    print(f"成功率: {metrics.success_rate:.3%}")

asyncio.run(test_async_concurrency())
```

### 3. 内存泄漏检测

```python
from tests.performance.memory_leak_detector import detect_memory_leak

def memory_intensive_function():
    # 你的内存密集型函数
    data = []
    for i in range(10000):
        data.append([j for j in range(100)])
    return len(data)

# 检测内存泄漏
analysis = detect_memory_leak(
    memory_intensive_function,
    duration=60,  # 监控60秒
    interval=10,  # 每10秒检查一次
    test_name="内存泄漏检测"
)

print(f"检测到内存泄漏: {analysis.leak_detected}")
print(f"内存增长率: {analysis.growth_rate_mb_per_hour:.2f} MB/h")
print(f"优化建议: {analysis.recommendations}")
```

### 4. 资源利用率监控

```python
from tests.performance.resource_utilization_monitor import ResourceUtilizationMonitor
import time

monitor = ResourceUtilizationMonitor()

# 启动监控
monitor.start_monitoring()

# 执行你的代码
time.sleep(30)  # 模拟30秒的工作负载

# 获取监控结果
current_stats = monitor.get_current_stats()
historical_data = monitor.get_historical_data()

print(f"当前CPU使用率: {current_stats.cpu_percent:.1f}%")
print(f"当前内存使用率: {current_stats.memory_percent:.1f}%")
print(f"收集了 {len(historical_data)} 个数据点")

# 停止监控
monitor.stop_monitoring()
```

### 5. 执行效率测试

```python
from tests.performance.execution_efficiency_tests import ExecutionEfficiencyTester

tester = ExecutionEfficiencyTester()

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# 测试函数执行效率
metrics = tester.measure_function_execution(
    fibonacci,
    args=(20,),
    iterations=100,
    test_name="斐波那契性能测试"
)

print(f"平均执行时间: {metrics.average_execution_time:.6f}s")
print(f"性能等级: {metrics.performance_grade}")
print(f"内存使用: {metrics.memory_usage_mb:.2f} MB")
```

### 6. 性能报告生成

```python
from tests.performance.performance_report_generator import PerformanceReportGenerator, generate_quick_report
from tests.performance.core_performance_framework import PerformanceMetrics

# 创建性能指标数据
metrics = PerformanceMetrics(
    test_name="示例测试",
    start_time=datetime.now(),
    end_time=datetime.now(),
    duration=10.5,
    success_rate=0.95,
    error_count=5,
    memory_usage_mb=256.7,
    cpu_usage_percent=45.2
)

# 生成快速报告
report_path = generate_quick_report(
    [metrics],
    output_dir="./reports",
    report_title="示例性能报告"
)
print(f"报告已生成: {report_path}")
```

## 配置选项

### PerformanceConfig 配置

```python
from tests.performance.core_performance_framework import PerformanceConfig

config = PerformanceConfig(
    max_memory_usage_mb=2000,        # 最大内存使用限制 (MB)
    max_cpu_usage_percent=80,        # 最大CPU使用率限制 (%)
    max_response_time_ms=5000,       # 最大响应时间限制 (ms)
    min_success_rate=0.999,          # 最小成功率要求
    memory_check_interval=30,        # 内存检查间隔 (秒)
    resource_monitor_interval=5,     # 资源监控间隔 (秒)
    concurrent_users=50,             # 默认并发用户数
    requests_per_user=100,           # 每用户默认请求数
    test_timeout=300                 # 测试超时时间 (秒)
)
```

## 命令行参数

### 主运行器参数

```bash
python tests/performance/run_performance_tests.py [选项]

测试模式选择 (必选其一):
  --all                     运行所有性能测试
  --module {response_time,concurrency,memory_leak,resource_monitoring,execution_efficiency}
                           运行指定的测试模块
  --integration            运行集成测试
  --quick-test             运行快速集成测试

测试配置参数:
  --url URL                测试目标URL (默认: https://httpbin.org/delay/0.1)
  --users USERS            并发用户数 (默认: 50)
  --requests REQUESTS      每用户请求数 (默认: 100)
  --duration DURATION      测试持续时间（秒） (默认: 120)
  --output OUTPUT          报告输出目录

日志配置:
  --log-level {DEBUG,INFO,WARNING,ERROR}
                           日志级别 (默认: INFO)
  --quiet                  静默模式，只输出错误
```

## 报告格式

### HTML报告
- 包含详细的性能图表和分析
- 支持响应时间分布、内存使用趋势、CPU使用率等可视化
- 提供性能等级评估和改进建议

### JSON报告
- 结构化的测试结果数据
- 便于程序化处理和集成到CI/CD流程
- 包含完整的测试指标和元数据

## 性能等级评估

框架使用A+到F的等级系统评估性能：

- **A+**: 优秀 (响应时间 < 100ms, 成功率 > 99.9%)
- **A**: 良好 (响应时间 < 500ms, 成功率 > 99.5%)
- **B**: 一般 (响应时间 < 1000ms, 成功率 > 99%)
- **C**: 较差 (响应时间 < 2000ms, 成功率 > 95%)
- **D**: 差 (响应时间 < 5000ms, 成功率 > 90%)
- **F**: 不及格 (响应时间 >= 5000ms 或 成功率 <= 90%)

## 最佳实践

### 1. 测试环境准备
- 确保测试环境与生产环境尽可能相似
- 关闭不必要的后台程序以减少干扰
- 使用稳定的网络连接

### 2. 测试数据准备
- 使用真实的测试数据
- 准备足够的测试用例覆盖各种场景
- 考虑边界条件和异常情况

### 3. 测试执行
- 多次运行测试以获得稳定的结果
- 记录测试环境的详细信息
- 监控系统资源使用情况

### 4. 结果分析
- 关注趋势而不是单次测试结果
- 结合业务需求分析性能指标
- 识别性能瓶颈并制定优化计划

## 故障排查

### 常见问题

1. **导入错误**
   ```
   ModuleNotFoundError: No module named 'tests.performance.xxx'
   ```
   - 确保在项目根目录运行测试
   - 检查Python路径配置

2. **网络连接超时**
   ```
   requests.exceptions.ConnectTimeout
   ```
   - 检查目标URL是否可访问
   - 调整超时时间设置
   - 使用本地测试服务器

3. **内存不足**
   ```
   MemoryError
   ```
   - 减少并发用户数
   - 降低测试数据量
   - 增加系统内存

4. **权限错误**
   ```
   PermissionError: [Errno 13] Permission denied
   ```
   - 检查输出目录的写权限
   - 使用管理员权限运行

### 调试技巧

1. **启用详细日志**
   ```bash
   python run_performance_tests.py --log-level DEBUG
   ```

2. **使用快速测试验证**
   ```bash
   python run_performance_tests.py --quick-test
   ```

3. **单独测试各模块**
   ```bash
   python run_performance_tests.py --module response_time
   ```

## 扩展开发

### 添加新的测试模块

1. 创建新的测试类，继承适当的基类
2. 实现必要的测试方法
3. 添加到主运行器中
4. 更新文档和示例

### 自定义性能指标

1. 扩展 `PerformanceMetrics` 类
2. 实现新的收集器
3. 更新报告生成器
4. 添加相应的可视化

## 许可证

本项目遵循 MIT 许可证。详情请参阅 LICENSE 文件。

## 贡献指南

欢迎提交问题报告和功能请求。在提交代码前，请确保：

1. 遵循项目的代码规范
2. 添加适当的测试用例
3. 更新相关文档
4. 通过所有现有测试

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目仓库: [HarborAI GitHub](https://github.com/harborai/harborai)
- 问题报告: [GitHub Issues](https://github.com/harborai/harborai/issues)
- 邮件: support@harborai.com

---

*最后更新: 2024年*