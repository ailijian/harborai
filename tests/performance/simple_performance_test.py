#!/usr/bin/env python3
"""
简单的性能测试脚本
用于获取基本的性能指标
"""

import sys
import time
import psutil
import json
from datetime import datetime
from pathlib import Path
sys.path.append('../..')

# 导入统一报告管理器
sys.path.append(str(Path(__file__).parent.parent))
from utils.unified_report_manager import get_performance_report_path

from tests.performance.response_time_tests import test_api_response_time
from tests.performance.concurrency_tests import test_high_concurrency
from tests.performance.memory_leak_detector import detect_memory_leak

def run_simple_performance_tests():
    """运行简单的性能测试"""
    results = {
        'test_session': f'simple_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'start_time': datetime.now().isoformat(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'python_version': sys.version
        },
        'tests': {}
    }
    
    print("=== HarborAI 简单性能测试 ===")
    
    # 1. 响应时间测试
    print("\n1. 响应时间测试...")
    try:
        metrics = test_api_response_time('https://httpbin.org/delay/0.1', num_requests=50)
        results['tests']['response_time'] = {
            'average_response_time': metrics.average_response_time,
            'success_rate': metrics.success_rate,
            'performance_grade': metrics.performance_grade,
            'min_time': metrics.min_response_time,
            'max_time': metrics.max_response_time,
            'total_requests': metrics.total_requests
        }
        print(f"   平均响应时间: {metrics.average_response_time:.3f}s")
        print(f"   成功率: {metrics.success_rate:.1%}")
        print(f"   性能等级: {metrics.performance_grade}")
    except Exception as e:
        print(f"   响应时间测试失败: {e}")
        results['tests']['response_time'] = {'error': str(e)}
    
    # 2. 并发测试
    print("\n2. 并发处理能力测试...")
    try:
        metrics, validation = test_high_concurrency(
            'https://httpbin.org/delay/0.1', 
            concurrent_users=20, 
            requests_per_user=10
        )
        results['tests']['concurrency'] = {
            'requests_per_second': metrics.requests_per_second,
            'success_rate': metrics.success_rate,
            'average_response_time': metrics.average_response_time,
            'requirements_met': validation['requirements_met'],
            'total_requests': metrics.total_requests
        }
        print(f"   吞吐量: {metrics.requests_per_second:.2f} RPS")
        print(f"   成功率: {metrics.success_rate:.1%}")
        print(f"   需求满足: {validation['requirements_met']}")
    except Exception as e:
        print(f"   并发测试失败: {e}")
        results['tests']['concurrency'] = {'error': str(e)}
    
    # 3. 内存使用测试
    print("\n3. 内存使用测试...")
    try:
        def memory_test_function():
            # 模拟内存密集型操作
            data = []
            for i in range(5000):
                data.append([j for j in range(50)])
            return len(data)
        
        analysis = detect_memory_leak(memory_test_function, duration=30, monitoring_interval=5)
        results['tests']['memory'] = {
            'leak_detected': analysis.is_leak_detected,
            'leak_rate': analysis.leak_rate,
            'peak_memory': analysis.peak_memory,
            'average_memory': analysis.average_memory
        }
        print(f"   内存泄漏检测: {analysis.is_leak_detected}")
        print(f"   泄漏率: {analysis.leak_rate:.2f} bytes/s")
        print(f"   峰值内存: {analysis.peak_memory:.2f} MB")
    except Exception as e:
        print(f"   内存测试失败: {e}")
        results['tests']['memory'] = {'error': str(e)}
    
    # 4. 系统资源监控
    print("\n4. 系统资源状态...")
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('C:\\')
        
        results['tests']['system_resources'] = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
        print(f"   CPU使用率: {cpu_percent:.1f}%")
        print(f"   内存使用率: {memory.percent:.1f}%")
        print(f"   可用内存: {memory.available / (1024**3):.1f} GB")
        print(f"   磁盘使用率: {disk.percent:.1f}%")
    except Exception as e:
        print(f"   系统资源监控失败: {e}")
        results['tests']['system_resources'] = {'error': str(e)}
    
    results['end_time'] = datetime.now().isoformat()
    
    # 保存结果 - 使用统一报告管理器
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f'simple_test_{timestamp}.json'
    output_file = get_performance_report_path("metrics", "json", output_filename)
    
    # 确保目录存在
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n=== 测试完成 ===")
    print(f"结果已保存到: {output_file}")
    
    return results

if __name__ == "__main__":
    run_simple_performance_tests()