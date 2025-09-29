# -*- coding: utf-8 -*-
"""
HarborAI 性能基准测试模块

本模块提供了HarborAI项目的性能基准测试功能，包括：
- API响应时间基准测试
- 吞吐量基准测试
- 并发性能基准测试
- 资源使用基准测试
- 模型性能对比基准测试

基准测试用于建立性能基线，监控性能回归，并进行性能优化验证。

作者: HarborAI Team
版本: 1.0.0
创建时间: 2024-01-20
"""

__version__ = "1.0.0"
__author__ = "HarborAI Team"

# 基准测试配置
BENCHMARK_CONFIG = {
    # 基准测试运行配置
    'rounds': 5,  # 基准测试轮数
    'warmup_rounds': 2,  # 预热轮数
    'min_time': 0.1,  # 最小运行时间（秒）
    'max_time': 60.0,  # 最大运行时间（秒）
    
    # 性能基线阈值
    'baseline_thresholds': {
        'api_response_time': {
            'excellent': 0.5,  # 优秀：500ms以下
            'good': 1.0,       # 良好：1秒以下
            'acceptable': 2.0,  # 可接受：2秒以下
            'poor': 5.0        # 较差：5秒以下
        },
        'throughput': {
            'excellent': 10,     # 请求/秒 >= 10
            'good': 5,           # 请求/秒 >= 5
            'acceptable': 1,     # 请求/秒 >= 1
            'poor': 0.5          # 请求/秒 >= 0.5
        },
        'requests_per_second': {
            'excellent': 100,
            'good': 50,
            'acceptable': 20,
            'poor': 10
        },
        'tokens_per_second': {
            'excellent': 1000,
            'good': 500,
            'acceptable': 200,
            'poor': 100
        },
        'concurrent_performance': {
            'excellent': 50.0,   # 并发吞吐量 * 效率比 >= 50
            'good': 30.0,        # 并发吞吐量 * 效率比 >= 30
            'acceptable': 15.0,  # 并发吞吐量 * 效率比 >= 15
            'poor': 5.0          # 并发吞吐量 * 效率比 >= 5
        },
        'max_concurrent_users': {
            'excellent': 100,
            'good': 50,
            'acceptable': 20,
            'poor': 10
        },
        'response_time_degradation': {
            'excellent': 1.2,  # 响应时间增长不超过20%
            'good': 1.5,       # 响应时间增长不超过50%
            'acceptable': 2.0,  # 响应时间增长不超过100%
            'poor': 3.0        # 响应时间增长不超过200%
        },
        'resource_usage': {
            'cpu_usage_percent': {
                'excellent': 30,
                'good': 50,
                'acceptable': 70,
                'poor': 90
            },
            'memory_usage_mb': {
                'excellent': 512,
                'good': 1024,
                'acceptable': 2048,
                'poor': 4096
            }
        }
    },
    
    # 基准测试场景
    'benchmark_scenarios': {
        'quick': {
            'description': '快速基准测试',
            'duration': 30,  # 30秒
            'concurrent_users': [1, 5, 10],
            'request_types': ['simple_chat', 'reasoning']
        },
        'standard': {
            'description': '标准基准测试',
            'duration': 300,  # 5分钟
            'concurrent_users': [1, 5, 10, 20, 50],
            'request_types': ['simple_chat', 'reasoning', 'streaming']
        },
        'comprehensive': {
            'description': '全面基准测试',
            'duration': 1800,  # 30分钟
            'concurrent_users': [1, 5, 10, 20, 50, 100],
            'request_types': ['simple_chat', 'reasoning', 'streaming', 'complex_reasoning']
        }
    },
    
    # 支持的厂商和模型
    'supported_vendors': {
        'openai': ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo'],
        'anthropic': ['claude-3-haiku', 'claude-3-sonnet', 'claude-3-opus'],
        'google': ['gemini-pro', 'gemini-pro-vision'],
        'deepseek': ['deepseek-chat', 'deepseek-reasoner'],
        'ernie': ['ernie-bot', 'ernie-bot-turbo'],
        'doubao': ['doubao-pro', 'doubao-lite'],
        'local': ['llama2-7b', 'llama2-13b']
    },
    
    # 基准测试标记
    'benchmark_markers': {
        'quick_benchmark': 'Quick benchmark tests (< 1 minute)',
        'standard_benchmark': 'Standard benchmark tests (5-10 minutes)',
        'comprehensive_benchmark': 'Comprehensive benchmark tests (30+ minutes)',
        'regression_benchmark': 'Performance regression benchmark tests',
        'comparison_benchmark': 'Model/vendor comparison benchmark tests'
    }
}

# 基准测试结果等级
PERFORMANCE_GRADES = {
    'A+': 'excellent',
    'A': 'good', 
    'B': 'acceptable',
    'C': 'poor',
    'F': 'unacceptable'
}

# 导出的公共接口
__all__ = [
    'BENCHMARK_CONFIG',
    'PERFORMANCE_GRADES'
]