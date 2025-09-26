# -*- coding: utf-8 -*-
"""
HarborAI 负载测试模块

本模块包含HarborAI项目的负载测试功能，用于评估系统在不同负载条件下的性能表现。
负载测试专注于验证系统在预期和峰值负载下的稳定性和性能。

主要功能:
- 渐进式负载测试
- 峰值负载测试
- 持续负载测试
- 突发负载测试
- 多厂商负载对比测试
- 负载恢复测试

作者: HarborAI Team
版本: 1.0.0
创建时间: 2024-01-20
"""

__version__ = '1.0.0'
__author__ = 'HarborAI Team'
__email__ = 'team@harborai.com'

# 负载测试配置
LOAD_TEST_CONFIG = {
    # 负载级别定义
    'load_levels': {
        'light': {
            'concurrent_users': 10,
            'requests_per_second': 5,
            'duration_minutes': 5,
            'ramp_up_seconds': 30
        },
        'normal': {
            'concurrent_users': 50,
            'requests_per_second': 25,
            'duration_minutes': 10,
            'ramp_up_seconds': 60
        },
        'heavy': {
            'concurrent_users': 100,
            'requests_per_second': 50,
            'duration_minutes': 15,
            'ramp_up_seconds': 120
        },
        'peak': {
            'concurrent_users': 200,
            'requests_per_second': 100,
            'duration_minutes': 20,
            'ramp_up_seconds': 180
        },
        'extreme': {
            'concurrent_users': 500,
            'requests_per_second': 250,
            'duration_minutes': 30,
            'ramp_up_seconds': 300
        }
    },
    
    # 负载测试场景
    'test_scenarios': {
        'gradual_ramp': {
            'description': '渐进式负载增长测试',
            'phases': ['light', 'normal', 'heavy'],
            'phase_duration_minutes': 10
        },
        'spike_test': {
            'description': '突发负载测试',
            'base_load': 'normal',
            'spike_load': 'peak',
            'spike_duration_minutes': 5,
            'spike_interval_minutes': 15
        },
        'endurance_test': {
            'description': '持续负载测试',
            'load_level': 'normal',
            'duration_hours': 2,
            'monitoring_interval_minutes': 5
        },
        'capacity_test': {
            'description': '容量极限测试',
            'start_load': 'light',
            'max_load': 'extreme',
            'increment_step': 25,
            'step_duration_minutes': 10
        }
    },
    
    # 性能阈值
    'performance_thresholds': {
        'response_time': {
            'acceptable_ms': 2000,
            'good_ms': 1000,
            'excellent_ms': 500
        },
        'throughput': {
            'min_rps': 10,
            'target_rps': 50,
            'optimal_rps': 100
        },
        'error_rate': {
            'max_acceptable': 0.05,  # 5%
            'target': 0.01,  # 1%
            'excellent': 0.001  # 0.1%
        },
        'resource_usage': {
            'max_cpu_percent': 80,
            'max_memory_percent': 85,
            'max_disk_io_mbps': 100
        }
    },
    
    # 负载测试标记
    'test_markers': {
        'quick_load': 'light',
        'standard_load': 'normal',
        'stress_load': 'heavy',
        'capacity_load': 'peak',
        'extreme_load': 'extreme'
    },
    
    # 支持的厂商和模型
    'supported_vendors': {
        'deepseek': ['deepseek-chat', 'deepseek-r1'],
        'baidu': ['ernie-3.5-8k', 'ernie-4.0-turbo-8k', 'ernie-x1-turbo-32k'],
        'bytedance': ['doubao-1-5-pro-32k-character-250715', 'doubao-seed-1-6-250615']
    },
    
    # 负载测试报告配置
    'reporting': {
        'metrics_collection_interval': 10,  # 秒
        'detailed_logging': True,
        'export_formats': ['json', 'csv', 'html'],
        'charts_enabled': True
    }
}

# 负载测试性能等级
LOAD_PERFORMANCE_GRADES = {
    'A+': {'min_score': 95, 'description': '卓越负载性能'},
    'A': {'min_score': 85, 'description': '优秀负载性能'},
    'B': {'min_score': 75, 'description': '良好负载性能'},
    'C': {'min_score': 65, 'description': '可接受负载性能'},
    'D': {'min_score': 50, 'description': '较差负载性能'},
    'F': {'min_score': 0, 'description': '不可接受负载性能'}
}

# 导出主要配置
__all__ = [
    'LOAD_TEST_CONFIG',
    'LOAD_PERFORMANCE_GRADES',
    '__version__',
    '__author__'
]