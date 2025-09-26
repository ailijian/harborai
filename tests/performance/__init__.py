# -*- coding: utf-8 -*-
"""
性能测试模块

本模块包含 HarborAI 项目的所有性能测试，包括：
- 基础性能测试
- 并发性能测试
- 压力测试
- 资源监控测试
- 流式性能测试
- 基准测试
- 负载测试

测试框架：pytest + pytest-benchmark + locust
性能监控：psutil + memory_profiler + prometheus_client
"""

__version__ = "1.0.0"
__author__ = "HarborAI Team"

# 性能测试配置常量
PERFORMANCE_CONFIG = {
    # 基础性能阈值（秒）
    "basic_latency_threshold": 5.0,
    "reasoning_latency_threshold": 10.0,
    
    # 基础性能测试配置
    "basic": {
        "max_response_time": 2.0,
        "min_throughput": 1.0,
        "max_error_rate": 0.05
    },
    
    # 并发测试配置
    "concurrent": {
        "levels": [1, 5, 10, 20, 50],
        "max_concurrent_requests": 100,
        "max_response_time": 3.0,
        "min_throughput": 0.8,
        "max_error_rate": 0.1
    },
    
    # 压力测试配置
    "stress": {
        "duration": 300,  # 5分钟
        "ramp_up": 60,    # 1分钟
        "max_response_time": 5.0,
        "min_throughput": 0.5,
        "max_error_rate": 0.2,
        "max_concurrent_users": 200,
        "degradation_threshold": 100
    },
    
    # 资源监控阈值
    "resource": {
        "memory_threshold_mb": 500,
        "cpu_threshold_percent": 80,
        "monitoring_interval": 0.5
    },
    
    # 流式测试配置
    "streaming": {
        "chunk_timeout": 30,
        "total_timeout": 120,
        "max_chunk_delay": 1.0,
        "min_chunks_per_second": 1.0
    },
    
    # 基准测试配置
    "benchmark": {
        "rounds": 10,
        "iterations": 100,
        "warmup_rounds": 3
    },
    
    # 负载测试配置
    "load": {
        "users": 50,
        "spawn_rate": 5,
        "duration": 300
    }
}

# 支持的厂商和模型配置
SUPPORTED_VENDORS = {
    "deepseek": {
        "models": ["deepseek-chat", "deepseek-r1"],
        "reasoning_models": ["deepseek-r1"]
    },
    "ernie": {
        "models": ["ernie-3.5-8k", "ernie-4.0-turbo-8k", "ernie-x1-turbo-32k"],
        "reasoning_models": ["ernie-x1-turbo-32k"]
    },
    "doubao": {
        "models": ["doubao-1-5-pro-32k-character-250715", "doubao-seed-1-6-250615"],
        "reasoning_models": ["doubao-seed-1-6-250615"]
    }
}

# 性能测试标记
PERFORMANCE_MARKERS = {
    "basic": "基础性能测试",
    "concurrent": "并发性能测试", 
    "stress": "压力测试",
    "resource": "资源监控测试",
    "streaming": "流式性能测试",
    "benchmark": "基准测试",
    "load": "负载测试"
}