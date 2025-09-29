# -*- coding: utf-8 -*-
"""
HarborAI 集成测试模块

本模块包含 HarborAI 项目的集成测试，用于验证各组件之间的协作和整体功能。

测试模块包括：
- test_end_to_end.py: 端到端测试
- test_multi_vendor.py: 多厂商集成测试
- test_database_integration.py: 数据库集成测试
- test_docker_environment.py: Docker环境测试

测试要求：
- 使用 pytest 框架
- 支持异步测试
- 包含性能基准测试
- 支持多厂商API测试
- 包含完整的错误处理和重试机制
"""

__version__ = "1.0.0"
__author__ = "HarborAI Team"

# 集成测试配置
INTEGRATION_TEST_CONFIG = {
    "timeout": 60,  # 集成测试超时时间（秒）
    "retry_count": 3,  # 重试次数
    "retry_delay": 1.0,  # 重试延迟（秒）
    "enable_real_api": False,  # 是否启用真实API测试
    "database_url": "postgresql://test:test@localhost:5432/harborai_test",  # 测试数据库URL
    "docker_compose_file": "docker-compose.test.yml",  # Docker测试配置文件
}

# 支持的厂商和模型配置
SUPPORTED_VENDORS = {
    "deepseek": {
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "reasoning_models": ["deepseek-reasoner"],
        "base_url": "https://api.deepseek.com"
    },
    "ernie": {
        "models": ["ernie-3.5-8k", "ernie-4.0-turbo-8k", "ernie-x1-turbo-32k"],
        "reasoning_models": ["ernie-x1-turbo-32k"],
        "base_url": "https://aip.baidubce.com"
    },
    "doubao": {
        "models": ["doubao-1-5-pro-32k-character-250715", "doubao-seed-1-6-250615"],
        "reasoning_models": ["doubao-seed-1-6-250615"],
        "base_url": "https://ark.cn-beijing.volces.com"
    }
}

# 测试数据配置
TEST_DATA_CONFIG = {
    "simple_message": "你好，请介绍一下你自己。",
    "complex_message": "请详细分析机器学习在自然语言处理中的应用，包括主要算法、优势和挑战。",
    "reasoning_message": "请一步步分析以下数学问题：如果一个圆的半径是5cm，那么它的面积和周长分别是多少？",
    "json_schema_simple": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer", "minimum": 0}
        },
        "required": ["name", "age"]
    }
}

# 性能基准配置
PERFORMANCE_BENCHMARKS = {
    "max_response_time": 10.0,  # 最大响应时间（秒）
    "max_memory_usage": 500,  # 最大内存使用（MB）
    "concurrent_requests": [1, 5, 10, 20],  # 并发请求数量
    "stress_test_duration": 60,  # 压力测试持续时间（秒）
}