#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试工具包

功能：提供测试相关的工具函数和类，包括：
- test_helpers: 测试辅助函数
- mock_helpers: Mock辅助函数
- data_generators: 测试数据生成器
- performance_utils: 性能测试工具
- security_utils: 安全测试工具
- report_utils: 报告生成工具

作者：HarborAI测试团队
创建时间：2024年
"""

# 导入所有模块的主要类和函数
from .test_helpers import (
    TestConfig,
    TestTimer,
    MockDataGenerator,
    SecurityTestHelper,
    TestMetrics,
    measure_performance,
    generate_test_data,
    temporary_config,
    retry_on_failure,
    test_metrics
)

from .mock_helpers import (
    MockResponse,
    StreamChunk,
    APIResponseMocker,
    ErrorSimulator,
    HarborAIMocker,
    MockMetricsCollector,
    mock_harborai_client,
    async_mock_harborai_client,
    create_mock_response_factory,
    patch_harborai_method,
    mock_metrics_collector
)

from .data_generators import (
    TestDataGenerator,
    generate_chat_messages,
    generate_json_schema,
    generate_api_keys,
    generate_performance_data,
    generate_user_data,
    generate_malicious_inputs,
    bulk_generate_data,
    configure_data_generation
)

from .performance_utils import (
    PerformanceMetrics,
    PerformanceThresholds,
    SystemMonitor,
    PerformanceTester,
    BenchmarkSuite,
    measure_performance as perf_measure,
    performance_monitor
)

from .security_utils import (
    SecurityLevel,
    VulnerabilityType,
    SecurityIssue,
    SecurityTestConfig,
    DataSanitizer,
    VulnerabilityScanner,
    SecurityValidator,
    SecurityReporter,
    sanitize_data,
    scan_for_vulnerabilities,
    validate_security,
    generate_security_report,
    detect_sensitive_data,
    security_test,
    sanitize_output
)

from .report_utils import (
    TestSummary,
    PerformanceMetrics as ReportPerformanceMetrics,
    SecurityMetrics as ReportSecurityMetrics,
    ReportGenerator,
    MetricsCollector,
    TestMetricsExporter,
    generate_test_report,
    collect_and_export_metrics,
    save_test_session,
    report_generator,
    metrics_collector,
    metrics_exporter
)

# 版本信息
__version__ = "1.0.0"
__author__ = "HarborAI测试团队"

# 导出的主要接口
__all__ = [
    # test_helpers
    "TestConfig",
    "TestTimer", 
    "MockDataGenerator",
    "SecurityTestHelper",
    "TestMetrics",
    "measure_performance",
    "generate_test_data",
    "temporary_config",
    "retry_on_failure",
    "test_metrics",
    
    # mock_helpers
    "MockResponse",
    "StreamChunk",
    "APIResponseMocker",
    "ErrorSimulator",
    "HarborAIMocker",
    "MockMetricsCollector",
    "mock_harborai_client",
    "async_mock_harborai_client",
    "create_mock_response_factory",
    "patch_harborai_method",
    "mock_metrics_collector",
    
    # data_generators
    "TestDataGenerator",
    "generate_chat_messages",
    "generate_json_schema",
    "generate_api_keys",
    "generate_performance_data",
    "generate_user_data",
    "generate_malicious_inputs",
    "bulk_generate_data",
    "configure_data_generation",
    
    # performance_utils
    "PerformanceMetrics",
    "PerformanceThresholds",
    "SystemMonitor",
    "PerformanceTester",
    "BenchmarkSuite",
    "perf_measure",
    "performance_monitor",
    
    # security_utils
    "SecurityLevel",
    "VulnerabilityType",
    "SecurityIssue",
    "SecurityTestConfig",
    "DataSanitizer",
    "VulnerabilityScanner",
    "SecurityValidator",
    "SecurityReporter",
    "sanitize_data",
    "scan_for_vulnerabilities",
    "validate_security",
    "generate_security_report",
    "detect_sensitive_data",
    "security_test",
    "sanitize_output",
    
    # report_utils
    "TestSummary",
    "ReportPerformanceMetrics",
    "ReportSecurityMetrics",
    "ReportGenerator",
    "MetricsCollector",
    "TestMetricsExporter",
    "generate_test_report",
    "collect_and_export_metrics",
    "save_test_session",
    "report_generator",
    "metrics_collector",
    "metrics_exporter"
]