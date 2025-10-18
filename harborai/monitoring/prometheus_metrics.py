#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prometheus指标导出模块

提供API调用次数、响应时间分布、错误率、数据一致性、性能监控等关键指标的监控。
"""

import time
import functools
from typing import Optional, Dict, Any, Callable
from prometheus_client import (
    Counter, Histogram, Gauge, Info, Summary,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)
from ..utils.logger import get_logger
from ..utils.exceptions import HarborAIError

logger = get_logger(__name__)

# 全局Prometheus指标实例
_prometheus_metrics: Optional['PrometheusMetrics'] = None


class PrometheusMetrics:
    """Prometheus指标收集器"""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """初始化Prometheus指标收集器
        
        Args:
            registry: Prometheus注册表，默认使用全局注册表
        """
        self.registry = registry or CollectorRegistry()
        self._init_metrics()
        logger.info("Prometheus指标收集器已初始化")
    
    def _init_metrics(self):
        """初始化所有指标"""
        # === 原有API指标 ===
        # API调用计数器
        self.api_requests_total = Counter(
            'harborai_api_requests_total',
            'API请求总数',
            ['method', 'model', 'provider', 'status'],
            registry=self.registry
        )
        
        # API响应时间直方图
        self.api_request_duration_seconds = Histogram(
            'harborai_api_request_duration_seconds',
            'API请求持续时间（秒）',
            ['method', 'model', 'provider'],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0),
            registry=self.registry
        )
        
        # Token使用量计数器
        self.tokens_used_total = Counter(
            'harborai_tokens_used_total',
            'Token使用总量',
            ['model', 'provider', 'token_type'],
            registry=self.registry
        )
        
        # 成本计数器
        self.cost_total = Counter(
            'harborai_cost_total',
            'API调用总成本（美元）',
            ['model', 'provider'],
            registry=self.registry
        )
        
        # 错误率计数器
        self.api_errors_total = Counter(
            'harborai_api_errors_total',
            'API错误总数',
            ['method', 'model', 'provider', 'error_type'],
            registry=self.registry
        )
        
        # 当前活跃连接数
        self.active_connections = Gauge(
            'harborai_active_connections',
            '当前活跃连接数',
            ['provider'],
            registry=self.registry
        )
        
        # 重试次数计数器
        self.retries_total = Counter(
            'harborai_retries_total',
            '重试总次数',
            ['model', 'provider', 'retry_reason'],
            registry=self.registry
        )
        
        # 缓存命中率
        self.cache_hits_total = Counter(
            'harborai_cache_hits_total',
            '缓存命中总数',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_misses_total = Counter(
            'harborai_cache_misses_total',
            '缓存未命中总数',
            ['cache_type'],
            registry=self.registry
        )
        
        # 系统信息
        self.system_info = Info(
            'harborai_system_info',
            'HarborAI系统信息',
            registry=self.registry
        )
        
        # === 新增：数据一致性指标 ===
        # 数据一致性检查计数器
        self.data_consistency_checks_total = Counter(
            'harborai_data_consistency_checks_total',
            '数据一致性检查总数',
            ['check_type', 'status'],
            registry=self.registry
        )
        
        # 数据一致性问题计数器
        self.data_consistency_issues_total = Counter(
            'harborai_data_consistency_issues_total',
            '数据一致性问题总数',
            ['issue_type', 'severity', 'table_name'],
            registry=self.registry
        )
        
        # 自动修正操作计数器
        self.auto_correction_operations_total = Counter(
            'harborai_auto_correction_operations_total',
            '自动修正操作总数',
            ['operation_type', 'table_name', 'status'],
            registry=self.registry
        )
        
        # 数据库约束违反计数器
        self.database_constraint_violations_total = Counter(
            'harborai_database_constraint_violations_total',
            '数据库约束违反总数',
            ['constraint_type', 'table_name'],
            registry=self.registry
        )
        
        # 数据完整性评分
        self.data_integrity_score = Gauge(
            'harborai_data_integrity_score',
            '数据完整性评分（0-100）',
            ['table_name'],
            registry=self.registry
        )
        
        # === 新增：性能监控指标 ===
        # 数据库连接池指标
        self.database_connections_active = Gauge(
            'harborai_database_connections_active',
            '活跃数据库连接数',
            registry=self.registry
        )
        
        self.database_connections_idle = Gauge(
            'harborai_database_connections_idle',
            '空闲数据库连接数',
            registry=self.registry
        )
        
        # 数据库查询性能
        self.database_query_duration_seconds = Histogram(
            'harborai_database_query_duration_seconds',
            '数据库查询持续时间（秒）',
            ['query_type', 'table_name'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        # 内存使用量
        self.memory_usage_bytes = Gauge(
            'harborai_memory_usage_bytes',
            '内存使用量（字节）',
            ['component'],
            registry=self.registry
        )
        
        # CPU使用率
        self.cpu_usage_percent = Gauge(
            'harborai_cpu_usage_percent',
            'CPU使用率（百分比）',
            ['component'],
            registry=self.registry
        )
        
        # 队列长度
        self.queue_length = Gauge(
            'harborai_queue_length',
            '队列长度',
            ['queue_name'],
            registry=self.registry
        )
        
        # 处理延迟
        self.processing_latency_seconds = Summary(
            'harborai_processing_latency_seconds',
            '处理延迟（秒）',
            ['operation_type'],
            registry=self.registry
        )
        
        # === 新增：业务指标 ===
        # 用户活跃度
        self.active_users_total = Gauge(
            'harborai_active_users_total',
            '活跃用户总数',
            ['time_window'],  # 1h, 24h, 7d
            registry=self.registry
        )
        
        # 模型使用统计
        self.model_usage_requests_total = Counter(
            'harborai_model_usage_requests_total',
            '模型使用请求总数',
            ['model_name', 'provider', 'user_type'],
            registry=self.registry
        )
        
        # 成本效率指标
        self.cost_efficiency_ratio = Gauge(
            'harborai_cost_efficiency_ratio',
            '成本效率比（输出token/成本）',
            ['model_name', 'provider'],
            registry=self.registry
        )
        
        # 服务质量指标
        self.service_quality_score = Gauge(
            'harborai_service_quality_score',
            '服务质量评分（0-100）',
            ['service_type'],
            registry=self.registry
        )
        
        # 追踪覆盖率
        self.tracing_coverage_percent = Gauge(
            'harborai_tracing_coverage_percent',
            '追踪覆盖率（百分比）',
            ['operation_type'],
            registry=self.registry
        )
        
        # === 新增：健康检查指标 ===
        # 健康检查状态
        self.health_check_status = Gauge(
            'harborai_health_check_status',
            '健康检查状态（1=健康，0=不健康）',
            ['component', 'check_type'],
            registry=self.registry
        )
        
        # 健康检查响应时间
        self.health_check_duration_seconds = Histogram(
            'harborai_health_check_duration_seconds',
            '健康检查响应时间（秒）',
            ['component', 'check_type'],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry
        )
        
        # 降级状态
        self.degradation_status = Gauge(
            'harborai_degradation_status',
            '降级状态（1=降级，0=正常）',
            ['component', 'degradation_type'],
            registry=self.registry
        )
        
        # === 新增：高级监控指标 ===
        # API请求P95响应时间
        self.api_response_time_p95 = Gauge(
            'harborai_api_response_time_p95_seconds',
            'API响应时间P95（秒）',
            ['model', 'provider'],
            registry=self.registry
        )
        
        # API请求P99响应时间
        self.api_response_time_p99 = Gauge(
            'harborai_api_response_time_p99_seconds',
            'API响应时间P99（秒）',
            ['model', 'provider'],
            registry=self.registry
        )
        
        # API错误率百分比
        self.api_error_rate_percent = Gauge(
            'harborai_api_error_rate_percent',
            'API错误率（百分比）',
            ['model', 'provider'],
            registry=self.registry
        )
        
        # 并发连接数
        self.api_concurrent_connections = Gauge(
            'harborai_api_concurrent_connections',
            'API并发连接数',
            registry=self.registry
        )
        
        # 数据库连接池使用率
        self.database_connection_pool_usage = Gauge(
            'harborai_database_connection_pool_usage_percent',
            '数据库连接池使用率（百分比）',
            ['pool_name'],
            registry=self.registry
        )
        
        # 数据库查询延迟P99
        self.database_query_duration_p99 = Gauge(
            'harborai_database_query_duration_p99_seconds',
            '数据库查询延迟P99（秒）',
            ['query_type'],
            registry=self.registry
        )
        
        # 系统内存使用率
        self.memory_usage_percent = Gauge(
            'harborai_memory_usage_percent',
            '系统内存使用率（百分比）',
            registry=self.registry
        )
        
        # 系统CPU使用率
        self.system_cpu_usage_percent = Gauge(
            'harborai_system_cpu_usage_percent',
            '系统CPU使用率（百分比）',
            registry=self.registry
        )
        
        # 磁盘使用率
        self.disk_usage_percent = Gauge(
            'harborai_disk_usage_percent',
            '磁盘使用率（百分比）',
            ['mount_point'],
            registry=self.registry
        )
        
        # === 新增：数据质量监控指标 ===
        # Token数据质量评分
        self.token_data_quality_score = Gauge(
            'harborai_token_data_quality_score',
            'Token数据质量评分（0-100）',
            ['provider', 'model'],
            registry=self.registry
        )
        
        # 数据完整性检查
        self.data_completeness_percent = Gauge(
            'harborai_data_completeness_percent',
            '数据完整性百分比',
            ['table_name', 'field_name'],
            registry=self.registry
        )
        
        # 数据准确性评分
        self.data_accuracy_score = Gauge(
            'harborai_data_accuracy_score',
            '数据准确性评分（0-100）',
            ['data_type'],
            registry=self.registry
        )
        
        # 数据一致性检查失败次数
        self.data_consistency_failures_total = Counter(
            'harborai_data_consistency_failures_total',
            '数据一致性检查失败总数',
            ['check_type', 'table_name'],
            registry=self.registry
        )
        
        # 数据异常检测
        self.data_anomaly_detected_total = Counter(
            'harborai_data_anomaly_detected_total',
            '数据异常检测总数',
            ['anomaly_type', 'severity'],
            registry=self.registry
        )
        
        # 数据修复操作
        self.data_repair_operations_total = Counter(
            'harborai_data_repair_operations_total',
            '数据修复操作总数',
            ['repair_type', 'status'],
            registry=self.registry
        )
        
        # === 新增：业务质量指标 ===
        # Token解析成功率
        self.token_parsing_success_rate = Gauge(
            'harborai_token_parsing_success_rate_percent',
            'Token解析成功率（百分比）',
            ['provider', 'parsing_method'],
            registry=self.registry
        )
        
        # 成本计算准确性
        self.cost_calculation_accuracy = Gauge(
            'harborai_cost_calculation_accuracy_percent',
            '成本计算准确性（百分比）',
            ['model', 'provider'],
            registry=self.registry
        )
        
        # 价格配置更新次数
        self.price_config_updates_total = Counter(
            'harborai_price_config_updates_total',
            '价格配置更新总数',
            ['update_type', 'status'],
            registry=self.registry
        )
        
        # 告警规则触发次数
        self.alert_rule_triggers_total = Counter(
            'harborai_alert_rule_triggers_total',
            '告警规则触发总数',
            ['rule_id', 'severity', 'status'],
            registry=self.registry
        )
        
        # 告警抑制次数
        self.alert_suppressions_total = Counter(
            'harborai_alert_suppressions_total',
            '告警抑制总数',
            ['suppression_type', 'rule_id'],
            registry=self.registry
        )
        
        # 智能阈值调整次数
        self.intelligent_threshold_adjustments_total = Counter(
            'harborai_intelligent_threshold_adjustments_total',
            '智能阈值调整总数',
            ['metric_name', 'adjustment_type'],
            registry=self.registry
        )

    def record_api_request(self, method: str, model: str, provider: str, 
                          duration: float, status: str = 'success',
                          error_type: Optional[str] = None):
        """记录API请求指标"""
        # 记录请求总数
        self.api_requests_total.labels(
            method=method,
            model=model,
            provider=provider,
            status=status
        ).inc()
        
        # 记录响应时间
        self.api_request_duration_seconds.labels(
            method=method,
            model=model,
            provider=provider
        ).observe(duration)
        
        # 记录错误
        if status == 'error' and error_type:
            self.api_errors_total.labels(
                method=method,
                model=model,
                provider=provider,
                error_type=error_type
            ).inc()
    
    def record_token_usage(self, model: str, provider: str, 
                          prompt_tokens: int, completion_tokens: int):
        """记录Token使用量"""
        self.tokens_used_total.labels(
            model=model,
            provider=provider,
            token_type='prompt'
        ).inc(prompt_tokens)
        
        self.tokens_used_total.labels(
            model=model,
            provider=provider,
            token_type='completion'
        ).inc(completion_tokens)
    
    def record_cost(self, model: str, provider: str, cost: float):
        """记录API调用成本"""
        self.cost_total.labels(
            model=model,
            provider=provider
        ).inc(cost)
    
    def record_retry(self, model: str, provider: str, retry_reason: str):
        """记录重试事件"""
        self.retries_total.labels(
            model=model,
            provider=provider,
            retry_reason=retry_reason
        ).inc()
    
    def record_cache_hit(self, cache_type: str):
        """记录缓存命中"""
        self.cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """记录缓存未命中"""
        self.cache_misses_total.labels(cache_type=cache_type).inc()
    
    def set_active_connections(self, provider: str, count: int):
        """设置活跃连接数"""
        self.active_connections.labels(provider=provider).set(count)
    
    def set_system_info(self, info: Dict[str, str]):
        """设置系统信息"""
        self.system_info.info(info)
    
    # === 新增：数据一致性指标方法 ===
    def record_consistency_check(self, check_type: str, status: str):
        """记录数据一致性检查
        
        Args:
            check_type: 检查类型（token_consistency, cost_consistency, tracing_consistency等）
            status: 检查状态（success, failed, error）
        """
        self.data_consistency_checks_total.labels(
            check_type=check_type,
            status=status
        ).inc()
    
    def record_consistency_issue(self, issue_type: str, severity: str, table_name: str):
        """记录数据一致性问题
        
        Args:
            issue_type: 问题类型（missing_data, inconsistent_data, orphaned_record等）
            severity: 严重程度（low, medium, high, critical）
            table_name: 表名
        """
        self.data_consistency_issues_total.labels(
            issue_type=issue_type,
            severity=severity,
            table_name=table_name
        ).inc()
    
    def record_auto_correction(self, operation_type: str, table_name: str, status: str):
        """记录自动修正操作
        
        Args:
            operation_type: 操作类型（insert, update, delete, recalculate）
            table_name: 表名
            status: 操作状态（success, failed, skipped）
        """
        self.auto_correction_operations_total.labels(
            operation_type=operation_type,
            table_name=table_name,
            status=status
        ).inc()
    
    def record_constraint_violation(self, constraint_type: str, table_name: str):
        """记录数据库约束违反
        
        Args:
            constraint_type: 约束类型（foreign_key, unique, check, not_null）
            table_name: 表名
        """
        self.database_constraint_violations_total.labels(
            constraint_type=constraint_type,
            table_name=table_name
        ).inc()
    
    def set_data_integrity_score(self, table_name: str, score: float):
        """设置数据完整性评分
        
        Args:
            table_name: 表名
            score: 评分（0-100）
        """
        self.data_integrity_score.labels(table_name=table_name).set(score)
    
    # === 新增：性能监控指标方法 ===
    def set_database_connections(self, active: int, idle: int):
        """设置数据库连接数
        
        Args:
            active: 活跃连接数
            idle: 空闲连接数
        """
        self.database_connections_active.set(active)
        self.database_connections_idle.set(idle)
    
    def record_database_query(self, query_type: str, table_name: str, duration: float):
        """记录数据库查询性能
        
        Args:
            query_type: 查询类型（select, insert, update, delete）
            table_name: 表名
            duration: 查询持续时间（秒）
        """
        self.database_query_duration_seconds.labels(
            query_type=query_type,
            table_name=table_name
        ).observe(duration)
    
    def set_memory_usage(self, component: str, bytes_used: int):
        """设置内存使用量
        
        Args:
            component: 组件名称
            bytes_used: 使用的字节数
        """
        self.memory_usage_bytes.labels(component=component).set(bytes_used)
    
    def set_cpu_usage(self, component: str, percent: float):
        """设置CPU使用率
        
        Args:
            component: 组件名称
            percent: 使用率百分比
        """
        self.cpu_usage_percent.labels(component=component).set(percent)
    
    def set_queue_length(self, queue_name: str, length: int):
        """设置队列长度
        
        Args:
            queue_name: 队列名称
            length: 队列长度
        """
        self.queue_length.labels(queue_name=queue_name).set(length)
    
    def record_processing_latency(self, operation_type: str, latency: float):
        """记录处理延迟
        
        Args:
            operation_type: 操作类型
            latency: 延迟时间（秒）
        """
        self.processing_latency_seconds.labels(operation_type=operation_type).observe(latency)
    
    # === 新增：业务指标方法 ===
    def set_active_users(self, time_window: str, count: int):
        """设置活跃用户数
        
        Args:
            time_window: 时间窗口（1h, 24h, 7d）
            count: 用户数量
        """
        self.active_users_total.labels(time_window=time_window).set(count)
    
    def record_model_usage(self, model_name: str, provider: str, user_type: str):
        """记录模型使用
        
        Args:
            model_name: 模型名称
            provider: 提供商
            user_type: 用户类型（free, premium, enterprise）
        """
        self.model_usage_requests_total.labels(
            model_name=model_name,
            provider=provider,
            user_type=user_type
        ).inc()
    
    def set_cost_efficiency(self, model_name: str, provider: str, ratio: float):
        """设置成本效率比
        
        Args:
            model_name: 模型名称
            provider: 提供商
            ratio: 效率比（输出token/成本）
        """
        self.cost_efficiency_ratio.labels(
            model_name=model_name,
            provider=provider
        ).set(ratio)
    
    def set_service_quality_score(self, service_type: str, score: float):
        """设置服务质量评分
        
        Args:
            service_type: 服务类型
            score: 质量评分（0-100）
        """
        self.service_quality_score.labels(service_type=service_type).set(score)
    
    def set_tracing_coverage(self, operation_type: str, percent: float):
        """设置追踪覆盖率
        
        Args:
            operation_type: 操作类型
            percent: 覆盖率百分比
        """
        self.tracing_coverage_percent.labels(operation_type=operation_type).set(percent)
    
    # === 新增：健康检查指标方法 ===
    def set_health_check_status(self, component: str, check_type: str, is_healthy: bool):
        """设置健康检查状态
        
        Args:
            component: 组件名称
            check_type: 检查类型
            is_healthy: 是否健康
        """
        self.health_check_status.labels(
            component=component,
            check_type=check_type
        ).set(1 if is_healthy else 0)
    
    def record_health_check_duration(self, component: str, check_type: str, duration: float):
        """记录健康检查响应时间
        
        Args:
            component: 组件名称
            check_type: 检查类型
            duration: 响应时间（秒）
        """
        self.health_check_duration_seconds.labels(
            component=component,
            check_type=check_type
        ).observe(duration)
    
    def set_degradation_status(self, component: str, degradation_type: str, is_degraded: bool):
        """设置降级状态
        
        Args:
            component: 组件名称
            degradation_type: 降级类型
            is_degraded: 是否降级
        """
        self.degradation_status.labels(
            component=component,
            degradation_type=degradation_type
        ).set(1 if is_degraded else 0)
    
    def get_metrics_data(self) -> str:
        """获取Prometheus格式的指标数据
        
        Returns:
            str: Prometheus格式的指标数据
        """
        return generate_latest(self.registry)
    
    # === 新增：高级监控指标方法 ===
    def set_api_response_time_percentiles(self, model: str, provider: str, p95: float, p99: float):
        """设置API响应时间百分位数
        
        Args:
            model: 模型名称
            provider: 提供商
            p95: P95响应时间（秒）
            p99: P99响应时间（秒）
        """
        self.api_response_time_p95.labels(model=model, provider=provider).set(p95)
        self.api_response_time_p99.labels(model=model, provider=provider).set(p99)
    
    def set_api_error_rate(self, model: str, provider: str, error_rate: float):
        """设置API错误率
        
        Args:
            model: 模型名称
            provider: 提供商
            error_rate: 错误率（百分比）
        """
        self.api_error_rate_percent.labels(model=model, provider=provider).set(error_rate)
    
    def set_concurrent_connections(self, count: int):
        """设置并发连接数
        
        Args:
            count: 并发连接数
        """
        self.api_concurrent_connections.set(count)
    
    def set_database_pool_usage(self, pool_name: str, usage_percent: float):
        """设置数据库连接池使用率
        
        Args:
            pool_name: 连接池名称
            usage_percent: 使用率（百分比）
        """
        self.database_connection_pool_usage.labels(pool_name=pool_name).set(usage_percent)
    
    def set_database_query_p99(self, query_type: str, duration: float):
        """设置数据库查询P99延迟
        
        Args:
            query_type: 查询类型
            duration: P99延迟（秒）
        """
        self.database_query_duration_p99.labels(query_type=query_type).set(duration)
    
    def set_system_resource_usage(self, memory_percent: float, cpu_percent: float):
        """设置系统资源使用率
        
        Args:
            memory_percent: 内存使用率（百分比）
            cpu_percent: CPU使用率（百分比）
        """
        self.memory_usage_percent.set(memory_percent)
        self.system_cpu_usage_percent.set(cpu_percent)
    
    def set_disk_usage(self, mount_point: str, usage_percent: float):
        """设置磁盘使用率
        
        Args:
            mount_point: 挂载点
            usage_percent: 使用率（百分比）
        """
        self.disk_usage_percent.labels(mount_point=mount_point).set(usage_percent)
    
    # === 新增：数据质量监控方法 ===
    def set_token_data_quality_score(self, provider: str, model: str, score: float):
        """设置Token数据质量评分
        
        Args:
            provider: 提供商
            model: 模型名称
            score: 质量评分（0-100）
        """
        self.token_data_quality_score.labels(provider=provider, model=model).set(score)
    
    def set_data_completeness(self, table_name: str, field_name: str, completeness_percent: float):
        """设置数据完整性百分比
        
        Args:
            table_name: 表名
            field_name: 字段名
            completeness_percent: 完整性百分比
        """
        self.data_completeness_percent.labels(table_name=table_name, field_name=field_name).set(completeness_percent)
    
    def set_data_accuracy_score(self, data_type: str, score: float):
        """设置数据准确性评分
        
        Args:
            data_type: 数据类型
            score: 准确性评分（0-100）
        """
        self.data_accuracy_score.labels(data_type=data_type).set(score)
    
    def record_data_consistency_failure(self, check_type: str, table_name: str):
        """记录数据一致性检查失败
        
        Args:
            check_type: 检查类型
            table_name: 表名
        """
        self.data_consistency_failures_total.labels(check_type=check_type, table_name=table_name).inc()
    
    def record_data_anomaly(self, anomaly_type: str, severity: str):
        """记录数据异常检测
        
        Args:
            anomaly_type: 异常类型
            severity: 严重程度
        """
        self.data_anomaly_detected_total.labels(anomaly_type=anomaly_type, severity=severity).inc()
    
    def record_data_repair(self, repair_type: str, status: str):
        """记录数据修复操作
        
        Args:
            repair_type: 修复类型
            status: 操作状态
        """
        self.data_repair_operations_total.labels(repair_type=repair_type, status=status).inc()
    
    # === 新增：业务质量指标方法 ===
    def set_token_parsing_success_rate(self, provider: str, parsing_method: str, success_rate: float):
        """设置Token解析成功率
        
        Args:
            provider: 提供商
            parsing_method: 解析方法
            success_rate: 成功率（百分比）
        """
        self.token_parsing_success_rate.labels(provider=provider, parsing_method=parsing_method).set(success_rate)
    
    def set_cost_calculation_accuracy(self, model: str, provider: str, accuracy: float):
        """设置成本计算准确性
        
        Args:
            model: 模型名称
            provider: 提供商
            accuracy: 准确性（百分比）
        """
        self.cost_calculation_accuracy.labels(model=model, provider=provider).set(accuracy)
    
    def record_price_config_update(self, update_type: str, status: str):
        """记录价格配置更新
        
        Args:
            update_type: 更新类型
            status: 更新状态
        """
        self.price_config_updates_total.labels(update_type=update_type, status=status).inc()
    
    def record_alert_rule_trigger(self, rule_id: str, severity: str, status: str):
        """记录告警规则触发
        
        Args:
            rule_id: 规则ID
            severity: 严重程度
            status: 触发状态
        """
        self.alert_rule_triggers_total.labels(rule_id=rule_id, severity=severity, status=status).inc()
    
    def record_alert_suppression(self, suppression_type: str, rule_id: str):
        """记录告警抑制
        
        Args:
            suppression_type: 抑制类型
            rule_id: 规则ID
        """
        self.alert_suppressions_total.labels(suppression_type=suppression_type, rule_id=rule_id).inc()
    
    def record_intelligent_threshold_adjustment(self, metric_name: str, adjustment_type: str):
        """记录智能阈值调整
        
        Args:
            metric_name: 指标名称
            adjustment_type: 调整类型
        """
        self.intelligent_threshold_adjustments_total.labels(metric_name=metric_name, adjustment_type=adjustment_type).inc()

    def record_api_request(self, method: str, model: str, provider: str, 
                          duration: float, status: str = 'success',
                          error_type: Optional[str] = None):
        """记录API请求指标"""
        # 记录请求总数
        self.api_requests_total.labels(
            method=method,
            model=model,
            provider=provider,
            status=status
        ).inc()
        
        # 记录响应时间
        self.api_request_duration_seconds.labels(
            method=method,
            model=model,
            provider=provider
        ).observe(duration)
        
        # 记录错误
        if status == 'error' and error_type:
            self.api_errors_total.labels(
                method=method,
                model=model,
                provider=provider,
                error_type=error_type
            ).inc()
    
    def record_token_usage(self, model: str, provider: str, 
                          prompt_tokens: int, completion_tokens: int):
        """记录Token使用量"""
        self.tokens_used_total.labels(
            model=model,
            provider=provider,
            token_type='prompt'
        ).inc(prompt_tokens)
        
        self.tokens_used_total.labels(
            model=model,
            provider=provider,
            token_type='completion'
        ).inc(completion_tokens)
    
    def record_cost(self, model: str, provider: str, cost: float):
        """记录API调用成本"""
        self.cost_total.labels(
            model=model,
            provider=provider
        ).inc(cost)
    
    def record_retry(self, model: str, provider: str, retry_reason: str):
        """记录重试事件"""
        self.retries_total.labels(
            model=model,
            provider=provider,
            retry_reason=retry_reason
        ).inc()
    
    def record_cache_hit(self, cache_type: str):
        """记录缓存命中"""
        self.cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """记录缓存未命中"""
        self.cache_misses_total.labels(cache_type=cache_type).inc()
    
    def set_active_connections(self, provider: str, count: int):
        """设置活跃连接数"""
        self.active_connections.labels(provider=provider).set(count)
    
    def set_system_info(self, info: Dict[str, str]):
        """设置系统信息"""
        self.system_info.info(info)
    
    # === 新增：数据一致性指标方法 ===
    def record_consistency_check(self, check_type: str, status: str):
        """记录数据一致性检查
        
        Args:
            check_type: 检查类型（token_consistency, cost_consistency, tracing_consistency等）
            status: 检查状态（success, failed, error）
        """
        self.data_consistency_checks_total.labels(
            check_type=check_type,
            status=status
        ).inc()
    
    def record_consistency_issue(self, issue_type: str, severity: str, table_name: str):
        """记录数据一致性问题
        
        Args:
            issue_type: 问题类型（missing_data, inconsistent_data, orphaned_record等）
            severity: 严重程度（low, medium, high, critical）
            table_name: 表名
        """
        self.data_consistency_issues_total.labels(
            issue_type=issue_type,
            severity=severity,
            table_name=table_name
        ).inc()
    
    def record_auto_correction(self, operation_type: str, table_name: str, status: str):
        """记录自动修正操作
        
        Args:
            operation_type: 操作类型（insert, update, delete, recalculate）
            table_name: 表名
            status: 操作状态（success, failed, skipped）
        """
        self.auto_correction_operations_total.labels(
            operation_type=operation_type,
            table_name=table_name,
            status=status
        ).inc()
    
    def record_constraint_violation(self, constraint_type: str, table_name: str):
        """记录数据库约束违反
        
        Args:
            constraint_type: 约束类型（foreign_key, unique, check, not_null）
            table_name: 表名
        """
        self.database_constraint_violations_total.labels(
            constraint_type=constraint_type,
            table_name=table_name
        ).inc()
    
    def set_data_integrity_score(self, table_name: str, score: float):
        """设置数据完整性评分
        
        Args:
            table_name: 表名
            score: 评分（0-100）
        """
        self.data_integrity_score.labels(table_name=table_name).set(score)
    
    # === 新增：性能监控指标方法 ===
    def set_database_connections(self, active: int, idle: int):
        """设置数据库连接数
        
        Args:
            active: 活跃连接数
            idle: 空闲连接数
        """
        self.database_connections_active.set(active)
        self.database_connections_idle.set(idle)
    
    def record_database_query(self, query_type: str, table_name: str, duration: float):
        """记录数据库查询性能
        
        Args:
            query_type: 查询类型（select, insert, update, delete）
            table_name: 表名
            duration: 查询持续时间（秒）
        """
        self.database_query_duration_seconds.labels(
            query_type=query_type,
            table_name=table_name
        ).observe(duration)
    
    def set_memory_usage(self, component: str, bytes_used: int):
        """设置内存使用量
        
        Args:
            component: 组件名称
            bytes_used: 使用的字节数
        """
        self.memory_usage_bytes.labels(component=component).set(bytes_used)
    
    def set_cpu_usage(self, component: str, percent: float):
        """设置CPU使用率
        
        Args:
            component: 组件名称
            percent: 使用率百分比
        """
        self.cpu_usage_percent.labels(component=component).set(percent)
    
    def set_queue_length(self, queue_name: str, length: int):
        """设置队列长度
        
        Args:
            queue_name: 队列名称
            length: 队列长度
        """
        self.queue_length.labels(queue_name=queue_name).set(length)
    
    def record_processing_latency(self, operation_type: str, latency: float):
        """记录处理延迟
        
        Args:
            operation_type: 操作类型
            latency: 延迟时间（秒）
        """
        self.processing_latency_seconds.labels(operation_type=operation_type).observe(latency)
    
    # === 新增：业务指标方法 ===
    def set_active_users(self, time_window: str, count: int):
        """设置活跃用户数
        
        Args:
            time_window: 时间窗口（1h, 24h, 7d）
            count: 用户数量
        """
        self.active_users_total.labels(time_window=time_window).set(count)
    
    def record_model_usage(self, model_name: str, provider: str, user_type: str):
        """记录模型使用
        
        Args:
            model_name: 模型名称
            provider: 提供商
            user_type: 用户类型（free, premium, enterprise）
        """
        self.model_usage_requests_total.labels(
            model_name=model_name,
            provider=provider,
            user_type=user_type
        ).inc()
    
    def set_cost_efficiency(self, model_name: str, provider: str, ratio: float):
        """设置成本效率比
        
        Args:
            model_name: 模型名称
            provider: 提供商
            ratio: 效率比（输出token/成本）
        """
        self.cost_efficiency_ratio.labels(
            model_name=model_name,
            provider=provider
        ).set(ratio)
    
    def set_service_quality_score(self, service_type: str, score: float):
        """设置服务质量评分
        
        Args:
            service_type: 服务类型
            score: 质量评分（0-100）
        """
        self.service_quality_score.labels(service_type=service_type).set(score)
    
    def set_tracing_coverage(self, operation_type: str, percent: float):
        """设置追踪覆盖率
        
        Args:
            operation_type: 操作类型
            percent: 覆盖率百分比
        """
        self.tracing_coverage_percent.labels(operation_type=operation_type).set(percent)
    
    # === 新增：健康检查指标方法 ===
    def set_health_check_status(self, component: str, check_type: str, is_healthy: bool):
        """设置健康检查状态
        
        Args:
            component: 组件名称
            check_type: 检查类型
            is_healthy: 是否健康
        """
        self.health_check_status.labels(
            component=component,
            check_type=check_type
        ).set(1 if is_healthy else 0)
    
    def record_health_check_duration(self, component: str, check_type: str, duration: float):
        """记录健康检查响应时间
        
        Args:
            component: 组件名称
            check_type: 检查类型
            duration: 响应时间（秒）
        """
        self.health_check_duration_seconds.labels(
            component=component,
            check_type=check_type
        ).observe(duration)
    
    def set_degradation_status(self, component: str, degradation_type: str, is_degraded: bool):
        """设置降级状态
        
        Args:
            component: 组件名称
            degradation_type: 降级类型
            is_degraded: 是否降级
        """
        self.degradation_status.labels(
            component=component,
            degradation_type=degradation_type
        ).set(1 if is_degraded else 0)
    
    def get_metrics(self) -> str:
        """获取Prometheus格式的指标数据"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """获取指标数据的Content-Type"""
        return CONTENT_TYPE_LATEST


def get_prometheus_metrics() -> Optional[PrometheusMetrics]:
    """获取全局Prometheus指标实例
    
    Returns:
        PrometheusMetrics实例，如果未初始化则返回None
    """
    return _prometheus_metrics


def init_prometheus_metrics(registry: Optional[CollectorRegistry] = None) -> PrometheusMetrics:
    """初始化全局Prometheus指标实例
    
    Args:
        registry: Prometheus注册表
        
    Returns:
        PrometheusMetrics实例
    """
    global _prometheus_metrics
    _prometheus_metrics = PrometheusMetrics(registry)
    return _prometheus_metrics


def prometheus_middleware(func: Callable) -> Callable:
    """Prometheus监控中间件装饰器
    
    自动记录函数调用的指标信息。
    根据性能配置动态启用或禁用监控。
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 检查性能配置是否启用Prometheus监控
        from ..config.performance import get_performance_config
        perf_config = get_performance_config()
        middleware_config = perf_config.get_middleware_config()
        
        if not middleware_config.get('metrics_middleware', True):
            return func(*args, **kwargs)
        
        metrics = get_prometheus_metrics()
        if not metrics:
            return func(*args, **kwargs)
        
        # 提取参数
        method = func.__name__
        model = kwargs.get('model', 'unknown')
        provider = kwargs.get('provider', 'unknown')
        
        start_time = time.time()
        status = 'success'
        error_type = None
        
        try:
            result = func(*args, **kwargs)
            
            # 记录Token使用量和成本
            if hasattr(result, 'usage') and result.usage:
                usage = result.usage
                metrics.record_token_usage(
                    model=model,
                    provider=provider,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens
                )
                
                # 计算并记录成本
                from ..core.pricing import PricingCalculator
                cost = PricingCalculator.calculate_cost(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    model_name=model
                )
                if cost is not None:
                    metrics.record_cost(model=model, provider=provider, cost=cost)
            
            return result
            
        except HarborAIError as e:
            status = 'error'
            error_type = type(e).__name__
            raise
        except Exception as e:
            status = 'error'
            error_type = 'UnexpectedError'
            raise
        finally:
            duration = time.time() - start_time
            metrics.record_api_request(
                method=method,
                model=model,
                provider=provider,
                duration=duration,
                status=status,
                error_type=error_type
            )
    
    return wrapper


# 异步版本的中间件
def prometheus_async_middleware(func: Callable) -> Callable:
    """异步Prometheus监控中间件装饰器
    
    根据性能配置动态启用或禁用监控。
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # 检查性能配置是否启用Prometheus监控
        from ..config.performance import get_performance_config
        perf_config = get_performance_config()
        middleware_config = perf_config.get_middleware_config()
        
        if not middleware_config.get('metrics_middleware', True):
            return await func(*args, **kwargs)
        
        metrics = get_prometheus_metrics()
        if not metrics:
            return await func(*args, **kwargs)
        
        # 提取参数
        method = func.__name__
        model = kwargs.get('model', 'unknown')
        provider = kwargs.get('provider', 'unknown')
        
        start_time = time.time()
        status = 'success'
        error_type = None
        
        try:
            result = await func(*args, **kwargs)
            
            # 记录Token使用量和成本
            if hasattr(result, 'usage') and result.usage:
                usage = result.usage
                metrics.record_token_usage(
                    model=model,
                    provider=provider,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens
                )
                
                # 计算并记录成本
                from ..core.pricing import PricingCalculator
                cost = PricingCalculator.calculate_cost(
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    model_name=model
                )
                if cost is not None:
                    metrics.record_cost(model=model, provider=provider, cost=cost)
            
            return result
            
        except HarborAIError as e:
            status = 'error'
            error_type = type(e).__name__
            raise
        except Exception as e:
            status = 'error'
            error_type = 'UnexpectedError'
            raise
        finally:
            duration = time.time() - start_time
            metrics.record_api_request(
                method=method,
                model=model,
                provider=provider,
                duration=duration,
                status=status,
                error_type=error_type
            )
    
    return wrapper