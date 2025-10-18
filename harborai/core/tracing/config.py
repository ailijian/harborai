#!/usr/bin/env python3
"""
分布式追踪配置管理模块

负责管理OpenTelemetry和HarborAI追踪系统的配置，包括：
- 环境变量配置加载
- 追踪器初始化配置
- 服务发现和注册配置
- 采样策略配置

作者: HarborAI团队
创建时间: 2025-01-15
版本: v2.0.0
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

import structlog
try:
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased, AlwaysOn, AlwaysOff
except ImportError:
    # 兼容不同版本的OpenTelemetry
    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
    
    class AlwaysOn:
        """总是采样"""
        def should_sample(self, *args, **kwargs):
            from opentelemetry.sdk.trace.sampling import SamplingResult, Decision
            return SamplingResult(Decision.RECORD_AND_SAMPLE)
    
    class AlwaysOff:
        """从不采样"""
        def should_sample(self, *args, **kwargs):
            from opentelemetry.sdk.trace.sampling import SamplingResult, Decision
            return SamplingResult(Decision.DROP)


@dataclass
class OTLPConfig:
    """OTLP导出器配置"""
    endpoint: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30  # 秒
    compression: str = "gzip"  # gzip, deflate, none
    insecure: bool = False


@dataclass
class SamplingConfig:
    """采样配置"""
    strategy: str = "ratio"  # ratio, always_on, always_off, custom
    ratio: float = 1.0  # 采样比例 (0.0-1.0)
    
    # 自定义采样规则
    custom_rules: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ResourceConfig:
    """资源配置"""
    service_name: str = "harborai-logging"
    service_version: str = "2.0.0"
    service_namespace: str = "harborai"
    deployment_environment: str = "production"
    
    # 自定义资源属性
    custom_attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class TracingConfig:
    """完整的追踪配置"""
    enabled: bool = True
    
    # 基础配置
    resource: ResourceConfig = field(default_factory=ResourceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    otlp: OTLPConfig = field(default_factory=OTLPConfig)
    
    # HarborAI特定配置
    hb_trace_id_prefix: str = "hb"
    include_sensitive_data: bool = False
    
    # 批处理配置
    batch_export_timeout: int = 30000  # 毫秒
    max_export_batch_size: int = 512
    max_queue_size: int = 2048
    
    # 调试配置
    debug_mode: bool = False
    console_exporter: bool = False


class TracingConfigLoader:
    """追踪配置加载器"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
    
    def load_from_env(self) -> TracingConfig:
        """从环境变量加载配置"""
        try:
            # 基础配置
            enabled = self._get_bool_env("OTEL_ENABLED", True)
            debug_mode = self._get_bool_env("OTEL_DEBUG", False)
            console_exporter = self._get_bool_env("OTEL_CONSOLE_EXPORTER", False)
            
            # 资源配置
            resource = ResourceConfig(
                service_name=os.getenv("OTEL_SERVICE_NAME", "harborai-logging"),
                service_version=os.getenv("OTEL_SERVICE_VERSION", "2.0.0"),
                service_namespace=os.getenv("OTEL_SERVICE_NAMESPACE", "harborai"),
                deployment_environment=os.getenv("DEPLOYMENT_ENVIRONMENT", "production"),
                custom_attributes=self._parse_custom_attributes()
            )
            
            # 采样配置
            sampling = SamplingConfig(
                strategy=os.getenv("OTEL_SAMPLING_STRATEGY", "ratio"),
                ratio=float(os.getenv("OTEL_SAMPLING_RATIO", "1.0")),
                custom_rules=self._parse_sampling_rules()
            )
            
            # OTLP配置
            otlp = OTLPConfig(
                endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
                headers=self._parse_otlp_headers(),
                timeout=int(os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT", "30")),
                compression=os.getenv("OTEL_EXPORTER_OTLP_COMPRESSION", "gzip"),
                insecure=self._get_bool_env("OTEL_EXPORTER_OTLP_INSECURE", False)
            )
            
            # HarborAI特定配置
            hb_trace_id_prefix = os.getenv("HARBORAI_TRACE_ID_PREFIX", "hb")
            include_sensitive_data = self._get_bool_env("HARBORAI_INCLUDE_SENSITIVE_DATA", False)
            
            # 批处理配置
            batch_export_timeout = int(os.getenv("OTEL_BSP_EXPORT_TIMEOUT", "30000"))
            max_export_batch_size = int(os.getenv("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "512"))
            max_queue_size = int(os.getenv("OTEL_BSP_MAX_QUEUE_SIZE", "2048"))
            
            config = TracingConfig(
                enabled=enabled,
                resource=resource,
                sampling=sampling,
                otlp=otlp,
                hb_trace_id_prefix=hb_trace_id_prefix,
                include_sensitive_data=include_sensitive_data,
                batch_export_timeout=batch_export_timeout,
                max_export_batch_size=max_export_batch_size,
                max_queue_size=max_queue_size,
                debug_mode=debug_mode,
                console_exporter=console_exporter
            )
            
            self.logger.info(
                "追踪配置加载成功",
                enabled=config.enabled,
                service_name=config.resource.service_name,
                environment=config.resource.deployment_environment,
                otlp_endpoint=config.otlp.endpoint,
                sampling_strategy=config.sampling.strategy,
                sampling_ratio=config.sampling.ratio
            )
            
            return config
            
        except Exception as e:
            self.logger.error(
                "加载追踪配置失败",
                error=str(e)
            )
            # 返回默认配置
            return TracingConfig()
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """获取布尔类型环境变量"""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    def _parse_custom_attributes(self) -> Dict[str, str]:
        """解析自定义资源属性"""
        attributes = {}
        
        # 从环境变量中解析 OTEL_RESOURCE_ATTRIBUTES
        resource_attrs = os.getenv("OTEL_RESOURCE_ATTRIBUTES", "")
        if resource_attrs:
            try:
                for pair in resource_attrs.split(","):
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        attributes[key.strip()] = value.strip()
            except Exception as e:
                self.logger.warning(
                    "解析资源属性失败",
                    error=str(e),
                    resource_attrs=resource_attrs
                )
        
        # 添加HarborAI特定属性
        attributes.update({
            "ai.system": "harborai",
            "ai.framework": "harborai-logging",
            "telemetry.sdk.name": "opentelemetry",
            "telemetry.sdk.language": "python"
        })
        
        return attributes
    
    def _parse_otlp_headers(self) -> Dict[str, str]:
        """解析OTLP头部"""
        headers = {}
        
        # 从环境变量中解析头部
        headers_env = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        if headers_env:
            try:
                for pair in headers_env.split(","):
                    if "=" in pair:
                        key, value = pair.split("=", 1)
                        headers[key.strip()] = value.strip()
            except Exception as e:
                self.logger.warning(
                    "解析OTLP头部失败",
                    error=str(e),
                    headers_env=headers_env
                )
        
        # 从单独的环境变量中解析头部
        for key, value in os.environ.items():
            if key.startswith("OTEL_EXPORTER_OTLP_HEADERS_"):
                header_name = key[len("OTEL_EXPORTER_OTLP_HEADERS_"):]
                headers[header_name] = value
        
        return headers
    
    def _parse_sampling_rules(self) -> List[Dict[str, Any]]:
        """解析自定义采样规则"""
        rules = []
        
        # 从环境变量中解析采样规则
        rules_env = os.getenv("OTEL_SAMPLING_RULES", "")
        if rules_env:
            try:
                import json
                rules = json.loads(rules_env)
            except Exception as e:
                self.logger.warning(
                    "解析采样规则失败",
                    error=str(e),
                    rules_env=rules_env
                )
        
        return rules
    
    def validate_config(self, config: TracingConfig) -> bool:
        """验证配置的有效性"""
        try:
            # 验证基础配置
            if not isinstance(config.enabled, bool):
                self.logger.error("enabled必须是布尔值")
                return False
            
            # 验证采样比例
            if not 0.0 <= config.sampling.ratio <= 1.0:
                self.logger.error(
                    "采样比例必须在0.0-1.0之间",
                    ratio=config.sampling.ratio
                )
                return False
            
            # 验证OTLP端点
            if config.otlp.endpoint and not config.otlp.endpoint.startswith(("http://", "https://")):
                self.logger.error(
                    "OTLP端点必须是有效的HTTP/HTTPS URL",
                    endpoint=config.otlp.endpoint
                )
                return False
            
            # 验证批处理配置
            if config.max_export_batch_size <= 0:
                self.logger.error(
                    "批处理大小必须大于0",
                    batch_size=config.max_export_batch_size
                )
                return False
            
            if config.max_queue_size <= 0:
                self.logger.error(
                    "队列大小必须大于0",
                    queue_size=config.max_queue_size
                )
                return False
            
            self.logger.debug("配置验证通过")
            return True
            
        except Exception as e:
            self.logger.error(
                "配置验证失败",
                error=str(e)
            )
            return False
    
    def get_sampler(self, config: SamplingConfig):
        """根据配置获取采样器"""
        try:
            if config.strategy == "always_on":
                return AlwaysOn()
            elif config.strategy == "always_off":
                return AlwaysOff()
            elif config.strategy == "ratio":
                return TraceIdRatioBased(config.ratio)
            else:
                self.logger.warning(
                    "未知的采样策略，使用默认比例采样",
                    strategy=config.strategy
                )
                return TraceIdRatioBased(config.ratio)
                
        except Exception as e:
            self.logger.error(
                "创建采样器失败",
                error=str(e),
                strategy=config.strategy
            )
            return TraceIdRatioBased(1.0)


# 全局配置加载器实例
_config_loader = TracingConfigLoader()


def load_tracing_config() -> TracingConfig:
    """加载追踪配置"""
    return _config_loader.load_from_env()


def validate_tracing_config(config: TracingConfig) -> bool:
    """验证追踪配置"""
    return _config_loader.validate_config(config)


def get_sampler_from_config(config: SamplingConfig):
    """从配置获取采样器"""
    return _config_loader.get_sampler(config)