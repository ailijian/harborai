"""
HarborAI 配置验证器
提供环境变量和配置参数的严格验证机制
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """验证级别"""
    ERROR = "error"      # 错误，阻止启动
    WARNING = "warning"  # 警告，记录但继续
    INFO = "info"       # 信息，仅记录


@dataclass
class ValidationResult:
    """验证结果"""
    level: ValidationLevel
    key: str
    message: str
    current_value: Optional[str] = None
    expected_format: Optional[str] = None
    suggestion: Optional[str] = None


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        
        # 定义价格配置的正则表达式
        self.price_pattern = re.compile(r'^[A-Z_]+_(INPUT_|OUTPUT_)?PRICE$')
        
        # 定义支持的货币
        self.supported_currencies = {
            'CNY', 'USD', 'EUR', 'JPY', 'GBP', 
            'CAD', 'AUD', 'CHF', 'SEK', 'NOK'
        }
        
        # 定义支持的提供商
        self.supported_providers = {
            'OPENAI', 'DEEPSEEK', 'WENXIN', 'DOUBAO', 
            'ERNIE', 'CLAUDE', 'GEMINI'
        }
        
        # 定义OpenTelemetry导出器类型
        self.supported_otel_exporters = {
            'otlp', 'jaeger', 'zipkin', 'console', 'none'
        }

    def validate_all_configs(self) -> List[ValidationResult]:
        """验证所有配置"""
        self.results.clear()
        
        # 验证价格配置
        self._validate_pricing_configs()
        
        # 验证成本追踪配置
        self._validate_cost_tracking_configs()
        
        # 验证OpenTelemetry配置
        self._validate_opentelemetry_configs()
        
        # 验证数据库配置
        self._validate_database_configs()
        
        # 验证日志配置
        self._validate_logging_configs()
        
        # 验证监控配置
        self._validate_monitoring_configs()
        
        return self.results

    def _validate_pricing_configs(self):
        """验证价格配置"""
        logger.info("开始验证价格配置...")
        
        # 获取所有价格相关的环境变量
        price_vars = {k: v for k, v in os.environ.items() 
                     if self.price_pattern.match(k)}
        
        if not price_vars:
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                key="PRICING_CONFIG",
                message="未找到任何价格配置环境变量",
                suggestion="设置如 DEEPSEEK_INPUT_PRICE=0.0014 的环境变量"
            ))
            return
        
        for key, value in price_vars.items():
            self._validate_price_value(key, value)
            self._validate_price_key_format(key)

    def _validate_price_value(self, key: str, value: str):
        """验证价格值"""
        try:
            price = float(value)
            
            # 检查价格是否为非负数
            if price < 0:
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    key=key,
                    message=f"价格必须为非负数",
                    current_value=value,
                    expected_format="非负浮点数",
                    suggestion=f"设置为正数，如 {key}=0.001"
                ))
            
            # 检查价格是否过高（可能的配置错误）
            elif price > 1.0:
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    key=key,
                    message=f"价格似乎过高，请确认单位是否正确",
                    current_value=value,
                    suggestion="通常价格应该是每千Token的成本，单位为元或美分"
                ))
            
            # 检查价格是否为0（可能的配置遗漏）
            elif price == 0:
                self.results.append(ValidationResult(
                    level=ValidationLevel.INFO,
                    key=key,
                    message=f"价格设置为0，将不计算成本",
                    current_value=value
                ))
                
        except ValueError:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                key=key,
                message=f"价格值必须为有效数字",
                current_value=value,
                expected_format="浮点数，如 0.0014",
                suggestion=f"修正为有效数字，如 {key}=0.001"
            ))

    def _validate_price_key_format(self, key: str):
        """验证价格键格式"""
        # 检查提供商名称是否支持
        parts = key.split('_')
        if len(parts) >= 2:
            provider = parts[0]
            if provider not in self.supported_providers:
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    key=key,
                    message=f"未知的提供商: {provider}",
                    suggestion=f"支持的提供商: {', '.join(self.supported_providers)}"
                ))

    def _validate_cost_tracking_configs(self):
        """验证成本追踪配置"""
        logger.info("开始验证成本追踪配置...")
        
        # 验证成本追踪开关
        cost_tracking = os.getenv('HARBORAI_COST_TRACKING', 'true').lower()
        if cost_tracking not in ['true', 'false', '1', '0', 'yes', 'no']:
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                key='HARBORAI_COST_TRACKING',
                message="成本追踪配置值不标准",
                current_value=cost_tracking,
                expected_format="true/false 或 1/0",
                suggestion="设置为 true 或 false"
            ))
        
        # 验证货币配置
        currency = os.getenv('COST_CURRENCY', 'CNY').upper()
        if currency not in self.supported_currencies:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                key='COST_CURRENCY',
                message=f"不支持的货币代码: {currency}",
                current_value=currency,
                expected_format="ISO 4217货币代码",
                suggestion=f"使用支持的货币: {', '.join(self.supported_currencies)}"
            ))
        
        # 验证快速路径配置
        fast_path_skip = os.getenv('HARBORAI_FAST_PATH_SKIP_COST', 'false').lower()
        if fast_path_skip not in ['true', 'false', '1', '0']:
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                key='HARBORAI_FAST_PATH_SKIP_COST',
                message="快速路径成本跳过配置值不标准",
                current_value=fast_path_skip,
                expected_format="true/false",
                suggestion="设置为 true 或 false"
            ))

    def _validate_opentelemetry_configs(self):
        """验证OpenTelemetry配置"""
        logger.info("开始验证OpenTelemetry配置...")
        
        # 验证OpenTelemetry开关
        otel_enabled = os.getenv('OTEL_ENABLED', 'false').lower()
        if otel_enabled not in ['true', 'false', '1', '0']:
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                key='OTEL_ENABLED',
                message="OpenTelemetry启用配置值不标准",
                current_value=otel_enabled,
                expected_format="true/false",
                suggestion="设置为 true 或 false"
            ))
        
        # 如果启用了OpenTelemetry，验证相关配置
        if otel_enabled in ['true', '1']:
            self._validate_otel_service_config()
            self._validate_otel_exporter_config()
            self._validate_otel_resource_config()

    def _validate_otel_service_config(self):
        """验证OpenTelemetry服务配置"""
        service_name = os.getenv('OTEL_SERVICE_NAME', '')
        if not service_name:
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                key='OTEL_SERVICE_NAME',
                message="未设置OpenTelemetry服务名称",
                suggestion="设置为 OTEL_SERVICE_NAME=harborai"
            ))
        elif not re.match(r'^[a-zA-Z][a-zA-Z0-9._-]*$', service_name):
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                key='OTEL_SERVICE_NAME',
                message="OpenTelemetry服务名称格式不正确",
                current_value=service_name,
                expected_format="字母开头，可包含字母、数字、点、下划线、连字符",
                suggestion="使用如 harborai 或 harborai-api 的格式"
            ))

    def _validate_otel_exporter_config(self):
        """验证OpenTelemetry导出器配置"""
        exporter_type = os.getenv('OTEL_EXPORTER_TYPE', 'otlp').lower()
        if exporter_type not in self.supported_otel_exporters:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                key='OTEL_EXPORTER_TYPE',
                message=f"不支持的OpenTelemetry导出器类型: {exporter_type}",
                current_value=exporter_type,
                expected_format="支持的导出器类型",
                suggestion=f"使用: {', '.join(self.supported_otel_exporters)}"
            ))
        
        # 验证OTLP端点配置
        if exporter_type == 'otlp':
            otlp_endpoint = os.getenv('OTEL_EXPORTER_OTLP_ENDPOINT', '')
            if not otlp_endpoint:
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    key='OTEL_EXPORTER_OTLP_ENDPOINT',
                    message="未设置OTLP导出端点",
                    suggestion="设置为 http://jaeger:14268/api/traces"
                ))
            elif not self._is_valid_url(otlp_endpoint):
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    key='OTEL_EXPORTER_OTLP_ENDPOINT',
                    message="OTLP端点URL格式不正确",
                    current_value=otlp_endpoint,
                    expected_format="有效的HTTP/HTTPS URL",
                    suggestion="使用如 http://localhost:14268/api/traces 的格式"
                ))

    def _validate_otel_resource_config(self):
        """验证OpenTelemetry资源配置"""
        resource_attrs = os.getenv('OTEL_RESOURCE_ATTRIBUTES', '')
        if resource_attrs:
            # 验证资源属性格式
            try:
                # 资源属性应该是 key1=value1,key2=value2 的格式
                attrs = resource_attrs.split(',')
                for attr in attrs:
                    if '=' not in attr:
                        self.results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            key='OTEL_RESOURCE_ATTRIBUTES',
                            message=f"资源属性格式不正确: {attr}",
                            current_value=resource_attrs,
                            expected_format="key1=value1,key2=value2",
                            suggestion="使用如 service.version=1.0.0,deployment.environment=production 的格式"
                        ))
                        break
            except Exception as e:
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    key='OTEL_RESOURCE_ATTRIBUTES',
                    message=f"资源属性解析失败: {str(e)}",
                    current_value=resource_attrs
                ))

    def _validate_database_configs(self):
        """验证数据库配置"""
        logger.info("开始验证数据库配置...")
        
        # 验证PostgreSQL配置
        db_url = os.getenv('DATABASE_URL', '')
        if db_url:
            if not self._is_valid_postgres_url(db_url):
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    key='DATABASE_URL',
                    message="PostgreSQL连接URL格式不正确",
                    expected_format="postgresql://user:password@host:port/database",
                    suggestion="检查用户名、密码、主机和数据库名称"
                ))
        
        # 验证连接池配置
        max_connections = os.getenv('DB_MAX_CONNECTIONS', '20')
        try:
            max_conn = int(max_connections)
            if max_conn <= 0:
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    key='DB_MAX_CONNECTIONS',
                    message="数据库最大连接数必须大于0",
                    current_value=max_connections,
                    suggestion="设置为正整数，如 20"
                ))
            elif max_conn > 100:
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    key='DB_MAX_CONNECTIONS',
                    message="数据库最大连接数过高，可能影响性能",
                    current_value=max_connections,
                    suggestion="建议设置为 10-50 之间"
                ))
        except ValueError:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                key='DB_MAX_CONNECTIONS',
                message="数据库最大连接数必须为整数",
                current_value=max_connections,
                expected_format="正整数",
                suggestion="设置为如 20 的整数值"
            ))

    def _validate_logging_configs(self):
        """验证日志配置"""
        logger.info("开始验证日志配置...")
        
        # 验证日志级别
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if log_level not in valid_levels:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                key='LOG_LEVEL',
                message=f"无效的日志级别: {log_level}",
                current_value=log_level,
                expected_format="有效的日志级别",
                suggestion=f"使用: {', '.join(valid_levels)}"
            ))
        
        # 验证日志格式
        log_format = os.getenv('LOG_FORMAT', 'json').lower()
        if log_format not in ['json', 'text', 'structured']:
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                key='LOG_FORMAT',
                message=f"不推荐的日志格式: {log_format}",
                current_value=log_format,
                suggestion="推荐使用 json 格式以便于解析"
            ))

    def _validate_monitoring_configs(self):
        """验证监控配置"""
        logger.info("开始验证监控配置...")
        
        # 验证Prometheus配置
        prometheus_enabled = os.getenv('PROMETHEUS_ENABLED', 'true').lower()
        if prometheus_enabled not in ['true', 'false', '1', '0']:
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                key='PROMETHEUS_ENABLED',
                message="Prometheus启用配置值不标准",
                current_value=prometheus_enabled,
                expected_format="true/false",
                suggestion="设置为 true 或 false"
            ))
        
        # 验证Prometheus端口
        prometheus_port = os.getenv('PROMETHEUS_PORT', '8000')
        try:
            port = int(prometheus_port)
            if port <= 0 or port > 65535:
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    key='PROMETHEUS_PORT',
                    message="Prometheus端口号超出有效范围",
                    current_value=prometheus_port,
                    expected_format="1-65535之间的整数",
                    suggestion="使用如 8000 的有效端口号"
                ))
        except ValueError:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                key='PROMETHEUS_PORT',
                message="Prometheus端口号必须为整数",
                current_value=prometheus_port,
                expected_format="整数",
                suggestion="设置为如 8000 的整数值"
            ))

    def _is_valid_url(self, url: str) -> bool:
        """检查URL是否有效"""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return url_pattern.match(url) is not None

    def _is_valid_postgres_url(self, url: str) -> bool:
        """检查PostgreSQL URL是否有效"""
        postgres_pattern = re.compile(
            r'^postgresql://[^:]+:[^@]+@[^:]+:\d+/[^/]+$'
        )
        return postgres_pattern.match(url) is not None

    def get_error_count(self) -> int:
        """获取错误数量"""
        return len([r for r in self.results if r.level == ValidationLevel.ERROR])

    def get_warning_count(self) -> int:
        """获取警告数量"""
        return len([r for r in self.results if r.level == ValidationLevel.WARNING])

    def has_errors(self) -> bool:
        """是否有错误"""
        return self.get_error_count() > 0

    def print_results(self):
        """打印验证结果"""
        if not self.results:
            logger.info("✅ 所有配置验证通过")
            return
        
        error_count = self.get_error_count()
        warning_count = self.get_warning_count()
        
        logger.info(f"配置验证完成: {error_count} 个错误, {warning_count} 个警告")
        
        for result in self.results:
            level_icon = {
                ValidationLevel.ERROR: "❌",
                ValidationLevel.WARNING: "⚠️",
                ValidationLevel.INFO: "ℹ️"
            }[result.level]
            
            message = f"{level_icon} [{result.level.value.upper()}] {result.key}: {result.message}"
            
            if result.current_value:
                message += f" (当前值: {result.current_value})"
            
            if result.suggestion:
                message += f" 建议: {result.suggestion}"
            
            if result.level == ValidationLevel.ERROR:
                logger.error(message)
            elif result.level == ValidationLevel.WARNING:
                logger.warning(message)
            else:
                logger.info(message)

    def export_results_json(self) -> str:
        """导出验证结果为JSON"""
        results_data = []
        for result in self.results:
            results_data.append({
                'level': result.level.value,
                'key': result.key,
                'message': result.message,
                'current_value': result.current_value,
                'expected_format': result.expected_format,
                'suggestion': result.suggestion
            })
        
        return json.dumps({
            'summary': {
                'total': len(self.results),
                'errors': self.get_error_count(),
                'warnings': self.get_warning_count(),
                'info': len([r for r in self.results if r.level == ValidationLevel.INFO])
            },
            'results': results_data
        }, indent=2, ensure_ascii=False)


def validate_startup_configs() -> bool:
    """启动时验证配置，返回是否可以继续启动"""
    validator = ConfigValidator()
    results = validator.validate_all_configs()
    
    validator.print_results()
    
    # 如果有错误，阻止启动
    if validator.has_errors():
        logger.error("❌ 配置验证失败，存在错误配置，无法启动")
        return False
    
    logger.info("✅ 配置验证通过，可以启动")
    return True


if __name__ == "__main__":
    # 命令行工具模式
    import sys
    
    validator = ConfigValidator()
    results = validator.validate_all_configs()
    
    validator.print_results()
    
    # 输出JSON格式结果
    if "--json" in sys.argv:
        print(validator.export_results_json())
    
    # 如果有错误，退出码为1
    sys.exit(1 if validator.has_errors() else 0)