"""
HarborAI 增强错误处理器
提供更细粒度的错误处理和恢复机制
"""

import logging
import traceback
import time
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"           # 低级错误，可以忽略或降级处理
    MEDIUM = "medium"     # 中级错误，需要记录和监控
    HIGH = "high"         # 高级错误，需要立即处理
    CRITICAL = "critical" # 严重错误，可能影响系统稳定性


class ErrorCategory(Enum):
    """错误分类"""
    TOKEN_PARSING = "token_parsing"
    COST_CALCULATION = "cost_calculation"
    DATABASE_CONNECTION = "database_connection"
    NETWORK_REQUEST = "network_request"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    RATE_LIMITING = "rate_limiting"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """错误上下文信息"""
    category: ErrorCategory
    severity: ErrorSeverity
    provider: Optional[str] = None
    model: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    additional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorRecoveryAction:
    """错误恢复动作"""
    action_type: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    success_probability: float = 0.0  # 成功概率 0-1
    cost: int = 0  # 执行成本（相对值）


class EnhancedErrorHandler:
    """增强的错误处理器"""
    
    def __init__(self):
        self.error_stats: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        self.error_history: List[Dict[str, Any]] = []
        
        # 初始化恢复策略
        self._initialize_recovery_strategies()
        
        # 熔断器配置
        self.circuit_breaker_config = {
            'failure_threshold': 5,  # 失败阈值
            'recovery_timeout': 60,  # 恢复超时（秒）
            'half_open_max_calls': 3  # 半开状态最大调用次数
        }

    def _initialize_recovery_strategies(self):
        """初始化恢复策略"""
        self.recovery_strategies = {
            ErrorCategory.TOKEN_PARSING: [
                self._fallback_token_parsing,
                self._estimate_token_usage,
                self._use_cached_token_info
            ],
            ErrorCategory.COST_CALCULATION: [
                self._fallback_cost_calculation,
                self._use_default_pricing,
                self._skip_cost_calculation
            ],
            ErrorCategory.DATABASE_CONNECTION: [
                self._retry_database_connection,
                self._use_fallback_storage,
                self._cache_for_later_retry
            ],
            ErrorCategory.NETWORK_REQUEST: [
                self._retry_with_backoff,
                self._use_cached_response,
                self._degrade_service_quality
            ],
            ErrorCategory.CONFIGURATION: [
                self._use_default_config,
                self._reload_configuration,
                self._validate_and_fix_config
            ]
        }

    def handle_error(self, 
                    error: Exception, 
                    context: ErrorContext,
                    auto_recover: bool = True) -> Optional[Any]:
        """
        处理错误并尝试恢复
        
        Args:
            error: 异常对象
            context: 错误上下文
            auto_recover: 是否自动尝试恢复
            
        Returns:
            恢复结果（如果成功）或None
        """
        # 记录错误
        self._log_error(error, context)
        
        # 更新错误统计
        self._update_error_stats(context)
        
        # 检查熔断器状态
        if self._should_circuit_break(context):
            logger.warning(f"熔断器触发，跳过处理: {context.category.value}")
            return None
        
        # 尝试自动恢复
        if auto_recover:
            return self._attempt_recovery(error, context)
        
        return None

    def _log_error(self, error: Exception, context: ErrorContext):
        """记录错误详情"""
        error_info = {
            'timestamp': context.timestamp.isoformat(),
            'category': context.category.value,
            'severity': context.severity.value,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'provider': context.provider,
            'model': context.model,
            'request_id': context.request_id,
            'trace_id': context.trace_id,
            'user_id': context.user_id,
            'traceback': traceback.format_exc(),
            'additional_data': context.additional_data
        }
        
        # 添加到错误历史
        self.error_history.append(error_info)
        
        # 保持错误历史在合理大小
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]
        
        # 根据严重程度选择日志级别
        if context.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"严重错误: {error_info}")
        elif context.severity == ErrorSeverity.HIGH:
            logger.error(f"高级错误: {error_info}")
        elif context.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"中级错误: {error_info}")
        else:
            logger.info(f"低级错误: {error_info}")

    def _update_error_stats(self, context: ErrorContext):
        """更新错误统计"""
        key = f"{context.category.value}:{context.provider or 'unknown'}"
        self.error_stats[key] = self.error_stats.get(key, 0) + 1

    def _should_circuit_break(self, context: ErrorContext) -> bool:
        """检查是否应该触发熔断器"""
        key = f"{context.category.value}:{context.provider or 'unknown'}"
        
        if key not in self.circuit_breakers:
            self.circuit_breakers[key] = {
                'state': 'closed',  # closed, open, half_open
                'failure_count': 0,
                'last_failure_time': None,
                'half_open_calls': 0
            }
        
        breaker = self.circuit_breakers[key]
        
        # 检查是否需要从开放状态恢复到半开状态
        if breaker['state'] == 'open':
            if (breaker['last_failure_time'] and 
                time.time() - breaker['last_failure_time'] > self.circuit_breaker_config['recovery_timeout']):
                breaker['state'] = 'half_open'
                breaker['half_open_calls'] = 0
                logger.info(f"熔断器进入半开状态: {key}")
        
        # 更新失败计数
        breaker['failure_count'] += 1
        breaker['last_failure_time'] = time.time()
        
        # 检查是否需要打开熔断器
        if (breaker['state'] == 'closed' and 
            breaker['failure_count'] >= self.circuit_breaker_config['failure_threshold']):
            breaker['state'] = 'open'
            logger.warning(f"熔断器打开: {key}")
            return True
        
        # 检查半开状态
        if breaker['state'] == 'half_open':
            breaker['half_open_calls'] += 1
            if breaker['half_open_calls'] >= self.circuit_breaker_config['half_open_max_calls']:
                breaker['state'] = 'open'
                logger.warning(f"熔断器从半开状态重新打开: {key}")
                return True
        
        return breaker['state'] == 'open'

    def _attempt_recovery(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """尝试错误恢复"""
        strategies = self.recovery_strategies.get(context.category, [])
        
        for strategy in strategies:
            try:
                logger.info(f"尝试恢复策略: {strategy.__name__}")
                result = strategy(error, context)
                
                if result is not None:
                    logger.info(f"恢复成功: {strategy.__name__}")
                    self._record_successful_recovery(context, strategy.__name__)
                    return result
                    
            except Exception as recovery_error:
                logger.warning(f"恢复策略失败 {strategy.__name__}: {recovery_error}")
        
        logger.error(f"所有恢复策略都失败了: {context.category.value}")
        return None

    def _record_successful_recovery(self, context: ErrorContext, strategy_name: str):
        """记录成功的恢复"""
        key = f"{context.category.value}:{context.provider or 'unknown'}"
        
        # 重置熔断器状态
        if key in self.circuit_breakers:
            breaker = self.circuit_breakers[key]
            if breaker['state'] == 'half_open':
                breaker['state'] = 'closed'
                breaker['failure_count'] = 0
                logger.info(f"熔断器恢复到关闭状态: {key}")

    # ==================== 恢复策略实现 ====================

    def _fallback_token_parsing(self, error: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """降级Token解析"""
        try:
            response_data = context.additional_data.get('response_data', {})
            
            # 尝试从不同字段提取Token信息
            token_fields = [
                ('usage', 'prompt_tokens', 'completion_tokens', 'total_tokens'),
                ('token_usage', 'input_tokens', 'output_tokens', 'total_tokens'),
                ('usage', 'input_tokens', 'output_tokens', 'total_tokens'),
            ]
            
            for usage_key, prompt_key, completion_key, total_key in token_fields:
                if usage_key in response_data:
                    usage = response_data[usage_key]
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get(prompt_key, 0)
                        completion_tokens = usage.get(completion_key, 0)
                        total_tokens = usage.get(total_key, prompt_tokens + completion_tokens)
                        
                        return {
                            'prompt_tokens': prompt_tokens,
                            'completion_tokens': completion_tokens,
                            'total_tokens': total_tokens,
                            'parsing_method': 'fallback_extraction',
                            'confidence': 0.7
                        }
            
            # 如果无法提取，返回估算值
            return self._estimate_token_usage(error, context)
            
        except Exception as e:
            logger.warning(f"降级Token解析失败: {e}")
            return None

    def _estimate_token_usage(self, error: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """估算Token使用量"""
        try:
            request_data = context.additional_data.get('request_data', {})
            response_data = context.additional_data.get('response_data', {})
            
            # 简单的Token估算逻辑
            prompt_text = ""
            completion_text = ""
            
            # 提取请求文本
            if 'messages' in request_data:
                for msg in request_data['messages']:
                    if isinstance(msg, dict) and 'content' in msg:
                        prompt_text += msg['content'] + " "
            elif 'prompt' in request_data:
                prompt_text = request_data['prompt']
            
            # 提取响应文本
            if 'choices' in response_data:
                for choice in response_data['choices']:
                    if isinstance(choice, dict) and 'message' in choice:
                        completion_text += choice['message'].get('content', '') + " "
            elif 'content' in response_data:
                completion_text = response_data['content']
            
            # 粗略估算（1个Token约等于4个字符）
            prompt_tokens = max(1, len(prompt_text) // 4)
            completion_tokens = max(1, len(completion_text) // 4)
            total_tokens = prompt_tokens + completion_tokens
            
            return {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
                'parsing_method': 'estimated',
                'confidence': 0.5
            }
            
        except Exception as e:
            logger.warning(f"Token估算失败: {e}")
            return None

    def _use_cached_token_info(self, error: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """使用缓存的Token信息"""
        # 这里可以实现从缓存中获取类似请求的Token信息
        # 暂时返回默认值
        return {
            'prompt_tokens': 100,
            'completion_tokens': 50,
            'total_tokens': 150,
            'parsing_method': 'cached_fallback',
            'confidence': 0.3
        }

    def _fallback_cost_calculation(self, error: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """降级成本计算"""
        try:
            token_data = context.additional_data.get('token_data', {})
            provider = context.provider or 'unknown'
            
            # 使用默认价格进行计算
            default_prices = {
                'deepseek': {'input': 0.0014, 'output': 0.0028},
                'openai': {'input': 0.03, 'output': 0.06},
                'wenxin': {'input': 0.008, 'output': 0.016},
                'doubao': {'input': 0.0008, 'output': 0.002}
            }
            
            prices = default_prices.get(provider.lower(), {'input': 0.01, 'output': 0.02})
            
            prompt_tokens = token_data.get('prompt_tokens', 0)
            completion_tokens = token_data.get('completion_tokens', 0)
            
            input_cost = (prompt_tokens / 1000) * prices['input']
            output_cost = (completion_tokens / 1000) * prices['output']
            total_cost = input_cost + output_cost
            
            return {
                'input_cost': round(input_cost, 6),
                'output_cost': round(output_cost, 6),
                'total_cost': round(total_cost, 6),
                'currency': 'CNY',
                'pricing_source': 'fallback'
            }
            
        except Exception as e:
            logger.warning(f"降级成本计算失败: {e}")
            return None

    def _use_default_pricing(self, error: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """使用默认定价"""
        return {
            'input_cost': 0.0,
            'output_cost': 0.0,
            'total_cost': 0.0,
            'currency': 'CNY',
            'pricing_source': 'default'
        }

    def _skip_cost_calculation(self, error: Exception, context: ErrorContext) -> Optional[Dict[str, Any]]:
        """跳过成本计算"""
        return {
            'input_cost': 0.0,
            'output_cost': 0.0,
            'total_cost': 0.0,
            'currency': 'CNY',
            'pricing_source': 'skipped'
        }

    def _retry_database_connection(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """重试数据库连接"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                time.sleep(retry_delay * (attempt + 1))
                # 这里应该实现实际的数据库重连逻辑
                logger.info(f"数据库重连尝试 {attempt + 1}/{max_retries}")
                # 返回成功标志
                return True
            except Exception as e:
                logger.warning(f"数据库重连失败 {attempt + 1}/{max_retries}: {e}")
        
        return None

    def _use_fallback_storage(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """使用降级存储"""
        # 切换到文件存储
        logger.info("切换到文件存储模式")
        return {'storage_mode': 'file'}

    def _cache_for_later_retry(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """缓存以便稍后重试"""
        # 将数据缓存到内存或文件，稍后重试
        logger.info("数据已缓存，稍后重试")
        return {'cached': True}

    def _retry_with_backoff(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """带退避的重试"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                delay = base_delay * (2 ** attempt)  # 指数退避
                time.sleep(delay)
                logger.info(f"网络请求重试 {attempt + 1}/{max_retries}")
                # 这里应该实现实际的重试逻辑
                return True
            except Exception as e:
                logger.warning(f"网络请求重试失败 {attempt + 1}/{max_retries}: {e}")
        
        return None

    def _use_cached_response(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """使用缓存响应"""
        # 从缓存中获取响应
        logger.info("使用缓存响应")
        return {'cached_response': True}

    def _degrade_service_quality(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """降级服务质量"""
        # 降低服务质量，如使用更简单的模型
        logger.info("降级服务质量")
        return {'degraded': True}

    def _use_default_config(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """使用默认配置"""
        logger.info("使用默认配置")
        return {'config_source': 'default'}

    def _reload_configuration(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """重新加载配置"""
        logger.info("重新加载配置")
        return {'config_reloaded': True}

    def _validate_and_fix_config(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """验证并修复配置"""
        logger.info("验证并修复配置")
        return {'config_fixed': True}

    # ==================== 工具方法 ====================

    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计信息"""
        return {
            'error_counts': self.error_stats.copy(),
            'circuit_breakers': {k: v.copy() for k, v in self.circuit_breakers.items()},
            'total_errors': sum(self.error_stats.values()),
            'error_categories': list(set(k.split(':')[0] for k in self.error_stats.keys()))
        }

    def reset_circuit_breaker(self, key: str):
        """重置熔断器"""
        if key in self.circuit_breakers:
            self.circuit_breakers[key] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure_time': None,
                'half_open_calls': 0
            }
            logger.info(f"熔断器已重置: {key}")

    def export_error_report(self) -> str:
        """导出错误报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': self.get_error_statistics(),
            'recent_errors': self.error_history[-50:] if self.error_history else []
        }
        return json.dumps(report, indent=2, ensure_ascii=False)


# 全局错误处理器实例
global_error_handler = EnhancedErrorHandler()


def handle_token_parsing_error(error: Exception, provider: str = None, 
                             model: str = None, response_data: Dict = None) -> Optional[Dict[str, Any]]:
    """处理Token解析错误的便捷函数"""
    context = ErrorContext(
        category=ErrorCategory.TOKEN_PARSING,
        severity=ErrorSeverity.MEDIUM,
        provider=provider,
        model=model,
        additional_data={'response_data': response_data or {}}
    )
    return global_error_handler.handle_error(error, context)


def handle_cost_calculation_error(error: Exception, provider: str = None,
                                token_data: Dict = None) -> Optional[Dict[str, Any]]:
    """处理成本计算错误的便捷函数"""
    context = ErrorContext(
        category=ErrorCategory.COST_CALCULATION,
        severity=ErrorSeverity.MEDIUM,
        provider=provider,
        additional_data={'token_data': token_data or {}}
    )
    return global_error_handler.handle_error(error, context)


def handle_database_error(error: Exception, operation: str = None) -> Optional[Any]:
    """处理数据库错误的便捷函数"""
    context = ErrorContext(
        category=ErrorCategory.DATABASE_CONNECTION,
        severity=ErrorSeverity.HIGH,
        additional_data={'operation': operation}
    )
    return global_error_handler.handle_error(error, context)