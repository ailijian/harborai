#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警管理器

负责告警规则的管理、告警状态的跟踪和告警的触发
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from uuid import uuid4
import json

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重程度"""
    CRITICAL = "critical"  # 严重：系统不可用
    HIGH = "high"         # 高：功能受影响
    MEDIUM = "medium"     # 中：性能下降
    LOW = "low"          # 低：潜在问题
    INFO = "info"        # 信息：状态变化


class AlertStatus(Enum):
    """告警状态"""
    PENDING = "pending"      # 待处理
    FIRING = "firing"        # 触发中
    RESOLVED = "resolved"    # 已解决
    SUPPRESSED = "suppressed" # 已抑制
    ACKNOWLEDGED = "acknowledged" # 已确认


class AlertCondition(Enum):
    """告警条件类型"""
    THRESHOLD = "threshold"           # 阈值条件
    RATE_OF_CHANGE = "rate_of_change" # 变化率条件
    ANOMALY = "anomaly"              # 异常检测
    PATTERN = "pattern"              # 模式匹配
    COMPOSITE = "composite"          # 复合条件


@dataclass
class AlertRule:
    """告警规则"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition_type: AlertCondition
    metric_name: str
    threshold: float
    comparison: str  # >, <, >=, <=, ==, !=
    duration: int  # 持续时间（秒）
    evaluation_interval: int  # 评估间隔（秒）
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 高级条件
    rate_window: Optional[int] = None  # 变化率窗口（秒）
    anomaly_threshold: Optional[float] = None  # 异常阈值
    pattern_regex: Optional[str] = None  # 模式正则表达式
    composite_rules: List[str] = field(default_factory=list)  # 复合规则ID列表
    composite_operator: str = "AND"  # 复合操作符：AND, OR
    
    # 通知配置
    notification_channels: List[str] = field(default_factory=list)
    notification_template: Optional[str] = None
    escalation_rules: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity.value,
            "condition_type": self.condition_type.value,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "duration": self.duration,
            "evaluation_interval": self.evaluation_interval,
            "labels": self.labels,
            "annotations": self.annotations,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "rate_window": self.rate_window,
            "anomaly_threshold": self.anomaly_threshold,
            "pattern_regex": self.pattern_regex,
            "composite_rules": self.composite_rules,
            "composite_operator": self.composite_operator,
            "notification_channels": self.notification_channels,
            "notification_template": self.notification_template,
            "escalation_rules": self.escalation_rules
        }


@dataclass
class Alert:
    """告警实例"""
    id: str
    rule_id: str
    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    metric_value: float
    threshold: float
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    suppressed_until: Optional[datetime] = None
    notification_sent: bool = False
    escalation_level: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "labels": self.labels,
            "annotations": self.annotations,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "suppressed_until": self.suppressed_until.isoformat() if self.suppressed_until else None,
            "notification_sent": self.notification_sent,
            "escalation_level": self.escalation_level
        }


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.metric_providers: Dict[str, Callable] = {}
        self.notification_service = None
        self.suppression_manager = None
        self.alert_history_service = None
        self.evaluation_tasks: Dict[str, asyncio.Task] = {}
        self.running = False
        
    async def initialize(self):
        """初始化告警管理器"""
        logger.info("初始化告警管理器")
        await self._load_default_rules()
        
    def set_notification_service(self, service):
        """设置通知服务"""
        self.notification_service = service
        
    def set_suppression_manager(self, manager):
        """设置抑制管理器"""
        self.suppression_manager = manager
        
    def set_alert_history(self, history):
        """设置告警历史服务"""
        self.alert_history_service = history
        
    def register_metric_provider(self, metric_name: str, provider: Callable):
        """注册指标提供者"""
        self.metric_providers[metric_name] = provider
        logger.info(f"注册指标提供者: {metric_name}")
        
    async def add_rule(self, rule: AlertRule) -> bool:
        """添加告警规则"""
        try:
            self.rules[rule.id] = rule
            
            # 如果规则启用且管理器正在运行，启动评估任务
            if rule.enabled and self.running:
                await self._start_rule_evaluation(rule)
                
            logger.info(f"添加告警规则: {rule.name} ({rule.id})")
            return True
            
        except Exception as e:
            logger.error(f"添加告警规则失败: {e}")
            return False
            
    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新告警规则"""
        try:
            if rule_id not in self.rules:
                logger.warning(f"告警规则不存在: {rule_id}")
                return False
                
            rule = self.rules[rule_id]
            
            # 更新规则属性
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
                    
            rule.updated_at = datetime.now()
            
            # 重启评估任务
            if rule_id in self.evaluation_tasks:
                self.evaluation_tasks[rule_id].cancel()
                
            if rule.enabled and self.running:
                await self._start_rule_evaluation(rule)
                
            logger.info(f"更新告警规则: {rule.name} ({rule_id})")
            return True
            
        except Exception as e:
            logger.error(f"更新告警规则失败: {e}")
            return False
            
    async def remove_rule(self, rule_id: str) -> bool:
        """删除告警规则"""
        try:
            if rule_id not in self.rules:
                logger.warning(f"告警规则不存在: {rule_id}")
                return False
                
            # 停止评估任务
            if rule_id in self.evaluation_tasks:
                self.evaluation_tasks[rule_id].cancel()
                del self.evaluation_tasks[rule_id]
                
            # 解决相关的活跃告警
            alerts_to_resolve = [
                alert for alert in self.active_alerts.values()
                if alert.rule_id == rule_id
            ]
            
            for alert in alerts_to_resolve:
                await self._resolve_alert(alert.id, "规则已删除")
                
            # 删除规则
            rule_name = self.rules[rule_id].name
            del self.rules[rule_id]
            
            logger.info(f"删除告警规则: {rule_name} ({rule_id})")
            return True
            
        except Exception as e:
            logger.error(f"删除告警规则失败: {e}")
            return False
            
    async def start(self):
        """启动告警管理器"""
        if self.running:
            logger.warning("告警管理器已在运行")
            return
            
        self.running = True
        logger.info("启动告警管理器")
        
        # 启动所有启用规则的评估任务
        for rule in self.rules.values():
            if rule.enabled:
                await self._start_rule_evaluation(rule)
                
    async def stop(self):
        """停止告警管理器"""
        if not self.running:
            return
            
        self.running = False
        logger.info("停止告警管理器")
        
        # 取消所有评估任务
        for task in self.evaluation_tasks.values():
            task.cancel()
            
        self.evaluation_tasks.clear()
        
        # 等待任务完成
        await asyncio.sleep(0.1)
        
    async def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """确认告警"""
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"活跃告警不存在: {alert_id}")
                return False
                
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now()
            alert.acknowledged_by = user
            
            # 记录到历史
            if self.alert_history_service:
                await self.alert_history_service.record_event(
                    alert_id, "acknowledged", {"user": user}
                )
                
            logger.info(f"告警已确认: {alert_id} by {user}")
            return True
            
        except Exception as e:
            logger.error(f"确认告警失败: {e}")
            return False
            
    async def suppress_alert(self, alert_id: str, duration: int, reason: str) -> bool:
        """抑制告警"""
        try:
            if alert_id not in self.active_alerts:
                logger.warning(f"活跃告警不存在: {alert_id}")
                return False
                
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = datetime.now() + timedelta(seconds=duration)
            
            # 记录到历史
            if self.alert_history_service:
                await self.alert_history_service.record_event(
                    alert_id, "suppressed", {"duration": duration, "reason": reason}
                )
                
            logger.info(f"告警已抑制: {alert_id} for {duration}s - {reason}")
            return True
            
        except Exception as e:
            logger.error(f"抑制告警失败: {e}")
            return False
            
    async def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """获取活跃告警"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
            
        return sorted(alerts, key=lambda x: x.started_at, reverse=True)
        
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计信息"""
        now = datetime.now()
        
        # 按严重程度统计活跃告警
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                alert for alert in self.active_alerts.values()
                if alert.severity == severity
            ])
            
        # 按状态统计活跃告警
        status_counts = {}
        for status in AlertStatus:
            status_counts[status.value] = len([
                alert for alert in self.active_alerts.values()
                if alert.status == status
            ])
            
        # 24小时内的告警趋势
        last_24h = now - timedelta(hours=24)
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.started_at >= last_24h
        ]
        
        return {
            "total_rules": len(self.rules),
            "enabled_rules": len([r for r in self.rules.values() if r.enabled]),
            "active_alerts": len(self.active_alerts),
            "severity_distribution": severity_counts,
            "status_distribution": status_counts,
            "alerts_last_24h": len(recent_alerts),
            "mean_resolution_time": self._calculate_mean_resolution_time(),
            "top_firing_rules": self._get_top_firing_rules()
        }
        
    async def _start_rule_evaluation(self, rule: AlertRule):
        """启动规则评估任务"""
        async def evaluate_rule():
            while self.running and rule.enabled:
                try:
                    await self._evaluate_rule(rule)
                    await asyncio.sleep(rule.evaluation_interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"评估规则失败 {rule.name}: {e}")
                    await asyncio.sleep(rule.evaluation_interval)
                    
        task = asyncio.create_task(evaluate_rule())
        self.evaluation_tasks[rule.id] = task
        
    async def _evaluate_rule(self, rule: AlertRule):
        """评估单个规则"""
        try:
            # 获取指标值
            metric_value = await self._get_metric_value(rule.metric_name, rule.labels)
            if metric_value is None:
                return
                
            # 评估条件
            condition_met = await self._evaluate_condition(rule, metric_value)
            
            # 查找现有告警
            existing_alert = self._find_alert_by_rule(rule.id)
            
            if condition_met:
                if existing_alert is None:
                    # 创建新告警
                    await self._create_alert(rule, metric_value)
                elif existing_alert.status == AlertStatus.RESOLVED:
                    # 重新激活告警
                    await self._reactivate_alert(existing_alert, metric_value)
            else:
                if existing_alert and existing_alert.status == AlertStatus.FIRING:
                    # 解决告警
                    await self._resolve_alert(existing_alert.id, "条件不再满足")
                    
        except Exception as e:
            logger.error(f"评估规则失败 {rule.name}: {e}")
            
    async def _evaluate_condition(self, rule: AlertRule, metric_value: float) -> bool:
        """评估告警条件"""
        if rule.condition_type == AlertCondition.THRESHOLD:
            return self._evaluate_threshold_condition(rule, metric_value)
        elif rule.condition_type == AlertCondition.RATE_OF_CHANGE:
            return await self._evaluate_rate_condition(rule, metric_value)
        elif rule.condition_type == AlertCondition.ANOMALY:
            return await self._evaluate_anomaly_condition(rule, metric_value)
        elif rule.condition_type == AlertCondition.PATTERN:
            return await self._evaluate_pattern_condition(rule, metric_value)
        elif rule.condition_type == AlertCondition.COMPOSITE:
            return await self._evaluate_composite_condition(rule)
        else:
            logger.warning(f"未知的条件类型: {rule.condition_type}")
            return False
            
    def _evaluate_threshold_condition(self, rule: AlertRule, metric_value: float) -> bool:
        """评估阈值条件"""
        comparison = rule.comparison
        threshold = rule.threshold
        
        if comparison == ">":
            return metric_value > threshold
        elif comparison == "<":
            return metric_value < threshold
        elif comparison == ">=":
            return metric_value >= threshold
        elif comparison == "<=":
            return metric_value <= threshold
        elif comparison == "==":
            return metric_value == threshold
        elif comparison == "!=":
            return metric_value != threshold
        else:
            logger.warning(f"未知的比较操作符: {comparison}")
            return False
            
    async def _get_metric_value(self, metric_name: str, labels: Dict[str, str]) -> Optional[float]:
        """获取指标值"""
        if metric_name not in self.metric_providers:
            logger.warning(f"未找到指标提供者: {metric_name}")
            return None
            
        try:
            provider = self.metric_providers[metric_name]
            return await provider(labels)
        except Exception as e:
            logger.error(f"获取指标值失败 {metric_name}: {e}")
            return None
            
    def _find_alert_by_rule(self, rule_id: str) -> Optional[Alert]:
        """根据规则ID查找告警"""
        for alert in self.active_alerts.values():
            if alert.rule_id == rule_id:
                return alert
        return None
        
    async def _create_alert(self, rule: AlertRule, metric_value: float):
        """创建新告警"""
        alert_id = str(uuid4())
        
        alert = Alert(
            id=alert_id,
            rule_id=rule.id,
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            message=self._generate_alert_message(rule, metric_value),
            metric_value=metric_value,
            threshold=rule.threshold,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy()
        )
        
        self.active_alerts[alert_id] = alert
        
        # 检查是否被抑制
        if self.suppression_manager:
            is_suppressed = await self.suppression_manager.is_suppressed(alert)
            if is_suppressed:
                alert.status = AlertStatus.SUPPRESSED
                
        # 发送通知
        if alert.status != AlertStatus.SUPPRESSED and self.notification_service:
            await self.notification_service.send_alert_notification(alert, rule)
            alert.notification_sent = True
            
        # 记录到历史
        if self.alert_history_service:
            await self.alert_history_service.record_event(
                alert_id, "created", {"metric_value": metric_value}
            )
            
        logger.info(f"创建告警: {rule.name} - {alert.message}")
        
    async def _resolve_alert(self, alert_id: str, reason: str):
        """解决告警"""
        if alert_id not in self.active_alerts:
            return
            
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        # 移动到历史记录
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]
        
        # 发送解决通知
        if self.notification_service:
            await self.notification_service.send_resolution_notification(alert, reason)
            
        # 记录到历史
        if self.alert_history_service:
            await self.alert_history_service.record_event(
                alert_id, "resolved", {"reason": reason}
            )
            
        logger.info(f"解决告警: {alert.rule_name} - {reason}")
        
    def _generate_alert_message(self, rule: AlertRule, metric_value: float) -> str:
        """生成告警消息"""
        return f"{rule.name}: {rule.metric_name} = {metric_value} {rule.comparison} {rule.threshold}"
        
    def _calculate_mean_resolution_time(self) -> float:
        """计算平均解决时间（分钟）"""
        resolved_alerts = [
            alert for alert in self.alert_history
            if alert.resolved_at is not None
        ]
        
        if not resolved_alerts:
            return 0.0
            
        total_time = sum([
            (alert.resolved_at - alert.started_at).total_seconds()
            for alert in resolved_alerts
        ])
        
        return total_time / len(resolved_alerts) / 60  # 转换为分钟
        
    def _get_top_firing_rules(self, limit: int = 5) -> List[Dict[str, Any]]:
        """获取触发最多的规则"""
        rule_counts = {}
        
        for alert in self.alert_history:
            rule_id = alert.rule_id
            if rule_id not in rule_counts:
                rule_counts[rule_id] = {
                    "rule_id": rule_id,
                    "rule_name": alert.rule_name,
                    "count": 0
                }
            rule_counts[rule_id]["count"] += 1
            
        # 按触发次数排序
        sorted_rules = sorted(
            rule_counts.values(),
            key=lambda x: x["count"],
            reverse=True
        )
        
        return sorted_rules[:limit]
        
    async def _load_default_rules(self):
        """加载默认告警规则"""
        default_rules = [
            # 数据库连接告警
            AlertRule(
                id="db_connection_failure",
                name="数据库连接失败",
                description="数据库连接失败率过高",
                severity=AlertSeverity.CRITICAL,
                condition_type=AlertCondition.THRESHOLD,
                metric_name="database_connection_failure_rate",
                threshold=0.1,  # 10%
                comparison=">",
                duration=60,
                evaluation_interval=30,
                labels={"component": "database"},
                annotations={"runbook": "检查数据库连接配置和网络状态"}
            ),
            
            # API响应时间告警
            AlertRule(
                id="api_response_time_high",
                name="API响应时间过高",
                description="API平均响应时间超过阈值",
                severity=AlertSeverity.HIGH,
                condition_type=AlertCondition.THRESHOLD,
                metric_name="api_response_time_avg",
                threshold=2000,  # 2秒
                comparison=">",
                duration=300,  # 5分钟
                evaluation_interval=60,
                labels={"component": "api"},
                annotations={"runbook": "检查API性能和数据库查询优化"}
            ),
            
            # 错误率告警
            AlertRule(
                id="error_rate_high",
                name="错误率过高",
                description="系统错误率超过正常水平",
                severity=AlertSeverity.HIGH,
                condition_type=AlertCondition.THRESHOLD,
                metric_name="error_rate",
                threshold=0.05,  # 5%
                comparison=">",
                duration=180,  # 3分钟
                evaluation_interval=60,
                labels={"component": "system"},
                annotations={"runbook": "检查错误日志和系统状态"}
            ),
            
            # 内存使用率告警
            AlertRule(
                id="memory_usage_high",
                name="内存使用率过高",
                description="系统内存使用率超过安全阈值",
                severity=AlertSeverity.MEDIUM,
                condition_type=AlertCondition.THRESHOLD,
                metric_name="memory_usage_percent",
                threshold=85,  # 85%
                comparison=">",
                duration=600,  # 10分钟
                evaluation_interval=120,
                labels={"component": "system"},
                annotations={"runbook": "检查内存泄漏和优化内存使用"}
            ),
            
            # 磁盘空间告警
            AlertRule(
                id="disk_space_low",
                name="磁盘空间不足",
                description="磁盘可用空间低于安全阈值",
                severity=AlertSeverity.MEDIUM,
                condition_type=AlertCondition.THRESHOLD,
                metric_name="disk_free_percent",
                threshold=15,  # 15%
                comparison="<",
                duration=300,  # 5分钟
                evaluation_interval=300,
                labels={"component": "system"},
                annotations={"runbook": "清理磁盘空间或扩容"}
            )
        ]
        
        for rule in default_rules:
            await self.add_rule(rule)
            
        logger.info(f"加载了 {len(default_rules)} 个默认告警规则")