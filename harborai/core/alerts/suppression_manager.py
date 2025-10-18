#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抑制管理器

负责告警抑制规则的管理和执行，避免告警风暴
"""

import asyncio
import logging
import re
import hashlib
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Pattern, Tuple
from dataclasses import dataclass, field
from uuid import uuid4
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class SuppressionType(Enum):
    """抑制类型"""
    TIME_BASED = "time_based"        # 基于时间的抑制
    LABEL_BASED = "label_based"      # 基于标签的抑制
    PATTERN_BASED = "pattern_based"  # 基于模式的抑制
    DEPENDENCY = "dependency"        # 基于依赖的抑制
    MAINTENANCE = "maintenance"      # 维护期间抑制
    RATE_LIMIT = "rate_limit"       # 基于频率的抑制
    DUPLICATE = "duplicate"          # 重复告警抑制
    SMART = "smart"                  # 智能抑制


class SuppressionStatus(Enum):
    """抑制状态"""
    ACTIVE = "active"      # 激活中
    INACTIVE = "inactive"  # 未激活
    EXPIRED = "expired"    # 已过期
    DISABLED = "disabled"  # 已禁用


@dataclass
class SuppressionRule:
    """抑制规则"""
    id: str
    name: str
    description: str
    type: SuppressionType
    status: SuppressionStatus = SuppressionStatus.ACTIVE
    
    # 时间相关
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration: Optional[int] = None  # 持续时间（秒）
    
    # 匹配条件
    alert_name_pattern: Optional[str] = None
    severity_levels: List[str] = field(default_factory=list)
    label_matchers: Dict[str, str] = field(default_factory=dict)  # 标签匹配器
    annotation_matchers: Dict[str, str] = field(default_factory=dict)
    
    # 依赖抑制
    dependency_alerts: List[str] = field(default_factory=list)  # 依赖的告警ID
    dependency_rules: List[str] = field(default_factory=list)   # 依赖的规则ID
    
    # 频率限制
    rate_limit_count: Optional[int] = None  # 频率限制次数
    rate_limit_window: Optional[int] = None  # 频率限制窗口（秒）
    
    # 维护窗口
    maintenance_windows: List[Dict[str, Any]] = field(default_factory=list)
    
    # 重复告警抑制
    duplicate_window: Optional[int] = None  # 重复检测窗口（秒）
    duplicate_threshold: Optional[int] = None  # 重复阈值
    
    # 智能抑制参数
    smart_threshold: Optional[float] = None  # 智能抑制阈值
    smart_window: Optional[int] = None  # 智能分析窗口（秒）
    smart_algorithm: Optional[str] = None  # 智能算法类型
    
    # 元数据
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # 编译后的正则表达式（运行时使用）
    _compiled_pattern: Optional[Pattern] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.alert_name_pattern:
            try:
                self._compiled_pattern = re.compile(self.alert_name_pattern)
            except re.error as e:
                logger.error(f"编译正则表达式失败 {self.alert_name_pattern}: {e}")
                
    def matches_alert(self, alert) -> bool:
        """检查告警是否匹配抑制规则"""
        try:
            # 检查告警名称模式
            if self.alert_name_pattern and self._compiled_pattern:
                if not self._compiled_pattern.match(alert.rule_name):
                    return False
                    
            # 检查严重程度
            if self.severity_levels and alert.severity.value not in self.severity_levels:
                return False
                
            # 检查标签匹配
            if self.label_matchers:
                for key, pattern in self.label_matchers.items():
                    if key not in alert.labels:
                        return False
                    if not re.match(pattern, alert.labels[key]):
                        return False
                        
            # 检查注释匹配
            if self.annotation_matchers:
                for key, pattern in self.annotation_matchers.items():
                    if key not in alert.annotations:
                        return False
                    if not re.match(pattern, alert.annotations[key]):
                        return False
                        
            return True
            
        except Exception as e:
            logger.error(f"检查告警匹配失败: {e}")
            return False
            
    def is_active(self, current_time: Optional[datetime] = None) -> bool:
        """检查抑制规则是否激活"""
        if self.status != SuppressionStatus.ACTIVE:
            return False
            
        if current_time is None:
            current_time = datetime.now()
            
        # 检查时间范围
        if self.start_time and current_time < self.start_time:
            return False
        if self.end_time and current_time > self.end_time:
            return False
            
        # 检查维护窗口
        if self.maintenance_windows:
            return self._is_in_maintenance_window(current_time)
            
        return True
        
    def _is_in_maintenance_window(self, current_time: datetime) -> bool:
        """检查是否在维护窗口内"""
        for window in self.maintenance_windows:
            start_time = datetime.fromisoformat(window['start_time'])
            end_time = datetime.fromisoformat(window['end_time'])
            
            if start_time <= current_time <= end_time:
                return True
                
            # 检查周期性维护窗口
            if window.get('recurring'):
                # 简单的周期性检查（每周重复）
                if window['recurring'] == 'weekly':
                    days_diff = (current_time - start_time).days
                    if days_diff % 7 == 0:
                        # 检查时间是否在窗口内
                        current_time_of_day = current_time.time()
                        start_time_of_day = start_time.time()
                        end_time_of_day = end_time.time()
                        
                        if start_time_of_day <= current_time_of_day <= end_time_of_day:
                            return True
                            
        return False
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type.value,
            "status": self.status.value,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration": self.duration,
            "alert_name_pattern": self.alert_name_pattern,
            "severity_levels": self.severity_levels,
            "label_matchers": self.label_matchers,
            "annotation_matchers": self.annotation_matchers,
            "dependency_alerts": self.dependency_alerts,
            "dependency_rules": self.dependency_rules,
            "rate_limit_count": self.rate_limit_count,
            "rate_limit_window": self.rate_limit_window,
            "maintenance_windows": self.maintenance_windows,
            "duplicate_window": self.duplicate_window,
            "duplicate_threshold": self.duplicate_threshold,
            "smart_threshold": self.smart_threshold,
            "smart_window": self.smart_window,
            "smart_algorithm": self.smart_algorithm,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class SuppressionEvent:
    """抑制事件"""
    id: str
    rule_id: str
    alert_id: str
    event_type: str  # suppressed, unsuppressed
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertFingerprint:
    """告警指纹，用于重复检测"""
    rule_name: str
    labels_hash: str
    severity: str
    
    @classmethod
    def from_alert(cls, alert) -> 'AlertFingerprint':
        """从告警创建指纹"""
        # 创建标签哈希
        labels_str = "|".join(f"{k}={v}" for k, v in sorted(alert.labels.items()))
        labels_hash = hashlib.md5(labels_str.encode()).hexdigest()
        
        return cls(
            rule_name=alert.rule_name,
            labels_hash=labels_hash,
            severity=alert.severity.value
        )
    
    def __hash__(self):
        return hash((self.rule_name, self.labels_hash, self.severity))
    
    def __eq__(self, other):
        return (self.rule_name == other.rule_name and 
                self.labels_hash == other.labels_hash and 
                self.severity == other.severity)


class SuppressionManager:
    """抑制管理器"""
    
    def __init__(self):
        self.rules: Dict[str, SuppressionRule] = {}
        self.suppressed_alerts: Set[str] = set()
        self.suppression_events: List[SuppressionEvent] = []
        self.rate_limit_counters: Dict[str, List[datetime]] = defaultdict(list)
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # 重复告警检测
        self.alert_history: Dict[AlertFingerprint, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.duplicate_cache: Dict[str, datetime] = {}  # alert_id -> last_seen
        
        # 智能抑制
        self.alert_patterns: Dict[str, List[Tuple[datetime, Any]]] = defaultdict(list)
        self.smart_cache: Dict[str, Dict[str, Any]] = {}
        
    async def initialize(self):
        """初始化抑制管理器"""
        await self._load_default_rules()
        logger.info("抑制管理器初始化完成")
        
    async def add_rule(self, rule: SuppressionRule) -> bool:
        """添加抑制规则"""
        try:
            if rule.id in self.rules:
                logger.warning(f"抑制规则已存在: {rule.id}")
                return False
                
            self.rules[rule.id] = rule
            self._update_dependency_graph(rule)
            
            logger.info(f"添加抑制规则: {rule.name} ({rule.id})")
            return True
            
        except Exception as e:
            logger.error(f"添加抑制规则失败: {e}")
            return False
            
    async def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新抑制规则"""
        try:
            if rule_id not in self.rules:
                logger.warning(f"抑制规则不存在: {rule_id}")
                return False
                
            rule = self.rules[rule_id]
            
            # 更新字段
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
                    
            rule.updated_at = datetime.now()
            
            # 重新编译正则表达式
            if 'alert_name_pattern' in updates:
                try:
                    rule._compiled_pattern = re.compile(rule.alert_name_pattern)
                except re.error as e:
                    logger.error(f"编译正则表达式失败: {e}")
                    return False
                    
            self._update_dependency_graph(rule)
            logger.info(f"更新抑制规则: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"更新抑制规则失败: {e}")
            return False
            
    async def remove_rule(self, rule_id: str) -> bool:
        """删除抑制规则"""
        try:
            if rule_id not in self.rules:
                logger.warning(f"抑制规则不存在: {rule_id}")
                return False
                
            # 清理依赖图
            self._cleanup_dependency_graph(rule_id)
            
            # 删除规则
            del self.rules[rule_id]
            
            # 清理相关计数器
            keys_to_remove = [key for key in self.rate_limit_counters.keys() if key.startswith(f"{rule_id}_")]
            for key in keys_to_remove:
                del self.rate_limit_counters[key]
                
            logger.info(f"删除抑制规则: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"删除抑制规则失败: {e}")
            return False
            
    async def is_suppressed(self, alert) -> bool:
        """检查告警是否被抑制"""
        try:
            current_time = datetime.now()
            
            for rule in self.rules.values():
                if not rule.is_active(current_time):
                    continue
                    
                if await self._check_suppression_rule(rule, alert, current_time):
                    # 记录抑制事件
                    await self._record_suppression_event(
                        rule.id, 
                        alert.id, 
                        "suppressed", 
                        f"匹配抑制规则: {rule.name}"
                    )
                    
                    self.suppressed_alerts.add(alert.id)
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"检查告警抑制失败: {e}")
            return False
            
    async def _check_suppression_rule(self, rule: SuppressionRule, alert, current_time: datetime) -> bool:
        """检查单个抑制规则"""
        try:
            if rule.type == SuppressionType.TIME_BASED:
                return await self._check_time_based_suppression(rule, alert, current_time)
            elif rule.type == SuppressionType.LABEL_BASED:
                return await self._check_label_based_suppression(rule, alert)
            elif rule.type == SuppressionType.PATTERN_BASED:
                return await self._check_pattern_based_suppression(rule, alert)
            elif rule.type == SuppressionType.DEPENDENCY:
                return await self._check_dependency_suppression(rule, alert)
            elif rule.type == SuppressionType.MAINTENANCE:
                return await self._check_maintenance_suppression(rule, alert, current_time)
            elif rule.type == SuppressionType.RATE_LIMIT:
                return await self._check_rate_limit_suppression(rule, alert, current_time)
            elif rule.type == SuppressionType.DUPLICATE:
                return await self._check_duplicate_suppression(rule, alert, current_time)
            elif rule.type == SuppressionType.SMART:
                return await self._check_smart_suppression(rule, alert, current_time)
            else:
                logger.warning(f"未知的抑制类型: {rule.type}")
                return False
                
        except Exception as e:
            logger.error(f"检查抑制规则失败 {rule.name}: {e}")
            return False
            
    async def _check_time_based_suppression(self, rule: SuppressionRule, alert, current_time: datetime) -> bool:
        """检查基于时间的抑制"""
        if not rule.matches_alert(alert):
            return False
            
        # 检查时间范围
        if rule.start_time and current_time < rule.start_time:
            return False
        if rule.end_time and current_time > rule.end_time:
            return False
            
        return True
        
    async def _check_label_based_suppression(self, rule: SuppressionRule, alert) -> bool:
        """检查基于标签的抑制"""
        return rule.matches_alert(alert)
        
    async def _check_pattern_based_suppression(self, rule: SuppressionRule, alert) -> bool:
        """检查基于模式的抑制"""
        return rule.matches_alert(alert)
        
    async def _check_dependency_suppression(self, rule: SuppressionRule, alert) -> bool:
        """检查基于依赖的抑制"""
        if not rule.matches_alert(alert):
            return False
            
        # 检查依赖的告警是否存在
        for dep_alert_id in rule.dependency_alerts:
            if dep_alert_id in self.suppressed_alerts:
                return True
                
        # 检查依赖的规则是否激活
        for dep_rule_id in rule.dependency_rules:
            if dep_rule_id in self.rules:
                dep_rule = self.rules[dep_rule_id]
                if dep_rule.is_active():
                    return True
                    
        return False
        
    async def _check_maintenance_suppression(self, rule: SuppressionRule, alert, current_time: datetime) -> bool:
        """检查维护期间抑制"""
        if not rule.matches_alert(alert):
            return False
            
        return rule._is_in_maintenance_window(current_time)
        
    async def _check_rate_limit_suppression(self, rule: SuppressionRule, alert, current_time: datetime) -> bool:
        """检查基于频率的抑制"""
        if not rule.matches_alert(alert):
            return False
            
        if not rule.rate_limit_count or not rule.rate_limit_window:
            return False
            
        # 获取计数器键
        counter_key = f"{rule.id}_{alert.rule_id}"
        
        # 初始化计数器
        if counter_key not in self.rate_limit_counters:
            self.rate_limit_counters[counter_key] = []
            
        # 清理过期记录
        window_start = current_time - timedelta(seconds=rule.rate_limit_window)
        self.rate_limit_counters[counter_key] = [
            ts for ts in self.rate_limit_counters[counter_key]
            if ts > window_start
        ]
        
        # 检查是否超过限制
        if len(self.rate_limit_counters[counter_key]) >= rule.rate_limit_count:
            return True
            
        # 记录当前时间
        self.rate_limit_counters[counter_key].append(current_time)
        return False
        
    async def _check_duplicate_suppression(self, rule: SuppressionRule, alert, current_time: datetime) -> bool:
        """检查重复告警抑制"""
        if not rule.matches_alert(alert):
            return False
            
        if not rule.duplicate_window or not rule.duplicate_threshold:
            return False
            
        # 创建告警指纹
        fingerprint = AlertFingerprint.from_alert(alert)
        
        # 清理过期记录
        window_start = current_time - timedelta(seconds=rule.duplicate_window)
        history = self.alert_history[fingerprint]
        
        # 移除过期记录
        while history and history[0] < window_start:
            history.popleft()
            
        # 检查重复次数
        if len(history) >= rule.duplicate_threshold:
            logger.info(f"检测到重复告警: {alert.rule_name}, 次数: {len(history)}")
            return True
            
        # 记录当前告警
        history.append(current_time)
        return False
        
    async def _check_smart_suppression(self, rule: SuppressionRule, alert, current_time: datetime) -> bool:
        """检查智能抑制"""
        if not rule.matches_alert(alert):
            return False
            
        if not rule.smart_threshold or not rule.smart_window or not rule.smart_algorithm:
            return False
            
        # 根据算法类型执行不同的智能抑制逻辑
        if rule.smart_algorithm == "anomaly_detection":
            return await self._check_anomaly_suppression(rule, alert, current_time)
        elif rule.smart_algorithm == "pattern_learning":
            return await self._check_pattern_learning_suppression(rule, alert, current_time)
        elif rule.smart_algorithm == "correlation_analysis":
            return await self._check_correlation_suppression(rule, alert, current_time)
        else:
            logger.warning(f"未知的智能算法: {rule.smart_algorithm}")
            return False
            
    async def _check_anomaly_suppression(self, rule: SuppressionRule, alert, current_time: datetime) -> bool:
        """异常检测抑制"""
        # 简单的异常检测：基于历史频率判断
        pattern_key = f"{alert.rule_name}_{alert.severity.value}"
        
        # 获取历史模式
        if pattern_key not in self.alert_patterns:
            self.alert_patterns[pattern_key] = []
            
        patterns = self.alert_patterns[pattern_key]
        
        # 清理过期数据
        window_start = current_time - timedelta(seconds=rule.smart_window)
        patterns[:] = [(ts, data) for ts, data in patterns if ts > window_start]
        
        # 计算当前频率
        current_frequency = len(patterns)
        
        # 如果历史数据不足，不抑制
        if len(patterns) < 10:
            patterns.append((current_time, alert))
            return False
            
        # 计算平均频率
        avg_frequency = len(patterns) / (rule.smart_window / 3600)  # 每小时平均频率
        
        # 如果当前频率超过阈值倍数，则抑制
        if current_frequency > avg_frequency * rule.smart_threshold:
            logger.info(f"智能抑制 - 异常频率检测: {pattern_key}, 当前: {current_frequency}, 平均: {avg_frequency}")
            return True
            
        patterns.append((current_time, alert))
        return False
        
    async def _check_pattern_learning_suppression(self, rule: SuppressionRule, alert, current_time: datetime) -> bool:
        """模式学习抑制"""
        # 简单的模式学习：检测周期性模式
        pattern_key = f"{alert.rule_name}_{alert.severity.value}"
        
        if pattern_key not in self.smart_cache:
            self.smart_cache[pattern_key] = {
                "last_occurrences": [],
                "learned_intervals": []
            }
            
        cache = self.smart_cache[pattern_key]
        last_occurrences = cache["last_occurrences"]
        
        # 记录当前发生时间
        last_occurrences.append(current_time)
        
        # 保持最近的记录
        if len(last_occurrences) > 20:
            last_occurrences.pop(0)
            
        # 学习间隔模式
        if len(last_occurrences) >= 5:
            intervals = []
            for i in range(1, len(last_occurrences)):
                interval = (last_occurrences[i] - last_occurrences[i-1]).total_seconds()
                intervals.append(interval)
                
            # 检测是否有规律的间隔
            if len(intervals) >= 3:
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
                
                # 如果方差较小，说明有规律性
                if variance < (avg_interval * 0.1) ** 2:  # 10%的变异系数
                    # 预测下次发生时间
                    predicted_next = last_occurrences[-1] + timedelta(seconds=avg_interval)
                    
                    # 如果在预测时间附近，可能是预期的告警
                    time_diff = abs((current_time - predicted_next).total_seconds())
                    if time_diff < avg_interval * 0.2:  # 20%的误差范围
                        logger.info(f"智能抑制 - 模式学习: {pattern_key}, 预期告警")
                        return True
                        
        return False
        
    async def _check_correlation_suppression(self, rule: SuppressionRule, alert, current_time: datetime) -> bool:
        """关联分析抑制"""
        # 简单的关联分析：检查相关告警的发生情况
        correlation_window = timedelta(minutes=5)  # 5分钟关联窗口
        
        # 检查最近是否有相关告警
        related_patterns = [
            pattern for pattern in self.alert_patterns.keys()
            if pattern != f"{alert.rule_name}_{alert.severity.value}"
        ]
        
        recent_alerts = 0
        for pattern in related_patterns:
            if pattern in self.alert_patterns:
                recent_occurrences = [
                    ts for ts, _ in self.alert_patterns[pattern]
                    if current_time - correlation_window <= ts <= current_time
                ]
                recent_alerts += len(recent_occurrences)
                
        # 如果最近有大量相关告警，可能是系统性问题，抑制部分告警
        if recent_alerts > rule.smart_threshold:
            logger.info(f"智能抑制 - 关联分析: {alert.rule_name}, 相关告警数: {recent_alerts}")
            return True
            
        return False
        
    async def _record_suppression_event(self, rule_id: str, alert_id: str, event_type: str, reason: str):
        """记录抑制事件"""
        event = SuppressionEvent(
            id=str(uuid4()),
            rule_id=rule_id,
            alert_id=alert_id,
            event_type=event_type,
            reason=reason
        )
        
        self.suppression_events.append(event)
        logger.info(f"记录抑制事件: {event_type} - {alert_id} by {rule_id}")
        
    def _update_dependency_graph(self, rule: SuppressionRule):
        """更新依赖图"""
        if rule.id not in self.dependency_graph:
            self.dependency_graph[rule.id] = set()
            
        self.dependency_graph[rule.id].update(rule.dependency_rules)
        
    def _cleanup_dependency_graph(self, rule_id: str):
        """清理依赖图"""
        # 删除规则的依赖
        if rule_id in self.dependency_graph:
            del self.dependency_graph[rule_id]
            
        # 删除其他规则对此规则的依赖
        for deps in self.dependency_graph.values():
            deps.discard(rule_id)
            
    async def get_active_rules(self) -> List[SuppressionRule]:
        """获取激活的抑制规则"""
        current_time = datetime.now()
        return [
            rule for rule in self.rules.values()
            if rule.is_active(current_time)
        ]
        
    async def get_suppressed_alerts(self) -> Set[str]:
        """获取被抑制的告警ID集合"""
        return self.suppressed_alerts.copy()
        
    async def get_suppression_statistics(self) -> Dict[str, Any]:
        """获取抑制统计信息"""
        now = datetime.now()
        last_24h = now - timedelta(hours=24)
        
        recent_events = [
            event for event in self.suppression_events
            if event.timestamp >= last_24h
        ]
        
        # 按类型统计规则
        type_counts = {}
        for rule_type in SuppressionType:
            type_counts[rule_type.value] = len([
                rule for rule in self.rules.values()
                if rule.type == rule_type
            ])
            
        # 按状态统计规则
        status_counts = {}
        for status in SuppressionStatus:
            status_counts[status.value] = len([
                rule for rule in self.rules.values()
                if rule.status == status
            ])
            
        return {
            "total_rules": len(self.rules),
            "active_rules": len(await self.get_active_rules()),
            "suppressed_alerts": len(self.suppressed_alerts),
            "type_distribution": type_counts,
            "status_distribution": status_counts,
            "events_last_24h": len(recent_events),
            "suppression_events": len([e for e in recent_events if e.event_type == "suppressed"]),
            "unsuppression_events": len([e for e in recent_events if e.event_type == "unsuppressed"]),
            "duplicate_patterns": len(self.alert_history),
            "smart_patterns": len(self.alert_patterns)
        }
        
    async def create_maintenance_window(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        alert_patterns: List[str],
        recurring: bool = False
    ) -> str:
        """创建维护窗口"""
        rule_id = str(uuid4())
        
        maintenance_windows = [{
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "recurring": "weekly" if recurring else None
        }]
        
        rule = SuppressionRule(
            id=rule_id,
            name=f"维护窗口: {name}",
            description=f"维护期间抑制告警: {start_time} - {end_time}",
            type=SuppressionType.MAINTENANCE,
            alert_name_pattern="|".join(alert_patterns) if alert_patterns else ".*",
            maintenance_windows=maintenance_windows,
            start_time=start_time,
            end_time=end_time
        )
        
        await self.add_rule(rule)
        logger.info(f"创建维护窗口: {name} ({rule_id})")
        return rule_id
        
    async def create_duplicate_suppression_rule(
        self,
        name: str,
        alert_pattern: str,
        window_seconds: int = 300,
        threshold: int = 3
    ) -> str:
        """创建重复告警抑制规则"""
        rule_id = str(uuid4())
        
        rule = SuppressionRule(
            id=rule_id,
            name=f"重复抑制: {name}",
            description=f"抑制重复告警: {alert_pattern}",
            type=SuppressionType.DUPLICATE,
            alert_name_pattern=alert_pattern,
            duplicate_window=window_seconds,
            duplicate_threshold=threshold
        )
        
        await self.add_rule(rule)
        logger.info(f"创建重复抑制规则: {name} ({rule_id})")
        return rule_id
        
    async def create_smart_suppression_rule(
        self,
        name: str,
        alert_pattern: str,
        algorithm: str = "anomaly_detection",
        threshold: float = 2.0,
        window_seconds: int = 3600
    ) -> str:
        """创建智能抑制规则"""
        rule_id = str(uuid4())
        
        rule = SuppressionRule(
            id=rule_id,
            name=f"智能抑制: {name}",
            description=f"智能抑制告警: {alert_pattern}",
            type=SuppressionType.SMART,
            alert_name_pattern=alert_pattern,
            smart_algorithm=algorithm,
            smart_threshold=threshold,
            smart_window=window_seconds
        )
        
        await self.add_rule(rule)
        logger.info(f"创建智能抑制规则: {name} ({rule_id})")
        return rule_id
        
    async def cleanup_expired_data(self):
        """清理过期数据"""
        now = datetime.now()
        
        # 清理过期的抑制事件（保留30天）
        cutoff_time = now - timedelta(days=30)
        self.suppression_events = [
            event for event in self.suppression_events
            if event.timestamp > cutoff_time
        ]
        
        # 清理过期的频率限制计数器
        for key, timestamps in list(self.rate_limit_counters.items()):
            # 保留最近1小时的数据
            recent_timestamps = [
                ts for ts in timestamps
                if ts > now - timedelta(hours=1)
            ]
            if recent_timestamps:
                self.rate_limit_counters[key] = recent_timestamps
            else:
                del self.rate_limit_counters[key]
                
        # 清理过期的重复检测缓存
        for fingerprint, history in list(self.alert_history.items()):
            # 清理超过24小时的记录
            cutoff = now - timedelta(hours=24)
            while history and history[0] < cutoff:
                history.popleft()
            if not history:
                del self.alert_history[fingerprint]
                
        # 清理过期的智能模式数据
        for pattern_key, patterns in list(self.alert_patterns.items()):
            # 保留最近7天的数据
            cutoff = now - timedelta(days=7)
            patterns[:] = [(ts, data) for ts, data in patterns if ts > cutoff]
            if not patterns:
                del self.alert_patterns[pattern_key]
                
        logger.info("清理过期抑制数据完成")
        
    async def _load_default_rules(self):
        """加载默认抑制规则"""
        default_rules = [
            # 维护期间抑制所有告警
            SuppressionRule(
                id="maintenance_suppression",
                name="维护期间抑制",
                description="系统维护期间抑制所有告警",
                type=SuppressionType.MAINTENANCE,
                status=SuppressionStatus.INACTIVE,  # 默认不激活
                alert_name_pattern=".*",
                maintenance_windows=[]
            ),
            
            # 低优先级告警频率限制
            SuppressionRule(
                id="low_priority_rate_limit",
                name="低优先级告警频率限制",
                description="限制低优先级告警的频率",
                type=SuppressionType.RATE_LIMIT,
                severity_levels=["low", "info"],
                rate_limit_count=5,
                rate_limit_window=300  # 5分钟内最多5个
            ),
            
            # 数据库连接失败时抑制相关告警
            SuppressionRule(
                id="db_failure_dependency",
                name="数据库故障依赖抑制",
                description="数据库连接失败时抑制相关告警",
                type=SuppressionType.DEPENDENCY,
                dependency_rules=["db_connection_failure"],
                label_matchers={"component": "database|api|statistics"}
            ),
            
            # 重复告警抑制
            SuppressionRule(
                id="duplicate_alert_suppression",
                name="重复告警抑制",
                description="抑制5分钟内重复出现3次以上的告警",
                type=SuppressionType.DUPLICATE,
                alert_name_pattern=".*",
                duplicate_window=300,  # 5分钟
                duplicate_threshold=3
            ),
            
            # 智能异常检测抑制
            SuppressionRule(
                id="smart_anomaly_suppression",
                name="智能异常检测抑制",
                description="基于异常检测的智能抑制",
                type=SuppressionType.SMART,
                alert_name_pattern=".*",
                smart_algorithm="anomaly_detection",
                smart_threshold=3.0,  # 3倍标准差
                smart_window=3600  # 1小时窗口
            ),
            
            # 夜间低优先级抑制
            SuppressionRule(
                id="night_low_priority_suppression",
                name="夜间低优先级抑制",
                description="夜间时段抑制低优先级告警",
                type=SuppressionType.TIME_BASED,
                severity_levels=["low", "info"],
                maintenance_windows=[{
                    "start_time": "22:00:00",
                    "end_time": "08:00:00",
                    "recurring": "daily"
                }]
            )
        ]
        
        for rule in default_rules:
            await self.add_rule(rule)
            
        logger.info(f"加载了 {len(default_rules)} 个默认抑制规则")