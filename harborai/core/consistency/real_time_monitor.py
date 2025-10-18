#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据一致性实时监控器

提供实时数据一致性监控功能，包括：
- 实时数据变更监控
- 一致性问题检测
- 自动告警触发
- 性能指标收集
- 监控状态管理
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict, deque

from ...database.async_manager import DatabaseManager
from .data_consistency_checker import DataConsistencyChecker, ConsistencyIssue, IssueType, IssueSeverity
from .auto_correction_service import AutoCorrectionService
from ..alerts.alert_manager import AlertManager, AlertRule, AlertSeverity, AlertCondition


class MonitoringMode(Enum):
    """监控模式"""
    CONTINUOUS = "continuous"    # 连续监控
    SCHEDULED = "scheduled"      # 定时监控
    TRIGGERED = "triggered"      # 触发式监控
    ADAPTIVE = "adaptive"        # 自适应监控


class MonitoringStatus(Enum):
    """监控状态"""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class MonitoringMetrics:
    """监控指标"""
    checks_performed: int = 0
    issues_detected: int = 0
    issues_resolved: int = 0
    false_positives: int = 0
    avg_check_duration: float = 0.0
    last_check_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "checks_performed": self.checks_performed,
            "issues_detected": self.issues_detected,
            "issues_resolved": self.issues_resolved,
            "false_positives": self.false_positives,
            "avg_check_duration": self.avg_check_duration,
            "last_check_time": self.last_check_time.isoformat() if self.last_check_time else None,
            "uptime_seconds": self.uptime_seconds,
            "error_count": self.error_count
        }


@dataclass
class MonitoringConfig:
    """监控配置"""
    mode: MonitoringMode = MonitoringMode.CONTINUOUS
    check_interval: int = 60  # 检查间隔（秒）
    batch_size: int = 1000    # 批处理大小
    max_concurrent_checks: int = 5  # 最大并发检查数
    auto_correction_enabled: bool = True  # 启用自动修正
    alert_enabled: bool = True  # 启用告警
    
    # 检查配置
    token_consistency_enabled: bool = True
    cost_consistency_enabled: bool = True
    tracing_completeness_enabled: bool = True
    foreign_key_integrity_enabled: bool = True
    data_range_validation_enabled: bool = True
    performance_anomaly_enabled: bool = True
    
    # 告警阈值
    critical_issue_threshold: int = 10  # 严重问题阈值
    high_issue_threshold: int = 50      # 高级问题阈值
    error_rate_threshold: float = 0.1   # 错误率阈值
    
    # 自适应配置
    adaptive_interval_min: int = 30     # 最小检查间隔
    adaptive_interval_max: int = 300    # 最大检查间隔
    load_factor_threshold: float = 0.8  # 负载因子阈值


class RealTimeConsistencyMonitor:
    """
    数据一致性实时监控器
    
    提供实时数据一致性监控功能，包括：
    - 连续或定时的一致性检查
    - 自动问题检测和告警
    - 性能指标收集和分析
    - 自适应监控策略
    - 监控状态管理
    """
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        consistency_checker: DataConsistencyChecker,
        auto_correction_service: AutoCorrectionService,
        alert_manager: AlertManager,
        config: Optional[MonitoringConfig] = None
    ):
        """
        初始化实时监控器
        
        Args:
            db_manager: 数据库管理器
            consistency_checker: 数据一致性检查器
            auto_correction_service: 自动修正服务
            alert_manager: 告警管理器
            config: 监控配置
        """
        self.db_manager = db_manager
        self.consistency_checker = consistency_checker
        self.auto_correction_service = auto_correction_service
        self.alert_manager = alert_manager
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        
        # 监控状态
        self.status = MonitoringStatus.STOPPED
        self.start_time: Optional[datetime] = None
        self.monitoring_task: Optional[asyncio.Task] = None
        
        # 指标和统计
        self.metrics = MonitoringMetrics()
        self.recent_issues: deque = deque(maxlen=1000)  # 最近的问题
        self.check_durations: deque = deque(maxlen=100)  # 最近的检查耗时
        
        # 回调函数
        self.issue_callbacks: List[Callable[[ConsistencyIssue], None]] = []
        self.status_callbacks: List[Callable[[MonitoringStatus], None]] = []
        
        # 内部状态
        self._stop_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # 初始状态为非暂停
        
    async def initialize(self):
        """初始化监控器"""
        self.logger.info("初始化数据一致性实时监控器")
        
        # 注册默认告警规则
        await self._register_default_alert_rules()
        
        # 初始化组件
        if hasattr(self.consistency_checker, 'initialize'):
            await self.consistency_checker.initialize()
            
        self.logger.info("数据一致性实时监控器初始化完成")
        
    async def start(self):
        """启动监控"""
        if self.status in [MonitoringStatus.RUNNING, MonitoringStatus.STARTING]:
            self.logger.warning("监控器已在运行或正在启动")
            return
            
        self.logger.info("启动数据一致性实时监控")
        self.status = MonitoringStatus.STARTING
        self._notify_status_change()
        
        try:
            self.start_time = datetime.now()
            self._stop_event.clear()
            self._pause_event.set()
            
            # 启动监控任务
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())
            
            self.status = MonitoringStatus.RUNNING
            self._notify_status_change()
            self.logger.info("数据一致性实时监控已启动")
            
        except Exception as e:
            self.status = MonitoringStatus.ERROR
            self._notify_status_change()
            self.logger.error(f"启动监控失败: {e}")
            raise
            
    async def stop(self):
        """停止监控"""
        if self.status == MonitoringStatus.STOPPED:
            return
            
        self.logger.info("停止数据一致性实时监控")
        self.status = MonitoringStatus.STOPPING
        self._notify_status_change()
        
        # 设置停止事件
        self._stop_event.set()
        
        # 等待监控任务完成
        if self.monitoring_task and not self.monitoring_task.done():
            try:
                await asyncio.wait_for(self.monitoring_task, timeout=10.0)
            except asyncio.TimeoutError:
                self.logger.warning("监控任务停止超时，强制取消")
                self.monitoring_task.cancel()
                
        self.status = MonitoringStatus.STOPPED
        self._notify_status_change()
        self.logger.info("数据一致性实时监控已停止")
        
    async def pause(self):
        """暂停监控"""
        if self.status != MonitoringStatus.RUNNING:
            return
            
        self.logger.info("暂停数据一致性监控")
        self._pause_event.clear()
        self.status = MonitoringStatus.PAUSED
        self._notify_status_change()
        
    async def resume(self):
        """恢复监控"""
        if self.status != MonitoringStatus.PAUSED:
            return
            
        self.logger.info("恢复数据一致性监控")
        self._pause_event.set()
        self.status = MonitoringStatus.RUNNING
        self._notify_status_change()
        
    def add_issue_callback(self, callback: Callable[[ConsistencyIssue], None]):
        """添加问题回调函数"""
        self.issue_callbacks.append(callback)
        
    def add_status_callback(self, callback: Callable[[MonitoringStatus], None]):
        """添加状态回调函数"""
        self.status_callbacks.append(callback)
        
    def get_metrics(self) -> MonitoringMetrics:
        """获取监控指标"""
        # 更新运行时间
        if self.start_time:
            self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
        # 更新平均检查耗时
        if self.check_durations:
            self.metrics.avg_check_duration = sum(self.check_durations) / len(self.check_durations)
            
        return self.metrics
        
    def get_recent_issues(self, limit: int = 100) -> List[ConsistencyIssue]:
        """获取最近的问题"""
        return list(self.recent_issues)[-limit:]
        
    async def force_check(self) -> Dict[str, Any]:
        """强制执行一次检查"""
        self.logger.info("执行强制一致性检查")
        
        start_time = time.time()
        try:
            # 执行检查
            issues = await self._perform_consistency_check()
            
            # 处理问题
            if issues:
                await self._handle_issues(issues)
                
            duration = time.time() - start_time
            self.check_durations.append(duration)
            
            return {
                "success": True,
                "issues_found": len(issues),
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.metrics.error_count += 1
            self.logger.error(f"强制检查失败: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
            
    async def _monitoring_loop(self):
        """监控主循环"""
        self.logger.info("启动监控主循环")
        
        while not self._stop_event.is_set():
            try:
                # 等待暂停状态解除
                await self._pause_event.wait()
                
                if self._stop_event.is_set():
                    break
                    
                # 执行一致性检查
                start_time = time.time()
                issues = await self._perform_consistency_check()
                duration = time.time() - start_time
                
                # 更新指标
                self.metrics.checks_performed += 1
                self.metrics.last_check_time = datetime.now()
                self.check_durations.append(duration)
                
                # 处理发现的问题
                if issues:
                    await self._handle_issues(issues)
                    
                # 自适应调整检查间隔
                if self.config.mode == MonitoringMode.ADAPTIVE:
                    self._adjust_check_interval(issues, duration)
                    
                # 等待下次检查
                await asyncio.sleep(self.config.check_interval)
                
            except asyncio.CancelledError:
                self.logger.info("监控循环被取消")
                break
            except Exception as e:
                self.metrics.error_count += 1
                self.logger.error(f"监控循环出错: {e}")
                
                # 错误恢复：等待一段时间后继续
                await asyncio.sleep(min(self.config.check_interval, 60))
                
        self.logger.info("监控主循环结束")
        
    async def _perform_consistency_check(self) -> List[ConsistencyIssue]:
        """执行一致性检查"""
        all_issues = []
        
        try:
            # 并发执行各种检查
            check_tasks = []
            
            if self.config.token_consistency_enabled:
                check_tasks.append(self.consistency_checker.check_token_consistency())
                
            if self.config.cost_consistency_enabled:
                check_tasks.append(self.consistency_checker.check_cost_consistency())
                
            if self.config.tracing_completeness_enabled:
                check_tasks.append(self.consistency_checker.check_tracing_completeness())
                
            if self.config.foreign_key_integrity_enabled:
                check_tasks.append(self.consistency_checker.check_foreign_key_integrity())
                
            if self.config.data_range_validation_enabled:
                check_tasks.append(self.consistency_checker.check_data_range_consistency())
                
            if self.config.performance_anomaly_enabled:
                check_tasks.append(self.consistency_checker.check_performance_anomalies())
                
            # 限制并发数
            semaphore = asyncio.Semaphore(self.config.max_concurrent_checks)
            
            async def limited_check(check_coro):
                async with semaphore:
                    return await check_coro
                    
            # 执行检查
            results = await asyncio.gather(
                *[limited_check(task) for task in check_tasks],
                return_exceptions=True
            )
            
            # 处理结果
            successful_checks = 0
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"检查任务失败: {result}")
                    continue
                    
                successful_checks += 1
                if isinstance(result, tuple) and len(result) == 2:
                    issues, _ = result
                    all_issues.extend(issues)
                elif isinstance(result, list):
                    all_issues.extend(result)
            
            # 如果所有检查都失败，抛出异常
            if successful_checks == 0 and len(check_tasks) > 0:
                raise Exception("所有一致性检查都失败")
                    
        except Exception as e:
            self.logger.error(f"执行一致性检查失败: {e}")
            raise
            
        return all_issues
        
    async def _handle_issues(self, issues: List[ConsistencyIssue]):
        """处理发现的问题"""
        if not issues:
            return
            
        self.logger.info(f"发现 {len(issues)} 个一致性问题")
        self.metrics.issues_detected += len(issues)
        
        # 记录问题
        for issue in issues:
            self.recent_issues.append(issue)
            
        # 触发回调
        for issue in issues:
            for callback in self.issue_callbacks:
                try:
                    callback(issue)
                except Exception as e:
                    self.logger.error(f"问题回调执行失败: {e}")
                    
        # 触发告警
        if self.config.alert_enabled:
            await self._trigger_alerts(issues)
            
        # 自动修正
        if self.config.auto_correction_enabled:
            await self._auto_correct_issues(issues)
            
    async def _trigger_alerts(self, issues: List[ConsistencyIssue]):
        """触发告警"""
        try:
            # 按严重程度分组
            critical_issues = [i for i in issues if i.severity == IssueSeverity.CRITICAL]
            high_issues = [i for i in issues if i.severity == IssueSeverity.HIGH]
            
            # 检查是否超过阈值
            if len(critical_issues) >= self.config.critical_issue_threshold:
                await self._create_alert(
                    "critical_consistency_issues",
                    AlertSeverity.CRITICAL,
                    f"发现 {len(critical_issues)} 个严重一致性问题",
                    {"issues": [i.to_dict() for i in critical_issues]}
                )
                
            if len(high_issues) >= self.config.high_issue_threshold:
                await self._create_alert(
                    "high_consistency_issues",
                    AlertSeverity.HIGH,
                    f"发现 {len(high_issues)} 个高级一致性问题",
                    {"issues": [i.to_dict() for i in high_issues]}
                )
                
        except Exception as e:
            self.logger.error(f"触发告警失败: {e}")
            
    async def _auto_correct_issues(self, issues: List[ConsistencyIssue]):
        """自动修正问题"""
        try:
            # 过滤可自动修正的问题
            correctable_issues = [
                issue for issue in issues
                if issue.auto_fixable and issue.issue_type in [
                    IssueType.TOKEN_MISMATCH,
                    IssueType.COST_MISMATCH,
                    IssueType.MISSING_TRACING,
                    IssueType.ORPHANED_RECORD
                ]
            ]
            
            if not correctable_issues:
                return
                
            self.logger.info(f"尝试自动修正 {len(correctable_issues)} 个问题")
            
            # 执行自动修正
            result = await self.auto_correction_service.auto_correct_issues(
                correctable_issues, dry_run=False
            )
            
            if result.success:
                self.metrics.issues_resolved += result.total_records_affected
                self.logger.info(f"成功自动修正 {result.total_records_affected} 个问题")
            else:
                self.logger.warning(f"自动修正失败: {result.errors}")
                
        except Exception as e:
            self.logger.error(f"自动修正失败: {e}")
            
    def _adjust_check_interval(self, issues: List[ConsistencyIssue], duration: float):
        """自适应调整检查间隔"""
        try:
            # 基于问题数量和检查耗时调整间隔
            issue_factor = min(len(issues) / 10.0, 1.0)  # 问题越多，间隔越短
            duration_factor = min(duration / 60.0, 1.0)  # 耗时越长，间隔越长
            
            # 计算新的间隔
            base_interval = self.config.check_interval
            adjustment = (issue_factor - duration_factor) * 0.5
            new_interval = base_interval * (1 - adjustment)
            
            # 限制在配置范围内
            new_interval = max(
                self.config.adaptive_interval_min,
                min(new_interval, self.config.adaptive_interval_max)
            )
            
            if abs(new_interval - self.config.check_interval) > 5:
                self.logger.info(f"调整检查间隔: {self.config.check_interval}s -> {new_interval}s")
                self.config.check_interval = int(new_interval)
                
        except Exception as e:
            self.logger.error(f"调整检查间隔失败: {e}")
            
    async def _create_alert(self, rule_name: str, severity: AlertSeverity, message: str, metadata: Dict[str, Any]):
        """创建告警"""
        try:
            # 这里应该调用告警管理器的方法
            # 由于AlertManager的接口可能不同，这里提供一个通用的实现
            self.logger.warning(f"[{severity.value.upper()}] {rule_name}: {message}")
            
            # 如果告警管理器支持，可以调用其方法
            # await self.alert_manager.create_alert(rule_name, severity, message, metadata)
            
        except Exception as e:
            self.logger.error(f"创建告警失败: {e}")
            
    async def _register_default_alert_rules(self):
        """注册默认告警规则"""
        try:
            # 这里可以注册一些默认的告警规则
            # 具体实现取决于AlertManager的接口
            pass
        except Exception as e:
            self.logger.error(f"注册默认告警规则失败: {e}")
            
    def _notify_status_change(self):
        """通知状态变更"""
        for callback in self.status_callbacks:
            try:
                callback(self.status)
            except Exception as e:
                self.logger.error(f"状态回调执行失败: {e}")