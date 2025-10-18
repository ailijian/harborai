#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
告警历史服务

负责告警历史记录的存储、查询和分析
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from uuid import uuid4
import sqlite3
import aiosqlite

logger = logging.getLogger(__name__)


class AlertEventType(Enum):
    """告警事件类型"""
    CREATED = "created"           # 告警创建
    UPDATED = "updated"           # 告警更新
    RESOLVED = "resolved"         # 告警解决
    ACKNOWLEDGED = "acknowledged" # 告警确认
    SUPPRESSED = "suppressed"     # 告警抑制
    ESCALATED = "escalated"       # 告警升级
    NOTIFICATION_SENT = "notification_sent"  # 通知发送
    NOTIFICATION_FAILED = "notification_failed"  # 通知失败


@dataclass
class AlertEvent:
    """告警事件"""
    id: str
    alert_id: str
    event_type: AlertEventType
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "event_type": self.event_type.value,
            "message": self.message,
            "metadata": self.metadata,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AlertHistoryRecord:
    """告警历史记录"""
    alert_id: str
    rule_id: str
    rule_name: str
    severity: str
    status: str
    message: str
    metric_value: float
    threshold: float
    labels: Dict[str, str]
    annotations: Dict[str, str]
    started_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolution_time: Optional[int] = None  # 解决时间（秒）
    notification_count: int = 0
    escalation_level: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity,
            "status": self.status,
            "message": self.message,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "labels": self.labels,
            "annotations": self.annotations,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolution_time": self.resolution_time,
            "notification_count": self.notification_count,
            "escalation_level": self.escalation_level
        }


class AlertHistory:
    """告警历史服务"""
    
    def __init__(self, db_path: str = "alert_history.db"):
        self.db_path = db_path
        self.events: List[AlertEvent] = []
        self.records: List[AlertHistoryRecord] = []
        
    async def initialize(self):
        """初始化告警历史服务"""
        logger.info("初始化告警历史服务")
        await self._init_database()
        
    async def _init_database(self):
        """初始化数据库"""
        async with aiosqlite.connect(self.db_path) as db:
            # 创建告警历史表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    alert_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    labels TEXT NOT NULL,
                    annotations TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    resolved_at TEXT,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    resolution_time INTEGER,
                    notification_count INTEGER DEFAULT 0,
                    escalation_level INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 创建告警事件表
            await db.execute("""
                CREATE TABLE IF NOT EXISTS alert_events (
                    id TEXT PRIMARY KEY,
                    alert_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metadata TEXT,
                    user_id TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (alert_id) REFERENCES alert_history (alert_id)
                )
            """)
            
            # 创建索引
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alert_history_rule_id ON alert_history(rule_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alert_history_severity ON alert_history(severity)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alert_history_started_at ON alert_history(started_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alert_events_alert_id ON alert_events(alert_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alert_events_event_type ON alert_events(event_type)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_alert_events_timestamp ON alert_events(timestamp)")
            
            await db.commit()
            
        logger.info("告警历史数据库初始化完成")
        
    async def record_alert(self, alert) -> bool:
        """记录告警"""
        try:
            record = AlertHistoryRecord(
                alert_id=alert.id,
                rule_id=alert.rule_id,
                rule_name=alert.rule_name,
                severity=alert.severity.value,
                status=alert.status.value,
                message=alert.message,
                metric_value=alert.metric_value,
                threshold=alert.threshold,
                labels=alert.labels,
                annotations=alert.annotations,
                started_at=alert.started_at,
                resolved_at=alert.resolved_at,
                acknowledged_at=alert.acknowledged_at,
                acknowledged_by=alert.acknowledged_by,
                escalation_level=alert.escalation_level
            )
            
            # 计算解决时间
            if alert.resolved_at and alert.started_at:
                record.resolution_time = int((alert.resolved_at - alert.started_at).total_seconds())
                
            await self._save_alert_record(record)
            
            # 记录创建事件
            await self.record_event(
                alert.id,
                AlertEventType.CREATED,
                f"告警创建: {alert.message}",
                {"metric_value": alert.metric_value, "threshold": alert.threshold}
            )
            
            logger.info(f"记录告警历史: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"记录告警历史失败: {e}")
            return False
            
    async def update_alert(self, alert) -> bool:
        """更新告警记录"""
        try:
            # 计算解决时间
            resolution_time = None
            if alert.resolved_at and alert.started_at:
                resolution_time = int((alert.resolved_at - alert.started_at).total_seconds())
                
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    UPDATE alert_history SET
                        status = ?,
                        resolved_at = ?,
                        acknowledged_at = ?,
                        acknowledged_by = ?,
                        resolution_time = ?,
                        escalation_level = ?
                    WHERE alert_id = ?
                """, (
                    alert.status.value,
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
                    alert.acknowledged_by,
                    resolution_time,
                    alert.escalation_level,
                    alert.id
                ))
                await db.commit()
                
            logger.info(f"更新告警历史: {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"更新告警历史失败: {e}")
            return False
            
    async def record_event(
        self,
        alert_id: str,
        event_type: AlertEventType,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """记录告警事件"""
        try:
            event = AlertEvent(
                id=str(uuid4()),
                alert_id=alert_id,
                event_type=event_type,
                message=message,
                metadata=metadata or {},
                user_id=user_id
            )
            
            await self._save_event(event)
            
            logger.debug(f"记录告警事件: {alert_id} - {event_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"记录告警事件失败: {e}")
            return False
            
    async def _save_alert_record(self, record: AlertHistoryRecord):
        """保存告警记录到数据库"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO alert_history (
                    alert_id, rule_id, rule_name, severity, status, message,
                    metric_value, threshold, labels, annotations, started_at,
                    resolved_at, acknowledged_at, acknowledged_by, resolution_time,
                    notification_count, escalation_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.alert_id,
                record.rule_id,
                record.rule_name,
                record.severity,
                record.status,
                record.message,
                record.metric_value,
                record.threshold,
                json.dumps(record.labels, ensure_ascii=False),
                json.dumps(record.annotations, ensure_ascii=False),
                record.started_at.isoformat(),
                record.resolved_at.isoformat() if record.resolved_at else None,
                record.acknowledged_at.isoformat() if record.acknowledged_at else None,
                record.acknowledged_by,
                record.resolution_time,
                record.notification_count,
                record.escalation_level
            ))
            await db.commit()
            
    async def _save_event(self, event: AlertEvent):
        """保存事件到数据库"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO alert_events (
                    id, alert_id, event_type, message, metadata, user_id, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.alert_id,
                event.event_type.value,
                event.message,
                json.dumps(event.metadata, ensure_ascii=False),
                event.user_id,
                event.timestamp.isoformat()
            ))
            await db.commit()
            
    async def get_alert_history(
        self,
        alert_id: Optional[str] = None,
        rule_id: Optional[str] = None,
        severity: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AlertHistoryRecord]:
        """查询告警历史"""
        try:
            conditions = []
            params = []
            
            if alert_id:
                conditions.append("alert_id = ?")
                params.append(alert_id)
                
            if rule_id:
                conditions.append("rule_id = ?")
                params.append(rule_id)
                
            if severity:
                conditions.append("severity = ?")
                params.append(severity)
                
            if start_time:
                conditions.append("started_at >= ?")
                params.append(start_time.isoformat())
                
            if end_time:
                conditions.append("started_at <= ?")
                params.append(end_time.isoformat())
                
            where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
                SELECT * FROM alert_history
                {where_clause}
                ORDER BY started_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
            
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
            records = []
            for row in rows:
                record = AlertHistoryRecord(
                    alert_id=row[0],
                    rule_id=row[1],
                    rule_name=row[2],
                    severity=row[3],
                    status=row[4],
                    message=row[5],
                    metric_value=row[6],
                    threshold=row[7],
                    labels=json.loads(row[8]),
                    annotations=json.loads(row[9]),
                    started_at=datetime.fromisoformat(row[10]),
                    resolved_at=datetime.fromisoformat(row[11]) if row[11] else None,
                    acknowledged_at=datetime.fromisoformat(row[12]) if row[12] else None,
                    acknowledged_by=row[13],
                    resolution_time=row[14],
                    notification_count=row[15],
                    escalation_level=row[16]
                )
                records.append(record)
                
            return records
            
        except Exception as e:
            logger.error(f"查询告警历史失败: {e}")
            return []
            
    async def get_alert_events(
        self,
        alert_id: str,
        event_type: Optional[AlertEventType] = None,
        limit: int = 100
    ) -> List[AlertEvent]:
        """获取告警事件"""
        try:
            conditions = ["alert_id = ?"]
            params = [alert_id]
            
            if event_type:
                conditions.append("event_type = ?")
                params.append(event_type.value)
                
            where_clause = " WHERE " + " AND ".join(conditions)
            
            query = f"""
                SELECT * FROM alert_events
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            """
            params.append(limit)
            
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
            events = []
            for row in rows:
                event = AlertEvent(
                    id=row[0],
                    alert_id=row[1],
                    event_type=AlertEventType(row[2]),
                    message=row[3],
                    metadata=json.loads(row[4]) if row[4] else {},
                    user_id=row[5],
                    timestamp=datetime.fromisoformat(row[6])
                )
                events.append(event)
                
            return events
            
        except Exception as e:
            logger.error(f"获取告警事件失败: {e}")
            return []
            
    async def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """获取告警统计信息"""
        try:
            if not start_time:
                start_time = datetime.now() - timedelta(days=30)
            if not end_time:
                end_time = datetime.now()
                
            async with aiosqlite.connect(self.db_path) as db:
                # 总体统计
                async with db.execute("""
                    SELECT 
                        COUNT(*) as total_alerts,
                        COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_alerts,
                        AVG(CASE WHEN resolution_time IS NOT NULL THEN resolution_time END) as avg_resolution_time,
                        MAX(resolution_time) as max_resolution_time,
                        MIN(resolution_time) as min_resolution_time
                    FROM alert_history
                    WHERE started_at BETWEEN ? AND ?
                """, (start_time.isoformat(), end_time.isoformat())) as cursor:
                    overall_stats = await cursor.fetchone()
                    
                # 按严重程度统计
                async with db.execute("""
                    SELECT severity, COUNT(*) as count
                    FROM alert_history
                    WHERE started_at BETWEEN ? AND ?
                    GROUP BY severity
                """, (start_time.isoformat(), end_time.isoformat())) as cursor:
                    severity_stats = await cursor.fetchall()
                    
                # 按规则统计
                async with db.execute("""
                    SELECT rule_name, COUNT(*) as count
                    FROM alert_history
                    WHERE started_at BETWEEN ? AND ?
                    GROUP BY rule_name
                    ORDER BY count DESC
                    LIMIT 10
                """, (start_time.isoformat(), end_time.isoformat())) as cursor:
                    rule_stats = await cursor.fetchall()
                    
                # 按小时统计（最近24小时）
                last_24h = end_time - timedelta(hours=24)
                async with db.execute("""
                    SELECT 
                        strftime('%H', started_at) as hour,
                        COUNT(*) as count
                    FROM alert_history
                    WHERE started_at BETWEEN ? AND ?
                    GROUP BY hour
                    ORDER BY hour
                """, (last_24h.isoformat(), end_time.isoformat())) as cursor:
                    hourly_stats = await cursor.fetchall()
                    
            return {
                "period": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat()
                },
                "overall": {
                    "total_alerts": overall_stats[0] or 0,
                    "resolved_alerts": overall_stats[1] or 0,
                    "resolution_rate": (
                        (overall_stats[1] / overall_stats[0] * 100)
                        if overall_stats[0] > 0 else 0
                    ),
                    "avg_resolution_time": overall_stats[2] or 0,
                    "max_resolution_time": overall_stats[3] or 0,
                    "min_resolution_time": overall_stats[4] or 0
                },
                "by_severity": {
                    row[0]: row[1] for row in severity_stats
                },
                "top_rules": [
                    {"rule_name": row[0], "count": row[1]}
                    for row in rule_stats
                ],
                "hourly_distribution": {
                    f"{row[0]:02d}:00": row[1] for row in hourly_stats
                }
            }
            
        except Exception as e:
            logger.error(f"获取告警统计失败: {e}")
            return {}
            
    async def get_alert_trends(self, days: int = 7) -> Dict[str, Any]:
        """获取告警趋势"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            async with aiosqlite.connect(self.db_path) as db:
                # 按天统计
                async with db.execute("""
                    SELECT 
                        DATE(started_at) as date,
                        COUNT(*) as total,
                        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical,
                        COUNT(CASE WHEN severity = 'high' THEN 1 END) as high,
                        COUNT(CASE WHEN severity = 'medium' THEN 1 END) as medium,
                        COUNT(CASE WHEN severity = 'low' THEN 1 END) as low,
                        AVG(CASE WHEN resolution_time IS NOT NULL THEN resolution_time END) as avg_resolution_time
                    FROM alert_history
                    WHERE started_at BETWEEN ? AND ?
                    GROUP BY DATE(started_at)
                    ORDER BY date
                """, (start_time.isoformat(), end_time.isoformat())) as cursor:
                    daily_trends = await cursor.fetchall()
                    
            trends = []
            for row in daily_trends:
                trends.append({
                    "date": row[0],
                    "total": row[1],
                    "by_severity": {
                        "critical": row[2],
                        "high": row[3],
                        "medium": row[4],
                        "low": row[5]
                    },
                    "avg_resolution_time": row[6] or 0
                })
                
            return {
                "period_days": days,
                "trends": trends
            }
            
        except Exception as e:
            logger.error(f"获取告警趋势失败: {e}")
            return {}
            
    async def cleanup_old_records(self, retention_days: int = 90) -> int:
        """清理旧记录"""
        try:
            cutoff_time = datetime.now() - timedelta(days=retention_days)
            
            async with aiosqlite.connect(self.db_path) as db:
                # 删除旧的告警事件
                async with db.execute("""
                    DELETE FROM alert_events
                    WHERE timestamp < ?
                """, (cutoff_time.isoformat(),)) as cursor:
                    events_deleted = cursor.rowcount
                    
                # 删除旧的告警历史
                async with db.execute("""
                    DELETE FROM alert_history
                    WHERE started_at < ?
                """, (cutoff_time.isoformat(),)) as cursor:
                    records_deleted = cursor.rowcount
                    
                await db.commit()
                
            total_deleted = events_deleted + records_deleted
            logger.info(f"清理旧记录完成: 删除 {total_deleted} 条记录")
            return total_deleted
            
        except Exception as e:
            logger.error(f"清理旧记录失败: {e}")
            return 0