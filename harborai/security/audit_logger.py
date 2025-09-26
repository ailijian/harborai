"""审计日志模块

提供安全事件记录和审计日志功能。
"""

import json
import time
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict


class AuditEventType(Enum):
    """审计事件类型枚举"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"
    API_CALL = "api_call"
    CONFIG_CHANGE = "config_change"


class SeverityLevel(Enum):
    """严重级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """审计事件数据类"""
    event_id: str
    event_type: AuditEventType
    timestamp: float
    user_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: str
    result: str
    severity: SeverityLevel
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        data['severity'] = self.severity.value
        return data


class AuditLogger:
    """审计日志记录器
    
    提供安全事件记录和审计日志功能。
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """初始化审计日志记录器
        
        Args:
            log_file: 日志文件路径
        """
        self.log_file = log_file or "audit.log"
        self.events: List[AuditEvent] = []
        self.max_events_in_memory = 1000
    
    def log_event(self, 
                  event_type: AuditEventType,
                  action: str,
                  result: str,
                  severity: SeverityLevel = SeverityLevel.LOW,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  resource: Optional[str] = None,
                  details: Optional[Dict[str, Any]] = None) -> str:
        """记录审计事件
        
        Args:
            event_type: 事件类型
            action: 操作描述
            result: 操作结果
            severity: 严重级别
            user_id: 用户ID
            session_id: 会话ID
            ip_address: IP地址
            user_agent: 用户代理
            resource: 资源
            details: 详细信息
            
        Returns:
            str: 事件ID
        """
        import uuid
        
        event_id = str(uuid.uuid4())
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            severity=severity,
            details=details or {}
        )
        
        self.events.append(event)
        
        # 限制内存中的事件数量
        if len(self.events) > self.max_events_in_memory:
            self.events = self.events[-self.max_events_in_memory:]
        
        # 写入日志文件
        self._write_to_file(event)
        
        return event_id
    
    def _write_to_file(self, event: AuditEvent):
        """写入日志文件
        
        Args:
            event: 审计事件
        """
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict(), ensure_ascii=False) + '\n')
        except Exception as e:
            # 记录写入失败，但不抛出异常
            print(f"Failed to write audit log: {e}")
    
    def log_login_success(self, user_id: str, session_id: str, ip_address: str = None):
        """记录登录成功事件"""
        self.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            action="用户登录",
            result="成功",
            severity=SeverityLevel.LOW,
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address
        )
    
    def log_login_failure(self, username: str, ip_address: str = None, reason: str = None):
        """记录登录失败事件"""
        self.log_event(
            event_type=AuditEventType.LOGIN_FAILURE,
            action="用户登录",
            result="失败",
            severity=SeverityLevel.MEDIUM,
            ip_address=ip_address,
            details={"username": username, "reason": reason or "密码错误"}
        )
    
    def log_logout(self, user_id: str, session_id: str):
        """记录登出事件"""
        self.log_event(
            event_type=AuditEventType.LOGOUT,
            action="用户登出",
            result="成功",
            severity=SeverityLevel.LOW,
            user_id=user_id,
            session_id=session_id
        )
    
    def log_permission_change(self, user_id: str, permission: str, granted: bool, admin_user: str):
        """记录权限变更事件"""
        event_type = AuditEventType.PERMISSION_GRANTED if granted else AuditEventType.PERMISSION_REVOKED
        action = f"{'授予' if granted else '撤销'}权限: {permission}"
        
        self.log_event(
            event_type=event_type,
            action=action,
            result="成功",
            severity=SeverityLevel.MEDIUM,
            user_id=admin_user,
            details={"target_user": user_id, "permission": permission}
        )
    
    def log_data_access(self, user_id: str, resource: str, action: str, result: str):
        """记录数据访问事件"""
        self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            action=f"访问数据: {action}",
            result=result,
            severity=SeverityLevel.LOW,
            user_id=user_id,
            resource=resource
        )
    
    def log_security_violation(self, user_id: str, violation_type: str, details: Dict[str, Any]):
        """记录安全违规事件"""
        self.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            action=f"安全违规: {violation_type}",
            result="检测到",
            severity=SeverityLevel.HIGH,
            user_id=user_id,
            details=details
        )
    
    def log_api_call(self, user_id: str, endpoint: str, method: str, status_code: int, 
                     ip_address: str = None, user_agent: str = None):
        """记录API调用事件"""
        severity = SeverityLevel.LOW
        if status_code >= 400:
            severity = SeverityLevel.MEDIUM
        if status_code >= 500:
            severity = SeverityLevel.HIGH
            
        self.log_event(
            event_type=AuditEventType.API_CALL,
            action=f"{method} {endpoint}",
            result=f"HTTP {status_code}",
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=endpoint,
            details={"method": method, "status_code": status_code}
        )
    
    def get_events(self, 
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   event_type: Optional[AuditEventType] = None,
                   user_id: Optional[str] = None,
                   severity: Optional[SeverityLevel] = None,
                   limit: int = 100) -> List[AuditEvent]:
        """获取审计事件
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            event_type: 事件类型
            user_id: 用户ID
            severity: 严重级别
            limit: 限制数量
            
        Returns:
            List[AuditEvent]: 事件列表
        """
        filtered_events = self.events
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
        
        # 按时间倒序排列
        filtered_events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_events[:limit]
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取安全摘要
        
        Args:
            hours: 统计时间范围（小时）
            
        Returns:
            Dict[str, Any]: 安全摘要
        """
        start_time = time.time() - (hours * 3600)
        recent_events = self.get_events(start_time=start_time, limit=10000)
        
        summary = {
            "total_events": len(recent_events),
            "login_attempts": 0,
            "failed_logins": 0,
            "security_violations": 0,
            "api_calls": 0,
            "high_severity_events": 0,
            "unique_users": set(),
            "event_types": {}
        }
        
        for event in recent_events:
            if event.event_type == AuditEventType.LOGIN_SUCCESS:
                summary["login_attempts"] += 1
            elif event.event_type == AuditEventType.LOGIN_FAILURE:
                summary["failed_logins"] += 1
            elif event.event_type == AuditEventType.SECURITY_VIOLATION:
                summary["security_violations"] += 1
            elif event.event_type == AuditEventType.API_CALL:
                summary["api_calls"] += 1
            
            if event.severity in [SeverityLevel.HIGH, SeverityLevel.CRITICAL]:
                summary["high_severity_events"] += 1
            
            if event.user_id:
                summary["unique_users"].add(event.user_id)
            
            event_type_str = event.event_type.value
            summary["event_types"][event_type_str] = summary["event_types"].get(event_type_str, 0) + 1
        
        summary["unique_users"] = len(summary["unique_users"])
        
        return summary