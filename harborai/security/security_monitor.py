"""安全监控模块

提供实时安全监控、威胁检测和告警功能。
"""

import time
import threading
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, deque


class ThreatLevel(Enum):
    """威胁级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型枚举"""
    BRUTE_FORCE = "brute_force"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    SYSTEM_ANOMALY = "system_anomaly"


@dataclass
class SecurityAlert:
    """安全告警数据类"""
    alert_id: str
    alert_type: AlertType
    threat_level: ThreatLevel
    timestamp: float
    source_ip: Optional[str]
    user_id: Optional[str]
    description: str
    details: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None


class SecurityMonitor:
    """安全监控器
    
    提供实时安全监控、威胁检测和告警功能。
    """
    
    def __init__(self):
        """初始化安全监控器"""
        self.alerts: List[SecurityAlert] = []
        self.alert_handlers: Dict[AlertType, List[Callable]] = defaultdict(list)
        self.monitoring_enabled = True
        
        # 监控统计
        self.login_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.api_requests: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.failed_operations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 阈值配置
        self.brute_force_threshold = 5  # 5次失败登录
        self.brute_force_window = 300   # 5分钟窗口
        self.rate_limit_threshold = 100 # 100次请求
        self.rate_limit_window = 60     # 1分钟窗口
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def record_login_attempt(self, username: str, ip_address: str, success: bool):
        """记录登录尝试
        
        Args:
            username: 用户名
            ip_address: IP地址
            success: 是否成功
        """
        if not self.monitoring_enabled:
            return
            
        current_time = time.time()
        key = f"{username}:{ip_address}"
        
        self.login_attempts[key].append({
            "timestamp": current_time,
            "success": success,
            "username": username,
            "ip_address": ip_address
        })
        
        if not success:
            self._check_brute_force_attack(username, ip_address)
    
    def record_api_request(self, user_id: str, ip_address: str, endpoint: str, status_code: int):
        """记录API请求
        
        Args:
            user_id: 用户ID
            ip_address: IP地址
            endpoint: 端点
            status_code: 状态码
        """
        if not self.monitoring_enabled:
            return
            
        current_time = time.time()
        key = f"{user_id}:{ip_address}"
        
        self.api_requests[key].append({
            "timestamp": current_time,
            "endpoint": endpoint,
            "status_code": status_code,
            "user_id": user_id,
            "ip_address": ip_address
        })
        
        self._check_rate_limit(user_id, ip_address)
        
        if status_code >= 400:
            self._record_failed_operation(user_id, ip_address, f"API请求失败: {endpoint}")
    
    def record_security_event(self, event_type: str, user_id: str, ip_address: str, details: Dict[str, Any]):
        """记录安全事件
        
        Args:
            event_type: 事件类型
            user_id: 用户ID
            ip_address: IP地址
            details: 事件详情
        """
        if not self.monitoring_enabled:
            return
            
        # 检查是否为可疑活动
        if self._is_suspicious_activity(event_type, user_id, ip_address, details):
            self._create_alert(
                AlertType.SUSPICIOUS_ACTIVITY,
                ThreatLevel.MEDIUM,
                ip_address,
                user_id,
                f"检测到可疑活动: {event_type}",
                details
            )
    
    def _check_brute_force_attack(self, username: str, ip_address: str):
        """检查暴力破解攻击"""
        current_time = time.time()
        key = f"{username}:{ip_address}"
        
        # 统计时间窗口内的失败登录次数
        recent_failures = [
            attempt for attempt in self.login_attempts[key]
            if not attempt["success"] and 
               current_time - attempt["timestamp"] <= self.brute_force_window
        ]
        
        if len(recent_failures) >= self.brute_force_threshold:
            self._create_alert(
                AlertType.BRUTE_FORCE,
                ThreatLevel.HIGH,
                ip_address,
                None,
                f"检测到暴力破解攻击，用户: {username}",
                {
                    "username": username,
                    "failed_attempts": len(recent_failures),
                    "time_window": self.brute_force_window
                }
            )
    
    def _check_rate_limit(self, user_id: str, ip_address: str):
        """检查速率限制"""
        current_time = time.time()
        key = f"{user_id}:{ip_address}"
        
        # 统计时间窗口内的请求次数
        recent_requests = [
            request for request in self.api_requests[key]
            if current_time - request["timestamp"] <= self.rate_limit_window
        ]
        
        if len(recent_requests) >= self.rate_limit_threshold:
            self._create_alert(
                AlertType.RATE_LIMIT_EXCEEDED,
                ThreatLevel.MEDIUM,
                ip_address,
                user_id,
                f"API请求频率超限，用户: {user_id}",
                {
                    "request_count": len(recent_requests),
                    "time_window": self.rate_limit_window,
                    "threshold": self.rate_limit_threshold
                }
            )
    
    def _record_failed_operation(self, user_id: str, ip_address: str, operation: str):
        """记录失败操作"""
        current_time = time.time()
        key = f"{user_id}:{ip_address}"
        
        self.failed_operations[key].append({
            "timestamp": current_time,
            "operation": operation,
            "user_id": user_id,
            "ip_address": ip_address
        })
    
    def _is_suspicious_activity(self, event_type: str, user_id: str, ip_address: str, details: Dict[str, Any]) -> bool:
        """判断是否为可疑活动"""
        # 简单的可疑活动检测逻辑
        suspicious_patterns = [
            "权限提升",
            "敏感数据访问",
            "系统配置修改",
            "批量数据下载",
            "异常时间访问"
        ]
        
        for pattern in suspicious_patterns:
            if pattern in event_type or pattern in str(details):
                return True
                
        # 检查是否在异常时间访问（凌晨2-6点）
        current_hour = time.localtime().tm_hour
        if 2 <= current_hour <= 6:
            return True
            
        return False
    
    def _create_alert(self, alert_type: AlertType, threat_level: ThreatLevel, 
                     source_ip: str, user_id: str, description: str, details: Dict[str, Any]):
        """创建安全告警"""
        import uuid
        
        alert = SecurityAlert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            threat_level=threat_level,
            timestamp=time.time(),
            source_ip=source_ip,
            user_id=user_id,
            description=description,
            details=details
        )
        
        self.alerts.append(alert)
        
        # 触发告警处理器
        for handler in self.alert_handlers[alert_type]:
            try:
                handler(alert)
            except Exception as e:
                print(f"Alert handler error: {e}")
        
        # 记录告警日志
        print(f"SECURITY ALERT [{threat_level.value.upper()}]: {description}")
    
    def add_alert_handler(self, alert_type: AlertType, handler: Callable[[SecurityAlert], None]):
        """添加告警处理器
        
        Args:
            alert_type: 告警类型
            handler: 处理器函数
        """
        self.alert_handlers[alert_type].append(handler)
    
    def get_alerts(self, 
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   alert_type: Optional[AlertType] = None,
                   threat_level: Optional[ThreatLevel] = None,
                   resolved: Optional[bool] = None,
                   limit: int = 100) -> List[SecurityAlert]:
        """获取安全告警
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            alert_type: 告警类型
            threat_level: 威胁级别
            resolved: 是否已解决
            limit: 限制数量
            
        Returns:
            List[SecurityAlert]: 告警列表
        """
        filtered_alerts = self.alerts
        
        if start_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp >= start_time]
        
        if end_time:
            filtered_alerts = [a for a in filtered_alerts if a.timestamp <= end_time]
        
        if alert_type:
            filtered_alerts = [a for a in filtered_alerts if a.alert_type == alert_type]
        
        if threat_level:
            filtered_alerts = [a for a in filtered_alerts if a.threat_level == threat_level]
        
        if resolved is not None:
            filtered_alerts = [a for a in filtered_alerts if a.resolved == resolved]
        
        # 按时间倒序排列
        filtered_alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_alerts[:limit]
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """解决告警
        
        Args:
            alert_id: 告警ID
            resolved_by: 解决人
            
        Returns:
            bool: 操作结果
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = time.time()
                alert.resolved_by = resolved_by
                return True
        return False
    
    def get_security_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """获取安全指标
        
        Args:
            hours: 统计时间范围（小时）
            
        Returns:
            Dict[str, Any]: 安全指标
        """
        start_time = time.time() - (hours * 3600)
        recent_alerts = self.get_alerts(start_time=start_time, limit=10000)
        
        metrics = {
            "total_alerts": len(recent_alerts),
            "unresolved_alerts": len([a for a in recent_alerts if not a.resolved]),
            "critical_alerts": len([a for a in recent_alerts if a.threat_level == ThreatLevel.CRITICAL]),
            "high_alerts": len([a for a in recent_alerts if a.threat_level == ThreatLevel.HIGH]),
            "brute_force_attempts": len([a for a in recent_alerts if a.alert_type == AlertType.BRUTE_FORCE]),
            "rate_limit_violations": len([a for a in recent_alerts if a.alert_type == AlertType.RATE_LIMIT_EXCEEDED]),
            "suspicious_activities": len([a for a in recent_alerts if a.alert_type == AlertType.SUSPICIOUS_ACTIVITY]),
            "unique_threat_sources": len(set(a.source_ip for a in recent_alerts if a.source_ip)),
            "alert_types": {}
        }
        
        for alert in recent_alerts:
            alert_type_str = alert.alert_type.value
            metrics["alert_types"][alert_type_str] = metrics["alert_types"].get(alert_type_str, 0) + 1
        
        return metrics
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_enabled:
            try:
                # 清理过期数据
                self._cleanup_old_data()
                
                # 执行周期性检查
                self._periodic_checks()
                
                # 休眠1分钟
                time.sleep(60)
            except Exception as e:
                print(f"Monitor loop error: {e}")
    
    def _cleanup_old_data(self):
        """清理过期数据"""
        current_time = time.time()
        cleanup_threshold = 24 * 3600  # 24小时
        
        # 清理过期的登录尝试记录
        for key in list(self.login_attempts.keys()):
            attempts = self.login_attempts[key]
            while attempts and current_time - attempts[0]["timestamp"] > cleanup_threshold:
                attempts.popleft()
            if not attempts:
                del self.login_attempts[key]
        
        # 清理过期的API请求记录
        for key in list(self.api_requests.keys()):
            requests = self.api_requests[key]
            while requests and current_time - requests[0]["timestamp"] > cleanup_threshold:
                requests.popleft()
            if not requests:
                del self.api_requests[key]
    
    def _periodic_checks(self):
        """周期性检查"""
        # 检查系统异常
        self._check_system_anomalies()
    
    def _check_system_anomalies(self):
        """检查系统异常"""
        # 简单的系统异常检测
        current_time = time.time()
        
        # 检查是否有大量失败的API请求
        total_failed_requests = 0
        for requests in self.api_requests.values():
            failed_requests = [
                req for req in requests
                if req["status_code"] >= 500 and 
                   current_time - req["timestamp"] <= 300  # 5分钟内
            ]
            total_failed_requests += len(failed_requests)
        
        if total_failed_requests > 50:  # 5分钟内超过50个500错误
            self._create_alert(
                AlertType.SYSTEM_ANOMALY,
                ThreatLevel.HIGH,
                None,
                None,
                "检测到系统异常：大量API请求失败",
                {"failed_requests": total_failed_requests, "time_window": 300}
            )
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_enabled = False
    
    def start_monitoring(self):
        """启动监控"""
        self.monitoring_enabled = True
        if not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()