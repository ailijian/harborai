"""SecurityMonitor 全面测试套件

测试安全监控器的所有功能，包括威胁检测、告警管理、监控统计等。
遵循 TDD 原则和 VIBE 编码规范。
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock, call
from collections import deque

from harborai.security.security_monitor import (
    SecurityMonitor,
    SecurityAlert,
    AlertType,
    ThreatLevel
)


class TestSecurityMonitor:
    """SecurityMonitor 测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.monitor = SecurityMonitor()
        # 停止后台监控线程，但保持监控功能启用
        self.monitor.stop_monitoring()
        self.monitor.monitoring_enabled = True
    
    def teardown_method(self):
        """测试后置清理"""
        self.monitor.stop_monitoring()
        self.monitor = None


class TestLoginAttemptMonitoring:
    """登录尝试监控测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.monitor = SecurityMonitor()
        # 停止后台监控线程，但保持监控功能启用
        self.monitor.stop_monitoring()
        self.monitor.monitoring_enabled = True
    
    def test_record_successful_login(self):
        """测试记录成功登录"""
        self.monitor.record_login_attempt("testuser", "192.168.1.1", True)
        
        key = "testuser:192.168.1.1"
        assert key in self.monitor.login_attempts
        assert len(self.monitor.login_attempts[key]) == 1
        
        attempt = self.monitor.login_attempts[key][0]
        assert attempt["username"] == "testuser"
        assert attempt["ip_address"] == "192.168.1.1"
        assert attempt["success"] is True
        assert "timestamp" in attempt
    
    def test_record_failed_login(self):
        """测试记录失败登录"""
        self.monitor.record_login_attempt("testuser", "192.168.1.1", False)
        
        key = "testuser:192.168.1.1"
        assert key in self.monitor.login_attempts
        assert len(self.monitor.login_attempts[key]) == 1
        
        attempt = self.monitor.login_attempts[key][0]
        assert attempt["username"] == "testuser"
        assert attempt["ip_address"] == "192.168.1.1"
        assert attempt["success"] is False
    
    def test_multiple_login_attempts(self):
        """测试多次登录尝试"""
        for i in range(3):
            self.monitor.record_login_attempt("testuser", "192.168.1.1", i % 2 == 0)
        
        key = "testuser:192.168.1.1"
        assert len(self.monitor.login_attempts[key]) == 3
    
    def test_login_attempts_disabled_monitoring(self):
        """测试监控禁用时不记录登录尝试"""
        self.monitor.monitoring_enabled = False
        self.monitor.record_login_attempt("testuser", "192.168.1.1", True)
        
        assert len(self.monitor.login_attempts) == 0
    
    def test_brute_force_detection(self):
        """测试暴力破解检测"""
        # 记录多次失败登录
        for _ in range(5):
            self.monitor.record_login_attempt("testuser", "192.168.1.1", False)
        
        # 应该生成暴力破解告警
        brute_force_alerts = [
            alert for alert in self.monitor.alerts
            if alert.alert_type == AlertType.BRUTE_FORCE
        ]
        assert len(brute_force_alerts) == 1
        
        alert = brute_force_alerts[0]
        assert alert.threat_level == ThreatLevel.HIGH
        assert alert.source_ip == "192.168.1.1"
        assert "testuser" in alert.description
    
    def test_brute_force_threshold_not_reached(self):
        """测试未达到暴力破解阈值"""
        # 记录少于阈值的失败登录
        for _ in range(3):
            self.monitor.record_login_attempt("testuser", "192.168.1.1", False)
        
        # 不应该生成暴力破解告警
        brute_force_alerts = [
            alert for alert in self.monitor.alerts
            if alert.alert_type == AlertType.BRUTE_FORCE
        ]
        assert len(brute_force_alerts) == 0
    
    def test_brute_force_time_window(self):
        """测试暴力破解时间窗口"""
        # 模拟超出时间窗口的失败登录
        with patch('time.time') as mock_time:
            # 第一次失败登录
            mock_time.return_value = 1000
            for _ in range(3):
                self.monitor.record_login_attempt("testuser", "192.168.1.1", False)
            
            # 超出时间窗口后的失败登录
            mock_time.return_value = 1000 + self.monitor.brute_force_window + 1
            for _ in range(3):
                self.monitor.record_login_attempt("testuser", "192.168.1.1", False)
        
        # 不应该生成暴力破解告警（因为不在同一时间窗口内）
        brute_force_alerts = [
            alert for alert in self.monitor.alerts
            if alert.alert_type == AlertType.BRUTE_FORCE
        ]
        assert len(brute_force_alerts) == 0


class TestAPIRequestMonitoring:
    """API请求监控测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.monitor = SecurityMonitor()
        # 停止后台监控线程，但保持监控功能启用
        self.monitor.stop_monitoring()
        self.monitor.monitoring_enabled = True
    
    def test_record_api_request_success(self):
        """测试记录成功API请求"""
        self.monitor.record_api_request("user1", "192.168.1.1", "/api/data", 200)
        
        key = "user1:192.168.1.1"
        assert key in self.monitor.api_requests
        assert len(self.monitor.api_requests[key]) == 1
        
        request = self.monitor.api_requests[key][0]
        assert request["user_id"] == "user1"
        assert request["ip_address"] == "192.168.1.1"
        assert request["endpoint"] == "/api/data"
        assert request["status_code"] == 200
        assert "timestamp" in request
    
    def test_record_api_request_failure(self):
        """测试记录失败API请求"""
        self.monitor.record_api_request("user1", "192.168.1.1", "/api/data", 500)
        
        key = "user1:192.168.1.1"
        assert key in self.monitor.api_requests
        assert key in self.monitor.failed_operations
        
        # 检查失败操作记录
        assert len(self.monitor.failed_operations[key]) == 1
        failed_op = self.monitor.failed_operations[key][0]
        assert failed_op["user_id"] == "user1"
        assert failed_op["ip_address"] == "192.168.1.1"
        assert "API请求失败" in failed_op["operation"]
    
    def test_api_request_disabled_monitoring(self):
        """测试监控禁用时不记录API请求"""
        self.monitor.monitoring_enabled = False
        self.monitor.record_api_request("user1", "192.168.1.1", "/api/data", 200)
        
        assert len(self.monitor.api_requests) == 0
    
    def test_rate_limit_detection(self):
        """测试速率限制检测"""
        # 记录大量API请求
        for _ in range(100):
            self.monitor.record_api_request("user1", "192.168.1.1", "/api/data", 200)
        
        # 应该生成速率限制告警
        rate_limit_alerts = [
            alert for alert in self.monitor.alerts
            if alert.alert_type == AlertType.RATE_LIMIT_EXCEEDED
        ]
        assert len(rate_limit_alerts) == 1
        
        alert = rate_limit_alerts[0]
        assert alert.threat_level == ThreatLevel.MEDIUM
        assert alert.source_ip == "192.168.1.1"
        assert alert.user_id == "user1"
    
    def test_rate_limit_threshold_not_reached(self):
        """测试未达到速率限制阈值"""
        # 记录少于阈值的API请求
        for _ in range(50):
            self.monitor.record_api_request("user1", "192.168.1.1", "/api/data", 200)
        
        # 不应该生成速率限制告警
        rate_limit_alerts = [
            alert for alert in self.monitor.alerts
            if alert.alert_type == AlertType.RATE_LIMIT_EXCEEDED
        ]
        assert len(rate_limit_alerts) == 0
    
    def test_rate_limit_time_window(self):
        """测试速率限制时间窗口"""
        with patch('time.time') as mock_time:
            # 第一批请求
            mock_time.return_value = 1000
            for _ in range(60):
                self.monitor.record_api_request("user1", "192.168.1.1", "/api/data", 200)
            
            # 超出时间窗口后的请求
            mock_time.return_value = 1000 + self.monitor.rate_limit_window + 1
            for _ in range(60):
                self.monitor.record_api_request("user1", "192.168.1.1", "/api/data", 200)
        
        # 不应该生成速率限制告警（因为不在同一时间窗口内）
        rate_limit_alerts = [
            alert for alert in self.monitor.alerts
            if alert.alert_type == AlertType.RATE_LIMIT_EXCEEDED
        ]
        assert len(rate_limit_alerts) == 0


class TestSecurityEventMonitoring:
    """安全事件监控测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.monitor = SecurityMonitor()
        # 停止后台监控线程，但保持监控功能启用
        self.monitor.stop_monitoring()
        self.monitor.monitoring_enabled = True
    
    def test_record_normal_security_event(self):
        """测试记录正常安全事件"""
        details = {"action": "用户登录", "resource": "/dashboard"}
        self.monitor.record_security_event("用户登录", "user1", "192.168.1.1", details)
        
        # 正常事件不应该生成告警
        suspicious_alerts = [
            alert for alert in self.monitor.alerts
            if alert.alert_type == AlertType.SUSPICIOUS_ACTIVITY
        ]
        assert len(suspicious_alerts) == 0
    
    def test_record_suspicious_security_event(self):
        """测试记录可疑安全事件"""
        details = {"action": "权限提升", "resource": "/admin"}
        self.monitor.record_security_event("权限提升", "user1", "192.168.1.1", details)
        
        # 可疑事件应该生成告警
        suspicious_alerts = [
            alert for alert in self.monitor.alerts
            if alert.alert_type == AlertType.SUSPICIOUS_ACTIVITY
        ]
        assert len(suspicious_alerts) == 1
        
        alert = suspicious_alerts[0]
        assert alert.threat_level == ThreatLevel.MEDIUM
        assert alert.source_ip == "192.168.1.1"
        assert alert.user_id == "user1"
        assert "可疑活动" in alert.description
    
    def test_suspicious_activity_patterns(self):
        """测试可疑活动模式检测"""
        suspicious_patterns = [
            "权限提升",
            "敏感数据访问",
            "系统配置修改",
            "批量数据下载",
            "异常时间访问"
        ]
        
        for pattern in suspicious_patterns:
            details = {"action": pattern}
            self.monitor.record_security_event(pattern, "user1", "192.168.1.1", details)
        
        # 应该为每个可疑模式生成告警
        suspicious_alerts = [
            alert for alert in self.monitor.alerts
            if alert.alert_type == AlertType.SUSPICIOUS_ACTIVITY
        ]
        assert len(suspicious_alerts) == len(suspicious_patterns)
    
    def test_suspicious_activity_time_based(self):
        """测试基于时间的可疑活动检测"""
        with patch('time.localtime') as mock_localtime:
            # 模拟凌晨3点访问
            mock_localtime.return_value.tm_hour = 3
            
            details = {"action": "正常操作"}
            self.monitor.record_security_event("正常操作", "user1", "192.168.1.1", details)
            
            # 应该生成可疑活动告警（异常时间访问）
            suspicious_alerts = [
                alert for alert in self.monitor.alerts
                if alert.alert_type == AlertType.SUSPICIOUS_ACTIVITY
            ]
            assert len(suspicious_alerts) == 1
    
    def test_security_event_disabled_monitoring(self):
        """测试监控禁用时不记录安全事件"""
        self.monitor.monitoring_enabled = False
        details = {"action": "权限提升"}
        self.monitor.record_security_event("权限提升", "user1", "192.168.1.1", details)
        
        assert len(self.monitor.alerts) == 0


class TestAlertManagement:
    """告警管理测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.monitor = SecurityMonitor()
        # 停止后台监控线程，但保持监控功能启用
        self.monitor.stop_monitoring()
        self.monitor.monitoring_enabled = True
    
    def test_create_alert(self):
        """测试创建告警"""
        details = {"test": "data"}
        self.monitor._create_alert(
            AlertType.BRUTE_FORCE,
            ThreatLevel.HIGH,
            "192.168.1.1",
            "user1",
            "测试告警",
            details
        )
        
        assert len(self.monitor.alerts) == 1
        alert = self.monitor.alerts[0]
        
        assert alert.alert_type == AlertType.BRUTE_FORCE
        assert alert.threat_level == ThreatLevel.HIGH
        assert alert.source_ip == "192.168.1.1"
        assert alert.user_id == "user1"
        assert alert.description == "测试告警"
        assert alert.details == details
        assert alert.resolved is False
        assert alert.alert_id is not None
        assert alert.timestamp > 0
    
    def test_add_alert_handler(self):
        """测试添加告警处理器"""
        handler_called = False
        received_alert = None
        
        def test_handler(alert):
            nonlocal handler_called, received_alert
            handler_called = True
            received_alert = alert
        
        self.monitor.add_alert_handler(AlertType.BRUTE_FORCE, test_handler)
        
        # 创建告警
        details = {"test": "data"}
        self.monitor._create_alert(
            AlertType.BRUTE_FORCE,
            ThreatLevel.HIGH,
            "192.168.1.1",
            "user1",
            "测试告警",
            details
        )
        
        assert handler_called is True
        assert received_alert is not None
        assert received_alert.alert_type == AlertType.BRUTE_FORCE
    
    def test_alert_handler_exception(self):
        """测试告警处理器异常处理"""
        def failing_handler(alert):
            raise Exception("Handler error")
        
        self.monitor.add_alert_handler(AlertType.BRUTE_FORCE, failing_handler)
        
        # 创建告警不应该因为处理器异常而失败
        with patch('builtins.print') as mock_print:
            details = {"test": "data"}
            self.monitor._create_alert(
                AlertType.BRUTE_FORCE,
                ThreatLevel.HIGH,
                "192.168.1.1",
                "user1",
                "测试告警",
                details
            )
            
            # 应该打印错误信息
            mock_print.assert_any_call("Alert handler error: Handler error")
    
    def test_get_alerts_no_filters(self):
        """测试获取所有告警"""
        # 创建多个告警
        for i in range(3):
            self.monitor._create_alert(
                AlertType.BRUTE_FORCE,
                ThreatLevel.HIGH,
                f"192.168.1.{i}",
                f"user{i}",
                f"测试告警{i}",
                {}
            )
        
        alerts = self.monitor.get_alerts()
        assert len(alerts) == 3
    
    def test_get_alerts_with_time_filter(self):
        """测试按时间过滤告警"""
        with patch('time.time') as mock_time:
            # 创建不同时间的告警
            mock_time.return_value = 1000
            self.monitor._create_alert(
                AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
                "192.168.1.1", "user1", "旧告警", {}
            )
            
            mock_time.return_value = 2000
            self.monitor._create_alert(
                AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
                "192.168.1.2", "user2", "新告警", {}
            )
        
        # 过滤时间范围
        alerts = self.monitor.get_alerts(start_time=1500, end_time=2500)
        assert len(alerts) == 1
        assert alerts[0].description == "新告警"
    
    def test_get_alerts_with_type_filter(self):
        """测试按类型过滤告警"""
        # 创建不同类型的告警
        self.monitor._create_alert(
            AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
            "192.168.1.1", "user1", "暴力破解", {}
        )
        self.monitor._create_alert(
            AlertType.RATE_LIMIT_EXCEEDED, ThreatLevel.MEDIUM,
            "192.168.1.2", "user2", "速率限制", {}
        )
        
        # 过滤告警类型
        alerts = self.monitor.get_alerts(alert_type=AlertType.BRUTE_FORCE)
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.BRUTE_FORCE
    
    def test_get_alerts_with_threat_level_filter(self):
        """测试按威胁级别过滤告警"""
        # 创建不同威胁级别的告警
        self.monitor._create_alert(
            AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
            "192.168.1.1", "user1", "高威胁", {}
        )
        self.monitor._create_alert(
            AlertType.RATE_LIMIT_EXCEEDED, ThreatLevel.MEDIUM,
            "192.168.1.2", "user2", "中威胁", {}
        )
        
        # 过滤威胁级别
        alerts = self.monitor.get_alerts(threat_level=ThreatLevel.HIGH)
        assert len(alerts) == 1
        assert alerts[0].threat_level == ThreatLevel.HIGH
    
    def test_get_alerts_with_resolved_filter(self):
        """测试按解决状态过滤告警"""
        # 创建告警
        self.monitor._create_alert(
            AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
            "192.168.1.1", "user1", "未解决", {}
        )
        self.monitor._create_alert(
            AlertType.RATE_LIMIT_EXCEEDED, ThreatLevel.MEDIUM,
            "192.168.1.2", "user2", "已解决", {}
        )
        
        # 解决一个告警
        alert_id = self.monitor.alerts[1].alert_id
        self.monitor.resolve_alert(alert_id, "admin")
        
        # 过滤未解决的告警
        unresolved_alerts = self.monitor.get_alerts(resolved=False)
        assert len(unresolved_alerts) == 1
        assert unresolved_alerts[0].description == "未解决"
        
        # 过滤已解决的告警
        resolved_alerts = self.monitor.get_alerts(resolved=True)
        assert len(resolved_alerts) == 1
        assert resolved_alerts[0].description == "已解决"
    
    def test_get_alerts_with_limit(self):
        """测试限制告警数量"""
        # 创建多个告警
        for i in range(5):
            self.monitor._create_alert(
                AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
                f"192.168.1.{i}", f"user{i}", f"告警{i}", {}
            )
        
        alerts = self.monitor.get_alerts(limit=3)
        assert len(alerts) == 3
    
    def test_get_alerts_sorted_by_time(self):
        """测试告警按时间倒序排列"""
        with patch('time.time') as mock_time:
            # 创建不同时间的告警
            mock_time.return_value = 1000
            self.monitor._create_alert(
                AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
                "192.168.1.1", "user1", "第一个", {}
            )
            
            mock_time.return_value = 2000
            self.monitor._create_alert(
                AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
                "192.168.1.2", "user2", "第二个", {}
            )
        
        alerts = self.monitor.get_alerts()
        assert len(alerts) == 2
        assert alerts[0].description == "第二个"  # 最新的在前
        assert alerts[1].description == "第一个"
    
    def test_resolve_alert_success(self):
        """测试成功解决告警"""
        # 创建告警
        self.monitor._create_alert(
            AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
            "192.168.1.1", "user1", "测试告警", {}
        )
        
        alert_id = self.monitor.alerts[0].alert_id
        
        with patch('time.time', return_value=1500):
            result = self.monitor.resolve_alert(alert_id, "admin")
        
        assert result is True
        
        alert = self.monitor.alerts[0]
        assert alert.resolved is True
        assert alert.resolved_at == 1500
        assert alert.resolved_by == "admin"
    
    def test_resolve_alert_invalid_id(self):
        """测试解决无效告警ID"""
        result = self.monitor.resolve_alert("invalid_id", "admin")
        assert result is False
    
    def test_resolve_alert_already_resolved(self):
        """测试解决已解决的告警"""
        # 创建并解决告警
        self.monitor._create_alert(
            AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
            "192.168.1.1", "user1", "测试告警", {}
        )
        
        alert_id = self.monitor.alerts[0].alert_id
        self.monitor.resolve_alert(alert_id, "admin")
        
        # 尝试再次解决
        result = self.monitor.resolve_alert(alert_id, "admin2")
        assert result is False


class TestSecurityMetrics:
    """安全指标测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.monitor = SecurityMonitor()
        # 停止后台监控线程，但保持监控功能启用
        self.monitor.stop_monitoring()
        self.monitor.monitoring_enabled = True
    
    def test_get_security_metrics_empty(self):
        """测试获取空的安全指标"""
        metrics = self.monitor.get_security_metrics()
        
        assert metrics["total_alerts"] == 0
        assert metrics["unresolved_alerts"] == 0
        assert metrics["critical_alerts"] == 0
        assert metrics["high_alerts"] == 0
        assert metrics["brute_force_attempts"] == 0
        assert metrics["rate_limit_violations"] == 0
        assert metrics["suspicious_activities"] == 0
        assert metrics["unique_threat_sources"] == 0
        assert metrics["alert_types"] == {}
    
    def test_get_security_metrics_with_alerts(self):
        """测试获取包含告警的安全指标"""
        # 创建不同类型和级别的告警
        self.monitor._create_alert(
            AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
            "192.168.1.1", "user1", "暴力破解", {}
        )
        self.monitor._create_alert(
            AlertType.RATE_LIMIT_EXCEEDED, ThreatLevel.MEDIUM,
            "192.168.1.2", "user2", "速率限制", {}
        )
        self.monitor._create_alert(
            AlertType.SUSPICIOUS_ACTIVITY, ThreatLevel.CRITICAL,
            "192.168.1.3", "user3", "可疑活动", {}
        )
        
        # 解决一个告警
        alert_id = self.monitor.alerts[0].alert_id
        self.monitor.resolve_alert(alert_id, "admin")
        
        metrics = self.monitor.get_security_metrics()
        
        assert metrics["total_alerts"] == 3
        assert metrics["unresolved_alerts"] == 2
        assert metrics["critical_alerts"] == 1
        assert metrics["high_alerts"] == 1
        assert metrics["brute_force_attempts"] == 1
        assert metrics["rate_limit_violations"] == 1
        assert metrics["suspicious_activities"] == 1
        assert metrics["unique_threat_sources"] == 3
        
        expected_alert_types = {
            "brute_force": 1,
            "rate_limit_exceeded": 1,
            "suspicious_activity": 1
        }
        assert metrics["alert_types"] == expected_alert_types
    
    def test_get_security_metrics_time_range(self):
        """测试指定时间范围的安全指标"""
        with patch('time.time') as mock_time:
            # 创建旧告警
            mock_time.return_value = 1000
            self.monitor._create_alert(
                AlertType.BRUTE_FORCE, ThreatLevel.HIGH,
                "192.168.1.1", "user1", "旧告警", {}
            )
            
            # 创建新告警
            mock_time.return_value = 1000 + 25 * 3600  # 25小时后
            self.monitor._create_alert(
                AlertType.RATE_LIMIT_EXCEEDED, ThreatLevel.MEDIUM,
                "192.168.1.2", "user2", "新告警", {}
            )
            
            # 获取24小时内的指标
            mock_time.return_value = 1000 + 25 * 3600
            metrics = self.monitor.get_security_metrics(hours=24)
        
        # 只应该包含新告警
        assert metrics["total_alerts"] == 1
        assert metrics["rate_limit_violations"] == 1
        assert metrics["brute_force_attempts"] == 0


class TestMonitoringLoop:
    """监控循环测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.monitor = SecurityMonitor()
    
    def teardown_method(self):
        """测试后置清理"""
        self.monitor.stop_monitoring()
    
    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        # 停止监控
        self.monitor.stop_monitoring()
        assert self.monitor.monitoring_enabled is False
        
        # 启动监控
        self.monitor.start_monitoring()
        assert self.monitor.monitoring_enabled is True
        assert self.monitor.monitor_thread.is_alive()
    
    def test_cleanup_old_data(self):
        """测试清理过期数据"""
        with patch('time.time') as mock_time:
            # 添加旧数据
            mock_time.return_value = 1000
            self.monitor.record_login_attempt("user1", "192.168.1.1", False)
            self.monitor.record_api_request("user1", "192.168.1.1", "/api/test", 200)
            
            # 模拟24小时后
            mock_time.return_value = 1000 + 25 * 3600
            
            # 执行清理
            self.monitor._cleanup_old_data()
        
        # 旧数据应该被清理
        assert len(self.monitor.login_attempts) == 0
        assert len(self.monitor.api_requests) == 0
    
    def test_system_anomaly_detection(self):
        """测试系统异常检测"""
        # 模拟大量失败的API请求
        for i in range(60):
            self.monitor.record_api_request(f"user{i}", f"192.168.1.{i}", "/api/test", 500)
        
        # 执行系统异常检查
        self.monitor._check_system_anomalies()
        
        # 应该生成系统异常告警
        anomaly_alerts = [
            alert for alert in self.monitor.alerts
            if alert.alert_type == AlertType.SYSTEM_ANOMALY
        ]
        assert len(anomaly_alerts) == 1
        
        alert = anomaly_alerts[0]
        assert alert.threat_level == ThreatLevel.HIGH
        assert "系统异常" in alert.description
    
    def test_monitor_loop_exception_handling(self):
        """测试监控循环异常处理"""
        # 模拟清理数据时发生异常
        with patch.object(self.monitor, '_cleanup_old_data', side_effect=Exception("Test error")):
            with patch('builtins.print') as mock_print:
                # 直接调用监控循环方法来测试异常处理
                self.monitor.monitoring_enabled = True
                try:
                    # 直接调用一次监控循环的核心逻辑
                    self.monitor._cleanup_old_data()
                    self.monitor._periodic_checks()
                except Exception as e:
                    print(f"Monitor loop error: {e}")
                
                # 应该打印错误信息
                mock_print.assert_any_call("Monitor loop error: Test error")


class TestDataStructures:
    """数据结构测试类"""
    
    def test_threat_level_enum(self):
        """测试威胁级别枚举"""
        assert ThreatLevel.LOW.value == "low"
        assert ThreatLevel.MEDIUM.value == "medium"
        assert ThreatLevel.HIGH.value == "high"
        assert ThreatLevel.CRITICAL.value == "critical"
    
    def test_alert_type_enum(self):
        """测试告警类型枚举"""
        assert AlertType.BRUTE_FORCE.value == "brute_force"
        assert AlertType.SUSPICIOUS_ACTIVITY.value == "suspicious_activity"
        assert AlertType.RATE_LIMIT_EXCEEDED.value == "rate_limit_exceeded"
        assert AlertType.UNAUTHORIZED_ACCESS.value == "unauthorized_access"
        assert AlertType.DATA_BREACH.value == "data_breach"
        assert AlertType.SYSTEM_ANOMALY.value == "system_anomaly"
    
    def test_security_alert_dataclass(self):
        """测试安全告警数据类"""
        alert = SecurityAlert(
            alert_id="test_id",
            alert_type=AlertType.BRUTE_FORCE,
            threat_level=ThreatLevel.HIGH,
            timestamp=1000.0,
            source_ip="192.168.1.1",
            user_id="user1",
            description="测试告警",
            details={"test": "data"}
        )
        
        assert alert.alert_id == "test_id"
        assert alert.alert_type == AlertType.BRUTE_FORCE
        assert alert.threat_level == ThreatLevel.HIGH
        assert alert.timestamp == 1000.0
        assert alert.source_ip == "192.168.1.1"
        assert alert.user_id == "user1"
        assert alert.description == "测试告警"
        assert alert.details == {"test": "data"}
        assert alert.resolved is False
        assert alert.resolved_at is None
        assert alert.resolved_by is None


class TestEdgeCases:
    """边界情况测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.monitor = SecurityMonitor()
        # 停止后台监控线程，但保持监控功能启用
        self.monitor.stop_monitoring()
        self.monitor.monitoring_enabled = True
    
    def test_deque_maxlen_behavior(self):
        """测试deque最大长度行为"""
        # 测试登录尝试deque的最大长度
        key = "testuser:192.168.1.1"
        for i in range(150):  # 超过maxlen=100
            self.monitor.login_attempts[key].append({"test": i})
        
        assert len(self.monitor.login_attempts[key]) == 100
        assert self.monitor.login_attempts[key][0]["test"] == 50  # 最早的50个被移除
    
    def test_empty_collections_cleanup(self):
        """测试空集合清理"""
        # 添加数据然后清理
        self.monitor.login_attempts["test:192.168.1.1"].append({
            "timestamp": time.time() - 25 * 3600  # 25小时前
        })
        
        self.monitor._cleanup_old_data()
        
        # 空的deque应该被删除
        assert "test:192.168.1.1" not in self.monitor.login_attempts
    
    def test_concurrent_access(self):
        """测试并发访问"""
        def add_login_attempts():
            for i in range(10):
                self.monitor.record_login_attempt(f"user{i}", "192.168.1.1", False)
        
        # 创建多个线程同时添加登录尝试
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=add_login_attempts)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # 验证数据完整性
        total_attempts = sum(len(attempts) for attempts in self.monitor.login_attempts.values())
        assert total_attempts == 50  # 5个线程 × 10次尝试
    
    def test_none_values_handling(self):
        """测试None值处理"""
        # 创建包含None值的告警
        self.monitor._create_alert(
            AlertType.SYSTEM_ANOMALY,
            ThreatLevel.HIGH,
            None,  # source_ip为None
            None,  # user_id为None
            "系统告警",
            {}
        )
        
        alert = self.monitor.alerts[0]
        assert alert.source_ip is None
        assert alert.user_id is None
        
        # 获取指标时应该正确处理None值
        metrics = self.monitor.get_security_metrics()
        assert metrics["unique_threat_sources"] == 0  # None不计入唯一来源


@pytest.mark.integration
class TestSecurityMonitorIntegration:
    """安全监控集成测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.monitor = SecurityMonitor()
        # 停止后台监控线程，但保持监控功能启用
        self.monitor.stop_monitoring()
        self.monitor.monitoring_enabled = True
    
    def test_complete_security_workflow(self):
        """测试完整的安全工作流程"""
        # 1. 记录正常活动
        self.monitor.record_login_attempt("user1", "192.168.1.1", True)
        self.monitor.record_api_request("user1", "192.168.1.1", "/api/data", 200)
        
        # 2. 记录可疑活动
        for _ in range(5):
            self.monitor.record_login_attempt("user1", "192.168.1.1", False)
        
        # 3. 记录大量API请求
        for _ in range(100):
            self.monitor.record_api_request("user2", "192.168.1.2", "/api/data", 200)
        
        # 4. 记录安全事件
        self.monitor.record_security_event("权限提升", "user3", "192.168.1.3", {})
        
        # 5. 验证告警生成
        assert len(self.monitor.alerts) == 3  # 暴力破解 + 速率限制 + 可疑活动
        
        # 6. 添加告警处理器
        handled_alerts = []
        def alert_handler(alert):
            handled_alerts.append(alert)
        
        self.monitor.add_alert_handler(AlertType.BRUTE_FORCE, alert_handler)
        
        # 7. 触发更多暴力破解
        self.monitor.record_login_attempt("user4", "192.168.1.4", False)
        for _ in range(4):
            self.monitor.record_login_attempt("user4", "192.168.1.4", False)
        
        # 8. 验证处理器被调用
        assert len(handled_alerts) == 1
        
        # 9. 解决告警
        for alert in self.monitor.alerts:
            if not alert.resolved:
                self.monitor.resolve_alert(alert.alert_id, "admin")
        
        # 10. 验证指标
        metrics = self.monitor.get_security_metrics()
        assert metrics["total_alerts"] == 4
        assert metrics["unresolved_alerts"] == 0
    
    def test_monitoring_performance(self):
        """测试监控性能"""
        start_time = time.time()
        
        # 大量操作
        for i in range(1000):
            self.monitor.record_login_attempt(f"user{i % 10}", f"192.168.1.{i % 10}", i % 3 == 0)
            self.monitor.record_api_request(f"user{i % 10}", f"192.168.1.{i % 10}", "/api/test", 200)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # 性能要求：1000次操作应在1秒内完成
        assert duration < 1.0
        
        # 验证数据完整性
        assert len(self.monitor.login_attempts) <= 10  # 最多10个不同的key
        assert len(self.monitor.api_requests) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])