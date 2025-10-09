#!/usr/bin/env python3
"""
审计日志模块测试

测试 harborai.security.audit_logger 模块的所有功能，包括：
- 审计事件记录
- 日志文件写入
- 事件查询和过滤
- 安全摘要生成

遵循TDD流程和VIBE编码规范，目标覆盖率≥90%
"""

import pytest
import time
import json
import tempfile
import os
from unittest.mock import patch, mock_open, Mock
from typing import Dict, Any

from harborai.security.audit_logger import (
    AuditLogger,
    AuditEvent,
    AuditEventType,
    SeverityLevel
)


class TestAuditEventType:
    """测试审计事件类型枚举"""
    
    def test_audit_event_type_values(self):
        """测试审计事件类型值"""
        assert AuditEventType.LOGIN_SUCCESS.value == "login_success"
        assert AuditEventType.LOGIN_FAILURE.value == "login_failure"
        assert AuditEventType.LOGOUT.value == "logout"
        assert AuditEventType.PERMISSION_GRANTED.value == "permission_granted"
        assert AuditEventType.PERMISSION_REVOKED.value == "permission_revoked"
        assert AuditEventType.DATA_ACCESS.value == "data_access"
        assert AuditEventType.DATA_MODIFICATION.value == "data_modification"
        assert AuditEventType.SECURITY_VIOLATION.value == "security_violation"
        assert AuditEventType.SYSTEM_ERROR.value == "system_error"
        assert AuditEventType.API_CALL.value == "api_call"
        assert AuditEventType.CONFIG_CHANGE.value == "config_change"


class TestSeverityLevel:
    """测试严重级别枚举"""
    
    def test_severity_level_values(self):
        """测试严重级别值"""
        assert SeverityLevel.LOW.value == "low"
        assert SeverityLevel.MEDIUM.value == "medium"
        assert SeverityLevel.HIGH.value == "high"
        assert SeverityLevel.CRITICAL.value == "critical"


class TestAuditEvent:
    """测试审计事件数据类"""
    
    def test_audit_event_creation(self):
        """测试审计事件创建"""
        # Given: 创建审计事件
        event = AuditEvent(
            event_id="test-event-id",
            event_type=AuditEventType.LOGIN_SUCCESS,
            timestamp=1234567890.0,
            user_id="user123",
            session_id="session123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            resource="/api/login",
            action="用户登录",
            result="成功",
            severity=SeverityLevel.LOW,
            details={"key": "value"}
        )
        
        # Then: 验证事件属性
        assert event.event_id == "test-event-id"
        assert event.event_type == AuditEventType.LOGIN_SUCCESS
        assert event.timestamp == 1234567890.0
        assert event.user_id == "user123"
        assert event.session_id == "session123"
        assert event.ip_address == "192.168.1.1"
        assert event.user_agent == "Mozilla/5.0"
        assert event.resource == "/api/login"
        assert event.action == "用户登录"
        assert event.result == "成功"
        assert event.severity == SeverityLevel.LOW
        assert event.details == {"key": "value"}
    
    def test_audit_event_to_dict(self):
        """测试审计事件转换为字典"""
        # Given: 创建审计事件
        event = AuditEvent(
            event_id="test-event-id",
            event_type=AuditEventType.LOGIN_SUCCESS,
            timestamp=1234567890.0,
            user_id="user123",
            session_id="session123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            resource="/api/login",
            action="用户登录",
            result="成功",
            severity=SeverityLevel.LOW,
            details={"key": "value"}
        )
        
        # When: 转换为字典
        event_dict = event.to_dict()
        
        # Then: 验证字典内容
        assert event_dict["event_id"] == "test-event-id"
        assert event_dict["event_type"] == "login_success"  # 枚举值
        assert event_dict["timestamp"] == 1234567890.0
        assert event_dict["user_id"] == "user123"
        assert event_dict["session_id"] == "session123"
        assert event_dict["ip_address"] == "192.168.1.1"
        assert event_dict["user_agent"] == "Mozilla/5.0"
        assert event_dict["resource"] == "/api/login"
        assert event_dict["action"] == "用户登录"
        assert event_dict["result"] == "成功"
        assert event_dict["severity"] == "low"  # 枚举值
        assert event_dict["details"] == {"key": "value"}
    
    def test_audit_event_with_none_values(self):
        """测试包含None值的审计事件"""
        # Given: 创建包含None值的审计事件
        event = AuditEvent(
            event_id="test-event-id",
            event_type=AuditEventType.SYSTEM_ERROR,
            timestamp=1234567890.0,
            user_id=None,
            session_id=None,
            ip_address=None,
            user_agent=None,
            resource=None,
            action="系统错误",
            result="失败",
            severity=SeverityLevel.HIGH,
            details={}
        )
        
        # When: 转换为字典
        event_dict = event.to_dict()
        
        # Then: 验证None值正确处理
        assert event_dict["user_id"] is None
        assert event_dict["session_id"] is None
        assert event_dict["ip_address"] is None
        assert event_dict["user_agent"] is None
        assert event_dict["resource"] is None
        assert event_dict["details"] == {}


class TestAuditLoggerInitialization:
    """测试审计日志记录器初始化"""
    
    def test_audit_logger_default_initialization(self):
        """测试默认初始化"""
        # When: 创建审计日志记录器
        logger = AuditLogger()
        
        # Then: 验证默认值
        assert logger.log_file == "audit.log"
        assert logger.events == []
        assert logger.max_events_in_memory == 1000
    
    def test_audit_logger_custom_initialization(self):
        """测试自定义初始化"""
        # When: 创建自定义审计日志记录器
        logger = AuditLogger(log_file="custom_audit.log")
        
        # Then: 验证自定义值
        assert logger.log_file == "custom_audit.log"
        assert logger.events == []
        assert logger.max_events_in_memory == 1000


class TestAuditLoggerEventLogging:
    """测试审计日志记录器事件记录"""
    
    @pytest.fixture
    def logger(self):
        """创建审计日志记录器实例"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name
        return AuditLogger(log_file=log_file)
    
    @pytest.fixture
    def cleanup_log_file(self, logger):
        """清理日志文件"""
        yield
        try:
            os.unlink(logger.log_file)
        except FileNotFoundError:
            pass
    
    @patch('uuid.uuid4')
    @patch('time.time')
    def test_log_event_basic(self, mock_time, mock_uuid, logger, cleanup_log_file):
        """测试基本事件记录"""
        # Given: Mock时间和UUID
        mock_time.return_value = 1234567890.0
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-event-id")
        
        # When: 记录事件
        event_id = logger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            action="用户登录",
            result="成功",
            user_id="user123"
        )
        
        # Then: 验证事件记录
        assert event_id == "test-event-id"
        assert len(logger.events) == 1
        
        event = logger.events[0]
        assert event.event_id == "test-event-id"
        assert event.event_type == AuditEventType.LOGIN_SUCCESS
        assert event.timestamp == 1234567890.0
        assert event.user_id == "user123"
        assert event.action == "用户登录"
        assert event.result == "成功"
        assert event.severity == SeverityLevel.LOW  # 默认值
    
    @patch('uuid.uuid4')
    @patch('time.time')
    def test_log_event_with_all_parameters(self, mock_time, mock_uuid, logger, cleanup_log_file):
        """测试记录包含所有参数的事件"""
        # Given: Mock时间和UUID
        mock_time.return_value = 1234567890.0
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-event-id")
        
        # When: 记录包含所有参数的事件
        event_id = logger.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            action="安全违规",
            result="检测到",
            severity=SeverityLevel.HIGH,
            user_id="user123",
            session_id="session123",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            resource="/api/sensitive",
            details={"violation_type": "unauthorized_access"}
        )
        
        # Then: 验证事件记录
        assert event_id == "test-event-id"
        assert len(logger.events) == 1
        
        event = logger.events[0]
        assert event.event_id == "test-event-id"
        assert event.event_type == AuditEventType.SECURITY_VIOLATION
        assert event.severity == SeverityLevel.HIGH
        assert event.user_id == "user123"
        assert event.session_id == "session123"
        assert event.ip_address == "192.168.1.1"
        assert event.user_agent == "Mozilla/5.0"
        assert event.resource == "/api/sensitive"
        assert event.details == {"violation_type": "unauthorized_access"}
    
    def test_log_event_memory_limit(self, logger, cleanup_log_file):
        """测试内存中事件数量限制"""
        # Given: 设置较小的内存限制
        logger.max_events_in_memory = 5
        
        # When: 记录超过限制的事件
        for i in range(10):
            logger.log_event(
                event_type=AuditEventType.API_CALL,
                action=f"API调用 {i}",
                result="成功"
            )
        
        # Then: 验证内存中只保留最新的事件
        assert len(logger.events) == 5
        
        # 验证保留的是最新的事件
        actions = [event.action for event in logger.events]
        expected_actions = [f"API调用 {i}" for i in range(5, 10)]
        assert actions == expected_actions
    
    @patch('builtins.open', mock_open())
    @patch('json.dumps')
    def test_write_to_file_success(self, mock_json_dumps, logger):
        """测试成功写入文件"""
        # Given: Mock JSON序列化
        mock_json_dumps.return_value = '{"test": "data"}'
        
        # When: 记录事件
        logger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            action="用户登录",
            result="成功"
        )
        
        # Then: 验证文件写入
        mock_json_dumps.assert_called_once()
    
    @patch('builtins.open', side_effect=IOError("File write error"))
    @patch('builtins.print')
    def test_write_to_file_failure(self, mock_print, mock_open_error, logger):
        """测试文件写入失败"""
        # When: 记录事件（文件写入失败）
        logger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            action="用户登录",
            result="成功"
        )
        
        # Then: 验证错误处理
        mock_print.assert_called_once()
        assert "Failed to write audit log" in str(mock_print.call_args)
        
        # 验证事件仍然在内存中
        assert len(logger.events) == 1


class TestAuditLoggerSpecificMethods:
    """测试审计日志记录器特定方法"""
    
    @pytest.fixture
    def logger(self):
        """创建审计日志记录器实例"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name
        return AuditLogger(log_file=log_file)
    
    @pytest.fixture
    def cleanup_log_file(self, logger):
        """清理日志文件"""
        yield
        try:
            os.unlink(logger.log_file)
        except FileNotFoundError:
            pass
    
    def test_log_login_success(self, logger, cleanup_log_file):
        """测试记录登录成功事件"""
        # When: 记录登录成功
        logger.log_login_success(
            user_id="user123",
            session_id="session123",
            ip_address="192.168.1.1"
        )
        
        # Then: 验证事件记录
        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == AuditEventType.LOGIN_SUCCESS
        assert event.action == "用户登录"
        assert event.result == "成功"
        assert event.severity == SeverityLevel.LOW
        assert event.user_id == "user123"
        assert event.session_id == "session123"
        assert event.ip_address == "192.168.1.1"
    
    def test_log_login_failure(self, logger, cleanup_log_file):
        """测试记录登录失败事件"""
        # When: 记录登录失败
        logger.log_login_failure(
            username="testuser",
            ip_address="192.168.1.1",
            reason="密码错误"
        )
        
        # Then: 验证事件记录
        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == AuditEventType.LOGIN_FAILURE
        assert event.action == "用户登录"
        assert event.result == "失败"
        assert event.severity == SeverityLevel.MEDIUM
        assert event.ip_address == "192.168.1.1"
        assert event.details == {"username": "testuser", "reason": "密码错误"}
    
    def test_log_login_failure_default_reason(self, logger, cleanup_log_file):
        """测试记录登录失败事件（默认原因）"""
        # When: 记录登录失败（不提供原因）
        logger.log_login_failure(username="testuser")
        
        # Then: 验证默认原因
        event = logger.events[0]
        assert event.details == {"username": "testuser", "reason": "密码错误"}
    
    def test_log_logout(self, logger, cleanup_log_file):
        """测试记录登出事件"""
        # When: 记录登出
        logger.log_logout(user_id="user123", session_id="session123")
        
        # Then: 验证事件记录
        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == AuditEventType.LOGOUT
        assert event.action == "用户登出"
        assert event.result == "成功"
        assert event.severity == SeverityLevel.LOW
        assert event.user_id == "user123"
        assert event.session_id == "session123"
    
    def test_log_permission_change_granted(self, logger, cleanup_log_file):
        """测试记录权限授予事件"""
        # When: 记录权限授予
        logger.log_permission_change(
            user_id="user123",
            permission="read",
            granted=True,
            admin_user="admin"
        )
        
        # Then: 验证事件记录
        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == AuditEventType.PERMISSION_GRANTED
        assert event.action == "授予权限: read"
        assert event.result == "成功"
        assert event.severity == SeverityLevel.MEDIUM
        assert event.user_id == "admin"
        assert event.details == {"target_user": "user123", "permission": "read"}
    
    def test_log_permission_change_revoked(self, logger, cleanup_log_file):
        """测试记录权限撤销事件"""
        # When: 记录权限撤销
        logger.log_permission_change(
            user_id="user123",
            permission="write",
            granted=False,
            admin_user="admin"
        )
        
        # Then: 验证事件记录
        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == AuditEventType.PERMISSION_REVOKED
        assert event.action == "撤销权限: write"
        assert event.result == "成功"
        assert event.severity == SeverityLevel.MEDIUM
        assert event.user_id == "admin"
        assert event.details == {"target_user": "user123", "permission": "write"}
    
    def test_log_data_access(self, logger, cleanup_log_file):
        """测试记录数据访问事件"""
        # When: 记录数据访问
        logger.log_data_access(
            user_id="user123",
            resource="/api/users",
            action="查询用户列表",
            result="成功"
        )
        
        # Then: 验证事件记录
        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == AuditEventType.DATA_ACCESS
        assert event.action == "访问数据: 查询用户列表"
        assert event.result == "成功"
        assert event.severity == SeverityLevel.LOW
        assert event.user_id == "user123"
        assert event.resource == "/api/users"
    
    def test_log_security_violation(self, logger, cleanup_log_file):
        """测试记录安全违规事件"""
        # When: 记录安全违规
        violation_details = {"violation_type": "unauthorized_access", "attempted_resource": "/admin"}
        logger.log_security_violation(
            user_id="user123",
            violation_type="未授权访问",
            details=violation_details
        )
        
        # Then: 验证事件记录
        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == AuditEventType.SECURITY_VIOLATION
        assert event.action == "安全违规: 未授权访问"
        assert event.result == "检测到"
        assert event.severity == SeverityLevel.HIGH
        assert event.user_id == "user123"
        assert event.details == violation_details
    
    def test_log_api_call_success(self, logger, cleanup_log_file):
        """测试记录成功API调用事件"""
        # When: 记录成功API调用
        logger.log_api_call(
            user_id="user123",
            endpoint="/api/users",
            method="GET",
            status_code=200,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0"
        )
        
        # Then: 验证事件记录
        assert len(logger.events) == 1
        event = logger.events[0]
        assert event.event_type == AuditEventType.API_CALL
        assert event.action == "GET /api/users"
        assert event.result == "HTTP 200"
        assert event.severity == SeverityLevel.LOW
        assert event.user_id == "user123"
        assert event.ip_address == "192.168.1.1"
        assert event.user_agent == "Mozilla/5.0"
        assert event.resource == "/api/users"
        assert event.details == {"method": "GET", "status_code": 200}
    
    def test_log_api_call_client_error(self, logger, cleanup_log_file):
        """测试记录客户端错误API调用事件"""
        # When: 记录客户端错误API调用
        logger.log_api_call(
            user_id="user123",
            endpoint="/api/users",
            method="POST",
            status_code=400
        )
        
        # Then: 验证事件记录
        event = logger.events[0]
        assert event.severity == SeverityLevel.MEDIUM
        assert event.result == "HTTP 400"
    
    def test_log_api_call_server_error(self, logger, cleanup_log_file):
        """测试记录服务器错误API调用事件"""
        # When: 记录服务器错误API调用
        logger.log_api_call(
            user_id="user123",
            endpoint="/api/users",
            method="POST",
            status_code=500
        )
        
        # Then: 验证事件记录
        event = logger.events[0]
        assert event.severity == SeverityLevel.HIGH
        assert event.result == "HTTP 500"


class TestAuditLoggerEventQuerying:
    """测试审计日志记录器事件查询"""
    
    @pytest.fixture
    def logger_with_events(self):
        """创建包含测试事件的审计日志记录器"""
        logger = AuditLogger()
        
        # 添加测试事件
        base_time = 1234567890.0
        
        # 登录成功事件
        logger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            action="用户登录",
            result="成功",
            user_id="user1",
            severity=SeverityLevel.LOW
        )
        logger.events[-1].timestamp = base_time
        
        # 登录失败事件
        logger.log_event(
            event_type=AuditEventType.LOGIN_FAILURE,
            action="用户登录",
            result="失败",
            user_id="user2",
            severity=SeverityLevel.MEDIUM
        )
        logger.events[-1].timestamp = base_time + 100
        
        # 安全违规事件
        logger.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            action="安全违规",
            result="检测到",
            user_id="user1",
            severity=SeverityLevel.HIGH
        )
        logger.events[-1].timestamp = base_time + 200
        
        # API调用事件
        logger.log_event(
            event_type=AuditEventType.API_CALL,
            action="API调用",
            result="成功",
            user_id="user3",
            severity=SeverityLevel.LOW
        )
        logger.events[-1].timestamp = base_time + 300
        
        return logger
    
    def test_get_events_no_filters(self, logger_with_events):
        """测试获取所有事件（无过滤器）"""
        # When: 获取所有事件
        events = logger_with_events.get_events()
        
        # Then: 返回所有事件，按时间倒序
        assert len(events) == 4
        assert events[0].event_type == AuditEventType.API_CALL  # 最新的
        assert events[1].event_type == AuditEventType.SECURITY_VIOLATION
        assert events[2].event_type == AuditEventType.LOGIN_FAILURE
        assert events[3].event_type == AuditEventType.LOGIN_SUCCESS  # 最旧的
    
    def test_get_events_by_time_range(self, logger_with_events):
        """测试按时间范围获取事件"""
        # When: 获取特定时间范围的事件
        start_time = 1234567890.0 + 50
        end_time = 1234567890.0 + 250
        events = logger_with_events.get_events(start_time=start_time, end_time=end_time)
        
        # Then: 返回时间范围内的事件
        assert len(events) == 2
        assert events[0].event_type == AuditEventType.SECURITY_VIOLATION
        assert events[1].event_type == AuditEventType.LOGIN_FAILURE
    
    def test_get_events_by_event_type(self, logger_with_events):
        """测试按事件类型获取事件"""
        # When: 获取特定类型的事件
        events = logger_with_events.get_events(event_type=AuditEventType.LOGIN_SUCCESS)
        
        # Then: 返回指定类型的事件
        assert len(events) == 1
        assert events[0].event_type == AuditEventType.LOGIN_SUCCESS
    
    def test_get_events_by_user_id(self, logger_with_events):
        """测试按用户ID获取事件"""
        # When: 获取特定用户的事件
        events = logger_with_events.get_events(user_id="user1")
        
        # Then: 返回指定用户的事件
        assert len(events) == 2
        assert all(event.user_id == "user1" for event in events)
        assert events[0].event_type == AuditEventType.SECURITY_VIOLATION
        assert events[1].event_type == AuditEventType.LOGIN_SUCCESS
    
    def test_get_events_by_severity(self, logger_with_events):
        """测试按严重级别获取事件"""
        # When: 获取高严重级别的事件
        events = logger_with_events.get_events(severity=SeverityLevel.HIGH)
        
        # Then: 返回指定严重级别的事件
        assert len(events) == 1
        assert events[0].severity == SeverityLevel.HIGH
        assert events[0].event_type == AuditEventType.SECURITY_VIOLATION
    
    def test_get_events_with_limit(self, logger_with_events):
        """测试限制返回事件数量"""
        # When: 限制返回数量
        events = logger_with_events.get_events(limit=2)
        
        # Then: 返回限制数量的事件
        assert len(events) == 2
        assert events[0].event_type == AuditEventType.API_CALL
        assert events[1].event_type == AuditEventType.SECURITY_VIOLATION
    
    def test_get_events_multiple_filters(self, logger_with_events):
        """测试多个过滤器组合"""
        # When: 使用多个过滤器
        events = logger_with_events.get_events(
            user_id="user1",
            severity=SeverityLevel.LOW,
            limit=1
        )
        
        # Then: 返回符合所有条件的事件
        assert len(events) == 1
        assert events[0].user_id == "user1"
        assert events[0].severity == SeverityLevel.LOW
        assert events[0].event_type == AuditEventType.LOGIN_SUCCESS
    
    def test_get_events_no_matches(self, logger_with_events):
        """测试无匹配事件"""
        # When: 查询不存在的条件
        events = logger_with_events.get_events(user_id="nonexistent")
        
        # Then: 返回空列表
        assert len(events) == 0


class TestAuditLoggerSecuritySummary:
    """测试审计日志记录器安全摘要"""
    
    @pytest.fixture
    def logger_with_recent_events(self):
        """创建包含最近事件的审计日志记录器"""
        logger = AuditLogger()
        
        current_time = time.time()
        
        # 添加最近24小时内的事件
        events_data = [
            (AuditEventType.LOGIN_SUCCESS, "user1", SeverityLevel.LOW),
            (AuditEventType.LOGIN_SUCCESS, "user2", SeverityLevel.LOW),
            (AuditEventType.LOGIN_FAILURE, "user3", SeverityLevel.MEDIUM),
            (AuditEventType.LOGIN_FAILURE, "user3", SeverityLevel.MEDIUM),
            (AuditEventType.SECURITY_VIOLATION, "user1", SeverityLevel.HIGH),
            (AuditEventType.API_CALL, "user1", SeverityLevel.LOW),
            (AuditEventType.API_CALL, "user2", SeverityLevel.LOW),
            (AuditEventType.API_CALL, "user3", SeverityLevel.MEDIUM),
            (AuditEventType.SYSTEM_ERROR, None, SeverityLevel.CRITICAL),
        ]
        
        for i, (event_type, user_id, severity) in enumerate(events_data):
            logger.log_event(
                event_type=event_type,
                action="测试操作",
                result="测试结果",
                user_id=user_id,
                severity=severity
            )
            # 设置为最近时间
            logger.events[-1].timestamp = current_time - (i * 100)
        
        # 添加24小时前的事件（应该被排除）
        logger.log_event(
            event_type=AuditEventType.LOGIN_SUCCESS,
            action="旧事件",
            result="成功",
            user_id="old_user",
            severity=SeverityLevel.LOW
        )
        logger.events[-1].timestamp = current_time - (25 * 3600)  # 25小时前
        
        return logger
    
    def test_get_security_summary_default_24_hours(self, logger_with_recent_events):
        """测试获取默认24小时安全摘要"""
        # When: 获取安全摘要
        summary = logger_with_recent_events.get_security_summary()
        
        # Then: 验证摘要内容
        assert summary["total_events"] == 9  # 排除24小时前的事件
        assert summary["login_attempts"] == 2  # LOGIN_SUCCESS事件
        assert summary["failed_logins"] == 2   # LOGIN_FAILURE事件
        assert summary["security_violations"] == 1
        assert summary["api_calls"] == 3
        assert summary["high_severity_events"] == 2  # HIGH + CRITICAL
        assert summary["unique_users"] == 3  # user1, user2, user3（排除None）
        
        # 验证事件类型统计
        expected_event_types = {
            "login_success": 2,
            "login_failure": 2,
            "security_violation": 1,
            "api_call": 3,
            "system_error": 1
        }
        assert summary["event_types"] == expected_event_types
    
    def test_get_security_summary_custom_hours(self, logger_with_recent_events):
        """测试获取自定义时间范围安全摘要"""
        # When: 获取最近1小时的安全摘要
        summary = logger_with_recent_events.get_security_summary(hours=1)
        
        # Then: 验证摘要内容（应该包含更少的事件）
        assert summary["total_events"] <= 9
        assert summary["unique_users"] <= 3
    
    def test_get_security_summary_empty_events(self):
        """测试空事件列表的安全摘要"""
        # Given: 空的审计日志记录器
        logger = AuditLogger()
        
        # When: 获取安全摘要
        summary = logger.get_security_summary()
        
        # Then: 验证空摘要
        assert summary["total_events"] == 0
        assert summary["login_attempts"] == 0
        assert summary["failed_logins"] == 0
        assert summary["security_violations"] == 0
        assert summary["api_calls"] == 0
        assert summary["high_severity_events"] == 0
        assert summary["unique_users"] == 0
        assert summary["event_types"] == {}


class TestAuditLoggerIntegrationScenarios:
    """测试审计日志记录器集成场景"""
    
    @pytest.fixture
    def logger(self):
        """创建审计日志记录器实例"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
            log_file = f.name
        return AuditLogger(log_file=log_file)
    
    @pytest.fixture
    def cleanup_log_file(self, logger):
        """清理日志文件"""
        yield
        try:
            os.unlink(logger.log_file)
        except FileNotFoundError:
            pass
    
    def test_complete_user_session_audit(self, logger, cleanup_log_file):
        """测试完整用户会话审计"""
        user_id = "user123"
        session_id = "session123"
        ip_address = "192.168.1.1"
        
        # 1. 用户登录成功
        logger.log_login_success(user_id, session_id, ip_address)
        
        # 2. 用户进行数据访问
        logger.log_data_access(user_id, "/api/users", "查询用户列表", "成功")
        
        # 3. 用户进行API调用
        logger.log_api_call(user_id, "/api/profile", "GET", 200, ip_address)
        
        # 4. 用户登出
        logger.log_logout(user_id, session_id)
        
        # Then: 验证完整的审计轨迹
        assert len(logger.events) == 4
        
        # 验证所有预期的事件类型都存在
        events = logger.get_events()
        event_types = [event.event_type for event in events]
        
        assert AuditEventType.LOGIN_SUCCESS in event_types
        assert AuditEventType.DATA_ACCESS in event_types
        assert AuditEventType.API_CALL in event_types
        assert AuditEventType.LOGOUT in event_types
        
        # 验证用户ID一致性
        for event in events:
            assert event.user_id == user_id
    
    def test_security_incident_audit(self, logger, cleanup_log_file):
        """测试安全事件审计"""
        # 1. 多次登录失败
        for i in range(3):
            logger.log_login_failure("attacker", "192.168.1.100", "密码错误")
        
        # 2. 安全违规检测
        logger.log_security_violation(
            "attacker",
            "暴力破解",
            {"attempts": 3, "ip": "192.168.1.100"}
        )
        
        # 3. 系统错误（可能由攻击引起）
        logger.log_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            action="系统过载",
            result="服务暂停",
            severity=SeverityLevel.CRITICAL,
            details={"cause": "too_many_requests"}
        )
        
        # Then: 验证安全事件记录
        assert len(logger.events) == 5
        
        # 验证高严重级别事件
        high_severity_events = logger.get_events(severity=SeverityLevel.HIGH)
        critical_events = logger.get_events(severity=SeverityLevel.CRITICAL)
        assert len(high_severity_events) == 1
        assert len(critical_events) == 1
        
        # 验证安全摘要
        summary = logger.get_security_summary()
        assert summary["failed_logins"] == 3
        assert summary["security_violations"] == 1
        assert summary["high_severity_events"] == 2  # HIGH + CRITICAL
    
    def test_audit_log_file_persistence(self, logger, cleanup_log_file):
        """测试审计日志文件持久化"""
        # When: 记录多个事件
        logger.log_login_success("user1", "session1")
        logger.log_api_call("user1", "/api/test", "GET", 200)
        logger.log_logout("user1", "session1")
        
        # Then: 验证文件内容
        with open(logger.log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        assert len(lines) == 3
        
        # 验证每行都是有效的JSON
        for line in lines:
            event_data = json.loads(line.strip())
            assert "event_id" in event_data
            assert "event_type" in event_data
            assert "timestamp" in event_data
            assert "action" in event_data
            assert "result" in event_data


class TestAuditLoggerPerformanceAndScalability:
    """测试审计日志记录器性能和可扩展性"""
    
    def test_large_number_of_events_performance(self):
        """测试大量事件的性能"""
        logger = AuditLogger()
        
        # When: 记录大量事件
        start_time = time.time()
        for i in range(1000):
            logger.log_event(
                event_type=AuditEventType.API_CALL,
                action=f"API调用 {i}",
                result="成功",
                user_id=f"user_{i % 10}"  # 10个不同用户
            )
        end_time = time.time()
        
        # Then: 验证性能
        total_time = end_time - start_time
        avg_time_per_event = total_time / 1000
        assert avg_time_per_event < 0.001  # 每个事件小于1毫秒
        
        # 验证内存限制生效
        assert len(logger.events) == logger.max_events_in_memory
    
    def test_event_querying_performance(self):
        """测试事件查询性能"""
        logger = AuditLogger()
        
        # Given: 创建大量事件
        for i in range(1000):
            logger.log_event(
                event_type=AuditEventType.API_CALL,
                action=f"API调用 {i}",
                result="成功",
                user_id=f"user_{i % 10}"
            )
        
        # When: 执行查询
        start_time = time.time()
        events = logger.get_events(user_id="user_1", limit=50)
        end_time = time.time()
        
        # Then: 验证查询性能
        query_time = end_time - start_time
        assert query_time < 0.1  # 查询时间小于100毫秒
        assert len(events) <= 50
    
    def test_security_summary_performance(self):
        """测试安全摘要生成性能"""
        logger = AuditLogger()
        
        # Given: 创建大量不同类型的事件
        event_types = [
            AuditEventType.LOGIN_SUCCESS,
            AuditEventType.LOGIN_FAILURE,
            AuditEventType.API_CALL,
            AuditEventType.SECURITY_VIOLATION
        ]
        
        for i in range(1000):
            logger.log_event(
                event_type=event_types[i % len(event_types)],
                action=f"操作 {i}",
                result="结果",
                user_id=f"user_{i % 20}"
            )
        
        # When: 生成安全摘要
        start_time = time.time()
        summary = logger.get_security_summary()
        end_time = time.time()
        
        # Then: 验证摘要生成性能
        summary_time = end_time - start_time
        assert summary_time < 0.1  # 摘要生成时间小于100毫秒
        
        # 验证摘要内容正确性
        assert summary["total_events"] == logger.max_events_in_memory
        assert summary["unique_users"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])