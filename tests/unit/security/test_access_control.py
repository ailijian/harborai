#!/usr/bin/env python3
"""
访问控制模块测试

测试 harborai.security.access_control 模块的所有功能，包括：
- 用户认证
- 权限管理
- 会话管理
- 安全控制

遵循TDD流程和VIBE编码规范，目标覆盖率≥90%
"""

import pytest
import time
import uuid
from unittest.mock import patch, Mock
from typing import Set

from harborai.security.access_control import (
    AccessControlManager,
    User,
    PermissionType
)


class TestUser:
    """测试User数据类"""
    
    def test_user_creation_with_defaults(self):
        """测试用户创建时的默认值"""
        # Given: 创建用户时只提供必需参数
        user = User(
            user_id="test_user",
            username="testuser",
            email="test@example.com"
        )
        
        # Then: 验证默认值设置正确
        assert user.user_id == "test_user"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.permissions == set()
        assert user.last_login is None
        assert user.failed_login_attempts == 0
    
    def test_user_creation_with_custom_values(self):
        """测试用户创建时的自定义值"""
        # Given: 创建用户时提供所有参数
        permissions = {PermissionType.READ, PermissionType.WRITE}
        user = User(
            user_id="custom_user",
            username="customuser",
            email="custom@example.com",
            is_active=False,
            permissions=permissions,
            last_login=1234567890.0,
            failed_login_attempts=2
        )
        
        # Then: 验证所有值设置正确
        assert user.user_id == "custom_user"
        assert user.username == "customuser"
        assert user.email == "custom@example.com"
        assert user.is_active is False
        assert user.permissions == permissions
        assert user.last_login == 1234567890.0
        assert user.failed_login_attempts == 2
    
    def test_user_permissions_modification(self):
        """测试用户权限修改"""
        # Given: 创建用户
        user = User(
            user_id="test_user",
            username="testuser",
            email="test@example.com"
        )
        
        # When: 添加权限
        user.permissions.add(PermissionType.READ)
        user.permissions.add(PermissionType.WRITE)
        
        # Then: 验证权限添加成功
        assert PermissionType.READ in user.permissions
        assert PermissionType.WRITE in user.permissions
        assert len(user.permissions) == 2


class TestAccessControlManagerInitialization:
    """测试AccessControlManager初始化"""
    
    def test_manager_initialization(self):
        """测试管理器初始化"""
        # When: 创建访问控制管理器
        manager = AccessControlManager()
        
        # Then: 验证初始化状态
        assert isinstance(manager.users, dict)
        assert isinstance(manager.sessions, dict)
        assert manager.max_failed_attempts == 3
        assert manager.session_timeout == 3600
        assert manager.lockout_duration == 300
        
        # 验证默认用户创建
        assert "admin" in manager.users
        assert "user1" in manager.users
        
        admin_user = manager.users["admin"]
        assert admin_user.username == "admin"
        assert admin_user.email == "admin@example.com"
        assert PermissionType.ADMIN in admin_user.permissions
        assert PermissionType.READ in admin_user.permissions
        assert PermissionType.WRITE in admin_user.permissions
        assert PermissionType.DELETE in admin_user.permissions
        
        regular_user = manager.users["user1"]
        assert regular_user.username == "user1"
        assert regular_user.email == "user1@example.com"
        assert PermissionType.READ in regular_user.permissions
        assert PermissionType.WRITE in regular_user.permissions
        assert PermissionType.ADMIN not in regular_user.permissions
    
    def test_default_users_creation(self):
        """测试默认用户创建"""
        # When: 创建管理器
        manager = AccessControlManager()
        
        # Then: 验证默认用户
        assert len(manager.users) == 2
        
        # 验证admin用户
        admin = manager.users["admin"]
        assert admin.user_id == "admin"
        assert admin.is_active is True
        expected_admin_permissions = {
            PermissionType.READ, 
            PermissionType.WRITE, 
            PermissionType.DELETE, 
            PermissionType.ADMIN
        }
        assert admin.permissions == expected_admin_permissions
        
        # 验证普通用户
        user1 = manager.users["user1"]
        assert user1.user_id == "user1"
        assert user1.is_active is True
        expected_user_permissions = {PermissionType.READ, PermissionType.WRITE}
        assert user1.permissions == expected_user_permissions


class TestUserAuthentication:
    """测试用户认证功能"""
    
    @pytest.fixture
    def manager(self):
        """创建访问控制管理器实例"""
        return AccessControlManager()
    
    def test_successful_authentication(self, manager):
        """测试成功认证"""
        # Given: 有效的用户名和密码
        username = "admin"
        password = "password123"
        
        # When: 进行认证
        result = manager.authenticate_user(username, password)
        
        # Then: 认证成功
        assert result is True
        
        # 验证用户状态更新
        user = manager.users[username]
        assert user.last_login is not None
        assert user.failed_login_attempts == 0
    
    def test_authentication_with_invalid_username(self, manager):
        """测试无效用户名认证"""
        # Given: 无效的用户名
        username = "nonexistent"
        password = "password123"
        
        # When: 进行认证
        result = manager.authenticate_user(username, password)
        
        # Then: 认证失败
        assert result is False
    
    def test_authentication_with_invalid_password(self, manager):
        """测试无效密码认证"""
        # Given: 有效用户名但无效密码
        username = "admin"
        password = "wrongpassword"
        
        # When: 进行认证
        result = manager.authenticate_user(username, password)
        
        # Then: 认证失败
        assert result is False
        
        # 验证失败次数增加
        user = manager.users[username]
        assert user.failed_login_attempts == 1
    
    def test_authentication_with_inactive_user(self, manager):
        """测试非活跃用户认证"""
        # Given: 设置用户为非活跃状态
        username = "admin"
        password = "password123"
        manager.users[username].is_active = False
        
        # When: 进行认证
        result = manager.authenticate_user(username, password)
        
        # Then: 认证失败
        assert result is False
    
    def test_authentication_lockout_after_max_failures(self, manager):
        """测试达到最大失败次数后的锁定"""
        # Given: 用户已达到最大失败次数
        username = "admin"
        password_wrong = "wrongpassword"
        password_correct = "password123"
        
        # When: 连续失败认证直到锁定
        for i in range(manager.max_failed_attempts):
            result = manager.authenticate_user(username, password_wrong)
            assert result is False
        
        # Then: 即使密码正确也无法认证
        result = manager.authenticate_user(username, password_correct)
        assert result is False
        
        # 验证失败次数
        user = manager.users[username]
        assert user.failed_login_attempts >= manager.max_failed_attempts
    
    def test_failed_login_attempts_reset_on_success(self, manager):
        """测试成功登录后失败次数重置"""
        # Given: 用户有一些失败尝试
        username = "admin"
        manager.users[username].failed_login_attempts = 2
        
        # When: 成功认证
        result = manager.authenticate_user(username, "password123")
        
        # Then: 失败次数重置
        assert result is True
        assert manager.users[username].failed_login_attempts == 0


class TestSessionManagement:
    """测试会话管理功能"""
    
    @pytest.fixture
    def manager(self):
        """创建访问控制管理器实例"""
        return AccessControlManager()
    
    def test_create_session_for_valid_user(self, manager):
        """测试为有效用户创建会话"""
        # Given: 有效用户
        username = "admin"
        
        # When: 创建会话
        session_id = manager.create_session(username)
        
        # Then: 会话创建成功
        assert session_id is not None
        assert session_id in manager.sessions
        
        session = manager.sessions[session_id]
        assert session["user_id"] == "admin"
        assert session["username"] == username
        assert "created_at" in session
        assert "last_accessed" in session
    
    def test_create_session_for_invalid_user(self, manager):
        """测试为无效用户创建会话"""
        # Given: 无效用户
        username = "nonexistent"
        
        # When: 创建会话
        session_id = manager.create_session(username)
        
        # Then: 会话创建失败
        assert session_id is None
    
    def test_create_session_for_inactive_user(self, manager):
        """测试为非活跃用户创建会话"""
        # Given: 非活跃用户
        username = "admin"
        manager.users[username].is_active = False
        
        # When: 创建会话
        session_id = manager.create_session(username)
        
        # Then: 会话创建失败
        assert session_id is None
    
    @patch('uuid.uuid4')
    def test_session_id_generation(self, mock_uuid, manager):
        """测试会话ID生成"""
        # Given: Mock UUID生成
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="test-session-id")
        
        # When: 创建会话
        session_id = manager.create_session("admin")
        
        # Then: 使用UUID生成会话ID
        assert session_id == "test-session-id"
        mock_uuid.assert_called_once()
    
    def test_validate_valid_session(self, manager):
        """测试验证有效会话"""
        # Given: 创建会话
        session_id = manager.create_session("admin")
        
        # When: 验证会话
        result = manager.validate_session(session_id)
        
        # Then: 验证成功
        assert result is True
        
        # 验证最后访问时间更新
        session = manager.sessions[session_id]
        assert "last_accessed" in session
    
    def test_validate_invalid_session(self, manager):
        """测试验证无效会话"""
        # Given: 无效会话ID
        session_id = "invalid-session-id"
        
        # When: 验证会话
        result = manager.validate_session(session_id)
        
        # Then: 验证失败
        assert result is False
    
    def test_validate_expired_session(self, manager):
        """测试验证过期会话"""
        # Given: 创建会话并设置为过期
        session_id = manager.create_session("admin")
        session = manager.sessions[session_id]
        session["last_accessed"] = time.time() - manager.session_timeout - 1
        
        # When: 验证会话
        result = manager.validate_session(session_id)
        
        # Then: 验证失败且会话被删除
        assert result is False
        assert session_id not in manager.sessions
    
    def test_get_user_from_valid_session(self, manager):
        """测试从有效会话获取用户"""
        # Given: 创建会话
        session_id = manager.create_session("admin")
        
        # When: 从会话获取用户
        user = manager.get_user_from_session(session_id)
        
        # Then: 获取成功
        assert user is not None
        assert user.username == "admin"
        assert user.user_id == "admin"
    
    def test_get_user_from_invalid_session(self, manager):
        """测试从无效会话获取用户"""
        # Given: 无效会话ID
        session_id = "invalid-session-id"
        
        # When: 从会话获取用户
        user = manager.get_user_from_session(session_id)
        
        # Then: 获取失败
        assert user is None
    
    def test_get_user_from_expired_session(self, manager):
        """测试从过期会话获取用户"""
        # Given: 创建过期会话
        session_id = manager.create_session("admin")
        session = manager.sessions[session_id]
        session["last_accessed"] = time.time() - manager.session_timeout - 1
        
        # When: 从会话获取用户
        user = manager.get_user_from_session(session_id)
        
        # Then: 获取失败
        assert user is None


class TestPermissionManagement:
    """测试权限管理功能"""
    
    @pytest.fixture
    def manager(self):
        """创建访问控制管理器实例"""
        return AccessControlManager()
    
    def test_check_permission_for_valid_user_with_permission(self, manager):
        """测试检查有权限的有效用户"""
        # Given: 用户有READ权限
        username = "admin"
        permission = PermissionType.READ
        
        # When: 检查权限
        result = manager.check_permission(username, permission)
        
        # Then: 权限检查通过
        assert result is True
    
    def test_check_permission_for_valid_user_without_permission(self, manager):
        """测试检查无权限的有效用户"""
        # Given: 普通用户没有ADMIN权限
        username = "user1"
        permission = PermissionType.ADMIN
        
        # When: 检查权限
        result = manager.check_permission(username, permission)
        
        # Then: 权限检查失败
        assert result is False
    
    def test_check_permission_for_invalid_user(self, manager):
        """测试检查无效用户权限"""
        # Given: 无效用户
        username = "nonexistent"
        permission = PermissionType.READ
        
        # When: 检查权限
        result = manager.check_permission(username, permission)
        
        # Then: 权限检查失败
        assert result is False
    
    def test_check_permission_for_inactive_user(self, manager):
        """测试检查非活跃用户权限"""
        # Given: 非活跃用户
        username = "admin"
        manager.users[username].is_active = False
        permission = PermissionType.READ
        
        # When: 检查权限
        result = manager.check_permission(username, permission)
        
        # Then: 权限检查失败
        assert result is False
    
    def test_grant_permission_to_valid_user(self, manager):
        """测试为有效用户授予权限"""
        # Given: 用户没有EXECUTE权限
        username = "user1"
        permission = PermissionType.EXECUTE
        assert permission not in manager.users[username].permissions
        
        # When: 授予权限
        result = manager.grant_permission(username, permission)
        
        # Then: 权限授予成功
        assert result is True
        assert permission in manager.users[username].permissions
    
    def test_grant_permission_to_invalid_user(self, manager):
        """测试为无效用户授予权限"""
        # Given: 无效用户
        username = "nonexistent"
        permission = PermissionType.READ
        
        # When: 授予权限
        result = manager.grant_permission(username, permission)
        
        # Then: 权限授予失败
        assert result is False
    
    def test_grant_duplicate_permission(self, manager):
        """测试授予重复权限"""
        # Given: 用户已有READ权限
        username = "admin"
        permission = PermissionType.READ
        assert permission in manager.users[username].permissions
        original_permissions = manager.users[username].permissions.copy()
        
        # When: 再次授予相同权限
        result = manager.grant_permission(username, permission)
        
        # Then: 操作成功但权限集合不变
        assert result is True
        assert manager.users[username].permissions == original_permissions
    
    def test_revoke_permission_from_valid_user(self, manager):
        """测试撤销有效用户权限"""
        # Given: 用户有READ权限
        username = "admin"
        permission = PermissionType.READ
        assert permission in manager.users[username].permissions
        
        # When: 撤销权限
        result = manager.revoke_permission(username, permission)
        
        # Then: 权限撤销成功
        assert result is True
        assert permission not in manager.users[username].permissions
    
    def test_revoke_permission_from_invalid_user(self, manager):
        """测试撤销无效用户权限"""
        # Given: 无效用户
        username = "nonexistent"
        permission = PermissionType.READ
        
        # When: 撤销权限
        result = manager.revoke_permission(username, permission)
        
        # Then: 权限撤销失败
        assert result is False
    
    def test_revoke_nonexistent_permission(self, manager):
        """测试撤销不存在的权限"""
        # Given: 用户没有EXECUTE权限
        username = "user1"
        permission = PermissionType.EXECUTE
        assert permission not in manager.users[username].permissions
        original_permissions = manager.users[username].permissions.copy()
        
        # When: 撤销不存在的权限
        result = manager.revoke_permission(username, permission)
        
        # Then: 操作成功但权限集合不变
        assert result is True
        assert manager.users[username].permissions == original_permissions


class TestPermissionTypes:
    """测试权限类型枚举"""
    
    def test_permission_type_values(self):
        """测试权限类型值"""
        assert PermissionType.READ.value == "read"
        assert PermissionType.WRITE.value == "write"
        assert PermissionType.DELETE.value == "delete"
        assert PermissionType.ADMIN.value == "admin"
        assert PermissionType.EXECUTE.value == "execute"
    
    def test_permission_type_comparison(self):
        """测试权限类型比较"""
        assert PermissionType.READ == PermissionType.READ
        assert PermissionType.READ != PermissionType.WRITE
        
        # 测试在集合中的使用
        permissions = {PermissionType.READ, PermissionType.WRITE}
        assert PermissionType.READ in permissions
        assert PermissionType.ADMIN not in permissions


class TestSecurityBoundaryConditions:
    """测试安全边界条件"""
    
    @pytest.fixture
    def manager(self):
        """创建访问控制管理器实例"""
        return AccessControlManager()
    
    def test_empty_username_authentication(self, manager):
        """测试空用户名认证"""
        result = manager.authenticate_user("", "password123")
        assert result is False
    
    def test_empty_password_authentication(self, manager):
        """测试空密码认证"""
        result = manager.authenticate_user("admin", "")
        assert result is False
    
    def test_none_username_authentication(self, manager):
        """测试None用户名认证"""
        result = manager.authenticate_user(None, "password123")
        assert result is False
    
    def test_none_password_authentication(self, manager):
        """测试None密码认证"""
        result = manager.authenticate_user("admin", None)
        assert result is False
    
    def test_session_timeout_boundary(self, manager):
        """测试会话超时边界条件"""
        # Given: 创建会话
        session_id = manager.create_session("admin")
        session = manager.sessions[session_id]
        
        # When: 设置会话刚好到达超时边界
        session["last_accessed"] = time.time() - manager.session_timeout
        
        # Then: 会话应该仍然有效（边界条件）
        result = manager.validate_session(session_id)
        assert result is True
        
        # When: 设置会话超过超时时间1秒
        session["last_accessed"] = time.time() - manager.session_timeout - 1
        
        # Then: 会话应该过期
        result = manager.validate_session(session_id)
        assert result is False
    
    def test_max_failed_attempts_boundary(self, manager):
        """测试最大失败次数边界条件"""
        username = "admin"
        
        # When: 失败次数刚好达到最大值
        manager.users[username].failed_login_attempts = manager.max_failed_attempts
        
        # Then: 认证应该失败
        result = manager.authenticate_user(username, "password123")
        assert result is False
        
        # When: 失败次数少于最大值
        manager.users[username].failed_login_attempts = manager.max_failed_attempts - 1
        
        # Then: 认证应该成功
        result = manager.authenticate_user(username, "password123")
        assert result is True


class TestUserLockingAndUnlocking:
    """测试用户锁定和解锁功能"""
    
    @pytest.fixture
    def manager(self):
        """创建访问控制管理器实例"""
        return AccessControlManager()
    
    def test_is_user_locked_for_valid_unlocked_user(self, manager):
        """测试检查有效未锁定用户"""
        # Given: 有效用户未被锁定
        username = "admin"
        
        # When: 检查是否锁定
        result = manager.is_user_locked(username)
        
        # Then: 用户未被锁定
        assert result is False
    
    def test_is_user_locked_for_locked_user(self, manager):
        """测试检查被锁定用户"""
        # Given: 用户被锁定
        username = "admin"
        manager.users[username].failed_login_attempts = manager.max_failed_attempts
        
        # When: 检查是否锁定
        result = manager.is_user_locked(username)
        
        # Then: 用户被锁定
        assert result is True
    
    def test_is_user_locked_for_invalid_user(self, manager):
        """测试检查无效用户锁定状态"""
        # Given: 无效用户
        username = "nonexistent"
        
        # When: 检查是否锁定
        result = manager.is_user_locked(username)
        
        # Then: 返回True（无效用户视为锁定）
        assert result is True
    
    def test_unlock_valid_locked_user(self, manager):
        """测试解锁有效被锁定用户"""
        # Given: 用户被锁定
        username = "admin"
        manager.users[username].failed_login_attempts = manager.max_failed_attempts
        
        # When: 解锁用户
        result = manager.unlock_user(username)
        
        # Then: 解锁成功
        assert result is True
        assert manager.users[username].failed_login_attempts == 0
        assert not manager.is_user_locked(username)
    
    def test_unlock_invalid_user(self, manager):
        """测试解锁无效用户"""
        # Given: 无效用户
        username = "nonexistent"
        
        # When: 解锁用户
        result = manager.unlock_user(username)
        
        # Then: 解锁失败
        assert result is False
    
    def test_unlock_already_unlocked_user(self, manager):
        """测试解锁已解锁用户"""
        # Given: 用户未被锁定
        username = "admin"
        assert manager.users[username].failed_login_attempts == 0
        
        # When: 解锁用户
        result = manager.unlock_user(username)
        
        # Then: 操作成功
        assert result is True
        assert manager.users[username].failed_login_attempts == 0


class TestSessionLogout:
    """测试会话登出功能"""
    
    @pytest.fixture
    def manager(self):
        """创建访问控制管理器实例"""
        return AccessControlManager()
    
    def test_logout_valid_session(self, manager):
        """测试登出有效会话"""
        # Given: 创建会话
        session_id = manager.create_session("admin")
        assert session_id in manager.sessions
        
        # When: 登出
        result = manager.logout_user(session_id)
        
        # Then: 登出成功
        assert result is True
        assert session_id not in manager.sessions
    
    def test_logout_invalid_session(self, manager):
        """测试登出无效会话"""
        # Given: 无效会话ID
        session_id = "invalid-session-id"
        
        # When: 登出
        result = manager.logout_user(session_id)
        
        # Then: 登出失败
        assert result is False
    
    def test_logout_already_logged_out_session(self, manager):
        """测试登出已登出会话"""
        # Given: 创建会话然后登出
        session_id = manager.create_session("admin")
        manager.logout_user(session_id)
        
        # When: 再次登出
        result = manager.logout_user(session_id)
        
        # Then: 登出失败
        assert result is False


class TestActiveSessionsManagement:
    """测试活跃会话管理"""
    
    @pytest.fixture
    def manager(self):
        """创建访问控制管理器实例"""
        return AccessControlManager()
    
    def test_get_active_sessions_empty(self, manager):
        """测试获取空的活跃会话列表"""
        # When: 获取活跃会话
        active_sessions = manager.get_active_sessions()
        
        # Then: 返回空列表
        assert active_sessions == []
    
    def test_get_active_sessions_with_valid_sessions(self, manager):
        """测试获取有效活跃会话"""
        # Given: 创建多个会话
        session1 = manager.create_session("admin")
        session2 = manager.create_session("user1")
        
        # When: 获取活跃会话
        active_sessions = manager.get_active_sessions()
        
        # Then: 返回所有活跃会话
        assert len(active_sessions) == 2
        
        session_ids = [s["session_id"] for s in active_sessions]
        assert session1 in session_ids
        assert session2 in session_ids
        
        # 验证会话信息完整性
        for session in active_sessions:
            assert "session_id" in session
            assert "username" in session
            assert "created_at" in session
            assert "last_accessed" in session
    
    def test_get_active_sessions_excludes_expired(self, manager):
        """测试获取活跃会话排除过期会话"""
        # Given: 创建会话并设置为过期
        session1 = manager.create_session("admin")
        session2 = manager.create_session("user1")
        
        # 设置session1为过期
        manager.sessions[session1]["last_accessed"] = time.time() - manager.session_timeout - 1
        
        # When: 获取活跃会话
        active_sessions = manager.get_active_sessions()
        
        # Then: 只返回未过期的会话
        assert len(active_sessions) == 1
        assert active_sessions[0]["session_id"] == session2
        assert active_sessions[0]["username"] == "user1"
    
    def test_get_active_sessions_boundary_timeout(self, manager):
        """测试活跃会话超时边界条件"""
        # Given: 创建会话并设置为刚好到达超时边界
        session_id = manager.create_session("admin")
        manager.sessions[session_id]["last_accessed"] = time.time() - manager.session_timeout
        
        # When: 获取活跃会话
        active_sessions = manager.get_active_sessions()
        
        # Then: 会话仍然活跃
        assert len(active_sessions) == 1
        assert active_sessions[0]["session_id"] == session_id


class TestIntegrationScenarios:
    """测试集成场景"""
    
    @pytest.fixture
    def manager(self):
        """创建访问控制管理器实例"""
        return AccessControlManager()
    
    def test_complete_user_workflow(self, manager):
        """测试完整的用户工作流程"""
        username = "admin"
        password = "password123"
        
        # 1. 用户认证
        auth_result = manager.authenticate_user(username, password)
        assert auth_result is True
        
        # 2. 创建会话
        session_id = manager.create_session(username)
        assert session_id is not None
        
        # 3. 验证会话
        session_valid = manager.validate_session(session_id)
        assert session_valid is True
        
        # 4. 从会话获取用户
        user = manager.get_user_from_session(session_id)
        assert user is not None
        assert user.username == username
        
        # 5. 检查权限
        has_read = manager.check_permission(username, PermissionType.READ)
        assert has_read is True
        
        # 6. 授予新权限
        grant_result = manager.grant_permission(username, PermissionType.EXECUTE)
        assert grant_result is True
        
        # 7. 验证新权限
        has_execute = manager.check_permission(username, PermissionType.EXECUTE)
        assert has_execute is True
    
    def test_security_violation_scenario(self, manager):
        """测试安全违规场景"""
        username = "user1"  # 普通用户
        
        # 1. 普通用户尝试获取管理员权限
        has_admin = manager.check_permission(username, PermissionType.ADMIN)
        assert has_admin is False
        
        # 2. 尝试多次错误登录
        for i in range(manager.max_failed_attempts):
            result = manager.authenticate_user(username, "wrongpassword")
            assert result is False
        
        # 3. 用户被锁定，即使密码正确也无法登录
        result = manager.authenticate_user(username, "password123")
        assert result is False
        
        # 4. 无法创建会话
        session_id = manager.create_session(username)
        assert session_id is None
    
    def test_concurrent_sessions_scenario(self, manager):
        """测试并发会话场景"""
        username = "admin"
        
        # 1. 创建多个会话
        session1 = manager.create_session(username)
        session2 = manager.create_session(username)
        session3 = manager.create_session(username)
        
        assert session1 is not None
        assert session2 is not None
        assert session3 is not None
        assert session1 != session2 != session3
        
        # 2. 所有会话都应该有效
        assert manager.validate_session(session1) is True
        assert manager.validate_session(session2) is True
        assert manager.validate_session(session3) is True
        
        # 3. 从所有会话都能获取用户
        user1 = manager.get_user_from_session(session1)
        user2 = manager.get_user_from_session(session2)
        user3 = manager.get_user_from_session(session3)
        
        assert user1.username == username
        assert user2.username == username
        assert user3.username == username


class TestPerformanceAndScalability:
    """测试性能和可扩展性"""
    
    @pytest.fixture
    def manager(self):
        """创建访问控制管理器实例"""
        return AccessControlManager()
    
    def test_large_number_of_users(self, manager):
        """测试大量用户场景"""
        # Given: 创建大量用户
        num_users = 1000
        for i in range(num_users):
            user = User(
                user_id=f"user_{i}",
                username=f"user_{i}",
                email=f"user_{i}@example.com",
                permissions={PermissionType.READ}
            )
            manager.users[f"user_{i}"] = user
        
        # When: 检查权限
        start_time = time.time()
        for i in range(100):  # 检查100个用户的权限
            result = manager.check_permission(f"user_{i}", PermissionType.READ)
            assert result is True
        end_time = time.time()
        
        # Then: 性能应该在可接受范围内（每次检查<1ms）
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.001  # 小于1毫秒
    
    def test_large_number_of_sessions(self, manager):
        """测试大量会话场景"""
        # Given: 创建大量会话
        sessions = []
        for i in range(100):
            session_id = manager.create_session("admin")
            sessions.append(session_id)
        
        # When: 验证所有会话
        start_time = time.time()
        for session_id in sessions:
            result = manager.validate_session(session_id)
            assert result is True
        end_time = time.time()
        
        # Then: 性能应该在可接受范围内
        avg_time = (end_time - start_time) / len(sessions)
        assert avg_time < 0.001  # 小于1毫秒
    
    def test_memory_usage_with_many_permissions(self, manager):
        """测试大量权限的内存使用"""
        # Given: 为用户添加大量权限
        username = "admin"
        user = manager.users[username]
        
        # 添加所有可能的权限类型
        all_permissions = set(PermissionType)
        user.permissions = all_permissions
        
        # When: 检查所有权限
        for permission in all_permissions:
            result = manager.check_permission(username, permission)
            assert result is True
        
        # Then: 操作应该成功完成
        assert len(user.permissions) == len(all_permissions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])