"""AccessControlManager 全面测试套件

测试访问控制管理器的所有功能，包括用户认证、权限管理、会话管理等。
遵循 TDD 原则和 VIBE 编码规范。
"""

import pytest
import time
import uuid
from unittest.mock import patch, MagicMock

from harborai.security.access_control import (
    AccessControlManager,
    User,
    PermissionType
)


class TestAccessControlManager:
    """AccessControlManager 测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.access_manager = AccessControlManager()
    
    def teardown_method(self):
        """测试后置清理"""
        self.access_manager = None


class TestUserAuthentication:
    """用户认证测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.access_manager = AccessControlManager()
    
    def test_authenticate_valid_user(self):
        """测试有效用户认证"""
        # 测试管理员用户认证
        result = self.access_manager.authenticate_user("admin", "password123")
        assert result is True
        
        # 验证登录时间已更新
        admin_user = self.access_manager.users["admin"]
        assert admin_user.last_login is not None
        assert admin_user.failed_login_attempts == 0
    
    def test_authenticate_invalid_password(self):
        """测试无效密码认证"""
        result = self.access_manager.authenticate_user("admin", "wrongpassword")
        assert result is False
        
        # 验证失败次数增加
        admin_user = self.access_manager.users["admin"]
        assert admin_user.failed_login_attempts == 1
    
    def test_authenticate_nonexistent_user(self):
        """测试不存在的用户认证"""
        result = self.access_manager.authenticate_user("nonexistent", "password123")
        assert result is False
    
    def test_authenticate_inactive_user(self):
        """测试非活跃用户认证"""
        # 设置用户为非活跃状态
        self.access_manager.users["admin"].is_active = False
        
        result = self.access_manager.authenticate_user("admin", "password123")
        assert result is False
    
    def test_user_lockout_after_max_attempts(self):
        """测试用户达到最大失败次数后被锁定"""
        # 连续失败登录
        for _ in range(3):
            result = self.access_manager.authenticate_user("admin", "wrongpassword")
            assert result is False
        
        # 验证用户被锁定
        admin_user = self.access_manager.users["admin"]
        assert admin_user.failed_login_attempts >= self.access_manager.max_failed_attempts
        
        # 即使密码正确也无法登录
        result = self.access_manager.authenticate_user("admin", "password123")
        assert result is False
    
    def test_reset_failed_attempts_on_successful_login(self):
        """测试成功登录后重置失败次数"""
        # 先失败一次
        self.access_manager.authenticate_user("admin", "wrongpassword")
        admin_user = self.access_manager.users["admin"]
        assert admin_user.failed_login_attempts == 1
        
        # 成功登录
        result = self.access_manager.authenticate_user("admin", "password123")
        assert result is True
        assert admin_user.failed_login_attempts == 0


class TestSessionManagement:
    """会话管理测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.access_manager = AccessControlManager()
    
    def test_create_session_valid_user(self):
        """测试为有效用户创建会话"""
        session_id = self.access_manager.create_session("admin")
        
        assert session_id is not None
        assert session_id in self.access_manager.sessions
        
        session = self.access_manager.sessions[session_id]
        assert session["user_id"] == "admin"
        assert session["username"] == "admin"
        assert "created_at" in session
        assert "last_accessed" in session
    
    def test_create_session_invalid_user(self):
        """测试为无效用户创建会话"""
        session_id = self.access_manager.create_session("nonexistent")
        assert session_id is None
    
    def test_create_session_inactive_user(self):
        """测试为非活跃用户创建会话"""
        self.access_manager.users["admin"].is_active = False
        session_id = self.access_manager.create_session("admin")
        assert session_id is None
    
    def test_validate_session_valid(self):
        """测试验证有效会话"""
        session_id = self.access_manager.create_session("admin")
        
        result = self.access_manager.validate_session(session_id)
        assert result is True
    
    def test_validate_session_invalid(self):
        """测试验证无效会话"""
        result = self.access_manager.validate_session("invalid_session_id")
        assert result is False
    
    def test_validate_session_expired(self):
        """测试验证过期会话"""
        session_id = self.access_manager.create_session("admin")
        
        # 模拟会话过期
        session = self.access_manager.sessions[session_id]
        session["last_accessed"] = time.time() - self.access_manager.session_timeout - 1
        
        result = self.access_manager.validate_session(session_id)
        assert result is False
        assert session_id not in self.access_manager.sessions
    
    def test_validate_session_updates_last_accessed(self):
        """测试验证会话时更新最后访问时间"""
        session_id = self.access_manager.create_session("admin")
        original_time = self.access_manager.sessions[session_id]["last_accessed"]
        
        # 等待一小段时间
        time.sleep(0.1)
        
        self.access_manager.validate_session(session_id)
        new_time = self.access_manager.sessions[session_id]["last_accessed"]
        
        assert new_time > original_time
    
    def test_get_user_from_session_valid(self):
        """测试从有效会话获取用户"""
        session_id = self.access_manager.create_session("admin")
        
        user = self.access_manager.get_user_from_session(session_id)
        assert user is not None
        assert user.username == "admin"
        assert user.user_id == "admin"
    
    def test_get_user_from_session_invalid(self):
        """测试从无效会话获取用户"""
        user = self.access_manager.get_user_from_session("invalid_session_id")
        assert user is None
    
    def test_logout_user_valid_session(self):
        """测试有效会话用户登出"""
        session_id = self.access_manager.create_session("admin")
        
        result = self.access_manager.logout_user(session_id)
        assert result is True
        assert session_id not in self.access_manager.sessions
    
    def test_logout_user_invalid_session(self):
        """测试无效会话用户登出"""
        result = self.access_manager.logout_user("invalid_session_id")
        assert result is False
    
    def test_get_active_sessions(self):
        """测试获取活跃会话列表"""
        # 创建多个会话
        session1 = self.access_manager.create_session("admin")
        session2 = self.access_manager.create_session("user1")
        
        active_sessions = self.access_manager.get_active_sessions()
        
        assert len(active_sessions) == 2
        session_ids = [s["session_id"] for s in active_sessions]
        assert session1 in session_ids
        assert session2 in session_ids
    
    def test_get_active_sessions_excludes_expired(self):
        """测试获取活跃会话列表排除过期会话"""
        session_id = self.access_manager.create_session("admin")
        
        # 模拟会话过期
        session = self.access_manager.sessions[session_id]
        session["last_accessed"] = time.time() - self.access_manager.session_timeout - 1
        
        active_sessions = self.access_manager.get_active_sessions()
        assert len(active_sessions) == 0


class TestPermissionManagement:
    """权限管理测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.access_manager = AccessControlManager()
    
    def test_check_permission_valid_user_has_permission(self):
        """测试检查有效用户拥有的权限"""
        result = self.access_manager.check_permission("admin", PermissionType.ADMIN)
        assert result is True
        
        result = self.access_manager.check_permission("user1", PermissionType.READ)
        assert result is True
    
    def test_check_permission_valid_user_no_permission(self):
        """测试检查有效用户没有的权限"""
        result = self.access_manager.check_permission("user1", PermissionType.ADMIN)
        assert result is False
        
        result = self.access_manager.check_permission("user1", PermissionType.DELETE)
        assert result is False
    
    def test_check_permission_invalid_user(self):
        """测试检查无效用户权限"""
        result = self.access_manager.check_permission("nonexistent", PermissionType.READ)
        assert result is False
    
    def test_check_permission_inactive_user(self):
        """测试检查非活跃用户权限"""
        self.access_manager.users["admin"].is_active = False
        
        result = self.access_manager.check_permission("admin", PermissionType.ADMIN)
        assert result is False
    
    def test_grant_permission_valid_user(self):
        """测试为有效用户授予权限"""
        # 确认用户原本没有该权限
        assert not self.access_manager.check_permission("user1", PermissionType.DELETE)
        
        # 授予权限
        result = self.access_manager.grant_permission("user1", PermissionType.DELETE)
        assert result is True
        
        # 验证权限已授予
        assert self.access_manager.check_permission("user1", PermissionType.DELETE)
    
    def test_grant_permission_invalid_user(self):
        """测试为无效用户授予权限"""
        result = self.access_manager.grant_permission("nonexistent", PermissionType.READ)
        assert result is False
    
    def test_revoke_permission_valid_user(self):
        """测试撤销有效用户权限"""
        # 确认用户拥有该权限
        assert self.access_manager.check_permission("admin", PermissionType.READ)
        
        # 撤销权限
        result = self.access_manager.revoke_permission("admin", PermissionType.READ)
        assert result is True
        
        # 验证权限已撤销
        assert not self.access_manager.check_permission("admin", PermissionType.READ)
    
    def test_revoke_permission_invalid_user(self):
        """测试撤销无效用户权限"""
        result = self.access_manager.revoke_permission("nonexistent", PermissionType.READ)
        assert result is False
    
    def test_revoke_nonexistent_permission(self):
        """测试撤销用户不拥有的权限"""
        # 确认用户没有该权限
        assert not self.access_manager.check_permission("user1", PermissionType.DELETE)
        
        # 撤销权限（应该成功，但不会有实际影响）
        result = self.access_manager.revoke_permission("user1", PermissionType.DELETE)
        assert result is True


class TestUserLockout:
    """用户锁定测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.access_manager = AccessControlManager()
    
    def test_is_user_locked_normal_user(self):
        """测试检查正常用户是否被锁定"""
        result = self.access_manager.is_user_locked("admin")
        assert result is False
    
    def test_is_user_locked_after_max_attempts(self):
        """测试检查达到最大失败次数后用户是否被锁定"""
        # 连续失败登录
        for _ in range(3):
            self.access_manager.authenticate_user("admin", "wrongpassword")
        
        result = self.access_manager.is_user_locked("admin")
        assert result is True
    
    def test_is_user_locked_nonexistent_user(self):
        """测试检查不存在用户是否被锁定"""
        result = self.access_manager.is_user_locked("nonexistent")
        assert result is True
    
    def test_unlock_user_valid(self):
        """测试解锁有效用户"""
        # 先锁定用户
        for _ in range(3):
            self.access_manager.authenticate_user("admin", "wrongpassword")
        
        assert self.access_manager.is_user_locked("admin")
        
        # 解锁用户
        result = self.access_manager.unlock_user("admin")
        assert result is True
        assert not self.access_manager.is_user_locked("admin")
        
        # 验证可以正常登录
        auth_result = self.access_manager.authenticate_user("admin", "password123")
        assert auth_result is True
    
    def test_unlock_user_invalid(self):
        """测试解锁无效用户"""
        result = self.access_manager.unlock_user("nonexistent")
        assert result is False


class TestUserDataClass:
    """User 数据类测试"""
    
    def test_user_creation_with_defaults(self):
        """测试使用默认值创建用户"""
        user = User(
            user_id="test_user",
            username="testuser",
            email="test@example.com"
        )
        
        assert user.user_id == "test_user"
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.permissions == set()
        assert user.last_login is None
        assert user.failed_login_attempts == 0
    
    def test_user_creation_with_permissions(self):
        """测试创建带权限的用户"""
        permissions = {PermissionType.READ, PermissionType.WRITE}
        user = User(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            permissions=permissions
        )
        
        assert user.permissions == permissions
    
    def test_user_post_init_permissions(self):
        """测试用户初始化后权限设置"""
        user = User(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            permissions=None
        )
        
        # __post_init__ 应该将 None 转换为空集合
        assert user.permissions == set()


class TestPermissionType:
    """PermissionType 枚举测试"""
    
    def test_permission_type_values(self):
        """测试权限类型枚举值"""
        assert PermissionType.READ.value == "read"
        assert PermissionType.WRITE.value == "write"
        assert PermissionType.DELETE.value == "delete"
        assert PermissionType.ADMIN.value == "admin"
        assert PermissionType.EXECUTE.value == "execute"
    
    def test_permission_type_comparison(self):
        """测试权限类型比较"""
        assert PermissionType.READ == PermissionType.READ
        assert PermissionType.READ != PermissionType.WRITE


class TestDefaultUsers:
    """默认用户测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.access_manager = AccessControlManager()
    
    def test_default_admin_user(self):
        """测试默认管理员用户"""
        admin = self.access_manager.users.get("admin")
        
        assert admin is not None
        assert admin.user_id == "admin"
        assert admin.username == "admin"
        assert admin.email == "admin@example.com"
        assert admin.is_active is True
        
        expected_permissions = {
            PermissionType.READ,
            PermissionType.WRITE,
            PermissionType.DELETE,
            PermissionType.ADMIN
        }
        assert admin.permissions == expected_permissions
    
    def test_default_regular_user(self):
        """测试默认普通用户"""
        user = self.access_manager.users.get("user1")
        
        assert user is not None
        assert user.user_id == "user1"
        assert user.username == "user1"
        assert user.email == "user1@example.com"
        assert user.is_active is True
        
        expected_permissions = {PermissionType.READ, PermissionType.WRITE}
        assert user.permissions == expected_permissions


class TestEdgeCases:
    """边界情况测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.access_manager = AccessControlManager()
    
    def test_session_timeout_boundary(self):
        """测试会话超时边界情况"""
        session_id = self.access_manager.create_session("admin")
        
        # 设置会话刚好在超时边界
        session = self.access_manager.sessions[session_id]
        session["last_accessed"] = time.time() - self.access_manager.session_timeout
        
        # 应该仍然有效（等于超时时间）
        result = self.access_manager.validate_session(session_id)
        assert result is True
        
        # 超过超时时间一秒
        session["last_accessed"] = time.time() - self.access_manager.session_timeout - 1
        result = self.access_manager.validate_session(session_id)
        assert result is False
    
    def test_multiple_sessions_same_user(self):
        """测试同一用户的多个会话"""
        session1 = self.access_manager.create_session("admin")
        session2 = self.access_manager.create_session("admin")
        
        assert session1 != session2
        assert self.access_manager.validate_session(session1)
        assert self.access_manager.validate_session(session2)
        
        # 登出一个会话不应影响另一个
        self.access_manager.logout_user(session1)
        assert not self.access_manager.validate_session(session1)
        assert self.access_manager.validate_session(session2)
    
    def test_permission_set_operations(self):
        """测试权限集合操作"""
        user = self.access_manager.users["user1"]
        original_permissions = user.permissions.copy()
        
        # 添加已存在的权限
        self.access_manager.grant_permission("user1", PermissionType.READ)
        assert user.permissions == original_permissions
        
        # 移除不存在的权限
        self.access_manager.revoke_permission("user1", PermissionType.ADMIN)
        assert user.permissions == original_permissions


@pytest.mark.integration
class TestAccessControlIntegration:
    """访问控制集成测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.access_manager = AccessControlManager()
    
    def test_complete_user_workflow(self):
        """测试完整的用户工作流程"""
        # 1. 用户认证
        auth_result = self.access_manager.authenticate_user("admin", "password123")
        assert auth_result is True
        
        # 2. 创建会话
        session_id = self.access_manager.create_session("admin")
        assert session_id is not None
        
        # 3. 验证会话
        session_valid = self.access_manager.validate_session(session_id)
        assert session_valid is True
        
        # 4. 检查权限
        has_admin_permission = self.access_manager.check_permission("admin", PermissionType.ADMIN)
        assert has_admin_permission is True
        
        # 5. 获取用户信息
        user = self.access_manager.get_user_from_session(session_id)
        assert user is not None
        assert user.username == "admin"
        
        # 6. 用户登出
        logout_result = self.access_manager.logout_user(session_id)
        assert logout_result is True
        
        # 7. 验证会话已失效
        session_valid = self.access_manager.validate_session(session_id)
        assert session_valid is False
    
    def test_permission_management_workflow(self):
        """测试权限管理工作流程"""
        # 1. 检查初始权限
        has_delete = self.access_manager.check_permission("user1", PermissionType.DELETE)
        assert has_delete is False
        
        # 2. 授予权限
        grant_result = self.access_manager.grant_permission("user1", PermissionType.DELETE)
        assert grant_result is True
        
        # 3. 验证权限已授予
        has_delete = self.access_manager.check_permission("user1", PermissionType.DELETE)
        assert has_delete is True
        
        # 4. 撤销权限
        revoke_result = self.access_manager.revoke_permission("user1", PermissionType.DELETE)
        assert revoke_result is True
        
        # 5. 验证权限已撤销
        has_delete = self.access_manager.check_permission("user1", PermissionType.DELETE)
        assert has_delete is False
    
    def test_user_lockout_and_unlock_workflow(self):
        """测试用户锁定和解锁工作流程"""
        # 1. 验证用户未被锁定
        is_locked = self.access_manager.is_user_locked("admin")
        assert is_locked is False
        
        # 2. 连续失败登录导致锁定
        for _ in range(3):
            auth_result = self.access_manager.authenticate_user("admin", "wrongpassword")
            assert auth_result is False
        
        # 3. 验证用户被锁定
        is_locked = self.access_manager.is_user_locked("admin")
        assert is_locked is True
        
        # 4. 即使密码正确也无法登录
        auth_result = self.access_manager.authenticate_user("admin", "password123")
        assert auth_result is False
        
        # 5. 解锁用户
        unlock_result = self.access_manager.unlock_user("admin")
        assert unlock_result is True
        
        # 6. 验证用户已解锁
        is_locked = self.access_manager.is_user_locked("admin")
        assert is_locked is False
        
        # 7. 可以正常登录
        auth_result = self.access_manager.authenticate_user("admin", "password123")
        assert auth_result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])