"""访问控制模块

提供用户认证、权限管理、会话管理等访问控制功能。
"""

import time
from typing import Dict, List, Optional, Set
from enum import Enum
from dataclasses import dataclass


class PermissionType(Enum):
    """权限类型枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"


@dataclass
class User:
    """用户数据类"""
    user_id: str
    username: str
    email: str
    is_active: bool = True
    permissions: Set[PermissionType] = None
    last_login: Optional[float] = None
    failed_login_attempts: int = 0
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = set()


class AccessControlManager:
    """访问控制管理器
    
    提供用户认证、权限管理、会话管理等功能。
    """
    
    def __init__(self):
        """初始化访问控制管理器"""
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Dict] = {}
        self.max_failed_attempts = 3
        self.session_timeout = 3600  # 1小时
        self.lockout_duration = 300  # 5分钟
        
        # 创建默认用户
        self._create_default_users()
    
    def _create_default_users(self):
        """创建默认用户"""
        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@example.com",
            permissions={PermissionType.READ, PermissionType.WRITE, PermissionType.DELETE, PermissionType.ADMIN}
        )
        
        regular_user = User(
            user_id="user1",
            username="user1",
            email="user1@example.com",
            permissions={PermissionType.READ, PermissionType.WRITE}
        )
        
        self.users["admin"] = admin_user
        self.users["user1"] = regular_user
    
    def authenticate_user(self, username: str, password: str) -> bool:
        """用户认证
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            bool: 认证结果
        """
        user = self.users.get(username)
        if not user:
            return False
            
        if not user.is_active:
            return False
            
        # 检查是否被锁定
        if user.failed_login_attempts >= self.max_failed_attempts:
            return False
            
        # 简单的密码验证（实际应用中应使用哈希验证）
        if password == "password123":  # 模拟正确密码
            user.last_login = time.time()
            user.failed_login_attempts = 0
            return True
        else:
            user.failed_login_attempts += 1
            return False
    
    def create_session(self, username: str) -> Optional[str]:
        """创建会话
        
        Args:
            username: 用户名
            
        Returns:
            Optional[str]: 会话ID
        """
        user = self.users.get(username)
        if not user or not user.is_active:
            return None
            
        import uuid
        session_id = str(uuid.uuid4())
        
        self.sessions[session_id] = {
            "user_id": user.user_id,
            "username": username,
            "created_at": time.time(),
            "last_accessed": time.time()
        }
        
        return session_id
    
    def validate_session(self, session_id: str) -> bool:
        """验证会话
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 验证结果
        """
        session = self.sessions.get(session_id)
        if not session:
            return False
            
        # 检查会话是否过期
        current_time = time.time()
        if current_time - session["last_accessed"] > self.session_timeout:
            del self.sessions[session_id]
            return False
            
        # 更新最后访问时间
        session["last_accessed"] = current_time
        return True
    
    def get_user_from_session(self, session_id: str) -> Optional[User]:
        """从会话获取用户
        
        Args:
            session_id: 会话ID
            
        Returns:
            Optional[User]: 用户对象
        """
        if not self.validate_session(session_id):
            return None
            
        session = self.sessions.get(session_id)
        if not session:
            return None
            
        return self.users.get(session["username"])
    
    def check_permission(self, username: str, permission: PermissionType) -> bool:
        """检查权限
        
        Args:
            username: 用户名
            permission: 权限类型
            
        Returns:
            bool: 是否有权限
        """
        user = self.users.get(username)
        if not user or not user.is_active:
            return False
            
        return permission in user.permissions
    
    def grant_permission(self, username: str, permission: PermissionType) -> bool:
        """授予权限
        
        Args:
            username: 用户名
            permission: 权限类型
            
        Returns:
            bool: 操作结果
        """
        user = self.users.get(username)
        if not user:
            return False
            
        user.permissions.add(permission)
        return True
    
    def revoke_permission(self, username: str, permission: PermissionType) -> bool:
        """撤销权限
        
        Args:
            username: 用户名
            permission: 权限类型
            
        Returns:
            bool: 操作结果
        """
        user = self.users.get(username)
        if not user:
            return False
            
        user.permissions.discard(permission)
        return True
    
    def logout_user(self, session_id: str) -> bool:
        """用户登出
        
        Args:
            session_id: 会话ID
            
        Returns:
            bool: 操作结果
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def is_user_locked(self, username: str) -> bool:
        """检查用户是否被锁定
        
        Args:
            username: 用户名
            
        Returns:
            bool: 是否被锁定
        """
        user = self.users.get(username)
        if not user:
            return True
            
        return user.failed_login_attempts >= self.max_failed_attempts
    
    def unlock_user(self, username: str) -> bool:
        """解锁用户
        
        Args:
            username: 用户名
            
        Returns:
            bool: 操作结果
        """
        user = self.users.get(username)
        if not user:
            return False
            
        user.failed_login_attempts = 0
        return True
    
    def get_active_sessions(self) -> List[Dict]:
        """获取活跃会话列表
        
        Returns:
            List[Dict]: 活跃会话列表
        """
        current_time = time.time()
        active_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_accessed"] <= self.session_timeout:
                active_sessions.append({
                    "session_id": session_id,
                    "username": session["username"],
                    "created_at": session["created_at"],
                    "last_accessed": session["last_accessed"]
                })
                
        return active_sessions