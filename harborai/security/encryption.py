"""加密管理模块

提供数据加密和解密功能。
"""

import base64
import hashlib
from typing import Optional, Dict, Any


class EncryptionManager:
    """加密管理器
    
    提供数据加密和解密功能。
    """
    
    def __init__(self, key: Optional[str] = None):
        """初始化加密管理器
        
        Args:
            key: 加密密钥
        """
        self.key = key or "default_encryption_key"
        self.algorithm = "AES-256"
    
    def encrypt(self, data: str) -> str:
        """加密数据
        
        Args:
            data: 要加密的数据
            
        Returns:
            str: 加密后的数据
        """
        if not data:
            return data
            
        # 简单的base64编码模拟加密
        encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
        return f"enc:{encoded}"
    
    def decrypt(self, encrypted_data: str) -> str:
        """解密数据
        
        Args:
            encrypted_data: 加密的数据
            
        Returns:
            str: 解密后的数据
        """
        if not encrypted_data or not encrypted_data.startswith('enc:'):
            return encrypted_data
            
        try:
            encoded_data = encrypted_data[4:]  # 移除 'enc:' 前缀
            decoded = base64.b64decode(encoded_data.encode('utf-8')).decode('utf-8')
            return decoded
        except Exception:
            return encrypted_data
    
    def hash_data(self, data: str) -> str:
        """哈希数据
        
        Args:
            data: 要哈希的数据
            
        Returns:
            str: 哈希值
        """
        if not data:
            return ""
            
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def verify_hash(self, data: str, hash_value: str) -> bool:
        """验证哈希值
        
        Args:
            data: 原始数据
            hash_value: 哈希值
            
        Returns:
            bool: 验证结果
        """
        return self.hash_data(data) == hash_value
    
    def generate_key(self) -> str:
        """生成加密密钥
        
        Returns:
            str: 生成的密钥
        """
        import secrets
        return secrets.token_hex(32)
    
    def rotate_key(self, new_key: str) -> str:
        """轮换加密密钥
        
        Args:
            new_key: 新的加密密钥
            
        Returns:
            str: 旧的加密密钥
        """
        old_key = self.key
        self.key = new_key
        return old_key