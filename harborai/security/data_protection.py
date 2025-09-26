"""数据保护模块

提供敏感数据掩码、加密等数据保护功能。
"""

import re
import json
from typing import Any, Dict, List, Union


class DataProtectionManager:
    """数据保护管理器
    
    提供敏感数据的掩码、加密等保护功能。
    """
    
    def __init__(self):
        """初始化数据保护管理器"""
        # 敏感数据模式
        self.sensitive_patterns = {
            'api_key': re.compile(r'\b[a-zA-Z0-9]{20,}\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10,11}\b'),
            'credit_card': re.compile(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'password': re.compile(r'(password|pwd|pass)\s*[:=]\s*[^\s]+', re.IGNORECASE)
        }
    
    def mask_api_key(self, api_key: str) -> str:
        """掩码API密钥
        
        Args:
            api_key: 原始API密钥
            
        Returns:
            str: 掩码后的API密钥
        """
        if not api_key or not isinstance(api_key, str):
            return api_key
            
        if len(api_key) <= 8:
            return '*' * len(api_key)
            
        # 显示前4位和后4位，中间用*替代
        return api_key[:4] + '*' * (len(api_key) - 8) + api_key[-4:]
    
    def mask_log_data(self, log_data: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """掩码日志数据
        
        Args:
            log_data: 原始日志数据
            
        Returns:
            Union[str, Dict[str, Any]]: 掩码后的日志数据
        """
        if isinstance(log_data, str):
            return self._mask_string_data(log_data)
        elif isinstance(log_data, dict):
            return self._mask_dict_data(log_data)
        else:
            return log_data
    
    def _mask_string_data(self, text: str) -> str:
        """掩码字符串数据
        
        Args:
            text: 原始文本
            
        Returns:
            str: 掩码后的文本
        """
        if not text:
            return text
            
        masked_text = text
        
        # 掩码API密钥
        for match in self.sensitive_patterns['api_key'].finditer(text):
            original = match.group()
            masked = self.mask_api_key(original)
            masked_text = masked_text.replace(original, masked)
        
        # 掩码邮箱
        for match in self.sensitive_patterns['email'].finditer(masked_text):
            original = match.group()
            parts = original.split('@')
            if len(parts) == 2:
                username = parts[0]
                domain = parts[1]
                if len(username) > 2:
                    masked_username = username[0] + '*' * (len(username) - 2) + username[-1]
                else:
                    masked_username = '*' * len(username)
                masked = f"{masked_username}@{domain}"
                masked_text = masked_text.replace(original, masked)
        
        # 掩码密码
        for match in self.sensitive_patterns['password'].finditer(masked_text):
            original = match.group()
            # 保留键名，掩码值
            if ':' in original:
                key, value = original.split(':', 1)
                masked = f"{key}:***"
            elif '=' in original:
                key, value = original.split('=', 1)
                masked = f"{key}=***"
            else:
                masked = "***"
            masked_text = masked_text.replace(original, masked)
        
        return masked_text
    
    def _mask_dict_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """掩码字典数据
        
        Args:
            data: 原始字典数据
            
        Returns:
            Dict[str, Any]: 掩码后的字典数据
        """
        if not isinstance(data, dict):
            return data
            
        masked_data = {}
        sensitive_keys = ['api_key', 'password', 'token', 'secret', 'key', 'pwd', 'pass']
        
        for key, value in data.items():
            key_lower = key.lower()
            
            # 检查键名是否敏感
            if any(sensitive_key in key_lower for sensitive_key in sensitive_keys):
                if isinstance(value, str):
                    masked_data[key] = self.mask_api_key(value)
                else:
                    masked_data[key] = "***"
            elif isinstance(value, str):
                masked_data[key] = self._mask_string_data(value)
            elif isinstance(value, dict):
                masked_data[key] = self._mask_dict_data(value)
            elif isinstance(value, list):
                masked_data[key] = [self._mask_dict_data(item) if isinstance(item, dict) 
                                  else self._mask_string_data(item) if isinstance(item, str)
                                  else item for item in value]
            else:
                masked_data[key] = value
                
        return masked_data
    
    def encrypt_sensitive_data(self, data: str, key: str = None) -> str:
        """加密敏感数据
        
        Args:
            data: 要加密的数据
            key: 加密密钥
            
        Returns:
            str: 加密后的数据
        """
        # 简单的加密实现（实际应用中应使用更强的加密算法）
        if not data:
            return data
            
        # 模拟加密过程
        import base64
        encoded = base64.b64encode(data.encode('utf-8')).decode('utf-8')
        return f"encrypted:{encoded}"
    
    def decrypt_sensitive_data(self, encrypted_data: str, key: str = None) -> str:
        """解密敏感数据
        
        Args:
            encrypted_data: 加密的数据
            key: 解密密钥
            
        Returns:
            str: 解密后的数据
        """
        if not encrypted_data or not encrypted_data.startswith('encrypted:'):
            return encrypted_data
            
        # 模拟解密过程
        import base64
        try:
            encoded_data = encrypted_data[10:]  # 移除 'encrypted:' 前缀
            decoded = base64.b64decode(encoded_data.encode('utf-8')).decode('utf-8')
            return decoded
        except Exception:
            return encrypted_data
    
    def is_sensitive_data(self, text: str) -> bool:
        """检查是否包含敏感数据
        
        Args:
            text: 要检查的文本
            
        Returns:
            bool: 是否包含敏感数据
        """
        if not text:
            return False
            
        for pattern in self.sensitive_patterns.values():
            if pattern.search(text):
                return True
                
        return False
    
    def get_data_classification(self, data: Union[str, Dict[str, Any]]) -> str:
        """获取数据分类
        
        Args:
            data: 要分类的数据
            
        Returns:
            str: 数据分类（public, internal, confidential, restricted）
        """
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            text = json.dumps(data)
        else:
            text = str(data)
            
        if self.is_sensitive_data(text):
            return "restricted"
        elif any(keyword in text.lower() for keyword in ['internal', 'private', 'confidential']):
            return "confidential"
        elif any(keyword in text.lower() for keyword in ['company', 'business', 'proprietary']):
            return "internal"
        else:
            return "public"