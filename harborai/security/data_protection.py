"""数据保护模块

提供敏感数据掩码、加密等数据保护功能。
"""

import json
from typing import Any, Dict, List, Union


class DataProtectionManager:
    """数据保护管理器
    
    提供敏感数据的掩码、加密等保护功能。
    """
    
    def __init__(self):
        """初始化数据保护管理器"""
        # 敏感数据模式 - 使用简单字符串匹配避免正则表达式问题
        self.sensitive_keywords = {
            'api_key': ['api_key', 'apikey', 'api-key'],
            'email': ['@'],
            'phone': ['phone', 'tel', 'mobile'],
            'credit_card': ['card', 'credit'],
            'ssn': ['ssn', 'social'],
            'password': ['password', 'pwd', 'pass']
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
        
        # 简化的敏感数据掩码 - 使用字符串匹配
        # 掩码包含@的邮箱
        if '@' in masked_text:
            words = masked_text.split()
            for i, word in enumerate(words):
                if '@' in word and '.' in word:
                    parts = word.split('@')
                    if len(parts) == 2:
                        username = parts[0]
                        domain = parts[1]
                        if len(username) > 2:
                            masked_username = username[0] + '*' * (len(username) - 2) + username[-1]
                        else:
                            masked_username = '*' * len(username)
                        words[i] = f"{masked_username}@{domain}"
            masked_text = ' '.join(words)
        
        # 掩码密码相关字段
        for keyword in self.sensitive_keywords['password']:
            if keyword in masked_text.lower():
                # 简单替换：将密码值替换为***
                lines = masked_text.split('\n')
                for i, line in enumerate(lines):
                    if keyword in line.lower():
                        if ':' in line:
                            parts = line.split(':', 1)
                            if len(parts) == 2:
                                lines[i] = f"{parts[0]}:***"
                        elif '=' in line:
                            parts = line.split('=', 1)
                            if len(parts) == 2:
                                lines[i] = f"{parts[0]}=***"
                masked_text = '\n'.join(lines)
        
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
            
        # 使用简单字符串匹配检查敏感数据
        text_lower = text.lower()
        for category, keywords in self.sensitive_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
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