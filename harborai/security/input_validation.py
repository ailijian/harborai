"""输入验证模块

提供各种输入数据的验证功能，包括email、URL、危险模式检测等。
"""

import re
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse


class InputValidator:
    """输入验证器
    
    提供各种输入数据的验证功能。
    """
    
    def __init__(self):
        """初始化验证器"""
        # Email正则表达式
        self.email_pattern = re.compile(
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        # 危险模式列表
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS脚本
            r'javascript:',  # JavaScript协议
            r'on\w+\s*=',  # 事件处理器
            r'\beval\s*\(',  # eval函数
            r'\bexec\s*\(',  # exec函数
            r'\b(union|select|insert|update|delete|drop)\b',  # SQL注入
            r'\.\./',  # 路径遍历
            r'\\x[0-9a-fA-F]{2}',  # 十六进制编码
        ]
        
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
    
    def validate_email(self, email: str) -> bool:
        """验证email格式
        
        Args:
            email: 要验证的email地址
            
        Returns:
            bool: 验证结果
        """
        if not email or not isinstance(email, str):
            return False
            
        return bool(self.email_pattern.match(email.strip()))
    
    def validate_url(self, url: str) -> bool:
        """验证URL格式
        
        Args:
            url: 要验证的URL
            
        Returns:
            bool: 验证结果
        """
        if not url or not isinstance(url, str):
            return False
            
        try:
            result = urlparse(url.strip())
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    def detect_dangerous_patterns(self, text: str) -> List[str]:
        """检测危险模式
        
        Args:
            text: 要检测的文本
            
        Returns:
            List[str]: 检测到的危险模式列表
        """
        if not text or not isinstance(text, str):
            return []
            
        detected = []
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(text):
                detected.append(self.dangerous_patterns[i])
                
        return detected
    
    def is_safe_input(self, text: str) -> bool:
        """检查输入是否安全
        
        Args:
            text: 要检查的文本
            
        Returns:
            bool: 是否安全
        """
        return len(self.detect_dangerous_patterns(text)) == 0
    
    def validate_input_length(self, text: str, max_length: int = 1000) -> bool:
        """验证输入长度
        
        Args:
            text: 要验证的文本
            max_length: 最大长度
            
        Returns:
            bool: 验证结果
        """
        if not isinstance(text, str):
            return False
            
        return len(text) <= max_length
    
    def sanitize_input(self, text: str) -> str:
        """清理输入文本
        
        Args:
            text: 要清理的文本
            
        Returns:
            str: 清理后的文本
        """
        if not isinstance(text, str):
            return ""
            
        # 移除危险字符
        sanitized = text
        for pattern in self.compiled_patterns:
            sanitized = pattern.sub('', sanitized)
            
        return sanitized.strip()
    
    def validate_api_key_format(self, api_key: str) -> bool:
        """验证API密钥格式
        
        Args:
            api_key: API密钥
            
        Returns:
            bool: 验证结果
        """
        if not api_key or not isinstance(api_key, str):
            return False
            
        # 基本格式检查：长度和字符
        if len(api_key) < 10 or len(api_key) > 200:
            return False
            
        # 检查是否包含有效字符
        valid_chars = re.compile(r'^[a-zA-Z0-9._-]+$')
        return bool(valid_chars.match(api_key))
    
    def validate_json_structure(self, data: Any, required_fields: List[str] = None) -> bool:
        """验证JSON结构
        
        Args:
            data: 要验证的数据
            required_fields: 必需字段列表
            
        Returns:
            bool: 验证结果
        """
        if not isinstance(data, dict):
            return False
            
        if required_fields:
            for field in required_fields:
                if field not in data:
                    return False
                    
        return True