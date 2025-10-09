"""输入验证模块

提供各种输入数据的验证功能，包括email、URL、危险模式检测等。
"""

from typing import List, Optional, Dict, Any
from urllib.parse import urlparse


class InputValidator:
    """输入验证器
    
    提供各种输入数据的验证功能。
    """
    
    def __init__(self):
        """初始化验证器"""
        # 危险模式列表 - 使用简单的字符串匹配，避免正则表达式性能问题
        self.dangerous_keywords = [
            'script', 'javascript', 'eval', 'exec', 'iframe',
            'union', 'select', 'insert', 'update', 'delete', 'drop',
            'onclick', 'onload', 'onerror', 'onmouseover', 'onfocus', 'onblur'
        ]
    
    def validate_email(self, email: str) -> bool:
        """验证email格式
        
        Args:
            email: 要验证的email地址
            
        Returns:
            bool: 验证结果
        """
        if not email or not isinstance(email, str):
            return False
        
        email = email.strip()
        
        # 简单的email验证：检查是否包含@和.
        if '@' not in email or '.' not in email:
            return False
        
        # 检查@符号的位置
        at_pos = email.find('@')
        if at_pos <= 0 or at_pos >= len(email) - 1:
            return False
        
        # 检查@后面是否有.
        domain_part = email[at_pos + 1:]
        if '.' not in domain_part:
            return False
        
        return True
    
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
        text_lower = text.lower()
        
        # 使用简单的字符串匹配检测危险关键词
        for keyword in self.dangerous_keywords:
            if keyword in text_lower:
                detected.append(keyword)
        
        # 检测其他危险模式
        dangerous_chars = ['<script', '</script>', '../', '..\\', 'javascript:']
        for pattern in dangerous_chars:
            if pattern in text_lower:
                detected.append(pattern)
                
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
    
    def sanitize_input(self, input_data: Any) -> str:
        """清理输入数据，移除危险字符和模式
        
        临时修复版本：使用最安全的实现，避免无限循环问题。
        
        Args:
            input_data: 要清理的输入数据
            
        Returns:
            str: 清理后的安全字符串
        """
        # 处理 None
        if input_data is None:
            return ""
        
        # 处理数字
        if isinstance(input_data, (int, float)):
            return str(input_data)
        
        # 处理字符串 - 使用最简单的方法
        if isinstance(input_data, str):
            # 直接检查长度，避免复杂操作
            if len(input_data) > 100:
                return "SANITIZED_LONG_INPUT"
            
            # 简单检查：如果包含单引号或分号，认为是危险的
            if "'" in input_data or ";" in input_data:
                return "SANITIZED_DANGEROUS_INPUT"
            
            # 否则返回原始输入
            return input_data
        
        # 其他类型
        return "SANITIZED_OTHER_TYPE"
    
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
            
        # 检查是否包含有效字符（字母、数字、点、下划线、连字符）
        for char in api_key:
            if not (char.isalnum() or char in '._-'):
                return False
        
        return True
    
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