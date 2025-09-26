"""安全测试模块

本模块包含HarborAI项目的安全相关测试，包括：
- 数据脱敏测试
- API密钥安全测试
- 输入验证测试

遵循VIBE Coding规则，确保安全性和合规性。
"""

__version__ = "1.0.0"
__author__ = "HarborAI Team"

# 安全测试相关常量
SECURITY_TEST_TIMEOUT = 30  # 安全测试超时时间（秒）
MAX_INPUT_LENGTH = 10000    # 最大输入长度
SENSITIVE_PATTERNS = {      # 敏感信息模式
    'api_key': [
        r'sk-[a-zA-Z0-9]{32,64}',  # 通用API密钥格式
    ],
    'generic_key': [
        r'[a-zA-Z0-9]{32}',     # 通用32位密钥
    ],
    'credit_card': [
        r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # 信用卡号
    ],
    'email': [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 邮箱
    ],
    'ssn': [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN格式
    ],
}

# 测试数据脱敏配置
SANITIZATION_CONFIG = {
    'mask_char': '*',
    'preserve_length': True,
    'preserve_format': True,
    'min_mask_length': 4,
}

# API密钥安全配置
API_KEY_SECURITY_CONFIG = {
    'min_key_length': 16,
    'max_key_length': 256,
    'allowed_key_prefixes': ['sk-', 'ak-', 'Bearer '],
    'forbidden_patterns': ['test', 'demo', 'example', '123456'],
    'secure_storage_path': '/tmp/secure_keys',
    'max_access_per_hour': 100,
    'encryption_algorithm': 'AES-256',
    'key_rotation_days': 90,
    'audit_log_level': 'high',
    'transmission_encryption': True,
    'log_sensitive_operations': True,
}

# 输入验证配置
INPUT_VALIDATION_CONFIG = {
    'max_message_length': 32768,
    'max_messages_count': 100,
    'forbidden_content_types': ['image/svg+xml'],
    'sql_injection_patterns': [
        r"('|(\-\-)|(;)|(\|)|(\*)|(%))",
        r"(union|select|insert|delete|update|drop|create|alter)",
    ],
    'xss_patterns': [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
    ],
}