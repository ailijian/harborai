"""数据脱敏测试模块

功能：测试HarborAI项目中的数据脱敏功能
参数：包含敏感信息检测、脱敏处理、合规性验证等测试
返回：测试结果和安全性评估
边界条件：处理各种敏感信息格式和边界情况
假设：敏感信息遵循常见格式模式
不确定点：不同厂商API对敏感信息的处理方式可能不同
验证方法：pytest tests/security/test_data_sanitization.py
"""

import pytest
import re
import json
import logging
import time
from typing import Dict, List, Any, Tuple
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

from tests.security import (
    SENSITIVE_PATTERNS, 
    SANITIZATION_CONFIG,
    SECURITY_TEST_TIMEOUT
)
from tests.utils.test_helpers import TestConfig, measure_performance
from tests.fixtures.data_fixtures import *


@pytest.fixture
def mock_harborai_client():
    """模拟HarborAI客户端"""
    mock_client = Mock()
    mock_client.chat = Mock()
    mock_client.chat.completions = Mock()
    mock_client.chat.completions.create = Mock()
    return mock_client


@dataclass
class SensitiveDataCase:
    """敏感数据测试用例"""
    name: str
    content: str
    expected_pattern: str
    should_be_detected: bool
    sanitized_format: str


class DataSanitizer:
    """数据脱敏工具类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化脱敏器"""
        self.config = config or SANITIZATION_CONFIG
        self.sensitive_patterns = SENSITIVE_PATTERNS
    
    def detect_sensitive_info(self, text: str) -> List[Dict[str, Any]]:
        """检测敏感信息"""
        detections = []
        
        for pattern_type, patterns in self.sensitive_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    detections.append({
                        'match': match.group(),
                        'pattern': pattern,
                        'start': match.start(),
                        'end': match.end(),
                        'type': pattern_type
                    })
        return detections
    
    def sanitize_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """脱敏文本内容"""
        detections = self.detect_sensitive_info(text)
        sanitized_text = text
        
        # 按位置倒序处理，避免位置偏移
        for detection in sorted(detections, key=lambda x: x['start'], reverse=True):
            start, end = detection['start'], detection['end']
            original = detection['match']
            masked = self._mask_sensitive_data(original, detection['type'])
            sanitized_text = sanitized_text[:start] + masked + sanitized_text[end:]
        
        return sanitized_text, detections
    
    def _get_pattern_type(self, pattern_id: int) -> str:
        """获取模式类型"""
        pattern_types = [
            'api_key',
            'generic_key', 
            'credit_card',
            'email',
            'ssn'
        ]
        return pattern_types[pattern_id] if pattern_id < len(pattern_types) else 'unknown'
    
    def _mask_sensitive_data(self, data: str, data_type: str) -> str:
        """脱敏敏感数据"""
        mask_char = self.config.get('mask_char', '*')
        preserve_length = self.config.get('preserve_length', True)
        preserve_format = self.config.get('preserve_format', True)
        min_mask_length = self.config.get('min_mask_length', 4)
        
        if data_type == 'email':
            # 邮箱脱敏：保留首尾字符和@域名
            parts = data.split('@')
            if len(parts) == 2:
                username, domain = parts
                if len(username) > 2:
                    masked_username = username[0] + mask_char * (len(username) - 2) + username[-1]
                else:
                    masked_username = mask_char * len(username)
                return f"{masked_username}@{domain}"
        
        elif data_type == 'credit_card':
            # 信用卡脱敏：只显示后4位
            digits_only = re.sub(r'\D', '', data)
            if len(digits_only) >= 4:
                masked_digits = mask_char * (len(digits_only) - 4) + digits_only[-4:]
                # 保持原格式
                result = data
                for i, char in enumerate(data):
                    if char.isdigit():
                        digit_index = len([c for c in data[:i] if c.isdigit()])
                        if digit_index < len(masked_digits):
                            result = result[:i] + masked_digits[digit_index] + result[i+1:]
                return result
        
        elif data_type in ['api_key', 'generic_key']:
            # API密钥脱敏：保留前缀和后4位
            if len(data) > 8:
                if data.startswith(('sk-', 'ak-')):
                    prefix = data[:3]
                    suffix = data[-4:]
                    middle_length = len(data) - 7
                    return prefix + mask_char * middle_length + suffix
                else:
                    return data[:2] + mask_char * (len(data) - 6) + data[-4:]
        
        # 默认脱敏：保留首尾，中间用*替换
        if preserve_length and len(data) > min_mask_length:
            if len(data) <= 4:
                return mask_char * len(data)
            else:
                return data[0] + mask_char * (len(data) - 2) + data[-1]
        else:
            return mask_char * max(min_mask_length, len(data) // 2)


class TestDataSanitization:
    """数据脱敏测试类"""
    
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """测试方法设置"""
        self.sanitizer = DataSanitizer()
        self.test_config = TestConfig()
    
    @pytest.fixture
    def sensitive_test_cases(self) -> List[SensitiveDataCase]:
        """敏感数据测试用例"""
        return [
            SensitiveDataCase(
                name="DeepSeek API Key",
                content="我的DeepSeek API密钥是 sk-1234567890abcdef1234567890abcdef1234567890abcdef 请保密",
                expected_pattern=r'sk-[a-zA-Z0-9]{48,}',
                should_be_detected=True,
                sanitized_format="sk-************************************cdef"
            ),
            SensitiveDataCase(
                name="Email Address",
                content="请联系 user@example.com 获取更多信息",
                expected_pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                should_be_detected=True,
                sanitized_format="u**r@example.com"
            ),
            SensitiveDataCase(
                name="Credit Card",
                content="信用卡号：4532-1234-5678-9012",
                expected_pattern=r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
                should_be_detected=True,
                sanitized_format="****-****-****-9012"
            ),
            SensitiveDataCase(
                name="SSN",
                content="社会安全号：123-45-6789",
                expected_pattern=r'\b\d{3}-\d{2}-\d{4}\b',
                should_be_detected=True,
                sanitized_format="***-**-****"
            ),
            SensitiveDataCase(
                name="Generic Key",
                content="密钥：abcd1234efgh5678ijkl9012mnop3456",
                expected_pattern=r'[a-zA-Z0-9]{32}',
                should_be_detected=True,
                sanitized_format="ab**************************3456"
            ),
            SensitiveDataCase(
                name="Normal Text",
                content="这是一段正常的文本，不包含敏感信息。",
                expected_pattern="",
                should_be_detected=False,
                sanitized_format="这是一段正常的文本，不包含敏感信息。"
            )
        ]
    
    @pytest.mark.security
    @pytest.mark.p0
    def test_sensitive_info_detection(self, sensitive_test_cases):
        """测试敏感信息检测功能"""
        for case in sensitive_test_cases:
            detections = self.sanitizer.detect_sensitive_info(case.content)
            
            if case.should_be_detected:
                assert len(detections) > 0, f"应该检测到敏感信息：{case.name}"
                
                # 验证检测结果
                found_match = False
                for detection in detections:
                    if re.search(case.expected_pattern, detection['match'], re.IGNORECASE):
                        found_match = True
                        break
                
                assert found_match, f"未找到预期的敏感信息模式：{case.expected_pattern}"
            else:
                assert len(detections) == 0, f"不应该检测到敏感信息：{case.name}"
    
    @pytest.mark.security
    @pytest.mark.p0
    def test_data_sanitization(self, sensitive_test_cases):
        """测试数据脱敏功能"""
        for case in sensitive_test_cases:
            sanitized_text, detections = self.sanitizer.sanitize_text(case.content)
            
            if case.should_be_detected:
                # 验证敏感信息已被脱敏
                assert sanitized_text != case.content, f"敏感信息未被脱敏：{case.name}"
                
                # 验证脱敏后不包含原始敏感信息
                for detection in detections:
                    assert detection['match'] not in sanitized_text, \
                        f"脱敏后仍包含原始敏感信息：{detection['match']}"
                
                # 验证脱敏格式合理性
                assert '*' in sanitized_text or len(sanitized_text) < len(case.content), \
                    f"脱敏格式不正确：{sanitized_text}"
            else:
                # 正常文本不应被修改
                assert sanitized_text == case.content, f"正常文本被错误修改：{case.name}"
    
    @pytest.mark.security
    @pytest.mark.p1
    def test_sanitization_preserves_structure(self):
        """测试脱敏保持文本结构"""
        test_text = """
        用户信息：
        - 邮箱：admin@company.com
        - DeepSeek API密钥：sk-abcdef1234567890abcdef1234567890abcdef1234567890
        - 备注：这是测试数据
        """
        
        sanitized_text, detections = self.sanitizer.sanitize_text(test_text)
        
        # 验证结构保持
        assert "用户信息：" in sanitized_text
        assert "- 邮箱：" in sanitized_text
        assert "- DeepSeek API密钥：" in sanitized_text
        assert "- 备注：这是测试数据" in sanitized_text
        
        # 验证敏感信息被脱敏
        assert "admin@company.com" not in sanitized_text
        assert "sk-abcdef1234567890abcdef1234567890abcdef1234567890" not in sanitized_text
        
        # 验证检测到敏感信息
        assert len(detections) >= 2
    
    @pytest.mark.security
    @pytest.mark.p1
    def test_multiple_sensitive_info_in_text(self):
        """测试文本中包含多个敏感信息的情况"""
        test_text = (
            "请使用DeepSeek API密钥 sk-test1234567890abcdef1234567890abcdef1234567890 "
            "联系邮箱 support@example.com 或拨打电话，"
            "信用卡号 4532-1234-5678-9012 用于付款。"
        )
        
        sanitized_text, detections = self.sanitizer.sanitize_text(test_text)
        
        # 验证检测到多个敏感信息
        assert len(detections) >= 3
        
        # 验证所有敏感信息都被脱敏
        sensitive_items = [
            "sk-test1234567890abcdef1234567890abcdef1234567890",
            "support@example.com",
            "4532-1234-5678-9012"
        ]
        
        for item in sensitive_items:
            assert item not in sanitized_text, f"敏感信息未被脱敏：{item}"
    
    @pytest.mark.security
    @pytest.mark.p2
    def test_sanitization_performance(self):
        """测试脱敏性能"""
        # 生成大量测试数据
        large_text = ""
        for i in range(100):
            large_text += f"用户{i}的邮箱是 user{i}@example.com，DeepSeek API密钥是 sk-{i:048d}。\n"
        
        # 性能测试
        start_time = time.time()
        sanitized_text, detections = self.sanitizer.sanitize_text(large_text)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # 验证性能要求（应在1秒内完成）
        assert processing_time < 1.0, f"脱敏处理时间过长：{processing_time:.2f}秒"
        
        # 验证处理结果
        assert len(detections) >= 200  # 至少检测到200个敏感信息（100个邮箱+100个DeepSeek API密钥）
        assert "user0@example.com" not in sanitized_text  # 完整邮箱被脱敏
        assert "sk-" in sanitized_text  # API密钥前缀保留
    
    @pytest.mark.security
    @pytest.mark.p2
    def test_edge_cases(self):
        """测试边界情况"""
        edge_cases = [
            "",  # 空字符串
            "   ",  # 只有空格
            "sk-",  # 不完整的API密钥
            "@example.com",  # 不完整的邮箱
            "1234-5678-9012",  # 不完整的信用卡号
            "sk-" + "a" * 100,  # 超长API密钥
            "user@" + "a" * 100 + ".com",  # 超长邮箱
        ]
        
        for case in edge_cases:
            try:
                sanitized_text, detections = self.sanitizer.sanitize_text(case)
                # 验证不会抛出异常
                assert isinstance(sanitized_text, str)
                assert isinstance(detections, list)
            except Exception as e:
                pytest.fail(f"边界情况处理失败：{case} - {str(e)}")
    
    @pytest.mark.security
    @pytest.mark.p1
    def test_sanitization_config_customization(self):
        """测试脱敏配置自定义"""
        custom_config = {
            'mask_char': '#',
            'preserve_length': False,
            'preserve_format': False,
            'min_mask_length': 8,
        }
        
        custom_sanitizer = DataSanitizer(custom_config)
        test_text = "DeepSeek API密钥：sk-1234567890abcdef1234567890abcdef1234567890abcdef"
        
        sanitized_text, detections = custom_sanitizer.sanitize_text(test_text)
        
        # 验证使用了自定义配置
        assert '#' in sanitized_text
        assert '*' not in sanitized_text
        assert len(detections) > 0
    
    @pytest.mark.security
    @pytest.mark.p1
    @pytest.mark.mock
    def test_harborai_response_sanitization(self, mock_harborai_client):
        """测试HarborAI响应的脱敏处理"""
        # 模拟包含敏感信息的响应
        sensitive_response = Mock()
        sensitive_response.choices = [Mock()]
        sensitive_response.choices[0].message = Mock()
        sensitive_response.choices[0].message.content = (
            "您的DeepSeek API密钥是 sk-1234567890abcdef1234567890abcdef1234567890abcdef，"
            "请联系 support@example.com 获取帮助。"
        )
        
        mock_harborai_client.chat.completions.create.return_value = sensitive_response
        
        # 获取响应并脱敏
        response = mock_harborai_client.chat.completions.create(
            model="test-model",
            messages=[{"role": "user", "content": "测试消息"}]
        )
        
        original_content = response.choices[0].message.content
        sanitized_content, detections = self.sanitizer.sanitize_text(original_content)
        
        # 验证脱敏效果
        assert len(detections) >= 2
        assert "sk-1234567890abcdef1234567890abcdef1234567890abcdef" not in sanitized_content
        assert "support@example.com" not in sanitized_content
        assert "sk-" in sanitized_content  # 前缀保留
        assert "@example.com" in sanitized_content  # 邮箱域名保留
    
    @pytest.mark.security
    @pytest.mark.p2
    def test_sanitization_logging_compliance(self, caplog):
        """测试脱敏日志合规性"""
        with caplog.at_level(logging.INFO):
            test_text = "用户DeepSeek API密钥：sk-1234567890abcdef1234567890abcdef1234567890abcdef"
            
            # 模拟日志记录
            logging.info(f"处理用户请求：{test_text}")
            
            # 检查日志中是否包含敏感信息
            for record in caplog.records:
                sanitized_message, detections = self.sanitizer.sanitize_text(record.message)
                
                if detections:
                    # 如果检测到敏感信息，验证应该被脱敏
                    for detection in detections:
                        assert detection['match'] not in record.message or \
                               "*" in sanitized_message, \
                               f"日志中包含未脱敏的敏感信息：{detection['match']}"
    
    @pytest.mark.security
    @pytest.mark.p1
    def test_sanitization_reversibility_prevention(self):
        """测试脱敏不可逆性"""
        original_text = "DeepSeek API密钥：sk-1234567890abcdef1234567890abcdef1234567890abcdef"
        sanitized_text, detections = self.sanitizer.sanitize_text(original_text)
        
        # 验证脱敏后无法恢复原始信息
        assert "1234567890abcdef1234567890abcdef1234567890abcdef" not in sanitized_text
        
        # 尝试多次脱敏，结果应该一致
        sanitized_again, _ = self.sanitizer.sanitize_text(sanitized_text)
        assert sanitized_again == sanitized_text
    
    @pytest.mark.security
    @pytest.mark.p3
    def test_international_sensitive_data(self):
        """测试国际化敏感数据处理"""
        international_cases = [
            "IBAN: GB82 WEST 1234 5698 7654 32",  # 国际银行账号
            "护照号：G12345678",  # 护照号
            "身份证：110101199001011234",  # 身份证号
        ]
        
        for case in international_cases:
            sanitized_text, detections = self.sanitizer.sanitize_text(case)
            
            # 虽然可能不在预定义模式中，但应该能安全处理
            assert isinstance(sanitized_text, str)
            assert len(sanitized_text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])