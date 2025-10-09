"""EncryptionManager 全面测试套件

测试加密管理器的所有功能，包括数据加密、解密、哈希等。
遵循 TDD 原则和 VIBE 编码规范。
"""

import pytest
import base64
import hashlib
from unittest.mock import patch

from harborai.security.encryption import EncryptionManager


class TestEncryptionManagerInitialization:
    """EncryptionManager 初始化测试类"""
    
    def test_initialization_default(self):
        """测试默认初始化"""
        manager = EncryptionManager()
        
        assert manager.key == "default_encryption_key"
        assert manager.algorithm == "AES-256"
    
    def test_initialization_with_custom_key(self):
        """测试使用自定义密钥初始化"""
        custom_key = "my_custom_encryption_key"
        manager = EncryptionManager(key=custom_key)
        
        assert manager.key == custom_key
        assert manager.algorithm == "AES-256"
    
    def test_initialization_with_none_key(self):
        """测试使用None密钥初始化"""
        manager = EncryptionManager(key=None)
        
        assert manager.key == "default_encryption_key"
        assert manager.algorithm == "AES-256"
    
    def test_initialization_with_empty_key(self):
        """测试使用空字符串密钥初始化"""
        manager = EncryptionManager(key="")
        
        assert manager.key == "default_encryption_key"
        assert manager.algorithm == "AES-256"


class TestDataEncryption:
    """数据加密测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.manager = EncryptionManager()
    
    def test_encrypt_basic_string(self):
        """测试基本字符串加密"""
        data = "Hello World"
        encrypted = self.manager.encrypt(data)
        
        assert encrypted != data
        assert encrypted.startswith("enc:")
        assert isinstance(encrypted, str)
    
    def test_encrypt_empty_string(self):
        """测试空字符串加密"""
        data = ""
        encrypted = self.manager.encrypt(data)
        
        assert encrypted == data  # 空字符串直接返回
    
    def test_encrypt_chinese_text(self):
        """测试中文文本加密"""
        data = "中文测试数据"
        encrypted = self.manager.encrypt(data)
        
        assert encrypted != data
        assert encrypted.startswith("enc:")
        
        # 验证可以正确解码
        encoded_part = encrypted[4:]  # 移除 'enc:' 前缀
        decoded = base64.b64decode(encoded_part.encode('utf-8')).decode('utf-8')
        assert decoded == data
    
    def test_encrypt_special_characters(self):
        """测试特殊字符加密"""
        data = "Special chars: !@#$%^&*()\n\t\r"
        encrypted = self.manager.encrypt(data)
        
        assert encrypted != data
        assert encrypted.startswith("enc:")
    
    def test_encrypt_long_text(self):
        """测试长文本加密"""
        data = "Very long text " * 1000
        encrypted = self.manager.encrypt(data)
        
        assert encrypted != data
        assert encrypted.startswith("enc:")
    
    def test_encrypt_json_data(self):
        """测试JSON数据加密"""
        data = '{"name": "test", "age": 25, "active": true}'
        encrypted = self.manager.encrypt(data)
        
        assert encrypted != data
        assert encrypted.startswith("enc:")
    
    def test_encrypt_consistency(self):
        """测试加密一致性"""
        data = "test data for consistency"
        
        encrypted1 = self.manager.encrypt(data)
        encrypted2 = self.manager.encrypt(data)
        encrypted3 = self.manager.encrypt(data)
        
        # 相同输入应产生相同输出
        assert encrypted1 == encrypted2 == encrypted3
    
    @pytest.mark.parametrize("data", [
        "Hello World",
        "中文测试",
        "Special: !@#$%^&*()",
        "Multi\nLine\nText",
        '{"json": "data"}',
        "123456789",
        "A" * 100,
    ])
    def test_encrypt_various_inputs(self, data):
        """测试各种输入的加密"""
        encrypted = self.manager.encrypt(data)
        
        assert isinstance(encrypted, str)
        if data:  # 非空字符串
            assert encrypted != data
            assert encrypted.startswith("enc:")


class TestDataDecryption:
    """数据解密测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.manager = EncryptionManager()
    
    def test_decrypt_basic_string(self):
        """测试基本字符串解密"""
        original = "Hello World"
        encrypted = self.manager.encrypt(original)
        decrypted = self.manager.decrypt(encrypted)
        
        assert decrypted == original
    
    def test_decrypt_empty_string(self):
        """测试空字符串解密"""
        original = ""
        encrypted = self.manager.encrypt(original)
        decrypted = self.manager.decrypt(encrypted)
        
        assert decrypted == original
    
    def test_decrypt_chinese_text(self):
        """测试中文文本解密"""
        original = "中文测试数据"
        encrypted = self.manager.encrypt(original)
        decrypted = self.manager.decrypt(encrypted)
        
        assert decrypted == original
    
    def test_decrypt_special_characters(self):
        """测试特殊字符解密"""
        original = "Special chars: !@#$%^&*()\n\t\r"
        encrypted = self.manager.encrypt(original)
        decrypted = self.manager.decrypt(encrypted)
        
        assert decrypted == original
    
    def test_decrypt_non_encrypted_data(self):
        """测试解密非加密数据"""
        data = "not encrypted data"
        decrypted = self.manager.decrypt(data)
        
        # 非加密数据应该原样返回
        assert decrypted == data
    
    def test_decrypt_invalid_format(self):
        """测试解密无效格式数据"""
        invalid_data = "enc:invalid_base64!"
        decrypted = self.manager.decrypt(invalid_data)
        
        # 无效数据应该原样返回
        assert decrypted == invalid_data
    
    def test_decrypt_corrupted_data(self):
        """测试解密损坏的数据"""
        original = "test data"
        encrypted = self.manager.encrypt(original)
        
        # 损坏加密数据
        corrupted = encrypted[:-5] + "XXXXX"
        decrypted = self.manager.decrypt(corrupted)
        
        # 损坏的数据应该原样返回
        assert decrypted == corrupted
    
    @pytest.mark.parametrize("original", [
        "Hello World",
        "中文测试",
        "Special: !@#$%^&*()",
        "Multi\nLine\nText",
        '{"json": "data"}',
        "123456789",
        "A" * 100,
        "",  # 空字符串
    ])
    def test_encrypt_decrypt_roundtrip(self, original):
        """测试加密解密往返一致性"""
        encrypted = self.manager.encrypt(original)
        decrypted = self.manager.decrypt(encrypted)
        
        assert decrypted == original


class TestDataHashing:
    """数据哈希测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.manager = EncryptionManager()
    
    def test_hash_basic_string(self):
        """测试基本字符串哈希"""
        data = "Hello World"
        hashed = self.manager.hash_data(data)
        
        assert isinstance(hashed, str)
        assert len(hashed) == 64  # SHA256哈希长度
        assert hashed != data
        
        # 验证是有效的十六进制字符串
        int(hashed, 16)
    
    def test_hash_empty_string(self):
        """测试空字符串哈希"""
        data = ""
        hashed = self.manager.hash_data(data)
        
        assert hashed == ""  # 空字符串返回空字符串
    
    def test_hash_chinese_text(self):
        """测试中文文本哈希"""
        data = "中文测试数据"
        hashed = self.manager.hash_data(data)
        
        assert isinstance(hashed, str)
        assert len(hashed) == 64
        assert hashed != data
    
    def test_hash_consistency(self):
        """测试哈希一致性"""
        data = "test data for hash consistency"
        
        hash1 = self.manager.hash_data(data)
        hash2 = self.manager.hash_data(data)
        hash3 = self.manager.hash_data(data)
        
        # 相同输入应产生相同哈希
        assert hash1 == hash2 == hash3
    
    def test_hash_uniqueness(self):
        """测试哈希唯一性"""
        test_cases = [
            "data1",
            "data2",
            "data1 ",  # 末尾空格
            "Data1",   # 大小写
            "data11",  # 长度差异
        ]
        
        hashes = []
        for data in test_cases:
            hash_result = self.manager.hash_data(data)
            hashes.append(hash_result)
        
        # 验证所有哈希值都不相同
        assert len(set(hashes)) == len(hashes)
    
    @pytest.mark.parametrize("data", [
        "Hello World",
        "中文测试",
        "Special: !@#$%^&*()",
        "Multi\nLine\nText",
        '{"json": "data"}',
        "123456789",
        "A" * 1000,
    ])
    def test_hash_various_inputs(self, data):
        """测试各种输入的哈希"""
        hashed = self.manager.hash_data(data)
        
        assert isinstance(hashed, str)
        assert len(hashed) == 64
        assert hashed != data
        
        # 验证是有效的十六进制字符串
        int(hashed, 16)


class TestHashVerification:
    """哈希验证测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.manager = EncryptionManager()
    
    def test_verify_hash_valid(self):
        """测试验证有效哈希"""
        data = "test data"
        hash_value = self.manager.hash_data(data)
        
        assert self.manager.verify_hash(data, hash_value) is True
    
    def test_verify_hash_invalid(self):
        """测试验证无效哈希"""
        data = "test data"
        wrong_hash = "invalid_hash_value"
        
        assert self.manager.verify_hash(data, wrong_hash) is False
    
    def test_verify_hash_different_data(self):
        """测试验证不同数据的哈希"""
        data1 = "test data 1"
        data2 = "test data 2"
        hash1 = self.manager.hash_data(data1)
        
        assert self.manager.verify_hash(data2, hash1) is False
    
    def test_verify_hash_empty_string(self):
        """测试验证空字符串哈希"""
        data = ""
        hash_value = self.manager.hash_data(data)
        
        assert self.manager.verify_hash(data, hash_value) is True
    
    @pytest.mark.parametrize("data", [
        "Hello World",
        "中文测试",
        "Special: !@#$%^&*()",
        "Multi\nLine\nText",
        '{"json": "data"}',
        "123456789",
    ])
    def test_verify_hash_various_inputs(self, data):
        """测试验证各种输入的哈希"""
        hash_value = self.manager.hash_data(data)
        
        assert self.manager.verify_hash(data, hash_value) is True


class TestKeyManagement:
    """密钥管理测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.manager = EncryptionManager()
    
    def test_generate_key(self):
        """测试生成密钥"""
        key = self.manager.generate_key()
        
        assert isinstance(key, str)
        assert len(key) == 64  # 32字节的十六进制表示
        
        # 验证是有效的十六进制字符串
        int(key, 16)
    
    def test_generate_key_uniqueness(self):
        """测试生成密钥的唯一性"""
        keys = []
        for _ in range(10):
            key = self.manager.generate_key()
            keys.append(key)
        
        # 验证所有密钥都不相同
        assert len(set(keys)) == len(keys)
    
    def test_rotate_key(self):
        """测试密钥轮换"""
        original_key = self.manager.key
        new_key = "new_encryption_key"
        
        old_key = self.manager.rotate_key(new_key)
        
        assert old_key == original_key
        assert self.manager.key == new_key
    
    def test_rotate_key_with_generated_key(self):
        """测试使用生成的密钥进行轮换"""
        original_key = self.manager.key
        new_key = self.manager.generate_key()
        
        old_key = self.manager.rotate_key(new_key)
        
        assert old_key == original_key
        assert self.manager.key == new_key
        assert len(self.manager.key) == 64


class TestEdgeCases:
    """边界情况测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.manager = EncryptionManager()
    
    def test_encrypt_very_long_text(self):
        """测试加密非常长的文本"""
        data = "A" * 100000  # 100KB文本
        encrypted = self.manager.encrypt(data)
        decrypted = self.manager.decrypt(encrypted)
        
        assert decrypted == data
    
    def test_encrypt_unicode_characters(self):
        """测试加密Unicode字符"""
        data = "🔒🛡️🚨 Unicode测试 àáâãäåæçèéêë"
        encrypted = self.manager.encrypt(data)
        decrypted = self.manager.decrypt(encrypted)
        
        assert decrypted == data
    
    def test_encrypt_binary_like_data(self):
        """测试加密类似二进制的数据"""
        data = "\x00\x01\x02\x03\x04\x05"
        encrypted = self.manager.encrypt(data)
        decrypted = self.manager.decrypt(encrypted)
        
        assert decrypted == data
    
    def test_hash_very_long_text(self):
        """测试哈希非常长的文本"""
        data = "B" * 100000  # 100KB文本
        hashed = self.manager.hash_data(data)
        
        assert isinstance(hashed, str)
        assert len(hashed) == 64
    
    def test_multiple_operations_consistency(self):
        """测试多次操作的一致性"""
        data = "consistency test data"
        
        # 多次加密解密
        for _ in range(100):
            encrypted = self.manager.encrypt(data)
            decrypted = self.manager.decrypt(encrypted)
            assert decrypted == data
        
        # 多次哈希
        hashes = []
        for _ in range(100):
            hashed = self.manager.hash_data(data)
            hashes.append(hashed)
        
        # 所有哈希应该相同
        assert len(set(hashes)) == 1


@pytest.mark.integration
class TestEncryptionManagerIntegration:
    """EncryptionManager 集成测试类"""
    
    def setup_method(self):
        """测试前置设置"""
        self.manager = EncryptionManager()
    
    def test_complete_encryption_workflow(self):
        """测试完整的加密工作流程"""
        # 1. 生成新密钥
        new_key = self.manager.generate_key()
        
        # 2. 轮换密钥
        old_key = self.manager.rotate_key(new_key)
        
        # 3. 加密数据
        sensitive_data = "敏感数据：用户密码123456"
        encrypted = self.manager.encrypt(sensitive_data)
        
        # 4. 验证加密结果
        assert encrypted != sensitive_data
        assert encrypted.startswith("enc:")
        
        # 5. 解密数据
        decrypted = self.manager.decrypt(encrypted)
        assert decrypted == sensitive_data
        
        # 6. 哈希数据
        hashed = self.manager.hash_data(sensitive_data)
        
        # 7. 验证哈希
        assert self.manager.verify_hash(sensitive_data, hashed) is True
        assert self.manager.verify_hash("错误数据", hashed) is False
    
    def test_cross_instance_compatibility(self):
        """测试跨实例兼容性"""
        # 使用第一个实例加密
        manager1 = EncryptionManager(key="shared_key")
        data = "cross instance test"
        encrypted = manager1.encrypt(data)
        hashed = manager1.hash_data(data)
        
        # 使用第二个实例解密
        manager2 = EncryptionManager(key="shared_key")
        decrypted = manager2.decrypt(encrypted)
        hash_verified = manager2.verify_hash(data, hashed)
        
        assert decrypted == data
        assert hash_verified is True
    
    def test_performance_benchmark(self):
        """测试性能基准"""
        import time
        
        data = "performance test data " * 100  # 约2KB数据
        
        # 加密性能测试
        start_time = time.time()
        for _ in range(1000):
            self.manager.encrypt(data)
        encrypt_time = time.time() - start_time
        
        # 解密性能测试
        encrypted = self.manager.encrypt(data)
        start_time = time.time()
        for _ in range(1000):
            self.manager.decrypt(encrypted)
        decrypt_time = time.time() - start_time
        
        # 哈希性能测试
        start_time = time.time()
        for _ in range(1000):
            self.manager.hash_data(data)
        hash_time = time.time() - start_time
        
        # 性能要求：1000次操作应在1秒内完成
        assert encrypt_time < 1.0
        assert decrypt_time < 1.0
        assert hash_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])