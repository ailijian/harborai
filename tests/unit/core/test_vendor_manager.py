#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HarborAI 厂商管理器测试

测试厂商管理器的配置、切换和故障转移功能。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from harborai.core.vendor_manager import (
    VendorManager, VendorType, VendorConfig
)
from harborai.core.exceptions import HarborAIError, APIError


class TestVendorType:
    """测试厂商类型枚举"""
    
    def test_vendor_type_values(self):
        """测试厂商类型值"""
        assert VendorType.DEEPSEEK.value == "deepseek"
        assert VendorType.ERNIE.value == "ernie"
        assert VendorType.DOUBAO.value == "doubao"
    
    def test_vendor_type_from_string(self):
        """测试从字符串创建厂商类型"""
        assert VendorType("deepseek") == VendorType.DEEPSEEK
        assert VendorType("ernie") == VendorType.ERNIE
        assert VendorType("doubao") == VendorType.DOUBAO
    
    def test_vendor_type_invalid_string(self):
        """测试无效字符串创建厂商类型"""
        with pytest.raises(ValueError):
            VendorType("invalid_vendor")


class TestVendorConfig:
    """测试厂商配置数据类"""
    
    def test_vendor_config_creation(self):
        """测试厂商配置创建"""
        config = VendorConfig(
            name="Test Vendor",
            api_key="test_key",
            base_url="https://api.test.com",
            models=["model1", "model2"],
            max_tokens=4096,
            supports_streaming=True,
            supports_function_calling=False,
            rate_limit_rpm=60
        )
        
        assert config.name == "Test Vendor"
        assert config.api_key == "test_key"
        assert config.base_url == "https://api.test.com"
        assert config.models == ["model1", "model2"]
        assert config.max_tokens == 4096
        assert config.supports_streaming is True
        assert config.supports_function_calling is False
        assert config.rate_limit_rpm == 60
    
    def test_vendor_config_equality(self):
        """测试厂商配置相等性"""
        config1 = VendorConfig(
            name="Test",
            api_key="key",
            base_url="url",
            models=["model"],
            max_tokens=1000,
            supports_streaming=True,
            supports_function_calling=True,
            rate_limit_rpm=60
        )
        
        config2 = VendorConfig(
            name="Test",
            api_key="key",
            base_url="url",
            models=["model"],
            max_tokens=1000,
            supports_streaming=True,
            supports_function_calling=True,
            rate_limit_rpm=60
        )
        
        assert config1 == config2


class TestVendorManager:
    """测试厂商管理器"""
    
    def setup_method(self):
        """每个测试前的设置"""
        self.test_config = {
            "deepseek_api_key": "test_deepseek_key",
            "ernie_api_key": "test_ernie_key",
            "doubao_api_key": "test_doubao_key"
        }
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_vendor_manager_creation(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试厂商管理器创建"""
        manager = VendorManager(self.test_config)
        
        assert manager.config == self.test_config
        assert len(manager._vendors) == 3
        assert VendorType.DEEPSEEK in manager._vendors
        assert VendorType.ERNIE in manager._vendors
        assert VendorType.DOUBAO in manager._vendors
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_vendor_manager_creation_without_config(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试无配置创建厂商管理器"""
        manager = VendorManager()
        
        assert manager.config == {}
        assert len(manager._vendors) == 3
        
        # 检查默认配置
        deepseek_config = manager._vendors[VendorType.DEEPSEEK]
        assert deepseek_config.name == "DeepSeek"
        assert deepseek_config.api_key == ""
        assert deepseek_config.base_url == "https://api.deepseek.com"
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_initialize_vendors(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试初始化厂商配置"""
        manager = VendorManager(self.test_config)
        
        # 检查DeepSeek配置
        deepseek_config = manager._vendors[VendorType.DEEPSEEK]
        assert deepseek_config.name == "DeepSeek"
        assert deepseek_config.api_key == "test_deepseek_key"
        assert deepseek_config.models == ["deepseek-chat", "deepseek-coder"]
        assert deepseek_config.supports_streaming is True
        assert deepseek_config.supports_function_calling is True
        
        # 检查ERNIE配置
        ernie_config = manager._vendors[VendorType.ERNIE]
        assert ernie_config.name == "百度ERNIE"
        assert ernie_config.api_key == "test_ernie_key"
        assert ernie_config.models == ["ernie-3.5-8k", "ernie-4.0-8k"]
        assert ernie_config.supports_function_calling is False
        
        # 检查Doubao配置
        doubao_config = manager._vendors[VendorType.DOUBAO]
        assert doubao_config.name == "字节跳动Doubao"
        assert doubao_config.api_key == "test_doubao_key"
        assert doubao_config.models == ["doubao-lite-4k", "doubao-pro-4k"]
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_initialize_plugins_success(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试成功初始化插件"""
        mock_deepseek_instance = Mock()
        mock_wenxin_instance = Mock()
        mock_doubao_instance = Mock()
        
        mock_deepseek.return_value = mock_deepseek_instance
        mock_wenxin.return_value = mock_wenxin_instance
        mock_doubao.return_value = mock_doubao_instance
        
        manager = VendorManager()
        
        assert manager._plugins[VendorType.DEEPSEEK] == mock_deepseek_instance
        assert manager._plugins[VendorType.ERNIE] == mock_wenxin_instance
        assert manager._plugins[VendorType.DOUBAO] == mock_doubao_instance
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_initialize_plugins_failure(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试插件初始化失败"""
        mock_deepseek.side_effect = Exception("DeepSeek plugin error")
        mock_wenxin.side_effect = Exception("Wenxin plugin error")
        mock_doubao.side_effect = Exception("Doubao plugin error")
        
        with patch('harborai.core.vendor_manager.logger') as mock_logger:
            manager = VendorManager()
            
            # 验证警告日志被记录
            assert mock_logger.warning.call_count == 3
            
            # 验证插件字典为空
            assert len(manager._plugins) == 0
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_available_vendors(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试获取可用厂商列表"""
        manager = VendorManager()
        vendors = manager.get_available_vendors()
        
        assert len(vendors) == 3
        assert "deepseek" in vendors
        assert "ernie" in vendors
        assert "doubao" in vendors
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_vendor_config_by_enum(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试通过枚举获取厂商配置"""
        manager = VendorManager(self.test_config)
        
        config = manager.get_vendor_config(VendorType.DEEPSEEK)
        assert config is not None
        assert config.name == "DeepSeek"
        assert config.api_key == "test_deepseek_key"
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_vendor_config_by_string(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试通过字符串获取厂商配置"""
        manager = VendorManager(self.test_config)
        
        config = manager.get_vendor_config("deepseek")
        assert config is not None
        assert config.name == "DeepSeek"
        
        config = manager.get_vendor_config("ernie")
        assert config is not None
        assert config.name == "百度ERNIE"
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_vendor_config_invalid(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试获取无效厂商配置"""
        manager = VendorManager()
        
        config = manager.get_vendor_config("invalid_vendor")
        assert config is None
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_vendor_plugin(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试获取厂商插件"""
        mock_deepseek_instance = Mock()
        mock_deepseek.return_value = mock_deepseek_instance
        
        manager = VendorManager()
        
        plugin = manager.get_vendor_plugin(VendorType.DEEPSEEK)
        assert plugin == mock_deepseek_instance
        
        plugin = manager.get_vendor_plugin("deepseek")
        assert plugin == mock_deepseek_instance
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_vendor_plugin_invalid(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试获取无效厂商插件"""
        manager = VendorManager()
        
        plugin = manager.get_vendor_plugin("invalid_vendor")
        assert plugin is None
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_is_vendor_available_with_api_key(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试有API密钥的厂商可用性"""
        manager = VendorManager(self.test_config)
        
        assert manager.is_vendor_available(VendorType.DEEPSEEK) is True
        assert manager.is_vendor_available("ernie") is True
        assert manager.is_vendor_available("doubao") is True
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_is_vendor_available_without_api_key(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试无API密钥的厂商可用性"""
        manager = VendorManager()  # 无配置
        
        assert manager.is_vendor_available(VendorType.DEEPSEEK) is False
        assert manager.is_vendor_available("ernie") is False
        assert manager.is_vendor_available("doubao") is False
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_is_vendor_available_invalid_vendor(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试无效厂商的可用性"""
        manager = VendorManager()
        
        assert manager.is_vendor_available("invalid_vendor") is False
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_models_for_vendor(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试获取厂商支持的模型"""
        manager = VendorManager()
        
        models = manager.get_models_for_vendor(VendorType.DEEPSEEK)
        assert models == ["deepseek-chat", "deepseek-coder"]
        
        models = manager.get_models_for_vendor("ernie")
        assert models == ["ernie-3.5-8k", "ernie-4.0-8k"]
        
        models = manager.get_models_for_vendor("doubao")
        assert models == ["doubao-lite-4k", "doubao-pro-4k"]
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_models_for_invalid_vendor(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试获取无效厂商的模型"""
        manager = VendorManager()
        
        models = manager.get_models_for_vendor("invalid_vendor")
        assert models == []
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_vendor_for_model(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试根据模型获取厂商"""
        manager = VendorManager()
        
        vendor = manager.get_vendor_for_model("deepseek-chat")
        assert vendor == VendorType.DEEPSEEK
        
        vendor = manager.get_vendor_for_model("ernie-3.5-8k")
        assert vendor == VendorType.ERNIE
        
        vendor = manager.get_vendor_for_model("doubao-lite-4k")
        assert vendor == VendorType.DOUBAO
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_vendor_for_unknown_model(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试获取未知模型的厂商"""
        manager = VendorManager()
        
        vendor = manager.get_vendor_for_model("unknown-model")
        assert vendor is None
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_switch_vendor_success(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试成功切换厂商"""
        manager = VendorManager(self.test_config)
        
        with patch('harborai.core.vendor_manager.logger') as mock_logger:
            result = manager.switch_vendor(VendorType.DEEPSEEK, VendorType.ERNIE)
            assert result is True
            mock_logger.info.assert_called_once()
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_switch_vendor_unavailable_target(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试切换到不可用的厂商"""
        manager = VendorManager()  # 无API密钥
        
        with patch('harborai.core.vendor_manager.logger') as mock_logger:
            result = manager.switch_vendor(VendorType.DEEPSEEK, VendorType.ERNIE)
            assert result is False
            mock_logger.error.assert_called_once()
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_failover_sequence(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试获取故障转移序列"""
        manager = VendorManager(self.test_config)
        
        sequence = manager.get_failover_sequence(VendorType.ERNIE)
        
        # 应该包含主要厂商和其他可用厂商
        assert VendorType.ERNIE in sequence
        assert VendorType.DEEPSEEK in sequence
        assert VendorType.DOUBAO in sequence
        
        # 主要厂商应该在第一位
        assert sequence[0] == VendorType.ERNIE
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_failover_sequence_partial_availability(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试部分厂商可用时的故障转移序列"""
        config = {"deepseek_api_key": "test_key"}  # 只有DeepSeek可用
        manager = VendorManager(config)
        
        sequence = manager.get_failover_sequence(VendorType.ERNIE)
        
        # 应该只包含主要厂商和可用的厂商
        assert VendorType.ERNIE in sequence
        assert VendorType.DEEPSEEK in sequence
        assert VendorType.DOUBAO not in sequence
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_get_failover_sequence_invalid_vendor(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试无效厂商的故障转移序列"""
        manager = VendorManager()
        
        sequence = manager.get_failover_sequence("invalid_vendor")
        assert sequence == []
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_update_vendor_config_success(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试成功更新厂商配置"""
        manager = VendorManager()
        
        updates = {
            "api_key": "new_api_key",
            "max_tokens": 8192,
            "rate_limit_rpm": 120
        }
        
        with patch('harborai.core.vendor_manager.logger') as mock_logger:
            result = manager.update_vendor_config(VendorType.DEEPSEEK, updates)
            assert result is True
            mock_logger.info.assert_called_once()
        
        # 验证配置已更新
        config = manager.get_vendor_config(VendorType.DEEPSEEK)
        assert config.api_key == "new_api_key"
        assert config.max_tokens == 8192
        assert config.rate_limit_rpm == 120
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_update_vendor_config_invalid_vendor(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试更新无效厂商配置"""
        manager = VendorManager()
        
        result = manager.update_vendor_config("invalid_vendor", {"api_key": "test"})
        assert result is False
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_update_vendor_config_invalid_field(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试更新无效配置字段"""
        manager = VendorManager()
        
        updates = {
            "api_key": "new_key",
            "invalid_field": "value"  # 无效字段
        }
        
        result = manager.update_vendor_config(VendorType.DEEPSEEK, updates)
        assert result is True
        
        # 验证有效字段已更新，无效字段被忽略
        config = manager.get_vendor_config(VendorType.DEEPSEEK)
        assert config.api_key == "new_key"
        assert not hasattr(config, "invalid_field")


class TestVendorManagerIntegration:
    """测试厂商管理器集成场景"""
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_full_vendor_lifecycle(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试完整厂商生命周期"""
        config = {
            "deepseek_api_key": "deepseek_key",
            "ernie_api_key": "ernie_key"
        }
        
        manager = VendorManager(config)
        
        # 1. 检查初始状态
        assert len(manager.get_available_vendors()) == 3
        assert manager.is_vendor_available("deepseek") is True
        assert manager.is_vendor_available("ernie") is True
        assert manager.is_vendor_available("doubao") is False
        
        # 2. 获取模型和厂商映射
        models = manager.get_models_for_vendor("deepseek")
        assert "deepseek-chat" in models
        
        vendor = manager.get_vendor_for_model("deepseek-chat")
        assert vendor == VendorType.DEEPSEEK
        
        # 3. 测试故障转移
        sequence = manager.get_failover_sequence("deepseek")
        assert len(sequence) >= 2  # 至少包含主要厂商和一个备用厂商
        
        # 4. 更新配置
        result = manager.update_vendor_config("doubao", {"api_key": "doubao_key"})
        assert result is True
        assert manager.is_vendor_available("doubao") is True
        
        # 5. 测试厂商切换
        result = manager.switch_vendor("deepseek", "doubao")
        assert result is True
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_vendor_failover_scenario(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试厂商故障转移场景"""
        config = {
            "deepseek_api_key": "deepseek_key",
            "ernie_api_key": "ernie_key",
            "doubao_api_key": "doubao_key"
        }
        
        manager = VendorManager(config)
        
        # 模拟主要厂商故障
        primary_vendor = "deepseek"
        
        # 获取故障转移序列
        failover_sequence = manager.get_failover_sequence(primary_vendor)
        
        # 验证故障转移序列
        assert len(failover_sequence) == 3
        assert VendorType.DEEPSEEK in failover_sequence
        
        # 模拟逐个尝试厂商
        for vendor in failover_sequence[1:]:  # 跳过主要厂商
            if manager.is_vendor_available(vendor):
                success = manager.switch_vendor(primary_vendor, vendor)
                assert success is True
                break
    
    @patch('harborai.core.vendor_manager.DeepSeekPlugin')
    @patch('harborai.core.vendor_manager.WenxinPlugin')
    @patch('harborai.core.vendor_manager.DoubaoPlugin')
    def test_vendor_configuration_management(self, mock_doubao, mock_wenxin, mock_deepseek):
        """测试厂商配置管理"""
        manager = VendorManager()
        
        # 初始状态：所有厂商都不可用
        for vendor_type in VendorType:
            assert manager.is_vendor_available(vendor_type) is False
        
        # 逐个配置厂商
        vendors_to_configure = [
            ("deepseek", "deepseek_test_key"),
            ("ernie", "ernie_test_key"),
            ("doubao", "doubao_test_key")
        ]
        
        for vendor_name, api_key in vendors_to_configure:
            # 更新API密钥
            result = manager.update_vendor_config(vendor_name, {"api_key": api_key})
            assert result is True
            
            # 验证厂商现在可用
            assert manager.is_vendor_available(vendor_name) is True
            
            # 验证配置正确
            config = manager.get_vendor_config(vendor_name)
            assert config.api_key == api_key
        
        # 验证所有厂商都可用
        available_vendors = manager.get_available_vendors()
        assert len(available_vendors) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])