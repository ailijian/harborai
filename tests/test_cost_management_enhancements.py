#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成本管理增强功能测试

测试动态价格更新机制、价格配置热更新、审计日志和回滚功能
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timezone

from harborai.core.enhanced_pricing import EnhancedPricingCalculator, EnhancedModelPricing
from harborai.core.dynamic_pricing_manager import (
    DynamicPricingManager,
    PricingChangeType,
    PricingChangeStatus,
    PricingChangeRecord,
    PricingSnapshot
)


class TestCostManagementEnhancements:
    """测试成本管理增强功能"""
    
    @pytest.fixture
    def pricing_calculator(self):
        """创建增强价格计算器实例"""
        return EnhancedPricingCalculator()
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def pricing_manager(self, pricing_calculator, temp_dir):
        """创建动态价格管理器实例"""
        audit_log_path = Path(temp_dir) / "pricing_audit.jsonl"
        snapshot_path = Path(temp_dir) / "snapshots"
        return DynamicPricingManager(
            pricing_calculator=pricing_calculator,
            audit_log_path=str(audit_log_path),
            snapshot_path=str(snapshot_path)
        )
    
    @pytest.mark.asyncio
    async def test_dynamic_pricing_hot_update(self, pricing_manager):
        """测试动态价格热更新机制"""
        # 添加新的价格配置
        change_record = await pricing_manager.update_pricing(
            provider="openai",
            model="gpt-4",
            input_price_per_1k=0.03,
            output_price_per_1k=0.06,
            currency="USD",
            operator="admin",
            reason="价格调整"
        )
        
        # 验证变更记录
        assert change_record.change_type == PricingChangeType.CREATE
        assert change_record.status == PricingChangeStatus.APPLIED
        assert change_record.provider == "openai"
        assert change_record.model == "gpt-4"
        assert change_record.operator == "admin"
        assert change_record.reason == "价格调整"
        
        # 验证价格已应用到计算器
        dynamic_key = "openai:gpt-4"
        assert dynamic_key in pricing_manager.pricing_calculator.dynamic_pricing
        
        pricing = pricing_manager.pricing_calculator.dynamic_pricing[dynamic_key]
        assert pricing.input_price_per_1k == 0.03
        assert pricing.output_price_per_1k == 0.06
        assert pricing.currency == "USD"
        assert pricing.source == "dynamic"
    
    @pytest.mark.asyncio
    async def test_pricing_update_validation(self, pricing_manager):
        """测试价格配置验证"""
        # 测试价格超出范围
        with pytest.raises(ValueError, match="输入价格.*超出允许范围"):
            await pricing_manager.update_pricing(
                provider="test",
                model="test-model",
                input_price_per_1k=-1.0,  # 负价格
                output_price_per_1k=0.01,
                operator="admin"
            )
        
        # 测试不支持的货币
        with pytest.raises(ValueError, match="货币.*不在允许列表中"):
            await pricing_manager.update_pricing(
                provider="test",
                model="test-model",
                input_price_per_1k=0.01,
                output_price_per_1k=0.02,
                currency="JPY",  # 不支持的货币
                operator="admin"
            )
    
    @pytest.mark.asyncio
    async def test_pricing_audit_log(self, pricing_manager):
        """测试价格变更审计日志"""
        # 执行多个价格变更操作
        await pricing_manager.update_pricing(
            provider="openai", model="gpt-3.5-turbo",
            input_price_per_1k=0.001, output_price_per_1k=0.002,
            operator="admin", reason="初始配置"
        )
        
        await pricing_manager.update_pricing(
            provider="openai", model="gpt-3.5-turbo",
            input_price_per_1k=0.0015, output_price_per_1k=0.003,
            operator="admin", reason="价格调整"
        )
        
        await pricing_manager.delete_pricing(
            provider="openai", model="gpt-3.5-turbo",
            operator="admin", reason="停用模型"
        )
        
        # 获取变更历史
        history = await pricing_manager.get_change_history(
            provider="openai", model="gpt-3.5-turbo"
        )
        
        assert len(history) == 3
        assert history[0].change_type == PricingChangeType.DELETE  # 最新的在前
        assert history[1].change_type == PricingChangeType.UPDATE
        assert history[2].change_type == PricingChangeType.CREATE
        
        # 验证审计日志文件存在
        assert pricing_manager.audit_log_path.exists()
        
        # 读取审计日志文件内容
        with open(pricing_manager.audit_log_path, 'r', encoding='utf-8') as f:
            log_lines = f.readlines()
        
        assert len(log_lines) == 3
        
        # 验证日志格式
        for line in log_lines:
            log_data = json.loads(line.strip())
            assert "id" in log_data
            assert "change_type" in log_data
            assert "provider" in log_data
            assert "model" in log_data
            assert "timestamp" in log_data
            assert "operator" in log_data
    
    @pytest.mark.asyncio
    async def test_pricing_snapshot_and_rollback(self, pricing_manager):
        """测试价格配置快照和回滚功能"""
        # 设置初始价格配置
        await pricing_manager.update_pricing(
            provider="openai", model="gpt-4",
            input_price_per_1k=0.03, output_price_per_1k=0.06,
            operator="admin", reason="初始配置"
        )
        
        await pricing_manager.update_pricing(
            provider="deepseek", model="deepseek-chat",
            input_price_per_1k=0.001, output_price_per_1k=0.002,
            operator="admin", reason="初始配置"
        )
        
        # 创建快照
        snapshot = await pricing_manager.create_snapshot(
            description="版本1.0配置",
            operator="admin"
        )
        
        assert snapshot.id.startswith("snapshot_")
        assert snapshot.description == "版本1.0配置"
        assert len(snapshot.pricing_data) == 2
        assert "openai:gpt-4" in snapshot.pricing_data
        assert "deepseek:deepseek-chat" in snapshot.pricing_data
        
        # 修改价格配置
        await pricing_manager.update_pricing(
            provider="openai", model="gpt-4",
            input_price_per_1k=0.05, output_price_per_1k=0.10,
            operator="admin", reason="价格上调"
        )
        
        await pricing_manager.delete_pricing(
            provider="deepseek", model="deepseek-chat",
            operator="admin", reason="停用模型"
        )
        
        # 验证当前配置已改变
        assert len(pricing_manager.pricing_calculator.dynamic_pricing) == 1
        current_gpt4 = pricing_manager.pricing_calculator.dynamic_pricing["openai:gpt-4"]
        assert current_gpt4.input_price_per_1k == 0.05
        
        # 回滚到快照
        rollback_records = await pricing_manager.rollback_to_snapshot(
            snapshot_id=snapshot.id,
            operator="admin",
            reason="回滚到稳定版本"
        )
        
        assert len(rollback_records) == 2
        
        # 验证配置已回滚
        assert len(pricing_manager.pricing_calculator.dynamic_pricing) == 2
        restored_gpt4 = pricing_manager.pricing_calculator.dynamic_pricing["openai:gpt-4"]
        assert restored_gpt4.input_price_per_1k == 0.03
        assert restored_gpt4.output_price_per_1k == 0.06
        
        restored_deepseek = pricing_manager.pricing_calculator.dynamic_pricing["deepseek:deepseek-chat"]
        assert restored_deepseek.input_price_per_1k == 0.001
        assert restored_deepseek.output_price_per_1k == 0.002
    
    @pytest.mark.asyncio
    async def test_batch_pricing_update(self, pricing_manager):
        """测试批量价格更新"""
        pricing_updates = [
            {
                "provider": "openai",
                "model": "gpt-3.5-turbo",
                "input_price_per_1k": 0.001,
                "output_price_per_1k": 0.002,
                "currency": "USD"
            },
            {
                "provider": "openai",
                "model": "gpt-4",
                "input_price_per_1k": 0.03,
                "output_price_per_1k": 0.06,
                "currency": "USD"
            },
            {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "input_price_per_1k": 0.001,
                "output_price_per_1k": 0.002,
                "currency": "CNY"
            }
        ]
        
        change_records = await pricing_manager.batch_update_pricing(
            pricing_updates=pricing_updates,
            operator="admin",
            reason="批量初始化"
        )
        
        assert len(change_records) == 3
        assert all(record.status == PricingChangeStatus.APPLIED for record in change_records)
        assert len(pricing_manager.pricing_calculator.dynamic_pricing) == 3
        
        # 验证每个配置都正确应用
        for update in pricing_updates:
            key = f"{update['provider']}:{update['model']}"
            pricing = pricing_manager.pricing_calculator.dynamic_pricing[key]
            assert pricing.input_price_per_1k == update["input_price_per_1k"]
            assert pricing.output_price_per_1k == update["output_price_per_1k"]
            assert pricing.currency == update["currency"]
    
    @pytest.mark.asyncio
    async def test_pricing_statistics(self, pricing_manager):
        """测试价格配置统计信息"""
        # 执行一些操作
        await pricing_manager.update_pricing(
            provider="openai", model="gpt-4",
            input_price_per_1k=0.03, output_price_per_1k=0.06,
            operator="admin"
        )
        
        await pricing_manager.update_pricing(
            provider="openai", model="gpt-4",
            input_price_per_1k=0.035, output_price_per_1k=0.07,
            operator="admin"
        )
        
        await pricing_manager.create_snapshot(operator="admin")
        
        # 获取统计信息
        stats = await pricing_manager.get_pricing_statistics()
        
        assert stats["total_changes"] == 2
        assert stats["successful_changes"] == 2
        assert stats["failed_changes"] == 0
        assert stats["success_rate"] == 1.0
        assert stats["unique_providers"] == 1
        assert stats["unique_models"] == 1
        assert stats["total_snapshots"] == 1
        assert stats["current_dynamic_configs"] == 1
        assert "latest_change" in stats
    
    @pytest.mark.asyncio
    async def test_pricing_config_persistence(self, pricing_manager):
        """测试价格配置持久化"""
        # 添加价格配置
        await pricing_manager.update_pricing(
            provider="test", model="test-model",
            input_price_per_1k=0.01, output_price_per_1k=0.02,
            operator="admin"
        )
        
        # 创建快照
        snapshot = await pricing_manager.create_snapshot(operator="admin")
        
        # 验证快照文件存在
        snapshot_file = pricing_manager.snapshot_path / f"{snapshot.id}.json"
        assert snapshot_file.exists()
        
        # 读取快照文件内容
        with open(snapshot_file, 'r', encoding='utf-8') as f:
            snapshot_data = json.load(f)
        
        assert snapshot_data["id"] == snapshot.id
        assert "test:test-model" in snapshot_data["pricing_data"]
        assert snapshot_data["pricing_data"]["test:test-model"]["input_price_per_1k"] == 0.01
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, pricing_manager):
        """测试错误处理和恢复机制"""
        # 模拟价格计算器错误
        with patch.object(pricing_manager.pricing_calculator, 'add_dynamic_pricing', 
                         side_effect=Exception("模拟错误")):
            
            # 尝试更新价格（应该失败）
            with pytest.raises(Exception, match="模拟错误"):
                await pricing_manager.update_pricing(
                    provider="test", model="test-model",
                    input_price_per_1k=0.01, output_price_per_1k=0.02,
                    operator="admin"
                )
            
            # 验证失败记录被正确记录
            history = await pricing_manager.get_change_history(limit=1)
            assert len(history) == 1
            assert history[0].status == PricingChangeStatus.FAILED
            assert history[0].error_message == "模拟错误"
    
    def test_enhanced_pricing_calculator_integration(self, pricing_calculator):
        """测试增强价格计算器集成"""
        # 添加动态价格
        pricing_calculator.add_dynamic_pricing(
            provider="test",
            model="test-model",
            input_price_per_1k=0.01,
            output_price_per_1k=0.02,
            currency="CNY"
        )
        
        # 验证动态价格已添加
        assert len(pricing_calculator.dynamic_pricing) == 1
        
        # 获取价格摘要
        summary = pricing_calculator.get_pricing_summary()
        assert summary["dynamic_models_count"] == 1
        assert "test:test-model" in summary["dynamic_models"]
        
        # 移除动态价格
        pricing_calculator.remove_dynamic_pricing("test", "test-model")
        assert len(pricing_calculator.dynamic_pricing) == 0